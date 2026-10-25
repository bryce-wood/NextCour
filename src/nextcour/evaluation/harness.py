"""The evaluation harness (roadmap Phase 2): every method, scored the same way on the split's users.

`uv run nextcour eval <config>` runs it on dev users; `--users test` is for reported results only.

Two jobs are measured (design note, Metrics):
- **A. Planning list:** ordering accuracy among the shows a user started after the cutoff, which
  stand in for their Planning list, since MAL data doesn't record what was on it.
- **D. Continue a series:** ordering accuracy among the targets that continue a franchise the user
  had started before the cutoff, a stand-in for "next entries the user started".
- **B. Discovery:** nDCG@10 and Recall@100 over the whole catalog. Candidates are franchise entry
  points that had aired by the end of the data, minus anything on the user's list before the
  cutoff and any franchise they'd started. Relevant shows are the targets among those candidates
  that the user enjoyed (0.6 or more), with their enjoyment as the gain.

Settled 2026-10-24, before the first result:
- Labels: enjoyment from the user's whole list (the truth), as in the "works for me" test.
  Histories and their labels use only entries from before the cutoff.
- Crowd numbers (members, average score) come from train users' entries before the cutoff, never
  from MAL's own 2018 numbers, which include later ratings. An average needs at least 50 scores;
  shows with fewer get the average of all averages, so they sit in the middle, not at the bottom.
- Ties in a method's ranking are broken by one fixed random order of the shows.
- Every metric is per user, then averaged over users. Intervals are a paired bootstrap over users.

The methods themselves are in methods.py. Each says which jobs it's measured on.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import duckdb
import numpy as np

from nextcour.core import guardrail
from nextcour.core.entries import WATCHED, ListEntry
from nextcour.evaluation import methods as method_library
from nextcour.evaluation import metrics, split
from nextcour.profile import builder, enjoyment

MIN_SCORES = 50  # scores an average needs before it counts
ENJOYED = 0.6  # design note, Enjoyment
DISCOVERY_K = 10
RECALL_K = 100
TIEBREAK_SEED = 7


# --- The catalog, with crowd numbers from train users ----------------------------------------


@dataclass(frozen=True)
class Catalog:
    ids: np.ndarray  # AniList IDs
    index: Mapping[int, int]  # AniList ID -> position
    franchise: np.ndarray  # franchise ID; a show outside any franchise gets its own (minus its ID)
    entry_point: np.ndarray
    year: np.ndarray  # year it started airing; NaN if unknown
    aired_day: np.ndarray  # the day it started airing, as days since 1970; -1 if unknown
    aired_after_cutoff: np.ndarray
    available: np.ndarray  # aired by the end of the data, and not a music video
    members: np.ndarray  # train users with it on their list before the cutoff
    scores: np.ndarray  # how many of them scored it
    average: np.ndarray  # their average score, 0-100; NaN with fewer than MIN_SCORES
    band: np.ndarray  # "popular" (top 10% by members), "middle" or "niche" (bottom 50%)
    top_percent: np.ndarray  # in the top 1% by members
    watch_hours: np.ndarray
    tiebreak: np.ndarray

    def __len__(self) -> int:
        return len(self.ids)


def fit_crowd_stats(con: duckdb.DuckDBPyConnection, folder: Path) -> dict[int, tuple[int, int, float]]:
    """AniList ID -> (members, scores, average score) from train users before the cutoff."""
    con.execute(f"CREATE OR REPLACE VIEW train AS SELECT * FROM '{(folder / 'train.parquet').as_posix()}'")
    guardrail.check_training_table(con, "train")
    rows = con.execute(
        """
        SELECT anilist_id, count(*),
               count(*) FILTER (score > 0 AND status <> 'PLANNING'),
               avg(score) FILTER (score > 0 AND status <> 'PLANNING')
        FROM train WHERE before GROUP BY 1
        """
    ).fetchall()
    return {a: (members, scored, average) for a, members, scored, average in rows}


def load_catalog(folder: Path, cutoff: dt.date, con: duckdb.DuckDBPyConnection | None = None) -> Catalog:
    con = con or duckdb.connect()
    crowd = fit_crowd_stats(con, folder)
    rows = con.execute(
        f"SELECT anilist_id, franchise_id, entry_point, aired, format, episodes, duration"
        f" FROM '{(folder / 'shows.parquet').as_posix()}' ORDER BY anilist_id"
    ).fetchall()
    n = len(rows)
    ids = np.array([r[0] for r in rows], dtype=np.int64)
    franchise = np.array([r[1] if r[1] is not None else -r[0] for r in rows], dtype=np.int64)
    aired = [r[3] for r in rows]
    year = np.array([a.year if a else np.nan for a in aired], dtype=float)
    available = np.array(
        [a is not None and a <= split.DATA_END and r[4] != "MUSIC" for a, r in zip(aired, rows, strict=True)]
    )
    members = np.array([crowd.get(i, (0, 0, None))[0] for i in ids], dtype=np.int64)
    scores = np.array([crowd.get(i, (0, 0, None))[1] for i in ids], dtype=np.int64)
    average = np.array(
        [crowd[i][2] if i in crowd and crowd[i][1] >= MIN_SCORES else np.nan for i in ids], dtype=float
    )
    order = np.lexsort((ids, -members))  # most members first
    order = order[available[order]]
    band = np.full(n, "middle", dtype=object)
    band[order[: math.ceil(0.10 * len(order))]] = "popular"
    band[order[len(order) - len(order) // 2 :]] = "niche"
    top_percent = np.zeros(n, dtype=bool)
    top_percent[order[: math.ceil(0.01 * len(order))]] = True
    minutes = [
        (r[5] or 1) * (r[6] or builder.DEFAULT_EPISODE_MINUTES) for r in rows
    ]  # unknown counts as one episode
    return Catalog(
        ids=ids,
        index={int(a): i for i, a in enumerate(ids)},
        franchise=franchise,
        entry_point=np.array([bool(r[2]) for r in rows]),
        year=year,
        aired_day=np.array([(a - dt.date(1970, 1, 1)).days if a else -1 for a in aired], dtype=np.int64),
        aired_after_cutoff=np.array([a is not None and a >= cutoff for a in aired]),
        available=available,
        members=members,
        scores=scores,
        average=average,
        band=band,
        top_percent=top_percent,
        watch_hours=np.array(minutes, dtype=float) / 60,
        tiebreak=np.random.default_rng(TIEBREAK_SEED).permutation(n),
    )


# --- Users -----------------------------------------------------------------------------------


@dataclass
class EvalUser:
    name: str
    history: tuple[ListEntry, ...]
    targets: np.ndarray  # catalog positions of the shows started after the cutoff
    labels: np.ndarray  # their enjoyment, from the whole list
    sequel: np.ndarray  # whether each target continues a franchise started before the cutoff
    seen: np.ndarray  # catalog positions of everything on the list before the cutoff
    started: frozenset[int]  # franchises with a watched entry before the cutoff
    has_dates: bool  # at least half the completed history has start and finish dates
    watched: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))  # watched before
    franchise_of: Mapping[int, int] = field(repr=False, default_factory=dict)
    _profile: builder.Profile | None = field(default=None, repr=False)

    @property
    def profile(self) -> builder.Profile:
        """The profile built from the history, as the app builds mine."""
        if self._profile is None:
            self._profile = builder.build(self.history, self.franchise_of)
        return self._profile


def prepare(user: split.SplitUser, catalog: Catalog, franchise_of: Mapping[int, int]) -> EvalUser:
    kept = [e for e in user.entries if e.format not in builder.EXCLUDED_FORMATS]
    truth = enjoyment.labels(kept, (), franchise_of)
    targets = [
        e
        for e in user.after
        if e.format not in builder.EXCLUDED_FORMATS
        and e.anilist_id in truth
        and e.anilist_id in catalog.index
    ]
    positions = np.array([catalog.index[e.anilist_id] for e in targets], dtype=np.int64)
    started = frozenset(
        int(catalog.franchise[catalog.index[e.anilist_id]])
        for e in user.history
        if e.status in WATCHED and e.anilist_id in catalog.index
    )
    completed = [e for e in user.history if e.status in enjoyment.DONE]
    dated = sum(1 for e in completed if e.started and e.completed)
    return EvalUser(
        name=user.name,
        history=user.history,
        targets=positions,
        labels=np.array([truth[e.anilist_id] for e in targets], dtype=float),
        sequel=np.array([int(catalog.franchise[p]) in started for p in positions], dtype=bool),
        seen=np.array([catalog.index[e.anilist_id] for e in user.history if e.anilist_id in catalog.index]),
        started=started,
        has_dates=bool(completed) and dated >= len(completed) / 2,
        watched=np.array(
            [
                catalog.index[e.anilist_id]
                for e in user.history
                if e.status in WATCHED and e.anilist_id in catalog.index
            ],
            dtype=np.int64,
        ),
        franchise_of=franchise_of,
    )


@dataclass(frozen=True)
class Config:
    name: str
    methods: tuple[str, ...]
    reference: str  # every method is compared with this one
    description: str


CONFIGS = {
    "baselines": Config(
        "baselines",
        (
            "random",
            "popularity",
            "average_score",
            "item_item",
            "content_similarity",
            "sequel_rule",
            "scorer_v0_mal",
        ),
        "average_score",
        "Every baseline that MAL data supports: random order (a floor), popularity, average score,"
        " item-item similarity, content similarity, the sequel rule (Job D only) and scorer v0 on MAL"
        " data.",
    ),
    "llm": Config(
        "llm",
        (
            "random",
            "popularity",
            "average_score",
            "item_item",
            "content_similarity",
            "sequel_rule",
            "scorer_v0_mal",
            "llm_ordering",
            "llm_open",
        ),
        "average_score",
        "Every baseline, plus the LLM baseline (both versions, Gemini 3.5 Flash-Lite), on the users"
        " whose answers have been collected with `nextcour llm-baseline`.",
    ),
    "simple": Config(
        "simple",
        ("random", "popularity", "average_score"),
        "average_score",
        "The three simplest methods, as first run in E4.",
    ),
}


# --- Measuring -------------------------------------------------------------------------------


def _decade(year: float) -> str:
    if math.isnan(year):
        return "unknown"
    return "before 1990" if year < 1990 else f"{int(year) // 10 * 10}s"


def job_a(user: EvalUser, catalog: Catalog, scores: np.ndarray) -> dict[str, float]:
    """Ordering accuracy among the targets, overall and within each slice, and Job D's."""
    out = {
        "A ordering accuracy": metrics.ordering_accuracy(user.labels, scores),
        "D ordering accuracy": (
            metrics.ordering_accuracy(user.labels[user.sequel], scores[user.sequel])
            if user.sequel.sum() > 1
            else float("nan")
        ),
    }
    groups: dict[str, np.ndarray] = {
        "new franchises": ~user.sequel,
        "aired after the cutoff": catalog.aired_after_cutoff[user.targets],
        "aired before the cutoff": ~catalog.aired_after_cutoff[user.targets],
    }
    for band in ("popular", "middle", "niche"):
        groups[band] = catalog.band[user.targets] == band
    decades = np.array([_decade(y) for y in catalog.year[user.targets]], dtype=object)
    for decade in sorted(set(decades) - {"unknown"}):
        groups[decade] = decades == decade
    for name, mask in groups.items():
        out[f"A: {name}"] = (
            metrics.ordering_accuracy(user.labels[mask], scores[mask]) if mask.sum() > 1 else float("nan")
        )
    group = "with dates" if user.has_dates else "without dates"
    out[f"A: users {group}"] = out["A ordering accuracy"]
    return out


def discovery_candidates(user: EvalUser, catalog: Catalog, base: np.ndarray) -> np.ndarray:
    """Entry points that had aired, minus the history and every franchise already started."""
    keep = np.ones(len(base), dtype=bool)
    keep[np.isin(base, user.seen)] = False
    if user.started:
        keep[np.isin(catalog.franchise[base], np.fromiter(user.started, dtype=np.int64))] = False
    return base[keep]


@dataclass
class TopTen:
    """Beyond accuracy: what the top 10 looks like, summed over users."""

    users: int = 0
    popular: float = 0.0
    years: list[float] = field(default_factory=list)
    hours: float = 0.0
    shown: set[int] = field(default_factory=set)

    def add(self, top: np.ndarray, catalog: Catalog) -> None:
        self.users += 1
        self.popular += float(catalog.top_percent[top].mean())
        self.years.extend(catalog.year[top][~np.isnan(catalog.year[top])].tolist())
        self.hours += float(catalog.watch_hours[top].sum())
        self.shown.update(top.tolist())

    def summary(self, candidates: int) -> dict[str, float]:
        return {
            "share of the top 10 in the most popular 1%": self.popular / self.users
            if self.users
            else math.nan,
            "median year of the top 10": float(np.median(self.years)) if self.years else math.nan,
            "watch hours of the top 10": self.hours / self.users if self.users else math.nan,
            "catalog coverage (shows ever in a top 10)": len(self.shown) / candidates
            if candidates
            else math.nan,
        }


def discovery_gains(user: EvalUser, candidates: np.ndarray) -> np.ndarray:
    """Each candidate's gain: its enjoyment if it's a target the user enjoyed, else 0."""
    enjoyed = user.labels >= ENJOYED
    gain = np.zeros(len(candidates))
    where = np.searchsorted(candidates, user.targets[enjoyed])  # candidates are sorted positions
    found = (where < len(candidates)) & (
        candidates[np.minimum(where, len(candidates) - 1)] == user.targets[enjoyed]
    )
    gain[where[found]] = user.labels[enjoyed][found]
    return gain


def job_b(
    user: EvalUser,
    catalog: Catalog,
    candidates: np.ndarray,
    gain: np.ndarray,
    scores: np.ndarray,
    top_ten: TopTen,
) -> dict[str, float]:
    relevant = gain > 0
    out: dict[str, float] = {}
    order = metrics.rank(scores, catalog.tiebreak[candidates])
    top_ten.add(candidates[order[:DISCOVERY_K]], catalog)
    if not relevant.any():
        return {"B nDCG@10": math.nan, "B Recall@100": math.nan}
    out["B nDCG@10"] = metrics.ndcg(gain[order], gain[relevant], DISCOVERY_K)
    out["B Recall@100"] = metrics.recall(relevant[order], int(relevant.sum()), RECALL_K)
    in_top = np.zeros(len(candidates), dtype=bool)
    in_top[order[:RECALL_K]] = True
    groups = {
        "aired after the cutoff": catalog.aired_after_cutoff[candidates],
        "aired before the cutoff": ~catalog.aired_after_cutoff[candidates],
    }
    for band in ("popular", "middle", "niche"):
        groups[band] = catalog.band[candidates] == band
    for name, mask in groups.items():
        hits = relevant & mask
        out[f"B recall: {name}"] = float(in_top[hits].mean()) if hits.any() else math.nan
    group = "with dates" if user.has_dates else "without dates"
    out[f"B nDCG: users {group}"] = out["B nDCG@10"]
    return out


@dataclass
class Run:
    config: Config
    part: str
    users: int
    per_user: dict[str, dict[str, np.ndarray]]  # method -> metric -> one value per user
    top_ten: dict[str, dict[str, float]]
    candidates: int
    seconds: float
    record: dict[str, Any]


def evaluate(
    config: Config,
    part: str = "dev",
    sample: int | None = None,
    cutoff: dt.date = split.CUTOFF,
    folder: Path | None = None,
    log: Callable[[str], None] = print,
) -> Run:
    if part not in ("dev", "test"):
        raise ValueError("evaluate dev or test users")
    start = time.time()
    folder = folder or split.split_dir(cutoff)
    con = duckdb.connect()
    catalog = load_catalog(folder, cutoff, con)
    franchise_of = split.franchise_lookup(cutoff, folder)
    record = json.loads((folder / "split.json").read_text(encoding="utf-8"))
    ctx = method_library.Context(folder, con, record["catalog_release"], franchise_of, log, part)
    methods = [method_library.BUILDERS[name](catalog, ctx) for name in config.methods]
    base = np.flatnonzero(catalog.available & catalog.entry_point)  # sorted, as discovery_gains needs
    rows: dict[str, list[dict[str, float]]] = {m.name: [] for m in methods}
    top_ten = {m.name: TopTen() for m in methods}
    users = 0
    for raw in split.load_users(part, limit=sample, cutoff=cutoff, folder=folder, con=con):
        user = prepare(raw, catalog, franchise_of)
        candidates = discovery_candidates(user, catalog, base)
        gain = discovery_gains(user, candidates)
        for method in methods:
            row: dict[str, float] = {}
            if "A" in method.jobs or "D" in method.jobs:
                for key, value in job_a(user, catalog, method.scores(user, user.targets)).items():
                    if key[0] in method.jobs:
                        row[key] = value
            if "B" in method.jobs:
                scores = method.scores(user, candidates)
                row.update(job_b(user, catalog, candidates, gain, scores, top_ten[method.name]))
            rows[method.name].append(row)
        users += 1
        if users % 1000 == 0:
            log(f"  {users:,} users, {time.time() - start:.0f} s")
    keys = sorted({key for method_rows in rows.values() for r in method_rows for key in r})
    per_user = {
        name: {key: np.array([r.get(key, math.nan) for r in method_rows]) for key in keys}
        for name, method_rows in rows.items()
    }
    return Run(
        config=config,
        part=part,
        users=users,
        per_user=per_user,
        top_ten={name: t.summary(len(base)) for name, t in top_ten.items()},
        candidates=len(base),
        seconds=time.time() - start,
        record=record,
    )


# --- Reporting -------------------------------------------------------------------------------

HEADLINE = ("A ordering accuracy", "B nDCG@10", "B Recall@100", "D ordering accuracy")
SLICE_ORDER = (
    "new franchises",
    "popular",
    "middle",
    "niche",
    "before 1990",
    "1990s",
    "2000s",
    "2010s",
    "aired before the cutoff",
    "aired after the cutoff",
    "users with dates",
    "users without dates",
)


def slice_order(key: str) -> tuple[str, int]:
    """Job first, then slices in the design note's order: franchises, popularity, age, cold start, dates."""
    job, _, name = key.partition(" ")
    name = name.split(": ", 1)[-1]
    return job, SLICE_ORDER.index(name) if name in SLICE_ORDER else len(SLICE_ORDER)


def summarize(run: Run, seed: int = 0) -> dict[str, Any]:
    boot = metrics.Bootstrap(run.users, seed=seed)
    reference = run.per_user[run.config.reference]
    out: dict[str, Any] = {}
    for name, series in run.per_user.items():
        out[name] = {}
        for key in sorted(series):
            estimate = boot.estimate(series[key])
            entry: dict[str, Any] = {"estimate": estimate.__dict__}
            if name != run.config.reference and key in reference:
                diff = boot.difference(series[key], reference[key])
                entry["difference"] = {**diff.__dict__, "verdict": diff.verdict}
            out[name][key] = entry
    return out


def _cell(entry: Mapping[str, Any]) -> str:
    e = entry["estimate"]
    if math.isnan(e["mean"]):
        return "—"
    return f"{e['mean']:.3f} ({e['low']:.3f} to {e['high']:.3f})"


def _difference_cell(entry: Mapping[str, Any]) -> str:
    d = entry.get("difference")
    if not d or math.isnan(d["mean"]):
        return "—"
    return f"{d['mean']:+.3f} ({d['low']:+.3f} to {d['high']:+.3f}), {d['verdict']}"


def markdown(run: Run, summary: Mapping[str, Any], commit: Mapping[str, Any]) -> str:
    methods = list(run.per_user)
    reference = run.config.reference
    first = summary[methods[0]]
    job_a_users = first["A ordering accuracy"]["estimate"]["users"]
    job_b_users = first["B nDCG@10"]["estimate"]["users"]
    job_d_users = first["D ordering accuracy"]["estimate"]["users"]
    dirty = " (uncommitted changes)" if commit.get("uncommitted_changes") else ""
    lines = [
        f"# Evaluation: {run.config.name} on {run.users:,} {run.part} users",
        "",
        f"- **Config:** `{run.config.name}`: {run.config.description}",
        f"- **Commit:** `{commit.get('commit', '?')}`{dirty}",
        f"- **Split:** {run.record['dataset']}, cutoff {run.record['cutoff']}, seed {run.record['seed']},"
        f" catalog {run.record['catalog_release']}",
        f"- **Users:** {run.users:,} {run.part} users; Job A defined for {job_a_users:,}, Job B for"
        f" {job_b_users:,} (the rest had no enjoyed new franchise after the cutoff), Job D for"
        f" {job_d_users:,} (two or more sequels with different labels)",
        f"- **Discovery candidates:** {run.candidates:,} entry points that had aired by {split.DATA_END}",
        f"- **Intervals:** 95%, paired bootstrap over users ({metrics.BOOTSTRAP_DRAWS:,} draws).",
        f"  Differences are against `{reference}`.",
        f"- **Time:** {run.seconds:.0f} s",
        "",
        "## Headline",
        "",
        "| Method | A. Ordering accuracy | B. nDCG@10 | B. Recall@100 | D. Ordering accuracy |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name in methods:
        cells = [_cell(summary[name][k]) for k in HEADLINE]
        lines.append(f"| `{name}` | {' | '.join(cells)} |")
    lines += [
        "",
        f"### Differences from `{reference}`",
        "",
        "| Method | Metric | Difference (95% interval) |",
    ]
    lines.append("| --- | --- | --- |")
    for name in methods:
        if name == reference:
            continue
        for key in HEADLINE:
            lines.append(f"| `{name}` | {key} | {_difference_cell(summary[name][key])} |")
    slices = sorted({k for s in summary.values() for k in s if k not in HEADLINE}, key=slice_order)
    lines += ["", "## Slices", "", f"| Slice | {' | '.join(f'`{m}`' for m in methods)} | Users |"]
    lines.append(f"| --- | {' | '.join('---' for _ in methods)} | --- |")
    for key in slices:
        cells = [_cell(summary[m][key]) if key in summary[m] else "—" for m in methods]
        users = summary[methods[0]].get(key, {}).get("estimate", {}).get("users", 0)
        lines.append(f"| {key} | {' | '.join(cells)} | {users:,} |")
    lines += [
        "",
        "## Beyond accuracy (discovery's top 10)",
        "",
        f"| Measure | {' | '.join(f'`{m}`' for m in methods)} |",
    ]
    lines.append(f"| --- | {' | '.join('---' for _ in methods)} |")
    for key, digits in zip(next(iter(run.top_ten.values())), (3, 0, 1, 3), strict=True):
        values = [run.top_ten[m][key] for m in methods]
        cells = ["—" if math.isnan(v) else f"{v:.{digits}f}" for v in values]
        lines.append(f"| {key} | {' | '.join(cells)} |")
    lines += [
        "",
        "## Experiment log stub",
        "",
        "```markdown",
        "### E<n> — <short name>",
        "",
        f"**Date:** {dt.date.today()} · **Phase:** 2 · **Commit:** `{commit.get('commit', '?')}`"
        f" · **Config:** `uv run nextcour eval {run.config.name}"
        f"{' --users test' if run.part == 'test' else ''}`",
        f"**Data:** {run.record['dataset']} split at {run.record['cutoff']} (seed {run.record['seed']}),"
        f" catalog {run.record['catalog_release']}; {run.users:,} {run.part} users",
        "",
        "1. **Hypothesis.**",
        "2. **Change.**",
        "3. **Metrics.** (the tables above)",
        "4. **Error analysis.** For a model experiment: my notes from the examples, written first,"
        " then each one checked on all dev users.",
        "5. **Decision.**",
        "```",
    ]
    return "\n".join(lines) + "\n"


def headline_lines(run: Run, summary: Mapping[str, Any]) -> Iterable[str]:
    yield f"{run.users:,} {run.part} users, {run.seconds:.0f} s"
    yield f"  {'method':<20}" + "".join(f"{k:>30}" for k in HEADLINE)
    for name in run.per_user:
        yield f"  {name:<20}" + "".join(f"{_cell(summary[name][k]):>30}" for k in HEADLINE)

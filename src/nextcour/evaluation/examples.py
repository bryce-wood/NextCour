"""Examples for my error analysis of a model experiment (experiment log, Error analysis for model
experiments): cases picked by fixed rules from dev users, so they aren't cherry-picked.

For scorer v1 (E7), on Job A's pairs (which include Job D's sequels):
1. **Most wrong:** the users with at least MIN_PAIRS pairs whose pairs v1 ordered worst, and for
   each, the three pairs it ordered most confidently the wrong way.
2. **Disagreements with average score:** pairs where v1 and average score ordered two shows in
   opposite directions, with average score at least MIN_GAP points apart, so it's a real
   disagreement. The pairs where v1 was most confident, at most one per user: half where v1 was
   right, half where it was wrong.

Each case shows the user's history in short, the shows, their labels, and v1's reasons. Users
appear as numbers, never MAL user names, and the file stays in data/results/ (MAL list data).
Never run on test users or my frozen set.
"""

from __future__ import annotations

import html
import itertools
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import numpy as np

from nextcour.core.entries import ListEntry
from nextcour.core.shows import Feature, Show
from nextcour.evaluation import harness, methods, metrics, split
from nextcour.models.scorer_v1 import ScorerV1, ScoreV1

MOST_WRONG = 15
DISAGREE_RIGHT, DISAGREE_WRONG = 8, 7
MIN_PAIRS = 30
MIN_GAP = 3.0  # average-score points (out of 100) for a disagreement to count
WORST_PAIRS = 3


@dataclass
class Pair:
    user: int  # the user's number in this file
    first: int  # catalog positions
    second: int
    labels: tuple[float, float]
    v1: tuple[float, float]
    average: tuple[float, float]

    @property
    def v1_right(self) -> bool:
        return (self.v1[0] > self.v1[1]) == (self.labels[0] > self.labels[1])

    @property
    def confidence(self) -> float:
        return abs(self.v1[0] - self.v1[1])


@dataclass
class Case:
    number: int
    accuracy: float
    pairs: int
    worst: list[Pair] = field(default_factory=list)


@dataclass
class Picked:
    """The chosen cases, and the users they need: their prepared profile and later entries."""

    catalog: harness.Catalog
    ctx: methods.Context
    worst: list[Case]
    disagreements: list[Pair]
    users: dict[int, harness.EvalUser]
    after: dict[int, tuple[ListEntry, ...]]


def _pairs(number: int, user: harness.EvalUser, v1: np.ndarray, average: np.ndarray) -> list[Pair]:
    out = []
    for i, j in itertools.combinations(range(len(user.targets)), 2):
        if user.labels[i] == user.labels[j]:
            continue
        out.append(
            Pair(
                number,
                int(user.targets[i]),
                int(user.targets[j]),
                (float(user.labels[i]), float(user.labels[j])),
                (float(v1[i]), float(v1[j])),
                (float(average[i]), float(average[j])),
            )
        )
    return out


def collect(folder: Path, log: Callable[[str], None] = print) -> Picked:
    """Pick the cases in one pass over the dev users, then load only the users they need."""
    con = duckdb.connect()
    catalog = harness.load_catalog(folder, split.CUTOFF, con)
    franchise_of = split.franchise_lookup(folder=folder)
    release = json.loads((folder / "split.json").read_text(encoding="utf-8"))["catalog_release"]
    ctx = methods.Context(folder, con, release, franchise_of, log)
    features = methods.load_features(catalog, ctx)
    v1 = methods.ScorerV1Mal(catalog, features.matrix)
    average = methods.AverageScore(catalog)
    cases: list[Case] = []
    disagreements: list[Pair] = []
    for number, raw in enumerate(split.load_users("dev", folder=folder, con=con), start=1):
        user = harness.prepare(raw, catalog, franchise_of)
        if len(user.targets) < 2:
            continue
        s_v1 = v1.scores(user, user.targets)
        pairs = _pairs(number, user, s_v1, average.scores(user, user.targets))
        if len(pairs) >= MIN_PAIRS:
            wrong = sorted((p for p in pairs if not p.v1_right), key=lambda p: -p.confidence)
            cases.append(
                Case(number, metrics.ordering_accuracy(user.labels, s_v1), len(pairs), wrong[:WORST_PAIRS])
            )
        opposite = [
            p
            for p in pairs
            if (p.v1[0] > p.v1[1]) != (p.average[0] > p.average[1])
            and p.v1[0] != p.v1[1]
            and abs(p.average[0] - p.average[1]) >= MIN_GAP
        ]
        if opposite:
            disagreements.append(max(opposite, key=lambda p: p.confidence))  # this user's strongest
        if number % 2000 == 0:
            log(f"  {number:,} users")
    worst = sorted(cases, key=lambda c: (c.accuracy, c.number))[:MOST_WRONG]
    right = sorted((p for p in disagreements if p.v1_right), key=lambda p: (-p.confidence, p.user))
    wrong = sorted((p for p in disagreements if not p.v1_right), key=lambda p: (-p.confidence, p.user))
    picked = right[:DISAGREE_RIGHT] + wrong[:DISAGREE_WRONG]
    needed = {c.number for c in worst} | {p.user for p in picked}
    users: dict[int, harness.EvalUser] = {}
    after: dict[int, tuple[ListEntry, ...]] = {}
    for number, raw in enumerate(split.load_users("dev", folder=folder, con=con), start=1):
        if number in needed:
            users[number] = harness.prepare(raw, catalog, franchise_of)
            after[number] = raw.after
    return Picked(catalog, ctx, worst, picked, users, after)


# --- Writing it up ---------------------------------------------------------------------------


def _shows(catalog: harness.Catalog, ctx: methods.Context) -> dict[int, Show]:
    """Show objects for the split's shows, with the features scorer v1 uses, for its reasons."""
    features = methods.load_features(catalog, ctx)
    titles = dict(
        ctx.con.execute(
            f"SELECT anilist_id, title FROM '{(ctx.folder / 'shows.parquet').as_posix()}'"
        ).fetchall()
    )
    shows = {}
    for i, anilist_id in enumerate(catalog.ids.tolist()):
        names = [features.names[j] for j in np.flatnonzero(features.matrix[i])]
        average = catalog.average[i]
        shows[anilist_id] = Show(
            anilist_id=anilist_id,
            title=html.unescape(titles.get(anilist_id) or f"Show {anilist_id}"),
            format=None,
            aired=None,
            average_score=None if np.isnan(average) else float(average),
            popularity=None,
            features=tuple(Feature(n.split(":")[0], n, n.split(":", 1)[1], 1.0) for n in names),
        )
    return shows


def _score(entry: ListEntry) -> str:
    return f"{entry.score / 10:g}/10" if entry.score > 0 else "no score"


def history_summary(history: Sequence[ListEntry]) -> list[str]:
    watched = [e for e in history if e.status in ("COMPLETED", "DROPPED", "CURRENT", "PAUSED")]
    scored = [e for e in watched if e.score > 0]
    top = sorted(scored, key=lambda e: -e.score)[:6]
    drops = [e for e in reversed(watched) if e.status == "DROPPED"][:4]
    median = f"{np.median([e.score for e in scored]) / 10:g}/10" if scored else "none"
    lines = [f"- **History:** {len(watched)} shows watched, {len(scored)} scored, median score {median}."]
    if top:
        lines.append(
            "- **Top scores:** " + "; ".join(f"{html.unescape(e.title)} ({_score(e)})" for e in top) + "."
        )
    if drops:
        lines.append(
            "- **Latest drops:** "
            + "; ".join(
                f"{html.unescape(e.title)} ({e.progress} of {e.episodes or '?'} eps, {_score(e)})"
                for e in drops
            )
            + "."
        )
    return lines


def _reasons(score: ScoreV1) -> str:
    """v1's reasons in a line: the crowd prior, the largest taste effects, and the series point."""
    parts = [f"crowd prior {score.crowd_prior:.2f}"]
    if score.series is not None:
        parts.append(f"series: your average enjoyment {score.series:.2f}, a {score.series_share:.0%} share")
    tastes = [c for c in score.contributions[1:] if c.kind not in ("series", "crowd_link")]
    if tastes:
        total = sum(c.value for c in tastes)
        largest = ", ".join(f"{c.label} ({'+' if c.value > 0 else '-'})" for c in tastes[:3])
        parts.append(f"taste match {total:+.3f}, most from {largest}")
    return "; ".join(parts)


def _pair_rows(
    pair: Pair,
    catalog: harness.Catalog,
    shows: Mapping[int, Show],
    scorer: ScorerV1,
    after: Mapping[int, ListEntry],
    sequel: Mapping[int, bool],
) -> list[str]:
    rows = []
    for position, label, v1 in zip((pair.first, pair.second), pair.labels, pair.v1, strict=True):
        anilist_id = int(catalog.ids[position])
        show = shows[anilist_id]
        year = "" if np.isnan(catalog.year[position]) else f", {int(catalog.year[position])}"
        kind = (
            "sequel" if sequel.get(position) else ("new show" if catalog.aired_after_cutoff[position] else "")
        )
        tag = f" ({kind})" if kind else ""
        entry = after.get(anilist_id)
        their = _score(entry) if entry else "?"
        if entry and entry.status == "DROPPED":
            their = f"dropped, {their}"
        avg = (
            "too few scores"
            if np.isnan(catalog.average[position])
            else f"{catalog.average[position] / 10:.2f}"
        )
        rows.append(
            f"| {show.title}{year}{tag} | {their} | {label:.2f} | {v1:.2f} | {avg}"
            f" | {_reasons(scorer.score(show))} |"
        )
    return rows


TABLE_HEAD = [
    "| Show | Their score | Enjoyment (the truth) | v1's prediction | Average score (train users)"
    " | v1's reasons |",
    "| --- | --- | --- | --- | --- | --- |",
]


def write(path: Path, picked: Picked) -> None:
    catalog, ctx, worst, users, after_entries = (
        picked.catalog,
        picked.ctx,
        picked.worst,
        picked.users,
        picked.after,
    )
    shows = _shows(catalog, ctx)
    lines = [
        "# Scorer v1: examples for my error analysis (E7)",
        "",
        "Dev users only, picked by fixed rules (see `src/nextcour/evaluation/examples.py`). Users are"
        " numbered; the numbers mean nothing outside this file. This is MAL list data: keep it here.",
        "",
        "How to read a pair: the two shows' enjoyment (from the user's whole list) says which they"
        " liked more. v1 got the pair wrong when its prediction orders them the other way. Its reasons"
        " are the parts of its score: the crowd prior (where the show's average falls among the user's"
        " shows), the series point (if the show continues a franchise they'd watched) and taste match"
        " (tags and studios of shows they liked or didn't).",
        "",
        "Write what you notice in E7's part 4 before reading Claude's analysis.",
        "",
        f"## Part 1. The {len(worst)} users v1 ordered worst",
        "",
    ]
    for case in worst:
        user = users[case.number]
        scorer = ScorerV1(user.profile, shows, ctx.franchise_of)
        after = {e.anilist_id: e for e in after_entries[case.number]}
        sequel = dict(zip(user.targets.tolist(), user.sequel.tolist(), strict=True))
        lines += [
            f"### User {case.number}: ordering accuracy {case.accuracy:.2f} over {case.pairs} pairs",
            "",
            *history_summary(user.history),
            "",
            f"Its {len(case.worst)} most confident wrong pairs:",
            "",
        ]
        for n, pair in enumerate(case.worst, start=1):
            lines += [
                f"**Pair {n}**",
                "",
                *TABLE_HEAD,
                *_pair_rows(pair, catalog, shows, scorer, after, sequel),
                "",
            ]
    lines += [
        f"## Part 2. Where v1 and average score disagreed most ({len(picked.disagreements)} pairs)",
        "",
    ]
    for n, pair in enumerate(picked.disagreements, start=1):
        user = users[pair.user]
        scorer = ScorerV1(user.profile, shows, ctx.franchise_of)
        after = {e.anilist_id: e for e in after_entries[pair.user]}
        sequel = dict(zip(user.targets.tolist(), user.sequel.tolist(), strict=True))
        verdict = "v1 was right" if pair.v1_right else "v1 was wrong"
        lines += [
            f"### Disagreement {n}: user {pair.user}, {verdict}",
            "",
            *history_summary(user.history),
            "",
            *TABLE_HEAD,
            *_pair_rows(pair, catalog, shows, scorer, after, sequel),
            "",
        ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

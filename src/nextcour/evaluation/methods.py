"""The methods the harness compares (design note, Baselines). Each scores catalog positions for one
user, from that user's history and from what train users did before the cutoff.

Choices for the MAL evaluation, settled 2026-10-24 before any of these ran:
- **Item-item similarity:** "people who watched your shows also watched". Cosine similarity between
  shows, from which train users watched both (watching, completed, paused or dropped) before the
  cutoff. A show's score is the sum of its similarities to every show the user watched. No tuned
  settings: no neighbor cutoff, no weighting.
- **Content similarity:** shows whose tags and studios match what the user enjoyed. Tags come from
  the catalog (anime-offline-database, about 30 per show, merged from several sites, genres
  included), studios from MAL 2018. Each is weighted by how rare it is (TF-IDF), and a feature
  needs at least two shows to count. The user's profile is the enjoyed shows (0.6 or more) in their
  history, each weighted by its enjoyment and its share of its franchise's weight, so a long
  franchise can't swamp it. A show's score is its cosine similarity to that profile.
- **The sequel rule** (Job D only): a next entry scores the user's enjoyment of the latest earlier
  entry of its franchise they'd watched; the robotic "watch the sequel to whatever you liked".
- **Scorer v0 on MAL data:** v0's formula exactly, with what MAL data has. Its crowd prior uses the
  train users' averages; its taste match uses the catalog's tags and MAL's studios, each at full
  weight, since there's no tag relevance. No staff and no crowd links: those need AniList's show
  details. It's vectorized, and tests check it against ScorerV0 itself.
"""

from __future__ import annotations

import itertools
import json
import time
from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import duckdb
import numpy as np

from nextcour.core import guardrail
from nextcour.ingest import catalog as catalog_data
from nextcour.models import scorer_v0
from nextcour.profile import enjoyment

if TYPE_CHECKING:
    from nextcour.evaluation.harness import Catalog, EvalUser

ALL_JOBS = ("A", "B", "D")
ENJOYED = 0.6
MIN_FEATURE_SHOWS = 2  # a tag or studio on a single show can't link two shows
WATCHED_STATUSES = ("CURRENT", "COMPLETED", "PAUSED", "DROPPED")


@dataclass
class Context:
    """What methods may be fitted from: the split's folder and its train table."""

    folder: Path
    con: duckdb.DuckDBPyConnection
    release: str
    franchise_of: Mapping[int, int]
    log: Callable[[str], None] = print
    part: str = "dev"  # whose answers the LLM methods read
    _features: ShowFeatures | None = None


class Method(Protocol):
    name: str
    jobs: tuple[str, ...]  # which jobs it's measured on

    def scores(self, user: EvalUser, shows: np.ndarray) -> np.ndarray:
        """A score for each catalog position in `shows`; higher means more likely to be enjoyed."""
        ...


class PerUser:
    """Methods that score the whole catalog at once keep the last user's scores, since the harness
    asks twice per user (once for each job)."""

    _user: str | None = None
    _all: np.ndarray | None = None

    def scores(self, user: EvalUser, shows: np.ndarray) -> np.ndarray:
        if self._user != user.name or self._all is None:
            self._all = self.score_all(user)
            self._user = user.name
        return self._all[shows]

    def score_all(self, user: EvalUser) -> np.ndarray:
        raise NotImplementedError


# --- The simplest baselines ------------------------------------------------------------------


@dataclass
class RandomOrder:
    """A floor: a fixed random order per user. Shows what chance looks like on each metric."""

    catalog: Catalog
    name: str = "random"
    jobs: tuple[str, ...] = ALL_JOBS

    def scores(self, user: EvalUser, shows: np.ndarray) -> np.ndarray:
        from nextcour.evaluation import split

        rng = np.random.default_rng(int(split.user_draw(user.name, seed=1) * 2**32))
        return rng.random(len(self.catalog))[shows]


@dataclass
class Popularity:
    """The shows the most train users had on their lists before the cutoff."""

    catalog: Catalog
    name: str = "popularity"
    jobs: tuple[str, ...] = ALL_JOBS

    def scores(self, user: EvalUser, shows: np.ndarray) -> np.ndarray:
        return self.catalog.members[shows].astype(float)


@dataclass
class AverageScore:
    """The shows train users scored highest before the cutoff. Too few scores: the middle."""

    catalog: Catalog
    name: str = "average_score"
    jobs: tuple[str, ...] = ALL_JOBS

    def __post_init__(self) -> None:
        known = self.catalog.average[~np.isnan(self.catalog.average)]
        self._filled = np.where(np.isnan(self.catalog.average), known.mean(), self.catalog.average)

    def scores(self, user: EvalUser, shows: np.ndarray) -> np.ndarray:
        return self._filled[shows]


# --- Item-item similarity --------------------------------------------------------------------


def fit_item_similarity(catalog: Catalog, ctx: Context, chunk: int = 16384) -> np.ndarray:
    """Cosine similarity between shows, from which train users watched both before the cutoff.

    About a minute, so the result is cached next to the split, keyed by when the split was built.
    """
    cache = ctx.folder / "item_similarity.npy"
    stamp = ctx.folder / "item_similarity.json"
    built = json.loads((ctx.folder / "split.json").read_text(encoding="utf-8"))["built_at"]
    if (
        cache.exists()
        and stamp.exists()
        and json.loads(stamp.read_text(encoding="utf-8")).get("split") == built
    ):
        return np.load(cache, mmap_mode="r")

    start = time.time()
    ctx.con.execute(
        f"CREATE OR REPLACE VIEW train AS SELECT * FROM '{(ctx.folder / 'train.parquet').as_posix()}'"
    )
    guardrail.check_training_table(ctx.con, "train")
    statuses = ", ".join(f"'{s}'" for s in WATCHED_STATUSES)
    rows = ctx.con.execute(
        f"""
        WITH positions AS (
            SELECT anilist_id, row_number() OVER (ORDER BY anilist_id) - 1 AS p
            FROM '{(ctx.folder / "shows.parquet").as_posix()}'
        ),
        users AS (
            SELECT user_name, row_number() OVER (ORDER BY user_name) - 1 AS u
            FROM (SELECT DISTINCT user_name FROM train)
        )
        SELECT users.u, positions.p FROM train JOIN users USING (user_name) JOIN positions USING (anilist_id)
        WHERE train.before AND train.status IN ({statuses})
        ORDER BY users.u
        """
    ).fetchnumpy()
    user, show = rows["u"].astype(np.int64), rows["p"].astype(np.int64)
    n = len(catalog)
    counts = np.zeros((n, n), dtype=np.float32)  # exact: counts stay far below 2**24
    total_users = int(user.max()) + 1 if len(user) else 0
    starts = range(0, total_users, chunk)
    bounds = np.searchsorted(user, [*starts, total_users])
    for first, (a, b) in zip(starts, itertools.pairwise(bounds), strict=True):
        if a == b:
            continue
        block = np.zeros((chunk, n), dtype=np.float32)
        block[user[a:b] - first, show[a:b]] = 1
        counts += block.T @ block
    norms = np.sqrt(counts.diagonal()).copy()
    norms[norms == 0] = 1
    counts /= norms[:, None]
    counts /= norms[None, :]
    np.fill_diagonal(counts, 0)
    np.save(cache, counts)
    stamp.write_text(json.dumps({"split": built, "rows": len(user)}), encoding="utf-8")
    ctx.log(f"  item similarity from {len(user):,} train entries in {time.time() - start:.0f} s (cached)")
    return counts


@dataclass
class ItemItem(PerUser):
    """People who watched your shows also watched: the sum of similarities to your watched shows."""

    similarity: np.ndarray
    name: str = "item_item"
    jobs: tuple[str, ...] = ALL_JOBS

    def score_all(self, user: EvalUser) -> np.ndarray:
        if not len(user.watched):
            return np.zeros(self.similarity.shape[0])
        return np.asarray(self.similarity[np.sort(user.watched)].sum(axis=0, dtype=np.float64))


# --- Show features: tags and studios ---------------------------------------------------------


@dataclass(frozen=True)
class ShowFeatures:
    names: tuple[str, ...]  # "tag:comedy", "studio:Sunrise"
    matrix: np.ndarray  # shows x features, 1 where the show has it


def load_features(catalog: Catalog, ctx: Context) -> ShowFeatures:
    """The catalog's tags and MAL's studios for every show, as a 0/1 matrix."""
    if ctx._features is not None:
        return ctx._features
    catalog_data.load(ctx.con, ctx.release)
    rows = ctx.con.execute(
        f"""
        WITH tags AS (
            SELECT a AS anilist_id, list_distinct(flatten(list(tags))) AS tags
            FROM catalog, unnest(anilist_ids) AS t(a) GROUP BY a
        )
        SELECT s.anilist_id, coalesce(tags.tags, []), coalesce(s.studios, [])
        FROM '{(ctx.folder / "shows.parquet").as_posix()}' s LEFT JOIN tags USING (anilist_id)
        """
    ).fetchall()
    per_show: dict[int, list[str]] = {}
    for anilist_id, tags, studios in rows:
        per_show[anilist_id] = [f"tag:{t}" for t in tags] + [f"studio:{s}" for s in studios]
    count: dict[str, int] = defaultdict(int)
    for names in per_show.values():
        for name in set(names):
            count[name] += 1
    kept = sorted(name for name, c in count.items() if c >= MIN_FEATURE_SHOWS)
    column = {name: j for j, name in enumerate(kept)}
    matrix = np.zeros((len(catalog), len(kept)), dtype=np.float32)
    for anilist_id, names in per_show.items():
        row = catalog.index[anilist_id]
        for name in names:
            if name in column:
                matrix[row, column[name]] = 1
    ctx._features = ShowFeatures(tuple(kept), matrix)
    return ctx._features


def _profile_rows(user: EvalUser, catalog: Catalog) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The profile's entries as catalog positions, with their franchise weights and enjoyment."""
    entries = [p for p in user.profile.entries if p.entry.anilist_id in catalog.index]
    return (
        np.array([catalog.index[p.entry.anilist_id] for p in entries], dtype=np.int64),
        np.array([p.weight for p in entries], dtype=np.float64),
        np.array([p.enjoyment for p in entries], dtype=np.float64),
    )


@dataclass
class ContentSimilarity(PerUser):
    """Shows whose tags and studios match the shows the user enjoyed, rarer features counting more."""

    catalog: Catalog
    tfidf: np.ndarray  # shows x features, each row scaled to length 1
    name: str = "content_similarity"
    jobs: tuple[str, ...] = ALL_JOBS

    @classmethod
    def fit(cls, catalog: Catalog, features: ShowFeatures) -> ContentSimilarity:
        shows_with = features.matrix.sum(axis=0)
        idf = np.log(len(catalog) / np.maximum(shows_with, 1)).astype(np.float32)
        weighted = features.matrix * idf
        lengths = np.linalg.norm(weighted, axis=1, keepdims=True)
        lengths[lengths == 0] = 1
        return cls(catalog, weighted / lengths)

    def score_all(self, user: EvalUser) -> np.ndarray:
        positions, weights, enjoyed = _profile_rows(user, self.catalog)
        keep = enjoyed >= ENJOYED
        if not keep.any():
            return np.zeros(len(self.catalog))
        profile = (weights[keep] * enjoyed[keep]) @ self.tfidf[positions[keep]]
        length = np.linalg.norm(profile)
        if not length:
            return np.zeros(len(self.catalog))
        return self.tfidf @ (profile / length).astype(self.tfidf.dtype)  # same type: no copy of the matrix


# --- The sequel rule -------------------------------------------------------------------------


@dataclass
class SequelRule:
    """For Continue a series: the user's enjoyment of the latest earlier entry they'd watched in the
    show's franchise. Measured only on Job D; anything outside a started franchise scores -1."""

    catalog: Catalog
    name: str = "sequel_rule"
    jobs: tuple[str, ...] = ("D",)

    def scores(self, user: EvalUser, shows: np.ndarray) -> np.ndarray:
        positions, _, enjoyed = _profile_rows(user, self.catalog)
        by_franchise: dict[int, list[tuple[int, float]]] = defaultdict(list)
        for p, e in zip(positions.tolist(), enjoyed.tolist(), strict=True):
            by_franchise[int(self.catalog.franchise[p])].append((int(self.catalog.aired_day[p]), e))
        out = np.full(len(shows), -1.0)
        for i, show in enumerate(shows.tolist()):
            franchise = int(self.catalog.franchise[show])
            if franchise not in user.started:
                continue
            watched = by_franchise.get(franchise, [])
            earlier = [(day, e) for day, e in watched if day < self.catalog.aired_day[show]]
            if earlier:
                out[i] = max(earlier)[1]  # the latest-aired earlier entry
            elif watched:
                out[i] = sum(e for _, e in watched) / len(watched)
            else:
                out[i] = 0.5  # started, but nothing labeled yet
        return out


# --- Scorer v0 on MAL data -------------------------------------------------------------------


def mid_ranks(values: np.ndarray, population: np.ndarray) -> np.ndarray:
    """scorer_v0.mid_rank for many values at once: the share below, ties counting half."""
    if not len(population):
        return np.full(len(values), scorer_v0.NEUTRAL)
    ordered = np.sort(population)
    below = np.searchsorted(ordered, values, side="left")
    equal = np.searchsorted(ordered, values, side="right") - below
    return (below + 0.5 * equal) / len(ordered)


@dataclass
class ScorerV0Mal(PerUser):
    """Scorer v0's formula on MAL data: crowd prior + 0.5 x taste match, with no crowd links."""

    catalog: Catalog
    features: np.ndarray  # shows x features, 1 where the show has it (full weight)
    name: str = "scorer_v0_mal"
    jobs: tuple[str, ...] = ALL_JOBS
    _totals: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.features = self.features.astype(np.float64)  # in 64 bits, as ScorerV0 adds them up
        self._totals = self.features.sum(axis=1)

    def crowd_priors(self, positions: np.ndarray, trust: float) -> np.ndarray:
        average = self.catalog.average
        history = average[positions][~np.isnan(average[positions])]
        priors = 0.5 + trust * (mid_ranks(np.nan_to_num(average), history) - 0.5)
        return np.where(np.isnan(average), scorer_v0.NEUTRAL, priors)

    def score_all(self, user: EvalUser) -> np.ndarray:
        profile = user.profile
        positions, weights, enjoyed = _profile_rows(user, self.catalog)
        trust = profile.scored / (profile.scored + enjoyment.SCORE_TRUST)
        priors = self.crowd_priors(positions, trust)
        if not len(positions):
            return priors
        residuals = enjoyed - priors[positions]
        rows = self.features[positions]
        evidence = weights @ rows
        sums = (weights * residuals) @ rows
        affinity = sums / (evidence + scorer_v0.TASTE_SHRINK)
        with np.errstate(invalid="ignore", divide="ignore"):
            taste = np.where(self._totals > 0, (self.features @ affinity) / self._totals, 0.0)
        return priors + scorer_v0.TASTE_WEIGHT * taste


def _llm():  # imported late: llm_baseline builds on this module
    from nextcour.evaluation import llm_baseline

    return llm_baseline


# --- Building them ---------------------------------------------------------------------------

BUILDERS: dict[str, Callable[[Catalog, Context], Method]] = {
    "random": lambda catalog, ctx: RandomOrder(catalog),
    "popularity": lambda catalog, ctx: Popularity(catalog),
    "average_score": lambda catalog, ctx: AverageScore(catalog),
    "item_item": lambda catalog, ctx: ItemItem(fit_item_similarity(catalog, ctx)),
    "content_similarity": lambda catalog, ctx: ContentSimilarity.fit(catalog, load_features(catalog, ctx)),
    "sequel_rule": lambda catalog, ctx: SequelRule(catalog),
    "scorer_v0_mal": lambda catalog, ctx: ScorerV0Mal(catalog, load_features(catalog, ctx).matrix),
    "llm_ordering": lambda catalog, ctx: _llm().ordering_method(catalog, ctx),
    "llm_open": lambda catalog, ctx: _llm().open_method(catalog, ctx),
}

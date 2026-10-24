"""The guardrail that keeps my own data out of training (CLAUDE.md; roadmap Phase 0).

The code does the remembering, so my AniList list can't reach a model by accident:
1. Training data comes only from the MAL datasets, never from an AniList list or any account
   of mine. Every training pipeline calls check_training_table() before it fits anything, and
   tests/test_guardrail.py fails if a pipeline doesn't.
2. When a frozen show is predicted, its history holds only entries whose outcome was known by
   the show's as-of date (the day I started it), and never the show itself or a later entry of
   its franchise. Other frozen shows may be in it if they came earlier: the test predicts each
   show from what came before, like real use.
3. The frozen test set never changes. tests/test_guardrail.py pins its checksum, and the list
   snapshot its labels come from is pinned by a checksum stored in it.

The frozen test set is personal, so it isn't in the public repo. Without it (in CI, or for anyone
else running NextCour), rule 1 still holds: training data must come from the MAL datasets.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb

from nextcour.core import paths
from nextcour.core.entries import FINISHED, MAX_WATCH, ListEntry

log = logging.getLogger(__name__)

TRAINING_DATASETS = frozenset({"mal-2018", "mal-2020", "mal-2023"})


class GuardrailViolation(AssertionError):
    """My own data reached training, or a frozen show's history holds something it mustn't."""


@dataclass(frozen=True)
class FrozenShow:
    anilist_id: int
    mal_id: int | None
    title: str
    franchise_id: int | None
    aired: dt.date | None
    as_of: dt.date  # predictions for this show may only use what was known by this day
    set: str  # "main" or "sequel"


@dataclass(frozen=True)
class FrozenSet:
    frozen_on: dt.date
    anilist_user_id: int
    anilist_user_name: str
    joined: dt.datetime  # when the user's first list entry was created (UTC)
    mal_user_names: tuple[str, ...]  # my MAL accounts, if any could be in the datasets
    catalog_release: str  # the release whose franchise IDs the shows use
    snapshot_file: str
    snapshot_sha256: str
    shows: tuple[FrozenShow, ...]

    @property
    def anilist_ids(self) -> frozenset[int]:
        return frozenset(s.anilist_id for s in self.shows)

    def of(self, which: str) -> list[FrozenShow]:
        return [s for s in self.shows if s.set == which]


def canonical_sha256(data: Any) -> str:
    """Checksum of JSON data, independent of formatting and line endings."""
    text = json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _date(value: str | None) -> dt.date | None:
    return dt.date.fromisoformat(value) if value else None


def load_frozen_set(path: Path | None = None) -> FrozenSet:
    raw = json.loads((path or paths.FROZEN_TEST_SET).read_text(encoding="utf-8"))
    shows = tuple(
        FrozenShow(
            anilist_id=s["anilist_id"],
            mal_id=s["mal_id"],
            title=s["title"],
            franchise_id=s["franchise_id"],
            aired=_date(s["aired"]),
            as_of=dt.date.fromisoformat(s["as_of"]),
            set=which,
        )
        for which in ("main", "sequel")
        for s in raw[which]
    )
    return FrozenSet(
        frozen_on=dt.date.fromisoformat(raw["frozen_on"]),
        anilist_user_id=raw["anilist_user"]["id"],
        anilist_user_name=raw["anilist_user"]["name"],
        joined=dt.datetime.fromisoformat(raw["anilist_user"]["joined"]),
        mal_user_names=tuple(raw["mal_user_names"]),
        catalog_release=raw["catalog_release"],
        snapshot_file=raw["snapshot"]["file"],
        snapshot_sha256=raw["snapshot"]["sha256"],
        shows=shows,
    )


def frozen_set_if_present() -> FrozenSet | None:
    return load_frozen_set() if paths.FROZEN_TEST_SET.exists() else None


# --- 1. Training data ------------------------------------------------------------------------


def check_training_table(
    con: duckdb.DuckDBPyConnection,
    table: str,
    *,
    dataset_column: str = "dataset",
    user_column: str = "user_name",
    frozen: FrozenSet | None = None,
) -> None:
    """Fail if a training table holds anything but MAL dataset rows, or rows from my accounts."""
    frozen = frozen or frozen_set_if_present()
    sources = {row[0] for row in con.execute(f"SELECT DISTINCT {dataset_column} FROM {table}").fetchall()}
    if not sources <= TRAINING_DATASETS:
        raise GuardrailViolation(f"{table} has rows from {sorted(sources - TRAINING_DATASETS)}")
    if frozen is None:
        log.warning("No frozen test set at %s; checked only that rows come from MAL", paths.FROZEN_TEST_SET)
        return
    mine = [name.lower() for name in (frozen.anilist_user_name, *frozen.mal_user_names)]
    count = con.execute(
        f"SELECT count(*) FROM {table} WHERE lower(CAST({user_column} AS VARCHAR)) IN (SELECT unnest(?))",
        [mine],
    ).fetchone()
    if count and count[0]:
        raise GuardrailViolation(f"{table} has {count[0]} rows from my own accounts")


# --- 2. History for a frozen show ------------------------------------------------------------


def known_by(entry: ListEntry, joined: dt.datetime) -> dt.date | None:
    """The latest day by which the entry's outcome (a finish or a drop) existed; None if unknown."""
    if entry.status not in FINISHED:
        return None  # still in progress: its current state is from now, not the past
    if entry.completed:
        return entry.completed
    if entry.started:
        return entry.started + MAX_WATCH
    if entry.created is not None and entry.created - joined < dt.timedelta(days=1):
        # Added undated on my first day on AniList: shows I'd already watched (confirmed 2026-10-11).
        return joined.date()
    return None


def history_for(
    show: FrozenShow,
    entries: Iterable[ListEntry],
    franchise_of: Mapping[int, int],
    joined: dt.datetime,
) -> list[ListEntry]:
    """The entries a prediction for `show` may use: finished and known by its as-of date."""
    allowed = []
    for entry in entries:
        if entry.anilist_id == show.anilist_id:
            continue
        known = known_by(entry, joined)
        if known is None or known > show.as_of:
            continue
        if _later_in_same_franchise(entry, show, franchise_of):
            continue
        allowed.append(entry)
    return allowed


def check_history(
    show: FrozenShow,
    history: Iterable[ListEntry],
    franchise_of: Mapping[int, int],
    joined: dt.datetime,
) -> None:
    """Fail if a history assembled for `show` holds the show itself or anything not yet known."""
    for entry in history:
        if entry.anilist_id == show.anilist_id:
            raise GuardrailViolation(f"{show.title} is in its own history")
        known = known_by(entry, joined)
        if known is None or known > show.as_of:
            raise GuardrailViolation(f"{entry.title} wasn't known by {show.as_of} ({show.title})")
        if _later_in_same_franchise(entry, show, franchise_of):
            raise GuardrailViolation(f"{entry.title} comes after {show.title} in its franchise")


def _later_in_same_franchise(entry: ListEntry, show: FrozenShow, franchise_of: Mapping[int, int]) -> bool:
    same = show.franchise_id is not None and franchise_of.get(entry.anilist_id) == show.franchise_id
    return bool(same and entry.aired and show.aired and entry.aired > show.aired)

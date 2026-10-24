"""Calibrating the labels for unscored entries (roadmap Phase 2; design note, Enjoyment).

An unscored entry gets the average relative score that users who did score gave in the same
situation: completed, completed and rewatched (not a catch-up), or dropped after a given share of
the episodes. Measured on train users only, from their history before the cutoff, with the same
label code the app uses, so neither dev nor test users feed the labels.

Relative scores here are the labels as enjoyment.py defines them, pulled toward 0.5 for users
with few scores, and the average is over entries, so users with long lists count for more.

This is a measurement, not tuning: the values come from what users did, and no metric is looked
at. The labels' code takes the measured values by hand, with the date they were measured.
"""

from __future__ import annotations

import datetime as dt
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import duckdb

from nextcour.core import guardrail
from nextcour.core.entries import ListEntry
from nextcour.evaluation import split
from nextcour.profile import builder, enjoyment

SAMPLE_USERS = 20_000  # train users, picked in the split's fixed random order
DROP_BINS = 10  # share of episodes watched, in tenths


def situation(entry: ListEntry, entries: Iterable[ListEntry], franchise_of: Mapping[int, int]) -> str | None:
    """Which unscored rule would label this entry if it had no score."""
    if entry.status in enjoyment.DONE:
        if entry.repeat > 0 and not enjoyment.is_catch_up(entry, entries, franchise_of):
            return "rewatched"
        return "completed"
    if entry.status == "DROPPED":
        if not entry.episodes:
            return "dropped, episode count unknown"
        tenth = min(int(enjoyment.share_watched(entry) * DROP_BINS), DROP_BINS - 1)
        return f"dropped, {tenth * 10}-{tenth * 10 + 10}% watched"
    return None


@dataclass
class Tally:
    scored: int = 0
    total: float = 0.0
    squares: float = 0.0
    unscored: int = 0  # how many entries this value would label

    def add(self, value: float) -> None:
        self.scored += 1
        self.total += value
        self.squares += value * value

    @property
    def mean(self) -> float:
        return self.total / self.scored if self.scored else float("nan")

    @property
    def standard_error(self) -> float:
        if self.scored < 2:
            return float("nan")
        variance = (self.squares - self.scored * self.mean**2) / (self.scored - 1)
        return math.sqrt(max(variance, 0.0) / self.scored)


def measure(histories: Iterable[Iterable[ListEntry]], franchise_of: Mapping[int, int]) -> dict[str, Tally]:
    tallies: dict[str, Tally] = defaultdict(Tally)
    for history in histories:
        kept = [e for e in history if e.format not in builder.EXCLUDED_FORMATS]
        relative = enjoyment.relative_scores(kept)
        for e in kept:
            where = situation(e, kept, franchise_of)
            if where is None:
                continue
            if e.anilist_id in relative:
                tallies[where].add(relative[e.anilist_id])
            else:
                tallies[where].unscored += 1
    return dict(tallies)


def fit_calibration(
    cutoff: dt.date = split.CUTOFF, sample: int = SAMPLE_USERS, folder: Path | None = None
) -> dict[str, Tally]:
    """Measure every situation on a fixed sample of train users, after the guardrail check."""
    folder = folder or split.split_dir(cutoff)
    con = duckdb.connect()
    con.execute(f"CREATE VIEW train AS SELECT * FROM '{(folder / 'train.parquet').as_posix()}'")
    guardrail.check_training_table(con, "train")
    franchise_of = split.franchise_lookup(cutoff, folder)
    users = split.load_users("train", limit=sample, cutoff=cutoff, folder=folder, con=con)
    return measure((u.history for u in users), franchise_of)


def order(name: str) -> tuple[int, str]:
    """Situations in reading order: completed, rewatched, then drops by share watched."""
    if name == "completed":
        return 0, ""
    if name == "rewatched":
        return 1, ""
    digits = name.split(", ")[1].split("-")[0]
    return (2, f"{int(digits):03d}") if digits.isdigit() else (3, name)

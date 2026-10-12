"""Profile builder: a user's history turned into labeled, weighted entries (roadmap Phase 1).

- Enjoyment labels v1, computed from the history alone (see enjoyment.py).
- The format filter: every format except music videos.
- Franchise weight (design note, Sequels and franchises): a long franchise counts more than a
  single show, but not once per cour. Its size is watch time in cour-equivalents, n = hours / 4.8,
  and its weight is log2(1 + n), shared across its entries by watch time. Rewatches add no hours.
  - Scorer v0 uses log2, the middle of the three candidate curves; Phase 5 picks the curve on dev
    data.
  - Anything shorter than a cour counts as one: a two-hour movie says as much about taste as a
    12-episode season. Size only adds weight beyond that.

It takes ListEntry objects and nothing AniList-specific except favourites (MAL users have none),
so the same code can run on MAL dataset users from Phase 2 on.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from nextcour.core.entries import ListEntry
from nextcour.profile import enjoyment

COUR_HOURS = 4.8  # 12 episodes of 24 minutes
DEFAULT_EPISODE_MINUTES = 24
EXCLUDED_FORMATS = frozenset({"MUSIC"})


@dataclass(frozen=True)
class ProfileEntry:
    entry: ListEntry
    enjoyment: float
    weight: float  # this entry's share of its franchise's weight


@dataclass(frozen=True)
class Profile:
    entries: tuple[ProfileEntry, ...]
    scored: int  # how many entries have a score; relative scores are pulled toward 0.5 by it


def watch_hours(entry: ListEntry) -> float:
    """Hours of one watch: rewatches add none."""
    done = entry.status in enjoyment.DONE
    episodes = (entry.episodes or entry.progress or 1) if done else entry.progress
    return episodes * (entry.duration or DEFAULT_EPISODE_MINUTES) / 60


def franchise_weight(hours: float) -> float:
    cours = max(hours / COUR_HOURS, 1.0)
    return math.log2(1 + cours)


def build(
    history: Iterable[ListEntry],
    franchise_of: Mapping[int, int],
    favourites: Iterable[int] = (),
) -> Profile:
    kept = [e for e in history if e.format not in EXCLUDED_FORMATS]
    labels = enjoyment.labels(kept, favourites, franchise_of)
    labeled = [e for e in kept if e.anilist_id in labels]

    franchises: dict[object, list[ListEntry]] = defaultdict(list)
    for e in labeled:
        franchises[franchise_of.get(e.anilist_id, ("anilist", e.anilist_id))].append(e)
    weights: dict[int, float] = {}
    for members in franchises.values():
        hours = {e.anilist_id: watch_hours(e) for e in members}
        total = sum(hours.values())
        franchise = franchise_weight(total)
        for e in members:
            share = hours[e.anilist_id] / total if total else 1 / len(members)
            weights[e.anilist_id] = franchise * share

    return Profile(
        entries=tuple(ProfileEntry(e, labels[e.anilist_id], weights[e.anilist_id]) for e in labeled),
        scored=sum(1 for e in labeled if e.score > 0),
    )

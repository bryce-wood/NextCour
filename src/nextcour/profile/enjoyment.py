"""Enjoyment labels v1 (design note, Enjoyment): one number from 0 to 1 per list entry.

An explicit judgment always wins. The first rule that matches decides:
1. Favorited: 1.
2. Scored, any status: the relative score.
3. Not scored, completed and rewatched (not a catch-up): 0.668.
4. Not scored, completed: 0.512.
5. Not scored, dropped: 0.184 after the first episode, up to 0.280 near the end; 0.355 when the
   show's episode count isn't known.
6. Not scored, and watching, paused or planning: no label.

Rules 3-5 were calibrated on 2026-10-24 (calibration.py): each is the average relative score that
scoring MAL 2018 users gave in the same situation, measured on 20,000 train users before the
2017-07-01 cutoff. The drop values are a straight line fitted to drops grouped by tenths of the
episodes watched. Placeholders before that: 0.8, 0.6, and 0.1 to 0.4. None of this is ever tuned to
improve a metric.

An unscored completion lands near the middle of the list (0.512), because completed shows are most
of what anyone scores, and relative scores average 0.5 by construction. So it doesn't count as
"enjoyed" (0.6 or more): finishing a show says it was about typical, no more.

Labels are computed from the entries passed in and nothing else. A profile for a past date passes
only that date's history, so even the relative scores use only scores given by then.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping

from nextcour.core.entries import WATCHED, ListEntry

FAVORITE = 1.0
REWATCHED = 0.668  # calibrated 2026-10-24 from 706 scored entries (standard error 0.008)
COMPLETED = 0.512  # from 1.6 million (standard error 0.0002)
DROPPED_EARLY, DROPPED_LATE = 0.184, 0.280  # after one episode and near the end, from 63,000
DROPPED_UNKNOWN = 0.355  # dropped, episode count not known (often a long-running show); from 914
SCORE_TRUST = 10  # with n scores, a percentile keeps n / (n + 10) of its distance from 0.5
UNKNOWN_SHARE = 0.5  # share watched when the episode count isn't known yet

DONE = frozenset({"COMPLETED", "REPEATING"})  # REPEATING: rewatching, so the first watch finished
IN_PROGRESS = frozenset({"CURRENT", "PAUSED"})


def share_watched(entry: ListEntry) -> float:
    if entry.status in DONE:
        return 1.0
    if not entry.episodes:
        return UNKNOWN_SHARE
    return min(1.0, entry.progress / entry.episodes)


def relative_scores(entries: Iterable[ListEntry]) -> dict[int, float]:
    """Each score's percentile among the user's own scores, pulled toward 0.5 when there are few.

    Ties count half, so equal scores get equal values. Scores on shows still in progress count
    with a weight equal to the share of episodes watched, since they can still change.
    """
    scored = [e for e in entries if e.score > 0 and e.status in WATCHED]
    weight_at: dict[float, float] = defaultdict(float)  # score -> total weight of entries with it
    for e in scored:
        weight_at[e.score] += share_watched(e) if e.status in IN_PROGRESS else 1.0
    total = sum(weight_at.values())
    if not total:
        return {}
    trust = len(scored) / (len(scored) + SCORE_TRUST)
    percentile: dict[float, float] = {}
    below = 0.0
    for score in sorted(weight_at):  # one pass over the distinct scores, so MAL-sized lists are fast
        percentile[score] = (below + 0.5 * weight_at[score]) / total
        below += weight_at[score]
    return {e.anilist_id: 0.5 + trust * (percentile[e.score] - 0.5) for e in scored}


def is_catch_up(entry: ListEntry, entries: Iterable[ListEntry], franchise_of: Mapping[int, int]) -> bool:
    """A rewatch before going on to a later entry of the same franchise."""
    franchise = franchise_of.get(entry.anilist_id)
    if franchise is None or entry.aired is None:
        return False
    return any(
        other.anilist_id != entry.anilist_id
        and franchise_of.get(other.anilist_id) == franchise
        and other.aired is not None
        and other.aired > entry.aired
        and other.status in WATCHED
        for other in entries
    )


def labels(
    entries: Iterable[ListEntry],
    favourites: Iterable[int] = (),
    franchise_of: Mapping[int, int] | None = None,
) -> dict[int, float]:
    """AniList ID -> enjoyment, for every entry that has a label."""
    entries = list(entries)
    franchise_of = franchise_of or {}
    favourites = set(favourites)
    relative = relative_scores(entries)
    out = {}
    for e in entries:
        if e.status not in WATCHED:
            continue
        if e.anilist_id in favourites:
            out[e.anilist_id] = FAVORITE
        elif e.anilist_id in relative:
            out[e.anilist_id] = relative[e.anilist_id]
        elif e.status in DONE and e.repeat > 0 and not is_catch_up(e, entries, franchise_of):
            out[e.anilist_id] = REWATCHED
        elif e.status in DONE:
            out[e.anilist_id] = COMPLETED
        elif e.status == "DROPPED" and not e.episodes:
            out[e.anilist_id] = DROPPED_UNKNOWN
        elif e.status == "DROPPED":
            out[e.anilist_id] = DROPPED_EARLY + (DROPPED_LATE - DROPPED_EARLY) * share_watched(e)
    return out

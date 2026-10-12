"""Enjoyment labels v1 (design note, Enjoyment): one number from 0 to 1 per list entry.

An explicit judgment always wins. The first rule that matches decides:
1. Favorited: 1.
2. Scored, any status: the relative score.
3. Not scored, completed and rewatched (not a catch-up): 0.8.
4. Not scored, completed: 0.6.
5. Not scored, dropped: 0.1 after the first episode, up to 0.4 near the end.
6. Not scored, and watching, paused or planning: no label.

Rules 3-5 are placeholders until Phase 2 calibrates them from MAL 2018. None of this is ever tuned
to improve a metric.

Labels are computed from the entries passed in and nothing else. A profile for a past date passes
only that date's history, so even the relative scores use only scores given by then.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from nextcour.core.entries import WATCHED, ListEntry

FAVORITE = 1.0
REWATCHED = 0.8  # placeholder
COMPLETED = 0.6  # placeholder
DROPPED_EARLY, DROPPED_LATE = 0.1, 0.4  # placeholders, after one episode and near the end
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
    weight = {e.anilist_id: share_watched(e) if e.status in IN_PROGRESS else 1.0 for e in scored}
    total = sum(weight.values())
    if not total:
        return {}
    trust = len(scored) / (len(scored) + SCORE_TRUST)
    out = {}
    for e in scored:
        below = sum(weight[o.anilist_id] for o in scored if o.score < e.score)
        equal = sum(weight[o.anilist_id] for o in scored if o.score == e.score)
        percentile = (below + 0.5 * equal) / total
        out[e.anilist_id] = 0.5 + trust * (percentile - 0.5)
    return out


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
        elif e.status == "DROPPED":
            out[e.anilist_id] = DROPPED_EARLY + (DROPPED_LATE - DROPPED_EARLY) * share_watched(e)
    return out

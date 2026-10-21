"""Job A's candidates: the shows on a user's Planning list that are ready to watch.

Watch order (decided 2026-10-21): a Planning entry is ready once it has started airing and
everything it follows is finished on the user's list. So an unstarted franchise offers its first
entry, and a started one offers only its next entry. What a show follows:
- its parent, if it's a side story or spin-off,
- its main-line prequels: TV series, movies and web series (ONA). Specials and OVAs that AniList
  calls prequels are extras: Another doesn't wait on its OVA.
- Either way, only entries released before it. A prequel released later is a story prequel, not
  a watch-order one (Fate/Zero came out after Fate/stay night), and AniList's relations have
  quirks (it lists a 2026 Slime movie as the parent of a 2022 one).

An entry that's still airing (or on hiatus) can't be finished by anyone, so being started is
enough: One Piece's films don't wait for One Piece to end. A Planning show that's airing now is
fine; one that hasn't started airing is held back.

Entries that aren't ready are held back, not dropped: they're listed apart with what they're
waiting on, so a franchise that doesn't need watching in order loses nothing.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from nextcour.core.shows import Related, Show

FINISHED = frozenset({"COMPLETED", "REPEATING"})
STARTED = FINISHED | {"CURRENT", "PAUSED"}
STILL_AIRING = frozenset({"RELEASING", "HIATUS"})
MAIN_LINE_FORMATS = frozenset({"TV", "TV_SHORT", "MOVIE", "ONA"})
NOT_RELEASED = "not released yet"


@dataclass(frozen=True)
class HeldBack:
    show: Show
    waiting_on: tuple[str, ...]  # why: each unfinished entry it follows, or that it isn't out yet


def must_follow(show: Show) -> list[Related]:
    """The entries that should be finished (or started, if still airing) before this show."""
    return [
        r
        for r in show.follows
        if (r.relation == "PARENT" or r.format in MAIN_LINE_FORMATS)
        and not (r.aired and show.aired and r.aired > show.aired)
    ]


def is_released(show: Show, today: dt.date) -> bool:
    """Has it started airing? Cached show data can be a few weeks old, so a start date that has
    certainly passed wins over a "not yet released" status."""
    if show.status != "NOT_YET_RELEASED":
        return True
    return show.aired_by is not None and show.aired_by <= today


def is_done(entry: Related, statuses: Mapping[int, str]) -> bool:
    """Is this entry far enough along on the user's list for what follows it?"""
    enough = STARTED if entry.status in STILL_AIRING else FINISHED
    return statuses.get(entry.anilist_id) in enough


def by_watch_order(
    planning: Iterable[Show], statuses: Mapping[int, str], today: dt.date
) -> tuple[list[Show], list[HeldBack]]:
    """Split Planning shows into those ready to watch and those held back.

    statuses: the user's list, AniList ID -> status (COMPLETED, PLANNING and so on).
    """
    ready: list[Show] = []
    held: list[HeldBack] = []
    for show in planning:
        if not is_released(show, today):
            held.append(HeldBack(show, (NOT_RELEASED,)))
            continue
        missing = [r for r in must_follow(show) if not is_done(r, statuses)]
        if not missing:
            ready.append(show)
            continue
        waiting = []
        for r in missing:
            status = statuses.get(r.anilist_id)
            waiting.append(f"{r.title} ({status.lower() if status else 'not on the list'})")
        held.append(HeldBack(show, tuple(waiting)))
    return ready, held

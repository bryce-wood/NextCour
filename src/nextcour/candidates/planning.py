"""Job A's candidates: the shows on a user's Planning list that are ready to watch.

Watch order (decided 2026-10-21): a Planning entry is ready once everything it follows is finished
on the user's list. So an unstarted franchise offers its first entry, and a started one offers
only its next entry. What a show follows:
- its parent, if it's a side story or spin-off,
- its main-line prequels: TV series, movies and web series (ONA). Specials and OVAs that AniList
  calls prequels are extras: Another doesn't wait on its OVA.
- Either way, only entries released before it. A prequel released later is a story prequel, not
  a watch-order one (Fate/Zero came out after Fate/stay night), and AniList's relations have
  quirks (it lists a 2026 Slime movie as the parent of a 2022 one).

Entries that aren't ready are held back, not dropped: they're listed apart with what they're
waiting on, so a franchise that doesn't need watching in order loses nothing.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from nextcour.core.shows import Related, Show

FINISHED = frozenset({"COMPLETED", "REPEATING"})
MAIN_LINE_FORMATS = frozenset({"TV", "TV_SHORT", "MOVIE", "ONA"})


@dataclass(frozen=True)
class HeldBack:
    show: Show
    waiting_on: tuple[str, ...]  # each unfinished entry it follows, with where it stands


def must_follow(show: Show) -> list[Related]:
    """The entries that should be finished before this show is recommended."""
    return [
        r
        for r in show.follows
        if (r.relation == "PARENT" or r.format in MAIN_LINE_FORMATS)
        and not (r.aired and show.aired and r.aired > show.aired)
    ]


def by_watch_order(
    planning: Iterable[Show], statuses: Mapping[int, str]
) -> tuple[list[Show], list[HeldBack]]:
    """Split Planning shows into those ready to watch and those waiting on an earlier entry.

    statuses: the user's list, AniList ID -> status (COMPLETED, PLANNING and so on).
    """
    ready: list[Show] = []
    held: list[HeldBack] = []
    for show in planning:
        missing = [r for r in must_follow(show) if statuses.get(r.anilist_id) not in FINISHED]
        if not missing:
            ready.append(show)
            continue
        waiting = []
        for r in missing:
            status = statuses.get(r.anilist_id)
            waiting.append(f"{r.title} ({status.lower() if status else 'not on the list'})")
        held.append(HeldBack(show, tuple(waiting)))
    return ready, held

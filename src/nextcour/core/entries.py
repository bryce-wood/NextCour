"""List entries: one show on one user's list, the same shape for AniList and (later) MAL users.

Dates are cautious on purpose, because they decide what counts as "the past" (no data leakage):
- AniList dates can be partial (a year, or a year and month). A start is read as the earliest day
  it could be, and a completion as the latest, so an entry never looks earlier than it was.
- Rewatches overwrite dates. An entry with repeat > 0 may carry its first watch's dates or a
  rewatch's, so its dates aren't trusted for ordering.
"""

from __future__ import annotations

import calendar
import datetime as dt
from dataclasses import dataclass
from typing import Any

# Statuses that end a watch, so the entry has an outcome.
FINISHED = frozenset({"COMPLETED", "DROPPED"})
WATCHED = frozenset({"COMPLETED", "DROPPED", "CURRENT", "PAUSED", "REPEATING"})

# No single watch takes longer than this; true of my list (checked 2026-10-11).
MAX_WATCH = dt.timedelta(days=365)


@dataclass(frozen=True)
class ListEntry:
    anilist_id: int
    mal_id: int | None
    title: str
    format: str | None
    aired: dt.date | None  # when the show started airing
    status: str
    score: float  # 0-100; 0 means unscored
    progress: int
    repeat: int  # how many times it was rewatched
    started: dt.date | None  # earliest day the recorded start could be
    completed: dt.date | None  # latest day the recorded completion could be
    created: dt.datetime  # when the entry was added to the list (UTC)
    episodes: int | None = None  # the show's episode count, if known
    duration: int | None = None  # minutes per episode, if known

    @property
    def watch_dates_trusted(self) -> bool:
        """Dates belong to the first watch only if no rewatch could have overwritten them."""
        return self.repeat == 0

    @property
    def watched_from(self) -> dt.date | None:
        """When the first watch began, as far as the dates can tell; None if they can't.

        A watch spanning more than MAX_WATCH has an untrustworthy start (usually set on the day
        the show was added), so the completion date stands in for it.
        """
        if not self.watch_dates_trusted:
            return None
        if self.started and self.completed and self.completed - self.started > MAX_WATCH:
            return self.completed
        return self.started or self.completed


def fuzzy_date(value: dict[str, int | None] | None, *, latest: bool) -> dt.date | None:
    """An AniList fuzzy date as a date: the earliest or latest day it could mean."""
    if not value or not value.get("year"):
        return None
    year = value["year"]
    month = value.get("month") or (12 if latest else 1)
    day = value.get("day") or (calendar.monthrange(year, month)[1] if latest else 1)
    return dt.date(year, month, day)


def from_anilist(entry: dict[str, Any]) -> ListEntry:
    media = entry["media"]
    title = media["title"]
    return ListEntry(
        anilist_id=media["id"],
        mal_id=media.get("idMal"),
        title=title.get("english") or title["romaji"],
        format=media.get("format"),
        aired=fuzzy_date(media.get("startDate"), latest=False),
        status=entry["status"],
        score=float(entry.get("score") or 0),
        progress=entry.get("progress") or 0,
        repeat=entry.get("repeat") or 0,
        started=fuzzy_date(entry.get("startedAt"), latest=False),
        completed=fuzzy_date(entry.get("completedAt"), latest=True),
        created=dt.datetime.fromtimestamp(entry["createdAt"], dt.UTC),
        episodes=media.get("episodes"),
        duration=media.get("duration"),
    )

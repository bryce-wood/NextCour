"""Builders for synthetic test data."""

import datetime as dt

from nextcour.core.entries import ListEntry

JOINED = dt.datetime(2021, 4, 7, 23, 50, tzinfo=dt.UTC)


def d(text: str | None) -> dt.date | None:
    return dt.date.fromisoformat(text) if text else None


def entry(
    anilist_id: int,
    *,
    aired: str | None = "2020-01-01",
    status: str = "COMPLETED",
    score: float = 80,
    repeat: int = 0,
    started: str | None = None,
    completed: str | None = None,
    created: dt.datetime | None = None,
    format: str = "TV",
    progress: int = 12,
    episodes: int | None = 12,
    duration: int | None = 24,
) -> ListEntry:
    return ListEntry(
        anilist_id=anilist_id,
        mal_id=None,
        title=f"Show {anilist_id}",
        format=format,
        aired=d(aired),
        status=status,
        score=score,
        progress=progress,
        repeat=repeat,
        started=d(started),
        completed=d(completed),
        created=created or dt.datetime(2022, 1, 1, tzinfo=dt.UTC),
        episodes=episodes,
        duration=duration,
    )

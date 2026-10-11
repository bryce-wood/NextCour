"""Freezing my "works for me" test set (roadmap Phase 0). It runs once; the result never changes.

Rules (the reasons are in the roadmap's Phase 0 and the design note's Evaluation section):
- The first entry of a franchise is the earliest-aired one I watched, since I watch in order.
- Main set: that first entry, for every franchise whose first watch has trustworthy dates
  (completed or dropped, never rewatched, dated). That's every franchise I started since
  joining AniList, because there aren't ~40 of them.
- Sequel set: every later entry of any franchise, watched in the same period, by the same rules.
- As-of date: the day the watch began. With no start date, the completion date, but never later
  than the start of a later entry in the franchise (I watch in order).
- Music videos are left out, as everywhere in NextCour.

Outputs:
- frozen/my_test_set.json: which shows, and their as-of dates. Private, so git ignores it.
- data/frozen/my_list_<date>.json, git-ignored: the list exactly as fetched, which holds the
  labels. Its checksum is recorded in the frozen set.
"""

from __future__ import annotations

import datetime as dt
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from nextcour.core import paths
from nextcour.core.entries import FINISHED, WATCHED, ListEntry, from_anilist
from nextcour.core.guardrail import FrozenShow, canonical_sha256

RULES_VERSION = 1


def as_of(entry: ListEntry, franchise_entries: Iterable[ListEntry], joined: dt.datetime) -> dt.date | None:
    """The day the first watch began, or None when the dates can't say."""
    own = entry.watched_from
    if own is None:
        return None
    if entry.created - joined < dt.timedelta(days=1) and own <= joined.date():
        return None  # dated on my first day on AniList: the import, not a watch
    if own == entry.started:
        return own
    # No trustworthy start: the completion, but I watch in order, so no later than the start of
    # an entry that aired after this one.
    later = [
        other.watched_from
        for other in franchise_entries
        if other is not entry and other.aired and entry.aired and other.aired > entry.aired
    ]
    return min([own, *(d for d in later if d is not None)])


def select(
    entries: Iterable[ListEntry],
    franchise_of: Mapping[int, int],
    joined: dt.datetime | None = None,
) -> tuple[list[FrozenShow], list[FrozenShow]]:
    """Choose the main and sequel sets from a list's entries.

    joined: when the user's first list entry was created; by default, the earliest entry here.
    """
    entries = list(entries)
    joined = joined or min(e.created for e in entries)
    by_franchise: dict[int | tuple[str, int], list[ListEntry]] = defaultdict(list)
    for entry in entries:
        if entry.status in WATCHED and entry.format != "MUSIC":
            # A show missing from the catalog is its own franchise.
            key = franchise_of.get(entry.anilist_id, ("anilist", entry.anilist_id))
            by_franchise[key].append(entry)

    def frozen(entry: ListEntry, date: dt.date, which: str) -> FrozenShow:
        return FrozenShow(
            anilist_id=entry.anilist_id,
            mal_id=entry.mal_id,
            title=entry.title,
            franchise_id=franchise_of.get(entry.anilist_id),
            aired=entry.aired,
            as_of=date,
            set=which,
        )

    main: list[FrozenShow] = []
    later: list[tuple[ListEntry, dt.date]] = []
    for franchise in by_franchise.values():
        franchise.sort(key=lambda e: (e.aired or dt.date.max, e.anilist_id))
        first, *rest = franchise
        if first.status in FINISHED and (date := as_of(first, franchise, joined)):
            main.append(frozen(first, date, "main"))
        for entry in rest:
            if entry.status in FINISHED and (date := as_of(entry, franchise, joined)):
                later.append((entry, date))

    if not main:
        return [], []
    period_start = min(show.as_of for show in main)
    sequel = [frozen(entry, date, "sequel") for entry, date in later if date >= period_start]
    order = lambda show: (show.as_of, show.anilist_id)  # noqa: E731
    return sorted(main, key=order), sorted(sequel, key=order)


def _show_json(show: FrozenShow) -> dict[str, Any]:
    return {
        "anilist_id": show.anilist_id,
        "mal_id": show.mal_id,
        "title": show.title,
        "franchise_id": show.franchise_id,
        "aired": show.aired.isoformat() if show.aired else None,
        "as_of": show.as_of.isoformat(),
    }


def freeze(
    user: dict[str, Any],
    raw_entries: list[dict[str, Any]],
    franchise_of: Mapping[int, int],
    catalog_release: str,
    today: dt.date,
) -> dict[str, Any]:
    """Write the snapshot and the frozen set. Refuses to run twice."""
    if paths.FROZEN_TEST_SET.exists():
        raise FileExistsError(f"{paths.FROZEN_TEST_SET} exists: the test set is frozen. Leave it alone.")

    entries = [from_anilist(e) for e in raw_entries]
    joined = min(e.created for e in entries)
    main, sequel = select(entries, franchise_of, joined)

    snapshot = {"fetched_on": today.isoformat(), "user": user, "entries": raw_entries}
    snapshot_path = paths.FROZEN_SNAPSHOT_DIR / f"my_list_{today.isoformat()}.json"
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=1), encoding="utf-8")

    frozen = {
        "about": "My frozen 'works for me' test set. Never tune on it; never change it. "
        "Rules: src/nextcour/evaluation/frozen_set.py. Labels: the git-ignored snapshot.",
        "rules_version": RULES_VERSION,
        "frozen_on": today.isoformat(),
        "anilist_user": {"id": user["id"], "name": user["name"], "joined": joined.isoformat()},
        "mal_user_names": [],  # I've never had a MyAnimeList account (confirmed 2026-10-11)
        "catalog_release": catalog_release,
        "snapshot": {
            "file": snapshot_path.relative_to(paths.DATA).as_posix(),
            "sha256": canonical_sha256(snapshot),
        },
        "main": [_show_json(s) for s in main],
        "sequel": [_show_json(s) for s in sequel],
    }
    paths.FROZEN_TEST_SET.parent.mkdir(parents=True, exist_ok=True)
    paths.FROZEN_TEST_SET.write_text(
        json.dumps(frozen, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return frozen

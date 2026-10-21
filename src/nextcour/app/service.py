"""The Planning list, assembled (roadmap Phase 1): ranked by scorer v0, in watch order, explained.

Shared by `nextcour rank`, `nextcour explain`, the writer comparison and the web page, so they
always agree on what's ranked and why.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from nextcour.app.feedback import FeedbackStore
from nextcour.candidates.planning import HeldBack, by_watch_order
from nextcour.core import shows
from nextcour.core.entries import WATCHED, ListEntry, from_anilist
from nextcour.core.shows import Show
from nextcour.explain import evidence
from nextcour.explain.writer import TEMPLATE, Explanation, Item, Writer, template
from nextcour.ingest import anilist
from nextcour.models.scorer_v0 import Score, ScorerV0
from nextcour.profile import builder


@dataclass(frozen=True)
class Row:
    rank: int
    show: Show
    score: Score
    average_rank: int  # where sorting by average score alone would put it
    item: Item  # its evidence, for the writer
    # Context, not a reason: entries of the same series already on the list. Scorer v0 doesn't use
    # them, so the explanation can't cite them; the page shows them beside it.
    series: tuple[str, ...] = ()

    @property
    def watch_time(self) -> str | None:
        return watch_time(self.show)


@dataclass(frozen=True)
class Ranking:
    user: str
    rows: list[Row]
    held: list[HeldBack]
    hidden: list[Show]  # marked "not interested"
    history_size: int  # shows the profile was built from


def watch_time(show: Show) -> str | None:
    """'About 9 hours', so the commitment is visible."""
    minutes = show.watch_minutes
    if minutes is None:
        return None
    if minutes < 90:
        return f"about {minutes} minutes"
    hours = round(minutes / 60, 1)
    return f"about {hours:.1f} hours" if hours < 10 else f"about {round(minutes / 60)} hours"


class Planner:
    def __init__(
        self,
        franchise_of: Mapping[int, int],
        anilist_client: Callable[[], anilist.AniListClient],
        feedback: FeedbackStore,
        today: Callable[[], dt.date] = dt.date.today,
    ):
        self.franchise_of = franchise_of
        self._anilist = anilist_client
        self.feedback = feedback
        self._today = today

    def ranking(self, user: str) -> Ranking:
        """Fetch the list (cached for an hour) and rank its ready Planning shows.

        Raises anilist.NotFound if the user doesn't exist or their list is private.
        """
        with self._anilist() as client:
            account = anilist.fetch_user(client, user)
            entries = [from_anilist(e) for e in anilist.fetch_list(client, user)]
            media = anilist.fetch_show_details(client, [e.anilist_id for e in entries])
        details = {m["id"]: shows.from_anilist(m) for m in media}

        history = [e for e in entries if e.status != "PLANNING"]
        planning = [
            details[e.anilist_id]
            for e in entries
            if e.status == "PLANNING" and e.format not in builder.EXCLUDED_FORMATS and e.anilist_id in details
        ]
        hidden_ids = self.feedback.not_interested(account["name"])
        hidden = [s for s in planning if s.anilist_id in hidden_ids]
        statuses = {e.anilist_id: e.status for e in entries}
        ready, held = by_watch_order(
            [s for s in planning if s.anilist_id not in hidden_ids], statuses, self._today()
        )

        profile = builder.build(history, self.franchise_of, account["favourite_anime"])
        scorer = ScorerV0(profile, details)
        scores = sorted((scorer.score(s) for s in ready), key=lambda s: (-s.predicted, s.anilist_id))
        by_average = sorted(scores, key=lambda s: (-s.crowd_prior, s.anilist_id))
        average_rank = {s.anilist_id: i for i, s in enumerate(by_average, 1)}
        rows = []
        for i, score in enumerate(scores, 1):
            show = details[score.anilist_id]
            item = Item(
                show.anilist_id, show.title, tuple(evidence.build(score, show.title, self.franchise_of))
            )
            series = series_watched(show.anilist_id, entries, self.franchise_of)
            rows.append(Row(i, show, score, average_rank[score.anilist_id], item, series))
        return Ranking(account["name"], rows, held, hidden, len(profile.entries))


def series_watched(
    anilist_id: int, entries: Sequence[ListEntry], franchise_of: Mapping[int, int]
) -> tuple[str, ...]:
    """The other entries of this show's series that are on the list as watched, earliest first."""
    franchise = franchise_of.get(anilist_id)
    if franchise is None:
        return ()
    same = [
        e
        for e in entries
        if e.anilist_id != anilist_id and franchise_of.get(e.anilist_id) == franchise and e.status in WATCHED
    ]
    same.sort(key=lambda e: (e.aired is None, e.aired, e.anilist_id))
    return tuple(f"{e.title} ({_take(e)})" for e in same)


def _take(entry: ListEntry) -> str:
    status = {
        "CURRENT": "watching",
        "PAUSED": "paused",
        "DROPPED": "dropped",
        "REPEATING": "rewatching",
    }.get(entry.status)
    score = f"you gave it {entry.score:g}/100" if entry.score else "unscored"
    return f"{status}, {score}" if status else score


def explanations(writer: Writer, ranking: Ranking) -> dict[int, Explanation]:
    """Cached explanations where they exist, templates for the rest. Never calls a model."""
    items = [row.item for row in ranking.rows]
    found = writer.cached(ranking.user, items)
    return {
        item.anilist_id: found.get(item.anilist_id) or template(item, "not written yet") for item in items
    }


def write_missing(
    writer: Writer, ranking: Ranking, rows: Sequence[Row] | None = None
) -> dict[int, Explanation]:
    """Have the models write the explanations that aren't cached yet; returns what they wrote."""
    items = [row.item for row in (rows if rows is not None else ranking.rows)]
    found = writer.cached(ranking.user, items)
    missing = [item for item in items if item.anilist_id not in found]
    return writer.write(ranking.user, missing) if missing else {}


def failures(written: Mapping[int, Explanation]) -> str | None:
    """Why some explanations are still templates after a writing run, if any are."""
    notes = sorted({e.note for e in written.values() if e.writer == TEMPLATE and e.note})
    return "; ".join(notes) or None

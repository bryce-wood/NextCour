import datetime as dt

from helpers import d

from nextcour.candidates.planning import NOT_RELEASED, by_watch_order
from nextcour.core.shows import Related, Show

TODAY = dt.date(2026, 10, 21)


def show(anilist_id, aired="2020-01-01", follows=(), status="FINISHED", aired_by=None):
    return Show(
        anilist_id, f"Show {anilist_id}", "TV", d(aired), 80, 100,
        follows=tuple(follows), status=status, aired_by=d(aired_by or aired),
    )  # fmt: skip


def prequel(anilist_id, format="TV", aired="2019-01-01", status="FINISHED"):
    return Related(anilist_id, "PREQUEL", f"Show {anilist_id}", format, d(aired), status)


def parent(anilist_id, aired="2019-01-01", status="FINISHED"):
    return Related(anilist_id, "PARENT", f"Show {anilist_id}", "TV", d(aired), status)


def ids(shows):
    return [s.anilist_id for s in shows]


def split(planning, statuses):
    return by_watch_order(planning, statuses, TODAY)


def test_first_entries_and_next_entries_are_ready():
    statuses = {1: "PLANNING", 2: "COMPLETED", 3: "PLANNING", 5: "REPEATING", 6: "PLANNING"}
    planning = [show(1), show(3, follows=[prequel(2)]), show(6, follows=[prequel(5)])]
    ready, held = split(planning, statuses)
    assert ids(ready) == [1, 3, 6]
    assert held == []


def test_later_entries_wait_and_say_on_what():
    statuses = {2: "CURRENT", 3: "PLANNING", 4: "PLANNING", 7: "DROPPED", 8: "PLANNING"}
    planning = [
        show(3, follows=[prequel(2)]),  # the season before is still being watched
        show(4, follows=[prequel(3)]),  # the season before is only planned
        show(8, follows=[prequel(7)]),  # the season before was dropped
        show(9, follows=[prequel(99)]),  # the season before isn't on the list at all
    ]
    ready, held = split(planning, statuses)
    assert ready == []
    assert [h.waiting_on for h in held] == [
        ("Show 2 (current)",),
        ("Show 3 (planning)",),
        ("Show 7 (dropped)",),
        ("Show 99 (not on the list)",),
    ]


def test_specials_and_later_released_prequels_dont_hold_a_show_back():
    planning = [
        show(1, follows=[prequel(10, format="OVA")]),  # like Another and its OVA
        show(2, follows=[prequel(11, format="SPECIAL")]),
        show(3, aired="2006-01-01", follows=[prequel(12, aired="2011-10-01")]),  # Fate/Zero came later
        show(4, follows=[prequel(13, format="MOVIE")]),  # a movie in the main line does count
    ]
    ready, held = split(planning, {})
    assert ids(ready) == [1, 2, 3]
    assert [h.show.anilist_id for h in held] == [4]


def test_a_side_story_waits_for_its_parent():
    ready, held = split([show(11, follows=[parent(10)])], {11: "PLANNING"})
    assert ready == [] and held[0].waiting_on == ("Show 10 (not on the list)",)
    ready, _ = split([show(11, follows=[parent(10)])], {10: "COMPLETED", 11: "PLANNING"})
    assert ids(ready) == [11]
    # A "parent" released after the show is a data quirk, not a prerequisite.
    ready, _ = split([show(11, follows=[parent(12, aired="2026-02-27")])], {})
    assert ids(ready) == [11]


def test_a_show_still_airing_only_needs_to_be_started():
    film = show(21, follows=[parent(20, aired="1999-10-20", status="RELEASING")])  # One Piece
    for status in ("CURRENT", "PAUSED", "COMPLETED"):
        ready, _ = split([film], {20: status})
        assert ids(ready) == [21], status
    for status in ("PLANNING", "DROPPED", None):
        ready, _ = split([film], {20: status} if status else {})
        assert ready == [], status
    on_hiatus = show(22, follows=[parent(20, status="HIATUS")])
    assert ids(split([on_hiatus], {20: "CURRENT"})[0]) == [22]


def test_unreleased_shows_wait_but_airing_ones_are_fine():
    planning = [
        show(1, aired="2027-01-01", aired_by="2027-12-31", status="NOT_YET_RELEASED"),  # "2027"
        show(2, aired="2026-10-03", status="RELEASING"),  # this season, still airing
        # Cached as not yet released, but its full start date has passed: it's out.
        show(3, aired="2026-10-05", status="NOT_YET_RELEASED"),
        # Only "2026" is known: it could still be December, so it waits.
        show(4, aired="2026-01-01", aired_by="2026-12-31", status="NOT_YET_RELEASED"),
    ]
    ready, held = split(planning, {})
    assert ids(ready) == [2, 3]
    assert [(h.show.anilist_id, h.waiting_on) for h in held] == [(1, (NOT_RELEASED,)), (4, (NOT_RELEASED,))]

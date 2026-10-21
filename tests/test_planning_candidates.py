from helpers import d

from nextcour.candidates.planning import by_watch_order
from nextcour.core.shows import Related, Show


def show(anilist_id, aired="2020-01-01", follows=()):
    return Show(anilist_id, f"Show {anilist_id}", "TV", d(aired), 80, 100, follows=tuple(follows))


def prequel(anilist_id, format="TV", aired="2019-01-01"):
    return Related(anilist_id, "PREQUEL", f"Show {anilist_id}", format, d(aired))


def parent(anilist_id, aired="2019-01-01"):
    return Related(anilist_id, "PARENT", f"Show {anilist_id}", "TV", d(aired))


def ids(shows):
    return [s.anilist_id for s in shows]


def test_first_entries_and_next_entries_are_ready():
    statuses = {1: "PLANNING", 2: "COMPLETED", 3: "PLANNING", 5: "REPEATING", 6: "PLANNING"}
    planning = [show(1), show(3, follows=[prequel(2)]), show(6, follows=[prequel(5)])]
    ready, held = by_watch_order(planning, statuses)
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
    ready, held = by_watch_order(planning, statuses)
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
    ready, held = by_watch_order(planning, {})
    assert ids(ready) == [1, 2, 3]
    assert [h.show.anilist_id for h in held] == [4]


def test_a_side_story_waits_for_its_parent():
    ready, held = by_watch_order([show(11, follows=[parent(10)])], {11: "PLANNING"})
    assert ready == [] and held[0].waiting_on == ("Show 10 (not on the list)",)
    ready, _ = by_watch_order([show(11, follows=[parent(10)])], {10: "COMPLETED", 11: "PLANNING"})
    assert ids(ready) == [11]
    # A "parent" released after the show is a data quirk, not a prerequisite.
    ready, _ = by_watch_order([show(11, follows=[parent(12, aired="2026-02-27")])], {})
    assert ids(ready) == [11]

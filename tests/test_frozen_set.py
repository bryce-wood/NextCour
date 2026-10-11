"""The rules that chose my frozen test set, on synthetic lists."""

import datetime as dt

from helpers import JOINED, entry

from nextcour.evaluation.frozen_set import select

IMPORT = JOINED + dt.timedelta(minutes=5)  # created during my first session on AniList


def ids(shows):
    return [s.anilist_id for s in shows]


def test_first_entry_of_each_franchise_goes_in_the_main_set():
    entries = [
        entry(1, aired="2019-01-01", started="2022-01-01", completed="2022-01-10", created=IMPORT),
        entry(2, aired="2021-01-01", started="2022-02-01", completed="2022-02-05"),
        entry(3, aired="2018-01-01", started="2022-03-01", completed="2022-03-02"),
    ]
    main, sequel = select(entries, {1: 100, 2: 100, 3: 300}, JOINED)
    assert ids(main) == [1, 3]
    assert ids(sequel) == [2]
    assert main[0].as_of == dt.date(2022, 1, 1)


def test_a_rewatched_first_entry_keeps_the_franchise_out_of_the_main_set():
    entries = [
        entry(1, aired="2016-01-01", repeat=1, started="2026-01-20", completed="2026-01-31", created=IMPORT),
        entry(2, aired="2020-01-01", started="2026-01-31", completed="2026-02-04"),
        entry(9, aired="2021-01-01", started="2025-01-01", completed="2025-01-02"),
    ]
    main, sequel = select(entries, {1: 100, 2: 100, 9: 900}, JOINED)
    assert ids(main) == [9]
    assert ids(sequel) == [2]  # the later season still counts


def test_the_import_itself_is_not_a_watch():
    entries = [
        entry(1, completed="2021-04-07", created=IMPORT),  # dated by the import
        entry(2, started="2021-04-10", completed="2021-05-04", created=IMPORT),  # planned, watched later
        entry(3, created=IMPORT),  # undated
    ]
    main, _ = select(entries, {}, JOINED)
    assert ids(main) == [2]


def test_missing_start_is_bounded_by_the_next_entry():
    # Solo Leveling: season 1 has only a completion date, a day after season 2 started.
    entries = [
        entry(1, aired="2024-01-06", completed="2025-05-13"),
        entry(2, aired="2025-01-04", started="2025-05-12", completed="2025-05-20"),
    ]
    main, sequel = select(entries, {1: 100, 2: 100}, JOINED)
    assert main[0].as_of == dt.date(2025, 5, 12)
    assert sequel[0].as_of == dt.date(2025, 5, 12)


def test_a_known_start_is_kept_even_when_watched_out_of_air_order():
    # Jujutsu Kaisen 0 aired before season 3 but was watched after it started.
    entries = [
        entry(1, aired="2020-10-01", started="2021-05-01", completed="2021-06-01"),
        entry(2, aired="2021-12-24", started="2026-01-20", completed="2026-01-20", format="MOVIE"),
        entry(3, aired="2026-01-08", started="2026-01-10", completed="2026-03-26"),
    ]
    _, sequel = select(entries, {1: 100, 2: 100, 3: 100}, JOINED)
    assert {s.anilist_id: s.as_of for s in sequel}[2] == dt.date(2026, 1, 20)


def test_sequels_before_the_period_and_unfinished_or_music_entries_are_left_out():
    entries = [
        entry(1, aired="2010-01-01", started="2021-05-01", completed="2021-05-02"),
        entry(2, aired="2012-01-01", started="2021-01-01", completed="2021-01-05"),  # before the period
        entry(5, aired="2015-01-01", started="2021-06-01", completed="2021-07-01"),
        entry(6, aired="2016-01-01", started="2022-01-01", completed="2022-01-02"),
        entry(7, aired="2023-01-01", status="CURRENT", started="2023-01-01"),
        entry(8, aired="2024-01-01", started="2024-01-01", completed="2024-01-01", format="MUSIC"),
        entry(10, aired="2010-01-01", status="CURRENT", started="2022-05-22"),  # first entry in progress
        entry(11, aired="2022-08-06", started="2023-08-01", completed="2023-08-01"),
    ]
    franchise_of = {1: 100, 2: 100, 5: 500, 6: 500, 7: 500, 8: 500, 10: 1000, 11: 1000}
    main, sequel = select(entries, franchise_of, JOINED)
    assert ids(main) == [1, 5]  # sorted by as-of date; the period starts 2021-05-01
    assert ids(sequel) == [6, 11]

import datetime as dt

import pytest
from helpers import JOINED, d, entry

from nextcour.core.guardrail import FrozenShow
from nextcour.core.shows import Show
from nextcour.evaluation import works_for_me as wfm


def frozen(anilist_id, as_of, aired="2015-01-01", franchise_id=None):
    return FrozenShow(anilist_id, None, f"Show {anilist_id}", franchise_id, d(aired), d(as_of), "main")


def test_a_pair_shares_the_history_of_its_earlier_show():
    a, b = frozen(1, "2023-01-01"), frozen(2, "2024-01-01")
    entries = [
        entry(1, started="2023-01-01", completed="2023-01-01"),  # a itself, a one-day movie
        entry(2, started="2024-01-01", completed="2024-02-01"),  # b itself
        entry(3, completed="2022-06-01"),  # before both: kept
        entry(4, completed="2023-06-01"),  # between the two: too late for the pair
    ]
    date, history = wfm.shared_history(a, b, entries, {}, JOINED)
    assert date == dt.date(2023, 1, 1)
    assert [e.anilist_id for e in history] == [3]  # a was finished that day, but never counts


def test_clean_means_airing_for_six_months_by_the_pair_date():
    date = dt.date(2023, 1, 1)
    assert wfm.is_clean(frozen(1, "2023-01-01", aired="2022-06-01"), date)
    assert not wfm.is_clean(frozen(1, "2023-01-01", aired="2022-09-01"), date)
    assert not wfm.is_clean(frozen(1, "2023-01-01", aired=None), date)


def by_id_method(profile, details):
    """A toy method: a higher AniList ID scores higher. Ignores the history."""
    return lambda show: float(show.anilist_id)


def flat_method(profile, details):
    return lambda show: 0.0


def test_pairs_skip_equal_labels_and_score_both_shows_from_the_same_profile():
    shows = [frozen(1, "2023-01-01"), frozen(2, "2023-03-01"), frozen(3, "2023-05-01")]
    entries = [
        entry(1, score=50, started="2023-01-01", completed="2023-01-20"),
        entry(2, score=90, started="2023-03-01", completed="2023-03-20"),
        entry(3, score=90, started="2023-05-01", completed="2023-05-20"),
        entry(4, score=70, completed="2022-01-01"),
    ]
    details = {i: Show(i, f"Show {i}", "TV", None, 70, 1) for i in (1, 2, 3, 4)}
    methods = {"by_id": by_id_method, "flat": flat_method}
    pairs = wfm.judge_pairs(shows, entries, {}, JOINED, details, methods)
    assert [(p.first.anilist_id, p.second.anilist_id) for p in pairs] == [(1, 2), (1, 3)]  # 2 and 3 tie
    assert all(p.outcome("by_id") == 1.0 and p.outcome("flat") == 0.5 for p in pairs)

    summary = wfm.summarize(pairs, ("by_id", "flat"))
    assert (summary.pairs, summary.shows) == (2, 3)
    assert summary.correct == {"by_id": 2.0, "flat": 1.0}
    assert summary.difference == pytest.approx(0.5)
    assert summary.disagreements == 2
    assert summary.disagreements_won == {"by_id": 2, "flat": 0}


def test_bootstrap_is_repeatable_and_brackets_a_sure_difference():
    shows = [frozen(i, f"2023-0{i}-01") for i in range(1, 6)]
    pairs = [
        wfm.Pair(a, b, a.as_of, 0, True, (0.2, 0.8), {"good": (0.1, 0.9), "bad": (0.9, 0.1)})
        for i, a in enumerate(shows)
        for b in shows[i + 1 :]
    ]
    interval = wfm.bootstrap_interval(pairs, ("good", "bad"), draws=200, seed=1)
    assert interval == wfm.bootstrap_interval(pairs, ("good", "bad"), draws=200, seed=1)
    assert interval == (1.0, 1.0)

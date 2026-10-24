"""Calibrating the labels for unscored entries."""

import pytest
from helpers import entry

from nextcour.profile import calibration


def test_situation_names_the_rule_that_would_apply():
    assert calibration.situation(entry(1), [], {}) == "completed"
    assert calibration.situation(entry(1, repeat=1), [], {}) == "rewatched"
    later = entry(2, aired="2021-01-01")
    assert calibration.situation(entry(1, repeat=1), [later], {1: 5, 2: 5}) == "completed"  # a catch-up
    dropped = entry(1, status="DROPPED", progress=3, episodes=12)
    assert calibration.situation(dropped, [], {}) == "dropped, 20-30% watched"
    assert (
        calibration.situation(entry(1, status="DROPPED", progress=12), [], {}) == "dropped, 90-100% watched"
    )
    assert calibration.situation(entry(1, status="DROPPED", episodes=None), [], {}) == (
        "dropped, episode count unknown"
    )
    assert calibration.situation(entry(1, status="CURRENT"), [], {}) is None


def test_measure_averages_relative_scores_and_counts_the_unscored():
    history = [
        entry(1, score=90),
        entry(2, score=70),
        entry(3, score=50, status="DROPPED", progress=1),
        entry(4, score=0),  # unscored: counted, not averaged
        entry(5, score=80, format="MUSIC"),  # left out, as the profile builder does
    ]
    tallies = calibration.measure([history], {})
    assert tallies["completed"].scored == 2
    assert tallies["completed"].unscored == 1
    # Percentiles among 90, 70 and 50 are 5/6, 1/2 and 1/6, pulled toward 0.5 by 3 / (3 + 10).
    trust = 3 / 13
    assert tallies["completed"].mean == pytest.approx(0.5 + trust * ((5 / 6 + 1 / 2) / 2 - 0.5))
    assert tallies["dropped, 0-10% watched"].mean == pytest.approx(0.5 + trust * (1 / 6 - 0.5))


def test_situations_sort_in_reading_order():
    names = [
        "dropped, 10-20% watched",
        "rewatched",
        "dropped, episode count unknown",
        "dropped, 0-10% watched",
        "completed",
    ]
    assert sorted(names, key=calibration.order) == [
        "completed",
        "rewatched",
        "dropped, 0-10% watched",
        "dropped, 10-20% watched",
        "dropped, episode count unknown",
    ]

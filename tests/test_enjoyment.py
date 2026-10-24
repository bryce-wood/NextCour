import pytest
from helpers import entry

from nextcour.profile import enjoyment


def trust(n):
    return n / (n + enjoyment.SCORE_TRUST)


def test_relative_scores_are_percentiles_with_ties_counting_half():
    entries = [entry(1, score=60), entry(2, score=80), entry(3, score=80), entry(4, score=95)]
    relative = enjoyment.relative_scores(entries)
    # Percentiles: 60 -> 0.5/4, 80 -> (1 + 1)/4, 95 -> 3.5/4; then pulled toward 0.5.
    expected = {1: 0.125, 2: 0.5, 3: 0.5, 4: 0.875}
    for show, percentile in expected.items():
        assert relative[show] == pytest.approx(0.5 + trust(4) * (percentile - 0.5))


def test_in_progress_scores_count_by_share_watched():
    entries = [
        entry(1, score=50),
        entry(2, score=90, status="CURRENT", progress=3, episodes=12),
    ]
    relative = enjoyment.relative_scores(entries)
    # Weights 1 and 0.25: the 50 has 0.5 * 1 below-or-equal, out of 1.25.
    assert relative[1] == pytest.approx(0.5 + trust(2) * (0.5 / 1.25 - 0.5))


def test_planning_scores_are_ignored():
    assert enjoyment.relative_scores([entry(1, score=70, status="PLANNING")]) == {}


def test_label_rules_in_order():
    entries = [
        entry(1, score=40),  # scored: relative score
        entry(2, score=40, status="DROPPED"),  # a scored drop keeps its relative score
        entry(3, score=0, repeat=1),  # rewatched, no later entry
        entry(4, score=0),  # completed
        entry(5, score=0, status="DROPPED", progress=6, episodes=12),  # dropped halfway
        entry(6, score=0, status="CURRENT"),  # no label
        entry(7, score=0, status="PLANNING"),  # no label
        entry(8, score=30),  # a favourite: 1, whatever the score
        entry(9, score=0, status="DROPPED", progress=40, episodes=None),  # episode count unknown
    ]
    labels = enjoyment.labels(entries, favourites=[8])
    relative = enjoyment.relative_scores(entries)
    assert labels[1] == relative[1]
    assert labels[2] == relative[2]
    assert labels[3] == enjoyment.REWATCHED
    assert labels[4] == enjoyment.COMPLETED
    assert labels[5] == pytest.approx((enjoyment.DROPPED_EARLY + enjoyment.DROPPED_LATE) / 2)
    assert 6 not in labels and 7 not in labels
    assert labels[8] == enjoyment.FAVORITE
    assert labels[9] == enjoyment.DROPPED_UNKNOWN


def test_calibrated_labels_keep_their_order():
    """Rewatched above completed, completed above any drop, and later drops above earlier ones."""
    assert enjoyment.REWATCHED > enjoyment.COMPLETED > enjoyment.DROPPED_LATE > enjoyment.DROPPED_EARLY
    assert enjoyment.COMPLETED > enjoyment.DROPPED_UNKNOWN


def test_a_catch_up_rewatch_counts_as_a_plain_completion():
    entries = [
        entry(1, score=0, repeat=1, aired="2020-01-01"),
        entry(2, score=0, aired="2021-01-01"),  # the next season, watched after the rewatch
    ]
    franchise_of = {1: 10, 2: 10}
    assert enjoyment.labels(entries, franchise_of=franchise_of)[1] == enjoyment.COMPLETED
    assert enjoyment.labels(entries[:1], franchise_of=franchise_of)[1] == enjoyment.REWATCHED

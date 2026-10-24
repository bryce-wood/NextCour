"""Metrics and the paired bootstrap."""

import math

import numpy as np
import pytest

from nextcour.evaluation import metrics


def test_ordering_accuracy_counts_pairs_with_different_labels():
    labels = np.array([0.9, 0.5, 0.1])
    assert metrics.ordering_accuracy(labels, np.array([3.0, 2.0, 1.0])) == 1.0
    assert metrics.ordering_accuracy(labels, np.array([1.0, 2.0, 3.0])) == 0.0
    # One pair tied in the scores counts half: (1 + 0.5 + 1) / 3.
    assert metrics.ordering_accuracy(labels, np.array([2.0, 2.0, 1.0])) == pytest.approx(2.5 / 3)


def test_ordering_accuracy_skips_pairs_with_equal_labels():
    labels = np.array([0.5, 0.5, 0.1])
    # Only the two pairs with 0.1 count, and both are right.
    assert metrics.ordering_accuracy(labels, np.array([1.0, 2.0, 0.0])) == 1.0
    assert math.isnan(metrics.ordering_accuracy(np.array([0.5, 0.5]), np.array([1.0, 2.0])))


def test_ndcg_is_one_for_the_best_order_and_discounts_lower_ranks():
    gains = np.array([0.9, 0.7, 0.0, 0.0])
    assert metrics.ndcg(gains, gains[gains > 0], k=10) == pytest.approx(1.0)
    swapped = np.array([0.0, 0.9, 0.7, 0.0])
    expected = (0.9 / math.log2(3) + 0.7 / math.log2(4)) / (0.9 + 0.7 / math.log2(3))
    assert metrics.ndcg(swapped, gains[gains > 0], k=10) == pytest.approx(expected)
    assert metrics.ndcg(np.array([0.0, 0.9]), np.array([0.9]), k=1) == 0.0
    assert math.isnan(metrics.ndcg(np.zeros(3), np.array([]), k=10))


def test_recall_is_the_share_of_relevant_shows_in_the_top_k():
    ranked = np.array([True, False, True, False, True])
    assert metrics.recall(ranked, 3, k=2) == pytest.approx(1 / 3)
    assert metrics.recall(ranked, 3, k=5) == 1.0
    assert math.isnan(metrics.recall(ranked, 0))


def test_rank_breaks_ties_by_the_fixed_order():
    scores = np.array([1.0, 2.0, 2.0, 0.5])
    assert metrics.rank(scores, np.array([0, 5, 1, 2])).tolist() == [2, 1, 0, 3]


def test_bootstrap_estimates_and_paired_differences():
    rng = np.random.default_rng(1)
    a = rng.normal(0.6, 0.1, 2000)
    b = a - 0.05 + rng.normal(0, 0.01, 2000)  # worse by 0.05, user by user
    boot = metrics.Bootstrap(2000, draws=500, seed=0)
    estimate = boot.estimate(a)
    assert estimate.low < estimate.mean < estimate.high
    assert estimate.users == 2000
    diff = boot.difference(b, a)
    assert diff.mean == pytest.approx(-0.05, abs=0.002)
    assert diff.verdict == "worse"
    # Pairing makes the interval far narrower than either estimate's own.
    assert diff.high - diff.low < (estimate.high - estimate.low) / 5


def test_bootstrap_ignores_users_a_metric_is_undefined_for():
    values = np.array([1.0, np.nan, 0.0, np.nan])
    boot = metrics.Bootstrap(4, draws=200, seed=0)
    estimate = boot.estimate(values)
    assert estimate.mean == 0.5 and estimate.users == 2
    same = metrics.Difference(0.0, -0.1, 0.1)
    assert same.verdict == "no clear difference"

"""Metrics and confidence intervals for the evaluation harness (design note, Metrics).

Every metric is computed per user, then averaged over users, so each user counts once however
long their list is. Intervals come from a paired bootstrap over users: resample the users with
replacement, recompute every method's average on the same resample, and take the middle 95%.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

BOOTSTRAP_DRAWS = 2000


def ordering_accuracy(labels: np.ndarray, scores: np.ndarray) -> float:
    """Share of pairs with different labels that the scores put in the same order.

    A tie in the scores counts half. NaN when no pair has different labels.
    """
    label_order = np.sign(labels[:, None] - labels[None, :])
    score_order = np.sign(scores[:, None] - scores[None, :])
    pairs = np.triu(label_order != 0, k=1)
    if not pairs.any():
        return float("nan")
    agree = np.where(score_order == 0, 0.5, (score_order == label_order).astype(float))
    return float(agree[pairs].mean())


def ndcg(ranked_gains: np.ndarray, all_gains: np.ndarray, k: int = 10) -> float:
    """nDCG@k with linear gains: the top k's gains, discounted by rank, over the best possible."""
    discounts = 1 / np.log2(np.arange(2, k + 2))
    top = ranked_gains[:k]
    ideal = np.sort(all_gains)[::-1][:k]
    best = float((ideal * discounts[: len(ideal)]).sum())
    if best <= 0:
        return float("nan")
    return float((top * discounts[: len(top)]).sum()) / best


def recall(ranked_relevant: np.ndarray, relevant: int, k: int = 100) -> float:
    """Share of the relevant shows that made the top k. NaN when there are none."""
    if relevant <= 0:
        return float("nan")
    return float(ranked_relevant[:k].sum()) / relevant


def rank(scores: np.ndarray, tiebreak: np.ndarray) -> np.ndarray:
    """Positions in order of score, highest first. Ties go by `tiebreak`, a fixed random order,
    so neither old nor new shows win them by their IDs."""
    return np.lexsort((tiebreak, -scores))


@dataclass(frozen=True)
class Estimate:
    mean: float
    low: float
    high: float
    users: int  # users the metric is defined for

    def text(self, digits: int = 3) -> str:
        return f"{self.mean:.{digits}f} ({self.low:.{digits}f} to {self.high:.{digits}f})"


@dataclass(frozen=True)
class Difference:
    mean: float
    low: float
    high: float

    @property
    def verdict(self) -> str:
        """'better' or 'worse' only when the interval excludes zero."""
        if self.low > 0:
            return "better"
        if self.high < 0:
            return "worse"
        return "no clear difference"

    def text(self, digits: int = 3) -> str:
        return f"{self.mean:+.{digits}f} ({self.low:+.{digits}f} to {self.high:+.{digits}f})"


class Bootstrap:
    """Resampled user weights, shared by every metric and method so the comparisons are paired."""

    def __init__(self, users: int, draws: int = BOOTSTRAP_DRAWS, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.weights = rng.multinomial(users, np.full(users, 1 / users), size=draws).astype(np.float64)

    def _means(self, values: np.ndarray) -> np.ndarray:
        defined = ~np.isnan(values)
        totals = self.weights @ np.where(defined, values, 0.0)
        counts = self.weights @ defined.astype(np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            return totals / counts

    def _interval(self, values: np.ndarray) -> tuple[float, float, float]:
        """Mean and 95% interval; all NaN when no user has a value."""
        if np.isnan(values).all():
            return math.nan, math.nan, math.nan
        means = self._means(values)
        means = means[~np.isnan(means)]  # a resample can miss every user who has a value
        low, high = np.percentile(means, [2.5, 97.5])
        return float(np.nanmean(values)), float(low), float(high)

    def estimate(self, values: np.ndarray) -> Estimate:
        return Estimate(*self._interval(values), users=int((~np.isnan(values)).sum()))

    def difference(self, values: np.ndarray, reference: np.ndarray) -> Difference:
        both = ~np.isnan(values) & ~np.isnan(reference)
        return Difference(*self._interval(np.where(both, values - reference, np.nan)))

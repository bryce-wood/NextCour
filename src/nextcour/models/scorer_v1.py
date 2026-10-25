"""Scorer v1 (roadmap Phase 2; design note, Components): scorer v0 with its three known flaws fixed.

1. Personal points come from my enjoyment of a past show relative to my other shows (its enjoyment
   minus 0.5, my middle), not from residuals. A residual credits shows I liked more than the crowd
   did, even ones I liked less than usual; on shows with no crowd average, that ordered them worse
   than chance (E5).
2. Near-ties count for nothing: a past show within NEAR of my middle adds evidence but no push.
3. A series point: for a show in a franchise I've watched, my average enjoyment of that franchise's
   entries in my history is blended in, with a share of W / (W + SERIES_SHRINK), where W is the
   franchise's weight in my history (1 for one cour).

    base = crowd prior + 0.5 x taste match + 0.5 x crowd links       (as in v0, with the new points)
    predicted = (1 - share) x base + share x series average

The crowd prior, the features and every other constant are v0's. Written 2026-10-24 before it was
measured (E7); never tuned on the test set or my frozen shows.

Contributions add up to the prediction, as v0's do: the crowd prior, then each taste and link
effect scaled by (1 - share), then the series point, share x (series average - crowd prior).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from typing import Any

from nextcour.core.shows import Show
from nextcour.models import scorer_v0
from nextcour.models.scorer_v0 import (
    LINK_SHRINK,
    LINK_WEIGHT,
    TASTE_SHRINK,
    TASTE_WEIGHT,
    Contribution,
    link_strength,
    mid_rank,
)
from nextcour.profile import enjoyment
from nextcour.profile.builder import Profile, ProfileEntry

MIDDLE = 0.5
NEAR = 0.10  # a past show this close to my middle pushes neither way
SERIES_SHRINK = 1.0  # one cour of a franchise counts as much as everything else together


def push(enjoyment_value: float) -> float:
    """How far a past show sits from my middle, or 0 for a near-tie."""
    distance = enjoyment_value - MIDDLE
    return 0.0 if abs(distance) < NEAR else distance


def franchise_key(anilist_id: int, franchise_of: Mapping[int, int]) -> Hashable:
    """The profile builder's key: a show outside any franchise is its own."""
    return franchise_of.get(anilist_id, ("anilist", anilist_id))


@dataclass(frozen=True)
class ScoreV1:
    anilist_id: int
    predicted: float
    crowd_prior: float
    taste: float
    links: float
    series: float | None  # my average enjoyment of the franchise's entries, if I've watched any
    series_share: float  # how much of the prediction the series average makes up
    contributions: tuple[Contribution, ...]  # the crowd prior first, then the largest effect first


class ScorerV1:
    def __init__(self, profile: Profile, shows: Mapping[int, Show], franchise_of: Mapping[int, int]):
        self.shows = shows
        self.franchise_of = franchise_of
        self.history: list[tuple[ProfileEntry, Show]] = [
            (p, shows[p.entry.anilist_id]) for p in profile.entries if p.entry.anilist_id in shows
        ]
        self._trust = profile.scored / (profile.scored + enjoyment.SCORE_TRUST)
        self._averages = [s.average_score for _, s in self.history if s.average_score is not None]
        self.push = {s.anilist_id: push(p.enjoyment) for p, s in self.history}

        self._evidence: dict[str, float] = defaultdict(float)
        self._sum: dict[str, float] = defaultdict(float)
        self._carriers: dict[str, list[tuple[ProfileEntry, float]]] = defaultdict(list)
        self._franchises: dict[Hashable, list[ProfileEntry]] = defaultdict(list)
        for p, s in self.history:
            for f in s.features:
                self._evidence[f.key] += p.weight * f.weight
                self._sum[f.key] += p.weight * f.weight * self.push[s.anilist_id]
                self._carriers[f.key].append((p, f.weight))
            self._franchises[franchise_key(s.anilist_id, franchise_of)].append(p)

    def crowd_prior(self, show: Show) -> float:
        if show.average_score is None:
            return scorer_v0.NEUTRAL
        return 0.5 + self._trust * (mid_rank(show.average_score, self._averages) - 0.5)

    def affinity(self, feature_key: str) -> float:
        return self._sum[feature_key] / (self._evidence[feature_key] + TASTE_SHRINK)

    def series(self, show: Show) -> tuple[float, float | None, list[ProfileEntry]]:
        """(share, average enjoyment, entries) for the show's franchise in my history."""
        members = [
            p
            for p in self._franchises.get(franchise_key(show.anilist_id, self.franchise_of), [])
            if p.entry.anilist_id != show.anilist_id
        ]
        weight = sum(p.weight for p in members)
        if weight <= 0:
            return 0.0, None, []
        average = sum(p.weight * p.enjoyment for p in members) / weight
        return weight / (weight + SERIES_SHRINK), average, members

    def score(self, show: Show) -> ScoreV1:
        prior = self.crowd_prior(show)
        share, series, members = self.series(show)
        keep = 1.0 - share

        effects: list[Contribution] = []
        taste = 0.0
        total = sum(f.weight for f in show.features)
        for f in show.features:
            if f.key not in self._evidence:
                continue
            effect = f.weight * self.affinity(f.key) / total
            taste += effect
            effects.append(
                Contribution(
                    f.kind,
                    f.key,
                    f.name,
                    keep * TASTE_WEIGHT * effect,
                    {"relevance": f.weight, "role": f.role, "affinity": round(self.affinity(f.key), 3)},
                )
            )

        linked = [
            (p, s, link_strength(show.link_votes(s)))
            for p, s in self.history
            if s.anilist_id != show.anilist_id and show.link_votes(s) > 0
        ]
        denominator = sum(strength * p.weight for p, _, strength in linked) + LINK_SHRINK
        links = 0.0
        for p, s, strength in linked:
            effect = strength * p.weight * self.push[s.anilist_id] / denominator
            links += effect
            effects.append(
                Contribution(
                    "crowd_link",
                    f"show:{s.anilist_id}",
                    s.title,
                    keep * LINK_WEIGHT * effect,
                    {"votes": show.link_votes(s), **self._show_facts(p)},
                )
            )

        base = prior + TASTE_WEIGHT * taste + LINK_WEIGHT * links
        predicted = keep * base + share * series if series is not None else base
        if series is not None:
            effects.append(
                Contribution(
                    "series",
                    "series",
                    show.title,
                    share * (series - prior),
                    {
                        "average": round(series, 3),
                        "share": round(share, 3),
                        "shows": [self._show_facts(p) for p in members],
                    },
                )
            )
        prior_item = Contribution(
            "average_score",
            "average_score",
            show.title,
            prior,
            {"average_score": show.average_score, "history_size": len(self._averages)},
        )
        effects.sort(key=lambda c: abs(c.value), reverse=True)
        return ScoreV1(
            anilist_id=show.anilist_id,
            predicted=predicted,
            crowd_prior=prior,
            taste=taste,
            links=links,
            series=series,
            series_share=share,
            contributions=(prior_item, *effects),
        )

    def predict(self, show: Show) -> float:
        return self.score(show).predicted

    def _show_facts(self, p: ProfileEntry) -> dict[str, Any]:
        return {
            "title": p.entry.title,
            "anilist_id": p.entry.anilist_id,
            "my_score": p.entry.score or None,
            "status": p.entry.status,
            "enjoyment": round(p.enjoyment, 3),
            "push": round(self.push[p.entry.anilist_id], 3),
            "weight": round(p.weight, 3),
        }

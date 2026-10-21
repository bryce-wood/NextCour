"""Scorer v0 (roadmap Phase 1): a hand-weighted blend that predicts how much I'll enjoy a show.

Every part is in the units of the enjoyment label (0 to 1), so the parts can be added:
1. Crowd prior: where the show's AniList average score falls among the average scores of the
   shows in my history. If my taste matched the crowd's exactly, that's where the show would land
   among my own scores. It's coarser than the average score itself: shows whose averages fall
   between the same two history shows' averages get the same prior.
2. Taste match: how much more or less than the crowd's scores suggested I enjoyed past shows with
   this show's tags, genres, studios and key staff.
3. Crowd links: the same, for the past shows the crowd links to this one ("if you liked X, try
   this").

The personal parts work on residuals: my enjoyment of a past show minus its crowd prior. A studio
whose shows I liked only because they were good shows earns nothing, since the crowd prior already
counts that. Parts 2 and 3 estimate the same thing (how far I'll be from the crowd on this show),
so they're averaged:

    predicted enjoyment = crowd prior + 0.5 x taste match + 0.5 x crowd links

Sparse evidence is pulled toward zero (shrinkage): a tag seen on one show I liked says little.

The weights and constants were set once by reasoning on 2026-10-11, before the first "works for
me" result, and are never tuned on the test set. Phase 5's learned ranker replaces this scorer
only if it wins.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from nextcour.core.shows import Show
from nextcour.profile import enjoyment
from nextcour.profile.builder import Profile, ProfileEntry

TASTE_WEIGHT = 0.5
LINK_WEIGHT = 0.5
# Evidence, in full-weight shows, at which a feature's affinity keeps half its raw value.
TASTE_SHRINK = 3.0
# The same for crowd links, in certain links: one strong link from a show gets half its value.
LINK_SHRINK = 1.0
# A link with this many net votes counts half as much as a certain one.
LINK_VOTES_HALF = 10
NEUTRAL = 0.5  # the crowd prior when a show has no average score


@dataclass(frozen=True)
class Contribution:
    """One piece of evidence and how much it moved the prediction, in enjoyment units."""

    kind: str  # "average_score", "tag", "genre", "studio", "staff" or "crowd_link"
    key: str
    label: str  # what it's about, for people: a tag, a person, a show
    value: float
    facts: dict[str, Any] = field(default_factory=dict)  # the real numbers behind it


@dataclass(frozen=True)
class Score:
    anilist_id: int
    predicted: float
    crowd_prior: float
    taste: float
    links: float
    contributions: tuple[Contribution, ...]  # largest effect first


def mid_rank(value: float, population: Iterable[float]) -> float:
    """Share of the population below `value`, with ties counting half."""
    population = list(population)
    if not population:
        return NEUTRAL
    below = sum(1 for p in population if p < value)
    equal = sum(1 for p in population if p == value)
    return (below + 0.5 * equal) / len(population)


def link_strength(votes: int) -> float:
    return votes / (votes + LINK_VOTES_HALF) if votes > 0 else 0.0


class ScorerV0:
    def __init__(self, profile: Profile, shows: Mapping[int, Show]):
        self.shows = shows
        self.history: list[tuple[ProfileEntry, Show]] = [
            (p, shows[p.entry.anilist_id]) for p in profile.entries if p.entry.anilist_id in shows
        ]
        self._trust = profile.scored / (profile.scored + enjoyment.SCORE_TRUST)
        self._averages = [s.average_score for _, s in self.history if s.average_score is not None]
        self.residual = {s.anilist_id: p.enjoyment - self.crowd_prior(s) for p, s in self.history}

        self._evidence: dict[str, float] = defaultdict(float)  # feature -> weight of evidence
        self._sum: dict[str, float] = defaultdict(float)  # feature -> weighted sum of residuals
        self._carriers: dict[str, list[tuple[ProfileEntry, float]]] = defaultdict(list)
        for p, s in self.history:
            for f in s.features:
                self._evidence[f.key] += p.weight * f.weight
                self._sum[f.key] += p.weight * f.weight * self.residual[s.anilist_id]
                self._carriers[f.key].append((p, f.weight))

    def crowd_prior(self, show: Show) -> float:
        """Where the show's average score falls among my history's, on my enjoyment scale."""
        if show.average_score is None:
            return NEUTRAL
        percentile = mid_rank(show.average_score, self._averages)
        return 0.5 + self._trust * (percentile - 0.5)

    def average_percentile(self, show: Show) -> float | None:
        """Share of my history's average scores below this show's, ties counting half."""
        if show.average_score is None or not self._averages:
            return None
        return round(mid_rank(show.average_score, self._averages), 3)

    def affinity(self, feature_key: str) -> float:
        """How much more than the crowd suggested I enjoyed shows with this feature, shrunk."""
        return self._sum[feature_key] / (self._evidence[feature_key] + TASTE_SHRINK)

    def score(self, show: Show) -> Score:
        prior = self.crowd_prior(show)
        contributions = [
            Contribution(
                "average_score",
                "average_score",
                show.title,
                prior,
                {
                    "average_score": show.average_score,
                    "share_below": self.average_percentile(show),
                    "history_size": len(self._averages),
                },
            )
        ]

        taste = 0.0
        total = sum(f.weight for f in show.features)
        for f in show.features:
            if f.key not in self._evidence:
                continue
            effect = f.weight * self.affinity(f.key) / total
            taste += effect
            contributions.append(
                Contribution(
                    f.kind,
                    f.key,
                    f.name,
                    TASTE_WEIGHT * effect,
                    {
                        "relevance": f.weight,
                        "role": f.role,
                        "affinity": round(self.affinity(f.key), 3),
                        "shows": self._show_facts(self._carriers[f.key]),
                    },
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
            effect = strength * p.weight * self.residual[s.anilist_id] / denominator
            links += effect
            contributions.append(
                Contribution(
                    "crowd_link",
                    f"show:{s.anilist_id}",
                    s.title,
                    LINK_WEIGHT * effect,
                    {"votes": show.link_votes(s), **self._show_facts([(p, 1.0)])[0]},
                )
            )

        personal = sorted(contributions[1:], key=lambda c: abs(c.value), reverse=True)
        return Score(
            anilist_id=show.anilist_id,
            predicted=prior + TASTE_WEIGHT * taste + LINK_WEIGHT * links,
            crowd_prior=prior,
            taste=taste,
            links=links,
            contributions=(contributions[0], *personal),
        )

    def predict(self, show: Show) -> float:
        return self.score(show).predicted

    def _show_facts(self, carriers: list[tuple[ProfileEntry, float]]) -> list[dict[str, Any]]:
        return [
            {
                "title": p.entry.title,
                "anilist_id": p.entry.anilist_id,
                "my_score": p.entry.score or None,
                "status": p.entry.status,
                "average_score": self.shows[p.entry.anilist_id].average_score,
                "enjoyment": round(p.enjoyment, 3),
                "crowd_prior": round(p.enjoyment - self.residual[p.entry.anilist_id], 3),
                "residual": round(self.residual[p.entry.anilist_id], 3),
                "weight": round(p.weight * relevance, 3),
            }
            for p, relevance in carriers
        ]

"""The evidence list (design note, Explanations): why scorer v0 put a show where it did.

Each item is one fact with its real numbers, written as a plain sentence, and how much it moved
the predicted enjoyment. The items come straight from the scorer's contributions, which add up to
its prediction, so every reason an explanation gives is one that actually moved the score.

- The average score comes first: in scorer v0 it's the base everything else adjusts. Near the
  middle of the viewer's shows it's described as neutral, since it says little either way.
- Then the largest personal contributions (taste match and crowd links), up to MAX_PERSONAL.
  - Near-ties aren't reasons. A past show the viewer liked about as much as its average score
    predicts (an 84 against an 85 average) says nothing about them, so it isn't cited, even
    when the scorer counted it.
  - Crowd links from several entries of one series become one fact about the series.
- Effects are described in words ("a moderate point in its favor"), not decimals, so the writer
  has no meaningless numbers to quote.
- Staff facts carry the person's role, and the checker makes the writer keep it.

Leaving a reason out keeps an explanation faithful (everything it says still moved the score), but
it doesn't fix the scorer: scorer v1 (Phase 2) is where near-ties should stop counting at all.
"""

from __future__ import annotations

import hashlib
import json
import string
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from nextcour.models.scorer_v0 import Contribution, Score

MAX_PERSONAL = 5
MIN_EFFECT = 0.005  # smaller contributions don't move the order in practice
NEAR_TIE = 0.10  # residuals smaller than this: liked about as much as the average score predicts
CARRIERS = 2  # past shows named per taste-match fact
STRONG, MODERATE = 0.10, 0.03
# The average score's effect is its crowd prior minus 0.5, which is about 0.9 x (its percentile
# among the viewer's shows - 0.5). Between roughly the 35th and 65th percentiles it's neutral.
AVERAGE_MODERATE, AVERAGE_STRONG = 0.14, 0.28
# AniList's staff roles (core.shows.KEY_ROLES) as they read in a sentence.
ROLE_PHRASES = {
    "Director": "director",
    "Original Creator": "original creator",
    "Original Story": "original story author",
    "Series Composition": "series composition writer",
    "Character Design": "character designer",
    "Music": "music composer",
}


@dataclass(frozen=True)
class Evidence:
    id: str  # "a", "b", ...: unique within one show
    kind: str  # "average_score", "tag", "genre", "studio", "staff" or "crowd_link"
    label: str  # what it's about: the show, a tag, a person, a linked show or series
    fact: str  # one plain sentence with the real numbers, addressed to the viewer
    effect: float  # how much it moved the predicted enjoyment (0-1 units)
    role: str | None = None  # staff only: what they did on the show

    @property
    def described_effect(self) -> str:
        size = abs(self.effect)
        if self.kind == "average_score":
            if size < AVERAGE_MODERATE:
                return "about neutral"
            strength = "strong" if size >= AVERAGE_STRONG else "moderate"
        else:
            strength = "strong" if size >= STRONG else "moderate" if size >= MODERATE else "slight"
        return f"a {strength} point {'in its favor' if self.effect >= 0 else 'against it'}"

    @property
    def neutral(self) -> bool:
        return self.described_effect == "about neutral"


@dataclass(frozen=True)
class _Point:
    kind: str
    label: str
    fact: str
    effect: float
    role: str | None = None


def build(score: Score, title: str, franchise_of: Mapping[int, int] | None = None) -> list[Evidence]:
    """The evidence items for one scored show: the average score, then the largest personal
    points. franchise_of (AniList ID -> franchise) lets links from one series be grouped."""
    average, *personal = score.contributions
    kept = [c for c in personal if abs(c.value) >= MIN_EFFECT]
    points = _links(
        [c for c in kept if c.kind == "crowd_link" and abs(c.facts["residual"]) >= NEAR_TIE],
        franchise_of or {},
    )
    for c in kept:
        if c.kind != "crowd_link" and (examples := _carriers(c)):
            points.append(_Point(c.kind, c.label, _taste_fact(c, examples), c.value, c.facts.get("role")))
    points.sort(key=lambda p: abs(p.effect), reverse=True)
    items = [Evidence("a", "average_score", title, _average_fact(title, average), average.value - 0.5)]
    for letter, p in zip(string.ascii_lowercase[1:], points[:MAX_PERSONAL], strict=False):
        items.append(Evidence(letter, p.kind, p.label, p.fact, p.effect, p.role))
    return items


def evidence_hash(items: Sequence[Evidence]) -> str:
    """Identifies an evidence list: an explanation is rewritten only when this changes."""
    payload = [(e.kind, e.label, e.fact, e.described_effect) for e in items]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode()).hexdigest()


def _average_fact(title: str, c: Contribution) -> str:
    average, share = c.facts.get("average_score"), c.facts.get("share_below")
    if average is None:
        return f"AniList has no average score for {title} yet."
    if share is None:
        return f"{title} has an AniList average score of {average}."
    percent = round(100 * share)
    comparison = f"higher than {percent}%" if percent >= 50 else f"lower than {100 - percent}%"
    return f"{title} has an AniList average score of {average}, {comparison} of the shows you've watched."


# The personal facts compare where a show sits among the viewer's own shows with where its
# AniList average puts it (a residual), not the two raw numbers: a 73 can be high for someone
# whose scores run low. "Relative to your other shows" says so.
def _links(links: Sequence[Contribution], franchise_of: Mapping[int, int]) -> list[_Point]:
    """One point per linked show, or per series when several of its entries link here the same way."""
    groups: dict[object, list[Contribution]] = defaultdict(list)
    for c in links:
        series = franchise_of.get(c.facts["anilist_id"], ("show", c.facts["anilist_id"]))
        groups[(series, c.value >= 0)].append(c)
    points = []
    for members in groups.values():
        more = "more" if members[0].value >= 0 else "less"
        effect = sum(c.value for c in members)
        if len(members) == 1:
            c = members[0]
            fact = (
                f"AniList users who liked {c.label} recommend this show ({c.facts['votes']} votes), and, "
                f"relative to your other shows, you liked {c.label} {more} than AniList users did "
                f"({_score_and_average(c.facts)})."
            )
            points.append(_Point("crowd_link", c.label, fact, effect))
            continue
        series = min((c.label for c in members), key=len)
        votes = "; ".join(f"{c.label}: {c.facts['votes']} votes" for c in members)
        takes = "; ".join(f"{c.label}: {_score_and_average(c.facts)}" for c in members)
        fact = (
            f"AniList users who liked the {series} series recommend this show ({votes}), and, relative to "
            f"your other shows, you've liked that series {more} than AniList users did ({takes})."
        )
        points.append(_Point("crowd_link", f"{series} series", fact, effect))
    return points


def _taste_fact(c: Contribution, examples: Sequence[str]) -> str:
    more = "more" if c.value >= 0 else "less"
    such_as = f", such as {_join(examples)}"
    liked = f"relative to your other shows, you've liked {{shows}} {more} than AniList users did{such_as}"
    name = c.label
    if c.kind == "tag":
        relevance = round(100 * c.facts["relevance"])
        return (
            f"It's tagged {name} ({relevance}% relevance), and, {liked.format(shows='shows with that tag')}."
        )
    if c.kind == "genre":
        return f"It's in the {name} genre, and, {liked.format(shows=f'{name} shows')}."
    if c.kind == "studio":
        return f"It's made by {name}, and, {liked.format(shows=f'{name} shows')}."
    roles = [
        ROLE_PHRASES.get(r.strip(), r.strip().lower()) for r in (c.facts.get("role") or "staff").split(",")
    ]
    return f"{name} worked on it as its {_join(roles)}, and, {liked.format(shows=f'shows {name} worked on')}."


def _carriers(c: Contribution) -> list[str]:
    """The past shows that best show this feature's pull, in the direction it pulls. Near-ties
    aren't examples; a feature with no clear example isn't cited."""
    shows: list[Mapping[str, Any]] = c.facts.get("shows") or []
    sign = 1 if c.value >= 0 else -1
    same_way = [s for s in shows if sign * s["residual"] >= NEAR_TIE]
    same_way.sort(key=lambda s: s["weight"] * abs(s["residual"]), reverse=True)
    return [f"{s['title']} ({_score_and_average(s)})" for s in same_way[:CARRIERS]]


def _score_and_average(facts: Mapping[str, Any]) -> str:
    if facts.get("my_score"):
        yours = f"you gave it {_score(facts['my_score'])}/100"
    else:
        yours = "you finished it without scoring it"
    if facts.get("status") == "DROPPED":
        yours = (
            f"you dropped it and gave it {_score(facts['my_score'])}/100"
            if facts.get("my_score")
            else "you dropped it"
        )
    if facts.get("average_score") is None:
        return yours
    return f"{yours}; it averages {facts['average_score']}"


def _score(value: float) -> str:
    return f"{value:g}"


def _join(parts: Sequence[str]) -> str:
    return parts[0] if len(parts) == 1 else f"{', '.join(parts[:-1])} and {parts[-1]}"

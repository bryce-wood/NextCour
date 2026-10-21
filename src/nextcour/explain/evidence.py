"""The evidence list (design note, Explanations): why scorer v0 put a show where it did.

Each item is one fact with its real numbers, written as a plain sentence, and how much it moved
the predicted enjoyment. The items come straight from the scorer's contributions, which add up to
its prediction, so every reason an explanation gives is one that actually moved the score.

- The average score always comes first: in scorer v0 it's the base everything else adjusts.
- Then the largest personal contributions (taste match and crowd links), up to MAX_PERSONAL.
- Effects are described in words ("raises it moderately"), not decimals, so the writer has no
  meaningless numbers to quote.
"""

from __future__ import annotations

import hashlib
import json
import string
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from nextcour.models.scorer_v0 import Contribution, Score

MAX_PERSONAL = 5
MIN_EFFECT = 0.002  # smaller contributions don't move the order in practice
CARRIERS = 2  # past shows named per taste-match fact
STRONG, MODERATE = 0.10, 0.03


@dataclass(frozen=True)
class Evidence:
    id: str  # "a", "b", ...: unique within one show
    kind: str  # "average_score", "tag", "genre", "studio", "staff" or "crowd_link"
    label: str  # what it's about: the show, a tag, a person, a linked show
    fact: str  # one plain sentence with the real numbers, addressed to the viewer
    effect: float  # how much it moved the predicted enjoyment (0-1 units)

    @property
    def described_effect(self) -> str:
        size = abs(self.effect)
        strength = "strongly" if size >= STRONG else "moderately" if size >= MODERATE else "slightly"
        return f"{'raises' if self.effect >= 0 else 'lowers'} it {strength}"


def build(score: Score, title: str) -> list[Evidence]:
    """The evidence items for one scored show, largest effect first after the average score."""
    average, *personal = score.contributions
    chosen = [c for c in personal if abs(c.value) >= MIN_EFFECT][:MAX_PERSONAL]
    items = [Evidence("a", "average_score", title, _average_fact(title, average), average.value - 0.5)]
    for letter, c in zip(string.ascii_lowercase[1:], chosen, strict=False):
        items.append(Evidence(letter, c.kind, c.label, _personal_fact(c), c.value))
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
def _personal_fact(c: Contribution) -> str:
    more = "more" if c.value >= 0 else "less"
    if c.kind == "crowd_link":
        return (
            f"AniList users who liked {c.label} recommend this show ({c.facts['votes']} votes), and, "
            f"relative to your other shows, you liked {c.label} {more} than AniList users did "
            f"({_score_and_average(c.facts)})."
        )
    examples = _carriers(c)
    such_as = f", such as {_join(examples)}" if examples else ""
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
    role = c.facts.get("role") or "staff"
    return f"{name} ({role}) worked on it, and, {liked.format(shows=f'shows {name} worked on')}."


def _carriers(c: Contribution) -> list[str]:
    """The past shows that best show this feature's pull, in the direction it pulls."""
    shows: list[Mapping[str, Any]] = c.facts.get("shows") or []
    sign = 1 if c.value >= 0 else -1
    same_way = [s for s in shows if sign * s["residual"] > 0] or list(shows)
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

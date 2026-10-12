"""Baselines every model is compared against (design note, Baselines). Phase 2 adds the rest.

Each takes a profile and the show details and returns a function that scores a show, the shape
the evaluation code expects.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

from nextcour.core.shows import Show
from nextcour.profile.builder import Profile


def average_score(profile: Profile, shows: Mapping[int, Show]) -> Callable[[Show], float]:
    """Sort by AniList's average score, the same for everyone. Shows without one go last."""
    return lambda show: float(show.average_score if show.average_score is not None else -1)

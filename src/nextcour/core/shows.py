"""Show details: what the scorer knows about a show, parsed from AniList's show data.

Scorer v0 (roadmap Phase 1) uses three things:
- the crowd's average score,
- content features: tags with their relevance, genres, main studios and key staff,
- the crowd's recommendation links ("if you liked X, try Y") and their net votes.

AniList shows a recommendation on both shows' pages with the same rating (checked 2026-10-11),
so a link can be read from either side.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from nextcour.core.entries import fuzzy_date

# Roles that shape a whole show, not one episode. AniList adds qualifiers like "(eps 1-12)".
KEY_ROLES = frozenset(
    {"Director", "Original Creator", "Original Story", "Series Composition", "Character Design", "Music"}
)


@dataclass(frozen=True)
class Feature:
    kind: str  # "tag", "genre", "studio" or "staff"
    key: str  # unique across kinds, e.g. "tag:63" or "staff:95185"
    name: str  # for people to read
    weight: float  # how strongly the show has it: a tag's relevance (0-1); 1 for the rest
    role: str | None = None  # staff only, e.g. "Director, Series Composition"


@dataclass(frozen=True)
class Related:
    """A show this one follows from: a prequel, or the parent of a side story or spin-off."""

    anilist_id: int
    relation: str  # "PREQUEL" or "PARENT", read from this show's side
    title: str
    format: str | None
    aired: dt.date | None


@dataclass(frozen=True)
class Show:
    anilist_id: int
    title: str
    format: str | None
    aired: dt.date | None
    average_score: int | None  # AniList's average, 0-100
    popularity: int | None  # how many AniList users have it on their list
    features: tuple[Feature, ...] = ()
    recommendations: Mapping[int, int] = field(default_factory=dict)  # linked show -> net votes
    follows: tuple[Related, ...] = ()  # its prequels and parents, as AniList lists them

    def link_votes(self, other: Show) -> int:
        """Net votes for the crowd's link between this show and `other`, from either side."""
        return self.recommendations.get(other.anilist_id) or other.recommendations.get(self.anilist_id) or 0


def from_anilist(media: dict[str, Any]) -> Show:
    features: list[Feature] = []
    for tag in media.get("tags") or []:
        if tag["isMediaSpoiler"] or tag["isGeneralSpoiler"]:
            continue  # an explanation must never cite a spoiler
        features.append(Feature("tag", f"tag:{tag['id']}", tag["name"], tag["rank"] / 100))
    for genre in media.get("genres") or []:
        features.append(Feature("genre", f"genre:{genre}", genre, 1.0))
    for edge in (media.get("studios") or {}).get("edges") or []:
        if edge["isMain"]:
            studio = edge["node"]
            features.append(Feature("studio", f"studio:{studio['id']}", studio["name"], 1.0))

    roles: dict[int, list[str]] = {}
    names: dict[int, str] = {}
    for edge in (media.get("staff") or {}).get("edges") or []:
        role = edge["role"].split(" (")[0]
        if role in KEY_ROLES:
            person = edge["node"]
            names[person["id"]] = person["name"]["full"]
            if role not in roles.setdefault(person["id"], []):
                roles[person["id"]].append(role)
    for person, its_roles in roles.items():
        features.append(Feature("staff", f"staff:{person}", names[person], 1.0, ", ".join(its_roles)))

    recommendations = {
        node["mediaRecommendation"]["id"]: node["rating"]
        for node in (media.get("recommendations") or {}).get("nodes") or []
        if node.get("mediaRecommendation") and node["rating"] > 0
    }

    # Relations are read from this show's side: "PREQUEL: X" means X comes right before it.
    follows = [
        Related(
            anilist_id=edge["node"]["id"],
            relation=edge["relationType"],
            title=_title(edge["node"].get("title")) or f"AniList show {edge['node']['id']}",
            format=edge["node"].get("format"),
            aired=fuzzy_date(edge["node"].get("startDate"), latest=False),
        )
        for edge in (media.get("relations") or {}).get("edges") or []
        if edge["node"].get("type") == "ANIME" and edge["relationType"] in ("PREQUEL", "PARENT")
    ]

    title = media["title"]
    return Show(
        anilist_id=media["id"],
        title=title.get("english") or title["romaji"],
        format=media.get("format"),
        aired=fuzzy_date(media.get("startDate"), latest=False),
        average_score=media.get("averageScore") or media.get("meanScore"),
        popularity=media.get("popularity"),
        features=tuple(features),
        recommendations=recommendations,
        follows=tuple(follows),
    )


def _title(title: dict[str, str | None] | None) -> str | None:
    if not title:
        return None
    return title.get("english") or title.get("romaji")

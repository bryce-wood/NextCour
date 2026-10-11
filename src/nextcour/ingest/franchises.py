"""Franchise groups: a show's seasons, movies, side stories and specials, grouped together.

The catalog's related-show links are untyped, and crossovers (Isekai Quartet, collab commercials)
chain unrelated franchises together: on the 2026-27 release, one group swallows 5,723 shows. So:
1. Group shows by the catalog's untyped links.
2. Rebuild every group larger than OVERSIZED from AniList's typed relations. Groups holding shows
   we must get right (my list) are rebuilt whatever their size.
        - CHARACTER and OTHER relations (cameos, collabs, crossovers) are ignored.
        - Sequels, prequels, alternative versions and compilations form each franchise's backbone.
        - A side story, spin-off or recap joins its parent's franchise only if it has exactly one
          parent franchise. A crossover ("Mazinger Z vs. Devilman") has two, so it stands alone.
3. In rebuilt groups, a show without an AniList ID joins a franchise only if all its links point
   into that one franchise, by the same reasoning.

Output: data/processed/catalog-<release>/franchises.parquet with catalog_id, franchise_id
(the smallest catalog_id in the franchise) and method ('untyped' or 'typed').
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from itertools import pairwise

import duckdb

from nextcour.ingest import anilist, catalog

OVERSIZED = 10

# AniList relation types (version 2), read from the show that lists the relation.
BACKBONE = frozenset({"PREQUEL", "SEQUEL", "ALTERNATIVE", "COMPILATION", "CONTAINS"})
CHILD_OF_RELATED = frozenset({"PARENT"})  # the related show is this show's parent
PARENT_OF_RELATED = frozenset({"SIDE_STORY", "SPIN_OFF", "SUMMARY"})  # the related show derives from this one
# Ignored: CHARACTER and OTHER (crossovers, cameos, collabs), ADAPTATION and SOURCE (non-anime).


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[int, int] = {}

    def find(self, x: int) -> int:
        self.parent.setdefault(x, x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:  # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)

    def groups(self, items: Iterable[int]) -> dict[int, list[int]]:
        out: dict[int, list[int]] = defaultdict(list)
        for item in items:
            out[self.find(item)].append(item)
        return out


def untyped_groups(related: Mapping[int, list[int]]) -> dict[int, list[int]]:
    """Connected groups of shows over the catalog's untyped links."""
    uf = UnionFind()
    for show, neighbours in related.items():
        uf.find(show)
        for other in neighbours:
            uf.union(show, other)
    return uf.groups(related)


def typed_groups(
    members: list[int],
    anilist_ids: Mapping[int, list[int]],
    typed_edges: Iterable[tuple[int, int, str]],
    related: Mapping[int, list[int]],
) -> dict[int, list[int]]:
    """Split one untyped group using AniList's typed relations.

    typed_edges are (anilist_id, related anilist_id, relation type). Shows without an AniList ID
    attach through their untyped links, but only when those links all lead into one franchise.
    """
    in_group = set(members)
    by_anilist = {a: show for show in members for a in anilist_ids.get(show, [])}
    uf = UnionFind()
    for show in members:
        uf.find(show)
        ids = anilist_ids.get(show, [])
        for a, b in pairwise(ids):  # one catalog entry merging several AniList IDs
            uf.union(by_anilist[a], by_anilist[b])

    parents: dict[int, set[int]] = defaultdict(set)
    for a, b, relation in typed_edges:
        if a not in by_anilist or b not in by_anilist:
            continue
        show, related_show = by_anilist[a], by_anilist[b]
        if relation in BACKBONE:
            uf.union(show, related_show)
        elif relation in CHILD_OF_RELATED:
            parents[show].add(related_show)
        elif relation in PARENT_OF_RELATED:
            parents[related_show].add(show)

    # Attach every part with exactly one parent franchise; repeat, since attaching can make two
    # parents turn out to be one franchise.
    changed = True
    while changed:
        changed = False
        parent_roots: dict[int, set[int]] = defaultdict(set)
        for child, its_parents in parents.items():
            parent_roots[uf.find(child)].update(uf.find(p) for p in its_parents)
        for root, roots in parent_roots.items():
            roots.discard(root)
            if len(roots) == 1 and uf.find(root) != uf.find(next(iter(roots))):
                uf.union(root, roots.pop())
                changed = True

    for show in members:
        if anilist_ids.get(show):
            continue
        roots = {
            uf.find(other) for other in related.get(show, []) if other in in_group and anilist_ids.get(other)
        }
        if len(roots) == 1:
            uf.union(show, roots.pop())
    return uf.groups(members)


def build(
    con: duckdb.DuckDBPyConnection,
    client: anilist.AniListClient,
    must_type: Iterable[int] = (),
    release: str = catalog.RELEASE,
) -> None:
    """Build the `franchises` table from the `catalog` table and write it to Parquet.

    must_type: AniList IDs whose groups are rebuilt from typed relations whatever their size.
    """
    rows = con.execute("SELECT catalog_id, anilist_ids, related_ids FROM catalog").fetchall()
    anilist_ids = {show: ids for show, ids, _ in rows}
    related = {show: rel for show, _, rel in rows}
    show_of = {a: show for show, ids in anilist_ids.items() for a in ids}

    groups = untyped_groups(related)
    must = {show_of[a] for a in must_type if a in show_of}
    to_rebuild = [m for m in groups.values() if len(m) > OVERSIZED or must.intersection(m)]

    needed = [a for members in to_rebuild for show in members for a in anilist_ids[show]]
    edges: list[tuple[int, int, str]] = []
    for media in anilist.fetch_media_with_relations(client, needed):
        for edge in media["relations"]["edges"]:
            if edge["node"]["type"] == "ANIME":
                edges.append((media["id"], edge["node"]["id"], edge["relationType"]))

    assignments: list[tuple[int, int, str]] = []
    rebuilt = {show for members in to_rebuild for show in members}
    for members in groups.values():
        if members[0] not in rebuilt:
            assignments += [(show, min(members), "untyped") for show in members]
    for members in to_rebuild:
        for franchise in typed_groups(members, anilist_ids, edges, related).values():
            assignments += [(show, min(franchise), "typed") for show in franchise]

    shows, franchise_ids, methods = (list(column) for column in zip(*sorted(assignments), strict=True))
    con.execute(
        "CREATE OR REPLACE TABLE franchises AS SELECT unnest(?::INTEGER[]) AS catalog_id,"
        " unnest(?::INTEGER[]) AS franchise_id, unnest(?::VARCHAR[]) AS method",
        [shows, franchise_ids, methods],
    )
    out = catalog.processed_dir(release) / "franchises.parquet"
    con.execute(f"COPY franchises TO '{out.as_posix()}' (FORMAT parquet)")


def load(con: duckdb.DuckDBPyConnection, release: str = catalog.RELEASE) -> None:
    """Create a `franchises` view over the processed Parquet file."""
    path = (catalog.processed_dir(release) / "franchises.parquet").as_posix()
    con.execute(f"CREATE OR REPLACE VIEW franchises AS SELECT * FROM '{path}'")


def lookup(con: duckdb.DuckDBPyConnection, release: str = catalog.RELEASE) -> dict[int, int]:
    """AniList ID -> franchise ID. Needs the `catalog` table or view."""
    load(con, release)
    rows = con.execute(
        "SELECT a, franchise_id FROM franchises JOIN catalog USING (catalog_id), unnest(anilist_ids) AS t(a)"
    ).fetchall()
    return dict(rows)


def summary_lines(con: duckdb.DuckDBPyConnection, largest: int = 10) -> list[str]:
    """Franchise counts by method, and the largest franchises."""
    lines = []
    for method, shows, count in con.execute(
        "SELECT method, count(*), count(DISTINCT franchise_id) FROM franchises GROUP BY 1 ORDER BY 1"
    ).fetchall():
        lines.append(f"{method}: {shows:,} shows in {count:,} franchises")
    lines.append(f"Largest {largest}:")
    for size, titles in con.execute(
        "SELECT count(*), list(title ORDER BY year NULLS LAST, catalog_id)[1:3]"
        " FROM franchises JOIN catalog USING (catalog_id) GROUP BY franchise_id ORDER BY 1 DESC LIMIT ?",
        [largest],
    ).fetchall():
        lines.append(f"  {size:>4}  {'; '.join(titles)}")
    return lines

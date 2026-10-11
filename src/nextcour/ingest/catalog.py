"""The show catalog, from anime-offline-database.

One row per show with its IDs on AniList and MyAnimeList, so the MAL datasets and AniList can be
joined without crawling anything. Outputs, under data/processed/catalog-<release>/:
- catalog.parquet: one row per show,
- mapping.parquet: every (anilist_id, mal_id) pair that the catalog links.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb
import httpx

from nextcour.core import paths

RELEASE = "2026-27"  # 2026-07-04; check https://github.com/manami-project/anime-offline-database/releases
FILE_NAME = "anime-offline-database-minified.json"
DOWNLOAD_URL = "https://github.com/manami-project/anime-offline-database/releases/download/{release}/{file}"
SOURCE = "https://github.com/manami-project/anime-offline-database"
LICENSE = "ODbL 1.0 + DbCL 1.0"


def raw_path(release: str = RELEASE) -> Path:
    return paths.RAW / "anime-offline-database" / release / FILE_NAME


def processed_dir(release: str = RELEASE) -> Path:
    return paths.PROCESSED / f"catalog-{release}"


def download(release: str = RELEASE) -> Path:
    target = raw_path(release)
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    url = DOWNLOAD_URL.format(release=release, file=FILE_NAME)
    partial = target.with_suffix(".part")
    with httpx.stream("GET", url, follow_redirects=True, timeout=120) as response:
        response.raise_for_status()
        with partial.open("wb") as out:
            for chunk in response.iter_bytes():
                out.write(chunk)
    partial.replace(target)
    return target


def _ids(site_prefix: str) -> str:
    """SQL: the sorted integer IDs from `sources` whose URL starts with site_prefix."""
    return (
        f"list_sort(list_transform(list_filter(sources, s -> starts_with(s, '{site_prefix}')),"
        f" s -> CAST(regexp_extract(s, '(\\d+)$', 1) AS INTEGER)))"
    )


CATALOG_SQL = f"""
CREATE OR REPLACE TABLE catalog AS
WITH raw AS (
    -- catalog_id is the show's position in the release file: stable within a release only.
    -- Use anilist_id and mal_id to join across releases.
    SELECT generate_subscripts(data, 1) AS catalog_id, unnest(data, recursive := true)
    FROM read_json(?, maximum_object_size = 500000000)
),
source_index AS (SELECT catalog_id, unnest(sources) AS url FROM raw),
related AS (
    SELECT r.catalog_id, list_sort(list(DISTINCT s.catalog_id)) AS related_ids
    FROM (SELECT catalog_id, unnest(relatedAnime) AS url FROM raw) r
    JOIN source_index s USING (url)
    WHERE s.catalog_id <> r.catalog_id
    GROUP BY r.catalog_id
)
SELECT
    raw.catalog_id,
    title,
    type,
    episodes,
    status,
    season,
    year,
    CASE WHEN unit = 'SECONDS' THEN value END AS duration_seconds,
    arithmeticMean AS score_mean,
    {_ids("https://anilist.co/anime/")} AS anilist_ids,
    {_ids("https://myanimelist.net/anime/")} AS mal_ids,
    studios,
    producers,
    tags,
    synonyms,
    coalesce(related.related_ids, []) AS related_ids
FROM raw LEFT JOIN related USING (catalog_id)
ORDER BY raw.catalog_id
"""

MAPPING_SQL = """
CREATE OR REPLACE TABLE mapping AS
SELECT DISTINCT catalog_id, a.anilist_id, m.mal_id
FROM catalog,
     unnest(anilist_ids) AS a(anilist_id),
     unnest(mal_ids) AS m(mal_id)
ORDER BY anilist_id, mal_id
"""


@dataclass(frozen=True)
class CatalogSummary:
    release: str
    shows: int
    with_anilist: int
    with_mal: int
    with_both: int
    several_anilist_ids: int
    several_mal_ids: int

    def lines(self) -> list[str]:
        return [
            f"Catalog release {self.release}: {self.shows:,} shows",
            f"  with an AniList ID: {self.with_anilist:,}",
            f"  with a MAL ID: {self.with_mal:,}",
            f"  with both: {self.with_both:,} "
            f"({self.with_both / self.with_anilist:.1%} of AniList shows map to MAL; "
            f"{self.with_both / self.with_mal:.1%} of MAL shows map to AniList)",
            f"  merged entries (several IDs on one site): AniList {self.several_anilist_ids}, "
            f"MAL {self.several_mal_ids}",
        ]


def build(con: duckdb.DuckDBPyConnection, release: str = RELEASE) -> CatalogSummary:
    """Load the release into `catalog` and `mapping` tables and write them to Parquet."""
    con.execute(CATALOG_SQL, [str(raw_path(release))])
    con.execute(MAPPING_SQL)
    out = processed_dir(release)
    out.mkdir(parents=True, exist_ok=True)
    for table in ("catalog", "mapping"):
        con.execute(f"COPY {table} TO '{(out / f'{table}.parquet').as_posix()}' (FORMAT parquet)")
    return summarize(con, release)


def summarize(con: duckdb.DuckDBPyConnection, release: str = RELEASE) -> CatalogSummary:
    row = con.execute(
        """
        SELECT count(*),
               count(*) FILTER (len(anilist_ids) > 0),
               count(*) FILTER (len(mal_ids) > 0),
               count(*) FILTER (len(anilist_ids) > 0 AND len(mal_ids) > 0),
               count(*) FILTER (len(anilist_ids) > 1),
               count(*) FILTER (len(mal_ids) > 1)
        FROM catalog
        """
    ).fetchone()
    assert row is not None
    return CatalogSummary(release, *row)


def load(con: duckdb.DuckDBPyConnection, release: str = RELEASE) -> None:
    """Create `catalog` and `mapping` views over the processed Parquet files."""
    out = processed_dir(release)
    for table in ("catalog", "mapping"):
        con.execute(
            f"CREATE OR REPLACE VIEW {table} AS SELECT * FROM '{(out / f'{table}.parquet').as_posix()}'"
        )

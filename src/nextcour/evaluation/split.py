"""The temporal split (roadmap Phase 2): MAL 2018 users' lists, divided at a cutoff date.

Each user's entries from before the cutoff are their history, the input. Their labeled entries
after it are the targets. Decided 2026-10-24 (design note, Evaluation):
- **Cutoff: 2017-07-01.** About 10 months of targets before the data ends (2018-05-22), and years
  of history before it. Most eligible users stayed active to the end (median last update
  2018-05-06), so the window is real for nearly all of them.
- **When an entry happened:** its finish date, if the user recorded one, or else its last update.
  A finish date after the last update can't be right, so the earlier of the two counts. 84% of
  entries have no dates, so most are placed by their last update, which for dated entries came
  within a week of the finish 92% of the time. Entries with neither can't be placed and are left
  out.
- **Eligible users:** at least 10 labeled entries before the cutoff and at least 5 after
  ("labeled" as in enjoyment.py: scored, completed or dropped).
- **Users:** dev and test are each 10% of the eligible users, picked by a hash of the user name
  and a fixed seed. Train is every other MAL 2018 user, but only their entries from before the
  cutoff are kept, so no model can learn how shows after the cutoff were received.
- **Dev and test users feed nothing:** no model or crowd number is built from them. Their history
  goes in only as input at prediction time, the way mine does in the app. Test is for reported
  results only.
- **Shows:** only MAL shows with an AniList ID (96.4% of entries) are kept, so the profile
  builder, the franchises and the labels work as they do for AniList lists. The mapping is made
  one-to-one (4 MAL IDs link to two AniList IDs, and 4 the other way round).
- MAL's 1-10 scores are multiplied by 10, so they sit on AniList's 0-100 scale; relative scores
  make the scale irrelevant anyway. MAL 2018 has no rewatch count, only a "re-watching" flag,
  which counts as one rewatch.

Outputs, under data/processed/mal-2018-split-<cutoff>/: train, dev and test (list entries),
users, shows, and split.json (the settings and counts).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb

from nextcour.core import guardrail, manifest, paths
from nextcour.core.entries import ListEntry
from nextcour.ingest import catalog, franchises, mal

DATASET = "mal-2018"
CUTOFF = dt.date(2017, 7, 1)
DATA_END = dt.date(2018, 5, 22)  # MAL 2018's newest last-updated time (measured 2026-10-21)
SEED = 20261024
DEV_SHARE = TEST_SHARE = 0.10
MIN_BEFORE, MIN_AFTER = 10, 5
PARTS = ("train", "dev", "test")

MAL_STATUS = {1: "CURRENT", 2: "COMPLETED", 3: "PAUSED", 4: "DROPPED", 6: "PLANNING"}
MAL_FORMAT = {
    "TV": "TV",
    "Movie": "MOVIE",
    "OVA": "OVA",
    "ONA": "ONA",
    "Special": "SPECIAL",
    "Music": "MUSIC",
}
# Formats that can start a franchise's main line. Specials and music videos never do.
MAIN_FORMATS = ("TV", "MOVIE", "ONA", "OVA")


def split_dir(cutoff: dt.date = CUTOFF) -> Path:
    return paths.PROCESSED / f"{DATASET}-split-{cutoff.isoformat()}"


def user_draw(user_name: str, seed: int = SEED) -> float:
    """A number in [0, 1) fixed by the user name and seed, the same on every machine and version."""
    digest = hashlib.md5(f"{seed}:{user_name}".encode()).hexdigest()
    return int(digest[:12], 16) / 16**12


def _mal_date(column: str, *, latest: bool) -> str:
    """SQL: a MAL date string as a date. '2008-00-00' means some day in 2008: a start is read as the
    earliest day it could be and a finish as the latest, so an entry never looks earlier than it was."""
    y, m, d = f"substr({column}, 1, 4)", f"substr({column}, 6, 2)", f"substr({column}, 9, 2)"
    if latest:
        value = (
            f"CASE WHEN {m} = '00' THEN try_cast({y} || '-12-31' AS DATE)"
            f" WHEN {d} = '00' THEN last_day(try_cast({y} || '-' || {m} || '-01' AS DATE))"
            f" ELSE try_cast({column} AS DATE) END"
        )
    else:
        value = (
            f"CASE WHEN {m} = '00' THEN try_cast({y} || '-01-01' AS DATE)"
            f" WHEN {d} = '00' THEN try_cast({y} || '-' || {m} || '-01' AS DATE)"
            f" ELSE try_cast({column} AS DATE) END"
        )
    return f"CASE WHEN {y} BETWEEN '1960' AND '2018' THEN {value} END"


def _shows_sql(anime_list: str) -> str:
    formats = " ".join(f"WHEN '{k}' THEN '{v}'" for k, v in MAL_FORMAT.items())
    main = ", ".join(f"'{f}'" for f in MAIN_FORMATS)
    return f"""
    CREATE OR REPLACE TABLE shows AS
    WITH pairs AS (
        SELECT DISTINCT mal_id, anilist_id FROM mapping WHERE mal_id IS NOT NULL AND anilist_id IS NOT NULL
    ),
    by_mal AS (SELECT mal_id, min(anilist_id) AS anilist_id FROM pairs GROUP BY mal_id),
    one_to_one AS (SELECT min(mal_id) AS mal_id, anilist_id FROM by_mal GROUP BY anilist_id),
    franchise AS (
        SELECT a AS anilist_id, min(franchise_id) AS franchise_id
        FROM franchises JOIN catalog USING (catalog_id), unnest(anilist_ids) AS t(a)
        GROUP BY a
    ),
    mal_shows AS (
        SELECT
            anime_id AS mal_id,
            coalesce(nullif(title_english, ''), title) AS title,
            CASE type {formats} END AS format,
            try_cast(regexp_extract(aired, '''from'': ''([0-9-]+)''', 1) AS DATE) AS aired,
            nullif(episodes, 0) AS episodes,
            nullif(coalesce(try_cast(regexp_extract(duration, '(\\d+) hr', 1) AS INTEGER), 0) * 60
                   + coalesce(try_cast(regexp_extract(duration, '(\\d+) min', 1) AS INTEGER), 0), 0)
                AS duration,
            list_filter(string_split(coalesce(studio, ''), ', '), s -> s NOT IN ('', 'None found'))
                AS studios,
            list_filter(string_split(coalesce(genre, ''), ', '), g -> g <> '') AS genres,
            nullif(source, 'Unknown') AS source
        FROM '{anime_list}'
    ),
    joined AS (
        SELECT o.anilist_id, m.*, f.franchise_id
        FROM mal_shows m JOIN one_to_one o USING (mal_id) LEFT JOIN franchise f USING (anilist_id)
    ),
    -- A franchise's entry point: its earliest main-line entry, or its earliest entry of any kind
    -- if it has no main-line one. Shows outside any franchise are their own entry point.
    ranked AS (
        SELECT *, row_number() OVER (
            PARTITION BY coalesce(franchise_id, -anilist_id)
            ORDER BY (format IN ({main})) DESC NULLS LAST, aired NULLS LAST, mal_id
        ) AS position
        FROM joined
    )
    SELECT * EXCLUDE (position), position = 1 AS entry_point FROM ranked
    """


def draw_sql(column: str, seed: int = SEED) -> str:
    """SQL for user_draw(), which tests check it against."""
    return f"CAST('0x' || substr(md5('{seed}:' || {column}), 1, 12) AS UBIGINT) / 16^12"


def _entries_sql(user_lists: str, cutoff: dt.date) -> str:
    statuses = " ".join(f"WHEN {k} THEN '{v}'" for k, v in MAL_STATUS.items())
    started = _mal_date("my_start_date", latest=False)
    completed = _mal_date("my_finish_date", latest=True)
    return f"""
    CREATE OR REPLACE TABLE entries AS
    WITH raw AS (
        SELECT
            '{DATASET}' AS dataset,
            username AS user_name,
            s.anilist_id,
            anime_id AS mal_id,
            s.format,
            s.aired,
            CASE my_status {statuses} END AS status,
            my_score * 10 AS score,
            my_watched_episodes AS progress,
            coalesce(my_rewatching, 0) > 0 AS rewatching,
            {started} AS started_raw,
            {completed} AS completed_raw,
            CASE WHEN my_last_updated > 86400 THEN to_timestamp(my_last_updated) END AS updated
        FROM '{user_lists}' JOIN shows s ON s.mal_id = anime_id
        WHERE my_status IN ({", ".join(map(str, MAL_STATUS))})
    ),
    dated AS (
        -- A date before the show started airing can't be right.
        SELECT * EXCLUDE (started_raw, completed_raw),
            CASE WHEN aired IS NULL OR started_raw >= aired THEN started_raw END AS started,
            CASE WHEN aired IS NULL OR completed_raw >= aired THEN completed_raw END AS completed
        FROM raw
    ),
    placed AS (
        SELECT *,
            CASE
                WHEN completed IS NOT NULL AND updated IS NOT NULL
                    THEN least(completed, CAST(updated AS DATE))
                WHEN completed IS NOT NULL THEN completed
                ELSE CAST(updated AS DATE)
            END AS happened,
            status IN ('CURRENT', 'COMPLETED', 'PAUSED', 'DROPPED')
                AND (score > 0 OR status IN ('COMPLETED', 'DROPPED'))
                AND coalesce(format, '') <> 'MUSIC' AS labeled
        FROM dated
    )
    SELECT *, happened < DATE '{cutoff.isoformat()}' AS before FROM placed
    """


@dataclass(frozen=True)
class SplitSummary:
    cutoff: dt.date
    users: dict[str, int]
    eligible: int
    entries: dict[str, int]
    targets: dict[str, int]
    unplaced: int
    shows: int

    def lines(self) -> list[str]:
        return [
            f"Cutoff {self.cutoff}: {self.eligible:,} eligible users "
            f"({MIN_BEFORE}+ labeled entries before, {MIN_AFTER}+ after)",
            *(
                f"  {part}: {self.users[part]:,} users, {self.entries[part]:,} entries"
                + (f", {self.targets[part]:,} targets" if part != "train" else " (before the cutoff only)")
                for part in PARTS
            ),
            f"  shows with an AniList ID: {self.shows:,}",
            f"  entries left out because they have no date or update time: {self.unplaced:,}",
        ]


def build_training_split(
    con: duckdb.DuckDBPyConnection | None = None,
    cutoff: dt.date = CUTOFF,
    release: str = catalog.RELEASE,
    out: Path | None = None,
) -> SplitSummary:
    """Write the split's tables to Parquet, after the guardrail has checked them."""
    con = con or duckdb.connect()
    catalog.load(con, release)
    franchises.load(con, release)
    source = mal.parquet_dir(DATASET)
    con.execute(_shows_sql((source / "AnimeList.parquet").as_posix()))
    con.execute(_entries_sql((source / f"{mal.LIST_TABLES[DATASET].file}.parquet").as_posix(), cutoff))
    unplaced = con.execute("SELECT count(*) FROM entries WHERE happened IS NULL").fetchone()[0]
    con.execute("DELETE FROM entries WHERE happened IS NULL")

    con.execute(
        f"""
        CREATE OR REPLACE TABLE users AS
        WITH counted AS (
            SELECT user_name,
                   count(*) FILTER (labeled AND before) AS labeled_before,
                   count(*) FILTER (labeled AND NOT before) AS labeled_after,
                   {draw_sql("user_name")} AS draw
            FROM entries GROUP BY 1
        )
        SELECT user_name,
               CASE WHEN NOT eligible THEN 'train' WHEN draw < {DEV_SHARE} THEN 'dev'
                    WHEN draw < {DEV_SHARE + TEST_SHARE} THEN 'test' ELSE 'train' END AS part,
               eligible, labeled_before, labeled_after, draw
        FROM (SELECT *, labeled_before >= {MIN_BEFORE} AND labeled_after >= {MIN_AFTER} AS eligible
              FROM counted)
        """
    )

    con.execute(
        "CREATE OR REPLACE TABLE train AS SELECT e.* FROM entries e JOIN users USING (user_name)"
        " WHERE part = 'train' AND before"
    )
    for part in ("dev", "test"):
        con.execute(
            f"CREATE OR REPLACE TABLE {part} AS SELECT e.* FROM entries e JOIN users USING (user_name)"
            f" WHERE part = '{part}'"
        )
    check_split(con, cutoff)

    out = out or split_dir(cutoff)
    out.mkdir(parents=True, exist_ok=True)
    order = {"users": "user_name", "shows": "anilist_id"}
    for table in ("train", "dev", "test", "users", "shows"):
        path = (out / f"{table}.parquet").as_posix()
        by = order.get(table, "user_name, happened, anilist_id")
        con.execute(
            f"COPY (SELECT * FROM {table} ORDER BY {by}) TO '{path}' (FORMAT parquet, COMPRESSION zstd)"
        )

    summary = SplitSummary(
        cutoff=cutoff,
        users=dict(con.execute("SELECT part, count(*) FROM users GROUP BY 1").fetchall()),
        eligible=con.execute("SELECT count(*) FROM users WHERE eligible").fetchone()[0],
        entries={p: con.execute(f"SELECT count(*) FROM {p}").fetchone()[0] for p in PARTS},
        targets={
            p: con.execute(f"SELECT count(*) FROM {p} WHERE labeled AND NOT before").fetchone()[0]
            for p in ("dev", "test")
        }
        | {"train": 0},
        unplaced=unplaced,
        shows=con.execute("SELECT count(*) FROM shows").fetchone()[0],
    )
    record: dict[str, Any] = {
        "dataset": DATASET,
        "dataset_files": [
            {k: f[k] for k in ("file", "sha256")}
            for f in manifest.load()["datasets"].get(DATASET, {}).get("files", [])
        ],
        "catalog_release": release,
        "cutoff": cutoff.isoformat(),
        "seed": SEED,
        "dev_share": DEV_SHARE,
        "test_share": TEST_SHARE,
        "min_before": MIN_BEFORE,
        "min_after": MIN_AFTER,
        "built_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "users": summary.users,
        "eligible": summary.eligible,
        "entries": summary.entries,
        "targets": summary.targets,
        "unplaced": unplaced,
        "shows": summary.shows,
    }
    (out / "split.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    return summary


def check_split(con: duckdb.DuckDBPyConnection, cutoff: dt.date) -> None:
    """Fail if the split could leak: training rows from after the cutoff, a user in two parts, or
    rows from my own accounts anywhere."""
    for part in PARTS:
        guardrail.check_training_table(con, part)
    late = con.execute(
        f"SELECT count(*) FROM train WHERE NOT before OR happened >= DATE '{cutoff}'"
    ).fetchone()
    if late and late[0]:
        raise guardrail.GuardrailViolation(f"train has {late[0]} entries from after the cutoff")
    shared = con.execute(
        "SELECT count(*) FROM (SELECT user_name FROM ("
        " SELECT DISTINCT 'train' AS p, user_name FROM train"
        " UNION ALL SELECT DISTINCT 'dev', user_name FROM dev"
        " UNION ALL SELECT DISTINCT 'test', user_name FROM test"
        ") GROUP BY 1 HAVING count(*) > 1)"
    ).fetchone()
    if shared and shared[0]:
        raise guardrail.GuardrailViolation(f"{shared[0]} users are in more than one part of the split")


# --- Reading the split back ------------------------------------------------------------------


@dataclass(frozen=True)
class SplitUser:
    name: str
    history: tuple[ListEntry, ...]  # every entry from before the cutoff, whatever its status
    after: tuple[ListEntry, ...]  # every entry from after it; the labeled ones are the targets

    @property
    def entries(self) -> tuple[ListEntry, ...]:
        return self.history + self.after


def _entry(row: tuple[Any, ...]) -> ListEntry:
    (
        anilist_id,
        mal_id,
        title,
        format_,
        aired,
        status,
        score,
        progress,
        rewatching,
        started,
        completed,
        episodes,
        duration,
    ) = row
    return ListEntry(
        anilist_id=anilist_id,
        mal_id=mal_id,
        title=title,
        format=format_,
        aired=aired,
        status=status,
        score=float(score),
        progress=progress or 0,
        repeat=1 if rewatching else 0,
        started=started,
        completed=completed,
        created=None,
        episodes=episodes,
        duration=duration,
    )


def load_users(
    part: str,
    *,
    limit: int | None = None,
    min_draw: float = 0.0,
    cutoff: dt.date = CUTOFF,
    folder: Path | None = None,
    con: duckdb.DuckDBPyConnection | None = None,
    batch: int = 1000,
) -> Iterator[SplitUser]:
    """Users of one part, each with their history and later entries.

    `limit` takes the first users in a fixed random order (by their draw), so a sample is the same
    every run; `min_draw` skips users before that point in the order.
    """
    if part not in PARTS:
        raise ValueError(f"part must be one of {PARTS}")
    folder = folder or split_dir(cutoff)
    con = con or duckdb.connect()
    users = f"'{(folder / 'users.parquet').as_posix()}'"
    picked = con.execute(
        f"SELECT user_name FROM {users} WHERE part = ? AND draw >= ? AND (eligible OR ? = 'train')"
        " ORDER BY draw, user_name LIMIT ?",
        [part, min_draw, part, limit if limit is not None else 2**62],
    ).fetchall()
    names = [n for (n,) in picked]
    for start in range(0, len(names), batch):  # in batches, so memory stays small
        chunk = names[start : start + batch]
        rows = con.execute(
            f"""
            SELECT e.user_name, e.before, e.anilist_id, e.mal_id, s.title, e.format, e.aired, e.status,
                   e.score, e.progress, e.rewatching, e.started, e.completed, s.episodes, s.duration
            FROM '{(folder / f"{part}.parquet").as_posix()}' e
            JOIN '{(folder / "shows.parquet").as_posix()}' s USING (anilist_id)
            WHERE e.user_name IN (SELECT unnest(?))
            ORDER BY e.user_name, e.happened, e.anilist_id
            """,
            [chunk],
        ).fetchall()
        by_user: dict[str, tuple[list[ListEntry], list[ListEntry]]] = {n: ([], []) for n in chunk}
        for name, before, *rest in rows:
            by_user[name][0 if before else 1].append(_entry(tuple(rest)))
        for name in chunk:
            history, after = by_user[name]
            yield SplitUser(name, tuple(history), tuple(after))


def franchise_lookup(cutoff: dt.date = CUTOFF, folder: Path | None = None) -> dict[int, int]:
    """AniList ID -> franchise ID, for the split's shows."""
    folder = folder or split_dir(cutoff)
    rows = duckdb.sql(
        f"SELECT anilist_id, franchise_id FROM '{(folder / 'shows.parquet').as_posix()}'"
        " WHERE franchise_id IS NOT NULL"
    ).fetchall()
    return dict(rows)

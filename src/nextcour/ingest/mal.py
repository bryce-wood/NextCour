"""The three MAL datasets from Kaggle, converted once from CSV to Parquet.

Download each dataset from its Kaggle page and unzip it into data/raw/<name>/. Conversion keeps
the data exactly as published (cleaning happens in queries later), so a type DuckDB can't read
for every row, such as MAL's "0000-00-00" dates, stays text. DuckDB reads the whole file before
choosing column types, so a rare odd value can't break the conversion halfway through.

Some files aren't strict enough CSV for DuckDB: MAL 2018's list notes contain a few stray
carriage returns before a closing quote. Those files are read leniently, and the result is only
kept if its row count matches an independent count by Python's csv module.
"""

from __future__ import annotations

import csv
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb

from nextcour.core import manifest, paths

DATASETS: dict[str, dict[str, str]] = {
    "mal-2018": {
        "title": "MyAnimeList Dataset (2018), by azathoth42",
        "source": "https://www.kaggle.com/datasets/azathoth42/myanimelist",
        "license": "ODbL",
    },
    "mal-2020": {
        "title": "Anime Recommendation Database 2020, by hernan4444",
        "source": "https://www.kaggle.com/datasets/hernan4444/anime-recommendation-database-2020",
        "license": "CC0",
    },
    "mal-2023": {
        "title": "Anime Dataset 2023, by dbdmobile",
        "source": "https://www.kaggle.com/datasets/dbdmobile/myanimelist-dataset",
        "license": "ODbL",
        "note": "Only users-score-2023.csv is new. user-filtered.csv repeats mal-2020's animelist.csv "
        "exactly, and final_animedataset.csv repeats mal-2018's list entries (checked 2026-10-11).",
    },
}


class RowCountMismatch(RuntimeError):
    """A lenient read disagreed with Python's csv module about how many rows a file has."""


def raw_dir(name: str) -> Path:
    return paths.RAW / name


def parquet_dir(name: str) -> Path:
    return paths.PARQUET / name


def csv_row_count(path: Path) -> int:
    """Data rows in a CSV, by Python's csv module (quoted line breaks don't start new rows)."""
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        return sum(1 for _ in csv.reader(f)) - 1  # minus the header


def _copy(con: duckdb.DuckDBPyConnection, csv_path: Path, out: Path, strict: bool) -> None:
    options = "header = true, sample_size = -1" + ("" if strict else ", strict_mode = false")
    con.execute(
        f"COPY (SELECT * FROM read_csv('{csv_path.as_posix()}', {options}))"
        f" TO '{out.as_posix()}' (FORMAT parquet, COMPRESSION zstd)"
    )


def convert(
    name: str, con: duckdb.DuckDBPyConnection | None = None, log: Callable[[str], None] = print
) -> dict[str, Any]:
    """Convert every CSV in data/raw/<name>/ to Parquet, record it in the manifest, return the record."""
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset {name!r}; expected one of {sorted(DATASETS)}")
    source = raw_dir(name)
    csvs = sorted(source.rglob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV files in {source}. Download and unzip the dataset there first.")
    con = con or duckdb.connect()
    previous = {f["file"]: f for f in manifest.load()["datasets"].get(name, {}).get("files", [])}
    files: list[dict[str, Any]] = []
    for csv_path in csvs:
        relative = csv_path.relative_to(source).as_posix()
        out = parquet_dir(name) / Path(relative).with_suffix(".parquet")
        size = csv_path.stat().st_size
        log(f"{name}/{relative}: {size / 1e9:.2f} GB, checksumming...")
        sha256 = manifest.file_sha256(csv_path)
        old = previous.get(relative)
        if old and old["sha256"] == sha256 and out.exists():
            log("  unchanged, already converted")
            files.append(old)
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        start = time.time()
        parsing = "strict"
        try:
            _copy(con, csv_path, out, strict=True)
        except duckdb.InvalidInputException:
            log("  not strict CSV for DuckDB; reading leniently and checking the row count")
            _copy(con, csv_path, out, strict=False)
            parsing = "lenient, row count checked with Python's csv module"
        rows = con.execute(f"SELECT count(*) FROM '{out.as_posix()}'").fetchone()[0]
        if parsing != "strict" and rows != (expected := csv_row_count(csv_path)):
            out.unlink()
            raise RowCountMismatch(f"{relative}: DuckDB read {rows:,} rows, Python's csv module {expected:,}")
        columns = [
            {"name": c[0], "type": c[1]}
            for c in con.execute(f"DESCRIBE SELECT * FROM '{out.as_posix()}'").fetchall()
        ]
        log(f"  {rows:,} rows in {time.time() - start:.0f} s ({parsing})")
        files.append(
            {
                "file": relative,
                "bytes": size,
                "sha256": sha256,
                "rows": rows,
                "parsing": parsing,
                "parquet_bytes": out.stat().st_size,
                "columns": columns,
            }
        )
        # Save after every file, so a failure later keeps the work done so far.
        manifest.update(name, {**DATASETS[name], "files": [*files, *_rest(previous, files, csvs, source)]})
    record = {**DATASETS[name], "files": files}
    manifest.update(name, record)
    return record


def _rest(
    previous: dict[str, dict[str, Any]], done: list[dict[str, Any]], csvs: list[Path], source: Path
) -> list[dict[str, Any]]:
    """Earlier manifest records for files not converted yet in this run."""
    finished = {f["file"] for f in done}
    pending = [p.relative_to(source).as_posix() for p in csvs]
    return [previous[f] for f in pending if f not in finished and f in previous]


def available() -> list[str]:
    """Datasets whose raw files have been downloaded."""
    return [name for name in DATASETS if any(raw_dir(name).rglob("*.csv"))]


# Each dataset's table of list entries.
@dataclass(frozen=True)
class ListTable:
    file: str  # Parquet file stem
    user: str  # column identifying the user
    show: str  # column with the MAL anime ID
    user_name: str | None  # column with the MAL user name, if the dataset has one


LIST_TABLES = {
    "mal-2018": ListTable("UserAnimeList", "username", "anime_id", "username"),
    "mal-2020": ListTable("animelist", "user_id", "anime_id", None),  # user names were anonymized
    "mal-2023": ListTable("users-score-2023", "user_id", "anime_id", "Username"),  # the only new part
}


@dataclass(frozen=True)
class ListReport:
    name: str
    users: int
    entries: int
    shows: int
    shows_mapped: int  # shows with an AniList ID in the catalog
    entries_mapped: int
    protected_entries: int | None  # entries under one of my user names; None if names were anonymized

    def lines(self) -> list[str]:
        protected = (
            "can't be checked (user names anonymized)"
            if self.protected_entries is None
            else f"{self.protected_entries} entries"
        )
        return [
            f"{self.name}: {self.users:,} users, {self.entries:,} list entries, {self.shows:,} shows",
            f"  shows that map to AniList: {self.shows_mapped:,} ({self.shows_mapped / self.shows:.1%})",
            f"  list entries whose show maps: {self.entries_mapped:,} "
            f"({self.entries_mapped / self.entries:.1%})",
            f"  under my user names: {protected}",
        ]


def list_report(con: duckdb.DuckDBPyConnection, name: str, my_names: list[str]) -> ListReport:
    """Size, AniList mapping and a check for my own accounts. Needs the `mapping` table or view."""
    table = LIST_TABLES[name]
    path = (parquet_dir(name) / f"{table.file}.parquet").as_posix()
    row = con.execute(
        f"""
        WITH entries AS (SELECT "{table.user}" AS u, "{table.show}" AS s FROM '{path}'),
             mapped AS (SELECT DISTINCT mal_id FROM mapping)
        SELECT count(DISTINCT u), count(*), count(DISTINCT s),
               count(DISTINCT s) FILTER (s IN (SELECT mal_id FROM mapped)),
               count(*) FILTER (s IN (SELECT mal_id FROM mapped))
        FROM entries
        """
    ).fetchone()
    assert row is not None
    protected = None
    if table.user_name:
        name_column = f'lower(CAST("{table.user_name}" AS VARCHAR))'
        protected = con.execute(
            f"SELECT count(*) FROM '{path}' WHERE {name_column} IN (SELECT unnest(?))",
            [[n.lower() for n in my_names]],
        ).fetchone()[0]
    return ListReport(name, *row, protected)

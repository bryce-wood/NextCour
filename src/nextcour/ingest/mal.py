"""The three MAL datasets from Kaggle, converted once from CSV to Parquet.

Download each dataset from its Kaggle page and unzip it into data/raw/<name>/. Conversion keeps
the data exactly as published (cleaning happens in queries later), so a type DuckDB can't read
for every row, such as MAL's "0000-00-00" dates, stays text. DuckDB reads the whole file before
choosing column types, so a rare odd value can't break the conversion halfway through.
"""

from __future__ import annotations

import time
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
    },
}


def raw_dir(name: str) -> Path:
    return paths.RAW / name


def parquet_dir(name: str) -> Path:
    return paths.PARQUET / name


def convert(name: str, con: duckdb.DuckDBPyConnection | None = None, log=print) -> dict[str, Any]:
    """Convert every CSV in data/raw/<name>/ to Parquet, record it in the manifest, return the record."""
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset {name!r}; expected one of {sorted(DATASETS)}")
    source = raw_dir(name)
    csvs = sorted(source.rglob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV files in {source}. Download and unzip the dataset there first.")
    con = con or duckdb.connect()
    previous = {f["file"]: f for f in manifest.load()["datasets"].get(name, {}).get("files", [])}
    files = []
    for csv in csvs:
        relative = csv.relative_to(source).as_posix()
        out = parquet_dir(name) / Path(relative).with_suffix(".parquet")
        size = csv.stat().st_size
        log(f"{name}/{relative}: {size / 1e9:.2f} GB, checksumming...")
        sha256 = manifest.file_sha256(csv)
        old = previous.get(relative)
        if old and old["sha256"] == sha256 and out.exists():
            log("  unchanged, already converted")
            files.append(old)
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        start = time.time()
        con.execute(
            f"COPY (SELECT * FROM read_csv('{csv.as_posix()}', header = true, sample_size = -1))"
            f" TO '{out.as_posix()}' (FORMAT parquet, COMPRESSION zstd)"
        )
        rows = con.execute(f"SELECT count(*) FROM '{out.as_posix()}'").fetchone()[0]
        columns = [
            {"name": c[0], "type": c[1]}
            for c in con.execute(f"DESCRIBE SELECT * FROM '{out.as_posix()}'").fetchall()
        ]
        log(f"  {rows:,} rows in {time.time() - start:.0f} s")
        files.append(
            {
                "file": relative,
                "bytes": size,
                "sha256": sha256,
                "rows": rows,
                "parquet_bytes": out.stat().st_size,
                "columns": columns,
            }
        )
    record = {**DATASETS[name], "files": files}
    manifest.update(name, record)
    return record


def available() -> list[str]:
    """Datasets whose raw files have been downloaded."""
    return [name for name in DATASETS if any(raw_dir(name).rglob("*.csv"))]

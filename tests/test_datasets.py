"""Catalog building, MAL conversion and date coverage, on tiny synthetic files."""

import json

import duckdb
import pytest

from nextcour.core import manifest, paths
from nextcour.ingest import catalog, coverage, mal


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "RAW", tmp_path / "raw")
    monkeypatch.setattr(paths, "PARQUET", tmp_path / "parquet")
    monkeypatch.setattr(paths, "PROCESSED", tmp_path / "processed")
    monkeypatch.setattr(paths, "DATASET_MANIFEST", tmp_path / "manifests" / "datasets.json")
    return tmp_path


def show(title, sources, related=(), unit="SECONDS"):
    return {
        "sources": sources,
        "title": title,
        "type": "TV",
        "episodes": 12,
        "status": "FINISHED",
        "animeSeason": {"season": "SPRING", "year": 2020},
        "duration": {"value": 1440, "unit": unit},
        "score": {"arithmeticGeometricMean": 7.5, "arithmeticMean": 7.6, "median": 7.5},
        "synonyms": [],
        "studios": ["madhouse"],
        "producers": [],
        "relatedAnime": list(related),
        "tags": ["action"],
    }


def test_catalog_maps_anilist_to_mal(data_dir):
    release = "test"
    raw = catalog.raw_path(release)
    raw.parent.mkdir(parents=True)
    shows = [
        show(
            "A",
            ["https://anilist.co/anime/10", "https://myanimelist.net/anime/20"],
            ["https://kitsu.app/anime/5"],
        ),
        show("B", ["https://kitsu.app/anime/5", "https://myanimelist.net/anime/21"]),
        show("C", ["https://anilist.co/anime/11", "https://anilist.co/anime/12"]),
    ]
    raw.write_text(json.dumps({"lastUpdate": "2026-01-01", "data": shows}), encoding="utf-8")

    con = duckdb.connect()
    summary = catalog.build(con, release)
    assert (summary.shows, summary.with_anilist, summary.with_mal, summary.with_both) == (3, 2, 2, 1)
    assert summary.several_anilist_ids == 1
    assert con.execute("SELECT anilist_id, mal_id FROM mapping").fetchall() == [(10, 20)]
    assert con.execute("SELECT related_ids FROM catalog WHERE title = 'A'").fetchone() == ([2],)
    assert (catalog.processed_dir(release) / "catalog.parquet").exists()


MAL_CSV = """username,anime_id,my_watched_episodes,my_start_date,my_finish_date,my_score,my_status
alice,1,12,2015-01-01,2015-01-10,8,2
alice,2,12,0000-00-00,0000-00-00,7,2
alice,3,3,2015-02-00,0000-00-00,0,4
bob,1,12,0000-00-00,2016-05-05,9,2
"""


def test_mal_conversion_and_date_coverage(data_dir):
    folder = mal.raw_dir("mal-2018")
    folder.mkdir(parents=True)
    (folder / "UserAnimeList.csv").write_text(MAL_CSV, encoding="utf-8")
    logs = []
    record = mal.convert("mal-2018", log=logs.append)

    [file] = record["files"]
    assert (file["file"], file["rows"]) == ("UserAnimeList.csv", 4)
    assert len(file["sha256"]) == 64
    assert manifest.load()["datasets"]["mal-2018"]["license"] == "ODbL"

    mal.convert("mal-2018", log=logs.append)  # unchanged files aren't converted again
    assert logs[-1] == "  unchanged, already converted"

    con = duckdb.connect()
    result = coverage.mal_list_entries(con, mal.parquet_dir("mal-2018") / "UserAnimeList.parquet", "test")
    assert (result.completed, result.with_start, result.with_finish, result.with_both) == (3, 1, 2, 1)
    assert (result.users, result.users_any, result.users_most) == (2, 1, 1)


def test_anilist_coverage_counts_only_full_dates():
    full = {"year": 2024, "month": 1, "day": 2}
    partial = {"year": 2024, "month": None, "day": None}
    entries = [
        {"status": "COMPLETED", "startedAt": full, "completedAt": full},
        {"status": "COMPLETED", "startedAt": partial, "completedAt": full},
        {"status": "PLANNING", "startedAt": full, "completedAt": full},
    ]
    result = coverage.anilist_list(entries, "test")
    assert (result.completed, result.with_start, result.with_finish, result.with_both) == (2, 1, 2, 1)


def test_unknown_dataset_is_refused(data_dir):
    with pytest.raises(ValueError, match="Unknown dataset"):
        mal.convert("mal-1999")


NOTES_CSV = (
    'username,anime_id,my_tags\nalice,1,"two\r\nlines"\nbob,2,"stray carriage return\r"\ncarol,3,plain\n'
)


def test_csv_row_count_ignores_line_breaks_inside_quotes(tmp_path):
    path = tmp_path / "notes.csv"
    path.write_bytes(NOTES_CSV.encode())
    assert mal.csv_row_count(path) == 3


def strict_fails(original):
    def copy(con, csv_path, out, strict):
        if strict:
            raise duckdb.InvalidInputException("CSV Parser state machine reached an invalid state")
        original(con, csv_path, out, strict)

    return copy


def test_lenient_fallback_keeps_the_file_when_row_counts_agree(data_dir, monkeypatch):
    folder = mal.raw_dir("mal-2018")
    folder.mkdir(parents=True)
    (folder / "UserAnimeList.csv").write_bytes(NOTES_CSV.encode())
    monkeypatch.setattr(mal, "_copy", strict_fails(mal._copy))
    [file] = mal.convert("mal-2018", log=lambda _: None)["files"]
    assert file["rows"] == 3
    assert file["parsing"].startswith("lenient")


def test_lenient_fallback_refuses_a_row_count_mismatch(data_dir, monkeypatch):
    folder = mal.raw_dir("mal-2018")
    folder.mkdir(parents=True)
    (folder / "UserAnimeList.csv").write_bytes(NOTES_CSV.encode())
    monkeypatch.setattr(mal, "_copy", strict_fails(mal._copy))
    monkeypatch.setattr(mal, "csv_row_count", lambda _: 4)
    with pytest.raises(mal.RowCountMismatch):
        mal.convert("mal-2018", log=lambda _: None)
    assert not (mal.parquet_dir("mal-2018") / "UserAnimeList.parquet").exists()


def test_list_report(data_dir):
    folder = mal.raw_dir("mal-2018")
    folder.mkdir(parents=True)
    (folder / "UserAnimeList.csv").write_text(MAL_CSV, encoding="utf-8")
    mal.convert("mal-2018", log=lambda _: None)
    con = duckdb.connect()
    con.execute("CREATE VIEW mapping AS SELECT 1 AS mal_id")
    report = mal.list_report(con, "mal-2018", ["Bob"])
    assert (report.users, report.entries, report.shows) == (2, 4, 3)
    assert (report.shows_mapped, report.entries_mapped) == (1, 2)
    assert report.protected_entries == 1

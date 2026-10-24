"""The temporal split, on tiny synthetic MAL files."""

import datetime as dt

import duckdb
import pytest
from test_guardrail import fake_frozen_set

from nextcour.core import guardrail, paths
from nextcour.core.guardrail import GuardrailViolation
from nextcour.evaluation import split

CUTOFF = dt.date(2017, 7, 1)
AFTER = int(dt.datetime(2017, 9, 1, tzinfo=dt.UTC).timestamp())
BEFORE = int(dt.datetime(2017, 3, 1, tzinfo=dt.UTC).timestamp())

# (mal_id, anilist_id, franchise_id, type, aired). Shows 1-3 are one franchise; 3 aired first but is
# a special, so 1 is the entry point. Show 9 has no AniList ID.
SHOWS = [
    (1, 101, 1, "TV", "2014-04-01"),
    (2, 102, 1, "TV", "2016-04-01"),
    (3, 103, 1, "Special", "2013-01-01"),
    (4, 104, None, "Movie", "2015-01-01"),
    (5, 105, None, "TV", "2017-10-01"),
    (6, 106, None, "Music", "2015-01-01"),
    (7, 107, None, "TV", "2015-01-01"),
    (9, None, None, "TV", "2015-01-01"),
] + [(10 + i, 110 + i, None, "TV", "2010-01-01") for i in range(20)]


def write_inputs(tmp_path, monkeypatch, lists):
    monkeypatch.setattr(paths, "PARQUET", tmp_path / "parquet")
    monkeypatch.setattr(paths, "PROCESSED", tmp_path / "processed")
    monkeypatch.setattr(paths, "DATASET_MANIFEST", tmp_path / "datasets.json")
    con = duckdb.connect()
    catalog_dir = tmp_path / "processed" / "catalog-test"
    mal_dir = tmp_path / "parquet" / "mal-2018"
    catalog_dir.mkdir(parents=True)
    mal_dir.mkdir(parents=True)
    con.execute("CREATE TABLE catalog (catalog_id INTEGER, anilist_ids INTEGER[])")
    con.execute("CREATE TABLE mapping (catalog_id INTEGER, anilist_id INTEGER, mal_id INTEGER)")
    con.execute("CREATE TABLE franchises (catalog_id INTEGER, franchise_id INTEGER, method VARCHAR)")
    con.execute(
        "CREATE TABLE anime (anime_id BIGINT, title VARCHAR, title_english VARCHAR, type VARCHAR,"
        " aired VARCHAR, episodes BIGINT, duration VARCHAR, studio VARCHAR, genre VARCHAR, source VARCHAR)"
    )
    for catalog_id, (mal_id, anilist_id, franchise_id, kind, aired) in enumerate(SHOWS, start=1):
        con.execute("INSERT INTO catalog VALUES (?, ?)", [catalog_id, [anilist_id] if anilist_id else []])
        con.execute("INSERT INTO mapping VALUES (?, ?, ?)", [catalog_id, anilist_id, mal_id])
        if franchise_id:
            con.execute("INSERT INTO franchises VALUES (?, ?, 'typed')", [catalog_id, franchise_id])
        con.execute(
            "INSERT INTO anime VALUES"
            " (?, ?, '', ?, ?, 12, '24 min. per ep.', 'Madhouse', 'Action, Drama', 'Manga')",
            [mal_id, f"Show {mal_id}", kind, f"{{'from': '{aired}', 'to': None}}"],
        )
    con.execute(
        "CREATE TABLE lists (username VARCHAR, anime_id BIGINT, my_watched_episodes BIGINT,"
        " my_start_date VARCHAR, my_finish_date VARCHAR, my_score BIGINT, my_status BIGINT,"
        " my_rewatching BIGINT, my_rewatching_ep BIGINT, my_last_updated BIGINT, my_tags VARCHAR)"
    )
    for row in lists:
        con.execute("INSERT INTO lists VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?, '')", row)
    for table in ("catalog", "mapping", "franchises"):
        con.execute(f"COPY {table} TO '{(catalog_dir / f'{table}.parquet').as_posix()}' (FORMAT parquet)")
    con.execute(f"COPY anime TO '{(mal_dir / 'AnimeList.parquet').as_posix()}' (FORMAT parquet)")
    con.execute(f"COPY lists TO '{(mal_dir / 'UserAnimeList.parquet').as_posix()}' (FORMAT parquet)")


def filler(user, count, updated=BEFORE):
    """`count` completed, scored entries on the filler shows."""
    return [(user, 10 + i, 12, "0000-00-00", "0000-00-00", 7, 2, 0, updated) for i in range(count)]


def row(
    user,
    mal_id,
    *,
    finished="0000-00-00",
    updated=AFTER,
    score=8,
    status=2,
    rewatching=0,
    started="0000-00-00",
):
    return (user, mal_id, 12, started, finished, score, status, rewatching, updated)


@pytest.fixture
def built(tmp_path, monkeypatch):
    monkeypatch.setattr(split, "DEV_SHARE", 1.0)  # every eligible user goes to dev
    monkeypatch.setattr(guardrail, "frozen_set_if_present", lambda: fake_frozen_set())
    lists = [
        *filler("eligible", 10),
        row("eligible", 1, finished="2017-06-20", updated=AFTER),  # finished before, updated after
        row("eligible", 2, updated=AFTER),  # no dates: placed by its update
        row("eligible", 4, finished="2017-12-01", updated=BEFORE),  # finish after update: the update wins
        row("eligible", 5, finished="2017-11-00", updated=AFTER, rewatching=1),  # a partial date
        row("eligible", 6, updated=AFTER),  # a music video: on the list, never a target
        row("eligible", 3, updated=AFTER, score=0, status=4),  # an unscored drop
        row("eligible", 7, updated=0),  # no finish date and no update time: left out
        row("eligible", 9, updated=AFTER),  # no AniList ID: left out
        row("eligible", 20, updated=AFTER),
        row("eligible", 21, updated=AFTER),  # with 2, 3 and 5, that's the 5 labeled entries needed after
        *filler("few_after", 10),
        row("few_after", 2, updated=AFTER),
        *filler("trainee", 3),
        row("trainee", 2, updated=AFTER),  # a train user's later entry: never stored
    ]
    write_inputs(tmp_path, monkeypatch, lists)
    summary = split.build_training_split(cutoff=CUTOFF, release="test")
    return summary, split.split_dir(CUTOFF)


def test_user_draw_matches_its_sql():
    names = ["abc", "Some_User", "名前", "x" * 40]
    con = duckdb.connect()
    for name in names:
        (value,) = con.execute(f"SELECT {split.draw_sql('?')}", [name]).fetchone()
        assert value == pytest.approx(split.user_draw(name), abs=1e-12)
    assert 0 <= split.user_draw("abc") < 1
    assert split.user_draw("abc") != split.user_draw("abd")


def test_split_places_entries_by_finish_date_or_update(built):
    summary, folder = built
    (user,) = split.load_users("dev", folder=folder)
    assert user.name == "eligible"
    before = {e.mal_id for e in user.history}
    after = {e.mal_id for e in user.after}
    assert 1 in before  # its finish date was before the cutoff
    assert 4 in before  # a finish date after the last update can't be right
    assert {2, 5, 3, 6} <= after
    assert 9 not in before | after  # no AniList ID
    assert summary.unplaced == 1  # show 7 has no date and no update time


def test_split_converts_mal_entries(built):
    _, folder = built
    (user,) = split.load_users("dev", folder=folder)
    by_id = {e.mal_id: e for e in user.entries}
    assert by_id[2].anilist_id == 102
    assert by_id[2].score == 80  # MAL's 8 of 10
    assert by_id[3].status == "DROPPED" and by_id[3].score == 0
    assert by_id[5].completed == dt.date(2017, 11, 30)  # "2017-11-00" is the latest day it could be
    assert by_id[5].repeat == 1  # the re-watching flag
    assert by_id[6].format == "MUSIC"
    assert by_id[2].episodes == 12 and by_id[2].duration == 24


def test_train_users_keep_only_entries_from_before_the_cutoff(built):
    summary, folder = built
    con = duckdb.connect()
    parts = dict(
        con.execute(f"SELECT user_name, part FROM '{(folder / 'users.parquet').as_posix()}'").fetchall()
    )
    assert parts == {"eligible": "dev", "few_after": "train", "trainee": "train"}
    train = con.execute(
        "SELECT user_name, count(*), bool_and(before)"
        f" FROM '{(folder / 'train.parquet').as_posix()}' GROUP BY 1"
    ).fetchall()
    assert sorted(train) == [("few_after", 10, True), ("trainee", 3, True)]
    assert summary.eligible == 1


def test_entry_point_is_the_first_main_line_entry(built):
    _, folder = built
    con = duckdb.connect()
    points = dict(
        con.execute(f"SELECT mal_id, entry_point FROM '{(folder / 'shows.parquet').as_posix()}'").fetchall()
    )
    assert points[1] and not points[2] and not points[3]  # the special aired first, but it's a special
    assert points[4] and points[5]  # shows outside any franchise


def test_my_account_in_the_mal_data_stops_the_split(tmp_path, monkeypatch):
    monkeypatch.setattr(guardrail, "frozen_set_if_present", lambda: fake_frozen_set())
    write_inputs(tmp_path, monkeypatch, filler("SomeAniListUser", 3))
    with pytest.raises(GuardrailViolation, match="my own accounts"):
        split.build_training_split(cutoff=CUTOFF, release="test")

"""The guardrail: my own AniList data never reaches training, and frozen shows stay frozen.

When a new training pipeline is built, it must call guardrail.check_training_table() on its
training data; test_every_training_pipeline_calls_the_guardrail enforces that.
"""

import dataclasses
import datetime as dt
import json
import re
from pathlib import Path

import duckdb
import pytest
from helpers import JOINED, d, entry

from nextcour.core import guardrail, paths
from nextcour.core.entries import from_anilist
from nextcour.core.guardrail import FrozenSet, FrozenShow, GuardrailViolation
from nextcour.ingest import catalog, franchises

# Pinned when the set was frozen. If this test fails, the frozen set changed: undo the change.
FROZEN_SET_SHA256 = "8f0f7f9f2fa2d68665cc65cd0ae2b0ea0b88981d07dcc0dec29f128e652e84b3"

SRC = Path(__file__).resolve().parents[1] / "src" / "nextcour"


def show(anilist_id=1, as_of="2023-01-01", aired="2020-01-01", franchise_id=10):
    return FrozenShow(anilist_id, None, f"Show {anilist_id}", franchise_id, d(aired), d(as_of), "main")


def fake_frozen_set(mal_user_names=()):
    return FrozenSet(
        frozen_on=dt.date(2026, 10, 11),
        anilist_user_id=1001,
        anilist_user_name="SomeAniListUser",
        joined=JOINED,
        mal_user_names=tuple(mal_user_names),
        catalog_release="2026-27",
        snapshot_file="frozen/x.json",
        snapshot_sha256="0",
        shows=(show(),),
    )


# --- My frozen set (private: these run only where it exists) --------------------------------


@pytest.fixture
def frozen_file():
    if not paths.FROZEN_TEST_SET.exists():
        pytest.skip("no frozen test set here (it's private)")
    return paths.FROZEN_TEST_SET


@pytest.mark.local_data
def test_frozen_set_is_unchanged(frozen_file):
    data = json.loads(frozen_file.read_text(encoding="utf-8"))
    assert guardrail.canonical_sha256(data) == FROZEN_SET_SHA256


@pytest.mark.local_data
def test_frozen_set_is_well_formed(frozen_file):
    frozen = guardrail.load_frozen_set(frozen_file)
    assert frozen.of("main") and frozen.of("sequel")
    assert len(frozen.anilist_ids) == len(frozen.shows), "a show appears twice"
    assert all(s.as_of <= frozen.frozen_on for s in frozen.shows)
    assert frozen.mal_user_names == ()  # I've never had a MAL account


@pytest.mark.local_data
def test_snapshot_and_every_frozen_history_pass_locally(frozen_file):
    """The labels' snapshot is intact, and each frozen show's history holds only what was known
    before it."""
    frozen = guardrail.load_frozen_set(frozen_file)
    snapshot_path = paths.DATA / frozen.snapshot_file
    if not snapshot_path.exists() or not catalog.processed_dir(frozen.catalog_release).exists():
        pytest.skip("no local snapshot or catalog")
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    assert guardrail.canonical_sha256(snapshot) == frozen.snapshot_sha256

    entries = [from_anilist(e) for e in snapshot["entries"]]
    con = duckdb.connect()
    catalog.load(con, frozen.catalog_release)
    franchise_of = franchises.lookup(con, frozen.catalog_release)
    for frozen_show in frozen.shows:
        history = guardrail.history_for(frozen_show, entries, franchise_of, frozen.joined)
        guardrail.check_history(frozen_show, history, franchise_of, frozen.joined)
        assert frozen_show.anilist_id not in {e.anilist_id for e in history}


# --- History for a frozen show ---------------------------------------------------------------


def test_known_by():
    assert guardrail.known_by(entry(1, completed="2022-03-01"), JOINED) == dt.date(2022, 3, 1)
    assert guardrail.known_by(entry(1, status="DROPPED", started="2022-03-27"), JOINED) == dt.date(
        2023, 3, 27
    )
    assert guardrail.known_by(entry(1, created=JOINED + dt.timedelta(hours=1)), JOINED) == dt.date(2021, 4, 7)
    assert guardrail.known_by(entry(1), JOINED) is None  # undated, added later: unknown
    assert guardrail.known_by(entry(1, status="CURRENT", started="2020-01-01"), JOINED) is None


def test_history_keeps_only_what_was_known_before():
    target = show(1, as_of="2023-01-01", aired="2020-01-01", franchise_id=10)
    entries = [
        entry(1, completed="2022-01-01"),  # the show itself
        entry(2, completed="2022-06-01"),  # known before: kept
        entry(3, completed="2023-01-01"),  # finished the day it started: kept
        entry(4, completed="2023-01-02"),  # finished after: dropped
        entry(5, status="CURRENT", started="2022-01-01"),  # in progress: dropped
        entry(6, aired="2021-01-01", completed="2022-02-01"),  # later season of the same franchise
        entry(7, aired="2019-01-01", completed="2022-02-01"),  # earlier season: kept
    ]
    franchise_of = {1: 10, 6: 10, 7: 10}
    history = guardrail.history_for(target, entries, franchise_of, JOINED)
    assert [e.anilist_id for e in history] == [2, 3, 7]
    guardrail.check_history(target, history, franchise_of, JOINED)


@pytest.mark.parametrize(
    ("bad", "message"),
    [
        (entry(1, completed="2020-01-01"), "own history"),
        (entry(2, completed="2024-01-01"), "wasn't known"),
        (entry(3, aired="2021-01-01", completed="2022-01-01"), "comes after"),
    ],
)
def test_check_history_catches_leaks(bad, message):
    with pytest.raises(GuardrailViolation, match=message):
        guardrail.check_history(show(1), [bad], {1: 10, 3: 10}, JOINED)


# --- Training data ---------------------------------------------------------------------------


def training_table(rows):
    con = duckdb.connect()
    con.execute("CREATE TABLE train (dataset VARCHAR, user_name VARCHAR, anime_id INTEGER)")
    con.executemany("INSERT INTO train VALUES (?, ?, ?)", rows)
    return con


def test_clean_mal_training_data_passes():
    con = training_table([("mal-2018", "someone", 1), ("mal-2023", "else", 2)])
    guardrail.check_training_table(con, "train", frozen=fake_frozen_set())


def test_anilist_rows_in_training_data_fail():
    con = training_table([("mal-2018", "someone", 1), ("anilist", "SomeAniListUser", 2)])
    with pytest.raises(GuardrailViolation, match="anilist"):
        guardrail.check_training_table(con, "train", frozen=fake_frozen_set())


def test_without_a_frozen_set_only_mal_rows_are_allowed(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "FROZEN_TEST_SET", tmp_path / "missing.json")
    guardrail.check_training_table(training_table([("mal-2020", "anyone", 1)]), "train")
    with pytest.raises(GuardrailViolation, match="anilist"):
        guardrail.check_training_table(training_table([("anilist", "anyone", 1)]), "train")


def test_my_accounts_in_training_data_fail():
    con = training_table([("mal-2023", "someanilistuser", 1)])
    with pytest.raises(GuardrailViolation, match="my own accounts"):
        guardrail.check_training_table(con, "train", frozen=fake_frozen_set())
    con = training_table([("mal-2020", "old_mal_name", 1)])
    with pytest.raises(GuardrailViolation, match="my own accounts"):
        guardrail.check_training_table(con, "train", frozen=fake_frozen_set(["Old_MAL_Name"]))


def test_every_training_pipeline_calls_the_guardrail():
    """Any module that builds training data or fits a model must check its training table."""
    pipeline = re.compile(r"^def (train\w*|fit\w*|build_training\w*)\(", re.MULTILINE)
    unguarded = [
        path.relative_to(SRC).as_posix()
        for path in SRC.rglob("*.py")
        if pipeline.search(text := path.read_text(encoding="utf-8")) and "check_training_table(" not in text
    ]
    assert unguarded == []


def test_frozen_show_is_immutable():
    with pytest.raises(dataclasses.FrozenInstanceError):
        show().as_of = dt.date(2030, 1, 1)  # type: ignore[misc]

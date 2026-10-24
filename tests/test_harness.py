"""The evaluation harness, end to end on a tiny synthetic split."""

import math

import numpy as np
import pytest
from helpers import entry
from test_guardrail import fake_frozen_set
from test_split import AFTER, BEFORE, CUTOFF, filler, row, write_inputs

from nextcour.core import guardrail
from nextcour.evaluation import harness, split


@pytest.fixture
def folder(tmp_path, monkeypatch):
    monkeypatch.setattr(split, "DEV_SHARE", 1.0)
    monkeypatch.setattr(guardrail, "frozen_set_if_present", lambda: fake_frozen_set())
    monkeypatch.setattr(harness, "MIN_SCORES", 1)
    lists = [
        # The dev user: history on the filler shows and show 1, then five shows after the cutoff.
        *filler("viewer", 10),
        row("viewer", 1, updated=BEFORE, score=9),  # starts franchise 1
        row("viewer", 2, updated=AFTER, score=9),  # a sequel in franchise 1
        row("viewer", 4, updated=AFTER, score=10),  # a new franchise, enjoyed
        row("viewer", 5, updated=AFTER, score=3),  # aired after the cutoff
        row("viewer", 20, updated=AFTER, score=5),
        row("viewer", 21, updated=AFTER, score=6),
        # Train users, whose history before the cutoff makes the crowd numbers.
        *[row(f"t{i}", 4, updated=BEFORE, score=9) for i in range(3)],
        *[row(f"t{i}", 20, updated=BEFORE, score=4) for i in range(5)],
        *[row(f"t{i}", 21, updated=BEFORE, score=6) for i in range(2)],
    ]
    write_inputs(tmp_path, monkeypatch, lists)
    split.build_training_split(cutoff=CUTOFF, release="test")
    return split.split_dir(CUTOFF)


def test_crowd_numbers_come_from_train_users_before_the_cutoff(folder):
    catalog = harness.load_catalog(folder, CUTOFF)
    at = catalog.index
    assert catalog.members[at[120]] == 5  # filler show 20 (AniList 120): five train users
    assert catalog.average[at[120]] == pytest.approx(40)
    assert catalog.average[at[104]] == pytest.approx(90)
    assert catalog.members[at[102]] == 0  # only the dev user watched it, after the cutoff
    assert math.isnan(catalog.average[at[102]])
    assert catalog.aired_after_cutoff[at[105]] and not catalog.aired_after_cutoff[at[104]]


def test_discovery_leaves_out_the_history_and_started_franchises(folder):
    catalog = harness.load_catalog(folder, CUTOFF)
    franchise_of = split.franchise_lookup(CUTOFF, folder)
    (raw,) = split.load_users("dev", folder=folder)
    user = harness.prepare(raw, catalog, franchise_of)
    base = np.flatnonzero(catalog.available & catalog.entry_point)
    candidates = set(catalog.ids[harness.discovery_candidates(user, catalog, base)].tolist())
    assert 101 not in candidates and 102 not in candidates  # franchise 1 is started
    assert 110 not in candidates  # a filler show in the history
    assert 104 in candidates and 105 in candidates
    assert user.sequel.tolist() == [
        catalog.franchise[p] == catalog.franchise[catalog.index[101]] for p in user.targets
    ]


def test_discovery_gains_mark_only_enjoyed_targets():
    user = harness.EvalUser(
        name="u",
        history=(),
        targets=np.array([3, 7, 9]),
        labels=np.array([0.9, 0.2, 0.6]),
        sequel=np.zeros(3, dtype=bool),
        seen=np.array([], dtype=np.int64),
        started=frozenset(),
        has_dates=False,
    )
    gains = harness.discovery_gains(user, np.array([1, 3, 7, 8, 9]))
    assert gains.tolist() == [0.0, 0.9, 0.0, 0.0, 0.6]


def test_evaluate_scores_every_method_on_both_jobs(folder):
    run = harness.evaluate(
        harness.CONFIGS["baselines"], "dev", folder=folder, cutoff=CUTOFF, log=lambda _: None
    )
    assert run.users == 1
    for name in ("random", "popularity", "average_score"):
        assert not math.isnan(run.per_user[name]["A ordering accuracy"][0])
        assert not math.isnan(run.per_user[name]["B nDCG@10"][0])
    summary = harness.summarize(run)
    assert "difference" in summary["popularity"]["A ordering accuracy"]
    assert "difference" not in summary["average_score"]["A ordering accuracy"]
    report = harness.markdown(run, summary, {"commit": "abc"})
    assert "| `average_score` |" in report and "## Slices" in report


def test_profile_is_built_from_the_history_only():
    history = (entry(1, score=90), entry(2, score=50))
    user = harness.EvalUser(
        "u", history, np.array([]), np.array([]), np.array([]), np.array([]), frozenset(), False
    )
    assert {p.entry.anilist_id for p in user.profile.entries} == {1, 2}


def test_slices_sort_by_job_then_the_design_notes_order():
    keys = ["B recall: niche", "A: users with dates", "A: sequels", "A: popular", "B recall: popular"]
    assert sorted(keys, key=harness.slice_order) == [
        "A: sequels",
        "A: popular",
        "A: users with dates",
        "B recall: popular",
        "B recall: niche",
    ]

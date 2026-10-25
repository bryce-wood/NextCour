"""The evaluation harness and its methods, end to end on a tiny synthetic split."""

import math

import duckdb
import numpy as np
import pytest
from helpers import entry
from test_guardrail import fake_frozen_set
from test_split import AFTER, BEFORE, CUTOFF, filler, row, write_inputs

from nextcour.core import guardrail
from nextcour.core.shows import Feature, Show
from nextcour.evaluation import harness, methods, split
from nextcour.models.scorer_v0 import ScorerV0


@pytest.fixture
def folder(tmp_path, monkeypatch):
    monkeypatch.setattr(split, "DEV_SHARE", 1.0)
    monkeypatch.setattr(guardrail, "frozen_set_if_present", lambda: fake_frozen_set())
    monkeypatch.setattr(harness, "MIN_SCORES", 1)
    lists = [
        # The dev user: history on the filler shows and show 1, then six shows after the cutoff.
        *filler("viewer", 10),
        row("viewer", 1, updated=BEFORE, score=9),  # starts franchise 1
        row("viewer", 2, updated=AFTER, score=9),  # a sequel in franchise 1
        row("viewer", 3, updated=AFTER, score=6),  # another entry of franchise 1, aired before show 1
        row("viewer", 4, updated=AFTER, score=10),  # a new franchise, enjoyed
        row("viewer", 5, updated=AFTER, score=3),  # aired after the cutoff
        row("viewer", 20, updated=AFTER, score=5),
        row("viewer", 21, updated=AFTER, score=6),
        # Train users, whose history before the cutoff makes the crowd numbers and similarities.
        *[row(f"t{i}", 4, updated=BEFORE, score=9) for i in range(3)],
        *[row(f"t{i}", 20, updated=BEFORE, score=4) for i in range(5)],
        *[row(f"t{i}", 21, updated=BEFORE, score=6) for i in range(2)],
    ]
    write_inputs(tmp_path, monkeypatch, lists)
    split.build_training_split(cutoff=CUTOFF, release="test")
    return split.split_dir(CUTOFF)


@pytest.fixture
def setup(folder):
    """The catalog, a fitting context and the dev user, ready for a method."""
    con = duckdb.connect()
    catalog = harness.load_catalog(folder, CUTOFF, con)
    franchise_of = split.franchise_lookup(CUTOFF, folder)
    ctx = methods.Context(folder, con, "test", franchise_of, log=lambda _: None)
    (raw,) = split.load_users("dev", folder=folder)
    return catalog, ctx, harness.prepare(raw, catalog, franchise_of)


def test_crowd_numbers_come_from_train_users_before_the_cutoff(folder):
    catalog = harness.load_catalog(folder, CUTOFF)
    at = catalog.index
    assert catalog.members[at[120]] == 5  # filler show 20 (AniList 120): five train users
    assert catalog.average[at[120]] == pytest.approx(40)
    assert catalog.average[at[104]] == pytest.approx(90)
    assert catalog.members[at[102]] == 0  # only the dev user watched it, after the cutoff
    assert math.isnan(catalog.average[at[102]])
    assert catalog.aired_after_cutoff[at[105]] and not catalog.aired_after_cutoff[at[104]]


def test_discovery_leaves_out_the_history_and_started_franchises(setup):
    catalog, _, user = setup
    base = np.flatnonzero(catalog.available & catalog.entry_point)
    candidates = set(catalog.ids[harness.discovery_candidates(user, catalog, base)].tolist())
    assert 101 not in candidates and 102 not in candidates  # franchise 1 is started
    assert 110 not in candidates  # a filler show in the history
    assert 104 in candidates and 105 in candidates
    sequels = {int(catalog.ids[p]) for p, s in zip(user.targets, user.sequel, strict=True) if s}
    assert sequels == {102, 103}


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


def test_evaluate_scores_each_method_on_its_own_jobs(folder):
    run = harness.evaluate(
        harness.CONFIGS["baselines"], "dev", folder=folder, cutoff=CUTOFF, log=lambda _: None
    )
    assert run.users == 1
    for name in ("random", "popularity", "average_score", "item_item", "content_similarity", "scorer_v0_mal"):
        for key in ("A ordering accuracy", "B nDCG@10", "D ordering accuracy"):
            assert not math.isnan(run.per_user[name][key][0]), (name, key)
    rule = run.per_user["sequel_rule"]
    assert not math.isnan(rule["D ordering accuracy"][0])
    assert math.isnan(rule["A ordering accuracy"][0]) and math.isnan(rule["B nDCG@10"][0])
    summary = harness.summarize(run)
    assert "difference" in summary["popularity"]["A ordering accuracy"]
    assert "difference" not in summary["average_score"]["A ordering accuracy"]
    report = harness.markdown(run, summary, {"commit": "abc"})
    assert "| `sequel_rule` | — | — | — |" in report and "## Slices" in report


def test_item_similarity_is_the_cosine_of_who_watched_both(setup):
    catalog, ctx, _ = setup
    similarity = methods.fit_item_similarity(catalog, ctx)
    at = catalog.index
    # Shows 4, 20 and 21 were watched by 3, 5 and 2 train users; 4 and 20 together by 3.
    assert similarity[at[104], at[120]] == pytest.approx(3 / math.sqrt(3 * 5))
    assert similarity[at[120], at[121]] == pytest.approx(2 / math.sqrt(5 * 2))
    assert similarity[at[104], at[104]] == 0  # a show isn't its own neighbor
    assert similarity[at[101], at[104]] == 0  # nobody in train watched show 1
    again = methods.fit_item_similarity(catalog, ctx)  # read back from the cache
    assert np.array_equal(np.asarray(again), np.asarray(similarity))


def test_content_similarity_favors_rare_shared_features(setup):
    catalog, ctx, user = setup
    method = methods.ContentSimilarity.fit(catalog, methods.load_features(catalog, ctx))
    at = catalog.index
    # The only enjoyed show in the history is show 1: tags "common" and "odd". "Common" is on
    # every show, so it carries no weight; show 5 shares "odd", show 4 doesn't.
    scores = method.scores(user, np.array([at[105], at[104]]))
    assert scores[0] > 0 and scores[1] == pytest.approx(0)


def test_sequel_rule_uses_the_latest_earlier_entry(setup):
    catalog, _, user = setup
    at = catalog.index
    (show_1,) = [p for p in user.profile.entries if p.entry.anilist_id == 101]
    scores = methods.SequelRule(catalog).scores(user, np.array([at[102], at[103], at[104]]))
    # Show 2 aired after show 1, so it takes show 1's enjoyment; show 3 aired before show 1, so it
    # takes the franchise's average, which is the same here; show 4 isn't a sequel.
    assert scores.tolist() == pytest.approx([show_1.enjoyment, show_1.enjoyment, -1.0])


def v0_shows(catalog, features):
    """Show objects with the same features and averages, for scorer v0 itself."""
    shows = {}
    for i, anilist_id in enumerate(catalog.ids.tolist()):
        names = [features.names[j] for j in np.flatnonzero(features.matrix[i])]
        average = catalog.average[i]
        shows[anilist_id] = Show(
            anilist_id=anilist_id,
            title=f"Show {anilist_id}",
            format=None,
            aired=None,
            average_score=None if math.isnan(average) else float(average),
            popularity=None,
            features=tuple(Feature(n.split(":")[0], n, n, 1.0) for n in names),
        )
    return shows


def assert_v0_parity(catalog, features, user, positions):
    fast = methods.ScorerV0Mal(catalog, features.matrix).scores(user, positions)
    shows = v0_shows(catalog, features)
    scorer = ScorerV0(user.profile, shows)
    slow = [scorer.predict(shows[int(catalog.ids[p])]) for p in positions]
    assert fast.tolist() == pytest.approx(slow, abs=1e-9)


def test_scorer_v0_mal_matches_scorer_v0(setup):
    catalog, ctx, user = setup
    features = methods.load_features(catalog, ctx)
    assert_v0_parity(catalog, features, user, np.arange(len(catalog)))


@pytest.mark.local_data
def test_scorer_v0_mal_matches_scorer_v0_on_real_dev_users():
    folder = split.split_dir()
    if not (folder / "dev.parquet").exists():
        pytest.skip("no local split")
    con = duckdb.connect()
    catalog = harness.load_catalog(folder, split.CUTOFF, con)
    franchise_of = split.franchise_lookup(folder=folder)
    record = split.json.loads((folder / "split.json").read_text(encoding="utf-8"))
    ctx = methods.Context(folder, con, record["catalog_release"], franchise_of, log=lambda _: None)
    features = methods.load_features(catalog, ctx)
    rng = np.random.default_rng(0)
    for raw in split.load_users("dev", limit=5, folder=folder, con=con):
        user = harness.prepare(raw, catalog, franchise_of)
        assert_v0_parity(catalog, features, user, rng.choice(len(catalog), 200, replace=False))


def test_profile_is_built_from_the_history_only():
    history = (entry(1, score=90), entry(2, score=50))
    user = harness.EvalUser(
        "u", history, np.array([]), np.array([]), np.array([]), np.array([]), frozenset(), False
    )
    assert {p.entry.anilist_id for p in user.profile.entries} == {1, 2}


def test_slices_sort_by_job_then_the_design_notes_order():
    keys = ["B recall: niche", "A: users with dates", "A: new franchises", "A: popular", "B recall: popular"]
    assert sorted(keys, key=harness.slice_order) == [
        "A: new franchises",
        "A: popular",
        "A: users with dates",
        "B recall: popular",
        "B recall: niche",
    ]

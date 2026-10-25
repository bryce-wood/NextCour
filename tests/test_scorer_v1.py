"""Scorer v1: personal points from enjoyment, no near-ties, and a series point."""

import pytest
from helpers import entry

from nextcour.core.shows import Feature, Show
from nextcour.models import scorer_v1
from nextcour.models.scorer_v0 import ScorerV0
from nextcour.models.scorer_v1 import ScorerV1
from nextcour.profile import builder


def show(anilist_id, average, *tags):
    features = tuple(Feature("tag", f"tag:{t}", t, 1.0) for t in tags)
    return Show(anilist_id, f"Show {anilist_id}", "TV", None, average, None, features)


def test_push_is_zero_for_near_ties():
    assert scorer_v1.push(0.9) == pytest.approx(0.4)
    assert scorer_v1.push(0.2) == pytest.approx(-0.3)
    assert scorer_v1.push(0.55) == 0.0 and scorer_v1.push(0.41) == 0.0


def test_a_show_liked_less_than_usual_counts_against_its_features():
    # Show 2 has the lowest average but I liked it more than that, so v0's residual is positive;
    # it's still below my middle, so v1 counts it against its tag.
    # Twelve shows, so relative scores aren't pulled into the near-tie zone.
    scores = {2: 30, 3: 20, **{i: 70 + 2 * i for i in range(4, 14)}}
    averages = {2: 40, 3: 85, **{i: 75 + i for i in range(4, 14)}}
    history = [entry(i, score=s) for i, s in scores.items()]
    shows = {i: show(i, averages[i], "fanservice" if i == 2 else "other") for i in scores}
    shows[9] = show(9, None, "fanservice")
    profile = builder.build(history, {})
    v0, v1 = ScorerV0(profile, shows), ScorerV1(profile, shows, {})
    assert v0.residual[2] > 0
    assert v1.push[2] < 0
    assert v0.affinity("tag:fanservice") > 0 > v1.affinity("tag:fanservice")
    # A new show with no average: v0 pushes it up for the tag, v1 down.
    assert v0.predict(shows[9]) > 0.5 > v1.predict(shows[9])


def test_near_ties_add_evidence_but_no_push():
    history = [entry(1, score=90), entry(2, score=70), entry(3, score=10)]
    shows = {1: show(1, 80, "a"), 2: show(2, 80, "b"), 3: show(3, 80, "c")}
    v1 = ScorerV1(builder.build(history, {}), shows, {})
    assert v1.push[2] == 0.0  # the middle score of three
    assert v1.affinity("tag:b") == 0.0
    assert v1._evidence["tag:b"] > 0


def test_the_series_point_blends_my_enjoyment_of_the_franchise():
    history = [entry(1, score=90, aired="2019-01-01"), entry(2, score=40), entry(3, score=60)]
    franchise_of = {1: 7, 5: 7}
    shows = {1: show(1, 80), 2: show(2, 80), 3: show(3, 80), 5: show(5, 80)}
    profile = builder.build(history, franchise_of)
    v1 = ScorerV1(profile, shows, franchise_of)
    score = v1.score(shows[5])
    (first,) = [p for p in profile.entries if p.entry.anilist_id == 1]
    share = first.weight / (first.weight + scorer_v1.SERIES_SHRINK)
    assert score.series == pytest.approx(first.enjoyment) and score.series_share == pytest.approx(share)
    base = score.crowd_prior  # no features and no links: the base is the crowd prior
    assert score.predicted == pytest.approx((1 - share) * base + share * first.enjoyment)
    assert v1.score(shows[2]).series is None  # not in a franchise I've watched


def test_contributions_add_up_to_the_prediction():
    history = [entry(1, score=90), entry(2, score=40), entry(3, score=70), entry(4, score=20)]
    franchise_of = {1: 7, 6: 7}
    shows = {
        1: show(1, 80, "a", "b"),
        2: show(2, 60, "b"),
        3: show(3, 75, "c"),
        4: show(4, 50, "a"),
        6: show(6, 78, "a", "c"),
    }
    score = ScorerV1(builder.build(history, franchise_of), shows, franchise_of).score(shows[6])
    assert score.contributions[0].kind == "average_score"
    assert sum(c.value for c in score.contributions) == pytest.approx(score.predicted)
    assert any(c.kind == "series" for c in score.contributions)

import pytest
from helpers import entry

from nextcour.core.shows import Feature, Show
from nextcour.models import scorer_v0
from nextcour.models.scorer_v0 import ScorerV0
from nextcour.profile import builder

TAG = Feature("tag", "tag:1", "Found Family", 1.0)
STUDIO = Feature("studio", "studio:1", "Some Studio", 1.0)


def show(anilist_id, average, features=(), recommendations=None):
    return Show(
        anilist_id, f"Show {anilist_id}", "TV", None, average, 1000, tuple(features), recommendations or {}
    )


def scorer(history, shows):
    return ScorerV0(builder.build(history, {}), {s.anilist_id: s for s in shows})


def test_crowd_prior_orders_like_average_score_and_is_neutral_when_missing():
    s = scorer([entry(1, score=60), entry(2, score=80)], [show(1, 60), show(2, 80)])
    assert s.crowd_prior(show(9, 90)) > s.crowd_prior(show(8, 70)) > s.crowd_prior(show(7, 50))
    assert s.crowd_prior(show(6, None)) == scorer_v0.NEUTRAL


def test_taste_credits_features_of_shows_i_liked_more_than_the_crowd():
    # Same crowd scores; I liked the two shows with the tag far more than the two without.
    history = [entry(1, score=95), entry(2, score=90), entry(3, score=50), entry(4, score=40)]
    shows = [show(1, 70, [TAG]), show(2, 70, [TAG]), show(3, 70), show(4, 70)]
    s = scorer(history, shows)
    assert s.affinity("tag:1") > 0
    with_tag, without = s.score(show(9, 75, [TAG])), s.score(show(8, 75, [STUDIO]))
    assert with_tag.predicted > without.predicted
    assert without.taste == 0  # an unseen feature is no evidence


def test_a_feature_of_shows_i_rated_as_the_crowd_did_earns_nothing():
    # My scores track the crowd's exactly, so the studio's shows have no residual.
    history = [entry(1, score=90), entry(2, score=50)]
    s = scorer(history, [show(1, 90, [STUDIO]), show(2, 50, [STUDIO])])
    assert s.affinity("studio:1") == pytest.approx(0)


def test_links_from_a_show_i_liked_more_than_the_crowd_raise_the_score():
    history = [entry(1, score=95), entry(2, score=40), entry(3, score=60)]
    shows = [show(1, 60, recommendations={9: 30}), show(2, 80), show(3, 70)]
    s = scorer(history, shows)
    linked, unlinked = s.score(show(9, 75)), s.score(show(8, 75))
    assert linked.links > 0 and unlinked.links == 0
    assert linked.predicted > unlinked.predicted
    assert linked.contributions[1].kind == "crowd_link"


def test_prediction_is_the_stated_blend_of_its_parts():
    history = [entry(1, score=95), entry(2, score=40), entry(3, score=60)]
    shows = [show(1, 60, [TAG], {9: 30}), show(2, 80, [STUDIO]), show(3, 70)]
    result = scorer(history, shows).score(show(9, 75, [TAG, STUDIO]))
    assert result.predicted == pytest.approx(
        result.crowd_prior + scorer_v0.TASTE_WEIGHT * result.taste + scorer_v0.LINK_WEIGHT * result.links
    )
    personal = sum(c.value for c in result.contributions[1:])
    assert personal == pytest.approx(result.predicted - result.crowd_prior)

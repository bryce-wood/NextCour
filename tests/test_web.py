import json

import pytest
from fastapi.testclient import TestClient

from nextcour.app.feedback import FeedbackStore
from nextcour.app.service import Ranking, Row, watch_time
from nextcour.app.web import Jobs, create_app
from nextcour.candidates.planning import HeldBack
from nextcour.core.shows import Show
from nextcour.explain import compare, evidence
from nextcour.explain.gemini import ModelUnavailable, Reply
from nextcour.explain.writer import ExplanationStore, Item, Writer
from nextcour.ingest import anilist
from nextcour.models.scorer_v0 import Contribution, Score


def show(anilist_id, title, episodes=12):
    return Show(anilist_id, title, "TV", None, 80, 100, episodes=episodes, duration=24)


def row(rank, anilist_id, title):
    contributions = (
        Contribution("average_score", "average_score", title, 0.7, {"average_score": 80, "share_below": 0.7}),
        Contribution(
            "crowd_link",
            "show:9",
            "Mob Psycho 100",
            0.04,
            {
                "votes": 42,
                "title": "Mob Psycho 100",
                "anilist_id": 9,
                "my_score": 95,
                "status": "COMPLETED",
                "average_score": 86,
                "residual": 0.2,
                "weight": 1.0,
            },
        ),
    )
    score = Score(anilist_id, 0.74 - rank / 100, 0.7, 0.0, 0.08, contributions)
    item = Item(anilist_id, title, tuple(evidence.build(score, title)))
    return Row(rank, show(anilist_id, title), score, rank, item)


class FakePlanner:
    def __init__(self, feedback):
        self.feedback = feedback
        self.shows = [(1, "Frieren"), (2, "Bocchi"), (3, "Mushishi")]

    def ranking(self, user):
        if user == "nobody":
            raise anilist.NotFound("not found", 404)
        if user == "outage":
            raise anilist.AniListError(
                "AniList error 403: The AniList API has been temporarily disabled", 403
            )
        hidden = self.feedback.not_interested("Someone")
        shown = [(i, t) for i, t in self.shows if i not in hidden]
        rows = [row(rank, i, t) for rank, (i, t) in enumerate(shown, 1)]
        held = [HeldBack(show(4, "Frieren Season 2"), ("Frieren (planning)",))]
        hidden_shows = [show(i, t) for i, t in self.shows if i in hidden]
        return Ranking("Someone", rows, held, hidden_shows, 42)


class FakeModels:
    def __init__(self):
        self.calls = 0

    def generate_json(self, model, system, prompt_text, schema):
        self.calls += 1
        keys = [s["show_id"] for s in json.loads(prompt_text)["shows"]]
        return Reply(
            {
                "explanations": [
                    {
                        "show_id": k,
                        "sentences": [
                            {"text": "Fans of Mob Psycho 100 recommend it.", "evidence": [f"{k}b"]}
                        ],
                    }
                    for k in keys
                ]
            },
            model,
            {},
        )


@pytest.fixture
def app_parts(tmp_path):
    feedback = FeedbackStore(":memory:")
    writer = Writer(ExplanationStore(":memory:"), None)
    votes = compare.Votes(":memory:")
    planner = FakePlanner(feedback)
    app = create_app(planner, writer, feedback, votes, comparisons=tmp_path)
    return TestClient(app), feedback, writer, votes, tmp_path


def test_index_and_redirect(app_parts):
    client, *_ = app_parts
    assert "AniList user name" in client.get("/").text
    response = client.get("/", params={"user": " Someone "}, follow_redirects=False)
    assert response.status_code == 303 and response.headers["location"] == "/u/Someone"
    response = client.get("/", params={"user": "a/b c"}, follow_redirects=False)
    assert response.headers["location"] == "/u/a%2Fb%20c"


def test_the_page_says_why_explanations_are_still_plain(tmp_path):
    class Busy:
        def generate_json(self, *args):
            raise ModelUnavailable("m is busy or unreachable")

    feedback = FeedbackStore(":memory:")
    writer = Writer(ExplanationStore(":memory:"), Busy(), ("m",))
    jobs = Jobs()
    app = create_app(FakePlanner(feedback), writer, feedback, compare.Votes(":memory:"), tmp_path, jobs)
    client = TestClient(app)
    client.get("/u/Someone")
    jobs.running["someone"].join(timeout=5)
    page = client.get("/u/Someone").text
    assert "3 explanations still use the plain version" in page
    assert "m is busy or unreachable" in page
    assert "Write them now" in page  # no automatic retry right after a failure


def test_planning_page_shows_the_ranked_list(app_parts):
    client, *_ = app_parts
    page = client.get("/u/Someone").text
    assert page.index("Frieren") < page.index("Bocchi") < page.index("Mushishi")
    assert "about 4.8 hours" in page
    assert "Frieren Season 2" in page and "Frieren (planning)" in page  # held back, with the reason
    assert "AniList average score of 80" in page  # the template stands in
    assert "GEMINI_API_KEY" in page  # no writer configured


def test_titles_from_anilist_are_escaped():
    feedback = FeedbackStore(":memory:")
    planner = FakePlanner(feedback)
    planner.shows = [(1, "<script>alert(1)</script>")]
    writer = Writer(ExplanationStore(":memory:"), None)
    page = TestClient(create_app(planner, writer, feedback, compare.Votes(":memory:"))).get("/u/Someone").text
    assert "<script>alert(1)</script>" not in page
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in page


def test_not_interested_hides_a_show_until_undone(app_parts):
    client, feedback, *_ = app_parts
    assert client.post("/u/Someone/not-interested/2").headers["HX-Refresh"] == "true"
    assert feedback.not_interested("Someone") == {2}
    page = client.get("/u/Someone").text
    assert 'id="show-2"' not in page and "Undo" in page
    client.post("/u/Someone/interested/2")
    assert 'id="show-2"' in client.get("/u/Someone").text


def test_a_pick_is_logged_with_its_rank_and_explanation(app_parts):
    client, feedback, *_ = app_parts
    assert "Noted" in client.post("/u/Someone/pick/3").text
    (pick,) = feedback.picks("Someone")
    assert (pick.anilist_id, pick.rank, pick.title) == (3, 3, "Mushishi")
    assert "average score of 80" in pick.explanation
    assert client.post("/u/Someone/pick/99").status_code == 404


def test_json_matches_the_page(app_parts):
    client, *_ = app_parts
    data = client.get("/api/u/Someone/planning").json()
    assert [r["title"] for r in data["ranked"]] == ["Frieren", "Bocchi", "Mushishi"]
    first = data["ranked"][0]
    assert first["watch_minutes"] == 288 and first["evidence"][0]["id"] == "a"
    assert first["explanation"]["writer"] == "template"
    assert data["held_back"][0]["waiting_on"] == ["Frieren (planning)"]


def test_unknown_users_get_a_404(app_parts):
    client, *_ = app_parts
    assert client.get("/u/nobody").status_code == 404
    response = client.get("/api/u/nobody/planning")
    assert response.status_code == 404 and "No public AniList" in response.json()["error"]


def test_an_anilist_outage_says_so(app_parts):
    client, *_ = app_parts
    response = client.get("/u/outage")
    assert response.status_code == 503
    assert "available right now" in response.text and "/compare" in response.text


def test_explanations_are_written_in_the_background(tmp_path):
    feedback = FeedbackStore(":memory:")
    models = FakeModels()
    writer = Writer(ExplanationStore(":memory:"), models, ("m",))
    jobs = Jobs()
    app = create_app(FakePlanner(feedback), writer, feedback, compare.Votes(":memory:"), tmp_path, jobs)
    client = TestClient(app)
    first = client.get("/u/Someone").text
    assert "Writing explanations" in first
    jobs.running["someone"].join(timeout=5)
    assert jobs.progress["someone"] == (3, 3)  # written, out of those to write
    assert client.get("/u/Someone/status").headers["HX-Refresh"] == "true"
    page = client.get("/u/Someone").text
    assert page.count("Fans of Mob Psycho 100 recommend it.") == 3
    assert "written by m" in page
    assert models.calls == 1  # one batch for all three shows, and no rewrite after


def test_the_comparison_hides_models_until_the_results(app_parts):
    client, _, _, votes, folder = app_parts
    assert client.get("/compare").status_code == 404
    items = [
        {
            "anilist_id": 1,
            "title": "Frieren",
            "rank": 1,
            "evidence": [{"id": "a", "fact": "A fact.", "effect": "a strong point in its favor"}],
            "a_model": "secret-model-x",
            "b_model": "secret-model-y",
            "a": {
                "sentences": [{"text": "Text from A.", "evidence": ["a"]}],
                "writer": "secret-model-x",
                "dropped": [],
            },
            "b": {
                "sentences": [{"text": "Text from B.", "evidence": ["a"]}],
                "writer": "template",
                "dropped": ["x"],
            },
        }
    ]
    comparison = {
        "id": "c1",
        "user": "Someone",
        "created_at": "2026-10-21T07:00:00+00:00",
        "models": ["secret-model-x", "secret-model-y"],
        "seconds_per_batch": {"secret-model-x": [3.0], "secret-model-y": [240.0]},
        "batch_size": 10,
        "prompt_version": 3,
        "items": items,
    }
    compare.save(comparison, folder)
    page = client.get("/compare").text
    assert "Text from A." in page and "Text from B." in page
    assert "secret-model" not in page
    early = client.get("/compare/results").text
    assert "reveal it now" in early and "Your vote" not in early  # no A/B answers before the votes
    assert "Your vote" in client.get("/compare/results", params={"reveal": "true"}).text
    response = client.post("/compare/c1/1/a")
    assert "chosen" in response.text and "All done" in response.text
    assert votes.of("c1") == {1: "a"}
    results = client.get("/compare/results").text
    assert "secret-model-x" in results and "1 wins" in results
    assert client.post("/compare/c1/1/maybe").status_code == 400


def test_watch_time_reads_naturally():
    assert watch_time(show(1, "x", episodes=1)) == "about 24 minutes"
    assert watch_time(show(1, "x", episodes=26)) == "about 10 hours"
    assert watch_time(show(1, "x", episodes=None)) is None


def test_series_watched_lists_earlier_entries_as_context():
    from helpers import entry

    from nextcour.app.service import series_watched

    entries = [
        entry(1, aired="2018-04-01", score=92),
        entry(2, aired="2020-01-01", score=0, status="DROPPED", progress=3),
        entry(3, aired="2021-01-01", status="PLANNING", score=0),
        entry(4, aired="2019-01-01", score=70),
    ]
    franchise_of = {1: 10, 2: 10, 3: 10, 5: 10, 4: 20}
    assert series_watched(5, entries, franchise_of) == (
        "Show 1 (you gave it 92/100)",
        "Show 2 (dropped, unscored)",
    )
    assert series_watched(99, entries, franchise_of) == ()

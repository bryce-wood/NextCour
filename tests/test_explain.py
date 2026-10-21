import json

import pytest

from nextcour.explain import compare, evidence
from nextcour.explain.evidence import Evidence
from nextcour.explain.gemini import ModelUnavailable, Reply
from nextcour.explain.writer import (
    TEMPLATE,
    ExplanationStore,
    Item,
    Writer,
    check,
    prompt,
    template,
    write_batch,
)
from nextcour.models.scorer_v0 import Contribution, Score


def carrier(title, score, average, residual, weight=1.0, status="COMPLETED", anilist_id=7):
    return {
        "title": title,
        "anilist_id": anilist_id,
        "my_score": score,
        "status": status,
        "average_score": average,
        "residual": residual,
        "weight": weight,
    }


def scored(anilist_id=1):
    contributions = (
        Contribution(
            "average_score", "average_score", "Show", 0.8, {"average_score": 84, "share_below": 0.9}
        ),
        Contribution(
            "crowd_link",
            "show:7",
            "Mob Psycho 100",
            0.05,
            {"votes": 42, **carrier("Mob Psycho 100", 95, 86, 0.2)},
        ),
        Contribution(
            "tag",
            "tag:3",
            "Found Family",
            -0.02,
            {
                "relevance": 0.8,
                "role": None,
                "shows": [
                    carrier("A", 60, 80, -0.3, 1.0),
                    carrier("B", 90, 70, 0.2),
                    carrier("C", 50, 75, -0.1),
                ],
            },
        ),
        Contribution("genre", "genre:Drama", "Drama", 0.0001, {"relevance": 1.0, "shows": []}),
    )
    return Score(anilist_id, 0.815, 0.8, -0.04, 0.1, contributions)


def item(anilist_id=1, title="Show"):
    return Item(anilist_id, title, tuple(evidence.build(scored(anilist_id), title)))


# --- Evidence --------------------------------------------------------------------------------


def test_evidence_states_real_numbers_and_skips_tiny_effects():
    items = evidence.build(scored(), "Show")
    assert [e.id for e in items] == ["a", "b", "c"]  # the tiny Drama effect is left out
    average, link, tag = items
    assert (
        average.fact
        == "Show has an AniList average score of 84, higher than 90% of the shows you've watched."
    )
    assert average.described_effect == "a strong point in its favor"
    assert "Mob Psycho 100" in link.fact and "42 votes" in link.fact and "95/100" in link.fact
    assert link.described_effect == "a moderate point in its favor"
    # The tag lowers the score, so the examples are the shows that pull it down: A, then C.
    assert "less than AniList users did" in tag.fact
    assert "A (you gave it 60/100; it averages 80)" in tag.fact and "C (" in tag.fact
    assert "B (" not in tag.fact
    assert tag.described_effect == "a slight point against it"


def test_evidence_hash_changes_with_the_facts():
    one, two = evidence.build(scored(), "Show"), evidence.build(scored(), "Other title")
    assert evidence.evidence_hash(one) == evidence.evidence_hash(evidence.build(scored(), "Show"))
    assert evidence.evidence_hash(one) != evidence.evidence_hash(two)


# --- The checker and templates ---------------------------------------------------------------


def test_checker_keeps_grounded_sentences_and_says_why_it_drops_others():
    raw = [
        {"text": "It averages 84 on AniList, above 90% of what you've watched.", "evidence": ["S1a"]},
        {"text": "Fans of Mob Psycho 100 recommend it.", "evidence": ["S1b"]},
        {"text": "It has 12 episodes.", "evidence": ["S1a"]},  # 12 isn't in the evidence
        {"text": "You liked it.", "evidence": []},
        {"text": "You liked it.", "evidence": ["S2a"]},  # another show's evidence
        {"text": "See S1b for why.", "evidence": ["S1b"]},
        {"text": "The prediction is high.", "evidence": ["S1a"]},
    ]
    kept, dropped = check(item(), "S1", raw)
    assert [s.text for s in kept] == [raw[0]["text"], raw[1]["text"]]
    assert [s.evidence for s in kept] == [("a",), ("b",)]
    assert [d.split(":")[0] for d in dropped] == [
        "has numbers its evidence doesn't (12)",
        "cites no evidence",
        "cites evidence that isn't this show's (S2a)",
        "puts an evidence ID in its text",
        "talks about the prediction or the evidence",
    ]


def test_numbers_in_the_title_are_allowed():
    titled = item(title="Mob Psycho 100 III")
    kept, _ = check(titled, "S1", [{"text": "Mob Psycho 100 III averages 84.", "evidence": ["S1a"]}])
    assert len(kept) == 1


def test_template_leads_with_the_average_then_the_biggest_personal_fact():
    explanation = template(item())
    assert explanation.writer == TEMPLATE
    assert [s.evidence for s in explanation.sentences] == [("a",), ("b",)]


def test_prompt_holds_only_the_evidence():
    data = json.loads(prompt([("S1", item())]))
    assert data["shows"][0]["show_id"] == "S1"
    assert [e["id"] for e in data["shows"][0]["evidence"]] == ["S1a", "S1b", "S1c"]
    assert set(data["shows"][0]["evidence"][0]) == {"id", "fact", "effect"}


# --- Writing, with a fake model ----------------------------------------------------------------


class FakeModels:
    """Answers per model: a dict to return as JSON, or an exception to raise."""

    def __init__(self, answers):
        self.answers = answers
        self.calls = []

    def generate_json(self, model, system, prompt_text, schema):
        self.calls.append(model)
        answer = self.answers[model]
        if isinstance(answer, Exception):
            raise answer
        return Reply(answer, model, {})


def good_answer(*keys):
    return {
        "explanations": [
            {"show_id": key, "sentences": [{"text": "It averages 84 on AniList.", "evidence": [f"{key}a"]}]}
            for key in keys
        ]
    }


def test_the_backup_model_steps_in_and_templates_come_last():
    items = [item(1), item(2)]
    models = FakeModels({"main": ModelUnavailable("busy"), "backup": good_answer("S1", "S2")})
    results, answered = write_batch(items, models, ["main", "backup"])
    assert answered == "backup"
    assert {r.writer for r in results.values()} == {"backup"}

    models = FakeModels({"main": ModelUnavailable("busy"), "backup": ModelUnavailable("quota")})
    results, answered = write_batch(items, models, ["main", "backup"])
    assert answered is None
    assert {r.writer for r in results.values()} == {TEMPLATE}
    assert "busy" in results[1].note and "quota" in results[1].note


def test_answers_are_matched_even_without_the_show_id():
    items = [item(1, "First"), item(2, "Second")]
    answer = {
        "explanations": [
            {"show_id": "Second", "sentences": [{"text": "Second averages 84.", "evidence": ["S2a"]}]},
            {"show_id": "", "sentences": [{"text": "First averages 84.", "evidence": ["S1a"]}]},
        ]
    }
    results, _ = write_batch(items, FakeModels({"m": answer}), ["m"])
    assert results[1].text == "First averages 84."
    assert results[2].text == "Second averages 84."


def test_a_show_with_no_usable_sentence_gets_the_template():
    answer = {
        "explanations": [
            {"show_id": "S1", "sentences": [{"text": "It has 99 episodes.", "evidence": ["S1a"]}]}
        ]
    }
    results, answered = write_batch([item()], FakeModels({"m": answer}), ["m"])
    assert answered == "m"
    assert results[1].writer == TEMPLATE
    assert results[1].dropped and "no usable sentence" in results[1].note


def test_an_explanation_without_numbers_gets_the_template():
    vague = {
        "explanations": [
            {"show_id": "S1", "sentences": [{"text": "Fans of that show like it.", "evidence": ["S1b"]}]}
        ]
    }
    results, _ = write_batch([item()], FakeModels({"m": vague}), ["m"])
    assert results[1].writer == TEMPLATE and "no numbers" in results[1].note


def test_the_writer_caches_what_a_model_answered_but_not_failures():
    store = ExplanationStore(":memory:")
    models = FakeModels({"m": ModelUnavailable("busy")})
    writer = Writer(store, models, ("m",))
    assert writer.write("Someone", [item()])[1].writer == TEMPLATE
    assert writer.cached("Someone", [item()]) == {}  # a later run can still try a model

    models.answers["m"] = good_answer("S1")
    assert writer.write("Someone", [item()])[1].writer == "m"
    calls = len(models.calls)
    assert writer.write("someone", [item()])[1].writer == "m"  # cached, and user names ignore case
    assert len(models.calls) == calls


def test_cached_explanations_expire(tmp_path):
    now = [1_000_000.0]
    path = tmp_path / "app.sqlite"
    ExplanationStore(path, clock=lambda: now[0]).put("x", item(), template(item()))
    assert ExplanationStore(path, clock=lambda: now[0]).get("x", item()) is not None
    now[0] += 31 * 86400  # old explanations are deleted when a store opens
    assert ExplanationStore(path, clock=lambda: now[0]).get("x", item()) is None


# --- The blind comparison ----------------------------------------------------------------------


def test_spread_includes_both_ends():
    assert compare.spread(60, 20)[0] == 0 and compare.spread(60, 20)[-1] == 59
    assert len(compare.spread(60, 20)) == 20
    assert compare.spread(5, 20) == [0, 1, 2, 3, 4]


def test_comparison_hides_the_models_and_tallies_votes():
    rows = [(1, item(1)), (5, item(2))]
    models = FakeModels({"x": good_answer("S1", "S2"), "y": ModelUnavailable("busy")})
    comparison = compare.prepare("Someone", rows, models, ("x", "y"), seed=3, clock=lambda: 0.0)
    assert [i["rank"] for i in comparison["items"]] == [1, 5]
    for i in comparison["items"]:
        assert {i["a_model"], i["b_model"]} == {"x", "y"}
    first = comparison["items"][0]
    votes = {first["anilist_id"]: "a" if first["a_model"] == "x" else "b", 2: "same"}
    tallies = {t.model: t for t in compare.tally(comparison, votes)}
    assert (tallies["x"].wins, tallies["x"].same, tallies["y"].losses) == (1, 1, 1)
    assert tallies["y"].templates == 2  # y failed, so the template stood in for both shows


def test_votes_are_stored_and_checked():
    votes = compare.Votes(":memory:")
    votes.vote("c1", 10, "a")
    votes.vote("c1", 10, "both_bad")
    assert votes.of("c1") == {10: "both_bad"}
    with pytest.raises(ValueError):
        votes.vote("c1", 10, "maybe")


def test_evidence_ids_are_letters():
    assert all(isinstance(e, Evidence) and e.id.isalpha() for e in item().evidence)


# --- What gets cited (prompt version 5) --------------------------------------------------------


def link(anilist_id, title, value, residual, votes=30):
    facts = {"votes": votes, **carrier(title, 80, 80, residual, anilist_id=anilist_id)}
    return Contribution("crowd_link", f"show:{anilist_id}", title, value, facts)


def score_of(*personal, share=0.9):
    average = Contribution(
        "average_score", "average_score", "Show", 0.5, {"average_score": 84, "share_below": share}
    )
    prior = 0.5 + 0.92 * (share - 0.5)
    average = Contribution(average.kind, average.key, average.label, prior, average.facts)
    return Score(1, prior, prior, 0.0, 0.0, (average, *personal))


def test_near_ties_are_not_cited():
    items = evidence.build(score_of(link(7, "Near Tie", -0.02, -0.05), link(8, "Clear", 0.03, 0.3)), "Show")
    assert [e.label for e in items[1:]] == ["Clear"]


def test_links_from_one_series_become_one_point():
    links = (link(7, "Slime Season 2", 0.02, 0.2), link(8, "Slime", 0.03, 0.25), link(9, "Other", 0.01, 0.2))
    items = evidence.build(score_of(*links), "Show", franchise_of={7: 100, 8: 100})
    series = items[1]
    assert series.label == "Slime series" and series.effect == pytest.approx(0.05)
    assert "Slime Season 2: 30 votes" in series.fact and "Slime: 30 votes" in series.fact
    assert [e.label for e in items] == ["Show", "Slime series", "Other"]


def test_a_middling_average_is_neutral_and_the_template_leads_with_the_personal_fact():
    items = evidence.build(score_of(link(8, "Clear", 0.03, 0.3), share=0.55), "Show")
    assert items[0].described_effect == "about neutral"
    explanation = template(Item(1, "Show", tuple(items)))
    assert [s.evidence for s in explanation.sentences] == [("b",), ("a",)]


def test_staff_must_come_with_their_role():
    staff = Evidence(
        "b", "staff", "Taku Iwasaki", "Taku Iwasaki worked on it as its music composer.", 0.02, "Music"
    )
    shown = Item(1, "Show", (evidence.build(score_of(), "Show")[0], staff))
    kept, dropped = check(
        shown,
        "S1",
        [
            {"text": "You've liked shows Taku Iwasaki worked on.", "evidence": ["S1b"]},
            {"text": "Taku Iwasaki composed its music.", "evidence": ["S1b"]},
        ],
    )
    assert [s.text for s in kept] == ["Taku Iwasaki composed its music."]
    assert dropped[0].startswith("names Taku Iwasaki without saying what they did")

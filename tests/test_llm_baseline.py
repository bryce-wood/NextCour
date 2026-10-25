"""The LLM baseline: prompts, reading answers, title matching, and scoring as a method."""

import numpy as np
import pytest
from helpers import entry

from nextcour.core import paths
from nextcour.evaluation import llm_baseline as llm
from nextcour.evaluation import split
from nextcour.explain.gemini import BadResponse, ModelUnavailable, Reply


def user(name="secret_user_name", history=(), after=()):
    return split.SplitUser(name, tuple(history), tuple(after))


def test_history_lines_say_what_happened():
    assert llm.history_line(entry(1, score=80)) == "- Show 1 (TV, 2020): 8/10"
    assert llm.show_label("Centaur&#039;s Life", "TV", None) == "Centaur's Life (TV)"
    assert llm.history_line(entry(1, score=0)) == "- Show 1 (TV, 2020): completed, no score"
    dropped = entry(1, score=40, status="DROPPED", progress=3, episodes=12)
    assert llm.history_line(dropped) == "- Show 1 (TV, 2020): dropped after 3 of 12 episodes, 4/10"
    watching = entry(1, score=0, status="CURRENT", progress=5, episodes=None, format="ONA", aired=None)
    assert llm.history_line(watching) == "- Show 1 (ONA): watching, 5 episodes so far"


def test_history_is_most_recent_first_and_capped(monkeypatch):
    monkeypatch.setattr(llm, "MAX_HISTORY", 2)
    history = [entry(i, score=70) for i in (1, 2, 3)] + [entry(4, status="PLANNING", score=0)]
    block = llm.history_block(history)
    assert block.splitlines()[1:] == [
        "- Show 3 (TV, 2020): 7/10",
        "- Show 2 (TV, 2020): 7/10",
        "- (and 1 older shows, left out for length)",
    ]
    assert "3 shows" in block  # planning entries aren't history


def test_ordering_prompt_hides_the_user_and_the_labels():
    targets = [entry(10, score=100), entry(11, score=20), entry(12, score=60)]
    who = user(history=[entry(1, score=70)], after=targets)
    prompt, candidates = llm.ordering_prompt(who, targets)
    assert "secret_user_name" not in prompt
    shown = prompt.split("Anime to order (3):\n")[1]
    assert "/10" not in shown  # the candidates' own scores are the answers; they never appear
    assert sorted(candidates) == [10, 11, 12]
    assert [line.split(". ")[1] for line in shown.splitlines()] == [
        f"Show {a} (TV, 2020)" for a in candidates
    ]
    assert llm.ordering_prompt(who, targets) == (prompt, candidates)  # the shuffle is fixed per user


def test_open_prompt_lists_the_plan_to_watch_shows():
    who = user(history=[entry(1, score=70), entry(2, status="PLANNING", score=0)])
    prompt = llm.open_prompt(who)
    assert "Their plan-to-watch list (1 shows):\n- Show 2 (TV, 2020)" in prompt
    assert "secret_user_name" not in prompt


def answer(version="ordering", data=None, candidates=(), error=None, key="k"):
    return llm.Answer(key, version, llm.MODEL, llm.PROMPT_VERSION, list(candidates), data, error)


def test_parse_order_counts_left_out_and_unknown_ids():
    a = answer(data={"order": ["c2", "C9", "C2", " C1 "]}, candidates=[101, 102, 103])
    ordering = llm.parse_order(a)
    assert ordering.shows == [102, 101]
    assert ordering.missing == 1  # 103 was left out
    assert ordering.unknown == 2  # C9 isn't a candidate; C2 came twice
    assert llm.parse_order(answer(data=None, candidates=[1, 2])).missing == 2


FULL_WIDTH = "".join(chr(ord(c) + 0xFEE0) for c in "KAGUYA")  # as some Japanese text writes Latin letters


def test_titles_are_normalized_and_tried_without_extras():
    assert llm.normalize("Steins;Gate") == "steins gate"
    assert llm.normalize("Centaur&#039;s Life") == llm.normalize("Centaur's Life")
    assert llm.normalize(FULL_WIDTH + " & Friends!") == "kaguya and friends"
    assert llm.title_variants("Mob Psycho 100 (TV)") == ["mob psycho 100 tv", "mob psycho 100"]
    assert llm.title_variants("Haikyu!! Season 1") == ["haikyu season 1", "haikyu"]


def index():
    return llm.TitleIndex(
        shows={"hunter x hunter": [11, 12], "made in abyss": [20], "steins gate": [30]},
        years={11: 1999.0, 12: 2011.0, 20: 2017.0, 30: 2011.0},
        entry_points=frozenset({11, 20, 30}),
    )


def test_title_index_uses_the_year_to_pick_between_shows_with_one_title():
    assert index().match("Hunter x Hunter", 2011) == 12
    assert index().match(f"Hunter {llm.TIMES} Hunter", 1999) == 11  # the official title's multiplication sign
    assert index().match("HUNTER X HUNTER", None) == 11  # no year: the entry point
    assert index().match("Made in Abyss (TV)", 2017) == 20
    assert index().match("Some Show That Isn't There", 2015) is None


def test_parse_picks_counts_unmatched_titles():
    data = {
        "shows": [
            {"title": "Steins;Gate", "year": 2011},
            {"title": "Not A Real Show", "year": 2010},
            {"title": "Steins Gate", "year": 2011},  # the same show again
            {"title": "Made in Abyss", "year": 2017},
        ]
    }
    picks = llm.parse_picks(answer("open", data), index())
    assert picks.shows == [30, 20]
    assert (picks.named, picks.unmatched) == (4, 1)


class StubClient:
    def __init__(self, outcome):
        self.outcome = outcome
        self.calls = []

    def generate_json(self, model, system, prompt, schema, temperature):
        self.calls.append((model, system, prompt, temperature))
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return Reply(self.outcome, model, {"totalTokenCount": 42}, 1.5)


def test_ask_records_answers_and_bad_answers_but_not_a_busy_model():
    targets = [entry(10), entry(11)]
    who = user(history=[entry(1)], after=targets)
    good = llm.ask(StubClient({"order": ["C1", "C2"]}), "ordering", who, targets)
    assert (
        good.data == {"order": ["C1", "C2"]} and good.error is None and good.tokens["totalTokenCount"] == 42
    )
    assert good.key == llm.user_key("secret_user_name") and "secret_user_name" not in good.key
    bad = llm.ask(StubClient(BadResponse("not JSON")), "open", who)
    assert bad.data is None and bad.error == "not JSON"
    with pytest.raises(ModelUnavailable):
        llm.ask(StubClient(ModelUnavailable("busy")), "open", who)
    client = StubClient({"shows": []})
    llm.ask(client, "open", who)
    assert client.calls[0][0] == llm.MODEL and client.calls[0][3] == 0.0


def test_answers_are_saved_and_read_back(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "RESULTS", tmp_path)
    llm.save_answer("dev", answer(data={"order": ["C1"]}, candidates=[5], key="a"))
    llm.save_answer("dev", answer(data={"order": []}, candidates=[5], key="a"))  # a later answer wins
    llm.save_answer("dev", answer(key="b", error="bad"))
    answers = llm.load_answers("dev", "ordering")
    assert set(answers) == {"a", "b"} and answers["a"].data == {"order": []}
    assert llm.load_answers("test", "ordering") == {}


class TinyCatalog:
    def __init__(self, ids):
        self.ids = np.array(ids)
        self.index = {a: i for i, a in enumerate(ids)}

    def __len__(self):
        return len(self.ids)


class Named:
    name = "secret_user_name"


def test_the_methods_score_what_the_model_said():
    catalog = TinyCatalog([101, 102, 103, 104])
    key = llm.user_key(Named.name)
    ordering = llm.LLMOrdering(
        catalog, {key: answer(data={"order": ["C2", "C1"]}, candidates=[103, 101, 104])}
    )
    assert ordering.scores(Named(), np.arange(4)).tolist() == [
        2.0,
        0.0,
        1.0,
        0.0,
    ]  # C2 is 101, C1 is 103; 104 left out
    open_ = llm.LLMOpen(
        catalog,
        {key: answer("open", {"shows": [{"title": "Made in Abyss", "year": 2017}]})},
        llm.TitleIndex({"made in abyss": [102]}, {102: 2017.0}, frozenset({102})),
    )
    assert open_.scores(Named(), np.arange(4)).tolist() == [0.0, float(llm.PICKS), 0.0, 0.0]
    nobody = llm.LLMOrdering(catalog, {})
    assert nobody.scores(Named(), np.arange(4)).tolist() == [0.0] * 4  # no answer: everything ties

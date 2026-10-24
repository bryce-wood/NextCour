import datetime as dt
import json

import httpx
import pytest

from nextcour.explain import gemini
from nextcour.explain.gemini import DailyQuotaUsed, GeminiClient, Limits, ModelUnavailable, UsageLog

# 2026-10-21 12:00 UTC is 05:00 in Los Angeles.
NOON_UTC = dt.datetime(2026, 10, 21, 12, tzinfo=dt.UTC).timestamp()
LIMITS = {"fast": Limits(per_minute=6, per_day=3)}


class FakeTime:
    def __init__(self, now: float = NOON_UTC) -> None:
        self.now = now
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(round(seconds, 2))
        self.now += seconds


def answer(data, thought=None):
    parts = [{"text": thought, "thought": True}] if thought else []
    parts.append({"text": json.dumps(data)})
    return httpx.Response(
        200,
        json={
            "candidates": [{"content": {"parts": parts}, "finishReason": "STOP"}],
            "usageMetadata": {"totalTokenCount": 10},
        },
    )


def limited(quota_id, delay="7s"):
    return httpx.Response(
        429,
        json={
            "error": {
                "code": 429,
                "status": "RESOURCE_EXHAUSTED",
                "details": [
                    {
                        "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                        "violations": [{"quotaId": quota_id}],
                    },
                    {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": delay},
                ],
            }
        },
    )


def make_client(responses, usage=None, now=NOON_UTC):
    fake = FakeTime(now)
    sent = []
    replies = iter(responses)

    def handler(request):
        sent.append((fake.now, request))
        reply = next(replies)
        if isinstance(reply, Exception):
            raise reply
        return reply

    client = GeminiClient(
        "test-key",
        usage or UsageLog(":memory:"),
        limits=LIMITS,
        transport=httpx.MockTransport(handler),
        clock=fake.clock,
        sleep=fake.sleep,
    )
    return client, fake, sent


def ask(client):
    return client.generate_json("fast", "system", "prompt", {"type": "OBJECT"})


def test_returns_the_json_answer_and_skips_thinking():
    client, _, sent = make_client([answer({"ok": True}, thought="hmm")])
    reply = ask(client)
    assert reply.data == {"ok": True}
    assert reply.usage == {"totalTokenCount": 10}
    request = sent[0][1]
    assert request.headers["x-goog-api-key"] == "test-key"
    assert "key=" not in str(request.url)  # the key never goes in the URL
    body = json.loads(request.content)
    assert body["generationConfig"]["responseMimeType"] == "application/json"


def test_spaces_requests_to_the_per_minute_limit():
    client, _, sent = make_client([answer({}), answer({})])
    ask(client)
    ask(client)
    assert sent[1][0] - sent[0][0] == pytest.approx(gemini.SPACING_MARGIN * 60 / 6)


def test_stops_before_the_daily_limit_and_counts_by_pacific_day():
    usage = UsageLog(":memory:")
    client, fake, sent = make_client([answer({})] * 4, usage)
    for _ in range(3):
        ask(client)
    with pytest.raises(DailyQuotaUsed):
        ask(client)
    assert len(sent) == 3
    assert client.remaining_today("fast") == 0
    fake.now += 19 * 3600  # past midnight in Los Angeles: a new day's quota
    ask(client)
    assert client.remaining_today("fast") == 2


def test_a_used_up_daily_quota_marks_the_model_done_for_the_day():
    usage = UsageLog(":memory:")
    client, _, sent = make_client([limited("GenerateRequestsPerDayPerProjectPerModel-FreeTier")], usage)
    with pytest.raises(DailyQuotaUsed):
        ask(client)
    assert client.remaining_today("fast") == 0
    with pytest.raises(DailyQuotaUsed):
        ask(client)  # refused locally, without another request
    assert len(sent) == 1


def test_a_per_minute_limit_waits_the_delay_google_gives():
    client, fake, sent = make_client(
        [limited("GenerateRequestsPerMinutePerProjectPerModel-FreeTier", "7s"), answer({"ok": 1})]
    )
    assert ask(client).data == {"ok": 1}
    assert 8.0 in fake.sleeps
    assert len(sent) == 2


def test_a_busy_model_is_retried_then_given_up_on():
    busy = httpx.Response(503, json={"error": {"message": "The model is overloaded."}})
    client, _, sent = make_client([busy, httpx.ConnectError("refused"), answer({"ok": 1})])
    assert ask(client).data == {"ok": 1}
    client, _, sent = make_client([busy, busy, busy])
    with pytest.raises(ModelUnavailable, match="busy"):
        ask(client)
    assert len(sent) == 3


def test_bad_answers_raise():
    not_json = httpx.Response(
        200, json={"candidates": [{"content": {"parts": [{"text": "nope"}]}, "finishReason": "STOP"}]}
    )
    cut_off = httpx.Response(
        200, json={"candidates": [{"content": {"parts": [{"text": "{"}]}, "finishReason": "MAX_TOKENS"}]}
    )
    for reply in (not_json, cut_off, httpx.Response(400, json={"error": {"message": "bad schema"}})):
        client, _, _ = make_client([reply])
        with pytest.raises(gemini.LLMError):
            ask(client)


def test_unknown_models_are_refused():
    client, _, sent = make_client([])
    with pytest.raises(ModelUnavailable):
        client.generate_json("no-such-model", "s", "p", {})
    assert sent == []


def test_a_timeout_goes_straight_to_the_backup():
    client, _, sent = make_client([httpx.ReadTimeout("slow"), answer({"ok": 1})])
    with pytest.raises(ModelUnavailable, match="didn't answer within"):
        ask(client)
    assert len(sent) == 1  # no second wait on the same slow model

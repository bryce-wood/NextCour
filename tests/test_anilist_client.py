import json

import httpx
import pytest

from nextcour.ingest import anilist
from nextcour.ingest.anilist import AniListClient, AniListError, NotFound, ResponseCache


class FakeTime:
    """A clock that only moves when something sleeps."""

    def __init__(self) -> None:
        self.now = 1_000_000.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(round(seconds, 3))
        self.now += seconds


def ok(data, limit=30, remaining=29, **headers):
    return httpx.Response(
        200,
        json={"data": data},
        headers={"X-RateLimit-Limit": str(limit), "X-RateLimit-Remaining": str(remaining), **headers},
    )


def make_client(responses, cache=None):
    """A client whose transport replays `responses` in order and records each request."""
    fake = FakeTime()
    sent = []
    replies = iter(responses)

    def handler(request):
        sent.append((fake.now, json.loads(request.content)))
        return next(replies)

    client = AniListClient(cache, transport=httpx.MockTransport(handler), clock=fake.clock, sleep=fake.sleep)
    return client, fake, sent


def test_returns_data_and_spaces_requests_by_the_reported_limit():
    client, _, sent = make_client([ok({"n": 1}, limit=30), ok({"n": 2}, limit=30)])
    assert client.query("{ a }", max_age=0) == {"n": 1}
    assert client.query("{ b }", max_age=0) == {"n": 2}
    assert sent[1][0] - sent[0][0] == pytest.approx(2.0)  # 60 s / 30 requests


def test_follows_a_raised_limit():
    client, _, sent = make_client([ok({}, limit=90), ok({}, limit=90)])
    client.query("{ a }", max_age=0)
    client.query("{ b }", max_age=0)
    assert sent[1][0] - sent[0][0] == pytest.approx(60 / 90)


def test_waits_for_retry_after_on_429_then_retries():
    limited = httpx.Response(429, headers={"Retry-After": "5", "X-RateLimit-Remaining": "0"})
    client, fake, sent = make_client([limited, ok({"n": 1})])
    assert client.query("{ a }", max_age=0) == {"n": 1}
    assert 6 in fake.sleeps
    assert len(sent) == 2


def test_waits_for_reset_when_no_requests_remain():
    client, _, sent = make_client([ok({}, remaining=0, **{"X-RateLimit-Reset": "1000060"}), ok({})])
    client.query("{ a }", max_age=0)
    client.query("{ b }", max_age=0)
    assert sent[1][0] == pytest.approx(1_000_060)


def test_backs_off_on_server_errors():
    client, fake, _ = make_client([httpx.Response(502), ok({"n": 1})])
    assert client.query("{ a }", max_age=0) == {"n": 1}
    assert fake.sleeps[0] == 2.0


def test_gives_up_after_repeated_failures():
    client, _, sent = make_client([httpx.Response(500)] * anilist.MAX_ATTEMPTS)
    with pytest.raises(AniListError, match="failed after"):
        client.query("{ a }", max_age=0)
    assert len(sent) == anilist.MAX_ATTEMPTS


def test_private_or_missing_user_raises_not_found():
    reply = httpx.Response(404, json={"errors": [{"message": "Private User", "status": 404}], "data": None})
    client, _, _ = make_client([reply])
    with pytest.raises(NotFound, match="Private User"):
        client.query("{ a }", max_age=0)


def test_cache_serves_fresh_answers_and_refetches_stale_ones():
    cache = ResponseCache(":memory:")
    client, fake, sent = make_client([ok({"n": 1}), ok({"n": 2})], cache=cache)
    assert client.query("{ a }", {"x": 1}, max_age=100) == {"n": 1}
    assert client.query("{  a }", {"x": 1}, max_age=100) == {"n": 1}  # whitespace doesn't matter
    assert len(sent) == 1
    fake.now += 101
    assert client.query("{ a }", {"x": 1}, max_age=100) == {"n": 2}
    assert len(sent) == 2


def test_errors_are_not_cached():
    cache = ResponseCache(":memory:")
    error = httpx.Response(400, json={"errors": [{"message": "Bad query", "status": 400}]})
    client, _, _ = make_client([error, ok({"n": 1})], cache=cache)
    with pytest.raises(AniListError):
        client.query("{ a }", max_age=100)
    assert client.query("{ a }", max_age=100) == {"n": 1}


def test_fetch_list_reads_every_chunk_and_skips_custom_lists():
    def collection(has_next, lists):
        return ok({"MediaListCollection": {"hasNextChunk": has_next, "lists": lists}})

    chunk1 = collection(
        True,
        [
            {"name": "Completed", "isCustomList": False, "entries": [{"id": 1}, {"id": 2}]},
            {"name": "Favs", "isCustomList": True, "entries": [{"id": 1}]},
        ],
    )
    chunk2 = collection(False, [{"name": "Planning", "isCustomList": False, "entries": [{"id": 3}]}])
    client, _, sent = make_client([chunk1, chunk2])
    entries = anilist.fetch_list(client, "someone")
    assert [e["id"] for e in entries] == [1, 2, 3]
    assert [body["variables"]["chunk"] for _, body in sent] == [1, 2]


def test_fetch_media_batches_fifty_ids_per_request():
    pages = [ok({"Page": {"media": [{"id": i}]}}) for i in range(3)]
    client, _, sent = make_client(pages)
    anilist.fetch_media_with_relations(client, range(120))
    assert [len(body["variables"]["ids"]) for _, body in sent] == [50, 50, 20]


def test_user_data_is_deleted_from_the_cache_after_an_hour():
    cache = ResponseCache(":memory:")
    lists = ok({"MediaListCollection": {"hasNextChunk": False, "lists": []}})
    show = ok({"Page": {"media": [{"id": 1}]}})
    client, fake, _ = make_client([lists, show, ok({"Page": {"media": []}})], cache=cache)
    anilist.fetch_list(client, "someone")
    anilist.fetch_media_with_relations(client, [1])
    count = "SELECT count(*) FROM anilist_cache"
    assert cache._db.execute(count).fetchone()[0] == 2
    fake.now += anilist.USER_LIST_MAX_AGE + 1
    client.query("{ other }", max_age=0)  # any query clears expired user data first
    remaining = cache._db.execute("SELECT query FROM anilist_cache").fetchall()
    assert [q for (q,) in remaining if q in anilist.USER_DATA_QUERIES] == []
    assert any(q == anilist.RELATIONS_QUERY for (q,) in remaining)  # show data stays

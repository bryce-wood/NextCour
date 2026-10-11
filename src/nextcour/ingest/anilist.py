"""AniList GraphQL client.

- Paces requests to the rate limit AniList reports in its response headers (X-RateLimit-Limit;
  30 requests a minute as of 2026-10), spacing them evenly so it never bursts.
- On a 429 it waits for Retry-After, then retries. On server or network errors it backs off.
- Caches successful responses in SQLite, keyed by query and variables. Each call says how old a
  cached answer may be: long for show data, short for user lists, which are only cached briefly
  (see Data rights in the design note).
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import httpx

log = logging.getLogger(__name__)

API_URL = "https://graphql.anilist.co"
USER_AGENT = "NextCour/0.1 (personal non-commercial project)"
DEFAULT_LIMIT_PER_MINUTE = 30
MAX_ATTEMPTS = 6

HOUR = 3600.0
DAY = 24 * HOUR
SHOW_MAX_AGE = 30 * DAY
USER_LIST_MAX_AGE = HOUR
MEDIA_PER_PAGE = 50  # AniList's page size limit


class AniListError(RuntimeError):
    """AniList answered with GraphQL errors, or kept failing after retries."""

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


class NotFound(AniListError):
    """The user or show doesn't exist, or the list is private."""


class ResponseCache:
    """Successful GraphQL responses, stored in SQLite."""

    def __init__(self, path: Path | str):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS anilist_cache ("
            " key TEXT PRIMARY KEY, query TEXT NOT NULL, variables TEXT NOT NULL,"
            " data TEXT NOT NULL, fetched_at REAL NOT NULL)"
        )
        self._db.commit()

    @staticmethod
    def key(query: str, variables: dict[str, Any]) -> str:
        normalized = " ".join(query.split())
        payload = json.dumps({"query": normalized, "variables": variables}, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()

    def get(self, key: str, max_age: float, now: float) -> Any | None:
        row = self._db.execute("SELECT data, fetched_at FROM anilist_cache WHERE key = ?", (key,)).fetchone()
        if row is None or now - row[1] > max_age:
            return None
        return json.loads(row[0])

    def put(self, key: str, query: str, variables: dict[str, Any], data: Any, now: float) -> None:
        self._db.execute(
            "INSERT OR REPLACE INTO anilist_cache VALUES (?, ?, ?, ?, ?)",
            (key, query, json.dumps(variables, sort_keys=True), json.dumps(data), now),
        )
        self._db.commit()

    def close(self) -> None:
        self._db.close()


class AniListClient:
    def __init__(
        self,
        cache: ResponseCache | None = None,
        *,
        transport: httpx.BaseTransport | None = None,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.cache = cache
        self._http = httpx.Client(
            transport=transport,
            timeout=30.0,
            headers={
                "User-Agent": USER_AGENT,
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        self._clock = clock
        self._sleep = sleep
        self.limit_per_minute = DEFAULT_LIMIT_PER_MINUTE
        self.remaining: int | None = None
        self._reset_at: float | None = None
        self._last_request: float | None = None
        self.requests_sent = 0

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> AniListClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def query(self, query: str, variables: dict[str, Any] | None = None, *, max_age: float) -> Any:
        """Run a GraphQL query and return its `data`, from the cache if it's fresh enough."""
        variables = variables or {}
        key = ResponseCache.key(query, variables)
        if self.cache is not None:
            cached = self.cache.get(key, max_age, self._clock())
            if cached is not None:
                return cached
        data = self._send(query, variables)
        if self.cache is not None:
            self.cache.put(key, query, variables, data, self._clock())
        return data

    def _wait_for_turn(self) -> None:
        now = self._clock()
        if self.remaining is not None and self.remaining <= 0:
            resume_at = self._reset_at or now + 60.0
        elif self._last_request is not None:
            resume_at = self._last_request + 60.0 / self.limit_per_minute
        else:
            return
        if resume_at > now:
            self._sleep(resume_at - now)

    def _read_limits(self, headers: httpx.Headers) -> None:
        if limit := _as_int(headers.get("X-RateLimit-Limit")):
            self.limit_per_minute = limit
        self.remaining = _as_int(headers.get("X-RateLimit-Remaining"))
        reset = _as_int(headers.get("X-RateLimit-Reset"))
        self._reset_at = float(reset) if reset else None

    def _send(self, query: str, variables: dict[str, Any]) -> Any:
        for attempt in range(1, MAX_ATTEMPTS + 1):
            self._wait_for_turn()
            self._last_request = self._clock()
            self.requests_sent += 1
            try:
                response = self._http.post(API_URL, json={"query": query, "variables": variables})
            except httpx.TransportError as error:
                self._back_off(attempt, f"network error ({error})")
                continue
            self._read_limits(response.headers)

            if response.status_code == 429:
                wait = (_as_int(response.headers.get("Retry-After")) or 60) + 1
                log.warning("AniList rate limit hit; waiting %d s", wait)
                self._sleep(wait)
                continue
            if response.status_code >= 500:
                self._back_off(attempt, f"server error {response.status_code}")
                continue

            try:
                body = response.json()
            except ValueError as error:
                raise AniListError(f"AniList sent a non-JSON reply ({response.status_code})") from error
            if body.get("errors"):
                first = body["errors"][0]
                status = first.get("status") or response.status_code
                error_type = NotFound if status == 404 else AniListError
                raise error_type(f"AniList error {status}: {first.get('message')}", status)
            if response.status_code != 200 or "data" not in body:
                raise AniListError(f"Unexpected AniList reply ({response.status_code})", response.status_code)
            return body["data"]
        raise AniListError(f"AniList request failed after {MAX_ATTEMPTS} attempts")

    def _back_off(self, attempt: int, reason: str) -> None:
        wait = 2.0**attempt
        log.warning("AniList %s; retrying in %.0f s", reason, wait)
        self._sleep(wait)


def _as_int(value: str | None) -> int | None:
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None


# --- Queries ---------------------------------------------------------------------------------

FUZZY_DATE = "{ year month day }"
MEDIA_FIELDS = f"""
    id idMal title {{ romaji english }} format episodes duration status
    startDate {FUZZY_DATE} season seasonYear
"""

USER_QUERY = """
query ($name: String) {
  User(name: $name) {
    id name
    mediaListOptions { scoreFormat }
    favourites { anime(perPage: 50) { pageInfo { hasNextPage } nodes { id } } }
  }
}
"""

LIST_QUERY = f"""
query ($name: String, $chunk: Int) {{
  MediaListCollection(userName: $name, type: ANIME, chunk: $chunk, perChunk: 500,
                      forceSingleCompletedList: true) {{
    hasNextChunk
    lists {{
      name isCustomList
      entries {{
        id status score(format: POINT_100) progress repeat
        startedAt {FUZZY_DATE} completedAt {FUZZY_DATE} createdAt updatedAt
        media {{ {MEDIA_FIELDS} }}
      }}
    }}
  }}
}}
"""

RELATIONS_QUERY = f"""
query ($ids: [Int]) {{
  Page(page: 1, perPage: {MEDIA_PER_PAGE}) {{
    media(id_in: $ids, type: ANIME) {{
      {MEDIA_FIELDS}
      relations {{ edges {{ relationType(version: 2) node {{ id type }} }} }}
    }}
  }}
}}
"""


def fetch_user(client: AniListClient, name: str) -> dict[str, Any]:
    """The user's ID, score format and favourite anime IDs."""
    user = client.query(USER_QUERY, {"name": name}, max_age=USER_LIST_MAX_AGE)["User"]
    favourites = user["favourites"]["anime"]
    if favourites["pageInfo"]["hasNextPage"]:
        raise AniListError("More than 50 favourite anime; add paging to fetch_user")
    return {
        "id": user["id"],
        "name": user["name"],
        "score_format": user["mediaListOptions"]["scoreFormat"],
        "favourite_anime": [node["id"] for node in favourites["nodes"]],
    }


def fetch_list(client: AniListClient, name: str) -> list[dict[str, Any]]:
    """Every entry on a user's anime list (custom lists skipped: they repeat entries)."""
    entries: list[dict[str, Any]] = []
    chunk = 1
    while True:
        page = client.query(LIST_QUERY, {"name": name, "chunk": chunk}, max_age=USER_LIST_MAX_AGE)
        collection = page["MediaListCollection"]
        for media_list in collection["lists"]:
            if not media_list["isCustomList"]:
                entries.extend(media_list["entries"])
        if not collection["hasNextChunk"]:
            return entries
        chunk += 1


def fetch_media_with_relations(client: AniListClient, ids: Iterable[int]) -> list[dict[str, Any]]:
    """Shows with their typed relations, fetched 50 per request."""
    unique = sorted(set(ids))
    media: list[dict[str, Any]] = []
    for start in range(0, len(unique), MEDIA_PER_PAGE):
        batch = unique[start : start + MEDIA_PER_PAGE]
        page = client.query(RELATIONS_QUERY, {"ids": batch}, max_age=SHOW_MAX_AGE)
        media.extend(page["Page"]["media"])
    return media

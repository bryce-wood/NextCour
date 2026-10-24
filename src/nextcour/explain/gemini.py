"""Gemini client for NextCour's LLM calls: the explanation writer now, the Phase 2 baseline later.

Plain HTTP with httpx, like the AniList client: it's one endpoint, and the error details (which
quota ran out, how long to wait) decide whether to wait, retry or move on to the backup model.

Free tier (design note, Deferred decisions): limits are per project and per model, and Google
sends no rate-limit headers, so the client keeps count itself:
- It spaces each model's requests to its per-minute limit.
- It counts each model's requests per Pacific day in SQLite and stops before the daily limit.
  The quota resets at midnight Pacific time.
- On a 429, a used-up daily quota marks the model as done for the day. Otherwise it waits the
  delay Google gives and tries again.
- On a busy model (500, 503 or 504) or a network error, it retries twice with a short backoff,
  then gives up so the caller can move to the backup model.
- A request that times out isn't retried: a model that slow is busy, and waiting out two more
  timeouts would cost half an hour before the backup got a turn (seen 2026-10-24).

Free-tier prompts may be used by Google and read by people, so only public data goes into them.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import re
import sqlite3
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import httpx

log = logging.getLogger(__name__)

API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
PACIFIC = ZoneInfo("America/Los_Angeles")
BUSY_RETRIES = 2
RATE_RETRIES = 3
SPACING_MARGIN = 1.1  # space requests 10% wider than the limit requires
# Busy free-tier models can take minutes: 3.8 Flash took 3.5 to 4.5 minutes to answer a
# one-line request on 2026-10-21, with or without thinking, and up to 5 minutes for ten shows.
TIMEOUT = 600.0


@dataclass(frozen=True)
class Limits:
    per_minute: int
    per_day: int


# Free-tier limits from my Google AI Studio, 2026-10-21. Models not listed have no free quota.
FREE_TIER: Mapping[str, Limits] = {
    "gemini-3.8-flash": Limits(5, 20),
    "gemini-3.6-flash": Limits(5, 20),
    "gemini-2.5-flash": Limits(5, 20),
    "gemini-3.5-flash-lite": Limits(15, 500),
    "gemini-3.1-flash-lite": Limits(15, 500),
    "gemini-2.5-flash-lite": Limits(10, 20),
}


class LLMError(RuntimeError):
    """The model couldn't give a usable answer; the caller moves on to its backup."""


class ModelUnavailable(LLMError):
    """Busy, rate-limited, or failing: try another model."""


class DailyQuotaUsed(ModelUnavailable):
    """This model's free requests for today (Pacific time) are used up."""


class BadResponse(LLMError):
    """The model answered, but not with the JSON asked for."""


def pacific_day(now: float) -> str:
    return dt.datetime.fromtimestamp(now, PACIFIC).date().isoformat()


class UsageLog:
    """Requests sent per model per Pacific day, so a restart doesn't forget them."""

    def __init__(self, path: Path | str):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS gemini_usage ("
            " model TEXT NOT NULL, day TEXT NOT NULL, requests INTEGER NOT NULL,"
            " PRIMARY KEY (model, day))"
        )
        self._db.commit()

    def count(self, model: str, day: str) -> int:
        with self._lock:
            row = self._db.execute(
                "SELECT requests FROM gemini_usage WHERE model = ? AND day = ?", (model, day)
            ).fetchone()
        return row[0] if row else 0

    def add(self, model: str, day: str) -> None:
        with self._lock:
            self._db.execute(
                "INSERT INTO gemini_usage VALUES (?, ?, 1)"
                " ON CONFLICT (model, day) DO UPDATE SET requests = requests + 1",
                (model, day),
            )
            self._db.commit()

    def use_up(self, model: str, day: str, limit: int) -> None:
        with self._lock:
            self._db.execute(
                "INSERT INTO gemini_usage VALUES (?, ?, ?)"
                " ON CONFLICT (model, day) DO UPDATE SET requests = max(requests, excluded.requests)",
                (model, day, limit),
            )
            self._db.commit()


@dataclass(frozen=True)
class Reply:
    data: Any  # the parsed JSON answer
    model: str
    usage: Mapping[str, Any]  # Google's token counts
    seconds: float = 0.0  # how long the answer took, retries included


class GeminiClient:
    def __init__(
        self,
        api_key: str,
        usage: UsageLog,
        *,
        limits: Mapping[str, Limits] = FREE_TIER,
        timeout: float = TIMEOUT,
        transport: httpx.BaseTransport | None = None,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self._http = httpx.Client(transport=transport, timeout=timeout, headers={"x-goog-api-key": api_key})
        self._timeout = timeout
        self.usage = usage
        self.limits = limits
        self._clock = clock
        self._sleep = sleep
        self._last: dict[str, float] = {}
        self.requests_sent = 0

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> GeminiClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def remaining_today(self, model: str) -> int:
        limit = self.limits[model].per_day
        return max(0, limit - self.usage.count(model, pacific_day(self._clock())))

    def generate_json(
        self, model: str, system: str, prompt: str, schema: Mapping[str, Any], temperature: float = 0.4
    ) -> Reply:
        """Ask `model` for JSON matching `schema`. Raises an LLMError if it can't."""
        if model not in self.limits:
            raise ModelUnavailable(f"{model} has no known free-tier limits")
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": schema,
                "temperature": temperature,
            },
        }
        busy = rated = 0
        started = self._clock()
        while True:
            response = self._send(model, body)
            if response is None or response.status_code in (500, 502, 503, 504):
                busy += 1
                if busy > BUSY_RETRIES:
                    raise ModelUnavailable(f"{model} is busy or unreachable")
                self._sleep(2.0 * 2**busy)
                continue
            if response.status_code == 429:
                daily, delay = _read_429(response)
                if daily:
                    self.usage.use_up(model, pacific_day(self._clock()), self.limits[model].per_day)
                    raise DailyQuotaUsed(f"{model}'s free requests for today are used up")
                rated += 1
                if rated > RATE_RETRIES:
                    raise ModelUnavailable(f"{model} keeps hitting its per-minute limit")
                log.warning("%s hit its per-minute limit; waiting %.0f s", model, delay)
                self._sleep(delay)
                continue
            if response.status_code != 200:
                raise ModelUnavailable(f"{model} answered {response.status_code}: {_message(response)}")
            reply = _parse(model, response)
            return Reply(reply.data, model, reply.usage, self._clock() - started)

    def _send(self, model: str, body: Mapping[str, Any]) -> httpx.Response | None:
        limits = self.limits[model]
        day = pacific_day(self._clock())
        if self.usage.count(model, day) >= limits.per_day:
            raise DailyQuotaUsed(f"{model}'s {limits.per_day} free requests for today are used up")
        last = self._last.get(model)
        if last is not None:
            wait = last + SPACING_MARGIN * 60.0 / limits.per_minute - self._clock()
            if wait > 0:
                self._sleep(wait)
        self._last[model] = self._clock()
        self.usage.add(model, pacific_day(self._clock()))
        self.requests_sent += 1
        try:
            return self._http.post(API_URL.format(model=model), json=body)
        except httpx.TimeoutException as error:
            raise ModelUnavailable(f"{model} didn't answer within {self._timeout:.0f} s") from error
        except httpx.TransportError as error:
            log.warning("%s: network error (%s)", model, error)
            return None


def _read_429(response: httpx.Response) -> tuple[bool, float]:
    """(is the daily quota used up, seconds to wait) from a 429's error details."""
    try:
        details = response.json()["error"].get("details") or []
    except (ValueError, KeyError, TypeError):
        return False, 60.0
    daily = any(
        "PerDay" in (violation.get("quotaId") or "")
        for detail in details
        for violation in detail.get("violations") or []
    )
    delay = 60.0
    for detail in details:
        if match := re.fullmatch(r"([\d.]+)s", str(detail.get("retryDelay", ""))):
            delay = float(match.group(1)) + 1.0
    return daily, delay


def _message(response: httpx.Response) -> str:
    try:
        return str(response.json()["error"]["message"])[:200]
    except (ValueError, KeyError, TypeError):
        return response.text[:200]


def _parse(model: str, response: httpx.Response) -> Reply:
    body = response.json()
    if reason := (body.get("promptFeedback") or {}).get("blockReason"):
        raise BadResponse(f"{model} blocked the prompt ({reason})")
    candidates = body.get("candidates") or []
    if not candidates:
        raise BadResponse(f"{model} returned no answer")
    candidate = candidates[0]
    if candidate.get("finishReason") not in (None, "STOP"):
        raise BadResponse(f"{model} stopped early ({candidate.get('finishReason')})")
    parts = (candidate.get("content") or {}).get("parts") or []
    text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
    try:
        data = json.loads(text)
    except ValueError as error:
        raise BadResponse(f"{model} returned text that isn't JSON") from error
    return Reply(data=data, model=model, usage=body.get("usageMetadata") or {})

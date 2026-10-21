"""The blind comparison of two writer models (design note, Deferred decisions).

Both models explain the same 20 shows, spread evenly down the ranked Planning list, from the same
evidence and prompt. Each model gets one try, with no backup, so a failure counts against it.
For every show, which model is "A" and which is "B" is random. The page shows the evidence and
both explanations; I vote, and only then see which model wrote what.

The comparison is saved as JSON under data/results/ (it holds my list, so it stays private), and
the votes go in SQLite.
"""

from __future__ import annotations

import datetime as dt
import json
import random
import sqlite3
import threading
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nextcour.core import paths
from nextcour.explain.gemini import GeminiClient
from nextcour.explain.writer import BATCH_SIZE, PROMPT_VERSION, Item, write_batch

SHOWS = 20
CHOICES = {
    "a": "A is better",
    "b": "B is better",
    "same": "About the same",
    "both_bad": "Both are bad",
}


def spread(n: int, k: int) -> list[int]:
    """k positions spread evenly over range(n), first and last included."""
    if k >= n:
        return list(range(n))
    if k == 1:
        return [0]
    return sorted({round(i * (n - 1) / (k - 1)) for i in range(k)})


def prepare(
    user: str,
    rows: Sequence[tuple[int, Item]],
    client: GeminiClient,
    models: tuple[str, str],
    seed: int | None = None,
    clock: Callable[[], float] = time.time,
) -> dict[str, Any]:
    """Have both models explain the same shows; returns the comparison, ready to save.

    rows: (rank, item) for each show to compare.
    """
    outputs: dict[str, dict[int, dict[str, Any]]] = {}
    seconds: dict[str, list[float]] = {}
    for model in models:
        outputs[model] = {}
        seconds[model] = []
        for start in range(0, len(rows), BATCH_SIZE):
            batch = [item for _, item in rows[start : start + BATCH_SIZE]]
            began = clock()
            results, _ = write_batch(batch, client, [model])
            seconds[model].append(round(clock() - began, 1))
            for item in batch:
                outputs[model][item.anilist_id] = results[item.anilist_id].as_dict()
    rng = random.Random(seed)
    items = []
    for rank, item in rows:
        a, b = rng.sample(list(models), 2)
        items.append(
            {
                "anilist_id": item.anilist_id,
                "title": item.title,
                "rank": rank,
                "evidence": [
                    {"id": e.id, "fact": e.fact, "effect": e.described_effect} for e in item.evidence
                ],
                "a_model": a,
                "b_model": b,
                "a": outputs[a][item.anilist_id],
                "b": outputs[b][item.anilist_id],
            }
        )
    created = dt.datetime.fromtimestamp(clock(), dt.UTC)
    return {
        "id": f"{created:%Y-%m-%dT%H%M%S}",
        "user": user,
        "created_at": created.isoformat(timespec="seconds"),
        "models": list(models),
        "seconds_per_batch": seconds,
        "batch_size": BATCH_SIZE,
        "prompt_version": PROMPT_VERSION,
        "items": items,
    }


def save(comparison: Mapping[str, Any], folder: Path | None = None) -> Path:
    folder = folder or paths.RESULTS
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"writer-comparison-{comparison['id']}.json"
    path.write_text(json.dumps(comparison, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def latest(folder: Path | None = None) -> dict[str, Any] | None:
    files = sorted((folder or paths.RESULTS).glob("writer-comparison-*.json"))
    return json.loads(files[-1].read_text(encoding="utf-8")) if files else None


class Votes:
    def __init__(self, path: Path | str, clock: Callable[[], float] = time.time):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        self._clock = clock
        with self._lock:
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS writer_votes ("
                " comparison TEXT NOT NULL, anilist_id INTEGER NOT NULL, choice TEXT NOT NULL,"
                " voted_at REAL NOT NULL, PRIMARY KEY (comparison, anilist_id))"
            )
            self._db.commit()

    def vote(self, comparison: str, anilist_id: int, choice: str) -> None:
        if choice not in CHOICES:
            raise ValueError(f"unknown choice {choice!r}")
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO writer_votes VALUES (?, ?, ?, ?)",
                (comparison, anilist_id, choice, self._clock()),
            )
            self._db.commit()

    def of(self, comparison: str) -> dict[int, str]:
        with self._lock:
            rows = self._db.execute(
                "SELECT anilist_id, choice FROM writer_votes WHERE comparison = ?", (comparison,)
            ).fetchall()
        return dict(rows)


@dataclass(frozen=True)
class Tally:
    model: str
    wins: int
    losses: int
    same: int
    both_bad: int
    templates: int  # shows where the model's answer failed and the template stood in
    dropped: int  # sentences the checker dropped

    @property
    def summary(self) -> str:
        return f"{self.wins} wins, {self.losses} losses, {self.same} about the same, {self.both_bad} both bad"


def tally(comparison: Mapping[str, Any], votes: Mapping[int, str]) -> list[Tally]:
    counts: dict[str, Counter[str]] = {m: Counter() for m in comparison["models"]}
    for item in comparison["items"]:
        for side in ("a", "b"):
            model = item[f"{side}_model"]
            counts[model]["templates"] += item[side]["writer"] == "template"
            counts[model]["dropped"] += len(item[side]["dropped"])
        choice = votes.get(item["anilist_id"])
        if choice in ("a", "b"):
            winner = item[f"{choice}_model"]
            loser = item["b_model" if choice == "a" else "a_model"]
            counts[winner]["wins"] += 1
            counts[loser]["losses"] += 1
        elif choice in ("same", "both_bad"):
            for model in (item["a_model"], item["b_model"]):
                counts[model][choice] += 1
    return [
        Tally(m, c["wins"], c["losses"], c["same"], c["both_bad"], c["templates"], c["dropped"])
        for m, c in counts.items()
    ]

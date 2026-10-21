"""What a user tells NextCour, kept in SQLite (design note, Signals and Evaluation):

- "Not interested": a hard filter on that user's lists, until undone.
- Picks: a show chosen from a list, with its rank, prediction and explanation at that moment.
  These are the "real use" layer: later, the list says whether I finished it and liked it.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Pick:
    user: str
    anilist_id: int
    title: str
    rank: int
    predicted: float
    explanation: str
    picked_at: float


class FeedbackStore:
    def __init__(self, path: Path | str, clock: Callable[[], float] = time.time):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        self._clock = clock
        with self._lock:
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS not_interested ("
                " user TEXT NOT NULL, anilist_id INTEGER NOT NULL, marked_at REAL NOT NULL,"
                " PRIMARY KEY (user, anilist_id))"
            )
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS picks ("
                " user TEXT NOT NULL, anilist_id INTEGER NOT NULL, title TEXT NOT NULL,"
                " rank INTEGER NOT NULL, predicted REAL NOT NULL, explanation TEXT NOT NULL,"
                " picked_at REAL NOT NULL)"
            )
            self._db.commit()

    def not_interested(self, user: str) -> set[int]:
        with self._lock:
            rows = self._db.execute(
                "SELECT anilist_id FROM not_interested WHERE user = ?", (user.lower(),)
            ).fetchall()
        return {row[0] for row in rows}

    def mark_not_interested(self, user: str, anilist_id: int) -> None:
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO not_interested VALUES (?, ?, ?)",
                (user.lower(), anilist_id, self._clock()),
            )
            self._db.commit()

    def undo_not_interested(self, user: str, anilist_id: int) -> None:
        with self._lock:
            self._db.execute(
                "DELETE FROM not_interested WHERE user = ? AND anilist_id = ?", (user.lower(), anilist_id)
            )
            self._db.commit()

    def add_pick(self, pick: Pick) -> None:
        with self._lock:
            self._db.execute(
                "INSERT INTO picks VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    pick.user.lower(),
                    pick.anilist_id,
                    pick.title,
                    pick.rank,
                    pick.predicted,
                    pick.explanation,
                    pick.picked_at,
                ),
            )
            self._db.commit()

    def picks(self, user: str) -> list[Pick]:
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM picks WHERE user = ? ORDER BY picked_at DESC", (user.lower(),)
            ).fetchall()
        return [Pick(*row) for row in rows]

    def now(self) -> float:
        return self._clock()

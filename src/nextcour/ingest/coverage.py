"""Date coverage (roadmap Phase 0): the share of completed entries with start and finish dates.

It decides how much the speed and airing features can matter: they only exist where dates do.
Only full dates (year, month and day) count, since speed needs days.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb


@dataclass(frozen=True)
class DateCoverage:
    label: str
    completed: int
    with_start: int
    with_finish: int
    with_both: int
    users: int | None = None  # users with at least one completed entry
    users_any: int | None = None  # ... of whom at least one completed entry has both dates
    users_most: int | None = None  # ... of whom at least half of completed entries have both

    def lines(self) -> list[str]:
        def share(n: int, of: int) -> str:
            return f"{n:,} ({n / of:.1%})" if of else "0"

        out = [
            f"{self.label}: {self.completed:,} completed entries",
            f"  with a start date: {share(self.with_start, self.completed)}",
            f"  with a finish date: {share(self.with_finish, self.completed)}",
            f"  with both: {share(self.with_both, self.completed)}",
        ]
        if self.users is not None:
            out += [
                f"  users with a completed entry: {self.users:,}",
                f"    with at least one fully dated: {share(self.users_any or 0, self.users)}",
                f"    with at least half fully dated: {share(self.users_most or 0, self.users)}",
            ]
        return out


def anilist_list(raw_entries: list[dict[str, Any]], label: str) -> DateCoverage:
    def full(date: dict[str, int | None]) -> bool:
        return bool(date and date.get("year") and date.get("month") and date.get("day"))

    completed = [e for e in raw_entries if e["status"] == "COMPLETED"]
    return DateCoverage(
        label=label,
        completed=len(completed),
        with_start=sum(full(e["startedAt"]) for e in completed),
        with_finish=sum(full(e["completedAt"]) for e in completed),
        with_both=sum(full(e["startedAt"]) and full(e["completedAt"]) for e in completed),
    )


MAL_SQL = """
WITH completed AS (
    SELECT
        username,
        TRY_CAST(CAST(my_start_date AS VARCHAR) AS DATE) IS NOT NULL AS has_start,
        TRY_CAST(CAST(my_finish_date AS VARCHAR) AS DATE) IS NOT NULL AS has_finish
    FROM read_parquet(?)
    WHERE CAST(my_status AS VARCHAR) IN ('2', 'Completed')
),
per_user AS (
    SELECT username, avg(CAST(has_start AND has_finish AS INTEGER)) AS dated_share
    FROM completed GROUP BY username
)
SELECT
    (SELECT count(*) FROM completed),
    (SELECT count(*) FILTER (has_start) FROM completed),
    (SELECT count(*) FILTER (has_finish) FROM completed),
    (SELECT count(*) FILTER (has_start AND has_finish) FROM completed),
    (SELECT count(*) FROM per_user),
    (SELECT count(*) FILTER (dated_share > 0) FROM per_user),
    (SELECT count(*) FILTER (dated_share >= 0.5) FROM per_user)
"""


def mal_list_entries(con: duckdb.DuckDBPyConnection, parquet: Path, label: str) -> DateCoverage:
    """Coverage in a MAL 2018 list-entry table (UserAnimeList), where a missing date is 0000-00-00."""
    row = con.execute(MAL_SQL, [parquet.as_posix()]).fetchone()
    assert row is not None
    return DateCoverage(label, *row)

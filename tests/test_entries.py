import datetime as dt

from helpers import entry

from nextcour.core.entries import from_anilist, fuzzy_date


def test_partial_dates_read_cautiously():
    assert fuzzy_date({"year": 2024, "month": 2, "day": None}, latest=False) == dt.date(2024, 2, 1)
    assert fuzzy_date({"year": 2024, "month": 2, "day": None}, latest=True) == dt.date(2024, 2, 29)
    assert fuzzy_date({"year": 2023, "month": None, "day": None}, latest=True) == dt.date(2023, 12, 31)
    assert fuzzy_date({"year": None, "month": None, "day": None}, latest=False) is None


def test_watched_from():
    assert entry(1, started="2022-01-01", completed="2022-02-01").watched_from == dt.date(2022, 1, 1)
    assert entry(1, completed="2022-02-01").watched_from == dt.date(2022, 2, 1)
    assert entry(1).watched_from is None
    # A rewatch may have overwritten the dates.
    assert entry(1, repeat=1, started="2022-01-01").watched_from is None
    # No watch takes over a year, so a start that far back isn't trusted.
    assert entry(1, started="2022-09-25", completed="2025-05-01").watched_from == dt.date(2025, 5, 1)


def test_from_anilist():
    raw = {
        "status": "COMPLETED",
        "score": 84,
        "progress": 22,
        "repeat": 0,
        "startedAt": {"year": 2025, "month": 6, "day": None},
        "completedAt": {"year": None, "month": None, "day": None},
        "createdAt": 1680000000,
        "media": {
            "id": 20665,
            "idMal": 23273,
            "title": {"romaji": "Shigatsu wa Kimi no Uso", "english": "Your lie in April"},
            "format": "TV",
            "startDate": {"year": 2014, "month": 10, "day": 10},
        },
    }
    e = from_anilist(raw)
    assert (e.anilist_id, e.mal_id, e.title) == (20665, 23273, "Your lie in April")
    assert e.started == dt.date(2025, 6, 1)
    assert e.completed is None
    assert e.aired == dt.date(2014, 10, 10)
    assert e.created.tzinfo is not None

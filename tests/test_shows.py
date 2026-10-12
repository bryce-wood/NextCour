from nextcour.core.shows import Show, from_anilist


def media(**overrides):
    base = {
        "id": 1,
        "title": {"romaji": "Romaji", "english": None},
        "format": "TV",
        "startDate": {"year": 2020, "month": 4, "day": None},
        "averageScore": 81,
        "meanScore": 82,
        "popularity": 5000,
        "genres": ["Drama"],
        "tags": [
            {"id": 10, "name": "Time Skip", "rank": 70, "isMediaSpoiler": False, "isGeneralSpoiler": False},
            {"id": 11, "name": "Twist", "rank": 90, "isMediaSpoiler": True, "isGeneralSpoiler": False},
        ],
        "studios": {
            "edges": [
                {"isMain": True, "node": {"id": 7, "name": "Main Studio"}},
                {"isMain": False, "node": {"id": 8, "name": "Producer"}},
            ]
        },
        "staff": {
            "edges": [
                {"role": "Director", "node": {"id": 100, "name": {"full": "A Director"}}},
                {"role": "Series Composition", "node": {"id": 100, "name": {"full": "A Director"}}},
                {"role": "Music (eps 1-12)", "node": {"id": 101, "name": {"full": "A Composer"}}},
                {"role": "Episode Director (ep 3)", "node": {"id": 102, "name": {"full": "Someone"}}},
            ]
        },
        "recommendations": {
            "nodes": [
                {"rating": 40, "mediaRecommendation": {"id": 2}},
                {"rating": 0, "mediaRecommendation": {"id": 3}},
                {"rating": 9, "mediaRecommendation": None},
            ]
        },
    }
    return base | overrides


def test_features_skip_spoilers_minor_studios_and_episode_roles():
    show = from_anilist(media())
    keys = {f.key: f for f in show.features}
    assert set(keys) == {"tag:10", "genre:Drama", "studio:7", "staff:100", "staff:101"}
    assert keys["tag:10"].weight == 0.7
    assert keys["staff:100"].role == "Director, Series Composition"
    assert keys["staff:101"].role == "Music"


def test_basic_fields_and_links():
    show = from_anilist(media())
    assert (show.title, show.average_score, show.popularity) == ("Romaji", 81, 5000)
    assert show.recommendations == {2: 40}
    other = Show(2, "Other", "TV", None, 70, 10)
    assert show.link_votes(other) == 40
    assert other.link_votes(show) == 40  # a link reads the same from either side
    assert from_anilist(media(averageScore=None)).average_score == 82

import math

import pytest
from helpers import entry

from nextcour.profile import builder


def test_franchise_weight_grows_slowly_and_starts_at_one():
    assert builder.franchise_weight(4.8) == pytest.approx(1)  # one cour
    assert builder.franchise_weight(2.0) == pytest.approx(1)  # a movie still counts once
    assert builder.franchise_weight(8 * 4.8) == pytest.approx(math.log2(9))


def test_watch_hours():
    assert builder.watch_hours(entry(1, episodes=12, duration=24)) == pytest.approx(4.8)
    assert builder.watch_hours(entry(1, repeat=2)) == pytest.approx(4.8)  # rewatches add nothing
    dropped = entry(1, status="DROPPED", progress=3, episodes=12, duration=24)
    assert builder.watch_hours(dropped) == pytest.approx(1.2)


def test_a_franchise_shares_its_weight_by_watch_time():
    history = [
        entry(1, episodes=12, duration=24),  # 4.8 hours
        entry(2, episodes=24, duration=24),  # 9.6 hours
        entry(3, episodes=1, duration=120, format="MOVIE"),  # 2 hours, its own franchise
    ]
    profile = builder.build(history, {1: 10, 2: 10, 3: 30})
    weight = {p.entry.anilist_id: p.weight for p in profile.entries}
    total = builder.franchise_weight(14.4)
    assert weight[1] == pytest.approx(total / 3)
    assert weight[2] == pytest.approx(total * 2 / 3)
    assert weight[3] == pytest.approx(1)


def test_music_videos_and_unlabeled_entries_are_left_out():
    history = [
        entry(1),
        entry(2, format="MUSIC"),
        entry(3, score=0, status="CURRENT"),
    ]
    profile = builder.build(history, {})
    assert [p.entry.anilist_id for p in profile.entries] == [1]
    assert profile.scored == 1

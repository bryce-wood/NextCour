"""The "works for me" test (roadmap Phase 1; design note, Evaluation).

Does a method put the frozen shows I enjoyed above the ones I enjoyed less, more often than sorting
by average score does? The measure is ordering accuracy over pairs of frozen shows.

Settled on 2026-10-11, before the first result:
- Each pair is judged from one shared history: what I'd finished by the earlier of the two shows'
  as-of dates (and never either show, or a later entry of either franchise). Both scores in a pair
  then come from the same profile, so comparing them is fair for any method, including the Phase 5
  ranker, whose scores only mean something within one history. It also matches the MAL
  evaluation, where all of a user's targets are scored from one history.
- AniList's crowd numbers (average score, recommendations) are today's. A pair is "clean" when
  both shows had been airing for at least six months by the pair's date, so today's numbers are
  close to what was known then. The headline uses clean pairs; all pairs are reported too, as
  using later crowd information.
- Labels are enjoyment v1, from the frozen snapshot. Pairs with equal labels say nothing about
  order and are skipped. A tie in a method's scores counts half.
- The main and sequel sets are reported separately, never mixed.
- Confidence intervals come from a bootstrap over shows: resample the shows with replacement,
  keep every pair between the resampled shows, and recompute.

Every history is built with guardrail.history_for() and checked with check_history().
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import itertools
import json
import random
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from nextcour.core import guardrail, paths
from nextcour.core.entries import ListEntry, from_anilist
from nextcour.core.guardrail import FrozenShow
from nextcour.core.shows import Show
from nextcour.profile import builder, enjoyment

CLEAN_AFTER = dt.timedelta(days=182)  # about six months of airing
BOOTSTRAP_DRAWS = 2000

# A method turns a profile and the show details into a function that scores a show.
Method = Callable[[builder.Profile, Mapping[int, Show]], Callable[[Show], float]]


@dataclass(frozen=True)
class Pair:
    first: FrozenShow  # the one started first (or the same day)
    second: FrozenShow
    date: dt.date  # the shared as-of date: the earlier of the two
    history_size: int
    clean: bool
    labels: tuple[float, float]
    scores: Mapping[str, tuple[float, float]]  # method -> (first, second)

    def outcome(self, method: str) -> float:
        """1 if the method orders the pair like the labels, 0.5 for a tie, 0 otherwise."""
        a, b = self.scores[method]
        if a == b:
            return 0.5
        return 1.0 if (a > b) == (self.labels[0] > self.labels[1]) else 0.0


def is_clean(show: FrozenShow, date: dt.date) -> bool:
    return show.aired is not None and show.aired <= date - CLEAN_AFTER


def shared_history(
    a: FrozenShow,
    b: FrozenShow,
    entries: Sequence[ListEntry],
    franchise_of: Mapping[int, int],
    joined: dt.datetime,
) -> tuple[dt.date, list[ListEntry]]:
    """The history both shows of a pair are scored from, checked by the guardrail."""
    date = min(a.as_of, b.as_of)
    a, b = dataclasses.replace(a, as_of=date), dataclasses.replace(b, as_of=date)
    allowed_for_b = {e.anilist_id for e in guardrail.history_for(b, entries, franchise_of, joined)}
    history = [
        e for e in guardrail.history_for(a, entries, franchise_of, joined) if e.anilist_id in allowed_for_b
    ]
    guardrail.check_history(a, history, franchise_of, joined)
    guardrail.check_history(b, history, franchise_of, joined)
    return date, history


def judge_pairs(
    shows_to_test: Iterable[FrozenShow],
    entries: Sequence[ListEntry],
    franchise_of: Mapping[int, int],
    joined: dt.datetime,
    details: Mapping[int, Show],
    methods: Mapping[str, Method],
    favourites: Iterable[int] = (),
) -> list[Pair]:
    """Every pair of the given frozen shows whose labels differ, scored by every method."""
    favourites = list(favourites)
    labels = enjoyment.labels(entries, favourites, franchise_of)  # the truth: the whole snapshot
    ordered = sorted(shows_to_test, key=lambda s: (s.as_of, s.anilist_id))
    profiles: dict[tuple[int, ...], dict[str, Callable[[Show], float]]] = {}
    pairs = []
    for a, b in itertools.combinations(ordered, 2):
        label_a, label_b = labels.get(a.anilist_id), labels.get(b.anilist_id)
        if label_a is None or label_b is None or label_a == label_b:
            continue
        date, history = shared_history(a, b, entries, franchise_of, joined)
        key = tuple(sorted(e.anilist_id for e in history))
        if key not in profiles:
            profile = builder.build(history, franchise_of, favourites)
            profiles[key] = {name: make(profile, details) for name, make in methods.items()}
        scorers = profiles[key]
        pairs.append(
            Pair(
                first=a,
                second=b,
                date=date,
                history_size=len(history),
                clean=is_clean(a, date) and is_clean(b, date),
                labels=(label_a, label_b),
                scores={
                    name: (score(details[a.anilist_id]), score(details[b.anilist_id]))
                    for name, score in scorers.items()
                },
            )
        )
    return pairs


@dataclass(frozen=True)
class Summary:
    pairs: int
    shows: int
    correct: dict[str, float]  # method -> pairs ordered right (ties count half)
    disagreements: int  # pairs the two methods order differently
    disagreements_won: dict[str, int]  # method -> how many of those it got right
    difference: float  # accuracy of the first method minus the second's
    interval: tuple[float, float] | None  # 95% bootstrap interval for the difference

    def accuracy(self, method: str) -> float:
        return self.correct[method] / self.pairs if self.pairs else float("nan")


def summarize(pairs: Sequence[Pair], methods: tuple[str, str], seed: int = 0) -> Summary:
    first, second = methods
    correct = {m: sum(p.outcome(m) for p in pairs) for m in methods}
    disagree = [p for p in pairs if p.outcome(first) != p.outcome(second)]
    n = len(pairs)
    return Summary(
        pairs=n,
        shows=len({s.anilist_id for p in pairs for s in (p.first, p.second)}),
        correct=correct,
        disagreements=len(disagree),
        disagreements_won={m: sum(1 for p in disagree if p.outcome(m) == 1.0) for m in methods},
        difference=(correct[first] - correct[second]) / n if n else float("nan"),
        interval=bootstrap_interval(pairs, methods, seed=seed) if n else None,
    )


def bootstrap_interval(
    pairs: Sequence[Pair], methods: tuple[str, str], draws: int = BOOTSTRAP_DRAWS, seed: int = 0
) -> tuple[float, float]:
    """95% interval for the accuracy difference, resampling shows (not pairs), since pairs that
    share a show aren't independent."""
    first, second = methods
    shows = sorted({s.anilist_id for p in pairs for s in (p.first, p.second)})
    outcomes = [
        (p.first.anilist_id, p.second.anilist_id, p.outcome(first) - p.outcome(second)) for p in pairs
    ]
    rng = random.Random(seed)
    differences = []
    for _ in range(draws):
        count = dict.fromkeys(shows, 0)
        for show in rng.choices(shows, k=len(shows)):
            count[show] += 1
        weight = total = 0.0
        for a, b, diff in outcomes:
            w = count[a] * count[b]
            weight += w
            total += w * diff
        if weight:
            differences.append(total / weight)
    differences.sort()
    return differences[int(0.025 * len(differences))], differences[int(0.975 * len(differences)) - 1]


def load_snapshot(frozen: guardrail.FrozenSet) -> tuple[list[ListEntry], list[int], dict[str, Any]]:
    """The list as it stood at the freeze, which holds the labels: entries, favourites, raw JSON.
    Fails if the snapshot changed."""
    path = paths.DATA / frozen.snapshot_file
    snapshot = json.loads(path.read_text(encoding="utf-8"))
    if guardrail.canonical_sha256(snapshot) != frozen.snapshot_sha256:
        raise guardrail.GuardrailViolation(f"{path} doesn't match the checksum in the frozen set")
    entries = [from_anilist(e) for e in snapshot["entries"]]
    return entries, snapshot["user"]["favourite_anime"], snapshot


def pair_record(pair: Pair) -> dict[str, Any]:
    return {
        "first": {
            "title": pair.first.title,
            "anilist_id": pair.first.anilist_id,
            "as_of": str(pair.first.as_of),
        },
        "second": {
            "title": pair.second.title,
            "anilist_id": pair.second.anilist_id,
            "as_of": str(pair.second.as_of),
        },
        "date": str(pair.date),
        "history_size": pair.history_size,
        "clean": pair.clean,
        "labels": [round(x, 4) for x in pair.labels],
        "scores": {m: [round(x, 4) for x in s] for m, s in pair.scores.items()},
    }

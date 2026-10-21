"""Command line: `uv run nextcour <command>`. Each step is one command."""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import logging
import subprocess
import sys

import duckdb

from nextcour.core import manifest, paths, shows
from nextcour.core.entries import from_anilist
from nextcour.core.shows import Show
from nextcour.ingest import anilist, catalog, coverage, franchises, mal


def _client() -> anilist.AniListClient:
    return anilist.AniListClient(anilist.ResponseCache(paths.APP_DB))


def _watched_ids(client: anilist.AniListClient, user: str) -> list[int]:
    return [e["media"]["id"] for e in anilist.fetch_list(client, user) if e["status"] != "PLANNING"]


def cmd_catalog(args: argparse.Namespace) -> None:
    path = catalog.download(args.release)
    con = duckdb.connect()
    summary = catalog.build(con, args.release)
    print("\n".join(summary.lines()))
    last_update = json.loads(path.read_text(encoding="utf-8"))["lastUpdate"]
    manifest.update(
        "anime-offline-database",
        {
            "title": "anime-offline-database, by manami-project",
            "source": catalog.SOURCE,
            "license": catalog.LICENSE,
            "release": args.release,
            "last_update": last_update,
            "files": [
                {
                    "file": path.relative_to(paths.RAW).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": manifest.file_sha256(path),
                    "rows": summary.shows,
                }
            ],
        },
    )


def cmd_franchises(args: argparse.Namespace) -> None:
    con = duckdb.connect()
    catalog.load(con, args.release)
    with _client() as client:
        must = _watched_ids(client, args.user) if args.user else []
        franchises.build(con, client, must_type=must, release=args.release)
        print(f"AniList requests: {client.requests_sent}")
    for line in franchises.summary_lines(con):
        print(line)


def cmd_convert_mal(args: argparse.Namespace) -> None:
    names = args.datasets or mal.available()
    if not names:
        sys.exit(f"No datasets downloaded. Unzip them into {paths.RAW / 'mal-2018'} and so on.")
    con = duckdb.connect()
    for name in names:
        mal.convert(name, con)


def cmd_data_report(args: argparse.Namespace) -> None:
    from nextcour.core import guardrail

    con = duckdb.connect()
    catalog.load(con, args.release)
    frozen = guardrail.frozen_set_if_present()
    my_names = [frozen.anilist_user_name, *frozen.mal_user_names] if frozen else []
    if not my_names:
        print("No frozen test set here, so there are no user names of mine to look for.")
    for name, table in mal.LIST_TABLES.items():
        if not (mal.parquet_dir(name) / f"{table.file}.parquet").exists():
            print(f"{name}: not converted yet")
            continue
        print("\n".join(mal.list_report(con, name, my_names).lines()))


def cmd_coverage(args: argparse.Namespace) -> None:
    table = mal.parquet_dir("mal-2018") / "UserAnimeList.parquet"
    if table.exists():
        con = duckdb.connect()
        print("\n".join(coverage.mal_list_entries(con, table, "MAL 2018 (UserAnimeList)").lines()))
    else:
        print(f"MAL 2018 not converted yet ({table} missing); run convert-mal first.")
    if args.user:
        with _client() as client:
            entries = anilist.fetch_list(client, args.user)
        print("\n".join(coverage.anilist_list(entries, f"AniList list of {args.user}").lines()))


def cmd_freeze(args: argparse.Namespace) -> None:
    from nextcour.evaluation import frozen_set

    con = duckdb.connect()
    catalog.load(con, args.release)
    franchise_of = franchises.lookup(con, args.release)
    with _client() as client:
        user = anilist.fetch_user(client, args.user)
        entries = anilist.fetch_list(client, args.user)
    frozen = frozen_set.freeze(user, entries, franchise_of, args.release, dt.date.today())
    for which in ("main", "sequel"):
        print(f"{which}: {len(frozen[which])} shows")
        for show in frozen[which]:
            print(f"  {show['as_of']}  {show['title']}")
    print(f"Wrote {paths.FROZEN_TEST_SET} and the snapshot {paths.DATA / frozen['snapshot']['file']}")


def _show_details(client: anilist.AniListClient, ids: list[int]) -> tuple[dict[int, Show], list[dict]]:
    media = anilist.fetch_show_details(client, ids)
    return {m["id"]: shows.from_anilist(m) for m in media}, media


def _commit() -> dict[str, object]:
    def git(*args: str) -> str:
        result = subprocess.run(["git", *args], cwd=paths.ROOT, capture_output=True, text=True, check=False)
        return result.stdout.strip()

    return {"commit": git("rev-parse", "HEAD"), "uncommitted_changes": bool(git("status", "--porcelain"))}


def cmd_rank(args: argparse.Namespace) -> None:
    from nextcour.candidates import planning as planning_candidates
    from nextcour.models.scorer_v0 import ScorerV0
    from nextcour.profile import builder

    con = duckdb.connect()
    catalog.load(con, args.release)
    franchise_of = franchises.lookup(con, args.release)
    with _client() as client:
        user = anilist.fetch_user(client, args.user)
        entries = [from_anilist(e) for e in anilist.fetch_list(client, args.user)]
        details, _ = _show_details(client, [e.anilist_id for e in entries])
    history = [e for e in entries if e.status != "PLANNING"]
    planning = [
        details[e.anilist_id]
        for e in entries
        if e.status == "PLANNING" and e.format not in builder.EXCLUDED_FORMATS and e.anilist_id in details
    ]
    statuses = {e.anilist_id: e.status for e in entries}
    ready, held = planning_candidates.by_watch_order(planning, statuses, dt.date.today())
    scorer = ScorerV0(builder.build(history, franchise_of, user["favourite_anime"]), details)
    scores = sorted((scorer.score(s) for s in ready), key=lambda s: s.predicted, reverse=True)
    by_average = sorted(scores, key=lambda s: s.crowd_prior, reverse=True)
    average_rank = {s.anilist_id: i for i, s in enumerate(by_average, 1)}
    print(f"{args.user}'s Planning list: {len(scores)} shows ready to watch, ranked by scorer v0")
    print("  rank  predicted = prior + taste + links   (rank by average score)")
    for i, s in enumerate(scores[: args.top], 1):
        print(
            f"  {i:>4}  {s.predicted:.3f} = {s.crowd_prior:.3f} {0.5 * s.taste:+.3f} {0.5 * s.links:+.3f}"
            f"   ({average_rank[s.anilist_id]:>3})  {details[s.anilist_id].title}"
        )
        for c in s.contributions[1 : 1 + args.reasons]:
            print(f"          {c.value:+.3f}  {c.kind}: {c.label}")
    if held:
        print(f"\nHeld back, {len(held)} shows: not out yet, or waiting on an earlier entry:")
        for h in sorted(held, key=lambda h: h.show.title):
            print(f"  {h.show.title}  <- {'; '.join(h.waiting_on)}")


def cmd_works_for_me(args: argparse.Namespace) -> None:
    from nextcour.core import guardrail
    from nextcour.evaluation import works_for_me as wfm
    from nextcour.models import baselines, scorer_v0
    from nextcour.models.scorer_v0 import ScorerV0

    frozen = guardrail.load_frozen_set()
    entries, favourites, _ = wfm.load_snapshot(frozen)
    con = duckdb.connect()
    catalog.load(con, frozen.catalog_release)
    franchise_of = franchises.lookup(con, frozen.catalog_release)
    with _client() as client:
        details, media = _show_details(client, [e.anilist_id for e in entries])
    missing = [s.title for s in frozen.shows if s.anilist_id not in details]
    if missing:
        sys.exit(f"No AniList details for {', '.join(missing)}")

    methods: dict[str, wfm.Method] = {
        "scorer_v0": lambda profile, known: ScorerV0(profile, known).predict,
        "average_score": baselines.average_score,
    }
    compared = ("scorer_v0", "average_score")
    today = dt.date.today().isoformat()
    run_at = dt.datetime.now(dt.UTC)
    paths.RESULTS.mkdir(parents=True, exist_ok=True)
    details_file = paths.RESULTS / f"show-details-{today}.json"
    details_file.write_text(json.dumps(media, ensure_ascii=False), encoding="utf-8")

    result: dict[str, object] = {
        "run_at": run_at.isoformat(timespec="seconds"),
        **_commit(),
        "catalog_release": frozen.catalog_release,
        "snapshot_sha256": frozen.snapshot_sha256,
        "show_details": details_file.name,
        "scorer_v0": {
            name: getattr(scorer_v0, name)
            for name in ("TASTE_WEIGHT", "LINK_WEIGHT", "TASTE_SHRINK", "LINK_SHRINK", "LINK_VOTES_HALF")
        },
        "sets": {},
    }
    for which in ("main", "sequel"):
        pairs = wfm.judge_pairs(
            frozen.of(which), entries, franchise_of, frozen.joined, details, methods, favourites
        )
        print(f"\n{which.capitalize()} set: {len(frozen.of(which))} shows")
        record: dict[str, object] = {"pairs": [wfm.pair_record(p) for p in pairs]}
        for name, subset in (("clean pairs", [p for p in pairs if p.clean]), ("all pairs", pairs)):
            summary = wfm.summarize(subset, compared)
            record[name] = dataclasses.asdict(summary)
            print(f"  {name}: {summary.pairs} pairs over {summary.shows} shows")
            for method in compared:
                print(
                    f"    {method:<14} {summary.correct[method]:g} right"
                    f"  (ordering accuracy {summary.accuracy(method):.3f})"
                )
            low, high = summary.interval or (float("nan"), float("nan"))
            print(f"    difference     {summary.difference:+.3f}  (95% interval {low:+.3f} to {high:+.3f})")
            won = summary.opposite_won
            print(
                f"    ordered in opposite directions: {summary.opposite} pairs;"
                f" scorer v0 right on {won['scorer_v0']}, average score on {won['average_score']}"
            )
            right, wrong = summary.tie_breaks["scorer_v0"]
            print(
                f"    average score tied: {summary.ties['average_score']} pairs;"
                f" scorer v0 broke {right} right and {wrong} wrong"
            )
        result["sets"][which] = record  # type: ignore[index]
    out = paths.RESULTS / f"works-for-me-{run_at:%Y-%m-%dT%H%M}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nWrote {out}")


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # show titles aren't all ASCII
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(
        prog="nextcour", description="NextCour: explainable anime recommendations"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser(
        "catalog", help="download anime-offline-database; build the catalog and AniList-MAL mapping"
    )
    p.add_argument("--release", default=catalog.RELEASE)
    p.set_defaults(run=cmd_catalog)

    p = sub.add_parser("franchises", help="group shows into franchises (needs the catalog)")
    p.add_argument(
        "--user", help="AniList user whose watched shows' franchises are always checked by typed relations"
    )
    p.add_argument("--release", default=catalog.RELEASE)
    p.set_defaults(run=cmd_franchises)

    p = sub.add_parser("convert-mal", help="convert downloaded MAL datasets to Parquet and record them")
    p.add_argument(
        "datasets", nargs="*", help=f"any of {', '.join(mal.DATASETS)}; default: every downloaded one"
    )
    p.set_defaults(run=cmd_convert_mal)

    p = sub.add_parser(
        "data-report", help="each MAL dataset's size and AniList mapping; checks for my accounts"
    )
    p.add_argument("--release", default=catalog.RELEASE)
    p.set_defaults(run=cmd_data_report)

    p = sub.add_parser("coverage", help="measure date coverage in MAL 2018 and in a user's list")
    p.add_argument("--user", help="AniList user")
    p.set_defaults(run=cmd_coverage)

    p = sub.add_parser("freeze", help="freeze my 'works for me' test set (runs once, then never again)")
    p.add_argument("--user", required=True, help="my AniList user name")
    p.add_argument("--release", default=catalog.RELEASE)
    p.set_defaults(run=cmd_freeze)

    p = sub.add_parser("rank", help="rank a user's Planning list with scorer v0")
    p.add_argument("--user", required=True, help="AniList user")
    p.add_argument("--top", type=int, default=30, help="how many shows to print")
    p.add_argument("--reasons", type=int, default=3, help="how many pieces of evidence to print per show")
    p.add_argument("--release", default=catalog.RELEASE)
    p.set_defaults(run=cmd_rank)

    p = sub.add_parser(
        "works-for-me", help="the 'works for me' test: scorer v0 vs. average score on my frozen shows"
    )
    p.set_defaults(run=cmd_works_for_me)

    args = parser.parse_args(argv)
    args.run(args)


if __name__ == "__main__":
    main()

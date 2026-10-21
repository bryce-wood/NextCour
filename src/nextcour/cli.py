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


def _planner(release: str = catalog.RELEASE):
    from nextcour.app.feedback import FeedbackStore
    from nextcour.app.service import Planner

    con = duckdb.connect()
    catalog.load(con, release)
    return Planner(franchises.lookup(con, release), _client, FeedbackStore(paths.APP_DB))


def _writer(models: list[str] | None = None, timeout: float | None = None):
    """The explanation writer: Gemini's free tier if GEMINI_API_KEY is set, templates otherwise."""
    from nextcour.core import settings
    from nextcour.explain import gemini
    from nextcour.explain.writer import DEFAULT_MODELS, ExplanationStore, Writer

    key = settings.gemini_api_key()
    client = (
        gemini.GeminiClient(key, gemini.UsageLog(paths.APP_DB), timeout=timeout or gemini.TIMEOUT)
        if key
        else None
    )
    if client is None:
        print("No GEMINI_API_KEY in .env: explanations will use templates.")
    return Writer(ExplanationStore(paths.APP_DB), client, tuple(models or DEFAULT_MODELS))


def cmd_rank(args: argparse.Namespace) -> None:
    ranking = _planner(args.release).ranking(args.user)
    print(
        f"{ranking.user}'s Planning list: {len(ranking.rows)} shows ready to watch, ranked by scorer v0"
        f" (profile from {ranking.history_size} shows)"
    )
    print("  rank  predicted = prior + taste + links   (rank by average score)")
    for row in ranking.rows[: args.top]:
        s = row.score
        parts = f"{s.predicted:.3f} = {s.crowd_prior:.3f} {0.5 * s.taste:+.3f} {0.5 * s.links:+.3f}"
        print(f"  {row.rank:>4}  {parts}   ({row.average_rank:>3})  {row.show.title}")
        for c in s.contributions[1 : 1 + args.reasons]:
            print(f"          {c.value:+.3f}  {c.kind}: {c.label}")
    if ranking.held:
        print(f"\nHeld back, {len(ranking.held)} shows: not out yet, or waiting on an earlier entry:")
        for h in sorted(ranking.held, key=lambda h: h.show.title):
            print(f"  {h.show.title}  <- {'; '.join(h.waiting_on)}")
    if ranking.hidden:
        print(f"\nNot interested: {', '.join(s.title for s in ranking.hidden)}")


def cmd_explain(args: argparse.Namespace) -> None:
    from nextcour.app import service

    ranking = _planner(args.release).ranking(args.user)
    rows = ranking.rows[: args.top]
    writer = _writer(args.model)
    if args.rewrite:
        writer.store.forget_user(ranking.user)
    written = len(service.write_missing(writer, ranking, rows)) if not args.cached_only else 0
    found = service.explanations(writer, ranking)
    client = writer.client
    print(f"{ranking.user}: {len(rows)} shows; {written} explanations written now")
    if client is not None:
        for model in writer.models:
            print(f"  {model}: {client.remaining_today(model)} free requests left today (Pacific time)")
    for row in rows:
        explanation = found[row.show.anilist_id]
        print(f"\n{row.rank:>3}. {row.show.title}  [{explanation.writer}]  {row.watch_time or ''}")
        print(f"     {explanation.text}")
        for reason in explanation.dropped:
            print(f"     (checker dropped a sentence that {reason})")
        if explanation.note and explanation.writer == "template":
            print(f"     (template: {explanation.note})")
        if args.evidence:
            for e in row.item.evidence:
                print(f"       {e.id}. [{e.described_effect}] {e.fact}")


def cmd_compare_writers(args: argparse.Namespace) -> None:
    from nextcour.explain import compare

    writer = _writer(timeout=600.0)
    if writer.client is None:
        sys.exit("The comparison needs GEMINI_API_KEY in .env.")
    ranking = _planner(args.release).ranking(args.user)
    positions = compare.spread(len(ranking.rows), args.shows)
    rows = [(ranking.rows[i].rank, ranking.rows[i].item) for i in positions]
    models = (args.models[0], args.models[1])
    for model in models:
        print(f"{model}: {writer.client.remaining_today(model)} free requests left today")
    comparison = compare.prepare(ranking.user, rows, writer.client, models, seed=args.seed)
    path = compare.save(comparison)
    for t in compare.tally(comparison, {}):
        print(
            f"{t.model}: template stood in for {t.templates} shows; the checker dropped {t.dropped} sentences"
        )
    print(f"Wrote {path}. Vote at http://127.0.0.1:8000/compare after `uv run nextcour serve`.")


def cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn

    from nextcour.app.web import create_app
    from nextcour.explain import compare

    planner = _planner(args.release)
    app = create_app(planner, _writer(), planner.feedback, compare.Votes(paths.APP_DB))
    print(f"NextCour is at http://{args.host}:{args.port}/ (Ctrl+C stops it)")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


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

    p = sub.add_parser("explain", help="write (and cache) explanations for a user's ranked Planning list")
    p.add_argument("--user", required=True, help="AniList user")
    p.add_argument("--top", type=int, default=1000, help="only the top N shows")
    p.add_argument("--model", action="append", help="Gemini model to use, in order (repeatable)")
    p.add_argument("--evidence", action="store_true", help="print each show's evidence too")
    p.add_argument("--cached-only", action="store_true", help="don't call any model")
    p.add_argument("--rewrite", action="store_true", help="forget this user's cached explanations first")
    p.add_argument("--release", default=catalog.RELEASE)
    p.set_defaults(run=cmd_explain)

    p = sub.add_parser("serve", help="run the web page at http://127.0.0.1:8000/")
    p.add_argument("--host", default="127.0.0.1", help="127.0.0.1 keeps it on this computer")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--release", default=catalog.RELEASE)
    p.set_defaults(run=cmd_serve)

    p = sub.add_parser("compare-writers", help="both writer models explain the same shows, for a blind vote")
    p.add_argument("--user", required=True, help="AniList user")
    p.add_argument("--shows", type=int, default=20, help="how many shows, spread down the list")
    p.add_argument(
        "--models", nargs=2, default=["gemini-3.8-flash", "gemini-3.5-flash-lite"], help="the two models"
    )
    p.add_argument("--seed", type=int, help="fix which model is A and which is B")
    p.add_argument("--release", default=catalog.RELEASE)
    p.set_defaults(run=cmd_compare_writers)

    p = sub.add_parser(
        "works-for-me", help="the 'works for me' test: scorer v0 vs. average score on my frozen shows"
    )
    p.set_defaults(run=cmd_works_for_me)

    args = parser.parse_args(argv)
    args.run(args)


if __name__ == "__main__":
    main()

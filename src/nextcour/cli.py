"""Command line: `uv run nextcour <command>`. Each Phase 0 step is one command."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sys

import duckdb

from nextcour.core import manifest, paths
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

    args = parser.parse_args(argv)
    args.run(args)


if __name__ == "__main__":
    main()

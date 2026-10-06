"""Command line for acquisition and compilation.

`paa site` builds the browser. The data commands are the ones that matter.
"""


import argparse
import json
from pathlib import Path

from paa.config import REPORTS, ensure_dirs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="paa")
    sub = parser.add_subparsers(dest="command", required=True)

    vaalit = sub.add_parser("vaalit", help="Aggregate official candidate files already downloaded, or download them")
    vaalit.add_argument("--year", type=int, action="append")

    sub.add_parser("yle", help="Fetch named 2011/2023 campaign texts and note the anonymous 2019 file")
    sub.add_parser("members", help="Fetch the parliament member register")
    votes = sub.add_parser("votes", help="Fetch Finnish plenary-vote rows")
    votes.add_argument("--year", type=int, action="append")
    ballots = sub.add_parser("ballots", help="Fetch per-MP ballots for the given years")
    ballots.add_argument("--year", type=int, action="append")
    sub.add_parser("repair-ballots", help="Repair short vote slices from bounded official-source retries")
    initiatives = sub.add_parser("initiatives", help="Fetch the official legislative-initiative register")
    initiatives.add_argument("--year", type=int, action="append", help="Registry year; repeat for several years")
    initiatives.add_argument("--matter", action="append", help="Fetch an exact matter (e.g. LA 72/2017 vp); repeatable")
    initiatives.add_argument(
        "--partition-identifiers",
        action="store_true",
        help="Partition registry requests by the first LA identifier digit to avoid a broad-page repeat",
    )
    initiatives.add_argument(
        "--refresh",
        action="store_true",
        help="Fetch registry pages again and preserve prior raw checkpoints by their content hash",
    )
    initiatives.add_argument("--db", type=Path, help="Database to receive normalized official objects")
    questions = sub.add_parser("questions", help="Acquire written questions and separate government answers")
    questions.add_argument("--year", type=int, action="append")
    questions.add_argument("--refresh", action="store_true")
    questions.add_argument("--db", type=Path)
    speeches = sub.add_parser("speeches", help="Acquire source-attributed parliamentary speeches by year")
    speeches.add_argument("--year", type=int, action="append")
    speeches.add_argument("--refresh", action="store_true")
    speeches.add_argument("--db", type=Path)
    llm = sub.add_parser("llm", help="Run or evaluate resumable local-model source extraction")
    llm.add_argument("llm_args", nargs=argparse.REMAINDER)
    evidence = sub.add_parser("import-evidence", help="Import reviewed official-object evidence without replacing campaign data")
    evidence.add_argument("--file", type=Path, required=True, help="JSONL evidence bundle")
    evidence.add_argument("--db", type=Path, help="Database to receive reviewed records")
    compiler = sub.add_parser("compile", help="Link identities, compile statements, write exports")
    compiler.add_argument("--llm-run", type=Path, help="Use a finished, source-matched local-model candidate run")
    compiler.add_argument("--db", type=Path)
    compiler.add_argument("--output", type=Path)
    compiler.add_argument("--exports", type=Path)
    compiler.add_argument("--reports", type=Path)
    check = sub.add_parser("check", help="Fail if a known invariant is broken")
    check.add_argument("--db", type=Path, help="Database to inspect")
    check.add_argument("--slice", action="store_true", help="Check a declared frozen slice, not full-corpus counts")
    site = sub.add_parser("site", help="Write the browser into dist/browser")
    site.add_argument("--db", type=Path)
    site.add_argument("--output", type=Path)
    site.add_argument("--structure-results", type=Path, help="Expose the source-bound model comparison as proposals")
    site.add_argument("--research-packet", type=Path, action="append",
                      help="Include a research-only inquiry JSON export; repeat for multiple runs")
    sub.add_parser("acquire", help="Run vaalit, yle, members and vote metadata")
    frozen = sub.add_parser("frozen", help="Build the offline source -> trace -> browser acceptance fixture")
    frozen.add_argument("--root", type=Path, default=Path("dist/frozen"), help="Build root for data, reports and dist")
    frozen.add_argument("--fixture", type=Path, help="Override the bundled frozen JSONL source slice")
    frozen.add_argument("--overwrite", action="store_true", help="Replace outputs below --root")

    args = parser.parse_args(argv)
    if args.command == "llm":
        from paa.llm_run import main as llm_main

        return llm_main(args.llm_args)
    if args.command == "frozen":
        from paa.frozen import DEFAULT_FIXTURE, build_frozen

        result = build_frozen(
            args.root,
            fixture_path=args.fixture or DEFAULT_FIXTURE,
            overwrite=args.overwrite,
        )
        benchmark = result["benchmark"]
        summary = {**result, "benchmark": {
            "report_path": str(Path(result["report_dir"]) / "relation_benchmark.json"),
            "evaluation_scope": benchmark["evaluation_scope"],
            "initiative_top_k_recall": benchmark["initiative_benchmark"]["top_k_recall"],
            "admitted_relation_precision": benchmark["admitted_relation_precision"],
            "review_abstention_rate": benchmark["abstention_rate"],
        }}
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    ensure_dirs()
    if args.command == "vaalit":
        from paa.acquire_vaalit import acquire

        print(json.dumps(acquire(args.year), ensure_ascii=False, indent=2))
    elif args.command == "yle":
        from paa.acquire_yle import acquire_2011, acquire_2019_anonymous, acquire_2023, probe_2015

        result = {"2011": acquire_2011(), "2023": acquire_2023(), "2019_anonymous": acquire_2019_anonymous(), "2015_probe": probe_2015()}
        (REPORTS / "yle_acquire.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "members":
        from paa.acquire_eduskunta import acquire_members

        print(json.dumps(acquire_members(), ensure_ascii=False, indent=2))
    elif args.command == "votes":
        from paa.acquire_eduskunta import acquire_votes

        print(json.dumps(acquire_votes(args.year), ensure_ascii=False, indent=2))
    elif args.command == "ballots":
        from paa.acquire_eduskunta import acquire_ballots

        print(json.dumps(acquire_ballots(args.year), ensure_ascii=False, indent=2))
    elif args.command == "repair-ballots":
        from paa.acquire_eduskunta import repair_short_ballots

        print(json.dumps(repair_short_ballots(), ensure_ascii=False, indent=2))
    elif args.command == "initiatives":
        from paa.initiative_ledger import acquire_registry, import_result
        from paa.store import connect

        if args.matter:
            from paa.acquire_initiatives import acquire_initiatives

            result = acquire_initiatives(args.matter)
        else:
            result = acquire_registry(
                args.year,
                partition_identifiers=args.partition_identifiers,
                refresh=args.refresh,
            )
        conn = connect(args.db)
        try:
            imported = import_result(conn, result)
            conn.commit()
        finally:
            conn.close()
        print(json.dumps({"imported": imported, "coverage": result.get("coverage")}, ensure_ascii=False, indent=2))
    elif args.command == "questions":
        from paa.question_ledger import acquire_questions, import_result
        from paa.store import connect

        result = acquire_questions(args.year, refresh=args.refresh)
        conn = connect(args.db)
        try:
            imported = import_result(conn, result)
            conn.commit()
        finally:
            conn.close()
        print(json.dumps({"imported": imported, "coverage": result.get("coverage")}, ensure_ascii=False, indent=2))
    elif args.command == "speeches":
        from paa.speech_ledger import acquire_full_term, import_speeches
        from paa.store import connect

        result = acquire_full_term(years=args.year or (2023, 2024, 2025, 2026), refresh=args.refresh)
        conn = connect(args.db)
        try:
            imported = import_speeches(conn, result)
            conn.commit()
        finally:
            conn.close()
        print(json.dumps({"imported": imported, "coverage": result.get("coverage")}, ensure_ascii=False, indent=2))
    elif args.command == "import-evidence":
        from paa.import_evidence import import_records
        from paa.store import connect

        conn = connect(args.db)
        try:
            result = import_records(conn, args.file)
        finally:
            conn.close()
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "compile":
        from paa.pipeline import compile_all

        print(json.dumps(compile_all(db_path=args.db, output_dir=args.output, export_dir=args.exports,
                                    report_dir=args.reports, llm_run=args.llm_run), ensure_ascii=False, indent=2))
    elif args.command == "check":
        from paa.check import main as check_main

        return check_main(args.db, full_corpus=not args.slice)
    elif args.command == "site":
        from paa.site import build_from_db

        build_from_db(args.db, args.output, research_packets=args.research_packet,
                      structure_results=args.structure_results)
    elif args.command == "acquire":
        from paa.acquire_eduskunta import acquire_members, acquire_votes
        from paa.acquire_vaalit import acquire
        from paa.acquire_yle import acquire_2011, acquire_2019_anonymous, acquire_2023

        result = {
            "vaalit": acquire(None),
            "yle_2011": acquire_2011(),
            "yle_2023": acquire_2023(),
            "yle_2019_anonymous": acquire_2019_anonymous(),
            "members": acquire_members(),
            "votes": acquire_votes(None),
        }
        (REPORTS / "acquire.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

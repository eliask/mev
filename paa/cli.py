"""Command line for acquisition and compilation.

`paa site` builds the browser. The data commands are the ones that matter.
"""


import argparse
import json
from pathlib import Path

from paa.config import FIXTURE_DIR, REPORTS, ensure_dirs


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
    release = sub.add_parser("release", help="Build a fresh selected public-beta site and static archive")
    release.add_argument("--selection", type=Path, default=FIXTURE_DIR / "public_beta_selection.json")
    release.add_argument("--output", type=Path, required=True)
    release.add_argument("--source-revision", required=True, help="Code revision label; actual producer hashes are retained")
    release.add_argument("--base-url", default="", help="Authorized public hosting URL; omit for a local preview")
    purpose = sub.add_parser(
        "purpose-demo",
        help="Build the hospital/Oulaskangas purpose demonstration from bundled evidence",
    )
    purpose.add_argument("--output", type=Path, required=True)
    purpose.add_argument("--source-revision", required=True, help="Code revision label; actual producer hashes are retained")
    purpose.add_argument("--selection", type=Path, default=FIXTURE_DIR / "purpose_demo_selection.json")
    purpose.add_argument("--corpus", type=Path, default=FIXTURE_DIR / "ballot_disposition_corpus.json")
    purpose.add_argument("--base-url", default="", help="Authorized public hosting URL; omit for a local preview")
    matter = sub.add_parser(
        "matter",
        help="Query exact matter-to-actor interventions, or list lexical matter candidates",
    )
    matter_query = matter.add_mutually_exclusive_group(required=True)
    matter_query.add_argument("--matter", dest="matter_id", help="Exact official matter identifier")
    matter_query.add_argument("--search", help="Text search for candidate matters; never an exact relation")
    matter.add_argument("--actor-id", help="Exact actor identifier for an actor+matter query")
    matter.add_argument("--person-id", help="Exact source person identifier for an actor+matter query")
    matter.add_argument("--actor-name", help="Unique exact normalized display name for an actor+matter query")
    matter.add_argument("--limit", type=int, default=50, help="Candidate search result limit (1..500)")
    matter.add_argument("--db", type=Path, help="Database to inspect (read-only)")
    matter.add_argument("--output", type=Path, help="Write the JSON source packet here")
    matter.add_argument("--html", dest="html_output", type=Path, help="Write a standalone readable HTML packet here")
    ask = sub.add_parser(
        "source-question",
        help="Answer a question from a self-contained source bank with integer pointer evidence",
    )
    ask.add_argument("--questions", type=Path, required=True, help="Question bank JSON")
    ask.add_argument("--sources", type=Path, required=True, help="Candidate-safe source bank JSON")
    ask_task = ask.add_mutually_exclusive_group(required=True)
    ask_task.add_argument("--task", dest="task_id", help="Single task id (e.g. D07)")
    ask_task.add_argument("--all", dest="all_tasks", action="store_true", help="Run every task in the bank")
    ask_task.add_argument("--split", help="Run every task with this split label (e.g. dev)")
    ask.add_argument(
        "--provider",
        choices=("local", "muse"),
        default="local",
        help="Inference route: local llama.cpp (default) or selected Muse via OpenCode Go",
    )
    ask.add_argument(
        "--key-file",
        type=Path,
        help="Literal OPENCODE_API_KEY assignment file for --provider muse (or OPENCODE_KEY_FILE)",
    )
    ask.add_argument(
        "--db",
        type=Path,
        help="Optional read-only DB; ledger projection is retained separately, not same-source synthesis input",
    )
    ask.add_argument(
        "--include-db-ledger",
        action="store_true",
        help="Attach deterministic ledger projection for separate rendering (still excluded from model synthesis)",
    )
    ask.add_argument("--output", type=Path, required=True, help="Directory for task packets and SUMMARY.json")
    ask.add_argument(
        "--context-limit-tokens",
        type=int,
        default=None,
        help="Optional explicit Muse planning ceiling (not advertised model capacity); unsupported for local",
    )
    research = sub.add_parser("research-import", help="Bind an external research answer to retained source pointers")
    research.add_argument("--questions", type=Path, required=True)
    research.add_argument("--sources", type=Path, required=True)
    research.add_argument("--submission", type=Path, required=True)
    research.add_argument("--output", type=Path, required=True, help="New candidate answer JSON; never overwrites")
    revise = sub.add_parser(
        "source-revision",
        help="One bounded source-bound revision of a candidate using a source-backed critique",
    )
    revise.add_argument("--candidate", type=Path, required=True, help="Original candidate answer JSON")
    revise.add_argument("--critique", type=Path, required=True, help="Source-backed critique JSON")
    revise.add_argument("--questions", type=Path, required=True, help="Question bank JSON")
    revise.add_argument("--sources", type=Path, required=True, help="Source bank JSON")
    revise.add_argument("--output", type=Path, required=True, help="Unique output directory")
    revise.add_argument("--provider", choices=("local", "muse"), default="muse")
    revise.add_argument("--key-file", type=Path, help="OPENCODE_API_KEY assignment file for muse")
    revise.add_argument(
        "--context-limit-tokens",
        type=int,
        default=None,
        help="Optional explicit Muse planning ceiling for the single revision request",
    )
    enrich = sub.add_parser(
        "enrich-ballots",
        help="Bind portable current-official ballot witnesses into matter JSON snapshots (no DB writes)",
    )
    enrich.add_argument("--matter", type=Path, action="append", required=True, help="Matter JSON; repeatable")
    enrich.add_argument("--witness-bank", type=Path, required=True, help="portable_ballot_witness_bank.json")
    enrich.add_argument(
        "--comparison",
        type=Path,
        required=True,
        help="per_event_version_comparison.json covering every bank event",
    )
    enrich.add_argument("--output", type=Path, required=True, help="Unique output directory")
    sourcebank_matter = sub.add_parser(
        "matter-sourcebank",
        help="Adapt KKV/committee sourcebanks into immutable matter JSON + public HTML (no DB writes)",
    )
    sourcebank_matter.add_argument("--kkv-bank", type=Path, help="kkv_response_sourcebank.json")
    sourcebank_matter.add_argument(
        "--committee-bank", type=Path, help="committee_role_sourcebank.json"
    )
    sourcebank_matter.add_argument(
        "--matter-id",
        help="Optional matter id when adapting a multi-matter committee bank",
    )
    sourcebank_matter.add_argument("--output", type=Path, required=True, help="Unique output directory")
    answer_pdf = sub.add_parser(
        "export-answer-attachment",
        help="Fetch official KKV/KK answer PDF via AttachmentGroup metadata and export normalized artifact",
    )
    answer_pdf.add_argument("--attachment-group-id", required=True, help="Official AttachmentGroupId")
    answer_pdf.add_argument("--declared-md5", help="Optional Vaski HashValue MD5 control")
    answer_pdf.add_argument("--record-json", type=Path, help="Optional parsed question/answer record JSON")
    answer_pdf.add_argument("--output", type=Path, required=True, help="Unique output directory")
    route = sub.add_parser(
        "route",
        help="Build a deterministic cross-matter documentary route/component site from a source bank",
    )
    route.add_argument("--questions", type=Path, required=True, help="Question bank JSON")
    route.add_argument("--sources", type=Path, required=True, help="Self-contained source bank JSON")
    route_task = route.add_mutually_exclusive_group(required=True)
    route_task.add_argument("--task", dest="task_id", help="Single task id")
    route_task.add_argument("--all", dest="all_tasks", action="store_true", help="Run every task in the bank")
    route_task.add_argument("--split", help="Run every task with this split label")
    route.add_argument("--as-of", dest="as_of", help="Optional ISO as-of date for applicability propositions")
    route.add_argument(
        "--answer",
        type=Path,
        help="Optional source-question packet JSON or directory; bound by task/question/source hashes",
    )
    route.add_argument("--output", type=Path, required=True, help="New artifact directory (must be empty unless --overwrite)")
    route.add_argument("--overwrite", action="store_true", help="Replace files inside the selected output directory")
    bootstrap = sub.add_parser(
        "bootstrap-site",
        help="Build a selectable static reader from matter packets and source-question candidates",
    )
    bootstrap.add_argument("--output", type=Path, required=True, help="New artifact directory (does not overwrite older dist trees)")
    bootstrap.add_argument("--matter", type=Path, action="append", default=[], help="Matter ledger JSON packet; repeatable")
    bootstrap.add_argument("--question", type=Path, action="append", default=[], help="Source-question result JSON; repeatable")
    bootstrap.add_argument("--oulaskangas-group", type=Path, action="append", default=[], help="Group-response candidate JSON")
    bootstrap.add_argument("--oulaskangas-cost", type=Path, action="append", default=[], help="Cost-scope candidate JSON")
    bootstrap.add_argument("--oulaskangas-monitoring", type=Path, action="append", default=[], help="Monitoring candidate JSON")
    bootstrap.add_argument("--editorial-packet", type=Path, help="Source-bound editorial rows JSON (no renderer constants)")
    bootstrap.add_argument(
        "--editorial-overview",
        type=Path,
        help="Optional source-bound overview JSON (question/answer/takeaway/publisher)",
    )
    bootstrap.add_argument("--source-bank", type=Path, action="append", default=[], help="Source bank for title/URL metadata")
    bootstrap.add_argument(
        "--route-dir",
        type=Path,
        action="append",
        default=[],
        help="Directory of validated route JSON(+HTML)/manifest to copy into the artifact; repeatable",
    )
    bootstrap.add_argument(
        "--review",
        type=Path,
        action="append",
        default=[],
        help="Source-bound AI review receipt (paa.source_answer_review.v1); tied to exact candidate/bank hashes",
    )
    bootstrap.add_argument(
        "--reviewed-candidate",
        action="append",
        default=[],
        metavar="TASK_ID=PATH",
        help="Exact reviewed candidate file for TASK_ID (byte-bound to review candidate_sha256)",
    )
    bootstrap.add_argument("--title", default="Poliittinen muisti")
    bootstrap.add_argument(
        "--journey",
        type=Path,
        action="append",
        default=[],
        help="Composed question-journey packet JSON; repeatable (projection review must already be bound if claimed)",
    )
    integrated = sub.add_parser(
        "integrated-reader",
        help="Build a portable question-driven reader from a relative-path inputs manifest",
    )
    integrated.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Portable inputs manifest (paa.integrated_reader_manifest.v1 or BS02 carrier + journey_builds)",
    )
    integrated.add_argument(
        "--inputs",
        type=Path,
        help="Optional inputs root for relative paths (default: manifest directory)",
    )
    integrated.add_argument("--output", type=Path, required=True, help="New empty artifact directory")
    journey = sub.add_parser(
        "journey",
        help="Compose a question journey projection from a route packet and explicit mapping",
    )
    journey.add_argument("--route-packet", type=Path, required=True, help="Validated route packet JSON")
    journey.add_argument("--mapping", type=Path, required=True, help="Explicit journey mapping JSON")
    journey.add_argument("--answer", type=Path, help="Optional source-question answer packet for claim binding")
    journey.add_argument(
        "--source-answer-review",
        type=Path,
        help="Optional paa.source_answer_review.v1 receipt (answer review; not projection review)",
    )
    journey.add_argument(
        "--projection-review",
        type=Path,
        help="Optional paa.journey_projection_review.v1 receipt binding the displayed graph",
    )
    journey.add_argument("--output", type=Path, required=True, help="New journey site directory")
    journey.add_argument("--overwrite", action="store_true", help="Replace files inside the selected output directory")

    args = parser.parse_args(argv)
    if args.command == "release":
        from paa.release import build_release

        result = build_release(args.selection, args.output, source_revision=args.source_revision,
                               base_url=args.base_url)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "purpose-demo":
        from paa.purpose_demo import build_purpose_demo

        result = build_purpose_demo(
            args.output,
            source_revision=args.source_revision,
            selection_path=args.selection,
            corpus_path=args.corpus,
            base_url=args.base_url,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "matter":
        from paa.matter_ledger import matter_cli

        packet = matter_cli(
            db_path=args.db,
            matter_id=args.matter_id,
            search=args.search,
            actor_id=args.actor_id,
            person_id=args.person_id,
            actor_name=args.actor_name,
            limit=args.limit,
            output=args.output,
            html_output=args.html_output,
        )
        print(json.dumps(packet, ensure_ascii=False, indent=2))
        return 0
    if args.command == "research-import":
        from paa.research_import import import_research_answer

        packet = import_research_answer(questions=args.questions, sources=args.sources,
                                       submission=args.submission, output=args.output)
        print(json.dumps({"task_id": packet["task_id"], "binding_state": packet["binding_state"],
                          "admission": packet["admission"]}, ensure_ascii=False))
        return 0
    if args.command == "source-question":
        from paa.source_question import source_question_cli

        packet = source_question_cli(
            questions=args.questions,
            sources=args.sources,
            task_id=args.task_id,
            all_tasks=args.all_tasks,
            split=args.split,
            db_path=args.db,
            output=args.output,
            include_db_context=args.include_db_ledger,
            provider=args.provider,
            key_file=args.key_file,
            context_limit_tokens=args.context_limit_tokens,
        )
        print(json.dumps(packet["summary"], ensure_ascii=False, indent=2))
        return 0
    if args.command == "source-revision":
        from paa.source_revision import source_revision_cli

        packet = source_revision_cli(
            candidate=args.candidate,
            critique=args.critique,
            questions=args.questions,
            sources=args.sources,
            output=args.output,
            provider=args.provider,
            key_file=args.key_file,
            context_limit_tokens=args.context_limit_tokens,
        )
        print(json.dumps({"task_id": packet.get("task_id"), "status": packet.get("status"), "revision": packet.get("revision")}, ensure_ascii=False, indent=2))
        return 0
    if args.command == "enrich-ballots":
        from paa.ballot_snapshot_enrichment import enrich_matter_files

        manifest = enrich_matter_files(
            matter_paths=args.matter,
            witness_bank_path=args.witness_bank,
            output_dir=args.output,
            comparison_path=args.comparison,
        )
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0
    if args.command == "matter-sourcebank":
        from paa.matter_sourcebank_adapters import write_sourcebank_matter_artifacts

        if not args.kkv_bank and not args.committee_bank:
            raise SystemExit("matter-sourcebank requires --kkv-bank and/or --committee-bank")
        manifest = write_sourcebank_matter_artifacts(
            kkv_bank_path=args.kkv_bank,
            committee_bank_path=args.committee_bank,
            matter_id=args.matter_id,
            output_dir=args.output,
        )
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0
    if args.command == "export-answer-attachment":
        from paa.question_ledger import (
            export_answer_attachment_artifact,
            fetch_official_attachment_pdf,
        )

        attachment = fetch_official_attachment_pdf(
            attachment_group_id=args.attachment_group_id,
            declared_md5=args.declared_md5,
        )
        record = {}
        if args.record_json:
            record = json.loads(Path(args.record_json).read_text(encoding="utf-8"))
        export = export_answer_attachment_artifact(
            record=record,
            attachment=attachment,
            output_dir=args.output,
        )
        # Do not print raw PDF bytes.
        public = dict(export)
        print(json.dumps(public, ensure_ascii=False, indent=2))
        return 0
    if args.command == "route":
        from paa.route_packet import route_cli

        manifest = route_cli(
            questions=args.questions,
            sources=args.sources,
            output=args.output,
            task_id=args.task_id,
            all_tasks=args.all_tasks,
            split=args.split,
            as_of=args.as_of,
            answer=args.answer,
            overwrite=args.overwrite,
        )
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0
    if args.command == "bootstrap-site":
        from paa.bootstrap_site import bootstrap_site_cli

        reviewed_candidates: dict[str, Path] = {}
        for item in args.reviewed_candidate or []:
            if "=" not in item:
                raise SystemExit(
                    f"--reviewed-candidate expects TASK_ID=PATH, got {item!r}"
                )
            task_id, path_text = item.split("=", 1)
            reviewed_candidates[task_id.strip()] = Path(path_text)
        manifest = bootstrap_site_cli(
            output=args.output,
            matters=args.matter,
            questions=args.question,
            oulaskangas_group=args.oulaskangas_group,
            oulaskangas_cost=args.oulaskangas_cost,
            oulaskangas_monitoring=args.oulaskangas_monitoring,
            editorial_packet=args.editorial_packet,
            editorial_overview=args.editorial_overview,
            source_banks=args.source_bank,
            route_dirs=args.route_dir,
            reviews=args.review,
            reviewed_candidates=reviewed_candidates,
            journeys=args.journey,
            title=args.title,
        )
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0
    if args.command == "integrated-reader":
        from paa.integrated_reader import build_integrated_reader

        output = Path(args.output)
        if output.exists() and any(output.iterdir()):
            raise SystemExit(f"output directory is not empty: {output}")
        output.mkdir(parents=True, exist_ok=True)
        receipt = build_integrated_reader(
            manifest_path=args.manifest,
            output_dir=output,
            inputs_root=args.inputs,
        )
        print(json.dumps(receipt, ensure_ascii=False, indent=2))
        return 0
    if args.command == "journey":
        from paa.question_journey import journey_cli

        manifest = journey_cli(
            route_packet=args.route_packet,
            mapping=args.mapping,
            output=args.output,
            answer=args.answer,
            source_answer_review=args.source_answer_review,
            projection_review=args.projection_review,
            overwrite=args.overwrite,
        )
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0
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

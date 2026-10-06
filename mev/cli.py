"""mev — Mekanismirealismi pipeline CLI.

Replaces build_pipeline.sh with a proper Python entry point.
Each command delegates to mev package modules.

Usage:
    mev status
    mev build he-index | atoms | enrich | causal-map | aggregate-entities | lawvm
    mev detect unreason | scrutiny | sabotage | drift | noise-lausunto [options]
    mev pipeline all | all-llm
"""
from __future__ import annotations

import argparse
import sqlite3
import subprocess
import sys
from pathlib import Path

from mev.config import (
    BOOK_ROOT,
    CAUSAL_MAP_DB,
    CENSUS_DIR,
    ENRICHMENTS_DB,
    GRAPH_DIR,
    HE_DB_DIR,
    HE_INDEX_DB,
    INDEX_DB,
    LAWVM_DIR,
    LLAMA_API_BASE,
    ROOT,
    STATUTE_ZIP,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(cmd: list[str], cwd: Path = ROOT) -> int:
    """Run a command and return its exit code."""
    result = subprocess.run(cmd, cwd=cwd)
    return result.returncode


def _uv_module(module: str, *args: str, cwd: Path = ROOT) -> int:
    """Run a mev package module via uv run python -m."""
    return _run(["uv", "run", "python", "-m", module] + list(args), cwd=cwd)


def _die(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def _extract_he(extra: list[str]) -> str | None:
    """Extract --he VALUE or bare he-* from extra args."""
    for i, a in enumerate(extra):
        if a == '--he' and i + 1 < len(extra):
            return extra[i + 1]
        if a.startswith('he-'):
            return a
    return None


def _count_he_dbs() -> int:
    if not HE_DB_DIR.exists():
        return 0
    return sum(1 for _ in HE_DB_DIR.glob("he-*.db"))


async def _run_with_monitor(make_coro, interval: float = 60):
    """Run an async stage with periodic LLM throughput printing."""
    from mev.llm import TRACKER
    TRACKER.start_monitor(interval=interval)
    try:
        return await make_coro()
    finally:
        TRACKER.stop_monitor()


def _llm_ok() -> bool:
    import urllib.request
    for path in ("/health", "/v1/models"):
        try:
            urllib.request.urlopen(LLAMA_API_BASE + path, timeout=2)
            return True
        except urllib.error.HTTPError:
            return True  # got a response = server is up
        except Exception:
            continue
    return False


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_status() -> int:
    print("=== mev pipeline status ===\n")

    failures: list[str] = []

    def ok(msg: str) -> None:
        print(f"  \033[32m✓\033[0m  {msg}")

    def warn(msg: str) -> None:
        print(f"  \033[33m~\033[0m  {msg}")

    def err(msg: str) -> None:
        print(f"  \033[31m✗\033[0m  {msg}")
        failures.append(msg)

    # statute.zip
    if STATUTE_ZIP.exists():
        size = STATUTE_ZIP.stat().st_size // (1024 * 1024)
        ok(f"statute.zip: {size} MB")
    else:
        err(f"statute.zip: MISSING ({STATUTE_ZIP})")

    # HE master index
    if HE_INDEX_DB.exists():
        try:
            conn = sqlite3.connect(HE_INDEX_DB)
            n = conn.execute("SELECT COUNT(*) FROM he_index").fetchone()[0]
            r = conn.execute("SELECT COUNT(*) FROM he_statute_refs").fetchone()[0]
            conn.close()
            ok(f"HE master index: {n} HEs, {r} statute refs")
        except Exception:
            warn("HE master index: exists but unreadable")
    else:
        warn("HE master index: not built  (mev build he-index)")

    # HE databases
    n_dbs = _count_he_dbs()
    if n_dbs:
        ok(f"HE databases: {n_dbs} DBs in {HE_DB_DIR.relative_to(ROOT)}/")
    else:
        err("HE databases: none  (mev build atoms)")

    # Legislative index
    if INDEX_DB.exists():
        ok(f"Legislative index: {INDEX_DB.relative_to(BOOK_ROOT)}")
    else:
        warn("Legislative index: missing")

    # he_enrichments.db — report CONTENTS, not just size. Every table here is an
    # LLM detector output, so empty is legitimate on a fresh build; counts are
    # informational. The hard contract lives on state_causal_map.db below.
    if ENRICHMENTS_DB.exists():
        size_kb = ENRICHMENTS_DB.stat().st_size // 1024
        try:
            conn = sqlite3.connect(f"file:{ENRICHMENTS_DB}?mode=ro", uri=True)
            counts = []
            for t in ("sentence_tag", "lausunto_tag", "mietinto_tag", "discourse_node"):
                try:
                    n = conn.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
                except sqlite3.Error:
                    n = None
                counts.append(f"{t}={n if n is not None else 'MISSING'}")
            conn.close()
            ok(f"he_enrichments.db: {size_kb} KB · " + " ".join(counts))
        except Exception:
            warn(f"he_enrichments.db: {size_kb} KB but unreadable")
    else:
        warn("he_enrichments.db: not built")

    # LawVM graph artifact
    graph_meta = GRAPH_DIR / "meta.json"
    if graph_meta.exists():
        import json
        try:
            meta = json.loads(graph_meta.read_text())
            n_stat = meta.get("corpus_size", meta.get("n_statutes", "?"))
            ok(f"LawVM graph: {n_stat} statutes in {GRAPH_DIR}/")
        except Exception:
            ok(f"LawVM graph: exists ({GRAPH_DIR}/)")
    else:
        warn("LawVM graph: not built  (mev build lawvm)")

    # Census
    census_report = CENSUS_DIR / "census_report.md"
    if census_report.exists():
        ok(f"Census report: {census_report.relative_to(ROOT)}")
    else:
        warn("Census report: not generated  (mev build lawvm)")

    # State causal map DB — SIZE IS NOT A HEALTH CHECK.
    # An all-empty table set is a healthy-sized file; that is exactly how nine
    # tables stayed at zero rows for five months while this line printed green.
    # See mev/db_contract.py for the contract and the incident.
    contract_breaches: list[tuple[str, str, str]] = []
    if CAUSAL_MAP_DB.exists():
        size_kb = CAUSAL_MAP_DB.stat().st_size // 1024
        from mev.db_contract import (
            CAUSAL_MAP_CONTRACT,
            causal_map_optional_counts,
            check_causal_map,
        )
        try:
            contract_breaches = check_causal_map()
        except Exception as exc:
            err(f"state_causal_map.db: {size_kb} KB but contract check failed: {exc}")
        else:
            n_contract = len(CAUSAL_MAP_CONTRACT)
            if contract_breaches:
                err(
                    f"state_causal_map.db: {size_kb} KB but "
                    f"{len(contract_breaches)}/{n_contract} contracted tables unpopulated"
                )
                for table, producer, problem in contract_breaches:
                    print(f"       \033[31m{problem:8}\033[0m {table}  →  run: {producer}")
            else:
                ok(f"state_causal_map.db: {size_kb} KB · {n_contract}/{n_contract} contracted tables populated")
            for table, producer, n in causal_map_optional_counts():
                if not n:
                    warn(f"  optional table {table} is empty  ({producer})")
    else:
        err("state_causal_map.db: not built  (mev build causal-map)")

    # LLM server
    if _llm_ok():
        ok(f"LLM server: reachable at {LLAMA_API_BASE}")
    else:
        warn(f"LLM server: not reachable at {LLAMA_API_BASE}")

    print()
    if contract_breaches:
        print("\033[31m  DATA CONTRACT VIOLATION\033[0m — a producer wrote nothing and nobody noticed.")
        print("  Empty tables are not a cosmetic issue: every downstream reader")
        print("  (lakikartta.js, any island query) gets zero rows with no error.")
        print()
    if failures:
        print(f"\033[31mmev status: FAIL\033[0m ({len(failures)} problem(s))")
        for msg in failures:
            print(f"  - {msg}")
        print()
        return 1
    return 0


def cmd_build(stage: str, extra: list[str]) -> int:
    if stage == "he-index":
        print("=== Stage A0: Build HE master index ===")
        if not STATUTE_ZIP.exists():
            _die(f"statute.zip not found at {STATUTE_ZIP}")
        return _uv_module("mev.pipeline.he", "--build-index", *extra)

    elif stage == "atoms":
        print("=== Stage A1: Build per-HE atom databases ===")
        if not STATUTE_ZIP.exists():
            _die(f"statute.zip not found at {STATUTE_ZIP}")
        years = extra or [
            "--year", "2026", "--year", "2025", "--year", "2024",
            "--year", "2023", "--year", "2022", "--year", "2021",
            "--year", "2020", "--year", "2019", "--year", "2018",
            "--year", "2017", "--year", "2016", "--year", "2015",
            "--year", "2014", "--year", "2013", "--year", "2012",
            "--year", "2011", "--year", "2010",
        ]
        return _uv_module("mev.pipeline.he", *years)

    elif stage == "enrich":
        print("=== Stage B2: Enrich per-HE DBs ===")
        if not INDEX_DB.exists():
            _die("legislative_index.sqlite missing — run: mev build index")
        return _uv_module("mev.pipeline.enrich_he_db", "--all", *extra)

    elif stage == "bake-spans":
        print("=== Stage B3: Bake span highlights into per-HE DBs ===")
        return _uv_module("mev.pipeline.bake_spans", *extra)

    elif stage == "export-json":
        print("=== Stage B4: Export per-HE JSON summaries ===")
        he_id = _extract_he(extra)
        if he_id:
            return _uv_module("mev.pipeline.export_he_json", he_id)
        return _uv_module("mev.pipeline.export_he_json", "--all")

    elif stage == "index":
        print("=== Stage B1: Build legislative index ===")
        return _uv_module("mev.pipeline.legislative_index", *extra)

    elif stage == "tag":
        print("=== Stage A2: Sentence tagging (LLM) ===")
        if _count_he_dbs() == 0:
            _die("No HE databases — run: mev build atoms")
        return _uv_module("mev.detectors.tag", "--write-db", *extra)

    elif stage == "drift":
        print("=== Stage A2c+A3a: Mechanism + delegation drift ===")
        if not STATUTE_ZIP.exists():
            _die(f"statute.zip not found at {STATUTE_ZIP}")
        rc = _uv_module("mev.pipeline.mechanism_drift", "--embed", *extra)
        if rc != 0:
            return rc
        return _uv_module("mev.pipeline.delegation_drift", "--embed")

    elif stage == "causal-map":
        print("=== Stage C: Build state_causal_map.db ===")
        if not HE_INDEX_DB.exists():
            print("  Warning: HE master index missing — building it first")
            rc = cmd_build("he-index", [])
            if rc != 0:
                return rc
        extra_args = []
        if GRAPH_DIR.exists():
            print(f"  Using LawVM graph artifact: {GRAPH_DIR}/")
            extra_args = ["--graph-dir", str(GRAPH_DIR)]
        else:
            print("  No LawVM graph artifact — run 'mev build lawvm' for graph_* columns")
        return _uv_module("mev.pipeline.causal_map_db", *extra_args, *extra)

    elif stage == "aggregate-entities":
        print("=== Stage C2: Aggregate committee/expert/minister entities ===")
        if not CAUSAL_MAP_DB.exists():
            _die("state_causal_map.db missing — run: mev build causal-map")
        if _count_he_dbs() == 0:
            _die("No HE databases — run: mev build atoms")
        return _uv_module("mev.pipeline.aggregate_entity_data", *extra)

    elif stage == "lawvm":
        print("=== Stage L: Build LawVM corpus graph + census ===")
        if not LAWVM_DIR.exists():
            _die(f"LawVM not found at {LAWVM_DIR}")
        GRAPH_DIR.mkdir(parents=True, exist_ok=True)
        CENSUS_DIR.mkdir(parents=True, exist_ok=True)
        print("  L1: lawvm build --full")
        rc = _run(["uv", "run", "lawvm", "build", "--full", "--output", str(GRAPH_DIR)], cwd=LAWVM_DIR)
        if rc != 0:
            return rc
        print("  L2: lawvm census")
        return _run(
            ["uv", "run", "lawvm", "census",
             "--graph", str(GRAPH_DIR),
             "--output", str(CENSUS_DIR),
             "--report"],
            cwd=LAWVM_DIR,
        )

    else:
        _die(f"Unknown build stage: {stage}")
        return 1


def cmd_detect(what: str, extra: list[str]) -> int:
    if what == "unreason":
        print("=== Detect: semantic unreason (LLM) ===")
        return _uv_module("mev.detectors.unreason", *extra)

    elif what == "unreason-regex":
        print("=== Detect: regex unreason (Tier 0, no LLM) ===")
        from mev.detectors.unreason_regex import run as urx_run
        he_id = _extract_he(extra)
        min_chars = 0
        for i, a in enumerate(extra):
            if a == '--min-impact-chars' and i + 1 < len(extra):
                min_chars = int(extra[i + 1])
        clear = '--clear' in extra
        urx_run(he_id=he_id, clear=clear, min_impact_chars=min_chars)
        return 0

    elif what == "scrutiny":
        print("=== Detect: scrutiny matching (LLM) ===")
        if not INDEX_DB.exists():
            _die("legislative_index.sqlite missing")
        return _uv_module("mev.detectors.scrutiny", *extra)

    elif what == "sabotage":
        print("=== Detect: parliamentary sabotage (LLM) ===")
        return _uv_module("mev.detectors.sabotage", *extra)

    elif what == "drift":
        print("=== Detect: delegation drift at scale ===")
        return _uv_module("mev.detectors.drift_scale", *extra)

    elif what == "summarize":
        print("=== Detect: summarize delegations (LLM) ===")
        return _uv_module("mev.detectors.summarize", *extra)

    elif what == "tag":
        print("=== Detect: HE sentence tagging (LLM) ===")
        if _count_he_dbs() == 0:
            _die("No HE databases — run: mev build atoms")
        import asyncio
        from mev.detectors.tag import run as tag_run
        he_id = _extract_he(extra)
        write_db = '--write-db' in extra
        force = '--force' in extra
        asyncio.run(tag_run(he_id=he_id, write_db=write_db, force=force))
        return 0

    elif what == "noise-lausunto":
        print("=== Detect: lausunto noise tagging (LLM) ===")
        if _count_he_dbs() == 0:
            _die("No HE databases — run: mev build atoms")
        return _uv_module("mev.detectors.lausunto_noise", *extra)

    elif what == "tag-lausunto":
        print("=== Detect: lausunto claim tagging (LLM) ===")
        if _count_he_dbs() == 0:
            _die("No HE databases — run: mev build atoms")
        import asyncio
        from mev.detectors.tag_lausunto import run as tag_lau_run
        he_id = _extract_he(extra)
        write_db = '--write-db' in extra
        force = '--force' in extra
        asyncio.run(tag_lau_run(he_id=he_id, write_db=write_db, force=force))
        return 0

    elif what == "cross-doc":
        print("=== Cross-document claim matching ===")
        from mev.pipeline.cross_doc import run as cross_run
        he_id = _extract_he(extra)
        write_db = '--write-db' in extra
        cross_run(he_id=he_id, write_db=write_db)
        return 0

    elif what == "discourse":
        print("=== Discourse graph: unified cross-document matching ===")
        from mev.pipeline.discourse import run as disc_run
        he_id = _extract_he(extra)
        write_db = '--write-db' in extra
        disc_run(he_id=he_id, write_db=write_db)
        return 0

    elif what == "tag-mietinto":
        print("=== Detect: mietinto paragraph tagging (LLM) ===")
        import asyncio
        from mev.detectors.tag_mietinto import run as tm_run
        he_id = _extract_he(extra)
        write_db = '--write-db' in extra
        force = '--force' in extra
        asyncio.run(tm_run(he_id=he_id, write_db=write_db, force=force))
        return 0

    elif what == "tag-ptk":
        print("=== Detect: PTK speech tagging (LLM) ===")
        import asyncio
        from mev.detectors.tag_ptk import run as ptk_run
        he_id = _extract_he(extra)
        write_db = '--write-db' in extra
        force = '--force' in extra
        asyncio.run(ptk_run(he_id=he_id, write_db=write_db, force=force))
        return 0

    elif what == "tag-spans":
        print("=== Detect: span-level highlighting (LLM + regex) ===")
        import asyncio
        from mev.detectors.tag_spans import run as spans_run
        he_id = _extract_he(extra)
        write_db = '--write-db' in extra
        force = '--force' in extra
        doc_types = [a for i, a in enumerate(extra) if extra[i-1:i] == ['--doc-type']] if '--doc-type' in extra else None
        asyncio.run(spans_run(he_id=he_id, doc_types=doc_types, write_db=write_db, force=force))
        return 0

    elif what == "comm-evade":
        print("=== Detect: committee evasion ===")
        from mev.detectors.comm_evade import run as ce_run
        ce_run()
        return 0

    elif what == "ev-outcome":
        print("=== Detect: EV outcome classification ===")
        from mev.detectors.ev_outcome import run as evo_run
        evo_run()
        return 0

    elif what == "longitudinal":
        print("=== Detect: longitudinal strike ===")
        from mev.detectors.longitudinal import run as long_run
        long_run()
        return 0

    elif what == "feedback-gaps":
        print("=== Detect: feedback-impact gaps ===")
        from mev.detectors.feedback_gaps import run as fg_run
        fg_run()
        return 0

    elif what == "corroboration":
        print("=== Pipeline: corroboration scoring ===")
        from mev.pipeline.corroboration import run as corr_run
        corr_run()
        return 0

    elif what == "scrutiny-summary":
        print("=== Pipeline: scrutiny summary ===")
        from mev.pipeline.scrutiny_summary import run as ss_run
        ss_run()
        return 0

    elif what == "all-he":
        print("=== Full pipeline for one HE ===")
        he_id = _extract_he(extra)
        write_db = '--write-db' in extra
        if not he_id:
            _die("--he required for all-he mode")
        if not he_id.startswith('he-'):
            he_id = f'he-{he_id}'

        import asyncio
        from mev.detectors.tag import run as tag_run
        from mev.detectors.tag_lausunto import run as tag_lau_run
        from mev.detectors.tag_mietinto import run as tag_mie_run
        from mev.detectors.tag_ptk import run as ptk_run
        from mev.detectors.tag_spans import run as spans_run
        from mev.llm import TRACKER
        from mev.pipeline.discourse import run as disc_run

        TRACKER.reset()

        async_stages = [
            ("tag (HE sentences)",          lambda: tag_run(he_id=he_id, write_db=write_db, force=True)),
            ("tag-lausunto (expert claims)", lambda: tag_lau_run(he_id=he_id, write_db=write_db, force=True)),
            ("tag-mietinto (committee)",     lambda: tag_mie_run(he_id=he_id, write_db=write_db, force=True)),
            ("tag-ptk (plenary speeches)",   lambda: ptk_run(he_id=he_id, write_db=write_db, force=True)),
            ("tag-spans (span highlights)",  lambda: spans_run(he_id=he_id, write_db=write_db, force=True)),
        ]
        for label, make_coro in async_stages:
            print(f"\n{'='*60}\n  Stage: {label}\n{'='*60}")
            asyncio.run(_run_with_monitor(make_coro))

        print(f"\n{'='*60}\n  Stage: discourse (cross-doc matching)\n{'='*60}")
        disc_run(he_id=he_id, write_db=write_db)

        print(f"\n{'='*60}\n  Stage: resolve-refs (link V-spans to documents)\n{'='*60}")
        _uv_module("mev.pipeline.resolve_refs", he_id)

        print(f"\n{'='*60}\n  Stage: unreason-regex (Tier 0 regex, no LLM)\n{'='*60}")
        from mev.detectors.unreason_regex import run as urx_run
        urx_run(he_id=he_id)

        print(f"\n{'='*60}\n  Stage: enrich (sync to per-HE DB)\n{'='*60}")
        cmd_build("enrich", [he_id])

        print(f"\n{'='*60}\n  Stage: noise-lausunto (boilerplate hiding)\n{'='*60}")
        _uv_module("mev.detectors.lausunto_noise", "--he", he_id)

        print(f"\n{'='*60}\n  Stage: bake-spans (highlight signal in HTML)\n{'='*60}")
        _uv_module("mev.pipeline.bake_spans", he_id)

        final = TRACKER.summary_line()
        if final:
            print(f"\n{'='*60}\n  {final}\n{'='*60}")
        return 0

    elif what == "all":
        print("=== Run ALL detectors (corpus-wide) ===")
        write_db = '--write-db' in extra
        db_flag = ['--write-db'] if write_db else []

        import asyncio
        from mev.detectors.tag import run as tag_run
        from mev.detectors.tag_lausunto import run as tag_lau_run
        from mev.detectors.tag_mietinto import run as tag_mie_run
        from mev.detectors.tag_ptk import run as ptk_run
        from mev.llm import TRACKER
        from mev.pipeline.discourse import run as disc_run

        TRACKER.reset()

        async_stages = [
            ("tag (HE sentences)",          lambda: tag_run(all_hes=True, write_db=write_db)),
            ("tag-lausunto (expert claims)", lambda: tag_lau_run(write_db=write_db)),
            ("tag-mietinto (committee)",     lambda: tag_mie_run(write_db=write_db)),
            ("tag-ptk (plenary speeches)",   lambda: ptk_run(write_db=write_db)),
        ]
        for label, make_coro in async_stages:
            print(f"\n{'='*60}\n  Stage: {label}\n{'='*60}")
            asyncio.run(_run_with_monitor(make_coro))

        mid = TRACKER.summary_line()
        if mid:
            print(f"\n  {mid}")

        print(f"\n{'='*60}\n  Stage: discourse (cross-doc matching)\n{'='*60}")
        disc_run(write_db=write_db)

        print(f"\n{'='*60}\n  Stage: unreason (LLM)\n{'='*60}")
        _uv_module("mev.detectors.unreason", *db_flag)

        print(f"\n{'='*60}\n  Stage: unreason-regex (Tier 0, no LLM)\n{'='*60}")
        from mev.detectors.unreason_regex import run as urx_run
        urx_run()

        print(f"\n{'='*60}\n  Stage: resolve-refs (link V-spans to documents)\n{'='*60}")
        _uv_module("mev.pipeline.resolve_refs", "--all")

        print(f"\n{'='*60}\n  Stage: enrich (sync to per-HE DBs)\n{'='*60}")
        cmd_build("enrich", ["--all"])

        print(f"\n{'='*60}\n  Stage: noise-lausunto (boilerplate hiding)\n{'='*60}")
        _uv_module("mev.detectors.lausunto_noise")

        print(f"\n{'='*60}\n  Stage: bake-spans (highlight signal in HTML)\n{'='*60}")
        _uv_module("mev.pipeline.bake_spans", "--all")

        final = TRACKER.summary_line()
        if final:
            print(f"\n{'='*60}\n  {final}\n{'='*60}")
        return 0

    elif what == "bench-format":
        print("=== Bench: LLM output format benchmark ===")
        from mev.bench.runner import run_benchmark_cli
        run_benchmark_cli(extra)
        return 0

    else:
        _die(f"Unknown detector: {what}")
        return 1


def cmd_pipeline(mode: str) -> int:
    # Non-LLM build stages in dependency order
    build_stages = [
        "lawvm",       # Stage L: corpus graph + census (independent)
        "he-index",    # Stage A0: HE master index
        "atoms",       # Stage A1: per-HE atom DBs
        "index",       # Stage B1: legislative index (Lakitutka, Eduskunta, etc.)
        "enrich",      # Stage B2: enrich per-HE DBs with expert/committee data
        "drift",       # Stage A2c+A3a: mechanism + delegation drift
        "causal-map",  # Stage C: state_causal_map.db (uses LawVM graph if available)
        "aggregate-entities",  # Stage C2: committee/expert/org/minister roll-ups (needs C + atoms)
    ]

    for stage in build_stages:
        print()
        rc = cmd_build(stage, [])
        if rc != 0:
            print(f"Pipeline aborted at build stage: {stage}", file=sys.stderr)
            return rc

    if mode == "all-llm":
        # LLM stages in dependency order
        llm_steps: list[tuple[str, str, list[str]]] = [
            ("build", "tag",      []),              # tag sentences (needs atoms)
            ("detect", "unreason", []),              # semantic unreason
            ("detect", "sabotage", []),              # parliamentary sabotage
            ("detect", "summarize", []),             # delegation summaries
            ("detect", "scrutiny", []),              # expert concern matching
            ("detect", "drift", ["--top", "20", "--llm", "--embed"]),  # drift at scale
        ]
        for kind, what, extra in llm_steps:
            print()
            rc = cmd_build(what, extra) if kind == "build" else cmd_detect(what, extra)
            if rc != 0:
                print(f"Pipeline aborted at LLM stage: {what}", file=sys.stderr)
                return rc

    print("\nPipeline complete.")
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        prog="mev",
        description="Mekanismirealismi pipeline CLI (replaces build_pipeline.sh)",
    )
    sub = ap.add_subparsers(dest="command", metavar="COMMAND")

    # status
    sub.add_parser("status", help="Show pipeline component status")

    # build
    build_p = sub.add_parser("build", help="Build a pipeline stage")
    build_p.add_argument(
        "stage",
        choices=["he-index", "atoms", "enrich", "bake-spans", "export-json", "index", "tag", "drift", "causal-map", "aggregate-entities", "lawvm"],
        metavar="STAGE",
        help="he-index | atoms | enrich | bake-spans | export-json | index | tag | drift | causal-map | aggregate-entities | lawvm",
    )
    build_p.add_argument("extra", nargs=argparse.REMAINDER, help="Extra args passed to the script")

    # detect
    detect_p = sub.add_parser("detect", help="Run an LLM detector")
    detect_p.add_argument(
        "what",
        choices=[
            "unreason", "unreason-regex", "scrutiny", "sabotage", "drift", "summarize",
            "noise-lausunto", "tag", "tag-lausunto", "tag-ptk", "tag-mietinto", "tag-spans",
            "cross-doc", "discourse",
            "comm-evade", "ev-outcome", "longitudinal",
            "feedback-gaps", "corroboration", "scrutiny-summary",
            "all-he", "all",
            "bench-format",
        ],
        metavar="DETECTOR",
    )
    detect_p.add_argument("extra", nargs=argparse.REMAINDER, help="Extra args passed to the script")

    # pipeline
    pipeline_p = sub.add_parser("pipeline", help="Run full pipeline")
    pipeline_p.add_argument(
        "mode",
        choices=["all", "all-llm"],
        help="all (no LLM) | all-llm (includes LLM stages)",
    )

    # aux-sources
    aux_p = sub.add_parser("aux-sources", help="Fetch/index auxiliary evidence source seeds")
    aux_p.add_argument("extra", nargs=argparse.REMAINDER, help="Args passed to mev.pipeline.aux_sources")

    args = ap.parse_args()

    if args.command == "status":
        sys.exit(cmd_status())
    elif args.command == "build":
        sys.exit(cmd_build(args.stage, args.extra))
    elif args.command == "detect":
        sys.exit(cmd_detect(args.what, args.extra))
    elif args.command == "pipeline":
        sys.exit(cmd_pipeline(args.mode))
    elif args.command == "aux-sources":
        from mev.pipeline.aux_sources import main as aux_sources_main

        sys.exit(aux_sources_main(args.extra))
    else:
        ap.print_help()
        sys.exit(0)

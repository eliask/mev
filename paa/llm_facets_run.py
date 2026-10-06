"""Bounded execution and source-packet integration for the facet extractor.

The driver owns filesystem/SQLite/model effects.  The semantic module
``paa.llm_facets`` owns source validation and normalization.  Development and
held-out source loading deliberately have separate functions: the blind
loader strips every selection, gold and adjudication field before a request is
constructed, and reference loading is available only to the post-run
evaluator.
"""

import argparse
import asyncio
import json
import sqlite3
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import final

from paa.llm_client import LocalLLMClient, digest
from paa.llm_facets import (
    FACET_PROMPT_VERSION,
    FACET_SCHEMA_VERSION,
    FACET_VALIDATOR_VERSION,
    FacetResearchPacket,
    FacetRunOutcome,
    SourceDocument,
    SourceUnit,
    build_facet_request,
    facet_response_schema,
    normalize_facet_output,
    research_packet,
    source_document,
    source_only_fixture_row,
    source_units,
)


@final
@dataclass(frozen=True, slots=True, kw_only=True, match_args=False)
class FacetCase:
    """One source packet selected before inference; no labels are carried."""

    item_id: str
    document: SourceDocument
    units: tuple[SourceUnit, ...]


@final
@dataclass(frozen=True, slots=True, kw_only=True, match_args=False)
class FacetRunConfig:
    prompt_version: str = FACET_PROMPT_VERSION
    enable_thinking: bool = False
    max_workers: int = 1
    max_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.max_workers < 1 or self.max_workers > 3:
            raise ValueError("facet workers must be between one and three")
        if self.max_tokens is not None and self.max_tokens < 128:
            raise ValueError("max_tokens is too small for a facet response")


def _db_connection(db: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"file:{db.resolve()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def load_sqlite_cases(
    db: Path,
    *,
    document_ids: Iterable[str] | None = None,
    limit: int | None = None,
) -> tuple[FacetCase, ...]:
    """Load exact source units from the maintained DB without model labels."""

    requested = set(document_ids) if document_ids is not None else None
    connection = _db_connection(db)
    try:
        source_rows = list(connection.execute("SELECT * FROM documents ORDER BY document_id"))
        proposition_rows: dict[str, list[dict[str, object]]] = {}
        for row in connection.execute("SELECT statement_id,json FROM propositions ORDER BY proposition_id"):
            value = json.loads(row["json"])
            proposition_rows.setdefault(str(row["statement_id"]), []).append(value)
    finally:
        connection.close()
    cases: list[FacetCase] = []
    for row in source_rows:
        document_id = str(row["document_id"])
        if requested is not None and document_id not in requested:
            continue
        document = source_document(
            {
                "document_id": document_id,
                "source_id": row["source_id"],
                "url": row["url"],
                "actor_id": row["actor_id"],
                "language": row["language"],
                "source_text": row["text"],
                "stated_earliest": row["stated_earliest"],
                "source_sha256": row["sha256"],
                "question": row["field_label"],
            }
        )
        raw_units: list[dict[str, object]] = []
        for proposition in proposition_rows.get(document_id, []):
            span = proposition.get("source_span")
            if not isinstance(span, Mapping):
                continue
            raw_units.append(
                {
                    "unit_id": proposition.get("proposition_id"),
                    "text": proposition.get("source_text"),
                    "start": span.get("start"),
                    "end": span.get("end"),
                }
            )
        if not raw_units:
            raw_units = [{"unit_id": document_id + "-source", "text": document.text, "start": 0, "end": len(document.text)}]
        # A malformed historical proposition is a source-slice failure, not a
        # reason to silently widen the unit to the complete document. Raise it
        # here so the declared input remains accounted and repairable.
        units = source_units(document, raw_units)
        cases.append(FacetCase(item_id=document_id, document=document, units=units))
        if limit is not None and len(cases) >= limit:
            break
    if requested is not None:
        missing = requested - {case.item_id for case in cases}
        if missing:
            raise ValueError(f"requested source documents are missing: {sorted(missing)}")
    if not cases:
        raise ValueError("no source cases selected")
    return tuple(cases)


def load_blind_source_only(path: Path, *, limit: int | None = None) -> tuple[FacetCase, ...]:
    """Load blind source text while stripping gold/selection/adjudication.

    This is the only loader used by ``run_blind_frozen``.  It never returns the
    fixture row and therefore cannot accidentally include its evaluation
    fields in a prompt or receipt.
    """

    cases: list[FacetCase] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            document, units = source_only_fixture_row(row)
            if document.document_id in seen:
                raise ValueError(f"duplicate blind document_id: {document.document_id}")
            seen.add(document.document_id)
            cases.append(FacetCase(item_id=document.document_id, document=document, units=units))
            if limit is not None and len(cases) >= limit:
                break
    if not cases:
        raise ValueError(f"blind fixture has no source rows: {path}")
    return tuple(cases)


def load_blind_references_for_evaluation(path: Path) -> tuple[dict[str, object], ...]:
    """Read blind adjudication only after a frozen run, never for prompting."""

    references: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("blind reference row must be an object")
                references.append(row)
    return tuple(references)


def _atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _source_snapshot(cases: Sequence[FacetCase]) -> str:
    return digest(
        [
            {
                "item_id": case.item_id,
                "document_id": case.document.document_id,
                "source_sha256": case.document.source_sha256,
                "units": [unit.as_dict() for unit in case.units],
            }
            for case in cases
        ]
    )


def _receipt_summary(receipt: Mapping[str, object]) -> dict[str, object]:
    """Keep transport telemetry while excluding model internal reasoning."""

    keys = (
        "request_id",
        "status",
        "cache_hit",
        "model_id",
        "server_build",
        "started_at",
        "elapsed_seconds",
        "attempts",
        "usage",
        "timings",
        "shared_slot_wait_seconds",
        "error",
    )
    return {key: receipt[key] for key in keys if key in receipt}


def _request_message(request: Mapping[str, object], index: int) -> str:
    """Narrow the dynamic request mapping at the model-call boundary."""

    messages = request.get("messages")
    if not isinstance(messages, list) or index < 0 or index >= len(messages):
        raise ValueError("facet request has an incomplete messages list")
    message = messages[index]
    if not isinstance(message, Mapping):
        raise TypeError("facet request message must be an object")
    content = message.get("content")
    if not isinstance(content, str):
        raise TypeError("facet request message content must be a string")
    return content


def _write_outcome(path: Path, outcome: FacetRunOutcome, request: Mapping[str, object]) -> None:
    metadata = request.get("metadata")
    source_sha256 = metadata.get("source_sha256") if isinstance(metadata, Mapping) else None
    unit_ids = metadata.get("unit_ids") if isinstance(metadata, Mapping) else None
    output: dict[str, object] = {
        "item_id": outcome.item_id,
        "request": {
            "prompt_version": request.get("prompt_version"),
            "source_sha256": source_sha256,
            "unit_ids": unit_ids,
            "enable_thinking": outcome.enable_thinking,
        },
        "outcome": {
            "status": outcome.status,
            "request_id": outcome.request_id,
            "receipt_status": outcome.receipt_status,
            "elapsed_seconds": outcome.elapsed_seconds,
            "generation_tokens": outcome.generation_tokens,
            "cache_hit": outcome.cache_hit,
            "enable_thinking": outcome.enable_thinking,
            "error": outcome.error,
        },
        "extraction": outcome.output.as_dict() if outcome.output is not None else None,
        "admission_state": "PROPOSED",
    }
    _atomic_write_json(path, output)


async def run_facet_cases(
    cases: Sequence[FacetCase],
    output: Path,
    *,
    config: FacetRunConfig | None = None,
    cache_dir: Path | None = None,
    timeout: float = 600,
) -> dict[str, object]:
    """Run a bounded source→facet pass with one owned queue.

    The queue is bounded to ``max_workers`` and each worker retains the case
    ID when merging.  LocalLLMClient's shared slot lock coordinates with other
    PAA jobs; this driver never assumes that agent count equals GPU capacity.
    """

    if config is None:
        config = FacetRunConfig()
    if not cases:
        raise ValueError("facet run requires at least one case")
    output.mkdir(parents=True, exist_ok=True)
    client = LocalLLMClient(cache_dir=cache_dir, timeout=timeout)
    started = time.monotonic()
    manifest_path = output / "manifest.json"
    try:
        model = await client.discover()
        sample_request = build_facet_request(cases[0].document, cases[0].units, prompt_version=config.prompt_version)
        sample_system = _request_message(sample_request, 0)
        identity = {
            "contract": FACET_SCHEMA_VERSION,
            "validator_version": FACET_VALIDATOR_VERSION,
            "model": model,
            "prompt_version": config.prompt_version,
            "prompt_sha256": digest(sample_system),
            "schema_sha256": digest(facet_response_schema(compact=config.prompt_version == "facets_v2_linked_propositions")),
            "source_snapshot_sha256": _source_snapshot(cases),
            "enable_thinking": config.enable_thinking,
            "max_workers": config.max_workers,
            "max_tokens": config.max_tokens,
        }
        if manifest_path.exists():
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
            if previous.get("identity") != identity:
                raise ValueError("facet run directory belongs to a different configuration or source snapshot")
        else:
            _atomic_write_json(
                manifest_path,
                {
                    "identity": identity,
                    "started_at": datetime.now(UTC).isoformat(),
                    "document_count": len(cases),
                    "unit_count": sum(len(case.units) for case in cases),
                    "admission_state": "PROPOSED",
                    "source_scope": "source-only cases; blind references excluded from requests",
                },
            )
        result_dir = output / "cases"
        result_dir.mkdir(exist_ok=True)
        queue: asyncio.Queue[FacetCase | None] = asyncio.Queue(maxsize=config.max_workers)
        results: dict[str, FacetRunOutcome] = {}
        result_lock = asyncio.Lock()

        async def producer() -> None:
            for case in cases:
                await queue.put(case)
            for _ in range(config.max_workers):
                await queue.put(None)

        async def worker() -> None:
            while True:
                case = await queue.get()
                try:
                    if case is None:
                        return
                    request = build_facet_request(case.document, case.units, prompt_version=config.prompt_version)
                    requested_max_tokens = request.get("max_tokens")
                    if not isinstance(requested_max_tokens, int):
                        raise TypeError("facet request max_tokens must be an integer")
                    max_tokens = config.max_tokens or requested_max_tokens
                    system_message = _request_message(request, 0)
                    user_message = _request_message(request, 1)
                    receipt = await client.request(
                        "facet-extraction:" + config.prompt_version,
                        system_message,
                        user_message,
                        schema=facet_response_schema(compact=config.prompt_version == "facets_v2_linked_propositions"),
                        max_tokens=max_tokens,
                        enable_thinking=config.enable_thinking,
                    )
                    parsed = receipt.get("parsed") if receipt.get("status") == "OK" else None
                    extraction = normalize_facet_output(
                        parsed if isinstance(parsed, (Mapping, str)) else (receipt.get("content") or ""),
                        case.document,
                        case.units,
                        prompt_version=config.prompt_version,
                    )
                    usage = receipt.get("usage")
                    generation_tokens = usage.get("completion_tokens") if isinstance(usage, Mapping) else None
                    outcome = FacetRunOutcome(
                        item_id=case.item_id,
                        status=extraction.status.value if receipt.get("status") == "OK" else str(receipt.get("status", "FAILED")),
                        request_id=str(receipt.get("request_id")) if receipt.get("request_id") is not None else None,
                        receipt_status=str(receipt.get("status")) if receipt.get("status") is not None else None,
                        elapsed_seconds=float(receipt["elapsed_seconds"]) if isinstance(receipt.get("elapsed_seconds"), (int, float)) else None,
                        generation_tokens=int(generation_tokens) if isinstance(generation_tokens, int) else None,
                        cache_hit=bool(receipt.get("cache_hit", False)),
                        enable_thinking=config.enable_thinking,
                        output=extraction,
                        error=str(receipt.get("error")) if receipt.get("error") else None,
                    )
                    _write_outcome(result_dir / f"{case.item_id}.json", outcome, request)
                    async with result_lock:
                        results[case.item_id] = outcome
                finally:
                    queue.task_done()

        await asyncio.gather(producer(), *(worker() for _ in range(config.max_workers)))
        ordered = [results[item_id] for item_id in sorted(results)]
        status_counts: dict[str, int] = {}
        receipt_counts: dict[str, int] = {}
        for outcome in ordered:
            status_counts[outcome.status] = status_counts.get(outcome.status, 0) + 1
            receipt = outcome.receipt_status or "MISSING"
            receipt_counts[receipt] = receipt_counts.get(receipt, 0) + 1
        summary = {
            "schema_version": "paa.facets.run.v1",
            "output": str(output),
            "identity": identity,
            "document_count": len(cases),
            "unit_count": sum(len(case.units) for case in cases),
            "outcome_status": status_counts,
            "receipt_status": receipt_counts,
            "facet_count": sum(len(outcome.output.facets) for outcome in ordered if outcome.output is not None),
            "invalid_record_count": sum(len(outcome.output.invalid_records) for outcome in ordered if outcome.output is not None),
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "admission_state": "PROPOSED",
            "internal_reasoning_published": False,
        }
        _atomic_write_json(output / "summary.json", summary)
        return summary
    finally:
        await client.close()


async def run_development_matrix(
    cases: Sequence[FacetCase],
    output_root: Path,
    *,
    prompt_versions: Sequence[str],
    thinking_modes: Sequence[bool] = (False, True),
    max_workers: int = 1,
    cache_dir: Path | None = None,
) -> tuple[dict[str, object], ...]:
    """Run a finite prompt/thinking matrix on development source only."""

    summaries: list[dict[str, object]] = []
    for prompt_version in prompt_versions:
        for enable_thinking in thinking_modes:
            run_dir = output_root / f"{prompt_version}-thinking-{str(enable_thinking).lower()}"
            summaries.append(
                await run_facet_cases(
                    cases,
                    run_dir,
                    config=FacetRunConfig(prompt_version=prompt_version, enable_thinking=enable_thinking, max_workers=max_workers),
                    cache_dir=cache_dir,
                )
            )
    _atomic_write_json(output_root / "development_matrix.json", {"runs": summaries, "selection": "development-only; no blind references read"})
    return tuple(summaries)


async def run_blind_frozen(
    fixture: Path,
    output: Path,
    *,
    prompt_version: str,
    enable_thinking: bool,
    max_workers: int = 1,
    cache_dir: Path | None = None,
) -> dict[str, object]:
    """Run a frozen prompt on source-only held-out rows.

    The function does not call ``load_blind_references_for_evaluation`` and
    therefore cannot tune against held-out adjudication.
    """

    cases = load_blind_source_only(fixture)
    return await run_facet_cases(
        cases,
        output,
        config=FacetRunConfig(prompt_version=prompt_version, enable_thinking=enable_thinking, max_workers=max_workers),
        cache_dir=cache_dir,
    )


def load_research_packet(
    result_file: Path,
    document: Mapping[str, object] | SourceDocument,
    units: Sequence[Mapping[str, object]] | Sequence[SourceUnit],
) -> FacetResearchPacket:
    """Expose one stored proposal packet to a consumer without admission."""

    value = json.loads(result_file.read_text(encoding="utf-8"))
    extraction_value = value.get("extraction") if isinstance(value, Mapping) else None
    if not isinstance(extraction_value, Mapping):
        raise TypeError("result file does not contain an extraction")
    # Revalidate from the retained normalized wire result. This protects a
    # consumer from stale/corrupted facet JSON and keeps source checks in one
    # place.
    extraction = normalize_facet_output(
        {
            "schema_version": FACET_SCHEMA_VERSION,
            "document_id": extraction_value.get("document_id"),
            "source_id": extraction_value.get("source_id"),
            "abstain": extraction_value.get("status") == "ABSTAIN",
            "abstention_reason": extraction_value.get("abstention_reason"),
            "coverage": {"status": "COMPLETE"},
            "units": [
                {
                    "unit_id": unit.get("unit_id"),
                    "status": unit.get("status"),
                    "abstention_reason": unit.get("abstention_reason"),
                    "facets": unit.get("facets", []),
                }
                for unit in extraction_value.get("units", [])
                if isinstance(unit, Mapping)
            ],
        },
        document,
        units,
        prompt_version=str(extraction_value.get("prompt_version", FACET_PROMPT_VERSION)),
    )
    return research_packet(document, units, extraction)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("data/paa.sqlite"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", default=FACET_PROMPT_VERSION)
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--blind", type=Path)
    parser.add_argument("--fixture", type=Path, default=Path("paa/contracts/fixtures/semantic_blind_heldout_20261007.jsonl"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.blind is not None:
        summary = asyncio.run(
            run_blind_frozen(
                args.fixture,
                args.output,
                prompt_version=args.prompt,
                enable_thinking=args.thinking,
                max_workers=args.max_workers,
            )
        )
    else:
        cases = load_sqlite_cases(args.db, limit=args.limit)
        summary = asyncio.run(
            run_facet_cases(
                cases,
                args.output,
                config=FacetRunConfig(prompt_version=args.prompt, enable_thinking=args.thinking, max_workers=args.max_workers),
            )
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FacetCase",
    "FacetRunConfig",
    "load_blind_references_for_evaluation",
    "load_blind_source_only",
    "load_research_packet",
    "load_sqlite_cases",
    "main",
    "run_blind_frozen",
    "run_development_matrix",
    "run_facet_cases",
]

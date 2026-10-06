"""Offline audit and retrieval report for a proposed local-model run.

The command in this module never contacts a model endpoint and never changes a
run directory or the SQLite store.  It replays the raw rows in every receipt
through the source validator, checks exact source anchors and critical fields,
checks that proposed output did not cross an admission/terminal boundary, and
optionally measures candidate retrieval against the live official-object
corpus.

Example::

    uv run python -m paa.model_audit \
      --run data/llm_runs/corpus-v2-compact \
      --db data/paa.sqlite \
      --output /tmp/paa-model-audit.json

The retrieval result is candidate recall only.  A target missing from top-k is
not a negative relation judgment and cannot be rendered as absence.
"""


import argparse
import hashlib
import json
import sqlite3
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from paa.llm_run import load_corpus, revalidate_document
from paa.relations import ObjectRetriever, fingerprint

REPORT_SCHEMA_VERSION = "paa.model-audit.v1"
DEFAULT_RUN = Path("data/llm_runs/corpus-v2-compact")
DEFAULT_DB = Path("data/paa.sqlite")
DEFAULT_RELATION_GOLD = Path("paa/contracts/fixtures/llm_relation_gold.jsonl")
DEFAULT_SEMANTIC_GOLD = Path("paa/contracts/fixtures/llm_semantic_gold.jsonl")
DEFAULT_PROMPT = "extract_batch_multi_v2"

_TRANSPORT_RECEIPTS = frozenset({
    "FAILED",
    "TRUNCATED",
    "TIMEOUT",
    "NETWORK_ERROR",
    "CONNECTION_ERROR",
})
_PROTOCOL_RECEIPTS = frozenset({"INVALID_OUTPUT", "MALFORMED", "PROTOCOL_ERROR"})
_STATE_KEYS = frozenset({
    "admission_state",
    "validation_state",
    "review_state",
    "terminal_state",
    "terminal_eligibility",
    "finding_state",
})
_PROMOTION_VALUES = frozenset({
    "ADMITTED",
    "PUBLIC_ADMITTED",
    "ACCEPTED",
    "VERIFIED",
    "OBSERVED_ALIGNED_ACTION",
    "FULFILLED",
    "FAILED_TO_FULFILL",
    "ADMISSIBLE",
})
_CRITICAL_FIELDS = (
    "semantic_type",
    "source_quote",
    "source_start",
    "source_end",
    "negation",
    "issuer_scope",
    "target_quote",
    "condition_quote",
    "condition_inherited",
    "deadline_quote",
    "deadline_normalized",
    "deadline_basis",
    "action_kind",
    "required_capability",
    "observable_action",
    "validation_state",
)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"{type(exc).__name__}: {exc}"
    if not isinstance(value, dict):
        return None, "JSON root is not an object"
    return value, None


def _read_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    try:
        lines = path.read_text(encoding="utf-8").split("\n")
    except OSError as exc:
        return [], [f"{type(exc).__name__}: {exc}"]
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"line {line_number}: {exc}")
            continue
        if not isinstance(value, dict):
            errors.append(f"line {line_number}: JSON root is not an object")
            continue
        rows.append(value)
    return rows, errors


def _receipt_class(status: Any) -> str:
    value = str(status or "MISSING").upper()
    if value in {"MISSING", "PENDING"}:
        return "MISSING_RECEIPT"
    if value == "OK":
        return "COMPLETE_RESPONSE"
    if value in _TRANSPORT_RECEIPTS:
        return "TRANSPORT_FAILURE"
    if value in _PROTOCOL_RECEIPTS:
        return "MODEL_PROTOCOL_FAILURE"
    return "UNKNOWN_RECEIPT_STATUS"


def _counter(counter: Counter[Any]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items(), key=lambda item: str(item[0]))}


def _state_scan(value: Any, path: str, states: Counter[str], violations: list[str]) -> None:
    """Collect admission/terminal fields without treating status=VALID as promotion."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            child = f"{path}.{key}" if path else str(key)
            key_lower = str(key).casefold()
            if key_lower in _STATE_KEYS:
                state = str(item) if item is not None else "<NULL>"
                states[f"{key_lower}={state}"] += 1
                upper = state.upper()
                if upper in _PROMOTION_VALUES:
                    violations.append(f"{child}={state}")
                elif key_lower in {"admission_state", "validation_state"} and upper not in {
                    "PROPOSED",
                    "UNREVIEWED",
                    "<NULL>",
                }:
                    # A non-proposed value in one of the model admission
                    # fields is suspicious even when it uses a new spelling.
                    violations.append(f"{child}={state}")
            _state_scan(item, child, states, violations)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            _state_scan(item, f"{path}[{index}]", states, violations)


def _anchor_issues(prop: Mapping[str, Any], unit: Mapping[str, Any], document_text: str) -> list[str]:
    issues: list[str] = []
    quote = prop.get("source_quote")
    start = prop.get("source_start")
    end = prop.get("source_end")
    expected_quote = unit.get("text")
    if not isinstance(quote, str) or not quote:
        issues.append("SOURCE_QUOTE_MISSING")
    if quote != expected_quote:
        issues.append("SOURCE_QUOTE_NOT_CANONICAL_UNIT")
    if not isinstance(start, int) or not isinstance(end, int):
        issues.append("SOURCE_SPAN_NOT_INTEGER")
    elif start < 0 or end < start or end > len(document_text):
        issues.append("SOURCE_SPAN_OUT_OF_DOCUMENT")
    elif document_text[start:end] != quote:
        issues.append("SOURCE_SPAN_TEXT_MISMATCH")
    if start != unit.get("start") or end != unit.get("end"):
        issues.append("SOURCE_SPAN_NOT_CANONICAL_UNIT")
    return issues


def _critical_issues(prop: Mapping[str, Any]) -> list[str]:
    issues: list[str] = []
    for field in _CRITICAL_FIELDS:
        if field not in prop:
            issues.append(f"CRITICAL_FIELD_MISSING:{field}")
    if not isinstance(prop.get("semantic_type"), str) or not prop.get("semantic_type"):
        issues.append("CRITICAL_FIELD_INVALID:semantic_type")
    if not isinstance(prop.get("issuer_scope"), str) or not prop.get("issuer_scope"):
        issues.append("CRITICAL_FIELD_INVALID:issuer_scope")
    if not isinstance(prop.get("negation"), bool):
        issues.append("CRITICAL_FIELD_INVALID:negation")
    if not isinstance(prop.get("observable_action"), bool):
        issues.append("CRITICAL_FIELD_INVALID:observable_action")
    for field in ("source_quote", "target_quote", "condition_quote", "deadline_quote"):
        value = prop.get(field)
        if value is not None and not isinstance(value, str):
            issues.append(f"CRITICAL_FIELD_INVALID:{field}")
    if prop.get("validation_state") != "PROPOSED":
        issues.append("MODEL_VALIDATION_STATE_NOT_PROPOSED")
    return issues


def _load_receipts(run: Path) -> tuple[dict[str, dict[str, Any]], Counter[str], list[str], list[str]]:
    """Load receipt objects by embedded document ID without mutating them."""

    directory = run / "documents"
    by_id: dict[str, dict[str, Any]] = {}
    parse_errors: list[str] = []
    duplicate_ids: list[str] = []
    file_names: set[str] = set()
    for path in sorted(directory.glob("*.json")) if directory.is_dir() else []:
        file_names.add(path.name)
        value, error = _read_json(path)
        if error:
            parse_errors.append(f"{path.name}: {error}")
            continue
        assert value is not None
        document_id = value.get("document_id") or path.stem
        if not isinstance(document_id, str) or not document_id:
            parse_errors.append(f"{path.name}: missing document_id")
            continue
        if document_id in by_id:
            duplicate_ids.append(document_id)
            continue
        by_id[document_id] = value
    return by_id, Counter({"receipt_files": len(file_names)}), parse_errors, duplicate_ids


def _load_live_objects(db: Path) -> tuple[list[dict[str, Any]], list[str]]:
    objects: list[dict[str, Any]] = []
    errors: list[str] = []
    try:
        conn = sqlite3.connect(f"file:{db.resolve()}?mode=ro", uri=True)
    except sqlite3.Error as exc:
        return [], [f"{type(exc).__name__}: {exc}"]
    try:
        for row in conn.execute("SELECT object_id, json FROM official_objects ORDER BY object_id"):
            object_id, raw = row
            try:
                value = json.loads(raw)
            except (TypeError, json.JSONDecodeError) as exc:
                errors.append(f"{object_id}: {type(exc).__name__}: {exc}")
                continue
            if not isinstance(value, dict):
                errors.append(f"{object_id}: object JSON is not an object")
                continue
            objects.append(value)
    except sqlite3.Error as exc:
        errors.append(f"official_objects query: {type(exc).__name__}: {exc}")
    finally:
        conn.close()
    return objects, errors


def audit_retrieval(
    db: Path,
    relation_gold: Path = DEFAULT_RELATION_GOLD,
    *,
    topks: Sequence[int] = (1, 3, 5, 10),
) -> dict[str, Any]:
    """Measure candidate recall for all gold-positive pairs against live objects."""

    rows, gold_errors = _read_jsonl(relation_gold)
    status_counts = Counter(str(row.get("gold", {}).get("status", "<MISSING>")) for row in rows)
    split_counts = Counter(str(row.get("split", "<MISSING>")) for row in rows)
    if not db.exists():
        return {
            "status": "SKIPPED",
            "reason": f"database not found: {db}",
            "pairs_total": len(rows),
            "gold_status_counts": _counter(status_counts),
            "gold_split_counts": _counter(split_counts),
            "gold_read_errors": gold_errors,
        }
    objects, object_errors = _load_live_objects(db)
    object_by_id = {str(item.get("object_id")): item for item in objects if item.get("object_id")}
    positives = [row for row in rows if row.get("gold", {}).get("status") == "SAME_POLICY_OBJECT"]
    coverage_counts = Counter()
    ranks: dict[str, int | None] = {}
    retriever = ObjectRetriever(objects)
    max_k = max(topks, default=0)
    hits = Counter({int(k): 0 for k in topks})
    for row in positives:
        source = row.get("source") if isinstance(row.get("source"), Mapping) else {}
        obj = row.get("object") if isinstance(row.get("object"), Mapping) else {}
        source_quote = source.get("source_quote")
        object_id = obj.get("object_id")
        live = object_by_id.get(str(object_id))
        source_text = source.get("source_text")
        object_quote = obj.get("object_quote")
        source_hash = source.get("source_text_sha256") or source.get("source_sha256")
        object_hash = obj.get("object_sha256")
        if isinstance(source_quote, str) and isinstance(source_text, str) and source_quote in source_text:
            coverage_counts["source_quote_exact"] += 1
        if isinstance(source_hash, str) and isinstance(source_text, str) and fingerprint(source_text) == source_hash:
            coverage_counts["source_full_hash_match"] += 1
        if live is not None:
            coverage_counts["gold_object_id_present"] += 1
            live_text = live.get("text") or ""
            if isinstance(object_hash, str) and fingerprint(live_text) == object_hash:
                coverage_counts["live_object_full_hash_match"] += 1
            if isinstance(object_quote, str) and object_quote in live_text:
                coverage_counts["object_quote_exact"] += 1
        query = source_quote if isinstance(source_quote, str) else ""
        ranked = retriever.search(query, k=max_k) if query and max_k else []
        ranked_ids = [str(item.get("object_id")) for item in ranked]
        rank = ranked_ids.index(str(object_id)) + 1 if str(object_id) in ranked_ids else None
        ranks[str(row.get("pair_id", object_id))] = rank
        if rank is not None:
            for k in topks:
                if rank <= k:
                    hits[int(k)] += 1
    covered = sum(
        coverage_counts.get(key, 0) == len(positives)
        for key in (
            "source_quote_exact",
            "source_full_hash_match",
            "gold_object_id_present",
            "live_object_full_hash_match",
            "object_quote_exact",
        )
    )
    positive_coverage = {
        "denominator": len(positives),
        "counts": _counter(coverage_counts),
        "all_required_exact_anchors": (
            len(positives) > 0
            and all(
                coverage_counts.get(key, 0) == len(positives)
                for key in (
                    "source_quote_exact",
                    "source_full_hash_match",
                    "gold_object_id_present",
                    "live_object_full_hash_match",
                    "object_quote_exact",
                )
            )
        ),
        "complete_check_count": covered,
    }
    recall = {
        str(k): {
            "hits": int(hits[int(k)]),
            "denominator": len(positives),
            "recall": round(hits[int(k)] / len(positives), 6) if positives else None,
        }
        for k in topks
    }
    return {
        "status": "OK" if not gold_errors and not object_errors else "PARTIAL",
        "pairs_total": len(rows),
        "gold_status_counts": _counter(status_counts),
        "gold_split_counts": _counter(split_counts),
        "positive_status": "SAME_POLICY_OBJECT",
        "positive_denominator": len(positives),
        "positive_clause_coverage": positive_coverage,
        "recall_at_k": recall,
        "positive_ranks": ranks,
        "live_official_objects": len(objects),
        "live_object_parse_errors": object_errors,
        "gold_read_errors": gold_errors,
        "query": "exact source.source_quote only",
        "interpretation": "Candidate retrieval recall; missing from top-k is not relation absence or rejection.",
    }


def evaluation_limits(
    semantic_gold: Path = DEFAULT_SEMANTIC_GOLD,
    relation_gold: Path = DEFAULT_RELATION_GOLD,
) -> dict[str, Any]:
    semantic, semantic_errors = _read_jsonl(semantic_gold)
    relation, relation_errors = _read_jsonl(relation_gold)
    semantic_splits = Counter(str(row.get("split", "<MISSING>")) for row in semantic)
    relation_splits = Counter(str(row.get("split", "<MISSING>")) for row in relation)
    relation_status = Counter(str(row.get("gold", {}).get("status", "<MISSING>")) for row in relation)
    return {
        "semantic_gold": {
            "total": len(semantic),
            "splits": _counter(semantic_splits),
            "heldout_rows": int(semantic_splits.get("heldout", 0)),
            "development_rows": int(semantic_splits.get("development", 0)),
            "not_an_independent_model_benchmark": True,
            "note": "Only the heldout rows are 30 source-text-reviewed cases; this is project evaluation data, not an independently collected human/model benchmark or a corpus-wide precision estimate.",
            "read_errors": semantic_errors,
        },
        "relation_gold": {
            "total": len(relation),
            "splits": _counter(relation_splits),
            "status_counts": _counter(relation_status),
            "positive_denominator": int(relation_status.get("SAME_POLICY_OBJECT", 0)),
            "not_an_independent_model_benchmark": True,
            "note": "The 400 pair reviews are source-text evaluation fixtures; their labels do not establish model precision, population prevalence or absence outside the enumerated corpus.",
            "read_errors": relation_errors,
        },
    }


def audit_run(
    run: Path = DEFAULT_RUN,
    db: Path = DEFAULT_DB,
    *,
    prompt: str | None = None,
    relation_gold: Path | None = DEFAULT_RELATION_GOLD,
    semantic_gold: Path = DEFAULT_SEMANTIC_GOLD,
    include_retrieval: bool = True,
) -> dict[str, Any]:
    """Audit every expected document/unit in a model receipt directory."""

    specs = load_corpus(db)
    expected = {spec["document"]["document_id"]: spec for spec in specs}
    manifest, manifest_error = _read_json(run / "manifest.json")
    selected_prompt = prompt or (manifest or {}).get("identity", {}).get("prompt") or DEFAULT_PROMPT
    receipts, file_counter, parse_errors, duplicate_ids = _load_receipts(run)
    expected_ids = set(expected)
    receipt_ids = set(receipts)
    extra_ids = sorted(receipt_ids - expected_ids)
    missing_ids = sorted(expected_ids - receipt_ids)

    receipt_status = Counter()
    receipt_classes = Counter()
    cached_doc_status = Counter()
    revalidated_doc_status = Counter()
    cached_unit_status = Counter()
    revalidated_unit_status = Counter()
    cached_errors = Counter()
    revalidated_errors = Counter()
    semantic_counts = Counter()
    unit_counts = Counter({"expected": sum(len(spec["units"]) for spec in specs)})
    source_hash = Counter()
    anchor_counts = Counter()
    anchor_errors = Counter()
    critical_counts = Counter()
    critical_errors = Counter()
    classification_counts: dict[str, Counter[str]] = {
        "semantic_type": Counter(),
        "issuer_scope": Counter(),
        "action_kind": Counter(),
        "negation": Counter(),
        "observable_action": Counter(),
    }
    raw_revalidation = Counter()
    transport_failures = Counter()
    promotion_states = Counter()
    promotion_violations: list[str] = []

    # The manifest is part of the no-promotion audit, but not part of the
    # source-unit counts.
    if manifest is not None:
        _state_scan(manifest, "manifest", promotion_states, promotion_violations)
    for document_id, spec in expected.items():
        result = receipts.get(document_id)
        if result is None:
            semantic_counts["missing_document_receipt"] += 1
            unit_counts["missing_document_receipt"] += len(spec["units"])
            continue
        status = result.get("receipt_status", "MISSING")
        receipt_status[str(status)] += 1
        receipt_class = _receipt_class(status)
        receipt_classes[receipt_class] += 1
        if receipt_class == "TRANSPORT_FAILURE":
            transport_failures["documents"] += 1
            transport_failures["expected_units"] += len(spec["units"])
        elif receipt_class == "MODEL_PROTOCOL_FAILURE":
            transport_failures["protocol_failure_documents"] += 1
        cached_doc_status[str(result.get("status", "<MISSING>"))] += 1
        for error in result.get("invalid_records", []) if isinstance(result.get("invalid_records"), list) else []:
            if isinstance(error, Mapping):
                cached_errors[str(error.get("code", "<MISSING>"))] += 1
        _state_scan(result, f"document[{document_id}]", promotion_states, promotion_violations)

        canonical_text = spec["document"]["text"]
        expected_hash = _sha256(canonical_text)
        reported_hash = result.get("source_text_sha256")
        if reported_hash == expected_hash:
            source_hash["match"] += 1
        elif reported_hash is None:
            source_hash["missing"] += 1
        else:
            source_hash["mismatch"] += 1

        observed_units = result.get("units") if isinstance(result.get("units"), list) else []
        by_unit: dict[str, dict[str, Any]] = {}
        for observed in observed_units:
            if not isinstance(observed, Mapping):
                unit_counts["malformed_unit_record"] += 1
                continue
            unit_id = observed.get("unit_id")
            if not isinstance(unit_id, str):
                unit_counts["unit_id_missing"] += 1
                continue
            if unit_id in by_unit:
                unit_counts["duplicate_unit_receipt"] += 1
                continue
            by_unit[unit_id] = dict(observed)
            cached_unit_status[str(observed.get("status", "<MISSING>"))] += 1
            _state_scan(observed, f"document[{document_id}].unit[{unit_id}]", promotion_states, promotion_violations)
            prop = observed.get("proposition")
            if isinstance(prop, Mapping):
                for field, values in classification_counts.items():
                    values[str(prop.get(field, "<MISSING>"))] += 1
                _state_scan(prop, f"document[{document_id}].unit[{unit_id}].proposition", promotion_states, promotion_violations)
                issues = _anchor_issues(prop, next((u for u in spec["units"] if u["unit_id"] == unit_id), {}), canonical_text)
                anchor_counts["checked"] += 1
                if not issues:
                    anchor_counts["exact_valid"] += 1
                for issue in issues:
                    anchor_errors[issue] += 1
                cissues = _critical_issues(prop)
                critical_counts["checked"] += 1
                if not cissues:
                    critical_counts["valid"] += 1
                for issue in cissues:
                    critical_errors[issue] += 1
            elif observed.get("status") == "VALID":
                critical_counts["valid_without_proposition"] += 1
            else:
                anchor_counts["no_proposition"] += 1
                critical_counts["no_proposition"] += 1

        expected_unit_ids = {unit["unit_id"] for unit in spec["units"]}
        missing_units = expected_unit_ids - set(by_unit)
        unit_counts["unknown_unit_receipt"] += len(set(by_unit) - expected_unit_ids)
        unit_counts["missing_unit_receipt"] += len(missing_units)
        unit_counts["observed"] += len(by_unit)

        try:
            replayed = revalidate_document(result, spec, selected_prompt)
            raw_revalidation["documents_attempted"] += 1
        except (AttributeError, KeyError, TypeError, ValueError) as exc:  # pragma: no cover - old receipts
            raw_revalidation["exceptions"] += 1
            revalidated = None
            raw_revalidation[f"exception:{type(exc).__name__}"] += 1
        else:
            revalidated = replayed
            raw_revalidation["documents_succeeded"] += 1
        if revalidated is None:
            continue
        revalidated_doc_status[str(revalidated.get("status", "<MISSING>"))] += 1
        for error in revalidated.get("invalid_records", []) if isinstance(revalidated.get("invalid_records"), list) else []:
            if isinstance(error, Mapping):
                code = str(error.get("code", "<MISSING>"))
                revalidated_errors[code] += 1
        replay_units = {
            str(unit.get("unit_id")): unit
            for unit in revalidated.get("units", [])
            if isinstance(unit, Mapping) and isinstance(unit.get("unit_id"), str)
        }
        replay_status = str(revalidated.get("status", "<MISSING>"))
        if _receipt_class(status) == "COMPLETE_RESPONSE":
            if replay_status == "ABSTAIN":
                semantic_counts["semantic_abstention_documents"] += 1
            elif replay_status in {"INVALID", "PARTIAL"}:
                semantic_counts["semantic_validation_invalid_documents"] += 1
        for unit in spec["units"]:
            unit_id = unit["unit_id"]
            replay = replay_units.get(unit_id)
            if replay is None:
                continue
            replay_unit_status = str(replay.get("status", "<MISSING>"))
            revalidated_unit_status[replay_unit_status] += 1
            if _receipt_class(status) == "COMPLETE_RESPONSE":
                if replay_unit_status == "ABSTAIN":
                    semantic_counts["semantic_abstention_units"] += 1
                elif replay_unit_status == "INVALID":
                    semantic_counts["semantic_validation_invalid_units"] += 1
                elif replay_unit_status == "VALID":
                    semantic_counts["source_valid_units"] += 1
            if replay_unit_status == "VALID" and isinstance(replay.get("proposition"), Mapping):
                replay_prop = replay["proposition"]
                replay_anchor_issues = _anchor_issues(replay_prop, unit, canonical_text)
                replay_critical_issues = _critical_issues(replay_prop)
                if replay_anchor_issues:
                    anchor_counts["revalidated_invalid"] += 1
                if replay_critical_issues:
                    critical_counts["revalidated_invalid"] += 1
            elif replay_unit_status == "INVALID":
                anchor_counts["revalidated_invalid_units"] += 1
                critical_counts["revalidated_invalid_units"] += 1

    # A missing receipt is distinct from a model abstention.  It is also not a
    # source-semantic negative result.
    if missing_ids:
        receipt_classes["MISSING_RECEIPT"] += len(missing_ids)
    for code, count in cached_errors.items():
        if code in {"INVALID_MULTI_OUTPUT", "EMPTY_WITHOUT_ABSTENTION", "MISSING_UNIT_ROW"}:
            transport_failures[f"error_records:{code}"] += count

    # De-duplicate violation paths so a single bad field is not inflated by a
    # repeated traversal.  Keep a bounded list in the report for diagnostics.
    unique_violations = list(dict.fromkeys(promotion_violations))
    retrieval = (
        audit_retrieval(db, relation_gold)
        if include_retrieval and relation_gold is not None
        else {"status": "SKIPPED", "reason": "disabled"}
    )
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "run": str(run),
        "database": str(db),
        "prompt_version_replayed": selected_prompt,
        "manifest": {
            "present": manifest is not None,
            "read_error": manifest_error,
            "document_count": (manifest or {}).get("document_count"),
            "unit_count": (manifest or {}).get("unit_count"),
            "admission_state": (manifest or {}).get("admission_state"),
        },
        "population": {
            "expected_documents": len(specs),
            "expected_units": sum(len(spec["units"]) for spec in specs),
            "receipt_files": file_counter.get("receipt_files", 0),
            "receipt_documents_loaded": len(receipts),
            "missing_document_ids": len(missing_ids),
            "extra_document_ids": len(extra_ids),
            "malformed_receipts": len(parse_errors),
            "duplicate_document_ids": len(duplicate_ids),
        },
        "receipt_status": _counter(receipt_status),
        "receipt_class": _counter(receipt_classes),
        "cached_document_status": _counter(cached_doc_status),
        "revalidated_document_status": _counter(revalidated_doc_status),
        "cached_unit_status": _counter(cached_unit_status),
        "revalidated_unit_status": _counter(revalidated_unit_status),
        "source_hash": _counter(source_hash),
        "source_anchor": {
            "counts": _counter(anchor_counts),
            "errors": _counter(anchor_errors),
            "cached_proposition_audit_is_not_semantic_review": True,
        },
        "critical_fields": {
            "counts": _counter(critical_counts),
            "errors": _counter(critical_errors),
            "classification_counts": {
                field: _counter(values) for field, values in classification_counts.items()
            },
            "fields_checked": list(_CRITICAL_FIELDS),
        },
        "raw_revalidation": {
            **_counter(raw_revalidation),
            "error_codes": _counter(revalidated_errors),
        },
        "semantic_outcomes": {
            **_counter(semantic_counts),
            "transport_and_protocol_failures_are_not_semantic_abstentions": True,
        },
        "transport_failures": {
            **_counter(transport_failures),
            "definition": "Receipt-level FAILED/TRUNCATED/network/protocol failures; these are not semantic abstentions and are not semantic negatives.",
        },
        "unit_accounting": _counter(unit_counts),
        "terminal_no_promotion": {
            "passed": not unique_violations,
            "state_counts": _counter(promotion_states),
            "violation_count": len(unique_violations),
            "violation_paths": unique_violations[:100],
            "rule": "Model receipts may remain PROPOSED only; status=VALID means source/schema validation, not admission or observed action.",
        },
        "gold_evaluation_limits": evaluation_limits(semantic_gold, relation_gold or DEFAULT_RELATION_GOLD),
        "retrieval_benchmark": retrieval,
        "diagnostics": {
            "missing_document_ids_sample": missing_ids[:20],
            "extra_document_ids_sample": extra_ids[:20],
            "receipt_parse_errors_sample": parse_errors[:20],
            "duplicate_document_ids_sample": duplicate_ids[:20],
            "cached_error_codes": _counter(cached_errors),
        },
    }
    return report


def _write_report(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--prompt", help="prompt version used for raw-row replay; defaults to the run manifest")
    parser.add_argument("--relation-gold", type=Path, default=DEFAULT_RELATION_GOLD)
    parser.add_argument("--semantic-gold", type=Path, default=DEFAULT_SEMANTIC_GOLD)
    parser.add_argument("--skip-retrieval", action="store_true")
    parser.add_argument("--output", type=Path, help="optional path for the deterministic JSON report")
    args = parser.parse_args(argv)
    report = audit_run(
        args.run,
        args.db,
        prompt=args.prompt,
        relation_gold=args.relation_gold,
        semantic_gold=args.semantic_gold,
        include_retrieval=not args.skip_retrieval,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        _write_report(args.output, report)
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

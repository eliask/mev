"""Run source-grounded relation proposals and reviews on the frozen fixtures.

This module is intentionally a runner, not an admission path.  It presents
only :func:`paa.llm_evaluation.public_relation_records` to the model, keeps
the adjudication rows in a separate local map for scoring, and writes a
replayable receipt for every pair.  A proposal is sent to the separate-context
verifier only when its validated status is an admitted relation candidate.

The command is deliberately resumable at pair granularity.  Existing rows are
reused only when the public source/object identity, prompt/schema hashes, and
served model identity all match.  The runner never contacts a non-loopback
endpoint; ``LocalLLMClient`` enforces that restriction as well.
"""


import argparse
import asyncio
import hashlib
import json
import time
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from paa.llm_client import LocalLLMClient, digest
from paa.llm_evaluation import (
    RELATION_GOLD_PATH,
    evaluate_relation_predictions,
    load_relation_gold,
    public_relation_records,
)
from paa.llm_relations import (
    MODEL_PROJECTION_VERSION,
    VALIDATOR_VERSION,
    batch_proposal_schema,
    batch_verification_schema,
    build_batch_relation_request,
    build_batch_verification_request,
    fingerprint,
    load_prompt,
    validate_batch_relation_response,
    validate_batch_verification_response,
)

DEFAULT_BATCH_SIZE = 8
DEFAULT_MAX_TOKENS = 6400
RELATION_CANDIDATE_STATUSES = frozenset({"SAME_POLICY_OBJECT", "SAME_MATTER"})
ABSTENTION_STATUSES = frozenset({"UNRESOLVED", "UNKNOWN", "ABSTAIN", "MISSING"})


def _json(value: Any) -> str:
    """Serialize model input without whitespace so receipts are stable."""

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    text = "".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for row in rows)
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def load_relation_input(path: str | Path) -> list[dict[str, Any]]:
    """Load native source/object pair records from JSONL or a JSON array.

    Native records are intentionally not interpreted as adjudication data.  A
    row may use the public fixture shape (``source``/``object``) or canonical
    relation inputs (``proposition``/``statement``/``official_object``).
    """

    path = Path(path)
    raw = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        value = json.loads(raw)
        if not isinstance(value, list):
            raise ValueError(f"{path}: expected a JSON array")
        rows = value
    else:
        rows = []
        # JSONL records are delimited by LF.  ``str.splitlines()`` also
        # treats U+2028/U+2029 as line boundaries, but those characters can
        # legitimately occur inside a quoted source excerpt (notably in
        # Finnish campaign text).  Splitting only on LF keeps the JSON record
        # intact and preserves the source bytes supplied to the model.
        for line_number, line in enumerate(raw.split("\n"), 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            rows.append(value)
    if any(not isinstance(row, Mapping) for row in rows):
        raise ValueError(f"{path}: every input row must be an object")
    result = [dict(row) for row in rows]
    if not result:
        raise ValueError(f"{path}: no relation input rows")
    return result


def _chunked(values: Sequence[Mapping[str, Any]], size: int) -> Iterable[list[Mapping[str, Any]]]:
    if size < 1 or size > 8:
        raise ValueError("batch_size must be between 1 and 8")
    for start in range(0, len(values), size):
        yield list(values[start : start + size])


def _relation_batches(
    values: Sequence[Mapping[str, Any]],
    size: int,
    *,
    group_by_proposition: bool = False,
) -> Iterable[list[Mapping[str, Any]]]:
    """Yield model batches, optionally keeping one proposition's objects together.

    Native retrieval deliberately yields several official objects for the same
    source proposition.  Keeping those rows together lets the request builder
    share the source context and makes the model normalize the source target
    once before comparing the three objects.  Rows without a canonical
    proposition identity remain ordinary singleton groups rather than being
    guessed together by lexical similarity.
    """

    if not group_by_proposition:
        yield from _chunked(values, size)
        return
    for rows in _proposition_groups(values):
        yield from _chunked(rows, size)


def _proposition_groups(values: Sequence[Mapping[str, Any]]) -> list[list[Mapping[str, Any]]]:
    """Return stable proposition groups without splitting their official objects."""

    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for index, row in enumerate(values):
        proposition = row.get("proposition") if isinstance(row, Mapping) else None
        statement = row.get("statement") if isinstance(row, Mapping) else None
        proposition_id = proposition.get("proposition_id") if isinstance(proposition, Mapping) else None
        statement_id = statement.get("statement_id") if isinstance(statement, Mapping) else None
        if not proposition_id:
            source = row.get("source") if isinstance(row, Mapping) else None
            proposition_id = source.get("source_quote") if isinstance(source, Mapping) else None
        if not statement_id:
            source = row.get("source") if isinstance(row, Mapping) else None
            statement_id = source.get("document_id") if isinstance(source, Mapping) else None
        key = (str(statement_id or ""), str(proposition_id or f"row-{index}"))
        groups.setdefault(key, []).append(row)
    return list(groups.values())


def _relation_partitions(
    values: Sequence[Mapping[str, Any]],
    concurrency: int,
    *,
    group_by_proposition: bool,
) -> list[list[Mapping[str, Any]]]:
    """Partition rows while keeping each proposition's objects together."""

    if concurrency < 1:
        raise ValueError("concurrency must be positive")
    if not group_by_proposition:
        return [
            list(values[start::concurrency])
            for start in range(concurrency)
            if values[start::concurrency]
        ]
    partitions = [[] for _ in range(concurrency)]
    for index, group in enumerate(_proposition_groups(values)):
        partitions[index % concurrency].extend(group)
    return [partition for partition in partitions if partition]


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _evidence_id(prefix: str, value: Any) -> str:
    return f"fixture-{prefix}:{fingerprint(str(value))}"


def _evidence_records(record: Mapping[str, Any], fallback_id: str, fallback_quote: str) -> list[dict[str, Any]]:
    """Normalize optional public evidence objects without copying gold data."""

    raw = record.get("evidence") or record.get("evidence_objects") or record.get("evidence_records")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raw = []
    result: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        evidence_id = item.get("evidence_id") or item.get("id") or item.get("source_id")
        if not evidence_id:
            continue
        evidence = dict(item)
        evidence["evidence_id"] = str(evidence_id)
        if not evidence.get("quote") and isinstance(evidence.get("text"), str):
            evidence["quote"] = evidence["text"]
        result.append(evidence)
    explicit_ids = record.get("evidence_ids") or record.get("source_evidence_ids") or record.get("object_evidence_ids")
    if isinstance(explicit_ids, Sequence) and not isinstance(explicit_ids, (str, bytes)):
        for evidence_id in explicit_ids:
            if isinstance(evidence_id, str) and evidence_id.strip() and not any(item["evidence_id"] == evidence_id for item in result):
                result.append({"evidence_id": evidence_id.strip()})
    if not result:
        result = [{"evidence_id": fallback_id, "quote": fallback_quote}]
    elif not any(item["evidence_id"] == fallback_id for item in result):
        # The fallback is an excerpt/hash-bound local anchor only when the
        # public fixture did not expose an upstream evidence object.
        pass
    return result


def public_record_to_relation_pair(record: Mapping[str, Any]) -> dict[str, Any]:
    """Adapt a public fixture record to ``build_relation_request`` input.

    The fixture may store a focused source clause and an official-object
    excerpt rather than the repository's canonical proposition object.  When
    full ``source_text``/``object_text`` fields are present they are supplied
    as context while the focused quotes remain exact anchors; otherwise the
    runner marks the request's coverage as excerpt-only.  No gold status,
    selection stratum, or adjudication field is copied into the model request.
    """

    if not isinstance(record, Mapping):
        raise TypeError("relation record must be an object")
    pair_id = _required_text(record.get("pair_id"), "pair_id")
    if all(isinstance(record.get(key), Mapping) for key in ("proposition", "statement", "official_object")):
        # Native corpus mode: callers may provide the canonical source/object
        # records directly.  The adapter still removes semantic adjudication
        # labels before generation and keeps the caller's full evidence context.
        proposition = dict(record["proposition"])
        proposition["semantic_type"] = "UNRESOLVED"
        proposition.pop("gold", None)
        statement = dict(record["statement"])
        official_object = dict(record["official_object"])
        source_text = _required_text(statement.get("original_text") or statement.get("text"), f"{pair_id}.statement.original_text")
        proposition_text = _required_text(proposition.get("source_text") or source_text, f"{pair_id}.proposition.source_text")
        object_text = _required_text(official_object.get("text") or official_object.get("normalized_text") or official_object.get("title"), f"{pair_id}.official_object.text")
        if proposition_text not in source_text:
            raise ValueError(f"{pair_id}: proposition source_text is not an exact span of statement text")
        if "proposition_id" not in proposition:
            proposition["proposition_id"] = f"{pair_id}:proposition"
        if "statement_id" not in statement and "document_version_id" in statement:
            statement["statement_id"] = statement["document_version_id"]
        if not statement.get("statement_id") or not official_object.get("object_id") or not official_object.get("matter_id"):
            raise ValueError(f"{pair_id}: canonical relation pair lacks statement/object identity")
        return {
            "pair_id": pair_id,
            "proposition": proposition,
            "statement": statement,
            "official_object": official_object,
            "alternatives": record.get("alternatives") or official_object.get("alternatives") or [],
            "_receipt_context": {
                "source_document_id": statement.get("statement_id"),
                "source_id": statement.get("source_id"),
                "source_year": statement.get("source_year"),
                "language": statement.get("language"),
                "object_id": official_object.get("object_id"),
                "matter_id": official_object.get("matter_id"),
                "object_year": official_object.get("object_year"),
                "statement_context_complete": True,
                "object_context_complete": True,
                "statement_focus_quote": proposition_text,
                "object_focus_quote": record.get("object_quote") or object_text[:320],
            },
        }
    source = record.get("source")
    official = record.get("object")
    if not isinstance(source, Mapping) or not isinstance(official, Mapping):
        raise TypeError(f"{pair_id}: public relation record lacks source/object")
    source_focus = _required_text(source.get("source_quote"), f"{pair_id}.source.source_quote")
    source_text = _required_text(
        source.get("source_text") or source.get("original_text") or source.get("full_text") or source_focus,
        f"{pair_id}.source.source_text",
    )
    object_focus = _required_text(official.get("object_quote"), f"{pair_id}.object.object_quote")
    object_text = _required_text(
        official.get("object_text") or official.get("full_text") or official.get("text") or object_focus,
        f"{pair_id}.object.object_text",
    )
    if source_focus not in source_text:
        raise ValueError(f"{pair_id}: source_quote is not an exact span of source_text")
    if object_focus not in object_text:
        raise ValueError(f"{pair_id}: object_quote is not an exact span of object_text")
    statement_evidence = _evidence_id("statement", source.get("document_id") or pair_id)
    object_evidence = _evidence_id("object", official.get("object_id") or pair_id)
    source_evidence = _evidence_records(source, statement_evidence, source_focus)
    object_evidence_records = _evidence_records(official, object_evidence, object_focus)
    statement_evidence_ids = [item["evidence_id"] for item in source_evidence]
    object_evidence_ids = [item["evidence_id"] for item in object_evidence_records]
    candidate = source.get("candidate_context")
    candidate = candidate if isinstance(candidate, Mapping) else {}
    actor_id = candidate.get("actor_id")
    actor_name = candidate.get("display_name")
    actor_context = official.get("actor_context")
    actor_context = actor_context if isinstance(actor_context, Mapping) else {}
    authors: list[dict[str, Any]] = []
    person_id = actor_context.get("person_id")
    role = actor_context.get("role")
    if person_id or actor_context.get("matched_name") or role:
        authors.append({
            "person_id": person_id,
            "name": actor_context.get("matched_name"),
            "role": role or "UNRESOLVED",
            "evidence_ids": object_evidence_ids[:1],
        })
    proposition: dict[str, Any] = {
        "proposition_id": f"{pair_id}:fixture-proposition",
        "source_text": source_focus,
        # This is deliberately not the fixture's gold semantic type.  Relation
        # generation receives no semantic adjudication label.
        "semantic_type": "UNRESOLVED",
        "issuer_scope": "UNRESOLVED",
        "subject_actor_ids": [str(actor_id)] if actor_id else [],
        "evidence_ids": statement_evidence_ids,
    }
    statement = {
        "statement_id": _required_text(source.get("document_id"), f"{pair_id}.source.document_id"),
        "source_id": str(source.get("source_id") or "fixture-source"),
        "source_field_label": str(source.get("original_question") or ""),
        "language": str(source.get("language") or ""),
        "original_text": source_text,
        "question_text": str(source.get("original_question") or ""),
        "evidence": source_evidence,
    }
    official_object = {
        "object_id": _required_text(official.get("object_id"), f"{pair_id}.object.object_id"),
        "matter_id": _required_text(official.get("matter_id"), f"{pair_id}.object.matter_id"),
        "kind": str(official.get("kind") or "UNRESOLVED"),
        "title": str(official.get("title") or ""),
        "text": object_text,
        "date": str(official.get("publication_date") or ""),
        "action_date": str(official.get("action_date") or ""),
        "source_id": "fixture-official-object",
        "authors": authors,
        "evidence": object_evidence_records,
        "evidence_ids": object_evidence_ids,
    }
    # Keep source/object metadata available to the runner for receipts, but do
    # not add it to any model-facing mapping returned by build_relation_request.
    return {
        "pair_id": pair_id,
        "proposition": proposition,
        "statement": statement,
        "official_object": official_object,
        "alternatives": [],
        "_receipt_context": {
            "source_document_id": source.get("document_id"),
            "source_id": source.get("source_id"),
            "source_year": source.get("source_year"),
            "language": source.get("language"),
            "object_id": official.get("object_id"),
            "matter_id": official.get("matter_id"),
            "object_year": official.get("object_year"),
            "candidate_actor_id": actor_id,
            "candidate_display_name": actor_name,
            "official_actor_name": actor_context.get("matched_name"),
            "official_actor_role": actor_context.get("role"),
            "statement_context_complete": bool(source.get("source_text") or source.get("original_text") or source.get("full_text")),
            "object_context_complete": bool(official.get("object_text") or official.get("full_text") or official.get("text")),
            "statement_focus_quote": source_focus,
            "object_focus_quote": object_focus,
        },
    }


def _manifest_model_id(manifest: Mapping[str, Any] | None) -> str | None:
    if not isinstance(manifest, Mapping):
        return None
    value = manifest.get("model_id")
    return str(value) if value is not None else None


def _source_hashes(gold_row: Mapping[str, Any] | None, public_record: Mapping[str, Any]) -> tuple[str, str]:
    source = gold_row.get("source") if isinstance(gold_row, Mapping) else None
    obj = gold_row.get("object") if isinstance(gold_row, Mapping) else None
    source_hash = source.get("source_sha256") if isinstance(source, Mapping) else None
    object_hash = obj.get("object_sha256") if isinstance(obj, Mapping) else None
    public_source = public_record.get("source") or {}
    public_object = public_record.get("object") or {}
    if not isinstance(public_source, Mapping) or not public_source:
        public_source = public_record.get("statement") or {}
    if not isinstance(public_object, Mapping) or not public_object:
        public_object = public_record.get("official_object") or {}
    source_hash = source_hash or public_source.get("source_sha256")
    object_hash = object_hash or public_object.get("object_sha256")
    source_text = public_source.get("source_text") or public_source.get("original_text") or public_source.get("full_text") or public_source.get("text") or ""
    object_text = public_object.get("object_text") or public_object.get("full_text") or public_object.get("text") or public_object.get("normalized_text") or ""
    return (
        str(source_hash or hashlib.sha256(str(source_text).encode("utf-8")).hexdigest()),
        str(object_hash or hashlib.sha256(str(object_text).encode("utf-8")).hexdigest()),
    )


def _record_source_context(public_record: Mapping[str, Any], gold_row: Mapping[str, Any] | None) -> dict[str, Any]:
    source = public_record.get("source") if isinstance(public_record.get("source"), Mapping) else {}
    obj = public_record.get("object") if isinstance(public_record.get("object"), Mapping) else {}
    if not source and isinstance(public_record.get("statement"), Mapping):
        source = public_record["statement"]
    if not obj and isinstance(public_record.get("official_object"), Mapping):
        obj = public_record["official_object"]
    source_hash, object_hash = _source_hashes(gold_row, public_record)
    return {
        "source_document_id": source.get("document_id"),
        "source_id": source.get("source_id"),
        "source_year": source.get("source_year"),
        "language": source.get("language"),
        "source_sha256": source_hash,
        "object_id": obj.get("object_id"),
        "matter_id": obj.get("matter_id"),
        "object_year": obj.get("object_year"),
        "object_sha256": object_hash,
        "statement_context_complete": bool(source.get("source_text") or source.get("original_text") or source.get("full_text") or source.get("text")),
        "object_context_complete": bool(obj.get("object_text") or obj.get("full_text") or obj.get("text") or obj.get("normalized_text")),
    }


def _model_input_hash(public_record: Mapping[str, Any]) -> str:
    source = public_record.get("source") or public_record.get("statement")
    official = public_record.get("object") or public_record.get("official_object")
    proposition = public_record.get("proposition")
    return digest({
        "pair_id": public_record.get("pair_id"),
        "source": source,
        "object": official,
        "proposition": proposition,
    })


def _model_projection_hash(public_record: Mapping[str, Any], *, prompt_version: str) -> str:
    """Hash the actual single-pair model projection used for resumability.

    ``_model_input_hash`` protects the canonical input record, but it cannot
    detect a projection-only change such as author-roster compaction or a
    coverage annotation policy.  Build the same deterministic request shape
    used by the batch runner (with one pair), apply the receipt-facing
    coverage overlay, and hash only the model-visible projection plus its
    explicit version.  Grouped batches may omit a duplicated statement, but
    the single-pair projection is the stable per-pair contract whose changes
    must invalidate old receipts.
    """

    try:
        relation_pair = public_record_to_relation_pair(public_record)
        request = build_batch_relation_request(
            [relation_pair],
            max_pairs=1,
            max_statement_chars=4200,
            max_object_chars=3200,
            prompt_version=prompt_version,
        )
        _apply_context_coverage(request, [relation_pair])
        model_input = request.get("model_input")
        if not isinstance(model_input, Mapping):
            raise TypeError("model projection lacks model_input")
        return digest({
            "projection_version": MODEL_PROJECTION_VERSION,
            "model_input": model_input,
        })
    except Exception as exc:  # noqa: BLE001 - invalid rows remain pending
        # Keep identity construction total so one malformed native row is
        # persisted as an invalid receipt rather than aborting the whole run.
        return digest({
            "projection_version": MODEL_PROJECTION_VERSION,
            "projection_error": f"{type(exc).__name__}: {exc}",
        })


def _safe_model_manifest(manifest: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(manifest, Mapping):
        return None
    # The manifest is already local metadata, but keep receipts compact and do
    # not copy arbitrary future server fields into the public report.
    keys = (
        "client_version", "endpoint", "model_id", "model_digest", "model_path",
        "model_ftype", "chat_template_sha256", "context_length", "server_build",
    )
    return {key: manifest.get(key) for key in keys if key in manifest}


def _response_parsed(response: Mapping[str, Any]) -> Mapping[str, Any] | None:
    parsed = response.get("parsed")
    if isinstance(parsed, Mapping):
        return parsed
    content = response.get("content")
    if isinstance(content, str) and content.strip():
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, Mapping) else None
    return None


def _stage_receipt(
    request: Mapping[str, Any],
    response: Mapping[str, Any] | None,
    validation: Mapping[str, Any] | None,
    *,
    manifest: Mapping[str, Any] | None,
) -> dict[str, Any]:
    response = response if isinstance(response, Mapping) else {}
    result = {
        "request_id": response.get("request_id") or request.get("request_id"),
        "envelope_request_id": request.get("request_id"),
        "status": response.get("status", "NOT_RUN"),
        "prompt_version": request.get("prompt_version"),
        "prompt_sha256": request.get("prompt_sha256"),
        "schema_sha256": request.get("schema_sha256"),
        "model_id": _manifest_model_id(manifest),
        "cache_hit": bool(response.get("cache_hit", False)),
    }
    if isinstance(validation, Mapping):
        result["validation"] = {
            "valid": bool(validation.get("valid")),
            "abstained": bool(validation.get("abstained")),
            "errors": list(validation.get("errors") or []),
            "warnings": list(validation.get("warnings") or []),
        }
        proposal = validation.get("proposal")
        if isinstance(proposal, Mapping):
            result["proposal"] = dict(proposal)
        if validation.get("verdict") is not None:
            result["verdict"] = validation.get("verdict")
        if validation.get("independence") is not None:
            result["independence"] = dict(validation["independence"])
    if response.get("error") is not None:
        result["error"] = str(response.get("error"))
    return result


def _gold_map(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        pair_id = _required_text(row.get("pair_id"), "gold.pair_id")
        if pair_id in result:
            raise ValueError(f"duplicate relation gold pair_id: {pair_id}")
        result[pair_id] = row
    return result


def _receipt_identity(
    public_record: Mapping[str, Any],
    gold_row: Mapping[str, Any] | None,
    *,
    prompt_version: str,
    manifest: Mapping[str, Any] | None,
) -> dict[str, Any]:
    source_hash, object_hash = _source_hashes(gold_row, public_record)
    manifest_fingerprint = fingerprint(json.dumps(dict(manifest or {}), sort_keys=True, ensure_ascii=False, default=str))
    return {
        "pair_id": public_record.get("pair_id"),
        "input_sha256": _model_input_hash(public_record),
        "model_projection_version": MODEL_PROJECTION_VERSION,
        "model_projection_sha256": _model_projection_hash(public_record, prompt_version=prompt_version),
        "source_sha256": source_hash,
        "object_sha256": object_hash,
        "prompt_version": prompt_version,
        "proposal_prompt_sha256": fingerprint(load_prompt(prompt_version, stage="propose")),
        "proposal_schema_sha256": fingerprint(json.dumps(batch_proposal_schema(prompt_version), sort_keys=True, ensure_ascii=False)),
        "verification_prompt_sha256": fingerprint(load_prompt(prompt_version, stage="verify")),
        "verification_schema_sha256": fingerprint(json.dumps(batch_verification_schema(prompt_version), sort_keys=True, ensure_ascii=False)),
        "validator_version": VALIDATOR_VERSION,
        "model_id": _manifest_model_id(manifest),
        "served_manifest_sha256": manifest_fingerprint,
    }


def _is_resumable(receipt: Mapping[str, Any], identity: Mapping[str, Any]) -> bool:
    if not isinstance(receipt, Mapping) or not isinstance(receipt.get("prediction"), Mapping):
        return False
    stored = receipt.get("identity")
    if not isinstance(stored, Mapping):
        return False
    for key, value in identity.items():
        if stored.get(key) != value:
            return False
    proposal = receipt.get("proposal")
    if not isinstance(proposal, Mapping) or proposal.get("status") not in {"OK", "VALID"}:
        return False
    validation = proposal.get("validation")
    if not isinstance(validation, Mapping) or not validation.get("valid"):
        return False
    run = receipt.get("run")
    if not isinstance(run, Mapping) or run.get("status") != "COMPLETE":
        return False
    proposed = proposal.get("proposal")
    if isinstance(proposed, Mapping) and proposed.get("status") in RELATION_CANDIDATE_STATUSES:
        verification = receipt.get("verification")
        if not isinstance(verification, Mapping) or verification.get("status") not in {"OK", "VALID"}:
            return False
        verification_validation = verification.get("validation")
        if not isinstance(verification_validation, Mapping) or not verification_validation.get("valid"):
            return False
    return True


def _proposal_from_validation(validation: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if not isinstance(validation, Mapping) or not validation.get("valid"):
        return None
    proposal = validation.get("proposal")
    return proposal if isinstance(proposal, Mapping) else None


def _actor_eligibility(public_record: Mapping[str, Any], proposal: Mapping[str, Any] | None) -> str:
    source = public_record.get("source") if isinstance(public_record.get("source"), Mapping) else {}
    obj = public_record.get("object") if isinstance(public_record.get("object"), Mapping) else {}
    candidate = source.get("candidate_context") if isinstance(source.get("candidate_context"), Mapping) else {}
    context = obj.get("actor_context") if isinstance(obj.get("actor_context"), Mapping) else {}
    role = str(context.get("role") or "").upper()
    candidate_name = str(candidate.get("display_name") or "").strip().casefold()
    official_name = str(context.get("matched_name") or "").strip().casefold()
    if candidate_name and official_name and candidate_name != official_name:
        return "NO_IDENTIFIED_MATCH"
    if role in {"AUTHOR", "FIRST_AUTHOR"} and candidate_name and official_name:
        return "MATCHED_AUTHOR"
    if role == "COSIGNER" and candidate_name and official_name:
        return "COSIGNER_ONLY"
    return "NO_IDENTIFIED_MATCH"


def _time_eligibility(public_record: Mapping[str, Any]) -> str:
    source = public_record.get("source") if isinstance(public_record.get("source"), Mapping) else {}
    obj = public_record.get("object") if isinstance(public_record.get("object"), Mapping) else {}
    source_year = source.get("source_year")
    object_year = obj.get("object_year")
    if not isinstance(source_year, int) or not isinstance(object_year, int):
        return "UNRESOLVED"
    if object_year > source_year:
        return "AFTER_SOURCE_YEAR"
    if object_year < source_year:
        return "BEFORE_SOURCE_YEAR"
    return "SAME_SOURCE_YEAR"


def _candidate_prediction(
    public_record: Mapping[str, Any],
    proposal: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Keep relation status separate from actor/time terminal eligibility."""

    proposal = proposal if isinstance(proposal, Mapping) else {}
    status = str(proposal.get("status") or "UNRESOLVED").upper()
    action = str(proposal.get("action_alignment") or "UNRESOLVED").upper()
    actor = _actor_eligibility(public_record, proposal)
    time_eligibility = _time_eligibility(public_record)
    if status in RELATION_CANDIDATE_STATUSES:
        domain = "SAME"
    elif status == "REJECTED":
        # The model rejected this supplied pair, but that is not a claim that
        # the domains are wholly unrelated; retain a conservative unknown.
        domain = "UNKNOWN"
    else:
        domain = "UNKNOWN"
    if status in RELATION_CANDIDATE_STATUSES:
        # Policy identity alone is not a testable personal commitment.  Only
        # an explicit observable action alignment can open an action-terminal
        # candidate; RELATED/UNRESOLVED remains NOT_TESTABLE.
        if action in {"RELATED", "UNRESOLVED"}:
            terminal = "NOT_TESTABLE"
        elif actor == "COSIGNER_ONLY":
            terminal = "ADMISSIBLE_WITH_COSIGNER_CAVEAT"
        elif actor == "MATCHED_AUTHOR" and time_eligibility != "UNRESOLVED":
            terminal = "ADMISSIBLE"
        else:
            terminal = "ABSTAIN"
    elif status == "REJECTED":
        terminal = "NOT_ADMISSIBLE"
    else:
        terminal = "ABSTAIN"
    return {
        "action_alignment": action,
        "actor_eligibility": actor,
        "time_eligibility": time_eligibility,
        "domain_relation": domain,
        "terminal_eligibility": terminal,
    }


def _prediction_from_proposal(public_record: Mapping[str, Any], proposal: Mapping[str, Any] | None) -> dict[str, Any]:
    proposal = proposal if isinstance(proposal, Mapping) else {}
    result = {
        "status": str(proposal.get("status") or "UNRESOLVED").upper(),
        "matter_id": proposal.get("matter_id"),
        "identity_basis": proposal.get("identity_basis"),
        "statement_quote": proposal.get("statement_quote"),
        "object_quote": proposal.get("object_quote"),
        "normalized_target": proposal.get("normalized_target"),
        "target_scope": proposal.get("target_scope") or "UNRESOLVED",
        "action_alignment": proposal.get("action_alignment") or "UNRESOLVED",
        "candidate": _candidate_prediction(public_record, proposal),
    }
    # Keep the fields flat too for consumers that predate the candidate
    # envelope.  The evaluator reads the candidate values first.
    result.update(result["candidate"])
    return result


def _response_validation(
    stage: str,
    response: Mapping[str, Any],
    request: Mapping[str, Any],
) -> dict[str, Any]:
    if response.get("status") not in {"OK", "VALID"}:
        return {
            "valid": False,
            "errors": [f"{stage} client status is {response.get('status', 'MISSING')}"],
            "warnings": [],
            "results": [],
        }
    parsed = _response_parsed(response)
    if parsed is None:
        return {
            "valid": False,
            "errors": [f"{stage} response has no parsed JSON object"],
            "warnings": [],
            "results": [],
        }
    if stage == "proposal":
        return validate_batch_relation_response(parsed, request)
    return validate_batch_verification_response(parsed, request)


def _apply_context_coverage(
    batch_request: dict[str, Any],
    relation_pairs: Sequence[Mapping[str, Any]],
) -> None:
    """Attach model-clip coverage and raw-original availability separately."""

    pair_flags = {
        str(pair["pair_id"]): dict(pair.get("_receipt_context") or {})
        for pair in relation_pairs
    }
    validation_requests = batch_request.get("validation_requests")
    model_input = batch_request.get("model_input")
    model_pairs = model_input.get("pairs") if isinstance(model_input, Mapping) else None
    for pair_id, request in (validation_requests.items() if isinstance(validation_requests, Mapping) else []):
        flags = pair_flags.get(str(pair_id), {})
        coverage = dict(request.get("coverage") or {})
        # ``statement_context_complete`` and ``object_context_complete`` are
        # receipt-facing model-coverage fields.  The request builder already
        # calculated these from the clipped text; do not replace them with
        # the mere existence of a longer raw source/object record.
        statement_model_complete = bool(coverage.get("statement_complete"))
        object_model_complete = bool(coverage.get("object_complete"))
        originals_available = {
            "statement": bool(flags.get("statement_context_complete")),
            "object": bool(flags.get("object_context_complete")),
        }
        statement_original_complete = statement_model_complete and originals_available["statement"]
        object_original_complete = object_model_complete and originals_available["object"]
        coverage.update({
            # ``*_clip_complete`` records what was submitted to the model;
            # ``*_context_complete`` is reserved for a complete original
            # source, never a short fixture excerpt that merely fits the clip.
            "statement_clip_complete": statement_model_complete,
            "object_clip_complete": object_model_complete,
            "statement_context_complete": statement_original_complete,
            "object_context_complete": object_original_complete,
            "source_context_kind": "FULL_SOURCE_TEXT" if statement_original_complete else "FOCUSED_SOURCE_QUOTE_ONLY",
            "official_context_kind": "FULL_OFFICIAL_OBJECT_TEXT" if object_original_complete else "OFFICIAL_OBJECT_EXCERPT_ONLY",
            "originals_available": originals_available,
        })
        request["coverage"] = coverage
        nested_input = request.get("model_input")
        if isinstance(nested_input, Mapping):
            nested_input["coverage"] = coverage
        if isinstance(model_pairs, list):
            for item in model_pairs:
                if isinstance(item, Mapping) and str(item.get("pair_id")) == str(pair_id):
                    item_coverage = dict(item.get("coverage") or {})
                    item_coverage.update(coverage)
                    item["coverage"] = item_coverage
                    pair_input = item.get("input")
                    if isinstance(pair_input, Mapping):
                        pair_input["coverage"] = item_coverage


async def _request_batch(
    client: Any,
    request: Mapping[str, Any],
    *,
    stage: str,
    max_tokens: int,
) -> dict[str, Any]:
    task = f"paa-relation-{stage}:{request['request_id']}"
    try:
        response = await client.request(
            task,
            str(request["model_prompt"]),
            _json(request["model_input"]),
            schema=request["output_schema"],
            max_tokens=max_tokens,
            cache=True,
            seed=42,
        )
    except Exception as exc:  # noqa: BLE001 - preserve one receipt per endpoint failure
        return {"status": "FAILED", "error": f"{type(exc).__name__}: {exc}"}
    return dict(response) if isinstance(response, Mapping) else {"status": "INVALID_OUTPUT", "error": "client returned non-object"}


def _empty_stage(request: Mapping[str, Any], *, manifest: Mapping[str, Any] | None, status: str, error: str | None = None) -> dict[str, Any]:
    stage = {
        "request_id": request.get("request_id"),
        "envelope_request_id": request.get("request_id"),
        "status": status,
        "prompt_version": request.get("prompt_version"),
        "prompt_sha256": request.get("prompt_sha256"),
        "schema_sha256": request.get("schema_sha256"),
        "model_id": _manifest_model_id(manifest),
    }
    if error:
        stage["error"] = error
    return stage


def _run_quality(receipts: Mapping[str, Mapping[str, Any]], pair_ids: Sequence[str]) -> dict[str, Any]:
    """Separate endpoint/format failures from valid semantic abstentions.

    Evaluation status alone cannot tell a cold reader whether a pair was
    unresolved because the model abstained or because the endpoint returned a
    truncated/invalid response.  Keep those failure classes explicit in every
    native-run report.
    """

    transport: set[str] = set()
    truncated: set[str] = set()
    format_invalid: set[str] = set()
    valid: set[str] = set()
    semantic_abstentions: set[str] = set()
    terminal_not_testable: set[str] = set()
    not_run: set[str] = set()

    for pair_id in pair_ids:
        receipt = receipts.get(pair_id)
        if not isinstance(receipt, Mapping):
            not_run.add(pair_id)
            continue
        stages = [receipt.get("proposal"), receipt.get("verification")]
        pair_transport = pair_truncated = pair_invalid = False
        candidate_required = False
        candidate_verified = True
        for stage in stages:
            if not isinstance(stage, Mapping):
                continue
            status = str(stage.get("status") or "").upper()
            if status in {"FAILED", "ERROR", "TIMEOUT", "MISSING"}:
                pair_transport = True
            elif status == "TRUNCATED":
                pair_truncated = True
            elif status == "INVALID_OUTPUT":
                pair_invalid = True
            validation = stage.get("validation")
            if status in {"OK", "VALID"} and isinstance(validation, Mapping) and not validation.get("valid"):
                pair_invalid = True
        proposal_stage = receipt.get("proposal")
        if isinstance(proposal_stage, Mapping):
            proposal = proposal_stage.get("proposal")
            proposal_valid = (
                str(proposal_stage.get("status") or "").upper() in {"OK", "VALID"}
                and isinstance(proposal_stage.get("validation"), Mapping)
                and bool(proposal_stage["validation"].get("valid"))
                and isinstance(proposal, Mapping)
            )
            candidate_required = proposal_valid and str(proposal.get("status") or "").upper() in RELATION_CANDIDATE_STATUSES
            verification_stage = receipt.get("verification")
            if candidate_required:
                candidate_verified = (
                    isinstance(verification_stage, Mapping)
                    and str(verification_stage.get("status") or "").upper() in {"OK", "VALID"}
                    and isinstance(verification_stage.get("validation"), Mapping)
                    and bool(verification_stage["validation"].get("valid"))
                )
            if proposal_valid and (not candidate_required or candidate_verified):
                valid.add(pair_id)
                prediction = receipt.get("prediction")
                status = str(prediction.get("status") or "").upper() if isinstance(prediction, Mapping) else ""
                if status in ABSTENTION_STATUSES:
                    semantic_abstentions.add(pair_id)
                if isinstance(prediction, Mapping) and prediction.get("terminal_eligibility") == "NOT_TESTABLE":
                    terminal_not_testable.add(pair_id)
        if pair_transport:
            transport.add(pair_id)
        if pair_truncated:
            truncated.add(pair_id)
        if pair_invalid:
            format_invalid.add(pair_id)

    return {
        "pair_count": len(pair_ids),
        "valid_output_count": len(valid),
        "valid_output_ids": sorted(valid),
        "transport_failure_count": len(transport),
        "transport_failure_ids": sorted(transport),
        "truncation_count": len(truncated),
        "truncation_ids": sorted(truncated),
        "format_invalid_count": len(format_invalid),
        "format_invalid_ids": sorted(format_invalid),
        "semantic_abstention_count": len(semantic_abstentions),
        "semantic_abstention_ids": sorted(semantic_abstentions),
        "terminal_not_testable_count": len(terminal_not_testable),
        "terminal_not_testable_ids": sorted(terminal_not_testable),
        "not_run_count": len(not_run),
        "not_run_ids": sorted(not_run),
    }


async def run_relation_evaluation(
    *,
    split: str,
    prompt_version: str = "relation_v5",
    output: str | Path,
    client: Any | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    records: Sequence[Mapping[str, Any]] | None = None,
    gold_rows: Sequence[Mapping[str, Any]] | None = None,
    dry_run: bool = False,
    limit: int | None = None,
    input_path: str | Path | None = None,
    no_score: bool = False,
    concurrency: int = 1,
    group_by_proposition: bool = False,
) -> dict[str, Any]:
    """Run one frozen split and return its all-pair score/report.

    ``records``/``input_path`` and ``gold_rows`` are injectable for tests and
    native corpus runs.  With neither input option, the public adapter and
    checked-in fixture are loaded.  Gold rows are never passed to a client;
    ``no_score`` explicitly produces receipt/status output without any gold
    comparison.
    """

    if split not in {"development", "heldout"}:
        raise ValueError("split must be development or heldout")
    if batch_size < 1 or batch_size > 8:
        raise ValueError("batch_size must be between 1 and 8")
    if concurrency < 1 or concurrency > 3:
        raise ValueError("concurrency must be between 1 and 3")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive when supplied")
    # Validate prompt names before touching a client or output.
    proposal_prompt = load_prompt(prompt_version, stage="propose")
    verification_prompt = load_prompt(prompt_version, stage="verify")
    del proposal_prompt, verification_prompt
    output_path = Path(output)
    if input_path is not None and records is not None:
        raise ValueError("pass either records or input_path, not both")
    selected_public = (
        load_relation_input(input_path)
        if input_path is not None
        else list(records) if records is not None else public_relation_records(split)
    )
    selected_public = [dict(row) for row in selected_public]
    if limit is not None:
        selected_public = selected_public[:limit]
    if not selected_public:
        raise ValueError(f"no public relation records for split {split}")
    public_ids = {str(row.get("pair_id")) for row in selected_public}
    if no_score:
        selected_gold = []
    elif gold_rows is not None:
        selected_gold = [dict(row) for row in gold_rows if row.get("split") == split and str(row.get("pair_id")) in public_ids]
    elif records is not None:
        # Native corpus mode is source-only unless the caller explicitly
        # supplies adjudication rows.  Do not force canonical corpus IDs to
        # match the frozen benchmark's pair IDs.
        selected_gold = []
    else:
        selected_gold = [
            dict(row)
            for row in load_relation_gold(RELATION_GOLD_PATH, split)
            if str(row.get("pair_id")) in public_ids
        ]
    gold_by_id = _gold_map(selected_gold)
    public_by_id = {str(row.get("pair_id")): row for row in selected_public}
    if selected_gold and set(public_by_id) != set(gold_by_id):
        raise ValueError("public relation records and gold rows have different pair IDs")

    # Native inventories are large enough that one sequential stream is
    # needlessly slow, while the benchmark/default path must remain exactly
    # reproducible.  Each child keeps its own durable JSONL and cache, and
    # the parent atomically merges completed rows in public-record order.
    # When a prior serial/parallel run already produced a combined receipt,
    # seed the partition files from it so a later capacity increase resumes
    # rather than silently falling back to one stream.
    if concurrency > 1 and not dry_run:
        parallel_owns_client = client is None
        if parallel_owns_client:
            # Keep one cache namespace across partitions.  Request identities
            # already include all prompt/input/model fields, so sharing this
            # directory is safe and lets a capacity increase replay a prior
            # serial run instead of repeating model calls.
            client = LocalLLMClient(cache_dir=output_path.with_name(output_path.name + ".cache"))
            try:
                if not dry_run:
                    await client.discover()
            except Exception:
                await client.close()
                raise
        parallel_started = time.monotonic()
        partitions = _relation_partitions(
            selected_public,
            concurrency,
            group_by_proposition=group_by_proposition,
        )
        part_paths = [
            output_path.with_name(f"{output_path.name}.part{index}")
            for index in range(len(partitions))
        ]
        if output_path.exists():
            existing: dict[str, dict[str, Any]] = {}
            for line_number, line in enumerate(output_path.read_text(encoding="utf-8").split("\n"), 1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{output_path}:{line_number}: invalid receipt JSON") from exc
                if not isinstance(item, Mapping) or not item.get("pair_id"):
                    raise ValueError(f"{output_path}:{line_number}: receipt lacks pair_id")
                existing[str(item["pair_id"])] = dict(item)
        else:
            existing = {}
        # A prior parallel run may have durable rows only in its partition
        # files because the parent merge had not completed.  Treat those rows
        # as seeds too, while the per-partition loop below reassigns them to
        # the current stable proposition partition.
        for part_path in part_paths:
            if not part_path.exists():
                continue
            for line_number, line in enumerate(part_path.read_text(encoding="utf-8").split("\n"), 1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{part_path}:{line_number}: invalid receipt JSON") from exc
                if not isinstance(item, Mapping) or not item.get("pair_id"):
                    raise ValueError(f"{part_path}:{line_number}: receipt lacks pair_id")
                existing.setdefault(str(item["pair_id"]), dict(item))
        for part_path, rows in zip(part_paths, partitions, strict=True):
            partition_existing: dict[str, dict[str, Any]] = {}
            if part_path.exists():
                for line_number, line in enumerate(part_path.read_text(encoding="utf-8").split("\n"), 1):
                    if not line.strip():
                        continue
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"{part_path}:{line_number}: invalid receipt JSON") from exc
                    if not isinstance(item, Mapping) or not item.get("pair_id"):
                        raise ValueError(f"{part_path}:{line_number}: receipt lacks pair_id")
                    partition_existing[str(item["pair_id"])] = dict(item)
            changed = False
            for row in rows:
                pair_id = str(row["pair_id"])
                if pair_id in existing and pair_id not in partition_existing:
                    partition_existing[pair_id] = existing[pair_id]
                    changed = True
            if changed:
                ordered_seed = [
                    partition_existing[str(row["pair_id"])]
                    for row in rows
                    if str(row["pair_id"]) in partition_existing
                ]
                _write_jsonl(part_path, ordered_seed)

        async def _run_partition(index: int, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
            return await run_relation_evaluation(
                split=split,
                prompt_version=prompt_version,
                output=part_paths[index],
                client=client,
                batch_size=batch_size,
                max_tokens=max_tokens,
                records=rows,
                gold_rows=selected_gold,
                dry_run=False,
                limit=None,
                input_path=None,
                no_score=no_score,
                concurrency=1,
                group_by_proposition=group_by_proposition,
            )

        try:
            partition_reports = await asyncio.gather(*(
                _run_partition(index, rows)
                for index, rows in enumerate(partitions)
            ))
        finally:
            if parallel_owns_client:
                await client.close()
        merged: dict[str, dict[str, Any]] = {}
        for part_path in part_paths:
            for line_number, line in enumerate(part_path.read_text(encoding="utf-8").split("\n"), 1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{part_path}:{line_number}: invalid receipt JSON") from exc
                if not isinstance(item, Mapping) or not item.get("pair_id"):
                    raise ValueError(f"{part_path}:{line_number}: receipt lacks pair_id")
                merged[str(item["pair_id"])] = dict(item)
        ordered = [merged[str(row["pair_id"])] for row in selected_public if str(row["pair_id"]) in merged]
        _write_jsonl(output_path, ordered)
        predictions = {
            pair_id: merged[pair_id].get("prediction", {})
            for pair_id in public_by_id
            if pair_id in merged
        }
        if selected_gold:
            report = evaluate_relation_predictions(selected_gold, predictions)
        else:
            statuses = Counter(
                str(prediction.get("status") or "MISSING").upper()
                for prediction in predictions.values()
                if isinstance(prediction, Mapping)
            )
            report = {
                "evaluation_scope": "source_only_no_gold",
                "scored": False,
                "pair_count": len(selected_public),
                "prediction_count": len(predictions),
                "missing_pair_count": len(selected_public) - len(predictions),
                "status_counts": dict(sorted(statuses.items())),
                "note": "No adjudication rows supplied; this run reports receipts/model statuses only and makes no semantic precision claim.",
            }
        model_ids = sorted({str(item.get("model_id")) for item in partition_reports if item.get("model_id")})
        report.update({
            "split": split,
            "prompt_version": prompt_version,
            "batch_size": batch_size,
            "concurrency": concurrency,
            "group_by_proposition": group_by_proposition,
            "model_id": model_ids[0] if len(model_ids) == 1 else model_ids,
            "receipt_path": str(output_path),
            "partition_receipts": [str(path) for path in part_paths],
            "elapsed_seconds": round(time.monotonic() - parallel_started, 3),
            "resumed_pair_count": sum(int(item.get("resumed_pair_count", 0)) for item in partition_reports),
            "new_pair_count": sum(int(item.get("new_pair_count", 0)) for item in partition_reports),
            "admission_state": "PROPOSED",
            "run_quality": _run_quality(merged, [str(row["pair_id"]) for row in selected_public]),
        })
        _write_json(output_path.with_name(output_path.name + ".summary.json"), report)
        return report

    owns_client = client is None
    if owns_client:
        client = LocalLLMClient(cache_dir=output_path.with_name(output_path.name + ".cache"))
    manifest: Mapping[str, Any] | None = None
    started = time.monotonic()
    receipts: dict[str, dict[str, Any]] = {}
    # A JSONL file is the durable unit.  Loading malformed trailing lines is a
    # hard error: silently resuming from a corrupt receipt could mix models.
    if output_path.exists():
        for line_number, line in enumerate(output_path.read_text(encoding="utf-8").split("\n"), 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{output_path}:{line_number}: invalid receipt JSON") from exc
            if not isinstance(item, Mapping) or not item.get("pair_id"):
                raise ValueError(f"{output_path}:{line_number}: receipt lacks pair_id")
            receipts[str(item["pair_id"])] = dict(item)

    try:
        if dry_run:
            manifest = {"model_id": "dry-run", "client_version": "runner-dry-run"}
        else:
            manifest = getattr(client, "manifest", None) or await client.discover()
        model_id = _manifest_model_id(manifest)
        identities = {
            pair_id: _receipt_identity(public_by_id[pair_id], gold_by_id.get(pair_id), prompt_version=prompt_version, manifest=manifest)
            for pair_id in public_by_id
        }
        pending = [
            row for row in selected_public
            if not _is_resumable(receipts.get(str(row.get("pair_id")), {}), identities[str(row.get("pair_id"))])
        ]
        if dry_run:
            for record in pending:
                pair_id = str(record["pair_id"])
                receipts[pair_id] = {
                    "pair_id": pair_id,
                    "split": split,
                    "identity": identities[pair_id],
                    "source": _record_source_context(record, gold_by_id.get(pair_id)),
                    "proposal": {"status": "DRY_RUN"},
                    "prediction": _prediction_from_proposal(record, None),
                    "run": {"status": "DRY_RUN", "admission_state": "PROPOSED"},
                }
            ordered = [receipts[str(row["pair_id"])] for row in selected_public]
            _write_jsonl(output_path, ordered)
        else:
            for batch_records in _relation_batches(
                pending,
                batch_size,
                group_by_proposition=group_by_proposition,
            ):
                pair_ids = [str(row["pair_id"]) for row in batch_records]
                relation_pairs = [public_record_to_relation_pair(row) for row in batch_records]
                try:
                    proposal_request = build_batch_relation_request(
                        relation_pairs,
                        max_pairs=batch_size,
                        prompt_version=prompt_version,
                    )
                    _apply_context_coverage(proposal_request, relation_pairs)
                    proposal_response = await _request_batch(client, proposal_request, stage="proposal", max_tokens=max_tokens)
                    proposal_validation = _response_validation("proposal", proposal_response, proposal_request)
                except Exception as exc:  # noqa: BLE001 - persist invalid batch receipts
                    proposal_request = None
                    proposal_response = {"status": "FAILED", "error": f"{type(exc).__name__}: {exc}"}
                    proposal_validation = {"valid": False, "errors": [proposal_response["error"]], "results": []}

                proposal_results = {
                    str(item["pair_id"]): item
                    for item in proposal_validation.get("results", [])
                    if isinstance(item, Mapping) and item.get("pair_id")
                }
                verification_items: list[dict[str, Any]] = []
                pair_validations: dict[str, Mapping[str, Any]] = {}
                for pair_id in pair_ids:
                    item = proposal_results.get(pair_id)
                    if isinstance(item, Mapping):
                        pair_validations[pair_id] = item
                        proposal = _proposal_from_validation(item)
                        if proposal is not None and proposal.get("status") in RELATION_CANDIDATE_STATUSES and proposal_request is not None:
                            verification_items.append({
                                "pair_id": pair_id,
                                "proposal": proposal,
                                "request": proposal_request["validation_requests"][pair_id],
                            })
                verification_request: Mapping[str, Any] | None = None
                verification_response: Mapping[str, Any] = {"status": "NOT_RUN"}
                verification_validation: Mapping[str, Any] = {"valid": False, "results": []}
                if verification_items:
                    try:
                        verification_request = build_batch_verification_request(
                            verification_items,
                            max_pairs=batch_size,
                            prompt_version=prompt_version,
                        )
                        verification_response = await _request_batch(
                            client,
                            verification_request,
                            stage="verification",
                            max_tokens=max_tokens,
                        )
                        verification_validation = _response_validation("verification", verification_response, verification_request)
                    except Exception as exc:  # noqa: BLE001 - persist verifier failure per pair
                        verification_response = {"status": "FAILED", "error": f"{type(exc).__name__}: {exc}"}
                        verification_validation = {"valid": False, "errors": [verification_response["error"]], "results": []}
                verification_results = {
                    str(item["pair_id"]): item
                    for item in verification_validation.get("results", [])
                    if isinstance(item, Mapping) and item.get("pair_id")
                }
                verification_pair_ids = {str(item["pair_id"]) for item in verification_items}
                for record in batch_records:
                    pair_id = str(record["pair_id"])
                    validation = pair_validations.get(pair_id)
                    proposal = _proposal_from_validation(validation)
                    verifier = verification_results.get(pair_id)
                    if pair_id in verification_pair_ids:
                        # A candidate relation must survive the separate
                        # verifier; endpoint/parse/validation failure is an
                        # abstention, not silent acceptance of the proposal.
                        final_proposal = (
                            _proposal_from_validation(verifier)
                            if isinstance(verifier, Mapping) and verifier.get("valid")
                            else None
                        )
                    else:
                        final_proposal = proposal
                    proposal_stage_request = proposal_request or {
                        "request_id": None,
                        "prompt_version": prompt_version,
                        "prompt_sha256": identities[pair_id]["proposal_prompt_sha256"],
                        "schema_sha256": identities[pair_id]["proposal_schema_sha256"],
                    }
                    verifier_stage_request = verification_request or {
                        "request_id": None,
                        "prompt_version": prompt_version,
                        "prompt_sha256": identities[pair_id]["verification_prompt_sha256"],
                        "schema_sha256": identities[pair_id]["verification_schema_sha256"],
                    }
                    proposal_stage = _stage_receipt(
                        proposal_stage_request,
                        proposal_response,
                        validation,
                        manifest=manifest,
                    )
                    if pair_id in verification_pair_ids:
                        verification_stage = _stage_receipt(
                            verifier_stage_request,
                            verification_response,
                            verifier if isinstance(verifier, Mapping) else None,
                            manifest=manifest,
                        )
                    else:
                        verification_stage = {
                            "status": "NOT_REQUESTED",
                            "prompt_version": prompt_version,
                            "prompt_sha256": identities[pair_id]["verification_prompt_sha256"],
                            "schema_sha256": identities[pair_id]["verification_schema_sha256"],
                            "model_id": model_id,
                        }
                    prediction = _prediction_from_proposal(record, final_proposal)
                    if isinstance(verifier, Mapping) and verifier.get("valid") and verifier.get("verdict") in {"CONTESTED", "INSUFFICIENT_EVIDENCE"}:
                        prediction = _prediction_from_proposal(record, final_proposal)
                    receipts[pair_id] = {
                        "pair_id": pair_id,
                        "split": split,
                        "identity": identities[pair_id],
                        "source": _record_source_context(record, gold_by_id.get(pair_id)),
                        "proposal": proposal_stage,
                        "verification": verification_stage,
                        "prediction": prediction,
                        "run": {
                            "status": (
                                "COMPLETE"
                                if validation and validation.get("valid")
                                and (pair_id not in verification_pair_ids or (isinstance(verifier, Mapping) and verifier.get("valid")))
                                else "INVALID_OR_MISSING"
                            ),
                            "admission_state": "PROPOSED",
                            "proposal_batch_id": proposal_request.get("request_id") if proposal_request else None,
                            "verification_batch_id": verification_request.get("request_id") if verification_request else None,
                            "model": _safe_model_manifest(manifest),
                            "source_gold_not_sent_to_model": True,
                            "same_model_verifier_not_independent": bool(verifier),
                        },
                    }
                # A partial file is valid during an interrupted run; later
                # batches append/replace their pair rows on the next pass.
                _write_jsonl(
                    output_path,
                    [receipts[str(row["pair_id"])] for row in selected_public if str(row["pair_id"]) in receipts],
                )

        predictions = {
            pair_id: receipts[pair_id].get("prediction", {})
            for pair_id in public_by_id
            if pair_id in receipts
        }
        if selected_gold:
            report = evaluate_relation_predictions(selected_gold, predictions)
        else:
            statuses = Counter(
                str(prediction.get("status") or "MISSING").upper()
                for prediction in predictions.values()
                if isinstance(prediction, Mapping)
            )
            report = {
                "evaluation_scope": "source_only_no_gold",
                "scored": False,
                "pair_count": len(selected_public),
                "prediction_count": len(predictions),
                "missing_pair_count": len(selected_public) - len(predictions),
                "status_counts": dict(sorted(statuses.items())),
                "note": "No adjudication rows supplied; this run reports receipts/model statuses only and makes no semantic precision claim.",
            }
        report.update({
            "split": split,
            "prompt_version": prompt_version,
            "batch_size": batch_size,
            "concurrency": 1,
            "group_by_proposition": group_by_proposition,
            "model_id": model_id,
            "receipt_path": str(output_path),
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "resumed_pair_count": len(selected_public) - len(pending),
            "new_pair_count": len(pending),
            "admission_state": "PROPOSED",
            "run_quality": _run_quality(receipts, [str(row["pair_id"]) for row in selected_public]),
        })
        _write_json(output_path.with_name(output_path.name + ".summary.json"), report)
        return report
    finally:
        if owns_client and client is not None:
            await client.close()


def score_relation_receipts(
    output: str | Path,
    *,
    split: str,
    limit: int | None = None,
) -> dict[str, Any]:
    """Score completed JSONL receipts without contacting the local server."""

    if split not in {"development", "heldout"}:
        raise ValueError("split must be development or heldout")
    path = Path(output)
    if not path.is_file():
        raise FileNotFoundError(path)
    receipts: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").split("\n"), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid receipt JSON") from exc
        if not isinstance(item, Mapping) or not item.get("pair_id"):
            raise ValueError(f"{path}:{line_number}: receipt lacks pair_id")
        receipts.append(dict(item))
    gold = load_relation_gold(RELATION_GOLD_PATH, split)
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive when supplied")
        gold = gold[:limit]
    receipt_by_id = {str(item["pair_id"]): item for item in receipts}
    source_identity_mismatches: list[str] = []
    for row in gold:
        pair_id = str(row["pair_id"])
        receipt = receipt_by_id.get(pair_id)
        identity = receipt.get("identity") if isinstance(receipt, Mapping) else None
        expected_source, expected_object = _source_hashes(row, row)
        if not isinstance(identity, Mapping) or identity.get("source_sha256") != expected_source or identity.get("object_sha256") != expected_object:
            source_identity_mismatches.append(pair_id)
    prediction_map = {
        str(item["pair_id"]): item.get("prediction", {})
        for item in receipts
        if isinstance(item.get("prediction"), Mapping) and str(item["pair_id"]) not in source_identity_mismatches
    }
    report = evaluate_relation_predictions(gold, prediction_map)
    report.update({
        "split": split,
        "receipt_path": str(path),
        "score_only": True,
        "source_identity_mismatch_count": len(source_identity_mismatches),
        "source_identity_mismatch_ids": source_identity_mismatches,
        "admission_state": "PROPOSED",
    })
    _write_json(path.with_name(path.name + ".summary.json"), report)
    return report


def run_relation(**kwargs: Any) -> dict[str, Any]:
    """Synchronous convenience wrapper used by scripts and notebooks."""

    return asyncio.run(run_relation_evaluation(**kwargs))


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("development", "heldout"), required=True)
    parser.add_argument("--prompt", choices=("relation_v1", "relation_v2", "relation_v3", "relation_v4", "relation_v5", "relation_v6", "relation_v7", "relation_v7b", "relation_v7c"), default="relation_v5")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input", dest="input_path", type=Path, help="native relation-pair JSONL/JSON input; implies source-only mode unless gold is supplied by API")
    parser.add_argument("--batch-size", type=int, choices=tuple(range(1, 9)), default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--concurrency", type=int, choices=(1, 2, 3), default=1, help="parallel native partitions; default 1 preserves sequential benchmark behavior")
    parser.add_argument("--group-by-proposition", action="store_true", help="keep native official-object candidates for one proposition in one batch")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--limit", type=int, help="process only the first N stable records (e.g. development pilot 16)")
    parser.add_argument("--phase", choices=("run", "score"), default="run")
    parser.add_argument("--score", action="store_true", help="score existing receipts without contacting the local server")
    parser.add_argument("--no-score", action="store_true", help="run native/fixture inputs without comparing to semantic gold")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    report = (
        score_relation_receipts(args.output, split=args.split, limit=args.limit)
        if args.score or args.phase == "score"
        else run_relation(
            split=args.split,
            prompt_version=args.prompt,
            output=args.output,
            batch_size=args.batch_size,
            concurrency=args.concurrency,
            max_tokens=args.max_tokens,
            dry_run=args.dry_run,
            limit=args.limit,
            input_path=args.input_path,
            no_score=args.no_score,
            group_by_proposition=args.group_by_proposition,
        )
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point kept public for orchestration/tests."""

    return _main(argv)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_MAX_TOKENS",
    "load_relation_input",
    "main",
    "public_record_to_relation_pair",
    "run_relation",
    "run_relation_evaluation",
    "score_relation_receipts",
]

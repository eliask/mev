"""Normalize frozen primary institutional-action source slices.

This adapter deliberately accepts a structured, already retrieved source
slice.  It does not turn a news report or a guessed URL into an official
action.  The caller must provide the publisher URL, raw-body hash/size,
locator, and the normalized quotes extracted from the primary record.  That
boundary keeps a frozen replay honest while allowing the same normalizer to
be used by a live CloudNC/Dynasty fetcher later.
"""


import hashlib
import json
import sqlite3
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SOURCE_KIND = "OFFICIAL_INSTITUTIONAL_ACTIONS"
NORMALIZATION_VERSION = "html-visible-text-nfc-whitespace-1"


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _required(source: Mapping[str, Any], key: str) -> str:
    value = str(source.get(key) or "").strip()
    if not value:
        raise ValueError(f"official action source lacks {key}")
    return value


def _evidence(
    *,
    evidence_id: str,
    source: Mapping[str, Any],
    quote: str,
    field_path: str,
) -> dict[str, Any]:
    if not quote.strip():
        raise ValueError(f"official action source has empty {field_path} quote")
    raw_sha256 = _required(source, "raw_sha256")
    if len(raw_sha256) != 64:
        raise ValueError("raw_sha256 must be a SHA-256 hex digest")
    return {
        "evidence_id": evidence_id,
        "document_version_id": source["object_id"],
        "kind": "text_span",
        "text_sha256": _digest(quote),
        "span_start": 0,
        "span_end": len(quote),
        "quote": quote,
        "normalization_version": NORMALIZATION_VERSION,
        "record_locator": _required(source, "record_locator"),
        "field_path": field_path,
        "context_evidence_ids": [],
        "url": _required(source, "url"),
        "raw_sha256": raw_sha256,
        "byte_length": int(source.get("raw_bytes") or 0),
        "source_raw_sha256": raw_sha256,
        "source_raw_bytes": int(source.get("raw_bytes") or 0),
    }


def parse_resignation_source(source: Mapping[str, Any]) -> dict[str, Any]:
    """Build one source-grounded ``RESIGN_ROLE`` object and its evidence.

    ``date``/``action_date`` is the institutional grant/decision date.  The
    actor's request date is retained separately; it is not mislabeled as the
    date on which the institution granted the resignation.
    """

    source = dict(source)
    object_id = _required(source, "object_id")
    request_quote = _required(source, "request_quote")
    decision_quote = _required(source, "decision_quote")
    decision_date = _required(source, "decision_date")
    request_date = _required(source, "request_date")
    actor_id = _required(source, "actor_id")
    actor_name = _required(source, "actor_name")
    role = _required(source, "role")
    raw_sha256 = _required(source, "raw_sha256")
    raw_bytes = int(source.get("raw_bytes") or 0)
    if raw_bytes <= 0:
        raise ValueError("official action source raw_bytes must be positive")
    text = f"{request_quote}\n{decision_quote}"
    request_evidence_id = f"{object_id}:request"
    decision_evidence_id = f"{object_id}:decision"
    request_evidence = _evidence(
        evidence_id=request_evidence_id,
        source=source,
        quote=request_quote,
        field_path="request_quote",
    )
    decision_evidence = _evidence(
        evidence_id=decision_evidence_id,
        source=source,
        quote=decision_quote,
        field_path="decision_quote",
    )
    evidence_ids = [request_evidence_id, decision_evidence_id]
    obj = {
        "object_id": object_id,
        "kind": "RESIGN_ROLE",
        "matter_id": _required(source, "matter_id"),
        "title": _required(source, "title"),
        "text": text,
        "date": decision_date,
        "action_date": decision_date,
        "action_date_basis": "INSTITUTIONAL_DECISION_DATE",
        "request_date": request_date,
        "url": _required(source, "url"),
        "source_id": _required(source, "source_id"),
        "source_raw_sha256": raw_sha256,
        "source_raw_bytes": raw_bytes,
        "source_locator": _required(source, "record_locator"),
        "authors": [{
            "actor_id": actor_id,
            "name": actor_name,
            "role": "ACTOR",
            "identity_basis": "INDEPENDENT_CASE_REVIEW",
            "evidence_ids": evidence_ids,
        }],
        "evidence_ids": evidence_ids,
        "action": {
            "kind": "RESIGN_ROLE",
            "actor_id": actor_id,
            "role": role,
            "request_date": request_date,
            "institutional_decision_date": decision_date,
            "state": "GRANTED",
            "evidence_ids": evidence_ids,
        },
        "disposition": {
            "state": "GRANTED",
            "date": decision_date,
            "evidence_ids": [decision_evidence_id],
        },
        "coverage_scope": source.get("coverage_scope") or "single_official_decision_item",
        "void": False,
    }
    return {
        "object": obj,
        "evidence": [request_evidence, decision_evidence],
        "source": {
            "source_id": obj["source_id"],
            "url": obj["url"],
            "raw_sha256": raw_sha256,
            "raw_bytes": raw_bytes,
            "record_locator": obj["source_locator"],
        },
    }


def acquire_official_actions(
    sources: Iterable[Mapping[str, Any]],
    *,
    output_path: Path | None = None,
    retrieved_at: str | None = None,
) -> dict[str, Any]:
    """Normalize explicit primary source slices into importable records."""

    retrieved = retrieved_at or datetime.now(UTC).replace(microsecond=0).isoformat()
    parsed = [parse_resignation_source(source) for source in sources]
    objects = [item["object"] for item in parsed]
    evidence = [record for item in parsed for record in item["evidence"]]
    source_rows = [item["source"] for item in parsed]
    coverage_id = "official-actions-" + _digest(
        "|".join(row["source_id"] + ":" + row["raw_sha256"] for row in source_rows)
    )[:20]
    coverage = {
        "schema_version": "1.0",
        "coverage_id": coverage_id,
        "source_id": SOURCE_KIND,
        "population_definition": "Explicitly supplied primary institutional-action source slices",
        "record_type": "institutional_action",
        "enumeration_basis": "caller_supplied_primary_source_slices",
        "retrieved_at": retrieved,
        "expected_count": len(objects),
        "retrieved_count": len(objects),
        "parsed_count": len(objects),
        "linked_count": len(objects),
        "assessed_count": len(objects),
        "excluded_count": 0,
        "unresolved_count": 0,
        "counts_are_disjoint_partition": True,
        "status": "RECONCILED_FOR_DECLARED_SLICE",
        "scope_residuals": [
            "The result covers only the supplied primary records, not a complete municipal or regional register.",
            "A decision for one office does not establish a decision for another office named in a campaign statement.",
        ],
        "source_records": source_rows,
    }
    result = {"objects": objects, "evidence": evidence, "coverage": coverage, "retrieved_at": retrieved}
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            "".join(json.dumps({"kind": "official_object", "row": obj}, ensure_ascii=False) + "\n" for obj in objects)
            + "".join(json.dumps({"kind": "evidence", "row": row}, ensure_ascii=False) + "\n" for row in evidence)
            + json.dumps({"kind": "source_coverage", "row": coverage}, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return result


def import_official_actions(conn: sqlite3.Connection, result: Mapping[str, Any]) -> dict[str, int | str]:
    """Upsert only this adapter's rows into an existing PAA database.

    The function intentionally does not clear or rebuild any roster, document,
    proposition, or initiative-ledger table.  A caller can therefore import a
    reviewed institutional slice into a live database and then run the normal
    compiler in its own transaction.
    """

    objects = list(result.get("objects") or [])
    evidence = list(result.get("evidence") or [])
    coverage = dict(result.get("coverage") or {})
    if not coverage.get("coverage_id"):
        raise ValueError("official action result lacks coverage_id")
    for obj in objects:
        conn.execute(
            "INSERT OR REPLACE INTO official_objects(object_id, json) VALUES (?, ?)",
            (obj["object_id"], json.dumps(obj, ensure_ascii=False)),
        )
    for item in evidence:
        conn.execute(
            "INSERT OR REPLACE INTO evidence(evidence_id, json) VALUES (?, ?)",
            (item["evidence_id"], json.dumps(item, ensure_ascii=False)),
        )
    conn.execute(
        "INSERT OR REPLACE INTO source_coverage(coverage_id, json) VALUES (?, ?)",
        (coverage["coverage_id"], json.dumps(coverage, ensure_ascii=False)),
    )
    for source in coverage.get("source_records", []):
        conn.execute(
            "INSERT INTO manifest(source_id, url, sha256, bytes, http_status, retrieved_at, note) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                source.get("source_id"), source.get("url"), source.get("raw_sha256"), source.get("raw_bytes"),
                200, result.get("retrieved_at"), "Official institutional-action source slice; normalized by acquire_official_actions.",
            ),
        )
    return {"official_objects": len(objects), "evidence": len(evidence), "source_coverage": 1}


def import_relation_review(conn: sqlite3.Connection, review: Mapping[str, Any]) -> str:
    """Upsert one explicitly reviewed, version-bound relation row."""

    row = dict(review.get("row") or review)
    review_id = _required(row, "review_id")
    if not row.get("proposition_id") or not row.get("object_id"):
        raise ValueError("relation review needs proposition_id and object_id")
    conn.execute(
        "INSERT OR REPLACE INTO relation_reviews(review_id, json) VALUES (?, ?)",
        (review_id, json.dumps(row, ensure_ascii=False)),
    )
    return review_id


__all__ = [
    "NORMALIZATION_VERSION",
    "SOURCE_KIND",
    "acquire_official_actions",
    "import_official_actions",
    "import_relation_review",
    "parse_resignation_source",
]

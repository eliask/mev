"""A bounded receipt adapter for LawVM legal-text runs.

This module deliberately stops at the LawVM boundary.  LawVM can provide a
versioned reconstruction of a provision and a comparison with a cached
consolidated view; this adapter records those observations, their source
locators and residuals.  It does *not* decide that either view is operative
law, infer legal effects, or turn a reconcile result into an implementation
finding.

The public functions accept the JSON mappings emitted by the ``lawvm`` CLI.
They return ordinary JSON-compatible mappings so that the PAA store can carry
the receipt without importing LawVM or depending on its Python environment.
"""


import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from datetime import date
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "paa.legal_state_receipt.v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PLANES = ("replay", "oracle")


class LegalStateError(ValueError):
    """Raised when a receipt cannot preserve its identity safely."""


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise LegalStateError(f"{label} must be a JSON object")
    return dict(value)


def _text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _date_value(value: Any) -> str:
    candidate = _text(value)
    if not candidate:
        return ""
    try:
        date.fromisoformat(candidate)
    except ValueError:
        return ""
    return candidate


def _first(values: Iterable[Any]) -> str:
    for value in values:
        candidate = _text(value)
        if candidate:
            return candidate
    return ""


def _unique(values: Iterable[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        candidate = _text(value)
        if candidate and candidate not in seen:
            result.append(candidate)
            seen.add(candidate)
    return result


def _residual(code: str, severity: str, message: str, **details: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "code": code,
        "severity": severity,
        "message": message,
    }
    if details:
        row["details"] = details
    return row


def _payload_text(payload: Mapping[str, Any], *, oracle: bool = False) -> str:
    """Extract a displayed text view without treating it as raw source."""

    if oracle:
        return _first((payload.get("full_text"), payload.get("text")))
    text = payload.get("text")
    if isinstance(text, Mapping):
        return _text(text.get("rendered") or text.get("text"))
    return _text(text)


def _payload_available(payload: Mapping[str, Any], *, oracle: bool = False) -> bool:
    if oracle and "found" in payload:
        return bool(payload.get("found"))
    if "available" in payload:
        return bool(payload.get("available"))
    if isinstance(payload.get("text"), Mapping) and "available" in payload["text"]:
        return bool(payload["text"].get("available"))
    return bool(_payload_text(payload, oracle=oracle))


def _normalise_source_views(
    source_views: Mapping[str, Any] | Iterable[Mapping[str, Any]] | None,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Return explicit source views and residuals for malformed view rows.

    A source view is intentionally a receipt, not an assertion that the
    referenced bytes are authoritative.  ``raw_sha256`` is kept separate from
    hashes of rendered text or canonical JSON; callers must state its role.
    """

    if source_views is None:
        return {}, []
    if isinstance(source_views, Mapping):
        rows: list[Any] = []
        for plane, value in source_views.items():
            if isinstance(value, Mapping):
                row = dict(value)
                row.setdefault("plane", plane)
                rows.append(row)
            else:
                rows.append({"plane": plane, "value": value})
    else:
        rows = list(source_views)

    normalised: dict[str, dict[str, Any]] = {}
    residuals: list[dict[str, Any]] = []
    for raw in rows:
        if not isinstance(raw, Mapping):
            residuals.append(_residual(
                "SOURCE_VIEW_ROW_INVALID",
                "BLOCKING",
                "A source-view receipt was not a JSON object.",
            ))
            continue
        row = dict(raw)
        plane = _text(row.get("plane") or row.get("kind") or row.get("view"))
        if plane not in _PLANES:
            residuals.append(_residual(
                "SOURCE_VIEW_PLANE_UNRECOGNIZED",
                "BLOCKING",
                "Only replay and oracle source views are admitted by this adapter.",
                plane=plane,
            ))
            continue
        locator = _first((row.get("locator"), row.get("source_locator"), row.get("document_uri")))
        raw_hash = _first((row.get("raw_sha256"), row.get("sha256"), row.get("raw_hash")))
        if raw_hash and not _SHA256.fullmatch(raw_hash):
            raise LegalStateError(f"{plane} source-view raw_sha256 is not a lowercase SHA-256 digest")
        if not locator:
            residuals.append(_residual(
                "SOURCE_VIEW_LOCATOR_MISSING",
                "BLOCKING",
                f"The {plane} source view has no stable locator.",
                plane=plane,
            ))
        if not raw_hash:
            residuals.append(_residual(
                "SOURCE_VIEW_RAW_HASH_MISSING",
                "BLOCKING",
                f"The {plane} source view has no raw-byte hash receipt.",
                plane=plane,
            ))
        output = {
            "plane": plane,
            "source_id": _first((row.get("source_id"), row.get("id"))),
            "locator": locator,
            "record_locator": _first((row.get("record_locator"), row.get("structural_path"), row.get("path"))),
            "raw_sha256": raw_hash or None,
            "raw_hash_role": _first((row.get("raw_hash_role"), row.get("hash_role"))) or "unspecified_sha256",
            "quote": _text(row.get("quote")),
            "quote_hash": _first((row.get("quote_hash"), row.get("preview_sha256"))),
            "quote_role": _first((row.get("quote_role"), row.get("quote_semantics"))) or "unspecified_quote",
            "effective_date": _date_value(row.get("effective_date") or row.get("effective")) or None,
            "enacted_date": _date_value(row.get("enacted_date") or row.get("enacted")) or None,
            "cutoff_date": _date_value(row.get("cutoff_date") or row.get("oracle_cutoff_date")) or None,
            "edition_date": _date_value(row.get("edition_date")) or None,
            "edition_id": _first((row.get("edition_id"), row.get("oracle_version_amendment_id"))) or None,
            "capture": row.get("capture") if isinstance(row.get("capture"), Mapping) else None,
        }
        if plane in normalised:
            prior = normalised[plane]
            if prior != output:
                residuals.append(_residual(
                    "SOURCE_VIEW_DUPLICATE_CONFLICT",
                    "BLOCKING",
                    f"Multiple conflicting {plane} source-view receipts were supplied.",
                    plane=plane,
                ))
                continue
        normalised[plane] = output
    return normalised, residuals


def _normalise_source_artifacts(
    source_artifacts: Mapping[str, Any] | Iterable[Mapping[str, Any]] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate optional captured source artifacts and their text hashes.

    A LawVM locator/hash pair identifies a source, but it does not prove that
    the bytes are available to a clean-clone audit.  When a capture includes
    the decoded UTF-8 XML, this boundary recomputes the raw-byte digest.  A
    mismatch is an error rather than a warning.  Human-readable text is kept
    as a separately hashed view because it is not interchangeable with raw
    XML bytes.
    """

    if source_artifacts is None:
        return [], []
    if isinstance(source_artifacts, Mapping):
        rows: list[Any] = []
        for plane, value in source_artifacts.items():
            if isinstance(value, Mapping):
                item = dict(value)
                item.setdefault("plane", plane)
                rows.append(item)
            else:
                rows.append({"plane": plane, "value": value})
    else:
        rows = list(source_artifacts)

    normalised: list[dict[str, Any]] = []
    residuals: list[dict[str, Any]] = []
    for raw in rows:
        if not isinstance(raw, Mapping):
            residuals.append(_residual(
                "SOURCE_ARTIFACT_ROW_INVALID",
                "BLOCKING",
                "A captured source artifact was not a JSON object.",
            ))
            continue
        row = dict(raw)
        plane = _text(row.get("plane") or row.get("kind") or row.get("view"))
        if plane not in _PLANES:
            residuals.append(_residual(
                "SOURCE_ARTIFACT_PLANE_UNRECOGNIZED",
                "BLOCKING",
                "Only replay and oracle source artifacts are admitted by this adapter.",
                plane=plane,
            ))
            continue
        locator = _first((row.get("locator"), row.get("source_locator"), row.get("document_uri")))
        raw_hash = _first((row.get("raw_sha256"), row.get("raw_hash"), row.get("artifact_digest")))
        if raw_hash and not _SHA256.fullmatch(raw_hash):
            raise LegalStateError(f"{plane} source-artifact raw_sha256 is not a lowercase SHA-256 digest")
        raw_text = row.get("raw_text")
        if raw_text is not None and not isinstance(raw_text, str):
            raise LegalStateError(f"{plane} source-artifact raw_text must be a UTF-8 string")
        if raw_text is not None:
            observed_raw_hash = _sha256_text(raw_text)
            if raw_hash and raw_hash != observed_raw_hash:
                raise LegalStateError(
                    f"{plane} source-artifact raw_sha256 does not match captured raw_text"
                )
            raw_hash = raw_hash or observed_raw_hash
        human_text = row.get("human_text")
        if human_text is not None and not isinstance(human_text, str):
            raise LegalStateError(f"{plane} source-artifact human_text must be a string")
        human_hash = _first((row.get("human_text_sha256"), row.get("text_sha256")))
        if human_hash and not _SHA256.fullmatch(human_hash):
            raise LegalStateError(f"{plane} source-artifact human_text_sha256 is invalid")
        if human_text is not None:
            observed_human_hash = _sha256_text(human_text)
            if human_hash and human_hash != observed_human_hash:
                raise LegalStateError(
                    f"{plane} source-artifact human_text_sha256 does not match captured human_text"
                )
            human_hash = human_hash or observed_human_hash
        if not locator:
            residuals.append(_residual(
                "SOURCE_ARTIFACT_LOCATOR_MISSING",
                "BLOCKING",
                f"The {plane} source artifact has no stable locator.",
                plane=plane,
            ))
        if not raw_hash:
            residuals.append(_residual(
                "SOURCE_ARTIFACT_RAW_HASH_MISSING",
                "BLOCKING",
                f"The {plane} source artifact has no raw-byte hash.",
                plane=plane,
            ))
        commencement = row.get("commencement")
        if commencement is not None and not isinstance(commencement, Mapping):
            raise LegalStateError(f"{plane} source-artifact commencement must be a JSON object")
        commencement_row = dict(commencement) if isinstance(commencement, Mapping) else None
        if commencement_row:
            commencement_date = _date_value(
                commencement_row.get("date") or commencement_row.get("latest_start_date")
            )
            if not commencement_date:
                raise LegalStateError(f"{plane} source-artifact commencement date is invalid")
            commencement_row["date"] = commencement_date
        temporal_fields: dict[str, str | None] = {}
        for output_key, *input_keys in (
            ("effective_date", "effective_date", "effective"),
            ("enacted_date", "enacted_date", "enacted"),
            ("edition_date", "edition_date"),
        ):
            raw_date = _first(row.get(key) for key in input_keys)
            if raw_date and not _date_value(raw_date):
                raise LegalStateError(f"{plane} source-artifact {output_key} is invalid")
            temporal_fields[output_key] = _date_value(raw_date) or None
        normalised.append({
            "plane": plane,
            "artifact_kind": _first((row.get("artifact_kind"), row.get("kind"))),
            "source_id": _first((row.get("source_id"), row.get("id"))),
            "locator": locator,
            "record_locator": _first((row.get("record_locator"), row.get("structural_path"), row.get("path"))),
            "raw_sha256": raw_hash or None,
            "raw_hash_role": _first((row.get("raw_hash_role"), row.get("hash_role"))) or "unspecified_sha256",
            "raw_bytes": row.get("raw_bytes"),
            "raw_text": raw_text,
            "human_text": human_text,
            "human_text_sha256": human_hash or None,
            "human_text_role": _first((row.get("human_text_role"), row.get("text_role"))) or "unspecified_text_view",
            **temporal_fields,
            "commencement": commencement_row,
            "capture_role": _first((row.get("capture_role"), row.get("capture"))) or "source_artifact_capture",
        })
    return normalised, residuals


def _source_view_from_provision(replay: Mapping[str, Any]) -> dict[str, Any] | None:
    locator = replay.get("source_locator")
    if not isinstance(locator, Mapping):
        return None
    detail = locator.get("detail") if isinstance(locator.get("detail"), Mapping) else {}
    witness = detail.get("source_witness") if isinstance(detail.get("source_witness"), Mapping) else {}
    source = replay.get("source") if isinstance(replay.get("source"), Mapping) else {}
    version = replay.get("version") if isinstance(replay.get("version"), Mapping) else {}
    source_locator = _first((witness.get("locator"), locator.get("document_uri")))
    raw_hash = _first((
        witness.get("digest"),
        detail.get("artifact_digest"),
        locator.get("artifact_digest"),
    ))
    if not source_locator and not raw_hash:
        return None
    return {
        "plane": "replay",
        "source_id": _first((locator.get("source_id"), witness.get("artifact_id"))),
        "locator": source_locator,
        "record_locator": _first((locator.get("structural_path"), detail.get("target_legal_address_kind"))),
        "raw_sha256": raw_hash,
        "raw_hash_role": _first((
            detail.get("artifact_digest_status"),
            witness.get("source_role"),
            locator.get("artifact_digest_algorithm"),
        )) or "source_artifact_sha256",
        "quote": _text(witness.get("quote")),
        "quote_hash": _first((witness.get("quote_hash"), locator.get("quote_hash"))),
        "quote_role": _first((witness.get("source_role"), witness.get("quote_hash_semantics"))) or "lawvm_source_witness_quote",
        "effective_date": _first((version.get("effective"), source.get("effective"))) or None,
        "enacted_date": _first((version.get("enacted"), source.get("enacted"))) or None,
        "capture": {"derived_from": "lawvm.provision_state.source_locator"},
    }


def _source_view_from_oracle(oracle: Mapping[str, Any]) -> dict[str, Any] | None:
    locator = _first((oracle.get("locator"), oracle.get("source_locator")))
    raw_hash = _first((oracle.get("raw_sha256"), oracle.get("raw_hash"), oracle.get("artifact_digest")))
    if not locator and not raw_hash:
        return None
    quote = _text(oracle.get("full_text") or oracle.get("text"))
    return {
        "plane": "oracle",
        "source_id": _first((oracle.get("source_id"), oracle.get("oracle_version_amendment_id"))),
        "locator": locator,
        "record_locator": _first((oracle.get("section_filter"), oracle.get("resolved_section"))),
        "raw_sha256": raw_hash,
        "raw_hash_role": _first((oracle.get("raw_hash_role"), oracle.get("hash_role"))) or "unspecified_sha256",
        "quote": quote,
        "quote_hash": _first((oracle.get("quote_hash"), oracle.get("full_text_sha256"))) or (_sha256_text(quote) if quote else ""),
        "quote_role": _first((oracle.get("quote_role"), oracle.get("quote_semantics"))) or "lawvm_oracle_text_view",
        "cutoff_date": _first((oracle.get("oracle_cutoff_date"), oracle.get("cutoff_date"))) or None,
        "edition_date": _first((oracle.get("edition_date"),)) or None,
        "edition_id": _first((oracle.get("oracle_version_amendment_id"), oracle.get("edition_id"))) or None,
        "capture": {"derived_from": "lawvm.oracle_text"},
    }


def _merge_source_view(primary: Mapping[str, Any], fallback: Mapping[str, Any]) -> dict[str, Any]:
    """Fill omitted descriptive fields from a LawVM payload-derived view.

    An explicit capture manifest owns its hash and locator.  It may still omit
    the exact source quote or structural path, which the CLI payload can
    provide without replacing the manifest's provenance.
    """

    merged = dict(fallback)
    merged.update(dict(primary))
    for key, value in fallback.items():
        if merged.get(key) in (None, "", []) and value not in (None, "", []):
            merged[key] = value
    return merged


def _identity(replay: Mapping[str, Any], oracle: Mapping[str, Any], reconcile: Mapping[str, Any]) -> dict[str, Any]:
    query = replay.get("query") if isinstance(replay.get("query"), Mapping) else {}
    oracle_address = oracle.get("section_filter") or oracle.get("resolved_section")
    ids = _unique((
        replay.get("statute_id"), query.get("statute_id"), oracle.get("statute_id"), reconcile.get("statute_id"),
    ))
    if not ids:
        raise LegalStateError("LawVM receipt has no statute_id")
    if len(ids) > 1:
        raise LegalStateError("LawVM receipt mixes statute IDs: " + ", ".join(ids))
    addresses = _unique((
        (replay.get("resolved_address") or {}).get("text") if isinstance(replay.get("resolved_address"), Mapping) else None,
        query.get("provision"),
        oracle_address,
        reconcile.get("selector"),
    ))
    if not addresses:
        raise LegalStateError("LawVM receipt has no exact legal address")
    as_of_values = _unique((replay.get("as_of"), query.get("as_of"), reconcile.get("as_of")))
    if not as_of_values:
        raise LegalStateError("LawVM receipt has no as_of date")
    as_of = as_of_values[0]
    if not _date_value(as_of):
        raise LegalStateError("LawVM receipt as_of must be an ISO date")
    return {
        "statute_id": ids[0],
        "legal_address": addresses[0],
        "address_views": addresses,
        "as_of": as_of,
        "as_of_views": as_of_values,
        "query_types": _unique((query.get("query_type"), reconcile.get("query_type"))),
    }


def _view_record(
    plane: str,
    payload: Mapping[str, Any],
    source_view: Mapping[str, Any] | None,
    *,
    oracle: bool,
) -> dict[str, Any]:
    text = _payload_text(payload, oracle=oracle)
    row: dict[str, Any] = {
        "plane": plane,
        "available": _payload_available(payload, oracle=oracle),
        "status": _first((
            payload.get("provision_status"),
            payload.get("replay_status"),
            payload.get("selection_status"),
            payload.get("found") if oracle else None,
        )) or None,
        "text_sha256": _sha256_text(text) if text else None,
        "text_length": len(text) if text else 0,
    }
    if oracle:
        row.update({
            "cutoff_date": _first((payload.get("oracle_cutoff_date"), payload.get("cutoff_date"))),
            "edition_date": _first((payload.get("edition_date"),)),
            "version_amendment": _first((payload.get("oracle_version_amendment_id"),)),
        })
    else:
        row.update({
            "effective": _first(((payload.get("version") or {}).get("effective") if isinstance(payload.get("version"), Mapping) else None, payload.get("effective"))),
            "enacted": _first(((payload.get("version") or {}).get("enacted") if isinstance(payload.get("version"), Mapping) else None, payload.get("enacted"))),
            "source_amendment": _first(((payload.get("source") or {}).get("statute_id") if isinstance(payload.get("source"), Mapping) else None, payload.get("source_amendment"))),
        })
    if source_view:
        row["source_view"] = dict(source_view)
    return row


def _temporal_source_assessment(
    plane: str,
    payload: Mapping[str, Any],
    source_view: Mapping[str, Any] | None,
    as_of: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Check only whether a declared source date is later than the query.

    This is a guard against using a future edition for a historical question,
    not a legal-effect interpreter.  A date that is not later than ``as_of``
    is merely *not future by this check*; it is not certified operative.
    """

    view = source_view or {}
    residuals: list[dict[str, Any]] = []
    if plane == "oracle":
        source_date = _date_value(
            view.get("cutoff_date")
            or payload.get("oracle_cutoff_date")
            or payload.get("cutoff_date")
            or view.get("edition_date")
            or payload.get("edition_date")
        )
        basis = "oracle_cutoff_date_or_edition_date"
    else:
        version = payload.get("version") if isinstance(payload.get("version"), Mapping) else {}
        source = payload.get("source") if isinstance(payload.get("source"), Mapping) else {}
        source_date = _date_value(
            view.get("effective_date")
            or version.get("effective")
            or source.get("effective")
            or view.get("enacted_date")
            or version.get("enacted")
            or source.get("enacted")
        )
        basis = "effective_date_or_enacted_date"
    if not source_date:
        return {
            "status": "SOURCE_DATE_UNRESOLVED",
            "historical_verification": "NOT_VALIDATED",
            "as_of": as_of,
            "source_date": None,
            "basis": basis,
        }, [
            _residual(
                "TEMPORAL_SOURCE_DATE_UNRESOLVED",
                "WARNING",
                f"The {plane} source has no independently declared temporal date for the as-of guard.",
                plane=plane,
                as_of=as_of,
                basis=basis,
            )
        ]
    if source_date > as_of:
        return {
            "status": "FUTURE_SOURCE_NOT_VALIDATED",
            "historical_verification": "BLOCKED",
            "as_of": as_of,
            "source_date": source_date,
            "basis": basis,
        }, [
            _residual(
                "TEMPORAL_SOURCE_NOT_VALIDATED",
                "BLOCKING",
                f"The {plane} source edition/cutoff is later than the requested as-of date; it is not used to validate historical legal state.",
                plane=plane,
                source_date=source_date,
                as_of=as_of,
                basis=basis,
                source_id=view.get("source_id"),
                locator=view.get("locator"),
            )
        ]
    return {
        "status": "DATE_NOT_AFTER_AS_OF",
        "historical_verification": "NOT_BLOCKED_BY_FUTURE_DATE",
        "as_of": as_of,
        "source_date": source_date,
        "basis": basis,
    }, residuals


def _scope_assessment(
    artifact: Mapping[str, Any],
    as_of: str,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    commencement = artifact.get("commencement")
    if not isinstance(commencement, Mapping):
        return None, []
    commencement_date = _date_value(commencement.get("date"))
    if not commencement_date:
        return {
            "status": "COMMENCEMENT_DATE_UNRESOLVED",
            "historical_verification": "NOT_VALIDATED",
            "as_of": as_of,
        }, [
            _residual(
                "OPERATIVE_PROVISION_SCOPE_NOT_VALIDATED",
                "BLOCKING",
                "A commencement/scope constraint was captured without a valid date.",
                plane=artifact.get("plane"),
            )
        ]
    scope = {
        "status": "FUTURE_SCOPE_DEADLINE",
        "historical_verification": "NOT_VALIDATED",
        "as_of": as_of,
        "date": commencement_date,
        "basis": commencement.get("date_basis") or "explicit_source_text",
        "quote": _text(commencement.get("quote")) or None,
    }
    if commencement_date <= as_of:
        scope["status"] = "SCOPE_DATE_NOT_AFTER_AS_OF"
        scope["historical_verification"] = "NOT_BLOCKED_BY_FUTURE_DATE"
        return scope, []
    return scope, [
        _residual(
            "OPERATIVE_PROVISION_SCOPE_NOT_VALIDATED",
            "BLOCKING",
            "The captured commencement text places the latest required use after the requested as-of date; source inclusion is not a claim that the duty applied by as-of.",
            plane=artifact.get("plane"),
            scope_date=commencement_date,
            as_of=as_of,
            basis=scope["basis"],
        )
    ]


def build_legal_state_receipt(
    replay: Mapping[str, Any] | None,
    oracle: Mapping[str, Any] | None,
    reconcile: Mapping[str, Any] | None,
    *,
    source_views: Mapping[str, Any] | Iterable[Mapping[str, Any]] | None = None,
    source_artifacts: Mapping[str, Any] | Iterable[Mapping[str, Any]] | None = None,
    invocation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a source-bound LawVM comparison receipt.

    ``replay`` is normally the complete ``provision-state`` payload.  ``oracle``
    is normally the ``oracle-text`` payload, and ``reconcile`` is the JSON
    emitted by ``lawvm reconcile --json``.  Missing planes remain visible as
    residuals; they are never replaced with a default legal state.
    """

    replay_map = _mapping(replay, "replay")
    oracle_map = _mapping(oracle, "oracle")
    reconcile_map = _mapping(reconcile, "reconcile")
    identity = _identity(replay_map, oracle_map, reconcile_map)

    explicit_views, view_residuals = _normalise_source_views(source_views)
    artifacts, artifact_residuals = _normalise_source_artifacts(source_artifacts)
    derived_replay = _source_view_from_provision(replay_map)
    derived_oracle = _source_view_from_oracle(oracle_map)
    if "replay" not in explicit_views and derived_replay:
        # The provision-state source locator carries an actual Finlex source
        # artifact digest, so it is a valid receipt even without a separate
        # caller-supplied view row.
        explicit_views["replay"] = _normalise_source_views([derived_replay])[0]["replay"]
    elif "replay" in explicit_views and derived_replay:
        explicit_views["replay"] = _merge_source_view(explicit_views["replay"], derived_replay)
    if "oracle" not in explicit_views and derived_oracle:
        explicit_views["oracle"] = _normalise_source_views([derived_oracle])[0]["oracle"]
    elif "oracle" in explicit_views and derived_oracle:
        explicit_views["oracle"] = _merge_source_view(explicit_views["oracle"], derived_oracle)

    residuals = [*view_residuals, *artifact_residuals]

    def view_residual_present(code: str, plane: str) -> bool:
        return any(
            item.get("code") == code and (item.get("details") or {}).get("plane") == plane
            for item in residuals
        )

    for plane in _PLANES:
        view = explicit_views.get(plane)
        if view is None:
            residuals.append(_residual(
                "SOURCE_VIEW_RECEIPT_MISSING",
                "BLOCKING",
                f"No {plane} source-view receipt was supplied.",
                plane=plane,
            ))
            continue
        if not view.get("locator") and not view_residual_present("SOURCE_VIEW_LOCATOR_MISSING", plane):
            residuals.append(_residual(
                "SOURCE_VIEW_LOCATOR_MISSING",
                "BLOCKING",
                f"The {plane} source view has no stable locator.",
                plane=plane,
            ))
        if not view.get("raw_sha256") and not view_residual_present("SOURCE_VIEW_RAW_HASH_MISSING", plane):
            residuals.append(_residual(
                "SOURCE_VIEW_RAW_HASH_MISSING",
                "BLOCKING",
                f"The {plane} source view has no raw-byte hash receipt.",
                plane=plane,
            ))
        if view.get("raw_hash_role") == "unspecified_sha256":
            residuals.append(_residual(
                "SOURCE_VIEW_RAW_HASH_ROLE_UNSPECIFIED",
                "WARNING",
                f"The {plane} hash is present but its byte/hash role was not declared.",
                plane=plane,
            ))

    if not replay_map:
        residuals.append(_residual("REPLAY_OUTPUT_MISSING", "BLOCKING", "No LawVM replay/provision-state output was supplied."))
    if not oracle_map:
        residuals.append(_residual("ORACLE_OUTPUT_MISSING", "BLOCKING", "No LawVM oracle output was supplied."))
    if not reconcile_map:
        residuals.append(_residual("RECONCILIATION_OUTPUT_MISSING", "BLOCKING", "No LawVM reconcile output was supplied."))

    if len(identity["address_views"]) > 1:
        residuals.append(_residual(
            "LEGAL_ADDRESS_VIEW_MISMATCH",
            "BLOCKING",
            "Replay, oracle and reconcile outputs do not all name the same legal address.",
            addresses=identity["address_views"],
        ))
    if len(identity["as_of_views"]) > 1:
        residuals.append(_residual(
            "AS_OF_VIEW_MISMATCH",
            "BLOCKING",
            "Replay and reconcile outputs carry different as-of dates.",
            as_of_dates=identity["as_of_views"],
        ))
    if len(identity["query_types"]) > 1:
        residuals.append(_residual(
            "QUERY_TYPE_VIEW_MISMATCH",
            "WARNING",
            "The replay and reconcile commands used different query types; both are preserved.",
            query_types=identity["query_types"],
        ))

    engine = replay_map.get("engine") if isinstance(replay_map.get("engine"), Mapping) else {}
    if engine.get("git_dirty") in {True, "true", "True"}:
        residuals.append(_residual(
            "LAWVM_BUILD_DIRTY",
            "WARNING",
            "The LawVM producer reported a dirty working tree; replay reproducibility is qualified.",
            build_id=engine.get("build_id"),
        ))

    replay_view = _view_record("replay", replay_map, explicit_views.get("replay"), oracle=False)
    oracle_view = _view_record("oracle", oracle_map, explicit_views.get("oracle"), oracle=True)
    replay_temporal, replay_temporal_residuals = _temporal_source_assessment(
        "replay", replay_map, explicit_views.get("replay"), identity["as_of"]
    )
    oracle_temporal, oracle_temporal_residuals = _temporal_source_assessment(
        "oracle", oracle_map, explicit_views.get("oracle"), identity["as_of"]
    )
    replay_view["temporal"] = replay_temporal
    oracle_view["temporal"] = oracle_temporal
    residuals.extend([*replay_temporal_residuals, *oracle_temporal_residuals])
    oracle_cutoff = _date_value(oracle_view.get("cutoff_date"))
    if oracle_cutoff and oracle_cutoff < identity["as_of"]:
        residuals.append(_residual(
            "ORACLE_CUTOFF_PRECEDES_AS_OF",
            "WARNING",
            "The consolidated oracle cutoff predates the requested as-of date; this is a comparison residual, not a legal conclusion.",
            oracle_cutoff=oracle_cutoff,
            as_of=identity["as_of"],
        ))

    for artifact in artifacts:
        artifact_temporal, artifact_temporal_residuals = _temporal_source_assessment(
            str(artifact["plane"]),
            replay_map if artifact["plane"] == "replay" else oracle_map,
            artifact,
            identity["as_of"],
        )
        artifact["temporal"] = artifact_temporal
        scope, scope_residuals = _scope_assessment(artifact, identity["as_of"])
        if scope is not None:
            artifact["scope_temporal"] = scope
        residuals.extend([*artifact_temporal_residuals, *scope_residuals])

    verdict = _text(reconcile_map.get("verdict")).upper()
    divergence_class = _text(reconcile_map.get("divergence_class")) or None
    if verdict == "DISAGREE" or divergence_class:
        comparison_state = "DIVERGENT"
    elif verdict == "AGREE":
        comparison_state = "AGREED_FOR_DECLARED_COMPARISON"
    else:
        comparison_state = "UNRESOLVED"

    if reconcile_map:
        residuals.append(_residual(
            "RECONCILIATION_NOT_OPERATIVE_AUTHORITY",
            "BLOCKING",
            "A replay/oracle comparison is not by itself an operative legal-state authorization.",
            verdict=verdict or None,
        ))
    residuals.append(_residual(
        "OPERATIVE_VERIFICATION_WITHHELD",
        "BLOCKING",
        "This adapter does not certify operative legal effect from LawVM outputs alone.",
    ))
    residuals.append(_residual(
        "LEGAL_EFFECTS_NOT_ASSESSED",
        "INFO",
        "Implementation, service delivery, behavioural and causal effects are outside this receipt.",
    ))

    temporal_blocking_codes = {
        "TEMPORAL_SOURCE_NOT_VALIDATED",
        "OPERATIVE_PROVISION_SCOPE_NOT_VALIDATED",
    }
    temporal_blocked = any(item.get("code") in temporal_blocking_codes for item in residuals)
    temporal_guard = {
        "status": "BLOCKED" if temporal_blocked else "BOUNDED",
        "historical_verification": "NOT_VALIDATED" if temporal_blocked else "NOT_BLOCKED_BY_FUTURE_DATE",
        "as_of": identity["as_of"],
        "sources": {
            "replay": replay_temporal,
            "oracle": oracle_temporal,
        },
        "note": "This guard only rejects future source editions and exposes explicit commencement constraints; it does not certify operative law.",
    }
    if artifacts:
        temporal_guard["artifacts"] = [
            {
                "plane": artifact["plane"],
                "locator": artifact.get("locator"),
                "temporal": artifact.get("temporal"),
                "scope_temporal": artifact.get("scope_temporal"),
            }
            for artifact in artifacts
        ]

    present_planes = [plane for plane in ("replay", "oracle", "reconcile") if {
        "replay": replay_map,
        "oracle": oracle_map,
        "reconcile": reconcile_map,
    }[plane]]
    missing = [plane for plane in ("replay", "oracle", "reconcile") if plane not in present_planes]
    missing.extend(f"{plane}_source_view" for plane in _PLANES if not explicit_views.get(plane, {}).get("raw_sha256"))
    missing.extend(
        f"{plane}_source_hash_role"
        for plane in _PLANES
        if explicit_views.get(plane, {}).get("raw_sha256")
        and explicit_views.get(plane, {}).get("raw_hash_role") == "unspecified_sha256"
    )
    if any(
        item.get("code") in {"TEMPORAL_SOURCE_NOT_VALIDATED", "OPERATIVE_PROVISION_SCOPE_NOT_VALIDATED"}
        for item in residuals
    ):
        missing.append("historical_temporal_validation")
    completeness_state = "COMPLETE_FOR_DECLARED_INPUTS" if not missing else "PARTIAL"

    evidence: list[dict[str, Any]] = []
    for plane in _PLANES:
        view = explicit_views.get(plane)
        if not view:
            continue
        seed = f"{plane}|{view.get('locator')}|{view.get('raw_sha256')}"
        evidence.append({
            "evidence_id": f"LAWVM-{_sha256_text(seed)[:24]}",
            "plane": plane,
            "source_id": view.get("source_id") or None,
            "locator": view.get("locator") or None,
            "record_locator": view.get("record_locator") or None,
            "raw_sha256": view.get("raw_sha256") or None,
            "raw_hash_role": view.get("raw_hash_role") or None,
            "quote": view.get("quote") or None,
            "quote_hash": view.get("quote_hash") or None,
            "quote_role": view.get("quote_role") or None,
        })

    comparison_summary = {
        "DIVERGENT": (
            "The LawVM replay and consolidated oracle diverge for the declared provision and date; "
            "this receipt does not choose either view as operative law."
        ),
        "AGREED_FOR_DECLARED_COMPARISON": (
            "The LawVM replay and consolidated oracle agree for the declared comparison; "
            "this receipt still does not certify operative law or implementation."
        ),
        "UNRESOLVED": "The declared LawVM comparison is incomplete or unresolved.",
    }[comparison_state]
    unknowns = [
        {
            "code": residual["code"],
            "message": residual["message"],
        }
        for residual in residuals
        if residual.get("severity") == "BLOCKING"
    ]
    unknowns.extend([
        {
            "code": "OPERATIVE_LEGAL_STATE",
            "message": "Whether the reconstructed text is operative law is not answered by this adapter.",
        },
        {
            "code": "IMPLEMENTATION_AND_EFFECTS",
            "message": "Implementation, service, behavioural and causal effects are not part of this source slice.",
        },
    ])
    question_id = "LAWVM-Q-" + _sha256_text(
        f"{identity['statute_id']}|{identity['legal_address']}|{identity['as_of']}"
    )[:24]

    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "receipt_id": "LAWVM-RECEIPT-" + _sha256_text(
            f"{identity['statute_id']}|{identity['legal_address']}|{identity['as_of']}|{_canonical_hash(reconcile_map)}"
        )[:24],
        "jurisdiction": _text(replay_map.get("jurisdiction")) or "fi",
        "statute_id": identity["statute_id"],
        "legal_address": identity["legal_address"],
        "address_views": identity["address_views"],
        "as_of": identity["as_of"],
        "query_types": identity["query_types"],
        "title": _first((
            (replay_map.get("base") or {}).get("title") if isinstance(replay_map.get("base"), Mapping) else None,
            replay_map.get("title"),
            oracle_map.get("title"),
        )),
        "reconstruction": {
            "replay": replay_view,
            "oracle": oracle_view,
        },
        "comparison": {
            "state": comparison_state,
            "verdict": verdict or None,
            "divergence_class": divergence_class,
            "agree_ratio": reconcile_map.get("agree_ratio"),
            "scope": reconcile_map.get("scope"),
            "scope_note": reconcile_map.get("scope_note"),
            "selector": reconcile_map.get("selector"),
            "locator": reconcile_map.get("locator"),
            "temporal": temporal_guard,
            "replay": {
                "status": (reconcile_map.get("replay") or {}).get("replay_status") if isinstance(reconcile_map.get("replay"), Mapping) else None,
                "source_amendment": (reconcile_map.get("replay") or {}).get("source_amendment") if isinstance(reconcile_map.get("replay"), Mapping) else None,
            },
            "oracle": {
                "basis": (reconcile_map.get("oracle") or {}).get("basis") if isinstance(reconcile_map.get("oracle"), Mapping) else None,
                "version_markers": (reconcile_map.get("oracle") or {}).get("version_markers") if isinstance(reconcile_map.get("oracle"), Mapping) else [],
            },
        },
        "question": {
            "question_id": question_id,
            "text": (
                "What do the source-bound LawVM replay and consolidated oracle show for "
                f"{identity['statute_id']} {identity['legal_address']} as of {identity['as_of']}?"
            ),
            "scope": {
                "statute_id": identity["statute_id"],
                "legal_address": identity["legal_address"],
                "as_of": identity["as_of"],
            },
            "answer": {
                "state": f"{comparison_state}_SOURCE_VIEWS",
                "summary": comparison_summary,
                "decisive_evidence_ids": [row["evidence_id"] for row in evidence],
            },
            "unknowns": unknowns,
        },
        "source_views": [explicit_views[plane] for plane in _PLANES if plane in explicit_views],
        "source_artifacts": artifacts,
        "evidence": evidence,
        "coverage": {
            "state": completeness_state,
            "scope": "one LawVM statute provision, two text views, one declared as-of date",
            "present_planes": present_planes,
            "missing": missing,
            "temporal_guard": temporal_guard,
            "note": "Completeness describes the captured receipt, not legal validity or policy implementation.",
        },
        "operative": {
            "verified": False,
            "state": "NOT_OPERATIVE_VERIFIED",
            "basis": "LAWVM_RECONSTRUCTION_AND_COMPARISON_ONLY",
        },
        "legal_effects": {
            "state": "NOT_ASSESSED",
            "items": [],
            "note": "No operative, implementation, service, outcome or causal effect is inferred.",
        },
        "lawvm": {
            "schema": replay_map.get("schema") or None,
            "spec_version": replay_map.get("spec_version") or None,
            "engine": dict(engine),
            "payload_sha256": {
                "replay": _canonical_hash(replay_map) if replay_map else None,
                "oracle": _canonical_hash(oracle_map) if oracle_map else None,
                "reconcile": _canonical_hash(reconcile_map) if reconcile_map else None,
            },
        },
        "residuals": residuals,
    }
    if invocation is not None:
        if not isinstance(invocation, Mapping):
            raise LegalStateError("invocation must be a JSON object")
        receipt["invocation"] = dict(invocation)
    validate_legal_state_receipt(receipt)
    return receipt


def validate_legal_state_receipt(receipt: Mapping[str, Any]) -> bool:
    """Validate structural and safety invariants of an adapted receipt."""

    if not isinstance(receipt, Mapping):
        raise LegalStateError("legal-state receipt must be a JSON object")
    required = {
        "schema_version", "receipt_id", "statute_id", "legal_address", "as_of",
        "reconstruction", "comparison", "question", "source_views", "source_artifacts", "coverage", "operative",
        "legal_effects", "residuals",
    }
    missing = sorted(required - set(receipt))
    if missing:
        raise LegalStateError("legal-state receipt missing: " + ", ".join(missing))
    if receipt.get("schema_version") != SCHEMA_VERSION:
        raise LegalStateError("unsupported legal-state receipt schema")
    if not _text(receipt.get("statute_id")) or not _text(receipt.get("legal_address")):
        raise LegalStateError("legal-state receipt identity is incomplete")
    if not _date_value(receipt.get("as_of")):
        raise LegalStateError("legal-state receipt as_of must be an ISO date")
    operative = receipt.get("operative")
    if not isinstance(operative, Mapping) or operative.get("verified") is not False or operative.get("state") == "OPERATIVE_VERIFIED":
        raise LegalStateError("LawVM receipt must not claim operative legal verification")
    effects = receipt.get("legal_effects")
    if not isinstance(effects, Mapping) or effects.get("state") != "NOT_ASSESSED" or effects.get("items") != []:
        raise LegalStateError("legal effects must remain NOT_ASSESSED with no inferred items")
    source_views = receipt.get("source_views")
    if not isinstance(source_views, list):
        raise LegalStateError("source_views must be a list")
    for view in source_views:
        if not isinstance(view, Mapping) or view.get("plane") not in _PLANES:
            raise LegalStateError("source view has an unknown plane")
        raw_hash = view.get("raw_sha256")
        if raw_hash is not None and not _SHA256.fullmatch(_text(raw_hash)):
            raise LegalStateError("source view raw_sha256 is invalid")
    source_artifacts = receipt.get("source_artifacts")
    if not isinstance(source_artifacts, list):
        raise LegalStateError("source_artifacts must be a list")
    for artifact in source_artifacts:
        if not isinstance(artifact, Mapping) or artifact.get("plane") not in _PLANES:
            raise LegalStateError("source artifact has an unknown plane")
        raw_hash = artifact.get("raw_sha256")
        if raw_hash is not None and not _SHA256.fullmatch(_text(raw_hash)):
            raise LegalStateError("source artifact raw_sha256 is invalid")
        raw_text = artifact.get("raw_text")
        if raw_text is not None and _sha256_text(str(raw_text)) != raw_hash:
            raise LegalStateError("source artifact raw_text hash does not match")
        human_hash = artifact.get("human_text_sha256")
        human_text = artifact.get("human_text")
        if human_hash is not None and human_text is not None and _sha256_text(str(human_text)) != human_hash:
            raise LegalStateError("source artifact human_text hash does not match")
    coverage = receipt.get("coverage")
    if not isinstance(coverage, Mapping) or coverage.get("state") not in {"PARTIAL", "COMPLETE_FOR_DECLARED_INPUTS"}:
        raise LegalStateError("legal-state coverage state is invalid")
    if not isinstance(receipt.get("residuals"), list):
        raise LegalStateError("legal-state residuals must be a list")
    for residual in receipt["residuals"]:
        if not isinstance(residual, Mapping) or not _text(residual.get("code")):
            raise LegalStateError("legal-state residual is not typed")
    return True


def load_lawvm_capture(path: str | Path) -> dict[str, Any]:
    """Load a JSON capture containing ``replay``, ``oracle`` and ``reconcile``."""

    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LegalStateError(f"cannot load LawVM capture {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise LegalStateError("LawVM capture must be a JSON object")
    return dict(value)


def build_legal_state_receipt_from_capture(
    capture: Mapping[str, Any],
    *,
    invocation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Adapt a persisted capture envelope without importing LawVM.

    Capture metadata is kept as invocation provenance, while the three LawVM
    result planes and explicit source-view receipts remain the only inputs to
    the legal-state comparison itself.
    """

    capture_map = _mapping(capture, "capture")
    if invocation is None:
        invocation = {
            "commands": capture_map.get("commands") or [],
            "captured_cli_artifacts": capture_map.get("captured_cli_artifacts") or [],
            "fixture_kind": capture_map.get("fixture_kind") or None,
        }
    return build_legal_state_receipt(
        capture_map.get("replay"),
        capture_map.get("oracle"),
        capture_map.get("reconcile"),
        source_views=capture_map.get("source_views"),
        source_artifacts=capture_map.get("source_artifacts"),
        invocation=invocation,
    )


# Descriptive aliases keep the boundary easy to discover without making a
# second implementation or implying that this is a legal oracle.
adapt_lawvm_outputs = build_legal_state_receipt
build_lawvm_receipt = build_legal_state_receipt


__all__ = [
    "SCHEMA_VERSION",
    "LegalStateError",
    "adapt_lawvm_outputs",
    "build_lawvm_receipt",
    "build_legal_state_receipt",
    "build_legal_state_receipt_from_capture",
    "load_lawvm_capture",
    "validate_legal_state_receipt",
]

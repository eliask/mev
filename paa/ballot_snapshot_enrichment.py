"""Enrich accepted matter ballot snapshots with current official raw witnesses.

Validates portable witness-bank receipts by hashing retained body bytes,
comparing normalized person/group/response rows to decoded raw rows, and
attaching ``CURRENT_OFFICIAL_RAW_BOUND`` only after independent proof.
Normalized snapshot hashes stay separate from raw witnesses. Never claims
historical unrecorded HTTP bytes. Stale/mismatch fails the affected binding
only; prior numbers/actors/groups are preserved.
"""

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

SCHEMA_VERSION: Final = "paa.ballot_snapshot_enrichment.v1"
BINDING_STATUS: Final = "CURRENT_OFFICIAL_RAW_BOUND"
MATCH_STATUS: Final = "MATCH_CURRENT_RAW"
_KNOWN_RESPONSES: Final = frozenset({"JAA", "EI", "TYHJÄÄ", "POISSA"})


class BallotEnrichmentError(ValueError):
    """Raised when a witness bank or matter packet cannot be enriched safely."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _response_norm(value: object) -> str:
    text = str(value or "").strip().casefold()
    aliases = {
        "jaa": "JAA",
        "ei": "EI",
        "tyhjää": "TYHJÄÄ",
        "tyhjaa": "TYHJÄÄ",
        "tyhja": "TYHJÄÄ",
        "poissa": "POISSA",
    }
    return aliases.get(text, str(value or "").strip().upper() or "UNKNOWN")


def _group_norm(value: object) -> str:
    return str(value or "").strip()


def _resolve_body_path(path_value: object, *, bank_dir: Path | None) -> Path | None:
    if not path_value:
        return None
    path = Path(str(path_value))
    if path.is_file():
        return path
    if bank_dir is not None:
        candidate = bank_dir / path
        if candidate.is_file():
            return candidate
    return None


def _receipt_body_path(receipt: Mapping[str, Any], *, bank_dir: Path | None) -> Path | None:
    if receipt.get("body_path"):
        return _resolve_body_path(receipt["body_path"], bank_dir=bank_dir)
    artifact = receipt.get("artifact_body")
    if not artifact or bank_dir is None:
        return None
    relative = Path(str(artifact))
    if relative.is_absolute():
        return None
    path = (bank_dir / relative).resolve()
    if not path.is_relative_to(bank_dir.resolve()) or not path.is_file():
        return None
    return path


def _decode_table_rows(body: Mapping[str, Any]) -> list[dict[str, Any]]:
    columns = [str(item) for item in (body.get("columnNames") or [])]
    row_data = body.get("rowData")
    if isinstance(row_data, list):
        out: list[dict[str, Any]] = []
        for row in row_data:
            if isinstance(row, Mapping):
                out.append(dict(row))
                continue
            if isinstance(row, list) and columns:
                out.append({columns[i]: row[i] for i in range(min(len(columns), len(row)))})
        return out
    # Compact fixture shape: {"rows":[...]} or a single metadata object.
    rows = body.get("rows")
    if isinstance(rows, list):
        return [dict(row) for row in rows if isinstance(row, Mapping)]
    if body.get("AanestysId") is not None:
        return [dict(body)]
    return []


def _url_mentions_event(url: object, event_id: str) -> bool:
    text = str(url or "")
    return bool(event_id) and event_id in text


def _verify_receipt_body(
    receipt: Mapping[str, Any],
    *,
    bank_dir: Path | None,
    label: str,
) -> tuple[bytes | None, list[str]]:
    errors: list[str] = []
    status = receipt.get("status")
    if status is not None and int(status) != 200:
        errors.append(f"{label} HTTP status {status!r} is not 200")
    content_type = str(receipt.get("content_type") or "")
    if content_type and "json" not in content_type.casefold():
        errors.append(f"{label} content_type {content_type!r} is not JSON")
    body_path = _receipt_body_path(receipt, bank_dir=bank_dir)
    if body_path is None:
        errors.append(f"{label} body_path missing or unreadable")
        return None, errors
    data = body_path.read_bytes()
    digest = _sha256_bytes(data)
    declared = str(receipt.get("sha256") or "").strip().casefold()
    if not declared:
        errors.append(f"{label} missing receipt sha256")
    elif digest != declared:
        errors.append(f"{label} body sha256 mismatch")
    declared_bytes = receipt.get("body_bytes")
    if declared_bytes is not None and int(declared_bytes) != len(data):
        errors.append(
            f"{label} body_bytes mismatch: declared={declared_bytes} actual={len(data)}"
        )
    return data, errors


def validate_witness_event(
    event: Mapping[str, Any],
    *,
    bank_dir: Path | None = None,
) -> dict[str, Any]:
    """Validate one portable witness event against retained raw bodies."""

    errors: list[str] = []
    event_id = str(event.get("event_id") or "")
    if not event_id:
        errors.append("missing event_id")
    bank_status = str(event.get("comparison_status") or "")
    if bank_status and bank_status != MATCH_STATUS:
        errors.append(f"bank comparison_status is {bank_status!r}, want {MATCH_STATUS}")
    bank_errors = list(event.get("comparison_errors") or [])
    if bank_errors:
        errors.append("bank comparison_errors is non-empty")
    meta = event.get("metadata") or {}
    receipt = event.get("metadata_receipt") or {}
    if not isinstance(meta, Mapping):
        errors.append("metadata is not an object")
        meta = {}
    if not isinstance(receipt, Mapping):
        errors.append("metadata_receipt is not an object")
        receipt = {}
    if str(meta.get("AanestysId") or "") != event_id:
        errors.append("metadata AanestysId does not match event_id")
    for url_key in ("request_url", "final_url"):
        if receipt.get(url_key) and not _url_mentions_event(receipt.get(url_key), event_id):
            errors.append(f"metadata {url_key} does not mention event_id")
    meta_bytes, meta_errors = _verify_receipt_body(receipt, bank_dir=bank_dir, label="metadata")
    errors.extend(meta_errors)
    meta_rows: list[dict[str, Any]] = []
    if meta_bytes is not None:
        try:
            meta_body = json.loads(meta_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            errors.append(f"metadata body is not JSON: {error}")
            meta_body = None
        if isinstance(meta_body, Mapping):
            meta_rows = _decode_table_rows(meta_body)
            if len(meta_rows) != 1:
                errors.append(f"metadata row cardinality {len(meta_rows)} != 1")
            elif (
                "AanestysId" in meta_rows[0]
                and str(meta_rows[0].get("AanestysId") or "") not in {"", event_id}
            ):
                errors.append("metadata body AanestysId mismatch")
    pages = list(event.get("ballot_pages") or [])
    if not pages:
        errors.append("no ballot_pages")
    seen_people: set[str] = set()
    response_counts: Counter[str] = Counter()
    page_numbers: list[int] = []
    person_index: dict[str, dict[str, Any]] = {}
    verified_pages: list[dict[str, Any]] = []
    for page_offset, page in enumerate(pages):
        if not isinstance(page, Mapping):
            errors.append("ballot page is not an object")
            continue
        try:
            page_no = int(page.get("page"))
        except (TypeError, ValueError):
            errors.append(f"page {page.get('page')!r} is not an integer")
            continue
        page_numbers.append(page_no)
        page_receipt = page.get("receipt") or {}
        if not isinstance(page_receipt, Mapping):
            errors.append(f"page {page_no} receipt missing")
            page_receipt = {}
        for url_key in ("request_url", "final_url"):
            url = page_receipt.get(url_key) or page.get(url_key)
            if url and not _url_mentions_event(url, event_id):
                errors.append(f"page {page_no} {url_key} does not mention event_id")
        page_bytes, page_errors = _verify_receipt_body(
            page_receipt, bank_dir=bank_dir, label=f"page {page_no}"
        )
        errors.extend(page_errors)
        decoded_rows: list[dict[str, Any]] = []
        if page_bytes is not None:
            try:
                page_body = json.loads(page_bytes.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                errors.append(f"page {page_no} body is not JSON: {error}")
                page_body = None
            if isinstance(page_body, Mapping):
                decoded_rows = _decode_table_rows(page_body)
                declared_count = page.get("row_count")
                if declared_count is not None and int(declared_count) != len(decoded_rows):
                    errors.append(
                        f"page {page_no} row_count {declared_count} != len(rows) {len(decoded_rows)}"
                    )
                if page_offset == len(pages) - 1 and not decoded_rows:
                    errors.append(f"final page {page_no} has no rows")
        # Prefer body-decoded rows; fall back to embedded rows only when body absent.
        rows = decoded_rows or list(page.get("rows") or [])
        if decoded_rows and isinstance(page.get("rows"), list) and page["rows"]:
            embedded = list(page["rows"])
            if len(embedded) != len(decoded_rows):
                errors.append(f"page {page_no} embedded row count differs from body")
        for row_index, row in enumerate(rows):
            if not isinstance(row, Mapping):
                errors.append(f"page {page_no} row {row_index} is not an object")
                continue
            if str(row.get("AanestysId") or "").strip() not in {"", event_id}:
                errors.append(f"page {page_no} row AanestysId mismatch")
            person = str(row.get("EdustajaHenkiloNumero") or "").strip()
            if not person:
                errors.append(f"page {page_no} row missing EdustajaHenkiloNumero")
                continue
            if person in seen_people:
                errors.append(f"duplicate person {person}")
            seen_people.add(person)
            response = _response_norm(row.get("EdustajaAanestys"))
            if response not in _KNOWN_RESPONSES:
                errors.append(f"unknown response {response!r} for person {person}")
            response_counts[response] += 1
            person_index[person] = {
                "person_id": person,
                "response": response,
                "group_code": _group_norm(row.get("EdustajaRyhmaLyhenne")),
                "page": page_no,
                "row_index": row_index,
                "raw_json_path": f"ballot_pages[{page_offset}].rows[{row_index}]",
                "page_sha256": page_receipt.get("sha256"),
                "first_name": row.get("EdustajaEtunimi"),
                "last_name": row.get("EdustajaSukunimi"),
            }
        if page is pages[-1] and page.get("has_more") is True:
            errors.append("last page still has_more=true; pagination incomplete")
        if page is not pages[-1] and page.get("has_more") is not True:
            errors.append(f"intermediate page {page_no} missing has_more=true")
        verified_pages.append(
            {
                "page": page_no,
                "sha256": page_receipt.get("sha256"),
                "url": page_receipt.get("final_url")
                or page_receipt.get("request_url")
                or page.get("request_url"),
                "row_count": len(rows),
                "has_more": page.get("has_more"),
                "body_verified": page_bytes is not None,
            }
        )
    if page_numbers:
        expected_pages = list(range(min(page_numbers), min(page_numbers) + len(page_numbers)))
        if sorted(page_numbers) != expected_pages or len(set(page_numbers)) != len(page_numbers):
            errors.append(f"page numbers not contiguous/unique: {page_numbers}")
    expected_totals = {
        "JAA": int(meta.get("AanestysTulosJaa") or 0),
        "EI": int(meta.get("AanestysTulosEi") or 0),
        "TYHJÄÄ": int(meta.get("AanestysTulosTyhjia") or 0),
        "POISSA": int(meta.get("AanestysTulosPoissa") or 0),
    }
    for key, expected in expected_totals.items():
        if response_counts.get(key, 0) != expected:
            errors.append(
                f"response total mismatch for {key}: rows={response_counts.get(key, 0)} meta={expected}"
            )
    published_total = meta.get("AanestysTulosYhteensa")
    if published_total is not None and int(published_total) != len(seen_people):
        errors.append(
            f"unique people {len(seen_people)} != AanestysTulosYhteensa {published_total}"
        )
    if sum(expected_totals.values()) != len(seen_people):
        errors.append("sum of published buckets does not equal unique people")
    return {
        "event_id": event_id,
        "ok": not errors,
        "errors": errors,
        "unique_people": len(seen_people),
        "response_counts": dict(response_counts),
        "expected_totals": expected_totals,
        "page_count": len(pages),
        "metadata_receipt_sha256": receipt.get("sha256"),
        "metadata_url": receipt.get("final_url") or receipt.get("request_url") or meta.get("Url"),
        "capture_date": receipt.get("captured_at") or receipt.get("completed_at"),
        "pages": verified_pages,
        "person_index": person_index,
        "matter": event.get("matter"),
    }


def _row_person_id(row: Mapping[str, Any]) -> str:
    for key in ("person_id", "henkilo_numero", "EdustajaHenkiloNumero", "actor_id"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    actor = row.get("actor") or {}
    if isinstance(actor, Mapping):
        for key in ("person_id", "henkilo_numero", "actor_id"):
            value = actor.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
    return ""


def _event_id_from_row(row: Mapping[str, Any]) -> str:
    """Resolve AanestysId/event identity from common snapshot field names."""

    for key in ("event_id", "vote_id", "AanestysId", "aanestys_id"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    source = row.get("source")
    if isinstance(source, Mapping):
        for key in ("event_id", "vote_id", "AanestysId"):
            value = source.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
    return ""


def _normalized_hash_witnesses(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Retain local normalized snapshot hashes separately from raw HTTP receipts."""

    out: list[dict[str, Any]] = []
    for evidence in row.get("source_evidence") or []:
        if not isinstance(evidence, Mapping):
            continue
        digest = evidence.get("text_sha256") or evidence.get("source_sha256")
        if not digest:
            continue
        out.append(
            {
                "kind": evidence.get("kind") or "NORMALIZED_LOCAL_RECORD",
                "hash_basis": evidence.get("source_version")
                or "NORMALIZED_SNAPSHOT_NOT_RAW_HTTP_PAYLOAD",
                "text_sha256": digest,
                "record_locator": evidence.get("record_locator"),
                "evidence_id": evidence.get("evidence_id"),
            }
        )
    for key in ("group_peer_source_sha256", "source_sha256"):
        digest = row.get(key)
        if digest:
            out.append(
                {
                    "kind": "NORMALIZED_PACKET_HASH",
                    "hash_basis": "NORMALIZED_SNAPSHOT_NOT_RAW_HTTP_PAYLOAD",
                    "text_sha256": digest,
                    "field": key,
                }
            )
    return out


def _people_container(row: Mapping[str, Any]) -> tuple[str | None, list[Any]]:
    for key in ("rows", "ballots", "members"):
        people = row.get(key)
        if isinstance(people, list):
            return key, people
    return None, []


def _compare_normalized_to_raw(
    people: Sequence[Mapping[str, Any]],
    person_index: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    errors: list[str] = []
    if not people:
        return ["normalized person rows missing for comparison"]
    seen: set[str] = set()
    for person in people:
        if not isinstance(person, Mapping):
            errors.append("normalized person row is not an object")
            continue
        person_id = _row_person_id(person)
        if not person_id:
            errors.append("normalized person missing person_id")
            continue
        if person_id in seen:
            errors.append(f"duplicate normalized person {person_id}")
        seen.add(person_id)
        raw = person_index.get(person_id)
        if raw is None:
            errors.append(f"normalized person {person_id} missing from raw witness")
            continue
        old_response = _response_norm(person.get("response") or person.get("EdustajaAanestys"))
        if old_response != raw["response"]:
            errors.append(
                f"response mismatch for {person_id}: normalized={old_response} raw={raw['response']}"
            )
        old_group = _group_norm(person.get("group_code") or person.get("EdustajaRyhmaLyhenne"))
        raw_group = _group_norm(raw.get("group_code"))
        if old_group and raw_group and old_group.casefold() != raw_group.casefold():
            errors.append(
                f"group mismatch for {person_id}: normalized={old_group} raw={raw_group}"
            )
    if len(seen) != len(person_index):
        errors.append(
            f"normalized unique people {len(seen)} != raw unique people {len(person_index)}"
        )
    return errors


def _require_comparison_coverage(
    bank_event_ids: set[str],
    comparison: Mapping[str, Any] | None,
) -> dict[str, Mapping[str, Any]]:
    if comparison is None:
        raise BallotEnrichmentError(
            "comparison object is required and must cover every bank event"
        )
    rows = list(comparison.get("events") or [])
    by_id: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or not row.get("event_id"):
            raise BallotEnrichmentError("comparison event missing event_id")
        event_id = str(row["event_id"])
        if event_id in by_id:
            raise BallotEnrichmentError(f"duplicate comparison event_id {event_id}")
        status = str(row.get("comparison_status") or row.get("status") or "")
        if status != MATCH_STATUS:
            raise BallotEnrichmentError(
                f"comparison event {event_id} status {status!r}, want {MATCH_STATUS}"
            )
        mismatches = row.get("mismatches") or {}
        if isinstance(mismatches, Mapping):
            for key, value in mismatches.items():
                if value:
                    raise BallotEnrichmentError(
                        f"comparison event {event_id} has nonempty mismatches.{key}"
                    )
        if row.get("errors"):
            raise BallotEnrichmentError(f"comparison event {event_id} has nonempty errors")
        by_id[event_id] = row
    missing = sorted(bank_event_ids - set(by_id))
    extra = sorted(set(by_id) - bank_event_ids)
    if missing or extra:
        raise BallotEnrichmentError(
            "comparison must cover exactly the bank events; "
            f"missing={missing[:8]} extra={extra[:8]}"
        )
    return by_id


def _index_bank_events(witness_bank: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    events: dict[str, Mapping[str, Any]] = {}
    for event in witness_bank.get("events") or []:
        if not isinstance(event, Mapping) or not event.get("event_id"):
            continue
        event_id = str(event["event_id"])
        if event_id in events:
            raise BallotEnrichmentError(f"duplicate bank event_id {event_id}")
        events[event_id] = event
    if not events:
        raise BallotEnrichmentError("witness bank has no events")
    return events


def enrich_matter_packet(
    matter: Mapping[str, Any],
    witness_bank: Mapping[str, Any],
    *,
    comparison: Mapping[str, Any] | None = None,
    bank_dir: Path | None = None,
) -> dict[str, Any]:
    """Return a new matter packet with successful bindings attached."""

    if str(witness_bank.get("schema_version") or "") != "bs02.portable_ballot_witness_bank.v1":
        raise BallotEnrichmentError(
            f"unsupported witness bank schema {witness_bank.get('schema_version')!r}"
        )
    events = _index_bank_events(witness_bank)
    comparison_by_id = _require_comparison_coverage(set(events), comparison)
    out = json.loads(json.dumps(matter, ensure_ascii=False))
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "bound_event_ids": [],
        "failed_event_ids": [],
        "skipped_event_ids": [],
        "failures": [],
        "preserved_prior_rows": True,
        "historical_http_claim": False,
        "comparison_required": True,
        "raw_body_verified": True,
    }
    targets: list[tuple[list[Any], int, dict[str, Any], str]] = []
    if isinstance(out.get("vote_sources"), list):
        for index, row in enumerate(out["vote_sources"]):
            if isinstance(row, Mapping):
                targets.append((out["vote_sources"], index, dict(row), f"vote_sources[{index}]"))
    matter_obj = out.get("matter")
    if isinstance(matter_obj, Mapping) and isinstance(matter_obj.get("vote_sources"), list):
        for index, row in enumerate(matter_obj["vote_sources"]):
            if isinstance(row, Mapping):
                targets.append(
                    (matter_obj["vote_sources"], index, dict(row), f"matter.vote_sources[{index}]")
                )
    for key in ("group_peer_sources", "ballot_packets", "vote_packets"):
        rows = out.get(key)
        if not isinstance(rows, list):
            continue
        for index, row in enumerate(rows):
            if isinstance(row, Mapping):
                targets.append((rows, index, dict(row), f"{key}[{index}]"))

    by_event: dict[str, list[tuple[list[Any], int, dict[str, Any], str]]] = {}
    for target in targets:
        event_id = _event_id_from_row(target[2])
        if event_id and event_id in events:
            by_event.setdefault(event_id, []).append(target)

    bound_ids: set[str] = set()
    failed_ids: set[str] = set()
    for event_id, group in sorted(by_event.items()):
        comparison_row = comparison_by_id[event_id]
        validation = validate_witness_event(events[event_id], bank_dir=bank_dir)
        if not validation["ok"]:
            failed_ids.add(event_id)
            report["failures"].append({"event_id": event_id, "errors": validation["errors"]})
            continue
        person_maps: list[Mapping[str, Any]] = []
        for _container, _index, row, _path in group:
            _key, people = _people_container(row)
            person_maps.extend(person for person in people if isinstance(person, Mapping))
        if not person_maps:
            failed_ids.add(event_id)
            report["failures"].append(
                {
                    "event_id": event_id,
                    "errors": ["normalized person rows missing for comparison"],
                }
            )
            continue
        # Compare using the largest person list (group_peer packet) when duplicates exist.
        unique_people = { _row_person_id(person): person for person in person_maps if _row_person_id(person) }
        compare_errors = _compare_normalized_to_raw(
            list(unique_people.values()),
            validation["person_index"],
        )
        if compare_errors:
            failed_ids.add(event_id)
            report["failures"].append({"event_id": event_id, "errors": compare_errors})
            continue
        historical = comparison_row.get("historical_version_status") or (
            "NOT_RECONSTRUCTIBLE_WITHOUT_PRIOR_RECEIPT"
        )
        for container, index, row, path in group:
            enriched = dict(row)
            enriched["source_binding"] = {
                "status": BINDING_STATUS,
                "event_id": event_id,
                "json_path": path,
                "metadata_raw_sha256": validation.get("metadata_receipt_sha256"),
                "metadata_url": validation.get("metadata_url"),
                "capture_date": validation.get("capture_date"),
                "page_receipts": validation.get("pages"),
                "unique_people": validation.get("unique_people"),
                "normalized_snapshot_hashes": _normalized_hash_witnesses(row),
                "raw_witness_separate_from_normalized": True,
                "historical_version_status": historical,
                "matter": validation.get("matter") or events[event_id].get("matter"),
                "note": (
                    "Current official raw witness bound after body hash verification and "
                    "per-person response comparison. Normalized local snapshot hashes are "
                    "retained separately and are not historical HTTP payload bytes."
                ),
            }
            people_key, people = _people_container(row)
            if people_key is not None:
                person_out_rows: list[Any] = []
                for person_index, person in enumerate(people):
                    if not isinstance(person, Mapping):
                        person_out_rows.append(person)
                        continue
                    person_out = dict(person)
                    person_id = _row_person_id(person_out)
                    raw = validation["person_index"].get(person_id) or {}
                    person_out["source_binding"] = {
                        "status": BINDING_STATUS,
                        "event_id": event_id,
                        "json_path": f"{path}.{people_key}[{person_index}]",
                        "person_id": person_id,
                        "raw_json_path": raw.get("raw_json_path"),
                        "raw_page": raw.get("page"),
                        "raw_row_index": raw.get("row_index"),
                        "raw_page_sha256": raw.get("page_sha256"),
                        "raw_response": raw.get("response"),
                        "raw_group_code": raw.get("group_code"),
                        "metadata_raw_sha256": validation.get("metadata_receipt_sha256"),
                        "capture_date": validation.get("capture_date"),
                        "historical_version_status": historical,
                    }
                    person_out_rows.append(person_out)
                enriched[people_key] = person_out_rows
            container[index] = enriched
        bound_ids.add(event_id)
    report["bound_event_ids"] = sorted(bound_ids)
    report["failed_event_ids"] = sorted(failed_ids)
    report["skipped_event_ids"] = sorted(set(events) - bound_ids - failed_ids)
    # Drop scoped ballot-version gaps only for events that now carry a verified binding.
    if bound_ids:
        cleared_codes = {
            "BALLOT_SOURCE_VERSION_NOT_BOUND",
            "VOTE_SOURCE_ANCHOR_NOT_LOADED",
        }
        remaining: list[Any] = []
        removed = 0
        for row in out.get("missingness") or []:
            if not isinstance(row, Mapping):
                remaining.append(row)
                continue
            code = str(row.get("code") or "")
            locator = str(row.get("record_locator") or "")
            event_hit = any(
                event_id in locator or f"AanestysId={event_id}" in locator
                for event_id in bound_ids
            )
            if code in cleared_codes and event_hit:
                removed += 1
                continue
            remaining.append(row)
        out["missingness"] = remaining
        report["cleared_missingness_rows"] = removed
    out["ballot_enrichment"] = report
    return out


def enrich_matter_files(
    *,
    matter_paths: Sequence[Path],
    witness_bank_path: Path,
    output_dir: Path,
    comparison_path: Path | None = None,
) -> dict[str, Any]:
    """Apply a portable witness bank to matter JSON files into a unique output dir."""

    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise BallotEnrichmentError(f"output directory is not empty: {output_dir}")
    if comparison_path is None:
        raise BallotEnrichmentError("comparison_path is required for enrichment")
    output_dir.mkdir(parents=True, exist_ok=True)
    bank = _load_json(witness_bank_path)
    comparison = _load_json(comparison_path)
    export_raw = output_dir / "raw_witnesses"
    export_raw.mkdir(exist_ok=True)
    bank_dir = Path(witness_bank_path).parent
    written: list[dict[str, Any]] = []
    for matter_path in matter_paths:
        matter = _load_json(matter_path)
        enriched = enrich_matter_packet(
            matter, bank, comparison=comparison, bank_dir=bank_dir
        )
        out_name = Path(matter_path).name
        (output_dir / out_name).write_text(
            json.dumps(enriched, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        written.append(
            {
                "input": Path(matter_path).name,
                "output": out_name,
                "bound": enriched.get("ballot_enrichment", {}).get("bound_event_ids"),
                "failed": enriched.get("ballot_enrichment", {}).get("failed_event_ids"),
            }
        )
    portable = json.loads(json.dumps(bank, ensure_ascii=False))
    for event in portable.get("events") or []:
        receipt = event.get("metadata_receipt") or {}
        body_path = receipt.get("body_path") or receipt.get("artifact_body")
        if body_path:
            src = _receipt_body_path(receipt, bank_dir=bank_dir)
            if src is not None:
                digest = receipt.get("sha256") or _sha256_bytes(src.read_bytes())
                dest = export_raw / f"{event.get('event_id')}_metadata_{digest[:12]}.body"
                dest.write_bytes(src.read_bytes())
                receipt["artifact_body"] = str(dest.relative_to(output_dir))
        for key in ("body_path", "headers_path", "result_path", "curl_line"):
            receipt.pop(key, None)
        event["metadata_receipt"] = receipt
        for page in event.get("ballot_pages") or []:
            page_receipt = page.get("receipt") or {}
            body_path = page_receipt.get("body_path") or page_receipt.get("artifact_body")
            if body_path:
                src = _receipt_body_path(page_receipt, bank_dir=bank_dir)
                if src is not None:
                    digest = page_receipt.get("sha256") or _sha256_bytes(src.read_bytes())
                    dest = export_raw / f"{event.get('event_id')}_p{page.get('page')}_{digest[:12]}.body"
                    dest.write_bytes(src.read_bytes())
                    page_receipt["artifact_body"] = str(dest.relative_to(output_dir))
            for key in ("body_path", "headers_path", "result_path", "curl_line"):
                page_receipt.pop(key, None)
            page["receipt"] = page_receipt
    dumped = json.dumps(portable, ensure_ascii=False)
    if "/.tmp/" in dumped or "/mnt/" in dumped:
        raise BallotEnrichmentError("portable witness bank still contains private paths")
    (output_dir / "portable_ballot_witness_bank.json").write_text(
        dumped + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "matter_files": written,
        "witness_bank_sha256": _sha256_bytes(Path(witness_bank_path).read_bytes()),
        "comparison_sha256": _sha256_bytes(Path(comparison_path).read_bytes()),
        "db_writes": False,
        "private_paths_stripped": True,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


__all__ = [
    "BINDING_STATUS",
    "MATCH_STATUS",
    "SCHEMA_VERSION",
    "BallotEnrichmentError",
    "enrich_matter_files",
    "enrich_matter_packet",
    "validate_witness_event",
]

"""Generic adapters from source-rescue banks into immutable matter packets.

Keeps source-native role text, date fields, hashes and locators. Does not invent
membership intervals, drafting authorship, or institutional outcomes.
"""

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

KKV_BANK_SCHEMA: Final = "bs02.sourcebank_document.v1"
COMMITTEE_BANK_SCHEMA: Final = "bs02.committee_role_sourcebank.v1"
PACKET_SCHEMA: Final = "paa.matter_packet.sourcebank.v1"
_MEMBER_ROLES: Final = frozenset(
    {"jäsen", "varajäsen", "puheenjohtaja", "varapuheenjohtaja"}
)
_SECRETARY_ROLES: Final = frozenset({"valiokuntaneuvos", "sihteeri"})
_ROLE_TOKEN_RE = re.compile(
    r"\b(valiokuntaneuvos|sihteeri|varapuheenjohtaja|puheenjohtaja|varajäsen|jäsen)\b",
    re.IGNORECASE,
)


class MatterSourcebankError(ValueError):
    """Raised when a sourcebank cannot be adapted without inventing fields."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _anchor_value(anchors: Sequence[Mapping[str, Any]], field: str) -> dict[str, Any] | None:
    for row in anchors:
        if not isinstance(row, Mapping):
            continue
        if str(row.get("field") or "") != field:
            continue
        return {
            "field": field,
            "value": row.get("value"),
            "xml_path": row.get("xml_path"),
            "basis": row.get("basis") or "source field_anchors",
        }
    return None


def _role_tokens(*texts: str) -> set[str]:
    found: set[str] = set()
    for text in texts:
        for match in _ROLE_TOKEN_RE.finditer(str(text or "")):
            found.add(match.group(1).casefold())
    return found


def classify_committee_role_family(
    *,
    role_values: Sequence[str],
    xml_path: str,
    source_statement: str,
    classification: str,
) -> str:
    """Map source role tokens/paths to member, secretary, expert, or other.

    Structural hearing/expert XML paths take precedence over member-like title
    tokens in an outside person's source statement (chair of another body can
    appear as an expert witness).
    """

    roles = {str(value).strip().casefold() for value in role_values if str(value).strip()}
    path = str(xml_path or "").casefold()
    # Structure wins: Asiantuntija path is hearing capacity, not committee seat.
    if "asiantuntija" in path:
        return "EXPERT_OR_HEARING"
    roles |= _role_tokens(source_statement)
    if roles & _SECRETARY_ROLES:
        return "COMMITTEE_SECRETARY"
    if roles & _MEMBER_ROLES:
        return "COMMITTEE_MEMBER"
    if str(classification or "") == "EXPLICIT_ROLE" and roles:
        return "OTHER_NAMED_ROLE"
    if str(classification or "") == "DOCUMENTED_PARTICIPATION":
        return "DOCUMENTED_PARTICIPATION"
    return "OTHER_NAMED_ROLE"


def _participation_date_scope(date_semantics: Mapping[str, Any] | None) -> dict[str, Any]:
    blob = date_semantics or {}
    document_dates = [
        {
            "field": row.get("field"),
            "value": row.get("value"),
            "kind": row.get("kind") or "document_date",
            "xml_path": row.get("xml_path"),
        }
        for row in (blob.get("document_date") or [])
        if isinstance(row, Mapping)
    ]
    meeting_dates = [
        {
            "field": row.get("field"),
            "value": row.get("value"),
            "kind": row.get("kind") or "meeting_date",
            "xml_path": row.get("xml_path"),
        }
        for row in (blob.get("meeting_date") or [])
        if isinstance(row, Mapping)
    ]
    signature_dates = [
        {
            "field": row.get("field"),
            "value": row.get("value"),
            "kind": row.get("kind") or "signature_date",
            "xml_path": row.get("xml_path"),
        }
        for row in (blob.get("signature_date") or [])
        if isinstance(row, Mapping)
    ]
    return {
        "document_dates": document_dates,
        "meeting_dates": meeting_dates,
        "signature_dates": signature_dates,
        "meeting_date_status": blob.get("meeting_date_status")
        or ("SUPPLIED" if meeting_dates else "NOT_SUPPLIED_BY_SOURCE"),
        "record_dates": dict(blob.get("record_dates") or {}),
        "capture_time": blob.get("capture_time"),
        "membership_interval_asserted": False,
        "note": (
            "Date scope is source-defined document/meeting/signature/record/capture "
            "fields only; full membership intervals are not inferred."
        ),
    }


def adapt_kkv_response_sourcebank(bank: Mapping[str, Any]) -> dict[str, Any]:
    """Build an immutable matter packet slice from a KKV response sourcebank."""

    if str(bank.get("schema_version") or "") != KKV_BANK_SCHEMA:
        raise MatterSourcebankError(
            f"unsupported KKV bank schema {bank.get('schema_version')!r}"
        )
    source_object = bank.get("source_object") or {}
    document = bank.get("document") or {}
    metadata = bank.get("source_metadata") or {}
    anchors = list(metadata.get("field_anchors") or [])
    people = list(source_object.get("people") or [])
    body = str(document.get("full_text") or "")
    body_sha = str(document.get("full_text_sha256") or "")
    if body and body_sha and _sha256_bytes(body.encode("utf-8")) != body_sha:
        raise MatterSourcebankError("KKV full_text does not match full_text_sha256")
    role_person = people[0] if people else {}
    laadinta = _anchor_value(anchors, "laadintaPvm")
    role_anchor = None
    for row in anchors:
        if str(row.get("field") or "") == "rooliKoodi" and str(row.get("value") or "") == str(
            role_person.get("role_code") or "Laatija"
        ):
            role_anchor = row
            break
    matter_id = str(source_object.get("matter_id") or "")
    document_id = str(
        (_anchor_value(anchors, "eduskuntaTunnus") or {}).get("value")
        or source_object.get("object_id")
        or ""
    )
    answer_document = {
        "kind": source_object.get("kind") or "GOVERNMENT_ANSWER",
        "document_id": document_id,
        "matter_id": matter_id,
        "object_id": source_object.get("object_id"),
        "title": source_object.get("title"),
        "body_text": body,
        "body_text_sha256": body_sha,
        "pdf_sha256": document.get("pdf_sha256"),
        "page_count": document.get("page_count"),
        "source_url": (bank.get("source_questionbank") or {}).get("source_url"),
        "role": {
            "code": role_person.get("role_code"),
            "position": role_person.get("position"),
            "given_name": role_person.get("given_name"),
            "family_name": role_person.get("family_name"),
            "xml_path": role_person.get("xml_path") or (role_anchor or {}).get("xml_path"),
            "identity_basis": source_object.get("identity_basis") or "source XML fields only",
            "anchor": {
                "field": (role_anchor or {}).get("field"),
                "value": (role_anchor or {}).get("value"),
                "basis": (role_anchor or {}).get("basis"),
            }
            if role_anchor
            else None,
        },
        "dates": {
            "laadintaPvm": (laadinta or {}).get("value") or source_object.get("date"),
            "field": (laadinta or {}).get("field") or "laadintaPvm",
            "xml_path": (laadinta or {}).get("xml_path"),
            "basis": (laadinta or {}).get("basis") or source_object.get("identity_basis"),
        },
        "source_records": [
            {
                "record_locator": row.get("record_locator"),
                "source_url": row.get("source_url"),
                "raw_sha256": row.get("raw_sha256"),
                "body_bytes": row.get("body_bytes"),
                "captured_at": row.get("captured_at"),
            }
            for row in (bank.get("source_records") or [])
            if isinstance(row, Mapping)
        ],
        "questions": list((bank.get("source_questionbank") or {}).get("questions") or []),
        "raw_span": (bank.get("source_questionbank") or {}).get("raw_span"),
        "pages": [
            {
                "page": page.get("page"),
                "text": page.get("text"),
                "text_sha256": page.get("text_sha256"),
                "source_locator": page.get("source_locator"),
            }
            for page in (document.get("pages") or [])
            if isinstance(page, Mapping)
        ],
        "pdf_metadata": {
            "pdf_sha256": document.get("pdf_sha256"),
            "page_count": document.get("page_count"),
            "content_type": document.get("content_type")
            or (metadata.get("attachment_row") or {}).get("ContentType"),
            "md5": document.get("md5")
            or (metadata.get("attachment_row") or {}).get("HashValue")
            or (metadata.get("attachment_row") or {}).get("Md5"),
            "last_modified": document.get("last_modified")
            or (metadata.get("attachment_row") or {}).get("LastModified"),
            "capture_time_separate_from_answer_date": True,
        },
        "admission": "SOURCE_NATIVE_FIELDS_ONLY",
    }
    interventions = []
    if role_person:
        name = " ".join(
            part
            for part in (role_person.get("given_name"), role_person.get("family_name"))
            if part
        ).strip()
        interventions.append(
            {
                "kind": "GOVERNMENT_ANSWER",
                "role_label": "RESPONDENT",
                "action_kind": {"value": "ANSWER_ATTRIBUTED", "basis": "SOURCE_XML_ROLE"},
                "action_date": answer_document["dates"].get("laadintaPvm"),
                "action_date_basis": "SOURCE_LAADINTA_PVM",
                "object_id": source_object.get("object_id"),
                "title": source_object.get("title"),
                "source_text": body,
                "source_text_excerpt": body[:2000],
                "source_text_is_excerpt": len(body) > 2000,
                "actor": {
                    "name": name or None,
                    "identity_state": "SOURCE_NAME_ONLY_UNJOINED",
                    "source_display_name": name or None,
                },
                "source": {
                    "url": answer_document.get("source_url"),
                    "record_locator": (
                        (answer_document["source_records"][0].get("record_locator"))
                        if answer_document["source_records"]
                        else document_id
                    ),
                    "raw_sha256": (
                        answer_document["source_records"][0].get("raw_sha256")
                        if answer_document["source_records"]
                        else None
                    ),
                    "text_sha256": body_sha,
                },
                "institutional_result": {
                    "state": "RECORDED",
                    "raw_state": (_anchor_value(anchors, "tilaKoodi") or {}).get("value"),
                    "scope": "ANSWER_DOCUMENT_RECORDING_ONLY",
                },
                "influence": {"state": "INFLUENCE_NOT_INFERRED"},
                "drafting": {"state": "UNRESOLVED", "value": None},
                "evidence_ids": [
                    row.get("raw_sha256")
                    for row in answer_document["source_records"]
                    if row.get("raw_sha256")
                ],
            }
        )
    return {
        "schema_version": PACKET_SCHEMA,
        "query": {"mode": "EXACT_MATTER", "text": matter_id, "actor": None},
        "matter": {
            "matter_id": matter_id,
            "subject_title": source_object.get("title"),
            "subject_title_state": "COMMON_OBJECT_TITLE" if source_object.get("title") else "MISSING",
            "subject_title_candidates": [source_object.get("title")] if source_object.get("title") else [],
            "objects": [
                {
                    "kind": source_object.get("kind") or "GOVERNMENT_ANSWER",
                    "object_id": source_object.get("object_id"),
                    "title": source_object.get("title"),
                    "date": source_object.get("date"),
                }
            ],
            "vote_sources": [],
        },
        "answer_document": answer_document,
        "interventions": interventions,
        "coverage": [
            {
                "kind": "GOVERNMENT_ANSWER",
                "declared_scope": "sourcebank answer document body and XML role/date anchors",
                "state": "DECLARED",
            }
        ],
        "missingness": [],
        "limits": [
            "Answer role, date and body are source-native fields from the KKV bank.",
            "No membership interval, drafting authorship, influence, or final adoption is asserted.",
        ],
        "sourcebank_provenance": {
            "bank_schema": KKV_BANK_SCHEMA,
            "source_id": bank.get("source_id"),
            "generated_at": bank.get("generated_at"),
        },
    }


def adapt_committee_role_sourcebank(
    bank: Mapping[str, Any],
    *,
    matter_id: str | None = None,
) -> dict[str, Any]:
    """Build an immutable matter packet with named committee participation rows."""

    if str(bank.get("schema_version") or "") != COMMITTEE_BANK_SCHEMA:
        raise MatterSourcebankError(
            f"unsupported committee bank schema {bank.get('schema_version')!r}"
        )
    matters = [row for row in (bank.get("matters") or []) if isinstance(row, Mapping)]
    if matter_id:
        matters = [row for row in matters if str(row.get("matter") or "") == matter_id]
    if not matters:
        raise MatterSourcebankError(f"no committee matter rows for {matter_id!r}")
    selected = matters[0]
    selected_matter_id = str(selected.get("matter") or "")
    report = selected.get("report") or {}
    rows_out: list[dict[str, Any]] = []
    interventions: list[dict[str, Any]] = []
    for index, row in enumerate(selected.get("named_participation_or_role") or []):
        if not isinstance(row, Mapping):
            continue
        role_fields = [item for item in (row.get("role_fields") or []) if isinstance(item, Mapping)]
        role_values = [str(item.get("value") or "") for item in role_fields]
        locator = row.get("source_locator") or {}
        xml_path = str(locator.get("xml_path") or "")
        family = classify_committee_role_family(
            role_values=role_values,
            xml_path=xml_path,
            source_statement=str(row.get("source_statement") or ""),
            classification=str(row.get("classification") or ""),
        )
        date_scope = _participation_date_scope(
            row.get("date_semantics") if isinstance(row.get("date_semantics"), Mapping) else None
        )
        if row.get("membership_interval_asserted"):
            raise MatterSourcebankError("bank asserted membership interval; refuse to adapt")
        if row.get("drafting_authorship_asserted"):
            raise MatterSourcebankError("bank asserted drafting authorship; refuse to adapt")
        action_date = None
        if date_scope["document_dates"]:
            action_date = date_scope["document_dates"][0].get("value")
        elif date_scope["meeting_dates"]:
            action_date = date_scope["meeting_dates"][0].get("value")
        adapted = {
            "person_name": row.get("person_name"),
            "source_statement": row.get("source_statement"),
            "role_family": family,
            "source_roles": role_values,
            "role_fields": role_fields,
            "classification": row.get("classification"),
            "committee_fields": row.get("committee_fields") or [],
            "participation_date_scope": date_scope,
            "membership_interval_asserted": False,
            "drafting_authorship_asserted": False,
            "source_locator": {
                "record_id": locator.get("record_id"),
                "record_identifier": locator.get("record_identifier"),
                "xml_sha256": locator.get("xml_sha256"),
                "xml_path": locator.get("xml_path"),
            },
        }
        rows_out.append(adapted)
        report_url = None
        for receipt in report.get("report_page_receipts") or []:
            if isinstance(receipt, Mapping) and (
                receipt.get("final_url") or receipt.get("request_url")
            ):
                report_url = receipt.get("final_url") or receipt.get("request_url")
                break
        interventions.append(
            {
                "kind": "COMMITTEE_NAMED_PARTICIPATION",
                "role_label": family,
                "action_kind": {
                    "value": "OTHER_OBSERVABLE_ACTION",
                    "basis": "COMMITTEE_ROLE_SOURCEBANK",
                },
                "action_date": action_date,
                "action_date_basis": "SOURCE_DOCUMENT_OR_MEETING_DATE_IF_SUPPLIED",
                "object_id": f"committee:{selected_matter_id}:{index}",
                "title": report.get("source_reference_text") or report.get("report_identifier"),
                "source_text": row.get("source_statement"),
                "actor": {
                    "name": row.get("person_name"),
                    "identity_state": "SOURCE_NAME_ONLY_UNJOINED",
                    "source_display_name": row.get("person_name"),
                },
                "source": {
                    "url": report_url,
                    "record_locator": (
                        f"{locator.get('record_identifier') or ''}#"
                        f"{locator.get('xml_path') or ''}"
                    ).strip("#"),
                    "raw_sha256": locator.get("xml_sha256"),
                },
                "committee_role": adapted,
                "institutional_result": {
                    "state": "RECORDED",
                    "scope": "NAMED_PARTICIPATION_RECORDING_ONLY",
                },
                "influence": {"state": "INFLUENCE_NOT_INFERRED"},
                "drafting": {"state": "UNRESOLVED", "value": None},
                "evidence_ids": [locator.get("xml_sha256")] if locator.get("xml_sha256") else [],
            }
        )
    report_receipts = []
    for receipt in report.get("report_page_receipts") or []:
        if not isinstance(receipt, Mapping):
            continue
        report_receipts.append(
            {
                key: receipt.get(key)
                for key in (
                    "schema_version",
                    "kind",
                    "request_url",
                    "final_url",
                    "status",
                    "content_type",
                    "body_bytes",
                    "sha256",
                    "captured_at",
                    "completed_at",
                )
                if key in receipt
            }
        )
    procedural = report.get("source_reference_procedural_event")
    return {
        "schema_version": PACKET_SCHEMA,
        "query": {"mode": "EXACT_MATTER", "text": selected_matter_id, "actor": None},
        "matter": {
            "matter_id": selected_matter_id,
            "subject_title": report.get("source_reference_text"),
            "subject_title_state": (
                "COMMON_OBJECT_TITLE" if report.get("source_reference_text") else "MISSING"
            ),
            "subject_title_candidates": (
                [report.get("source_reference_text")] if report.get("source_reference_text") else []
            ),
            "objects": [
                {
                    "kind": "COMMITTEE_REPORT",
                    "object_id": report.get("report_identifier"),
                    "title": report.get("source_reference_text"),
                    "committee_code": report.get("committee_code"),
                }
            ],
            "vote_sources": [],
        },
        "committee_named_participation": {
            "schema_version": "paa.committee_named_participation.v1",
            "matter_id": selected_matter_id,
            "classification_basis": selected.get("classification_basis"),
            "controls": dict(bank.get("controls") or {}),
            "report": {
                "report_identifier": report.get("report_identifier"),
                "committee_code": report.get("committee_code"),
                "source_reference_text": report.get("source_reference_text"),
                "source_reference_record_id": report.get("source_reference_record_id"),
                "source_reference_xml_path": report.get("source_reference_xml_path"),
                "source_reference_procedural_event": procedural if isinstance(procedural, Mapping) else None,
                "report_page_receipts": report_receipts,
            },
            "rows": rows_out,
        },
        "interventions": interventions,
        "coverage": [
            {
                "kind": "COMMITTEE_NAMED_PARTICIPATION",
                "declared_scope": "Exact XML source nodes exported in the committee role bank",
                "state": "DECLARED",
            }
        ],
        "missingness": [],
        "limits": [
            "Named participation and explicit roles are retained separately from membership intervals.",
            "Drafting authorship is not inferred.",
            "Meeting-date absence remains explicit when the inspected report node does not supply it.",
        ],
        "sourcebank_provenance": {
            "bank_schema": COMMITTEE_BANK_SCHEMA,
            "source_id": bank.get("source_id"),
            "generated_at": bank.get("generated_at"),
            "matter_status": selected.get("status"),
        },
    }


def merge_matter_packets(*packets: Mapping[str, Any]) -> dict[str, Any]:
    """Deterministically merge adapted packets for the same matter_id."""

    if not packets:
        raise MatterSourcebankError("no packets to merge")
    matter_ids = {
        str((packet.get("matter") or {}).get("matter_id") or "") for packet in packets
    }
    if len(matter_ids) != 1 or not next(iter(matter_ids)):
        raise MatterSourcebankError(f"merge requires one matter_id, got {sorted(matter_ids)}")
    out = json.loads(json.dumps(packets[0], ensure_ascii=False))
    for packet in packets[1:]:
        for key in ("answer_document", "committee_named_participation"):
            if packet.get(key) is not None:
                out[key] = packet[key]
        out.setdefault("interventions", []).extend(list(packet.get("interventions") or []))
        out.setdefault("limits", []).extend(list(packet.get("limits") or []))
        out.setdefault("coverage", []).extend(list(packet.get("coverage") or []))
        prov = out.setdefault("sourcebank_provenance", {})
        if isinstance(prov, dict) and packet.get("sourcebank_provenance"):
            merged = prov.setdefault("merged", [])
            if isinstance(merged, list):
                merged.append(packet.get("sourcebank_provenance"))
        matter = out.setdefault("matter", {})
        objects = matter.setdefault("objects", [])
        if isinstance(objects, list):
            objects.extend(list((packet.get("matter") or {}).get("objects") or []))
    out["schema_version"] = PACKET_SCHEMA
    return out


def write_sourcebank_matter_artifacts(
    *,
    kkv_bank_path: Path | None = None,
    committee_bank_path: Path | None = None,
    matter_id: str | None = None,
    output_dir: Path,
    html_name: str = "matter.html",
) -> dict[str, Any]:
    """Adapt banks into JSON + public HTML under a unique output directory."""

    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise MatterSourcebankError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    packets: list[dict[str, Any]] = []
    inputs: dict[str, str] = {}
    if kkv_bank_path is not None:
        path = Path(kkv_bank_path)
        bank = _load_json(path)
        packets.append(adapt_kkv_response_sourcebank(bank))
        inputs["kkv_bank"] = str(path)
        inputs["kkv_bank_sha256"] = _sha256_bytes(path.read_bytes())
    if committee_bank_path is not None:
        path = Path(committee_bank_path)
        bank = _load_json(path)
        packets.append(adapt_committee_role_sourcebank(bank, matter_id=matter_id))
        inputs["committee_bank"] = str(path)
        inputs["committee_bank_sha256"] = _sha256_bytes(path.read_bytes())
    if not packets:
        raise MatterSourcebankError("provide --kkv-bank and/or --committee-bank")
    packet = packets[0] if len(packets) == 1 else merge_matter_packets(*packets)
    packet_path = output_dir / "matter_packet.json"
    packet_path.write_text(json.dumps(packet, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    from paa.matter_site import write_matter_site

    site = write_matter_site(
        packet,
        output_dir / html_name,
        packet_url="matter_packet.json",
    )
    manifest = {
        "schema_version": "paa.matter_sourcebank_build.v1",
        "inputs": inputs,
        "matter_id": (packet.get("matter") or {}).get("matter_id"),
        "packet": packet_path.name,
        "site": site,
        "db_writes": False,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


__all__ = [
    "COMMITTEE_BANK_SCHEMA",
    "KKV_BANK_SCHEMA",
    "PACKET_SCHEMA",
    "MatterSourcebankError",
    "adapt_committee_role_sourcebank",
    "adapt_kkv_response_sourcebank",
    "classify_committee_role_family",
    "merge_matter_packets",
    "write_sourcebank_matter_artifacts",
]

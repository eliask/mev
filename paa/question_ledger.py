"""Acquire the current-term written-question register from VaskiData.

The Vaski register stores a written question, its processing history, and a
government reply as different document rows.  This adapter keeps those
records separate.  A question object may point at a separate answer object,
but an institutional answer is never treated as policy implementation.

The public rows endpoint has no stable ordering parameter.  The acquisition
therefore persists every raw page and records page overlap instead of silently
calling a repeated page a complete register.  Cached pages are verified by
their receipt hash before replay.  The parser itself is deterministic and can
be exercised entirely from frozen JSON page slices.
"""


import hashlib
import json
import re
import sqlite3
from collections import defaultdict
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import httpx
from lxml import etree

from paa.acquire_initiatives import SOURCE_ID, VASKI_ROWS_URL
from paa.config import CORPUS_CUTOFF, RAW, USER_AGENT

QUESTION_SOURCE_ID = SOURCE_ID
QUESTION_KIND = "WRITTEN_QUESTION"
ANSWER_KIND = "GOVERNMENT_ANSWER"
DEFAULT_YEARS = (2023, 2024, 2025, 2026)
_MATTER = re.compile(r"^(?P<prefix>KKV|KK) (?P<number>\d+)/(?P<year>\d{4}) vp$", re.IGNORECASE)
_DATE = re.compile(r"^(?P<year>\d{4})-(?P<month>\d{1,2})-(?P<day>\d{1,2})")
_FINNISH_DATE = re.compile(r"^(?P<day>\d{1,2})[.]\s*(?P<month>\d{1,2})[.]\s*(?P<year>\d{4})")


def _clean(value: Any) -> str:
    return " ".join(str(value or "").split())


def _local(element: etree._Element) -> str:
    return etree.QName(element).localname if isinstance(element.tag, str) else ""


def _attrs(element: etree._Element | None) -> dict[str, str]:
    if element is None:
        return {}
    return {key.rsplit("}", 1)[-1]: str(value).strip() for key, value in element.attrib.items()}


def _date(value: Any) -> str | None:
    text = _clean(value)
    match = _DATE.search(text) or _FINNISH_DATE.search(text)
    if not match:
        return None
    groups = match.groupdict()
    if "day" in groups and groups.get("day") is not None:
        return f"{int(groups['year']):04d}-{int(groups['month']):02d}-{int(groups['day']):02d}"
    return f"{int(groups['year']):04d}-{int(groups['month']):02d}-{int(groups['day']):02d}"


def _element_text(element: etree._Element) -> str:
    return _clean(" ".join(element.itertext()))


def _metadata(root: etree._Element) -> etree._Element | None:
    return next((element for element in root.iter() if _local(element) == "JulkaisuMetatieto"), None)


def _first_text(root: etree._Element, names: set[str]) -> str:
    for element in root.iter():
        if _local(element) in names:
            text = _element_text(element)
            if text:
                return text
    return ""


def _document_id(root: etree._Element) -> str:
    metadata = _metadata(root)
    candidate = _attrs(metadata).get("eduskuntaTunnus", "")
    if _MATTER.match(candidate):
        return candidate
    for element in root.iter():
        candidate = _attrs(element).get("eduskuntaTunnus", "")
        if _MATTER.match(candidate):
            return candidate
        if _local(element) in {"EduskuntaTunnus", "Vireilletulo"}:
            candidate = _element_text(element)
            if _MATTER.match(candidate):
                return candidate
    return ""


def _question_matter(document_id: str, root: etree._Element) -> str:
    match = _MATTER.match(document_id)
    if not match:
        return ""
    if match.group("prefix").upper() == "KK":
        return document_id
    # KKV metadata intentionally points back to the KK matter.  Require an
    # explicit cross-reference before accepting that join.
    for element in root.iter():
        if _local(element) not in {"Vireilletulo", "EduskuntaTunnus"}:
            continue
        candidate = _element_text(element)
        if re.match(r"^KK \d+/\d{4} vp$", candidate, re.IGNORECASE):
            return candidate
    return "KK " + document_id[4:]


def _document_kind(document_id: str) -> str | None:
    match = _MATTER.match(document_id)
    if not match:
        return None
    prefix = match.group("prefix").upper()
    if prefix == "KK":
        return QUESTION_KIND
    if prefix == "KKV":
        return ANSWER_KIND
    return None


def _content_root(root: etree._Element, kind: str) -> etree._Element | None:
    names = {"Kysymys", "KirjallinenKysymys"} if kind == QUESTION_KIND else {"Vastaus", "VastausKirjalliseenKysymykseen"}
    return next((element for element in root.iter() if _local(element) in names), None)


def _title(root: etree._Element) -> str:
    return _first_text(root, {"NimekeTeksti"})


def _signature_date(root: etree._Element) -> str | None:
    """Return an explicit signer date, never the publication metadata date."""

    for element in root.iter():
        if _local(element) != "PaivaysKooste":
            continue
        parsed = _date(_attrs(element).get("allekirjoitusPvm"))
        if parsed:
            return parsed
        parsed = _date(_element_text(element))
        if parsed:
            return parsed
    return None


def _body(root: etree._Element, kind: str, title: str) -> str:
    content = _content_root(root, kind)
    if content is None:
        return title
    values: list[str] = []
    seen: set[str] = set()
    for element in content.iter():
        if _local(element) not in {
            "OtsikkoTeksti",
            "KappaleKooste",
            "MomenttiKooste",
            "PykalaTunnusKooste",
            "PykalaNimekeKooste",
            "SaadosNimekeKooste",
            "FraasiKappaleKooste",
            # Written questions keep the actual interrogative in a separate
            # petition block.  It is not a ``KappaleKooste`` and must not be
            # dropped merely because the preceding background text is long.
            "JohdantoTeksti",
            "SisennettyKappaleKooste",
            "KursiiviTeksti",
        }:
            continue
        text = _element_text(element)
        if text and text not in seen:
            values.append(text)
            seen.add(text)
    if not values:
        values = [title] if title else []
    return "\n\n".join(values)


def _person_name(person: etree._Element) -> str:
    first = _first_text(person, {"EtuNimi", "Etunimi"})
    last = _first_text(person, {"SukuNimi", "Sukunimi"})
    if first or last:
        return _clean(f"{first} {last}")
    return _first_text(person, {"Nimi", "NimiTeksti"}) or _element_text(person)


def _person_id(person: etree._Element, parent: etree._Element | None = None) -> str | None:
    for element in (person, parent):
        attrs = _attrs(element)
        for name in ("muuTunnus", "personId", "henkiloTunnus"):
            if attrs.get(name):
                return attrs[name]
    return None


def _authors(root: etree._Element, kind: str, evidence_id: str) -> list[dict[str, Any]]:
    """Extract source-backed signers/respondents without inventing actor IDs."""

    found: list[dict[str, Any]] = []
    index: dict[str, int] = {}
    name_index: dict[str, int] = {}

    def add(person: etree._Element, parent: etree._Element | None, role: str) -> None:
        name = _clean(_person_name(person))
        person_id = _person_id(person, parent)
        name_key = name.casefold()
        if not person_id and not name_key:
            return
        # The content row repeats a signer in both ``Allekirjoittaja`` and
        # ``Toimija`` metadata.  One copy may have only a name and the other
        # a source person ID; merge that safe identity upgrade while keeping
        # two conflicting source IDs separate.
        position = index.get(person_id) if person_id else None
        if position is None and name_key:
            candidate = name_index.get(name_key)
            if candidate is not None:
                existing_id = found[candidate].get("person_id")
                # A name-only duplicate can be upgraded to the already
                # observed source ID.  Two different non-empty IDs are kept
                # distinct even when display names collide.
                if not existing_id or not person_id or existing_id == person_id:
                    position = candidate
        if position is not None:
            existing = found[position]
            if person_id and not existing.get("person_id"):
                existing["person_id"] = person_id
                existing["identity_basis"] = "SOURCE_PERSON_ID"
                index[person_id] = position
            if evidence_id not in existing["evidence_ids"]:
                existing["evidence_ids"].append(evidence_id)
            if existing["role"] == "UNRESOLVED" and role != "UNRESOLVED":
                existing["role"] = role
            return
        position = len(found)
        if person_id:
            index[person_id] = position
        if name_key:
            name_index[name_key] = position
        found.append({
            "person_id": person_id,
            "name": name,
            "role": role,
            "identity_basis": "SOURCE_PERSON_ID" if person_id else "SOURCE_NAME_ONLY",
            "evidence_ids": [evidence_id],
        })

    signatures = [element for element in root.iter() if _local(element) == "Allekirjoittaja"]
    for number, signature in enumerate(signatures):
        person = next((child for child in signature.iter() if _local(child) == "Henkilo"), None)
        if person is not None:
            add(person, signature, "AUTHOR" if number == 0 else "COSIGNER")
    for actor in root.iter():
        if _local(actor) != "Toimija":
            continue
        role_text = _clean(_attrs(actor).get("rooliKoodi") or _element_text(actor)).casefold()
        person = next((child for child in actor.iter() if _local(child) == "Henkilo"), None)
        if person is None:
            continue
        if kind == QUESTION_KIND:
            # Procedural question rows also name the minister who answered
            # the question.  That respondent must not become a cosigner.
            if not any(token in role_text for token in ("laatija", "ensimm", "allekirjoittaja")):
                continue
            role = "AUTHOR" if "laatija" in role_text or "ensimm" in role_text else "COSIGNER"
        else:
            role = "RESPONDENT" if "vastannut" in role_text or "minister" in role_text or "laatija" in role_text else "UNRESOLVED"
        add(person, actor, role)
    return found


def _procedure_events(root: etree._Element, evidence_id: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for element in root.iter():
        if _local(element) != "ToimenpideJulkaisu":
            continue
        attrs = _attrs(element)
        when = _date(attrs.get("tapahtumaPvm"))
        if not when:
            when = _date(_first_text(element, {"TapahtumaPvmTeksti"}))
        if not when:
            continue
        code = _clean(attrs.get("kasittelyvaiheKoodi")).upper()
        label = _first_text(element, {"ValiotsikkoTeksti", "OtsikkoTeksti"})
        body = _first_text(element, {"FraasiKappaleKooste", "FraasiPerus"})
        text = _clean(" ".join(part for part in (when, label, body) if part))
        lower = f"{code} {label} {body}".casefold()
        if code in {"JATTO", "VIR"} or "kysymys jätetty" in lower or "kysymys jatetty" in lower:
            event_kind = "SUBMISSION"
            basis = "VIREILLETULO_EVENT"
        elif code.startswith("VAST") or "vastaus annettu" in lower:
            event_kind = "ANSWER_RECEIVED"
            basis = "ANSWER_EVENT"
        elif code.startswith("ILM") and "vastaus" in lower:
            event_kind = "ANSWER_ANNOUNCED"
            basis = "ANSWER_ANNOUNCEMENT_EVENT"
        else:
            continue
        events.append({
            "kind": event_kind,
            "date": when,
            "basis": basis,
            "code": code,
            "label": label,
            "quote": text or when,
            "evidence_ids": [evidence_id],
        })
    return events


def _source_url(document_id: str) -> str:
    query = urlencode({"columnName": "Eduskuntatunnus", "columnValue": document_id, "perPage": 100, "page": 0})
    return f"{VASKI_ROWS_URL}?{query}"


def _record_class(root: etree._Element, kind: str) -> str:
    if _content_root(root, kind) is not None:
        return "CONTENT"
    return "PROCEDURAL"


def parse_question_xml(
    xml: str,
    *,
    row_metadata: Mapping[str, Any] | None = None,
    source_url: str | None = None,
    retrieved_at: str | None = None,
) -> dict[str, Any]:
    """Parse one KK/KKV XML row into a provenance-bearing source record."""

    if not isinstance(xml, str) or not xml.strip():
        raise ValueError("Vaski question XML must be non-empty")
    raw_bytes = xml.encode("utf-8")
    raw_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    try:
        parser = etree.XMLParser(resolve_entities=False, no_network=True, load_dtd=False, recover=False)
        root = etree.fromstring(raw_bytes, parser=parser)
    except etree.XMLSyntaxError as exc:
        raise ValueError(f"invalid Vaski question XML: {exc}") from exc
    document_id = _document_id(root)
    kind = _document_kind(document_id)
    if not kind:
        raise ValueError(f"not a KK or KKV document: {document_id or '<missing>'}")
    matter_id = _question_matter(document_id, root)
    if not matter_id:
        raise ValueError(f"{document_id} has no linked KK matter")
    row = dict(row_metadata or {})
    row_id = str(row.get("Id") or row.get("id") or "").strip() or None
    record_locator = f"VaskiData/Id={row_id}" if row_id else f"VaskiData/{document_id}"
    source_url = source_url or _source_url(document_id)
    metadata = _metadata(root)
    publication_date = _date(_attrs(metadata).get("laadintaPvm"))
    title = _title(root)
    text = _body(root, kind, title)
    signature_date = _signature_date(root)
    evidence_id = f"{QUESTION_SOURCE_ID}:{raw_sha256[:16]}:record"
    events = _procedure_events(root, evidence_id)
    evidence = {
        "evidence_id": evidence_id,
        "document_version_id": record_locator,
        "kind": "text_span" if text else "structured_field",
        "text_sha256": hashlib.sha256((text or title).encode("utf-8")).hexdigest() if (text or title) else None,
        "span_start": 0 if (text or title) else None,
        "span_end": len(text or title) if (text or title) else None,
        "quote": text or title or None,
        "normalization_version": "nfc-1",
        "record_locator": record_locator,
        "field_path": "question_text" if kind == QUESTION_KIND else "answer_metadata",
        "context_evidence_ids": [],
        "source_id": QUESTION_SOURCE_ID,
        "source_url": source_url,
        "url": source_url,
        "raw_sha256": raw_sha256,
        "byte_length": len(raw_bytes),
        "retrieved_at": retrieved_at or datetime.now(UTC).replace(microsecond=0).isoformat(),
    }
    return {
        "document_id": document_id,
        "matter_id": matter_id,
        "kind": kind,
        "title": title,
        "text": text,
        "publication_date": publication_date,
        "date": publication_date,
        "signature_date": signature_date,
        "signature_date_basis": "SIGNATURE_DATE" if signature_date else None,
        "authors": _authors(root, kind, evidence_id),
        "events": events,
        "evidence_ids": [evidence_id],
        "evidence": [evidence],
        "source_id": QUESTION_SOURCE_ID,
        "record_class": _record_class(root, kind),
        "record_locator": record_locator,
        "raw_sha256": raw_sha256,
        "source_row": {
            "id": row_id,
            "status": row.get("Status"),
            "created": row.get("Created"),
            "imported": row.get("Imported"),
            "attachment_group_id": row.get("AttachmentGroupId"),
        },
    }


def parse_question_row(
    row: Mapping[str, Any], *, source_url: str | None = None, retrieved_at: str | None = None
) -> dict[str, Any]:
    xml = row.get("XmlData") or row.get("XmlDataFi") or row.get("xml")
    if not isinstance(xml, str) or not xml.strip():
        raise ValueError("Vaski question row has no XmlData")
    return parse_question_xml(xml, row_metadata=row, source_url=source_url, retrieved_at=retrieved_at)


def _event_evidence(record: Mapping[str, Any], event: Mapping[str, Any]) -> dict[str, Any]:
    raw_sha256 = str(record["raw_sha256"])
    event_id = f"{QUESTION_SOURCE_ID}:{raw_sha256[:16]}:event-{str(event['kind']).lower()}-{event['date']}"
    return {
        "evidence_id": event_id,
        "document_version_id": record["record_locator"],
        "kind": "structured_field",
        "text_sha256": hashlib.sha256(str(event["quote"]).encode("utf-8")).hexdigest(),
        "span_start": None,
        "span_end": None,
        "quote": event["quote"],
        "normalization_version": "nfc-1",
        "record_locator": record["record_locator"],
        "field_path": f"procedure.{str(event['kind']).lower()}",
        "context_evidence_ids": [record["evidence_ids"][0]],
        "source_id": QUESTION_SOURCE_ID,
        "source_url": record["evidence"][0]["source_url"],
        "url": record["evidence"][0]["source_url"],
        "raw_sha256": raw_sha256,
        "byte_length": record["evidence"][0]["byte_length"],
    }


def _dedupe_records(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for record in records:
        key = (str(record.get("record_locator") or ""), str(record.get("raw_sha256") or ""))
        if key in seen:
            continue
        seen.add(key)
        result.append(dict(record))
    return result


def _record_summary(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "document_id": record.get("document_id"),
        "record_locator": record.get("record_locator"),
        "raw_sha256": record.get("raw_sha256"),
        "record_class": record.get("record_class"),
        "publication_date": record.get("publication_date"),
        "evidence_ids": list(record.get("evidence_ids") or []),
        "event_count": len(record.get("events") or []),
    }


def normalize_question_records(records: Iterable[Mapping[str, Any]], *, coverage_scope: str = "current_term") -> dict[str, Any]:
    """Build distinct WRITTEN_QUESTION and GOVERNMENT_ANSWER objects."""

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in _dedupe_records(records):
        grouped[str(record.get("matter_id") or "")].append(dict(record))
    objects: list[dict[str, Any]] = []
    evidence: dict[str, dict[str, Any]] = {}
    identity_counts = {"questions": 0, "answers": 0, "authors_with_person_id": 0, "authors_name_only": 0,
                       "questions_with_submission_event": 0, "questions_without_submission_event": 0,
                       "questions_with_answer_event": 0, "questions_without_answer_event": 0}
    for matter_id, group in sorted(grouped.items()):
        if not matter_id:
            continue
        questions = [record for record in group if record["kind"] == QUESTION_KIND]
        answers = [record for record in group if record["kind"] == ANSWER_KIND]
        if not questions and not answers:
            continue
        content_questions = [record for record in questions if record["record_class"] == "CONTENT" and record.get("text")]
        question_base = max(content_questions or questions, key=lambda record: len(record.get("text") or ""), default=None)
        if question_base is None:
            # A reply record can be present even when the public question body
            # is unavailable.  Keep the answer trace, but do not fabricate a
            # question object from the answer title.
            question_base = None
        all_events = [event | {"record": record} for record in group for event in record.get("events") or []]
        submissions = [event for event in all_events if event["kind"] == "SUBMISSION"]
        answer_events = [event for event in all_events if event["kind"] in {"ANSWER_RECEIVED", "ANSWER_ANNOUNCED"}]
        submission = min(submissions, key=lambda event: (event["date"], event["record"]["record_locator"])) if submissions else None
        answer_event = min(answer_events, key=lambda event: (event["date"], event["record"]["record_locator"])) if answer_events else None
        for record in group:
            for item in record.get("evidence") or []:
                evidence[item["evidence_id"]] = item
            for event in record.get("events") or []:
                event_record = _event_evidence(record, event)
                event["evidence_ids"] = [event_record["evidence_id"]]
                evidence[event_record["evidence_id"]] = event_record
        answer_document = max(answers, key=lambda record: len(record.get("text") or ""), default=None)
        answer_object_id = f"eduskunta:{answer_document['document_id']}" if answer_document else f"eduskunta:{matter_id}:answer"
        answer_date = answer_event["date"] if answer_event else None
        answer_refs = list(answer_event["evidence_ids"]) if answer_event else []
        if answer_document:
            answer_refs.extend(answer_document.get("evidence_ids") or [])
        answer_refs = list(dict.fromkeys(answer_refs))
        if question_base is not None:
            question_id = f"eduskunta:{matter_id}"
            question_evidence_ids: list[str] = []
            question_evidence: list[dict[str, Any]] = []
            for record in questions:
                question_evidence_ids.extend(record.get("evidence_ids") or [])
                question_evidence.extend(record.get("evidence") or [])
            if submission:
                question_evidence_ids.extend(submission["evidence_ids"])
            if answer_event:
                question_evidence_ids.extend(answer_event["evidence_ids"])
            question_evidence_ids = list(dict.fromkeys(question_evidence_ids))
            question_evidence = [evidence[key] for key in question_evidence_ids if key in evidence]
            authors = question_base.get("authors") or []
            authors_with_id = sum(bool(author.get("person_id")) for author in authors)
            identity_counts["authors_with_person_id"] += authors_with_id
            identity_counts["authors_name_only"] += sum(not bool(author.get("person_id")) for author in authors)
            question_obj = {
                "object_id": question_id,
                "kind": QUESTION_KIND,
                "matter_id": matter_id,
                "document_id": question_base.get("document_id"),
                "title": question_base.get("title") or matter_id,
                "text": question_base.get("text") or question_base.get("title") or "",
                "date": question_base.get("publication_date"),
                "publication_date": question_base.get("publication_date"),
                "signature_date": question_base.get("signature_date"),
                "signature_date_basis": question_base.get("signature_date_basis"),
                "action_date": submission["date"] if submission else None,
                # The trace contract currently admits SUBMISSION_DATE for a
                # written question.  Keep the exact Vireilletulo event basis
                # alongside it rather than calling metadata publication date a
                # filing date.
                "action_date_basis": "SUBMISSION_DATE" if submission else None,
                "action_date_provenance": {
                    "basis": submission["basis"],
                    "event_code": submission.get("code"),
                    "evidence_ids": submission["evidence_ids"],
                } if submission else None,
                "authors": authors,
                "evidence_ids": question_evidence_ids,
                "evidence": question_evidence,
                "source_id": QUESTION_SOURCE_ID,
                "url": question_base["evidence"][0]["source_url"],
                "source_records": [_record_summary(record) for record in questions],
                "answer_object_id": answer_object_id if answer_event or answer_document else None,
                "disposition": {
                    "state": "ANSWERED" if answer_event else "UNRESOLVED",
                    "date": answer_date,
                    "evidence_ids": answer_refs[:1] if answer_event else [],
                    "institutional_action": "GOVERNMENT_RESPONSE" if answer_event else "UNRESOLVED",
                    "policy_implementation": "NOT_ASSESSED",
                },
                "coverage_scope": coverage_scope,
                "void": False,
            }
            objects.append(question_obj)
            identity_counts["questions"] += 1
            if submission:
                identity_counts["questions_with_submission_event"] += 1
            else:
                identity_counts["questions_without_submission_event"] += 1
            if answer_event:
                identity_counts["questions_with_answer_event"] += 1
            else:
                identity_counts["questions_without_answer_event"] += 1
        if answer_document or answer_event:
            answer_base = answer_document or answer_event["record"]
            answer_evidence_ids = list(dict.fromkeys(answer_refs))
            answer_obj = {
                "object_id": answer_object_id,
                "kind": ANSWER_KIND,
                "matter_id": matter_id,
                # An event-only answer has no KKV document identity.  Do not
                # relabel the KK procedural record as an answer document.
                "document_id": answer_document.get("document_id") if answer_document else None,
                "title": answer_document.get("title") if answer_document else f"Vastaus kirjalliseen kysymykseen {matter_id}",
                "text": answer_document.get("text") if answer_document else (answer_event.get("quote") if answer_event else ""),
                "date": answer_base.get("publication_date") or (answer_event["date"] if answer_event else None),
                "publication_date": answer_base.get("publication_date"),
                "action_date": answer_event["date"] if answer_event else None,
                "action_date_basis": "ANSWER_EVENT" if answer_event else None,
                "action_date_provenance": {"basis": answer_event["basis"], "evidence_ids": answer_event["evidence_ids"]} if answer_event else None,
                # The question signer in a procedural record is not the
                # institutional respondent.  With no KKV row, leave authors
                # empty rather than attributing the answer to that signer.
                "authors": answer_document.get("authors") if answer_document else [],
                "evidence_ids": answer_evidence_ids,
                "evidence": [evidence[key] for key in answer_evidence_ids if key in evidence],
                "source_id": QUESTION_SOURCE_ID,
                "url": (answer_base.get("evidence") or [{}])[0].get("source_url"),
                "question_object_id": f"eduskunta:{matter_id}",
                "source_records": [_record_summary(record) for record in answers] + ([ _record_summary(answer_event["record"]) ] if answer_event and answer_event["record"] not in answers else []),
                "disposition": {
                    "state": "ANSWERED" if answer_event else "UNRESOLVED",
                    "date": answer_event["date"] if answer_event else None,
                    "evidence_ids": answer_event["evidence_ids"] if answer_event else [],
                    "institutional_action": "GOVERNMENT_RESPONSE" if answer_event else "UNRESOLVED",
                    "policy_implementation": "NOT_ASSESSED",
                },
                "coverage_scope": coverage_scope,
                "void": False,
            }
            objects.append(answer_obj)
            identity_counts["answers"] += 1
    return {"objects": objects, "evidence": list(evidence.values()), "identity_counts": identity_counts}


def _page_query(prefix: str, year: int, digit: str, page: int) -> dict[str, Any]:
    return {"columnName": "Eduskuntatunnus", "columnValue": f"{prefix} {digit}%/{year} vp", "perPage": 100, "page": page}


def _load_page(path: Path, receipt: Path, *, getter: Any, params: Mapping[str, Any], refresh: bool, retrieved_at: str) -> tuple[dict[str, Any], dict[str, Any]]:
    if path.exists() and receipt.exists() and not refresh:
        body = path.read_bytes()
        manifest = json.loads(receipt.read_text(encoding="utf-8"))
        if hashlib.sha256(body).hexdigest() != manifest.get("sha256"):
            raise ValueError(f"corrupt question checkpoint: {path}")
    else:
        response = getter(VASKI_ROWS_URL, params=dict(params), headers={"User-Agent": USER_AGENT}, timeout=90)
        response.raise_for_status()
        body = response.content
        payload = json.loads(body)
        if "columnNames" not in payload or "rowData" not in payload:
            raise ValueError("VaskiData question API returned no table payload")
        manifest = {
            "url": str(getattr(response, "url", VASKI_ROWS_URL)),
            "query": json.dumps(dict(params), sort_keys=True),
            "sha256": hashlib.sha256(body).hexdigest(),
            "bytes": len(body),
            "retrieved_at": retrieved_at,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
        receipt.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    payload = json.loads(body)
    return payload, manifest


def acquire_questions(
    years: Iterable[int] | None = None,
    *,
    raw_dir: Path | None = None,
    client: Any = None,
    max_pages: int = 200,
    refresh: bool = False,
    include_answers: bool = True,
    partition_identifiers: bool = True,
) -> dict[str, Any]:
    """Fetch KK/KKV rows for the declared current-term years.

    Page overlap is retained as a coverage limitation.  It does not make the
    parser silently discard rows, and it does not become an ``ENUMERATED``
    certificate unless every requested partition is disjoint and parsed.
    """

    selected_years = sorted({int(year) for year in (years or DEFAULT_YEARS)})
    if not selected_years or any(year < 2023 or year > 2026 for year in selected_years):
        raise ValueError("question acquisition is limited to current-term years 2023–2026")
    destination = raw_dir or RAW / "eduskunta" / "question_register"
    destination.mkdir(parents=True, exist_ok=True)
    getter = client.get if client is not None else httpx.get
    retrieved_at = datetime.now(UTC).replace(microsecond=0).isoformat()
    all_records: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    seen_row_ids: dict[str, str] = {}
    overlap_rows: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    prefixes = ["KK", "KKV"] if include_answers else ["KK"]
    for year in selected_years:
        for prefix in prefixes:
            digits = list("0123456789") if partition_identifiers else [""]
            for digit in digits:
                partition_seen: set[str] = set()
                for page in range(max_pages):
                    params = _page_query(prefix, year, digit, page)
                    suffix = f"-prefix{digit}" if digit else ""
                    path = destination / f"{prefix.lower()}-{year}{suffix}-{page}.json"
                    receipt = path.with_suffix(".manifest.json")
                    payload, manifest = _load_page(path, receipt, getter=getter, params=params, refresh=refresh, retrieved_at=retrieved_at)
                    rows = [dict(zip(payload["columnNames"], raw)) for raw in payload.get("rowData", [])]
                    manifest = dict(manifest)
                    manifest.update({"page": page, "prefix": prefix, "year": year, "digit": digit, "row_count": len(rows)})
                    page_ids: list[str] = []
                    new_count = 0
                    page_overlap = 0
                    for row in rows:
                        record_id = str(row.get("Id") or "").strip()
                        matter = _clean(row.get("Eduskuntatunnus"))
                        if not record_id:
                            excluded.append({"matter_id": matter, "reason": "row lacks Id"})
                            continue
                        match = _MATTER.match(matter)
                        expected_prefix = prefix.upper()
                        if not match or match.group("prefix").upper() != expected_prefix or int(match.group("year")) != year:
                            raise ValueError(f"server ignored question identifier filter: {matter}")
                        page_ids.append(record_id)
                        xml = row.get("XmlData") or row.get("XmlDataFi") or ""
                        body_hash = hashlib.sha256(str(xml).encode("utf-8")).hexdigest()
                        if record_id in partition_seen or record_id in seen_row_ids:
                            page_overlap += 1
                            overlap_rows.append({"id": record_id, "matter_id": matter, "page": page, "prefix": prefix, "year": year})
                            if seen_row_ids.get(record_id) != body_hash:
                                conflicts.append({"id": record_id, "matter_id": matter, "old_sha256": seen_row_ids.get(record_id), "new_sha256": body_hash})
                            continue
                        partition_seen.add(record_id)
                        seen_row_ids[record_id] = body_hash
                        new_count += 1
                        try:
                            record = parse_question_row(row, source_url=_source_url(matter), retrieved_at=retrieved_at)
                        except ValueError as error:
                            excluded.append({"id": record_id, "matter_id": matter, "reason": str(error)})
                            continue
                        all_records.append(record)
                        archive = destination / "by-sha256" / (body_hash + ".xml")
                        archive.parent.mkdir(parents=True, exist_ok=True)
                        if archive.exists() and hashlib.sha256(archive.read_bytes()).hexdigest() != body_hash:
                            raise ValueError(f"corrupt immutable question source: {archive}")
                        if not archive.exists():
                            archive.write_bytes(str(xml).encode("utf-8"))
                    manifest.update({"row_ids": page_ids, "new_row_count": new_count, "overlap_row_count": page_overlap,
                                    "has_more": bool(payload.get("hasMore")), "artifact_paths": [str(destination / "by-sha256" / (seen_row_ids[row_id] + ".xml")) for row_id in page_ids if row_id in seen_row_ids]})
                    manifests.append(manifest)
                    if not payload.get("hasMore"):
                        break
                else:
                    raise RuntimeError(f"question register pagination exceeded {max_pages} pages for {prefix} {year} digit {digit}")
    normalized = normalize_question_records(all_records, coverage_scope=f"term_{min(selected_years)}_{max(selected_years)}")
    manifest_digest = hashlib.sha256("".join(str(item["sha256"]) for item in manifests).encode()).hexdigest()
    state = "ENUMERATED"
    if overlap_rows:
        state = "ENUMERATED_WITH_PAGE_OVERLAP"
    if excluded or conflicts:
        state = "ENUMERATED_WITH_EXCLUSIONS"
    coverage = {
        "schema_version": "1.0",
        "coverage_id": "written-question-register-" + manifest_digest[:20],
        "source_id": QUESTION_SOURCE_ID,
        "kind": "WRITTEN_QUESTION_REGISTER",
        "url": VASKI_ROWS_URL,
        "years": selected_years,
        "enumeration": "VASKI_IDENTIFIER_FIRST_DIGIT_PARTITIONS",
        "query_partition": "DOCUMENT_PREFIX_AND_FIRST_IDENTIFIER_DIGIT" if partition_identifiers else "DOCUMENT_PREFIX_AND_YEAR",
        "state": state,
        "complete": state == "ENUMERATED",
        "retrieved_at": retrieved_at,
        "source_record_count": len(seen_row_ids),
        "parsed_record_count": len(all_records),
        "object_count": len(normalized["objects"]),
        "question_object_count": normalized["identity_counts"]["questions"],
        "answer_object_count": normalized["identity_counts"]["answers"],
        "excluded_count": len(excluded),
        "overlap_row_count": len(overlap_rows),
        "conflict_count": len(conflicts),
        "excluded_records": excluded,
        # Keep every observed duplicate row in the receipt.  A count alone
        # would make it impossible to audit which raw record was repeated.
        "page_overlaps": overlap_rows,
        "conflicts": conflicts,
        "identity_counts": normalized["identity_counts"],
        "page_manifests": manifests,
        "window": {"earliest": f"{min(selected_years)}-01-01", "latest": min(f"{max(selected_years)}-12-31", CORPUS_CUTOFF)},
        "limitations": [
            "Coverage is limited to the accessible Finnish KK/KKV VaskiData rows in the declared years.",
            "The API exposes no stable ordering parameter; page overlap is reported and prevents a complete certificate.",
            "Question and government-answer documents are separate official objects.",
            "An ANSWERED disposition records an institutional response only; it is not evidence of policy implementation or outcome.",
            "Answer PDFs may be referenced by Vaski metadata while the answer body is not present in the JSON row.",
        ],
    }
    result = {
        "objects": normalized["objects"],
        "evidence": normalized["evidence"],
        "coverage": coverage,
        "manifests": manifests,
        "retrieved_at": retrieved_at,
    }
    (destination / "normalized.jsonl").write_text("".join(json.dumps(obj, ensure_ascii=False) + "\n" for obj in result["objects"]), encoding="utf-8")
    (destination / "coverage.json").write_text(json.dumps(coverage, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def import_result(conn: sqlite3.Connection, result: Mapping[str, Any]) -> dict[str, Any]:
    """Upsert question objects/evidence/coverage without rebuilding other data."""

    objects = list(result.get("objects") or [])
    evidence = list(result.get("evidence") or [])
    coverage = dict(result.get("coverage") or {})
    if not coverage.get("coverage_id"):
        raise ValueError("question result lacks coverage_id")
    evidence_by_id = {str(item.get("evidence_id")): item for item in evidence if item.get("evidence_id")}
    for obj in objects:
        conn.execute("INSERT OR REPLACE INTO official_objects(object_id, json) VALUES (?, ?)", (obj["object_id"], json.dumps(obj, ensure_ascii=False)))
        for item in obj.get("evidence") or []:
            evidence_id = item.get("evidence_id")
            if evidence_id and evidence_id not in evidence_by_id:
                evidence_by_id[evidence_id] = item
    evidence = list(evidence_by_id.values())
    for item in evidence:
        conn.execute("INSERT OR REPLACE INTO evidence(evidence_id, json) VALUES (?, ?)", (item["evidence_id"], json.dumps(item, ensure_ascii=False)))
    conn.execute("INSERT OR REPLACE INTO source_coverage(coverage_id, json) VALUES (?, ?)", (coverage["coverage_id"], json.dumps(coverage, ensure_ascii=False)))
    for manifest in result.get("manifests") or []:
        conn.execute(
            "INSERT INTO manifest(source_id, url, sha256, bytes, http_status, retrieved_at, note) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (coverage["source_id"], manifest.get("url"), manifest.get("sha256"), manifest.get("bytes"), 200, manifest.get("retrieved_at"), "VaskiData written-question page receipt"),
        )
    return {"official_objects": len(objects), "evidence": len(evidence), "source_coverage": 1}


# Keep the adapter discoverable under the same neutral name used by the
# initiative register without making the two source universes interchangeable.
acquire_registry = acquire_questions


__all__ = [
    "ANSWER_KIND",
    "QUESTION_KIND",
    "QUESTION_SOURCE_ID",
    "acquire_questions",
    "acquire_registry",
    "import_result",
    "normalize_question_records",
    "parse_question_row",
    "parse_question_xml",
]

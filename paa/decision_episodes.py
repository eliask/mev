"""Source-grounded decision episodes.

This module is a deliberately small waist between the official-object
adapters and any later attribution, semantic, or browser code.  It does not
decide whether a policy was implemented and it does not infer a legal effect
from a parliamentary question.  Instead it records the observable sequence
of source objects and leaves the dimensions that are not covered by those
objects as explicit residuals.

The first episode adapter is for a written question and a government answer.
Both are already canonical ``official_objects`` produced by
``paa.question_ledger``.  The builder is pure; the database helper only reads
the three source tables needed to assemble an episode.
"""


import hashlib
import json
import re
import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

QUESTION_KIND = "WRITTEN_QUESTION"
ANSWER_KIND = "GOVERNMENT_ANSWER"
EPISODE_KIND = "WRITTEN_QUESTION_RESPONSE"
EPISODE_SCHEMA_VERSION = "1.0"

_DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
_RECORD_CLASSES = {"CONTENT", "PROCEDURAL"}
_EVENT_FIELDS = {
    "submission": ("QUESTION_SUBMITTED", "FILED_QUESTION"),
    "answer_received": ("GOVERNMENT_RESPONSE_RECEIVED", "RESPONSE_RECEIVED"),
    "answer_announced": ("GOVERNMENT_RESPONSE_ANNOUNCED", "RESPONSE_ANNOUNCED"),
}


class EpisodeError(ValueError):
    """Raised when source objects cannot form one same-matter episode."""


def _text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _unique(values: Iterable[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value is None:
            continue
        item = str(value)
        if item and item not in seen:
            result.append(item)
            seen.add(item)
    return result


def _unique_nested(values: Iterable[Iterable[Any]]) -> list[str]:
    """Flatten evidence/locator lists before applying stable de-duplication."""

    return _unique(item for group in values for item in group)


def _date_value(value: Any) -> str | None:
    if value is None:
        return None
    match = _DATE_RE.search(str(value))
    return match.group(1) if match else None


def _safe_slug(value: str) -> str:
    value = value.casefold().replace("/", "-")
    value = re.sub(r"[^a-z0-9]+", "-", value).strip("-")
    return value[:100] or "matter"


def _object_evidence_ids(obj: Mapping[str, Any]) -> list[str]:
    values: list[Any] = list(obj.get("evidence_ids") or [])
    disposition = obj.get("disposition")
    if isinstance(disposition, Mapping):
        values.extend(disposition.get("evidence_ids") or [])
    provenance = obj.get("action_date_provenance")
    if isinstance(provenance, Mapping):
        values.extend(provenance.get("evidence_ids") or [])
    for author in obj.get("authors") or []:
        if isinstance(author, Mapping):
            values.extend(author.get("evidence_ids") or [])
    for record in obj.get("source_records") or []:
        if isinstance(record, Mapping):
            values.extend(record.get("evidence_ids") or [])
    return _unique(values)


def _embedded_evidence(objects: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for obj in objects:
        for item in obj.get("evidence") or []:
            if not isinstance(item, Mapping) or not item.get("evidence_id"):
                continue
            result[str(item["evidence_id"])] = dict(item)
    return result


def _normalise_evidence(
    objects: Sequence[Mapping[str, Any]],
    evidence: Mapping[str, Mapping[str, Any]] | Iterable[Mapping[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    result = _embedded_evidence(objects)
    if evidence is None:
        return result
    if isinstance(evidence, Mapping):
        items = evidence.values()
    else:
        items = evidence
    for item in items:
        if isinstance(item, Mapping) and item.get("evidence_id"):
            result[str(item["evidence_id"])] = dict(item)
    return result


def _source_record_key(record: Mapping[str, Any]) -> tuple[str, str]:
    return (str(record.get("record_locator") or ""), str(record.get("raw_sha256") or ""))


def _source_records(objects: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Merge duplicate source rows while retaining every owning object."""

    merged: dict[tuple[str, str], dict[str, Any]] = {}
    fallback_counter = 0
    for obj in objects:
        object_id = str(obj.get("object_id") or "")
        for raw_record in obj.get("source_records") or []:
            if not isinstance(raw_record, Mapping):
                continue
            record = dict(raw_record)
            key = _source_record_key(record)
            if not key[0] and not key[1]:
                fallback_counter += 1
                key = (f"object:{object_id}:{fallback_counter}", "")
            current = merged.get(key)
            if current is None:
                current = {
                    "record_locator": record.get("record_locator"),
                    "raw_sha256": record.get("raw_sha256"),
                    "document_id": record.get("document_id"),
                    "record_class": record.get("record_class"),
                    "publication_date": record.get("publication_date"),
                    "object_ids": [],
                    "evidence_ids": [],
                    "event_count": record.get("event_count", 0),
                }
                merged[key] = current
            if object_id and object_id not in current["object_ids"]:
                current["object_ids"].append(object_id)
            current["evidence_ids"] = _unique([
                *current["evidence_ids"],
                *(record.get("evidence_ids") or []),
            ])
            if current.get("event_count", 0) < record.get("event_count", 0):
                current["event_count"] = record.get("event_count", 0)
            # A substantive content row is more informative than a shell
            # when two adapters describe the same locator.
            if current.get("record_class") not in _RECORD_CLASSES and record.get("record_class") in _RECORD_CLASSES:
                current["record_class"] = record.get("record_class")
    return sorted(
        merged.values(),
        key=lambda item: (
            item.get("publication_date") or "9999-99-99",
            item.get("record_locator") or "",
            item.get("raw_sha256") or "",
        ),
    )


def _coverage_summary(
    coverage: Mapping[str, Any] | Iterable[Mapping[str, Any]] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    if coverage is None:
        return [], []
    items = [coverage] if isinstance(coverage, Mapping) else list(coverage)
    summaries: list[dict[str, Any]] = []
    residuals: list[dict[str, str]] = []
    for item in items:
        if not isinstance(item, Mapping):
            continue
        summary = {
            "coverage_id": item.get("coverage_id"),
            "source_id": item.get("source_id"),
            "kind": item.get("kind"),
            "state": item.get("state"),
            "complete": item.get("complete"),
            "source_record_count": item.get("source_record_count"),
            "parsed_record_count": item.get("parsed_record_count"),
            "object_count": item.get("object_count"),
            "years": item.get("years"),
            "limitations": list(item.get("limitations") or item.get("scope_residuals") or []),
        }
        summaries.append(summary)
        if item.get("complete") is False or item.get("state") not in {None, "ENUMERATED", "RECONCILED_FOR_DECLARED_SLICE"}:
            residuals.append({
                "code": "SOURCE_COVERAGE_LIMITED",
                "severity": "CONTEXT",
                "reason": f"Coverage {item.get('coverage_id') or '<unnamed>'} is not a complete enumeration.",
            })
    return summaries, residuals


def _actor_records(
    objects: Sequence[Mapping[str, Any]],
    actor_map: Mapping[str, Mapping[str, Any]] | None,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for obj in objects:
        object_id = str(obj.get("object_id") or "")
        for source in obj.get("authors") or []:
            if not isinstance(source, Mapping):
                continue
            person_id = str(source.get("person_id")) if source.get("person_id") is not None else None
            name = _text(source.get("name"))
            role = str(source.get("role") or "UNRESOLVED")
            key = (object_id, person_id or "", name, role)
            if key in seen:
                continue
            seen.add(key)
            actor = {
                "source_object_id": object_id,
                "person_id": person_id,
                "name": name,
                "role": role,
                "identity_basis": source.get("identity_basis") or ("SOURCE_PERSON_ID" if person_id else "SOURCE_NAME_ONLY"),
                "evidence_ids": _unique(source.get("evidence_ids") or []),
            }
            # Attribution is intentionally an input, never an inference.  A
            # future identity/attribution pass can supply actor_id and its
            # own basis without changing the source actor record.
            if actor_map:
                mapped = actor_map.get(person_id) if person_id else None
                if mapped is None and name:
                    mapped = actor_map.get(name)
                if isinstance(mapped, Mapping) and mapped.get("actor_id"):
                    actor["actor_id"] = str(mapped["actor_id"])
                    actor["actor_identity_basis"] = mapped.get("identity_basis") or "EXTERNAL_ATTRIBUTION"
                    actor["actor_evidence_ids"] = _unique(mapped.get("evidence_ids") or [])
            result.append(actor)
    return result


def _event_candidates(
    objects: Sequence[Mapping[str, Any]],
    evidence: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Extract only explicitly named procedure events from evidence fields."""

    candidates: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for obj in objects:
        object_id = str(obj.get("object_id") or "")
        ids = _object_evidence_ids(obj)
        record_locators = {
            str(record.get("record_locator"))
            for record in obj.get("source_records") or []
            if isinstance(record, Mapping) and record.get("record_locator")
        }
        # ``normalize_question_records`` historically kept the procedure
        # record ID in the object evidence list while the finer-grained event
        # evidence lived in the evidence table.  A source-record locator is
        # the safe join key here; it never searches by text or matter title.
        candidate_ids = _unique([
            *ids,
            *(evidence_id for evidence_id, item in evidence.items()
              if str(item.get("record_locator") or "") in record_locators),
        ])
        # Some callers pass parsed records before materializing embedded
        # evidence; accept explicit event dictionaries as well.
        for event in obj.get("events") or []:
            if not isinstance(event, Mapping):
                continue
            kind = str(event.get("kind") or "").upper()
            mapping = {
                "SUBMISSION": ("QUESTION_SUBMITTED", "FILED_QUESTION"),
                "ANSWER_RECEIVED": ("GOVERNMENT_RESPONSE_RECEIVED", "RESPONSE_RECEIVED"),
                "ANSWER_ANNOUNCED": ("GOVERNMENT_RESPONSE_ANNOUNCED", "RESPONSE_ANNOUNCED"),
            }.get(kind)
            if not mapping:
                continue
            for evidence_id in _unique(event.get("evidence_ids") or ids):
                key = (mapping[0], evidence_id)
                if key in seen:
                    continue
                seen.add(key)
                candidates.append({
                    "stage": mapping[0],
                    "action": mapping[1],
                    "object_id": (
                        next((str(candidate.get("object_id")) for candidate in objects
                              if candidate.get("kind") == ANSWER_KIND), object_id)
                        if mapping[0].startswith("GOVERNMENT_RESPONSE") else object_id
                    ),
                    "date": _date_value(event.get("date")),
                    "date_basis": event.get("basis") or "SOURCE_EVENT",
                    "quote": event.get("quote"),
                    "evidence_ids": [evidence_id],
                    "record_locators": [],
                    "sort_priority": 20 if kind == "SUBMISSION" else 40 if kind == "ANSWER_RECEIVED" else 50,
                })
        for evidence_id in candidate_ids:
            item = evidence.get(evidence_id)
            if not isinstance(item, Mapping):
                continue
            field = str(item.get("field_path") or "")
            if not field.startswith("procedure."):
                continue
            procedure = field.rsplit(".", 1)[-1].casefold()
            mapping = _EVENT_FIELDS.get(procedure)
            if mapping is None:
                continue
            key = (mapping[0], evidence_id)
            if key in seen:
                continue
            seen.add(key)
            quote = item.get("quote")
            candidates.append({
                "stage": mapping[0],
                "action": mapping[1],
                "object_id": (
                    next((str(candidate.get("object_id")) for candidate in objects
                          if candidate.get("kind") == ANSWER_KIND), object_id)
                    if mapping[0].startswith("GOVERNMENT_RESPONSE") else object_id
                ),
                "date": _date_value(quote) or _date_value(item.get("date")),
                "date_basis": "SOURCE_EVENT",
                "quote": quote,
                "evidence_ids": [evidence_id],
                "record_locators": _unique([item.get("record_locator")]),
                "sort_priority": 20 if procedure == "submission" else 40 if procedure == "answer_received" else 50,
            })
    return candidates


def _object_summary(obj: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "object_id": obj.get("object_id"),
        "kind": obj.get("kind"),
        "matter_id": obj.get("matter_id"),
        "document_id": obj.get("document_id"),
        "title": obj.get("title"),
        "text": obj.get("text"),
        "date": obj.get("date"),
        "publication_date": obj.get("publication_date"),
        "action_date": obj.get("action_date"),
        "action_date_basis": obj.get("action_date_basis"),
        "action_date_provenance": obj.get("action_date_provenance"),
        "answer_object_id": obj.get("answer_object_id"),
        "question_object_id": obj.get("question_object_id"),
        "authors": [dict(item) for item in obj.get("authors") or [] if isinstance(item, Mapping)],
        "evidence_ids": _object_evidence_ids(obj),
        "source_id": obj.get("source_id"),
        "url": obj.get("url"),
        "source_records": [dict(item) for item in obj.get("source_records") or [] if isinstance(item, Mapping)],
        "disposition": dict(obj.get("disposition") or {}),
    }


def _record_locators_for_evidence(
    evidence_ids: Iterable[str],
    evidence: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    return _unique(evidence.get(item, {}).get("record_locator") for item in evidence_ids)


def _episode_id(matter_id: str, objects: Sequence[Mapping[str, Any]], records: Sequence[Mapping[str, Any]]) -> str:
    raw = "\x1f".join([
        EPISODE_KIND,
        matter_id,
        *(str(obj.get("object_id") or "") for obj in objects),
        *(str(record.get("raw_sha256") or "") for record in records),
    ])
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"episode-{_safe_slug(matter_id)}-{digest}"


def _unknown(code: str, reason: str, *, severity: str = "BLOCKING", evidence_ids: Iterable[str] = ()) -> dict[str, Any]:
    return {
        "code": code,
        "severity": severity,
        "reason": reason,
        "evidence_ids": _unique(evidence_ids),
    }


def build_written_question_episode(
    question: Mapping[str, Any],
    answer: Mapping[str, Any] | None = None,
    *,
    evidence: Mapping[str, Mapping[str, Any]] | Iterable[Mapping[str, Any]] | None = None,
    coverage: Mapping[str, Any] | Iterable[Mapping[str, Any]] | None = None,
    actor_map: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build one immutable, source-sequenced written-question episode.

    ``question`` and ``answer`` must be canonical official objects.  The
    function never searches for a related matter and never fills a missing
    answer from a different object.  That makes a missing answer an explicit
    residual rather than a silent join.
    """

    if str(question.get("kind")) != QUESTION_KIND:
        raise EpisodeError(f"expected {QUESTION_KIND}, got {question.get('kind')!r}")
    matter_id = str(question.get("matter_id") or "").strip()
    if not matter_id:
        raise EpisodeError("written question has no matter_id")
    if answer is not None:
        if str(answer.get("kind")) != ANSWER_KIND:
            raise EpisodeError(f"expected {ANSWER_KIND}, got {answer.get('kind')!r}")
        if str(answer.get("matter_id") or "") != matter_id:
            raise EpisodeError(
                f"answer matter mismatch: {answer.get('matter_id')!r} != {matter_id!r}"
            )
        linked_question = answer.get("question_object_id")
        if linked_question and str(linked_question) != str(question.get("object_id")):
            raise EpisodeError("answer question_object_id does not point to supplied question")

    objects = [question] + ([answer] if answer is not None else [])
    source_evidence = _normalise_evidence(objects, evidence)
    records = _source_records(objects)
    coverage_rows, coverage_residuals = _coverage_summary(coverage)
    question_eids = _object_evidence_ids(question)
    answer_eids = _object_evidence_ids(answer) if answer is not None else []
    all_eids = _unique([*question_eids, *answer_eids])
    event_rows = _event_candidates(objects, source_evidence)

    # An answer event is admitted only when the source event is explicit.  A
    # date copied from publication metadata cannot close an episode.
    answer_event_rows = [
        row for row in event_rows
        if row["stage"] in {"GOVERNMENT_RESPONSE_RECEIVED", "GOVERNMENT_RESPONSE_ANNOUNCED"}
    ]
    has_response_event = bool(answer_event_rows)
    if answer is not None and not has_response_event:
        answer_action_date = _date_value(answer.get("action_date"))
        if answer_action_date and answer.get("action_date_basis") == "ANSWER_EVENT":
            # The canonical answer object carries a source-backed action-date
            # provenance even when callers did not pass the evidence table.
            provenance = answer.get("action_date_provenance") or {}
            refs = _unique(provenance.get("evidence_ids") or answer.get("evidence_ids") or [])
            event_rows.append({
                "stage": "GOVERNMENT_RESPONSE_RECEIVED",
                "action": "RESPONSE_RECEIVED",
                "object_id": str(answer.get("object_id") or ""),
                "date": answer_action_date,
                "date_basis": "ANSWER_EVENT",
                "quote": None,
                "evidence_ids": refs,
                "record_locators": _record_locators_for_evidence(refs, source_evidence),
                "sort_priority": 40,
            })
            has_response_event = True

    if has_response_event:
        episode_state = "ANSWERED_INSTITUTIONALLY"
    elif answer is not None:
        episode_state = "ANSWER_OBJECT_PRESENT_UNRESOLVED"
    else:
        episode_state = "OPEN_QUESTION"

    # Add document stages separately from procedural events.  These are
    # observations about the registered source rows, not claims about legal
    # effect.
    sequence_rows: list[dict[str, Any]] = []
    question_records = [record for record in records if str(question.get("object_id")) in record.get("object_ids", [])]
    answer_records = [
        record for record in records
        if answer is not None and str(answer.get("object_id")) in record.get("object_ids", [])
    ]
    q_document_records = [record for record in question_records if record.get("record_class") == "CONTENT"]
    if not q_document_records:
        q_document_records = question_records
    if q_document_records:
        q_record_eids = _unique(eid for record in q_document_records for eid in record.get("evidence_ids") or [])
        sequence_rows.append({
            "stage": "QUESTION_DOCUMENT",
            "action": "QUESTION_DOCUMENT_PUBLISHED",
            "object_id": question.get("object_id"),
            "date": _date_value(question.get("publication_date") or question.get("date")),
            "date_basis": "PUBLICATION_DATE" if question.get("publication_date") or question.get("date") else "UNRESOLVED",
            "actor_refs": [str(question.get("object_id"))],
            "record_locators": _unique(record.get("record_locator") for record in q_document_records),
            "evidence_ids": q_record_eids,
            "sort_priority": 10,
        })
    if answer is not None:
        answer_document_id = answer.get("document_id")
        answer_specific_records = [
            record for record in answer_records
            if answer_document_id and record.get("document_id") == answer_document_id
        ]
        answer_document_records = [record for record in answer_specific_records if record.get("record_class") == "CONTENT"]
        if not answer_document_records and answer_specific_records:
            answer_document_records = answer_specific_records
        # KKV rows in the current public register can be procedural shells;
        # retaining that shell as a source stage is still useful, but it is
        # named as a registered answer object rather than an answer body.
        if not answer_document_records:
            answer_document_records = answer_records
        if answer_document_records:
            answer_record_eids = _unique(eid for record in answer_document_records for eid in record.get("evidence_ids") or [])
            sequence_rows.append({
                "stage": "GOVERNMENT_RESPONSE_DOCUMENT",
                "action": "ANSWER_OBJECT_REGISTERED",
                "object_id": answer.get("object_id"),
                "date": _date_value(answer.get("publication_date") or answer.get("date")),
                "date_basis": "PUBLICATION_DATE" if answer.get("publication_date") or answer.get("date") else "UNRESOLVED",
                "actor_refs": [str(answer.get("object_id"))],
                "record_locators": _unique(record.get("record_locator") for record in answer_document_records),
                "evidence_ids": answer_record_eids,
                "content_state": (
                    "BODY_PRESENT"
                    if any(record.get("record_class") == "CONTENT" for record in answer_document_records)
                    else "REGISTERED_RECORD_WITHOUT_BODY"
                ),
                "sort_priority": 30,
            })

    for event in event_rows:
        row = dict(event)
        row["actor_refs"] = [
            str(question.get("object_id"))
            if event["stage"] == "QUESTION_SUBMITTED"
            else str(answer.get("object_id")) if answer is not None else ""
        ]
        row["actor_refs"] = [ref for ref in row["actor_refs"] if ref]
        if not row.get("record_locators"):
            row["record_locators"] = _record_locators_for_evidence(row.get("evidence_ids") or [], source_evidence)
        sequence_rows.append(row)

    sequence_rows.sort(key=lambda row: (
        row.get("date") or "9999-99-99",
        int(row.get("sort_priority") or 99),
        row.get("stage") or "",
        ",".join(row.get("evidence_ids") or []),
    ))
    for number, row in enumerate(sequence_rows, start=1):
        row["sequence"] = number
        row.pop("sort_priority", None)
        if not row.get("date"):
            row["date_state"] = "UNRESOLVED"

    actors = _actor_records(objects, actor_map)
    actor_by_object: dict[str, list[dict[str, Any]]] = {}
    for actor in actors:
        actor_by_object.setdefault(str(actor["source_object_id"]), []).append(actor)
    for row in sequence_rows:
        # Replace object-id placeholders with source-backed actor references;
        # an institutional response intentionally has no implied individual
        # actor when no source respondent is identified.
        if row["stage"] == "QUESTION_SUBMITTED":
            row["actor_refs"] = [dict(item) for item in actor_by_object.get(str(question.get("object_id")), [])]
        elif row["stage"] in {"GOVERNMENT_RESPONSE_RECEIVED", "GOVERNMENT_RESPONSE_ANNOUNCED"}:
            respondent = [
                dict(item) for item in actor_by_object.get(str(answer.get("object_id")), [])
                if item.get("role") == "RESPONDENT"
            ] if answer is not None else []
            row["actor_refs"] = respondent
            row["institution"] = "GOVERNMENT"
        elif row["stage"] == "GOVERNMENT_RESPONSE_DOCUMENT":
            row["actor_refs"] = [
                dict(item) for item in actor_by_object.get(str(answer.get("object_id")), [])
                if item.get("role") == "RESPONDENT"
            ] if answer is not None else []
            row["institution"] = "GOVERNMENT"
        else:
            row["actor_refs"] = [dict(item) for item in actor_by_object.get(str(question.get("object_id")), [])]

    unknowns: list[dict[str, Any]] = [
        _unknown(
            "POLICY_IMPLEMENTATION_NOT_SOURCED",
            "A written-question answer records an institutional response; this source surface does not establish policy implementation.",
            evidence_ids=all_eids,
        ),
        _unknown(
            "CAUSAL_OUTCOME_NOT_SOURCED",
            "No outcome or counterfactual source is part of this written-question episode.",
            evidence_ids=all_eids,
        ),
        _unknown(
            "LEGAL_EFFECT_NOT_ASSESSED",
            "A question and its response are not an enactment, commencement, or other legal-effect record.",
            evidence_ids=all_eids,
        ),
    ]
    if answer is None:
        unknowns.append(_unknown(
            "ANSWER_OBJECT_NOT_LOADED",
            "The question points to no loaded answer object; no cross-matter fallback was attempted.",
            evidence_ids=question.get("evidence_ids") or [],
        ))
    elif not has_response_event:
        unknowns.append(_unknown(
            "ANSWER_EVENT_NOT_FOUND",
            "An answer object is present, but no explicit answer event was supplied by the source records.",
            evidence_ids=answer_eids,
        ))
    respondents = [actor for actor in actors if actor.get("role") == "RESPONDENT"]
    if answer is not None and not respondents:
        unknowns.append(_unknown(
            "RESPONDENT_IDENTITY_UNRESOLVED",
            "The answer source has no source-backed respondent identity.",
            severity="CONTEXT",
            evidence_ids=answer_eids,
        ))
    if not coverage_rows:
        unknowns.append(_unknown(
            "SOURCE_COVERAGE_NOT_ATTACHED",
            "No register coverage receipt was attached to this episode.",
            severity="CONTEXT",
        ))
    unknowns.extend(coverage_residuals)
    if not sequence_rows:
        unknowns.append(_unknown(
            "SOURCE_SEQUENCE_EMPTY",
            "Neither a source record nor an explicit procedure event was available.",
        ))

    source_objects = [_object_summary(obj) for obj in objects]
    all_eids = _unique([
        *all_eids,
        *(eid for row in sequence_rows for eid in row.get("evidence_ids") or []),
        *(eid for actor in actors for eid in actor.get("evidence_ids") or []),
    ])
    episode = {
        "schema_version": EPISODE_SCHEMA_VERSION,
        "episode_id": _episode_id(matter_id, objects, records),
        "kind": EPISODE_KIND,
        "matter_id": matter_id,
        "episode_state": episode_state,
        "source_objects": source_objects,
        "source_records": records,
        "source_sequence": sequence_rows,
        "actors": actors,
        "action_channels": [
            {
                "channel": "PARLIAMENTARY_WRITTEN_QUESTION",
                "action": "FILED_QUESTION",
                "object_id": question.get("object_id"),
                "actor_roles": ["AUTHOR", "COSIGNER"],
                "evidence_ids": _unique_nested(
                    row.get("evidence_ids") or [] for row in sequence_rows if row.get("stage") == "QUESTION_SUBMITTED"
                ),
            },
            {
                "channel": "GOVERNMENT_RESPONSE",
                "action": "ISSUED_INSTITUTIONAL_RESPONSE",
                "object_id": answer.get("object_id") if answer is not None else None,
                "institution": "GOVERNMENT",
                "actor_roles": ["RESPONDENT"],
                "evidence_ids": _unique_nested(
                    row.get("evidence_ids") or []
                    for row in sequence_rows
                    if row.get("stage") in {"GOVERNMENT_RESPONSE_RECEIVED", "GOVERNMENT_RESPONSE_ANNOUNCED", "GOVERNMENT_RESPONSE_DOCUMENT"}
                ),
                "state": "OBSERVED" if has_response_event else "UNRESOLVED",
            },
        ],
        "institutional_disposition": {
            "state": "ANSWERED" if has_response_event else "UNRESOLVED",
            "date": min((row["date"] for row in answer_event_rows if row.get("date")), default=None),
            "institution": "GOVERNMENT" if has_response_event else None,
            "evidence_ids": _unique_nested(row.get("evidence_ids") or [] for row in answer_event_rows),
            "basis": "EXPLICIT_ANSWER_EVENT" if has_response_event else "NO_EXPLICIT_ANSWER_EVENT",
        },
        "legal_state": {
            "state": "NOT_ASSESSED",
            "source_scope": "WRITTEN_QUESTION_REGISTER",
            "note": "No legal enactment, commencement, or operative-state conclusion is made from this episode.",
            "evidence_ids": all_eids,
        },
        "policy_implementation_state": {
            "state": "NOT_ASSESSED",
            "note": "Institutional response is not implementation evidence.",
            "evidence_ids": all_eids,
        },
        "causal_outcome_state": {
            "state": "NOT_ASSESSED",
            "note": "No causal outcome or counterfactual is identified.",
            "evidence_ids": all_eids,
        },
        "coverage": {
            "state": "DECLARED" if coverage_rows else "NOT_ATTACHED",
            "complete": bool(coverage_rows) and all(item.get("complete") is True for item in coverage_rows),
            "items": coverage_rows,
        },
        "evidence_ids": all_eids,
        "residuals": unknowns,
        "limitations": [
            "This episode is a source sequence, not a claim that the response changed policy or law.",
            "Source-register coverage limitations remain attached to the episode and are not converted into absence claims.",
        ],
    }
    validate_episode(episode, evidence=source_evidence)
    return episode


def validate_episode(
    episode: Mapping[str, Any],
    *,
    evidence: Mapping[str, Mapping[str, Any]] | Iterable[Mapping[str, Any]] | None = None,
) -> None:
    """Validate episode invariants and evidence references.

    This intentionally does not validate a causal or legal conclusion: those
    fields must remain ``NOT_ASSESSED`` for this episode kind.
    """

    required = {"schema_version", "episode_id", "kind", "matter_id", "episode_state", "source_sequence", "residuals"}
    missing = sorted(required - set(episode))
    if missing:
        raise EpisodeError(f"episode missing required fields: {', '.join(missing)}")
    if episode.get("schema_version") != EPISODE_SCHEMA_VERSION:
        raise EpisodeError("unsupported episode schema version")
    if episode.get("kind") != EPISODE_KIND:
        raise EpisodeError("unexpected episode kind")
    source_objects = list(episode.get("source_objects") or [])
    if not source_objects or source_objects[0].get("kind") != QUESTION_KIND:
        raise EpisodeError("episode must start with a written-question object")
    matter_id = str(episode.get("matter_id") or "")
    if any(str(obj.get("matter_id") or "") != matter_id for obj in source_objects):
        raise EpisodeError("episode source objects have different matter IDs")
    answers = [obj for obj in source_objects if obj.get("kind") == ANSWER_KIND]
    if len(answers) > 1:
        raise EpisodeError("episode contains more than one answer object")
    if answers:
        linked_answer = source_objects[0].get("answer_object_id")
        if linked_answer != answers[0].get("object_id"):
            raise EpisodeError("question answer_object_id does not point to episode answer")
        linked_question = answers[0].get("question_object_id")
        if linked_question and linked_question != source_objects[0].get("object_id"):
            raise EpisodeError("answer question_object_id does not point to episode question")
    dates: list[str] = []
    sequence_numbers: list[int] = []
    for row in episode.get("source_sequence") or []:
        if row.get("sequence") is not None:
            sequence_numbers.append(int(row["sequence"]))
        row_date = _date_value(row.get("date"))
        if row_date:
            dates.append(row_date)
        if row.get("stage") in {"GOVERNMENT_RESPONSE_RECEIVED", "GOVERNMENT_RESPONSE_ANNOUNCED"} and not row.get("evidence_ids"):
            raise EpisodeError("response event has no source evidence")
    if sequence_numbers != list(range(1, len(sequence_numbers) + 1)):
        raise EpisodeError("source sequence numbers are not contiguous")
    if dates != sorted(dates):
        raise EpisodeError("source sequence is not chronological")
    legal = episode.get("legal_state") or {}
    policy = episode.get("policy_implementation_state") or {}
    causal = episode.get("causal_outcome_state") or {}
    if legal.get("state") != "NOT_ASSESSED" or policy.get("state") != "NOT_ASSESSED" or causal.get("state") != "NOT_ASSESSED":
        raise EpisodeError("episode builder cannot admit legal, implementation, or causal states")
    if evidence is not None:
        known = _normalise_evidence([], evidence)
        refs = {str(item) for item in episode.get("evidence_ids") or []}
        missing_evidence = sorted(ref for ref in refs if ref not in known)
        if missing_evidence:
            raise EpisodeError("episode has unresolved evidence references: " + ", ".join(missing_evidence[:5]))


def build_episodes_from_objects(
    objects: Iterable[Mapping[str, Any]] | Mapping[str, Mapping[str, Any]],
    *,
    evidence: Mapping[str, Mapping[str, Any]] | Iterable[Mapping[str, Any]] | None = None,
    coverage: Mapping[str, Any] | Iterable[Mapping[str, Any]] | None = None,
    actor_map: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Build one episode for each loaded written question.

    Mapping values are accepted because that is the shape used by the live
    ``official_objects`` reader.  No question is joined to an answer merely
    by title or text: only the canonical ``answer_object_id`` link is used.
    """

    object_list = list(objects.values()) if isinstance(objects, Mapping) else list(objects)
    by_id = {str(obj.get("object_id")): obj for obj in object_list if obj.get("object_id")}
    result: list[dict[str, Any]] = []
    for question in sorted(
        (obj for obj in object_list if obj.get("kind") == QUESTION_KIND),
        key=lambda obj: (str(obj.get("matter_id") or ""), str(obj.get("object_id") or "")),
    ):
        answer_id = question.get("answer_object_id")
        answer = by_id.get(str(answer_id)) if answer_id else None
        result.append(build_written_question_episode(
            question,
            answer,
            evidence=evidence,
            coverage=coverage,
            actor_map=actor_map,
        ))
    return result


def build_written_question_episode_from_db(
    conn: sqlite3.Connection,
    matter_id: str,
    *,
    actor_map: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Read one episode from ``official_objects``/``evidence``/coverage.

    The connection is never written to and no transaction is opened.  This is
    suitable for browser compilation or audit inspection while another build
    owns the writable database.
    """

    rows = conn.execute(
        "SELECT object_id, json FROM official_objects WHERE json_extract(json, '$.matter_id') = ?",
        (matter_id,),
    ).fetchall()
    objects = [json.loads(row["json"] if isinstance(row, sqlite3.Row) else row[1]) for row in rows]
    question = next((obj for obj in objects if obj.get("kind") == QUESTION_KIND), None)
    if question is None:
        raise KeyError(f"no written question object for matter {matter_id}")
    answer_id = question.get("answer_object_id")
    answer = next((obj for obj in objects if str(obj.get("object_id")) == str(answer_id)), None) if answer_id else None
    object_refs = _unique(
        ref
        for obj in [question, answer]
        if obj is not None
        for ref in _object_evidence_ids(obj)
    )
    record_locators = _unique(
        record.get("record_locator")
        for obj in [question, answer]
        if obj is not None
        for record in obj.get("source_records") or []
        if isinstance(record, Mapping)
    )
    evidence_rows: list[dict[str, Any]] = []
    evidence_where: list[str] = []
    evidence_params: list[str] = []
    if object_refs:
        placeholders = ",".join("?" for _ in object_refs)
        evidence_where.append(f"evidence_id IN ({placeholders})")
        evidence_params.extend(object_refs)
    if record_locators:
        placeholders = ",".join("?" for _ in record_locators)
        evidence_where.append(f"json_extract(json, '$.record_locator') IN ({placeholders})")
        evidence_params.extend(record_locators)
    if evidence_where:
        evidence_rows = [
            json.loads(row["json"] if isinstance(row, sqlite3.Row) else row[1])
            for row in conn.execute(
                f"SELECT evidence_id, json FROM evidence WHERE {' OR '.join(evidence_where)}",
                evidence_params,
            ).fetchall()
        ]
    coverage_rows = [
        json.loads(row["json"] if isinstance(row, sqlite3.Row) else row[1])
        for row in conn.execute("SELECT coverage_id, json FROM source_coverage").fetchall()
        if json.loads(row["json"] if isinstance(row, sqlite3.Row) else row[1]).get("kind") == "WRITTEN_QUESTION_REGISTER"
    ]
    return build_written_question_episode(
        question,
        answer,
        evidence=evidence_rows,
        coverage=coverage_rows,
        actor_map=actor_map,
    )


__all__ = [
    "ANSWER_KIND",
    "EPISODE_KIND",
    "EPISODE_SCHEMA_VERSION",
    "QUESTION_KIND",
    "EpisodeError",
    "build_episodes_from_objects",
    "build_written_question_episode",
    "build_written_question_episode_from_db",
    "validate_episode",
]

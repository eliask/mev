"""Exact matter-to-actor intervention ledgers.

The ledger is a read-only projection of the normalized official records.  It
answers a narrow question: *which source-identified people have an observable
intervention on this exact matter?*  A matter identifier is the only join key
between records.  Text search is exposed separately as a candidate-only
projection and is never promoted into an intervention.

This module deliberately keeps several tempting inferences out of the result:

* an initiative's ``AUTHOR`` field is a first signature, not proof of drafting;
  the maintained ``action_kind`` for that role is ``FIRST_SIGNATORY`` (not
  ``INITIATIVE_AUTHORED``);
* a ballot belongs to the published vote/matter package, not to every component
  phrase in a motion title;
* member-group/role intervals establish an overlapping recorded role, not
  committee influence or causal responsibility;
* a respondent named without a source person ID remains name-only.

The database helper only reads.  ``build_matter_ledger`` is also usable with a
small in-memory database in tests, while the CLI supplies JSON and a sharded
static HTML site as an actual consumer.
"""

import hashlib
import json
import re
import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from paa.chair_title_proposers import (
    finnish_person_name_keys,
    parse_chair_title_named_proposers,
    published_alternative_disposition,
)
from paa.config import CORPUS_CUTOFF, DB_PATH
from paa.group_agreement import (
    DEFAULT_MINIMUM_PEERS,
    build_vote_ballot_source,
    compact_peer_comparison,
    compare_people_on_source,
)
from paa.records import alternatives_from_title

LEDGER_SCHEMA_VERSION = "paa.matter-ledger.v4"

# Retained v3 (and earlier) packets may still carry INITIATIVE_AUTHORED for an
# official first signature. That label is unsafe: drafting stays UNKNOWN.
# Decode only the matter-ledger signature projection; do not rewrite
# opportunity/facet commitment action kinds that legitimately use the same
# token for source-text claims about initiating a bill.
_LEGACY_FIRST_SIGNATURE_ACTION = "INITIATIVE_AUTHORED"
_FIRST_SIGNATORY_ACTION = "FIRST_SIGNATORY"

_OBJECT_KINDS = frozenset(
    {
        "LEGISLATIVE_INITIATIVE",
        "WRITTEN_QUESTION",
        "GOVERNMENT_ANSWER",
        "SPEECH",
        "VOTE",
        "RESIGN_ROLE",
    }
)
_SOURCE_KIND_TO_COVERAGE = {
    "LEGISLATIVE_INITIATIVE": "LEGISLATIVE_INITIATIVE",
    "WRITTEN_QUESTION": "WRITTEN_QUESTION_REGISTER",
    "GOVERNMENT_ANSWER": "WRITTEN_QUESTION_REGISTER",
    "SPEECH": "PARLIAMENTARY_SPEECH_REGISTER",
}
_ROLE_KINDS = frozenset({"VaaliPiiri", "Edustajatoimi", "Eduskuntaryhma"})
_PERSON_ID_FIELDS = frozenset({"person_id", "person_number"})


class MatterLedgerError(ValueError):
    """Raised when an exact ledger query cannot be formed."""


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _clean(value: Any) -> str | None:
    value = _text(value)
    return value or None


def matter_subject_title_from_objects(
    objects: Sequence[Mapping[str, Any]] | None,
) -> dict[str, Any]:
    """Retain a source-derived subject title only when official objects agree.

    Vote chair titles are alternatives, not issue subjects.  Conflicting object
    titles stay ambiguous; the identifier remains the join key and fallback.
    """

    titles: list[str] = []
    seen: set[str] = set()
    for obj in objects or ():
        title = _clean(obj.get("title"))
        if not title:
            continue
        key = title.casefold()
        if key in seen:
            continue
        seen.add(key)
        titles.append(title)
    if len(titles) == 1:
        return {
            "subject_title": titles[0],
            "subject_title_state": "COMMON_OBJECT_TITLE",
            "subject_title_candidates": titles,
        }
    if len(titles) > 1:
        return {
            "subject_title": None,
            "subject_title_state": "AMBIGUOUS_OBJECT_TITLES",
            "subject_title_candidates": titles,
        }
    return {
        "subject_title": None,
        "subject_title_state": "MISSING",
        "subject_title_candidates": [],
    }


def _unique(values: Iterable[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _text(value) if not isinstance(value, str) else value.strip()
        if text and text not in seen:
            result.append(text)
            seen.add(text)
    return result


def _mapping(value: Any) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise MatterLedgerError("normalized record mapping has a non-object value")
    return value


def _list_mapping(value: Any) -> list[Mapping[str, Any]]:
    """Validate a normalized list before semantic code consumes it.

    A malformed author/evidence list is a processing failure, not an empty
    source list.  Silently filtering bad members would turn corruption into a
    false absence of an intervention or source anchor.
    """

    if value is None:
        return []
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise MatterLedgerError("normalized record list is not a sequence")
    result: list[Mapping[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise MatterLedgerError(f"normalized record list item {index} is not an object")
        result.append(item)
    return result


def _list_text(value: Any, *, field: str) -> list[str]:
    """Validate string-ID lists rather than iterating malformed scalar data."""

    if value is None:
        return []
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise MatterLedgerError(f"normalized {field} is not a sequence")
    result: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise MatterLedgerError(f"normalized {field}[{index}] is not a non-empty string")
        result.append(item.strip())
    return result


def _json_row(row: sqlite3.Row | Sequence[Any]) -> dict[str, Any]:
    if isinstance(row, sqlite3.Row):
        raw = row["json"]
    else:
        raw = row[1]
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise MatterLedgerError("stored official record is not a JSON object")
    return value


def official_object_matches_matter(obj: Mapping[str, Any], matter_id: str) -> bool:
    """Exact top-level ``matter_id`` equality after parse; serialization-insensitive."""

    return _text(obj.get("matter_id")) == _text(matter_id)


def _matter_object_rows(conn: sqlite3.Connection, matter_id: str) -> list[sqlite3.Row]:
    """Prefetch official-object rows whose top-level JSON ``matter_id`` matches.

    Uses ``json_extract`` so compact, spaced, newline and escaped-slash
    serializers of the same identifier are equal after decode. Nested-only
    occurrences are excluded because they are not ``$.matter_id``. Callers
    still apply ``official_object_matches_matter`` as an exact post-decode
    guard.
    """

    return list(
        conn.execute(
            "SELECT object_id, json FROM official_objects WHERE json_extract(json, '$.matter_id') = ?",
            (_text(matter_id),),
        )
    )


def _actor_from_display_name(
    source_display_name: str,
    *,
    by_name: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Join a chair-title name: exact source spelling first, then stem fallback."""

    source_name = _clean(source_display_name) or ""
    exact_key = source_name.casefold()
    exact_matches = list(by_name.get(exact_key, ()))
    if len(exact_matches) == 1:
        actor = exact_matches[0]
        return {
            "actor_id": actor.get("actor_id"),
            "person_id": actor.get("person_id"),
            "name": source_name,
            "resolved_display_name": actor.get("display_name") or source_name,
            "source_display_name": source_name,
            "identity_state": "EXACT_SOURCE_DISPLAY_NAME",
            "identity_basis": "CHAIR_TITLE_SOURCE_DISPLAY_NAME",
            "source_role": "CHAIR_TITLE_NAMED_PROPOSER",
            "source_party": None,
            "source_parliamentary_group": None,
        }
    if len(exact_matches) > 1:
        return {
            "actor_id": None,
            "person_id": None,
            "name": source_name,
            "resolved_display_name": None,
            "source_display_name": source_name,
            "identity_state": "CHAIR_TITLE_NAME_AMBIGUOUS",
            "identity_basis": "CHAIR_TITLE_SOURCE_DISPLAY_NAME",
            "source_role": "CHAIR_TITLE_NAMED_PROPOSER",
            "source_party": None,
            "source_parliamentary_group": None,
            "candidate_actor_ids": [_clean(item.get("actor_id")) for item in exact_matches],
        }
    # Conservative inflection candidates only after exact miss.
    matches: list[Mapping[str, Any]] = []
    seen_actor_ids: set[str] = set()
    for key in finnish_person_name_keys(source_name):
        if key == exact_key:
            continue
        for actor in by_name.get(key, ()):
            actor_id = _text(actor.get("actor_id"))
            if actor_id and actor_id in seen_actor_ids:
                continue
            if actor_id:
                seen_actor_ids.add(actor_id)
            matches.append(actor)
    if len(matches) == 1:
        actor = matches[0]
        return {
            "actor_id": actor.get("actor_id"),
            "person_id": actor.get("person_id"),
            "name": source_name,
            "resolved_display_name": actor.get("display_name") or source_name,
            "source_display_name": source_name,
            "identity_state": "INFLECTED_SOURCE_DISPLAY_NAME",
            "identity_basis": "CHAIR_TITLE_INFLECTED_FALLBACK",
            "source_role": "CHAIR_TITLE_NAMED_PROPOSER",
            "source_party": None,
            "source_parliamentary_group": None,
        }
    return {
        "actor_id": None,
        "person_id": None,
        "name": source_name,
        "resolved_display_name": None,
        "source_display_name": source_name,
        "identity_state": "CHAIR_TITLE_NAME_UNRESOLVED" if not matches else "CHAIR_TITLE_NAME_AMBIGUOUS",
        "identity_basis": "CHAIR_TITLE_SOURCE_DISPLAY_NAME",
        "source_role": "CHAIR_TITLE_NAMED_PROPOSER",
        "source_party": None,
        "source_parliamentary_group": None,
        "candidate_actor_ids": [_clean(item.get("actor_id")) for item in matches] or None,
    }


def _object_evidence_ids(obj: Mapping[str, Any]) -> list[str]:
    values: list[Any] = []
    values.extend(_list_text(obj.get("evidence_ids"), field="evidence_ids"))
    provenance = _mapping(obj.get("action_date_provenance"))
    values.extend(_list_text(provenance.get("evidence_ids"), field="action_date_provenance.evidence_ids"))
    disposition = _mapping(obj.get("disposition"))
    values.extend(_list_text(disposition.get("evidence_ids"), field="disposition.evidence_ids"))
    for author in _list_mapping(obj.get("authors")):
        values.extend(_list_text(author.get("evidence_ids"), field="authors[].evidence_ids"))
    for source_record in _list_mapping(obj.get("source_records")):
        values.extend(_list_text(source_record.get("evidence_ids"), field="source_records[].evidence_ids"))
    return _unique(values)


def _object_evidence_rows(obj: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Retain embedded source anchors instead of reducing them to IDs only."""

    return [dict(item) for item in _list_mapping(obj.get("evidence"))]


def _resolve_object_evidence(conn: sqlite3.Connection, obj: Mapping[str, Any]) -> dict[str, Any]:
    """Attach retained evidence-table rows for direct object evidence IDs.

    Most official objects embed their evidence packet.  The normalized VOTE
    shell is an important exception: it carries ``evidence_ids`` while the
    canonical evidence row lives in ``evidence``.  Resolve that row before
    publishing action-date references, preserving the stored object itself.
    """

    embedded = _object_evidence_rows(obj)
    embedded_ids = {_clean(item.get("evidence_id")) for item in embedded}
    requested_ids = _object_evidence_ids(obj)
    missing_ids = [evidence_id for evidence_id in requested_ids if evidence_id not in embedded_ids]
    if not missing_ids:
        return dict(obj)
    placeholders = ",".join("?" for _ in missing_ids)
    rows = conn.execute(
        f"SELECT json FROM evidence WHERE evidence_id IN ({placeholders}) ORDER BY evidence_id",
        tuple(missing_ids),
    )
    resolved = list(embedded)
    for row in rows:
        evidence = _json_row(row)
        evidence_id = _clean(evidence.get("evidence_id"))
        if evidence_id and evidence_id not in embedded_ids:
            resolved.append(evidence)
            embedded_ids.add(evidence_id)
    result = dict(obj)
    result["evidence"] = resolved
    return result


def _action_date(obj: Mapping[str, Any]) -> tuple[str | None, str | None, list[str]]:
    date = _clean(obj.get("action_date"))
    basis = _clean(obj.get("action_date_basis"))
    provenance = _mapping(obj.get("action_date_provenance"))
    if not date and _text(obj.get("kind")) == "VOTE":
        # The normalized vote fixture predates action_date naming but retains
        # its dated vote record.  Preserve that source field with an explicit
        # fallback basis rather than treating it as a publication date.
        date = _clean(obj.get("date"))
        basis = "VOTE_DATE" if date else None
        return date, basis, _object_evidence_ids(obj)
    return date, basis, _list_text(provenance.get("evidence_ids"), field="action_date_provenance.evidence_ids")


def _object_source_record(obj: Mapping[str, Any]) -> dict[str, Any]:
    source_records = _list_mapping(obj.get("source_records"))
    first = source_records[0] if source_records else {}
    return {
        "source_id": _clean(obj.get("source_id")),
        "record_locator": _clean(obj.get("record_locator") or obj.get("source_locator") or first.get("record_locator")),
        "raw_sha256": _clean(obj.get("raw_sha256") or obj.get("source_raw_sha256") or first.get("raw_sha256")),
        "url": _clean(obj.get("url") or obj.get("source_url") or first.get("url")),
        # Keep every retained source-record anchor.  The first record is only
        # a compatibility summary; date provenance may point to a later one.
        "source_records": [dict(record) for record in source_records],
    }


def _vote_id_from_object(obj: Mapping[str, Any]) -> str | None:
    object_id = _text(obj.get("object_id"))
    match = re.search(r"(?:vote-|AanestysId=)(\d+)$", object_id, re.IGNORECASE)
    return match.group(1) if match else None


def _row_fields(row: sqlite3.Row) -> dict[str, Any]:
    """Return normalized SQLite fields in a stable key order for hashing."""

    keys = row.keys()
    return {keys[index]: row[index] for index in range(len(keys))}


def _local_record_evidence(
    *,
    record_kind: str,
    record_locator: str,
    fields: Mapping[str, Any],
) -> dict[str, Any]:
    """Make evidence for a normalized local row without posing as source capture.

    The canonical JSON serialization and its digest bind the evidence to the
    exact normalized values used by this projection.  ``raw_sha256`` remains
    null: this is reproducible local-record evidence, not an independently
    checked upstream response body.
    """

    serialized = json.dumps(
        {"fields": dict(fields), "record_kind": record_kind, "record_locator": record_locator},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    prefix = record_kind.casefold().replace("_", "-")
    return {
        "evidence_id": f"local-{prefix}-{digest[:24]}",
        "kind": "NORMALIZED_LOCAL_RECORD",
        "status": "NORMALIZED_LOCAL_RECORD",
        "observation_status": "NORMALIZED_LOCAL_RECORD",
        "serialization": "canonical-json-v1",
        "quote": serialized,
        "text_sha256": digest,
        "record_locator": record_locator,
        "source_id": "SRC-LOCAL-PAA-SQLITE",
        "source_version": "paa.sqlite.normalized",
        "raw_sha256": None,
    }


def _local_vote_event_evidence(vote: sqlite3.Row) -> dict[str, Any]:
    vote_id = _text(vote["aanestys_id"])
    return _local_record_evidence(
        record_kind="VOTE_EVENT",
        record_locator=f"vote_events.aanestys_id={vote_id}",
        fields=_row_fields(vote),
    )


def _local_ballot_evidence(vote_id: str, ballot: sqlite3.Row) -> dict[str, Any]:
    person_number = _text(ballot["person_number"])
    return _local_record_evidence(
        record_kind="BALLOT_ROW",
        record_locator=(
            f"ballots.aanestys_id={vote_id};ballots.person_number={person_number}"
        ),
        fields=_row_fields(ballot),
    )


def _object_summary(obj: Mapping[str, Any]) -> dict[str, Any]:
    action_date, action_basis, action_date_evidence_ids = _action_date(obj)
    source = _object_source_record(obj)
    return {
        "object_id": _clean(obj.get("object_id")),
        "kind": _clean(obj.get("kind")),
        "matter_id": _clean(obj.get("matter_id")),
        "title": obj.get("title"),
        "text": obj.get("text"),
        "action_date": action_date,
        "action_date_basis": action_basis,
        "action_date_evidence_ids": action_date_evidence_ids,
        "signature_date": _clean(obj.get("signature_date")),
        "signature_date_basis": _clean(obj.get("signature_date_basis")),
        "publication_date": _clean(obj.get("publication_date")),
        "source": source,
        "source_id": source["source_id"],
        "evidence_ids": _object_evidence_ids(obj),
        "evidence": _object_evidence_rows(obj),
        "disposition": dict(_mapping(obj.get("disposition"))),
        "authors": [dict(author) for author in _list_mapping(obj.get("authors"))],
        "url": source["url"],
    }


def _normalise_actor_map(
    conn: sqlite3.Connection,
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, list[dict[str, Any]]],
    dict[str, dict[str, Any]],
    dict[str, list[dict[str, Any]]],
]:
    person_candidates: dict[str, list[dict[str, Any]]] = {}
    by_name: dict[str, list[dict[str, Any]]] = {}
    by_actor: dict[str, dict[str, Any]] = {}
    for row in conn.execute(
        "SELECT actor_id, display_name, name_key, birth_year, person_id, identity_status FROM actors ORDER BY actor_id"
    ):
        actor = {
            "actor_id": _clean(row["actor_id"]),
            "display_name": _clean(row["display_name"]),
            "name_key": _clean(row["name_key"]),
            "birth_year": row["birth_year"],
            "person_id": _clean(row["person_id"]),
            "identity_status": _clean(row["identity_status"]),
        }
        if actor["actor_id"]:
            by_actor[actor["actor_id"]] = actor
        person_id = actor["person_id"]
        if person_id:
            person_candidates.setdefault(person_id, []).append(actor)
        name = (actor["display_name"] or "").casefold().strip()
        if name:
            by_name.setdefault(name, []).append(actor)
    by_person = {
        person_id: candidates[0]
        for person_id, candidates in person_candidates.items()
        if len(candidates) == 1
    }
    ambiguous_persons = {
        person_id: candidates
        for person_id, candidates in person_candidates.items()
        if len(candidates) > 1
    }
    return by_person, by_name, by_actor, ambiguous_persons


def _resolve_actor_filter(
    conn: sqlite3.Connection,
    *,
    actor_id: str | None,
    person_id: str | None,
    actor_name: str | None,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    supplied = [bool(actor_id), bool(person_id), bool(actor_name)]
    if sum(supplied) > 1:
        raise MatterLedgerError("at most one of actor_id, person_id, and actor_name may be supplied")
    by_person, by_name, _by_actor, ambiguous_persons = _normalise_actor_map(conn)
    if actor_id:
        row = conn.execute("SELECT * FROM actors WHERE actor_id = ?", (actor_id,)).fetchone()
        if row is None:
            return {
                "match_basis": "EXACT_ACTOR_ID_UNRESOLVED",
                "actor_id": actor_id,
                "person_id": None,
                "display_name": None,
                "unresolved": True,
            }, [{"code": "ACTOR_NOT_FOUND", "reason": "No exact actor_id exists in the local actor table."}]
        selected = {
            "match_basis": "EXACT_ACTOR_ID",
            "actor_id": _clean(row["actor_id"]),
            "person_id": _clean(row["person_id"]),
            "display_name": _clean(row["display_name"]),
        }
        if selected["person_id"] in ambiguous_persons:
            selected["person_id_ambiguity"] = True
        return selected, []
    if person_id:
        if person_id in ambiguous_persons:
            return {
                "match_basis": "EXACT_PERSON_ID_AMBIGUOUS",
                "actor_id": None,
                "person_id": person_id,
                "display_name": None,
                "candidate_actor_ids": [item["actor_id"] for item in ambiguous_persons[person_id]],
                "unresolved": True,
            }, [{
                "code": "PERSON_ID_AMBIGUOUS",
                "reason": "More than one actor row carries this person_id; no actor was selected.",
                "person_id": person_id,
                "candidate_actor_ids": [item["actor_id"] for item in ambiguous_persons[person_id]],
            }]
        actor = by_person.get(person_id)
        if actor is None:
            return {
                "match_basis": "EXACT_PERSON_ID_UNRESOLVED",
                "actor_id": None,
                "person_id": person_id,
                "display_name": None,
                "unresolved": True,
            }, [{"code": "PERSON_ID_NOT_FOUND", "reason": "No exact person_id-to-actor row exists."}]
        return {
            "match_basis": "EXACT_PERSON_ID",
            "actor_id": actor["actor_id"],
            "person_id": actor["person_id"],
            "display_name": actor["display_name"],
        }, []
    if actor_name:
        matches = by_name.get(actor_name.casefold().strip(), [])
        if len(matches) == 1:
            actor = matches[0]
            selected = {
                "match_basis": "EXACT_NORMALIZED_DISPLAY_NAME_UNIQUE",
                "actor_id": actor["actor_id"],
                "person_id": actor["person_id"],
                "display_name": actor["display_name"],
            }
            if selected["person_id"] in ambiguous_persons:
                selected["person_id_ambiguity"] = True
            return selected, []
        if len(matches) > 1:
            return {
                "match_basis": "EXACT_NORMALIZED_DISPLAY_NAME_AMBIGUOUS",
                "actor_id": None,
                "person_id": None,
                "display_name": actor_name,
                "unresolved": True,
            }, [{"code": "ACTOR_NAME_AMBIGUOUS", "reason": "More than one actor has the exact normalized display name."}]
        return {
            "match_basis": "EXACT_NORMALIZED_DISPLAY_NAME_UNRESOLVED",
            "actor_id": None,
            "person_id": None,
            "display_name": actor_name,
            "unresolved": True,
        }, [{"code": "ACTOR_NAME_NOT_FOUND", "reason": "No exact normalized display name exists in the local actor table."}]
    return None, []


def _actor_matches(actor: Mapping[str, Any], selected: Mapping[str, Any] | None) -> bool:
    if selected is None:
        return True
    if selected.get("unresolved"):
        return False
    basis = _text(selected.get("match_basis"))
    # An actor_id query is an actor-row query.  Do not let a source record
    # carrying the same (ambiguous) person_id leak another actor into it.
    if basis.startswith("EXACT_ACTOR_ID"):
        return bool(selected.get("actor_id") and actor.get("actor_id") == selected.get("actor_id"))
    if basis.startswith("EXACT_PERSON_ID"):
        return bool(selected.get("person_id") and actor.get("person_id") == selected.get("person_id"))
    if basis.startswith("EXACT_NORMALIZED_DISPLAY_NAME"):
        if selected.get("person_id_ambiguity"):
            return bool(selected.get("actor_id") and actor.get("actor_id") == selected.get("actor_id"))
        return bool(
            (selected.get("actor_id") and actor.get("actor_id") == selected.get("actor_id"))
            or (selected.get("person_id") and actor.get("person_id") == selected.get("person_id"))
        )
    return False


def _role_evidence_id(person_id: str, row: Mapping[str, Any]) -> str:
    start = _text(row.get("start_date")) or "unknown"
    kind = _text(row.get("kind")) or "unknown"
    return f"role-{person_id}-{kind}-{start}"


def _roles_for_action(
    conn: sqlite3.Connection,
    person_id: str | None,
    action_date: str | None,
    *,
    ambiguous_persons: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
) -> dict[str, Any]:
    base = {
        "state": "DATE_UNKNOWN" if not action_date else "NO_OVERLAP",
        "roles": [],
        "evidence_ids": [],
        "evidence": [],
        "limitations": [],
    }
    if not person_id:
        base["state"] = "PERSON_ID_UNRESOLVED"
        base["limitations"].append("No source person ID is available for an exact role-interval join.")
        return base
    if ambiguous_persons and person_id in ambiguous_persons:
        base["state"] = "PERSON_ID_AMBIGUOUS"
        base["limitations"].append("Multiple actor rows carry this person_id; no role interval was attributed.")
        return base
    if not action_date:
        base["limitations"].append("The action has no source action date, so a role interval cannot be joined.")
        return base
    imprecise_interval = False
    rows = conn.execute(
        """SELECT person_id, kind, label, start_date, end_date, precision
           FROM mp_periods WHERE person_id = ? ORDER BY start_date, kind, label""",
        (person_id,),
    )
    for row in rows:
        role = dict(row)
        if role["kind"] not in _ROLE_KINDS:
            continue
        if role.get("precision") != "day" or not _text(role.get("start_date")):
            imprecise_interval = True
            continue
        start = _text(role.get("start_date"))
        end = _text(role.get("end_date")) or "9999-12-31"
        if start > action_date or end < action_date:
            continue
        evidence_id = _role_evidence_id(person_id, role)
        role_locator = f"MemberOfParliament.personId={person_id};{role['kind']}"
        role_quote = json.dumps(role, ensure_ascii=False, sort_keys=True)
        role.update(
            {
                "match_state": "EXACT_DATE_OVERLAP",
                "source_id": "SRC-EDUSKUNTA-MEMBER-REGISTER",
                "source_url": (
                    "https://avoindata.eduskunta.fi/api/v1/tables/MemberOfParliament/rows?"
                    f"columnName=personId&columnValue={person_id}"
                ),
                "record_locator": role_locator,
                "evidence_ids": [evidence_id],
                "influence_state": "NOT_ASSESSED",
            }
        )
        base["evidence"].append(
            {
                "evidence_id": evidence_id,
                "kind": "structured_field",
                "quote": role_quote,
                "text_sha256": hashlib.sha256(role_quote.encode("utf-8")).hexdigest(),
                "record_locator": role_locator,
                "source_id": "SRC-EDUSKUNTA-MEMBER-REGISTER",
                "source_url": role["source_url"],
            }
        )
        base["roles"].append(role)
        base["evidence_ids"].append(evidence_id)
    base['imprecise_intervals_omitted'] = imprecise_interval
    if base["roles"]:
        base["state"] = "EXACT_DATE_OVERLAP"
        if imprecise_interval:
            base["limitations"].append(
                "Some loaded member-register intervals lack day precision and were not considered for exact overlap."
            )
    elif imprecise_interval:
        base["state"] = "ROLE_PRECISION_INSUFFICIENT"
        base["limitations"].append("Loaded role intervals lack a day-precision start date for an exact overlap.")
    else:
        base["limitations"].append("No loaded member-register interval overlaps the action date.")
    # The member register has no committee membership/influence relation in
    # this normalized slice.  Make that missingness explicit instead of
    # treating an Eduskuntaryhma or seat interval as committee influence.
    base["limitations"].append("Committee membership and influence are not established by this role interval join.")
    return base


def _actor_payload(
    source: Mapping[str, Any],
    *,
    by_person: Mapping[str, Mapping[str, Any]],
    by_name: Mapping[str, Sequence[Mapping[str, Any]]],
    by_actor: Mapping[str, Mapping[str, Any]],
    ambiguous_persons: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    source_actor_id = _clean(source.get("actor_id"))
    person_id = _clean(source.get("person_id"))
    name = _clean(source.get("name"))
    actor = by_actor.get(source_actor_id) if source_actor_id else None
    if source_actor_id and actor is not None:
        if person_id and actor.get("person_id") != person_id:
            return {
                "actor_id": None,
                "person_id": person_id,
                "name": name or actor.get("display_name"),
                "source_actor_id": source_actor_id,
                "identity_state": "SOURCE_ACTOR_ID_PERSON_CONFLICT",
                "identity_basis": source.get("identity_basis") or "SOURCE_ACTOR_ID",
                "source_role": _clean(source.get("role")),
                "source_party": _clean(source.get("party")),
                "source_parliamentary_group": _clean(source.get("parliamentary_group")),
                "unresolved": True,
            }
        return {
            "actor_id": actor.get("actor_id"),
            "person_id": actor.get("person_id") or person_id,
            "name": name or actor.get("display_name"),
            "identity_state": "EXACT_SOURCE_ACTOR_ID" if source_actor_id else "EXACT_SOURCE_PERSON_ID",
            "identity_basis": source.get("identity_basis") or ("SOURCE_ACTOR_ID" if source_actor_id else "SOURCE_PERSON_ID"),
            "source_role": _clean(source.get("role")),
            "source_party": _clean(source.get("party")),
            "source_parliamentary_group": _clean(source.get("parliamentary_group")),
        }
    if source_actor_id:
        # Never fall back from an unknown source actor ID to a person row and
        # label the result as an exact actor join.  The source identity is
        # retained, but remains unresolved for actor-level attribution.
        return {
            "actor_id": source_actor_id,
            "person_id": person_id,
            "name": name,
            "identity_state": "SOURCE_ACTOR_ID_NO_ACTOR_ROW",
            "identity_basis": source.get("identity_basis") or "SOURCE_ACTOR_ID",
            "source_role": _clean(source.get("role")),
            "source_party": _clean(source.get("party")),
            "source_parliamentary_group": _clean(source.get("parliamentary_group")),
            "unresolved": True,
        }
    actor = by_person.get(person_id) if person_id else None
    if actor:
        return {
            "actor_id": actor.get("actor_id"),
            "person_id": actor.get("person_id") or person_id,
            "name": name or actor.get("display_name"),
            "identity_state": "EXACT_SOURCE_PERSON_ID",
            "identity_basis": source.get("identity_basis") or "SOURCE_PERSON_ID",
            "source_role": _clean(source.get("role")),
            "source_party": _clean(source.get("party")),
            "source_parliamentary_group": _clean(source.get("parliamentary_group")),
        }
    if person_id and person_id in ambiguous_persons:
        return {
            "actor_id": None,
            "person_id": person_id,
            "name": name,
            "identity_state": "PERSON_ID_AMBIGUOUS",
            "identity_basis": source.get("identity_basis") or "SOURCE_PERSON_ID",
            "source_role": _clean(source.get("role")),
            "source_party": _clean(source.get("party")),
            "source_parliamentary_group": _clean(source.get("parliamentary_group")),
            "candidate_actor_ids": [item.get("actor_id") for item in ambiguous_persons[person_id]],
        }
    if person_id:
        return {
            "actor_id": None,
            "person_id": person_id,
            "name": name,
            "identity_state": "PERSON_ID_NO_ACTOR_ROW",
            "identity_basis": source.get("identity_basis") or "SOURCE_PERSON_ID",
            "source_role": _clean(source.get("role")),
            "source_party": _clean(source.get("party")),
            "source_parliamentary_group": _clean(source.get("parliamentary_group")),
        }
    name_matches = by_name.get((name or "").casefold().strip(), []) if name else []
    return {
        "actor_id": None,
        "person_id": None,
        "name": name,
        "identity_state": "SOURCE_NAME_ONLY_UNJOINED" if len(name_matches) != 1 else "NAME_MATCH_NOT_PROMOTED",
        "identity_basis": source.get("identity_basis") or "SOURCE_NAME_ONLY",
        "source_role": _clean(source.get("role")),
        "source_party": _clean(source.get("party")),
        "source_parliamentary_group": _clean(source.get("parliamentary_group")),
    }


def _role_label(kind: str, role: str | None) -> str:
    if kind == "LEGISLATIVE_INITIATIVE" and role == "AUTHOR":
        return "FIRST_SIGNATURE"
    if kind == "LEGISLATIVE_INITIATIVE" and role == "COSIGNER":
        return "CO_SIGNATURE"
    if kind == "WRITTEN_QUESTION" and role == "AUTHOR":
        return "QUESTION_AUTHOR"
    if kind == "WRITTEN_QUESTION" and role == "COSIGNER":
        return "QUESTION_CO_SIGNATURE"
    if kind == "SPEECH":
        return "SPEAKER"
    if kind == "GOVERNMENT_ANSWER" and role == "RESPONDENT":
        return "RESPONDENT"
    return role or "SOURCE_NAMED_ACTOR"


def decode_ledger_action_kind(
    action_kind: Mapping[str, Any] | str | None,
    *,
    role_label: str | None = None,
    signature_value: str | None = None,
    kind: str | None = None,
) -> dict[str, Any]:
    """Normalize a matter-ledger ``action_kind`` for maintained consumers.

    Fresh ``paa.matter-ledger.v4`` packets emit ``FIRST_SIGNATORY`` for an
    initiative first signature. Legacy retained packets may still store
    ``INITIATIVE_AUTHORED`` for that same official role. Decode the legacy
    token to ``FIRST_SIGNATORY`` when the signature/role evidence is first
    signature; leave other action kinds untouched. This does not invent
    drafting authorship and does not rewrite non-ledger commitment artifacts.
    """

    if isinstance(action_kind, Mapping):
        payload = {
            "value": action_kind.get("value"),
            "basis": action_kind.get("basis"),
        }
        for key, value in action_kind.items():
            if key not in payload:
                payload[key] = value
    elif action_kind is None:
        payload = {"value": None, "basis": None}
    else:
        payload = {"value": str(action_kind), "basis": None}
    value = payload.get("value")
    first_signature = (
        role_label == "FIRST_SIGNATURE"
        or signature_value == "FIRST_SIGNATURE"
        or (kind == "LEGISLATIVE_INITIATIVE" and role_label == "FIRST_SIGNATURE")
    )
    if value == _LEGACY_FIRST_SIGNATURE_ACTION and first_signature:
        payload = dict(payload)
        payload["value"] = _FIRST_SIGNATORY_ACTION
        payload["basis"] = payload.get("basis") or "SOURCE_RECORD_KIND_AND_ROLE"
        payload["decoded_from"] = _LEGACY_FIRST_SIGNATURE_ACTION
        payload["decode_note"] = (
            "Legacy matter-ledger INITIATIVE_AUTHORED for FIRST_SIGNATURE "
            "decodes to FIRST_SIGNATORY; drafting remains unresolved."
        )
    return payload


def _intervention_from_object(
    obj: Mapping[str, Any],
    source_actor: Mapping[str, Any],
    *,
    by_person: Mapping[str, Mapping[str, Any]],
    by_name: Mapping[str, Sequence[Mapping[str, Any]]],
    by_actor: Mapping[str, Mapping[str, Any]],
    ambiguous_persons: Mapping[str, Sequence[Mapping[str, Any]]],
    conn: sqlite3.Connection,
) -> dict[str, Any]:
    kind = _text(obj.get("kind"))
    action_date, action_basis, action_date_evidence_ids = _action_date(obj)
    actor = _actor_payload(
        source_actor,
        by_person=by_person,
        by_name=by_name,
        by_actor=by_actor,
        ambiguous_persons=ambiguous_persons,
    )
    role = _clean(source_actor.get("role"))
    object_evidence = _object_evidence_rows(obj)
    object_evidence_ids = {
        _clean(item.get("evidence_id")) for item in object_evidence if _clean(item.get("evidence_id"))
    }
    unresolved_date_evidence_ids = [
        evidence_id for evidence_id in action_date_evidence_ids if evidence_id not in object_evidence_ids
    ]
    role_intervals = _roles_for_action(
        conn,
        actor.get("person_id"),
        action_date,
        ambiguous_persons=ambiguous_persons,
    )
    evidence_ids = _unique(
        [
            *_object_evidence_ids(obj),
            *_list_text(source_actor.get("evidence_ids"), field="source_actor.evidence_ids"),
            *role_intervals.get("evidence_ids", []),
        ]
    )
    drafting = {
        "state": "UNRESOLVED",
        "value": None,
        "evidence_ids": [],
        "limitation": "Official signature/author fields do not identify who drafted the text.",
    }
    signature = {
        "state": "NOT_APPLICABLE",
        "value": None,
        "role": role,
        "evidence_ids": _list_text(source_actor.get("evidence_ids"), field="source_actor.evidence_ids"),
    }
    if kind == "LEGISLATIVE_INITIATIVE" and role in {"AUTHOR", "COSIGNER"}:
        signature.update(
            {
                "state": "SOURCE_ROLE",
                "value": "FIRST_SIGNATURE" if role == "AUTHOR" else "CO_SIGNATURE",
            }
        )
    elif kind == "WRITTEN_QUESTION" and role in {"AUTHOR", "COSIGNER"}:
        signature.update(
            {
                "state": "SOURCE_ROLE",
                "value": "QUESTION_AUTHOR" if role == "AUTHOR" else "QUESTION_CO_SIGNATURE",
            }
        )
    if kind == "LEGISLATIVE_INITIATIVE":
        if role == "AUTHOR":
            # Official first signature only; drafting stays in ``drafting``.
            action_value = _FIRST_SIGNATORY_ACTION
        elif role == "COSIGNER":
            action_value = "INITIATIVE_COSIGNED"
        else:
            action_value = "OTHER_OBSERVABLE_ACTION"
    elif kind == "WRITTEN_QUESTION":
        if role == "AUTHOR":
            action_value = "QUESTION_FILED"
        elif role == "COSIGNER":
            action_value = "QUESTION_COSIGNED"
        else:
            action_value = "OTHER_OBSERVABLE_ACTION"
    else:
        action_value = {
            "GOVERNMENT_ANSWER": "ANSWER_ATTRIBUTED",
            "SPEECH": "SPEECH_DELIVERED",
            "RESIGN_ROLE": "RESIGN_ROLE",
        }.get(kind, kind)
    return {
        "intervention_id": f"{_text(obj.get('object_id'))}:{actor.get('person_id') or actor.get('name') or 'unresolved'}",
        "matter_id": _clean(obj.get("matter_id")),
        "object_id": _clean(obj.get("object_id")),
        "kind": kind,
        "action_kind": {
            "value": action_value,
            "basis": "SOURCE_RECORD_KIND_AND_ROLE",
        },
        "role": role,
        "role_label": _role_label(kind, role),
        "signature": signature,
        "actor": actor,
        "title": obj.get("title"),
        "source_text": obj.get("text"),
        "action_date": action_date,
        "action_date_basis": action_basis,
        "action_date_evidence_ids": action_date_evidence_ids,
        "date_provenance": {
            "event": {
                "value": action_date,
                "basis": action_basis,
                "evidence_ids": action_date_evidence_ids,
            },
            "signature": {
                "value": _clean(obj.get("signature_date")),
                "basis": _clean(obj.get("signature_date_basis")),
                "evidence_ids": [],
            },
            "publication": {
                "value": _clean(obj.get("publication_date")),
                "evidence_ids": [],
            },
            "capture": {
                "retrieved_at": _unique(item.get("retrieved_at") for item in object_evidence),
                "evidence_ids": _unique(item.get("evidence_id") for item in object_evidence),
            },
            "date_evidence_state": "UNRESOLVED" if unresolved_date_evidence_ids else "RESOLVED",
            "unresolved_evidence_ids": unresolved_date_evidence_ids,
        },
        "source": _object_source_record(obj),
        "evidence_ids": evidence_ids,
        "source_evidence": [*object_evidence, *role_intervals.get("evidence", [])],
        "drafting": drafting,
        "role_intervals": role_intervals,
        "institutional_result": dict(_mapping(obj.get("disposition"))),
        "limitations": [
            "A recorded intervention does not establish influence, causation, implementation, or effects.",
        ],
    }


def _vote_alternatives(vote: Mapping[str, Any], evidence_ids: Sequence[str]) -> dict[str, Any]:
    vote = dict(vote)
    title = _text(vote.get("title"))
    # Keep the delimiter parse useful for review, but never manufacture an
    # evidence identifier when the local source packet has no vote anchor.
    parsed = alternatives_from_title(title, evidence_ids[0] if evidence_ids else "")
    if parsed and not evidence_ids:
        for alternative in parsed:
            alternative["evidence_ids"] = []
    if parsed:
        # ``alternatives_from_title`` is a source-preserving delimiter split;
        # this ledger does not claim the two strings are policy components.
        state = "DELIMITED_TITLE_CANDIDATE_NOT_COMPONENT_PROOF"
    else:
        state = "UNRESOLVED_NO_EXACT_TWO_OPTION_TITLE_SPLIT"
    return {
        "raw_title": title,
        "raw_matter_text": vote.get("kohta") or "",
        "alternatives": parsed,
        "state": state,
        "evidence_ids": list(evidence_ids),
    }


def _vote_source_evidence(
    conn: sqlite3.Connection,
    matter_id: str,
    vote_objects: Sequence[Mapping[str, Any]] | None = None,
    *,
    vote_ids: Sequence[str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Resolve vote anchors per vote ID; never reuse another vote's anchor."""

    by_vote: dict[str, dict[str, dict[str, Any]]] = {}
    object_rows = vote_objects
    if object_rows is None:
        object_rows = []
        for _object_id, raw in _matter_object_rows(conn, matter_id):
            obj = _resolve_object_evidence(conn, json.loads(raw))
            if not official_object_matches_matter(obj, matter_id):
                continue
            object_rows.append(obj)
    for obj in object_rows:
        if _text(obj.get("kind")) != "VOTE" or _text(obj.get("matter_id")) != matter_id:
            continue
        vote_id = _vote_id_from_object(obj)
        if vote_id is None:
            continue
        bucket = by_vote.setdefault(vote_id, {})
        for item in _object_evidence_rows(obj):
            evidence_id = _clean(item.get("evidence_id"))
            if evidence_id:
                bucket[evidence_id] = item
    wanted = {*_unique(vote_ids or ()), *by_vote.keys()}
    if not wanted:
        return {vote_id: list(items.values()) for vote_id, items in by_vote.items()}
    clauses = " OR ".join(["evidence_id LIKE ? OR json LIKE ?"] * len(wanted))
    params: list[str] = []
    for vote_id in sorted(wanted):
        token = f"%{vote_id}%"
        params.extend([token, token])
    for row in conn.execute(f"SELECT evidence_id, json FROM evidence WHERE {clauses}", params):
        item = _json_row(row)
        evidence_id = _clean(item.get("evidence_id"))
        locator = _text(item.get("record_locator"))
        match = re.search(r"(?:vote-|AanestysId=)(\d+)", " ".join([evidence_id or "", locator]))
        if not match:
            continue
        vote_id = match.group(1)
        if vote_id not in wanted:
            continue
        # The official VOTE object is the canonical source packet when it is
        # present.  A legacy normalized ``vote-*`` evidence row may also be
        # in the table, but mixing it into this action would make the event
        # provenance look like two independent anchors (and can reintroduce a
        # synthetic/non-canonical locator).  Use the fallback rows only when
        # no exact VOTE object anchor was retained.
        if by_vote.get(vote_id):
            continue
        if vote_id not in by_vote:
            # Evidence rows can exist without an official-object shell.  They
            # remain usable only when the vote event itself proves the exact
            # matter join; the caller supplies that check by matter_id.
            by_vote[vote_id] = {}
        by_vote[vote_id][evidence_id or locator] = item
    return {vote_id: list(items.values()) for vote_id, items in by_vote.items()}


_KNOWN_BALLOT_RESPONSES = frozenset({"JAA", "EI", "TYHJA", "POISSA"})
_BALLOT_BUCKET_FIELDS = {
    "JAA": "jaa",
    "EI": "ei",
    "TYHJA": "tyhjaa",
    "POISSA": "poissa",
}


def _declared_count(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value if value >= 0 else None
    if isinstance(value, str) and re.fullmatch(r"[0-9]+", value.strip()):
        return int(value.strip())
    return None


def _vote_completeness(vote: sqlite3.Row, ballot_rows: Sequence[sqlite3.Row]) -> dict[str, Any]:
    """Reconcile the published vote buckets with retained ballot rows.

    A row-count match alone is not sufficient: status codes and each declared
    bucket must agree before a vote is labelled complete.  ``None`` remains
    unknown rather than becoming zero through ``or`` coercion.
    """

    declared = {
        field: _declared_count(vote[field])
        for field in ("jaa", "ei", "tyhjaa", "poissa", "yhteensa")
    }
    observed: dict[str, int] = {code: 0 for code in _KNOWN_BALLOT_RESPONSES}
    unknown_responses: list[str | None] = []
    for ballot in ballot_rows:
        response = _clean(ballot["raw_response"])
        if response in observed:
            observed[response] += 1
        else:
            unknown_responses.append(response)
    observed_buckets = {
        field: observed[code] for code, field in _BALLOT_BUCKET_FIELDS.items()
    }
    declared_buckets = [declared[field] for field in ("jaa", "ei", "tyhjaa", "poissa")]
    expected_total = sum(item for item in declared_buckets if item is not None) if all(
        item is not None for item in declared_buckets
    ) else None
    row_count = len(ballot_rows)
    total = declared["yhteensa"]
    bucket_match = (
        all(declared[field] == observed_buckets[field] for field in observed_buckets)
        if all(declared[field] is not None for field in observed_buckets)
        else None
    )
    total_match = total == expected_total if total is not None and expected_total is not None else None
    row_count_match = total == row_count if total is not None else None
    if not ballot_rows:
        state = "NO_LOCAL_BALLOT_ROWS"
    elif unknown_responses:
        state = "INVALID_BALLOT_RESPONSES"
    elif total is None or bucket_match is None or total_match is None:
        state = "DECLARED_TOTAL_OR_BUCKET_UNKNOWN"
    elif not row_count_match or not bucket_match or not total_match:
        state = "INCOMPLETE_OR_UNRECONCILED"
    else:
        state = "COMPLETE_FOR_DECLARED_VOTE"
    return {
        "published_total": total,
        "stored_ballot_rows": row_count,
        "declared_counts": declared,
        "observed_counts": observed_buckets,
        "unknown_responses": unknown_responses,
        "expected_total_from_buckets": expected_total,
        "reconciliation": {
            "row_count_matches_published_total": row_count_match,
            "buckets_match_published_counts": bucket_match,
            "published_total_matches_bucket_sum": total_match,
        },
        "state": state,
    }


def _vote_interventions(
    conn: sqlite3.Connection,
    matter_id: str,
    selected_actor: Mapping[str, Any] | None,
    *,
    by_person: Mapping[str, Mapping[str, Any]],
    by_name: Mapping[str, Sequence[Mapping[str, Any]]],
    by_actor: Mapping[str, Mapping[str, Any]],
    ambiguous_persons: Mapping[str, Sequence[Mapping[str, Any]]],
    vote_objects: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    interventions: list[dict[str, Any]] = []
    vote_sources: list[dict[str, Any]] = []
    missingness: list[dict[str, Any]] = []
    group_peer_sources: list[dict[str, Any]] = []
    vote_ids = [
        _text(row["aanestys_id"])
        for row in conn.execute(
            "SELECT aanestys_id FROM vote_events WHERE matter = ? ORDER BY session_date, aanestys_id",
            (matter_id,),
        )
        if _text(row["aanestys_id"])
    ]
    vote_source_evidence = _vote_source_evidence(
        conn, matter_id, vote_objects, vote_ids=vote_ids
    )
    vote_object_ids: dict[str, str] = {}
    vote_object_sources: dict[str, dict[str, Any]] = {}
    for obj in vote_objects or []:
        if _text(obj.get("kind")) != "VOTE" or _text(obj.get("matter_id")) != matter_id:
            continue
        vote_id = _vote_id_from_object(obj)
        object_id = _clean(obj.get("object_id"))
        if vote_id:
            if object_id:
                vote_object_ids[vote_id] = object_id
            vote_object_sources[vote_id] = _object_source_record(obj)
    for vote in conn.execute("SELECT * FROM vote_events WHERE matter = ? ORDER BY session_date, aanestys_id", (matter_id,)):
        vote_id = _text(vote["aanestys_id"])
        original_source_evidence = vote_source_evidence.get(vote_id, [])
        local_event_evidence = _local_vote_event_evidence(vote)
        source_evidence = [*original_source_evidence, local_event_evidence]
        original_source_evidence_ids = _unique(
            item.get("evidence_id") for item in original_source_evidence
        )
        local_event_evidence_id = _clean(local_event_evidence.get("evidence_id"))
        event_evidence_ids = _unique(
            [*original_source_evidence_ids, local_event_evidence_id]
        )
        ballot_rows = list(conn.execute("SELECT * FROM ballots WHERE aanestys_id = ? ORDER BY person_number", (vote_id,)))
        completeness = _vote_completeness(vote, ballot_rows)
        # One normalised peer source per vote; batch compares reuse its private prepare.
        peer_source = build_vote_ballot_source(vote, ballot_rows, cutoff=CORPUS_CUTOFF)
        group_peer_sources.append(peer_source)
        peer_person_ids = [
            person
            for person in (_text(row["person_number"]) for row in ballot_rows)
            if person
        ]
        peer_comparisons = (
            compare_people_on_source(
                peer_source,
                peer_person_ids,
                minimum_peer_count=DEFAULT_MINIMUM_PEERS,
            )
            if peer_person_ids
            else {}
        )
        canonical_source = vote_object_sources.get(vote_id)
        title = _text(vote["title"])
        chair_proposers = parse_chair_title_named_proposers(title)
        disposition = published_alternative_disposition(
            jaa=_declared_count(vote["jaa"]),
            ei=_declared_count(vote["ei"]),
        )
        vote_sources.append(
            {
                "source_id": "SRC-EDUSKUNTA-VOTES",
                "kind": "VOTE_METADATA",
                "vote_id": vote_id,
                "record_locator": f"vote_events.aanestys_id={vote_id}",
                "event_record_locator": f"vote_events.aanestys_id={vote_id}",
                "canonical_source": canonical_source,
                "action_date": _clean(vote["session_date"]),
                "title": vote["title"],
                "matter_text": vote["kohta"],
                "completeness": completeness,
                "chair_title_proposers": [item.to_wire() for item in chair_proposers],
                "published_disposition": disposition,
                "jaa": _declared_count(vote["jaa"]),
                "ei": _declared_count(vote["ei"]),
                "tyhjaa": _declared_count(vote["tyhjaa"]),
                "poissa": _declared_count(vote["poissa"]),
                "group_peer_source_id": peer_source.get("source_id"),
                "group_peer_source_sha256": peer_source.get("source_sha256"),
                "evidence_ids": event_evidence_ids,
                "original_source_evidence_ids": original_source_evidence_ids,
                "local_evidence_ids": [local_event_evidence_id] if local_event_evidence_id else [],
                "source_evidence": source_evidence,
                "source_url": (canonical_source or {}).get("url") or _clean(vote["url"]),
            }
        )
        for proposer in chair_proposers:
            actor = _actor_from_display_name(proposer.source_display_name, by_name=by_name)
            if not _actor_matches(actor, selected_actor):
                continue
            action_date = _clean(vote["session_date"])
            role_intervals = _roles_for_action(
                conn,
                actor.get("person_id"),
                action_date,
                ambiguous_persons=ambiguous_persons,
            )
            side_key = "jaa_side_state" if proposer.side == "JAA" else "ei_side_state"
            side_state = _text(disposition.get(side_key)) or "UNRESOLVED"
            interventions.append(
                {
                    "intervention_id": (
                        f"formal-alternative:{vote_id}:{proposer.slot}:{proposer.source_display_name}"
                    ),
                    "matter_id": matter_id,
                    "object_id": vote_object_ids.get(vote_id),
                    "event_id": vote_id,
                    "kind": "FORMAL_ALTERNATIVE",
                    "action_kind": {
                        "value": "ALTERNATIVE_PUBLICLY_ATTRIBUTED",
                        "basis": proposer.basis,
                    },
                    "role": "CHAIR_TITLE_NAMED_PROPOSER",
                    "role_label": "CHAIR_TITLE_NAMED_PROPOSER",
                    "signature": {
                        "state": "NOT_APPLICABLE",
                        "value": None,
                        "role": "CHAIR_TITLE_NAMED_PROPOSER",
                        "evidence_ids": event_evidence_ids,
                    },
                    "actor": actor,
                    "title": vote["title"],
                    "source_text": vote["kohta"],
                    "action_date": action_date,
                    "action_date_basis": "SESSION_DATE" if action_date else None,
                    "action_date_evidence_ids": event_evidence_ids if action_date else [],
                    "source": {
                        "source_id": "SRC-EDUSKUNTA-VOTES",
                        "record_locator": f"vote_events.aanestys_id={vote_id};title",
                        "event_record_locator": f"vote_events.aanestys_id={vote_id}",
                        "raw_sha256": None,
                        "url": _clean(vote["url"]),
                    },
                    "evidence_ids": _unique([*event_evidence_ids, *role_intervals.get("evidence_ids", [])]),
                    "source_evidence": [*source_evidence, *role_intervals.get("evidence", [])],
                    "chair_title_attribution": proposer.to_wire(),
                    "alternatives": _vote_alternatives(vote, event_evidence_ids),
                    "drafting": {
                        "state": "UNRESOLVED",
                        "value": None,
                        "evidence_ids": [],
                        "limitation": "A chair-title name attributes the published alternative, not private drafting.",
                    },
                    "role_intervals": role_intervals,
                    "institutional_result": {
                        "state": side_state,
                        "side": proposer.side,
                        "slot": proposer.slot,
                        "basis": disposition.get("basis"),
                        "published_disposition": disposition,
                        "final_adoption": disposition.get("final_adoption"),
                        "claim": (
                            "Published JAA/EI totals decide only which side prevailed at this "
                            "vote. They are not final adoption, enactment or causation."
                        ),
                        "evidence_ids": event_evidence_ids,
                    },
                    "limitations": [
                        "Chair-title naming is a published attribution of the alternative, not drafting authorship.",
                        "Package vote linkage is not component-level policy support for every phrase.",
                        "Vote-stage prevalence is not final adoption unless a final-stage witness binds it.",
                    ],
                }
            )
        if not ballot_rows:
            missingness.append(
                {
                    "code": "NO_BALLOT_ROWS_FOR_VOTE",
                    "severity": "CONTEXT",
                    "reason": f"Vote {vote_id} has metadata but no local ballot rows.",
                    "source_ids": ["SRC-EDUSKUNTA-SALIDBAANESTYSEDUSTAJA"],
                }
            )
            if completeness["published_total"] is None:
                missingness.append(
                    {
                        "code": "VOTE_COMPLETENESS_UNKNOWN",
                        "severity": "CONTEXT",
                        "reason": "The published vote total is null and no local ballot rows are available; zero was not substituted.",
                        "vote_id": vote_id,
                        "record_locator": f"vote_events.aanestys_id={vote_id}",
                    }
                )
        if completeness["state"] == "INVALID_BALLOT_RESPONSES":
            missingness.append(
                {
                    "code": "BALLOT_RESPONSES_INVALID",
                    "severity": "CONTEXT",
                    "reason": "At least one retained ballot response is outside the known source vocabulary.",
                    "vote_id": vote_id,
                    "record_locator": f"vote_events.aanestys_id={vote_id}",
                    "unknown_responses": completeness["unknown_responses"],
                }
            )
        elif completeness["state"] == "DECLARED_TOTAL_OR_BUCKET_UNKNOWN":
            missingness.append(
                {
                    "code": "VOTE_COMPLETENESS_UNKNOWN",
                    "severity": "CONTEXT",
                    "reason": "A published total or response bucket is null/non-numeric; zero was not substituted.",
                    "vote_id": vote_id,
                    "record_locator": f"vote_events.aanestys_id={vote_id}",
                }
            )
        elif completeness["state"] == "INCOMPLETE_OR_UNRECONCILED":
            missingness.append(
                {
                    "code": "VOTE_BALLOT_BUCKET_MISMATCH",
                    "severity": "CONTEXT",
                    "reason": "Retained ballot rows do not reconcile with the published response buckets/total.",
                    "vote_id": vote_id,
                    "record_locator": f"vote_events.aanestys_id={vote_id}",
                }
            )
        if not original_source_evidence:
            missingness.append(
                {
                    "code": "VOTE_SOURCE_ANCHOR_NOT_LOADED",
                    "severity": "CONTEXT",
                    "reason": f"Vote {vote_id} has no retained source-version evidence row in this database.",
                    "record_locator": f"AanestysId={vote_id}; KohtaOtsikko",
                }
            )
        if ballot_rows:
            missingness.append(
                {
                    "code": "BALLOT_SOURCE_VERSION_NOT_BOUND",
                    "severity": "CONTEXT",
                    "reason": "Normalized ballot rows have actor locators but no retained per-row raw hash/source-version evidence binding.",
                    "source_id": "SRC-EDUSKUNTA-SALIDBAANESTYSEDUSTAJA",
                    "record_locator": f"SaliDBAanestysEdustaja.AanestysId={vote_id}",
                }
            )
        alternatives = _vote_alternatives(vote, event_evidence_ids)
        for ballot in ballot_rows:
            person_number = _text(ballot["person_number"])
            if not person_number:
                missingness.append(
                    {
                        "code": "BALLOT_PERSON_ID_MISSING",
                        "severity": "CONTEXT",
                        "reason": "A retained ballot row has no person number and cannot be exact-joined to an actor.",
                        "vote_id": vote_id,
                        "record_locator": f"SaliDBAanestysEdustaja.AanestysId={vote_id}",
                    }
                )
            raw_response = _clean(ballot["raw_response"])
            local_ballot_evidence = _local_ballot_evidence(vote_id, ballot)
            local_ballot_evidence_id = _clean(local_ballot_evidence.get("evidence_id"))
            if raw_response in {"JAA", "EI", "TYHJA"}:
                action_kind = {"value": "VOTE_CAST", "basis": "BALLOT_REGISTER"}
                ballot_category = "CAST"
            elif raw_response == "POISSA":
                action_kind = {"value": None, "basis": "BALLOT_STATUS_NOT_ACTION"}
                ballot_category = "ABSENT"
            else:
                action_kind = {"value": None, "basis": "UNKNOWN_BALLOT_STATUS"}
                ballot_category = "UNKNOWN"
                missingness.append(
                    {
                        "code": "BALLOT_STATUS_UNDECODABLE",
                        "severity": "CONTEXT",
                        "reason": "A stored ballot response is retained verbatim but is not mapped to a policy action.",
                        "vote_id": vote_id,
                        "record_locator": f"SaliDBAanestysEdustaja.AanestysId={vote_id};EdustajaHenkiloNumero={_text(ballot['person_number'])}",
                        "raw_response": raw_response,
                    }
                )
            ballot_locator = (
                f"SaliDBAanestysEdustaja.AanestysId={vote_id};"
                f"EdustajaHenkiloNumero={person_number}"
            )
            role_intervals = _roles_for_action(
                conn,
                _text(ballot["person_number"]),
                _clean(vote["session_date"]),
                ambiguous_persons=ambiguous_persons,
            )
            action_evidence_ids = _unique(
                [
                    *event_evidence_ids,
                    local_ballot_evidence_id,
                    *role_intervals.get("evidence_ids", []),
                ]
            )
            source_actor = {
                "person_id": person_number,
                "name": " ".join(part for part in [ballot["first_name"], ballot["last_name"]] if part),
                "role": "BALLOT_MEMBER",
                "party": _text(ballot["party"]),
                "identity_basis": "BALLOT_PERSON_NUMBER",
                # The event anchor is context for the package/date.  The
                # normalized ballot row has no retained raw hash of its own.
                "evidence_ids": event_evidence_ids,
            }
            actor = _actor_payload(
                source_actor,
                by_person=by_person,
                by_name=by_name,
                by_actor=by_actor,
                ambiguous_persons=ambiguous_persons,
            )
            if not _actor_matches(actor, selected_actor):
                continue
            action_date = _clean(vote["session_date"])
            intervention = {
                "intervention_id": f"vote:{vote_id}:{source_actor['person_id']}",
                "matter_id": matter_id,
                "object_id": vote_object_ids.get(vote_id),
                "event_id": vote_id,
                "kind": "VOTE",
                "action_kind": action_kind,
                "role": "BALLOT_MEMBER",
                "role_label": "BALLOT_MEMBER",
                "actor": actor,
                "title": vote["title"],
                "source_text": vote["kohta"],
                "action_date": action_date,
                "action_date_basis": "SESSION_DATE" if action_date else None,
                "action_date_evidence_ids": event_evidence_ids if action_date else [],
                "source": {
                    "source_id": "SRC-EDUSKUNTA-SALIDBAANESTYSEDUSTAJA",
                    "metadata_source_id": "SRC-EDUSKUNTA-VOTES",
                    "record_locator": ballot_locator,
                    "event_record_locator": f"vote_events.aanestys_id={vote_id}",
                    "raw_sha256": None,
                    "url": _clean(vote["url"]),
                },
                # Keep event evidence visible as package context, while the
                # separate ballot-source object prevents a false claim that
                # the individual action has its own retained raw hash.
                "evidence_ids": action_evidence_ids,
                # The exact vote-event anchor is retained once in
                # ``matter.vote_sources`` and referenced here by IDs.  Keep
                # the local ballot row and actor-specific role evidence on
                # each action so one 199-member vote does not duplicate the
                # same event source packet 199 times.
                "source_evidence": [local_ballot_evidence, *role_intervals.get("evidence", [])],
                "event_source_evidence_ids": event_evidence_ids,
                "ballot_evidence_ids": [local_ballot_evidence_id] if local_ballot_evidence_id else [],
                "ballot_source_version": {
                    "state": "NORMALIZED_ROW_NO_RAW_HASH",
                    "record_status": "NORMALIZED_LOCAL_RECORD",
                    "source_id": "SRC-EDUSKUNTA-SALIDBAANESTYSEDUSTAJA",
                    "record_locator": ballot_locator,
                    "raw_sha256": None,
                    "evidence_ids": [local_ballot_evidence_id] if local_ballot_evidence_id else [],
                    "limitation": "The local ballots table retains normalized response fields but no per-ballot raw source hash or source-version evidence row.",
                },
                "ballot_status": {
                    "state": "RECORDED" if raw_response else "MISSING",
                    "raw_response": raw_response,
                    "category": ballot_category,
                    "evidence_ids": [local_ballot_evidence_id] if local_ballot_evidence_id else [],
                },
                "raw_response": raw_response,
                "party": _clean(ballot["party"]),
                "package_attribution": {
                    "state": "EXACT_VOTE_MATTER",
                    "matter_id": matter_id,
                    "claim": "The ballot is joined to this exact published vote-event matter identifier.",
                    "evidence_ids": event_evidence_ids,
                },
                "component_attribution": {
                    "state": "UNRESOLVED",
                    "value": None,
                    "limitation": "A package vote does not establish support or opposition to each component phrase.",
                    "evidence_ids": [],
                },
                "alternatives": alternatives,
                "drafting": {"state": "NOT_APPLICABLE", "value": None, "evidence_ids": []},
                "role_intervals": role_intervals,
                "institutional_result": {"state": "VOTE_RECORDED", "evidence_ids": event_evidence_ids},
                "own_group_peer_comparison": (
                    compact_peer_comparison(
                        peer_comparisons[person_number],
                        source=peer_source,
                        minimum_peer_count=DEFAULT_MINIMUM_PEERS,
                    )
                    if person_number in peer_comparisons
                    else None
                ),
                "limitations": [
                    "Raw response is retained; only known ballot responses are marked as VOTE_CAST, while status/unknown values are not policy actions.",
                    "The individual ballot row has no retained raw source hash; event evidence identifies the package/date, not a separately versioned ballot record.",
                    "Committee membership and influence are not inferred from a ballot or role interval.",
                    "Own-group peer comparison is a same-event descriptive baseline, not party discipline or component stance.",
                ],
            }
            interventions.append(intervention)
    return interventions, vote_sources, missingness, group_peer_sources


def _coverage_rows(
    conn: sqlite3.Connection,
    *,
    object_kinds: set[str],
    source_ids: set[str],
    vote_sources: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    coverage: list[dict[str, Any]] = []
    coverage_kinds = {_SOURCE_KIND_TO_COVERAGE.get(kind) for kind in object_kinds}
    coverage_kinds.update(kind for kind in object_kinds if kind in {"LEGISLATIVE_INITIATIVE", "SPEECH"})
    if "VOTE" in object_kinds:
        coverage_kinds.add("VOTE")
    coverage_kinds.discard(None)
    for row in conn.execute("SELECT json FROM source_coverage ORDER BY coverage_id"):
        item = _json_row(row)
        source_id = _clean(item.get("source_id"))
        kind = _clean(item.get("kind"))
        if kind in coverage_kinds:
            coverage.append(
                {
                    "coverage_id": _clean(item.get("coverage_id")),
                    "source_id": source_id,
                    "kind": kind,
                    "state": _clean(item.get("state")),
                    "complete": item.get("complete"),
                    "declared_scope": item.get("coverage_scope") or item.get("years") or item.get("limitations"),
                    "source_record_count": (
                        item["source_record_count"]
                        if "source_record_count" in item
                        else item.get("source_records")
                    ),
                    "object_count": item.get("object_count"),
                    "limitations": list(item.get("limitations") or item.get("scope_residuals") or []),
                }
            )
    coverage.extend(
        {
            "coverage_id": f"local-vote-{item['vote_id']}",
            "source_id": item["source_id"],
            "kind": item["kind"],
            "state": "DECLARED_LOCAL_NORMALIZED_SLICE",
            "complete": item["completeness"]["state"] == "COMPLETE_FOR_DECLARED_VOTE",
            "declared_scope": "Rows currently stored in vote_events for this exact matter.",
            "source_record_count": 1,
            "object_count": 1,
            "limitations": ["Local acquisition coverage is not a claim of all historical votes."],
        }
        for item in vote_sources
    )
    return coverage


def _missingness(
    *,
    objects: Sequence[Mapping[str, Any]],
    vote_sources: Sequence[Mapping[str, Any]],
    interventions: Sequence[Mapping[str, Any]],
    selected_actor: Mapping[str, Any] | None,
    selected_actor_errors: Sequence[Mapping[str, Any]],
    coverage: Sequence[Mapping[str, Any]],
    initial: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    result = [dict(item) for item in initial]
    if not objects and not vote_sources:
        result.append(
            {
                "code": "NO_EXACT_MATTER_RECORD",
                "severity": "BLOCKING",
                "reason": "No official object or vote event carries this exact matter identifier.",
            }
        )
    if selected_actor_errors:
        result.extend(dict(item) for item in selected_actor_errors)
    for obj in objects:
        if not _action_date(obj)[0]:
            result.append(
                {
                    "code": "ACTION_DATE_MISSING",
                    "severity": "CONTEXT",
                    "object_id": obj.get("object_id"),
                    "reason": "The official object has no source action_date; it is not assigned one from publication metadata.",
                }
            )
        date_evidence_ids = _action_date(obj)[2]
        known_evidence_ids = {
            _clean(item.get("evidence_id"))
            for item in _object_evidence_rows(obj)
            if _clean(item.get("evidence_id"))
        }
        unresolved_date_evidence_ids = [
            evidence_id for evidence_id in date_evidence_ids if evidence_id not in known_evidence_ids
        ]
        if unresolved_date_evidence_ids:
            result.append(
                {
                    "code": "ACTION_DATE_EVIDENCE_UNRESOLVED",
                    "severity": "CONTEXT",
                    "object_id": obj.get("object_id"),
                    "evidence_ids": unresolved_date_evidence_ids,
                    "reason": "The action-date provenance names evidence IDs that are not embedded in this object source packet.",
                }
            )
        for author in _list_mapping(obj.get("authors")):
            if not _clean(author.get("person_id")):
                result.append(
                    {
                        "code": "SOURCE_PERSON_ID_MISSING",
                        "severity": "CONTEXT",
                        "object_id": obj.get("object_id"),
                        "name": author.get("name"),
                        "reason": "The named source actor is retained but cannot be exact-joined to an individual actor row.",
                    }
                )
    if any(
        _text(obj.get("kind")) in {"LEGISLATIVE_INITIATIVE", "WRITTEN_QUESTION"}
        and _list_mapping(obj.get("authors"))
        for obj in objects
    ):
        result.append(
            {
                "code": "DRAFTING_NOT_IDENTIFIED",
                "severity": "CONTEXT",
                "reason": "Author and signature fields do not establish drafting without a source drafting field.",
            }
        )
    if vote_sources:
        result.append(
            {
                "code": "PACKAGE_VOTE_COMPONENT_UNRESOLVED",
                "severity": "CONTEXT",
                "reason": "Exact matter/vote linkage is retained, while component-level support remains unresolved.",
            }
        )
    if coverage and any(item.get("complete") is False for item in coverage):
        result.append(
            {
                "code": "SOURCE_COVERAGE_LIMITED",
                "severity": "CONTEXT",
                "reason": "At least one attached source coverage receipt is incomplete or declared as a local slice.",
            }
        )
    if coverage and any(item.get("complete") is None for item in coverage):
        result.append(
            {
                "code": "SOURCE_COVERAGE_COMPLETENESS_UNKNOWN",
                "severity": "CONTEXT",
                "reason": "At least one attached source coverage receipt does not declare whether its scope is complete.",
            }
        )
    result.extend(
        [
            {
                "code": "COMMITTEE_MEMBERSHIP_NOT_LOADED",
                "severity": "CONTEXT",
                "reason": "The normalized role table has member, district, and parliamentary-group intervals, not committee membership.",
            },
            {
                "code": "INFLUENCE_NOT_INFERRED",
                "severity": "CONTEXT",
                "reason": "Recorded interventions and overlapping roles do not establish private influence or causation.",
            },
        ]
    )
    if selected_actor is not None and not interventions:
        result.append(
            {
                "code": "NO_EXACT_ACTOR_INTERVENTION",
                "severity": "CONTEXT",
                "reason": "No retained intervention on this exact matter matched the selected actor identity.",
            }
        )
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str, str]] = set()
    for item in result:
        key = (
            _text(item.get("code")),
            _text(item.get("object_id")),
            _text(item.get("name")),
            _text(item.get("vote_id")),
            _text(item.get("record_locator")),
        )
        if key not in seen:
            deduped.append(item)
            seen.add(key)
    return deduped


def build_matter_ledger(
    conn: sqlite3.Connection,
    matter_id: str,
    *,
    actor_id: str | None = None,
    person_id: str | None = None,
    actor_name: str | None = None,
) -> dict[str, Any]:
    """Build one exact matter ledger from local normalized tables.

    ``matter_id`` is compared byte-for-byte after surrounding whitespace is
    removed.  No title, topic, or related-matter fallback is attempted.
    """

    matter_id = _text(matter_id)
    if not matter_id:
        raise MatterLedgerError("matter_id is required for an exact ledger query")
    selected_actor, actor_errors = _resolve_actor_filter(
        conn, actor_id=actor_id, person_id=person_id, actor_name=actor_name
    )
    by_person, by_name, by_actor, ambiguous_persons = _normalise_actor_map(conn)
    objects: list[dict[str, Any]] = []
    all_raw_objects: list[dict[str, Any]] = []
    raw_objects: list[dict[str, Any]] = []
    object_missingness: list[dict[str, Any]] = []
    for row in _matter_object_rows(conn, matter_id):
        obj = _resolve_object_evidence(conn, _json_row(row))
        if not official_object_matches_matter(obj, matter_id):
            continue
        objects.append(_object_summary(obj))
        all_raw_objects.append(obj)
        kind = _text(obj.get("kind"))
        if kind not in _OBJECT_KINDS:
            object_missingness.append(
                {
                    "code": "UNSUPPORTED_OFFICIAL_OBJECT_KIND",
                    "severity": "CONTEXT",
                    "object_id": obj.get("object_id"),
                    "kind": kind or None,
                    "reason": "An exact matter row is retained, but its kind is outside this ledger's supported action projection.",
                }
            )
            continue
        raw_objects.append(obj)
    interventions: list[dict[str, Any]] = []
    for obj in raw_objects:
        kind = _text(obj.get("kind"))
        if kind == "VOTE":
            # Vote object metadata is retained in ``objects``.  Individual
            # ballot actions come from the normalized vote_events/ballots
            # source, avoiding duplicate rows for one event.
            continue
        source_actors = _list_mapping(obj.get("authors"))
        if not source_actors:
            object_missingness.append(
                {
                    "code": "NO_SOURCE_ACTOR",
                    "severity": "CONTEXT",
                    "object_id": obj.get("object_id"),
                    "kind": kind,
                    "reason": "The official object is retained, but no source actor field was loaded for an intervention join.",
                }
            )
        for source_actor in source_actors:
            action = _intervention_from_object(
                obj,
                source_actor,
                by_person=by_person,
                by_name=by_name,
                by_actor=by_actor,
                ambiguous_persons=ambiguous_persons,
                conn=conn,
            )
            if _actor_matches(action["actor"], selected_actor):
                interventions.append(action)
    vote_actions, vote_sources, vote_missingness, group_peer_sources = _vote_interventions(
        conn,
        matter_id,
        selected_actor,
        by_person=by_person,
        by_name=by_name,
        by_actor=by_actor,
        ambiguous_persons=ambiguous_persons,
        vote_objects=[obj for obj in raw_objects if _text(obj.get("kind")) == "VOTE"],
    )
    interventions.extend(vote_actions)
    vote_source_ids = {item.get("vote_id") for item in vote_sources}
    for obj in raw_objects:
        if _text(obj.get("kind")) != "VOTE":
            continue
        vote_id = _vote_id_from_object(obj)
        if vote_id is None or vote_id not in vote_source_ids:
            object_missingness.append(
                {
                    "code": "VOTE_EVENT_NOT_LOADED",
                    "severity": "CONTEXT",
                    "object_id": obj.get("object_id"),
                    "reason": "The exact VOTE object is retained, but no matching vote_events row supplied ballot actions.",
                    "record_locator": _object_source_record(obj).get("record_locator"),
                }
            )
    interventions.sort(
        key=lambda item: (
            item.get("action_date") or "9999-99-99",
            item.get("kind") or "",
            item.get("object_id") or "",
            item.get("actor", {}).get("person_id") or item.get("actor", {}).get("name") or "",
        )
    )
    source_ids = {_text(item.get("source", {}).get("source_id")) for item in interventions}
    source_ids.update(_text(obj.get("source", {}).get("source_id")) for obj in objects)
    source_ids.discard("")
    coverage = _coverage_rows(
        conn,
        object_kinds={_text(obj.get("kind")) for obj in objects},
        source_ids=source_ids,
        vote_sources=vote_sources,
    )
    missingness = _missingness(
        objects=all_raw_objects,
        vote_sources=vote_sources,
        interventions=interventions,
        selected_actor=selected_actor,
        selected_actor_errors=actor_errors,
        coverage=coverage,
        initial=[*object_missingness, *vote_missingness],
    )
    actor_rows: dict[tuple[str, str], dict[str, Any]] = {}
    for action in interventions:
        actor = action["actor"]
        key = (_text(actor.get("actor_id")), _text(actor.get("person_id")) or _text(actor.get("name")))
        if not key[0] and not key[1]:
            continue
        row = actor_rows.setdefault(
            key,
            {
                "actor_id": actor.get("actor_id"),
                "person_id": actor.get("person_id"),
                "name": actor.get("name"),
                "identity_states": [],
                "roles": [],
                "intervention_count": 0,
                "parties": [],
                "source_ids": [],
            },
        )
        for field, value in (("identity_states", actor.get("identity_state")), ("parties", actor.get("source_party"))):
            if value and value not in row[field]:
                row[field].append(value)
        row["intervention_count"] += 1
        source_id = _text(action.get("source", {}).get("source_id"))
        if source_id and source_id not in row["source_ids"]:
            row["source_ids"].append(source_id)
        for role in action.get("role_intervals", {}).get("roles", []):
            role_key = (role.get("kind"), role.get("label"), role.get("start_date"), role.get("end_date"))
            if role_key not in {(r.get("kind"), r.get("label"), r.get("start_date"), r.get("end_date")) for r in row["roles"]}:
                row["roles"].append(role)
    return {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "query": {
            "mode": "EXACT_MATTER",
            "matter_id": matter_id,
            "actor": selected_actor,
            "join": "official_objects.matter_id = vote_events.matter = query.matter_id where applicable",
            "lexical_candidates_included": False,
        },
        "matter": {
            "matter_id": matter_id,
            "objects": objects,
            "vote_sources": vote_sources,
            "object_count": len(objects),
            "vote_event_count": len(vote_sources),
            **matter_subject_title_from_objects(objects),
        },
        "group_peer_sources": group_peer_sources,
        "actors": sorted(actor_rows.values(), key=lambda item: (item.get("name") or "", item.get("actor_id") or "")),
        "interventions": interventions,
        "coverage": coverage,
        "missingness": missingness,
        "limits": [
            "Exact matter linkage is not semantic issue equivalence.",
            "First signature/cosignature is not drafting attribution.",
            "Chair-title naming attributes a published alternative, not private drafting.",
            "Vote-stage prevalence from published totals is not final adoption.",
            "Package vote linkage is not component-level policy support.",
            "Role overlap is not committee influence, causal contribution, implementation, or effect.",
            "Coverage describes loaded declared slices and does not establish population completeness.",
            "Own-group peer comparison uses same-event ballot group codes only; it is not discipline, motive, or influence.",
        ],
    }


def search_matter_candidates(
    conn: sqlite3.Connection,
    query: str,
    *,
    limit: int = 50,
) -> dict[str, Any]:
    """Return lexical matter candidates without constructing interventions."""

    query = _text(query)
    if not query:
        raise MatterLedgerError("search query is required")
    if limit < 1 or limit > 500:
        raise MatterLedgerError("candidate limit must be between 1 and 500")
    terms = [term.casefold() for term in re.split(r"\s+", query) if term]
    candidates: dict[str, dict[str, Any]] = {}

    def add_candidate(
        matter_id: str,
        *,
        kind: str,
        title: str | None,
        text: str | None,
        source_id: str | None,
        action_date: str | None,
        record_locator: str | None,
        url: str | None,
    ) -> None:
        searchable = " ".join([matter_id, title or "", text or ""]).casefold()
        if not all(term in searchable for term in terms):
            return
        row = candidates.setdefault(
            matter_id,
            {
                "matter_id": matter_id,
                "candidate_state": "LEXICAL_CANDIDATE_NOT_EXACT_RELATION",
                "match_basis": "CASEFOLDED_SUBSTRING_ALL_TERMS",
                "records": [],
                "limitations": [
                    "Text overlap proposes a matter for review; no actor intervention or cross-record relation is admitted.",
                    "Candidate results are not a negative finding when absent.",
                ],
            },
        )
        row["records"].append(
            {
                "kind": kind,
                "title": title,
                "text_excerpt": (text or "")[:500],
                "source_id": source_id,
                "action_date": action_date,
                "record_locator": record_locator,
                "url": url,
            }
        )

    for row in conn.execute("SELECT object_id, json FROM official_objects ORDER BY object_id"):
        obj = _json_row(row)
        matter_id = _clean(obj.get("matter_id"))
        if not matter_id or _text(obj.get("kind")) not in _OBJECT_KINDS:
            continue
        date, _basis, _date_evidence = _action_date(obj)
        source = _object_source_record(obj)
        add_candidate(
            matter_id,
            kind=_text(obj.get("kind")),
            title=_clean(obj.get("title")),
            text=_clean(obj.get("text")),
            source_id=source["source_id"],
            action_date=date,
            record_locator=source["record_locator"],
            url=source["url"],
        )
    for row in conn.execute("SELECT * FROM vote_events ORDER BY session_date, aanestys_id"):
        matter_id = _clean(row["matter"])
        if not matter_id:
            continue
        add_candidate(
            matter_id,
            kind="VOTE",
            title=_clean(row["title"]),
            text=_clean(row["kohta"]),
            source_id="SRC-EDUSKUNTA-VOTES",
            action_date=_clean(row["session_date"]),
            record_locator=f"vote_events.aanestys_id={row['aanestys_id']}",
            url=_clean(row["url"]),
        )
    ordered = sorted(candidates.values(), key=lambda item: item["matter_id"])
    return {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "query": {
            "mode": "LEXICAL_CANDIDATE",
            "text": query,
            "normalization": "casefolded whitespace-separated substring terms",
            "exact_join_performed": False,
        },
        "candidates": ordered[:limit],
        "coverage": {
            "state": "LOCAL_NORMALIZED_SLICE",
            "candidate_matter_count_before_limit": len(ordered),
            "returned_count": min(limit, len(ordered)),
            "complete": False,
            "limitations": ["Candidate search covers loaded official_objects and vote_events only."],
        },
        "missingness": [
            {
                "code": "LEXICAL_SEARCH_NOT_EXACT_RELATION",
                "severity": "CONTEXT",
                "reason": "Run an exact matter query before treating a record as an intervention on a matter.",
            }
        ],
    }


def render_matter_html(packet: Mapping[str, Any], *, packet_url: str | None = None) -> str:
    """Delegate the public HTML hub to the maintained site renderer."""

    from paa.matter_site import render_matter_html as render_site_matter_html

    return render_site_matter_html(packet, packet_url=packet_url)


def write_matter_packet(
    packet: Mapping[str, Any],
    *,
    output: Path | None = None,
    html_output: Path | None = None,
) -> dict[str, Any] | None:
    """Write optional JSON and a sharded static HTML site; never mutate the DB."""

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(packet, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if html_output is None:
        return None
    from paa.matter_site import write_matter_site

    packet_url = output.name if output is not None else None
    if output is not None and output.parent.resolve() != html_output.parent.resolve():
        # Hub beside a differently placed JSON: keep a relative download when possible.
        try:
            packet_url = str(output.resolve().relative_to(html_output.parent.resolve()))
        except ValueError:
            packet_url = output.name
    return write_matter_site(packet, html_output, packet_url=packet_url)


def matter_cli(
    *,
    db_path: Path | None = None,
    matter_id: str | None = None,
    search: str | None = None,
    actor_id: str | None = None,
    person_id: str | None = None,
    actor_name: str | None = None,
    limit: int = 50,
    output: Path | None = None,
    html_output: Path | None = None,
) -> dict[str, Any]:
    """Build one CLI packet from a read-only database connection."""

    if bool(matter_id) == bool(search):
        raise MatterLedgerError("provide exactly one of matter_id or search")
    if search and any((actor_id, person_id, actor_name)):
        raise MatterLedgerError("actor filters are available only for an exact matter query")
    path = db_path or DB_PATH
    if not path.exists():
        raise MatterLedgerError(f"database does not exist: {path}")
    conn = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        packet = (
            build_matter_ledger(
                conn,
                matter_id or "",
                actor_id=actor_id,
                person_id=person_id,
                actor_name=actor_name,
            )
            if matter_id
            else search_matter_candidates(conn, search or "", limit=limit)
        )
    finally:
        conn.close()
    write_matter_packet(packet, output=output, html_output=html_output)
    return packet


def project_retained_matter_packet_for_current_emit(
    packet: Mapping[str, Any],
    *,
    retained_bytes_sha256: str | None = None,
) -> dict[str, Any]:
    """Emit a current public packet from a retained snapshot without rewriting the archive.

    Legacy ``INITIATIVE_AUTHORED`` first-signature rows decode to
    ``FIRST_SIGNATORY``. Drafting stays unresolved. The retained schema/hash
    basis is recorded; the original artifact is not mutated.
    """

    out = json.loads(json.dumps(packet, ensure_ascii=False))
    retained_schema = out.get("schema_version")
    out["schema_version"] = LEDGER_SCHEMA_VERSION
    out["retained_source_codec"] = {
        "source_schema_version": retained_schema,
        "source_sha256": retained_bytes_sha256,
        "note": (
            "Current emit projects maintained action_kind labels; retained "
            "snapshot bytes/hashes stay the archival basis."
        ),
    }
    for intervention in out.get("interventions") or []:
        if not isinstance(intervention, dict):
            continue
        role_label = intervention.get("role_label")
        role = intervention.get("role")
        if isinstance(role, Mapping):
            role_label = role_label or role.get("label")
        decoded = decode_ledger_action_kind(
            intervention.get("action_kind"),
            role_label=str(role_label) if role_label else None,
            signature_value=str(role_label) if role_label else None,
            kind=str(intervention.get("kind") or "") or None,
        )
        intervention["action_kind"] = decoded
        if not isinstance(intervention.get("drafting"), Mapping):
            intervention["drafting"] = {
                "state": "UNRESOLVED",
                "value": None,
                "limitation": "Signature/role evidence is not drafting authorship.",
            }
    return out


__all__ = [
    "LEDGER_SCHEMA_VERSION",
    "MatterLedgerError",
    "build_matter_ledger",
    "decode_ledger_action_kind",
    "matter_cli",
    "matter_subject_title_from_objects",
    "official_object_matches_matter",
    "project_retained_matter_packet_for_current_emit",
    "render_matter_html",
    "search_matter_candidates",
    "write_matter_packet",
]

"""Read-only MeV source adapter for the PAA-02 discovery atlas.

The adapter deliberately stops at source-grounded case packets.  It does not
classify warnings, infer authorship, admit detector output, or decide whether
a legal change was implemented.  Case specifications contain generic question
contracts and source references; the SQLite index supplies the complete stored
text and public provenance.  Any later semantic or causal review is a separate
record with its own status and evidence.

The institutional index is opened with SQLite ``mode=ro`` and ``query_only``.
No writes, schema changes, detector reads, or network calls are performed.

CLI examples::

    python -m paa.mev_cases snapshot \
      --index-db /path/to/legislative_index.sqlite \
      --specs paa/contracts/fixtures/mev_discovery_specs.jsonl \
      --output paa/contracts/fixtures/mev_cases_source_slices.jsonl

    python -m paa.mev_cases reviews \
      --index-db /path/to/legislative_index.sqlite \
      --specs paa/contracts/fixtures/mev_warning_review_specs.jsonl \
      --output paa/contracts/fixtures/mev_case_reviews.jsonl
"""


import argparse
import hashlib
import html
import json
import re
import sqlite3
import urllib.parse
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "MEV-CASE-0.1"
REVIEW_SCHEMA_VERSION = "MEV-REVIEW-0.1"
PROPOSED = "PROPOSED"
RESEARCH_ONLY = "RESEARCH_ONLY"

WARNING_CONTROLS = frozenset(
    {"REAL_REPAIR", "REASONED_REBUTTAL", "APPARENT_GAP", "FALSE_GAP"}
)

_TABLES: dict[str, dict[str, Any]] = {
    "he": {
        "key": "canonical_id",
        "kind": "GOVERNMENT_PROPOSAL",
        "text": "content",
        "id_field": "canonical_id",
    },
    "expert_statement": {
        "key": "statement_id",
        "kind": "EXPERT_STATEMENT",
        "text": "content",
        "id_field": "statement_id",
    },
    "committee_report": {
        "key": "report_id",
        "kind": "COMMITTEE_REPORT",
        "text": "content",
        "id_field": "report_id",
    },
    "parliamentary_debate": {
        "key": "debate_id",
        "kind": "PARLIAMENTARY_DEBATE",
        "text": "content",
        "id_field": "debate_id",
    },
}

_LEGACY_KEYS = frozenset(
    {
        "detector_labels",
        "legacy_labels",
        "legacy_status",
        "classifier_labels",
        "gold_label",
        "selection_stratum",
        "sampling_category",
    }
)


class MevCaseError(ValueError):
    """Raised when a source packet or review cannot satisfy its contract."""


def sha256_text(text: str) -> str:
    """Return the UTF-8 SHA-256 for an exact stored string."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_text(raw: str) -> str:
    """Make a deterministic human-readable text view without changing raw text.

    The index stores proposal and Lakitutka/committee content as HTML-ish
    fields.  Tags are removed only for the canonical display/hash view; the
    original field remains in ``raw_text`` and its own hash.
    """

    value = str(raw or "")
    value = re.sub(r"(?is)<\s*(?:br|/p|/div|/li|/h[1-6]|/tr)\s*/?\s*>", "\n", value)
    value = re.sub(r"(?is)<[^>]*>", "", value)
    value = html.unescape(value)
    lines = [" ".join(line.split()) for line in value.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def _public_finlex_url(year: int, number: int) -> str:
    return (
        "https://opendata.finlex.fi/finlex/avoindata/v1/akn/fi/doc/"
        f"government-proposal/{year}/{number}/fin@"
    )


def _public_lakitutka_url(doc_id: str) -> str:
    encoded = urllib.parse.quote(str(doc_id), safe="")
    return f"https://lakitutka.fi/api/docs/laki_vk/{encoded}?lang=fi"


def _public_vaski_url(tunnus: str) -> str:
    encoded = urllib.parse.quote(str(tunnus), safe="")
    return (
        "https://avoindata.eduskunta.fi/api/v1/tables/VaskiData/rows"
        f"?columnName=Eduskuntatunnus&columnValue={encoded}&perPage=5"
    )


def read_only_connection(path: Path | str) -> sqlite3.Connection:
    """Open one SQLite database read-only and reject accidental writes."""

    db_path = Path(path)
    if not db_path.exists():
        raise MevCaseError(f"source database does not exist: {db_path}")
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    return connection


def _source_url(table: str, row: Mapping[str, Any]) -> tuple[str | None, str]:
    if table == "he":
        year = row.get("year")
        number = row.get("number")
        if year is not None and number is not None:
            return _public_finlex_url(int(year), int(number)), "PUBLIC_AKN_API"
    lakitutka_id = row.get("lakitutka_id")
    if lakitutka_id:
        return _public_lakitutka_url(str(lakitutka_id)), "PUBLIC_LAKITUTKA_API"
    tunnus = row.get("tunnus") or row.get("eduskunta_tunnus")
    if tunnus:
        return _public_vaski_url(str(tunnus)), "PUBLIC_VASKI_API_QUERY"
    return None, "PUBLIC_URL_NOT_CAPTURED"


def _clean_optional(value: Any) -> str | None:
    """Return a source-field string, preserving ``None`` for absent facts."""

    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _committee_code(
    table: str,
    document_identifier: str | None,
    expert_title: str | None,
) -> str | None:
    """Extract a code only when it is written in the source metadata itself.

    Committee reports carry the formal code in ``tunnus`` (for example
    ``StVM 18/2020 vp``).  Expert records do not have a separate code column,
    but their stored expert title includes the committee abbreviation after
    the HE identifier (for example ``HE 4/2020 vp StV ...``).  We retain only
    that literal substring and never map a committee name to a code.
    """

    if table == "committee_report" and document_identifier:
        match = re.match(r"^([A-Za-zÅÄÖåäö]+)\s+\d+/\d+\s+vp(?:\s|$)", document_identifier)
        if match:
            return match.group(1)
    if table == "expert_statement" and expert_title:
        match = re.search(
            r"\bvp\s+([A-Za-zÅÄÖåäö]+)\s+\d{1,2}\.\d{1,2}\.\d{4}\b",
            expert_title,
        )
        if match:
            return match.group(1)
    return None


def _query_source(
    connection: sqlite3.Connection,
    table: str,
    record_id: str,
) -> sqlite3.Row:
    metadata = _TABLES.get(table)
    if metadata is None:
        raise MevCaseError(f"unsupported source table: {table}")
    key = metadata["key"]
    if table == "committee_report":
        # Permit either the stable report_id or the human report number.  No
        # committee/name inference is performed.
        row = connection.execute(
            "SELECT * FROM committee_report WHERE report_id = ? OR tunnus = ? LIMIT 1",
            (record_id, record_id),
        ).fetchone()
    else:
        row = connection.execute(
            f'SELECT * FROM "{table}" WHERE "{key}" = ? LIMIT 1', (record_id,)
        ).fetchone()
        if row is None and table == "committee_report":
            row = connection.execute(
                "SELECT * FROM committee_report WHERE tunnus = ? LIMIT 1", (record_id,)
            ).fetchone()
    if row is None:
        raise MevCaseError(f"source row not found: {table}:{record_id}")
    raw = row[metadata["text"]]
    if not raw:
        raise MevCaseError(f"source row has no full text: {table}:{record_id}")
    return row


def source_record(
    connection: sqlite3.Connection,
    table: str,
    record_id: str,
) -> dict[str, Any]:
    """Load one complete index record with raw/canonical hashes and URL."""

    metadata = _TABLES.get(table)
    if metadata is None:
        raise MevCaseError(f"unsupported source table: {table}")
    row = _query_source(connection, table, record_id)
    raw_text = str(row[metadata["text"]])
    text = canonical_text(raw_text)
    if not text:
        raise MevCaseError(f"source row canonicalizes to empty text: {table}:{record_id}")
    row_dict = dict(row)
    decision_text_value = row_dict.get("decision_text")
    decision_text = str(decision_text_value) if decision_text_value else None
    public_url, url_kind = _source_url(table, row_dict)
    stable_id = f"MEV-LEGISLATIVE:{table}:{row_dict[metadata['id_field']]}"
    he_id = row_dict.get("canonical_id") if table == "he" else row_dict.get("he_id")
    matter_title = _clean_optional(
        row_dict.get("title") or row_dict.get("session_title")
    )
    expert_title = _clean_optional(row_dict.get("expert_title"))
    committee_name = _clean_optional(row_dict.get("committee"))
    report_type = _clean_optional(row_dict.get("report_type"))
    if table == "he":
        document_identifier = _clean_optional(row_dict.get("eduskunta_tunnus"))
        document_identifier_kind = "EDUSKUNTA_TUNNUS" if document_identifier else None
    elif table == "committee_report":
        document_identifier = _clean_optional(row_dict.get("tunnus"))
        document_identifier_kind = "COMMITTEE_TUNNUS" if document_identifier else None
    elif table == "expert_statement":
        document_identifier = _clean_optional(row_dict.get("lakitutka_id")) or _clean_optional(
            row_dict.get("statement_id")
        )
        document_identifier_kind = "LAKITUTKA_ID" if row_dict.get("lakitutka_id") else "STATEMENT_ID"
    else:
        document_identifier = _clean_optional(row_dict.get(metadata["id_field"]))
        document_identifier_kind = "RECORD_ID" if document_identifier else None
    committee_code = _committee_code(table, document_identifier, expert_title)
    # A committee row's stored ``title`` is the HE matter title, not the
    # committee document's identity.  Use its formal identifier as the
    # display title and preserve the matter title separately.  Expert titles
    # are already the source's own title and identify the submitting expert.
    if table == "committee_report":
        display_title = document_identifier or matter_title
    elif table == "expert_statement":
        display_title = expert_title or document_identifier or matter_title
    else:
        display_title = matter_title or document_identifier
    return {
        "source_id": stable_id,
        "source_kind": metadata["kind"],
        "source_table": table,
        "record_id": str(row_dict[metadata["id_field"]]),
        "he_id": str(he_id) if he_id else None,
        "title": display_title,
        "matter_title": matter_title,
        "document_identifier": document_identifier,
        "document_identifier_kind": document_identifier_kind,
        "committee_name": committee_name,
        "committee_code": committee_code,
        "report_type": report_type,
        "expert_title": expert_title,
        "publisher": row_dict.get("committee") or row_dict.get("expert_name"),
        "event_date": row_dict.get("date") or row_dict.get("date_issued"),
        "source_url": public_url,
        "source_url_kind": url_kind,
        "record_locator": f"legislative_index.sqlite:{table}:{row_dict[metadata['id_field']]}",
        "coverage_basis": "LOCAL_READ_ONLY_INSTITUTIONAL_INDEX",
        "rights_state": "PUBLIC_SOURCE_REFERENCE",
        "content_format": "HTML_OR_AKN_FIELD",
        "raw_text": raw_text,
        "raw_sha256": sha256_text(raw_text),
        "text": text,
        "text_sha256": sha256_text(text),
        "decision_text": decision_text,
        "decision_text_sha256": sha256_text(decision_text) if decision_text else None,
    }


def _source_refs(spec: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    refs = spec.get("source_refs")
    if not isinstance(refs, Sequence) or isinstance(refs, (str, bytes)) or not refs:
        raise MevCaseError("case requires a non-empty source_refs list")
    result: list[Mapping[str, Any]] = []
    for ref in refs:
        if not isinstance(ref, Mapping) or not ref.get("table") or not ref.get("record_id"):
            raise MevCaseError("source_refs require table and record_id")
        result.append(ref)
    return result


def validate_question_contract(question: Mapping[str, Any]) -> None:
    required = {
        "question_id",
        "text",
        "target_scope",
        "period",
        "comparison",
        "evidence_needed",
        "valid_outputs",
        "unknowns",
        "practical_use",
    }
    missing = sorted(required - set(question))
    if missing:
        raise MevCaseError(f"question contract missing fields: {', '.join(missing)}")
    if not isinstance(question["valid_outputs"], list) or not question["valid_outputs"]:
        raise MevCaseError("question.valid_outputs must be a non-empty list")
    if not isinstance(question["unknowns"], list):
        raise MevCaseError("question.unknowns must be a list")
    if not isinstance(question["evidence_needed"], list) or not question["evidence_needed"]:
        raise MevCaseError("question.evidence_needed must be a non-empty list")
    if not isinstance(question["text"], str) or not question["text"].strip():
        raise MevCaseError("question.text must be non-empty")


def build_case(connection: sqlite3.Connection, spec: Mapping[str, Any]) -> dict[str, Any]:
    """Build a source-grounded, still-``PROPOSED`` case packet."""

    unknown_legacy = sorted(_LEGACY_KEYS.intersection(spec))
    if unknown_legacy:
        raise MevCaseError(
            "legacy detector/selection fields are not admitted: " + ", ".join(unknown_legacy)
        )
    episode_id = str(spec.get("episode_id") or "")
    if not episode_id:
        raise MevCaseError("case requires episode_id")
    episode_kind = str(spec.get("episode_kind") or "")
    if not episode_kind:
        raise MevCaseError("case requires episode_kind")
    question = spec.get("question_contract")
    if not isinstance(question, Mapping):
        raise MevCaseError("case requires question_contract")
    validate_question_contract(question)
    sources = [
        source_record(connection, str(ref["table"]), str(ref["record_id"]))
        for ref in _source_refs(spec)
    ]
    source_by_ref = {
        (source["source_table"], source["record_id"]): source for source in sources
    }
    transformations: list[dict[str, Any]] = []
    for item in spec.get("transformations") or []:
        if not isinstance(item, Mapping):
            raise MevCaseError("transformations must contain objects")
        before = item.get("before")
        after = item.get("after")
        if not isinstance(before, Mapping) or not isinstance(after, Mapping):
            raise MevCaseError("transformation requires before and after source refs")
        before_key = (str(before.get("table")), str(before.get("record_id")))
        after_key = (str(after.get("table")), str(after.get("record_id")))
        if before_key not in source_by_ref or after_key not in source_by_ref:
            raise MevCaseError("transformation refs must be present in source_refs")
        before_source = source_by_ref[before_key]
        after_source = source_by_ref[after_key]
        before_quote = item.get("before_quote")
        after_quote = item.get("after_quote")
        before_field = str(item.get("before_field") or "text")
        after_field = str(item.get("after_field") or "text")
        if bool(before_quote) != bool(after_quote):
            raise MevCaseError("transformation before_quote and after_quote must be supplied together")
        transformation: dict[str, Any] = {
            "transformation_id": str(
                item.get("transformation_id")
                or f"{episode_id}:transformation-{len(transformations) + 1}"
            ),
            "kind": str(item.get("kind") or "DOCUMENTARY_VERSION_CHANGE"),
            "status": PROPOSED,
            "before_source_id": before_source["source_id"],
            "after_source_id": after_source["source_id"],
            "scope": str(item.get("scope") or "SOURCE_TEXT_COMPARISON"),
            "unknowns": list(item.get("unknowns") or []),
        }
        if before_quote and after_quote:
            _quote_check(before_source, str(before_quote), field=before_field)
            _quote_check(after_source, str(after_quote), field=after_field)
            transformation["before_evidence"] = {
                "source_id": before_source["source_id"],
                "field": before_field,
                "quote": str(before_quote),
                "quote_sha256": sha256_text(str(before_quote)),
            }
            transformation["after_evidence"] = {
                "source_id": after_source["source_id"],
                "field": after_field,
                "quote": str(after_quote),
                "quote_sha256": sha256_text(str(after_quote)),
            }
        transformations.append(transformation)
    return {
        "schema_version": SCHEMA_VERSION,
        "fixture_kind": "MEV_DISCOVERY_CASE",
        "episode_id": episode_id,
        "episode_kind": episode_kind,
        "admission_state": PROPOSED,
        "semantic_state": "SOURCE_PACKET_ONLY",
        "question_contract": dict(question),
        "source_ids": [source["source_id"] for source in sources],
        "sources": sources,
        "transformations": transformations,
        "coverage": {
            "source_count": len(sources),
            "source_tables": sorted({source["source_table"] for source in sources}),
            "full_text_present": all(bool(source["raw_text"]) for source in sources),
            "legacy_detector_labels_admitted": False,
            "causal_effect_assessed": False,
        },
        "review_state": "NO_INDEPENDENT_REVIEW_IN_CASE_PACKET",
    }


def _quote_check(source: Mapping[str, Any], quote: str, *, field: str = "text") -> None:
    if not isinstance(quote, str) or not quote.strip():
        raise MevCaseError("review quote must be non-empty")
    if field not in {"text", "decision_text"}:
        raise MevCaseError(f"unsupported quote field: {field}")
    if quote not in str(source.get(field) or ""):
        raise MevCaseError(
            f"review quote is not an exact substring of {source.get('source_id')} {field}"
        )


def build_review(
    connection: sqlite3.Connection,
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    """Build an AI source-reading review; never an admitted detector label."""

    control = str(spec.get("warning_control") or "")
    if control not in WARNING_CONTROLS:
        raise MevCaseError(f"warning_control must be one of {sorted(WARNING_CONTROLS)}")
    review_id = str(spec.get("review_id") or "")
    episode_id = str(spec.get("episode_id") or "")
    rationale = str(spec.get("rationale") or "").strip()
    if not review_id or not episode_id or not rationale:
        raise MevCaseError("review requires review_id, episode_id and rationale")
    refs = _source_refs(spec)
    sources = [
        source_record(connection, str(ref["table"]), str(ref["record_id"])) for ref in refs
    ]
    source_by_ref = {(s["source_table"], s["record_id"]): s for s in sources}
    quote_items = spec.get("quotes")
    if not isinstance(quote_items, Sequence) or isinstance(quote_items, (str, bytes)) or not quote_items:
        raise MevCaseError("review requires quotes")
    quotes: list[dict[str, Any]] = []
    for item in quote_items:
        if not isinstance(item, Mapping):
            raise MevCaseError("review quotes must contain objects")
        key = (str(item.get("table")), str(item.get("record_id")))
        source = source_by_ref.get(key)
        if source is None:
            raise MevCaseError(f"review quote source is not in source_refs: {key}")
        quote = str(item.get("quote") or "")
        field = str(item.get("field") or "text")
        _quote_check(source, quote, field=field)
        quotes.append(
            {
                "source_id": source["source_id"],
                "source_table": source["source_table"],
                "record_id": source["record_id"],
                "field": field,
                "quote": quote,
                "quote_sha256": sha256_text(quote),
            }
        )
    return {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "fixture_kind": "MEV_WARNING_RESPONSE_REVIEW",
        "review_id": review_id,
        "episode_id": episode_id,
        "warning_control": control,
        "review_state": RESEARCH_ONLY,
        "reviewer": "AI_SOURCE_READING",
        "review_method": "AI_SOURCE_READING",
        "review_provenance": "complete-cited-source-text-read; no template label assignment",
        "label_basis": "actual meaning after reading the complete cited source texts",
        "template_label_assignment": False,
        "rationale": rationale,
        "source_ids": [source["source_id"] for source in sources],
        "source_text_sha256": [source["text_sha256"] for source in sources],
        "quotes": quotes,
        "evidence_boundary": "documentary_response_only; no authorship, motive, implementation, or causal claim",
    }


def read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    # JSON permits U+2028/U+2029 inside a string.  ``str.splitlines`` treats
    # both as physical line boundaries; JSONL records are delimited by LF.
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").split("\n"), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise MevCaseError(f"invalid JSONL at {path}:{line_no}: {exc}") from exc
        if not isinstance(value, dict):
            raise MevCaseError(f"JSONL row is not an object at {path}:{line_no}")
        rows.append(value)
    return rows


def write_jsonl(path: Path | str, rows: Iterable[Mapping[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    destination.write_text(text, encoding="utf-8")


def build_snapshot(index_db: Path | str, specs_path: Path | str) -> list[dict[str, Any]]:
    connection = read_only_connection(index_db)
    try:
        return [build_case(connection, spec) for spec in read_jsonl(specs_path)]
    finally:
        connection.close()


def build_reviews(index_db: Path | str, specs_path: Path | str) -> list[dict[str, Any]]:
    connection = read_only_connection(index_db)
    try:
        return [build_review(connection, spec) for spec in read_jsonl(specs_path)]
    finally:
        connection.close()


def _cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("snapshot", "reviews"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--index-db", type=Path, required=True)
        sub.add_argument("--specs", type=Path, required=True)
        sub.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = build_snapshot(args.index_db, args.specs) if args.command == "snapshot" else build_reviews(args.index_db, args.specs)
    write_jsonl(args.output, rows)
    print(json.dumps({"command": args.command, "rows": len(rows), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())

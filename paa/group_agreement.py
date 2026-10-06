"""Source-bound group-majority agreement calculations.

This module is deliberately a small, read-only analytical layer over the
Eduskunta ballot rows already reconciled by :mod:`paa.acquire_eduskunta`.
It answers one descriptive question only:

    did a member's recorded JAA/EI ballot match the strict majority of the
    other substantive JAA/EI ballots from the same parliamentary group at
    that event?

The result is *not* a measure of obedience, independent thought, influence,
competence or policy agreement.  ``ballots.party`` is the event-time
``EdustajaRyhmaLyhenne`` value; current candidacy parties and present-day
identity tables are intentionally not consulted.

``build_group_context`` returns a compact global source collection (one
normalised ballot snapshot per vote) and one person summary per member.  A
summary references the shared source snapshots by ID; it never embeds the
same ballot rows again.  The source hash is over the normalised snapshot,
not a claim about an unavailable raw HTTP payload.  ``validate_group_packet``
recomputes source reconciliation and every emitted comparison from those
shared inputs before accepting a summary.
"""


import hashlib
import json
import sqlite3
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import date
from typing import Any

from paa.config import CORPUS_CUTOFF

SCHEMA_VERSION = "paa.group_agreement.v1"
NORMALIZATION_VERSION = "eduskunta-ballot-group-v1"
SOURCE_KIND = "EDUSKUNTA_BALLOT_ROWS"
DEFAULT_MINIMUM_PEERS = 2

SUBSTANTIVE = frozenset({"JAA", "EI"})
RESPONSE_BUCKETS = ("JAA", "EI", "TYHJA", "POISSA")

# These are the compact group codes actually used by the current Eduskunta
# ballot endpoint, plus established historical spellings.  An unrecognised
# code is not silently promoted to a group: it remains UNKNOWN and cannot
# enter a peer denominator.  ``erk`` means a group-less member in this
# source, not a party.
KNOWN_GROUP_CODES = frozenset(
    {
        "kok",
        "ps",
        "sd",
        "kesk",
        "vihr",
        "vas",
        "r",
        "kd",
        "liik",
        "tv",
        # Common historical compact forms, retained for reusable fixtures.
        "sfp",
        "rkp",
        "ml",
        "smp",
        "skdl",
        "lib",
        "sin",
        "sit",
        "uv",
        "vkk",
        "ed",
    }
)
GROUPLESS_CODES = frozenset({"", "erk", "group-less", "groupless", "none", "null", "n/a", "na"})


class GroupAgreementError(ValueError):
    """Raised for malformed group-agreement source or packet data."""


def _value(row: Mapping[str, Any] | sqlite3.Row, key: str, default: Any = None) -> Any:
    try:
        return row[key]
    except (KeyError, IndexError):
        return default


def _text(value: Any) -> str:
    return str(value or "").strip()


def _normalise_group(value: Any) -> str:
    return _text(value).casefold()


def _group_state(code: str) -> str:
    if code in GROUPLESS_CODES:
        return "GROUPLESS_EXCLUDED"
    if code in KNOWN_GROUP_CODES:
        return "KNOWN"
    return "UNKNOWN"


def _normalise_response(value: Any) -> str:
    # The acquisition adapter already emits these four forms.  The small
    # accent/whitespace handling keeps the source adapter honest for fixtures
    # copied from the Finnish endpoint without accepting a guessed vote.
    value = _text(value).upper().replace("Ä", "A")
    if value in RESPONSE_BUCKETS:
        return value
    if not value:
        return ""
    return "UNKNOWN"


def _as_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().casefold() in {"1", "true", "yes", "kyllä", "mitatoity"}
    return bool(value)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _parse_iso_day(value: Any) -> date | None:
    raw = _text(value)
    if not raw:
        return None
    try:
        return date.fromisoformat(raw[:10])
    except ValueError:
        return None


def _source_id(vote_id: str) -> str:
    return f"eduskunta:ballots:{vote_id}"


def _row_as_dict(row: Mapping[str, Any] | sqlite3.Row) -> dict[str, Any]:
    person_id = _text(_value(row, "person_number"))
    return {
        "person_id": person_id,
        "group_code": _normalise_group(_value(row, "party")),
        "response": _normalise_response(_value(row, "raw_response")),
        "first_name": _text(_value(row, "first_name")),
        "last_name": _text(_value(row, "last_name")),
    }


def _published_totals(vote: Mapping[str, Any] | sqlite3.Row) -> dict[str, int | None]:
    return {
        "JAA": _as_int(_value(vote, "jaa")),
        "EI": _as_int(_value(vote, "ei")),
        "TYHJA": _as_int(_value(vote, "tyhjaa")),
        "POISSA": _as_int(_value(vote, "poissa")),
        "TOTAL": _as_int(_value(vote, "yhteensa")),
    }


def _observed_totals(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = Counter(str(row.get("response") or "") for row in rows)
    return {bucket: int(counts.get(bucket, 0)) for bucket in RESPONSE_BUCKETS} | {
        "UNKNOWN": int(counts.get("UNKNOWN", 0)),
        "BLANK": int(counts.get("", 0)),
        "TOTAL": len(rows),
    }


def _source_payload(source: Mapping[str, Any]) -> dict[str, Any]:
    """Return the exact normalised fields covered by ``source_sha256``."""

    # Keep provenance and all four published buckets in the signed input.
    # ``source_sha256`` itself is intentionally excluded to avoid recursion.
    return {
        "schema_version": source.get("schema_version", SCHEMA_VERSION),
        "normalization_version": source.get("normalization_version", NORMALIZATION_VERSION),
        "source_kind": source.get("source_kind", SOURCE_KIND),
        "source_id": source.get("source_id"),
        "vote_id": source.get("vote_id"),
        "record_locator": source.get("record_locator"),
        "source_url": source.get("source_url"),
        "source_basis": source.get("source_basis"),
        "group_code_field": source.get("group_code_field"),
        "year": source.get("year"),
        "session_date": source.get("session_date"),
        "number": source.get("number"),
        "title": source.get("title"),
        "matter": source.get("matter"),
        "published_totals": source.get("published_totals"),
        "void": bool(source.get("void")),
        "cutoff": source.get("cutoff"),
        "rows": source.get("rows", []),
    }


def _source_hash(source: Mapping[str, Any]) -> str:
    return _sha256(_source_payload(source))


def _source_diagnostics(source: Mapping[str, Any], *, cutoff: str | None = None) -> dict[str, Any]:
    """Reconcile one normalised source independently of any person packet."""

    rows = source.get("rows")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
        rows = []
    normalized_rows: list[dict[str, Any]] = []
    for raw in rows:
        if not isinstance(raw, Mapping):
            normalized_rows.append({"person_id": "", "group_code": "", "response": "", "first_name": "", "last_name": ""})
            continue
        # A source is already normalised, but re-normalising here prevents a
        # case/whitespace edit from bypassing validation.
        normalized_rows.append(
            {
                "person_id": _text(raw.get("person_id", raw.get("person_number"))),
                "group_code": _normalise_group(raw.get("group_code", raw.get("party"))),
                "response": _normalise_response(raw.get("response", raw.get("raw_response"))),
                "first_name": _text(raw.get("first_name")),
                "last_name": _text(raw.get("last_name")),
            }
        )

    published = source.get("published_totals")
    if not isinstance(published, Mapping):
        published = {}
    expected = {bucket: _as_int(published.get(bucket)) for bucket in RESPONSE_BUCKETS}
    expected_total = _as_int(published.get("TOTAL"))
    observed = _observed_totals(normalized_rows)
    ids = [row["person_id"] for row in normalized_rows]
    duplicates = sorted({person_id for person_id, n in Counter(ids).items() if n > 1 or not person_id})
    structural_reasons: list[str] = []
    if any(expected[bucket] is None for bucket in RESPONSE_BUCKETS) or expected_total is None:
        structural_reasons.append("MISSING_PUBLISHED_TOTAL")
    else:
        if expected_total != sum(expected[bucket] for bucket in RESPONSE_BUCKETS):
            structural_reasons.append("PUBLISHED_TOTAL_BUCKET_SUM_MISMATCH")
        if len(normalized_rows) != expected_total:
            structural_reasons.append("ROW_COUNT_MISMATCH")
        for bucket in RESPONSE_BUCKETS:
            if observed[bucket] != expected[bucket]:
                structural_reasons.append(f"{bucket}_BUCKET_MISMATCH")
    if observed["UNKNOWN"] or observed["BLANK"]:
        structural_reasons.append("UNRECOGNISED_OR_BLANK_RESPONSE")
    if duplicates:
        structural_reasons.append("DUPLICATE_OR_BLANK_PERSON_ID")
    if _text(source.get("vote_id")) != _text(source.get("source_id", "")).removeprefix("eduskunta:ballots:"):
        structural_reasons.append("SOURCE_ID_VOTE_ID_MISMATCH")
    source_day = _parse_iso_day(source.get("session_date"))
    if source.get("session_date") and source_day is None:
        structural_reasons.append("INVALID_SESSION_DATE")

    reasons: list[str] = []
    if _truthy(source.get("void")):
        reasons.append("VOID_VOTE")
    if cutoff and source_day and source_day > (_parse_iso_day(cutoff) or date.max):
        reasons.append("FUTURE_CUTOFF")
    if not source_day:
        reasons.append("MISSING_SESSION_DATE")
    reasons.extend(structural_reasons)
    if "VOID_VOTE" in reasons:
        state = "EXCLUDED_VOID"
    elif "FUTURE_CUTOFF" in reasons:
        state = "EXCLUDED_FUTURE"
    elif reasons:
        state = "EXCLUDED_SOURCE_MISMATCH"
    else:
        state = "VALID"
    published_jaa = expected.get("JAA")
    published_ei = expected.get("EI")
    global_contested = bool(published_jaa is not None and published_ei is not None and published_jaa > 0 and published_ei > 0)
    canonical_published = {bucket: expected[bucket] for bucket in RESPONSE_BUCKETS}
    canonical_published["TOTAL"] = expected_total
    return {
        "state": state,
        "valid": state == "VALID",
        "reasons": sorted(set(reasons)),
        "structural_reasons": sorted(set(structural_reasons)),
        "normalized_rows": normalized_rows,
        "normalized_row_count": len(normalized_rows),
        "unique_person_count": len(set(ids)),
        "duplicate_person_ids": duplicates,
        "observed_totals": observed,
        "published_totals": canonical_published,
        "global_contested": global_contested,
    }


def _make_source(vote: Mapping[str, Any] | sqlite3.Row, rows: Sequence[Mapping[str, Any] | sqlite3.Row], *, cutoff: str) -> dict[str, Any]:
    vote_id = _text(_value(vote, "aanestys_id"))
    normalized_rows = sorted((_row_as_dict(row) for row in rows), key=lambda row: (row["person_id"], row["group_code"], row["response"]))
    source: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "normalization_version": NORMALIZATION_VERSION,
        "source_kind": SOURCE_KIND,
        "source_id": _source_id(vote_id),
        "vote_id": vote_id,
        "record_locator": f"vote_events.aanestys_id={vote_id};ballots.aanestys_id={vote_id}",
        "source_url": _text(_value(vote, "url")) or None,
        "source_basis": "LOCAL_NORMALIZED_DB_ROWS_FROM_OFFICIAL_EDUSKUNTA_BALLOT_REGISTER",
        "group_code_field": "EdustajaRyhmaLyhenne -> ballots.party",
        "year": _as_int(_value(vote, "year")),
        "session_date": _text(_value(vote, "session_date")) or None,
        "number": _as_int(_value(vote, "number")),
        "title": _text(_value(vote, "title")),
        # This is the source's formal matter label verbatim.  No semantic
        # matter clustering is performed here.
        "matter": _text(_value(vote, "matter")) or None,
        "published_totals": _published_totals(vote),
        "rows": normalized_rows,
        "void": _truthy(_value(vote, "mitatoity")),
        "cutoff": cutoff,
    }
    source["source_sha256"] = _source_hash(source)
    diagnostics = _source_diagnostics(source, cutoff=cutoff)
    source.update(
        {
            "state": diagnostics["state"],
            "valid": diagnostics["valid"],
            "exclusion_reasons": diagnostics["reasons"],
            "structural_reasons": diagnostics["structural_reasons"],
            "observed_totals": diagnostics["observed_totals"],
            "row_count": diagnostics["normalized_row_count"],
            "unique_person_count": diagnostics["unique_person_count"],
            "duplicate_person_ids": diagnostics["duplicate_person_ids"],
            "global_contested": diagnostics["global_contested"],
            "hash_basis": "NORMALIZED_SNAPSHOT_NOT_RAW_HTTP_PAYLOAD",
        }
    )
    return source


def _prepare_source(
    source: Mapping[str, Any],
    *,
    diagnostics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Prepare one source once for all target members.

    A source has at most one row per person after reconciliation.  Group
    totals let each target comparison subtract itself in O(1), rather than
    scanning the 199-row vote once for every member.
    """

    diagnostics = diagnostics or _source_diagnostics(source, cutoff=source.get("cutoff"))
    rows = diagnostics["normalized_rows"]
    by_person: dict[str, dict[str, Any]] = {}
    group_totals: dict[str, dict[str, int]] = defaultdict(lambda: {"JAA": 0, "EI": 0})
    for row in rows:
        person_id = row["person_id"]
        # Duplicate IDs make the source invalid.  Keep the first row only for
        # deterministic audit output; no invalid source can become a valid
        # comparison.
        by_person.setdefault(person_id, row)
        if row["response"] in SUBSTANTIVE and _group_state(row["group_code"]) == "KNOWN":
            group_totals[row["group_code"]][row["response"]] += 1
    return {"diagnostics": diagnostics, "rows": rows, "by_person": by_person, "group_totals": dict(group_totals)}


def _compare_source(
    source: Mapping[str, Any],
    person_id: str,
    *,
    minimum_peer_count: int,
    prepared: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute one target's event result from source rows only."""

    prepared = prepared or _prepare_source(source)
    diagnostics = prepared["diagnostics"]
    state = diagnostics["state"]
    target = prepared["by_person"].get(person_id)
    base: dict[str, Any] = {
        "source_ref": source.get("source_id"),
        "vote_id": source.get("vote_id"),
        "session_date": source.get("session_date"),
        "matter": source.get("matter"),
        "contested": bool(diagnostics["global_contested"] and state == "VALID"),
        "target_group_code": target.get("group_code") if target else None,
        "target_group_state": _group_state(target.get("group_code", "")) if target else None,
        "target_response": target.get("response") if target else None,
        "peer_count": 0,
        "peer_jaa": 0,
        "peer_ei": 0,
        "peer_majority": None,
        "matches_peer_majority": None,
        "status": None,
        "reason": None,
    }
    if state != "VALID":
        base["status"] = "SOURCE_EXCLUDED"
        base["reason"] = ";".join(diagnostics["reasons"]) or state
        # Retain a separate target-state marker for audit reports, while the
        # source exclusion remains the denominator gate.
        if target is None:
            base["target_state"] = "ABSENT"
        elif target["response"] == "":
            base["target_state"] = "BLANK"
        elif target["response"] == "UNKNOWN":
            base["target_state"] = "UNKNOWN_RESPONSE"
        else:
            base["target_state"] = "PRESENT"
        return base
    if target is None:
        base["status"] = "TARGET_ABSENT"
        base["reason"] = "target person has no row in this reconciled vote"
        return base
    response = target["response"]
    group_code = target["group_code"]
    group_state = _group_state(group_code)
    if response == "":
        base["status"] = "TARGET_BLANK"
        base["reason"] = "target row has a blank ballot response"
        return base
    if response == "UNKNOWN":
        base["status"] = "TARGET_UNKNOWN_RESPONSE"
        base["reason"] = "target row has an unrecognised ballot response"
        return base
    if response not in SUBSTANTIVE:
        base["status"] = "TARGET_NON_SUBSTANTIVE"
        base["reason"] = "target ballot is abstention or absence, not JAA/EI"
        return base
    if group_state == "GROUPLESS_EXCLUDED":
        base["status"] = "TARGET_GROUPLESS_EXCLUDED"
        base["reason"] = "event group code is group-less (ERK), not a peer group"
        return base
    if group_state == "UNKNOWN":
        base["status"] = "TARGET_UNKNOWN_GROUP"
        base["reason"] = "event group code is not in the declared source group vocabulary"
        return base

    group_totals = prepared["group_totals"].get(group_code, {"JAA": 0, "EI": 0})
    peer_jaa = int(group_totals.get("JAA", 0)) - int(target["response"] == "JAA")
    peer_ei = int(group_totals.get("EI", 0)) - int(target["response"] == "EI")
    peer_count = peer_jaa + peer_ei
    base.update({"peer_count": peer_count, "peer_jaa": peer_jaa, "peer_ei": peer_ei})
    if peer_count < minimum_peer_count:
        base["status"] = "PEER_COUNT_BELOW_MINIMUM"
        base["reason"] = f"fewer than {minimum_peer_count} substantive peers"
        return base
    if peer_jaa == peer_ei:
        base["status"] = "PEER_TIE"
        base["reason"] = "peer JAA/EI counts are tied"
        return base
    majority = "JAA" if peer_jaa > peer_ei else "EI"
    base["peer_majority"] = majority
    base["matches_peer_majority"] = response == majority
    base["status"] = "COMPARABLE"
    base["reason"] = "strict majority among substantive same-group peers"
    return base


def _empty_summary(person_id: str, display_name: str | None, source_refs: list[str]) -> dict[str, Any]:
    return {
        "packet_type": "GROUP_AGREEMENT_PERSON",
        "schema_version": SCHEMA_VERSION,
        "person_id": person_id,
        "display_name": display_name or None,
        "source_refs": list(source_refs),
        "comparison_refs": [],
        "summary": {
            "source_count": len(source_refs),
            "valid_source_count": 0,
            "source_excluded_count": 0,
            "present_count": 0,
            # A missing row is a source-row absence, not a recorded POISSA
            # ballot and not evidence that the member missed the sitting.
            "source_row_missing_count": 0,
            "absent_count": 0,
            "target_blank_count": 0,
            "target_unknown_response_count": 0,
            "group_known_count": 0,
            "group_less_excluded_count": 0,
            "unknown_group_count": 0,
            "recorded_absence_count": 0,
            "recorded_abstention_count": 0,
            "target_abstention_count": 0,
            "target_non_substantive_count": 0,
            "target_substantive_count": 0,
            "peer_eligible_count": 0,
            "peer_undefined_count": 0,
            "comparable_count": 0,
            "matching_count": 0,
            "agreement": None,
            "contested_source_count": 0,
            "contested_comparable_count": 0,
            "contested_matching_count": 0,
            "contested_agreement": None,
            "status_counts": {},
            "reason_counts": {},
            "matter_label_counts": {},
            "matter_cluster_diagnostic": {
                "basis": "EXACT_SOURCE_FORMAL_MATTER_LABEL_ONLY",
                "labels_denominator": 0,
                "comparable_votes_with_label": 0,
                "comparable_votes_without_label": 0,
                "mean_within_label_agreement": None,
                "largest_label": None,
                "largest_label_count": 0,
                "largest_label_matching_count": 0,
                "largest_label_agreement": None,
                "agreement_with_largest_label_removed": None,
                "semantic_policy_clustering": "NOT_PERFORMED",
                "excluded_label_reason": "MISSING_OR_BLANK_EXACT_SOURCE_MATTER_LABEL",
            },
        },
        "comparisons": [],
    }


def _finalise_summary(packet: dict[str, Any]) -> None:
    summary = packet["summary"]
    comparable = int(summary["comparable_count"])
    matching = int(summary["matching_count"])
    contested_comparable = int(summary["contested_comparable_count"])
    contested_matching = int(summary["contested_matching_count"])
    summary["agreement"] = matching / comparable if comparable else None
    summary["contested_agreement"] = contested_matching / contested_comparable if contested_comparable else None
    # Exact source formal matter labels are a sensitivity diagnostic, not a
    # policy ontology.  Restrict this calculation to comparable rows with a
    # non-blank source label; invalid sources and unlabeled rows remain
    # explicit in the denominators below.
    by_label: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    labeled_comparable = 0
    for comparison in packet.get("comparisons", []):
        if comparison.get("status") != "COMPARABLE":
            continue
        label = comparison.get("matter")
        if not isinstance(label, str) or not label.strip():
            continue
        label = label.strip()
        by_label[label][0] += 1
        by_label[label][1] += int(bool(comparison.get("matches_peer_majority")))
        labeled_comparable += 1
    rates = [matching_count / count for count, matching_count in by_label.values() if count]
    largest_label = None
    largest_count = 0
    largest_matching = 0
    if by_label:
        largest_label = min(by_label, key=lambda label: (-by_label[label][0], label))
        largest_count, largest_matching = by_label[largest_label]
    remaining_count = labeled_comparable - largest_count
    remaining_matching = sum(matching_count for label, (_count, matching_count) in by_label.items() if label != largest_label)
    diagnostic = {
        "basis": "EXACT_SOURCE_FORMAL_MATTER_LABEL_ONLY",
        "labels_denominator": len(by_label),
        "comparable_votes_with_label": labeled_comparable,
        "comparable_votes_without_label": comparable - labeled_comparable,
        "mean_within_label_agreement": sum(rates) / len(rates) if rates else None,
        "largest_label": largest_label,
        "largest_label_count": largest_count,
        "largest_label_matching_count": largest_matching,
        "largest_label_agreement": largest_matching / largest_count if largest_count else None,
        "agreement_with_largest_label_removed": remaining_matching / remaining_count if remaining_count else None,
        "semantic_policy_clustering": "NOT_PERFORMED",
        "excluded_label_reason": "MISSING_OR_BLANK_EXACT_SOURCE_MATTER_LABEL",
    }
    summary["matter_cluster_diagnostic"] = diagnostic
    packet["summary"] = summary


def _display_names(conn: sqlite3.Connection, person_ids: set[str]) -> dict[str, str]:
    names: dict[str, str] = {}
    # Names are merely a display convenience.  Group membership and the
    # comparison itself always come from the event ballot row.
    if not person_ids:
        return names
    try:
        for row in conn.execute("SELECT person_id, first_name, last_name FROM mp_people"):
            person_id = _text(row["person_id"])
            if person_id in person_ids:
                names[person_id] = " ".join(part for part in (_text(row["first_name"]), _text(row["last_name"])) if part)
    except sqlite3.OperationalError:
        pass
    return names


def _snapshot_hash(sources: Sequence[Mapping[str, Any]], *, cutoff: str, minimum_peer_count: int) -> str:
    return _sha256(
        {
            "schema_version": SCHEMA_VERSION,
            "normalization_version": NORMALIZATION_VERSION,
            "cutoff": cutoff,
            "minimum_peer_count": minimum_peer_count,
            "source_sha256": [str(source.get("source_sha256")) for source in sorted(sources, key=lambda item: str(item.get("source_id")))],
        }
    )


def _matter_report(sources: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    valid_sources = [source for source in sources if source.get("valid")]
    labels = Counter(
        str(source.get("matter")).strip()
        for source in valid_sources
        if isinstance(source.get("matter"), str) and source.get("matter").strip()
    )
    repeated = [
        {"label": label, "vote_count": count}
        for label, count in sorted(labels.items(), key=lambda item: (str(item[0]), item[1]))
        if count > 1 and label is not None
    ]
    largest_label = None
    largest_count = 0
    if labels:
        largest_label = min(labels, key=lambda label: (-labels[label], str(label)))
        largest_count = int(labels[largest_label])
    valid_count = len(valid_sources)
    return {
        "basis": "EXACT_SOURCE_FORMAL_MATTER_LABEL",
        "distinct_labels": len(labels),
        "missing_label_votes": sum(
            not (isinstance(source.get("matter"), str) and source.get("matter").strip())
            for source in valid_sources
        ),
        "repeated_labels": repeated,
        "largest_label": largest_label,
        "largest_label_count": largest_count,
        "largest_label_share_of_valid_sources": largest_count / valid_count if valid_count else None,
        "policy_clustering": "NOT_PERFORMED",
        "warning": "Repeated procedural rounds remain separate source events; exact formal labels do not establish one policy object.",
    }


def _load_group_sources(
    conn: sqlite3.Connection,
    *,
    cutoff: str,
) -> tuple[list[dict[str, Any]], int, int]:
    """Load and normalise shared vote sources without allocating packets.

    The source collection is deliberately independent from person summaries:
    callers that only need an integrity receipt can rebuild the normalized
    vote snapshots and compare their hashes without constructing every target
    packet.  ``vote_count`` and ``orphan_ballot_rows`` are returned so the
    full context report can retain its existing coverage disclosures.
    """

    try:
        votes = conn.execute("SELECT * FROM vote_events ORDER BY session_date, aanestys_id").fetchall()
        ballot_rows = conn.execute(
            "SELECT aanestys_id, person_number, first_name, last_name, party, raw_response "
            "FROM ballots ORDER BY aanestys_id, person_number"
        ).fetchall()
    except sqlite3.OperationalError as error:
        raise GroupAgreementError(f"group agreement requires vote_events and ballots tables: {error}") from error

    grouped: dict[str, list[sqlite3.Row]] = defaultdict(list)
    orphan_rows = 0
    known_vote_ids = {str(_value(vote, "aanestys_id")) for vote in votes}
    for row in ballot_rows:
        vote_id = _text(_value(row, "aanestys_id"))
        if vote_id not in known_vote_ids:
            orphan_rows += 1
            continue
        grouped[vote_id].append(row)

    source_by_vote: dict[str, dict[str, Any]] = {}
    for vote in votes:
        vote_id = _text(_value(vote, "aanestys_id"))
        rows = grouped.get(vote_id)
        if not rows:
            continue
        source_by_vote[vote_id] = _make_source(vote, rows, cutoff=cutoff)
    sources = [
        source_by_vote[key]
        for key in sorted(
            source_by_vote,
            key=lambda item: (source_by_vote[item].get("session_date") or "", item),
        )
    ]
    return sources, len(votes), orphan_rows


def build_group_sources(
    conn: sqlite3.Connection,
    *,
    cutoff: str = CORPUS_CUTOFF,
) -> list[dict[str, Any]]:
    """Rebuild the normalized, source-only vote collection.

    This is intentionally cheaper and less identity-sensitive than
    :func:`build_group_context`: it returns one reconciled source object per
    vote with stored ballot rows and performs no person-packet allocation.
    Source hashes cover vote metadata, all four published response buckets,
    event-time group codes, and normalized ballot rows.
    """

    sources, _vote_count, _orphan_rows = _load_group_sources(conn, cutoff=cutoff)
    return sources


def build_group_context(
    conn: sqlite3.Connection,
    *,
    cutoff: str = CORPUS_CUTOFF,
    minimum_peer_count: int = DEFAULT_MINIMUM_PEERS,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Build person packets, shared vote sources and an audit report.

    Only votes with at least one stored ballot row are source inputs.  Votes
    without ballots are disclosed as ``votes_without_ballots`` rather than
    treated as universal absences.  No writes are performed on ``conn``.
    """

    if minimum_peer_count < 1:
        raise ValueError("minimum_peer_count must be positive")
    sources, vote_count, orphan_rows = _load_group_sources(conn, cutoff=cutoff)
    source_refs = [str(source["source_id"]) for source in sources]
    prepared_sources = {str(source["source_id"]): _prepare_source(source) for source in sources}
    snapshot_sha256 = _snapshot_hash(sources, cutoff=cutoff, minimum_peer_count=minimum_peer_count)

    person_ids: set[str] = set()
    display_from_ballots: dict[str, str] = {}
    for source in sources:
        for row in source["rows"]:
            person_id = _text(row.get("person_id"))
            if not person_id:
                continue
            person_ids.add(person_id)
            display = " ".join(part for part in (_text(row.get("first_name")), _text(row.get("last_name"))) if part)
            if display and person_id not in display_from_ballots:
                display_from_ballots[person_id] = display
    display_names = display_from_ballots | _display_names(conn, person_ids)

    packets: list[dict[str, Any]] = []
    for person_id in sorted(person_ids, key=lambda value: (int(value) if value.isdigit() else 10**12, value)):
        packet = _empty_summary(person_id, display_names.get(person_id), source_refs)
        packet.update(
            {
                "cutoff": cutoff,
                "minimum_peer_count": minimum_peer_count,
                "snapshot_sha256": snapshot_sha256,
            }
        )
        summary = packet["summary"]
        comparisons: list[dict[str, Any]] = []
        for source in sources:
            comparison = _compare_source(
                source,
                person_id,
                minimum_peer_count=minimum_peer_count,
                prepared=prepared_sources[str(source["source_id"])],
            )
            state = comparison["status"]
            summary["status_counts"][state] = int(summary["status_counts"].get(state, 0)) + 1
            reason = str(comparison.get("reason") or state)
            summary["reason_counts"][reason] = int(summary["reason_counts"].get(reason, 0)) + 1
            if source.get("matter") is not None:
                matter = str(source["matter"])
                summary["matter_label_counts"][matter] = int(summary["matter_label_counts"].get(matter, 0)) + 1
            if state == "SOURCE_EXCLUDED":
                summary["source_excluded_count"] += 1
                if comparison.get("target_state") == "ABSENT":
                    summary["absent_count"] += 1
                elif comparison.get("target_state") == "BLANK":
                    summary["target_blank_count"] += 1
                elif comparison.get("target_state") == "UNKNOWN_RESPONSE":
                    summary["target_unknown_response_count"] += 1
                continue
            summary["valid_source_count"] += 1
            if state == "TARGET_ABSENT":
                summary["source_row_missing_count"] += 1
                summary["absent_count"] += 1
                continue
            # Every valid, present-target result is retained as a compact
            # comparison record.  The shared source rows are not duplicated.
            comparisons.append(comparison)
            summary["present_count"] += 1
            group_state = comparison.get("target_group_state")
            if group_state == "KNOWN":
                summary["group_known_count"] += 1
            elif group_state == "GROUPLESS_EXCLUDED":
                summary["group_less_excluded_count"] += 1
            elif group_state == "UNKNOWN":
                summary["unknown_group_count"] += 1
            if state == "TARGET_BLANK":
                summary["target_blank_count"] += 1
            elif state == "TARGET_UNKNOWN_RESPONSE":
                summary["target_unknown_response_count"] += 1
            elif state == "TARGET_NON_SUBSTANTIVE":
                summary["target_non_substantive_count"] += 1
                if comparison.get("target_response") == "TYHJA":
                    summary["recorded_abstention_count"] += 1
                    summary["target_abstention_count"] += 1
                elif comparison.get("target_response") == "POISSA":
                    summary["recorded_absence_count"] += 1
            elif comparison.get("target_response") in SUBSTANTIVE:
                summary["target_substantive_count"] += 1
            if state == "COMPARABLE":
                summary["peer_eligible_count"] += 1
                summary["comparable_count"] += 1
                if comparison.get("matches_peer_majority"):
                    summary["matching_count"] += 1
                if comparison.get("contested"):
                    summary["contested_comparable_count"] += 1
                    if comparison.get("matches_peer_majority"):
                        summary["contested_matching_count"] += 1
            elif comparison.get("target_response") in SUBSTANTIVE and comparison.get("target_group_state") == "KNOWN":
                summary["peer_undefined_count"] += 1
            if comparison.get("contested"):
                summary["contested_source_count"] += 1
        packet["comparisons"] = comparisons
        packet["comparison_refs"] = [comparison["source_ref"] for comparison in comparisons]
        _finalise_summary(packet)
        packets.append(packet)

    state_counts = Counter(str(source.get("state")) for source in sources)
    source_reasons = Counter(reason for source in sources for reason in source.get("exclusion_reasons", []))
    observed_group_codes = Counter(
        str(row.get("group_code") or "")
        for source in sources
        for row in source.get("rows", [])
    )
    valid_sources = [source for source in sources if source.get("valid")]
    observed_rows = sum(int(source.get("row_count") or 0) for source in sources)
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "normalization_version": NORMALIZATION_VERSION,
        "source_kind": SOURCE_KIND,
        "cutoff": cutoff,
        "minimum_peer_count": minimum_peer_count,
        "membership_basis": "EVENT_BALLOT_EdustajaRyhmaLyhenne",
        "current_actor_party_not_used": True,
        "source_count": len(sources),
        "valid_source_count": len(valid_sources),
        "excluded_source_count": len(sources) - len(valid_sources),
        "source_state_counts": dict(sorted(state_counts.items())),
        "source_exclusion_reason_counts": dict(sorted(source_reasons.items())),
        "ballot_row_count": observed_rows,
        "person_count": len(packets),
        "packet_count": len(packets),
        "comparison_count": sum(len(packet["comparisons"]) for packet in packets),
        "comparable_count": sum(int(packet["summary"]["comparable_count"]) for packet in packets),
        "matching_count": sum(int(packet["summary"]["matching_count"]) for packet in packets),
        "source_row_missing_count": sum(int(packet["summary"]["source_row_missing_count"]) for packet in packets),
        "recorded_absence_count": sum(int(packet["summary"]["recorded_absence_count"]) for packet in packets),
        "recorded_abstention_count": sum(int(packet["summary"]["recorded_abstention_count"]) for packet in packets),
        "target_blank_count": sum(int(packet["summary"]["target_blank_count"]) for packet in packets),
        "target_unknown_response_count": sum(int(packet["summary"]["target_unknown_response_count"]) for packet in packets),
        "contested_source_count": sum(bool(source.get("valid") and source.get("global_contested")) for source in sources),
        "contested_comparable_count": sum(int(packet["summary"]["contested_comparable_count"]) for packet in packets),
        "votes_without_ballots": vote_count - len(sources),
        "orphan_ballot_rows": orphan_rows,
        "all_four_published_buckets_reconciled": not any(
            source.get("state") == "EXCLUDED_SOURCE_MISMATCH" for source in sources
        ),
        "group_vocabulary": {
            "known_codes": sorted(KNOWN_GROUP_CODES),
            "groupless_codes": sorted(GROUPLESS_CODES),
            "unknown_policy": "EXCLUDED_FROM_PEER_REFERENCE",
        },
        "observed_group_code_counts": dict(sorted(observed_group_codes.items())),
        "denominator_policy": {
            "source_row_missing_count": "a target person has no row in a reconciled source; it is not a recorded POISSA and not evidence of missing the sitting",
            "recorded_absence_count": "the target has an explicit POISSA ballot",
            "recorded_abstention_count": "the target has an explicit TYHJA ballot",
            "target_absent": "legacy alias for source_row_missing_count; separate from recorded_absence_count",
            "target_blank": "separate from target_absent; source reconciliation normally excludes unknown blank rows",
            "target_abstention": "legacy alias for recorded_abstention_count; POISSA is recorded_absence_count",
            "peer_tie": "undefined, not agreement or dissent",
            "peer_count_below_minimum": "undefined, not agreement or dissent",
            "contest_subset": "published global JAA>0 and EI>0; descriptive only",
        },
        "matter_sensitivity": _matter_report(sources),
        "snapshot_sha256": _snapshot_hash(sources, cutoff=cutoff, minimum_peer_count=minimum_peer_count),
        "hash_basis": "NORMALIZED_SNAPSHOT_NOT_RAW_HTTP_PAYLOAD",
        "source_refs": source_refs,
        "limitations": [
            "Group-majority agreement is descriptive and does not establish obedience, private influence, independent thought or policy meaning.",
            "Votes without stored ballot rows are not treated as universal absence.",
            "A missing person row within a reconciled source is a denominator state only; it does not establish that the member missed a sitting.",
            "Exact formal matter labels are retained; semantic matter clustering and repeated-round independence claims are not performed.",
            "A one-member or group-less event group has no peer reference by design.",
        ],
    }
    return packets, sources, report


def _source_map(sources: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    if isinstance(sources, Mapping):
        # Accept either {source_id: source} or a full build result mapping.
        if "sources" in sources and isinstance(sources["sources"], Sequence):
            return _source_map(sources["sources"])
        return {str(key): value for key, value in sources.items() if isinstance(value, Mapping)}
    return {
        str(source.get("source_id")): source
        for source in sources
        if isinstance(source, Mapping) and source.get("source_id")
    }


def _expected_packet(
    packet: Mapping[str, Any],
    sources: Mapping[str, Mapping[str, Any]],
    *,
    minimum_peer_count: int,
    prepared_sources: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    person_id = _text(packet.get("person_id"))
    source_refs = packet.get("source_refs")
    errors: list[str] = []
    if not isinstance(source_refs, Sequence) or isinstance(source_refs, (str, bytes, bytearray)):
        source_refs = []
        errors.append("source_refs is not a list")
    source_refs = [str(ref) for ref in source_refs]
    if len(source_refs) != len(set(source_refs)):
        errors.append("source_refs contains duplicates")
    missing = [ref for ref in source_refs if ref not in sources]
    if missing:
        errors.append(f"missing source refs: {missing[:3]}")
    expected = _empty_summary(person_id, _text(packet.get("display_name")) or None, source_refs)
    summary = expected["summary"]
    comparisons: list[dict[str, Any]] = []
    prepared_sources = prepared_sources or {
        source_ref: _prepare_source(source)
        for source_ref, source in sources.items()
        if source_ref in source_refs
    }
    for source_ref in source_refs:
        source = sources.get(source_ref)
        if source is None:
            continue
        comparison = _compare_source(
            source,
            person_id,
            minimum_peer_count=minimum_peer_count,
            prepared=prepared_sources[source_ref],
        )
        state = comparison["status"]
        summary["status_counts"][state] = int(summary["status_counts"].get(state, 0)) + 1
        reason = str(comparison.get("reason") or state)
        summary["reason_counts"][reason] = int(summary["reason_counts"].get(reason, 0)) + 1
        if source.get("matter") is not None:
            label = str(source["matter"])
            summary["matter_label_counts"][label] = int(summary["matter_label_counts"].get(label, 0)) + 1
        if state == "SOURCE_EXCLUDED":
            summary["source_excluded_count"] += 1
            target_state = comparison.get("target_state")
            if target_state == "ABSENT":
                summary["absent_count"] += 1
            elif target_state == "BLANK":
                summary["target_blank_count"] += 1
            elif target_state == "UNKNOWN_RESPONSE":
                summary["target_unknown_response_count"] += 1
            continue
        summary["valid_source_count"] += 1
        if state == "TARGET_ABSENT":
            summary["source_row_missing_count"] += 1
            summary["absent_count"] += 1
            continue
        comparisons.append(comparison)
        summary["present_count"] += 1
        group_state = comparison.get("target_group_state")
        if group_state == "KNOWN":
            summary["group_known_count"] += 1
        elif group_state == "GROUPLESS_EXCLUDED":
            summary["group_less_excluded_count"] += 1
        elif group_state == "UNKNOWN":
            summary["unknown_group_count"] += 1
        state_response = comparison.get("target_response")
        if state == "TARGET_BLANK":
            summary["target_blank_count"] += 1
        elif state == "TARGET_UNKNOWN_RESPONSE":
            summary["target_unknown_response_count"] += 1
        elif state == "TARGET_NON_SUBSTANTIVE":
            summary["target_non_substantive_count"] += 1
            if state_response == "TYHJA":
                summary["recorded_abstention_count"] += 1
                summary["target_abstention_count"] += 1
            elif state_response == "POISSA":
                summary["recorded_absence_count"] += 1
        elif state_response in SUBSTANTIVE:
            summary["target_substantive_count"] += 1
        if state == "COMPARABLE":
            summary["peer_eligible_count"] += 1
            summary["comparable_count"] += 1
            if comparison.get("matches_peer_majority"):
                summary["matching_count"] += 1
            if comparison.get("contested"):
                summary["contested_comparable_count"] += 1
                if comparison.get("matches_peer_majority"):
                    summary["contested_matching_count"] += 1
        elif state_response in SUBSTANTIVE and group_state == "KNOWN":
            summary["peer_undefined_count"] += 1
        if comparison.get("contested"):
            summary["contested_source_count"] += 1
    expected["comparisons"] = comparisons
    expected["comparison_refs"] = [comparison["source_ref"] for comparison in comparisons]
    _finalise_summary(expected)
    return expected, errors


def validate_group_packet(
    packet: Mapping[str, Any],
    sources: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    minimum_peer_count: int | None = None,
) -> dict[str, Any]:
    """Recompute and validate one person packet against shared sources.

    The return value is structured so callers can retain an audit receipt:
    ``{"valid": bool, "errors": [...], "checked_comparisons": int}``.
    Source hashes, published totals, duplicate IDs, void/future gates and all
    comparison fields are independently recomputed here.
    """

    if not isinstance(packet, Mapping):
        return {"valid": False, "errors": ["packet is not an object"], "checked_sources": 0, "checked_comparisons": 0}
    source_map = _source_map(sources)
    errors: list[str] = []
    if packet.get("packet_type") != "GROUP_AGREEMENT_PERSON":
        errors.append("packet_type is not GROUP_AGREEMENT_PERSON")
    if packet.get("schema_version") != SCHEMA_VERSION:
        errors.append("packet schema_version is not supported")
    minimum = minimum_peer_count
    if minimum is None:
        raw = packet.get("minimum_peer_count")
        minimum = _as_int(raw) or DEFAULT_MINIMUM_PEERS
    if minimum < 1:
        return {"valid": False, "errors": errors + ["minimum_peer_count must be positive"], "checked_sources": 0, "checked_comparisons": 0}
    if packet.get("minimum_peer_count") != minimum:
        errors.append("packet minimum_peer_count is missing or differs from validator input")
    if not packet.get("cutoff"):
        errors.append("packet cutoff is missing")
    if not packet.get("snapshot_sha256"):
        errors.append("packet snapshot_sha256 is missing")
    refs = packet.get("source_refs") if isinstance(packet.get("source_refs"), Sequence) and not isinstance(packet.get("source_refs"), (str, bytes, bytearray)) else []
    refs = [str(ref) for ref in refs]
    checked_sources = 0
    for source_ref in refs:
        source = source_map.get(source_ref)
        if source is None:
            continue
        checked_sources += 1
        expected_hash = _source_hash(source)
        if source.get("source_sha256") != expected_hash:
            errors.append(f"{source_ref}: source_sha256 does not cover normalized source")
        diagnostics = _source_diagnostics(source, cutoff=source.get("cutoff"))
        claimed_state = source.get("state")
        if claimed_state != diagnostics["state"]:
            errors.append(f"{source_ref}: claimed source state differs from independent reconciliation")
        if bool(source.get("valid")) != bool(diagnostics["valid"]):
            errors.append(f"{source_ref}: claimed source validity differs from independent reconciliation")
        if source.get("rows") != diagnostics["normalized_rows"]:
            errors.append(f"{source_ref}: rows are not in the declared normalized form")
        if source.get("observed_totals") != diagnostics["observed_totals"]:
            errors.append(f"{source_ref}: observed response buckets changed")
        if source.get("published_totals") != diagnostics["published_totals"]:
            errors.append(f"{source_ref}: published totals changed")
    prepared_sources = {
        source_ref: _prepare_source(source_map[source_ref])
        for source_ref in refs
        if source_ref in source_map
    }
    expected, expected_errors = _expected_packet(
        packet,
        source_map,
        minimum_peer_count=minimum,
        prepared_sources=prepared_sources,
    )
    errors.extend(expected_errors)
    actual_comparisons = packet.get("comparisons")
    if not isinstance(actual_comparisons, Sequence) or isinstance(actual_comparisons, (str, bytes, bytearray)):
        errors.append("comparisons is not a list")
        actual_comparisons = []
    actual_by_ref = {str(item.get("source_ref")): item for item in actual_comparisons if isinstance(item, Mapping)}
    if len(actual_by_ref) != len(actual_comparisons):
        errors.append("comparisons contains duplicate or malformed source refs")
    expected_by_ref = {str(item["source_ref"]): item for item in expected["comparisons"]}
    if set(actual_by_ref) != set(expected_by_ref):
        errors.append("comparison source refs do not match independently emitted target-present results")
    for source_ref, expected_comparison in expected_by_ref.items():
        actual_comparison = actual_by_ref.get(source_ref)
        if actual_comparison is None:
            continue
        # JSON numeric/bool/null values are compared exactly.  Ignore no
        # fields: a changed rationale or matter label is also a changed audit
        # record, not a harmless presentation edit.
        if dict(actual_comparison) != expected_comparison:
            errors.append(f"{source_ref}: comparison differs from independent recomputation")
    if packet.get("comparison_refs") != [item["source_ref"] for item in expected["comparisons"]]:
        errors.append("comparison_refs differ from independently recomputed comparisons")
    actual_summary = packet.get("summary")
    if not isinstance(actual_summary, Mapping) or dict(actual_summary) != expected["summary"]:
        errors.append("summary differs from independently recomputed denominators")
    # Snapshot hashes bind packet references to the shared source collection.
    if packet.get("snapshot_sha256"):
        expected_snapshot = _snapshot_hash(
            [source_map[ref] for ref in refs if ref in source_map],
            cutoff=str(packet.get("cutoff") or (next((source_map[ref].get("cutoff") for ref in refs if ref in source_map), CORPUS_CUTOFF))),
            minimum_peer_count=minimum,
        )
        if packet.get("snapshot_sha256") != expected_snapshot:
            errors.append("packet snapshot_sha256 does not bind its source refs")
    return {
        "valid": not errors,
        "errors": errors,
        "checked_sources": checked_sources,
        "checked_comparisons": len(expected["comparisons"]),
        "person_id": packet.get("person_id"),
    }


def validate_group_packets(
    packets: Sequence[Mapping[str, Any]] | Mapping[str, Any],
    sources: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    minimum_peer_count: int | None = None,
) -> dict[str, Any]:
    """Validate a complete group-context collection in one source pass.

    ``validate_group_packet`` is useful for an individual profile, but using
    it repeatedly would reparse and rehash every shared vote for every person.
    This collection validator prepares each source exactly once, then
    independently recomputes every packet's comparison rows and denominators
    against that prepared source map.  The complete collection is required to
    contain exactly one packet for each person observed in the shared source
    rows and exactly the same source-reference set in every packet.

    The return value is intentionally a bounded receipt: counts, set hashes,
    the source snapshot hash, and errors.  It does not copy packet or ballot
    data into the receipt.
    """

    if isinstance(packets, Mapping):
        packet_values = packets.get("packets")
        if not isinstance(packet_values, Sequence) or isinstance(packet_values, (str, bytes, bytearray)):
            return {
                "valid": False,
                "errors": ["packets mapping must contain a packet list"],
                "checked_sources": 0,
                "checked_packets": 0,
                "checked_comparisons": 0,
            }
        packet_list = list(packet_values)
    elif isinstance(packets, Sequence) and not isinstance(packets, (str, bytes, bytearray)):
        packet_list = list(packets)
    else:
        return {
            "valid": False,
            "errors": ["packets is not a sequence"],
            "checked_sources": 0,
            "checked_packets": 0,
            "checked_comparisons": 0,
        }

    # Preserve duplicate source IDs long enough to report them instead of
    # letting _source_map silently overwrite one source with another.
    if isinstance(sources, Mapping) and "sources" in sources:
        raw_sources = sources.get("sources")
    else:
        raw_sources = sources.values() if isinstance(sources, Mapping) else sources
    if not isinstance(raw_sources, Sequence) or isinstance(raw_sources, (str, bytes, bytearray)):
        raw_sources = list(raw_sources) if raw_sources is not None else []
    else:
        raw_sources = list(raw_sources)

    errors: list[str] = []
    source_ids_in_order: list[str] = []
    source_map: dict[str, Mapping[str, Any]] = {}
    duplicate_source_ids: set[str] = set()
    for source in raw_sources:
        if not isinstance(source, Mapping):
            errors.append("source collection contains a non-object")
            continue
        source_id = str(source.get("source_id") or "")
        if not source_id:
            errors.append("source collection contains a source without source_id")
            continue
        if source_id in source_map:
            duplicate_source_ids.add(source_id)
        source_ids_in_order.append(source_id)
        source_map[source_id] = source
    if duplicate_source_ids:
        errors.append(f"duplicate source IDs: {sorted(duplicate_source_ids)[:5]}")
    source_ids = sorted(source_map)

    # Validate source integrity and prepare group totals once.  The source
    # diagnostics are reused by _prepare_source instead of being recomputed
    # for every person packet.
    diagnostics_by_source: dict[str, Mapping[str, Any]] = {}
    prepared_sources: dict[str, Mapping[str, Any]] = {}
    for source_id in source_ids:
        source = source_map[source_id]
        expected_hash = _source_hash(source)
        if source.get("source_sha256") != expected_hash:
            errors.append(f"{source_id}: source_sha256 does not cover normalized source")
        diagnostics = _source_diagnostics(source, cutoff=source.get("cutoff"))
        diagnostics_by_source[source_id] = diagnostics
        if source.get("schema_version") != SCHEMA_VERSION:
            errors.append(f"{source_id}: unsupported source schema_version")
        if source.get("normalization_version") != NORMALIZATION_VERSION:
            errors.append(f"{source_id}: unsupported normalization_version")
        if source.get("source_kind") != SOURCE_KIND:
            errors.append(f"{source_id}: unsupported source_kind")
        if source.get("state") != diagnostics["state"]:
            errors.append(f"{source_id}: source state differs from independent reconciliation")
        if bool(source.get("valid")) != bool(diagnostics["valid"]):
            errors.append(f"{source_id}: source validity differs from independent reconciliation")
        if source.get("rows") != diagnostics["normalized_rows"]:
            errors.append(f"{source_id}: rows are not in declared normalized form")
        if source.get("published_totals") != diagnostics["published_totals"]:
            errors.append(f"{source_id}: published totals changed")
        if source.get("observed_totals") != diagnostics["observed_totals"]:
            errors.append(f"{source_id}: observed totals changed")
        prepared_sources[source_id] = _prepare_source(source, diagnostics=diagnostics)

    cutoffs = {str(source_map[source_id].get("cutoff")) for source_id in source_ids}
    if len(cutoffs) > 1:
        errors.append(f"source cutoff set is inconsistent: {sorted(cutoffs)}")
    inferred_cutoff = next(iter(cutoffs), CORPUS_CUTOFF)
    if not inferred_cutoff or inferred_cutoff == "None":
        inferred_cutoff = CORPUS_CUTOFF

    if minimum_peer_count is None:
        packet_minima = {
            _as_int(packet.get("minimum_peer_count"))
            for packet in packet_list
            if isinstance(packet, Mapping) and _as_int(packet.get("minimum_peer_count")) is not None
        }
        if len(packet_minima) > 1:
            errors.append(f"packet minimum_peer_count set is inconsistent: {sorted(packet_minima)}")
        minimum = next(iter(packet_minima), DEFAULT_MINIMUM_PEERS)
    else:
        minimum = minimum_peer_count
    if minimum < 1:
        errors.append("minimum_peer_count must be positive")
        minimum = DEFAULT_MINIMUM_PEERS

    expected_snapshot = _snapshot_hash(
        [source_map[source_id] for source_id in source_ids],
        cutoff=inferred_cutoff,
        minimum_peer_count=minimum,
    )
    source_ref_set_sha256 = _sha256(source_ids)

    expected_person_ids = sorted(
        {
            str(row.get("person_id"))
            for source in source_map.values()
            for row in source.get("rows", [])
            if isinstance(row, Mapping) and str(row.get("person_id") or "")
        },
        key=lambda value: (int(value) if value.isdigit() else 10**12, value),
    )
    expected_person_set_sha256 = _sha256(expected_person_ids)
    packet_person_ids = [
        str(packet.get("person_id")) if isinstance(packet, Mapping) else ""
        for packet in packet_list
    ]
    duplicate_person_ids = sorted({person_id for person_id, count in Counter(packet_person_ids).items() if count > 1})
    if duplicate_person_ids:
        errors.append(f"duplicate packet person IDs: {duplicate_person_ids[:5]}")
    if sorted(packet_person_ids, key=lambda value: (int(value) if value.isdigit() else 10**12, value)) != expected_person_ids:
        missing_people = sorted(set(expected_person_ids) - set(packet_person_ids))
        extra_people = sorted(set(packet_person_ids) - set(expected_person_ids))
        errors.append(f"packet person set differs: missing={missing_people[:5]} extra={extra_people[:5]}")

    valid_packets = 0
    checked_comparisons = 0
    packet_errors: list[dict[str, Any]] = []
    expected_source_id_set = set(source_ids)
    for packet in packet_list:
        if not isinstance(packet, Mapping):
            packet_errors.append({"person_id": None, "errors": ["packet is not an object"]})
            continue
        person_id = str(packet.get("person_id") or "")
        local_errors: list[str] = []
        if packet.get("packet_type") != "GROUP_AGREEMENT_PERSON":
            local_errors.append("packet_type is not GROUP_AGREEMENT_PERSON")
        if packet.get("schema_version") != SCHEMA_VERSION:
            local_errors.append("unsupported packet schema_version")
        if packet.get("minimum_peer_count") != minimum:
            local_errors.append("minimum_peer_count differs from collection")
        if packet.get("cutoff") != inferred_cutoff:
            local_errors.append("cutoff differs from collection")
        if packet.get("snapshot_sha256") != expected_snapshot:
            local_errors.append("snapshot_sha256 differs from shared source collection")
        raw_refs = packet.get("source_refs")
        if not isinstance(raw_refs, Sequence) or isinstance(raw_refs, (str, bytes, bytearray)):
            refs: list[str] = []
            local_errors.append("source_refs is not a list")
        else:
            refs = [str(ref) for ref in raw_refs]
        if len(refs) != len(set(refs)):
            local_errors.append("source_refs contains duplicates")
        ref_set = set(refs)
        if ref_set != expected_source_id_set:
            missing_refs = sorted(expected_source_id_set - ref_set)
            extra_refs = sorted(ref_set - expected_source_id_set)
            local_errors.append(f"source reference set differs: missing={missing_refs[:5]} extra={extra_refs[:5]}")

        expected, expected_errors = _expected_packet(
            packet,
            source_map,
            minimum_peer_count=minimum,
            prepared_sources=prepared_sources,
        )
        local_errors.extend(expected_errors)
        actual_comparisons = packet.get("comparisons")
        if not isinstance(actual_comparisons, Sequence) or isinstance(actual_comparisons, (str, bytes, bytearray)):
            local_errors.append("comparisons is not a list")
            actual_comparisons = []
        actual_by_ref = {
            str(item.get("source_ref")): item
            for item in actual_comparisons
            if isinstance(item, Mapping)
        }
        if len(actual_by_ref) != len(actual_comparisons):
            local_errors.append("comparisons contains duplicate or malformed source refs")
        expected_by_ref = {str(item["source_ref"]): item for item in expected["comparisons"]}
        if set(actual_by_ref) != set(expected_by_ref):
            local_errors.append("comparison source refs differ from independent recomputation")
        for source_ref, expected_comparison in expected_by_ref.items():
            actual_comparison = actual_by_ref.get(source_ref)
            if actual_comparison is not None and dict(actual_comparison) != expected_comparison:
                local_errors.append(f"{source_ref}: comparison differs from independent recomputation")
        if packet.get("comparison_refs") != [item["source_ref"] for item in expected["comparisons"]]:
            local_errors.append("comparison_refs differ from independent recomputation")
        if not isinstance(packet.get("summary"), Mapping) or dict(packet["summary"]) != expected["summary"]:
            local_errors.append("summary differs from independent recomputation")
        checked_comparisons += len(expected["comparisons"])
        if local_errors:
            packet_errors.append({"person_id": person_id, "errors": local_errors})
        else:
            valid_packets += 1

    errors.extend(f"packet {item['person_id']}: {error}" for item in packet_errors for error in item["errors"])
    return {
        "valid": not errors,
        "errors": errors,
        "checked_sources": len(source_ids),
        "checked_packets": len(packet_list),
        "valid_packets": valid_packets,
        "invalid_packets": len(packet_errors),
        "checked_comparisons": checked_comparisons,
        "source_count": len(source_ids),
        "packet_count": len(packet_list),
        "minimum_peer_count": minimum,
        "cutoff": inferred_cutoff,
        "source_snapshot_sha256": expected_snapshot,
        "source_ref_set_sha256": source_ref_set_sha256,
        "packet_person_id_set_sha256": expected_person_set_sha256,
        "expected_person_count": len(expected_person_ids),
    }


__all__ = [
    "DEFAULT_MINIMUM_PEERS",
    "SCHEMA_VERSION",
    "GroupAgreementError",
    "build_group_context",
    "build_group_sources",
    "validate_group_packet",
    "validate_group_packets",
]

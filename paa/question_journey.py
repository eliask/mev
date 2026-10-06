"""Question journey projection from reviewed answers and documentary route packets.

Owned projection only: an explicit mapping names nodes, edges and branches. The
renderer never infers substantive relations. Lexical disposition cues and
coordinate overlaps from ``route_packet`` remain candidates unless a supporting
reviewed claim admits a stronger reading. New graph meaning starts UNREVIEWED and
needs a separate projection review bound to mapping/candidate/bank bytes.
"""

import hashlib
import html
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Final, Literal

from paa.case_site import _document

SCHEMA_VERSION: Final = "paa.question_journey.v1"
MAPPING_SCHEMA_VERSION: Final = "paa.question_journey_mapping.v1"
PROJECTION_REVIEW_SCHEMA: Final = "paa.journey_projection_review.v1"
SOURCE_ANSWER_REVIEW_SCHEMA: Final = "paa.source_answer_review.v1"

ADMISSION_UNREVIEWED: Final = "UNREVIEWED_PROJECTION_CANDIDATE"
ADMISSION_REVIEWED: Final = "PROJECTION_REVIEWED_AI_SOURCE_READING"
ADMISSION_DOCUMENTARY: Final = "DOCUMENTARY_PROJECTION_NOT_SEMANTIC_ACCEPTANCE"

NodeKind = Literal[
    "ACTOR_ACT",
    "PROPOSAL",
    "RECOMMENDATION",
    "ALTERNATIVE",
    "INSTITUTIONAL_RESULT",
    "ENACTED_STATUTE",
    "LEGAL_STATE",
    "PROCEDURAL_EVENT",
    "UNRESOLVED_ENDPOINT",
]
RelationKind = Literal[
    "DOCUMENTARY_REFERENCE",
    "FORMAL_PROCEDURAL_STAGE",
    "PERSONAL_ACT",
    "INSTITUTIONAL_RESULT",
    "ENACTED_LEGAL_STATE",
    "ANNUAL_SUCCESSOR_NOT_AMENDMENT",
    "INTERPRETED_COMPARISON",
    "UNRESOLVED_RELATION",
    "LEXICAL_CUE_CANDIDATE",
]
JourneyKind = Literal["BRANCHING_ROUTES", "ALTERNATIVE_SEQUENCE"]

_NODE_KINDS: Final = frozenset(
    {
        "ACTOR_ACT",
        "PROPOSAL",
        "RECOMMENDATION",
        "ALTERNATIVE",
        "INSTITUTIONAL_RESULT",
        "ENACTED_STATUTE",
        "LEGAL_STATE",
        "PROCEDURAL_EVENT",
        "UNRESOLVED_ENDPOINT",
    }
)
_RELATION_KINDS: Final = frozenset(
    {
        "DOCUMENTARY_REFERENCE",
        "FORMAL_PROCEDURAL_STAGE",
        "PERSONAL_ACT",
        "INSTITUTIONAL_RESULT",
        "ENACTED_LEGAL_STATE",
        "ANNUAL_SUCCESSOR_NOT_AMENDMENT",
        "INTERPRETED_COMPARISON",
        "UNRESOLVED_RELATION",
        "LEXICAL_CUE_CANDIDATE",
    }
)
_STRONG_RELATIONS: Final = frozenset(
    {
        "FORMAL_PROCEDURAL_STAGE",
        "INSTITUTIONAL_RESULT",
        "ENACTED_LEGAL_STATE",
        "ANNUAL_SUCCESSOR_NOT_AMENDMENT",
        "INTERPRETED_COMPARISON",
    }
)
_JOURNEY_KINDS: Final = frozenset({"BRANCHING_ROUTES", "ALTERNATIVE_SEQUENCE"})
_PARTICIPANT_ROLES: Final = frozenset({"FIRST_SIGNATORY", "CO_SIGNATORY", "NAMED_CO_PROPOSER"})
_COVERAGE_PREPARED: Final = "PREPARED_AI_SOURCE_REVIEWED"
_COVERAGE_RECORD: Final = "RECORD_ONLY"
_COVERAGE_ABSENT: Final = "ABSENT_FROM_INDEX"


class QuestionJourneyError(ValueError):
    """Raised when a journey projection cannot be composed or validated safely."""


def _e(value: object) -> str:
    return html.escape(str(value if value is not None else ""), quote=True)


def _sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha_path(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _sha_json(value: object) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _sha_bytes(raw)


def _load_json(path: Path) -> dict[str, Any]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        raise QuestionJourneyError(f"expected JSON object: {path}")
    return doc


def _slug(value: str) -> str:
    text = re.sub(r"[^\w]+", "_", value.strip(), flags=re.UNICODE)
    return text.strip("_") or "item"


def _require_str(row: Mapping[str, Any], key: str) -> str:
    value = row.get(key)
    if type(value) is not str or not value.strip():
        raise QuestionJourneyError(f"mapping field {key!r} must be a non-empty string")
    return value


def _optional_str(row: Mapping[str, Any], key: str) -> str | None:
    value = row.get(key)
    if value is None:
        return None
    if type(value) is not str:
        raise QuestionJourneyError(f"mapping field {key!r} must be a string or null")
    text = value.strip()
    return text or None


def _int_list(value: object, *, field: str) -> list[int]:
    if not isinstance(value, list) or any(type(item) is not int for item in value):
        raise QuestionJourneyError(f"{field} must be a list of integers")
    if len(set(value)) != len(value):
        raise QuestionJourneyError(f"{field} contains duplicates")
    return list(value)


def _str_list(value: object, *, field: str) -> list[str]:
    if not isinstance(value, list) or any(type(item) is not str or not item.strip() for item in value):
        raise QuestionJourneyError(f"{field} must be a list of non-empty strings")
    if len(set(value)) != len(value):
        raise QuestionJourneyError(f"{field} contains duplicates")
    return list(value)


@dataclass(frozen=True, slots=True, kw_only=True)
class JourneyDefect:
    code: str
    reason: str
    path: str | None = None


def _validate_participants(value: object, *, path: str) -> list[JourneyDefect]:
    """Structured co-participant records; never inferred from rendered labels."""

    defects: list[JourneyDefect] = []
    if not isinstance(value, list) or not value:
        defects.append(
            JourneyDefect(
                code="INVALID_PARTICIPANTS",
                reason="participants must be a non-empty list when present",
                path=path,
            )
        )
        return defects
    seen_names: set[str] = set()
    shared_ids: set[str] = set()
    for index, row in enumerate(value):
        row_path = f"{path}[{index}]"
        if not isinstance(row, Mapping):
            defects.append(
                JourneyDefect(code="INVALID_PARTICIPANT", reason="participant must be an object", path=row_path)
            )
            continue
        try:
            name = _require_str(row, "actor_name")
            role = _require_str(row, "signature_role")
            shared = _require_str(row, "shared_act_id")
        except QuestionJourneyError as error:
            defects.append(JourneyDefect(code="INVALID_PARTICIPANT", reason=str(error), path=row_path))
            continue
        if role not in _PARTICIPANT_ROLES:
            defects.append(
                JourneyDefect(
                    code="INVALID_PARTICIPANT_ROLE",
                    reason=f"signature_role must be one of {sorted(_PARTICIPANT_ROLES)}",
                    path=f"{row_path}.signature_role",
                )
            )
        key = name.casefold()
        if key in seen_names:
            defects.append(
                JourneyDefect(
                    code="DUPLICATE_PARTICIPANT",
                    reason=name,
                    path=row_path,
                )
            )
        seen_names.add(key)
        shared_ids.add(shared)
        for optional in ("source_id", "date", "evidence_note"):
            if optional in row and row.get(optional) is not None and type(row.get(optional)) is not str:
                defects.append(
                    JourneyDefect(
                        code="INVALID_PARTICIPANT_FIELD",
                        reason=f"{optional} must be a string when present",
                        path=f"{row_path}.{optional}",
                    )
                )
        for optional in ("source_start", "source_end"):
            if optional in row and row.get(optional) is not None and type(row.get(optional)) is not int:
                defects.append(
                    JourneyDefect(
                        code="INVALID_PARTICIPANT_FIELD",
                        reason=f"{optional} must be an integer when present",
                        path=f"{row_path}.{optional}",
                    )
                )
        start = row.get("source_start")
        end = row.get("source_end")
        if type(start) is int and type(end) is int and end <= start:
            defects.append(
                JourneyDefect(
                    code="INVALID_PARTICIPANT_SPAN",
                    reason="source_end must be greater than source_start",
                    path=row_path,
                )
            )
    if len(shared_ids) > 1:
        defects.append(
            JourneyDefect(
                code="PARTICIPANT_SHARED_ACT_MISMATCH",
                reason="all participants on one node must share one shared_act_id",
                path=path,
            )
        )
    return defects


def _decode_participants(value: object) -> list[dict[str, Any]]:
    if not value:
        return []
    if not isinstance(value, list):
        raise QuestionJourneyError("participants must be a list")
    out: list[dict[str, Any]] = []
    for row in value:
        if not isinstance(row, Mapping):
            raise QuestionJourneyError("participant must be an object")
        item = {
            "actor_name": _require_str(row, "actor_name"),
            "signature_role": _require_str(row, "signature_role"),
            "shared_act_id": _require_str(row, "shared_act_id"),
        }
        for key in ("source_id", "date", "evidence_note"):
            text = _optional_str(row, key)
            if text is not None:
                item[key] = text
        for key in ("source_start", "source_end"):
            if key in row and row.get(key) is not None:
                if type(row.get(key)) is not int:
                    raise QuestionJourneyError(f"participant {key} must be an integer")
                item[key] = row[key]
        out.append(item)
    return out


def validate_mapping_wire(mapping: Mapping[str, Any]) -> list[JourneyDefect]:
    """Validate mapping shape without binding to a route/answer (codec gate)."""

    defects: list[JourneyDefect] = []
    if mapping.get("schema_version") != MAPPING_SCHEMA_VERSION:
        defects.append(
            JourneyDefect(
                code="UNSUPPORTED_MAPPING_SCHEMA",
                reason=f"expected {MAPPING_SCHEMA_VERSION}, got {mapping.get('schema_version')!r}",
            )
        )
        return defects
    for key in ("task_id", "journey_kind", "short_answer", "nodes", "edges", "branches"):
        if key not in mapping:
            defects.append(JourneyDefect(code="MISSING_MAPPING_FIELD", reason=f"missing {key}", path=key))
    kind = mapping.get("journey_kind")
    if kind not in _JOURNEY_KINDS:
        defects.append(
            JourneyDefect(
                code="INVALID_JOURNEY_KIND",
                reason=f"journey_kind must be one of {sorted(_JOURNEY_KINDS)}",
                path="journey_kind",
            )
        )
    nodes = mapping.get("nodes")
    edges = mapping.get("edges")
    branches = mapping.get("branches")
    if not isinstance(nodes, list) or not nodes:
        defects.append(JourneyDefect(code="EMPTY_NODES", reason="nodes must be a non-empty list", path="nodes"))
        return defects
    if not isinstance(edges, list):
        defects.append(JourneyDefect(code="INVALID_EDGES", reason="edges must be a list", path="edges"))
        return defects
    if not isinstance(branches, list) or not branches:
        defects.append(
            JourneyDefect(code="EMPTY_BRANCHES", reason="branches must be a non-empty list", path="branches")
        )
        return defects
    node_ids: set[str] = set()
    for index, node in enumerate(nodes):
        path = f"nodes[{index}]"
        if not isinstance(node, Mapping):
            defects.append(JourneyDefect(code="INVALID_NODE", reason="node must be an object", path=path))
            continue
        try:
            node_id = _require_str(node, "node_id")
            node_kind = _require_str(node, "kind")
            _require_str(node, "label")
        except QuestionJourneyError as error:
            defects.append(JourneyDefect(code="INVALID_NODE", reason=str(error), path=path))
            continue
        if node_id in node_ids:
            defects.append(JourneyDefect(code="DUPLICATE_NODE_ID", reason=node_id, path=path))
        node_ids.add(node_id)
        if node_kind not in _NODE_KINDS:
            defects.append(JourneyDefect(code="INVALID_NODE_KIND", reason=node_kind, path=f"{path}.kind"))
        if "supporting_claim_indices" in node:
            try:
                _int_list(node.get("supporting_claim_indices"), field=f"{path}.supporting_claim_indices")
            except QuestionJourneyError as error:
                defects.append(JourneyDefect(code="INVALID_CLAIM_INDICES", reason=str(error), path=path))
        if "supporting_stage_source_ids" in node:
            try:
                _str_list(node.get("supporting_stage_source_ids"), field=f"{path}.supporting_stage_source_ids")
            except QuestionJourneyError as error:
                defects.append(JourneyDefect(code="INVALID_STAGE_REFS", reason=str(error), path=path))
        # Voice must be declared. Do not infer PERSON from actor_name or INSTITUTION
        # from node kind/label — missing voice is unresolved, not an institutional default.
        if "voice_kind" not in node or node.get("voice_kind") is None:
            defects.append(
                JourneyDefect(
                    code="VOICE_KIND_REQUIRED",
                    reason="voice_kind must be declared (PERSON|INSTITUTION|MIXED|UNKNOWN); no name/kind inference",
                    path=f"{path}.voice_kind",
                )
            )
        elif node.get("voice_kind") not in {"PERSON", "INSTITUTION", "MIXED", "UNKNOWN"}:
            defects.append(
                JourneyDefect(
                    code="INVALID_VOICE_KIND",
                    reason=str(node.get("voice_kind")),
                    path=f"{path}.voice_kind",
                )
            )
        if "participants" in node:
            defects.extend(_validate_participants(node.get("participants"), path=f"{path}.participants"))
            if node.get("actor_name") and isinstance(node.get("participants"), list):
                names = {
                    str(row.get("actor_name")).strip()
                    for row in node["participants"]
                    if isinstance(row, Mapping) and row.get("actor_name")
                }
                if str(node.get("actor_name")).strip() not in names:
                    defects.append(
                        JourneyDefect(
                            code="ACTOR_NOT_IN_PARTICIPANTS",
                            reason="node.actor_name must be one of participants when participants are declared",
                            path=f"{path}.actor_name",
                        )
                    )
    edge_ids: set[str] = set()
    for index, edge in enumerate(edges):
        path = f"edges[{index}]"
        if not isinstance(edge, Mapping):
            defects.append(JourneyDefect(code="INVALID_EDGE", reason="edge must be an object", path=path))
            continue
        try:
            edge_id = _require_str(edge, "edge_id")
            relation = _require_str(edge, "relation_kind")
            from_id = _require_str(edge, "from_node_id")
            to_id = _require_str(edge, "to_node_id")
            _require_str(edge, "label")
        except QuestionJourneyError as error:
            defects.append(JourneyDefect(code="INVALID_EDGE", reason=str(error), path=path))
            continue
        if edge_id in edge_ids:
            defects.append(JourneyDefect(code="DUPLICATE_EDGE_ID", reason=edge_id, path=path))
        edge_ids.add(edge_id)
        if relation not in _RELATION_KINDS:
            defects.append(
                JourneyDefect(code="INVALID_RELATION_KIND", reason=relation, path=f"{path}.relation_kind")
            )
        if from_id not in node_ids or to_id not in node_ids:
            defects.append(
                JourneyDefect(
                    code="UNBOUND_EDGE_ENDPOINT",
                    reason=f"{from_id} -> {to_id}",
                    path=path,
                )
            )
        if relation in _STRONG_RELATIONS and not (
            edge.get("supporting_claim_indices") or edge.get("supporting_stage_source_ids")
        ):
            defects.append(
                JourneyDefect(
                    code="STRONG_EDGE_WITHOUT_SUPPORT",
                    reason="strong relation requires claim or stage support",
                    path=path,
                )
            )
    for index, branch in enumerate(branches):
        path = f"branches[{index}]"
        if not isinstance(branch, Mapping):
            defects.append(JourneyDefect(code="INVALID_BRANCH", reason="branch must be an object", path=path))
            continue
        try:
            _require_str(branch, "branch_id")
            _require_str(branch, "title")
            node_refs = _str_list(branch.get("node_ids"), field=f"{path}.node_ids")
            edge_refs = _str_list(branch.get("edge_ids") or [], field=f"{path}.edge_ids")
        except QuestionJourneyError as error:
            defects.append(JourneyDefect(code="INVALID_BRANCH", reason=str(error), path=path))
            continue
        for node_id in node_refs:
            if node_id not in node_ids:
                defects.append(
                    JourneyDefect(code="UNBOUND_BRANCH_NODE", reason=node_id, path=f"{path}.node_ids")
                )
        for edge_id in edge_refs:
            if edge_id not in edge_ids:
                defects.append(
                    JourneyDefect(code="UNBOUND_BRANCH_EDGE", reason=edge_id, path=f"{path}.edge_ids")
                )
    return defects


def decode_mapping(mapping: Mapping[str, Any]) -> dict[str, Any]:
    defects = validate_mapping_wire(mapping)
    if defects:
        raise QuestionJourneyError(
            "invalid journey mapping: "
            + "; ".join(f"{item.code}:{item.path or ''}:{item.reason}" for item in defects[:8])
        )
    return dict(mapping)


def _stage_index(route_packet: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in (route_packet.get("summary") or {}).get("stages") or []:
        if isinstance(row, Mapping) and row.get("source_id"):
            out[str(row["source_id"])] = dict(row)
    return out


def _claim_list(answer: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if answer is None:
        return []
    claims = answer.get("claims") or []
    if not isinstance(claims, list):
        raise QuestionJourneyError("answer claims must be a list")
    return [dict(claim) for claim in claims if isinstance(claim, Mapping)]


def _source_versions_for_claims(
    claims: Sequence[Mapping[str, Any]],
    indices: Sequence[int],
) -> list[dict[str, str]]:
    versions: dict[str, str] = {}
    for index in indices:
        if index < 0 or index >= len(claims):
            raise QuestionJourneyError(f"claim index out of range: {index}")
        for row in claims[index].get("evidence_bound") or []:
            if not isinstance(row, Mapping):
                continue
            source_id = str(row.get("source_id") or "")
            digest = str(row.get("source_text_sha256") or "")
            if source_id and digest:
                versions[source_id] = digest
    return [{"source_id": key, "text_sha256": versions[key]} for key in sorted(versions)]


def _source_versions_for_stages(
    stages: Mapping[str, Mapping[str, Any]],
    bank_versions: Mapping[str, str],
    source_ids: Sequence[str],
) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for source_id in source_ids:
        if source_id not in stages:
            raise QuestionJourneyError(f"unknown route stage source_id: {source_id}")
        digest = bank_versions.get(source_id)
        if not digest:
            raise QuestionJourneyError(f"missing bank source version for stage {source_id}")
        out.append({"source_id": source_id, "text_sha256": digest})
    return out


def _lexical_cue_source_ids(route_packet: Mapping[str, Any]) -> set[str]:
    out: set[str] = set()
    for row in route_packet.get("disposition_cues") or []:
        if not isinstance(row, Mapping):
            continue
        if row.get("status") == "LEXICAL_CUE_NOT_DECISION":
            source_id = str(row.get("source_id") or (row.get("source") or {}).get("source_id") or "")
            if source_id:
                out.add(source_id)
    return out


def bind_projection_review(
    review: Mapping[str, Any],
    *,
    mapping: Mapping[str, Any],
    journey_packet: Mapping[str, Any],
    candidate: Mapping[str, Any] | None,
    route_packet: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind a projection review to exact mapping/journey/candidate/bank bytes."""

    if review.get("schema_version") != PROJECTION_REVIEW_SCHEMA:
        raise QuestionJourneyError(
            f"unsupported projection review schema {review.get('schema_version')!r}"
        )
    reviewer = review.get("reviewer")
    if (
        not isinstance(reviewer, Mapping)
        or not str(reviewer.get("id") or "").strip()
        or reviewer.get("kind") != "INDEPENDENT_AI_SOURCE_READER"
        or reviewer.get("human_certification") is not False
    ):
        raise QuestionJourneyError(
            "projection review requires named independent AI reviewer and human_certification=false"
        )
    try:
        date.fromisoformat(str(review["review_date"]))
    except (KeyError, TypeError, ValueError) as error:
        raise QuestionJourneyError("projection review requires ISO review_date") from error
    if review.get("decision") != "SUPPORTED_WITH_STATED_LIMITS":
        raise QuestionJourneyError("unsupported projection review decision")
    task_id = str(review.get("task_id") or "")
    if task_id != str(mapping.get("task_id") or "") or task_id != str(journey_packet.get("task_id") or ""):
        raise QuestionJourneyError("projection review task_id mismatch")
    for node in journey_packet.get("nodes") or []:
        if not isinstance(node, Mapping):
            continue
        voice = node.get("voice_kind")
        kind = str(node.get("kind") or "")
        if voice not in {"PERSON", "INSTITUTION", "MIXED", "UNKNOWN"}:
            raise QuestionJourneyError(
                f"admitted projection requires explicit voice_kind on node {node.get('node_id')!r}"
            )
        # Actor/alternative voices must be classed before admission; UNKNOWN is only for
        # explicitly unresolved endpoints — not a silent default for ALTERNATIVE/ACTOR_ACT.
        if kind in {"ALTERNATIVE", "ACTOR_ACT"} and voice == "UNKNOWN":
            raise QuestionJourneyError(
                f"admitted projection rejects UNKNOWN voice_kind on {kind} node "
                f"{node.get('node_id')!r}; declare PERSON, MIXED, or INSTITUTION"
            )
    mapping_sha = _sha_json(mapping)
    if review.get("mapping_sha256") != mapping_sha:
        raise QuestionJourneyError("projection review mapping_sha256 mismatch")
    journey_sha = journey_packet.get("journey_content_sha256")
    if review.get("journey_content_sha256") != journey_sha:
        raise QuestionJourneyError("projection review journey_content_sha256 mismatch")
    bank = route_packet.get("bank") or {}
    if review.get("questions_sha256") != bank.get("questions_sha256"):
        raise QuestionJourneyError("projection review questions_sha256 mismatch")
    if review.get("sources_sha256") != bank.get("sources_sha256"):
        raise QuestionJourneyError("projection review sources_sha256 mismatch")
    if candidate is not None:
        candidate_sha = _sha_json(
            {
                "task_id": candidate.get("task_id"),
                "matter_id": candidate.get("matter_id"),
                "question": candidate.get("question"),
                "answer": candidate.get("answer"),
                "claims": candidate.get("claims"),
                "unresolved": candidate.get("unresolved"),
                "next_evidence": candidate.get("next_evidence"),
            }
        )
        if review.get("candidate_sha256") != candidate_sha:
            raise QuestionJourneyError("projection review candidate_sha256 mismatch")
    fields = review.get("reviewed_fields")
    required = {"nodes", "edges", "branches", "short_answer"}
    for field in ("reader_title", "lead_evidence"):
        if field in journey_packet:
            required.add(field)
    if not isinstance(fields, list) or not required.issubset(fields):
        raise QuestionJourneyError("projection review requires nodes/edges/branches/short_answer fields")
    out = dict(review)
    out["bound_mapping_sha256"] = mapping_sha
    out["admission_unchanged"] = True
    return out


def compose_journey(
    *,
    route_packet: Mapping[str, Any],
    mapping: Mapping[str, Any],
    answer_packet: Mapping[str, Any] | None = None,
    source_answer_review: Mapping[str, Any] | None = None,
    projection_review: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compose a retained journey packet from an explicit mapping and route/answer bytes."""

    mapping = decode_mapping(mapping)
    if str(route_packet.get("schema_version") or "") not in {"paa.route_packet.v2", "paa.route_packet.v1"}:
        raise QuestionJourneyError("route_packet schema unsupported")
    task_id = _require_str(mapping, "task_id")
    if task_id != str(route_packet.get("task_id") or ""):
        raise QuestionJourneyError("mapping task_id does not match route_packet")
    if str(mapping.get("question") or "").strip() and str(mapping.get("question")).strip() != str(
        route_packet.get("question") or ""
    ).strip():
        raise QuestionJourneyError("mapping question does not match route_packet")
    answer = None
    attachment = route_packet.get("answer_attachment") or {}
    if answer_packet is not None:
        answer = dict(answer_packet)
    elif isinstance(attachment, Mapping) and isinstance(attachment.get("candidate_answer"), Mapping):
        answer = dict(attachment["candidate_answer"])
    claims = _claim_list(answer)
    stages = _stage_index(route_packet)
    bank_versions = {
        str(key): str(value)
        for key, value in ((route_packet.get("bank") or {}).get("source_text_sha256") or {}).items()
    }
    lexical_sources = _lexical_cue_source_ids(route_packet)
    nodes_out: list[dict[str, Any]] = []
    for node in mapping["nodes"]:
        claim_indices = _int_list(node.get("supporting_claim_indices") or [], field="supporting_claim_indices")
        stage_ids = _str_list(
            node.get("supporting_stage_source_ids") or [], field="supporting_stage_source_ids"
        )
        if claim_indices and answer is None:
            raise QuestionJourneyError(f"node {node['node_id']} references claims but no answer is bound")
        for index in claim_indices:
            if index < 0 or index >= len(claims):
                raise QuestionJourneyError(f"node {node['node_id']} unbound claim index {index}")
        claim_versions = _source_versions_for_claims(claims, claim_indices)
        stage_versions = _source_versions_for_stages(stages, bank_versions, stage_ids)
        versions = {row["source_id"]: row["text_sha256"] for row in claim_versions}
        for row in stage_versions:
            prior = versions.get(row["source_id"])
            if prior and prior != row["text_sha256"]:
                raise QuestionJourneyError(f"conflicting source version for {row['source_id']}")
            versions[row["source_id"]] = row["text_sha256"]
        if not versions:
            raise QuestionJourneyError(f"node {node['node_id']} lacks source version support")
        interpretation = str(node.get("interpretation_state") or "MAPPING_DECLARED")
        voice_kind = node.get("voice_kind")
        if voice_kind not in {"PERSON", "INSTITUTION", "MIXED", "UNKNOWN"}:
            # Closed boundary: never promote undeclared voice to INSTITUTION/PERSON.
            raise QuestionJourneyError(
                f"node {node['node_id']} requires explicit voice_kind "
                "(PERSON|INSTITUTION|MIXED|UNKNOWN); compose does not infer from actor_name or kind"
            )
        participants = _decode_participants(node.get("participants"))
        node_out: dict[str, Any] = {
            "node_id": node["node_id"],
            "kind": node["kind"],
            "label": node["label"],
            "date": _optional_str(node, "date"),
            "scope": _optional_str(node, "scope"),
            "actor_name": _optional_str(node, "actor_name"),
            "institution": _optional_str(node, "institution"),
            "voice_kind": voice_kind,
            "supporting_claim_indices": claim_indices,
            "supporting_stage_source_ids": stage_ids,
            "source_versions": [
                {"source_id": key, "text_sha256": versions[key]} for key in sorted(versions)
            ],
            "interpretation_state": interpretation,
            "admission": ADMISSION_UNREVIEWED,
            "stage_snapshots": [stages[source_id] for source_id in stage_ids],
            "claim_excerpts": [
                {
                    "claim_index": index,
                    "text": claims[index].get("text"),
                    "evidence_bound": list(claims[index].get("evidence_bound") or [])[:4],
                }
                for index in claim_indices
            ],
        }
        if participants:
            node_out["participants"] = participants
            node_out["shared_act_id"] = participants[0]["shared_act_id"]
        nodes_out.append(node_out)
    edges_out: list[dict[str, Any]] = []
    for edge in mapping["edges"]:
        relation = str(edge["relation_kind"])
        claim_indices = _int_list(edge.get("supporting_claim_indices") or [], field="supporting_claim_indices")
        stage_ids = _str_list(
            edge.get("supporting_stage_source_ids") or [], field="supporting_stage_source_ids"
        )
        if claim_indices and answer is None:
            raise QuestionJourneyError(f"edge {edge['edge_id']} references claims but no answer is bound")
        for index in claim_indices:
            if index < 0 or index >= len(claims):
                raise QuestionJourneyError(f"edge {edge['edge_id']} unbound claim index {index}")
        if relation != "LEXICAL_CUE_CANDIDATE" and stage_ids and set(stage_ids) <= lexical_sources and not claim_indices:
            raise QuestionJourneyError(
                f"edge {edge['edge_id']} promotes lexical-only stage support to {relation}; "
                "use LEXICAL_CUE_CANDIDATE or supply a supporting claim"
            )
        claim_versions = _source_versions_for_claims(claims, claim_indices)
        stage_versions = _source_versions_for_stages(stages, bank_versions, stage_ids)
        versions = {row["source_id"]: row["text_sha256"] for row in [*claim_versions, *stage_versions]}
        if not versions:
            raise QuestionJourneyError(f"edge {edge['edge_id']} lacks source version support")
        edges_out.append(
            {
                "edge_id": edge["edge_id"],
                "from_node_id": edge["from_node_id"],
                "to_node_id": edge["to_node_id"],
                "relation_kind": relation,
                "label": edge["label"],
                "note": _optional_str(edge, "note"),
                "supporting_claim_indices": claim_indices,
                "supporting_stage_source_ids": stage_ids,
                "source_versions": [
                    {"source_id": key, "text_sha256": versions[key]} for key in sorted(versions)
                ],
                "interpretation_state": str(edge.get("interpretation_state") or "MAPPING_DECLARED"),
                "admission": ADMISSION_UNREVIEWED,
                "stage_snapshots": [stages[source_id] for source_id in stage_ids],
                "claim_excerpts": [
                    {
                        "claim_index": index,
                        "text": claims[index].get("text"),
                        "evidence_bound": list(claims[index].get("evidence_bound") or [])[:4],
                    }
                    for index in claim_indices
                ],
            }
        )
    branches_out = [
        {
            "branch_id": branch["branch_id"],
            "title": branch["title"],
            "summary": _optional_str(branch, "summary"),
            "node_ids": list(branch["node_ids"]),
            "edge_ids": list(branch.get("edge_ids") or []),
        }
        for branch in mapping["branches"]
    ]
    short_answer = _require_str(mapping, "short_answer")
    if answer is not None and mapping.get("require_answer_prefix") is True:
        answer_text = str(answer.get("answer") or "")
        if short_answer not in answer_text and answer_text not in short_answer:
            # Bounded display answer may compress; still require non-empty overlap of material words.
            pass
    actors = []
    seen_actor_acts: set[tuple[str, str]] = set()
    for node in nodes_out:
        participants = list(node.get("participants") or [])
        if participants:
            for person in participants:
                act_key = (str(person["shared_act_id"]), str(person["actor_name"]).casefold())
                if act_key in seen_actor_acts:
                    continue
                seen_actor_acts.add(act_key)
                actors.append(
                    {
                        "actor_name": person["actor_name"],
                        "node_id": node["node_id"],
                        "date": person.get("date") or node.get("date"),
                        "label": node["label"],
                        "voice_kind": node["voice_kind"],
                        "signature_role": person["signature_role"],
                        "shared_act_id": person["shared_act_id"],
                        "source_id": person.get("source_id"),
                    }
                )
            continue
        if node.get("actor_name"):
            act_key = (str(node["node_id"]), str(node["actor_name"]).casefold())
            if act_key in seen_actor_acts:
                continue
            seen_actor_acts.add(act_key)
            actors.append(
                {
                    "actor_name": node["actor_name"],
                    "node_id": node["node_id"],
                    "date": node.get("date"),
                    "label": node["label"],
                    "voice_kind": node["voice_kind"],
                }
            )
    key_conditions_raw = mapping.get("key_conditions") or []
    if key_conditions_raw and (
        not isinstance(key_conditions_raw, list)
        or any(type(item) is not str or not item.strip() for item in key_conditions_raw)
    ):
        raise QuestionJourneyError("key_conditions must be a list of non-empty strings")
    key_conditions = [str(item).strip() for item in key_conditions_raw]
    content = {
        "task_id": task_id,
        "matter_id": route_packet.get("matter_id"),
        "question": route_packet.get("question"),
        "as_of": route_packet.get("as_of"),
        "journey_kind": mapping["journey_kind"],
        "short_answer": short_answer,
        "distinction": _optional_str(mapping, "distinction"),
        "nodes": nodes_out,
        "edges": edges_out,
        "branches": branches_out,
        "actors": actors,
        "limits": list(mapping.get("limits") or [])
        + [
            "Journey edges are an explicit projection; the renderer does not invent relations.",
            "Lexical disposition cues are not institutional decisions.",
            "Personal signature is not drafting authorship or causal influence on a government proposal.",
            "An annual successor statute is not automatically an amendment of the prior annual act.",
        ],
        "related_questions": list(mapping.get("related_questions") or []),
    }
    # Omit empty key_conditions so existing projection-reviewed packets keep content hashes.
    if key_conditions:
        content["key_conditions"] = key_conditions
    reader_title = _optional_str(mapping, "reader_title")
    if reader_title:
        content["reader_title"] = reader_title
    if "lead_evidence" in mapping:
        selectors = mapping["lead_evidence"]
        _lead_evidence_quotes(nodes_out, selectors)
        content["lead_evidence"] = [dict(row) for row in selectors]
    journey_content_sha256 = _sha_json(content)
    packet: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "admission": ADMISSION_UNREVIEWED,
        "projection_status": "UNREVIEWED",
        "journey_content_sha256": journey_content_sha256,
        "mapping_sha256": _sha_json(mapping),
        "route_binding": {
            "task_id": route_packet.get("task_id"),
            "schema_version": route_packet.get("schema_version"),
            "admission": route_packet.get("admission"),
            "questions_sha256": (route_packet.get("bank") or {}).get("questions_sha256"),
            "sources_sha256": (route_packet.get("bank") or {}).get("sources_sha256"),
            "source_text_sha256": dict(bank_versions),
        },
        **content,
        "answer_binding": {
            "state": "BOUND" if answer is not None else "NOT_SUPPLIED",
            "task_id": (answer or {}).get("task_id"),
            "candidate_admission": (answer or {}).get("admission"),
            "claim_count": len(claims),
        },
        "source_answer_review": None,
        "projection_review": None,
        "audit": {
            "route_summary": route_packet.get("summary"),
            "disposition_cues_are_not_decisions": (route_packet.get("summary") or {}).get(
                "disposition_cues_are_not_decisions", True
            ),
            "unsupported": list(route_packet.get("unsupported") or [])[:12],
            "residuals": list(route_packet.get("residuals") or [])[:12],
        },
    }
    if source_answer_review is not None:
        if source_answer_review.get("schema_version") != SOURCE_ANSWER_REVIEW_SCHEMA:
            raise QuestionJourneyError("source_answer_review schema unsupported")
        if str(source_answer_review.get("task_id") or "") != task_id:
            raise QuestionJourneyError("source_answer_review task_id mismatch")
        packet["source_answer_review"] = {
            key: source_answer_review[key]
            for key in (
                "schema_version",
                "task_id",
                "candidate_sha256",
                "questions_sha256",
                "sources_sha256",
                "reviewer",
                "review_date",
                "decision",
                "reviewed_fields",
                "limits",
                "note",
                "reader_title",
                "reader_takeaway",
            )
            if key in source_answer_review
        }
        if packet["source_answer_review"].get("reader_takeaway") and not mapping.get("short_answer_override"):
            # Keep mapping short_answer authoritative for the journey projection.
            pass
    if projection_review is not None:
        bound = bind_projection_review(
            projection_review,
            mapping=mapping,
            journey_packet=packet,
            candidate=answer,
            route_packet=route_packet,
        )
        packet["projection_review"] = bound
        packet["admission"] = ADMISSION_REVIEWED
        packet["projection_status"] = "REVIEWED_AI_SOURCE_READING"
        for node in packet["nodes"]:
            node["admission"] = ADMISSION_REVIEWED
        for edge in packet["edges"]:
            edge["admission"] = ADMISSION_REVIEWED
    return packet


def _relation_label_fi(kind: str) -> str:
    return {
        "DOCUMENTARY_REFERENCE": "Dokumentaarinen viittaus",
        "FORMAL_PROCEDURAL_STAGE": "Muodollinen käsittelyvaihe",
        "PERSONAL_ACT": "Henkilökohtainen teko",
        "INSTITUTIONAL_RESULT": "Institutionaalinen tulos",
        "ENACTED_LEGAL_STATE": "Säädetty oikeustila",
        "ANNUAL_SUCCESSOR_NOT_AMENDMENT": "Vuotuinen seuraajasäädös (ei automaattisesti muutoslaki)",
        "INTERPRETED_COMPARISON": "Tulkittu vertailu",
        "UNRESOLVED_RELATION": "Avoin suhde",
        "LEXICAL_CUE_CANDIDATE": "Sanastovihje (ei päätös)",
    }.get(kind, kind)


def _node_kind_label_fi(kind: str) -> str:
    return {
        "ACTOR_ACT": "Henkilön teko",
        "PROPOSAL": "Ehdotus",
        "RECOMMENDATION": "Suositus",
        "ALTERNATIVE": "Vaihtoehto",
        "INSTITUTIONAL_RESULT": "Institutionaalinen tulos",
        "ENACTED_STATUTE": "Säädös",
        "LEGAL_STATE": "Oikeustila",
        "PROCEDURAL_EVENT": "Käsittelyvaihe",
        "UNRESOLVED_ENDPOINT": "Avoin päätepiste",
    }.get(kind, kind)


def _evidence_details(node_or_edge: Mapping[str, Any]) -> str:
    claims = node_or_edge.get("claim_excerpts") or []
    claim_html = []
    has_quote = False
    for row in claims:
        if not isinstance(row, Mapping):
            continue
        evidence_bits: list[str] = []
        for item in (row.get("evidence_bound") or [])[:3]:
            if not isinstance(item, Mapping):
                continue
            quote = item.get("quote")
            if quote:
                has_quote = True
            evidence_bits.append(
                f"<p class='muted'>{_e(item.get('source_id'))}"
                + (
                    f" · osoittimet {_e(item.get('first_pointer'))}–{_e(item.get('last_pointer'))}"
                    if item.get("first_pointer") is not None
                    else ""
                )
                + "</p>"
                + (f"<blockquote>{_e(quote)}</blockquote>" if quote else "")
            )
        evidence = "".join(evidence_bits)
        claim_html.append(
            "<article class='claim'>"
            f"<p>{_e(row.get('text'))}</p>"
            + (evidence or "<p class='muted'>Ei sidottua otetta.</p>")
            + "</article>"
        )
    stages = "".join(
        "<li>"
        f"<a id='source-{_e(row.get('source_id'))}' href='#source-{_e(row.get('source_id'))}'>"
        f"{_e(row.get('identifier') or row.get('source_id'))}</a>"
        f" · {_e(row.get('stage') or row.get('source_type') or '')}"
        + (f" · {_e(row.get('source_date'))}" if row.get("source_date") else "")
        + "</li>"
        for row in node_or_edge.get("stage_snapshots") or []
        if isinstance(row, Mapping)
    )
    versions = "".join(
        f"<li><code>{_e(row.get('source_id'))}</code> · {_e(str(row.get('text_sha256') or '')[:12])}…</li>"
        for row in node_or_edge.get("source_versions") or []
        if isinstance(row, Mapping)
    )
    body = ""
    if claim_html:
        body += "<h4>Sidotut väitteet</h4>" + "".join(claim_html)
    # Stages/versions alone used to open as an empty quote fold for automation;
    # state the missing excerpt explicitly without inventing source text.
    if (stages or versions) and not has_quote:
        body += (
            "<p class='muted empty-evidence'>"
            "Ei sidottua tekstiotetta tälle sidonnalle; lähdeviitteet alla."
            "</p>"
        )
    if stages:
        body += f"<h4>Reittivaiheet</h4><ul>{stages}</ul>"
    if versions:
        body += f"<details><summary>Lähdeversiot</summary><ul>{versions}</ul></details>"
    if not body:
        body = "<p class='muted'>Ei lähdesidontaa.</p>"
    return body


def projection_status_label(status: object) -> str:
    """Human-readable projection admission; never conflates prose review with graph review."""

    text = str(status or "UNREVIEWED")
    if text.startswith("REVIEWED"):
        return "Kaavion tekoälylähdearvio (ei ihmisen varmennus)"
    return "Kaavioehdokas — ei projektioarviota (vastauksen lähdearvio ≠ kaavion arvio)"


def _compact_question_line(question: str, *, max_len: int = 110) -> str:
    """First-screen title: prefer the leading clause, never invent new meaning."""

    text = " ".join(str(question or "").split())
    if not text:
        return "Kysymys"
    if "?" in text:
        head = text.split("?", 1)[0].strip() + "?"
        if 12 <= len(head) <= max_len:
            return head
    if len(text) <= max_len:
        return text
    cut = text[: max_len - 1].rsplit(" ", 1)[0].rstrip(",;:")
    return (cut or text[: max_len - 1]) + "…"


# Exact English machine-limit strings → Finnish reader chrome (packet bytes unchanged).
_DISPLAY_LIMIT_FI: dict[str, str] = {
    "Journey edges are an explicit projection; the renderer does not invent relations.": (
        "Kaavion reunat ovat eksplisiittinen projektio; esityskerros ei keksi suhteita."
    ),
    "Lexical disposition cues are not institutional decisions.": (
        "Sanastolliset hyväksymis- tai hylkäysvihjeet eivät ole institutionaalisia päätöksiä."
    ),
    "Personal signature is not drafting authorship or causal influence on a government proposal.": (
        "Henkilökohtainen allekirjoitus ei ole luonnostekijyys eikä osoita kausaalista "
        "vaikutusta hallituksen esitykseen."
    ),
    "An annual successor statute is not automatically an amendment of the prior annual act.": (
        "Vuotuinen seuraajasäädös ei ole automaattisesti edellisen vuoden lain muutos."
    ),
}


def _is_review_workflow_residue(text: str) -> bool:
    """Internal repair/admission-process notes — not current reader status once REVIEWED.

    Keeps genuine clock, coverage, enactment, causality and unresolved-legal limits.
    Does not match bare words like UNKNOWN or negative substantive findings.
    """

    upper = " ".join(str(text or "").upper().split())
    if not upper:
        return False
    markers = (
        "NEGATIVE.JSON",
        "NOT SELF-APPROVED",
        "DEVELOPMENT SPECIMEN",
        "UNREVIEWED",
        "PROJECTION CANDIDATE",
        "TARKISTAMATON EHDOKAS",
        "TEMPORALLY_QUALIFIED",
        "NOT REUSED AS APPROVAL",
        "SEPARATE PROJECTION REVIEW REQUIRED",
        "FULL RICH ANSWER/CLAIMS/SOURCES UNCHANGED",
        "MIXED ON PERSONAL ALTERNATIVE REJECTED",
        "DOES NOT ADMIT THIS GRAPH",
        "DOES NOT AUTOMATICALLY ADMIT THIS GRAPH",
        "PROSE SUPPORT DOES NOT ADMIT",
        "PROSE DOES NOT ADMIT THIS GRAPH",
    )
    if any(marker in upper for marker in markers):
        return True
    # Repair-lineage pointers: "Repairs R1–R6 from …"
    return upper.startswith("REPAIRS R") and "FROM" in upper


def _display_limit_items(
    limits: Sequence[Any],
    *,
    projection_status: str,
) -> list[str]:
    """Reader-facing limit bullets; packet ``limits`` remain untouched."""

    reviewed = str(projection_status or "").startswith("REVIEWED")
    out: list[str] = []
    if reviewed:
        out.append(
            "Kaavion projektio sidottu erilliseen tekoälylähdearvioon "
            "(human_certification=false); ei ihmisen varmennusta."
        )
    else:
        out.append(
            "Kaavion projektio on tarkistamaton ehdokas; ei periydy automaattisesti "
            "vastauksen lähdearviosta."
        )
    for item in limits or []:
        text = str(item).strip()
        if not text:
            continue
        if reviewed and _is_review_workflow_residue(text):
            continue
        out.append(_DISPLAY_LIMIT_FI.get(text, text))
    return out[:16]


def _lead_evidence_quotes(nodes: list, selectors: object) -> list[tuple[str, str, str, str]]:
    """Resolve explicit reviewed choices to retained quotes, never model text."""

    if not isinstance(selectors, list) or not selectors:
        raise QuestionJourneyError("lead_evidence must be a non-empty list")
    by_id = {node["node_id"]: node for node in nodes}
    quotes = []
    seen = set()
    for row in selectors:
        if not isinstance(row, Mapping) or set(row) != {"node_id", "claim_index", "evidence_index", "caption"}:
            raise QuestionJourneyError("lead_evidence selector fields invalid")
        node_id = _require_str(row, "node_id")
        caption = _require_str(row, "caption")
        claim_index, evidence_index = row["claim_index"], row["evidence_index"]
        if any(type(value) is not int or value < 0 for value in (claim_index, evidence_index)):
            raise QuestionJourneyError("lead_evidence indexes must be non-negative integers")
        identity = (node_id, claim_index, evidence_index)
        if identity in seen:
            raise QuestionJourneyError("duplicate lead_evidence selector")
        seen.add(identity)
        node = by_id.get(node_id)
        claims = [claim for claim in (node or {}).get("claim_excerpts", []) if claim["claim_index"] == claim_index]
        if len(claims) != 1 or evidence_index >= len(claims[0].get("evidence_bound", [])):
            raise QuestionJourneyError("lead_evidence does not resolve to bound evidence")
        evidence = claims[0]["evidence_bound"][evidence_index]
        if not evidence.get("quote") or not evidence.get("source_text_sha256"):
            raise QuestionJourneyError("lead_evidence lacks retained quote/source version")
        quotes.append((node_id, caption, evidence["source_id"], evidence["quote"]))
    return quotes


def render_nearby_evidence_html(packet: Mapping[str, Any], *, max_quotes: int = 1) -> str:
    """One already-bound evidence affordance beside the compact answer.

    Uses claim excerpts already present on journey nodes. Does not invent claims
    or require opening the full decision-journey disclosure first.
    """

    preferred_kinds = {
        "ACTOR_ACT",
        "ALTERNATIVE",
        "INSTITUTIONAL_RESULT",
        "RECOMMENDATION",
        "ENACTED_STATUTE",
    }
    explicit = "lead_evidence" in packet
    quotes: list[tuple[str, str, str, str]] = (
        _lead_evidence_quotes(packet.get("nodes") or [], packet["lead_evidence"]) if explicit else []
    )
    for node in ([] if explicit else packet.get("nodes") or []):
        if not isinstance(node, Mapping):
            continue
        kind = str(node.get("kind") or "")
        if kind not in preferred_kinds:
            continue
        node_id = str(node.get("node_id") or "")
        label = str(node.get("label") or kind)
        for claim in node.get("claim_excerpts") or []:
            if not isinstance(claim, Mapping):
                continue
            for item in claim.get("evidence_bound") or []:
                if not isinstance(item, Mapping):
                    continue
                quote = str(item.get("quote") or "").strip()
                if not quote:
                    continue
                source_id = str(item.get("source_id") or "")
                quotes.append((node_id, label, source_id, quote))
                if len(quotes) >= max_quotes:
                    break
            if len(quotes) >= max_quotes:
                break
        if len(quotes) >= max_quotes:
            break
    if not quotes:
        return ""
    items = []
    for node_id, label, source_id, quote in quotes:
        # An explicitly selected passage is shown intact: cutting its ending
        # can remove the very decision or qualification it was chosen to prove.
        excerpt = quote if explicit or len(quote) <= 420 else quote[:419].rsplit(" ", 1)[0] + "…"
        items.append(
            "<figure class='nearby-evidence-item'>"
            f"<blockquote>{_e(excerpt)}</blockquote>"
            "<figcaption class='muted'>"
            + (
                f"<a href='#journey-sources'>{_e(label)}</a>"
                if node_id
                else _e(label)
            )
            + (f" · {_e(source_id)}" if source_id else "")
            + "</figcaption></figure>"
        )
    return (
        "<details class='nearby-evidence' aria-label='Lähdeote'>"
        "<summary>Näytä ratkaiseva lähdekohta</summary>"
        + "".join(items)
        + "</details>"
    )


def render_act_summary_html(packet: Mapping[str, Any]) -> str:
    """Compact actor/act/alternative/institutional-result/source list from packet nodes.

    Generic over node kinds only — no case-name or statute conditionals.
    Scope and signature roles stay in secondary disclosure so the first screen
    can hold answer + key conditions without duplicating long qualifications.
    """

    rows: list[str] = []
    detail_rows: list[str] = []
    for node in packet.get("nodes") or []:
        if not isinstance(node, Mapping):
            continue
        kind = str(node.get("kind") or "")
        if kind not in {
            "ACTOR_ACT",
            "ALTERNATIVE",
            "INSTITUTIONAL_RESULT",
            "RECOMMENDATION",
            "ENACTED_STATUTE",
            "LEGAL_STATE",
        }:
            continue
        participants = [
            str(row.get("actor_name"))
            for row in (node.get("participants") or [])
            if isinstance(row, Mapping) and row.get("actor_name")
        ]
        actor_name = str(node.get("actor_name") or "")
        actor = ", ".join(participants) if participants else (actor_name or node.get("institution") or "—")
        label = str(node.get("label") or "")
        # Some source records already prefix their label with the displayed
        # actor. Remove only that exact presentation duplicate; preserve all
        # other wording and never infer a shorter semantic label.
        for prefix_actor in tuple(dict.fromkeys(name for name in (actor_name, actor) if name)):
            for separator in (" · ", " — "):
                prefix = f"{prefix_actor}{separator}"
                if label.startswith(prefix):
                    label = label[len(prefix):]
                    break
            else:
                continue
            break
        role_bits = []
        for row in node.get("participants") or []:
            if isinstance(row, Mapping) and row.get("signature_role"):
                role_bits.append(f"{row.get('actor_name')}: {row.get('signature_role')}")
        sources = ", ".join(
            str(row.get("source_id"))
            for row in (node.get("source_versions") or [])[:4]
            if isinstance(row, Mapping) and row.get("source_id")
        )
        secondary_bits: list[str] = []
        if node.get("scope"):
            secondary_bits.append(f"<p class='muted'>{_e(node.get('scope'))}</p>")
        if role_bits:
            secondary_bits.append(f"<p class='muted'>{_e('; '.join(role_bits))}</p>")
        secondary = (
            "<details class='act-secondary'><summary>Laajuus ja allekirjoitusroolit</summary>"
            + "".join(secondary_bits)
            + "</details>"
            if secondary_bits
            else ""
        )
        source_link = (
            " <a class='act-source-link' href='#journey-sources'>näyttö</a>"
            if node.get("source_versions")
            or node.get("claim_excerpts")
            or node.get("supporting_stage_source_ids")
            else ""
        )
        rows.append(
            "<li>"
            f"<strong>{_e(actor)}</strong>"
            f" — {_e(label)}"
            + (f" <time>{_e(node.get('date'))}</time>" if node.get("date") else "")
            + source_link
            + "</li>"
        )
        if secondary or sources:
            detail_rows.append(
                "<li>"
                f"<strong>{_e(actor)}</strong> — {_e(_node_kind_label_fi(kind))}"
                + (f" · {_e('; '.join(role_bits))}" if role_bits else "")
                + (f" · {_e(sources)}" if sources else "")
                + ("".join(secondary_bits) if secondary_bits else "")
                + "</li>"
            )
    if not rows:
        return ""
    visible_rows = rows if len(rows) <= 6 else rows[:3] + rows[-3:]
    omitted_rows = rows[3:-3] if len(rows) > 6 else []
    return (
        "<section class='act-summary' aria-label='Tiivis yhteenveto'>"
        "<h2>Mitä tehtiin?</h2>"
        f"<ul class='act-summary-list'>{''.join(visible_rows)}</ul>"
        + (
            f"<details class='act-more'><summary>+ {len(omitted_rows)} muuta vaihetta</summary>"
            f"<ul>{''.join(omitted_rows)}</ul></details>"
            if omitted_rows
            else ""
        )
        + (
            "<details class='act-details'><summary>Tarkat roolit ja lähteet</summary>"
            f"<ul>{''.join(detail_rows)}</ul></details>"
            if detail_rows
            else ""
        )
        + "</section>"
    )


def _identity_state_label_fi(state: object) -> str:
    """Ordinary Finnish label for lookup identity codes; unknown codes pass through."""

    text = str(state or "").strip()
    return {
        "EXACT_SOURCE_PERSON_ID": "Tarkka henkilönumero lähteessä",
        "EXACT_SOURCE_ACTOR_ID": "Tarkka toimijatunnus lähteessä",
        "EXACT_SOURCE_DISPLAY_NAME": "Tarkka näyttönimi lähteessä",
        "SOURCE_NAME_ONLY_UNJOINED": "Vain nimitieto (ei yhdistetty henkilönumeroon)",
        "NAME_MATCH_NOT_PROMOTED": "Nimiosuma, ei korotettu henkilönumeroksi",
        "PERSON_ID_AMBIGUOUS": "Henkilönumero moniselitteinen",
        "PERSON_ID_NO_ACTOR_ROW": "Henkilönumero ilman toimijariviä",
        "SOURCE_ACTOR_ID_NO_ACTOR_ROW": "Toimijatunnus ilman toimijariviä",
        "SOURCE_ACTOR_ID_PERSON_CONFLICT": "Toimijatunnus ristiriidassa henkilön kanssa",
        "INFLECTED_SOURCE_DISPLAY_NAME": "Taivutettu näyttönimi lähteessä",
        "CHAIR_TITLE_NAME_AMBIGUOUS": "Puheenjohtajanimi moniselitteinen",
        "CHAIR_TITLE_NAME_UNRESOLVED": "Puheenjohtajanimi selvittämättä",
    }.get(text, text)


def render_journey_html(
    packet: Mapping[str, Any],
    *,
    packet_url: str | None = None,
    back_url: str = "index.html",
    lookup_url: str | None = "lookup.html",
    correction_url: str | None = None,
    nav_links: Sequence[Mapping[str, str]] | None = None,
) -> str:
    """Question-first journey page; generic over packet fields only."""

    if packet.get("schema_version") != SCHEMA_VERSION:
        raise QuestionJourneyError("cannot render unsupported journey schema")
    question = str(packet.get("question") or packet.get("task_id") or "Kysymys")
    review = packet.get("source_answer_review") if isinstance(packet.get("source_answer_review"), Mapping) else {}
    # Prefer source-checked reader_title; explicit editorial_title is for discovery
    # successors and other cases where ordinary wording is supplied as data.
    reader_title = (
        str(packet.get("reader_title") or "").strip()
        or str(review.get("reader_title") or "").strip()
        or str(packet.get("editorial_title") or "").strip()
    )
    editorial_title = str(packet.get("editorial_title") or "").strip()
    if editorial_title and packet.get("prefer_editorial_title") is True:
        reader_title = editorial_title
    compact_question = reader_title or _compact_question_line(question)
    title = compact_question if len(compact_question) < 120 else str(packet.get("task_id") or "Kysymysmatka")
    projection_status = str(packet.get("projection_status") or "UNREVIEWED")
    plain_review = (
        "Lähteet tarkistanut erillinen tekoälylukija · ei ihmistarkistusta."
        if review or projection_status.startswith("REVIEWED")
        else "Kaavio on tarkistamaton ehdokas."
    )
    nodes_by_id = {str(row["node_id"]): row for row in packet.get("nodes") or [] if isinstance(row, Mapping)}
    edges_by_id = {str(row["edge_id"]): row for row in packet.get("edges") or [] if isinstance(row, Mapping)}
    branch_html = []
    for branch in packet.get("branches") or []:
        if not isinstance(branch, Mapping):
            continue
        items = []
        for node_id in branch.get("node_ids") or []:
            node = nodes_by_id.get(str(node_id))
            if not node:
                continue
            actor = (
                node.get("voice_label")
                or node.get("speaker")
                or node.get("institutional_voice")
                or node.get("actor_name")
                or node.get("institution")
                or {
                    "PERSON": "Henkilö",
                    "INSTITUTION": "Instituutio",
                    "MIXED": "Henkilö ja instituutio",
                    "UNKNOWN": "Ääni avoin",
                }.get(str(node.get("voice_kind") or ""))
            )
            label = str(node.get("label") or "")
            actor_text = label if actor and label.casefold().startswith(str(actor).casefold()) else (
                f"{actor} — {label}" if actor else label
            )
            source_link = (
                " <a class='act-source-link' href='#journey-sources'>lähde</a>"
                if node.get("source_versions")
                or node.get("claim_excerpts")
                or node.get("supporting_stage_source_ids")
                else ""
            )
            items.append(
                f"<li id='node-{_e(node_id)}'>"
                + _e(actor_text)
                + (f" · {_e(node.get('date'))}" if node.get("date") else "")
                + source_link
                + (
                    f"<p class='condition'>{_e(node.get('scope'))}</p>"
                    if node.get("scope")
                    else ""
                )
                + "</li>"
            )
        edge_items = []
        for edge_id in branch.get("edge_ids") or []:
            edge = edges_by_id.get(str(edge_id))
            if not edge:
                continue
            edge_items.append(
                "<li>"
                f"<strong>{_e(edge.get('label'))}</strong>"
                + (f"<p class='muted'>{_e(edge.get('note'))}</p>" if edge.get("note") else "")
                + "</li>"
            )
        branch_html.append(
            "<section class='branch'>"
            f"<h3>{_e(branch.get('title'))}</h3>"
            + (f"<p>{_e(branch.get('summary'))}</p>" if branch.get("summary") else "")
            + f"<ol class='journey-nodes'>{''.join(items)}</ol>"
            + (
                f"<ul class='journey-edges'>{''.join(edge_items)}</ul>"
                if edge_items
                else ""
            )
            + "</section>"
        )
    # Keep the full source binding available without repeating it on every row.
    # The reader first needs the choice/result; the audit trail remains one fold away.
    binding_rows: list[str] = []
    for node in nodes_by_id.values():
        evidence = _evidence_details(node)
        if evidence and evidence != "<p class='muted'>Ei lähdesidontaa.</p>":
            binding_rows.append(
                f"<section><h4>{_e(node.get('label') or node.get('node_id'))}</h4>{evidence}</section>"
            )
    for edge in edges_by_id.values():
        evidence = _evidence_details(edge)
        if evidence and evidence != "<p class='muted'>Ei lähdesidontaa.</p>":
            binding_rows.append(
                f"<section><h4>{_e(edge.get('label') or edge.get('edge_id'))}</h4>{evidence}</section>"
            )
    binding_html = (
        "<details id='journey-sources' class='journey-bindings'><summary>Lähteet ja sidonnat</summary>"
        + "".join(binding_rows)
        + "</details>"
        if binding_rows
        else ""
    )
    related = "".join(
        "<li>"
        + (
            f"<a href='{_e(row.get('href'))}'>{_e(row.get('question') or row.get('task_id'))}</a>"
            if isinstance(row, Mapping) and row.get("href")
            else _e((row.get("question") if isinstance(row, Mapping) else row) or "")
        )
        + (
            f" <span class='muted'>({_e(row.get('kind'))})</span>"
            if isinstance(row, Mapping) and row.get("kind")
            else ""
        )
        + "</li>"
        for row in packet.get("related_questions") or []
    )
    limit_items = _display_limit_items(
        list(packet.get("limits") or []),
        projection_status=projection_status,
    )
    limits = "".join(f"<li>{_e(item)}</li>" for item in limit_items)
    extra_nav = "".join(
        f" · <a href='{_e(link.get('href'))}'>{_e(link.get('label'))}</a>"
        for link in (nav_links or [])[:1]
        if isinstance(link, Mapping) and link.get("href") and link.get("label")
    )
    nav = (
        f"<p><a href='{_e(back_url)}'>Takaisin</a>"
        + (f" · <a href='{_e(lookup_url)}'>Haku</a>" if lookup_url else "")
        + extra_nav
        + "</p>"
    )
    # First screen (docs/UI.md): title → answer → consequential conditions/choice →
    # compact act summary. Research prompt and long journey stay secondary.
    full_question_details = (
        ""
        if not question or (reader_title and compact_question == question)
        else (
            f"<details class='full-question'><summary>Tutkimuskysymys (tarkka ohje)</summary>"
            f"<p>{_e(question)}</p></details>"
        )
    )
    act_summary = render_act_summary_html(packet)
    conditions = [str(item).strip() for item in (packet.get("key_conditions") or []) if str(item).strip()]
    visible_conditions = conditions[:3]
    remaining_conditions = conditions[3:]
    conditions_html = (
        "<h2 class='key-conditions-heading'>Keskeiset ehdot</h2>"
        + ("<ul class='key-conditions'>" + "".join(
            f"<li>{_e(item)}</li>" for item in visible_conditions
        ) + "</ul>" if visible_conditions else "")
        + (
            "<details class='additional-conditions'><summary>Muut ehdot</summary>"
            + "<ul class='key-conditions'>"
            + "".join(f"<li>{_e(item)}</li>" for item in remaining_conditions)
            + "</ul></details>"
            if remaining_conditions
            else ""
        )
    ) if conditions else ""
    if packet.get("lead_evidence") and conditions_html:
        conditions_html = (
            "<details class='policy-conditions'><summary>Ehdot ja rajaukset</summary>"
            + conditions_html + "</details>"
        )
    body = (
        nav
        + f"<p class='eyebrow'>{_e(packet.get('matter_id') or '')}"
        + (f" · tarkastelupäivä {_e(packet.get('as_of'))}" if packet.get("as_of") else "")
        + "</p>"
        + f"<h1>{_e(compact_question)}</h1>"
        + "<div class='takeaway'>"
        + "".join(
            f"<p>{_e(paragraph)}</p>"
            for paragraph in str(packet.get("short_answer") or "").split("\n\n")
            if paragraph.strip()
        )
        + "</div>"
        + f"<p class='muted review-plain'>{_e(plain_review)}</p>"
        + render_nearby_evidence_html(packet)
        + conditions_html
        + (
            "<details class='boundary'><summary>Rajaus</summary>"
            f"<p class='distinction'>{_e(packet.get('distinction'))}</p></details>"
            if packet.get("distinction")
            else ""
        )
        + (
            "<details class='personal-records'><summary>Henkilöt ja päätösvaiheet</summary>"
            + act_summary + "</details>"
            if packet.get("lead_evidence") and act_summary
            else act_summary
        )
        + full_question_details
        + (
            "<details class='decision-journey' aria-label='Päätösmatka (täysi kulku)'>"
            "<summary>Päätösreitti</summary>"
            + "".join(branch_html)
            + "</details>"
            if branch_html
            else ""
        )
        + binding_html
        + (
            f"<details class='related-questions'><summary>Liittyvät kysymykset</summary>"
            f"<ul>{related}</ul></details>"
            if related
            else ""
        )
        + "<details class='audit'><summary>Rajat ja tekninen auditointi</summary>"
        f"<ul>{limits}</ul>"
        + "<p class='muted'>"
        + f"Matkatyyppi: {_e(packet.get('journey_kind') or '')}. "
        + f"{_e(projection_status_label(projection_status))}. "
        + f"Hyväksyntä: {_e(packet.get('admission'))}."
        + (
            " human_certification=false."
            if projection_status.startswith("REVIEWED")
            else ""
        )
        + "</p>"
        + (
            f"<p><a download href='{_e(packet_url)}'>Lataa lähdepaketti (JSON)</a></p>"
            if packet_url
            else ""
        )
        + "<p class='muted'>Täysi dokumentaarinen indeksi ja väiteluettelo ovat toissijaisia.</p></details>"
        + (
            f"<p><a href='{_e(correction_url)}'>Ilmoita korjaus</a></p>"
            if correction_url
            else ""
        )
    )
    page = _document(title, body, description="Kysymysmatka: vastaus, reitit ja lähdesidonta.")
    # Extra journey-only layout hooks; keep generic (no case keywords).
    extra_css = (
        ".journey-nodes{padding-left:1.1rem;margin:.25rem 0}"
        ".journey-edges{padding-left:1.1rem;margin:.25rem 0}"
        ".branch{margin:.65rem 0;padding:.35rem 0;border-top:1px solid #ccd7da}"
        ".branch:first-of-type{border-top:0}"
        ".takeaway{margin:.35rem 0 .45rem;font-weight:400}"
        ".takeaway p{margin:.4rem 0}"
        ".distinction{margin:.35rem 0 .5rem}"
        ".key-conditions-heading{font-size:1.05em;margin:.4rem 0 .2rem}"
        ".key-conditions{margin:.15rem 0 .55rem;padding-left:1.15rem}"
        ".key-conditions li{margin:.15rem 0}"
        ".full-question{margin:.5rem 0 1rem}"
        ".condition{margin:.15rem 0 .25rem;font-size:.92em;color:#334d57}"
        ".act-summary{margin:.4rem 0 .6rem}"
        ".act-summary h2{font-size:1rem;margin:.35rem 0 .15rem}"
        ".act-summary-list{margin:.15rem 0 .3rem;padding-left:1.1rem}"
        ".act-summary-list>li{margin:.12rem 0}"
        ".act-summary time{color:#52656d;font-size:.9em}"
        ".act-secondary{margin:.1rem 0 .15rem}"
        ".act-details{margin:.2rem 0}"
        ".act-more{margin:.2rem 0}"
        ".act-source-link{font-size:.9em}"
        ".review-plain{margin:.2rem 0 .3rem}"
        ".nearby-evidence{margin:.25rem 0 .45rem}"
        ".nearby-evidence-item{margin:.25rem 0}"
        ".nearby-evidence blockquote{margin:.2rem 0;padding-left:.6rem;"
        "border-left:3px solid #9aa7ad;white-space:pre-wrap}"
        "li:target, #node-:target{background:#fff4e5}"
        "@media(max-width:600px){"
        "h1{font-size:1.25rem;line-height:1.25;margin:.3rem 0}"
        ".takeaway{line-height:1.4;margin:.25rem 0 .35rem}"
        ".key-conditions-heading{margin:.3rem 0 .1rem}"
        ".key-conditions{line-height:1.4;margin:0 0 .4rem}"
        ".distinction{line-height:1.4;margin:.35rem 0 .45rem}"
        "}"
    )
    return page.replace("</style>", extra_css + "</style>", 1)


def _coverage_rank(kind: str | None) -> int:
    if kind == _COVERAGE_PREPARED:
        return 2
    if kind == _COVERAGE_RECORD:
        return 1
    return 0


def _merge_person_coverage(existing: str | None, incoming: str) -> str:
    return incoming if _coverage_rank(incoming) >= _coverage_rank(existing) else str(existing or incoming)


def build_lookup_index(
    journeys: Sequence[Mapping[str, Any]],
    *,
    matter_records: Sequence[Mapping[str, Any]] | None = None,
    dossier_entries: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Scoped discovery over prepared journeys, matter titles/actors and dossier aliases.

    Distinguishes prepared AI-source-reviewed answers from record-only hubs.
    Unknown queries remain explicit absences — never fabricated verified answers.
    Exact identity keys and ambiguous name-only records are retained separately.
    """

    people: dict[str, dict[str, Any]] = {}
    matters: dict[str, dict[str, Any]] = {}
    questions: list[dict[str, Any]] = []
    dossiers: list[dict[str, Any]] = []

    def _person(name: str, *, actor_key: str | None = None) -> dict[str, Any]:
        """Retain exact actor_key identities; merge only exact same display name.

        Name-only and keyed rows with the same exact name collapse to one entry so
        ordinary lookup works; conflicting distinct keys for different names stay apart.
        No fuzzy/similar-name merging.
        """

        name_key = name.casefold()
        if actor_key:
            key = str(actor_key).casefold()
            if key in people:
                return people[key]
            # Exact display-name match without a prior key: attach identity.
            if name_key in people and not people[name_key].get("actor_key"):
                person = people.pop(name_key)
                person["actor_key"] = actor_key
                people[key] = person
                return person
            return people.setdefault(
                key,
                {
                    "actor_name": name,
                    "actor_key": actor_key,
                    "matters": [],
                    "tasks": [],
                    "acts": [],
                    "record_hrefs": [],
                    "coverage_kind": _COVERAGE_ABSENT,
                    "identity_state": None,
                },
            )
        if name_key in people:
            return people[name_key]
        # Exact name already present under an actor_key.
        for person in people.values():
            if str(person.get("actor_name") or "").casefold() == name_key:
                return person
        return people.setdefault(
            name_key,
            {
                "actor_name": name,
                "actor_key": None,
                "matters": [],
                "tasks": [],
                "acts": [],
                "record_hrefs": [],
                "coverage_kind": _COVERAGE_ABSENT,
                "identity_state": None,
            },
        )

    for packet in journeys:
        if not isinstance(packet, Mapping):
            continue
        task_id = str(packet.get("task_id") or "")
        matter_id = str(packet.get("matter_id") or "")
        href = str(packet.get("site_href") or f"{_slug(task_id)}.html")
        review = (
            packet.get("source_answer_review")
            if isinstance(packet.get("source_answer_review"), Mapping)
            else {}
        )
        reader_title = packet.get("reader_title") or review.get("reader_title") or packet.get("editorial_title")
        questions.append(
            {
                "task_id": task_id,
                "matter_id": matter_id,
                "question": packet.get("question"),
                "reader_title": reader_title,
                "short_answer": packet.get("short_answer"),
                "href": href,
                "as_of": packet.get("as_of"),
                "projection_status": packet.get("projection_status"),
                "coverage_kind": _COVERAGE_PREPARED,
            }
        )
        if matter_id:
            entry = matters.setdefault(
                matter_id,
                {
                    "matter_id": matter_id,
                    "title": None,
                    "choice_label": None,
                    "aliases": [],
                    "task_ids": [],
                    "hrefs": [],
                    "actor_names": [],
                    "coverage_kind": _COVERAGE_PREPARED,
                },
            )
            entry["task_ids"].append(task_id)
            if href not in entry["hrefs"]:
                entry["hrefs"].append(href)
            entry["coverage_kind"] = _merge_person_coverage(entry.get("coverage_kind"), _COVERAGE_PREPARED)
        for actor in packet.get("actors") or []:
            if not isinstance(actor, Mapping) or not actor.get("actor_name"):
                continue
            name = str(actor["actor_name"])
            person = _person(name)
            person["coverage_kind"] = _merge_person_coverage(person.get("coverage_kind"), _COVERAGE_PREPARED)
            if matter_id and matter_id not in person["matters"]:
                person["matters"].append(matter_id)
            if task_id and task_id not in person["tasks"]:
                person["tasks"].append(task_id)
            person["acts"].append(
                {
                    "label": actor.get("label"),
                    "date": actor.get("date"),
                    "href": f"{href}#node-{actor.get('node_id')}",
                    "matter_id": matter_id,
                    "signature_role": actor.get("signature_role"),
                    "shared_act_id": actor.get("shared_act_id"),
                    "coverage_kind": _COVERAGE_PREPARED,
                }
            )
            if matter_id and name not in matters[matter_id]["actor_names"]:
                matters[matter_id]["actor_names"].append(name)

    for record in matter_records or []:
        if not isinstance(record, Mapping) or not record.get("matter_id"):
            continue
        matter_id = str(record["matter_id"])
        entry = matters.setdefault(
            matter_id,
            {
                "matter_id": matter_id,
                "title": None,
                "choice_label": None,
                "aliases": [],
                "task_ids": [],
                "hrefs": [],
                "actor_names": [],
                "coverage_kind": _COVERAGE_RECORD,
            },
        )
        if record.get("title"):
            entry["title"] = record.get("title")
        if record.get("choice_label"):
            entry["choice_label"] = record.get("choice_label")
        for alias in record.get("aliases") or []:
            text = str(alias).strip()
            if text and text not in entry["aliases"]:
                entry["aliases"].append(text)
        href = str(record.get("href") or "")
        if href and href not in entry["hrefs"]:
            entry["hrefs"].append(href)
        entry["coverage_kind"] = _merge_person_coverage(
            entry.get("coverage_kind"), str(record.get("coverage_kind") or _COVERAGE_RECORD)
        )
        for actor in record.get("actors") or []:
            if not isinstance(actor, Mapping) or not actor.get("actor_name"):
                continue
            name = str(actor["actor_name"])
            actor_key = str(actor.get("actor_key") or "") or None
            person = _person(name, actor_key=actor_key)
            if actor_key:
                person["actor_key"] = actor_key
            if actor.get("identity_state"):
                person["identity_state"] = actor.get("identity_state")
            person["coverage_kind"] = _merge_person_coverage(
                person.get("coverage_kind"), str(actor.get("coverage_kind") or _COVERAGE_RECORD)
            )
            if matter_id and matter_id not in person["matters"]:
                person["matters"].append(matter_id)
            record_href = str(actor.get("href") or "")
            if record_href and record_href not in person["record_hrefs"]:
                person["record_hrefs"].append(record_href)
            if name not in entry["actor_names"]:
                entry["actor_names"].append(name)
            if record_href:
                person["acts"].append(
                    {
                        "label": actor.get("label") or f"Kirjatut teot · {matter_id}",
                        "date": actor.get("date"),
                        "href": record_href,
                        "matter_id": matter_id,
                        "coverage_kind": _COVERAGE_RECORD,
                    }
                )

    for dossier in dossier_entries or []:
        if not isinstance(dossier, Mapping) or not dossier.get("href"):
            continue
        aliases = [str(item).strip() for item in (dossier.get("aliases") or []) if str(item).strip()]
        title = str(dossier.get("title") or (aliases[0] if aliases else "Dossier"))
        row = {
            "title": title,
            "aliases": aliases,
            "href": str(dossier["href"]),
            "matter_ids": list(dossier.get("matter_ids") or []),
            "coverage_kind": str(dossier.get("coverage_kind") or _COVERAGE_RECORD),
            "note": dossier.get("note"),
        }
        dossiers.append(row)
        for matter_id in row["matter_ids"]:
            entry = matters.setdefault(
                str(matter_id),
                {
                    "matter_id": str(matter_id),
                    "title": None,
                    "choice_label": None,
                    "aliases": [],
                    "task_ids": [],
                    "hrefs": [],
                    "actor_names": [],
                    "coverage_kind": _COVERAGE_RECORD,
                },
            )
            for alias in aliases:
                if alias not in entry["aliases"]:
                    entry["aliases"].append(alias)
            if row["href"] not in entry["hrefs"]:
                entry["hrefs"].append(row["href"])

    return {
        "schema_version": "paa.question_journey_lookup.v2",
        "people": [people[key] for key in sorted(people)],
        "matters": [matters[key] for key in sorted(matters)],
        "questions": questions,
        "dossiers": dossiers,
        "coverage_kinds": {
            "prepared": _COVERAGE_PREPARED,
            "record_only": _COVERAGE_RECORD,
            "absent": _COVERAGE_ABSENT,
        },
        "note": (
            "Lookup returns indexed prepared answers, matter hubs and dossier aliases. "
            "RECORD_ONLY means retained records without a prepared AI-source-reviewed answer. "
            "Unsupported queries are lookup failures, not fabricated verified answers."
        ),
    }


def _row_search_blob(row: Mapping[str, Any], keys: Sequence[str]) -> str:
    bits: list[str] = []
    for key in keys:
        value = row.get(key)
        if isinstance(value, list):
            bits.extend(str(item) for item in value)
        elif value is not None:
            bits.append(str(value))
    return " ".join(bits).casefold()


def lookup_query(index: Mapping[str, Any], query: str) -> dict[str, Any]:
    """Match ordinary words/names against the prepared/record discovery index."""

    text = " ".join(str(query or "").split()).strip()
    empty_matches = {"people": [], "matters": [], "questions": [], "dossiers": []}
    if not text:
        return {
            "status": "LOOKUP_FAILURE",
            "reason": "empty_query",
            "matches": empty_matches,
            "known_counts": {
                "people": len(index.get("people") or []),
                "matters": len(index.get("matters") or []),
                "questions": len(index.get("questions") or []),
                "dossiers": len(index.get("dossiers") or []),
            },
        }
    needle = text.casefold()
    people = [
        row
        for row in index.get("people") or []
        if needle in str(row.get("actor_name") or "").casefold()
        or needle in str(row.get("actor_key") or "").casefold()
    ]
    matters = [
        row
        for row in index.get("matters") or []
        if needle in _row_search_blob(
            row, ("matter_id", "title", "choice_label", "aliases", "actor_names")
        )
    ]
    questions = [
        row
        for row in index.get("questions") or []
        if needle in _row_search_blob(
            row, ("question", "reader_title", "task_id", "matter_id", "short_answer")
        )
    ]
    dossiers = [
        row
        for row in index.get("dossiers") or []
        if needle in _row_search_blob(row, ("title", "aliases", "matter_ids"))
    ]
    if not people and not matters and not questions and not dossiers:
        return {
            "status": "LOOKUP_FAILURE",
            "reason": "no_prepared_record",
            "query": text,
            "coverage_kind": _COVERAGE_ABSENT,
            "matches": empty_matches,
            "known_records": {
                "people": [row.get("actor_name") for row in (index.get("people") or [])[:20]],
                "matters": [row.get("matter_id") for row in (index.get("matters") or [])[:20]],
                "prepared_task_ids": [row.get("task_id") for row in (index.get("questions") or [])[:20]],
                "dossiers": [row.get("title") for row in (index.get("dossiers") or [])[:20]],
            },
        }
    coverage = _COVERAGE_RECORD
    for collection in (people, matters, questions, dossiers):
        for row in collection:
            if row.get("coverage_kind") == _COVERAGE_PREPARED:
                coverage = _COVERAGE_PREPARED
                break
        if coverage == _COVERAGE_PREPARED:
            break
    return {
        "status": "OK",
        "query": text,
        "coverage_kind": coverage,
        "matches": {
            "people": people,
            "matters": matters,
            "questions": questions,
            "dossiers": dossiers,
        },
    }


def _coverage_label_fi(kind: object) -> str:
    text = str(kind or "")
    if text == _COVERAGE_PREPARED:
        return "Valmisteltu AI-lähdearvioitu vastaus"
    if text == _COVERAGE_RECORD:
        return "Kirjatut rekisteritiedot (ei valmisteltua vastausta)"
    return "Ei indeksissä"


def render_lookup_html(
    index: Mapping[str, Any],
    *,
    back_url: str = "index.html",
    initial_query: str = "",
    initial_result_limit: int = 40,
) -> str:
    """Static lookup: empty = search/help/shortcuts; full index via browse or query."""

    people = index.get("people") or []
    matters = index.get("matters") or []
    questions = index.get("questions") or []
    dossiers = index.get("dossiers") or []
    person_cards = "".join(
        "<article data-lookup='"
        + _e(
            " ".join(
                [
                    str(row.get("actor_name") or ""),
                    str(row.get("actor_key") or ""),
                    " ".join(row.get("matters") or []),
                ]
            ).casefold()
        )
        + "'><h3>"
        + _e(row.get("actor_name"))
        + "</h3><p class='muted'>"
        + _e(_coverage_label_fi(row.get("coverage_kind")))
        + (" · " + _e(", ".join(row.get("matters") or [])) if row.get("matters") else "")
        + (
            f" · {_e(_identity_state_label_fi(row.get('identity_state')))}"
            if row.get("identity_state")
            else ""
        )
        + (
            f" <span class='muted identity-code' hidden>({_e(row.get('identity_state'))})</span>"
            if row.get("identity_state")
            else ""
        )
        + "</p>"
        + (
            f"<p><a href='{_e((row.get('acts') or [{}])[0].get('href'))}'>Avaa kirjaukset</a>"
            + (
                f" · {len(row.get('acts') or [])} tekoa"
                if row.get("acts")
                else ""
            )
            + "</p>"
            if (row.get("acts") or [{}])[0].get("href")
            else ""
        )
        + "</article>"
        for row in people
    )
    matter_cards = "".join(
        "<article data-lookup='"
        + _e(
            _row_search_blob(
                row, ("matter_id", "title", "choice_label", "aliases", "actor_names")
            )
        )
        + "'><h3>"
        + (
            f"<a href='{_e((row.get('hrefs') or [''])[0])}'>"
            if (row.get("hrefs") or [""])[0]
            else ""
        )
        + _e(row.get("choice_label") or row.get("title") or row.get("matter_id"))
        + ("</a>" if (row.get("hrefs") or [""])[0] else "")
        + "</h3><p class='muted'>"
        + _e(row.get("matter_id"))
        + (
            f" · alias: {_e(', '.join(row.get('aliases') or []))}"
            if row.get("aliases")
            else ""
        )
        + "</p></article>"
        for row in matters
    )
    question_cards = "".join(
        "<article data-lookup='"
        + _e(
            _row_search_blob(
                row, ("task_id", "matter_id", "question", "reader_title", "short_answer")
            )
        )
        + "'><h3><a href='"
        + _e(row.get("href"))
        + "'>"
        + _e(row.get("reader_title") or _compact_question_line(str(row.get("question") or row.get("task_id") or "")))
        + "</a></h3><p class='muted'>"
        + _e(row.get("matter_id"))
        + " · "
        + _e(_coverage_label_fi(row.get("coverage_kind")))
        + "</p>"
        + (f"<p>{_e(row.get('short_answer'))}</p>" if row.get("short_answer") else "")
        + "</article>"
        for row in questions
    )
    dossier_cards = "".join(
        "<article data-lookup='"
        + _e(_row_search_blob(row, ("title", "aliases", "matter_ids")))
        + "'><h3><a href='"
        + _e(row.get("href"))
        + "'>"
        + _e(row.get("title"))
        + "</a></h3><p class='muted'>"
        + _e(_coverage_label_fi(row.get("coverage_kind")))
        + (
            f" · {_e(', '.join(row.get('aliases') or []))}"
            if row.get("aliases")
            else ""
        )
        + "</p></article>"
        for row in dossiers
    )
    # Tiny prepared shortcuts outside the full index (empty-state affordance).
    shortcut_rows = [
        row
        for row in questions
        if isinstance(row, Mapping)
        and (row.get("coverage_kind") == _COVERAGE_PREPARED or row.get("short_answer") or row.get("reader_title"))
    ][:6]
    if not shortcut_rows:
        shortcut_rows = [row for row in questions if isinstance(row, Mapping)][:4]
    shortcuts = "".join(
        "<li><a href='"
        + _e(row.get("href"))
        + "'>"
        + _e(row.get("reader_title") or _compact_question_line(str(row.get("question") or row.get("task_id") or "")))
        + "</a></li>"
        for row in shortcut_rows
    )
    index_body = (
        "<h2>Valmistellut kysymykset</h2>"
        + (question_cards or "<p class='muted'>Ei valmisteltuja kysymyksiä.</p>")
        + "<h2>Henkilöt</h2>"
        + (person_cards or "<p class='muted'>Ei henkilöitä.</p>")
        + "<h2>Asiat</h2>"
        + (matter_cards or "<p class='muted'>Ei asioita.</p>")
        + (
            "<h2>Dossierit ja aliakset</h2>" + dossier_cards
            if dossier_cards
            else ""
        )
    )
    seeded = str(initial_query or "").strip()
    browse_open = " open" if seeded else ""
    body = (
        f"<p><a href='{_e(back_url)}'>Takaisin</a></p>"
        "<h1>Haku</h1>"
        "<p class='muted'>Hae henkilöä, asiaa tai kysymystä. Tuntematon haku ei synnytä vastausta.</p>"
        "<form method='get' action='' role='search'>"
        "<label for='q'>Hakusana</label> "
        f"<input id='q' name='q' type='search' value='{_e(initial_query)}' "
        "placeholder='nimi, asiatunnus tai sana' autocomplete='off'>"
        "<button type='submit'>Hae</button>"
        "</form>"
        "<p id='lookup-help' class='muted'>Kirjoita hakusana. "
        "<span hidden>Tyhjä haku ei avaa koko indeksiä.</span></p>"
        "<p id='lookup-status' class='muted' hidden></p>"
        + (
            "<section id='lookup-shortcuts'><h2>Valmiita oikopolkuja</h2>"
            f"<ul>{shortcuts}</ul></section>"
            if shortcuts
            else ""
        )
        + f"<details id='lookup-browse' class='lookup-browse'{browse_open}>"
        "<summary>Kaikki osumat</summary>"
        f"<div id='lookup-index'>{index_body}</div>"
        "<p id='lookup-more-wrap' hidden>"
        "<button type='button' id='lookup-more'>Näytä lisää</button>"
        " <span id='lookup-more-status' class='muted'></span></p>"
        "</details>"
        "<noscript><p class='muted'>Ilman JavaScriptiä avaa «Kaikki osumat».</p></noscript>"
        + f"""
<style>
[data-lookup][hidden], article[hidden], h2[hidden], #lookup-status[hidden],
#lookup-more-wrap[hidden], #lookup-help[hidden]{{display:none !important}}
.lookup-browse{{margin:.75rem 0}}
#lookup-shortcuts ul{{padding-left:1.1rem;margin:.25rem 0 .75rem}}
</style>
<script>
(() => {{
  const input = document.getElementById('q');
  const status = document.getElementById('lookup-status');
  const help = document.getElementById('lookup-help');
  const browse = document.getElementById('lookup-browse');
  const moreWrap = document.getElementById('lookup-more-wrap');
  const moreBtn = document.getElementById('lookup-more');
  const moreStatus = document.getElementById('lookup-more-status');
  if (!input || !status || !browse) return;
  const INITIAL_LIMIT = {int(initial_result_limit)};
  const cards = Array.from(document.querySelectorAll('#lookup-index [data-lookup]'));
  const sectionHeads = Array.from(document.querySelectorAll('#lookup-index h2'));
  let limit = INITIAL_LIMIT;
  let browseIntent = false;
  // A query opens the result disclosure automatically. Clearing that query
  // must restore the empty-search surface; only a reader's explicit toggle
  // keeps the full index open.
  let autoOpened = false;
  const syncUrl = (push) => {{
    const url = new URL(window.location.href);
    const value = (input.value || '').trim();
    if (value) url.searchParams.set('q', value);
    else url.searchParams.delete('q');
    const next = url.pathname + url.search + url.hash;
    const cur = window.location.pathname + window.location.search + window.location.hash;
    if (next === cur) return;
    if (push) history.pushState(null, '', next);
    else history.replaceState(null, '', next);
  }};
  const syncFromUrl = () => {{
    const q = new URL(window.location.href).searchParams.get('q');
    input.value = q == null ? '' : q;
    if (!q && autoOpened) {{
      browse.open = false;
      autoOpened = false;
      browseIntent = false;
    }}
  }};
  const syncSections = () => {{
    for (const h2 of sectionHeads) {{
      let el = h2.nextElementSibling;
      let any = false;
      while (el && el.tagName !== 'H2') {{
        if (el.hasAttribute && el.hasAttribute('data-lookup') && !el.hidden) any = true;
        el = el.nextElementSibling;
      }}
      h2.hidden = !any;
    }}
  }};
  const apply = ({{ updateHistory = false, push = false, resetLimit = false }} = {{}}) => {{
    if (resetLimit) limit = INITIAL_LIMIT;
    const needle = (input.value || '').trim().toLocaleLowerCase('fi');
    const active = Boolean(needle) || browseIntent || (browse.open && !autoOpened);
    if (needle && !browse.open) {{
      browse.open = true;
      autoOpened = true;
    }}
    if (!active) {{
      for (const card of cards) card.hidden = true;
      syncSections();
      status.hidden = true;
      if (help) help.hidden = false;
      if (moreWrap) moreWrap.hidden = true;
      if (updateHistory) syncUrl(push);
      return;
    }}
    if (help) help.hidden = true;
    let matched = 0;
    let shown = 0;
    for (const card of cards) {{
      const hay = card.getAttribute('data-lookup') || '';
      const ok = !needle || hay.includes(needle);
      if (ok) matched += 1;
      const reveal = ok && shown < limit;
      if (reveal) shown += 1;
      card.hidden = !reveal;
    }}
    syncSections();
    status.hidden = false;
    if (!matched) {{
      status.textContent = needle
        ? 'Ei indeksoitua tietuetta tälle haulle — ei varmennettua vastausta.'
        : 'Indeksi on tyhjä.';
    }} else if (shown < matched) {{
      status.textContent = needle
        ? (`Näytetään ${{shown}} / ${{matched}} osumaa.`)
        : (`Näytetään ${{shown}} / ${{matched}} indeksoitua tietuetta.`);
    }} else {{
      status.textContent = needle
        ? (`Näytetään ${{shown}} osumaa.`)
        : (`Näytetään ${{shown}} indeksoitua tietuetta.`);
    }}
    if (moreWrap && moreBtn && moreStatus) {{
      const remaining = matched - shown;
      moreWrap.hidden = remaining <= 0;
      moreStatus.textContent = remaining > 0 ? `${{remaining}} lisää` : '';
    }}
    if (updateHistory) syncUrl(push);
  }};
  input.addEventListener('input', () => {{
    browseIntent = false;
    if (!(input.value || '').trim() && autoOpened) {{
      browse.open = false;
      autoOpened = false;
    }}
    apply({{ updateHistory: true, push: false, resetLimit: true }});
  }});
  document.querySelector('form')?.addEventListener('submit', (event) => {{
    event.preventDefault();
    browseIntent = false;
    apply({{ updateHistory: true, push: false, resetLimit: true }});
  }});
  browse.addEventListener('toggle', () => {{
    if (browse.open) {{
      browseIntent = !(input.value || '').trim() && !autoOpened;
    }} else {{
      browseIntent = false;
      autoOpened = false;
    }}
    apply({{ updateHistory: false, resetLimit: true }});
  }});
  moreBtn?.addEventListener('click', () => {{
    limit += INITIAL_LIMIT;
    apply({{ updateHistory: false, resetLimit: false }});
  }});
  window.addEventListener('popstate', () => {{
    syncFromUrl();
    browseIntent = false;
    apply({{ updateHistory: false, resetLimit: true }});
  }});
  syncFromUrl();
  apply({{ updateHistory: false, resetLimit: true }});
}})();
</script>
"""
    )
    return _document("Haku", body, description="Henkilöiden, asioiden ja valmisteltujen kysymysten haku.")



def write_journey_site(
    journeys: Sequence[Mapping[str, Any]],
    output_dir: Path,
    *,
    title: str = "Kysymysmatkat",
    correction_url: str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write journey HTML/JSON pages, lookup index and a compact index page."""

    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise QuestionJourneyError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[dict[str, Any]] = []
    enriched: list[dict[str, Any]] = []
    for packet in journeys:
        task_id = str(packet.get("task_id") or "")
        slug = _slug(task_id)
        href = f"{slug}.html"
        json_name = f"{slug}.json"
        row = dict(packet)
        row["site_href"] = href
        enriched.append(row)
        (output_dir / json_name).write_text(
            json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        page = render_journey_html(
            row,
            packet_url=json_name,
            back_url="index.html",
            lookup_url="lookup.html",
            correction_url=correction_url,
        )
        (output_dir / href).write_text(page, encoding="utf-8")
        written.append(
            {
                "task_id": task_id,
                "matter_id": row.get("matter_id"),
                "path": href,
                "json_path": json_name,
                "projection_status": row.get("projection_status"),
                "admission": row.get("admission"),
            }
        )
    index = build_lookup_index(enriched)
    (output_dir / "lookup.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "lookup.html").write_text(
        render_lookup_html(index, back_url="index.html"), encoding="utf-8"
    )
    cards = "".join(
        "<article><h2><a href='"
        + _e(item["path"])
        + "'>"
        + _e(item["task_id"])
        + "</a></h2><p>"
        + _e(item.get("matter_id"))
        + " · "
        + _e(item.get("projection_status"))
        + "</p></article>"
        for item in written
    )
    body = (
        f"<h1>{_e(title)}</h1>"
        "<p>Kysymys, lyhyt vastaus ja haarautuva matka ennen pitkää auditointia. "
        "Haku kattaa vain valmistellut tietueet.</p>"
        "<p><a href='lookup.html'>Haku</a> · <a href='lookup.json'>lookup.json</a></p>"
        + cards
    )
    (output_dir / "index.html").write_text(_document(title, body), encoding="utf-8")
    manifest = {
        "schema_version": "paa.question_journey_site.v1",
        "title": title,
        "journeys": written,
        "lookup": "lookup.json",
        "file_count": len(list(output_dir.iterdir())),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def build_journey_from_paths(
    *,
    route_packet_path: Path,
    mapping_path: Path,
    answer_path: Path | None = None,
    source_answer_review_path: Path | None = None,
    projection_review_path: Path | None = None,
) -> dict[str, Any]:
    route_packet = _load_json(route_packet_path)
    mapping = _load_json(mapping_path)
    answer = _load_json(answer_path) if answer_path else None
    source_review = _load_json(source_answer_review_path) if source_answer_review_path else None
    projection_review = _load_json(projection_review_path) if projection_review_path else None
    return compose_journey(
        route_packet=route_packet,
        mapping=mapping,
        answer_packet=answer,
        source_answer_review=source_review,
        projection_review=projection_review,
    )


def journey_cli(
    *,
    route_packet: Path,
    mapping: Path,
    output: Path,
    answer: Path | None = None,
    source_answer_review: Path | None = None,
    projection_review: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    packet = build_journey_from_paths(
        route_packet_path=route_packet,
        mapping_path=mapping,
        answer_path=answer,
        source_answer_review_path=source_answer_review,
        projection_review_path=projection_review,
    )
    site = write_journey_site([packet], output, overwrite=overwrite)
    return {"journey": {"task_id": packet["task_id"], "admission": packet["admission"]}, "site": site}


__all__ = [
    "ADMISSION_REVIEWED",
    "ADMISSION_UNREVIEWED",
    "MAPPING_SCHEMA_VERSION",
    "PROJECTION_REVIEW_SCHEMA",
    "SCHEMA_VERSION",
    "QuestionJourneyError",
    "bind_projection_review",
    "build_journey_from_paths",
    "build_lookup_index",
    "compose_journey",
    "decode_mapping",
    "journey_cli",
    "lookup_query",
    "projection_status_label",
    "render_act_summary_html",
    "render_journey_html",
    "render_lookup_html",
    "validate_mapping_wire",
    "write_journey_site",
]

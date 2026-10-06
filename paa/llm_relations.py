"""Offline contracts for local-LLM relation proposals and verification.

This module deliberately does not own a model client.  The root orchestrator
can send ``request["model_input"]`` to an approved local endpoint and pass the
JSON response back to the validators here.  The validators keep the model in
the proposed/review layer: they never admit a relation or a finding.

The relation task is source-grounded.  A lexical candidate, a Finnish name,
or a ballot value is not an identity proof.  Exact quotes, source hashes,
matter IDs and evidence references are checked before a proposal can be
converted to the repository's ``verified_review`` shape.
"""


import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

PROMPT_DIR = Path(__file__).resolve().parent / "prompts"

RELATION_STATUSES = frozenset({"SAME_POLICY_OBJECT", "SAME_MATTER", "REJECTED", "UNRESOLVED"})
ACTION_ALIGNMENTS = frozenset({"ALIGNED", "CONTRARY", "RELATED", "UNRESOLVED"})
IDENTITY_BASES = frozenset({
    "EXPLICIT_MATTER_ID",
    "EXPLICIT_TARGET_AND_MATTER",
    "REVIEWED_SOURCE_LINK",
    "AUTHOR_OR_SIGNATORY_RECORD",
    "UNRESOLVED",
})
# Relation-v5 intentionally narrows the model-facing identity vocabulary.  A
# model may describe the substantive target with ``normalized_target``; that
# is a value, not an identity basis.  In particular, never emit the former
# pseudo-basis ``NORMALIZED_POLICY_TARGET``.
V5_IDENTITY_BASES = frozenset({
    "EXPLICIT_TARGET_AND_MATTER",
    "REVIEWED_SOURCE_LINK",
    "UNRESOLVED",
})
TARGET_SCOPES = frozenset({"EXACT", "PARTIAL_COMPONENT", "BROAD_RELATED", "UNRESOLVED"})
VERIFICATION_VERDICTS = frozenset({
    "SUPPORTED_WITHIN_SCOPE",
    "NARROWER_CLAIM_SUPPORTED",
    "CONTESTED",
    "INSUFFICIENT_EVIDENCE",
})
NARROW_ACTION_TYPES = frozenset({"PERSONAL_ACTION_COMMITMENT", "PERSONAL_RESTRAINT_COMMITMENT"})
VOTE_KINDS = frozenset({"VOTE", "PARLIAMENTARY_VOTE", "VOTE_EVENT"})
INITIATIVE_KINDS = frozenset({
    "LEGISLATIVE_INITIATIVE",
    "LAKIALOITE",
    "TOIMENPIDEALOITE",
    "CITIZEN_INITIATIVE",
})
_OMISSION = "\n[… context omitted; source coverage is recorded separately …]\n"
V3_MAX_QUOTE_CHARS = 160
V3_MAX_SUMMARY_CHARS = 120
VALIDATOR_VERSION = "relation-validator-v2-source-grounding-title-anchor"
# This version names the deterministic model-facing projection contract.  The
# runner also records a hash of the actual single-pair projection, so a future
# projection edit is caught even when this constant is accidentally left
# unchanged.  Bump it when the shape/meaning of model context changes.
MODEL_PROJECTION_VERSION = "relation-model-projection-v3-statement-payload-sharing"

# Finnish parliamentary identifiers normally have a short series code, a
# number and a four-digit year (for example ``LA 72/2017 vp``).  Keep this
# deliberately narrower than a generic identifier parser: a bare internal
# object id or an actor/author record is not matter identity evidence.
_FORMAL_MATTER_ID_RE = re.compile(
    r"(?<![A-ZÅÄÖ0-9])([A-ZÅÄÖ]{1,8}\s*\d{1,5}\s*/\s*\d{4}(?:\s+VP)?)(?![A-ZÅÄÖ0-9])",
    re.IGNORECASE,
)

_TARGET_WORD_RE = re.compile(r"[^\W\d_]{4,}", re.UNICODE)


def _source_grounded_target(target: str, source_text: str) -> bool:
    """Require a positive target to retain source-side lexical anchors.

    This is a safety check, not a relation matcher: it cannot admit a pair and
    it deliberately does not assert that overlapping words mean the same
    policy.  It only blocks a model from copying specificity that appears only
    in the official object.  A six-character prefix tolerates common Finnish
    and Swedish inflection while requiring at least two independent anchors
    for a multi-term target.
    """

    target_terms = [item.casefold() for item in _TARGET_WORD_RE.findall(target)]
    source_terms = [item.casefold() for item in _TARGET_WORD_RE.findall(source_text)]
    if not target_terms or not source_terms:
        return False
    matched = 0
    for target_term in target_terms:
        prefix_length = min(6, len(target_term))
        if any(
            source_term == target_term
            or source_term.startswith(target_term[:prefix_length])
            or target_term.startswith(source_term[:prefix_length])
            for source_term in source_terms
        ):
            matched += 1
    required = 1 if len(target_terms) == 1 else 2
    return matched >= required and matched * 2 >= len(target_terms)


PROPOSAL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "status",
        "matter_id",
        "identity_basis",
        "statement_quote",
        "object_quote",
        "normalized_target",
        "target_scope",
        "bounded_claim",
        "rationale",
        "action_alignment",
        "evidence_ids",
        "counterevidence_ids",
    ],
    "properties": {
        "status": {"enum": sorted(RELATION_STATUSES)},
        "matter_id": {"type": "string", "minLength": 1},
        "identity_basis": {"enum": sorted(IDENTITY_BASES)},
        "statement_quote": {"type": "string", "minLength": 1},
        "object_quote": {"type": "string", "minLength": 1},
        "normalized_target": {"type": ["string", "null"]},
        "target_scope": {"enum": sorted(TARGET_SCOPES)},
        "bounded_claim": {"type": "string", "minLength": 1},
        "rationale": {"type": "string", "minLength": 1},
        "action_alignment": {"enum": sorted(ACTION_ALIGNMENTS)},
        "evidence_ids": {"type": "array", "items": {"type": "string", "minLength": 1}, "minItems": 1},
        "counterevidence_ids": {"type": "array", "items": {"type": "string", "minLength": 1}},
        "actor_role": {"type": ["string", "null"]},
        "vote_interpretation": {"type": ["object", "null"]},
        "alternative_interpretations": {"type": "array", "items": {"type": "string"}},
    },
}

VERIFICATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "verdict",
        "status",
        "matter_id",
        "identity_basis",
        "statement_quote",
        "object_quote",
        "normalized_target",
        "target_scope",
        "bounded_claim",
        "rationale",
        "evidence_ids",
        "counterevidence_ids",
    ],
    "properties": {
        "verdict": {"enum": sorted(VERIFICATION_VERDICTS)},
        "status": {"enum": sorted(RELATION_STATUSES)},
        "matter_id": {"type": "string", "minLength": 1},
        "identity_basis": {"enum": sorted(IDENTITY_BASES)},
        "statement_quote": {"type": "string", "minLength": 1},
        "object_quote": {"type": "string", "minLength": 1},
        "normalized_target": {"type": ["string", "null"]},
        "target_scope": {"enum": sorted(TARGET_SCOPES)},
        "bounded_claim": {"type": "string", "minLength": 1},
        "rationale": {"type": "string", "minLength": 1},
        "evidence_ids": {"type": "array", "items": {"type": "string", "minLength": 1}, "minItems": 1},
        "counterevidence_ids": {"type": "array", "items": {"type": "string", "minLength": 1}},
        "missing_evidence": {"type": "array", "items": {"type": "string"}},
        "safe_claim": {"type": ["string", "null"]},
        "action_alignment": {"enum": sorted(ACTION_ALIGNMENTS)},
        "actor_role": {"type": ["string", "null"]},
        "vote_interpretation": {"type": ["object", "null"]},
        "alternative_interpretations": {"type": "array", "items": {"type": "string"}},
    },
}

BATCH_PROPOSAL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["pairs"],
    "properties": {
        "pairs": {
            "type": "array",
            "minItems": 1,
            "maxItems": 8,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["pair_id", *PROPOSAL_SCHEMA["required"]],
                "properties": {"pair_id": {"type": "string", "minLength": 1}, **PROPOSAL_SCHEMA["properties"]},
            },
        }
    },
}

BATCH_VERIFICATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["pairs"],
    "properties": {
        "pairs": {
            "type": "array",
            "minItems": 1,
            "maxItems": 8,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["pair_id", *VERIFICATION_SCHEMA["required"]],
                "properties": {"pair_id": {"type": "string", "minLength": 1}, **VERIFICATION_SCHEMA["properties"]},
            },
        }
    },
}


def _v3_schema(base: Mapping[str, Any]) -> dict[str, Any]:
    """Return the compact relation-v3 variant of a legacy object schema.

    The wire contract stays field-compatible with v1/v2 so old runners can
    still replay their artifacts.  V3 adds generation-time bounds for the
    fields that most often caused local-model truncation; the Python
    validator below repeats these limits because an endpoint may ignore a
    JSON-schema keyword.
    """

    result = deepcopy(base)
    properties = result.get("properties", {})
    for field in ("statement_quote", "object_quote"):
        if isinstance(properties.get(field), dict):
            properties[field]["maxLength"] = V3_MAX_QUOTE_CHARS
    for field in ("bounded_claim", "rationale"):
        if isinstance(properties.get(field), dict):
            properties[field]["maxLength"] = V3_MAX_SUMMARY_CHARS
    if isinstance(properties.get("normalized_target"), dict):
        properties["normalized_target"]["maxLength"] = V3_MAX_QUOTE_CHARS
    # Batch schemas contain the proposal/verification properties one level
    # deeper under pairs.items.properties.
    item = properties.get("pairs", {}).get("items") if isinstance(properties.get("pairs"), dict) else None
    if isinstance(item, Mapping):
        item_properties = item.get("properties", {})
        for field in ("statement_quote", "object_quote"):
            if isinstance(item_properties.get(field), dict):
                item_properties[field]["maxLength"] = V3_MAX_QUOTE_CHARS
        for field in ("bounded_claim", "rationale"):
            if isinstance(item_properties.get(field), dict):
                item_properties[field]["maxLength"] = V3_MAX_SUMMARY_CHARS
        if isinstance(item_properties.get("normalized_target"), dict):
            item_properties["normalized_target"]["maxLength"] = V3_MAX_QUOTE_CHARS
    return result


V3_PROPOSAL_SCHEMA: dict[str, Any] = _v3_schema(PROPOSAL_SCHEMA)
V3_VERIFICATION_SCHEMA: dict[str, Any] = _v3_schema(VERIFICATION_SCHEMA)
V3_BATCH_PROPOSAL_SCHEMA: dict[str, Any] = _v3_schema(BATCH_PROPOSAL_SCHEMA)
V3_BATCH_VERIFICATION_SCHEMA: dict[str, Any] = _v3_schema(BATCH_VERIFICATION_SCHEMA)


def _v5_schema(base: Mapping[str, Any]) -> dict[str, Any]:
    """Return the compact v5 contract with a minimal identity vocabulary.

    V5 keeps the proven short quote/summary limits from v3/v4.  Optional
    explanatory fields are removed from the wire schema so an eight-pair
    batch spends its output budget on the relation, evidence and action gate.
    The normalized internal proposal remains field-compatible with older
    receipts after validation.
    """

    result = _v3_schema(base)

    def patch_properties(properties: Any) -> None:
        if not isinstance(properties, dict):
            return
        identity = properties.get("identity_basis")
        if isinstance(identity, dict):
            identity["enum"] = sorted(V5_IDENTITY_BASES)
        # These fields are useful in legacy verbose review records but are not
        # required for the compact relation decision.  Vote interpretation is
        # retained because a ballot cannot establish policy support without it.
        properties.pop("alternative_interpretations", None)
        if "verdict" in properties:
            properties.pop("missing_evidence", None)
            properties.pop("safe_claim", None)

    properties = result.get("properties")
    patch_properties(properties)
    if isinstance(properties, dict):
        pairs = properties.get("pairs")
        item = pairs.get("items") if isinstance(pairs, dict) else None
        if isinstance(item, Mapping):
            patch_properties(item.get("properties"))
    return result


V5_PROPOSAL_SCHEMA: dict[str, Any] = _v5_schema(PROPOSAL_SCHEMA)
V5_VERIFICATION_SCHEMA: dict[str, Any] = _v5_schema(VERIFICATION_SCHEMA)
V5_BATCH_PROPOSAL_SCHEMA: dict[str, Any] = _v5_schema(BATCH_PROPOSAL_SCHEMA)
V5_BATCH_VERIFICATION_SCHEMA: dict[str, Any] = _v5_schema(BATCH_VERIFICATION_SCHEMA)


def fingerprint(text: str) -> str:
    """Return the canonical UTF-8 source fingerprint used by review records."""

    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def load_prompt(version: str = "relation_v1", *, stage: str = "propose") -> str:
    """Load a checked-in prompt; no endpoint or network access is involved."""

    if stage not in {"propose", "verify"}:
        raise ValueError("stage must be 'propose' or 'verify'")
    path = PROMPT_DIR / f"{version}_{stage}.txt"
    if not path.is_file():
        raise ValueError(f"unknown relation prompt: {version}/{stage}")
    return path.read_text(encoding="utf-8").strip()


def _is_v3_prompt(version: Any) -> bool:
    return _text(version).lower().startswith(("relation_v3", "relation_v4", "relation_v5", "relation_v6", "relation_v7"))


def _is_v5_prompt(version: Any) -> bool:
    return _text(version).lower().startswith(("relation_v5", "relation_v6", "relation_v7"))


def proposal_schema(prompt_version: str = "relation_v1") -> dict[str, Any]:
    """Return a detached copy of the versioned model proposal schema."""

    if _is_v5_prompt(prompt_version):
        return deepcopy(V5_PROPOSAL_SCHEMA)
    return deepcopy(V3_PROPOSAL_SCHEMA if _is_v3_prompt(prompt_version) else PROPOSAL_SCHEMA)


def verification_schema(prompt_version: str = "relation_v1") -> dict[str, Any]:
    """Return a detached copy of the versioned model verification schema."""

    if _is_v5_prompt(prompt_version):
        return deepcopy(V5_VERIFICATION_SCHEMA)
    return deepcopy(V3_VERIFICATION_SCHEMA if _is_v3_prompt(prompt_version) else VERIFICATION_SCHEMA)


def batch_proposal_schema(prompt_version: str = "relation_v1") -> dict[str, Any]:
    if _is_v5_prompt(prompt_version):
        return deepcopy(V5_BATCH_PROPOSAL_SCHEMA)
    return deepcopy(V3_BATCH_PROPOSAL_SCHEMA if _is_v3_prompt(prompt_version) else BATCH_PROPOSAL_SCHEMA)


def batch_verification_schema(prompt_version: str = "relation_v1") -> dict[str, Any]:
    if _is_v5_prompt(prompt_version):
        return deepcopy(V5_BATCH_VERIFICATION_SCHEMA)
    return deepcopy(V3_BATCH_VERIFICATION_SCHEMA if _is_v3_prompt(prompt_version) else BATCH_VERIFICATION_SCHEMA)


def schema(stage: str = "proposal", prompt_version: str = "relation_v1") -> dict[str, Any]:
    """Return the versioned JSON schema for the requested model-output stage."""

    if stage == "proposal":
        return proposal_schema(prompt_version)
    if stage == "verification":
        return verification_schema(prompt_version)
    if stage == "batch_proposal":
        return batch_proposal_schema(prompt_version)
    if stage == "batch_verification":
        return batch_verification_schema(prompt_version)
    raise ValueError("stage must be proposal, verification, batch_proposal, or batch_verification")


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _nonempty(value: Any, field: str) -> str:
    result = _text(value)
    if not result:
        raise ValueError(f"{field} must be a non-empty string")
    return result


def _ids(value: Any) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def _evidence_ids(value: Any) -> list[str]:
    """Collect explicit evidence IDs without treating arbitrary IDs as evidence."""

    found: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key == "evidence_id" and isinstance(item, str) and item.strip():
                found.append(item.strip())
            elif key.endswith("evidence_ids") and isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
                found.extend(_ids(item))
            elif key in {"evidence", "authors", "disposition", "action", "alternatives"}:
                found.extend(_evidence_ids(item))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            found.extend(_evidence_ids(item))
    return list(dict.fromkeys(found))


def _statement_text(statement: Mapping[str, Any]) -> str:
    return _text(statement.get("original_text") or statement.get("text"))


def _object_text(obj: Mapping[str, Any]) -> str:
    return _text(obj.get("text") or obj.get("normalized_text") or obj.get("title"))


def _canonical_formal_matter_id(value: Any) -> str | None:
    """Normalize one parliamentary matter identifier for comparison only."""

    text = _text(value)
    if not text:
        return None
    match = _FORMAL_MATTER_ID_RE.fullmatch(text)
    if not match:
        return None
    normalized = " ".join(match.group(1).upper().split())
    normalized = re.sub(r"\s*/\s*", "/", normalized)
    normalized = re.sub(r"^([A-ZÅÄÖ]{1,8})\s*(\d)", r"\1 \2", normalized)
    # ``vp`` is a citation suffix, not a distinct matter identifier.
    return normalized.removesuffix(" VP")


def _formal_matter_ids_from_text(value: Any) -> set[str]:
    text = _text(value)
    if not text:
        return set()
    return {
        canonical
        for match in _FORMAL_MATTER_ID_RE.finditer(text)
        if (canonical := _canonical_formal_matter_id(match.group(1)))
    }


def _formal_matter_ids_from_value(value: Any) -> set[str]:
    """Read identifiers from explicit ID fields and supplied source text."""

    found: set[str] = set()
    if isinstance(value, str):
        return _formal_matter_ids_from_text(value)
    if not isinstance(value, Mapping):
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for item in value:
                found.update(_formal_matter_ids_from_value(item))
        return found
    for key in (
        "matter_id",
        "formal_matter_id",
        "formal_matter_ids",
        "matter_ids",
        "related_matter_id",
        "related_matter_ids",
        "context_matter_ids",
    ):
        item = value.get(key)
        if isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
            for candidate in item:
                canonical = _canonical_formal_matter_id(candidate)
                if canonical:
                    found.add(canonical)
        else:
            canonical = _canonical_formal_matter_id(item)
            if canonical:
                found.add(canonical)
    for key in ("text", "original_text", "normalized_text", "title", "context_text", "proposition_text"):
        found.update(_formal_matter_ids_from_text(value.get(key)))
    return found


def _source_formal_matter_ids(statement: Mapping[str, Any], validation: Mapping[str, Any]) -> set[str]:
    found = _formal_matter_ids_from_value(statement)
    found.update(_formal_matter_ids_from_text(validation.get("statement_text")))
    found.update(_formal_matter_ids_from_text(validation.get("proposition_text")))
    return found


def _object_formal_matter_ids(obj: Mapping[str, Any], validation: Mapping[str, Any]) -> set[str]:
    found = _formal_matter_ids_from_value(obj)
    found.update(_formal_matter_ids_from_text(validation.get("object_text")))
    return found


def _is_bare_formal_matter_id(value: Any) -> bool:
    return _canonical_formal_matter_id(value) is not None


def _request_prompt_version(request: Mapping[str, Any]) -> str:
    version = _text(request.get("prompt_version"))
    if version:
        return version
    model_input = request.get("model_input")
    if isinstance(model_input, Mapping):
        return _text(model_input.get("prompt_version")) or "relation_v1"
    return "relation_v1"


def _quote_limit(prompt_version: Any, default: int = 320) -> int:
    return V3_MAX_QUOTE_CHARS if _is_v3_prompt(prompt_version) else default


def _clip_text(text: str, max_chars: int, required_spans: Iterable[str] = ()) -> dict[str, Any]:
    """Clip text while retaining required exact anchors where physically possible.

    The returned marker is not source text.  ``preserved_anchors`` and
    ``complete`` are part of the coverage certificate, so a caller cannot
    mistake a clipped context for a complete motion or statement.
    """

    if max_chars < 32:
        raise ValueError("max_chars must be at least 32")
    source = str(text)
    # Keep the omission marker inside a caller's hard context budget even for
    # the small diagnostic windows used by tests or retry prompts.
    marker = _OMISSION if len(_OMISSION) < max_chars else "…"
    required = list(dict.fromkeys(span for span in required_spans if _text(span)))
    locations: list[tuple[int, int, str]] = []
    missing: list[str] = []
    for span in required:
        start = source.find(span)
        if start < 0:
            missing.append(span)
        else:
            locations.append((start, start + len(span), span))
    locations.sort()
    if len(source) <= max_chars:
        return {
            "text": source,
            "complete": not missing,
            "original_chars": len(source),
            "omitted_chars": 0,
            "preserved_anchors": [span for _start, _end, span in locations],
            "missing_anchors": missing,
        }

    if not locations:
        tail = max(1, (max_chars - len(marker)) // 2)
        head = max_chars - len(marker) - tail
        clipped = source[:head] + marker + source[-tail:]
        return {
            "text": clipped,
            "complete": False,
            "original_chars": len(source),
            "omitted_chars": len(source) - len(clipped),
            "preserved_anchors": [],
            "missing_anchors": missing or required,
        }

    separator_budget = len(marker) * (len(locations) - 1)
    anchor_budget = sum(end - start for start, end, _span in locations)
    if anchor_budget + separator_budget > max_chars:
        clipped = source[:max_chars]
        return {
            "text": clipped,
            "complete": False,
            "original_chars": len(source),
            "omitted_chars": len(source) - len(clipped),
            "preserved_anchors": [span for start, end, span in locations if end <= max_chars],
            "missing_anchors": missing + [span for start, end, span in locations if end > max_chars],
        }

    windows = [(start, end, span) for start, end, span in locations]
    remaining = max_chars - anchor_budget - separator_budget
    # Expand each required span into local context without crossing the next
    # required span.  Round-robin expansion makes coverage deterministic.
    while remaining:
        changed = False
        for index, (start, end, span) in enumerate(windows):
            left_bound = windows[index - 1][1] if index else 0
            right_bound = windows[index + 1][0] if index + 1 < len(windows) else len(source)
            if start > left_bound:
                windows[index] = (start - 1, end, span)
                remaining -= 1
                changed = True
            if remaining and end < right_bound:
                windows[index] = (windows[index][0], end + 1, span)
                remaining -= 1
                changed = True
            if not remaining:
                break
        if not changed:
            break
    pieces = [source[start:end] for start, end, _span in windows]
    clipped = marker.join(pieces)
    return {
        "text": clipped,
        "complete": False,
        "original_chars": len(source),
        "omitted_chars": len(source) - len(clipped),
        "preserved_anchors": [span for _start, _end, span in windows],
        "missing_anchors": missing,
    }


def clip_context(text: str, max_chars: int = 6000, required_spans: Iterable[str] = ()) -> dict[str, Any]:
    """Public alias used by the orchestrator and tests."""

    return _clip_text(text, max_chars, required_spans)


def _clean_author(author: Any) -> dict[str, Any]:
    if not isinstance(author, Mapping):
        return {"name": _text(author), "role": "UNRESOLVED", "evidence_ids": []}
    return {
        key: author.get(key)
        for key in ("actor_id", "person_id", "name", "role", "identity_basis")
        if author.get(key) is not None
    } | {"evidence_ids": _ids(author.get("evidence_ids"))}


def _clean_alternatives(values: Any) -> list[dict[str, Any]]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return []
    result = []
    for index, item in enumerate(values):
        if isinstance(item, Mapping):
            result.append({
                "alternative_id": str(item.get("alternative_id") or item.get("id") or item.get("option_id") or item.get("raw_code") or index + 1),
                "label": _text(item.get("label") or item.get("title") or item.get("name") or item.get("raw_code")),
                "text": _text(
                    item.get("text")
                    or item.get("motion_text")
                    or item.get("description")
                    or item.get("alternative_text")
                    or item.get("substance")
                ),
                "substantive_effect": _text(item.get("substantive_effect") or item.get("effect") or item.get("substance")),
                "evidence_ids": _ids(item.get("evidence_ids")),
            })
        else:
            result.append({"alternative_id": str(index + 1), "label": _text(item), "text": "", "evidence_ids": []})
    return result


def _clean_ballot(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    return {
        key: value.get(key)
        for key in ("actor_id", "person_id", "vote_id", "aanestys_id", "raw_response", "alternative_id", "stage")
        if value.get(key) is not None
    } | {"evidence_ids": _ids(value.get("evidence_ids"))}


def _kind(obj: Mapping[str, Any]) -> str:
    return _text(obj.get("kind")).upper()


def _is_vote(obj: Mapping[str, Any]) -> bool:
    return _kind(obj) in VOTE_KINDS or "VOTE" in _kind(obj)


def _is_initiative(obj: Mapping[str, Any]) -> bool:
    return _kind(obj) in INITIATIVE_KINDS or "INITIATIVE" in _kind(obj) or "ALOITE" in _kind(obj)


def _actor_key(value: Any) -> str:
    text = _text(value).casefold()
    return text.removeprefix("mp-")


def _actor_matches(left: Any, right: Any) -> bool:
    return bool(_actor_key(left) and _actor_key(left) == _actor_key(right))


def _select_model_authors(
    authors: Sequence[Mapping[str, Any]],
    proposition: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, int | bool]]:
    """Keep subject matches and one first-author row in the model context.

    The raw official object remains available for validation and replay.  The
    model only needs the candidate's matching author rows plus a first-author
    control row; a complete co-signer roster is both expensive and irrelevant
    to policy identity.  Explicit counts prevent an omitted roster from being
    mistaken for evidence that nobody else signed the object.
    """

    subject_ids = _ids(proposition.get("subject_actor_ids"))
    selected: list[dict[str, Any]] = []
    selected_keys: set[tuple[str, str, str]] = set()

    def add(author: Mapping[str, Any]) -> None:
        key = (
            _text(author.get("actor_id")),
            _text(author.get("person_id")),
            _text(author.get("name")),
        )
        if key in selected_keys:
            return
        selected_keys.add(key)
        selected.append(dict(author))

    for author in authors:
        if any(
            _actor_matches(subject_id, author.get("actor_id"))
            or _actor_matches(subject_id, author.get("person_id"))
            for subject_id in subject_ids
        ):
            add(author)
    for author in authors:
        if _text(author.get("role")).upper() in {"FIRST_AUTHOR", "AUTHOR"}:
            add(author)
            break
    return selected, {
        "total": len(authors),
        "omitted": max(0, len(authors) - len(selected)),
        "complete": len(selected) == len(authors),
    }


def _matching_authors(request: Mapping[str, Any]) -> list[dict[str, Any]]:
    model_input = _relation_model_input(request)
    actors = model_input.get("proposition", {}).get("subject_actor_ids", [])
    authors = model_input.get("official_object", {}).get("authors", [])
    return [
        author for author in authors
        if any(_actor_matches(actor, author.get("actor_id")) or _actor_matches(actor, author.get("person_id")) for actor in actors)
    ]


def _source_validation(request: Mapping[str, Any]) -> dict[str, Any]:
    validation = request.get("validation")
    if not isinstance(validation, Mapping):
        raise TypeError("relation request lacks validation source context")
    statement_text = _nonempty(validation.get("statement_text"), "validation.statement_text")
    object_text = _nonempty(validation.get("object_text"), "validation.object_text")
    if fingerprint(statement_text) != validation.get("statement_sha256"):
        raise ValueError("statement validation hash does not match source text")
    if fingerprint(object_text) != validation.get("object_sha256"):
        raise ValueError("object validation hash does not match source text")
    return dict(validation)


def _parse_response(response: Mapping[str, Any] | str) -> tuple[dict[str, Any] | None, list[str]]:
    if isinstance(response, str):
        try:
            response = json.loads(response)
        except json.JSONDecodeError as exc:
            return None, [f"model response is not JSON: {exc.msg}"]
    if not isinstance(response, Mapping):
        return None, ["model response must be a JSON object"]
    return dict(response), []


def _relation_model_input(request: Mapping[str, Any]) -> Mapping[str, Any]:
    model_input = request.get("model_input")
    if not isinstance(model_input, Mapping):
        return {}
    nested = model_input.get("source_relation_request")
    return nested if isinstance(nested, Mapping) else model_input


def _base_request_fields(request: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    model_input = _relation_model_input(request)
    if not isinstance(model_input, Mapping):
        raise TypeError("relation request lacks model_input")
    statement = model_input.get("statement")
    obj = model_input.get("official_object")
    if not isinstance(statement, Mapping) or not isinstance(obj, Mapping):
        raise TypeError("relation request lacks statement or official_object")
    validation = _source_validation(request)
    if _text(statement.get("statement_id")) != _text(validation.get("statement_id")):
        raise ValueError("model statement ID does not match validation source")
    if _text(obj.get("object_id")) != _text(validation.get("object_id")):
        raise ValueError("model official-object ID does not match validation source")
    if _text(obj.get("matter_id")) != _text(validation.get("matter_id")):
        raise ValueError("model matter ID does not match validation source")
    model_proposition = model_input.get("proposition")
    if isinstance(model_proposition, Mapping):
        model_proposition_text = _text(model_proposition.get("source_text"))
        if model_proposition_text and model_proposition_text != _text(validation.get("proposition_text")):
            raise ValueError("model proposition source text does not match validation source")
    allowed = list(dict.fromkeys(
        _ids(request.get("statement_evidence_ids"))
        + _ids(request.get("object_evidence_ids"))
        + _ids(request.get("ballot_evidence_ids"))
        + _ids(request.get("counterevidence_ids"))
    ))
    return dict(statement), dict(obj), allowed


def _common_validation(response: Mapping[str, Any] | str, request: Mapping[str, Any], *, stage: str) -> dict[str, Any]:
    parsed, errors = _parse_response(response)
    if parsed is None:
        return {"valid": False, "errors": errors, "warnings": [], "proposal": None, "abstained": False}
    try:
        _statement, obj, allowed_evidence = _base_request_fields(request)
        source = _source_validation(request)
    except (TypeError, ValueError) as exc:
        return {"valid": False, "errors": [str(exc)], "warnings": [], "proposal": None, "abstained": False}

    errors = []
    warnings: list[str] = []
    prompt_version = _request_prompt_version(request)
    versioned_schema = verification_schema(prompt_version) if stage == "verification" else proposal_schema(prompt_version)
    allowed_response_keys = set(versioned_schema["properties"])
    if stage == "verification":
        allowed_response_keys.add("verdict")
    unknown_keys = sorted(set(parsed) - allowed_response_keys)
    if unknown_keys:
        errors.append("model response contains unknown keys: " + ", ".join(unknown_keys))
    status = _text(parsed.get("status")).upper()
    matter_id = _text(parsed.get("matter_id"))
    identity_basis = _text(parsed.get("identity_basis") or "UNRESOLVED").upper()
    statement_quote = _text(parsed.get("statement_quote"))
    object_quote = _text(parsed.get("object_quote"))
    normalized_target = parsed.get("normalized_target")
    target_scope = _text(parsed.get("target_scope") or "UNRESOLVED").upper()
    bounded_claim = _text(parsed.get("bounded_claim"))
    rationale = _text(parsed.get("rationale"))
    alignment = _text(parsed.get("action_alignment") or "UNRESOLVED").upper()
    evidence_ids = _ids(parsed.get("evidence_ids"))
    counterevidence_ids = _ids(parsed.get("counterevidence_ids"))

    if _is_v3_prompt(prompt_version):
        if len(statement_quote) > V3_MAX_QUOTE_CHARS:
            errors.append(f"statement_quote exceeds relation_v3 limit of {V3_MAX_QUOTE_CHARS} characters")
        if len(object_quote) > V3_MAX_QUOTE_CHARS:
            errors.append(f"object_quote exceeds relation_v3 limit of {V3_MAX_QUOTE_CHARS} characters")
        for field, value in (("bounded_claim", bounded_claim), ("rationale", rationale)):
            if len(value) > V3_MAX_SUMMARY_CHARS:
                errors.append(f"{field} exceeds relation_v3 limit of {V3_MAX_SUMMARY_CHARS} characters")

    if status not in RELATION_STATUSES:
        errors.append(f"status must be one of {sorted(RELATION_STATUSES)}")
    if matter_id != _text(obj.get("matter_id")):
        errors.append("matter_id does not exactly match the official object")
    if identity_basis not in IDENTITY_BASES:
        errors.append("identity_basis is not a declared source-grounded basis")
    if _is_v5_prompt(prompt_version) and identity_basis not in V5_IDENTITY_BASES:
        errors.append(
            "relation_v5 identity_basis must be REVIEWED_SOURCE_LINK, "
            "EXPLICIT_TARGET_AND_MATTER, or UNRESOLVED"
        )
    if not statement_quote or statement_quote not in source["statement_text"]:
        errors.append("statement_quote is not an exact source anchor")
    proposition_text = _text(source.get("proposition_text"))
    if proposition_text and statement_quote and statement_quote not in proposition_text:
        errors.append("statement_quote extends beyond the proposition source text")
    official_anchor_texts = [
        source["object_text"],
        _text(obj.get("title")),
        _text(obj.get("normalized_text")),
    ]
    if not object_quote or not any(object_quote in candidate for candidate in official_anchor_texts if candidate):
        errors.append("object_quote is not an exact official-object anchor")
    if not bounded_claim:
        errors.append("bounded_claim is required; a rationale alone is not scope")
    if not rationale:
        errors.append("rationale is required")
    if alignment not in ACTION_ALIGNMENTS:
        errors.append(f"action_alignment must be one of {sorted(ACTION_ALIGNMENTS)}")
    if target_scope not in TARGET_SCOPES:
        errors.append(f"target_scope must be one of {sorted(TARGET_SCOPES)}")
    if status in {"SAME_POLICY_OBJECT", "SAME_MATTER"}:
        if not isinstance(normalized_target, str) or not normalized_target.strip():
            errors.append("admitted relation status requires normalized_target")
        if identity_basis == "UNRESOLVED":
            errors.append("admitted relation status requires a non-UNRESOLVED identity_basis")
        if target_scope == "UNRESOLVED":
            errors.append("admitted relation status requires target_scope")
        if (
            _is_v5_prompt(prompt_version)
            and isinstance(normalized_target, str)
            and normalized_target.strip()
            and not _source_grounded_target(normalized_target, proposition_text or source["statement_text"])
        ):
            errors.append("normalized_target is not grounded in the proposition source text")
    elif normalized_target not in (None, ""):
        warnings.append("normalized_target was removed because the relation was not admitted")
        normalized_target = None
    if not evidence_ids:
        errors.append("evidence_ids must not be empty")
    if any(item not in allowed_evidence for item in evidence_ids + counterevidence_ids):
        errors.append("model cited an evidence ID absent from the supplied source context")
    statement_evidence = set(_ids(request.get("statement_evidence_ids")))
    object_evidence = set(_ids(request.get("object_evidence_ids")))
    if evidence_ids and not statement_evidence.intersection(evidence_ids):
        errors.append("relation evidence must include a statement-side evidence ID")
    if evidence_ids and not object_evidence.intersection(evidence_ids):
        errors.append("relation evidence must include an official-object evidence ID")

    # V3 separates matter identity from actor/action identity.  In
    # particular, an author or signatory record can prove who acted on an
    # object, but cannot prove that a campaign statement and the object are
    # the same matter.  SAME_MATTER therefore needs one formal identifier in
    # both supplied source contexts; an exact object-side matter_id alone is
    # insufficient when the statement never names that matter.
    if _is_v3_prompt(prompt_version) and status in {"SAME_POLICY_OBJECT", "SAME_MATTER"}:
        source_matter_ids = _source_formal_matter_ids(_statement, source)
        object_matter_ids = _object_formal_matter_ids(obj, source)
        if identity_basis == "AUTHOR_OR_SIGNATORY_RECORD":
            errors.append("AUTHOR_OR_SIGNATORY_RECORD cannot establish matter identity")
        if status == "SAME_MATTER" and not source_matter_ids.intersection(object_matter_ids):
            errors.append("SAME_MATTER requires a shared formal matter identifier in statement and object sources")
        if isinstance(normalized_target, str) and _is_bare_formal_matter_id(normalized_target):
            errors.append("normalized_target cannot be only a bare formal matter identifier")
        if status == "SAME_POLICY_OBJECT" and target_scope == "BROAD_RELATED":
            errors.append("SAME_POLICY_OBJECT requires an exact or partial policy target, not BROAD_RELATED")

    proposal = {
        "status": status,
        "matter_id": matter_id,
        "identity_basis": identity_basis,
        "statement_quote": statement_quote,
        "object_quote": object_quote,
        "normalized_target": normalized_target,
        "target_scope": target_scope,
        "bounded_claim": bounded_claim,
        "rationale": rationale,
        "action_alignment": alignment,
        "evidence_ids": evidence_ids,
        "counterevidence_ids": counterevidence_ids,
        "actor_role": _text(parsed.get("actor_role")) or None,
        "vote_interpretation": parsed.get("vote_interpretation") if isinstance(parsed.get("vote_interpretation"), Mapping) else None,
        "alternative_interpretations": [item for item in parsed.get("alternative_interpretations", []) if isinstance(item, str)],
    }

    coverage = request.get("coverage") if isinstance(request.get("coverage"), Mapping) else {}
    proposition = _relation_model_input(request).get("proposition", {})
    semantic_type = _text(proposition.get("semantic_type")).upper()
    if alignment in {"ALIGNED", "CONTRARY"} and semantic_type not in NARROW_ACTION_TYPES:
        proposal["action_alignment"] = "RELATED"
        warnings.append("broad, positional, collective or process text cannot be marked action-aligned or contrary")
    elif _is_v3_prompt(prompt_version) and alignment in {"ALIGNED", "CONTRARY"}:
        action_kind = _text(proposition.get("action_kind")).upper()
        issuer_scope = _text(proposition.get("issuer_scope")).upper()
        if proposition.get("observable_action") is not True or not action_kind or issuer_scope not in {"SELF", "PERSONAL"}:
            proposal["action_alignment"] = "RELATED"
            warnings.append("relation_v3 requires an explicit self-scoped observable action kind for action alignment")
    if status not in {"SAME_POLICY_OBJECT", "SAME_MATTER"} and proposal["action_alignment"] in {"ALIGNED", "CONTRARY"}:
        proposal["action_alignment"] = "UNRESOLVED"
        warnings.append("non-related object status cannot carry an action alignment")

    if _is_vote(obj) and status in {"SAME_POLICY_OBJECT", "SAME_MATTER"}:
        if not coverage.get("decisive_motion_present"):
            proposal.update({"status": "UNRESOLVED", "normalized_target": None, "target_scope": "UNRESOLVED", "action_alignment": "UNRESOLVED"})
            warnings.append("vote relation abstained because the decisive motion/alternatives were not supplied")
        elif proposal["action_alignment"] in {"ALIGNED", "CONTRARY"}:
            interpretation = proposal.get("vote_interpretation")
            if not isinstance(interpretation, Mapping) or not _text(interpretation.get("substantive_effect")):
                proposal["action_alignment"] = "UNRESOLVED"
                warnings.append("raw ballot response is not a policy interpretation; vote alignment abstained")
    if _is_initiative(obj) and proposal["action_alignment"] == "ALIGNED":
        authors = _matching_authors(request)
        author_roles = {_text(item.get("role")).upper() for item in authors}
        if not author_roles.intersection({"AUTHOR", "FIRST_AUTHOR", "ACTOR"}):
            proposal["action_alignment"] = "RELATED"
            warnings.append("initiative relation is not action-aligned without an explicit AUTHOR role for the actor")
        elif proposal.get("actor_role") and proposal["actor_role"].upper() not in author_roles:
            proposal["action_alignment"] = "RELATED"
            warnings.append("model actor_role does not match the official author role")

    if stage == "verification" and not _text(parsed.get("verdict")):
        errors.append("verification response lacks verdict")
    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "proposal": proposal if not errors else None,
        "abstained": any("abstained" in warning for warning in warnings),
        "raw": parsed,
    }


def validate_relation_response(response: Mapping[str, Any] | str, request: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a proposal and apply deterministic safety downgrades."""

    return _common_validation(response, request, stage="proposal")


def validate_proposal(response: Mapping[str, Any] | str, request: Mapping[str, Any]) -> dict[str, Any]:
    """Readable alias for callers that use proposal terminology."""

    return validate_relation_response(response, request)


def validate(response: Mapping[str, Any] | str, request: Mapping[str, Any], *, stage: str = "proposal") -> dict[str, Any]:
    """Stage-dispatched validation convenience function."""

    if stage == "proposal":
        return validate_relation_response(response, request)
    if stage == "verification":
        return validate_verification_response(response, request)
    raise ValueError("stage must be 'proposal' or 'verification'")


def build_relation_request(
    proposition: Mapping[str, Any],
    statement: Mapping[str, Any],
    official_object: Mapping[str, Any],
    *,
    alternatives: Sequence[Mapping[str, Any]] | None = None,
    actor_ballot: Mapping[str, Any] | None = None,
    counterevidence: Sequence[Mapping[str, Any]] | None = None,
    max_statement_chars: int = 6000,
    max_object_chars: int = 9000,
    prompt_version: str = "relation_v1",
) -> dict[str, Any]:
    """Create a compact, source-grounded model request without calling a model."""

    proposition_id = _nonempty(proposition.get("proposition_id"), "proposition.proposition_id")
    statement_id = _nonempty(statement.get("statement_id") or statement.get("document_version_id"), "statement.statement_id")
    object_id = _nonempty(official_object.get("object_id"), "official_object.object_id")
    matter_id = _nonempty(official_object.get("matter_id"), "official_object.matter_id")
    statement_text = _nonempty(_statement_text(statement), "statement.original_text")
    object_text = _nonempty(_object_text(official_object), "official_object.text")
    proposition_text = _text(proposition.get("source_text") or proposition.get("original_text"))
    if not proposition_text:
        raise ValueError("proposition.source_text must be present")
    statement_formal_matter_ids = sorted(
        _formal_matter_ids_from_value(statement)
        | _formal_matter_ids_from_text(statement_text)
        | _formal_matter_ids_from_text(proposition_text)
    )
    object_formal_matter_ids = sorted(
        _formal_matter_ids_from_value(official_object)
        | _formal_matter_ids_from_text(object_text)
    )
    selected_alternatives = _clean_alternatives(
        alternatives if alternatives is not None else official_object.get("alternatives")
    )
    if not selected_alternatives and official_object.get("alternative_labels"):
        selected_alternatives = _clean_alternatives(official_object.get("alternative_labels"))
    raw_authors = [
        author for author in official_object.get("authors", [])
        if isinstance(author, Mapping)
    ]
    authors, author_coverage = _select_model_authors(
        [_clean_author(author) for author in raw_authors],
        proposition,
    )
    object_kind = _kind(official_object)
    motion_text = _text(official_object.get("motion_text") or official_object.get("title"))
    explicit_alternative_coverage = official_object.get("alternatives_complete")
    alternatives_complete = bool(
        len(selected_alternatives) >= 2
        and all(item.get("label") or item.get("text") for item in selected_alternatives)
        and (explicit_alternative_coverage is None or bool(explicit_alternative_coverage))
    )
    decisive_motion_present = not _is_vote(official_object) or bool(motion_text and alternatives_complete)
    clipped_statement = _clip_text(statement_text, max_statement_chars, [proposition_text])
    clipped_object = _clip_text(object_text, max_object_chars, [])
    counter_records = []
    for item in counterevidence or []:
        if not isinstance(item, Mapping):
            continue
        raw = _text(item.get("text") or item.get("quote") or item.get("title"))
        if not raw:
            continue
        counter_records.append({
            "evidence_ids": _ids(item.get("evidence_ids")) or ([item["evidence_id"]] if item.get("evidence_id") else []),
            "source_id": _text(item.get("source_id")),
            "kind": _text(item.get("kind")),
            "title": _text(item.get("title")),
            "text": _clip_text(raw, 2400, []),
        })
    statement_evidence_ids = _evidence_ids(statement) or _ids(proposition.get("evidence_ids"))
    object_evidence_ids = _evidence_ids(official_object)
    ballot_evidence_ids = _evidence_ids(actor_ballot)
    counterevidence_ids = list(dict.fromkeys(item for record in counter_records for item in record["evidence_ids"]))
    model_input = {
        "schema": "paa.relation.request.v1",
        "task": "propose a source-grounded relation; do not adjudicate fulfillment or responsibility",
        "proposition": {
            key: proposition.get(key)
            for key in (
                "proposition_id", "source_text", "semantic_type", "subject_actor_ids", "issuer_scope",
                "predicate", "target", "modality", "negation", "conditions", "deadline", "jurisdiction",
                "testability", "missing_specification", "action_kind", "required_capability", "observable_action",
                "alternative_interpretations", "evidence_ids",
            )
            if proposition.get(key) is not None
        },
        "statement": {
            "statement_id": statement_id,
            "source_id": _text(statement.get("source_id")),
            "record_locator": _text(statement.get("record_locator")),
            "source_field_label": _text(statement.get("source_field_label") or statement.get("field_label")),
            "language": _text(statement.get("language")),
            "question_text": _text(statement.get("question_text")),
            "answer_options": statement.get("answer_options") if isinstance(statement.get("answer_options"), list) else [],
            "context_statement_ids": _ids(statement.get("context_statement_ids")),
            "stated_at": statement.get("stated_at"),
            "context_text": clipped_statement["text"],
            "proposition_text": proposition_text,
            "formal_matter_ids": statement_formal_matter_ids,
            "evidence_ids": statement_evidence_ids,
        },
        "official_object": {
            "object_id": object_id,
            "matter_id": matter_id,
            "formal_matter_ids": object_formal_matter_ids,
            "kind": object_kind,
            "title": _text(official_object.get("title")),
            "normalized_text": _text(official_object.get("normalized_text")),
            "text": clipped_object["text"],
            "date": _text(official_object.get("date")),
            "action_date": _text(official_object.get("action_date")),
            "action_date_basis": _text(official_object.get("action_date_basis")),
            "url": _text(official_object.get("url")),
            "source_id": _text(official_object.get("source_id")),
            "record_locator": _text(official_object.get("record_locator") or official_object.get("source_locator")),
            "raw_sha256": _text(official_object.get("raw_sha256") or official_object.get("source_raw_sha256")),
            "authors": authors,
            "authors_total_count": author_coverage["total"],
            "authors_omitted_count": author_coverage["omitted"],
            "authors_roster_complete": author_coverage["complete"],
            "evidence_ids": object_evidence_ids,
            "disposition": official_object.get("disposition") or {"state": "UNRESOLVED", "evidence_ids": []},
            "motion_text": motion_text,
            "stage": _text(official_object.get("stage")),
            "substantive_interpretation": _text(official_object.get("substantive_interpretation")),
            "alternatives": selected_alternatives,
        },
        "actor_ballot": _clean_ballot(actor_ballot),
        "counterevidence": counter_records,
        "source_evidence_ids": {
            "statement": statement_evidence_ids,
            "official_object": object_evidence_ids,
            "actor_ballot": ballot_evidence_ids,
            "counterevidence": counterevidence_ids,
        },
        "instructions": {
            "candidate_only_until_review": True,
            "raw_ballot_is_not_policy_support": True,
            "lexical_overlap_is_not_identity": True,
            "broad_or_process_text_is_not_fulfillment": True,
            "author_and_cosigner_are_distinct": True,
            "private_hypotheses_are_not_evidence": True,
        },
    }
    validation = {
        "statement_id": statement_id,
        "proposition_id": proposition_id,
        "object_id": object_id,
        "matter_id": matter_id,
        "statement_text": statement_text,
        "proposition_text": proposition_text,
        "object_text": object_text,
        "statement_formal_matter_ids": statement_formal_matter_ids,
        "object_formal_matter_ids": object_formal_matter_ids,
        "statement_sha256": fingerprint(statement_text),
        "object_sha256": fingerprint(object_text),
    }
    coverage = {
        "statement_complete": clipped_statement["complete"],
        "statement_anchor_preserved": proposition_text in clipped_statement["text"],
        "object_complete": clipped_object["complete"],
        "object_anchor_preserved": clipped_object["complete"] or bool(clipped_object["preserved_anchors"]),
        "alternatives_complete": alternatives_complete,
        "decisive_motion_present": decisive_motion_present,
        "counterevidence_supplied": bool(counter_records),
        "counterevidence_closed": False,
        "omitted_source_chars": clipped_statement["omitted_chars"] + clipped_object["omitted_chars"],
    }
    # The client normally serializes only model_input.  Keep the coverage
    # certificate there as well as at the request envelope so the model sees
    # exactly what was clipped and which decisive fields are absent.
    model_input["coverage"] = coverage
    prompt = load_prompt(prompt_version, stage="propose")
    request_identity = {
        "prompt_version": prompt_version,
        "proposition_id": proposition_id,
        "statement_id": statement_id,
        "object_id": object_id,
        "statement_sha256": validation["statement_sha256"],
        "object_sha256": validation["object_sha256"],
        "ballot_evidence_ids": ballot_evidence_ids,
        "counterevidence_ids": counterevidence_ids,
    }
    return {
        "schema_version": "paa.relation.request.v1",
        "request_id": "relation-request-" + fingerprint(json.dumps(request_identity, sort_keys=True, ensure_ascii=False))[:24],
        "prompt_version": prompt_version,
        "prompt_sha256": fingerprint(prompt),
        "schema_sha256": fingerprint(json.dumps(proposal_schema(prompt_version), sort_keys=True, ensure_ascii=False)),
        "output_schema": proposal_schema(prompt_version),
        "model_input": model_input,
        "model_prompt": prompt,
        "validation": validation,
        "coverage": coverage,
        "statement_evidence_ids": statement_evidence_ids,
        "object_evidence_ids": object_evidence_ids,
        "ballot_evidence_ids": ballot_evidence_ids,
        "counterevidence_ids": counterevidence_ids,
    }


def request_relation(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Compatibility alias for the root client's request builder."""

    return build_relation_request(*args, **kwargs)


_STATEMENT_COVERAGE_FIELDS = (
    "statement_complete",
    "statement_anchor_preserved",
    "statement_context_complete",
    "statement_clip_complete",
    "source_context_kind",
)


def _statement_model_payload(
    model_input: Mapping[str, Any],
    coverage: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Return the statement payload used by a batch-sharing decision.

    A statement can contain several propositions, so source ID/hash equality
    alone does not establish that the model-facing statement block is safe to
    share.  Include the proposition anchor, clipped context, and
    statement-specific coverage while excluding object-only coverage.
    """

    statement = model_input.get("statement")
    if not isinstance(statement, Mapping):
        return None
    visible_coverage: dict[str, Any] = {}
    if isinstance(coverage, Mapping):
        visible_coverage.update(coverage)
    model_coverage = model_input.get("coverage")
    if isinstance(model_coverage, Mapping):
        # The nested model-facing coverage wins if a caller has deliberately
        # changed it after request construction.
        visible_coverage.update(model_coverage)
    return {
        "statement": dict(statement),
        "coverage": {
            field: visible_coverage.get(field)
            for field in _STATEMENT_COVERAGE_FIELDS
            if field in visible_coverage
        },
    }


def build_batch_relation_request(
    pairs: Sequence[Mapping[str, Any]],
    *,
    max_pairs: int = 8,
    max_statement_chars: int = 4200,
    max_object_chars: int = 3200,
    prompt_version: str = "relation_v2",
) -> dict[str, Any]:
    """Build one compact generation request for up to eight source pairs.

    Each input mapping contains ``pair_id``, ``proposition``, ``statement``
    and ``official_object``.  Only identical model-facing statement payloads
    (statement fields plus statement-specific coverage) are placed once in
    ``shared_statement``; source ID/hash equality alone is insufficient
    because one statement can contain several propositions.  Individual pair
    records retain a validation request with the complete source text.  The
    complete source never needs to be sent to the model when the clipped
    ``model_input`` is used by the client.
    """

    if not isinstance(pairs, Sequence) or isinstance(pairs, (str, bytes)):
        raise TypeError("pairs must be a sequence of mappings")
    if not pairs:
        raise ValueError("batch needs at least one relation pair")
    if max_pairs < 1 or max_pairs > 8:
        raise ValueError("max_pairs must be between 1 and 8")
    if len(pairs) > max_pairs:
        raise ValueError(f"batch has {len(pairs)} pairs; maximum is {max_pairs}")

    individual: list[tuple[str, dict[str, Any]]] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(pairs):
        if not isinstance(item, Mapping):
            raise TypeError(f"pair {index} must be a mapping")
        object_input = item.get("official_object")
        pair_id = _text(item.get("pair_id") or item.get("id"))
        if not pair_id and isinstance(object_input, Mapping):
            pair_id = _text(object_input.get("object_id"))
        pair_id = _nonempty(pair_id, f"pair[{index}].pair_id")
        if pair_id in seen_ids:
            raise ValueError(f"duplicate batch pair_id: {pair_id}")
        seen_ids.add(pair_id)
        if not isinstance(item.get("proposition"), Mapping):
            raise TypeError(f"pair {pair_id} lacks proposition")
        if not isinstance(item.get("statement"), Mapping):
            raise TypeError(f"pair {pair_id} lacks statement")
        if not isinstance(object_input, Mapping):
            raise TypeError(f"pair {pair_id} lacks official_object")
        request = build_relation_request(
            item["proposition"],
            item["statement"],
            object_input,
            alternatives=item.get("alternatives"),
            actor_ballot=item.get("actor_ballot"),
            counterevidence=item.get("counterevidence"),
            max_statement_chars=max_statement_chars,
            max_object_chars=max_object_chars,
            prompt_version=prompt_version,
        )
        individual.append((pair_id, request))

    first_request = individual[0][1]
    first_statement = first_request["model_input"]["statement"]
    first_statement_payload = _statement_model_payload(
        first_request["model_input"], first_request.get("coverage")
    )
    shared = first_statement_payload is not None and all(
        _statement_model_payload(request["model_input"], request.get("coverage")) == first_statement_payload
        and request["validation"]["statement_id"] == individual[0][1]["validation"]["statement_id"]
        and request["validation"]["statement_sha256"] == individual[0][1]["validation"]["statement_sha256"]
        for _pair_id, request in individual
    )
    model_pairs = []
    for pair_id, request in individual:
        pair_input = dict(request["model_input"])
        if shared:
            pair_input.pop("statement", None)
            pair_input["statement_ref"] = first_statement["statement_id"]
        model_pairs.append({
            "pair_id": pair_id,
            "input": pair_input,
            "coverage": {
                **request["coverage"],
                "shared_statement_context": shared,
                "fixed_short_quote_limit": _quote_limit(prompt_version),
            },
        })
    prompt = load_prompt(prompt_version, stage="propose")
    identity = {
        "prompt_version": prompt_version,
        "pair_ids": [pair_id for pair_id, _request in individual],
        "request_ids": [request["request_id"] for _pair_id, request in individual],
    }
    return {
        "schema_version": "paa.relation.batch_request.v1",
        "request_id": "relation-batch-" + fingerprint(json.dumps(identity, sort_keys=True, ensure_ascii=False))[:24],
        "prompt_version": prompt_version,
        "prompt_sha256": fingerprint(prompt),
        "schema_sha256": fingerprint(json.dumps(batch_proposal_schema(prompt_version), sort_keys=True, ensure_ascii=False)),
        "output_schema": batch_proposal_schema(prompt_version),
        "model_prompt": prompt,
        "model_input": {
            "schema": "paa.relation.batch_input.v1",
            "task": "propose one conservative source-grounded relation for every pair",
            "shared_statement": first_statement if shared else None,
            "pairs": model_pairs,
            "output_constraints": {
                "one_result_per_pair": True,
                "no_duplicate_or_missing_pair_ids": True,
                "max_quote_chars": _quote_limit(prompt_version),
                "rationale_max_sentences": 1,
                "candidate_only_until_review": True,
            },
        },
        "pair_ids": [pair_id for pair_id, _request in individual],
        "validation_requests": {pair_id: request for pair_id, request in individual},
    }


def _batch_response_items(response: Mapping[str, Any] | str) -> tuple[list[dict[str, Any]], list[str]]:
    parsed, errors = _parse_response(response)
    if parsed is None:
        return [], errors
    unknown = sorted(set(parsed) - {"pairs"})
    if unknown:
        errors.append("batch model response contains unknown keys: " + ", ".join(unknown))
    items = parsed.get("pairs")
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
        return [], ["batch model response must contain a pairs array"]
    if len(items) > 8:
        errors.append("batch model response contains more than eight pairs")
    if any(not isinstance(item, Mapping) for item in items):
        return [], ["every batch response pair must be an object"]
    return [dict(item) for item in items], []


def validate_batch_relation_response(response: Mapping[str, Any] | str, request: Mapping[str, Any]) -> dict[str, Any]:
    """Validate each batch result against its own complete source request."""

    items, errors = _batch_response_items(response)
    expected = list(request.get("pair_ids") or [])
    actual = [_text(item.get("pair_id")) for item in items]
    if len(actual) != len(set(actual)):
        errors.append("batch response contains duplicate pair IDs")
    if set(actual) != set(expected):
        errors.append("batch response pair IDs do not exactly match the request")
    requests = request.get("validation_requests")
    if not isinstance(requests, Mapping):
        errors.append("batch request lacks validation_requests")
        requests = {}
    results = []
    quote_limit = _quote_limit(request.get("prompt_version"))
    for item in items:
        pair_id = _text(item.get("pair_id"))
        if pair_id not in requests:
            continue
        payload = {key: value for key, value in item.items() if key != "pair_id"}
        result = validate_relation_response(payload, requests[pair_id])
        results.append({"pair_id": pair_id, **result})
        if not result["valid"]:
            errors.extend(f"{pair_id}: {error}" for error in result["errors"])
        proposal = result.get("proposal")
        if isinstance(proposal, Mapping):
            for field in ("statement_quote", "object_quote"):
                if len(_text(proposal.get(field))) > quote_limit:
                    errors.append(f"{pair_id}: {field} exceeds the compact batch quote limit of {quote_limit}")
    return {"valid": not errors and len(results) == len(expected), "errors": errors, "results": results}


def build_batch_verification_request(
    proposals: Sequence[Mapping[str, Any]],
    *,
    max_pairs: int = 8,
    prompt_version: str = "relation_v2",
) -> dict[str, Any]:
    """Build a verifier batch from ``{pair_id, proposal, request}`` mappings."""

    if not isinstance(proposals, Sequence) or isinstance(proposals, (str, bytes)):
        raise TypeError("proposals must be a sequence of mappings")
    if not proposals:
        raise ValueError("verification batch needs at least one proposal")
    if max_pairs < 1 or max_pairs > 8:
        raise ValueError("max_pairs must be between 1 and 8")
    if len(proposals) > max_pairs or len(proposals) > 8:
        raise ValueError("verification batch has more than eight proposals")
    individual: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(proposals):
        if not isinstance(item, Mapping):
            raise TypeError(f"proposal {index} must be a mapping")
        relation_request = item.get("request")
        proposal = item.get("proposal")
        if not isinstance(relation_request, Mapping) or not isinstance(proposal, Mapping):
            raise TypeError("each verifier pair needs request and proposal")
        pair_id = _text(item.get("pair_id") or relation_request.get("request_id"))
        pair_id = _nonempty(pair_id, f"proposal[{index}].pair_id")
        if pair_id in individual:
            raise ValueError(f"duplicate verification pair_id: {pair_id}")
        individual[pair_id] = build_verification_request(
            proposal,
            relation_request,
            counterevidence=item.get("counterevidence"),
            prompt_version=prompt_version,
        )
    prompt = load_prompt(prompt_version, stage="verify")
    # Verifier batches usually revisit several candidate objects against one
    # statement.  Keep it in one shared context block only when the complete
    # model-facing statement payload (including proposition and coverage) is
    # identical; a statement ID/hash alone may cover several propositions.
    requests = list(individual.items())
    first_source = requests[0][1]["model_input"].get("source_relation_request")
    first_statement_payload = (
        _statement_model_payload(first_source, requests[0][1].get("coverage"))
        if isinstance(first_source, Mapping)
        else None
    )
    shared = first_statement_payload is not None and all(
        isinstance(req["model_input"].get("source_relation_request"), Mapping)
        and _statement_model_payload(
            req["model_input"]["source_relation_request"], req.get("coverage")
        ) == first_statement_payload
        and req["validation"].get("statement_id") == requests[0][1]["validation"].get("statement_id")
        and req["validation"].get("statement_sha256") == requests[0][1]["validation"].get("statement_sha256")
        for _pair_id, req in requests
    )
    shared_statement = None
    if shared and isinstance(first_source, Mapping):
        shared_statement = first_source.get("statement")
    model_pairs = []
    for pair_id, req in requests:
        pair_input = dict(req["model_input"])
        source_input = pair_input.get("source_relation_request")
        if shared and isinstance(source_input, Mapping):
            source_input = dict(source_input)
            source_input.pop("statement", None)
            source_input["statement_ref"] = _text((shared_statement or {}).get("statement_id"))
            pair_input["source_relation_request"] = source_input
        model_pairs.append({
            "pair_id": pair_id,
            "input": pair_input,
            "coverage": {**req["coverage"], "shared_statement_context": shared},
        })
    identity = {"prompt_version": prompt_version, "pair_ids": list(individual), "request_ids": [req["request_id"] for req in individual.values()]}
    return {
        "schema_version": "paa.relation.batch_verification_request.v1",
        "request_id": "relation-verification-batch-" + fingerprint(json.dumps(identity, sort_keys=True, ensure_ascii=False))[:24],
        "prompt_version": prompt_version,
        "prompt_sha256": fingerprint(prompt),
        "schema_sha256": fingerprint(json.dumps(batch_verification_schema(prompt_version), sort_keys=True, ensure_ascii=False)),
        "output_schema": batch_verification_schema(prompt_version),
        "model_prompt": prompt,
        "model_input": {
            "schema": "paa.relation.batch_verification_input.v1",
            "task": "verify each proposal against primary source and supplied counterevidence",
            "shared_statement": shared_statement,
            "pairs": model_pairs,
            "output_constraints": {
                "one_result_per_pair": True,
                "no_duplicate_or_missing_pair_ids": True,
                "max_quote_chars": _quote_limit(prompt_version),
                "rationale_max_sentences": 1,
            },
        },
        "pair_ids": list(individual),
        "validation_requests": individual,
    }


def validate_batch_verification_response(response: Mapping[str, Any] | str, request: Mapping[str, Any]) -> dict[str, Any]:
    items, errors = _batch_response_items(response)
    expected = list(request.get("pair_ids") or [])
    actual = [_text(item.get("pair_id")) for item in items]
    if len(actual) != len(set(actual)):
        errors.append("batch verification response contains duplicate pair IDs")
    if set(actual) != set(expected):
        errors.append("batch verification response pair IDs do not exactly match the request")
    requests = request.get("validation_requests")
    if not isinstance(requests, Mapping):
        errors.append("batch verification request lacks validation_requests")
        requests = {}
    results = []
    quote_limit = _quote_limit(request.get("prompt_version"))
    for item in items:
        pair_id = _text(item.get("pair_id"))
        if pair_id not in requests:
            continue
        payload = {key: value for key, value in item.items() if key != "pair_id"}
        result = validate_verification_response(payload, requests[pair_id])
        results.append({"pair_id": pair_id, **result})
        if not result["valid"]:
            errors.extend(f"{pair_id}: {error}" for error in result["errors"])
        proposal = result.get("proposal")
        if isinstance(proposal, Mapping):
            for field in ("statement_quote", "object_quote"):
                if len(_text(proposal.get(field))) > quote_limit:
                    errors.append(f"{pair_id}: {field} exceeds the compact batch quote limit of {quote_limit}")
    return {"valid": not errors and len(results) == len(expected), "errors": errors, "results": results}


def build_verification_request(
    proposal: Mapping[str, Any],
    relation_request: Mapping[str, Any],
    *,
    counterevidence: Sequence[Mapping[str, Any]] | None = None,
    prompt_version: str = "relation_v1",
    max_counterevidence_chars: int = 2400,
) -> dict[str, Any]:
    """Build a separate-context verifier request with explicit counterevidence."""

    if not isinstance(proposal, Mapping):
        raise TypeError("proposal must be a mapping")
    prompt_version = _text(prompt_version) or "relation_v1"
    proposal_payload = {
        key: proposal.get(key)
        for key in proposal_schema(prompt_version)["properties"]
        if key in proposal
    }
    result = validate_relation_response(proposal_payload, relation_request)
    if not result["valid"] or result["proposal"] is None:
        raise ValueError("cannot verify an invalid proposal: " + "; ".join(result["errors"]))
    records = []
    source_counterevidence = counterevidence
    if source_counterevidence is None:
        relation_input = _relation_model_input(relation_request)
        source_counterevidence = relation_input.get("counterevidence", []) if isinstance(relation_input, Mapping) else []
    for item in source_counterevidence or []:
        if not isinstance(item, Mapping):
            continue
        raw_text = item.get("text")
        if isinstance(raw_text, Mapping):
            raw_text = raw_text.get("text")
        source_text = _text(raw_text or item.get("quote") or item.get("title"))
        if not source_text:
            continue
        records.append({
            "evidence_ids": _ids(item.get("evidence_ids")) or ([item["evidence_id"]] if item.get("evidence_id") else []),
            "source_id": _text(item.get("source_id")),
            "kind": _text(item.get("kind")),
            "title": _text(item.get("title")),
            "text": _clip_text(source_text, max_counterevidence_chars, []) ,
        })
    counter_ids = list(dict.fromkeys(item for record in records for item in record["evidence_ids"]))
    prompt = load_prompt(prompt_version, stage="verify")
    return {
        "schema_version": "paa.relation.verification_request.v1",
        "request_id": relation_request.get("request_id"),
        "prompt_version": prompt_version,
        "prompt_sha256": fingerprint(prompt),
        "schema_sha256": fingerprint(json.dumps(verification_schema(prompt_version), sort_keys=True, ensure_ascii=False)),
        "output_schema": verification_schema(prompt_version),
        "model_prompt": prompt,
        "model_input": {
            "task": "verify the proposal against the supplied primary source and counterevidence",
            "source_relation_request": relation_request["model_input"],
            "proposal": dict(result["proposal"]),
            "counterevidence": records,
            "coverage": {
                **dict(relation_request.get("coverage") or {}),
                "counterevidence_supplied": bool(records),
                "counterevidence_closed": False,
            },
            "reviewer": {
                "mode": "SAME_MODEL_SEPARATE_CONTEXT",
                "independence": "NOT_INDEPENDENT",
                "generator_hidden_reasoning_reused": False,
                "counterevidence_required": True,
            },
        },
        "validation": relation_request["validation"],
        "coverage": {
            **dict(relation_request.get("coverage") or {}),
            "counterevidence_supplied": bool(records),
            "counterevidence_closed": False,
        },
        "statement_evidence_ids": relation_request.get("statement_evidence_ids", []),
        "object_evidence_ids": relation_request.get("object_evidence_ids", []),
        "ballot_evidence_ids": relation_request.get("ballot_evidence_ids", []),
        "counterevidence_ids": counter_ids,
        "proposal_review": dict(result["proposal"]),
    }


def validate_verification_response(response: Mapping[str, Any] | str, request: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a verifier output without promoting it to an independent review."""

    result = _common_validation(response, request, stage="verification")
    parsed = result.get("raw") or {}
    verdict = _text(parsed.get("verdict")).upper()
    if verdict not in VERIFICATION_VERDICTS:
        result["valid"] = False
        result["errors"].append(f"verdict must be one of {sorted(VERIFICATION_VERDICTS)}")
    proposal = request.get("proposal_review") or request.get("model_input", {}).get("proposal") or {}
    if _text(parsed.get("status")).upper() not in {"", _text(proposal.get("status")).upper(), "UNRESOLVED"}:
        result["valid"] = False
        result["errors"].append("verification cannot change a proposal relation except by abstaining")
    supplied_counter = set(_ids(request.get("counterevidence_ids")))
    cited_counter = set(_ids(parsed.get("counterevidence_ids")))
    if not cited_counter <= supplied_counter | set(_ids(request.get("object_evidence_ids"))):
        result["valid"] = False
        result["errors"].append("verification cited counterevidence outside the supplied context")
    if verdict == "SUPPORTED_WITHIN_SCOPE" and supplied_counter and not cited_counter.intersection(supplied_counter):
        result["proposal"] = None
        result["valid"] = False
        result["errors"].append("supported verdict must address supplied counterevidence")
    if verdict in {"CONTESTED", "INSUFFICIENT_EVIDENCE"}:
        if result.get("proposal"):
            result["proposal"]["status"] = "UNRESOLVED"
            result["proposal"]["normalized_target"] = None
            result["proposal"]["target_scope"] = "UNRESOLVED"
            result["proposal"]["action_alignment"] = "UNRESOLVED"
        result["abstained"] = True
    result["verdict"] = verdict
    result["independence"] = {
        "status": "NOT_INDEPENDENT",
        "basis": "same-model-separate-context",
        "counterevidence_context": bool(supplied_counter),
    }
    return result


def proposal_to_review(
    response: Mapping[str, Any] | str,
    request: Mapping[str, Any],
    *,
    model_id: str = "unspecified-local-model",
    prompt_version: str | None = None,
    review_id: str | None = None,
) -> dict[str, Any]:
    """Convert a structurally valid proposal into a PROPOSED ``verified_review`` record."""

    result = validate_relation_response(response, request)
    if not result["valid"] or result["proposal"] is None:
        raise ValueError("invalid relation proposal: " + "; ".join(result["errors"]))
    proposal = result["proposal"]
    validation = _source_validation(request)
    version = prompt_version or _text(request.get("prompt_version")) or "relation_v1"
    request_id = _nonempty(request.get("request_id"), "request.request_id")
    generated_id = review_id or (
        "llm-proposed:" + request_id + ":" + fingerprint(model_id + ":" + version)[:12]
    )
    return {
        "review_id": generated_id,
        "proposition_id": validation["proposition_id"],
        "object_id": validation["object_id"],
        "matter_id": validation["matter_id"],
        "status": proposal["status"],
        "relation": proposal["status"],
        "normalized_target": proposal["normalized_target"],
        "target_scope": proposal["target_scope"],
        "bounded_claim": proposal["bounded_claim"],
        "statement_sha256": validation["statement_sha256"],
        "object_sha256": validation["object_sha256"],
        "statement_quote": proposal["statement_quote"],
        "object_quote": proposal["object_quote"],
        "reviewer": f"local-llm-proposer:{model_id}:{version}",
        "review_method": "LOCAL_LLM_PROPOSAL",
        "review_state": "PROPOSED",
        "validation_state": "PROPOSED",
        "admission_state": "PROPOSED",
        "admission_route": "MODEL_PROPOSAL_NOT_ADMITTED",
        "rationale": proposal["rationale"],
        "evidence_ids": proposal["evidence_ids"],
        "counterevidence_ids": proposal["counterevidence_ids"],
        "action_alignment": proposal["action_alignment"],
        "identity_basis": proposal["identity_basis"],
        "actor_role": proposal["actor_role"],
        "vote_interpretation": proposal["vote_interpretation"],
        "alternative_interpretations": proposal["alternative_interpretations"],
        "model_id": model_id,
        "prompt_version": version,
        "request_id": request_id,
        "reviewer_independence": {
            "status": "NOT_INDEPENDENT",
            "basis": "model-generated proposal; requires source review/admission gate",
        },
        "validation_warnings": result["warnings"],
    }


def verification_artifact(
    response: Mapping[str, Any] | str,
    request: Mapping[str, Any],
    *,
    model_id: str = "unspecified-local-model",
    prompt_version: str | None = None,
) -> dict[str, Any]:
    """Persist verifier output metadata while keeping it non-independent/non-admitted."""

    result = validate_verification_response(response, request)
    if not result["valid"] or result.get("proposal") is None:
        raise ValueError("invalid relation verification: " + "; ".join(result["errors"]))
    proposal = result["proposal"]
    validation = _source_validation(request)
    version = prompt_version or _text(request.get("prompt_version")) or "relation_v1"
    return {
        "verification_id": "llm-verification:" + fingerprint(
            str(request.get("request_id")) + ":" + model_id + ":" + version
        )[:24],
        "request_id": request.get("request_id"),
        "proposition_id": validation["proposition_id"],
        "object_id": validation["object_id"],
        "status": proposal["status"],
        "matter_id": proposal["matter_id"],
        "verdict": result["verdict"],
        "identity_basis": proposal["identity_basis"],
        "normalized_target": proposal["normalized_target"],
        "target_scope": proposal["target_scope"],
        "action_alignment": proposal["action_alignment"],
        "rationale": proposal["rationale"],
        "bounded_claim": proposal["bounded_claim"],
        "statement_quote": proposal["statement_quote"],
        "object_quote": proposal["object_quote"],
        "evidence_ids": proposal["evidence_ids"],
        "counterevidence_ids": proposal["counterevidence_ids"],
        "review_state": "PROPOSED",
        "validation_state": "PROPOSED",
        "admission_state": "PROPOSED",
        "admission_route": "MODEL_VERIFICATION_NOT_INDEPENDENT",
        "reviewer": f"local-llm-verifier:{model_id}:{version}",
        "reviewer_independence": result["independence"],
        "validation_warnings": result["warnings"],
    }


def measure_relation_pairs(
    pairs: Iterable[Mapping[str, Any]],
    reviews: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Measure relation statuses against an independent, source-anchored pair set.

    This is a development measurement only.  It does not alter review state or
    turn a predicted status into an admitted relation.
    """

    pair_rows = list(pairs)
    review_map = {
        _text(row.get("pair_id") or row.get("object_id")): row
        for row in reviews
        if _text(row.get("pair_id") or row.get("object_id"))
    }
    positive = 0
    predicted_positive = 0
    true_positive = 0
    false_positive = 0
    abstained = 0
    rejected = 0
    judgments = []
    for pair in pair_rows:
        key = _text(pair.get("pair_id") or pair.get("object_id"))
        gold = pair.get("gold_status") or pair.get("gold_label")
        gold_positive = gold in {"SAME_POLICY_OBJECT", "SAME_MATTER", "POSITIVE_SAME_POLICY_OBJECT", "POSITIVE"}
        if gold_positive:
            positive += 1
        review = review_map.get(key)
        predicted = _text(
            (review or {}).get("status")
            or (review or {}).get("review_status")
            or (review or {}).get("relation")
        ).upper() if review else "UNRESOLVED"
        is_positive = predicted in {"SAME_POLICY_OBJECT", "SAME_MATTER"}
        if is_positive:
            predicted_positive += 1
        if is_positive and gold_positive:
            true_positive += 1
        if is_positive and not gold_positive:
            false_positive += 1
        if predicted in {"UNRESOLVED", ""}:
            abstained += 1
        elif predicted == "REJECTED":
            rejected += 1
        judgments.append({"pair_id": key, "gold": gold, "predicted": predicted, "gold_positive": gold_positive})
    return {
        "pair_count": len(pair_rows),
        "positive_count": positive,
        "predicted_positive_count": predicted_positive,
        "true_positive_count": true_positive,
        "false_positive_count": false_positive,
        "abstention_count": abstained,
        "rejected_count": rejected,
        "precision": true_positive / predicted_positive if predicted_positive else 0.0,
        "recall": true_positive / positive if positive else 0.0,
        "evaluation_scope": "independent_source_anchored_pairs_only",
        "judgments": judgments,
    }


__all__ = [
    "ACTION_ALIGNMENTS",
    "BATCH_PROPOSAL_SCHEMA",
    "BATCH_VERIFICATION_SCHEMA",
    "IDENTITY_BASES",
    "PROPOSAL_SCHEMA",
    "RELATION_STATUSES",
    "TARGET_SCOPES",
    "V3_BATCH_PROPOSAL_SCHEMA",
    "V3_BATCH_VERIFICATION_SCHEMA",
    "V3_MAX_QUOTE_CHARS",
    "V3_MAX_SUMMARY_CHARS",
    "V3_PROPOSAL_SCHEMA",
    "V3_VERIFICATION_SCHEMA",
    "V5_BATCH_PROPOSAL_SCHEMA",
    "V5_BATCH_VERIFICATION_SCHEMA",
    "V5_IDENTITY_BASES",
    "V5_PROPOSAL_SCHEMA",
    "V5_VERIFICATION_SCHEMA",
    "VERIFICATION_SCHEMA",
    "VERIFICATION_VERDICTS",
    "batch_proposal_schema",
    "batch_verification_schema",
    "build_batch_relation_request",
    "build_batch_verification_request",
    "build_relation_request",
    "build_verification_request",
    "clip_context",
    "fingerprint",
    "load_prompt",
    "measure_relation_pairs",
    "proposal_schema",
    "proposal_to_review",
    "request_relation",
    "schema",
    "validate",
    "validate_batch_relation_response",
    "validate_batch_verification_response",
    "validate_proposal",
    "validate_relation_response",
    "validate_verification_response",
    "verification_artifact",
    "verification_schema",
]

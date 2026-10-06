"""Deterministic carrier, decision-episode, and actor-attribution helpers.

This module is downstream of the canonical trace packet. It does not discover
policy relations, infer who drafted a document, or turn a recorded act into a
causal claim. It composes source-backed dimensions already present in a packet
and keeps proposed/unresolved inputs out of supported claims.

The public entry point is attach_attribution. It returns a copy of a trace
packet with three optional top-level fields:

* commitment_carrier: statement scope and explicit actor attribution;
* decision_episodes: official objects grouped only by exact matter_id;
* attribution_envelopes: actor x episode dimensions.

Callers that validate against the current evidence-trace schema can persist
the separate artifact returned by attribution_artifact, or add the optional
fields to their schema.
"""


import hashlib
import re
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from typing import Any

SCHEMA_VERSION = "paa.attribution.v1"

SUPPORTED = "SUPPORTED"
UNKNOWN = "UNKNOWN"
UNRESOLVED = "UNRESOLVED"
PROPOSED = "PROPOSED"

_POSITIVE_RELATION_STATES = frozenset({"SAME_MATTER", "SAME_POLICY_OBJECT"})
_PROPOSED_MARKERS = frozenset({
    "PROPOSED",
    "CANDIDATE",
    "UNREVIEWED",
    "STALE_OR_UNGROUNDED",
    "MODEL_PROPOSAL_NOT_ADMITTED",
    "MODEL_VERIFICATION_NOT_INDEPENDENT",
})
_REVIEWED_MARKERS = frozenset({"VALID", "VALIDATED", "SOURCE_REVIEWED", "INDEPENDENT_REVIEW", "ADMITTED"})
_PERSONAL_TYPES = frozenset({"PERSONAL_ACTION_COMMITMENT", "PERSONAL_RESTRAINT_COMMITMENT"})
_COLLECTIVE_TYPES = frozenset({"COLLECTIVE_ACTION_COMMITMENT"})
_COMMITMENT_TYPES = frozenset({
    "PERSONAL_ACTION_COMMITMENT",
    "PERSONAL_RESTRAINT_COMMITMENT",
    "COLLECTIVE_ACTION_COMMITMENT",
    "OUTCOME_COMMITMENT",
    "PROCESS_COMMITMENT",
    "MAINTAIN_COMMITMENT",
    "PREVENT_COMMITMENT",
})
_ACTION_KINDS = frozenset({
    "INITIATIVE_AUTHORED",
    "VOTE_CAST",
    "QUESTION_FILED",
    "SPEECH_DELIVERED",
    "RESIGN_ROLE",
    "DONATION",
    "PUBLIC_ADVOCACY",
    "POLICY_RESTRAINT",
    "OTHER_OBSERVABLE_ACTION",
})
_FORMAL_MATTER = re.compile(
    r"(?<![A-ZÅÄÖ0-9])[A-ZÅÄÖ]{1,8}\s*\d{1,6}\s*/\s*\d{4}(?:\s+VP)?(?![A-ZÅÄÖ0-9])",
    re.IGNORECASE,
)
_PERSONAL_MARKER = re.compile(
    r"\b(?:lupaan|teen|jätän|eroan|äänestän|esitän|kirjoitan|laadin|kysyn|pyrin|"
    r"kannatan|vastustan|vaadin|haluan|toivon|en|minä)\b",
    re.IGNORECASE,
)
_COLLECTIVE_MARKER = re.compile(
    r"\b(?:lupaamme|teemme|jätämme|eroamme|äänestämme|esitämme|kirjoitamme|"
    r"laadimme|kysymme|pyrimme|kannatamme|vastustamme|vaadimme|haluamme|emme|me|puolueemme)\b",
    re.IGNORECASE,
)
_PASSIVE_MARKER = re.compile(
    r"\bon\s+(?:\w+\s+){0,3}\w*(?:tava|ttava|tävä|ttävä)\b|"
    r"\bon\s+(?:poistettava|saatava|tehtävä|turvattava|huolehdittava)\b",
    re.IGNORECASE,
)

# A named institution is a carrier only when it is the grammatical subject of
# a commitment-like verb at the beginning of the supplied source span.  This
# deliberately does not search for institution words anywhere in the text:
# ``Vaadin että hallitus tekee ...`` attributes the demand to the speaker, and
# ``Hallitus leikkasi ...`` records an event rather than a commitment.  The
# proposition classifier still decides whether the span is a commitment at
# all; these patterns only identify a source-explicit institutional subject.
_INSTITUTIONAL_COMMITMENT_VERB = (
    r"(?:lup(?:aa|aamme|aavat)|sitout(?:uu|umme|uvat)|aik(?:oo|omme|ovat)|"
    r"pyrk(?:ii|imme|ivät)|esitt(?:ää|ämme|ävät)|laati(?:i|mme|vat)|"
    r"jättä(?:ä|mme|vät)|te(?:kee|emme|kevät)|edistä(?:ä|mme|vät)|"
    r"turva(?:a|mme|vat)|vahvista(?:a|mme|vat)|kannatta(?:a|mme|vat)|"
    r"vastusta(?:a|mme|vat))"
)
_INSTITUTIONAL_SUBJECT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (
        carrier_type,
        re.compile(
            r"^\s*[\"'«]?(?:"
            + subject
            + r")\b[^.!?]{0,140}\b"
            + _INSTITUTIONAL_COMMITMENT_VERB
            + r"\b",
            re.IGNORECASE,
        ),
    )
    for carrier_type, subject in (
        ("PARLIAMENTARY_GROUP", r"(?:meidän\s+)?eduskuntaryhm(?:ä|ämme|än)"),
        ("COALITION", r"(?:meidän\s+)?koalitio(?:mme|n)?"),
        ("MINISTRY", r"(?:maa-\s*ja\s*metsätalous)?ministeriö(?:mme|n)?"),
        ("GOVERNMENT", r"(?:meidän\s+)?hallitus(?:mme|n)?"),
        ("PARTY", r"(?:meidän\s+)?puolue(?:emme|en|et)?"),
        ("PARLIAMENT", r"(?:meidän\s+)?eduskunta(?:mme|n)?"),
        ("OTHER_INSTITUTION", r"(?:meidän\s+)?(?:kunta|kaupunki|aluevaltuusto)(?:mme|n)?"),
    )
)


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _dedupe(values: Iterable[Any]) -> list[str]:
    return list(dict.fromkeys(str(value).strip() for value in values if _text(value)))


def _ids(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return _dedupe(value)
    return []


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _provenance_state(value: Any) -> str:
    marker = _text(value).upper()
    return marker or UNKNOWN


def _is_proposed(value: Any) -> bool:
    marker = _provenance_state(value)
    return marker in _PROPOSED_MARKERS or marker.startswith("MODEL_")


def _is_reviewed(value: Any) -> bool:
    return _provenance_state(value) in _REVIEWED_MARKERS


def _packet_evidence_ids(packet: Mapping[str, Any]) -> set[str]:
    values: set[str] = set()
    evidence = packet.get("evidence")
    if isinstance(evidence, Sequence) and not isinstance(evidence, (str, bytes)):
        for item in evidence:
            if isinstance(item, Mapping) and _text(item.get("evidence_id")):
                values.add(_text(item.get("evidence_id")))
    return values


def _known_ids(packet: Mapping[str, Any], values: Iterable[Any]) -> list[str]:
    supplied = _dedupe(values)
    available = _packet_evidence_ids(packet)
    return [value for value in supplied if value in available]


def _dimension(
    state: str,
    value: Any = None,
    *,
    claim: str | None = None,
    basis: str | None = None,
    evidence_ids: Iterable[Any] = (),
    limitation: str | None = None,
    packet: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a typed dimension, downgrading unsupported evidence safely."""

    refs = _dedupe(evidence_ids)
    if state == SUPPORTED:
        available = _packet_evidence_ids(packet or {})
        missing = [item for item in refs if item not in available]
        if not refs or missing:
            return {
                "state": UNKNOWN,
                "value": None,
                "claim": None,
                "basis": basis or "UNRESOLVED",
                "evidence_ids": [],
                "limitation": limitation or "Supported attribution lacks complete packet evidence.",
                # Do not call these evidence_ids: records.evidence_reference_ids
                # intentionally treats every *_evidence_ids field as a source
                # reference.  Missing diagnostics are not source evidence and
                # must never become a false proof reference downstream.
                "missing_reference_ids": missing or refs,
            }
    result = {
        "state": state,
        "value": value,
        "claim": claim,
        "basis": basis or ("SOURCE" if state == SUPPORTED else "UNRESOLVED"),
        "evidence_ids": refs if state == SUPPORTED else [],
    }
    if limitation:
        result["limitation"] = limitation
    return result


def _safe_supplied_dimension(
    packet: Mapping[str, Any],
    value: Any,
    *,
    limitation: str,
) -> dict[str, Any]:
    """Keep externally supplied supported dimensions evidence-complete.

    ``attach_attribution`` can receive episodes or a carrier assembled by an
    upstream component.  Do not let a raw ``state=SUPPORTED`` mapping bypass
    the packet-evidence check that our own dimensions use.
    """

    if not isinstance(value, Mapping):
        return _dimension(UNKNOWN, limitation=limitation, packet=packet)
    if _text(value.get("state")).upper() != SUPPORTED:
        return dict(value)
    refs = _ids(value.get("evidence_ids"))
    return _dimension(
        SUPPORTED,
        value.get("value"),
        claim=_text(value.get("claim")) or None,
        basis=_text(value.get("basis")) or "EXTERNAL_SOURCE_DIMENSION",
        evidence_ids=refs,
        limitation=limitation,
        packet=packet,
    )


def _source_text(packet: Mapping[str, Any], proposition: Mapping[str, Any]) -> str:
    return _text(
        proposition.get("text")
        or proposition.get("source_text")
        or _mapping(packet.get("statement")).get("text")
    )


def _semantic_type(proposition: Mapping[str, Any]) -> str:
    return _text(proposition.get("semantic_type")).upper()


def _proposition_evidence(packet: Mapping[str, Any], proposition: Mapping[str, Any]) -> list[str]:
    statement = _mapping(packet.get("statement"))
    return _dedupe(_ids(proposition.get("evidence_ids")) + _ids(statement.get("evidence_ids")))


def _explicit_actor_ids(
    packet: Mapping[str, Any],
    proposition: Mapping[str, Any],
) -> tuple[list[str], list[str], str | None]:
    """Return explicit source actors, their evidence, and attribution basis.

    packet.actor_id is an identity join, not by itself a statement-side
    attribution proof. It is accepted only when an upstream source-bound
    attribution field accompanies it.
    """

    statement = _mapping(packet.get("statement"))
    candidates: list[str] = []
    for source in (statement, proposition, packet):
        for key in ("issuer_actor_ids", "subject_actor_ids", "actor_ids", "issuer_actor_id"):
            candidates.extend(_ids(source.get(key)))
    candidates = _dedupe(candidates)
    evidence = _proposition_evidence(packet, proposition)
    basis = (
        _text(statement.get("attribution_basis"))
        or _text(proposition.get("attribution_basis"))
        or _text(packet.get("attribution_basis"))
        or None
    )
    return candidates, evidence, basis


def _source_scope(
    packet: Mapping[str, Any],
    proposition: Mapping[str, Any],
    source: str,
) -> tuple[str, str]:
    """Resolve grammatical scope from source wording and explicit fields."""

    if _PASSIVE_MARKER.search(source):
        return "PASSIVE", "PASSIVE_CONSTRUCTION"
    statement = _mapping(packet.get("statement"))
    explicit_scope = _text(proposition.get("issuer_scope") or statement.get("issuer_scope")).upper()
    if any(pattern.search(source) for _, pattern in _INSTITUTIONAL_SUBJECT_PATTERNS):
        return "COLLECTIVE", "EXPLICIT_SOURCE_INSTITUTIONAL_SUBJECT"
    collective_marker = bool(_COLLECTIVE_MARKER.search(source))
    personal_marker = bool(_PERSONAL_MARKER.search(source))
    if explicit_scope in {"PARTY", "OTHER_COLLECTIVE", "COLLECTIVE", "UNSPECIFIED_WE"} and (
        collective_marker or explicit_scope in {"PARTY", "COLLECTIVE"}
    ):
        return "COLLECTIVE", "EXPLICIT_ISSUER_SCOPE_AND_SOURCE" if collective_marker else "EXPLICIT_ISSUER_SCOPE"
    if collective_marker:
        return "COLLECTIVE", "COLLECTIVE_SOURCE_PRONOUN"
    if explicit_scope in {"SELF", "PERSONAL"} and personal_marker:
        return "PERSONAL", "EXPLICIT_ISSUER_SCOPE_AND_SOURCE"
    if personal_marker:
        return "PERSONAL", "PERSONAL_SOURCE_PRONOUN"
    return "UNKNOWN", "NO_SOURCE_CARRIER"


def _carrier_type(
    scope: str,
    source: str,
    explicit_attribution: Mapping[str, Any],
    proposition: Mapping[str, Any],
) -> tuple[str, str]:
    """Map source-supported scope to the bounded carrier type vocabulary."""

    if scope == "PASSIVE":
        return "UNRESOLVED", "PASSIVE_ACTOR_OMITTED"
    if scope == "PERSONAL" and explicit_attribution.get("state") == SUPPORTED:
        return "PERSON", "EXPLICIT_SELF_AND_ACTOR"
    # A broad/value/position sentence can mention an institution without
    # making it the source of a commitment.  Do not turn that mention into a
    # carrier even when issuer_scope was supplied by an upstream classifier.
    semantic_type = _semantic_type(proposition)
    if semantic_type not in _COMMITMENT_TYPES:
        return "UNRESOLVED", "NON_COMMITMENT_SOURCE"
    if scope == "COLLECTIVE":
        for carrier_type, pattern in _INSTITUTIONAL_SUBJECT_PATTERNS:
            if pattern.search(source):
                return carrier_type, "EXPLICIT_SOURCE_INSTITUTIONAL_SUBJECT"
    # A collective first-person pronoun without an identified institution is
    # not enough to assign the candidate's party or parliamentary group.
    return "UNRESOLVED", "CARRIER_INSTITUTION_NOT_NAMED"


def resolve_commitment_carrier(
    packet: Mapping[str, Any],
    *,
    proposition: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve statement scope and explicit actor attribution separately.

    The source can support that a sentence is personal/collective/passive even
    when semantic classification is still PROPOSED. In that case the
    commitment dimension remains PROPOSED and no supported fulfilment claim is
    possible. Passive constructions never receive a guessed actor.
    """

    if not isinstance(packet, Mapping):
        raise TypeError("packet must be a mapping")
    prop = dict(proposition or _mapping(packet.get("proposition")))
    source = _source_text(packet, prop)
    evidence = _proposition_evidence(packet, prop)
    semantic_type = _semantic_type(prop)
    validation_state = _provenance_state(prop.get("validation_state"))
    scope, scope_basis = _source_scope(packet, prop, source)
    actor_ids, attribution_evidence, attribution_basis = _explicit_actor_ids(packet, prop)
    packet_actor = _text(packet.get("actor_id"))
    actor_mismatch = bool(
        packet_actor
        and actor_ids
        and not any(_actor_matches(packet_actor, candidate) for candidate in actor_ids)
    )
    if actor_mismatch:
        explicit_attribution = _dimension(
            UNKNOWN,
            claim=None,
            basis="ACTOR_MISMATCH",
            limitation="Source attribution names a different actor than the canonical packet actor.",
            packet=packet,
        )
    elif actor_ids and attribution_basis and not _is_proposed(attribution_basis):
        explicit_attribution = _dimension(
            SUPPORTED,
            actor_ids,
            claim="The supplied statement metadata explicitly attributes the statement to these actor IDs.",
            basis=attribution_basis,
            evidence_ids=attribution_evidence,
            packet=packet,
        )
    else:
        explicit_attribution = _dimension(
            UNKNOWN,
            claim=None,
            basis="NO_EXPLICIT_ACTOR_ATTRIBUTION",
            limitation="A packet actor identity alone does not prove who the statement grammatically attributes.",
            packet=packet,
        )

    scope_state = SUPPORTED if scope != "UNKNOWN" and evidence else UNKNOWN
    scope_dimension = _dimension(
        scope_state,
        scope if scope_state == SUPPORTED else None,
        claim="The source wording supports this grammatical carrier scope." if scope_state == SUPPORTED else None,
        basis=scope_basis,
        evidence_ids=evidence,
        limitation="The source does not identify a personal, collective, or passive carrier." if scope_state != SUPPORTED else None,
        packet=packet,
    )
    if semantic_type not in _COMMITMENT_TYPES or not source:
        commitment_state = UNKNOWN
        commitment_value = None
        commitment_basis = "NON_COMMITMENT_OR_EMPTY"
        commitment_limit = "This proposition does not establish a commitment. A statement and its speaker can still be attributable."
    elif _is_proposed(validation_state):
        commitment_state = PROPOSED
        commitment_value = semantic_type or scope
        commitment_basis = "PROPOSED_CLASSIFICATION"
        commitment_limit = "Semantic classification is proposed and has not been promoted to a supported commitment."
    elif _is_reviewed(validation_state):
        commitment_state = SUPPORTED if scope in {"PERSONAL", "COLLECTIVE"} else UNKNOWN
        commitment_value = semantic_type or scope if commitment_state == SUPPORTED else None
        commitment_basis = "SOURCE_REVIEWED_CLASSIFICATION"
        commitment_limit = "Passive or actor-unresolved wording cannot establish a commitment carrier." if commitment_state != SUPPORTED else None
    else:
        commitment_state = UNKNOWN
        commitment_value = None
        commitment_basis = "UNRESOLVED_CLASSIFICATION"
        commitment_limit = "Missing or unresolved proposition validation cannot establish a commitment carrier."
    commitment = _dimension(
        commitment_state,
        commitment_value,
        claim="The source-reviewed proposition identifies this commitment class." if commitment_state == SUPPORTED else None,
        basis=commitment_basis,
        evidence_ids=evidence,
        limitation=commitment_limit,
        packet=packet,
    )

    raw_action_kind = _text(prop.get("action_kind")).upper() or None
    if raw_action_kind and raw_action_kind not in _ACTION_KINDS:
        raw_action_kind = None
    action_state = (
        PROPOSED if raw_action_kind and _is_proposed(validation_state)
        else SUPPORTED if raw_action_kind and commitment_state == SUPPORTED
        else UNKNOWN
    )
    action_kind = _dimension(
        action_state,
        raw_action_kind,
        claim="The source-reviewed proposition names this observable action kind." if action_state == SUPPORTED else None,
        basis="PROPOSED_CLASSIFICATION" if action_state == PROPOSED else "SOURCE_REVIEWED_CLASSIFICATION" if action_state == SUPPORTED else "UNRESOLVED",
        evidence_ids=evidence,
        limitation="Action kind is not promoted while proposition classification remains proposed." if action_state == PROPOSED else None,
        packet=packet,
    )
    carrier_type, carrier_type_basis = _carrier_type(scope, source, explicit_attribution, prop)
    type_state = SUPPORTED if carrier_type != "UNRESOLVED" and evidence else UNKNOWN
    type_dimension = _dimension(
        type_state,
        carrier_type if type_state == SUPPORTED else None,
        claim="The source identifies this carrier type." if type_state == SUPPORTED else None,
        basis=carrier_type_basis,
        evidence_ids=evidence,
        limitation="The source does not identify a bounded institutional or personal carrier type." if type_state != SUPPORTED else None,
        packet=packet,
    )
    carrier_state = (
        PROPOSED if commitment_state == PROPOSED
        else SUPPORTED if scope_state == SUPPORTED and explicit_attribution["state"] == SUPPORTED and commitment_state == SUPPORTED
        else UNKNOWN
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "state": carrier_state,
        "commitment_state": commitment["state"],
        "semantic_type": semantic_type or None,
        "validation_state": validation_state,
        "source_text": source or None,
        "commitment": commitment,
        "type": carrier_type if type_state == SUPPORTED else "UNRESOLVED",
        "type_dimension": type_dimension,
        "scope": scope_dimension,
        "explicit_attribution": explicit_attribution,
        "actor_ids": actor_ids if explicit_attribution["state"] == SUPPORTED and carrier_type == "PERSON" else [],
        "speaker_actor_ids": actor_ids if explicit_attribution["state"] == SUPPORTED else [],
        "action_kind": action_kind,
        "evidence_ids": sorted(set(scope_dimension.get("evidence_ids", []) + commitment.get("evidence_ids", []) + explicit_attribution.get("evidence_ids", []))),
        "limitations": [
            "Scope does not establish actor identity, authority, fulfilment, institutional control, or causation.",
            *(["The proposition remains PROPOSED; it is not a supported commitment finding."] if commitment_state == PROPOSED else []),
            *(["Passive wording leaves the actor unresolved."] if scope == "PASSIVE" else []),
        ],
    }


def _matter_id(obj: Mapping[str, Any]) -> str | None:
    value = obj.get("matter_id")
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        value = next((item for item in value if _text(item)), None)
    matter = _text(value)
    if not matter or matter.upper() in {"UNKNOWN", "UNRESOLVED", "NONE"}:
        return None
    return matter


def _object_evidence(obj: Mapping[str, Any]) -> list[str]:
    values = _ids(obj.get("evidence_ids"))
    for author in obj.get("authors") or []:
        if isinstance(author, Mapping):
            values.extend(_ids(author.get("evidence_ids")))
    disposition = _mapping(obj.get("disposition"))
    values.extend(_ids(disposition.get("evidence_ids")))
    action = _mapping(obj.get("action"))
    values.extend(_ids(action.get("evidence_ids")))
    return _dedupe(values)


def _relation_is_admitted(relation: Mapping[str, Any]) -> bool:
    if _text(relation.get("status")).upper() not in _POSITIVE_RELATION_STATES:
        return False
    if _text(relation.get("validation_state")).upper() != "VALID":
        return False
    for key in ("review_state", "admission_state", "admission_route", "validation_state"):
        if _is_proposed(relation.get(key)):
            return False
    return True


def _action_is_admitted(action: Mapping[str, Any]) -> bool:
    if _text(action.get("state")).upper() not in {"OBSERVED_ALIGNED_ACTION", "DOCUMENTED_RELATED_ACTION", "RECORDED_ACTION"}:
        return False
    for key in ("validation_state", "admission_state", "review_state", "admission_route"):
        if _is_proposed(action.get(key)):
            return False
    return True


def _date_for_object(obj: Mapping[str, Any]) -> str | None:
    value = _text(obj.get("action_date"))
    if value:
        return value[:10]
    basis = _text(obj.get("action_date_basis")).upper()
    allowed = {
        "SIGNATURE_DATE",
        "VIREILLETULO_EVENT",
        "INSTITUTIONAL_DECISION_DATE",
        "SUBMISSION_DATE",
        "ANSWER_EVENT",
        "SESSION_DATE",
        "SPEECH_DATE",
    }
    if basis in allowed:
        value = _text(obj.get("date"))
        if value:
            return value[:10]
    return None


def _object_stage(obj: Mapping[str, Any]) -> str:
    kind = _text(obj.get("kind")).upper()
    if kind == "LEGISLATIVE_INITIATIVE" and obj.get("action_date_basis") == "SIGNATURE_DATE":
        return "INITIATIVE_SIGNED"
    return {
        "LEGISLATIVE_INITIATIVE": "INITIATIVE_FILED",
        "WRITTEN_QUESTION": "QUESTION_FILED",
        "GOVERNMENT_ANSWER": "ANSWER_RECORDED",
        "VOTE": "VOTE_RECORDED",
        "SPEECH": "SPEECH_DELIVERED",
        "RESIGN_ROLE": "ROLE_DECISION_RECORDED",
    }.get(kind, "OFFICIAL_OBJECT_RECORDED")


def _candidate_object_ids(packet: Mapping[str, Any]) -> set[str]:
    ids: set[str] = set()
    for relation in packet.get("relations") or []:
        if isinstance(relation, Mapping) and _relation_is_admitted(relation):
            object_id = _text(relation.get("object_id"))
            if object_id:
                ids.add(object_id)
    for action in packet.get("actions") or []:
        if isinstance(action, Mapping) and _action_is_admitted(action):
            object_id = _text(action.get("object_id"))
            if object_id:
                ids.add(object_id)
    return ids


def _episode_id(matter_id: str, object_ids: Sequence[str]) -> str:
    raw = matter_id + "|" + "|".join(sorted(object_ids))
    return "decision-episode:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def derive_decision_episodes(packet: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Group admitted official objects by exact source matter identifier.

    This intentionally does not use normalized targets, lexical similarity, or
    relation status to merge distinct matters. A question and its answer can
    share an episode only when the official records carry the same matter_id.
    """

    if not isinstance(packet, Mapping):
        raise TypeError("packet must be a mapping")
    objects = {
        _text(item.get("object_id")): item
        for item in packet.get("retrieved_objects") or []
        if isinstance(item, Mapping) and _text(item.get("object_id"))
    }
    selected = _candidate_object_ids(packet)
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for object_id in sorted(selected):
        obj = objects.get(object_id)
        matter_id = _matter_id(obj or {})
        if obj is None or matter_id is None:
            continue
        grouped.setdefault(matter_id, []).append(obj)
    action_by_object: dict[str, list[Mapping[str, Any]]] = {}
    for action in packet.get("actions") or []:
        if not isinstance(action, Mapping) or not _action_is_admitted(action):
            continue
        action_by_object.setdefault(_text(action.get("object_id")), []).append(action)

    episodes: list[dict[str, Any]] = []
    for matter_id in sorted(grouped):
        group = grouped[matter_id]
        object_ids = sorted(_text(obj.get("object_id")) for obj in group)
        object_refs = _dedupe(ref for obj in group for ref in _object_evidence(obj))
        relation_refs = _dedupe(
            ref
            for relation in packet.get("relations") or []
            if isinstance(relation, Mapping)
            and _relation_is_admitted(relation)
            and _text(relation.get("object_id")) in object_ids
            for ref in _ids(relation.get("evidence_ids"))
        )
        episode_refs = _dedupe(object_refs + relation_refs)
        sequence: list[dict[str, Any]] = []
        decisions: list[dict[str, Any]] = []
        recorded_actions: list[dict[str, Any]] = []
        for obj in sorted(
            group,
            key=lambda item: (_date_for_object(item) or "9999-99-99", _text(item.get("object_id"))),
        ):
            object_id = _text(obj.get("object_id"))
            obj_refs = _object_evidence(obj)
            date = _date_for_object(obj)
            sequence.append({
                "stage": _object_stage(obj),
                "object_id": object_id,
                "kind": _text(obj.get("kind")).upper() or "UNRESOLVED",
                "date": date,
                "date_basis": _text(obj.get("action_date_basis") or "UNRESOLVED"),
                "evidence_ids": _known_ids(packet, obj_refs),
            })
            disposition = _mapping(obj.get("disposition"))
            disposition_state = _text(disposition.get("state")).upper()
            disposition_refs = _ids(disposition.get("evidence_ids"))
            if disposition_state and disposition_state not in {"UNRESOLVED", "UNKNOWN", "NOT_ASSESSED"}:
                decision = {
                    "object_id": object_id,
                    "state": disposition_state,
                    "date": _text(disposition.get("date")) or date,
                    "evidence_ids": _known_ids(packet, disposition_refs or obj_refs),
                }
                decisions.append(decision)
            for action in action_by_object.get(object_id, []):
                action_refs = _known_ids(packet, _ids(action.get("evidence_ids")) + obj_refs)
                recorded_actions.append({
                    "object_id": object_id,
                    "kind": _text(action.get("kind") or obj.get("kind")).upper() or "UNRESOLVED",
                    "date": _text(action.get("date")) or date,
                    "actor_id": _text(action.get("actor_id")) or None,
                    "state": _text(action.get("state")).upper() or "RECORDED_ACTION",
                    "evidence_ids": action_refs,
                })
        policy_identity = _dimension(
            SUPPORTED,
            matter_id,
            claim="Official records share this exact matter identifier.",
            basis="OFFICIAL_OBJECT_MATTER_ID",
            evidence_ids=episode_refs,
            packet=packet,
        )
        decision_dimension = (
            _dimension(
                SUPPORTED,
                decisions,
                claim="The official matter sequence contains a source-backed institutional disposition.",
                basis="OFFICIAL_DISPOSITION",
                evidence_ids=_dedupe(ref for decision in decisions for ref in decision["evidence_ids"]),
                packet=packet,
            )
            if decisions
            else _dimension(UNKNOWN, limitation="No source-backed disposition was supplied for this matter.", packet=packet)
        )
        action_dimension = (
            _dimension(
                SUPPORTED,
                recorded_actions,
                claim="The canonical trace records an action on this exact official matter.",
                basis="CANONICAL_TRACE_ACTION",
                evidence_ids=_dedupe(ref for action in recorded_actions for ref in action["evidence_ids"]),
                packet=packet,
            )
            if recorded_actions
            else _dimension(UNKNOWN, limitation="No admitted actor action is linked to this matter episode.", packet=packet)
        )
        episodes.append({
            "schema_version": SCHEMA_VERSION,
            "episode_id": _episode_id(matter_id, object_ids),
            "state": SUPPORTED if policy_identity["state"] == SUPPORTED else UNKNOWN,
            "matter_id": matter_id,
            "matter_identity": policy_identity,
            "object_ids": object_ids,
            "sequence": sequence,
            "actions": recorded_actions,
            "action": action_dimension,
            "decision": decision_dimension,
            "decisions": decisions,
            "evidence_ids": sorted(set(
                episode_refs
                + policy_identity.get("evidence_ids", [])
                + decision_dimension.get("evidence_ids", [])
                + action_dimension.get("evidence_ids", [])
            )),
            "limitations": [
                "Exact matter identity does not establish that the actor caused the institutional decision or outcome.",
                "Objects with different matter IDs are intentionally separate episodes even when their topics are similar.",
            ],
        })
    return episodes


def _actor_matches(left: Any, right: Any) -> bool:
    left_text = _text(left).casefold().removeprefix("mp-")
    right_text = _text(right).casefold().removeprefix("mp-")
    return bool(left_text and right_text and left_text == right_text)


def _author_candidate(obj: Mapping[str, Any]) -> dict[str, Any] | None:
    authors = obj.get("authors")
    if not isinstance(authors, Sequence) or isinstance(authors, (str, bytes)):
        return None
    priorities = {
        "FIRST_AUTHOR": 0,
        "AUTHOR": 1,
    }
    candidates = [
        item for item in authors
        if isinstance(item, Mapping) and _text(item.get("role")).upper() in priorities
    ]
    if not candidates:
        return None
    return min(enumerate(candidates), key=lambda pair: (priorities[_text(pair[1].get("role")).upper()], pair[0]))[1]


def _first_signatory(
    packet: Mapping[str, Any],
    objects: Sequence[Mapping[str, Any]],
    actor_id: str | None,
) -> dict[str, Any]:
    for obj in objects:
        author = _author_candidate(obj)
        if author is None:
            continue
        author_ids = _dedupe([author.get("actor_id"), author.get("person_id")])
        if actor_id and not any(_actor_matches(actor_id, item) for item in author_ids):
            continue
        refs = _known_ids(packet, _ids(author.get("evidence_ids")) + _ids(obj.get("evidence_ids")))
        value = {
            "actor_id": _text(author.get("actor_id")) or None,
            "person_id": _text(author.get("person_id")) or None,
            "name": _text(author.get("name")) or None,
            "role": _text(author.get("role")).upper() or "UNRESOLVED",
            "object_id": _text(obj.get("object_id")),
        }
        if refs:
            return _dimension(
                SUPPORTED,
                value,
                claim="The official record identifies this first/primary signatory role.",
                basis="OFFICIAL_AUTHOR_ROLE",
                evidence_ids=refs,
                packet=packet,
            )
    return _dimension(
        UNKNOWN,
        limitation="This actor is not the explicit first author/signatory in the supplied official record, or that role lacks evidence.",
        packet=packet,
    )


def _actor_role(packet: Mapping[str, Any], objects: Sequence[Mapping[str, Any]], actor_id: str | None) -> dict[str, Any]:
    if not actor_id:
        return _dimension(UNKNOWN, limitation="The canonical packet has no actor ID.", packet=packet)
    for obj in objects:
        authors = obj.get("authors")
        if not isinstance(authors, Sequence) or isinstance(authors, (str, bytes)):
            continue
        for author in authors:
            if not isinstance(author, Mapping):
                continue
            author_ids = _dedupe([author.get("actor_id"), author.get("person_id")])
            if not any(_actor_matches(actor_id, item) for item in author_ids):
                continue
            refs = _known_ids(packet, _ids(author.get("evidence_ids")) + _ids(obj.get("evidence_ids")))
            if refs:
                return _dimension(
                    SUPPORTED,
                    {
                        "actor_id": _text(author.get("actor_id")) or None,
                        "person_id": _text(author.get("person_id")) or None,
                        "name": _text(author.get("name")) or None,
                        "role": _text(author.get("role")).upper() or "UNRESOLVED",
                        "object_id": _text(obj.get("object_id")),
                    },
                    claim="The official record attributes this typed role to the canonical actor.",
                    basis="OFFICIAL_ACTOR_ROLE",
                    evidence_ids=refs,
                    packet=packet,
                )
    return _dimension(
        UNKNOWN,
        limitation="The official episode does not source a role for this canonical actor.",
        packet=packet,
    )


def _explicit_drafter(
    packet: Mapping[str, Any],
    objects: Sequence[Mapping[str, Any]],
    actor_id: str | None = None,
) -> dict[str, Any]:
    """Return a drafter fact only for the actor being enveloped.

    An official record may name a drafter, but that does not make every
    packet actor the drafter.  In particular, a co-signer or a speech actor
    must not inherit another person's drafter field.  When ``actor_id`` is
    supplied, an explicit candidate without a matching identity is therefore
    retained as an unresolved fact rather than promoted for this envelope.
    """

    for obj in objects:
        candidate: Any = None
        for key in ("drafter", "drafted_by", "drafter_actor_id", "drafted_by_actor_id"):
            if obj.get(key) is not None:
                candidate = obj.get(key)
                break
        if candidate is None:
            authors = obj.get("authors")
            if isinstance(authors, Sequence) and not isinstance(authors, (str, bytes)):
                candidate = next(
                    (author for author in authors if isinstance(author, Mapping) and _text(author.get("role")).upper() == "DRAFTER"),
                    None,
                )
        if candidate is None:
            continue
        if isinstance(candidate, Mapping):
            value = {
                "actor_id": _text(candidate.get("actor_id")) or None,
                "person_id": _text(candidate.get("person_id")) or None,
                "name": _text(candidate.get("name")) or None,
                "role": _text(candidate.get("role")).upper() or "DRAFTER",
                "object_id": _text(obj.get("object_id")),
            }
            refs = _known_ids(packet, _ids(candidate.get("evidence_ids")) + _ids(obj.get("evidence_ids")))
        else:
            value = {"actor_id": _text(candidate), "object_id": _text(obj.get("object_id"))}
            refs = _known_ids(packet, _ids(obj.get("evidence_ids")))
        candidate_ids = _dedupe([value.get("actor_id"), value.get("person_id")])
        if actor_id and (not candidate_ids or not any(_actor_matches(actor_id, item) for item in candidate_ids)):
            continue
        if refs:
            return _dimension(
                SUPPORTED,
                value,
                claim="The official source explicitly records a drafter; this is not inferred from signatory order.",
                basis="EXPLICIT_DRAFTER_FIELD",
                evidence_ids=refs,
                packet=packet,
            )
    return _dimension(
        UNKNOWN,
        limitation="The supplied official records do not identify a drafter; first signatory is not treated as drafter.",
        packet=packet,
    )


def _authority_dimension(packet: Mapping[str, Any]) -> dict[str, Any]:
    authority = _mapping(packet.get("authority"))
    opportunity = _text(authority.get("opportunity_state")).upper()
    if opportunity not in {"OBSERVABLE_OPPORTUNITY", "NO_OBSERVABLE_OPPORTUNITY"}:
        return _dimension(
            UNKNOWN,
            limitation="Role interval or opportunity state is unresolved.",
            packet=packet,
        )
    refs = _known_ids(packet, _ids(authority.get("evidence_ids")))
    roles = authority.get("roles") if isinstance(authority.get("roles"), list) else []
    refs = _dedupe(refs + [ref for role in roles if isinstance(role, Mapping) for ref in _ids(role.get("evidence_ids"))])
    value = {
        "opportunity_state": opportunity,
        "required_capability": _text(authority.get("required_capability")) or None,
        "required_role": _text(authority.get("required_role")) or None,
        "window": authority.get("window") if isinstance(authority.get("window"), Mapping) else None,
    }
    return _dimension(
        SUPPORTED,
        value,
        claim="The canonical authority interval records this opportunity state; it does not imply control of outcomes.",
        basis="CANONICAL_AUTHORITY_INTERVAL",
        evidence_ids=refs,
        limitation="Authority state was present but its role evidence was incomplete." if not refs else None,
        packet=packet,
    )


def _recorded_action_dimension(
    packet: Mapping[str, Any],
    actor_id: str | None,
    episode: Mapping[str, Any],
) -> dict[str, Any]:
    actions = episode.get("actions")
    if not isinstance(actions, Sequence) or isinstance(actions, (str, bytes)) or not actions:
        return _dimension(UNKNOWN, limitation="No admitted action is linked to this episode.", packet=packet)
    matching: list[Mapping[str, Any]] = []
    mismatched = False
    for action in actions:
        if not isinstance(action, Mapping):
            continue
        action_actor = _text(action.get("actor_id"))
        if actor_id and action_actor and _actor_matches(actor_id, action_actor):
            matching.append(action)
        elif actor_id and action_actor:
            mismatched = True
    if not matching:
        return _dimension(
            UNKNOWN,
            limitation="The official action is attributed to a different or unresolved actor; no actor action is promoted.",
            basis="ACTOR_MISMATCH" if mismatched else "ACTOR_UNRESOLVED",
            packet=packet,
        )
    refs = _dedupe(ref for action in matching for ref in _known_ids(packet, _ids(action.get("evidence_ids"))))
    return _dimension(
        SUPPORTED,
        [dict(action) for action in matching],
        claim="The canonical trace records this actor performing the typed official action.",
        basis="CANONICAL_ACTOR_ACTION",
        evidence_ids=refs,
        packet=packet,
    )


def _episode_objects(packet: Mapping[str, Any], episode: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    ids = set(_ids(episode.get("object_ids")))
    return [
        item for item in packet.get("retrieved_objects") or []
        if isinstance(item, Mapping) and _text(item.get("object_id")) in ids
    ]


def build_attribution_envelope(
    packet: Mapping[str, Any],
    episode: Mapping[str, Any],
    carrier: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one actor x exact-matter envelope without causal inference."""

    resolved_carrier = carrier or resolve_commitment_carrier(packet)
    actor_id = _text(packet.get("actor_id")) or None
    objects = _episode_objects(packet, episode)
    explicit_actor_ids = _ids(_mapping(resolved_carrier.get("explicit_attribution")).get("value"))
    if actor_id and explicit_actor_ids and any(_actor_matches(actor_id, item) for item in explicit_actor_ids):
        actor_identity = _dimension(
            SUPPORTED,
            actor_id,
            claim="The canonical actor matches the explicitly attributed statement actor.",
            basis="CANONICAL_ACTOR_AND_SOURCE_ATTRIBUTION",
            evidence_ids=_mapping(resolved_carrier.get("explicit_attribution")).get("evidence_ids", []),
            packet=packet,
        )
    elif explicit_actor_ids:
        actor_identity = _dimension(
            UNKNOWN,
            limitation="The canonical actor does not match the source-attributed actor IDs.",
            basis="ACTOR_MISMATCH",
            packet=packet,
        )
    else:
        actor_identity = _dimension(
            UNKNOWN,
            limitation="No explicit statement-side actor attribution is available.",
            basis="NO_EXPLICIT_ACTOR_ATTRIBUTION",
            packet=packet,
        )
    action = _recorded_action_dimension(packet, actor_id, episode)
    authority = _authority_dimension(packet)
    decision = _safe_supplied_dimension(
        packet,
        episode.get("decision"),
        limitation="The episode decision lacks complete packet evidence or is not source-supported.",
    )
    first_signatory = _first_signatory(packet, objects, actor_id)
    actor_role = _actor_role(packet, objects, actor_id)
    drafter = _explicit_drafter(packet, objects, actor_id)
    statement_evidence = _ids(resolved_carrier.get("evidence_ids"))
    source_text = _text(resolved_carrier.get("source_text")) or _source_text(packet, _mapping(packet.get("proposition")))
    statement_said = _dimension(
        SUPPORTED if source_text and statement_evidence and actor_identity["state"] == SUPPORTED else UNKNOWN,
        source_text or None,
        claim="The supplied statement source records these words; this does not establish their truth or fulfillment.",
        basis="STATEMENT_SOURCE",
        evidence_ids=statement_evidence,
        limitation="The statement span is not source-attributed to this actor." if actor_identity["state"] != SUPPORTED else None,
        packet=packet,
    )
    if actor_role.get("state") == SUPPORTED and _text(_mapping(actor_role.get("value")).get("role")).upper() == "COSIGNER":
        cosigned = _dimension(
            SUPPORTED,
            actor_role.get("value"),
            claim="The official record identifies the canonical actor as a co-signer; co-signature is not authorship or drafting.",
            basis="OFFICIAL_COSIGNER_ROLE",
            evidence_ids=actor_role.get("evidence_ids", []),
            packet=packet,
        )
    else:
        cosigned = _dimension(
            UNKNOWN,
            limitation="No source-backed co-signature for this canonical actor is present in the episode.",
            basis="NO_COSIGNER_RECORD",
            packet=packet,
        )
    # These dimensions are deliberately explicit even when the current
    # official records cannot answer them.  A recorded object or vote does
    # not by itself prove support/opposition, implementation responsibility,
    # institutional responsibility, policy outcome, or pivotality.
    exact_support_opposition = _dimension(
        UNKNOWN,
        limitation="No complete motion, alternatives, and source-backed position mapping was supplied.",
        basis="POSITION_NOT_ESTABLISHED",
        packet=packet,
    )
    documented_constraints = _dimension(
        UNKNOWN,
        limitation="Public records supplied here do not establish private or contextual constraints on the actor.",
        basis="CONSTRAINTS_NOT_OBSERVED",
        packet=packet,
    )
    institutional_responsibility = _dimension(
        UNKNOWN,
        limitation="An institutional decision is not evidence that this actor was institutionally responsible for it.",
        basis="RESPONSIBILITY_NOT_ESTABLISHED",
        packet=packet,
    )
    implementation_responsibility = _dimension(
        UNKNOWN,
        limitation="The supplied records do not establish implementation authority or responsibility.",
        basis="IMPLEMENTATION_RESPONSIBILITY_NOT_ESTABLISHED",
        packet=packet,
    )
    pivotality = _dimension(
        UNKNOWN,
        limitation="The supplied records do not establish that this actor or action was pivotal to the institutional result.",
        basis="PIVOTALITY_NOT_ESTABLISHED",
        packet=packet,
    )
    policy_outcome = _dimension(
        UNKNOWN,
        limitation="An official disposition is not a measured policy implementation or outcome.",
        basis="POLICY_OUTCOME_NOT_OBSERVED",
        packet=packet,
    )
    carrier_commitment = _safe_supplied_dimension(
        packet,
        resolved_carrier.get("commitment"),
        limitation="The carrier commitment lacks complete packet evidence or is not source-supported.",
    )
    carrier_type_dimension = _safe_supplied_dimension(
        packet,
        resolved_carrier.get("type_dimension"),
        limitation="The carrier type lacks complete packet evidence or is not source-supported.",
    )
    carrier_scope_dimension = _safe_supplied_dimension(
        packet,
        resolved_carrier.get("scope"),
        limitation="The carrier scope lacks complete packet evidence or is not source-supported.",
    )
    commitment_state = _text(resolved_carrier.get("commitment_state") or carrier_commitment.get("state")).upper() or UNKNOWN
    if commitment_state == PROPOSED:
        action_congruence = _dimension(
            PROPOSED,
            limitation="A proposed commitment classification cannot be promoted to an action-congruence finding.",
            basis="PROPOSED_COMMITMENT",
            packet=packet,
        )
        fulfillment = _dimension(
            PROPOSED,
            limitation="A proposed commitment classification cannot be promoted to full fulfillment.",
            basis="PROPOSED_COMMITMENT",
            packet=packet,
        )
    elif commitment_state != SUPPORTED or action.get("state") != SUPPORTED:
        action_congruence = _dimension(
            UNKNOWN,
            limitation="A source-reviewed commitment and a matching recorded action are both required for action congruence.",
            basis="INSUFFICIENT_COMMITMENT_OR_ACTION",
            packet=packet,
        )
        fulfillment = _dimension(
            UNKNOWN,
            limitation="Full fulfillment, implementation, and outcome cannot be established from the supplied records.",
            basis="FULL_FULFILLMENT_NOT_OBSERVED",
            packet=packet,
        )
    else:
        action_congruence = _dimension(
            SUPPORTED,
            "RECORDED_ACTION_ONLY",
            claim="The source-reviewed commitment and the actor's recorded action are linked to this exact episode; this does not establish full fulfillment or outcome.",
            basis="ACTION_COMMITMENT_LINK",
            evidence_ids=_dedupe(
                _ids(action.get("evidence_ids"))
                + _ids(carrier_commitment.get("evidence_ids"))
            ),
            packet=packet,
        )
        fulfillment = _dimension(
            UNKNOWN,
            limitation="A recorded action establishes neither complete fulfillment nor policy implementation or outcome.",
            basis="FULL_FULFILLMENT_NOT_OBSERVED",
            packet=packet,
        )
    causal_effect = _dimension(
        UNKNOWN,
        limitation="An action or institutional disposition is not evidence of cause, effect, or responsibility.",
        basis="CAUSALITY_NOT_ESTABLISHED",
        packet=packet,
    )
    private_constraints = _dimension(
        UNKNOWN,
        limitation="Private constraints, intent, and unobserved influence are outside these public records.",
        basis="PRIVATE_STATE_UNOBSERVED",
        packet=packet,
    )
    supported_dimensions = [
        actor_identity,
        statement_said,
        action,
        authority,
        decision,
        first_signatory,
        actor_role,
        cosigned,
        drafter,
        action_congruence,
    ]
    envelope_state = (
        PROPOSED if commitment_state == PROPOSED
        else SUPPORTED if any(item.get("state") == SUPPORTED for item in supported_dimensions)
        else UNKNOWN
    )
    refs = sorted({
        ref
        for item in supported_dimensions
        if item.get("state") == SUPPORTED
        for ref in _ids(item.get("evidence_ids"))
    })
    return {
        "schema_version": SCHEMA_VERSION,
        "envelope_id": "attribution:" + _text(packet.get("trace_id")) + ":" + _text(episode.get("episode_id")),
        "state": envelope_state,
        "actor_id": actor_id,
        "episode_id": _text(episode.get("episode_id")),
        "matter_id": _text(episode.get("matter_id")) or None,
        "dimensions": {
            "actor_identity": actor_identity,
            "said": statement_said,
            "statement_said": statement_said,
            "committed": carrier_commitment,
            "carrier_type": carrier_type_dimension,
            "scope": carrier_scope_dimension,
            "authority": authority,
            "recorded_action": action,
            "actor_role": actor_role,
            "cosigned": cosigned,
            "institutional_decision": decision,
            "first_signatory": first_signatory,
            "drafter": drafter,
            "action_congruence": action_congruence,
            "commitment_fulfillment": fulfillment,
            "exact_support_opposition": exact_support_opposition,
            "documented_constraints": documented_constraints,
            "institutional_responsibility": institutional_responsibility,
            "implementation_responsibility": implementation_responsibility,
            "policy_outcome": policy_outcome,
            "pivotality": pivotality,
            "causal_effect": causal_effect,
            "private_constraints": private_constraints,
        },
        "evidence_ids": refs,
        "limitations": [
            "First signatory is not treated as drafter unless a drafter field or role is explicitly sourced.",
            "An actor action is not a causal-effect or responsibility finding.",
            "A model/unresolved/proposed relation cannot be promoted through this envelope.",
        ],
    }


def build_attribution_envelopes(
    packet: Mapping[str, Any],
    *,
    carrier: Mapping[str, Any] | None = None,
    episodes: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    resolved_carrier = carrier or resolve_commitment_carrier(packet)
    selected = list(episodes) if episodes is not None else derive_decision_episodes(packet)
    return [build_attribution_envelope(packet, episode, resolved_carrier) for episode in selected]


def attribution_artifact(
    packet: Mapping[str, Any],
    *,
    proposition: Mapping[str, Any] | None = None,
    episodes: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return the standalone optional attribution artifact for a trace packet."""

    carrier = resolve_commitment_carrier(packet, proposition=proposition)
    selected = list(episodes) if episodes is not None else derive_decision_episodes(packet)
    return {
        "schema_version": SCHEMA_VERSION,
        "trace_id": _text(packet.get("trace_id")) or None,
        "proposition_id": _text(packet.get("proposition_id")) or None,
        "commitment_carrier": carrier,
        "decision_episodes": selected,
        "attribution_envelopes": build_attribution_envelopes(packet, carrier=carrier, episodes=selected),
    }


def attach_attribution(
    packet: Mapping[str, Any],
    *,
    proposition: Mapping[str, Any] | None = None,
    episodes: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return a packet copy with the optional attribution fields attached."""

    if not isinstance(packet, Mapping):
        raise TypeError("packet must be a mapping")
    result = deepcopy(dict(packet))
    artifact = attribution_artifact(packet, proposition=proposition, episodes=episodes)
    result.update({
        "commitment_carrier": artifact["commitment_carrier"],
        "decision_episodes": artifact["decision_episodes"],
        "attribution_envelopes": artifact["attribution_envelopes"],
    })
    return result


__all__ = [
    "PROPOSED",
    "SCHEMA_VERSION",
    "SUPPORTED",
    "UNKNOWN",
    "UNRESOLVED",
    "attach_attribution",
    "attribution_artifact",
    "build_attribution_envelope",
    "build_attribution_envelopes",
    "derive_decision_episodes",
    "resolve_commitment_carrier",
]

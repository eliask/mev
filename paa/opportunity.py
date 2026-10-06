"""Authority and opportunity checks for personal commitments.

The absence of a later record is never itself a finding that somebody did
nothing. This module answers the narrower prerequisite question: could the
named act have been performed by this actor, in the relevant role and
interval? It accepts both the old positional call shape and proposition/role
records used by the trace backend.
"""


import re
from collections.abc import Mapping, Sequence
from typing import Any

from paa.opportunity_codec import plan_to_wire
from paa.opportunity_records import (
    ActionKind,
    ActionRequirements,
    RequiredCapability,
    UnresolvedRequirementReason,
    VerificationNotApplicable,
    VerificationPlan,
    VerificationUnresolved,
)

_ELECTION_CONDITION = re.compile(
    r"\b(?:jos|mikäli|kun)\b[^.!?]{0,180}?"
    r"(?:pääsen\s+eduskuntaan|valit(?:aan|uksi)|valitset\s+minut\s+eduskuntaan|"
    r"tulen\s+valituksi|kansanedustajaksi)\b",
    re.IGNORECASE,
)

_ACTION_TYPES = frozenset({"PERSONAL_ACTION_COMMITMENT", "PERSONAL_RESTRAINT_COMMITMENT"})

_ACTION_LABELS = {
    "INITIATIVE_AUTHORED": "aloitteen jättämistä",
    "VOTE_CAST": "äänestämistä",
    "QUESTION_FILED": "kirjallisen kysymyksen jättämistä",
    "SPEECH_DELIVERED": "eduskuntapuheenvuoroa",
    "RESIGN_ROLE": "luottamustoimen tai paikan jättämistä",
    "DONATION": "lahjoittamista",
    "PUBLIC_ADVOCACY": "kampanjointia tai muuta nimettyä julkista toimintaa",
    "POLICY_RESTRAINT": "nimettyyn politiikkatoimeen osallistumista koskevaa pidättyvyyttä",
    "OTHER_OBSERVABLE_ACTION": "nimettyä julkista tekoa",
}

_ROLE_LABELS = {
    "MP_INITIATE_BILL": "kansanedustajan aloiteoikeutta",
    "PARLIAMENTARY_VOTE": "kansanedustajan äänestysoikeutta",
    "FILE_PARLIAMENTARY_QUESTION": "kansanedustajan kysymysoikeutta",
    "SPEAK_IN_PARLIAMENT": "kansanedustajan puheoikeutta",
    "HOLD_ELECTED_ROLE": "voimassa olevaa luottamustoimea",
    "PARLIAMENTARY_INFLUENCE": "parlamentaarista vaikutusmahdollisuutta",
    "POLICYMAKING_ROLE": "toimivaltaista päätöksentekoroolia",
    "PUBLIC_ADVOCACY": "julkista vaikuttamismahdollisuutta",
    "PERSONAL_FUNDS": "henkilökohtaisia varoja",
    "OTHER": "lähteissä osoitettua toimivaltaa",
}

_ACTION_CAPABILITIES = {
    # An initiative can be parliamentary, municipal, civic or unspecified.
    # Its model label alone cannot establish parliamentary authority.
    "INITIATIVE_AUTHORED": "OTHER",
    "VOTE_CAST": "PARLIAMENTARY_VOTE",
    "QUESTION_FILED": "FILE_PARLIAMENTARY_QUESTION",
    "SPEECH_DELIVERED": "SPEAK_IN_PARLIAMENT",
    "RESIGN_ROLE": "HOLD_ELECTED_ROLE",
    "DONATION": "PERSONAL_FUNDS",
    "PUBLIC_ADVOCACY": "PUBLIC_ADVOCACY",
    "POLICY_RESTRAINT": "POLICYMAKING_ROLE",
    "OTHER_OBSERVABLE_ACTION": "OTHER",
}


def _value(record: Any, key: str, default: Any = None) -> Any:
    if isinstance(record, Mapping):
        return record.get(key, default)
    return getattr(record, key, default)


def _record_text(proposition: Any, fallback: str | None = None) -> str:
    if proposition is None:
        return fallback or ""
    return str(
        _value(proposition, "text", None)
        or _value(proposition, "source_text", None)
        or fallback
        or ""
    ).strip()


def _record_types(proposition: Any, semantic_types: Sequence[str] | None) -> list[str]:
    if semantic_types is not None:
        return [str(value) for value in semantic_types]
    if proposition is None:
        return []
    semantic_type = _value(proposition, "semantic_type", None)
    return [str(semantic_type)] if semantic_type else []


def _condition(proposition: Any, text: str) -> str | None:
    condition = _value(proposition, "condition", None) if proposition is not None else None
    if condition:
        return str(condition)
    conditions = _value(proposition, "conditions", None) if proposition is not None else None
    if conditions:
        first = conditions[0] if isinstance(conditions, Sequence) else conditions
        if isinstance(first, Mapping):
            value = first.get("value")
            if value:
                return str(value)
        elif first:
            return str(first)
    match = _ELECTION_CONDITION.search(text)
    return match.group(0).strip() if match else None


def _infer_action(text: str) -> tuple[str | None, str | None]:
    # Keep action extraction canonical in semantics.py. The opportunity layer
    # only adapts its result to role/capability reasoning.
    from paa.semantics import action_metadata

    kind, capability = action_metadata(text)
    if kind:
        return kind, capability
    if re.search(
        r"\b(?:en|emme)\s+(?:aio\s+|tule\s+)?(?:leikka|korota|lakkauta|poista|"
        r"heikennä|supista)|\blupaan\s+olla\s+(?:leikkaamatta|korottamatta)",
        text,
        re.IGNORECASE,
    ):
        return action_metadata(text, restraint=True)
    return None, None


def opportunity_requirements(
    proposition: Any = None,
    *,
    text: str | None = None,
    semantic_types: Sequence[str] | None = None,
    action_kind: str | None = None,
    required_capability: str | None = None,
) -> VerificationPlan:
    """Normalize a proposition into an explicit action/capability request.

    Broad objectives produce an explicit not-applicable plan. An ambiguous
    interpretation or unidentified action channel produces an unresolved plan.
    Neither launches an action ledger merely because a promise verb occurs.
    """

    source = _record_text(proposition, text)
    types = _record_types(proposition, semantic_types)
    semantic_type = next((item for item in types if item in _ACTION_TYPES), types[0] if types else "AMBIGUOUS")
    if not any(item in _ACTION_TYPES for item in types):
        if semantic_type == "AMBIGUOUS":
            return VerificationUnresolved(semantic_type=semantic_type,
                                          reason=UnresolvedRequirementReason.INTERPRETATION_AMBIGUOUS)
        return VerificationNotApplicable(semantic_type=semantic_type)
    inferred_kind = _value(proposition, "action_kind", None) if proposition is not None else None
    inferred_capability = _value(proposition, "required_capability", None) if proposition is not None else None
    kind = action_kind or inferred_kind
    capability = required_capability or inferred_capability
    if not kind or not capability:
        inferred_kind, inferred_capability = _infer_action(source)
        if not capability and (not kind or kind == inferred_kind):
            capability = inferred_capability
        kind = kind or inferred_kind
    if kind and not capability:
        capability = _ACTION_CAPABILITIES.get(str(kind))
    # A personal-action label without a recoverable action register is not
    # enough to launch a verification plan. Keep it open for adjudication
    # instead of inventing an OTHER event.
    if not kind:
        return VerificationUnresolved(semantic_type=semantic_type,
                                      reason=UnresolvedRequirementReason.ACTION_NOT_SPECIFIED)
    if not capability:
        return VerificationUnresolved(semantic_type=semantic_type,
                                      reason=UnresolvedRequirementReason.CAPABILITY_NOT_SPECIFIED)
    try:
        owned_kind = ActionKind(kind)
    except ValueError:
        return VerificationUnresolved(semantic_type=semantic_type,
                                      reason=UnresolvedRequirementReason.UNSUPPORTED_ACTION)
    try:
        owned_capability = RequiredCapability(capability)
    except ValueError:
        return VerificationUnresolved(semantic_type=semantic_type,
                                      reason=UnresolvedRequirementReason.UNSUPPORTED_CAPABILITY)
    condition = _condition(proposition, source)
    condition_requires_election = bool(
        condition
        and re.search(
            r"eduskuntaan|kansanedustajaksi|valit(?:aan|uksi)", condition, re.IGNORECASE
        )
    )
    action_label = _ACTION_LABELS.get(kind, "nimettyä julkista tekoa")
    if kind == "INITIATIVE_AUTHORED" and re.search(r"lakialoit", source, re.IGNORECASE):
        action_label = "lakialoitteen jättämistä"
    return ActionRequirements(
        action_kind=owned_kind, required_capability=owned_capability,
        required_role=_ROLE_LABELS.get(capability, capability), action_label=action_label,
        condition=condition, condition_requires_election=condition_requires_election,
        source_text=source, evidence_ids=tuple(_value(proposition, "evidence_ids", []) or []),
    )


def _role_capabilities(role: Mapping[str, Any]) -> set[str]:
    capabilities = role.get("capabilities") or role.get("formal_capabilities") or []
    found: set[str] = set()
    for item in capabilities:
        value = item.get("capability") if isinstance(item, Mapping) else item
        if value:
            found.add(str(value))
    return found


def _role_capabilities_declared(role: Mapping[str, Any]) -> bool:
    # A missing (or explicit null) capability field is not evidence that the
    # role has no authority. An empty list, on the other hand, is a complete
    # declaration that no capability was recorded.
    return any(role.get(key) is not None for key in ("capabilities", "formal_capabilities"))


def _role_evidence(role: Mapping[str, Any]) -> list[str]:
    evidence = list(role.get("evidence_ids") or [])
    for key in ("capabilities", "formal_capabilities"):
        for item in role.get(key) or []:
            if isinstance(item, Mapping):
                evidence.extend(str(value) for value in item.get("evidence_ids") or [])
    return sorted(set(evidence))


def _role_covers(role: Mapping[str, Any], when: str | None = None) -> bool:
    if when is None:
        return True
    start = role.get("start")
    end = role.get("end")
    # For an uncertain timebox, use the latest possible start and earliest
    # possible end. That avoids granting authority at a date that the source
    # interval does not actually establish.
    start_date = (
        (start.get("latest") or start.get("earliest"))
        if isinstance(start, Mapping)
        else role.get("start_date")
    )
    end_date = (
        (end.get("earliest") or end.get("latest"))
        if isinstance(end, Mapping)
        else role.get("end_date")
    )
    if not start_date:
        return False
    return (not start_date or str(when) >= str(start_date)) and (not end_date or str(when) <= str(end_date))


def _role_records(
    role_intervals: Sequence[Mapping[str, Any]] | None,
    role_interval: Mapping[str, Any] | None,
) -> list[Mapping[str, Any]]:
    records = [item for item in (role_intervals or []) if isinstance(item, Mapping)]
    if isinstance(role_interval, Mapping):
        records.append(role_interval)
    return records


def _provenance(
    requirement: Mapping[str, Any],
    *,
    evidence_ids: Sequence[str] | None = None,
    role_evidence_ids: Sequence[str] | None = None,
    role_records: Sequence[Mapping[str, Any]] | None = None,
    opportunity_evidence: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    statement_ids = list(requirement.get("evidence_ids") or evidence_ids or [])
    provenance: list[dict[str, Any]] = [
        {
            "kind": "STATEMENT_TEXT",
            "basis": "EXPLICIT_TEXT",
            "quote": requirement.get("source_text") or "",
            "evidence_ids": statement_ids,
        }
    ]
    if role_evidence_ids or role_records:
        role_item: dict[str, Any] = {
            "kind": "ROLE_INTERVAL",
            "basis": "SOURCE_RECORD",
            "evidence_ids": sorted(set(role_evidence_ids or [])),
        }
        role_ids = sorted({str(role["role_id"]) for role in role_records or [] if role.get("role_id")})
        if role_ids:
            role_item["role_ids"] = role_ids
        intervals = []
        for role in role_records or []:
            start = role.get("start")
            end = role.get("end")
            intervals.append(
                {
                    "role_id": role.get("role_id"),
                    "start": start if isinstance(start, Mapping) else role.get("start_date"),
                    "end": end if isinstance(end, Mapping) else role.get("end_date"),
                }
            )
        if intervals:
            role_item["intervals"] = intervals
        provenance.append(role_item)
    for item in opportunity_evidence or []:
        provenance.append(dict(item))
    return provenance


def assess_opportunity(
    text: Any = "",
    semantic_types: Sequence[str] | None = None,
    elected: bool | None = None,
    *,
    proposition: Any = None,
    role_intervals: Sequence[Mapping[str, Any]] | None = None,
    role_interval: Mapping[str, Any] | None = None,
    opportunity_evidence: Sequence[Mapping[str, Any]] | None = None,
    evidence_ids: Sequence[str] | None = None,
    action_kind: str | None = None,
    required_capability: str | None = None,
    action_date: str | None = None,
    role_evidence_complete: bool = False,
) -> dict[str, Any] | None:
    """Assess whether the source-named act had the required opportunity.

    Old callers can continue to pass (text, semantic_types, elected). New
    callers should pass a proposition record and role intervals. A positive
    opportunity is not a closure finding, so the legacy call returns None
    when the role is available.
    """

    record = proposition if proposition is not None else (text if not isinstance(text, str) else None)
    source_text = _record_text(record, text if isinstance(text, str) else None)
    plan = opportunity_requirements(
        record,
        text=source_text,
        semantic_types=semantic_types,
        action_kind=action_kind,
        required_capability=required_capability,
    )
    if type(plan) is not ActionRequirements:
        return None
    requirements = plan_to_wire(plan)

    condition = requirements["condition"]
    role_records = _role_records(role_intervals, role_interval)
    capability = requirements["required_capability"]
    has_role = any(capability in _role_capabilities(role) and _role_covers(role, action_date) for role in role_records)
    role_evidence_ids = [evidence for role in role_records for evidence in _role_evidence(role)]
    provenance = _provenance(
        requirements,
        evidence_ids=evidence_ids,
        role_evidence_ids=role_evidence_ids,
        role_records=role_records,
        opportunity_evidence=opportunity_evidence,
    )

    # A dated role/capability record is stronger than a coarse current
    # election boolean. This matters for a former MP whose promised action
    # falls inside a documented earlier mandate.
    if role_records and has_role:
        condition_state = "NOT_APPLICABLE" if not condition else "UNRESOLVED"
        if condition and requirements["condition_requires_election"] and elected is not None:
            condition_state = "NOT_SATISFIED" if elected is False else "SATISFIED"
        return {
            "state": "OPPORTUNITY_AVAILABLE",
            "action_kind": requirements["action_kind"],
            "required_capability": capability,
            "required_role": requirements["required_role"],
            "condition": condition,
            "condition_state": condition_state,
            "provenance": provenance,
        }

    if role_records and not has_role and not all(_role_capabilities_declared(role) for role in role_records):
        return {
            "state": "INSUFFICIENT_EVIDENCE",
            "claim": (
                "Roolitieto ei ilmoita muodollisia valmiuksia, joten vaaditun "
                f"{requirements['required_role']} olemassaoloa ei voi päätellä."
            ),
            "action_kind": requirements["action_kind"],
            "required_capability": capability,
            "required_role": requirements["required_role"],
            "condition": condition,
            "condition_state": "UNRESOLVED",
            "provenance": provenance,
        }

    if role_records and action_date is not None and any(
        capability in _role_capabilities(role) for role in role_records
    ):
        return {
            "state": "INSUFFICIENT_EVIDENCE",
            "claim": (
                f"Roolitieto nimeää vaaditun {requirements['required_role']}, mutta sen aikaväli "
                f"ei osoita toimivaltaa päivänä {action_date}."
            ),
            "action_kind": requirements["action_kind"],
            "required_capability": capability,
            "required_role": requirements["required_role"],
            "condition": condition,
            "condition_state": "UNRESOLVED",
            "provenance": provenance,
        }

    # Election-dependent promises have a fully explicit counterfactual state:
    # no seat means the promised parliamentary act was not available. This is
    # not a claim that the actor did nothing elsewhere.
    election_required = requirements["condition_requires_election"] or capability in {
        "MP_INITIATE_BILL",
        "PARLIAMENTARY_VOTE",
        "FILE_PARLIAMENTARY_QUESTION",
        "SPEAK_IN_PARLIAMENT",
    }
    if elected is False and election_required:
        action_kind = requirements["action_kind"]
        if action_kind == "INITIATIVE_AUTHORED" and re.search(
            r"lakialoit", requirements["source_text"], re.IGNORECASE
        ):
            claim = (
                "Teksti edellyttää kansanedustajan paikkaa samana vuonna. "
                "Vaalitulos ei tuonut paikkaa, joten lakialoitetta ei voinut jättää. "
                "Muu toiminta jää tämän aineiston ulkopuolelle."
            )
        elif requirements["condition_requires_election"]:
            claim = (
                f"Teksti on ehdollinen: {condition}. Vaalituloksen perusteella ehto ei täyttynyt, "
                f"joten lupausta vastaavaa {requirements['action_label']} ei käynnistynyt. "
                "Muu toiminta jää tämän aineiston ulkopuolelle."
            )
        else:
            claim = (
                f"Teksti edellyttää {requirements['required_role']}. "
                "Vaalitulos ei osoittanut tämän toimivallan syntyneen, joten "
                f"lupausta vastaavaa {requirements['action_label']} ei voitu tässä roolissa arvioida. "
                "Muu toiminta jää tämän aineiston ulkopuolelle."
            )
        return {
            "state": "NO_OBSERVABLE_OPPORTUNITY",
            "claim": claim,
            "action_kind": action_kind,
            "required_capability": capability,
            "required_role": requirements["required_role"],
            "condition": condition,
            "condition_state": "NOT_SATISFIED" if condition else "UNRESOLVED",
            "provenance": provenance,
        }

    if role_records and role_evidence_complete and not has_role:
        return {
            "state": "NO_OBSERVABLE_OPPORTUNITY",
            "claim": (
                f"Toimitettu rooliaineisto ei osoita vaadittua {requirements['required_role']} "
                f"ajankohtana, joten {requirements['action_label']} ei ole arvioitavissa tämän aineiston perusteella."
            ),
            "action_kind": requirements["action_kind"],
            "required_capability": capability,
            "required_role": requirements["required_role"],
            "condition": condition,
            "condition_state": "UNRESOLVED",
            "provenance": provenance,
        }

    # No negative inference when election/role state is unknown.
    return None

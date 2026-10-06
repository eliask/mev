"""Versioned retained codec for proposed authority-verification plans.

Malformed retained plans are integrity failures, never negative observations.
The boundary rejects duplicates/extra fields/non-finite numbers before creating
the owned semantic records. Source interpretation is still checked separately.
"""

import json
from typing import assert_never

from paa.opportunity_records import (
    ActionKind,
    ActionRequirements,
    RequiredCapability,
    UnresolvedRequirementReason,
    VerificationNotApplicable,
    VerificationPlan,
    VerificationUnresolved,
)

SCHEMA_VERSION = "paa.opportunity.plan.v1"
type PlanWire = dict[str, str | bool | None | list[str]]


class OpportunityCodecError(ValueError):
    """A retained plan violated its declared wire contract."""


def plan_to_wire(plan: VerificationPlan) -> PlanWire:
    if type(plan) not in (ActionRequirements, VerificationNotApplicable, VerificationUnresolved):
        raise TypeError("Expected an exact owned verification-plan variant")
    match plan:
        case ActionRequirements() as action:
            return {
                "schema_version": SCHEMA_VERSION, "state": "ACTION_REQUIREMENTS",
                "action_kind": action.action_kind.value,
                "required_capability": action.required_capability.value,
                "required_role": action.required_role, "action_label": action.action_label,
                "condition": action.condition, "condition_requires_election": action.condition_requires_election,
                "source_text": action.source_text, "evidence_ids": list(action.evidence_ids),
            }
        case VerificationNotApplicable() as inactive:
            return {"schema_version": SCHEMA_VERSION, "state": "NOT_APPLICABLE",
                    "semantic_type": inactive.semantic_type}
        case VerificationUnresolved() as unresolved:
            return {"schema_version": SCHEMA_VERSION, "state": "UNRESOLVED",
                    "semantic_type": unresolved.semantic_type, "reason": unresolved.reason.value}
        case _ as unreachable:
            assert_never(unreachable)


def encode_plan(plan: VerificationPlan) -> str:
    return json.dumps(plan_to_wire(plan), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _pairs(rows: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in rows:
        if key in result:
            raise OpportunityCodecError("Duplicate plan field")
        result[key] = value
    return result


def _constant(value: str) -> None:
    raise OpportunityCodecError(f"Non-finite plan number: {value}")


def _text(value: object) -> str:
    if type(value) is not str:
        raise OpportunityCodecError("Plan text field is not a string")
    return value


def decode_plan(text: str) -> VerificationPlan:
    if type(text) is not str or len(text.encode("utf-8")) > 2_100_000:
        raise OpportunityCodecError("Plan input must be bounded UTF-8 text")
    try:
        row = json.loads(text, object_pairs_hook=_pairs, parse_constant=_constant)
    except (json.JSONDecodeError, RecursionError) as error:
        raise OpportunityCodecError("Malformed or excessively nested plan JSON") from error
    if type(row) is not dict or row.get("schema_version") != SCHEMA_VERSION:
        raise OpportunityCodecError("Unsupported verification-plan schema")
    try:
        match row.get("state"):
            case "ACTION_REQUIREMENTS":
                fields = {"schema_version", "state", "action_kind", "required_capability", "required_role",
                          "action_label", "condition", "condition_requires_election", "source_text", "evidence_ids"}
                if set(row) != fields:
                    raise OpportunityCodecError("Unknown or missing action-plan fields")
                condition = None if row["condition"] is None else _text(row["condition"])
                ids = row["evidence_ids"]
                if type(ids) is not list or len(ids) > 4096:
                    raise OpportunityCodecError("Invalid plan reference population")
                return ActionRequirements(
                    action_kind=ActionKind(_text(row["action_kind"])),
                    required_capability=RequiredCapability(_text(row["required_capability"])),
                    required_role=_text(row["required_role"]), action_label=_text(row["action_label"]),
                    condition=condition, condition_requires_election=row["condition_requires_election"],
                    source_text=_text(row["source_text"]), evidence_ids=tuple(_text(key) for key in ids),
                )
            case "NOT_APPLICABLE":
                if set(row) != {"schema_version", "state", "semantic_type"}:
                    raise OpportunityCodecError("Unknown or missing inactive-plan fields")
                return VerificationNotApplicable(semantic_type=_text(row["semantic_type"]))
            case "UNRESOLVED":
                if set(row) != {"schema_version", "state", "semantic_type", "reason"}:
                    raise OpportunityCodecError("Unknown or missing unresolved-plan fields")
                return VerificationUnresolved(semantic_type=_text(row["semantic_type"]),
                                              reason=UnresolvedRequirementReason(_text(row["reason"])))
            case _:
                raise OpportunityCodecError("Unsupported verification-plan state")
    except (TypeError, ValueError) as error:
        raise OpportunityCodecError(f"Invalid plan value: {error}") from error

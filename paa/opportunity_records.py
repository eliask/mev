"""Owned action requirements at the interpretation-to-authority boundary.

These records describe a verification plan, never an admitted interpretation.
Source/classification adapters own extraction; authority consumers receive an
immutable plan or a distinct explanation for why no plan can be constructed.
This narrow semantic scope targets the pinned Python profile. It makes no
whole-module or whole-repository conformance claim about older adapters.
"""

from dataclasses import dataclass
from enum import Enum
from typing import assert_never, final


class ActionKind(Enum):
    INITIATIVE_AUTHORED = "INITIATIVE_AUTHORED"
    VOTE_CAST = "VOTE_CAST"
    QUESTION_FILED = "QUESTION_FILED"
    SPEECH_DELIVERED = "SPEECH_DELIVERED"
    RESIGN_ROLE = "RESIGN_ROLE"
    DONATION = "DONATION"
    PUBLIC_ADVOCACY = "PUBLIC_ADVOCACY"
    POLICY_RESTRAINT = "POLICY_RESTRAINT"
    OTHER_OBSERVABLE_ACTION = "OTHER_OBSERVABLE_ACTION"


class RequiredCapability(Enum):
    MP_INITIATE_BILL = "MP_INITIATE_BILL"
    PARLIAMENTARY_VOTE = "PARLIAMENTARY_VOTE"
    FILE_PARLIAMENTARY_QUESTION = "FILE_PARLIAMENTARY_QUESTION"
    SPEAK_IN_PARLIAMENT = "SPEAK_IN_PARLIAMENT"
    HOLD_ELECTED_ROLE = "HOLD_ELECTED_ROLE"
    PARLIAMENTARY_INFLUENCE = "PARLIAMENTARY_INFLUENCE"
    POLICYMAKING_ROLE = "POLICYMAKING_ROLE"
    PUBLIC_ADVOCACY = "PUBLIC_ADVOCACY"
    PERSONAL_FUNDS = "PERSONAL_FUNDS"
    OTHER = "OTHER"


class UnresolvedRequirementReason(Enum):
    INTERPRETATION_AMBIGUOUS = "INTERPRETATION_AMBIGUOUS"
    ACTION_NOT_SPECIFIED = "ACTION_NOT_SPECIFIED"
    CAPABILITY_NOT_SPECIFIED = "CAPABILITY_NOT_SPECIFIED"
    UNSUPPORTED_ACTION = "UNSUPPORTED_ACTION"
    UNSUPPORTED_CAPABILITY = "UNSUPPORTED_CAPABILITY"


def _check_action_capability(action: ActionKind, capability: RequiredCapability) -> None:
    match action:
        case ActionKind.INITIATIVE_AUTHORED:
            allowed = (RequiredCapability.MP_INITIATE_BILL, RequiredCapability.POLICYMAKING_ROLE,
                       RequiredCapability.PUBLIC_ADVOCACY, RequiredCapability.OTHER)
        case ActionKind.VOTE_CAST:
            allowed = (RequiredCapability.PARLIAMENTARY_VOTE,)
        case ActionKind.QUESTION_FILED:
            allowed = (RequiredCapability.FILE_PARLIAMENTARY_QUESTION,)
        case ActionKind.SPEECH_DELIVERED:
            allowed = (RequiredCapability.SPEAK_IN_PARLIAMENT,)
        case ActionKind.RESIGN_ROLE:
            allowed = (RequiredCapability.HOLD_ELECTED_ROLE,)
        case ActionKind.DONATION:
            allowed = (RequiredCapability.PERSONAL_FUNDS,)
        case ActionKind.PUBLIC_ADVOCACY:
            allowed = (RequiredCapability.PUBLIC_ADVOCACY,)
        case ActionKind.POLICY_RESTRAINT:
            allowed = (RequiredCapability.POLICYMAKING_ROLE, RequiredCapability.PARLIAMENTARY_INFLUENCE)
        case ActionKind.OTHER_OBSERVABLE_ACTION:
            allowed = tuple(RequiredCapability)
        case _ as unreachable:
            assert_never(unreachable)
    if capability not in allowed:
        raise ValueError("Capability is incompatible with the named action channel")


def _nonempty_text(value: str, *, limit: int) -> None:
    if type(value) is not str:
        raise TypeError("Plan text must be an owned string")
    if not value or len(value) > limit:
        raise ValueError("Plan text is empty or exceeds its declared limit")


@final
@dataclass(frozen=True, slots=True, kw_only=True, match_args=False)
class ActionRequirements:
    action_kind: ActionKind
    required_capability: RequiredCapability
    required_role: str
    action_label: str
    condition: str | None
    condition_requires_election: bool
    source_text: str
    evidence_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.action_kind) is not ActionKind or type(self.required_capability) is not RequiredCapability:
            raise TypeError("Action and capability require exact domain enums")
        _check_action_capability(self.action_kind, self.required_capability)
        _nonempty_text(self.required_role, limit=512)
        _nonempty_text(self.action_label, limit=512)
        _nonempty_text(self.source_text, limit=1_000_000)
        if self.condition is not None:
            _nonempty_text(self.condition, limit=1_000_000)
        if type(self.condition_requires_election) is not bool:
            raise TypeError("Election-condition marker must be a boolean")
        if self.condition_requires_election and self.condition is None:
            raise ValueError("An election condition needs a retained condition")
        if type(self.evidence_ids) is not tuple:
            raise TypeError("Evidence references must be an owned tuple")
        if len(self.evidence_ids) > 4096:
            raise ValueError("Evidence-reference population exceeds its declared limit")
        for reference in self.evidence_ids:
            _nonempty_text(reference, limit=512)
        if len(self.evidence_ids) != len(set(self.evidence_ids)):
            raise ValueError("Duplicate evidence references in action requirements")


@final
@dataclass(frozen=True, slots=True, kw_only=True, match_args=False)
class VerificationNotApplicable:
    semantic_type: str

    def __post_init__(self) -> None:
        _nonempty_text(self.semantic_type, limit=128)


@final
@dataclass(frozen=True, slots=True, kw_only=True, match_args=False)
class VerificationUnresolved:
    semantic_type: str
    reason: UnresolvedRequirementReason

    def __post_init__(self) -> None:
        _nonempty_text(self.semantic_type, limit=128)
        if type(self.reason) is not UnresolvedRequirementReason:
            raise TypeError("Unresolved plans require an exact reason enum")


type VerificationPlan = ActionRequirements | VerificationNotApplicable | VerificationUnresolved

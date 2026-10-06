"""Source-grounded multi-facet extraction for PAA-02.

This module is deliberately a successor to :mod:`paa.llm_semantics`, not a
replacement for it.  The compact v2 corpus pass gives one proposition label
to each canonical unit.  The facet pass keeps several linked propositions in
one unit and carries the source fields needed to distinguish an actor, a
carrier, a predicate, a target, and a condition.

Model output is untrusted boundary data.  ``normalize_facet_output`` checks
the complete source text, exact offsets, explicit source anchors, links and
terminal per-unit accounting.  A valid result is still ``PROPOSED``; this
module never admits a political interpretation or a fulfilment finding.
"""

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import final

PROMPT_DIR = Path(__file__).with_name("prompts")
FACET_SCHEMA_VERSION = "paa.facets.v1"
FACET_VALIDATOR_VERSION = "paa.facets.validator.v3-owned-compact-binding"
FACET_PROMPT_VERSION = "facets_v3_source_fidelity"
FACET_PROMPT_VERSIONS = (
    "facets_v1_source_first",
    "facets_v2_linked_propositions",
    FACET_PROMPT_VERSION,
)


class FacetKind(Enum):
    """Small transport vocabulary; none of these values is an admission."""

    FUTURE_COMMITMENT = "FUTURE_COMMITMENT"
    PAST_FACT = "PAST_FACT"
    CURRENT_FACT = "CURRENT_FACT"
    POLICY_GOAL = "POLICY_GOAL"
    POSITION = "POSITION"
    FORECAST = "FORECAST"
    QUESTION = "QUESTION"
    REPORTED_SPEECH = "REPORTED_SPEECH"
    AMBIGUOUS = "AMBIGUOUS"


class HorizonKind(Enum):
    PAST = "PAST"
    CURRENT = "CURRENT"
    FUTURE = "FUTURE"
    ATEMPORAL = "ATEMPORAL"
    UNRESOLVED = "UNRESOLVED"


class ActorScope(Enum):
    SELF = "SELF"
    COLLECTIVE = "COLLECTIVE"
    INSTITUTION = "INSTITUTION"
    OTHER = "OTHER"
    UNRESOLVED = "UNRESOLVED"


class FacetUnitStatus(Enum):
    COMPLETE = "COMPLETE"
    ABSTAIN = "ABSTAIN"
    INVALID = "INVALID"


class PublicationState(Enum):
    """Closed publication state for model-derived facet records."""

    PROPOSED = "PROPOSED"


class ActionKind(Enum):
    """Small closed action transport vocabulary; not an admission."""

    INITIATIVE_AUTHORED = "INITIATIVE_AUTHORED"
    LAW_INITIATIVE = "LAW_INITIATIVE"
    VOTE_CAST = "VOTE_CAST"
    QUESTION_FILED = "QUESTION_FILED"
    SPEECH_DELIVERED = "SPEECH_DELIVERED"
    RESIGN_ROLE = "RESIGN_ROLE"
    DONATION = "DONATION"
    PUBLIC_ADVOCACY = "PUBLIC_ADVOCACY"
    POLICY_RESTRAINT = "POLICY_RESTRAINT"
    OTHER_OBSERVABLE_ACTION = "OTHER_OBSERVABLE_ACTION"


class FacetResultStatus(Enum):
    VALID = "VALID"
    PARTIAL = "PARTIAL"
    ABSTAIN = "ABSTAIN"
    INVALID = "INVALID"


class InvalidCode(Enum):
    INVALID_JSON = "INVALID_JSON"
    OUTPUT_NOT_OBJECT = "OUTPUT_NOT_OBJECT"
    UNSUPPORTED_TOP_LEVEL_FIELDS = "UNSUPPORTED_TOP_LEVEL_FIELDS"
    SCHEMA_VERSION_MISMATCH = "SCHEMA_VERSION_MISMATCH"
    DOCUMENT_ID_MISMATCH = "DOCUMENT_ID_MISMATCH"
    SOURCE_ID_MISMATCH = "SOURCE_ID_MISMATCH"
    UNITS_NOT_LIST = "UNITS_NOT_LIST"
    UNIT_NOT_OBJECT = "UNIT_NOT_OBJECT"
    UNIT_ID_REQUIRED = "UNIT_ID_REQUIRED"
    UNIT_ID_UNKNOWN = "UNIT_ID_UNKNOWN"
    UNIT_ID_DUPLICATE = "UNIT_ID_DUPLICATE"
    UNIT_FIELDS_MISSING = "UNIT_FIELDS_MISSING"
    COVERAGE_REQUIRED = "COVERAGE_REQUIRED"
    ABSTENTION_COVERAGE_MISMATCH = "ABSTENTION_COVERAGE_MISMATCH"
    FACETS_NOT_LIST = "FACETS_NOT_LIST"
    EMPTY_WITHOUT_ABSTENTION = "EMPTY_WITHOUT_ABSTENTION"
    ABSTENTION_REASON_REQUIRED = "ABSTENTION_REASON_REQUIRED"
    FACET_NOT_OBJECT = "FACET_NOT_OBJECT"
    FACET_FIELDS_MISSING = "FACET_FIELDS_MISSING"
    FACET_ID_REQUIRED = "FACET_ID_REQUIRED"
    FACET_ID_DUPLICATE = "FACET_ID_DUPLICATE"
    UNKNOWN_FACET_KIND = "UNKNOWN_FACET_KIND"
    UNKNOWN_HORIZON_KIND = "UNKNOWN_HORIZON_KIND"
    UNKNOWN_ACTOR_SCOPE = "UNKNOWN_ACTOR_SCOPE"
    SOURCE_QUOTE_REQUIRED = "SOURCE_QUOTE_REQUIRED"
    SOURCE_SPAN_REQUIRED = "SOURCE_SPAN_REQUIRED"
    SOURCE_SPAN_MISMATCH = "SOURCE_SPAN_MISMATCH"
    FACET_OUTSIDE_UNIT = "FACET_OUTSIDE_UNIT"
    ANCHOR_NOT_OBJECT = "ANCHOR_NOT_OBJECT"
    ANCHOR_FIELDS_MISSING = "ANCHOR_FIELDS_MISSING"
    ANCHOR_NOT_IN_SOURCE = "ANCHOR_NOT_IN_SOURCE"
    ANCHOR_AMBIGUOUS = "ANCHOR_AMBIGUOUS"
    ANCHOR_SPAN_MISMATCH = "ANCHOR_SPAN_MISMATCH"
    NEGATION_NOT_BOOLEAN = "NEGATION_NOT_BOOLEAN"
    NEGATION_NOT_SOURCE_GROUNDED = "NEGATION_NOT_SOURCE_GROUNDED"
    HORIZON_REQUIRED = "HORIZON_REQUIRED"
    ACTION_KIND_NOT_STRING = "ACTION_KIND_NOT_STRING"
    UNKNOWN_ACTION_KIND = "UNKNOWN_ACTION_KIND"
    BROAD_GOAL_ACTION_KIND_FORBIDDEN = "BROAD_GOAL_ACTION_KIND_FORBIDDEN"
    ACTION_NOT_SOURCE_EXPLICIT = "ACTION_NOT_SOURCE_EXPLICIT"
    LINK_NOT_LIST = "LINK_NOT_LIST"
    LINK_UNKNOWN = "LINK_UNKNOWN"
    LINK_SELF = "LINK_SELF"
    INVALID_DOCUMENT_OR_UNITS = "INVALID_DOCUMENT_OR_UNITS"


@final
@dataclass(frozen=True, slots=True, kw_only=True, match_args=False)
class SourceDocument:
    """The source-only view exposed to an inference request."""

    document_id: str
    source_id: str | None
    text: str
    language: str | None
    question: str | None
    stated_earliest: str | None
    source_year: int | None
    source_sha256: str
    url: str | None = None
    actor_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.document_id, str) or not self.document_id or len(self.document_id) > 256:
            raise ValueError("document_id must be a bounded non-empty string")
        if self.source_id is not None and (not isinstance(self.source_id, str) or len(self.source_id) > 256):
            raise ValueError("source_id must be a bounded string or null")
        if not isinstance(self.text, str) or not self.text or len(self.text) > 2_000_000:
            raise ValueError("source text must be non-empty and bounded")
        for name, value, limit in (
            ("language", self.language, 64),
            ("question", self.question, 10_000),
            ("stated_earliest", self.stated_earliest, 128),
            ("url", self.url, 8_192),
            ("actor_id", self.actor_id, 256),
        ):
            if value is not None and (not isinstance(value, str) or len(value) > limit):
                raise ValueError(f"{name} must be a bounded string or null")
        if self.source_year is not None and (not isinstance(self.source_year, int) or isinstance(self.source_year, bool)):
            raise TypeError("source_year must be an integer or null")
        if not isinstance(self.source_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", self.source_sha256):
            raise ValueError("source_sha256 must be a lowercase SHA-256 digest")
        if self.source_sha256 != _digest_text(self.text):
            raise ValueError("source_sha256 does not match source text")

    def as_request_dict(self) -> dict[str, object]:
        """Return only fields needed for source reading, never review metadata."""

        result: dict[str, object] = {
            "document_id": self.document_id,
            "source_id": self.source_id,
            "language": self.language,
            "question": self.question,
            "stated_earliest": self.stated_earliest,
            "source_year": self.source_year,
            "source_sha256": self.source_sha256,
            "source_text": self.text,
            "url": self.url,
        }
        if self.actor_id is not None:
            result["actor_id"] = self.actor_id
        return result


@final
@dataclass(frozen=True, slots=True, kw_only=True, match_args=False)
class SourceUnit:
    """A caller-owned exact slice; labels are intentionally absent."""

    unit_id: str
    text: str
    start: int
    end: int

    def __post_init__(self) -> None:
        if not isinstance(self.unit_id, str) or not self.unit_id or len(self.unit_id) > 256:
            raise ValueError("unit_id must be a bounded non-empty string")
        if not isinstance(self.text, str) or not self.text or len(self.text) > 2_000_000:
            raise ValueError("unit text must be non-empty and bounded")
        if (
            not isinstance(self.start, int)
            or isinstance(self.start, bool)
            or not isinstance(self.end, int)
            or isinstance(self.end, bool)
            or self.start < 0
            or self.end <= self.start
            or self.end - self.start != len(self.text)
        ):
            raise ValueError("unit offsets must be bounded integer offsets matching unit text length")

    def as_dict(self) -> dict[str, object]:
        return {"unit_id": self.unit_id, "text": self.text, "start": self.start, "end": self.end}


@final
@dataclass(frozen=True, slots=True, kw_only=True, match_args=False)
class SourceAnchor:
    """An exact source quote with absolute document offsets."""

    quote: str
    start: int
    end: int

    def __post_init__(self) -> None:
        if not isinstance(self.quote, str) or not self.quote or len(self.quote) > 2_000_000:
            raise ValueError("anchor quote must be non-empty and bounded")
        if (
            not isinstance(self.start, int)
            or isinstance(self.start, bool)
            or not isinstance(self.end, int)
            or isinstance(self.end, bool)
            or self.start < 0
            or self.end <= self.start
            or self.end - self.start != len(self.quote)
        ):
            raise ValueError("anchor offsets must be bounded integer offsets matching quote length")

    def as_dict(self) -> dict[str, object]:
        return {"quote": self.quote, "start": self.start, "end": self.end}


@final
@dataclass(frozen=True, slots=True, kw_only=True, match_args=False)
class FacetProposal:
    """One source-grounded facet; all interpretation remains proposed."""

    facet_id: str
    kind: FacetKind
    source: SourceAnchor
    actor: SourceAnchor | None
    carrier: SourceAnchor | None
    predicate: SourceAnchor | None
    target: SourceAnchor | None
    population: SourceAnchor | None
    magnitude: SourceAnchor | None
    instrument: SourceAnchor | None
    condition: SourceAnchor | None
    horizon: SourceAnchor | None
    horizon_kind: HorizonKind
    actor_scope: ActorScope
    negation: bool
    action_kind: ActionKind | None
    linked_subproposition_ids: tuple[str, ...]
    validation_state: PublicationState = PublicationState.PROPOSED

    def __post_init__(self) -> None:
        if not isinstance(self.facet_id, str) or not self.facet_id or len(self.facet_id) > 256:
            raise ValueError("facet_id must be a bounded non-empty string")
        if not isinstance(self.kind, FacetKind):
            raise TypeError("kind must be a FacetKind")
        if not isinstance(self.source, SourceAnchor):
            raise TypeError("source must be a SourceAnchor")
        for value in (self.actor, self.carrier, self.predicate, self.target, self.population, self.magnitude, self.instrument, self.condition, self.horizon):
            if value is not None and not isinstance(value, SourceAnchor):
                raise TypeError("facet anchors must be SourceAnchor values or null")
        if not isinstance(self.horizon_kind, HorizonKind):
            raise TypeError("horizon_kind must be a HorizonKind")
        if not isinstance(self.actor_scope, ActorScope):
            raise TypeError("actor_scope must be an ActorScope")
        if not isinstance(self.negation, bool):
            raise TypeError("negation must be boolean")
        if self.action_kind is not None and not isinstance(self.action_kind, ActionKind):
            raise TypeError("action_kind must be an ActionKind or null")
        links = tuple(self.linked_subproposition_ids)
        if len(links) > 128 or any(not isinstance(link, str) or not link or len(link) > 256 for link in links):
            raise ValueError("linked_subproposition_ids must be bounded non-empty strings")
        object.__setattr__(self, "linked_subproposition_ids", links)
        if not isinstance(self.validation_state, PublicationState):
            raise TypeError("validation_state must be a PublicationState")

    @property
    def source_quote(self) -> str:
        return self.source.quote

    @property
    def source_start(self) -> int:
        return self.source.start

    @property
    def source_end(self) -> int:
        return self.source.end

    def as_dict(self) -> dict[str, object]:
        return {
            "facet_id": self.facet_id,
            "kind": self.kind.value,
            "source_quote": self.source.quote,
            "source_start": self.source.start,
            "source_end": self.source.end,
            "actor": self.actor.as_dict() if self.actor else None,
            "carrier": self.carrier.as_dict() if self.carrier else None,
            "predicate": self.predicate.as_dict() if self.predicate else None,
            "target": self.target.as_dict() if self.target else None,
            "population": self.population.as_dict() if self.population else None,
            "magnitude": self.magnitude.as_dict() if self.magnitude else None,
            "instrument": self.instrument.as_dict() if self.instrument else None,
            "condition": self.condition.as_dict() if self.condition else None,
            "horizon": self.horizon.as_dict() if self.horizon else None,
            "horizon_kind": self.horizon_kind.value,
            "actor_scope": self.actor_scope.value,
            "negation": self.negation,
            "action_kind": self.action_kind.value if self.action_kind else None,
            "linked_subproposition_ids": list(self.linked_subproposition_ids),
            "validation_state": self.validation_state.value,
        }


@final
@dataclass(frozen=True, slots=True, kw_only=True, match_args=False)
class FacetInvalidRecord:
    """A visible boundary rejection; never silently dropped."""

    code: InvalidCode
    message: str
    index: int | None = None
    field: str | None = None
    raw: object = None

    def __post_init__(self) -> None:
        if not isinstance(self.code, InvalidCode):
            raise TypeError("invalid record code must be an InvalidCode")
        if not isinstance(self.message, str) or not self.message:
            raise ValueError("invalid record message must be non-empty")
        object.__setattr__(self, "raw", deepcopy(self.raw))

    def as_dict(self) -> dict[str, object]:
        result: dict[str, object] = {"code": self.code.value, "message": self.message, "index": self.index}
        if self.field is not None:
            result["field"] = self.field
        if self.raw is not None:
            result["raw"] = self.raw
        return result


@final
@dataclass(frozen=True, slots=True, kw_only=True, match_args=False)
class FacetUnitResult:
    unit_id: str
    status: FacetUnitStatus
    facets: tuple[FacetProposal, ...] = ()
    invalid_records: tuple[FacetInvalidRecord, ...] = ()
    abstention_reason: str | None = None
    raw: object = None

    def __post_init__(self) -> None:
        if not isinstance(self.unit_id, str) or not self.unit_id:
            raise ValueError("unit result requires unit_id")
        if not isinstance(self.status, FacetUnitStatus):
            raise TypeError("unit result status must be a FacetUnitStatus")
        facets = tuple(self.facets)
        invalid_records = tuple(self.invalid_records)
        if any(not isinstance(facet, FacetProposal) for facet in facets):
            raise TypeError("unit facets must be FacetProposal values")
        if any(not isinstance(error, FacetInvalidRecord) for error in invalid_records):
            raise TypeError("unit invalid_records must be FacetInvalidRecord values")
        object.__setattr__(self, "facets", facets)
        object.__setattr__(self, "invalid_records", invalid_records)
        object.__setattr__(self, "raw", deepcopy(self.raw))

    def as_dict(self) -> dict[str, object]:
        return {
            "unit_id": self.unit_id,
            "status": self.status.value,
            "facets": [facet.as_dict() for facet in self.facets],
            "invalid_records": [error.as_dict() for error in self.invalid_records],
            "abstention_reason": self.abstention_reason,
            "raw": self.raw,
        }


@final
@dataclass(frozen=True, slots=True, kw_only=True, match_args=False)
class FacetExtractionResult:
    status: FacetResultStatus
    document_id: str | None
    source_id: str | None
    prompt_version: str
    units: tuple[FacetUnitResult, ...]
    invalid_records: tuple[FacetInvalidRecord, ...] = ()
    abstention_reason: str | None = None
    coverage: Mapping[str, object] = field(default_factory=lambda: MappingProxyType({}))
    raw_sha256: str | None = None
    validator_version: str = FACET_VALIDATOR_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.status, FacetResultStatus):
            raise TypeError("extraction status must be a FacetResultStatus")
        units = tuple(self.units)
        errors = tuple(self.invalid_records)
        if any(not isinstance(unit, FacetUnitResult) for unit in units):
            raise TypeError("extraction units must be FacetUnitResult values")
        if any(not isinstance(error, FacetInvalidRecord) for error in errors):
            raise TypeError("extraction invalid_records must be FacetInvalidRecord values")
        object.__setattr__(self, "units", units)
        object.__setattr__(self, "invalid_records", errors)
        if not isinstance(self.coverage, Mapping):
            raise TypeError("coverage must be a mapping")
        object.__setattr__(self, "coverage", _owned_coverage(self.coverage))
        if self.validator_version != FACET_VALIDATOR_VERSION:
            raise ValueError("unsupported facet validator version")

    @property
    def facets(self) -> tuple[FacetProposal, ...]:
        return tuple(facet for unit in self.units for facet in unit.facets)

    @property
    def linked_subpropositions(self) -> tuple[FacetProposal, ...]:
        """Integration-friendly alias for the richer source packet consumer."""

        return self.facets

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "document_id": self.document_id,
            "source_id": self.source_id,
            "prompt_version": self.prompt_version,
            "units": [unit.as_dict() for unit in self.units],
            "facets": [facet.as_dict() for facet in self.facets],
            "invalid_records": [error.as_dict() for error in self.invalid_records],
            "abstention_reason": self.abstention_reason,
            "coverage": dict(self.coverage),
            "raw_sha256": self.raw_sha256,
            "validator_version": self.validator_version,
            "admission_state": "PROPOSED",
        }


@final
@dataclass(frozen=True, slots=True, kw_only=True, match_args=False)
class FacetResearchPacket:
    """A source + proposed facets packet for a maintained consumer."""

    packet_version: str
    source: SourceDocument
    units: tuple[SourceUnit, ...]
    extraction: FacetExtractionResult

    def as_dict(self) -> dict[str, object]:
        return {
            "packet_version": self.packet_version,
            "source": self.source.as_request_dict(),
            "units": [unit.as_dict() for unit in self.units],
            "extraction": self.extraction.as_dict(),
            "admission_state": "PROPOSED",
        }


@final
@dataclass(frozen=True, slots=True, kw_only=True, match_args=False)
class FacetRunOutcome:
    """Receipt-independent run summary used by the experiment driver."""

    item_id: str
    status: str
    request_id: str | None
    receipt_status: str | None
    elapsed_seconds: float | None
    generation_tokens: int | None
    cache_hit: bool
    enable_thinking: bool
    output: FacetExtractionResult | None
    error: str | None = None


_ANCHOR_FIELDS = (
    "actor",
    "carrier",
    "predicate",
    "target",
    "population",
    "magnitude",
    "instrument",
    "condition",
    "horizon",
)
_TOP_LEVEL_KEYS = frozenset(
    {"schema_version", "document_id", "source_id", "abstain", "abstention_reason", "coverage", "units"}
)
_UNIT_KEYS = frozenset({"unit_id", "status", "abstention_reason", "facets"})
_FACET_KEYS = frozenset(
    {
        "facet_id",
        "kind",
        "source_quote",
        "source_start",
        "source_end",
        "source_occurrence",
        "actor",
        "carrier",
        "predicate",
        "target",
        "population",
        "magnitude",
        "instrument",
        "condition",
        "horizon",
        "horizon_kind",
        "actor_scope",
        "negation",
        "action_kind",
        "linked_subproposition_ids",
    }
)
_NEGATION = re.compile(
    r"\b(?:ei|eikä|eivät|eivätkä|emme|emmekä|enkä|ettei|etteivät|ilman|älä|älkää|"
    r"inte|ingen|inget|inga|aldrig|utan|not|never|no)\b|\w+matta\b",
    re.IGNORECASE,
)
_HORIZON = re.compile(
    r"\b(?:tämän|tänä|kuluvan)\s+vuoden|\bvuoden\s+(?:\d{4}\s+)?loppuun|\b(?:ennen|ennen kuin|ensi)\b|"
    r"\b(?:vuonna|vuoteen|vuoden)\s+(?:19|20)\d{2}|\b(?:19|20)\d{2}\b|\b(?:nyt|nykyisin|tällä hetkellä|aiemmin|ennen)\b|"
    r"\b(?:future|currently|presently|past|before|by)\b",
    re.IGNORECASE,
)
_ACTION_TEXT = re.compile(
    r"\b(?:aloit\w*|lakiehdot\w*|äänest\w*|kys\w*|puheenvuoro\w*|"
    r"esit\w*|teen|teemme|tehd\w*|tekem\w*|laad\w*|kirjoit\w*|jät\w*|ero\w*|lahjoit\w*|kampanjo\w*|"
    r"advocat\w*|vote\w*|file\w*|speak\w*)\b",
    re.IGNORECASE,
)


def _digest_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _error(
    errors: list[FacetInvalidRecord],
    code: InvalidCode,
    message: str,
    index: int | None = None,
    field_name: str | None = None,
    raw: object = None,
) -> None:
    errors.append(
        FacetInvalidRecord(
            code=code,
            message=message,
            index=index,
            field=field_name,
            raw=deepcopy(raw),
        )
    )


def _owned_coverage(value: Mapping[str, object]) -> Mapping[str, object]:
    """Freeze a small scalar coverage summary before it crosses the boundary."""

    return MappingProxyType(dict(value))


def source_document(value: Mapping[str, object]) -> SourceDocument:
    """Convert a source-only mapping into an owned document record.

    This function intentionally ignores all unknown input keys.  In
    particular, held-out rows may contain ``gold``, ``selection`` and
    ``adjudication`` metadata; none can cross into an inference request.
    """

    document_id = value.get("document_id")
    text = value.get("source_text", value.get("text"))
    source_id = value.get("source_id")
    language = value.get("language")
    question = value.get("question", value.get("original_question"))
    stated_earliest = value.get("stated_earliest")
    source_year = value.get("source_year")
    url = value.get("url")
    actor_id = value.get("actor_id")
    identity = value.get("identity_context")
    if actor_id is None and isinstance(identity, Mapping):
        actor_id = identity.get("actor_id")
    if stated_earliest is None and isinstance(identity, Mapping):
        stated_earliest = identity.get("stated_earliest")
    if not isinstance(document_id, str) or not document_id.strip():
        raise ValueError("source document requires document_id")
    if not isinstance(text, str) or not text:
        raise ValueError("source document requires non-empty source_text")
    if source_id is not None and not isinstance(source_id, str):
        raise ValueError("source_id must be a string or null")
    if language is not None and not isinstance(language, str):
        raise ValueError("language must be a string or null")
    if question is not None and not isinstance(question, str):
        raise ValueError("question must be a string or null")
    if stated_earliest is not None and not isinstance(stated_earliest, str):
        raise ValueError("stated_earliest must be a string or null")
    if source_year is not None and (not isinstance(source_year, int) or isinstance(source_year, bool)):
        raise ValueError("source_year must be an integer or null")
    supplied_hash = value.get("source_sha256", value.get("source_text_sha256"))
    actual_hash = _digest_text(text)
    if supplied_hash is not None and supplied_hash != actual_hash:
        raise ValueError("source_sha256 does not match source_text")
    if not isinstance(url, str) and url is not None:
        raise ValueError("url must be a string or null")
    if not isinstance(actor_id, str) and actor_id is not None:
        raise ValueError("actor_id must be a string or null")
    return SourceDocument(
        document_id=document_id,
        source_id=source_id,
        text=text,
        language=language,
        question=question,
        stated_earliest=stated_earliest,
        source_year=source_year,
        source_sha256=actual_hash,
        url=url,
        actor_id=actor_id,
    )


def source_units(document: SourceDocument, values: Sequence[Mapping[str, object]]) -> tuple[SourceUnit, ...]:
    """Own and validate exact canonical unit slices before model prompting."""

    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence) or not values:
        raise ValueError("source units must be a non-empty sequence")
    seen: set[str] = set()
    result: list[SourceUnit] = []
    for index, value in enumerate(values):
        if not isinstance(value, Mapping):
            raise TypeError(f"source unit {index} must be an object")
        unit_id = value.get("unit_id")
        text = value.get("text", value.get("source_text"))
        start = value.get("start", value.get("source_start"))
        end = value.get("end", value.get("source_end"))
        if not isinstance(unit_id, str) or not unit_id.strip():
            raise ValueError(f"source unit {index} requires unit_id")
        if unit_id in seen:
            raise ValueError(f"duplicate source unit_id: {unit_id}")
        if not isinstance(text, str) or not text:
            raise ValueError(f"source unit {unit_id} requires text")
        if (
            not isinstance(start, int)
            or isinstance(start, bool)
            or not isinstance(end, int)
            or isinstance(end, bool)
            or start < 0
            or end <= start
            or end > len(document.text)
            or document.text[start:end] != text
        ):
            raise ValueError(f"source unit {unit_id} is not an exact source slice")
        seen.add(unit_id)
        result.append(SourceUnit(unit_id=unit_id, text=text, start=start, end=end))
    return tuple(result)


def _prompt_text(prompt_version: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", prompt_version):
        raise ValueError("invalid prompt version")
    path = PROMPT_DIR / f"{prompt_version}.txt"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.read_text(encoding="utf-8")


def _schema_anchor(*, compact: bool = False) -> dict[str, object]:
    if compact:
        return {"type": ["string", "null"]}
    return {
        "anyOf": [
            {"type": "null"},
            {
                "type": "object",
                "additionalProperties": False,
                "required": ["quote", "start", "end"],
                "properties": {
                    "quote": {"type": "string", "minLength": 1},
                    "start": {"type": "integer", "minimum": 0},
                    "end": {"type": "integer", "minimum": 1},
                },
            },
        ]
    }


def facet_response_schema(*, compact: bool = False) -> dict[str, object]:
    """Return a copy of the constrained-decoding transport schema."""

    facet = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "facet_id",
            "kind",
            "source_quote",
            *([] if compact else ["source_start", "source_end"]),
            *( ["source_occurrence"] if compact else []),
            *_ANCHOR_FIELDS,
            "horizon_kind",
            "actor_scope",
            "negation",
            "action_kind",
            "linked_subproposition_ids",
        ],
        "properties": {
            "facet_id": {"type": "string", "minLength": 1},
            "kind": {"enum": [item.value for item in FacetKind]},
            "source_quote": {"type": "string", "minLength": 1},
            **({} if compact else {
                "source_start": {"type": "integer", "minimum": 0},
                "source_end": {"type": "integer", "minimum": 1},
            }),
            **({"source_occurrence": {"type": "integer", "minimum": 0}} if compact else {}),
            **{name: _schema_anchor(compact=compact) for name in _ANCHOR_FIELDS},
            "horizon_kind": {"enum": [item.value for item in HorizonKind]},
            "actor_scope": {"enum": [item.value for item in ActorScope]},
            "negation": {"type": "boolean"},
            "action_kind": {"type": ["string", "null"]},
            "linked_subproposition_ids": {"type": "array", "items": {"type": "string"}},
        },
    }
    unit = {
        "type": "object",
        "additionalProperties": False,
        "required": ["unit_id", "status", "abstention_reason", "facets"],
        "properties": {
            "unit_id": {"type": "string", "minLength": 1},
            "status": {"enum": [FacetUnitStatus.COMPLETE.value, FacetUnitStatus.ABSTAIN.value]},
            "abstention_reason": {"type": ["string", "null"]},
            "facets": {"type": "array", "items": facet},
        },
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["schema_version", "document_id", "source_id", "abstain", "abstention_reason", "coverage", "units"],
        "properties": {
            "schema_version": {"const": FACET_SCHEMA_VERSION},
            "document_id": {"type": "string"},
            "source_id": {"type": ["string", "null"]},
            "abstain": {"type": "boolean"},
            "abstention_reason": {"type": ["string", "null"]},
            "coverage": {
                "type": "object",
                "additionalProperties": False,
                "required": ["status"],
                "properties": {"status": {"enum": ["COMPLETE", "PARTIAL", "ABSTAIN"]}},
            },
            "units": {"type": "array", "items": unit},
        },
    }


def build_facet_request(
    document: Mapping[str, object] | SourceDocument,
    units: Sequence[Mapping[str, object]] | Sequence[SourceUnit],
    *,
    prompt_version: str = FACET_PROMPT_VERSION,
) -> dict[str, object]:
    """Build a source-only request; this function never calls the model."""

    owned_document = document if isinstance(document, SourceDocument) else source_document(document)
    unit_values: list[Mapping[str, object]] = []
    for unit in units:
        unit_values.append(unit.as_dict() if isinstance(unit, SourceUnit) else unit)
    owned_units = source_units(owned_document, unit_values)
    payload = {
        "task": "extract_linked_source_facets",
        "contract": FACET_SCHEMA_VERSION,
        "source_document": owned_document.as_request_dict(),
        "units": [unit.as_dict() for unit in owned_units],
    }
    system = _prompt_text(prompt_version)
    return {
        "prompt_version": prompt_version,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)},
        ],
        "temperature": 0,
        "max_tokens": max(1400, min(12000, 700 + 240 * len(owned_units) + len(owned_document.text) // 2)),
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "paa_source_facets",
                "strict": True,
                "schema": facet_response_schema(compact=prompt_version == "facets_v2_linked_propositions"),
            },
        },
        "metadata": {
            "document_id": owned_document.document_id,
            "source_id": owned_document.source_id,
            "source_sha256": owned_document.source_sha256,
            "unit_ids": [unit.unit_id for unit in owned_units],
            "unit_count": len(owned_units),
            "prompt_version": prompt_version,
            "admission_state": "PROPOSED",
        },
    }


def _parse_raw(raw: Mapping[str, object] | str, errors: list[FacetInvalidRecord]) -> dict[str, object] | None:
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            _error(errors, InvalidCode.INVALID_JSON, f"model output is not JSON: {exc.msg}")
            return None
    elif isinstance(raw, Mapping):
        parsed = dict(raw)
    else:
        _error(errors, InvalidCode.OUTPUT_NOT_OBJECT, "model output must be a JSON object")
        return None
    if not isinstance(parsed, dict):
        _error(errors, InvalidCode.OUTPUT_NOT_OBJECT, "model output must decode to an object")
        return None
    unknown = sorted(set(parsed) - _TOP_LEVEL_KEYS)
    if unknown:
        _error(errors, InvalidCode.UNSUPPORTED_TOP_LEVEL_FIELDS, f"unsupported top-level fields: {unknown}")
    return parsed


def _anchor(
    raw: object,
    *,
    name: str,
    source: str,
    source_start: int,
    source_end: int,
    document_length: int,
    allow_outside_unit: bool,
    errors: list[FacetInvalidRecord],
    index: int,
) -> SourceAnchor | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        if not raw:
            _error(errors, InvalidCode.ANCHOR_FIELDS_MISSING, f"{name} cannot be an empty quote", index, name, raw)
            return None
        candidates: list[int] = []
        cursor = source_start
        while True:
            found = source.find(raw, cursor, source_end)
            if found < 0:
                break
            candidates.append(found)
            cursor = found + 1
        if len(candidates) != 1:
            code = InvalidCode.ANCHOR_NOT_IN_SOURCE if not candidates else InvalidCode.ANCHOR_AMBIGUOUS
            _error(errors, code, f"{name} quote must have one deterministic occurrence in the facet span", index, name, raw)
            return None
        start = candidates[0]
        return SourceAnchor(quote=raw, start=start, end=start + len(raw))
    if not isinstance(raw, Mapping):
        _error(errors, InvalidCode.ANCHOR_NOT_OBJECT, f"{name} must be null or an anchor object", index, name, raw)
        return None
    quote = raw.get("quote")
    start = raw.get("start")
    end = raw.get("end")
    if (
        not isinstance(quote, str)
        or not quote
        or not isinstance(start, int)
        or isinstance(start, bool)
        or not isinstance(end, int)
        or isinstance(end, bool)
    ):
        _error(errors, InvalidCode.ANCHOR_FIELDS_MISSING, f"{name} needs quote/start/end", index, name, raw)
        return None
    if start < 0 or end <= start or end > document_length:
        _error(errors, InvalidCode.ANCHOR_SPAN_MISMATCH, f"{name} span is outside source document", index, name, raw)
        return None
    if source[start:end] != quote:
        _error(errors, InvalidCode.ANCHOR_SPAN_MISMATCH, f"{name} quote does not match source offsets", index, name, raw)
        return None
    if not allow_outside_unit and (start < source_start or end > source_end):
        _error(errors, InvalidCode.ANCHOR_NOT_IN_SOURCE, f"{name} must be inside the facet source span", index, name, raw)
        return None
    return SourceAnchor(quote=quote, start=start, end=end)


def _source_anchor(
    row: Mapping[str, object],
    document: SourceDocument,
    unit: SourceUnit,
    errors: list[FacetInvalidRecord],
    index: int,
) -> SourceAnchor | None:
    quote = row.get("source_quote")
    start = row.get("source_start")
    end = row.get("source_end")
    occurrence = row.get("source_occurrence")
    if not isinstance(quote, str) or not quote:
        _error(errors, InvalidCode.SOURCE_QUOTE_REQUIRED, "source_quote must be non-empty", index, "source_quote", row)
        return None
    if start is None and end is None:
        if not isinstance(occurrence, int) or isinstance(occurrence, bool) or occurrence < 0:
            _error(errors, InvalidCode.SOURCE_SPAN_REQUIRED, "compact facet source needs a non-negative source_occurrence", index, "source_occurrence", row)
            return None
        matches: list[int] = []
        cursor = unit.start
        while True:
            found = document.text.find(quote, cursor, unit.end)
            if found < 0:
                break
            matches.append(found)
            cursor = found + 1
        if occurrence >= len(matches):
            _error(errors, InvalidCode.SOURCE_SPAN_MISMATCH, "source_occurrence does not identify a source quote", index, "source_occurrence", row)
            return None
        start = matches[occurrence]
        end = start + len(quote)
    if (
        not isinstance(start, int)
        or isinstance(start, bool)
        or not isinstance(end, int)
        or isinstance(end, bool)
    ):
        _error(errors, InvalidCode.SOURCE_SPAN_REQUIRED, "source_start/source_end must be integer offsets", index, "source_span", row)
        return None
    if start < unit.start or end > unit.end or end <= start or document.text[start:end] != quote:
        code = InvalidCode.FACET_OUTSIDE_UNIT if start < unit.start or end > unit.end else InvalidCode.SOURCE_SPAN_MISMATCH
        _error(errors, code, "facet source quote must exactly match a subspan of its caller unit", index, "source_span", row)
        return None
    return SourceAnchor(quote=quote, start=start, end=end)


def _source_has_explicit_negation(text: str) -> bool:
    return _NEGATION.search(text) is not None


def _source_has_horizon(text: str) -> bool:
    return _HORIZON.search(text) is not None


def _unit_condition_anchor(
    raw: object,
    source: SourceAnchor,
    document: SourceDocument,
    unit: SourceUnit,
    units: Sequence[SourceUnit],
    errors: list[FacetInvalidRecord],
    index: int,
) -> SourceAnchor | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        inside_matches: list[int] = []
        cursor = source.start
        while True:
            found = document.text.find(raw, cursor, source.end)
            if found < 0:
                break
            inside_matches.append(found)
            cursor = found + 1
        if len(inside_matches) == 1:
            return SourceAnchor(quote=raw, start=inside_matches[0], end=inside_matches[0] + len(raw))
        if len(inside_matches) > 1:
            _error(errors, InvalidCode.ANCHOR_AMBIGUOUS, "condition quote has multiple occurrences in the facet span", index, "condition", raw)
            return None
        previous = [candidate for candidate in units if candidate.end <= unit.start]
        if previous:
            previous_unit = max(previous, key=lambda candidate: candidate.end)
            matches: list[int] = []
            cursor = previous_unit.start
            while True:
                found = document.text.find(raw, cursor, previous_unit.end)
                if found < 0:
                    break
                matches.append(found)
                cursor = found + 1
            if len(matches) == 1 and re.fullmatch(r"[\s,;:()\[\]{}—–\-]*", document.text[previous_unit.end : unit.start]):
                return SourceAnchor(quote=raw, start=matches[0], end=matches[0] + len(raw))
        return None
    if not isinstance(raw, Mapping):
        _error(errors, InvalidCode.ANCHOR_NOT_OBJECT, "condition must be null or an anchor object", index, "condition", raw)
        return None
    quote = raw.get("quote")
    start = raw.get("start")
    end = raw.get("end")
    if not isinstance(quote, str) or not quote or not isinstance(start, int) or isinstance(start, bool) or not isinstance(end, int) or isinstance(end, bool):
        _error(errors, InvalidCode.ANCHOR_FIELDS_MISSING, "condition needs quote/start/end", index, "condition", raw)
        return None
    if start < 0 or end <= start or end > len(document.text) or document.text[start:end] != quote:
        _error(errors, InvalidCode.ANCHOR_SPAN_MISMATCH, "condition quote does not match source offsets", index, "condition", raw)
        return None
    if source.start <= start and end <= source.end:
        return SourceAnchor(quote=quote, start=start, end=end)
    # A condition immediately before a canonical unit is allowed, but its
    # exact span remains retained. This is the only inherited context escape.
    previous = [candidate for candidate in units if candidate.end <= unit.start]
    if previous:
        previous_unit = max(previous, key=lambda candidate: candidate.end)
        if (
            previous_unit.start <= start < end <= previous_unit.end
            and re.fullmatch(r"[\s,;:()\[\]{}—–\-]*", document.text[previous_unit.end : unit.start])
        ):
            return SourceAnchor(quote=quote, start=start, end=end)
    _error(errors, InvalidCode.ANCHOR_NOT_IN_SOURCE, "condition must be inside the facet or adjacent preceding unit", index, "condition", raw)
    return None


def _facet_from_raw(
    raw: object,
    *,
    document: SourceDocument,
    unit: SourceUnit,
    units: Sequence[SourceUnit],
    facet_index: int,
    errors: list[FacetInvalidRecord],
) -> FacetProposal | None:
    if not isinstance(raw, Mapping):
        _error(errors, InvalidCode.FACET_NOT_OBJECT, "facet must be an object", facet_index, raw=raw)
        return None
    unknown = sorted(set(raw) - _FACET_KEYS)
    if unknown:
        _error(errors, InvalidCode.FACET_FIELDS_MISSING, f"unsupported facet fields: {unknown}", facet_index, raw=raw)
    facet_id = raw.get("facet_id")
    if not isinstance(facet_id, str) or not facet_id.strip():
        _error(errors, InvalidCode.FACET_ID_REQUIRED, "facet_id must be non-empty", facet_index, "facet_id", raw)
        return None
    kind_value = raw.get("kind")
    try:
        kind = FacetKind(kind_value)
    except (TypeError, ValueError):
        _error(errors, InvalidCode.UNKNOWN_FACET_KIND, f"unsupported facet kind: {kind_value!r}", facet_index, "kind", raw)
        return None
    source = _source_anchor(raw, document, unit, errors, facet_index)
    if source is None:
        return None
    horizon_value = raw.get("horizon_kind")
    try:
        horizon_kind = HorizonKind(horizon_value)
    except (TypeError, ValueError):
        _error(errors, InvalidCode.UNKNOWN_HORIZON_KIND, f"unsupported horizon kind: {horizon_value!r}", facet_index, "horizon_kind", raw)
        return None
    actor_scope_value = raw.get("actor_scope")
    try:
        actor_scope = ActorScope(actor_scope_value)
    except (TypeError, ValueError):
        _error(errors, InvalidCode.UNKNOWN_ACTOR_SCOPE, f"unsupported actor scope: {actor_scope_value!r}", facet_index, "actor_scope", raw)
        return None
    negation = raw.get("negation")
    if not isinstance(negation, bool):
        _error(errors, InvalidCode.NEGATION_NOT_BOOLEAN, "negation must be boolean", facet_index, "negation", raw)
        return None
    if negation != _source_has_explicit_negation(source.quote):
        _error(errors, InvalidCode.NEGATION_NOT_SOURCE_GROUNDED, "negation must agree with explicit source wording", facet_index, "negation", raw)
    if not negation and _source_has_explicit_negation(source.quote):
        _error(errors, InvalidCode.NEGATION_NOT_SOURCE_GROUNDED, "negative source wording cannot be marked positive", facet_index, "negation", raw)
    horizon_raw = raw.get("horizon")
    compact = "source_occurrence" in raw and raw.get("source_start") is None
    anchor_start = unit.start if compact else source.start
    anchor_end = unit.end if compact else source.end
    horizon = _anchor(
        horizon_raw,
        name="horizon",
        source=document.text,
        source_start=anchor_start,
        source_end=anchor_end,
        document_length=len(document.text),
        allow_outside_unit=False,
        errors=errors,
        index=facet_index,
    )
    if _source_has_horizon(source.quote) and horizon is None:
        _error(errors, InvalidCode.HORIZON_REQUIRED, "explicit temporal wording needs a horizon anchor", facet_index, "horizon", raw)
    # A temporal reading (for example a future verb) is distinct from an
    # explicit calendar/deadline anchor. ``horizon_kind`` may therefore be
    # FUTURE/CURRENT/PAST without a horizon quote; an explicit time phrase,
    # when present, still requires its exact source anchor.
    if horizon_kind is HorizonKind.FUTURE and kind in {FacetKind.PAST_FACT, FacetKind.CURRENT_FACT}:
        _error(errors, InvalidCode.HORIZON_REQUIRED, "past/current factual acts cannot be labelled future", facet_index, "horizon_kind", raw)
    action_kind = raw.get("action_kind")
    if action_kind is not None and not isinstance(action_kind, str):
        _error(errors, InvalidCode.ACTION_KIND_NOT_STRING, "action_kind must be a string or null", facet_index, "action_kind", raw)
        action_kind = None
    if isinstance(action_kind, str):
        try:
            action_kind = ActionKind(action_kind)
        except (TypeError, ValueError):
            _error(errors, InvalidCode.UNKNOWN_ACTION_KIND, f"unsupported action kind: {action_kind!r}", facet_index, "action_kind", raw)
            action_kind = None
    if kind is FacetKind.POLICY_GOAL and action_kind is not None:
        _error(errors, InvalidCode.BROAD_GOAL_ACTION_KIND_FORBIDDEN, "a broad policy goal cannot launch an action kind", facet_index, "action_kind", raw)
    if action_kind is not None and _ACTION_TEXT.search(source.quote) is None:
        _error(errors, InvalidCode.ACTION_NOT_SOURCE_EXPLICIT, "action kind needs an explicit source action phrase", facet_index, "action_kind", raw)
    anchors: dict[str, SourceAnchor | None] = {}
    for name in _ANCHOR_FIELDS:
        if name == "horizon":
            anchors[name] = horizon
        elif name == "condition":
            anchors[name] = _unit_condition_anchor(raw.get(name), source, document, unit, units, errors, facet_index)
        else:
            anchors[name] = _anchor(
                raw.get(name),
                name=name,
                source=document.text,
                source_start=anchor_start,
                source_end=anchor_end,
                document_length=len(document.text),
                allow_outside_unit=False,
                errors=errors,
                index=facet_index,
            )
    links = raw.get("linked_subproposition_ids")
    if not isinstance(links, list) or any(not isinstance(item, str) or not item for item in links):
        _error(errors, InvalidCode.LINK_NOT_LIST, "linked_subproposition_ids must be a list of non-empty strings", facet_index, "linked_subproposition_ids", raw)
        links_tuple: tuple[str, ...] = ()
    else:
        links_tuple = tuple(links)
    return FacetProposal(
        facet_id=facet_id,
        kind=kind,
        source=source,
        actor=anchors["actor"],
        carrier=anchors["carrier"],
        predicate=anchors["predicate"],
        target=anchors["target"],
        population=anchors["population"],
        magnitude=anchors["magnitude"],
        instrument=anchors["instrument"],
        condition=anchors["condition"],
        horizon=anchors["horizon"],
        horizon_kind=horizon_kind,
        actor_scope=actor_scope,
        negation=negation,
        action_kind=action_kind,
        linked_subproposition_ids=links_tuple,
    )


def _status_for_units(units: Sequence[FacetUnitResult], errors: Sequence[FacetInvalidRecord], abstain: bool) -> FacetResultStatus:
    if abstain and not errors:
        return FacetResultStatus.ABSTAIN
    if abstain:
        return FacetResultStatus.INVALID
    if units and all(unit.status is FacetUnitStatus.ABSTAIN for unit in units) and not errors:
        return FacetResultStatus.ABSTAIN
    if not units:
        return FacetResultStatus.INVALID
    if all(unit.status is FacetUnitStatus.COMPLETE for unit in units) and not errors:
        return FacetResultStatus.VALID
    if any(unit.status is FacetUnitStatus.COMPLETE for unit in units):
        return FacetResultStatus.PARTIAL
    return FacetResultStatus.INVALID


def normalize_facet_output(
    raw: Mapping[str, object] | str,
    document: Mapping[str, object] | SourceDocument,
    units: Sequence[Mapping[str, object]] | Sequence[SourceUnit],
    *,
    prompt_version: str = FACET_PROMPT_VERSION,
) -> FacetExtractionResult:
    """Validate model output and account for every requested source unit."""

    try:
        owned_document = document if isinstance(document, SourceDocument) else source_document(document)
        unit_values: list[Mapping[str, object]] = [item.as_dict() if isinstance(item, SourceUnit) else item for item in units]
        owned_units = source_units(owned_document, unit_values)
    except (TypeError, ValueError) as exc:
        error = FacetInvalidRecord(code=InvalidCode.INVALID_DOCUMENT_OR_UNITS, message=str(exc))
        return FacetExtractionResult(
            status=FacetResultStatus.INVALID,
            document_id=None,
            source_id=None,
            prompt_version=prompt_version,
            units=(),
            invalid_records=(error,),
            coverage=_owned_coverage({"computed_status": FacetResultStatus.INVALID.value, "unit_count": 0, "invalid_count": 1}),
        )
    raw_bytes = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False, sort_keys=True)
    raw_hash = _digest_text(raw_bytes)
    errors: list[FacetInvalidRecord] = []
    parsed = _parse_raw(raw, errors)
    if parsed is None:
        return FacetExtractionResult(
            status=FacetResultStatus.INVALID,
            document_id=owned_document.document_id,
            source_id=owned_document.source_id,
            prompt_version=prompt_version,
            units=tuple(FacetUnitResult(unit_id=unit.unit_id, status=FacetUnitStatus.INVALID) for unit in owned_units),
            invalid_records=tuple(errors),
            coverage=_owned_coverage({"computed_status": FacetResultStatus.INVALID.value, "unit_count": len(owned_units), "covered_unit_count": 0, "facet_count": 0, "invalid_count": len(errors)}),
            raw_sha256=raw_hash,
        )
    if parsed.get("schema_version") != FACET_SCHEMA_VERSION:
        _error(errors, InvalidCode.SCHEMA_VERSION_MISMATCH, f"expected {FACET_SCHEMA_VERSION!r}", field_name="schema_version")
    if parsed.get("document_id") != owned_document.document_id:
        _error(errors, InvalidCode.DOCUMENT_ID_MISMATCH, "model output is for another document", field_name="document_id")
    if parsed.get("source_id") != owned_document.source_id:
        _error(errors, InvalidCode.SOURCE_ID_MISMATCH, "model output is for another source", field_name="source_id")
    abstain = parsed.get("abstain")
    if not isinstance(abstain, bool):
        _error(errors, InvalidCode.EMPTY_WITHOUT_ABSTENTION, "abstain must be boolean", field_name="abstain")
        abstain = False
    reason = parsed.get("abstention_reason")
    if abstain and (not isinstance(reason, str) or not reason.strip()):
        _error(errors, InvalidCode.ABSTENTION_REASON_REQUIRED, "top-level abstention needs a reason", field_name="abstention_reason")
        reason = None
    declared_coverage = parsed.get("coverage")
    coverage_status = declared_coverage.get("status") if isinstance(declared_coverage, Mapping) else None
    if coverage_status not in {"COMPLETE", "PARTIAL", "ABSTAIN"}:
        _error(errors, InvalidCode.COVERAGE_REQUIRED, "coverage.status must be COMPLETE, PARTIAL, or ABSTAIN", field_name="coverage")
    elif abstain and coverage_status != "ABSTAIN":
        _error(errors, InvalidCode.ABSTENTION_COVERAGE_MISMATCH, "top-level abstention requires ABSTAIN coverage", field_name="coverage")
    elif not abstain and coverage_status == "ABSTAIN":
        _error(errors, InvalidCode.ABSTENTION_COVERAGE_MISMATCH, "non-abstaining output cannot use ABSTAIN coverage", field_name="coverage")
    raw_units = parsed.get("units")
    if not isinstance(raw_units, list):
        _error(errors, InvalidCode.UNITS_NOT_LIST, "units must be a list", field_name="units")
        raw_units = []
    by_id = {unit.unit_id: unit for unit in owned_units}
    seen_unit_ids: set[str] = set()
    staged: dict[str, tuple[FacetUnitResult, int]] = {}
    all_facet_ids: set[str] = set()
    for row_index, raw_unit in enumerate(raw_units):
        if not isinstance(raw_unit, Mapping):
            _error(errors, InvalidCode.UNIT_NOT_OBJECT, "unit result must be an object", row_index, raw=raw_unit)
            continue
        unknown = sorted(set(raw_unit) - _UNIT_KEYS)
        if unknown:
            _error(errors, InvalidCode.UNIT_FIELDS_MISSING, f"unsupported unit fields: {unknown}", row_index, raw=raw_unit)
        unit_id = raw_unit.get("unit_id")
        if not isinstance(unit_id, str) or not unit_id:
            _error(errors, InvalidCode.UNIT_ID_REQUIRED, "unit_id is required", row_index, "unit_id", raw_unit)
            continue
        if unit_id not in by_id:
            _error(errors, InvalidCode.UNIT_ID_UNKNOWN, f"unknown unit_id: {unit_id}", row_index, "unit_id", raw_unit)
            continue
        if unit_id in seen_unit_ids:
            _error(errors, InvalidCode.UNIT_ID_DUPLICATE, f"duplicate unit_id: {unit_id}", row_index, "unit_id", raw_unit)
            continue
        seen_unit_ids.add(unit_id)
        unit = by_id[unit_id]
        row_status = raw_unit.get("status")
        if row_status == FacetUnitStatus.ABSTAIN.value:
            row_reason = raw_unit.get("abstention_reason")
            if not isinstance(row_reason, str) or not row_reason.strip():
                _error(errors, InvalidCode.ABSTENTION_REASON_REQUIRED, "unit abstention needs a reason", row_index, "abstention_reason", raw_unit)
                row_reason = None
            staged[unit_id] = (
                FacetUnitResult(
                    unit_id=unit_id,
                    status=FacetUnitStatus.ABSTAIN,
                    abstention_reason=row_reason,
                    raw=deepcopy(raw_unit),
                ),
                row_index,
            )
            continue
        if row_status != FacetUnitStatus.COMPLETE.value:
            _error(errors, InvalidCode.EMPTY_WITHOUT_ABSTENTION, "unit status must be COMPLETE or ABSTAIN", row_index, "status", raw_unit)
        raw_facets = raw_unit.get("facets")
        if not isinstance(raw_facets, list):
            _error(errors, InvalidCode.FACETS_NOT_LIST, "facets must be a list", row_index, "facets", raw_unit)
            raw_facets = []
        if not raw_facets:
            _error(errors, InvalidCode.EMPTY_WITHOUT_ABSTENTION, "empty facets require explicit unit abstention", row_index, "facets", raw_unit)
        local_errors_before = len(errors)
        facets: list[FacetProposal] = []
        local_ids: set[str] = set()
        for facet_index, raw_facet in enumerate(raw_facets):
            facet = _facet_from_raw(raw_facet, document=owned_document, unit=unit, units=owned_units, facet_index=facet_index, errors=errors)
            if facet is None:
                continue
            if facet.facet_id in local_ids or facet.facet_id in all_facet_ids:
                _error(errors, InvalidCode.FACET_ID_DUPLICATE, f"duplicate facet_id: {facet.facet_id}", facet_index, "facet_id", raw_facet)
                continue
            local_ids.add(facet.facet_id)
            all_facet_ids.add(facet.facet_id)
            facets.append(facet)
        status = FacetUnitStatus.COMPLETE if len(errors) == local_errors_before else FacetUnitStatus.INVALID
        staged[unit_id] = (
            FacetUnitResult(
                unit_id=unit_id,
                status=status,
                facets=tuple(facets),
                invalid_records=tuple(errors[local_errors_before:]),
                raw=deepcopy(raw_unit),
            ),
            row_index,
        )
    if abstain:
        unit_results = tuple(
            FacetUnitResult(unit_id=unit.unit_id, status=FacetUnitStatus.ABSTAIN, abstention_reason=str(reason) if reason else None)
            for unit in owned_units
        )
    else:
        unit_results_list: list[FacetUnitResult] = []
        for unit in owned_units:
            if unit.unit_id in staged:
                unit_results_list.append(staged[unit.unit_id][0])
            else:
                _error(errors, InvalidCode.UNIT_ID_UNKNOWN, f"missing result for unit_id: {unit.unit_id}", field_name="units")
                unit_results_list.append(FacetUnitResult(unit_id=unit.unit_id, status=FacetUnitStatus.INVALID))
        unit_results = tuple(unit_results_list)
    # Validate links after all facet IDs are known. A link to itself is not a
    # meaningful linked proposition and unknown IDs indicate incomplete output.
    repaired: list[FacetUnitResult] = []
    for unit_result in unit_results:
        unit_facets: list[FacetProposal] = []
        for facet in unit_result.facets:
            for link in facet.linked_subproposition_ids:
                if link == facet.facet_id:
                    _error(errors, InvalidCode.LINK_SELF, f"facet {facet.facet_id} links to itself", field_name="linked_subproposition_ids")
                elif link not in all_facet_ids:
                    _error(errors, InvalidCode.LINK_UNKNOWN, f"facet link points to unknown id: {link}", field_name="linked_subproposition_ids")
            unit_facets.append(facet)
        repaired.append(FacetUnitResult(unit_id=unit_result.unit_id, status=unit_result.status, facets=tuple(unit_facets), invalid_records=unit_result.invalid_records, abstention_reason=unit_result.abstention_reason, raw=unit_result.raw))
    unit_results = tuple(repaired)
    status = _status_for_units(unit_results, errors, abstain)
    covered = sum(unit.status is FacetUnitStatus.COMPLETE for unit in unit_results)
    facet_count = sum(len(unit.facets) for unit in unit_results)
    coverage = {
        "computed_status": status.value,
        "unit_count": len(owned_units),
        "covered_unit_count": covered,
        "abstained_unit_count": sum(unit.status is FacetUnitStatus.ABSTAIN for unit in unit_results),
        "facet_count": facet_count,
        "linked_facet_count": sum(bool(facet.linked_subproposition_ids) for facet in (item for unit in unit_results for item in unit.facets)),
        "invalid_count": len(errors),
    }
    return FacetExtractionResult(
        status=status,
        document_id=owned_document.document_id,
        source_id=owned_document.source_id,
        prompt_version=prompt_version,
        units=unit_results,
        invalid_records=tuple(errors),
        abstention_reason=str(reason) if reason else None,
        coverage=_owned_coverage(coverage),
        raw_sha256=raw_hash,
    )


def research_packet(
    document: Mapping[str, object] | SourceDocument,
    units: Sequence[Mapping[str, object]] | Sequence[SourceUnit],
    extraction: FacetExtractionResult,
) -> FacetResearchPacket:
    """Bind a validated proposal to the exact source packet for consumers."""

    owned_document = document if isinstance(document, SourceDocument) else source_document(document)
    unit_values: list[Mapping[str, object]] = [item.as_dict() if isinstance(item, SourceUnit) else item for item in units]
    owned_units = source_units(owned_document, unit_values)
    if extraction.document_id != owned_document.document_id:
        raise ValueError("extraction document does not match source packet")
    return FacetResearchPacket(packet_version="paa.facets.packet.v1", source=owned_document, units=owned_units, extraction=extraction)


def source_only_fixture_row(value: Mapping[str, object]) -> tuple[SourceDocument, tuple[SourceUnit, ...]]:
    """Read a blind row without ever exposing selection/gold/adjudication fields."""

    raw_document = value.get("document")
    if not isinstance(raw_document, Mapping):
        raise TypeError("fixture row requires a document object")
    document = source_document(raw_document)
    unit = SourceUnit(unit_id=document.document_id + "-source", text=document.text, start=0, end=len(document.text))
    return document, (unit,)


__all__ = [
    "FACET_PROMPT_VERSION",
    "FACET_PROMPT_VERSIONS",
    "FACET_SCHEMA_VERSION",
    "FACET_VALIDATOR_VERSION",
    "ActionKind",
    "ActorScope",
    "FacetExtractionResult",
    "FacetInvalidRecord",
    "FacetKind",
    "FacetProposal",
    "FacetResearchPacket",
    "FacetResultStatus",
    "FacetRunOutcome",
    "FacetUnitResult",
    "FacetUnitStatus",
    "HorizonKind",
    "PublicationState",
    "SourceAnchor",
    "SourceDocument",
    "SourceUnit",
    "build_facet_request",
    "facet_response_schema",
    "normalize_facet_output",
    "research_packet",
    "source_document",
    "source_only_fixture_row",
    "source_units",
]

"""Source-grounded local-LLM proposition extraction.

The local model is an extraction assistant, not an authority.  This module
owns the boundary around its JSON: every quote and span must be found in the
source document, relative dates need an explicit source year, and all output
remains ``PROPOSED`` until the normal review/record pipeline promotes it.

There are deliberately no HTTP or model calls here.  ``build_request`` and
``build_verification_request`` return payloads for the caller-owned local
client; ``normalize_output`` is safe to run on an untrusted model response.
"""


import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from paa.semantics import Proposition

PROMPT_DIR = Path(__file__).with_name("prompts")
EXTRACTION_SCHEMA_VERSION = "paa.extract.v1"
VERIFICATION_SCHEMA_VERSION = "paa.extract.verify.v1"
BATCH_SCHEMA_VERSION = "paa.extract.batch.v1"
MULTI_BATCH_SCHEMA_VERSION = "paa.extract.multi.v1"
DEFAULT_PROMPT_VERSION = "extract_v1"
DEFAULT_VERIFY_PROMPT_VERSION = "extract_v1_verify"
DEFAULT_BATCH_PROMPT_VERSION = "extract_batch_v1"
DEFAULT_MULTI_BATCH_PROMPT_VERSION = "extract_batch_multi_v1"

# Keep this list aligned with the proposition contract.  The extractor may
# propose a type; it never promotes the type to a reviewed judgment.
SEMANTIC_TYPES = frozenset(
    {
        "FACTUAL_CLAIM",
        "OBSERVED_STATE_FORECAST",
        "CAUSAL_EFFECT_FORECAST",
        "CAUSAL_CLAIM",
        "POSITION",
        "PERSONAL_ACTION_COMMITMENT",
        "PERSONAL_RESTRAINT_COMMITMENT",
        "COLLECTIVE_ACTION_COMMITMENT",
        "OUTCOME_COMMITMENT",
        "PROCESS_COMMITMENT",
        "MAINTAIN_COMMITMENT",
        "PREVENT_COMMITMENT",
        "POLICY_DESIDERATUM",
        "BROAD_OBJECTIVE",
        "VALUE_OR_SLOGAN",
        "QUESTION",
        "REPORTED_SPEECH",
        "AMBIGUOUS",
    }
)

ISSUER_SCOPES = frozenset(
    {"SELF", "PARTY", "GOVERNMENT", "OTHER_COLLECTIVE", "UNSPECIFIED_WE", "OTHER", "UNRESOLVED"}
)
ACTION_KINDS = frozenset(
    {
        "INITIATIVE_AUTHORED",
        "VOTE_CAST",
        "QUESTION_FILED",
        "SPEECH_DELIVERED",
        "RESIGN_ROLE",
        "DONATION",
        "PUBLIC_ADVOCACY",
        "POLICY_RESTRAINT",
        "OTHER_OBSERVABLE_ACTION",
    }
)
CAPABILITIES = frozenset(
    {
        "MP_INITIATE_BILL",
        "PARLIAMENTARY_VOTE",
        "FILE_PARLIAMENTARY_QUESTION",
        "SPEAK_IN_PARLIAMENT",
        "HOLD_ELECTED_ROLE",
        "PARLIAMENTARY_INFLUENCE",
        "PUBLIC_ADVOCACY",
        "POLICYMAKING_ROLE",
        "PERSONAL_FUNDS",
        "OTHER",
    }
)
DEADLINE_BASES = frozenset({"EXPLICIT", "CONTEXT_DERIVED", "UNRESOLVED"})

# Some action kinds have more than one legitimate institutional capability:
# a generic "aloite" may concern Parliament, a council, or public advocacy.
ACTION_CAPABILITIES = {
    "INITIATIVE_AUTHORED": frozenset({"MP_INITIATE_BILL", "POLICYMAKING_ROLE", "PUBLIC_ADVOCACY", "OTHER"}),
    "VOTE_CAST": frozenset({"PARLIAMENTARY_VOTE"}),
    "QUESTION_FILED": frozenset({"FILE_PARLIAMENTARY_QUESTION"}),
    "SPEECH_DELIVERED": frozenset({"SPEAK_IN_PARLIAMENT"}),
    "RESIGN_ROLE": frozenset({"HOLD_ELECTED_ROLE"}),
    "DONATION": frozenset({"PERSONAL_FUNDS"}),
    "PUBLIC_ADVOCACY": frozenset({"PUBLIC_ADVOCACY"}),
    "POLICY_RESTRAINT": frozenset({"POLICYMAKING_ROLE", "PARLIAMENTARY_INFLUENCE"}),
    "OTHER_OBSERVABLE_ACTION": frozenset(CAPABILITIES),
}

PERSONAL_ACTION_TYPES = frozenset({"PERSONAL_ACTION_COMMITMENT", "PERSONAL_RESTRAINT_COMMITMENT"})
NON_ACTION_TYPES = frozenset(
    {
        "FACTUAL_CLAIM",
        "OBSERVED_STATE_FORECAST",
        "CAUSAL_EFFECT_FORECAST",
        "CAUSAL_CLAIM",
        "POSITION",
        "OUTCOME_COMMITMENT",
        "PROCESS_COMMITMENT",
        "MAINTAIN_COMMITMENT",
        "PREVENT_COMMITMENT",
        "POLICY_DESIDERATUM",
        "BROAD_OBJECTIVE",
        "VALUE_OR_SLOGAN",
        "QUESTION",
        "REPORTED_SPEECH",
        "AMBIGUOUS",
    }
)

# The batch protocol intentionally uses short codes.  A full proposition
# object is expensive and error-prone for a large corpus; the caller already
# owns the source-grounded unit/span, so the model only supplies the semantic
# labels and a few quoted qualifiers.  Keep the codebook explicit and stable.
BATCH_TYPE_CODES = {
    "FC": "FACTUAL_CLAIM",
    "OS": "OBSERVED_STATE_FORECAST",
    "CF": "CAUSAL_EFFECT_FORECAST",
    "CC": "CAUSAL_CLAIM",
    "PO": "POSITION",
    "PA": "PERSONAL_ACTION_COMMITMENT",
    "NR": "PERSONAL_RESTRAINT_COMMITMENT",
    "CO": "COLLECTIVE_ACTION_COMMITMENT",
    "OC": "OUTCOME_COMMITMENT",
    "PR": "PROCESS_COMMITMENT",
    "MC": "MAINTAIN_COMMITMENT",
    "PV": "PREVENT_COMMITMENT",
    "PD": "POLICY_DESIDERATUM",
    "BO": "BROAD_OBJECTIVE",
    "SL": "VALUE_OR_SLOGAN",
    "QU": "QUESTION",
    "RS": "REPORTED_SPEECH",
    "AM": "AMBIGUOUS",
}
BATCH_TYPE_CODES_REVERSE = {value: key for key, value in BATCH_TYPE_CODES.items()}
BATCH_SCOPE_CODES = {
    "S": "SELF",
    "P": "PARTY",
    "G": "GOVERNMENT",
    "C": "OTHER_COLLECTIVE",
    "W": "UNSPECIFIED_WE",
    "O": "OTHER",
    "U": "UNRESOLVED",
}
BATCH_SCOPE_CODES_REVERSE = {value: key for key, value in BATCH_SCOPE_CODES.items()}
BATCH_ACTION_CODES = {
    "IA": "INITIATIVE_AUTHORED",
    "VC": "VOTE_CAST",
    "QF": "QUESTION_FILED",
    "SD": "SPEECH_DELIVERED",
    "RR": "RESIGN_ROLE",
    "DN": "DONATION",
    "AD": "PUBLIC_ADVOCACY",
    "RT": "POLICY_RESTRAINT",
    "OA": "OTHER_OBSERVABLE_ACTION",
}
BATCH_ACTION_CODES_REVERSE = {value: key for key, value in BATCH_ACTION_CODES.items()}

# JSON-schema form for clients that support constrained decoding.  Cardinality
# (one row per caller-supplied unit) remains a source-pipeline concern because
# this schema cannot know the current unit IDs; ``normalize_batch_output``
# performs that check after decoding.
BATCH_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["schema_version", "document_id", "source_id", "abstain", "abstention_reason", "coverage", "rows"],
    "properties": {
        "schema_version": {"const": BATCH_SCHEMA_VERSION},
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
        "rows": {
            "type": "array",
            "items": {
                "type": "array",
                "minItems": 8,
                "maxItems": 8,
                "prefixItems": [
                    {"type": "string"},
                    {"enum": sorted(BATCH_TYPE_CODES)},
                    {"enum": sorted(BATCH_SCOPE_CODES)},
                    {"type": "boolean"},
                    {"enum": [*sorted(BATCH_ACTION_CODES), None]},
                    {"type": ["string", "null"]},
                    {"type": ["string", "null"]},
                    {"type": ["string", "null"]},
                ],
                "items": False,
            },
        },
    },
}

_DATE_DMY = re.compile(r"\b(\d{1,2})\.(\d{1,2})\.(\d{4})\b")
_DATE_ISO = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_YEAR_END = re.compile(r"\bvuoden\s+(\d{4})\s+loppuun\b", re.IGNORECASE)
_RELATIVE_YEAR_END = re.compile(
    r"\b(?:tämän|tänä|kuluvan)\s+vuoden(?:\s+(?:puolella|aikana))?\b|\bvuoden\s+loppuun\b",
    re.IGNORECASE,
)
_RELATIVE_YEAR_END_SV = re.compile(
    r"\b(?:i\s+år|detta\s+år|under\s+året|innan\s+årets\s+slut|före\s+årets\s+slut)\b",
    re.IGNORECASE,
)
_YEAR_END_SV = re.compile(
    r"\b(?:år|året)\s+(\d{4})\s+(?:slut|utgång)\b|\b(?:före|innan)\s+utgången\s+av\s+(\d{4})\b",
    re.IGNORECASE,
)
_NEGATION = re.compile(r"\b(?:en|enkä|emme|emmekä|et|etkä|ette|ettekä|ei|eikä|eivät|eivätkä|ettei|etteivät|ellen|ellei|älä|älkää|ilman)\b|\w+matta\b", re.IGNORECASE)
_NEGATION_SV = re.compile(r"\b(?:inte|ingen|inget|inga|utan|ej|aldrig)\b", re.IGNORECASE)
_PLURAL_ISSUER = re.compile(
    r"^(?:me\s+)?(?:lupaamme|teemme|esitämme|äänestämme|kampanjoimme|lahjoitamme|pyrimme)\b",
    re.IGNORECASE,
)
_ALLOWED_PUNCTUATION = re.compile(r"^[\s,;:()\[\]{}—–\-]*$")

_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "document_id",
        "source_id",
        "abstain",
        "abstention_reason",
        "coverage",
        "propositions",
    }
)
_PROPOSITION_KEYS = frozenset(
    {
        "source_quote",
        "source_start",
        "source_end",
        "semantic_type",
        "target_quote",
        "condition_quote",
        "condition_inherited",
        "deadline_quote",
        "deadline_normalized",
        "deadline_basis",
        "negation",
        "issuer_scope",
        "action_kind",
        "required_capability",
        "observable_action",
    }
)
_BATCH_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "document_id",
        "source_id",
        "abstain",
        "abstention_reason",
        "coverage",
        "rows",
    }
)
_BATCH_ROW_KEYS = frozenset(
    {
        "unit_id",
        "type_code",
        "scope_code",
        "negation",
        "action_code",
        "target_quote",
        "condition_quote",
        "deadline_quote",
    }
)
_MULTI_BATCH_TOP_LEVEL_KEYS = frozenset({"schema_version", "documents"})


@dataclass
class InvalidRecord:
    """A visible rejection; invalid model rows are never silently dropped."""

    index: int | None
    code: str
    message: str
    field: str | None = None
    raw: Any = None

    def as_dict(self) -> dict[str, Any]:
        result = {"index": self.index, "code": self.code, "message": self.message}
        if self.field is not None:
            result["field"] = self.field
        if self.raw is not None:
            result["raw"] = self.raw
        return result


@dataclass
class ExtractionResult:
    """Normalized output and its explicit audit state."""

    status: str
    document_id: str | None
    source_id: str | None
    prompt_version: str
    propositions: list[Proposition] = field(default_factory=list)
    invalid_records: list[InvalidRecord] = field(default_factory=list)
    abstention: dict[str, Any] | None = None
    coverage: dict[str, Any] = field(default_factory=dict)
    raw_sha256: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "document_id": self.document_id,
            "source_id": self.source_id,
            "prompt_version": self.prompt_version,
            "propositions": [_proposition_dict(prop) for prop in self.propositions],
            "invalid_records": [item.as_dict() for item in self.invalid_records],
            "abstention": self.abstention,
            "coverage": self.coverage,
            "raw_sha256": self.raw_sha256,
        }


@dataclass(frozen=True)
class BatchUnit:
    """A caller-owned, source-grounded classification unit."""

    unit_id: str
    text: str
    start: int
    end: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "text": self.text,
            "start": self.start,
            "end": self.end,
        }


@dataclass
class BatchUnitResult:
    """One explicit batch outcome, including a visible rejection if invalid."""

    unit_id: str
    status: str
    proposition: Proposition | None = None
    type_code: str | None = None
    scope_code: str | None = None
    negation: bool | None = None
    action_code: str | None = None
    target_quote: str | None = None
    condition_quote: str | None = None
    deadline_quote: str | None = None
    condition_inherited: bool = False
    invalid_records: list[InvalidRecord] = field(default_factory=list)
    raw: Any = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "status": self.status,
            "type_code": self.type_code,
            "scope_code": self.scope_code,
            "negation": self.negation,
            "action_code": self.action_code,
            "target_quote": self.target_quote,
            "condition_quote": self.condition_quote,
            "deadline_quote": self.deadline_quote,
            "condition_inherited": self.condition_inherited,
            "proposition": _proposition_dict(self.proposition) if self.proposition is not None else None,
            "invalid_records": [item.as_dict() for item in self.invalid_records],
            "raw": self.raw,
        }


@dataclass
class BatchExtractionResult:
    """Normalized compact extraction with complete per-unit accounting."""

    status: str
    document_id: str | None
    source_id: str | None
    prompt_version: str
    units: list[BatchUnitResult] = field(default_factory=list)
    invalid_records: list[InvalidRecord] = field(default_factory=list)
    abstention: dict[str, Any] | None = None
    coverage: dict[str, Any] = field(default_factory=dict)
    raw_sha256: str | None = None

    @property
    def propositions(self) -> list[Proposition]:
        return [item.proposition for item in self.units if item.proposition is not None]

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "document_id": self.document_id,
            "source_id": self.source_id,
            "prompt_version": self.prompt_version,
            "units": [item.as_dict() for item in self.units],
            "propositions": [_proposition_dict(prop) for prop in self.propositions],
            "invalid_records": [item.as_dict() for item in self.invalid_records],
            "abstention": self.abstention,
            "coverage": self.coverage,
            "raw_sha256": self.raw_sha256,
        }


@dataclass
class MultiBatchExtractionResult:
    """Results for one compact inference containing several documents."""

    status: str
    prompt_version: str
    documents: list[BatchExtractionResult] = field(default_factory=list)
    invalid_records: list[InvalidRecord] = field(default_factory=list)
    raw_sha256: str | None = None

    @property
    def propositions(self) -> list[Proposition]:
        return [prop for result in self.documents for prop in result.propositions]

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "prompt_version": self.prompt_version,
            "documents": [result.as_dict() for result in self.documents],
            "propositions": [_proposition_dict(prop) for prop in self.propositions],
            "invalid_records": [item.as_dict() for item in self.invalid_records],
            "raw_sha256": self.raw_sha256,
        }


def _proposition_dict(prop: Proposition) -> dict[str, Any]:
    """Serialize a ``semantics.Proposition`` without inventing evidence."""

    return {
        "source_quote": prop.text,
        "source_start": prop.source_start,
        "source_end": prop.source_end,
        "semantic_type": prop.semantic_type,
        "target_quote": prop.targets[0] if prop.targets else None,
        "condition_quote": prop.condition,
        "condition_inherited": False,
        "deadline_quote": None,
        "deadline_normalized": prop.deadline,
        "deadline_basis": prop.deadline_basis,
        "negation": prop.negation,
        "issuer_scope": prop.issuer_scope,
        "action_kind": prop.action_kind,
        "required_capability": prop.required_capability,
        "observable_action": prop.observable_action,
        "validation_state": prop.validation_state,
    }


def _context_year(document: Mapping[str, Any]) -> int | None:
    for key in ("stated_earliest", "stated_latest", "statement_date", "context_date"):
        candidate = document.get(key)
        if isinstance(candidate, Mapping):
            candidate = candidate.get("earliest") or candidate.get("latest")
        if isinstance(candidate, str):
            match = re.search(r"\b(\d{4})\b", candidate)
            if match:
                return int(match.group(1))
    return None


def _has_explicit_negation(text: str, language: str | None = None) -> bool:
    """Detect only language-appropriate explicit negation markers.

    In Swedish, ``en`` is an indefinite article, not the Finnish first-person
    negative auxiliary.  Applying the Finnish regex to Swedish corpus text
    therefore creates a systematic false-negative/positive validation error.
    """

    if isinstance(language, str) and language.casefold().startswith("sv"):
        return bool(_NEGATION_SV.search(text))
    return bool(_NEGATION.search(text))


def _document_context(document: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(document, Mapping):
        raise TypeError("document must be a mapping")
    document_id = str(document.get("document_id") or "").strip()
    text = document.get("text")
    if not document_id:
        raise ValueError("document requires document_id")
    if not isinstance(text, str) or not text:
        raise ValueError("document requires non-empty text")
    language = document.get("language") or "fi"
    # The 2011 source import labels all responses Finnish. Several real
    # responses are plainly Swedish. Use multiple unambiguous Swedish words
    # only for language-sensitive validation; preserve the declared metadata.
    swedish_words = re.findall(r"\b(?:och|skall|ska|jag|att|för|inte|det|som|är|ett|med|på|till|av)\b", text.casefold())
    finnish_words = re.findall(r"\b(?:ja|että|on|olen|haluan|suomen|sekä|mutta|myös|ei)\b", text.casefold())
    if not str(language).startswith("sv") and len(swedish_words) >= 3 and len(swedish_words) > len(finnish_words) + 2:
        language = "sv"
    context = {
        "document_id": document_id,
        "source_id": document.get("source_id"),
        "language": language,
        "field_label": document.get("field_label"),
        "question": document.get("question") or document.get("question_text"),
        "stated_earliest": document.get("stated_earliest"),
        "stated_latest": document.get("stated_latest"),
        "context_year": _context_year(document),
        "text": text,
    }
    return context


def _prompt(prompt_version: str) -> str:
    path = PROMPT_DIR / f"{prompt_version}.txt"
    if not path.is_file():
        raise ValueError(f"unknown local extraction prompt: {prompt_version}")
    return path.read_text(encoding="utf-8")


def build_request(document: Mapping[str, Any], prompt_version: str = DEFAULT_PROMPT_VERSION) -> dict[str, Any]:
    """Build a deterministic client payload; this function never calls a model."""

    context = _document_context(document)
    system = _prompt(prompt_version)
    user = json.dumps(
        {
            "task": "extract_source_grounded_propositions",
            "contract": EXTRACTION_SCHEMA_VERSION,
            "document": context,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return {
        "prompt_version": prompt_version,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": 0,
        "max_tokens": max(1200, min(8000, 500 + len(context["text"]) // 2)),
        "response_format": {"type": "json_object"},
        "metadata": {
            "document_id": context["document_id"],
            "source_id": context["source_id"],
            "source_text_sha256": hashlib.sha256(context["text"].encode("utf-8")).hexdigest(),
            "context_year": context["context_year"],
        },
    }


def batch_response_schema() -> dict[str, Any]:
    """Return an independent constrained-decoding schema for batch rows."""

    return deepcopy(BATCH_RESPONSE_SCHEMA)


def _normalize_batch_units(document: Mapping[str, Any], units: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], list[BatchUnit]]:
    """Validate caller-owned units before exposing them to the model."""

    context = _document_context(document)
    if isinstance(units, (str, bytes)) or not isinstance(units, Sequence):
        raise TypeError("units must be a sequence of source-grounded mappings")
    normalized: list[BatchUnit] = []
    seen: set[str] = set()
    for index, raw in enumerate(units):
        if not isinstance(raw, Mapping):
            raise TypeError(f"unit {index} must be an object")
        unit_id = raw.get("unit_id")
        unit_text = raw.get("text")
        start = raw.get("start")
        end = raw.get("end")
        if not isinstance(unit_id, str) or not unit_id.strip():
            raise ValueError(f"unit {index} requires a non-empty unit_id")
        if unit_id in seen:
            raise ValueError(f"duplicate unit_id: {unit_id}")
        seen.add(unit_id)
        if not isinstance(unit_text, str) or not unit_text:
            raise ValueError(f"unit {unit_id} requires non-empty text")
        if (
            not isinstance(start, int)
            or isinstance(start, bool)
            or not isinstance(end, int)
            or isinstance(end, bool)
            or start < 0
            or end <= start
            or end > len(context["text"])
            or context["text"][start:end] != unit_text
        ):
            raise ValueError(f"unit {unit_id} span is not an exact source slice")
        normalized.append(BatchUnit(unit_id=unit_id, text=unit_text, start=start, end=end))
    if not normalized:
        raise ValueError("units must contain at least one source-grounded unit")
    return context, normalized


def build_batch_request(
    document: Mapping[str, Any],
    units: Sequence[Mapping[str, Any]],
    *,
    prompt_version: str = DEFAULT_BATCH_PROMPT_VERSION,
) -> dict[str, Any]:
    """Build the compact corpus-classification payload without calling a model.

    ``units`` are already segmented by the canonical source pipeline.  Their
    text and offsets are checked here and sent unchanged; no classifier labels
    or inferred targets are supplied as model context.
    """

    context, normalized = _normalize_batch_units(document, units)
    system = _prompt(prompt_version)
    payload = {
        "task": "classify_source_units",
        "contract": BATCH_SCHEMA_VERSION,
        "document": context,
        "units": [unit.as_dict() for unit in normalized],
    }
    user = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    # The protocol is intentionally short: roughly 20--30 output tokens per
    # unit plus a small envelope.  The caller should chunk very large corpora.
    max_tokens = max(256, min(12000, 96 + 30 * len(normalized)))
    return {
        "prompt_version": prompt_version,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
        "metadata": {
            "document_id": context["document_id"],
            "source_id": context["source_id"],
            "source_text_sha256": hashlib.sha256(context["text"].encode("utf-8")).hexdigest(),
            "context_year": context["context_year"],
            "unit_count": len(normalized),
            "unit_ids": [unit.unit_id for unit in normalized],
        },
    }


def _normalize_multi_specs(
    documents: Sequence[Mapping[str, Any]],
) -> list[tuple[dict[str, Any], list[BatchUnit]]]:
    """Accept ``{document..., units:[...]}`` or ``{document, units}`` specs."""

    if isinstance(documents, (str, bytes)) or not isinstance(documents, Sequence):
        raise TypeError("documents must be a sequence of document/unit mappings")
    normalized: list[tuple[dict[str, Any], list[BatchUnit]]] = []
    seen: set[str] = set()
    for index, spec in enumerate(documents):
        if not isinstance(spec, Mapping):
            raise TypeError(f"document spec {index} must be an object")
        units = spec.get("units")
        document = spec.get("document")
        if document is None:
            document = {key: value for key, value in spec.items() if key != "units"}
        if not isinstance(document, Mapping):
            raise TypeError(f"document spec {index} requires a document object")
        context, canonical_units = _normalize_batch_units(document, units)
        if context["document_id"] in seen:
            raise ValueError(f"duplicate document_id: {context['document_id']}")
        seen.add(context["document_id"])
        normalized.append((context, canonical_units))
    if not normalized:
        raise ValueError("documents must contain at least one document")
    return normalized


def build_multi_batch_request(
    documents: Sequence[Mapping[str, Any]],
    *,
    prompt_version: str = DEFAULT_MULTI_BATCH_PROMPT_VERSION,
) -> dict[str, Any]:
    """Build one compact request for several documents.

    Each document carries its full text once.  Unit IDs on the wire are short
    ordinal strings (``"1"``, ``"2"`` ...); the returned metadata retains the
    mapping to canonical source IDs and spans for the normalizer.
    """

    normalized = _normalize_multi_specs(documents)
    system = _prompt(prompt_version)
    payload_documents: list[dict[str, Any]] = []
    unit_maps: dict[str, dict[str, str]] = {}
    unit_counts: dict[str, int] = {}
    for context, units in normalized:
        wire_map = {str(index): unit.unit_id for index, unit in enumerate(units, start=1)}
        unit_maps[context["document_id"]] = wire_map
        unit_counts[context["document_id"]] = len(units)
        payload_documents.append(
            {
                "document": context,
                "units": [
                    {"unit_id": wire_id, "text": unit.text}
                    for wire_id, unit in zip(wire_map, units, strict=True)
                ],
            }
        )
    payload = {
        "task": "classify_source_documents",
        "contract": MULTI_BATCH_SCHEMA_VERSION,
        "documents": payload_documents,
    }
    user = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    total_units = sum(unit_counts.values())
    return {
        "prompt_version": prompt_version,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": 0,
        "max_tokens": max(512, min(16000, 128 + 30 * total_units)),
        "response_format": {"type": "json_object"},
        "metadata": {
            "document_ids": [context["document_id"] for context, _ in normalized],
            "document_count": len(normalized),
            "unit_count": total_units,
            "unit_maps": unit_maps,
            "unit_counts": unit_counts,
        },
    }


def _candidate_dict(candidate: Proposition | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(candidate, Proposition):
        return _proposition_dict(candidate)
    if isinstance(candidate, Mapping):
        return dict(candidate)
    raise ValueError("candidate must be a Proposition or mapping")


def needs_stage2_verification(candidate: Proposition | Mapping[str, Any], *, disagreement: bool = False) -> bool:
    """Return whether a critical candidate deserves a second independent pass."""

    if disagreement:
        return True
    row = _candidate_dict(candidate)
    return bool(
        row.get("negation")
        or row.get("observable_action")
        or row.get("action_kind")
        or row.get("semantic_type") in PERSONAL_ACTION_TYPES
    )


def build_verification_request(
    document: Mapping[str, Any],
    candidate: Proposition | Mapping[str, Any],
    *,
    prompt_version: str = DEFAULT_VERIFY_PROMPT_VERSION,
) -> dict[str, Any]:
    """Build an optional second-pass request for critical fields only."""

    context = _document_context(document)
    row = _candidate_dict(candidate)
    if not row.get("source_quote") and not row.get("text"):
        raise ValueError("verification candidate requires a source quote")
    return {
        "prompt_version": prompt_version,
        "messages": [
            {"role": "system", "content": _prompt(prompt_version)},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "task": "verify_critical_extraction_fields",
                        "contract": VERIFICATION_SCHEMA_VERSION,
                        "document": context,
                        "candidate": row,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            },
        ],
        "temperature": 0,
        "max_tokens": 900,
        "response_format": {"type": "json_object"},
    }


def _invalid(
    result: list[InvalidRecord], index: int | None, code: str, message: str, field: str | None = None, raw: Any = None
) -> None:
    result.append(InvalidRecord(index=index, code=code, message=message, field=field, raw=raw))


def _parse_raw(raw: Mapping[str, Any] | str) -> tuple[dict[str, Any] | None, list[InvalidRecord]]:
    errors: list[InvalidRecord] = []
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            _invalid(errors, None, "INVALID_JSON", f"model output is not JSON: {exc.msg}")
            return None, errors
    elif isinstance(raw, Mapping):
        parsed = dict(raw)
    else:
        _invalid(errors, None, "INVALID_OUTPUT_TYPE", "model output must be a JSON object")
        return None, errors
    if not isinstance(parsed, dict):
        _invalid(errors, None, "OUTPUT_NOT_OBJECT", "model output must decode to a JSON object")
        return None, errors
    unknown = sorted(set(parsed) - _TOP_LEVEL_KEYS)
    if unknown:
        _invalid(errors, None, "UNSUPPORTED_TOP_LEVEL_FIELDS", f"unsupported top-level fields: {unknown}")
    return parsed, errors


def _span_for_quote(text: str, quote: str, start: Any, end: Any, index: int, errors: list[InvalidRecord]) -> tuple[int, int] | None:
    if not isinstance(quote, str) or not quote:
        _invalid(errors, index, "SOURCE_QUOTE_REQUIRED", "source_quote must be a non-empty exact source substring", "source_quote")
        return None
    if not isinstance(start, int) or isinstance(start, bool) or not isinstance(end, int) or isinstance(end, bool):
        _invalid(errors, index, "SOURCE_SPAN_REQUIRED", "source_start/source_end must be integer offsets", "source_span")
        return None
    if start < 0 or end <= start or end > len(text) or text[start:end] != quote:
        _invalid(errors, index, "SOURCE_SPAN_MISMATCH", "source span is not an exact slice of the document", "source_span")
        return None
    return start, end


def _exact_anchor(text: str, value: Any, index: int, field_name: str, errors: list[InvalidRecord]) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        _invalid(errors, index, "ANCHOR_NOT_STRING", f"{field_name} must be null or a non-empty exact quote", field_name)
        return None
    if value not in text:
        _invalid(errors, index, "ANCHOR_NOT_IN_SOURCE", f"{field_name} is not an exact source substring", field_name)
        return None
    return value


def _date_from_quote(quote: str, context_year: int | None, language: str | None = None) -> tuple[str | None, str]:
    match = _DATE_DMY.search(quote)
    if match:
        day, month, year = (int(value) for value in match.groups())
        try:
            return date(year, month, day).isoformat(), "EXPLICIT"
        except ValueError:
            return None, "UNRESOLVED"
    match = _DATE_ISO.search(quote)
    if match:
        try:
            return date(int(match.group(1)), int(match.group(2)), int(match.group(3))).isoformat(), "EXPLICIT"
        except ValueError:
            return None, "UNRESOLVED"
    match = _YEAR_END.search(quote)
    if match:
        return f"{match.group(1)}-12-31", "EXPLICIT"
    if isinstance(language, str) and language.casefold().startswith("sv"):
        match = _YEAR_END_SV.search(quote)
        if match:
            year = match.group(1) or match.group(2)
            return f"{year}-12-31", "EXPLICIT"
    relative = _RELATIVE_YEAR_END_SV.search(quote) if isinstance(language, str) and language.casefold().startswith("sv") else _RELATIVE_YEAR_END.search(quote)
    if relative:
        if context_year is None:
            return None, "UNRESOLVED"
        return f"{context_year:04d}-12-31", "CONTEXT_DERIVED"
    return None, "UNRESOLVED"


def _has_deadline_anchor(text: str, language: str | None = None) -> bool:
    relative = _RELATIVE_YEAR_END_SV if isinstance(language, str) and language.casefold().startswith("sv") else _RELATIVE_YEAR_END
    year_end = _YEAR_END_SV if isinstance(language, str) and language.casefold().startswith("sv") else _YEAR_END
    # A past publication/reference date is not a deadline. Exact day dates
    # require explicit deadline phrasing before omission becomes an error.
    dated_deadline = bool((_DATE_DMY.search(text) or _DATE_ISO.search(text)) and
        re.search(r"\b(?:mennessä|viimeistään|ennen|by|senast|före)\b", text, re.IGNORECASE))
    return bool(dated_deadline or year_end.search(text) or relative.search(text))


def _condition_is_adjacent(text: str, source_start: int, condition: str) -> bool:
    positions = [match.start() for match in re.finditer(re.escape(condition), text)]
    for position in positions:
        end = position + len(condition)
        if end <= source_start and _ALLOWED_PUNCTUATION.fullmatch(text[end:source_start]):
            return True
    return False


def _validate_proposition(
    raw: Any,
    index: int,
    context: Mapping[str, Any],
    errors: list[InvalidRecord],
) -> Proposition | None:
    if not isinstance(raw, Mapping):
        _invalid(errors, index, "PROPOSITION_NOT_OBJECT", "proposition must be an object", raw=raw)
        return None
    row = dict(raw)
    unknown = sorted(set(row) - _PROPOSITION_KEYS)
    if unknown:
        _invalid(errors, index, "UNSUPPORTED_PROPOSITION_FIELDS", f"unsupported proposition fields: {unknown}", raw=row)
    missing = sorted(_PROPOSITION_KEYS - set(row))
    if missing:
        _invalid(errors, index, "PROPOSITION_FIELDS_MISSING", f"missing proposition fields: {missing}", raw=row)
        return None
    text = str(context["text"])
    span = _span_for_quote(text, row["source_quote"], row["source_start"], row["source_end"], index, errors)
    if span is None:
        return None
    source_start, source_end = span
    semantic_type = row["semantic_type"]
    semantic_type_key = semantic_type if isinstance(semantic_type, str) else ""
    if semantic_type_key not in SEMANTIC_TYPES:
        _invalid(errors, index, "UNKNOWN_SEMANTIC_TYPE", f"unsupported semantic_type: {semantic_type!r}", "semantic_type")
    issuer_scope = row["issuer_scope"]
    if not isinstance(issuer_scope, str) or issuer_scope not in ISSUER_SCOPES:
        _invalid(errors, index, "UNKNOWN_ISSUER_SCOPE", f"unsupported issuer_scope: {issuer_scope!r}", "issuer_scope")
    if not isinstance(row["negation"], bool):
        _invalid(errors, index, "NEGATION_NOT_BOOLEAN", "negation must be boolean", "negation")
    source_negation = _has_explicit_negation(row["source_quote"], context.get("language"))
    if isinstance(row["negation"], bool) and row["negation"] != source_negation:
        _invalid(errors, index, "NEGATION_NOT_SOURCE_GROUNDED", "negation disagrees with explicit source wording", "negation")
    if semantic_type == "PERSONAL_RESTRAINT_COMMITMENT" and row["negation"] is not True:
        _invalid(errors, index, "RESTRAINT_NEEDS_NEGATION", "personal restraint must carry explicit negation", "semantic_type")

    target_quote = _exact_anchor(text, row["target_quote"], index, "target_quote", errors)
    condition_quote = _exact_anchor(text, row["condition_quote"], index, "condition_quote", errors)
    deadline_quote = _exact_anchor(text, row["deadline_quote"], index, "deadline_quote", errors)
    if _PLURAL_ISSUER.search(row["source_quote"]) and issuer_scope not in {
        "OTHER_COLLECTIVE",
        "PARTY",
        "GOVERNMENT",
        "UNSPECIFIED_WE",
    }:
        _invalid(errors, index, "PLURAL_SCOPE_CONFLICT", "plural source wording cannot be assigned to SELF", "issuer_scope")

    condition = condition_quote
    if condition is None and row["condition_inherited"] is not False:
        _invalid(errors, index, "CONDITION_INHERITANCE_INVALID", "condition_inherited must be false when condition_quote is null", "condition_inherited")
    elif not isinstance(row["condition_inherited"], bool):
        _invalid(errors, index, "CONDITION_INHERITANCE_NOT_BOOLEAN", "condition_inherited must be boolean", "condition_inherited")
    elif condition is not None:
        if condition in row["source_quote"] and row["condition_inherited"]:
            _invalid(errors, index, "CONDITION_INHERITANCE_FALSE", "an in-span condition cannot be marked inherited", "condition_inherited")
        if condition not in row["source_quote"] and (
            not row["condition_inherited"] or not _condition_is_adjacent(text, source_start, condition)
        ):
            _invalid(errors, index, "CONDITION_NOT_BOUND", "condition is neither in the proposition nor an adjacent inherited qualifier", "condition_quote")

    normalized_deadline = row["deadline_normalized"]
    deadline_basis = row["deadline_basis"]
    if not isinstance(deadline_basis, str) or deadline_basis not in DEADLINE_BASES:
        _invalid(errors, index, "UNKNOWN_DEADLINE_BASIS", f"unsupported deadline_basis: {deadline_basis!r}", "deadline_basis")
    if deadline_quote is None:
        if normalized_deadline is not None or deadline_basis != "UNRESOLVED":
            _invalid(errors, index, "DEADLINE_WITHOUT_QUOTE", "a normalized deadline requires an exact deadline quote", "deadline")
        if _has_deadline_anchor(row["source_quote"], context.get("language")):
            _invalid(errors, index, "DEADLINE_OMITTED", "source proposition contains a supported date/deadline anchor", "deadline_quote")
    else:
        if deadline_quote not in row["source_quote"]:
            _invalid(errors, index, "DEADLINE_NOT_IN_PROPOSITION", "deadline quote must be inside source_quote", "deadline_quote")
        expected, expected_basis = _date_from_quote(
            deadline_quote, context.get("context_year"), context.get("language")
        )
        if expected is None:
            _invalid(errors, index, "UNSUPPORTED_DEADLINE_ANCHOR", "deadline wording is not safely normalizable", "deadline_quote")
        if normalized_deadline != expected:
            _invalid(errors, index, "DEADLINE_NORMALIZATION_MISMATCH", "normalized deadline does not match the quoted anchor", "deadline_normalized")
        if deadline_basis != expected_basis:
            _invalid(errors, index, "DEADLINE_BASIS_MISMATCH", "deadline basis does not match the quoted anchor", "deadline_basis")
        if not isinstance(normalized_deadline, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", normalized_deadline):
            _invalid(errors, index, "DEADLINE_NOT_ISO_DATE", "normalized deadline must be YYYY-MM-DD or null", "deadline_normalized")

    action_kind = row["action_kind"]
    capability = row["required_capability"]
    action_kind_key = action_kind if isinstance(action_kind, str) else None
    capability_key = capability if isinstance(capability, str) else None
    if action_kind is not None and action_kind_key not in ACTION_KINDS:
        _invalid(errors, index, "UNKNOWN_ACTION_KIND", f"unsupported action_kind: {action_kind!r}", "action_kind")
    if capability is not None and capability_key not in CAPABILITIES:
        _invalid(errors, index, "UNKNOWN_CAPABILITY", f"unsupported required_capability: {capability!r}", "required_capability")
    if action_kind is None and capability is not None:
        _invalid(errors, index, "CAPABILITY_WITHOUT_ACTION", "required_capability requires an action_kind", "required_capability")
    if action_kind_key is not None and capability_key not in ACTION_CAPABILITIES.get(action_kind_key, set()):
        _invalid(errors, index, "ACTION_CAPABILITY_MISMATCH", "action_kind and required_capability are incompatible", "required_capability")
    if not isinstance(row["observable_action"], bool):
        _invalid(errors, index, "OBSERVABLE_ACTION_NOT_BOOLEAN", "observable_action must be boolean", "observable_action")
    if semantic_type_key in NON_ACTION_TYPES and (action_kind is not None or row["observable_action"]):
        _invalid(errors, index, "ACTION_ON_NON_ACTION_TYPE", "non-action semantic types cannot launch an action ledger", "action_kind")
    if semantic_type_key in PERSONAL_ACTION_TYPES and issuer_scope != "SELF":
        _invalid(errors, index, "PERSONAL_SCOPE_CONFLICT", "personal action types require SELF issuer_scope", "issuer_scope")
    if semantic_type_key in PERSONAL_ACTION_TYPES and action_kind is not None and row["observable_action"] is not True:
        _invalid(errors, index, "ACTION_NOT_MARKED_OBSERVABLE", "a canonical personal action must be observable_action=true", "observable_action")
    if semantic_type_key == "COLLECTIVE_ACTION_COMMITMENT" and (action_kind is not None or row["observable_action"]):
        _invalid(errors, index, "COLLECTIVE_ACTION_NOT_PERSONAL", "collective commitments cannot be emitted as personal observable actions", "action_kind")

    if errors and any(item.index == index for item in errors):
        return None

    personal = semantic_type_key in PERSONAL_ACTION_TYPES and issuer_scope == "SELF"
    if personal:
        testability = "NARROW" if action_kind and normalized_deadline else "PARTIAL"
    elif semantic_type_key in {"VALUE_OR_SLOGAN", "POSITION", "QUESTION", "REPORTED_SPEECH"}:
        testability = "NOT_A_COMMITMENT"
    elif semantic_type_key == "PROCESS_COMMITMENT":
        testability = "CASE_REVIEW"
    elif semantic_type_key == "AMBIGUOUS":
        testability = "UNRESOLVED"
    elif semantic_type_key in {"BROAD_OBJECTIVE", "POLICY_DESIDERATUM"}:
        testability = "PARTIAL" if target_quote or normalized_deadline else "NOT_TESTABLE_AS_WRITTEN"
    else:
        testability = "PARTIAL"
    missing_specification: list[str] = []
    if personal and not normalized_deadline:
        missing_specification.append("deadline")
    if personal and not target_quote:
        missing_specification.append("target")
    return Proposition(
        text=row["source_quote"],
        semantic_type=semantic_type_key,
        testability=testability,
        personal_action_commitment=personal,
        issuer_scope=issuer_scope,
        deadline=normalized_deadline,
        deadline_basis=deadline_basis,
        negation=row["negation"],
        condition=condition,
        targets=[target_quote] if target_quote else [],
        guarantees_implementation=False,
        effect_is_counterfactual=semantic_type_key in {"CAUSAL_EFFECT_FORECAST", "CAUSAL_CLAIM", "OBSERVED_STATE_FORECAST"},
        reported_speech=semantic_type_key == "REPORTED_SPEECH",
        missing_specification=missing_specification,
        action_kind=action_kind_key,
        required_capability=capability_key,
        observable_action=row["observable_action"],
        source_start=source_start,
        source_end=source_end,
        validation_state="PROPOSED",
    )


def _coverage(context: Mapping[str, Any], props: Sequence[Proposition], declared: Any, invalid_count: int) -> dict[str, Any]:
    text = str(context["text"])
    spans = sorted((prop.source_start, prop.source_end) for prop in props if prop.source_start is not None and prop.source_end is not None)
    covered = 0
    cursor = 0
    overlaps = 0
    for start, end in spans:
        if start < cursor:
            overlaps += 1
        if end > cursor:
            covered += end - max(start, cursor)
            cursor = end
    non_whitespace = sum(not char.isspace() for char in text)
    covered_non_whitespace = sum(
        not char.isspace() for start, end in spans for char in text[start:end]
    )
    # The count above intentionally counts overlaps; it is diagnostic only.
    status = declared.get("status") if isinstance(declared, Mapping) else None
    if not props:
        computed = "NONE"
    elif invalid_count:
        computed = "PARTIAL"
    elif status in {"PARTIAL", "ABSTAIN"}:
        computed = status
    else:
        computed = "COMPLETE"
    return {
        "declared": dict(declared) if isinstance(declared, Mapping) else None,
        "computed_status": computed,
        "proposition_count": len(props),
        "invalid_count": invalid_count,
        "source_length": len(text),
        "covered_span_characters": covered,
        "covered_non_whitespace_characters": covered_non_whitespace,
        "source_non_whitespace_characters": non_whitespace,
        "overlapping_span_count": overlaps,
    }


def normalize_output(
    raw: Mapping[str, Any] | str,
    document: Mapping[str, Any],
    *,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
) -> ExtractionResult:
    """Validate and convert one model response into proposed propositions."""

    try:
        context = _document_context(document)
    except ValueError as exc:
        return ExtractionResult(
            status="INVALID",
            document_id=None,
            source_id=None,
            prompt_version=prompt_version,
            invalid_records=[InvalidRecord(None, "INVALID_DOCUMENT", str(exc))],
            coverage={"computed_status": "INVALID", "proposition_count": 0, "invalid_count": 1},
        )
    parsed, errors = _parse_raw(raw)
    raw_hash = hashlib.sha256(
        (raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False, sort_keys=True)).encode("utf-8")
    ).hexdigest()
    base = ExtractionResult(
        status="INVALID",
        document_id=context["document_id"],
        source_id=context["source_id"],
        prompt_version=prompt_version,
        invalid_records=errors,
        raw_sha256=raw_hash,
    )
    if parsed is None:
        base.coverage = {"computed_status": "INVALID", "proposition_count": 0, "invalid_count": len(errors)}
        return base
    if parsed.get("schema_version") != EXTRACTION_SCHEMA_VERSION:
        _invalid(errors, None, "SCHEMA_VERSION_MISMATCH", f"expected {EXTRACTION_SCHEMA_VERSION!r}", "schema_version")
    if parsed.get("document_id") != context["document_id"]:
        _invalid(errors, None, "DOCUMENT_ID_MISMATCH", "model output is for a different document", "document_id")
    if context["source_id"] is not None and parsed.get("source_id") != context["source_id"]:
        _invalid(errors, None, "SOURCE_ID_MISMATCH", "model output is for a different source", "source_id")
    if not isinstance(parsed.get("abstain"), bool):
        _invalid(errors, None, "ABSTAIN_NOT_BOOLEAN", "abstain must be boolean", "abstain")
    if not isinstance(parsed.get("propositions"), list):
        _invalid(errors, None, "PROPOSITIONS_NOT_LIST", "propositions must be a list", "propositions")
        parsed["propositions"] = []
    abstain = parsed.get("abstain") is True
    reason = parsed.get("abstention_reason")
    if abstain and (not isinstance(reason, str) or not reason.strip()):
        _invalid(errors, None, "ABSTENTION_REASON_REQUIRED", "explicit abstention requires a reason", "abstention_reason")
    if not abstain and parsed.get("propositions") == []:
        _invalid(errors, None, "EMPTY_WITHOUT_ABSTENTION", "empty extraction must be explicit abstention", "propositions")
    declared_coverage = parsed.get("coverage")
    if not isinstance(declared_coverage, Mapping) or declared_coverage.get("status") not in {"COMPLETE", "PARTIAL", "ABSTAIN"}:
        _invalid(errors, None, "COVERAGE_REQUIRED", "coverage.status must be COMPLETE, PARTIAL, or ABSTAIN", "coverage")
    if abstain and isinstance(declared_coverage, Mapping) and declared_coverage.get("status") != "ABSTAIN":
        _invalid(errors, None, "ABSTENTION_COVERAGE_MISMATCH", "abstain=true requires coverage.status=ABSTAIN", "coverage")
    if not abstain and isinstance(declared_coverage, Mapping) and declared_coverage.get("status") == "ABSTAIN":
        _invalid(errors, None, "COVERAGE_ABSTENTION_MISMATCH", "non-abstaining output cannot declare ABSTAIN coverage", "coverage")

    propositions: list[Proposition] = []
    for index, row in enumerate(parsed.get("propositions", [])):
        before = len(errors)
        proposition = _validate_proposition(row, index, context, errors)
        if proposition is not None and len(errors) == before:
            propositions.append(proposition)
    base.propositions = propositions
    base.abstention = {"reason": reason, "source": "MODEL_EXPLICIT"} if abstain else None
    base.coverage = _coverage(context, propositions, declared_coverage, len(errors))
    base.invalid_records = errors
    if abstain and not any(item.index is None and item.code in {"ABSTENTION_REASON_REQUIRED", "ABSTENTION_COVERAGE_MISMATCH"} for item in errors):
        base.status = "ABSTAIN"
    elif propositions and not errors:
        base.status = "VALID"
    elif propositions:
        base.status = "PARTIAL"
    else:
        base.status = "INVALID"
    return base


def _parse_batch_raw(raw: Mapping[str, Any] | str) -> tuple[dict[str, Any] | None, list[InvalidRecord]]:
    errors: list[InvalidRecord] = []
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            _invalid(errors, None, "INVALID_JSON", f"model output is not JSON: {exc.msg}")
            return None, errors
    elif isinstance(raw, Mapping):
        parsed = dict(raw)
    else:
        _invalid(errors, None, "INVALID_OUTPUT_TYPE", "model output must be a JSON object")
        return None, errors
    if not isinstance(parsed, dict):
        _invalid(errors, None, "OUTPUT_NOT_OBJECT", "model output must decode to a JSON object")
        return None, errors
    unknown = sorted(set(parsed) - _BATCH_TOP_LEVEL_KEYS)
    if unknown:
        _invalid(errors, None, "UNSUPPORTED_TOP_LEVEL_FIELDS", f"unsupported top-level fields: {unknown}")
    return parsed, errors


def _batch_row_mapping(raw: Any, index: int, errors: list[InvalidRecord]) -> dict[str, Any] | None:
    """Convert the compact array row to named fields for safe validation.

    Objects are accepted as a compatibility escape hatch for hand-authored
    fixtures, but the prompt asks the model for the eight-field array because
    it is materially cheaper in a large batch.
    """

    if isinstance(raw, Mapping):
        row = dict(raw)
        unknown = sorted(set(row) - _BATCH_ROW_KEYS)
        if unknown:
            _invalid(errors, index, "UNSUPPORTED_BATCH_ROW_FIELDS", f"unsupported row fields: {unknown}", raw=raw)
        missing = sorted(_BATCH_ROW_KEYS - set(row))
        if missing:
            _invalid(errors, index, "BATCH_ROW_FIELDS_MISSING", f"missing row fields: {missing}", raw=raw)
            return None
        return row
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        if len(raw) != 8:
            _invalid(errors, index, "BATCH_ROW_LENGTH", "compact row must contain exactly eight fields", raw=raw)
            return None
        return dict(
            zip(
                (
                    "unit_id",
                    "type_code",
                    "scope_code",
                    "negation",
                    "action_code",
                    "target_quote",
                    "condition_quote",
                    "deadline_quote",
                ),
                raw,
                strict=True,
            )
        )
    _invalid(errors, index, "BATCH_ROW_NOT_ARRAY", "batch row must be an eight-field array", raw=raw)
    return None


def _batch_condition(
    context: Mapping[str, Any],
    unit: BatchUnit,
    units: Sequence[BatchUnit],
    value: Any,
    index: int,
    errors: list[InvalidRecord],
) -> tuple[str | None, bool]:
    if value is None:
        return None, False
    if not isinstance(value, str) or not value:
        _invalid(errors, index, "ANCHOR_NOT_STRING", "condition_quote must be null or a non-empty exact quote", "condition_quote")
        return None, False
    if value in unit.text:
        return value, False
    # Inheritance is intentionally local: only the immediately preceding
    # canonical unit may provide a qualifier, with punctuation/whitespace
    # between the two spans.  A quote found elsewhere is not enough.
    previous = [candidate for candidate in units if candidate.end <= unit.start]
    if previous:
        previous_unit = max(previous, key=lambda candidate: candidate.end)
        gap = str(context["text"])[previous_unit.end : unit.start]
        if value in previous_unit.text and _ALLOWED_PUNCTUATION.fullmatch(gap):
            return value, True
    _invalid(
        errors,
        index,
        "CONDITION_NOT_LOCAL",
        "condition_quote must occur in this unit or the immediately adjacent preceding unit",
        "condition_quote",
    )
    return None, False


def _batch_exact_local_anchor(
    value: Any,
    unit: BatchUnit,
    index: int,
    field_name: str,
    errors: list[InvalidRecord],
) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        _invalid(errors, index, "ANCHOR_NOT_STRING", f"{field_name} must be null or a non-empty exact quote", field_name)
        return None
    if value not in unit.text:
        _invalid(errors, index, "ANCHOR_NOT_IN_UNIT", f"{field_name} must be an exact substring of this unit", field_name)
        return None
    return value


def _batch_testability(semantic_type: str, action_kind: str | None, deadline: str | None, target: str | None) -> str:
    if semantic_type in PERSONAL_ACTION_TYPES:
        if action_kind and deadline:
            return "NARROW"
        if action_kind or target or deadline:
            return "PARTIAL"
        return "UNRESOLVED"
    if semantic_type in {"VALUE_OR_SLOGAN", "POSITION", "QUESTION", "REPORTED_SPEECH"}:
        return "NOT_A_COMMITMENT"
    if semantic_type == "PROCESS_COMMITMENT":
        return "CASE_REVIEW"
    if semantic_type == "AMBIGUOUS":
        return "UNRESOLVED"
    if semantic_type in {"BROAD_OBJECTIVE", "POLICY_DESIDERATUM"}:
        return "PARTIAL" if target or deadline else "NOT_TESTABLE_AS_WRITTEN"
    return "PARTIAL"


def _validate_batch_row(
    raw: Any,
    index: int,
    context: Mapping[str, Any],
    unit: BatchUnit,
    units: Sequence[BatchUnit],
    errors: list[InvalidRecord],
) -> BatchUnitResult:
    row = _batch_row_mapping(raw, index, errors)
    if row is None:
        return BatchUnitResult(unit_id=unit.unit_id, status="INVALID", raw=raw)
    unit_id = row.get("unit_id")
    if unit_id != unit.unit_id:
        _invalid(errors, index, "UNIT_ID_INTERNAL_MISMATCH", "row was dispatched to the wrong unit", "unit_id", raw=raw)
        return BatchUnitResult(unit_id=unit.unit_id, status="INVALID", raw=raw)
    type_code = row.get("type_code")
    scope_code = row.get("scope_code")
    action_code = row.get("action_code")
    negation = row.get("negation")
    semantic_type = BATCH_TYPE_CODES.get(type_code) if isinstance(type_code, str) else None
    issuer_scope = BATCH_SCOPE_CODES.get(scope_code) if isinstance(scope_code, str) else None
    action_kind = BATCH_ACTION_CODES.get(action_code) if isinstance(action_code, str) else None
    if semantic_type is None:
        _invalid(errors, index, "UNKNOWN_TYPE_CODE", f"unsupported type_code: {type_code!r}", "type_code")
    if issuer_scope is None:
        _invalid(errors, index, "UNKNOWN_SCOPE_CODE", f"unsupported scope_code: {scope_code!r}", "scope_code")
    if not isinstance(negation, bool):
        _invalid(errors, index, "NEGATION_NOT_BOOLEAN", "negation must be boolean", "negation")
    else:
        source_negation = _has_explicit_negation(unit.text, context.get("language"))
        # Negation inside a subordinate clause ("regardless of income",
        # "unless elected") does not prove that the main proposition is
        # negated. A positive model flag must have a source anchor; an
        # explicitly negative personal action cannot become a positive act.
        negative_personal = bool(re.match(r"\s*(?:en|enkä|jag\s+(?:ska\s+)?inte)\b", unit.text, re.IGNORECASE))
        if (negation and not source_negation) or (not negation and negative_personal and semantic_type in PERSONAL_ACTION_TYPES):
            _invalid(errors, index, "NEGATION_NOT_SOURCE_GROUNDED", "negation disagrees with explicit source wording", "negation")
    if action_code is not None and action_kind is None:
        _invalid(errors, index, "UNKNOWN_ACTION_CODE", f"unsupported action_code: {action_code!r}", "action_code")
    if semantic_type == "PERSONAL_RESTRAINT_COMMITMENT" and negation is not True:
        _invalid(errors, index, "RESTRAINT_NEEDS_NEGATION", "personal restraint must carry explicit negation", "type_code")
    if semantic_type in PERSONAL_ACTION_TYPES and issuer_scope != "SELF":
        _invalid(errors, index, "PERSONAL_SCOPE_CONFLICT", "personal action types require SELF scope", "scope_code")
    if semantic_type == "COLLECTIVE_ACTION_COMMITMENT" and action_kind is not None:
        _invalid(errors, index, "COLLECTIVE_ACTION_NOT_PERSONAL", "collective commitments cannot launch a personal action ledger", "action_code")
    if semantic_type in NON_ACTION_TYPES and action_kind is not None:
        _invalid(errors, index, "ACTION_ON_NON_ACTION_TYPE", "broad/process/value types cannot carry a personal action code", "action_code")
    if action_kind == "POLICY_RESTRAINT" and semantic_type != "PERSONAL_RESTRAINT_COMMITMENT":
        _invalid(errors, index, "RESTRAINT_ACTION_TYPE_MISMATCH", "policy restraint action requires the negative restraint type", "action_code")
    if action_kind is not None and issuer_scope != "SELF":
        _invalid(errors, index, "ACTION_SCOPE_CONFLICT", "an observable action code requires SELF scope", "scope_code")
    if _PLURAL_ISSUER.search(unit.text) and issuer_scope == "SELF":
        _invalid(errors, index, "PLURAL_SCOPE_CONFLICT", "plural source wording cannot be assigned to SELF", "scope_code")

    target_quote = _batch_exact_local_anchor(row.get("target_quote"), unit, index, "target_quote", errors)
    deadline_quote = _batch_exact_local_anchor(row.get("deadline_quote"), unit, index, "deadline_quote", errors)
    condition_quote, condition_inherited = _batch_condition(
        context, unit, units, row.get("condition_quote"), index, errors
    )
    normalized_deadline: str | None = None
    deadline_basis = "UNRESOLVED"
    if deadline_quote is None:
        if _has_deadline_anchor(unit.text, context.get("language")):
            _invalid(errors, index, "DEADLINE_OMITTED", "source unit contains a supported deadline anchor", "deadline_quote")
    else:
        normalized_deadline, deadline_basis = _date_from_quote(
            deadline_quote, context.get("context_year"), context.get("language")
        )
        if normalized_deadline is None and not re.search(
            r"\b(?:\d{4}\s+mennessä|\d+\s+vuoden\s+(?:sisään|kuluessa)|år\s+\d{4}|till\s+\d{4})\b",
            deadline_quote, re.IGNORECASE,
        ):
            _invalid(errors, index, "UNSUPPORTED_DEADLINE_ANCHOR", "deadline wording is not safely normalizable", "deadline_quote")

    result = BatchUnitResult(
        unit_id=unit.unit_id,
        status="INVALID",
        type_code=type_code if isinstance(type_code, str) else None,
        scope_code=scope_code if isinstance(scope_code, str) else None,
        negation=negation if isinstance(negation, bool) else None,
        action_code=action_code if isinstance(action_code, str) else None,
        target_quote=target_quote,
        condition_quote=condition_quote,
        deadline_quote=deadline_quote,
        condition_inherited=condition_inherited,
        raw=raw,
    )
    if any(item.index == index for item in errors) or semantic_type is None or issuer_scope is None:
        return result
    personal = semantic_type in PERSONAL_ACTION_TYPES and issuer_scope == "SELF"
    observable_action = personal and action_kind is not None
    missing_specification: list[str] = []
    if personal and not action_kind:
        missing_specification.append("observable_action_kind")
    if personal and not normalized_deadline:
        missing_specification.append("deadline")
    if personal and not target_quote:
        missing_specification.append("target")
    proposition = Proposition(
        text=unit.text,
        semantic_type=semantic_type,
        testability=_batch_testability(semantic_type, action_kind, normalized_deadline, target_quote),
        personal_action_commitment=personal,
        issuer_scope=issuer_scope,
        deadline=normalized_deadline,
        deadline_basis=deadline_basis,
        negation=bool(negation),
        condition=condition_quote,
        targets=[target_quote] if target_quote else [],
        guarantees_implementation=False,
        effect_is_counterfactual=semantic_type in {"CAUSAL_EFFECT_FORECAST", "CAUSAL_CLAIM", "OBSERVED_STATE_FORECAST"},
        reported_speech=semantic_type == "REPORTED_SPEECH",
        missing_specification=missing_specification,
        action_kind=action_kind,
        # The compact protocol has no capability field.  Do not infer one
        # from a verb; opportunity resolution must use the action evidence and
        # actor role interval later.
        required_capability=None,
        observable_action=observable_action,
        source_start=unit.start,
        source_end=unit.end,
        validation_state="PROPOSED",
    )
    result.proposition = proposition
    result.status = "VALID"
    return result


def normalize_batch_output(
    raw: Mapping[str, Any] | str,
    document: Mapping[str, Any],
    units: Sequence[Mapping[str, Any]],
    *,
    prompt_version: str = DEFAULT_BATCH_PROMPT_VERSION,
) -> BatchExtractionResult:
    """Validate compact rows while preserving a terminal outcome per unit."""

    try:
        context, normalized_units = _normalize_batch_units(document, units)
    except (TypeError, ValueError) as exc:
        return BatchExtractionResult(
            status="INVALID",
            document_id=None,
            source_id=None,
            prompt_version=prompt_version,
            invalid_records=[InvalidRecord(None, "INVALID_DOCUMENT_OR_UNITS", str(exc))],
            coverage={"computed_status": "INVALID", "unit_count": 0, "invalid_count": 1},
        )
    raw_hash = hashlib.sha256(
        (raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False, sort_keys=True)).encode("utf-8")
    ).hexdigest()
    parsed, errors = _parse_batch_raw(raw)
    base = BatchExtractionResult(
        status="INVALID",
        document_id=context["document_id"],
        source_id=context["source_id"],
        prompt_version=prompt_version,
        invalid_records=errors,
        raw_sha256=raw_hash,
    )
    if parsed is None:
        base.units = [BatchUnitResult(unit_id=unit.unit_id, status="INVALID") for unit in normalized_units]
        base.coverage = {
            "computed_status": "INVALID",
            "unit_count": len(normalized_units),
            "covered_unit_count": 0,
            "proposition_count": 0,
            "invalid_count": len(errors),
        }
        return base
    if parsed.get("schema_version") != BATCH_SCHEMA_VERSION:
        _invalid(errors, None, "SCHEMA_VERSION_MISMATCH", f"expected {BATCH_SCHEMA_VERSION!r}", "schema_version")
    if parsed.get("document_id") != context["document_id"]:
        _invalid(errors, None, "DOCUMENT_ID_MISMATCH", "model output is for a different document", "document_id")
    if context["source_id"] is not None and parsed.get("source_id") != context["source_id"]:
        _invalid(errors, None, "SOURCE_ID_MISMATCH", "model output is for a different source", "source_id")
    if not isinstance(parsed.get("abstain"), bool):
        _invalid(errors, None, "ABSTAIN_NOT_BOOLEAN", "abstain must be boolean", "abstain")
    if not isinstance(parsed.get("rows"), list):
        _invalid(errors, None, "ROWS_NOT_LIST", "rows must be a list", "rows")
        parsed["rows"] = []
    declared_coverage = parsed.get("coverage")
    if not isinstance(declared_coverage, Mapping) or declared_coverage.get("status") not in {"COMPLETE", "PARTIAL", "ABSTAIN"}:
        _invalid(errors, None, "COVERAGE_REQUIRED", "coverage.status must be COMPLETE, PARTIAL, or ABSTAIN", "coverage")
    abstain = parsed.get("abstain") is True
    reason = parsed.get("abstention_reason")
    if abstain and (not isinstance(reason, str) or not reason.strip()):
        _invalid(errors, None, "ABSTENTION_REASON_REQUIRED", "explicit abstention requires a reason", "abstention_reason")
    if abstain and parsed.get("rows"):
        _invalid(errors, None, "ABSTAIN_ROWS_PRESENT", "abstention must not include classification rows", "rows")
    if abstain and isinstance(declared_coverage, Mapping) and declared_coverage.get("status") != "ABSTAIN":
        _invalid(errors, None, "ABSTENTION_COVERAGE_MISMATCH", "abstain=true requires coverage.status=ABSTAIN", "coverage")
    if not abstain and not parsed.get("rows"):
        _invalid(errors, None, "EMPTY_WITHOUT_ABSTENTION", "empty batch must be explicit abstention", "rows")
    if not abstain and isinstance(declared_coverage, Mapping) and declared_coverage.get("status") == "ABSTAIN":
        _invalid(errors, None, "COVERAGE_ABSTENTION_MISMATCH", "non-abstaining output cannot declare ABSTAIN coverage", "coverage")

    by_id = {unit.unit_id: unit for unit in normalized_units}
    seen: set[str] = set()
    unit_results: dict[str, BatchUnitResult] = {}
    if abstain:
        # Explicit batch abstention is itself a complete per-unit accounting;
        # missing classification rows are not additional model errors here.
        unit_results = {unit.unit_id: BatchUnitResult(unit_id=unit.unit_id, status="ABSTAIN") for unit in normalized_units}
    else:
        for index, raw_row in enumerate(parsed.get("rows", [])):
            row_mapping = raw_row if isinstance(raw_row, Mapping) else raw_row if isinstance(raw_row, Sequence) else None
            candidate_id = row_mapping.get("unit_id") if isinstance(row_mapping, Mapping) else row_mapping[0] if row_mapping else None
            if not isinstance(candidate_id, str) or candidate_id not in by_id:
                _invalid(errors, index, "UNKNOWN_UNIT_ID", "row unit_id is not one of the supplied units", "unit_id", raw=raw_row)
                continue
            if candidate_id in seen:
                _invalid(errors, index, "DUPLICATE_UNIT_ROW", "each supplied unit must have exactly one row", "unit_id", raw=raw_row)
                continue
            seen.add(candidate_id)
            before = len(errors)
            result = _validate_batch_row(raw_row, index, context, by_id[candidate_id], normalized_units, errors)
            row_errors = [item for item in errors[before:] if item.index == index]
            result.invalid_records.extend(row_errors)
            unit_results[candidate_id] = result
        for unit in normalized_units:
            if unit.unit_id not in seen:
                _invalid(errors, None, "MISSING_UNIT_ROW", "every supplied unit requires a row or explicit batch abstention", "unit_id", raw=unit.unit_id)
                unit_results[unit.unit_id] = BatchUnitResult(unit_id=unit.unit_id, status="INVALID")
    base.units = [unit_results[unit.unit_id] for unit in normalized_units]
    base.abstention = {"reason": reason, "source": "MODEL_EXPLICIT"} if abstain else None
    valid_count = sum(item.status == "VALID" for item in base.units)
    base.coverage = {
        "declared": dict(declared_coverage) if isinstance(declared_coverage, Mapping) else None,
        "computed_status": "ABSTAIN" if abstain and not errors else "COMPLETE" if valid_count == len(normalized_units) and not errors else "PARTIAL" if valid_count else "INVALID",
        "unit_count": len(normalized_units),
        "covered_unit_count": len(seen),
        "proposition_count": len(base.propositions),
        "valid_unit_count": valid_count,
        "invalid_count": len(errors),
    }
    base.invalid_records = errors
    if abstain and not errors:
        base.status = "ABSTAIN"
    elif valid_count == len(normalized_units) and not errors:
        base.status = "VALID"
    elif valid_count:
        base.status = "PARTIAL"
    else:
        base.status = "INVALID"
    return base


def _parse_multi_batch_raw(raw: Mapping[str, Any] | str) -> tuple[dict[str, Any] | None, list[InvalidRecord]]:
    errors: list[InvalidRecord] = []
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            _invalid(errors, None, "INVALID_JSON", f"model output is not JSON: {exc.msg}")
            return None, errors
    elif isinstance(raw, Mapping):
        parsed = dict(raw)
    else:
        _invalid(errors, None, "INVALID_OUTPUT_TYPE", "model output must be a JSON object")
        return None, errors
    if not isinstance(parsed, dict):
        _invalid(errors, None, "OUTPUT_NOT_OBJECT", "model output must decode to a JSON object")
        return None, errors
    unknown = sorted(set(parsed) - _MULTI_BATCH_TOP_LEVEL_KEYS)
    if unknown:
        _invalid(errors, None, "UNSUPPORTED_TOP_LEVEL_FIELDS", f"unsupported top-level fields: {unknown}")
    return parsed, errors


def _invalid_batch_result(context: Mapping[str, Any], units: Sequence[BatchUnit], code: str, message: str) -> BatchExtractionResult:
    error = InvalidRecord(None, code, message)
    return BatchExtractionResult(
        status="INVALID",
        document_id=context.get("document_id"),
        source_id=context.get("source_id"),
        prompt_version=DEFAULT_BATCH_PROMPT_VERSION,
        units=[BatchUnitResult(unit_id=unit.unit_id, status="INVALID") for unit in units],
        invalid_records=[error],
        coverage={
            "computed_status": "INVALID",
            "unit_count": len(units),
            "covered_unit_count": 0,
            "proposition_count": 0,
            "invalid_count": 1,
        },
    )


def _translate_wire_rows(rows: Any, wire_to_canonical: Mapping[str, str]) -> Any:
    if not isinstance(rows, list):
        return rows
    translated: list[Any] = []
    for row in rows:
        if isinstance(row, Mapping):
            item = dict(row)
            wire_id = item.get("unit_id")
            if isinstance(wire_id, str) and wire_id in wire_to_canonical:
                item["unit_id"] = wire_to_canonical[wire_id]
            translated.append(item)
        elif isinstance(row, Sequence) and not isinstance(row, (str, bytes)):
            item = list(row)
            if item and isinstance(item[0], str) and item[0] in wire_to_canonical:
                item[0] = wire_to_canonical[item[0]]
            translated.append(item)
        else:
            translated.append(row)
    return translated


def normalize_multi_batch_output(
    raw: Mapping[str, Any] | str,
    documents: Sequence[Mapping[str, Any]],
    *,
    prompt_version: str = DEFAULT_MULTI_BATCH_PROMPT_VERSION,
) -> MultiBatchExtractionResult:
    """Normalize a multi-document response through the single-document gate."""

    try:
        normalized = _normalize_multi_specs(documents)
    except (TypeError, ValueError) as exc:
        return MultiBatchExtractionResult(
            status="INVALID",
            prompt_version=prompt_version,
            invalid_records=[InvalidRecord(None, "INVALID_DOCUMENTS", str(exc))],
        )
    raw_hash = hashlib.sha256(
        (raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False, sort_keys=True)).encode("utf-8")
    ).hexdigest()
    parsed, errors = _parse_multi_batch_raw(raw)
    base = MultiBatchExtractionResult(status="INVALID", prompt_version=prompt_version, invalid_records=errors, raw_sha256=raw_hash)
    if parsed is None:
        base.documents = [
            _invalid_batch_result(context, units, "INVALID_MULTI_OUTPUT", "multi-document output could not be parsed")
            for context, units in normalized
        ]
        base.invalid_records.extend(error for result in base.documents for error in result.invalid_records)
        return base
    if parsed.get("schema_version") != MULTI_BATCH_SCHEMA_VERSION:
        _invalid(errors, None, "SCHEMA_VERSION_MISMATCH", f"expected {MULTI_BATCH_SCHEMA_VERSION!r}", "schema_version")
    if not isinstance(parsed.get("documents"), list):
        _invalid(errors, None, "DOCUMENT_RESULTS_NOT_LIST", "documents must be a list", "documents")
        parsed["documents"] = []
    expected = {context["document_id"]: (context, units) for context, units in normalized}
    seen: set[str] = set()
    results: dict[str, BatchExtractionResult] = {}
    for index, raw_document in enumerate(parsed.get("documents", [])):
        if not isinstance(raw_document, Mapping):
            _invalid(errors, index, "DOCUMENT_RESULT_NOT_OBJECT", "document result must be an object", raw=raw_document)
            continue
        document_id = raw_document.get("document_id")
        if not isinstance(document_id, str) or document_id not in expected:
            _invalid(errors, index, "UNKNOWN_DOCUMENT_ID", "document result is not one of the supplied documents", "document_id", raw=raw_document)
            continue
        if document_id in seen:
            _invalid(errors, index, "DUPLICATE_DOCUMENT_RESULT", "each supplied document must have exactly one result", "document_id", raw=raw_document)
            continue
        seen.add(document_id)
        context, units = expected[document_id]
        translated = dict(raw_document)
        translated["schema_version"] = BATCH_SCHEMA_VERSION
        wire_to_canonical = {str(number): unit.unit_id for number, unit in enumerate(units, start=1)}
        translated["rows"] = _translate_wire_rows(raw_document.get("rows"), wire_to_canonical)
        result = normalize_batch_output(
            translated,
            context,
            [unit.as_dict() for unit in units],
            prompt_version=prompt_version,
        )
        results[document_id] = result
    for context, units in normalized:
        document_id = context["document_id"]
        if document_id not in seen:
            _invalid(errors, None, "MISSING_DOCUMENT_RESULT", "every supplied document requires one result", "document_id", raw=document_id)
            results[document_id] = _invalid_batch_result(
                context, units, "MISSING_DOCUMENT_RESULT", "model omitted this document result"
            )
    base.documents = [results[context["document_id"]] for context, _ in normalized]
    base.invalid_records = errors + [error for result in base.documents for error in result.invalid_records]
    all_explicitly_abstained = all(result.status == "ABSTAIN" for result in base.documents) and not errors
    all_complete = all(result.status == "VALID" for result in base.documents) and not errors
    if all_explicitly_abstained:
        base.status = "ABSTAIN"
    elif all_complete:
        base.status = "VALID"
    elif any(result.status in {"VALID", "PARTIAL", "ABSTAIN"} for result in base.documents):
        base.status = "PARTIAL"
    else:
        base.status = "INVALID"
    return base


def validate_output(
    raw: Mapping[str, Any] | str,
    document: Mapping[str, Any],
    *,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
) -> ExtractionResult:
    """Compatibility alias emphasizing the validation boundary."""

    return normalize_output(raw, document, prompt_version=prompt_version)


def proposition_records(
    result: ExtractionResult,
    *,
    statement_id: str,
    evidence_id: str,
    actor_ids: list[str],
    id_prefix: str,
) -> list[dict[str, Any]]:
    """Convert proposed objects through the canonical record constructor."""

    from paa.records import proposition_record

    return [
        proposition_record(prop, statement_id, evidence_id, actor_ids, f"{id_prefix}-p{index}")
        for index, prop in enumerate(result.propositions, start=1)
    ]


__all__ = [
    "ACTION_CAPABILITIES",
    "ACTION_KINDS",
    "BATCH_ACTION_CODES",
    "BATCH_RESPONSE_SCHEMA",
    "BATCH_SCHEMA_VERSION",
    "BATCH_SCOPE_CODES",
    "BATCH_TYPE_CODES",
    "CAPABILITIES",
    "MULTI_BATCH_SCHEMA_VERSION",
    "SEMANTIC_TYPES",
    "BatchExtractionResult",
    "BatchUnit",
    "BatchUnitResult",
    "ExtractionResult",
    "InvalidRecord",
    "MultiBatchExtractionResult",
    "batch_response_schema",
    "build_batch_request",
    "build_multi_batch_request",
    "build_request",
    "build_verification_request",
    "needs_stage2_verification",
    "normalize_batch_output",
    "normalize_multi_batch_output",
    "normalize_output",
    "proposition_records",
    "validate_output",
]

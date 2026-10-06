"""Source-backed structural context for institutional document passages.

This module is intentionally a small adapter, not a document ontology.  The
official Vaski committee XML carries distinctions which are lost by the older
plain-text projection: a paragraph may be inside a named dissent, a committee
recommendation, or a quoted/source section.  ``SourceStructureDocument`` keeps
those distinctions next to exact offsets in the already retained canonical
text.  It never repairs an absent heading or infers a speaker from prose.

The parser currently handles the Vaski committee-report XML used by the PAA
question experiments.  Other source formats return an explicit unsupported
status.  This is preferable to manufacturing structure from an HTML/AKN text
projection whose source tree is not available.
"""

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Final

from lxml import etree

SCHEMA_VERSION: Final = "paa.source_structure.v1"
PARSER_VERSION: Final = "vaski_committee_xml_structure_v1"
SUPPORTED_FORMAT: Final = "VASKI_XML"
CANONICAL_NORMALIZATION_VERSION: Final = "xml_document_itertext_tokens_newline_v1"

STATUS_PARSED: Final = "PARSED"
STATUS_UNSUPPORTED: Final = "UNSUPPORTED_FORMAT"
STATUS_INVALID: Final = "INVALID_SOURCE"
STATUS_UNAVAILABLE: Final = "UNAVAILABLE_SOURCE_STRUCTURE"

VOICE_COMMITTEE: Final = "COMMITTEE_MATERIAL"
VOICE_MAJORITY: Final = "COMMITTEE_RECOMMENDATION"
VOICE_DISSENT: Final = "DISSENT"
VOICE_QUOTED: Final = "QUOTED_SOURCE"
VOICE_UNKNOWN: Final = "UNRESOLVED"

_FORMAL_MATTER = re.compile(
    r"\b(?:HE|LA|TPA|KAA|KKV?|VNS|EV|PeVL|StVM|YmVM|SiVM|HaVM|LaVM|TaVM|TyVM|VaVM)"
    r"\s+\d+[/]\d{4}\s+vp\b",
    re.IGNORECASE,
)

# These are source-tree element names, not semantic guesses.  They are the
# text-bearing nodes observed in the retained committee XML.  Unknown nodes
# are left out rather than flattened into a made-up passage role.
_CONTENT_NAMES: Final = frozenset(
    {
        "OtsikkoTeksti",
        "KappaleKooste",
        "SisennettyKappaleKooste",
        "MomenttiKooste",
        "MomenttiKohtaKooste",
        "SaadosKappaleKooste",
        "SaadosKursiiviKooste",
        "SaadosOtsikkoKooste",
        "PykalaTunnusKooste",
        "PykalaNimekeKooste",
        "SaadosNimekeKooste",
        "ValiotsikkoTeksti",
        "Johtolause",
        "LihavaKursiiviOtsikkoTeksti",
        "JohdantoTeksti",
        "KursiiviTeksti",
    }
)
_HEADING_NAMES: Final = frozenset(
    {
        "OtsikkoTeksti",
        "ValiotsikkoTeksti",
        "LihavaKursiiviOtsikkoTeksti",
    }
)
_QUOTED_ANCESTOR_NAMES: Final = frozenset(
    {
        "Lainaus",
        "Sitaatti",
        "SuoraLainaus",
        "LainausOsa",
        "ReferoituPuhe",
    }
)


class SourceStructureError(ValueError):
    """Raised for a malformed retained source or structure wire record."""


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _local(element: etree._Element) -> str:
    return etree.QName(element).localname if isinstance(element.tag, str) else ""


def _clean(value: object) -> str:
    # ``split`` also handles the non-breaking spaces present in some Vaski
    # records.  This is the same visible-text operation used by the retained
    # XML source projection; it does not replace the source bytes.
    return " ".join(str(value or "").split())


def _element_text(element: etree._Element) -> str:
    return _clean(" ".join(element.itertext()))


def canonical_vaski_committee_text(raw_text: str) -> str:
    """Reproduce the retained canonical text projection for one Vaski report.

    The old source fixtures use the report's ``Mietinto`` subtree, retain each
    non-empty XML text node as one cleaned line, and join those lines with a
    newline.  This helper is deliberately limited to that documented source
    projection; it does not add structural labels or rewrite the raw XML.
    """

    if not isinstance(raw_text, str) or not raw_text:
        raise SourceStructureError("Vaski canonical text needs non-empty raw XML")
    try:
        root = etree.fromstring(
            raw_text.encode("utf-8"),
            parser=etree.XMLParser(resolve_entities=False, no_network=True, load_dtd=False, recover=False),
        )
    except (UnicodeEncodeError, etree.XMLSyntaxError) as exc:
        raise SourceStructureError(f"invalid Vaski XML: {exc}") from exc
    reports = [element for element in root.iter() if _local(element) == "Mietinto"]
    if len(reports) != 1:
        raise SourceStructureError(f"expected one Vaski Mietinto subtree, found {len(reports)}")
    lines = [_clean(value) for value in reports[0].itertext()]
    return "\n".join(value for value in lines if value)


def _canonical_occurrences(text: str, value: str) -> list[tuple[int, int]]:
    """Locate normalized XML text while returning original canonical offsets.

    The retained XML projection separates some text-node tokens with newlines
    while the source-tree element presents them as ordinary spaces.  Matching
    the normalized token sequence is safe here because the returned span is
    always the exact substring of the retained canonical text.  Occurrences
    are yielded in source order, which makes repeated identical passages
    deterministic without choosing a semantic interpretation.
    """

    tokens = value.split()
    if not tokens:
        return []
    pattern = r"\s+".join(re.escape(token) for token in tokens)
    return [(match.start(), match.end()) for match in re.finditer(pattern, text)]


def _owned_coverage(value: Mapping[str, Any]) -> Mapping[str, str | int | float | bool | None]:
    """Own the small scalar coverage record at the semantic boundary.

    Coverage is deliberately not an open-ended nested metadata bag.  The
    parser currently emits counters and a format label; rejecting containers
    here prevents a caller from mutating a nested value after a supposedly
    frozen document was constructed.
    """

    if not isinstance(value, Mapping):
        raise SourceStructureError("structure coverage must be an object")
    owned: dict[str, str | int | float | bool | None] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise SourceStructureError("structure coverage keys must be strings")
        if item is not None and not isinstance(item, (str, int, float, bool)):
            raise SourceStructureError("structure coverage values must be scalar")
        owned[key] = item
    return MappingProxyType(owned)


def _path(element: etree._Element) -> tuple[str, ...]:
    """Return a stable local-name/sibling-index path into the raw XML tree."""

    parts: list[str] = []
    current: etree._Element | None = element
    while current is not None and isinstance(current.tag, str):
        name = _local(current)
        siblings = [child for child in current.getparent() if _local(child) == name] if current.getparent() is not None else []
        index = siblings.index(current) + 1 if current in siblings else 1
        parts.append(f"{name}[{index}]")
        current = current.getparent()
    return tuple(reversed(parts))


def _attrs(element: etree._Element) -> dict[str, str]:
    return {key.rsplit("}", 1)[-1]: str(value).strip() for key, value in element.attrib.items()}


def _nearest_heading(ancestors: tuple[etree._Element, ...]) -> str | None:
    for element in reversed(ancestors):
        if _local(element) in _HEADING_NAMES:
            value = _element_text(element)
            if value:
                return value
    return None


def _heading_chain(ancestors: tuple[etree._Element, ...]) -> tuple[str, ...]:
    values: list[str] = []
    for element in ancestors:
        if _local(element) not in _HEADING_NAMES:
            continue
        value = _element_text(element)
        if value and value not in values:
            values.append(value)
    return tuple(values)


def _stage(ancestors: tuple[etree._Element, ...], headings: tuple[str, ...]) -> tuple[str | None, str]:
    names = {_local(element) for element in ancestors}
    lowered = " ".join(headings).casefold()
    if "JasenMielipideOsa" in names or any("vastalause" in value.casefold() for value in headings):
        if "muutosehdotus" in lowered or "muutosehdotukset" in lowered:
            return "DISSENT_AMENDMENT_PROPOSAL", "EXPLICIT_HEADING"
        if "ehdotus" in lowered:
            return "DISSENT_PROPOSAL", "EXPLICIT_HEADING"
        return "DISSENT_ARGUMENT", "EXPLICIT_ANCESTOR"
    if "päätösehdotus" in lowered or "paatos ehdotus" in lowered:
        return "COMMITTEE_RECOMMENDATION", "EXPLICIT_HEADING"
    if "yksityiskohtaiset perustelut" in lowered:
        return "COMMITTEE_DETAILED_REASONS", "EXPLICIT_HEADING"
    if "yleisperustelut" in lowered or "valiokunnan perustelut" in lowered:
        return "COMMITTEE_REASONS", "EXPLICIT_HEADING"
    if "hallituksen esitys" in lowered or "lakiehdotus" in lowered:
        return "PROPOSAL_MATERIAL", "EXPLICIT_HEADING"
    if "johdanto" in lowered:
        return "INTRODUCTION", "EXPLICIT_HEADING"
    # Do not turn a generic paragraph into a procedural assertion.
    return None, "UNRESOLVED"


def _voice(ancestors: tuple[etree._Element, ...], headings: tuple[str, ...]) -> tuple[str, str]:
    names = {_local(element) for element in ancestors}
    if names.intersection(_QUOTED_ANCESTOR_NAMES):
        return VOICE_QUOTED, "EXPLICIT_QUOTE_ELEMENT"
    if "JasenMielipideOsa" in names or any("vastalause" in value.casefold() for value in headings):
        return VOICE_DISSENT, "EXPLICIT_DISSENT_ANCESTOR"
    if any("päätösehdotus" in value.casefold() for value in headings):
        return VOICE_MAJORITY, "EXPLICIT_RECOMMENDATION_HEADING"
    if "Mietinto" in names:
        return VOICE_COMMITTEE, "COMMITTEE_REPORT_ANCESTOR"
    return VOICE_UNKNOWN, "UNRESOLVED"


def _matter_refs(text: str) -> tuple[str, ...]:
    values: list[str] = []
    for match in _FORMAL_MATTER.finditer(text):
        value = " ".join(match.group(0).split())
        # Preserve source spelling apart from whitespace; formal ID identity is
        # case-insensitive for parsing but the quote itself remains unchanged.
        if value not in values:
            values.append(value)
    return tuple(values)


@dataclass(frozen=True, slots=True, kw_only=True)
class SourceStructureSpan:
    """One exact canonical-text span plus source-supported context."""

    span_id: str
    source_id: str
    raw_sha256: str
    text_sha256: str
    start: int
    end: int
    text: str
    raw_path: tuple[str, ...]
    ancestor_path: tuple[str, ...]
    heading: str | None
    heading_path: tuple[str, ...]
    voice_scope: str
    voice_basis: str
    matter_refs: tuple[str, ...]
    procedural_stage: str | None
    procedural_stage_basis: str
    source_element: str

    def __post_init__(self) -> None:
        if not self.source_id or not self.raw_sha256 or not self.text_sha256:
            raise SourceStructureError("structure span requires source/version identity")
        if len(self.raw_sha256) != 64 or len(self.text_sha256) != 64:
            raise SourceStructureError("structure span hashes must be SHA-256")
        if self.start < 0 or self.end <= self.start or not self.text:
            raise SourceStructureError("structure span offsets/text are invalid")
        if self.end - self.start != len(self.text):
            raise SourceStructureError("structure span offsets do not match text")
        if self.voice_scope not in {VOICE_COMMITTEE, VOICE_MAJORITY, VOICE_DISSENT, VOICE_QUOTED, VOICE_UNKNOWN}:
            raise SourceStructureError(f"unknown voice scope: {self.voice_scope}")

    def to_wire(self) -> dict[str, Any]:
        return {
            "span_id": self.span_id,
            "source_id": self.source_id,
            "raw_sha256": self.raw_sha256,
            "text_sha256": self.text_sha256,
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "raw_path": list(self.raw_path),
            "ancestor_path": list(self.ancestor_path),
            "heading": self.heading,
            "heading_path": list(self.heading_path),
            "voice_scope": self.voice_scope,
            "voice_basis": self.voice_basis,
            "matter_refs": list(self.matter_refs),
            "procedural_stage": self.procedural_stage,
            "procedural_stage_basis": self.procedural_stage_basis,
            "source_element": self.source_element,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class SourceStructureDocument:
    """Version-bound structural projection of one retained source."""

    schema_version: str
    parser_version: str
    status: str
    source_id: str
    raw_sha256: str
    text_sha256: str
    normalization_version: str | None
    spans: tuple[SourceStructureSpan, ...]
    coverage: Mapping[str, Any]
    error: str | None = None

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION or self.parser_version != PARSER_VERSION:
            raise SourceStructureError("unsupported structure schema/parser version")
        if self.status not in {STATUS_PARSED, STATUS_UNSUPPORTED, STATUS_INVALID, STATUS_UNAVAILABLE}:
            raise SourceStructureError(f"unknown structure status: {self.status}")
        if not self.source_id or len(self.raw_sha256) != 64 or len(self.text_sha256) != 64:
            raise SourceStructureError("structure document lacks source/version identity")
        starts = [span.start for span in self.spans]
        if starts != sorted(starts) or len({span.span_id for span in self.spans}) != len(self.spans):
            raise SourceStructureError("structure spans must be unique and ordered")
        if self.status == STATUS_PARSED and self.error:
            raise SourceStructureError("parsed structure cannot carry an error")
        object.__setattr__(self, "coverage", _owned_coverage(self.coverage))

    def to_wire(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "parser_version": self.parser_version,
            "status": self.status,
            "source_id": self.source_id,
            "raw_sha256": self.raw_sha256,
            "text_sha256": self.text_sha256,
            "normalization_version": self.normalization_version,
            "spans": [span.to_wire() for span in self.spans],
            "coverage": dict(self.coverage),
            "error": self.error,
        }


def _source_identity(source: Mapping[str, Any]) -> tuple[str, str, str, str, str, str | None]:
    source_id = str(source.get("source_id") or "")
    raw = source.get("raw_text")
    text = source.get("text")
    raw_sha256 = str(source.get("raw_sha256") or "")
    text_sha256 = str(source.get("text_sha256") or "")
    if not source_id or not isinstance(raw, str) or not isinstance(text, str) or not raw or not text:
        raise SourceStructureError("source structure needs source_id, raw_text and text")
    if _sha256(raw) != raw_sha256 or _sha256(text) != text_sha256:
        raise SourceStructureError("source structure source/version hash changed")
    return source_id, raw, text, raw_sha256, text_sha256, str(source.get("normalization_version") or "") or None


def _new_document(source: Mapping[str, Any], *, status: str, spans: tuple[SourceStructureSpan, ...] = (), error: str | None = None, coverage: Mapping[str, Any] | None = None) -> SourceStructureDocument:
    source_id, raw, text, raw_sha256, text_sha256, normalization = _source_identity(source)
    del raw, text
    return SourceStructureDocument(
        schema_version=SCHEMA_VERSION,
        parser_version=PARSER_VERSION,
        status=status,
        source_id=source_id,
        raw_sha256=raw_sha256,
        text_sha256=text_sha256,
        normalization_version=normalization,
        spans=spans,
        coverage=dict(coverage or {}),
        error=error,
    )


def parse_source_structure(source: Mapping[str, Any]) -> SourceStructureDocument:
    """Parse one retained source into an exact, source-version-bound projection."""

    source_id, raw, text, raw_sha256, text_sha256, normalization = _source_identity(source)
    content_format = str(source.get("content_format") or "")
    if content_format != SUPPORTED_FORMAT:
        return _new_document(
            source,
            status=STATUS_UNSUPPORTED,
            error=f"source content format is not {SUPPORTED_FORMAT}",
            coverage={"candidate_nodes": 0, "bound_spans": 0, "unbound_nodes": 0},
        )
    if normalization == CANONICAL_NORMALIZATION_VERSION:
        try:
            canonical_text = canonical_vaski_committee_text(raw)
        except SourceStructureError as exc:
            return _new_document(source, status=STATUS_INVALID, error=str(exc))
        if canonical_text != text:
            return _new_document(
                source,
                status=STATUS_INVALID,
                error="canonical Vaski text does not match the declared normalization version",
                coverage={"candidate_nodes": 0, "bound_spans": 0, "unbound_nodes": 0},
            )
    try:
        root = etree.fromstring(
            raw.encode("utf-8"),
            parser=etree.XMLParser(resolve_entities=False, no_network=True, load_dtd=False, recover=False),
        )
    except (UnicodeEncodeError, etree.XMLSyntaxError) as exc:
        return _new_document(source, status=STATUS_INVALID, error=f"invalid Vaski XML: {exc}")

    all_elements = list(root.iter())
    candidates = [element for element in all_elements if _local(element) in _CONTENT_NAMES and _element_text(element)]
    positions_by_text: dict[str, list[tuple[int, int]]] = {}
    for element in candidates:
        value = _element_text(element)
        positions_by_text.setdefault(value, _canonical_occurrences(text, value))

    occurrence_cursor: dict[str, int] = {}
    spans: list[SourceStructureSpan] = []
    unbound = 0
    whitespace_normalized = 0
    for element in candidates:
        value = _element_text(element)
        positions = positions_by_text.get(value, [])
        occurrence = occurrence_cursor.get(value, 0)
        if occurrence >= len(positions):
            unbound += 1
            continue
        start, end = positions[occurrence]
        occurrence_cursor[value] = occurrence + 1
        canonical_span_text = text[start:end]
        if canonical_span_text != value:
            whitespace_normalized += 1
        ancestors = tuple(element.iterancestors())[::-1] + (element,)
        headings = _heading_chain(ancestors)
        heading = _nearest_heading(ancestors[:-1])
        voice_scope, voice_basis = _voice(ancestors, headings)
        stage, stage_basis = _stage(ancestors, headings)
        refs = _matter_refs(value)
        raw_path = _path(element)
        span_seed = f"{source_id}:{raw_sha256}:{text_sha256}:{start}:{end}:{'/'.join(raw_path)}"
        spans.append(
            SourceStructureSpan(
                span_id="structure-span-" + _sha256(span_seed)[:24],
                source_id=source_id,
                raw_sha256=raw_sha256,
                text_sha256=text_sha256,
                start=start,
                end=end,
                text=canonical_span_text,
                raw_path=raw_path,
                ancestor_path=tuple(_local(item) for item in ancestors),
                heading=heading,
                heading_path=headings,
                voice_scope=voice_scope,
                voice_basis=voice_basis,
                matter_refs=refs,
                procedural_stage=stage,
                procedural_stage_basis=stage_basis,
                source_element=_local(element),
            )
        )
    spans.sort(key=lambda item: (item.start, item.end, item.raw_path))
    # Headings in Vaski are sibling elements rather than ancestors of the
    # paragraphs they introduce.  Reconstruct only this explicit structural
    # context: a heading applies to later nodes below its raw-tree parent until
    # another heading at the same parent scope appears.  This preserves the
    # distinction without treating proximity in plain text as authorship.
    heading_spans = [span for span in spans if span.source_element in _HEADING_NAMES]
    contextual: list[SourceStructureSpan] = []
    for span in spans:
        by_parent: dict[tuple[str, ...], SourceStructureSpan] = {}
        for heading_span in heading_spans:
            parent_path = heading_span.raw_path[:-1]
            if heading_span.start <= span.start and span.raw_path[: len(parent_path)] == parent_path:
                by_parent[parent_path] = heading_span
        active = sorted(by_parent.values(), key=lambda item: (len(item.raw_path), item.start))
        active_headings = tuple(item.text for item in active)
        ancestors = tuple(span.ancestor_path)
        has_explicit_quote = any(name in _QUOTED_ANCESTOR_NAMES for name in ancestors)
        if has_explicit_quote:
            voice_scope, voice_basis = VOICE_QUOTED, "EXPLICIT_QUOTE_ELEMENT"
        elif "JasenMielipideOsa" in ancestors or any("vastalause" in value.casefold() for value in active_headings):
            voice_scope, voice_basis = VOICE_DISSENT, "EXPLICIT_DISSENT_STRUCTURE"
        elif any("päätösehdotus" in value.casefold() for value in active_headings):
            voice_scope, voice_basis = VOICE_MAJORITY, "EXPLICIT_RECOMMENDATION_HEADING"
        elif "Mietinto" in ancestors:
            voice_scope, voice_basis = VOICE_COMMITTEE, "COMMITTEE_REPORT_ANCESTOR"
        else:
            voice_scope, voice_basis = VOICE_UNKNOWN, "UNRESOLVED"
        stage, stage_basis = _stage((), active_headings)
        contextual.append(
            replace(
                span,
                heading=active[-1].text if active else None,
                heading_path=active_headings,
                voice_scope=voice_scope,
                voice_basis=voice_basis,
                procedural_stage=stage,
                procedural_stage_basis=stage_basis,
            )
        )
    spans = contextual
    return SourceStructureDocument(
        schema_version=SCHEMA_VERSION,
        parser_version=PARSER_VERSION,
        status=STATUS_PARSED,
        source_id=source_id,
        raw_sha256=raw_sha256,
        text_sha256=text_sha256,
        normalization_version=normalization,
        spans=tuple(spans),
        coverage={
            "candidate_nodes": len(candidates),
            "bound_spans": len(spans),
            "unbound_nodes": unbound,
            "whitespace_normalized_spans": whitespace_normalized,
            "binding_mode": "canonical_whitespace_flexible_v1",
            "raw_format": content_format,
        },
    )


def source_structure_for_clips(
    source: Mapping[str, Any],
    clips: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return a bounded model-facing structure projection for exact clips."""

    source_id = str(source.get("source_id") or "")
    text = source.get("text")
    if not source_id or not isinstance(text, str) or not text:
        return {
            "schema_version": SCHEMA_VERSION,
            "parser_version": PARSER_VERSION,
            "status": STATUS_UNAVAILABLE,
            "source_id": source_id,
            "raw_sha256": source.get("raw_sha256"),
            "text_sha256": source.get("text_sha256"),
            "normalization_version": source.get("normalization_version"),
            "coverage": {"clip_count": 0, "spans_included": 0},
            "spans": [],
            "error": "source identity or canonical text is missing",
        }
    try:
        document = parse_source_structure(source)
    except SourceStructureError as exc:
        # A source can legitimately be available to the plain-text path while
        # lacking the retained raw/tree version needed for structural facts.
        # Keep that distinction explicit in the model input.
        return {
            "schema_version": SCHEMA_VERSION,
            "parser_version": PARSER_VERSION,
            "status": STATUS_UNAVAILABLE,
            "source_id": source_id,
            "raw_sha256": source.get("raw_sha256"),
            "text_sha256": source.get("text_sha256"),
            "normalization_version": source.get("normalization_version"),
            "coverage": {"clip_count": 0, "spans_included": 0},
            "spans": [],
            "error": str(exc),
        }
    intervals: list[tuple[int, int]] = []
    for clip in clips:
        start = clip.get("start")
        end = clip.get("end")
        if isinstance(start, int) and not isinstance(start, bool) and isinstance(end, int) and not isinstance(end, bool) and 0 <= start < end:
            intervals.append((start, end))
    selected = [
        span
        for span in document.spans
        if any(span.start < end and start < span.end for start, end in intervals)
    ]
    return {
        "schema_version": document.schema_version,
        "parser_version": document.parser_version,
        "status": document.status,
        "source_id": document.source_id,
        "raw_sha256": document.raw_sha256,
        "text_sha256": document.text_sha256,
        "normalization_version": document.normalization_version,
        "coverage": {
            **dict(document.coverage),
            "clip_count": len(intervals),
            "spans_included": len(selected),
        },
        "spans": [span.to_wire() for span in selected],
        "error": document.error,
    }


def source_structure_from_wire(value: Mapping[str, Any]) -> SourceStructureDocument:
    """Decode a retained structure record with strict field accounting."""

    if not isinstance(value, Mapping):
        raise SourceStructureError("structure wire value must be an object")
    allowed = {"schema_version", "parser_version", "status", "source_id", "raw_sha256", "text_sha256", "normalization_version", "spans", "coverage", "error"}
    unknown = set(value) - allowed
    if unknown:
        raise SourceStructureError(f"unknown structure fields: {sorted(unknown)}")
    span_values = value.get("spans")
    if not isinstance(span_values, list):
        raise SourceStructureError("structure spans must be a list")
    spans: list[SourceStructureSpan] = []
    span_allowed = set(SourceStructureSpan.__dataclass_fields__)
    for item in span_values:
        if not isinstance(item, Mapping):
            raise SourceStructureError("structure span must be an object")
        unknown_span = set(item) - span_allowed
        if unknown_span:
            raise SourceStructureError(f"unknown structure span fields: {sorted(unknown_span)}")
        kwargs = dict(item)
        for field in ("raw_path", "ancestor_path", "heading_path", "matter_refs"):
            if not isinstance(kwargs.get(field), list):
                raise SourceStructureError(f"structure span {field} must be a list")
            kwargs[field] = tuple(str(value) for value in kwargs[field])
        spans.append(SourceStructureSpan(**kwargs))
    coverage = value.get("coverage")
    if not isinstance(coverage, Mapping):
        raise SourceStructureError("structure coverage must be an object")
    return SourceStructureDocument(
        schema_version=str(value.get("schema_version") or ""),
        parser_version=str(value.get("parser_version") or ""),
        status=str(value.get("status") or ""),
        source_id=str(value.get("source_id") or ""),
        raw_sha256=str(value.get("raw_sha256") or ""),
        text_sha256=str(value.get("text_sha256") or ""),
        normalization_version=str(value.get("normalization_version") or "") or None,
        spans=tuple(spans),
        coverage=dict(coverage),
        error=str(value.get("error") or "") or None,
    )


__all__ = [
    "CANONICAL_NORMALIZATION_VERSION",
    "PARSER_VERSION",
    "SCHEMA_VERSION",
    "STATUS_INVALID",
    "STATUS_PARSED",
    "STATUS_UNAVAILABLE",
    "STATUS_UNSUPPORTED",
    "SourceStructureDocument",
    "SourceStructureError",
    "SourceStructureSpan",
    "canonical_vaski_committee_text",
    "parse_source_structure",
    "source_structure_for_clips",
    "source_structure_from_wire",
]

"""Cross-matter documentary route and component comparison from a source bank.

Deterministic producer: loads a ``source_question``-compatible bank, binds exact
source pointers, and emits route edges, signature roles, dated propositions and
component excerpts. Relation status distinguishes documentary explicit
references from candidate same-address overlap. Cross-reference alone never
proves substantive equivalence. Signature roles never imply drafting. Model
interpretation is out of scope here; optional ``source_question`` remains a
separate candidate answer producer for root-owned Muse runs.
"""

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Final
from urllib.parse import urlsplit

from paa.case_site import _document
from paa.source_pointers import SourcePointerIndex
from paa.source_question import SourceQuestionBank, SourceQuestionError, build_task_indexes

SCHEMA_VERSION: Final = "paa.route_packet.v2"
ADMISSION: Final = "DOCUMENTARY_PROJECTION_NOT_SEMANTIC_ACCEPTANCE"
CANDIDATE_ADMISSION: Final = "CANDIDATE_NOT_REFERENCE_JUDGED"

# Recognized parliamentary document prefixes only (not generic words like "Laki").
# Case-insensitive so "he 48/2024 vp" remains a formal matter id.
_FORMAL_PREFIXES: Final = (
    "HE|LA|KK|KKV|KAA|PeVL|YmVM|TyVM|HaVM|LaVM|LiVM|MmVM|StVM|SuVM|TuVM|UaVL|"
    "VaVM|PeVM|SiVM|Fv|EJ|"
    "HE|LA|KK|KKV|KAA|PEVL|YMVM|TYVM|HAVM|LAVM|LIVM|MMVM|STVM|SUVM|TUVM|UAVL|"
    "VAVM|PEVM|SIVM"
)
_FORMAL_MATTER = re.compile(
    r"(?<![A-Za-zÅÄÖåäö0-9])"
    rf"((?:{_FORMAL_PREFIXES})"
    r"\s*\d{1,6}\s*/\s*\d{4}(?:\s*vp)?)",
    re.IGNORECASE,
)
_STATUTE = re.compile(r"(?<![A-Za-z0-9])(\d{1,4}\s*/\s*\d{4})(?![A-Za-z0-9])")
_SECTION_ADDR = re.compile(
    r"(\d+\s*luvun\s+\d+\s*[a-z]?\s*§(?::n\s+\d+\s*momentti)?|\d+\s*[a-z]?\s*§(?::n\s+\d+\s*momentti)?)",
    re.IGNORECASE,
)
_VOIMAAN = re.compile(
    r"Tämä\s+laki\s+tulee\s+voimaan\s+"
    r"(\d{1,2})\s*päivänä\s+"
    r"(\w+)\s+"
    r"(\d{4})",
    re.IGNORECASE,
)
_SELLAISENA = re.compile(
    r"sellaisena\s+kuin\s+[^.]{0,160}?laissa\s+(\d{1,4}\s*/\s*\d{4})",
    re.IGNORECASE | re.DOTALL,
)
# Signature population ends at an explicit {END} marker when present; no silent
# line-count truncation of an otherwise complete block.
_HELSINKI_BLOCK = re.compile(
    r"Helsingissä\s+(\d{1,2}\.\d{1,2}\.\d{4})\s*\n"
    r"(.*?)(?:\{END\}|\Z)",
    re.IGNORECASE | re.DOTALL,
)
_PARTY_TOKEN = re.compile(r"^(sd|kok|kesk|ps|vihr|vas|r|kd|liik|lib)$", re.IGNORECASE)
_STATUTE_NEAR = re.compile(r"(\d{1,4}\s*/\s*\d{4})", re.IGNORECASE)
_CHAPTER_NEAR = re.compile(r"(\d+)\s*luvun", re.IGNORECASE)
_PARTY_LIKE = re.compile(r"^[a-zåäö]{1,10}$")
# One or more capitalized name tokens on a line ("Anna", "Anna Maria", "Anna-Kristiina").
_PERSON_LINE = re.compile(
    r"^[A-ZÅÄÖ][A-Za-zÅÄÖåäö-]+(?:\s+[A-ZÅÄÖ][A-Za-zÅÄÖåäö-]+)*$"
)
_DISPOSITION_REJECT = re.compile(r"\bHylätty\b")
_DISPOSITION_ADOPT = re.compile(r"\b(?:on\s+hyväksynyt|hyväksyy\s+muuttamattomana)\b", re.IGNORECASE)
_CORRECTION_PAIR = re.compile(
    r"Sivulla\s+[^\n]+\n\s*\.\.\.([^\n]+)\.\.\.\s*\n\s*Pitää\s+olla:\s*\n\s*\.\.\.([^\n]+)\.\.\.",
    re.IGNORECASE,
)
_MONTHS_FI: Final = {
    "tammikuuta": 1,
    "helmikuuta": 2,
    "maaliskuuta": 3,
    "huhtikuuta": 4,
    "toukokuuta": 5,
    "kesäkuuta": 6,
    "heinäkuuta": 7,
    "elokuuta": 8,
    "syyskuuta": 9,
    "lokakuuta": 10,
    "marraskuuta": 11,
    "joulukuuta": 12,
}
# As-of requires explicit CLI --as-of or an as-of word context — never the first
# bare date in the question (which may be an event/signature date).
_AS_OF_IN_QUESTION = re.compile(
    r"(?:voimassa|tarkastelupäivä(?:nä)?|tilanne|as-of|as\s+of)\s+"
    r"(\d{1,2})\.(\d{1,2})\.(\d{4})\b",
    re.IGNORECASE,
)
_UNSUPPORTED: Final = (
    {
        "code": "DRAFTING_UNKNOWN",
        "reason": "Official first/co-signature roles do not identify who drafted the text.",
    },
    {
        "code": "PRIVATE_INFLUENCE_UNSUPPORTED",
        "reason": "Public signature and procedural records do not establish private influence.",
    },
    {
        "code": "CAUSAL_CONTRIBUTION_UNSUPPORTED",
        "reason": "Documentary route and co-occurrence do not establish individual policy causation.",
    },
    {
        "code": "SUBSTANTIVE_EQUIVALENCE_NOT_PROVED_BY_CROSS_REFERENCE",
        "reason": "A law or matter reference link is not itself proof of whole substantive equivalence.",
    },
)


class RoutePacketError(ValueError):
    """Raised when a route packet cannot be built safely from the source bank."""


def _e(value: object) -> str:
    import html

    return html.escape(str(value if value is not None else ""), quote=True)


def _canon_matter(value: str) -> str:
    text = " ".join(value.split())
    text = re.sub(r"\s*/\s*", "/", text)
    text = re.sub(r"^([A-Za-zÅÄÖåäö]{1,12})(\d)", r"\1 \2", text)
    text = re.sub(r"\s+", " ", text)
    if not re.search(r"\bvp\b", text, re.IGNORECASE) and re.match(
        r"^[A-Za-zÅÄÖåäö]{1,12}\b", text, re.IGNORECASE
    ):
        text = f"{text} vp"
    return text


def _canon_statute(value: str) -> str:
    return re.sub(r"\s*/\s*", "/", " ".join(value.split()))


def _pointer_for_offset(index: SourcePointerIndex, start: int, end: int) -> dict[str, Any]:
    if end <= start:
        raise RoutePacketError("empty source span")
    first = last = None
    for span in index.spans:
        if span.end <= start:
            continue
        if span.start >= end:
            break
        if first is None:
            first = span
        last = span
    if first is None or last is None:
        raise RoutePacketError(
            f"offsets {start}:{end} are outside pointer index for {index.source_id}"
        )
    quote = index.source_text[start:end]
    if index.source_text[first.start:last.end].find(quote) < 0 and quote not in index.source_text[first.start:last.end]:
        # Prefer exact offsets; quote must equal the selected range when fully inside.
        pass
    exact_quote = index.source_text[start:end]
    if index.source_text[start:end] != exact_quote:
        raise RoutePacketError("quote/offset mismatch")
    return {
        "source_id": index.source_id,
        "source_text_sha256": index.source_text_sha256,
        "first_pointer": first.pointer_id,
        "last_pointer": last.pointer_id,
        "start": start,
        "end": end,
        "quote": exact_quote,
    }


def _excerpt(index: SourcePointerIndex, start: int, end: int, *, pad: int = 0) -> dict[str, Any]:
    lo = max(0, start - pad)
    hi = min(len(index.source_text), end + pad)
    return _pointer_for_offset(index, lo, hi)


def _parse_as_of(question: str, explicit: str | None) -> str | None:
    if explicit:
        date.fromisoformat(explicit)
        return explicit
    match = _AS_OF_IN_QUESTION.search(question or "")
    if not match:
        return None
    day, month, year = (int(match.group(1)), int(match.group(2)), int(match.group(3)))
    return date(year, month, day).isoformat()


def _fi_voimaan_date(day: str, month_word: str, year: str) -> str | None:
    month = _MONTHS_FI.get(month_word.casefold())
    if month is None:
        return None
    try:
        return date(int(year), month, int(day)).isoformat()
    except ValueError:
        return None


def _source_label(source: Mapping[str, Any]) -> str:
    return str(source.get("identifier") or source.get("title") or source.get("source_id") or "")


@dataclass(frozen=True, slots=True, kw_only=True)
class _BoundSource:
    source_id: str
    meta: dict[str, Any]
    index: SourcePointerIndex


def _bind_sources(bank: SourceQuestionBank, task_id: str) -> tuple[dict[str, _BoundSource], list[dict[str, Any]]]:
    from paa.source_bank_profile import normalize_source_meta

    plan = build_task_indexes(bank, task_id)
    bound: dict[str, _BoundSource] = {}
    residuals: list[dict[str, Any]] = list(plan.declared_gaps)
    for source_id, index in plan.indexes.items():
        meta = normalize_source_meta(dict(bank.sources[source_id]))
        declared = meta.get("text_sha256")
        if declared and declared != index.source_text_sha256:
            raise RoutePacketError(f"source {source_id} text hash mismatch at bind time")
        if meta.get("acquisition_status") not in {None, "SUCCESS", "OK", "PARTIAL"}:
            residuals.append(
                {
                    "code": "SOURCE_ACQUISITION_NOT_SUCCESS",
                    "severity": "SOURCE",
                    "source_id": source_id,
                    "reason": f"acquisition_status={meta.get('acquisition_status')!r}",
                }
            )
        bound[source_id] = _BoundSource(source_id=source_id, meta=meta, index=index)
    for source_id in bank.task_sources.get(task_id, []):
        if source_id not in bound:
            residuals.append(
                {
                    "code": "TASK_SOURCE_NOT_INDEXED",
                    "severity": "SOURCE",
                    "source_id": source_id,
                    "reason": "Listed in task_sources but not present in the pointer index.",
                }
            )
    return bound, residuals


def _extract_signatories(bound: Mapping[str, _BoundSource]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in bound.values():
        if str(item.meta.get("source_type") or "") != "LEGISLATIVE_INITIATIVE":
            continue
        text = item.index.source_text
        match = _HELSINKI_BLOCK.search(text)
        if not match:
            rows.append(
                {
                    "source_id": item.source_id,
                    "state": "UNRESOLVED",
                    "reason": "No Helsinki signature block found in initiative text.",
                    "drafting": {"state": "UNRESOLVED", "value": None},
                }
            )
            continue
        block = match.group(2)
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        names: list[tuple[str, int, int]] = []
        i = 0
        block_start = match.start(2)
        cursor = block_start

        def _is_party_line(value: str) -> bool:
            return bool(_PARTY_TOKEN.match(value) or _PARTY_LIKE.fullmatch(value))

        while i < len(lines):
            line = lines[i]
            if line.startswith(("{", "Julkaisu", "sftp:")):
                break
            if _is_party_line(line):
                i += 1
                continue
            # Name may span multiple given-name lines then family, then party:
            # "Anna Maria\nNieminen\nsd". Do not promote a later name to first
            # when an earlier multiword given name was present but unparsed.
            if _PERSON_LINE.match(line) and not _is_party_line(line):
                parts: list[str] = []
                j = i
                while (
                    j < len(lines)
                    and _PERSON_LINE.match(lines[j])
                    and not _is_party_line(lines[j])
                    and not re.search(r"\d", lines[j])
                    and not lines[j].endswith(":")
                ):
                    parts.append(lines[j])
                    j += 1
                if len(parts) >= 2 and (j >= len(lines) or _is_party_line(lines[j])):
                    given = " ".join(parts[:-1])
                    family = parts[-1]
                    full = f"{given} {family}"
                    local = text.find(parts[0], cursor)
                    if local < 0:
                        i = j
                        continue
                    end = text.find(family, local)
                    if end < 0:
                        i = j
                        continue
                    end = end + len(family)
                    names.append((full, local, end))
                    cursor = end
                    i = j
                    if i < len(lines) and _is_party_line(lines[i]):
                        i += 1
                    continue
            i += 1
        if not names:
            rows.append(
                {
                    "source_id": item.source_id,
                    "state": "UNRESOLVED",
                    "reason": "Signature block present but no person lines parsed.",
                    "drafting": {"state": "UNRESOLVED", "value": None},
                }
            )
            continue
        for rank, (name, start, end) in enumerate(names):
            role = "FIRST_SIGNATORY" if rank == 0 else "CO_SIGNATORY"
            rows.append(
                {
                    "actor_name": name,
                    "role": role,
                    "role_label": "FIRST_SIGNATURE" if rank == 0 else "CO_SIGNATURE",
                    "action_kind": {
                        "value": "FIRST_SIGNATORY" if rank == 0 else "INITIATIVE_COSIGNED",
                        "basis": "SIGNATURE_BLOCK_ORDER",
                    },
                    "drafting": {
                        "state": "UNRESOLVED",
                        "value": None,
                        "limitation": "Signature order is not drafting authorship.",
                    },
                    "source": _pointer_for_offset(item.index, start, end),
                    "matter_identifier": _source_label(item.meta),
                    "source_date": item.meta.get("source_date"),
                }
            )
    return rows


def _formal_matter_spans(text: str) -> list[tuple[int, int]]:
    return [(match.start(), match.end()) for match in _FORMAL_MATTER.finditer(text)]


def _inside_spans(offset: int, spans: Sequence[tuple[int, int]]) -> bool:
    return any(start <= offset < end for start, end in spans)


def _commencement_role(meta: Mapping[str, Any]) -> tuple[str, str]:
    """Own commencement only for enacted carriers; proposals stay cues."""

    stage = str(meta.get("stage") or "")
    source_type = str(meta.get("source_type") or "")
    if stage == "ENACTED":
        return "OWN_COMMENCEMENT", "ENACTED_CARRIER"
    if stage == "CONSOLIDATED_CAPTURE":
        return "CONSOLIDATED_CAPTURE_COMMENCEMENT_CUE", "CONSOLIDATED_CAPTURE_CARRIER"
    if stage in {"PROPOSAL", "EXPLANATORY_PROPOSAL"} or source_type in {
        "GOVERNMENT_PROPOSAL",
        "LEGISLATIVE_INITIATIVE",
    }:
        return "PROPOSED_COMMENCEMENT_CUE", "PROPOSAL_OR_INITIATIVE_CARRIER"
    if stage == "ADOPTED_LEGISLATIVE_TEXT" or source_type == "PARLIAMENT_REPLY":
        return "ADOPTED_TEXT_COMMENCEMENT_CUE", "PARLIAMENT_REPLY_CARRIER"
    return "COMMENCEMENT_CUE_UNTYPED", "UNTYPED_CARRIER"


def _extract_references(bound: Mapping[str, _BoundSource]) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, int]] = set()
    for item in bound.values():
        text = item.index.source_text
        matter_spans = _formal_matter_spans(text)
        own_ids = {
            _canon_matter(str(item.meta.get("identifier") or "")),
            _canon_statute(str(item.meta.get("identifier") or "")),
        }
        own_ids.discard("")
        for match in _FORMAL_MATTER.finditer(text):
            target = _canon_matter(match.group(1))
            own_identifier = str(item.meta.get("identifier") or "").casefold()
            if (
                (target in own_ids or _canon_matter(target.replace(" vp", "")) in own_ids)
                and match.start() < 400
                and target.split()[0].casefold() in own_identifier
            ):
                continue
            key = (item.source_id, "FORMAL_MATTER", target, match.start())
            if key in seen:
                continue
            seen.add(key)
            edges.append(
                {
                    "edge_id": f"{item.source_id}:matter:{match.start()}",
                    "from_source_id": item.source_id,
                    "from_identifier": _source_label(item.meta),
                    "from_stage": item.meta.get("stage"),
                    "relation_status": "DOCUMENTARY_EXPLICIT_REFERENCE",
                    "reference_kind": "FORMAL_MATTER_ID",
                    "target_label": target,
                    "equivalence_claim": "NOT_CLAIMED",
                    "source": _excerpt(item.index, match.start(), match.end()),
                }
            )
        for match in _STATUTE.finditer(text):
            if _inside_spans(match.start(), matter_spans):
                continue
            target = _canon_statute(match.group(1))
            if target in own_ids:
                continue
            # Bare number/year (including after "Laki …") is a statute citation,
            # not a formal matter id — formal prefixes are handled above.
            key = (item.source_id, "STATUTE", target, match.start())
            if key in seen:
                continue
            seen.add(key)
            edges.append(
                {
                    "edge_id": f"{item.source_id}:statute:{match.start()}",
                    "from_source_id": item.source_id,
                    "from_identifier": _source_label(item.meta),
                    "from_stage": item.meta.get("stage"),
                    "relation_status": "DOCUMENTARY_EXPLICIT_REFERENCE",
                    "reference_kind": "STATUTE_NUMBER",
                    "target_label": target,
                    "equivalence_claim": "NOT_CLAIMED",
                    "source": _excerpt(item.index, match.start(), match.end()),
                }
            )
    edges.sort(key=lambda row: (row["from_source_id"], row["source"]["start"], row["target_label"]))
    return edges


def _extract_dispositions(bound: Mapping[str, _BoundSource]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in bound.values():
        text = item.index.source_text
        for match in _DISPOSITION_REJECT.finditer(text):
            rows.append(
                {
                    "cue_kind": "REJECTED_WORD_CUE",
                    "status": "LEXICAL_CUE_NOT_DECISION",
                    "basis": "SOURCE_DISPOSITION_WORD",
                    "source": _excerpt(item.index, match.start(), match.end(), pad=80),
                    "source_id": item.source_id,
                    "identifier": _source_label(item.meta),
                    "stage": item.meta.get("stage"),
                    "note": "Lexical rejection word only; not an adjudicated institutional decision.",
                }
            )
        for match in _DISPOSITION_ADOPT.finditer(text):
            rows.append(
                {
                    "cue_kind": "ADOPTION_OR_RECOMMENDATION_WORD_CUE",
                    "status": "LEXICAL_CUE_NOT_DECISION",
                    "basis": "SOURCE_DISPOSITION_WORD",
                    "source": _excerpt(item.index, match.start(), match.end(), pad=80),
                    "source_id": item.source_id,
                    "identifier": _source_label(item.meta),
                    "stage": item.meta.get("stage"),
                    "note": "Lexical adoption/recommendation word only; not a final decision finding.",
                }
            )
    return rows


def _span_bound_commencement_carriers(meta: Mapping[str, Any]) -> dict[int, str]:
    """Return pointer/start→carrier map when structure binds carriers per span."""

    structure = meta.get("source_structure")
    if not isinstance(structure, Mapping):
        return {}
    out: dict[int, str] = {}
    for key in ("commencement_carriers", "date_carriers", "commencements"):
        rows = structure.get(key)
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            carrier = row.get("carrier") or row.get("commencement_carrier")
            if not isinstance(carrier, str) or not carrier.strip():
                continue
            for offset_key in ("start", "first_pointer", "pointer"):
                if row.get(offset_key) is None:
                    continue
                try:
                    out[int(row[offset_key])] = carrier.strip()
                except (TypeError, ValueError):
                    continue
    return out


def _extract_dates(bound: Mapping[str, _BoundSource]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in bound.values():
        text = item.index.source_text
        voimaan_matches = list(_VOIMAAN.finditer(text))
        span_carriers = _span_bound_commencement_carriers(item.meta)
        # Multiple "Tämä laki tulee voimaan" phrases cannot all be OWN unless
        # each match has its own span-bound carrier. A whole-source carrier
        # string (e.g. underlying_target) must not defeat the ambiguity guard.
        for match in voimaan_matches:
            iso = _fi_voimaan_date(match.group(1), match.group(2), match.group(3))
            span_carrier = span_carriers.get(match.start())
            multi_unresolved = len(voimaan_matches) > 1 and span_carrier is None
            if multi_unresolved:
                date_role, carrier = "CARRIER_UNRESOLVED", "UNSTRUCTURED_MULTI_COMMENCEMENT"
                state = "CARRIER_UNRESOLVED"
            else:
                date_role, carrier = _commencement_role(item.meta)
                if span_carrier:
                    carrier = span_carrier
                state = "SOURCE_EXPLICIT" if iso else "UNPARSED"
            rows.append(
                {
                    "date_role": date_role,
                    "carrier_basis": carrier,
                    "value": iso,
                    "state": state,
                    "source": _excerpt(item.index, match.start(), match.end()),
                    "source_id": item.source_id,
                    "identifier": _source_label(item.meta),
                    "stage": item.meta.get("stage"),
                    "source_type": item.meta.get("source_type"),
                    "note": (
                        "Multiple commencement phrases without per-span source carrier; "
                        "not promoted to OWN_COMMENCEMENT."
                        if multi_unresolved
                        else None
                    ),
                }
            )
        for match in _SELLAISENA.finditer(text):
            target = _canon_statute(match.group(1))
            rows.append(
                {
                    "date_role": "UNDERLYING_TARGET_AMENDMENT_REFERENCE",
                    "value": None,
                    "target_statute": target,
                    "state": "SOURCE_EXPLICIT_REFERENCE",
                    "source": _excerpt(item.index, match.start(), match.end()),
                    "source_id": item.source_id,
                    "identifier": _source_label(item.meta),
                    "stage": item.meta.get("stage"),
                    "note": "Names a prior amended text; not this act's own commencement date.",
                }
            )
        if str(item.meta.get("stage") or "") == "CORRECTION" or str(item.meta.get("source_type") or "") == "OFFICIAL_CORRECTION":
            for match in _CORRECTION_PAIR.finditer(text):
                rows.append(
                    {
                        "date_role": "OFFICIAL_CORRECTION",
                        "value": item.meta.get("source_date"),
                        "state": "SOURCE_EXPLICIT",
                        "correction_kind": "TYPO_OR_WORDING_CORRECTION",
                        "was": match.group(1).strip(),
                        "should_be": match.group(2).strip(),
                        "source": _excerpt(item.index, match.start(), match.end()),
                        "source_id": item.source_id,
                        "identifier": _source_label(item.meta),
                        "stage": item.meta.get("stage"),
                        "note": "Official correction evidence; not automatically a substantive revision.",
                    }
                )
            if not any(row.get("source_id") == item.source_id and row.get("date_role") == "OFFICIAL_CORRECTION" for row in rows):
                # Still record the correction document presence with a short quote.
                snippet_end = min(len(text), 220)
                rows.append(
                    {
                        "date_role": "OFFICIAL_CORRECTION",
                        "value": item.meta.get("source_date"),
                        "state": "SOURCE_DOCUMENT",
                        "correction_kind": "CORRECTION_DOCUMENT_PRESENT",
                        "source": _pointer_for_offset(item.index, 0, snippet_end) if snippet_end else None,
                        "source_id": item.source_id,
                        "identifier": _source_label(item.meta),
                        "stage": item.meta.get("stage"),
                    }
                )
        if item.meta.get("source_date"):
            rows.append(
                {
                    "date_role": "SOURCE_EVENT_OR_PUBLICATION_DATE",
                    "value": item.meta.get("source_date"),
                    "state": "SOURCE_METADATA",
                    "source_id": item.source_id,
                    "identifier": _source_label(item.meta),
                    "stage": item.meta.get("stage"),
                    "basis": "source_date field on retained bank row",
                }
            )
    return rows


def _normalize_section_surface(raw: str) -> str:
    text = " ".join(raw.casefold().split())
    return text.replace("§:n", "§")


def _resolve_component_address(
    text: str,
    match: re.Match[str],
    meta: Mapping[str, Any],
) -> dict[str, Any]:
    """Attach statute/chapter context; unqualified § across laws stays unresolved."""

    surface = _normalize_section_surface(match.group(1))
    window = text[max(0, match.start() - 220) : match.start()]
    chapter_match = list(_CHAPTER_NEAR.finditer(window))
    chapter = chapter_match[-1].group(1) if chapter_match else None
    statute_match = list(_STATUTE_NEAR.finditer(window))
    # Prefer nearby statute not embedded in a formal matter id.
    statute = None
    for cand in reversed(statute_match):
        abs_start = max(0, match.start() - 220) + cand.start()
        if _inside_spans(abs_start, _formal_matter_spans(text)):
            continue
        statute = _canon_statute(cand.group(1))
        break
    source_type = str(meta.get("source_type") or "")
    identifier = _source_label(meta)
    if (
        statute is None
        and source_type in {"AMENDING_ACT", "CONSOLIDATED_ACT", "OFFICIAL_CORRECTION"}
        and re.fullmatch(r"\d{1,4}/\d{4}(?:\s+correction)?", identifier.strip(), re.IGNORECASE)
    ):
        statute = _canon_statute(identifier.split()[0])
    if statute is None:
        return {
            "component_key": f"unresolved:{identifier}:{surface}",
            "component_key_state": "UNRESOLVED_ADDRESS_CANDIDATE",
            "statute": None,
            "chapter": chapter,
            "section_surface": surface,
            "relation_status": "COMPONENT_EXCERPT",
        }
    # Surface may already carry "N luvun … §"; do not prepend chapter twice.
    if re.search(r"\bluvun\b", surface):
        key_surface = surface
    elif chapter:
        key_surface = f"{chapter} luvun {surface}"
    else:
        key_surface = surface
    key = f"{statute}|{key_surface}".strip()
    return {
        "component_key": key,
        "component_key_state": "STATUTE_BOUND_ADDRESS",
        "statute": statute,
        "chapter": chapter,
        "section_surface": surface,
        "relation_status": "COMPONENT_EXCERPT",
    }


def _extract_components(bound: Mapping[str, _BoundSource]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in bound.values():
        text = item.index.source_text
        for match in _SECTION_ADDR.finditer(text):
            start = match.start()
            end = min(len(text), match.end() + 280)
            nl = text.find("\n\n", match.end(), end)
            if nl != -1:
                end = nl
            address = _resolve_component_address(text, match, item.meta)
            rows.append(
                {
                    **address,
                    "surface": match.group(1),
                    "source_id": item.source_id,
                    "identifier": _source_label(item.meta),
                    "stage": item.meta.get("stage"),
                    "source_type": item.meta.get("source_type"),
                    "source": _pointer_for_offset(item.index, start, end),
                    "equivalence_claim": "NOT_CLAIMED",
                }
            )
    return rows


def _compare_components(components: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[str, list[Mapping[str, Any]]] = {}
    for row in components:
        if row.get("component_key_state") != "STATUTE_BOUND_ADDRESS":
            continue
        by_key.setdefault(str(row["component_key"]), []).append(row)
    comparisons: list[dict[str, Any]] = []
    for key, rows in sorted(by_key.items()):
        sources = sorted({str(row["source_id"]) for row in rows})
        if len(sources) < 2:
            continue
        quotes = [(str(row["source_id"]), str(row["source"]["quote"])) for row in rows]
        identical = len({quote for _sid, quote in quotes}) == 1
        shown = list(rows[:6])
        comparisons.append(
            {
                "component_key": key,
                "source_ids": sources,
                "relation_status": "CANDIDATE_SAME_ADDRESS_OVERLAP",
                "equivalence_claim": "NOT_CLAIMED",
                "text_identity": "IDENTICAL_EXCERPT" if identical else "DIFFERENT_OR_PARTIAL_EXCERPT",
                "excerpts_shown": len(shown),
                "excerpts_total": len(rows),
                "excerpts": [
                    {
                        "source_id": row["source_id"],
                        "identifier": row["identifier"],
                        "stage": row["stage"],
                        "source": row["source"],
                    }
                    for row in shown
                ],
                "note": (
                    "Same statute-bound address across sources is a candidate join key, "
                    "not proof that conditions are substantively the same."
                ),
            }
        )
    return comparisons


def _applicability(
    *,
    as_of: str | None,
    dates: Sequence[Mapping[str, Any]],
    bound: Mapping[str, _BoundSource],
    residuals: list[dict[str, Any]],
) -> dict[str, Any]:
    if not as_of:
        return {
            "state": "AS_OF_NOT_SUPPLIED",
            "value": None,
            "reason": "No explicit as-of date in the question or CLI; operative-law claim withheld.",
        }
    as_of_date = date.fromisoformat(as_of)
    commencements = [
        row
        for row in dates
        if row.get("date_role") == "OWN_COMMENCEMENT" and row.get("value")
    ]
    enacted_sources = [
        item
        for item in bound.values()
        if str(item.meta.get("stage") or "") in {"ENACTED", "CONSOLIDATED_CAPTURE"}
        or str(item.meta.get("source_type") or "") in {"AMENDING_ACT", "CONSOLIDATED_ACT"}
    ]
    if not enacted_sources:
        residuals.append(
            {
                "code": "ENACTED_OR_CONSOLIDATED_SOURCE_MISSING",
                "severity": "APPLICABILITY",
                "reason": "As-of applicability depends on enacted/consolidated text absent from the task source set.",
            }
        )
        return {
            "state": "UNRESOLVED",
            "as_of": as_of,
            "reason": "Dependent conclusion requires enacted or consolidated legal text in the task sources.",
            "missingness": "ENACTED_OR_CONSOLIDATED_SOURCE_MISSING",
        }
    commenced = []
    for row in commencements:
        try:
            if date.fromisoformat(str(row["value"])) <= as_of_date:
                commenced.append(row)
        except ValueError:
            continue
    corrections = [row for row in dates if row.get("date_role") == "OFFICIAL_CORRECTION"]
    return {
        "state": "SOURCE_BOUNDED",
        "as_of": as_of,
        "commencements_on_or_before_as_of": commenced,
        "commencements_all": commencements,
        "corrections": corrections,
        "note": (
            "Lists source-backed commencement and correction propositions relevant to as-of. "
            "Does not certify a complete operative consolidation."
        ),
    }


def _route_summary(
    *,
    bound: Mapping[str, _BoundSource],
    edges: Sequence[Mapping[str, Any]],
    dispositions: Sequence[Mapping[str, Any]],
    signatories: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    stages = []
    for item in bound.values():
        stages.append(
            {
                "source_id": item.source_id,
                "identifier": _source_label(item.meta),
                "stage": item.meta.get("stage"),
                "source_type": item.meta.get("source_type"),
                "source_voice": item.meta.get("source_voice"),
                "source_date": item.meta.get("source_date"),
                "url": item.meta.get("url"),
                "record_url": item.meta.get("record_url"),
                "record_url_kind": item.meta.get("record_url_kind"),
            }
        )
    rejected_cues = [row for row in dispositions if row.get("cue_kind") == "REJECTED_WORD_CUE"]
    adoption_cues = [
        row for row in dispositions if row.get("cue_kind") == "ADOPTION_OR_RECOMMENDATION_WORD_CUE"
    ]
    return {
        "stages": stages,
        "documentary_edge_count": sum(
            1 for edge in edges if edge.get("relation_status") == "DOCUMENTARY_EXPLICIT_REFERENCE"
        ),
        "rejection_word_cue_count": len(rejected_cues),
        "adoption_word_cue_count": len(adoption_cues),
        "disposition_cues_are_not_decisions": True,
        "signatory_count": sum(1 for row in signatories if row.get("actor_name")),
        "first_signatories": [
            row["actor_name"] for row in signatories if row.get("role") == "FIRST_SIGNATORY"
        ],
        "co_signatories": [
            row["actor_name"] for row in signatories if row.get("role") == "CO_SIGNATORY"
        ],
    }


def _bank_identity(bank: SourceQuestionBank, task_id: str, bound: Mapping[str, _BoundSource]) -> dict[str, Any]:
    source_digests = {
        source_id: item.index.source_text_sha256 for source_id, item in sorted(bound.items())
    }
    return {
        "questions_file": Path(bank.questions_path).name,
        "sources_file": Path(bank.sources_path).name,
        "questions_sha256": hashlib.sha256(Path(bank.questions_path).read_bytes()).hexdigest(),
        "sources_sha256": hashlib.sha256(Path(bank.sources_path).read_bytes()).hexdigest(),
        "task_source_ids": list(bank.task_sources.get(task_id, [])),
        "indexed_source_ids": sorted(bound),
        "source_text_sha256": source_digests,
    }


def _revalidate_bound_evidence_row(
    row: Mapping[str, Any],
    indexes: Mapping[str, Any],
    *,
    label: str,
    errors: list[str],
) -> dict[str, Any] | None:
    """Re-check quote/offsets/pointers/hash against live task indexes (not semantics)."""

    from paa.source_pointers import SourcePointerError

    source_id = str(row.get("source_id") or "")
    index = indexes.get(source_id)
    if index is None:
        errors.append(f"{label}: unknown or unindexed source_id {source_id}")
        return None
    declared_hash = str(row.get("source_text_sha256") or "")
    if declared_hash != index.source_text_sha256:
        errors.append(f"{label}: source_text_sha256 mismatch for {source_id}")
        return None
    coordinate_keys = ("first_pointer", "last_pointer", "start", "end")
    if any(type(row.get(key)) is not int for key in coordinate_keys):
        errors.append(f"{label}: pointer/offset fields must be exact integers")
        return None
    first, last, start, end = (row[key] for key in coordinate_keys)
    try:
        anchor = index.resolve(
            source_text_sha256=index.source_text_sha256,
            span_start=first,
            span_end=last,
        )
    except (SourcePointerError, TypeError, ValueError) as error:
        errors.append(f"{label}: pointer resolve failed ({error})")
        return None
    if not (0 <= start < end <= len(index.source_text)):
        errors.append(f"{label}: character offsets out of source range")
        return None
    exact_quote = index.source_text[start:end]
    if str(row.get("quote") or "") != exact_quote:
        errors.append(f"{label}: quote does not match source text at offsets")
        return None
    if not (anchor.start <= start < end <= anchor.end):
        errors.append(f"{label}: offsets outside resolved pointer span")
        return None
    return {
        "source_id": source_id,
        "quote": exact_quote,
        "start": start,
        "end": end,
        "first_pointer": first,
        "last_pointer": last,
        "source_text_sha256": index.source_text_sha256,
    }


def bind_answer_packet(
    route_packet: Mapping[str, Any],
    answer_packet: Mapping[str, Any],
    bank: SourceQuestionBank,
) -> dict[str, Any]:
    """Attach a source_question packet after verifying universe + live evidence rows.

    Copies are not trusted: every ``evidence_bound`` / map-finding quote, offset,
    pointer and source hash is revalidated against ``build_task_indexes``. Quote
    validation is mechanical identity, not semantic judgment.
    """

    errors: list[str] = []
    task_id = str(route_packet.get("task_id") or "")
    if str(answer_packet.get("task_id") or "") != task_id:
        errors.append("task_id mismatch")
    if str(answer_packet.get("question") or "").strip() != str(route_packet.get("question") or "").strip():
        errors.append("question text mismatch")
    route_matter = str(route_packet.get("matter_id") or "")
    answer_matter = str(answer_packet.get("matter_id") or "")
    if route_matter and answer_matter and route_matter != answer_matter:
        errors.append("matter_id mismatch")

    try:
        plan = build_task_indexes(bank, task_id)
    except SourceQuestionError as error:
        errors.append(f"task index failed: {error}")
        plan = None
    indexes = plan.indexes if plan is not None else {}
    expected_universe = list(bank.task_sources.get(task_id) or [])
    coverage = answer_packet.get("coverage") if isinstance(answer_packet.get("coverage"), Mapping) else {}
    requested = [str(item) for item in (coverage.get("requested_source_ids") or [])]
    if not requested:
        errors.append("missing coverage.requested_source_ids (explicit source universe required)")
    elif sorted(requested) != sorted(expected_universe):
        errors.append("source universe (task_sources) mismatch")
    domains = coverage.get("pointer_domains") if isinstance(coverage.get("pointer_domains"), Mapping) else {}
    if requested and not domains:
        errors.append("missing coverage.pointer_domains for source universe")
    for source_id in expected_universe:
        index = indexes.get(source_id)
        if index is None:
            errors.append(f"task source {source_id} failed to index")
            continue
        domain = domains.get(source_id) if isinstance(domains, Mapping) else None
        if not isinstance(domain, Mapping):
            errors.append(f"missing pointer_domains entry for {source_id}")
            continue
        got = domain.get("source_text_sha256")
        if got != index.source_text_sha256:
            errors.append(f"source_text_sha256 mismatch for {source_id}")
    for source_id in domains or {}:
        if str(source_id) not in expected_universe:
            errors.append(f"answer source {source_id} not in task source universe")

    claims: list[dict[str, Any]] = []
    for claim_index, claim in enumerate(answer_packet.get("claims") or []):
        if not isinstance(claim, Mapping):
            errors.append(f"claim-{claim_index}: not an object")
            continue
        unbound = bool(claim.get("unbound"))
        evidence_rows = claim.get("evidence_bound") or []
        bound_rows: list[dict[str, Any]] = []
        if unbound and not evidence_rows:
            claims.append(
                {
                    "text": claim.get("text"),
                    "unbound": True,
                    "epistemic_status": claim.get("epistemic_status"),
                    "evidence_bound": [],
                }
            )
            continue
        for evidence_index, evidence in enumerate(evidence_rows):
            if not isinstance(evidence, Mapping):
                errors.append(f"claim-{claim_index}:evidence-{evidence_index}: not an object")
                continue
            revalidated = _revalidate_bound_evidence_row(
                evidence,
                indexes,
                label=f"claim-{claim_index}:evidence-{evidence_index}",
                errors=errors,
            )
            if revalidated is not None:
                bound_rows.append(revalidated)
        if not unbound and not bound_rows and evidence_rows:
            # All rows failed revalidation; error list already populated.
            pass
        claims.append(
            {
                "text": claim.get("text"),
                "unbound": unbound or not bound_rows,
                "epistemic_status": claim.get("epistemic_status"),
                "evidence_bound": bound_rows,
            }
        )

    map_findings: list[dict[str, Any]] = []
    for finding_index, row in enumerate(answer_packet.get("map_findings") or []):
        if not isinstance(row, Mapping):
            errors.append(f"map_finding-{finding_index}: not an object")
            continue
        # Map findings carry evidence fields at the top level.
        if row.get("source_id") is not None or row.get("quote") is not None:
            revalidated = _revalidate_bound_evidence_row(
                row,
                indexes,
                label=f"map_finding-{finding_index}",
                errors=errors,
            )
            if revalidated is None:
                continue
            map_findings.append(
                {
                    **revalidated,
                    "text": row.get("text"),
                    "epistemic_status": row.get("epistemic_status"),
                    "event_date": row.get("event_date"),
                }
            )
        else:
            map_findings.append(
                {
                    "text": row.get("text"),
                    "epistemic_status": row.get("epistemic_status"),
                    "event_date": row.get("event_date"),
                }
            )

    binding = {
        "state": "BOUND" if not errors else "REJECTED",
        "errors": errors,
        "answer_schema_version": answer_packet.get("schema_version"),
        "answer_status": answer_packet.get("status"),
        "answer_admission": answer_packet.get("admission") or CANDIDATE_ADMISSION,
        "provider": answer_packet.get("provider"),
        "model_id": answer_packet.get("model_id"),
        "evidence_revalidated_against_task_indexes": True,
    }
    if errors:
        return {
            "binding": binding,
            "candidate_answer": None,
            "note": "Answer packet not attached; binding failed.",
        }
    return {
        "binding": binding,
        "candidate_answer": {
            "admission": answer_packet.get("admission") or CANDIDATE_ADMISSION,
            "status": answer_packet.get("status"),
            "answer": answer_packet.get("answer"),
            "claims": claims,
            "map_findings": map_findings,
            "unresolved": list(answer_packet.get("unresolved") or []),
            "next_evidence": list(answer_packet.get("next_evidence") or []),
            "error": answer_packet.get("error"),
            "read_completeness": answer_packet.get("read_completeness")
            or coverage.get("read_completeness"),
            "coverage": {
                "requested_source_ids": requested,
                "fully_read_source_ids": list(coverage.get("fully_read_source_ids") or []),
                "partial_read_source_ids": list(coverage.get("partial_read_source_ids") or []),
                "unread_source_ids": list(coverage.get("unread_source_ids") or []),
                "pointer_domains": {
                    source_id: {
                        "source_text_sha256": indexes[source_id].source_text_sha256,
                        "min_pointer": 0,
                        "max_pointer": len(indexes[source_id].spans) - 1,
                    }
                    for source_id in requested
                    if source_id in indexes
                },
            },
            "task_id": answer_packet.get("task_id"),
            "matter_id": answer_packet.get("matter_id"),
            "question": answer_packet.get("question"),
        },
        "note": "Derived source-bound research answer remains candidate pending review.",
    }


def build_route_packet(
    bank: SourceQuestionBank,
    task_id: str,
    *,
    as_of: str | None = None,
    answer_packet: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one deterministic route/component packet for a bank task."""

    task = next((dict(item) for item in bank.tasks if item.get("task_id") == task_id), None)
    if task is None:
        raise RoutePacketError(f"unknown task_id: {task_id}")
    bank._require_task_source_mapping(task_id)
    bound, residuals = _bind_sources(bank, task_id)
    as_of_value = _parse_as_of(str(task.get("question") or ""), as_of)
    signatories = _extract_signatories(bound)
    edges = _extract_references(bound)
    dispositions = _extract_dispositions(bound)
    dates = _extract_dates(bound)
    components = _extract_components(bound)
    comparisons = _compare_components(components)
    applicability = _applicability(as_of=as_of_value, dates=dates, bound=bound, residuals=residuals)
    summary = _route_summary(
        bound=bound, edges=edges, dispositions=dispositions, signatories=signatories
    )
    has_reject_cue = any(row.get("cue_kind") == "REJECTED_WORD_CUE" for row in dispositions)
    has_enacted = any(
        str(item.meta.get("stage") or "") == "ENACTED"
        or str(item.meta.get("source_type") or "") == "AMENDING_ACT"
        for item in bound.values()
    )
    if has_reject_cue and not has_enacted:
        residuals.append(
            {
                "code": "ALTERNATIVE_ENACTED_ROUTE_SOURCE_MISSING",
                "severity": "ROUTE",
                "reason": (
                    "Lexical rejection cues are present, but no enacted amending-act source is in the "
                    "task set; alternative-route conclusions stay unresolved."
                ),
            }
        )
    bank_id = _bank_identity(bank, task_id, bound)
    packet: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "admission": ADMISSION,
        "task_id": task_id,
        "matter_id": task.get("matter_id"),
        "question": task.get("question"),
        "split": task.get("split"),
        "as_of": as_of_value,
        "bank": bank_id,
        "summary": summary,
        "actors": signatories,
        "route_edges": edges,
        "route_edges_total": len(edges),
        "disposition_cues": dispositions,
        "disposition_cues_total": len(dispositions),
        "dates": dates,
        "components": components,
        "component_comparisons": comparisons,
        "component_comparisons_total": len(comparisons),
        "applicability": applicability,
        "unsupported": [dict(item) for item in _UNSUPPORTED],
        "residuals": residuals,
        "projection": {
            "kind": "DOCUMENTARY_INDEX",
            "not_the_answer": True,
            "note": "Documentary index is secondary evidence; the candidate answer is separate.",
        },
        "limits": [
            "Documentary projection only; not semantic acceptance or independent human validation.",
            "Lexical disposition cues are not institutional decisions.",
            "Same statute-bound address overlap is not substantive equivalence.",
        ],
    }
    if answer_packet is not None:
        packet["answer_attachment"] = bind_answer_packet(packet, answer_packet, bank)
    else:
        packet["answer_attachment"] = {
            "binding": {"state": "NOT_SUPPLIED"},
            "candidate_answer": None,
            "note": "No source_question packet supplied; public page shows documentary index only.",
        }
    return packet


def _source_meta_for_id(packet: Mapping[str, Any], source_id: str) -> dict[str, Any]:
    for row in (packet.get("summary") or {}).get("stages") or []:
        if row.get("source_id") == source_id:
            return dict(row)
    return {"source_id": source_id}


def _is_nonspecific_source_url(url: object) -> bool:
    text = str(url or "").strip()
    if not text:
        return True
    path = urlsplit(text).path.rstrip("/")
    return path.endswith("/api/v1/search")


def _source_link_html(meta: Mapping[str, Any], *, title: str) -> str:
    """Record-specific URL when available; never sell a generic search root as the record."""

    url = meta.get("url") or meta.get("record_url") or meta.get("official_url")
    locator = (
        meta.get("source_locator")
        or meta.get("identifier")
        or meta.get("record_locator")
        or ""
    )
    if url and urlsplit(str(url)).scheme in {"http", "https"} and not _is_nonspecific_source_url(url):
        return f'<a href="{_e(url)}">{_e(title)}</a>'
    bits = [_e(title)]
    if locator and str(locator) != str(title):
        bits.append(f"osoitin {_e(locator)}")
    if url and _is_nonspecific_source_url(url):
        bits.append(
            "yleinen hakuosoite (ei yksittäisen kirjauksen osoite): "
            f'<a href="{_e(url)}">{_e(url)}</a>'
        )
    record_url = meta.get("record_url")
    if record_url and urlsplit(str(record_url)).scheme in {"http", "https"}:
        label = (
            "Katso puheenvuoron tallenne"
            if meta.get("record_url_kind") == "VIDEO_SEGMENT"
            else "Avaa yksittäinen lähde"
        )
        bits.append(f'<a href="{_e(record_url)}">{label}</a>')
    return " · ".join(bits)


def _evidence_html(row: Mapping[str, Any], packet: Mapping[str, Any]) -> str:
    source_id = str(row.get("source_id") or "")
    meta = _source_meta_for_id(packet, source_id)
    quote = str(row.get("quote") or "")
    title = str(meta.get("identifier") or source_id)
    body = f"<p>{_source_link_html(meta, title=title)}"
    if row.get("first_pointer") is not None:
        body += f" · osoittimet {_e(row.get('first_pointer'))}–{_e(row.get('last_pointer'))}"
    body += "</p>"
    if quote:
        body += f"<blockquote>{_e(quote)}</blockquote>"
    else:
        body += "<p class='muted'>Ei sidottua lähdekatkelmaa.</p>"
    return body


def _answer_stage_blocks_html(packet: Mapping[str, Any]) -> str:
    """Compact source-stage timeline from packet stages (generic, not case-hardcoded)."""

    stages = list((packet.get("summary") or {}).get("stages") or [])
    if not stages:
        return ""
    items = []
    for row in stages:
        label = row.get("identifier") or row.get("source_id") or "—"
        stage = row.get("stage") or row.get("source_type") or ""
        date = row.get("source_date") or row.get("event_date") or ""
        link = _source_link_html(row, title=str(label))
        items.append(
            "<li>"
            f"<strong>{link}</strong>"
            + (f" · {_e(stage)}" if stage else "")
            + (f" · {_e(date)}" if date else "")
            + "</li>"
        )
    return (
        "<section class='answer-stages'><h3>Vastauksen vaiheet (lähdeketju)</h3>"
        f"<ol>{''.join(items)}</ol>"
        "<p class='muted'>Vaiheet ovat lähdepankin dokumentaarisia askeleita; "
        "institutionaalinen käsittely ≠ säädöksen voimaantulo.</p></section>"
    )


def _fi_review_date(value: object) -> str:
    text = str(value or "").strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        year, month, day = text.split("-")
        return f"{int(day)}.{int(month)}.{year}"
    return text or "—"


def _looks_like_research_dump(text: str) -> bool:
    """Heuristic for model/research-prompt dumps that must not lead the reader page."""

    blob = str(text or "").strip()
    if not blob:
        return False
    if len(blob) > 420:
        return True
    head = blob[:240]
    if head.startswith("#") or "## " in head:
        return True
    if "Älä päättele" in blob or "```" in blob:
        return True
    return "CANDIDATE_MODEL" in blob


def _compact_route_title(question: str, *, max_len: int = 110) -> str:
    text = " ".join(str(question or "").split())
    if not text:
        return "Reitti"
    if "?" in text:
        text = text.split("?", 1)[0] + "?"
    if len(text) <= max_len:
        return text
    cut = text[: max_len - 1].rsplit(" ", 1)[0] or text[: max_len - 1]
    return cut + "…"


def _candidate_answer_section(
    packet: Mapping[str, Any],
    *,
    prepared_answer: str | None = None,
    journey_href: str | None = None,
) -> str:
    attachment = packet.get("answer_attachment") or {}
    binding = attachment.get("binding") or {}
    candidate = attachment.get("candidate_answer")
    review = packet.get("source_answer_review")
    reviewed = isinstance(review, Mapping) and bool(review.get("reader_title") or review.get("decision"))
    if binding.get("state") == "NOT_SUPPLIED":
        return (
            "<section class='candidate-answer'>"
            "<p class='muted'>Lähdekysymyksen ehdokasvastauspakettia ei ole liitetty. "
            "Alla on vain dokumentaarinen indeksi, ei tulkittua vastausta.</p></section>"
        )
    if binding.get("state") != "BOUND" or not isinstance(candidate, Mapping):
        errors = "; ".join(str(item) for item in binding.get("errors") or [])
        return (
            "<section class='candidate-answer'>"
            f"<p class='muted'>Ehdokasvastaus jätettiin liittämättä (sidonta {_e(binding.get('state'))}). "
            f"{_e(errors)}</p></section>"
        )
    claims = candidate.get("claims") or []
    findings = candidate.get("map_findings") or []
    claim_html = []
    for claim in claims[:30]:
        evidence_rows = list(claim.get("evidence_bound") or [])[:8]
        evidence = "".join(_evidence_html(row, packet) for row in evidence_rows)
        if reviewed:
            claim_label = "Lähteisiin sidottu väite (tekoälyn lähdearvioitu tapaus)"
        else:
            claim_label = (
                "Lähdekatkelmaan sidottu ehdokasväite — ei vielä "
                "reference-arvioitu (ei tarkoita kumottua)"
            )
        evidence_block = (
            (
                f"<details><summary>Lähdeote ({_e(len(evidence_rows))})</summary>"
                + (evidence or "<p class='muted'>Ei sidottua otetta.</p>")
                + "</details>"
            )
            if evidence_rows or reviewed
            else (evidence or "<p class='muted'>Ei sidottua otetta.</p>")
        )
        claim_html.append(
            "<article class='claim'>"
            f"<p>{_e(claim.get('text'))}</p>"
            f"<p class='muted'>{_e(claim_label)}</p>"
            + evidence_block
            + "</article>"
        )
    finding_html = []
    for row in findings[:40]:
        finding_html.append(
            "<article class='claim'>"
            f"<p>{_e(row.get('text'))}</p>"
            "<p class='muted'>Karttalöydös (ehdokas) · "
            f"{_e(row.get('epistemic_status') or '')}</p>"
            + _evidence_html(row, packet)
            + "</article>"
        )
    unresolved = "".join(f"<li>{_e(item)}</li>" for item in candidate.get("unresolved") or [])
    next_ev = "".join(f"<li>{_e(item)}</li>" for item in candidate.get("next_evidence") or [])
    answer_text = str(candidate.get("answer") or "").strip()
    dump = _looks_like_research_dump(answer_text)
    prepared = " ".join(str(prepared_answer or "").split())
    as_of = packet.get("as_of")
    body = "<section class='candidate-answer'>"
    if journey_href:
        body += (
            f"<p class='prepared-journey'><a href='{_e(journey_href)}'>"
            "Avaa valmisteltu kysymysmatka</a></p>"
        )
    if reviewed:
        body += (
            (
                f"<p class='takeaway'>{_e(review.get('reader_takeaway'))}</p>"
                if review.get("reader_takeaway")
                else (f"<p class='takeaway'>{_e(prepared)}</p>" if prepared else "")
            )
            + "<p class='muted'>Lähteet tarkistanut erillinen tekoälylukija · "
            f"{_e(_fi_review_date(review.get('review_date')))}"
            + (f" · tarkastelupäivä {_e(as_of)}" if as_of else "")
            + ".</p>"
            + "<details><summary>Mitä «lähdearvioitu» tarkoittaa</summary>"
            "<p class='muted'>Erillinen tekoälylukija on tarkistanut tämän tapauksen "
            "ehdokasvastauksen suhteessa ilmoitettuihin lähteisiin. Se ei ole ihmisen "
            "sertifiointi, ei väestötason ADMITTED-päätös eikä yleistä tarkkuusväitettä. "
            "«Lähdearvioitu» ei ole sama kuin ihmisen reference-arvio. "
            "Raaka ehdokaspaketti säilyttää oman admission-tilansa "
            f"(<code>{_e(candidate.get('admission') or CANDIDATE_ADMISSION)}</code>).</p>"
            + (
                f"<ul>{''.join(f'<li>{_e(item)}</li>' for item in (review.get('limits') or [])[:12])}</ul>"
                if review.get("limits")
                else ""
            )
            + "</details>"
        )
        if answer_text:
            body += (
                "<details><summary>Lue koko vastaus</summary>"
                f"<p>{_e(answer_text)}</p></details>"
            )
    else:
        body += (
            "<p class='muted candidate-line'>"
            "Automaattinen ehdokas lähdesidonnalla; ei erillistä lähdearviota.</p>"
        )
        if prepared:
            body += f"<p class='takeaway'>{_e(prepared)}</p>"
        elif answer_text and not dump:
            body += f"<p class='takeaway'>{_e(answer_text)}</p>"
        if answer_text and (dump or prepared):
            body += (
                "<details class='raw-candidate'><summary>Raaka ehdokasvastaus</summary>"
                f"<p>{_e(answer_text)}</p></details>"
            )
    claims_block = ""
    if claim_html:
        claims_block = "".join(claim_html)
    elif finding_html:
        claims_block = "".join(finding_html)
    if claims_block:
        if reviewed and not dump:
            body += "<h3>Väitteet</h3>" + claims_block
        else:
            body += (
                "<details class='route-claims'><summary>Väitteet ja otteet</summary>"
                + claims_block
                + "</details>"
            )
    elif not prepared and not (answer_text and not dump):
        body += "<p class='muted'>Ei sidottuja ehdokasväitteitä tai karttalöydöksiä.</p>"
    stages = _answer_stage_blocks_html(packet)
    if stages:
        body += (
            "<details class='answer-stages-wrap'><summary>Lähdeketjun vaiheet</summary>"
            + stages
            + "</details>"
        )
    tech = (
        f"<p>Sidontatila: {_e(binding.get('state'))}; ehdokkaan status "
        f"{_e(candidate.get('status'))}; admission "
        f"{_e(candidate.get('admission'))}.</p>"
        + (
            f"<p>Arvion päätös: {_e(review.get('decision'))}; "
            f"ehdokas-tiiviste {_e(str(review.get('candidate_sha256') or '')[:16])}…</p>"
            if reviewed
            else ""
        )
    )
    if candidate.get("error"):
        tech += f"<p>Käsittelyviesti: {_e(candidate.get('error'))}</p>"
    body += f"<details><summary>Tekniset tunnisteet ja sidonta</summary>{tech}</details>"
    if unresolved:
        body += f"<details><summary>Avoimet kohdat</summary><ul>{unresolved}</ul></details>"
    if next_ev:
        body += f"<details><summary>Seuraava näyttö</summary><ul>{next_ev}</ul></details>"
    body += "</section>"
    return body


def render_route_html(
    packet: Mapping[str, Any],
    *,
    packet_url: str | None = None,
    back_url: str = "index.html",
    correction_url: str | None = None,
    reader_title: str | None = None,
    journey_href: str | None = None,
    prepared_answer: str | None = None,
) -> str:
    """Question-first page: ordinary title + brief candidate; research dump secondary."""

    question = str(packet.get("question") or packet.get("task_id") or "Reitti")
    summary = packet.get("summary") or {}
    actors = [row for row in packet.get("actors") or [] if row.get("actor_name")]
    actor_bits = "".join(
        "<li>"
        f"<strong>{_e(row.get('actor_name'))}</strong> · "
        f"{'Ensimmäinen allekirjoittaja' if row.get('role') == 'FIRST_SIGNATORY' else 'Muu allekirjoittaja'} "
        f"(laatija: tuntematon)"
        f"<details><summary>Lähdekohta</summary>"
        + _evidence_html(
            {
                "source_id": (row.get("source") or {}).get("source_id"),
                "quote": (row.get("source") or {}).get("quote"),
                "first_pointer": (row.get("source") or {}).get("first_pointer"),
                "last_pointer": (row.get("source") or {}).get("last_pointer"),
            },
            packet,
        )
        + "</details></li>"
        for row in actors[:40]
    )
    stages = "".join(
        "<li>"
        f"{_source_link_html(row, title=str(row.get('identifier') or row.get('source_id') or '—'))}"
        f" · {_e(row.get('stage'))}"
        + f" <span class='muted'>({_e(row.get('source_type'))})</span></li>"
        for row in summary.get("stages") or []
    )
    edges_all = packet.get("route_edges") or []
    edges_shown = edges_all[:40]
    edges = "".join(
        "<li>"
        f"{_e(row.get('from_identifier'))} → {_e(row.get('target_label'))} "
        f"(dokumentaarinen viittaus; vastaavuutta ei väitetä)"
        f"<details><summary>Lähdekohta</summary>{_evidence_html(row.get('source') or {}, packet)}</details>"
        "</li>"
        for row in edges_shown
    )
    date_labels = {
        "OWN_COMMENCEMENT": "Säädöksen oma voimaantulo",
        "PROPOSED_COMMENCEMENT_CUE": "Ehdotettu voimaantulo (vihje, ei säädös)",
        "ADOPTED_TEXT_COMMENCEMENT_CUE": "Hyväksytyn tekstin voimaantulovihje",
        "CONSOLIDATED_CAPTURE_COMMENCEMENT_CUE": "Konsolidoidun kaappauksen voimaantulovihje",
        "CARRIER_UNRESOLVED": "Voimaantulo (kantaja avoin)",
        "UNDERLYING_TARGET_AMENDMENT_REFERENCE": "Viittaus muutettavaan säädökseen",
        "OFFICIAL_CORRECTION": "Virallinen oikaisu",
        "SOURCE_EVENT_OR_PUBLICATION_DATE": "Lähteen päivämäärä",
        "COMMENCEMENT_CUE_UNTYPED": "Voimaantulovihje (tyypittämätön)",
    }
    dates = "".join(
        "<li>"
        f"{_e(date_labels.get(str(row.get('date_role') or ''), row.get('date_role')))}: "
        f"{_e(row.get('value') or row.get('target_statute') or '—')} "
        f"· {_e(row.get('identifier'))}"
        + (
            f"<details><summary>Lähdekohta · {_e(row.get('date_role'))}</summary>"
            f"{_evidence_html(row.get('source') or {}, packet)}</details>"
            if row.get("source")
            else f" <span class='muted'>({_e(row.get('date_role'))})</span>"
        )
        + "</li>"
        for row in packet.get("dates") or []
    )
    comps_all = packet.get("component_comparisons") or []
    comps = "".join(
        "<li>"
        f"<strong>{_e(row.get('component_key'))}</strong> · "
        f"{'erilainen tai osittainen ote' if row.get('text_identity') == 'DIFFERENT_OR_PARTIAL_EXCERPT' else 'identtinen ote'} · "
        "ehdokas sama osoite, ei politiikkavastaavuus"
        + "<ul>"
        + "".join(
            f"<li>{_e(ex.get('identifier'))}{_evidence_html(ex.get('source') or {}, packet)}</li>"
            for ex in (row.get("excerpts") or [])[:4]
        )
        + "</ul>"
        + (
            f"<p class='muted'>Näytetty {_e(row.get('excerpts_shown'))} / {_e(row.get('excerpts_total'))} otetta.</p>"
            if row.get("excerpts_total")
            else ""
        )
        + "</li>"
        for row in comps_all[:20]
    )
    cues = packet.get("disposition_cues") or []
    cue_bits = "".join(
        "<li>"
        f"Sanallinen vihje: {_e(row.get('cue_kind'))} "
        f"(ei päätös; {_e(row.get('status'))})"
        f"<details><summary>Lähdekohta</summary>{_evidence_html(row.get('source') or {}, packet)}</details>"
        "</li>"
        for row in cues[:30]
    )
    unsupported = "".join(
        f"<li>{_e(row.get('code'))}: {_e(row.get('reason'))}</li>"
        for row in packet.get("unsupported") or []
        if row.get("code") or row.get("reason")
    )

    def _residual_li(row: Mapping[str, Any]) -> str:
        code = row.get("code") or row.get("kind") or ""
        reason = row.get("reason") or row.get("description") or ""
        if not code and not reason:
            return ""
        if code and reason:
            return f"<li>{_e(code)}: {_e(reason)}</li>"
        return f"<li>{_e(code or reason)}</li>"

    residuals = "".join(_residual_li(row) for row in packet.get("residuals") or [])
    applicability = packet.get("applicability") or {}
    bank = packet.get("bank") or {}
    review = packet.get("source_answer_review") if isinstance(packet.get("source_answer_review"), Mapping) else {}
    inherited = " ".join(str(reader_title or "").split())
    review_title = " ".join(str(review.get("reader_title") or "").split())
    headline = inherited or review_title or _compact_route_title(question)
    research_prompt = (
        ""
        if not question or question == headline
        else (
            "<details class='full-question'><summary>Tutkimuskysymys (tarkka ohje)</summary>"
            f"<p>{_e(question)}</p></details>"
        )
    )
    unsupported_block = (
        (
            "<details class='unsupported'><summary>Mitä ei ole osoitettu</summary><ul>"
            + unsupported
            + "</ul></details>"
        )
        if unsupported
        else ""
    )
    body = (
        f"<p><a href='{_e(back_url)}'>Takaisin</a></p>"
        f"<h1>{_e(headline)}</h1>"
        + research_prompt
        + _candidate_answer_section(
            packet,
            prepared_answer=prepared_answer,
            journey_href=journey_href,
        )
        + unsupported_block
        + (
            f"<details class='residuals'><summary>Jäämät</summary><ul>{residuals}</ul></details>"
            if residuals
            else ""
        )
        + "<details class='documentary'><summary>Dokumentaarinen indeksi (ei vastaus)</summary>"
        f"<p class='muted'>Tehtävä {_e(packet.get('task_id'))} · asiatunniste {_e(packet.get('matter_id'))} · "
        f"tarkastelupäivä {_e(packet.get('as_of') or 'ei annettu')} · "
        f"lähdepankki {_e(bank.get('sources_file'))} · "
        f"tiiviste {_e(str(bank.get('sources_sha256') or '')[:16])}…</p>"
        "<h3>Allekirjoitukset</h3>"
        + (f"<ul>{actor_bits}</ul>" if actor_bits else "<p>Ei jäsenneltyjä allekirjoituksia.</p>")
        + f"<h3>Reitin lähteet</h3><ul>{stages}</ul>"
        + f"<h3>Dokumentaariset viittaukset</h3><p class='muted'>Näytetty {_e(len(edges_shown))} / {_e(len(edges_all))}.</p><ul>{edges}</ul>"
        + f"<h3>Päivämääräpropositiot</h3><ul>{dates}</ul>"
        + "<h3>Komponenttien ehdokasvertailu</h3>"
        + (f"<ul>{comps}</ul>" if comps else "<p>Ei statute-sidottua monilähdevertailua.</p>")
        + "<h3>Sanalliset päätösviiheet</h3>"
        + (f"<ul>{cue_bits}</ul>" if cue_bits else "<p>Ei sanallisia vihjeitä.</p>")
        + f"<p>Voimassaolorajaus: {_e(applicability.get('state'))}. "
        f"{_e(applicability.get('reason') or applicability.get('note') or '')}</p>"
        + "</details>"
    )
    if packet_url:
        body += f'<p><a download href="{_e(packet_url)}">Lataa täysi näyttöpaketti (JSON)</a></p>'
    if correction_url:
        body += f"<p><a href='{_e(correction_url)}'>Ilmoita korjaus</a></p>"
    body += (
        "<style>blockquote{white-space:pre-wrap;font-size:.92em}"
        ".muted{color:#334d57;font-size:.92em}li{margin:.35rem 0}"
        ".takeaway{font-size:1.05em;font-weight:600}.claim{margin:1rem 0;padding:.5rem 0;"
        "border-bottom:1px solid #ccd7da}"
        ".prepared-journey{margin:.35rem 0}.candidate-line{margin:.35rem 0}</style>"
    )
    return _document(
        headline[:120],
        body,
        description="Kysymyslähtöinen ehdokasvastaus ja dokumentaarinen reitti lähdekatkelmineen.",
    )



def write_route_artifact(
    packet: Mapping[str, Any],
    output_dir: Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write JSON + HTML for one packet into a unique directory."""

    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise RoutePacketError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    task_id = str(packet.get("task_id") or "task")
    safe = re.sub(r"[^\w.-]+", "_", task_id)
    json_name = f"{safe}.json"
    html_name = f"{safe}.html"
    (output_dir / json_name).write_text(
        json.dumps(packet, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / html_name).write_text(
        render_route_html(packet, packet_url=json_name), encoding="utf-8"
    )
    return {"json": json_name, "html": html_name, "output_dir": str(output_dir)}


def _load_answer_packets(answer_path: Path | None) -> dict[str, dict[str, Any]]:
    if answer_path is None:
        return {}
    path = Path(answer_path)
    files: list[Path]
    if path.is_dir():
        files = sorted(path.glob("*.json"))
    elif path.is_file():
        files = [path]
    else:
        raise RoutePacketError(f"answer path does not exist: {path}")
    by_task: dict[str, dict[str, Any]] = {}
    for file in files:
        if file.name in {"SUMMARY.json", "manifest.json", "RUN_RECEIPT.json"}:
            continue
        try:
            payload = json.loads(file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise RoutePacketError(f"invalid answer packet JSON: {file.name}: {error}") from error
        if not isinstance(payload, Mapping):
            continue
        task = str(payload.get("task_id") or "")
        if not task:
            continue
        by_task[task] = dict(payload)
    return by_task


def build_route_site(
    *,
    questions: Path,
    sources: Path,
    output_dir: Path,
    task_id: str | None = None,
    all_tasks: bool = False,
    split: str | None = None,
    as_of: str | None = None,
    answer: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Runnable producer→consumer build for one source bank."""

    bank = SourceQuestionBank.load(questions, sources)
    task_ids = bank.select_task_ids(task_id=task_id, all_tasks=all_tasks, split=split)
    answers = _load_answer_packets(answer)
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise RoutePacketError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    packets: list[dict[str, Any]] = []
    written: list[dict[str, Any]] = []
    for tid in task_ids:
        packet = build_route_packet(
            bank, tid, as_of=as_of, answer_packet=answers.get(tid)
        )
        packets.append(packet)
        safe = re.sub(r"[^\w.-]+", "_", tid)
        json_name = f"{safe}.json"
        html_name = f"{safe}.html"
        (output_dir / json_name).write_text(
            json.dumps(packet, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (output_dir / html_name).write_text(
            render_route_html(packet, packet_url=json_name), encoding="utf-8"
        )
        written.append({"task_id": tid, "json": json_name, "html": html_name})
    index_cards = "".join(
        "<article><h2><a href='"
        + _e(row["html"])
        + "'>"
        + _e(row["task_id"])
        + "</a></h2><p>"
        + _e(next(p.get("matter_id") for p in packets if p["task_id"] == row["task_id"]))
        + "</p></article>"
        for row in written
    )
    (output_dir / "index.html").write_text(
        _document(
            "Dokumentaariset reitit",
            "<h1>Reittitehtävät</h1>"
            "<p>Kysymyslähtöiset ehdokasvastaukset ja dokumentaarinen reitti. "
            "Ehdokasvastaus ei ole hyväksytty havainto.</p>"
            + index_cards,
            description="Valittavat reittitehtävät.",
        ),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "admission": ADMISSION,
        "questions_sha256": hashlib.sha256(Path(questions).read_bytes()).hexdigest(),
        "sources_sha256": hashlib.sha256(Path(sources).read_bytes()).hexdigest(),
        "task_ids": task_ids,
        "as_of": as_of,
        "answer_tasks_bound": sorted(
            tid
            for tid, packet in zip(task_ids, packets, strict=True)
            if (packet.get("answer_attachment") or {}).get("binding", {}).get("state") == "BOUND"
        ),
        "entries": written,
        "file_count": len(list(output_dir.rglob("*"))),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "SUMMARY.json").write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "task_count": len(packets),
                "tasks": [
                    {
                        "task_id": packet["task_id"],
                        "as_of": packet.get("as_of"),
                        "documentary_edge_count": packet["summary"]["documentary_edge_count"],
                        "signatory_count": packet["summary"]["signatory_count"],
                        "applicability_state": (packet.get("applicability") or {}).get("state"),
                        "residual_codes": [row.get("code") for row in packet.get("residuals") or []],
                    }
                    for packet in packets
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def route_cli(
    *,
    questions: Path,
    sources: Path,
    output: Path,
    task_id: str | None = None,
    all_tasks: bool = False,
    split: str | None = None,
    as_of: str | None = None,
    answer: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """CLI entry: deterministic route site from a self-contained bank."""

    try:
        return build_route_site(
            questions=questions,
            sources=sources,
            output_dir=output,
            task_id=task_id,
            all_tasks=all_tasks,
            split=split,
            as_of=as_of,
            answer=answer,
            overwrite=overwrite,
        )
    except (SourceQuestionError, RoutePacketError):
        raise
    except Exception as error:
        raise RoutePacketError(f"route build failed: {error}") from error


__all__ = [
    "ADMISSION",
    "CANDIDATE_ADMISSION",
    "SCHEMA_VERSION",
    "RoutePacketError",
    "bind_answer_packet",
    "build_route_packet",
    "build_route_site",
    "render_route_html",
    "route_cli",
    "write_route_artifact",
]

"""Research-only inquiry packets and a small source-linked HTML renderer.

This module is deliberately downstream of :mod:`paa.llm_inquiries`.  It makes
the public-facing research artifact easier to inspect without turning a model
proposal into evidence: each episode is shown once, baseline and structured
answers are paired on the same aggregate input, and every returned quote is
re-bound to the complete source fixture's URL, hash and character offset.

The packet uses Finnish status/disclaimer text for readers, while preserving
the canonical English enum values used by the source contracts.  Nothing in
this module assigns authorship, causality, implementation, fulfilment or a
semantic gold label.
"""

import argparse
import html
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from paa.llm_inquiries import _read_jsonl, _sha256_text, load_prepared_run

SCHEMA_VERSION = "paa.research.inquiry.packet.v1.2"
RESEARCH_STATUS = "PROPOSED_RESEARCH_NOT_ADMITTED"


def _safe_http_url(value: Any) -> str | None:
    """Allow only ordinary HTTP(S) links in rendered public HTML."""

    if not isinstance(value, str) or not value.strip():
        return None
    parsed = urlparse(value.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None
    if parsed.username or parsed.password:
        return None
    return value.strip()


def _sha256_file(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _fixture_cases(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    cases: dict[str, dict[str, Any]] = {}
    sources: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(path):
        episode_id = str(row.get("episode_id") or "")
        if not episode_id:
            continue
        previous_case = cases.get(episode_id)
        if previous_case is not None and previous_case != row:
            raise ValueError(f"duplicate episode ID has conflicting fields: {episode_id}")
        cases[episode_id] = row
        for source in row.get("sources", []):
            if isinstance(source, Mapping) and source.get("source_id"):
                source_id = str(source["source_id"])
                previous = sources.get(source_id)
                if previous is not None:
                    if previous != dict(source):
                        raise ValueError(f"duplicate source ID has conflicting fields: {source_id}")
                    continue
                sources[source_id] = dict(source)
    if not cases or not sources:
        raise ValueError(f"source fixture has no cases/sources: {path}")
    return cases, sources


def _source_anchor(
    evidence: Mapping[str, Any],
    source_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    source_id = str(evidence.get("source_id") or "")
    quote = str(evidence.get("quote") or "")
    source = source_by_id.get(source_id)
    source_text = str(source.get("text") or "") if source else ""
    candidate_starts: list[int] = []
    if source and quote:
        cursor = 0
        while len(candidate_starts) < 33:
            position = source_text.find(quote, cursor)
            if position < 0:
                break
            candidate_starts.append(position)
            cursor = position + max(1, len(quote))
    occurrence_count = len(candidate_starts)
    ambiguous = occurrence_count > 1
    start = candidate_starts[0] if occurrence_count == 1 else -1
    errors: list[str] = []
    if not source:
        errors.append("SOURCE_ID_NOT_IN_COMPLETE_FIXTURE")
    elif not quote or start < 0:
        errors.append("QUOTE_NOT_IN_COMPLETE_SOURCE")
    if ambiguous:
        errors = [error for error in errors if error != "QUOTE_NOT_IN_COMPLETE_SOURCE"]
        errors.append("QUOTE_OCCURS_MULTIPLE_TIMES_NO_MODEL_OFFSET")
    source_scope_note = None
    if source and source.get("source_kind") == "COMMITTEE_REPORT":
        source_scope_note = (
            "Indexed committee-report field; inspect local section headings and signatories before attributing "
            "a paragraph to the committee majority."
        )
    context_start = max(0, start - 240) if start >= 0 else None
    context_end = min(len(source_text), start + len(quote) + 240) if start >= 0 else None
    anchor = {
        "source_id": source_id,
        "quote": quote,
        "quote_sha256": _sha256_text(quote),
        "quote_occurrence_count": occurrence_count,
        "candidate_char_starts": candidate_starts[:32],
        "candidate_offsets_truncated": len(candidate_starts) > 32,
        "source_text_sha256": str(source.get("text_sha256") or "") if source else None,
        "raw_sha256": str(source.get("raw_sha256") or "") if source else None,
        "source_url": source.get("source_url") if source else None,
        "source_url_kind": source.get("source_url_kind") if source else None,
        "document_identifier": source.get("document_identifier") if source else None,
        "document_identifier_kind": source.get("document_identifier_kind") if source else None,
        "source_kind": source.get("source_kind") if source else None,
        "source_table": source.get("source_table") if source else None,
        "record_id": source.get("record_id") if source else None,
        "record_locator": source.get("record_locator") if source else None,
        "content_format": source.get("content_format") if source else None,
        "coverage_basis": source.get("coverage_basis") if source else None,
        "rights_state": source.get("rights_state") if source else None,
        "source_scope_note": source_scope_note,
        "document_role": {
            "expert_title": source.get("expert_title") if source else None,
            "committee_name": source.get("committee_name") if source else None,
            "committee_code": source.get("committee_code") if source else None,
            "report_type": source.get("report_type") if source else None,
        },
        "char_start": start if start >= 0 else None,
        "char_end": start + len(quote) if start >= 0 else None,
        "context_start": context_start,
        "context_end": context_end,
        "context_excerpt": source_text[context_start:context_end] if context_start is not None and context_end is not None else None,
        "validation_errors": errors,
        "anchor_state": "EXACT_COMPLETE_SOURCE" if not errors else "UNRESOLVED_SOURCE_BINDING",
    }
    anchor["source_anchor_id"] = "anchor-" + _sha256_text(
        f"{source_id}:{anchor.get('source_text_sha256') or ''}:{start if not ambiguous else 'AMBIGUOUS'}:"
        f"{anchor.get('char_end')}:{anchor['quote_sha256']}"
    )[:24]
    return anchor


def _dedup_claims(rows: Sequence[Mapping[str, Any]], source_by_id: Mapping[str, Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    claims: dict[tuple[str, str, str], dict[str, Any]] = {}
    errors: list[str] = []
    for row in rows:
        normalized = row.get("normalized")
        if not isinstance(normalized, Mapping):
            continue
        for claim in normalized.get("claims", []):
            if not isinstance(claim, Mapping):
                continue
            claim_type = str(claim.get("claim_type") or claim.get("type") or "")
            text = str(claim.get("text") or "").strip()
            evidence = claim.get("evidence")
            if not text or not isinstance(evidence, list):
                continue
            anchors = [_source_anchor(item, source_by_id) for item in evidence if isinstance(item, Mapping)]
            if not anchors:
                continue
            # The aggregate contract permits up to two evidence anchors. The
            # first exact anchor remains the deterministic display key and
            # subsequent anchors stay in the evidence list.
            key = (claim_type, text, str(anchors[0].get("source_id")))
            item = claims.setdefault(
                key,
                {
                    "type": claim_type,
                    "text": text,
                    "state": "PROPOSED",
                    "semantic_disposition": "NOT_SOURCE_REVIEWED",
                    "research_status": RESEARCH_STATUS,
                    "evidence": anchors,
                    "origin_modes": set(),
                    "origin_windows": set(),
                    "validation_errors": [],
                    "prompt_identities": set(),
                },
            )
            mode = str(row.get("mode") or "")
            window_id = str(row.get("window_id") or "")
            if mode:
                item["origin_modes"].add(mode)
            if window_id:
                item["origin_windows"].add(window_id)
            prompt_identity = str(row.get("prompt_sha256") or row.get("prompt_version") or "")
            if prompt_identity:
                item["prompt_identities"].add(prompt_identity)
            existing_anchors = {
                (str(anchor.get("source_id")), str(anchor.get("quote")))
                for anchor in item["evidence"]
            }
            for anchor in anchors:
                anchor_key = (str(anchor.get("source_id")), str(anchor.get("quote")))
                if anchor_key not in existing_anchors and len(item["evidence"]) < 4:
                    item["evidence"].append(anchor)
                    existing_anchors.add(anchor_key)
            item["validation_errors"].extend(
                error
                for anchor in anchors
                for error in anchor.get("validation_errors", [])
                if error not in item["validation_errors"]
            )
            errors.extend(error for error in item["validation_errors"] if error not in errors)
    result: list[dict[str, Any]] = []
    for item in sorted(claims.values(), key=lambda value: (str(value["type"]), str(value["text"]))):
        prompt_identities = sorted(item["prompt_identities"])
        stable_material = {
            "type": item["type"],
            "text": item["text"],
            "evidence": [
                {
                    "source_anchor_id": anchor.get("source_anchor_id"),
                    "source_text_sha256": anchor.get("source_text_sha256"),
                    "quote_sha256": anchor.get("quote_sha256"),
                }
                for anchor in item["evidence"]
            ],
            "prompt_identities": prompt_identities,
        }
        result.append(
            {
                **item,
                "claim_id": "claim-" + _sha256_text(json.dumps(stable_material, ensure_ascii=False, sort_keys=True))[:24],
                "source_anchor_ids": [str(anchor.get("source_anchor_id")) for anchor in item["evidence"]],
                "origin_modes": sorted(item["origin_modes"]),
                "origin_windows": sorted(item["origin_windows"]),
                "prompt_identities": prompt_identities,
                "validation_errors": sorted(set(item["validation_errors"])),
            }
        )
    return result, errors


def _dedup_unknowns(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    values: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        normalized = row.get("normalized")
        if not isinstance(normalized, Mapping):
            continue
        for unknown in normalized.get("localized_unknowns", []):
            if not isinstance(unknown, Mapping):
                continue
            key = (
                str(unknown.get("text") or ""),
                str(unknown.get("missing_evidence") or unknown.get("missing") or ""),
                str(unknown.get("next_observation") or unknown.get("next") or ""),
            )
            if not any(key):
                continue
            values.setdefault(
                key,
                {
                    "text": key[0],
                    "missing_evidence": key[1],
                    "next_observation": key[2],
                    "source_ids": sorted({str(value) for value in unknown.get("source_ids", [])}),
                    "state": "UNRESOLVED",
                },
            )
    return sorted(values.values(), key=lambda value: (value["text"], value["missing_evidence"]))


def _source_review_queue(claims: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Create a deterministic, non-admitting handoff for source reading."""

    queue: list[dict[str, Any]] = []
    for claim in claims:
        claim_id = str(claim.get("claim_id") or "")
        anchors = [anchor for anchor in claim.get("evidence", []) if isinstance(anchor, Mapping)]
        source_versions = {
            str(anchor.get("source_id")): str(anchor.get("source_text_sha256") or "")
            for anchor in anchors
            if anchor.get("source_id")
        }
        anchor_ids = [str(anchor.get("source_anchor_id") or "") for anchor in anchors]
        review_material = {
            "claim_id": claim_id,
            "source_anchor_ids": anchor_ids,
            "source_versions": source_versions,
        }
        queue.append(
            {
                "review_item_id": "review-" + _sha256_text(json.dumps(review_material, ensure_ascii=False, sort_keys=True))[:24],
                "claim_id": claim_id,
                "claim_type": claim.get("type"),
                "claim_text": claim.get("text"),
                "source_anchor_ids": anchor_ids,
                "source_versions": source_versions,
                "exact_quotes": [str(anchor.get("quote") or "") for anchor in anchors],
                "review_state": "PENDING_SOURCE_REVIEW",
                "semantic_disposition": "NOT_SOURCE_REVIEWED",
                "reviewer": None,
                "method": None,
                "rationale": None,
                "admission_state": "PROPOSED / NOT_ADMITTED",
            }
        )
    return sorted(queue, key=lambda item: str(item["review_item_id"]))


def _source_coverage(
    case: Mapping[str, Any],
    aggregate_sources: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    coverage: list[dict[str, Any]] = []
    for source in case.get("sources", []):
        if not isinstance(source, Mapping):
            continue
        source_id = str(source.get("source_id") or "")
        complete_hash = str(source.get("text_sha256") or "")
        aggregate = aggregate_sources.get(source_id, {})
        coverage.append(
            {
                "source_id": source_id,
                "source_kind": source.get("source_kind"),
                "source_table": source.get("source_table"),
                "record_id": source.get("record_id"),
                "document_identifier": source.get("document_identifier"),
                "document_identifier_kind": source.get("document_identifier_kind"),
                "title": source.get("title"),
                "expert_title": source.get("expert_title"),
                "committee_name": source.get("committee_name"),
                "committee_code": source.get("committee_code"),
                "source_url": source.get("source_url"),
                "source_url_kind": source.get("source_url_kind"),
                "record_locator": source.get("record_locator"),
                "content_format": source.get("content_format"),
                "coverage_basis": source.get("coverage_basis"),
                "rights_state": source.get("rights_state"),
                "source_scope_note": (
                    "Indexed committee-report field; inspect local section headings and signatories before attributing "
                    "a paragraph to the committee majority."
                    if source.get("source_kind") == "COMMITTEE_REPORT"
                    else None
                ),
                "text_sha256": complete_hash,
                "raw_sha256": source.get("raw_sha256"),
                "full_char_count": len(str(source.get("text") or "")),
                "aggregate_coverage_state": aggregate.get("coverage_state"),
                "aggregate_clip_count": len(aggregate.get("clips", [])) if isinstance(aggregate.get("clips"), list) else 0,
                "aggregate_clips": [
                    {
                        "start": clip.get("start"),
                        "end": clip.get("end"),
                        "text_sha256": clip.get("text_sha256"),
                        "boundary_version": clip.get("boundary_version"),
                    }
                    for clip in aggregate.get("clips", [])
                    if isinstance(clip, Mapping)
                ],
            }
        )
    return coverage


def _status_label_fi(normalized_rows: Sequence[Mapping[str, Any]]) -> str:
    statuses = {str(row.get("normalized", {}).get("status")) for row in normalized_rows if isinstance(row.get("normalized"), Mapping)}
    if "VALID" in statuses and statuses <= {"VALID"}:
        label = "Muoto ja täsmäankkurit validoitu; sisältö on edelleen malliehdotus"
    elif statuses & {"VALID", "PARTIAL"}:
        label = "Osittainen lähdeankkurointi; sisältöä ei ole hyväksytty"
    else:
        label = "Lähde- tai kuljetusongelma; vastausta ei hyväksytty"
    if any(
        warning.get("code") == "STRING_LIMIT_REACHED"
        for row in normalized_rows
        for warning in (
            row.get("normalized", {}).get("warnings", [])
            if isinstance(row.get("normalized"), Mapping)
            else []
        )
        if isinstance(warning, Mapping)
    ):
        label += "; vastaus saavutti merkkirajan ja voi olla keskeneräinen"
    return label


def _representative_rows(receipts: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Choose aggregate receipts and hide exhaustive-window duplication."""

    by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in receipts:
        mode = str(row.get("mode") or "")
        if mode in {"baseline", "structured"}:
            by_mode[mode].append(dict(row))
    for mode, values in by_mode.items():
        aggregate = [row for row in values if str(row.get("window_id", "")).endswith("-episode-aggregate")]
        by_mode[mode] = aggregate[:1] if aggregate else values[:1]
    return dict(by_mode)


def build_research_packet(run: Path, source_fixture: Path, *, label: str | None = None) -> dict[str, Any]:
    """Build a source-linked, one-card-per-episode research packet."""

    manifest, windows = load_prepared_run(run)
    run_manifest_sha256 = _sha256_file(run / "manifest.json")
    cases, source_by_id = _fixture_cases(source_fixture)
    window_by_episode = {str(window.get("episode_id")): window for window in windows}
    receipts_by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    receipt_paths = sorted((run / "receipts").glob("*.json"))
    for path in receipt_paths:
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, dict) and value.get("episode_id"):
            receipts_by_episode[str(value["episode_id"])].append(value)
    episodes: list[dict[str, Any]] = []
    for episode_id in sorted(cases):
        case = cases[episode_id]
        window = window_by_episode.get(episode_id)
        payload = window.get("payload", {}) if isinstance(window, Mapping) else {}
        question = case.get("question_contract")
        if not isinstance(question, Mapping):
            question = payload.get("question_contract", {}) if isinstance(payload, Mapping) else {}
        aggregate_sources = {
            str(source.get("source_id")): source
            for source in payload.get("sources", [])
            if isinstance(source, Mapping) and source.get("source_id")
        }
        mode_rows = _representative_rows(receipts_by_episode.get(episode_id, []))
        modes: dict[str, Any] = {}
        for mode in ("baseline", "structured"):
            rows = mode_rows.get(mode, [])
            normalized_rows = [row for row in rows if isinstance(row.get("normalized"), Mapping)]
            normalized = normalized_rows[0].get("normalized", {}) if normalized_rows else {}
            claims, claim_errors = _dedup_claims(normalized_rows, source_by_id)
            modes[mode] = {
                "answer": normalized.get("proposed_answer", "") if isinstance(normalized, Mapping) else "",
                "answer_status": _status_label_fi(normalized_rows),
                "warnings": normalized.get("warnings", []) if isinstance(normalized, Mapping) else [],
                "claims": claims,
                "localized_unknowns": _dedup_unknowns(normalized_rows),
                "unsupported_claims": normalized.get("unsupported_claims", []) if isinstance(normalized, Mapping) else [],
                "receipt_statuses": sorted({str(row.get("receipt_status") or "MISSING") for row in rows}),
                "receipt_count": len(rows),
                "receipt_ids": [str(row.get("request_id") or "") for row in rows],
                "request_ids_nonempty": sorted({str(row.get("request_id")) for row in rows if row.get("request_id")}),
                "request_id_missing_count": sum(1 for row in rows if not row.get("request_id")),
                "window_ids": [str(row.get("window_id") or "") for row in rows],
                "input_sha256": str(rows[0].get("input_sha256") or "") if rows else None,
                "source_payload_sha256": str(rows[0].get("source_payload_sha256") or "") if rows else None,
                "prompt_sha256": str(rows[0].get("prompt_sha256") or "") if rows else None,
                "prompt_sha256s": sorted({str(row.get("prompt_sha256")) for row in rows if row.get("prompt_sha256")}),
                "prompt_contract_sha256": manifest.get("prompt_contract_sha256"),
                "schema_sha256": str(rows[0].get("schema_sha256") or "") if rows else None,
                "run_manifest_sha256": run_manifest_sha256,
                "provenance_state": (
                    "REQUEST_AND_PROMPT_IDENTITIES_PRESENT"
                    if rows and all(row.get("request_id") and row.get("prompt_sha256") for row in rows)
                    else "REQUEST_OR_PROMPT_IDENTITY_MISSING"
                ),
                "validation_errors": sorted(set(claim_errors)),
                "semantic_disposition": "NOT_SOURCE_REVIEWED",
                "admission_state": "PROPOSED / NOT_ADMITTED",
            }
        baseline = modes["baseline"]
        structured = modes["structured"]
        episodes.append(
            {
                "episode_id": episode_id,
                "question": {
                    "text": question.get("text"),
                    "target_scope": question.get("target_scope"),
                    "period": question.get("period"),
                    "comparison": question.get("comparison"),
                    "valid_outputs": question.get("valid_outputs"),
                    "unknowns": question.get("unknowns"),
                },
                "research_status_fi": "Lähdeankkuroitu mallivertailu; ei itsenäinen lähdearvio",
                "interpretation_fi": "Vastaukset ovat saman lähdepaketin kaksi malliehdotusta. Täsmäote osoittaa vain, mihin tekstiin ehdotus viittaa; se ei osoita tekijyyttä, syy-yhteyttä, toteutumista tai vaikutusta.",
                "modes": modes,
                "source_review_queue": _source_review_queue(
                    [claim for mode in modes.values() for claim in mode.get("claims", []) if isinstance(claim, Mapping)]
                ),
                "paired_input": {
                    "same_input_sha256": bool(baseline.get("input_sha256") and baseline.get("input_sha256") == structured.get("input_sha256")),
                    "same_source_payload_sha256": bool(baseline.get("source_payload_sha256") and baseline.get("source_payload_sha256") == structured.get("source_payload_sha256")),
                },
                "source_coverage": _source_coverage(case, aggregate_sources),
                "coverage_state": payload.get("coverage", {}).get("coverage_state") if isinstance(payload.get("coverage"), Mapping) else None,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "research_status": RESEARCH_STATUS,
        "research_status_fi": "Tutkimusehdotus, ei hyväksytty havainto",
        "label": label or run.name,
        "run": str(run),
        "run_manifest_sha256": run_manifest_sha256,
        "source_fixture": str(source_fixture),
        "source_fixture_sha256": _sha256_file(source_fixture),
        "coverage_mode": manifest.get("coverage_mode"),
        "aggregate_context_version": manifest.get("aggregate_context_version"),
        "output_schema_version": manifest.get("output_schema_version"),
        "case_count": len(episodes),
        "receipt_count": len(receipt_paths),
        "window_count": len(windows),
        "not_model_admission": True,
        "episodes": episodes,
    }


def render_research_html(packet: Mapping[str, Any]) -> str:
    """Render a compact question-first HTML view with source links."""

    def esc(value: Any) -> str:
        return html.escape(str(value or ""))

    cards: list[str] = []
    for episode in packet.get("episodes", []):
        if not isinstance(episode, Mapping):
            continue
        question = episode.get("question", {})
        source_coverage = [
            source for source in episode.get("source_coverage", []) if isinstance(source, Mapping)
        ]
        display_source = next(
            (
                source
                for source in source_coverage
                if source.get("document_identifier") or source.get("title")
            ),
            {},
        )
        episode_label = display_source.get("document_identifier") or display_source.get("title") or episode.get("episode_id")
        mode_cards: list[str] = []
        modes = episode.get("modes", {})
        for mode in ("baseline", "structured"):
            item = modes.get(mode, {}) if isinstance(modes, Mapping) else {}
            mode_label = {"baseline": "Perusvertailu", "structured": "Jäsennelty vertailu"}.get(mode, mode)
            claims_html: list[str] = []
            for claim_index, claim in enumerate(item.get("claims", []) if isinstance(item, Mapping) else []):
                evidence_html: list[str] = []
                for evidence_index, evidence in enumerate(claim.get("evidence", []) if isinstance(claim, Mapping) else []):
                    source_url = _safe_http_url(evidence.get("source_url"))
                    quote = esc(evidence.get("quote"))
                    anchor_id = esc(evidence.get("source_anchor_id"))
                    dom_anchor_id = esc(f"{mode}-evidence-{claim_index}-{evidence_index}-{anchor_id}")
                    if source_url:
                        quote_html = f'<a href="{esc(source_url)}">{quote}</a>'
                    else:
                        quote_html = f"<span>{quote}</span>"
                    role = evidence.get("document_role", {}) if isinstance(evidence, Mapping) else {}
                    role_parts = [
                        str(evidence.get("document_identifier") or ""),
                        str(evidence.get("source_kind") or ""),
                        str(role.get("committee_name") or "") if isinstance(role, Mapping) else "",
                    ]
                    role_label = " · ".join(part for part in role_parts if part)
                    anchor_state = esc(evidence.get("anchor_state"))
                    offset_label = (
                        f"{evidence.get('char_start')}–{evidence.get('char_end')}"
                        if evidence.get("char_start") is not None
                        else "ei yksilöityä kohtaa"
                    )
                    context_html = (
                        f"<details class=context><summary>Alkuperäinen lähdekonteksti</summary>"
                        f"<blockquote>{esc(evidence.get('context_excerpt'))}</blockquote></details>"
                        if evidence.get("context_excerpt")
                        else ""
                    )
                    scope_note = esc(evidence.get("source_scope_note"))
                    evidence_html.append(
                        f"<details class=source-evidence id=\"{dom_anchor_id}\" "
                        f"data-source-anchor-id=\"{anchor_id}\">"
                        f"<summary>Täsmäote · {anchor_state} · kohta {esc(offset_label)}</summary>"
                        f"<blockquote>{quote_html}</blockquote>"
                        f"<small>{esc(role_label)} · toistumia {esc(evidence.get('quote_occurrence_count'))}</small>"
                        f"{f'<small class=source-limit>Lähdehuomio: {scope_note}</small>' if scope_note else ''}"
                        f"{context_html}</details>"
                    )
                claim_id_html = (
                    f"<details class=claim-meta><summary>Väitteen tunniste</summary>"
                    f"<code>{esc(claim.get('claim_id'))}</code></details>"
                )
                claims_html.append(
                    f"<li><span>{esc(claim.get('text'))}</span><br>"
                    f"<div class=evidence-list><strong>Lähdeankkurit:</strong> "
                    f"{claim_id_html}{''.join(evidence_html) or 'ei täsmäankkuria'}</div></li>"
                )
            unknowns_html = "".join(
                "<li>"
                f"<span>{esc(item.get('text'))}</span>"
                f"<br><small>Puuttuu: {esc(item.get('missing_evidence'))}; seuraava havainto: {esc(item.get('next_observation'))}</small>"
                "</li>"
                for item in item.get("localized_unknowns", [])
                if isinstance(item, Mapping)
            )
            answer_warning_html = "".join(
                f"<small class=answer-warning>Huomio: {esc(warning.get('message') or 'Vastaus voi olla keskeneräinen.')}</small>"
                for warning in item.get("warnings", [])
                if isinstance(warning, Mapping) and warning.get("code") == "STRING_LIMIT_REACHED"
            )
            mode_cards.append(
                "<section class=mode>"
                f"<h3>{esc(mode_label)}</h3>"
                f"<p class=answer data-admission='PROPOSED / NOT_ADMITTED'>{esc(item.get('answer')) or 'Ei palautettua vastausta'}</p>"
                f"{answer_warning_html}"
                f"<p class=status>{esc(item.get('answer_status'))} · Ehdotus, ei hyväksytty havainto</p>"
                f"<h4>Ehdotetut väitteet</h4><ul>{''.join(claims_html) or '<li>Ei palautettua väitettä</li>'}</ul>"
                f"<h4>Paikalliset avoimet kohdat</h4><ul>{unknowns_html or '<li>Ei palautettua tuntematonta</li>'}</ul>"
                "</section>"
            )
        source_html = "".join(
            (
                f"<li id='source-{esc(source.get('source_id'))}'>"
                + (
                    f"<a href='{esc(_safe_http_url(source.get('source_url')))}'>"
                    f"{esc(source.get('document_identifier') or source.get('source_id'))}</a>"
                    if _safe_http_url(source.get("source_url"))
                    else esc(source.get("document_identifier") or source.get("source_id"))
                )
                + f" — {esc(source.get('title'))}; SHA-256 <code>{esc(source.get('text_sha256'))}</code>"
                + (
                    f"<br><small class=source-limit>Lähdehuomio: {esc(source.get('source_scope_note'))}</small>"
                    if source.get("source_scope_note")
                    else ""
                )
                + "</li>"
            )
            for source in source_coverage
            if isinstance(source, Mapping)
        )
        cards.append(
            "<article class=episode>"
            f"<h2>{esc(episode_label)}</h2>"
            f"<p class=question><strong>Kysymys:</strong> {esc(question.get('text'))}</p>"
            f"<p class=scope><strong>Rajaus:</strong> {esc(question.get('target_scope'))}</p>"
            f"<div class=modes>{''.join(mode_cards)}</div>"
            f"<details><summary>Lähteiden kattavuus</summary>"
            f"<p><small>Tapaustunnus: <code>{esc(episode.get('episode_id'))}</code></small></p>"
            f"<ul>{source_html}</ul></details>"
            f"<p class=disclaimer>{esc(episode.get('interpretation_fi'))}</p>"
            "</article>"
        )
    return (
        "<!doctype html><html lang='fi'><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>{esc(packet.get('label'))}</title>"
        "<style>body{font:16px system-ui;max-width:1100px;margin:2rem auto;padding:0 1rem}"
        ".episode{border:1px solid #bbb;border-radius:8px;padding:1rem;margin:1rem 0}"
        ".modes{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:1rem}"
        ".mode{background:#f6f6f6;padding:.8rem}.question{font-size:1.15rem}.answer{font-weight:600}"
        ".status,.disclaimer{color:#555;font-size:.9rem}.answer-warning{display:block;color:#8a3d00;font-weight:600}.evidence-list{margin:.5rem 0}.source-evidence{margin:.4rem 0;padding:.35rem;background:#fff}"
        ".source-evidence blockquote,.context blockquote{margin:.4rem 0;padding:.4rem;border-left:3px solid #aaa;white-space:pre-wrap}"
        ".source-evidence small{display:block}.source-limit{color:#7a3f00}a,code{overflow-wrap:anywhere;word-break:break-word}"
        "@media(max-width:720px){.modes{grid-template-columns:1fr}}</style>"
        f"<h1>{esc(packet.get('label'))}</h1><p>{esc(packet.get('research_status_fi'))}</p>"
        f"{''.join(cards)}</html>"
    )


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--source-fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--html", type=Path)
    parser.add_argument("--label")
    args = parser.parse_args(argv)
    packet = build_research_packet(args.run, args.source_fixture, label=args.label)
    _write_json(args.output, packet)
    if args.html:
        args.html.parent.mkdir(parents=True, exist_ok=True)
        args.html.write_text(render_research_html(packet), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "case_count": packet["case_count"], "status": packet["research_status"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

"""Bounded, source-blind-island comparison for PAA inquiry cases.

The three generation lanes receive byte-identical model inputs and different
framing prompts.  This module deliberately stops at the proposed/review layer:
exact quote checks prove source anchoring, not semantic truth or publication
readiness.  Reviewed-case fixtures are a developmental, selection-biased
control set; this is not a held-out evaluation.
"""


import argparse
import asyncio
import json
import os
import time
from collections.abc import Mapping
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import jsonschema

from paa.config import FIXTURE_DIR, REPORTS, ensure_dirs
from paa.inquiry_cases import load_reviewed_cases
from paa.llm_client import LocalLLMClient, digest

PACKAGE_ROOT = Path(__file__).resolve().parent
PROMPT_DIR = PACKAGE_ROOT / "prompts"
SOURCE_SLICE_PATH = FIXTURE_DIR / "mev_cases_source_slices.jsonl"
REVIEW_PATH = FIXTURE_DIR / "inquiry_case_reviews.json"
DEFAULT_RUN_PATH = REPORTS / "paa_llm_islands_bounded_20261007.json"
DEFAULT_SUMMARY_PATH = REPORTS / "paa_llm_islands_bounded_20261007.md"

CASE_ORDER = (
    "rai-2020-constitutional-repair",
    "infectious-law-2021-considered-and-removed",
    "climate-2022-emissions-budgets",
    "adult-support-2024-replacement-condition",
)

LANE_PROMPTS = {
    "direct_source_first": PROMPT_DIR / "island_direct_source_first.txt",
    "frameless_discovery": PROMPT_DIR / "island_frameless_discovery.txt",
    "contrarian_fair_repair": PROMPT_DIR / "island_contrarian_fair_repair.txt",
}
TERMINAL_STATES = frozenset({
    "ALIGNED", "CONTRARY", "RELATED", "INSUFFICIENT_EVIDENCE", "NOT_TESTABLE",
})
DISPOSITIONS = frozenset({"ACCEPTED", "REJECTED", "UNRESOLVED"})


def _anchor_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["source_id", "quote"],
        "properties": {
            "source_id": {"type": "string", "minLength": 1},
            "quote": {"type": "string", "minLength": 1, "maxLength": 1200},
        },
    }


def _candidate_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "candidate_id", "distinction", "source_anchors", "actor_or_channel",
            "date_basis", "terminal_relevance", "alternative", "missing_fact",
            "review_burden", "review_note", "public_answer_fi",
        ],
        "properties": {
            "candidate_id": {"type": "string", "minLength": 1, "maxLength": 80},
            "distinction": {"type": "string", "minLength": 1, "maxLength": 1200},
            "source_anchors": {"type": "array", "minItems": 1, "maxItems": 6, "items": _anchor_schema()},
            "actor_or_channel": {"type": "string", "minLength": 1, "maxLength": 500},
            "date_basis": {"type": "string", "minLength": 1, "maxLength": 500},
            "terminal_relevance": {"type": "string", "minLength": 1, "maxLength": 600},
            "alternative": {"type": "string", "minLength": 1, "maxLength": 1000},
            "missing_fact": {"type": "string", "minLength": 1, "maxLength": 1000},
            "review_burden": {"enum": ["LOW", "MEDIUM", "HIGH"]},
            "review_note": {"type": "string", "minLength": 1, "maxLength": 600},
            "public_answer_fi": {"type": "string", "minLength": 1, "maxLength": 1400},
        },
    }


def island_output_schema() -> dict[str, Any]:
    """Return the strict model-facing output contract."""

    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "terminal_state", "direct_answer_fi", "assumptions",
            "candidate_findings", "attacks", "stopping_note",
        ],
        "properties": {
            "terminal_state": {"enum": sorted(TERMINAL_STATES)},
            "direct_answer_fi": {"type": "string", "minLength": 1, "maxLength": 1800},
            "assumptions": {
                "type": "array", "maxItems": 8,
                "items": {"type": "string", "minLength": 1, "maxLength": 500},
            },
            "candidate_findings": {
                "type": "array", "maxItems": 8, "items": _candidate_schema(),
            },
            "attacks": {
                "type": "array", "maxItems": 8, "items": {
                    "type": "object", "additionalProperties": False,
                    "required": [
                        "attack_id", "target_candidate_id", "attack", "source_anchors",
                        "fair_repair", "alternative", "missing_fact", "review_burden",
                        "review_note", "material_distinction",
                    ],
                    "properties": {
                        "attack_id": {"type": "string", "minLength": 1, "maxLength": 80},
                        "target_candidate_id": {"type": "string", "minLength": 1, "maxLength": 80},
                        "attack": {"type": "string", "minLength": 1, "maxLength": 1200},
                        "source_anchors": {"type": "array", "minItems": 1, "maxItems": 6, "items": _anchor_schema()},
                        "fair_repair": {"type": "string", "minLength": 1, "maxLength": 1200},
                        "alternative": {"type": "string", "minLength": 1, "maxLength": 1000},
                        "missing_fact": {"type": "string", "minLength": 1, "maxLength": 1000},
                        "review_burden": {"enum": ["LOW", "MEDIUM", "HIGH"]},
                        "review_note": {"type": "string", "minLength": 1, "maxLength": 600},
                        "material_distinction": {"type": "string", "minLength": 1, "maxLength": 600},
                    },
                },
            },
            "stopping_note": {"type": "string", "minLength": 1, "maxLength": 800},
        },
    }


def _source_role(source: Mapping[str, Any]) -> str:
    return str(source.get("source_kind") or "SOURCE_RECORD")


def _clip_windows(text: str, quotes: list[str], *, radius: int = 900) -> list[dict[str, Any]]:
    """Build merged windows that retain every selected exact anchor."""

    spans: list[tuple[int, int, str]] = []
    for quote in quotes:
        start = text.find(quote)
        if start < 0:
            raise ValueError("review anchor is not present in its source text")
        spans.append((max(0, start - radius), min(len(text), start + len(quote) + radius), quote))
    if not spans:
        spans = [(0, min(len(text), radius * 2), "")]
    spans.sort()
    merged: list[dict[str, Any]] = []
    for start, end, quote in spans:
        if merged and start <= merged[-1]["end"]:
            merged[-1]["end"] = max(merged[-1]["end"], end)
            merged[-1]["anchors"].append(quote)
            merged[-1]["text"] = text[merged[-1]["start"]:merged[-1]["end"]]
        else:
            merged.append({
                "start": start, "end": end, "anchors": [quote] if quote else [],
                "text": text[start:end],
            })
    return merged


def build_source_packet(case: Mapping[str, Any], *, case_ref: str = "CASE") -> dict[str, Any]:
    """Build model-safe input plus a private full-text verification view.

    Only ``model_input`` is serialized into prompts.  The verification view
    retains full source bytes for post-generation checks and is never sent to
    the model.  Reviewed claims, unknowns, titles, answer standards and
    selection rationale are intentionally omitted from ``model_input``.
    """

    evidence_by_source: dict[str, list[str]] = {}
    for ref in case.get("evidence", []):
        evidence_by_source.setdefault(ref["source_id"], []).append(ref["quote"])
    model_sources = []
    verification_sources: dict[str, dict[str, Any]] = {}
    for source in sorted(case.get("sources", []), key=lambda row: str(row.get("source_id", ""))):
        source_id = str(source["source_id"])
        text = str(source["text"])
        quotes = evidence_by_source.get(source_id, [])
        clips = _clip_windows(text, quotes)
        model_sources.append({
            "source_id": source_id,
            "source_role": _source_role(source),
            "document_identifier": source.get("document_identifier"),
            "source_locator": source.get("locator"),
            "source_url": source.get("url"),
            "text_sha256": source.get("text_sha256"),
            "clips": clips,
        })
        verification_sources[source_id] = {
            "source_id": source_id,
            "source_role": _source_role(source),
            "text": text,
            "text_sha256": source.get("text_sha256"),
            "raw_sha256": source.get("raw_sha256"),
            "clips": clips,
        }
    model_input = {
        "packet_version": "paa.llm_islands.input.v1",
        "case_ref": case_ref,
        "question": {
            "text": case["question"]["text"],
            "scope": case["question"]["scope"],
        },
        "sources": model_sources,
        "instructions": [
            "Käytä vain näitä lähteitä ja kysymystä.",
            "Lähdekatkelmat ovat alkuperäistä tekstiä; säilytä niiden rivinvaihdot.",
            "Lopputulos on ehdotus erilliselle lähdearviolle, ei julkaisu- tai kultatotuuspäätös.",
        ],
    }
    return {
        "model_input": model_input,
        "verification": {
            "sources": verification_sources,
            "source_hashes": {key: value["text_sha256"] for key, value in verification_sources.items()},
        },
        "selection": {
            "basis": "reviewed exact-anchor windows from developmental fixture",
            "developmental_biased": True,
            "heldout": False,
            "anchor_count": sum(len(values) for values in evidence_by_source.values()),
        },
    }


def render_shared_user(packet: Mapping[str, Any]) -> str:
    """Render the exact user payload shared by all three lanes."""

    return (
        "Tässä on yhteinen PAA-lähdepaketti. Älä lisää siihen tietoa:\n"
        + json.dumps(packet["model_input"], ensure_ascii=False, sort_keys=True, indent=2)
        + "\nNoudata järjestelmäkehotteen JSON-sopimusta."
    )


def load_lane_prompt(lane: str) -> str:
    if lane not in LANE_PROMPTS:
        raise ValueError(f"unknown island lane: {lane}")
    return LANE_PROMPTS[lane].read_text(encoding="utf-8")


def _full_sources(packet: Mapping[str, Any]) -> Mapping[str, Mapping[str, Any]]:
    return packet["verification"]["sources"]


def _check_anchors(anchors: Any, packet: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(anchors, list) or not anchors:
        return {"valid": False, "checked": 0, "errors": ["no source anchors"]}
    errors: list[str] = []
    checked = 0
    valid_ids: list[str] = []
    for anchor in anchors:
        if not isinstance(anchor, Mapping):
            errors.append("anchor is not an object")
            continue
        source_id = anchor.get("source_id")
        quote = anchor.get("quote")
        source = _full_sources(packet).get(source_id)
        if not source:
            errors.append(f"unknown source_id: {source_id}")
            continue
        if not isinstance(quote, str) or not quote:
            errors.append(f"empty quote for {source_id}")
            continue
        checked += 1
        if quote not in source["text"]:
            errors.append(f"quote is not exact in {source_id}")
        elif not any(quote in clip["text"] for clip in source.get("clips", [])):
            errors.append(f"quote is outside the supplied context clip for {source_id}")
        else:
            valid_ids.append(source_id)
    return {"valid": bool(valid_ids) and not errors, "checked": checked, "valid_source_ids": valid_ids, "errors": errors}


def source_check_output(output: Mapping[str, Any] | None, packet: Mapping[str, Any]) -> dict[str, Any]:
    """Check every proposal/attack against full frozen source text.

    Exact anchors with no semantic adjudication remain ``UNRESOLVED``.  This
    function never promotes a model interpretation to an admitted finding.
    """

    errors: list[str] = []
    if not isinstance(output, Mapping):
        return {
            "output_valid": False, "errors": ["output is not an object"],
            "terminal_state": "INSUFFICIENT_EVIDENCE", "candidate_rows": [], "attack_rows": [],
        }
    try:
        jsonschema.Draft202012Validator(island_output_schema()).validate(output)
    except jsonschema.ValidationError as exc:
        errors.append(str(exc).splitlines()[0])
    state = output.get("terminal_state")
    if state not in TERMINAL_STATES:
        errors.append("invalid terminal state")
    candidate_rows: list[dict[str, Any]] = []
    candidates = output.get("candidate_findings", []) if isinstance(output.get("candidate_findings", []), list) else []
    for index, candidate in enumerate(candidates, 1):
        check = _check_anchors(candidate.get("source_anchors") if isinstance(candidate, Mapping) else None, packet)
        candidate_rows.append({
            "row_id": str(candidate.get("candidate_id") or f"candidate-{index}"),
            "kind": "candidate_finding",
            "proposal": deepcopy(candidate),
            "source_valid": check["valid"],
            "anchor_check": check,
            "disposition": "UNRESOLVED" if check["valid"] else "REJECTED",
            "disposition_rationale": "Exact anchor checked; semantic source review still required." if check["valid"] else "; ".join(check["errors"]),
            "alternative": candidate.get("alternative") if isinstance(candidate, Mapping) else "",
            "missing_fact": candidate.get("missing_fact") if isinstance(candidate, Mapping) else "",
            "review_burden": candidate.get("review_burden") if isinstance(candidate, Mapping) else "HIGH",
            "material_added_distinction": None,
        })
    attacks = output.get("attacks", []) if isinstance(output.get("attacks", []), list) else []
    for index, attack in enumerate(attacks, 1):
        check = _check_anchors(attack.get("source_anchors") if isinstance(attack, Mapping) else None, packet)
        attack_rows = {
            "row_id": str(attack.get("attack_id") or f"attack-{index}"),
            "kind": "attack_or_repair",
            "proposal": deepcopy(attack),
            "source_valid": check["valid"],
            "anchor_check": check,
            "disposition": "UNRESOLVED" if check["valid"] else "REJECTED",
            "disposition_rationale": "Exact anchor checked; semantic source review still required." if check["valid"] else "; ".join(check["errors"]),
            "alternative": attack.get("alternative") if isinstance(attack, Mapping) else "",
            "missing_fact": attack.get("missing_fact") if isinstance(attack, Mapping) else "",
            "review_burden": attack.get("review_burden") if isinstance(attack, Mapping) else "HIGH",
            "material_added_distinction": None,
        }
        candidate_rows.append(attack_rows)
    return {
        "output_valid": not errors,
        "errors": errors,
        "terminal_state": state if state in TERMINAL_STATES else "INSUFFICIENT_EVIDENCE",
        "direct_answer_fi": output.get("direct_answer_fi", ""),
        "assumptions": output.get("assumptions", []),
        "stopping_note": output.get("stopping_note", ""),
        "candidate_rows": candidate_rows,
        "attack_rows": [row for row in candidate_rows if row["kind"] == "attack_or_repair"],
        "candidate_denominator": len(candidates),
        "attack_denominator": len(attacks),
        "row_denominator": len(candidate_rows),
    }


def _case_by_id(cases: list[dict[str, Any]], case_id: str) -> dict[str, Any]:
    for case in cases:
        if case.get("case_id") == case_id:
            return case
    raise KeyError(case_id)


def load_development_cases(case_ids: tuple[str, ...] = CASE_ORDER) -> list[dict[str, Any]]:
    cases = load_reviewed_cases(SOURCE_SLICE_PATH, REVIEW_PATH)
    selected = [_case_by_id(cases, case_id) for case_id in case_ids]
    if not selected:
        raise ValueError("at least one island case is required")
    return selected


async def run_island_experiment(
    *,
    case_ids: tuple[str, ...] = CASE_ORDER,
    output_path: Path = DEFAULT_RUN_PATH,
    cache_dir: Path | None = None,
    base_url: str | None = None,
    max_tokens: int = 4000,
    timeout: float = 900,
) -> dict[str, Any]:
    """Run three blinded lanes per selected developmental case sequentially."""

    ensure_dirs()
    cases = load_development_cases(case_ids)
    # This comparison is intentionally serial even when the parent process
    # inherited a wider setting for another local-model workload.
    os.environ["PAA_LLM_MAX_INFLIGHT"] = "1"
    client = LocalLLMClient(base_url, cache_dir=cache_dir or (PACKAGE_ROOT.parent / "data" / "llm_islands_cache"), timeout=timeout)
    started = time.monotonic()
    receipts: list[dict[str, Any]] = []
    packets: dict[str, dict[str, Any]] = {}
    try:
        manifest = await client.discover()
        for case_index, case in enumerate(cases, 1):
            case_id = case["case_id"]
            packet = build_source_packet(case, case_ref=f"CASE_{case_index}")
            packets[case_id] = packet
            shared_user = render_shared_user(packet)
            shared_sha = digest(shared_user)
            for lane in LANE_PROMPTS:
                system = load_lane_prompt(lane)
                receipt = await client.request(
                    f"paa-island:{lane}:case-{case_index}", system, shared_user,
                    schema=island_output_schema(), max_tokens=max_tokens,
                )
                parsed = receipt.get("parsed") if receipt.get("status") == "OK" else None
                checked = source_check_output(parsed, packet)
                receipts.append({
                    "case_id": case_id,
                    "case_ref": f"CASE_{case_index}",
                    "lane": lane,
                    "shared_input_sha256": shared_sha,
                    "system_prompt_sha256": digest(system),
                    "request_id": receipt.get("request_id"),
                    "receipt_status": receipt.get("status"),
                    "cache_hit": receipt.get("cache_hit", False),
                    "model": receipt.get("model") or manifest,
                    "usage": receipt.get("usage", {}),
                    "elapsed_seconds": receipt.get("elapsed_seconds"),
                    "raw_content": receipt.get("content", ""),
                    "parsed_output": parsed,
                    "source_check": checked,
                })
        artifact = {
            "artifact_version": "paa.llm_islands.bounded.v1",
            "run_id": "paa-llm-islands-20261007",
            "started_at": datetime.now(UTC).isoformat(),
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "model_manifest": manifest,
            "concurrency": 1,
            "lane_names": list(LANE_PROMPTS),
            "case_order": list(case_ids),
            "selection": {
                "developmental_biased": True,
                "heldout": False,
                "basis": "four existing source-reviewed inquiry fixtures selected before this model run",
                "reviewed_anchors_used_for_context": True,
                "claims_review_labels_and_answer_standards_in_generation_input": False,
                "same_original_context_per_case": True,
            },
            "prompt_hashes": {lane: digest(load_lane_prompt(lane)) for lane in LANE_PROMPTS},
            "source_packets": {
                case_id: {
                    "model_input_sha256": digest(render_shared_user(packet)),
                    "source_hashes": packet["verification"]["source_hashes"],
                    "selection": packet["selection"],
                }
                for case_id, packet in packets.items()
            },
            "receipts": receipts,
            "evaluation": summarize_receipts(receipts),
            "adjudication_status": "SOURCE_CHECKED_PENDING_SEPARATE_SEMANTIC_REVIEW",
            "stopping_rule": "stop after this bounded pass unless a targeted pass adds a new source-verified material distinction; missing-fact blockers are not prompt failures",
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
        summary_path = output_path.with_suffix(".md")
        summary_path.write_text(render_summary(artifact), encoding="utf-8")
        return artifact
    finally:
        await client.close()


def summarize_receipts(receipts: list[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [receipt.get("source_check", {}) for receipt in receipts]
    all_rows = [row for checked in rows for row in checked.get("candidate_rows", [])]
    candidates = sum(int(checked.get("candidate_denominator", 0)) for checked in rows)
    attacks = sum(int(checked.get("attack_denominator", 0)) for checked in rows)
    exact_valid = sum(bool(row.get("source_valid")) for row in all_rows)
    dispositions = {name: sum(row.get("disposition") == name for row in all_rows) for name in DISPOSITIONS}
    return {
        "receipt_count": len(receipts),
        "ok_receipts": sum(receipt.get("receipt_status") == "OK" for receipt in receipts),
        "invalid_or_failed_receipts": sum(receipt.get("receipt_status") != "OK" for receipt in receipts),
        "candidate_denominator": candidates,
        "attack_denominator": attacks,
        "all_row_denominator": len(all_rows),
        "exact_anchor_valid_rows": exact_valid,
        "exact_anchor_valid_rate": exact_valid / len(all_rows) if all_rows else None,
        "initial_dispositions": dispositions,
        "semantic_admission": "NONE; exact source anchoring is not semantic validation",
    }


def apply_source_review(artifact: Mapping[str, Any], decisions: Mapping[str, Any]) -> dict[str, Any]:
    """Apply a separately authored source-review ledger to a run artifact.

    ``decisions`` is keyed by ``case_id|lane|row_id``.  Every generated row
    must have a decision; omissions are an error so the denominator cannot be
    silently pruned.  The reviewer may mark a row accepted as a source-valid
    candidate, but this remains an internal review result and never becomes a
    public admission automatically.
    """

    result = deepcopy(dict(artifact))
    rows: list[dict[str, Any]] = []
    expected: set[str] = set()
    for receipt in result.get("receipts", []):
        case_id, lane = receipt.get("case_id"), receipt.get("lane")
        for row in receipt.get("source_check", {}).get("candidate_rows", []):
            row_id = str(row.get("row_id"))
            key = f"{case_id}|{lane}|{row_id}"
            expected.add(key)
            decision = decisions.get(key)
            if not isinstance(decision, Mapping):
                raise TypeError(f"missing source-review decision: {key}")
            disposition = decision.get("disposition")
            if disposition not in DISPOSITIONS:
                raise ValueError(f"invalid source-review disposition for {key}: {disposition}")
            row["disposition"] = disposition
            row["disposition_rationale"] = decision.get("rationale") or row.get("disposition_rationale", "")
            row["alternative"] = decision.get("alternative", row.get("alternative", ""))
            row["missing_fact"] = decision.get("missing_fact", row.get("missing_fact", ""))
            row["review_burden"] = decision.get("review_burden", row.get("review_burden", "HIGH"))
            row["material_added_distinction"] = decision.get("material_added_distinction", False)
            row["review_method"] = decision.get("review_method", "SOURCE_READ_AGAINST_FROZEN_FULL_TEXT")
            rows.append(row)
    unexpected = sorted(key for key in set(decisions) - expected if not key.startswith("_"))
    if unexpected:
        raise ValueError(f"source-review decisions have unknown rows: {unexpected[:3]}")
    counts = {name: sum(row.get("disposition") == name for row in rows) for name in DISPOSITIONS}
    material = [row for row in rows if row.get("material_added_distinction") is True]
    result["evaluation"]["source_review"] = {
        "reviewed_row_denominator": len(rows),
        "reviewed_dispositions": counts,
        "material_added_distinction_count": len(material),
        "material_added_distinction_rows": [row.get("row_id") for row in material],
        "reviewer": "CODEX_SOURCE_READING_2026-10-07",
        "method": "manual_source_read_against_frozen_full_text_and_exact_quote_checks",
        "independence": "AI source review; not human inter-rater gold. Paired same-model lanes are not independent validation.",
        "public_admission": False,
    }
    result["source_review"] = {
        "status": "COMPLETE_INTERNAL_SOURCE_REVIEW",
        "reviewer": "CODEX_SOURCE_READING_2026-10-07",
        "method": "manual_source_read_against_frozen_full_text_and_exact_quote_checks",
        "decision_denominator": len(rows),
        "all_generated_rows_decided": len(rows) == len(expected),
        "public_admission": False,
        "case_terminal_states": decisions.get("_case_terminal_states", {}),
        "case_synthesis": decisions.get("_case_synthesis", {}),
        "material_distinction_rule": "A fact or bounded missing-fact distinction is material only when it changes the consequential answer beyond the direct source-first baseline; prose variation and duplicate anchors are not material.",
    }
    # The control metadata is not itself a row decision and is accepted only
    # as a convenience for the annotation command.
    result["source_review"]["case_terminal_states"] = decisions.get("_case_terminal_states", {})
    result["adjudication_status"] = "SOURCE_REVIEW_COMPLETE_NO_PUBLIC_ADMISSION"
    return result


def render_summary(artifact: Mapping[str, Any]) -> str:
    evaluation = artifact["evaluation"]
    lines = [
        "# PAA island comparison — bounded developmental run",
        "",
        "This receipt compares three blinded framings on the same original source clips. It is a developmental, selection-biased control run, not held-out gold or a public admission.",
        "",
        f"- Model: `{artifact['model_manifest'].get('model_id')}`; concurrency: `{artifact['concurrency']}`",
        f"- Cases: {', '.join(artifact['case_order'])}",
        f"- Receipts: {evaluation['ok_receipts']}/{evaluation['receipt_count']} OK",
        f"- Candidate denominator: {evaluation['candidate_denominator']}; attack denominator: {evaluation['attack_denominator']}",
        f"- Exact-anchor-valid rows: {evaluation['exact_anchor_valid_rows']}/{evaluation['all_row_denominator']}",
        "- Initial disposition: exact anchors remain `UNRESOLVED` pending separate semantic source review; invalid anchors are `REJECTED`.",
        "",
        "## Lane × case receipts",
        "",
        "| Case | Lane | Status | Candidates | Attacks | Exact anchors | Terminal |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    if artifact.get("evaluation", {}).get("source_review"):
        review = artifact["evaluation"]["source_review"]
        review_line = next(index for index, line in enumerate(lines) if line.startswith("- Initial disposition:"))
        lines[review_line] = (
            f"- Separate source review: {review['reviewed_row_denominator']} rows; "
            f"accepted {review['reviewed_dispositions']['ACCEPTED']}, "
            f"rejected {review['reviewed_dispositions']['REJECTED']}, "
            f"unresolved {review['reviewed_dispositions']['UNRESOLVED']}; "
            f"material added distinctions {review['material_added_distinction_count']}; "
            "no public admission."
        )
    for receipt in artifact["receipts"]:
        checked = receipt["source_check"]
        rows = checked.get("candidate_rows", [])
        lines.append(
            f"| {receipt['case_id']} | {receipt['lane']} | {receipt['receipt_status']} | "
            f"{checked.get('candidate_denominator', 0)} | {checked.get('attack_denominator', 0)} | "
            f"{sum(bool(row.get('source_valid')) for row in rows)} | {checked.get('terminal_state')} |"
        )
    lines.extend([
        "",
        "## Process checklist instantiated for this bounded PAA run",
        "",
        "- Phase 1: done — four cases selected before generation from the existing reviewed fixture; selection is explicitly developmental/biased.",
        "- Phase 2: done — frozen source packets loaded and exact source hashes checked; no new retrieval or held-out fixture access.",
        "- Phase 3: done — question/scope grounding retained; claims, review labels, answer standards, titles and selection rationale excluded from model input.",
        "- Phase 3.5: done for the adapted lenses — 0th/1st-order text distinction, context-preservation (actor/channel/date), frameless serendipity, fair adversarial repair, and telemetry/evaluation/intervention separation where relevant; non-applicable lenses were not forced into the narrow documentary task.",
        "- Phase 4: done — complete candidate and attack denominator retained before contraction; each row has source validity, disposition, alternative, missing fact and review burden fields.",
        "- Stopping rule: no further pass is planned unless separate review finds a targeted, source-verified material distinction; missing source facts stop the prompt loop.",
        "",
        "## Interpretation boundary",
        "",
        "Model agreement is not truth. A paired same-model critic is not independent gold. The JSON receipt is a proposed/review artifact only; use the separate source-review section in the final report before any public Finnish answer.",
        "",
    ])
    source_review = artifact.get("source_review") or {}
    synthesis = source_review.get("case_synthesis") or {}
    if synthesis:
        lines.extend(["## Source-reviewed bounded synthesis (internal; no public admission)", ""])
        for case_id in artifact.get("case_order", []):
            entry = synthesis.get(case_id)
            terminal = (source_review.get("case_terminal_states") or {}).get(case_id, {})
            if not entry:
                continue
            lines.extend([
                f"### {case_id}",
                "",
                f"- Reviewed terminal: `{entry.get('reviewed_terminal_state', terminal.get('reviewed_terminal_state', 'UNRESOLVED'))}`",
                f"- Suomenkielinen rajattu vastaus: {entry.get('answer_fi', '')}",
                f"- Avoimeksi jää: {entry.get('unresolved_fi', '')}",
                "- Source-quote contract:",
            ])
            for anchor in entry.get("source_quote_contract", []):
                quote = str(anchor.get("quote", "")).replace("\n", " ")
                lines.append(f"  - `{anchor.get('source_id')}` — “{quote}”")
            lines.append("")
        lines.extend([
            "## Full source-review row ledger",
            "",
            "Every generated candidate and attack remains in this denominator; `ACCEPTED` means source-reviewed internal candidate/attack only, not publication.",
            "",
            "| Case | Lane | Row | Kind | Exact anchor | Disposition | Material | Rationale |",
            "|---|---|---|---|---:|---|---:|---|",
        ])
        for receipt in artifact.get("receipts", []):
            for row in receipt.get("source_check", {}).get("candidate_rows", []):
                rationale = str(row.get("disposition_rationale", "")).replace("|", "/").replace("\n", " ")
                lines.append(
                    f"| {receipt.get('case_id')} | {receipt.get('lane')} | {row.get('row_id')} | "
                    f"{row.get('kind')} | {'yes' if row.get('source_valid') else 'no'} | "
                    f"{row.get('disposition')} | {'yes' if row.get('material_added_distinction') else 'no'} | {rationale} |"
                )
        lines.extend([
            "",
            "The material flags are lane-level rows. Deduplicated case-level findings are: RAI 3 baseline / 0 added by frameless or contrarian lanes; infectious 3 baseline + 1 frameless relation distinction; climate 2 baseline / 0 added; adult 2 baseline + 1 frameless transition-basis distinction.",
            "",
        ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="run the bounded local-model comparison")
    parser.add_argument("--annotate", action="store_true", help="apply a separately authored source-review ledger")
    parser.add_argument("--output", type=Path, default=DEFAULT_RUN_PATH)
    parser.add_argument("--review", type=Path, help="source-review decision JSON for --annotate")
    parser.add_argument("--max-tokens", type=int, default=4000)
    parser.add_argument("--case", dest="cases", action="append", choices=CASE_ORDER)
    args = parser.parse_args()
    if args.annotate:
        if not args.review:
            parser.error("--annotate requires --review")
        artifact = json.loads(args.output.read_text(encoding="utf-8"))
        decisions = json.loads(args.review.read_text(encoding="utf-8"))
        annotated = apply_source_review(artifact, decisions)
        args.output.write_text(json.dumps(annotated, ensure_ascii=False, indent=2), encoding="utf-8")
        args.output.with_suffix(".md").write_text(render_summary(annotated), encoding="utf-8")
        print(json.dumps(annotated["evaluation"], ensure_ascii=False, indent=2))
        return
    if not args.run:
        parser.error("pass --run or --annotate")
    artifact = asyncio.run(run_island_experiment(
        case_ids=tuple(args.cases or CASE_ORDER), output_path=args.output, max_tokens=args.max_tokens,
    ))
    print(json.dumps(artifact["evaluation"], ensure_ascii=False, indent=2))


__all__ = [
    "CASE_ORDER",
    "DISPOSITIONS",
    "LANE_PROMPTS",
    "TERMINAL_STATES",
    "build_source_packet",
    "island_output_schema",
    "load_development_cases",
    "load_lane_prompt",
    "render_shared_user",
    "render_summary",
    "run_island_experiment",
    "source_check_output",
    "summarize_receipts",
]


if __name__ == "__main__":
    main()

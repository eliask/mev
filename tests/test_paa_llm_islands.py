from __future__ import annotations

import json
from pathlib import Path

import jsonschema

from paa.llm_islands import (
    CASE_ORDER,
    LANE_PROMPTS,
    build_source_packet,
    island_output_schema,
    load_development_cases,
    load_lane_prompt,
    render_shared_user,
    source_check_output,
    summarize_receipts,
)

FIXTURES = Path("paa/contracts/fixtures")


def test_island_selection_is_four_reviewed_developmental_cases():
    cases = load_development_cases()
    assert tuple(case["case_id"] for case in cases) == CASE_ORDER
    assert len(cases) == 4


def test_model_packet_excludes_review_labels_and_answer_expectations():
    case = load_development_cases((CASE_ORDER[0],))[0]
    packet = build_source_packet(case, case_ref="CASE_1")
    model_text = json.dumps(packet["model_input"], ensure_ascii=False)
    assert case["question"]["text"] in model_text
    assert case["question"]["scope"] in model_text
    assert "answer_standard" not in model_text
    assert "selection_basis" not in model_text
    assert "claims" not in model_text
    assert "unknowns" not in model_text
    assert case["title"] not in model_text
    assert packet["selection"]["developmental_biased"] is True
    assert packet["selection"]["heldout"] is False
    assert packet["verification"]["source_hashes"]


def test_all_lanes_use_identical_shared_input_but_unique_framing_prompts():
    case = load_development_cases((CASE_ORDER[1],))[0]
    packet = build_source_packet(case, case_ref="CASE_2")
    users = [render_shared_user(packet) for _ in LANE_PROMPTS]
    assert len(set(users)) == 1
    prompts = [load_lane_prompt(lane) for lane in LANE_PROMPTS]
    assert len(set(prompts)) == 3
    assert all("lähde" in prompt.lower() for prompt in prompts)


def _valid_output(packet):
    source_id = next(iter(packet["verification"]["sources"]))
    quote = packet["verification"]["sources"][source_id]["clips"][0]["anchors"][0]
    return {
        "terminal_state": "RELATED",
        "direct_answer_fi": "Lähde osoittaa rajatun käsittelyn, mutta ei toimeenpanoa.",
        "assumptions": ["Vain näkyvä lähdepaketti arvioidaan."],
        "candidate_findings": [{
            "candidate_id": "c1", "distinction": "Dokumentoitu käsittely.",
            "source_anchors": [{"source_id": source_id, "quote": quote}],
            "actor_or_channel": "Toimielin", "date_basis": "Asiakirjan päivämäärä",
            "terminal_relevance": "Rajaa kysymyksen vastausta.",
            "alternative": "Käsittely ei ole hyväksyminen.", "missing_fact": "Lopullinen vaikutus.",
            "review_burden": "LOW", "review_note": "Tarkista merkityssisältö.",
            "public_answer_fi": "Asiakirja osoittaa käsittelyn, ei vaikutusta.",
        }],
        "attacks": [],
        "stopping_note": "Ei muuta lähdepakettia.",
    }


def test_source_check_keeps_exact_anchor_candidate_unresolved():
    case = load_development_cases((CASE_ORDER[1],))[0]
    packet = build_source_packet(case, case_ref="CASE_2")
    output = _valid_output(packet)
    jsonschema.Draft202012Validator(island_output_schema()).validate(output)
    checked = source_check_output(output, packet)
    assert checked["output_valid"] is True
    assert checked["candidate_denominator"] == 1
    assert checked["candidate_rows"][0]["source_valid"] is True
    assert checked["candidate_rows"][0]["disposition"] == "UNRESOLVED"
    assert checked["candidate_rows"][0]["material_added_distinction"] is None


def test_source_check_rejects_fabricated_quote_and_preserves_denominator():
    case = load_development_cases((CASE_ORDER[1],))[0]
    packet = build_source_packet(case, case_ref="CASE_2")
    output = _valid_output(packet)
    output["candidate_findings"][0]["source_anchors"][0]["quote"] = "Tätä ei ole lähteessä."
    checked = source_check_output(output, packet)
    assert checked["candidate_denominator"] == 1
    assert checked["candidate_rows"][0]["disposition"] == "REJECTED"
    assert checked["candidate_rows"][0]["source_valid"] is False
    assert "not exact" in checked["candidate_rows"][0]["disposition_rationale"]


def test_summary_preserves_candidate_and_attack_denominators():
    receipts = [{
        "receipt_status": "OK",
        "source_check": {
            "candidate_denominator": 2, "attack_denominator": 1,
            "candidate_rows": [
                {"source_valid": True, "disposition": "UNRESOLVED"},
                {"source_valid": False, "disposition": "REJECTED"},
                {"source_valid": True, "disposition": "UNRESOLVED"},
            ],
        },
    }]
    summary = summarize_receipts(receipts)
    assert summary["candidate_denominator"] == 2
    assert summary["attack_denominator"] == 1
    assert summary["all_row_denominator"] == 3
    assert summary["exact_anchor_valid_rows"] == 2

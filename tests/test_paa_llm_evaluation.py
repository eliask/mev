"""Contract tests for the AI source-read evaluation fixtures."""

from __future__ import annotations

import json
from pathlib import Path

from paa.llm_evaluation import (
    evaluate_relation_predictions,
    evaluate_semantic_predictions,
    load_jsonl,
    load_relation_gold,
    load_semantic_gold,
    public_relation_records,
    public_semantic_records,
)

FIXTURES = Path("paa/contracts/fixtures")


def test_load_jsonl_preserves_unicode_line_separators_inside_json_strings(tmp_path: Path):
    path = tmp_path / "unicode.jsonl"
    path.write_text(json.dumps({"text": "ensimmäinen\u2028toinen\u2029kolmas"}) + "\n", encoding="utf-8")

    assert load_jsonl(path) == [{"text": "ensimmäinen\u2028toinen\u2029kolmas"}]


def test_real_semantic_gold_has_frozen_development_and_blind_heldout_slices():
    rows = load_semantic_gold()
    assert len(rows) == 60
    assert sum(row["split"] == "development" for row in rows) == 30
    assert sum(row["split"] == "heldout" for row in rows) == 30
    assert len({row["document"]["document_id"] for row in rows}) == 60
    assert {row["document"]["source_year"] for row in rows} == {2011, 2023}
    assert {row["document"]["language"] for row in rows} >= {"fi", "sv"}
    assert {"PERSONAL_ACTION_COMMITMENT", "POLICY_DESIDERATUM", "REPORTED_SPEECH"} <= {
        row["gold"]["primary_type"] for row in rows
    }
    for row in rows:
        document = row["document"]
        proposition = row["gold"]["propositions"][0]
        assert proposition["source_quote"] in document["source_text"]
        assert row["adjudication"]["origin"] == "REAL_DB_SOURCE_TEXT_INDEPENDENT_REVIEW"
        assert "retrieval" in row["adjudication"]["basis"]
        assert "semantic precision" in row["adjudication"]["independence_caveat"]


def test_public_semantic_inputs_do_not_leak_gold_or_stratification():
    rows = public_semantic_records("heldout")
    assert len(rows) == 30
    assert all("gold" not in row and "adjudication" not in row for row in rows)
    assert all("selection" not in row for row in rows)
    assert all("source_text" in row["document"] for row in rows)
    assert all("source_quote" not in row["document"] for row in rows)


def test_perfect_semantic_predictions_report_all_requested_metrics():
    rows = load_semantic_gold()
    predictions = {
        row["record_id"]: {"propositions": [row["gold"]["propositions"][0]]}
        for row in rows
    }
    report = evaluate_semantic_predictions(rows, predictions)
    assert report["missing_record_count"] == 0
    assert report["type_accuracy"]["accuracy"] == 1
    assert report["exact_anchor_coverage"]["accuracy"] == 1
    assert report["critical_field_accuracy"]["accuracy"] == 1
    assert report["field_accuracy"]["negation"]["accuracy"] == 1
    assert report["field_accuracy"]["condition"]["accuracy"] == 1
    assert report["field_accuracy"]["deadline_quote"]["accuracy"] == 1
    assert report["field_accuracy"]["deadline"]["accuracy"] == 1
    assert report["abstention_count"] == 0


def test_missing_and_abstained_records_are_visible():
    rows = load_semantic_gold()[:3]
    predictions = {
        rows[0]["record_id"]: {"abstain": True, "abstention_reason": "insufficient context"},
    }
    report = evaluate_semantic_predictions(rows, predictions)
    assert report["missing_record_count"] == 2
    assert report["abstention_count"] == 1
    assert report["unknown_prediction_count"] == 1
    assert set(report["missing_record_ids"]) == {
        rows[1]["record_id"],
        rows[2]["record_id"],
    }


def test_relation_fixture_is_400_real_pairs_with_actor_and_terminal_fields():
    rows = load_relation_gold()
    assert len(rows) == 400
    assert sum(row["split"] == "development" for row in rows) == 200
    assert sum(row["split"] == "heldout" for row in rows) == 200
    assert len({row["pair_id"] for row in rows}) == 400
    assert len({row["source"]["document_id"] for row in rows}) == 80
    assert len({row["object"]["object_id"] for row in rows}) == 40
    assert {
        row["gold"]["phenomenon"]
        for row in rows
    } >= {
        "same_policy",
        "same_vocabulary_compound_different_policy",
        "broad_vs_own_act",
        "changed_scope_or_date",
    }
    assert any(row["gold"]["actor_eligibility"] == "MATCHED_AUTHOR" for row in rows)
    assert any(row["gold"]["actor_eligibility"] == "COSIGNER_ONLY" for row in rows)
    assert all(
        field in row["gold"]
        for row in rows
        for field in ("actor_eligibility", "time_eligibility", "domain_relation", "terminal_eligibility")
    )
    assert all(row["source"]["source_quote"] for row in rows)
    assert all(row["object"]["object_quote"] for row in rows)


def test_relation_public_inputs_and_perfect_report():
    rows = load_relation_gold()
    public = public_relation_records("heldout")
    assert len(public) == 200
    assert all("gold" not in row and "adjudication" not in row for row in public)
    predictions = {row["pair_id"]: row["gold"] for row in rows}
    report = evaluate_relation_predictions(rows, predictions)
    assert report["missing_pair_count"] == 0
    assert report["field_accuracy"]["status"]["accuracy"] == 1
    assert report["field_accuracy"]["terminal_eligibility"]["accuracy"] == 1

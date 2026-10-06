"""Source-only MeV case packets and warning-response review controls."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from paa.mev_cases import (
    MevCaseError,
    build_case,
    build_review,
    canonical_text,
    read_jsonl,
    read_only_connection,
    sha256_text,
    source_record,
)

FIXTURES = Path(__file__).parents[1] / "paa/contracts/fixtures"


def _make_index(path: Path) -> None:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE he (
            canonical_id TEXT PRIMARY KEY,
            year INTEGER,
            number INTEGER,
            title TEXT,
            content TEXT
        );
        CREATE TABLE expert_statement (
            statement_id TEXT PRIMARY KEY,
            he_id TEXT,
            committee TEXT,
            expert_title TEXT,
            date TEXT,
            lakitutka_id TEXT,
            content TEXT
        );
        CREATE TABLE committee_report (
            report_id TEXT PRIMARY KEY,
            he_id TEXT,
            committee TEXT,
            report_type TEXT,
            title TEXT,
            date TEXT,
            lakitutka_id TEXT,
            tunnus TEXT,
            content TEXT,
            decision_text TEXT
        );
        INSERT INTO he VALUES (
            'he-test-2024', 2024, 9, 'Test proposal',
            '<p>Proposal text.</p><p>Requires a safeguard.</p>'
        );
        INSERT INTO expert_statement VALUES (
            'expert-test', 'he-test-2024', 'Test committee',
            'HE 9/2024 vp TeV 01.02.2024 Test expert Asiantuntijalausunto',
            '2024-02-01', 'expert-doc',
            '<p>The warning asks for a safeguard.</p>'
        );
        INSERT INTO committee_report VALUES (
            'committee-test', 'he-test-2024', 'Test committee', 'Valiokunnan mietintö', 'Test report', '2024-03-01',
            'committee-doc', 'TEST 9/2024 vp',
            '<p>The committee records the safeguard.</p>',
            'Committee decision: accepted with amendment.'
        );
        """
    )
    connection.commit()
    connection.close()


def _question() -> dict:
    return {
        "question_id": "q-test",
        "text": "What documentary response is recorded?",
        "target_scope": "the declared safeguard",
        "period": "2024 source dates",
        "comparison": "proposal, expert statement and committee report",
        "evidence_needed": ["complete source text", "exact response quote"],
        "valid_outputs": ["documentary response", "unresolved implementation"],
        "unknowns": ["later implementation"],
        "practical_use": "separate record from effect",
    }


def test_canonical_text_preserves_exact_hashable_source_view() -> None:
    value = "<p>One&nbsp;line.</p><p>Two lines.</p>"
    text = canonical_text(value)
    assert text == "One line.\nTwo lines."
    assert sha256_text(text) == sha256_text(text)


def test_read_only_source_record_keeps_raw_and_canonical_hashes(tmp_path: Path) -> None:
    db = tmp_path / "index.sqlite"
    _make_index(db)
    connection = read_only_connection(db)
    try:
        record = source_record(connection, "he", "he-test-2024")
        assert record["raw_text"].startswith("<p>Proposal")
        assert record["text"] == "Proposal text.\nRequires a safeguard."
        assert record["raw_sha256"] == sha256_text(record["raw_text"])
        assert record["text_sha256"] == sha256_text(record["text"])
        assert record["source_url"].endswith("government-proposal/2024/9/fin@")
        assert record["document_identifier"] is None
        assert record["matter_title"] == "Test proposal"
        with pytest.raises(sqlite3.OperationalError):
            connection.execute("CREATE TABLE should_not_exist (value TEXT)")
        expert = source_record(connection, "expert_statement", "expert-test")
        assert expert["document_identifier"] == "expert-doc"
        assert expert["committee_name"] == "Test committee"
        assert expert["committee_code"] == "TeV"
        assert expert["expert_title"].startswith("HE 9/2024 vp TeV")
        assert expert["title"] == expert["expert_title"]
        committee = source_record(connection, "committee_report", "committee-test")
        assert committee["document_identifier"] == "TEST 9/2024 vp"
        assert committee["committee_name"] == "Test committee"
        assert committee["committee_code"] == "TEST"
        assert committee["report_type"] == "Valiokunnan mietintö"
        assert committee["title"] == "TEST 9/2024 vp"
        assert committee["matter_title"] == "Test report"
        assert committee["decision_text"] == "Committee decision: accepted with amendment."
        assert committee["decision_text_sha256"] == sha256_text(committee["decision_text"])
    finally:
        connection.close()


def test_build_case_is_source_packet_only_and_rejects_legacy_labels(tmp_path: Path) -> None:
    db = tmp_path / "index.sqlite"
    _make_index(db)
    spec = {
        "episode_id": "mev-test",
        "episode_kind": "TEST_RESPONSE",
        "question_contract": _question(),
        "source_refs": [
            {"table": "he", "record_id": "he-test-2024"},
            {"table": "committee_report", "record_id": "committee-test"},
        ],
        "transformations": [
            {
                "transformation_id": "test-change",
                "kind": "PROPOSAL_TO_COMMITTEE_WORDING_AMENDMENT",
                "before": {"table": "he", "record_id": "he-test-2024"},
                "after": {"table": "committee_report", "record_id": "committee-test"},
                "scope": "safeguard",
                "before_quote": "Requires a safeguard.",
                "after_field": "decision_text",
                "after_quote": "Committee decision: accepted with amendment.",
            }
        ],
    }
    connection = read_only_connection(db)
    try:
        case = build_case(connection, spec)
        assert case["admission_state"] == "PROPOSED"
        assert case["semantic_state"] == "SOURCE_PACKET_ONLY"
        assert case["coverage"]["legacy_detector_labels_admitted"] is False
        assert len(case["sources"]) == 2
        assert case["sources"][0]["raw_text"]
        assert case["transformations"][0]["status"] == "PROPOSED"
        assert case["transformations"][0]["before_evidence"]["quote"] == "Requires a safeguard."
        assert case["transformations"][0]["after_evidence"]["field"] == "decision_text"
        assert case["transformations"][0]["after_evidence"]["quote"] == "Committee decision: accepted with amendment."

        bad = dict(spec)
        bad["detector_labels"] = ["REPAIR"]
        with pytest.raises(MevCaseError, match="legacy detector"):
            build_case(connection, bad)
    finally:
        connection.close()


def test_build_review_requires_exact_quote_and_separate_research_state(tmp_path: Path) -> None:
    db = tmp_path / "index.sqlite"
    _make_index(db)
    spec = {
        "review_id": "review-test",
        "episode_id": "mev-test",
        "warning_control": "REAL_REPAIR",
        "source_refs": [{"table": "expert_statement", "record_id": "expert-test"}],
        "quotes": [
            {
                "table": "expert_statement",
                "record_id": "expert-test",
                "quote": "The warning asks for a safeguard.",
            }
        ],
        "rationale": "The complete cited source records the warning; no implementation claim is made.",
    }
    connection = read_only_connection(db)
    try:
        review = build_review(connection, spec)
        assert review["review_state"] == "RESEARCH_ONLY"
        assert review["reviewer"] == "AI_SOURCE_READING"
        assert review["review_method"] == "AI_SOURCE_READING"
        assert review["template_label_assignment"] is False
        assert review["quotes"][0]["quote_sha256"] == sha256_text(review["quotes"][0]["quote"])

        bad = json.loads(json.dumps(spec))
        bad["quotes"][0]["quote"] = "The warning is absent."
        with pytest.raises(MevCaseError, match="exact substring"):
            build_review(connection, bad)
    finally:
        connection.close()


def test_warning_review_fixture_covers_four_controls_without_gold_labels() -> None:
    rows = read_jsonl(FIXTURES / "mev_warning_review_specs.jsonl")
    assert {row["warning_control"] for row in rows} == {
        "REAL_REPAIR",
        "REASONED_REBUTTAL",
        "APPARENT_GAP",
        "FALSE_GAP",
    }
    assert all(row["rationale"] for row in rows)
    assert not any("detector_labels" in row or "gold_label" in row for row in rows)


def test_read_jsonl_preserves_unicode_line_separators_inside_source_text(tmp_path: Path) -> None:
    path = tmp_path / "source.jsonl"
    text = "before\u2028between\u2029after"
    path.write_text(json.dumps({"text": text}, ensure_ascii=False) + "\n", encoding="utf-8")

    assert read_jsonl(path) == [{"text": text}]


def test_transfer_holdout_is_frozen_before_model_input_and_has_distinct_matters() -> None:
    manifest = json.loads((FIXTURES / "mev_transfer_heldout_manifest.json").read_text(encoding="utf-8"))
    rows = read_jsonl(FIXTURES / "mev_transfer_source_slices.jsonl")
    discovery = read_jsonl(FIXTURES / "mev_discovery_specs.jsonl")
    discovery_matters = {
        str(ref["record_id"])
        for row in discovery
        for ref in row.get("source_refs", [])
        if ref.get("table") == "he"
    }
    assert manifest["selection_state"] == "FROZEN_BEFORE_MODEL_TUNING"
    assert manifest["model_or_gold_present_at_freeze"] is False
    assert len(rows) == 6 == manifest["transfer_episode_count"]
    assert {row["episode_id"] for row in rows} == {row["episode_id"] for row in manifest["episodes"]}
    assert not (set(manifest["development_matter_keys"]) & {row["matter_key"] for row in manifest["episodes"]})
    assert set(manifest["development_matter_keys"]) == discovery_matters
    fixture_bytes = (FIXTURES / "mev_transfer_source_slices.jsonl").read_bytes()
    assert hashlib.sha256(fixture_bytes).hexdigest() == manifest["source_snapshot"]["source_fixture_sha256"]
    for row in rows:
        assert row["admission_state"] == "PROPOSED"
        assert all(source["text"] and source["text_sha256"] for source in row["sources"])
        assert "warning_control" not in row
        assert not any("gold_label" in source for source in row["sources"])

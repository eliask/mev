"""Frozen source-sequence tests for PAA decision episodes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from paa.decision_episodes import (
    EPISODE_KIND,
    EpisodeError,
    build_episodes_from_objects,
    build_written_question_episode,
    build_written_question_episode_from_db,
)
from paa.question_ledger import ANSWER_KIND, QUESTION_KIND, normalize_question_records, parse_question_row
from paa.store import connect

FIXTURES = Path(__file__).parents[1] / "paa" / "contracts" / "fixtures"


def _rows(name: str) -> list[dict]:
    payload = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
    return [dict(zip(payload["columnNames"], raw)) for raw in payload["rowData"]]


def _normalised_fixture() -> tuple[dict, dict, dict]:
    records = [
        parse_question_row(row, retrieved_at="2026-10-06T00:00:00+00:00")
        for name in ("vaski_kk1_2023_rows.json", "vaski_kkv1_2023_rows.json")
        for row in _rows(name)
    ]
    result = normalize_question_records(records)
    question = next(item for item in result["objects"] if item["kind"] == QUESTION_KIND)
    answer = next(item for item in result["objects"] if item["kind"] == ANSWER_KIND)
    return question, answer, result


def test_real_question_answer_episode_preserves_registered_sequence_and_residuals() -> None:
    question, answer, result = _normalised_fixture()
    episode = build_written_question_episode(
        question,
        answer,
        evidence=result["evidence"],
        coverage={
            "coverage_id": "frozen-question-slice",
            "source_id": "SRC-EDUSKUNTA-VASKI",
            "kind": "WRITTEN_QUESTION_REGISTER",
            "state": "PARTIAL_DECLARED_SLICE",
            "complete": False,
            "limitations": ["One frozen KK/KKV slice; not the whole register."],
        },
    )

    assert episode["kind"] == EPISODE_KIND
    assert episode["matter_id"] == "KK 1/2023 vp"
    assert episode["episode_state"] == "ANSWERED_INSTITUTIONALLY"
    assert [row["stage"] for row in episode["source_sequence"]] == [
        "QUESTION_DOCUMENT",
        "QUESTION_SUBMITTED",
        "GOVERNMENT_RESPONSE_DOCUMENT",
        "GOVERNMENT_RESPONSE_RECEIVED",
        "GOVERNMENT_RESPONSE_ANNOUNCED",
    ]
    assert {row["date"] for row in episode["source_sequence"]} == {"2023-04-20", "2023-05-09", "2023-05-30"}
    assert {row["record_locator"] for row in episode["source_records"]} == {
        "VaskiData/Id=252304",
        "VaskiData/Id=252305",
        "VaskiData/Id=252306",
        "VaskiData/Id=252633",
    }
    actors = {(actor["name"], actor["role"], actor.get("person_id")) for actor in episode["actors"]}
    assert ("Jussi Saramo", "AUTHOR", "1400") in actors
    assert ("Tytti Tuppurainen", "RESPONDENT", None) in actors
    assert episode["institutional_disposition"]["state"] == "ANSWERED"
    assert episode["institutional_disposition"]["date"] == "2023-05-09"
    assert episode["legal_state"]["state"] == "NOT_ASSESSED"
    assert episode["policy_implementation_state"]["state"] == "NOT_ASSESSED"
    assert episode["causal_outcome_state"]["state"] == "NOT_ASSESSED"
    residual_codes = {item["code"] for item in episode["residuals"]}
    assert {"POLICY_IMPLEMENTATION_NOT_SOURCED", "CAUSAL_OUTCOME_NOT_SOURCED", "LEGAL_EFFECT_NOT_ASSESSED"} <= residual_codes
    assert "SOURCE_COVERAGE_LIMITED" in residual_codes
    assert all(ref in {item["evidence_id"] for item in result["evidence"]} for ref in episode["evidence_ids"])


def test_missing_answer_is_not_joined_to_another_matter() -> None:
    question, _answer, result = _normalised_fixture()
    episode = build_written_question_episode(question, evidence=result["evidence"])

    # The procedural record can truthfully say that a response was received,
    # while the separate answer object is not loaded.  The residual keeps that
    # distinction visible instead of turning it into a missing-answer claim.
    assert episode["episode_state"] == "ANSWERED_INSTITUTIONALLY"
    assert any(item["code"] == "ANSWER_OBJECT_NOT_LOADED" for item in episode["residuals"])
    assert all(row["stage"] != "GOVERNMENT_RESPONSE_DOCUMENT" for row in episode["source_sequence"])


def test_mismatched_answer_is_rejected() -> None:
    question, answer, _result = _normalised_fixture()
    mismatched = dict(answer, matter_id="KK 2/2023 vp")
    with pytest.raises(EpisodeError, match="matter mismatch"):
        build_written_question_episode(question, mismatched)


def test_db_reader_is_read_only_and_uses_record_locator_for_event_evidence(tmp_path: Path) -> None:
    question, answer, result = _normalised_fixture()
    conn = connect(tmp_path / "episodes.sqlite")
    try:
        for obj in (question, answer):
            conn.execute(
                "INSERT INTO official_objects(object_id, json) VALUES (?, ?)",
                (obj["object_id"], json.dumps(obj, ensure_ascii=False)),
            )
        for item in result["evidence"]:
            conn.execute(
                "INSERT INTO evidence(evidence_id, json) VALUES (?, ?)",
                (item["evidence_id"], json.dumps(item, ensure_ascii=False)),
            )
        conn.execute(
            "INSERT INTO source_coverage(coverage_id, json) VALUES (?, ?)",
            (
                "frozen-question-slice",
                json.dumps({
                    "coverage_id": "frozen-question-slice",
                    "source_id": "SRC-EDUSKUNTA-VASKI",
                    "kind": "WRITTEN_QUESTION_REGISTER",
                    "state": "PARTIAL_DECLARED_SLICE",
                    "complete": False,
                }),
            ),
        )
        conn.commit()
        changes_before = conn.total_changes
        episode = build_written_question_episode_from_db(conn, "KK 1/2023 vp")
        assert conn.total_changes == changes_before
        assert [row["stage"] for row in episode["source_sequence"]] == [
            "QUESTION_DOCUMENT",
            "QUESTION_SUBMITTED",
            "GOVERNMENT_RESPONSE_DOCUMENT",
            "GOVERNMENT_RESPONSE_RECEIVED",
            "GOVERNMENT_RESPONSE_ANNOUNCED",
        ]
        assert episode["coverage"]["state"] == "DECLARED"
    finally:
        conn.close()


def test_batch_builder_only_uses_explicit_answer_object_link() -> None:
    question, answer, result = _normalised_fixture()
    objects = {question["object_id"]: question, answer["object_id"]: answer}
    episodes = build_episodes_from_objects(objects, evidence=result["evidence"])
    assert len(episodes) == 1
    assert episodes[0]["episode_id"] == build_written_question_episode(question, answer, evidence=result["evidence"])["episode_id"]

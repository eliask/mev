from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from paa.legal_state import (
    LegalStateError,
    build_legal_state_receipt,
    build_legal_state_receipt_from_capture,
    load_lawvm_capture,
    validate_legal_state_receipt,
)

FIXTURE = Path(__file__).parents[1] / "paa/contracts/fixtures/lawvm_2003_1281_section11.json"
STAFFING_FIXTURE = Path(__file__).parents[1] / "paa/contracts/fixtures/lawvm_2012_980_section3_15a_2020.json"


def _capture() -> dict:
    return load_lawvm_capture(FIXTURE)


def _staffing_capture() -> dict:
    return load_lawvm_capture(STAFFING_FIXTURE)


def _receipt(capture: dict, *, source_views=None) -> dict:
    return build_legal_state_receipt(
        capture["replay"],
        capture["oracle"],
        capture["reconcile"],
        source_views=capture["source_views"] if source_views is None else source_views,
        source_artifacts=capture.get("source_artifacts"),
        invocation={"commands": capture["commands"], "fixture_kind": capture["fixture_kind"]},
    )


def test_real_lawvm_capture_is_source_bound_and_keeps_temporal_divergence() -> None:
    capture = _capture()
    receipt = _receipt(capture)
    captured_receipt = build_legal_state_receipt_from_capture(capture)

    assert receipt["statute_id"] == "2003/1281"
    assert receipt["legal_address"] == "chapter:2/section:11"
    assert receipt["comparison"]["state"] == "DIVERGENT"
    assert receipt["comparison"]["verdict"] == "DISAGREE"
    assert receipt["comparison"]["divergence_class"] == "temporal"
    assert receipt["comparison"]["agree_ratio"] == 0.9133
    assert receipt["question"]["answer"]["state"] == "DIVERGENT_SOURCE_VIEWS"
    assert len(receipt["question"]["answer"]["decisive_evidence_ids"]) == 2
    assert any(item["code"] == "OPERATIVE_LEGAL_STATE" for item in receipt["question"]["unknowns"])
    assert receipt["coverage"]["state"] == "COMPLETE_FOR_DECLARED_INPUTS"
    assert receipt["operative"] == {
        "verified": False,
        "state": "NOT_OPERATIVE_VERIFIED",
        "basis": "LAWVM_RECONSTRUCTION_AND_COMPARISON_ONLY",
    }
    assert receipt["legal_effects"] == {
        "state": "NOT_ASSESSED",
        "items": [],
        "note": "No operative, implementation, service, outcome or causal effect is inferred.",
    }
    assert {row["plane"] for row in receipt["source_views"]} == {"replay", "oracle"}
    assert {row["raw_hash_role"] for row in receipt["source_views"]} == {
        "source_xml_bytes_sha256",
        "consolidated_oracle_xml_bytes_sha256",
    }
    assert all(row["locator"].startswith("finlex://") for row in receipt["source_views"])
    assert any(item["code"] == "ORACLE_CUTOFF_PRECEDES_AS_OF" for item in receipt["residuals"])
    assert any(item["code"] == "RECONCILIATION_NOT_OPERATIVE_AUTHORITY" for item in receipt["residuals"])
    assert validate_legal_state_receipt(receipt) is True
    assert captured_receipt["receipt_id"] == receipt["receipt_id"]


def test_real_source_quotes_and_hashes_survive_adapter() -> None:
    receipt = _receipt(_capture())
    by_plane = {row["plane"]: row for row in receipt["evidence"]}

    assert "ajoneuvoverolain" in by_plane["replay"]["quote"]
    assert by_plane["replay"]["quote_hash"] == "e664e9975d7af8df25cef29a21b9a008b81dd62004d47db076225bfa3b31e56c"
    assert by_plane["replay"]["quote_role"] == "operation_source_raw_text"
    assert "Käyttövoimavero" in by_plane["oracle"]["quote"]
    assert by_plane["oracle"]["quote_role"] == "lawvm_oracle_text_view"
    assert by_plane["replay"]["raw_sha256"] == "905b8d23943229c4521aecf74e6a7d9279a93c0fc5039454236c8829db896380"
    assert by_plane["oracle"]["raw_sha256"] == "9cdd278e46b4b2fbe3bc0fd6f82caa99dc8b8836e6e6d041dc924d6771a87177"


def test_real_staffing_case_records_2020_rai_provision_and_later_oracle_difference() -> None:
    capture = _staffing_capture()
    receipt = _receipt(capture)

    assert receipt["statute_id"] == "2012/980"
    assert receipt["legal_address"] == "chapter:3/section:15a"
    assert receipt["as_of"] == "2020-12-31"
    assert receipt["comparison"]["state"] == "DIVERGENT"
    assert receipt["comparison"]["divergence_class"] == "editorial"
    assert receipt["comparison"]["agree_ratio"] == 0.4522
    assert "Kunnan on käytettävä" in capture["replay"]["text"]["rendered"]
    assert "Hyvinvointialueen on käytettävä" in receipt["reconstruction"]["oracle"]["source_view"]["quote"]
    assert receipt["evidence"][0]["raw_sha256"] == "1c61506a5379afc516f049cd07903de5dfdad6e80cdc3ffe65d9609ed1b9c9bf"
    assert receipt["evidence"][1]["raw_sha256"] == "0fd59a2d1c5e13b2fbd44cb21988e7bf16b8c2369788279acf9733c8fdc4416f"
    assert receipt["operative"]["verified"] is False
    assert any(item["code"] == "ORACLE_CUTOFF_PRECEDES_AS_OF" for item in receipt["residuals"]) is False
    assert any(item["code"] == "TEMPORAL_SOURCE_NOT_VALIDATED" for item in receipt["residuals"])
    assert any(item["code"] == "OPERATIVE_PROVISION_SCOPE_NOT_VALIDATED" for item in receipt["residuals"])
    assert receipt["comparison"]["verdict"] == "DISAGREE"
    assert receipt["comparison"]["temporal"]["status"] == "BLOCKED"
    assert receipt["reconstruction"]["oracle"]["temporal"]["status"] == "FUTURE_SOURCE_NOT_VALIDATED"
    assert receipt["source_artifacts"][0]["temporal"]["status"] == "DATE_NOT_AFTER_AS_OF"
    assert receipt["source_artifacts"][0]["scope_temporal"]["status"] == "FUTURE_SCOPE_DEADLINE"
    assert receipt["coverage"]["state"] == "PARTIAL"
    assert receipt["question"]["scope"]["statute_id"] == "2012/980"


def test_full_source_artifact_hash_is_rechecked() -> None:
    import copy

    capture = _staffing_capture()
    capture = copy.deepcopy(capture)
    capture["source_artifacts"][0]["raw_text"] += "\n"
    with pytest.raises(LegalStateError, match="raw_sha256 does not match"):
        _receipt(capture)


def test_missing_oracle_raw_hash_is_partial_and_never_promoted() -> None:
    capture = _capture()
    source_views = [
        dict(capture["source_views"][0]),
        {"plane": "oracle", "locator": capture["source_views"][1]["locator"]},
    ]
    receipt = _receipt(capture, source_views=source_views)

    assert receipt["coverage"]["state"] == "PARTIAL"
    assert "oracle_source_view" in receipt["coverage"]["missing"]
    assert any(item["code"] == "SOURCE_VIEW_RAW_HASH_MISSING" for item in receipt["residuals"])
    assert receipt["operative"]["verified"] is False


def test_invalid_source_hash_is_rejected_without_guessing() -> None:
    capture = _capture()
    source_views = [dict(row) for row in capture["source_views"]]
    source_views[0]["raw_sha256"] = "not-a-hash"

    with pytest.raises(LegalStateError, match="lowercase SHA-256"):
        _receipt(capture, source_views=source_views)


def test_validator_rejects_operative_claim_and_keeps_fixture_json_plain() -> None:
    receipt = _receipt(_capture())
    tampered = copy.deepcopy(receipt)
    tampered["operative"]["verified"] = True
    with pytest.raises(LegalStateError, match="operative legal verification"):
        validate_legal_state_receipt(tampered)

    # The capture and receipt are both serialisable without custom encoders;
    # this is the boundary used by SQLite/static-build callers.
    json.dumps(receipt, ensure_ascii=False)

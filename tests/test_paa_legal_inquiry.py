"""The RAI follow-up keeps adopted-law evidence separate from legal truth."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from paa.inquiry_cases import anchor, load_reviewed_cases
from paa.legal_inquiry import (
    LegalInquiryError,
    augment_rai_inquiry,
    validate_legal_inquiry,
)
from paa.legal_state import LegalStateError, load_lawvm_capture

FIXTURES = Path(__file__).parents[1] / "paa/contracts/fixtures"
INQUIRY_FIXTURE = FIXTURES / "mev_cases_source_slices.jsonl"
REVIEW_FIXTURE = FIXTURES / "inquiry_case_reviews.json"
CAPTURE_FIXTURE = FIXTURES / "lawvm_2012_980_section3_15a_2020.json"


def _packet() -> dict:
    return load_reviewed_cases(INQUIRY_FIXTURE, REVIEW_FIXTURE)[0]


def _capture() -> dict:
    return load_lawvm_capture(CAPTURE_FIXTURE)


def test_real_rai_packet_reaches_adopted_law_bounded_stage() -> None:
    augmented = augment_rai_inquiry(_packet(), _capture())

    assert validate_legal_inquiry(augmented)
    inquiry = augmented["legal_inquiry"]
    assert inquiry["stage"] == "ADOPTED_LAW_BOUNDED"
    assert inquiry["status"] == "DOCUMENTED_BOUNDED"
    assert inquiry["statute_id"] == "2012/980"
    assert inquiry["amending_statute_id"] == "2020/565"
    assert inquiry["latest_start_date"] == "2023-04-01"
    assert inquiry["legal_address"] == "chapter:3/section:15a"

    legal_claim = next(row for row in augmented["claims"] if row.get("claim_id") == inquiry["claim_id"])
    assert legal_claim["state"] == "SOURCE_REVIEWED"
    assert "Muutossäädös 2020/565" in legal_claim["text"]
    assert "viimeistään" in legal_claim["text"]
    assert "ei ratkaise aikaisempaa" in legal_claim["text"]
    assert len(legal_claim["evidence_ids"]) == 3
    assert {row["view_role"] for row in augmented["sources"] if row.get("view_role")} == {
        "ACT_AMENDMENT_SCOPE",
        "OPERATIVE_PROVISION_WORDING",
        "COMMENCEMENT_LATEST_START",
    }


def test_full_raw_source_human_text_and_deadline_are_hash_bound() -> None:
    augmented = augment_rai_inquiry(_packet(), _capture())
    source_rows = {row["view_role"]: row for row in augmented["sources"] if row.get("view_role")}
    evidence_rows = {row["quote_role"]: row for row in augmented["evidence"] if row.get("quote_role")}

    assert source_rows["ACT_AMENDMENT_SCOPE"]["raw_sha256"] == (
        "1c61506a5379afc516f049cd07903de5dfdad6e80cdc3ffe65d9609ed1b9c9bf"
    )
    assert source_rows["ACT_AMENDMENT_SCOPE"]["document_identifier"] == "2020/565"
    assert source_rows["ACT_AMENDMENT_SCOPE"]["url"] == "https://www.finlex.fi/fi/laki/alkup/2020/20200565"
    assert "565/2020" in source_rows["ACT_AMENDMENT_SCOPE"]["title"]
    assert source_rows["ACT_AMENDMENT_SCOPE"]["raw_text"].startswith("<akomaNtoso")
    assert "RAI-arviointivälineistön käyttäminen" in source_rows["OPERATIVE_PROVISION_WORDING"]["text"]
    assert "viimeistään 1 päivänä huhtikuuta 2023" in source_rows["COMMENCEMENT_LATEST_START"]["text"]
    assert source_rows["COMMENCEMENT_LATEST_START"]["raw_quote_binding"] == {
        "basis": "NORMALIZED_XML_TEXT_SURFACE",
        "surface": "entry_into_force",
        "raw_sha256": "1c61506a5379afc516f049cd07903de5dfdad6e80cdc3ffe65d9609ed1b9c9bf",
    }
    assert evidence_rows["COMMENCEMENT_LATEST_START"]["quote_sha256"] == (
        "53802aed8d853189c2c9859e14577a67484c81366d9767ee1389bad4c24d7668"
    )
    assert all(row["capture_locator"] == "finlex://sd/2020/565/fin/main.xml" for row in evidence_rows.values())


def test_lawvm_receipt_is_full_but_explicitly_comparison_only() -> None:
    augmented = augment_rai_inquiry(_packet(), _capture())
    artifact = augmented["legal_comparison_artifacts"][0]
    receipt = artifact["receipt"]

    assert artifact["role"] == "COMPARISON_ONLY_NOT_LEGAL_TRUTH"
    assert artifact["receipt_id"] == receipt["receipt_id"]
    assert receipt["operative"]["verified"] is False
    assert receipt["legal_effects"]["state"] == "NOT_ASSESSED"
    assert receipt["comparison"]["temporal"]["status"] == "BLOCKED"
    assert any(row["code"] == "TEMPORAL_SOURCE_NOT_VALIDATED" for row in receipt["residuals"])
    # The full receipt remains JSON data for the store/static compiler.
    json.dumps(augmented, ensure_ascii=False)


def test_augmentation_is_idempotent_and_does_not_mutate_inputs() -> None:
    packet = _packet()
    capture = _capture()
    packet_before = copy.deepcopy(packet)
    capture_before = copy.deepcopy(capture)

    first = augment_rai_inquiry(packet, capture)
    second = augment_rai_inquiry(first, capture)

    assert first == second
    assert packet == packet_before
    assert capture == capture_before


def test_packet_and_capture_context_cannot_be_silently_mixed() -> None:
    packet = _packet()
    packet["episode_id"] = "different-episode"
    with pytest.raises(LegalInquiryError, match="conflicting episode identities"):
        augment_rai_inquiry(packet, _capture())


def test_missing_episode_or_documentary_source_context_is_rejected() -> None:
    packet = _packet()
    packet.pop("episode_id")
    packet.pop("source_episode_id")
    with pytest.raises(LegalInquiryError, match="explicit episode identity"):
        augment_rai_inquiry(packet, _capture())

    packet = _packet()
    capture = _capture()
    capture["case"]["documentary_context"]["source_ids"] = ["unrelated-source"]
    with pytest.raises(LegalInquiryError, match="no shared documentary source identity"):
        augment_rai_inquiry(packet, capture)


def test_changed_capture_raw_source_is_rejected_before_admission() -> None:
    capture = _capture()
    capture["source_artifacts"][0]["raw_text"] += "\nchanged"
    with pytest.raises(LegalStateError, match="raw_sha256 does not match"):
        augment_rai_inquiry(_packet(), capture)


def test_deadline_quote_cannot_be_rewritten_without_matching_raw_xml() -> None:
    capture = _capture()
    commencement = capture["source_artifacts"][0]["commencement"]
    commencement["date"] = "2024-04-01"
    commencement["quote"] = commencement["quote"].replace("2023", "2024")
    with pytest.raises(LegalInquiryError, match="differs from the raw XML surface"):
        augment_rai_inquiry(_packet(), capture)


def test_tampered_comparison_or_missing_earlier_unknown_is_rejected() -> None:
    augmented = augment_rai_inquiry(_packet(), _capture())
    tampered = copy.deepcopy(augmented)
    tampered["legal_comparison_artifacts"][0]["receipt"]["operative"]["verified"] = True
    with pytest.raises(LegalInquiryError, match="comparison artifact"):
        validate_legal_inquiry(tampered)

    tampered = copy.deepcopy(augmented)
    tampered["legal_inquiry"]["unknowns"] = ["implementation remains unknown"]
    with pytest.raises(LegalInquiryError, match="earlier-operativity"):
        validate_legal_inquiry(tampered)


def test_validator_rebuilds_xml_surface_after_display_hashes_are_rewritten() -> None:
    augmented = augment_rai_inquiry(_packet(), _capture())
    tampered = copy.deepcopy(augmented)
    source = next(row for row in tampered["sources"] if row.get("view_role") == "ACT_AMENDMENT_SCOPE")
    original_ref = next(row for row in tampered["evidence"] if row["source_id"] == source["source_id"])
    source["text"] = "spoofed operation wording"
    source["text_sha256"] = hashlib.sha256(source["text"].encode()).hexdigest()
    source["canonical_text_sha256"] = source["text_sha256"]
    ref = anchor(source, source["text"])
    ref.update({key: original_ref[key] for key in ("quote_role", "capture_locator", "capture_source_id", "evidence_basis", "raw_quote_binding", "canonical_text_sha256")})
    ref["quote_sha256"] = source["text_sha256"]
    tampered["evidence"] = [ref if row["evidence_id"] == original_ref["evidence_id"] else row for row in tampered["evidence"]]
    tampered["legal_inquiry"]["evidence_ids"] = [ref["evidence_id"] if key == original_ref["evidence_id"] else key for key in tampered["legal_inquiry"]["evidence_ids"]]
    claim = next(row for row in tampered["claims"] if row.get("claim_id") == tampered["legal_inquiry"]["claim_id"])
    claim["evidence_ids"] = list(tampered["legal_inquiry"]["evidence_ids"])
    claim["review"]["source_versions"][source["source_id"]] = source["text_sha256"]
    with pytest.raises(LegalInquiryError, match="display text differs from regenerated raw XML"):
        validate_legal_inquiry(tampered)

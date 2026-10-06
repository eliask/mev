"""The therapy-guarantee successor is bound to enacted primary text.

This is a focused producer-to-packet/browser check for the new legal endpoint.
It deliberately does not claim that the statutory obligations were delivered
in practice.
"""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from paa.case_site import write_cases
from paa.inquiry_cases import InquiryError, load_reviewed_cases, validate_case

FIXTURES = Path(__file__).parents[1] / "paa" / "contracts" / "fixtures"
SOURCE_FIXTURE = FIXTURES / "therapy_completion_source_versions.jsonl"
REVIEW_FIXTURE = FIXTURES / "therapy_completion_case_reviews.json"


def _case() -> dict:
    cases = load_reviewed_cases(SOURCE_FIXTURE, REVIEW_FIXTURE)
    assert len(cases) == 1
    return cases[0]


def test_enacted_therapy_bundle_reaches_review_packet_and_browser(tmp_path):
    case = _case()
    assert validate_case(case)
    assert case["case_id"] == "therapy-2024-final-adopted-scope"

    enacted = {
        source["record_id"]: source
        for source in case["sources"]
        if source["record_id"].startswith("act-")
    }
    assert set(enacted) == {
        "act-1107-2024",
        "act-1108-2024",
        "act-1109-2024",
        "act-1110-2024",
    }
    assert all(source["captured_at"] == "2026-10-07T07:14:57Z" for source in enacted.values())
    assert all(source["content_format"] == "AKN_XML" for source in enacted.values())
    assert all(source["raw_sha256"] == __import__("hashlib").sha256(
        source["raw_text"].encode("utf-8")
    ).hexdigest() for source in enacted.values())

    claims = {claim["dimension"]: claim for claim in case["claims"]}
    assert "Terveydenhuollon 28 vuorokauden sääntö" in claims
    assert "Sosiaalihuollon määräaika ja sisältö" in claims
    assert "Säädös ei ole toimitustulos" in claims
    health = claims["Terveydenhuollon 28 vuorokauden sääntö"]["text"]
    assert "tarve on todettu" in health
    assert "ensimmäisestä yhteydenotosta" in health
    assert "poikkeus" in health
    assert len(case["unknowns"]) == 3
    assert any("käytännössä" in unknown["question"] for unknown in case["unknowns"])

    output = tmp_path / "browser-cases"
    write_cases([case], output)
    page = (output / f'{case["case_id"]}.html').read_text(encoding="utf-8")
    assert "Tämä laki tulee voimaan 1 päivänä toukokuuta 2025." in page
    assert "evidence_id" not in page  # labels are rendered, raw field names are not required
    assert all(ref["evidence_id"] in page for ref in case["evidence"])
    assert "palvelujen tosiasiallista käynnistymistä" in page


def test_enacted_source_revision_invalidates_successor_review(tmp_path):
    source_rows = [json.loads(line) for line in SOURCE_FIXTURE.read_text(encoding="utf-8").splitlines()]
    changed = deepcopy(source_rows[0])
    changed["sources"] = deepcopy(changed["sources"])
    final = next(source for source in changed["sources"] if source["record_id"] == "act-1107-2024")
    final["text"] += " Muutettu lähdeversio."
    source_path = tmp_path / "changed.jsonl"
    source_path.write_text(json.dumps(changed, ensure_ascii=False) + "\n", encoding="utf-8")

    with pytest.raises(InquiryError, match="stale"):
        load_reviewed_cases(source_path, REVIEW_FIXTURE)

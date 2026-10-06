"""Successor source-review check for the unresolved Sastamala component.

This is intentionally separate from the original Pirkanmaa case.  It proves
that the new secondary-request evidence reaches the generic packet and browser
without turning a reported request or later membership into a final decision.
"""

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from paa.case_site import write_cases
from paa.inquiry_cases import InquiryError, load_reviewed_cases, validate_case

FIXTURES = Path(__file__).parents[1] / "paa" / "contracts" / "fixtures"
SOURCE_FIXTURE = FIXTURES / "public_beta_person_sastamala_sources.jsonl"
REVIEW_FIXTURE = FIXTURES / "public_beta_person_sastamala_reviews.json"


def _case() -> dict:
    cases = load_reviewed_cases(SOURCE_FIXTURE, REVIEW_FIXTURE)
    assert len(cases) == 1
    return cases[0]


def test_sastamala_successor_reaches_packet_and_browser_without_final_badge(tmp_path):
    case = _case()
    assert validate_case(case)
    assert case["case_id"] == "person-miko-bergbom-sastamala-request-2023"
    assert case["question_id"] == "q-public-beta-person-miko-bergbom-sastamala-request"
    assert case["canonical_target_scope"] == "CITY_COUNCIL_COMPONENT_REQUEST_ONLY"

    sources = {source["source_id"]: source for source in case["sources"]}
    assert set(sources) == {
        "SRC-BERGBOM-PROMISE-2023",
        "SRC-VAALIT-2023",
        "SRC-ALUEVIESTI-20230427",
        "SRC-YLE-20230411",
        "SRC-SASTAMALA-COUNCIL-20241209",
    }
    frozen_sources = {
        source["source_id"]: source
        for source in json.loads(SOURCE_FIXTURE.read_text(encoding="utf-8"))["sources"]
    }
    assert frozen_sources["SRC-ALUEVIESTI-20230427"]["publication_date"] == "2023-04-27"
    assert sources["SRC-SASTAMALA-COUNCIL-20241209"]["event_date"] == "2024-12-09"
    assert sources["SRC-ALUEVIESTI-20230427"]["source_projection"] == (
        "SHORT_SECONDARY_REPORT_PROJECTION"
    )
    assert sources["SRC-SASTAMALA-COUNCIL-20241209"]["source_projection"] == (
        "STRUCTURED_PARTICIPANT_FIELD_PROJECTION"
    )

    quotes = {ref["quote"] for ref in case["evidence"]}
    assert any("pyytää eroa valtuuston jäsenyydestä" in quote for quote in quotes)
    assert "Asia on kaupunginhallituksen käsiteltävänä ja lopulta valtuuston päätettävänä." in quotes
    assert "Kulmala Aulis — jäsen" in quotes
    assert all(claim["state"] == "SOURCE_REVIEWED" for claim in case["claims"])
    assert "ei sisällä Sastamalan" in case["claims"][-1]["text"]
    assert case["unknowns"]

    output = tmp_path / "browser-cases"
    write_cases([case], output)
    page = (output / f'{case["case_id"]}.html').read_text(encoding="utf-8")
    assert "Bergbomin Sastamalan valtuustopaikan pyyntö" in page
    assert "27.4.2023" in page
    assert "Kulmala Aulis" in page
    assert "ei sisällä Sastamalan kaupunginhallituksen käsittelyä" in page
    assert (output / f'{case["case_id"]}.json').exists()


def test_successor_projection_hashes_and_capture_hashes_are_bound():
    row = json.loads(SOURCE_FIXTURE.read_text(encoding="utf-8"))
    assert len(row["sources"]) == 5
    for source in row["sources"]:
        assert hashlib.sha256(source["raw_text"].encode("utf-8")).hexdigest() == source["raw_sha256"]
        assert hashlib.sha256(source["text"].encode("utf-8")).hexdigest() == source["text_sha256"]
        if source['source_id'] != 'SRC-ALUEVIESTI-20230427':
            assert len(source["capture_raw_sha256"]) == 64
    alue = next(source for source in row["sources"] if source["record_id"] == "alueviesti-request-2023-04-27")
    assert 'capture_raw_sha256' not in alue
    assert len(alue['blocked_response_sha256']) == 64
    assert alue['capture_response_status'] == 403
    assert alue["raw_sha256"] != alue["blocked_response_sha256"]
    assert "403" in alue["access_note"]
    later = next(source for source in row["sources"] if source["record_id"] == "sastamala-council-membership-2024-12-09")
    assert later["text"] == "Kulmala Aulis — jäsen"
    assert "päätöspäivää" in later["access_note"]


def test_changed_secondary_projection_requires_successor_review(tmp_path):
    rows = [json.loads(line) for line in SOURCE_FIXTURE.read_text(encoding="utf-8").splitlines()]
    changed = deepcopy(rows[0])
    changed["sources"] = deepcopy(changed["sources"])
    article = next(source for source in changed["sources"] if source["record_id"] == "alueviesti-request-2023-04-27")
    article["text"] += " Muutettu lähdeversio."
    source_path = tmp_path / "changed.jsonl"
    source_path.write_text(json.dumps(changed, ensure_ascii=False) + "\n", encoding="utf-8")

    with pytest.raises(InquiryError, match="stale"):
        load_reviewed_cases(source_path, REVIEW_FIXTURE)

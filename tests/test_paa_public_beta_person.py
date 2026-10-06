"""Focused public-beta producer-to-consumer check for the Bergbom trace.

The fixture deliberately retains only exact quote projections and one explicitly
structured election-field projection from the existing canonical sources. It
does not adjudicate the missing city-council component.
"""

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from paa.case_site import write_cases
from paa.inquiry_cases import InquiryError, load_reviewed_cases, validate_case

FIXTURES = Path(__file__).parents[1] / "paa" / "contracts" / "fixtures"
SOURCE_FIXTURE = FIXTURES / "public_beta_person_sources.jsonl"
REVIEW_FIXTURE = FIXTURES / "public_beta_person_reviews.json"


def _case() -> dict:
    cases = load_reviewed_cases(SOURCE_FIXTURE, REVIEW_FIXTURE)
    assert len(cases) == 1
    return cases[0]


def test_bergbom_public_beta_trace_reaches_review_packet_and_browser(tmp_path):
    case = _case()
    assert validate_case(case)
    assert case["case_id"] == "person-miko-bergbom-pirha-2023"
    assert case["question_id"] == "q-public-beta-person-miko-bergbom-pirha"
    assert case["source_episode_id"] == "public-beta-person-miko-bergbom-pirha-2023"

    sources = {source["source_id"]: source for source in case["sources"]}
    assert set(sources) == {"SRC-YLE-2023-C7", "SRC-VAALIT-2023", "SRC-PIRHA-2023-7933"}

    quotes = {ref["quote"] for ref in case["evidence"]}
    assert any("aluevaltuustossa että kaupunginvaltuustossa" in quote for quote in quotes)
    assert "Miko Bergbom · Pirkanmaa · ehdokasnumero 149 · valittu 2023" in quotes
    assert any("7.4.2023 pyytänyt eroa" in quote for quote in quotes)
    assert any("Aluevaltuusto päätti myöntää" in quote for quote in quotes)
    review = json.loads(REVIEW_FIXTURE.read_text(encoding="utf-8"))[0]
    assert {quote["key"] for quote in review["quotes"]} == {
        "original_promise",
        "election_role_projection",
        "request_2023-04-07",
        "final_grant_2023-06-05",
    }
    election_quote = next(quote for quote in review["quotes"] if quote["key"] == "election_role_projection")
    assert election_quote["quote_kind"] == "STRUCTURED_FIELD_PROJECTION"
    assert election_quote["verbatim"] is False
    assert "ei lähdeasiakirjan yhtenäinen" in election_quote["note_fi"]
    assert case["canonical_target_scope"] == "PARTIAL_COMPONENT"
    assert case["question"]["target_scope"] == "PARTIAL_COMPONENT"

    claims = {claim["dimension"]: claim for claim in case["claims"]}
    assert len(claims) == 4
    assert all(claim["review"]["method"] == "AI_SOURCE_READING" for claim in claims.values())
    assert "Sastamalan" in claims["Vain osittainen komponentti on varmennettu"]["text"]
    assert any("ei ole näyttöä" in unknown["missing_evidence"] or "Tarvitaan lähteeseen sidottu" in unknown["missing_evidence"]
               for unknown in case["unknowns"])

    output = tmp_path / "browser-cases"
    write_cases([case], output)
    page = (output / f'{case["case_id"]}.html').read_text(encoding="utf-8")
    assert case["title"] in page
    assert "molemmat valtuustopaikkansa" in page
    assert "Miko Bergbom · Pirkanmaa · ehdokasnumero 149 · valittu 2023" in page
    assert "7.4.2023" in page and "Aluevaltuusto päätti myöntää" in page
    assert "Sastamalan kaupunginvaltuuston" in page
    assert (output / f'{case["case_id"]}.json').exists()


def test_bergbom_fixture_is_quote_projection_not_full_campaign_or_corpus():
    row = json.loads(SOURCE_FIXTURE.read_text(encoding="utf-8"))
    assert len(row["sources"]) == 3

    promise = next(source for source in row["sources"] if source["record_id"] == "miko-promise-2023-03")
    assert promise["source_projection"] == "QUOTE_ONLY_CAMPAIGN_PROJECTION"
    assert promise["rights_state"] == "PUBLIC_SOURCE_REFERENCE_QUOTE_ONLY_RIGHTS_UNCONFIRMED"
    assert promise["capture_raw_sha256"] == (
        "0ccd5beea7ea8fba1ec02cf33dc198b5a4702aee4b306ac17e07e1c77b0412bc"
    )
    assert promise["raw_sha256"] == hashlib.sha256(promise["raw_text"].encode("utf-8")).hexdigest()
    assert promise["raw_sha256"] != promise["canonical_slice_sha256"]
    assert "Haluan antaa kaiken aikani" not in promise["text"]
    assert "Kun johonkin ryhdyn" not in promise["text"]

    election = next(source for source in row["sources"] if source["record_id"] == "miko-election-role-2023")
    assert election["canonical_evidence_id"] == "candidacy-vaalit-2023-07-149-e"
    assert election["actor_id"] == "mp-1514"
    assert election["source_projection"] == "STRUCTURED_CANDIDACY_FIELDS_PROJECTION"
    assert election["normalization_version"] == "candidacy_field_projection_v1"
    assert election["source_text_role"] == "STRUCTURED_RECORD_VIEW_NOT_VERBATIM_QUOTE"
    assert election["capture_raw_sha256"] == (
        "a0af4a015ac30b432485026ee83a8b6a6525013b6f39735335368d4cc90e9b3a"
    )
    assert election["canonical_slice_sha256"] == "c8d592c105c40d9de83d6c16393b2214298fb750914f06a1d48ac82df65019aa"
    frozen_fields = json.loads(election["raw_text"])
    assert frozen_fields["candidacy_id"] == "vaalit-2023-07-149"
    assert frozen_fields["candidate_number"] == 149
    assert frozen_fields["elected"] == 1
    assert election["raw_sha256"] == hashlib.sha256(election["raw_text"].encode("utf-8")).hexdigest()
    assert election["text_sha256"] == hashlib.sha256(election["text"].encode("utf-8")).hexdigest()

    decision = next(source for source in row["sources"] if source["record_id"] == "miko-pirha-9112-2023")
    assert decision["rights_state"] == "EXCERPT_ALLOWED"
    assert decision["canonical_evidence_ids"] == [
        "pirha-2023-9112-bergbom:request",
        "pirha-2023-9112-bergbom:decision",
    ]
    assert decision["request_date"] == "2023-04-07"
    assert decision["institutional_decision_date"] == "2023-06-05"
    assert decision["target_scope"] == "PARTIAL_COMPONENT"
    assert decision["canonical_slice_sha256"] == decision["text_sha256"]
    assert all("Sanna" not in source["text"] for source in row["sources"])


def test_changed_bergbom_projection_requires_successor_review(tmp_path):
    rows = [json.loads(line) for line in SOURCE_FIXTURE.read_text(encoding="utf-8").splitlines()]
    changed = deepcopy(rows[0])
    changed["sources"] = deepcopy(changed["sources"])
    promise = next(source for source in changed["sources"] if source["record_id"] == "miko-promise-2023-03")
    promise["text"] += " Muutettu lähdeversio."
    source_path = tmp_path / "changed.jsonl"
    source_path.write_text(json.dumps(changed, ensure_ascii=False) + "\n", encoding="utf-8")

    with pytest.raises(InquiryError, match="stale"):
        load_reviewed_cases(source_path, REVIEW_FIXTURE)

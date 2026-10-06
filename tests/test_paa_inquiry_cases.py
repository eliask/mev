"""Review staleness and candidate boundaries for source-to-inquiry pages."""
import json
from copy import deepcopy
from pathlib import Path

import pytest

from paa.case_site import render_case, write_cases
from paa.inquiry_cases import (
    InquiryError,
    anchor,
    compile_case,
    digest,
    load_reviewed_cases,
    source_record,
    validate_case,
)


def packet():
    # Synthetic control: source acquisition is tested separately with real cases.
    source = source_record("control", "The proposal requires condition A. The amendment removes condition A.",
                           url="https://example.org/control", locator="control:1", raw_sha256=digest("raw"), title="Control")
    ref = anchor(source, "The amendment removes condition A.")
    claim = {"dimension": "Ehdon muutos", "text": "Lähdeteksti sanoo ehdon poistuvan.", "state": "SOURCE_REVIEWED",
             "evidence_ids": [ref["evidence_id"]], "review": {"reviewer": "test-review", "method": "SOURCE_READING",
             "rationale": "Explicit documentary statement; no causal inference.", "source_versions": {"control": source["text_sha256"]}}}
    return compile_case(question={"text": "Muuttuiko ehto?", "scope": "Control only.", "answer_standard": "Exact before/after evidence."},
                        sources=[source], evidence=[ref], claims=[claim], title="Control inquiry", selection_basis="Synthetic control",
                        unknowns=[{"question": "Kuka aiheutti muutoksen?", "missing_evidence": "Causal identification absent."}])


def test_versioned_inquiry_to_static_page_and_artifact(tmp_path):
    case = packet()
    assert validate_case(case)
    write_cases([case], tmp_path)
    page = (tmp_path / f'{case["case_id"]}.html').read_text()
    assert "Muuttuiko ehto?" in page
    assert "Lähdeteksti sanoo ehdon poistuvan." in page
    assert case["evidence"][0]["evidence_id"] in page
    assert case["sources"][0]["raw_sha256"] in page
    assert "Kuka aiheutti muutoksen?" in page
    assert (tmp_path / f'{case["case_id"]}.json').exists()


@pytest.mark.parametrize("mutation", ["source", "quote", "review", "dangling", "method", "cause"])
def test_changed_or_candidate_evidence_cannot_support_reviewed_answer(mutation):
    case = deepcopy(packet())
    if mutation == "source":
        case["sources"][0]["text"] += " Changed."
    elif mutation == "quote":
        case["evidence"][0]["quote"] = "Wrong quotation"
    elif mutation == "review":
        case["claims"][0]["review"]["source_versions"]["control"] = digest("old")
    elif mutation == "dangling":
        case["claims"][0]["evidence_ids"] = ["missing"]
    elif mutation == "method":
        case["claims"][0]["review"]["method"] = "KEYWORD_OVERLAP"
    else:
        case["claims"][0]["causal_claim"] = True
    with pytest.raises(InquiryError):
        validate_case(case)


def test_candidate_is_visible_only_as_unreviewed_and_html_is_escaped():
    case = packet()
    case["claims"][0]["state"] = "CANDIDATE"
    case["claims"][0]["text"] = "<script>alert('candidate')</script>"
    page = render_case(case)
    assert "ei vielä tue tarkistettua vastausta" in page
    assert "ei varmennettu päätelmä" in page
    assert "<script>" not in page
    assert "&lt;script&gt;" in page


def test_anchor_requires_explicit_offset_for_duplicate_quote_and_keeps_unique_control():
    text = (
        "MAJORITY REASONS\n"
        "The same sentence appears in both institutional voices.\n"
        "DISSENT 1 REASONS\n"
        "The same sentence appears in both institutional voices."
    )
    source = source_record("duplicate-voice", text, url="https://example.org/voice",
                           locator="voice:1", raw_sha256=digest("voice-raw"), title="Voice control")
    quote = "The same sentence appears in both institutional voices."
    with pytest.raises(InquiryError, match="multiple times"):
        anchor(source, quote)

    second_start = text.index(quote, text.index(quote) + 1)
    selected = anchor(source, quote, start=second_start)
    assert selected["start"] == second_start
    assert selected["end"] == second_start + len(quote)
    assert anchor(source, "MAJORITY REASONS")["start"] == 0


@pytest.mark.parametrize("start", [True, False, -1, 1.5, "1"])
def test_anchor_rejects_non_integer_or_negative_explicit_offsets(start):
    source = source_record("offset-control", "A unique quotation.", url="https://example.org/offset",
                           locator="offset:1", raw_sha256=digest("offset-raw"), title="Offset control")
    with pytest.raises(InquiryError, match="start"):
        anchor(source, "A unique quotation.", start=start)


@pytest.mark.parametrize("start", [len("A unique quotation."), len("A unique quotation.") + 1])
def test_anchor_rejects_explicit_offsets_past_quote_end(start):
    source = source_record("range-control", "A unique quotation.", url="https://example.org/range",
                           locator="range:1", raw_sha256=digest("range-raw"), title="Range control")
    with pytest.raises(InquiryError, match="outside"):
        anchor(source, "A unique quotation.", start=start)


def test_review_loader_explicit_duplicate_offset_survives_packet_and_browser(tmp_path):
    text = (
        "MAJORITY REASONS\n"
        "The same sentence appears in both institutional voices.\n"
        "DISSENT 1 REASONS\n"
        "The same sentence appears in both institutional voices."
    )
    source_row = {
        "record_id": "duplicate-review-source",
        "source_id": "TEST:duplicate-review-source",
        "text": text,
        "text_sha256": digest(text),
        "raw_text": text,
        "raw_sha256": digest(text),
        "source_url": "https://example.org/duplicate-review-source",
        "record_locator": "test:duplicate-review-source",
        "title": "Duplicate review source",
        "source_kind": "COMMITTEE_REPORT",
    }
    source_path = tmp_path / "sources.jsonl"
    source_path.write_text(json.dumps({"episode_id": "duplicate-review-episode", "sources": [source_row]}))
    quote = "The same sentence appears in both institutional voices."
    second_start = text.index(quote, text.index(quote) + 1)
    review = [{
        "case_id": "duplicate-review-case",
        "source_episode_id": "duplicate-review-episode",
        "source_versions": {"duplicate-review-source": digest(text)},
        "title": "Duplicate review binding",
        "question": {"text": "Kumman ääni kuuluu?", "scope": "Two source sections.",
                      "answer_standard": "The selected source occurrence is explicit."},
        "quotes": [{"key": "dissent", "record_id": "duplicate-review-source", "quote": quote,
                    "start": second_start}],
        "claims": [{"dimension": "Lähdekohta", "text": "Valittu katkelma on vastalauseen kohdasta.",
                    "quote_keys": ["dissent"], "rationale": "The retained offset selects the second occurrence."}],
        "unknowns": [], "reviewer": "test-review", "method": "AI_SOURCE_READING",
        "selection_basis": "Ambiguous quote binding control.",
    }]
    review_path = tmp_path / "reviews.json"
    review_path.write_text(json.dumps(review, ensure_ascii=False))

    packets = load_reviewed_cases(source_path, review_path)
    assert len(packets) == 1
    packet = packets[0]
    ref = packet["evidence"][0]
    assert ref["start"] == second_start
    assert ref["quote"] == quote
    write_cases(packets, tmp_path / "browser")
    page = (tmp_path / "browser" / "duplicate-review-case.html").read_text()
    assert ref["evidence_id"] in page
    assert f"Merkit {second_start}–{second_start + len(quote)}" in page


def test_real_frozen_sources_build_database_inquiry_and_page(tmp_path):
    import sqlite3

    from paa.frozen import build_frozen

    build_frozen(root=tmp_path / "frozen")
    conn = sqlite3.connect(tmp_path / "frozen/data/paa.sqlite")
    try:
        case = json.loads(conn.execute("SELECT json FROM inquiry_cases WHERE case_id=?",
                                      ("rai-2020-constitutional-repair",)).fetchone()[0])
        voice = json.loads(conn.execute("SELECT json FROM inquiry_cases WHERE case_id=?",
                                       ("climate-2024-dissent-source-voice",)).fetchone()[0])
        nature = json.loads(conn.execute("SELECT json FROM inquiry_cases WHERE case_id=?",
                                         ("ymvm17-2022-matter-scope",)).fetchone()[0])
    finally:
        conn.close()
    assert len(case["claims"]) == 5
    assert all(claim["review"]["method"] == "AI_SOURCE_READING" for claim in case["claims"])
    assert {s["source_id"].split(":")[-1] for s in case["sources"]} == {
        "he-4-2020", "pevl15-2020", "stvm18-2020", "operation", "provision", "commencement"}
    assert case["legal_inquiry"]["latest_start_date"] == "2023-04-01"
    assert case["legal_comparison_artifacts"][0]["receipt"]["operative"]["verified"] is False
    page = (tmp_path / "frozen/dist/browser/cases/rai-2020-constitutional-repair.html").read_text()
    assert "uusi 15 a §" in page
    assert "tavallisen lain säätämisjärjestyksen edellytyksenä" in page
    assert "Ratkaisevat lähdekatkelmat" in page
    assert "2024-12-05" in page and "2020-12-31" in page
    assert "Takaraja ei osoita" in page
    assert "palvelun järjestäjän ja välineistön tarjoajan tehtävät" in page
    assert case["evidence"][0]["raw_sha256"] in page
    indexed = next(source for source in voice["sources"] if source["record_id"] == "ymvm8-2024")
    official = next(source for source in voice["sources"] if source["record_id"] == "291583")
    assert "Vastalause 1" not in indexed["text"]
    assert "Vastalause 1" in official["text"]
    assert "Vastalause 2" in official["text"]
    assert "SuppeaAllekirjoitusOsa" in official["raw_text"]
    assert official["captured_at"] == "2026-10-07T03:47:52.602734+00:00"
    assert official["normalization_version"] == "xml_document_itertext_tokens_newline_v1"
    assert indexed["text_sha256"] != official["text_sha256"]
    assert {ref["source_id"] for ref in voice["evidence"]} == {indexed["source_id"], official["source_id"]}
    voice_page = (tmp_path / "frozen/dist/browser/cases/climate-2024-dissent-source-voice.html").read_text()
    assert "valiokunnan enemmistön kantana" in voice_page
    assert "kuka tekstin kirjoitti" in voice_page
    assert "Marko Asell" in voice_page
    assert all(ref["evidence_id"] in voice_page for ref in voice["evidence"])
    assert len(nature["claims"]) == 3
    assert nature["sources"][0]["record_id"] == "236013"
    assert nature["sources"][0]["raw_sha256"] == "721ab746f461b72ae1520e6968e84c6ef3f851ee17e12a313082c5be25a09682"
    nature_claims = " ".join(claim["text"] for claim in nature["claims"])
    nature_quotes = " ".join(ref["quote"] for ref in nature["evidence"])
    assert "HE 76/2022 vp" in nature_claims
    assert "LA 19/2021 vp" in nature_claims
    assert "Valiokunta esittää lakiehdotusten hyväksymistä muutettuina" in nature_quotes
    assert "LA 19/2021 vp" in nature_quotes and "lakialoite hylätään" in nature_quotes
    assert "lopullinen" in nature_claims and "lopullisesta" in nature_claims
    nature_page = (tmp_path / "frozen/dist/browser/cases/ymvm17-2022-matter-scope.html").read_text()
    assert "HE 76/2022 vp" in nature_page and "LA 19/2021 vp" in nature_page
    assert "valiokunnan suositus" in nature_page
    assert "Hyväksyikö eduskunta" in nature_page
    assert all(ref["evidence_id"] in nature_page for ref in nature["evidence"])


def test_real_review_is_not_silently_rebound_when_source_changes(tmp_path):
    fixtures = Path(__file__).parents[1] / "paa/contracts/fixtures"
    source = fixtures / "mev_cases_source_slices.jsonl"
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    case = next(row for row in rows if row["episode_id"] == "mev-he4-2020-staffing")
    case["sources"][0]["text"] += " Source updated while reviewed quotes still remain."
    changed = tmp_path / "changed.jsonl"
    changed.write_text("\n".join(json.dumps(row) for row in rows))
    with pytest.raises(InquiryError, match="stale"):
        load_reviewed_cases(changed, fixtures / "inquiry_case_reviews.json")


def test_changed_legal_capture_requires_re_review_even_if_documentary_quotes_match(tmp_path):
    fixtures = Path(__file__).parents[1] / "paa/contracts/fixtures"
    review = tmp_path / "reviews.json"
    review.write_text((fixtures / "inquiry_case_reviews.json").read_text())
    capture = tmp_path / "lawvm_2012_980_section3_15a_2020.json"
    capture.write_text((fixtures / capture.name).read_text() + "\n")
    with pytest.raises(InquiryError, match="legal capture changed"):
        load_reviewed_cases(fixtures / "mev_cases_source_slices.jsonl", review)


@pytest.mark.parametrize("duplicate", ["episode", "record", "review"])
def test_duplicate_source_or_review_cannot_silently_replace_a_case(tmp_path, duplicate):
    fixtures = Path(__file__).parents[1] / "paa/contracts/fixtures"
    source_path = fixtures / "inquiry_official_source_versions.jsonl"
    review_path = fixtures / "inquiry_official_source_reviews.json"
    assert len(load_reviewed_cases(source_path, review_path)) == 2
    rows = [json.loads(line) for line in source_path.read_text().split("\n") if line.strip()]
    reviews = json.loads(review_path.read_text())
    if duplicate == "episode":
        rows.append(deepcopy(rows[0]))
    elif duplicate == "record":
        rows[0]["sources"].append(deepcopy(rows[0]["sources"][0]))
    else:
        reviews.append(deepcopy(reviews[0]))
    changed_source = tmp_path / "sources.jsonl"
    changed_source.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n")
    changed_review = tmp_path / "reviews.json"
    changed_review.write_text(json.dumps(reviews, ensure_ascii=False))
    with pytest.raises(InquiryError, match="duplicate"):
        load_reviewed_cases(changed_source, changed_review)

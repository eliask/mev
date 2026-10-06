"""Source-first implementation/measurement successor for the therapy case."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from paa.case_site import write_cases
from paa.inquiry_cases import InquiryError, load_reviewed_cases, validate_case

FIXTURES = Path(__file__).parents[1] / "paa" / "contracts" / "fixtures"
SOURCE_FIXTURE = FIXTURES / "therapy_implementation_source_versions.jsonl"
REVIEW_FIXTURE = FIXTURES / "therapy_implementation_case_reviews.json"


def _case() -> dict:
    cases = load_reviewed_cases(SOURCE_FIXTURE, REVIEW_FIXTURE)
    assert len(cases) == 1
    return cases[0]


def test_implementation_successor_reaches_packet_and_browser(tmp_path):
    case = _case()
    assert validate_case(case)
    assert case["case_id"] == "therapy-2025-2026-implementation-measurement"
    assert case["question_id"] == "q-therapy-implementation-measurement-2026"

    source_by_id = {source["source_id"]: source for source in case["sources"]}
    assert len(source_by_id) == 4
    assert source_by_id["THL-TERAPIA:GUIDANCE:2026-01"]["response_raw_sha256"] == (
        "0af2f29455b8de2a89549c2cbce65316266a4ea5008d7af709f7bc09e746b8a4"
    )
    assert {
        source["record_id"]: source["captured_at"]
        for source in source_by_id.values()
    } == {
        "helsinki-monitoring-2026-09-01": "2026-10-08T18:00:51Z",
        "stm-announcement-2026-06-10": "2026-10-08T18:01:29Z",
        "stm-interim-report-2026-06-10": "2026-10-08T18:01:30Z",
        "thl-guidance-2026-01": "2026-10-08T18:01:30Z",
    }
    assert all(source["coverage_basis"].endswith("SHORT_PROJECTION") for source in source_by_id.values())
    thl_text = source_by_id["THL-TERAPIA:GUIDANCE:2026-01"]["text"]
    assert thl_text.startswith("Takuu koskee ")
    assert "koska ryhmämuotoisten terapiatakuun interventioiden" in thl_text
    assert "Y58 Terapiatakuun interventio Kun interventio toteutuu oman organisaation sisällä" in thl_text
    report_text = source_by_id["STM-TERAPIA:INTERIM-REPORT:2026-06-10"]["text"]
    assert "Enimmäisajan (28 vrk) toteutuminen yhteensä 310 30 291 43." not in report_text

    claims = {claim["dimension"]: claim for claim in case["claims"]}
    assert len(claims) == 8
    assert "91 prosenttia" in claims["STM:n julkaisema havainto rajatusta kirjausaineistosta"]["text"]
    assert "eivät kaikkien oikeutettujen henkilöiden kansallinen toimitusaste" in (
        claims["STM:n julkaisema havainto rajatusta kirjausaineistosta"]["text"]
    )
    assert "ensimmäiseen toteutuneeseen interventiokäyntiin" in claims[
        "Mittarin kellon alku, loppu ja palvelukanava"
    ]["text"]
    assert "28 vuorokauden ja sosiaalihuollon yhden kuukauden" in claims[
        "Tietopohjan kattavuus rajoittaa kansallista johtopäätöstä"
    ]["text"]
    assert "310 havaintoa 340:stä" in claims["YTHS:n osuus yhdistetystä taulukosta"]["text"]
    assert "178 havaintoa 334:stä" in claims["YTHS:n osuus yhdistetystä taulukosta"]["text"]
    assert "1.5.2027" in case["question"]["text"]

    output = tmp_path / "browser-cases"
    write_cases([case], output)
    page = (output / f'{case["case_id"]}.html').read_text(encoding="utf-8")
    assert "Mitä terapiatakuun julkaistu toimeenpanotieto" in page
    assert "91 prosenttia" in page and "87 prosenttia" in page
    assert "1.5.2027" in page
    assert "Y58" in page and "Y78" in page and "Y29" in page
    assert "kansallinen toimitusaste" in page
    assert all(ref["evidence_id"] in page for ref in case["evidence"])
    assert (output / f'{case["case_id"]}.json').exists()


def test_source_projection_hashes_and_full_capture_hashes_are_distinct_and_bound():
    row = json.loads(SOURCE_FIXTURE.read_text(encoding="utf-8"))
    for source in row["sources"]:
        assert hashlib.sha256(source["raw_text"].encode("utf-8")).hexdigest() == source["raw_sha256"]
        assert hashlib.sha256(source["text"].encode("utf-8")).hexdigest() == source["text_sha256"]
        assert len(source["capture_raw_sha256"]) == 64
        assert source["source_projection"].startswith("SHORT_")
    thl = next(source for source in row["sources"] if source["record_id"] == "thl-guidance-2026-01")
    assert thl["raw_sha256"] != thl["capture_raw_sha256"]


def test_changed_implementation_source_requires_successor_review(tmp_path):
    rows = [json.loads(line) for line in SOURCE_FIXTURE.read_text(encoding="utf-8").splitlines()]
    changed = deepcopy(rows[0])
    report = next(source for source in changed["sources"] if source["record_id"] == "stm-interim-report-2026-06-10")
    report["text"] += " Muutettu lähdeversio."
    source_path = tmp_path / "changed.jsonl"
    source_path.write_text(json.dumps(changed, ensure_ascii=False) + "\n", encoding="utf-8")

    with pytest.raises(InquiryError, match="stale"):
        load_reviewed_cases(source_path, REVIEW_FIXTURE)

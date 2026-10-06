"""Frozen-source tests for the Vaski written-question ledger."""

from __future__ import annotations

import json
from pathlib import Path

from paa.question_ledger import (
    ANSWER_KIND,
    QUESTION_KIND,
    VASKI_ROWS_URL,
    acquire_questions,
    import_result,
    normalize_question_records,
    parse_question_row,
)
from paa.store import connect

FIXTURES = Path(__file__).parents[1] / "paa" / "contracts" / "fixtures"


def _rows(name: str) -> list[dict]:
    payload = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
    return [dict(zip(payload["columnNames"], raw)) for raw in payload["rowData"]]


def _records(*names: str) -> list[dict]:
    return [
        parse_question_row(row, retrieved_at="2026-10-06T00:00:00+00:00")
        for name in names
        for row in _rows(name)
    ]


def test_real_frozen_question_and_answer_are_distinct_source_objects() -> None:
    normalized = normalize_question_records(
        _records("vaski_kk1_2023_rows.json", "vaski_kkv1_2023_rows.json")
    )
    question = next(item for item in normalized["objects"] if item["kind"] == QUESTION_KIND)
    answer = next(item for item in normalized["objects"] if item["kind"] == ANSWER_KIND)

    assert question["object_id"] == "eduskunta:KK 1/2023 vp"
    assert answer["object_id"] == "eduskunta:KKV 1/2023 vp"
    assert question["answer_object_id"] == answer["object_id"]
    assert [(author["name"], author["person_id"], author["role"]) for author in question["authors"]] == [
        ("Jussi Saramo", "1400", "AUTHOR")
    ]
    assert question["action_date"] == "2023-04-20"
    assert question["action_date_basis"] == "SUBMISSION_DATE"
    assert question["action_date_provenance"]["basis"] == "VIREILLETULO_EVENT"
    assert question["signature_date"] == "2023-04-20"
    assert question["signature_date_basis"] == "SIGNATURE_DATE"
    assert "Mikä oli syy Fortumin omistajaohjauksen epäonnistumiselle" in question["text"]
    assert question["disposition"] == {
        "state": "ANSWERED",
        "date": "2023-05-09",
        "evidence_ids": question["disposition"]["evidence_ids"],
        "institutional_action": "GOVERNMENT_RESPONSE",
        "policy_implementation": "NOT_ASSESSED",
    }
    assert question["disposition"]["evidence_ids"]
    assert all(ref in {item["evidence_id"] for item in normalized["evidence"]} for ref in question["evidence_ids"])
    assert answer["authors"][0]["name"] == "Tytti Tuppurainen"
    assert answer["authors"][0]["role"] == "RESPONDENT"
    assert answer["disposition"]["policy_implementation"] == "NOT_ASSESSED"


def test_frozen_fixture_receipt_declares_the_real_source_slices() -> None:
    receipt = json.loads((FIXTURES / "vaski_kk_fixture_coverage.json").read_text(encoding="utf-8"))
    assert receipt["source_id"] == "SRC-EDUSKUNTA-VASKI"
    assert receipt["source_records"] == 4
    assert {item["object_id"] for item in receipt["normalized_objects"]} == {
        "eduskunta:KK 1/2023 vp",
        "eduskunta:KKV 1/2023 vp",
    }
    assert all(item["fixture_sha256"] for item in receipt["queries"])


def test_publication_metadata_never_becomes_filing_date_without_event() -> None:
    xml = """
    <root xmlns:m="urn:meta">
      <m:JulkaisuMetatieto m:eduskuntaTunnus="KK 99/2026 vp" m:laadintaPvm="2026-08-02">
        <m:NimekeTeksti>Question without public filing event</m:NimekeTeksti>
      </m:JulkaisuMetatieto>
      <m:Kysymys><m:OtsikkoTeksti>Question without public filing event</m:OtsikkoTeksti></m:Kysymys>
    </root>
    """
    record = parse_question_row(
        {"Id": 99, "XmlData": xml, "Status": 5},
        retrieved_at="2026-10-06T00:00:00+00:00",
    )
    question = normalize_question_records([record])["objects"][0]

    assert question["publication_date"] == "2026-08-02"
    assert question["action_date"] is None
    assert question["action_date_basis"] is None
    assert question["disposition"]["state"] == "UNRESOLVED"


def test_answer_metadata_without_answer_event_is_unresolved() -> None:
    normalized = normalize_question_records(_records("vaski_kkv1_2023_rows.json"))
    answer = normalized["objects"][0]

    assert answer["kind"] == ANSWER_KIND
    assert answer["publication_date"] == "2023-05-09"
    assert answer["action_date"] is None
    assert answer["action_date_basis"] is None
    assert answer["disposition"]["state"] == "UNRESOLVED"
    assert answer["disposition"]["policy_implementation"] == "NOT_ASSESSED"


def test_event_only_answer_does_not_relabel_question_signer_as_respondent() -> None:
    normalized = normalize_question_records(_records("vaski_kk1_2023_rows.json"))
    answer = next(item for item in normalized["objects"] if item["kind"] == ANSWER_KIND)

    assert answer["object_id"] == "eduskunta:KK 1/2023 vp:answer"
    assert answer["document_id"] is None
    assert answer["authors"] == []
    assert answer["title"] == "Vastaus kirjalliseen kysymykseen KK 1/2023 vp"
    assert answer["action_date"] == "2023-05-09"


class _Response:
    def __init__(self, payload: dict) -> None:
        self.content = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.url = VASKI_ROWS_URL

    def raise_for_status(self) -> None:
        return None


class _FixtureClient:
    def __init__(self, payload: dict, *, repeat_page: bool = False) -> None:
        self.payload = payload
        self.repeat_page = repeat_page
        self.calls: list[dict] = []

    def get(self, url: str, **kwargs):
        assert url == VASKI_ROWS_URL
        self.calls.append(dict(kwargs["params"]))
        page = int(kwargs["params"]["page"])
        if page == 0:
            payload = dict(self.payload)
            payload["hasMore"] = self.repeat_page
            return _Response(payload)
        if self.repeat_page and page == 1:
            payload = dict(self.payload)
            payload["hasMore"] = False
            return _Response(payload)
        raise AssertionError(f"unexpected page {page}")


def test_callable_acquisition_records_page_overlap_and_replays_checkpoint(tmp_path: Path) -> None:
    payload = json.loads((FIXTURES / "vaski_kk1_2023_rows.json").read_text(encoding="utf-8"))
    client = _FixtureClient(payload, repeat_page=True)
    result = acquire_questions(
        years=[2023],
        raw_dir=tmp_path,
        client=client,
        include_answers=False,
        partition_identifiers=False,
    )

    assert result["coverage"]["state"] == "ENUMERATED_WITH_PAGE_OVERLAP"
    assert result["coverage"]["complete"] is False
    assert result["coverage"]["overlap_row_count"] == len(payload["rowData"])
    assert result["coverage"]["source_record_count"] == len(payload["rowData"])
    assert len(client.calls) == 2

    class NoNetwork:
        def get(self, *_args, **_kwargs):
            raise AssertionError("a verified checkpoint should be replayed without network")

    replay = acquire_questions(
        years=[2023],
        raw_dir=tmp_path,
        client=NoNetwork(),
        include_answers=False,
        partition_identifiers=False,
    )
    assert replay["coverage"]["coverage_id"] == result["coverage"]["coverage_id"]


def test_import_result_persists_objects_evidence_and_coverage(tmp_path: Path) -> None:
    payload = json.loads((FIXTURES / "vaski_kk1_2023_rows.json").read_text(encoding="utf-8"))
    result = acquire_questions(
        years=[2023],
        raw_dir=tmp_path,
        client=_FixtureClient(payload),
        include_answers=False,
        partition_identifiers=False,
    )
    conn = connect(tmp_path / "questions.sqlite")
    try:
        summary = import_result(conn, result)
        conn.commit()
        assert summary["official_objects"] == 2
        assert summary["source_coverage"] == 1
        assert conn.execute("SELECT COUNT(*) FROM official_objects").fetchone()[0] == 2
        assert conn.execute("SELECT COUNT(*) FROM evidence").fetchone()[0] == len(result["evidence"])
        stored = conn.execute("SELECT json FROM official_objects").fetchone()[0]
        assert json.loads(stored)["kind"] == QUESTION_KIND
    finally:
        conn.close()

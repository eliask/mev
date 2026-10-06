"""Offline tests for the source-grounded Vaski initiative adapter."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from paa.acquire_initiatives import (
    SOURCE_ID,
    VASKI_ROWS_URL,
    acquire_initiatives,
    merge_vaski_records,
    parse_vaski_xml,
    reconcile_action_dates,
)

FIXTURES = Path(__file__).parents[1] / "paa" / "contracts" / "fixtures"


def _parse(name: str, record_id: int) -> dict:
    return parse_vaski_xml(
        (FIXTURES / name).read_text(encoding="utf-8"),
        row_metadata={"Id": record_id, "Status": 5, "Created": "2017-10-24"},
        retrieved_at="2026-10-06T00:00:00+00:00",
    )


def test_frozen_content_slice_keeps_author_target_and_policy_text() -> None:
    record = _parse("vaski_la72_content.xml", 85933)

    assert record["object_id"] == "eduskunta:LA 72/2017 vp"
    assert record["matter_id"] == "LA 72/2017 vp"
    assert record["kind"] == "LEGISLATIVE_INITIATIVE"
    assert record["record_class"] == "CONTENT"
    assert record["date"] == "2017-10-24"
    assert record["title"].startswith("Lakialoite laiksi ikääntyneen väestön")
    assert record["authors"][0]["person_id"] == "1129"
    assert record["authors"][0]["name"] == "Arja Juvonen"
    assert record["authors"][0]["role"] == "AUTHOR"
    assert record["authors"][1]["role"] == "COSIGNER"
    assert "Hoitajamitoituksessa havaitut epäkohdat" in record["text"]
    assert "Henkilöstön vähimmäismäärä tehostetussa palveluasumisessa on 0,60" in record["text"]
    assert record["disposition"] is None
    assert record["evidence"][0]["source_id"] == SOURCE_ID
    assert record["evidence"][0]["record_locator"] == "VaskiData/Id=85933"
    assert record["evidence"][0]["source_url"].startswith(VASKI_ROWS_URL)
    assert record["evidence"][0]["url"] == record["evidence"][0]["source_url"]
    assert record["evidence"][0]["quote"] == record["text"]
    assert record["raw_sha256"] == hashlib.sha256(
        (FIXTURES / "vaski_la72_content.xml").read_bytes()
    ).hexdigest()


def test_procedural_slice_is_not_substituted_for_submitted_initiative() -> None:
    record = _parse("vaski_la72_disposition.xml", 67283)

    assert record["record_class"] == "PROCEDURAL"
    assert record["authors"][0]["person_id"] == "1129"
    assert record["disposition"] == {
        "state": "EXPIRED",
        "date": "2019-04-16",
        "raw_state": "Rauennut",
        "code": "Expired",
        "evidence_ids": [record["evidence_ids"][0]],
    }


def test_approval_is_not_enactment() -> None:
    xml = """
    <root xmlns="urn:test" xmlns:m="urn:meta">
      <m:JulkaisuMetatieto m:eduskuntaTunnus="LA 1/2024 vp" m:laadintaPvm="2024-01-02" />
      <m:KasittelytiedotValtiopaivaasia m:paattymisPvm="2024-02-03">
        <m:NimekeTeksti>Testialoite</m:NimekeTeksti>
        <m:EduskuntakasittelyPaatosKuvaus m:eduskuntakasittelyPaatosKoodi="Approved">Hyväksytty</m:EduskuntakasittelyPaatosKuvaus>
      </m:KasittelytiedotValtiopaivaasia>
    </root>
    """
    record = parse_vaski_xml(xml, row_metadata={"Id": 1}, retrieved_at="2026-10-06T00:00:00+00:00")
    assert record["disposition"]["state"] == "APPROVED"
    assert record["disposition"]["state"] != "ENACTED"


def test_xml_parser_does_not_resolve_external_entities() -> None:
    xml = """<!DOCTYPE root [<!ENTITY secret SYSTEM \"file:///etc/passwd\">]>
    <root><EduskuntaTunnus>LA 1/2024 vp</EduskuntaTunnus><NimekeTeksti>&secret;</NimekeTeksti></root>"""
    record = parse_vaski_xml(xml, row_metadata={"Id": 2}, retrieved_at="2026-10-06T00:00:00+00:00")
    assert "/etc/passwd" not in record["text"]


def test_reference_shell_is_preserved_but_substantive_content_wins() -> None:
    shell = """
    <root xmlns:m="urn:meta">
      <m:JulkaisuMetatieto m:eduskuntaTunnus="LA 72/2017 vp" m:laadintaPvm="2025-12-19">
        <m:NimekeTeksti>Shell title</m:NimekeTeksti>
      </m:JulkaisuMetatieto>
    </root>
    """
    reference = parse_vaski_xml(shell, row_metadata={"Id": 10}, retrieved_at="2026-10-06T00:00:00+00:00")
    content = _parse("vaski_la72_content.xml", 85933)

    assert reference["record_class"] == "CONTENT_REFERENCE"
    assert reference["record_quality"] == "REFERENCE_SHELL"
    merged = merge_vaski_records([reference, content])
    assert merged["record_class"] == "CONTENT"
    assert merged["selection"]["reason"] == "SUBSTANTIVE_CONTENT_OVER_REFERENCE_SHELL"
    assert {row["record_quality"] for row in merged["source_records"]} == {"REFERENCE_SHELL", "SUBSTANTIVE"}
    assert merged["selection"]["preserved_record_count"] == 2


def test_procedural_filing_event_is_separate_from_publication_date() -> None:
    xml = """
    <root xmlns:m="urn:meta">
      <m:KasittelytiedotValtiopaivaasia m:eduskuntaTunnus="LA 35/2025 vp" m:laadintaPvm="2025-12-19">
        <m:NimekeTeksti>Procedural record</m:NimekeTeksti>
        <m:ToimenpideJulkaisu m:kasittelyvaiheKoodi="VIR" m:tapahtumaPvm="2025-12-18">
          <m:ValiotsikkoTeksti>Vireilletulo</m:ValiotsikkoTeksti>
        </m:ToimenpideJulkaisu>
      </m:KasittelytiedotValtiopaivaasia>
    </root>
    """
    record = parse_vaski_xml(xml, row_metadata={"Id": 11}, retrieved_at="2026-10-06T00:00:00+00:00")
    assert record["publication_date"] == "2025-12-19"
    assert record["action_date"] == "2025-12-18"
    assert record["action_date_basis"] == "VIREILLETULO_EVENT"


def test_merge_uses_only_explicit_procedural_filing_date_when_content_is_dateless() -> None:
    content_xml = (FIXTURES / "vaski_la72_content.xml").read_text(encoding="utf-8")
    # Simulate a later registry publication of the same content without a
    # signature date.  The metadata date must remain publication metadata.
    content_xml = content_xml.replace(
        'met1:laadintaPvm="2017-10-24"',
        'met1:laadintaPvm="2026-02-03"',
        1,
    )
    content = parse_vaski_xml(
        content_xml,
        row_metadata={"Id": 85933},
        retrieved_at="2026-10-06T00:00:00+00:00",
    )
    procedure_xml = """
    <Siirto xmlns="http://www.eduskunta.fi/skeemat/siirto/2011/09/07"
            xmlns:met1="http://www.vn.fi/skeemat/metatietoelementit/2010/04/27"
            xmlns:vsk1="http://www.eduskunta.fi/skeemat/vaskielementit/2011/01/04">
      <KasittelytiedotValtiopaivaasia met1:eduskuntaTunnus="LA 72/2017 vp"
                                      met1:laadintaPvm="2026-02-03">
        <NimekeTeksti>Procedural filing</NimekeTeksti>
        <ToimenpideJulkaisu vsk1:kasittelyvaiheKoodi="VIR"
                            vsk1:tapahtumaPvm="2017-10-24">
          <ValiotsikkoTeksti>Vireilletulo</ValiotsikkoTeksti>
        </ToimenpideJulkaisu>
      </KasittelytiedotValtiopaivaasia>
    </Siirto>
    """
    procedure = parse_vaski_xml(
        procedure_xml,
        row_metadata={"Id": 67283},
        retrieved_at="2026-10-06T00:00:00+00:00",
    )
    assert content["action_date"] is None
    assert procedure["action_date_basis"] == "VIREILLETULO_EVENT"

    merged = merge_vaski_records([content, procedure])

    assert merged["publication_date"] == "2026-02-03"
    assert merged["action_date"] == "2017-10-24"
    assert merged["action_date_basis"] == "VIREILLETULO_EVENT"
    assert merged["action_date_provenance"] == {
        "record_locator": "VaskiData/Id=67283",
        "raw_sha256": procedure["raw_sha256"],
        "evidence_ids": procedure["evidence_ids"],
        "basis": "VIREILLETULO_EVENT",
    }
    assert merged["action_date_source_record_locator"] == "VaskiData/Id=67283"


def test_real_registry_drift_checkpoint_keeps_shells_and_action_dates_distinct() -> None:
    checkpoint = json.loads((FIXTURES / "vaski_registry_revision_excerpts.json").read_text(encoding="utf-8"))
    la35 = next(item for item in checkpoint["matters"] if item["matter_id"] == "LA 35/2025 vp")
    la35_shell, la35_content, la35_procedure = la35["rows"]
    assert la35_shell["record_quality"] == "REFERENCE_SHELL"
    assert la35_content["record_quality"] == "SUBSTANTIVE"
    assert la35_procedure["record_quality"] == "PROCEDURAL"
    assert la35_content["publication_date"] == "2026-02-03"
    assert la35_content["action_date"] == "2025-12-19"
    assert la35_content["action_date"] != la35_content["publication_date"]
    assert la35_procedure["action_date_basis"] == "VIREILLETULO_EVENT"

    la22 = next(item for item in checkpoint["matters"] if item["matter_id"] == "LA 22/2025 vp")
    assert {row["record_class"] for row in la22["rows"]} == {"CONTENT_REFERENCE", "CONTENT", "PROCEDURAL"}
    assert next(row for row in la22["rows"] if row["record_class"] == "CONTENT")["text_length"] > 100_000


def test_merge_preserves_both_source_rows_and_bounded_disposition() -> None:
    content = _parse("vaski_la72_content.xml", 85933)
    procedure = _parse("vaski_la72_disposition.xml", 67283)
    merged = merge_vaski_records([content, procedure])

    assert merged["object_id"] == "eduskunta:LA 72/2017 vp"
    assert merged["record_class"] == "CONTENT"
    assert merged["authors"][0]["name"] == "Arja Juvonen"
    assert merged["disposition"]["state"] == "EXPIRED"
    assert merged["disposition"]["date"] == "2019-04-16"
    assert {row["record_class"] for row in merged["source_records"]} == {"CONTENT", "PROCEDURAL"}
    assert len(merged["evidence_ids"]) == 2
    assert len(merged["evidence"]) == 2


def test_signing_before_deadline_does_not_replace_later_formal_filing() -> None:
    content = _parse("vaski_la72_content.xml", 85933)
    content.update(action_date="2017-12-31", action_date_basis="SIGNATURE_DATE")
    procedure = _parse("vaski_la72_disposition.xml", 67283)
    procedure.update(action_date="2018-01-02", action_date_basis="VIREILLETULO_EVENT")
    merged = merge_vaski_records([content, procedure])
    assert merged["signature_date"] == "2017-12-31"
    assert merged["action_date"] == "2018-01-02"
    assert merged["action_date_basis"] == "VIREILLETULO_EVENT"
    assert merged["action_date_source_evidence_ids"] == procedure["evidence_ids"]
    assert reconcile_action_dates(merged) == merged
    # Nearest valid control: a content-only source proves signing, not filing.
    signed = merge_vaski_records([content])
    assert signed["date_binding_state"] == "SIGNATURE_ONLY"
    assert signed["action_date_basis"] == "SIGNATURE_DATE"


def test_later_signature_and_conflicting_filing_dates_remain_explicit() -> None:
    content = _parse("vaski_la72_content.xml", 85933)
    content.update(action_date="2017-10-25", action_date_basis="SIGNATURE_DATE")
    procedure = _parse("vaski_la72_disposition.xml", 67283)
    procedure.update(action_date="2017-10-24", action_date_basis="VIREILLETULO_EVENT")
    merged = merge_vaski_records([content, procedure])
    assert merged["date_binding_state"] == "SIGNATURE_AFTER_FILING"
    another = {**procedure, "action_date": "2017-10-26", "record_locator": "another-source"}
    conflict = merge_vaski_records([content, procedure, another])
    assert conflict["action_date"] is None
    assert conflict["date_binding_state"] == "CONFLICTING_FILING_DATES"
    assert "action_date_source_evidence_ids" not in conflict


def test_callable_acquisition_replays_a_page_without_network(tmp_path: Path) -> None:
    content = (FIXTURES / "vaski_la72_content.xml").read_text(encoding="utf-8")
    procedure = (FIXTURES / "vaski_la72_disposition.xml").read_text(encoding="utf-8")

    class Response:
        def __init__(self, payload: dict) -> None:
            self.payload = payload

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return self.payload

    class Client:
        def get(self, url: str, **kwargs):
            assert url == VASKI_ROWS_URL
            assert kwargs["params"]["columnName"] == "Eduskuntatunnus"
            assert kwargs["params"]["columnValue"] == "LA 72/2017 vp"
            return Response(
                {
                    "columnNames": ["Id", "XmlData", "Status", "Created", "Imported"],
                    "rowData": [
                        [85933, content, 5, "2017-10-24", "2017-10-25"],
                        [67283, procedure, 5, "2019-04-16", "2019-04-17"],
                    ],
                    "hasMore": False,
                }
            )

    output = tmp_path / "initiatives.jsonl"
    result = acquire_initiatives(
        ["LA 72/2017 vp"],
        raw_dir=tmp_path / "raw",
        output_path=output,
        client=Client(),
        retrieved_at="2026-10-06T00:00:00+00:00",
    )

    assert len(result["objects"]) == 1
    assert result["objects"][0]["disposition"]["state"] == "EXPIRED"
    assert result["coverage"]["enumeration"] == "EXACT_MATTER_IDENTIFIERS"
    assert result["coverage"]["requested_row_count"] == 2
    assert result["coverage"]["parsed_row_count"] == 2
    assert len(output.read_text(encoding="utf-8").splitlines()) == 1
    assert len(list((tmp_path / "raw").glob("*.xml"))) == 2

    decoded = json.loads(output.read_text(encoding="utf-8"))
    assert decoded["matter_id"] == "LA 72/2017 vp"
    assert decoded["source_records"][1]["record_class"] == "PROCEDURAL"


def test_frozen_case_labels_broad_policy_relation_without_overclaiming_pledge_completion() -> None:
    case = json.loads((FIXTURES / "initiative_case_arja_juvonen.json").read_text(encoding="utf-8"))
    review = case["relation_review"]

    assert review["statement_quote"] in case["candidate"]["quote"]
    assert review["object_quote"] in case["official_source"]["title"] or review["object_quote"].startswith(
        "Hoitajamitoituksessa"
    )
    assert case["candidate"]["promise_class"] == "POLICY_DESIDERATUM"
    assert case["candidate"]["personal_observable_action"] is False
    assert case["bounded_finding"]["pledge_adjudication"] == "NOT_TESTABLE_AS_WRITTEN"
    assert case["bounded_finding"]["state"] == "DOCUMENTED_RELATED_ACTION"
    assert not review["statement_quote"].endswith("mm.")
    assert review["object_sha256"] == case["normalized_object"]["object_text_sha256"]
    assert "2017" in case["bounded_finding"]["temporal_scope_note"]


def test_kaa_snapshot_has_exact_row_hashes_and_only_official_status_label() -> None:
    rows = [
        json.loads(line)
        for line in (FIXTURES / "vaski_kaa_dispositions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    coverage = json.loads((FIXTURES / "vaski_kaa_coverage.json").read_text(encoding="utf-8"))

    assert {row["matter_id"] for row in rows} == {
        "KAA 1/2023 vp",
        "KAA 2/2023 vp",
        "KAA 5/2023 vp",
        "KAA 11/2021 vp",
    }
    assert all(row["kind"] == "CITIZEN_INITIATIVE" for row in rows)
    assert all(row["disposition"]["state"] == "REJECTED" for row in rows)
    assert all(row["disposition"]["quote"] == "Hylätty" for row in rows)
    assert all(len(row["content_record"]["raw_sha256"]) == 64 for row in rows)
    assert all(len(row["procedural_record"]["raw_sha256"]) == 64 for row in rows)
    assert coverage["requested_row_count"] == 8
    assert coverage["merged_object_count"] == 4

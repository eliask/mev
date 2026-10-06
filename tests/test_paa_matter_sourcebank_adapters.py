"""Producer→consumer tests for KKV/committee sourcebank adapters."""

from pathlib import Path

from paa.matter_site import render_matter_html, write_matter_site
from paa.matter_sourcebank_adapters import (
    adapt_committee_role_sourcebank,
    adapt_kkv_response_sourcebank,
    classify_committee_role_family,
    write_sourcebank_matter_artifacts,
)

FIXTURE = Path("paa/contracts/fixtures/matter_sourcebank")


def test_classify_expert_member_secretary() -> None:
    assert (
        classify_committee_role_family(
            role_values=["valiokuntaneuvos"],
            xml_path="OsallistujaOsa/Toimija",
            source_statement="valiokuntaneuvos X",
            classification="EXPLICIT_ROLE",
        )
        == "COMMITTEE_SECRETARY"
    )
    assert (
        classify_committee_role_family(
            role_values=["puheenjohtaja"],
            xml_path="OsallistujaOsa",
            source_statement="puheenjohtaja Y",
            classification="EXPLICIT_ROLE",
        )
        == "COMMITTEE_MEMBER"
    )
    assert (
        classify_committee_role_family(
            role_values=["hallitusneuvos"],
            xml_path=".../AsiantuntijatToimenpide/.../Asiantuntija/...",
            source_statement="hallitusneuvos Z",
            classification="EXPLICIT_ROLE",
        )
        == "EXPERT_OR_HEARING"
    )
    # Outside-body chair title must not defeat Asiantuntija path structure.
    assert (
        classify_committee_role_family(
            role_values=[],
            xml_path="AsiantuntijatToimenpide/Asiantuntija/Toimija",
            source_statement="puheenjohtaja Hans Duncker",
            classification="EXPLICIT_ROLE",
        )
        == "EXPERT_OR_HEARING"
    )
    assert (
        classify_committee_role_family(
            role_values=[],
            xml_path="OsallistujaOsa",
            source_statement="varajäsen Ville Valkonen kok (osittain)",
            classification="DOCUMENTED_PARTICIPATION",
        )
        == "COMMITTEE_MEMBER"
    )


def test_kkv_adapter_preserves_role_date_body_hashes(tmp_path: Path) -> None:
    import json

    bank = json.loads((FIXTURE / "kkv_response_sourcebank.json").read_text(encoding="utf-8"))
    packet = adapt_kkv_response_sourcebank(bank)
    answer = packet["answer_document"]
    assert answer["document_id"] == "KKV 1/2023 vp"
    assert answer["matter_id"] == "KK 1/2023 vp"
    assert answer["role"]["code"] == "Laatija"
    assert answer["role"]["family_name"] == "Tuppurainen"
    assert answer["dates"]["laadintaPvm"] == "2023-05-09"
    assert answer["body_text_sha256"] == bank["document"]["full_text_sha256"]
    assert answer["pdf_sha256"] == bank["document"]["pdf_sha256"]
    assert answer["source_records"][0]["raw_sha256"]
    hub = render_matter_html(packet)
    assert "Vastausasiakirja" in hub
    assert "Tuppurainen" in hub
    assert "2023-05-09" in hub
    site = write_matter_site(packet, tmp_path / "kkv.html", packet_url="matter_packet.json")
    assert site["intervention_count"] >= 1


def test_committee_adapter_keeps_families_and_date_scope(tmp_path: Path) -> None:
    import json

    bank = json.loads((FIXTURE / "committee_role_sourcebank.json").read_text(encoding="utf-8"))
    packet = adapt_committee_role_sourcebank(bank, matter_id="HE 12/2024 vp")
    section = packet["committee_named_participation"]
    families = {row["role_family"] for row in section["rows"]}
    assert "COMMITTEE_SECRETARY" in families
    assert "COMMITTEE_MEMBER" in families
    assert "EXPERT_OR_HEARING" in families
    for row in section["rows"]:
        assert row["membership_interval_asserted"] is False
        assert row["drafting_authorship_asserted"] is False
        assert "participation_date_scope" in row
        assert row["participation_date_scope"]["membership_interval_asserted"] is False
    hub = render_matter_html(packet)
    assert "Nimetty osallistuminen" in hub
    assert "Asiantuntija" in hub or "EXPERT" in hub
    assert "jäsenyyden täyttä aikaväliä" in hub.casefold() or "Jäsenyyden koko aikaväliä" in hub
    write_matter_site(packet, tmp_path / "committee.html", packet_url="matter_packet.json")


def test_write_artifacts_cli_path(tmp_path: Path) -> None:
    out = tmp_path / "build"
    manifest = write_sourcebank_matter_artifacts(
        kkv_bank_path=FIXTURE / "kkv_response_sourcebank.json",
        output_dir=out,
    )
    assert manifest["db_writes"] is False
    assert (out / "matter_packet.json").is_file()
    assert (out / "matter.html").is_file()
    html = (out / "matter.html").read_text(encoding="utf-8")
    assert "Vastausasiakirja" in html
    assert html.index("Valitse henkilö") < html.index("Asian tiivis yhteenveto")

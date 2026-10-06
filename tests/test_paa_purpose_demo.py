"""Purpose demonstration builds from bundled evidence without live DB."""

import json
import zipfile
from pathlib import Path

from paa.ballot_disposition import validate_packet
from paa.config import FIXTURE_DIR
from paa.inquiry_cases import load_reviewed_cases, validate_case
from paa.purpose_demo import build_purpose_demo
from paa.release import load_selection


def test_purpose_demo_source_to_static_site(tmp_path: Path) -> None:
    result = build_purpose_demo(
        tmp_path / "purpose-demo",
        source_revision="test-purpose-demo",
    )
    root = Path(result["destination"])
    assert result["hospital_rows"] == 25
    assert result["wage_rows"] == 3
    assert result["publication_dependencies"]
    assert (root / "ballot_disposition.json").is_file()
    assert (root / "ballot_disposition.html").is_file()
    assert (root / "ballot_disposition_corpus.json").is_file()
    computation = json.loads((root / "ballot_disposition.json").read_text(encoding="utf-8"))
    assert "corpus_path" not in computation
    assert computation["corpus_filename"] == "ballot_disposition_corpus.json"
    assert ".tmp/" not in json.dumps(computation)
    for packet in computation["instances"].values():
        validate_packet(packet)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert "ballot_disposition.json" in manifest["files"]
    assert "ballot_disposition_corpus.json" in manifest["files"]
    assert manifest["ballot_disposition"]["result_sha256"] == computation["result_sha256"]
    assert manifest["ballot_disposition"]["corpus_sha256"] == computation["corpus_sha256"]
    with zipfile.ZipFile(result["archive"]) as archive:
        names = set(archive.namelist())
        assert "ballot_disposition.json" in names
        assert "ballot_disposition.html" in names
        assert "ballot_disposition_corpus.json" in names
        assert archive.read("ballot_disposition.json") == (root / "ballot_disposition.json").read_bytes()
        # ZIP must match the sealed static artifact exactly (no loose extras outside dest).
        for name in names:
            if name.endswith("/"):
                continue
            assert archive.read(name) == (root / name).read_bytes()
    index = (root / "index.html").read_text(encoding="utf-8")
    assert "ballot_disposition.html" in index
    page = (root / "cases/hospital-hanninen-oulaskangas-attribution.html").read_text(encoding="utf-8")
    assert "Hänninen" in page
    assert "ei riippumatonta ihmistarkistusta" in page
    assert "ballot_disposition.html" in page
    rai_page = (root / "cases/rai-2020-constitutional-repair.html").read_text(encoding="utf-8")
    assert "ballot_disposition.html" not in rai_page
    table = (root / "ballot_disposition.html").read_text(encoding="utf-8")
    assert "Valittu vaihtoehto" in table
    assert "table-wrap" in table
    assert "Sidotut vaihtoehtotekstit" in table
    monitoring = (root / "cases/hospital-monitoring-cost-shift-repair.html").read_text(encoding="utf-8")
    mon_json = (root / "cases/hospital-monitoring-cost-shift-repair.json").read_text(encoding="utf-8")
    assert "1.10.2025" in monitoring or "1.10.2025" in mon_json
    assert "583/2017" not in monitoring
    assert "SHORT_EXCERPTS" in mon_json
    assert "valuma-aluetasolla" not in json.loads(mon_json)["unknowns"][0]["missing_evidence"]
    for case_id in (
        "hospital-hanninen-oulaskangas-attribution",
        "hospital-monitoring-cost-shift-repair",
        "rai-2020-constitutional-repair",
    ):
        packet = json.loads((root / f"cases/{case_id}.json").read_text(encoding="utf-8"))
        assert validate_case(packet)


def test_purpose_selection_compiles_reviewed_packets() -> None:
    selection = load_selection(FIXTURE_DIR / "purpose_demo_selection.json")
    assert selection.release_id == "paa-purpose-demo-2026-10-09"
    assert selection.cases[0].audience == "GENERAL"
    assert selection.cases[2].audience == "SPECIALIST"
    for entry in selection.cases[:2]:
        packets = load_reviewed_cases(FIXTURE_DIR / entry.sources_file, FIXTURE_DIR / entry.reviews_file)
        assert packets[0]["case_id"] == entry.case_id
        assert all(claim["review"]["method"] == "AI_SOURCE_READING" for claim in packets[0]["claims"])


def test_purpose_selection_keeps_public_sections_reader_facing() -> None:
    selection = load_selection(FIXTURE_DIR / "purpose_demo_selection.json")
    hanninen = next(case for case in selection.cases if case.case_id == "hospital-hanninen-oulaskangas-attribution")
    rai = next(case for case in selection.cases if case.case_id == "rai-2020-constitutional-repair")

    hanninen_titles = {section.title for section in hanninen.sections}
    assert "Sama laskenta eri asiassa" not in hanninen_titles
    assert all("kontroll" not in title.casefold() for title in hanninen_titles)

    rai_last = rai.sections[-1]
    assert rai_last.title == "Mitä asiakirjoista voi päätellä?"
    assert "kontroll" not in rai_last.title.casefold()
    assert rai_last.evidence_ids == (
        "inquiry-evidence:21d5e1d64b9c3d011a638621",
        "inquiry-evidence:a294e30f424ccf3b27d4bb28",
    )
    assert rai.audience == "SPECIALIST"

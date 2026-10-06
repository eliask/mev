"""Focused tests for source-tree facts carried into PAA model packets."""

import hashlib
import json
from pathlib import Path

import pytest

from paa.llm_inquiries import build_request, load_prepared_run, prepare_run
from paa.source_structure import (
    CANONICAL_NORMALIZATION_VERSION,
    STATUS_UNAVAILABLE,
    STATUS_UNSUPPORTED,
    SourceStructureError,
    canonical_vaski_committee_text,
    parse_source_structure,
    source_structure_for_clips,
    source_structure_from_wire,
)

FIXTURES = Path(__file__).parents[1] / "paa/contracts/fixtures"


def _official_sources() -> list[dict]:
    rows = [
        json.loads(line)
        for line in (FIXTURES / "inquiry_official_source_versions.jsonl").read_text(encoding="utf-8").split("\n")
        if line.strip()
    ]
    return [source for row in rows for source in row["sources"]]


def test_vaski_structure_preserves_dissent_voice_and_exact_offsets() -> None:
    source = next(item for item in _official_sources() if item["record_id"] == "291583")
    assert source["normalization_version"] == CANONICAL_NORMALIZATION_VERSION
    assert canonical_vaski_committee_text(source["raw_text"]) == source["text"]
    document = parse_source_structure(source)

    assert document.status == "PARSED"
    assert document.raw_sha256 == source["raw_sha256"]
    assert document.text_sha256 == source["text_sha256"]
    assert document.coverage["candidate_nodes"] >= document.coverage["bound_spans"]

    dissent = next(span for span in document.spans if span.text == "Vastalause 1")
    assert dissent.voice_scope == "DISSENT"
    assert "JasenMielipideOsa" in dissent.ancestor_path
    paragraph = next(span for span in document.spans if "Poukkoileva politiikka" in span.text)
    assert paragraph.voice_scope == "DISSENT"
    assert paragraph.heading_path[:2] == ("Vastalause 1", "Perustelut")
    assert source["text"][paragraph.start:paragraph.end] == paragraph.text
    assert paragraph.raw_sha256 == source["raw_sha256"]


def test_vaski_structure_keeps_matter_reference_and_committee_stage_local() -> None:
    source = next(item for item in _official_sources() if item["record_id"] == "236013")
    assert canonical_vaski_committee_text(source["raw_text"]) == source["text"]
    document = parse_source_structure(source)

    recommendation = next(
        span
        for span in document.spans
        if "Eduskunta hylkää lakialoitteeseen LA 19/2021 vp" in span.text
    )
    assert recommendation.matter_refs == ("LA 19/2021 vp",)
    assert recommendation.procedural_stage == "COMMITTEE_RECOMMENDATION"
    assert recommendation.voice_scope == "COMMITTEE_RECOMMENDATION"
    assert source["text"][recommendation.start:recommendation.end] == recommendation.text
    assert "LA 19/2021 vp" in recommendation.text


def test_declared_canonical_version_rejects_text_projection_drift() -> None:
    source = next(item for item in _official_sources() if item["record_id"] == "291583")
    changed = dict(source)
    changed["text"] = source["text"] + " drift"
    changed["text_sha256"] = hashlib.sha256(changed["text"].encode()).hexdigest()
    document = parse_source_structure(changed)
    assert document.status == "INVALID_SOURCE"
    assert "does not match" in (document.error or "")


def test_plain_indexed_source_does_not_receive_xml_structure() -> None:
    source = next(item for item in _official_sources() if item["record_id"] == "ymvm8-2024")
    document = parse_source_structure(source)
    assert document.status == STATUS_UNSUPPORTED
    assert document.spans == ()
    context = source_structure_for_clips(source, [{"start": 0, "end": 120, "text": source["text"][:120]}])
    assert context["status"] == STATUS_UNSUPPORTED
    assert context["spans"] == []


def test_missing_raw_source_is_explicit_in_model_context() -> None:
    source = {
        "source_id": "synthetic",
        "text": "plain source",
        "text_sha256": hashlib.sha256(b"plain source").hexdigest(),
        "content_format": "HTML_OR_AKN_FIELD",
    }
    context = source_structure_for_clips(source, [{"start": 0, "end": 12}])
    assert context["status"] == STATUS_UNAVAILABLE
    assert context["spans"] == []
    assert "raw" in context["error"]


def test_structure_codec_round_trip_and_unknown_field_rejection() -> None:
    source = next(item for item in _official_sources() if item["record_id"] == "291583")
    document = parse_source_structure(source)
    restored = source_structure_from_wire(document.to_wire())
    assert restored == document
    bad = document.to_wire()
    bad["unexpected_meaning"] = "do not accept"
    with pytest.raises(SourceStructureError, match="unknown structure fields"):
        source_structure_from_wire(bad)


def test_structure_owns_scalar_coverage_and_binds_repeated_text_in_source_order() -> None:
    raw = """<Mietinto><OtsikkoTeksti>Perustelut</OtsikkoTeksti><KappaleKooste>Toistuva kohta.</KappaleKooste><KappaleKooste>Toistuva kohta.</KappaleKooste></Mietinto>"""
    text = "Perustelut\nToistuva\nkohta.\nToistuva\nkohta."
    source = {
        "source_id": "synthetic-repeated",
        "raw_text": raw,
        "raw_sha256": hashlib.sha256(raw.encode()).hexdigest(),
        "text": text,
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "content_format": "VASKI_XML",
        "normalization_version": "test-canonical-v1",
    }
    document = parse_source_structure(source)
    repeats = [span for span in document.spans if "Toistuva" in span.text]
    assert [(span.start, span.end) for span in repeats] == [(11, 26), (27, 42)]
    assert text[repeats[0].start : repeats[0].end] == repeats[0].text
    assert text[repeats[1].start : repeats[1].end] == repeats[1].text
    assert document.coverage["candidate_nodes"] == 3
    with pytest.raises(TypeError):
        document.coverage["candidate_nodes"] = 0  # type: ignore[index]
    original = {"candidate_nodes": 3}
    owned = source_structure_from_wire(
        {
            **document.to_wire(),
            "coverage": original,
        }
    )
    original["candidate_nodes"] = 0
    assert owned.coverage["candidate_nodes"] == 3


def test_prepared_model_input_carries_structure_with_source_offsets(tmp_path: Path) -> None:
    row = next(
        row
        for row in (
            json.loads(line)
            for line in (FIXTURES / "inquiry_official_source_versions.jsonl").read_text(encoding="utf-8").split("\n")
            if line.strip()
        )
        if row["episode_id"] == "source-matter-ymvm17-2022"
    )
    row["question_contract"] = {
        "text": "Mitä lähdeasiakirjan rakenteesta voidaan todeta?",
        "target_scope": "one official committee report",
        "period": "2022",
        "comparison": "source structure",
        "evidence_needed": ["exact source span", "structural path"],
        "valid_outputs": ["documentary structure"],
        "unknowns": ["private drafting"],
    }
    source_fixture = tmp_path / "sources.jsonl"
    source_fixture.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    run = tmp_path / "prepared"
    prepare_run(source_fixture, run, clip_limit=2_000, window_limit=20_000, coverage_mode="FOCUSED")
    _, windows = load_prepared_run(run)
    window = windows[0]
    source = next(item for item in window["payload"]["sources"] if item["source_id"].endswith(":236013"))
    structure = source["source_structure"]
    assert structure["status"] == "PARSED"
    official = next(item for item in row["sources"] if item["record_id"] == "236013")
    assert structure["raw_sha256"] == official["raw_sha256"]
    assert structure["spans"]
    span = structure["spans"][0]
    assert official["text"][span["start"]:span["end"]] == span["text"]
    assert span["text_sha256"] == structure["text_sha256"]
    baseline = build_request(window, "baseline")
    structured = build_request(window, "structured")
    assert baseline["source_payload_sha256"] == structured["source_payload_sha256"]
    model_source = next(item for item in structured["model_payload"]["sources"] if item["source_id"].endswith(":236013"))
    assert model_source["source_structure"]["spans"]

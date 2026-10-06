"""Ballot snapshot enrichment producer/consumer on a tiny frozen bank."""

import json
from pathlib import Path

import pytest

from paa.ballot_snapshot_enrichment import (
    BINDING_STATUS,
    BallotEnrichmentError,
    enrich_matter_files,
    enrich_matter_packet,
    validate_witness_event,
)

FIXTURE = Path("paa/contracts/fixtures/ballot_enrichment_bank")


def _comparison() -> dict:
    return json.loads((FIXTURE / "comparison.json").read_text(encoding="utf-8"))


def test_valid_three_person_event_binds_and_bad_event_fails(tmp_path: Path) -> None:
    bank = json.loads((FIXTURE / "witness_bank.json").read_text(encoding="utf-8"))
    matter = json.loads((FIXTURE / "matter.json").read_text(encoding="utf-8"))
    ok = validate_witness_event(bank["events"][0], bank_dir=FIXTURE)
    assert ok["ok"] is True
    assert ok["unique_people"] == 3
    bad = validate_witness_event(bank["events"][1], bank_dir=FIXTURE)
    assert bad["ok"] is False
    assert any("has_more" in err or "mismatch" in err for err in bad["errors"])

    enriched = enrich_matter_packet(matter, bank, comparison=_comparison(), bank_dir=FIXTURE)
    report = enriched["ballot_enrichment"]
    assert "90001" in report["bound_event_ids"]
    assert "90002" in report["failed_event_ids"]
    good = enriched["vote_sources"][0]
    assert good["source_binding"]["status"] == BINDING_STATUS
    assert good["source_binding"]["metadata_raw_sha256"]
    assert good["source_binding"]["raw_witness_separate_from_normalized"] is True
    assert good["rows"][0]["source_binding"]["raw_json_path"]
    assert good["rows"][0]["source_binding"]["raw_page_sha256"]
    assert len(good["rows"]) == 3
    assert good["rows"][0]["person_id"] == "1"
    assert "source_binding" not in enriched["vote_sources"][1]

    out = tmp_path / "enriched"
    manifest = enrich_matter_files(
        matter_paths=[FIXTURE / "matter.json"],
        witness_bank_path=FIXTURE / "witness_bank.json",
        comparison_path=FIXTURE / "comparison.json",
        output_dir=out,
    )
    assert manifest["db_writes"] is False
    assert (out / "matter.json").is_file()
    portable = json.loads((out / "portable_ballot_witness_bank.json").read_text(encoding="utf-8"))
    dumped = json.dumps(portable)
    assert "/.tmp/" not in dumped
    assert "/mnt/" not in dumped
    assert (out / "raw_witnesses").is_dir()
    assert any((out / "raw_witnesses").glob("*metadata*"))


def test_changed_normalized_response_fails_before_binding() -> None:
    bank = json.loads((FIXTURE / "witness_bank.json").read_text(encoding="utf-8"))
    matter = json.loads((FIXTURE / "matter.json").read_text(encoding="utf-8"))
    matter["vote_sources"][0]["rows"][0]["response"] = "EI"
    enriched = enrich_matter_packet(matter, bank, comparison=_comparison(), bank_dir=FIXTURE)
    assert "90001" in enriched["ballot_enrichment"]["failed_event_ids"]
    assert "source_binding" not in enriched["vote_sources"][0]


def test_tampered_receipt_hash_rejected() -> None:
    bank = json.loads((FIXTURE / "witness_bank.json").read_text(encoding="utf-8"))
    bank["events"][0]["metadata_receipt"]["sha256"] = "0" * 64
    bad = validate_witness_event(bank["events"][0], bank_dir=FIXTURE)
    assert bad["ok"] is False
    assert any("sha256 mismatch" in err for err in bad["errors"])


def test_missing_comparison_coverage_rejected() -> None:
    bank = json.loads((FIXTURE / "witness_bank.json").read_text(encoding="utf-8"))
    matter = json.loads((FIXTURE / "matter.json").read_text(encoding="utf-8"))
    with pytest.raises(BallotEnrichmentError, match="comparison"):
        enrich_matter_packet(matter, bank, comparison=None, bank_dir=FIXTURE)
    thin = {"events": [{"event_id": "90001", "status": "MATCH_CURRENT_RAW", "mismatches": {}, "errors": []}]}
    with pytest.raises(BallotEnrichmentError, match="exactly the bank events"):
        enrich_matter_packet(matter, bank, comparison=thin, bank_dir=FIXTURE)


def test_vote_id_alias_and_refuse_nonempty_output(tmp_path: Path) -> None:
    bank = json.loads((FIXTURE / "witness_bank.json").read_text(encoding="utf-8"))
    matter = json.loads((FIXTURE / "matter.json").read_text(encoding="utf-8"))
    matter["vote_sources"][0].pop("event_id", None)
    matter["vote_sources"][0]["vote_id"] = "90001"
    matter["vote_sources"][0]["source_evidence"] = [
        {
            "kind": "NORMALIZED_LOCAL_RECORD",
            "text_sha256": "a" * 64,
            "record_locator": "vote_events.aanestys_id=90001",
        }
    ]
    enriched = enrich_matter_packet(matter, bank, comparison=_comparison(), bank_dir=FIXTURE)
    binding = enriched["vote_sources"][0]["source_binding"]
    assert binding["event_id"] == "90001"
    assert binding["normalized_snapshot_hashes"][0]["text_sha256"] == "a" * 64
    assert binding["metadata_raw_sha256"] != "a" * 64

    out = tmp_path / "out"
    out.mkdir()
    (out / "keep.txt").write_text("x", encoding="utf-8")
    with pytest.raises(BallotEnrichmentError, match="not empty"):
        enrich_matter_files(
            matter_paths=[FIXTURE / "matter.json"],
            witness_bank_path=FIXTURE / "witness_bank.json",
            comparison_path=FIXTURE / "comparison.json",
            output_dir=out,
        )


def test_exported_raw_bank_replays_without_original_paths(tmp_path: Path) -> None:
    first = tmp_path / "first"
    enrich_matter_files(
        matter_paths=[FIXTURE / "matter.json"], witness_bank_path=FIXTURE / "witness_bank.json",
        comparison_path=FIXTURE / "comparison.json", output_dir=first,
    )
    bank_path = first / "portable_ballot_witness_bank.json"
    bank = json.loads(bank_path.read_text())
    assert "body_path" not in bank["events"][0]["metadata_receipt"]
    second = tmp_path / "second"
    result = enrich_matter_files(
        matter_paths=[FIXTURE / "matter.json"], witness_bank_path=bank_path,
        comparison_path=FIXTURE / "comparison.json", output_dir=second,
    )
    assert result["matter_files"][0]["bound"] == ["90001"]
    assert result["matter_files"][0]["failed"] == ["90002"]  # retained incomplete-page control
    assert len(list((second / "raw_witnesses").glob("*.body"))) == 3
    assert sorted(p.read_bytes() for p in (first / "raw_witnesses").glob("*.body")) == sorted(
        p.read_bytes() for p in (second / "raw_witnesses").glob("*.body")
    )
    # A retained-artifact locator must stay inside its selected artifact root.
    bank["events"][0]["metadata_receipt"]["artifact_body"] = "../outside.body"
    (tmp_path / "outside.body").write_bytes(next((first / "raw_witnesses").glob("90001_metadata*")).read_bytes())
    matter = json.loads((FIXTURE / "matter.json").read_text())
    invalid = enrich_matter_packet(matter, bank, comparison=_comparison(), bank_dir=first)
    assert invalid["ballot_enrichment"]["bound_event_ids"] == []
    assert invalid["ballot_enrichment"]["failed_event_ids"] == ["90001", "90002"]

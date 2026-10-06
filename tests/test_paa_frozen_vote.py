"""Offline replay tests for the genuine 52877 Eduskunta vote slice."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from paa.frozen_vote import FrozenVoteError, import_frozen_vote_fixture, load_frozen_vote_fixture
from paa.group_agreement import build_group_context, validate_group_packets
from paa.store import connect

FIXTURE_DIR = Path(__file__).parents[1] / "paa" / "contracts" / "fixtures"
FIXTURE = FIXTURE_DIR / "eduskunta_vote_52877_fixture.json"


def _copy_fixture(tmp_path: Path) -> Path:
    spec = json.loads(FIXTURE.read_text(encoding="utf-8"))
    destination = tmp_path / FIXTURE.name
    shutil.copy2(FIXTURE_DIR / spec["metadata_fixture"]["file"], tmp_path / spec["metadata_fixture"]["file"])
    for page in spec["ballot_source"]["pages"]:
        shutil.copy2(FIXTURE_DIR / page["file"], tmp_path / page["file"])
    shutil.copy2(FIXTURE, destination)
    return destination


def test_genuine_vote_fixture_reconciles_metadata_all_buckets_and_raw_receipts() -> None:
    fixture = load_frozen_vote_fixture(FIXTURE)
    assert fixture["vote_id"] == "52877"
    assert fixture["row_count"] == fixture["unique_person_count"] == 199
    assert fixture["published_totals"] == {
        "JAA": 120,
        "EI": 62,
        "TYHJA": 0,
        "POISSA": 17,
        "TOTAL": 199,
    }
    assert fixture["observed_totals"] == fixture["published_totals"]
    assert len(fixture["raw_receipts"]) == 3
    assert {receipt["raw_sha256"] for receipt in fixture["raw_receipts"]} == {
        "270589cd5bc00905f3882b7c94e787c98862d6d47e90a46bdfb98d11809e2ab7",
        "b57e97ee0ab074d5cfabbbf3589efb168375b67f07f3e5ed15f7d4adcb284459",
        "a6183d41a01b34b51543f8eb7859f7faa46890c3e9ae8d012cd8f64c790e6999",
    }
    assert fixture["metadata"]["matter"] == "KAA 1/2023 vp"
    assert fixture["metadata"]["session_date"] == "2024-04-05"
    assert fixture["metadata"]["mitatoity"] == 0
    assert len({row["person_number"] for row in fixture["rows"]}) == 199


def test_imported_vote_reaches_group_context_without_invented_membership(tmp_path: Path) -> None:
    db_path = tmp_path / "data" / "paa.sqlite"
    conn = connect(db_path)
    imported = import_frozen_vote_fixture(conn, FIXTURE)
    conn.commit()
    packets, sources, report = build_group_context(conn)
    assert imported["row_count"] == 199
    assert conn.execute("SELECT COUNT(*) FROM vote_events WHERE aanestys_id='52877'").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM ballots WHERE aanestys_id='52877'").fetchone()[0] == 199
    source = next(source for source in sources if source["vote_id"] == "52877")
    assert source["state"] == "VALID"
    assert source["row_count"] == 199
    assert source["published_totals"] == imported["published_totals"]
    assert report["source_count"] == 1
    assert report["ballot_row_count"] == 199
    assert validate_group_packets(packets, sources)["valid"] is True
    conn.close()


def test_changed_raw_page_is_rejected_instead_of_becoming_a_new_fixture(tmp_path: Path) -> None:
    fixture = _copy_fixture(tmp_path)
    spec = json.loads(fixture.read_text(encoding="utf-8"))
    page_path = tmp_path / spec["ballot_source"]["pages"][0]["file"]
    body = page_path.read_bytes()
    assert b"Jaa                 " in body
    page_path.write_bytes(body.replace(b"Jaa                 ", b"Ei                  ", 1))
    with pytest.raises(FrozenVoteError, match="raw receipt mismatch"):
        load_frozen_vote_fixture(fixture)


def test_fixture_page_count_and_response_conflicts_cannot_be_silently_filled(tmp_path: Path) -> None:
    fixture = _copy_fixture(tmp_path)
    spec = json.loads(fixture.read_text(encoding="utf-8"))
    page_path = tmp_path / spec["ballot_source"]["pages"][1]["file"]
    payload = json.loads(page_path.read_text(encoding="utf-8"))
    payload["rowData"] = payload["rowData"][:-1]
    # The retained raw hash is deliberately not updated: this is an invalid
    # incomplete capture, not an opportunity to fabricate the missing row.
    page_path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    with pytest.raises(FrozenVoteError, match="raw receipt mismatch"):
        load_frozen_vote_fixture(fixture)


def test_full_frozen_build_checks_group_math_and_invalidates_changed_source(tmp_path: Path) -> None:
    from paa.check import run
    from paa.frozen import build_frozen

    root = tmp_path / "frozen"
    build_frozen(root)
    db_path = root / "data" / "paa.sqlite"
    assert run(db_path, full_corpus=False) == []
    conn = connect(db_path)
    conn.execute("UPDATE ballots SET party='erk' WHERE aanestys_id='52877' AND person_number="
                 "(SELECT person_number FROM ballots WHERE aanestys_id='52877' AND raw_response='JAA' LIMIT 1)")
    conn.commit()
    conn.close()
    problems = run(db_path, full_corpus=False)
    assert any("group context:" in problem and "changed; recompile" in problem for problem in problems)

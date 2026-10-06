from __future__ import annotations

import hashlib
import json
from pathlib import Path

from paa.acquire_official_actions import (
    acquire_official_actions,
    import_official_actions,
    import_relation_review,
    parse_resignation_source,
)
from paa.store import connect

REQUEST = (
    "Aluevaltuuston varsinainen jäsen Miko Bergbom on 7.4.2023 pyytänyt eroa "
    "aluevaltuuston jäsenyydestä ja aluevaltuuston 2. varapuheenjohtajan tehtävästä. "
    "Perusteluina hän on esittänyt kansanedustajan tehtävää. Eroanomus on liitteenä."
)
DECISION = (
    "Aluevaltuusto päätti myöntää Miko Bergbomille eron aluevaltuuston jäsenyydestä "
    "ja aluevaltuuston 2. varapuheenjohtajan tehtävästä."
)


def _source() -> dict[str, str | int]:
    return {
        "object_id": "pirha-2023-12670-bergbom",
        "source_id": "SRC-PIRHA-2023-7933",
        "url": "https://pirha.cloudnc.fi/fi-FI/content/7933/12670",
        "raw_sha256": hashlib.sha256(b"frozen-primary-body").hexdigest(),
        "raw_bytes": 82712,
        "record_locator": "Pirkanmaan aluevaltuusto, kokous 05.06.2023, §54, Päätös",
        "matter_id": "9112/2023",
        "title": "Aluevaltuuston jäsenen ja 2. varapuheenjohtajan vaihtuminen",
        "request_quote": REQUEST,
        "decision_quote": DECISION,
        "request_date": "2023-04-07",
        "decision_date": "2023-06-05",
        "actor_id": "mp-1514",
        "actor_name": "Miko Bergbom",
        "role": "Pirkanmaan hyvinvointialueen aluevaltuuston jäsen ja 2. varapuheenjohtaja",
    }


def test_primary_decision_keeps_request_and_grant_dates_distinct() -> None:
    result = parse_resignation_source(_source())
    obj = result["object"]
    assert obj["date"] == "2023-06-05"
    assert obj["action_date"] == "2023-06-05"
    assert obj["request_date"] == "2023-04-07"
    assert obj["authors"][0]["actor_id"] == "mp-1514"
    assert obj["authors"][0]["identity_basis"] == "INDEPENDENT_CASE_REVIEW"
    assert DECISION in obj["text"]
    assert result["evidence"][1]["quote"] == DECISION
    assert result["evidence"][1]["url"] == _source()["url"]


def test_frozen_pirha_primary_slice_replays_with_record_hash() -> None:
    source = json.loads((Path(__file__).parents[1] / "paa/contracts/fixtures/pirha_2023_7933_final.json").read_text())
    result = parse_resignation_source(source)
    assert result["object"]["source_raw_sha256"] == "9ff99c9682c7d6c98b590402785a86f2d1c480f54d2eab911d66e36225fd07d5"
    assert result["object"]["action_date"] == "2023-06-05"
    assert result["object"]["request_date"] == "2023-04-07"
    assert result["evidence"][1]["quote"].endswith("Pykälä tarkastettiin kokouksessa.")


def test_callable_import_writes_source_shaped_jsonl(tmp_path: Path) -> None:
    output = tmp_path / "official-actions.jsonl"
    result = acquire_official_actions([_source()], output_path=output, retrieved_at="2026-10-06T00:00:00+00:00")
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert result["coverage"]["status"] == "RECONCILED_FOR_DECLARED_SLICE"
    assert result["coverage"]["expected_count"] == 1
    assert [row["kind"] for row in rows] == ["official_object", "evidence", "evidence", "source_coverage"]


def test_live_import_upserts_only_action_tables(tmp_path: Path) -> None:
    result = acquire_official_actions([_source()], retrieved_at="2026-10-06T00:00:00+00:00")
    conn = connect(tmp_path / "paa.sqlite")
    conn.execute("INSERT INTO meta(key, value) VALUES ('roster-sentinel', 'keep')")
    conn.commit()
    stats = import_official_actions(conn, result)
    conn.commit()
    assert stats == {"official_objects": 1, "evidence": 2, "source_coverage": 1}
    assert conn.execute("SELECT value FROM meta WHERE key='roster-sentinel'").fetchone()[0] == "keep"
    assert conn.execute("SELECT COUNT(*) FROM official_objects").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM evidence").fetchone()[0] == 2
    conn.close()


def test_relation_review_import_does_not_clear_existing_rows(tmp_path: Path) -> None:
    review = {
        "review_id": "review-test",
        "proposition_id": "prop-test",
        "object_id": "object-test",
        "status": "UNRESOLVED",
        "rationale": "test review",
    }
    conn = connect(tmp_path / "paa.sqlite")
    conn.execute("INSERT INTO relation_reviews(review_id, json) VALUES ('keep', '{}')")
    assert import_relation_review(conn, review) == "review-test"
    conn.commit()
    assert conn.execute("SELECT COUNT(*) FROM relation_reviews").fetchone()[0] == 2
    conn.close()

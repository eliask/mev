import json

import pytest

from paa.config import FIXTURE_DIR
from paa.frozen import DEFAULT_FIXTURE, seed_frozen_database
from paa.import_evidence import import_records
from paa.pipeline import compile_database
from paa.store import connect


def test_review_bundle_replays_without_replacing_source_corpus(tmp_path):
    database = tmp_path / "paa.sqlite"
    seed_frozen_database(database)
    conn = connect(database)
    for table in ("official_objects", "evidence", "source_coverage", "relation_reviews"):
        conn.execute(f"DELETE FROM {table}")
    conn.commit()
    source_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    result = import_records(conn, DEFAULT_FIXTURE)
    assert result["imported"]["relation_review"] == 2
    assert result["ignored_source_rows"]["document"] == 3
    assert conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == source_count
    compile_database(conn)
    packet = json.loads(conn.execute("SELECT json FROM evidence_traces WHERE proposition_id = 'yle2023-1029-3-p1'").fetchone()[0])
    assert packet["assessment"]["state"] == "OBSERVED_ALIGNED_ACTION"
    conn.close()


def test_invalid_later_record_cannot_leave_a_partial_import(tmp_path):
    bundle = tmp_path / "invalid.jsonl"
    bundle.write_text(json.dumps({"kind": "evidence", "row": {"evidence_id": "first"}}) + "\n" +
                      json.dumps({"kind": "relation_review", "row": {"review_id": "bad"}}))
    conn = connect(tmp_path / "paa.sqlite")
    with pytest.raises(ValueError, match="incomplete relation review"):
        import_records(conn, bundle)
    assert conn.execute("SELECT COUNT(*) FROM evidence").fetchone()[0] == 0
    conn.close()


def test_pretty_json_review_bundle_is_importable(tmp_path):
    conn = connect(tmp_path / "paa.sqlite")
    result = import_records(conn, FIXTURE_DIR / "initiative_case_arja_juvonen_live_review.json")
    assert result["imported"]["relation_review"] == 1
    assert conn.execute("SELECT COUNT(*) FROM relation_reviews").fetchone()[0] == 1
    conn.close()


def test_wrong_review_field_types_cannot_enter_the_ledger(tmp_path):
    rows = [json.loads((FIXTURE_DIR / "initiative_case_arja_juvonen_live_review.json").read_text())]
    review = next(row["row"] for row in rows if row["kind"] == "relation_review")
    review["normalized_target"] = ["elder-care"]
    path = tmp_path / "malformed.json"
    path.write_text(json.dumps(rows))
    conn = connect(tmp_path / "paa.sqlite")
    with pytest.raises(ValueError, match="invalid field types"):
        import_records(conn, path)
    assert conn.execute("SELECT COUNT(*) FROM relation_reviews").fetchone()[0] == 0
    conn.close()


def test_jsonl_source_separators_survive_import_and_frozen_replay(tmp_path):
    from paa.frozen import _read_rows

    source_quote = "Ensimmäinen katkelma\u2028toinen katkelma\u2029kolmas katkelma"
    record = {"kind": "evidence", "row": {"evidence_id": "unicode-source", "quote": source_quote}}
    control = {"kind": "evidence", "row": {"evidence_id": "ordinary-source", "quote": "Tavallinen katkelma"}}
    path = tmp_path / "source.jsonl"
    path.write_text('\n'.join(json.dumps(row, ensure_ascii=False) for row in (record, control)) + '\n', encoding="utf-8")
    assert _read_rows(path) == [record, control]
    conn = connect(tmp_path / "paa.sqlite")
    try:
        assert import_records(conn, path)["imported"]["evidence"] == 2
        retained = json.loads(conn.execute("SELECT json FROM evidence WHERE evidence_id=?", ("unicode-source",)).fetchone()[0])
        assert retained["quote"] == source_quote
    finally:
        conn.close()

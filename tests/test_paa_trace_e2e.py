"""Cold-reader acceptance tests for the offline source -> trace -> site path.

These tests intentionally inspect the generated SQLite packets and static
files.  A hand-written HTML string cannot satisfy them: the campaign source,
identity join, proposition compiler, relation review, authority window,
action ledger, finding, export, and browser must all be connected.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from paa.frozen import DEFAULT_FIXTURE, build_frozen, seed_frozen_database
from paa.pipeline import compile_database
from paa.records import evidence_reference_ids
from paa.store import connect


def _trace_rows(db_path: Path) -> dict[str, dict]:
    conn = sqlite3.connect(db_path)
    rows = {
        row[0]: json.loads(row[1])
        for row in conn.execute("SELECT proposition_id, json FROM evidence_traces")
    }
    conn.close()
    return rows


def test_frozen_seed_is_source_shaped_and_network_free(tmp_path, monkeypatch):
    def fail_network(*_args, **_kwargs):  # pragma: no cover - called only on regression
        raise AssertionError("frozen source build attempted network access")

    monkeypatch.setattr("paa.http_client.get_bytes", fail_network)
    db_path = tmp_path / "data" / "paa.sqlite"
    stats = seed_frozen_database(db_path, DEFAULT_FIXTURE)
    assert stats["document"] == 3
    assert stats["official_object"] == 2
    assert stats["relation_review"] == 2
    conn = sqlite3.connect(db_path)
    assert conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 3
    assert conn.execute("SELECT COUNT(*) FROM official_objects").fetchone()[0] == 2
    assert conn.execute("SELECT COUNT(*) FROM relation_reviews").fetchone()[0] == 2
    assert conn.execute("SELECT COUNT(*) FROM source_coverage").fetchone()[0] == 2
    assert conn.execute("SELECT COUNT(*) FROM evidence").fetchone()[0] == 4
    manifest = {
        row[0]: row[1:]
        for row in conn.execute(
            "SELECT source_id, sha256, bytes, note FROM manifest WHERE source_id != 'SRC-FROZEN-FIXTURE'"
        )
    }
    assert manifest["SRC-VAALIT-2023"][0] == "a0af4a015ac30b432485026ee83a8b6a6525013b6f39735335368d4cc90e9b3a"
    assert manifest["SRC-VAALIT-2011"][0] == "27d056493ff46c4d19a76a912bae867dce8c83370eb5b5ee2299de68ce80dba2"
    assert manifest["SRC-YLE-2023-C7"][0] == "0ccd5beea7ea8fba1ec02cf33dc198b5a4702aee4b306ac17e07e1c77b0412bc"
    assert manifest["SRC-PIRHA-2023-7933"][0] == "9ff99c9682c7d6c98b590402785a86f2d1c480f54d2eab911d66e36225fd07d5"
    assert manifest["SRC-PIRHA-2023-7933"][1] == 82712
    assert "source_slice_sha256=" in manifest["SRC-EDUSKUNTA-VOTE-52877"][2]
    pirha = json.loads(conn.execute(
        "SELECT json FROM official_objects WHERE object_id = 'pirha-2023-9112-bergbom'"
    ).fetchone()[0])
    assert pirha["source_raw_sha256"] == manifest["SRC-PIRHA-2023-7933"][0]
    assert pirha["source_raw_bytes"] == manifest["SRC-PIRHA-2023-7933"][1]
    assert pirha["source_raw_bytes"] > len(pirha["text"].encode("utf-8"))
    conn.close()


def test_frozen_sources_compile_to_bounded_trace_states_and_site(tmp_path):
    result = build_frozen(tmp_path)
    db_path = Path(result["db_path"])
    browser = Path(result["output_dir"])
    traces = _trace_rows(db_path)

    assert {"yle2023-1029-3-p1", "yle2011-2875-p2", "yle2023-482-1-p4"} <= set(traces)
    assert traces["yle2023-1029-3-p1"]["assessment"]["state"] == "OBSERVED_ALIGNED_ACTION"
    assert traces["yle2011-2875-p2"]["assessment"]["state"] == "NO_OBSERVABLE_OPPORTUNITY"
    assert traces["yle2023-482-1-p4"]["assessment"]["state"] == "INSUFFICIENT_EVIDENCE"

    miko = traces["yle2023-1029-3-p1"]
    assert miko["authority"]["condition_state"] == "SATISFIED"
    assert miko["authority"]["opportunity_state"] == "OBSERVABLE_OPPORTUNITY"
    assert miko["actions"]
    assert miko["actions"][0]["state"] == "OBSERVED_ALIGNED_ACTION"
    assert miko["actions"][0]["date"] == "2023-06-05"
    assert any(item["status"] == "SAME_POLICY_OBJECT" and item["validation_state"] == "VALID" for item in miko["relations"])
    miko_relation = next(item for item in miko["relations"] if item["object_id"] == "pirha-2023-9112-bergbom")
    assert miko_relation["normalized_target"] == "Pirkanmaan aluevaltuuston jäsenyydestä ja 2. varapuheenjohtajan tehtävästä eroaminen"
    assert miko_relation["target_scope"] == "PARTIAL_COMPONENT"
    assert miko["target"]["normalized_object"] == miko_relation["normalized_target"]
    assert miko["actions"][0]["evidence_ids"]
    assert all(ref in {item["evidence_id"] for item in miko["evidence"]} for ref in miko["actions"][0]["evidence_ids"])
    assert any("city-council" in residual for coverage in miko["coverage"] for residual in coverage["scope_residuals"])
    miko_object = next(item for item in miko["retrieved_objects"] if item["object_id"] == "pirha-2023-9112-bergbom")
    miko_actor = next(iter(miko_object["authors"]))
    assert miko_actor["name"] == "Miko Bergbom"
    assert miko_actor["actor_id"] == "mp-1514"
    assert miko_actor["role"] == "ACTOR"
    assert miko_actor["identity_basis"] == "INDEPENDENT_CASE_REVIEW"
    assert not miko_actor.get("person_id")
    assert {"yle2023-1029-3-e", "candidacy-vaalit-2023-07-149-e", "pirha-2023-9112-bergbom:request", "pirha-2023-9112-bergbom:decision"} <= set(miko_actor["evidence_ids"])

    armi = traces["yle2011-2875-p2"]
    assert armi["authority"]["condition_state"] == "NOT_SATISFIED"
    assert armi["authority"]["opportunity_state"] == "NO_OBSERVABLE_OPPORTUNITY"
    assert not armi["actions"]
    assert "ei päätellä" in armi["assessment"]["claim"]

    sanna = traces["yle2023-482-1-p4"]
    assert not sanna["actions"]
    assert any(item["status"] == "UNRESOLVED" and item["validation_state"] == "VALID" for item in sanna["relations"])
    assert "samaa asiaa" in sanna["assessment"]["claim"]

    for packet in traces.values():
        current = {key: value for key, value in packet.items() if key not in {"evidence", "correction_history"}}
        assert evidence_reference_ids(current) <= {ref["evidence_id"] for ref in packet["evidence"]}

    assert (browser / "index.html").exists()
    index = json.loads((browser / "index.json").read_text(encoding="utf-8"))
    assert {person["name"] for person in index["people"]} >= {"Miko Bergbom", "Armi Lindell", "Sanna Antikainen"}
    assert any(trace["state"] == "OBSERVED_ALIGNED_ACTION" for trace in index["traces"])
    assert "OBSERVED_ALIGNED_ACTION" in (browser / "traces" / "trace-yle2023-1029-3-p1.json").read_text(encoding="utf-8")
    object_page = (browser / "objects" / "pirha-2023-9112-bergbom.html").read_text(encoding="utf-8")
    assert "Miko Bergbom" in object_page
    assert "9ff99c9682c7d6c98b590402785a86f2d1c480f54d2eab911d66e36225fd07d5" in object_page
    assert "Ratkaisua ei varmennettu tästä lähteestä" not in object_page
    assert (Path(result["export_dir"]) / "evidence_traces.jsonl").exists()
    assert (Path(result["report_dir"]) / "relation_benchmark.json").exists()


def test_stale_version_bound_review_abstains(tmp_path):
    db_path = tmp_path / "data" / "paa.sqlite"
    seed_frozen_database(db_path, DEFAULT_FIXTURE)
    conn = connect(db_path)
    review = json.loads(conn.execute("SELECT json FROM relation_reviews WHERE review_id = 'review-miko-pirha'").fetchone()[0])
    review["statement_sha256"] = "0" * 64
    conn.execute("UPDATE relation_reviews SET json = ? WHERE review_id = 'review-miko-pirha'", (json.dumps(review, ensure_ascii=False),))
    conn.commit()
    compile_database(conn)
    conn.commit()
    row = conn.execute("SELECT json FROM evidence_traces WHERE proposition_id = 'yle2023-1029-3-p1'").fetchone()
    packet = json.loads(row[0])
    assert packet["assessment"]["state"] == "INSUFFICIENT_EVIDENCE"
    assert not packet["actions"]
    assert all(item.get("validation_state") != "VALID" for item in packet["relations"] if item["object_id"] == "pirha-2023-9112-bergbom")
    conn.close()


def test_action_outside_statement_window_does_not_close_promise(tmp_path):
    db_path = tmp_path / "data" / "paa.sqlite"
    seed_frozen_database(db_path, DEFAULT_FIXTURE)
    conn = connect(db_path)
    conn.execute(
        "UPDATE official_objects SET json = replace(json, '2023-06-05', '2022-01-01') WHERE object_id = 'pirha-2023-9112-bergbom'"
    )
    compile_database(conn)
    packet = json.loads(conn.execute("SELECT json FROM evidence_traces WHERE proposition_id = 'yle2023-1029-3-p1'").fetchone()[0])
    assert packet["actions"] == []
    assert packet["assessment"]["state"] == "INSUFFICIENT_EVIDENCE"

    conn.execute(
        "UPDATE official_objects SET json = replace(json, '2022-01-01', '2027-01-01') WHERE object_id = 'pirha-2023-9112-bergbom'"
    )
    compile_database(conn)
    packet = json.loads(conn.execute("SELECT json FROM evidence_traces WHERE proposition_id = 'yle2023-1029-3-p1'").fetchone()[0])
    assert packet["actions"] == []
    assert packet["assessment"]["state"] == "INSUFFICIENT_EVIDENCE"
    conn.close()


def test_frozen_build_is_repeatable(tmp_path):
    first = build_frozen(tmp_path)
    first_traces = (Path(first["export_dir"]) / "evidence_traces.jsonl").read_text(encoding="utf-8")
    second = build_frozen(tmp_path, overwrite=True)
    second_traces = (Path(second["export_dir"]) / "evidence_traces.jsonl").read_text(encoding="utf-8")
    assert first_traces == second_traces
    assert first["seed"]["fixture_sha256"] == second["seed"]["fixture_sha256"]


def test_frozen_seed_overwrite_clears_live_vote_rows(tmp_path):
    """An explicit frozen overwrite cannot inherit a prior live corpus."""
    db_path = tmp_path / "data" / "paa.sqlite"
    seed_frozen_database(db_path, DEFAULT_FIXTURE)
    conn = connect(db_path)
    conn.execute(
        "INSERT INTO vote_events(aanestys_id, year, session_date, title, mitatoity) VALUES (?, ?, ?, ?, ?)",
        ("stale-vote", 2026, "2026-01-01", "stale", 0),
    )
    conn.execute(
        "INSERT INTO ballots(aanestys_id, person_number, raw_response) VALUES (?, ?, ?)",
        ("stale-vote", "9999", "JAA"),
    )
    conn.commit()
    conn.close()

    seed_frozen_database(db_path, DEFAULT_FIXTURE, overwrite=True)
    conn = sqlite3.connect(db_path)
    assert conn.execute("SELECT COUNT(*) FROM vote_events").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM ballots").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM meta").fetchone()[0] == 0
    conn.close()


def test_relation_benchmark_declares_illustrative_scope(tmp_path):
    result = build_frozen(tmp_path)
    report = json.loads((Path(result["report_dir"]) / "relation_benchmark.json").read_text(encoding="utf-8"))
    assert report["evaluation_scope"] == "frozen_fixture_only_not_held_out"
    assert report["gold_reviewed_relations"] == 1
    # The expanded frozen corpus contains 277 real speeches and one question
    # episode. Lexical retrieval misses the resignation object; its persisted
    # review is injected for replay and must never count as retrieval success.
    assert report["top_k_recall"] == {"1": 0.0, "3": 0.0, "5": 0.0}
    assert report["admitted_relation_precision"] == 1.0
    assert report["abstention_rate"] == 0.5
    assert report["review_reference_injections_excluded_from_recall"] == 1


def test_frozen_question_and_speech_sources_reach_real_browser(tmp_path):
    result = build_frozen(tmp_path)
    conn = sqlite3.connect(result["db_path"])
    sources = [json.loads(row[0]) for row in conn.execute("SELECT json FROM official_objects")]
    assert sum(s["kind"] == "SPEECH" for s in sources) == 277
    assert sum(s["kind"] == "WRITTEN_QUESTION" for s in sources) == 1
    episode = json.loads(conn.execute("SELECT json FROM decision_episodes").fetchone()[0])
    assert episode["matter_id"] == "KK 1/2023 vp"
    assert episode["episode_state"] == "ANSWERED_INSTITUTIONALLY"
    conn.close()
    browser = Path(result["output_dir"])
    index = (browser / "index.html").read_text()
    assert 'id="person-issue"' in index
    assert (browser / "episodes/index.html").is_file()
    question = next(s for s in sources if s["kind"] == "WRITTEN_QUESTION")
    from paa.site import _safe
    page = (browser / "objects" / (_safe(question["object_id"])+".html")).read_text()
    assert "Fortumin" in page
    assert "2023-04-20" in page

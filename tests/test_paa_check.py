import json
from pathlib import Path

from paa.check import ballot_discrepancies, run
from paa.frozen import build_frozen
from paa.store import connect


def test_matching_person_count_does_not_hide_wrong_vote_choices(tmp_path):
    conn = connect(tmp_path / "paa.sqlite")
    conn.execute("INSERT INTO vote_events(aanestys_id,year,jaa,ei,tyhjaa,poissa,yhteensa) VALUES ('v',2024,1,1,0,0,2)")
    conn.execute("INSERT INTO ballots(aanestys_id,person_number,raw_response) VALUES ('v','a','JAA')")
    conn.execute("INSERT INTO ballots(aanestys_id,person_number,raw_response) VALUES ('v','b','EI')")
    assert ballot_discrepancies(conn) == []
    conn.execute("UPDATE ballots SET raw_response='JAA' WHERE person_number='b'")
    failures = ballot_discrepancies(conn)
    assert len(failures) == 1
    assert failures[0]["stored"] == failures[0]["published"] == 2
    assert failures[0]["observed_jaa"] != failures[0]["jaa"]
    conn.close()


def test_coverage_source_references_must_resolve_inside_downloaded_packet(tmp_path):
    result = build_frozen(tmp_path)
    database = Path(result["db_path"])
    assert run(database, full_corpus=False) == []
    conn = connect(database)
    row = conn.execute("SELECT trace_id,json FROM evidence_traces WHERE proposition_id='yle2023-1029-3-p1'").fetchone()
    packet = json.loads(row["json"])
    packet["coverage"][0]["enumeration_evidence_ids"].append("missing-source-page")
    conn.execute("UPDATE evidence_traces SET json=? WHERE trace_id=?", (json.dumps(packet), row["trace_id"]))
    conn.commit()
    conn.close()
    assert any("dangling evidence IDs" in problem for problem in run(database, full_corpus=False))

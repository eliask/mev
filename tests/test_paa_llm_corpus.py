"""Candidate inventory must preserve actor/time limits and failed coverage."""
import json
import sqlite3

from paa.llm_corpus import inventory
from paa.semantics import Proposition


def test_inventory_cannot_turn_failed_receipts_or_other_people_into_missing_action(tmp_path, monkeypatch):
    db = tmp_path / "source.sqlite"
    conn = sqlite3.connect(db)
    conn.executescript("""
        CREATE TABLE documents(document_id TEXT, text TEXT, field_label TEXT, stated_earliest TEXT);
        CREATE TABLE statements(statement_id TEXT, json TEXT);
        CREATE TABLE actors(actor_id TEXT, person_id TEXT, identity_status TEXT);
        CREATE TABLE official_objects(json TEXT);
    """)
    conn.execute("INSERT INTO actors VALUES ('actor', '42', 'MP_UNIQUE')")
    for doc_id in ("valid", "failed"):
        conn.execute("INSERT INTO documents VALUES (?, 'Teen koulutusaloitteen.', 'promise', '2023-03-01')", (doc_id,))
        statement = {"statement_id": doc_id, "issuer_actor_ids": ["actor"],
                     "stated_at": {"latest": "2023-03-01"}, "evidence": [{"evidence_id": "source"}]}
        conn.execute("INSERT INTO statements VALUES (?, ?)", (doc_id, json.dumps(statement)))
    for obj_id, person, action_date in (("own-later", "42", "2023-06-01"),
                                       ("other-person", "43", "2023-06-01"),
                                       ("before-statement", "42", "2023-01-01"),
                                       ("after-cutoff", "42", "2027-01-01")):
        obj = {"object_id": obj_id, "matter_id": obj_id, "kind": "LEGISLATIVE_INITIATIVE",
               "text": "koulutusaloitteen koulutus", "title": "Koulutus", "action_date": action_date,
               "authors": [{"person_id": person}]}
        conn.execute("INSERT INTO official_objects VALUES (?)", (json.dumps(obj),))
    conn.commit()
    conn.close()
    run = tmp_path / "run"
    (run / "documents").mkdir(parents=True)
    for doc_id, state in (("valid", "OK"), ("failed", "FAILED")):
        (run / "documents" / f"{doc_id}.json").write_text(json.dumps({"receipt_status": state}))

    class Interpretations:
        def __init__(self, path):
            self.run_id, self.manifest = "run", {"identity": {"prompt": "test"}}
            self.applied = self.abstained = 0

        def propositions(self, doc, original):
            assert doc["document_id"] == "valid", "failed receipts must never launch retrieval"
            self.applied += 1
            return [Proposition(text=doc["text"], semantic_type="PERSONAL_ACTION_COMMITMENT",
                                testability="UNRESOLVED", targets=["koulutus"], validation_state="PROPOSED")]

    monkeypatch.setattr("paa.llm_corpus.ModelInterpretations", Interpretations)
    output = tmp_path / "pairs.jsonl"
    result = inventory(db, run, output)
    pairs = [json.loads(line) for line in output.read_text().splitlines()]
    assert [pair["official_object"]["object_id"] for pair in pairs] == ["own-later"]
    coverage = json.loads(output.with_suffix(".coverage.json").read_text())
    assert coverage["documents"][1]["state"] == "CANDIDATE_RETRIEVAL_COMPLETED"  # sorted documents
    assert coverage["documents"][0]["state"] == "MODEL_RECEIPT_FAILED"
    assert all(not row["absence_claim"] for row in coverage["documents"])
    assert result["counts"]["model_receipts_FAILED"] == 1

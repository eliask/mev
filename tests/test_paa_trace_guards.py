"""Meaningful adversarial checks for the canonical admission boundary."""
import json

import pytest

from paa.frozen import seed_frozen_database
from paa.pipeline import compile_database
from paa.relations import ObjectRetriever, fingerprint, verified_review
from paa.store import connect


def sample_review():
    statement = {"original_text": "Teen lakialoitteen koulutuksesta.", "statement_id": "s", "evidence": [{"evidence_id": "s-e"}]}
    prop = {"proposition_id": "s-p1", "original_text": statement["original_text"]}
    obj = {"object_id": "o", "matter_id": "LA 1/2026 vp", "text": "Lakialoite koulutuksesta", "evidence_ids": ["o-e"]}
    evidence = {"s-e": {}, "o-e": {}}
    review = {"review_id": "review-s-o", "proposition_id": "s-p1", "object_id": "o", "matter_id": "LA 1/2026 vp", "status": "SAME_POLICY_OBJECT",
              "normalized_target": "education bill",
              "statement_sha256": fingerprint(statement["original_text"]), "object_sha256": fingerprint(obj["text"]),
              "statement_quote": "lakialoitteen koulutuksesta", "object_quote": obj["text"],
              "reviewer": "test reviewer", "rationale": "same named object", "evidence_ids": ["s-e", "o-e"]}
    return review, statement, prop, obj, evidence


@pytest.mark.parametrize("field,value", [("normalized_target", ["x"]), ("reviewer", 123),
                                       ("status", ["SAME_POLICY_OBJECT"]), ("evidence_ids", "s-e"),
                                       ("evidence_ids", ["s-e", {}])])
def test_malformed_review_fields_cannot_admit(field, value):
    review, statement, prop, obj, evidence = sample_review()
    review[field] = value
    assert not verified_review(review, statement, prop, obj, evidence)


def test_keyword_retrieval_never_admits_identity():
    retriever = ObjectRetriever([{"object_id": "wrong", "title": "Koulutuksesta", "text": ""}])
    result = retriever.search("Teen lakialoitteen koulutuksesta.")
    assert result and all(row["status"] == "CANDIDATE" for row in result)


def test_changed_source_invalidates_review_even_if_id_survives():
    review, statement, prop, obj, evidence = sample_review()
    assert verified_review(review, statement, prop, obj, evidence)
    obj["text"] += " (ehdotus hylättiin)"
    assert not verified_review(review, statement, prop, obj, evidence)


def test_vote_metadata_shell_cannot_replace_reviewed_official_source(tmp_path, monkeypatch):
    database = tmp_path / "paa.sqlite"
    seed_frozen_database(database)
    conn = connect(database)
    obj = next(json.loads(row["json"]) for row in conn.execute("SELECT json FROM official_objects")
               if json.loads(row["json"])["kind"] == "VOTE")
    shell = {**obj, "text": "Metadata title without the decisive motion"}
    monkeypatch.setattr("paa.traces._vote_objects", lambda _: [shell])
    compile_database(conn)
    packet = json.loads(conn.execute("SELECT json FROM evidence_traces WHERE proposition_id='yle2023-482-1-p4'").fetchone()[0])
    source = next(item for item in packet["retrieved_objects"] if item["object_id"] == obj["object_id"])
    assert source["text"] == obj["text"]
    assert any(item["object_id"] == obj["object_id"] and item["validation_state"] == "VALID" for item in packet["relations"])
    conn.close()


def test_dangling_evidence_or_unquoted_rationale_cannot_admit():
    review, statement, prop, obj, evidence = sample_review()
    evidence.pop("o-e")
    assert not verified_review(review, statement, prop, obj, evidence)
    review, statement, prop, obj, evidence = sample_review()
    review["object_quote"] = "other matter"
    assert not verified_review(review, statement, prop, obj, evidence)


def test_existing_unrelated_evidence_cannot_substitute_for_source_refs():
    review, statement, prop, obj, evidence = sample_review()
    evidence["unrelated-e"] = {}
    review["evidence_ids"] = ["unrelated-e"]
    assert not verified_review(review, statement, prop, obj, evidence)


def test_changed_matter_identity_invalidates_unchanged_text_review():
    review, statement, prop, obj, evidence = sample_review()
    obj["matter_id"] = "LA 2/2026 vp"
    assert not verified_review(review, statement, prop, obj, evidence)


def test_uninterpreted_target_cannot_admit_a_review():
    review, statement, prop, obj, evidence = sample_review()
    review.pop("normalized_target")
    assert not verified_review(review, statement, prop, obj, evidence)


def test_unexplained_action_date_cannot_close_a_real_promise(tmp_path):
    database = tmp_path / "paa.sqlite"
    seed_frozen_database(database)
    conn = connect(database)
    compile_database(conn)
    baseline = json.loads(conn.execute("SELECT json FROM evidence_traces WHERE proposition_id = ?", ("yle2023-1029-3-p1",)).fetchone()[0])
    assert baseline["assessment"]["state"] == "OBSERVED_ALIGNED_ACTION"
    row = conn.execute(
        "SELECT object_id, json FROM official_objects WHERE object_id = ?", ("pirha-2023-9112-bergbom",)
    ).fetchone()
    obj = json.loads(row["json"])
    obj["action_date_basis"] = "UNVERIFIED_METADATA_DATE"
    conn.execute("UPDATE official_objects SET json = ? WHERE object_id = ?", (json.dumps(obj), row["object_id"]))
    compile_database(conn)
    packet = json.loads(conn.execute("SELECT json FROM evidence_traces WHERE proposition_id = ?", ("yle2023-1029-3-p1",)).fetchone()[0])
    assert not packet["actions"]
    assert packet["assessment"]["state"] == "INSUFFICIENT_EVIDENCE"
    conn.close()


def test_model_proposal_cannot_reach_an_aligned_trace(tmp_path):
    """Anchored model output remains a proposal until a source review admits it."""

    database = tmp_path / "paa.sqlite"
    seed_frozen_database(database)
    conn = connect(database)
    review = json.loads(
        conn.execute(
            "SELECT json FROM relation_reviews WHERE review_id = ?", ("review-miko-pirha",)
        ).fetchone()[0]
    )
    review.update(
        {
            "review_method": "LOCAL_LLM_PROPOSAL",
            "review_state": "PROPOSED",
            "validation_state": "PROPOSED",
            "admission_state": "PROPOSED",
            "admission_route": "MODEL_PROPOSAL_NOT_ADMITTED",
            "reviewer": "local-llm-proposer:test-model:relation_v1",
        }
    )
    conn.execute(
        "UPDATE relation_reviews SET json = ? WHERE review_id = ?",
        (json.dumps(review, ensure_ascii=False), "review-miko-pirha"),
    )
    conn.commit()
    compile_database(conn)
    packet = json.loads(
        conn.execute(
            "SELECT json FROM evidence_traces WHERE proposition_id = ?", ("yle2023-1029-3-p1",)
        ).fetchone()[0]
    )
    relation = next(item for item in packet["relations"] if item["object_id"] == "pirha-2023-9112-bergbom")
    assert relation["validation_state"] != "VALID"
    assert relation["status"] == "UNRESOLVED"
    assert packet["actions"] == []
    assert packet["assessment"]["state"] != "OBSERVED_ALIGNED_ACTION"
    conn.close()


def test_unknown_statement_upper_bound_cannot_prove_a_later_action(tmp_path):
    database = tmp_path / "paa.sqlite"
    seed_frozen_database(database)
    conn = connect(database)
    compile_database(conn)
    baseline = json.loads(conn.execute("SELECT json FROM evidence_traces WHERE proposition_id = ?", ("yle2023-1029-3-p1",)).fetchone()[0])
    assert baseline["assessment"]["state"] == "OBSERVED_ALIGNED_ACTION"
    conn.execute("UPDATE documents SET source_id = 'SRC-UNBOUNDED' WHERE document_id = 'yle2023-1029-3'")
    compile_database(conn)
    packet = json.loads(conn.execute("SELECT json FROM evidence_traces WHERE proposition_id = ?", ("yle2023-1029-3-p1",)).fetchone()[0])
    assert packet["authority"]["window"]["earliest"] is None
    assert not packet["actions"]
    assert packet["assessment"]["state"] == "INSUFFICIENT_EVIDENCE"
    conn.close()


def test_election_condition_does_not_spread_to_other_source_propositions(tmp_path):
    database = tmp_path / "paa.sqlite"
    seed_frozen_database(database)
    conn = connect(database)
    compile_database(conn)
    packets = {r["proposition_id"]: json.loads(r["json"]) for r in conn.execute(
        "SELECT proposition_id,json FROM evidence_traces WHERE statement_id='yle2011-2875'"
    )}
    assert packets["yle2011-2875-p1"]["authority"]["condition_state"] == "NOT_APPLICABLE"
    assert packets["yle2011-2875-p2"]["authority"]["condition_state"] == "NOT_SATISFIED"
    conn.close()


def test_unresolved_interpretation_does_not_become_known_non_testable(tmp_path):
    database = tmp_path / "paa.sqlite"
    seed_frozen_database(database)
    conn = connect(database)
    compile_database(conn)
    ambiguous = [json.loads(row["json"]) for row in conn.execute("SELECT json FROM evidence_traces")
                 if json.loads(row["json"])["proposition"]["semantic_type"] == "AMBIGUOUS"]
    assert ambiguous
    assert all(packet["assessment"]["state"] == "INSUFFICIENT_EVIDENCE" for packet in ambiguous)
    assert all("tulkinta ei riitä" in packet["assessment"]["claim"] for packet in ambiguous)
    conn.close()

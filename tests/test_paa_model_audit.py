"""Regression tests for the offline proposed-model audit."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

from paa.model_audit import audit_retrieval, audit_run, evaluation_limits


def _make_db(path: Path, *, text: str = "Hoitajamitoitus kirjataan lakiin.") -> None:
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE documents (
            document_id TEXT PRIMARY KEY, source_id TEXT, url TEXT, actor_id TEXT,
            field_label TEXT, language TEXT, text TEXT, stated_earliest TEXT,
            sha256 TEXT, http_status INTEGER
        );
        CREATE TABLE propositions (proposition_id TEXT PRIMARY KEY, statement_id TEXT, json TEXT NOT NULL);
        CREATE TABLE official_objects (object_id TEXT PRIMARY KEY, json TEXT NOT NULL);
        """
    )
    document_id = "doc-1"
    conn.execute(
        "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (document_id, "SRC-1", "https://example.test/1", "actor-1", "answer", "fi", text, "2023-01-01", hashlib.sha256(text.encode()).hexdigest(), 200),
    )
    conn.execute(
        "INSERT INTO propositions VALUES (?, ?, ?)",
        (
            "doc-1-p1",
            document_id,
            json.dumps({"source_text": text, "source_span": {"start": 0, "end": len(text)}}),
        ),
    )
    conn.commit()
    conn.close()


def test_audit_separates_failed_receipt_from_semantic_abstention(tmp_path: Path):
    db = tmp_path / "paa.sqlite"
    _make_db(db)
    run = tmp_path / "run"
    (run / "documents").mkdir(parents=True)
    (run / "manifest.json").write_text(json.dumps({"admission_state": "PROPOSED"}), encoding="utf-8")
    (run / "documents" / "doc-1.json").write_text(
        json.dumps(
            {
                "document_id": "doc-1",
                "receipt_status": "FAILED",
                "status": "INVALID",
                "validation_state": "PROPOSED",
                "source_text_sha256": hashlib.sha256(b"Hoitajamitoitus kirjataan lakiin.").hexdigest(),
                "invalid_records": [{"code": "INVALID_MULTI_OUTPUT"}],
                "units": [{"unit_id": "doc-1-p1", "status": "INVALID", "raw": None}],
            }
        ),
        encoding="utf-8",
    )
    report = audit_run(db=db, run=run, include_retrieval=False)
    assert report["receipt_class"] == {"TRANSPORT_FAILURE": 1}
    assert report["transport_failures"]["documents"] == 1
    assert "semantic_abstention_documents" not in report["semantic_outcomes"]
    assert report["terminal_no_promotion"]["passed"] is True


def test_retrieval_reports_positive_clause_coverage_and_recall(tmp_path: Path):
    db = tmp_path / "paa.sqlite"
    _make_db(db)
    object_text = "Lakialoitteessa hoitajamitoitus kirjataan lakiin."
    object_id = "eduskunta:LA 1/2023 vp"
    conn = sqlite3.connect(db)
    conn.execute(
        "INSERT INTO official_objects VALUES (?, ?)",
        (
            object_id,
            json.dumps(
                {
                    "object_id": object_id,
                    "title": "Lakialoite hoitajamitoituksesta",
                    "text": object_text,
                    "matter_id": object_id,
                }
            ),
        ),
    )
    conn.commit()
    conn.close()
    source_text = "Pidän tärkeänä hoitajamitoitusta."
    source_quote = source_text
    object_quote = object_text
    gold = {
        "pair_id": "pair-1",
        "split": "heldout",
        "source": {
            "source_quote": source_quote,
            "source_text": source_text,
            "source_text_sha256": hashlib.sha256(source_text.encode()).hexdigest(),
        },
        "object": {
            "object_id": object_id,
            "object_quote": object_quote,
            "object_sha256": hashlib.sha256(object_text.encode()).hexdigest(),
        },
        "gold": {"status": "SAME_POLICY_OBJECT"},
    }
    fixture = tmp_path / "relation.jsonl"
    fixture.write_text(json.dumps(gold, ensure_ascii=False) + "\n", encoding="utf-8")
    report = audit_retrieval(db, fixture)
    assert report["positive_denominator"] == 1
    assert report["positive_clause_coverage"]["all_required_exact_anchors"] is True
    assert report["recall_at_k"]["1"]["hits"] == 1
    assert "not relation absence" in report["interpretation"]


def test_evaluation_limits_expose_small_non_independent_semantic_holdout():
    report = evaluation_limits()
    semantic = report["semantic_gold"]
    relation = report["relation_gold"]
    assert semantic["total"] == 60
    assert semantic["heldout_rows"] == 30
    assert semantic["not_an_independent_model_benchmark"] is True
    assert relation["total"] == 400
    assert relation["positive_denominator"] == 17

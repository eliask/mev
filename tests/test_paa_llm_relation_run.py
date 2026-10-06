"""Offline tests for the resumable local relation runner."""

from __future__ import annotations

import json
from pathlib import Path

from paa.llm_evaluation import evaluate_relation_predictions
from paa.llm_relation_run import (
    _apply_context_coverage,
    _is_resumable,
    _receipt_identity,
    _relation_batches,
    _relation_partitions,
    load_relation_input,
    public_record_to_relation_pair,
    run_relation_evaluation,
)


class FakeRelationClient:
    """Schema-shaped fake; no loopback request is made by these tests."""

    def __init__(self) -> None:
        self.manifest = {"model_id": "fixture-fake-model", "client_version": "test"}
        self.calls: list[tuple[str, dict]] = []

    async def discover(self) -> dict:
        return self.manifest

    async def close(self) -> None:
        return None

    async def request(self, task: str, system: str, user: str, **_kwargs) -> dict:
        payload = json.loads(user)
        self.calls.append((task, payload))
        pairs = []
        for item in payload["pairs"]:
            pair_id = item["pair_id"]
            pair_input = item["input"]
            if ":verification-batch-" in task or "-verification:" in task:
                proposal = pair_input["proposal"]
                pairs.append({"pair_id": pair_id, "verdict": "SUPPORTED_WITHIN_SCOPE", **proposal})
                continue
            proposition = pair_input["proposition"]
            official = pair_input["official_object"]
            statement = pair_input.get("statement") or payload["shared_statement"]
            status = "SAME_POLICY_OBJECT" if pair_id.endswith("-1") else "REJECTED"
            admitted = status in {"SAME_POLICY_OBJECT", "SAME_MATTER"}
            pairs.append({
                "pair_id": pair_id,
                "status": status,
                "matter_id": official["matter_id"],
                "identity_basis": "EXPLICIT_MATTER_ID" if admitted else "UNRESOLVED",
                "statement_quote": proposition["source_text"],
                "object_quote": official["text"][:120],
                "normalized_target": "fixture target" if admitted else None,
                "target_scope": "EXACT" if admitted else "UNRESOLVED",
                "bounded_claim": "The supplied passages support only this bounded relation.",
                "rationale": "The exact source anchors support this conservative candidate.",
                "action_alignment": "RELATED" if admitted else "UNRESOLVED",
                "evidence_ids": [statement["evidence_ids"][0], official["evidence_ids"][0]],
                "counterevidence_ids": [],
            })
        return {"status": "OK", "request_id": task, "parsed": {"pairs": pairs}}


def test_context_coverage_keeps_model_clip_separate_from_raw_original_availability():
    batch = {
        "validation_requests": {
            "pair-1": {
                "coverage": {
                    "statement_complete": True,
                    "object_complete": False,
                },
                "model_input": {},
            },
        },
        "model_input": {
            "pairs": [{"pair_id": "pair-1", "input": {}}],
        },
    }

    _apply_context_coverage(
        batch,
        [{
            "pair_id": "pair-1",
            "_receipt_context": {
                "statement_context_complete": True,
                "object_context_complete": True,
            },
        }],
    )

    coverage = batch["validation_requests"]["pair-1"]["coverage"]
    assert coverage["statement_context_complete"] is True
    assert coverage["object_context_complete"] is False
    assert coverage["statement_clip_complete"] is True
    assert coverage["object_clip_complete"] is False
    assert coverage["official_context_kind"] == "OFFICIAL_OBJECT_EXCERPT_ONLY"
    assert coverage["originals_available"] == {"statement": True, "object": True}
    assert batch["model_input"]["pairs"][0]["coverage"] == coverage


def test_short_fixture_excerpt_is_not_reported_as_complete_original_context():
    batch = {
        "validation_requests": {
            "pair-1": {
                "coverage": {"statement_complete": True, "object_complete": True},
                "model_input": {},
            },
        },
        "model_input": {"pairs": [{"pair_id": "pair-1", "input": {}}]},
    }

    _apply_context_coverage(
        batch,
        [{
            "pair_id": "pair-1",
            "_receipt_context": {
                "statement_context_complete": False,
                "object_context_complete": False,
            },
        }],
    )

    coverage = batch["validation_requests"]["pair-1"]["coverage"]
    assert coverage["statement_clip_complete"] is True
    assert coverage["object_clip_complete"] is True
    assert coverage["statement_context_complete"] is False
    assert coverage["object_context_complete"] is False
    assert coverage["source_context_kind"] == "FOCUSED_SOURCE_QUOTE_ONLY"


def test_receipt_identity_binds_served_runtime_manifest_for_same_alias():
    public, gold = _tiny_public_and_gold()
    manifest = {"model_id": "same-alias", "server_build": "build-a", "chat_template_sha256": "template-a"}
    unchanged = _receipt_identity(public[0], gold[0], prompt_version="relation_v1", manifest=manifest)
    changed = _receipt_identity(
        public[0],
        gold[0],
        prompt_version="relation_v1",
        manifest={**manifest, "server_build": "build-b"},
    )
    receipt = {
        "identity": unchanged,
        "prediction": {},
        "proposal": {"status": "OK", "validation": {"valid": True}},
        "run": {"status": "COMPLETE"},
    }

    assert _is_resumable(receipt, unchanged) is True
    assert _is_resumable(receipt, changed) is False


def test_receipt_identity_binds_single_pair_model_projection(monkeypatch):
    public, gold = _tiny_public_and_gold()
    manifest = {"model_id": "same-alias", "server_build": "build-a"}
    unchanged = _receipt_identity(public[0], gold[0], prompt_version="relation_v1", manifest=manifest)
    receipt = {
        "identity": unchanged,
        "prediction": {},
        "proposal": {"status": "OK", "validation": {"valid": True}},
        "run": {"status": "COMPLETE"},
    }

    original_builder = __import__("paa.llm_relation_run", fromlist=["build_batch_relation_request"]).build_batch_relation_request

    def changed_projection(*args, **kwargs):
        request = original_builder(*args, **kwargs)
        request["model_input"]["projection_test_sentinel"] = "changed"
        return request

    monkeypatch.setattr("paa.llm_relation_run.build_batch_relation_request", changed_projection)
    changed = _receipt_identity(public[0], gold[0], prompt_version="relation_v1", manifest=manifest)

    assert changed["model_projection_version"] == unchanged["model_projection_version"]
    assert changed["model_projection_sha256"] != unchanged["model_projection_sha256"]
    assert _is_resumable(receipt, unchanged) is True
    assert _is_resumable(receipt, changed) is False


def test_receipt_identity_invalidates_when_projection_contract_changes(monkeypatch):
    public, gold = _tiny_public_and_gold()
    manifest = {"model_id": "same-alias", "server_build": "build-a"}
    unchanged = _receipt_identity(public[0], gold[0], prompt_version="relation_v1", manifest=manifest)
    receipt = {
        "identity": unchanged,
        "prediction": {},
        "proposal": {"status": "OK", "validation": {"valid": True}},
        "run": {"status": "COMPLETE"},
    }

    monkeypatch.setattr(
        "paa.llm_relation_run.MODEL_PROJECTION_VERSION",
        "relation-model-projection-test-next",
    )
    changed = _receipt_identity(public[0], gold[0], prompt_version="relation_v1", manifest=manifest)

    assert changed["model_projection_version"] != unchanged["model_projection_version"]
    assert changed["model_projection_sha256"] != unchanged["model_projection_sha256"]
    assert _is_resumable(receipt, changed) is False


def _tiny_public_and_gold() -> tuple[list[dict], list[dict]]:
    """Stable runner mechanics fixture independent of the editable 400-row set."""

    source_text = "Pidän tärkeänä vanhuspalvelujen riittäviä resursseja."
    object_text = "Lakialoitteessa ehdotetaan vanhuspalvelulain resurssien turvaamista."
    public = []
    gold = []
    for index in range(1, 4):
        pair_id = f"test-relation-{index}"
        source = {
            "document_id": f"test-source-{index}",
            "source_id": "TEST-SOURCE",
            "source_year": 2023,
            "language": "fi",
            "original_question": "Mitä asioita haluat edistää?",
            "source_quote": source_text,
            "candidate_context": {"actor_id": "actor-1", "display_name": "Test Actor"},
        }
        official = {
            "object_id": f"test-object-{index}",
            "matter_id": f"TEST {index}/2023 vp",
            "kind": "LEGISLATIVE_INITIATIVE",
            "title": "Vanhuspalvelulain resurssit",
            "object_year": 2024,
            "object_quote": object_text,
            "actor_context": {"matched_name": "Test Actor", "person_id": "actor-1", "role": "AUTHOR"},
        }
        public.append({"pair_id": pair_id, "source": source, "object": official})
        gold.append({
            "pair_id": pair_id,
            "split": "development",
            "source": source,
            "object": official,
            "gold": {
                "status": "SAME_POLICY_OBJECT" if index == 1 else "REJECTED",
                "target_scope": "EXACT" if index == 1 else "UNRESOLVED",
                "domain_relation": "SAME" if index == 1 else "RELATED",
                "actor_eligibility": "MATCHED_AUTHOR",
                "time_eligibility": "AFTER_SOURCE_YEAR",
                "terminal_eligibility": "ADMISSIBLE" if index == 1 else "NOT_ADMISSIBLE",
                "action_alignment": "RELATED" if index == 1 else "UNRESOLVED",
            },
        })
    return public, gold


def test_native_jsonl_loader_preserves_unicode_line_separator_inside_source(tmp_path: Path):
    path = tmp_path / "native.jsonl"
    rows = [
        {"pair_id": "unicode-1", "source": {"source_quote": "Ensimmäinen\u2028lause"}, "object": {}},
        {"pair_id": "unicode-2", "source": {"source_quote": "Toinen"}, "object": {}},
    ]
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")

    loaded = load_relation_input(path)

    assert [row["pair_id"] for row in loaded] == ["unicode-1", "unicode-2"]
    assert loaded[0]["source"]["source_quote"] == "Ensimmäinen\u2028lause"


def test_native_batches_keep_one_proposition_objects_together():
    rows = [
        {"pair_id": "a", "statement": {"statement_id": "s"}, "proposition": {"proposition_id": "p1"}},
        {"pair_id": "b", "statement": {"statement_id": "s"}, "proposition": {"proposition_id": "p1"}},
        {"pair_id": "c", "statement": {"statement_id": "s"}, "proposition": {"proposition_id": "p1"}},
        {"pair_id": "d", "statement": {"statement_id": "s"}, "proposition": {"proposition_id": "p2"}},
    ]

    batches = list(_relation_batches(rows, 8, group_by_proposition=True))

    assert [[row["pair_id"] for row in batch] for batch in batches] == [["a", "b", "c"], ["d"]]


def test_parallel_partitions_keep_proposition_objects_together():
    rows = [
        {"pair_id": "a", "statement": {"statement_id": "s1"}, "proposition": {"proposition_id": "p1"}},
        {"pair_id": "b", "statement": {"statement_id": "s1"}, "proposition": {"proposition_id": "p1"}},
        {"pair_id": "c", "statement": {"statement_id": "s1"}, "proposition": {"proposition_id": "p1"}},
        {"pair_id": "d", "statement": {"statement_id": "s2"}, "proposition": {"proposition_id": "p2"}},
    ]

    partitions = _relation_partitions(rows, 2, group_by_proposition=True)

    assert [[row["pair_id"] for row in partition] for partition in partitions] == [["a", "b", "c"], ["d"]]


def test_public_adapter_keeps_model_input_free_of_gold_and_selection(tmp_path: Path):
    public, _gold = _tiny_public_and_gold()
    public = public[:1]
    pair = public_record_to_relation_pair(public[0])
    assert "gold" not in pair and "selection" not in pair
    assert pair["statement"]["original_text"] == public[0]["source"]["source_quote"]
    assert pair["official_object"]["text"] == public[0]["object"]["object_quote"]
    assert pair["proposition"]["semantic_type"] == "UNRESOLVED"


def test_runner_batches_verifies_same_status_and_resumes_per_pair(tmp_path: Path):
    public, gold = _tiny_public_and_gold()
    output = tmp_path / "relation.jsonl"
    first = FakeRelationClient()
    report = __import__("asyncio").run(run_relation_evaluation(
        split="development",
        prompt_version="relation_v1",
        output=output,
        client=first,
        records=public,
        gold_rows=gold,
        batch_size=2,
    ))
    assert report["pair_count"] == 3
    assert report["missing_pair_count"] == 0
    assert report["new_pair_count"] == 3
    assert len(output.read_text(encoding="utf-8").splitlines()) == 3
    assert any("proposal" in task for task, _payload in first.calls)
    assert any("verification" in task for task, _payload in first.calls)
    for task, payload in first.calls:
        assert "gold" not in json.dumps(payload, ensure_ascii=False)
        assert "adjudication" not in json.dumps(payload, ensure_ascii=False)
        assert "selection" not in json.dumps(payload, ensure_ascii=False)
    second = FakeRelationClient()
    resumed = __import__("asyncio").run(run_relation_evaluation(
        split="development",
        prompt_version="relation_v1",
        output=output,
        client=second,
        records=public,
        gold_rows=gold,
        batch_size=2,
    ))
    assert resumed["resumed_pair_count"] == 3
    assert resumed["new_pair_count"] == 0
    assert second.calls == []


def test_relation_score_counts_missing_in_all_pair_denominators():
    _public, rows = _tiny_public_and_gold()
    predictions = {rows[0]["pair_id"]: {"status": "UNRESOLVED"}}
    report = evaluate_relation_predictions(rows, predictions)
    assert report["missing_pair_count"] == 2
    assert report["field_accuracy"]["status"]["total"] == 3
    assert report["status_metrics"]["accuracy"]["total"] == 3
    assert report["abstention_count"] == 1
    assert report["status_metrics"]["false_negative"] == 1


def test_native_input_mode_accepts_canonical_pairs_without_gold(tmp_path: Path):
    public, _gold = _tiny_public_and_gold()
    canonical = public_record_to_relation_pair(public[0])
    input_path = tmp_path / "native.jsonl"
    input_path.write_text(json.dumps({key: canonical[key] for key in ("pair_id", "proposition", "statement", "official_object")}) + "\n", encoding="utf-8")
    output = tmp_path / "native-relation.jsonl"
    report = __import__("asyncio").run(run_relation_evaluation(
        split="development",
        prompt_version="relation_v2",
        output=output,
        input_path=input_path,
        no_score=True,
        client=FakeRelationClient(),
    ))
    assert report["evaluation_scope"] == "source_only_no_gold"
    assert report["scored"] is False
    assert report["pair_count"] == 1


def test_native_partition_concurrency_merges_receipts_without_gold_leak(tmp_path: Path):
    public, gold = _tiny_public_and_gold()
    output = tmp_path / "parallel.jsonl"
    client = FakeRelationClient()
    report = __import__("asyncio").run(run_relation_evaluation(
        split="development",
        prompt_version="relation_v1",
        output=output,
        client=client,
        records=public,
        gold_rows=gold,
        batch_size=1,
        concurrency=2,
    ))
    assert report["concurrency"] == 2
    assert report["pair_count"] == 3
    assert report["missing_pair_count"] == 0
    assert len(output.read_text(encoding="utf-8").splitlines()) == 3
    assert len(report["partition_receipts"]) == 2
    assert all("gold" not in json.dumps(payload, ensure_ascii=False) for _task, payload in client.calls)


def test_partition_concurrency_can_resume_combined_serial_receipts(tmp_path: Path):
    public, gold = _tiny_public_and_gold()
    output = tmp_path / "resume-parallel.jsonl"
    first = FakeRelationClient()
    __import__("asyncio").run(run_relation_evaluation(
        split="development",
        prompt_version="relation_v1",
        output=output,
        client=first,
        records=public,
        gold_rows=gold,
        batch_size=2,
    ))

    # A previous parallel attempt may have left a partial partition behind;
    # a later combined serial run must seed newly added main receipts into it.
    part0 = output.with_name(output.name + ".part0")
    part0.write_text(output.read_text(encoding="utf-8").splitlines()[0] + "\n", encoding="utf-8")

    second = FakeRelationClient()
    report = __import__("asyncio").run(run_relation_evaluation(
        split="development",
        prompt_version="relation_v1",
        output=output,
        client=second,
        records=public,
        gold_rows=gold,
        batch_size=1,
        concurrency=2,
    ))

    assert report["concurrency"] == 2
    assert report["resumed_pair_count"] == 3
    assert report["new_pair_count"] == 0
    assert second.calls == []
    assert len(output.read_text(encoding="utf-8").splitlines()) == 3

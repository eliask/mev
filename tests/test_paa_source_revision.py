"""Bounded source-revision validation and mocked producer path (no live inference)."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from paa.source_question import (
    MUSE_ANSWER_MAX_TOKENS,
    RESERVED_SYSTEM_SCHEMA_TOKENS,
    SourceQuestionBank,
    _context_budget_tokens,
)
from paa.source_revision import (
    SourceRevisionError,
    _strip_private_paths,
    revise_candidate,
    validate_critique_binding,
)

FIXTURE = Path("paa/contracts/fixtures/source_revision_bank")


def _load_bank() -> SourceQuestionBank:
    return SourceQuestionBank.load(FIXTURE / "questions.json", FIXTURE / "sources.json")


def _critique() -> dict:
    return json.loads((FIXTURE / "critique.json").read_text(encoding="utf-8"))


def test_critique_binding_accepts_exact_anchors() -> None:
    bank = _load_bank()
    candidate_bytes = (FIXTURE / "candidate.json").read_bytes()
    candidate = json.loads(candidate_bytes.decode())
    critique = _critique()
    binding = validate_critique_binding(
        candidate=candidate,
        candidate_bytes=candidate_bytes,
        critique=critique,
        bank=bank,
    )
    assert binding["task_id"] == "REV-D1"
    assert binding["candidate_sha256"] == critique["candidate"]["sha256"]
    assert binding["task_source_ids"] == ["HE1"]
    assert binding["reviewer"]["human_certification"] is True


def test_stale_source_hash_and_invented_quote_rejected() -> None:
    bank = _load_bank()
    candidate_bytes = (FIXTURE / "candidate.json").read_bytes()
    candidate = json.loads(candidate_bytes.decode())
    critique = json.loads((FIXTURE / "critique_stale_source.json").read_text(encoding="utf-8"))
    with pytest.raises(SourceRevisionError, match="mismatch|does not match"):
        validate_critique_binding(
            candidate=candidate,
            candidate_bytes=candidate_bytes,
            critique=critique,
            bank=bank,
        )


def test_bad_candidate_anchor_rejected() -> None:
    bank = _load_bank()
    candidate_bytes = (FIXTURE / "candidate.json").read_bytes()
    candidate = json.loads(candidate_bytes.decode())
    critique = json.loads((FIXTURE / "critique_bad_candidate_anchor.json").read_text(encoding="utf-8"))
    with pytest.raises(SourceRevisionError, match="excerpt mismatch"):
        validate_critique_binding(
            candidate=candidate,
            candidate_bytes=candidate_bytes,
            critique=critique,
            bank=bank,
        )


def test_missing_required_bindings_rejected_before_provider() -> None:
    bank = _load_bank()
    candidate_bytes = (FIXTURE / "candidate.json").read_bytes()
    candidate = json.loads(candidate_bytes.decode())
    critique = _critique()
    del critique["reviewer"]
    critique["issues"] = []
    critique["input_contract"]["task_source_ids"] = []
    with pytest.raises(SourceRevisionError, match="reviewer|issues|task_source_ids"):
        validate_critique_binding(
            candidate=candidate,
            candidate_bytes=candidate_bytes,
            critique=critique,
            bank=bank,
        )


def test_evidence_outside_task_universe_rejected() -> None:
    bank = _load_bank()
    candidate_bytes = (FIXTURE / "candidate.json").read_bytes()
    candidate = json.loads(candidate_bytes.decode())
    critique = _critique()
    critique["issues"][0]["source_evidence"][0]["source_id"] = "OTHER"
    with pytest.raises(SourceRevisionError, match="not in task universe"):
        validate_critique_binding(
            candidate=candidate,
            candidate_bytes=candidate_bytes,
            critique=critique,
            bank=bank,
        )


def test_private_paths_stripped_keeps_source_evidence() -> None:
    critique = _critique()
    public = _strip_private_paths(critique)
    dumped = json.dumps(public)
    assert "raw_path" not in dumped
    assert "/tmp/private" not in dumped
    evidence = public["issues"][0]["source_evidence"][0]
    assert evidence["source_id"] == "HE1"
    assert evidence["quote"].startswith("Valiokunta")
    assert evidence["start"] == 107
    assert evidence["text_sha256"]


@pytest.mark.asyncio
async def test_revise_candidate_mocked_provider_binds_and_preserves(tmp_path: Path, monkeypatch) -> None:
    bank = _load_bank()
    out = tmp_path / "revision_out"
    client = SimpleNamespace(manifest=None, discover_calls=0)

    async def discover() -> dict:
        client.discover_calls += 1
        client.manifest = {
            "context_length": 160_000,
            "provider": "TEST",
            "model_id": "test-model",
            "context_length_basis": "EXPLICIT_CALLER_PLANNING_LIMIT",
        }
        return client.manifest

    client.discover = discover

    captured: dict = {}

    async def fake_provider(client_arg, task, system, user, *, schema, max_tokens, reasoning_effort=None, allow_syntax_retry=True):
        assert allow_syntax_retry is False
        captured["budget"] = _context_budget_tokens(client_arg, max_tokens=max_tokens)
        captured["max_tokens"] = max_tokens
        captured["user_has_critique"] = "SOURCE-BACKED CRITIQUE" in user
        captured["user_has_raw_path"] = "/tmp/private" in user
        return {
            "status": "OK",
            "parsed": {
                "answer": "Lähde ei tue vain-varoitus -rajausta.",
                "claims": [
                    {
                        "text": "Valiokunta ei sano, että aloite koskisi vain varoitusta.",
                        "epistemic_status": "SOURCE_SUPPORTED",
                        "evidence": [
                            {"source_id": "HE1", "first_pointer": 2, "last_pointer": 2},
                        ],
                    }
                ],
                "unresolved": [],
                "next_evidence": [],
            },
            "request_id": "req-test-1",
            "cache_hit": False,
            "elapsed_seconds": 0.01,
            "request_attempts": [{"attempt": 1, "status": "OK"}],
            "request_attempt_count": 1,
            "model": client.manifest,
            "provider": "TEST",
            "model_id": "test-model",
        }

    monkeypatch.setattr("paa.source_revision._provider_request", fake_provider)
    packet = await revise_candidate(
        candidate_path=FIXTURE / "candidate.json",
        critique_path=FIXTURE / "critique.json",
        questions=FIXTURE / "questions.json",
        sources=FIXTURE / "sources.json",
        client=client,
        output_dir=out,
        bank=bank,
    )
    assert client.discover_calls == 1
    expected_budget = 160_000 - MUSE_ANSWER_MAX_TOKENS - RESERVED_SYSTEM_SCHEMA_TOKENS
    assert captured["budget"] == expected_budget
    assert expected_budget > 49_152  # not the default context ceiling
    assert captured["user_has_critique"] is True
    assert captured["user_has_raw_path"] is False
    assert packet["status"] == "OK"
    assert packet["binding_ok"] is True
    assert packet["reading_mode"] == "DIRECT_WHOLE_BUNDLE"
    assert packet["admission"] == "CANDIDATE_NOT_REFERENCE_JUDGED"
    assert packet["revision"]["automatic_repeat"] is False
    assert packet["revision"]["budget_tokens"] == expected_budget
    assert packet["claims"]
    assert packet["claims"][0]["evidence_bound"] or packet["claims"][0].get("evidence")
    preserved = out / packet["preserved_predecessor_file"]
    assert preserved.is_file()
    assert preserved.read_bytes() == (FIXTURE / "candidate.json").read_bytes()
    principal = out / f"{packet['task_id']}.revised.json"
    assert principal.is_file()
    assert principal.name != preserved.name
    receipt = json.loads((out / "REVISION_RECEIPT.json").read_text(encoding="utf-8"))
    assert receipt["principal_candidate_artifact"] == principal.name
    assert receipt["preserved_predecessor_file"] == preserved.name


@pytest.mark.asyncio
async def test_revise_candidate_preserves_truncated_status(tmp_path: Path, monkeypatch) -> None:
    bank = _load_bank()
    out = tmp_path / "trunc_out"
    client = SimpleNamespace(
        manifest={
            "context_length": 160_000,
            "provider": "TEST",
            "model_id": "test-model",
        }
    )

    async def fake_provider(*_args, **_kwargs):
        return {
            "status": "TRUNCATED",
            "error": "output token budget exhausted",
            "failure_class": "OUTPUT_BUDGET_OR_CONFIGURATION",
            "request_id": "req-trunc",
            "cache_hit": False,
            "elapsed_seconds": 0.02,
            "request_attempts": [{"attempt": 1, "status": "TRUNCATED"}],
            "request_attempt_count": 1,
            "model": client.manifest,
            "provider": "TEST",
            "model_id": "test-model",
        }

    monkeypatch.setattr("paa.source_revision._provider_request", fake_provider)
    packet = await revise_candidate(
        candidate_path=FIXTURE / "candidate.json",
        critique_path=FIXTURE / "critique.json",
        questions=FIXTURE / "questions.json",
        sources=FIXTURE / "sources.json",
        client=client,
        output_dir=out,
        bank=bank,
    )
    assert packet["status"] == "TRUNCATED"
    assert packet["failure_class"] == "OUTPUT_BUDGET_OR_CONFIGURATION"
    assert packet["binding_ok"] is False
    assert packet["read_completeness"] == "FAILED"
    assert packet["coverage"]["fully_read_source_ids"] == []
    assert packet["reading_mode"] == "DIRECT_WHOLE_BUNDLE"
    assert (out / packet["preserved_predecessor_file"]).is_file()
    written = json.loads((out / "REV-D1.revised.json").read_text(encoding="utf-8"))
    assert written["status"] == "TRUNCATED"
    assert written["error"] == "output token budget exhausted"


@pytest.mark.asyncio
async def test_stale_candidate_hash_fails_before_provider(tmp_path: Path, monkeypatch) -> None:
    bank = _load_bank()
    called = {"n": 0}

    async def fake_provider(*_args, **_kwargs):
        called["n"] += 1
        raise AssertionError("provider must not run")

    monkeypatch.setattr("paa.source_revision._provider_request", fake_provider)
    bad_candidate = tmp_path / "bad_candidate.json"
    payload = json.loads((FIXTURE / "candidate.json").read_text(encoding="utf-8"))
    payload["answer"] = "tampered"
    bad_candidate.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    client = SimpleNamespace(manifest={"context_length": 160_000})
    with pytest.raises(SourceRevisionError, match="candidate byte hash mismatch"):
        await revise_candidate(
            candidate_path=bad_candidate,
            critique_path=FIXTURE / "critique.json",
            questions=FIXTURE / "questions.json",
            sources=FIXTURE / "sources.json",
            client=client,
            output_dir=tmp_path / "out",
            bank=bank,
        )
    assert called["n"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_pointer", [999999, None])
async def test_revision_does_not_promote_invalid_or_empty_citations(tmp_path, monkeypatch, bad_pointer):
    async def provider(*args, **kwargs):
        assert kwargs["allow_syntax_retry"] is False
        claims = [] if bad_pointer is None else [{
            "text": "Unsupported", "epistemic_status": "open",
            "evidence": [{"source_id": "HE1", "first_pointer": bad_pointer, "last_pointer": bad_pointer}],
        }]
        return {"status": "OK", "parsed": {
            "answer": "candidate", "claims": claims, "unresolved": [], "next_evidence": [],
        }}
    monkeypatch.setattr("paa.source_revision._provider_request", provider)
    packet = await revise_candidate(
        candidate_path=FIXTURE / "candidate.json", critique_path=FIXTURE / "critique.json",
        questions=FIXTURE / "questions.json", sources=FIXTURE / "sources.json",
        client=SimpleNamespace(manifest={"context_length": 160_000}), output_dir=tmp_path,
    )
    assert packet["status"] == "INVALID_OUTPUT"
    assert packet["binding_ok"] is False
    receipt = json.loads((tmp_path / "REVISION_RECEIPT.json").read_text())
    assert receipt["input_binding_ok"] is True
    assert receipt["binding_ok"] is False


@pytest.mark.asyncio
async def test_interrupted_revision_keeps_predecessor_and_pending_lineage(tmp_path, monkeypatch):
    async def provider(*args, **kwargs):
        raise RuntimeError("Internal defect must propagate")
    monkeypatch.setattr("paa.source_revision._provider_request", provider)
    with pytest.raises(RuntimeError, match="Internal defect"):
        await revise_candidate(
            candidate_path=FIXTURE / "candidate.json", critique_path=FIXTURE / "critique.json",
            questions=FIXTURE / "questions.json", sources=FIXTURE / "sources.json",
            client=SimpleNamespace(manifest={"context_length": 160_000}), output_dir=tmp_path,
        )
    receipt = json.loads((tmp_path / "REVISION_RECEIPT.json").read_text())
    assert receipt["status"] == "REQUEST_PENDING"
    assert receipt["binding_ok"] is False
    assert (tmp_path / receipt["preserved_predecessor_file"]).read_bytes() == (FIXTURE / "candidate.json").read_bytes()


@pytest.mark.asyncio
async def test_single_request_policy_disables_syntax_retry():
    from paa.source_question import ANSWER_SCHEMA, _muse_validate_or_syntax_retry
    async def unexpected_request(*args, **kwargs):
        raise AssertionError("A bounded revision must not silently submit a second request")
    result = await _muse_validate_or_syntax_retry(
        SimpleNamespace(request=unexpected_request), task="revision", system_with_schema="schema",
        user="source", schema=ANSWER_SCHEMA, max_tokens=8192, reasoning_effort="medium",
        first_response={"status": "OK", "content": "{invalid", "request_id": "first"},
        allow_syntax_retry=False,
    )
    assert result["status"] == "INVALID_OUTPUT"
    assert result["request_attempt_count"] == 1
    assert result["request_attempts"][0]["request_id"] == "first"

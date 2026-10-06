from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from paa.llm_semantics import ACTION_KINDS, CAPABILITIES, ISSUER_SCOPES, SEMANTIC_TYPES

ROOT = Path(__file__).parents[1]
FIXTURE = ROOT / "paa/contracts/fixtures/semantic_blind_heldout_20261007.jsonl"
EXCLUSIONS = (
    ROOT / "paa/contracts/fixtures/llm_semantic_gold.jsonl",
    ROOT / "paa/contracts/fixtures/llm_relation_gold.jsonl",
)


def _ids(path: Path) -> set[str]:
    result: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        document = item.get("document") or item.get("source") or {}
        result.add(document["document_id"])
    return result


def test_source_only_heldout_fixture_is_anchored_and_disjoint() -> None:
    rows = [json.loads(line) for line in FIXTURE.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 80
    document_ids = [row["document"]["document_id"] for row in rows]
    assert len(set(document_ids)) == 80
    assert not set(document_ids).intersection(*(_ids(path) for path in EXCLUSIONS))
    assert not set(document_ids).intersection(set().union(*(_ids(path) for path in EXCLUSIONS)))

    assert Counter(row["selection"]["stratum"] for row in rows) == Counter(
        {
            "collective": 12,
            "explicit_formal_act": 10,
            "negative_or_process": 14,
            "ordinary_random": 14,
            "policy_goal": 16,
            "strong_outcome": 14,
        }
    )
    for row in rows:
        document = row["document"]
        text = document["source_text"]
        assert hashlib.sha256(text.encode()).hexdigest() == document["source_sha256"]
        assert row["selection"]["seed"] == 20261007
        assert row["split"] == "heldout_blind_source_only"
        assert row["adjudication"]["reviewer"] == "CODEX_SOURCE_ONLY_2026-10-07"
        assert row["adjudication"]["method"] == "AI_SOURCE_READING"
        assert "LLM output" in row["adjudication"]["basis"]
        assert "human annotation" in row["adjudication"]["independence_caveat"]

        gold = row["gold"]
        assert gold["primary_type"] in SEMANTIC_TYPES
        assert len(gold["propositions"]) >= 1
        for proposition in gold["propositions"]:
            assert proposition["source_quote"] in text
            for key in ("target_quote", "condition_quote", "deadline_quote"):
                value = proposition[key]
                assert value is None or value in text
            assert proposition["semantic_type"] in SEMANTIC_TYPES
            assert proposition["issuer_scope"] in ISSUER_SCOPES
            assert proposition["action_kind"] is None or proposition["action_kind"] in ACTION_KINDS
            assert proposition["required_capability"] is None or proposition["required_capability"] in CAPABILITIES
            assert proposition["reported_speech"] is False

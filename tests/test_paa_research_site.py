"""Tests for source-linked research-only inquiry packets."""

import asyncio
import hashlib
import json
import re
from pathlib import Path

from paa.llm_inquiries import infer_run, prepare_aggregate_run, prepare_run
from paa.research_site import _source_anchor, build_research_packet, render_research_html


def _source(source_id: str, text: str) -> dict:
    return {
        "source_id": source_id,
        "source_table": "test",
        "record_id": source_id,
        "title": "Test source",
        "source_url": "https://example.test/" + source_id,
        "source_kind": "TEST",
        "document_identifier": source_id,
        "text": text,
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
    }


def _case() -> dict:
    return {
        "episode_id": "mev-research-episode",
        "question_contract": {
            "question_id": "q-research",
            "text": "What documentary response is recorded?",
            "target_scope": "the safeguard",
            "period": "2024",
            "comparison": "proposal and committee",
            "evidence_needed": ["exact source quote"],
            "valid_outputs": ["documentary response"],
            "unknowns": ["later implementation"],
        },
        "sources": [
            _source("source-proposal", "The proposal describes a safeguard."),
            _source("source-committee", "The committee records the safeguard."),
        ],
    }


def _write_fixture(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def test_packet_deduplicates_episode_and_rebinds_original_source_anchor(tmp_path: Path, monkeypatch) -> None:
    source_fixture = tmp_path / "sources.jsonl"
    _write_fixture(source_fixture, [_case()])
    broad = tmp_path / "broad"
    prepare_run(source_fixture, broad, clip_limit=200, window_limit=500, coverage_mode="EXHAUSTIVE")

    class FakeClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        async def discover(self):
            return {"model_id": "fake", "model_digest": "fake"}

        async def request(self, task, system, user, *, schema, max_tokens):
            return {
                "status": "OK",
                "request_id": "request-research",
                "parsed": {
                    "schema_version": schema["properties"]["schema_version"]["const"],
                    "episode_id": schema["properties"]["episode_id"]["const"],
                    "window_id": schema["properties"]["window_id"]["const"],
                    "answer": "The record documents a safeguard.",
                    "claims": [{
                        "type": "DOCUMENTARY_ANSWER",
                        "text": "The committee records a safeguard.",
                        "state": "PROPOSED",
                        "evidence": [{"source_id": "source-committee", "quote": "The committee records the safeguard."}],
                    }],
                    "unknowns": [],
                    "unsupported": [],
                },
            }

        async def close(self):
            return None

    monkeypatch.setattr("paa.llm_inquiries.LocalLLMClient", FakeClient)
    asyncio.run(infer_run(broad))
    aggregate = tmp_path / "aggregate"
    prepare_aggregate_run(broad, source_fixture, aggregate)
    asyncio.run(infer_run(aggregate))
    packet = build_research_packet(aggregate, source_fixture, label="test packet")
    assert packet["case_count"] == 1
    episode = packet["episodes"][0]
    assert episode["question"]["text"] == _case()["question_contract"]["text"]
    review_queue = episode["source_review_queue"]
    assert review_queue
    assert all(item["review_state"] == "PENDING_SOURCE_REVIEW" for item in review_queue)
    assert all(item["semantic_disposition"] == "NOT_SOURCE_REVIEWED" for item in review_queue)
    assert episode["paired_input"]["same_input_sha256"] is True
    for mode in ("baseline", "structured"):
        claim = episode["modes"][mode]["claims"][0]
        assert claim["state"] == "PROPOSED"
        assert claim["semantic_disposition"] == "NOT_SOURCE_REVIEWED"
        assert claim["claim_id"].startswith("claim-")
        assert claim["source_anchor_ids"][0].startswith("anchor-")
        anchor = claim["evidence"][0]
        assert anchor["anchor_state"] == "EXACT_COMPLETE_SOURCE"
        assert anchor["char_start"] == _case()["sources"][1]["text"].find(anchor["quote"])
        assert anchor["quote"] in anchor["context_excerpt"]
    rendered = render_research_html(packet)
    assert "test packet" in rendered
    assert "PROPOSED_RESEARCH_NOT_ADMITTED" not in rendered
    assert "Ehdotetut väitteet" in rendered
    assert "Perusvertailu" in rendered
    assert "The record documents a safeguard." in rendered
    assert 'id="baseline-evidence-' in rendered
    assert 'id="structured-evidence-' in rendered
    dom_ids = re.findall(r'\sid="([^"]+)"', rendered)
    assert len(dom_ids) == len(set(dom_ids))
    assert "Alkuperäinen lähdekonteksti" in rendered
    assert "data-source-anchor-id=" in rendered
    assert "viewport" in rendered
    warning_packet = json.loads(json.dumps(packet))
    warning_packet["episodes"][0]["modes"]["baseline"]["warnings"] = [{
        "code": "STRING_LIMIT_REACHED",
        "message": "Answer reached the schema length bound and may be incomplete.",
    }]
    warning_packet["episodes"][0]["modes"]["baseline"]["answer_status"] = (
        "Muoto ja täsmäankkurit validoitu; vastaus saavutti merkkirajan ja voi olla keskeneräinen"
    )
    warning_html = render_research_html(warning_packet)
    assert "vastaus saavutti merkkirajan" in warning_html
    assert "Answer reached the schema length bound" in warning_html
    unsafe = json.loads(json.dumps(packet))
    unsafe["episodes"][0]["source_coverage"][0]["source_url"] = "javascript:alert(1)"
    unsafe["episodes"][0]["modes"]["baseline"]["claims"][0]["evidence"][0]["source_url"] = "javascript:alert(1)"
    assert "javascript:" not in render_research_html(unsafe)


def test_repeated_quote_is_not_bound_to_first_source_occurrence() -> None:
    source = _source("source-repeated", "same phrase; context one. same phrase; context two.")

    anchor = _source_anchor({"source_id": source["source_id"], "quote": "same phrase"}, {source["source_id"]: source})

    assert anchor["anchor_state"] == "UNRESOLVED_SOURCE_BINDING"
    assert "QUOTE_OCCURS_MULTIPLE_TIMES_NO_MODEL_OFFSET" in anchor["validation_errors"]
    assert anchor["char_start"] is None
    assert anchor["candidate_char_starts"] == [0, 26]

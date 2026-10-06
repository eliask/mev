"""Tests for source-windowed baseline/structured inquiry comparison."""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

import pytest

from paa.llm_inquiries import (
    AGGREGATE_OUTPUT_SCHEMA_VERSION,
    AGGREGATE_OUTPUT_SCHEMA_VERSION_V6,
    AGGREGATE_OUTPUT_SCHEMA_VERSION_V7,
    AGGREGATE_OUTPUT_SCHEMA_VERSION_V8,
    CLIP_BOUNDARY_VERSION,
    MIN_AGGREGATE_SOURCE_CONTEXT_CHARS,
    OUTPUT_SCHEMA_VERSION,
    SCHEMA_VERSION,
    InquiryModelError,
    _aggregate_source,
    _compact_model_payload,
    _read_jsonl,
    _source_fixture_map,
    _source_payload,
    build_request,
    evaluate_run,
    infer_run,
    load_prepared_run,
    normalize_inquiry,
    prepare_aggregate_run,
    prepare_run,
)


def _source(source_id: str, text: str) -> dict:
    return {
        "source_id": source_id,
        "source_table": "test",
        "record_id": source_id,
        "title": "Test source",
        "source_url": "https://example.test/" + source_id,
        "text": text,
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "raw_sha256": hashlib.sha256(("raw:" + text).encode()).hexdigest(),
    }


def _case() -> dict:
    return {
        "episode_id": "mev-test-episode",
        "episode_kind": "TEST",
        "question_contract": {
            "question_id": "q-test",
            "text": "What documentary response is recorded?",
            "target_scope": "the safeguard",
            "period": "2024",
            "comparison": "proposal and committee",
            "evidence_needed": ["exact source quote"],
            "valid_outputs": ["documentary response", "unresolved implementation"],
            "unknowns": ["later implementation"],
            "practical_use": "separate record from effect",
        },
        "sources": [
            _source("source-proposal", "Proposal requires a safeguard. " + "background " * 500),
            _source("source-committee", "The committee records the safeguard."),
        ],
        "transformations": [],
    }


def _write_fixture(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def test_jsonl_preserves_unicode_line_separators_inside_source_text(tmp_path: Path) -> None:
    source_fixture = tmp_path / "sources.jsonl"
    special_text = "Before\u2028between\u2029after"
    _write_fixture(source_fixture, [{**_case(), "sources": [_source("source-special", special_text)]}])

    rows = _read_jsonl(source_fixture)

    assert rows[0]["sources"][0]["text"] == special_text


def test_source_fixture_map_rejects_conflicting_duplicate_source_ids(tmp_path: Path) -> None:
    source_fixture = tmp_path / "sources.jsonl"
    _write_fixture(
        source_fixture,
        [
            {**_case(), "episode_id": "episode-one", "sources": [_source("source-duplicate", "first")]},
            {**_case(), "episode_id": "episode-two", "sources": [_source("source-duplicate", "second")]},
        ],
    )

    with pytest.raises(InquiryModelError, match="conflicting"):
        _source_fixture_map(source_fixture)


def test_prepare_clips_sources_and_keeps_review_labels_out_of_model_payload(tmp_path: Path) -> None:
    source_fixture = tmp_path / "sources.jsonl"
    reviews_fixture = tmp_path / "reviews.jsonl"
    _write_fixture(source_fixture, [_case()])
    _write_fixture(
        reviews_fixture,
        [{"review_id": f"r-{i}", "warning_control": control} for i, control in enumerate(("REAL_REPAIR", "REASONED_REBUTTAL", "APPARENT_GAP", "FALSE_GAP"))],
    )
    run = tmp_path / "run"
    manifest = prepare_run(source_fixture, run, reviews_fixture=reviews_fixture, clip_limit=300, window_limit=1_000, coverage_mode="FOCUSED")
    loaded, windows = load_prepared_run(run)
    assert manifest == loaded
    assert manifest["case_count"] == 1
    assert manifest["review_control_count"] == 4
    assert manifest["gold_or_review_labels_in_model_input"] is False
    assert manifest["prompt_contract"]["modes"]["baseline"]["system"]
    assert manifest["prompt_contract_sha256"]
    assert manifest["request_config"]["chat_template_kwargs"]["enable_thinking"] is False
    assert windows[0]["payload"]["coverage"]["full_source_text_in_model_input"] is False
    assert any(source["coverage_state"] == "CLIPPED" for source in windows[0]["payload"]["sources"])
    serialized = json.dumps(windows[0]["payload"])
    assert "warning_control" not in serialized
    assert "gold" not in serialized


def test_exhaustive_mode_covers_source_text_with_overlapped_chunks(tmp_path: Path) -> None:
    source_fixture = tmp_path / "sources.jsonl"
    _write_fixture(source_fixture, [_case()])
    run = tmp_path / "run"
    manifest = prepare_run(source_fixture, run, clip_limit=300, window_limit=500, coverage_mode="EXHAUSTIVE")
    _, windows = load_prepared_run(run)
    assert manifest["coverage_mode"] == "EXHAUSTIVE"
    source_text = {source["source_id"]: source["text"] for source in _case()["sources"]}
    intervals: dict[str, list[tuple[int, int]]] = {source_id: [] for source_id in source_text}
    for window in windows:
        assert window["payload"]["coverage"]["coverage_mode"] == "EXHAUSTIVE"
        for source in window["payload"]["sources"]:
            intervals[source["source_id"]].extend((clip["start"], clip["end"]) for clip in source["clips"])
            for clip in source["clips"]:
                assert clip["boundary_version"] == CLIP_BOUNDARY_VERSION
                if clip["end"] < len(source_text[source["source_id"]]):
                    assert source_text[source["source_id"]][clip["end"] - 1].isspace()
    for source_id, text in source_text.items():
        covered = [False] * len(text)
        for start, end in intervals[source_id]:
            for index in range(start, end):
                covered[index] = True
        assert all(covered)


def test_baseline_and_structured_use_identical_source_input(tmp_path: Path) -> None:
    source_fixture = tmp_path / "sources.jsonl"
    _write_fixture(source_fixture, [_case()])
    run = tmp_path / "run"
    prepare_run(source_fixture, run, clip_limit=300, window_limit=1_000, coverage_mode="FOCUSED")
    _, windows = load_prepared_run(run)
    baseline = build_request(windows[0], "baseline")
    structured = build_request(windows[0], "structured")
    assert baseline["user"] == structured["user"]
    assert baseline["input_sha256"] == structured["input_sha256"]
    assert baseline["source_payload_sha256"] == structured["source_payload_sha256"]
    assert baseline["schema_sha256"] == structured["schema_sha256"]
    assert baseline["prompt_sha256"] != structured["prompt_sha256"]


def test_legacy_prepared_projection_omits_absent_optional_structure(tmp_path: Path) -> None:
    source_fixture = tmp_path / "sources.jsonl"
    _write_fixture(source_fixture, [_case()])
    run = tmp_path / "run"
    prepare_run(source_fixture, run, clip_limit=300, window_limit=1_000, coverage_mode="FOCUSED")
    _, windows = load_prepared_run(run)

    absent = json.loads(json.dumps(windows[0]))
    for source in absent["payload"]["sources"]:
        source.pop("source_structure", None)
    compact_absent = _compact_model_payload(absent)
    assert all("source_structure" not in source for source in compact_absent["sources"])

    explicit_none = json.loads(json.dumps(absent))
    for source in explicit_none["payload"]["sources"]:
        source["source_structure"] = None
    assert _compact_model_payload(explicit_none) == compact_absent

    prepared_source = windows[0]["payload"]["sources"][0]
    legacy_projection = _source_payload(
        [{key: value for key, value in prepared_source.items() if key != "source_structure"}]
    )
    assert "source_structure" not in legacy_projection[0]


def test_compact_request_is_versioned_and_keeps_exact_clip_anchors(tmp_path: Path) -> None:
    source_fixture = tmp_path / "sources.jsonl"
    _write_fixture(source_fixture, [_case()])
    run = tmp_path / "run"
    prepare_run(source_fixture, run, clip_limit=300, window_limit=1_000, coverage_mode="FOCUSED")
    _, windows = load_prepared_run(run)
    request = build_request(windows[0], "structured")
    assert request["prompt_version"] == "inquiry_structured_compact_v4"
    assert request["output_schema_version"] == OUTPUT_SCHEMA_VERSION
    assert request["schema"]["properties"]["schema_version"]["const"] == OUTPUT_SCHEMA_VERSION
    assert request["schema"]["properties"]["claims"]["maxItems"] == 2
    assert request["schema"]["properties"]["claims"]["items"]["properties"]["evidence"]["maxItems"] == 1
    assert all("start" in clip and "end" in clip and "text" in clip
               for source in request["model_payload"]["sources"] for clip in source["clips"])
    assert "source_url" not in request["model_payload"]["sources"][0]
    assert len(request["user"]) < len(json.dumps(windows[0]["payload"], ensure_ascii=False, sort_keys=True))


def test_infer_rejects_more_than_three_local_streams(tmp_path: Path) -> None:
    source_fixture = tmp_path / "sources.jsonl"
    _write_fixture(source_fixture, [_case()])
    run = tmp_path / "run"
    prepare_run(source_fixture, run, clip_limit=300, window_limit=1_000, coverage_mode="FOCUSED")
    try:
        asyncio.run(infer_run(run, concurrency=4))
    except InquiryModelError as error:
        assert "concurrency" in str(error)
    else:
        raise AssertionError("concurrency > 3 must be rejected before model discovery")


def test_normalize_requires_exact_window_quote_and_keeps_claim_proposed(tmp_path: Path) -> None:
    source_fixture = tmp_path / "sources.jsonl"
    _write_fixture(source_fixture, [_case()])
    run = tmp_path / "run"
    prepare_run(source_fixture, run, clip_limit=300, window_limit=1_000, coverage_mode="FOCUSED")
    _, windows = load_prepared_run(run)
    window = windows[0]
    quote = "The committee records the safeguard."
    valid = {
        "schema_version": SCHEMA_VERSION,
        "episode_id": window["episode_id"],
        "window_id": window["window_id"],
        "proposed_answer": "The committee records a safeguard in the supplied text.",
        "claims": [{
            "claim_type": "DOCUMENTARY_ANSWER",
            "text": "The committee records a safeguard.",
            "state": "PROPOSED",
            "evidence": [{"source_id": "source-committee", "quote": quote}],
        }],
        "localized_unknowns": [{
            "text": "Implementation is not in the packet.",
            "missing_evidence": "An implementation record.",
            "next_observation": "Check the later implementation record.",
            "source_ids": ["source-committee"],
            "state": "UNRESOLVED",
        }],
        "unsupported_claims": [],
    }
    normalized = normalize_inquiry(valid, window, "structured")
    assert normalized["status"] == "VALID"
    assert normalized["claims"][0]["state"] == "PROPOSED"
    assert normalized["claims"][0]["evidence"][0]["quote_sha256"] == hashlib.sha256(quote.encode()).hexdigest()

    bounded = json.loads(json.dumps(valid))
    bounded["answer"] = "x" * 420
    bounded_result = normalize_inquiry(
        bounded,
        window,
        "structured",
        expected_schema_version=AGGREGATE_OUTPUT_SCHEMA_VERSION,
    )
    assert bounded_result["status"] == "VALID"
    assert bounded_result["warnings"] == [{
        "code": "STRING_LIMIT_REACHED",
        "field": "answer",
        "max_length": 420,
        "message": "Answer reached the schema length bound and may be incomplete.",
    }]
    v6_bounded = json.loads(json.dumps(valid))
    v6_bounded["schema_version"] = AGGREGATE_OUTPUT_SCHEMA_VERSION_V6
    v6_bounded["answer"] = "x" * 1_200
    v6_result = normalize_inquiry(
        v6_bounded,
        window,
        "structured",
        expected_schema_version=AGGREGATE_OUTPUT_SCHEMA_VERSION_V6,
    )
    assert v6_result["status"] == "VALID"
    assert v6_result["normalization_version"] == "inquiry_normalization_aggregate_v6"
    assert v6_result["warnings"][0]["max_length"] == 1_200
    assert bounded_result["normalization_version"] == "inquiry_normalization_v2"
    v7_bounded = json.loads(json.dumps(valid))
    v7_bounded["schema_version"] = AGGREGATE_OUTPUT_SCHEMA_VERSION_V7
    v7_bounded["answer"] = "x" * 1_200
    v7_result = normalize_inquiry(
        v7_bounded,
        window,
        "structured",
        expected_schema_version=AGGREGATE_OUTPUT_SCHEMA_VERSION_V7,
    )
    assert v7_result["status"] == "VALID"
    assert v7_result["normalization_version"] == "inquiry_normalization_aggregate_v7b"
    assert v7_result["warnings"][0]["max_length"] == 1_200

    v7_bound_claim = json.loads(json.dumps(valid))
    v7_bound_claim["schema_version"] = AGGREGATE_OUTPUT_SCHEMA_VERSION_V7
    v7_bound_claim["claims"][0]["text"] = "x" * 260
    v7_bound_claim_result = normalize_inquiry(
        v7_bound_claim,
        window,
        "structured",
        expected_schema_version=AGGREGATE_OUTPUT_SCHEMA_VERSION_V7,
    )
    assert v7_bound_claim_result["status"] == "PARTIAL"
    assert v7_bound_claim_result["claims"] == []
    assert v7_bound_claim_result["withheld_count"] == 1
    assert any(
        warning["code"] == "STRING_LIMIT_REACHED_LOAD_BEARING"
        and warning["field"] == "claims[0].text"
        for warning in v7_bound_claim_result["warnings"]
    )

    invalid = json.loads(json.dumps(valid))
    invalid["claims"][0]["evidence"][0]["quote"] = "A quote not in the source clip."
    rejected = normalize_inquiry(invalid, window, "structured")
    # The malformed claim is discarded, but the independently valid
    # localized unknown remains usable; this is a partial, source-abstaining
    # result rather than an admission of the bad claim.
    assert rejected["status"] == "PARTIAL"
    assert rejected["claims"] == []
    assert any(error["code"] == "EVIDENCE_QUOTE_NOT_EXACT" for error in rejected["errors"])


def test_infer_is_resumable_for_both_modes_and_eval_reports_proposed_counts(tmp_path: Path, monkeypatch) -> None:
    source_fixture = tmp_path / "sources.jsonl"
    _write_fixture(source_fixture, [_case()])
    run = tmp_path / "run"
    prepare_run(source_fixture, run, clip_limit=300, window_limit=1_000, coverage_mode="FOCUSED")
    calls: list[str] = []

    class FakeClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        async def discover(self):
            return {"model_id": "fake-inquiry", "model_digest": "fake-digest"}

        async def request(self, task, system, user, *, schema, max_tokens):
            calls.append(task)
            episode_id = schema["properties"]["episode_id"]["const"]
            window_id = schema["properties"]["window_id"]["const"]
            return {
                "status": "OK",
                "request_id": "fake-request-" + str(len(calls)),
                "parsed": {
                    "schema_version": SCHEMA_VERSION,
                    "episode_id": episode_id,
                    "window_id": window_id,
                    "proposed_answer": "A source-bound proposal.",
                    "claims": [{
                        "claim_type": "DOCUMENTARY_ANSWER",
                        "text": "The committee records a safeguard.",
                        "state": "PROPOSED",
                        "evidence": [{"source_id": "source-committee", "quote": "The committee records the safeguard."}],
                    }],
                    "localized_unknowns": [],
                    "unsupported_claims": [],
                },
            }

        async def close(self):
            return None

    monkeypatch.setattr("paa.llm_inquiries.LocalLLMClient", FakeClient)
    first = asyncio.run(infer_run(run))
    assert first["counts"]["completed"] == 2
    assert len(calls) == 2
    second = asyncio.run(infer_run(run))
    assert second["counts"]["cache_reused"] == 2
    assert len(calls) == 2
    report = evaluate_run(run)
    assert report["prompt_modes"]["baseline"]["useful_claims"] == 1
    assert report["prompt_modes"]["structured"]["useful_claims"] == 1
    per_case = report["per_case"]["mev-test-episode"]["baseline"]
    assert per_case["window_char_count_total"] > 0
    assert per_case["window_estimated_tokens_total"] > 0
    assert per_case["coverage_state"] == "SOURCE_CLIPS_EXPLICIT"
    assert per_case["source_ids"] == ["source-committee", "source-proposal"]
    assert per_case["unsupported_claims"] == 0
    assert report["model_admission"] == "PROPOSED / NOT_ADMITTED"


def test_infer_reuses_terminal_partial_receipts_without_new_model_calls(tmp_path: Path, monkeypatch) -> None:
    source_fixture = tmp_path / "sources.jsonl"
    _write_fixture(source_fixture, [_case()])
    run = tmp_path / "run"
    prepare_run(source_fixture, run, clip_limit=300, window_limit=1_000, coverage_mode="FOCUSED")
    calls: list[str] = []

    class PartialClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        async def discover(self):
            return {"model_id": "fake-partial", "model_digest": "fake-partial-digest"}

        async def request(self, task, system, user, *, schema, max_tokens):
            calls.append(task)
            return {
                "status": "OK",
                "request_id": f"fake-partial-{len(calls)}",
                "parsed": {
                    "schema_version": schema["properties"]["schema_version"]["const"],
                    "episode_id": schema["properties"]["episode_id"]["const"],
                    "window_id": schema["properties"]["window_id"]["const"],
                    "proposed_answer": "A bounded partial result.",
                    "claims": [{
                        "type": "DOCUMENTARY_ANSWER",
                        "text": "This claim has no exact source quote.",
                        "state": "PROPOSED",
                        "evidence": [{"source_id": "source-committee", "quote": "not in source"}],
                    }],
                    "localized_unknowns": [{
                        "text": "A later record is missing.",
                        "missing_evidence": "A later record.",
                        "next_observation": "Inspect the later record.",
                        "source_ids": ["source-committee"],
                        "state": "UNRESOLVED",
                    }],
                    "unsupported_claims": [],
                },
            }

        async def close(self):
            return None

    monkeypatch.setattr("paa.llm_inquiries.LocalLLMClient", PartialClient)
    first = asyncio.run(infer_run(run))
    assert first["counts"]["semantic_or_source_abstention"] == 2
    assert len(calls) == 2
    second = asyncio.run(infer_run(run))
    assert second["counts"]["cache_reused"] == 2
    assert len(calls) == 2


def test_episode_aggregate_rebinds_broad_candidates_to_complete_source_context(tmp_path: Path, monkeypatch) -> None:
    source_fixture = tmp_path / "sources.jsonl"
    _write_fixture(source_fixture, [_case()])
    broad_run = tmp_path / "broad"
    prepare_run(source_fixture, broad_run, clip_limit=300, window_limit=1_000, coverage_mode="EXHAUSTIVE")

    class FakeClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        async def discover(self):
            return {"model_id": "fake-inquiry", "model_digest": "fake-digest"}

        async def request(self, task, system, user, *, schema, max_tokens):
            return {
                "status": "OK",
                "request_id": "fake-aggregate-source",
                "parsed": {
                    "schema_version": schema["properties"]["schema_version"]["const"],
                    "episode_id": schema["properties"]["episode_id"]["const"],
                    "window_id": schema["properties"]["window_id"]["const"],
                    "answer": "A source-bound proposal.",
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
    result = asyncio.run(infer_run(broad_run))
    assert result["stage"] == "INFERRED"
    aggregate_run = tmp_path / "aggregate"
    manifest = prepare_aggregate_run(broad_run, source_fixture, aggregate_run)
    assert manifest["coverage_mode"] == "EPISODE_AGGREGATED"
    assert manifest["output_schema_version"] == AGGREGATE_OUTPUT_SCHEMA_VERSION_V8
    assert manifest["candidate_quote_count"] == 1
    _, windows = load_prepared_run(aggregate_run)
    payload = windows[0]["payload"]
    assert payload["coverage"]["coverage_state"] == "AGGREGATED_ORIGINAL_CONTEXT"
    assert payload["candidate_hints"][0]["status"] == "PROPOSED_CANDIDATE"
    committee = next(source for source in payload["sources"] if source["source_id"] == "source-committee")
    assert any("The committee records the safeguard." in clip["text"] for clip in committee["clips"])
    baseline = build_request(windows[0], "baseline", prompt_contract=manifest["prompt_contract"])
    structured = build_request(windows[0], "structured", prompt_contract=manifest["prompt_contract"])
    assert baseline["user"] == structured["user"]
    assert "candidate_hints" in baseline["model_payload"]
    assert baseline["schema"]["properties"]["claims"]["items"]["properties"]["evidence"]["maxItems"] == 2
    assert "suomeksi" in baseline["system"]
    aggregate_v6 = tmp_path / "aggregate-v6"
    manifest_v6 = prepare_aggregate_run(
        broad_run,
        source_fixture,
        aggregate_v6,
        aggregate_version="v6",
    )
    assert manifest_v6["aggregate_contract_version"] == "v6"
    assert manifest_v6["output_schema_version"] == AGGREGATE_OUTPUT_SCHEMA_VERSION_V6
    assert manifest_v6["prompt_contract"]["response_schema_template"]["properties"]["answer"]["maxLength"] == 1_200
    assert "420" not in manifest_v6["prompt_contract"]["modes"]["baseline"]["system"]
    aggregate_v7 = tmp_path / "aggregate-v7"
    manifest_v7 = prepare_aggregate_run(
        broad_run,
        source_fixture,
        aggregate_v7,
        aggregate_version="v7",
    )
    assert manifest_v7["aggregate_contract_version"] == "v7"
    assert manifest_v7["output_schema_version"] == AGGREGATE_OUTPUT_SCHEMA_VERSION_V7
    assert manifest_v7["prompt_contract"]["response_schema_template"]["properties"]["answer"]["maxLength"] == 1_200
    assert "hyväksytty tai säädetty laki" in manifest_v7["prompt_contract"]["modes"]["baseline"]["system"]
    assert manifest["prompt_contract"]["response_schema_template"]["properties"]["claims"]["items"]["properties"]["text"]["maxLength"] == 600
    assert manifest["prompt_contract"]["response_schema_template"]["properties"]["unknowns"]["items"]["properties"]["text"]["maxLength"] == 300


def test_aggregate_source_fallback_is_readable_and_budget_fair() -> None:
    first = _source("source-prefix", "No declared question term occurs here. " * 300)
    second = _source("source-nearest", "The safeguard is discussed in this source. " * 300)
    third = _source("source-budget", "A separate institutional record follows. " * 300)

    selected_total = 0
    payloads = []
    for source, terms in (
        (first, ["missing-term"]),
        (second, ["safeguard"]),
        (third, ["missing-term"]),
    ):
        payload, selected_total, _ = _aggregate_source(
            source,
            [],
            terms,
            selected_total=selected_total,
            max_total=6_000,
            source_budget=2_000,
            minimum_chars=MIN_AGGREGATE_SOURCE_CONTEXT_CHARS,
        )
        payloads.append(payload)

    assert selected_total <= 6_000
    assert all(payload["provided_char_count"] >= MIN_AGGREGATE_SOURCE_CONTEXT_CHARS for payload in payloads)
    assert all(payload["provided_char_count"] > 1 for payload in payloads)
    assert payloads[0]["context_selection"]["selection_methods"] == ["SOURCE_PREFIX_CONTEXT"]
    assert payloads[1]["context_selection"]["selection_methods"] == ["QUESTION_TERM_CONTEXT"]
    assert payloads[2]["context_selection"]["selection_methods"] == ["SOURCE_PREFIX_CONTEXT"]

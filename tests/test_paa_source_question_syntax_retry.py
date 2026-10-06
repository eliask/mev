"""Finite Muse JSON-syntax retry retains the first invalid receipt through consumers."""

from __future__ import annotations

import asyncio
import json

import jsonschema
import pytest

from paa.source_question import (
    ANSWER_SCHEMA,
    SYNTAX_RETRY_SUFFIX,
    _apply_stage_attempt_accounting,
    _copy_syntax_attempt_fields,
    _extract_json_object,
    _muse_validate_or_syntax_retry,
    map_schema_for_window,
)


class _FakeMuse:
    def __init__(self, contents: list[str], *, cache_hits: list[bool] | None = None) -> None:
        self.contents = list(contents)
        self.cache_hits = list(cache_hits or [])
        self.tasks: list[str] = []

    async def request(self, task: str, system: str, user: str, **kwargs: object) -> dict:
        self.tasks.append(task)
        content = self.contents.pop(0)
        cache_hit = self.cache_hits.pop(0) if self.cache_hits else False
        return {
            "status": "OK",
            "content": content,
            "request_id": f"req-{len(self.tasks)}",
            "cache_hit": cache_hit,
            "elapsed_seconds": 0.01,
            "raw_response": {"content": content},
        }


def _valid_answer() -> str:
    return (
        '{"answer":"ok","claims":[{"text":"c","epistemic_status":"open",'
        '"evidence":[{"source_id":"s","first_pointer":0,"last_pointer":0}]}],'
        '"unresolved":[],"next_evidence":[]}'
    )


def _valid_map_findings(*, first: int = 0, last: int = 2) -> str:
    return json.dumps(
        {
            "findings": [
                {
                    "text": "fact",
                    "epistemic_status": "open",
                    "first_pointer": first,
                    "last_pointer": last,
                }
            ],
            "notes": "",
        },
        ensure_ascii=False,
    )


def test_extract_json_fenced_without_newline_is_invalid_output_not_indexerror() -> None:
    with pytest.raises(json.JSONDecodeError):
        _extract_json_object("```")
    with pytest.raises(json.JSONDecodeError):
        _extract_json_object("```json")
    parsed = _extract_json_object('```json{"answer":"x","claims":[],"unresolved":[],"next_evidence":[]}```')
    assert parsed["answer"] == "x"


def test_json_parse_error_triggers_one_syntax_retry_with_new_task_id() -> None:
    client = _FakeMuse([_valid_answer()])

    async def _run() -> dict:
        return await _muse_validate_or_syntax_retry(
            client,
            task="paa:source-question:v5:direct:D04",
            system_with_schema="sys",
            user="user",
            schema=ANSWER_SCHEMA,
            max_tokens=100,
            reasoning_effort="medium",
            first_response={
                "status": "OK",
                "content": "{not-json",
                "request_id": "req-0",
                "cache_hit": False,
                "elapsed_seconds": 0.01,
                "raw_response": {"content": "{not-json"},
            },
        )

    result = asyncio.run(_run())
    assert result["status"] == "OK"
    assert result["parsed"]["answer"] == "ok"
    assert result["request_attempt_count"] == 2
    assert result["request_attempts"][0]["parse_status"] == "INVALID_OUTPUT"
    assert result["request_attempts"][0]["request_id"] == "req-0"
    assert result["request_attempts"][1]["parse_status"] == "OK"
    assert result["syntax_retry"]["attempted"] is True
    assert result["syntax_retry"]["retry_task"].endswith(SYNTAX_RETRY_SUFFIX)
    assert result["first_invalid_output"]["status"] == "INVALID_OUTPUT"
    assert result["first_invalid_output"]["request_id"] == "req-0"
    assert "content" not in result["first_invalid_output"]
    assert "raw_response" not in result["first_invalid_output"]
    assert client.tasks == ["paa:source-question:v5:direct:D04" + SYNTAX_RETRY_SUFFIX]


def test_direct_consumer_count_uses_distinct_request_attempts() -> None:
    client = _FakeMuse([_valid_answer()])
    response = asyncio.run(
        _muse_validate_or_syntax_retry(
            client,
            task="paa:source-question:v5:direct:D04",
            system_with_schema="sys",
            user="user",
            schema=ANSWER_SCHEMA,
            max_tokens=100,
            reasoning_effort="medium",
            first_response={
                "status": "OK",
                "content": "{not-json",
                "request_id": "req-0",
                "cache_hit": False,
                "elapsed_seconds": 0.01,
            },
        )
    )
    # Producer→consumer: direct subrequests.count must not hardcode 1.
    subrequests = {
        "mode": "DIRECT_WHOLE_BUNDLE",
        "count": int(response["request_attempt_count"]),
        "request_attempt_count": int(response["request_attempt_count"]),
        "request_attempts": response["request_attempts"],
    }
    assert subrequests["count"] == 2
    assert subrequests["request_attempts"][0]["parse_status"] == "INVALID_OUTPUT"
    assert subrequests["request_attempts"][1]["request_id"] == "req-1"


def test_map_receipt_retains_first_invalid_when_retry_ok() -> None:
    schema = map_schema_for_window(first_pointer=0, last_pointer=5)
    client = _FakeMuse([_valid_map_findings()])
    response = asyncio.run(
        _muse_validate_or_syntax_retry(
            client,
            task="paa:source-question:v5:map:D03:w0",
            system_with_schema="sys",
            user="user",
            schema=schema,
            max_tokens=100,
            reasoning_effort="low",
            first_response={
                "status": "OK",
                "content": "{broken",
                "request_id": "map-first",
                "cache_hit": False,
                "elapsed_seconds": 0.2,
            },
        )
    )
    assert response["status"] == "OK"
    receipt: dict = {
        "window_id": "w0",
        "source_id": "src",
        "status": response.get("status"),
        "request_id": response.get("request_id"),
    }
    _copy_syntax_attempt_fields(receipt, response)
    assert receipt["request_attempt_count"] == 2
    assert receipt["first_invalid_output"]["request_id"] == "map-first"
    assert receipt["syntax_retry"]["attempted"] is True
    assert receipt["request_attempts"][0]["parse_status"] == "INVALID_OUTPUT"
    assert receipt["request_attempts"][1]["parse_status"] == "OK"
    # Terminal OK window still exposes the failed first attempt.
    assert receipt["status"] == "OK"
    assert receipt["first_invalid_output"]["status"] == "INVALID_OUTPUT"


def test_invalid_both_attempts_remain_terminal_invalid() -> None:
    client = _FakeMuse(["{still-bad"])
    result = asyncio.run(
        _muse_validate_or_syntax_retry(
            client,
            task="paa:source-question:v5:direct:D04",
            system_with_schema="sys",
            user="user",
            schema=ANSWER_SCHEMA,
            max_tokens=100,
            reasoning_effort="medium",
            first_response={
                "status": "OK",
                "content": "{first-bad",
                "request_id": "req-0",
                "cache_hit": False,
            },
        )
    )
    assert result["status"] == "INVALID_OUTPUT"
    assert result["request_attempt_count"] == 2
    assert result["request_attempts"][0]["parse_status"] == "INVALID_OUTPUT"
    assert result["request_attempts"][1]["parse_status"] == "INVALID_OUTPUT"
    assert result["first_invalid_output"]["request_id"] == "req-0"
    _apply_stage_attempt_accounting(
        result,
        reduce_or_direct_attempts=result["request_attempts"],
        reduce_or_direct_syntax_retry=result["syntax_retry"],
    )
    assert result["request_attempt_count"] == 2
    assert result["syntax_retry_attempt_count"] == 1


def test_schema_validation_error_does_not_retry() -> None:
    client = _FakeMuse(['{"answer":"","claims":[],"unresolved":[],"next_evidence":[]}'])

    result = asyncio.run(
        _muse_validate_or_syntax_retry(
            client,
            task="paa:source-question:v5:direct:D04",
            system_with_schema="sys",
            user="user",
            schema=ANSWER_SCHEMA,
            max_tokens=100,
            reasoning_effort="medium",
            first_response={
                "status": "OK",
                "content": '{"answer":"","claims":[],"unresolved":[],"next_evidence":[]}',
                "request_id": "req-0",
                "cache_hit": False,
            },
        )
    )
    assert result["status"] == "INVALID_OUTPUT"
    assert "syntax_retry" not in result
    assert result["request_attempt_count"] == 1
    assert client.tasks == []


def test_already_retried_task_does_not_retry_again() -> None:
    client = _FakeMuse(["still-bad"])
    result = asyncio.run(
        _muse_validate_or_syntax_retry(
            client,
            task="paa:source-question:v5:direct:D04" + SYNTAX_RETRY_SUFFIX,
            system_with_schema="sys",
            user="user",
            schema=ANSWER_SCHEMA,
            max_tokens=100,
            reasoning_effort=None,
            first_response={
                "status": "OK",
                "content": "still-bad",
                "request_id": "req-r",
                "cache_hit": False,
            },
        )
    )
    assert result["status"] == "INVALID_OUTPUT"
    assert result["request_attempt_count"] == 1
    assert client.tasks == []


def test_cached_ok_replay_records_single_attempt_without_retry() -> None:
    client = _FakeMuse([_valid_answer()], cache_hits=[True])
    # First response already OK+valid, as a cache hit would present.
    result = asyncio.run(
        _muse_validate_or_syntax_retry(
            client,
            task="paa:source-question:v5:direct:D02",
            system_with_schema="sys",
            user="user",
            schema=ANSWER_SCHEMA,
            max_tokens=100,
            reasoning_effort="medium",
            first_response={
                "status": "OK",
                "content": _valid_answer(),
                "request_id": "cached-req",
                "cache_hit": True,
                "elapsed_seconds": 0.0,
            },
        )
    )
    assert result["status"] == "OK"
    assert result["request_attempt_count"] == 1
    assert result["request_attempts"][0]["cache_hit"] is True
    assert result["request_attempts"][0]["parse_status"] == "OK"
    assert "syntax_retry" not in result
    assert client.tasks == []


def test_out_of_window_pointers_are_consumer_finding_errors_not_syntax_retries() -> None:
    # Invalid pointer relative to window is a finding validation issue, not JSON retry.
    schema = map_schema_for_window(first_pointer=10, last_pointer=12)
    payload = _valid_map_findings(first=0, last=2)
    with pytest.raises(jsonschema.ValidationError):
        # Schema itself rejects out-of-domain pointers before findings consumer.
        jsonschema.Draft202012Validator(schema).validate(json.loads(payload))

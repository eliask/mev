"""answer_task → answer_tasks accounting via FakeMuse (OpenCodeClient path)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from paa.opencode_client import OpenCodeClient
from paa.source_pointers import text_sha256
from paa.source_question import (
    ACCOUNTING_PROJECTION_VERSION,
    MUSE_ANSWER_MAX_TOKENS,
    RESERVED_SYSTEM_SCHEMA_TOKENS,
    SYNTAX_RETRY_SUFFIX,
    SourceQuestionBank,
    _context_budget_tokens,
    answer_task,
    answer_tasks,
    build_task_indexes,
    plan_direct_or_windows,
)


def _valid_answer() -> str:
    return (
        '{"answer":"ok","claims":[{"text":"c","epistemic_status":"open",'
        '"evidence":[{"source_id":"src-long","first_pointer":0,"last_pointer":0}]}],'
        '"unresolved":[],"next_evidence":[]}'
    )


def _valid_map(*, first: int = 0, last: int = 0) -> str:
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


class FakeMuse(OpenCodeClient):
    """OpenCodeClient subclass that never opens network or key files."""

    def __init__(
        self,
        *,
        context_length: int = 200_000,
        direct_script: list[str] | None = None,
        map_first_invalid: bool = False,
        reduce_first_invalid: bool = False,
        cache_hit_direct: bool = False,
        elapsed_seconds: float = 0.11,
    ) -> None:
        self.manifest = {
            "provider": "fake-muse",
            "model_id": "fake-muse",
            "context_length": context_length,
        }
        self.tasks: list[str] = []
        self.direct_script = list(direct_script or [])
        self.map_first_invalid = map_first_invalid
        self.reduce_first_invalid = reduce_first_invalid
        self.cache_hit_direct = cache_hit_direct
        self.elapsed_seconds = elapsed_seconds
        self._map_seen: set[str] = set()
        self._reduce_seen = False
        self.cache_dir = Path("/tmp/paa-fake-muse-cache")

    async def discover(self) -> dict:
        return self.manifest

    async def close(self) -> None:
        return None

    def _answer_for_reduce(self, user: str) -> str:
        try:
            payload = json.loads(user)
        except json.JSONDecodeError:
            return _valid_answer()
        transport = payload.get("quotation_transport") or {}
        witnesses = list(transport.get("presented_witnesses") or [])
        if not witnesses:
            return _valid_answer()
        wit = witnesses[0]
        return json.dumps(
            {
                "answer": "ok",
                "claims": [
                    {
                        "text": "c",
                        "epistemic_status": "open",
                        "evidence": [
                            {
                                "source_id": wit.get("source_id") or "src-long",
                                "first_pointer": wit.get("first_pointer", 0),
                                "last_pointer": wit.get("last_pointer", wit.get("first_pointer", 0)),
                            }
                        ],
                    }
                ],
                "unresolved": [],
                "next_evidence": [],
            },
            ensure_ascii=False,
        )

    async def request(self, task: str, system: str, user: str, **kwargs: object) -> dict:
        self.tasks.append(task)
        cache_hit = False
        if ":direct:" in task:
            if self.direct_script:
                content = self.direct_script.pop(0)
            else:
                content = _valid_answer()
            cache_hit = self.cache_hit_direct and not task.endswith(SYNTAX_RETRY_SUFFIX)
        elif ":map:" in task:
            base = task.removesuffix(SYNTAX_RETRY_SUFFIX)
            if self.map_first_invalid and base not in self._map_seen and not task.endswith(
                SYNTAX_RETRY_SUFFIX
            ):
                self._map_seen.add(base)
                content = "{not-json-map"
            else:
                head = user.split("\n\n", 1)[0]
                window = json.loads(head).get("window") or {}
                first = int(window.get("first_pointer") or 0)
                last = int(window.get("last_pointer") or first)
                # One in-window pointer keeps reduce payload small enough to fit.
                content = _valid_map(first=first, last=min(first, last))
        elif ":reduce:" in task:
            if (
                self.reduce_first_invalid
                and not self._reduce_seen
                and not task.endswith(SYNTAX_RETRY_SUFFIX)
            ):
                self._reduce_seen = True
                content = "{not-json-reduce"
            else:
                content = self._answer_for_reduce(user)
        else:
            content = _valid_answer()
        return {
            "status": "OK",
            "content": content,
            "request_id": f"req-{len(self.tasks)}",
            "cache_hit": cache_hit,
            "elapsed_seconds": self.elapsed_seconds,
            "raw_response": {"content": content},
            "model": {"model_id": "fake-muse"},
        }


def _bank(tmp_path: Path, *, text: str, task_id: str = "D99") -> SourceQuestionBank:
    source_id = "src-long"
    q_path = tmp_path / "questions.json"
    s_path = tmp_path / "sources.json"
    q_path.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "task_id": task_id,
                        "split": "dev",
                        "matter_id": "HE 1/2023 vp",
                        "question": "Testikysymys?",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    s_path.write_text(
        json.dumps(
            {
                "sources": [
                    {
                        "source_id": source_id,
                        "title": "Lähde",
                        "text": text,
                        "text_sha256": text_sha256(text),
                        "event_date": "2023-01-01",
                    }
                ],
                "task_sources": {task_id: [source_id]},
                "source_gaps": [],
            }
        ),
        encoding="utf-8",
    )
    return SourceQuestionBank.load(q_path, s_path)


def _map_reduce_retry_fixture(
    tmp_path: Path, *, task_id: str
) -> tuple[SourceQuestionBank, int, dict[str, int]]:
    """Derive FakeMuse context + short-line source so the declared scenario holds.

    Intended shape (planner-measured, not a fixed prompt-length contract):
    - direct whole-bundle just over budget
    - exactly one scheduled map window
    - reduce (with one short in-window witness quote) fits, so syntax-retry
      accounting can run — genuine REDUCE_OVERBUDGET stays covered elsewhere
    """

    # Short lines keep map→reduce witness quotes small; enough lines push direct over.
    line = "Sana.\n"
    line_count = 80
    margin = 32
    text = line * line_count
    tmp_path.mkdir(parents=True, exist_ok=True)
    bank = _bank(tmp_path, text=text, task_id=task_id)
    probe = FakeMuse(context_length=200_000)
    sized = plan_direct_or_windows(
        build_task_indexes(bank, task_id),
        bank,
        next(task for task in bank.tasks if task["task_id"] == task_id),
        budget_tokens=10**9,
        client=probe,
    )
    direct_est = int(sized.coverage["direct_estimated_tokens"])
    budget_target = direct_est - margin
    if budget_target < 4_000:
        raise AssertionError(
            f"fixture text too small for map-reduce band: direct_est={direct_est}"
        )
    context_length = budget_target + MUSE_ANSWER_MAX_TOKENS + RESERVED_SYSTEM_SCHEMA_TOKENS
    client = FakeMuse(context_length=context_length)
    budget = _context_budget_tokens(client, max_tokens=MUSE_ANSWER_MAX_TOKENS)
    plan = plan_direct_or_windows(
        build_task_indexes(bank, task_id),
        bank,
        next(task for task in bank.tasks if task["task_id"] == task_id),
        budget_tokens=budget,
        client=client,
    )
    if plan.reading_mode == "DIRECT_WHOLE_BUNDLE" or len(plan.windows) != 1:
        raise AssertionError(
            "map-reduce fixture calibration failed: "
            f"mode={plan.reading_mode} windows={len(plan.windows)} "
            f"budget={budget} direct_est={direct_est} context_length={context_length}"
        )
    return bank, context_length, {
        "budget_tokens": budget,
        "direct_estimated_tokens": direct_est,
        "context_length": context_length,
        "source_chars": len(text),
        "line_count": line_count,
        "margin_tokens": margin,
        "window_count": 1,
    }


def test_direct_parse_retry_and_cache_hit_through_answer_tasks(tmp_path: Path) -> None:
    bank = _bank(tmp_path, text="Lyhyt lähdeteksti rivillä.\n")
    client = FakeMuse(direct_script=["{broken", _valid_answer()])
    out = asyncio.run(answer_tasks(bank, ["D99"], client=client, output_dir=tmp_path / "out1"))
    row = out["summary"]["tasks"][0]
    assert row["request_attempt_count"] == 2
    assert row["direct_request_attempt_count"] == 2
    assert row["window_count"] in {0, None}
    assert row["syntax_retry_any"] is True
    assert row["syntax_retry_attempt_count"] == 1
    assert row["accounting_projection_version"] == ACCOUNTING_PROJECTION_VERSION
    assert isinstance(row["elapsed_seconds"], float)
    assert row["provider_attempt_elapsed_sum_seconds"] == 0.22
    assert row["provider_receipt_elapsed_seconds"] == 0.11
    packet = out["results"][0]
    assert packet["first_invalid_output"]["request_id"]
    assert "content" not in packet["first_invalid_output"]

    client_hit = FakeMuse(cache_hit_direct=True, elapsed_seconds=1.5)
    hit = asyncio.run(answer_task(bank, "D99", client=client_hit))
    assert hit["request_attempt_count"] == 1
    assert hit["subrequests"]["request_attempts"][0]["cache_hit"] is True
    assert hit["syntax_retry_any"] is False
    assert hit["provider_receipt_elapsed_seconds"] == 1.5
    # Wall clock is current end-to-end; cached provider latency is not treated as fresh work sum alone.
    assert hit["elapsed_seconds"] is not None
    assert hit["elapsed_seconds"] < hit["provider_receipt_elapsed_seconds"] or hit[
        "provider_attempt_elapsed_sum_seconds"
    ] == 1.5


def test_map_reduce_retries_aggregate_total_attempts(tmp_path: Path) -> None:
    bank, context_length, _calib = _map_reduce_retry_fixture(tmp_path / "d03", task_id="D03")
    client = FakeMuse(
        context_length=context_length,
        map_first_invalid=True,
        reduce_first_invalid=True,
        elapsed_seconds=0.05,
    )
    out = asyncio.run(answer_tasks(bank, ["D03"], client=client, output_dir=tmp_path / "out2"))
    row = out["summary"]["tasks"][0]
    packet = out["results"][0]
    assert packet["reading_mode"] != "DIRECT_WHOLE_BUNDLE"
    assert packet.get("error") != "REDUCE_OVERBUDGET_NO_SILENT_TRUNCATION"
    assert int(row["window_count"] or 0) == 1
    assert int(row["map_request_attempt_count"] or 0) == 2
    assert int(row["reduce_request_attempt_count"] or 0) == 2
    assert row["request_attempt_count"] == 4
    assert row["syntax_retry_any"] is True
    assert row["syntax_retry"] is True
    assert any(task.endswith(SYNTAX_RETRY_SUFFIX) for task in client.tasks if ":map:" in task)
    assert any(task.endswith(SYNTAX_RETRY_SUFFIX) for task in client.tasks if ":reduce:" in task)
    assert packet["request_attempts"][0]["parse_status"] == "INVALID_OUTPUT"


def test_map_only_retry_sets_summary_retry_without_reducer_retry(tmp_path: Path) -> None:
    bank, context_length, _calib = _map_reduce_retry_fixture(tmp_path / "m01", task_id="M01")
    client = FakeMuse(
        context_length=context_length,
        map_first_invalid=True,
        reduce_first_invalid=False,
    )
    out = asyncio.run(answer_tasks(bank, ["M01"], client=client, output_dir=tmp_path / "out3"))
    row = out["summary"]["tasks"][0]
    packet = out["results"][0]
    assert packet.get("error") != "REDUCE_OVERBUDGET_NO_SILENT_TRUNCATION"
    assert row["syntax_retry_any"] is True
    assert packet.get("reduce_syntax_retry") is None
    assert int(row["reduce_request_attempt_count"] or 0) == 1
    assert int(row["map_request_attempt_count"] or 0) == 2
    assert row["request_attempt_count"] == 3

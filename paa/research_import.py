"""Retain externally researched answers through the existing pointer boundary.

Discovery and interpretation happen upstream. Import verifies the supplied
record and exact source spans; it neither repeats inference nor admits meaning.
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path

import jsonschema

from paa.source_question import (
    ANSWER_SCHEMA,
    SCHEMA_VERSION,
    SourceQuestionBank,
    SourceQuestionError,
    bind_answer,
    build_task_indexes,
)

SUBMISSION_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["schema_version", "task_id", "researcher_id", "produced_at", "reading_scope", "result"],
    "properties": {
        "schema_version": {"const": "paa.research_submission.v1"},
        "task_id": {"type": "string", "minLength": 1},
        "researcher_id": {"type": "string", "minLength": 1},
        "produced_at": {"type": "string", "minLength": 1},
        "reading_scope": {"type": "string", "minLength": 1},
        "result": ANSWER_SCHEMA,
    },
}


def import_research_answer(*, questions: Path, sources: Path, submission: Path, output: Path) -> dict:
    """Bind a declared research result, preserving the parent question and gaps.

    Pointer IDs use ``build_task_indexes``' default source domains: lines,
    windows for long text, or ballot rows. No quotation search or positional
    fallback occurs. All claims must bind; valid subanswers may still leave
    declared parts of the parent question unresolved.
    """
    raw = submission.read_bytes()
    record = json.loads(raw)
    jsonschema.Draft202012Validator(SUBMISSION_SCHEMA).validate(record)
    produced_at = datetime.fromisoformat(record["produced_at"])
    if produced_at.tzinfo is None or produced_at.utcoffset() is None:
        raise SourceQuestionError("research produced_at requires an explicit timezone")
    bank = SourceQuestionBank.load(questions, sources)
    task_id = bank.select_task_ids(task_id=record["task_id"])[0]
    task = next(row for row in bank.tasks if row["task_id"] == task_id)
    # Use the same source domains as the existing route/review consumers.
    plan = build_task_indexes(bank, task_id)
    if plan.coverage["state"] != "INDEXED":
        raise SourceQuestionError("research import requires all declared task sources to index")
    bound = bind_answer(record["result"], plan.indexes)
    if not bound["binding_ok"] or any(claim["unbound"] for claim in bound["claims"]):
        raise SourceQuestionError(f"research evidence binding failed: {bound['binding_errors']}")
    packet = {
        "schema_version": SCHEMA_VERSION, "task_id": task_id,
        "matter_id": task.get("matter_id"), "question": task["question"],
        **bound, "coverage": plan.coverage, "source_gaps": plan.declared_gaps,
        "status": "SUCCESS", "admission": "CANDIDATE_NOT_REFERENCE_JUDGED",
        "reading_mode": "EXTERNAL_RESEARCH_SUBMISSION",
        "provider": "external_research_submission", "model_id": None,
        "research_provenance": {
            "researcher_id": record["researcher_id"], "produced_at": record["produced_at"],
            "reading_scope": record["reading_scope"],
            "submission_sha256": hashlib.sha256(raw).hexdigest(),
            "questions_sha256": hashlib.sha256(questions.read_bytes()).hexdigest(),
            "sources_sha256": hashlib.sha256(sources.read_bytes()).hexdigest(),
            "semantic_review_performed_by_import": False,
        },
        "read_completeness": {"state": "UPSTREAM_SCOPE_DECLARED_NOT_INDEPENDENTLY_VERIFIED"},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(packet, ensure_ascii=False, indent=2) + "\n")
    return packet

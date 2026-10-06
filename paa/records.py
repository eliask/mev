"""Schema-shaped records. Extra keys are rejected by the package schemas."""


import hashlib
import json
from functools import cache
from pathlib import Path

from paa.config import RUN_ID, SCHEMA_DIR

try:
    import jsonschema
except ImportError:  # pragma: no cover
    jsonschema = None


def slot(value, basis: str = "EXPLICIT", evidence_ids: list[str] | None = None, note: str | None = None) -> dict:
    return {"value": value, "basis": basis, "evidence_ids": list(evidence_ids or []), "note": note}


def evidence_reference_ids(value) -> set[str]:
    """Collect source references, including nested dates and coverage certificates."""
    refs: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key.endswith("evidence_ids") and isinstance(item, list):
                refs.update(item)
            else:
                refs.update(evidence_reference_ids(item))
    elif isinstance(value, list):
        for item in value:
            refs.update(evidence_reference_ids(item))
    return refs


def timebox(earliest: str | None, latest: str | None, precision: str, basis: str, evidence_ids: list[str]) -> dict:
    return {
        "earliest": earliest,
        "latest": latest,
        "precision": precision,
        "basis": basis,
        "timezone": "Europe/Helsinki",
        "source_evidence_ids": evidence_ids,
    }


def text_evidence(document_id: str, evidence_id: str, text: str, quote: str) -> dict:
    start = text.find(quote)
    if start < 0:
        start = 0
        quote = text[:1] or " "
    end = start + len(quote)
    return {
        "document_version_id": document_id,
        "evidence_id": evidence_id,
        "kind": "text_span",
        "text_sha256": hashlib.sha256(quote.encode("utf-8")).hexdigest(),
        "span_start": start,
        "span_end": max(end, start + 1),
        "quote": quote,
        "normalization_version": "nfc-1",
        "record_locator": None,
        "field_path": "original_text",
        "segment_start_seconds": None,
        "segment_end_seconds": None,
        "context_evidence_ids": [],
    }


def statement_record(document: dict, evidence_id: str, actor_ids: list[str], basis: str) -> dict:
    text = document["text"]
    stated_earliest = document.get("stated_earliest")
    stated_latest = document.get("stated_latest")
    # A missing upper bound is not permission to invent an election-day
    # timestamp.  Keep the open interval explicit; consumers can still use
    # the known lower bound without treating it as an exact publication day.
    if stated_earliest is None and stated_latest is None:
        stated_precision = "unknown"
        stated_basis = "UNRESOLVED"
    elif stated_earliest and stated_latest and stated_earliest == stated_latest:
        stated_precision = "day"
        stated_basis = "CONTEXT_DERIVED"
    else:
        stated_precision = "range"
        stated_basis = "CONTEXT_DERIVED"
    return {
        "schema_version": "1.0",
        "statement_id": document["document_id"],
        "document_version_id": document["document_id"],
        "source_id": document["source_id"],
        "issuer_actor_ids": actor_ids,
        "attribution_basis": basis,
        "source_field_label": document["field_label"],
        "language": document["language"],
        "original_text": text,
        "question_text": None,
        "answer_options": [],
        "selected_answer": None,
        "answer_state": "PRESENT",
        "context_statement_ids": [],
        "stated_at": timebox(
            stated_earliest,
            stated_latest,
            stated_precision,
            stated_basis,
            [evidence_id],
        ),
        "retrieved_at": document.get("retrieved_at"),
        "evidence": [text_evidence(document["document_id"], evidence_id, text, text.strip() or text)],
    }


def proposition_record(
    prop,
    statement_id: str,
    evidence_id: str,
    actor_ids: list[str],
    proposition_id: str,
    *,
    validation_state: str | None = None,
) -> dict:
    modality = {
        "VALUE_OR_SLOGAN": "slogan",
        "BROAD_OBJECTIVE": "broad",
        "POSITION": "evaluative",
        "PROCESS_COMMITMENT": "process",
        "PERSONAL_ACTION_COMMITMENT": "commissive",
        "PERSONAL_RESTRAINT_COMMITMENT": "restraint",
        "POLICY_DESIDERATUM": "desiderative",
        "COLLECTIVE_ACTION_COMMITMENT": "collective",
        "OUTCOME_COMMITMENT": "commissive",
        "MAINTAIN_COMMITMENT": "commissive",
        "PREVENT_COMMITMENT": "commissive",
        "FACTUAL_CLAIM": "assertive",
        "CAUSAL_CLAIM": "assertive",
        "QUESTION": "interrogative",
        "CAUSAL_EFFECT_FORECAST": "forecast",
        "REPORTED_SPEECH": "report",
        "AMBIGUOUS": "unresolved",
    }.get(prop.semantic_type, "unresolved")
    return {
        "schema_version": "1.0",
        "proposition_id": proposition_id,
        "statement_ids": [statement_id],
        "semantic_type": prop.semantic_type,
        "source_text": getattr(prop, "text", ""),
        "source_span": {
            "start": getattr(prop, "source_start", None),
            "end": getattr(prop, "source_end", None),
            "basis": "SOURCE_TEXT",
        },
        "subject_actor_ids": actor_ids,
        "issuer_scope": prop.issuer_scope,
        "predicate": slot(prop.semantic_type, "EXPLICIT", [evidence_id], None),
        "target": slot(prop.targets or None, "EXPLICIT" if prop.targets else "UNRESOLVED", [evidence_id], None),
        "modality": slot(modality, "EXPLICIT", [evidence_id], None),
        "negation": slot(prop.negation, "EXPLICIT", [evidence_id], None),
        "conditions": [slot(prop.condition, "EXPLICIT", [evidence_id], None)] if prop.condition else [],
        "quantity": slot(None, "UNRESOLVED", [], "no metric invented"),
        "baseline": slot(None, "UNRESOLVED", [], None),
        "deadline": slot(
            prop.deadline,
            getattr(prop, "deadline_basis", "EXPLICIT") if prop.deadline else "UNRESOLVED",
            [evidence_id] if prop.deadline else [],
            None,
        ),
        "jurisdiction": slot("FI", "CONTEXT_DERIVED", [evidence_id], "national campaign field"),
        "testability": prop.testability,
        "missing_specification": prop.missing_specification,
        "action_kind": getattr(prop, "action_kind", None),
        "required_capability": getattr(prop, "required_capability", None),
        "observable_action": getattr(prop, "observable_action", False),
        "alternative_interpretations": [],
        "evidence_ids": [evidence_id],
        "run_id": RUN_ID,
        "validation_state": validation_state or getattr(prop, "validation_state", None) or "PROPOSED",
    }


def finding_record(**kwargs) -> dict:
    base = {
        "schema_version": "1.0",
        "plan_id": None,
        "condition_state": "NOT_APPLICABLE",
        "target_state": "NOT_APPLICABLE",
        "action_congruence": "NOT_APPLICABLE",
        "attribution": "UNASSIGNED",
        "support_evidence_ids": [],
        "counterevidence_ids": [],
        "coverage_ids": [],
        "limitations": [],
        "defeaters": [],
        "admission_state": "CANDIDATE",
        "admission_route": "NONE",
        "validation_artifact_ids": [],
        "causal_estimand": None,
        "identification_assumptions": [],
        "run_id": RUN_ID,
    }
    base.update(kwargs)
    if (
        base["admission_state"] == "ADMITTED"
        and (
            not base["support_evidence_ids"]
            or not base["validation_artifact_ids"]
            or base["admission_route"] == "NONE"
        )
    ):
        raise ValueError(f"admitted finding {base.get('finding_id')} is missing support or validation")
    return base


def vote_record(
    vote_id: str,
    event_id: str,
    actor_id: str,
    raw: str,
    title: str,
    alternatives: list[dict],
    evidence_ids: list[str],
    matter: str,
) -> dict:
    interpreted = len(alternatives) >= 2
    return {
        "schema_version": "1.0",
        "vote_id": vote_id,
        "event_id": event_id,
        "matter_ids": [matter] if matter.strip() else [],
        "stage": "UNRESOLVED",
        "motion_text": title,
        "motion_version_id": None,
        "alternatives": alternatives,
        "parent_vote_id": None,
        "voter_actor_id": actor_id,
        "raw_response": raw if raw in {"JAA", "EI", "TYHJA", "POISSA", "OTHER", "UNRESOLVED"} else "OTHER",
        "eligibility_state": "ELIGIBLE",
        "substantive_interpretation": (
            "JAA on otsikon vasen vaihtoehto ja EI oikea. JAA ei tarkoita kannatusta lakiesitykselle."
            if interpreted
            else None
        ),
        "interpretation_state": "VALIDATED" if interpreted else "UNRESOLVED",
        "evidence_ids": evidence_ids,
    }


def alternatives_from_title(title: str, evidence_id: str) -> list[dict]:
    text = (title or "").strip()
    if " / " in text:
        left, right = text.split(" / ", 1)
    elif text.count("/") == 1:
        left, right = text.split("/", 1)
    else:
        return []
    left, right = left.strip(), right.strip()
    if not left or not right:
        return []
    return [
        {"raw_code": "JAA", "alternative_text": left, "evidence_ids": [evidence_id]},
        {"raw_code": "EI", "alternative_text": right, "evidence_ids": [evidence_id]},
    ]


def event_record(event_id: str, when: str | None, evidence_id: str, attributes: dict) -> dict:
    return {
        "schema_version": "1.0",
        "event_id": event_id,
        "event_type": "VOTE",
        "actor_ids": [],
        "role_ids": [],
        "policy_ids": [],
        "event_time": timebox(when, when, "day" if when else "unknown", "EXPLICIT" if when else "UNRESOLVED", [evidence_id]),
        "known_at": timebox(when, when, "day" if when else "unknown", "EXPLICIT" if when else "UNRESOLVED", [evidence_id]),
        "jurisdiction": "FI",
        "document_version_ids": [],
        "evidence_ids": [evidence_id],
        "attributes": attributes,
        "observation_status": "SOURCE_RECORD",
    }


@cache
def _validator(kind: str):
    """Build and cache the schema validator used for one record kind."""

    path = Path(SCHEMA_DIR) / f"{kind}.schema.json"
    schema = json.loads(path.read_text(encoding="utf-8"))
    validator_cls = jsonschema.validators.validator_for(schema)
    validator_cls.check_schema(schema)
    return validator_cls(schema)


def validate(kind: str, record: dict) -> None:
    if jsonschema is None:
        return
    _validator(kind).validate(record)

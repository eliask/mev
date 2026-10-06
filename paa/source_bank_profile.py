"""Explicit boundary profiles for source-native bank metadata.

Frozen banks may use lowercase ``official_*`` source_type/stage/voice values.
This module maps them to the canonical documentary vocabulary used by route and
revision producers without editing the retained bank files or inventing
authority for unknown labels.
"""

from collections.abc import Mapping
from typing import Any, Final

PROFILE_SOURCE_NATIVE_OFFICIAL_V1: Final = "source_native_official_v1"

_STAGE_MAP: Final = {
    "bill_text": "PROPOSAL",
    "proposal": "PROPOSAL",
    "explanatory_proposal": "EXPLANATORY_PROPOSAL",
    "successor_bill": "PROPOSAL",
    "enacted_law": "ENACTED",
    "successor_enacted_law": "ENACTED",
    "prior_enacted_temporal_state": "ENACTED",
    "enacted": "ENACTED",
    "correction": "CORRECTION",
    "procedural_events": "PROCEDURAL_EVENTS",
    "recommendation": "RECOMMENDATION",
    "plenary_speech_and_personal_alternative": "PROPOSAL",
    "first_handling_named_alternatives_and_disposition": "PROCEDURAL_EVENTS",
    "second_handling_institutional_disposition": "PROCEDURAL_EVENTS",
    "consolidated_capture": "CONSOLIDATED_CAPTURE",
    "adopted_legislative_text": "ADOPTED_LEGISLATIVE_TEXT",
}

_SOURCE_TYPE_MAP: Final = {
    "official_bill": "GOVERNMENT_PROPOSAL",
    "official_successor_bill": "GOVERNMENT_PROPOSAL",
    "government_proposal": "GOVERNMENT_PROPOSAL",
    "official_enacted_law": "AMENDING_ACT",
    "official_successor_enacted_law": "AMENDING_ACT",
    "official_prior_enacted_law": "AMENDING_ACT",
    "amending_act": "AMENDING_ACT",
    "official_parliament_speech_api": "PLENARY_SPEECH",
    "official_plenary_record": "PLENARY_RECORD",
    "legislative_initiative": "LEGISLATIVE_INITIATIVE",
    "matter_register": "MATTER_REGISTER",
    "official_correction": "OFFICIAL_CORRECTION",
    "consolidated_act": "CONSOLIDATED_ACT",
    "consolidated_capture": "CONSOLIDATED_CAPTURE",
    "adopted_legislative_text": "ADOPTED_LEGISLATIVE_TEXT",
    "parliament_reply": "PARLIAMENT_REPLY",
}

_VOICE_MAP: Final = {
    "government_submission": "GOVERNMENT",
    "government_submission_successor": "GOVERNMENT",
    "enacted_statute": "LEGISLATURE",
    "enacted_statute_successor": "LEGISLATURE",
    "enacted_statute_prior_temporal_state": "LEGISLATURE",
    "named_individual_speaker": "NAMED_SPEAKER",
    "institutional_chair_record": "PARLIAMENT_RECORD",
    "institutional_record": "PARLIAMENT_RECORD",
    "finlex_consolidation": "CONSOLIDATION_PUBLISHER",
    "parliament": "PARLIAMENT",
}


def normalize_source_meta(meta: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy with canonical stage/type/voice when a known mapping exists.

    Native labels are retained on first normalization and never overwritten by a
    second pass that only re-maps already-canonical values to themselves.
    Unknown labels stay unknown.
    """

    out = dict(meta)
    stage = str(out.get("stage") or "")
    source_type = str(out.get("source_type") or out.get("source_kind") or "")
    voice = str(out.get("source_voice") or "")
    mapped_stage = _STAGE_MAP.get(stage.casefold())
    mapped_type = _SOURCE_TYPE_MAP.get(source_type.casefold())
    mapped_voice = _VOICE_MAP.get(voice.casefold())
    if mapped_stage and stage != mapped_stage:
        out.setdefault("stage_native", stage)
        out["stage"] = mapped_stage
    if mapped_type and source_type != mapped_type:
        out.setdefault("source_type_native", source_type)
        out["source_type"] = mapped_type
    if mapped_voice and voice != mapped_voice:
        out.setdefault("source_voice_native", voice)
        out["source_voice"] = mapped_voice
    out["boundary_profile"] = PROFILE_SOURCE_NATIVE_OFFICIAL_V1
    return out


def task_source_ids_from_bank(bank: Any, task_id: str) -> list[str]:
    """Resolve task→source ids from sources mapping or task-embedded lists."""

    mapped = list(bank.task_sources.get(task_id) or [])
    if mapped:
        return mapped
    for task in bank.tasks:
        if str(task.get("task_id") or "") != task_id:
            continue
        for key in ("task_sources", "task_source_ids"):
            rows = task.get(key)
            if isinstance(rows, list) and rows:
                return [str(item) for item in rows]
    return []


__all__ = [
    "PROFILE_SOURCE_NATIVE_OFFICIAL_V1",
    "normalize_source_meta",
    "task_source_ids_from_bank",
]

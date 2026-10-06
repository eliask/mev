"""Corruption and ownership witnesses for the live action-requirement waist."""

import json
import sqlite3
from dataclasses import FrozenInstanceError

import pytest

from paa.opportunity import opportunity_requirements
from paa.opportunity_codec import OpportunityCodecError, decode_plan, encode_plan, plan_to_wire
from paa.opportunity_records import (
    ActionKind,
    ActionRequirements,
    RequiredCapability,
    VerificationNotApplicable,
    VerificationUnresolved,
)


def _record() -> dict:
    return {"text": "Teen kansalaisaloitteen koulutuksesta.",
            "semantic_type": "PERSONAL_ACTION_COMMITMENT", "action_kind": "INITIATIVE_AUTHORED",
            "required_capability": None, "evidence_ids": ["source-quote"]}


def test_live_adapter_owns_plan_and_retains_civic_capability():
    source = _record()
    plan = opportunity_requirements(source)
    assert type(plan) is ActionRequirements
    assert plan.required_capability is RequiredCapability.PUBLIC_ADVOCACY
    source["evidence_ids"].append("later-caller-mutation")
    assert plan.evidence_ids == ("source-quote",)
    with pytest.raises(FrozenInstanceError):
        plan.source_text = "changed"
    assert decode_plan(encode_plan(plan)) == plan
    wire = plan_to_wire(plan)
    wire["evidence_ids"].append("wire-mutation")
    assert plan.evidence_ids == ("source-quote",)


def test_unknown_action_channel_and_broad_objective_are_distinct():
    assert type(opportunity_requirements({"text": "Tasa-arvoa.", "semantic_type": "BROAD_OBJECTIVE"})) is VerificationNotApplicable
    assert type(opportunity_requirements({"text": "Teen jotakin.", "semantic_type": "PERSONAL_ACTION_COMMITMENT"})) is VerificationUnresolved
    assert type(opportunity_requirements({"text": "Avoin.", "semantic_type": "AMBIGUOUS"})) is VerificationUnresolved


def test_bad_plan_cannot_decode_as_a_valid_negative_or_action():
    plan = opportunity_requirements(_record())
    encoded = encode_plan(plan)
    with pytest.raises(OpportunityCodecError, match="Duplicate"):
        decode_plan(encoded[:-1] + ',"state":"NOT_APPLICABLE"}')
    row = plan_to_wire(plan)
    row["unexpected_authority"] = "MP"
    with pytest.raises(OpportunityCodecError, match="fields"):
        decode_plan(json.dumps(row))
    row = plan_to_wire(plan)
    row["condition_requires_election"] = 1
    with pytest.raises(OpportunityCodecError, match="boolean"):
        decode_plan(json.dumps(row))
    row = plan_to_wire(plan)
    row["required_capability"] = "PERSONAL_FUNDS"
    with pytest.raises(OpportunityCodecError, match="incompatible"):
        decode_plan(json.dumps(row))
    with pytest.raises(OpportunityCodecError, match="Non-finite"):
        decode_plan('{"state":NaN}')


def test_constructor_rejects_unowned_children_and_wire_tokens():
    fields = {"action_kind": ActionKind.INITIATIVE_AUTHORED,
              "required_capability": RequiredCapability.PUBLIC_ADVOCACY,
              "required_role": "Julkinen toiminta", "action_label": "Aloitteen tekeminen",
              "condition": None, "condition_requires_election": False,
              "source_text": "Teen kansalaisaloitteen.", "evidence_ids": ["source"]}
    with pytest.raises(TypeError, match="tuple"):
        ActionRequirements(**fields)
    fields["evidence_ids"] = ("source",)
    fields["action_kind"] = "INITIATIVE_AUTHORED"
    with pytest.raises(TypeError, match="enums"):
        ActionRequirements(**fields)


def test_real_frozen_consumer_rejects_a_schema_valid_capability_change(tmp_path):
    from paa.check import run
    from paa.frozen import build_frozen

    root = tmp_path / "frozen"
    build_frozen(root)
    database = root / "data/paa.sqlite"
    assert run(database, full_corpus=False) == []
    with sqlite3.connect(database) as connection:
        trace_id, content = connection.execute(
            "SELECT trace_id,json FROM evidence_traces WHERE "
            "json_extract(json,'$.authority.verification_plan.action_kind')='INITIATIVE_AUTHORED' LIMIT 1"
        ).fetchone()
        packet = json.loads(content)
        packet["authority"]["verification_plan"]["required_capability"] = "PUBLIC_ADVOCACY"
        # The altered combination is well-formed. The decisive check is its
        # disagreement with the current source proposition's actual channel.
        decode_plan(json.dumps(packet["authority"]["verification_plan"]))
        connection.execute("UPDATE evidence_traces SET json=? WHERE trace_id=?",
                           (json.dumps(packet), trace_id))
    problems = run(database, full_corpus=False)
    assert problems == [f"{trace_id}: action requirements changed; recompile"]

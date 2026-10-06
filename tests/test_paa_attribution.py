"""Focused source/actor attribution tests for canonical trace packets."""

from __future__ import annotations

import json

from paa.attribution import (
    PROPOSED,
    SUPPORTED,
    UNKNOWN,
    attach_attribution,
    derive_decision_episodes,
    resolve_commitment_carrier,
)


def _evidence(*ids: str) -> list[dict]:
    return [
        {"evidence_id": evidence_id, "quote": evidence_id, "record_locator": evidence_id}
        for evidence_id in ids
    ]


def _packet(
    *,
    text: str,
    actor_id: str | None = "mp-1",
    proposition: dict | None = None,
    statement: dict | None = None,
    objects: list[dict] | None = None,
    relations: list[dict] | None = None,
    actions: list[dict] | None = None,
    authority: dict | None = None,
    evidence_ids: tuple[str, ...] = ("statement-e",),
) -> dict:
    proposition = proposition or {
        "text": text,
        "semantic_type": "PERSONAL_ACTION_COMMITMENT",
        "issuer_scope": "SELF",
        "validation_state": "PROPOSED",
        "evidence_ids": ["statement-e"],
    }
    statement = statement or {
        "text": text,
        "evidence_ids": ["statement-e"],
    }
    return {
        "schema_version": "1.0",
        "trace_id": "trace-test",
        "proposition_id": "prop-test",
        "actor_id": actor_id,
        "statement": statement,
        "proposition": proposition,
        "retrieved_objects": objects or [],
        "relations": relations or [],
        "actions": actions or [],
        "authority": authority or {},
        "evidence": _evidence(*evidence_ids),
    }


def _object(
    object_id: str,
    matter_id: str,
    *,
    kind: str = "LEGISLATIVE_INITIATIVE",
    date: str = "2024-01-01",
    evidence_id: str = "object-e",
    authors: list[dict] | None = None,
    disposition: dict | None = None,
) -> dict:
    return {
        "object_id": object_id,
        "kind": kind,
        "matter_id": matter_id,
        "text": "Official source text",
        "action_date": date,
        "action_date_basis": "SIGNATURE_DATE" if kind == "LEGISLATIVE_INITIATIVE" else "SPEECH_DATE",
        "authors": authors or [],
        "evidence_ids": [evidence_id],
        "disposition": disposition or {"state": "UNRESOLVED", "evidence_ids": []},
    }


def _relation(object_id: str, evidence_id: str = "relation-e", **extra) -> dict:
    return {
        "object_id": object_id,
        "status": "SAME_MATTER",
        "validation_state": "VALID",
        "evidence_ids": [evidence_id],
        **extra,
    }


def _action(object_id: str, actor_id: str, kind: str, evidence_id: str = "action-e") -> dict:
    return {
        "object_id": object_id,
        "actor_id": actor_id,
        "kind": kind,
        "date": "2024-01-01",
        "role": "ACTOR",
        "state": "OBSERVED_ALIGNED_ACTION",
        "evidence_ids": [evidence_id],
    }


def test_personal_scope_and_explicit_actor_are_separate_from_proposed_commitment() -> None:
    packet = _packet(
        text="Lupaan tehdä lakialoitteen.",
        statement={
            "text": "Lupaan tehdä lakialoitteen.",
            "evidence_ids": ["statement-e"],
            "issuer_actor_ids": ["mp-1"],
            "attribution_basis": "SOURCE_IDENTITY_REVIEW",
        },
        proposition={
            "text": "Lupaan tehdä lakialoitteen.",
            "semantic_type": "PERSONAL_ACTION_COMMITMENT",
            "issuer_scope": "SELF",
            "action_kind": "INITIATIVE_AUTHORED",
            "validation_state": "PROPOSED",
            "evidence_ids": ["statement-e"],
        },
    )
    carrier = resolve_commitment_carrier(packet)

    assert carrier["type"] == "PERSON"
    assert carrier["scope"]["state"] == SUPPORTED
    assert carrier["scope"]["value"] == "PERSONAL"
    assert carrier["explicit_attribution"]["state"] == SUPPORTED
    assert carrier["commitment_state"] == PROPOSED
    assert carrier["action_kind"]["state"] == PROPOSED


def test_passive_sentence_has_no_guessed_carrier() -> None:
    packet = _packet(
        text="Dieselvero on poistettava.",
        actor_id="candidate-1",
        proposition={
            "text": "Dieselvero on poistettava.",
            "semantic_type": "POLICY_DESIDERATUM",
            "issuer_scope": "UNSPECIFIED_WE",
            "validation_state": "PROPOSED",
            "evidence_ids": ["statement-e"],
        },
    )
    carrier = resolve_commitment_carrier(packet)

    assert carrier["type"] == "UNRESOLVED"
    assert carrier["scope"]["state"] == SUPPORTED
    assert carrier["scope"]["value"] == "PASSIVE"
    assert carrier["explicit_attribution"]["state"] == UNKNOWN
    assert carrier["actor_ids"] == []
    assert carrier["commitment_state"] == UNKNOWN


def test_process_commitment_retains_a_carrier_without_an_observable_act():
    text = "Lupaan perustaa päätökseni parhaaseen saatavilla olevaan tietoon."
    packet = _packet(text=text, proposition={"text": text, "semantic_type": "PROCESS_COMMITMENT",
        "issuer_scope": "SELF", "validation_state": "PROPOSED", "evidence_ids": ["statement-e"]})
    carrier = resolve_commitment_carrier(packet)
    assert carrier["commitment_state"] == PROPOSED
    assert carrier["action_kind"]["state"] != SUPPORTED
    for semantic_type in ["FACTUAL_CLAIM", "CAUSAL_CLAIM", "QUESTION", "UNKNOWN_CLASS"]:
        packet["proposition"]["semantic_type"] = semantic_type
        assert resolve_commitment_carrier(packet)["commitment_state"] == UNKNOWN


def test_collective_scope_does_not_invent_party_or_candidate_actor() -> None:
    packet = _packet(
        text="Lupaamme tehdä aloitteen.",
        proposition={
            "text": "Lupaamme tehdä aloitteen.",
            "semantic_type": "COLLECTIVE_ACTION_COMMITMENT",
            "issuer_scope": "UNSPECIFIED_WE",
            "validation_state": "PROPOSED",
            "evidence_ids": ["statement-e"],
        },
    )
    carrier = resolve_commitment_carrier(packet)
    assert carrier["scope"]["value"] == "COLLECTIVE"
    assert carrier["type"] == "UNRESOLVED"
    assert carrier["actor_ids"] == []

    party_packet = _packet(
        text="Puolueemme ohjelmassa lupaamme tehdä aloitteen.",
        proposition={
            "text": "Puolueemme ohjelmassa lupaamme tehdä aloitteen.",
            "semantic_type": "COLLECTIVE_ACTION_COMMITMENT",
            "issuer_scope": "PARTY",
            "validation_state": "VALIDATED",
            "evidence_ids": ["statement-e"],
        },
    )
    party_carrier = resolve_commitment_carrier(party_packet)
    assert party_carrier["type"] == "PARTY"
    assert party_carrier["actor_ids"] == []


def test_institution_mentions_do_not_become_carriers_without_commitment_subject() -> None:
    demand = _packet(
        text="Vaadin että hallitus tekee muutoksen.",
        statement={
            "text": "Vaadin että hallitus tekee muutoksen.",
            "evidence_ids": ["statement-e"],
            "issuer_actor_ids": ["mp-1"],
            "attribution_basis": "SOURCE_IDENTITY_REVIEW",
        },
        proposition={
            "text": "Vaadin että hallitus tekee muutoksen.",
            "semantic_type": "POSITION",
            "issuer_scope": "SELF",
            "validation_state": "VALIDATED",
            "evidence_ids": ["statement-e"],
        },
    )
    carrier = resolve_commitment_carrier(demand)
    assert carrier["type"] == "PERSON"
    assert carrier["commitment_state"] == UNKNOWN

    event = _packet(
        text="Hallitus leikkasi koulutuksesta.",
        actor_id=None,
        proposition={
            "text": "Hallitus leikkasi koulutuksesta.",
            "semantic_type": "POSITION",
            "issuer_scope": "COLLECTIVE",
            "validation_state": "VALIDATED",
            "evidence_ids": ["statement-e"],
        },
    )
    event_carrier = resolve_commitment_carrier(event)
    assert event_carrier["type"] == "UNRESOLVED"

    institutional_commitment = _packet(
        text="Hallitus lupaa vahvistaa hoitoa.",
        actor_id=None,
        proposition={
            "text": "Hallitus lupaa vahvistaa hoitoa.",
            "semantic_type": "COLLECTIVE_ACTION_COMMITMENT",
            "issuer_scope": "COLLECTIVE",
            "validation_state": "PROPOSED",
            "evidence_ids": ["statement-e"],
        },
    )
    institutional_carrier = resolve_commitment_carrier(institutional_commitment)
    assert institutional_carrier["type"] == "GOVERNMENT"
    assert institutional_carrier["commitment_state"] == PROPOSED


def test_decision_episodes_group_only_exact_matter_ids_and_ignore_model_relation() -> None:
    objects = [
        _object("initiative", "LA 1/2024 vp", evidence_id="object-e"),
        _object("disposition", "LA 1/2024 vp", kind="VOTE", evidence_id="decision-e"),
        _object("other", "LA 2/2024 vp", evidence_id="other-e"),
    ]
    objects[1]["disposition"] = {"state": "RECORDED", "date": "2024-02-01", "evidence_ids": ["decision-e"]}
    packet = _packet(
        text="Lupaan tehdä aloitteen.",
        objects=objects,
        relations=[
            _relation("initiative"),
            _relation("disposition", "decision-relation-e"),
            _relation("other", "other-relation-e", admission_state="PROPOSED"),
        ],
        actions=[_action("initiative", "mp-1", "INITIATIVE_AUTHORED")],
        evidence_ids=("statement-e", "object-e", "decision-e", "other-e", "relation-e", "decision-relation-e", "other-relation-e", "action-e"),
    )
    episodes = derive_decision_episodes(packet)

    assert len(episodes) == 1
    assert episodes[0]["matter_id"] == "LA 1/2024 vp"
    assert episodes[0]["object_ids"] == ["disposition", "initiative"]
    assert episodes[0]["decision"]["state"] == SUPPORTED
    assert episodes[0]["action"]["state"] == SUPPORTED


def test_question_and_speech_actions_are_typed_but_speech_is_not_first_signatory() -> None:
    question = _object(
        "question",
        "KK 1/2023 vp",
        kind="WRITTEN_QUESTION",
        date="2023-04-20",
        evidence_id="question-e",
        authors=[{"person_id": "1", "role": "AUTHOR", "evidence_ids": ["question-e"]}],
    )
    speech = _object(
        "speech",
        "HE 1/2023 vp",
        kind="SPEECH",
        date="2023-04-25",
        evidence_id="speech-e",
        authors=[{"person_id": "1503", "role": "ACTOR", "evidence_ids": ["speech-e"]}],
    )
    packet = _packet(
        text="Lupaan puhua eduskunnassa.",
        actor_id="1503",
        objects=[question, speech],
        relations=[_relation("question", "question-relation-e"), _relation("speech", "speech-relation-e")],
        actions=[
            _action("question", "1", "QUESTION_FILED", "question-action-e"),
            _action("speech", "1503", "SPEECH_DELIVERED", "speech-action-e"),
        ],
        evidence_ids=(
            "statement-e", "question-e", "speech-e", "question-relation-e", "speech-relation-e",
            "question-action-e", "speech-action-e",
        ),
    )
    attached = attach_attribution(packet)
    envelopes = {item["matter_id"]: item for item in attached["attribution_envelopes"]}
    assert envelopes["KK 1/2023 vp"]["dimensions"]["recorded_action"]["state"] == UNKNOWN
    speech_envelope = envelopes["HE 1/2023 vp"]
    assert speech_envelope["dimensions"]["recorded_action"]["state"] == SUPPORTED
    assert speech_envelope["dimensions"]["actor_role"]["value"]["role"] == "ACTOR"
    assert speech_envelope["dimensions"]["first_signatory"]["state"] == UNKNOWN


def test_first_signatory_is_not_drafter_and_different_actor_is_not_promoted() -> None:
    obj = _object(
        "initiative",
        "LA 3/2024 vp",
        authors=[
            {"person_id": "2", "role": "AUTHOR", "evidence_ids": ["object-e"]},
            {"person_id": "1", "role": "COSIGNER", "evidence_ids": ["cosigner-e"]},
        ],
    )
    packet = _packet(
        text="Lupaan tehdä aloitteen.",
        actor_id="mp-1",
        objects=[obj],
        relations=[_relation("initiative")],
        actions=[_action("initiative", "mp-1", "INITIATIVE_AUTHORED")],
        evidence_ids=("statement-e", "object-e", "cosigner-e", "relation-e", "action-e"),
    )
    envelope = attach_attribution(packet)["attribution_envelopes"][0]
    dimensions = envelope["dimensions"]
    assert dimensions["first_signatory"]["state"] == UNKNOWN
    assert dimensions["actor_role"]["state"] == SUPPORTED
    assert dimensions["actor_role"]["value"]["role"] == "COSIGNER"
    assert dimensions["drafter"]["state"] == UNKNOWN


def test_drafter_field_for_another_actor_is_not_promoted_to_current_envelope() -> None:
    obj = _object(
        "initiative",
        "LA 6/2024 vp",
        authors=[
            {"person_id": "2", "role": "AUTHOR", "evidence_ids": ["object-e"]},
            {"person_id": "1", "role": "COSIGNER", "evidence_ids": ["cosigner-e"]},
        ],
    )
    obj["drafter_actor_id"] = "2"
    packet = _packet(
        text="Lupaan tehdä aloitteen.",
        actor_id="mp-1",
        objects=[obj],
        relations=[_relation("initiative")],
        actions=[_action("initiative", "mp-1", "INITIATIVE_AUTHORED")],
        evidence_ids=("statement-e", "object-e", "cosigner-e", "relation-e", "action-e"),
    )
    dimensions = attach_attribution(packet)["attribution_envelopes"][0]["dimensions"]
    assert dimensions["drafter"]["state"] == UNKNOWN


def test_recorded_action_is_bounded_congruence_not_full_fulfillment() -> None:
    obj = _object("initiative", "LA 7/2024 vp")
    packet = _packet(
        text="Lupaan tehdä aloitteen.",
        statement={
            "text": "Lupaan tehdä aloitteen.",
            "evidence_ids": ["statement-e"],
            "issuer_actor_ids": ["mp-1"],
            "attribution_basis": "SOURCE_IDENTITY_REVIEW",
        },
        proposition={
            "text": "Lupaan tehdä aloitteen.",
            "semantic_type": "PERSONAL_ACTION_COMMITMENT",
            "issuer_scope": "SELF",
            "action_kind": "INITIATIVE_AUTHORED",
            "validation_state": "VALIDATED",
            "evidence_ids": ["statement-e"],
        },
        objects=[obj],
        relations=[_relation("initiative")],
        actions=[_action("initiative", "mp-1", "INITIATIVE_AUTHORED")],
        evidence_ids=("statement-e", "object-e", "relation-e", "action-e"),
    )
    dimensions = attach_attribution(packet)["attribution_envelopes"][0]["dimensions"]
    assert dimensions["action_congruence"]["state"] == SUPPORTED
    assert dimensions["commitment_fulfillment"]["state"] == UNKNOWN
    assert dimensions["commitment_fulfillment"]["value"] is None


def test_model_or_unresolved_relation_cannot_create_episode_or_causal_claim() -> None:
    obj = _object("initiative", "LA 4/2024 vp")
    packet = _packet(
        text="Lupaan tehdä aloitteen.",
        objects=[obj],
        relations=[_relation("initiative", admission_state="PROPOSED")],
        actions=[{
            **_action("initiative", "mp-1", "INITIATIVE_AUTHORED"),
            "admission_state": "PROPOSED",
        }],
        evidence_ids=("statement-e", "object-e", "relation-e", "action-e"),
    )
    artifact = attach_attribution(packet)
    assert artifact["decision_episodes"] == []
    assert artifact["attribution_envelopes"] == []
    assert "causal" not in json.dumps(artifact["commitment_carrier"]).casefold()


def test_action_is_not_cause_or_effect_even_when_recorded() -> None:
    obj = _object("initiative", "LA 5/2024 vp")
    packet = _packet(
        text="Lupaan tehdä aloitteen.",
        objects=[obj],
        relations=[_relation("initiative")],
        actions=[_action("initiative", "mp-1", "INITIATIVE_AUTHORED")],
        evidence_ids=("statement-e", "object-e", "relation-e", "action-e"),
    )
    envelope = attach_attribution(packet)["attribution_envelopes"][0]
    assert envelope["dimensions"]["recorded_action"]["state"] == SUPPORTED
    assert envelope["dimensions"]["causal_effect"]["state"] == UNKNOWN
    assert envelope["dimensions"]["private_constraints"]["state"] == UNKNOWN

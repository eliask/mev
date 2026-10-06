"""Offline contract tests for local-model relation proposals."""

import json
from pathlib import Path

import pytest

from paa.initiative_benchmark import build_benchmark
from paa.llm_relations import (
    batch_proposal_schema,
    build_batch_relation_request,
    build_batch_verification_request,
    build_relation_request,
    build_verification_request,
    clip_context,
    fingerprint,
    load_prompt,
    measure_relation_pairs,
    proposal_schema,
    proposal_to_review,
    validate_batch_relation_response,
    validate_batch_verification_response,
    validate_relation_response,
    validate_verification_response,
    verification_artifact,
    verification_schema,
)
from paa.relations import verified_review

FIXTURES = Path("paa/contracts/fixtures")


def _real_case():
    case = json.loads((FIXTURES / "initiative_case_arja_juvonen.json").read_text(encoding="utf-8"))
    objects = [
        json.loads(line)
        for line in (FIXTURES / "initiative_retrieval_objects.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    obj = next(item for item in objects if item["object_id"] == "eduskunta:LA 72/2017 vp")
    statement_text = case["candidate"]["quote"]
    statement = {
        "statement_id": case["candidate"]["source_document_id"],
        "source_id": case["candidate"]["source_id"],
        "field_label": case["candidate"]["field_label"],
        "language": "fi",
        "original_text": statement_text,
        "evidence": [{"evidence_id": "yle2011-3823-e"}],
    }
    proposition = {
        "proposition_id": case["relation_review"]["proposition_id"],
        "source_text": statement_text,
        "semantic_type": "POLICY_DESIDERATUM",
        "subject_actor_ids": [case["candidate"]["actor_id"]],
        "target": {"value": ["vanhuslaki", "hoitajavahvuus"]},
        "evidence_ids": ["yle2011-3823-e"],
    }
    return case, proposition, statement, obj


def _mixed_proposition_case():
    """Two propositions from one source document for sharing controls."""

    _case, proposition, statement, first_object = _real_case()
    statement = {
        **statement,
        "statement_id": "multi-proposition-source",
        "original_text": "Vanhuspalvelut on turvattava. Koulutus on pidettävä maksuttomana.",
    }
    first_text, second_text = statement["original_text"].split(" ", 1)
    proposition_a = {**proposition, "proposition_id": "multi-proposition-a", "source_text": first_text}
    proposition_b = {
        **proposition,
        "proposition_id": "multi-proposition-b",
        "source_text": second_text,
        "target": {"value": ["koulutus"]},
    }
    return proposition_a, proposition_b, statement, first_object


def _response(request, *, status="SAME_POLICY_OBJECT", action_alignment="RELATED"):
    obj = request["model_input"]["official_object"]
    prop = request["model_input"]["proposition"]
    object_text = request["validation"]["object_text"]
    return {
        "status": status,
        "matter_id": obj["matter_id"],
        "identity_basis": "EXPLICIT_MATTER_ID" if status.startswith("SAME_") else "UNRESOLVED",
        "statement_quote": prop["source_text"][:160],
        "object_quote": object_text[:180],
        "normalized_target": "vanhuspalvelulain hoitajamitoituksen vähimmäismäärä" if status.startswith("SAME_") else None,
        "target_scope": "PARTIAL_COMPONENT" if status.startswith("SAME_") else "UNRESOLVED",
        "bounded_claim": "Official text addresses the cited policy object; it does not establish fulfilment.",
        "rationale": "The exact source passages support only this scoped relation.",
        "action_alignment": action_alignment,
        "evidence_ids": [request["statement_evidence_ids"][0], *request["object_evidence_ids"][:1]],
        "counterevidence_ids": [],
    }


def test_prompts_are_local_and_versioned():
    assert "JAA" in load_prompt("relation_v1", stage="propose")
    assert "counterevidence" in load_prompt("relation_v1", stage="verify").lower()
    assert load_prompt("relation_v2", stage="propose") != load_prompt("relation_v1", stage="propose")
    with pytest.raises(ValueError):
        load_prompt("relation_v9", stage="propose")


def test_real_source_pair_becomes_proposed_review_but_broad_text_is_not_aligned():
    _case, proposition, statement, obj = _real_case()
    request = build_relation_request(proposition, statement, obj)
    assert request["model_input"]["source_evidence_ids"]["statement"] == request["statement_evidence_ids"]
    assert request["model_input"]["source_evidence_ids"]["official_object"] == request["object_evidence_ids"]
    assert request["model_input"]["coverage"] == request["coverage"]
    response = _response(request, action_alignment="ALIGNED")

    result = validate_relation_response(response, request)

    assert result["valid"] is True
    assert result["proposal"]["status"] == "SAME_POLICY_OBJECT"
    assert result["proposal"]["action_alignment"] == "RELATED"
    assert any("broad" in warning for warning in result["warnings"])

    review = proposal_to_review(response, request, model_id="test-local")
    assert review["review_state"] == "PROPOSED"
    assert review["admission_state"] == "PROPOSED"
    assert review["reviewer_independence"]["status"] == "NOT_INDEPENDENT"
    evidence = {
        evidence_id: {"evidence_id": evidence_id}
        for evidence_id in request["statement_evidence_ids"] + request["object_evidence_ids"]
    }
    statement_record = {
        "original_text": request["validation"]["statement_text"],
        "evidence": [{"evidence_id": "yle2011-3823-e"}],
    }
    proposition_record = {"proposition_id": proposition["proposition_id"], "source_text": proposition["source_text"]}
    object_record = {"object_id": obj["object_id"], "matter_id": obj["matter_id"], "text": obj["text"], "evidence_ids": obj["evidence_ids"]}
    # Source quotes and hashes make this a replayable proposal, but its
    # explicit model admission state must not cross the canonical review gate.
    assert not verified_review(review, statement_record, proposition_record, object_record, evidence)


def test_relation_model_context_compacts_large_author_roster_without_losing_raw_validation():
    _case, proposition, statement, obj = _real_case()
    expanded = dict(obj)
    expanded["authors"] = list(obj["authors"]) + [
        {"person_id": f"extra-{index}", "name": f"Extra {index}", "role": "COSIGNER"}
        for index in range(140)
    ]

    request = build_relation_request(proposition, statement, expanded)
    model_object = request["model_input"]["official_object"]

    assert model_object["authors_total_count"] == len(expanded["authors"])
    assert model_object["authors_omitted_count"] == len(expanded["authors"]) - len(model_object["authors"])
    assert model_object["authors_roster_complete"] is False
    subject_id = proposition["subject_actor_ids"][0].removeprefix("mp-")
    assert any(author.get("person_id") == subject_id for author in model_object["authors"])
    assert all(not author.get("person_id", "").startswith("extra-") for author in model_object["authors"])
    assert request["validation"]["object_text"] == expanded["text"]


def test_official_title_is_an_exact_anchor_when_body_omits_title():
    _case, proposition, statement, obj = _real_case()
    object_with_separate_title = dict(obj)
    object_with_separate_title["title"] = "Official title retained as source field"
    object_with_separate_title["text"] = "Official body does not repeat its title."
    request = build_relation_request(
        proposition,
        statement,
        object_with_separate_title,
        prompt_version="relation_v5",
    )
    response = _v3_response(request)
    response["identity_basis"] = "REVIEWED_SOURCE_LINK"

    result = validate_relation_response(response, request)

    assert result["valid"] is True


def test_vote_without_decisive_motion_abstains_even_when_model_claims_same_and_aligned():
    proposition = {
        "proposition_id": "p-vote-1",
        "source_text": "Äänestän tämän lakimuutoksen puolesta.",
        "semantic_type": "PERSONAL_ACTION_COMMITMENT",
        "subject_actor_ids": ["mp-1"],
        "evidence_ids": ["statement-e"],
    }
    statement = {
        "statement_id": "s-vote-1",
        "original_text": proposition["source_text"],
        "evidence": [{"evidence_id": "statement-e"}],
    }
    obj = {
        "object_id": "vote-1",
        "matter_id": "HE 1/2024 vp",
        "kind": "VOTE",
        "title": "Lakimuutosta koskeva äänestys",
        "text": "Päätösesitys ja sen käsittely.",
        "evidence_ids": ["vote-e"],
    }
    request = build_relation_request(
        proposition,
        statement,
        obj,
        actor_ballot={"actor_id": "mp-1", "raw_response": "JAA", "evidence_ids": ["ballot-e"]},
        max_object_chars=80,
    )
    result = validate_relation_response(
        {
            "status": "SAME_MATTER",
            "matter_id": obj["matter_id"],
            "identity_basis": "EXPLICIT_MATTER_ID",
            "statement_quote": proposition["source_text"],
            "object_quote": obj["text"],
            "normalized_target": "lakimuutos",
            "target_scope": "EXACT",
            "bounded_claim": "The object is the same matter if the motion is supplied.",
            "rationale": "The model saw a candidate matter but not its alternatives.",
            "action_alignment": "ALIGNED",
            "evidence_ids": ["statement-e", "vote-e"],
            "counterevidence_ids": [],
        },
        request,
    )
    assert result["valid"] is True
    assert result["abstained"] is True
    assert result["proposal"]["status"] == "UNRESOLVED"
    assert result["proposal"]["action_alignment"] == "UNRESOLVED"
    assert any("decisive motion" in warning for warning in result["warnings"])

    complete_vote = {
        **obj,
        "alternatives": [
            {"raw_code": "JAA", "alternative_text": "Hyväksytään muutos", "evidence_ids": ["vote-e"]},
            {"raw_code": "EI", "alternative_text": "Hylätään muutos", "evidence_ids": ["vote-e"]},
        ],
    }
    complete_request = build_relation_request(
        proposition,
        statement,
        complete_vote,
        actor_ballot={"actor_id": "mp-1", "raw_response": "JAA", "evidence_ids": ["ballot-e"]},
    )
    complete_response = {
        "status": "SAME_MATTER",
        "matter_id": complete_vote["matter_id"],
        "identity_basis": "EXPLICIT_MATTER_ID",
        "statement_quote": proposition["source_text"],
        "object_quote": complete_vote["text"],
        "normalized_target": "lakimuutos",
        "target_scope": "EXACT",
        "bounded_claim": "The source records a vote on this motion.",
        "rationale": "The vote alternatives are present, but their policy effect was not interpreted.",
        "action_alignment": "ALIGNED",
        "evidence_ids": ["statement-e", "vote-e"],
        "counterevidence_ids": [],
    }
    complete_result = validate_relation_response(complete_response, complete_request)
    assert complete_result["valid"] is True
    assert complete_result["proposal"]["action_alignment"] == "UNRESOLVED"
    assert any("raw ballot" in warning for warning in complete_result["warnings"])


def test_cosigner_is_not_author_and_is_not_action_aligned():
    proposition = {
        "proposition_id": "p-init-1",
        "source_text": "Teen tästä lakialoitteen.",
        "semantic_type": "PERSONAL_ACTION_COMMITMENT",
        "subject_actor_ids": ["mp-1129"],
        "evidence_ids": ["statement-e"],
    }
    statement = {"statement_id": "s-init-1", "original_text": proposition["source_text"], "evidence": [{"evidence_id": "statement-e"}]}
    obj = {
        "object_id": "eduskunta:LA 1/2024 vp",
        "matter_id": "LA 1/2024 vp",
        "kind": "LEGISLATIVE_INITIATIVE",
        "title": "Lakialoite",
        "text": "Eduskunnalle ehdotetaan lakia.",
        "authors": [{"person_id": "1129", "name": "Actor", "role": "COSIGNER", "evidence_ids": ["object-e"]}],
        "evidence_ids": ["object-e"],
    }
    request = build_relation_request(proposition, statement, obj)
    result = validate_relation_response(_response(request, action_alignment="ALIGNED"), request)
    assert result["valid"] is True
    assert result["proposal"]["action_alignment"] == "RELATED"
    assert any("AUTHOR role" in warning for warning in result["warnings"])


def test_verifier_receives_counterevidence_and_remains_non_independent():
    _case, proposition, statement, obj = _real_case()
    # This is a narrow action-shaped test input so the guard does not turn the
    # verifier exercise into a broad-policy alignment assertion.
    proposition = {**proposition, "semantic_type": "PERSONAL_ACTION_COMMITMENT"}
    request = build_relation_request(
        proposition,
        statement,
        obj,
        counterevidence=[{"evidence_id": "counter-e", "kind": "official", "text": "A later source narrows the claim."}],
    )
    proposal = _response(request, action_alignment="ALIGNED")
    proposal["actor_role"] = "AUTHOR"
    review = proposal_to_review(proposal, request, model_id="local-test")
    verification_request = build_verification_request(
        review,
        request,
        counterevidence=[{"evidence_id": "counter-e", "kind": "official", "text": "A later source narrows the claim."}],
    )
    verification_response = {
        **proposal,
        "verdict": "NARROWER_CLAIM_SUPPORTED",
        "counterevidence_ids": ["counter-e"],
    }
    result = validate_verification_response(verification_response, verification_request)
    assert result["valid"] is True
    assert result["independence"]["status"] == "NOT_INDEPENDENT"
    artifact = verification_artifact(verification_response, verification_request, model_id="local-test")
    assert artifact["admission_state"] == "PROPOSED"
    assert artifact["reviewer_independence"]["basis"] == "same-model-separate-context"


def test_measurement_uses_real_source_anchored_pairs_without_admitting_them():
    benchmark = build_benchmark()
    pairs = [{"object_id": item["object_id"], "gold_label": item["gold_label"]} for item in benchmark["judgments"]]
    reviews = [{"object_id": item["object_id"], "status": item["review_status"]} for item in benchmark["judgments"]]
    report = measure_relation_pairs(pairs, reviews)
    assert report["evaluation_scope"] == "independent_source_anchored_pairs_only"
    assert report["pair_count"] == 7
    assert report["positive_count"] == 2
    assert report["true_positive_count"] == 2
    assert report["false_positive_count"] == 0
    assert report["rejected_count"] == 5
    assert report["abstention_count"] == 0
    assert report["precision"] == 1.0
    assert report["recall"] == 1.0
    assert all(item["predicted"] != "ADMITTED" for item in report["judgments"])


def test_exact_anchor_or_source_hash_tampering_is_rejected():
    _case, proposition, statement, obj = _real_case()
    request = build_relation_request(proposition, statement, obj)
    tampered = _response(request)
    tampered["statement_quote"] = "same words, different source"
    result = validate_relation_response(tampered, request)
    assert result["valid"] is False
    assert any("statement_quote" in error for error in result["errors"])

    request["validation"]["object_sha256"] = "0" * 64
    result = validate_relation_response(_response(request), request)
    assert result["valid"] is False
    assert any("object validation hash" in error for error in result["errors"])


def test_hashes_are_exact_utf8_anchors():
    assert fingerprint("ä") == "33e6d73fee82904c8d7afb78de1154d1e8dc2a0edb08120e63df5b9385c2d9cc"


def test_clipping_never_exceeds_small_budget_and_preserves_required_anchor():
    clipped = clip_context("alku " + "x" * 120 + " loppu", max_chars=32, required_spans=("x" * 120,))
    assert len(clipped["text"]) <= 32
    assert clipped["complete"] is False
    assert clipped["missing_anchors"] == ["x" * 120]


def test_relation_batch_shares_statement_and_validates_every_pair_against_full_sources():
    _case, proposition, statement, first_object = _real_case()
    all_objects = [
        json.loads(line)
        for line in (FIXTURES / "initiative_retrieval_objects.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    second_object = next(item for item in all_objects if item["object_id"] != first_object["object_id"])
    batch = build_batch_relation_request([
        {"pair_id": "positive-candidate", "proposition": proposition, "statement": statement, "official_object": first_object},
        {"pair_id": "hard-negative-candidate", "proposition": proposition, "statement": statement, "official_object": second_object},
    ])

    assert batch["pair_ids"] == ["positive-candidate", "hard-negative-candidate"]
    assert batch["model_input"]["shared_statement"] is not None
    assert all(pair["coverage"]["shared_statement_context"] for pair in batch["model_input"]["pairs"])
    assert all("statement" not in pair["input"] for pair in batch["model_input"]["pairs"])
    assert batch["output_schema"]["properties"]["pairs"]["items"]["additionalProperties"] is False

    responses = []
    for pair_id, request in batch["validation_requests"].items():
        responses.append({"pair_id": pair_id, **_response(request, status="SAME_POLICY_OBJECT" if pair_id == "positive-candidate" else "REJECTED")})
    checked = validate_batch_relation_response({"pairs": responses}, batch)
    assert checked["valid"] is True
    assert {item["pair_id"] for item in checked["results"]} == set(batch["pair_ids"])

    with pytest.raises(ValueError, match="maximum"):
        build_batch_relation_request([{
            "pair_id": str(index),
            "proposition": proposition,
            "statement": statement,
            "official_object": first_object,
        } for index in range(9)])


def test_verifier_batch_shares_statement_but_keeps_pair_specific_validation():
    _case, proposition, statement, obj = _real_case()
    request = build_relation_request(proposition, statement, obj)
    proposal = _response(request)
    proposal_result = validate_relation_response(proposal, request)
    assert proposal_result["valid"] is True
    verification_batch = build_batch_verification_request([
        {"pair_id": "review-a", "proposal": proposal_result["proposal"], "request": request},
        {"pair_id": "review-b", "proposal": proposal_result["proposal"], "request": request},
    ])
    assert verification_batch["model_input"]["shared_statement"] is not None
    assert all(pair["coverage"]["shared_statement_context"] for pair in verification_batch["model_input"]["pairs"])
    assert all(
        "statement" not in pair["input"]["source_relation_request"]
        for pair in verification_batch["model_input"]["pairs"]
    )

    responses = []
    for pair_id in verification_batch["validation_requests"]:
        responses.append({
            "pair_id": pair_id,
            **proposal,
            "verdict": "SUPPORTED_WITHIN_SCOPE",
            "counterevidence_ids": [],
        })
    checked = validate_batch_verification_response({"pairs": responses}, verification_batch)
    assert checked["valid"] is True
    assert all(item["independence"]["status"] == "NOT_INDEPENDENT" for item in checked["results"])


def test_relation_batch_does_not_share_mixed_propositions_from_one_statement():
    proposition_a, proposition_b, statement, first_object = _mixed_proposition_case()
    batch = build_batch_relation_request([
        {"pair_id": "mixed-a", "proposition": proposition_a, "statement": statement, "official_object": first_object},
        {"pair_id": "mixed-b", "proposition": proposition_b, "statement": statement, "official_object": first_object},
    ])

    assert batch["model_input"]["shared_statement"] is None
    assert all(pair["coverage"]["shared_statement_context"] is False for pair in batch["model_input"]["pairs"])
    assert all("statement" in pair["input"] for pair in batch["model_input"]["pairs"])
    assert batch["model_input"]["pairs"][0]["input"]["statement"]["proposition_text"] != batch["model_input"]["pairs"][1]["input"]["statement"]["proposition_text"]


def test_verifier_batch_does_not_share_mixed_propositions_from_one_statement():
    proposition_a, proposition_b, statement, first_object = _mixed_proposition_case()
    request_a = build_relation_request(proposition_a, statement, first_object)
    request_b = build_relation_request(proposition_b, statement, first_object)
    proposal_a = validate_relation_response(_response(request_a), request_a)
    proposal_b = validate_relation_response(_response(request_b), request_b)
    assert proposal_a["valid"] is True
    assert proposal_b["valid"] is True

    batch = build_batch_verification_request([
        {"pair_id": "mixed-review-a", "proposal": proposal_a["proposal"], "request": request_a},
        {"pair_id": "mixed-review-b", "proposal": proposal_b["proposal"], "request": request_b},
    ])

    assert batch["model_input"]["shared_statement"] is None
    assert all(pair["coverage"]["shared_statement_context"] is False for pair in batch["model_input"]["pairs"])
    assert all("statement" in pair["input"]["source_relation_request"] for pair in batch["model_input"]["pairs"])


def _v3_response(request, *, status="SAME_POLICY_OBJECT", action_alignment="RELATED"):
    """Build a short response suitable for relation_v3 adversarial cases."""

    obj = request["model_input"]["official_object"]
    prop = request["model_input"]["proposition"]
    return {
        "status": status,
        "matter_id": obj["matter_id"],
        "identity_basis": "EXPLICIT_MATTER_ID" if status.startswith("SAME_") else "UNRESOLVED",
        "statement_quote": prop["source_text"][:160],
        "object_quote": obj.get("title") or request["validation"]["object_text"][:120],
        "normalized_target": "vanhuspalvelulain hoitajamitoituksen vähimmäismäärä" if status.startswith("SAME_") else None,
        "target_scope": "PARTIAL_COMPONENT" if status.startswith("SAME_") else "UNRESOLVED",
        "bounded_claim": "The supplied sources support this scoped relation only.",
        "rationale": "The exact anchors support this bounded interpretation.",
        "action_alignment": action_alignment,
        "evidence_ids": [request["statement_evidence_ids"][0], *request["object_evidence_ids"][:1]],
        "counterevidence_ids": [],
    }


def test_relation_v3_prompt_and_schemas_are_compact_and_versioned():
    propose = load_prompt("relation_v3", stage="propose")
    verify = load_prompt("relation_v3", stage="verify")
    assert "SAME_MATTER" in propose
    assert "AUTHOR_OR_SIGNATORY_RECORD" in propose
    assert "SAME_MODEL" not in propose
    assert "counterevidence" in verify
    assert proposal_schema("relation_v3")["properties"]["statement_quote"]["maxLength"] == 160
    assert proposal_schema("relation_v3")["properties"]["rationale"]["maxLength"] == 120
    assert verification_schema("relation_v3")["properties"]["object_quote"]["maxLength"] == 160
    assert "maxLength" not in proposal_schema("relation_v2")["properties"]["statement_quote"]


def test_relation_v3_same_matter_requires_shared_formal_id_and_not_author_record():
    _case, proposition, statement, obj = _real_case()
    request = build_relation_request(proposition, statement, obj, prompt_version="relation_v3")
    response = _v3_response(request, status="SAME_MATTER")
    response["identity_basis"] = "AUTHOR_OR_SIGNATORY_RECORD"
    result = validate_relation_response(response, request)
    assert result["valid"] is False
    assert any("formal matter identifier" in error for error in result["errors"])
    assert any("AUTHOR_OR_SIGNATORY_RECORD" in error for error in result["errors"])


def test_relation_v3_accepts_same_matter_only_when_source_names_same_formal_id():
    _case, _old_prop, old_statement, obj = _real_case()
    proposition = {
        "proposition_id": "p-formal-1",
        "source_text": "Sitoumus koskee LA 72/2017 vp -esityksen hoitajamitoitusta.",
        "semantic_type": "POLICY_DESIDERATUM",
        "subject_actor_ids": ["actor-1"],
        "evidence_ids": ["statement-e"],
    }
    statement = {
        **old_statement,
        "statement_id": "s-formal-1",
        "original_text": proposition["source_text"],
        "evidence": [{"evidence_id": "statement-e"}],
    }
    request = build_relation_request(proposition, statement, obj, prompt_version="relation_v3")
    response = _v3_response(request, status="SAME_MATTER")
    result = validate_relation_response(response, request)
    assert result["valid"] is True
    assert result["proposal"]["status"] == "SAME_MATTER"
    assert request["validation"]["statement_formal_matter_ids"] == ["LA 72/2017"]
    assert "LA 72/2017" in request["validation"]["object_formal_matter_ids"]


def test_relation_v3_rejects_bare_matter_id_target_and_broad_policy_object():
    _case, proposition, statement, obj = _real_case()
    request = build_relation_request(proposition, statement, obj, prompt_version="relation_v3")
    bare = _v3_response(request)
    bare["normalized_target"] = "LA 72/2017 vp"
    result = validate_relation_response(bare, request)
    assert result["valid"] is False
    assert any("bare formal matter identifier" in error for error in result["errors"])

    broad = _v3_response(request)
    broad["target_scope"] = "BROAD_RELATED"
    result = validate_relation_response(broad, request)
    assert result["valid"] is False
    assert any("BROAD_RELATED" in error for error in result["errors"])


def test_relation_v3_broad_goal_can_be_related_but_never_action_aligned():
    _case, proposition, statement, obj = _real_case()
    request = build_relation_request(proposition, statement, obj, prompt_version="relation_v3")
    response = _v3_response(request, action_alignment="ALIGNED")
    result = validate_relation_response(response, request)
    assert result["valid"] is True
    assert result["proposal"]["action_alignment"] == "RELATED"
    assert any("broad" in warning for warning in result["warnings"])


def test_relation_v3_rejects_verbose_summary_even_if_anchor_is_valid():
    _case, proposition, statement, obj = _real_case()
    request = build_relation_request(proposition, statement, obj, prompt_version="relation_v3")
    response = _v3_response(request)
    response["rationale"] = "x" * 121
    result = validate_relation_response(response, request)
    assert result["valid"] is False
    assert any("rationale" in error and "120" in error for error in result["errors"])


def test_relation_v5_uses_source_review_identity_basis_and_minimal_wire_schema():
    propose = load_prompt("relation_v5", stage="propose")
    verify = load_prompt("relation_v5", stage="verify")
    assert "NORMALIZED_POLICY_TARGET" not in propose
    assert "REVIEWED_SOURCE_LINK" in propose
    assert "NORMALIZED_POLICY_TARGET" not in verify
    assert "REVIEWED_SOURCE_LINK" in verify
    v5 = proposal_schema("relation_v5")
    assert v5["properties"]["identity_basis"]["enum"] == [
        "EXPLICIT_TARGET_AND_MATTER",
        "REVIEWED_SOURCE_LINK",
        "UNRESOLVED",
    ]
    assert "alternative_interpretations" not in v5["properties"]
    assert "alternative_interpretations" not in batch_proposal_schema("relation_v5")["properties"]["pairs"]["items"]["properties"]


def test_relation_v5_rejects_pseudo_identity_basis_without_touching_v4():
    _case, proposition, statement, obj = _real_case()
    request = build_relation_request(proposition, statement, obj, prompt_version="relation_v5")
    response = _v3_response(request)
    response["identity_basis"] = "NORMALIZED_POLICY_TARGET"
    result = validate_relation_response(response, request)
    assert result["valid"] is False
    assert any("identity_basis" in error for error in result["errors"])

    v4_request = build_relation_request(proposition, statement, obj, prompt_version="relation_v4")
    v4_response = _v3_response(v4_request)
    v4_result = validate_relation_response(v4_response, v4_request)
    assert v4_result["valid"] is True


def test_relation_v5_positive_target_must_retain_source_anchors():
    _case, proposition, statement, obj = _real_case()
    request = build_relation_request(proposition, statement, obj, prompt_version="relation_v5")
    response = _v3_response(request)
    response["identity_basis"] = "REVIEWED_SOURCE_LINK"
    assert validate_relation_response(response, request)["valid"] is True

    response["normalized_target"] = "eläinten hyvinvointia koskeva kattava lainsäädäntö"
    result = validate_relation_response(response, request)
    assert result["valid"] is False
    assert any("grounded in the proposition source" in error for error in result["errors"])


def test_relation_v6_is_distinct_development_prompt_with_same_safe_schema():
    propose = load_prompt("relation_v6", stage="propose")
    verify = load_prompt("relation_v6", stage="verify")
    assert "bounded target inside a broad platform" in propose
    assert "NORMALIZED_POLICY_TARGET" not in propose
    assert "NORMALIZED_POLICY_TARGET" not in verify
    assert proposal_schema("relation_v6")["properties"]["identity_basis"]["enum"] == [
        "EXPLICIT_TARGET_AND_MATTER",
        "REVIEWED_SOURCE_LINK",
        "UNRESOLVED",
    ]


def test_relation_v7_is_source_first_compact_prompt_with_safe_schema():
    propose = load_prompt("relation_v7", stage="propose")
    verify = load_prompt("relation_v7", stage="verify")
    assert "Freeze that source target" in propose
    assert "object-only invention" in propose
    assert "source-first identity gate" in verify
    assert proposal_schema("relation_v7")["properties"]["identity_basis"]["enum"] == [
        "EXPLICIT_TARGET_AND_MATTER",
        "REVIEWED_SOURCE_LINK",
        "UNRESOLVED",
    ]


def test_relation_v7b_preserves_explicit_subordinate_source_targets():
    propose = load_prompt("relation_v7b", stage="propose")
    verify = load_prompt("relation_v7b", stage="verify")
    assert "hoitajavahvuus" in propose
    assert "subordinate clause" in verify
    assert proposal_schema("relation_v7b")["properties"]["identity_basis"]["enum"] == [
        "EXPLICIT_TARGET_AND_MATTER",
        "REVIEWED_SOURCE_LINK",
        "UNRESOLVED",
    ]


def test_relation_v7c_allows_bounded_partial_without_source_bill_id():
    propose = load_prompt("relation_v7c", stage="propose")
    verify = load_prompt("relation_v7c", stage="verify")
    assert "source does not need to name the bill number" in propose
    assert "PARTIAL_COMPONENT" in verify
    assert proposal_schema("relation_v7c")["properties"]["identity_basis"]["enum"] == [
        "EXPLICIT_TARGET_AND_MATTER",
        "REVIEWED_SOURCE_LINK",
        "UNRESOLVED",
    ]


def test_v5_policy_relation_without_personal_action_is_not_testable_terminal():
    _case, proposition, statement, obj = _real_case()
    request = build_relation_request(proposition, statement, obj, prompt_version="relation_v5")
    response = _v3_response(request, action_alignment="RELATED")
    response["identity_basis"] = "REVIEWED_SOURCE_LINK"
    result = validate_relation_response(response, request)
    assert result["valid"] is True
    assert result["proposal"]["identity_basis"] == "REVIEWED_SOURCE_LINK"

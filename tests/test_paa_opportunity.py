from paa.opportunity import assess_opportunity
from paa.semantics import analyze_text


def test_bill_promise_if_elected_has_no_opportunity_when_not_elected():
    text = "Teen näistä lakialoitteet tämän vuoden puolella jos pääsen eduskuntaan."
    analysis = analyze_text(text)
    types = [prop.semantic_type for prop in analysis.propositions]
    assert "PERSONAL_ACTION_COMMITMENT" in types
    assert any(prop.condition for prop in analysis.propositions)
    result = assess_opportunity(text, types, elected=False)
    assert result["state"] == "NO_OBSERVABLE_OPPORTUNITY"
    assert "lakialoitetta ei voinut jättää" in result["claim"]
    assert "ulkopuolelle" in result["claim"]


def test_same_promise_is_not_closed_when_the_person_was_elected():
    text = "Teen näistä lakialoitteet tämän vuoden puolella jos pääsen eduskuntaan."
    types = ["PERSONAL_ACTION_COMMITMENT"]
    assert assess_opportunity(text, types, elected=True) is None


def test_generic_role_resignation_uses_action_kind_instead_of_bill_language():
    text = "Mikäli valitset minut eduskuntaan, jätän paikkani valtuustossa."
    analysis = analyze_text(text)
    result = assess_opportunity(
        text,
        [prop.semantic_type for prop in analysis.propositions],
        elected=False,
    )
    assert result["state"] == "NO_OBSERVABLE_OPPORTUNITY"
    assert result["action_kind"] == "RESIGN_ROLE"
    assert "lakialoitetta" not in result["claim"]
    assert "paikan" in result["claim"]
    assert result["condition_state"] == "NOT_SATISFIED"


def test_generic_initiative_is_not_described_as_a_lakialoite():
    text = "Teen aloitteen X jos pääsen eduskuntaan."
    analysis = analyze_text(text)
    result = assess_opportunity(
        text,
        [prop.semantic_type for prop in analysis.propositions],
        elected=False,
    )
    assert result["state"] == "NO_OBSERVABLE_OPPORTUNITY"
    assert "aloitteen" in result["claim"]
    assert "lakialoitetta" not in result["claim"]


def test_unqualified_initiative_does_not_assume_parliamentary_capability():
    prop = analyze_text("Teen aloitteen X.").propositions[0]
    result = assess_opportunity(proposition=prop, role_evidence_complete=True, role_intervals=[])
    assert result is None
    from paa.opportunity import opportunity_requirements

    requirements = opportunity_requirements(prop)
    assert requirements.action_kind.value == "INITIATIVE_AUTHORED"
    assert requirements.required_capability.value == "OTHER"


def test_candidate_initiative_label_does_not_impose_a_parliamentary_seat():
    from paa.opportunity import opportunity_requirements

    for text, capability in [
        ("Teen kansalaisaloitteen koulutuksesta.", "PUBLIC_ADVOCACY"),
        ("Teen aloitteen kaupunginvaltuustossa.", "POLICYMAKING_ROLE"),
        ("Teen aloitteen koulutuksesta.", "OTHER"),
        ("Lupaan istuttaa yhden puun jokaista ääntä kohti.", "OTHER"),
        ("Teen lakialoitteen koulutuksesta.", "MP_INITIATE_BILL"),
    ]:
        prop = {"text": text, "semantic_type": "PERSONAL_ACTION_COMMITMENT",
                "action_kind": "INITIATIVE_AUTHORED", "required_capability": None}
        result = opportunity_requirements(prop)
        assert result.required_capability.value == capability


def test_broad_commitment_has_no_opportunity_plan():
    text = "Lupaan edistää arvojeni mukaista politiikkaa."
    analysis = analyze_text(text)
    assert assess_opportunity(
        text,
        [prop.semantic_type for prop in analysis.propositions],
        elected=False,
    ) is None


def test_role_interval_capability_is_returned_with_provenance():
    text = "Teen lakialoitteen X."
    analysis = analyze_text(text)
    prop = analysis.propositions[0]
    result = assess_opportunity(
        proposition=prop,
        role_intervals=[
            {
                "role_id": "role-mp-1",
                "start_date": "2023-04-05",
                "end_date": "2027-04-04",
                "formal_capabilities": [
                    {"capability": "MP_INITIATE_BILL", "evidence_ids": ["role-e1"]}
                ],
                "evidence_ids": ["role-e0"],
            }
        ],
        action_date="2024-01-01",
    )
    assert result["state"] == "OPPORTUNITY_AVAILABLE"
    assert result["required_capability"] == "MP_INITIATE_BILL"
    role_trace = next(item for item in result["provenance"] if item["kind"] == "ROLE_INTERVAL")
    assert "role-e1" in role_trace["evidence_ids"]
    assert role_trace["role_ids"] == ["role-mp-1"]


def test_election_does_not_satisfy_an_unrelated_action_condition():
    prop = {"text": "Teen lakialoitteen jos rahoitus järjestyy.",
            "semantic_type": "PERSONAL_ACTION_COMMITMENT",
            "action_kind": "INITIATIVE_AUTHORED", "required_capability": "MP_INITIATE_BILL",
            "condition": "jos rahoitus järjestyy"}
    result = assess_opportunity(proposition=prop, elected=True, role_intervals=[{
        "role_id": "mp-role", "capabilities": ["MP_INITIATE_BILL"],
    }])
    assert result["state"] == "OPPORTUNITY_AVAILABLE"
    assert result["condition_state"] == "UNRESOLVED"


def test_overlapping_mp_role_does_not_turn_lost_election_into_no_opportunity():
    """A replacement/overlapping MP can have authority despite losing the own ballot."""
    text = "Teen lakialoitteen X jos pääsen eduskuntaan."
    prop = analyze_text(text).propositions[0]
    result = assess_opportunity(
        proposition=prop,
        elected=False,
        role_intervals=[
            {
                "role_id": "role-replacement-mp",
                "start_date": "2023-04-12",
                "end_date": "2027-04-12",
                "formal_capabilities": [
                    {"capability": "MP_INITIATE_BILL", "evidence_ids": ["role-replacement-e"]}
                ],
                "evidence_ids": ["role-replacement-e"],
            }
        ],
        action_date="2024-01-01",
    )
    assert result["state"] == "OPPORTUNITY_AVAILABLE"
    # The role answers opportunity; it must not erase the separate election
    # condition, whose failure remains a reason not to admit positive action.
    assert result["condition_state"] == "NOT_SATISFIED"


def test_untyped_personal_action_does_not_invent_an_other_event():
    assert assess_opportunity(
        proposition={
            "semantic_type": "PERSONAL_ACTION_COMMITMENT",
            "source_text": "Teen jotain.",
            "action_kind": None,
            "required_capability": None,
        },
        elected=False,
    ) is None


def test_explicit_action_kind_supplies_its_canonical_capability():
    result = assess_opportunity(
        proposition={
            "semantic_type": "PERSONAL_ACTION_COMMITMENT",
            "source_text": "Jätän tehtävän, jos ehto täyttyy.",
            "action_kind": "RESIGN_ROLE",
            "required_capability": None,
        },
        role_intervals=[
            {
                "formal_capabilities": ["HOLD_ELECTED_ROLE"],
                "start_date": "2020-01-01",
                "end_date": "2030-01-01",
            }
        ],
        action_date="2024-01-01",
    )
    assert result["state"] == "OPPORTUNITY_AVAILABLE"
    assert result["required_capability"] == "HOLD_ELECTED_ROLE"


def test_missing_role_capabilities_stays_insufficient_with_provenance():
    prop = analyze_text("Teen lakialoitteen X.").propositions[0]
    result = assess_opportunity(
        proposition=prop,
        role_intervals=[
            {
                "role_id": "role-without-capability-field",
                "start_date": "2023-04-05",
                "end_date": "2027-04-04",
                "evidence_ids": ["role-e0"],
            }
        ],
        action_date="2024-01-01",
    )
    assert result["state"] == "INSUFFICIENT_EVIDENCE"
    assert any("role-e0" in item.get("evidence_ids", []) for item in result["provenance"])


def test_empty_capability_declaration_can_support_no_opportunity():
    prop = analyze_text("Teen lakialoitteen X.").propositions[0]
    result = assess_opportunity(
        proposition=prop,
        role_intervals=[
            {
                "role_id": "role-with-no-capability",
                "start_date": "2023-04-05",
                "end_date": "2027-04-04",
                "formal_capabilities": [],
                "evidence_ids": ["role-e0"],
            }
        ],
        action_date="2024-01-01",
        role_evidence_complete=True,
    )
    assert result["state"] == "NO_OBSERVABLE_OPPORTUNITY"


def test_uncertain_role_start_does_not_grant_dated_opportunity():
    prop = analyze_text("Teen lakialoitteen X.").propositions[0]
    result = assess_opportunity(
        proposition=prop,
        role_intervals=[
            {
                "role_id": "role-uncertain-start",
                "start": {
                    "earliest": None,
                    "latest": None,
                    "precision": "unknown",
                    "basis": "UNRESOLVED",
                    "timezone": "Europe/Helsinki",
                    "source_evidence_ids": ["role-e0"],
                },
                "end": {
                    "earliest": None,
                    "latest": None,
                    "precision": "unknown",
                    "basis": "UNRESOLVED",
                    "timezone": "Europe/Helsinki",
                    "source_evidence_ids": ["role-e0"],
                },
                "formal_capabilities": [{"capability": "MP_INITIATE_BILL"}],
                "evidence_ids": ["role-e0"],
            }
        ],
        action_date="2024-01-01",
    )
    assert result["state"] == "INSUFFICIENT_EVIDENCE"

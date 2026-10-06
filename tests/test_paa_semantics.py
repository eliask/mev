"""Fixture rules for statement reading. These tests do not call the network."""

from __future__ import annotations

import json
from pathlib import Path

from paa.records import proposition_record
from paa.semantics import analyze_text, evaluate_fixture

FIXTURES = Path(__file__).resolve().parents[1] / "paa" / "contracts" / "fixtures"


def _rows(name: str) -> list[dict]:
    path = FIXTURES / name
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_every_semantic_fixture_key():
    failures = []
    for row in _rows("semantic_cases.jsonl"):
        actual = evaluate_fixture(row)
        for key, expected in row["expected"].items():
            if actual.get(key) != expected:
                failures.append(f"{row['fixture_id']} {key}: expected {expected!r} got {actual.get(key)!r}")
    assert failures == []


def test_field_heading_does_not_create_a_promise():
    analysis = analyze_text("Suomi sydämessä.", {"field_label": "Vaalilupaukset"})
    assert analysis.propositions[0].semantic_type == "VALUE_OR_SLOGAN"
    assert analysis.flags["personal_action_commitment"] is False


def test_antikainen_anchor_stays_a_desideratum():
    text = "Polttoaineen hinta on saatava alaspäin ja dieselvero on poistettava."
    analysis = analyze_text(text, {"field_label": "Vaalilupaukset"})
    kinds = [prop.semantic_type for prop in analysis.propositions]
    assert kinds.count("POLICY_DESIDERATUM") >= 2
    assert all(not prop.personal_action_commitment for prop in analysis.propositions)


def test_orpo_anchor_has_two_domains_and_no_invented_metric():
    analysis = analyze_text("Laittaa Suomen talous ja koulutus kuntoon.", {"field_label": "Vaalilupaukset"})
    assert len(analysis.propositions) == 1
    assert analysis.propositions[0].semantic_type == "BROAD_OBJECTIVE"
    assert set(analysis.propositions[0].targets) >= {"talous", "koulutus"}
    assert analysis.flags["invent_metric"] is False


def test_harjanne_anchor_is_a_process_commitment():
    analysis = analyze_text(
        "Lupaan pohjata päätökseni parhaaseen mahdolliseen tietoon.",
        {"field_label": "Vaalilupaukset"},
    )
    assert analysis.propositions[0].semantic_type == "PROCESS_COMMITMENT"
    assert analysis.flags["guarantees_implementation"] is False


def test_short_denial_stays_a_position():
    analysis = analyze_text("En kannata X:n kieltämistä.")
    assert analysis.propositions[0].semantic_type == "POSITION"
    assert analysis.flags["speaker_supports_ban"] is False


def test_bare_promise_word_is_not_an_action():
    analysis = analyze_text("Lupaan.")
    assert analysis.propositions[0].personal_action_commitment is False


def test_bare_action_word_is_ambiguous_not_a_commitment():
    prop = analyze_text("Teen.").propositions[0]
    assert prop.semantic_type == "AMBIGUOUS"
    assert prop.personal_action_commitment is False


def test_colloquial_bare_action_slogan_is_not_a_commitment():
    prop = analyze_text("Teen enkä meinaa!").propositions[0]
    assert prop.semantic_type == "VALUE_OR_SLOGAN"
    assert prop.personal_action_commitment is False


def test_adverb_between_on_and_the_passive_is_still_a_desideratum():
    analysis = analyze_text("Lupaan, että vero on heti poistettava.")
    assert analysis.propositions[0].semantic_type == "POLICY_DESIDERATUM"
    assert analysis.propositions[0].personal_action_commitment is False


def test_seka_splits_two_desiderata():
    text = "Polttoaineen hinta on saatava alaspäin sekä dieselvero on poistettava."
    analysis = analyze_text(text)
    assert sum(prop.semantic_type == "POLICY_DESIDERATUM" for prop in analysis.propositions) >= 2


def test_quoted_promise_is_not_the_speakers():
    analysis = analyze_text('Vastustajani sanoi: "Poistan veron X." Minä en kannata sitä.')
    assert any(prop.reported_speech for prop in analysis.propositions)
    assert analysis.flags["attribute_quoted_promise_to_speaker"] is False
    assert analysis.flags["speaker_supports_repeal"] is False


def test_broad_value_policy_promise_does_not_open_action_ledger():
    analysis = analyze_text("Lupaan edistää arvojeni mukaista politiikkaa.")
    prop = analysis.propositions[0]
    assert prop.semantic_type == "BROAD_OBJECTIVE"
    assert prop.personal_action_commitment is False
    assert prop.observable_action is False
    assert prop.action_kind is None
    assert analysis.flags["observable_action_commitment"] is False


def test_negative_personal_policy_commitment_is_a_distinct_observable_kind():
    analysis = analyze_text("En leikkaa koulutuksesta.")
    prop = analysis.propositions[0]
    assert prop.semantic_type == "PERSONAL_RESTRAINT_COMMITMENT"
    assert prop.negation is True
    assert prop.personal_action_commitment is True
    assert prop.observable_action is True
    assert prop.action_kind == "POLICY_RESTRAINT"
    assert prop.required_capability == "POLICYMAKING_ROLE"


def test_conditional_resignation_has_role_action_metadata_and_source_span():
    text = (
        "Mikäli valitset minut eduskuntaan, jätän paikkani sekä aluevaltuustossa "
        "että kaupunginvaltuustossa."
    )
    prop = analyze_text(text).propositions[0]
    assert prop.semantic_type == "PERSONAL_ACTION_COMMITMENT"
    assert prop.action_kind == "RESIGN_ROLE"
    assert prop.required_capability == "HOLD_ELECTED_ROLE"
    assert prop.condition == "Mikäli valitset minut eduskuntaan"
    assert text[prop.source_start : prop.source_end] == prop.text


def test_relative_year_deadline_uses_statement_context_and_basis():
    text = "Teen näistä lakialoitteet tämän vuoden puolella jos pääsen eduskuntaan."
    contextual = analyze_text(text, {"stated_earliest": "2011-04-17"})
    prop = contextual.propositions[0]
    assert prop.deadline == "2011-12-31"
    assert prop.deadline_basis == "CONTEXT_DERIVED"
    assert contextual.flags["deadline_basis"] == "CONTEXT_DERIVED"

    without_context = analyze_text(text)
    assert without_context.propositions[0].deadline is None
    assert without_context.propositions[0].deadline_basis == "UNRESOLVED"


def test_decision_method_is_process_not_an_untyped_public_act():
    analysis = analyze_text("Teen päätökseni parhaaseen tietoon pohjautuen.")
    prop = analysis.propositions[0]
    assert prop.semantic_type == "PROCESS_COMMITMENT"
    assert prop.personal_action_commitment is False
    assert prop.action_kind is None


def test_possessive_work_phrase_is_process_not_public_act():
    prop = analyze_text("Teen työtäni ajatellen suomalaisen keskiöön.").propositions[0]
    assert prop.semantic_type == "PROCESS_COMMITMENT"
    assert prop.personal_action_commitment is False
    assert prop.action_kind is None


def test_first_person_advocacy_without_promise_is_a_position():
    analysis = analyze_text("Puolustan tasa-arvoa ja kaikkien ihmisten ihmisoikeuksia.")
    assert analysis.propositions[0].semantic_type == "POSITION"
    assert analysis.propositions[0].personal_action_commitment is False


def test_collective_negative_commitment_is_not_personal():
    analysis = analyze_text("Emme leikkaa koulutuksesta.")
    prop = analysis.propositions[0]
    assert prop.semantic_type == "COLLECTIVE_ACTION_COMMITMENT"
    assert prop.issuer_scope == "OTHER_COLLECTIVE"
    assert prop.personal_action_commitment is False
    assert prop.observable_action is False


def test_collective_negative_promise_is_not_personal():
    prop = analyze_text("Lupaamme olla leikkaamatta koulutuksesta.").propositions[0]
    assert prop.semantic_type == "COLLECTIVE_ACTION_COMMITMENT"
    assert prop.issuer_scope == "OTHER_COLLECTIVE"
    assert prop.personal_action_commitment is False


def test_plural_action_commitment_does_not_become_one_persons_act():
    prop = analyze_text("Lupaamme tehdä aloitteen X.").propositions[0]
    assert prop.semantic_type == "COLLECTIVE_ACTION_COMMITMENT"
    assert prop.issuer_scope == "OTHER_COLLECTIVE"
    assert prop.personal_action_commitment is False
    assert prop.observable_action is False


def test_common_finnish_abbreviation_does_not_break_source_proposition():
    text = "Pidän tärkeänä päästä vaikuttamaan syntyvään vanhuslakiin: siihen on kirjattava mm. hoitajavahvuus."
    analysis = analyze_text(text)
    assert len(analysis.propositions) == 1
    assert analysis.propositions[0].text == text
    assert text[analysis.propositions[0].source_start : analysis.propositions[0].source_end] == text


def test_source_span_keeps_leading_document_whitespace_offset():
    text = "  Teen lakialoitteen X."
    prop = analyze_text(text).propositions[0]
    assert text[prop.source_start : prop.source_end] == prop.text


def test_regex_classification_is_proposed_until_explicitly_reviewed():
    prop = analyze_text("Teen lakialoitteen X.").propositions[0]
    record = proposition_record(prop, "statement-1", "evidence-1", ["actor-1"], "prop-1")
    assert record["validation_state"] == "PROPOSED"
    reviewed = proposition_record(
        prop,
        "statement-1",
        "evidence-1",
        ["actor-1"],
        "prop-1",
        validation_state="REVIEWED",
    )
    assert reviewed["validation_state"] == "REVIEWED"

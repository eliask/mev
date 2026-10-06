from __future__ import annotations

import json

from paa.llm_semantics import (
    BATCH_RESPONSE_SCHEMA,
    BATCH_SCHEMA_VERSION,
    EXTRACTION_SCHEMA_VERSION,
    MULTI_BATCH_SCHEMA_VERSION,
    batch_response_schema,
    build_batch_request,
    build_multi_batch_request,
    build_request,
    build_verification_request,
    needs_stage2_verification,
    normalize_batch_output,
    normalize_multi_batch_output,
    normalize_output,
    proposition_records,
)
from paa.records import validate
from paa.semantics import Proposition


def _document(text: str = "Teen näistä lakialoitteet tämän vuoden puolella jos pääsen eduskuntaan.") -> dict:
    return {
        "document_id": "yle2011-test-1",
        "source_id": "SRC-YLE-2011",
        "language": "fi",
        "field_label": "Vaalilupaukset",
        "stated_earliest": "2011-04-17",
        "text": text,
    }


def _valid_output(document: dict | None = None) -> dict:
    document = document or _document()
    text = document["text"]
    start = text.index("Teen") if "Teen" in text else 0
    end = len(text)
    return {
        "schema_version": EXTRACTION_SCHEMA_VERSION,
        "document_id": document["document_id"],
        "source_id": document["source_id"],
        "abstain": False,
        "abstention_reason": None,
        "coverage": {"status": "COMPLETE", "note": "one proposition"},
        "propositions": [
            {
                "source_quote": text[start:end],
                "source_start": start,
                "source_end": end,
                "semantic_type": "PERSONAL_ACTION_COMMITMENT",
                "target_quote": "näistä",
                "condition_quote": "jos pääsen eduskuntaan",
                "condition_inherited": False,
                "deadline_quote": "tämän vuoden puolella",
                "deadline_normalized": "2011-12-31",
                "deadline_basis": "CONTEXT_DERIVED",
                "negation": False,
                "issuer_scope": "SELF",
                "action_kind": "INITIATIVE_AUTHORED",
                "required_capability": "MP_INITIATE_BILL",
                "observable_action": True,
            }
        ],
    }


def test_build_request_is_source_bound_and_does_not_call_a_model():
    request = build_request(_document())
    user = json.loads(request["messages"][1]["content"])
    assert request["prompt_version"] == "extract_v1"
    assert user["document"]["context_year"] == 2011
    assert user["document"]["text"] == _document()["text"]
    assert request["temperature"] == 0
    assert request["response_format"] == {"type": "json_object"}


def test_valid_output_becomes_proposed_compatible_proposition():
    document = _document()
    result = normalize_output(_valid_output(document), document)
    assert result.status == "VALID"
    assert len(result.propositions) == 1
    prop = result.propositions[0]
    assert isinstance(prop, Proposition)
    assert prop.deadline == "2011-12-31"
    assert prop.deadline_basis == "CONTEXT_DERIVED"
    assert prop.validation_state == "PROPOSED"
    assert prop.text == document["text"]
    records = proposition_records(
        result,
        statement_id=document["document_id"],
        evidence_id="e-1",
        actor_ids=["actor-1"],
        id_prefix=document["document_id"],
    )
    validate("proposition", records[0])


def test_exact_source_anchors_and_spans_are_required():
    document = _document()
    output = _valid_output(document)
    output["propositions"][0]["source_end"] -= 1
    result = normalize_output(output, document)
    assert result.status == "INVALID"
    assert any(item.code == "SOURCE_SPAN_MISMATCH" for item in result.invalid_records)


def test_model_cannot_invent_deadline_or_use_current_year():
    document = _document("Teen lakialoitteen X.")
    output = _valid_output(document)
    row = output["propositions"][0]
    row.update(
        source_quote=document["text"],
        source_start=0,
        source_end=len(document["text"]),
        target_quote="X",
        condition_quote=None,
        condition_inherited=False,
        deadline_quote="tämän vuoden puolella",
        deadline_normalized="2026-12-31",
        deadline_basis="CONTEXT_DERIVED",
    )
    result = normalize_output(output, document)
    assert result.status == "INVALID"
    assert any(item.code == "ANCHOR_NOT_IN_SOURCE" for item in result.invalid_records)


def test_relative_deadline_without_context_year_is_rejected_not_invented():
    document = _document("Teen lakialoitteen tämän vuoden puolella.")
    document.pop("stated_earliest")
    output = _valid_output(document)
    row = output["propositions"][0]
    row.update(
        source_quote=document["text"],
        source_start=0,
        source_end=len(document["text"]),
        target_quote="lakialoitteen",
        condition_quote=None,
        condition_inherited=False,
        deadline_quote="tämän vuoden puolella",
        deadline_normalized=None,
        deadline_basis="UNRESOLVED",
    )
    result = normalize_output(output, document)
    assert result.status == "INVALID"
    assert any(item.code == "UNSUPPORTED_DEADLINE_ANCHOR" for item in result.invalid_records)


def test_adjacent_inherited_condition_is_preserved_but_distant_one_is_not():
    text = "Mikäli valitset minut eduskuntaan, Teen aloitteen X."
    document = _document(text)
    output = _valid_output(document)
    row = output["propositions"][0]
    row.update(
        source_quote="Teen aloitteen X.",
        source_start=text.index("Teen"),
        source_end=len(text),
        target_quote="X",
        condition_quote="Mikäli valitset minut eduskuntaan",
        condition_inherited=True,
        deadline_quote=None,
        deadline_normalized=None,
        deadline_basis="UNRESOLVED",
        action_kind="INITIATIVE_AUTHORED",
        required_capability="OTHER",
    )
    result = normalize_output(output, document)
    assert result.status == "VALID"
    assert result.propositions[0].condition == row["condition_quote"]

    row["condition_quote"] = "Jos kaikki muuttuu ensi vaalikaudella"
    result = normalize_output(output, document)
    assert result.status == "INVALID"
    assert any(item.code == "ANCHOR_NOT_IN_SOURCE" for item in result.invalid_records)


def test_empty_response_requires_explicit_abstention_and_coverage():
    document = _document()
    empty = {
        "schema_version": EXTRACTION_SCHEMA_VERSION,
        "document_id": document["document_id"],
        "source_id": document["source_id"],
        "abstain": False,
        "abstention_reason": None,
        "coverage": {"status": "COMPLETE", "note": ""},
        "propositions": [],
    }
    invalid = normalize_output(empty, document)
    assert invalid.status == "INVALID"
    assert any(item.code == "EMPTY_WITHOUT_ABSTENTION" for item in invalid.invalid_records)

    empty["abstain"] = True
    empty["abstention_reason"] = "Teksti ei ole riittävän selkeä."
    empty["coverage"] = {"status": "ABSTAIN", "note": "unclear"}
    abstained = normalize_output(empty, document)
    assert abstained.status == "ABSTAIN"
    assert abstained.abstention["source"] == "MODEL_EXPLICIT"
    assert abstained.coverage["computed_status"] == "NONE"


def test_negative_and_action_capability_conflicts_are_visible():
    document = _document("En leikkaa koulutuksesta.")
    output = _valid_output(document)
    row = output["propositions"][0]
    row.update(
        source_quote=document["text"],
        source_start=0,
        source_end=len(document["text"]),
        semantic_type="PERSONAL_RESTRAINT_COMMITMENT",
        target_quote="koulutuksesta",
        condition_quote=None,
        condition_inherited=False,
        deadline_quote=None,
        deadline_normalized=None,
        deadline_basis="UNRESOLVED",
        negation=False,
        action_kind="POLICY_RESTRAINT",
        required_capability="MP_INITIATE_BILL",
        observable_action=True,
    )
    result = normalize_output(output, document)
    assert result.status == "INVALID"
    codes = {item.code for item in result.invalid_records}
    assert "NEGATION_NOT_SOURCE_GROUNDED" in codes
    assert "ACTION_CAPABILITY_MISMATCH" in codes


def test_plural_source_cannot_be_assigned_to_one_person():
    document = _document("Lupaamme tehdä aloitteen X.")
    output = _valid_output(document)
    row = output["propositions"][0]
    row.update(
        source_quote=document["text"],
        source_start=0,
        source_end=len(document["text"]),
        semantic_type="COLLECTIVE_ACTION_COMMITMENT",
        target_quote="X",
        condition_quote=None,
        condition_inherited=False,
        deadline_quote=None,
        deadline_normalized=None,
        deadline_basis="UNRESOLVED",
        issuer_scope="SELF",
        action_kind=None,
        required_capability=None,
        observable_action=False,
    )
    result = normalize_output(output, document)
    assert result.status == "INVALID"
    assert any(item.code == "PLURAL_SCOPE_CONFLICT" for item in result.invalid_records)


def test_partial_output_does_not_silently_drop_invalid_rows():
    document = _document()
    output = _valid_output(document)
    bad = dict(output["propositions"][0])
    bad["source_quote"] = "not in source"
    bad["source_start"] = 0
    bad["source_end"] = len(bad["source_quote"])
    output["propositions"].append(bad)
    output["coverage"] = {"status": "PARTIAL", "note": "one row rejected"}
    result = normalize_output(output, document)
    assert result.status == "PARTIAL"
    assert len(result.propositions) == 1
    assert any(item.index == 1 and item.code == "SOURCE_SPAN_MISMATCH" for item in result.invalid_records)
    assert result.coverage["invalid_count"] >= 1


def test_stage2_is_requested_for_narrow_negative_or_disagreement():
    result = normalize_output(_valid_output(), _document())
    prop = result.propositions[0]
    assert needs_stage2_verification(prop)
    request = build_verification_request(_document(), prop)
    payload = json.loads(request["messages"][1]["content"])
    assert payload["contract"] == "paa.extract.verify.v1"
    assert needs_stage2_verification({"semantic_type": "POSITION"}, disagreement=True)


def _batch_document() -> tuple[dict, list[dict]]:
    text = (
        "Mikäli valitset minut eduskuntaan, "
        "Teen lakialoitteen X tämän vuoden puolella. "
        "En leikkaa koulutuksesta."
    )
    condition_end = text.index(",")
    action_start = text.index("Teen")
    action_end = text.index(".", action_start) + 1
    restraint_start = text.index("En leikkaa")
    return (
        {
            "document_id": "batch-doc-1",
            "source_id": "SRC-YLE-2011",
            "stated_earliest": "2011-04-17",
            "question": "Mitä lupasit tehdä?",
            "text": text,
        },
        [
            {"unit_id": "u-condition", "text": text[:condition_end], "start": 0, "end": condition_end},
            {
                "unit_id": "u-action",
                "text": text[action_start:action_end],
                "start": action_start,
                "end": action_end,
            },
            {
                "unit_id": "u-restraint",
                "text": text[restraint_start:],
                "start": restraint_start,
                "end": len(text),
            },
        ],
    )


def _valid_batch_output(document: dict, units: list[dict]) -> dict:
    return {
        "schema_version": BATCH_SCHEMA_VERSION,
        "document_id": document["document_id"],
        "source_id": document["source_id"],
        "abstain": False,
        "abstention_reason": None,
        "coverage": {"status": "COMPLETE"},
        "rows": [
            ["u-condition", "AM", "U", False, None, None, None, None],
            [
                "u-action",
                "PA",
                "S",
                False,
                "IA",
                "lakialoitteen X",
                "Mikäli valitset minut eduskuntaan",
                "tämän vuoden puolella",
            ],
            ["u-restraint", "NR", "S", True, "RT", "koulutuksesta", None, None],
        ],
    }


def test_batch_request_contains_full_source_and_unlabeled_caller_units():
    document, units = _batch_document()
    request = build_batch_request(document, units)
    payload = json.loads(request["messages"][1]["content"])
    assert request["prompt_version"] == "extract_batch_v1"
    assert payload["contract"] == BATCH_SCHEMA_VERSION
    assert payload["document"]["text"] == document["text"]
    assert payload["document"]["question"] == document["question"]
    assert payload["units"] == units
    assert all("semantic_type" not in unit for unit in payload["units"])
    assert request["temperature"] == 0
    assert request["max_tokens"] >= 30 * len(units)


def test_batch_rows_cover_units_and_preserve_source_spans_and_context_date():
    document, units = _batch_document()
    result = normalize_batch_output(_valid_batch_output(document, units), document, units)
    assert result.status == "VALID"
    assert [item.unit_id for item in result.units] == ["u-condition", "u-action", "u-restraint"]
    action = result.units[1].proposition
    assert action is not None
    assert action.source_start == units[1]["start"]
    assert action.source_end == units[1]["end"]
    assert action.deadline == "2011-12-31"
    assert action.deadline_basis == "CONTEXT_DERIVED"
    assert action.action_kind == "INITIATIVE_AUTHORED"
    assert action.required_capability is None
    assert action.condition == "Mikäli valitset minut eduskuntaan"
    assert result.units[1].condition_inherited is True
    restraint = result.units[2].proposition
    assert restraint is not None
    assert restraint.semantic_type == "PERSONAL_RESTRAINT_COMMITMENT"
    assert restraint.action_kind == "POLICY_RESTRAINT"
    assert restraint.negation is True


def test_batch_missing_rows_are_visible_not_silently_dropped():
    document, units = _batch_document()
    output = _valid_batch_output(document, units)
    output["rows"] = output["rows"][:1]
    output["coverage"] = {"status": "PARTIAL"}
    result = normalize_batch_output(output, document, units)
    assert result.status == "PARTIAL"
    assert len(result.units) == len(units)
    assert sum(item.status == "INVALID" for item in result.units) == 2
    assert sum(item.code == "MISSING_UNIT_ROW" for item in result.invalid_records) == 2


def test_batch_rejects_action_on_broad_process_and_distant_condition():
    document, units = _batch_document()
    output = _valid_batch_output(document, units)
    output["rows"][1][2:] = ["S", False, "OA", "lakialoitteen X", None, None]
    # This row is now an action on a personal action type, so it remains
    # valid; use a broad type to assert the actual gate.
    output["rows"][1][1] = "PR"
    output["rows"][1][4] = "OA"
    output["rows"][1][6] = None
    output["rows"][1][7] = None
    result = normalize_batch_output(output, document, units)
    assert result.status == "PARTIAL"
    assert any(item.code == "ACTION_ON_NON_ACTION_TYPE" for item in result.invalid_records)

    text = "Mikäli valitset minut eduskuntaan, Sama teksti. Teen aloitteen X."
    remote_document = {**document, "document_id": "batch-remote", "text": text}
    first_end = text.index(",")
    second_start = text.index("Sama")
    second_end = text.index(".", second_start) + 1
    third_start = text.index("Teen")
    remote_units = [
        {"unit_id": "c", "text": text[:first_end], "start": 0, "end": first_end},
        {"unit_id": "middle", "text": text[second_start:second_end], "start": second_start, "end": second_end},
        {"unit_id": "act", "text": text[third_start:], "start": third_start, "end": len(text)},
    ]
    remote_output = {
        "schema_version": BATCH_SCHEMA_VERSION,
        "document_id": remote_document["document_id"],
        "source_id": remote_document["source_id"],
        "abstain": False,
        "abstention_reason": None,
        "coverage": {"status": "PARTIAL"},
        "rows": [
            ["c", "AM", "U", False, None, None, None, None],
            ["middle", "AM", "U", False, None, None, None, None],
            ["act", "PA", "S", False, "IA", "aloitteen X", "Mikäli valitset minut eduskuntaan", None],
        ],
    }
    remote_result = normalize_batch_output(remote_output, remote_document, remote_units)
    assert remote_result.status == "PARTIAL"
    assert any(item.code == "CONDITION_NOT_LOCAL" for item in remote_result.invalid_records)


def test_batch_explicit_ambiguity_and_abstention_are_distinct():
    document, units = _batch_document()
    output = _valid_batch_output(document, units)
    output["rows"][0] = ["u-condition", "AM", "U", False, None, None, None, None]
    result = normalize_batch_output(output, document, units)
    assert result.units[0].status == "VALID"
    assert result.units[0].proposition.semantic_type == "AMBIGUOUS"

    abstain = {
        "schema_version": BATCH_SCHEMA_VERSION,
        "document_id": document["document_id"],
        "source_id": document["source_id"],
        "abstain": True,
        "abstention_reason": "Batch context was unreadable.",
        "coverage": {"status": "ABSTAIN"},
        "rows": [],
    }
    abstained = normalize_batch_output(abstain, document, units)
    assert abstained.status == "ABSTAIN"
    assert abstained.coverage["computed_status"] == "ABSTAIN"
    assert all(item.status == "ABSTAIN" for item in abstained.units)


def test_batch_relative_deadline_without_source_year_is_rejected():
    document, units = _batch_document()
    document = {key: value for key, value in document.items() if key != "stated_earliest"}
    output = _valid_batch_output(document, units)
    result = normalize_batch_output(output, document, units)
    assert result.status == "PARTIAL"
    assert any(item.code == "UNSUPPORTED_DEADLINE_ANCHOR" for item in result.invalid_records)


def test_batch_swedish_negation_and_relative_deadline_are_language_aware():
    text = "En fråga behandlas i år. Jag gör inte detta."
    document = {
        "document_id": "batch-sv-1",
        "source_id": "SRC-SV-2024",
        "language": "sv",
        "stated_earliest": "2024-01-01",
        "text": text,
    }
    first_end = text.index(".") + 1
    units = [
        {"unit_id": "sv-1", "text": text[:first_end], "start": 0, "end": first_end},
        {"unit_id": "sv-2", "text": text[first_end + 1 :], "start": first_end + 1, "end": len(text)},
    ]
    output = {
        "schema_version": BATCH_SCHEMA_VERSION,
        "document_id": document["document_id"],
        "source_id": document["source_id"],
        "abstain": False,
        "abstention_reason": None,
        "coverage": {"status": "COMPLETE"},
        "rows": [
            ["sv-1", "FC", "U", False, None, None, None, "i år"],
            ["sv-2", "PA", "S", True, "OA", "detta", None, None],
        ],
    }
    result = normalize_batch_output(output, document, units)
    assert result.status == "VALID"
    assert result.units[0].proposition.deadline == "2024-12-31"
    assert result.units[0].proposition.deadline_basis == "CONTEXT_DERIVED"
    assert result.units[0].proposition.negation is False
    assert result.units[1].proposition.negation is True


def test_batch_response_schema_is_typed_and_copy_safe():
    schema = batch_response_schema()
    assert schema == BATCH_RESPONSE_SCHEMA
    schema["properties"]["rows"]["items"]["prefixItems"][0]["type"] = "integer"
    assert BATCH_RESPONSE_SCHEMA["properties"]["rows"]["items"]["prefixItems"][0]["type"] == "string"
    assert schema["properties"]["rows"]["items"]["prefixItems"][1]["enum"]
    assert schema["properties"]["rows"]["items"]["prefixItems"][4]["enum"][-1] is None


def test_multi_batch_uses_short_wire_ids_and_normalizes_back_to_canonical_units():
    document, units = _batch_document()
    swedish_text = "En fråga behandlas i år."
    swedish_document = {
        "document_id": "batch-sv-multi",
        "source_id": "SRC-SV-2024",
        "language": "sv",
        "stated_earliest": "2024-01-01",
        "text": swedish_text,
    }
    swedish_units = [{"unit_id": "long-canonical-sv-id", "text": swedish_text, "start": 0, "end": len(swedish_text)}]
    specs = [{"document": document, "units": units}, {"document": swedish_document, "units": swedish_units}]
    request = build_multi_batch_request(specs)
    payload = json.loads(request["messages"][1]["content"])
    assert payload["contract"] == MULTI_BATCH_SCHEMA_VERSION
    assert payload["documents"][0]["units"][0]["unit_id"] == "1"
    assert payload["documents"][0]["units"][1]["unit_id"] == "2"
    assert payload["documents"][1]["units"][0]["unit_id"] == "1"
    assert request["metadata"]["unit_maps"]["batch-sv-multi"]["1"] == "long-canonical-sv-id"

    first_output = _valid_batch_output(document, units)
    wire_first_rows = [row[:] for row in first_output["rows"]]
    for row in wire_first_rows:
        row[0] = str(int(units.index(next(unit for unit in units if unit["unit_id"] == row[0]))) + 1)
    multi_output = {
        "schema_version": MULTI_BATCH_SCHEMA_VERSION,
        "documents": [
            {
                "document_id": document["document_id"],
                "source_id": document["source_id"],
                "abstain": False,
                "abstention_reason": None,
                "coverage": {"status": "COMPLETE"},
                "rows": wire_first_rows,
            },
            {
                "document_id": swedish_document["document_id"],
                "source_id": swedish_document["source_id"],
                "abstain": False,
                "abstention_reason": None,
                "coverage": {"status": "COMPLETE"},
                "rows": [["1", "FC", "U", False, None, None, None, "i år"]],
            },
        ],
    }
    result = normalize_multi_batch_output(multi_output, specs)
    assert result.status == "VALID"
    assert [item.unit_id for item in result.documents[0].units] == [unit["unit_id"] for unit in units]
    assert result.documents[1].units[0].proposition.deadline == "2024-12-31"


def test_multi_batch_missing_document_is_visible():
    document, units = _batch_document()
    specs = [{"document": document, "units": units}]
    raw = {
        "schema_version": MULTI_BATCH_SCHEMA_VERSION,
        "documents": [],
    }
    result = normalize_multi_batch_output(raw, specs)
    assert result.status == "INVALID"
    assert any(item.code == "MISSING_DOCUMENT_RESULT" for item in result.invalid_records)

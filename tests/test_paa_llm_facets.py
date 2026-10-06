import json

from paa.llm_facets import (
    FACET_PROMPT_VERSION,
    FACET_SCHEMA_VERSION,
    FacetKind,
    FacetResultStatus,
    FacetUnitStatus,
    PublicationState,
    build_facet_request,
    normalize_facet_output,
    research_packet,
    source_document,
    source_only_fixture_row,
    source_units,
)


def _document(text: str) -> dict[str, object]:
    return {
        "document_id": "facet-doc-1",
        "source_id": "SRC-TEST",
        "language": "fi",
        "question": "Vaalilupaukset",
        "source_year": 2023,
        "source_text": text,
    }


def _anchor(text: str, quote: str, start_at: int = 0) -> dict[str, object]:
    start = text.index(quote, start_at)
    return {"quote": quote, "start": start, "end": start + len(quote)}


def _facet(
    text: str,
    quote: str,
    *,
    facet_id: str,
    kind: str,
    start_at: int = 0,
    negation: bool = False,
    horizon: str | None = None,
    horizon_kind: str = "UNRESOLVED",
    action_kind: str | None = None,
    target: str | None = None,
    condition: dict[str, object] | None = None,
    links: list[str] | None = None,
) -> dict[str, object]:
    source = _anchor(text, quote, start_at)
    return {
        "facet_id": facet_id,
        "kind": kind,
        "source_quote": quote,
        "source_start": source["start"],
        "source_end": source["end"],
        "actor": None,
        "carrier": None,
        "predicate": None,
        "target": _anchor(text, target, start_at) if target else None,
        "population": None,
        "magnitude": None,
        "instrument": None,
        "condition": condition,
        "horizon": _anchor(text, horizon, start_at) if horizon else None,
        "horizon_kind": horizon_kind,
        "actor_scope": "SELF",
        "negation": negation,
        "action_kind": action_kind,
        "linked_subproposition_ids": links or [],
    }


def _output(document: dict[str, object], units: list[dict[str, object]], rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "schema_version": FACET_SCHEMA_VERSION,
        "document_id": document["document_id"],
        "source_id": document["source_id"],
        "abstain": False,
        "abstention_reason": None,
        "coverage": {"status": "COMPLETE"},
        "units": rows,
    }


def test_build_request_exposes_source_only_fields_and_full_context() -> None:
    text = "Aloitan työn."
    document = _document(text)
    document["gold"] = {"primary_type": "SECRET"}
    document["selection"] = {"stratum": "SECRET"}
    units = [{"unit_id": "u1", "text": text, "start": 0, "end": len(text)}]
    request = build_facet_request(document, units)
    payload = json.loads(request["messages"][1]["content"])
    assert request["prompt_version"] == FACET_PROMPT_VERSION
    assert payload["source_document"]["source_text"] == text
    assert "gold" not in payload["source_document"]
    assert "selection" not in payload["source_document"]
    assert payload["units"] == units
    assert payload["contract"] == FACET_SCHEMA_VERSION
    assert request["metadata"]["admission_state"] == "PROPOSED"


def test_two_linked_facets_preserve_exact_subspans() -> None:
    text = "Kaupunginvaltuutettuna tein aloitteen X. Ensi vuonna teen Y."
    document = _document(text)
    units = [{"unit_id": "u1", "text": text, "start": 0, "end": len(text)}]
    first_quote = "Kaupunginvaltuutettuna tein aloitteen X."
    second_quote = "Ensi vuonna teen Y."
    raw = _output(
        document,
        units,
        [
            {
                "unit_id": "u1",
                "status": "COMPLETE",
                "abstention_reason": None,
                "facets": [
                    _facet(text, first_quote, facet_id="f1", kind="PAST_FACT", action_kind="INITIATIVE_AUTHORED", target="X", links=["f2"]),
                    _facet(text, second_quote, facet_id="f2", kind="FUTURE_COMMITMENT", horizon="Ensi vuonna", horizon_kind="FUTURE", target="Y", start_at=text.index("Ensi"), links=["f1"]),
                ],
            }
        ],
    )
    result = normalize_facet_output(raw, document, units)
    assert result.status is FacetResultStatus.VALID
    assert len(result.facets) == 2
    assert result.facets[0].source_quote == first_quote
    assert result.facets[1].horizon.quote == "Ensi vuonna"
    assert result.facets[1].linked_subproposition_ids == ("f1",)
    assert all(facet.validation_state is PublicationState.PROPOSED for facet in result.facets)


def test_compact_quote_protocol_binds_absolute_offsets_deterministically() -> None:
    text = "Teen aloitteen X."
    document = _document(text)
    units = [{"unit_id": "u1", "text": text, "start": 0, "end": len(text)}]
    compact_facet = {
        "facet_id": "f1",
        "kind": "FUTURE_COMMITMENT",
        "source_quote": text,
        "source_occurrence": 0,
        "actor": None,
        "carrier": None,
        "predicate": "Teen",
        "target": "X",
        "population": None,
        "magnitude": None,
        "instrument": "aloitteen",
        "condition": None,
        "horizon": None,
        "horizon_kind": "FUTURE",
        "actor_scope": "SELF",
        "negation": False,
        "action_kind": "LAW_INITIATIVE",
        "linked_subproposition_ids": [],
    }
    raw = _output(document, units, [{"unit_id": "u1", "status": "COMPLETE", "abstention_reason": None, "facets": [compact_facet]}])
    result = normalize_facet_output(raw, document, units, prompt_version="facets_v2_linked_propositions")
    assert result.status is FacetResultStatus.VALID
    assert result.facets[0].source_start == 0
    assert result.facets[0].target.start == text.index("X")
    assert result.facets[0].action_kind.value == "LAW_INITIATIVE"


def test_past_and_current_facts_are_not_future_commitments() -> None:
    text = "Vuonna 2020 tein aloitteen X. Nyt puolustan palvelua."
    document = _document(text)
    units = [{"unit_id": "u1", "text": text, "start": 0, "end": len(text)}]
    raw = _output(
        document,
        units,
        [
            {
                "unit_id": "u1",
                "status": "COMPLETE",
                "abstention_reason": None,
                "facets": [
                    _facet(text, "Vuonna 2020 tein aloitteen X.", facet_id="past", kind="PAST_FACT", horizon="Vuonna 2020", horizon_kind="PAST", action_kind="INITIATIVE_AUTHORED", target="X"),
                    _facet(text, "Nyt puolustan palvelua.", facet_id="now", kind="CURRENT_FACT", horizon="Nyt", horizon_kind="CURRENT", target="palvelua", start_at=text.index("Nyt")),
                ],
            }
        ],
    )
    result = normalize_facet_output(raw, document, units)
    assert result.status is FacetResultStatus.VALID
    assert [facet.kind for facet in result.facets] == [FacetKind.PAST_FACT, FacetKind.CURRENT_FACT]


def test_broad_policy_goal_cannot_carry_initiative_action() -> None:
    text = "Tavoittelen parempaa koulutusta kaikille."
    document = _document(text)
    units = [{"unit_id": "u1", "text": text, "start": 0, "end": len(text)}]
    raw = _output(
        document,
        units,
        [
            {
                "unit_id": "u1",
                "status": "COMPLETE",
                "abstention_reason": None,
                "facets": [_facet(text, text, facet_id="goal", kind="POLICY_GOAL", action_kind="INITIATIVE_AUTHORED", target="koulutusta")],
            }
        ],
    )
    result = normalize_facet_output(raw, document, units)
    assert result.status is FacetResultStatus.INVALID
    assert any(error.code.value == "BROAD_GOAL_ACTION_KIND_FORBIDDEN" for error in result.invalid_records)


def test_source_offsets_and_nested_anchors_are_checked() -> None:
    text = "Teen aloitteen X."
    document = _document(text)
    units = [{"unit_id": "u1", "text": text, "start": 0, "end": len(text)}]
    facet = _facet(text, text, facet_id="f1", kind="FUTURE_COMMITMENT", target="X")
    facet["source_end"] = int(facet["source_end"]) - 1
    raw = _output(document, units, [{"unit_id": "u1", "status": "COMPLETE", "abstention_reason": None, "facets": [facet]}])
    result = normalize_facet_output(raw, document, units)
    assert result.status is FacetResultStatus.INVALID
    assert any(error.code.value == "SOURCE_SPAN_MISMATCH" for error in result.invalid_records)

    facet = _facet(text, text, facet_id="f1", kind="FUTURE_COMMITMENT", target="X")
    facet["target"] = {"quote": "X", "start": len(text) + 1, "end": len(text) + 2}
    raw["units"][0]["facets"] = [facet]
    result = normalize_facet_output(raw, document, units)
    assert result.status is FacetResultStatus.INVALID
    assert any(error.code.value == "ANCHOR_SPAN_MISMATCH" for error in result.invalid_records)


def test_adjacent_condition_is_retained_with_absolute_offsets() -> None:
    text = "Jos minut valitaan, Teen aloitteen X."
    split = text.index("Teen")
    document = _document(text)
    units = [
        {"unit_id": "condition", "text": text[:split].rstrip(", "), "start": 0, "end": text.index(",")},
        {"unit_id": "action", "text": text[split:], "start": split, "end": len(text)},
    ]
    condition = _anchor(text, "Jos minut valitaan")
    raw = _output(
        document,
        units,
        [
            {"unit_id": "condition", "status": "ABSTAIN", "abstention_reason": "context only", "facets": []},
            {
                "unit_id": "action",
                "status": "COMPLETE",
                "abstention_reason": None,
                "facets": [_facet(text, "Teen aloitteen X.", facet_id="f1", kind="FUTURE_COMMITMENT", target="X", condition=condition, start_at=split)],
            },
        ],
    )
    result = normalize_facet_output(raw, document, units)
    assert result.status is FacetResultStatus.PARTIAL
    assert result.facets[0].condition.quote == "Jos minut valitaan"
    assert result.facets[0].condition.start == 0
    assert result.units[0].status is FacetUnitStatus.ABSTAIN


def test_missing_unit_is_explicit_invalid_not_negative() -> None:
    text = "Teen aloitteen X."
    document = _document(text)
    units = [{"unit_id": "u1", "text": text, "start": 0, "end": len(text)}]
    raw = _output(document, units, [])
    result = normalize_facet_output(raw, document, units)
    assert result.status is FacetResultStatus.INVALID
    assert result.units[0].status is FacetUnitStatus.INVALID
    assert any(error.code.value == "UNIT_ID_UNKNOWN" for error in result.invalid_records)


def test_explicit_abstention_is_distinct_from_invalid_output() -> None:
    text = "Epäselvä vastaus."
    document = _document(text)
    units = [{"unit_id": "u1", "text": text, "start": 0, "end": len(text)}]
    raw = {
        "schema_version": FACET_SCHEMA_VERSION,
        "document_id": document["document_id"],
        "source_id": document["source_id"],
        "abstain": True,
        "abstention_reason": "source does not support a stable facet",
        "coverage": {"status": "ABSTAIN"},
        "units": [],
    }
    result = normalize_facet_output(raw, document, units)
    assert result.status is FacetResultStatus.ABSTAIN
    assert result.units[0].status is FacetUnitStatus.ABSTAIN


def test_source_only_blind_loader_drops_gold_and_selection() -> None:
    row = {
        "document": {
            **_document("Teen aloitteen X."),
            "identity_context": {"actor_id": "actor-1", "stated_earliest": "2023-01-01"},
        },
        "gold": {"primary_type": "SHOULD NEVER BE SENT"},
        "selection": {"stratum": "SECRET"},
        "adjudication": {"basis": "SECRET"},
    }
    document, units = source_only_fixture_row(row)
    request = build_facet_request(document, units)
    payload = json.loads(request["messages"][1]["content"])
    assert set(payload["source_document"]) <= {
        "document_id", "source_id", "language", "question", "stated_earliest", "source_year", "source_sha256", "source_text", "url", "actor_id"
    }
    assert "gold" not in json.dumps(payload)
    assert "selection" not in json.dumps(payload)


def test_research_packet_is_source_bound_and_proposed() -> None:
    text = "Tavoittelen parempaa koulutusta."
    document = source_document(_document(text))
    units = source_units(document, [{"unit_id": "u1", "text": text, "start": 0, "end": len(text)}])
    raw = _output(_document(text), [unit.as_dict() for unit in units], [{"unit_id": "u1", "status": "COMPLETE", "abstention_reason": None, "facets": [_facet(text, text, facet_id="f1", kind="POLICY_GOAL", target="koulutusta")]}])
    extraction = normalize_facet_output(raw, document, units)
    packet = research_packet(document, units, extraction)
    assert packet.source.document_id == document.document_id
    assert packet.extraction.status is FacetResultStatus.VALID
    assert packet.as_dict()["admission_state"] == "PROPOSED"


def test_source_fixture_row_has_one_exact_whole_source_unit() -> None:
    text = "A. B."
    row = {"document": {**_document(text), "source_sha256": ""}}
    row["document"].pop("source_sha256")
    _document_value, units = source_only_fixture_row(row)
    assert units[0].text == text
    assert units[0].start == 0
    assert units[0].end == len(text)

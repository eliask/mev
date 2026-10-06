"""Offline semantic records; source-to-site acceptance is in test_paa_trace_e2e."""

from paa.records import proposition_record, statement_record, validate
from paa.semantics import analyze_text


def test_slogan_specimen_validates_and_is_not_a_personal_act():
    text = "Isänmaa sydämessä."
    analysis = analyze_text(text, {"field_label": "Vaalilupaukset"})
    prop = analysis.propositions[0]
    assert prop.semantic_type == "VALUE_OR_SLOGAN"
    assert prop.personal_action_commitment is False
    document = {
        "document_id": "specimen-purra",
        "source_id": "SRC-SPECIMEN",
        "field_label": "Vaalilupaukset",
        "language": "fi",
        "text": text,
        "stated_earliest": "2023-03-01",
        "stated_latest": "2023-04-02",
        "retrieved_at": "2026-10-06T00:00:00Z",
    }
    statement = statement_record(document, "specimen-purra-e", ["mp-specimen"], "EXPLICIT")
    proposition = proposition_record(prop, statement["statement_id"], "specimen-purra-e", ["mp-specimen"], "specimen-purra-p1")
    validate("statement", statement)
    validate("proposition", proposition)


def test_statement_without_upper_bound_does_not_invent_2023_election_day():
    document = {
        "document_id": "official-2024-statement",
        "source_id": "SRC-OFFICIAL-2024",
        "field_label": "Lausunto",
        "language": "fi",
        "text": "Vuonna 2024 julkaistu lausunto.",
        "stated_earliest": "2024-01-15",
    }

    statement = statement_record(document, "official-2024-statement-e", [], "EXPLICIT")

    assert statement["stated_at"] == {
        "earliest": "2024-01-15",
        "latest": None,
        "precision": "range",
        "basis": "CONTEXT_DERIVED",
        "timezone": "Europe/Helsinki",
        "source_evidence_ids": ["official-2024-statement-e"],
    }
    validate("statement", statement)

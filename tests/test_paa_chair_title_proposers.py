"""Chair-title named proposer parsing controls."""

from paa.chair_title_proposers import (
    finnish_person_name_keys,
    parse_chair_title_named_proposers,
    published_alternative_disposition,
)
from paa.matter_ledger import _actor_from_display_name


def test_named_lausuma_proposer_is_extracted() -> None:
    proposers = parse_chair_title_named_proposers("Lausumaehdotus, mietintö / Hilkka Kemppi")
    assert len(proposers) == 1
    assert proposers[0].source_display_name == "Hilkka Kemppi"
    assert proposers[0].side == "EI"
    assert proposers[0].slot == "EI:1"
    assert proposers[0].basis == "CHAIR_TITLE_NAMED_ALTERNATIVE"


def test_both_side_named_alternatives_are_retained() -> None:
    proposers = parse_chair_title_named_proposers(
        "Lausumaehdotus, Hanna-Leena Mattila 1 / Nina Merinen 1"
    )
    assert [(item.side, item.slot, item.source_display_name) for item in proposers] == [
        ("JAA", "JAA:1", "Hanna-Leena Mattila"),
        ("EI", "EI:1", "Nina Merinen"),
    ]


def test_preserves_nominative_names_ending_in_n() -> None:
    proposers = parse_chair_title_named_proposers(
        "Lausumaehdotus, mietintö / Bella Forsgrén 1, Ilmari Nurminen 1"
    )
    assert [item.source_display_name for item in proposers] == ["Bella Forsgrén", "Ilmari Nurminen"]
    assert "Bella Forsgré" not in {item.source_display_name for item in proposers}
    for title_name in ("Aki Lindén", "Li Andersson", "Peter Östman"):
        parsed = parse_chair_title_named_proposers(f"Lausumaehdotus, mietintö / {title_name}")
        assert parsed[0].source_display_name == title_name


def test_genitive_source_spelling_preserved_with_inflection_key() -> None:
    proposers = parse_chair_title_named_proposers(
        "Lausumaehdotus, mietintö / Ilmari Nurmisen monisteena jaettu lausumaehdotus 1"
    )
    assert len(proposers) == 1
    assert proposers[0].source_display_name == "Ilmari Nurmisen"
    keys = finnish_person_name_keys("Ilmari Nurmisen")
    assert keys[0] == "ilmari nurmisen"
    assert "ilmari nurminen" in keys
    assert parse_chair_title_named_proposers(
        "Nauttii luottamusta / Matias Mäkysen ehdotus"
    )[0].source_display_name == "Matias Mäkysen"


def test_procedural_title_is_negative_control() -> None:
    assert parse_chair_title_named_proposers("Lakiehdotusten hyväksyminen / hylkääminen") == ()
    assert parse_chair_title_named_proposers("Hyväksyminen / Hylkääminen") == ()
    assert parse_chair_title_named_proposers("Option A / Option B") == ()


def test_terminal_polarity_and_annotation_before_proposal_term() -> None:
    proposers = parse_chair_title_named_proposers(
        "Mietintö JAA / Hanna-Leena Mattilan ehdotus (vl) EI"
    )
    assert len(proposers) == 1
    assert proposers[0].source_display_name == "Hanna-Leena Mattilan"
    assert proposers[0].side == "EI"
    assert proposers[0].slot == "EI:1"
    assert proposers[0].basis == "CHAIR_TITLE_NAMED_ALTERNATIVE"
    keys = finnish_person_name_keys("Hanna-Leena Mattilan")
    assert keys[0] == "hanna-leena mattilan"
    assert "hanna-leena mattila" in keys


def test_both_polarity_markers_on_named_sides() -> None:
    proposers = parse_chair_title_named_proposers(
        "Hanna-Leena Mattila JAA / Nina Merinen EI"
    )
    assert [(item.side, item.source_display_name) for item in proposers] == [
        ("JAA", "Hanna-Leena Mattila"),
        ("EI", "Nina Merinen"),
    ]


def test_conflicting_polarity_markers_are_rejected() -> None:
    assert parse_chair_title_named_proposers(
        "Hanna-Leena Mattila EI / Nina Merinen JAA"
    ) == ()
    assert parse_chair_title_named_proposers(
        "Hanna-Leena Mattila JAA / Nina Merinen JAA"
    ) == ()


def test_annotations_cannot_hide_conflicting_or_duplicate_polarity() -> None:
    for title in (
        "Mietintö EI (vl) / Hanna-Leena Mattilan ehdotus (vl) EI",
        "Mietintö JAA / Hanna-Leena Mattilan ehdotus JAA (vl)",
        "Mietintö JAA / Hanna-Leena Mattilan ehdotus JAA (vl) EI",
        "Mietintö JAA / Hanna-Leena Mattilan ehdotus EI (vl) EI",
    ):
        assert parse_chair_title_named_proposers(title) == ()
    parsed = parse_chair_title_named_proposers(
        "Mietintö JAA (vl) / Hanna-Leena Mattilan ehdotus EI (vl)"
    )
    assert [(item.source_display_name, item.side) for item in parsed] == [
        ("Hanna-Leena Mattilan", "EI")
    ]


def test_procedural_description_is_not_a_title_case_person() -> None:
    for description in ("Asian Palauttaminen", "asian palauttaminen", "ASIAN PALAUTTAMINEN"):
        assert parse_chair_title_named_proposers(f"Mietintö JAA / {description} EI") == ()


def test_procedural_sides_with_polarity_markers_stay_empty() -> None:
    assert parse_chair_title_named_proposers("Hyväksyminen JAA / Hylkääminen EI") == ()
    assert parse_chair_title_named_proposers("Mietintö JAA / Hylkääminen EI") == ()


def test_unknown_annotation_does_not_invent_a_proposer() -> None:
    assert parse_chair_title_named_proposers(
        "Mietintö JAA / valiokunnan ehdotus (xyz) EI"
    ) == ()
    for qualifier in ("tuntematon rooli", "xyz", "VL tuntematon"):
        assert parse_chair_title_named_proposers(
            f"Mietintö JAA / Hanna-Leena Mattilan ehdotus ({qualifier}) EI"
        ) == ()
        assert parse_chair_title_named_proposers(
            f"Mietintö JAA / Hanna-Leena Mattilan ehdotus EI ({qualifier})"
        ) == ()


def test_genitive_la_stem_key_preserves_nominative_n_names() -> None:
    keys = finnish_person_name_keys("Hanna-Leena Mattilan")
    assert "hanna-leena mattila" in keys
    assert finnish_person_name_keys("Atte Harjanne") == ("atte harjanne",)
    assert finnish_person_name_keys("Nina Merinen") == ("nina merinen",)
    assert "bella forsgré" not in finnish_person_name_keys("Bella Forsgrén")


def test_roster_exact_nominative_precedes_inflection_and_collision_stays_unresolved() -> None:
    exact = {"actor_id": "exact", "person_id": "1", "display_name": "Matti Jensen"}
    fallback = {"actor_id": "fallback", "person_id": "2", "display_name": "Matti Jennen"}
    roster = {"matti jensen": [exact], "matti jennen": [fallback]}
    parsed = parse_chair_title_named_proposers("Mietintö JAA / Matti Jensen EI")
    resolved = _actor_from_display_name(parsed[0].source_display_name, by_name=roster)
    assert resolved["actor_id"] == "exact"
    assert resolved["identity_state"] == "EXACT_SOURCE_DISPLAY_NAME"
    roster["matti jensen"].append(
        {"actor_id": "collision", "person_id": "3", "display_name": "Matti Jensen"}
    )
    unresolved = _actor_from_display_name(parsed[0].source_display_name, by_name=roster)
    assert unresolved["actor_id"] is None
    assert unresolved["person_id"] is None
    assert unresolved["identity_state"] == "CHAIR_TITLE_NAME_AMBIGUOUS"


def test_vote_totals_are_stage_local_not_final_adoption() -> None:
    lost = published_alternative_disposition(jaa=94, ei=75)
    assert lost["ei_side_state"] == "LOST_AT_THIS_VOTE"
    assert lost["final_adoption"]["state"] == "UNRESOLVED"
    assert lost["scope"] == "THIS_VOTE_ONLY"
    prevailed = published_alternative_disposition(jaa=15, ei=165)
    assert prevailed["ei_side_state"] == "PREVAILED_AT_THIS_VOTE"
    assert prevailed["final_adoption"]["state"] == "UNRESOLVED"
    # VK 7/2014 bracket witness: intermediate EI victories are not adoption.
    first = published_alternative_disposition(jaa=15, ei=165)
    second = published_alternative_disposition(jaa=43, ei=143)
    final = published_alternative_disposition(jaa=97, ei=81)
    assert first["ei_side_state"] == "PREVAILED_AT_THIS_VOTE"
    assert second["ei_side_state"] == "PREVAILED_AT_THIS_VOTE"
    assert final["ei_side_state"] == "LOST_AT_THIS_VOTE"
    assert all(row["final_adoption"]["state"] == "UNRESOLVED" for row in (first, second, final))
    missing = published_alternative_disposition(jaa=None, ei=10)
    assert missing["ei_side_state"] == "UNRESOLVED"

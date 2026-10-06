"""Deterministic cross-matter route/component projection from a bundled bank."""

import copy
import json
from pathlib import Path

import pytest

from paa.route_packet import (
    SCHEMA_VERSION,
    _BoundSource,
    _extract_dates,
    _extract_references,
    _extract_signatories,
    _parse_as_of,
    bind_answer_packet,
    build_route_packet,
    build_route_site,
    render_route_html,
)
from paa.source_pointers import build_source_pointer_index
from paa.source_question import SourceQuestionBank

FIXTURE = Path("paa/contracts/fixtures/route_comparison_bank")


def _bound(source_id: str, text: str, meta: dict) -> _BoundSource:
    return _BoundSource(
        source_id=source_id,
        meta=meta,
        index=build_source_pointer_index(source_id, text, granularity="line"),
    )


@pytest.fixture
def bank() -> SourceQuestionBank:
    return SourceQuestionBank.load(FIXTURE / "questions.json", FIXTURE / "sources.json")


def test_committee_prefix_not_detached_as_statute() -> None:
    text = "Valiokunnan mukaan tämä toteutuu (PeVL 9/2010 vp).\n"
    edges = _extract_references(
        {
            "X": _bound(
                "X",
                text,
                {"identifier": "HE 1/2099 vp", "stage": "PROPOSAL", "source_type": "GOVERNMENT_PROPOSAL"},
            )
        }
    )
    assert any(edge["target_label"] == "PeVL 9/2010 vp" for edge in edges)
    assert not any(
        edge.get("reference_kind") == "STATUTE_NUMBER" and edge.get("target_label") == "9/2010"
        for edge in edges
    )


def test_multiple_commencement_phrases_are_carrier_unresolved() -> None:
    text = (
        "Muutetaan lain 999/2025 voimaantulosäännös seuraavasti:\n"
        "Tämä laki tulee voimaan 1 päivänä tammikuuta 2028.\n"
        "Tämä laki tulee voimaan 1 päivänä joulukuuta 2026.\n"
    )
    dates = _extract_dates(
        {
            "SYNTH": _bound(
                "SYNTH",
                text,
                {"identifier": "123/2026", "stage": "ENACTED", "source_type": "AMENDING_ACT"},
            )
        }
    )
    roles = [row["date_role"] for row in dates if row.get("value")]
    assert roles.count("CARRIER_UNRESOLVED") == 2
    assert "OWN_COMMENCEMENT" not in roles
    single = _extract_dates(
        {
            "ONE": _bound(
                "ONE",
                "Tämä laki tulee voimaan 1 päivänä maaliskuuta 2026.\n",
                {"identifier": "1/2026", "stage": "ENACTED", "source_type": "AMENDING_ACT"},
            )
        }
    )
    assert any(row.get("date_role") == "OWN_COMMENCEMENT" for row in single)


def test_thirty_signatories_are_not_silently_truncated() -> None:
    names = "".join(f"Anna\nSukunimi{chr(65 + i // 26)}{chr(65 + i % 26)}\nsd\n" for i in range(30))
    text = f"Lakialoite\nHelsingissä 9.2.2024\n{names}{{END}}\n"
    rows = _extract_signatories(
        {
            "SIG": _bound(
                "SIG",
                text,
                {
                    "identifier": "LA 999/2024 vp",
                    "source_type": "LEGISLATIVE_INITIATIVE",
                    "source_date": "2024-02-09",
                },
            )
        }
    )
    assert len([row for row in rows if row.get("actor_name")]) == 30


def test_as_of_requires_explicit_cli_or_word_context() -> None:
    assert _parse_as_of("mitä oli voimassa 9.10.2100?", None) == "2100-10-09"
    assert _parse_as_of("allekirjoitettiin 1.3.2099", None) is None
    assert _parse_as_of("allekirjoitettiin 1.3.2099", "2100-10-09") == "2100-10-09"


def test_rejected_initiative_vs_other_enacted_route(bank: SourceQuestionBank) -> None:
    packet = build_route_packet(bank, "ROUTE-FULL", as_of="2100-10-09")
    assert packet["schema_version"] == SCHEMA_VERSION
    cues = packet["disposition_cues"]
    assert cues
    assert all(row.get("status") == "LEXICAL_CUE_NOT_DECISION" for row in cues)
    assert any(row.get("cue_kind") == "REJECTED_WORD_CUE" for row in cues)
    assert packet["summary"]["disposition_cues_are_not_decisions"] is True
    assert any(
        edge.get("target_label") == "HE 9/2099 vp"
        and edge.get("relation_status") == "DOCUMENTARY_EXPLICIT_REFERENCE"
        for edge in packet["route_edges"]
    )
    assert any(
        edge.get("target_label") == "100/2099" and edge.get("from_source_id") == "LAW-B"
        for edge in packet["route_edges"]
    )
    stages = {row["source_id"]: row["stage"] for row in packet["summary"]["stages"]}
    assert stages["INIT-TEXT"] == "PROPOSAL"
    assert stages["LAW-A"] == "ENACTED"
    assert packet["applicability"]["state"] == "SOURCE_BOUNDED"
    assert "questions_sha256" in packet["bank"]
    assert "sources_sha256" in packet["bank"]
    assert ".tmp" not in json.dumps(packet["bank"])


def test_formal_matter_number_is_not_detached_as_statute(bank: SourceQuestionBank) -> None:
    packet = build_route_packet(bank, "ROUTE-FULL")
    he_edges = [edge for edge in packet["route_edges"] if edge["from_source_id"] == "HE-TEXT"]
    assert any(edge["target_label"] in {"HE48/2024 vp", "HE 48/2024 vp"} for edge in he_edges)
    assert not any(
        edge.get("reference_kind") == "STATUTE_NUMBER" and edge.get("target_label") == "48/2024"
        for edge in he_edges
    )


def test_first_and_co_sign_distinct_from_drafting(bank: SourceQuestionBank) -> None:
    packet = build_route_packet(bank, "ROUTE-SIGNERS")
    named = [row for row in packet["actors"] if row.get("actor_name")]
    assert named[0]["actor_name"] == "Ada First"
    assert named[0]["role"] == "FIRST_SIGNATORY"
    assert named[0]["action_kind"]["value"] == "FIRST_SIGNATORY"
    assert named[0]["drafting"]["state"] == "UNRESOLVED"
    assert named[1]["actor_name"] == "Bea Second"
    assert named[1]["role"] == "CO_SIGNATORY"
    assert named[1]["action_kind"]["value"] == "INITIATIVE_COSIGNED"
    assert "DRAFTING_UNKNOWN" in {row["code"] for row in packet["unsupported"]}
    text = bank.sources["INIT-TEXT"]["text"]
    assert named[0]["source"]["quote"] in text
    assert named[1]["source"]["quote"] in text
    html = render_route_html(packet)
    assert "laatija: tuntematon" in html
    assert "(ei laatija)" not in html
    assert "laatijaa ei osoitettu" not in html


def test_same_address_different_conditions_are_not_equivalence(bank: SourceQuestionBank) -> None:
    packet = build_route_packet(bank, "ROUTE-FULL")
    comps = packet["component_comparisons"]
    assert comps
    assert all(row["equivalence_claim"] == "NOT_CLAIMED" for row in comps)
    assert any(row["text_identity"] == "DIFFERENT_OR_PARTIAL_EXCERPT" for row in comps)
    assert all(row["relation_status"] == "CANDIDATE_SAME_ADDRESS_OVERLAP" for row in comps)
    assert any(row["component_key"].startswith("100/2099|") for row in comps)
    # Bare § under a different statute must not join the 100/2099 address group.
    assert not any("300/2101" in row["component_key"] for row in comps)


def test_unqualified_section_stays_unresolved_without_statute(bank: SourceQuestionBank) -> None:
    packet = build_route_packet(bank, "ROUTE-FULL")
    init_comps = [row for row in packet["components"] if row["source_id"] == "INIT-TEXT"]
    assert init_comps
    assert all(row["component_key_state"] == "UNRESOLVED_ADDRESS_CANDIDATE" for row in init_comps)


def test_amending_own_commencement_vs_proposed_and_target(bank: SourceQuestionBank) -> None:
    packet = build_route_packet(bank, "ROUTE-FULL")
    own = [
        row
        for row in packet["dates"]
        if row.get("date_role") == "OWN_COMMENCEMENT" and row.get("source_id") == "LAW-B"
    ]
    proposed = [
        row
        for row in packet["dates"]
        if row.get("date_role") == "PROPOSED_COMMENCEMENT_CUE"
        and row.get("source_id") in {"HE-TEXT", "INIT-TEXT"}
    ]
    target = [
        row
        for row in packet["dates"]
        if row.get("date_role") == "UNDERLYING_TARGET_AMENDMENT_REFERENCE"
        and row.get("source_id") == "LAW-B"
    ]
    assert own and own[0]["value"] == "2100-01-01"
    assert own[0]["carrier_basis"] == "ENACTED_CARRIER"
    assert proposed
    assert not any(
        row.get("date_role") == "OWN_COMMENCEMENT" and row.get("source_id") in {"HE-TEXT", "INIT-TEXT"}
        for row in packet["dates"]
    )
    assert target and target[0]["target_statute"] == "100/2099"
    assert target[0]["value"] is None


def test_omitted_enacted_source_leaves_dependent_route_unresolved(bank: SourceQuestionBank) -> None:
    packet = build_route_packet(bank, "ROUTE-OMIT-ENACTED", as_of="2100-10-09")
    codes = {row["code"] for row in packet["residuals"]}
    assert "ALTERNATIVE_ENACTED_ROUTE_SOURCE_MISSING" in codes
    assert packet["applicability"]["state"] == "UNRESOLVED"
    assert packet["applicability"]["missingness"] == "ENACTED_OR_CONSOLIDATED_SOURCE_MISSING"
    assert any(row.get("cue_kind") == "REJECTED_WORD_CUE" for row in packet["disposition_cues"])


def test_typo_correction_distinct_from_substantive_revision(bank: SourceQuestionBank) -> None:
    packet = build_route_packet(bank, "ROUTE-FULL")
    corrections = [row for row in packet["dates"] if row.get("date_role") == "OFFICIAL_CORRECTION"]
    assert corrections
    typed = [row for row in corrections if row.get("correction_kind") == "TYPO_OR_WORDING_CORRECTION"]
    assert typed
    assert typed[0]["was"] == "kulussa"
    assert typed[0]["should_be"] == "kuluessa"
    assert "not automatically a substantive" in (typed[0].get("note") or "").casefold()


def test_answer_packet_binds_and_renders_first(bank: SourceQuestionBank, tmp_path: Path) -> None:
    answer = json.loads((FIXTURE / "answer_ROUTE-SIGNERS.json").read_text(encoding="utf-8"))
    packet = build_route_packet(bank, "ROUTE-SIGNERS", answer_packet=answer)
    attachment = packet["answer_attachment"]
    assert attachment["binding"]["state"] == "BOUND"
    assert attachment["candidate_answer"]["answer"]
    html = render_route_html(packet, packet_url="ROUTE-SIGNERS.json")
    # Compact candidate chrome leads; documentary index stays secondary.
    answer_pos = html.index("candidate-answer")
    documentary_pos = html.index("Dokumentaarinen indeksi")
    assert answer_pos < documentary_pos
    assert "Ada First on ensimmäinen allekirjoittaja" in html
    assert "https://example.test/INIT-TEXT" in html
    assert "Ada\nFirst" in html or "Ada" in html
    # Ordinary title — not the full research prompt as H1 when a reader title is given.
    titled = render_route_html(
        packet,
        reader_title="Lyhyt otsikko allekirjoittajista",
        prepared_answer="Ada First on ensimmäinen allekirjoittaja.",
        journey_href="../journeys/ROUTE_SIGNERS.html",
    )
    assert "<h1>Lyhyt otsikko allekirjoittajista</h1>" in titled
    assert "Avaa valmisteltu kysymysmatka" in titled
    assert "Tutkimuskysymys (tarkka ohje)" in titled
    bad = dict(answer)
    bad["question"] = "wrong question"
    rejected = bind_answer_packet(packet, bad, bank)
    assert rejected["binding"]["state"] == "REJECTED"
    manifest = build_route_site(
        questions=FIXTURE / "questions.json",
        sources=FIXTURE / "sources.json",
        output_dir=tmp_path / "site",
        task_id="ROUTE-SIGNERS",
        answer=FIXTURE / "answer_ROUTE-SIGNERS.json",
    )
    assert manifest["answer_tasks_bound"] == ["ROUTE-SIGNERS"]
    page = (tmp_path / "site" / "ROUTE-SIGNERS.html").read_text(encoding="utf-8")
    assert "candidate-answer" in page
    assert ".tmp/" not in page


def test_bind_answer_rejects_tampered_evidence_and_accepts_valid(bank: SourceQuestionBank) -> None:
    answer = json.loads((FIXTURE / "answer_ROUTE-SIGNERS.json").read_text(encoding="utf-8"))
    base = build_route_packet(bank, "ROUTE-SIGNERS")
    ok = bind_answer_packet(base, answer, bank)
    assert ok["binding"]["state"] == "BOUND"
    assert ok["candidate_answer"] is not None
    assert ok["binding"]["evidence_revalidated_against_task_indexes"] is True

    forged = copy.deepcopy(answer)
    forged["claims"][0]["evidence_bound"][0]["quote"] = "FORGED QUOTE NOT IN SOURCE"
    assert bind_answer_packet(base, forged, bank)["binding"]["state"] == "REJECTED"

    offsets = copy.deepcopy(answer)
    offsets["claims"][0]["evidence_bound"][0]["start"] += 1
    assert bind_answer_packet(base, offsets, bank)["binding"]["state"] == "REJECTED"

    digest = copy.deepcopy(answer)
    digest["claims"][0]["evidence_bound"][0]["source_text_sha256"] = "0" * 64
    assert bind_answer_packet(base, digest, bank)["binding"]["state"] == "REJECTED"

    unknown = copy.deepcopy(answer)
    unknown["claims"][0]["evidence_bound"][0]["source_id"] = "NOT_IN_BANK"
    assert bind_answer_packet(base, unknown, bank)["binding"]["state"] == "REJECTED"

    for key in ("first_pointer", "last_pointer", "start", "end"):
        for invalid_value in (0.5, True, "0"):
            fractional = copy.deepcopy(answer)
            fractional["claims"][0]["evidence_bound"][0][key] = invalid_value
            result = bind_answer_packet(base, fractional, bank)
            assert result["binding"]["state"] == "REJECTED"
            assert any("exact integers" in error for error in result["binding"]["errors"])

    omitted = copy.deepcopy(answer)
    omitted.pop("coverage", None)
    assert bind_answer_packet(base, omitted, bank)["binding"]["state"] == "REJECTED"


def test_global_carrier_does_not_bypass_multi_commencement_guard() -> None:
    text = (
        "Muutetaan lain 999/2025 voimaantulosäännös seuraavasti:\n"
        "Tämä laki tulee voimaan 1 päivänä tammikuuta 2028.\n"
        "Tämä laki tulee voimaan 1 päivänä joulukuuta 2026.\n"
    )
    dates = _extract_dates(
        {
            "DATE": _bound(
                "DATE",
                text,
                {
                    "identifier": "123/2026",
                    "stage": "ENACTED",
                    "source_type": "AMENDING_ACT",
                    "source_structure": {"carrier": "underlying_target"},
                },
            )
        }
    )
    roles = [row["date_role"] for row in dates if row.get("value")]
    assert roles.count("CARRIER_UNRESOLVED") == 2
    assert "OWN_COMMENCEMENT" not in roles


def test_multiword_given_name_keeps_first_signatory() -> None:
    text = "Lakialoite\nHelsingissä 9.2.2024\nAnna Maria\nNieminen\nsd\nMatti\nVirtanen\nkok\n{END}\n"
    rows = _extract_signatories(
        {
            "MULTI": _bound(
                "MULTI",
                text,
                {
                    "identifier": "LA 998/2024 vp",
                    "source_type": "LEGISLATIVE_INITIATIVE",
                    "source_date": "2024-02-09",
                },
            )
        }
    )
    named = [row for row in rows if row.get("actor_name")]
    assert named[0]["actor_name"] == "Anna Maria Nieminen"
    assert named[0]["role"] == "FIRST_SIGNATORY"
    assert named[1]["actor_name"] == "Matti Virtanen"
    assert named[1]["role"] == "CO_SIGNATORY"


def test_laki_statute_and_lowercase_formal_matter_reference() -> None:
    text = "Laki 428/2024 tuli voimaan. HE 48/2024 vp ja PeVL 9/2010 vp. he 48/2024 vp."
    edges = _extract_references(
        {
            "REF": _bound(
                "REF",
                text,
                {"identifier": "control", "stage": "ENACTED", "source_type": "AMENDING_ACT"},
            )
        }
    )
    assert not any(
        edge.get("reference_kind") == "FORMAL_MATTER_ID"
        and "Laki" in str(edge.get("target_label") or "")
        for edge in edges
    )
    assert any(
        edge.get("reference_kind") == "STATUTE_NUMBER" and edge.get("target_label") == "428/2024"
        for edge in edges
    )
    assert any(
        edge.get("reference_kind") == "FORMAL_MATTER_ID"
        and str(edge.get("target_label") or "").casefold().startswith("he 48/2024")
        for edge in edges
    )
    assert not any(
        edge.get("reference_kind") == "STATUTE_NUMBER" and edge.get("target_label") == "48/2024"
        for edge in edges
    )


def test_source_profile_preserves_native_and_maps_consolidated_labels() -> None:
    from paa.source_bank_profile import normalize_source_meta

    first = normalize_source_meta(
        {
            "stage": "successor_enacted_law",
            "source_type": "official_successor_enacted_law",
            "source_voice": "enacted_statute_successor",
        }
    )
    assert first["stage"] == "ENACTED"
    assert first["stage_native"] == "successor_enacted_law"
    second = normalize_source_meta(first)
    assert second["stage_native"] == "successor_enacted_law"
    assert second["source_type_native"] == "official_successor_enacted_law"
    consolidated = normalize_source_meta(
        {"stage": "consolidated_capture", "source_type": "consolidated_act"}
    )
    assert consolidated["stage"] == "CONSOLIDATED_CAPTURE"
    assert consolidated["source_type"] == "CONSOLIDATED_ACT"
    reply = normalize_source_meta(
        {"stage": "adopted_legislative_text", "source_type": "parliament_reply"}
    )
    assert reply["stage"] == "ADOPTED_LEGISLATIVE_TEXT"
    assert reply["source_type"] == "PARLIAMENT_REPLY"


def test_public_page_is_question_first_with_download_and_no_research_notes(
    bank: SourceQuestionBank, tmp_path: Path
) -> None:
    packet = build_route_packet(bank, "ROUTE-FULL", as_of="2100-10-09")
    html = render_route_html(packet, packet_url="ROUTE-FULL.json")
    assert packet["question"][:40] in html
    assert "Ada First" in html
    assert "Ensimmäinen allekirjoittaja" in html
    assert "laatija: tuntematon" in html
    assert "ehdokas sama osoite, ei politiikkavastaavuus" in html
    assert "Sanallinen vihje" in html
    assert 'download href="ROUTE-FULL.json"' in html
    assert "OPENCODE" not in html
    assert "prompt" not in html.casefold()
    assert ".tmp/" not in html
    assert ".tmp/" not in json.dumps(packet)
    manifest = build_route_site(
        questions=FIXTURE / "questions.json",
        sources=FIXTURE / "sources.json",
        output_dir=tmp_path / "site",
        all_tasks=True,
    )
    assert manifest["task_ids"] == ["ROUTE-FULL", "ROUTE-OMIT-ENACTED", "ROUTE-SIGNERS"]
    assert (tmp_path / "site" / "index.html").is_file()
    assert (tmp_path / "site" / "ROUTE-FULL.html").is_file()

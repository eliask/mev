"""Question journey projection: explicit mapping, review binding, generic render."""

import copy
import json
from pathlib import Path

import pytest

from paa.question_journey import (
    ADMISSION_UNREVIEWED,
    MAPPING_SCHEMA_VERSION,
    PROJECTION_REVIEW_SCHEMA,
    SCHEMA_VERSION,
    QuestionJourneyError,
    _display_limit_items,
    _evidence_details,
    build_lookup_index,
    compose_journey,
    lookup_query,
    render_journey_html,
    render_lookup_html,
    validate_mapping_wire,
    write_journey_site,
)
from paa.route_packet import build_route_packet
from paa.source_question import SourceQuestionBank

ROUTE_FIXTURE = Path("paa/contracts/fixtures/route_comparison_bank")
JOURNEY_FIXTURE = Path("paa/contracts/fixtures/question_journey_bank")


@pytest.fixture
def bank() -> SourceQuestionBank:
    return SourceQuestionBank.load(ROUTE_FIXTURE / "questions.json", ROUTE_FIXTURE / "sources.json")


@pytest.fixture
def route_full(bank: SourceQuestionBank) -> dict:
    answer = json.loads((JOURNEY_FIXTURE / "answer_ROUTE-FULL.json").read_text(encoding="utf-8"))
    return build_route_packet(bank, "ROUTE-FULL", as_of="2100-10-09", answer_packet=answer)


@pytest.fixture
def mapping_full() -> dict:
    return json.loads((JOURNEY_FIXTURE / "mapping_ROUTE-FULL.json").read_text(encoding="utf-8"))


def test_mapping_codec_accepts_fixture(mapping_full: dict) -> None:
    assert mapping_full["schema_version"] == MAPPING_SCHEMA_VERSION
    assert validate_mapping_wire(mapping_full) == []


def test_compose_branching_journey_is_unreviewed(route_full: dict, mapping_full: dict) -> None:
    answer = json.loads((JOURNEY_FIXTURE / "answer_ROUTE-FULL.json").read_text(encoding="utf-8"))
    packet = compose_journey(route_packet=route_full, mapping=mapping_full, answer_packet=answer)
    assert packet["schema_version"] == SCHEMA_VERSION
    assert packet["admission"] == ADMISSION_UNREVIEWED
    assert packet["projection_status"] == "UNREVIEWED"
    assert packet["journey_kind"] == "BRANCHING_ROUTES"
    assert len(packet["branches"]) == 2
    assert {row["branch_id"] for row in packet["branches"]} == {"initiative", "government"}
    assert any(row["relation_kind"] == "LEXICAL_CUE_CANDIDATE" for row in packet["edges"])
    assert any(row["relation_kind"] == "ANNUAL_SUCCESSOR_NOT_AMENDMENT" for row in packet["edges"])
    assert all(row["admission"] == ADMISSION_UNREVIEWED for row in packet["nodes"])
    assert all(row["source_versions"] for row in packet["nodes"])


def test_lexical_only_stage_cannot_become_formal_decision(route_full: dict, mapping_full: dict) -> None:
    bad = copy.deepcopy(mapping_full)
    for edge in bad["edges"]:
        if edge["edge_id"] == "e-la-cue":
            edge["relation_kind"] = "INSTITUTIONAL_RESULT"
            edge.pop("supporting_claim_indices", None)
    answer = json.loads((JOURNEY_FIXTURE / "answer_ROUTE-FULL.json").read_text(encoding="utf-8"))
    with pytest.raises(QuestionJourneyError, match="lexical-only|STRONG_EDGE|promotes lexical"):
        compose_journey(route_packet=route_full, mapping=bad, answer_packet=answer)


def test_missing_voice_kind_is_rejected_and_not_inferred_as_institution(
    route_full: dict, mapping_full: dict
) -> None:
    """Closed boundary: no PERSON-from-name or INSTITUTION-from-absence default."""

    import hashlib

    answer = json.loads((JOURNEY_FIXTURE / "answer_ROUTE-FULL.json").read_text(encoding="utf-8"))
    missing = copy.deepcopy(mapping_full)
    missing["nodes"][2]["kind"] = "ALTERNATIVE"
    missing["nodes"][2].pop("voice_kind", None)
    defects = validate_mapping_wire(missing)
    assert any(d.code == "VOICE_KIND_REQUIRED" for d in defects)
    with pytest.raises(QuestionJourneyError, match="VOICE_KIND_REQUIRED|explicit voice_kind"):
        compose_journey(route_packet=route_full, mapping=missing, answer_packet=answer)

    # actor_name alone must not become PERSON
    named = copy.deepcopy(mapping_full)
    named["nodes"][2]["kind"] = "ALTERNATIVE"
    named["nodes"][2]["actor_name"] = "Someone"
    named["nodes"][2].pop("voice_kind", None)
    with pytest.raises(QuestionJourneyError, match="VOICE_KIND_REQUIRED|explicit voice_kind"):
        compose_journey(route_packet=route_full, mapping=named, answer_packet=answer)

    # Explicit INSTITUTION on ALTERNATIVE remains a valid control
    institutional = copy.deepcopy(mapping_full)
    institutional["nodes"][2]["kind"] = "ALTERNATIVE"
    institutional["nodes"][2]["voice_kind"] = "INSTITUTION"
    institutional["nodes"][2].pop("actor_name", None)
    packet = compose_journey(route_packet=route_full, mapping=institutional, answer_packet=answer)
    alt = next(n for n in packet["nodes"] if n["kind"] == "ALTERNATIVE")
    assert alt["voice_kind"] == "INSTITUTION"

    # UNKNOWN on ACTOR_ACT/ALTERNATIVE cannot be admitted; unresolved endpoints may use UNKNOWN.
    unknown_actor = copy.deepcopy(mapping_full)
    unknown_actor["nodes"][0]["voice_kind"] = "UNKNOWN"
    packet_u = compose_journey(route_packet=route_full, mapping=unknown_actor, answer_packet=answer)
    candidate_sha = hashlib.sha256(
        json.dumps(
            {
                "task_id": answer.get("task_id"),
                "matter_id": answer.get("matter_id"),
                "question": answer.get("question"),
                "answer": answer.get("answer"),
                "claims": answer.get("claims"),
                "unresolved": answer.get("unresolved"),
                "next_evidence": answer.get("next_evidence"),
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    review = {
        "schema_version": PROJECTION_REVIEW_SCHEMA,
        "task_id": "ROUTE-FULL",
        "mapping_sha256": packet_u["mapping_sha256"],
        "journey_content_sha256": packet_u["journey_content_sha256"],
        "questions_sha256": route_full["bank"]["questions_sha256"],
        "sources_sha256": route_full["bank"]["sources_sha256"],
        "candidate_sha256": candidate_sha,
        "review_date": "2026-10-10",
        "decision": "SUPPORTED_WITH_STATED_LIMITS",
        "reviewed_fields": ["nodes", "edges", "branches", "short_answer"],
        "reviewer": {
            "id": "fixture-reviewer",
            "kind": "INDEPENDENT_AI_SOURCE_READER",
            "human_certification": False,
        },
    }
    with pytest.raises(QuestionJourneyError, match="rejects UNKNOWN voice_kind on ACTOR_ACT"):
        compose_journey(
            route_packet=route_full,
            mapping=unknown_actor,
            answer_packet=answer,
            projection_review=review,
        )


def test_unbound_claim_index_is_rejected(route_full: dict, mapping_full: dict) -> None:
    bad = copy.deepcopy(mapping_full)
    bad["nodes"][0]["supporting_claim_indices"] = [99]
    answer = json.loads((JOURNEY_FIXTURE / "answer_ROUTE-FULL.json").read_text(encoding="utf-8"))
    with pytest.raises(QuestionJourneyError, match="unbound claim index"):
        compose_journey(route_packet=route_full, mapping=bad, answer_packet=answer)


def test_stale_projection_review_is_rejected(route_full: dict, mapping_full: dict) -> None:
    answer = json.loads((JOURNEY_FIXTURE / "answer_ROUTE-FULL.json").read_text(encoding="utf-8"))
    packet = compose_journey(route_packet=route_full, mapping=mapping_full, answer_packet=answer)
    review = {
        "schema_version": PROJECTION_REVIEW_SCHEMA,
        "task_id": "ROUTE-FULL",
        "mapping_sha256": "0" * 64,
        "journey_content_sha256": packet["journey_content_sha256"],
        "questions_sha256": route_full["bank"]["questions_sha256"],
        "sources_sha256": route_full["bank"]["sources_sha256"],
        "candidate_sha256": "1" * 64,
        "review_date": "2026-10-10",
        "decision": "SUPPORTED_WITH_STATED_LIMITS",
        "reviewed_fields": ["nodes", "edges", "branches", "short_answer"],
        "reviewer": {
            "id": "fixture-reviewer",
            "kind": "INDEPENDENT_AI_SOURCE_READER",
            "human_certification": False,
        },
    }
    with pytest.raises(QuestionJourneyError, match="mapping_sha256"):
        compose_journey(
            route_packet=route_full,
            mapping=mapping_full,
            answer_packet=answer,
            projection_review=review,
        )


def test_renderer_is_generic_and_preserves_deep_links(route_full: dict, mapping_full: dict) -> None:
    answer = json.loads((JOURNEY_FIXTURE / "answer_ROUTE-FULL.json").read_text(encoding="utf-8"))
    packet = compose_journey(route_packet=route_full, mapping=mapping_full, answer_packet=answer)
    page = render_journey_html(packet, packet_url="ROUTE-FULL.json")
    assert "Ada First" in page
    assert "id='node-la-sign-ada'" in page or 'id="node-la-sign-ada"' in page
    assert "Päätösmatka" in page
    assert "tarkistamaton ehdokas" in page
    assert "LA 9/2099" in page  # data may name matters
    # Generic renderer must not hard-code real development case identifiers.
    assert "Kymäläinen" not in Path("paa/question_journey.py").read_text(encoding="utf-8")
    assert "Räsänen" not in Path("paa/question_journey.py").read_text(encoding="utf-8")
    assert "428/2024" not in Path("paa/question_journey.py").read_text(encoding="utf-8")


def test_alternative_sequence_specimen(bank: SourceQuestionBank) -> None:
    mapping = json.loads((JOURNEY_FIXTURE / "mapping_ALT-SEQ.json").read_text(encoding="utf-8"))
    answer = json.loads((ROUTE_FIXTURE / "answer_ROUTE-SIGNERS.json").read_text(encoding="utf-8"))
    route = build_route_packet(bank, "ROUTE-SIGNERS", answer_packet=answer)
    packet = compose_journey(route_packet=route, mapping=mapping, answer_packet=answer)
    assert packet["journey_kind"] == "ALTERNATIVE_SEQUENCE"
    assert packet["branches"][0]["branch_id"] == "signature-sequence"


def test_lookup_failure_does_not_fabricate_answer(route_full: dict, mapping_full: dict) -> None:
    answer = json.loads((JOURNEY_FIXTURE / "answer_ROUTE-FULL.json").read_text(encoding="utf-8"))
    packet = compose_journey(route_packet=route_full, mapping=mapping_full, answer_packet=answer)
    index = build_lookup_index([packet])
    miss = lookup_query(index, "totally-unknown-person-xyz")
    assert miss["status"] == "LOOKUP_FAILURE"
    assert miss["matches"]["questions"] == []
    assert miss.get("coverage_kind") == "ABSENT_FROM_INDEX"
    hit = lookup_query(index, "Ada First")
    assert hit["status"] == "OK"
    assert hit["matches"]["people"]


def test_participants_expand_once_per_shared_act(route_full: dict, mapping_full: dict) -> None:
    answer = json.loads((JOURNEY_FIXTURE / "answer_ROUTE-FULL.json").read_text(encoding="utf-8"))
    packet = compose_journey(route_packet=route_full, mapping=mapping_full, answer_packet=answer)
    names = [row["actor_name"] for row in packet["actors"]]
    assert names.count("Ada First") == 1
    assert names.count("Bea Second") == 1
    bea = next(row for row in packet["actors"] if row["actor_name"] == "Bea Second")
    assert bea["signature_role"] == "CO_SIGNATORY"
    assert bea["shared_act_id"] == "la-9-2099-signature"
    index = build_lookup_index([packet])
    hit = lookup_query(index, "Bea Second")
    assert hit["status"] == "OK"
    assert hit["matches"]["people"][0]["actor_name"] == "Bea Second"
    page = render_journey_html(packet)
    assert "Tiivis yhteenveto" in page
    assert "CO_SIGNATORY" in page
    assert "human_certification=false" not in page.split("Rajat ja tekninen auditointi")[0]


def test_discovery_merges_matter_records_and_dossier_aliases(
    route_full: dict, mapping_full: dict
) -> None:
    answer = json.loads((JOURNEY_FIXTURE / "answer_ROUTE-FULL.json").read_text(encoding="utf-8"))
    packet = compose_journey(route_packet=route_full, mapping=mapping_full, answer_packet=answer)
    index = build_lookup_index(
        [packet],
        matter_records=[
            {
                "matter_id": "HE 149/2099 vp",
                "title": "Sairaalaverkko",
                "choice_label": "Sairaalaverkko",
                "href": "matters/HE_149_2099_vp.html",
                "actors": [
                    {
                        "actor_name": "Juha Example",
                        "actor_key": "actor:mp-1537",
                        "href": "matters/HE_149_2099_vp/people/actor_mp-1537.html",
                        "identity_state": "EXACT_SOURCE_PERSON_ID",
                        "coverage_kind": "RECORD_ONLY",
                    }
                ],
            }
        ],
        dossier_entries=[
            {
                "title": "Oulaskangas",
                "aliases": ["Oulaskangas", "Oulaskankaan"],
                "href": "oulaskangas.html",
                "matter_ids": ["HE 149/2099 vp"],
                "coverage_kind": "RECORD_ONLY",
            }
        ],
    )
    person = lookup_query(index, "Juha Example")
    assert person["status"] == "OK"
    assert person["coverage_kind"] == "RECORD_ONLY"
    assert person["matches"]["people"][0]["record_hrefs"]
    dossier = lookup_query(index, "Oulaskangas")
    assert dossier["status"] == "OK"
    assert dossier["matches"]["dossiers"]
    assert dossier["matches"]["matters"]


def test_key_conditions_and_node_scope_appear_in_compact_display(
    route_full: dict, mapping_full: dict
) -> None:
    answer = json.loads((JOURNEY_FIXTURE / "answer_ROUTE-FULL.json").read_text(encoding="utf-8"))
    mapping = copy.deepcopy(mapping_full)
    mapping["key_conditions"] = ["Condition A is decisive.", "Condition B differs from A."]
    mapping["nodes"][0]["scope"] = "Scope text must not stay audit-only."
    packet = compose_journey(route_packet=route_full, mapping=mapping, answer_packet=answer)
    assert packet["key_conditions"] == mapping["key_conditions"]
    page = render_journey_html(packet)
    assert "key-conditions" in page
    assert "Condition A is decisive." in page
    assert "Scope text must not stay audit-only." in page
    assert "class='condition'" in page or 'class="condition"' in page
    # Answer + key choice precede the act summary (docs/UI.md reading order).
    assert page.index("class='key-conditions'") < page.index("class='act-summary'")
    assert page.index("class='takeaway'") < page.index("class='key-conditions'")
    assert "act-summary-list" in page
    # Full branch walk is progressive disclosure; conditions stay on the first screen.
    assert "decision-journey" in page
    assert "Päätösmatka (täysi kulku)" in page
    before_journey = page.split("Päätösmatka (täysi kulku)", 1)[0]
    assert "Condition A is decisive." in before_journey
    assert "Scope text must not stay audit-only." in page
    # Nearby already-bound evidence belongs beside the takeaway, not only in the journey fold.
    if "nearby-evidence" in page:
        assert page.index("nearby-evidence") < page.index("Päätösmatka (täysi kulku)")


def test_compact_surface_keeps_overflow_conditions_and_real_source_target(
    route_full: dict, mapping_full: dict
) -> None:
    answer = json.loads((JOURNEY_FIXTURE / "answer_ROUTE-FULL.json").read_text(encoding="utf-8"))
    mapping = copy.deepcopy(mapping_full)
    mapping["key_conditions"] = [f"Ehto {index}" for index in range(1, 6)]
    packet = compose_journey(route_packet=route_full, mapping=mapping, answer_packet=answer)
    page = render_journey_html(packet)
    assert "Ehto 1" in page and "Ehto 3" in page
    assert "Ehto 4" in page and "Ehto 5" in page
    assert "Muut ehdot" in page
    assert "id='journey-sources'" in page or 'id="journey-sources"' in page
    assert "href='#node-la-sign-ada'" not in page


def test_act_summary_removes_only_exact_actor_prefix(
    route_full: dict, mapping_full: dict
) -> None:
    answer = json.loads((JOURNEY_FIXTURE / "answer_ROUTE-FULL.json").read_text(encoding="utf-8"))
    mapping = copy.deepcopy(mapping_full)
    mapping["nodes"][0]["label"] = "Ada First · ensimmäinen allekirjoittaja"
    packet = compose_journey(route_packet=route_full, mapping=mapping, answer_packet=answer)
    page = render_journey_html(packet)
    assert "<strong>Ada First, Bea Second</strong> — ensimmäinen allekirjoittaja" in page
    assert "<strong>Ada First, Bea Second</strong> — Ada First · ensimmäinen allekirjoittaja" not in page


def test_lookup_identity_state_uses_finnish_primary_label(
    route_full: dict, mapping_full: dict
) -> None:
    answer = json.loads((JOURNEY_FIXTURE / "answer_ROUTE-FULL.json").read_text(encoding="utf-8"))
    packet = compose_journey(route_packet=route_full, mapping=mapping_full, answer_packet=answer)
    index = build_lookup_index(
        [packet],
        matter_records=[
            {
                "matter_id": "HE 149/2099 vp",
                "title": "Sairaalaverkko",
                "href": "matters/HE_149_2099_vp.html",
                "actors": [
                    {
                        "actor_name": "Juha Example",
                        "actor_key": "actor:mp-1537",
                        "href": "matters/HE_149_2099_vp/people/actor_mp-1537.html",
                        "identity_state": "EXACT_SOURCE_PERSON_ID",
                        "coverage_kind": "RECORD_ONLY",
                    },
                    {
                        "actor_name": "Name Only Example",
                        "actor_key": "name-only:Name Only Example",
                        "href": "matters/HE_149_2099_vp/people/name-only_Name_Only_Example.html",
                        "identity_state": "SOURCE_NAME_ONLY_UNJOINED",
                        "coverage_kind": "RECORD_ONLY",
                    },
                ],
            }
        ],
    )
    page = render_lookup_html(index)
    assert "Tarkka henkilönumero lähteessä" in page
    assert "Vain nimitieto (ei yhdistetty henkilönumeroon)" in page
    assert "identity-code" in page
    assert "EXACT_SOURCE_PERSON_ID" in page
    assert "SOURCE_NAME_ONLY_UNJOINED" in page
    # Ambiguous rows stay distinct keys; no fuzzy merge.
    assert "actor:mp-1537" in page or "Juha Example" in page
    assert "name-only:Name Only Example" in page or "Name Only Example" in page


def test_write_journey_site_emits_lookup_and_static_pages(
    route_full: dict, mapping_full: dict, tmp_path: Path
) -> None:
    answer = json.loads((JOURNEY_FIXTURE / "answer_ROUTE-FULL.json").read_text(encoding="utf-8"))
    packet = compose_journey(route_packet=route_full, mapping=mapping_full, answer_packet=answer)
    manifest = write_journey_site([packet], tmp_path / "site")
    assert (tmp_path / "site" / "index.html").is_file()
    assert (tmp_path / "site" / "lookup.html").is_file()
    assert (tmp_path / "site" / "lookup.json").is_file()
    assert manifest["journeys"][0]["projection_status"] == "UNREVIEWED"
    html = (tmp_path / "site" / "ROUTE_FULL.html").read_text(encoding="utf-8")
    assert "Haku" in html
    lookup_page = (tmp_path / "site" / "lookup.html").read_text(encoding="utf-8")
    assert "<noscript>" in lookup_page


def test_lookup_filter_script_syncs_query_and_hides_via_hidden_not_inline_display(
    route_full: dict, mapping_full: dict
) -> None:
    """Regression: ?q= must seed the input; hide via hidden/CSS, not style.display."""

    answer = json.loads((JOURNEY_FIXTURE / "answer_ROUTE-FULL.json").read_text(encoding="utf-8"))
    packet = compose_journey(route_packet=route_full, mapping=mapping_full, answer_packet=answer)
    page = render_lookup_html(build_lookup_index([packet]), initial_query="")
    assert "syncFromUrl" in page
    assert "searchParams.get('q')" in page
    assert "popstate" in page
    assert "article[hidden]" in page
    assert "display:none !important" in page
    assert "card.hidden" in page
    assert "card.style.display" not in page
    assert "Ilman JavaScriptiä" in page
    assert "lookup-browse" in page
    assert "Valmiita oikopolkuja" in page or "lookup-shortcuts" in page
    assert "Tyhjä haku ei avaa koko indeksiä" in page
    assert "Näytä lisää" in page
    # Full index lives under browse disclosure; empty default must not paint it open.
    assert "lookup-browse" in page and "id=" in page
    assert "limit +=" in page or "INITIAL_LIMIT" in page


def test_lookup_query_clear_restores_empty_surface_and_preserves_explicit_browse(
    route_full: dict, mapping_full: dict
) -> None:
    """A typed query is temporary; only an explicit browse toggle opens the index."""

    answer = json.loads((JOURNEY_FIXTURE / "answer_ROUTE-FULL.json").read_text(encoding="utf-8"))
    packet = compose_journey(route_packet=route_full, mapping=mapping_full, answer_packet=answer)
    page = render_lookup_html(build_lookup_index([packet]), initial_query="")
    assert "let autoOpened = false" in page
    assert "browse.open = false" in page
    assert "browse.open && !autoOpened" in page
    assert "window.addEventListener('popstate'" in page
    assert "if (!(input.value || '').trim() && autoOpened)" in page
    # The distinction is still available, but does not repeat the first answer by default.
    assert "<details class='boundary'><summary>Rajaus</summary>" in render_journey_html(packet)


def test_reviewed_display_drops_workflow_residue_keeps_substantive_limits() -> None:
    """D1: NEGATIVE.json / specimen / not-self-approved stay out of REVIEWED chrome."""

    packet_limits = [
        (
            "Repairs R1–R6 from BS02-D1.journey_projection_review_NEGATIVE.json; "
            "this file is not self-approved."
        ),
        "Development specimen from retained BS02 witnesses; not a fresh transfer evaluation matter.",
        "Source-answer review of BS02-D1 prose does not admit this graph.",
        "as_of/tarkastelupäivä 2025-03-07 unchanged; later register deferred.",
        "Personal signature is not drafting authorship or causal influence on a government proposal.",
        "An annual successor statute is not automatically an amendment of the prior annual act.",
        "Journey edges are an explicit projection; the renderer does not invent relations.",
        "Lexical disposition cues are not institutional decisions.",
        "Unknown causal contribution of the named signer remains open.",
    ]
    shown = _display_limit_items(packet_limits, projection_status="REVIEWED_AI_SOURCE_READING")
    blob = "\n".join(shown)
    assert "NEGATIVE.json" not in blob
    assert "not self-approved" not in blob
    assert "Development specimen" not in blob
    assert "does not admit this graph" not in blob
    assert "2025-03-07" in blob
    assert "Unknown causal contribution" in blob
    assert "Vuotuinen seuraajasäädös" in blob
    assert "Henkilökohtainen allekirjoitus" in blob
    assert "eksplisiittinen projektio" in blob
    assert "Sanastolliset" in blob
    # Machine packet list must remain the caller's object (display is a projection).
    assert packet_limits[0].startswith("Repairs R1")


def test_unreviewed_display_keeps_candidate_status_and_does_not_strip_unknown() -> None:
    limits = [
        "Unknown causal contribution remains open.",
        "UNREVIEWED projection candidate — separate projection review required.",
    ]
    shown = _display_limit_items(limits, projection_status="UNREVIEWED")
    blob = "\n".join(shown)
    assert "tarkistamaton ehdokas" in blob
    assert "Unknown causal contribution remains open." in blob
    # Candidate chrome already states prose≠graph; still show non-workflow unknowns.
    assert "UNREVIEWED projection candidate" in blob


def test_evidence_details_states_empty_excerpt_when_only_stage_refs() -> None:
    html = _evidence_details(
        {
            "stage_snapshots": [
                {
                    "source_id": "VOTES-HE21",
                    "identifier": "HE 21/2025 vp votes",
                    "stage": "BALLOT",
                }
            ],
            "source_versions": [{"source_id": "VOTES-HE21", "text_sha256": "abc123def456"}],
        }
    )
    assert "empty-evidence" in html
    assert "Ei sidottua tekstiotetta" in html
    assert "VOTES-HE21" in html
    assert "<blockquote>" not in html


def test_explicit_lead_evidence_matches_the_question_not_first_signature(
    route_full: dict, mapping_full: dict
) -> None:
    """Policy explanation opens policy evidence, with the declared source voice."""
    from paa.question_journey import render_nearby_evidence_html

    mapping = copy.deepcopy(mapping_full)
    mapping['reader_title'] = 'Mitä ilmoitustavalle ehdotettiin?'
    mapping['short_answer'] = 'Aloite hylättiin.\n\nHallituksen eri ehdotus hyväksyttiin.'
    mapping['lead_evidence'] = [{
        'node_id': 'la-proposal', 'claim_index': 2, 'evidence_index': 0,
        'caption': 'Aloitteen ehdottama ilmoitustapa',
    }]
    packet = compose_journey(route_packet=route_full, mapping=mapping)
    quote = next(n for n in packet['nodes'] if n['node_id'] == 'la-proposal')['claim_excerpts'][0]['evidence_bound'][0]['quote']
    excerpt = render_nearby_evidence_html(packet)
    assert quote in excerpt
    assert 'Ada\nFirst' not in excerpt
    assert 'Aloitteen ehdottama ilmoitustapa' in excerpt
    packet['source_answer_review'] = {'reader_title': 'Vanha tekninen otsikko'}
    page = render_journey_html(packet)
    assert '<h1>Mitä ilmoitustavalle ehdotettiin?</h1>' in page
    assert '<p>Aloite hylättiin.</p><p>Hallituksen eri ehdotus hyväksyttiin.</p>' in page
    assert "class='personal-records'" in page
    assert packet['lead_evidence'] == mapping['lead_evidence']
    plain = compose_journey(route_packet=route_full, mapping=mapping_full)
    assert plain['journey_content_sha256'] != packet['journey_content_sha256']


@pytest.mark.parametrize('selector', [
    {'node_id': 'missing', 'claim_index': 2, 'evidence_index': 0, 'caption': 'Lähde'},
    {'node_id': 'la-proposal', 'claim_index': 0, 'evidence_index': 0, 'caption': 'Lähde'},
    {'node_id': 'la-proposal', 'claim_index': 2, 'evidence_index': 999, 'caption': 'Lähde'},
    {'node_id': 'la-proposal', 'claim_index': 2, 'evidence_index': True, 'caption': 'Lähde'},
])
def test_unbound_lead_evidence_cannot_fall_back_to_an_unrelated_quote(
    route_full: dict, mapping_full: dict, selector: dict
) -> None:
    bad = copy.deepcopy(mapping_full)
    bad['lead_evidence'] = [selector]
    with pytest.raises(QuestionJourneyError, match='lead_evidence'):
        compose_journey(route_packet=route_full, mapping=bad)

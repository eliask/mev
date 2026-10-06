"""The person/matter reading view preserves records and uncertain identity."""

from copy import deepcopy

from paa.matter_site import (
    compact_actor_summaries,
    render_matter_html,
    render_matter_index,
    render_matter_person_html,
    write_matter_site,
)


def _packet():
    action = {
        "actor": {
            "actor_id": "mp-1",
            "person_id": "1",
            "name": "First Person",
            "identity_state": "EXACT_PERSON_ID",
        },
        "source": {"url": "https://example.test/vote", "record_locator": "vote:1/person:1"},
        "kind": "VOTE",
        "role_label": "BALLOT_MEMBER",
        "action_kind": {"value": "VOTE_CAST", "basis": "BALLOT_REGISTER"},
        "action_date": "2024-01-02",
        "action_date_basis": "VOTE_EVENT_DATE",
        "object_id": "vote:1",
        "title": "First alternative",
        "raw_response": "POISSA",
        "evidence_ids": ["vote-1-e"],
        "ballot_source_version": {"state": "NORMALIZED_ROW_NO_RAW_HASH"},
        "role_intervals": {
            "state": "EXACT_DATE_OVERLAP",
            "imprecise_intervals_omitted": True,
            "roles": [
                {
                    "label": "Eduskuntaryhmä Kokoomus",
                    "kind": "Eduskuntaryhma",
                    "start_date": "2023-04-01",
                    "end_date": "2027-04-01",
                    "match_state": "EXACT_DATE_OVERLAP",
                    "influence_state": "NOT_INFERRED",
                },
                {"label": "Uncertain historical role", "match_state": "UNKNOWN"},
            ],
        },
        "own_group_peer_comparison": {
            "status": "TARGET_NON_SUBSTANTIVE",
            "reason": "target ballot is abstention or absence, not JAA/EI",
            "target_group_code": "kok",
            "target_response": "POISSA",
            "source_id": "eduskunta:ballots:1",
            "source_sha256": "a" * 64,
            "peer_count": 0,
        },
    }
    unnamed = deepcopy(action)
    unnamed["actor"] = {
        "actor_id": None,
        "person_id": None,
        "name": "Named Only",
        "identity_state": "SOURCE_NAME_ONLY_UNJOINED",
    }
    unnamed["source"] = {
        "url": "https://api.eduskunta.fi/api/v1/search",
        "record_locator": "api.eduskunta.fi/search/puheenvuoro/PUH 1/2024/1/1/1",
    }
    unnamed.update(
        kind="SPEECH",
        role_label="SPEAKER",
        action_kind={"value": "SPEECH_DELIVERED", "basis": "SPEECH_REGISTER"},
        source_text="<script>not executable</script>",
        institutional_result={"state": "RECORDED"},
        influence={"state": "INFLUENCE_NOT_INFERRED"},
    )
    unnamed.pop("own_group_peer_comparison", None)
    comparable = deepcopy(action)
    comparable["actor"] = {
        "actor_id": "mp-2",
        "person_id": "2",
        "name": "Second Person",
        "identity_state": "EXACT_PERSON_ID",
    }
    comparable["raw_response"] = "JAA"
    comparable["own_group_peer_comparison"] = {
        "status": "COMPARABLE",
        "target_group_code": "kok",
        "target_response": "JAA",
        "peer_count": 4,
        "peer_jaa": 3,
        "peer_ei": 1,
        "peer_majority": "JAA",
        "matches_peer_majority": True,
        "source_id": "eduskunta:ballots:1",
        "source_sha256": "b" * 64,
    }
    return {
        "query": {"mode": "EXACT_MATTER", "actor": None},
        "matter": {
            "matter_id": "HE 1/2024 vp",
            "subject_title": "Paikallisen sopimisen esitys",
            "subject_title_state": "COMMON_OBJECT_TITLE",
            "subject_title_candidates": ["Paikallisen sopimisen esitys"],
            "objects": [
                {
                    "kind": "SPEECH",
                    "title": "Paikallisen sopimisen esitys",
                }
            ],
            "vote_sources": [
                {
                    "action_date": "2024-01-02",
                    "title": "First alternative",
                    "matter_text": "Bundle context",
                    "source_url": "https://example.test/vote",
                    "published_disposition": {
                        "state": "JAA_SIDE_PREVAILED_AT_THIS_VOTE",
                        "scope": "THIS_VOTE_ONLY",
                        "jaa": 10,
                        "ei": 5,
                        "final_adoption": {
                            "state": "UNRESOLVED",
                            "limitation": (
                                "Published JAA/EI totals decide only which side prevailed at this "
                                "vote. Final adoption requires a final-stage/order witness."
                            ),
                        },
                    },
                }
            ],
        },
        "interventions": [action, unnamed, comparable],
        "coverage": [{"kind": "VOTE", "declared_scope": "loaded slice", "state": "DECLARED"}],
        "missingness": [{"code": "EXAMPLE_GAP", "reason": "Example missingness"}],
        "limits": ["Vote-stage prevalence from published totals is not final adoption."],
    }


def test_matter_hub_bootstrap_nav_exposes_takaisin_like_journeys() -> None:
    """D2: matter hubs need the same Takaisin affordance as journey pages."""

    hub = render_matter_html(
        _packet(),
        packet_url="records.json",
        back_url="../index.html",
        lookup_url="../lookup.html",
    )
    assert "Takaisin" in hub and "../index.html" in hub
    assert "Haku" in hub and "../lookup.html" in hub
    # Standalone render without site nav stays free of a dangling back link.
    bare = render_matter_html(_packet())
    assert ">Takaisin</a>" not in bare


def _default_visible_text(markup: str) -> str:
    """Closed bodies are hidden; their summary labels remain visible."""
    from html.parser import HTMLParser

    class Reader(HTMLParser):
        def __init__(self):
            super().__init__()
            self.stack = []
            self.text = []

        def handle_starttag(self, tag, attrs):
            attrs = dict(attrs)
            hidden = bool(self.stack and self.stack[-1][1])
            if self.stack and self.stack[-1][0] == "details" and self.stack[-1][2] and tag != "summary":
                hidden = True
            hidden |= tag in {"head", "script", "style", "noscript", "option"} or "hidden" in attrs
            if tag not in {"meta", "link", "input", "br", "hr", "img"}:
                self.stack.append((tag, hidden, tag == "details" and "open" not in attrs))

        def handle_endtag(self, tag):
            for index in range(len(self.stack) - 1, -1, -1):
                if self.stack[index][0] == tag:
                    self.stack = self.stack[:index]
                    break

        def handle_data(self, data):
            if self.stack and not self.stack[-1][1] and (self.stack[-1][0] != "details" or not self.stack[-1][2]):
                self.text.append(data)

    reader = Reader()
    reader.feed(markup)
    return " ".join(" ".join(reader.text).split())


def test_matter_hub_is_compact_and_person_pages_keep_literal_evidence(tmp_path):
    packet = _packet()
    hub = render_matter_html(packet, packet_url="records.json")
    assert '<article class="intervention"' not in hub
    assert 'id="actor-filter"' in hub and "class='person-links'" in hub
    assert "rebaseRelativeLinks(main, response.url)" in hub
    assert "loadGeneration" in hub and "henkilo" in hub
    assert "<h1>HE 1/2024 vp</h1>" in hub
    assert "vain nimi, identiteetti avoin" in hub
    assert _default_visible_text(hub).count("Paikallisen sopimisen esitys") == 1
    assert 'download href="records.json"' in hub
    assert "<noscript>" in hub and "staattinen sivu" in hub
    assert "INFLUENCE_NOT_INFERRED" not in _default_visible_text(hub)

    site = write_matter_site(packet, tmp_path / "matter.html", packet_url="records.json")
    assert site["actor_count"] == 3 and site["intervention_count"] == 3
    first = (tmp_path / "matter/people/actor_mp-1.html").read_text()
    visible = _default_visible_text(first)
    assert "First alternative" in visible and "POISSA" in visible
    assert "Miten äänesti?" in visible
    assert "INFLUENCE_NOT_INFERRED" not in visible and "tiiviste" not in visible
    assert visible.count("Paikallisen sopimisen esitys") == 1
    assert "alkuperäisen vastauksen lähdeversio on vahvistamatta" in first
    assert "../../records.json" in first
    record = (tmp_path / "matter/records/0000_vote_1.html").read_text()
    assert "https://example.test/vote" in record
    assert "2023-04-01" in record and "Uncertain historical role" not in record
    assert "Osa lähteen roolijaksoista jäi liian epätarkoiksi" in record
    assert "Oman ryhmän vertailu" in record
    assert "TARGET_NON_SUBSTANTIVE" in record
    assert "INFLUENCE_NOT_INFERRED" not in _default_visible_text(record)

    relative_site = write_matter_site(packet, tmp_path / "shard.html", packet_url="../shared/records.json")
    assert relative_site["hub"] == "shard.html"
    assert "../../../shared/records.json" in (tmp_path / "shard/people/actor_mp-1.html").read_text()

    unnamed = (tmp_path / "matter/people/name-only_Named_Only.html").read_text()
    assert "Mitä hän sanoi?" in _default_visible_text(unnamed)
    assert "Puhui" not in _default_visible_text(unnamed)
    assert "<article" not in unnamed
    assert '&lt;script&gt;' not in _default_visible_text(unnamed)
    source = (tmp_path / "matter/records/0001_vote_1.html").read_text()
    assert "&lt;script&gt;not executable&lt;/script&gt;" in source
    assert '<script>not executable</script>' not in source
    assert "PUH 1/2024/1/1/1" in source
    assert 'href="https://api.eduskunta.fi/api/v1/search">Avaa alkuperäinen lähde' not in source
    assert "Yleinen hakuosoite" in source
    assert "INFLUENCE_NOT_INFERRED" not in _default_visible_text(source)
    second_record = (tmp_path / "matter/records/0002_vote_1.html").read_text()
    assert "Vertailukelpoiset muut JAA/EI-äänet: 4" in second_record
    assert "sama kuin ryhmän enemmistö" in second_record

    summaries = compact_actor_summaries(packet["interventions"])
    assert {row["record_count"] for row in summaries} == {1}
    assert all(row["kinds"] for row in summaries)


def test_actor_without_records_is_an_explicit_bounded_empty_result():
    packet = _packet()
    packet["interventions"] = []
    packet["query"]["actor"] = {
        "actor_id": "mp-missing",
        "display_name": "Unresolved Person",
        "unresolved": True,
    }
    page = render_matter_html(packet)
    assert "Unresolved Person" in page
    assert "identiteetti jäi avoimeksi" in page
    assert "Tyhjä tulos ei osoita" in page


def test_matter_index_is_record_exploration_and_escapes_source_titles():
    page = render_matter_index(
        [
            {
                "path": "matter.html",
                "title": "<Law>",
                "choice_label": "HE 1/2024 vp — <Law>",
                "matter_id": "HE 1/2024 vp",
                "subject_title": "<Law>",
                "intervention_count": 2,
                "description": "Two records",
            }
        ]
    )
    assert "matter.html" in page
    assert "&lt;Law&gt;" in page
    assert "<Law>" not in page
    assert "lähdekirjausta" not in _default_visible_text(page)
    assert "<article" not in page


def test_ambiguous_object_titles_keep_identifier_without_arbitrary_subject():
    packet = _packet()
    packet["matter"]["subject_title"] = None
    packet["matter"]["subject_title_state"] = "AMBIGUOUS_OBJECT_TITLES"
    packet["matter"]["subject_title_candidates"] = ["Title A", "Title B"]
    packet["matter"]["objects"] = [
        {"kind": "SPEECH", "title": "Title A"},
        {"kind": "SPEECH", "title": "Title B"},
    ]
    page = render_matter_html(packet)
    assert "HE 1/2024 vp" in page
    assert "otsikot eivät ole yhteneviä" in page
    assert "Title A" not in page
    assert "Title B" not in page


def test_legacy_initiative_authored_decodes_on_person_page_without_drafting_claim():
    packet = _packet()
    packet["interventions"] = [
        {
            "actor": {
                "actor_id": "mp-3",
                "person_id": "3",
                "name": "Signer",
                "identity_state": "EXACT_PERSON_ID",
            },
            "source": {"url": "https://example.test/la", "record_locator": "la:1"},
            "kind": "LEGISLATIVE_INITIATIVE",
            "role_label": "FIRST_SIGNATURE",
            "signature": {"state": "SOURCE_ROLE", "value": "FIRST_SIGNATURE"},
            "action_kind": {"value": "INITIATIVE_AUTHORED", "basis": "SOURCE_RECORD_KIND_AND_ROLE"},
            "drafting": {"state": "UNRESOLVED", "value": None},
            "action_date": "2024-01-01",
            "action_date_basis": "SIGNATURE_DATE",
            "object_id": "la:1",
            "title": "LA example",
            "evidence_ids": ["la-e"],
            "source_text": "Aloite",
        }
    ]
    page = render_matter_person_html(packet, "actor:mp-3")
    assert "FIRST_SIGNATORY" not in _default_visible_text(page)
    assert "INITIATIVE_AUTHORED" not in page
    assert "Ensimmäinen allekirjoittaja" in page


def test_person_answers_choices_instead_of_listing_every_activity():
    packet = _packet()
    vote = deepcopy(packet["interventions"][2])
    vote.update(title="Hyväksyminen / Hylkääminen", event_id="1", raw_response="EI")
    vote["actor"] = packet["interventions"][0]["actor"]
    vote["own_group_peer_comparison"] = {}
    routine = deepcopy(vote)
    routine.update(title="Lausumaehdotus / vastaehdotus", event_id="2")
    speech = deepcopy(packet["interventions"][1])
    speech["actor"] = vote["actor"]
    speech["title"] = "Long official matter title repeated by every speech"
    speech["source_text"] = "Vastustan tätä ehdotusta sen vaikutusten vuoksi."
    packet["interventions"] = [speech, speech, routine, vote]
    packet["matter"]["vote_sources"][0]["vote_id"] = "1"
    page = render_matter_person_html(packet, "actor:mp-1")
    visible = _default_visible_text(page)
    assert "Hyväksyminen / Hylkääminen" in visible and "EI" in visible
    assert "JAA 10–5" in visible
    assert "Lausumaehdotus / vastaehdotus" not in visible
    assert "Long official matter title" not in visible
    assert "Vastustan" not in visible  # participation is not a compiled stance
    assert "Puhui" not in visible and "Puheenvuoro" not in visible
    assert "Hyväksytty" not in visible  # a vote winner is not enactment
    assert "<article" not in page
    # Every omitted primary record still has a source link; no data is discarded.
    assert page.count("../records/") == 4
    assert "Vastustan tätä ehdotusta" in page


def test_group_departure_appears_only_for_a_comparable_retained_baseline():
    packet = _packet()
    packet["interventions"] = [deepcopy(packet["interventions"][2]) for _ in range(4)]
    for index, row in enumerate(packet["interventions"]):
        row["title"] = f"Alternative {index}"
        row["own_group_peer_comparison"]["status"] = "TARGET_NON_SUBSTANTIVE"
    packet["interventions"][1]["own_group_peer_comparison"].update(
        status="COMPARABLE", matches_peer_majority=False, peer_majority="EI",
    )
    page = render_matter_person_html(packet, "actor:mp-2")
    visible = _default_visible_text(page)
    assert "Alternative 1" in visible and "ryhmä EI" in visible
    assert "Alternative 0" not in visible and "Alternative 2" not in visible
    packet["interventions"][1]["own_group_peer_comparison"]["status"] = "TIED_PEERS"
    assert "ryhmä EI" not in _default_visible_text(render_matter_person_html(packet, "actor:mp-2"))


def test_result_is_not_attached_to_a_same_id_with_a_different_event_date():
    packet = _packet()
    packet["interventions"][0]["event_id"] = "1"
    packet["matter"]["vote_sources"][0].update(vote_id="1", action_date="2024-02-01")
    page = render_matter_person_html(packet, "actor:mp-1")
    assert "JAA 10–5" not in _default_visible_text(page)

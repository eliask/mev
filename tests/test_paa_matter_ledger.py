"""Exact matter-to-actor ledger tests."""

import json
import sqlite3
from pathlib import Path

from paa.matter_ledger import (
    LEDGER_SCHEMA_VERSION,
    MatterLedgerError,
    build_matter_ledger,
    matter_subject_title_from_objects,
    official_object_matches_matter,
    render_matter_html,
    search_matter_candidates,
)
from paa.store import connect


def _object(
    object_id: str,
    matter_id: str,
    *,
    kind: str = "LEGISLATIVE_INITIATIVE",
    title: str = "Official matter",
    text: str = "Official source text",
    action_date: str | None = "2024-01-02",
    authors: list[dict] | None = None,
    source_id: str = "SRC-EDUSKUNTA-VASKI",
    evidence_id: str = "source-record-e",
) -> dict:
    return {
        "object_id": object_id,
        "matter_id": matter_id,
        "kind": kind,
        "title": title,
        "text": text,
        "action_date": action_date,
        "action_date_basis": "SIGNATURE_DATE" if action_date else None,
        "publication_date": "2024-01-03",
        "source_id": source_id,
        "record_locator": f"fixture/{object_id}",
        "url": "https://example.test/source",
        "authors": authors or [],
        "evidence_ids": [evidence_id],
        "evidence": [{"evidence_id": evidence_id, "quote": text, "record_locator": f"fixture/{object_id}"}],
        "disposition": {"state": "UNRESOLVED", "evidence_ids": []},
    }


def _actor(actor_id: str, person_id: str, name: str) -> tuple:
    return (actor_id, name, name.casefold(), None, person_id, "MP_UNIQUE")


def _seed_db(tmp_path: Path) -> sqlite3.Connection:
    conn = connect(tmp_path / "ledger.sqlite")
    conn.executemany(
        "INSERT INTO actors(actor_id, display_name, name_key, birth_year, person_id, identity_status) VALUES (?, ?, ?, ?, ?, ?)",
        [_actor("mp-10", "10", "Ada Author"), _actor("mp-11", "11", "Co Signer")],
    )
    conn.executemany(
        "INSERT INTO mp_periods(person_id, kind, label, start_date, end_date, precision) VALUES (?, ?, ?, ?, ?, ?)",
        [
            ("10", "Edustajatoimi", "Kansanedustaja", "2023-04-01", "2025-04-01", "day"),
            ("10", "Eduskuntaryhma", "Esimerkkiryhmä", "2023-04-01", "2025-04-01", "day"),
        ],
    )
    return conn


def _insert_object(conn: sqlite3.Connection, obj: dict) -> None:
    conn.execute(
        "INSERT INTO official_objects(object_id, json) VALUES (?, ?)",
        (obj["object_id"], json.dumps(obj, ensure_ascii=False)),
    )


def test_exact_matter_join_keeps_first_signature_and_cosignature_distinct(tmp_path: Path) -> None:
    conn = _seed_db(tmp_path)
    try:
        _insert_object(
            conn,
            _object(
                "eduskunta:LA 1/2024 vp",
                "LA 1/2024 vp",
                authors=[
                    {"person_id": "10", "name": "Ada Author", "role": "AUTHOR", "evidence_ids": ["author-e"]},
                    {"person_id": "11", "name": "Co Signer", "role": "COSIGNER", "evidence_ids": ["author-e"]},
                ],
            ),
        )
        # Similar title, different official matter: it must not be joined.
        _insert_object(
            conn,
            _object("eduskunta:LA 2/2024 vp", "LA 2/2024 vp", title="Official matter"),
        )
        conn.commit()
        before = conn.total_changes
        packet = build_matter_ledger(conn, "LA 1/2024 vp")
        assert conn.total_changes == before
        assert packet["schema_version"] == LEDGER_SCHEMA_VERSION
        assert packet["matter"]["object_count"] == 1
        assert packet["matter"]["objects"][0]["evidence"][0]["quote"] == "Official source text"
        assert {item["role_label"] for item in packet["interventions"]} == {"FIRST_SIGNATURE", "CO_SIGNATURE"}
        first = next(item for item in packet["interventions"] if item["role_label"] == "FIRST_SIGNATURE")
        assert first["action_kind"]["value"] == "FIRST_SIGNATORY"
        assert first["signature"]["value"] == "FIRST_SIGNATURE"
        co = next(item for item in packet["interventions"] if item["role_label"] == "CO_SIGNATURE")
        assert co["action_kind"]["value"] == "INITIATIVE_COSIGNED"
        assert co["signature"]["value"] == "CO_SIGNATURE"
        assert first["source_evidence"][0]["quote"] == "Official source text"
        assert first["drafting"]["state"] == "UNRESOLVED"
        # Legacy retained packets may still store INITIATIVE_AUTHORED for the
        # same first-signature role; decode without inventing drafting.
        from paa.matter_ledger import decode_ledger_action_kind

        legacy = decode_ledger_action_kind(
            {"value": "INITIATIVE_AUTHORED", "basis": "SOURCE_RECORD_KIND_AND_ROLE"},
            role_label="FIRST_SIGNATURE",
            signature_value="FIRST_SIGNATURE",
            kind="LEGISLATIVE_INITIATIVE",
        )
        assert legacy["value"] == "FIRST_SIGNATORY"
        assert legacy["decoded_from"] == "INITIATIVE_AUTHORED"
        from paa.matter_ledger import project_retained_matter_packet_for_current_emit

        retained = {
            "schema_version": "paa.matter-ledger.v3",
            "matter": {"matter_id": "LA 1/2024 vp"},
            "interventions": [
                {
                    "kind": "LEGISLATIVE_INITIATIVE",
                    "role_label": "FIRST_SIGNATURE",
                    "action_kind": {
                        "value": "INITIATIVE_AUTHORED",
                        "basis": "SOURCE_RECORD_KIND_AND_ROLE",
                    },
                }
            ],
        }
        projected = project_retained_matter_packet_for_current_emit(
            retained, retained_bytes_sha256="a" * 64
        )
        assert projected["schema_version"] == LEDGER_SCHEMA_VERSION
        assert projected["interventions"][0]["action_kind"]["value"] == "FIRST_SIGNATORY"
        assert projected["retained_source_codec"]["source_schema_version"] == "paa.matter-ledger.v3"
        assert first["role_intervals"]["state"] == "EXACT_DATE_OVERLAP"
        assert "COMMITTEE_MEMBERSHIP_NOT_LOADED" in {item["code"] for item in packet["missingness"]}
    finally:
        conn.close()


def test_actor_filter_uses_exact_person_identity(tmp_path: Path) -> None:
    conn = _seed_db(tmp_path)
    try:
        _insert_object(
            conn,
            _object(
                "eduskunta:KK 1/2024 vp",
                "KK 1/2024 vp",
                kind="WRITTEN_QUESTION",
                authors=[
                    {"person_id": "10", "name": "Ada Author", "role": "AUTHOR", "evidence_ids": ["question-e"]},
                    {"person_id": "11", "name": "Co Signer", "role": "COSIGNER", "evidence_ids": ["question-e"]},
                ],
            ),
        )
        conn.commit()
        packet = build_matter_ledger(conn, "KK 1/2024 vp", actor_id="mp-11")
        assert packet["query"]["actor"]["match_basis"] == "EXACT_ACTOR_ID"
        assert len(packet["interventions"]) == 1
        assert packet["interventions"][0]["actor"]["person_id"] == "11"
    finally:
        conn.close()


def test_unresolved_actor_filter_does_not_return_unfiltered_matter(tmp_path: Path) -> None:
    conn = _seed_db(tmp_path)
    try:
        _insert_object(
            conn,
            _object(
                "eduskunta:KK 2/2024 vp",
                "KK 2/2024 vp",
                kind="WRITTEN_QUESTION",
                authors=[{"person_id": "10", "name": "Ada Author", "role": "AUTHOR"}],
            ),
        )
        conn.commit()
        packet = build_matter_ledger(conn, "KK 2/2024 vp", actor_id="mp-not-in-db")
        assert packet["interventions"] == []
        assert "ACTOR_NOT_FOUND" in {item["code"] for item in packet["missingness"]}
        assert "NO_EXACT_ACTOR_INTERVENTION" in {item["code"] for item in packet["missingness"]}
    finally:
        conn.close()


def test_name_only_answer_is_retained_without_actor_promotion(tmp_path: Path) -> None:
    conn = _seed_db(tmp_path)
    try:
        _insert_object(
            conn,
            _object(
                "eduskunta:KKV 1/2024 vp",
                "KK 1/2024 vp",
                kind="GOVERNMENT_ANSWER",
                authors=[{"name": "Tytti Tuppurainen", "role": "RESPONDENT", "evidence_ids": ["answer-e"]}],
            ),
        )
        conn.commit()
        packet = build_matter_ledger(conn, "KK 1/2024 vp")
        answer = packet["interventions"][0]
        assert answer["actor"]["actor_id"] is None
        assert answer["actor"]["identity_state"] == "SOURCE_NAME_ONLY_UNJOINED"
        assert "SOURCE_PERSON_ID_MISSING" in {item["code"] for item in packet["missingness"]}
    finally:
        conn.close()


def test_source_actor_id_is_exactly_joined_for_non_document_action(tmp_path: Path) -> None:
    conn = _seed_db(tmp_path)
    conn.execute(
        "INSERT INTO actors(actor_id, display_name, name_key, birth_year, person_id, identity_status) VALUES (?, ?, ?, ?, ?, ?)",
        _actor("mp-1514", "1514", "Miko Bergbom"),
    )
    conn.execute(
        "INSERT INTO mp_periods(person_id, kind, label, start_date, end_date, precision) VALUES (?, ?, ?, ?, ?, ?)",
        ("1514", "Edustajatoimi", "Edustajatoimi", "2023-04-05", None, "day"),
    )
    try:
        _insert_object(
            conn,
            _object(
                "pirha:9112/2023",
                "9112/2023",
                kind="RESIGN_ROLE",
                authors=[{"actor_id": "mp-1514", "name": "Miko Bergbom", "role": "ACTOR", "evidence_ids": ["resign-e"]}],
            ),
        )
        conn.commit()
        packet = build_matter_ledger(conn, "9112/2023")
        actor = packet["interventions"][0]["actor"]
        assert actor["identity_state"] == "EXACT_SOURCE_ACTOR_ID"
        assert actor["person_id"] == "1514"
        assert packet["interventions"][0]["role_intervals"]["state"] == "EXACT_DATE_OVERLAP"
    finally:
        conn.close()


def test_matter_membership_ignores_json_whitespace_and_nested_ids(tmp_path: Path) -> None:
    conn = _seed_db(tmp_path)
    try:
        matter = "HE 12/2024 vp"
        variants = [
            {"object_id": "spaced", "matter_id": matter, "kind": "SPEECH", "title": "spaced", "authors": []},
            json.loads('{"object_id":"compact","matter_id":"HE 12/2024 vp","kind":"SPEECH","title":"compact","authors":[]}'),
            json.loads(
                '{\n  "object_id": "newline",\n  "matter_id":\n  "HE 12/2024 vp",\n  "kind": "SPEECH",\n  "title": "newline",\n  "authors": []\n}'
            ),
            {
                "object_id": "nested-only",
                "kind": "SPEECH",
                "title": "nested",
                "authors": [],
                "topic": {"related_matter_id": matter},
            },
            {
                "object_id": "other-matter",
                "matter_id": "HE 9/2023 vp",
                "kind": "SPEECH",
                "title": "other",
                "authors": [],
                "note": matter,
            },
        ]
        # Persist with distinct serializers so the SQL gate cannot rely on one dump style.
        conn.execute(
            "INSERT INTO official_objects(object_id, json) VALUES (?, ?)",
            ("spaced", json.dumps(variants[0], ensure_ascii=False, indent=2)),
        )
        conn.execute(
            "INSERT INTO official_objects(object_id, json) VALUES (?, ?)",
            ("compact", json.dumps(variants[1], ensure_ascii=False, separators=(",", ":"))),
        )
        conn.execute(
            "INSERT INTO official_objects(object_id, json) VALUES (?, ?)",
            (
                "newline",
                '{\n"object_id": "newline",\n"matter_id":\n"HE 12/2024 vp",\n"kind": "SPEECH",\n"title": "newline",\n"authors": []\n}',
            ),
        )
        conn.execute(
            "INSERT INTO official_objects(object_id, json) VALUES (?, ?)",
            ("nested-only", json.dumps(variants[3], ensure_ascii=False)),
        )
        conn.execute(
            "INSERT INTO official_objects(object_id, json) VALUES (?, ?)",
            ("other-matter", json.dumps(variants[4], ensure_ascii=False)),
        )
        # Escaped-slash serializer: '/' may appear as '\/' in stored JSON text.
        escaped = '{"object_id":"escaped","matter_id":"HE 12\\/2024 vp","kind":"SPEECH","title":"escaped","authors":[]}'
        conn.execute(
            "INSERT INTO official_objects(object_id, json) VALUES (?, ?)",
            ("escaped", escaped),
        )
        conn.commit()
        packet = build_matter_ledger(conn, matter)
        ids = {item["object_id"] for item in packet["matter"]["objects"]}
        assert ids == {"spaced", "compact", "newline", "escaped"}
        assert official_object_matches_matter(variants[0], matter)
        assert not official_object_matches_matter(variants[3], matter)
        assert official_object_matches_matter(json.loads(escaped), matter)
    finally:
        conn.close()


def test_forsgren_source_name_is_not_truncated(tmp_path: Path) -> None:
    conn = _seed_db(tmp_path)
    conn.execute(
        "INSERT INTO actors(actor_id, display_name, name_key, birth_year, person_id, identity_status) VALUES (?, ?, ?, ?, ?, ?)",
        _actor("mp-21", "21", "Bella Forsgrén"),
    )
    try:
        vote = {
            "aanestys_id": "51081",
            "year": 2023,
            "session_date": "2023-09-27",
            "number": 3,
            "title": "Lausumaehdotus, mietintö / Bella Forsgrén 1, Ilmari Nurminen 1",
            "lisa": "",
            "kohta": "HE 9/2023",
            "jaa": 94,
            "ei": 75,
            "tyhjaa": 0,
            "poissa": 30,
            "yhteensa": 199,
            "url": "https://example.test/vote/51081",
            "ptk": "PTK 32/2023 vp",
            "matter": "HE 9/2023 vp",
            "mitatoity": 0,
            "json": "{}",
        }
        columns = ", ".join(vote)
        placeholders = ", ".join("?" for _ in vote)
        conn.execute(f"INSERT INTO vote_events({columns}) VALUES ({placeholders})", tuple(vote.values()))
        conn.commit()
        packet = build_matter_ledger(conn, "HE 9/2023 vp")
        names = {
            item["actor"]["source_display_name"]
            for item in packet["interventions"]
            if item["kind"] == "FORMAL_ALTERNATIVE"
        }
        assert "Bella Forsgrén" in names
        assert "Bella Forsgré" not in names
        bella = next(
            item
            for item in packet["interventions"]
            if item["kind"] == "FORMAL_ALTERNATIVE" and item["actor"]["source_display_name"] == "Bella Forsgrén"
        )
        assert bella["actor"]["identity_state"] == "EXACT_SOURCE_DISPLAY_NAME"
        assert bella["actor"]["person_id"] == "21"
    finally:
        conn.close()


def test_matter_subject_title_requires_common_object_title() -> None:
    common = matter_subject_title_from_objects(
        [
            {"title": "Oulaskankaan sairaalassa tulee mahdollistaa ympärivuorokautinen perusterveydenhuollon päivystys"},
            {"title": "Oulaskankaan sairaalassa tulee mahdollistaa ympärivuorokautinen perusterveydenhuollon päivystys"},
        ]
    )
    assert common["subject_title_state"] == "COMMON_OBJECT_TITLE"
    assert common["subject_title"].startswith("Oulaskankaan")
    ambiguous = matter_subject_title_from_objects(
        [{"title": "Title A"}, {"title": "Title B"}]
    )
    assert ambiguous["subject_title"] is None
    assert ambiguous["subject_title_state"] == "AMBIGUOUS_OBJECT_TITLES"
    assert ambiguous["subject_title_candidates"] == ["Title A", "Title B"]
    assert matter_subject_title_from_objects([])["subject_title_state"] == "MISSING"


def test_genitive_chair_title_with_polarity_joins_roster_after_exact_miss(tmp_path: Path) -> None:
    conn = _seed_db(tmp_path)
    conn.execute(
        "INSERT INTO actors(actor_id, display_name, name_key, birth_year, person_id, identity_status) VALUES (?, ?, ?, ?, ?, ?)",
        _actor("mp-1537", "1537", "Hanna-Leena Mattila"),
    )
    try:
        _insert_object(
            conn,
            _object(
                "eduskunta:KAA 2/2025 vp",
                "KAA 2/2025 vp",
                kind="SPEECH",
                title="Oulaskankaan sairaalassa tulee mahdollistaa ympärivuorokautinen perusterveydenhuollon päivystys",
                authors=[],
            ),
        )
        vote = {
            "aanestys_id": "55560",
            "year": 2025,
            "session_date": "2025-10-10",
            "number": 1,
            "title": "Mietintö JAA / Hanna-Leena Mattilan ehdotus (vl) EI",
            "lisa": "",
            "kohta": "KAA 2/2025",
            "jaa": 95,
            "ei": 85,
            "tyhjaa": 0,
            "poissa": 19,
            "yhteensa": 199,
            "url": "https://example.test/vote/55560",
            "ptk": "PTK 1/2025 vp",
            "matter": "KAA 2/2025 vp",
            "mitatoity": 0,
            "json": "{}",
        }
        columns = ", ".join(vote)
        placeholders = ", ".join("?" for _ in vote)
        conn.execute(f"INSERT INTO vote_events({columns}) VALUES ({placeholders})", tuple(vote.values()))
        conn.commit()
        packet = build_matter_ledger(conn, "KAA 2/2025 vp")
        assert packet["matter"]["subject_title_state"] == "COMMON_OBJECT_TITLE"
        assert packet["matter"]["subject_title"].startswith("Oulaskankaan")
        formal = [
            item
            for item in packet["interventions"]
            if item["kind"] == "FORMAL_ALTERNATIVE"
        ]
        assert len(formal) == 1
        assert formal[0]["actor"]["source_display_name"] == "Hanna-Leena Mattilan"
        assert formal[0]["actor"]["identity_state"] == "INFLECTED_SOURCE_DISPLAY_NAME"
        assert formal[0]["actor"]["person_id"] == "1537"
        assert formal[0]["chair_title_attribution"]["side"] == "EI"
        assert formal[0]["institutional_result"]["state"] == "LOST_AT_THIS_VOTE"
        published = formal[0]["institutional_result"]["published_disposition"]
        assert published["jaa"] == 95
        assert published["ei"] == 85
        named = packet["matter"]["vote_sources"][0]
        assert named["vote_id"] == "55560"
        assert named["chair_title_proposers"][0]["source_display_name"] == "Hanna-Leena Mattilan"
    finally:
        conn.close()


def test_both_side_chair_title_proposers_keep_side_local_result(tmp_path: Path) -> None:
    conn = _seed_db(tmp_path)
    conn.execute(
        "INSERT INTO actors(actor_id, display_name, name_key, birth_year, person_id, identity_status) VALUES (?, ?, ?, ?, ?, ?)",
        _actor("mp-1537", "1537", "Hanna-Leena Mattila"),
    )
    conn.execute(
        "INSERT INTO actors(actor_id, display_name, name_key, birth_year, person_id, identity_status) VALUES (?, ?, ?, ?, ?, ?)",
        _actor("mp-1600", "1600", "Nina Merinen"),
    )
    try:
        vote = {
            "aanestys_id": "53546",
            "year": 2024,
            "session_date": "2024-12-11",
            "number": 14,
            "title": "Lausumaehdotus, Hanna-Leena Mattila 1 / Nina Merinen 1",
            "lisa": "",
            "kohta": "HE 149/2024",
            "jaa": 80,
            "ei": 70,
            "tyhjaa": 0,
            "poissa": 20,
            "yhteensa": 170,
            "url": "https://example.test/vote/53546",
            "ptk": "PTK 1/2024 vp",
            "matter": "HE 149/2024 vp",
            "mitatoity": 0,
            "json": "{}",
        }
        columns = ", ".join(vote)
        placeholders = ", ".join("?" for _ in vote)
        conn.execute(f"INSERT INTO vote_events({columns}) VALUES ({placeholders})", tuple(vote.values()))
        conn.commit()
        packet = build_matter_ledger(conn, "HE 149/2024 vp")
        formal = {
            item["actor"]["source_display_name"]: item
            for item in packet["interventions"]
            if item["kind"] == "FORMAL_ALTERNATIVE"
        }
        assert set(formal) == {"Hanna-Leena Mattila", "Nina Merinen"}
        assert formal["Hanna-Leena Mattila"]["chair_title_attribution"]["side"] == "JAA"
        assert formal["Nina Merinen"]["chair_title_attribution"]["side"] == "EI"
        assert formal["Hanna-Leena Mattila"]["institutional_result"]["state"] == "PREVAILED_AT_THIS_VOTE"
        assert formal["Nina Merinen"]["institutional_result"]["state"] == "LOST_AT_THIS_VOTE"
        assert formal["Hanna-Leena Mattila"]["institutional_result"]["final_adoption"]["state"] == "UNRESOLVED"
        named = next(item for item in packet["matter"]["vote_sources"] if item["vote_id"] == "53546")
        assert {row["source_display_name"] for row in named["chair_title_proposers"]} == {
            "Hanna-Leena Mattila",
            "Nina Merinen",
        }
    finally:
        conn.close()


def test_chair_title_named_proposer_is_positive_formal_alternative(tmp_path: Path) -> None:
    conn = _seed_db(tmp_path)
    conn.execute(
        "INSERT INTO actors(actor_id, display_name, name_key, birth_year, person_id, identity_status) VALUES (?, ?, ?, ?, ?, ?)",
        _actor("mp-20", "20", "Hilkka Kemppi"),
    )
    conn.execute(
        "INSERT INTO mp_periods(person_id, kind, label, start_date, end_date, precision) VALUES (?, ?, ?, ?, ?, ?)",
        ("20", "Edustajatoimi", "Kansanedustaja", "2023-04-01", "2027-04-01", "day"),
    )
    try:
        vote = {
            "aanestys_id": "51079",
            "year": 2023,
            "session_date": "2023-09-27",
            "number": 2,
            "title": "Lausumaehdotus, mietintö / Hilkka Kemppi",
            "lisa": "",
            "kohta": "Hallituksen esitys eduskunnalle vammaispalvelulain voimaantulon lykkäämisestä",
            "jaa": 94,
            "ei": 75,
            "tyhjaa": 0,
            "poissa": 30,
            "yhteensa": 199,
            "url": "https://example.test/vote/51079",
            "ptk": "PTK 32/2023 vp",
            "matter": "HE 9/2023 vp",
            "mitatoity": 0,
            "json": "{}",
        }
        columns = ", ".join(vote)
        placeholders = ", ".join("?" for _ in vote)
        conn.execute(f"INSERT INTO vote_events({columns}) VALUES ({placeholders})", tuple(vote.values()))
        # Procedural twin: must not invent a named proposer.
        procedural = dict(vote)
        procedural.update(
            {
                "aanestys_id": "51077",
                "title": "Lakiehdotusten hyväksyminen / hylkääminen",
                "jaa": 92,
                "ei": 75,
                "yhteensa": 167,
            }
        )
        conn.execute(f"INSERT INTO vote_events({columns}) VALUES ({placeholders})", tuple(procedural.values()))
        conn.commit()
        packet = build_matter_ledger(conn, "HE 9/2023 vp")
        formal = [item for item in packet["interventions"] if item["kind"] == "FORMAL_ALTERNATIVE"]
        assert len(formal) == 1
        action = formal[0]
        assert action["actor"]["person_id"] == "20"
        assert action["actor"]["source_display_name"] == "Hilkka Kemppi"
        assert action["actor"]["name"] == "Hilkka Kemppi"
        assert action["action_kind"]["value"] == "ALTERNATIVE_PUBLICLY_ATTRIBUTED"
        assert action["institutional_result"]["state"] == "LOST_AT_THIS_VOTE"
        assert action["institutional_result"]["final_adoption"]["state"] == "UNRESOLVED"
        assert action["drafting"]["state"] == "UNRESOLVED"
        named_vote = next(item for item in packet["matter"]["vote_sources"] if item["vote_id"] == "51079")
        assert named_vote["chair_title_proposers"][0]["source_display_name"] == "Hilkka Kemppi"
        procedural_vote = next(item for item in packet["matter"]["vote_sources"] if item["vote_id"] == "51077")
        assert procedural_vote["chair_title_proposers"] == []
        hub = render_matter_html(packet)
        assert "Hilkka Kemppi" in hub
        assert "Nimetyt äänestysvaihtoehdot" in hub or "FORMAL_ALTERNATIVE" in hub
        from paa.matter_site import render_matter_person_html

        person_key = "actor:" + action["actor"]["actor_id"]
        person = render_matter_person_html(packet, person_key)
        # The compact row must retain the actual vote outcome, rather than a
        # prescribed verbose status heading or a claim about final enactment.
        assert "Hävisi tässä äänestyksessä" in person
        assert "Nimetty äänestysvaihtoehdossa" in person
        assert "Hilkka Kemppi" in person
    finally:
        conn.close()


def test_vote_retains_exact_package_and_raw_alternatives_without_component_claim(tmp_path: Path) -> None:
    conn = _seed_db(tmp_path)
    try:
        vote = {
            "aanestys_id": "9",
            "year": 2024,
            "session_date": "2024-02-03",
            "number": 1,
            "title": "Package option A / Package option B",
            "lisa": "",
            "kohta": "Package matter | Component one | Component two",
            "jaa": 1,
            "ei": 0,
            "tyhjaa": 0,
            "poissa": 0,
            "yhteensa": 1,
            "url": "https://example.test/vote",
            "ptk": "PTK 1/2024 vp",
            "matter": "HE 9/2024 vp",
            "mitatoity": 0,
            "json": "{}",
        }
        columns = ", ".join(vote)
        placeholders = ", ".join("?" for _ in vote)
        conn.execute(f"INSERT INTO vote_events({columns}) VALUES ({placeholders})", tuple(vote.values()))
        conn.execute(
            "INSERT INTO ballots VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("9", "10", "Ada", "Author", "author|ada", "x", "JAA"),
        )
        conn.commit()
        packet = build_matter_ledger(conn, "HE 9/2024 vp")
        action = packet["interventions"][0]
        assert action["kind"] == "VOTE"
        assert action["package_attribution"]["state"] == "EXACT_VOTE_MATTER"
        assert action["component_attribution"]["state"] == "UNRESOLVED"
        assert action["alternatives"]["alternatives"][0]["alternative_text"] == "Package option A"
        assert action["alternatives"]["alternatives"][1]["alternative_text"] == "Package option B"
        assert action["source"]["source_id"] == "SRC-EDUSKUNTA-SALIDBAANESTYSEDUSTAJA"
        assert action["role_intervals"]["state"] == "EXACT_DATE_OVERLAP"
        assert action["ballot_status"]["category"] == "CAST"
        assert action["action_kind"]["value"] == "VOTE_CAST"
        assert action["ballot_source_version"]["state"] == "NORMALIZED_ROW_NO_RAW_HASH"
    finally:
        conn.close()


def test_missing_action_date_does_not_use_publication_date(tmp_path: Path) -> None:
    conn = _seed_db(tmp_path)
    try:
        _insert_object(
            conn,
            _object(
                "eduskunta:LA 3/2024 vp",
                "LA 3/2024 vp",
                action_date=None,
                authors=[{"person_id": "10", "name": "Ada Author", "role": "AUTHOR"}],
            ),
        )
        conn.commit()
        packet = build_matter_ledger(conn, "LA 3/2024 vp")
        action = packet["interventions"][0]
        assert action["action_date"] is None
        assert action["role_intervals"]["state"] == "DATE_UNKNOWN"
        assert "ACTION_DATE_MISSING" in {item["code"] for item in packet["missingness"]}
    finally:
        conn.close()


def test_lexical_search_is_a_separate_candidate_projection(tmp_path: Path) -> None:
    conn = _seed_db(tmp_path)
    try:
        _insert_object(
            conn,
            _object("eduskunta:LA 4/2024 vp", "LA 4/2024 vp", title="Diesel verotus", text="Diesel verotus"),
        )
        conn.commit()
        packet = search_matter_candidates(conn, "diesel verotus")
        assert packet["query"]["mode"] == "LEXICAL_CANDIDATE"
        assert packet["candidates"][0]["candidate_state"] == "LEXICAL_CANDIDATE_NOT_EXACT_RELATION"
        assert "interventions" not in packet["candidates"][0]
        assert packet["missingness"][0]["code"] == "LEXICAL_SEARCH_NOT_EXACT_RELATION"
    finally:
        conn.close()


def test_html_consumer_escapes_source_content() -> None:
    html = render_matter_html(
        {
            "query": {"mode": "LEXICAL_CANDIDATE", "text": "<script>"},
            "candidates": [
                {
                    "matter_id": "LA <1>",
                    "records": [{"kind": "SPEECH", "title": "<unsafe>", "source_id": "SRC"}],
                }
            ],
        }
    )
    assert "&lt;script&gt;" in html
    assert "<script>" not in html


def test_actual_db_receipt_covers_three_matter_stages() -> None:
    db = Path("data/paa.sqlite")
    if not db.exists():
        return
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    try:
        cases = {
            "LA 1/2025 vp": {"kind": "LEGISLATIVE_INITIATIVE", "source": "SRC-EDUSKUNTA-VASKI"},
            "KK 1/2023 vp": {"kind": "WRITTEN_QUESTION", "source": "SRC-EDUSKUNTA-VASKI"},
            "KAA 1/2023 vp": {"kind": "VOTE", "source": "SRC-EDUSKUNTA-SALIDBAANESTYSEDUSTAJA"},
        }
        for matter_id, expected in cases.items():
            packet = build_matter_ledger(conn, matter_id)
            assert packet["matter"]["object_count"] > 0
            assert packet["interventions"]
            sources = {
                source
                for item in packet["interventions"]
                for source in [item.get("source", {}).get("source_id")]
            }
            if matter_id == "KAA 1/2023 vp":
                assert expected["source"] in sources
            else:
                assert expected["source"] in sources
            assert any(item.get("action_date") for item in packet["interventions"])
    finally:
        conn.close()


def test_invalid_query_shapes_are_rejected(tmp_path: Path) -> None:
    conn = _seed_db(tmp_path)
    try:
        try:
            build_matter_ledger(conn, "")
        except MatterLedgerError:
            pass
        else:
            raise AssertionError("empty matter ID should be rejected")
    finally:
        conn.close()


def _insert_vote(
    conn: sqlite3.Connection,
    *,
    vote_id: str,
    matter: str,
    jaa: int | None,
    ei: int | None,
    tyhjaa: int | None,
    poissa: int | None,
    yhteensa: int | None,
    title: str = "Option A / Option B",
) -> None:
    vote = {
        "aanestys_id": vote_id,
        "year": 2024,
        "session_date": "2024-02-03",
        "number": 1,
        "title": title,
        "lisa": "",
        "kohta": "Package matter",
        "jaa": jaa,
        "ei": ei,
        "tyhjaa": tyhjaa,
        "poissa": poissa,
        "yhteensa": yhteensa,
        "url": "https://example.test/vote",
        "ptk": "PTK 1/2024 vp",
        "matter": matter,
        "mitatoity": 0,
        "json": "{}",
    }
    columns = ", ".join(vote)
    placeholders = ", ".join("?" for _ in vote)
    conn.execute(f"INSERT INTO vote_events({columns}) VALUES ({placeholders})", tuple(vote.values()))


def test_fractional_published_count_cannot_be_truncated_to_a_complete_vote(tmp_path: Path) -> None:
    conn = _seed_db(tmp_path)
    try:
        _insert_vote(conn, vote_id='count-control', matter='HE count', jaa=1, ei=0, tyhjaa=0, poissa=0, yhteensa=1)
        conn.execute("INSERT INTO ballots VALUES (?, ?, ?, ?, ?, ?, ?)",
                     ('count-control', '10', 'Ada', 'Author', 'Example', 'Example', 'JAA'))
        valid = build_matter_ledger(conn, 'HE count')
        assert valid['matter']['vote_sources'][0]['completeness']['state'] == 'COMPLETE_FOR_DECLARED_VOTE'
        conn.execute("UPDATE vote_events SET jaa = ? WHERE aanestys_id = ?", (1.7, 'count-control'))
        invalid = build_matter_ledger(conn, 'HE count')
        assert invalid['matter']['vote_sources'][0]['completeness']['state'] == 'DECLARED_TOTAL_OR_BUCKET_UNKNOWN'
    finally:
        conn.close()


def test_duplicate_person_id_never_selects_first_actor(tmp_path: Path) -> None:
    conn = _seed_db(tmp_path)
    try:
        conn.execute(
            "INSERT INTO actors(actor_id, display_name, name_key, birth_year, person_id, identity_status) VALUES (?, ?, ?, ?, ?, ?)",
            _actor("mp-10-duplicate", "10", "Ada Historical Row"),
        )
        _insert_object(
            conn,
            _object(
                "eduskunta:LA duplicate",
                "LA duplicate",
                authors=[{"person_id": "10", "name": "Ada Author", "role": "AUTHOR"}],
            ),
        )
        conn.commit()
        packet = build_matter_ledger(conn, "LA duplicate")
        action = packet["interventions"][0]
        assert action["actor"]["identity_state"] == "PERSON_ID_AMBIGUOUS"
        assert action["actor"]["actor_id"] is None
        assert set(action["actor"]["candidate_actor_ids"]) == {"mp-10", "mp-10-duplicate"}

        by_actor = build_matter_ledger(conn, "LA duplicate", actor_id="mp-10")
        assert by_actor["interventions"] == []
        assert "NO_EXACT_ACTOR_INTERVENTION" in {item["code"] for item in by_actor["missingness"]}

        # A source actor ID is an independent disambiguator when it resolves.
        _insert_object(
            conn,
            _object(
                "eduskunta:LA duplicate-source-id",
                "LA duplicate-source-id",
                authors=[{"actor_id": "mp-10", "person_id": "10", "name": "Ada Author", "role": "AUTHOR"}],
            ),
        )
        conn.commit()
        exact = build_matter_ledger(conn, "LA duplicate-source-id", actor_id="mp-10")
        assert len(exact["interventions"]) == 1
        assert exact["interventions"][0]["actor"]["identity_state"] == "EXACT_SOURCE_ACTOR_ID"
    finally:
        conn.close()


def test_unknown_source_actor_id_is_not_promoted_by_person_id(tmp_path: Path) -> None:
    conn = _seed_db(tmp_path)
    try:
        _insert_object(
            conn,
            _object(
                "eduskunta:LA unknown-source",
                "LA unknown-source",
                authors=[{"actor_id": "mp-unknown", "person_id": "10", "name": "Ada Author", "role": "AUTHOR"}],
            ),
        )
        conn.commit()
        packet = build_matter_ledger(conn, "LA unknown-source")
        action = packet["interventions"][0]
        assert action["actor"]["identity_state"] == "SOURCE_ACTOR_ID_NO_ACTOR_ROW"
        assert action["actor"]["actor_id"] == "mp-unknown"
        assert action["actor"].get("unresolved") is True
        assert build_matter_ledger(conn, "LA unknown-source", actor_id="mp-10")["interventions"] == []
    finally:
        conn.close()


def test_imprecise_role_interval_does_not_prove_date_overlap(tmp_path: Path) -> None:
    conn = _seed_db(tmp_path)
    try:
        conn.execute(
            "INSERT INTO mp_periods(person_id, kind, label, start_date, end_date, precision) VALUES (?, ?, ?, ?, ?, ?)",
            ("11", "Eduskuntaryhma", "Unknown interval", None, None, "unknown"),
        )
        _insert_object(
            conn,
            _object(
                "eduskunta:LA imprecise",
                "LA imprecise",
                authors=[{"person_id": "11", "name": "Co Signer", "role": "AUTHOR"}],
            ),
        )
        conn.commit()
        action = build_matter_ledger(conn, "LA imprecise")["interventions"][0]
        assert action["role_intervals"]["state"] == "ROLE_PRECISION_INSUFFICIENT"
        assert action["role_intervals"]["roles"] == []
    finally:
        conn.close()


def test_imprecise_role_interval_is_reported_even_with_exact_overlap(tmp_path: Path) -> None:
    conn = _seed_db(tmp_path)
    try:
        conn.execute(
            "INSERT INTO mp_periods(person_id, kind, label, start_date, end_date, precision) VALUES (?, ?, ?, ?, ?, ?)",
            ("10", "VaaliPiiri", "Unknown district interval", None, None, "unknown"),
        )
        _insert_object(
            conn,
            _object(
                "eduskunta:LA mixed-precision",
                "LA mixed-precision",
                authors=[{"person_id": "10", "name": "Ada Author", "role": "AUTHOR"}],
            ),
        )
        conn.commit()
        action = build_matter_ledger(conn, "LA mixed-precision")["interventions"][0]
        assert action["role_intervals"]["state"] == "EXACT_DATE_OVERLAP"
        assert any("lack day precision" in item for item in action["role_intervals"]["limitations"])
    finally:
        conn.close()


def test_ballot_status_is_not_policy_action_and_bucket_reconciliation_is_exact(tmp_path: Path) -> None:
    conn = _seed_db(tmp_path)
    try:
        _insert_vote(conn, vote_id="9", matter="HE 9/2024 vp", jaa=1, ei=0, tyhjaa=0, poissa=1, yhteensa=2)
        conn.executemany(
            "INSERT INTO ballots VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                ("9", "10", "Ada", "Author", "author|ada", "x", "JAA"),
                ("9", "11", "Co", "Signer", "signer|co", "x", "POISSA"),
            ],
        )
        conn.commit()
        packet = build_matter_ledger(conn, "HE 9/2024 vp")
        assert packet["matter"]["vote_sources"][0]["completeness"]["state"] == "COMPLETE_FOR_DECLARED_VOTE"
        absent = next(item for item in packet["interventions"] if item["raw_response"] == "POISSA")
        assert absent["action_kind"]["value"] is None
        assert absent["action_kind"]["basis"] == "BALLOT_STATUS_NOT_ACTION"
        assert absent["ballot_status"]["category"] == "ABSENT"
    finally:
        conn.close()


def test_vote_completeness_does_not_turn_invalid_or_null_counts_into_complete(tmp_path: Path) -> None:
    conn = _seed_db(tmp_path)
    try:
        _insert_vote(conn, vote_id="10", matter="M mismatch", jaa=2, ei=0, tyhjaa=0, poissa=0, yhteensa=2)
        conn.execute("INSERT INTO ballots VALUES (?, ?, ?, ?, ?, ?, ?)", ("10", "10", "Ada", "Author", "x", "x", "JAA"))
        _insert_vote(conn, vote_id="11", matter="M invalid", jaa=1, ei=0, tyhjaa=0, poissa=0, yhteensa=1)
        conn.execute("INSERT INTO ballots VALUES (?, ?, ?, ?, ?, ?, ?)", ("11", "10", "Ada", "Author", "x", "x", "MAYBE"))
        _insert_vote(conn, vote_id="12", matter="M null", jaa=0, ei=0, tyhjaa=0, poissa=0, yhteensa=None)
        conn.commit()
        mismatch = build_matter_ledger(conn, "M mismatch")
        assert mismatch["matter"]["vote_sources"][0]["completeness"]["state"] == "INCOMPLETE_OR_UNRECONCILED"
        assert "VOTE_BALLOT_BUCKET_MISMATCH" in {item["code"] for item in mismatch["missingness"]}
        invalid = build_matter_ledger(conn, "M invalid")
        assert invalid["matter"]["vote_sources"][0]["completeness"]["state"] == "INVALID_BALLOT_RESPONSES"
        assert "BALLOT_RESPONSES_INVALID" in {item["code"] for item in invalid["missingness"]}
        null = build_matter_ledger(conn, "M null")
        assert null["matter"]["vote_sources"][0]["completeness"]["state"] == "NO_LOCAL_BALLOT_ROWS"
        assert null["matter"]["vote_sources"][0]["completeness"]["published_total"] is None
        null_codes = {item["code"] for item in null["missingness"]}
        assert "NO_BALLOT_ROWS_FOR_VOTE" in null_codes
        assert "VOTE_COMPLETENESS_UNKNOWN" in null_codes
    finally:
        conn.close()


def test_object_and_ballot_accounting_missingness_is_explicit(tmp_path: Path) -> None:
    conn = _seed_db(tmp_path)
    try:
        _insert_object(conn, _object("speech:no-actor", "M no actor", kind="SPEECH", authors=[]))
        _insert_object(conn, _object("eduskunta-vote-101", "M no event", kind="VOTE", authors=[]))
        _insert_object(conn, _object("unknown:object", "M unsupported", kind="UNSUPPORTED", authors=[]))
        _insert_vote(conn, vote_id="13", matter="M missing person", jaa=1, ei=0, tyhjaa=0, poissa=0, yhteensa=1)
        conn.execute("INSERT INTO ballots VALUES (?, ?, ?, ?, ?, ?, ?)", ("13", None, "Unknown", "Member", "x", "x", "JAA"))
        conn.commit()
        no_actor = build_matter_ledger(conn, "M no actor")
        assert no_actor["interventions"] == []
        assert "NO_SOURCE_ACTOR" in {item["code"] for item in no_actor["missingness"]}
        no_event = build_matter_ledger(conn, "M no event")
        assert "VOTE_EVENT_NOT_LOADED" in {item["code"] for item in no_event["missingness"]}
        unsupported = build_matter_ledger(conn, "M unsupported")
        codes = {item["code"] for item in unsupported["missingness"]}
        assert "UNSUPPORTED_OFFICIAL_OBJECT_KIND" in codes
        assert "NO_EXACT_MATTER_RECORD" not in codes
        missing_person = build_matter_ledger(conn, "M missing person")
        assert "BALLOT_PERSON_ID_MISSING" in {item["code"] for item in missing_person["missingness"]}
    finally:
        conn.close()


def test_coverage_unknown_and_zero_count_are_preserved(tmp_path: Path) -> None:
    conn = _seed_db(tmp_path)
    try:
        _insert_object(conn, _object("eduskunta:LA coverage", "LA coverage"))
        conn.execute(
            "INSERT INTO source_coverage(coverage_id, json) VALUES (?, ?)",
            (
                "coverage-zero",
                json.dumps(
                    {
                        "coverage_id": "coverage-zero",
                        "source_id": "SRC-EDUSKUNTA-VASKI",
                        "kind": "LEGISLATIVE_INITIATIVE",
                        "complete": None,
                        "source_record_count": 0,
                        "source_records": 99,
                    }
                ),
            ),
        )
        conn.commit()
        packet = build_matter_ledger(conn, "LA coverage")
        row = next(item for item in packet["coverage"] if item["coverage_id"] == "coverage-zero")
        assert row["source_record_count"] == 0
        assert "SOURCE_COVERAGE_COMPLETENESS_UNKNOWN" in {item["code"] for item in packet["missingness"]}
    finally:
        conn.close()


def test_malformed_normalized_authors_and_evidence_fail_before_empty_projection(tmp_path: Path) -> None:
    conn = _seed_db(tmp_path)
    try:
        malformed_authors = _object("malformed:authors", "M malformed authors")
        malformed_authors["authors"] = ["not-an-object"]
        _insert_object(conn, malformed_authors)
        conn.commit()
        try:
            build_matter_ledger(conn, "M malformed authors")
        except MatterLedgerError as error:
            assert "list item" in str(error)
        else:
            raise AssertionError("malformed authors must not become empty evidence")
    finally:
        conn.close()

    conn = _seed_db(tmp_path / "evidence")
    try:
        malformed_evidence = _object("malformed:evidence", "M malformed evidence")
        malformed_evidence["evidence"] = ["not-an-object"]
        _insert_object(conn, malformed_evidence)
        conn.commit()
        try:
            build_matter_ledger(conn, "M malformed evidence")
        except MatterLedgerError as error:
            assert "list item" in str(error)
        else:
            raise AssertionError("malformed evidence must not become empty evidence")
    finally:
        conn.close()


def test_action_date_evidence_id_is_resolved_from_evidence_table(tmp_path: Path) -> None:
    conn = _seed_db(tmp_path)
    try:
        obj = _object("eduskunta:date-provenance", "M date provenance")
        obj["action_date_provenance"] = {"basis": "SOURCE_EVENT", "evidence_ids": ["date-e"]}
        obj["evidence_ids"] = ["date-e"]
        obj["evidence"] = []
        obj["authors"] = [{"person_id": "10", "name": "Ada Author", "role": "AUTHOR"}]
        _insert_object(conn, obj)
        conn.execute(
            "INSERT INTO evidence(evidence_id, json) VALUES (?, ?)",
            ("date-e", json.dumps({"evidence_id": "date-e", "quote": "Official date"})),
        )
        conn.commit()
        action = build_matter_ledger(conn, "M date provenance")["interventions"][0]
        assert action["date_provenance"]["date_evidence_state"] == "RESOLVED"
        assert action["date_provenance"]["event"]["evidence_ids"] == ["date-e"]
        assert "ACTION_DATE_EVIDENCE_UNRESOLVED" not in {
            item["code"] for item in build_matter_ledger(conn, "M date provenance")["missingness"]
        }
    finally:
        conn.close()


def test_actual_vote_packet_uses_canonical_source_anchor_and_retains_all_status_rows() -> None:
    db = Path("data/paa.sqlite")
    if not db.exists():
        return
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    try:
        packet = build_matter_ledger(conn, "KAA 1/2023 vp")
        vote = next(item for item in packet["interventions"] if item["kind"] == "VOTE")
        assert vote["object_id"] == "eduskunta-vote-52877"
        assert "eduskunta-vote-52877-e" in vote["evidence_ids"]
        assert "vote-52877-e" not in vote["evidence_ids"]
        assert vote["event_source_evidence_ids"][0] == "eduskunta-vote-52877-e"
        assert vote["event_source_evidence_ids"][1].startswith("local-vote-event-")
        assert packet["matter"]["vote_sources"][0]["original_source_evidence_ids"] == [
            "eduskunta-vote-52877-e"
        ]
        assert packet["matter"]["vote_sources"][0]["canonical_source"]["source_id"] == "SRC-EDUSKUNTA-VOTE-52877"
        assert vote["ballot_evidence_ids"][0].startswith("local-ballot-row-")
        assert vote["source_evidence"][0]["status"] == "NORMALIZED_LOCAL_RECORD"
        assert len([item for item in packet["interventions"] if item["kind"] == "VOTE"]) == 199
        assert {item["ballot_status"]["category"] for item in packet["interventions"] if item["kind"] == "VOTE"} == {
            "CAST",
            "ABSENT",
        }
        assert "ACTION_DATE_EVIDENCE_UNRESOLVED" not in {item["code"] for item in packet["missingness"]}
    finally:
        conn.close()


def test_actual_therapy_guarantee_vote_matter_supports_actor_queries() -> None:
    db = Path("data/paa.sqlite")
    if not db.exists():
        return
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    try:
        packet = build_matter_ledger(conn, "HE 131/2024 vp")
        assert packet["matter"]["vote_event_count"] == 10
        assert len([item for item in packet["interventions"] if item["kind"] == "VOTE"]) == 10 * 199
        aino = build_matter_ledger(conn, "HE 131/2024 vp", actor_id="mp-1097")
        anna = build_matter_ledger(conn, "HE 131/2024 vp", actor_id="mp-1155")
        assert len(aino["interventions"]) >= 1
        assert len(anna["interventions"]) >= 1
        assert {item["actor"]["actor_id"] for item in aino["interventions"]} == {"mp-1097"}
        assert {item["actor"]["actor_id"] for item in anna["interventions"]} == {"mp-1155"}
        assert {"2024-12-05", "2024-12-11"}.issubset(
            {item["action_date"] for item in aino["interventions"]}
        )
    finally:
        conn.close()


def test_normalized_vote_evidence_id_and_digest_change_when_event_row_changes(tmp_path: Path) -> None:
    conn = _seed_db(tmp_path)
    try:
        _insert_vote(conn, vote_id="21", matter="M local hash", jaa=1, ei=0, tyhjaa=0, poissa=0, yhteensa=1)
        conn.execute(
            "INSERT INTO ballots VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("21", "10", "Ada", "Author", "x", "x", "JAA"),
        )
        conn.commit()
        first = build_matter_ledger(conn, "M local hash")
        first_evidence = next(
            item
            for item in first["matter"]["vote_sources"][0]["source_evidence"]
            if item["status"] == "NORMALIZED_LOCAL_RECORD"
        )
        conn.execute("UPDATE vote_events SET title = ? WHERE aanestys_id = ?", ("Changed / Option", "21"))
        conn.commit()
        second = build_matter_ledger(conn, "M local hash")
        second_evidence = next(
            item
            for item in second["matter"]["vote_sources"][0]["source_evidence"]
            if item["status"] == "NORMALIZED_LOCAL_RECORD"
        )
        assert first_evidence["evidence_id"].startswith("local-vote-event-")
        assert first_evidence["text_sha256"] != second_evidence["text_sha256"]
        assert first_evidence["evidence_id"] != second_evidence["evidence_id"]
        assert first_evidence["raw_sha256"] is None
        assert first_evidence["source_id"] == "SRC-LOCAL-PAA-SQLITE"
        assert first_evidence["record_locator"] == "vote_events.aanestys_id=21"
        first_ballot = first["interventions"][0]["source_evidence"][0]
        second_ballot = second["interventions"][0]["source_evidence"][0]
        assert first_ballot["status"] == "NORMALIZED_LOCAL_RECORD"
        assert first_ballot["evidence_id"].startswith("local-ballot-row-")
        assert first_ballot["evidence_id"] == second_ballot["evidence_id"]
        conn.execute("UPDATE ballots SET raw_response = ? WHERE aanestys_id = ?", ("EI", "21"))
        conn.commit()
        third = build_matter_ledger(conn, "M local hash")
        third_ballot = third["interventions"][0]["source_evidence"][0]
        assert third_ballot["evidence_id"] != first_ballot["evidence_id"]
        assert third_ballot["text_sha256"] != first_ballot["text_sha256"]
    finally:
        conn.close()


def test_normalized_event_and_ballot_evidence_cannot_bind_another_vote(tmp_path: Path) -> None:
    conn = _seed_db(tmp_path)
    try:
        for vote_id in ("31", "32"):
            _insert_vote(
                conn,
                vote_id=vote_id,
                matter="M two events",
                jaa=1,
                ei=0,
                tyhjaa=0,
                poissa=0,
                yhteensa=1,
                title=f"Option {vote_id} / Alternative {vote_id}",
            )
            conn.execute(
                "INSERT INTO ballots VALUES (?, ?, ?, ?, ?, ?, ?)",
                (vote_id, "10", "Ada", "Author", f"x-{vote_id}", "x", "JAA"),
            )
        conn.commit()
        packet = build_matter_ledger(conn, "M two events")
        vote_sources = {item["vote_id"]: item for item in packet["matter"]["vote_sources"]}
        event_ids = {
            vote_id: set(item["local_evidence_ids"])
            for vote_id, item in vote_sources.items()
        }
        assert event_ids["31"].isdisjoint(event_ids["32"])
        assert all(item["record_locator"] == f"vote_events.aanestys_id={vote_id}"
                   for vote_id, item in (
                       ("31", next(x for x in vote_sources["31"]["source_evidence"] if x["status"] == "NORMALIZED_LOCAL_RECORD")),
                       ("32", next(x for x in vote_sources["32"]["source_evidence"] if x["status"] == "NORMALIZED_LOCAL_RECORD")),
                   ))
        actions = {item["event_id"]: item for item in packet["interventions"] if item["kind"] == "VOTE"}
        assert event_ids["31"].issubset(set(actions["31"]["event_source_evidence_ids"]))
        assert event_ids["32"].issubset(set(actions["32"]["event_source_evidence_ids"]))
        assert actions["31"]["ballot_evidence_ids"] != actions["32"]["ballot_evidence_ids"]
        assert actions["31"]["ballot_source_version"]["raw_sha256"] is None
    finally:
        conn.close()


def test_own_group_peer_comparison_uses_shared_hashed_source_once(tmp_path: Path) -> None:
    conn = _seed_db(tmp_path)
    try:
        for actor_id, person_id, name in (
            ("mp-20", "20", "Peer Two"),
            ("mp-21", "21", "Peer Three"),
            ("mp-22", "22", "Peer Four"),
            ("mp-30", "30", "Other Party"),
        ):
            conn.execute(
                "INSERT INTO actors(actor_id, display_name, name_key, birth_year, person_id, identity_status) VALUES (?, ?, ?, ?, ?, ?)",
                (actor_id, name, name.casefold(), None, person_id, "MP_UNIQUE"),
            )
        _insert_vote(
            conn,
            vote_id="peer-main",
            matter="HE peer/2024 vp",
            jaa=3,
            ei=1,
            tyhjaa=1,
            poissa=1,
            yhteensa=6,
        )
        for row in (
            ("peer-main", "10", "Ada", "Author", "ada|author", "kok", "JAA"),
            ("peer-main", "20", "Peer", "Two", "peer|two", "kok", "JAA"),
            ("peer-main", "21", "Peer", "Three", "peer|three", "kok", "JAA"),
            ("peer-main", "22", "Peer", "Four", "peer|four", "kok", "EI"),
            ("peer-main", "11", "Co", "Signer", "co|signer", "kok", "TYHJA"),
            ("peer-main", "30", "Other", "Party", "other|party", "ps", "POISSA"),
        ):
            conn.execute("INSERT INTO ballots VALUES (?, ?, ?, ?, ?, ?, ?)", row)
        # Unrelated second matter must not leak into the first vote's peer source.
        _insert_vote(
            conn,
            vote_id="peer-other",
            matter="HE transfer/2024 vp",
            jaa=0,
            ei=3,
            tyhjaa=0,
            poissa=0,
            yhteensa=3,
        )
        for row in (
            ("peer-other", "10", "Ada", "Author", "ada|author", "kok", "EI"),
            ("peer-other", "20", "Peer", "Two", "peer|two", "kok", "EI"),
            ("peer-other", "21", "Peer", "Three", "peer|three", "kok", "EI"),
        ):
            conn.execute("INSERT INTO ballots VALUES (?, ?, ?, ?, ?, ?, ?)", row)
        conn.commit()
        packet = build_matter_ledger(conn, "HE peer/2024 vp")
        assert packet["schema_version"] == LEDGER_SCHEMA_VERSION
        assert len(packet["group_peer_sources"]) == 1
        peer_source = packet["group_peer_sources"][0]
        assert peer_source["vote_id"] == "peer-main"
        assert "rows" in peer_source
        votes = [item for item in packet["interventions"] if item["kind"] == "VOTE"]
        by_person = {item["actor"]["person_id"]: item for item in votes}
        ada = by_person["10"]["own_group_peer_comparison"]
        assert ada["status"] == "COMPARABLE"
        assert ada["target_group_code"] == "kok"
        assert ada["peer_count"] == 3
        assert ada["peer_jaa"] == 2
        assert ada["peer_ei"] == 1
        assert ada["peer_majority"] == "JAA"
        assert ada["matches_peer_majority"] is True
        assert ada["source_sha256"] == peer_source["source_sha256"]
        assert "rows" not in ada
        assert by_person["11"]["own_group_peer_comparison"]["status"] == "TARGET_NON_SUBSTANTIVE"
        assert by_person["30"]["own_group_peer_comparison"]["status"] == "TARGET_NON_SUBSTANTIVE"
        # Incomplete published totals exclude the source rather than inventing peers.
        conn.execute("UPDATE vote_events SET jaa = NULL WHERE aanestys_id = ?", ("peer-main",))
        incomplete = build_matter_ledger(conn, "HE peer/2024 vp")
        incomplete_ada = next(
            item["own_group_peer_comparison"]
            for item in incomplete["interventions"]
            if item["kind"] == "VOTE" and item["actor"]["person_id"] == "10"
        )
        assert incomplete_ada["status"] == "SOURCE_EXCLUDED"
        other = build_matter_ledger(conn, "HE transfer/2024 vp")
        other_cmp = next(
            item["own_group_peer_comparison"]
            for item in other["interventions"]
            if item["kind"] == "VOTE" and item["actor"]["person_id"] == "10"
        )
        assert other_cmp["vote_id"] == "peer-other"
        assert other_cmp["target_response"] == "EI"
        assert other_cmp["source_sha256"] != peer_source["source_sha256"]
    finally:
        conn.close()

"""Adversarial and source-shaped tests for group-majority agreement."""

from __future__ import annotations

import copy
import inspect
import json
import sqlite3
from collections import Counter
from pathlib import Path

import pytest

from paa.group_agreement import (
    GroupAgreementError,
    build_group_context,
    build_group_sources,
    build_vote_ballot_source,
    compact_peer_comparison,
    compare_people_on_source,
    compare_person_on_source,
    validate_group_packet,
    validate_group_packets,
)
from paa.store import connect


def _ballot_source(
    vote_id: str,
    rows: list[tuple[str, str, str]],
    *,
    matter: str = "HE 1/2026 vp",
    session_date: str = "2024-02-03",
    published: dict[str, int] | None = None,
) -> dict:
    counts = Counter(response for _person, _group, response in rows)
    published = published or {
        "jaa": counts.get("JAA", 0),
        "ei": counts.get("EI", 0),
        "tyhjaa": counts.get("TYHJA", 0),
        "poissa": counts.get("POISSA", 0),
    }
    event = {
        "aanestys_id": vote_id,
        "year": int(session_date[:4]),
        "session_date": session_date,
        "number": 1,
        "title": f"Test vote {vote_id}",
        "jaa": published["jaa"],
        "ei": published["ei"],
        "tyhjaa": published["tyhjaa"],
        "poissa": published["poissa"],
        "yhteensa": len(rows),
        "url": f"https://example.test/{vote_id}",
        "matter": matter,
        "mitatoity": 0,
    }
    ballots = [
        {
            "person_number": person,
            "first_name": f"First{person}",
            "last_name": "Tester",
            "party": group,
            "raw_response": response,
        }
        for person, group, response in rows
    ]
    return build_vote_ballot_source(event, ballots)


def _add_vote(
    conn: sqlite3.Connection,
    vote_id: str,
    rows: list[tuple[str, str, str]],
    *,
    session_date: str = "2026-01-01",
    matter: str = "HE 1/2026 vp",
    void: int = 0,
    published: dict[str, int] | None = None,
) -> None:
    counts = Counter(response for _person, _group, response in rows)
    published = published or {
        "jaa": counts.get("JAA", 0),
        "ei": counts.get("EI", 0),
        "tyhjaa": counts.get("TYHJA", 0),
        "poissa": counts.get("POISSA", 0),
    }
    conn.execute(
        """INSERT INTO vote_events(
             aanestys_id, year, session_date, number, title, jaa, ei, tyhjaa,
             poissa, yhteensa, url, ptk, matter, mitatoity, json
           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            vote_id,
            int(session_date[:4]),
            session_date,
            1,
            f"Test vote {vote_id}",
            published["jaa"],
            published["ei"],
            published["tyhjaa"],
            published["poissa"],
            len(rows),
            f"https://www.eduskunta.fi/aanestystulos/{vote_id}",
            f"PTK 1/{session_date[:4]} vp",
            matter,
            void,
            json.dumps({"test": True}, ensure_ascii=False),
        ),
    )
    for person, group, response in rows:
        conn.execute(
            """INSERT INTO ballots(
                 aanestys_id, person_number, first_name, last_name, name_key,
                 party, raw_response
               ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (vote_id, person, f"First{person}", "Tester", f"first{person}|tester", group, response),
        )


def _packet(packets: list[dict], person_id: str) -> dict:
    return next(packet for packet in packets if packet["person_id"] == person_id)


def _comparison(packet: dict, vote_id: str) -> dict:
    return next(row for row in packet["comparisons"] if row["vote_id"] == vote_id)


def test_excludes_target_and_requires_strict_peer_majority(tmp_path: Path) -> None:
    conn = connect(tmp_path / "paa.sqlite")
    _add_vote(
        conn,
        "vote-majority",
        [
            ("1", "kok", "JAA"),
            ("2", "kok", "JAA"),
            ("3", "kok", "JAA"),
            ("4", "kok", "EI"),
            ("5", "sd", "EI"),
        ],
    )
    conn.commit()
    packets, sources, report = build_group_context(conn)

    result = _comparison(_packet(packets, "1"), "vote-majority")
    assert result["status"] == "COMPARABLE"
    assert result["target_group_code"] == "kok"
    assert result["peer_count"] == 3
    assert result["peer_jaa"] == 2  # target's JAA is not counted as a peer
    assert result["peer_ei"] == 1
    assert result["peer_majority"] == "JAA"
    assert result["matches_peer_majority"] is True
    assert report["current_actor_party_not_used"] is True
    assert validate_group_packet(_packet(packets, "1"), sources)["valid"] is True
    conn.close()


def test_source_only_builder_matches_shared_sources_without_person_packets(tmp_path: Path) -> None:
    conn = connect(tmp_path / "paa.sqlite")
    _add_vote(
        conn,
        "vote-source-only",
        [("1", "kok", "JAA"), ("2", "kok", "EI"), ("3", "sd", "JAA")],
    )
    conn.commit()

    source_only = build_group_sources(conn)
    packets, shared_sources, _report = build_group_context(conn)

    assert len(source_only) == 1
    assert source_only == shared_sources
    assert {packet["person_id"] for packet in packets} == {"1", "2", "3"}
    conn.close()


def test_ties_and_one_peer_are_undefined(tmp_path: Path) -> None:
    conn = connect(tmp_path / "paa.sqlite")
    _add_vote(
        conn,
        "vote-tie",
        [("1", "kok", "JAA"), ("2", "kok", "JAA"), ("3", "kok", "EI"), ("4", "sd", "JAA")],
    )
    _add_vote(
        conn,
        "vote-one-peer",
        [("1", "kok", "EI"), ("2", "kok", "JAA"), ("3", "sd", "EI")],
    )
    conn.commit()
    packets, sources, _report = build_group_context(conn)
    assert _comparison(_packet(packets, "1"), "vote-tie")["status"] == "PEER_TIE"
    one = _comparison(_packet(packets, "1"), "vote-one-peer")
    assert one["status"] == "PEER_COUNT_BELOW_MINIMUM"
    assert one["peer_count"] == 1
    assert _packet(packets, "1")["summary"]["comparable_count"] == 0
    assert validate_group_packet(_packet(packets, "1"), sources)["valid"] is True
    conn.close()


def test_source_row_missing_and_blank_are_separate_denominators(tmp_path: Path) -> None:
    conn = connect(tmp_path / "paa.sqlite")
    _add_vote(
        conn,
        "vote-present",
        [("1", "kok", "JAA"), ("2", "kok", "JAA"), ("3", "kok", "EI")],
    )
    # Person 1 is deliberately absent from this otherwise reconciled source.
    _add_vote(
        conn,
        "vote-missing-row",
        [("2", "kok", "JAA"), ("3", "kok", "EI")],
    )
    # A blank response is not silently interpreted as POISSA or TYHJA.  It
    # makes this source unreconciled, while its target blank remains visible
    # as a separate audit denominator.
    _add_vote(
        conn,
        "vote-blank-row",
        [("1", "kok", ""), ("2", "kok", "JAA"), ("3", "kok", "EI")],
    )
    _add_vote(
        conn,
        "vote-recorded-abstention",
        [("1", "kok", "TYHJA"), ("2", "kok", "JAA"), ("3", "kok", "EI")],
    )
    _add_vote(
        conn,
        "vote-recorded-absence",
        [("1", "kok", "POISSA"), ("2", "kok", "JAA"), ("3", "kok", "EI")],
    )
    conn.commit()
    packets, sources, _report = build_group_context(conn)
    packet = _packet(packets, "1")
    summary = packet["summary"]
    assert summary["source_row_missing_count"] == 1
    assert summary["absent_count"] == 1  # compatibility alias, explicitly documented
    assert summary["target_blank_count"] == 1
    assert summary["source_excluded_count"] == 1
    assert summary["recorded_abstention_count"] == 1
    assert summary["target_abstention_count"] == 1
    assert summary["recorded_absence_count"] == 1
    assert summary["target_non_substantive_count"] == 2
    assert not any(row["vote_id"] == "vote-missing-row" for row in packet["comparisons"])
    blank_source = next(source for source in sources if source["vote_id"] == "vote-blank-row")
    assert blank_source["state"] == "EXCLUDED_SOURCE_MISMATCH"
    assert validate_group_packet(packet, sources)["valid"] is True
    conn.close()


def test_event_time_group_code_wins_over_current_candidacy_party(tmp_path: Path) -> None:
    conn = connect(tmp_path / "paa.sqlite")
    conn.execute(
        """INSERT INTO candidacies(
             candidacy_id, election_year, party, first_name, last_name, actor_id
           ) VALUES ('c1', 2023, 'ps', 'First1', 'Tester', 'actor-1')"""
    )
    _add_vote(
        conn,
        "vote-before-switch",
        [("1", "ps", "JAA"), ("2", "ps", "JAA"), ("3", "ps", "EI")],
        session_date="2024-04-26",
    )
    _add_vote(
        conn,
        "vote-after-switch",
        [("1", "tv", "JAA"), ("2", "tv", "EI"), ("3", "tv", "EI")],
        session_date="2024-05-24",
    )
    conn.commit()
    packets, sources, _report = build_group_context(conn)
    packet = _packet(packets, "1")
    after = _comparison(packet, "vote-after-switch")
    assert after["target_group_code"] == "tv"
    assert after["status"] == "COMPARABLE"
    assert after["peer_majority"] == "EI"
    assert after["matches_peer_majority"] is False
    assert validate_group_packet(packet, sources)["valid"] is True
    conn.close()


def test_unknown_and_group_less_codes_never_form_peer_references(tmp_path: Path) -> None:
    conn = connect(tmp_path / "paa.sqlite")
    _add_vote(
        conn,
        "vote-erk",
        [("1", "erk", "JAA"), ("2", "erk", "JAA"), ("3", "erk", "EI")],
    )
    _add_vote(
        conn,
        "vote-unknown-group",
        [("1", "mystery", "JAA"), ("2", "mystery", "JAA"), ("3", "mystery", "EI")],
    )
    conn.commit()
    packets, _sources, _report = build_group_context(conn)
    packet = _packet(packets, "1")
    assert _comparison(packet, "vote-erk")["status"] == "TARGET_GROUPLESS_EXCLUDED"
    assert _comparison(packet, "vote-unknown-group")["status"] == "TARGET_UNKNOWN_GROUP"
    assert packet["summary"]["group_less_excluded_count"] == 1
    assert packet["summary"]["unknown_group_count"] == 1
    assert packet["summary"]["comparable_count"] == 0
    conn.close()


def test_void_future_and_published_bucket_mismatch_are_excluded(tmp_path: Path) -> None:
    conn = connect(tmp_path / "paa.sqlite")
    base = [("1", "kok", "JAA"), ("2", "kok", "JAA"), ("3", "kok", "EI")]
    _add_vote(conn, "vote-void", base, void=1)
    _add_vote(conn, "vote-future", base, session_date="2027-01-01")
    _add_vote(conn, "vote-mismatch", base, published={"jaa": 1, "ei": 2, "tyhjaa": 0, "poissa": 0})
    conn.commit()
    _packets, sources, report = build_group_context(conn)
    by_id = {source["vote_id"]: source for source in sources}
    assert by_id["vote-void"]["state"] == "EXCLUDED_VOID"
    assert by_id["vote-future"]["state"] == "EXCLUDED_FUTURE"
    assert by_id["vote-mismatch"]["state"] == "EXCLUDED_SOURCE_MISMATCH"
    assert "VOID_VOTE" in by_id["vote-void"]["exclusion_reasons"]
    assert "FUTURE_CUTOFF" in by_id["vote-future"]["exclusion_reasons"]
    assert report["source_exclusion_reason_counts"]["JAA_BUCKET_MISMATCH"] == 1
    conn.close()


def test_duplicate_person_id_is_a_source_mismatch_without_primary_key_help(tmp_path: Path) -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE vote_events(
          aanestys_id TEXT, year INTEGER, session_date TEXT, number INTEGER,
          title TEXT, jaa INTEGER, ei INTEGER, tyhjaa INTEGER, poissa INTEGER,
          yhteensa INTEGER, url TEXT, ptk TEXT, matter TEXT, mitatoity INTEGER, json TEXT
        );
        CREATE TABLE ballots(
          aanestys_id TEXT, person_number TEXT, first_name TEXT, last_name TEXT,
          name_key TEXT, party TEXT, raw_response TEXT
        );
        """
    )
    _add_vote(conn, "vote-duplicate", [("1", "kok", "JAA"), ("1", "kok", "JAA"), ("2", "kok", "EI")])
    conn.commit()
    _packets, sources, _report = build_group_context(conn)
    source = sources[0]
    assert source["state"] == "EXCLUDED_SOURCE_MISMATCH"
    assert "DUPLICATE_OR_BLANK_PERSON_ID" in source["exclusion_reasons"]
    conn.close()


def test_contested_subset_and_exact_matter_labels_are_descriptive(tmp_path: Path) -> None:
    conn = connect(tmp_path / "paa.sqlite")
    rows = [
        ("1", "kok", "JAA"),
        ("2", "kok", "JAA"),
        ("3", "kok", "JAA"),
        ("4", "kok", "EI"),
    ]
    _add_vote(conn, "vote-contested-a", rows, matter="HE 1/2026 vp")
    _add_vote(conn, "vote-contested-b", rows, matter="HE 1/2026 vp")
    _add_vote(
        conn,
        "vote-uncontested",
        [("1", "kok", "JAA"), ("2", "kok", "JAA"), ("3", "kok", "JAA"), ("4", "kok", "JAA")],
        matter="HE 1/2026 vp § 2",
    )
    conn.commit()
    packets, _sources, report = build_group_context(conn)
    packet = _packet(packets, "1")
    assert _comparison(packet, "vote-contested-a")["matter"] == "HE 1/2026 vp"
    assert report["contested_source_count"] == 2
    assert report["matter_sensitivity"]["basis"] == "EXACT_SOURCE_FORMAL_MATTER_LABEL"
    assert {item["label"] for item in report["matter_sensitivity"]["repeated_labels"]} == {"HE 1/2026 vp"}
    assert report["matter_sensitivity"]["policy_clustering"] == "NOT_PERFORMED"
    assert packet["summary"]["contested_comparable_count"] == 2
    diagnostic = packet["summary"]["matter_cluster_diagnostic"]
    assert diagnostic["labels_denominator"] == 2
    assert diagnostic["largest_label"] == "HE 1/2026 vp"
    assert diagnostic["largest_label_count"] == 2
    assert diagnostic["mean_within_label_agreement"] == 1.0
    assert diagnostic["agreement_with_largest_label_removed"] == 1.0
    conn.close()


def test_validator_detects_tampered_comparison_and_source(tmp_path: Path) -> None:
    conn = connect(tmp_path / "paa.sqlite")
    _add_vote(
        conn,
        "vote-validate",
        [("1", "kok", "JAA"), ("2", "kok", "JAA"), ("3", "kok", "EI")],
    )
    conn.commit()
    packets, sources, _report = build_group_context(conn)
    packet = _packet(packets, "1")
    assert validate_group_packet(packet, sources)["valid"] is True

    changed_packet = copy.deepcopy(packet)
    changed_packet["comparisons"][0]["peer_count"] += 1
    result = validate_group_packet(changed_packet, sources)
    assert result["valid"] is False
    assert any("comparison differs" in error for error in result["errors"])

    changed_diagnostic = copy.deepcopy(packet)
    changed_diagnostic["summary"]["matter_cluster_diagnostic"]["largest_label_count"] += 1
    result = validate_group_packet(changed_diagnostic, sources)
    assert result["valid"] is False
    assert any("summary differs" in error for error in result["errors"])

    changed_sources = copy.deepcopy(sources)
    changed_sources[0]["rows"][0]["response"] = "EI"
    result = validate_group_packet(packet, changed_sources)
    assert result["valid"] is False
    assert any("source_sha256" in error or "rows are not" in error for error in result["errors"])
    conn.close()


def test_collection_validator_reuses_source_snapshot_and_checks_every_packet(tmp_path: Path) -> None:
    conn = connect(tmp_path / "paa.sqlite")
    rows = [("1", "kok", "JAA"), ("2", "kok", "JAA"), ("3", "kok", "EI")]
    _add_vote(conn, "vote-a", rows, matter="HE 1/2026 vp")
    _add_vote(conn, "vote-b", rows, matter="HE 2/2026 vp")
    conn.commit()
    packets, sources, report = build_group_context(conn)
    result = validate_group_packets(packets, sources)
    assert result["valid"] is True
    assert result["checked_sources"] == len(sources) == 2
    assert result["checked_packets"] == len(packets) == 3
    assert result["valid_packets"] == 3
    assert result["invalid_packets"] == 0
    assert result["checked_comparisons"] == sum(len(packet["comparisons"]) for packet in packets)
    assert result["source_snapshot_sha256"] == report["snapshot_sha256"]
    assert result["source_ref_set_sha256"]
    assert result["packet_person_id_set_sha256"]
    conn.close()


def test_collection_validator_rejects_missing_and_extra_source_refs(tmp_path: Path) -> None:
    conn = connect(tmp_path / "paa.sqlite")
    rows = [("1", "kok", "JAA"), ("2", "kok", "JAA"), ("3", "kok", "EI")]
    _add_vote(conn, "vote-a", rows)
    _add_vote(conn, "vote-b", rows)
    conn.commit()
    packets, sources, _report = build_group_context(conn)

    missing = copy.deepcopy(packets)
    missing[0]["source_refs"].pop()
    result = validate_group_packets(missing, sources)
    assert result["valid"] is False
    assert any("source reference set differs" in error for error in result["errors"])

    extra = copy.deepcopy(packets)
    extra[1]["source_refs"].append("eduskunta:ballots:not-in-source-set")
    result = validate_group_packets(extra, sources)
    assert result["valid"] is False
    assert any("source reference set differs" in error for error in result["errors"])
    conn.close()


def test_public_compare_rejects_cross_source_prepared_cache() -> None:
    """Root witness: prepared B must not bind EI ballots onto vote A identity."""

    rows_jaa = [("1", "kok", "JAA"), ("2", "kok", "JAA"), ("3", "kok", "JAA"), ("4", "ps", "EI")]
    rows_ei = [("1", "kok", "EI"), ("2", "kok", "EI"), ("3", "kok", "EI"), ("4", "ps", "JAA")]
    source_a = _ballot_source("vote-A", rows_jaa, matter="HE A/2024 vp")
    source_b = _ballot_source("vote-B", rows_ei, matter="HE B/2024 vp")
    assert source_a["valid"] and source_b["valid"]
    assert "prepared" not in inspect.signature(compare_person_on_source).parameters
    with pytest.raises(TypeError):
        compare_person_on_source(source_a, "1", prepared={"by_person": {}})  # type: ignore[call-arg]
    direct_a = compare_person_on_source(source_a, "1")
    direct_b = compare_person_on_source(source_b, "1")
    assert direct_a["vote_id"] == "vote-A"
    assert direct_a["target_response"] == "JAA"
    assert direct_a["status"] == "COMPARABLE"
    assert direct_b["vote_id"] == "vote-B"
    assert direct_b["target_response"] == "EI"
    assert direct_b["status"] == "COMPARABLE"
    batch_a = compare_people_on_source(source_a, ["1", "2", "3"])
    assert batch_a["1"] == direct_a
    assert batch_a["2"]["target_response"] == "JAA"
    compact = compact_peer_comparison(batch_a["1"], source=source_a)
    assert compact["source_sha256"] == source_a["source_sha256"]
    assert compact["vote_id"] == "vote-A"
    assert compact["target_response"] == "JAA"


def test_compare_people_same_source_batch_equals_direct_and_rejects_stale_hash() -> None:
    rows = [
        ("1", "kok", "JAA"),
        ("2", "kok", "JAA"),
        ("3", "kok", "EI"),
        ("4", "kok", "TYHJA"),
        ("5", "kok", "POISSA"),
        ("6", "ps", "JAA"),
    ]
    source = _ballot_source("vote-mix", rows)
    batch = compare_people_on_source(source, ["1", "3", "4", "5"])
    assert batch["1"] == compare_person_on_source(source, "1")
    # Person 1 peers: 2 JAA + 3 EI -> tie. Person 3 peers: 1+2 JAA -> majority JAA.
    assert batch["1"]["peer_jaa"] == 1
    assert batch["1"]["peer_ei"] == 1
    assert batch["1"]["status"] == "PEER_TIE"
    assert batch["3"]["status"] == "COMPARABLE"
    assert batch["3"]["peer_majority"] == "JAA"
    assert batch["3"]["matches_peer_majority"] is False
    assert batch["4"]["status"] == "TARGET_NON_SUBSTANTIVE"
    assert batch["5"]["status"] == "TARGET_NON_SUBSTANTIVE"
    # Target excluded; other same-group rows retain full/excluded denominators.
    assert batch["1"]["target_excluded"] is True
    assert batch["1"]["group_other_record_count"] == 4  # 2,3,4,5 in kok
    assert batch["1"]["peer_excluded_tyhja"] == 1
    assert batch["1"]["peer_excluded_poissa"] == 1
    assert batch["4"]["group_other_record_count"] == 4
    compact_non = compact_peer_comparison(batch["4"], source=source)
    assert compact_non["status"] == "TARGET_NON_SUBSTANTIVE"
    assert compact_non["group_other_record_count"] == 4
    assert compact_non["peer_excluded_tyhja"] == 0  # other TYHJA rows excluding target 4
    assert compact_non["peer_excluded_poissa"] == 1
    stale = copy.deepcopy(source)
    stale["rows"] = [
        {
            "person_id": person,
            "group_code": group,
            "response": "EI" if response == "JAA" else response,
            "first_name": f"First{person}",
            "last_name": "Tester",
        }
        for person, group, response in rows
    ]
    with pytest.raises(GroupAgreementError, match="source_sha256"):
        compare_people_on_source(stale, ["1"])


def test_invalid_or_partial_source_excludes_batch_comparisons() -> None:
    rows = [("1", "kok", "JAA"), ("2", "kok", "JAA"), ("3", "kok", "EI")]
    # Published totals disagree with observed rows -> source invalid.
    source = _ballot_source(
        "vote-bad",
        rows,
        published={"jaa": 9, "ei": 9, "tyhjaa": 0, "poissa": 0},
    )
    assert source["valid"] is False
    comparison = compare_person_on_source(source, "1")
    assert comparison["status"] == "SOURCE_EXCLUDED"
    assert comparison["vote_id"] == "vote-bad"
    compact = compact_peer_comparison(comparison, source=source)
    assert compact["status"] == "SOURCE_EXCLUDED"
    assert compact["source_sha256"] == source["source_sha256"]


def test_compact_rejects_same_vote_different_snapshot_and_policy_relabel() -> None:
    """Second witness: compact must not rebound comparison A onto source B."""

    rows_jaa = [("1", "kok", "JAA"), ("2", "kok", "JAA"), ("3", "kok", "JAA")]
    rows_ei = [("1", "kok", "EI"), ("2", "kok", "EI"), ("3", "kok", "EI")]
    source_a = _ballot_source("snapshot-witness", rows_jaa, matter="SYNTHETIC")
    source_b = _ballot_source("snapshot-witness", rows_ei, matter="SYNTHETIC")
    assert source_a["vote_id"] == source_b["vote_id"]
    assert source_a["source_sha256"] != source_b["source_sha256"]
    comparison_a = compare_people_on_source(source_a, ["1"])["1"]
    assert comparison_a["source_sha256"] == source_a["source_sha256"]
    assert comparison_a["person_id"] == "1"
    assert comparison_a["minimum_peer_count"] == 2
    assert comparison_a["status"] == "COMPARABLE"
    assert comparison_a["target_response"] == "JAA"
    with pytest.raises(GroupAgreementError, match="source_sha256"):
        compact_peer_comparison(comparison_a, source=source_b)
    valid_b = compact_peer_comparison(
        compare_people_on_source(source_b, ["1"])["1"],
        source=source_b,
    )
    assert valid_b["target_response"] == "EI"
    assert valid_b["peer_ei"] == 2
    assert valid_b["source_sha256"] == source_b["source_sha256"]
    with pytest.raises(GroupAgreementError, match="minimum_peer_count"):
        compact_peer_comparison(comparison_a, source=source_a, minimum_peer_count=3)
    policy3 = compare_person_on_source(source_a, "1", minimum_peer_count=3)
    assert policy3["status"] == "PEER_COUNT_BELOW_MINIMUM"
    assert policy3["matches_peer_majority"] is None
    stale = copy.deepcopy(source_a)
    stale["rows"] = copy.deepcopy(source_b["rows"])
    with pytest.raises(GroupAgreementError, match="stale hash|source_sha256"):
        compact_peer_comparison(comparison_a, source=stale)
    # Undefined controls remain undefined, not false agreement.
    unknown = _ballot_source(
        "ctrl-unknown",
        [("1", "xyz", "JAA"), ("2", "xyz", "JAA"), ("3", "xyz", "EI")],
    )
    assert compare_person_on_source(unknown, "1")["status"] == "TARGET_UNKNOWN_GROUP"
    assert compare_person_on_source(unknown, "1")["matches_peer_majority"] is None
    groupless = _ballot_source(
        "ctrl-erk",
        [("1", "erk", "JAA"), ("2", "kok", "JAA"), ("3", "kok", "EI")],
    )
    assert compare_person_on_source(groupless, "1")["status"] == "TARGET_GROUPLESS_EXCLUDED"
    tie = _ballot_source(
        "ctrl-tie",
        [("1", "kok", "JAA"), ("2", "kok", "JAA"), ("3", "kok", "EI")],
    )
    tie_cmp = compare_person_on_source(tie, "1")
    assert tie_cmp["status"] == "PEER_TIE"
    assert tie_cmp["matches_peer_majority"] is None
    absent = _ballot_source(
        "ctrl-tyhja",
        [("1", "kok", "TYHJA"), ("2", "kok", "JAA"), ("3", "kok", "JAA")],
    )
    absent_cmp = compare_person_on_source(absent, "1")
    assert absent_cmp["status"] == "TARGET_NON_SUBSTANTIVE"
    assert absent_cmp["matches_peer_majority"] is None

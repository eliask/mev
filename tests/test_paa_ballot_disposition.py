"""Generic ballot/alternative/disposition computation across two domains."""

import copy
import json
from pathlib import Path

import pytest

from paa.ballot_disposition import (
    BallotDispositionError,
    actor_row,
    bind_alternatives,
    compare_person_ballot,
    evaluate_corpus,
    evaluate_instance,
    load_corpus,
    packet_digest,
    select_alternative,
    title_side_labels,
    validate_packet,
    vote_disposition,
)
from paa.config import FIXTURE_DIR
from paa.group_agreement import compare_person_on_source

CORPUS = FIXTURE_DIR / "ballot_disposition_corpus.json"


def _clone_corpus(tmp_path: Path) -> tuple[dict, Path]:
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    path = tmp_path / "corpus.json"
    path.write_text(json.dumps(corpus), encoding="utf-8")
    return corpus, path


def test_two_instances_hospital_and_wage_mediation() -> None:
    result = evaluate_corpus(CORPUS)
    assert set(result["instance_ids"]) == {"hospital_oulaskangas", "wage_mediation"}
    hospital = result["instances"]["hospital_oulaskangas"]
    wage = result["instances"]["wage_mediation"]
    validate_packet(hospital)
    validate_packet(wage)

    hanninen_he149 = actor_row(hospital, person_id="1537", vote_id="53538")
    assert hanninen_he149["raw_response"] == "EI"
    assert hanninen_he149["selection_state"] == "PROCEDURAL_LABEL_ONLY"
    assert hanninen_he149["matches_peer_majority"] is False
    assert actor_row(hospital, person_id="1450", vote_id="53538")["raw_response"] == "JAA"
    assert actor_row(hospital, person_id="1527", vote_id="53538")["raw_response"] == "JAA"

    hanninen_kaa = actor_row(hospital, person_id="1537", vote_id="55560")
    assert hanninen_kaa["raw_response"] == "EI"
    assert hanninen_kaa["selection_state"] == "AUTHORED_TEXT_BOUND"
    assert "lainvalmisteluun ryhtymisestä" in (hanninen_kaa["selected_text"] or "")
    assert hanninen_kaa["selected_witness"]["source_id"] == "DOC-STVM-13-2025-PROJECTION"

    wage_h = actor_row(wage, person_id="1537", vote_id="53586")
    assert wage_h["matter"] == "HE 146/2024 vp"
    assert wage_h["raw_response"] == "JAA"
    assert actor_row(wage, person_id="1511", vote_id="53586")["raw_response"] == "EI"
    assert actor_row(wage, person_id="1468", vote_id="53586")["raw_response"] == "EI"


def test_bracket_does_not_adopt_eliminated_merinen4() -> None:
    hospital = evaluate_instance(load_corpus(CORPUS), "hospital_oulaskangas")
    assert hospital["dispositions"]["53546"]["winner_label"] == "Lausumaehdotus, Hanna-Leena Mattila 1"
    assert hospital["dispositions"]["53548"]["winner_label"] == "Lausumaehdotus, mietintö"
    assert "Merinen" not in (hospital["dispositions"]["53548"]["winner_label"] or "")
    merinen_text = hospital["bindings"]["53546"]["ei_text"]
    assert "31.12.2026" in merinen_text
    assert hospital["dispositions"]["53548"]["winner_text"] != merinen_text
    chain = hospital["bracket_chains"][0]
    assert chain["stages"][0]["winner_survives_as"] == "Hanna-Leena Mattila 1"
    assert chain["stages"][1]["eliminated_label"] == "Hanna-Leena Mattila 1"
    assert chain["final_adopted_side"]["winner_label"] == "Lausumaehdotus, mietintö"
    assert chain["adopted_note_authoritative"] is False
    assert hospital["bindings"]["53548"]["ei_witness"] == hospital["bindings"]["53546"]["jaa_witness"]


def test_tyhja_is_not_policy_opposition() -> None:
    hospital = evaluate_instance(load_corpus(CORPUS), "hospital_oulaskangas")
    row = actor_row(hospital, person_id="1537", vote_id="53546")
    assert row["raw_response"] == "TYHJA"
    assert row["selection_state"] == "NON_SUBSTANTIVE_NO_SIDE"
    assert row["selected_label"] is None


def test_named_fixture_controls_pass() -> None:
    """Fixture-specific positive controls live in tests, not the maintained module."""

    corpus = load_corpus(CORPUS)
    hospital = evaluate_instance(corpus, "hospital_oulaskangas")
    wage = evaluate_instance(corpus, "wage_mediation")
    # Wrong person: absent id does not inherit another member's ballot.
    missing = compare_person_on_source(corpus["votes"]["53538"], "999999")
    assert missing["status"] != "COMPARABLE" or missing.get("target_response") is None
    assert hospital["dispositions"]["53548"]["winner_label"] != hospital["bindings"]["53546"]["ei_label"]
    assert actor_row(hospital, person_id="1537", vote_id="53546")["selection_state"] == "NON_SUBSTANTIVE_NO_SIDE"
    assert actor_row(wage, person_id="1537", vote_id="53586")["matter"] == "HE 146/2024 vp"
    assert actor_row(hospital, person_id="1537", vote_id="53538")["matter"] == "HE 149/2024 vp"


def test_swapped_chair_labels_rejected() -> None:
    corpus = load_corpus(CORPUS)
    source = corpus["votes"]["53546"]
    jaa, ei = title_side_labels(source["title"])
    assert jaa and ei
    with pytest.raises(BallotDispositionError, match="exact chair sides"):
        bind_alternatives(
            "53546",
            source,
            {**corpus["alternatives"]["53546"], "jaa_label": ei, "ei_label": jaa},
            documents=corpus["documents"],
        )


def test_authored_text_change_without_witness_update_rejected(tmp_path: Path) -> None:
    corpus, path = _clone_corpus(tmp_path)
    corpus["alternatives"]["53546"]["jaa_witness"]["quote"] = "TAMPERED TEXT THAT IS NOT IN SOURCE"
    corpus["alternatives"]["53546"]["jaa_witness"]["end"] = (
        corpus["alternatives"]["53546"]["jaa_witness"]["start"] + len("TAMPERED TEXT THAT IS NOT IN SOURCE")
    )
    path.write_text(json.dumps(corpus), encoding="utf-8")
    with pytest.raises(BallotDispositionError, match="exact source span|witness"):
        load_corpus(path)


def test_duplicate_person_ids_rejected_before_selection(tmp_path: Path) -> None:
    corpus, path = _clone_corpus(tmp_path)
    row = copy.deepcopy(corpus["votes"]["53538"]["rows"][0])
    corpus["votes"]["53538"]["rows"].append(row)
    from paa.group_agreement import _source_hash

    corpus["votes"]["53538"]["source_sha256"] = _source_hash(corpus["votes"]["53538"])
    corpus["source_manifest"]["53538"]["source_sha256"] = corpus["votes"]["53538"]["source_sha256"]
    path.write_text(json.dumps(corpus), encoding="utf-8")
    with pytest.raises(BallotDispositionError, match="invalid source"):
        load_corpus(path)


def test_bool_and_negative_totals_rejected() -> None:
    corpus = load_corpus(CORPUS)
    source = copy.deepcopy(corpus["votes"]["53538"])
    binding = bind_alternatives("53538", source, corpus["alternatives"]["53538"], documents=corpus["documents"])
    source["published_totals"] = {"JAA": True, "EI": False, "TYHJA": 0, "POISSA": 0, "TOTAL": 1}
    with pytest.raises(BallotDispositionError, match="non-negative int"):
        vote_disposition(source, binding)
    source["published_totals"] = {"JAA": -1, "EI": 2, "TYHJA": 0, "POISSA": 0, "TOTAL": 1}
    with pytest.raises(BallotDispositionError, match="non-negative int"):
        vote_disposition(source, binding)


def test_future_cutoff_and_year_mismatch_rejected(tmp_path: Path) -> None:
    corpus, path = _clone_corpus(tmp_path)
    from paa.group_agreement import _source_hash

    corpus["votes"]["53538"]["session_date"] = "2027-01-01"
    corpus["votes"]["53538"]["year"] = 2027
    corpus["votes"]["53538"]["source_sha256"] = _source_hash(corpus["votes"]["53538"])
    corpus["source_manifest"]["53538"].update(
        {
            "session_date": "2027-01-01",
            "year": 2027,
            "source_sha256": corpus["votes"]["53538"]["source_sha256"],
        }
    )
    path.write_text(json.dumps(corpus), encoding="utf-8")
    with pytest.raises(BallotDispositionError, match="after corpus cutoff|invalid source"):
        load_corpus(path)

    corpus, path = _clone_corpus(tmp_path)
    corpus["votes"]["53538"]["year"] = 2025
    corpus["votes"]["53538"]["source_sha256"] = _source_hash(corpus["votes"]["53538"])
    corpus["source_manifest"]["53538"].update(
        {"year": 2025, "source_sha256": corpus["votes"]["53538"]["source_sha256"]}
    )
    path.write_text(json.dumps(corpus), encoding="utf-8")
    with pytest.raises(BallotDispositionError, match="year/session_date mismatch"):
        load_corpus(path)


def test_duplicate_instance_ids_and_foreign_matter_rejected() -> None:
    corpus = load_corpus(CORPUS)
    bad = copy.deepcopy(corpus)
    bad["instances"]["hospital_oulaskangas"]["vote_ids"] = ["53538", "53538"]
    with pytest.raises(BallotDispositionError, match="duplicate"):
        evaluate_instance(bad, "hospital_oulaskangas")
    bad = copy.deepcopy(corpus)
    bad["instances"]["hospital_oulaskangas"]["vote_ids"] = ["53538", "53586"]
    with pytest.raises(BallotDispositionError, match="outside instance matters"):
        evaluate_instance(bad, "hospital_oulaskangas")


def test_bracket_graph_mutations_rejected(tmp_path: Path) -> None:
    corpus, path = _clone_corpus(tmp_path)
    corpus["bracket_chains"][0]["stages"][0]["role"] = "BOGUS"
    path.write_text(json.dumps(corpus), encoding="utf-8")
    # load still ok; evaluation must fail
    loaded = load_corpus(path)
    with pytest.raises(BallotDispositionError, match="bracket role|FIRST_ROUND"):
        evaluate_instance(loaded, "hospital_oulaskangas")

    corpus, path = _clone_corpus(tmp_path)
    corpus["bracket_chains"][0]["stages"][0]["winner_label"] = "not-the-winner"
    path.write_text(json.dumps(corpus), encoding="utf-8")
    loaded = load_corpus(path)
    with pytest.raises(BallotDispositionError, match="winner_label"):
        evaluate_instance(loaded, "hospital_oulaskangas")

    corpus, path = _clone_corpus(tmp_path)
    corpus["bracket_chains"][0]["stages"][0]["winner_survives_as"] = "Lausumaehdotus, mietintö"
    path.write_text(json.dumps(corpus), encoding="utf-8")
    loaded = load_corpus(path)
    with pytest.raises(BallotDispositionError, match="does not resolve to continuing side"):
        evaluate_instance(loaded, "hospital_oulaskangas")


def test_valid_chair_label_prefix_alias_still_continues_by_witness() -> None:
    corpus = load_corpus(CORPUS)
    hospital = evaluate_instance(corpus, "hospital_oulaskangas")
    assert hospital["bracket_chains"][0]["stages"][0]["winner_survives_as"] == "Hanna-Leena Mattila 1"
    assert hospital["bindings"]["53548"]["ei_witness"]["quote"] == hospital["bindings"]["53546"]["jaa_witness"]["quote"]


def test_swapped_approved_witnesses_rejected_before_projection() -> None:
    corpus = load_corpus(CORPUS)
    bad = copy.deepcopy(corpus)
    bad["alternatives"]["53546"]["jaa_witness"], bad["alternatives"]["53546"]["ei_witness"] = (
        bad["alternatives"]["53546"]["ei_witness"],
        bad["alternatives"]["53546"]["jaa_witness"],
    )
    with pytest.raises(BallotDispositionError, match="approved side binding"):
        evaluate_instance(bad, "hospital_oulaskangas")


def test_same_document_different_span_rejected_for_mattila_and_kaa() -> None:
    """Approved binding pins exact offsets; another exact span in the same doc fails."""

    corpus = load_corpus(CORPUS)
    evaluate_instance(corpus, "hospital_oulaskangas")  # unmodified control
    mattila = corpus["documents"]["DOC-PUH-136-2024-5-1-33"]["text"]
    other = mattila[-80:]
    start, end = len(mattila) - 80, len(mattila)
    bad = copy.deepcopy(corpus)
    for vote_id, key in (("53546", "jaa_witness"), ("53548", "ei_witness")):
        bad["alternatives"][vote_id][key] = {
            **bad["alternatives"][vote_id][key],
            "start": start,
            "end": end,
            "quote": other,
        }
    with pytest.raises(BallotDispositionError, match="approved side binding"):
        evaluate_instance(bad, "hospital_oulaskangas")

    stvm = corpus["documents"]["DOC-STVM-13-2025-PROJECTION"]["text"]
    bad_kaa = copy.deepcopy(corpus)
    bad_kaa["alternatives"]["55560"]["ei_witness"] = {
        **bad_kaa["alternatives"]["55560"]["ei_witness"],
        "start": len(stvm) - 80,
        "end": len(stvm),
        "quote": stvm[-80:],
    }
    with pytest.raises(BallotDispositionError, match="approved side binding"):
        evaluate_instance(bad_kaa, "hospital_oulaskangas")


def test_adopted_note_is_non_authoritative() -> None:
    corpus = load_corpus(CORPUS)
    bad = copy.deepcopy(corpus)
    bad["bracket_chains"][0]["adopted_note"] = "ARBITRARY_MUTATED_MARKER"
    packet = evaluate_instance(bad, "hospital_oulaskangas")
    assert packet["bracket_chains"][0]["adopted_note"] == "ARBITRARY_MUTATED_MARKER"
    assert packet["bracket_chains"][0]["final_adopted_side"]["winner_label"] == "Lausumaehdotus, mietintö"
    assert packet["bracket_chains"][0]["adopted_note_authoritative"] is False


def test_packet_digest_covers_bindings_and_selected_text() -> None:
    hospital = evaluate_instance(load_corpus(CORPUS), "hospital_oulaskangas")
    digest = hospital["result_sha256"]
    hospital["bindings"]["53546"]["jaa_text"] = "changed"
    assert packet_digest(hospital) != digest
    with pytest.raises(BallotDispositionError, match="digest mismatch"):
        validate_packet(hospital)


def test_packet_digest_covers_display_name_and_row_limits() -> None:
    hospital = evaluate_instance(load_corpus(CORPUS), "hospital_oulaskangas")
    digest = hospital["result_sha256"]
    hospital["rows"][0]["display_name"] = "MUTATED NAME"
    assert packet_digest(hospital) != digest
    hospital = evaluate_instance(load_corpus(CORPUS), "hospital_oulaskangas")
    digest = hospital["result_sha256"]
    hospital["rows"][0]["limits"] = (*hospital["rows"][0]["limits"], "MUTATED LIMIT")
    assert packet_digest(hospital) != digest
    with pytest.raises(BallotDispositionError, match="digest mismatch"):
        validate_packet(hospital)


def test_direct_compare_refuses_invalid_source_meaning() -> None:
    corpus = load_corpus(CORPUS)
    source = copy.deepcopy(corpus["votes"]["53538"])
    source["session_date"] = "2027-01-01"
    result = compare_person_ballot(source, "1537", cutoff="2026-10-09")
    assert result["status"] == "SOURCE_EXCLUDED"
    assert result["validated_meaning"] is False


def test_altered_corpus_hash_rejected_on_load(tmp_path: Path) -> None:
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    corpus["votes"]["53538"]["title"] = "Tampered"
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(corpus), encoding="utf-8")
    with pytest.raises(BallotDispositionError, match="source hash changed"):
        load_corpus(path)


def test_raw_code_without_authored_text_stays_procedural() -> None:
    corpus = load_corpus(CORPUS)
    source = corpus["votes"]["53586"]
    binding = bind_alternatives("53586", source, corpus["alternatives"]["53586"])
    selected = select_alternative("JAA", binding)
    assert selected["selection_state"] == "PROCEDURAL_LABEL_ONLY"
    assert selected["selected_text"] is None


def test_group_agreement_helper_matches_event_semantics() -> None:
    corpus = load_corpus(CORPUS)
    source = corpus["votes"]["53538"]
    comparison = compare_person_on_source(source, "1537")
    assert comparison["status"] == "COMPARABLE"
    assert comparison["target_response"] == "EI"
    assert comparison["matches_peer_majority"] is False

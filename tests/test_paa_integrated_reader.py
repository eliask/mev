"""Integrated reader: journeys in bootstrap site with honest projection status."""

import json
from pathlib import Path

import pytest

from paa.bootstrap_site import BootstrapSiteError, build_bootstrap_artifact
from paa.integrated_reader import IntegratedReaderError, compose_journeys_from_manifest_entries
from paa.question_journey import SCHEMA_VERSION as JOURNEY_SCHEMA
from paa.question_journey import compose_journey
from paa.route_packet import build_route_packet
from paa.source_question import SourceQuestionBank

ROUTE_FIXTURE = Path("paa/contracts/fixtures/route_comparison_bank")
JOURNEY_FIXTURE = Path("paa/contracts/fixtures/question_journey_bank")


def _minimal_matter(tmp_path: Path, matter_id: str = "LA 9/2099 vp") -> Path:
    path = tmp_path / "matter.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "paa.matter_ledger.v2",
                "query": {"mode": "EXACT_MATTER", "matter_id": matter_id},
                "matter": {"matter_id": matter_id},
                "actors": [],
                "interventions": [],
                "coverage": {},
                "missingness": [],
                "limits": [],
            }
        ),
        encoding="utf-8",
    )
    return path


def _journey_packet(*, reviewed: bool = False) -> dict:
    bank = SourceQuestionBank.load(ROUTE_FIXTURE / "questions.json", ROUTE_FIXTURE / "sources.json")
    answer = json.loads((JOURNEY_FIXTURE / "answer_ROUTE-FULL.json").read_text(encoding="utf-8"))
    route = build_route_packet(bank, "ROUTE-FULL", as_of="2100-10-09", answer_packet=answer)
    mapping = json.loads((JOURNEY_FIXTURE / "mapping_ROUTE-FULL.json").read_text(encoding="utf-8"))
    if not reviewed:
        return compose_journey(route_packet=route, mapping=mapping, answer_packet=answer)
    # Build a bound projection review against current bytes.
    packet = compose_journey(route_packet=route, mapping=mapping, answer_packet=answer)
    from paa.question_journey import _sha_json

    review = {
        "schema_version": "paa.journey_projection_review.v1",
        "task_id": "ROUTE-FULL",
        "mapping_sha256": packet["mapping_sha256"],
        "journey_content_sha256": packet["journey_content_sha256"],
        "questions_sha256": route["bank"]["questions_sha256"],
        "sources_sha256": route["bank"]["sources_sha256"],
        "candidate_sha256": _sha_json(
            {
                "task_id": answer.get("task_id"),
                "matter_id": answer.get("matter_id"),
                "question": answer.get("question"),
                "answer": answer.get("answer"),
                "claims": answer.get("claims"),
                "unresolved": answer.get("unresolved"),
                "next_evidence": answer.get("next_evidence"),
            }
        ),
        "review_date": "2026-10-10",
        "decision": "SUPPORTED_WITH_STATED_LIMITS",
        "reviewed_fields": ["nodes", "edges", "branches", "short_answer"],
        "reviewer": {
            "id": "test-projection-reviewer",
            "kind": "INDEPENDENT_AI_SOURCE_READER",
            "human_certification": False,
        },
    }
    return compose_journey(
        route_packet=route,
        mapping=mapping,
        answer_packet=answer,
        projection_review=review,
    )


def test_bootstrap_indexes_unreviewed_journey_without_graph_admission(tmp_path: Path) -> None:
    packet = _journey_packet(reviewed=False)
    out = tmp_path / "site"
    manifest = build_bootstrap_artifact(
        output_dir=out,
        matter_packets=[_minimal_matter(tmp_path)],
        question_packets=[],
        journey_packets=[packet],
        title="Poliittinen muisti",
    )
    assert manifest["journey_count"] == 1
    assert manifest["projection_reviewed_journey_count"] == 0
    assert manifest["journeys"][0]["projection_status"] == "UNREVIEWED"
    index = (out / "index.html").read_text(encoding="utf-8")
    assert "Kysymysmatkat" in index
    assert "journeys/" in index
    assert "lookup.html" in index
    assert index.index("Kysymysmatkat") < index.index("Valmistellut kysymykset")
    assert "Kaavioehdokas" in index
    assert "journey-card" not in index
    assert "Kaavion tekoälylähdearvio" not in index
    journey_page = (out / "journeys" / "ROUTE_FULL.html").read_text(encoding="utf-8")
    assert "Asia ja henkilöt" in journey_page
    assert "tarkistamaton ehdokas" in journey_page
    assert (out / "lookup.html").is_file()


def test_bootstrap_projection_reviewed_journey_keeps_human_certification_false(tmp_path: Path) -> None:
    packet = _journey_packet(reviewed=True)
    out = tmp_path / "reviewed"
    manifest = build_bootstrap_artifact(
        output_dir=out,
        matter_packets=[_minimal_matter(tmp_path)],
        question_packets=[],
        journey_packets=[packet],
        title="Poliittinen muisti",
    )
    assert manifest["projection_reviewed_journey_count"] == 1
    assert manifest["journeys"][0]["projection_reviewed"] is True
    assert manifest["journeys"][0]["human_certification"] is False
    index = (out / "index.html").read_text(encoding="utf-8")
    assert "human_certification=false" in index
    assert "Kaavion tekoälylähdearvio" in index
    assert "journey-card" not in index


def test_prose_review_cannot_mark_journey_projection_reviewed(tmp_path: Path) -> None:
    packet = _journey_packet(reviewed=False)
    packet["projection_status"] = "REVIEWED_AI_SOURCE_READING"
    packet["source_answer_review"] = {
        "schema_version": "paa.source_answer_review.v1",
        "task_id": "ROUTE-FULL",
        "decision": "SUPPORTED_WITH_STATED_LIMITS",
    }
    packet.pop("projection_review", None)
    with pytest.raises(BootstrapSiteError, match="projection_review|cannot admit"):
        build_bootstrap_artifact(
            output_dir=tmp_path / "bad",
            matter_packets=[_minimal_matter(tmp_path)],
            question_packets=[],
            journey_packets=[packet],
            title="Poliittinen muisti",
        )


def test_negative_projection_receipt_rejected_as_projection_review(tmp_path: Path) -> None:
    mapping = JOURNEY_FIXTURE / "mapping_ROUTE-FULL.json"
    answer = JOURNEY_FIXTURE / "answer_ROUTE-FULL.json"
    bank = SourceQuestionBank.load(ROUTE_FIXTURE / "questions.json", ROUTE_FIXTURE / "sources.json")
    route = build_route_packet(
        bank,
        "ROUTE-FULL",
        as_of="2100-10-09",
        answer_packet=json.loads(answer.read_text(encoding="utf-8")),
    )
    route_path = tmp_path / "route.json"
    route_path.write_text(json.dumps(route), encoding="utf-8")
    negative = tmp_path / "negative.json"
    negative.write_text(
        json.dumps(
            {
                "schema_version": "paa.journey_projection_review_negative.v1",
                "task_id": "ROUTE-FULL",
                "decision": "NEEDS_MAPPING_SUCCESSOR_BEFORE_APPROVAL",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(IntegratedReaderError, match="paa.journey_projection_review.v1"):
        compose_journeys_from_manifest_entries(
            root=tmp_path,
            entries=[
                {
                    "task_id": "ROUTE-FULL",
                    "route_packet": route_path.name,
                    "mapping": str(mapping.resolve()),
                    "answer": str(answer.resolve()),
                    "projection_review": negative.name,
                }
            ],
        )


def test_journey_schema_constant_stable() -> None:
    assert JOURNEY_SCHEMA == "paa.question_journey.v1"

"""Source-to-page acceptance for the decision-episode renderer."""

from __future__ import annotations

import json
from pathlib import Path

from paa.decision_episodes import build_written_question_episode_from_db
from paa.episode_site import render_episode, render_episode_index
from paa.pipeline import _compile_episodes
from paa.question_ledger import normalize_question_records, parse_question_row
from paa.store import connect

FIXTURES = Path(__file__).parents[1] / "paa" / "contracts" / "fixtures"


def _fixture_rows(name: str) -> list[dict]:
    payload = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
    return [dict(zip(payload["columnNames"], row)) for row in payload["rowData"]]


def test_frozen_source_db_episode_and_page_keep_the_cold_reader_boundaries(tmp_path: Path) -> None:
    records = [
        parse_question_row(row, retrieved_at="2026-10-06T00:00:00+00:00")
        for filename in ("vaski_kk1_2023_rows.json", "vaski_kkv1_2023_rows.json")
        for row in _fixture_rows(filename)
    ]
    normalized = normalize_question_records(records)
    conn = connect(tmp_path / "episodes.sqlite")
    try:
        for obj in normalized["objects"]:
            conn.execute(
                "INSERT INTO official_objects(object_id, json) VALUES (?, ?)",
                (obj["object_id"], json.dumps(obj, ensure_ascii=False)),
            )
        for item in normalized["evidence"]:
            conn.execute(
                "INSERT INTO evidence(evidence_id, json) VALUES (?, ?)",
                (item["evidence_id"], json.dumps(item, ensure_ascii=False)),
            )
        conn.execute(
            "INSERT INTO source_coverage(coverage_id, json) VALUES (?, ?)",
            (
                "frozen-question-slice",
                json.dumps({
                    "coverage_id": "frozen-question-slice",
                    "source_id": "SRC-EDUSKUNTA-VASKI",
                    "kind": "WRITTEN_QUESTION_REGISTER",
                    "state": "PARTIAL_DECLARED_SLICE",
                    "complete": False,
                    "limitations": ["One real KK/KKV source slice."],
                }),
            ),
        )
        conn.execute("INSERT INTO source_coverage VALUES (?, ?)",
                     ("unrelated-speech-register", json.dumps({"coverage_id": "unrelated-speech-register", "kind": "SPEECH", "complete": True})))
        conn.commit()
        stats = _compile_episodes(conn)
        assert stats["decision_episodes"] == 1
        compiled = json.loads(conn.execute("SELECT json FROM decision_episodes").fetchone()[0])
        assert [item["coverage_id"] for item in compiled["coverage"]["items"]] == ["frozen-question-slice"]
        assert compiled["coverage"]["complete"] is False
        episode = build_written_question_episode_from_db(
            conn,
            "KK 1/2023 vp",
            actor_map={"1400": {"actor_id": "mp-1400", "identity_basis": "SOURCE_PERSON_ID"}},
        )
    finally:
        conn.close()

    page = render_episode(
        episode,
        normalized["evidence"],
        {item["object_id"] for item in normalized["objects"]},
        {"mp-1400": "Jussi Saramo"},
    )
    index = render_episode_index([episode])

    assert '<html lang="fi">' in page
    assert "Kirjallinen kysymys valtion omistajaohjauksen epäonnistumisesta" in page
    assert "Ensimmäinen allekirjoittaja / lähteessä nimetty laatija" in page
    assert "Rekisterin allekirjoittaja tai nimetty laatija ei yksin osoita" in page
    assert "Tytti Tuppurainen" in page
    assert "nimi ilman lähteen henkilötunnistetta" in page
    assert "../index.html#person=mp-1400" in page
    assert "Kirjallinen kysymys jätetty" in page
    assert "Hallituksen vastaus annettu" in page
    assert "Vastaus ilmoitettu täysistunnossa" in page
    assert page.index("2023-05-09") < page.index("2023-05-30")
    assert "Vastaus on rekisterissä ilman vastaustekstiä" in page
    assert "sisällöllisesti varmennetuksi" in page
    assert "NOT_ASSESSED" in page
    assert "PUBLIC_CONSTRAINTS_NOT_SOURCED" in page
    assert "ACTION_ALTERNATIVES_NOT_SOURCED" in page
    assert "../objects/eduskunta-KK-1-2023-vp.html" in page
    assert f"{episode['episode_id']}.json" in page
    assert "SRC-EDUSKUNTA-VASKI:" in page
    assert "https://avoindata.eduskunta.fi/" in page

    assert "Päätös- ja käsittelytapaukset" in index
    assert "episode-data" in index
    assert f"{episode['episode_id']}.html" in index
    assert "Jussi Saramo" in index

from __future__ import annotations

import json
from pathlib import Path

import paa.acquire_eduskunta as acquire
from paa.store import connect


def _ballot(person: str, response: str) -> dict:
    return {
        "aanestys_id": "vote-1",
        "person_number": person,
        "first_name": f"First{person}",
        "last_name": "Tester",
        "name_key": f"first{person} tester",
        "party": "x",
        "raw_response": response,
    }


def _payload(page: int, rows: list[list[str]], has_more: bool) -> dict:
    return {
        "page": page,
        "perPage": 100,
        "hasMore": has_more,
        "tableName": "SaliDBAanestysEdustaja",
        "columnNames": [
            "EdustajaId",
            "AanestysId",
            "EdustajaEtunimi",
            "EdustajaSukunimi",
            "EdustajaHenkiloNumero",
            "EdustajaRyhmaLyhenne",
            "EdustajaAanestys",
        ],
        "rowData": rows,
        "rowCount": len(rows),
    }


class _Response:
    status_code = 200

    def __init__(self, payload: dict, url: str):
        self.url = url
        self.content = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return json.loads(self.content)


class _Client:
    def __init__(self, payloads: list[dict]):
        self.payloads = payloads

    def get(self, url: str, *, params: dict) -> _Response:
        payload = self.payloads[params["page"]]
        return _Response(payload, f"{url}?page={params['page']}")


def _raw_row(person: str, response: str) -> list[str]:
    return [person, "vote-1", f"First{person}", "Tester", person, "x", response, ""]


def test_sync_fetch_keeps_hash_receipts_and_deduplicates_a_page_union(tmp_path: Path) -> None:
    client = _Client(
        [
            _payload(0, [_raw_row("1", "Jaa                 "), _raw_row("2", "Ei                  ")], True),
            _payload(1, [_raw_row("2", "Ei                  "), _raw_row("3", "Poissa              ")], False),
        ]
    )

    audit = acquire._fetch_vote_ballots_sync_audit("vote-1", raw_dir=tmp_path, client=client)

    assert {row["person_number"] for row in audit["rows"]} == {"1", "2", "3"}
    assert not audit["conflicts"]
    assert audit["pagination_complete"] is True
    assert len(audit["receipts"]) == 2
    for receipt in audit["receipts"]:
        body = tmp_path / f"{receipt['raw_sha256']}.json"
        manifest = tmp_path / f"{receipt['raw_sha256']}.manifest.json"
        observation = Path(receipt["observation_manifest_path"])
        assert body.exists() and manifest.exists()
        assert observation.exists()
        assert json.loads(manifest.read_text(encoding="utf-8"))["raw_sha256"] == receipt["raw_sha256"]


def test_attempt_union_refuses_conflicting_response_for_same_person() -> None:
    merged = acquire._merge_ballot_attempts(
        [
            {"rows": [_ballot("1", "JAA"), _ballot("2", "EI")], "conflicts": []},
            {"rows": [_ballot("1", "EI"), _ballot("3", "JAA")], "conflicts": []},
        ]
    )

    assert {row["person_number"] for row in merged["rows"]} == {"1", "2", "3"}
    assert merged["conflicts"] == [
        {
            "person_number": "1",
            "first_response": "JAA",
            "second_response": "EI",
            "kind": "cross_attempt_response_conflict",
        }
    ]


def test_metadata_buckets_must_match_before_complete() -> None:
    vote = {"jaa": 2, "ei": 1, "tyhjaa": 0, "poissa": 0, "yhteensa": 3}
    complete = acquire._assess_ballot_rows(
        [_ballot("1", "JAA"), _ballot("2", "JAA"), _ballot("3", "EI")], vote
    )
    incomplete = acquire._assess_ballot_rows(
        [_ballot("1", "JAA"), _ballot("2", "EI"), _ballot("3", "EI")], vote
    )

    assert complete["complete"] is True
    assert incomplete["complete"] is False
    assert incomplete["observed_buckets"] == {"JAA": 1, "EI": 2, "TYHJA": 0, "POISSA": 0}


def test_repair_writes_only_complete_union_and_preserves_existing_on_conflict(tmp_path: Path, monkeypatch) -> None:
    database = tmp_path / "paa.sqlite"
    conn = connect(database)
    conn.execute(
        """INSERT INTO vote_events(
            aanestys_id, year, session_date, jaa, ei, tyhjaa, poissa, yhteensa, mitatoity
        ) VALUES ('vote-1', 2026, '2026-01-01', 2, 1, 0, 0, 3, 0)"""
    )
    conn.execute(
        """INSERT INTO ballots(
            aanestys_id, person_number, first_name, last_name, name_key, party, raw_response
        ) VALUES ('vote-1', '1', 'First1', 'Tester', 'first1 tester', 'x', 'JAA')"""
    )
    conn.commit()
    conn.close()

    calls = []

    def fake_fetch(vote_id: str, *, attempt: int, **_kwargs) -> dict:
        calls.append(attempt)
        rows = [_ballot("1", "JAA"), _ballot("2", "JAA")] if attempt == 1 else [
            _ballot("1", "JAA"), _ballot("2", "JAA"), _ballot("3", "EI")
        ]
        return {"vote_id": vote_id, "attempt": attempt, "rows": rows, "conflicts": [], "raw_sha256": []}

    monkeypatch.setattr(acquire, "connect", lambda: connect(database))
    monkeypatch.setattr(acquire, "_fetch_vote_ballots_sync_audit", fake_fetch)
    result = acquire.repair_short_ballots()

    assert result["resolved"] == 1
    assert calls == [1, 2]
    conn = connect(database)
    assert conn.execute("SELECT COUNT(*) FROM ballots WHERE aanestys_id='vote-1'").fetchone()[0] == 3
    conn.close()


def test_repair_refuses_conflicting_attempt_and_keeps_existing_rows(tmp_path: Path, monkeypatch) -> None:
    database = tmp_path / "paa.sqlite"
    conn = connect(database)
    conn.execute(
        """INSERT INTO vote_events(
            aanestys_id, year, session_date, jaa, ei, tyhjaa, poissa, yhteensa, mitatoity
        ) VALUES ('vote-1', 2026, '2026-01-01', 2, 1, 0, 0, 3, 0)"""
    )
    conn.execute(
        """INSERT INTO ballots(
            aanestys_id, person_number, first_name, last_name, name_key, party, raw_response
        ) VALUES ('vote-1', '1', 'First1', 'Tester', 'first1 tester', 'x', 'JAA')"""
    )
    conn.commit()
    conn.close()

    def fake_fetch(vote_id: str, *, attempt: int, **_kwargs) -> dict:
        rows = [_ballot("1", "JAA"), _ballot("2", "JAA")] if attempt == 1 else [
            _ballot("1", "EI"), _ballot("2", "JAA"), _ballot("3", "EI")
        ]
        return {"vote_id": vote_id, "attempt": attempt, "rows": rows, "conflicts": [], "raw_sha256": []}

    monkeypatch.setattr(acquire, "connect", lambda: connect(database))
    monkeypatch.setattr(acquire, "_fetch_vote_ballots_sync_audit", fake_fetch)
    result = acquire.repair_short_ballots()

    assert result["resolved"] == 0
    assert result["unresolved"][0]["status"] == "CONFLICT"
    conn = connect(database)
    assert conn.execute("SELECT COUNT(*) FROM ballots WHERE aanestys_id='vote-1'").fetchone()[0] == 1
    assert conn.execute("SELECT raw_response FROM ballots WHERE aanestys_id='vote-1' AND person_number='1'").fetchone()[0] == "JAA"
    conn.close()

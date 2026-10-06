"""Source-grounded acquisition tests for Yle's named 2011 answer sheet."""

import csv
import hashlib
import io
import json
from pathlib import Path

from paa.acquire_yle import (
    PROMISE_API,
    YLE_2011_EXPORT,
    YLE_2011_LICENSE,
    YLE_2011_PUBLICATION,
    YLE_2011_SHEET_ID,
    acquire_2011,
    restore_cached_capture_dates,
)
from paa.records import statement_record
from paa.store import connect

QUESTION = "Mitä asioita haluat edistää tai ajaa tulevalla vaalikaudella"
CSV_HEADERS = ["Vaalipiiri", "id", "Sukunimi", "Etunimi", "Ehdokasnumero", QUESTION]


def test_missing_capture_time_stays_unknown_in_statement():
    document = {"document_id": "d", "source_id": "s", "field_label": "Answer",
                "language": "fi", "text": "Avoin lähde."}
    assert statement_record(document, "e", [], "UNRESOLVED")["retrieved_at"] is None
    document["retrieved_at"] = "2026-10-07T12:34:56+00:00"
    assert statement_record(document, "e", [], "UNRESOLVED")["retrieved_at"] == document["retrieved_at"]


def test_capture_date_recovery_requires_matching_raw_receipt_and_document(tmp_path):
    conn = connect(tmp_path / "capture.sqlite")
    text = "En leikkaa koulutuksesta."
    raw = json.dumps([{"id": 42, "info": {"election_promise_1": {"fi": text}}}]).encode()
    (tmp_path / "ekv2023-c1.json").write_bytes(raw)
    conn.execute("INSERT INTO documents(document_id, source_id, text, sha256) VALUES(?,?,?,?)",
                 ("yle2023-42-1", "SRC-YLE-2023", text, hashlib.sha256(text.encode()).hexdigest()))
    # A later acquisition date must remain later than the assessment cutoff.
    captured = "2026-10-07T12:34:56+00:00"
    conn.execute("INSERT INTO manifest(source_id,url,sha256,http_status,retrieved_at) VALUES(?,?,?,?,?)",
                 ("SRC-YLE-2023", PROMISE_API.format(cid=1), hashlib.sha256(raw).hexdigest(), 200, captured))
    assert restore_cached_capture_dates(conn, tmp_path)["documents_updated"] == 1
    assert conn.execute("SELECT retrieved_at FROM documents").fetchone()[0] == captured
    conn.execute("UPDATE documents SET retrieved_at=NULL")
    (tmp_path / "ekv2023-c1.json").write_bytes(raw + b" ")
    assert restore_cached_capture_dates(conn, tmp_path)["documents_updated"] == 0
    (tmp_path / "ekv2023-c1.json").write_bytes(raw)
    conn.execute("UPDATE documents SET text='changed text'")
    assert restore_cached_capture_dates(conn, tmp_path)["documents_updated"] == 0
    conn.close()


def _csv_body() -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(CSV_HEADERS)
    writer.writerow(
        [
            "01 Uusimaa",
            "row-1",
            "Doe",
            "Jane",
            "123",
            "Teen tästä lakialoitteen, jos minut valitaan.",
        ]
    )
    return output.getvalue().encode("utf-8")


def _publication_body() -> bytes:
    return (
        "<html><body>"
        f'<a href="https://docs.google.com/spreadsheets/d/{YLE_2011_SHEET_ID}/edit">answers</a>'
        f"<p>{YLE_2011_LICENSE}</p>"
        "<p>Yle Uutisten vaalikone 2011."
        " Lisenssi koskee datan käyttöä; linkitä tämä artikkeli.</p>"
        "</body></html>"
    ).encode()


def _fetcher(csv_body: bytes, publication_body: bytes, calls: list[str]):
    def fetch(url: str):
        calls.append(url)
        if url == YLE_2011_PUBLICATION:
            return 200, publication_body, "text/html"
        if url == YLE_2011_EXPORT:
            return 200, csv_body, "text/csv"
        raise AssertionError(f"unexpected URL: {url}")

    return fetch


def _candidate(conn) -> None:
    conn.execute(
        """INSERT INTO candidacies(
            candidacy_id, election_year, district_code, candidate_number,
            first_name, last_name, name_key
        ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
        ("vaalit-2011-01-123", 2011, "01", 123, "Jane", "Doe", "doe jane"),
    )
    conn.commit()


def test_acquire_2011_fetches_public_sheet_and_records_provenance(tmp_path: Path):
    csv_body = _csv_body()
    calls: list[str] = []
    raw_dir = tmp_path / "raw" / "yle"
    conn = connect(tmp_path / "paa.sqlite")
    _candidate(conn)

    result = acquire_2011(
        conn,
        raw_dir=raw_dir,
        fetcher=_fetcher(csv_body, _publication_body(), calls),
    )

    assert result["available"] is True
    assert result["cache_hit"] is False
    assert result["loaded"] == 1
    assert result["linked"] == 1
    assert calls == [YLE_2011_PUBLICATION, YLE_2011_EXPORT]

    manifest = json.loads((raw_dir / "yle2011.manifest.json").read_text(encoding="utf-8"))
    assert manifest["raw_sha256"] == hashlib.sha256(csv_body).hexdigest()
    assert manifest["raw_bytes"] == len(csv_body)
    assert manifest["known_hash_match"] is False
    assert manifest["publication_url"] == YLE_2011_PUBLICATION
    assert manifest["license"] == YLE_2011_LICENSE
    assert manifest["attribution"] == "Yle Uutisten vaalikone 2011"
    assert Path(manifest["archive_path"]).read_bytes() == csv_body

    document = conn.execute(
        "SELECT source_id,url,actor_id,text FROM documents WHERE document_id='yle2011-row-1'"
    ).fetchone()
    assert tuple(document) == (
        "SRC-YLE-2011",
        YLE_2011_PUBLICATION,
        "vaalit-2011-01-123",
        "Teen tästä lakialoitteen, jos minut valitaan.",
    )
    manifest_rows = conn.execute(
        "SELECT source_id,url,sha256 FROM manifest WHERE source_id LIKE 'SRC-YLE-2011%'"
    ).fetchall()
    assert {row[0] for row in manifest_rows} == {"SRC-YLE-2011", "SRC-YLE-2011-PUBLICATION"}
    conn.close()


def test_acquire_2011_replays_cache_and_refresh_can_fall_back(tmp_path: Path):
    csv_body = _csv_body()
    raw_dir = tmp_path / "raw"
    conn = connect(tmp_path / "paa.sqlite")
    first_calls: list[str] = []
    acquire_2011(conn, raw_dir=raw_dir, fetcher=_fetcher(csv_body, _publication_body(), first_calls))

    def should_not_be_called(_url: str):
        raise AssertionError("a valid cache should be replayed without network access")

    cached = acquire_2011(conn, raw_dir=raw_dir, fetcher=should_not_be_called)
    assert cached["available"] is True
    assert cached["cache_hit"] is True
    assert cached["cache_fallback"] is False

    def unavailable(_url: str):
        raise OSError("offline")

    fallback = acquire_2011(conn, raw_dir=raw_dir, refresh=True, fetcher=unavailable)
    assert fallback["available"] is True
    assert fallback["cache_hit"] is True
    assert fallback["cache_fallback"] is True
    assert "offline" in fallback["refresh_error"]
    conn.close()


def test_acquire_2011_rejects_unproven_publication_without_deleting_documents(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    conn = connect(tmp_path / "paa.sqlite")
    conn.execute(
        """INSERT INTO documents(
            document_id, source_id, url, actor_id, field_label, language, text,
            stated_earliest, sha256, http_status
        ) VALUES ('existing', 'SRC-YLE-2011', 'old', '', 'answer', 'fi', 'keep', '2011-03-01', 'x', 200)"""
    )
    conn.commit()

    def bad_publication(url: str):
        assert url == YLE_2011_PUBLICATION
        return 200, b"<html>This is not the cited source.</html>", "text/html"

    result = acquire_2011(conn, raw_dir=raw_dir, fetcher=bad_publication)

    assert result["available"] is False
    assert result["reason"] == "publication_not_source_grounded"
    assert conn.execute("SELECT text FROM documents WHERE document_id='existing'").fetchone()[0] == "keep"
    conn.close()

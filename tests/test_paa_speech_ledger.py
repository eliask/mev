"""Frozen-source tests for the official Eduskunta speech ledger."""

from __future__ import annotations

import hashlib
import json
from itertools import pairwise
from pathlib import Path

from paa.speech_ledger import (
    DEFAULT_WINDOWS,
    SPEECH_KIND,
    acquire_full_term,
    acquire_speeches,
    full_term_windows,
    import_speeches,
    normalize_speech_records,
    parse_speech_result,
)
from paa.store import connect

FIXTURES = Path(__file__).parents[1] / "paa" / "contracts" / "fixtures"


def _payload(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _request(window: dict, start: int = 0) -> dict:
    return {
        "category": "puheenvuoro",
        "maxResults": 100,
        "startFromIndex": start,
        "sort": [{"property": "aloitushetki", "ascending": True}],
        "expression": {
            "and": [
                {"property": "aloitushetki", "fromDate": window["from_date"], "toDate": window["to_date"]},
                {"property": "valtiopaivavuosi", "stringValue": window["year"]},
            ]
        },
    }


def test_full_term_windows_are_nonoverlapping_quarters() -> None:
    windows = full_term_windows((2023, 2024))

    assert len(windows) == 8
    assert windows[0] == {
        "window_id": "2023-Q1",
        "from_date": "2023-01-01",
        "to_date": "2023-04-01",
        "year": "2023",
    }
    assert windows[-1] == {
        "window_id": "2024-Q4",
        "from_date": "2024-10-01",
        "to_date": "2025-01-01",
        "year": "2024",
    }
    assert all(left["to_date"] == right["from_date"] for left, right in pairwise(windows))


def test_real_speech_result_has_source_speaker_date_topic_and_exact_text() -> None:
    path = FIXTURES / "eduskunta_speeches_2023-04-25_page0.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    result = payload["results"][0]
    window = DEFAULT_WINDOWS[0]
    obj = parse_speech_result(
        result,
        raw_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        raw_bytes=path.stat().st_size,
        request=_request(window),
        window_id=window["window_id"],
        result_index=0,
        page_start=0,
        retrieved_at="2026-10-06T00:00:00+00:00",
    )

    assert obj["object_id"] == "eduskunta:PUH 4/2023/3/1/1"
    assert obj["kind"] == SPEECH_KIND
    assert obj["matter_id"] == "HE 1/2023 vp"
    assert obj["action_date"] == "2023-04-25"
    assert obj["action_date_basis"] == "SPEECH_DATE"
    assert obj["authors"] == [{
        "person_id": "1503",
        "name": "Saku Nikkanen",
        "role": "ACTOR",
        "speaker_role": "SPEAKER",
        "position": None,
        "party": "sd",
        "parliamentary_group": "SD01~SOSIALIDEMOKRAATTINEN EDUSKUNTARYHMÄ",
        "identity_basis": "SOURCE_PERSON_ID",
        "evidence_ids": obj["authors"][0]["evidence_ids"],
    }]
    assert obj["topic"]["protocol_item_id"] == "PTK 4/2023 vp"
    assert obj["evidence"][0]["quote"] == obj["text"]
    assert obj["evidence"][0]["source_id"] == "SRC-EDUSKUNTA-SPEECHES"
    assert obj["disposition"]["policy_implementation"] == "NOT_ASSESSED"


def test_empty_official_speech_field_is_metadata_only_not_fabricated_quote() -> None:
    result = {
        "id": "PUH 1/2024/1/1/1",
        "puheenvuoro": {
            "id": "PUH 1/2024/1/1/1",
            "puheenvuoro": "",
            "aloitushetki": "2024-01-01T12:00:00+00:00",
            "lopetushetki": "2024-01-01T12:01:00+00:00",
            "valtiopaivavuosi": "2024",
            "puhuja": {"henkilonro": "1", "etunimi": "A", "sukunimi": "Testi"},
            "asia": {"fi": {"eduskuntatunnus": "HE 1/2024 vp", "nimeketeksti": "Testi"}},
            "poytakirjanasiankohta": {"fi": {"eduskuntatunnus": "PTK 1/2024 vp"}},
        },
    }
    obj = parse_speech_result(
        result,
        raw_sha256="raw",
        raw_bytes=1,
        request={},
        window_id="2024-Q1",
        result_index=0,
        page_start=0,
        retrieved_at="2026-10-06T00:00:00+00:00",
    )

    assert obj["text"] == ""
    assert obj["text_availability"] == "SOURCE_FIELD_EMPTY"
    assert obj["evidence"][0]["kind"] == "structured_field"
    assert obj["evidence"][0]["quote"] is None


def test_frozen_four_window_slice_normalizes_all_real_records() -> None:
    pages = [
        ("2023-04-25", "eduskunta_speeches_2023-04-25_page0.json", 0),
        ("2024-02-08", "eduskunta_speeches_2024-02-08_page0.json", 0),
        ("2025-02-06", "eduskunta_speeches_2025-02-06_page0.json", 0),
        ("2026-02-05", "eduskunta_speeches_2026-02-05_page0.json", 0),
        ("2026-02-05", "eduskunta_speeches_2026-02-05_page100.json", 100),
    ]
    windows = {window["window_id"]: window for window in DEFAULT_WINDOWS}
    records = []
    for window_id, name, start in pages:
        path = FIXTURES / name
        payload = _payload(name)
        for index, result in enumerate(payload["results"]):
            records.append(
                parse_speech_result(
                    result,
                    raw_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    raw_bytes=path.stat().st_size,
                    request=_request(windows[window_id], start),
                    window_id=window_id,
                    result_index=index,
                    page_start=start,
                    retrieved_at="2026-10-06T00:00:00+00:00",
                )
            )
    normalized = normalize_speech_records(records)
    receipt = _payload("eduskunta_speeches_coverage.json")

    assert len(normalized["objects"]) == receipt["declared_window_total"] == 277
    assert len({obj["object_id"] for obj in normalized["objects"]}) == 277
    assert normalized["identity_counts"] == {
        "speeches": 277,
        "speeches_with_text": 277,
        "speeches_without_text": 0,
        "speakers_with_person_id": 277,
        "speakers_name_only": 0,
        "speeches_without_speaker": 0,
    }
    assert {obj["action_date"][:4] for obj in normalized["objects"]} == {"2023", "2024", "2025", "2026"}
    assert receipt["full_current_term_count"]["total"] == 39_786
    assert receipt["full_term_acquisition_receipt"] == {
        "enumeration": "FULL_CURRENT_TERM_QUARTER_WINDOWS",
        "quarter_window_count": 16,
        "source_record_count": 39_786,
        "parsed_record_count": 39_786,
        "object_count": 39_786,
        "overlap_row_count": 0,
        "conflict_count": 0,
        "excluded_count": 0,
        "metadata_only_text_record_count": 4,
        "coverage_id": "speech-register-7584e97fb43ede1646e7",
        "raw_cache": "data/raw/eduskunta/speeches_fullterm_quarters",
    }
    expected_hashes = receipt["fixture_response_sha256"]
    fixture_paths = {
        "count-2023-04-25": "eduskunta_speeches_2023-04-25_count.json",
        "search-2023-04-25-0": "eduskunta_speeches_2023-04-25_page0.json",
        "count-2024-02-08": "eduskunta_speeches_2024-02-08_count.json",
        "search-2024-02-08-0": "eduskunta_speeches_2024-02-08_page0.json",
        "count-2025-02-06": "eduskunta_speeches_2025-02-06_count.json",
        "search-2025-02-06-0": "eduskunta_speeches_2025-02-06_page0.json",
        "count-2026-02-05": "eduskunta_speeches_2026-02-05_count.json",
        "search-2026-02-05-0": "eduskunta_speeches_2026-02-05_page0.json",
        "search-2026-02-05-100": "eduskunta_speeches_2026-02-05_page100.json",
    }
    assert set(expected_hashes) == set(fixture_paths)
    assert all(
        hashlib.sha256((FIXTURES / fixture_paths[key]).read_bytes()).hexdigest() == digest
        for key, digest in expected_hashes.items()
    )


class _Response:
    def __init__(self, payload: dict, url: str) -> None:
        self.content = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.url = url
        self.status_code = 200

    def raise_for_status(self) -> None:
        return None


class _RetryResponse(_Response):
    def __init__(self, url: str) -> None:
        super().__init__({}, url)
        self.status_code = 429
        self.headers = {"retry-after": "0"}


class _FixtureClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def post(self, url: str, **kwargs):
        request = dict(kwargs["json"])
        self.calls.append((url, request))
        year = request["expression"]["and"][1]["stringValue"]
        from_date = request["expression"]["and"][0]["fromDate"]
        if url.endswith("/count"):
            payload = _payload(f"eduskunta_speeches_{from_date}_count.json")
            return _Response(payload, url)
        start = int(request["startFromIndex"])
        suffix = f"page{start}.json"
        payload = _payload(f"eduskunta_speeches_{from_date}_{suffix}")
        assert payload["results"]
        assert year == from_date[:4]
        return _Response(payload, url)


class _RetryOnceClient(_FixtureClient):
    def __init__(self) -> None:
        super().__init__()
        self.retry_once = True

    def post(self, url: str, **kwargs):
        if url.endswith("/count") and self.retry_once:
            self.retry_once = False
            return _RetryResponse(url)
        return super().post(url, **kwargs)


class _EmptyFullTermClient:
    def post(self, url: str, **kwargs):
        return _Response({"count": 0}, url)

    def get(self, url: str, **kwargs):
        return _Response({"searchMetadata": {"totalResultCount": 0}, "results": []}, url)


def test_callable_acquisition_replays_hashed_pages_without_network(tmp_path: Path) -> None:
    client = _FixtureClient()
    result = acquire_speeches(raw_dir=tmp_path, client=client)
    coverage = result["coverage"]

    assert coverage["state"] == "ENUMERATED"
    assert coverage["complete"] is True
    assert coverage["expected_count"] == 277
    assert coverage["source_record_count"] == 277
    assert coverage["object_count"] == 277
    assert coverage["overlap_row_count"] == 0
    assert len(coverage["page_manifests"]) == 5
    assert len(client.calls) == 9  # one count plus one/two search pages per window

    class NoNetwork:
        def post(self, *_args, **_kwargs):
            raise AssertionError("verified speech checkpoints should replay without network")

    replay = acquire_speeches(raw_dir=tmp_path, client=NoNetwork())
    assert replay["coverage"]["coverage_id"] == coverage["coverage_id"]
    assert replay["coverage"]["expected_count"] == 277


def test_retryable_public_api_response_is_bounded_and_receipted(tmp_path: Path, monkeypatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr("paa.speech_ledger.time.sleep", sleeps.append)
    client = _RetryOnceClient()
    result = acquire_speeches(raw_dir=tmp_path, client=client, max_attempts=4)

    assert result["coverage"]["complete"] is True
    count_receipt = json.loads((tmp_path / "count-2023-04-25.manifest.json").read_text(encoding="utf-8"))
    assert count_receipt["attempts"] == 2
    assert sleeps == [0.0]


def test_full_term_wrapper_declares_quarter_population_and_get_search(tmp_path: Path) -> None:
    result = acquire_full_term(
        years=(2023,),
        raw_dir=tmp_path,
        client=_EmptyFullTermClient(),
        request_delay=0.0,
    )
    coverage = result["coverage"]

    assert coverage["enumeration"] == "FULL_CURRENT_TERM_QUARTER_WINDOWS"
    assert coverage["full_term_years"] == ["2023"]
    assert len(coverage["windows"]) == 4
    assert coverage["search_method"] == "GET"
    assert coverage["complete"] is True
    assert coverage["expected_count"] == 0


def test_speech_import_persists_only_declared_objects_and_receipts(tmp_path: Path) -> None:
    result = acquire_speeches(raw_dir=tmp_path, client=_FixtureClient())
    conn = connect(tmp_path / "speech.sqlite")
    try:
        summary = import_speeches(conn, result)
        conn.commit()
        assert summary == {"official_objects": 277, "evidence": 277, "source_coverage": 1}
        assert conn.execute("SELECT COUNT(*) FROM official_objects").fetchone()[0] == 277
        assert conn.execute("SELECT COUNT(*) FROM evidence").fetchone()[0] == 277
        stored = json.loads(conn.execute("SELECT json FROM official_objects ORDER BY object_id LIMIT 1").fetchone()[0])
        assert stored["kind"] == SPEECH_KIND
        assert conn.execute("SELECT COUNT(*) FROM source_coverage").fetchone()[0] == 1
    finally:
        conn.close()

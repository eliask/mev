from __future__ import annotations

import json
from pathlib import Path

import pytest

from paa.initiative_ledger import acquire_registry, import_result
from paa.store import connect


class PageClient:
    def __init__(self, payloads):
        self.payloads = iter(payloads)

    def get(self, url, **kwargs):
        payload = next(self.payloads)
        class Response:
            content = json.dumps(payload).encode()
            def raise_for_status(self):
                pass
        result = Response()
        result.url = url + "?" + str(kwargs["params"])
        return result


def page(has_more=False):
    xml = (Path(__file__).resolve().parents[1] / "paa/contracts/fixtures/vaski_la72_content.xml").read_text()
    return {"columnNames": ["Id", "Eduskuntatunnus", "XmlData"], "rowData": [["85933", "LA 72/2017 vp", xml]], "hasMore": has_more}


def test_paged_register_import_preserves_official_actor_and_raw_receipt(tmp_path):
    result = acquire_registry([2017], raw_dir=tmp_path / "raw", client=PageClient([page()]))
    assert result["coverage"]["source_record_count"] == result["coverage"]["parsed_records"] == 1
    conn = connect(tmp_path / "db.sqlite")
    import_result(conn, result)
    obj = json.loads(conn.execute("SELECT json FROM official_objects").fetchone()[0])
    assert obj["authors"][0]["person_id"] == "1129"
    evidence = json.loads(conn.execute("SELECT json FROM evidence").fetchone()[0])
    assert evidence["raw_sha256"] and evidence["record_locator"] and evidence["url"]
    conn.close()


def test_repeated_api_page_withholds_coverage(tmp_path):
    with pytest.raises(RuntimeError, match="repeated records"):
        acquire_registry([2017], raw_dir=tmp_path, client=PageClient([page(True), page(False)]))
    assert not (tmp_path / "coverage.json").exists()


def test_corrupt_checkpoint_cannot_be_replayed_as_valid_source(tmp_path):
    acquire_registry([2017], raw_dir=tmp_path, client=PageClient([page()]))
    (tmp_path / "la-2017-0.json").write_text("{}")
    with pytest.raises(ValueError, match="corrupt initiative checkpoint"):
        acquire_registry([2017], raw_dir=tmp_path, client=PageClient([]))


def test_disjoint_identifier_partitions_close_without_repeated_year_pages(tmp_path):
    empty = {"columnNames": ["Id", "Eduskuntatunnus", "XmlData"], "rowData": [], "hasMore": False}
    pages = [empty] * 7 + [page()] + [empty] * 2
    result = acquire_registry([2017], raw_dir=tmp_path, client=PageClient(pages), partition_identifiers=True)
    assert result["coverage"]["query_partition"] == "FIRST_IDENTIFIER_DIGIT_0_TO_9"
    assert result["coverage"]["source_record_count"] == 1
    assert len(result["manifests"]) == 10


def test_refresh_preserves_previous_immutable_source_body(tmp_path):
    first = acquire_registry([2017], raw_dir=tmp_path, client=PageClient([page()]))
    changed = page()
    changed["retrievalNote"] = "source revision"
    second = acquire_registry([2017], raw_dir=tmp_path, client=PageClient([changed]), refresh=True)
    old = Path(first["manifests"][0]["artifact_path"])
    new = Path(second["manifests"][0]["artifact_path"])
    assert old != new and old.exists() and new.exists()

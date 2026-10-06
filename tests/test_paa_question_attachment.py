"""Official attachment PDF fallback controls (no network)."""

from paa.question_ledger import export_answer_attachment_artifact, fetch_official_attachment_pdf


class _FakeResponse:
    def __init__(self, *, status_code=200, json_data=None, content=b"", headers=None):
        self.status_code = status_code
        self._json = json_data
        self.content = content
        self.headers = headers or {"content-type": "application/pdf"}

    def json(self):
        return self._json


class _FakeClient:
    def __init__(self, mapping):
        self.mapping = mapping

    def get(self, url, headers=None, timeout=None):
        for key in sorted(self.mapping, key=len, reverse=True):
            if key in url:
                return self.mapping[key]
        return _FakeResponse(status_code=404, content=b"missing")


def test_exact_attachment_chain_and_md5_control(tmp_path) -> None:
    pdf = b"%PDF-1.4 fixture body"
    import hashlib

    md5 = hashlib.md5(pdf).hexdigest()
    client = _FakeClient(
        {
            "tables/AttachmentGroup/rows": _FakeResponse(
                json_data={
                    "columnNames": ["Id", "Imported"],
                    "rowData": [["776811", "2025-01-01"]],
                }
            ),
            "tables/Attachment/rows": _FakeResponse(
                json_data={
                    "columnNames": ["Id", "AttachmentGroupId", "Url", "Imported"],
                    "rowData": [
                        [
                            "778538",
                            "776811",
                            '"https://s3.example.test/official.pdf"',
                            "2025-01-01",
                        ]
                    ],
                }
            ),
            "s3.example.test/official.pdf": _FakeResponse(content=pdf),
        }
    )
    attachment = fetch_official_attachment_pdf(
        attachment_group_id="776811",
        declared_md5=md5,
        client=client,
    )
    assert attachment["md5_matched_declared"] is True
    assert attachment["sha256"]
    export = export_answer_attachment_artifact(
        record={
            "document_id": "KKV 1/2023 vp",
            "matter_id": "KK 1/2023 vp",
            "title": "Vastaus",
            "publication_date": "2023-05-09",
            "authors": [{"name": "Tytti Tuppurainen", "role": "ministeri"}],
            "raw_sha256": "a" * 64,
            "source_row": {"attachment_group_id": "776811"},
        },
        attachment=attachment,
        output_dir=tmp_path / "kkv_export",
        pdf_text="Helsingissä 9.5.2023",
    )
    assert export["db_writes"] is False
    assert (tmp_path / "kkv_export" / "answer.pdf").read_bytes() == pdf
    assert export["authors"][0]["name"] == "Tytti Tuppurainen"


def test_http_failure_is_explicit() -> None:
    client = _FakeClient({"AttachmentGroup": _FakeResponse(status_code=503, json_data={})})
    try:
        fetch_official_attachment_pdf(attachment_group_id="776811", client=client)
        raise AssertionError("expected failure")
    except ValueError as error:
        assert "HTTP 503" in str(error)

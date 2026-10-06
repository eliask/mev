"""Replay one genuine Eduskunta vote and its complete ballot register offline.

The fixture is intentionally source-shaped: the metadata response and both
ballot pages are retained as raw JSON files with byte hashes.  This helper
does not infer parties, roles or policy positions.  It only verifies that the
officially published metadata, all four response buckets and every stable
delegate identifier reconcile before inserting the rows into a frozen SQLite
database.
"""


import hashlib
import json
import sqlite3
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from paa.acquire_eduskunta import _ballot_row
from paa.config import FIXTURE_DIR
from paa.store import add_manifest

DEFAULT_FIXTURE = FIXTURE_DIR / "eduskunta_vote_52877_fixture.json"
_BALLOT_COLUMNS = (
    "EdustajaId",
    "AanestysId",
    "EdustajaEtunimi",
    "EdustajaSukunimi",
    "EdustajaHenkiloNumero",
    "EdustajaRyhmaLyhenne",
    "EdustajaAanestys",
    "Imported",
)
_META_COLUMNS_REQUIRED = {
    "AanestysId",
    "KieliId",
    "IstuntoVPVuosi",
    "IstuntoAlkuaika",
    "AanestysNumero",
    "AanestysMitatoity",
    "AanestysOtsikko",
    "AanestysLisaOtsikko",
    "KohtaOtsikko",
    "KohtaKasittelyVaihe",
    "AanestysTulosJaa",
    "AanestysTulosEi",
    "AanestysTulosTyhjia",
    "AanestysTulosPoissa",
    "AanestysTulosYhteensa",
    "Url",
    "AanestysPoytakirja",
    "AanestysValtiopaivaasia",
}


class FrozenVoteError(ValueError):
    """Raised when an immutable official vote fixture cannot be reconciled."""


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_sha(value: Any) -> str:
    body = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _sha_bytes(body)


def _int(value: Any, field: str) -> int:
    try:
        return int(str(value or "").strip())
    except (TypeError, ValueError) as error:
        raise FrozenVoteError(f"invalid integer {field}: {value!r}") from error


def _text(value: Any) -> str:
    return str(value or "").strip()


def _read_json(path: Path) -> tuple[dict[str, Any], bytes, str]:
    body = path.read_bytes()
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as error:
        raise FrozenVoteError(f"invalid JSON fixture: {path}") from error
    if not isinstance(payload, dict):
        raise FrozenVoteError(f"fixture payload is not an object: {path}")
    return payload, body, _sha_bytes(body)


def _metadata(path: Path, spec: Mapping[str, Any], vote_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    payload, body, digest = _read_json(path)
    if digest != spec.get("raw_sha256") or len(body) != int(spec.get("raw_bytes") or -1):
        raise FrozenVoteError(f"metadata raw receipt mismatch: {path}")
    columns = payload.get("columnNames")
    rows = payload.get("rowData")
    if columns is None or rows is None or not _META_COLUMNS_REQUIRED <= set(columns):
        raise FrozenVoteError("metadata response does not expose the required published fields")
    if payload.get("tableName") != "SaliDBAanestys" or len(rows) != 1 or payload.get("rowCount") != 1:
        raise FrozenVoteError("metadata response is not one complete vote row")
    row = dict(zip(columns, rows[0]))
    if _text(row.get("AanestysId")) != vote_id or _text(row.get("KieliId")) not in {"1", "1.0"}:
        raise FrozenVoteError("metadata fixture identifies a different or non-Finnish vote")
    expected = {
        "row_count": int(spec.get("row_count") or 1),
        "unique_person_count": int(spec.get("unique_person_count") or 1),
        "published_totals": dict(spec["expected"]["published_totals"]),
        "mitatoity": int(spec["expected"]["mitatoity"]),
        "year": int(spec["expected"]["year"]),
        "session_date": str(spec["expected"]["session_date"]),
        "number": int(spec["expected"]["number"]),
        "matter": str(spec["expected"]["matter"]),
        "title": str(spec["expected"]["title"]),
    }
    metadata = {
        "aanestys_id": vote_id,
        "year": _int(row.get("IstuntoVPVuosi"), "IstuntoVPVuosi"),
        "session_date": _text(row.get("IstuntoAlkuaika"))[:10],
        "number": _int(row.get("AanestysNumero"), "AanestysNumero"),
        "title": _text(row.get("AanestysOtsikko")),
        "lisa": _text(row.get("AanestysLisaOtsikko")),
        "kohta": _text(row.get("KohtaOtsikko")),
        "jaa": _int(row.get("AanestysTulosJaa"), "AanestysTulosJaa"),
        "ei": _int(row.get("AanestysTulosEi"), "AanestysTulosEi"),
        "tyhjaa": _int(row.get("AanestysTulosTyhjia"), "AanestysTulosTyhjia"),
        "poissa": _int(row.get("AanestysTulosPoissa"), "AanestysTulosPoissa"),
        "yhteensa": _int(row.get("AanestysTulosYhteensa"), "AanestysTulosYhteensa"),
        "url": "https://www.eduskunta.fi" + _text(row.get("Url")),
        "ptk": _text(row.get("AanestysPoytakirja")),
        "matter": _text(row.get("AanestysValtiopaivaasia")),
        "mitatoity": _int(row.get("AanestysMitatoity"), "AanestysMitatoity"),
        "json": json.dumps(
            {
                "vaihe": _text(row.get("KohtaKasittelyVaihe")),
                "paakohta": _text(row.get("PaaKohtaOtsikko")),
                "timezone_assumption": "Europe/Helsinki",
                "metadata_raw_sha256": digest,
            },
            ensure_ascii=False,
        ),
        "raw_sha256": digest,
        "raw_bytes": len(body),
        "record_locator": spec.get("record_locator"),
    }
    checks = {
        "row_count": 1,
        "unique_person_count": 1,
        "published_totals": {
            "JAA": metadata["jaa"],
            "EI": metadata["ei"],
            "TYHJA": metadata["tyhjaa"],
            "POISSA": metadata["poissa"],
            "TOTAL": metadata["yhteensa"],
        },
        "mitatoity": metadata["mitatoity"],
        "year": metadata["year"],
        "session_date": metadata["session_date"],
        "number": metadata["number"],
        "matter": metadata["matter"],
        "title": metadata["title"],
    }
    if checks != expected:
        raise FrozenVoteError(f"metadata does not match declared fixture expectations: {checks!r} != {expected!r}")
    return metadata, {
        "source_id": spec["source_id"],
        "url": spec["url"],
        "request": spec["request"],
        "raw_sha256": digest,
        "raw_bytes": len(body),
        "record_locator": spec["record_locator"],
        "fixture": path.name,
    }


def _read_ballot_page(path: Path, spec: Mapping[str, Any], vote_id: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload, body, digest = _read_json(path)
    if digest != spec.get("raw_sha256") or len(body) != int(spec.get("raw_bytes") or -1):
        raise FrozenVoteError(f"ballot raw receipt mismatch: {path}")
    if payload.get("tableName") != "SaliDBAanestysEdustaja":
        raise FrozenVoteError(f"unexpected ballot table: {path}")
    if payload.get("page") != spec.get("page") or payload.get("rowCount") != spec.get("row_count"):
        raise FrozenVoteError(f"ballot page metadata mismatch: {path}")
    if bool(payload.get("hasMore")) != bool(spec.get("has_more")):
        raise FrozenVoteError(f"ballot page continuation mismatch: {path}")
    columns = payload.get("columnNames")
    if tuple(columns or ()) != _BALLOT_COLUMNS:
        raise FrozenVoteError(f"ballot columns changed: {path}")
    raw_rows = payload.get("rowData")
    if not isinstance(raw_rows, list) or len(raw_rows) != int(spec["row_count"]):
        raise FrozenVoteError(f"ballot row count mismatch: {path}")
    rows: list[dict[str, Any]] = []
    for raw in raw_rows:
        row = _ballot_row(vote_id, dict(zip(columns, raw)))
        if row["aanestys_id"] != vote_id or not row["person_number"]:
            raise FrozenVoteError(f"ballot row has wrong vote or no stable person ID: {path}")
        rows.append(row)
    receipt = {
        "source_id": spec.get("source_id") or f"SRC-EDUSKUNTA-BALLOTS-{vote_id}",
        "url": spec.get("url"),
        "request": spec.get("request"),
        "page": spec["page"],
        "raw_sha256": digest,
        "raw_bytes": len(body),
        "row_count": len(rows),
        "has_more": bool(payload.get("hasMore")),
        "record_locator": f"SaliDBAanestysEdustaja.AanestysId={vote_id};page={spec['page']}",
        "fixture": path.name,
    }
    return rows, receipt


def _reconcile_rows(rows: Sequence[Mapping[str, Any]], expected: Mapping[str, Any]) -> dict[str, Any]:
    by_person: dict[str, dict[str, Any]] = {}
    conflicts: list[dict[str, Any]] = []
    duplicate_ids: list[str] = []
    for row in rows:
        person_id = _text(row.get("person_number"))
        previous = by_person.get(person_id)
        if previous is not None:
            duplicate_ids.append(person_id)
            if previous.get("raw_response") != row.get("raw_response"):
                conflicts.append(
                    {
                        "person_number": person_id,
                        "first_response": previous.get("raw_response"),
                        "second_response": row.get("raw_response"),
                        "kind": "duplicate_person_response_conflict",
                    }
                )
            continue
        by_person[person_id] = dict(row)
    if duplicate_ids:
        raise FrozenVoteError(f"duplicate ballot person IDs: {sorted(set(duplicate_ids))}")
    if conflicts:
        raise FrozenVoteError(f"conflicting ballot responses: {conflicts[:3]}")
    merged = sorted(by_person.values(), key=lambda row: (int(row["person_number"]) if str(row["person_number"]).isdigit() else 10**12, str(row["person_number"])))
    observed = Counter(row.get("raw_response") for row in merged)
    buckets = {bucket: int(observed.get(bucket, 0)) for bucket in ("JAA", "EI", "TYHJA", "POISSA")}
    buckets["TOTAL"] = len(merged)
    expected_buckets = dict(expected["published_totals"])
    if len(merged) != int(expected["row_count"]) or len(by_person) != int(expected["unique_person_count"]):
        raise FrozenVoteError("ballot rows do not reconcile to the declared full row and unique-person counts")
    if buckets != expected_buckets:
        raise FrozenVoteError(f"all four ballot buckets do not reconcile: {buckets!r} != {expected_buckets!r}")
    return {
        "rows": merged,
        "row_count": len(merged),
        "unique_person_count": len(by_person),
        "observed_totals": buckets,
        "conflicts": conflicts,
        "duplicate_person_ids": sorted(set(duplicate_ids)),
    }


def load_frozen_vote_fixture(fixture_path: Path | str = DEFAULT_FIXTURE) -> dict[str, Any]:
    """Verify and normalize the complete official vote fixture offline."""

    fixture_path = Path(fixture_path)
    spec, fixture_body, fixture_sha256 = _read_json(fixture_path)
    if spec.get("fixture_kind") != "EDUSKUNTA_BALLOT_VOTE":
        raise FrozenVoteError("not an Eduskunta ballot fixture")
    vote_id = _text(spec.get("vote_id"))
    if not vote_id:
        raise FrozenVoteError("fixture has no vote ID")
    metadata_spec = dict(spec.get("metadata_fixture") or {})
    metadata_spec["expected"] = spec.get("expected") or {}
    metadata_path = fixture_path.parent / str(metadata_spec.get("file") or "")
    metadata, metadata_receipt = _metadata(metadata_path, metadata_spec, vote_id)
    pages_spec = list((spec.get("ballot_source") or {}).get("pages") or [])
    if not pages_spec:
        raise FrozenVoteError("fixture has no ballot pages")
    all_rows: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = [metadata_receipt]
    previous_page = -1
    for page_spec in pages_spec:
        page_number = int(page_spec.get("page"))
        if page_number != previous_page + 1:
            raise FrozenVoteError("ballot pages are not a contiguous bounded sequence")
        page_path = fixture_path.parent / str(page_spec.get("file") or "")
        page_spec = dict(page_spec)
        page_spec["url"] = (spec.get("ballot_source") or {}).get("url")
        page_spec["request"] = {
            **dict((spec.get("ballot_source") or {}).get("request_template") or {}),
            "page": page_number,
        }
        rows, receipt = _read_ballot_page(page_path, page_spec, vote_id)
        all_rows.extend(rows)
        receipts.append(receipt)
        previous_page = page_number
    if receipts[-1].get("has_more"):
        raise FrozenVoteError("last ballot fixture page still advertises more rows")
    reconciled = _reconcile_rows(all_rows, spec["expected"])
    normalized_rows = [
        {
            "person_id": row["person_number"],
            "group_code": row["party"].casefold(),
            "response": row["raw_response"],
            "first_name": row["first_name"],
            "last_name": row["last_name"],
        }
        for row in reconciled["rows"]
    ]
    normalized_source = {
        "normalization_version": spec.get("normalization_version"),
        "source_id": spec.get("source_id"),
        "vote_id": vote_id,
        "metadata_raw_sha256": metadata["raw_sha256"],
        "published_totals": spec["expected"]["published_totals"],
        "rows": normalized_rows,
    }
    normalized_sha256 = _canonical_sha(normalized_source)
    return {
        "schema_version": spec.get("schema_version"),
        "fixture_kind": spec.get("fixture_kind"),
        "fixture": fixture_path.name,
        "fixture_sha256": fixture_sha256,
        "source_id": spec.get("source_id"),
        "vote_id": vote_id,
        "metadata": metadata,
        "metadata_receipt": metadata_receipt,
        "ballot_receipts": receipts[1:],
        "raw_receipts": receipts,
        "rows": reconciled["rows"],
        "normalized_rows": normalized_rows,
        "row_count": reconciled["row_count"],
        "unique_person_count": reconciled["unique_person_count"],
        "published_totals": dict(spec["expected"]["published_totals"]),
        "observed_totals": reconciled["observed_totals"],
        "normalized_sha256": normalized_sha256,
        "record_locator": (spec.get("ballot_source") or {}).get("record_locator"),
        "rights_status": spec.get("rights_status"),
        "limitations": list(spec.get("limitations") or []),
        "raw_fixture_bytes": len(fixture_body),
    }


def import_frozen_vote_fixture(conn: sqlite3.Connection, fixture_path: Path | str = DEFAULT_FIXTURE) -> dict[str, Any]:
    """Insert the verified event and ballot rows into a frozen SQLite DB."""

    fixture = load_frozen_vote_fixture(fixture_path)
    metadata = fixture["metadata"]
    conn.execute(
        """INSERT OR REPLACE INTO vote_events(
             aanestys_id, year, session_date, number, title, lisa, kohta,
             jaa, ei, tyhjaa, poissa, yhteensa, url, ptk, matter, mitatoity, json
           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            metadata["aanestys_id"], metadata["year"], metadata["session_date"], metadata["number"],
            metadata["title"], metadata["lisa"], metadata["kohta"], metadata["jaa"], metadata["ei"],
            metadata["tyhjaa"], metadata["poissa"], metadata["yhteensa"], metadata["url"], metadata["ptk"],
            metadata["matter"], metadata["mitatoity"], metadata["json"],
        ),
    )
    conn.execute("DELETE FROM ballots WHERE aanestys_id = ?", (fixture["vote_id"],))
    conn.executemany(
        """INSERT INTO ballots(
             aanestys_id, person_number, first_name, last_name, name_key, party, raw_response
           ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
        [
            (
                fixture["vote_id"], row["person_number"], row["first_name"], row["last_name"],
                f"{row['last_name'].casefold()}|{row['first_name'].casefold()}", row["party"], row["raw_response"],
            )
            for row in fixture["rows"]
        ],
    )
    for receipt in fixture["raw_receipts"]:
        add_manifest(
            conn,
            source_id=receipt["source_id"],
            url=receipt["url"],
            sha256=receipt["raw_sha256"],
            bytes=receipt["raw_bytes"],
            http_status=200,
            retrieved_at=None,
            note=f"Frozen official source receipt; {receipt['record_locator']}; fixture={receipt['fixture']}",
        )
    return {
        "source_id": fixture["source_id"],
        "vote_id": fixture["vote_id"],
        "row_count": fixture["row_count"],
        "unique_person_count": fixture["unique_person_count"],
        "published_totals": fixture["published_totals"],
        "observed_totals": fixture["observed_totals"],
        "normalized_sha256": fixture["normalized_sha256"],
        "raw_receipt_count": len(fixture["raw_receipts"]),
        "fixture_sha256": fixture["fixture_sha256"],
    }


__all__ = [
    "DEFAULT_FIXTURE",
    "FrozenVoteError",
    "import_frozen_vote_fixture",
    "load_frozen_vote_fixture",
]

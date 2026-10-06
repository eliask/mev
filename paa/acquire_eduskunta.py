"""Parliament open data: members, plenary votes, and per-MP ballots.

Adapted from the request shape used by the public mev Vaski client
(`tables/{name}/rows`), with paths and storage local to this repository.
"""


import asyncio
import hashlib
import json
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import httpx
from lxml import etree

from paa.config import (
    BALLOT_YEARS,
    CORPUS_CUTOFF,
    EDUSKUNTA_API,
    RAW,
    TERM_START,
    USER_AGENT,
    VOTE_META_YEARS,
)
from paa.identity import name_key
from paa.store import connect

_DATE = re.compile(r"(\d{1,2})\.(\d{1,2})\.(\d{4})")
_YEAR = re.compile(r"(\d{4})")


def _local(el) -> str:
    return etree.QName(el).localname if isinstance(el.tag, str) else ""


def _text(el) -> str:
    return " ".join((el.text or "").split())


def _parse_date(value: str) -> tuple[str | None, str]:
    value = (value or "").strip()
    match = _DATE.search(value)
    if match:
        day, month, year = (int(part) for part in match.groups())
        return f"{year:04d}-{month:02d}-{day:02d}", "day"
    match = _YEAR.search(value)
    if match:
        return f"{match.group(1)}-01-01", "year"
    return None, "unknown"


def _child_text(el, name: str) -> str:
    for child in el.iter():
        if _local(child) == name and child is not el:
            value = _text(child)
            if value:
                return value
    return ""


def parse_member_xml(xml: str, person_id: str, first: str, last: str, minister: str) -> dict:
    periods = []
    birth_year = None
    death = None
    ended = None
    if xml:
        try:
            root = etree.fromstring(xml.encode("utf-8"))
        except etree.XMLSyntaxError:
            root = None
        if root is not None:
            for el in root.iter():
                tag = _local(el)
                if tag == "SyntymaPvm" and birth_year is None:
                    parsed, _precision = _parse_date(_text(el))
                    if parsed:
                        birth_year = int(parsed[:4])
                elif tag == "KuolemaPvm" and _text(el):
                    death, _precision = _parse_date(_text(el))
                elif tag == "KansanedustajuusPaattynytPvm" and _text(el):
                    ended, _precision = _parse_date(_text(el))
                elif tag in {"VaaliPiiri", "Eduskuntaryhma", "Edustajatoimi"}:
                    start, precision = _parse_date(_child_text(el, "AlkuPvm"))
                    end, _end_precision = _parse_date(_child_text(el, "LoppuPvm"))
                    label = _child_text(el, "Nimi") or tag
                    if start or end or label:
                        periods.append(
                            {
                                "person_id": person_id,
                                "kind": tag,
                                "label": label,
                                "start_date": start,
                                "end_date": end,
                                "precision": precision,
                            }
                        )
    return {
        "person_id": str(person_id).strip(),
        "first_name": (first or "").strip(),
        "last_name": (last or "").strip(),
        "name_key": name_key(first or "", last or ""),
        "birth_year": birth_year,
        "death_date": death,
        "ended_date": ended,
        "minister": 1 if str(minister).strip().lower() in {"t", "true", "1"} else 0,
        "periods": periods,
    }


def _rows(table: str, params: dict) -> list[dict]:
    page = 0
    collected = []
    while True:
        query = {"perPage": 100, "page": page, **params}
        response = httpx.get(
            f"{EDUSKUNTA_API}/{table}/rows",
            params=query,
            headers={"User-Agent": USER_AGENT},
            timeout=90,
        )
        response.raise_for_status()
        payload = response.json()
        columns = payload["columnNames"]
        for raw in payload["rowData"]:
            collected.append(dict(zip(columns, raw)))
        if not payload.get("hasMore"):
            break
        page += 1
        if page > 500:
            raise RuntimeError(f"pagination runaway for {table} {params}")
    return collected


def acquire_members() -> dict:
    conn = connect()
    people = _rows("MemberOfParliament", {})
    conn.execute("DELETE FROM mp_people")
    conn.execute("DELETE FROM mp_periods")
    current = 0
    for raw in people:
        parsed = parse_member_xml(
            raw.get("XmlDataFi") or "",
            raw.get("personId"),
            raw.get("firstname"),
            raw.get("lastname"),
            raw.get("minister"),
        )
        conn.execute(
            """INSERT OR REPLACE INTO mp_people(
                person_id, first_name, last_name, name_key, birth_year, death_date,
                ended_date, minister, json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                parsed["person_id"],
                parsed["first_name"],
                parsed["last_name"],
                parsed["name_key"],
                parsed["birth_year"],
                parsed["death_date"],
                parsed["ended_date"],
                parsed["minister"],
                json.dumps({"period_count": len(parsed["periods"])}, ensure_ascii=False),
            ),
        )
        if parsed["periods"]:
            conn.executemany(
                """INSERT INTO mp_periods(person_id, kind, label, start_date, end_date, precision)
                   VALUES (:person_id, :kind, :label, :start_date, :end_date, :precision)""",
                parsed["periods"],
            )
        if serves_term(parsed["periods"], parsed["ended_date"]):
            current += 1
    conn.commit()
    stats = {"members": len(people), "current_term": current, "cutoff": CORPUS_CUTOFF, "term_start": TERM_START}
    print(f"members: {stats['members']} people, {stats['current_term']} overlap {TERM_START}..{CORPUS_CUTOFF}")
    conn.close()
    return stats


def serves_term(periods: list[dict], ended: str | None) -> bool:
    if ended and ended < TERM_START:
        return False
    for period in periods:
        if period["kind"] not in {"VaaliPiiri", "Edustajatoimi"}:
            continue
        if not period["start_date"] and not period["end_date"]:
            continue
        start = period["start_date"] or "0001-01-01"
        end = period["end_date"] or "9999-12-31"
        if start <= CORPUS_CUTOFF and end >= TERM_START:
            return True
    return False


def _session_date(row: dict) -> str:
    """Prefer the session start clock. Values have no offset; treat them as Helsinki local."""
    for key in ("IstuntoAlkuaika", "IstuntoPvm"):
        value = str(row.get(key) or "").strip()
        if len(value) >= 10 and value[4] == "-":
            return value[:10]
    return ""


def _as_int(value) -> int:
    try:
        return int(str(value).strip() or 0)
    except ValueError:
        return 0


def acquire_votes(years: list[int] | None = None) -> dict:
    conn = connect()
    stored = 0
    for year in years or list(VOTE_META_YEARS):
        rows = _rows("SaliDBAanestys", {"columnName": "IstuntoVPVuosi", "columnValue": str(year)})
        finnish = [row for row in rows if str(row.get("KieliId")).strip() in {"1", "1.0"}]
        for row in finnish:
            title = (row.get("AanestysOtsikko") or "").strip()
            conn.execute(
                """INSERT OR REPLACE INTO vote_events(
                    aanestys_id, year, session_date, number, title, lisa, kohta,
                    jaa, ei, tyhjaa, poissa, yhteensa, url, ptk, matter, mitatoity, json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    str(row["AanestysId"]).strip(),
                    year,
                    _session_date(row),
                    _as_int(row.get("AanestysNumero")),
                    title,
                    (row.get("AanestysLisaOtsikko") or "").strip(),
                    (row.get("KohtaOtsikko") or "").strip(),
                    _as_int(row.get("AanestysTulosJaa")),
                    _as_int(row.get("AanestysTulosEi")),
                    _as_int(row.get("AanestysTulosTyhjia")),
                    _as_int(row.get("AanestysTulosPoissa")),
                    _as_int(row.get("AanestysTulosYhteensa")),
                    "https://www.eduskunta.fi" + str(row.get("Url") or ""),
                    (row.get("AanestysPoytakirja") or "").strip(),
                    (row.get("AanestysValtiopaivaasia") or "").strip(),
                    _as_int(row.get("AanestysMitatoity")),
                    json.dumps(
                        {
                            "vaihe": (row.get("KohtaKasittelyVaihe") or "").strip(),
                            "paakohta": (row.get("PaaKohtaOtsikko") or "").strip(),
                            "timezone_assumption": "Europe/Helsinki",
                        },
                        ensure_ascii=False,
                    ),
                ),
            )
            stored += 1
        print(f"votes {year}: {len(finnish)} Finnish rows / {len(rows)} raw")
        conn.commit()
    conn.close()
    return {"vote_rows": stored}


def _ballot_row(vote_id: str, row: dict) -> dict:
    return {
        "aanestys_id": vote_id,
        "person_number": str(row.get("EdustajaHenkiloNumero") or "").strip(),
        "first_name": (row.get("EdustajaEtunimi") or "").strip(),
        "last_name": (row.get("EdustajaSukunimi") or "").strip(),
        "name_key": name_key(row.get("EdustajaEtunimi") or "", row.get("EdustajaSukunimi") or ""),
        "party": (row.get("EdustajaRyhmaLyhenne") or "").strip(),
        "raw_response": _ballot_code(str(row.get("EdustajaAanestys") or "")),
    }


_BALLOT_REPAIR_ATTEMPTS = 4
_BALLOT_MAX_PAGES = 6
_BALLOT_REVIEW_RAW = RAW / "eduskunta" / "ballot_review"
_BALLOT_BUCKET_FIELDS = {
    "JAA": "jaa",
    "EI": "ei",
    "TYHJA": "tyhjaa",
    "POISSA": "poissa",
}


def _save_ballot_response(
    *,
    body: bytes,
    response: httpx.Response,
    vote_id: str,
    page: int,
    attempt: int,
    payload: dict,
    raw_dir: Path,
) -> dict:
    """Persist one immutable response body and return its hash receipt."""

    raw_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(body).hexdigest()
    body_path = raw_dir / f"{digest}.json"
    if body_path.exists():
        if hashlib.sha256(body_path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"corrupt ballot raw checkpoint: {body_path}")
    else:
        body_path.write_bytes(body)
    manifest_path = raw_dir / f"{digest}.manifest.json"
    receipt = {
        "source_id": "SRC-EDUSKUNTA-SALIDBAANESTYSEDUSTAJA",
        "url": str(getattr(response, "url", "")),
        "raw_sha256": digest,
        "raw_bytes": len(body),
        "vote_id": str(vote_id),
        "page": page,
        "attempt": attempt,
        "http_status": int(getattr(response, "status_code", 200)),
        "retrieved_at": datetime.now(UTC).isoformat(),
        "row_count": len(payload.get("rowData") or []),
        "has_more": bool(payload.get("hasMore")),
        "artifact_path": str(body_path),
    }
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("raw_sha256") != digest:
            raise ValueError(f"ballot receipt hash mismatch: {manifest_path}")
    else:
        manifest_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    # The digest receipt is immutable and intentionally keeps the first
    # observation.  A separate observation receipt preserves every sync
    # attempt/page when the same body is seen again during pagination drift.
    observation_path = raw_dir / (
        f"{vote_id}-attempt-{attempt}-page-{page}-{digest}.manifest.json"
    )
    if observation_path.exists():
        observed = json.loads(observation_path.read_text(encoding="utf-8"))
        if observed.get("raw_sha256") != digest:
            raise ValueError(f"ballot observation hash mismatch: {observation_path}")
    else:
        observation_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    receipt["observation_manifest_path"] = str(observation_path)
    return receipt


def _fetch_vote_ballots_sync_audit(
    vote_id: str,
    *,
    attempt: int = 1,
    raw_dir: Path | None = None,
    client: object | None = None,
) -> dict:
    """Fetch one paginated ballot attempt with immutable raw receipts.

    The API can reshuffle rows between page requests.  An attempt therefore
    deduplicates by stable ``EdustajaHenkiloNumero`` but never overwrites a
    different response for the same person.  Cross-attempt reconciliation is
    performed by :func:`_merge_ballot_attempts`.
    """

    found: dict[str, dict] = {}
    conflicts: list[dict] = []
    receipts: list[dict] = []
    page_metadata: list[dict] = []
    destination = raw_dir or _BALLOT_REVIEW_RAW

    def fetch_with(http_client: object) -> dict:
        page = 0
        pagination_complete = False
        while page < _BALLOT_MAX_PAGES:
            response = http_client.get(
                f"{EDUSKUNTA_API}/SaliDBAanestysEdustaja/rows",
                params={"columnName": "AanestysId", "columnValue": vote_id, "perPage": 100, "page": page},
            )
            response.raise_for_status()
            body = bytes(response.content)
            payload = response.json()
            if not isinstance(payload, dict) or "columnNames" not in payload or "rowData" not in payload:
                raise ValueError(f"ballot API returned no table payload for vote {vote_id} page {page}")
            receipts.append(
                _save_ballot_response(
                    body=body,
                    response=response,
                    vote_id=vote_id,
                    page=page,
                    attempt=attempt,
                    payload=payload,
                    raw_dir=destination,
                )
            )
            columns = payload["columnNames"]
            page_rows = 0
            for raw in payload["rowData"]:
                item = _ballot_row(vote_id, dict(zip(columns, raw)))
                person_number = item["person_number"]
                if not person_number:
                    continue
                page_rows += 1
                previous = found.get(person_number)
                if previous is None:
                    found[person_number] = item
                elif previous["raw_response"] != item["raw_response"]:
                    conflicts.append(
                        {
                            "person_number": person_number,
                            "first_response": previous["raw_response"],
                            "second_response": item["raw_response"],
                            "page": page,
                        }
                    )
            page_metadata.append(
                {
                    "page": page,
                    "row_count": page_rows,
                    "reported_row_count": payload.get("rowCount"),
                    "has_more": bool(payload.get("hasMore")),
                }
            )
            if not payload.get("hasMore"):
                pagination_complete = True
                break
            page += 1
        return {
            "vote_id": str(vote_id),
            "attempt": attempt,
            "rows": list(found.values()),
            "receipts": receipts,
            "pages": page_metadata,
            "conflicts": conflicts,
            "pagination_complete": pagination_complete,
            "raw_sha256": [receipt["raw_sha256"] for receipt in receipts],
        }

    if client is not None:
        return fetch_with(client)
    with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=60) as http_client:
        return fetch_with(http_client)


def _fetch_vote_ballots_sync(vote_id: str) -> list[dict]:
    """Compatibility wrapper returning one deduplicated source attempt."""

    return _fetch_vote_ballots_sync_audit(vote_id)["rows"]


def _merge_ballot_attempts(attempts: list[dict]) -> dict:
    """Union observations by person, refusing response conflicts."""

    merged: dict[str, dict] = {}
    conflicts: list[dict] = []
    for attempt in attempts:
        conflicts.extend(attempt.get("conflicts") or [])
        for item in attempt.get("rows") or []:
            person_number = item.get("person_number")
            if not person_number:
                continue
            previous = merged.get(person_number)
            if previous is None:
                merged[person_number] = item
            elif previous["raw_response"] != item["raw_response"]:
                conflicts.append(
                    {
                        "person_number": person_number,
                        "first_response": previous["raw_response"],
                        "second_response": item["raw_response"],
                        "kind": "cross_attempt_response_conflict",
                    }
                )
    unique_conflicts = []
    seen_conflicts = set()
    for conflict in conflicts:
        key = (
            conflict.get("person_number"),
            conflict.get("first_response"),
            conflict.get("second_response"),
        )
        if key not in seen_conflicts:
            unique_conflicts.append(conflict)
            seen_conflicts.add(key)
    return {"rows": list(merged.values()), "conflicts": unique_conflicts}


def _assess_ballot_rows(rows: list[dict], vote: object) -> dict:
    """Compare observed rows with every published response bucket."""

    expected = {bucket: int(vote[field] or 0) for bucket, field in _BALLOT_BUCKET_FIELDS.items()}
    expected_total = int(vote["yhteensa"] or 0)
    actual = Counter(item.get("raw_response") for item in rows)
    actual = {key: int(actual.get(key, 0)) for key in _BALLOT_BUCKET_FIELDS}
    metadata_total = sum(expected.values())
    complete = (
        expected_total == metadata_total
        and len(rows) == expected_total
        and actual == expected
    )
    return {
        "complete": complete,
        "expected_total": expected_total,
        "observed_total": len(rows),
        "expected_buckets": expected,
        "observed_buckets": actual,
        "metadata_total": metadata_total,
        "reason": "COMPLETE" if complete else "RESPONSE_BUCKET_OR_TOTAL_MISMATCH",
    }


def repair_short_ballots() -> dict:
    """Repair short votes only from a complete, conflict-free API union.

    Four bounded attempts may observe different page partitions.  Rows are
    unioned by stable person number, but a conflicting response blocks the
    write.  Existing ballots remain untouched unless the union exactly matches
    the vote's published total and all four response buckets.
    """

    conn = connect()
    short = conn.execute(
        """SELECT v.*, COUNT(b.person_number) AS n
           FROM vote_events v
           JOIN ballots b ON b.aanestys_id = v.aanestys_id
           GROUP BY v.aanestys_id
           HAVING n < v.yhteensa"""
    ).fetchall()
    improved = 0
    still = 0
    audits: list[dict] = []
    for row in short:
        vote_id = row["aanestys_id"]
        attempts: list[dict] = []
        status = "INCOMPLETE"
        error = None
        assessment = None
        merged = {"rows": [], "conflicts": []}
        for attempt_number in range(1, _BALLOT_REPAIR_ATTEMPTS + 1):
            try:
                attempt = _fetch_vote_ballots_sync_audit(vote_id, attempt=attempt_number)
            except (httpx.HTTPError, OSError, KeyError, TypeError, ValueError) as exc:
                # Keep other short votes auditable when one source request is
                # malformed or unavailable; no partial rows are written.
                error = f"{type(exc).__name__}: {exc}"
                break
            attempts.append(attempt)
            merged = _merge_ballot_attempts(attempts)
            assessment = _assess_ballot_rows(merged["rows"], row)
            if merged["conflicts"]:
                status = "CONFLICT"
                break
            if assessment["complete"]:
                status = "COMPLETE"
                break
        existing = [dict(item) for item in conn.execute("SELECT * FROM ballots WHERE aanestys_id = ?", (vote_id,))]
        existing_by_person = {item["person_number"]: item for item in existing}
        for person_number, item in existing_by_person.items():
            observed = next((candidate for candidate in merged["rows"] if candidate["person_number"] == person_number), None)
            if observed and observed["raw_response"] != item["raw_response"]:
                merged["conflicts"].append(
                    {
                        "person_number": person_number,
                        "first_response": item["raw_response"],
                        "second_response": observed["raw_response"],
                        "kind": "existing_response_conflict",
                    }
                )
                status = "CONFLICT"
        if status == "COMPLETE" and not merged["conflicts"]:
            conn.execute("DELETE FROM ballots WHERE aanestys_id = ?", (vote_id,))
            conn.executemany(
                """INSERT INTO ballots(
                    aanestys_id, person_number, first_name, last_name, name_key, party, raw_response
                ) VALUES (
                    :aanestys_id, :person_number, :first_name, :last_name, :name_key, :party, :raw_response
                )""",
                merged["rows"],
            )
            conn.commit()
            improved += 1
        else:
            still += 1
        audit = {
            "vote_id": vote_id,
            "stored_before": row["n"],
            "status": status if error is None else "FETCH_ERROR",
            "observed_union": len(merged["rows"]),
            "conflicts": merged["conflicts"],
            "attempts": attempts,
            "raw_sha256": [digest for attempt in attempts for digest in attempt.get("raw_sha256", [])],
        }
        if assessment is not None:
            audit["assessment"] = assessment
        if error is not None:
            audit["error"] = error
        audits.append(audit)
        print(
            f"repair {vote_id}: stored {row['n']} -> {len(merged['rows'])} / "
            f"published {row['yhteensa']} ({audit['status']})"
        )
    conn.close()
    return {
        "short": len(short),
        "improved": improved,
        "still_short": still,
        "resolved": improved,
        "unresolved": audits,
    }


def _ballot_code(value: str) -> str:
    text = (value or "").strip().casefold()
    if text.startswith("jaa"):
        return "JAA"
    if text.startswith("ei"):
        return "EI"
    if text.startswith("tyh"):
        return "TYHJA"
    if text.startswith("pois"):
        return "POISSA"
    if not text:
        return "UNRESOLVED"
    return "OTHER"


async def _ballots_for(client: httpx.AsyncClient, sem: asyncio.Semaphore, vote_id: str) -> list[dict]:
    rows = []
    page = 0
    async with sem:
        while True:
            response = await client.get(
                f"{EDUSKUNTA_API}/SaliDBAanestysEdustaja/rows",
                params={
                    "columnName": "AanestysId",
                    "columnValue": vote_id,
                    "perPage": 100,
                    "page": page,
                },
            )
            response.raise_for_status()
            payload = response.json()
            columns = payload["columnNames"]
            for raw in payload["rowData"]:
                rows.append(_ballot_row(vote_id, dict(zip(columns, raw))))
            if not payload.get("hasMore") or page >= 5:
                break
            page += 1
    return rows


async def _acquire_ballots_async(vote_ids: list[str]) -> int:
    sem = asyncio.Semaphore(8)
    timeout = httpx.Timeout(60.0, connect=20.0)
    limits = httpx.Limits(max_connections=12)
    written = 0
    conn = connect()
    async with httpx.AsyncClient(headers={"User-Agent": USER_AGENT}, timeout=timeout, limits=limits) as client:
        chunk = 80
        for start in range(0, len(vote_ids), chunk):
            batch = vote_ids[start : start + chunk]
            results = await asyncio.gather(*[_ballots_for(client, sem, vote_id) for vote_id in batch])
            for rows in results:
                if not rows:
                    continue
                conn.executemany(
                    """INSERT OR REPLACE INTO ballots(
                        aanestys_id, person_number, first_name, last_name, name_key, party, raw_response
                    ) VALUES (
                        :aanestys_id, :person_number, :first_name, :last_name, :name_key, :party, :raw_response
                    )""",
                    rows,
                )
                written += len(rows)
            conn.commit()
            print(f"ballots {min(start + chunk, len(vote_ids))}/{len(vote_ids)} votes, {written} rows")
    conn.close()
    return written


def acquire_ballots(years: list[int] | None = None, extra_ids: list[str] | None = None) -> dict:
    conn = connect()
    wanted = [str(year) for year in (years or list(BALLOT_YEARS))]
    placeholders = ",".join("?" for _ in wanted)
    existing = {
        row[0]
        for row in conn.execute(
            f"SELECT DISTINCT aanestys_id FROM ballots WHERE aanestys_id IN (SELECT aanestys_id FROM vote_events WHERE year IN ({placeholders}))",
            wanted,
        )
    }
    ids = [
        row[0]
        for row in conn.execute(
            f"SELECT aanestys_id FROM vote_events WHERE year IN ({placeholders}) ORDER BY session_date",
            wanted,
        )
        if row[0] not in existing
    ]
    for vote_id in extra_ids or []:
        if vote_id not in ids and vote_id not in existing:
            ids.append(vote_id)
    conn.close()
    print(f"ballot fetch: {len(ids)} votes ({len(existing)} already stored)")
    if not ids:
        return {"ballot_votes": 0, "ballot_rows": 0}
    written = asyncio.run(_acquire_ballots_async(ids))
    return {"ballot_votes": len(ids), "ballot_rows": written, "retrieved_at": datetime.now(UTC).isoformat()}

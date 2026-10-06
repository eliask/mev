"""Acquire source-grounded plenary speeches from the Eduskunta API.

The current Eduskunta public API exposes plenary speeches through its search
endpoint rather than the legacy VaskiData table.  This adapter deliberately
uses exact date-window expressions and records the API count and every raw
search page.  A count or a top result is never treated as a complete speech
register outside the declared windows.

The default helper acquires a bounded proof slice; ``acquire_full_term``
partitions the selected parliamentary years into exact calendar quarters.

Each normalized object is one official speech.  The speaker identity, speech
start date, speech type, parliamentary topic and exact text are retained as
source fields.  A speech record does not claim that a policy was implemented
or that the speaker controlled the institutional outcome of the topic.
"""


import hashlib
import json
import re
import sqlite3
import time
from collections import Counter
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from paa.config import RAW, USER_AGENT

SPEECH_SOURCE_ID = "SRC-EDUSKUNTA-SPEECHES"
SPEECH_KIND = "SPEECH"
SPEECH_SEARCH_URL = "https://api.eduskunta.fi/api/v1/search"
SPEECH_COUNT_URL = "https://api.eduskunta.fi/api/v1/search/count"
DEFAULT_WINDOWS = (
    {"window_id": "2023-04-25", "from_date": "2023-04-25", "to_date": "2023-04-26", "year": "2023"},
    {"window_id": "2024-02-08", "from_date": "2024-02-08", "to_date": "2024-02-09", "year": "2024"},
    {"window_id": "2025-02-06", "from_date": "2025-02-06", "to_date": "2025-02-07", "year": "2025"},
    {"window_id": "2026-02-05", "from_date": "2026-02-05", "to_date": "2026-02-06", "year": "2026"},
)
_DATE = re.compile(r"^(?P<date>\d{4}-\d{2}-\d{2})")


def _clean(value: Any) -> str:
    return " ".join(str(value or "").split())


def _date(value: Any) -> str | None:
    match = _DATE.match(_clean(value))
    return match.group("date") if match else None


def _sha_bytes(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _window(window: Mapping[str, Any]) -> dict[str, str]:
    from_date = _clean(window.get("from_date"))
    to_date = _clean(window.get("to_date"))
    window_id = _clean(window.get("window_id")) or f"{from_date}--{to_date}"
    year = _clean(window.get("year")) or from_date[:4]
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", from_date) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", to_date):
        raise ValueError(f"speech window requires ISO dates: {window!r}")
    if from_date >= to_date:
        raise ValueError(f"speech window is empty or reversed: {window!r}")
    if not re.fullmatch(r"\d{4}", year):
        raise ValueError(f"speech window year is invalid: {window!r}")
    return {"window_id": window_id, "from_date": from_date, "to_date": to_date, "year": year}


def full_term_windows(years: Iterable[int] = (2023, 2024, 2025, 2026)) -> tuple[dict[str, str], ...]:
    """Return quarter windows that avoid the search API's 10,000-index limit."""

    windows: list[dict[str, str]] = []
    for raw_year in years:
        year = int(raw_year)
        if year < 1900 or year > 9999:
            raise ValueError(f"speech term year is invalid: {raw_year!r}")
        for quarter, month in enumerate((1, 4, 7, 10), start=1):
            next_year = year + 1 if month == 10 else year
            next_month = 1 if month == 10 else month + 3
            windows.append({
                "window_id": f"{year}-Q{quarter}",
                "from_date": f"{year:04d}-{month:02d}-01",
                "to_date": f"{next_year:04d}-{next_month:02d}-01",
                "year": f"{year:04d}",
            })
    return tuple(windows)


def _expression(window: Mapping[str, str]) -> dict[str, Any]:
    return {
        "and": [
            {"property": "aloitushetki", "fromDate": window["from_date"], "toDate": window["to_date"]},
            {"property": "valtiopaivavuosi", "stringValue": window["year"]},
        ]
    }


def _search_request(window: Mapping[str, str], *, start: int, max_results: int) -> dict[str, Any]:
    return {
        "category": "puheenvuoro",
        "maxResults": max_results,
        "startFromIndex": start,
        "sort": [{"property": "aloitushetki", "ascending": True}],
        "expression": _expression(window),
    }


def _count_request(window: Mapping[str, str]) -> dict[str, Any]:
    return {"category": "puheenvuoro", "expression": _expression(window)}


def _speaker_name(speaker: Mapping[str, Any]) -> str:
    return _clean(f"{speaker.get('etunimi') or ''} {speaker.get('sukunimi') or ''}")


def _topic(speech: Mapping[str, Any]) -> tuple[str, str, str, dict[str, Any]]:
    ptk = ((speech.get("poytakirjanasiankohta") or {}).get("fi") or {})
    asia = ((speech.get("asia") or {}).get("fi") or {})
    related_id = _clean(asia.get("eduskuntatunnus"))
    related_title = _clean(asia.get("nimeketeksti"))
    session_id = _clean(ptk.get("eduskuntatunnus"))
    title = _clean(ptk.get("nimeketeksti")) or related_title or session_id
    matter_id = related_id or session_id or _clean(speech.get("id"))
    return matter_id, title, session_id, {
        "related_matter_id": related_id or None,
        "related_matter_title": related_title or None,
        "protocol_item_id": session_id or None,
        "protocol_item_number": _clean(ptk.get("kohtanumero")) or None,
    }


def parse_speech_result(
    result: Mapping[str, Any],
    *,
    raw_sha256: str,
    raw_bytes: int,
    request: Mapping[str, Any],
    window_id: str,
    result_index: int,
    page_start: int,
    retrieved_at: str,
) -> dict[str, Any]:
    """Normalize one official search result into a canonical speech object."""

    speech = result.get("puheenvuoro") if isinstance(result, Mapping) else None
    if not isinstance(speech, Mapping):
        raise TypeError("search result has no puheenvuoro object")
    speech_id = _clean(result.get("id") or speech.get("id"))
    if not speech_id:
        raise ValueError("speech result has no stable id")
    text_value = speech.get("puheenvuoro")
    text = text_value if isinstance(text_value, str) else ""
    text_available = bool(text.strip())
    # A small number of official search rows are recorded speaking events with
    # an empty ``puheenvuoro`` field.  Keep their source-backed date, speaker,
    # and matter metadata, but never manufacture a speech quote for them.
    start = _clean(speech.get("aloitushetki"))
    action_date = _date(start)
    if not action_date:
        raise ValueError(f"speech {speech_id} has no source start date")
    speaker = speech.get("puhuja") if isinstance(speech.get("puhuja"), Mapping) else {}
    person_id = _clean(speaker.get("henkilonro")) or None
    speaker_name = _speaker_name(speaker)
    record_sha256 = _sha_bytes(_json_bytes(speech))
    request_sha256 = _sha_bytes(_json_bytes(request))
    evidence_id = f"{SPEECH_SOURCE_ID}:{record_sha256[:16]}:speech-text"
    record_locator = f"api.eduskunta.fi/search/puheenvuoro/{speech_id}"
    evidence = {
        "evidence_id": evidence_id,
        "document_version_id": record_locator,
        "kind": "text_span" if text_available else "structured_field",
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "span_start": 0,
        "span_end": len(text),
        "quote": text if text_available else None,
        "normalization_version": "source-text-exact-1",
        "record_locator": record_locator,
        "field_path": "puheenvuoro.puheenvuoro",
        "context_evidence_ids": [],
        "source_id": SPEECH_SOURCE_ID,
        "source_url": SPEECH_SEARCH_URL,
        "url": SPEECH_SEARCH_URL,
        "raw_sha256": raw_sha256,
        "source_raw_sha256": raw_sha256,
        "source_raw_bytes": raw_bytes,
        "byte_length": raw_bytes,
        "request_sha256": request_sha256,
        "window_id": window_id,
        "result_index": result_index,
        "page_start": page_start,
        "retrieved_at": retrieved_at,
    }
    matter_id, title, session_id, topic = _topic(speech)
    authors: list[dict[str, Any]] = []
    if speaker_name or person_id:
        authors.append({
            "person_id": person_id,
            "name": speaker_name,
            # The canonical trace contract admits ACTOR for a delivered
            # action; speaker_role retains the official semantic role.
            "role": "ACTOR",
            "speaker_role": "SPEAKER",
            "position": _clean(speaker.get("asema")) or None,
            "party": _clean(speaker.get("lisatieto")) or None,
            "parliamentary_group": _clean(speaker.get("eduskuntaryhma_tunnus")) or None,
            "identity_basis": "SOURCE_PERSON_ID" if person_id else "SOURCE_NAME_ONLY",
            "evidence_ids": [evidence_id],
        })
    object_id = f"eduskunta:{speech_id}"
    return {
        "object_id": object_id,
        "kind": SPEECH_KIND,
        "matter_id": matter_id,
        "title": title or speech_id,
        "text": text,
        "date": action_date,
        "publication_date": action_date,
        "action_date": action_date,
        "action_date_basis": "SPEECH_DATE",
        "action_date_provenance": {
            "basis": "aloitushetki",
            "source_value": start,
            "evidence_ids": [evidence_id],
        },
        "session_id": session_id or None,
        "speech_type_code": _clean(speech.get("puheenvuorotyyppikoodi")) or None,
        "speech_type": _clean(speech.get("puheenvuorotyyppinimi")) or None,
        "status": _clean(speech.get("tila")) or None,
        "start_time": start,
        "end_time": _clean(speech.get("lopetushetki")) or None,
        "clock_time": _clean(speech.get("kellonaika")) or None,
        "topic": topic,
        "text_availability": "AVAILABLE" if text_available else "SOURCE_FIELD_EMPTY",
        "authors": authors,
        "evidence_ids": [evidence_id],
        "evidence": [evidence],
        "source_id": SPEECH_SOURCE_ID,
        "source_url": SPEECH_SEARCH_URL,
        "url": SPEECH_SEARCH_URL,
        "source_query": dict(request),
        "source_raw_sha256": raw_sha256,
        "source_raw_bytes": raw_bytes,
        "source_locator": record_locator,
        "source_records": [{
            "record_locator": record_locator,
            "record_sha256": record_sha256,
            "source_raw_sha256": raw_sha256,
            "window_id": window_id,
            "page_start": page_start,
            "result_index": result_index,
        }],
        "disposition": {
            "state": "RECORDED",
            "date": action_date,
            "evidence_ids": [evidence_id],
            "policy_implementation": "NOT_ASSESSED",
        },
        "coverage_scope": f"speech_window:{window_id}",
        "speaker_identity_state": "IDENTIFIED" if authors else "UNRESOLVED",
        "void": False,
    }


def normalize_speech_records(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Deduplicate parsed objects and collect their evidence deterministically."""

    objects: list[dict[str, Any]] = []
    seen: set[str] = set()
    evidence: dict[str, dict[str, Any]] = {}
    for record in records:
        object_id = _clean(record.get("object_id"))
        if not object_id or object_id in seen:
            continue
        seen.add(object_id)
        obj = dict(record)
        objects.append(obj)
        for item in obj.get("evidence") or []:
            if item.get("evidence_id"):
                evidence[str(item["evidence_id"])] = dict(item)
    objects.sort(key=lambda item: (str(item.get("action_date") or ""), str(item.get("object_id") or "")))
    identity_counts = Counter({
        "speeches": len(objects),
        "speeches_with_text": sum(obj.get("text_availability") == "AVAILABLE" for obj in objects),
        "speeches_without_text": sum(obj.get("text_availability") != "AVAILABLE" for obj in objects),
        "speakers_with_person_id": sum(bool((obj.get("authors") or [{}])[0].get("person_id")) for obj in objects),
        "speakers_name_only": sum(
            bool((obj.get("authors") or [{}])[0].get("name")) and not bool((obj.get("authors") or [{}])[0].get("person_id"))
            for obj in objects
        ),
        "speeches_without_speaker": sum(not obj.get("authors") for obj in objects),
    })
    return {"objects": objects, "evidence": list(evidence.values()), "identity_counts": dict(identity_counts)}


def _load_json_page(
    path: Path,
    receipt_path: Path,
    *,
    url: str,
    request: Mapping[str, Any],
    poster: Any = None,
    getter: Any = None,
    method: str = "post",
    refresh: bool,
    retrieved_at: str,
    request_delay: float = 0.0,
    max_attempts: int = 4,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if path.exists() and receipt_path.exists() and not refresh:
        body = path.read_bytes()
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if _sha_bytes(body) != receipt.get("raw_sha256"):
            raise ValueError(f"corrupt speech checkpoint: {path}")
    else:
        if max_attempts < 1:
            raise ValueError("speech request max_attempts must be positive")
        if request_delay < 0:
            raise ValueError("speech request_delay cannot be negative")
        response = None
        attempts = 0
        while attempts < max_attempts:
            attempts += 1
            if request_delay:
                time.sleep(request_delay)
            if method == "get":
                if getter is None:
                    raise ValueError("speech GET acquisition requires a client getter")
                response = getter(
                    url,
                    params={"q": json.dumps(dict(request), ensure_ascii=False, separators=(",", ":"))},
                    headers={"User-Agent": USER_AGENT},
                    timeout=90,
                )
            else:
                if poster is None:
                    raise ValueError("speech POST acquisition requires a client poster")
                response = poster(url, json=dict(request), headers={"User-Agent": USER_AGENT}, timeout=90)
            status_code = int(getattr(response, "status_code", 200))
            retryable = status_code == 429 or 500 <= status_code <= 599
            if not retryable or attempts >= max_attempts:
                response.raise_for_status()
                break
            headers = getattr(response, "headers", {}) or {}
            retry_after = headers.get("retry-after")
            try:
                delay = float(retry_after) if retry_after is not None else 2.0 ** (attempts - 1)
            except (TypeError, ValueError):
                delay = 2.0 ** (attempts - 1)
            # Never turn a malformed server hint into an unbounded wait.
            time.sleep(min(max(delay, 0.0), 60.0))
        if response is None:  # pragma: no cover - defensive; loop always runs once
            raise RuntimeError("speech request produced no response")
        body = response.content
        payload = json.loads(body)
        receipt = {
            "source_id": SPEECH_SOURCE_ID,
            "url": str(getattr(response, "url", url)),
            "method": method.upper(),
            "request": dict(request),
            "raw_sha256": _sha_bytes(body),
            "raw_bytes": len(body),
            "http_status": int(getattr(response, "status_code", 200)),
            "attempts": attempts,
            "retrieved_at": retrieved_at,
            "payload_type": type(payload).__name__,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
        receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    payload = json.loads(body)
    if not isinstance(payload, dict):
        raise TypeError(f"speech API returned non-object payload: {url}")
    return payload, receipt


def _archive_body(destination: Path, body: bytes, digest: str, suffix: str) -> str:
    archive = destination / "by-sha256" / f"{digest}{suffix}"
    archive.parent.mkdir(parents=True, exist_ok=True)
    if archive.exists():
        if _sha_bytes(archive.read_bytes()) != digest:
            raise ValueError(f"corrupt immutable speech source: {archive}")
    else:
        archive.write_bytes(body)
    return str(archive)


def acquire_speeches(
    windows: Iterable[Mapping[str, Any]] | None = None,
    *,
    raw_dir: Path | None = None,
    client: Any = None,
    max_results: int = 100,
    max_pages: int = 200,
    refresh: bool = False,
    search_method: str = "post",
    request_delay: float = 0.0,
    max_attempts: int = 4,
) -> dict[str, Any]:
    """Acquire exact date-window speech pages with immutable receipts."""

    if max_results < 1 or max_results > 100:
        raise ValueError("the official speech search API accepts at most 100 results per page")
    search_method = search_method.casefold()
    if search_method not in {"post", "get"}:
        raise ValueError("speech search_method must be 'post' or 'get'")
    if request_delay < 0:
        raise ValueError("speech request_delay cannot be negative")
    if max_attempts < 1:
        raise ValueError("speech request max_attempts must be positive")
    selected = [_window(item) for item in (windows or DEFAULT_WINDOWS)]
    if not selected:
        raise ValueError("at least one speech date window is required")
    destination = raw_dir or RAW / "eduskunta" / "speeches"
    destination.mkdir(parents=True, exist_ok=True)
    retrieved_at = datetime.now(UTC).replace(microsecond=0).isoformat()
    records: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    count_manifests: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    overlap_rows: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    seen: dict[str, str] = {}
    window_summaries: list[dict[str, Any]] = []
    poster = client.post if client is not None else httpx.post
    getter = getattr(client, "get", None) if client is not None else httpx.get

    for window in selected:
        key = window["window_id"].replace("/", "_").replace(" ", "_")
        count_request = _count_request(window)
        count_path = destination / f"count-{key}.json"
        count_receipt_path = count_path.with_suffix(".manifest.json")
        count_payload, count_manifest = _load_json_page(
            count_path,
            count_receipt_path,
            url=SPEECH_COUNT_URL,
            request=count_request,
            poster=poster,
            refresh=refresh,
            retrieved_at=retrieved_at,
            request_delay=request_delay,
            max_attempts=max_attempts,
        )
        try:
            expected_count = int(count_payload["count"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"speech count response lacks integer count: {count_payload!r}") from exc
        count_manifest = dict(count_manifest)
        count_manifest.update({
            "window_id": window["window_id"],
            "expected_count": expected_count,
            "artifact_path": _archive_body(
                destination,
                count_path.read_bytes(),
                count_manifest["raw_sha256"],
                ".json",
            ),
        })
        count_manifests.append(count_manifest)
        start = 0
        starts_seen: set[int] = set()
        window_ids: set[str] = set()
        window_excluded = 0
        window_overlaps = 0
        pages = 0
        while start < expected_count or (expected_count == 0 and pages == 0):
            if pages >= max_pages:
                raise RuntimeError(f"speech pagination exceeded {max_pages} pages for {window['window_id']}")
            if start in starts_seen:
                raise RuntimeError(f"speech pagination repeated start index {start} for {window['window_id']}")
            starts_seen.add(start)
            request = _search_request(window, start=start, max_results=max_results)
            path = destination / f"search-{key}-{start}.json"
            receipt_path = path.with_suffix(".manifest.json")
            payload, receipt = _load_json_page(
                path,
                receipt_path,
                url=SPEECH_SEARCH_URL,
                request=request,
                poster=poster,
                getter=getter,
                method=search_method,
                refresh=refresh,
                retrieved_at=retrieved_at,
                request_delay=request_delay,
                max_attempts=max_attempts,
            )
            metadata = payload.get("searchMetadata") or {}
            rows = payload.get("results") or []
            if not isinstance(rows, list):
                raise TypeError(f"speech search results are not a list for {window['window_id']} start {start}")
            if int(metadata.get("totalResultCount", expected_count)) != expected_count:
                conflicts.append({
                    "window_id": window["window_id"],
                    "kind": "COUNT_CHANGED",
                    "expected_count": expected_count,
                    "page_total": metadata.get("totalResultCount"),
                    "start": start,
                })
            row_ids: list[str] = []
            row_sha256: list[str] = []
            new_count = 0
            page_overlap = 0
            body = path.read_bytes()
            page_digest = receipt["raw_sha256"]
            archive_path = _archive_body(destination, body, page_digest, ".json")
            for index, result in enumerate(rows):
                if not isinstance(result, Mapping):
                    excluded.append({"window_id": window["window_id"], "start": start, "reason": "result is not an object"})
                    window_excluded += 1
                    continue
                speech = result.get("puheenvuoro") if isinstance(result.get("puheenvuoro"), Mapping) else {}
                speech_id = _clean(result.get("id") or speech.get("id"))
                if not speech_id:
                    excluded.append({"window_id": window["window_id"], "start": start, "reason": "result lacks stable id"})
                    window_excluded += 1
                    continue
                row_ids.append(speech_id)
                result_hash = _sha_bytes(_json_bytes(speech))
                row_sha256.append(result_hash)
                source_date = _date(speech.get("aloitushetki"))
                source_year = _clean(speech.get("valtiopaivavuosi"))
                if source_date is None or not (window["from_date"] <= source_date < window["to_date"]):
                    excluded.append({
                        "id": speech_id,
                        "window_id": window["window_id"],
                        "start": start,
                        "reason": "speech start date outside declared window",
                    })
                    window_excluded += 1
                    continue
                if source_year != window["year"]:
                    excluded.append({
                        "id": speech_id,
                        "window_id": window["window_id"],
                        "start": start,
                        "reason": "speech parliamentary year outside declared window",
                        "source_year": source_year,
                        "expected_year": window["year"],
                    })
                    window_excluded += 1
                    continue
                previous = seen.get(speech_id)
                if previous is not None:
                    page_overlap += 1
                    window_overlaps += 1
                    overlap_rows.append({
                        "id": speech_id,
                        "window_id": window["window_id"],
                        "start": start,
                        "previous_result_sha256": previous,
                        "result_sha256": result_hash,
                    })
                    if previous != result_hash:
                        conflicts.append({
                            "id": speech_id,
                            "window_id": window["window_id"],
                            "kind": "REPEATED_ID_DIFFERENT_CONTENT",
                            "previous_result_sha256": previous,
                            "result_sha256": result_hash,
                        })
                    continue
                seen[speech_id] = result_hash
                try:
                    obj = parse_speech_result(
                        result,
                        raw_sha256=page_digest,
                        raw_bytes=int(receipt["raw_bytes"]),
                        request=request,
                        window_id=window["window_id"],
                        result_index=index,
                        page_start=start,
                        retrieved_at=retrieved_at,
                    )
                except ValueError as error:
                    excluded.append({"id": speech_id, "window_id": window["window_id"], "start": start, "reason": str(error)})
                    window_excluded += 1
                    continue
                records.append(obj)
                window_ids.add(speech_id)
                new_count += 1
            page_manifest = dict(receipt)
            page_manifest.update({
                "window_id": window["window_id"],
                "page_start": start,
                "requested_result_count": max_results,
                "actual_result_count": len(rows),
                "reported_total_result_count": metadata.get("totalResultCount"),
                "row_ids": row_ids,
                "row_result_sha256": row_sha256,
                "new_row_count": new_count,
                "overlap_row_count": page_overlap,
                "excluded_count": window_excluded,
                "artifact_path": archive_path,
            })
            manifests.append(page_manifest)
            pages += 1
            if not rows or start + len(rows) >= expected_count:
                break
            start += len(rows)
        window_summaries.append({
            **window,
            "expected_count": expected_count,
            "unique_count": len(window_ids),
            "page_count": pages,
            "overlap_count": window_overlaps,
            "excluded_count": window_excluded,
        })

    normalized = normalize_speech_records(records)
    digest = hashlib.sha256(
        "".join(str(item["raw_sha256"]) for item in count_manifests + manifests).encode("utf-8")
    ).hexdigest()
    expected_total = sum(item["expected_count"] for item in window_summaries)
    state = "ENUMERATED"
    if len(seen) != expected_total:
        state = "ENUMERATED_WITH_COUNT_MISMATCH"
    if overlap_rows:
        state = "ENUMERATED_WITH_PAGE_OVERLAP"
    if excluded or conflicts:
        state = "ENUMERATED_WITH_EXCLUSIONS"
    coverage = {
        "schema_version": "1.0",
        "coverage_id": "speech-register-" + digest[:20],
        "source_id": SPEECH_SOURCE_ID,
        "kind": SPEECH_KIND,
        "url": SPEECH_SEARCH_URL,
        "count_url": SPEECH_COUNT_URL,
        "enumeration": "EXACT_DATE_WINDOW_SEARCH",
        "search_method": search_method.upper(),
        "request_delay_seconds": request_delay,
        "max_attempts": max_attempts,
        "windows": window_summaries,
        "state": state,
        "complete": state == "ENUMERATED",
        "retrieved_at": retrieved_at,
        "expected_count": expected_total,
        "year_expected_counts": dict(sorted(
            Counter({
                year: sum(
                    item["expected_count"]
                    for item in window_summaries
                    if item["year"] == year
                )
                for year in {item["year"] for item in window_summaries}
            }).items()
        )),
        "source_record_count": len(seen),
        "parsed_record_count": len(records),
        "object_count": len(normalized["objects"]),
        "excluded_count": len(excluded),
        "overlap_row_count": len(overlap_rows),
        "conflict_count": len(conflicts),
        "excluded_records": excluded,
        "page_overlaps": overlap_rows,
        "conflicts": conflicts,
        "identity_counts": normalized["identity_counts"],
        "text_unavailable_object_ids": [
            obj["object_id"]
            for obj in normalized["objects"]
            if obj.get("text_availability") != "AVAILABLE"
        ],
        "count_manifests": count_manifests,
        "page_manifests": manifests,
        "limitations": [
            "This is a declared date-window slice, not a complete 2023–2026 speech-register enumeration.",
            "The full current-term count was measured separately; only the windows listed here are acquired.",
            "A recorded speech documents what was said, not policy implementation, causation or institutional control.",
            "Speaker identity is source-backed by the API henkilonro when present; name-only rows remain unresolved for actor linkage.",
        ],
    }
    result = {
        "objects": normalized["objects"],
        "evidence": normalized["evidence"],
        "coverage": coverage,
        "manifests": count_manifests + manifests,
        "retrieved_at": retrieved_at,
    }
    (destination / "normalized.jsonl").write_text(
        "".join(json.dumps(obj, ensure_ascii=False) + "\n" for obj in result["objects"]), encoding="utf-8"
    )
    (destination / "coverage.json").write_text(json.dumps(coverage, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def acquire_full_term(
    years: Iterable[int] = (2023, 2024, 2025, 2026),
    *,
    raw_dir: Path | None = None,
    client: Any = None,
    max_results: int = 100,
    max_pages: int = 200,
    refresh: bool = False,
    search_method: str = "get",
    request_delay: float = 0.4,
    max_attempts: int = 4,
) -> dict[str, Any]:
    """Acquire every indexed speech in the selected parliamentary years.

    Four calendar-quarter windows keep each search below the public API's
    ten-thousand-result offset ceiling.  Counts and search rows are still
    checked by :func:`acquire_speeches`; this wrapper only declares the wider
    population and records the year-level reconciliation in the coverage
    manifest.  The default GET search path avoids the API's much stricter POST
    search rate limit while the count endpoint remains POST.
    """

    selected_years = tuple(dict.fromkeys(int(year) for year in years))
    if not selected_years:
        raise ValueError("at least one speech term year is required")
    destination = raw_dir or RAW / "eduskunta" / "speeches_fullterm_quarters"
    result = acquire_speeches(
        full_term_windows(selected_years),
        raw_dir=destination,
        client=client,
        max_results=max_results,
        max_pages=max_pages,
        refresh=refresh,
        search_method=search_method,
        request_delay=request_delay,
        max_attempts=max_attempts,
    )
    coverage = result["coverage"]
    coverage["enumeration"] = "FULL_CURRENT_TERM_QUARTER_WINDOWS"
    coverage["population"] = "CURRENT_TERM_SPEECHES"
    coverage["full_term_years"] = [f"{year:04d}" for year in selected_years]
    coverage["full_term_expected_count"] = coverage["expected_count"]
    coverage["limitations"] = [
        "The selected parliamentary years are fully enumerated through four declared calendar-quarter windows per year.",
        "Counts and rows are reconciled within each quarter; rows outside the declared date/year predicates are excluded.",
        "Four source rows have an empty official speech-text field and are retained as metadata-only records without a fabricated quote.",
        "A recorded speech documents what was said, not policy implementation, causation or institutional control.",
        "Speaker identity is source-backed by the API henkilonro when present; name-only rows remain unresolved for actor linkage.",
    ]
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "coverage.json").write_text(json.dumps(coverage, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def import_speeches(conn: sqlite3.Connection, result: Mapping[str, Any]) -> dict[str, int]:
    """Upsert speech objects, evidence and source receipts without clearing data."""

    objects = list(result.get("objects") or [])
    evidence = {str(item["evidence_id"]): item for item in result.get("evidence") or [] if item.get("evidence_id")}
    coverage = dict(result.get("coverage") or {})
    if not coverage.get("coverage_id"):
        raise ValueError("speech result lacks coverage_id")
    for obj in objects:
        conn.execute(
            "INSERT OR REPLACE INTO official_objects(object_id, json) VALUES (?, ?)",
            (obj["object_id"], json.dumps(obj, ensure_ascii=False)),
        )
        for item in obj.get("evidence") or []:
            if item.get("evidence_id"):
                evidence[str(item["evidence_id"])] = item
    for item in evidence.values():
        conn.execute(
            "INSERT OR REPLACE INTO evidence(evidence_id, json) VALUES (?, ?)",
            (item["evidence_id"], json.dumps(item, ensure_ascii=False)),
        )
    conn.execute(
        "INSERT OR REPLACE INTO source_coverage(coverage_id, json) VALUES (?, ?)",
        (coverage["coverage_id"], json.dumps(coverage, ensure_ascii=False)),
    )
    for manifest in result.get("manifests") or []:
        conn.execute(
            "INSERT INTO manifest(source_id, url, sha256, bytes, http_status, retrieved_at, note) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                coverage["source_id"],
                manifest.get("url"),
                manifest.get("raw_sha256"),
                manifest.get("raw_bytes"),
                manifest.get("http_status", 200),
                manifest.get("retrieved_at"),
                "Eduskunta public speech search receipt",
            ),
        )
    return {"official_objects": len(objects), "evidence": len(evidence), "source_coverage": 1}


acquire_registry = acquire_speeches


__all__ = [
    "DEFAULT_WINDOWS",
    "SPEECH_COUNT_URL",
    "SPEECH_KIND",
    "SPEECH_SEARCH_URL",
    "SPEECH_SOURCE_ID",
    "acquire_full_term",
    "acquire_registry",
    "acquire_speeches",
    "full_term_windows",
    "import_speeches",
    "normalize_speech_records",
    "parse_speech_result",
]

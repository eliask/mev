"""Campaign text.

2023 named promises come from Yle's still-public vaalikone JSON, one file per
electoral district. That file is not the CC-BY anonymous answer dump. The
anonymous dump has no names and is not joined to people.

2019 Yle open data is an anonymized CC-BY file. It is counted and stored, and
it is not attributed to candidates.
"""


import csv
import hashlib
import html
import io
import json
import zipfile
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from paa.config import RAW, ensure_dirs
from paa.http_client import get_bytes
from paa.identity import names_compatible
from paa.store import add_manifest, connect

CONSTITUENCIES = range(1, 14)
PROMISE_API = (
    "https://old-vaalikone.yle.fi/vaalikone/eduskuntavaalit2023/api/public/constituencies/{cid}/candidates"
)
ANON_2019 = "https://vaalit.beta.yle.fi/avoindata/avoin_data_eduskuntavaalit_2019.zip"
OPEN_2015 = "http://data.yle.fi/dokumentit/Eduskuntavaalit2015/vastaukset_avoimena_datana.csv"
PAGE = "https://vaalit.yle.fi/ev2023/tulospalvelu/fi/electoral-districts/{cid}/candidates/{number}/"
YLE_2011_PUBLICATION = "https://yle.fi/aihe/a/20-162059"
YLE_2011_SHEET = (
    "https://spreadsheets.google.com/ccc?hl=en&key="
    "0As57fWmFoiWhdENxclR6YmFBYmlhQTJjSS1kdmJlNFE"
)
YLE_2011_SHEET_ID = "1yOLYmnWXtIutqpojnvktnDpBdAxtNzcsc5MLlbAxNfg"
YLE_2011_EXPORT = f"https://docs.google.com/spreadsheets/d/{YLE_2011_SHEET_ID}/export?format=csv"
YLE_2011_LICENSE = "CC-BY-NC-SA 3.0"
YLE_2011_LICENSE_URL = "https://creativecommons.org/licenses/by-nc-sa/3.0/deed.fi"
YLE_2011_ATTRIBUTION = "Yle Uutisten vaalikone 2011"
YLE_2011_KNOWN_SHA256 = "4807535a042ae03bcc6ed4ee8c6dd58ca9526b938421d7802376ff23788ade39"
YLE_2011_QUESTION = "Mitä asioita haluat edistää tai ajaa tulevalla vaalikaudella"


def _fi(value) -> str:
    if isinstance(value, dict):
        return str(value.get("fi") or "").strip()
    if isinstance(value, str):
        return value.strip()
    return ""


def acquire_2023() -> dict:
    ensure_dirs()
    conn = connect()
    now = datetime.now(UTC).isoformat()
    conn.execute("DELETE FROM documents WHERE source_id = 'SRC-YLE-2023'")
    stored = 0
    people = 0
    empty = 0
    unmatched = 0
    for cid in CONSTITUENCIES:
        url = PROMISE_API.format(cid=cid)
        status, body, _ctype = get_bytes(url)
        path = RAW / "yle" / f"ekv2023-c{cid}.json"
        path.write_bytes(body)
        digest = hashlib.sha256(body).hexdigest()
        add_manifest(
            conn,
            source_id="SRC-YLE-2023",
            url=url,
            sha256=digest,
            bytes=len(body),
            http_status=status,
            retrieved_at=now,
            note="named vaalikone JSON; not the CC-BY anonymous Likert dump",
        )
        if status != 200:
            print(f"yle 2023 district {cid}: HTTP {status}")
            continue
        rows = json.loads(body)
        for row in rows:
            people += 1
            info = row.get("info") or {}
            promises = []
            for index in (1, 2, 3):
                raw_promise = info.get(f"election_promise_{index}") or {}
                finnish = _fi(raw_promise)
                swedish = str(raw_promise.get("se") or "").strip() if isinstance(raw_promise, dict) else ""
                if finnish and finnish != "-":
                    promises.append(("fi", finnish))
                elif swedish and swedish != "-":
                    # Åland answers are Swedish. An empty Finnish field is not a missing promise.
                    promises.append(("sv", swedish))
            if not promises:
                empty += 1
                continue
            district = f"{int(row['constituency_id']):02d}"
            number = int(row.get("election_number") or 0)
            match = conn.execute(
                """SELECT candidacy_id, first_name, last_name, name_key FROM candidacies
                   WHERE election_year = 2023 AND district_code = ? AND candidate_number = ?""",
                (district, number),
            ).fetchall()
            if len(match) == 1 and not names_compatible(
                row.get("first_name") or "",
                row.get("last_name") or "",
                match[0]["first_name"],
                match[0]["last_name"],
            ):
                match = []
            actor_hint = match[0]["candidacy_id"] if len(match) == 1 else ""
            if len(match) != 1:
                unmatched += 1
            page = PAGE.format(cid=row["constituency_id"], number=row.get("election_number"))
            for index, (promise_language, text) in enumerate(promises, start=1):
                document_id = f"yle2023-{row['id']}-{index}"
                conn.execute(
                    """INSERT OR REPLACE INTO documents(
                        document_id, source_id, url, actor_id, field_label, language,
                        text, stated_earliest, sha256, http_status, retrieved_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        document_id,
                        "SRC-YLE-2023",
                        page,
                        actor_hint,
                        "Vaalilupaukset",
                        promise_language,
                        text,
                        "2023-03-01",
                        hashlib.sha256(text.encode("utf-8")).hexdigest(),
                        200,
                        now,
                    ),
                )
                stored += 1
        print(f"yle 2023 district {cid}: {len(rows)} profiles")
    conn.commit()
    conn.close()
    stats = {"profiles": people, "promise_fields": stored, "profiles_without_promise": empty, "unmatched_name": unmatched}
    print("yle 2023", stats)
    return stats


def _publication_is_source_grounded(body: bytes) -> tuple[bool, str | None]:
    """Check that the fetched page is the cited Yle publication, not an error page."""

    text = html.unescape(body.decode("utf-8", errors="replace"))
    if YLE_2011_SHEET_ID not in text and "0As57fWmFoiWhdENxclR6YmFBYmlhQTJjSS1kdmJlNFE" not in text:
        return False, "publication does not contain the cited spreadsheet link"
    if YLE_2011_LICENSE not in text and "creativecommons.org/licenses/by-nc-sa/3.0" not in text:
        return False, "publication does not contain the declared CC license"
    return True, None


def _read_2011_csv(path: Path) -> tuple[list[str], list[list[str]]]:
    try:
        text = path.read_text(encoding="utf-8-sig")
        rows = list(csv.reader(io.StringIO(text)))
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ValueError(f"cannot read Yle 2011 CSV: {exc}") from exc
    if not rows:
        raise ValueError("Yle 2011 CSV is empty")
    header = rows[0]
    required = {"Vaalipiiri", "id", "Sukunimi", "Etunimi", "Ehdokasnumero", YLE_2011_QUESTION}
    missing = sorted(required - set(header))
    if missing:
        raise ValueError(f"Yle 2011 CSV lacks required columns: {', '.join(missing)}")
    malformed = next(
        ((line_number, len(row)) for line_number, row in enumerate(rows[1:], start=2) if len(row) != len(header)),
        None,
    )
    if malformed:
        line_number, width = malformed
        raise ValueError(f"Yle 2011 CSV row {line_number} has {width} fields; expected {len(header)}")
    return header, rows[1:]


def _cache_2011_source(
    *,
    raw_dir: Path,
    refresh: bool,
    fetcher: Callable[[str], tuple[int, bytes, str]],
) -> dict:
    """Fetch or replay the immutable Yle publication and CSV cache."""

    raw_dir.mkdir(parents=True, exist_ok=True)
    csv_path = raw_dir / "yle2011.csv"
    manifest_path = raw_dir / "yle2011.manifest.json"
    publication_path = raw_dir / "yle2011-publication.html"
    cache = None
    if csv_path.exists() and manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            body = csv_path.read_bytes()
            digest = hashlib.sha256(body).hexdigest()
            if digest != manifest.get("raw_sha256") or len(body) != int(manifest.get("raw_bytes") or -1):
                raise ValueError("cached Yle 2011 CSV hash/size mismatch")
            required_manifest = {
                "source_id",
                "url",
                "publication_url",
                "raw_sha256",
                "raw_bytes",
                "publication_raw_sha256",
                "publication_raw_bytes",
                "license",
                "attribution",
            }
            missing_manifest = sorted(required_manifest - set(manifest))
            if missing_manifest:
                raise ValueError(f"cached Yle 2011 manifest lacks: {', '.join(missing_manifest)}")
            _read_2011_csv(csv_path)
            cache = {"manifest": manifest, "body": body, "cache_hit": True}
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            cache = {"cache_error": str(exc)}
        if cache and "body" in cache and not refresh:
            return {
                "available": True,
                "path": csv_path,
                "manifest": cache["manifest"],
                "cache_hit": True,
                "cache_fallback": False,
            }

    retrieved_at = datetime.now(UTC).isoformat()
    publication_status = None
    publication_body = b""
    try:
        publication_status, publication_body, publication_type = fetcher(YLE_2011_PUBLICATION)
    except (OSError, RuntimeError, ValueError) as exc:
        if cache and "cache_error" not in cache:
            return {
                "available": True,
                "path": csv_path,
                "manifest": cache["manifest"],
                "cache_hit": True,
                "cache_fallback": True,
                "refresh_error": f"{type(exc).__name__}: {exc}",
            }
        return {
            "available": False,
            "reason": "publication_fetch_failed",
            "error": f"{type(exc).__name__}: {exc}",
            "url": YLE_2011_PUBLICATION,
        }
    if publication_status != 200:
        if cache and "cache_error" not in cache:
            return {
                "available": True,
                "path": csv_path,
                "manifest": cache["manifest"],
                "cache_hit": True,
                "cache_fallback": True,
                "refresh_error": f"publication HTTP {publication_status}",
            }
        return {
            "available": False,
            "reason": "publication_http_error",
            "http_status": publication_status,
            "url": YLE_2011_PUBLICATION,
        }
    grounded, grounding_error = _publication_is_source_grounded(publication_body)
    if not grounded:
        return {"available": False, "reason": "publication_not_source_grounded", "error": grounding_error}

    try:
        export_status, csv_body, export_type = fetcher(YLE_2011_EXPORT)
    except (OSError, RuntimeError, ValueError) as exc:
        if cache and "cache_error" not in cache:
            return {
                "available": True,
                "path": csv_path,
                "manifest": cache["manifest"],
                "cache_hit": True,
                "cache_fallback": True,
                "refresh_error": f"{type(exc).__name__}: {exc}",
            }
        return {
            "available": False,
            "reason": "sheet_export_fetch_failed",
            "error": f"{type(exc).__name__}: {exc}",
            "url": YLE_2011_EXPORT,
        }
    if export_status != 200:
        if cache and "cache_error" not in cache:
            return {
                "available": True,
                "path": csv_path,
                "manifest": cache["manifest"],
                "cache_hit": True,
                "cache_fallback": True,
                "refresh_error": f"sheet export HTTP {export_status}",
            }
        return {
            "available": False,
            "reason": "sheet_export_http_error",
            "http_status": export_status,
            "url": YLE_2011_EXPORT,
        }

    digest = hashlib.sha256(csv_body).hexdigest()
    temporary = raw_dir / ".yle2011.csv.download"
    temporary.write_bytes(csv_body)
    try:
        header, rows = _read_2011_csv(temporary)
    except ValueError as exc:
        temporary.unlink(missing_ok=True)
        return {"available": False, "reason": "sheet_export_invalid_csv", "error": str(exc)}
    temporary.replace(csv_path)
    archive_dir = raw_dir / "by-sha256"
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archive_dir / f"{digest}.csv"
    if archive_path.exists():
        if hashlib.sha256(archive_path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"corrupt Yle 2011 immutable source: {archive_path}")
    else:
        archive_path.write_bytes(csv_body)
    publication_digest = hashlib.sha256(publication_body).hexdigest()
    publication_path.write_bytes(publication_body)
    publication_archive = archive_dir / f"{publication_digest}.html"
    if not publication_archive.exists():
        publication_archive.write_bytes(publication_body)
    manifest = {
        "source_id": "SRC-YLE-2011",
        "url": YLE_2011_EXPORT,
        "publication_url": YLE_2011_PUBLICATION,
        "sheet_url": YLE_2011_SHEET,
        "raw_sha256": digest,
        "raw_bytes": len(csv_body),
        "publication_raw_sha256": publication_digest,
        "publication_raw_bytes": len(publication_body),
        "http_status": export_status,
        "publication_http_status": publication_status,
        "content_type": export_type,
        "publication_content_type": publication_type,
        "retrieved_at": retrieved_at,
        "row_count": len(rows),
        "headers": header,
        "license": YLE_2011_LICENSE,
        "license_url": YLE_2011_LICENSE_URL,
        "attribution": YLE_2011_ATTRIBUTION,
        "attribution_url": YLE_2011_PUBLICATION,
        "known_hash_match": digest == YLE_2011_KNOWN_SHA256,
        "archive_path": str(archive_path),
        "publication_archive_path": str(publication_archive),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"available": True, "path": csv_path, "manifest": manifest, "cache_hit": False, "cache_fallback": False}


def _load_2011_csv(conn, path: Path, *, source_url: str = YLE_2011_PUBLICATION, http_status: int = 200,
                   retrieved_at: str | None = None) -> dict:
    """Load only the named 2011 open-answer field; retain no false promise labels."""

    header, rows = _read_2011_csv(path)
    index = {name: position for position, name in enumerate(header)}
    question = YLE_2011_QUESTION
    conn.execute("DELETE FROM documents WHERE source_id = 'SRC-YLE-2011'")
    loaded = 0
    linked = 0
    for row in rows:
        text = (row[index[question]] or "").strip()
        if not text or text == "-":
            continue
        district = row[index["Vaalipiiri"]][:2]
        try:
            number = int(row[index["Ehdokasnumero"]] or 0)
        except ValueError:
            number = 0
        match = []
        if number:
            match = conn.execute(
                """SELECT candidacy_id, first_name, last_name FROM candidacies
                   WHERE election_year = 2011 AND district_code = ? AND candidate_number = ?""",
                (district, number),
            ).fetchall()
        hint = ""
        if len(match) == 1:
            yle_first, yle_last = row[index["Etunimi"]], row[index["Sukunimi"]]
            official_first, official_last = match[0]["first_name"], match[0]["last_name"]
            if names_compatible(yle_first, yle_last, official_first, official_last) or names_compatible(
                yle_last, yle_first, official_first, official_last
            ):
                hint = match[0]["candidacy_id"]
                linked += 1
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        conn.execute(
            """INSERT INTO documents(
                document_id, source_id, url, actor_id, field_label, language, text,
                stated_earliest, sha256, http_status, retrieved_at
            ) VALUES (?, 'SRC-YLE-2011', ?, ?, ?, 'fi', ?, '2011-03-01', ?, ?, ?)""",
            (
                f"yle2011-{row[index['id']]}",
                source_url,
                hint,
                question,
                text,
                digest,
                http_status,
                retrieved_at,
            ),
        )
        loaded += 1
    return {"open_answers": loaded, "linked": linked, "csv_rows": len(rows)}


def acquire_2011(
    conn=None,
    *,
    raw_dir: Path | None = None,
    refresh: bool = False,
    fetcher: Callable[[str], tuple[int, bytes, str]] | None = None,
) -> dict:
    """Acquire and load Yle's named 2011 open answers from the cited publication.

    The Google Sheet is the source body linked by Yle's article.  A valid local
    cache is replayed without network access; a fresh download is retained by
    content hash with publication, license, and attribution metadata.  Any
    unavailable or invalid source is returned explicitly and cannot silently
    delete existing documents.
    """

    ensure_dirs()
    own_connection = conn is None
    database = conn or connect()
    try:
        result = _cache_2011_source(
            raw_dir=raw_dir or RAW / "yle",
            refresh=refresh,
            fetcher=fetcher or get_bytes,
        )
        if not result.get("available"):
            return {"source_id": "SRC-YLE-2011", **result, "loaded": 0}
        manifest = result["manifest"]
        loaded = _load_2011_csv(
            database,
            result["path"],
            source_url=YLE_2011_PUBLICATION,
            http_status=int(manifest.get("http_status") or 200),
            retrieved_at=manifest.get("retrieved_at"),
        )
        add_manifest(
            database,
            source_id="SRC-YLE-2011",
            url=manifest["url"],
            sha256=manifest["raw_sha256"],
            bytes=manifest["raw_bytes"],
            http_status=manifest.get("http_status", 200),
            retrieved_at=manifest.get("retrieved_at"),
            note=(
                f"{YLE_2011_ATTRIBUTION}; license {YLE_2011_LICENSE}; "
                f"publication {YLE_2011_PUBLICATION}; publication_raw_sha256={manifest['publication_raw_sha256']}"
            ),
        )
        add_manifest(
            database,
            source_id="SRC-YLE-2011-PUBLICATION",
            url=manifest["publication_url"],
            sha256=manifest["publication_raw_sha256"],
            bytes=manifest["publication_raw_bytes"],
            http_status=manifest.get("publication_http_status", 200),
            retrieved_at=manifest.get("retrieved_at"),
            note=f"Official Yle publication; links the sheet and declares {YLE_2011_LICENSE}.",
        )
        database.commit()
        return {
            "source_id": "SRC-YLE-2011",
            **result,
            **loaded,
            "loaded": loaded["open_answers"],
            "license": manifest["license"],
            "attribution": manifest["attribution"],
        }
    finally:
        if own_connection:
            database.close()


def load_2011_open_answers(conn) -> dict:
    """Replay the cached 2011 open-answer file without doing network I/O."""

    path = RAW / "yle" / "yle2011.csv"
    if not path.exists():
        return {"loaded": 0, "reason": "file missing"}
    try:
        metadata_path = path.with_suffix(".manifest.json")
        metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        capture = metadata.get("retrieved_at") if hashlib.sha256(path.read_bytes()).hexdigest() == metadata.get("raw_sha256") else None
        loaded = _load_2011_csv(conn, path, retrieved_at=capture)
        return {
            "loaded": loaded["open_answers"],
            "linked": loaded["linked"],
            "rows": loaded["csv_rows"],
        }
    except ValueError as exc:
        return {"loaded": 0, "reason": str(exc)}


def restore_cached_capture_dates(conn, raw_dir: Path | None = None) -> dict:
    """Recover legacy capture dates only from hash-matched source receipts.

    This performs no network requests. An exact document ID/text match must
    exist in the preserved source body. The assessment cutoff is never used
    as a substitute for an absent acquisition timestamp.
    """
    directory = raw_dir or RAW / "yle"
    updated, receipts = 0, []
    documents = {row["document_id"]: dict(row) for row in conn.execute("SELECT * FROM documents")}

    def apply(document_id, text, captured_at):
        nonlocal updated
        document = documents.get(document_id)
        if not document or document.get("retrieved_at") or document["text"] != text or not captured_at:
            return
        if document["sha256"] != hashlib.sha256(text.encode()).hexdigest():
            return
        conn.execute("UPDATE documents SET retrieved_at=? WHERE document_id=?", (captured_at, document_id))
        updated += 1

    for cid in CONSTITUENCIES:
        path = directory / f"ekv2023-c{cid}.json"
        if not path.exists():
            continue
        body = path.read_bytes()
        raw_hash = hashlib.sha256(body).hexdigest()
        observations = conn.execute(
            "SELECT retrieved_at FROM manifest WHERE source_id='SRC-YLE-2023' AND url=? AND sha256=? AND http_status=200 AND retrieved_at IS NOT NULL",
            (PROMISE_API.format(cid=cid), raw_hash)).fetchall()
        if not observations:
            continue
        capture = max((row["retrieved_at"] for row in observations), key=datetime.fromisoformat)
        before = updated
        for record in json.loads(body):
            texts = []
            for index in (1, 2, 3):
                value = (record.get("info") or {}).get(f"election_promise_{index}") or {}
                fi = _fi(value)
                sv = str(value.get("se") or "").strip() if isinstance(value, dict) else ""
                if fi and fi != "-":
                    texts.append(fi)
                elif sv and sv != "-":
                    texts.append(sv)
            for index, text in enumerate(texts, 1):
                apply(f"yle2023-{record['id']}-{index}", text, capture)
        receipts.append({"source_id": "SRC-YLE-2023", "url": PROMISE_API.format(cid=cid),
                         "raw_sha256": raw_hash, "retrieved_at": capture, "documents_updated": updated - before})
    metadata_path = directory / "yle2011.manifest.json"
    csv_path = directory / "yle2011.csv"
    if metadata_path.exists() and csv_path.exists():
        metadata = json.loads(metadata_path.read_text())
        raw_hash = hashlib.sha256(csv_path.read_bytes()).hexdigest()
        if raw_hash == metadata.get("raw_sha256") and metadata.get("retrieved_at"):
            header, rows = _read_2011_csv(csv_path)
            indices = {name: index for index, name in enumerate(header)}
            before = updated
            for row in rows:
                text = row[indices[YLE_2011_QUESTION]].strip()
                if text and text != "-":
                    apply(f"yle2011-{row[indices['id']]}", text, metadata["retrieved_at"])
            receipts.append({"source_id": "SRC-YLE-2011", "url": metadata["url"], "raw_sha256": raw_hash,
                             "retrieved_at": metadata["retrieved_at"], "documents_updated": updated - before})
    return {"documents_updated": updated, "hash_bound_source_receipts": receipts,
            "unknown_capture_dates": conn.execute("SELECT COUNT(*) FROM documents WHERE retrieved_at IS NULL").fetchone()[0]}


def relink_local(conn) -> dict:
    """Reattach stored 2023 promises using the candidate number and a tolerant name check."""
    files = sorted((RAW / "yle").glob("ekv2023-c*.json"))
    linked = 0
    rejected = 0
    for path in files:
        for row in json.loads(path.read_text(encoding="utf-8")):
            district = f"{int(row['constituency_id']):02d}"
            number = int(row.get("election_number") or 0)
            match = conn.execute(
                """SELECT candidacy_id, first_name, last_name FROM candidacies
                   WHERE election_year = 2023 AND district_code = ? AND candidate_number = ?""",
                (district, number),
            ).fetchall()
            hint = ""
            if len(match) == 1 and names_compatible(
                row.get("first_name") or "",
                row.get("last_name") or "",
                match[0]["first_name"],
                match[0]["last_name"],
            ):
                hint = match[0]["candidacy_id"]
                linked += 1
            else:
                rejected += 1
            conn.execute(
                "UPDATE documents SET actor_id = ? WHERE document_id LIKE ?",
                (hint, f"yle2023-{row['id']}-%"),
            )
    return {"profiles_linked": linked, "profiles_rejected": rejected}


def acquire_2019_anonymous() -> dict:
    """Store the anonymized release as a count. Do not attach it to named people."""
    ensure_dirs()
    status, body, _ctype = get_bytes(ANON_2019)
    path = RAW / "yle" / "yle2019.zip"
    stats = {"http_status": status, "bytes": len(body), "rows": 0, "nonempty_promise_1": 0, "attributed": False}
    if status != 200:
        stats["note"] = "download failed"
        return stats
    path.write_bytes(body)
    with zipfile.ZipFile(io.BytesIO(body)) as archive:
        names = [name for name in archive.namelist() if name.endswith(".csv")]
        if not names:
            stats["note"] = "zip had no csv"
            return stats
        raw = archive.read(names[0])
    text = raw.decode("utf-8-sig", errors="replace")
    reader = csv.DictReader(io.StringIO(text))
    fieldnames = reader.fieldnames or []
    promise_cols = [name for name in fieldnames if "vaalilupaus" in name.casefold() or "lupaus" in name.casefold()]
    nonempty = 0
    rows = 0
    for row in reader:
        rows += 1
        if promise_cols and (row.get(promise_cols[0]) or "").strip() not in {"", "-"}:
            nonempty += 1
    stats.update(
        {
            "rows": rows,
            "nonempty_promise_1": nonempty,
            "promise_columns": promise_cols[:6],
            "name_column_present": any("nimi" in (name or "").casefold() or "name" in (name or "").casefold() for name in fieldnames),
            "note": "CC-BY anonymized 2019 release. Not joined to candidates.",
        }
    )
    conn = connect()
    add_manifest(
        conn,
        source_id="SRC-YLE-2019-ANON",
        url=ANON_2019,
        sha256=hashlib.sha256(body).hexdigest(),
        bytes=len(body),
        http_status=status,
        retrieved_at=datetime.now(UTC).isoformat(),
        note=stats["note"],
    )
    conn.commit()
    conn.close()
    print("yle 2019 anonymous", {k: stats[k] for k in ("rows", "nonempty_promise_1", "name_column_present")})
    return stats


def probe_2015() -> dict:
    """The 2015 open-data host has been timing out. One short try, then record the miss."""
    import httpx

    try:
        response = httpx.get(OPEN_2015, headers={"User-Agent": "paa/0.1"}, timeout=12, follow_redirects=True)
    except (httpx.HTTPError, OSError) as exc:
        return {"url": OPEN_2015, "ok": False, "error": type(exc).__name__}
    return {
        "url": OPEN_2015,
        "ok": response.status_code == 200 and len(response.content) > 1000,
        "http_status": response.status_code,
        "bytes": len(response.content),
    }

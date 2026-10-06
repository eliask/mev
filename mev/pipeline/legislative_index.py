"""Finnish Legislative Data Indexing Service
Build a cross-referenced SQLite index for mekanismitesti HEs.

Active tables (read by downstream code):
    he               — read by comm_evade.py, ev_outcome.py
    committee_report — read by enrich_he_db.py, tag_mietinto.py, comm_evade.py, scrutiny.py
    expert_statement — read by enrich_he_db.py, scrutiny.py

Data flow:
    APIs (Finlex, Hankeikkuna, Eduskunta, Lakitutka, ...)
      → data/legislative_index.sqlite  [THIS SCRIPT]
        → .tmp/he_dbs/he-*.db          [enrich_he_db.py]

Source notes:
    LAKITUTKA — expert statements + committee report metadata
        Search: https://lakitutka.fi/api/search/all?term=HE+N/YYYY&sort=date_desc&lang=fi&size=200
        Quality for expert statements: GOOD — full content via sisalto field.
        Quality for committee reports: GOOD via Vaski XML; Lakitutka fallback is single-<p>.
        Status: ACTIVE for expert_statement + committee_report.

    EDUSKUNTA AVOINDATA (VaskiData) — official, structured XML
        API: https://avoindata.eduskunta.fi/api/v1/tables/VaskiData/rows?...
        Returns: XmlData blobs with VN XML schema. STRUCTURED content with
                 PerusteluOsa > PerusteluLuku > KappaleKooste (individual paragraphs).
                 Coverage ~2015+.
        Status: ACTIVE for committee_report metadata + mietintö content.
        NOTE: eduskunta.fi HTML pages are bot-hostile (403/timeout). Don't scrape.

    FINLEX — HE full text (local ZIP)
        Local ZIP: ~/Downloads/government-proposal.zip
        Status: ACTIVE for he.content (primary source via local ZIP).

    HANKEIKKUNA — project/preparation metadata
        API: https://api.hankeikkuna.fi/hankkeet/...
        Returns: JSON with preparation timeline, ministry, LAN status.
        Status: ACTIVE for he metadata (uuid, lan_consulted, hanke_metadata).

    Known gaps (TODO):
        - Pre-2015 committee report content: Vaski doesn't cover these.
          Lakitutka has degraded single-<p> content. Acceptable for now.
        - Final law number (säädöskokoelma): not currently stored.

Usage:
    uv run mev pipeline index
    uv run mev pipeline index --he 38/2025
"""

import asyncio
import hashlib
import json
import logging
import os
import re
import sqlite3
import sys
import time
import urllib.parse
import argparse
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path

import httpx
from farchive import BatchItem, CompressionPolicy, Farchive
from lxml import etree
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

from mev.config import BOOK_ROOT as ROOT, ROOT as MEV_ROOT
DB_PATH = ROOT / "data" / "legislative_index.sqlite"
CACHE_DIR = ROOT / ".cache" / "megadoc"
CACHE_ARCHIVE = Path(os.environ.get("MEV_API_CACHE_FARCHIVE", MEV_ROOT / "data" / "mev_api_cache.farchive"))
USE_FARCHIVE_CACHE = os.environ.get("MEV_USE_FARCHIVE_CACHE", "1") != "0"
WRITE_LEGACY_CACHE = os.environ.get("MEV_WRITE_LEGACY_CACHE", "0") == "1"

# ---------------------------------------------------------------------------
# Local Finlex ZIP handling
# ---------------------------------------------------------------------------

class LocalFinlex:
    """Provides access to HE data from local ZIP file in ~/Downloads."""
    def __init__(self):
        self.he_zip_path = Path.home() / "Downloads" / "government-proposal.zip"
        self._he_zf = None
        self._he_index = {}
        self._indexed = False

    def _ensure_indexed(self):
        if self._indexed:
            return
        if self.he_zip_path.exists():
            log.info("Indexing local HE zip: %s", self.he_zip_path)
            self._he_zf = zipfile.ZipFile(self.he_zip_path, 'r')
            for name in self._he_zf.namelist():
                if name.endswith('/main.xml') and 'government-proposal' in name:
                    parts = name.split('/')
                    try:
                        year = int(parts[4])
                        number = int(parts[5])
                        lang = parts[6].rstrip('@')
                        if lang == 'fin':
                            self._he_index[(year, number)] = name
                    except (IndexError, ValueError):
                        continue
        self._indexed = True

    def get_he(self, year: int, number: int) -> bytes | None:
        if not self.he_zip_path.exists(): return None
        self._ensure_indexed()
        path = self._he_index.get((year, number))
        if path and self._he_zf:
            return self._he_zf.read(path)
        return None

    def close(self):
        if self._he_zf: self._he_zf.close()

local_finlex = LocalFinlex()

USER_AGENT = "MeV-Legislative-Indexer/0.1 (research; mekanismirealismi.fi/mev)"
RATE_LIMIT_SLEEP = 1.5  # seconds between uncached API calls

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("indexer")


# ---------------------------------------------------------------------------
# Caching layer
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class CacheMigrationStats:
    files_seen: int = 0
    files_imported: int = 0
    files_same_head: int = 0
    files_missing: int = 0
    bytes_imported: int = 0
    errors: int = 0


def cache_key(method: str, url: str, body: str | None = None) -> str:
    """Deterministic cache key from request parameters."""
    h = hashlib.sha256(f"{method}:{url}:{body or ''}".encode()).hexdigest()[:16]
    # Sanitize URL for filename
    safe = re.sub(r'[^\w\-.]', '_', url.split("//", 1)[-1])[:80]
    return f"{method}_{safe}_{h}"


def _cache_locator(key: str) -> str:
    return f"mev-cache://megadoc/{key}"


def _storage_class_for_cache_ext(ext: str) -> str:
    if ext == ".xml":
        return "xml"
    if ext == ".json":
        return "json"
    if ext in (".html", ".htm"):
        return "html"
    if ext == ".txt":
        return "text"
    return "bin"


def _cache_metadata(
    *,
    key: str,
    method: str | None = None,
    url: str | None = None,
    body_sha256: str | None = None,
    ext: str | None = None,
    source: str = "mev.legislative_index",
) -> dict:
    return {
        "source": source,
        "cache_key": key,
        "method": method,
        "url": url,
        "body_sha256": body_sha256,
        "ext": ext,
    }


def _open_cache_archive() -> Farchive:
    # The MeV API cache contains many unrelated payloads keyed by request
    # locator. Delta search across the whole cache is expensive and low-value;
    # farchive still provides content-addressing, deduplication, zstd storage,
    # and per-locator history with delta disabled.
    return Farchive(CACHE_ARCHIVE, compression=CompressionPolicy(delta_enabled=False))


def get_cached(key: str, *, archive: Farchive | None = None) -> bytes | None:
    if USE_FARCHIVE_CACHE:
        if archive is None:
            with _open_cache_archive() as fa:
                data = fa.get(_cache_locator(key))
        else:
            data = archive.get(_cache_locator(key))
        if data is not None:
            return data

    for ext in (".json", ".xml", ".bin"):
        p = CACHE_DIR / f"{key}{ext}"
        if p.exists():
            data = p.read_bytes()
            # Read-through migration for touched legacy entries only. This is
            # conservative: it never scans 85k loose files during normal runs,
            # but future reads become farchive-backed without refetching.
            if USE_FARCHIVE_CACHE:
                put_cache(
                    key,
                    data,
                    ext=ext,
                    archive=archive,
                    metadata=_cache_metadata(key=key, ext=ext, source="legacy.megadoc.readthrough"),
                )
            return data
    return None


def put_cache(
    key: str,
    data: bytes,
    ext: str = ".json",
    *,
    archive: Farchive | None = None,
    metadata: dict | None = None,
) -> None:
    if USE_FARCHIVE_CACHE:
        locator = _cache_locator(key)
        storage_class = _storage_class_for_cache_ext(ext)
        if archive is None:
            with _open_cache_archive() as fa:
                fa.store(
                    locator,
                    data,
                    storage_class=storage_class,
                    series_key=f"mev-cache://megadoc/{storage_class}",
                    metadata=metadata or _cache_metadata(key=key, ext=ext),
                )
        else:
            archive.store(
                locator,
                data,
                storage_class=storage_class,
                series_key=f"mev-cache://megadoc/{storage_class}",
                metadata=metadata or _cache_metadata(key=key, ext=ext),
            )

    if WRITE_LEGACY_CACHE:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        (CACHE_DIR / f"{key}{ext}").write_bytes(data)


def migrate_legacy_cache_to_farchive(*, limit: int | None = None) -> CacheMigrationStats:
    """Import existing loose cache files into farchive without fetching."""
    stats = CacheMigrationStats()
    if not CACHE_DIR.exists():
        return stats
    candidates = sorted(
        p for p in CACHE_DIR.iterdir()
        if p.is_file() and p.suffix in (".json", ".xml", ".bin")
    )
    with _open_cache_archive() as fa:
        batch: list[BatchItem] = []
        for path in candidates:
            if limit is not None and stats.files_seen >= limit:
                break
            stats.files_seen += 1
            key = path.stem
            locator = _cache_locator(key)
            try:
                data = path.read_bytes()
            except FileNotFoundError:
                stats.files_missing += 1
                continue
            except OSError as exc:
                log.warning("CACHE MIGRATION READ ERROR: %s — %s", path, exc)
                stats.errors += 1
                continue
            digest = hashlib.sha256(data).hexdigest()
            current = fa.compare_current(locator, digest=digest)
            if current.status == "same":
                stats.files_same_head += 1
                continue
            batch.append(
                BatchItem(
                    locator=locator,
                    data=data,
                    storage_class=_storage_class_for_cache_ext(path.suffix),
                    series_key=f"mev-cache://megadoc/{_storage_class_for_cache_ext(path.suffix)}",
                    metadata=_cache_metadata(
                        key=key,
                        ext=path.suffix,
                        source="legacy.megadoc.migration",
                    ),
                )
            )
            stats.files_imported += 1
            stats.bytes_imported += len(data)
            if len(batch) >= 1000:
                fa.store_batch(batch)
                log.info(
                    "CACHE MIGRATION PROGRESS: seen=%d imported=%d same_head=%d bytes_imported=%d",
                    stats.files_seen,
                    stats.files_imported,
                    stats.files_same_head,
                    stats.bytes_imported,
                )
                batch.clear()
        if batch:
            fa.store_batch(batch)
    return stats


# ---------------------------------------------------------------------------
# Rate-limited HTTP client
# ---------------------------------------------------------------------------

_last_api_call = 0.0


async def rate_limit():
    global _last_api_call
    now = time.monotonic()
    wait = RATE_LIMIT_SLEEP - (now - _last_api_call)
    if wait > 0:
        await asyncio.sleep(wait)
    _last_api_call = time.monotonic()


class RetryableHTTPError(Exception):
    pass


@retry(
    retry=retry_if_exception_type(RetryableHTTPError),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=30),
)
async def fetch(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    body: dict | str | None = None,
    headers: dict | None = None,
    cache_ext: str = ".json",
) -> bytes:
    """Cached, rate-limited, retrying HTTP fetch."""
    body_str = json.dumps(body, sort_keys=True) if isinstance(body, dict) else body
    key = cache_key(method, url, body_str)

    cached = get_cached(key)
    if cached is not None:
        log.debug("CACHE HIT: %s %s", method, url[:80])
        return cached

    await rate_limit()
    log.info("FETCH: %s %s", method, url[:80])

    req_headers = {"User-Agent": USER_AGENT}
    if headers:
        req_headers.update(headers)

    try:
        if method == "GET":
            resp = await client.get(url, headers=req_headers)
        elif method == "POST":
            if isinstance(body, dict):
                resp = await client.post(url, json=body, headers=req_headers)
            else:
                resp = await client.post(url, content=body, headers=req_headers)
        else:
            raise ValueError(f"Unsupported method: {method}")
    except httpx.TimeoutException as e:
        log.warning("TIMEOUT: %s %s — %s", method, url[:80], e)
        raise RetryableHTTPError(str(e)) from e
    except httpx.ConnectError as e:
        log.warning("CONNECT ERROR: %s %s — %s", method, url[:80], e)
        raise RetryableHTTPError(str(e)) from e

    if resp.status_code == 429:
        log.warning("RATE LIMITED (429): %s", url[:80])
        raise RetryableHTTPError("429 Too Many Requests")
    if resp.status_code >= 500:
        log.warning("SERVER ERROR (%d): %s", resp.status_code, url[:80])
        raise RetryableHTTPError(f"HTTP {resp.status_code}")

    if resp.status_code >= 400:
        log.warning("CLIENT ERROR (%d): %s — %s", resp.status_code, url[:80], resp.text[:200])
        return b""

    data = resp.content
    body_sha256 = hashlib.sha256(body_str.encode("utf-8")).hexdigest() if body_str else None
    put_cache(
        key,
        data,
        cache_ext,
        metadata=_cache_metadata(
            key=key,
            method=method,
            url=url,
            body_sha256=body_sha256,
            ext=cache_ext,
            source="mev.legislative_index.fetch",
        ),
    )
    return data


# ---------------------------------------------------------------------------
# API fetchers
# ---------------------------------------------------------------------------

async def fetch_hankeikkuna(client: httpx.AsyncClient, year: int, number: int) -> dict | None:
    """Search Hankeikkuna for an HE project. Returns parsed JSON or None."""
    he_str = f"HE {number}/{year}"
    url = "https://api.hankeikkuna.fi/api/v2/kohteet/haku"
    body = {"teksti": he_str, "tyyppi": ["LAINSAADANTO"], "size": 10}

    data = await fetch(client, "POST", url, body=body)
    if not data:
        log.warning("MISSING_HANKE: No response for %s", he_str)
        return None

    try:
        result = json.loads(data)
    except json.JSONDecodeError:
        log.warning("MISSING_HANKE: Invalid JSON for %s", he_str)
        return None

    # Response uses "result" array (not "kohteet")
    items = result.get("result", [])
    for item in items:
        # HE numbers at: lainsaadanto.heTiedot.heNumerot
        lainsaadanto = item.get("lainsaadanto", {})
        he_tiedot = lainsaadanto.get("heTiedot", {})
        he_numerot = he_tiedot.get("heNumerot", [])
        if he_str in he_numerot:
            uuid = item.get("kohde", {}).get("uuid", "?")
            log.info("  HANKE: Found project %s for %s", uuid[:12], he_str)
            return item

    # Fallback: check if HE string appears anywhere in stringified item
    for item in items:
        if he_str in json.dumps(item):
            log.info("  HANKE: Found project via text match for %s", he_str)
            return item

    log.warning("MISSING_HANKE: No matching project for %s among %d results", he_str, len(items))
    return None


def extract_lausunto_guid(hanke: dict | None) -> str | None:
    """Extract Lausuntopalvelu GUID from Hankeikkuna linkit array."""
    if not hanke:
        return None
    for link in hanke.get("linkit", []):
        # linkit[].url is a dict with language keys: {"fi": "https://...", "sv": "..."}
        url_obj = link.get("url", {})
        if isinstance(url_obj, str):
            urls = [url_obj]
        elif isinstance(url_obj, dict):
            urls = [v for v in url_obj.values() if v]
        else:
            continue
        for url in urls:
            m = re.search(r'proposalId=([0-9a-f\-]{36})', url, re.IGNORECASE)
            if m:
                return m.group(1)
    return None


def extract_hanke_metadata(hanke: dict | None) -> dict:
    """Extract structured metadata from Hankeikkuna response.

    Hankeikkuna result items have a nested structure:
    - kohde: {uuid, tunnus, nimi, tila, asettajaUuid, ...}
    - lainsaadanto: {heTiedot: {heNumerot: [...]}, lainsaadannonArviointineuvosto: bool, ...}
    - linkit: [{url: {fi: "...", sv: "..."}, nimi: {...}}, ...]
    - asiakirjat: [{tyyppi: "LAUSUNTO", ...}, ...]
    """
    if not hanke:
        return {}

    kohde = hanke.get("kohde", {})
    lainsaadanto = hanke.get("lainsaadanto", {})
    he_tiedot = lainsaadanto.get("heTiedot", {})

    meta = {
        "uuid": kohde.get("uuid"),
        "tunnus": kohde.get("tunnus"),
        "nimi": (kohde.get("nimi") or {}).get("fi"),
        "tila": kohde.get("tila"),
        "asettaja": kohde.get("asettajaUuid"),
        "lan_consulted": lainsaadanto.get("lainsaadannonArviointineuvosto"),
        "he_numerot": he_tiedot.get("heNumerot", []),
        "vastuuministeri": (he_tiedot.get("vastuuministeri") or {}).get("fi"),
    }

    # Count documents by type
    docs = hanke.get("asiakirjat", [])
    doc_types = {}
    for d in docs:
        t = d.get("tyyppi", "UNKNOWN")
        doc_types[t] = doc_types.get(t, 0) + 1
    meta["document_counts"] = doc_types
    meta["total_documents"] = len(docs)

    return meta


async def fetch_finlex(client: httpx.AsyncClient, year: int, number: int) -> bytes:
    """Fetch HE full text from local ZIP if available, otherwise fallback to Finlex API."""
    local = local_finlex.get_he(year, number)
    if local:
        log.info("  LOCAL ZIP: Found HE %d/%d", number, year)
        return local

    url = f"https://opendata.finlex.fi/finlex/avoindata/v1/akn/fi/doc/government-proposal/{year}/{number}/fin@"
    return await fetch(client, "GET", url, cache_ext=".xml")



def parse_finlex_refs(xml_data: bytes) -> list[dict]:
    """Parse <ref> elements from Finlex Akoma Ntoso XML to find law cross-references."""
    if not xml_data:
        return []
    refs = []
    try:
        tree = etree.fromstring(xml_data)
    except etree.XMLSyntaxError as e:
        log.warning("FINLEX_PARSE: XML syntax error — %s", e)
        return []

    # Find all <ref> elements (any namespace)
    for ref in tree.iter("{*}ref"):
        href = ref.get("href", "")
        # Match Finlex Akoma Ntoso references:
        #   /akn/fi/act/statute-consolidated/{year}/{number}
        #   /akn/fi/act/statute/{year}/{number}
        #   /eli/sd/{year}/{number}
        #   /sd/{year}/{number}
        m = re.search(r'/(?:akn/fi/act/statute(?:-consolidated)?|eli/sd|sd)/(\d{4})/(\d+)', href)
        if m:
            law_year, law_num = m.group(1), m.group(2)
            law_id = f"{law_num}/{law_year}"
            text = (ref.text or "").strip()
            refs.append({"law_id": law_id, "href": href, "text": text})
            continue
        # Also match references to other HEs (for cross-reference tracking)
        m = re.search(r'/akn/fi/doc/government-proposal/(\d{4})/(\d+)', href)
        if m:
            he_year, he_num = m.group(1), m.group(2)
            refs.append({"law_id": f"HE {he_num}/{he_year}", "href": href, "text": (ref.text or "").strip(), "is_he_ref": True})

    # Deduplicate by law_id
    seen = set()
    unique = []
    for r in refs:
        if r["law_id"] not in seen:
            seen.add(r["law_id"])
            unique.append(r)

    return unique


def parse_finlex_metadata(xml_data: bytes) -> dict:
    """Extract title and ministry from Finlex Akoma Ntoso XML."""
    meta = {"title": None, "ministry": None}
    if not xml_data:
        return meta
    try:
        tree = etree.fromstring(xml_data)
    except etree.XMLSyntaxError:
        return meta

    # Try to find title
    for tag in ["{*}docTitle", "{*}longTitle", "{*}shortTitle"]:
        for el in tree.iter(tag):
            text = etree.tostring(el, method="text", encoding="unicode").strip()
            if text:
                meta["title"] = text[:500]
                break
        if meta["title"]:
            break

    # Try to find ministry (administrativeBranch)
    for el in tree.iter():
        if "administrativeBranch" in el.tag:
            # Value may be in text or in a child element
            text = (el.text or "").strip()
            if not text:
                text = etree.tostring(el, method="text", encoding="unicode").strip()
            if text:
                meta["ministry"] = text
            break

    return meta


async def fetch_eduskunta(client: httpx.AsyncClient, year: int, number: int) -> list[dict]:
    """Fetch VaskiData entries from Eduskunta API for an HE.

    VaskiData format: Eduskuntatunnus = "HE 38/2025 vp" (with spaces and vp suffix).
    rowData is list-of-lists, not list-of-dicts. Must zip with columnNames.
    """
    tunnus = f"HE {number}/{year} vp"
    encoded_tunnus = urllib.parse.quote(tunnus)
    url = f"https://avoindata.eduskunta.fi/api/v1/tables/VaskiData/rows?columnName=Eduskuntatunnus&columnValue={encoded_tunnus}&page=0&perPage=100"

    data = await fetch(client, "GET", url)
    if not data:
        return []

    try:
        result = json.loads(data)
    except json.JSONDecodeError:
        log.warning("EDUSKUNTA: Invalid JSON for %s", tunnus)
        return []

    column_names = result.get("columnNames", [])
    raw_rows = result.get("rowData", [])

    # Convert list-of-lists to list-of-dicts
    rows = [dict(zip(column_names, row)) for row in raw_rows]
    log.info("  EDUSKUNTA: %d VaskiData rows for %s", len(rows), tunnus)
    return rows


def parse_vaski_xml(rows: list[dict]) -> list[dict]:
    """Parse VaskiData XML blobs to extract committee reports and document types."""
    documents = []
    for row in rows:
        xml_str = row.get("XmlData", "")
        if not xml_str:
            continue
        try:
            tree = etree.fromstring(xml_str.encode("utf-8") if isinstance(xml_str, str) else xml_str)
        except etree.XMLSyntaxError:
            continue

        doc = {
            "vaski_id": row.get("Id"),
            "eduskuntatunnus": row.get("Eduskuntatunnus"),
            "status": row.get("Status"),
            "created": row.get("Created"),
        }

        # Extract document type from XML (tags seen in VaskiData blobs)
        for el in tree.iter():
            tag = etree.QName(el).localname if isinstance(el.tag, str) else str(el.tag)
            if tag in ("AsiakirjatyyppiNimi",):
                doc["doc_type"] = (el.text or "").strip()
            elif tag == "AsiakirjatyyppiKoodi":
                doc["doc_type_code"] = (el.text or "").strip()
            elif tag in ("ValiokuntaNimi", "Valiokunta"):
                doc["committee"] = (el.text or "").strip()
            elif tag in ("NimekeTeksti", "Nimike"):
                doc["title"] = (el.text or "").strip()
            elif tag in ("LaadintaPvmTeksti", "PaatospaivaMuotoiltu", "Paatospaiva"):
                doc.setdefault("date", (el.text or "").strip())
            elif tag == "SanomatyyppiNimi":
                doc["message_type"] = (el.text or "").strip()

        documents.append(doc)
    return documents


async def fetch_vaski_by_tunnus(client: httpx.AsyncClient, tunnus: str) -> str:
    """Fetch VaskiData XML for a committee report by its tunnus (e.g. 'HaVM 17/2025 vp').

    Returns the XmlData string, or empty string if not found.
    Vaski coverage starts ~2015; older reports return empty.
    """
    # Normalize: ensure ' vp' suffix
    tunnus_vp = tunnus if tunnus.endswith(" vp") else f"{tunnus} vp"
    encoded = urllib.parse.quote(tunnus_vp)
    url = (f"https://avoindata.eduskunta.fi/api/v1/tables/VaskiData/rows"
           f"?columnName=Eduskuntatunnus&columnValue={encoded}&perPage=5")

    data = await fetch(client, "GET", url)
    if not data:
        return ""

    try:
        result = json.loads(data)
    except json.JSONDecodeError:
        return ""

    column_names = result.get("columnNames", [])
    raw_rows = result.get("rowData", [])
    if not raw_rows:
        return ""

    # Take the row with the largest XmlData (in case of duplicates)
    rows = [dict(zip(column_names, r)) for r in raw_rows]
    best = max(rows, key=lambda r: len(r.get("XmlData", "")))
    return best.get("XmlData", "")


def extract_vaski_structured_content(xml_str: str) -> str:
    """Extract structured paragraph-separated HTML from Vaski VN XML.

    Parses PerusteluOsa sections and preserves paragraph structure as
    <h3> headers and <p> tags. Returns structured HTML string.

    The VN XML schema uses:
      PerusteluOsa > OtsikkoTeksti — section title (e.g. "VALIOKUNNAN PERUSTELUT")
      PerusteluOsa > PerusteluLuku > ValiotsikkoTeksti — subsection headers
      PerusteluOsa > PerusteluLuku > KappaleKooste — paragraphs
    """
    if not xml_str:
        return ""

    try:
        tree = etree.fromstring(xml_str.encode("utf-8") if isinstance(xml_str, str) else xml_str)
    except etree.XMLSyntaxError:
        return ""

    parts = []

    # Also extract the JOHDANTO/expert list before perustelut
    # (useful context, but we mark it with a header)
    for el in tree.iter():
        tag = etree.QName(el).localname if isinstance(el.tag, str) else str(el.tag)

        if tag == "PerusteluOsa":
            # Get section title
            for child in el:
                child_tag = etree.QName(child).localname if isinstance(child.tag, str) else str(child.tag)
                if child_tag == "OtsikkoTeksti":
                    title = (child.text or "").strip()
                    if title:
                        parts.append(f"<h2>{title}</h2>")
                elif child_tag == "PerusteluLuku":
                    _extract_perusteluluku(child, parts)

        elif tag == "JohdantoOsa":
            parts.insert(0, "<h2>JOHDANTO</h2>")
            for child in el:
                child_tag = etree.QName(child).localname if isinstance(child.tag, str) else str(child.tag)
                if child_tag == "KappaleKooste":
                    text = _collect_text(child)
                    if text:
                        parts.insert(len([p for p in parts if p.startswith("<h2>JOHD")]) + 1,
                                     f"<p>{text}</p>")

        elif tag == "PasijOsa":
            # Päätösehdotus
            parts.append("<h2>PÄÄTÖSEHDOTUS</h2>")
            for child in el:
                child_tag = etree.QName(child).localname if isinstance(child.tag, str) else str(child.tag)
                if child_tag == "KappaleKooste":
                    text = _collect_text(child)
                    if text:
                        parts.append(f"<p>{text}</p>")

    return "\n".join(parts)


def _extract_perusteluluku(luku_el, parts: list):
    """Extract paragraphs and subheadings from a PerusteluLuku element."""
    for child in luku_el:
        tag = etree.QName(child).localname if isinstance(child.tag, str) else str(child.tag)
        if tag == "ValiotsikkoTeksti":
            text = (child.text or "").strip()
            if text:
                parts.append(f"<h3>{text}</h3>")
        elif tag == "KappaleKooste":
            text = _collect_text(child)
            if text:
                parts.append(f"<p>{text}</p>")
        elif tag == "SisennettyKappaleKooste":
            text = _collect_text(child)
            if text:
                parts.append(f"<p>{text}</p>")
        elif tag == "ListaKohtaKooste":
            text = _collect_text(child)
            if text:
                parts.append(f"<p>- {text}</p>")
        elif tag == "PerusteluLuku":
            # Nested chapter
            _extract_perusteluluku(child, parts)


def _collect_text(el) -> str:
    """Collect all text content from an element and its children."""
    texts = []
    if el.text:
        texts.append(el.text.strip())
    for child in el:
        # Include inline element text (references, emphasis, etc.)
        if child.text:
            texts.append(child.text.strip())
        if child.tail:
            texts.append(child.tail.strip())
    return " ".join(t for t in texts if t)



async def fetch_lakitutka(client: httpx.AsyncClient, year: int, number: int) -> dict:
    """Fetch expert statements and committee reports from Lakitutka ES API.

    Returns dict with keys: 'expert_statements', 'committee_reports'.
    Each is a list of parsed _source dicts.
    """
    he_str = f"HE {number}/{year}"
    he_str_vp = f"HE {number}/{year} vp"
    url = f"https://lakitutka.fi/api/search/all?term={urllib.parse.quote(he_str)}&sort=date_desc&lang=fi&size=200"

    data = await fetch(client, "GET", url)
    if not data:
        log.warning("LAKITUTKA: No response for %s", he_str)
        return {"expert_statements": [], "committee_reports": []}

    try:
        result = json.loads(data)
    except json.JSONDecodeError:
        log.warning("LAKITUTKA: Invalid JSON for %s", he_str)
        return {"expert_statements": [], "committee_reports": []}

    hits = result.get("hits", {}).get("hits", [])
    log.info("  LAKITUTKA: %d total hits for %s", len(hits), he_str)

    expert_statements = []
    committee_reports = []

    for hit in hits:
        idx = hit.get("_index", "")
        src = hit.get("_source", {})
        hit_id = hit.get("_id", "")

        if idx == "laki_vk":
            # Filter: liittyy or hanke must reference our HE
            liittyy = src.get("liittyy", "")
            hanke = src.get("hanke", "")
            if he_str in liittyy or he_str_vp in liittyy or hanke == he_str or hanke == he_str_vp:
                doc_type = src.get("type", "")
                if doc_type == "Asiantuntijalausunto":
                    expert_statements.append({"id": hit_id, **src})
                else:
                    committee_reports.append({"id": hit_id, **src})

    log.info("  LAKITUTKA: %d expert statements, %d committee reports for %s",
             len(expert_statements), len(committee_reports), he_str)
    return {
        "expert_statements": expert_statements,
        "committee_reports": committee_reports,
    }





# ---------------------------------------------------------------------------
# SQLite schema
# ---------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS he (
    canonical_id TEXT PRIMARY KEY,
    number INTEGER NOT NULL,
    year INTEGER NOT NULL,
    title TEXT,
    content TEXT,
    edilex_html TEXT,
    ministry TEXT,
    finlex_uri TEXT,
    eduskunta_tunnus TEXT,
    hankeikkuna_uuid TEXT,
    hankeikkuna_tunnukset TEXT,
    lausuntopalvelu_guid TEXT,
    lawsampo_uri TEXT,
    lan_consulted INTEGER,
    status TEXT,
    date_issued TEXT,
    laws_amended TEXT,
    hanke_metadata TEXT,
    last_indexed TEXT
);

CREATE TABLE IF NOT EXISTS committee_report (
    report_id TEXT PRIMARY KEY,
    he_id TEXT REFERENCES he(canonical_id),
    committee TEXT,
    report_type TEXT,
    title TEXT,
    tunnus TEXT,
    content TEXT,
    edilex_html TEXT,
    eduskunta_vaski_id TEXT,
    lakitutka_id TEXT,
    date TEXT,
    vaski_xml TEXT
);

CREATE TABLE IF NOT EXISTS expert_statement (
    statement_id TEXT PRIMARY KEY,
    he_id TEXT REFERENCES he(canonical_id),
    committee TEXT,
    expert_title TEXT,
    date TEXT,
    content TEXT,
    lakitutka_id TEXT,
    liittyy TEXT
);
"""


def migrate_db(conn: sqlite3.Connection):
    """Add columns to existing tables if missing."""
    # Check expert_statement
    cursor = conn.execute("PRAGMA table_info(expert_statement)")
    cols = [row[1] for row in cursor.fetchall()]
    if "content" not in cols:
        log.info("MIGRATION: Adding content column to expert_statement")
        conn.execute("ALTER TABLE expert_statement ADD COLUMN content TEXT")

    # Check he
    cursor = conn.execute("PRAGMA table_info(he)")
    cols = [row[1] for row in cursor.fetchall()]
    if "content" not in cols:
        log.info("MIGRATION: Adding content column to he")
        conn.execute("ALTER TABLE he ADD COLUMN content TEXT")
    if "edilex_html" not in cols:
        log.info("MIGRATION: Adding edilex_html column to he")
        conn.execute("ALTER TABLE he ADD COLUMN edilex_html TEXT")

    # Check committee_report
    cursor = conn.execute("PRAGMA table_info(committee_report)")
    cols = [row[1] for row in cursor.fetchall()]
    for col_name, col_type in [("content", "TEXT"), ("edilex_html", "TEXT"),
                                ("lakitutka_id", "TEXT"), ("tunnus", "TEXT"),
                                ("vaski_xml", "TEXT")]:
        if col_name not in cols:
            log.info("MIGRATION: Adding %s column to committee_report", col_name)
            conn.execute(f"ALTER TABLE committee_report ADD COLUMN {col_name} {col_type}")


def init_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    migrate_db(conn)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


# ---------------------------------------------------------------------------
# Upsert logic
# ---------------------------------------------------------------------------

async def upsert_he(
    conn: sqlite3.Connection,
    year: int,
    number: int,
    hanke: dict | None,
    finlex_xml: bytes,
    vaski_docs: list[dict],
    lakitutka: dict | None = None,
    edilex_html: str = "",
):
    canonical_id = f"he-{number}-{year}"

    # Parse Finlex
    finlex_meta = parse_finlex_metadata(finlex_xml)
    finlex_refs = parse_finlex_refs(finlex_xml)

    # Parse Hankeikkuna
    hanke_meta = extract_hanke_metadata(hanke)
    lausunto_guid = extract_lausunto_guid(hanke)

    # Build laws_amended list (exclude HE cross-refs, keep only statute refs)
    laws_amended = [r["law_id"] for r in finlex_refs if not r.get("is_he_ref")]

    # Upsert HE row
    conn.execute("""
        INSERT OR REPLACE INTO he
        (canonical_id, number, year, title, content, edilex_html, ministry, finlex_uri, eduskunta_tunnus,
         hankeikkuna_uuid, hankeikkuna_tunnukset, lausuntopalvelu_guid, lan_consulted,
         status, laws_amended, hanke_metadata, last_indexed)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        canonical_id, number, year,
        finlex_meta["title"] or hanke_meta.get("nimi"),
        finlex_xml.decode("utf-8") if finlex_xml else None,
        edilex_html or None,
        finlex_meta["ministry"] or hanke_meta.get("vastuuministeri"),
        f"/akn/fi/doc/government-proposal/{year}/{number}/fin@",
        f"HE {number}/{year} vp",
        hanke_meta.get("uuid"),
        json.dumps([hanke_meta["tunnus"]] if hanke_meta.get("tunnus") else []),
        lausunto_guid,
        1 if hanke_meta.get("lan_consulted") else (0 if hanke_meta.get("lan_consulted") is False else None),
        hanke_meta.get("tila"),
        json.dumps(laws_amended),
        json.dumps(hanke_meta) if hanke_meta else None,
        time.strftime("%Y-%m-%d"),
    ))

    # Upsert committee reports from VaskiData
    for doc in vaski_docs:
        doc_type = doc.get("doc_type", "")
        committee = doc.get("committee", "")
        if doc_type or committee:
            report_id = doc.get("vaski_id") or f"{canonical_id}_{doc.get('eduskuntatunnus', 'unknown')}"
            conn.execute("""
                INSERT OR REPLACE INTO committee_report
                (report_id, he_id, committee, report_type, title, eduskunta_vaski_id, date)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                report_id, canonical_id, committee, doc_type,
                doc.get("title"), doc.get("vaski_id"), doc.get("date"),
            ))

    # Upsert Lakitutka data
    if lakitutka:
        # Expert statements
        for stmt in lakitutka.get("expert_statements", []):
            stmt_id = stmt.get("id", "")
            if not stmt_id:
                continue
            conn.execute("""
                INSERT OR REPLACE INTO expert_statement
                (statement_id, he_id, committee, expert_title, date, content, lakitutka_id, liittyy)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                stmt_id, canonical_id,
                stmt.get("valiokunta"),
                stmt.get("title"),
                stmt.get("laadittu"),
                stmt.get("content"),
                stmt_id,
                stmt.get("liittyy"),
            ))

        # Committee reports (mietinnöt + lausunnot) from Lakitutka + Vaski
        for report in lakitutka.get("committee_reports", []):
            report_id = report.get("id", "")
            if not report_id:
                continue
            tunnus = report.get("tunnus", "")
            conn.execute("""
                INSERT OR REPLACE INTO committee_report
                (report_id, he_id, committee, report_type, title, tunnus,
                 content, edilex_html, lakitutka_id, date, vaski_xml)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                report_id, canonical_id,
                report.get("valiokunta"),
                report.get("type"),
                report.get("title"),
                tunnus,
                report.get("content"),
                report.get("edilex_html"),
                report_id,
                report.get("laadittu"),
                report.get("vaski_xml"),
            ))

    conn.commit()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

async def fetch_lakitutka_timeline(client: httpx.AsyncClient, he_id: str) -> list:
    """Fetch timeline of all documents for an HE from Lakitutka.
    
    he_id example: 'he38-2025'
    """
    url = f"https://lakitutka.fi/api/timeline/hallituksen_esitykset/{he_id}"
    data = await fetch(client, "GET", url)
    if not data:
        return []
    try:
        return json.loads(data)
    except json.JSONDecodeError:
        return []


async def fetch_lakitutka_content(client: httpx.AsyncClient, doc_id: str, index: str) -> str:
    """Fetch full content for a Lakitutka document using its index-specific group."""
    # Map index names to api/docs groups
    group_map = {
        "laki_he": "hallituksen_esitykset",
        "laki_puheenvuoro": "poytakirja",
        "laki_vk": "he_asiantuntijalausunnot",  # Fallback
        "laki_lausuntokierros_asiakirjat": "lausuntokierros_asiakirjat"
    }
    group = group_map.get(index, "all")
    
    url = f"https://lakitutka.fi/api/docs/{group}/{doc_id}?lang=fi"
    data = await fetch(client, "GET", url)
    if not data:
        return ""
    try:
        res = json.loads(data)
        doc_src = res.get("doc", {}).get("_source", {})
        # Different fields depending on index
        content = doc_src.get("sisalto") or doc_src.get("content") or doc_src.get("xml_data") or ""
        return content
    except json.JSONDecodeError:
        return ""




async def process_he(client: httpx.AsyncClient, conn: sqlite3.Connection, year: int, number: int):
    """Process a single HE through the full query cascade."""
    canonical_id = f"he-{number}-{year}"
    he_str = f"HE {number}/{year}"
    log.info("=" * 60)
    log.info("Processing %s (%s)", he_str, canonical_id)
    log.info("=" * 60)

    # Tier 1: parallel independent fetches
    hanke_task = fetch_hankeikkuna(client, year, number)
    finlex_task = fetch_finlex(client, year, number)
    eduskunta_task = fetch_eduskunta(client, year, number)
    lakitutka_task = fetch_lakitutka(client, year, number)

    # Lakitutka timeline (needs he_id format)
    lakitutka_id = f"he{number}-{year}"
    timeline_task = fetch_lakitutka_timeline(client, lakitutka_id)

    hanke, finlex_xml, eduskunta_rows, lakitutka, timeline = await asyncio.gather(
        hanke_task, finlex_task, eduskunta_task, lakitutka_task, timeline_task,
        return_exceptions=True,
    )

    # Handle exceptions from gather
    if isinstance(hanke, Exception):
        log.error("  Hankeikkuna failed: %s", hanke)
        hanke = None
    if isinstance(finlex_xml, Exception):
        log.error("  Finlex failed: %s", finlex_xml)
        finlex_xml = b""
    if isinstance(eduskunta_rows, Exception):
        log.error("  Eduskunta failed: %s", eduskunta_rows)
        eduskunta_rows = []
    if isinstance(lakitutka, Exception):
        log.error("  Lakitutka failed: %s", lakitutka)
        lakitutka = None
    if isinstance(timeline, Exception):
        log.error("  Lakitutka timeline failed: %s", timeline)
        timeline = []

    # Parse VaskiData XML blobs
    vaski_docs = parse_vaski_xml(eduskunta_rows) if eduskunta_rows else []

    # Fetch full content for Lakitutka hits (parallelized)
    if isinstance(lakitutka, dict):
        expert_tasks = []
        for s in lakitutka.get("expert_statements", []):
            existing = conn.execute("SELECT content FROM expert_statement WHERE statement_id = ?", (s["id"],)).fetchone()
            if existing and existing["content"] and existing["content"].strip():
                expert_tasks.append(asyncio.Future())
                expert_tasks[-1].set_result(existing["content"])
            else:
                expert_tasks.append(fetch_lakitutka_content(client, s["id"], "laki_vk"))

        # Fetch committee report content from Lakitutka + Vaski XML
        committee_reports = lakitutka.get("committee_reports", [])
        committee_lakitutka_tasks = []
        committee_vaski_tasks = []
        for cr in committee_reports:
            tunnus = cr.get("tunnus", "")
            # Check existing
            existing = conn.execute("SELECT content, edilex_html, vaski_xml FROM committee_report WHERE report_id = ?", (cr["id"],)).fetchone()

            # Lakitutka content task (fallback if Vaski unavailable)
            if existing and existing["content"] and existing["content"].strip():
                t1 = asyncio.Future()
                t1.set_result(existing["content"])
            else:
                t1 = fetch_lakitutka_content(client, cr["id"], "laki_vk")
            committee_lakitutka_tasks.append(t1)

            # Vaski XML task — structured source (available ~2015+)
            if existing and existing["vaski_xml"] and len(existing["vaski_xml"]) > 100:
                t2 = asyncio.Future()
                t2.set_result(existing["vaski_xml"])
            elif tunnus:
                t2 = fetch_vaski_by_tunnus(client, tunnus)
            else:
                t2 = asyncio.Future()
                t2.set_result("")
            committee_vaski_tasks.append(t2)

        all_content = await asyncio.gather(
            *(expert_tasks + committee_lakitutka_tasks + committee_vaski_tasks))

        # Distribute back
        n_experts = len(expert_tasks)
        n_committee = len(committee_lakitutka_tasks)

        expert_contents = all_content[:n_experts]
        committee_lt_contents = all_content[n_experts:n_experts + n_committee]
        committee_vaski_contents = all_content[n_experts + n_committee:]

        for i, content in enumerate(expert_contents):
            if content: lakitutka["expert_statements"][i]["content"] = content
        for i, cr in enumerate(committee_reports):
            vaski_xml = committee_vaski_contents[i] if i < len(committee_vaski_contents) else ""
            if vaski_xml:
                cr["vaski_xml"] = vaski_xml
                # Extract structured content from Vaski XML — replaces degraded Lakitutka content
                structured = extract_vaski_structured_content(vaski_xml)
                if structured:
                    cr["content"] = structured
                    log.info("  VASKI: Structured content for %s (%d chars)", cr.get("tunnus", "?"), len(structured))
                else:
                    # Vaski XML exists but extraction failed — fall back to Lakitutka
                    lt_content = committee_lt_contents[i] if i < len(committee_lt_contents) else ""
                    if lt_content:
                        cr["content"] = lt_content
            else:
                # No Vaski XML — use Lakitutka content
                lt_content = committee_lt_contents[i] if i < len(committee_lt_contents) else ""
                if lt_content:
                    cr["content"] = lt_content

    # Upsert everything into SQLite
    try:
        await upsert_he(
            conn, year, number,
            hanke, finlex_xml, vaski_docs,
            lakitutka=lakitutka if isinstance(lakitutka, dict) else None,
        )
    except sqlite3.Error as e:
        log.critical("  FATAL SQL ERROR: %s", e)
        raise

    # Summary
    finlex_refs = parse_finlex_refs(finlex_xml)
    lt_experts = len((lakitutka or {}).get("expert_statements", [])) if isinstance(lakitutka, dict) else 0
    lt_committee = len((lakitutka or {}).get("committee_reports", [])) if isinstance(lakitutka, dict) else 0
    log.info("  DONE: %s — %d law refs, %d vaski docs, %d expert stmts, %d committee reports",
             he_str, len(finlex_refs), len(vaski_docs), lt_experts, lt_committee)



async def main():
    parser = argparse.ArgumentParser(description="Build Finnish legislative index")
    parser.add_argument("he_ids", nargs="*", help="HE IDs to process (e.g., '38/2025 157/2024')")
    parser.add_argument("--he", help="Process single HE (e.g., '38/2025') [deprecated: use positional args]")
    parser.add_argument("--all-extant", action="store_true", help="Process all HEs already in the database")
    parser.add_argument("--clear-cache", action="store_true", help="Clear API cache before running")
    parser.add_argument(
        "--cache-archive",
        type=Path,
        help="override farchive cache path for testing or one-off migration",
    )
    parser.add_argument(
        "--migrate-cache-to-farchive",
        action="store_true",
        help="Import existing loose .cache/megadoc files into farchive and exit without fetching",
    )
    parser.add_argument(
        "--migration-limit",
        type=int,
        default=None,
        metavar="N",
        help="debug: import at most N loose cache files when using --migrate-cache-to-farchive",
    )
    args = parser.parse_args()

    global CACHE_ARCHIVE
    if args.cache_archive:
        CACHE_ARCHIVE = args.cache_archive

    if args.migrate_cache_to_farchive:
        stats = migrate_legacy_cache_to_farchive(limit=args.migration_limit)
        log.info(
            "CACHE MIGRATION: seen=%d imported=%d same_head=%d missing=%d bytes_imported=%d errors=%d archive=%s",
            stats.files_seen,
            stats.files_imported,
            stats.files_same_head,
            stats.files_missing,
            stats.bytes_imported,
            stats.errors,
            CACHE_ARCHIVE,
        )
        if stats.errors:
            sys.exit(1)
        return

    if args.clear_cache:
        if CACHE_DIR.exists():
            shutil.rmtree(CACHE_DIR)
            log.info("Cache cleared: %s", CACHE_DIR)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        log.info("Farchive cache preserved: %s", CACHE_ARCHIVE)

    conn = init_db()
    log.info("Database: %s", DB_PATH)
    log.info("Cache: farchive=%s legacy=%s", CACHE_ARCHIVE, CACHE_DIR)

    async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
        # Build list of HEs to process
        he_list: list[tuple[int, int]] = []  # (year, number)

        if args.he:
            # --he flag (single HE, backwards compatible)
            parts = args.he.split("/")
            if len(parts) != 2:
                print(f"Invalid HE format: {args.he}. Use 'number/year'")
                sys.exit(1)
            he_list = [(int(parts[1]), int(parts[0]))]
        elif args.he_ids:
            # Positional args: multiple HEs
            for hid in args.he_ids:
                parts = hid.split("/")
                if len(parts) != 2:
                    print(f"Invalid HE format: {hid}. Use 'number/year' (e.g., '38/2025')")
                    sys.exit(1)
                he_list.append((int(parts[1]), int(parts[0])))
            log.info("Processing %d HEs from command line", len(he_list))
        elif args.all_extant:
            rows = conn.execute("SELECT number, year FROM he").fetchall()
            log.info("Processing %d extant HEs from database", len(rows))
            he_list = [(row["year"], row["number"]) for row in rows]

        for year, number in he_list:
            try:
                await process_he(client, conn, year, number)
            except sqlite3.Error:
                log.critical("Aborting due to fatal SQL error.")
                sys.exit(1)
            except Exception as e:
                log.error("Error for HE %d/%d: %s", number, year, e)

    local_finlex.close()

    # Print summary
    log.info("\n" + "=" * 60)
    log.info("INDEX SUMMARY")
    log.info("=" * 60)

    for table in ["he", "committee_report", "expert_statement"]:
        count = conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        log.info("  %-20s %d rows", table, count)

    # Per-HE summary
    log.info("\nPer-HE breakdown:")
    for row in conn.execute("SELECT canonical_id, title, hankeikkuna_uuid, lan_consulted, laws_amended, last_indexed FROM he ORDER BY canonical_id"):
        cid, title, huuid, lan, laws, indexed = row
        laws_list = json.loads(laws) if laws else []
        log.info("  %s: %s", cid, (title or "?")[:60])
        idx_str = indexed or "unknown date"
        h_str = "YES" if huuid else f"NO ({idx_str})"
        lan_str = {1: "YES", 0: f"NO ({idx_str})", None: f"? ({idx_str})"}.get(lan, f"? ({idx_str})")
        log.info("    Hankeikkuna: %s | LAN: %s | Laws: %d", h_str, lan_str, len(laws_list))

    conn.close()
    log.info("\nDone. Database at: %s", DB_PATH)


if __name__ == "__main__":
    asyncio.run(main())

"""Auxiliary evidence source mirror for MeV.

This is an L0.5 raw-source layer: it fetches source bytes, stores plaintext-ish
observations in farchive, and records retrieval metadata in SQLite. It does not
interpret holdings, compile doctrine, or decide whether a source discharges a
MeVM finding.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from farchive import CompressionPolicy, Farchive

from mev.config import ROOT


USER_AGENT = "MeV-AuxEvidenceMirror/0.1 (research; mekanismirealismi.fi/mev)"
DEFAULT_ARCHIVE = ROOT / "data" / "aux_sources" / "raw.farchive"
DEFAULT_INDEX = ROOT / "data" / "aux_sources" / "index.sqlite"


class SourceTier(StrEnum):
    CORE = "core"
    TIER1 = "tier1"
    TIER2 = "tier2"
    TIER3 = "tier3"


class AcquisitionMethod(StrEnum):
    API = "api"
    BULK = "bulk"
    HTML = "html"
    OAI = "oai"
    RSS = "rss"
    PDF = "pdf"
    CATALOGUE = "catalogue"


class MediaFamily(StrEnum):
    JSON = "json"
    XML = "xml"
    HTML = "html"
    TEXT = "text"
    CSV = "csv"
    MARKDOWN = "markdown"
    PDF = "pdf"
    BINARY = "binary"
    UNKNOWN = "unknown"


class StorageDecision(StrEnum):
    STORED = "stored"
    SKIPPED_NON_TEXT = "skipped_non_text"
    FETCH_ERROR = "fetch_error"


@dataclass(frozen=True, slots=True)
class AuxEndpoint:
    endpoint_id: str
    source_id: str
    source_kind: str
    tier: SourceTier
    role: str
    url: str
    method: AcquisitionMethod
    expected_family: MediaFamily

    def locator(self) -> str:
        return f"aux-source://{self.source_id}/{self.endpoint_id}"


@dataclass(frozen=True, slots=True)
class FetchRecord:
    endpoint_id: str
    source_id: str
    url: str
    locator: str
    retrieved_at: str
    status_code: int | None
    content_type: str
    media_family: MediaFamily
    size_bytes: int
    sha256: str
    storage_decision: StorageDecision
    error: str = ""

    def to_row(self) -> tuple:
        return (
            self.endpoint_id,
            self.source_id,
            self.url,
            self.locator,
            self.retrieved_at,
            self.status_code,
            self.content_type,
            self.media_family.value,
            self.size_bytes,
            self.sha256,
            self.storage_decision.value,
            self.error,
        )


SOURCE_CATALOG: tuple[AuxEndpoint, ...] = (
    AuxEndpoint(
        "eduskunta_openapi",
        "eduskunta",
        "parliamentary_documents",
        SourceTier.TIER1,
        "OpenAPI/ReDoc seed for HE, PeVL, committee and expert-statement discovery.",
        "https://avoindata.eduskunta.fi/swagger/apidocs.html",
        AcquisitionMethod.HTML,
        MediaFamily.HTML,
    ),
    AuxEndpoint(
        "eduskunta_open_data_info",
        "eduskunta",
        "parliamentary_documents",
        SourceTier.TIER1,
        "Official open-data description.",
        "https://www.parliament.fi/fi/avoin-data/mita-on-avoin-data",
        AcquisitionMethod.HTML,
        MediaFamily.HTML,
    ),
    AuxEndpoint(
        "hankeikkuna_api_landing",
        "hankeikkuna",
        "project_spine",
        SourceTier.TIER1,
        "Public API landing page.",
        "https://api.hankeikkuna.fi/",
        AcquisitionMethod.HTML,
        MediaFamily.HTML,
    ),
    AuxEndpoint(
        "hankeikkuna_api_doc",
        "hankeikkuna",
        "project_spine",
        SourceTier.TIER1,
        "API documentation seed.",
        "https://api.hankeikkuna.fi/doc",
        AcquisitionMethod.HTML,
        MediaFamily.HTML,
    ),
    AuxEndpoint(
        "hankeikkuna_swagger",
        "hankeikkuna",
        "project_spine",
        SourceTier.TIER1,
        "Swagger/API entrypoint seed.",
        "https://api.hankeikkuna.fi/api",
        AcquisitionMethod.HTML,
        MediaFamily.HTML,
    ),
    AuxEndpoint(
        "lausuntopalvelu_api_root",
        "lausuntopalvelu",
        "consultation",
        SourceTier.TIER1,
        "OData API root.",
        "https://www.lausuntopalvelu.fi/api/v1/Lausuntopalvelu.svc/",
        AcquisitionMethod.API,
        MediaFamily.XML,
    ),
    AuxEndpoint(
        "lausuntopalvelu_proposals",
        "lausuntopalvelu",
        "consultation",
        SourceTier.TIER1,
        "Proposal collection seed.",
        "https://www.lausuntopalvelu.fi/api/v1/Lausuntopalvelu.svc/Proposals",
        AcquisitionMethod.API,
        MediaFamily.XML,
    ),
    AuxEndpoint(
        "lausuntopalvelu_list",
        "lausuntopalvelu",
        "consultation",
        SourceTier.TIER1,
        "Human-visible proposal list fallback.",
        "https://www.lausuntopalvelu.fi/FI/Proposal/List",
        AcquisitionMethod.HTML,
        MediaFamily.HTML,
    ),
    AuxEndpoint(
        "eoa_decisions_landing",
        "eoa",
        "oversight_praxis",
        SourceTier.TIER1,
        "EOA decisions landing/list page.",
        "https://oikeusasiamies.fi/ratkaisut",
        AcquisitionMethod.HTML,
        MediaFamily.HTML,
    ),
    AuxEndpoint(
        "eoa_statements_landing",
        "eoa",
        "oversight_praxis",
        SourceTier.TIER1,
        "EOA statements landing/list page.",
        "https://oikeusasiamies.fi/lausunnot",
        AcquisitionMethod.HTML,
        MediaFamily.HTML,
    ),
    AuxEndpoint(
        "okv_finlex_index",
        "okv",
        "legality_supervision",
        SourceTier.TIER1,
        "Finlex-hosted OKV corpus index.",
        "https://www.finlex.fi/fi/viranomaiset/oikeuskansleri",
        AcquisitionMethod.HTML,
        MediaFamily.HTML,
    ),
    AuxEndpoint(
        "okv_search",
        "okv",
        "legality_supervision",
        SourceTier.TIER1,
        "OKV own decision/statement search.",
        "https://oikeuskansleri.fi/ratkaisu-ja-lausuntohaku",
        AcquisitionMethod.HTML,
        MediaFamily.HTML,
    ),
    AuxEndpoint(
        "vtv_reports",
        "vtv",
        "audit_evidence",
        SourceTier.TIER1,
        "VTV report listing.",
        "https://vtv.fi/raportit/",
        AcquisitionMethod.HTML,
        MediaFamily.HTML,
    ),
    AuxEndpoint(
        "vtv_wp_json_probe",
        "vtv",
        "audit_evidence",
        SourceTier.TIER1,
        "WordPress REST probe; useful only if available.",
        "https://vtv.fi/wp-json/",
        AcquisitionMethod.API,
        MediaFamily.JSON,
    ),
    AuxEndpoint(
        "valto_oai_identify",
        "valto",
        "ministry_publications",
        SourceTier.TIER1,
        "DSpace OAI-PMH identify probe.",
        "https://julkaisut.valtioneuvosto.fi/server/oai/request?verb=Identify",
        AcquisitionMethod.OAI,
        MediaFamily.XML,
    ),
    AuxEndpoint(
        "valto_oai_formats",
        "valto",
        "ministry_publications",
        SourceTier.TIER1,
        "DSpace OAI-PMH metadata formats.",
        "https://julkaisut.valtioneuvosto.fi/server/oai/request?verb=ListMetadataFormats",
        AcquisitionMethod.OAI,
        MediaFamily.XML,
    ),
    AuxEndpoint(
        "finlex_case_law_index",
        "finlex_case_law",
        "court_praxis",
        SourceTier.TIER1,
        "Finlex case-law index for KHO/KKO/MAO/HAO/etc.",
        "https://www.finlex.fi/fi/oikeuskaytanto",
        AcquisitionMethod.HTML,
        MediaFamily.HTML,
    ),
    AuxEndpoint(
        "court_decisions_latest",
        "courts",
        "court_praxis",
        SourceTier.TIER1,
        "Court-system latest decisions page.",
        "https://www.tuomioistuimet.fi/tuomioistuinten-ratkaisut/",
        AcquisitionMethod.HTML,
        MediaFamily.HTML,
    ),
    AuxEndpoint(
        "eurlex_reuse",
        "eurlex",
        "eu_doctrine_cluster",
        SourceTier.TIER2,
        "EUR-Lex reuse guidance; selective EU mirror only.",
        "https://eur-lex.europa.eu/content/help/data-reuse/reuse-contents-eurlex-details.html",
        AcquisitionMethod.HTML,
        MediaFamily.HTML,
    ),
    AuxEndpoint(
        "hudoc_database",
        "hudoc",
        "echr_doctrine_cluster",
        SourceTier.TIER2,
        "HUDOC database guidance; selective ECHR mirror only.",
        "https://www.echr.coe.int/hudoc-database",
        AcquisitionMethod.HTML,
        MediaFamily.HTML,
    ),
    AuxEndpoint(
        "statfin_open_data",
        "statfin",
        "statistics_context",
        SourceTier.TIER2,
        "Statistics Finland open-data/API guidance.",
        "https://stat.fi/fi/palvelut/tilastodatapalvelut/avoin-data-ja-rajapinnat",
        AcquisitionMethod.HTML,
        MediaFamily.HTML,
    ),
    AuxEndpoint(
        "sotkanet_landing",
        "sotkanet",
        "health_social_statistics",
        SourceTier.TIER2,
        "Sotkanet landing page.",
        "https://sotkanet.fi/",
        AcquisitionMethod.HTML,
        MediaFamily.HTML,
    ),
    AuxEndpoint(
        "tutkihallintoa_apis",
        "tutkihallintoa",
        "public_finance_context",
        SourceTier.TIER2,
        "Tutkihallintoa API guidance.",
        "https://www.tutkihallintoa.fi/avoimet-rajapinnat/",
        AcquisitionMethod.HTML,
        MediaFamily.HTML,
    ),
    AuxEndpoint(
        "avoindata_package_search_probe",
        "avoindata",
        "open_data_catalogue",
        SourceTier.TIER2,
        "CKAN package-search probe for dataset discovery.",
        "https://www.avoindata.fi/data/api/3/action/package_search?q=tuomioistuimet",
        AcquisitionMethod.API,
        MediaFamily.JSON,
    ),
)


def endpoints_by_id() -> dict[str, AuxEndpoint]:
    return {endpoint.endpoint_id: endpoint for endpoint in SOURCE_CATALOG}


def endpoints_for_source(source_id: str) -> tuple[AuxEndpoint, ...]:
    return tuple(endpoint for endpoint in SOURCE_CATALOG if endpoint.source_id == source_id)


def classify_media(content_type: str, url: str, body: bytes = b"") -> MediaFamily:
    lowered = content_type.lower().split(";", 1)[0].strip()
    suffix = Path(url.split("?", 1)[0]).suffix.lower()
    if lowered in ("application/json", "application/ld+json") or suffix == ".json":
        return MediaFamily.JSON
    if lowered in ("application/xml", "text/xml", "application/atom+xml") or suffix in (".xml", ".rdf"):
        return MediaFamily.XML
    if lowered in ("text/html", "application/xhtml+xml") or suffix in (".html", ".htm"):
        return MediaFamily.HTML
    if lowered in ("text/csv", "application/csv") or suffix == ".csv":
        return MediaFamily.CSV
    if lowered in ("text/markdown", "text/x-markdown") or suffix in (".md", ".markdown"):
        return MediaFamily.MARKDOWN
    if lowered == "application/pdf" or suffix == ".pdf" or body.startswith(b"%PDF"):
        return MediaFamily.PDF
    if lowered.startswith("text/") or suffix == ".txt":
        return MediaFamily.TEXT
    if lowered:
        return MediaFamily.BINARY
    return MediaFamily.UNKNOWN


def is_plaintextish(media_family: MediaFamily) -> bool:
    return media_family in {
        MediaFamily.JSON,
        MediaFamily.XML,
        MediaFamily.HTML,
        MediaFamily.TEXT,
        MediaFamily.CSV,
        MediaFamily.MARKDOWN,
    }


def _open_archive(path: Path) -> Farchive:
    return Farchive(path, compression=CompressionPolicy(delta_enabled=False))


def init_index(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS aux_source_observation (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            endpoint_id TEXT NOT NULL,
            source_id TEXT NOT NULL,
            url TEXT NOT NULL,
            locator TEXT NOT NULL,
            retrieved_at TEXT NOT NULL,
            status_code INTEGER,
            content_type TEXT NOT NULL,
            media_family TEXT NOT NULL,
            size_bytes INTEGER NOT NULL,
            sha256 TEXT NOT NULL,
            storage_decision TEXT NOT NULL,
            error TEXT NOT NULL
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_aux_source_obs_source ON aux_source_observation(source_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_aux_source_obs_endpoint ON aux_source_observation(endpoint_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_aux_source_obs_decision ON aux_source_observation(storage_decision)")
    return conn


def record_observation(conn: sqlite3.Connection, record: FetchRecord) -> None:
    conn.execute(
        """
        INSERT INTO aux_source_observation (
            endpoint_id, source_id, url, locator, retrieved_at, status_code,
            content_type, media_family, size_bytes, sha256, storage_decision, error
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        record.to_row(),
    )
    conn.commit()


def fetch_endpoint(
    endpoint: AuxEndpoint,
    *,
    archive_path: Path = DEFAULT_ARCHIVE,
    index_path: Path = DEFAULT_INDEX,
    include_binary: bool = False,
    timeout: float = 30.0,
) -> FetchRecord:
    retrieved_at = datetime.now(timezone.utc).isoformat()
    headers = {"User-Agent": USER_AGENT, "Accept": "*/*"}
    request = Request(endpoint.url, headers=headers)
    conn = init_index(index_path)
    try:
        with urlopen(request, timeout=timeout) as response:
            data = response.read()
            status_code = getattr(response, "status", None) or response.getcode()
            content_type = response.headers.get("Content-Type", "")
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        record = FetchRecord(
            endpoint_id=endpoint.endpoint_id,
            source_id=endpoint.source_id,
            url=endpoint.url,
            locator=endpoint.locator(),
            retrieved_at=retrieved_at,
            status_code=getattr(exc, "code", None),
            content_type="",
            media_family=MediaFamily.UNKNOWN,
            size_bytes=0,
            sha256="",
            storage_decision=StorageDecision.FETCH_ERROR,
            error=str(exc),
        )
        record_observation(conn, record)
        conn.close()
        return record

    digest = hashlib.sha256(data).hexdigest()
    media_family = classify_media(content_type, endpoint.url, data[:8])
    decision = StorageDecision.STORED if (include_binary or is_plaintextish(media_family)) else StorageDecision.SKIPPED_NON_TEXT
    if decision is StorageDecision.STORED:
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "source_id": endpoint.source_id,
            "endpoint_id": endpoint.endpoint_id,
            "source_kind": endpoint.source_kind,
            "tier": endpoint.tier.value,
            "role": endpoint.role,
            "url": endpoint.url,
            "retrieved_at": retrieved_at,
            "status_code": status_code,
            "content_type": content_type,
            "media_family": media_family.value,
            "sha256": digest,
            "size_bytes": len(data),
        }
        with _open_archive(archive_path) as archive:
            archive.store(
                endpoint.locator(),
                data,
                storage_class=media_family.value,
                series_key=f"aux-source://{endpoint.source_id}/{media_family.value}",
                metadata=metadata,
            )

    record = FetchRecord(
        endpoint_id=endpoint.endpoint_id,
        source_id=endpoint.source_id,
        url=endpoint.url,
        locator=endpoint.locator(),
        retrieved_at=retrieved_at,
        status_code=status_code,
        content_type=content_type,
        media_family=media_family,
        size_bytes=len(data),
        sha256=digest,
        storage_decision=decision,
    )
    record_observation(conn, record)
    conn.close()
    return record


def load_observations(index_path: Path = DEFAULT_INDEX) -> list[dict[str, object]]:
    if not index_path.exists():
        return []
    conn = sqlite3.connect(index_path)
    conn.row_factory = sqlite3.Row
    rows = [dict(row) for row in conn.execute("SELECT * FROM aux_source_observation ORDER BY id")]
    conn.close()
    return rows


def render_catalog_json() -> str:
    return json.dumps([asdict(endpoint) for endpoint in SOURCE_CATALOG], ensure_ascii=False, indent=2, default=str) + "\n"


def render_catalog_markdown() -> str:
    lines = ["# Aux Source Catalogue", ""]
    for endpoint in SOURCE_CATALOG:
        lines.append(
            f"- `{endpoint.endpoint_id}` `{endpoint.source_id}` `{endpoint.tier.value}` "
            f"`{endpoint.expected_family.value}` — {endpoint.role}"
        )
        lines.append(f"  URL: {endpoint.url}")
    return "\n".join(lines) + "\n"


def render_status_json(index_path: Path = DEFAULT_INDEX) -> str:
    return json.dumps(load_observations(index_path), ensure_ascii=False, indent=2) + "\n"


def render_status_markdown(index_path: Path = DEFAULT_INDEX) -> str:
    rows = load_observations(index_path)
    lines = ["# Aux Source Mirror Status", ""]
    if not rows:
        lines.append("(no observations)")
        return "\n".join(lines) + "\n"
    lines.append("| source | endpoint | decision | media | bytes | status |")
    lines.append("|---|---|---:|---|---:|---:|")
    for row in rows:
        lines.append(
            f"| `{row['source_id']}` | `{row['endpoint_id']}` | `{row['storage_decision']}` | "
            f"`{row['media_family']}` | {row['size_bytes']} | {row['status_code'] or ''} |"
        )
    return "\n".join(lines) + "\n"


def _selected_endpoints(source: str | None, endpoint: str | None, limit: int | None) -> tuple[AuxEndpoint, ...]:
    if endpoint:
        by_id = endpoints_by_id()
        if endpoint not in by_id:
            raise SystemExit(f"unknown endpoint_id: {endpoint}")
        selected = (by_id[endpoint],)
    elif source:
        selected = endpoints_for_source(source)
        if not selected:
            raise SystemExit(f"unknown source_id or no endpoints: {source}")
    else:
        selected = SOURCE_CATALOG
    return selected[:limit] if limit is not None else selected


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch/index auxiliary evidence source seeds")
    sub = parser.add_subparsers(dest="command", required=True)

    catalog = sub.add_parser("catalog", help="print the curated endpoint catalogue")
    catalog.add_argument("--format", choices=("markdown", "json"), default="markdown")

    fetch = sub.add_parser("fetch", help="fetch selected source endpoints into local farchive")
    fetch.add_argument("--source", help="source_id to fetch")
    fetch.add_argument("--endpoint", help="single endpoint_id to fetch")
    fetch.add_argument("--limit", type=int, help="maximum endpoints to fetch")
    fetch.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    fetch.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    fetch.add_argument("--include-binary", action="store_true", help="store PDFs/binary responses too")
    fetch.add_argument("--timeout", type=float, default=30.0)
    fetch.add_argument("--format", choices=("markdown", "json"), default="markdown")

    status = sub.add_parser("status", help="print local mirror observation status")
    status.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    status.add_argument("--format", choices=("markdown", "json"), default="markdown")

    args = parser.parse_args(argv)
    if args.command == "catalog":
        sys.stdout.write(render_catalog_json() if args.format == "json" else render_catalog_markdown())
        return 0
    if args.command == "status":
        sys.stdout.write(render_status_json(args.index) if args.format == "json" else render_status_markdown(args.index))
        return 0
    if args.command == "fetch":
        records = [
            fetch_endpoint(
                endpoint,
                archive_path=args.archive,
                index_path=args.index,
                include_binary=args.include_binary,
                timeout=args.timeout,
            )
            for endpoint in _selected_endpoints(args.source, args.endpoint, args.limit)
        ]
        if args.format == "json":
            sys.stdout.write(json.dumps([asdict(record) for record in records], ensure_ascii=False, indent=2, default=str) + "\n")
        else:
            for record in records:
                print(
                    f"{record.storage_decision.value}: {record.source_id}/{record.endpoint_id} "
                    f"{record.media_family.value} {record.size_bytes} bytes"
                )
        return 1 if any(record.storage_decision is StorageDecision.FETCH_ERROR for record in records) else 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())

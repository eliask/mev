"""Enumerate a declared initiative-register slice and import official actions.

Raw API pages are hashed checkpoints. Failed, repeated or corrupt pages do
not become a certificate of an empty register. Nothing here matches promises.
"""

import hashlib
import json
import sqlite3
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import httpx

from paa.acquire_initiatives import VASKI_ROWS_URL, merge_vaski_records, parse_vaski_row
from paa.config import CORPUS_CUTOFF, RAW, USER_AGENT


def import_result(conn: sqlite3.Connection, result: dict) -> dict:
    for obj in result["objects"]:
        normalized = dict(obj)
        normalized["disposition"] = normalized.get("disposition") or {"state": "UNRESOLVED", "date": None, "evidence_ids": []}
        conn.execute("INSERT OR REPLACE INTO official_objects VALUES (?, ?)", (obj["object_id"], json.dumps(normalized, ensure_ascii=False)))
        for ref in obj.get("evidence", []):
            evidence = {**ref, "url": ref.get("url") or ref.get("source_url") or obj.get("url"),
                        "quote": ref.get("quote") if ref.get("quote") is not None else obj.get("text", "")}
            conn.execute("INSERT OR REPLACE INTO evidence VALUES (?, ?)", (evidence["evidence_id"], json.dumps(evidence, ensure_ascii=False)))
    coverage = result["coverage"]
    conn.execute("INSERT OR REPLACE INTO source_coverage VALUES (?, ?)", (coverage["coverage_id"], json.dumps(coverage, ensure_ascii=False)))
    for page in result.get("manifests", []):
        conn.execute("INSERT INTO manifest VALUES (?, ?, ?, ?, ?, ?, ?)",
                     (coverage["source_id"], page["url"], page["sha256"], page["bytes"], 200, page["retrieved_at"], page["query"]))
    return {"objects": len(result["objects"]), "coverage_id": coverage["coverage_id"]}


def acquire_registry(years: list[int] | None = None, *, raw_dir: Path | None = None,
                     client=None, max_pages: int = 200, refresh: bool = False,
                     partition_identifiers: bool = False) -> dict:
    """Fetch accessible Finnish LA matter rows, including signatories.

    The server's SQL-like identifier filter was verified against real payloads.
    This enumerates one source's API rows, not all activity anywhere.
    """
    selected = sorted(set(years or [2023, 2024, 2025, 2026]))
    destination = raw_dir or RAW / "eduskunta" / "initiative_register"
    destination.mkdir(parents=True, exist_ok=True)
    records = defaultdict(list)
    manifests = []
    excluded = []
    seen = set()
    now = datetime.now(UTC).isoformat()
    getter = client.get if client is not None else httpx.get
    queries = [(year, digit) for year in selected for digit in (list("0123456789") if partition_identifiers else [""])]
    for year, digit in queries:
        pattern = f"LA {digit}%/{year} vp"
        for page in range(max_pages):
            params = {"columnName": "Eduskuntatunnus", "columnValue": pattern, "perPage": 100, "page": page}
            suffix = f"-prefix{digit}" if digit else ""
            path = destination / f"la-{year}{suffix}-{page}.json"
            receipt = path.with_suffix(".manifest.json")
            if path.exists() and receipt.exists() and not refresh:
                body = path.read_bytes()
                manifest = json.loads(receipt.read_text())
                if hashlib.sha256(body).hexdigest() != manifest["sha256"]:
                    raise ValueError(f"corrupt initiative checkpoint: {path}")
            else:
                response = getter(VASKI_ROWS_URL, params=params, headers={"User-Agent": USER_AGENT}, timeout=90)
                response.raise_for_status()
                body = response.content
                # Decode before replacing a valid checkpoint with an error shell.
                payload = json.loads(body)
                if "columnNames" not in payload or "rowData" not in payload:
                    raise ValueError("initiative API returned no table payload")
                manifest = {"url": str(response.url), "query": json.dumps(params, sort_keys=True),
                            "sha256": hashlib.sha256(body).hexdigest(), "bytes": len(body), "retrieved_at": now}
                path.write_bytes(body)
                receipt.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            manifests.append(manifest)
            # Refresh may move a checkpoint forward. Keep every retrieved
            # source body addressable by its immutable digest for prior traces.
            archive = destination / "by-sha256" / (manifest["sha256"] + ".json")
            archive.parent.mkdir(exist_ok=True)
            if archive.exists():
                if hashlib.sha256(archive.read_bytes()).hexdigest() != manifest["sha256"]:
                    raise ValueError(f"corrupt immutable initiative source: {archive}")
            else:
                archive.write_bytes(body)
            manifest["artifact_path"] = str(archive)
            payload = json.loads(body)
            rows = [dict(zip(payload["columnNames"], raw)) for raw in payload["rowData"]]
            new_ids = {str(r["Id"]) for r in rows} - seen
            if rows and len(new_ids) != len(rows):
                raise RuntimeError(f"initiative register repeated records on {year} page {page}; coverage withheld")
            seen.update(new_ids)
            for row in rows:
                matter = row.get("Eduskuntatunnus") or ""
                if not (matter.startswith(f"LA {digit}") and matter.endswith(f"/{year} vp")):
                    raise ValueError(f"server ignored identifier filter: {matter}")
                try:
                    record = parse_vaski_row(row, source_url=manifest["url"], retrieved_at=manifest["retrieved_at"])
                except ValueError as error:
                    excluded.append({"record_id": str(row["Id"]), "matter_id": matter, "reason": str(error)})
                    continue
                records[matter].append(record)
            if not payload.get("hasMore"):
                break
        else:
            raise RuntimeError(f"initiative registry pagination exceeded {max_pages} pages for {year}")
        if not digit or digit == "9":
            print(f"initiative register {year}: {len(records)} matters accumulated")
    objects = [merge_vaski_records(group) for _, group in sorted(records.items())]
    digest = hashlib.sha256("".join(m["sha256"] for m in manifests).encode()).hexdigest()
    coverage = {"coverage_id": "initiative-register-" + digest[:20], "source_id": "SRC-EDUSKUNTA-VASKI",
                "kind": "LEGISLATIVE_INITIATIVE", "url": VASKI_ROWS_URL, "years": selected,
                "enumeration": "ACCESSIBLE_FINNISH_LA_IDENTIFIER_FILTER", "state": "ENUMERATED_WITH_EXCLUSIONS" if excluded else "ENUMERATED",
                "query_partition": "FIRST_IDENTIFIER_DIGIT_0_TO_9" if partition_identifiers else "YEAR",
                "retrieved_at": now, "source_record_count": len(seen), "parsed_records": sum(map(len, records.values())),
                "object_count": len(objects), "excluded_records": excluded, "page_manifests": manifests,
                "window": {"earliest": f"{min(selected)}-01-01", "latest": min(f"{max(selected)}-12-31", CORPUS_CUTOFF)},
                "limitations": ["The API's accessible Finnish LA rows are the declared source universe; Status codes are retained without assuming publication semantics.",
                                "This does not include every initiative type, private work, speeches or questions.",
                                "Register completeness beyond this API cannot be inferred from pagination.",
                                "Content-only rows do not establish a later institutional disposition."]}
    result = {"objects": objects, "coverage": coverage, "manifests": manifests, "retrieved_at": now}
    (destination / "normalized.jsonl").write_text("".join(json.dumps(o, ensure_ascii=False)+"\n" for o in objects), encoding="utf-8")
    (destination / "coverage.json").write_text(json.dumps(coverage, ensure_ascii=False, indent=2), encoding="utf-8")
    return result

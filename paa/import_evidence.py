"""Import reviewed evidence records without replacing the campaign corpus."""

import json
import re
import sqlite3
from collections import Counter
from pathlib import Path

from paa.relations import review_shape_valid
from paa.store import add_manifest

_TABLES = {
    "official_object": ("official_objects", "object_id"),
    "evidence": ("evidence", "evidence_id"),
    "source_coverage": ("source_coverage", "coverage_id"),
    "relation_review": ("relation_reviews", "review_id"),
}


def import_records(conn: sqlite3.Connection, path: Path) -> dict:
    """Validate then transactionally import a JSON/JSONL evidence bundle.

    Statement, candidacy and role rows are deliberately ignored: the source
    acquisition layer owns those records. Importing a review never admits a
    finding; the compiler checks its source hashes, quotes and references.
    """
    rows = []
    ignored = Counter()
    contents = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        value = json.loads(contents)
        entries = value if isinstance(value, list) else [value]
    else:
        # JSONL uses LF records; U+2028/U+2029 are valid source-string data.
        entries = [json.loads(line) for line in contents.split('\n') if line.strip()]
    for number, entry in enumerate(entries, 1):
        if not isinstance(entry, dict):
            raise TypeError(f"{path}:{number}: expected an evidence record object")
        kind = entry.get("kind")
        if kind == "manifest":
            digest = entry.get("raw_sha256") or entry.get("sha256")
            if not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest):
                raise ValueError(f"{path}:{number}: manifest needs a source-body hash")
            rows.append((kind, entry))
            continue
        if kind not in _TABLES:
            ignored[str(kind)] += 1
            continue
        record = entry.get("row")
        key = _TABLES[kind][1]
        if not isinstance(record, dict) or not isinstance(record.get(key), str) or not record[key].strip():
            raise ValueError(f"{path}:{number}: {kind} lacks {key}")
        if kind == "relation_review":
            if not review_shape_valid(record):
                raise ValueError(f"{path}:{number}: incomplete relation review or invalid field types")
            for field in ("statement_sha256", "object_sha256"):
                if not re.fullmatch(r"[a-f0-9]{64}", str(record.get(field) or "")):
                    raise ValueError(f"{path}:{number}: invalid {field}")
        rows.append((kind, record))
    counts = Counter()
    with conn:
        for kind, record in rows:
            if kind == "manifest":
                add_manifest(conn, source_id=record["source_id"], url=record.get("url") or "",
                             sha256=record.get("raw_sha256") or record["sha256"],
                             bytes=int(record.get("raw_bytes") or record.get("bytes") or 0),
                             http_status=200, retrieved_at=record.get("retrieved_at"),
                             note=record.get("note") or "Imported source slice; full raw body is identified by hash.")
            else:
                table, key = _TABLES[kind]
                conn.execute(f"INSERT OR REPLACE INTO {table}({key}, json) VALUES (?, ?)",
                             (record[key], json.dumps(record, ensure_ascii=False, sort_keys=True)))
            counts[kind] += 1
    return {"imported": dict(counts), "ignored_source_rows": dict(ignored), "admission": "REQUIRES_COMPILATION"}

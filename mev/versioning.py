"""Extractor/detector schema versioning for MeV pipeline.

Each extractor (tagger, detector) has a version derived from its prompt text +
key code paths. When the prompt or logic changes, the version changes, and
existing results from the old version are invalidated.

Usage in a detector:

    from mev.versioning import extractor_version, is_stale

    VERSION = extractor_version(SYSTEM_PROMPT, "tag_lausunto_v1")

    # Check if results exist and are current
    if not is_stale(ENRICHMENTS_DB, "lausunto_tag", "he_id", he_id, VERSION):
        print(f"  {he_id}: up to date (v={VERSION[:8]})")
        continue

    # ... run extractor ...

    # Write with version stamp
    stamp_version(ENRICHMENTS_DB, "lausunto_tag", "he_id", he_id, VERSION)

The version table is:
    extractor_version(table_name TEXT, key_column TEXT, key_value TEXT, version TEXT, updated_at TEXT)
"""
from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


def extractor_version(prompt: str, salt: str = "") -> str:
    """Compute a version hash from prompt text + salt.

    The salt should include anything that affects output: model name,
    temperature, post-processing logic version, etc.
    """
    content = f"{salt}\n{prompt}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def _ensure_table(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS extractor_version (
            table_name  TEXT NOT NULL,
            key_column  TEXT NOT NULL,
            key_value   TEXT NOT NULL,
            version     TEXT NOT NULL,
            updated_at  TEXT NOT NULL,
            PRIMARY KEY (table_name, key_column, key_value)
        )
    """)


def is_stale(db_path: Path, table_name: str, key_column: str, key_value: str,
             current_version: str) -> bool:
    """Check if results for this key need recomputation.

    Returns True if:
    - No version record exists (never run)
    - Version record exists but doesn't match current_version (prompt changed)
    - No actual data exists in the target table

    Returns False if version matches (up to date, skip).
    """
    if not db_path.exists():
        return True
    conn = sqlite3.connect(str(db_path), timeout=30)
    try:
        _ensure_table(conn)
        row = conn.execute(
            "SELECT version FROM extractor_version "
            "WHERE table_name=? AND key_column=? AND key_value=?",
            (table_name, key_column, key_value)
        ).fetchone()
        if not row or row[0] != current_version:
            return True
        # Also check that actual data exists
        try:
            n = conn.execute(
                f"SELECT COUNT(*) FROM {table_name} WHERE {key_column}=?",
                (key_value,)
            ).fetchone()[0]
            return n == 0
        except sqlite3.OperationalError:
            return True
    finally:
        conn.close()


def stamp_version(db_path: Path, table_name: str, key_column: str,
                  key_value: str, version: str) -> None:
    """Record that this key was processed with this version."""
    conn = sqlite3.connect(str(db_path), timeout=30)
    _ensure_table(conn)
    conn.execute(
        "INSERT OR REPLACE INTO extractor_version "
        "(table_name, key_column, key_value, version, updated_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (table_name, key_column, key_value, version,
         datetime.now(timezone.utc).isoformat())
    )
    conn.commit()
    conn.close()


def stale_keys(db_path: Path, table_name: str, key_column: str,
               all_keys: list[str], current_version: str) -> list[str]:
    """Filter a list of keys to only those needing recomputation.

    Efficient batch version of is_stale() — one DB query instead of N.
    """
    if not db_path.exists():
        return list(all_keys)
    conn = sqlite3.connect(str(db_path), timeout=30)
    try:
        _ensure_table(conn)
        rows = conn.execute(
            "SELECT key_value, version FROM extractor_version "
            "WHERE table_name=? AND key_column=?",
            (table_name, key_column)
        ).fetchall()
        current = {r[0]: r[1] for r in rows}
        return [k for k in all_keys if current.get(k) != current_version]
    finally:
        conn.close()

"""Shared parameterized tagging engine for all document-level taggers.

Provides the common LLM-call, parse, DB-write, and run-loop machinery.
Each tagger (tag.py, tag_lausunto.py, tag_mietinto.py, tag_ptk.py) is a thin
wrapper that defines a TagConfig and delegates to the functions here.

Architecture:
  - TagConfig: dataclass holding all per-tagger configuration
  - call_llm_windowed(): shared LLM call with semaphore + truncation retry
  - parse_tag_lines(): parse "ID ROLE QUAL TOPIC" lines into {idx: (r,q,t)}
  - ensure_table(): create target DB table from TagConfig schema
  - run_tagger(): shared async run loop (load → chunk → tag → write → version)

Each wrapper defines its TagConfig and calls run_tagger().
Unique logic (irony patterns, government detection, top-HE selection) stays
in the individual wrapper modules.
"""
from __future__ import annotations

import asyncio
import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import aiohttp

from mev.llm import call_llm_full
from mev.versioning import stamp_version, stale_keys


# ---------------------------------------------------------------------------
# TagConfig dataclass
# ---------------------------------------------------------------------------

@dataclass
class TagConfig:
    """All per-tagger configuration for the shared engine.

    Fields:
      system_prompt    -- LLM system prompt (determines EXTRACTOR_VERSION)
      extractor_salt   -- version salt (e.g. "tag_he_v1")
      valid_roles      -- set of valid role codes the LLM may produce
      valid_quals      -- set of valid quality codes the LLM may produce
      valid_topics     -- set of valid topic codes (shared across taggers)
      role_labels      -- {code: label} for DB serialization
      qual_labels      -- {code: label} for DB serialization
      topic_labels     -- {code: label} for DB serialization
      target_table     -- DB table name for results (e.g. "lausunto_tag")
      version_key_col  -- column name used as the versioning key (e.g. "he_id")
      enrichments_db   -- Path to he_enrichments.db (injected by wrapper)
      window           -- sentences/paragraphs per LLM call (0 = no windowing)
      overlap          -- overlap between windows (used when window > 0)
    """
    system_prompt: str
    extractor_salt: str
    valid_roles: set[str]
    valid_quals: set[str]
    valid_topics: set[str]
    role_labels: dict[str, str]
    qual_labels: dict[str, str]
    topic_labels: dict[str, str]
    target_table: str
    version_key_col: str
    enrichments_db: Path
    window: int = 80
    overlap: int = 10


# ---------------------------------------------------------------------------
# Shared LLM wrapper
# ---------------------------------------------------------------------------

async def call_llm_windowed(
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    system: str,
    user: str,
    max_tokens: int,
) -> dict:
    """Acquire semaphore, call LLM, return full response dict.

    Returns dict with keys: content, tokens_in, tokens_out, elapsed,
    finish_reason. On error: finish_reason='error', content=''.
    """
    async with sem:
        return await call_llm_full(session, system, user, max_tokens=max_tokens)


# ---------------------------------------------------------------------------
# Tag line parser
# ---------------------------------------------------------------------------

def parse_tag_lines(
    raw: str,
    valid_roles: set[str],
    valid_quals: set[str],
    valid_topics: set[str],
) -> dict[int, tuple[str, str, str]]:
    """Parse LLM output of form 'ID ROLE QUAL TOPIC' lines.

    Returns {sid: (role, qual, topic)}. Skips invalid/incomplete lines.
    Falls back topic to 'X' if not in valid_topics.
    """
    results: dict[int, tuple[str, str, str]] = {}
    for line in raw.split('\n'):
        line = line.strip()
        if not line or line == 'NONE':
            continue
        parts = line.split()
        if not parts:
            continue
        try:
            sid = int(parts[0])
        except ValueError:
            continue
        role = parts[1] if len(parts) > 1 else '?'
        qual = parts[2] if len(parts) > 2 else '?'
        topic = parts[3] if len(parts) > 3 else 'X'
        if role not in valid_roles or qual not in valid_quals:
            continue
        if topic not in valid_topics:
            topic = 'X'
        if sid not in results:
            results[sid] = (role, qual, topic)
    return results


# ---------------------------------------------------------------------------
# Windowed tagging: process a sequence of text units through LLM in windows
# ---------------------------------------------------------------------------

async def tag_units_windowed(
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    config: TagConfig,
    units: list[tuple[str, Any]],  # list of (text, metadata)
    prompt_prefix: str = "",       # optional prefix to add before window lines
) -> dict[int, tuple[str, str, str]]:
    """Tag a list of text units using windowed LLM calls.

    units: list of (text, metadata) where text is sent to LLM
    Returns {0-based_idx: (role, qual, topic)} for all tagged units.

    Windowing: sends WINDOW units at a time with OVERLAP overlap.
    LLM receives 1-based indices "[N] text". Results remapped to 0-based.
    """
    all_tags: dict[int, tuple[str, str, str]] = {}
    n = len(units)
    window = config.window if config.window > 0 else n
    overlap = config.overlap
    step = max(1, window - overlap)

    start = 0
    while start < n:
        end = min(start + window, n)
        window_units = units[start:end]

        lines = [f'[{start + i + 1}] {text[:200]}' for i, (text, _) in enumerate(window_units)]
        prompt = (prompt_prefix + '\n' + '\n'.join(lines)).strip() if prompt_prefix else '\n'.join(lines)
        max_tok = 20 + len(window_units) * 8

        resp = await call_llm_windowed(session, sem, config.system_prompt, prompt, max_tok)
        if resp.get('error') or not resp.get('content'):
            if end >= n:
                break
            start += step
            continue

        tags = parse_tag_lines(
            resp['content'],
            config.valid_roles, config.valid_quals, config.valid_topics,
        )

        # Retry on truncation
        if resp.get('finish_reason') == 'length':
            resp2 = await call_llm_windowed(session, sem, config.system_prompt, prompt, max_tok * 2)
            if not resp2.get('error') and resp2.get('finish_reason') != 'length':
                tags = parse_tag_lines(
                    resp2['content'],
                    config.valid_roles, config.valid_quals, config.valid_topics,
                )

        # Remap: LLM uses 1-based indices starting at (start+1)
        for sid, val in tags.items():
            global_idx = sid - 1  # 0-based
            if 0 <= global_idx < n and global_idx not in all_tags:
                all_tags[global_idx] = val

        if end >= n:
            break
        start += step

    return all_tags


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def ensure_table_from_ddl(db_path: Path, ddl: str) -> None:
    """Create table(s) from a DDL executescript string."""
    conn = sqlite3.connect(str(db_path), timeout=30)
    conn.executescript(ddl)
    conn.close()


def write_rows_to_table(
    db_path: Path,
    table: str,
    delete_where: tuple[str, str],   # (column, value)
    columns: list[str],
    rows: list[tuple],
) -> int:
    """Delete existing rows for key, insert new rows. Returns count inserted."""
    col_str = ', '.join(columns)
    placeholders = ', '.join('?' for _ in columns)
    key_col, key_val = delete_where
    conn = sqlite3.connect(str(db_path), timeout=30)
    conn.execute(f"DELETE FROM {table} WHERE {key_col}=?", (key_val,))
    conn.executemany(
        f"INSERT OR REPLACE INTO {table} ({col_str}) VALUES ({placeholders})",
        rows,
    )
    conn.commit()
    conn.close()
    return len(rows)


# ---------------------------------------------------------------------------
# Stale-key filtering helper
# ---------------------------------------------------------------------------

def filter_stale(
    db_path: Path,
    table: str,
    key_col: str,
    all_keys: list[str],
    version: str,
    force: bool,
) -> list[str]:
    """Return subset of all_keys that need recomputation.

    If force=True, returns all_keys unchanged.
    """
    if force:
        return list(all_keys)
    before = len(all_keys)
    keys = stale_keys(db_path, table, key_col, all_keys, version)
    skipped = before - len(keys)
    if skipped:
        print(f"Skipping {skipped} up-to-date HEs (use --force to recompute)")
    return keys

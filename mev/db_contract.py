"""Content contracts for pipeline output databases.

Why this module exists
----------------------
`mev status` used to check that an output database EXISTS and has a plausible
SIZE. A database whose tables are all empty is a perfectly healthy-sized file,
so a producer that silently wrote nothing still reported green.

That is not hypothetical. The nine tables written by
``mev.pipeline.aggregate_entity_data`` sat at **zero rows from 2026-03-24 to
2026-08-17** while `mev status` printed a green checkmark for
`state_causal_map.db: 94852 KB`. Root cause: commit 77137be4 renamed the per-HE
table ``expert_statements`` -> ``expert_statement``, the aggregator's reader was
updated, the per-HE databases were not yet backfilled under the new name, and
the reader's ``except Exception: continue`` swallowed the resulting
``no such table`` for all 8,439 inputs. Nothing raised. Nothing warned.

The contract below is derived from **what each producer is supposed to write**,
not from what happens to be non-empty today, so it cannot ratify a current bug.
An empty table in ``CAUSAL_MAP_CONTRACT`` raises.

Usage
-----
    from mev.db_contract import assert_causal_map, check_causal_map

    assert_causal_map()                  # raises EmptyTableError on any breach
    problems = check_causal_map()        # non-raising, for status reporting
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from mev.config import CAUSAL_MAP_DB

# --------------------------------------------------------------------------
# state_causal_map.db
# --------------------------------------------------------------------------

_BUILD_CAUSAL_MAP = "mev build causal-map"
_AGGREGATE_ENTITIES = "mev build aggregate-entities"

#: Tables whose producer writes them unconditionally on a successful run.
#: Empty here == the producer failed or never ran == hard failure.
CAUSAL_MAP_CONTRACT: dict[str, str] = {
    # --- mev.pipeline.causal_map_db (statute + HE core) ---
    "statutes": _BUILD_CAUSAL_MAP,
    "edges": _BUILD_CAUSAL_MAP,
    "momentit": _BUILD_CAUSAL_MAP,
    "mapping": _BUILD_CAUSAL_MAP,
    "tae_texts": _BUILD_CAUSAL_MAP,
    "cofog": _BUILD_CAUSAL_MAP,
    "cofog_mapping": _BUILD_CAUSAL_MAP,
    "institutions": _BUILD_CAUSAL_MAP,
    "institution_statute_edges": _BUILD_CAUSAL_MAP,
    "delegations": _BUILD_CAUSAL_MAP,
    "asetus_authority": _BUILD_CAUSAL_MAP,
    "he_nodes": _BUILD_CAUSAL_MAP,
    "he_statute_link": _BUILD_CAUSAL_MAP,
    "he_he_refs": _BUILD_CAUSAL_MAP,
    "he_claims": _BUILD_CAUSAL_MAP,
    "claim_statute_link": _BUILD_CAUSAL_MAP,
    "successors": _BUILD_CAUSAL_MAP,
    "amendment_parents": _BUILD_CAUSAL_MAP,
    "co_amendments": _BUILD_CAUSAL_MAP,
    "editorial_notes": _BUILD_CAUSAL_MAP,
    # --- mev.pipeline.aggregate_entity_data (constituency roll-ups) ---
    # These are the six that were silently empty for ~5 months, plus the three
    # AKN-sourced tables written by the same script.
    "committee_scrutiny": _AGGREGATE_ENTITIES,
    "committee_he_link": _AGGREGATE_ENTITIES,
    "expert_org_scrutiny": _AGGREGATE_ENTITIES,
    "expert_org_he_link": _AGGREGATE_ENTITIES,
    "expert_person": _AGGREGATE_ENTITIES,
    "expert_person_he_link": _AGGREGATE_ENTITIES,
    "he_signatories": _AGGREGATE_ENTITIES,
    "signatory_profile": _AGGREGATE_ENTITIES,
    "ministry_he_profile": _AGGREGATE_ENTITIES,
}

#: Tables whose producer only writes them when an optional upstream artifact
#: exists (LLM detector output). Empty is legitimate on a fresh build, so these
#: are reported but never raise. Kept explicit so nobody quietly promotes them
#: into the hard contract, and so nobody forgets they exist.
CAUSAL_MAP_OPTIONAL: dict[str, str] = {
    "he_scrutiny": "mev detect scrutiny (via .tmp/scrutiny/scrutiny_analysis.json)",
    "delegation_drift_scale": "mev detect drift (copied from he_enrichments.db)",
}


class EmptyTableError(RuntimeError):
    """A table the pipeline is supposed to populate is missing or empty."""


def _table_count(conn: sqlite3.Connection, table: str) -> int | None:
    """Row count, or None if the table does not exist."""
    try:
        return conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
    except sqlite3.Error:
        return None


def check_causal_map(db_path: Path | str | None = None) -> list[tuple[str, str, str]]:
    """Return contract breaches as ``(table, producer, problem)`` triples.

    Does not raise on breaches — only on an unreadable/absent database, which
    is a different failure that the caller must not confuse with an empty table.
    """
    path = Path(db_path) if db_path is not None else CAUSAL_MAP_DB
    if not path.exists():
        raise EmptyTableError(f"{path} does not exist")

    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        breaches: list[tuple[str, str, str]] = []
        for table, producer in CAUSAL_MAP_CONTRACT.items():
            n = _table_count(conn, table)
            if n is None:
                breaches.append((table, producer, "MISSING"))
            elif n == 0:
                breaches.append((table, producer, "EMPTY"))
        return breaches
    finally:
        conn.close()


def causal_map_optional_counts(
    db_path: Path | str | None = None,
) -> list[tuple[str, str, int | None]]:
    """Row counts for the optional (LLM-gated) tables, for reporting only."""
    path = Path(db_path) if db_path is not None else CAUSAL_MAP_DB
    if not path.exists():
        return []
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return [
            (t, producer, _table_count(conn, t))
            for t, producer in CAUSAL_MAP_OPTIONAL.items()
        ]
    finally:
        conn.close()


def assert_causal_map(
    db_path: Path | str | None = None,
    skip: tuple[str, ...] | set[str] = (),
) -> None:
    """Raise ``EmptyTableError`` if any contracted table is missing or empty.

    Call this at the end of any producer that writes ``state_causal_map.db``,
    so a silent zero fails the run that caused it rather than the next reader.

    ``skip`` exempts tables the caller deliberately did not attempt this run
    (e.g. the AKN-sourced tables when the source zip is absent). Use it only for
    "did not attempt", never for "attempted and got zero" — the second case is
    the bug this module exists to catch.
    """
    breaches = [b for b in check_causal_map(db_path) if b[0] not in skip]
    if not breaches:
        return
    path = Path(db_path) if db_path is not None else CAUSAL_MAP_DB
    by_producer: dict[str, list[str]] = {}
    for table, producer, problem in breaches:
        by_producer.setdefault(producer, []).append(f"{table} ({problem})")
    lines = [f"{len(breaches)} contracted table(s) unpopulated in {path}:"]
    for producer, tables in sorted(by_producer.items()):
        lines.append(f"  producer `{producer}`: " + ", ".join(sorted(tables)))
    raise EmptyTableError("\n".join(lines))

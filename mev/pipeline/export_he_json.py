"""Export structured JSON summaries from per-HE SQLite DBs.

For each per-HE DB, generates a companion .json file alongside it.
The JSON is structured for LLM consumption (llms.txt / /he-dbs/{he-id}.json).

Usage:
    mev build export-json              # all HEs
    mev build export-json he-112-2025  # one HE
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from mev.config import HE_DB_DIR


def export_he_json(he_db_path: Path, output_path: Path) -> bool:
    """Export structured JSON summary from a per-HE SQLite DB.

    Returns True on success, False if the DB has no metadata.
    Missing tables are handled gracefully (try/except).
    """
    conn = sqlite3.connect(str(he_db_path))
    conn.row_factory = sqlite3.Row

    try:
        meta_row = conn.execute("SELECT * FROM metadata LIMIT 1").fetchone()
        if not meta_row:
            conn.close()
            return False
        meta = dict(meta_row)
    except sqlite3.OperationalError:
        conn.close()
        return False

    # Expert statements with concern/proposal counts
    experts = []
    try:
        rows = conn.execute("""
            SELECT e.statement_id, e.expert_name,
                   count(CASE WHEN s.tag='concern' THEN 1 END) as concerns,
                   count(CASE WHEN s.tag='proposal' THEN 1 END) as proposals
            FROM expert_statement e
            LEFT JOIN span_tag s ON s.doc_type='lausunto' AND s.doc_id=e.statement_id
            GROUP BY e.statement_id
        """).fetchall()
        experts = [
            {
                "id": r["statement_id"],
                "name": r["expert_name"],
                "concerns": r["concerns"] or 0,
                "proposals": r["proposals"] or 0,
            }
            for r in rows
        ]
    except sqlite3.OperationalError:
        pass

    # Top concern snippets
    top_concerns = []
    try:
        rows = conn.execute("""
            SELECT snippet FROM span_tag
            WHERE tag='concern' ORDER BY rowid LIMIT 10
        """).fetchall()
        top_concerns = [r["snippet"] for r in rows if r["snippet"]]
    except sqlite3.OperationalError:
        pass

    # Unanswered questions
    unanswered_questions = []
    try:
        rows = conn.execute("""
            SELECT snippet FROM span_tag
            WHERE tag='question' ORDER BY rowid LIMIT 10
        """).fetchall()
        unanswered_questions = [r["snippet"] for r in rows if r["snippet"]]
    except sqlite3.OperationalError:
        pass

    # Fiscal mentions
    fiscal_mentions = []
    try:
        rows = conn.execute("""
            SELECT snippet FROM span_tag
            WHERE tag='fiscal' ORDER BY rowid LIMIT 10
        """).fetchall()
        fiscal_mentions = [r["snippet"] for r in rows if r["snippet"]]
    except sqlite3.OperationalError:
        pass

    # Discourse edge type counts
    discourse: dict[str, int] = {}
    try:
        rows = conn.execute("""
            SELECT edge_type, count(*) as n FROM discourse_edge GROUP BY edge_type
        """).fetchall()
        discourse = {r["edge_type"]: r["n"] for r in rows}
    except sqlite3.OperationalError:
        pass

    # A row in ev_decisions means a parliamentary reply was stored.
    # It does not by itself mean the bill passed.
    ev_passed = None
    ev_record_present = False
    try:
        cur = conn.execute("SELECT * FROM ev_decisions LIMIT 1")
        ev = cur.fetchone()
        if ev is not None:
            ev_record_present = True
            cols = [d[0].lower() for d in cur.description]
            row = {cols[i]: ev[i] for i in range(len(cols))}
            for key in ("passed", "hyvaksytty", "paatos", "decision", "tulos"):
                if key not in row or row[key] in (None, ""):
                    continue
                val = str(row[key]).strip().lower()
                if val in {"1", "true", "hyvaksytty", "hyväksytty", "passed", "yes"}:
                    ev_passed = True
                elif val in {"0", "false", "hylatty", "hylätty", "rejected", "no"}:
                    ev_passed = False
    except sqlite3.OperationalError:
        pass

    # Dissent count
    dissent_count = 0
    try:
        dissent_count = conn.execute("SELECT count(*) FROM dissents").fetchone()[0]
    except sqlite3.OperationalError:
        pass

    # laws_amended — stored as JSON string in metadata
    laws_amended = []
    try:
        raw = meta.get("laws_amended", "[]") or "[]"
        laws_amended = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        laws_amended = []

    he_id = meta.get("he_id", "")

    result = {
        "he_id": he_id,
        "year": meta.get("year", ""),
        "title": meta.get("title", ""),
        "ministry": meta.get("ministry", ""),
        "date_published": meta.get("date_published", ""),
        "laws_amended": laws_amended,
        "experts": experts,
        "top_concerns": top_concerns,
        "unanswered_questions": unanswered_questions,
        "fiscal_mentions": fiscal_mentions,
        "discourse": discourse,
        "ev_passed": ev_passed,
        "ev_record_present": ev_record_present,
        "dissent_count": dissent_count,
        "db_url": f"/he-dbs/{he_id}.db",
        "viewer_url": f"/mev/he-viewer.html#{he_id}",
    }

    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    conn.close()
    return True


def run_all(he_id: str | None = None) -> dict:
    """Export JSON for all (or one) per-HE DBs."""
    if he_id:
        dbs = [HE_DB_DIR / f"{he_id}.db"]
    else:
        dbs = sorted(HE_DB_DIR.glob("he-*.db"))

    n_ok = 0
    n_skip = 0
    for db_path in dbs:
        json_path = db_path.with_suffix(".json")
        ok = export_he_json(db_path, json_path)
        if ok:
            n_ok += 1
        else:
            n_skip += 1

    return {"exported": n_ok, "skipped": n_skip}


def main() -> None:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Export per-HE JSON summaries")
    parser.add_argument("he_id", nargs="?", help="Specific HE (e.g. he-112-2025)")
    parser.add_argument("--all", action="store_true", help="Export all HEs")
    args = parser.parse_args()

    if not args.he_id and not args.all:
        parser.print_help()
        sys.exit(1)

    he_id = args.he_id if not args.all else None
    result = run_all(he_id)
    print(f"Done: {result['exported']} exported, {result['skipped']} skipped")


if __name__ == "__main__":
    main()

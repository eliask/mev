"""
Build multi-source corroboration score per HE.

Experimental prioritization heuristic. Not a credibility measure and not a
failure score. The inputs can move together; disagreement, speeches and
dissents are not evidence that anyone failed. Weights are uncalibrated.

Inputs:
  1. unreason_flag
  2. scrutiny_ignored
  3. dissents
  4. ptk_speeches
  5. formal_consult
  6. ev_decision row presence (not a passage verdict)

Output:
    .tmp/corroboration_scores.csv  -- one row per HE, all signals + composite
    Prints top HEs by composite score

Usage:
    uv run mev pipeline corroboration
    uv run mev pipeline corroboration --top 30
    uv run mev pipeline corroboration --min-signals 3
"""

import argparse
import csv
import json
import re
import sqlite3
import sys

from mev.config import ROOT, HE_DB_DIR, ENRICHMENTS_DB, INDEX_DB

SCRUTINY_PATH = ROOT / ".tmp" / "scrutiny" / "scrutiny_analysis.json"
OUT_CSV = ROOT / ".tmp" / "corroboration_scores.csv"


def he_db_to_tunnus(he_id: str) -> str:
    m = re.match(r"he-(\d+)-(\d+)", he_id)
    return f"HE {m.group(1)}/{m.group(2)} vp" if m else ""


def load_scrutiny() -> dict[str, float]:
    """Returns {he_id: scrutiny_ignored_rate}"""
    if not SCRUTINY_PATH.exists():
        return {}
    with open(SCRUTINY_PATH) as f:
        data = json.load(f)
    return {
        he["he_id"]: he.get("scrutiny_ignored_rate", 0.0)
        for he in data
        if "error" not in he and he.get("scrutiny_ignored_rate") is not None
    }


def load_unreason(enr_conn: sqlite3.Connection) -> dict[str, dict]:
    """Returns {he_id: {count, max_severity, detectors}}"""
    rows = enr_conn.execute(
        "SELECT he_id, detector, severity FROM unreason_flag"
    ).fetchall()
    result: dict[str, dict] = {}
    for he_id, detector, severity in rows:
        if he_id not in result:
            result[he_id] = {"count": 0, "max_severity": 0, "detectors": []}
        result[he_id]["count"] += 1
        result[he_id]["max_severity"] = max(result[he_id]["max_severity"], severity)
        result[he_id]["detectors"].append(detector)
    return result


def load_ptk_speeches(idx_conn: sqlite3.Connection) -> dict[str, int]:
    """Returns {he_tunnus: speech_count}"""
    rows = idx_conn.execute(
        "SELECT he_tunnus, COUNT(*) FROM ptk_speech "
        "WHERE he_tunnus IS NOT NULL GROUP BY he_tunnus"
    ).fetchall()
    return {r[0]: r[1] for r in rows}


def load_dissents(idx_conn: sqlite3.Connection) -> dict[str, int]:
    """Returns {he_id: dissent_count}"""
    rows = idx_conn.execute(
        "SELECT cr.he_id, COUNT(*) "
        "FROM vaski_dissent vd "
        "JOIN committee_report cr ON cr.report_id = vd.report_id "
        "GROUP BY cr.he_id"
    ).fetchall()
    return {r[0]: r[1] for r in rows}


def load_formal_consults(idx_conn: sqlite3.Connection) -> dict[str, int]:
    """Returns {he_id: formal_consultation_count}"""
    rows = idx_conn.execute(
        "SELECT cr.he_id, COUNT(*) "
        "FROM vaski_cross_ref vcr "
        "JOIN committee_report cr ON cr.report_id = vcr.report_id "
        "WHERE vcr.relation_type = 'FORMAL_CONSULTATION' "
        "GROUP BY cr.he_id"
    ).fetchall()
    return {r[0]: r[1] for r in rows}


def load_ev_has(idx_conn: sqlite3.Connection) -> set[str]:
    """Returns set of he_tunnukset that have an EV decision."""
    rows = idx_conn.execute(
        "SELECT he_tunnus_json FROM ev_document WHERE he_tunnus_json != '[]'"
    ).fetchall()
    result: set[str] = set()
    for (json_str,) in rows:
        try:
            for t in json.loads(json_str):
                result.add(t)
        except Exception:
            pass
    return result


def composite_score(row: dict) -> float:
    """
    Composite failure signal score [0-1].
    Uses normalized, bounded inputs to prevent one signal from dominating.
    """
    # Normalize each signal to [0,1]
    unreason_norm = min(row["unreason_count"] / 7.0, 1.0)          # 7+ flags = max
    severity_norm = (row["unreason_max_sev"] - 1) / 2.0 if row["unreason_max_sev"] >= 1 else 0.0
    scrutiny_norm = min(row["scrutiny_ignored_rate"], 1.0)           # already 0-1
    dissent_norm = min(row["dissent_count"] / 4.0, 1.0)              # 4+ dissents = max
    ptk_norm = min(row["ptk_speech_count"] / 300.0, 1.0)             # 300+ speeches = max
    formal_norm = min(row["formal_consult_count"] / 5.0, 1.0)        # 5+ formal = max

    # Weight signals by independence and reliability
    # unreason: high weight (document's own text contradicts itself)
    # scrutiny: high weight (external expert + committee independent signal)
    # dissent: medium weight (strong signal but fewer data points)
    # ptk: low weight (controversy proxy, not failure signal per se)
    # formal_consult: low weight (process indicator)
    weights = {
        "unreason": 0.30,
        "severity": 0.10,
        "scrutiny": 0.30,
        "dissent": 0.15,
        "ptk": 0.10,
        "formal": 0.05,
    }
    score = (
        weights["unreason"] * unreason_norm
        + weights["severity"] * severity_norm
        + weights["scrutiny"] * scrutiny_norm
        + weights["dissent"] * dissent_norm
        + weights["ptk"] * ptk_norm
        + weights["formal"] * formal_norm
    )
    return round(score, 4)


def count_active_signals(row: dict) -> int:
    signals = 0
    if row["unreason_count"] > 0:
        signals += 1
    if row["scrutiny_ignored_rate"] > 0.1:
        signals += 1
    if row["dissent_count"] > 0:
        signals += 1
    if row["ptk_speech_count"] > 20:
        signals += 1
    if row["formal_consult_count"] > 0:
        signals += 1
    return signals


def main(args=None):
    if args is None:
        parser = argparse.ArgumentParser(description="Build multi-source corroboration scores")
        parser.add_argument("--top", type=int, default=20, help="Print top N HEs by score")
        parser.add_argument("--min-signals", type=int, default=2,
                            help="Only include HEs with >= N active signals")
        args = parser.parse_args()

    if not ENRICHMENTS_DB.exists():
        print(f"Error: {ENRICHMENTS_DB} not found")
        sys.exit(1)
    if not INDEX_DB.exists():
        print(f"Error: {INDEX_DB} not found")
        sys.exit(1)

    enr_conn = sqlite3.connect(str(ENRICHMENTS_DB))
    idx_conn = sqlite3.connect(str(INDEX_DB))
    idx_conn.row_factory = sqlite3.Row

    print("Loading signals...")
    unreason_data = load_unreason(enr_conn)
    scrutiny_data = load_scrutiny()
    ptk_data = load_ptk_speeches(idx_conn)
    dissent_data = load_dissents(idx_conn)
    formal_data = load_formal_consults(idx_conn)
    ev_set = load_ev_has(idx_conn)

    enr_conn.close()
    idx_conn.close()

    # Get all HE IDs from DB directory
    all_he_ids = sorted(p.stem for p in HE_DB_DIR.glob("he-*.db"))
    print(f"Total HE DBs: {len(all_he_ids)}")
    print(f"With unreason flags: {len(unreason_data)}")
    print(f"With scrutiny data: {len(scrutiny_data)}")
    print(f"With PTK speeches: {len(ptk_data)}")
    print(f"With dissents: {len(dissent_data)}")
    print(f"With formal consults: {len(formal_data)}")
    print(f"With EV decision: {len(ev_set)}")

    rows = []
    for he_id in all_he_ids:
        he_tunnus = he_db_to_tunnus(he_id)
        ur = unreason_data.get(he_id, {})
        row = {
            "he_id": he_id,
            "he_tunnus": he_tunnus,
            "unreason_count": ur.get("count", 0),
            "unreason_max_sev": ur.get("max_severity", 0),
            "unreason_detectors": "|".join(ur.get("detectors", [])),
            "scrutiny_ignored_rate": scrutiny_data.get(he_id, 0.0),
            "dissent_count": dissent_data.get(he_id, 0),
            "ptk_speech_count": ptk_data.get(he_tunnus, 0),
            "formal_consult_count": formal_data.get(he_id, 0),
            "has_ev": 1 if he_tunnus in ev_set else 0,
        }
        row["active_signals"] = count_active_signals(row)
        row["composite_score"] = composite_score(row)
        rows.append(row)

    # Filter to HEs with enough signals
    rows = [r for r in rows if r["active_signals"] >= args.min_signals]
    rows.sort(key=lambda r: -r["composite_score"])

    # Write CSV
    fieldnames = [
        "he_id", "he_tunnus", "composite_score", "active_signals",
        "unreason_count", "unreason_max_sev", "unreason_detectors",
        "scrutiny_ignored_rate", "dissent_count",
        "ptk_speech_count", "formal_consult_count", "has_ev",
    ]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"\nWrote {OUT_CSV} ({len(rows)} rows)")
    print(f"\nTop {args.top} by composite score:")
    print(f"{'he_id':20s} {'score':6s} {'sigs':5s} {'unrsn':6s} {'scr%':6s} {'diss':5s} {'ptk':5s}")
    print("-" * 65)
    for r in rows[:args.top]:
        print(
            f"{r['he_id']:20s} "
            f"{r['composite_score']:.3f}  "
            f"{r['active_signals']:3d}   "
            f"{r['unreason_count']:4d}  "
            f"{r['scrutiny_ignored_rate']:5.0%}  "
            f"{r['dissent_count']:3d}  "
            f"{r['ptk_speech_count']:5d}"
        )


def run(**kwargs):
    """Standard pipeline API entry point."""
    main()


if __name__ == "__main__":
    main()

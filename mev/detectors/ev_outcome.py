"""
EV outcome analysis: did parliament pass government proposals despite warning signals?

Computes the "rubber stamp" finding:
  - What fraction of submitted HEs (2015-2025) got an EV (i.e., passed)?
  - For HEs that DID pass: how many had unreason_flag / committee dissents / PTK speeches?
  - Do any EVs show rejection/return/amendment?

Reads:
    data/legislative_index.sqlite  (ev_document, he, ptk_speech, vaski_dissent, committee_report)
    .tmp/he_enrichments.db         (unreason_flag)
    .tmp/corroboration_scores.csv  (corroboration scores)

Writes:
    .tmp/ev_outcome_stats.json     (aggregate findings)
    Prints summary table

Usage:
    uv run mev detector ev-outcome
    uv run mev detector ev-outcome --csv  # also write per-HE CSV
"""

import argparse
import csv
import json
import re
import sqlite3
from collections import Counter

from mev.config import ROOT, ENRICHMENTS_DB, INDEX_DB

CORROBORATION_CSV = ROOT / ".tmp" / "corroboration_scores.csv"
OUT_JSON = ROOT / ".tmp" / "ev_outcome_stats.json"
OUT_CSV = ROOT / ".tmp" / "ev_outcome_per_he.csv"

# EV decision_text patterns — virtually all EVs are "passed verbatim"
# Rejections would be exceptional and visible in non-standard text
PASS_PATTERN = re.compile(r"Eduskunta on hyväksynyt", re.IGNORECASE)
REJECT_PATTERN = re.compile(r"hylkää|hylättiin|ei hyväksy", re.IGNORECASE)
RETURN_PATTERN = re.compile(r"palautetaan|palautettiin", re.IGNORECASE)


def classify_ev(decision_text: str | None) -> str:
    if not decision_text:
        return "no_text"
    if REJECT_PATTERN.search(decision_text):
        return "rejected"
    if RETURN_PATTERN.search(decision_text):
        return "returned"
    if PASS_PATTERN.search(decision_text):
        return "passed"
    return "other"


def main(args=None):
    if args is None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--csv", action="store_true", help="Write per-HE CSV")
        args = parser.parse_args()

    idx = sqlite3.connect(str(INDEX_DB))
    idx.row_factory = sqlite3.Row

    # All HEs submitted 2015-2025
    all_hes_2015_2025 = {
        r["canonical_id"]
        for r in idx.execute(
            "SELECT canonical_id FROM he WHERE year BETWEEN 2015 AND 2025"
        ).fetchall()
    }
    print(f"HEs submitted 2015-2025: {len(all_hes_2015_2025)}")

    # EV outcomes
    ev_rows = idx.execute(
        "SELECT ev_tunnus, he_tunnus_json, decision_text FROM ev_document"
    ).fetchall()

    # Map he_id -> ev outcome
    he_ev_outcome: dict[str, str] = {}
    ev_outcome_counts: Counter = Counter()

    for row in ev_rows:
        outcome = classify_ev(row["decision_text"])
        ev_outcome_counts[outcome] += 1
        try:
            he_tunnukset = json.loads(row["he_tunnus_json"] or "[]")
        except Exception:
            he_tunnukset = []
        for he_tunnus in he_tunnukset:
            # Convert "HE 149/2024 vp" -> "he-149-2024"
            m = re.match(r"HE (\d+)/(\d+) vp", he_tunnus)
            if m:
                he_id = f"he-{m.group(1)}-{m.group(2)}"
                he_ev_outcome[he_id] = outcome

    passed_he_ids = {k for k, v in he_ev_outcome.items() if v == "passed"}
    n_passed_2015_2025 = len(passed_he_ids & all_hes_2015_2025)
    n_no_ev_2015_2025 = len(all_hes_2015_2025 - set(he_ev_outcome.keys()))

    print(f"\nEV outcomes across all years:")
    for outcome, n in ev_outcome_counts.most_common():
        print(f"  {outcome:20s}: {n:4d}")

    print(f"\nFor HEs submitted 2015-2025:")
    print(f"  Passed (got EV):     {n_passed_2015_2025:4d} / {len(all_hes_2015_2025)} "
          f"({n_passed_2015_2025/len(all_hes_2015_2025):.1%})")
    print(f"  No EV found:         {n_no_ev_2015_2025:4d} / {len(all_hes_2015_2025)} "
          f"({n_no_ev_2015_2025/len(all_hes_2015_2025):.1%})")

    # Unreason flags per HE
    enr = sqlite3.connect(str(ENRICHMENTS_DB))
    unreason_by_he = {}
    for he_id, n in enr.execute(
        "SELECT he_id, COUNT(*) FROM unreason_flag GROUP BY he_id"
    ).fetchall():
        unreason_by_he[he_id] = n
    enr.close()

    # Dissents per HE
    dissents_by_he = {}
    for he_id, n in idx.execute(
        "SELECT cr.he_id, COUNT(*) FROM vaski_dissent vd "
        "JOIN committee_report cr ON cr.report_id = vd.report_id "
        "GROUP BY cr.he_id"
    ).fetchall():
        dissents_by_he[he_id] = n

    # PTK speech count per HE (using he_tunnus)
    speech_by_he = {}
    for row in idx.execute(
        "SELECT he_tunnus, COUNT(*) as n FROM ptk_speech "
        "WHERE he_tunnus IS NOT NULL GROUP BY he_tunnus"
    ).fetchall():
        m = re.match(r"HE (\d+)/(\d+) vp", row["he_tunnus"])
        if m:
            speech_by_he[f"he-{m.group(1)}-{m.group(2)}"] = row["n"]

    # For passed 2015-2025 HEs: how many had warnings?
    passed_2015_2025 = passed_he_ids & all_hes_2015_2025
    with_unreason = sum(1 for h in passed_2015_2025 if unreason_by_he.get(h, 0) > 0)
    with_dissents = sum(1 for h in passed_2015_2025 if dissents_by_he.get(h, 0) > 0)
    with_both = sum(1 for h in passed_2015_2025
                    if unreason_by_he.get(h, 0) > 0 and dissents_by_he.get(h, 0) > 0)
    sev3 = sum(1 for h in passed_2015_2025 if unreason_by_he.get(h, 0) >= 3)

    print(f"\nOf the {len(passed_2015_2025)} passed HEs (2015-2025):")
    print(f"  With >=1 unreason flag:          {with_unreason:4d} ({with_unreason/len(passed_2015_2025):.1%})")
    print(f"  With >=1 committee dissent:      {with_dissents:4d} ({with_dissents/len(passed_2015_2025):.1%})")
    print(f"  With BOTH (unreason + dissent): {with_both:4d} ({with_both/len(passed_2015_2025):.1%})")
    print(f"  With >=3 unreason flags:         {sev3:4d} ({sev3/len(passed_2015_2025):.1%})")

    # Corroboration set: passed HEs with >=3 active signals
    if CORROBORATION_CSV.exists():
        corr_rows = list(csv.DictReader(open(CORROBORATION_CSV)))
        high_signal = [r for r in corr_rows if int(r["active_signals"]) >= 3]
        passed_high_signal = [r for r in high_signal if r["he_id"] in passed_he_ids]
        print(f"\nHEs with >=3 corroboration signals AND a passing EV: "
              f"{len(passed_high_signal)} / {len(high_signal)}")

    # Optionally write per-HE CSV
    if args.csv:
        fieldnames = ["he_id", "year", "ev_outcome", "unreason_flag",
                      "dissent_count", "ptk_speeches"]
        rows_out = []
        for he_id in all_hes_2015_2025:
            m = re.match(r"he-\d+-(\d+)", he_id)
            year = m.group(1) if m else ""
            rows_out.append({
                "he_id": he_id,
                "year": year,
                "ev_outcome": he_ev_outcome.get(he_id, "no_ev"),
                "unreason_flag": unreason_by_he.get(he_id, 0),
                "dissent_count": dissents_by_he.get(he_id, 0),
                "ptk_speeches": speech_by_he.get(he_id, 0),
            })
        rows_out.sort(key=lambda r: (-r["unreason_flag"], -r["dissent_count"]))
        with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows_out)
        print(f"\nWrote {OUT_CSV}")

    # Save summary JSON
    stats = {
        "period": "2015-2025",
        "hes_submitted": len(all_hes_2015_2025),
        "hes_passed": n_passed_2015_2025,
        "hes_no_ev": n_no_ev_2015_2025,
        "passage_rate": round(n_passed_2015_2025 / len(all_hes_2015_2025), 4),
        "passed_with_unreason_flag": with_unreason,
        "passed_with_dissents": with_dissents,
        "passed_with_both": with_both,
        "passed_with_3plus_flags": sev3,
        "ev_outcome_counts": dict(ev_outcome_counts),
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {OUT_JSON}")

    idx.close()


def run(**kwargs):
    """Standard detector API entry point."""
    main()


if __name__ == "__main__":
    main()

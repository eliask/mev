"""Candidate flag: a health or social statute citation whose lead committee is not StV.

Routing to Hallintovaliokunta is not evasion and not proof of weaker scrutiny.
It is a mismatch with a hand-picked statute list, exported for review.

Two confidence levels:
  broad:  HE cites >=1 health/social statute -> HaV instead of StV
  strict: HE cites >=2 health/social statutes -> HaV instead of StV

Requires:
  data/legislative_index.sqlite             (committee_report, he tables)
  mekanismirealismi/data/statute_graph/state_causal_map.db  (he_statute_link)

Output:
  .tmp/comm_evade_stats.json   -- aggregate stats
  .tmp/comm_evade_per_he.csv   -- per-HE routing detail

Usage:
  uv run mev detector comm-evade
"""

import csv
import json
import sqlite3
from collections import Counter

from mev.config import ROOT, INDEX_DB, CAUSAL_MAP_DB

OUT_DIR = ROOT / ".tmp"

# Core health/social statutes — HEs touching these should go to StV
HEALTH_SOCIAL_STATUTES = {
    "2010/1326": "terveydenhuoltolaki",
    "2014/1301": "sosiaalihuoltolaki",
    "1989/1062": "erikoissairaanhoitolaki",
    "1972/66":   "kansanterveyslaki",
    "2022/612":  "laki hyvinvointialueista",
    "2022/615":  "laki sosiaali- ja terveydenhuollon järjestämisestä",
}

EXPECTED_COMMITTEE = "Sosiaali- ja terveysvaliokunta"
# Primary evasion target: Hallintovaliokunta handles admin/municipal/immigration law
EVASION_COMMITTEE = "Hallintovaliokunta"


def load_statute_refs(scm_conn: sqlite3.Connection) -> dict[str, list[str]]:
    """Returns he_id -> list of health/social statute_ids cited."""
    statute_ids = list(HEALTH_SOCIAL_STATUTES.keys())
    placeholders = ",".join("?" * len(statute_ids))
    rows = scm_conn.execute(
        f"SELECT he_id, statute_id FROM he_statute_link WHERE statute_id IN ({placeholders})",
        statute_ids,
    ).fetchall()
    result: dict[str, list[str]] = {}
    for he_id, statute_id in rows:
        result.setdefault(he_id, []).append(statute_id)
    return result


def load_lead_committees(idx_conn: sqlite3.Connection, he_ids: list[str]) -> dict[str, list[str]]:
    """Returns he_id -> list of lead committees (mietinto only)."""
    result: dict[str, list[str]] = {}
    for he_id in he_ids:
        rows = idx_conn.execute(
            "SELECT committee FROM committee_report "
            "WHERE he_id = ? AND report_type = 'Valiokunnan mietintö'",
            (he_id,),
        ).fetchall()
        comms = [r[0] for r in rows if r[0]]
        if comms:
            result[he_id] = comms
    return result


def load_he_meta(idx_conn: sqlite3.Connection, he_ids: list[str]) -> dict[str, dict]:
    """Returns he_id -> {year, title, ministry}."""
    result: dict[str, dict] = {}
    for he_id in he_ids:
        row = idx_conn.execute(
            "SELECT year, title, ministry FROM he WHERE canonical_id = ?",
            (he_id,),
        ).fetchone()
        if row:
            result[he_id] = {"year": row[0], "title": row[1] or "", "ministry": row[2] or ""}
        else:
            result[he_id] = {"year": None, "title": "", "ministry": ""}
    return result


def classify_routing(lead_committees: list[str]) -> str:
    """Classify the routing outcome for a set of lead committees."""
    if not lead_committees:
        return "unknown"
    if EXPECTED_COMMITTEE in lead_committees:
        return "expected"
    if EVASION_COMMITTEE in lead_committees:
        return "evade_hav"
    return "other"


def main(args=None):
    OUT_DIR.mkdir(exist_ok=True)

    scm_conn = sqlite3.connect(str(CAUSAL_MAP_DB))
    idx_conn = sqlite3.connect(str(INDEX_DB))

    # 1. Find HEs that cite health/social statutes
    he_statutes = load_statute_refs(scm_conn)
    scm_conn.close()
    print(f"HEs citing >=1 health/social statute: {len(he_statutes)}")

    he_ids = list(he_statutes.keys())

    # 2. Load lead committee per HE
    lead_committees = load_lead_committees(idx_conn, he_ids)
    print(f"HEs with known lead committee: {len(lead_committees)}")

    # 3. Load metadata
    meta = load_he_meta(idx_conn, he_ids)
    idx_conn.close()

    # 4. Build per-HE records
    records = []
    for he_id in he_ids:
        statutes = he_statutes[he_id]
        comms = lead_committees.get(he_id, [])
        routing = classify_routing(comms)
        m = meta.get(he_id, {})
        records.append({
            "he_id": he_id,
            "year": m.get("year"),
            "title": m.get("title", ""),
            "ministry": m.get("ministry", ""),
            "health_statutes_cited": len(statutes),
            "health_statute_ids": ";".join(sorted(statutes)),
            "lead_committees": ";".join(comms) if comms else "",
            "routing": routing,
            "is_evade": routing == "evade_hav",
            "is_evade_strict": routing == "evade_hav" and len(statutes) >= 2,
        })

    # 5. Aggregate stats
    with_committee = [r for r in records if r["routing"] != "unknown"]
    evade_broad = [r for r in records if r["is_evade"]]
    evade_strict = [r for r in records if r["is_evade_strict"]]

    # Committee distribution for all health/social HEs with known routing
    all_comms: list[str] = []
    for r in with_committee:
        all_comms.extend(r["lead_committees"].split(";") if r["lead_committees"] else [])
    comm_dist = dict(Counter(all_comms).most_common())

    # By year for evade_broad
    evade_by_year = dict(Counter(r["year"] for r in evade_broad if r["year"]).most_common())

    stats = {
        "total_he_citing_health_social": len(he_statutes),
        "with_known_committee": len(with_committee),
        "routing_expected_stv": sum(1 for r in with_committee if r["routing"] == "expected"),
        "routing_evade_hav_broad": len(evade_broad),
        "routing_evade_hav_strict_ge2": len(evade_strict),
        "routing_other": sum(1 for r in with_committee if r["routing"] == "other"),
        "routing_unknown": sum(1 for r in records if r["routing"] == "unknown"),
        "evade_rate_broad": (
            round(len(evade_broad) / len(with_committee), 4) if with_committee else 0
        ),
        "evade_rate_strict": (
            round(len(evade_strict) / len(with_committee), 4) if with_committee else 0
        ),
        "committee_distribution": comm_dist,
        "evade_broad_by_year": dict(sorted(evade_by_year.items())),
        "evade_broad_examples": [
            {"he_id": r["he_id"], "year": r["year"],
             "statutes": r["health_statute_ids"],
             "committee": r["lead_committees"],
             "title": r["title"][:120]}
            for r in sorted(evade_broad, key=lambda x: -(x["health_statutes_cited"]))[:20]
        ],
    }

    stats_path = OUT_DIR / "comm_evade_stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"\nWrote {stats_path}")

    # 6. Per-HE CSV
    csv_path = OUT_DIR / "comm_evade_per_he.csv"
    fieldnames = [
        "he_id", "year", "health_statutes_cited", "health_statute_ids",
        "lead_committees", "routing", "is_evade", "is_evade_strict",
        "ministry", "title",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in sorted(records, key=lambda x: (not x["is_evade"], x["he_id"])):
            writer.writerow({k: r[k] for k in fieldnames})
    print(f"Wrote {csv_path} ({len(records)} rows)")

    # 7. Summary printout
    print(f"\n--- Committee routing for health/social statute HEs ---")
    print(f"Total HEs citing health/social statute:  {len(he_statutes):4d}")
    print(f"With known lead committee:               {len(with_committee):4d}")
    print(f"  -> Sosiaali- ja terveysvaliokunta:      {stats['routing_expected_stv']:4d} ({100*stats['routing_expected_stv']//max(1,len(with_committee)):2d}%)")
    print(f"  -> Hallintovaliokunta (evasion, broad):  {len(evade_broad):4d} ({100*len(evade_broad)//max(1,len(with_committee)):2d}%)")
    print(f"  -> Hallintovaliokunta (strict >=2 cite):  {len(evade_strict):4d}")
    print(f"  -> Other committee:                     {stats['routing_other']:4d}")
    print(f"  -> Unknown (no mietinto found):         {stats['routing_unknown']:4d}")

    print(f"\nTop evade_strict cases (HEs citing >=2 health/social statutes -> HaV):")
    for r in sorted(evade_strict, key=lambda x: -(x["health_statutes_cited"]))[:10]:
        print(f"  {r['he_id']} ({r['year']})  [{r['health_statutes_cited']} statutes]"
              f"  {r['title'][:80]}")

    print(f"\nCommittee distribution (all health/social HEs):")
    for comm, n in sorted(comm_dist.items(), key=lambda x: -x[1])[:8]:
        print(f"  {n:4d}  {comm}")


def run(**kwargs):
    """Standard detector API entry point."""
    main()


if __name__ == "__main__":
    main()

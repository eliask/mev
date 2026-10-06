"""
Delegation drift at scale — Tier 2.1 (FINDINGS_ROADMAP.md).

For every delegation clause that HAS an implementing asetus, compare:
  - What the HE (government proposal) promised the asetus would do
  - The actual delegation scope (match_text / quote)

Join key: statute_id + section.

Steps:
  1. Read state_causal_map.db: delegations + asetus_authority + he_claims
  2. Build tractable pairs: (statute, section, asetus_id, he_claims)
  3. Without --llm: output CSV inventory of tractable pairs (default)
  4. With --llm: call llama-server to compare HE promise vs delegation scope
  5. With --embed: write results to he_enrichments.db delegation_drift_scale table

Usage:
    mev detect drift
    mev detect drift --top 20
    mev detect drift --statute 2014/917
    mev detect drift --llm
    mev detect drift --llm --embed

Output:
    .tmp/delegation_drift_scale/inventory.csv     — all tractable pairs (always)
    .tmp/delegation_drift_scale/comparisons.csv   — LLM results (with --llm)
    .tmp/delegation_drift_scale/report.txt        — human-readable summary
"""

from __future__ import annotations

import argparse
import asyncio
import os
import collections
import csv
import json
import re
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import aiohttp
from mev.llm import call_llm, LLMContextExhausted
from mev.config import ROOT, ENRICHMENTS_DB

DB_PATH = ROOT / 'data' / 'statute_graph' / 'state_causal_map.db'
OUTPUT_DIR = ROOT / '.tmp' / 'delegation_drift_scale'

_MANDATORY_RE = re.compile(r'\b(säädetään|on annettava|on säädettävä)\b', re.I)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_tractable_pairs(conn: sqlite3.Connection, statute_filter: str | None = None) -> list[dict]:
    """Build list of (statute, delegation_section, asetus_id, he_claims) tuples."""
    filter_clause = "AND d.statute_id = ?" if statute_filter else ""
    params: list = [statute_filter] if statute_filter else []

    rows = conn.execute(f"""
        SELECT
            d.statute_id,
            s.title as law_title,
            s.in_force,
            d.section,
            d.delegation_type,
            d.match_text,
            d.quote,
            aa.asetus_id,
            aa.parent_section as matched_section,
            aa.preamble_quote,
            at.title as asetus_title,
            at.date_issued as asetus_date,
            at.in_force as asetus_in_force
        FROM delegations d
        JOIN statutes s ON s.id = d.statute_id
        JOIN asetus_authority aa ON aa.parent_statute_id = d.statute_id
        LEFT JOIN statutes at ON at.id = aa.asetus_id
        WHERE s.in_force = 1 {filter_clause}
          AND (aa.parent_section = d.section OR aa.parent_section = '' OR aa.parent_section IS NULL)
        ORDER BY d.statute_id, d.section
    """, params).fetchall()

    if not rows:
        return []

    pairs = [dict(r) for r in rows]

    # Attach HE claims per statute
    claim_rows = conn.execute("""
        SELECT csl.statute_id, c.he_id, c.claim_type, c.text, c.amount_eur, c.confidence
        FROM claim_statute_link csl
        JOIN he_claims c ON c.claim_id = csl.claim_id
        WHERE csl.statute_id IN (SELECT DISTINCT statute_id FROM delegations)
        ORDER BY csl.statute_id, c.claim_type
    """).fetchall()

    claims_by_statute: dict[str, list[dict]] = defaultdict(list)
    for r in claim_rows:
        claims_by_statute[r[0]].append({
            'he_id': r[1], 'claim_type': r[2], 'text': r[3],
            'amount_eur': r[4], 'confidence': r[5],
        })

    for p in pairs:
        p['he_claims'] = claims_by_statute.get(p['statute_id'], [])
        p['he_claim_count'] = len(p['he_claims'])
        p['is_mandatory'] = bool(_MANDATORY_RE.search(p['match_text'] or ''))

    return pairs


def load_top_statutes(conn: sqlite3.Connection, top_n: int) -> list[str]:
    """Return top-N statute IDs by (delegation × claims) score."""
    rows = conn.execute("""
        SELECT d.statute_id,
               COUNT(DISTINCT d.section) as deleg_secs,
               COUNT(DISTINCT aa.asetus_id) as matched_asetukset,
               COUNT(DISTINCT csl.claim_id) as claims
        FROM delegations d
        JOIN statutes s ON s.id = d.statute_id
        LEFT JOIN asetus_authority aa ON aa.parent_statute_id = d.statute_id
        LEFT JOIN claim_statute_link csl ON csl.statute_id = d.statute_id
        WHERE s.in_force = 1 AND aa.asetus_id IS NOT NULL
        GROUP BY d.statute_id
        HAVING matched_asetukset > 0 AND claims > 0
        ORDER BY deleg_secs * claims DESC
        LIMIT ?
    """, [top_n]).fetchall()
    return [r[0] for r in rows]


# ---------------------------------------------------------------------------
# LLM comparison
# ---------------------------------------------------------------------------

_SYSTEM = (
    "Olet lakiasiantuntija joka arvioi, toteuttaako valtioneuvoston asetus "
    "lain delegaatioklausulin tarkoituksen. "
    'Vastaa aina täsmälleen JSON-muodossa: {"verdict": "...", "explanation": "..."} '
    "Verdict-arvot: ALIGNED (asetus toteuttaa delegaation) / "
    "PARTIAL (osittainen toteutus) / "
    "DRIFT (asetus ei vastaa delegaatiota) / "
    "CANNOT_ASSESS (ei riittävästi tietoa)"
)


async def _compare_one(session: aiohttp.ClientSession, pair: dict) -> dict:
    """LLM comparison for one delegation-asetus pair.

    Compares the delegation clause scope against the asetus title/preamble.
    Does NOT use HE claims — those are from wrong HEs for most statutes.
    """
    delegation_text = (pair.get('quote') or pair.get('match_text') or '').strip()[:500]
    asetus_title = pair.get('asetus_title', '') or ''
    preamble = (pair.get('preamble_quote') or '')[:300]

    user = (
        f"Delegaatioklausuli (§{pair['section']} laissa {pair['statute_id']} "
        f"\"{pair.get('law_title', '')}\"):\n"
        f"\"{delegation_text}\"\n\n"
        f"Asetus {pair['asetus_id']}: {asetus_title}\n"
        f"Johtolause: {preamble}\n\n"
        f'Luokittele JSON: {{"verdict": "ALIGNED|PARTIAL|DRIFT|CANNOT_ASSESS", '
        f'"explanation": "1-2 lausetta miksi"}}'
    )
    ctx = f"{pair['statute_id']} §{pair['section']} → {pair['asetus_id']}"
    try:
        content = await call_llm(session, _SYSTEM, user, max_tokens=150, ctx=ctx)
        m = re.search(r'\{.*?\}', content, re.DOTALL)
        if m:
            data = json.loads(m.group())
            return {
                'drift_verdict': data.get('verdict', 'UNKNOWN'),
                'drift_explanation': data.get('explanation', ''),
            }
        return {'drift_verdict': 'PARSE_ERROR', 'drift_explanation': content[:200]}
    except LLMContextExhausted:
        return {'drift_verdict': 'TRUNCATED', 'drift_explanation': 'context exceeded'}
    except Exception as e:
        return {'drift_verdict': 'ERROR', 'drift_explanation': str(e)[:100]}


async def run_llm_comparisons(pairs: list[dict], parallel: int = int(os.environ.get("LLM_PARALLEL", "4"))) -> None:
    """Run LLM comparisons for all pairs with bounded parallelism, mutating in-place."""
    sem = asyncio.Semaphore(parallel)
    print(f"  Running LLM on {len(pairs)} pairs (parallel={parallel}, results cached)...")

    async def _bounded(session, pair):
        async with sem:
            return await _compare_one(session, pair)

    async with aiohttp.ClientSession() as session:
        results = await asyncio.gather(*[_bounded(session, p) for p in pairs])

    for p, r in zip(pairs, results):
        p.update(r)
    print(f"  Verdicts: {dict(collections.Counter(r.get('drift_verdict') for r in results))}")


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_inventory_csv(pairs: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        'statute_id', 'law_title', 'section', 'delegation_type', 'is_mandatory',
        'asetus_id', 'asetus_title', 'asetus_date', 'asetus_in_force',
        'matched_section', 'he_claim_count', 'match_text',
    ]
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        for p in pairs:
            w.writerow({k: p.get(k, '') for k in fields})
    print(f"  Wrote {len(pairs)} pairs → {path}")


def write_comparisons_csv(pairs: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        'statute_id', 'law_title', 'section', 'delegation_type', 'is_mandatory',
        'asetus_id', 'asetus_title', 'he_claim_count',
        'drift_verdict', 'drift_explanation', 'match_text',
    ]
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        for p in pairs:
            w.writerow({k: p.get(k, '') for k in fields})
    print(f"  Wrote {len(pairs)} comparisons → {path}")


def write_report(pairs: list[dict], path: Path, run_llm: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    total = len(pairs)
    mandatory = sum(1 for p in pairs if p.get('is_mandatory'))
    with_claims = sum(1 for p in pairs if p.get('he_claim_count', 0) > 0)

    lines = [
        "=== Delegation Drift at Scale — Tier 2.1 ===",
        f"Tractable pairs: {total} (delegation + asetus + optional HE claims)",
        f"  Mandatory (säädetään/on annettava): {mandatory}/{total}",
        f"  With HE claims: {with_claims}/{total}",
        "",
    ]

    if run_llm:
        by_verdict: dict[str, int] = defaultdict(int)
        for p in pairs:
            by_verdict[p.get('drift_verdict', 'UNKNOWN')] += 1
        lines += ["LLM Verdicts:"]
        for verdict, count in sorted(by_verdict.items()):
            pct = count / total * 100 if total else 0
            lines.append(f"  {verdict}: {count} ({pct:.1f}%)")
        lines.append("")

        drifts = [p for p in pairs if p.get('drift_verdict') == 'DRIFT']
        if drifts:
            lines.append(f"Top DRIFT findings ({len(drifts)} total):")
            for p in drifts[:10]:
                lines.append(f"  {p['statute_id']} §{p['section']} → {p['asetus_id']}")
                lines.append(f"    {p.get('drift_explanation', '')[:120]}")
        lines.append("")

    # Top statutes by coverage
    by_statute: dict[str, list] = defaultdict(list)
    for p in pairs:
        by_statute[p['statute_id']].append(p)

    lines.append("Top statutes by tractable delegation pairs:")
    for sid, statute_pairs in sorted(by_statute.items(), key=lambda x: -len(x[1]))[:10]:
        title = statute_pairs[0].get('law_title', '')[:40]
        n_claims = max(p.get('he_claim_count', 0) for p in statute_pairs)
        n_mand = sum(1 for p in statute_pairs if p.get('is_mandatory'))
        lines.append(f"  {sid} — {title}: {len(statute_pairs)} pairs, {n_mand} mandatory, {n_claims} HE claims")

    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f"  Report → {path}")


def embed_in_db(pairs: list[dict], db_path: Path) -> None:
    """Write results to he_enrichments.db delegation_drift_scale table."""
    if not db_path.exists():
        print(f"  SKIP embed: {db_path} not found")
        return
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS delegation_drift_scale (
            statute_id TEXT,
            section TEXT,
            delegation_type TEXT,
            is_mandatory INTEGER,
            asetus_id TEXT,
            asetus_title TEXT,
            he_claim_count INTEGER,
            drift_verdict TEXT,
            drift_explanation TEXT,
            match_text TEXT,
            PRIMARY KEY (statute_id, section, asetus_id)
        )
    """)
    conn.execute("DELETE FROM delegation_drift_scale")
    conn.executemany("""
        INSERT OR REPLACE INTO delegation_drift_scale VALUES
        (?,?,?,?,?,?,?,?,?,?)
    """, [
        (
            p.get('statute_id'), p.get('section'), p.get('delegation_type'),
            int(p.get('is_mandatory', False)),
            p.get('asetus_id'), p.get('asetus_title'),
            p.get('he_claim_count', 0),
            p.get('drift_verdict'), p.get('drift_explanation'),
            (p.get('match_text') or '')[:500],
        )
        for p in pairs
    ])
    conn.commit()
    conn.close()
    print(f"  Embedded {len(pairs)} rows → {db_path}:delegation_drift_scale")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Delegation drift at scale (Tier 2.1)")
    ap.add_argument('--statute', help="Single statute ID, e.g. 2014/917")
    ap.add_argument('--top', type=int, default=0, help="Process top-N statutes by (deleg × claims) score")
    ap.add_argument('--llm', action='store_true', help="Run LLM comparison (needs llama-server on :8080)")
    ap.add_argument('--embed', action='store_true', help="Write results to he_enrichments.db")
    ap.add_argument('--db', default=str(DB_PATH), help="Path to state_causal_map.db")
    args = ap.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"ERROR: DB not found: {db_path}", file=sys.stderr)
        sys.exit(1)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    print("=== Delegation drift at scale ===")

    if args.statute:
        print(f"Single statute mode: {args.statute}")
        pairs = load_tractable_pairs(conn, statute_filter=args.statute)
    elif args.top > 0:
        top_ids = load_top_statutes(conn, args.top)
        print(f"Top-{args.top} statutes: {', '.join(top_ids)}")
        pairs = []
        for sid in top_ids:
            pairs.extend(load_tractable_pairs(conn, statute_filter=sid))
    else:
        print("Full corpus mode (all statutes with delegations + matching asetukset)")
        pairs = load_tractable_pairs(conn)

    conn.close()

    print(f"Found {len(pairs)} tractable (delegation, asetus) pairs")

    if args.llm:
        print(f"Running LLM comparison on {len(pairs)} pairs...")
        asyncio.run(run_llm_comparisons(pairs))

    write_inventory_csv(pairs, OUTPUT_DIR / 'inventory.csv')
    if args.llm:
        write_comparisons_csv(pairs, OUTPUT_DIR / 'comparisons.csv')
    write_report(pairs, OUTPUT_DIR / 'report.txt', run_llm=args.llm)

    if args.embed:
        embed_in_db(pairs, ENRICHMENTS_DB)

    # Quick summary
    total = len(pairs)
    mandatory = sum(1 for p in pairs if p.get('is_mandatory'))
    with_claims = sum(1 for p in pairs if p.get('he_claim_count', 0) > 0)
    print(f"\nSummary: {total} tractable pairs | {mandatory} mandatory | {with_claims} with HE claims")
    if args.llm:
        verdicts = collections.Counter(p.get('drift_verdict') for p in pairs)
        print("Verdicts:", dict(verdicts))

    print(f"\nOutputs: {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()

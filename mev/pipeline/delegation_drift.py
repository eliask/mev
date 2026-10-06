"""
Detect delegation drift: Does a matching asetus exist for each delegation clause?

For each enacted statute from our HEs:
  1. Find delegation clauses ("valtioneuvoston asetuksella säädetään X:stä")
  2. Look up asetus_authority.csv for asetukset citing the same law AND section
  3. Direct section match → MATCHED or NO_ASETUS

The johtolause (preamble) of each asetus explicitly names the parent law and
section it implements. asetus_authority.csv has this parsed as parent_section.
This gives a direct, metadata-based link — no lemma comparison or LLM needed.

For asetukset that don't specify a section (20% of corpus, mostly pre-2000),
they are matched to ALL delegations in their parent law as UNSPECIFIED.

Reads:
    .tmp/mechanism_drift/he_to_enacted.json       [HE→enacted mapping]
    data/statute_graph/delegations.csv            [delegation clauses]
    data/statute_graph/asetus_authority.csv        [asetus→parent law+section mapping]

Outputs:
    .tmp/mechanism_drift/delegation_drift_report.json
    .tmp/mechanism_drift/delegation_drift_summary.txt

With --embed: writes delegation_drift table into per-HE databases.

Usage:
    mev detect drift
    mev detect drift he-13-2025
    mev detect drift --embed
"""

import argparse
import csv
import json
import re
import sqlite3
import sys
from collections import defaultdict

from mev.config import ROOT, HE_DB_DIR
DATA_DIR = ROOT / 'data' / 'statute_graph'
OUTPUT_DIR = ROOT / '.tmp' / 'mechanism_drift'
HE_TO_ENACTED_PATH = OUTPUT_DIR / 'he_to_enacted.json'
DELEGATIONS_CSV = DATA_DIR / 'delegations.csv'
ASETUS_CSV = DATA_DIR / 'asetus_authority.csv'


def load_delegations() -> dict[str, list[dict]]:
    """Load delegations.csv → {statute_id: [delegation_rows]}."""
    result = defaultdict(list)
    with open(DELEGATIONS_CSV) as f:
        for row in csv.DictReader(f):
            result[row['statute_id']].append(row)
    return result


def load_asetus_parents() -> dict[str, list[dict]]:
    """Load asetus_authority.csv → {parent_statute_id: [asetus_rows]}."""
    result = defaultdict(list)
    with open(ASETUS_CSV) as f:
        for row in csv.DictReader(f):
            result[row['parent_statute_id']].append(row)
    return result


def extract_delegation_scope(quote: str) -> str:
    """Extract the delegated scope from a delegation clause quote.

    E.g. "Valtioneuvoston asetuksella säädetään tarkemmin
    palvelutarvekertoimen laskennasta" → "palvelutarvekertoimen laskennasta"
    """
    # Scope AFTER the delegation verb
    m = re.search(
        r'(?:valtioneuvoston\s+)?asetuksella\s+'
        r'(?:voidaan\s+)?(?:antaa|säätää|säädetään)\s+'
        r'(?:tarkempia?\s+(?:säännöksiä\s+)?)?'
        r'(?:tarkemmin\s+)?'
        r'(.+)',
        quote, re.IGNORECASE | re.DOTALL
    )
    if m:
        scope = m.group(1).strip().rstrip('.')
        if len(scope) > 10:
            return scope

    # Scope BEFORE the delegation verb
    m = re.search(
        r'(.+?)\s+'
        r'(?:säädetään|annetaan|voidaan\s+antaa)\s+'
        r'(?:valtioneuvoston\s+)?asetuksella',
        quote, re.IGNORECASE | re.DOTALL
    )
    if m:
        scope = m.group(1).strip()
        scope = re.sub(r'^Tarkempi[ae]\s+säännöksi[äa]\s+', '', scope, flags=re.IGNORECASE)
        if len(scope) > 10:
            return scope

    return quote


def normalize_section(s: str) -> str:
    """Normalize section number for matching: '13' → '13', '50 b' → '50 b'."""
    return s.strip().lower()


def analyze_he(he_id: str, enacted_ids: list[str], delegations_db: dict,
               asetus_parents: dict) -> dict:
    """Analyze delegation drift for one HE using direct section matching."""

    result = {
        'he_id': he_id,
        'statutes': [],
        'summary': {
            'enacted_count': len(enacted_ids),
            'delegations_total': 0,
            'matched': 0,
            'unspecified': 0,
            'no_asetus': 0,
        }
    }

    for sid in enacted_ids:
        delegations = delegations_db.get(sid, [])
        if not delegations:
            continue

        # Deduplicate delegations by (section, delegation_type)
        seen = set()
        unique_delegations = []
        for d in delegations:
            key = (d['section'], d['delegation_type'])
            if key not in seen:
                seen.add(key)
                unique_delegations.append(d)

        # All asetukset for this parent law
        all_asetukset = asetus_parents.get(sid, [])

        # Index asetukset by section for direct matching
        by_section: dict[str, list[dict]] = defaultdict(list)
        unspecified: list[dict] = []  # asetukset without parent_section
        for a in all_asetukset:
            sec = a.get('parent_section', '').strip()
            if sec:
                by_section[normalize_section(sec)].append(a)
            else:
                unspecified.append(a)

        statute_result = {
            'statute_id': sid,
            'delegations': [],
            'asetukset_total': len(all_asetukset),
        }

        for deleg in unique_delegations:
            result['summary']['delegations_total'] += 1
            scope = extract_delegation_scope(deleg['quote'])
            d_section = normalize_section(deleg['section'])

            # Direct section match
            section_matches = by_section.get(d_section, [])

            deleg_result = {
                'section': deleg['section'],
                'delegation_type': deleg['delegation_type'],
                'scope': scope,
                'quote': deleg['quote'][:300],
                'asetukset': [],
                'status': 'NO_ASETUS',
            }

            if section_matches:
                # Direct match: asetus explicitly cites this section
                deleg_result['status'] = 'MATCHED'
                result['summary']['matched'] += 1
                for a in section_matches:
                    deleg_result['asetukset'].append({
                        'asetus_id': a['asetus_id'],
                        'parent_section': a.get('parent_section', ''),
                        'match_type': 'DIRECT',
                        'preamble': a.get('preamble_quote', '')[:200],
                    })
            elif unspecified:
                # Asetukset exist but don't specify section — possible match
                deleg_result['status'] = 'UNSPECIFIED'
                result['summary']['unspecified'] += 1
                for a in unspecified:
                    deleg_result['asetukset'].append({
                        'asetus_id': a['asetus_id'],
                        'parent_section': '',
                        'match_type': 'UNSPECIFIED',
                        'preamble': a.get('preamble_quote', '')[:200],
                    })
            else:
                # No matching asetus at all
                result['summary']['no_asetus'] += 1

            statute_result['delegations'].append(deleg_result)

        result['statutes'].append(statute_result)

    return result


def embed_delegation_drift(result: dict):
    """Write delegation drift data into the per-HE database."""
    he_id = result['he_id']
    db_path = HE_DB_DIR / f'{he_id}.db'
    if not db_path.exists():
        return

    conn = sqlite3.connect(str(db_path))
    c = conn.cursor()

    c.execute("DROP TABLE IF EXISTS delegation_drift")
    c.execute('''CREATE TABLE delegation_drift (
        statute_id TEXT,
        section TEXT,
        delegation_type TEXT,
        scope TEXT,
        quote TEXT,
        status TEXT,
        asetus_id TEXT,
        match_type TEXT,
        preamble TEXT
    )''')

    for st in result.get('statutes', []):
        for deleg in st.get('delegations', []):
            if deleg['asetukset']:
                for a in deleg['asetukset']:
                    c.execute(
                        'INSERT INTO delegation_drift VALUES (?,?,?,?,?,?,?,?,?)',
                        (
                            st['statute_id'],
                            deleg['section'],
                            deleg['delegation_type'],
                            deleg['scope'],
                            deleg['quote'],
                            deleg['status'],
                            a['asetus_id'],
                            a['match_type'],
                            a['preamble'],
                        )
                    )
            else:
                c.execute(
                    'INSERT INTO delegation_drift VALUES (?,?,?,?,?,?,?,?,?)',
                    (
                        st['statute_id'],
                        deleg['section'],
                        deleg['delegation_type'],
                        deleg['scope'],
                        deleg['quote'],
                        'NO_ASETUS',
                        '',
                        '',
                        '',
                    )
                )

    # Summary
    c.execute("DROP TABLE IF EXISTS delegation_drift_summary")
    c.execute("CREATE TABLE delegation_drift_summary (key TEXT PRIMARY KEY, value TEXT)")
    for k, v in result.get('summary', {}).items():
        c.execute('INSERT INTO delegation_drift_summary VALUES (?,?)', (k, str(v)))

    conn.commit()
    conn.close()


def main():
    parser = argparse.ArgumentParser(description='Detect delegation drift (direct section match)')
    parser.add_argument('he_ids', nargs='*', help='HE IDs to process (default: all)')
    parser.add_argument('--embed', action='store_true',
                        help='Write drift data into per-HE databases')
    args = parser.parse_args()

    if not HE_TO_ENACTED_PATH.exists():
        print("Error: he_to_enacted.json not found. Run detect_mechanism_drift.py first.")
        sys.exit(1)

    with open(HE_TO_ENACTED_PATH) as f:
        he_to_enacted = json.load(f)
    print(f"HE→enacted mapping: {len(he_to_enacted)} HEs")

    delegations_db = load_delegations()
    print(f"Delegation clauses: {sum(len(v) for v in delegations_db.values())} "
          f"across {len(delegations_db)} statutes")

    asetus_parents = load_asetus_parents()
    total_asetukset = sum(len(v) for v in asetus_parents.values())
    has_section = sum(1 for v in asetus_parents.values()
                      for a in v if a.get('parent_section', '').strip())
    print(f"Asetus→parent links: {total_asetukset} asetukset "
          f"({has_section} with section, {total_asetukset - has_section} without)")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.he_ids:
        he_ids = args.he_ids
    else:
        he_ids = sorted(p.stem for p in HE_DB_DIR.glob('he-*.db'))

    all_results = []
    totals = defaultdict(int)

    for he_id in he_ids:
        enacted_ids = he_to_enacted.get(he_id, [])
        r = analyze_he(he_id, enacted_ids, delegations_db, asetus_parents)
        all_results.append(r)

        s = r['summary']
        if s['delegations_total'] == 0:
            continue

        for k, v in s.items():
            totals[k] += v

        flag = ''
        if s['no_asetus'] > 0:
            flag = f'  *** {s["no_asetus"]} no asetus'

        print(f"  {he_id}: {s['delegations_total']} delegations — "
              f"{s['matched']} matched, {s['unspecified']} unspecified, "
              f"{s['no_asetus']} no asetus{flag}")

    # Embed
    if args.embed:
        embedded = 0
        for r in all_results:
            if r['summary']['delegations_total'] > 0:
                embed_delegation_drift(r)
                embedded += 1
        print(f"\nEmbedded delegation drift into {embedded} HE databases")

    # Save JSON
    with open(OUTPUT_DIR / 'delegation_drift_report.json', 'w') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=list)

    # Save summary
    with open(OUTPUT_DIR / 'delegation_drift_summary.txt', 'w') as f:
        f.write("Delegation Drift: Direct Section Match\n")
        f.write("=" * 55 + "\n\n")
        f.write(f"HEs analyzed:            {len(he_ids)}\n")
        f.write(f"Delegation clauses:      {totals['delegations_total']}\n")
        f.write(f"  MATCHED (direct):      {totals['matched']}\n")
        f.write(f"  UNSPECIFIED (no §):    {totals['unspecified']}\n")
        f.write(f"  NO_ASETUS:             {totals['no_asetus']}\n\n")

        # Detail NO_ASETUS
        f.write("DELEGATIONS WITH NO ASETUS\n")
        f.write("-" * 55 + "\n")
        for r in all_results:
            for st in r.get('statutes', []):
                for deleg in st.get('delegations', []):
                    if deleg['status'] == 'NO_ASETUS':
                        f.write(f"\n{r['he_id']} | {st['statute_id']} §{deleg['section']} "
                                f"({deleg['delegation_type']})\n")
                        f.write(f"  {deleg['scope'][:200]}\n")

    print(f"\nTotals: {totals['delegations_total']} delegations")
    print(f"  {totals['matched']} matched (direct section), "
          f"{totals['unspecified']} unspecified, "
          f"{totals['no_asetus']} no asetus")
    print(f"\nResults: {OUTPUT_DIR / 'delegation_drift_report.json'}")
    print(f"Summary: {OUTPUT_DIR / 'delegation_drift_summary.txt'}")


if __name__ == '__main__':
    main()

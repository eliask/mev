"""Cross-document claim matching: HE claims vs expert lausunto claims.

Matches claims across document types by topic code. Detects:
- CONTRADICTION: expert and HE make factual claims on same topic, different direction
- CORROBORATION: expert supports HE claim with additional evidence
- UNADDRESSED: expert raises concern on topic that HE doesn't analyze
- EUR_DIVERGENCE: both cite EUR amounts on same topic, amounts differ significantly

Reads from: he_enrichments.db [sentence_tag (HE), lausunto_tag (expert)]
Writes to: he_enrichments.db [cross_doc_matches table]

Usage:
    mev pipeline cross-doc                      # all HEs with both HE + lausunto tags
    mev pipeline cross-doc --he he-241-2020     # specific HE
    mev pipeline cross-doc --write-db           # persist matches
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

from mev.config import ROOT, ENRICHMENTS_DB


# ---------------------------------------------------------------------------
# Match types
# ---------------------------------------------------------------------------

def _eur_from_json(s: str | None) -> list[float]:
    if not s:
        return []
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return []


def find_matches(
    he_tags: list[dict],
    lausunto_tag: list[dict],
) -> list[dict]:
    """Find cross-document matches between HE and expert claims.

    Both inputs are lists of dicts with keys:
        sent_text, role, quality, topic, eur_amounts, (+ source-specific fields)
    """
    matches = []

    # Group by topic
    he_by_topic: dict[str, list[dict]] = defaultdict(list)
    for t in he_tags:
        if t['role'] and t['topic']:
            he_by_topic[t['topic']].append(t)

    ex_by_topic: dict[str, list[dict]] = defaultdict(list)
    for t in lausunto_tag:
        if t['role'] and t['topic']:
            ex_by_topic[t['topic']].append(t)

    all_topics = set(he_by_topic.keys()) | set(ex_by_topic.keys())

    for topic in all_topics:
        he_group = he_by_topic.get(topic, [])
        ex_group = ex_by_topic.get(topic, [])

        if not he_group and ex_group:
            # Expert raises concerns on a topic HE doesn't address
            concerns = [e for e in ex_group if e['role'] in ('concern', 'fact_claim')]
            for c in concerns:
                matches.append({
                    'match_type': 'UNADDRESSED',
                    'topic': topic,
                    'expert_text': c['sent_text'][:200],
                    'expert_role': c['role'],
                    'expert_quality': c['quality'],
                    'expert_statement_id': c.get('statement_id', ''),
                    'expert_name': c.get('expert_name', ''),
                    'he_text': '',
                    'he_role': '',
                    'he_quality': '',
                })
            continue

        if not ex_group:
            continue

        # EUR divergence: compare amounts on same topic
        he_eurs = []
        for h in he_group:
            he_eurs.extend(_eur_from_json(h.get('eur_amounts')))
        ex_eurs_by_stmt: dict[str, list[float]] = defaultdict(list)
        for e in ex_group:
            eurs = _eur_from_json(e.get('eur_amounts'))
            if eurs:
                ex_eurs_by_stmt[e.get('statement_id', '')].extend(eurs)

        if he_eurs:
            he_max = max(he_eurs)
            he_min = min(he_eurs)
            for stmt_id, ex_eurs in ex_eurs_by_stmt.items():
                for ex_eur in ex_eurs:
                    # Significant divergence: >50% difference
                    if he_max > 0 and abs(ex_eur - he_max) / he_max > 0.5:
                        # Find the specific sentences
                        he_sent = next((h for h in he_group if _eur_from_json(h.get('eur_amounts'))), None)
                        ex_sent = next((e for e in ex_group
                                        if e.get('statement_id') == stmt_id
                                        and _eur_from_json(e.get('eur_amounts'))), None)
                        if he_sent and ex_sent:
                            matches.append({
                                'match_type': 'EUR_DIVERGENCE',
                                'topic': topic,
                                'he_eur': he_max,
                                'expert_eur': ex_eur,
                                'ratio': ex_eur / he_max if he_max else 0,
                                'he_text': he_sent['sent_text'][:200],
                                'he_role': he_sent['role'],
                                'he_quality': he_sent['quality'],
                                'expert_text': ex_sent['sent_text'][:200],
                                'expert_role': ex_sent['role'],
                                'expert_quality': ex_sent['quality'],
                                'expert_statement_id': ex_sent.get('statement_id', ''),
                                'expert_name': ex_sent.get('expert_name', ''),
                            })
                            break  # one divergence per statement per topic

        # Concern vs claim: expert concern on topic where HE has claim
        he_claims = [h for h in he_group if h['role'] in ('claim', 'estimate')]
        ex_concerns = [e for e in ex_group if e['role'] in ('concern', 'fact_claim')]

        if he_claims and ex_concerns:
            # Expert raises concern on a topic HE claims about
            for ec in ex_concerns[:3]:  # limit per topic
                best_he = he_claims[0]
                # Check quality mismatch: expert grounded vs HE asserted = contradiction signal
                he_weak = best_he['quality'] in ('asserted', 'hedged')
                ex_strong = ec['quality'] in ('grounded', 'modeled')
                mtype = 'CONTRADICTION' if (he_weak and ex_strong) else 'TENSION'
                matches.append({
                    'match_type': mtype,
                    'topic': topic,
                    'he_text': best_he['sent_text'][:200],
                    'he_role': best_he['role'],
                    'he_quality': best_he['quality'],
                    'expert_text': ec['sent_text'][:200],
                    'expert_role': ec['role'],
                    'expert_quality': ec['quality'],
                    'expert_statement_id': ec.get('statement_id', ''),
                    'expert_name': ec.get('expert_name', ''),
                })

        # Corroboration: expert references/grounded claims supporting HE claims
        ex_support = [e for e in ex_group
                      if e['role'] in ('reference', 'fact_claim', 'position')
                      and e['quality'] in ('grounded', 'modeled')]
        if he_claims and ex_support:
            for es in ex_support[:2]:
                matches.append({
                    'match_type': 'CORROBORATION',
                    'topic': topic,
                    'he_text': he_claims[0]['sent_text'][:200],
                    'he_role': he_claims[0]['role'],
                    'he_quality': he_claims[0]['quality'],
                    'expert_text': es['sent_text'][:200],
                    'expert_role': es['role'],
                    'expert_quality': es['quality'],
                    'expert_statement_id': es.get('statement_id', ''),
                    'expert_name': es.get('expert_name', ''),
                })

    return matches


# ---------------------------------------------------------------------------
# DB
# ---------------------------------------------------------------------------

def ensure_db_tables(db_path: Path) -> None:
    conn = sqlite3.connect(str(db_path))
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS cross_doc_matches (
            he_id               TEXT NOT NULL,
            match_type          TEXT NOT NULL,
            topic               TEXT,
            he_text             TEXT,
            he_role             TEXT,
            he_quality          TEXT,
            expert_text         TEXT,
            expert_role         TEXT,
            expert_quality      TEXT,
            expert_statement_id TEXT,
            expert_name         TEXT,
            he_eur              REAL,
            expert_eur          REAL,
            detail              TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_cdm_he ON cross_doc_matches(he_id);
        CREATE INDEX IF NOT EXISTS idx_cdm_type ON cross_doc_matches(match_type);
    """)
    conn.close()


def load_he_tags(db_path: Path, he_id: str) -> list[dict]:
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            "SELECT sent_text, role, quality, "
            "COALESCE((SELECT topic FROM sentence_tag s2 WHERE s2.he_id=s.he_id AND s2.atom_id=s.atom_id AND s2.sent_idx=s.sent_idx), 'X') as topic, "
            "eur_amounts FROM sentence_tag s WHERE he_id=?",
            (he_id,)
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()
    return [{'sent_text': r[0], 'role': r[1], 'quality': r[2], 'topic': r[3], 'eur_amounts': r[4]} for r in rows]


def load_lausunto_tag(db_path: Path, he_id: str) -> list[dict]:
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            "SELECT sent_text, role, quality, topic, eur_amounts, statement_id, expert_name "
            "FROM lausunto_tag WHERE he_id=?",
            (he_id,)
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()
    return [
        {'sent_text': r[0], 'role': r[1], 'quality': r[2], 'topic': r[3],
         'eur_amounts': r[4], 'statement_id': r[5], 'expert_name': r[6]}
        for r in rows
    ]


def write_matches(db_path: Path, he_id: str, matches: list[dict]) -> int:
    conn = sqlite3.connect(str(db_path))
    conn.execute("DELETE FROM cross_doc_matches WHERE he_id=?", (he_id,))
    rows = []
    for m in matches:
        detail = json.dumps({k: v for k, v in m.items()
                             if k not in ('match_type', 'topic', 'he_text', 'he_role',
                                          'he_quality', 'expert_text', 'expert_role',
                                          'expert_quality', 'expert_statement_id',
                                          'expert_name', 'he_eur', 'expert_eur')},
                            ensure_ascii=False) if m else None
        rows.append((
            he_id, m['match_type'], m.get('topic', ''),
            m.get('he_text', ''), m.get('he_role', ''), m.get('he_quality', ''),
            m.get('expert_text', ''), m.get('expert_role', ''), m.get('expert_quality', ''),
            m.get('expert_statement_id', ''), m.get('expert_name', ''),
            m.get('he_eur'), m.get('expert_eur'),
            detail,
        ))
    conn.executemany(
        "INSERT INTO cross_doc_matches "
        "(he_id, match_type, topic, he_text, he_role, he_quality, "
        "expert_text, expert_role, expert_quality, expert_statement_id, expert_name, "
        "he_eur, expert_eur, detail) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        rows
    )
    conn.commit()
    conn.close()
    return len(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_he(he_id: str, write_db: bool) -> dict:
    he_tags = load_he_tags(ENRICHMENTS_DB, he_id)
    lau_tags = load_lausunto_tag(ENRICHMENTS_DB, he_id)

    if not he_tags:
        return {'he_id': he_id, 'status': 'no_he_tags'}
    if not lau_tags:
        return {'he_id': he_id, 'status': 'no_lausunto_tag'}

    matches = find_matches(he_tags, lau_tags)

    by_type = defaultdict(int)
    for m in matches:
        by_type[m['match_type']] += 1

    print(f"{he_id}: {len(he_tags)} HE tags, {len(lau_tags)} lausunto tags → {len(matches)} matches")
    for mtype, n in sorted(by_type.items()):
        print(f"  {mtype}: {n}")

    # Show top findings
    for m in matches[:5]:
        print(f"  [{m['match_type']}] {m.get('topic','?')}: "
              f"HE({m.get('he_role','?')}/{m.get('he_quality','?')}): {m.get('he_text','')[:60]}")
        print(f"    vs Expert({m.get('expert_role','?')}/{m.get('expert_quality','?')}): "
              f"{m.get('expert_text','')[:60]}")

    if write_db and matches:
        ensure_db_tables(ENRICHMENTS_DB)
        n = write_matches(ENRICHMENTS_DB, he_id, matches)
        print(f"  → {n} matches written")

    return {'he_id': he_id, 'n_matches': len(matches), 'by_type': dict(by_type)}


def run(he_id: str | None = None, write_db: bool = False) -> dict:
    """Programmatic entry point. Returns summary dict."""
    if he_id:
        hid = he_id if he_id.startswith('he-') else f'he-{he_id}'
        result = process_he(hid, write_db)
        return result or {}

    # Find HEs that have BOTH sentence_tag AND lausunto_tag
    conn = sqlite3.connect(str(ENRICHMENTS_DB))
    try:
        he_with_tags = set(r[0] for r in conn.execute(
            "SELECT DISTINCT he_id FROM sentence_tag").fetchall())
        he_with_lau = set(r[0] for r in conn.execute(
            "SELECT DISTINCT he_id FROM lausunto_tag").fetchall())
    except sqlite3.OperationalError:
        print("No sentence_tag or lausunto_tag tables found. Run taggers first.")
        return {'n_processed': 0}
    finally:
        conn.close()

    both = sorted(he_with_tags & he_with_lau)
    print(f"HEs with both HE + lausunto tags: {len(both)}")
    results = []
    for hid in both:
        r = process_he(hid, write_db)
        if r:
            results.append(r)
    return {'n_processed': len(results)}


def main(args=None) -> None:
    if args is None:
        parser = argparse.ArgumentParser()
        parser.add_argument('--he', help='Specific HE')
        parser.add_argument('--write-db', action='store_true')
        args = parser.parse_args()

    run(he_id=args.he, write_db=args.write_db)


if __name__ == '__main__':
    main()

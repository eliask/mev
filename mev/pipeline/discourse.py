"""Unified discourse graph: cross-document claim matching across all legislative texts.

Models the legislative process as a discourse tree rooted in HE claims:
  HE (proposal) → Lausunto (expert response) → Mietinto (committee) → PTK (debate)

Each sentence/paragraph across all document types becomes a discourse_node with
unified dimensions. Edges represent response relationships (contradicts, corroborates,
addresses, echoes, ignores).

Join keys: topic code (coarse) + EUR amounts (fine) + § references (fine).
Temporal order constrains matching direction: experts respond to HE, not vice versa.

Reads from: he_enrichments.db [sentence_tag, lausunto_tag, mietinto_tag, ptk_speech_tag]
Writes to: he_enrichments.db [discourse_node, discourse_edge]

Usage:
    mev detect discourse --he he-241-2020 --write-db
    mev detect discourse                              # all HEs with >=2 tag tables populated
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import defaultdict
from pathlib import Path

from mev.config import ENRICHMENTS_DB

# ---------------------------------------------------------------------------
# EUR + § reference extraction (anchors for fine-grained matching)
# ---------------------------------------------------------------------------

_EUR_RE = re.compile(
    r'([\d,.]+)\s*(milj(?:ard[ia]|\.)?|mrd\.?|miljoon\w+)\s*(?:euroa|€)',
    re.IGNORECASE
)
_SECTION_RE = re.compile(r'(\d+)\s*(?:[a-z])?\s*§')


def _extract_eurs(text: str) -> list[float]:
    amounts = []
    for m in _EUR_RE.finditer(text):
        num_str = m.group(1).replace(' ', '').replace(',', '.')
        try:
            num = float(num_str)
        except ValueError:
            continue
        mult = m.group(2).lower()
        if 'mrd' in mult or 'miljard' in mult:
            amounts.append(num * 1e9)
        else:
            amounts.append(num * 1e6)
    return amounts


def _extract_section_refs(text: str) -> list[str]:
    return list(set(_SECTION_RE.findall(text)))


# ---------------------------------------------------------------------------
# Unified node schema
# ---------------------------------------------------------------------------

# Map all document-specific roles to a unified role vocabulary
_ROLE_MAP = {
    # HE sentence roles
    'premise': 'describes', 'estimate': 'claims', 'claim': 'claims',
    'caveat': 'qualifies', 'promise': 'commits',
    # Lausunto roles
    'position': 'positions', 'fact_claim': 'claims', 'concern': 'concerns',
    'amendment': 'proposes', 'reference': 'references', 'describes': 'describes',
    # Mietinto roles
    'endorses': 'endorses', 'modifies': 'proposes', 'rejects': 'concerns',
    'notes': 'describes', 'describes': 'describes',
    # PTK roles (stance-based)
    'support': 'endorses', 'oppose': 'concerns',
    'procedural': 'describes', 'neutral': 'describes',
}

# Unified quality: normalize across doc types
_QUAL_MAP = {
    'grounded': 'grounded', 'modeled': 'modeled', 'asserted': 'asserted',
    'hedged': 'hedged', 'uncertain': 'uncertain',
    'references_expert': 'references', 'cites_law': 'grounded',
    # PTK content types → quality proxy
    'concrete_concern': 'grounded', 'expert_reference': 'references',
    'rhetoric': 'asserted', 'other': 'asserted',
}


def _unify_role(role: str) -> str:
    return _ROLE_MAP.get(role, role or 'unknown')


def _unify_quality(qual: str) -> str:
    return _QUAL_MAP.get(qual, qual or 'unknown')


# ---------------------------------------------------------------------------
# Load from all tag tables
# ---------------------------------------------------------------------------

def _load_he_nodes(conn: sqlite3.Connection, he_id: str) -> list[dict]:
    try:
        rows = conn.execute(
            "SELECT sent_text, role, quality, eur_amounts, 'he' as doc_type, "
            "atom_id as doc_id, sent_idx, he_id, topic "
            "FROM sentence_tag WHERE he_id=?", (he_id,)
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    nodes = []
    for r in rows:
        nodes.append({
            'text': r[0] or '', 'role': r[1], 'quality': r[2],
            'eur_amounts': r[3], 'doc_type': 'he',
            'doc_id': r[5], 'sent_idx': r[6], 'he_id': r[7],
            'source_name': 'HE',
            'topic': r[8] or 'X',
        })
    return nodes


def _load_lausunto_nodes(conn: sqlite3.Connection, he_id: str) -> list[dict]:
    try:
        rows = conn.execute(
            "SELECT sent_text, role, quality, topic, eur_amounts, "
            "statement_id, sent_idx, expert_name, "
            "COALESCE(is_government, 0) "
            "FROM lausunto_tag WHERE he_id=?", (he_id,)
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    nodes = []
    for r in rows:
        role = r[1]
        # Skip D (describes/restates) — these are HE restatements, not expert substance
        if role == 'describes':
            continue
        is_gov = bool(r[8])
        nodes.append({
            'text': r[0] or '', 'role': role, 'quality': r[2], 'topic': r[3] or 'X',
            'eur_amounts': r[4], 'doc_type': 'lausunto_gov' if is_gov else 'lausunto',
            'doc_id': r[5], 'sent_idx': r[6], 'he_id': he_id,
            'source_name': r[7] or '',
            'is_government': is_gov,
        })
    return nodes


def _load_mietinto_nodes(conn: sqlite3.Connection, he_id: str) -> list[dict]:
    try:
        rows = conn.execute(
            "SELECT para_text, role, quality, topic, "
            "report_id, para_idx, committee "
            "FROM mietinto_tag WHERE he_id=?", (he_id,)
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    return [{
        'text': r[0] or '', 'role': r[1], 'quality': r[2], 'topic': r[3] or 'X',
        'eur_amounts': None, 'doc_type': 'mietinto',
        'doc_id': r[4], 'sent_idx': r[5], 'he_id': he_id,
        'source_name': r[6] or '',
    } for r in rows]


def _load_ptk_nodes(conn: sqlite3.Connection, he_id: str) -> list[dict]:
    try:
        rows = conn.execute(
            "SELECT he_id, speech_rowid, speaker, party, stance, content, topic "
            "FROM ptk_speech_tag WHERE he_id=?", (he_id,)
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    # PTK is speech-level not sentence-level; use stance as role, content as quality
    return [{
        'text': '',  # PTK text is in per-HE DB, not in enrichments
        'role': r[4], 'quality': r[5], 'topic': r[6] or 'X',
        'eur_amounts': None, 'doc_type': 'ptk',
        'doc_id': str(r[1]), 'sent_idx': 0, 'he_id': he_id,
        'source_name': f"{r[2] or ''} ({r[3] or ''})",
    } for r in rows]


# ---------------------------------------------------------------------------
# Matching engine
# ---------------------------------------------------------------------------

# Causal chain order: which doc types respond to which
RESPONSE_CHAIN = [
    ('he', 'lausunto'),       # experts respond to HE
    ('lausunto', 'mietinto'), # committee responds to experts
    ('lausunto', 'ptk'),      # MPs echo expert concerns
    ('he', 'mietinto'),       # committee evaluates HE
    ('he', 'ptk'),            # MPs debate HE
]


def _eur_from_json(s: str | None) -> list[float]:
    if not s:
        return []
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return []


def _compute_confidence(n1: dict, n2: dict) -> float:
    """Confidence that two nodes are about the same thing. 0.0-1.0."""
    conf = 0.0

    # Topic match (coarse)
    if n1.get('topic') and n2.get('topic') and n1['topic'] == n2['topic'] and n1['topic'] != 'X':
        conf += 0.35

    # EUR overlap (fine)
    e1 = _eur_from_json(n1.get('eur_amounts'))
    e2 = _eur_from_json(n2.get('eur_amounts'))
    if e1 and e2:
        # Any EUR amount within 2x of each other
        for a in e1:
            for b in e2:
                if a > 0 and b > 0 and 0.5 < a/b < 2.0:
                    conf += 0.3
                    break
            else:
                continue
            break

    # § reference overlap (fine)
    refs1 = set(_extract_section_refs(n1.get('text', '')))
    refs2 = set(_extract_section_refs(n2.get('text', '')))
    if refs1 and refs2 and refs1 & refs2:
        conf += 0.25

    # Keyword overlap (weak signal)
    words1 = set(re.findall(r'\b[a-zäöå]{4,}\b', (n1.get('text', '')).lower()))
    words2 = set(re.findall(r'\b[a-zäöå]{4,}\b', (n2.get('text', '')).lower()))
    if words1 and words2:
        overlap = len(words1 & words2) / max(len(words1 | words2), 1)
        if overlap > 0.2:
            conf += 0.1

    return min(conf, 1.0)


def _classify_edge(source: dict, target: dict) -> str:
    """Classify the response relationship between two matched nodes."""
    s_role = _unify_role(source.get('role', ''))
    t_role = _unify_role(target.get('role', ''))
    s_qual = _unify_quality(source.get('quality', ''))
    t_qual = _unify_quality(target.get('quality', ''))

    # Expert concerns about HE claims
    if s_role == 'claims' and t_role == 'concerns':
        if t_qual in ('grounded', 'modeled') and s_qual in ('asserted', 'hedged'):
            return 'CONTRADICTS'
        return 'CHALLENGES'

    # Expert supports HE
    if s_role == 'claims' and t_role in ('endorses', 'claims') and t_qual in ('grounded', 'references'):
        return 'CORROBORATES'

    # Committee endorses
    if t_role == 'endorses':
        return 'ENDORSES'

    # Committee proposes modification
    if t_role == 'proposes':
        return 'MODIFIES'

    # Expert concern not addressed by committee
    if source.get('doc_type') == 'lausunto' and target.get('doc_type') == 'mietinto':
        if s_role == 'concerns' and t_role == 'describes':
            return 'ACKNOWLEDGED'

    # PTK echoes expert
    if source.get('doc_type') in ('lausunto', 'he') and target.get('doc_type') == 'ptk':
        if t_role == 'concerns':
            return 'ECHOES'

    return 'RESPONDS_TO'


def build_discourse_graph(he_id: str) -> tuple[list[dict], list[dict]]:
    """Build discourse nodes + edges for one HE. Returns (nodes, edges)."""
    conn = sqlite3.connect(str(ENRICHMENTS_DB))

    # Load all nodes from all doc types
    all_nodes = []
    all_nodes.extend(_load_he_nodes(conn, he_id))
    all_nodes.extend(_load_lausunto_nodes(conn, he_id))
    all_nodes.extend(_load_mietinto_nodes(conn, he_id))
    all_nodes.extend(_load_ptk_nodes(conn, he_id))
    conn.close()

    if not all_nodes:
        return [], []

    # Unify roles/quality
    for n in all_nodes:
        n['unified_role'] = _unify_role(n.get('role', ''))
        n['unified_quality'] = _unify_quality(n.get('quality', ''))
        # Extract anchors
        n['_eurs'] = _eur_from_json(n.get('eur_amounts'))
        n['_refs'] = _extract_section_refs(n.get('text', ''))

    # Group by doc_type and topic
    by_type_topic: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for i, n in enumerate(all_nodes):
        n['_idx'] = i
        by_type_topic[(n['doc_type'], n.get('topic', 'X'))].append(n)

    # Match along causal chain
    edges = []
    for src_type, tgt_type in RESPONSE_CHAIN:
        # For each topic, match src→tgt
        all_topics = set(t for (dt, t) in by_type_topic if dt == src_type) | \
                     set(t for (dt, t) in by_type_topic if dt == tgt_type)

        for topic in all_topics:
            sources = by_type_topic.get((src_type, topic), [])
            targets = by_type_topic.get((tgt_type, topic), [])
            if not sources or not targets:
                continue

            # Match: for each target, find best source by confidence
            for tgt in targets:
                best_src = None
                best_conf = 0.0
                for src in sources:
                    conf = _compute_confidence(src, tgt)
                    if conf > best_conf:
                        best_conf = conf
                        best_src = src

                if best_src and best_conf >= 0.3:
                    edge_type = _classify_edge(best_src, tgt)
                    edges.append({
                        'source_idx': best_src['_idx'],
                        'target_idx': tgt['_idx'],
                        'source_type': src_type,
                        'target_type': tgt_type,
                        'topic': topic,
                        'edge_type': edge_type,
                        'confidence': round(best_conf, 2),
                        'source_text': best_src.get('text', ''),
                        'target_text': tgt.get('text', ''),
                        'source_name': best_src.get('source_name', ''),
                        'target_name': tgt.get('source_name', ''),
                    })

    # Find UNADDRESSED: expert concerns with no matching mietintö node
    # Exclude government officials (ministry presentations are not external critique)
    expert_concerns = [n for n in all_nodes
                       if n['doc_type'] == 'lausunto'
                       and not n.get('is_government')
                       and n.get('unified_role') == 'concerns']
    addressed_experts = {e['target_idx'] for e in edges
                         if e['source_type'] == 'lausunto' and e['target_type'] == 'mietinto'}
    for ec in expert_concerns:
        if ec['_idx'] not in {e['source_idx'] for e in edges if e['source_type'] == 'lausunto'}:
            edges.append({
                'source_idx': ec['_idx'],
                'target_idx': -1,
                'source_type': 'lausunto',
                'target_type': 'mietinto',
                'topic': ec.get('topic', 'X'),
                'edge_type': 'UNADDRESSED',
                'confidence': 0.5,
                'source_text': ec.get('text', ''),
                'target_text': '',
                'source_name': ec.get('source_name', ''),
                'target_name': '',
            })

    return all_nodes, edges


# ---------------------------------------------------------------------------
# DB
# ---------------------------------------------------------------------------

def ensure_db_tables(db_path: Path) -> None:
    conn = sqlite3.connect(str(db_path))
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS discourse_node (
            he_id       TEXT NOT NULL,
            node_idx    INTEGER NOT NULL,
            doc_type    TEXT NOT NULL,
            doc_id      TEXT,
            sent_idx    INTEGER,
            text        TEXT,
            role        TEXT,
            quality     TEXT,
            topic       TEXT,
            unified_role TEXT,
            unified_quality TEXT,
            source_name TEXT,
            eur_amounts TEXT,
            section_refs TEXT,
            PRIMARY KEY (he_id, node_idx)
        );
        CREATE INDEX IF NOT EXISTS idx_dn_he ON discourse_node(he_id);
        CREATE INDEX IF NOT EXISTS idx_dn_type ON discourse_node(doc_type);

        CREATE TABLE IF NOT EXISTS discourse_edge (
            he_id       TEXT NOT NULL,
            source_idx  INTEGER,
            target_idx  INTEGER,
            source_type TEXT,
            target_type TEXT,
            topic       TEXT,
            edge_type   TEXT,
            confidence  REAL,
            source_text TEXT,
            target_text TEXT,
            source_name TEXT,
            target_name TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_de_he ON discourse_edge(he_id);
        CREATE INDEX IF NOT EXISTS idx_de_type ON discourse_edge(edge_type);
    """)
    conn.close()


def write_graph(db_path: Path, he_id: str, nodes: list[dict], edges: list[dict]) -> tuple[int, int]:
    conn = sqlite3.connect(str(db_path))
    conn.execute("DELETE FROM discourse_node WHERE he_id=?", (he_id,))
    conn.execute("DELETE FROM discourse_edge WHERE he_id=?", (he_id,))

    node_rows = []
    for n in nodes:
        node_rows.append((
            he_id, n['_idx'], n['doc_type'], n.get('doc_id', ''),
            n.get('sent_idx', 0), n.get('text', '')[:500],
            n.get('role', ''), n.get('quality', ''), n.get('topic', 'X'),
            n.get('unified_role', ''), n.get('unified_quality', ''),
            n.get('source_name', ''),
            json.dumps(n.get('_eurs', [])) if n.get('_eurs') else None,
            json.dumps(n.get('_refs', [])) if n.get('_refs') else None,
        ))
    conn.executemany(
        "INSERT INTO discourse_node VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        node_rows)

    edge_rows = []
    for e in edges:
        edge_rows.append((
            he_id, e['source_idx'], e['target_idx'],
            e['source_type'], e['target_type'], e.get('topic', ''),
            e['edge_type'], e['confidence'],
            e.get('source_text', ''), e.get('target_text', ''),
            e.get('source_name', ''), e.get('target_name', ''),
        ))
    conn.executemany(
        "INSERT INTO discourse_edge VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        edge_rows)

    conn.commit()
    conn.close()
    return len(node_rows), len(edge_rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_he(he_id: str, write_db: bool) -> dict:
    nodes, edges = build_discourse_graph(he_id)

    if not nodes:
        return {'he_id': he_id, 'status': 'no_data'}

    by_type = defaultdict(int)
    for n in nodes:
        by_type[n['doc_type']] += 1

    edge_types = defaultdict(int)
    for e in edges:
        edge_types[e['edge_type']] += 1

    print(f"\n{he_id}: {len(nodes)} nodes ({', '.join(f'{t}={n}' for t,n in sorted(by_type.items()))})")
    print(f"  {len(edges)} edges ({', '.join(f'{t}={n}' for t,n in sorted(edge_types.items()))})")

    # Show top findings
    contradictions = [e for e in edges if e['edge_type'] in ('CONTRADICTS', 'CHALLENGES')]
    unaddressed = [e for e in edges if e['edge_type'] == 'UNADDRESSED']

    if contradictions:
        print(f"\n  CONTRADICTIONS/CHALLENGES ({len(contradictions)}):")
        for e in contradictions[:3]:
            print(f"    {e['source_type']}→{e['target_type']} [{e['topic']}] conf={e['confidence']}")
            print(f"      {e['source_text'][:80]}")
            print(f"      vs {e['target_text'][:80]}")

    if unaddressed:
        print(f"\n  UNADDRESSED expert concerns ({len(unaddressed)}):")
        for e in unaddressed[:3]:
            print(f"    [{e['topic']}] {e['source_name']}: {e['source_text'][:80]}")

    if write_db:
        ensure_db_tables(ENRICHMENTS_DB)
        nn, ne = write_graph(ENRICHMENTS_DB, he_id, nodes, edges)
        print(f"\n  → {nn} nodes, {ne} edges written")

    return {'he_id': he_id, 'n_nodes': len(nodes), 'n_edges': len(edges),
            'edge_types': dict(edge_types)}


def run(he_id: str | None = None, write_db: bool = False) -> dict:
    """Programmatic entry point. Returns summary dict."""
    if he_id:
        hid = he_id if he_id.startswith('he-') else f'he-{he_id}'
        result = process_he(hid, write_db)
        return result or {}

    # Find HEs with data in at least 2 tag tables
    conn = sqlite3.connect(str(ENRICHMENTS_DB))
    tables: dict[str, set] = {
        'sentence_tag': set(),
        'lausunto_tag': set(),
        'mietinto_tag': set(),
        'ptk_speech_tag': set(),
    }
    for tbl in tables:
        try:
            tables[tbl] = {r[0] for r in conn.execute(f"SELECT DISTINCT he_id FROM {tbl}").fetchall()}
        except sqlite3.OperationalError:
            pass
    conn.close()

    all_hes: set = set()
    for s in tables.values():
        all_hes |= s
    candidates = {h for h in all_hes if sum(1 for s in tables.values() if h in s) >= 2}

    print(f"HEs with >=2 tagged doc types: {len(candidates)}")
    results = []
    for hid in sorted(candidates):
        r = process_he(hid, write_db)
        if r:
            results.append(r)
    return {'n_processed': len(results)}


def main(args=None):
    if args is None:
        parser = argparse.ArgumentParser()
        parser.add_argument('--he', help='Specific HE')
        parser.add_argument('--write-db', action='store_true')
        args = parser.parse_args()

    run(he_id=args.he, write_db=args.write_db)


if __name__ == '__main__':
    main()

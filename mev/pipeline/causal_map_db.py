#!/usr/bin/env python3
"""
Build state_causal_map.db — single authoritative build from raw CSV sources.

Reads (all under data/statute_graph/):
  nodes.csv                      Base statute graph (metadata + degree)
  edges.csv                      Cross-reference edges
  statute_titles.csv             Law titles (extracted from Finlex ZIP)
  all_momentit_complete.csv      Budget momentit (state, 2024 actuals)
  momentti_statute_mapping.csv   Momentti -> statute mapping
  tae/tae_texts/*.txt            TAE explanatory texts per momentti
  public_sector_cofog_2023.csv   COFOG expenditure (full public sector, 2023)
  cofog_statute_mapping.csv      COFOG -> statute mapping
Also reads (under data/statute_graph/, if available):
  revenue_statute_mapping.csv    Revenue supplement (kunta taxes, service fees)
Also reads (under data/hankinnat/, if available):
  procurement_by_ministry.csv    Procurement EUR per ministry per year
  procurement_by_category.csv    Procurement EUR per hankintakategoria per year
Also reads (under data/statute_graph/, if available):
  institutions.csv                  Institutions (pension funds etc.) with AUM
  institution_statute_mapping.csv   Institution -> statute edges (governed_by, operates_under, etc.)

Computes:
  - Momentti-based budget per statute (direct aggregation)
  - COFOG-based budget per statute (weighted by Kela/ETK data where available, else equal split)
  - Revenue supplement per statute (municipal taxes, service fees)
  - Institution AUM per statute (governance exposure from pension funds etc.)
  - Combined budget: max(momentti, cofog, revenue, procurement) + institution per statute
  - PageRank propagation on COMBINED budget (full public sector scope)

Produces:
  data/statute_graph/state_causal_map.db
  → upload to vs.kunnas.com/fi_state_causal_map.db

Usage:
    mev build causal-map
"""

import argparse
import csv
import json
import re
import sqlite3
from collections import defaultdict
from pathlib import Path

from mev.config import ROOT, CAUSAL_MAP_DB as _CAUSAL_MAP_DB, HE_DB_DIR as _HE_DB_DIR, HE_INDEX_DB as _HE_INDEX_DB, ENRICHMENTS_DB as _ENRICHMENTS_DB
from mev.pipeline.aggregate_entity_data import main as _aggregate_entities

DATA_DIR = ROOT / 'data' / 'statute_graph'
DB_PATH = _CAUSAL_MAP_DB

# Mandatory delegation detection: "on annettava" / "säädetään" / "on säädettävä"
_MANDATORY_RE = re.compile(r'\b(säädetään|on annettava|on säädettävä)\b', re.I)


def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def compute_propagation(budget_direct, edges, alpha=0.3, iterations=20):
    """PageRank-style propagation, out-degree normalized.

    R(v) = budget(v) + alpha * sum(R(u) / outdeg(u) for u referencing v)
    """
    refs_to = defaultdict(list)
    out_degree = defaultdict(int)
    for source, target in edges:
        refs_to[target].append(source)
        out_degree[source] += 1

    all_ids = set(budget_direct.keys())
    for s, t in edges:
        all_ids.add(s)
        all_ids.add(t)

    R = {sid: budget_direct.get(sid, 0.0) for sid in all_ids}

    for i in range(iterations):
        R_new = {}
        max_delta = 0.0
        for sid in all_ids:
            base = budget_direct.get(sid, 0.0)
            prop = sum(R[src] / out_degree[src]
                       for src in refs_to.get(sid, [])
                       if out_degree[src] > 0)
            R_new[sid] = base + alpha * prop
            max_delta = max(max_delta, abs(R_new[sid] - R[sid]))
        R = R_new
        if max_delta < 1.0:
            print(f'    Converged at iteration {i+1} (delta={max_delta:.2f})')
            break

    return R


def compute_katz(budget_direct, edges, alpha=0.031, iterations=30):
    """Katz centrality — no out-degree normalization.

    katz(v) = direct(v) + alpha * sum(katz(u) for all u referencing v)

    Unlike PageRank, every citation carries full weight (not divided by
    out-degree). This rewards structural ubiquity — a law referenced by
    many budget-carrying laws accumulates their full load.

    alpha must be < 1/spectral_radius(A^T) for convergence.
    Empirical spectral radius ~27.04, so alpha=0.031 < 1/27.04 ≈ 0.037.
    """
    refs_to = defaultdict(list)
    for source, target in edges:
        refs_to[target].append(source)

    all_ids = set(budget_direct.keys())
    for s, t in edges:
        all_ids.add(s)
        all_ids.add(t)

    katz = {sid: budget_direct.get(sid, 0.0) for sid in all_ids}

    for i in range(iterations):
        katz_new = {}
        max_delta = 0.0
        for sid in all_ids:
            base = budget_direct.get(sid, 0.0)
            prop = sum(katz[src] for src in refs_to.get(sid, []))
            katz_new[sid] = base + alpha * prop
            max_delta = max(max_delta, abs(katz_new[sid] - katz[sid]))
        katz = katz_new
        if max_delta < 1.0:
            print(f'    Converged at iteration {i+1} (delta={max_delta:.2f})')
            break

    return katz


def _load_from_graph_artifact(graph_dir: Path):
    """Load titles, edges, amendment_parents, and census metrics from a lawvm build artifact.

    Returns (titles, edges_raw, amendment_parents, graph_census).
    graph_census: dict[sid -> {amendment_count, cites_in, cites_out, eu_ref_count,
                                stale_ref_count, stale_ref_pct,
                                unexercised_delegations, mandatory_unexercised}]
    """
    import json

    # Titles from statutes.json
    titles = {}
    statutes_path = graph_dir / 'statutes.json'
    if statutes_path.exists():
        with open(statutes_path, encoding='utf-8') as f:
            statutes_meta = json.load(f)
        for sid, meta in statutes_meta.items():
            if meta.get('title'):
                titles[sid] = meta['title']
        print(f'  [graph] {len(titles)} titles from statutes.json')

    # Edges from citations.jsonl
    edges_raw = []
    cite_path = graph_dir / 'citations.jsonl'
    if cite_path.exists():
        with open(cite_path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                edges_raw.append({
                    'source': d.get('source_statute_id', ''),
                    'target': d.get('target_statute_id', ''),
                    'edge_type': d.get('edge_type', ''),
                })
        print(f'  [graph] {len(edges_raw)} citation edges from citations.jsonl')

    # Amendment parents from amendments.json (inverted: parent→[amendments] → amendment→parent)
    amend_index: dict = {}
    amendment_parents = {}
    amend_path = graph_dir / 'amendments.json'
    if amend_path.exists():
        with open(amend_path, encoding='utf-8') as f:
            amend_index = json.load(f)
        for parent_id, amend_list in amend_index.items():
            for amend_id in amend_list:
                amendment_parents[amend_id] = parent_id
        print(f'  [graph] {len(amendment_parents)} amendment→parent links from amendments.json')

    # ── Census aggregates ──────────────────────────────────────────────────────
    # Amendment count + latest amendment year per statute (for stale-ref approximation)
    graph_amendment_count: dict = {}
    latest_amend_year: dict = {}
    for parent_id, amend_list in amend_index.items():
        graph_amendment_count[parent_id] = len(amend_list)
        years = [int(a.split('/')[0]) for a in amend_list if a.split('/')[0].isdigit()]
        if years:
            latest_amend_year[parent_id] = max(years)

    # Citation metrics: cites_in, cites_out, eu_ref_count, stale counts
    graph_cites_in: dict = defaultdict(int)
    graph_cites_out: dict = defaultdict(int)
    graph_eu_ref_count: dict = defaultdict(int)
    stale_ref_out: dict = defaultdict(int)
    total_fi_ref_out: dict = defaultdict(int)
    for edge in edges_raw:
        if edge.get('edge_type') != 'CITES':
            continue
        src = edge.get('source', '')
        tgt = edge.get('target', '')
        if not src or not tgt:
            continue
        if tgt.startswith('eu/'):
            graph_eu_ref_count[src] += 1
            continue
        graph_cites_out[src] += 1
        graph_cites_in[tgt] += 1
        total_fi_ref_out[src] += 1
        try:
            citing_year = int(src.split('/')[0])
        except (ValueError, IndexError):
            continue
        tgt_amend_yr = latest_amend_year.get(tgt)
        if tgt_amend_yr and citing_year < tgt_amend_yr:
            stale_ref_out[src] += 1

    # Delegation metrics from delegations.jsonl
    graph_deleg_total: dict = defaultdict(int)
    graph_deleg_mandatory: dict = defaultdict(int)
    deleg_path = graph_dir / 'delegations.jsonl'
    if deleg_path.exists():
        with open(deleg_path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                sid = d.get('statute_id', '')
                if sid:
                    graph_deleg_total[sid] += 1
                    if _MANDATORY_RE.search(d.get('match_text', '')):
                        graph_deleg_mandatory[sid] += 1

    # Statutes that have at least one ISSUED_UNDER child asetus (delegation exercised)
    has_issued_under: set = {
        edge.get('target', '')
        for edge in edges_raw
        if edge.get('edge_type') == 'ISSUED_UNDER' and edge.get('target')
    }

    # Assemble per-statute census dict
    all_census_sids = (set(graph_amendment_count) | set(graph_cites_in)
                       | set(graph_cites_out) | set(graph_deleg_total))
    graph_census: dict = {}
    for sid in all_census_sids:
        n_fi_out = total_fi_ref_out.get(sid, 0)
        n_stale = stale_ref_out.get(sid, 0)
        n_deleg = graph_deleg_total.get(sid, 0)
        exercised = sid in has_issued_under
        graph_census[sid] = {
            'amendment_count': graph_amendment_count.get(sid),
            'cites_in': graph_cites_in.get(sid, 0),
            'cites_out': graph_cites_out.get(sid, 0),
            'eu_ref_count': graph_eu_ref_count.get(sid, 0),
            'stale_ref_count': n_stale,
            'stale_ref_pct': n_stale / n_fi_out if n_fi_out > 0 else None,
            'unexercised_delegations': n_deleg if not exercised else 0,
            'mandatory_unexercised': graph_deleg_mandatory.get(sid, 0) if not exercised else 0,
        }

    stale_total = sum(1 for gc in graph_census.values() if gc.get('stale_ref_count', 0) > 0)
    print(f'  [graph] census: {len(graph_census)} statutes, '
          f'{stale_total} with stale refs, {len(has_issued_under)} with ISSUED_UNDER children')

    return titles, edges_raw, amendment_parents, graph_census


def build(graph_dir: Path | None = None):
    DB_PATH.unlink(missing_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # ========== LOAD ALL SOURCE DATA ==========

    # 1. Nodes (base metadata)
    print('Loading nodes.csv...')
    nodes = load_csv(DATA_DIR / 'nodes.csv')
    node_map = {n['id']: n for n in nodes}
    node_ids = set(node_map.keys())
    print(f'  {len(nodes)} statutes')

    def normalize_statute_id(sid_raw: str) -> str:
        """Normalize HE-style 'number/year' to graph-style 'year/number'.

        Uses node_ids lookup instead of heuristics — handles dash IDs
        (e.g. 39-001/1889 → 1889/39-001) correctly.
        """
        parts = sid_raw.split('/')
        if len(parts) != 2:
            return sid_raw
        as_is = sid_raw
        flipped = f'{parts[1]}/{parts[0]}'
        if as_is in node_ids:
            return as_is
        if flipped in node_ids:
            return flipped
        return flipped  # default: assume number/year → year/number

    graph_census: dict = {}
    if graph_dir is not None:
        print(f'Loading from lawvm graph artifact: {graph_dir}')
        titles, edges_raw, amendment_parents, graph_census = _load_from_graph_artifact(graph_dir)
    else:
        # 2. Titles
        titles_path = DATA_DIR / 'statute_titles.csv'
        titles = {}
        if titles_path.exists():
            print(f'Loading {titles_path.name}...')
            for r in load_csv(titles_path):
                titles[r['id']] = r['title']
            print(f'  {len(titles)} titles')

        # 3. Edges
        print('Loading edges.csv...')
        edges_raw = load_csv(DATA_DIR / 'edges.csv')
        print(f'  {len(edges_raw)} edges')

        # 3b. Amendment parents (muutoslaki → parent law)
        amend_parents_path = DATA_DIR / 'amendment_parents.csv'
        amendment_parents = {}
        if amend_parents_path.exists():
            print('Loading amendment_parents.csv...')
            amend_raw = load_csv(amend_parents_path)
            for r in amend_raw:
                amendment_parents[r['amendment_id']] = r['parent_id']
            print(f'  {len(amendment_parents)} amendment→parent links')

    edges = [(r['source'], r['target']) for r in edges_raw]
    # Extract REPEALS edges (repealer -> repealed)
    repeals_from_csv = set()
    for r in edges_raw:
        if r.get('edge_type') == 'REPEALS':
            repeals_from_csv.add((r['source'], r['target']))
    print(f'  {len(edges)} edges total, {len(repeals_from_csv)} REPEALS')

    # 3c. Editorial notes
    notes_path = DATA_DIR / 'editorial_notes.csv'
    editorial_notes = []
    if notes_path.exists():
        print('Loading editorial_notes.csv...')
        editorial_notes = load_csv(notes_path)
        print(f'  {len(editorial_notes)} editorial notes')

    # 4. Momentit + budget
    print('Loading all_momentit_complete.csv...')
    momentit_raw = load_csv(DATA_DIR / 'all_momentit_complete.csv')
    momentti_budget = {}
    for r in momentit_raw:
        try:
            momentti_budget[r['momentti_code']] = float(r['budget_2024_eur'])
        except (ValueError, KeyError):
            momentti_budget[r['momentti_code']] = 0.0
    print(f'  {len(momentit_raw)} momentit')

    # 5. Momentti -> statute mapping
    print('Loading momentti_statute_mapping.csv...')
    mapping_raw = load_csv(DATA_DIR / 'momentti_statute_mapping.csv')
    # Normalize reversed IDs
    normalized = 0
    for row in mapping_raw:
        old = row['statute_id']
        if old not in node_ids:
            parts = old.split('/')
            if len(parts) == 2:
                rev = f"{parts[1]}/{parts[0]}"
                if rev in node_ids:
                    row['statute_id'] = rev
                    normalized += 1
    print(f'  {len(mapping_raw)} mapping rows ({normalized} IDs normalized)')

    # 6. COFOG data
    cofog_path = DATA_DIR / 'public_sector_cofog_2023.csv'
    cofog_budget = {}  # code -> meur
    cofog_names = {}
    cofog_all_rows = []
    if cofog_path.exists():
        print(f'Loading {cofog_path.name}...')
        for r in load_csv(cofog_path):
            cofog_all_rows.append(r)
            if r['sector'] == 'S13' and len(r['cofog']) == 5:
                cofog_budget[r['cofog']] = float(r['meur_2023'])
                cofog_names[r['cofog']] = r['cofog_name']
        print(f'  {len(cofog_budget)} COFOG subcategories')

    # 7. COFOG -> statute mapping
    cofog_map_path = DATA_DIR / 'cofog_statute_mapping.csv'
    cofog_mapping = []
    if cofog_map_path.exists():
        print(f'Loading {cofog_map_path.name}...')
        cofog_mapping = load_csv(cofog_map_path)
        print(f'  {len(cofog_mapping)} COFOG mapping rows')

    # 8. TAE texts
    tae_dir = DATA_DIR / 'tae' / 'tae_texts'
    tae_texts = {}
    if tae_dir.exists():
        for txt_file in sorted(tae_dir.glob('*.txt')):
            tae_texts[txt_file.stem] = txt_file.read_text(encoding='utf-8')
    print(f'  {len(tae_texts)} TAE texts')

    # 9. Procurement data (per unit, from tutkihankintoja)
    HANKINNAT_DIR = ROOT / 'data' / 'hankinnat'
    proc_unit_path = HANKINNAT_DIR / 'procurement_by_unit.csv'
    proc_unit_map_path = HANKINNAT_DIR / 'procurement_unit_statute_mapping.csv'
    procurement_by_unit = {}  # unit_name -> eur (latest year, state only)
    procurement_unit_mapping = []  # unit -> statute edges
    proc_year = None
    if proc_unit_path.exists():
        print(f'Loading {proc_unit_path.name}...')
        proc_data = load_csv(proc_unit_path)
        # Use latest full year with >3B state procurement
        import datetime
        current_year = datetime.date.today().year
        years_seen = sorted(set(int(r['year']) for r in proc_data))
        for y in reversed(years_seen):
            if y >= current_year:
                continue
            state_total = sum(float(r['eur']) for r in proc_data
                              if int(r['year']) == y and r['sektori'] == 'Valtio')
            if state_total > 3e9:
                proc_year = y
                break
        if proc_year:
            for r in proc_data:
                if int(r['year']) == proc_year and r['sektori'] == 'Valtio':
                    procurement_by_unit[r['hankintayksikko']] = float(r['eur'])
            total_proc = sum(procurement_by_unit.values())
            print(f'  Using {proc_year}: {len(procurement_by_unit)} state units, {total_proc/1e6:,.0f} M')
        else:
            print('  No full-year procurement data found')
    else:
        print('  No procurement_by_unit.csv (run aggregate_procurement.py first)')

    if proc_unit_map_path.exists():
        print(f'Loading {proc_unit_map_path.name}...')
        procurement_unit_mapping = load_csv(proc_unit_map_path)
        print(f'  {len(procurement_unit_mapping)} unit-statute mapping rows')

    # 11. Institutions (pension funds etc.) with AUM
    inst_path = DATA_DIR / 'institutions.csv'
    institutions = []
    if inst_path.exists():
        print(f'Loading {inst_path.name}...')
        institutions = load_csv(inst_path)
        print(f'  {len(institutions)} institutions, total AUM: '
              f'{sum(float(r["aum_meur"]) for r in institutions):,.0f} M EUR')

    inst_map_path = DATA_DIR / 'institution_statute_mapping.csv'
    inst_mapping = []
    if inst_map_path.exists():
        print(f'Loading {inst_map_path.name}...')
        inst_mapping = load_csv(inst_map_path)
        print(f'  {len(inst_mapping)} institution-statute edges')

    # 12. Delegation data (from LawVM delegation.py → delegation_summary.csv)
    deleg_summary_path = DATA_DIR / 'delegation_summary.csv'
    delegation_data = {}
    if deleg_summary_path.exists():
        print(f'Loading {deleg_summary_path.name}...')
        for r in load_csv(deleg_summary_path):
            delegation_data[r['statute_id']] = r
        print(f'  {len(delegation_data)} statutes with delegation data')

    deleg_detail_path = DATA_DIR / 'delegations.csv'
    delegation_rows = []
    if deleg_detail_path.exists():
        print(f'Loading {deleg_detail_path.name}...')
        delegation_rows = load_csv(deleg_detail_path)
        print(f'  {len(delegation_rows)} delegation clauses')

    asetus_auth_path = DATA_DIR / 'asetus_authority.csv'
    asetus_authority = []
    if asetus_auth_path.exists():
        print(f'Loading {asetus_auth_path.name}...')
        asetus_authority = load_csv(asetus_auth_path)
        print(f'  {len(asetus_authority)} asetus authority linkages')

    # ========== COMPUTE BUDGET AGGREGATIONS ==========

    # Momentti-based budget per statute (deduplicate mapping pairs)
    print('\nComputing momentti-based budget per statute...')
    CONF_ORDER = {'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}
    seen_pairs = {}  # (momentti_code, statute_id) -> best confidence
    for row in mapping_raw:
        key = (row['momentti_code'], row['statute_id'])
        conf = CONF_ORDER.get(row.get('confidence', 'LOW'), 1)
        if key not in seen_pairs or conf > seen_pairs[key]:
            seen_pairs[key] = conf
    deduped = len(mapping_raw) - len(seen_pairs)
    if deduped:
        print(f'  Deduplicated {deduped} duplicate (momentti, statute) pairs')

    statute_momentti = defaultdict(lambda: {'budget': 0.0, 'count': 0})
    for (code, sid) in seen_pairs:
        budget = momentti_budget.get(code, 0.0)
        statute_momentti[sid]['budget'] += abs(budget)
        statute_momentti[sid]['count'] += 1
    print(f'  {len(statute_momentti)} statutes with momentti budget')

    # COFOG-based budget per statute (weighted by Kela/ETK data where available)
    print('Computing COFOG-based budget per statute...')
    cofog_to_entries = defaultdict(list)  # cofog -> [(statute_id, weight_eur_m or None)]
    for row in cofog_mapping:
        w = row.get('weight_eur_m', '').strip()
        weight = float(w) if w else None
        cofog_to_entries[row['cofog']].append((row['statute_id'], weight))

    statute_cofog = defaultdict(float)
    weighted_categories = 0
    momentti_weighted_categories = 0
    equal_categories = 0
    for cofog_code, entries in cofog_to_entries.items():
        meur = cofog_budget.get(cofog_code, 0.0)
        if meur <= 0:
            continue
        valid = [(sid, w) for sid, w in entries if sid in node_ids]
        if not valid:
            continue

        # Check if any entry has explicit weight
        has_weights = any(w is not None for _, w in valid)
        if has_weights:
            # Weighted distribution: entries with weight get their exact amount,
            # entries without weight split the remainder equally
            weighted_sum = sum(w for _, w in valid if w is not None)
            unweighted = [(sid, w) for sid, w in valid if w is None]
            for sid, w in valid:
                if w is not None:
                    statute_cofog[sid] += w * 1e6
                # Entries without weight split the remainder
            remainder = max(0, meur - weighted_sum)
            if unweighted and remainder > 0:
                per_unweighted = (remainder * 1e6) / len(unweighted)
                for sid, _ in unweighted:
                    statute_cofog[sid] += per_unweighted
            weighted_categories += 1
        else:
            # Use momentti budget as intra-category weight if available,
            # otherwise equal distribution
            mom_weights = {sid: statute_momentti.get(sid, {}).get('budget', 0.0)
                           for sid, _ in valid}
            total_mom = sum(mom_weights.values())
            if total_mom > 0:
                # Distribute proportionally by momentti budget
                for sid, _ in valid:
                    share = mom_weights[sid] / total_mom
                    statute_cofog[sid] += share * meur * 1e6
                momentti_weighted_categories += 1
            else:
                per_statute_eur = (meur * 1e6) / len(valid)
                for sid, _ in valid:
                    statute_cofog[sid] += per_statute_eur
                equal_categories += 1

    print(f'  {len(statute_cofog)} statutes with COFOG budget')
    print(f'  {weighted_categories} categories with Kela/ETK weights, {momentti_weighted_categories} with momentti weights, {equal_categories} with equal split')

    # Procurement-based budget per statute (via unit -> statute mapping)
    print('Computing procurement-based budget per statute...')
    statute_procurement = defaultdict(float)
    if procurement_by_unit and procurement_unit_mapping:
        # Build unit -> [statute_ids] from mapping
        unit_to_statutes = defaultdict(list)
        for row in procurement_unit_mapping:
            unit_to_statutes[row['hankintayksikko']].append(row['statute_id'])

        mapped_eur = 0.0
        unmapped_eur = 0.0
        for unit, eur in procurement_by_unit.items():
            statutes = unit_to_statutes.get(unit, [])
            if statutes:
                per_statute = eur / len(statutes)
                for sid in statutes:
                    statute_procurement[sid] += per_statute
                mapped_eur += eur
            else:
                unmapped_eur += eur

        total_proc = mapped_eur + unmapped_eur
        print(f'  {len(statute_procurement)} statutes with procurement budget')
        print(f'  Mapped: {mapped_eur/1e6:,.0f}M ({mapped_eur/total_proc*100:.0f}%), '
              f'unmapped: {unmapped_eur/1e6:,.0f}M ({unmapped_eur/total_proc*100:.0f}%)')
        # Show top
        top_proc = sorted(statute_procurement.items(), key=lambda x: -x[1])[:5]
        for sid, eur in top_proc:
            print(f'    {sid:>12}  {eur/1e6:8,.0f}M  {titles.get(sid, "")[:40]}')

    # 10. Revenue supplement (municipal taxes, service fees not in state momentit/COFOG)
    revenue_supplement_path = DATA_DIR / 'revenue_statute_mapping.csv'
    statute_revenue = defaultdict(float)
    if revenue_supplement_path.exists():
        print(f'Loading {revenue_supplement_path.name}...')
        for r in load_csv(revenue_supplement_path):
            sid = r['statute_id']
            if sid in node_ids or True:  # include even if not in nodes (may be stub)
                statute_revenue[sid] += float(r['revenue_eur'])
        print(f'  {len(statute_revenue)} statutes with revenue supplement, '
              f'total: {sum(statute_revenue.values())/1e9:.1f}B EUR')
    else:
        print('  No revenue supplement (run build_revenue_supplement.py first)')

    # Institution AUM per statute (governance/operational exposure)
    # Each institution's FULL AUM propagates to all its connected statutes.
    # A governance flaw in any connected law puts the full AUM at risk.
    # For combined budget: convert stock (AUM) to annual flow equivalent
    # using per-institution payout rate from CSV (pension ~13%, endowment ~4%, etc.)
    print('Computing institution AUM per statute...')
    inst_aum = {r['id']: float(r['aum_meur']) * 1e6 for r in institutions}  # EUR
    inst_payout = {r['id']: float(r['payout_rate']) for r in institutions}
    statute_inst_aum = defaultdict(float)  # raw AUM exposure
    statute_inst_annual = defaultdict(float)  # annualized flow equivalent
    for row in inst_mapping:
        iid = row['institution_id']
        sid = row['statute_id']
        aum = inst_aum.get(iid, 0.0)
        rate = inst_payout[iid]
        if aum > 0:
            statute_inst_aum[sid] += aum
            statute_inst_annual[sid] += aum * rate
    if statute_inst_aum:
        print(f'  {len(statute_inst_aum)} statutes with institution AUM exposure')
        top_aum = sorted(statute_inst_aum.items(), key=lambda x: -x[1])[:5]
        for sid, aum in top_aum:
            annual = statute_inst_annual[sid]
            print(f'    {sid:>12}  AUM={aum/1e9:6.1f}B  annual={annual/1e9:5.1f}B  {titles.get(sid, "")[:40]}')

    # Combined budget: max(momentti, cofog, revenue, procurement) + institution annual
    # All flow sources use max() — same public money, different attribution layers:
    #   momentti = which policy law authorizes spending
    #   COFOG = which functional area the spending serves
    #   revenue = municipal taxes/fees not in state budget
    #   procurement = which organizational law governs the spender
    # They rarely land on the same statute, but when they do it's the same euro.
    # Institution AUM annual equivalent is additive (genuinely separate pool — pension assets).
    print('Computing combined budget...')
    all_budget_ids = (set(statute_momentti.keys()) | set(statute_cofog.keys()) |
                      set(statute_revenue.keys()) | set(statute_inst_annual.keys()) |
                      set(statute_procurement.keys()))
    budget_combined = {}
    for sid in all_budget_ids:
        # Zero out budget for repealed laws — they don't exert direct flow
        if node_map.get(sid, {}).get('in_force') == 'False':
            budget_combined[sid] = 0.0
            continue

        mom = statute_momentti[sid]['budget'] if sid in statute_momentti else 0.0
        cof = statute_cofog.get(sid, 0.0)
        rev = statute_revenue.get(sid, 0.0)
        inst = statute_inst_annual.get(sid, 0.0)
        proc = statute_procurement.get(sid, 0.0)
        # Flow sources: same money, different layer → max()
        # Institution AUM annual: separate pool → additive
        budget_combined[sid] = max(mom, cof, rev, proc) + inst
    total_combined = sum(budget_combined.values())
    print(f'  {len(budget_combined)} statutes, total combined: {total_combined/1e9:.1f}B EUR')

    # ========== SUCCESSOR REDIRECTION ==========
    # Build successor_map from REPEALS edges: if A repeals B, then B's successor is A
    # Exclude mass repealers (>5 repeals) — these are cleanup decrees, not functional successors
    repeal_count = defaultdict(int)
    for repealer, repealed in repeals_from_csv:
        repeal_count[repealer] += 1
    successor_map = {}
    mass_repealers = set()
    for repealer, repealed in repeals_from_csv:
        if repealed in node_ids and repealer in node_ids:
            if repeal_count[repealer] <= 5:
                successor_map[repealed] = repealer
            else:
                mass_repealers.add(repealer)
    if mass_repealers:
        print(f'  Excluded {len(mass_repealers)} mass repealers (>5 repeals each) from successor map')
    print(f'  Found {len(successor_map)} successor relationships (from REPEALS edges)')

    # Redirect edges: if B -> A and A is superseded by C, redirect to B -> C
    print('Redirecting edges to successors...')
    redirected_edges = []
    for s, t in edges:
        curr = t
        visited = {curr}
        while curr in successor_map:
            curr = successor_map[curr]
            if curr in visited: break # cycle
            visited.add(curr)
        redirected_edges.append((s, curr))

    # PageRank propagation on COMBINED budget
    print('Computing propagation (combined, alpha=0.3)...')
    R_combined = compute_propagation(budget_combined, redirected_edges)
    nonzero = sum(1 for v in R_combined.values() if v > 0)
    print(f'  {nonzero} statutes with nonzero systemic score')

    # Also compute state-only propagation for comparison
    print('Computing propagation (state-only, alpha=0.3)...')
    budget_state = {sid: (info['budget'] if node_map.get(sid, {}).get('in_force') != 'False' else 0.0)
                    for sid, info in statute_momentti.items()}
    R_state = compute_propagation(budget_state, redirected_edges)

    # Katz centrality on COMBINED budget (structural load — no out-degree normalization)
    print('Computing Katz centrality (combined, alpha=0.031)...')
    R_katz = compute_katz(budget_combined, redirected_edges)
    nonzero_katz = sum(1 for v in R_katz.values() if v > 0)
    print(f'  {nonzero_katz} statutes with nonzero Katz score')

    # ========== BOW-TIE DECOMPOSITION ==========
    # Tarjan SCC → classify every statute as CORE / IN / OUT / TENDRIL
    # Following Broder et al. (2000) and Csete & Doyle (2002) applied to legislation.

    print('\nBow-tie decomposition (Tarjan SCC)...')

    # Build adjacency lists
    adj_out = defaultdict(list)  # forward edges
    adj_in = defaultdict(list)   # reverse edges
    all_bt_nodes = set(node_ids)
    for s, t in redirected_edges:
        adj_out[s].append(t)
        adj_in[t].append(s)

    # Iterative Tarjan SCC (avoids recursion limit on 92K nodes)
    scc_index = {}
    scc_lowlink = {}
    scc_on_stack = {}
    scc_stack = []
    scc_counter = [0]
    sccs = []

    def tarjan_iterative(v):
        """Non-recursive Tarjan using explicit call stack."""
        work_stack = [(v, 0)]  # (node, neighbor_index)
        scc_index[v] = scc_lowlink[v] = scc_counter[0]
        scc_counter[0] += 1
        scc_stack.append(v)
        scc_on_stack[v] = True

        while work_stack:
            node, ni = work_stack[-1]
            neighbors = adj_out.get(node, [])
            if ni < len(neighbors):
                work_stack[-1] = (node, ni + 1)
                w = neighbors[ni]
                if w not in scc_index:
                    scc_index[w] = scc_lowlink[w] = scc_counter[0]
                    scc_counter[0] += 1
                    scc_stack.append(w)
                    scc_on_stack[w] = True
                    work_stack.append((w, 0))
                elif scc_on_stack.get(w, False):
                    scc_lowlink[node] = min(scc_lowlink[node], scc_index[w])
            else:
                # All neighbors processed
                if scc_lowlink[node] == scc_index[node]:
                    component = []
                    while True:
                        w = scc_stack.pop()
                        scc_on_stack[w] = False
                        component.append(w)
                        if w == node:
                            break
                    sccs.append(component)
                if work_stack:
                    work_stack.pop()
                    if work_stack:
                        parent = work_stack[-1][0]
                        scc_lowlink[parent] = min(scc_lowlink[parent], scc_lowlink[node])

    for v in all_bt_nodes:
        if v not in scc_index:
            tarjan_iterative(v)

    # Find the largest SCC = CORE
    sccs.sort(key=len, reverse=True)
    core_set = set(sccs[0]) if sccs else set()
    scc2_size = len(sccs[1]) if len(sccs) > 1 else 0

    print(f'  Largest SCC (CORE): {len(core_set)} statutes')
    print(f'  2nd largest SCC: {scc2_size} statutes')
    print(f'  Total SCCs: {len(sccs)}')

    # BFS reachability from CORE (forward = OUT reachable, backward = IN reachable)
    def bfs_reach(start_set, adj):
        visited = set(start_set)
        frontier = list(start_set)
        while frontier:
            batch = frontier
            frontier = []
            for node in batch:
                for nb in adj.get(node, []):
                    if nb not in visited:
                        visited.add(nb)
                        frontier.append(nb)
        return visited

    reachable_from_core = bfs_reach(core_set, adj_out)   # CORE can reach these (forward)
    reaches_core = bfs_reach(core_set, adj_in)            # These can reach CORE (backward)

    # Classify
    bowtie_zone = {}
    counts = defaultdict(int)
    for sid in all_bt_nodes:
        if sid in core_set:
            zone = 'CORE'
        elif sid in reaches_core and sid not in reachable_from_core:
            zone = 'IN'
        elif sid in reachable_from_core and sid not in reaches_core:
            zone = 'OUT'
        elif sid in reachable_from_core and sid in reaches_core:
            zone = 'CORE'  # reachable both ways through non-core path → treat as TUBE (fold into CORE for simplicity)
        else:
            zone = 'TENDRIL'
        bowtie_zone[sid] = zone
        counts[zone] += 1

    for zone in ['CORE', 'IN', 'OUT', 'TENDRIL']:
        print(f'  {zone}: {counts[zone]} statutes')

    # Budget distribution per zone
    for zone in ['CORE', 'IN', 'OUT', 'TENDRIL']:
        zone_budget = sum(budget_combined.get(sid, 0) for sid in all_bt_nodes if bowtie_zone.get(sid) == zone)
        zone_katz = sum(R_katz.get(sid, 0) for sid in all_bt_nodes if bowtie_zone.get(sid) == zone)
        print(f'    {zone} direct budget: {zone_budget/1e9:.1f}B, Katz sum: {zone_katz/1e9:.1f}B')

    # ========== EDGE TYPING (hierarchical) ==========
    # Classify each edge by the type_statute of its endpoints.
    # This is the cheapest possible edge typing — no text parsing needed.
    # Types: PEER (Laki→Laki), IMPL (Laki→Asetus), AUTH (Asetus→Laki),
    #        CONST (to/from constitutional), HIER (other hierarchical)

    print('\nEdge typing (hierarchical, from type_statute)...')

    LAKI_TYPES = {'Laki'}
    ASETUS_TYPES = {'Asetus'}
    PAATOS_TYPES = {'Päätös', 'Määräys', 'Ohje'}  # executive decisions

    def classify_edge(src_type, tgt_type):
        """Classify edge by source→target document type pair."""
        if src_type in LAKI_TYPES and tgt_type in LAKI_TYPES:
            return 'PEER'        # Laki↔Laki: horizontal policy entanglement
        elif src_type in LAKI_TYPES and tgt_type in ASETUS_TYPES:
            return 'IMPL_DOWN'   # Laki→Asetus: implementation delegation
        elif src_type in ASETUS_TYPES and tgt_type in LAKI_TYPES:
            return 'AUTH_UP'     # Asetus→Laki: authority basis
        elif src_type in ASETUS_TYPES and tgt_type in ASETUS_TYPES:
            return 'ASETUS_PEER' # Asetus↔Asetus: decree cross-ref
        elif src_type in PAATOS_TYPES or tgt_type in PAATOS_TYPES:
            return 'EXEC'        # Involves executive decisions
        else:
            return 'OTHER'

    # Extract ISSUED_UNDER edges from CSV
    issued_under_from_csv = set()
    for r in edges_raw:
        if r.get('edge_type') == 'ISSUED_UNDER':
            issued_under_from_csv.add((r['source'], r['target']))
    print(f'  {len(issued_under_from_csv)} ISSUED_UNDER edges from CSV')

    edge_types = {}  # (source, target) -> type
    edge_type_counts = defaultdict(int)
    for s, t in redirected_edges:
        if (s, t) in repeals_from_csv:
            etype = 'REPEALS'
        elif (s, t) in issued_under_from_csv:
            etype = 'ISSUED_UNDER'
        else:
            src_type = node_map.get(s, {}).get('type_statute', '')
            tgt_type = node_map.get(t, {}).get('type_statute', '')
            etype = classify_edge(src_type, tgt_type)
        edge_types[(s, t)] = etype
        edge_type_counts[etype] += 1

    for etype, cnt in sorted(edge_type_counts.items(), key=lambda x: -x[1]):
        print(f'  {etype}: {cnt} edges ({100*cnt/len(redirected_edges):.1f}%)')

    # ========== LOCAL CLUSTERING COEFFICIENT ==========
    # For each node, compute clustering among its in-neighborhood.
    # High in-degree + low clustering = DEFINITIONAL hub (shared vocabulary)
    # High in-degree + high clustering = FUNCTIONAL core (operational module)
    # This distinguishes load-bearing walls from paint.

    print('\nComputing local clustering coefficients (in-neighborhood)...')

    # Build in-neighbor sets (who cites me?)
    in_neighbors = defaultdict(set)
    for s, t in redirected_edges:
        in_neighbors[t].add(s)

    # For clustering, we need to know edges among the in-neighbors.
    # Build adjacency set for fast lookup.
    edge_set = set(redirected_edges)

    clustering_in = {}
    high_degree_threshold = 10  # only compute for nodes with >=10 in-citations

    nodes_computed = 0
    for sid in node_ids:
        nb = in_neighbors.get(sid, set())
        k = len(nb)
        if k < 2:
            clustering_in[sid] = 0.0
            continue
        if k < high_degree_threshold:
            clustering_in[sid] = -1.0  # sentinel: too few neighbors to be meaningful
            continue

        # Count edges among in-neighbors
        # For large neighborhoods, sample to keep O(n) manageable
        nb_list = list(nb)
        if k > 200:
            # Sample: pick 200 random pairs to estimate
            import random
            sample_size = min(5000, k * (k - 1) // 2)
            edge_count = 0
            for _ in range(sample_size):
                i, j = random.sample(range(k), 2)
                if (nb_list[i], nb_list[j]) in edge_set or (nb_list[j], nb_list[i]) in edge_set:
                    edge_count += 1
            clustering_in[sid] = edge_count / sample_size
        else:
            edge_count = 0
            possible = k * (k - 1)  # directed pairs
            for a in nb_list:
                for b in nb_list:
                    if a != b and (a, b) in edge_set:
                        edge_count += 1
            clustering_in[sid] = edge_count / possible if possible > 0 else 0.0
        nodes_computed += 1

    print(f'  Computed for {nodes_computed} statutes (in-degree >= {high_degree_threshold})')

    # Classify hubs
    hub_definitional = []  # high in-degree, low clustering → shared vocabulary
    hub_functional = []    # high in-degree, high clustering → operational module
    for sid in node_ids:
        in_deg = int(node_map.get(sid, {}).get('in_degree', 0))
        cc = clustering_in.get(sid, 0.0)
        if in_deg < 50 or cc < 0:
            continue
        if cc < 0.05:
            hub_definitional.append((sid, in_deg, cc))
        elif cc > 0.15:
            hub_functional.append((sid, in_deg, cc))

    hub_definitional.sort(key=lambda x: -x[1])
    hub_functional.sort(key=lambda x: -x[1])

    print(f'\n  DEFINITIONAL hubs (in-degree>=50, clustering<0.05): {len(hub_definitional)}')
    for sid, deg, cc in hub_definitional[:10]:
        title = titles.get(sid, '')[:60]
        katz = R_katz.get(sid, 0) / 1e9
        print(f'    deg={deg:4d}  cc={cc:.3f}  katz={katz:6.1f}B  {sid:15s}  {title}')

    print(f'\n  FUNCTIONAL hubs (in-degree>=50, clustering>0.15): {len(hub_functional)}')
    for sid, deg, cc in hub_functional[:10]:
        title = titles.get(sid, '')[:60]
        katz = R_katz.get(sid, 0) / 1e9
        print(f'    deg={deg:4d}  cc={cc:.3f}  katz={katz:6.1f}B  {sid:15s}  {title}')

    # ========== DEBTRANK (Battiston et al. 2012) ==========
    # Propagate fiscal shock through the legal network.
    # Edge (A→B) = "A cites B" = "A depends on B".
    # If B is distressed, all A that cite B are affected → shock travels backward.
    # DebtRank(v) = total Katz weight lost when v is initially shocked to distress=1.0.
    #
    # Edge weights incorporate edge type:
    #   - PEER (Laki↔Laki): full contagion (1.0) — horizontal policy entanglement
    #   - IMPL_DOWN (Laki→Asetus): 0.3 — asetus failing rarely kills parent law
    #   - AUTH_UP (Asetus→Laki): 0.9 — authority basis failure is severe
    #   - ASETUS_PEER: 0.4
    #   - EXEC: 0.2 — executive decisions are operationally replaceable
    #   - OTHER: 0.3

    print('\nComputing DebtRank (fiscal contagion)...')

    EDGE_TYPE_WEIGHT = {
        'PEER': 1.0,
        'AUTH_UP': 0.9,
        'REPEALS': 1.0,
        'ISSUED_UNDER': 0.8,
        'ASETUS_PEER': 0.4,
        'IMPL_DOWN': 0.3,
        'EXEC': 0.2,
        'OTHER': 0.3,
    }

    # Total system value = sum of all Katz weights
    total_katz = sum(R_katz.get(sid, 0) for sid in node_ids)

    # Build reverse adjacency with weights: who_depends_on[B] = [(A, weight), ...]
    # Edge (A→B) means A depends on B. If B fails, A is affected.
    # Weight = type_multiplier / out_degree(A) — if A cites 10 laws, each is ~10% of A's dependency.
    who_depends_on = defaultdict(list)  # B -> [(A, weight)]
    out_deg = defaultdict(int)
    for s, t in redirected_edges:
        out_deg[s] += 1

    for s, t in redirected_edges:
        etype = edge_types.get((s, t), 'OTHER')
        type_w = EDGE_TYPE_WEIGHT.get(etype, 0.3)
        w = type_w / out_deg[s] if out_deg[s] > 0 else 0.0
        who_depends_on[t].append((s, w))

    # Run DebtRank for top-N nodes by Katz (all 92K too expensive — O(N*(V+E)))
    katz_ranked = sorted(node_ids, key=lambda x: R_katz.get(x, 0), reverse=True)
    TOP_N = 500

    debtrank = {}

    for idx, seed in enumerate(katz_ranked[:TOP_N]):
        if R_katz.get(seed, 0) == 0:
            break

        # State: 0=inactive, 1=distressed, 2=already-propagated
        state = {}
        h = defaultdict(float)  # health impact [0,1]

        state[seed] = 1
        h[seed] = 1.0
        active = [seed]

        # Propagate (BFS, each node propagates at most once per Battiston constraint)
        while active:
            next_active = []
            for v in active:
                for (dep, w) in who_depends_on.get(v, []):
                    if state.get(dep, 0) == 0:  # only inactive nodes
                        delta = min(1.0, h[dep] + h[v] * w)
                        h[dep] = delta
                        if delta > 0.01:  # noise threshold
                            state[dep] = 1
                            next_active.append(dep)
                state[v] = 2  # mark as propagated
            active = next_active

        # DebtRank = sum of (h(v) * Katz(v)) for all affected nodes, excluding seed
        dr = sum(h[v] * R_katz.get(v, 0) for v in h if v != seed)
        debtrank[seed] = dr

    # Print top results
    dr_ranked = sorted(debtrank.items(), key=lambda x: -x[1])
    print(f'  Computed for {len(debtrank)} statutes')
    print('\n  TOP 20 BY DEBTRANK (total Katz lost when this law fails):')
    for sid, dr in dr_ranked[:20]:
        katz = R_katz.get(sid, 0) / 1e9
        dr_b = dr / 1e9
        pct = 100 * dr / total_katz if total_katz > 0 else 0
        title = titles.get(sid, '')[:55]
        print(f'    DR={dr_b:7.1f}B ({pct:4.1f}%)  katz={katz:6.1f}B  {sid:15s}  {title}')

    # Divergence: what does DebtRank reveal that Katz doesn't?
    print('\n  BIGGEST DEBTRANK/KATZ DIVERGENCES (hidden systemic importance):')
    divergences = []
    for sid in debtrank:
        katz = R_katz.get(sid, 0)
        dr = debtrank[sid]
        if katz > 0 and dr > 0:
            divergences.append((sid, dr / katz, dr, katz))
    divergences.sort(key=lambda x: -x[1])
    for sid, ratio, dr, katz in divergences[:15]:
        title = titles.get(sid, '')[:55]
        print(f'    ratio={ratio:5.1f}x  DR={dr/1e9:6.1f}B  katz={katz/1e9:6.1f}B  {sid:15s}  {title}')

    # ========== HE SCRUTINY DATA ==========

    # ========== HE MASTER INDEX (from build_he_db.py --build-index) ==========
    HE_INDEX_PATH = _HE_INDEX_DB
    HE_DB_DIR = _HE_DB_DIR

    he_nodes_data = []             # (he_id, year, number, title, ministry, date_issued)
    he_statute_links = []          # (he_id, statute_id, ref_type)  — AMENDS or CITES
    he_amends_by_he = defaultdict(list)  # he_id -> [statute_id] (AMENDS only, for claim attribution)
    he_he_refs = []                # (source_he, target_he)

    if HE_INDEX_PATH.exists():
        hconn = sqlite3.connect(str(HE_INDEX_PATH))
        he_nodes_data = hconn.execute(
            'SELECT he_id, year, number, title, ministry, date_issued FROM he_index'
        ).fetchall()
        raw_links = hconn.execute(
            'SELECT he_id, statute_ref, ref_type FROM he_statute_refs'
        ).fetchall()
        he_he_refs = hconn.execute(
            'SELECT source_he, target_he FROM he_he_refs'
        ).fetchall()
        hconn.close()

        for he_id, sref, rtype in raw_links:
            sid = normalize_statute_id(sref)
            he_statute_links.append((he_id, sid, rtype))
            if rtype == 'AMENDS':
                he_amends_by_he[he_id].append(sid)

        n_amends = sum(1 for _, _, rt in he_statute_links if rt == 'AMENDS')
        n_cites = sum(1 for _, _, rt in he_statute_links if rt == 'CITES')
        print(f'\nLoaded HE master index: {len(he_nodes_data)} HEs')
        print(f'  {n_amends} AMENDS + {n_cites} CITES = {len(he_statute_links)} HE→statute links')
        print(f'  {len(he_he_refs)} HE→HE cross-references')
    else:
        print(f'\nNo HE master index at {HE_INDEX_PATH} — run: mev build he-index')

    # Load scrutiny analysis (from detect_scrutiny_ignored.py --all)
    SCRUTINY_PATH = ROOT / '.tmp' / 'scrutiny' / 'scrutiny_analysis.json'

    he_scrutiny = {}       # he_id -> {stats dict}
    statute_scrutiny = defaultdict(lambda: {
        'he_count': 0, 'expert_total': 0, 'silent_total': 0,
        'topic_only_total': 0, 'graveyard_total': 0,
        'max_ignored_rate': 0.0,
    })

    if SCRUTINY_PATH.exists():
        with open(SCRUTINY_PATH) as f:
            scrutiny_data = json.load(f)
        print(f'\nLoading HE scrutiny data: {len(scrutiny_data)} HEs')

        for he in scrutiny_data:
            he_id = he['he_id']
            if 'error' in he:
                continue
            he_scrutiny[he_id] = {
                'total_experts': he.get('total_experts_analyzed', 0),
                'silent': he.get('silent_count', 0),
                'topic_only': he.get('topic_only_count', 0),
                'graveyard': he.get('graveyard_count', 0),
                'addressed': he.get('addressed_count', 0),
                'ignored_rate': he.get('scrutiny_ignored_rate', 0.0),
            }

            # Aggregate scrutiny stats per statute using AMENDS links from index
            for sid in he_amends_by_he.get(he_id, []):
                ss = statute_scrutiny[sid]
                ss['he_count'] += 1
                ss['expert_total'] += he_scrutiny[he_id]['total_experts']
                ss['silent_total'] += he_scrutiny[he_id]['silent']
                ss['topic_only_total'] += he_scrutiny[he_id]['topic_only']
                ss['graveyard_total'] += he_scrutiny[he_id]['graveyard']
                ss['max_ignored_rate'] = max(
                    ss['max_ignored_rate'],
                    he_scrutiny[he_id]['ignored_rate']
                )

            # Fallback: if no AMENDS in index, try per-HE DB
            if he_id not in he_amends_by_he:
                he_db = HE_DB_DIR / f'{he_id}.db'
                if he_db.exists():
                    try:
                        hdb = sqlite3.connect(str(he_db))
                        row = hdb.execute('SELECT laws_amended FROM metadata LIMIT 1').fetchone()
                        if row and row[0]:
                            for sid_raw in json.loads(row[0]):
                                sid = normalize_statute_id(sid_raw)
                                he_amends_by_he[he_id].append(sid)
                                he_statute_links.append((he_id, sid, 'AMENDS'))
                                ss = statute_scrutiny[sid]
                                ss['he_count'] += 1
                                ss['expert_total'] += he_scrutiny[he_id]['total_experts']
                                ss['silent_total'] += he_scrutiny[he_id]['silent']
                                ss['topic_only_total'] += he_scrutiny[he_id]['topic_only']
                                ss['graveyard_total'] += he_scrutiny[he_id]['graveyard']
                                ss['max_ignored_rate'] = max(
                                    ss['max_ignored_rate'],
                                    he_scrutiny[he_id]['ignored_rate']
                                )
                        hdb.close()
                    except Exception:
                        pass

        linked_statutes = sum(1 for s in statute_scrutiny if s in node_ids)
        print(f'  {len(he_scrutiny)} HEs with scrutiny data')
        print(f'  {len(statute_scrutiny)} unique statutes with scrutiny ({linked_statutes} in graph)')
    else:
        print(f'\nNo scrutiny data at {SCRUTINY_PATH} — skipping')
        scrutiny_data = []

    # ========== HE CLAIMS DATA (from he_enrichments.db) ==========

    ENRICHMENTS_DB = _ENRICHMENTS_DB

    he_claims_rows = []  # (claim_id, he_id, claim_type, text, quote, quote_match,
                         #  source_atom, amount_eur, scale, time_horizon, verifiability, confidence)

    if ENRICHMENTS_DB.exists():
        econn = sqlite3.connect(str(ENRICHMENTS_DB))
        try:
            rows = econn.execute(
                'SELECT claim_id, he_id, claim_type, text, quote, quote_match, '
                'source_atom, amount_eur, scale, time_horizon, verifiability, confidence '
                'FROM claim'
            ).fetchall()
            he_claims_rows = list(rows)

            he_claims_by_type = defaultdict(int)
            he_ids_with_claims = set()
            for r in he_claims_rows:
                he_claims_by_type[r[2]] += 1
                he_ids_with_claims.add(r[1])

            print(f'\nLoaded {len(he_claims_rows)} claims from {len(he_ids_with_claims)} HEs (he_enrichments.db)')
            for ct in sorted(he_claims_by_type):
                print(f'    {ct}: {he_claims_by_type[ct]}')
        except sqlite3.OperationalError as e:
            print(f'\nWarning: could not read claims from {ENRICHMENTS_DB}: {e}')
        finally:
            econn.close()
    else:
        print(f'\nNo enrichments DB at {ENRICHMENTS_DB} — skipping claims')

    # Build claim→statute attribution links (AMENDS only — what the HE actually modifies)
    he_to_statutes = he_amends_by_he

    claim_statute_links = []  # (claim_id, he_id, statute_id, attribution)
    direct_count = 0
    he_wide_count = 0
    for claim_row in he_claims_rows:
        claim_id, he_id = claim_row[0], claim_row[1]
        statutes = he_to_statutes.get(he_id, [])
        if len(statutes) == 1:
            claim_statute_links.append((claim_id, he_id, statutes[0], 'DIRECT'))
            direct_count += 1
        else:
            for sid in statutes:
                claim_statute_links.append((claim_id, he_id, sid, 'HE_WIDE'))
                he_wide_count += 1

    if claim_statute_links:
        print(f'  claim→statute links: {len(claim_statute_links)} ({direct_count} DIRECT, {he_wide_count} HE_WIDE)')
        print(f'  {len(set(s for sl in he_to_statutes.values() for s in sl))} unique statutes linked')

    # ========== SYSTEMIC RISK INDEX ==========
    import math

    # Statutes covered by published mekanismitestit
    # Value = estimated audit coverage (0.0–1.0). A mekanismitesti tests a specific
    # HE (amendment), not the full law. Multiple tests on the same law accumulate.
    # These are rough estimates — the coverage concept is fuzzy by nature.
    TESTED_STATUTES = {
        '2002/1290': 0.4,  # Työttömyysturvalaki — HE 13 (porrastus) + HE 112 (yleistuki) + HE 116 (interaction)
        '2023/380':  0.2,  # Laki työvoimapalveluiden järjestämisestä — HE 13 (secondary)
        '2010/1386': 0.3,  # Kotoutumislaki — HE 21 (deep dive, single test)
        '2021/617':  0.5,  # HVA rahoituslaki — HE 38 + HE 189 (funding formula, two tests)
        '2021/611':  0.3,  # Laki hyvinvointialueesta — HE 38 + HE 189 (secondary)
        '1987/395':  0.15, # Lääkelaki — HE 111 (only apteekkivero interaction, not full drug reg)
        '1997/1412': 0.3,  # Laki toimeentulotuesta — HE 116 (single deep test)
        '2016/1397': 0.3,  # Hankintalaki — HE 2/2026 (single deep test)
        '2014/938':  0.25, # Laki yleisestä asumistuesta — one focused analysis
    }

    # risk = sqrt(ADMIN_FLOOR + propagated_budget) × (1 + in_degree)^0.6 × max(MIN_AMEND, log2(2 + amendments))
    # ADMIN_FLOOR (30k EUR): marginal cost of passing any law in Finland (lainvalmistelu,
    #   translation, impact assessment). No law is free — ensures zero-budget laws get nonzero risk.
    # sqrt(budget): asymmetric destruction — a 10B law isn't 100× riskier than 100M law,
    #   structural embedding (connectivity, amendment churn) matters more at the margin.
    # Power law on in-degree (0.6): log₂ over-dampened connectivity — 933 refs scored only 2×
    #   a 30-ref law. Power law gives ~10× ratio, much closer to actual structural criticality.
    # Amendment floor (3.0 ≈ 6 amendments): new critical laws have HIGHEST implementation risk
    #   (untested mechanisms). log₂ alone punishes them. Floor ensures new laws aren't crushed.
    ADMIN_FLOOR = 30_000  # EUR — marginal cost of a minor technical amendment
    MIN_AMEND_FACTOR = 3.0  # floor: equivalent to ~6 amendments (log2(8) = 3.0)
    CONNECTIVITY_EXP = 0.6  # sublinear power law on in-degree
    # Scrutiny multiplier: unaddressed expert concerns amplify existing risk.
    # Max boost = 1.5× (a 40%+ ignored rate on a high-budget law is 50% riskier).
    # Rationale: scrutiny-ignored doesn't CREATE risk (budget/connectivity do that),
    # it reveals that existing risk wasn't properly evaluated during legislation.
    # GRAVEYARD gets half weight — committee acknowledged but deferred.
    SCRUTINY_MAX_BOOST = 0.5  # at 100% ignored, multiply risk by 1.5
    GRAVEYARD_WEIGHT = 0.5    # GRAVEYARD counts as 50% of SILENT toward risk
    systemic_risk = {}
    for n in nodes:
        sid = n['id']
        prop = R_combined.get(sid, 0.0)
        in_deg = int(n['in_degree']) if n.get('in_degree') else 0
        amend = int(n['amendment_count']) if n.get('amendment_count') else 0
        in_force = n.get('in_force', '') == 'True'
        if not in_force:
            continue
        budget_factor = math.sqrt(ADMIN_FLOOR + prop)
        connectivity_factor = (1 + in_deg) ** CONNECTIVITY_EXP
        amendment_factor = max(MIN_AMEND_FACTOR, math.log2(2 + amend))

        # Scrutiny factor: multiplicative boost from unaddressed expert concerns
        scrutiny_factor = 1.0
        ss = statute_scrutiny.get(sid)
        if ss and ss['expert_total'] > 0:
            # Effective ignored = SILENT + TOPIC_ONLY + 0.5*GRAVEYARD
            effective_ignored = (ss['silent_total'] + ss['topic_only_total']
                                + GRAVEYARD_WEIGHT * ss['graveyard_total'])
            effective_rate = effective_ignored / ss['expert_total']
            scrutiny_factor = 1.0 + SCRUTINY_MAX_BOOST * min(effective_rate, 1.0)

        risk = budget_factor * connectivity_factor * amendment_factor * scrutiny_factor
        systemic_risk[sid] = risk

    # Normalize to 0-1000 scale (max = 1000)
    max_risk = max(systemic_risk.values()) if systemic_risk else 1.0
    for sid in systemic_risk:
        systemic_risk[sid] = (systemic_risk[sid] / max_risk) * 1000.0

    tested_in_risk = {s for s in TESTED_STATUTES if s in systemic_risk}
    scrutinized = sum(1 for s in systemic_risk if s in statute_scrutiny)
    print(f'\nSystemic risk index: {len(systemic_risk)} statutes scored')
    print(f'  Tested: {len(tested_in_risk)} of {len(TESTED_STATUTES)}')
    print(f'  With scrutiny data: {scrutinized}')
    top_risk = sorted(systemic_risk.items(), key=lambda x: -x[1])[:20]
    print('\n--- TOP 20 SYSTEMIC RISK ---')
    for sid, risk in top_risk:
        cov = TESTED_STATUTES.get(sid, 0.0)
        flag = f'{cov:.0%}' if cov > 0 else '   '
        title = titles.get(sid, '')[:40]
        prop = R_combined.get(sid, 0) / 1e6
        n = next((n for n in nodes if n['id'] == sid), {})
        amend = int(n.get('amendment_count', 0))
        in_deg = int(n.get('in_degree', 0))
        ss = statute_scrutiny.get(sid)
        scr = f'scr={ss["max_ignored_rate"]:.0%}' if ss else '      '
        print(f'  {flag:>4s} {risk:7.1f}  {sid:15s}  prop={prop:8.0f}M  deg={in_deg:4d}  amend={amend:4d}  {scr}  {title}')

    # ========== BUILD DB ==========

    print('\nBuilding database...')

    c.execute('''CREATE TABLE statutes (
        id TEXT PRIMARY KEY,
        title TEXT,
        common_name TEXT,
        year INTEGER,
        number TEXT,
        ministry TEXT,
        in_force BOOLEAN,
        date_issued TEXT,
        date_entry_into_force TEXT,
        date_in_force_end TEXT,
        type_statute TEXT,
        keywords TEXT,
        amendment_count INTEGER,
        editorial_note_count INTEGER,
        corrigendum_count INTEGER,
        chapters INTEGER,
        sections INTEGER,
        in_degree INTEGER,
        out_degree INTEGER,
        budget_direct_eur REAL,
        budget_momentti_count INTEGER,
        budget_cofog_eur REAL,
        budget_revenue_eur REAL,
        budget_combined_eur REAL,
        budget_combined_propagated_eur REAL,
        budget_procurement_eur REAL,
        budget_institution_aum_eur REAL,
        systemic_risk_score REAL,
        audit_coverage REAL,
        delegation_total INTEGER,
        delegation_vn INTEGER,
        delegation_min INTEGER,
        delegation_agency INTEGER,
        delegation_density REAL,
        delegation_child_count INTEGER,
        delegation_gap INTEGER,
        scrutiny_he_count INTEGER,
        scrutiny_expert_total INTEGER,
        scrutiny_silent INTEGER,
        scrutiny_graveyard INTEGER,
        scrutiny_ignored_rate REAL,
        budget_katz_eur REAL,
        bowtie_zone TEXT,
        clustering_in REAL,
        debtrank_eur REAL,
        churn_rate REAL,
        hollow_mandate_rate REAL,
        churn_contagion REAL,
        dead_ref_rate REAL,
        he_attention INTEGER DEFAULT 0,
        he_attention_recent INTEGER DEFAULT 0,
        he_pol_degree INTEGER DEFAULT 0,
        he_coamend_pagerank REAL DEFAULT 0,
        he_rank_divergence REAL DEFAULT 0,
        graph_eu_ref_count INTEGER,
        graph_stale_ref_pct REAL,
        mandatory_unexercised INTEGER
    )''')

    c.execute('''CREATE TABLE edges (
        source TEXT,
        target TEXT,
        edge_type TEXT,
        FOREIGN KEY (source) REFERENCES statutes(id),
        FOREIGN KEY (target) REFERENCES statutes(id)
    )''')

    c.execute('''CREATE TABLE momentit (
        code TEXT PRIMARY KEY,
        dotted TEXT,
        paaluokka TEXT,
        hallinnonala TEXT,
        name TEXT,
        budget_2024_eur REAL
    )''')

    c.execute('''CREATE TABLE mapping (
        momentti_code TEXT,
        statute_id TEXT,
        confidence TEXT,
        source TEXT,
        notes TEXT,
        FOREIGN KEY (momentti_code) REFERENCES momentit(code),
        FOREIGN KEY (statute_id) REFERENCES statutes(id)
    )''')

    c.execute('''CREATE TABLE tae_texts (
        momentti_code TEXT PRIMARY KEY,
        text_content TEXT,
        FOREIGN KEY (momentti_code) REFERENCES momentit(code)
    )''')

    c.execute('''CREATE TABLE cofog (
        code TEXT PRIMARY KEY,
        name TEXT,
        sector TEXT,
        sector_name TEXT,
        meur_2023 REAL
    )''')

    c.execute('''CREATE TABLE cofog_mapping (
        cofog_code TEXT,
        statute_id TEXT,
        confidence TEXT,
        notes TEXT,
        FOREIGN KEY (cofog_code) REFERENCES cofog(code),
        FOREIGN KEY (statute_id) REFERENCES statutes(id)
    )''')

    # Insert statutes
    for n in nodes:
        sid = n['id']
        mom_info = statute_momentti.get(sid, {'budget': 0.0, 'count': 0})
        dd = delegation_data.get(sid, {})
        ss = statute_scrutiny.get(sid)
        gc = graph_census.get(sid, {})
        yr = int(n['year']) if n['year'] else 0
        amend = int(n['amendment_count']) if n.get('amendment_count') else 0
        d_total = int(dd['delegation_total']) if dd.get('delegation_total') else 0
        d_gap = int(dd['delegation_gap']) if dd.get('delegation_gap') else 0
        # When graph_dir is provided, prefer graph artifact values for primary columns.
        # Fallback to nodes.csv values when graph_census is absent (CSV-only mode).
        if gc:
            amendment_count_val = gc.get('amendment_count') if gc.get('amendment_count') is not None else (int(n['amendment_count']) if n.get('amendment_count') else 0)
            in_degree_val = gc.get('cites_in') if gc.get('cites_in') is not None else (int(n['in_degree']) if n.get('in_degree') else 0)
            out_degree_val = gc.get('cites_out') if gc.get('cites_out') is not None else (int(n['out_degree']) if n.get('out_degree') else 0)
            delegation_gap_val = gc.get('unexercised_delegations') if gc.get('unexercised_delegations') is not None else d_gap
        else:
            amendment_count_val = int(n['amendment_count']) if n.get('amendment_count') else 0
            in_degree_val = int(n['in_degree']) if n.get('in_degree') else 0
            out_degree_val = int(n['out_degree']) if n.get('out_degree') else 0
            delegation_gap_val = d_gap
        c.execute(
            'INSERT INTO statutes VALUES (' + ','.join(['?'] * 57) + ')',
            (
                sid,
                titles.get(sid, ''),
                n.get('common_name', ''),
                int(n['year']) if n['year'] else None,
                n['number'] if n['number'] else None,
                n.get('ministry', ''),
                n.get('in_force', '') == 'True',
                n.get('date_issued', ''),
                n.get('date_entry_into_force', ''),
                n.get('date_in_force_end', ''),
                n.get('type_statute', ''),
                n.get('keywords', ''),
                amendment_count_val,
                int(n['editorial_note_count']) if n.get('editorial_note_count') else 0,
                int(n['corrigendum_count']) if n.get('corrigendum_count') else 0,
                int(n['chapters']) if n.get('chapters') else 0,
                int(n['sections']) if n.get('sections') else 0,
                in_degree_val,
                out_degree_val,
                mom_info['budget'],
                mom_info['count'],
                statute_cofog.get(sid, 0.0),
                statute_revenue.get(sid, 0.0),
                budget_combined.get(sid, 0.0),
                R_combined.get(sid, 0.0),
                statute_procurement.get(sid, 0.0),
                statute_inst_aum.get(sid, 0.0),
                systemic_risk.get(sid, 0.0),
                TESTED_STATUTES.get(sid, 0.0),
                int(dd['delegation_total']) if dd.get('delegation_total') else 0,
                int(dd['delegation_vn']) if dd.get('delegation_vn') else 0,
                int(dd['delegation_min']) if dd.get('delegation_min') else 0,
                int(dd['delegation_agency']) if dd.get('delegation_agency') else 0,
                float(dd['delegation_density']) if dd.get('delegation_density') else 0.0,
                int(dd['child_asetukset']) if dd.get('child_asetukset') else 0,
                delegation_gap_val,
                # Scrutiny columns
                ss['he_count'] if ss else 0,
                ss['expert_total'] if ss else 0,
                ss['silent_total'] + ss['topic_only_total'] if ss else 0,
                ss['graveyard_total'] if ss else 0,
                ss['max_ignored_rate'] if ss else 0.0,
                R_katz.get(sid, 0.0),
                bowtie_zone.get(sid, 'TENDRIL'),
                clustering_in.get(sid, 0.0),
                debtrank.get(sid, 0.0),
                # Derived metrics
                amend / (2026 - yr) if yr and yr < 2026 else 0.0,  # churn_rate
                delegation_gap_val / d_total if d_total > 0 else 0.0,  # hollow_mandate_rate
                debtrank.get(sid, 0.0) * (amend / (2026 - yr)) if yr and yr < 2026 else 0.0,  # churn_contagion
                0.0,  # dead_ref_rate — computed after edges INSERT
                0,    # he_attention — computed after he_statute_link INSERT
                0,    # he_attention_recent — computed after he_statute_link INSERT
                0,    # he_pol_degree — computed after co_amendments INSERT
                0.0,  # he_coamend_pagerank — computed after co_amendments INSERT
                0.0,  # he_rank_divergence — computed after all PageRanks
                # graph_* supplementary columns (NULL when graph_dir not set)
                gc.get('eu_ref_count'),              # graph_eu_ref_count
                gc.get('stale_ref_pct'),             # graph_stale_ref_pct (None = no FI refs)
                gc.get('mandatory_unexercised'),     # mandatory_unexercised
            )
        )
    print(f'  statutes: {len(nodes)}')

    # Edges
    c.executemany('INSERT INTO edges VALUES (?,?,?)',
                   [(s, t, edge_types.get((s, t), 'OTHER')) for s, t in edges])
    print(f'  edges: {len(edges)}')

    # Compute dead_ref_rate: fraction of PEER/OTHER outbound edges to kumottu (not-in-force) statutes
    # Exclude IMPL_DOWN (naturally expiring decrees), REPEALS, ISSUED_UNDER
    # Distinct from graph_stale_ref_pct (which counts refs where target was amended since citation).
    print('  Computing dead_ref_rate...')
    in_force_set = set(r[0] for r in c.execute('SELECT id FROM statutes WHERE in_force = 1'))
    stale_agg = defaultdict(lambda: [0, 0])  # source_id -> [dead_count, total_count]
    for source, target, etype in c.execute('SELECT source, target, edge_type FROM edges'):
        if etype in ('IMPL_DOWN', 'REPEALS', 'ISSUED_UNDER'):
            continue
        stale_agg[source][1] += 1
        if target not in in_force_set:
            stale_agg[source][0] += 1
    stale_updates = [(s / t if t > 0 else 0.0, sid) for sid, (s, t) in stale_agg.items() if t > 0]
    c.executemany('UPDATE statutes SET dead_ref_rate = ? WHERE id = ?', stale_updates)
    stale_count = sum(1 for _, (s, t) in stale_agg.items() if s > 0)
    print(f'    {stale_count} statutes have dead references')

    # Momentit
    for r in momentit_raw:
        c.execute('INSERT INTO momentit VALUES (?,?,?,?,?,?)', (
            r['momentti_code'], r['momentti_dotted'], r['paaluokka'],
            r.get('hallinnonala', ''), r['momentti_name'],
            float(r['budget_2024_eur']) if r.get('budget_2024_eur') else 0.0,
        ))
    print(f'  momentit: {len(momentit_raw)}')

    # Mapping (deduplicated: keep highest-confidence row per momentti+statute pair)
    mapping_deduped = {}
    for r in mapping_raw:
        key = (r['momentti_code'], r['statute_id'])
        conf = CONF_ORDER.get(r.get('confidence', 'LOW'), 1)
        if key not in mapping_deduped or conf > CONF_ORDER.get(mapping_deduped[key].get('confidence', 'LOW'), 1):
            mapping_deduped[key] = r
    for r in mapping_deduped.values():
        c.execute('INSERT INTO mapping VALUES (?,?,?,?,?)', (
            r['momentti_code'], r['statute_id'],
            r.get('confidence', ''), r.get('source', ''), r.get('notes', ''),
        ))
    print(f'  mapping: {len(mapping_deduped)} (deduplicated from {len(mapping_raw)})')

    # TAE texts
    for code, text in tae_texts.items():
        c.execute('INSERT INTO tae_texts VALUES (?,?)', (code, text))
    print(f'  tae_texts: {len(tae_texts)}')

    # COFOG
    for r in cofog_all_rows:
        if len(r['cofog']) == 5:
            c.execute('INSERT OR IGNORE INTO cofog VALUES (?,?,?,?,?)', (
                r['cofog'], r['cofog_name'], r['sector'], r['sector_name'],
                float(r['meur_2023']) if r['meur_2023'] else 0.0,
            ))
    cofog_count = c.execute('SELECT COUNT(*) FROM cofog').fetchone()[0]
    print(f'  cofog: {cofog_count}')

    # COFOG mapping
    for r in cofog_mapping:
        c.execute('INSERT INTO cofog_mapping VALUES (?,?,?,?)', (
            r['cofog'], r['statute_id'],
            r.get('confidence', ''), r.get('notes', ''),
        ))
    print(f'  cofog_mapping: {len(cofog_mapping)}')

    # Institutions and institution-statute edges (always created)
    c.execute('''CREATE TABLE institutions (
        id TEXT PRIMARY KEY,
        name TEXT,
        short_name TEXT,
        sector TEXT,
        type TEXT,
        aum_meur REAL,
        aum_year INTEGER,
        payout_rate REAL,
        source TEXT
    )''')
    c.execute('''CREATE TABLE institution_statute_edges (
        institution_id TEXT,
        statute_id TEXT,
        edge_type TEXT,
        notes TEXT,
        FOREIGN KEY (institution_id) REFERENCES institutions(id),
        FOREIGN KEY (statute_id) REFERENCES statutes(id)
    )''')
    if institutions:
        for r in institutions:
            c.execute('INSERT INTO institutions VALUES (?,?,?,?,?,?,?,?,?)', (
                r['id'], r['name'], r['short_name'], r['sector'],
                r['type'], float(r['aum_meur']), int(r['aum_year']),
                float(r['payout_rate']),
                r.get('source', ''),
            ))
        for r in inst_mapping:
            c.execute('INSERT INTO institution_statute_edges VALUES (?,?,?,?)', (
                r['institution_id'], r['statute_id'],
                r['edge_type'], r.get('notes', ''),
            ))
    print(f'  institutions: {len(institutions)}')
    print(f'  institution_statute_edges: {len(inst_mapping)}')

    # Delegation detail rows (always created)
    c.execute('''CREATE TABLE delegations (
        statute_id TEXT,
        section TEXT,
        eid TEXT,
        delegation_type TEXT,
        match_text TEXT,
        quote TEXT,
        FOREIGN KEY (statute_id) REFERENCES statutes(id)
    )''')
    if delegation_rows:
        for r in delegation_rows:
            c.execute('INSERT INTO delegations VALUES (?,?,?,?,?,?)', (
                r['statute_id'], r['section'], r.get('eid', ''),
                r['delegation_type'], r['match_text'], r.get('quote', ''),
            ))
    print(f'  delegations: {len(delegation_rows)}')

    # Asetus authority linkages (always created)
    c.execute('''CREATE TABLE asetus_authority (
        asetus_id TEXT,
        parent_statute_id TEXT,
        parent_section TEXT,
        parent_momentti TEXT,
        preamble_quote TEXT,
        FOREIGN KEY (asetus_id) REFERENCES statutes(id),
        FOREIGN KEY (parent_statute_id) REFERENCES statutes(id)
    )''')
    if asetus_authority:
        for r in asetus_authority:
            c.execute('INSERT INTO asetus_authority VALUES (?,?,?,?,?)', (
                r['asetus_id'], r['parent_statute_id'],
                r['parent_section'], r.get('parent_momentti', ''),
                r.get('preamble_quote', ''),
            ))
    print(f'  asetus_authority: {len(asetus_authority)}')

    # HE nodes (from master index — all government proposals)
    c.execute('''CREATE TABLE he_nodes (
        he_id TEXT PRIMARY KEY,
        year INTEGER,
        number INTEGER,
        title TEXT,
        ministry TEXT,
        date_issued TEXT,
        he_pagerank REAL DEFAULT 0
    )''')
    if he_nodes_data:
        c.executemany('INSERT INTO he_nodes VALUES (?,?,?,?,?,?,0)', he_nodes_data)
    print(f'  he_nodes: {len(he_nodes_data)}')

    # HE scrutiny tables (always created, may be empty)
    c.execute('''CREATE TABLE he_scrutiny (
        he_id TEXT PRIMARY KEY,
        total_experts INTEGER,
        silent INTEGER,
        topic_only INTEGER,
        graveyard INTEGER,
        addressed INTEGER,
        ignored_rate REAL
    )''')
    for he_id, s in he_scrutiny.items():
        c.execute('INSERT INTO he_scrutiny VALUES (?,?,?,?,?,?,?)', (
            he_id, s['total_experts'], s['silent'], s['topic_only'],
            s['graveyard'], s['addressed'], s['ignored_rate'],
        ))
    print(f'  he_scrutiny: {len(he_scrutiny)}')

    # HE→statute links (AMENDS + CITES from master index)
    c.execute('''CREATE TABLE he_statute_link (
        he_id TEXT,
        statute_id TEXT,
        ref_type TEXT,
        FOREIGN KEY (statute_id) REFERENCES statutes(id)
    )''')
    if he_statute_links:
        c.executemany('INSERT INTO he_statute_link VALUES (?,?,?)', he_statute_links)
    print(f'  he_statute_link: {len(he_statute_links)}')

    # HE→HE cross-references
    c.execute('''CREATE TABLE he_he_refs (
        source_he TEXT NOT NULL,
        target_he TEXT NOT NULL
    )''')
    if he_he_refs:
        c.executemany('INSERT INTO he_he_refs VALUES (?,?)', he_he_refs)
    print(f'  he_he_refs: {len(he_he_refs)}')

    # HE claims (from he_enrichments.db)
    c.execute('''CREATE TABLE he_claims (
        claim_id TEXT PRIMARY KEY,
        he_id TEXT NOT NULL,
        claim_type TEXT NOT NULL,
        text TEXT NOT NULL,
        quote TEXT,
        quote_match TEXT,
        source_atom TEXT,
        amount_eur REAL,
        scale TEXT,
        time_horizon TEXT,
        verifiability TEXT,
        confidence REAL
    )''')
    if he_claims_rows:
        c.executemany(
            'INSERT INTO he_claims VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
            he_claims_rows,
        )
    print(f'  he_claims: {len(he_claims_rows)}')

    # Claim → statute attribution (DIRECT for single-statute HEs, HE_WIDE for multi)
    c.execute('''CREATE TABLE claim_statute_link (
        claim_id TEXT NOT NULL,
        he_id TEXT NOT NULL,
        statute_id TEXT NOT NULL,
        attribution TEXT NOT NULL,
        PRIMARY KEY (claim_id, statute_id),
        FOREIGN KEY (statute_id) REFERENCES statutes(id)
    )''')
    if claim_statute_links:
        c.executemany(
            'INSERT INTO claim_statute_link VALUES (?,?,?,?)',
            claim_statute_links,
        )
    print(f'  claim_statute_link: {len(claim_statute_links)}')

    # Successors (repealed → replacement)
    c.execute('''CREATE TABLE successors (
        old_id TEXT NOT NULL,
        new_id TEXT NOT NULL,
        PRIMARY KEY (old_id),
        FOREIGN KEY (old_id) REFERENCES statutes(id),
        FOREIGN KEY (new_id) REFERENCES statutes(id)
    )''')
    if successor_map:
        c.executemany('INSERT INTO successors VALUES (?,?)',
                      [(old, new) for old, new in successor_map.items()])
    print(f'  successors: {len(successor_map)}')

    # Amendment parents (muutoslaki → parent law it modified)
    c.execute('''CREATE TABLE amendment_parents (
        amendment_id TEXT NOT NULL,
        parent_id TEXT NOT NULL,
        PRIMARY KEY (amendment_id),
        FOREIGN KEY (amendment_id) REFERENCES statutes(id),
        FOREIGN KEY (parent_id) REFERENCES statutes(id)
    )''')
    if amendment_parents:
        c.executemany('INSERT INTO amendment_parents VALUES (?,?)',
                      [(a, p) for a, p in amendment_parents.items()])
    print(f'  amendment_parents: {len(amendment_parents)}')

    # ---- Change coupling: co-amendment detection ----
    print('  Computing change coupling (co-amendments)...')
    c.execute('''CREATE TABLE co_amendments (
        statute_a TEXT NOT NULL,
        statute_b TEXT NOT NULL,
        co_count INTEGER NOT NULL,
        shared_dates TEXT,
        coupling_type TEXT NOT NULL DEFAULT 'DATE',
        PRIMARY KEY (statute_a, statute_b),
        FOREIGN KEY (statute_a) REFERENCES statutes(id),
        FOREIGN KEY (statute_b) REFERENCES statutes(id)
    )''')

    # Method 1: Date-based — amendments sharing entry-into-force dates
    co_amend_rows = c.execute('''
        WITH dated_amendments AS (
            SELECT ap.parent_id, s.date_entry_into_force as d
            FROM amendment_parents ap
            JOIN statutes s ON ap.amendment_id = s.id
            WHERE s.date_entry_into_force IS NOT NULL AND s.date_entry_into_force <> ''
        )
        SELECT a.parent_id, b.parent_id, COUNT(DISTINCT a.d) as co_count,
               GROUP_CONCAT(DISTINCT a.d) as dates
        FROM dated_amendments a
        JOIN dated_amendments b ON a.d = b.d AND a.parent_id < b.parent_id
        GROUP BY a.parent_id, b.parent_id
        HAVING co_count >= 3
        ORDER BY co_count DESC
    ''').fetchall()
    c.executemany('INSERT INTO co_amendments VALUES (?,?,?,?,?)',
                  [(a, b, n, d, 'DATE') for a, b, n, d in co_amend_rows])
    print(f'  co_amendments (DATE): {len(co_amend_rows)} pairs (>= 3 shared dates)')

    # Method 2: HE-based — statutes referenced together in the same government proposal
    # Uses CITES edges (broader) since AMENDS is sparse for older HEs
    he_statute_sets = defaultdict(set)
    for he_id, sid, rtype in he_statute_links:
        if sid in node_ids:  # only statutes in our graph
            he_statute_sets[he_id].add(sid)

    # Count co-occurrence: how many HEs reference both statute A and B
    from itertools import combinations
    he_coupling = defaultdict(lambda: [0, []])
    for he_id, statutes in he_statute_sets.items():
        if len(statutes) < 2 or len(statutes) > 50:  # skip trivially small or mega-HEs
            continue
        for a, b in combinations(sorted(statutes), 2):
            key = (a, b)
            he_coupling[key][0] += 1
            if len(he_coupling[key][1]) < 10:  # keep up to 10 HE IDs as evidence
                he_coupling[key][1].append(he_id)

    he_co_rows = [(a, b, n, ','.join(hes), 'HE')
                  for (a, b), (n, hes) in he_coupling.items()
                  if n >= 3 and (a, b) not in {(r[0], r[1]) for r in co_amend_rows}]
    he_co_rows.sort(key=lambda x: -x[2])
    if he_co_rows:
        c.executemany('INSERT OR IGNORE INTO co_amendments VALUES (?,?,?,?,?)', he_co_rows)
    print(f'  co_amendments (HE): {len(he_co_rows)} pairs (>= 3 shared HEs, not in DATE)')
    print(f'  co_amendments total: {len(co_amend_rows) + len(he_co_rows)}')

    # ---- Compute he_attention / he_attention_recent per statute ----
    print('  Computing HE attention per statute...')
    he_year_lookup = {r[0]: r[1] for r in c.execute('SELECT he_id, year FROM he_nodes')}
    attn_total = defaultdict(int)
    attn_recent = defaultdict(int)
    for he_id, sid, _ in c.execute('SELECT he_id, statute_id, ref_type FROM he_statute_link'):
        attn_total[sid] += 1
        yr = he_year_lookup.get(he_id, 0)
        if yr >= 2020:
            attn_recent[sid] += 1
    c.executemany('UPDATE statutes SET he_attention = ? WHERE id = ?',
                  [(n, sid) for sid, n in attn_total.items()])
    c.executemany('UPDATE statutes SET he_attention_recent = ? WHERE id = ?',
                  [(n, sid) for sid, n in attn_recent.items()])
    top_attn = c.execute('''
        SELECT id, title, he_attention, he_attention_recent, in_degree
        FROM statutes WHERE he_attention > 0
        ORDER BY he_attention DESC LIMIT 5
    ''').fetchall()
    n_with_attn = c.execute('SELECT COUNT(*) FROM statutes WHERE he_attention > 0').fetchone()[0]
    print(f'    {n_with_attn} statutes with HE attention')
    for r in top_attn:
        print(f'      {r[0]:>15}  attn={r[2]:>5}  recent={r[3]:>3}  deg={r[4]:>4}  {(r[1] or "")[:45]}')

    # ---- Compute he_pol_degree (political degree centrality on co-amendment network) ----
    print('  Computing political degree centrality...')
    pol_deg = defaultdict(int)
    for row in c.execute('SELECT statute_a, statute_b, co_count FROM co_amendments WHERE coupling_type = ?', ('HE',)):
        pol_deg[row[0]] += row[2]
        pol_deg[row[1]] += row[2]
    c.executemany('UPDATE statutes SET he_pol_degree = ? WHERE id = ?',
                  [(d, sid) for sid, d in pol_deg.items()])
    n_with_pol = len(pol_deg)
    top_pol = sorted(pol_deg.items(), key=lambda x: -x[1])[:5]
    print(f'    {n_with_pol} statutes with political degree')
    for sid, d in top_pol:
        title = c.execute('SELECT title FROM statutes WHERE id = ?', (sid,)).fetchone()
        print(f'      {sid:>15}  pol_deg={d:>6}  {(title[0] or "")[:45] if title else ""}')

    # ---- Co-amendment PageRank (political centrality with transitivity) ----
    print('  Computing co-amendment PageRank...')
    # Build adjacency from co_amendments (undirected weighted graph)
    # For PageRank on undirected: treat each edge as bidirectional
    coamend_adj = defaultdict(list)   # node → [(neighbor, weight), ...]
    coamend_out_w = defaultdict(float)  # node → total outgoing weight
    for row in c.execute('SELECT statute_a, statute_b, co_count FROM co_amendments WHERE coupling_type = ?', ('HE',)):
        a, b, w = row
        coamend_adj[a].append((b, w))
        coamend_adj[b].append((a, w))
        coamend_out_w[a] += w
        coamend_out_w[b] += w
    coamend_nodes = set(coamend_adj.keys())
    n_coamend = len(coamend_nodes)
    if n_coamend > 0:
        damping = 0.85
        rank = {n: 1.0 / n_coamend for n in coamend_nodes}
        for iteration in range(100):
            new_rank = {}
            for node in coamend_nodes:
                s = 0.0
                for neighbor, w in coamend_adj[node]:
                    s += rank[neighbor] * w / coamend_out_w[neighbor]
                new_rank[node] = (1 - damping) / n_coamend + damping * s
            # Check convergence
            diff = sum(abs(new_rank[n] - rank[n]) for n in coamend_nodes)
            rank = new_rank
            if diff < 1e-8:
                break
        # Normalize to 0-1 range
        max_rank = max(rank.values()) if rank else 1.0
        rank_norm = {n: r / max_rank for n, r in rank.items()}
        c.executemany('UPDATE statutes SET he_coamend_pagerank = ? WHERE id = ?',
                      [(r, sid) for sid, r in rank_norm.items()])
        top_cpr = sorted(rank_norm.items(), key=lambda x: -x[1])[:5]
        print(f'    {n_coamend} nodes, converged in {iteration+1} iterations')
        for sid, r in top_cpr:
            title = c.execute('SELECT title FROM statutes WHERE id = ?', (sid,)).fetchone()
            print(f'      {sid:>15}  coamend_pr={r:.4f}  {(title[0] or "")[:45] if title else ""}')
    else:
        print('    No co-amendment edges — skipping')

    # ---- HE→HE PageRank (policy genealogy importance) ----
    print('  Computing HE citation PageRank...')
    # Only include HEs that exist in he_nodes (exclude ghost refs to pre-1992 HEs)
    known_he_ids = set(r[0] for r in c.execute('SELECT he_id FROM he_nodes'))
    # Deduplicate HE→HE edges, filtering to known HEs only
    he_he_dedup = defaultdict(int)  # (source, target) → count of refs
    for row in c.execute('SELECT source_he, target_he FROM he_he_refs'):
        if row[0] in known_he_ids and row[1] in known_he_ids:
            he_he_dedup[(row[0], row[1])] += 1
    # Build adjacency for directed graph (source cites target)
    he_adj_in = defaultdict(list)    # target → [(source, weight)]
    he_out_w = defaultdict(float)
    for (src, tgt), w in he_he_dedup.items():
        he_adj_in[tgt].append((src, w))
        he_out_w[src] += w
    n_he = len(known_he_ids)
    if n_he > 0 and he_he_dedup:
        damping = 0.85
        he_rank = {h: 1.0 / n_he for h in known_he_ids}
        for iteration in range(100):
            new_rank = {}
            for node in known_he_ids:
                s = 0.0
                for src, w in he_adj_in.get(node, []):
                    s += he_rank[src] * w / he_out_w[src]
                new_rank[node] = (1 - damping) / n_he + damping * s
            diff = sum(abs(new_rank[n] - he_rank[n]) for n in known_he_ids)
            he_rank = new_rank
            if diff < 1e-8:
                break
        max_he_rank = max(he_rank.values()) if he_rank else 1.0
        he_rank_norm = {h: r / max_he_rank for h, r in he_rank.items()}
        c.executemany('UPDATE he_nodes SET he_pagerank = ? WHERE he_id = ?',
                      [(r, hid) for hid, r in he_rank_norm.items()])
        top_hepr = sorted(he_rank_norm.items(), key=lambda x: -x[1])[:10]
        print(f'    {n_he} HEs, {len(he_he_dedup)} unique edges, converged in {iteration+1} iterations')
        for hid, r in top_hepr:
            row = c.execute('SELECT year, title FROM he_nodes WHERE he_id = ?', (hid,)).fetchone()
            yr, title = row if row else (0, '')
            raw_cited = len(he_adj_in.get(hid, []))
            print(f'      {hid:>15}  pr={r:.4f}  cited={raw_cited:>4}  {yr}  {(title or "")[:40]}')
    else:
        print('    No HE→HE edges — skipping')

    # ---- Rank divergence: legal PageRank rank vs co-amendment PageRank rank ----
    print('  Computing rank divergence...')
    # Get all statutes with both metrics
    ranked_statutes = c.execute('''
        SELECT id, budget_combined_propagated_eur, he_coamend_pagerank
        FROM statutes
        WHERE budget_combined_propagated_eur > 0 OR he_coamend_pagerank > 0
        ORDER BY id
    ''').fetchall()
    if ranked_statutes:
        # Rank by legal PageRank (budget_combined_propagated_eur)
        by_legal = sorted(ranked_statutes, key=lambda x: -x[1])
        legal_rank = {r[0]: i + 1 for i, r in enumerate(by_legal)}
        # Rank by co-amendment PageRank
        by_pol = sorted(ranked_statutes, key=lambda x: -x[2])
        pol_rank = {r[0]: i + 1 for i, r in enumerate(by_pol)}
        # Divergence = log2(legal_rank / pol_rank)
        # Positive = politically more important than legally
        # Negative = legally more important than politically
        import math
        divergence = {}
        for sid in legal_rank:
            lr = legal_rank[sid]
            pr = pol_rank[sid]
            if lr > 0 and pr > 0:
                divergence[sid] = math.log2(lr / pr)
        c.executemany('UPDATE statutes SET he_rank_divergence = ? WHERE id = ?',
                      [(d, sid) for sid, d in divergence.items()])
        # Show extremes
        top_pol_over = sorted(divergence.items(), key=lambda x: -x[1])[:5]
        top_legal_over = sorted(divergence.items(), key=lambda x: x[1])[:5]
        print(f'    {len(divergence)} statutes with divergence scores')
        print('    Most POLITICALLY over-ranked (high positive divergence):')
        for sid, d in top_pol_over:
            title = c.execute('SELECT title FROM statutes WHERE id = ?', (sid,)).fetchone()
            print(f'      {sid:>15}  div={d:>+6.2f}  legal_rank={legal_rank[sid]:>5}  pol_rank={pol_rank[sid]:>5}  {(title[0] or "")[:35] if title else ""}')
        print('    Most LEGALLY over-ranked (high negative divergence):')
        for sid, d in top_legal_over:
            title = c.execute('SELECT title FROM statutes WHERE id = ?', (sid,)).fetchone()
            print(f'      {sid:>15}  div={d:>+6.2f}  legal_rank={legal_rank[sid]:>5}  pol_rank={pol_rank[sid]:>5}  {(title[0] or "")[:35] if title else ""}')

    # Editorial notes
    c.execute('''CREATE TABLE editorial_notes (
        statute_id TEXT NOT NULL,
        note_text TEXT NOT NULL,
        FOREIGN KEY (statute_id) REFERENCES statutes(id)
    )''')
    if editorial_notes:
        c.executemany('INSERT INTO editorial_notes VALUES (?,?)',
                      [(r['statute_id'], r['note_text']) for r in editorial_notes])
    print(f'  editorial_notes: {len(editorial_notes)}')

    # ========== INDEXES ==========

    print('Creating indexes...')
    c.execute('CREATE INDEX idx_edges_source ON edges(source)')
    c.execute('CREATE INDEX idx_edges_target ON edges(target)')
    c.execute('CREATE INDEX idx_edges_type ON edges(edge_type)')
    c.execute('CREATE INDEX idx_mapping_momentti ON mapping(momentti_code)')
    c.execute('CREATE INDEX idx_mapping_statute ON mapping(statute_id)')
    c.execute('CREATE INDEX idx_statutes_budget ON statutes(budget_direct_eur)')
    c.execute('CREATE INDEX idx_statutes_combined ON statutes(budget_combined_eur)')
    c.execute('CREATE INDEX idx_statutes_combined_prop ON statutes(budget_combined_propagated_eur)')
    c.execute('CREATE INDEX idx_statutes_in_degree ON statutes(in_degree)')
    c.execute('CREATE INDEX idx_statutes_ministry ON statutes(ministry)')
    c.execute('CREATE INDEX idx_statutes_year ON statutes(year)')
    c.execute('CREATE INDEX idx_statutes_keywords ON statutes(keywords)')
    c.execute('CREATE INDEX idx_statutes_cofog ON statutes(budget_cofog_eur)')
    c.execute('CREATE INDEX idx_statutes_katz ON statutes(budget_katz_eur)')
    c.execute('CREATE INDEX idx_cofog_mapping_cofog ON cofog_mapping(cofog_code)')
    c.execute('CREATE INDEX idx_cofog_mapping_statute ON cofog_mapping(statute_id)')
    c.execute('CREATE INDEX idx_statutes_inst_aum ON statutes(budget_institution_aum_eur)')
    c.execute('CREATE INDEX idx_inst_edges_inst ON institution_statute_edges(institution_id)')
    c.execute('CREATE INDEX idx_inst_edges_statute ON institution_statute_edges(statute_id)')
    c.execute('CREATE INDEX idx_statutes_delegation ON statutes(delegation_total)')
    c.execute('CREATE INDEX idx_deleg_statute ON delegations(statute_id)')
    c.execute('CREATE INDEX idx_deleg_type ON delegations(delegation_type)')
    c.execute('CREATE INDEX idx_auth_asetus ON asetus_authority(asetus_id)')
    c.execute('CREATE INDEX idx_auth_parent ON asetus_authority(parent_statute_id)')
    c.execute('CREATE INDEX idx_he_nodes_year ON he_nodes(year)')
    c.execute('CREATE INDEX idx_he_link_he ON he_statute_link(he_id)')
    c.execute('CREATE INDEX idx_he_link_statute ON he_statute_link(statute_id)')
    c.execute('CREATE INDEX idx_he_link_type ON he_statute_link(ref_type)')
    c.execute('CREATE INDEX idx_he_he_source ON he_he_refs(source_he)')
    c.execute('CREATE INDEX idx_he_he_target ON he_he_refs(target_he)')
    c.execute('CREATE INDEX idx_he_claims_he ON he_claims(he_id)')
    c.execute('CREATE INDEX idx_he_claims_type ON he_claims(claim_type)')
    c.execute('CREATE INDEX idx_he_claims_amount ON he_claims(amount_eur)')
    c.execute('CREATE INDEX idx_csl_statute ON claim_statute_link(statute_id)')
    c.execute('CREATE INDEX idx_csl_claim ON claim_statute_link(claim_id)')
    c.execute('CREATE INDEX idx_csl_attribution ON claim_statute_link(attribution)')
    c.execute('CREATE INDEX idx_statutes_bowtie ON statutes(bowtie_zone)')
    c.execute('CREATE INDEX idx_editorial_notes ON editorial_notes(statute_id)')
    c.execute('CREATE INDEX idx_statutes_churn ON statutes(churn_rate)')
    c.execute('CREATE INDEX idx_statutes_churn_contagion ON statutes(churn_contagion)')
    c.execute('CREATE INDEX idx_statutes_dead_ref ON statutes(dead_ref_rate)')
    c.execute('CREATE INDEX idx_co_amendments_a ON co_amendments(statute_a)')
    c.execute('CREATE INDEX idx_co_amendments_b ON co_amendments(statute_b)')

    # FTS5 (now includes title)
    print('Building FTS index...')
    c.execute('''CREATE VIRTUAL TABLE statutes_fts USING fts5(
        id, title, common_name, keywords, ministry, content=statutes
    )''')
    c.execute('''INSERT INTO statutes_fts(id, title, common_name, keywords, ministry)
        SELECT id, title, common_name, keywords, ministry FROM statutes''')

    conn.commit()

    # ========== SUMMARY ==========

    print('\n--- TOP 10 BY COMBINED PROPAGATED WEIGHT (PageRank) ---')
    for row in c.execute('''
        SELECT id, title, budget_combined_propagated_eur/1e6 as prop_m,
               budget_combined_eur/1e6 as direct_m, in_degree
        FROM statutes ORDER BY budget_combined_propagated_eur DESC LIMIT 10
    '''):
        print(f'  {row[0]:>12}  prop={row[2]:8,.0f}M  direct={row[3]:8,.0f}M  deg={row[4]:>4}  {(row[1] or "")[:50]}')

    print('\n--- TOP 10 BY KATZ CENTRALITY (structural load) ---')
    for row in c.execute('''
        SELECT id, title, budget_katz_eur/1e6 as katz_m,
               budget_combined_propagated_eur/1e6 as pr_m, in_degree
        FROM statutes ORDER BY budget_katz_eur DESC LIMIT 10
    '''):
        print(f'  {row[0]:>12}  katz={row[2]:8,.0f}M  pagerank={row[3]:8,.0f}M  deg={row[4]:>4}  {(row[1] or "")[:50]}')

    if he_claims_rows:
        print('\n--- HE CLAIMS SUMMARY ---')
        for row in c.execute('''
            SELECT claim_type, count(*), coalesce(sum(amount_eur), 0)/1e6
            FROM he_claims GROUP BY claim_type ORDER BY count(*) DESC
        '''):
            print(f'  {row[0]:15s}  n={row[1]:5d}  total_eur={row[2]:10,.0f}M')
        total = c.execute('SELECT count(*) FROM he_claims').fetchone()[0]
        he_count = c.execute('SELECT count(DISTINCT he_id) FROM he_claims').fetchone()[0]
        print(f'  Total: {total} claims from {he_count} HEs')

    print('\n--- TOP 15 BY CHURN CONTAGION (systemic instability) ---')
    for row in c.execute('''
        SELECT id, title, churn_contagion/1e9, churn_rate, debtrank_eur/1e9, amendment_count
        FROM statutes WHERE in_force = 1 AND churn_contagion > 0
        ORDER BY churn_contagion DESC LIMIT 15
    '''):
        print(f'  {row[0]:>15}  cc={row[2]:8.1f}B  churn={row[3]:.2f}/yr  dr={row[4]:.1f}B  amend={row[5]:>3}  {(row[1] or "")[:45]}')

    print('\n--- TOP 15 BY DEAD REFERENCE RATE (in-force Laki, >=5 PEER refs) ---')
    for row in c.execute('''
        SELECT id, title, dead_ref_rate, out_degree, amendment_count
        FROM statutes WHERE in_force = 1 AND dead_ref_rate > 0.3
            AND type_statute = 'Laki' AND out_degree >= 5
            AND title NOT LIKE '%kumo%' AND title NOT LIKE '%voimaansaat%'
        ORDER BY dead_ref_rate DESC LIMIT 15
    '''):
        print(f'  {row[0]:>15}  dead={row[2]:.1%}  deg_out={row[3]:>4}  amend={row[4]:>3}  {(row[1] or "")[:45]}')

    print('\n--- TOP 15 ARCHITECTURALLY BRITTLE (high delegation + high churn) ---')
    for row in c.execute('''
        SELECT id, title, delegation_total, churn_rate, amendment_count, delegation_gap
        FROM statutes WHERE in_force = 1 AND delegation_total >= 5 AND churn_rate >= 1.5
        ORDER BY delegation_total * churn_rate DESC LIMIT 15
    '''):
        print(f'  {row[0]:>15}  deleg={row[2]:>3}  churn={row[3]:.2f}/yr  amend={row[4]:>3}  gap={row[5]:>2}  {(row[1] or "")[:40]}')

    print('\n--- TOP 15 HOLLOW MANDATES (delegation_gap / delegation_total) ---')
    for row in c.execute('''
        SELECT id, title, delegation_total, delegation_gap, hollow_mandate_rate
        FROM statutes WHERE in_force = 1 AND delegation_total >= 5 AND hollow_mandate_rate > 0.5
        ORDER BY delegation_total DESC LIMIT 15
    '''):
        print(f'  {row[0]:>15}  deleg={row[2]:>3}  gap={row[3]:>3}  hollow={row[4]:.0%}  {(row[1] or "")[:45]}')

    co_date = c.execute("SELECT COUNT(*) FROM co_amendments WHERE coupling_type='DATE'").fetchone()[0]
    co_he = c.execute("SELECT COUNT(*) FROM co_amendments WHERE coupling_type='HE'").fetchone()[0]
    print(f'\n--- CHANGE COUPLING: {co_date + co_he} co-amendment pairs ({co_date} DATE, {co_he} HE) ---')
    for row in c.execute('''
        SELECT ca.statute_a, s1.title, ca.statute_b, s2.title, ca.co_count, ca.coupling_type
        FROM co_amendments ca
        JOIN statutes s1 ON ca.statute_a = s1.id
        JOIN statutes s2 ON ca.statute_b = s2.id
        ORDER BY ca.co_count DESC LIMIT 20
    '''):
        print(f'  {row[4]:>3}× [{row[5]:4s}]  {row[0]:>15} ({(row[1] or "")[:22]})  <->  {row[2]:>15} ({(row[3] or "")[:22]})')

    if he_nodes_data:
        he_link_count = c.execute('SELECT COUNT(*) FROM he_statute_link').fetchone()[0]
        he_statute_count = c.execute('SELECT COUNT(DISTINCT statute_id) FROM he_statute_link WHERE statute_id IN (SELECT id FROM statutes)').fetchone()[0]
        print(f'\n--- HE GRAPH: {len(he_nodes_data)} HEs, {he_link_count} edges, {he_statute_count} statutes linked ---')
        for row in c.execute('''
            SELECT hl.statute_id, s.title, COUNT(*) as n_hes,
                   SUM(CASE WHEN hl.ref_type='AMENDS' THEN 1 ELSE 0 END) as n_amends
            FROM he_statute_link hl
            JOIN statutes s ON hl.statute_id = s.id
            GROUP BY hl.statute_id
            ORDER BY n_hes DESC LIMIT 10
        '''):
            print(f'  {row[0]:>15}  {row[2]:>4} HEs ({row[3]:>2} amend)  {(row[1] or "")[:50]}')

    conn.close()

    db_size = DB_PATH.stat().st_size
    print(f'\nWrote {DB_PATH} ({db_size / 1e6:.1f} MB)')

    # Entity aggregation (committees, orgs, ministers, expert persons)
    # Runs as additive post-build step — scans per-HE DBs and AKN XML
    print('\n========== ENTITY AGGREGATION ==========')
    _aggregate_entities()

    # ── Delegation drift at scale (Tier 2.1) ──
    # Copy delegation_drift_scale from he_enrichments.db if available
    ENRICHMENTS_FOR_DRIFT = _ENRICHMENTS_DB
    if ENRICHMENTS_FOR_DRIFT.exists():
        print('\n========== DELEGATION DRIFT (Tier 2.1) ==========')
        try:
            econn = sqlite3.connect(str(ENRICHMENTS_FOR_DRIFT))
            drift_rows = econn.execute('''
                SELECT statute_id, section, delegation_type, is_mandatory, asetus_id, asetus_title,
                       he_claim_count, drift_verdict, drift_explanation, match_text
                FROM delegation_drift_scale
            ''').fetchall()
            econn.close()
            dconn = sqlite3.connect(str(DB_PATH))
            dconn.executescript('''
                DROP TABLE IF EXISTS delegation_drift_scale;
                CREATE TABLE delegation_drift_scale (
                    statute_id TEXT, section TEXT, delegation_type TEXT, is_mandatory INTEGER,
                    asetus_id TEXT, asetus_title TEXT, he_claim_count INTEGER,
                    drift_verdict TEXT, drift_explanation TEXT, match_text TEXT,
                    PRIMARY KEY (statute_id, section, asetus_id)
                );
            ''')
            dconn.executemany('''
                INSERT OR REPLACE INTO delegation_drift_scale
                VALUES (?,?,?,?,?,?,?,?,?,?)
            ''', drift_rows)
            for col in ('drift_total', 'drift_mandatory'):
                try:
                    dconn.execute(f'ALTER TABLE statutes ADD COLUMN {col} INTEGER')
                except Exception:
                    pass
            dconn.execute('''
                UPDATE statutes SET
                    drift_total = (SELECT COUNT(*) FROM delegation_drift_scale d
                                   WHERE d.statute_id = statutes.id AND d.drift_verdict = 'DRIFT'),
                    drift_mandatory = (SELECT COUNT(*) FROM delegation_drift_scale d
                                       WHERE d.statute_id = statutes.id AND d.drift_verdict = 'DRIFT'
                                       AND d.is_mandatory = 1)
            ''')
            dconn.commit()
            dconn.close()
            nd = sum(1 for r in drift_rows if r[7] == 'DRIFT')
            nm = sum(1 for r in drift_rows if r[7] == 'DRIFT' and r[3] == 1)
            print(f'  {len(drift_rows)} pairs | {nd} DRIFT | {nm} mandatory DRIFT')
        except Exception as e:
            print(f'  Warning: could not embed delegation drift: {e}')
    else:
        print('\nNo he_enrichments.db — skipping delegation drift (run: mev detect drift --llm --embed)')

    db_size = DB_PATH.stat().st_size
    print(f'\nFinal DB: {DB_PATH} ({db_size / 1e6:.1f} MB)')
    # Publishing is manual. No host or filesystem layout is embedded here.
    import os
    target = os.environ.get("DEPLOY_RSYNC_TARGET")
    if target:
        print("DEPLOY_RSYNC_TARGET is set. This command does not upload.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build state_causal_map.db')
    parser.add_argument(
        '--graph-dir',
        metavar='DIR',
        help='Path to lawvm build artifact directory (replaces edges.csv / statute_titles.csv)',
    )
    args = parser.parse_args()
    graph_dir = Path(args.graph_dir) if args.graph_dir else None
    build(graph_dir=graph_dir)

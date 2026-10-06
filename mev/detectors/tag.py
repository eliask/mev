"""
Sentence-level argument tagger for HE impact assessments.

Tags each sentence with three dimensions:
  Role:    P(premise) E(estimate) V(claim) K(caveat) L(promise)
  Quality: G(grounded) M(modeled) A(asserted) H(hedged) T(uncertain)
  Topic:   F(fiscal) H(human) I(institutional) C(cognitive) S(social)
           D(demographic) N(natural) R(moral) X(other)

Pre-filters table rows (tagged mechanically). Sends prose to LLM.
Detects irony patterns: caveats near claims, hedge-then-precision, ungrounded promises.
Extracts EUR amounts mechanically.

Replaces: extract_claims_llm.py, enrich_atoms_llm.py (LLM phase)

Usage:
    mev detect tag                    # HE 241/2020, console only
    mev detect tag he-38-2025         # specific HE
    mev detect tag --write-db         # persist to he_enrichments.db
    mev detect tag --max-atoms 5      # test on 5 atoms
    mev detect tag --parallel 4       # fewer concurrent requests
"""

import argparse
import asyncio
import json
import os
import re
import sqlite3
import sys
from collections import defaultdict

import aiohttp

from mev.config import ROOT, HE_DB_DIR, ENRICHMENTS_DB
from mev.llm import call_llm_full, LLMContextExhausted
from mev.versioning import extractor_version, stale_keys, stamp_version
from mev.detectors.tagger_engine import (
    TagConfig,
    call_llm_windowed,
    parse_tag_lines,
    tag_units_windowed,
)

RESULTS_DIR = ROOT / ".tmp" / "sentence_tag"

# ---------------------------------------------------------------------------
# LLM wire codes → human-readable labels for DB/JSON serialization
# ---------------------------------------------------------------------------

ROLE_LABELS = {
    'P': 'premise', 'E': 'estimate', 'V': 'claim',
    'K': 'caveat', 'L': 'promise', '!': 'anomaly',
}
QUALITY_LABELS = {
    'G': 'grounded', 'M': 'modeled', 'A': 'asserted',
    'H': 'hedged', 'T': 'uncertain', '!': 'anomaly',
}
# Topic codes match the 9 capital stocks from Kansallistase (mekanismirealismi ch.3)
# + fiscal (flow, not stock — but HEs obsess over it, tracking shows the blindness)
TOPIC_LABELS = {
    'F': 'fiscal',          # not a capital stock; flow (euroa, budjetti)
    'W': 'epistemic',       # kyky tietää mikä on totta (tietopohja, tilastointi)
    'S': 'social',          # luottamus, alhaiset transaktiokustannukset
    'C': 'cognitive',       # erikoisosaaminen, teknologia, innovaatio
    'I': 'institutional',   # oikeusvaltio, hallinto, viranomaisrakenteet
    'N': 'infrastructure',  # tilat, rakennukset, verkot, fyysinen infra
    'D': 'human',           # ihmiset: väestö, henkilöstö, terveys, ikärakenne
    'Y': 'coherence',       # jaettu kieli, normit, koordinaatiokyky
    'J': 'purpose',         # jaettu ymmärrys, miksi olemme olemassa
    'R': 'moral',           # legitimiteetti, johtajien uskottavuus
    'X': 'other',
}

VALID_ROLES = set(ROLE_LABELS.keys())
VALID_QUALS = set(QUALITY_LABELS.keys())
VALID_TOPICS = set(TOPIC_LABELS.keys())


def role_label(code):
    return ROLE_LABELS.get(code, code)

def qual_label(code):
    return QUALITY_LABELS.get(code, code)

def topic_label(code):
    return TOPIC_LABELS.get(code, code)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM = """Luokittele hallituksen esityksen lauseet kolmella ulottuvuudella.

ROOLI (P/E/V/K/L):
  P = premissi, taustatieto, nykytila
  E = estimaatti, laskelma, projektio
  V = väite ehdotetun muutoksen vaikutuksesta
  K = kaveat, rajoitteen tunnustus
  L = lupaus tulevasta seurannasta

LAATU (G/M/A/H/T):
  G = grounded, viittaa dataan
  M = mallinnettu, parametrit näkyvissä
  A = assertoitu ilman perustelua
  H = hedged ("arvioidaan", "noin")
  T = tunnustettu epävarma ("ei voida arvioida")

AIHE (F/W/S/C/I/N/D/Y/J/R/X):
  F = fiskaalinen  W = episteeminen  S = sosiaalinen
  C = kognitiivinen  I = institutionaalinen  N = infrastruktuuri
  D = inhimillinen  Y = yhtenäisyys  J = tarkoitus
  R = moraalinen  X = muu

Tulosta VAIN rivit: NUMERO ROOLI LAATU AIHE
Ei selityksiä, ei sulkeita, ei muuta tekstiä.
Jos lause ohitetaan, ÄLÄ tulosta sille riviä.

Esimerkki — syöte:
[3] Vuonna 2019 kuntien verotulot olivat 23,2 miljardia euroa.
[4] Sosiaali- ja terveystoimen kokonaismenot olivat 21,1 miljardia euroa.
[5] Lakiviittaus: Hallituksen esitys laiksi (HE 15/2017).
[6] Kustannusvaikutus on arviolta 150 miljoonaa euroa vuodessa.
[7] Uudistus parantaisi hallinnon toimivuutta.
[8] Vaikutuksia ei voida tarkasti arvioida.
[9] Muutos lisäisi kansalaisten luottamusta palveluihin.
[10] Hallitus seuraa uudistuksen toteutumista.

Esimerkki — tuloste:
3 P G F
4 P G F
6 E M F
7 V H I
8 K T X
9 V A S
10 L A I"""


# ---------------------------------------------------------------------------
# Data loading and sentence extraction
# ---------------------------------------------------------------------------

def load_atoms(he_id, min_len=500, max_len=20000):
    db_path = HE_DB_DIR / f"{he_id}.db"
    if not db_path.exists():
        print(f"Error: {db_path} not found", file=sys.stderr)
        sys.exit(1)
    conn = sqlite3.connect(str(db_path), timeout=30)
    rows = conn.execute(
        "SELECT atom_id, content FROM atoms WHERE atom_type='IMPACT' "
        "AND length(content) BETWEEN ? AND ? ORDER BY atom_id",
        (min_len, max_len)
    ).fetchall()
    conn.close()
    return rows


def index_sents(text):
    """Split text into sentences. Returns [(text, kind)] where kind is prose/table/caption."""
    results = []
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
        if line.startswith('|'):
            if len(line) > 15 and '---' not in line:
                results.append((line, 'table'))
            continue
        if line.startswith('Taulukko'):
            results.append((line, 'caption'))
            continue
        parts = re.split(r'(?<=[.!?])\s+(?=[A-ZÄÖÅ0-9(])', line)
        for p in parts:
            p = p.strip()
            if len(p) > 15:
                results.append((p, 'prose'))
    return results


def extract_eur(text):
    amounts = []
    for m in re.finditer(
        r'([\d,.\s]+)\s*(milj(?:ard[ia]|\.)?|mrd\.?|miljoon\w+)\s*(?:euroa|€)',
        text, re.IGNORECASE
    ):
        num_str = m.group(1).strip().replace(' ', '').replace(',', '.')
        try:
            num = float(num_str)
        except ValueError:
            continue
        mult = m.group(2).lower()
        if 'mrd' in mult or 'miljard' in mult:
            amounts.append(num * 1e9)
        elif 'milj' in mult:
            amounts.append(num * 1e6)
    for m in re.finditer(r'([\d\s,.]+)\s*euroa?\b', text, re.IGNORECASE):
        num_str = m.group(1).strip().replace(' ', '').replace(',', '.')
        try:
            num = float(num_str)
            if num > 1000 and num not in amounts:
                amounts.append(num)
        except ValueError:
            continue
    return amounts


RE_HEDGE = re.compile(
    r'arvioidaan|arvion mukaan|noin |todennäköis|oletettavasti|'
    r'voidaan arvioida|mahdollisesti|likimäärin',
    re.IGNORECASE
)
RE_CAVEAT = re.compile(
    r'ei voida|ei pystytä|mahdoton arvioi|ei ole mahdollista|'
    r'vaikea arvioida|epävarm|ei voitu|ei kyetä|ei tiedetä|'
    r'ei ole arvioitu|huomattav\w+ epävarmuut',
    re.IGNORECASE
)

# Regex topic detection for table rows (which don't go to LLM)
TOPIC_HINTS = {
    'F': re.compile(r'euroa|milj\.|mrd\.|kustannu|säästö|menot|tulot|budjetti|rahoitu|vero', re.I),
    'W': re.compile(r'tietopohj|arviointiky|tilastoi|tutkimusky|tieto(?:a|ja)\b', re.I),
    'S': re.compile(r'luottamu|yhteisö|osallistu|legitimiteetti|transaktiokustannu', re.I),
    'C': re.compile(r'osaami|erikoisosaami|tietojärjestelm|digitali|ICT|asiantunti|innovaati|teknolog', re.I),
    'I': re.compile(r'viranomai|hallinto|organisaatio|kapasiteetti|toimintakyky|järjestämi|oikeusvalt', re.I),
    'N': re.compile(r'rakennuk|sairaala|tiet|verkko|infrastru|tilat\b|toimitil', re.I),
    'D': re.compile(r'syntyvyy|ikäänty|väestö|huoltosuhde|muuttolii|maahanmuut|henkilöstö|työntekij|tervey', re.I),
    'Y': re.compile(r'yhtenäisyy|koordinaatio|normit|jaettu kieli|protokolla', re.I),
    'J': re.compile(r'tarkoitus|merkityks|miksi olemme', re.I),
    'R': re.compile(r'legitimiteetti|uskottavu|vaikeiden päätös|oikeudenmukai|perusoikeu', re.I),
}


def guess_topic_regex(text):
    """Best-effort topic from regex. Returns code or 'X'."""
    hits = {}
    lower = text.lower()
    for code, pat in TOPIC_HINTS.items():
        n = len(pat.findall(lower))
        if n:
            hits[code] = n
    if not hits:
        return 'X'
    return max(hits, key=hits.get)


# ---------------------------------------------------------------------------
# LLM interface (delegates to engine)
# ---------------------------------------------------------------------------

async def call_llm(session, sem, system, user, max_tokens=300):
    """Thin wrapper: acquires sem then delegates to shared call_llm_full (cached)."""
    return await call_llm_windowed(session, sem, system, user, max_tokens)


def parse_tags(raw):
    """Parse 'ID ROLE QUALITY TOPIC' lines. Returns {sid: (role, qual, topic)}."""
    return parse_tag_lines(raw, VALID_ROLES, VALID_QUALS, VALID_TOPICS)


# ---------------------------------------------------------------------------
# Irony pattern detection (unique to HE tagger)
# ---------------------------------------------------------------------------

def find_irony_pattern(tags, sents):
    patterns = []
    tagged_list = sorted(tags.items())

    # Pattern 1: Caveat/admitted-uncertain near assertive claim
    caveats = {sid for sid, (r, q, _t) in tags.items() if r == 'K' or q == 'T'}
    for sid, (role, qual, topic) in tagged_list:
        if role in ('E', 'V') and qual in ('A', 'G', 'M'):
            for c_sid in caveats:
                if 0 < abs(sid - c_sid) <= 4:
                    cr, cq, ct = tags[c_sid]
                    patterns.append({
                        'type': 'caveat_near_claim',
                        'claim_sid': sid,
                        'claim_role': role_label(role),
                        'claim_quality': qual_label(qual),
                        'claim_topic': topic_label(topic),
                        'caveat_sid': c_sid,
                        'caveat_role': role_label(cr),
                        'caveat_quality': qual_label(cq),
                        'claim_text': sents[sid][0][:100] if sid < len(sents) else '',
                        'caveat_text': sents[c_sid][0][:100] if c_sid < len(sents) else '',
                    })
                    break

    # Pattern 2: Hedge followed by precision (H then G/M within ±2)
    for i, (sid, (role, qual, topic)) in enumerate(tagged_list):
        if qual == 'H':
            for j in range(i + 1, min(i + 3, len(tagged_list))):
                sid2, (r2, q2, t2) = tagged_list[j]
                if q2 in ('G', 'M') and r2 in ('E', 'V'):
                    if abs(sid2 - sid) <= 3:
                        patterns.append({
                            'type': 'hedge_then_precision',
                            'hedge_sid': sid,
                            'hedge_role': role_label(role),
                            'hedge_quality': qual_label(qual),
                            'precise_sid': sid2,
                            'precise_role': role_label(r2),
                            'precise_quality': qual_label(q2),
                            'hedge_text': sents[sid][0][:100] if sid < len(sents) else '',
                            'precise_text': sents[sid2][0][:100] if sid2 < len(sents) else '',
                        })
                        break

    # Pattern 3: Promise without grounding (L + A/H)
    for sid, (role, qual, topic) in tagged_list:
        if role == 'L' and qual in ('A', 'H'):
            patterns.append({
                'type': 'ungrounded_promise',
                'sid': sid,
                'role': role_label(role),
                'quality': qual_label(qual),
                'topic': topic_label(topic),
                'text': sents[sid][0][:100] if sid < len(sents) else '',
            })

    # Pattern 4: Escalation (!)
    for sid, (role, qual, topic) in tagged_list:
        if role == '!' or qual == '!':
            patterns.append({
                'type': 'escalation',
                'sid': sid,
                'text': sents[sid][0][:100] if sid < len(sents) else '',
            })

    return patterns


# ---------------------------------------------------------------------------
# DB
# ---------------------------------------------------------------------------

def ensure_db_tables(db_path):
    conn = sqlite3.connect(str(db_path), timeout=30)
    # Migrate: drop old schema if missing topic column
    cur = conn.execute("PRAGMA table_info(sentence_tag)")
    cols = {row[1] for row in cur.fetchall()}
    if cols and 'topic' not in cols:
        conn.executescript("DROP TABLE IF EXISTS sentence_tag; DROP TABLE IF EXISTS irony_pattern;")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS sentence_tag (
            he_id       TEXT NOT NULL,
            atom_id     TEXT NOT NULL,
            sent_idx    INTEGER NOT NULL,
            sent_text   TEXT NOT NULL,
            sent_kind   TEXT NOT NULL,   -- prose/table/caption
            role        TEXT,            -- premise/estimate/claim/caveat/promise/anomaly
            quality     TEXT,            -- grounded/modeled/asserted/hedged/uncertain/anomaly
            topic       TEXT,            -- fiscal/human/institutional/cognitive/social/demographic/natural/moral/other
            eur_amounts TEXT,            -- JSON array of floats, NULL if none
            PRIMARY KEY (he_id, atom_id, sent_idx)
        );
        CREATE INDEX IF NOT EXISTS idx_st_he ON sentence_tag(he_id);
        CREATE INDEX IF NOT EXISTS idx_st_role ON sentence_tag(role);
        CREATE INDEX IF NOT EXISTS idx_st_quality ON sentence_tag(quality);
        CREATE INDEX IF NOT EXISTS idx_st_topic ON sentence_tag(topic);

        CREATE TABLE IF NOT EXISTS irony_pattern (
            he_id       TEXT NOT NULL,
            atom_id     TEXT NOT NULL,
            pattern_type TEXT NOT NULL,
            sid1        INTEGER NOT NULL,
            sid2        INTEGER,
            detail      TEXT,            -- JSON with roles, qualities, topics, text previews
            PRIMARY KEY (he_id, atom_id, pattern_type, sid1)
        );
        CREATE INDEX IF NOT EXISTS idx_ip_he ON irony_pattern(he_id);
        CREATE INDEX IF NOT EXISTS idx_ip_type ON irony_pattern(pattern_type);
    """)
    conn.close()


def write_to_db(db_path, he_id, all_results, all_sents_by_atom):
    conn = sqlite3.connect(str(db_path), timeout=30)
    conn.execute("DELETE FROM sentence_tag WHERE he_id=?", (he_id,))
    conn.execute("DELETE FROM irony_pattern WHERE he_id=?", (he_id,))

    tag_rows = []
    pattern_rows = []

    for atom_id, result in all_results.items():
        sents = all_sents_by_atom[atom_id]
        tags = result['_tags']

        for i, (text, kind) in enumerate(sents):
            tag = tags.get(i)
            if tag:
                code_r, code_q, code_t = tag
                r, q, t = role_label(code_r), qual_label(code_q), topic_label(code_t)
            else:
                r, q, t = None, None, None
            eurs = extract_eur(text)
            tag_rows.append((
                he_id, atom_id, i, text, kind,
                r, q, t,
                json.dumps(eurs) if eurs else None,
            ))

        for p in result['patterns']:
            ptype = p['type']
            if ptype == 'caveat_near_claim':
                sid1, sid2 = p['claim_sid'], p['caveat_sid']
            elif ptype == 'hedge_then_precision':
                sid1, sid2 = p['hedge_sid'], p['precise_sid']
            elif ptype == 'ungrounded_promise':
                sid1, sid2 = p['sid'], None
            elif ptype == 'escalation':
                sid1, sid2 = p['sid'], None
            else:
                continue
            detail = json.dumps({k: v for k, v in p.items()
                                 if k not in ('type', 'atom_id')},
                                ensure_ascii=False)
            pattern_rows.append((he_id, atom_id, ptype, sid1, sid2, detail))

    conn.executemany(
        "INSERT OR REPLACE INTO sentence_tag "
        "(he_id, atom_id, sent_idx, sent_text, sent_kind, role, quality, topic, eur_amounts) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        tag_rows
    )
    conn.executemany(
        "INSERT OR REPLACE INTO irony_pattern "
        "(he_id, atom_id, pattern_type, sid1, sid2, detail) "
        "VALUES (?,?,?,?,?,?)",
        pattern_rows
    )
    conn.commit()
    conn.close()
    return len(tag_rows), len(pattern_rows)


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

EXTRACTOR_VERSION = extractor_version(SYSTEM, "tag_he_v1")


def _list_all_he_ids() -> list[str]:
    """List all HE IDs that have IMPACT atoms."""
    return sorted(p.stem for p in HE_DB_DIR.glob('he-*.db'))


async def _process_one_he(he_id: str, args, session, sem) -> dict | None:
    """Process one HE. Returns summary dict or None on skip."""
    atoms = load_atoms(he_id)
    if not atoms:
        return None

    max_atoms = getattr(args, 'max_atoms', 0)
    if max_atoms > 0:
        atoms = atoms[:max_atoms]

    print(f"  {he_id}: {len(atoms)} IMPACT atoms", flush=True)

    all_results = {}
    all_sents_by_atom = {}
    total_tokens_in = 0
    total_tokens_out = 0
    total_patterns = defaultdict(int)
    total_sents = 0
    total_tagged = 0
    total_eur_sents = 0

    if True:  # session/sem passed from caller
        async def process_atom(atom_id, content):
            sents = index_sents(content)
            prose_sents = [(i, s, k) for i, (s, k) in enumerate(sents) if k == 'prose']
            table_sents = [(i, s, k) for i, (s, k) in enumerate(sents) if k == 'table']

            if not prose_sents:
                return atom_id, sents, None

            llm_input = '\n'.join(f'[{i}] {s}' for i, s, _ in prose_sents)
            max_tok = 20 + len(prose_sents) * 8  # ~6 tokens per tagged line + headroom

            resp = await call_llm(session, sem, SYSTEM, llm_input, max_tokens=max_tok)

            if resp.get('finish_reason') == 'context_overflow':
                # Section too long for single call — fall back to windowed processing
                _cfg = TagConfig(
                    system_prompt=SYSTEM,
                    valid_roles=VALID_ROLES, valid_quals=VALID_QUALS, valid_topics=VALID_TOPICS,
                    window=60, overlap=5,
                )
                _units = [(s, i) for i, s, _ in prose_sents]
                _windowed = await tag_units_windowed(session, sem, _cfg, _units)
                tags = {prose_sents[li][0]: v for li, v in _windowed.items()
                        if li < len(prose_sents)}
                resp = {'tokens_in': 0, 'tokens_out': 0, 'elapsed': 0.0,
                        'finish_reason': 'windowed'}
            elif 'error' in resp:
                return atom_id, sents, {'error': resp['error']}
            else:
                tags = parse_tags(resp['content'])
                # Retry on truncation
                if resp.get('finish_reason') == 'length':
                    resp2 = await call_llm(session, sem, SYSTEM, llm_input,
                                           max_tokens=max_tok * 2)
                    if 'error' not in resp2 and resp2.get('finish_reason') != 'length':
                        tags = parse_tags(resp2['content'])
                        resp = resp2

            # Mechanical tags for table rows
            for i, s, k in table_sents:
                eurs = extract_eur(s)
                topic = guess_topic_regex(s)
                if eurs:
                    tags[i] = ('E', 'M', topic if topic != 'X' else 'F')
                elif re.match(r'\|[^|]*[A-ZÄÖÅa-zäöå]{3}', s):
                    tags[i] = ('P', 'G', topic)

            # Regex post-processing: override quality when regex is more certain
            for sid, (role, qual, topic) in list(tags.items()):
                if sid < len(sents):
                    text = sents[sid][0]
                    if RE_CAVEAT.search(text) and qual != 'T':
                        tags[sid] = (role, 'T', topic)
                    elif RE_HEDGE.search(text) and qual == 'G':
                        tags[sid] = (role, 'H', topic)

            patterns = find_irony_pattern(tags, sents)

            eur_sents = []
            for sid, (role, qual, topic) in tags.items():
                if sid < len(sents):
                    amts = extract_eur(sents[sid][0])
                    if amts:
                        eur_sents.append({
                            'sid': sid, 'role': role_label(role),
                            'quality': qual_label(qual), 'topic': topic_label(topic),
                            'eur': amts, 'text': sents[sid][0][:120],
                        })

            role_dist = defaultdict(int)
            qual_dist = defaultdict(int)
            topic_dist = defaultdict(int)
            for r, q, t in tags.values():
                role_dist[role_label(r)] += 1
                qual_dist[qual_label(q)] += 1
                topic_dist[topic_label(t)] += 1

            return atom_id, sents, {
                'n_sents': len(sents),
                'n_prose': len(prose_sents),
                'n_table': len(table_sents),
                'n_tagged': len(tags),
                'role_dist': dict(role_dist),
                'qual_dist': dict(qual_dist),
                'topic_dist': dict(topic_dist),
                'patterns': patterns,
                'eur_sents': eur_sents,
                'tokens_in': resp['tokens_in'],
                'tokens_out': resp['tokens_out'],
                'elapsed_ms': resp['elapsed'] * 1000,
                'finish_reason': resp.get('finish_reason', ''),
                '_tags': dict(tags),
            }

        tasks = [process_atom(aid, content) for aid, content in atoms]
        for fut in asyncio.as_completed(tasks):
            atom_id, sents, result = await fut
            all_sents_by_atom[atom_id] = sents

            if result is None or 'error' in (result or {}):
                print(f"  {atom_id}: SKIP ({result})")
                continue

            all_results[atom_id] = result
            total_tokens_in += result['tokens_in']
            total_tokens_out += result['tokens_out']
            total_sents += result['n_sents']
            total_tagged += result['n_tagged']
            total_eur_sents += len(result['eur_sents'])
            for p in result['patterns']:
                total_patterns[p['type']] += 1

            n_patterns = len(result['patterns'])
            # Compact topic summary for per-atom line
            top_topics = sorted(result['topic_dist'].items(), key=lambda x: -x[1])[:3]
            topic_str = ' '.join(f"{t[0][:4]}={t[1]}" for t in top_topics)
            pat_str = f" [{n_patterns}pat]" if n_patterns else ""
            eur_str = f" [{len(result['eur_sents'])}€]" if result['eur_sents'] else ""
            trunc = " TRUNC" if result['finish_reason'] == 'length' else ""
            print(f"  {atom_id:25s} {result['n_prose']:3d}p/{result['n_table']:2d}t "
                  f"→ {result['n_tagged']:3d} tagged  "
                  f"{result['tokens_out']:3d}tok {result['elapsed_ms']:5.0f}ms  "
                  f"{topic_str}{pat_str}{eur_str}{trunc}")

    # --- Summary ---
    print(f"\n{'='*70}")
    print(f"SUMMARY — {he_id}")
    print(f"{'='*70}")
    print(f"Atoms processed: {len(all_results)}")
    print(f"Total sentences: {total_sents} ({total_tagged} tagged)")
    print(f"Tokens: {total_tokens_in} in / {total_tokens_out} out "
          f"({total_tokens_in/max(total_tokens_out,1):.0f}:1)")
    print(f"EUR-bearing sentences: {total_eur_sents}")

    agg_role = defaultdict(int)
    agg_qual = defaultdict(int)
    agg_topic = defaultdict(int)
    for r in all_results.values():
        for k, v in r['role_dist'].items():
            agg_role[k] += v
        for k, v in r['qual_dist'].items():
            agg_qual[k] += v
        for k, v in r['topic_dist'].items():
            agg_topic[k] += v

    print(f"\nRole distribution: {dict(sorted(agg_role.items()))}")
    print(f"Quality distribution: {dict(sorted(agg_qual.items()))}")
    print(f"Topic distribution: {dict(sorted(agg_topic.items(), key=lambda x: -x[1]))}")

    # Capital stock coverage gap analysis
    all_topics = set(TOPIC_LABELS.values()) - {'other'}
    present_topics = {t for t, c in agg_topic.items() if c > 0 and t != 'other'}
    missing = all_topics - present_topics
    if missing:
        print(f"\nCAPITAL STOCK GAPS (zero sentences): {', '.join(sorted(missing))}")

    print("\nIrony patterns:")
    for ptype, count in sorted(total_patterns.items(), key=lambda x: -x[1]):
        print(f"  {ptype}: {count}")

    all_patterns = []
    for atom_id, r in all_results.items():
        for p in r['patterns']:
            p['atom_id'] = atom_id
            all_patterns.append(p)

    caveat_claims = [p for p in all_patterns if p['type'] == 'caveat_near_claim']
    if caveat_claims:
        print(f"\nTOP CAVEAT-NEAR-CLAIM ({len(caveat_claims)} total):")
        for p in caveat_claims[:8]:
            print(f"  {p['atom_id']}:")
            ct = p.get('claim_topic', '')
            ct_str = f" [{ct}]" if ct else ""
            print(f"    CAVEAT [{p['caveat_sid']}] ({p['caveat_role']}/{p['caveat_quality']}): "
                  f"{p['caveat_text'][:75]}")
            print(f"    CLAIM  [{p['claim_sid']}] ({p['claim_role']}/{p['claim_quality']}{ct_str}): "
                  f"{p['claim_text'][:75]}")

    promises = [p for p in all_patterns if p['type'] == 'ungrounded_promise']
    if promises:
        print(f"\nUNGROUNDED PROMISES ({len(promises)} total):")
        for p in promises[:8]:
            t = p.get('topic', '')
            print(f"  [{p['atom_id']}:{p['sid']}] ({p['role']}/{p['quality']}"
                  f"{' '+t if t else ''}) {p['text'][:75]}")

    escalations = [p for p in all_patterns if p['type'] == 'escalation']
    if escalations:
        print(f"\nESCALATIONS ({len(escalations)} total):")
        for p in escalations[:8]:
            print(f"  [{p['atom_id']}:{p['sid']}] {p['text'][:80]}")

    # DB write
    if getattr(args, 'write_db', False):
        ensure_db_tables(ENRICHMENTS_DB)
        n_tags, n_pats = write_to_db(
            ENRICHMENTS_DB, he_id, all_results, all_sents_by_atom
        )
        stamp_version(ENRICHMENTS_DB, "sentence_tag", "he_id", he_id, EXTRACTOR_VERSION)
        print(f"  → DB: {n_tags} tags, {n_pats} patterns (v={EXTRACTOR_VERSION[:8]})")

    return {
        'he_id': he_id,
        'n_atoms': len(all_results),
        'n_sents': total_sents,
        'n_tagged': total_tagged,
        'tokens_in': total_tokens_in,
        'tokens_out': total_tokens_out,
    }


async def run(
    he_id: str | None = None,
    all_hes: bool = False,
    write_db: bool = False,
    force: bool = False,
    parallel: int = int(os.environ.get("LLM_PARALLEL", "4")),
    max_atoms: int = 0,
) -> dict:
    """Programmatic entry point. Returns summary dict."""
    import types
    args = types.SimpleNamespace(
        he_id=he_id,
        all=all_hes,
        write_db=write_db,
        force=force,
        parallel=parallel,
        max_atoms=max_atoms,
    )

    if all_hes:
        he_ids = _list_all_he_ids()
    elif he_id:
        he_ids = [he_id if he_id.startswith('he-') else f'he-{he_id}']
    else:
        he_ids = ['he-241-2020']

    if not force:
        before = len(he_ids)
        he_ids = stale_keys(ENRICHMENTS_DB, "sentence_tag", "he_id", he_ids, EXTRACTOR_VERSION)
        skipped = before - len(he_ids)
        if skipped:
            print(f"Skipping {skipped} up-to-date HEs (v={EXTRACTOR_VERSION[:8]}, use --force to recompute)")

    print(f"Processing {len(he_ids)} HEs\n")

    sem = asyncio.Semaphore(parallel)
    grand_total = {'sents': 0, 'tagged': 0, 'he_done': 0}

    async with aiohttp.ClientSession() as session:
        for hid in he_ids:
            result = await _process_one_he(hid, args, session, sem)
            if result:
                grand_total['sents'] += result['n_sents']
                grand_total['tagged'] += result['n_tagged']
                grand_total['he_done'] += 1

    print(f"\n{'='*60}")
    print(f"TOTAL: {grand_total['he_done']} HEs, "
          f"{grand_total['sents']} sentences, {grand_total['tagged']} tagged")

    return grand_total


async def main():
    parser = argparse.ArgumentParser(
        description="Tag HE impact assessment sentences by role, quality, and capital stock topic")
    parser.add_argument('he_id', nargs='?', default=None,
                        help='HE identifier (e.g. he-241-2020). Omit with --all for batch.')
    parser.add_argument('--all', action='store_true',
                        help='Process all HEs with IMPACT atoms')
    parser.add_argument('--write-db', action='store_true',
                        help='Write results to he_enrichments.db')
    parser.add_argument('--force', action='store_true',
                        help='Recompute even if already tagged in DB')
    parser.add_argument('--parallel', type=int, default=int(os.environ.get("LLM_PARALLEL", "4")),
                        help='Max concurrent LLM requests (default: 8)')
    parser.add_argument('--max-atoms', type=int, default=0,
                        help='Limit atoms per HE (0=all)')
    args = parser.parse_args()

    await run(
        he_id=args.he_id,
        all_hes=args.all,
        write_db=args.write_db,
        force=args.force,
        parallel=args.parallel,
        max_atoms=args.max_atoms,
    )


if __name__ == '__main__':
    asyncio.run(main())

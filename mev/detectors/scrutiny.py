"""
LLM-based scrutiny detector: match expert concerns to committee mietintö paragraphs.

Three-phase pipeline:
  Phase 0: Segment mietintö into numbered paragraphs (deterministic)
  Phase 1: Classify mietintö paragraphs via LLM (1 call per HE)
  Phase 2: Match expert sentences to mietintö paragraphs via LLM (1 call per expert)
  Phase 3: Aggregate into per-expert classifications (deterministic)
"""

import argparse
import asyncio
import os
import json
import re
import sqlite3
from collections import defaultdict

import aiohttp

from mev.config import ROOT, INDEX_DB, HE_DB_DIR
from mev.llm import call_llm, LLMContextExhausted

PARALLEL = int(os.environ.get("LLM_PARALLEL", "4"))
OUTPUT_DIR = ROOT / ".tmp" / "scrutiny_llm"

# ---------------------------------------------------------------------------
# Phase 0: Segment mietintö (deterministic)
# ---------------------------------------------------------------------------

PERUSTELUT_START = re.compile(
    r'(?:VALIOKUNNAN\s+(?:YLEIS)?PERUSTELUT|YLEISPERUSTELUT)',
    re.IGNORECASE
)
PERUSTELUT_END = re.compile(
    r'(?:YKSITYISKOHTAISET\s+PERUSTELUT|PÄÄTÖSEHDOTUS)',
    re.IGNORECASE
)

RE_TAGS = re.compile(r'<[^>]+>')
RE_MULTI_SPACE = re.compile(r'\s+')


def strip_html(text: str) -> str:
    clean = RE_TAGS.sub(' ', text)
    clean = clean.replace('&nbsp;', ' ').replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
    return RE_MULTI_SPACE.sub(' ', clean).strip()


def segment_mietinto(html: str) -> list[dict]:
    if not html:
        return []

    blocks = []
    for m in re.finditer(r'<(h[23]|p)\b[^>]*>(.*?)</\1>', html, re.IGNORECASE | re.DOTALL):
        tag = m.group(1).lower()
        raw = m.group(2)
        text = strip_html(raw)
        if not text or len(text) < 3:
            continue
        blocks.append({'tag': tag, 'text': text, 'is_header': tag.startswith('h'), 'pos': m.start()})

    if not blocks:
        return []

    start_idx = 0
    for i, b in enumerate(blocks):
        if PERUSTELUT_START.search(b['text']):
            start_idx = i
            break

    end_idx = len(blocks)
    for i in range(start_idx + 1, len(blocks)):
        if PERUSTELUT_END.search(blocks[i]['text']):
            end_idx = i
            break

    paragraphs = []
    idx = 1
    for b in blocks[start_idx:end_idx]:
        if len(b['text']) < 20:
            continue
        paragraphs.append({'idx': idx, 'text': b['text'], 'is_header': b['is_header']})
        idx += 1

    return paragraphs


# ---------------------------------------------------------------------------
# Phase 1: Classify mietintö paragraphs (LLM)
# ---------------------------------------------------------------------------

PHASE1_SYSTEM = """Luokittele mietinnön kappaleet sen mukaan, miten ne suhtautuvat asiantuntijoiden esittämiin huoliin tai muutosehdotuksiin.

Koodisto:
R = REQUIRES_CHANGE (valiokunta edellyttää tai ehdottaa muutosta)
N = NOTES_CONCERN (valiokunta kiinnittää huomiota huoleen ilman suoraa vaatimusta)
D = DEFERS (siirtää asian jatkovalmisteluun tai seurantaan)
A = APPROVES (puoltaa hallituksen esitystä, pitää kannatettavana)
X = DISMISSES (torjuu huolen, pitää nykyistä muotoilua riittävänä)
T = PROCEDURAL (vireilletulo, lausunnot, yleinen kuvaus)

Vastaa muodossa: NUMERO KOODI (yksi rivi per kappale).
Älä selitä, älä käytä sulkeita."""


def build_phase1_input(paragraphs: list[dict]) -> str:
    lines = []
    for p in paragraphs:
        prefix = " [OTSIKKO]" if p['is_header'] else ""
        lines.append(f"{{{p['idx']}{prefix}}} {p['text']}")
    return '\n'.join(lines)


VALID_PARA_CODES = {'R', 'N', 'D', 'A', 'X', 'T'}

CODE_LABEL = {
    'R': 'REQUIRES_CHANGE', 'N': 'NOTES_CONCERN', 'D': 'DEFERS',
    'A': 'APPROVES', 'X': 'DISMISSES', 'T': 'PROCEDURAL',
}


def parse_phase1(raw: str) -> dict[int, str]:
    results = {}
    for line in raw.split('\n'):
        line = line.strip()
        if not line or line == 'NONE':
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            idx = int(parts[0].replace('{', '').replace('}', ''))
        except ValueError:
            continue
        code = parts[1].upper()
        if code in VALID_PARA_CODES and idx not in results:
            results[idx] = code
    return results


# ---------------------------------------------------------------------------
# Phase 2: Match expert sentences to mietintö paragraphs (LLM)
# ---------------------------------------------------------------------------

PHASE2_SYSTEM = """Mikä mietinnön kappale käsittelee samaa aihetta kuin asiantuntijan lause?

Aihepiirin vastaavuus riittää. Vastaa: [LAUSE] > {KAPPALE} tai [LAUSE] > NONE
Yksi rivi per lause. Ei selityksiä."""

BOILERPLATE = re.compile(
    r'asiantuntijalausunto|valiokunnalle|eduskunnalle|'
    r'allekirjoitus|lisätiedot|y-tunnus|kirje|vastine|'
    r'puh\.\s*\d|www\.|@|\.fi\b|pvm|diaarinumero|'
    r'hallituksen esitys eduskunnalle',
    re.IGNORECASE
)

ADDRESS_BOILERPLATE = re.compile(
    r'^\s*(?:PL\s+\d|Snellman|Valtioneuvosto|puh\.|tfn\.|FO-nummer|'
    r'kirjaamo|registratorskontoret|Finansministeriet|'
    r'Valtiovarainministeriö puh)',
    re.IGNORECASE
)


def extract_expert_sentences(html: str) -> list[dict]:
    if not html:
        return []
    clean = strip_html(html)
    lines = clean.split('.')
    results = []
    idx = 1
    for line in lines:
        line = line.strip()
        if len(line) < 30 or BOILERPLATE.search(line) or ADDRESS_BOILERPLATE.search(line):
            continue
        results.append({'idx': idx, 'text': line})
        idx += 1
    return results


def build_phase2_input(sentences: list[dict], paragraphs: list[dict],
                       para_codes: dict[int, str]) -> str:
    lines = ["MIETINNÖN KAPPALEET:"]
    for p in paragraphs:
        code = para_codes.get(p['idx'], 'T')
        if code == 'T':
            continue
        lines.append(f"{{{p['idx']}}} {p['text'][:300]}")
    lines.append("\nASIANTUNTIJAN LAUSEET:")
    for s in sentences:
        lines.append(f"[{s['idx']}] {s['text']}")
    return '\n'.join(lines)


RE_PHASE2_LINE = re.compile(r'\[(\d+)\]\s*>\s*(?:\{(\d+)\}|(\d+)|NONE)', re.IGNORECASE)


def parse_phase2(raw: str) -> list[tuple[int, int | None]]:
    results = []
    for line in raw.split('\n'):
        line = line.strip()
        if not line or line == 'NONE':
            continue
        m = RE_PHASE2_LINE.search(line)
        if m:
            sent_idx = int(m.group(1))
            para_str = m.group(2) or m.group(3)
            para_idx = int(para_str) if para_str else None
            results.append((sent_idx, para_idx))
    return results


# ---------------------------------------------------------------------------
# Phase 3: Classify (deterministic)
# ---------------------------------------------------------------------------

CODE_TO_CLASS = {
    'R': 'STRONG', 'N': 'ATTENTION', 'A': 'IMPLICIT',
    'D': 'GRAVEYARD', 'X': 'DISMISSED', 'T': 'IMPLICIT',
}

CLASS_PRIORITY = {
    'STRONG': 0, 'ATTENTION': 1, 'IMPLICIT': 2,
    'GRAVEYARD': 3, 'DISMISSED': 4, 'SILENT': 5,
}


def classify_expert(matches: list[tuple], para_codes: dict[int, str], n_sentences: int) -> dict:
    matched_classes = []
    details = []

    for sent_idx, para_idx in matches:
        if para_idx is None:
            details.append(f"{sent_idx}>NONE")
        else:
            code = para_codes.get(para_idx, '?')
            cls = CODE_TO_CLASS.get(code, 'IMPLICIT')
            matched_classes.append(cls)
            details.append(f"{sent_idx}>{para_idx}{code}")

    by_class: dict[str, int] = defaultdict(int)
    for c in matched_classes:
        by_class[c] += 1

    n_matched = len(matched_classes)
    match_rate = n_matched / max(n_sentences, 1)

    # Lowest priority number is the strongest matched response (STRONG=0),
    # not the worst one. The historical key worst_class keeps that value so
    # existing tables still load. Thin matches are reported as SILENT.
    strongest_matched_class = None
    reported_class = 'SILENT'
    if matched_classes:
        strongest_matched_class = sorted(matched_classes, key=lambda x: CLASS_PRIORITY.get(x, 99))[0]
        reported_class = strongest_matched_class
    if n_matched <= 1 and match_rate < 0.15:
        reported_class = 'SILENT'

    return {
        'worst_class': reported_class,
        'strongest_matched_class': strongest_matched_class,
        'n_matched': n_matched,
        'n_returned': len(matches),
        'match_rate': round(match_rate, 3),
        'by_class': dict(by_class),
        'match_detail': ';'.join(details),
    }


# ---------------------------------------------------------------------------
# DB write
# ---------------------------------------------------------------------------

def ensure_he_db_tables(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS mietinto_paragraphs (
            para_idx    INTEGER PRIMARY KEY,
            para_text   TEXT NOT NULL,
            para_code   TEXT,
            is_header   INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS scrutiny_llm (
            expert_id      TEXT NOT NULL,
            sent_idx       INTEGER NOT NULL,
            sent_text      TEXT NOT NULL,
            match_para     INTEGER,
            match_code     TEXT,
            classification TEXT NOT NULL,
            PRIMARY KEY (expert_id, sent_idx)
        );
        CREATE TABLE IF NOT EXISTS scrutiny_llm_summary (
            expert_id      TEXT PRIMARY KEY,
            expert_name    TEXT,
            n_sentences    INTEGER,
            n_matched      INTEGER,
            n_silent       INTEGER,
            worst_class    TEXT,
            match_detail   TEXT
        );
    """)


def clear_he_db(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM mietinto_paragraphs")
    conn.execute("DELETE FROM scrutiny_llm")
    conn.execute("DELETE FROM scrutiny_llm_summary")
    conn.commit()


def write_expert_to_he_db(conn: sqlite3.Connection, expert_res: dict, para_codes: dict) -> None:
    if 'error' in expert_res:
        return

    expert_id = expert_res['expert_id']
    stats = expert_res['stats']

    conn.execute(
        "INSERT OR REPLACE INTO scrutiny_llm_summary VALUES (?,?,?,?,?,?,?)",
        (expert_id, expert_res['expert_name'], len(expert_res['sentences']),
         stats['n_matched'], 1 if stats['worst_class'] == 'SILENT' else 0,
         stats['worst_class'], stats['match_detail'])
    )

    sent_map = {s['idx']: s['text'] for s in expert_res['sentences']}
    rows = []
    for sent_idx, para_idx in expert_res['matches']:
        code = para_codes.get(para_idx) if para_idx else None
        cls = CODE_TO_CLASS.get(code, 'SILENT') if code else 'SILENT'
        rows.append((expert_id, sent_idx, sent_map.get(sent_idx, ''), para_idx, code, cls))

    conn.executemany(
        "INSERT OR REPLACE INTO scrutiny_llm "
        "(expert_id, sent_idx, sent_text, match_para, match_code, classification) "
        "VALUES (?,?,?,?,?,?)",
        rows
    )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

async def process_he(session: aiohttp.ClientSession, he_id: str,
                     write_db: bool = False, verbose: bool = True) -> dict | None:
    if verbose:
        print(f"Processing {he_id}...")

    idx_conn = sqlite3.connect(str(INDEX_DB))
    idx_conn.row_factory = sqlite3.Row

    mietinto = idx_conn.execute(
        "SELECT tunnus, content FROM committee_report WHERE he_id=? AND report_type='Valiokunnan mietintö'",
        (he_id,)
    ).fetchone()

    if not mietinto:
        if verbose:
            print(f"  No mietintö found for {he_id}")
        idx_conn.close()
        return None

    experts = idx_conn.execute(
        "SELECT statement_id, expert_title, content FROM expert_statement WHERE he_id=? ORDER BY statement_id",
        (he_id,)
    ).fetchall()
    idx_conn.close()

    if not experts:
        if verbose:
            print(f"  No expert statements found for {he_id}")
        return None

    paragraphs = segment_mietinto(mietinto['content'])
    if not paragraphs:
        return None

    # Phase 1: Classify paragraphs
    phase1_input = build_phase1_input(paragraphs)
    phase1_budget = 50 + len(paragraphs) * 6
    try:
        raw1 = await call_llm(session, PHASE1_SYSTEM, phase1_input,
                               max_tokens=phase1_budget, ctx=f"{he_id}/phase1")
    except LLMContextExhausted:
        raw1 = await call_llm(session, PHASE1_SYSTEM, phase1_input,
                               max_tokens=phase1_budget * 2, ctx=f"{he_id}/phase1")

    para_codes = parse_phase1(raw1)

    # Phase 2: Process experts concurrently
    sem = asyncio.Semaphore(PARALLEL)

    async def process_expert(expert_row: sqlite3.Row) -> dict:
        async with sem:
            expert_id = expert_row['statement_id']
            expert_name = expert_row['expert_title']
            sentences = extract_expert_sentences(expert_row['content'])[:30]
            if not sentences:
                return {'expert_id': expert_id, 'error': 'Empty'}

            phase2_input = build_phase2_input(sentences, paragraphs, para_codes)
            phase2_budget = 30 + len(sentences) * 8
            try:
                raw2 = await call_llm(session, PHASE2_SYSTEM, phase2_input,
                                       max_tokens=phase2_budget, ctx=f"{he_id}/{expert_id}")
            except LLMContextExhausted:
                raw2 = await call_llm(session, PHASE2_SYSTEM, phase2_input,
                                       max_tokens=phase2_budget * 2, ctx=f"{he_id}/{expert_id}")

            matches = parse_phase2(raw2)
            return {
                'expert_id': expert_id,
                'expert_name': expert_name,
                'sentences': sentences,
                'matches': matches,
                'stats': classify_expert(matches, para_codes, len(sentences)),
            }

    expert_results = list(await asyncio.gather(*[process_expert(e) for e in experts]))

    db_conn = None
    if write_db:
        db_path = HE_DB_DIR / f"{he_id}.db"
        if db_path.exists():
            db_conn = sqlite3.connect(str(db_path))
            ensure_he_db_tables(db_conn)
            clear_he_db(db_conn)
            db_conn.executemany(
                "INSERT INTO mietinto_paragraphs (para_idx, para_text, para_code, is_header) VALUES (?,?,?,?)",
                [(p['idx'], p['text'],
                  CODE_LABEL.get(para_codes.get(p['idx'], ''), para_codes.get(p['idx'])),
                  int(p['is_header'])) for p in paragraphs]
            )
            for res in expert_results:
                write_expert_to_he_db(db_conn, res, para_codes)
            db_conn.commit()
            db_conn.close()

    valid = [r for r in expert_results if 'error' not in r]
    by_class: dict[str, int] = defaultdict(int)
    for r in valid:
        by_class[r['stats']['worst_class']] += 1

    summary = {
        'he_id': he_id,
        'mietinto_tunnus': mietinto['tunnus'],
        'n_paragraphs': len(paragraphs),
        'n_experts_total': len(experts),
        'n_experts_analyzed': len(valid),
        'by_worst_class': dict(by_class),
    }

    if verbose:
        print(f"  Summary: {summary['n_experts_analyzed']} analyzed, {by_class.get('SILENT', 0)} silent")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / f"{he_id}.json", 'w') as f:
        json.dump(summary, f, indent=2)

    return summary


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('he_id', nargs='?')
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--write-db', action='store_true')
    args = parser.parse_args()

    async with aiohttp.ClientSession() as session:
        if args.he_id:
            await process_he(session, args.he_id, write_db=args.write_db)
        elif args.all:
            conn = sqlite3.connect(str(INDEX_DB))
            he_ids = [r[0] for r in conn.execute(
                "SELECT DISTINCT he_id FROM expert_statement"
            ).fetchall()]
            conn.close()
            for hid in he_ids:
                await process_he(session, hid, write_db=args.write_db)


if __name__ == '__main__':
    asyncio.run(main())

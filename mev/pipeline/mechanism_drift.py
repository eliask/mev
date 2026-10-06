"""
Detect mechanism drift: compare HE proposed law text against enacted statutes.

Drift Type A: Did Parliament change the bill?
  HE PROPOSED_SECTION atoms vs enacted statute sections from Finlex statute.zip

Reads:
    .tmp/he_dbs/he-*.db              [per-HE atom databases]
    ~/Downloads/statute.zip           [Finlex AKN XML bulk data]

Outputs:
    .tmp/mechanism_drift/drift_report.json   [structured results]
    .tmp/mechanism_drift/drift_summary.txt   [human-readable summary]

With --embed: writes drift data into per-HE databases (mechanism_drift table)
for display in the HE viewer.

Usage:
    mev detect drift
    mev detect drift he-10-2024
    mev detect drift --threshold 0.95
    mev detect drift --embed
"""

import argparse
import difflib
import json
import re
import sqlite3
import sys
import xml.etree.ElementTree as ET
import zipfile
from collections import defaultdict

from mev.config import ROOT, HE_DB_DIR, STATUTE_ZIP
OUTPUT_DIR = ROOT / '.tmp' / 'mechanism_drift'

NS = {'akn': 'http://docs.oasis-open.org/legaldocml/ns/akn/3.0'}


def normalize_sec_num(s: str) -> str:
    """Normalize section numbers: '6 a' → '6a', '19' → '19'."""
    return re.sub(r'\s+', '', s.strip().lower())


def normalize_text(s: str) -> str:
    """Strip § headers, trailing chapter/voimaantulo blocks, and normalize whitespace."""
    s = re.sub(r'^\d+\s*\w?\s*§\s*', '', s.strip())
    # Strip trailing chapter headers: "2 luku Yleistuen saamisen..."
    s = re.sub(r'\s+\d+\s+luku\s+\S.*$', '', s)
    # Strip trailing voimaantulo + transition provisions block
    # Pattern: "Tämä laki tulee voimaan ..." at end of section text
    s = re.sub(r'\s+Tämä laki tulee voimaan\s.*$', '', s)
    # Join hyphenated line breaks: "työttömyys-turvalain" → "työttömyysturvalain"
    s = re.sub(r'(\w)-\s+(\w)', r'\1\2', s)
    return re.sub(r'\s+', ' ', s).strip()


def deep_normalize(s: str) -> str:
    """Aggressive normalization: strip formatting-only differences."""
    s = normalize_text(s)
    # Normalize all dash types to hyphen
    s = re.sub(r'[–—−‒]', '-', s)
    # Remove spaces around parentheses/brackets
    s = re.sub(r'\(\s+', '(', s)
    s = re.sub(r'\s+\)', ')', s)
    # Normalize comma spacing
    s = re.sub(r'\s*,\s*', ', ', s)
    # Normalize semicolons
    s = re.sub(r'\s*;\s*', '; ', s)
    # Collapse whitespace
    return re.sub(r'\s+', ' ', s).strip()


HE_TO_ENACTED_PATH = OUTPUT_DIR / 'he_to_enacted.json'



def build_he_to_enacted(zf: zipfile.ZipFile) -> dict[str, list[str]]:
    """Scan statute ZIP for government-proposal refs → {he_id: [statute_ids]}.

    Caches result to he_to_enacted.json for subsequent runs.
    """
    if HE_TO_ENACTED_PATH.exists():
        with open(HE_TO_ENACTED_PATH) as f:
            return json.load(f)

    print("Building HE→enacted mapping from statute ZIP metadata (one-time scan)...")
    he_to_statutes: dict[str, set[str]] = defaultdict(set)
    ref_pat = re.compile(r'/akn/fi/doc/government-proposal/(\d{4})/(\d+)')

    for name in zf.namelist():
        if not name.endswith('/main.xml') or '/fin@/' not in name:
            continue
        parts = name.split('/')
        if len(parts) < 7 or parts[3] != 'statute':
            continue
        sid = f"{parts[4]}/{parts[5]}"
        try:
            root = ET.fromstring(zf.read(name))
        except Exception:
            continue
        for ref in root.iter(f'{{{NS["akn"]}}}ref'):
            m = ref_pat.match(ref.get('href', ''))
            if m:
                he_id = f"he-{m.group(2)}-{m.group(1)}"
                he_to_statutes[he_id].add(sid)
                break

    result = {k: sorted(v) for k, v in he_to_statutes.items()}
    HE_TO_ENACTED_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(HE_TO_ENACTED_PATH, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"  Mapped {len(result)} HEs to enacted statutes")
    return result


def get_enacted_title(zf: zipfile.ZipFile, statute_id: str) -> str:
    """Extract title from enacted statute XML."""
    year, num = statute_id.split('/')
    path = f'akn/fi/act/statute/{year}/{num}/fin@/main.xml'
    try:
        root = ET.fromstring(zf.read(path))
    except Exception:
        return ''
    dt = root.find(f'.//{{{NS["akn"]}}}docTitle')
    if dt is not None and dt.text:
        return dt.text.strip()
    lt = root.find(f'.//{{{NS["akn"]}}}longTitle')
    if lt is not None:
        return ''.join(lt.itertext()).strip()
    return ''


def match_proposed_to_enacted(proposed_title: str, enacted_ids: list[str],
                               zf: zipfile.ZipFile,
                               _title_cache: dict = {}) -> str | None:
    """Match a PROPOSED_STATUTE title to an enacted statute within the HE's set."""
    best_id = None
    best_ratio = 0.0
    for sid in enacted_ids:
        if sid not in _title_cache:
            _title_cache[sid] = get_enacted_title(zf, sid)
        enacted_title = _title_cache[sid]
        if enacted_title == proposed_title.strip():
            return sid  # exact match
        ratio = difflib.SequenceMatcher(None, proposed_title.strip(),
                                        enacted_title).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_id = sid
    # Accept fuzzy match if good enough
    if best_ratio > 0.75:
        return best_id
    return None


def get_enacted_sections(zf: zipfile.ZipFile, statute_id: str) -> dict[str, str]:
    """Extract section texts from enacted statute in ZIP."""
    year, num = statute_id.split('/')
    path = f'akn/fi/act/statute/{year}/{num}/fin@/main.xml'
    try:
        content = zf.read(path)
    except KeyError:
        return {}

    root = ET.fromstring(content)
    sections = {}
    for sec in root.iter(f'{{{NS["akn"]}}}section'):
        num_elem = sec.find(f'{{{NS["akn"]}}}num', NS)
        sec_num = ''
        if num_elem is not None and num_elem.text:
            sec_num = num_elem.text.strip().rstrip(' §')
        if not sec_num:
            eid = sec.get('eId', '')
            m = re.search(r'sec_(\d+\w*)', eid)
            sec_num = m.group(1) if m else eid

        # Skip <num> and <heading> — only extract body (<subsection>, <content>, etc.)
        # This matches HE PROPOSED_SECTION atoms which don't include section titles.
        skip_tags = {f'{{{NS["akn"]}}}num', f'{{{NS["akn"]}}}heading'}
        texts = []
        for child in sec:
            if child.tag in skip_tags:
                continue
            for elem in child.iter():
                if elem.text:
                    texts.append(elem.text.strip())
                if elem.tail:
                    texts.append(elem.tail.strip())
        full_text = ' '.join(t for t in texts if t)
        sections[normalize_sec_num(sec_num)] = normalize_text(full_text)

    return sections


def compute_diff(proposed: str, enacted: str) -> dict:
    """Compute structured diff between proposed and enacted text."""
    ratio = difflib.SequenceMatcher(None, proposed, enacted).ratio()

    p_words = proposed.split()
    e_words = enacted.split()
    diff_ops = list(difflib.unified_diff(p_words, e_words, lineterm='', n=0))
    additions = [d[1:] for d in diff_ops if d.startswith('+') and not d.startswith('+++')]
    deletions = [d[1:] for d in diff_ops if d.startswith('-') and not d.startswith('---')]

    return {
        'similarity': round(ratio, 4),
        'words_added': len(additions),
        'words_deleted': len(deletions),
        'additions_sample': ' '.join(additions[:30]),
        'deletions_sample': ' '.join(deletions[:30]),
    }


# Known false-positive patterns (not real drift)
VOIMAANTULO_RE = re.compile(
    r'(päivänä\s+kuuta\s+20\s*\.?)|(tammikuuta|helmikuuta|maaliskuuta|huhtikuuta'
    r'|toukokuuta|kesäkuuta|heinäkuuta|elokuuta|syyskuuta|lokakuuta|marraskuuta'
    r'|joulukuuta)\s+20\d{2}',
    re.IGNORECASE
)


VOIMAANTULO_PROPOSED = re.compile(
    r'Tämä\s+laki\s+tulee\s+voimaan\s+päivänä\s+kuuta\s+20\s*\.?',
    re.IGNORECASE
)


def is_voimaantulo_only(diff: dict, proposed: str) -> bool:
    """Check if the drift is only in the entry-into-force date placeholder."""
    if diff['similarity'] < 0.50:
        return False
    # Proposed has placeholder date "päivänä kuuta 20"
    if VOIMAANTULO_PROPOSED.search(proposed):
        # Check the whole proposed text is just voimaantulo
        stripped = VOIMAANTULO_PROPOSED.sub('', proposed).strip()
        if not stripped:
            return True  # entire section is just the date placeholder
    # Check if deletions are date placeholders (original logic)
    if diff['similarity'] > 0.80:
        dels = diff['deletions_sample'].lower()
        if 'päivänä' in dels and 'kuuta' in dels and '20' in dels:
            return True
    return False


def analyze_he(he_id: str, zf: zipfile.ZipFile, enacted_ids: list[str],
               threshold: float) -> dict:
    """Analyze mechanism drift for one HE."""
    db_path = HE_DB_DIR / f'{he_id}.db'
    if not db_path.exists():
        return {'he_id': he_id, 'error': f'DB not found: {db_path}'}

    he_year = int(he_id.split('-')[2])
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # Get proposed statutes
    proposed_statutes = conn.execute(
        "SELECT atom_id, title FROM atoms WHERE atom_type='PROPOSED_STATUTE' ORDER BY rowid"
    ).fetchall()

    # Track which enacted IDs are still available for matching
    remaining_enacted = list(enacted_ids)

    result = {
        'he_id': he_id,
        'he_year': he_year,
        'statutes': [],
        'summary': {
            'proposed_count': len(proposed_statutes),
            'matched_count': 0,
            'sections_total': 0,
            'sections_same': 0,
            'sections_changed': 0,
            'sections_voimaantulo': 0,
            'sections_missing': 0,
        }
    }

    for ps in proposed_statutes:
        ps_title = ps['title']
        enacted_id = match_proposed_to_enacted(ps_title, remaining_enacted, zf)

        statute_result = {
            'proposed_atom': ps['atom_id'],
            'proposed_title': ps_title,
            'enacted_id': enacted_id,
            'sections': [],
        }

        if not enacted_id:
            result['statutes'].append(statute_result)
            continue

        # Remove matched ID so it won't match again
        if enacted_id in remaining_enacted:
            remaining_enacted.remove(enacted_id)
        result['summary']['matched_count'] += 1

        # Get proposed sections for this statute
        proposed_sections = conn.execute(
            "SELECT atom_id, title, content FROM atoms "
            "WHERE atom_type='PROPOSED_SECTION' AND parent_id=? ORDER BY rowid",
            (ps['atom_id'],)
        ).fetchall()

        enacted_sections = get_enacted_sections(zf, enacted_id)

        for sec in proposed_sections:
            m = re.match(r'(\d+\s*\w*)\s*§', sec['title'] or '')
            sec_num = normalize_sec_num(m.group(1)) if m else '?'
            p_text = normalize_text(sec['content'] or '')
            e_text = enacted_sections.get(sec_num, '')

            result['summary']['sections_total'] += 1

            if not e_text:
                statute_result['sections'].append({
                    'section': sec_num,
                    'title': sec['title'],
                    'status': 'MISSING',
                })
                result['summary']['sections_missing'] += 1
                continue

            diff = compute_diff(p_text, e_text)

            # Formatting-only: deep-normalized texts match
            formatting_only = deep_normalize(p_text) == deep_normalize(e_text)

            if diff['similarity'] > threshold or formatting_only:
                status = 'SAME'
                result['summary']['sections_same'] += 1
            elif is_voimaantulo_only(diff, p_text):
                status = 'VOIMAANTULO'
                result['summary']['sections_voimaantulo'] += 1
            else:
                status = 'CHANGED'
                result['summary']['sections_changed'] += 1

            section_result = {
                'section': sec_num,
                'title': sec['title'],
                'status': status,
                **diff,
            }
            if status == 'CHANGED':
                section_result['proposed_text'] = p_text
                section_result['enacted_text'] = e_text
            statute_result['sections'].append(section_result)

        result['statutes'].append(statute_result)

    conn.close()
    return result


def embed_drift(result: dict):
    """Write drift data into the per-HE database."""
    he_id = result['he_id']
    db_path = HE_DB_DIR / f'{he_id}.db'
    if not db_path.exists():
        return

    conn = sqlite3.connect(str(db_path))
    c = conn.cursor()

    c.execute("DROP TABLE IF EXISTS mechanism_drift")
    c.execute('''CREATE TABLE mechanism_drift (
        proposed_atom TEXT,
        proposed_title TEXT,
        enacted_id TEXT,
        section TEXT,
        section_title TEXT,
        status TEXT,
        similarity REAL,
        words_added INTEGER,
        words_deleted INTEGER,
        additions_sample TEXT,
        deletions_sample TEXT,
        proposed_text TEXT,
        enacted_text TEXT
    )''')

    for st in result.get('statutes', []):
        enacted_id = st.get('enacted_id', '')
        for sec in st.get('sections', []):
            c.execute(
                'INSERT INTO mechanism_drift VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)',
                (
                    st['proposed_atom'],
                    st['proposed_title'],
                    enacted_id,
                    sec['section'],
                    sec.get('title', ''),
                    sec['status'],
                    sec.get('similarity', 0.0),
                    sec.get('words_added', 0),
                    sec.get('words_deleted', 0),
                    sec.get('additions_sample', ''),
                    sec.get('deletions_sample', ''),
                    sec.get('proposed_text', ''),
                    sec.get('enacted_text', ''),
                )
            )
        # If no sections but statute was matched, add a statute-level row
        if not st.get('sections') and enacted_id:
            c.execute(
                'INSERT INTO mechanism_drift VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)',
                (st['proposed_atom'], st['proposed_title'], enacted_id,
                 '', '', 'MATCHED', 1.0, 0, 0, '', '', '', '')
            )

    # Also store summary
    c.execute("DROP TABLE IF EXISTS drift_summary")
    c.execute('''CREATE TABLE drift_summary (key TEXT PRIMARY KEY, value TEXT)''')
    s = result.get('summary', {})
    for k, v in s.items():
        c.execute('INSERT INTO drift_summary VALUES (?,?)', (k, str(v)))

    conn.commit()
    conn.close()


def main():
    parser = argparse.ArgumentParser(description='Detect mechanism drift (Type A)')
    parser.add_argument('he_ids', nargs='*', help='HE IDs to process (default: all)')
    parser.add_argument('--threshold', type=float, default=0.98,
                        help='Similarity threshold for SAME (default: 0.98)')
    parser.add_argument('--embed', action='store_true',
                        help='Write drift data into per-HE databases')
    args = parser.parse_args()

    if not STATUTE_ZIP.exists():
        print(f"Error: statute.zip not found at {STATUTE_ZIP}")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    zf = zipfile.ZipFile(STATUTE_ZIP, 'r')
    he_to_enacted = build_he_to_enacted(zf)
    print(f"HE→enacted mapping: {len(he_to_enacted)} HEs")

    if args.he_ids:
        he_ids = args.he_ids
    else:
        he_ids = sorted(
            p.stem for p in HE_DB_DIR.glob('he-*.db')
        )

    all_results = []
    totals = {'proposed': 0, 'matched': 0, 'sections': 0,
              'same': 0, 'changed': 0, 'voimaantulo': 0, 'missing': 0}

    for he_id in he_ids:
        enacted_ids = he_to_enacted.get(he_id, [])
        r = analyze_he(he_id, zf, enacted_ids, args.threshold)
        all_results.append(r)

        if 'error' in r:
            print(f"  {he_id}: {r['error']}")
            continue

        s = r['summary']
        totals['proposed'] += s['proposed_count']
        totals['matched'] += s['matched_count']
        totals['sections'] += s['sections_total']
        totals['same'] += s['sections_same']
        totals['changed'] += s['sections_changed']
        totals['voimaantulo'] += s['sections_voimaantulo']
        totals['missing'] += s['sections_missing']

        changed = s['sections_changed']
        flag = ' ***' if changed > 0 else ''
        print(f"  {he_id}: {s['matched_count']}/{s['proposed_count']} matched, "
              f"{s['sections_same']} same, {changed} changed, "
              f"{s['sections_voimaantulo']} voimaantulo, "
              f"{s['sections_missing']} missing{flag}")

    zf.close()

    # Embed into per-HE databases if requested
    if args.embed:
        embedded = 0
        for r in all_results:
            if 'error' not in r:
                embed_drift(r)
                embedded += 1
        print(f"\nEmbedded drift data into {embedded} HE databases")

    # Save JSON results
    with open(OUTPUT_DIR / 'drift_report.json', 'w') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    # Save summary
    with open(OUTPUT_DIR / 'drift_summary.txt', 'w') as f:
        f.write("Mechanism Drift Type A: HE Proposed vs Enacted\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"HEs analyzed:        {len(he_ids)}\n")
        f.write(f"Proposed statutes:   {totals['proposed']}\n")
        f.write(f"Matched to enacted:  {totals['matched']} "
                f"({totals['matched']/max(totals['proposed'],1)*100:.0f}%)\n")
        f.write(f"Sections compared:   {totals['sections']}\n")
        f.write(f"  Same:              {totals['same']}\n")
        f.write(f"  Changed:           {totals['changed']}\n")
        f.write(f"  Voimaantulo only:  {totals['voimaantulo']}\n")
        f.write(f"  Missing:           {totals['missing']}\n\n")

        # List all changed sections
        f.write("CHANGED SECTIONS\n")
        f.write("-" * 50 + "\n")
        for r in all_results:
            if 'error' in r:
                continue
            for st in r['statutes']:
                for sec in st.get('sections', []):
                    if sec['status'] == 'CHANGED':
                        f.write(f"\n{r['he_id']} | {st['enacted_id']} §{sec['section']}\n")
                        f.write(f"  {sec['title']} ({sec['similarity']:.1%} similar)\n")
                        if sec['deletions_sample']:
                            f.write(f"  - {sec['deletions_sample'][:120]}\n")
                        if sec['additions_sample']:
                            f.write(f"  + {sec['additions_sample'][:120]}\n")

    print(f"\nTotals: {totals['sections']} sections compared across {len(he_ids)} HEs")
    print(f"  {totals['same']} same, {totals['changed']} changed, "
          f"{totals['voimaantulo']} voimaantulo-only, {totals['missing']} missing")
    print(f"\nResults: {OUTPUT_DIR / 'drift_report.json'}")
    print(f"Summary: {OUTPUT_DIR / 'drift_summary.txt'}")


if __name__ == '__main__':
    main()

"""
Build per-HE SQLite databases from AKN XML. First step in HE pipeline.

Source: AKN XML from government-proposal.zip (only source).
Output: .tmp/he_dbs/he-*.db — per-HE DBs with atoms, metadata, law_refs, diffs, claims (schema only).

Pipeline: build_he_db.py → extract_claims_llm.py → detect_unreason.py → enrich_he_db.py → upload
Per-HE DBs are ephemeral build artifacts (claims/audits/unreason live in he_enrichments.db,
copied to per-HE DBs by enrich_he_db.py). Viewer: mr_website_public/mev/he-viewer.js

Usage:
    mev build atoms he-38-2025
    mev build atoms --year 2025
    mev build atoms              # all HEs in zip
"""

import argparse
import json
import re
import sqlite3
import csv
import xml.etree.ElementTree as ET
import zipfile
import libvoikko
from pathlib import Path

from mev.config import ROOT as _ROOT, AKN_ZIP_PATH

# Law references: (617/2021) or lain 617/2021
RE_LAW_REF = re.compile(r'\(?(\d{1,5}/\d{4})\)?')

# --- Section type classification ---

SECTION_TYPE_MAP = {
    '1': 'BACKGROUND',
    '2': 'CURRENT_STATE',
    '3': 'OBJECTIVES',
    '4': 'IMPACT',
    '5': 'ALTERNATIVES',
    '6': 'FEEDBACK',
    '7': 'SECTION_JUSTIFICATION',
    '8': 'SECTION_JUSTIFICATION',
    '9': 'NARRATIVE',
    '10': 'NARRATIVE',
    '11': 'NARRATIVE',
    '12': 'CONSTITUTIONAL',
    '13': 'CONSTITUTIONAL',
}

class StatuteResolver:
    """Resolves statute IDs from phrases using the alias registry."""
    def __init__(self, registry_path: Path):
        self.registry = []
        if registry_path.exists():
            with open(registry_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                self.registry = list(reader)
        self.voikko = libvoikko.Voikko('fi')
        
    def lemmatize(self, text: str) -> str:
        """Lemmatize a phrase using Voikko. Handles compound words."""
        words = text.split()
        lemmas = []
        for word in words:
            analysis = self.voikko.analyze(word)
            if analysis and 'BASEFORM' in analysis[0]:
                lemmas.append(analysis[0]['BASEFORM'])
            else:
                lemmas.append(word)
        return ' '.join(lemmas).lower()

    def resolve(self, phrase: str, he_id: str = None, date: str = None) -> str | None:
        """
        Resolve a phrase to a statute ID.
        Priority:
        1. Contextual override (exact phrase or lemma match)
        2. Temporal window (exact phrase or lemma match)
        3. Global default
        """
        if not phrase: return None
        phrase_clean = phrase.lower().strip()
        lemma = self.lemmatize(phrase_clean)
        
        # 1. Contextual override
        if he_id:
            for entry in self.registry:
                if entry['context_he_id'] == he_id:
                    reg_phrase = entry['phrase'].lower().strip()
                    if reg_phrase == phrase_clean or reg_phrase == lemma:
                        return entry['referent_id']
        
        # 2. Temporal window
        if date:
            for entry in self.registry:
                reg_phrase = entry['phrase'].lower().strip()
                if reg_phrase == phrase_clean or reg_phrase == lemma:
                    start = entry['valid_from'] or '0000-00-00'
                    end = entry['valid_until'] or '9999-99-99'
                    if start <= date <= end:
                        return entry['referent_id']
        
        # 3. Global default
        for entry in self.registry:
            if not entry['context_he_id'] and not entry['valid_from'] and not entry['valid_until']:
                reg_phrase = entry['phrase'].lower().strip()
                if reg_phrase == phrase_clean or reg_phrase == lemma:
                    return entry['referent_id']
                    
        return None

# Global resolver instance, initialized in main
statute_resolver = None

def classify_section(section_num: str, title: str) -> str:
    """Classify a section by its number and title."""
    top = section_num.split('.')[0]
    title_lower = title.lower() if title else ''
    if 'pääasiallinen sisältö' in title_lower:
        return 'SUMMARY'
    if 'perustuslaki' in title_lower or 'suhde perustuslakiin' in title_lower:
        return 'CONSTITUTIONAL'
    if 'lausuntopalaute' in title_lower or 'lausuntomenettely' in title_lower:
        return 'FEEDBACK'
    if 'vaikutu' in title_lower:
        return 'IMPACT'
    if 'tavoite' in title_lower:
        return 'OBJECTIVES'
    if 'nykytila' in title_lower:
        return 'CURRENT_STATE'
    if 'toteuttamisvaihtoehto' in title_lower or 'vaihtoehtoiset' in title_lower:
        return 'ALTERNATIVES'
    if 'säännöskohtai' in title_lower:
        return 'SECTION_JUSTIFICATION'
    return SECTION_TYPE_MAP.get(top, 'NARRATIVE')


def make_atom_id(section_num: str, atom_type: str) -> str:
    safe = section_num.replace('.', '_')
    return f"sec_{safe}"


def extract_law_refs(text: str) -> list[tuple[str, str]]:
    """Extract all law references from text. Returns list of (statute_id, ref_type)."""
    refs = []
    for m in RE_LAW_REF.finditer(text):
        raw = m.group(1)
        parts = raw.split('/')
        if len(parts) == 2:
            num, year = parts
            statute_id = f"{year}/{num}"
            refs.append((statute_id, 'CITES'))
    return refs


def compute_parent_id(section_num: str) -> str | None:
    parts = section_num.split('.')
    if len(parts) <= 1:
        return None
    parent_num = '.'.join(parts[:-1])
    return f"sec_{parent_num.replace('.', '_')}"




# ===================================================================
# DATABASE BUILDER
# ===================================================================

DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS atoms (
    atom_id     TEXT PRIMARY KEY,
    atom_type   TEXT NOT NULL,
    parent_id   TEXT,
    seq         INTEGER NOT NULL,
    title       TEXT,
    content     TEXT NOT NULL,
    content_html TEXT,
    char_count  INTEGER,
    FOREIGN KEY (parent_id) REFERENCES atoms(atom_id)
);

CREATE TABLE IF NOT EXISTS metadata (
    he_id               TEXT PRIMARY KEY,
    number              INTEGER,
    year                INTEGER,
    title               TEXT,
    ministry            TEXT,
    finlex_uri          TEXT,
    eduskunta_tunnus    TEXT,
    hankeikkuna_uuid    TEXT,
    lausuntopalvelu_guid TEXT,
    status              TEXT,
    date_issued         TEXT,
    laws_amended        TEXT,
    page_count          INTEGER,
    atom_count          INTEGER,
    build_timestamp     TEXT,
    build_model         TEXT,
    source              TEXT
);

CREATE TABLE IF NOT EXISTS claims (
    claim_id    TEXT PRIMARY KEY,
    claim_type  TEXT NOT NULL,
    text        TEXT NOT NULL,
    quote       TEXT,
    source_atom TEXT NOT NULL,
    amount_eur  REAL,
    scale       TEXT,
    time_horizon TEXT,
    verifiability TEXT,
    confidence  REAL,
    FOREIGN KEY (source_atom) REFERENCES atoms(atom_id)
);

CREATE TABLE IF NOT EXISTS law_refs (
    ref_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    source_atom TEXT NOT NULL,
    target_statute TEXT NOT NULL,
    target_section TEXT,
    ref_type    TEXT NOT NULL,
    FOREIGN KEY (source_atom) REFERENCES atoms(atom_id)
);

CREATE TABLE IF NOT EXISTS diffs (
    diff_id     TEXT PRIMARY KEY,
    statute_id  TEXT NOT NULL,
    section     TEXT NOT NULL,
    old_text    TEXT,
    new_text    TEXT NOT NULL,
    diff_html   TEXT,
    justification_atom TEXT,
    FOREIGN KEY (justification_atom) REFERENCES atoms(atom_id)
);

CREATE TABLE IF NOT EXISTS lausunnot (
    lausunto_id TEXT PRIMARY KEY,
    organization TEXT NOT NULL,
    stance      TEXT,
    key_points  TEXT,
    source_url  TEXT,
    related_atoms TEXT
);

CREATE INDEX IF NOT EXISTS idx_atoms_type ON atoms(atom_type);
CREATE INDEX IF NOT EXISTS idx_atoms_parent ON atoms(parent_id);
CREATE INDEX IF NOT EXISTS idx_claims_type ON claims(claim_type);
CREATE INDEX IF NOT EXISTS idx_claims_source ON claims(source_atom);
CREATE INDEX IF NOT EXISTS idx_refs_statute ON law_refs(target_statute);
CREATE INDEX IF NOT EXISTS idx_refs_source ON law_refs(source_atom);
"""

# ===================================================================
# AKN XML PARSER (primary — from government-proposal.zip)
# ===================================================================

AKN_NS = 'http://docs.oasis-open.org/legaldocml/ns/akn/3.0'

def _akn_tag(local: str) -> str:
    return f'{{{AKN_NS}}}{local}'

def _akn_text(elem) -> str:
    """Extract all text content from an element and its children, stripping tags."""
    parts = []
    if elem.text:
        parts.append(elem.text)
    for child in elem:
        # Skip ref elements but include their text
        parts.append(_akn_text(child))
        if child.tail:
            parts.append(child.tail)
    return ''.join(parts)

def _akn_text_with_refs(elem) -> str:
    """Extract text, preserving ref hrefs as (NNN/YYYY) inline."""
    parts = []
    if elem.text:
        parts.append(elem.text)
    for child in elem:
        tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag
        if tag == 'authorialNote':
            # Render footnote marker only, skip body
            marker = child.get('marker', '')
            if marker:
                parts.append(f'[{marker}]')
        elif tag == 'ref':
            parts.append(_akn_text(child))
        elif tag == 'a':
            parts.append(_akn_text(child))
        else:
            parts.append(_akn_text_with_refs(child))
        if child.tail:
            parts.append(child.tail)
    # Collapse runs of whitespace from XML indentation
    return ' '.join(''.join(parts).split())

def _extract_akn_refs(elem) -> list[tuple[str, str]]:
    """Extract law references from <ref> elements in AKN XML."""
    refs = []
    for ref in elem.iter(_akn_tag('ref')):
        href = ref.get('href', '')
        # Pattern: /akn/fi/act/statute-consolidated/YYYY/NNN
        # or /akn/fi/doc/government-proposal/YYYY/NNN
        m = re.search(r'/(\d{4})/(\d+)(?:/|$)', href)
        if m:
            year, num = m.group(1), m.group(2)
            if 'statute' in href or 'act' in href:
                refs.append((f'{year}/{num}', 'CITES'))
            elif 'government-proposal' in href:
                refs.append((f'HE {num}/{year}', 'CITES_HE'))
    # Also extract (NNN/YYYY) patterns from text
    text = _akn_text(elem)
    for statute_id, ref_type in extract_law_refs(text):
        if (statute_id, ref_type) not in refs:
            refs.append((statute_id, ref_type))
    return refs

def _collect_content_parts(container) -> list[str]:
    """Collect text from direct children of container, rendering tables as markdown.

    Walks direct children in document order. <p> → text, <blockList> → list items,
    <table> → markdown pipe table. Skips <num>, <heading>, nested <tblock>/<hcontainer>.
    """
    SKIP_TAGS = {'num', 'heading', 'tblock', 'hcontainer', 'section'}
    parts = []
    for child in container:
        tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag
        if tag in SKIP_TAGS:
            continue
        if tag == 'p':
            t = _akn_text_with_refs(child).strip()
            if t:
                parts.append(t)
        elif tag == 'blockList':
            for item in child.iter(_akn_tag('p')):
                t = _akn_text_with_refs(item).strip()
                if t:
                    parts.append(t)
        elif tag == 'table':
            md = _akn_table_to_markdown(child)
            if md:
                parts.append(md)
        elif tag in ('content', 'subsection', 'intro', 'paragraph', 'wrapUp'):
            parts.extend(_collect_content_parts(child))
    return parts


def _norm_ws(text: str) -> str:
    """Collapse XML indentation whitespace to single spaces, preserving boundary spaces."""
    if not text:
        return ''
    prefix = ' ' if text[0] in ' \t\n\r' else ''
    suffix = ' ' if text[-1] in ' \t\n\r' else ''
    return prefix + ' '.join(text.split()) + suffix

def _akn_elem_to_html(elem) -> str:
    """Convert an AKN element to simple HTML, preserving refs as <a> tags."""
    parts = []
    if elem.text:
        parts.append(_esc_html(_norm_ws(elem.text)))
    for child in elem:
        tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag
        if tag == 'authorialNote':
            marker = child.get('marker', '')
            if marker:
                parts.append(f'<sup>[{_esc_html(marker)}]</sup>')
        elif tag == 'ref':
            href = child.get('href', '')
            ref_text = _esc_html(_norm_ws(_akn_text(child)))
            m = re.search(r'/act/statute-consolidated/(\d{4})/(\d+)', href)
            if m:
                sid = f'{m.group(1)}/{m.group(2)}'
                parts.append(f'<a href="/mev/lakikartta#detail/{sid}">{ref_text}</a>')
            else:
                parts.append(ref_text)
        elif tag == 'a':
            href = child.get('href', '')
            link_text = _esc_html(_norm_ws(_akn_text(child)))
            if href:
                parts.append(f'<a href="{_esc_html(href)}" target="_blank">{link_text}</a>')
            else:
                parts.append(link_text)
        elif tag == 'b':
            parts.append(f'<strong>{_esc_html(_norm_ws(_akn_text(child)))}</strong>')
        elif tag == 'i':
            parts.append(f'<em>{_esc_html(_norm_ws(_akn_text(child)))}</em>')
        else:
            parts.append(_akn_elem_to_html(child))
        if child.tail:
            parts.append(_esc_html(_norm_ws(child.tail)))
    return ''.join(parts)


def _esc_html(text: str) -> str:
    """Minimal HTML escaping."""
    return text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def _akn_table_to_html(table_elem) -> str:
    """Convert AKN <table> to HTML <table>."""
    rows = []
    for tr in table_elem.findall(_akn_tag('tr')):
        cells = []
        for td in tr:
            tag = td.tag.split('}')[-1] if '}' in td.tag else td.tag
            if tag in ('td', 'th'):
                cell_html = ' '.join(
                    _akn_elem_to_html(p).strip()
                    for p in td.findall(_akn_tag('p'))
                ).strip()
                cell_tag = 'th' if tag == 'th' else 'td'
                cells.append(f'<{cell_tag}>{cell_html}</{cell_tag}>')
        if cells:
            rows.append(f'<tr>{"".join(cells)}</tr>')
    if not rows:
        return ''
    return f'<table>{"".join(rows)}</table>'


def _collect_html_parts(container) -> list[str]:
    """Collect HTML from direct children of container (parallel to _collect_content_parts)."""
    SKIP_TAGS = {'num', 'heading', 'tblock', 'hcontainer', 'section'}
    parts = []
    for child in container:
        tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag
        if tag in SKIP_TAGS:
            continue
        if tag == 'p':
            html = _akn_elem_to_html(child).strip()
            if html:
                parts.append(f'<p>{html}</p>')
        elif tag == 'blockList':
            items = []
            for item in child.iter(_akn_tag('p')):
                t = _akn_elem_to_html(item).strip()
                if t:
                    items.append(f'<li>{t}</li>')
            if items:
                parts.append(f'<ul>{"".join(items)}</ul>')
        elif tag == 'table':
            html = _akn_table_to_html(child)
            if html:
                parts.append(html)
        elif tag in ('content', 'subsection', 'intro', 'paragraph', 'wrapUp'):
            parts.extend(_collect_html_parts(child))
    return parts


def _akn_table_to_markdown(table_elem) -> str:
    """Convert AKN <table> element to markdown pipe table."""
    rows = []
    for tr in table_elem.findall(_akn_tag('tr')):
        cells = []
        for td in tr:
            tag = td.tag.split('}')[-1] if '}' in td.tag else td.tag
            if tag in ('td', 'th'):
                cell_text = ' '.join(
                    _akn_text_with_refs(p).strip()
                    for p in td.findall(_akn_tag('p'))
                ).strip()
                cells.append(cell_text)
        if cells:
            rows.append(cells)
    if not rows:
        return ''
    # Normalize column count
    ncols = max(len(r) for r in rows)
    for r in rows:
        while len(r) < ncols:
            r.append('')
    lines = []
    lines.append('| ' + ' | '.join(rows[0]) + ' |')
    lines.append('|' + '|'.join(['---'] * ncols) + '|')
    for r in rows[1:]:
        lines.append('| ' + ' | '.join(r) + ' |')
    return '\n'.join(lines)


def _parse_tblock(tblock, depth: int = 0) -> list[dict]:
    """Recursively parse a tblock into section atoms."""
    sections = []
    num_elem = tblock.find(_akn_tag('num'))
    heading_elem = tblock.find(_akn_tag('heading'))

    section_num = num_elem.text.strip() if num_elem is not None and num_elem.text else ''
    title = _akn_text(heading_elem).strip() if heading_elem is not None else ''

    if not section_num:
        # Wrapper tblock (e.g. "YLEISPERUSTELUT") — no num, just recurse into children
        for child_tblock in tblock.findall(_akn_tag('tblock')):
            sections.extend(_parse_tblock(child_tblock, depth + 1))
        return sections

    # Collect direct content (p, blockList, table — not sub-tblocks)
    content_parts = _collect_content_parts(tblock)
    content = '\n\n'.join(p for p in content_parts if p)
    html_parts = _collect_html_parts(tblock)
    content_html = '\n'.join(html_parts)

    atom_type = classify_section(section_num, title)
    atom_id = make_atom_id(section_num, atom_type)

    sections.append({
        'section_num': section_num,
        'title': title,
        'atom_type': atom_type,
        'atom_id': atom_id,
        'content': content,
        'content_html': content_html,
        '_elem': tblock,  # keep for ref extraction
    })

    # Recurse into nested tblocks
    for child_tblock in tblock.findall(_akn_tag('tblock')):
        sections.extend(_parse_tblock(child_tblock, depth + 1))

    return sections


def _parse_enacting_ops(enacting) -> list[dict]:
    """Parse enactingClause into operation blobs (kumotaan/muutetaan/lisätään).

    Each <p> in the enactingClause that starts with <i>op</i> becomes one entry:
      {'op': 'kumotaan'|'muutetaan'|'lisätään', 'text': '...section refs...'}
    """
    if enacting is None:
        return []
    ops = []
    content = enacting.find(_akn_tag('content'))
    if content is None:
        content = enacting
    for p in content.findall(_akn_tag('p')):
        # Check if first child is <i> with operation verb
        i_elem = p.find(_akn_tag('i'))
        if i_elem is None:
            # Also check without namespace (some XMLs)
            i_elem = p.find('i')
        if i_elem is not None:
            op_text = (i_elem.text or '').strip().lower()
            if op_text in ('kumotaan', 'muutetaan', 'lisätään'):
                full_text = _akn_text(p).strip()
                ops.append({'op': op_text, 'text': full_text})
    return ops


def _classify_section_op(pyk_num: str, ops: list[dict]) -> str:
    """Classify a proposed section as DELETE/REPLACE/INSERT by substring matching.

    pyk_num is like '19 a', '2', '18'. We search each operation blob for this
    section number. Match requires the number to appear near '§' context.

    Key subtlety: "kumotaan ... 2 §:n 3 momentti" means subsection 2(3) is deleted,
    NOT that §2 itself is deleted. If the match is followed by ":n ... momentti",
    the kumotaan applies to a subsection, not the whole section. The section itself
    is likely muutetaan (since it will appear with proposed text).
    """
    pyk_clean = pyk_num.strip()
    escaped = re.escape(pyk_clean)

    # Check lisätään first (most specific — "uusi" + section number)
    # Then muutetaan, then kumotaan (kumotaan with proposed text = rare edge case)
    for op_entry in ops:
        op = op_entry['op']
        text = op_entry['text']

        # Match: section number near § context
        pattern = rf'(?<!\d){escaped}\s*(?:§|,|\sja\s|\ssekä\s|\sseuraavasti)'
        m = re.search(pattern, text)
        if not m:
            continue

        if op == 'kumotaan':
            # Check if this is a subsection-level deletion (N §:n M momentti)
            # If so, the section itself is NOT being deleted — skip this match
            after_match = text[m.start():]
            sub_pattern = rf'^{escaped}\s*§:n\s+\d+\s+momentti'
            if re.search(sub_pattern, after_match):
                continue  # subsection deletion, not whole-section deletion
            return 'DELETE'
        elif op == 'muutetaan':
            return 'REPLACE'
        elif op == 'lisätään':
            return 'INSERT'

    # Fallback: if section has letter suffix (19a, 19b) -> likely INSERT
    if re.search(r'[a-z]$', pyk_clean.replace(' ', '')):
        return 'INSERT'
    return 'REPLACE'  # default assumption for sections with proposed text


def _extract_kumotaan_sections(ops: list[dict]) -> list[dict]:
    """Extract section references from kumotaan clauses (deleted sections have no proposed text).

    Returns list of {'section_num': '2', 'momentti': '3', 'text': 'original clause text'}
    for sections/subsections being deleted.
    """
    deleted = []
    for op_entry in ops:
        if op_entry['op'] != 'kumotaan':
            continue
        text = op_entry['text']
        # Pattern: "N §:n M momentti" (subsection deletion)
        for m in re.finditer(r'(\d+(?:\s+[a-z])?)\s*§:n\s+(\d+)\s+momentti', text):
            deleted.append({
                'section_num': m.group(1).strip(),
                'momentti': m.group(2),
                'full_ref': m.group(0),
            })
        # Pattern: "N §" (full section deletion) — but NOT if followed by ":n"
        for m in re.finditer(r'(\d+(?:\s+[a-z])?)\s*§(?!:)', text):
            sec = m.group(1).strip()
            # Check it's not already captured as momentti ref
            if not any(d['section_num'] == sec and 'momentti' not in d for d in deleted):
                deleted.append({
                    'section_num': sec,
                    'momentti': None,
                    'full_ref': m.group(0),
                })
    return deleted


def _parse_akn_bills(bills_container) -> list[dict]:
    """Parse lakiehdotukset from AKN XML bills container."""
    statutes = []
    bill_seq = 0

    for bill in bills_container.findall(_akn_tag('hcontainer')):
        if bill.get('name') != 'bill':
            continue

        bill_seq += 1
        heading_elem = bill.find(_akn_tag('heading'))
        title = _akn_text(heading_elem).strip() if heading_elem is not None else ''

        # Extract statute ID from enacting clause refs
        statute_id = None
        enacting = bill.find(f".//{_akn_tag('hcontainer')}[@name='enactingClause']")
        if enacting is not None:
            for ref in enacting.iter(_akn_tag('ref')):
                href = ref.get('href', '')
                m = re.search(r'/act/statute-consolidated/(\d{4})/(\d+)', href)
                if m:
                    statute_id = f'{m.group(1)}/{m.group(2)}'
                    break

        # Parse operation types from enacting clause
        enacting_ops = _parse_enacting_ops(enacting)

        # Extract header text from enacting clause
        header_text = ''
        if enacting is not None:
            header_text = _akn_text(enacting).strip()

        # Find pykälät in statuteProvisionsWrapper
        pykalat = []
        provisions = bill.find(f".//{_akn_tag('hcontainer')}[@name='statuteProvisionsWrapper']")
        if provisions is not None:
            # Sections can be inside chapters or directly
            for section in provisions.iter(_akn_tag('section')):
                pyk_num_elem = section.find(_akn_tag('num'))
                pyk_heading_elem = section.find(_akn_tag('heading'))

                if pyk_num_elem is None:
                    continue

                pyk_num_text = _akn_text(pyk_num_elem).strip()
                # Extract just the number: "10 §" -> "10", "3 a §" -> "3 a"
                pyk_num = pyk_num_text.replace('§', '').strip()
                pyk_title = _akn_text(pyk_heading_elem).strip() if pyk_heading_elem is not None else ''

                # Extract chapter number from eId (e.g. "bill_16__chp_3__sec_33" -> "3")
                section_eid = section.get('eId', '')
                chp_match = re.search(r'__chp_(\d+)__', section_eid)
                chapter_num = chp_match.group(1) if chp_match else None

                # Collect content from subsections (tables as markdown for text, HTML for display)
                content_parts = _collect_content_parts(section)
                html_parts = _collect_html_parts(section)

                # Skip omission-only sections
                is_omission_only = (
                    len(list(section)) == 1 and
                    section[0].tag == _akn_tag('hcontainer') and
                    section[0].get('name') == 'omission'
                )
                if is_omission_only and not content_parts:
                    continue

                content = '\n\n'.join(content_parts)
                # Classify operation type via enacting clause matching
                op_type = _classify_section_op(pyk_num, enacting_ops)

                pykalat.append({
                    'pykala_num': pyk_num,
                    'title': pyk_title,
                    'atom_id': f"law_{bill_seq}_sec_{pyk_num.replace(' ', '')}",
                    'content': content,
                    'content_html': '\n'.join(html_parts),
                    'op_type': op_type,
                    'chapter_num': chapter_num,
                })

        # Extract kumotaan-only sections (no proposed text in XML)
        kumotaan_sections = _extract_kumotaan_sections(enacting_ops)

        statutes.append({
            'seq': bill_seq,
            'title': title,
            'statute_id': statute_id,
            'atom_id': f"proposed_statute_{bill_seq}",
            'header_text': header_text,
            'pykalat': pykalat,
            'enacting_ops': enacting_ops,
            'kumotaan_sections': kumotaan_sections,
        })

    return statutes


def parse_akn_xml(xml_str: str, he_id: str = None) -> dict:
    """Parse AKN XML into structured atoms."""
    root = ET.fromstring(xml_str)
    doc = root.find(_akn_tag('doc'))
    if doc is None:
        raise ValueError("No <doc> element found in AKN XML")

    main_body = doc.find(_akn_tag('mainBody'))
    if main_body is None:
        raise ValueError("No <mainBody> element found")

    # Extract summary from introduction
    summary = ''
    summary_html = ''
    for hc in main_body.findall(_akn_tag('hcontainer')):
        if hc.get('name') == 'introduction':
            content_elem = hc.find(_akn_tag('content'))
            if content_elem is not None:
                parts = _collect_content_parts(content_elem)
                summary = '\n\n'.join(p for p in parts if p)
                html_parts = _collect_html_parts(content_elem)
                summary_html = '\n'.join(html_parts)
            break

    # Extract sections from rationale tblocks
    sections = []
    for hc in main_body.findall(_akn_tag('hcontainer')):
        if hc.get('name') == 'rationale':
            content_elem = hc.find(_akn_tag('content'))
            if content_elem is not None:
                for tblock in content_elem.findall(_akn_tag('tblock')):
                    sections.extend(_parse_tblock(tblock))
            break

    # Extract proposed statutes from bills
    proposed_statutes = []
    for hc in main_body.findall(_akn_tag('hcontainer')):
        if hc.get('name') == 'bills' or hc.get('eId') == 'bills':
            proposed_statutes = _parse_akn_bills(hc)
            break

    return {
        'summary': summary,
        'summary_html': summary_html,
        'sections': sections,
        'proposed_statutes': proposed_statutes,
    }


def extract_akn_metadata(xml_str: str) -> dict:
    """Extract metadata from AKN XML meta block."""
    root = ET.fromstring(xml_str)
    doc = root.find(_akn_tag('doc'))
    meta = doc.find(_akn_tag('meta')) if doc is not None else None
    if meta is None:
        return {}

    result = {}

    # FRBRWork
    work = meta.find(f'.//{_akn_tag("FRBRWork")}')
    if work is not None:
        num = work.find(_akn_tag('FRBRnumber'))
        if num is not None:
            result['number'] = int(num.get('value', '0'))
        date = work.find(_akn_tag('FRBRdate'))
        if date is not None:
            result['date_issued'] = date.get('date')
            result['year'] = int(date.get('date', '0000')[:4])
        uri = work.find(_akn_tag('FRBRuri'))
        if uri is not None:
            result['finlex_uri'] = uri.get('value')

    # Title from preface
    preface = doc.find(_akn_tag('preface'))
    if preface is not None:
        doc_title = preface.find(f'.//{_akn_tag("docTitle")}')
        if doc_title is not None:
            result['title'] = _akn_text(doc_title).strip()

    # Ministry from references
    for org in meta.iter(_akn_tag('TLCOrganization')):
        eid = org.get('eId', '')
        if 'ministry' in eid:
            result['ministry'] = org.get('showAs', '')
            break

    # Laws amended from finlex:affects/statuteReference
    FINLEX_NS = 'http://data.finlex.fi/schema/finlex'
    affects = meta.find(f'.//{{{FINLEX_NS}}}affects')
    if affects is not None:
        laws = []
        for sr in affects.findall(f'{{{FINLEX_NS}}}statuteReference'):
            ref = sr.find(f'{{{FINLEX_NS}}}ref')
            if ref is not None and ref.text:
                # ref.text is "380/2023" format, convert to year/number for statute graph
                parts = ref.text.strip().split('/')
                if len(parts) == 2:
                    laws.append(ref.text.strip())  # keep as "380/2023"
        if laws:
            result['laws_amended'] = json.dumps(laws)

    return result


def akn_zip_path_for_he(year: int, number: int) -> str:
    """Return the path inside the zip for a given HE."""
    return f'akn/fi/doc/government-proposal/{year}/{number}/fin@/main.xml'

def list_akn_hes(zip_path: Path = AKN_ZIP_PATH) -> list[tuple[int, int]]:
    """List all (year, number) pairs available in the AKN zip."""
    hes = []
    with zipfile.ZipFile(zip_path, 'r') as zf:
        for name in zf.namelist():
            m = re.match(r'akn/fi/doc/government-proposal/(\d{4})/(\d+)/fin@/main\.xml$', name)
            if m:
                hes.append((int(m.group(1)), int(m.group(2))))
    return sorted(hes)

def read_akn_xml(year: int, number: int, zip_path: Path = AKN_ZIP_PATH) -> str | None:
    """Read AKN XML for a specific HE from the zip."""
    inner_path = akn_zip_path_for_he(year, number)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        try:
            return zf.read(inner_path).decode('utf-8')
        except KeyError:
            return None

def build_from_akn(year: int, number: int, output_path: Path,
                   zip_path: Path = AKN_ZIP_PATH, force: bool = False,
                   statutory_atoms_conn: sqlite3.Connection = None):
    """Build HE atom DB from AKN XML in government-proposal.zip.

    Per-HE DBs are ephemeral build artifacts (claims live in he_enrichments.db).
    The --force flag is kept for API compatibility but no longer guards claims.
    """
    global statute_resolver
    he_id = f'he-{number}-{year}'

    xml_str = read_akn_xml(year, number, zip_path)
    if xml_str is None:
        print(f"Error: HE {number}/{year} not found in {zip_path}")
        return

    parsed = parse_akn_xml(xml_str, he_id)
    meta = extract_akn_metadata(xml_str)
    meta['canonical_id'] = he_id

    # Extract refs from sections (using stored elements)
    for section in parsed['sections']:
        if '_elem' in section:
            section['_refs'] = _extract_akn_refs(section['_elem'])
            del section['_elem']

    if output_path.exists():
        output_path.unlink()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(output_path))
    populate_db(conn, parsed, meta, he_id, 'akn-xml',
                statutory_atoms_conn=statutory_atoms_conn)
    print_summary(conn, output_path, he_id, parsed)
    conn.close()


def _fetch_baseline_text(statutory_conn: sqlite3.Connection,
                         statute_id: str, section_num: str,
                         chapter_num: str = None) -> str | None:
    """Fetch current consolidated text from statutory_atoms.db for a section.

    statute_id: '2003/13' format
    section_num: '18', '2', '19 a' etc.
    chapter_num: '3', '11' etc. (for chapter-qualified statutes like ulosottokaari)
    Returns plain text or None if not found.
    """
    if statutory_conn is None:
        return None
    # Normalize section_num: '19 a' -> try both '19 a' and '19a'
    candidates = [section_num, section_num.replace(' ', '')]
    for sn in candidates:
        if chapter_num:
            row = statutory_conn.execute(
                "SELECT text FROM sections WHERE statute_id = ? AND section_num = ? AND chapter_num = ?",
                (statute_id, sn, chapter_num)
            ).fetchone()
        else:
            row = statutory_conn.execute(
                "SELECT text FROM sections WHERE statute_id = ? AND section_num = ?",
                (statute_id, sn)
            ).fetchone()
        if row:
            return row[0]
    return None


def populate_db(conn: sqlite3.Connection, parsed: dict, meta: dict, he_id: str, source: str,
                statutory_atoms_conn: sqlite3.Connection = None):
    """Populate the database from parsed HE data."""
    cur = conn.cursor()
    cur.executescript(DB_SCHEMA)

    seq = 0

    # Insert summary atom
    if parsed['summary']:
        seq += 1
        cur.execute(
            "INSERT INTO atoms (atom_id, atom_type, parent_id, seq, title, content, content_html, char_count) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ('summary', 'SUMMARY', None, seq, 'Pääasiallinen sisältö',
             parsed['summary'], parsed.get('summary_html', ''), len(parsed['summary']))
        )
        for statute_id, ref_type in extract_law_refs(parsed['summary']):
            cur.execute(
                "INSERT INTO law_refs (source_atom, target_statute, ref_type) VALUES (?, ?, ?)",
                ('summary', statute_id, ref_type)
            )

    # Insert narrative sections
    for section in parsed['sections']:
        seq += 1
        parent_id = compute_parent_id(section['section_num'])
        content = section.get('content', '')
        content_html = section.get('content_html', '')
        cur.execute(
            "INSERT OR IGNORE INTO atoms (atom_id, atom_type, parent_id, seq, title, content, content_html, char_count) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (section['atom_id'], section['atom_type'], parent_id, seq,
             section['title'], content, content_html, len(content))
        )
        # Use structured refs if available (AKN), else extract from text
        refs = section.get('_refs') or extract_law_refs(content)
        for statute_id, ref_type in refs:
            cur.execute(
                "INSERT INTO law_refs (source_atom, target_statute, ref_type) VALUES (?, ?, ?)",
                (section['atom_id'], statute_id, ref_type)
            )

    # Insert proposed statutes and pykälät
    diff_count = 0
    for statute in parsed['proposed_statutes']:
        seq += 1
        header_text = statute.get('header_text', '')
        cur.execute(
            "INSERT OR IGNORE INTO atoms (atom_id, atom_type, parent_id, seq, title, content, char_count) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (statute['atom_id'], 'PROPOSED_STATUTE', None, seq,
             statute['title'], header_text, len(header_text))
        )
        if statute['statute_id']:
            cur.execute(
                "INSERT INTO law_refs (source_atom, target_statute, ref_type) VALUES (?, ?, ?)",
                (statute['atom_id'], statute['statute_id'], 'AMENDS')
            )

        sid = statute['statute_id']  # e.g. '2003/13'

        for pykala in statute['pykalat']:
            seq += 1
            content = pykala.get('content', '')
            pyk_html = pykala.get('content_html', '')
            op_type = pykala.get('op_type', 'REPLACE')
            cur.execute(
                "INSERT OR IGNORE INTO atoms (atom_id, atom_type, parent_id, seq, title, content, content_html, char_count) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (pykala['atom_id'], 'PROPOSED_SECTION', statute['atom_id'],
                 seq, f"{pykala['pykala_num']} § {pykala['title']}",
                 content, pyk_html, len(content))
            )
            for statute_id, ref_type in extract_law_refs(content):
                cur.execute(
                    "INSERT INTO law_refs (source_atom, target_statute, ref_type) VALUES (?, ?, ?)",
                    (pykala['atom_id'], statute_id, ref_type)
                )

            # Build diff: fetch baseline text for REPLACE sections
            if sid and op_type == 'REPLACE':
                old_text = _fetch_baseline_text(
                    statutory_atoms_conn, sid, pykala['pykala_num'],
                    chapter_num=pykala.get('chapter_num'))
                if old_text is not None:
                    diff_id = f"{pykala['atom_id']}_diff"
                    cur.execute(
                        "INSERT OR IGNORE INTO diffs "
                        "(diff_id, statute_id, section, old_text, new_text, diff_html, justification_atom) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (diff_id, sid, pykala['pykala_num'], old_text,
                         content, op_type, pykala['atom_id'])
                    )
                    diff_count += 1
            elif sid and op_type == 'INSERT':
                diff_id = f"{pykala['atom_id']}_diff"
                cur.execute(
                    "INSERT OR IGNORE INTO diffs "
                    "(diff_id, statute_id, section, old_text, new_text, diff_html, justification_atom) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (diff_id, sid, pykala['pykala_num'], None,
                     content, op_type, pykala['atom_id'])
                )
                diff_count += 1

        # Insert kumotaan-only sections (deletions with no proposed text)
        for kum in statute.get('kumotaan_sections', []):
            if sid:
                sec_num = kum['section_num']
                old_text = _fetch_baseline_text(
                    statutory_atoms_conn, sid, sec_num)
                label = f"{sec_num} §"
                if kum.get('momentti'):
                    label += f" {kum['momentti']} mom."
                diff_id = f"law_{statute['seq']}_del_{sec_num.replace(' ', '')}{'_mom' + kum['momentti'] if kum.get('momentti') else ''}"
                cur.execute(
                    "INSERT OR IGNORE INTO diffs "
                    "(diff_id, statute_id, section, old_text, new_text, diff_html, justification_atom) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (diff_id, sid, label, old_text, '', 'DELETE', statute['atom_id'])
                )
                diff_count += 1

    total_atoms = seq

    # Insert metadata
    from datetime import datetime, timezone
    cur.execute(
        "INSERT OR REPLACE INTO metadata "
        "(he_id, number, year, title, ministry, finlex_uri, eduskunta_tunnus, "
        "hankeikkuna_uuid, lausuntopalvelu_guid, status, date_issued, "
        "laws_amended, page_count, atom_count, build_timestamp, build_model, source) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            he_id,
            meta.get('number'),
            meta.get('year'),
            meta.get('title'),
            meta.get('ministry'),
            meta.get('finlex_uri'),
            meta.get('eduskunta_tunnus'),
            meta.get('hankeikkuna_uuid'),
            meta.get('lausuntopalvelu_guid'),
            meta.get('status'),
            meta.get('date_issued'),
            meta.get('laws_amended'),
            None,
            total_atoms,
            datetime.now(timezone.utc).isoformat(),
            'phase1-deterministic',
            source,
        )
    )

    # Build FTS index
    cur.executescript("""
        DROP TABLE IF EXISTS atoms_fts;
        CREATE VIRTUAL TABLE atoms_fts USING fts5(
            atom_id, title, content,
            content='atoms',
            content_rowid='rowid'
        );
        INSERT INTO atoms_fts(atom_id, title, content)
            SELECT atom_id, COALESCE(title, ''), content FROM atoms;
    """)

    conn.commit()
    return total_atoms


def print_summary(conn: sqlite3.Connection, output_path: Path, he_id: str, parsed: dict):
    """Print build summary."""
    cur = conn.cursor()
    atom_counts = dict(cur.execute(
        "SELECT atom_type, COUNT(*) FROM atoms GROUP BY atom_type"
    ).fetchall())
    ref_count = cur.execute("SELECT COUNT(*) FROM law_refs").fetchone()[0]
    unique_statutes = cur.execute(
        "SELECT COUNT(DISTINCT target_statute) FROM law_refs"
    ).fetchone()[0]
    total_atoms = cur.execute("SELECT COUNT(*) FROM atoms").fetchone()[0]

    print(f"Built {output_path}")
    print(f"  HE: {he_id}")
    print(f"  Atoms: {total_atoms}")
    for atype, count in sorted(atom_counts.items()):
        print(f"    {atype}: {count}")
    print(f"  Law references: {ref_count} ({unique_statutes} unique statutes)")
    diff_counts = dict(cur.execute(
        "SELECT diff_html, COUNT(*) FROM diffs GROUP BY diff_html"
    ).fetchall())
    total_diffs = sum(diff_counts.values())
    print(f"  Proposed statutes: {len(parsed['proposed_statutes'])}")
    for s in parsed['proposed_statutes']:
        ops = {}
        for p in s['pykalat']:
            ot = p.get('op_type', '?')
            ops[ot] = ops.get(ot, 0) + 1
        kum = len(s.get('kumotaan_sections', []))
        if kum:
            ops['DELETE'] = ops.get('DELETE', 0) + kum
        ops_str = ', '.join(f'{v}{k[0]}' for k, v in sorted(ops.items()))
        print(f"    {s['seq']}. {s['title'][:60]} ({s['statute_id'] or '?'}) — {len(s['pykalat'])} § [{ops_str}]")
    if total_diffs:
        parts = [f"{diff_counts.get(k, 0)} {k}" for k in ('REPLACE', 'INSERT', 'DELETE') if diff_counts.get(k)]
        print(f"  Diffs: {total_diffs} ({', '.join(parts)})")
        with_baseline = cur.execute("SELECT COUNT(*) FROM diffs WHERE old_text IS NOT NULL AND old_text != ''").fetchone()[0]
        print(f"  Baseline text found: {with_baseline}/{total_diffs}")


# ===================================================================
# ENTRY POINTS
# ===================================================================

def _parse_he_id(he_id: str) -> tuple[int, int]:
    """Parse 'he-38-2025' into (year=2025, number=38)."""
    m = re.match(r'he-(\d+)-(\d{4})', he_id)
    if not m:
        raise ValueError(f"Invalid HE ID format: {he_id} (expected he-N-YYYY)")
    return int(m.group(2)), int(m.group(1))


def build_index(zip_path: Path, output_path: Path):
    """Scan all HEs in the zip and build a lightweight master index.

    Output: SQLite DB with he_index (metadata), he_statute_refs (HE→statute edges),
    and he_he_refs (HE→HE cross-references). Used by build_state_causal_map_db.py.
    """
    import time
    FINLEX_NS = 'http://data.finlex.fi/schema/finlex'

    zf = zipfile.ZipFile(str(zip_path), 'r')
    fins = sorted(n for n in zf.namelist()
                  if re.match(r'akn/fi/doc/government-proposal/\d{4}/\d+/fin@/main\.xml$', n))
    print(f'Building HE master index from {len(fins)} XMLs...')

    if output_path.exists():
        output_path.unlink()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(output_path))
    c = conn.cursor()
    c.executescript('''
        CREATE TABLE he_index (
            he_id TEXT PRIMARY KEY,
            year INTEGER,
            number INTEGER,
            title TEXT,
            ministry TEXT,
            date_issued TEXT
        );
        CREATE TABLE he_statute_refs (
            he_id TEXT NOT NULL,
            statute_ref TEXT NOT NULL,
            ref_type TEXT NOT NULL
        );
        CREATE TABLE he_he_refs (
            source_he TEXT NOT NULL,
            target_he TEXT NOT NULL
        );
        CREATE INDEX idx_hsr_he ON he_statute_refs(he_id);
        CREATE INDEX idx_hsr_statute ON he_statute_refs(statute_ref);
        CREATE INDEX idx_hhr_source ON he_he_refs(source_he);
    ''')

    t0 = time.time()
    n_refs = 0
    n_he_refs = 0

    for name in fins:
        m = re.match(r'akn/fi/doc/government-proposal/(\d{4})/(\d+)/fin@/main\.xml$', name)
        year, number = int(m.group(1)), int(m.group(2))
        he_id = f'he-{number}-{year}'

        with zf.open(name) as f:
            try:
                tree = ET.parse(f)
            except ET.ParseError:
                continue
        root = tree.getroot()
        ns = f'{{{AKN_NS}}}'

        # --- Metadata ---
        title = ''
        title_el = root.find(f'.//{ns}docTitle')
        if title_el is not None:
            title = _akn_text(title_el).strip()

        date_issued = ''
        work = root.find(f'.//{ns}FRBRWork')
        if work is not None:
            d = work.find(f'{ns}FRBRdate')
            if d is not None:
                date_issued = d.get('date', '')

        ministry = ''
        admin = root.find(f'.//{{{FINLEX_NS}}}administrativeBranch')
        if admin is not None:
            ministry = admin.get('refersTo', '').replace('#', '')

        c.execute('INSERT INTO he_index VALUES (?,?,?,?,?,?)',
                  (he_id, year, number, title, ministry, date_issued))

        # --- AMENDS refs (from finlex:affects — structured, high confidence) ---
        amends_set = set()
        affects = root.find(f'.//{{{FINLEX_NS}}}affects')
        if affects is not None:
            for sr in affects.findall(f'{{{FINLEX_NS}}}statuteReference'):
                ref_el = sr.find(f'{{{FINLEX_NS}}}ref')
                if ref_el is not None and ref_el.text:
                    raw = ref_el.text.strip()  # "380/2023" format
                    parts = raw.split('/')
                    if len(parts) == 2:
                        sid = f'{parts[1]}/{parts[0]}'  # -> "2023/380"
                        amends_set.add(sid)
                        c.execute('INSERT INTO he_statute_refs VALUES (?,?,?)',
                                  (he_id, sid, 'AMENDS'))
                        n_refs += 1

        # --- CITES refs (from <ref> hrefs in body — broader, includes rationale mentions) ---
        cites_set = set()
        for ref in root.iter(f'{ns}ref'):
            href = ref.get('href', '')
            if '/act/statute-consolidated/' in href:
                rm = re.search(r'/(\d{4})/(\d+(?:-\d+)?)', href)
                if rm:
                    sid = f'{rm.group(1)}/{rm.group(2)}'
                    if sid not in amends_set and sid not in cites_set:
                        cites_set.add(sid)
                        c.execute('INSERT INTO he_statute_refs VALUES (?,?,?)',
                                  (he_id, sid, 'CITES'))
                        n_refs += 1
            elif '/doc/government-proposal/' in href:
                rm = re.search(r'/(\d{4})/(\d+)', href)
                if rm:
                    target = f'he-{rm.group(2)}-{rm.group(1)}'
                    if target != he_id:
                        c.execute('INSERT INTO he_he_refs VALUES (?,?)',
                                  (he_id, target))
                        n_he_refs += 1

    conn.commit()

    # Summary
    elapsed = time.time() - t0
    total_hes = c.execute('SELECT COUNT(*) FROM he_index').fetchone()[0]
    total_amends = c.execute("SELECT COUNT(*) FROM he_statute_refs WHERE ref_type='AMENDS'").fetchone()[0]
    total_cites = c.execute("SELECT COUNT(*) FROM he_statute_refs WHERE ref_type='CITES'").fetchone()[0]
    distinct_statutes = c.execute('SELECT COUNT(DISTINCT statute_ref) FROM he_statute_refs').fetchone()[0]
    yr_min = c.execute('SELECT MIN(year) FROM he_index').fetchone()[0]
    yr_max = c.execute('SELECT MAX(year) FROM he_index').fetchone()[0]

    print(f'Done in {elapsed:.1f}s')
    print(f'  {total_hes} HEs ({yr_min}–{yr_max})')
    print(f'  {total_amends} AMENDS edges, {total_cites} CITES edges')
    print(f'  {distinct_statutes} distinct statutes referenced')
    print(f'  {n_he_refs} HE→HE cross-references')
    print(f'  Output: {output_path} ({output_path.stat().st_size / 1024:.0f} KB)')

    conn.close()
    zf.close()


def main():
    parser = argparse.ArgumentParser(description='Build HE atom database from AKN XML')
    parser.add_argument('he_ids', nargs='*',
                        help='HE IDs to build (e.g., he-38-2025). Default: all in zip.')
    parser.add_argument('--output', default=None,
                        help='Output .db path (default: .tmp/he_dbs/{he_id}.db)')
    parser.add_argument('--zip', default=None,
                        help=f'Path to government-proposal.zip (default: {AKN_ZIP_PATH})')
    parser.add_argument('--year', type=int, default=None,
                        help='Only process HEs from this year (with --all)')
    parser.add_argument('--force', action='store_true',
                        help='Kept for compatibility (per-HE DBs are now freely rebuildable; claims live in he_enrichments.db)')
    parser.add_argument('--build-index', action='store_true',
                        help='Build master HE index (metadata + refs) for state causal map')
    args = parser.parse_args()

    base = _ROOT
    zip_path = Path(args.zip) if args.zip else AKN_ZIP_PATH

    if not zip_path.exists():
        print(f"Error: AKN zip not found at {zip_path}")
        return

    if args.build_index:
        out = Path(args.output) if args.output else base / '.tmp' / 'he_master_index.db'
        build_index(zip_path, out)
        return

    # Initialize statute resolver (only needed for per-HE builds)
    registry_path = base / 'data' / 'statute_graph' / 'statute_alias_registry.csv'
    global statute_resolver
    statute_resolver = StatuteResolver(registry_path)

    # Open statutory_atoms.db for baseline text lookup (diffs)
    statutory_atoms_path = base / 'data' / 'statute_graph' / 'statutory_atoms.db'
    statutory_conn = None
    if statutory_atoms_path.exists():
        statutory_conn = sqlite3.connect(str(statutory_atoms_path))
        statutory_conn.row_factory = sqlite3.Row
        n_statutes = statutory_conn.execute("SELECT COUNT(*) FROM parse_stats").fetchone()[0]
        print(f"Loaded statutory_atoms.db: {n_statutes} statutes for diff baselines")
    else:
        print(f"Warning: {statutory_atoms_path} not found, diffs will have no baseline text")

    if args.he_ids:
        for he_id in args.he_ids:
            year, number = _parse_he_id(he_id)
            out = Path(args.output) if args.output else base / '.tmp' / 'he_dbs' / f'{he_id}.db'
            build_from_akn(year, number, out, zip_path, force=args.force,
                           statutory_atoms_conn=statutory_conn)
    else:
        # Build all HEs in zip
        all_hes = list_akn_hes(zip_path)
        if args.year:
            all_hes = [(y, n) for y, n in all_hes if y == args.year]
        print(f"Building {len(all_hes)} HEs from {zip_path.name}...")
        for year, number in all_hes:
            he_id = f'he-{number}-{year}'
            out = base / '.tmp' / 'he_dbs' / f'{he_id}.db'
            build_from_akn(year, number, out, zip_path, force=args.force,
                           statutory_atoms_conn=statutory_conn)

    if statutory_conn:
        statutory_conn.close()


if __name__ == '__main__':
    main()

"""Bake span_tag highlights into per-HE DB content_html fields.

Runs AFTER enrich + noise-lausunto + tag-spans. Inserts <span class="hl hl-{tag}">
directly into content_html columns so the viewer renders highlights without
client-side offset matching.

For documents with existing HTML (lausunto with noise-blocks, HE atoms, mietintö):
  - Matches span snippets against the plain-text `content` column to get offsets
  - Maps those offsets into the HTML by walking the HTML and tracking text position
  - Inserts highlight <span> tags at the right HTML positions

For PTK speeches (plain text only):
  - Builds HTML from plain text with paragraph splits + highlights

Usage:
    mev build bake-spans he-1-2025
    mev build bake-spans --all
"""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path

from mev.config import HE_DB_DIR, ENRICHMENTS_DB

TAG_FI = {
    'concern': 'Huoli',
    'question': 'Kysymys',
    'proposal': 'Ehdotus',
    'reference': 'Viittaus',
    'admission': 'Myönnytys',
    'fiscal': 'Talous',
    'law_ref': 'Lakiviittaus',
    'political': 'Poliittinen',
    'rhetoric': 'Retoriikka',
}


def _load_spans(enr_conn: sqlite3.Connection, he_id: str) -> dict[str, list[dict]]:
    """Load span_tag grouped by doc_type:doc_id."""
    try:
        rows = enr_conn.execute(
            "SELECT doc_type, doc_id, tag, snippet "
            "FROM span_tag WHERE he_id = ? ORDER BY doc_type, doc_id",
            (he_id,)
        ).fetchall()
    except sqlite3.OperationalError:
        return {}
    grouped: dict[str, list[dict]] = {}
    for r in rows:
        key = f"{r[0]}:{r[1]}"
        if key not in grouped:
            grouped[key] = []
        grouped[key].append({
            'tag': r[2], 'snippet': r[3],
        })
    return grouped


def _load_ref_links(enr_conn: sqlite3.Connection, he_id: str) -> dict[str, list[dict]]:
    """Load span_link grouped by source_doc_type:source_doc_id."""
    try:
        rows = enr_conn.execute(
            "SELECT source_doc_type, source_doc_id, snippet, "
            "target_doc_type, target_doc_id, target_name, confidence "
            "FROM span_link WHERE he_id = ? ORDER BY confidence DESC",
            (he_id,)
        ).fetchall()
    except sqlite3.OperationalError:
        return {}
    grouped: dict[str, list[dict]] = {}
    for r in rows:
        key = f"{r[0]}:{r[1]}"
        if key not in grouped:
            grouped[key] = []
        grouped[key].append({
            'snippet': r[2],
            'target_doc_type': r[3],
            'target_doc_id': r[4],
            'target_name': r[5],
            'confidence': r[6],
        })
    return grouped


def _find_snippet_in_html(html: str, snippet: str) -> tuple[int, int] | None:
    """Find a plain-text snippet within HTML, skipping tags.

    Returns (start_idx, end_idx) in the HTML string where the snippet's
    visible text begins and ends, or None if not found.
    """
    norm_snip = re.sub(r'\s+', ' ', snippet).strip()
    if not norm_snip:
        return None

    # Build list of (html_idx, char) for visible characters
    vis = []  # (html_index, char)
    i = 0
    while i < len(html):
        ch = html[i]
        if ch == '<':
            close = html.find('>', i)
            i = (close + 1) if close >= 0 else (i + 1)
            continue
        if ch == '&':
            semi = html.find(';', i, i + 12)
            if semi > 0:
                entity = html[i:semi + 1]
                # Decode common entities for matching
                decoded = entity.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>').replace('&nbsp;', ' ')
                if len(decoded) == 1:
                    vis.append((i, decoded))
                else:
                    vis.append((i, entity[1] if len(entity) > 1 else '?'))
                i = semi + 1
                continue
        vis.append((i, ch))
        i += 1

    # Build normalized visible text and index map
    vis_text = ''.join(c for _, c in vis)
    # Normalize whitespace in visible text for matching
    # We need to track which vis[] indices map to which norm positions
    norm_vis = []
    j = 0
    while j < len(vis_text):
        ch = vis_text[j]
        if ch in (' ', '\t', '\n', '\r'):
            norm_vis.append((vis[j][0], ' '))
            # Skip consecutive whitespace
            while j + 1 < len(vis_text) and vis_text[j + 1] in (' ', '\t', '\n', '\r'):
                j += 1
        else:
            norm_vis.append((vis[j][0], ch))
        j += 1

    norm_text = ''.join(c for _, c in norm_vis)
    idx = norm_text.find(norm_snip)
    if idx < 0:
        return None

    # Map back to HTML indices
    html_start = norm_vis[idx][0]
    end_norm_idx = idx + len(norm_snip) - 1
    if end_norm_idx < len(norm_vis):
        # Find the end: the HTML index AFTER the last matched char
        last_html_idx = norm_vis[end_norm_idx][0]
        # Advance past the last character in HTML (it might be multi-char entity)
        html_end = last_html_idx + 1
        # If it was an entity, advance past the semicolon
        if html[last_html_idx] == '&':
            semi = html.find(';', last_html_idx)
            if semi > 0:
                html_end = semi + 1
    else:
        html_end = len(html)

    return html_start, html_end


def _linkify_law_refs(html: str) -> str:
    """Convert hl-law_ref spans to clickable <a> links where a statute/HE ID can be parsed."""
    def _replace_law_span(m):
        full = m.group(0)
        text = m.group(1)

        # Try HE reference first (e.g. "HE 202/2024 vp")
        he_m = re.search(r'HE\s+(\d+)/(\d{4})', text)
        if he_m:
            href = f'/mev/he-viewer#he-{he_m.group(1)}-{he_m.group(2)}'
            return f'<a class="hl hl-law_ref" href="{href}" title="{TAG_FI["law_ref"]}">{text}</a>'

        # Try statute reference (e.g. "rikoslain (39/1889)" or "laki 417/2007")
        stat_m = re.search(r'(\d{1,5})/(\d{4})', text)
        if stat_m:
            href = f'/mev/lakikartta#detail/{stat_m.group(2)}/{stat_m.group(1)}'
            return f'<a class="hl hl-law_ref" href="{href}" title="{TAG_FI["law_ref"]}">{text}</a>'

        return full  # no parseable ID, keep as span

    return re.sub(
        r'<span class="hl hl-law_ref" title="[^"]*">(.*?)</span>',
        _replace_law_span,
        html,
    )


def _linkify_ref_spans(html: str, ref_links: list[dict]) -> str:
    """Convert hl-reference spans to clickable links using resolved span_link."""
    for link in ref_links:
        target = link.get('target_doc_type', '')
        tid = link.get('target_doc_id', '')
        if target == 'lausunto':
            href = f'#expert-{tid}'
            title_attr = link.get('target_name', 'lausunto') or 'lausunto'
        elif target == 'mietinto':
            href = f'#report-{tid.replace(" ", "_")}'
            title_attr = link.get('target_name', 'mietintö') or 'mietintö'
        else:
            continue  # ptk and unknown: no stable anchor

        snippet = link.get('snippet', '')
        if not snippet:
            continue
        # Match on first 50 chars of snippet (enough to be distinctive)
        snippet_prefix = re.escape(snippet[:50])
        pattern = (
            r'<span class="hl hl-reference" title="[^"]*">'
            r'([^<]*' + snippet_prefix + r'[^<]*)</span>'
        )
        replacement = (
            f'<a class="hl hl-reference" href="{href}" title="{title_attr}">'
            r'\1</a>'
        )
        html = re.sub(pattern, replacement, html, count=1)
    return html


def _insert_highlights_into_html(html: str, spans: list[dict]) -> str:
    """Insert <span class="hl"> around snippet matches in HTML.

    Finds each snippet in the HTML (tag-aware), wraps the matched range.
    Skips matches inside <details class="noise-block"> sections.
    """
    if not spans or not html:
        return html

    # Sort by snippet length descending (longer matches first to avoid partial overlaps)
    sorted_spans = sorted(spans, key=lambda s: -len(s.get('snippet', '')))

    # Find noise-block ranges to exclude
    noise_ranges = []
    for m in re.finditer(r'<details\s+class="noise-block"[^>]*>.*?</details>', html, re.DOTALL):
        noise_ranges.append((m.start(), m.end()))

    # Track which HTML character ranges are already highlighted
    used_ranges = list(noise_ranges)  # pre-seed with noise ranges

    insertions = []  # (start, end, tag)
    for s in sorted_spans:
        snippet = s.get('snippet', '')
        if not snippet or len(snippet) < 3:
            continue

        match = _find_snippet_in_html(html, snippet)
        if not match:
            continue

        h_start, h_end = match

        # Check overlap with existing highlights
        overlaps = any(not (h_end <= us or h_start >= ue) for us, ue in used_ranges)
        if overlaps:
            continue

        used_ranges.append((h_start, h_end))
        insertions.append((h_start, h_end, s['tag']))

    if not insertions:
        return html

    # Sort by position (left to right), process right to left to preserve indices
    insertions.sort(key=lambda x: x[0])
    for h_start, h_end, tag in reversed(insertions):
        tag_title = TAG_FI.get(tag, tag)
        open_tag = f'<span class="hl hl-{tag}" title="{tag_title}">'
        close_tag = '</span>'
        html = html[:h_start] + open_tag + html[h_start:h_end] + close_tag + html[h_end:]

    return _linkify_law_refs(html)


def _plain_text_to_highlighted_html(text: str, spans: list[dict]) -> str:
    """Build HTML from plain text with paragraph splits and span highlights.
    Used for PTK speeches which have no HTML source."""
    if not text:
        return ''

    clean = re.sub(r'[ \t]+', ' ', text).strip()
    if not clean:
        return ''

    def _esc(s):
        return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

    # Step 1: Build paragraph HTML from plain text
    breaks = set()
    for m in re.finditer(r'Arvoisa\s+(?:herra\s+|rouva\s+)?(?:puhemies|puheenjohtaja)!', clean, re.IGNORECASE):
        if m.start() > 50:
            breaks.add(m.start())
    last_break = 0
    for i in range(len(clean) - 1):
        if clean[i] == '.' and clean[i + 1] == ' ' and (i - last_break) > 450:
            breaks.add(i + 2)
            last_break = i + 2

    break_points = sorted(breaks)
    parts = []
    prev = 0
    for bp in break_points:
        if bp > prev:
            parts.append(f'<p>{_esc(clean[prev:bp])}</p>')
            prev = bp
    if prev < len(clean):
        parts.append(f'<p>{_esc(clean[prev:])}</p>')

    base_html = ''.join(parts) if parts else f'<p>{_esc(clean)}</p>'

    # Step 2: Insert span highlights using snippet matching (same as HTML docs)
    # Note: _insert_highlights_into_html already calls _linkify_law_refs internally.
    if spans:
        base_html = _insert_highlights_into_html(base_html, spans)
    else:
        base_html = _linkify_law_refs(base_html)

    return base_html


def bake_he(he_id: str, enr_conn: sqlite3.Connection) -> dict:
    """Bake span highlights into one per-HE DB."""
    db_path = HE_DB_DIR / f"{he_id}.db"
    if not db_path.exists():
        return {'n_baked': 0}

    spans_by_doc = _load_spans(enr_conn, he_id)
    ref_links_by_doc = _load_ref_links(enr_conn, he_id)
    if not spans_by_doc and not ref_links_by_doc:
        return {'n_baked': 0}

    conn = sqlite3.connect(str(db_path))
    n_baked = 0

    # Bake into expert_statement (lausunto)
    try:
        rows = conn.execute(
            "SELECT statement_id, content, content_html FROM expert_statement"
        ).fetchall()
        for r in rows:
            key = f"lausunto:{r[0]}"
            doc_spans = spans_by_doc.get(key)
            html = r[2] or ''
            if not html:
                continue
            new_html = _insert_highlights_into_html(html, doc_spans or []) if doc_spans else _linkify_law_refs(html)
            ref_links = ref_links_by_doc.get(key, [])
            if ref_links:
                new_html = _linkify_ref_spans(new_html, ref_links)
            if new_html != html:
                conn.execute(
                    "UPDATE expert_statement SET content_html = ? WHERE statement_id = ?",
                    (new_html, r[0])
                )
                n_baked += 1
    except sqlite3.OperationalError:
        pass

    # Bake into atoms (HE)
    try:
        rows = conn.execute(
            "SELECT atom_id, content_html FROM atoms WHERE content_html IS NOT NULL"
        ).fetchall()
        for r in rows:
            key = f"he:{r[0]}"
            doc_spans = spans_by_doc.get(key)
            html = r[1] or ''
            if not html:
                continue
            new_html = _insert_highlights_into_html(html, doc_spans or []) if doc_spans else _linkify_law_refs(html)
            ref_links = ref_links_by_doc.get(key, [])
            if ref_links:
                new_html = _linkify_ref_spans(new_html, ref_links)
            if new_html != html:
                conn.execute(
                    "UPDATE atoms SET content_html = ? WHERE atom_id = ?",
                    (new_html, r[0])
                )
                n_baked += 1
    except sqlite3.OperationalError:
        pass

    # Bake into committee_report (mietintö)
    try:
        rows = conn.execute(
            "SELECT tunnus, content_html FROM committee_report"
        ).fetchall()
        for r in rows:
            key = f"mietinto:{r[0]}"
            doc_spans = spans_by_doc.get(key)
            html = r[1] or ''
            if not html:
                continue
            new_html = _insert_highlights_into_html(html, doc_spans or []) if doc_spans else _linkify_law_refs(html)
            ref_links = ref_links_by_doc.get(key, [])
            if ref_links:
                new_html = _linkify_ref_spans(new_html, ref_links)
            if new_html != html:
                conn.execute(
                    "UPDATE committee_report SET content_html = ? WHERE tunnus = ?",
                    (new_html, r[0])
                )
                n_baked += 1
    except sqlite3.OperationalError:
        pass

    # Bake PTK speeches — create content_html from plain text + spans + para splits
    try:
        rows = conn.execute("SELECT rowid, text FROM ptk_speeches").fetchall()
        # Add content_html column if not exists
        try:
            conn.execute("ALTER TABLE ptk_speeches ADD COLUMN content_html TEXT")
        except sqlite3.OperationalError:
            pass  # already exists

        for r in rows:
            key = f"ptk:{r[0]}"
            doc_spans = spans_by_doc.get(key)
            ref_links = ref_links_by_doc.get(key, [])
            text = r[1] or ''
            if not text:
                continue
            new_html = _plain_text_to_highlighted_html(text, doc_spans or [])
            if ref_links and new_html:
                new_html = _linkify_ref_spans(new_html, ref_links)
            if new_html:
                conn.execute(
                    "UPDATE ptk_speeches SET content_html = ? WHERE rowid = ?",
                    (new_html, r[0])
                )
                n_baked += 1
    except sqlite3.OperationalError:
        pass

    conn.commit()
    conn.close()
    return {'n_baked': n_baked}


def main():
    import argparse
    import sys
    parser = argparse.ArgumentParser(description='Bake span highlights into per-HE DB HTML')
    parser.add_argument('he_id', nargs='?', help='Specific HE')
    parser.add_argument('--all', action='store_true')
    args = parser.parse_args()

    if not ENRICHMENTS_DB.exists():
        print("No enrichments DB")
        sys.exit(1)

    enr_conn = sqlite3.connect(str(ENRICHMENTS_DB))

    if args.all:
        he_ids = sorted(p.stem for p in HE_DB_DIR.glob("he-*.db"))
    elif args.he_id:
        he_ids = [args.he_id]
    else:
        parser.print_help()
        sys.exit(1)

    total = 0
    for he_id in he_ids:
        result = bake_he(he_id, enr_conn)
        n = result['n_baked']
        if n > 0:
            print(f"  {he_id}: {n} docs baked")
        total += n

    enr_conn.close()
    print(f"\nDone: {total} docs baked across {len(he_ids)} HEs")


if __name__ == '__main__':
    main()

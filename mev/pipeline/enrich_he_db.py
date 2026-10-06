"""
Copy enrichment data from corpus-wide DBs into per-HE SQLite DBs for he-viewer.

Pipeline position: runs AFTER build_he_db.py, detect_unreason.py, tag detectors.
BEFORE upload to vs.kunnas.com. Per-HE DBs are the viewer's sole data source.

Reads:
    data/legislative_index.sqlite       [expert_statement, committee_report, ptk_speech, ev_document]
    .tmp/scrutiny/scrutiny_analysis.json [scrutiny classifications]
    .tmp/he_enrichments.db              [claims, unreason_flag, sentence_tag, irony_pattern,
                                         lausunto_tag, mietinto_tag, ptk_speech_tag,
                                         discourse_node, discourse_edge]
    .tmp/he_dbs/he-*.db                 [existing atom databases]

Adds to each per-HE DB:
    claims table               — from he_enrichments.db (LLM-extracted empirical claims)
    expert_statement table     — from legislative_index (expert content + scrutiny)
    committee_report table     — from legislative_index (mietintö + lausunto)
    scrutiny_summary table     — aggregate scrutiny stats
    unreason_flag table       — from he_enrichments.db (Tier 0-1 self-indicting detections)
    sentence_tag table        — from he_enrichments.db (3-dim sentence tagger: role×quality×topic)
    irony_pattern table       — from he_enrichments.db (mechanical irony detection from tag proximity)
    lausunto_tag table        — from he_enrichments.db (per-sentence expert statement tags)
    mietinto_tag table        — from he_enrichments.db (per-paragraph committee report tags)
    ptk_speech_tag table      — from he_enrichments.db (per-speech PTK stance/content tags)
    discourse_node table      — from he_enrichments.db (unified discourse graph nodes)
    discourse_edge table      — from he_enrichments.db (cross-document discourse edges)

Consumed by: mr_website_public/mev/he-viewer.js (loads per-HE DB via sql.js)

Usage:
    mev build enrich he-38-2025
    mev build enrich --all
"""

import argparse
import json
import re
import sqlite3
import sys

from mev.config import ROOT, HE_DB_DIR, ENRICHMENTS_DB, INDEX_DB, CAUSAL_MAP_DB
SCRUTINY_PATH = ROOT / ".tmp" / "scrutiny" / "scrutiny_analysis.json"


SAFE_TAGS = {'p', 'br', 'strong', 'b', 'em', 'a', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
             'ul', 'ol', 'li', 'table', 'thead', 'tbody', 'tr', 'th', 'td',
             'div', 'span', 'blockquote', 'hr', 'aside', 'sup', 'sub', 'section',
             'strike', 'img'}
SAFE_ATTRS = {'href', 'id', 'class', 'title', 'colspan', 'rowspan', 'target',
              'src', 'alt', 'width', 'height', 'cellpadding', 'cellspacing',
              'valign', 'align', 'aria-describedby', 'aria-label'}


_sanitize_warnings = set()


def sanitize_html(html: str) -> str:
    """Allow only safe tags and attributes, strip everything else."""
    def replace_tag(m):
        full = m.group(0)
        cm = re.match(r'</([a-zA-Z][a-zA-Z0-9]*)\s*>', full)
        if cm:
            tag = cm.group(1).lower()
            if tag in SAFE_TAGS:
                return full
            if tag not in _sanitize_warnings:
                _sanitize_warnings.add(tag)
                print(f"  WARN: stripped tag </{tag}>")
            return ''
        om = re.match(r'<([a-zA-Z][a-zA-Z0-9]*)((?:\s+[^>]*)?)\s*/?\s*>', full)
        if not om:
            return ''
        tag = om.group(1).lower()
        if tag not in SAFE_TAGS:
            if tag not in _sanitize_warnings:
                _sanitize_warnings.add(tag)
                print(f"  WARN: stripped tag <{tag}>")
            return ''
        attrs_str = om.group(2) or ''
        safe_attrs = []
        for am in re.finditer(r'([a-zA-Z-]+)\s*=\s*"([^"]*)"', attrs_str):
            attr_name = am.group(1).lower()
            if attr_name in SAFE_ATTRS:
                safe_attrs.append(f'{am.group(1)}="{am.group(2)}"')
            else:
                key = f'@{attr_name}'
                if key not in _sanitize_warnings:
                    _sanitize_warnings.add(key)
                    print(f"  WARN: stripped attr {attr_name}= on <{tag}>")
        attr_part = (' ' + ' '.join(safe_attrs)) if safe_attrs else ''
        slash = '/' if full.rstrip('>').endswith('/') else ''
        return f'<{tag}{attr_part}{slash}>'
    return re.sub(r'<[^>]+>', replace_tag, html)


VOID_TAGS = frozenset(['br', 'hr', 'img', 'input', 'meta', 'link', 'col',
                       'area', 'base', 'embed', 'source', 'track', 'wbr'])


def balance_html_tags(html: str) -> str:
    """Close any unclosed tags to prevent DOM leakage."""
    stack = []
    for m in re.finditer(r'</?([a-zA-Z][a-zA-Z0-9]*)\b[^>]*/?\s*>', html):
        full = m.group(0)
        tag = m.group(1).lower()
        if tag in VOID_TAGS or full.rstrip('>').endswith('/'):
            continue
        if full.startswith('</'):
            for i in range(len(stack) - 1, -1, -1):
                if stack[i] == tag:
                    stack.pop(i)
                    break
        else:
            stack.append(tag)
    for tag in reversed(stack):
        html += f'</{tag}>'
    return html


def clean_edilex_structure(html: str) -> str:
    """Strip Edilex page structure artifacts from content HTML."""
    html = re.sub(r'^(\s*</[a-zA-Z]+>\s*)+', '', html)
    html = re.sub(r'<div\s+class="anchor-wrapper">.*?</div>', '', html, flags=re.DOTALL)
    html = html.strip()
    return balance_html_tags(html)


def rewrite_edilex_links(html: str) -> str:
    """Rewrite Edilex internal links to lakikartta / he-viewer / Finlex."""
    def rewrite_href(m):
        href = m.group(1)
        lm = re.match(r'/lainsaadanto/(\d{4})(\d+)', href)
        if lm:
            return f'href="/mev/lakikartta#detail/{lm.group(1)}/{lm.group(2)}"'
        hm = re.match(r'/he/(\d{4})(\d+)', href)
        if hm:
            return f'href="/mev/he-viewer#he-{hm.group(2)}-{hm.group(1)}"'
        mm = re.match(r'/mt/(\w+)', href)
        if mm:
            return f'href="/mev/he-viewer#mietinto/{mm.group(1)}"'
        if href.startswith('/'):
            print(f"  WARN: unrewritten edilex link: {href}")
            return f'href="https://www.edilex.fi{href}"'
        return m.group(0)
    return re.sub(r'href="(/[^"]*)"', rewrite_href, html)


def strip_html(text: str) -> str:
    """Strip HTML tags to plain text (for char_count / FTS)."""
    clean = re.sub(r'<[^>]+>', ' ', text)
    clean = clean.replace('&nbsp;', ' ').replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
    return re.sub(r'\s+', ' ', clean).strip()


def he_id_to_tunnus(he_id: str) -> str:
    """Convert he-241-2020 → HE 241/2020 vp"""
    m = re.match(r"he-(\d+)-(\d+)", he_id)
    return f"HE {m.group(1)}/{m.group(2)} vp" if m else ""


def load_scrutiny() -> dict:
    """Load scrutiny analysis, keyed by he_id."""
    if not SCRUTINY_PATH.exists():
        return {}
    with open(SCRUTINY_PATH) as f:
        data = json.load(f)
    return {he['he_id']: he for he in data if 'error' not in he}


def enrich_he(he_id: str, index_conn: sqlite3.Connection, scrutiny: dict, enr_conn=None):
    """Add expert + scrutiny + enrichment tables to one per-HE database."""
    db_path = HE_DB_DIR / f"{he_id}.db"
    if not db_path.exists():
        print(f"  {he_id}: DB not found at {db_path}")
        return

    conn = sqlite3.connect(str(db_path))
    c = conn.cursor()

    # Drop existing enrichment tables (idempotent rebuild)
    c.execute("DROP TABLE IF EXISTS expert_statement")
    c.execute("DROP TABLE IF EXISTS committee_report")
    c.execute("DROP TABLE IF EXISTS scrutiny_summary")

    # Create tables
    c.execute('''CREATE TABLE expert_statement (
        statement_id TEXT PRIMARY KEY,
        expert_name TEXT,
        date TEXT,
        committee TEXT,
        content TEXT,
        content_html TEXT,
        content_chars INTEGER,
        scrutiny_type TEXT,
        keyword_overlap REAL,
        claims_total INTEGER,
        claims_found INTEGER,
        sentences_total INTEGER,
        sentences_reflected INTEGER,
        matched_atoms TEXT,
        pykalat TEXT,
        mechanism_keywords TEXT
    )''')

    c.execute('''CREATE TABLE committee_report (
        tunnus TEXT PRIMARY KEY,
        committee TEXT,
        report_type TEXT,
        content TEXT,
        content_html TEXT,
        content_chars INTEGER
    )''')

    c.execute('''CREATE TABLE scrutiny_summary (
        key TEXT PRIMARY KEY,
        value TEXT
    )''')

    # Load expert statements from legislative_index
    experts = index_conn.execute(
        "SELECT statement_id, expert_title, date, committee, content "
        "FROM expert_statement WHERE he_id = ? AND length(content) > 0 "
        "ORDER BY date",
        (he_id,)
    ).fetchall()

    # Build scrutiny lookup keyed by statement_id
    scr_data = scrutiny.get(he_id, {})
    scr_by_id = {}
    for r in scr_data.get('expert_results', []):
        scr_by_id[r['statement_id']] = r

    expert_count = 0
    for exp in experts:
        sid = exp['statement_id']
        raw_html = exp['content'] or ''
        plain = strip_html(raw_html)
        safe_html = clean_edilex_structure(rewrite_edilex_links(sanitize_html(raw_html)))

        # Parse expert name
        title = exp['expert_title'] or ''
        name_match = re.search(
            r'\d{2}\.\d{2}\.\d{4}\s+(.+?)\s*(?:Asiantuntijalausunto|/\s*vastine)',
            title
        )
        expert_name = name_match.group(1).strip() if name_match else title[:80]

        # Clean malformed API names: "HE 1/2025 vp HaV 20.03.2025 erityisasiantuntija ..."
        if not name_match:
            prefix_match = re.match(
                r'HE\s+\d+/\d{4}\s+vp\s+\S+\s+\d{2}\.\d{2}\.\d{4}\s+(.+)',
                title
            )
            if prefix_match:
                expert_name = prefix_match.group(1).strip()

        # Scrutiny data for this expert
        scr = scr_by_id.get(sid, {})
        resp = scr.get('committee_response', {})

        c.execute(
            'INSERT OR REPLACE INTO expert_statement VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
            (
                sid,
                expert_name,
                exp['date'],
                exp['committee'],
                plain,
                safe_html,
                len(plain),
                resp.get('response_type', ''),
                resp.get('keyword_overlap_ratio', 0.0),
                resp.get('specific_claims_total', 0),
                resp.get('specific_claims_found', 0),
                resp.get('sentences_total', 0),
                resp.get('sentences_reflected', 0),
                json.dumps(scr.get('atom_matches', []), ensure_ascii=False),
                json.dumps(scr.get('pykalat_mentioned', []), ensure_ascii=False),
                json.dumps(scr.get('mechanism_keywords', []), ensure_ascii=False),
            )
        )
        expert_count += 1

    # Load committee reports
    reports = index_conn.execute(
        "SELECT tunnus, committee, report_type, edilex_html, content "
        "FROM committee_report WHERE he_id = ?",
        (he_id,)
    ).fetchall()

    report_count = 0
    for rpt in reports:
        raw_html = rpt['edilex_html'] or rpt['content'] or ''
        plain = strip_html(raw_html)
        safe_html = clean_edilex_structure(rewrite_edilex_links(sanitize_html(raw_html)))
        c.execute(
            'INSERT OR REPLACE INTO committee_report VALUES (?,?,?,?,?,?)',
            (
                rpt['tunnus'],
                rpt['committee'],
                rpt['report_type'],
                plain,
                safe_html,
                len(plain),
            )
        )
        report_count += 1

    # Scrutiny summary
    if scr_data:
        by_type = scr_data.get('by_response_type', {})
        summary = {
            'total_experts_analyzed': str(scr_data.get('total_experts_analyzed', 0)),
            'total_experts_raw': str(scr_data.get('total_experts_raw', 0)),
            'scrutiny_ignored_rate': f"{scr_data.get('scrutiny_ignored_rate', 0):.4f}",
            'by_response_type': json.dumps(by_type, ensure_ascii=False),
            'mietinto_tunnus': (scr_data.get('mietinto') or {}).get('tunnus', ''),
            'mietinto_committee': (scr_data.get('mietinto') or {}).get('committee', ''),
        }
        for k, v in summary.items():
            c.execute('INSERT INTO scrutiny_summary VALUES (?,?)', (k, v))

    # Drop legacy mechanism_audit (replaced by fine-grained sentence_tag + claims + unreason)
    c.execute("DROP TABLE IF EXISTS mechanism_audit")

    # Claims from he_enrichments.db
    claim_count = 0
    if enr_conn:
        c.execute("DELETE FROM claims")
        try:
            eclaims = enr_conn.execute(
                "SELECT claim_id, claim_type, text, quote, source_atom, "
                "amount_eur, scale, time_horizon, verifiability, confidence "
                "FROM claim WHERE he_id = ?",
                (he_id,)
            ).fetchall()
            for ec in eclaims:
                c.execute(
                    "INSERT OR REPLACE INTO claims VALUES (?,?,?,?,?,?,?,?,?,?)",
                    tuple(ec)
                )
            claim_count = len(eclaims)
        except sqlite3.OperationalError:
            pass  # claims table may not exist in enrichments DB

    # Sentence tags from he_enrichments.db
    tag_count = 0
    irony_count = 0
    if enr_conn:
        c.execute("DROP TABLE IF EXISTS sentence_tag")
        c.execute('''CREATE TABLE sentence_tag (
            atom_id     TEXT NOT NULL,
            sent_idx    INTEGER NOT NULL,
            sent_text   TEXT NOT NULL,
            sent_kind   TEXT NOT NULL,
            role        TEXT,
            quality     TEXT,
            topic       TEXT,
            eur_amounts TEXT,
            PRIMARY KEY (atom_id, sent_idx)
        )''')
        try:
            stags = enr_conn.execute(
                "SELECT atom_id, sent_idx, sent_text, sent_kind, "
                "role, quality, topic, eur_amounts "
                "FROM sentence_tag WHERE he_id = ?",
                (he_id,)
            ).fetchall()
            for st in stags:
                c.execute(
                    "INSERT OR REPLACE INTO sentence_tag VALUES (?,?,?,?,?,?,?,?)",
                    tuple(st)
                )
            tag_count = len(stags)
        except sqlite3.OperationalError:
            pass

        c.execute("DROP TABLE IF EXISTS irony_pattern")
        c.execute('''CREATE TABLE irony_pattern (
            atom_id      TEXT NOT NULL,
            pattern_type TEXT NOT NULL,
            sid1         INTEGER NOT NULL,
            sid2         INTEGER,
            detail       TEXT,
            PRIMARY KEY (atom_id, pattern_type, sid1)
        )''')
        try:
            ipatterns = enr_conn.execute(
                "SELECT atom_id, pattern_type, sid1, sid2, detail "
                "FROM irony_pattern WHERE he_id = ?",
                (he_id,)
            ).fetchall()
            for ip in ipatterns:
                c.execute(
                    "INSERT OR REPLACE INTO irony_pattern VALUES (?,?,?,?,?)",
                    tuple(ip)
                )
            irony_count = len(ipatterns)
        except sqlite3.OperationalError:
            pass

    # Unreason flags from he_enrichments.db
    unreason_count = 0
    if enr_conn:
        c.execute("DROP TABLE IF EXISTS unreason_flag")
        c.execute('''CREATE TABLE unreason_flag (
            detector TEXT PRIMARY KEY,
            severity INTEGER NOT NULL,
            evidence_atoms TEXT,
            evidence_text TEXT,
            meta TEXT
        )''')
        try:
            uflags = enr_conn.execute(
                "SELECT detector, severity, evidence_atoms, evidence_text, meta "
                "FROM unreason_flag WHERE he_id = ?",
                (he_id,)
            ).fetchall()
            for uf in uflags:
                c.execute(
                    "INSERT INTO unreason_flag VALUES (?,?,?,?,?)",
                    tuple(uf)
                )
            unreason_count = len(uflags)
        except sqlite3.OperationalError:
            pass  # unreason_flag table may not exist

    # PTK plenary speeches
    ptk_count = 0
    he_tunnus = he_id_to_tunnus(he_id)
    if he_tunnus:
        c.execute("DROP TABLE IF EXISTS ptk_speeches")
        c.execute('''CREATE TABLE ptk_speeches (
            ptk_tunnus TEXT,
            speaker    TEXT,
            party      TEXT,
            role       TEXT,
            speech_time TEXT,
            text       TEXT,
            content_html TEXT
        )''')
        c.execute("CREATE INDEX IF NOT EXISTS idx_ptksp_ptk ON ptk_speeches(ptk_tunnus)")
        ptk_rows = index_conn.execute(
            "SELECT ptk_tunnus, "
            "TRIM(COALESCE(speaker_first,'') || ' ' || COALESCE(speaker_last,'')), "
            "party, role, speech_time, text "
            "FROM ptk_speech WHERE he_tunnus = ? ORDER BY rowid",
            (he_tunnus,)
        ).fetchall()
        c.executemany("INSERT INTO ptk_speeches VALUES (?,?,?,?,?,?,NULL)", ptk_rows)
        # Initialize content_html from plain text (paragraph-wrapped); bake_spans will enrich further
        c.execute("""
            UPDATE ptk_speeches SET content_html =
            '<p>' || replace(replace(text, char(10)||char(10), '</p><p>'), char(10), '<br>') || '</p>'
            WHERE content_html IS NULL AND text IS NOT NULL
        """)
        ptk_count = len(ptk_rows)

    # EV decisions (parliament's enacted response to this HE)
    ev_count = 0
    if he_tunnus:
        c.execute("DROP TABLE IF EXISTS ev_decisions")
        c.execute('''CREATE TABLE ev_decisions (
            ev_tunnus     TEXT PRIMARY KEY,
            date          TEXT,
            title         TEXT,
            decision_text TEXT
        )''')
        ev_rows = index_conn.execute(
            "SELECT ev_tunnus, date, title, decision_text "
            "FROM ev_document, json_each(he_tunnus_json) "
            "WHERE json_each.value = ?",
            (he_tunnus,)
        ).fetchall()
        c.executemany("INSERT OR REPLACE INTO ev_decisions VALUES (?,?,?,?)", ev_rows)
        ev_count = len(ev_rows)

    # Committee dissenting opinions
    diss_count = 0
    c.execute("DROP TABLE IF EXISTS dissents")
    c.execute('''CREATE TABLE dissents (
        tunnus          TEXT,
        text            TEXT,
        signatories_json TEXT
    )''')
    diss_rows = index_conn.execute(
        "SELECT vd.tunnus, vd.text, vd.signatories_json "
        "FROM vaski_dissent vd "
        "JOIN committee_report cr ON cr.report_id = vd.report_id "
        "WHERE cr.he_id = ?",
        (he_id,)
    ).fetchall()
    c.executemany("INSERT INTO dissents VALUES (?,?,?)", diss_rows)
    diss_count = len(diss_rows)

    # Lausunto tags from he_enrichments.db
    lau_tag_count = 0
    if enr_conn:
        c.execute("DROP TABLE IF EXISTS lausunto_tag")
        c.execute('''CREATE TABLE lausunto_tag (
            statement_id    TEXT NOT NULL,
            expert_name     TEXT,
            sent_idx        INTEGER NOT NULL,
            sent_text       TEXT NOT NULL,
            role            TEXT,
            quality         TEXT,
            topic           TEXT,
            eur_amounts     TEXT,
            is_government   INTEGER DEFAULT 0,
            PRIMARY KEY (statement_id, sent_idx)
        )''')
        try:
            rows = enr_conn.execute(
                "SELECT statement_id, expert_name, sent_idx, sent_text, "
                "role, quality, topic, eur_amounts, is_government "
                "FROM lausunto_tag WHERE he_id = ?",
                (he_id,)
            ).fetchall()
            c.executemany(
                "INSERT OR REPLACE INTO lausunto_tag VALUES (?,?,?,?,?,?,?,?,?)",
                rows
            )
            lau_tag_count = len(rows)
        except sqlite3.OperationalError:
            pass

    # Mietinto tags from he_enrichments.db
    miet_tag_count = 0
    if enr_conn:
        c.execute("DROP TABLE IF EXISTS mietinto_tag")
        c.execute('''CREATE TABLE mietinto_tag (
            report_id   TEXT NOT NULL,
            committee   TEXT,
            para_idx    INTEGER NOT NULL,
            para_text   TEXT NOT NULL,
            section     TEXT,
            role        TEXT,
            quality     TEXT,
            topic       TEXT,
            PRIMARY KEY (report_id, para_idx)
        )''')
        try:
            rows = enr_conn.execute(
                "SELECT report_id, committee, para_idx, para_text, "
                "section, role, quality, topic "
                "FROM mietinto_tag WHERE he_id = ?",
                (he_id,)
            ).fetchall()
            c.executemany(
                "INSERT OR REPLACE INTO mietinto_tag VALUES (?,?,?,?,?,?,?,?)",
                rows
            )
            miet_tag_count = len(rows)
        except sqlite3.OperationalError:
            pass

    # PTK speech tags from he_enrichments.db
    ptk_tag_count = 0
    if enr_conn:
        c.execute("DROP TABLE IF EXISTS ptk_speech_tag")
        c.execute('''CREATE TABLE ptk_speech_tag (
            speech_rowid INTEGER NOT NULL PRIMARY KEY,
            ptk_tunnus   TEXT,
            speaker      TEXT,
            party        TEXT,
            stance       TEXT,
            content      TEXT,
            topic        TEXT
        )''')
        try:
            rows = enr_conn.execute(
                "SELECT speech_rowid, ptk_tunnus, speaker, party, "
                "stance, content, topic "
                "FROM ptk_speech_tag WHERE he_id = ?",
                (he_id,)
            ).fetchall()
            c.executemany(
                "INSERT OR REPLACE INTO ptk_speech_tag VALUES (?,?,?,?,?,?,?)",
                rows
            )
            ptk_tag_count = len(rows)
        except sqlite3.OperationalError:
            pass

    # Discourse nodes from he_enrichments.db
    disc_node_count = 0
    if enr_conn:
        c.execute("DROP TABLE IF EXISTS discourse_node")
        c.execute('''CREATE TABLE discourse_node (
            node_idx        INTEGER NOT NULL PRIMARY KEY,
            doc_type        TEXT NOT NULL,
            doc_id          TEXT,
            sent_idx        INTEGER,
            text            TEXT,
            role            TEXT,
            quality         TEXT,
            topic           TEXT,
            unified_role    TEXT,
            unified_quality TEXT,
            source_name     TEXT,
            eur_amounts     TEXT,
            section_refs    TEXT
        )''')
        try:
            rows = enr_conn.execute(
                "SELECT node_idx, doc_type, doc_id, sent_idx, text, "
                "role, quality, topic, unified_role, unified_quality, "
                "source_name, eur_amounts, section_refs "
                "FROM discourse_node WHERE he_id = ?",
                (he_id,)
            ).fetchall()
            c.executemany(
                "INSERT OR REPLACE INTO discourse_node VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                rows
            )
            disc_node_count = len(rows)
        except sqlite3.OperationalError:
            pass

    # Discourse edges from he_enrichments.db
    disc_edge_count = 0
    if enr_conn:
        c.execute("DROP TABLE IF EXISTS discourse_edge")
        c.execute('''CREATE TABLE discourse_edge (
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
        )''')
        try:
            rows = enr_conn.execute(
                "SELECT source_idx, target_idx, source_type, target_type, "
                "topic, edge_type, confidence, source_text, target_text, "
                "source_name, target_name "
                "FROM discourse_edge WHERE he_id = ?",
                (he_id,)
            ).fetchall()
            c.executemany(
                "INSERT INTO discourse_edge VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                rows
            )
            disc_edge_count = len(rows)
        except sqlite3.OperationalError:
            pass

    # Span tags from he_enrichments.db
    span_count = 0
    if enr_conn:
        c.execute("DROP TABLE IF EXISTS span_tag")
        c.execute('''CREATE TABLE span_tag (
            doc_type    TEXT NOT NULL,
            doc_id      TEXT NOT NULL,
            tag         TEXT NOT NULL,
            snippet     TEXT NOT NULL
        )''')
        try:
            rows = enr_conn.execute(
                "SELECT doc_type, doc_id, tag, snippet "
                "FROM span_tag WHERE he_id = ?",
                (he_id,)
            ).fetchall()
            c.executemany("INSERT INTO span_tag VALUES (?,?,?,?)", rows)
            span_count = len(rows)
        except sqlite3.OperationalError:
            pass

    # Span links (resolved V-references) from he_enrichments.db
    link_count = 0
    if enr_conn:
        c.execute("DROP TABLE IF EXISTS span_link")
        c.execute('''CREATE TABLE span_link (
            source_doc_type TEXT NOT NULL,
            source_doc_id TEXT NOT NULL,
            snippet TEXT,
            target_doc_type TEXT NOT NULL,
            target_doc_id TEXT NOT NULL,
            target_name TEXT,
            confidence REAL
        )''')
        try:
            rows = enr_conn.execute(
                "SELECT source_doc_type, source_doc_id, snippet, "
                "target_doc_type, target_doc_id, target_name, confidence "
                "FROM span_link WHERE he_id = ?",
                (he_id,)
            ).fetchall()
            c.executemany("INSERT INTO span_link VALUES (?,?,?,?,?,?,?)", rows)
            link_count = len(rows)
        except sqlite3.OperationalError:
            pass

    # Edge previews for cross-document hover tooltips
    c.execute("DROP TABLE IF EXISTS edge_previews")
    c.execute('''CREATE TABLE edge_previews (
        target_type TEXT NOT NULL,
        target_id   TEXT NOT NULL,
        title       TEXT,
        summary     TEXT,
        affordances TEXT,
        url         TEXT,
        PRIMARY KEY (target_type, target_id)
    )''')

    preview_count = 0

    # 1. Statute previews from causal_map.db
    if CAUSAL_MAP_DB.exists():
        try:
            cm_conn = sqlite3.connect(str(CAUSAL_MAP_DB))
            cm_conn.row_factory = sqlite3.Row
            # Collect unique statute ids from law_refs + diffs
            statute_ids = set()
            for row in c.execute("SELECT DISTINCT target_statute FROM law_refs"):
                statute_ids.add(row[0])
            for row in c.execute("SELECT DISTINCT statute_id FROM diffs"):
                statute_ids.add(row[0])
            for sid in statute_ids:
                row = cm_conn.execute(
                    "SELECT id, title, date_issued FROM statutes WHERE id = ?", (sid,)
                ).fetchone()
                if row:
                    year_num = sid.split('/') if '/' in sid else [None, None]
                    url = f"/mev/lakikartta#detail/{year_num[1]}/{year_num[0]}" if len(year_num) == 2 else f"/mev/lakikartta#detail/{sid}"
                    c.execute(
                        "INSERT OR REPLACE INTO edge_previews VALUES (?,?,?,?,?,?)",
                        ('statute', sid, row['title'], None, None, url)
                    )
                    preview_count += 1
            cm_conn.close()
        except Exception as e:
            print(f"  WARN: causal_map lookup failed: {e}")

    # 2. Expert previews (for V-span links within the HE)
    expert_rows = c.execute(
        "SELECT statement_id, expert_name, content FROM expert_statement"
    ).fetchall()
    for row in expert_rows:
        sid = row[0]
        name = row[1] or sid
        content = row[2] or ''
        # First 200 chars of non-empty plain text
        summary = content[:200].strip() if content else None
        if summary and len(content) > 200:
            summary += '…'
        # Tag counts as affordances
        tag_rows = c.execute(
            "SELECT role, COUNT(*) as n FROM lausunto_tag WHERE statement_id = ? GROUP BY role",
            (sid,)
        ).fetchall()
        affordances = None
        if tag_rows:
            aff = {r[0]: r[1] for r in tag_rows}
            affordances = json.dumps(aff, ensure_ascii=False)
        c.execute(
            "INSERT OR REPLACE INTO edge_previews VALUES (?,?,?,?,?,?)",
            ('expert', sid, name, summary, affordances, f'#expert-{sid}')
        )
        preview_count += 1

    # 3. Committee report previews
    report_rows = c.execute(
        "SELECT tunnus, committee FROM committee_report"
    ).fetchall()
    for row in report_rows:
        tunnus = row[0]
        if not tunnus:
            continue
        committee = row[1] or ''
        title = f"{tunnus} — {committee}" if committee else tunnus
        anchor_id = tunnus.replace(' ', '_')
        c.execute(
            "INSERT OR REPLACE INTO edge_previews VALUES (?,?,?,?,?,?)",
            ('committee', tunnus, title, None, None, f'#report-{anchor_id}')
        )
        preview_count += 1

    conn.commit()
    conn.close()

    parts = [f"{expert_count} experts", f"{report_count} reports"]
    if scr_data:
        parts.append(f"scrutiny rate {scr_data.get('scrutiny_ignored_rate', 0):.0%}")
    if claim_count:
        parts.append(f"{claim_count} claims")
    if unreason_count:
        parts.append(f"{unreason_count} unreason")
    if tag_count:
        parts.append(f"{tag_count} tags")
    if irony_count:
        parts.append(f"{irony_count} irony")
    if ptk_count:
        parts.append(f"{ptk_count} ptk_speeches")
    if ev_count:
        parts.append(f"{ev_count} ev")
    if diss_count:
        parts.append(f"{diss_count} dissents")
    if lau_tag_count:
        parts.append(f"{lau_tag_count} lausunto_tag")
    if miet_tag_count:
        parts.append(f"{miet_tag_count} mietinto_tag")
    if ptk_tag_count:
        parts.append(f"{ptk_tag_count} ptk_speech_tag")
    if disc_node_count:
        parts.append(f"{disc_node_count} disc_nodes")
    if disc_edge_count:
        parts.append(f"{disc_edge_count} disc_edges")
    if span_count:
        parts.append(f"{span_count} spans")
    if link_count:
        parts.append(f"{link_count} links")
    if preview_count:
        parts.append(f"{preview_count} previews")
    print(f"  {he_id}: {', '.join(parts)}")


def main():
    parser = argparse.ArgumentParser(description='Enrich per-HE databases with expert + scrutiny data')
    parser.add_argument('he_id', nargs='?', help='HE canonical ID (e.g., he-38-2025)')
    parser.add_argument('--all', action='store_true', help='Enrich all HE databases')
    args = parser.parse_args()

    if not INDEX_DB.exists():
        print(f"Error: Index DB not found at {INDEX_DB}")
        sys.exit(1)

    index_conn = sqlite3.connect(str(INDEX_DB))
    index_conn.row_factory = sqlite3.Row

    enr_conn = None
    if ENRICHMENTS_DB.exists():
        enr_conn = sqlite3.connect(str(ENRICHMENTS_DB))
        enr_conn.row_factory = sqlite3.Row
        n_claims = 0
        n_unreason = 0
        try:
            n_claims = enr_conn.execute("SELECT COUNT(*) FROM claim").fetchone()[0]
        except sqlite3.OperationalError:
            pass
        try:
            n_unreason = enr_conn.execute("SELECT COUNT(*) FROM unreason_flag").fetchone()[0]
        except sqlite3.OperationalError:
            pass
        print(f"Loaded enrichments DB: {n_claims} claims, {n_unreason} unreason")
    else:
        print(f"Warning: {ENRICHMENTS_DB} not found, skipping enrichments")

    scrutiny = load_scrutiny()
    print(f"Loaded scrutiny data for {len(scrutiny)} HEs")

    if args.all:
        he_ids = sorted(
            p.stem for p in HE_DB_DIR.glob("he-*.db")
        )
    elif args.he_id:
        he_ids = [args.he_id]
    else:
        parser.print_help()
        sys.exit(1)

    for he_id in he_ids:
        enrich_he(he_id, index_conn, scrutiny, enr_conn)

    index_conn.close()
    if enr_conn:
        enr_conn.close()
    print(f"\nDone: enriched {len(he_ids)} databases")


if __name__ == '__main__':
    main()

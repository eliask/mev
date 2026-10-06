"""Resolve V-type (reference) span_tag to actual documents within the same HE.

For each span tagged 'reference' in he_enrichments.db, extract the candidate
entity name from the snippet, then fuzzy-match against:
  - expert_statement.expert_name (lausunto)
  - committee_report.committee (mietinto/lausunto committee reports)
  - ptk_speeches.speaker (plenary speeches)

Resolved links are written to span_link in he_enrichments.db.

Match strategy (no LLM):
  1. Strip Finnish genitive/case suffixes from candidate tokens
  2. Substring match (case-insensitive) against target names → confidence 0.9
  3. Token overlap match (≥1 significant token shared) → confidence 0.7

Reads:
    .tmp/he_enrichments.db  [span_tag WHERE tag='reference']
    .tmp/he_dbs/he-*.db     [expert_statement, committee_report, ptk_speeches]

Writes to:
    .tmp/he_enrichments.db  [span_link table]

Usage:
    python -m mev.pipeline.resolve_refs he-1-2025
    python -m mev.pipeline.resolve_refs --all
"""
from __future__ import annotations

import argparse
import re
import sqlite3
import sys
from pathlib import Path

from mev.config import HE_DB_DIR, ENRICHMENTS_DB

# ---------------------------------------------------------------------------
# Finnish case suffix stripping
# ---------------------------------------------------------------------------

# Suffixes ordered longest-first so greedier matches win.
# Covers genitive (-n), allative (-lle), elative (-sta/-stä), inessive (-ssa/-ssä),
# illative (-on/-ön/-an/-ään/-een/-iin/-uun/-yyn etc.), partitive (-a/-ä),
# ablative (-lta/-ltä), adessive (-lla/-llä), translative (-ksi),
# comitative (-ne-), possessive suffix stubs (-nsa/-nsä/-ään/-aan),
# and the -ry / -oy / -oy organisational suffixes handled separately.
_SUFFIXES = [
    # Long suffixes first
    r'llensa', r'lleen', r'ltansa', r'ltänsä',
    r'stansa', r'stänsä', r'ssansa', r'ssänsä',
    r'kseen', r'nteen', r'ntään',
    r'tten', r'iden', r'itten', r'ille', r'ilta', r'iltä', r'illa', r'illä',
    r'ista', r'istä', r'issa', r'issä', r'ihin', r'ihan',
    r'naan', r'nään', r'neen', r'niin', r'noon', r'nuun', r'nään', r'nyyn',
    r'ään', r'aan', r'oon', r'een', r'iin', r'uun', r'yyn',
    r'lle', r'lta', r'ltä', r'lla', r'llä',
    r'sta', r'stä', r'ssa', r'ssä',
    r'ksi', r'han', r'hin', r'hun', r'hän', r'hön',
    r'kin', r'kaan', r'kään',
    r'jen', r'ten',
    r'an', r'en', r'in', r'on', r'un', r'yn', r'ön',
    r'ia', r'iä',
    r'ta', r'tä',
    r'na', r'nä',
    r'a', r'ä', r'n',
]

_SUFFIX_RE = re.compile(
    r'^(.{3,})(?:' + '|'.join(_SUFFIXES) + r')$',
    re.IGNORECASE,
)

# Tokens shorter than this are noise for matching purposes
_MIN_TOKEN_LEN = 4

# Tokens that are common Finnish words — not useful as entity discriminators
_STOPWORDS = frozenset({
    'lausuntoon', 'lausunnon', 'lausunto', 'lausunnossa', 'lausunnosta',
    'lausunnossaan', 'lausunnossansa', 'lausuntoaan', 'lausunnostaan',
    'selvityksen', 'selvitykseen', 'selvityksessä', 'selvitys',
    'mukaan', 'mukaisesti', 'mukainen',
    'saaman', 'saamaan', 'saatu', 'saadun', 'saama',
    'asiantuntija', 'asiantuntijalausunto', 'asiantuntijalausunnon',
    'asiantuntijalausuntoa', 'asiantuntijalausunnossa',
    'asiantuntijalausuntoon',
    'kirjallista', 'kirjallisen', 'kirjallinen',
    'vastine', 'vastineen', 'vastinetta',
    'viittaus', 'viittauksen', 'viittasin',
    'kanslian', 'kanslia',
    'ministeri', 'ministeriön', 'ministeriö',
    'professori', 'professorin', 'dosentti',
    'johtaja', 'ylijohtaja', 'pääjohtaja', 'ylitarkastaja', 'tarkastaja',
    'erityisasiantuntija', 'neuvotteleva', 'neuvotteluneuvos', 'neuvos',
    'esittelijä', 'esittelijäneuvos',
    'varatoimitusjohtaja', 'toimitusjohtaja', 'varapuheenjohtaja',
    'puheenjohtaja', 'vastaava',
    'lakimies', 'lakimiehen', 'asianajaja',
    'lausuntoaan', 'lausuntoa',
    'myös', 'sekä', 'että', 'kuin', 'joka', 'jotka', 'jonka',
    'tämä', 'tässä', 'tämän', 'siitä', 'siihen', 'siinä',
    'valiokunta', 'valiokunnan', 'valiokunnalle', 'valiokunnassa',
    'eduskunta', 'eduskunnan', 'hallitus', 'hallituksen',
    'laki', 'lain', 'laissa', 'lakiin', 'lakiehdotus', 'lakiehdotuksiin',
    'hyväksymiin', 'hyväksyttyihin',
    'perustuslaki', 'perustuslain',
    'asetus', 'asetuksen', 'asetuksessa', 'asetukseen',
    'rikoslaki', 'rikoslain',
    'hallinto', 'hallintoon', 'hallinnon',
    'oikeus', 'oikeuden', 'oikeuteen', 'oikeudessa',
    'virasto', 'viraston', 'virastolle',
})


def strip_case_suffix(token: str) -> str:
    """Strip one Finnish case suffix from token. Returns the stem."""
    m = _SUFFIX_RE.match(token)
    if m:
        return m.group(1)
    return token


def extract_candidate_tokens(snippet: str) -> list[str]:
    """Extract meaningful entity tokens from a reference snippet.

    Tokenises, strips case suffixes, filters stopwords and short tokens.
    Returns list of candidate stem strings (lowercase).
    """
    # Split on whitespace and punctuation
    raw_tokens = re.split(r'[\s,;:()\[\]/\-–—"\']+', snippet)
    stems = []
    for tok in raw_tokens:
        # Remove trailing punctuation
        tok = tok.strip('.,;:!?"\'-–—')
        if len(tok) < _MIN_TOKEN_LEN:
            continue
        lower = tok.lower()
        if lower in _STOPWORDS:
            continue
        stem = strip_case_suffix(lower)
        # After suffix strip, check length again
        if len(stem) < _MIN_TOKEN_LEN:
            continue
        if stem in _STOPWORDS:
            continue
        stems.append(stem)
    return stems


# ---------------------------------------------------------------------------
# Candidate name extraction (higher-level patterns)
# ---------------------------------------------------------------------------

# Patterns for extracting the referenced entity directly from snippet structure.
# Each pattern returns group(1) as the raw entity name fragment.
_ENTITY_PATTERNS = [
    # "Hallintovaliokunnan saaman selvityksen" → "Hallintovaliokunta"
    re.compile(r'^([A-ZÄÖÅ][a-zäöå]+(?:valiokunta|komitea|toimikunta)\w*)', re.IGNORECASE),
    # "Metsästäjäliiton lausuntoon" → "Metsästäjäliitto"
    re.compile(r'^([A-ZÄÖÅ][a-zäöå]{3,}(?:liitto|yhdistys|seura|järjestö|virasto|laitos|toimisto|osasto)\w*)', re.IGNORECASE),
    # "Sako Oy:n lausunnossa" → "Sako Oy"
    re.compile(r'^((?:[A-ZÄÖÅ][a-zäöå]+\s+){0,3}(?:Oy|Oyj|ry| rf|rf|rr|ry|rs|ky|ay|ab)\b)', re.IGNORECASE),
    # "ministeri Rantasen" / "professori Mäkelän" → take word after title
    re.compile(
        r'(?:ministeri|professori|dosentti|johtaja|tarkastaja|asiantuntija|'
        r'neuvos|lakimies|asianajaja|varapuheenjohtaja|puheenjohtaja|'
        r'toimitusjohtaja|ylitarkastaja|esittelijä)\s+([A-ZÄÖÅ][a-zäöå][\w]+)',
        re.IGNORECASE,
    ),
    # "THL:n" / "STM:n" → acronym
    re.compile(r'\b([A-ZÄÖÅ]{2,6})(?::n|:lle|:stä|:ssä|:sta|:ssa|:oon|:lle|:n)?\b'),
    # Organisation name in nominative: capitalised multi-word before colon/comma
    re.compile(r'^([A-ZÄÖÅ][a-zA-ZäöåÄÖÅ\s]{5,40})(?::|,|\s+(?:mukaan|lausunnossa|selvityksen))'),
]


def extract_entity_name(snippet: str) -> str | None:
    """Try to extract the most likely entity name from a reference snippet.

    Returns a cleaned string suitable for fuzzy-matching, or None if nothing
    identifiable was found.
    """
    snippet = snippet.strip()

    for pat in _ENTITY_PATTERNS:
        m = pat.search(snippet)
        if m:
            candidate = m.group(1).strip()
            # Strip case suffix from the last word
            words = candidate.split()
            if words:
                words[-1] = strip_case_suffix(words[-1])
            candidate = ' '.join(words)
            if len(candidate) >= 3:
                return candidate

    return None


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def _normalise(s: str) -> str:
    """Lowercase and collapse whitespace."""
    return re.sub(r'\s+', ' ', s.lower()).strip()


def match_substring(candidate: str, target_name: str) -> float:
    """Exact or substring match. Returns confidence or 0.0."""
    if not candidate or not target_name:
        return 0.0
    c = _normalise(candidate)
    t = _normalise(target_name)
    if c == t:
        return 0.95
    if c in t or t in c:
        return 0.9
    return 0.0


def _lcp_ratio(a: str, b: str) -> float:
    """Longest common prefix length as fraction of the shorter string."""
    shorter = min(len(a), len(b))
    if shorter == 0:
        return 0.0
    i = 0
    while i < shorter and a[i] == b[i]:
        i += 1
    return i / shorter


def match_snippet_against_target(snippet: str, target_name: str) -> float:
    """Match raw snippet tokens (pre-strip) against target name.

    Handles Finnish morphology:
    - Genitive -n: "Metsästäjäliiton" → "Metsästäjäliitto"
    - Consonant gradation: "valiokunnan" (kk→k, tt→t, etc.) vs "valiokunta"
    - Compound words: token may be prefix/suffix of compound in target

    Strategy:
    1. Exact token match after normalisation
    2. Suffix-stripped token match
    3. Long-common-prefix ≥0.85 ratio (handles consonant gradation)
    4. One token is prefix of the other (compound words, ≥6 chars)
    """
    if not snippet or not target_name:
        return 0.0

    norm_target = _normalise(target_name)
    norm_snippet = _normalise(snippet)

    snip_tokens = [
        t for t in re.split(r'[\s,;:()\[\]/\-–—"\']+', norm_snippet)
        if len(t) >= _MIN_TOKEN_LEN and t not in _STOPWORDS
    ]
    tgt_tokens = [
        t for t in re.split(r'[\s,;:()\[\]/\-–—"\']+', norm_target)
        if len(t) >= _MIN_TOKEN_LEN
    ]

    if not snip_tokens or not tgt_tokens:
        return 0.0

    # Pre-compute stripped versions
    snip_stripped = [strip_case_suffix(t) for t in snip_tokens]
    tgt_stripped = [strip_case_suffix(t) for t in tgt_tokens]

    for st, st_s in zip(snip_tokens, snip_stripped):
        for tt, tt_s in zip(tgt_tokens, tgt_stripped):
            # Exact match (raw or stripped)
            if st == tt or st == tt_s or st_s == tt or st_s == tt_s:
                return 0.85
            # LCP ratio ≥ 0.85: handles consonant gradation
            # e.g. "hallintovaliokunna" vs "hallintovaliokunta" → ratio ~0.93
            for a, b in [(st, tt), (st_s, tt), (st, tt_s), (st_s, tt_s)]:
                if len(a) >= 6 and len(b) >= 6 and _lcp_ratio(a, b) >= 0.85:
                    return 0.85
            # Prefix match (compound words): one is a prefix of the other
            for a, b in [(st, tt), (st_s, tt), (st, tt_s), (st_s, tt_s),
                         (tt, st), (tt_s, st), (tt, st_s), (tt_s, st_s)]:
                if len(a) >= 6 and b.startswith(a):
                    return 0.85

    return 0.0


def match_token_overlap(candidate_tokens: list[str], target_name: str) -> float:
    """Token overlap: at least one significant token shared → 0.7."""
    if not candidate_tokens or not target_name:
        return 0.0
    target_tokens = set(
        strip_case_suffix(tok)
        for tok in re.split(r'[\s,;:()\[\]/\-–—"\']+', target_name.lower())
        if len(tok) >= _MIN_TOKEN_LEN and tok not in _STOPWORDS
    )
    for tok in candidate_tokens:
        if tok in target_tokens:
            return 0.7
        # Also check if tok is a prefix/substring of a target token (≥5 chars)
        if len(tok) >= 5:
            for tt in target_tokens:
                if tok in tt or tt in tok:
                    return 0.65
    return 0.0


def best_match(
    snippet: str,
    targets: list[dict],
    name_field: str,
    id_field: str,
    doc_type: str,
) -> dict | None:
    """Find best match among targets for a snippet.

    Returns dict with target info + confidence, or None.
    Matching passes (highest confidence wins):
      1. Substring match on extracted entity name → 0.90–0.95
      2. Raw snippet token match (handles Finnish compound genitive) → 0.85
      3. Stemmed token overlap → 0.65–0.70
    """
    entity_name = extract_entity_name(snippet)
    candidate_tokens = extract_candidate_tokens(snippet)

    best: dict | None = None
    best_conf = 0.0

    for t in targets:
        target_name = t.get(name_field) or ''
        if not target_name:
            continue

        conf = 0.0

        # Pass 1: substring match on extracted entity name
        if entity_name:
            conf = match_substring(entity_name, target_name)

        # Pass 2: raw snippet token matching (compound word aware)
        if conf < 0.85:
            conf = max(conf, match_snippet_against_target(snippet, target_name))

        # Pass 3: stemmed token overlap
        if conf < 0.65 and candidate_tokens:
            conf = max(conf, match_token_overlap(candidate_tokens, target_name))

        if conf > best_conf:
            best_conf = conf
            best = {
                'target_doc_type': doc_type,
                'target_doc_id': str(t[id_field]),
                'target_name': target_name,
                'confidence': conf,
            }

    # Only return if confidence meets threshold
    return best if best_conf >= 0.6 else None


# ---------------------------------------------------------------------------
# DB schema
# ---------------------------------------------------------------------------

_CREATE_SPAN_LINKS = """
CREATE TABLE IF NOT EXISTS span_link (
    he_id           TEXT NOT NULL,
    source_doc_type TEXT NOT NULL,
    source_doc_id   TEXT NOT NULL,
    snippet         TEXT NOT NULL,
    target_doc_type TEXT NOT NULL,
    target_doc_id   TEXT NOT NULL,
    target_name     TEXT,
    confidence      REAL
);
CREATE INDEX IF NOT EXISTS idx_spanlinks_he ON span_link(he_id);
CREATE INDEX IF NOT EXISTS idx_spanlinks_src ON span_link(source_doc_type, source_doc_id);
CREATE INDEX IF NOT EXISTS idx_spanlinks_tgt ON span_link(target_doc_type, target_doc_id);
"""


def _ensure_span_link(enr_conn: sqlite3.Connection) -> None:
    enr_conn.executescript(_CREATE_SPAN_LINKS)
    enr_conn.commit()


# ---------------------------------------------------------------------------
# Core resolver
# ---------------------------------------------------------------------------

def resolve_he(he_id: str, enr_conn: sqlite3.Connection, he_conn: sqlite3.Connection) -> int:
    """Resolve V-spans for one HE. Returns count of resolved links."""
    # Load reference spans for this HE
    try:
        spans = enr_conn.execute(
            "SELECT doc_type, doc_id, snippet FROM span_tag "
            "WHERE he_id = ? AND tag = 'reference'",
            (he_id,)
        ).fetchall()
    except sqlite3.OperationalError:
        return 0

    if not spans:
        return 0

    # Load lookup tables from per-HE DB
    experts: list[dict] = []
    try:
        rows = he_conn.execute(
            "SELECT statement_id, expert_name FROM expert_statement"
        ).fetchall()
        for r in rows:
            experts.append({'statement_id': r[0], 'expert_name': r[1] or ''})
    except sqlite3.OperationalError:
        pass

    committees: list[dict] = []
    try:
        rows = he_conn.execute(
            "SELECT tunnus, committee FROM committee_report"
        ).fetchall()
        for r in rows:
            committees.append({'tunnus': r[0], 'committee': r[1] or ''})
    except sqlite3.OperationalError:
        pass

    speakers: list[dict] = []
    try:
        rows = he_conn.execute(
            "SELECT rowid, speaker FROM ptk_speeches WHERE speaker IS NOT NULL"
        ).fetchall()
        for r in rows:
            speakers.append({'rowid': r[0], 'speaker': r[1] or ''})
    except sqlite3.OperationalError:
        pass

    # Clear existing links for this HE
    enr_conn.execute("DELETE FROM span_link WHERE he_id = ?", (he_id,))

    links: list[tuple] = []

    for (doc_type, doc_id, snippet) in spans:
        candidates = []

        # Try experts
        m = best_match(snippet, experts, 'expert_name', 'statement_id', 'lausunto')
        if m:
            candidates.append(m)

        # Try committee reports
        m = best_match(snippet, committees, 'committee', 'tunnus', 'mietinto')
        if m:
            candidates.append(m)

        # Try PTK speakers
        m = best_match(snippet, speakers, 'speaker', 'rowid', 'ptk')
        if m:
            candidates.append(m)

        # Keep only the best match across all target types
        if candidates:
            best = max(candidates, key=lambda x: x['confidence'])
            links.append((
                he_id,
                doc_type,
                doc_id,
                snippet,
                best['target_doc_type'],
                best['target_doc_id'],
                best['target_name'],
                best['confidence'],
            ))

    if links:
        enr_conn.executemany(
            "INSERT INTO span_link VALUES (?,?,?,?,?,?,?,?)",
            links
        )
        enr_conn.commit()

    return len(links)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Resolve V-type reference spans to documents within the same HE'
    )
    parser.add_argument('he_id', nargs='?', help='HE canonical ID (e.g., he-1-2025)')
    parser.add_argument('--all', action='store_true', help='Resolve all HE databases')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Print resolved links with confidence scores')
    args = parser.parse_args()

    if not ENRICHMENTS_DB.exists():
        print(f"Error: enrichments DB not found at {ENRICHMENTS_DB}", file=sys.stderr)
        sys.exit(1)

    if args.all:
        he_ids = sorted(p.stem for p in HE_DB_DIR.glob('he-*.db'))
    elif args.he_id:
        he_id = args.he_id
        if not he_id.startswith('he-'):
            he_id = f'he-{he_id}'
        he_ids = [he_id]
    else:
        parser.print_help()
        sys.exit(1)

    enr_conn = sqlite3.connect(str(ENRICHMENTS_DB))
    _ensure_span_link(enr_conn)

    total_resolved = 0
    total_he = 0

    for he_id in he_ids:
        db_path = HE_DB_DIR / f'{he_id}.db'
        if not db_path.exists():
            continue

        he_conn = sqlite3.connect(str(db_path))
        n = resolve_he(he_id, enr_conn, he_conn)
        he_conn.close()

        if n > 0:
            total_resolved += n
            total_he += 1
            print(f"  {he_id}: {n} links resolved")

            if args.verbose:
                rows = enr_conn.execute(
                    "SELECT source_doc_type, source_doc_id, snippet, "
                    "target_doc_type, target_doc_id, target_name, confidence "
                    "FROM span_link WHERE he_id = ? ORDER BY confidence DESC",
                    (he_id,)
                ).fetchall()
                for r in rows:
                    print(
                        f"    [{r[0]}:{r[1]}] \"{r[2][:60]}\" "
                        f"→ {r[3]}:{r[4]} \"{r[5]}\" "
                        f"(conf={r[6]:.2f})"
                    )
        elif args.verbose:
            print(f"  {he_id}: 0 links resolved")

    enr_conn.close()
    print(f"\nDone: {total_resolved} links resolved across {total_he} HEs")


if __name__ == '__main__':
    main()

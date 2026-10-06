"""Span-level highlighter for all legislative document types.

Tags specific text spans within PTK speeches, lausunnot, mietintö paragraphs,
and HE atoms. Unlike block-level taggers (tag.py, tag_lausunto.py), this
identifies precise sub-sentence phrases that are SIGNAL against a background
of restatement/rhetoric noise.

Tag categories (wire codes for LLM):
  K = Konkreettinen huoli (concrete concern, specific problem identified)
  € = Euromäärä/luku (fiscal figure, statistic, data point)
  V = Viittaus (references expert, lausunto, study, data source)
  ? = Kysymys (question, especially unanswered)
  E = Ehdotus (proposal, amendment suggestion, "olisi syytä harkita")
  T = Tunnustus (admission, caveat, "ei ole voitu arvioida")
  P = Poliittinen (political positioning beyond the bill substance)
  § = Lakiviittaus (specific legal reference, section/article cite)
  R = Retoriikka (empty rhetoric, "tärkeä", "kannatettava", filler)

Everything NOT tagged is restatement/noise — rendered muted in viewer.
R-tagged spans are explicitly muted (distinguished from untagged for analysis).

Reads from: per-HE DBs (ptk_speeches, expert_statement, committee_report, atoms)
Writes to: he_enrichments.db [span_tag table]

Usage:
    mev detect tag-spans --he he-1-2025 --write-db
    mev detect tag-spans --doc-type ptk --write-db     # PTK only, all HEs
    mev detect tag-spans                                # all types, all HEs
"""
from __future__ import annotations

import asyncio
import os
import json
import re
import sqlite3
from pathlib import Path

import aiohttp

from mev.config import ROOT, HE_DB_DIR, ENRICHMENTS_DB
from mev.llm import call_llm_full
from mev.versioning import extractor_version, stale_keys, stamp_version

PARALLEL = int(os.environ.get("LLM_PARALLEL", "4"))

# ---------------------------------------------------------------------------
# Tag vocabulary
# ---------------------------------------------------------------------------

TAG_LABELS = {
    'K': 'concern',
    'V': 'reference',
    '?': 'question',
    'E': 'proposal',
    'T': 'admission',
    'P': 'political',
    'R': 'rhetoric',
    '€': 'fiscal',
    '§': 'law_ref',
}

VALID_TAGS = set(TAG_LABELS.keys())

# Signature/title lines that should never be tagged as concerns or proposals
_SIGNATURE_RE = re.compile(
    r'^(?:Hallituksen jäsen|Puheenjohtaja|Varapuheenjohtaja|'
    r'Toimitusjohtaja|Pääsihteeri|Lakimies|Johtaja|'
    r'Allekirjoitus|Kunnioittavasti)\b',
    re.IGNORECASE
)

# ---------------------------------------------------------------------------
# LLM prompt — all semantic tags including § and €
# ---------------------------------------------------------------------------

SYSTEM = """Merkitse tekstistä VAIN merkitykselliset kohdat (K/V/?/E/T/P/R/€/§).

K = konkreettinen huoli tai ongelma, jonka PUHUJA ITSE nostaa esiin — ei lain sisällön kuvaamista
V = viittaus asiantuntijaan, lausuntoon, tutkimukseen, lähteeseen nimeltä
? = kysymys, erityisesti vastaamaton
E = ehdotus, muutosesitys, "olisi syytä harkita"
T = tunnustus, varauma, rajoitus, "ei ole voitu arvioida", "on vaikea ennakoida"
P = poliittinen kannanotto joka ylittää asian sisällön
R = tyhjä retoriikka, yleinen tuki/vastustus ilman sisältöä
€ = euromäärä, luku, tilasto, konkreettinen datapisteimi
§ = lakiviittaus KOKONAISENA: "rikoslain (39/1889) 46 luvun 1–3 §:ssä", "perustuslain 80 §"

§-viittaus on KOKO ilmaisu alusta loppuun, ei vain "3 §" tai "80 §" erikseen.
"perustuslain 80 §:n 1 momentissa" = yksi § kokonaan, EI kahtena osana.

EI merkitä: lain sisällön selostamista, HE:n tai mietinnön uudelleenkerrontaa, prosessikuvausta.
"Lakia sovellettaisiin..." = uudelleenkerrontaa, EI K.
"Lupaprosessi saattaa ruuhkautua" = K (puhuja nostaa huolen).

Tulosta VAIN:
TAG "lainattu teksti suoraan alkuperäisestä"

Lainaa TARKASTI. Älä tiivistä, muokkaa tai parafrasoi.
0 merkintää jos vain uudelleenkerrontaa. Ei selityksiä."""


# ---------------------------------------------------------------------------
# Mechanical extractors (regex — no LLM needed)
# ---------------------------------------------------------------------------

# EUR amounts: "400 miljoonaa euroa", "1,6–2,0 miljoonaa euroa", "164 euroa", "5 mrd €"
# Kept as mechanical fallback — LLM handles € too but may miss plain numbers
_RE_EUR = re.compile(
    r'(?:\d[\d\s,.]*)(?:–\d[\d\s,.]*)?'
    r'\s*(?:(?:milj(?:oona[an]?|ardii?n?)?|mrd)\w*\s*)?'
    r'euro\w*'
    r'|'
    r'(?:\d[\d\s,.]*)(?:–\d[\d\s,.]*)?'
    r'\s*(?:milj\.|mrd\.?)\s*€',
    re.IGNORECASE
)


def _strip_noise_from_text(content: str, content_html: str) -> str:
    """Remove text that corresponds to noise-block sections in HTML.

    Strategy: extract visible text from non-noise HTML sections,
    return that as the cleaned text for LLM tagging.
    Only active when noise markup has already been applied to the HTML.
    """
    if not content_html or 'noise-block' not in content_html:
        return content

    # Remove noise-block sections from HTML
    clean_html = re.sub(
        r'<details\s+class="noise-block"[^>]*>.*?</details>',
        '', content_html, flags=re.DOTALL
    )
    # Strip remaining HTML tags to get plain text
    clean = re.sub(r'<[^>]+>', ' ', clean_html)
    clean = clean.replace('&nbsp;', ' ').replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
    clean = re.sub(r'\s+', ' ', clean).strip()
    return clean if len(clean) > 50 else content


def extract_mechanical_spans(text: str) -> list[dict]:
    """Extract € amounts mechanically. § references handled by LLM (too varied for regex)."""
    spans = []
    for m in _RE_EUR.finditer(text):
        spans.append({
            'tag': 'fiscal',
            'span_start': m.start(),
            'span_end': m.end(),
            'snippet': m.group(),
        })
    return spans


# ---------------------------------------------------------------------------
# LLM output parsing
# ---------------------------------------------------------------------------

def parse_span_tag(raw: str, source_text: str) -> list[dict]:
    """Parse LLM output into span annotations. Stores snippet text only (no offsets).
    Offsets are resolved at bake time against the actual HTML."""
    results = []
    seen = set()
    for line in raw.split('\n'):
        line = line.strip()
        if not line or len(line) < 4:
            continue
        # Parse: TAG "quoted text"
        m = re.match(r'^([K€V?ETP§R])\s+"(.+?)"?\s*$', line)
        if not m:
            m = re.match(r'^([K€V?ETP§R])\s+(.{10,})', line)
            if not m:
                continue
        tag = m.group(1)
        snippet = m.group(2).strip().rstrip('"')
        if tag not in VALID_TAGS or len(snippet) < 5:
            continue

        # Post-filter: € tags must look like fiscal figures
        if tag == '€':
            # Drop bare phone/ID numbers (digits, spaces, dashes, optional leading +)
            if re.search(r'^\+?\d[\d\s\-]{5,}$', snippet.strip()):
                continue
            # Must contain at least one digit
            if not re.search(r'\d', snippet):
                continue

        # Post-filter: K/E tags must not be signature/title lines
        if tag in ('K', 'E') and _SIGNATURE_RE.match(snippet.strip()):
            continue

        # Verify snippet exists in source (exact or whitespace-normalized)
        verified = _verify_snippet(source_text, snippet)
        if not verified:
            continue

        # Deduplicate
        key = f"{tag}:{verified[:50]}"
        if key in seen:
            continue
        seen.add(key)

        results.append({
            'tag': TAG_LABELS[tag],
            'snippet': verified,
        })

    return results


def _verify_snippet(text: str, snippet: str) -> str | None:
    """Verify snippet exists in text. Returns the actual matched text, or None."""
    if snippet in text:
        return snippet

    # Normalize whitespace and retry
    norm_text = re.sub(r'\s+', ' ', text)
    norm_snip = re.sub(r'\s+', ' ', snippet)
    idx = norm_text.find(norm_snip)
    if idx >= 0:
        return norm_snip  # return normalized form

    # Partial match on first 40 chars
    if len(norm_snip) > 40:
        prefix = norm_snip[:40]
        idx = norm_text.find(prefix)
        if idx >= 0:
            return norm_text[idx:idx + len(norm_snip)]

    return None


# ---------------------------------------------------------------------------
# Document loaders
# ---------------------------------------------------------------------------

def load_documents(he_id: str, doc_types: list[str] | None = None) -> list[dict]:
    """Load all taggable documents from a per-HE DB."""
    db_path = HE_DB_DIR / f"{he_id}.db"
    if not db_path.exists():
        return []

    conn = sqlite3.connect(str(db_path))
    docs = []

    if not doc_types or 'ptk' in doc_types:
        try:
            rows = conn.execute(
                "SELECT rowid, ptk_tunnus, speaker, party, role, text "
                "FROM ptk_speeches WHERE length(text) > 100 ORDER BY rowid"
            ).fetchall()
            for r in rows:
                docs.append({
                    'doc_type': 'ptk',
                    'doc_id': str(r[0]),  # rowid
                    'context': f"{r[2]} ({r[3] or ''}) {r[4] or ''}".strip(),
                    'text': r[5],
                })
        except sqlite3.OperationalError:
            pass

    if not doc_types or 'lausunto' in doc_types:
        try:
            rows = conn.execute(
                "SELECT statement_id, expert_name, content, content_html "
                "FROM expert_statement WHERE length(content) > 200 ORDER BY rowid"
            ).fetchall()
            for r in rows:
                text = _strip_noise_from_text(r[2], r[3])
                docs.append({
                    'doc_type': 'lausunto',
                    'doc_id': r[0],
                    'context': r[1] or '',
                    'text': text,
                })
        except sqlite3.OperationalError:
            pass

    if not doc_types or 'mietinto' in doc_types:
        try:
            rows = conn.execute(
                "SELECT tunnus, committee, content "
                "FROM committee_report WHERE length(content) > 200 ORDER BY rowid"
            ).fetchall()
            for r in rows:
                docs.append({
                    'doc_type': 'mietinto',
                    'doc_id': r[0],
                    'context': f"{r[0]} {r[1] or ''}",
                    'text': r[2],
                })
        except sqlite3.OperationalError:
            pass

    if not doc_types or 'he' in doc_types:
        try:
            rows = conn.execute(
                "SELECT atom_id, atom_type, title, content "
                "FROM atoms WHERE atom_type IN "
                "('IMPACT','CURRENT_STATE','FEEDBACK','ALTERNATIVES','OBJECTIVES','CONSTITUTIONAL') "
                "AND length(content) > 200 ORDER BY rowid"
            ).fetchall()
            for r in rows:
                docs.append({
                    'doc_type': 'he',
                    'doc_id': r[0],
                    'context': f"{r[1]} {r[2] or ''}",
                    'text': r[3],
                })
        except sqlite3.OperationalError:
            pass

    conn.close()
    return docs


# ---------------------------------------------------------------------------
# LLM tagging
# ---------------------------------------------------------------------------

WINDOW = 2000  # chars per LLM call


async def tag_document(session, sem, doc: dict) -> dict:
    """Tag one document's text spans."""
    text = doc['text']
    if not text or len(text) < 50:
        return {**doc, 'spans': []}

    all_spans = []
    start = 0

    while start < len(text):
        end = min(start + WINDOW, len(text))
        # Extend to sentence boundary
        if end < len(text):
            for sep in ['. ', '.\n', '.\t']:
                last = text.rfind(sep, start + WINDOW // 2, end + 200)
                if last > 0:
                    end = last + 1
                    break

        chunk = text[start:end]
        max_tok = max(400, len(chunk) // 2)

        async with sem:
            resp = await call_llm_full(session, SYSTEM, chunk, max_tokens=max_tok)

        if resp.get('error'):
            break

        if resp.get('finish_reason') == 'length':
            async with sem:
                resp = await call_llm_full(session, SYSTEM, chunk, max_tokens=max_tok * 2)

        spans = parse_span_tag(resp['content'], chunk)
        all_spans.extend(spans)

        if end >= len(text):
            break
        start = end

    # Merge mechanical spans (€)
    mech_spans = extract_mechanical_spans(text)
    # Only add mechanical spans whose snippet isn't already covered by LLM
    llm_snippets = {s['snippet'] for s in all_spans}
    for ms in mech_spans:
        if not any(ms['snippet'] in ls for ls in llm_snippets):
            all_spans.append({'tag': ms['tag'], 'snippet': ms['snippet']})

    return {**doc, 'spans': all_spans}


# ---------------------------------------------------------------------------
# DB
# ---------------------------------------------------------------------------

def ensure_db_tables(db_path: Path) -> None:
    conn = sqlite3.connect(str(db_path))
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS span_tag (
            he_id       TEXT NOT NULL,
            doc_type    TEXT NOT NULL,
            doc_id      TEXT NOT NULL,
            tag         TEXT NOT NULL,
            snippet     TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_span_he ON span_tag(he_id);
        CREATE INDEX IF NOT EXISTS idx_span_type ON span_tag(doc_type);
        CREATE INDEX IF NOT EXISTS idx_span_tag ON span_tag(tag);
    """)
    conn.close()


def write_results(db_path: Path, he_id: str, results: list[dict]) -> int:
    conn = sqlite3.connect(str(db_path))
    conn.execute("DELETE FROM span_tag WHERE he_id = ?", (he_id,))
    total = 0
    for doc in results:
        rows = [
            (he_id, doc['doc_type'], doc['doc_id'], s['tag'], s['snippet'])
            for s in doc.get('spans', [])
        ]
        conn.executemany(
            "INSERT INTO span_tag VALUES (?,?,?,?,?)", rows)
        total += len(rows)
    conn.commit()
    conn.close()
    return total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

EXTRACTOR_VERSION = extractor_version(SYSTEM, "tag_spans_v1")


async def run(
    he_id: str | None = None,
    doc_types: list[str] | None = None,
    write_db: bool = False,
    force: bool = False,
    parallel: int = PARALLEL,
) -> dict:
    """Programmatic entry point."""
    if he_id:
        if not he_id.startswith('he-'):
            he_id = f'he-{he_id}'
        he_ids = [he_id]
    else:
        he_ids = sorted(p.stem for p in HE_DB_DIR.glob('he-*.db'))

    if not force:
        before = len(he_ids)
        he_ids = stale_keys(ENRICHMENTS_DB, "span_tag", "he_id", he_ids, EXTRACTOR_VERSION)
        skipped = before - len(he_ids)
        if skipped:
            print(f"Skipping {skipped} up-to-date HEs (v={EXTRACTOR_VERSION[:8]})")

    print(f"Processing {len(he_ids)} HEs (doc_types={doc_types or 'all'})")

    sem = asyncio.Semaphore(parallel)
    grand = {'n_he': 0, 'n_docs': 0, 'n_spans': 0}

    if write_db:
        ensure_db_tables(ENRICHMENTS_DB)

    async with aiohttp.ClientSession() as session:
        for hid in he_ids:
            docs = load_documents(hid, doc_types)
            if not docs:
                continue

            results = []
            for doc in docs:
                tagged = await tag_document(session, sem, doc)
                results.append(tagged)

            n_spans = sum(len(r.get('spans', [])) for r in results)
            if n_spans > 0:
                # Print summary by doc_type
                by_type = {}
                for r in results:
                    dt = r['doc_type']
                    ns = len(r.get('spans', []))
                    if ns:
                        by_type[dt] = by_type.get(dt, 0) + ns
                parts = [f"{v} {k}" for k, v in sorted(by_type.items())]
                print(f"  {hid}: {n_spans} spans ({', '.join(parts)})")

                # Print tag distribution
                tag_dist = {}
                for r in results:
                    for s in r.get('spans', []):
                        tag_dist[s['tag']] = tag_dist.get(s['tag'], 0) + 1
                dist_parts = [f"{k}:{v}" for k, v in sorted(tag_dist.items(), key=lambda x: -x[1])]
                print(f"    tags: {' '.join(dist_parts)}")

            if write_db and n_spans > 0:
                n = write_results(ENRICHMENTS_DB, hid, results)
                stamp_version(ENRICHMENTS_DB, "span_tag", "he_id", hid, EXTRACTOR_VERSION)

            grand['n_he'] += 1
            grand['n_docs'] += len(docs)
            grand['n_spans'] += n_spans

    print(f"\nDone: {grand['n_he']} HEs, {grand['n_docs']} docs, {grand['n_spans']} spans")
    return grand


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Span-level highlighter for legislative texts")
    parser.add_argument('--he', help='Specific HE')
    parser.add_argument('--doc-type', action='append', dest='doc_types',
                        choices=['ptk', 'lausunto', 'mietinto', 'he'],
                        help='Limit to doc type (repeatable)')
    parser.add_argument('--write-db', action='store_true')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--parallel', type=int, default=PARALLEL)
    args = parser.parse_args()
    asyncio.run(run(
        he_id=args.he,
        doc_types=args.doc_types,
        write_db=args.write_db,
        force=args.force,
        parallel=args.parallel,
    ))


if __name__ == '__main__':
    main()

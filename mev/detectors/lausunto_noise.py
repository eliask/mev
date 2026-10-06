"""Tag noise paragraphs in lausunto expert_statement using local LLM.

Identifies header/footer/signature boilerplate in expert statements.
Wraps consecutive noise <p> blocks in <details class="noise-block"> directly
in the HTML — stores result back in content_html column.

The viewer renders content_html with CSS styling the
noise-block details/summary as collapsed =====.

Usage:
    python -m mev.detectors.lausunto_noise              # all HEs
    python -m mev.detectors.lausunto_noise --he he-112-2025
    python -m mev.detectors.lausunto_noise --dry-run    # no writes
    python -m mev.detectors.lausunto_noise --rerun      # re-tag already tagged

Via mev CLI:
    mev detect noise-lausunto
    mev detect noise-lausunto --he he-112-2025
    mev detect noise-lausunto --dry-run

FUTURE: HTML normalization pre-pass before noise tagging — two problems:

1. LINE JOINING (PDF column-wrap artifacts)
   PDF-to-text breaks mid-sentence at ~80 chars → each fragment becomes a <p>.
   Heuristic: if <p> doesn't end in [.!?:;] AND next <p> starts lowercase → merge.
   Mechanical, no LLM, high confidence.

2. HEADER PROMOTION
   Section titles like "Tavoitteet", "Tausta", "Nykytila", "Vaikutukset",
   "Perustelut", "Voimaantulo" appear as plain <p> but should be <h3>.
   Detection: short (1-5 words), no trailing punctuation, optionally matched
   against canonical Finnish HE section vocabulary.

Preferred approach: mechanical normalize_lausunto_html() pre-pass (join broken
lines + impute known headers -> <h3>), THEN run noise LLM on cleaner input.
Benefits: LLM sees proper paragraphs, viewer gets <h3> structure regardless of
noise tagging state, no added LLM complexity.
Consider separate `mev detect normalize-lausunto` stage for incremental control.
Store normalized output back to content_html before noise pass.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import html as html_mod
import json
import re
import sqlite3
import sys
from pathlib import Path

import aiohttp

from mev.config import HE_DB_DIR
from mev.llm import call_llm as _call_llm_shared, LLMContextExhausted

PARALLEL = int(os.environ.get("LLM_PARALLEL", "4"))
# Max paragraphs to classify per LLM call (not counting context window)
CLASSIFY_WINDOW = 120  # paragraphs to classify per call
CONTEXT_WINDOW = 20    # preceding paragraphs shown as read-only context at window boundary

SYSTEM = """\
Olet asiantuntija tunnistamaan asiantuntijalausunnoista kappaleet, jotka eivät lisää \
ymmärrystä lausunnon sisältöön. Tunnista kahdenlaista melua:

RAKENNEMELUKAPPALEET (kirjekuori):
- lähettäjäorganisaatio, vastaanottava valiokunta, sähköpostiosoite, asiakirjan otsikko
- PL-osoite, postinumero, Y-tunnus, puhelinnumero, sivunumero, jatkoviite ("YHTIÖ 3)")
- diaarinumero (ÅLR NNN/YYYY, dnr, VN/ -tunniste jne.)
- yhteyshenkilöblokki ("Yhteyshenkilö / nimi")
- HE-viittausrivi lausunnon ALUSSA ("HE NNN/YYYY vp Hallituksen esitys..." otsikkokappal.)
  — EI sisäisiä väliotsikoita, jotka sisältävät HE-numeron osana kappaleen navigointia
- allekirjoitusblokki (nimi, titteli, organisaatio)

SEMANTTISESTI TYHJÄT KAPPALEET — VAIN jos kappale koostuu PELKÄSTÄÄN tästä:
- muodolliset kiitokset ilman substanssia: "kiittää/toivottaa/arvostaa mahdollisuudesta lausua"

JÄTÄ LUOKITTELEMATTA (näitä EI koskaan merkitä meluksi):
- kappale joka sisältää substanssikritiikin, huolen, suosituksen, lain §-viittauksen, \
numeerisen väitteen tai konkreettisen ehdotuksen — vaikka se alkaisi kohteliaisuusfrasilla
- asiakirjan SISÄISET väliotsikot tai section-otsikot — vaikka sisältäisivät HE-numeron tai lain nimen
- "pitää esitystä kannatettavana", "suhtautuu myönteisesti", "haluaa kiinnittää huomion" \
ja vastaavat — voivat olla merkityksellisiä, älä merkitse meluksi

Vastaa: yksi kappaleindeksi tai väli (esim. 5-9) per rivi.
NONE jos yhtään melukappaletta ei löydy.
Ei selityksiä, ei sulkeita, ei muuta tekstiä."""

_P_RE = re.compile(r'(<p[^>]*>)([\s\S]*?)(</p>)', re.IGNORECASE)


def _strip_tags(html: str) -> str:
    return re.sub(r'<[^>]+>', '', html).replace('\n', ' ').strip()


def extract_p_chunks(html: str) -> list[dict]:
    """Split HTML into ordered list of {html, text} for each <p> element."""
    chunks = []
    for m in _P_RE.finditer(html):
        text = _strip_tags(m.group(2))
        chunks.append({'html': m.group(0), 'text': text, 'start': m.start(), 'end': m.end()})
    return chunks


def text_to_html(text: str) -> str:
    """Convert plain text (double-newline paragraphs) to <p> HTML — mirrors JS textToParas."""
    paras = re.split(r'\n\s*\n', text.strip())
    parts = []
    for p in paras:
        p = p.replace('\n', ' ').strip()
        if p:
            parts.append(f'<p>{html_mod.escape(p)}</p>')
    return ''.join(parts)


def build_prompt(chunks: list[dict], offset: int = 0) -> str:
    """Build numbered paragraph prompt. offset is the 0-based index of chunks[0] in the doc."""
    lines = []
    for i, c in enumerate(chunks):
        lines.append(f'[{offset + i + 1}] {c["text"][:200]}')
    return '\n'.join(lines)


def parse_response(response: str, n_chunks: int) -> list[int]:
    """Parse LLM output → 0-based list of noise chunk indices.

    Accepts:
      - single indices:  [5]  or  5
      - ranges:          5-9  or  [5-9]  or  5–9 (en-dash)

    TODO (future): support sub-paragraph span tagging for paragraphs that are
    partially noise (e.g. a paragraph starting with a PDF artifact before actual content).
    Format might be "PARTIAL 5:0-15" meaning chars 0-15 of paragraph 5 are noise.
    This would require a follow-up LLM call for candidate paragraphs to do
    word-level refinement, then wrap only the noise substring with <noise>.
    """
    if not response or response.strip().upper() == 'NONE':
        return []
    noise = []
    for line in response.strip().splitlines():
        line = line.strip()
        # Range: 5-9 or [5-9] or 5–9
        m = re.match(r'\[?(\d+)\]?\s*[-–]\s*\[?(\d+)\]?', line)
        if m:
            lo, hi = int(m.group(1)) - 1, int(m.group(2)) - 1
            for idx in range(max(0, lo), min(n_chunks, hi + 1)):
                noise.append(idx)
            continue
        # Single index
        m = re.match(r'\[?(\d+)\]?', line)
        if m:
            idx = int(m.group(1)) - 1  # 1-based → 0-based
            if 0 <= idx < n_chunks:
                noise.append(idx)
    return sorted(set(noise))


def inject_noise_tags(source_html: str, chunks: list[dict], noise_indices: set[int]) -> str:
    """Wrap consecutive noise <p> chunks in <details class="noise-block"> in the HTML.

    Works by rebuilding the HTML string, grouping consecutive noise chunks together.
    Non-p content between the p tags is carried through unchanged.
    """
    if not noise_indices:
        return source_html

    # Build reconstruction from source_html using chunk positions
    # We split source_html into segments: the <p>...</p> chunks and the gaps between them
    segments = []  # (is_p: bool, html_slice: str, chunk_idx: int or None)
    prev_end = 0
    for i, c in enumerate(chunks):
        gap = source_html[prev_end:c['start']]
        if gap:
            segments.append({'type': 'gap', 'html': gap})
        segments.append({'type': 'p', 'html': c['html'], 'idx': i, 'noise': i in noise_indices})
        prev_end = c['end']
    tail = source_html[prev_end:]
    if tail:
        segments.append({'type': 'gap', 'html': tail})

    # Mark gap segments between two consecutive noise p's as "noise gaps" (whitespace passthrough)
    for i, seg in enumerate(segments):
        if seg['type'] == 'gap' and not seg['html'].strip():
            prev_noise = next_noise = False
            for j in range(i - 1, -1, -1):
                if segments[j]['type'] == 'p':
                    prev_noise = segments[j]['noise']
                    break
                if segments[j]['html'].strip():
                    break
            for j in range(i + 1, len(segments)):
                if segments[j]['type'] == 'p':
                    next_noise = segments[j]['noise']
                    break
                if segments[j]['html'].strip():
                    break
            if prev_noise and next_noise:
                seg['noise'] = True

    # Reconstruct, grouping consecutive noise segments
    result = ''
    i = 0
    while i < len(segments):
        seg = segments[i]
        if seg.get('noise'):
            noise_html = ''
            while i < len(segments) and segments[i].get('noise'):
                noise_html += segments[i]['html']
                i += 1
            result += f'<details class="noise-block"><summary></summary>{noise_html}</details>'
        else:
            result += seg['html']
            i += 1
    return result


async def call_llm(session: aiohttp.ClientSession, sem: asyncio.Semaphore, prompt: str, n_paras: int = 50) -> str:
    """Thin wrapper: acquires sem then delegates to shared call_llm (cached). Returns str."""
    max_tokens = 20 + n_paras * 5
    try:
        async with sem:
            return await _call_llm_shared(session, SYSTEM, prompt, max_tokens=max_tokens)
    except LLMContextExhausted:
        async with sem:
            return await _call_llm_shared(session, SYSTEM, prompt, max_tokens=max_tokens * 2)


async def tag_statement(
    session: aiohttp.ClientSession, sem: asyncio.Semaphore, stmt: dict, dry_run: bool
) -> dict:
    """Tag one expert statement. Returns {statement_id, tagged_html, noise_indices, n_paras}."""
    raw_html = stmt.get('content_html') or ''
    raw_text = stmt.get('content') or ''

    # Strip existing noise blocks if present (supports --rerun)
    if raw_html and 'noise-block' in raw_html:
        raw_html = strip_existing_noise_blocks(raw_html)
    working_html = raw_html if raw_html else text_to_html(raw_text)
    if not working_html:
        return {'statement_id': stmt['statement_id'], 'tagged_html': None, 'noise_indices': [], 'n_paras': 0}

    chunks = extract_p_chunks(working_html)
    if not chunks:
        return {'statement_id': stmt['statement_id'], 'tagged_html': working_html, 'noise_indices': [], 'n_paras': 0}

    # Window through the document. Each window overlaps with the previous by
    # CONTEXT_WINDOW paragraphs so boundary noise is seen in both calls.
    # Duplicates from overlap are harmless — merged via set().
    all_noise: list[int] = []
    start = 0
    n = len(chunks)
    step = CLASSIFY_WINDOW - CONTEXT_WINDOW  # non-overlapping advance per call
    response = ''

    while start < n:
        window_end = min(start + CLASSIFY_WINDOW, n)
        window = chunks[start:window_end]
        prompt = build_prompt(window, offset=start)
        response = await call_llm(session, sem, prompt, n_paras=len(window))
        all_noise.extend(parse_response(response, n))
        if window_end == n:
            break
        start += step

    noise_indices = sorted(set(all_noise))

    tagged_html = inject_noise_tags(working_html, chunks, set(noise_indices))

    return {
        'statement_id': stmt['statement_id'],
        'tagged_html': tagged_html,
        'noise_indices': noise_indices,
        'n_paras': len(chunks),
        'llm_response': response if isinstance(response, str) else '',
    }


_NOISE_BLOCK_RE = re.compile(r'<details class="noise-block">.*?</details>', re.DOTALL)


def strip_existing_noise_blocks(html: str) -> str:
    """Unwrap <details class="noise-block"> — restores original <p> elements."""
    def unwrap(m: re.Match) -> str:
        # Keep the <p> content, discard <details><summary></summary>
        inner = re.sub(r'<details[^>]*>|</details>|<summary[^>]*>.*?</summary>', '', m.group(0), flags=re.DOTALL)
        return inner
    return _NOISE_BLOCK_RE.sub(unwrap, html)


def load_statements(con: sqlite3.Connection, rerun: bool) -> list[dict]:
    try:
        rows = con.execute('SELECT statement_id, content, content_html FROM expert_statement').fetchall()
    except Exception:
        return []  # table doesn't exist yet (enrich not run)
    result = []
    for r in rows:
        html = r[2] or ''
        already_tagged = 'noise-block' in html
        if already_tagged and not rerun:
            continue
        result.append({'statement_id': r[0], 'content': r[1], 'content_html': r[2]})
    return result


def write_results(con: sqlite3.Connection, results: list[dict]) -> None:
    for r in results:
        if r.get('tagged_html') is not None:
            con.execute(
                'UPDATE expert_statement SET content_html = ? WHERE statement_id = ?',
                (r['tagged_html'], r['statement_id'])
            )
    con.commit()


async def process_he_db(
    db_path: Path, args: argparse.Namespace,
    session: aiohttp.ClientSession, sem: asyncio.Semaphore
) -> tuple[int, int]:
    con = sqlite3.connect(str(db_path))
    stmts = load_statements(con, args.rerun)
    if not stmts:
        con.close()
        return 0, 0

    he_id = db_path.stem
    print(f'{he_id}: {len(stmts)} statements', flush=True)

    tasks = [tag_statement(session, sem, s, args.dry_run) for s in stmts]
    raw_results = await asyncio.gather(*tasks, return_exceptions=True)
    results = [r for r in raw_results if not isinstance(r, Exception)]

    if args.verbose or args.dry_run:
        for r in results:
            noise = r['noise_indices']
            if noise or args.verbose:
                print(f'  {r["statement_id"]}: {r["n_paras"]} paras, noise={noise}')
                if args.dry_run and r.get('llm_response'):
                    print(f'    LLM → {repr(r["llm_response"][:120])}')

    if not args.dry_run:
        write_results(con, results)

    tagged = sum(1 for r in results if r['noise_indices'])
    con.close()
    return len(stmts), tagged


async def _main_async(args: argparse.Namespace) -> None:
    if args.he:
        he_id = args.he if args.he.startswith('he-') else f'he-{args.he}'
        db_paths = [HE_DB_DIR / f'{he_id}.db']
    else:
        db_paths = sorted(HE_DB_DIR.glob('he-*.db'))

    missing = [p for p in db_paths if not p.exists()]
    if missing:
        print(f'Not found: {[str(p) for p in missing]}', file=sys.stderr)
        sys.exit(1)

    sem = asyncio.Semaphore(args.parallel)
    total_stmts = total_tagged = 0

    async with aiohttp.ClientSession() as session:
        for db_path in db_paths:
            n, tagged = await process_he_db(db_path, args, session, sem)
            total_stmts += n
            total_tagged += tagged

    print(f'\nDone: {total_stmts} statements processed, {total_tagged} with noise tags')


def main() -> None:
    parser = argparse.ArgumentParser(description='Tag lausunto noise paragraphs via LLM')
    parser.add_argument('--he', help='Specific HE DB (e.g. he-112-2025 or 112-2025)')
    parser.add_argument('--dry-run', action='store_true', help='Print results, no DB writes')
    parser.add_argument('--rerun', action='store_true', help='Re-tag already tagged statements')
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--parallel', type=int, default=PARALLEL)
    args = parser.parse_args()
    asyncio.run(_main_async(args))


if __name__ == '__main__':
    main()

"""Sentence-level claim/position tagger for expert lausunto statements.

Tags each sentence with three dimensions:
  Role:    P(position) F(factual claim) C(concern) A(amendment) R(reference)
  Quality: G(grounded) M(modeled) A(asserted) H(hedged) T(uncertain)
  Topic:   same 11 capital-stock codes as tag.py (F/W/S/C/I/N/D/Y/J/R/X)

Architecture mirrors tag.py (HE tagger) — same LLM infra, same topic codes,
same output format. Enables cross-document matching: expert F+A claims on topic F
vs HE V+M claims on topic F → contradiction detection.

Reads from: per-HE DBs (.tmp/he_dbs/<he_id>.db) [expert_statement table]
Strips: existing noise-block tags (from lausunto_noise.py)
Writes to: he_enrichments.db [lausunto_tag table]

Usage:
    mev detect tag-lausunto                     # all HEs with expert statements
    mev detect tag-lausunto --he he-241-2020    # specific HE
    mev detect tag-lausunto --write-db          # persist to he_enrichments.db
    mev detect tag-lausunto --dry-run           # print results only
"""
from __future__ import annotations

import argparse
import asyncio
import os
import json
import re
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import aiohttp

from mev.config import ROOT, HE_DB_DIR, ENRICHMENTS_DB
from mev.versioning import extractor_version, stale_keys, stamp_version
from mev.detectors.tagger_engine import (
    TagConfig,
    call_llm_windowed,
    parse_tag_lines,
    tag_units_windowed,
)

# Reuse tag.py's EUR extractor and topic hints
from mev.detectors.tag import extract_eur, guess_topic_regex, RE_HEDGE, RE_CAVEAT

PARALLEL = int(os.environ.get("LLM_PARALLEL", "4"))

# Same topic codes as tag.py — enables cross-document matching
ROLE_LABELS = {
    'P': 'position',    # explicit stance (support/oppose/conditional)
    'F': 'fact_claim',  # asserts a fact about reality (cost, impact, count)
    'C': 'concern',     # identifies risk, problem, unintended consequence
    'A': 'amendment',   # proposes specific change to law text
    'R': 'reference',   # cites study, law, international comparison, data
    'D': 'describes',   # restates/summarizes HE content without adding substance
}
QUALITY_LABELS = {
    'G': 'grounded', 'M': 'modeled', 'A': 'asserted',
    'H': 'hedged', 'T': 'uncertain',
}
TOPIC_LABELS = {
    'F': 'fiscal', 'W': 'epistemic', 'S': 'social', 'C': 'cognitive',
    'I': 'institutional', 'N': 'infrastructure', 'D': 'human',
    'Y': 'coherence', 'J': 'purpose', 'R': 'moral', 'X': 'other',
}

VALID_ROLES = set(ROLE_LABELS.keys())
VALID_QUALS = set(QUALITY_LABELS.keys())
VALID_TOPICS = set(TOPIC_LABELS.keys())

SYSTEM = """Luokittele asiantuntijalausunnon lauseet kolmella ulottuvuudella.

ROOLI (P/F/C/A/R/D):
  P = kannanotto esitykseen (kannattaa/vastustaa/ehdollinen)
  F = tosiasiallinen väite LAUSUNNONANTAJAN omasta tiedosta (kustannus, vaikutus, lukumäärä)
  C = huoli, riski, ei-toivottu seuraus
  A = konkreettinen muutosehdotus lakitekstiin ("ehdotamme", "esitämme", "tulisi muuttaa")
  R = viittaus tutkimukseen, lakiin, kansainväliseen vertailuun, dataan
  D = kuvaa/referoi HE:n sisältöä tai ehdotusta — EI lisää omaa substanssia

TÄRKEÄÄ: D-luokka erottaa HE:n referoinnin omista väitteistä.
  "Esityksessä ehdotetaan..." = D (referoi)
  "Kustannusarvio on mielestämme aliarvioitu" = F (oma väite)
  "Lakia sovellettaisiin hylsyihin..." = D (referoi ehdotettua soveltamisalaa)
  "Luvanvaraisuus aiheuttaisi yrityksille lisäkustannuksia" = F (oma arvio)

LAATU (G/M/A/H/T):
  G = grounded, viittaa dataan tai lähteeseen
  M = mallinnettu, parametrit näkyvissä
  A = assertoitu ilman perustelua
  H = hedged ("arvioidaan", "noin", "todennäköisesti")
  T = tunnustettu epävarma

AIHE (F/W/S/C/I/N/D/Y/J/R/X):
  F = fiskaalinen (kustannus, budjetti, lupa*maksu*, hallinnollinen taakka, liikevaihto, kannattavuus)
  W = episteeminen  S = sosiaalinen
  C = kognitiivinen (osaaminen, asiantuntemus, tietojärjestelmä)
  I = institutionaalinen (viranomainen, organisaatio, menettely, lupa*prosessi*)
  N = infrastruktuuri  D = inhimillinen  Y = yhtenäisyys  J = tarkoitus
  R = moraalinen  X = muu

Tulosta VAIN rivit: NUMERO ROOLI LAATU AIHE
Ei selityksiä, ei sulkeita, ei muuta tekstiä.
Jos lause ohitetaan, ÄLÄ tulosta sille riviä.

Esimerkki — syöte:
[1] Pidämme esitystä lähtökohtaisesti kannatettavana.
[2] Esityksessä ehdotetaan säädettäväksi uusi laki viennin luvanvaraisuudesta.
[3] Kustannusarvio 150 miljoonaa euroa on mielestämme aliarvioitu.
[4] OECD:n raportin (2023) mukaan vastaava uudistus maksoi Ruotsissa 2,3 miljardia.
[5] Ehdotamme 12 §:n 2 momenttiin lisättäväksi siirtymäaikaa koskevaa säännöstä.
[6] Uudistus saattaa heikentää pienten kuntien palvelukykyä.
[7] Luvan hakeminen aiheuttaisi yrityksille lisäkustannuksia ja toimitusaikataulujen pidentymistä.

Esimerkki — tuloste:
1 P A I
3 F A F
4 R G F
5 A A I
6 C H S
7 F A F"""


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

_NOISE_BLOCK_RE = re.compile(r'<details class="noise-block">.*?</details>', re.DOTALL)
_TAG_RE = re.compile(r'<[^>]+>')


def _strip_html(html: str) -> str:
    clean = _TAG_RE.sub(' ', html)
    clean = clean.replace('&nbsp;', ' ').replace('&amp;', '&')
    clean = clean.replace('&lt;', '<').replace('&gt;', '>')
    return re.sub(r'\s+', ' ', clean).strip()


def extract_sentences(content: str, content_html: str | None) -> list[tuple[str, str]]:
    """Extract sentences from expert statement. Returns [(text, kind)].

    Uses content_html if available (noise-block stripped), falls back to content.
    """
    text = content
    if content_html:
        # Strip noise blocks, then extract text
        cleaned = _NOISE_BLOCK_RE.sub('', content_html)
        text = _strip_html(cleaned)

    results = []
    for line in text.split('\n'):
        line = line.strip()
        if not line or len(line) < 15:
            continue
        # Split on sentence boundaries
        parts = re.split(r'(?<=[.!?])\s+(?=[A-ZÄÖÅ0-9(])', line)
        for p in parts:
            p = p.strip()
            if len(p) > 15:
                results.append((p, 'prose'))
    return results


# ---------------------------------------------------------------------------
# Per-statement processing
# ---------------------------------------------------------------------------

WINDOW = 100   # sentences per LLM call
OVERLAP = 15   # context overlap

_FISCAL_RE = re.compile(
    r'kustannu|euroa|milj\.|budjett|lupamaksu|hallinnollis\w+ taakk|'
    r'liikevaih|kannattavuu|säästö|menot\b|tulot\b|rahoitu|vero',
    re.IGNORECASE
)


async def tag_statement(
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    stmt: dict,
) -> dict:
    """Tag one expert statement. Returns result dict."""
    sents = extract_sentences(stmt.get('content', ''), stmt.get('content_html'))
    if not sents:
        return {'statement_id': stmt['statement_id'], 'expert_name': stmt.get('expert_name', ''),
                'n_sents': 0, 'n_tagged': 0, 'tags': {}, 'sents': sents,
                'tokens_in': 0, 'tokens_out': 0}

    # Build a temporary TagConfig for windowed tagging
    config = TagConfig(
        system_prompt=SYSTEM,
        extractor_salt="tag_lausunto_v2",
        valid_roles=VALID_ROLES,
        valid_quals=VALID_QUALS,
        valid_topics=VALID_TOPICS,
        role_labels=ROLE_LABELS,
        qual_labels=QUALITY_LABELS,
        topic_labels=TOPIC_LABELS,
        target_table="lausunto_tag",
        version_key_col="he_id",
        enrichments_db=ENRICHMENTS_DB,
        window=WINDOW,
        overlap=OVERLAP,
    )

    # Convert sents to units format expected by tag_units_windowed
    units = [(text, kind) for text, kind in sents]
    all_tags = await tag_units_windowed(session, sem, config, units)

    # Regex post-processing
    for sid, (role, qual, topic) in list(all_tags.items()):
        if sid < len(sents):
            text = sents[sid][0]
            if RE_CAVEAT.search(text) and qual != 'T':
                all_tags[sid] = (role, 'T', topic)
            elif RE_HEDGE.search(text) and qual == 'G':
                all_tags[sid] = (role, 'H', topic)
            # Topic override: fiscal keywords should be F, not I
            if topic == 'I' and _FISCAL_RE.search(text):
                all_tags[sid] = (role, qual, 'F')

    # Detect if this is a government/ministry official (not external expert)
    # — unique to lausunto tagger
    name = (stmt.get('expert_name') or '').lower()
    is_government = bool(re.search(
        r'minister|valtioneuvoston|hallitus', name
    ))

    return {
        'statement_id': stmt['statement_id'],
        'expert_name': stmt.get('expert_name', ''),
        'is_government': is_government,
        'n_sents': len(sents),
        'n_tagged': len(all_tags),
        'tags': all_tags,
        'sents': sents,
        'tokens_in': 0,   # windowed helper doesn't track per-statement tokens
        'tokens_out': 0,
    }


# ---------------------------------------------------------------------------
# DB
# ---------------------------------------------------------------------------

def ensure_db_tables(db_path: Path) -> None:
    conn = sqlite3.connect(str(db_path))
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS lausunto_tag (
            he_id           TEXT NOT NULL,
            statement_id    TEXT NOT NULL,
            expert_name     TEXT,
            is_government   INTEGER DEFAULT 0,
            sent_idx        INTEGER NOT NULL,
            sent_text       TEXT NOT NULL,
            role            TEXT,
            quality         TEXT,
            topic           TEXT,
            eur_amounts     TEXT,
            PRIMARY KEY (he_id, statement_id, sent_idx)
        );
        CREATE INDEX IF NOT EXISTS idx_lt_he ON lausunto_tag(he_id);
        CREATE INDEX IF NOT EXISTS idx_lt_role ON lausunto_tag(role);
        CREATE INDEX IF NOT EXISTS idx_lt_topic ON lausunto_tag(topic);
    """)
    conn.close()


def write_results(db_path: Path, he_id: str, results: list[dict]) -> int:
    conn = sqlite3.connect(str(db_path))
    conn.execute("DELETE FROM lausunto_tag WHERE he_id=?", (he_id,))
    rows = []
    for r in results:
        for i, (text, kind) in enumerate(r['sents']):
            tag = r['tags'].get(i)
            if tag is None:
                continue
            role, qual, topic = tag
            eurs = extract_eur(text)
            rows.append((
                he_id, r['statement_id'], r.get('expert_name', ''),
                1 if r.get('is_government') else 0,
                i, text,
                ROLE_LABELS.get(role, role),
                QUALITY_LABELS.get(qual, qual),
                TOPIC_LABELS.get(topic, topic),
                json.dumps(eurs) if eurs else None,
            ))
    conn.executemany(
        "INSERT OR REPLACE INTO lausunto_tag "
        "(he_id, statement_id, expert_name, is_government, sent_idx, sent_text, role, quality, topic, eur_amounts) "
        "VALUES (?,?,?,?,?,?,?,?,?,?)",
        rows
    )
    conn.commit()
    conn.close()
    return len(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_statements(he_id: str) -> list[dict]:
    db_path = HE_DB_DIR / f"{he_id}.db"
    if not db_path.exists():
        return []
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            "SELECT statement_id, expert_name, content, content_html "
            "FROM expert_statement WHERE length(content) > 100"
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()
    return [
        {'statement_id': r[0], 'expert_name': r[1], 'content': r[2], 'content_html': r[3]}
        for r in rows
    ]


async def process_he(
    session: aiohttp.ClientSession, sem: asyncio.Semaphore,
    he_id: str, write_db: bool, dry_run: bool,
) -> tuple[int, int]:
    stmts = load_statements(he_id)
    if not stmts:
        return 0, 0

    print(f"{he_id}: {len(stmts)} statements", flush=True)
    tasks = [tag_statement(session, sem, s) for s in stmts]
    raw_results = await asyncio.gather(*tasks, return_exceptions=True)

    results = []
    total_tagged = 0
    for r in raw_results:
        if isinstance(r, Exception):
            print(f"  ERROR: {r}")
            continue
        results.append(r)
        total_tagged += r['n_tagged']
        if dry_run or r['n_tagged'] > 0:
            print(f"  {r['statement_id'][:30]:30s} {r['n_sents']:3d} sents → {r['n_tagged']:3d} tagged  "
                  f"{r.get('expert_name', '')[:30]}")

    if write_db and not dry_run:
        ensure_db_tables(ENRICHMENTS_DB)
        n = write_results(ENRICHMENTS_DB, he_id, results)
        stamp_version(ENRICHMENTS_DB, "lausunto_tag", "he_id", he_id, EXTRACTOR_VERSION)
        print(f"  → {n} rows written to lausunto_tag (v={EXTRACTOR_VERSION[:8]})")

    return len(stmts), total_tagged


EXTRACTOR_VERSION = extractor_version(SYSTEM, "tag_lausunto_v2")


async def _main_async(args: argparse.Namespace) -> None:
    if args.he:
        he_id = args.he if args.he.startswith('he-') else f'he-{args.he}'
        he_ids = [he_id]
    else:
        he_ids = sorted(p.stem for p in HE_DB_DIR.glob('he-*.db'))

    if not getattr(args, 'force', False):
        before = len(he_ids)
        he_ids = stale_keys(ENRICHMENTS_DB, "lausunto_tag", "he_id", he_ids, EXTRACTOR_VERSION)
        skipped = before - len(he_ids)
        if skipped:
            print(f"Skipping {skipped} up-to-date HEs (v={EXTRACTOR_VERSION[:8]}, use --force to recompute)")

    sem = asyncio.Semaphore(args.parallel)
    total_stmts = total_tagged = 0

    async with aiohttp.ClientSession() as session:
        for he_id in he_ids:
            n, tagged = await process_he(session, sem, he_id, args.write_db, args.dry_run)
            total_stmts += n
            total_tagged += tagged

    print(f"\nDone: {total_stmts} statements, {total_tagged} tagged sentences")
    return {'n_stmts': total_stmts, 'n_tagged': total_tagged}


async def run(
    he_id: str | None = None,
    write_db: bool = False,
    force: bool = False,
    parallel: int = PARALLEL,
) -> dict:
    """Programmatic entry point. Returns summary dict."""
    import types
    args = types.SimpleNamespace(
        he=he_id,
        write_db=write_db,
        dry_run=False,
        force=force,
        parallel=parallel,
    )
    result = await _main_async(args)
    return result or {}


def main(args: argparse.Namespace | None = None) -> None:
    if args is None:
        parser = argparse.ArgumentParser()
        parser.add_argument('--he', help='Specific HE (e.g. he-241-2020)')
        parser.add_argument('--write-db', action='store_true')
        parser.add_argument('--dry-run', action='store_true')
        parser.add_argument('--force', action='store_true', help='Recompute even if already tagged')
        parser.add_argument('--parallel', type=int, default=PARALLEL)
        args = parser.parse_args()
    asyncio.run(_main_async(args))


if __name__ == '__main__':
    main()

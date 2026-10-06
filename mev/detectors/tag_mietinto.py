"""Paragraph-level tagger for committee mietintö reports.

Tags each paragraph with three dimensions:
  Role:    E(endorses) M(modifies) J(rejects/concerns) N(notes) D(describes)
  Quality: G(grounded) A(asserted) R(references expert) C(cites law)
  Topic:   same 11 capital-stock codes (F/W/S/C/I/N/D/Y/J/R/X)

Reads from: legislative_index.sqlite [committee_report table]
Writes to: he_enrichments.db [mietinto_tag table]

Usage:
    mev detect tag-mietinto --he he-241-2020 --write-db
    mev detect tag-mietinto                              # all with content
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

from mev.config import ROOT, INDEX_DB, ENRICHMENTS_DB
from mev.versioning import extractor_version, stale_keys, stamp_version
from mev.detectors.tagger_engine import (
    TagConfig,
    tag_units_windowed,
)

PARALLEL = int(os.environ.get("LLM_PARALLEL", "4"))

ROLE_LABELS = {
    'E': 'endorses',    # committee supports/endorses HE proposal
    'M': 'modifies',    # committee proposes amendment/modification
    'J': 'rejects',     # committee rejects or expresses concern
    'N': 'notes',       # acknowledges without taking position
    'D': 'describes',   # restates HE content or expert input
}
QUALITY_LABELS = {
    'G': 'grounded',    # cites data or evidence
    'A': 'asserted',    # states without evidence
    'R': 'references_expert', # references expert lausunto
    'C': 'cites_law',   # references statute or legal principle
}
TOPIC_LABELS = {
    'F': 'fiscal', 'W': 'epistemic', 'S': 'social', 'C': 'cognitive',
    'I': 'institutional', 'N': 'infrastructure', 'D': 'human',
    'Y': 'coherence', 'J': 'purpose', 'R': 'moral', 'X': 'other',
}

VALID_ROLES = set(ROLE_LABELS.keys())
VALID_QUALS = set(QUALITY_LABELS.keys())
VALID_TOPICS = set(TOPIC_LABELS.keys())

SYSTEM = """Luokittele valiokunnan mietinnön kappaleet kolmella ulottuvuudella.

ROOLI (E/M/J/N/D):
  E = valiokunta hyväksyy/tukee esitystä
  M = valiokunta ehdottaa muutosta
  J = valiokunta torjuu tai ilmaisee huolen
  N = valiokunta toteaa/huomioi (ei kantaa)
  D = kuvaa HE:n sisältöä tai asiantuntijoiden näkemyksiä

LAATU (G/A/R/C):
  G = perusteltu datalla tai selvityksellä
  A = todettu ilman perustelua
  R = viittaa asiantuntijalausuntoon
  C = viittaa lakiin tai oikeusperiaatteeseen

AIHE (F/W/S/C/I/N/D/Y/J/R/X):
  F=fiskaalinen  W=episteeminen  S=sosiaalinen  C=kognitiivinen
  I=institutionaalinen  N=infrastruktuuri  D=inhimillinen
  Y=yhtenäisyys  J=tarkoitus  R=moraalinen  X=muu

Tulosta VAIN rivit: NUMERO ROOLI LAATU AIHE
Ohita: otsikot, lyhyet prosessitekstikappaleet, lakitekstiesitykset.
Ei selityksiä, ei sulkeita, ei muuta tekstiä.

Esimerkki:
3 E A I
5 D R F
7 J G D
9 M C I"""


# ---------------------------------------------------------------------------
# Text extraction — reuse scrutiny.py's mietintö segmenter
# ---------------------------------------------------------------------------

_TAG_RE = re.compile(r'<[^>]+>')
_PERUSTELUT_START = re.compile(r'VALIOKUNNAN\s+(?:YLEIS)?PERUSTELUT|YLEISPERUSTELUT', re.I)
_PERUSTELUT_END = re.compile(r'YKSITYISKOHTAISET\s+PERUSTELUT|PÄÄTÖSEHDOTUS', re.I)


def _strip_html(html: str) -> str:
    clean = _TAG_RE.sub(' ', html)
    clean = clean.replace('&nbsp;', ' ').replace('&amp;', '&')
    return re.sub(r'\s+', ' ', clean).strip()


def extract_paragraphs(content: str) -> list[tuple[str, str]]:
    """Extract paragraphs from mietintö content. Returns [(text, section)]."""
    if not content:
        return []

    # Try to find PERUSTELUT section (the analytical part)
    paras = []
    section = 'preamble'
    for line in content.split('\n'):
        line = _strip_html(line).strip()
        if not line or len(line) < 20:
            continue
        if _PERUSTELUT_START.search(line):
            section = 'perustelut'
            continue
        if _PERUSTELUT_END.search(line):
            section = 'yksityiskohtaiset'
            continue
        paras.append((line, section))

    return paras


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

WINDOW = 80


async def tag_report(session, sem, report: dict) -> dict:
    """Tag one committee report."""
    paras = extract_paragraphs(report.get('content', ''))
    if not paras:
        return {'report_id': report['report_id'], 'he_id': report['he_id'],
                'n_paras': 0, 'tags': {}, 'paras': paras}

    config = TagConfig(
        system_prompt=SYSTEM,
        extractor_salt="tag_mietinto_v1",
        valid_roles=VALID_ROLES,
        valid_quals=VALID_QUALS,
        valid_topics=VALID_TOPICS,
        role_labels=ROLE_LABELS,
        qual_labels=QUALITY_LABELS,
        topic_labels=TOPIC_LABELS,
        target_table="mietinto_tag",
        version_key_col="he_id",
        enrichments_db=ENRICHMENTS_DB,
        window=WINDOW,
        overlap=10,
    )

    # paras is [(text, section)]; pass as units
    all_tags = await tag_units_windowed(session, sem, config, paras)

    return {
        'report_id': report['report_id'],
        'he_id': report['he_id'],
        'committee': report.get('committee', ''),
        'n_paras': len(paras),
        'n_tagged': len(all_tags),
        'tags': all_tags,
        'paras': paras,
    }


# ---------------------------------------------------------------------------
# DB
# ---------------------------------------------------------------------------

def ensure_db_tables(db_path: Path) -> None:
    conn = sqlite3.connect(str(db_path))
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS mietinto_tag (
            he_id       TEXT NOT NULL,
            report_id   TEXT NOT NULL,
            committee   TEXT,
            para_idx    INTEGER NOT NULL,
            para_text   TEXT NOT NULL,
            section     TEXT,
            role        TEXT,
            quality     TEXT,
            topic       TEXT,
            PRIMARY KEY (he_id, report_id, para_idx)
        );
        CREATE INDEX IF NOT EXISTS idx_mt_he ON mietinto_tag(he_id);
        CREATE INDEX IF NOT EXISTS idx_mt_role ON mietinto_tag(role);
        CREATE INDEX IF NOT EXISTS idx_mt_topic ON mietinto_tag(topic);
    """)
    conn.close()


def write_results(db_path: Path, results: list[dict]) -> int:
    conn = sqlite3.connect(str(db_path))
    total = 0
    for r in results:
        he_id = r['he_id']
        conn.execute("DELETE FROM mietinto_tag WHERE he_id=? AND report_id=?",
                      (he_id, r['report_id']))
        rows = []
        for i, (text, section) in enumerate(r['paras']):
            tag = r['tags'].get(i)
            if tag is None:
                continue
            role, qual, topic = tag
            rows.append((
                he_id, r['report_id'], r.get('committee', ''),
                i, text,
                section,
                ROLE_LABELS.get(role, role),
                QUALITY_LABELS.get(qual, qual),
                TOPIC_LABELS.get(topic, topic),
            ))
        conn.executemany(
            "INSERT OR REPLACE INTO mietinto_tag "
            "(he_id, report_id, committee, para_idx, para_text, section, role, quality, topic) "
            "VALUES (?,?,?,?,?,?,?,?,?)", rows)
        total += len(rows)
    conn.commit()
    conn.close()
    return total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

EXTRACTOR_VERSION = extractor_version(SYSTEM, "tag_mietinto_v1")


def load_reports(he_id: str | None = None) -> list[dict]:
    conn = sqlite3.connect(str(INDEX_DB))
    if he_id:
        rows = conn.execute(
            "SELECT report_id, he_id, committee, content FROM committee_report "
            "WHERE he_id=? AND content IS NOT NULL AND length(content) > 200",
            (he_id,)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT report_id, he_id, committee, content FROM committee_report "
            "WHERE content IS NOT NULL AND length(content) > 200 "
            "ORDER BY he_id"
        ).fetchall()
    conn.close()
    return [{'report_id': r[0], 'he_id': r[1], 'committee': r[2], 'content': r[3]} for r in rows]


async def _main_async(args) -> dict:
    he_id = args.he if hasattr(args, 'he') else None
    if he_id and not he_id.startswith('he-'):
        he_id = f'he-{he_id}'

    reports = load_reports(he_id)

    if not getattr(args, 'force', False):
        before = len(reports)
        all_he_ids = [r['he_id'] for r in reports]
        stale_he_ids = set(stale_keys(ENRICHMENTS_DB, "mietinto_tag", "he_id", all_he_ids, EXTRACTOR_VERSION))
        reports = [r for r in reports if r['he_id'] in stale_he_ids]
        skipped = before - len(reports)
        if skipped:
            print(f"Skipping {skipped} up-to-date reports (v={EXTRACTOR_VERSION[:8]}, use --force to recompute)")

    print(f"Processing {len(reports)} committee reports")

    sem = asyncio.Semaphore(args.parallel)
    results = []

    async with aiohttp.ClientSession() as session:
        for report in reports:
            result = await tag_report(session, sem, report)
            results.append(result)
            if result['n_tagged'] > 0:
                print(f"  {result['report_id']:25s} {result['n_paras']:3d} paras → {result['n_tagged']:3d} tagged  {result['committee']}")

    if args.write_db:
        ensure_db_tables(ENRICHMENTS_DB)
        n = write_results(ENRICHMENTS_DB, results)
        for r in results:
            if r['n_tagged'] > 0:
                stamp_version(ENRICHMENTS_DB, "mietinto_tag", "he_id", r['he_id'], EXTRACTOR_VERSION)
        print(f"\n{n} rows written to mietinto_tag (v={EXTRACTOR_VERSION[:8]})")

    total_tagged = sum(r['n_tagged'] for r in results)
    print(f"\nDone: {len(results)} reports, {total_tagged} tagged paragraphs")
    return {'n_reports': len(results), 'n_tagged': total_tagged}


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
        force=force,
        parallel=parallel,
    )
    return await _main_async(args)


def main(args=None):
    if args is None:
        parser = argparse.ArgumentParser()
        parser.add_argument('--he', help='Specific HE')
        parser.add_argument('--write-db', action='store_true')
        parser.add_argument('--force', action='store_true', help='Recompute even if already tagged')
        parser.add_argument('--parallel', type=int, default=PARALLEL)
        args = parser.parse_args()
    asyncio.run(_main_async(args))


if __name__ == '__main__':
    main()

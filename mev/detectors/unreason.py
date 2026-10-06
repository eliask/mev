"""
LLM-powered semantic unreason detection in HE impact assessments.
Targets complex patterns where regex fails due to linguistic variety or logical tension.

Detectors:
  1. juridical_shield_llm      — framing decision as forced compliance to avoid scrutiny
  2. epistemic_buffering_llm   — precision-disclaimer tension (numbers vs. "impossible to say")
  3. floor_ceiling_llm         — minimum level set in context where it will become the maximum
"""

import asyncio
import argparse
import json
import os
import re
import sqlite3
from typing import Any

import aiohttp

from mev.config import HE_DB_DIR, ENRICHMENTS_DB
from mev.llm import call_llm, batch_llm

PARALLEL = int(os.environ.get("LLM_PARALLEL", "4"))

SYSTEM_PROMPT = """Olet kokenut hallituksen esitysten (HE) kriittinen arvioija.
Tehtäväsi on tunnistaa episteemisiä tai loogisia vinoumia vaikutusarvioista.
Noudata tarkasti annettua koodistoa ja tulosformaattia.

Format rules:
- Input items numbered with [N]
- Output: [N] CODE
- If no finding for an item, omit it.
- If no findings at all, output: NONE
- Ei selityksiä, ei sulkeita.
"""


def parse_findings(content: str) -> list[tuple]:
    findings = []
    if content.strip().upper() == "NONE":
        return findings
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        match = re.match(r'^\[?(\d+)\]?\s+(.+)$', line)
        if match:
            try:
                idx = int(match.group(1))
                codes = match.group(2).split()
                findings.append((idx, codes))
            except ValueError:
                continue
    return findings


async def _scan_atom(session: aiohttp.ClientSession, args: tuple) -> list[dict[str, Any]]:
    """Scan a single atom for all LLM detectors. args = (he_year, atom_id, indexed_content)"""
    he_year, atom_id, indexed_content = args
    findings = []
    try:
        user_content = f"HE vuodelta {he_year}:\n{indexed_content}"
        result = await call_llm(
            session,
            SYSTEM_PROMPT,
            user_content,
            max_tokens=200,
            ctx=atom_id,
        )
        for _idx, codes in parse_findings(result):
            for code in codes:
                detector = {
                    'J': 'juridical_shield_llm',
                    'B': 'epistemic_buffering_llm',
                    'F': 'floor_ceiling_llm',
                }.get(code.upper())
                if detector:
                    findings.append({'detector': detector, 'atom_id': atom_id, 'raw': result})
    except Exception:
        pass
    return findings


async def process_he(he_id: str, force: bool = False) -> None:
    if not force:
        enr_conn = sqlite3.connect(str(ENRICHMENTS_DB))
        existing = enr_conn.execute(
            "SELECT 1 FROM unreason_flag WHERE he_id = ? AND detector LIKE '%_llm' LIMIT 1",
            (he_id,)
        ).fetchone()
        enr_conn.close()
        if existing:
            return

    db_path = HE_DB_DIR / f"{he_id}.db"
    if not db_path.exists():
        return

    try:
        he_year = int(he_id.split('-')[-1])
    except (ValueError, IndexError):
        he_year = 2024

    conn = sqlite3.connect(str(db_path))
    atoms = conn.execute(
        "SELECT atom_id, content FROM atoms WHERE atom_type='IMPACT' AND length(content) > 100 ORDER BY atom_id"
    ).fetchall()
    conn.close()

    if not atoms:
        return

    items = [
        (he_year, atom_id, f"[{i+1}] {content}")
        for i, (atom_id, content) in enumerate(atoms)
    ]

    all_findings = await batch_llm(items, _scan_atom, parallel=PARALLEL, desc=f"Scanning {he_id}")

    enr_conn = sqlite3.connect(str(ENRICHMENTS_DB))
    for findings in all_findings:
        if not findings:
            continue
        for f in findings:
            enr_conn.execute(
                "INSERT OR REPLACE INTO unreason_flag "
                "(he_id, detector, severity, evidence_atoms, evidence_text, meta) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    he_id,
                    f['detector'],
                    2,
                    json.dumps([f['atom_id']]),
                    f"LLM-havaittu vinouma: {f['detector']}",
                    json.dumps({'atom_id': f['atom_id'], 'raw_llm': f['raw']}, ensure_ascii=False),
                )
            )
    enr_conn.commit()
    enr_conn.close()


async def main() -> None:
    parser = argparse.ArgumentParser(description='LLM Unreason Scanner')
    parser.add_argument('he_id', nargs='?', help='Single HE to analyze')
    parser.add_argument('--force', action='store_true', help='Re-process even if flags exist')
    parser.add_argument('--write-db', action='store_true', help='Write to DB (pipeline compatibility)')
    args = parser.parse_args()

    if args.he_id:
        await process_he(args.he_id, force=args.force)
    else:
        enr_conn = sqlite3.connect(str(ENRICHMENTS_DB))
        he_ids = sorted([r[0] for r in enr_conn.execute(
            "SELECT DISTINCT he_id FROM atom_enrichment"
        ).fetchall()])
        enr_conn.close()
        for hid in he_ids:
            await process_he(hid, force=args.force)


if __name__ == "__main__":
    asyncio.run(main())

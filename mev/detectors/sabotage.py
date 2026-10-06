"""
Candidate scan for enacted text that is weaker or vaguer than the bill.

This does not establish motive. A weakening edit is a text difference, not
evidence of sabotage. Treat every hit as a candidate for later reading.

Detectors:
  1. weakened_authority_llm    — changing "shall" to "may", adding vague exceptions
  2. transparency_removal_llm  — removing reporting or oversight requirements
  3. delegation_injection_llm  — adding new delegation clauses not in the original HE
"""

import asyncio
import os
import argparse
import json
import sqlite3
from typing import Any

import aiohttp

from mev.config import HE_DB_DIR, ENRICHMENTS_DB
from mev.llm import call_llm, batch_llm

PARALLEL = int(os.environ.get("LLM_PARALLEL", "4"))

SYSTEM_PROMPT = """Olet kokenut lainsäädännön asiantuntija.
Tehtäväsi on verrata hallituksen esityksen (HE) lakitekstiä ja eduskunnan hyväksymää lopullista lakitekstiä.
Tunnista muutokset, jotka heikentävät lain tavoitteita, vähentävät läpinäkyvyyttä tai lisäävät mielivaltaista valtaa.

Koodisto:
W = Weakened (heikennetty velvoittavuutta, esim. "on" -> "voi", lisätty väljiä poikkeuksia)
T = Transparency removed (poistettu raportointi-, julkisuus- tai valvontavaatimuksia)
D = Delegation injected (lisätty uusia asetuksenantovaltuuksia tai delegointia)

Tulosformaatti:
CODE: LYHYT_PERUSTELU
Esimerkki: W: "on" muutettu muotoon "voi", mikä heikentää velvoittavuutta.

Jos muutoksella ei ole merkittävää negatiivista vaikutusta (esim. korjattu kirjoitusvirhe tai tekninen muutos), vastaa: NONE
Ei selityksiä, ei sulkeita.
"""


async def _analyze_drift(session: aiohttp.ClientSession, args: tuple) -> list[dict[str, Any]]:
    """Compare proposed vs enacted text for one section. args = (he_id, section, proposed, enacted)"""
    he_id, section, proposed, enacted = args
    user_prompt = (
        f"HE-ID: {he_id}\n"
        f"Pykälä: {section}\n\n"
        f"EHDOTETTU TEKSTI:\n{proposed}\n\n"
        f"HYVÄKSYTTY TEKSTI:\n{enacted}\n\n"
        "Analysoi muutos:"
    )
    findings = []
    try:
        content = await call_llm(session, SYSTEM_PROMPT, user_prompt, max_tokens=150, ctx=f"{he_id}/{section}")
        if content.strip().upper() == "NONE":
            return findings
        for line in content.splitlines():
            if ":" in line:
                parts = line.split(":", 1)
                code = parts[0].strip().upper()
                reason = parts[1].strip()
                det = {
                    'W': 'weakened_authority_llm',
                    'T': 'transparency_removal_llm',
                    'D': 'delegation_injection_llm',
                }.get(code)
                if det:
                    findings.append({'detector': det, 'section': section, 'reason': reason, 'raw': line})
    except Exception:
        pass
    return findings


async def process_he(he_id: str, force: bool = False) -> None:
    if not force:
        enr_conn = sqlite3.connect(str(ENRICHMENTS_DB))
        existing = enr_conn.execute(
            "SELECT 1 FROM unreason_flag WHERE he_id = ? AND detector LIKE '%_llm' "
            "AND (detector LIKE 'weakened%' OR detector LIKE 'transparency%' OR detector LIKE 'delegation_injection%') LIMIT 1",
            (he_id,)
        ).fetchone()
        enr_conn.close()
        if existing:
            return

    db_path = HE_DB_DIR / f"{he_id}.db"
    if not db_path.exists():
        return

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    has_drift = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='mechanism_drift'"
    ).fetchone()
    if not has_drift:
        conn.close()
        return

    changes = conn.execute(
        "SELECT section, proposed_text, enacted_text FROM mechanism_drift WHERE status='CHANGED' ORDER BY section"
    ).fetchall()
    conn.close()

    if not changes:
        return

    items = [(he_id, row['section'], row['proposed_text'], row['enacted_text']) for row in changes]
    all_findings = await batch_llm(items, _analyze_drift, parallel=PARALLEL, desc=f"Sabotage scan {he_id}")

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
                    3,
                    json.dumps([]),
                    f"Parlamentaarinen muutos ({f['section']}): {f['reason']}",
                    json.dumps({'section': f['section'], 'raw_llm': f['raw']}, ensure_ascii=False),
                )
            )
    enr_conn.commit()
    enr_conn.close()


async def main() -> None:
    parser = argparse.ArgumentParser(description='LLM Sabotage Scanner')
    parser.add_argument('he_id', nargs='?', help='Single HE to analyze')
    parser.add_argument('--force', action='store_true', help='Re-process even if results exist')
    args = parser.parse_args()

    if args.he_id:
        await process_he(args.he_id, force=args.force)
    else:
        he_ids = sorted([p.stem for p in HE_DB_DIR.glob("he-*.db")])
        for hid in he_ids:
            await process_he(hid, force=args.force)


if __name__ == "__main__":
    asyncio.run(main())

"""
LLM-powered summarization of delegation scopes.
Runs alongside regex-based scope extraction, adding a semantic summary column.

Reads:
    .tmp/he_dbs/he-*.db [delegation_drift table]

Writes:
    Back to delegation_drift table (adds/updates scope_llm column)
"""

import asyncio
import os
import argparse
import sqlite3

import aiohttp

from mev.config import HE_DB_DIR
from mev.llm import call_llm, batch_llm

PARALLEL = int(os.environ.get("LLM_PARALLEL", "4"))

SYSTEM_PROMPT = """Olet lainsäädännön asiantuntija.
Tehtäväsi on tiivistää asetuksenantovaltuuden (delegointi) sisältö lyhyeksi ja selkeäksi ilmaukseksi.
Vastaa vain tiivistelmällä, ei muuta tekstiä.

Esimerkki:
Syöte: "Valtioneuvoston asetuksella säädetään tarkemmin palvelutarvekertoimen laskennasta"
Tulos: palvelutarvekertoimen laskenta

Syöte: "Tarkemmat säännökset hakemuksen sisällöstä ja toimittamisesta annetaan opetus- ja kulttuuriministeriön asetuksella"
Tulos: hakemuksen sisältö ja toimittaminen
"""


async def _summarize_one(session: aiohttp.ClientSession, args: tuple) -> tuple[int, str]:
    """Summarize one scope. args = (rowid, quote). Returns (rowid, summary)."""
    rowid, quote = args
    if not quote or len(quote) < 10:
        return rowid, quote
    try:
        result = await call_llm(session, SYSTEM_PROMPT, quote, max_tokens=50, ctx=str(rowid))
        return rowid, result.strip().rstrip('.')
    except Exception:
        return rowid, quote


async def process_he(he_id: str) -> None:
    db_path = HE_DB_DIR / f"{he_id}.db"
    if not db_path.exists():
        return

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    has_table = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='delegation_drift'"
    ).fetchone()
    if not has_table:
        conn.close()
        return

    try:
        conn.execute("ALTER TABLE delegation_drift ADD COLUMN scope_llm TEXT")
    except Exception:
        pass  # column already exists

    rows = conn.execute("SELECT rowid, quote FROM delegation_drift ORDER BY rowid").fetchall()
    if not rows:
        conn.close()
        return

    print(f"Summarizing {len(rows)} delegation scopes for {he_id}")
    items = [(row['rowid'], row['quote']) for row in rows]
    results = await batch_llm(items, _summarize_one, parallel=PARALLEL, desc=f"Scopes {he_id}")

    for result in results:
        if result is None:
            continue
        rowid, scope_llm = result
        conn.execute("UPDATE delegation_drift SET scope_llm = ? WHERE rowid = ?", (scope_llm, rowid))
    conn.commit()
    conn.close()


async def main() -> None:
    parser = argparse.ArgumentParser(description='LLM Delegation Summarizer')
    parser.add_argument('he_id', nargs='?', help='Single HE to analyze')
    args = parser.parse_args()

    if args.he_id:
        await process_he(args.he_id)
    else:
        he_ids = [p.stem for p in HE_DB_DIR.glob("he-*.db")]
        for hid in he_ids:
            await process_he(hid)


if __name__ == "__main__":
    asyncio.run(main())

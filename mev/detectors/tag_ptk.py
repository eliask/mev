"""
Speech-level classifier for PTK plenary speeches.

Tags each speech with three dimensions:
  Stance:  S(support) O(oppose) T(procedural) N(neutral)
  Content: K(concrete_concern) A(expert_reference) R(rhetoric) X(other)
  Topic:   F/W/S/C/I/N/D/Y/J/R/X  (same capital-stock categories as tag_he_sentences.py)

Purpose: enables triple-corroboration chain analysis.
  Expert concern → committee ignores (scrutiny SILENT) → MP echoes in PTK
  The "O+K" speeches (oppose + concrete concern) are the key signal.

Reads from: per-HE DBs (.tmp/he_dbs/<he_id>.db)  [ptk_speeches table]
Scoped to: top HEs by corroboration score (.tmp/corroboration_scores.csv)
Writes to: he_enrichments.db  [ptk_speech_tag table]

Usage:
  uv run mev detector tag-ptk                  # top 20 HEs, console
  uv run mev detector tag-ptk --n-he 100       # top 100 HEs
  uv run mev detector tag-ptk --write-db       # persist to DB
  uv run mev detector tag-ptk he-241-2020      # specific HE
  uv run mev detector tag-ptk --parallel 12    # tune concurrency
"""

import argparse
import asyncio
import os
import csv
import json
import sqlite3
import sys
from collections import defaultdict

import aiohttp

from mev.config import ROOT, HE_DB_DIR, ENRICHMENTS_DB
from mev.versioning import extractor_version, stale_keys, stamp_version
from mev.detectors.tagger_engine import call_llm_windowed

SCORES_CSV = ROOT / ".tmp" / "corroboration_scores.csv"

# Wire codes → labels
STANCE_LABELS = {
    'S': 'support', 'O': 'oppose', 'T': 'procedural', 'N': 'neutral',
}
CONTENT_LABELS = {
    'K': 'concrete_concern', 'A': 'expert_reference', 'R': 'rhetoric', 'X': 'other',
}
TOPIC_LABELS = {
    'F': 'fiscal', 'W': 'epistemic', 'S': 'social', 'C': 'cognitive',
    'I': 'institutional', 'N': 'infrastructure', 'D': 'human',
    'Y': 'coherence', 'J': 'purpose', 'R': 'moral', 'X': 'other',
}
VALID_STANCES = set(STANCE_LABELS)
VALID_CONTENTS = set(CONTENT_LABELS)
VALID_TOPICS = set(TOPIC_LABELS)

# Party → likely stance prior (for sanity checking output)
GOVT_PARTIES_2023 = {'kok', 'ps', 'rkp', 'kd', 'kokoomus', 'perussuomalaiset'}


SYSTEM = """Luokittele eduskuntapuheenvuoro kolmella ulottuvuudella.

ASEMA (S/O/T/N):
  S = tukee esitystä / puolustaa hallitusta
  O = vastustaa / ilmaisee huolen esityksen sisällöstä tai vaikutuksista
  T = proseduraalinen (äänestykset, järjestys, puhujalistat)
  N = neutraali tai informatiivinen

SISÄLTÖ (K/A/R/X):
  K = konkreettinen kritiikki tai ennuste vaikutuksista (nimetty ongelma)
  A = asiantuntijoiden tai lausuntojen nimenomainen viittaus/siteeraus
  R = poliittinen retoriikka ilman konkreettisia väitteitä tai tietoja
  X = muu (epäselvä tai lyhyt)

AIHE (F/W/S/C/I/N/D/Y/J/R/X):
  F=fiskaalinen  W=episteeminen  S=sosiaalinen  C=kognitiivinen
  I=institutionaalinen  N=infrastruktuuri  D=inhimillinen
  Y=yhtenäisyys  J=tarkoitus  R=moraalinen  X=muu

Tulosta VAIN kolme kirjainta: ASEMA SISÄLTÖ AIHE
Esimerkki: "O K D"
Ei muuta tekstiä."""

EXTRACTOR_VERSION = extractor_version(SYSTEM, "tag_ptk_v1")


async def call_llm(session, sem, text: str) -> dict:
    """Thin wrapper: acquires sem, truncates long speeches, delegates to engine."""
    # Speeches state stance/topic early; truncate at sentence boundary near 3000 chars
    if len(text) > 3000:
        cut = text.rfind('. ', 2000, 3200)
        text = text[:cut + 1] if cut > 0 else text[:3000]
    return await call_llm_windowed(session, sem, SYSTEM, text, max_tokens=8)


def parse_tag(raw: str) -> tuple[str, str, str]:
    """Parse 'STANCE CONTENT TOPIC' from LLM output. Returns ('N','X','X') on failure."""
    parts = raw.strip().upper().split()
    stance = parts[0] if len(parts) > 0 and parts[0] in VALID_STANCES else 'N'
    content = parts[1] if len(parts) > 1 and parts[1] in VALID_CONTENTS else 'X'
    topic = parts[2] if len(parts) > 2 and parts[2] in VALID_TOPICS else 'X'
    return stance, content, topic


def load_top_he_ids(n: int) -> list[str]:
    """Load top N HE IDs by composite corroboration score.

    Unique to PTK tagger — scopes work to HEs with most signal.
    """
    if not SCORES_CSV.exists():
        print(f"Warning: {SCORES_CSV} not found — no corroboration scores available",
              file=sys.stderr)
        return []
    rows = []
    with open(SCORES_CSV, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((row['he_id'], float(row.get('composite_score', 0))))
    rows.sort(key=lambda x: -x[1])
    return [r[0] for r in rows[:n]]


def load_speeches(he_id: str) -> list[dict]:
    """Load ptk_speeches from per-HE DB."""
    db_path = HE_DB_DIR / f"{he_id}.db"
    if not db_path.exists():
        return []
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT rowid, ptk_tunnus, speaker, party, role, speech_time, text "
            "FROM ptk_speeches ORDER BY rowid"
        ).fetchall()
        return [dict(r) for r in rows]
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()


def ensure_db_tables():
    conn = sqlite3.connect(str(ENRICHMENTS_DB))
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS ptk_speech_tag (
            he_id       TEXT NOT NULL,
            speech_rowid INTEGER NOT NULL,
            ptk_tunnus  TEXT,
            speaker     TEXT,
            party       TEXT,
            stance      TEXT,   -- support/oppose/procedural/neutral
            content     TEXT,   -- concrete_concern/expert_reference/rhetoric/other
            topic       TEXT,   -- capital stock topic
            PRIMARY KEY (he_id, speech_rowid)
        );
        CREATE INDEX IF NOT EXISTS idx_pst_he    ON ptk_speech_tag(he_id);
        CREATE INDEX IF NOT EXISTS idx_pst_stance ON ptk_speech_tag(stance);
        CREATE INDEX IF NOT EXISTS idx_pst_topic  ON ptk_speech_tag(topic);
    """)
    conn.commit()
    conn.close()


def write_tags_to_db(he_id: str, rows: list[dict]):
    conn = sqlite3.connect(str(ENRICHMENTS_DB))
    conn.execute("DELETE FROM ptk_speech_tag WHERE he_id = ?", (he_id,))
    conn.executemany(
        "INSERT OR REPLACE INTO ptk_speech_tag "
        "(he_id, speech_rowid, ptk_tunnus, speaker, party, stance, content, topic) "
        "VALUES (?,?,?,?,?,?,?,?)",
        [
            (he_id, r['rowid'], r['ptk_tunnus'], r['speaker'], r['party'],
             r['stance'], r['content'], r['topic'])
            for r in rows
        ]
    )
    conn.commit()
    conn.close()


def print_he_summary(he_id: str, tagged: list[dict]):
    """Print compact summary for one HE."""
    n = len(tagged)
    if not n:
        print(f"  {he_id}: 0 speeches")
        return

    stance_dist = defaultdict(int)
    content_dist = defaultdict(int)
    topic_dist = defaultdict(int)
    for r in tagged:
        stance_dist[r['stance']] += 1
        content_dist[r['content']] += 1
        topic_dist[r['topic']] += 1

    oppose_concrete = sum(1 for r in tagged
                          if r['stance'] == 'oppose' and r['content'] == 'concrete_concern')
    expert_refs = sum(1 for r in tagged if r['content'] == 'expert_reference')

    print(f"  {he_id}: {n} speeches | "
          f"oppose={stance_dist['oppose']} support={stance_dist['support']} "
          f"proc={stance_dist['procedural']} | "
          f"O+K={oppose_concrete} A={expert_refs}")

    top_topics = sorted(topic_dist.items(), key=lambda x: -x[1])[:4]
    print(f"    Topics: {' '.join(f'{t}={c}' for t, c in top_topics)}")


async def tag_he_speeches(he_id: str, session, sem,
                          max_speeches: int = 0) -> list[dict]:
    """Tag all speeches for one HE. Returns list of tagged dicts."""
    speeches = load_speeches(he_id)
    if not speeches:
        return []

    if max_speeches > 0:
        speeches = speeches[:max_speeches]

    tasks = [call_llm(session, sem, s['text'] or '') for s in speeches]
    raw_results = await asyncio.gather(*tasks, return_exceptions=True)

    tagged = []
    for speech, result in zip(speeches, raw_results):
        if isinstance(result, Exception):
            print(f"  PTK tag error: {result}")
            continue
        stance, content, topic = parse_tag(result.get('content', ''))
        tagged.append({
            **speech,
            'stance': STANCE_LABELS[stance],
            'content': CONTENT_LABELS[content],
            'topic': TOPIC_LABELS[topic],
        })

    return tagged


async def _main_async(args):
    if args.he_id:
        he_ids = [args.he_id]
    else:
        he_ids = load_top_he_ids(args.n_he)
        if not he_ids:
            print("No HE IDs found. Run `mev detect corroboration` first.")
            sys.exit(1)
        print(f"Processing top {len(he_ids)} HEs by corroboration score")

    if args.write_db:
        ensure_db_tables()

    # Skip up-to-date HEs using versioning (replaces old skip_existing logic)
    if not getattr(args, 'force', False):
        before = len(he_ids)
        he_ids = stale_keys(ENRICHMENTS_DB, "ptk_speech_tag", "he_id", he_ids, EXTRACTOR_VERSION)
        skipped = before - len(he_ids)
        if skipped:
            print(f"Skipping {skipped} up-to-date HEs (v={EXTRACTOR_VERSION[:8]}, use --force to recompute)")

    sem = asyncio.Semaphore(args.parallel)
    total_speeches = 0
    total_oppose_k = 0
    total_expert_refs = 0

    print()
    async with aiohttp.ClientSession() as session:
        for he_id in he_ids:
            tagged = await tag_he_speeches(he_id, session, sem, args.max_speeches)
            if not tagged:
                print(f"  {he_id}: no speeches")
                continue

            print_he_summary(he_id, tagged)
            total_speeches += len(tagged)
            total_oppose_k += sum(1 for r in tagged
                                  if r['stance'] == 'oppose' and r['content'] == 'concrete_concern')
            total_expert_refs += sum(1 for r in tagged if r['content'] == 'expert_reference')

            if args.write_db:
                write_tags_to_db(he_id, tagged)
                stamp_version(ENRICHMENTS_DB, "ptk_speech_tag", "he_id", he_id, EXTRACTOR_VERSION)

    print(f"\n{'='*60}")
    print(f"TOTAL: {len(he_ids)} HEs, {total_speeches} speeches")
    print(f"Oppose+ConcreteK: {total_oppose_k}")
    print(f"Expert references: {total_expert_refs}")

    if args.write_db:
        print(f"Written to: {ENRICHMENTS_DB} (ptk_speech_tag table)")

    if not args.write_db:
        print("\n(dry run — add --write-db to persist)")

    return {'n_he': len(he_ids), 'n_speeches': total_speeches,
            'n_oppose_k': total_oppose_k, 'n_expert_refs': total_expert_refs}


async def run(
    he_id: str | None = None,
    write_db: bool = False,
    force: bool = False,
    parallel: int = int(os.environ.get("LLM_PARALLEL", "4")),
) -> dict:
    """Programmatic entry point. Returns summary dict."""
    import types
    args = types.SimpleNamespace(
        he_id=he_id,
        n_he=20,
        max_speeches=0,
        write_db=write_db,
        parallel=parallel,
        force=force,
    )
    return await _main_async(args)


def main(args=None):
    if args is None:
        parser = argparse.ArgumentParser(description="Tag PTK plenary speeches by stance/content/topic")
        parser.add_argument('he_id', nargs='?', default=None,
                            help='Specific HE ID (default: use top-N from corroboration scores)')
        parser.add_argument('--n-he', type=int, default=20,
                            help='Number of top HEs to process (default: 20)')
        parser.add_argument('--max-speeches', type=int, default=0,
                            help='Max speeches per HE (0=all)')
        parser.add_argument('--write-db', action='store_true',
                            help='Persist tags to he_enrichments.db')
        parser.add_argument('--parallel', type=int, default=int(os.environ.get("LLM_PARALLEL", "4")),
                            help='Max concurrent LLM requests')
        parser.add_argument('--force', action='store_true',
                            help='Recompute even if already tagged (ignores version cache)')
        args = parser.parse_args()
    asyncio.run(_main_async(args))


if __name__ == '__main__':
    main()

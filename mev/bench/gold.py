"""Gold set management for the output-format benchmark.

Gold set = a list of sentences with known (correct) labels drawn from the
existing tagged corpus (he_enrichments.db → sentence_tag table).

Each GoldItem stores the sentence text + its 3-dimensional ground-truth label
as single-char codes (P/E/V/K/L, G/M/A/H/T, F/W/S/C/I/N/D/Y/J/R/X).

Usage:
    gold = create_gold_set(n=200, seed=42)  # sample from DB
    gold = load_gold_set()                   # load saved JSON
    save_gold_set(gold, path)                # save to JSON
"""
from __future__ import annotations

import json
import random
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from mev.bench.formats import label_to_code
from mev.config import ENRICHMENTS_DB, ROOT

GOLD_DIR  = ROOT / "data" / "bench"
GOLD_PATH = GOLD_DIR / "tag_gold.json"


@dataclass
class GoldItem:
    he_id:    str
    atom_id:  str
    sent_idx: int
    text:     str
    role:     str   # single-char code: P/E/V/K/L
    quality:  str   # single-char code: G/M/A/H/T
    topic:    str   # single-char code: F/W/S/C/I/N/D/Y/J/R/X


def create_gold_set(
    n: int = 200,
    seed: int = 42,
    min_text_len: int = 30,
    source_db: Optional[Path] = None,
) -> list[GoldItem]:
    """Sample n sentences with complete labels from he_enrichments.db.

    Strategy: sample across diverse HEs (not all from one HE) to get a
    representative spread of roles, qualities, and topics.

    Args:
        n: Number of sentences to sample.
        seed: Random seed for reproducibility.
        min_text_len: Minimum sentence length in characters (filters short stubs).
        source_db: Path to enrichments DB. Defaults to ENRICHMENTS_DB.

    Returns:
        List of GoldItem instances.
    """
    db_path = source_db or ENRICHMENTS_DB
    if not db_path.exists():
        raise FileNotFoundError(
            f"Enrichments DB not found at {db_path}. "
            "Run: mev build tag (or mev detect tag --write-db)"
        )

    conn = sqlite3.connect(str(db_path))
    try:
        # Pull all tagged sentences with full labels
        rows = conn.execute(
            """
            SELECT he_id, atom_id, sent_idx, sent_text, role, quality, topic
            FROM sentence_tag
            WHERE role IS NOT NULL
              AND quality IS NOT NULL
              AND topic IS NOT NULL
              AND sent_kind = 'prose'
              AND length(sent_text) >= ?
            ORDER BY he_id, atom_id, sent_idx
            """,
            (min_text_len,)
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        raise RuntimeError("No tagged sentences found in enrichments DB.")

    # Convert to GoldItem list with code translation
    items: list[GoldItem] = []
    for he_id, atom_id, sent_idx, text, role_label, qual_label, topic_label in rows:
        r, q, t = label_to_code(role_label, qual_label, topic_label)
        if r == '?' or q == '?':
            continue  # skip anomaly / unknown labels
        items.append(GoldItem(
            he_id=he_id,
            atom_id=atom_id,
            sent_idx=sent_idx,
            text=text,
            role=r,
            quality=q,
            topic=t,
        ))

    if len(items) < n:
        raise RuntimeError(
            f"Only {len(items)} valid items available, requested n={n}. "
            "Run more tagging jobs or reduce n."
        )

    # Stratified sampling: pick items from diverse HEs
    rng = random.Random(seed)

    # Group by HE
    by_he: dict[str, list[GoldItem]] = {}
    for item in items:
        by_he.setdefault(item.he_id, []).append(item)

    he_ids = sorted(by_he.keys())
    rng.shuffle(he_ids)

    selected: list[GoldItem] = []
    # Round-robin across HEs until we have n items
    he_iters = {h: iter(by_he[h]) for h in he_ids}
    active_he = list(he_ids)
    while len(selected) < n and active_he:
        next_active = []
        for h in active_he:
            try:
                item = next(he_iters[h])
                selected.append(item)
                if len(selected) >= n:
                    break
                next_active.append(h)
            except StopIteration:
                pass  # this HE exhausted, drop it
        active_he = next_active

    # Final shuffle so ordering is random (not HE-grouped)
    rng.shuffle(selected)
    return selected[:n]


def save_gold_set(items: list[GoldItem], path: Optional[Path] = None) -> Path:
    """Save gold set to JSON. Returns the path written."""
    out = path or GOLD_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(
            [asdict(g) for g in items],
            f,
            ensure_ascii=False,
            indent=2,
        )
    return out


def load_gold_set(path: Optional[Path] = None) -> list[GoldItem]:
    """Load gold set from JSON. Falls back to default path."""
    p = path or GOLD_PATH
    if not p.exists():
        raise FileNotFoundError(
            f"Gold set not found at {p}. Run: mev detect bench-format --create-gold"
        )
    with open(p, encoding='utf-8') as f:
        data = json.load(f)
    return [GoldItem(**d) for d in data]


def gold_set_exists(path: Optional[Path] = None) -> bool:
    """Return True if gold set JSON exists."""
    return (path or GOLD_PATH).exists()


def validate_gold_set(items: list[GoldItem], warn: bool = True) -> dict[str, object]:
    """Check diversity properties of a gold set and warn if coverage is poor.

    Checks:
      - All 5 role types represented (P/E/V/K/L)
      - All 5 quality types represented (G/M/A/H/T)
      - Multiple topic types represented (ideally 5+)
      - Sentences from at least 10 different HEs

    Args:
        items: Gold set to validate.
        warn: If True, print warnings to stdout for failed checks.

    Returns:
        Dict with keys: roles_present, quals_present, topics_present,
        n_he_ids, missing_roles, missing_quals, missing_topics, ok (bool).
    """
    roles_present  = set(g.role    for g in items)
    quals_present  = set(g.quality for g in items)
    topics_present = set(g.topic   for g in items)
    he_ids_present = set(g.he_id   for g in items)

    all_roles   = set('PEVKL')
    all_quals   = set('GMAHT')

    missing_roles  = all_roles  - roles_present
    missing_quals  = all_quals  - quals_present
    missing_topics = topics_present  # just report what's present

    ok = True
    issues = []

    if missing_roles:
        ok = False
        issues.append(f"Missing role types: {sorted(missing_roles)}")
    if missing_quals:
        ok = False
        issues.append(f"Missing quality types: {sorted(missing_quals)}")
    if len(topics_present) < 5:
        ok = False
        issues.append(
            f"Only {len(topics_present)} topic types present (want >= 5): {sorted(topics_present)}"
        )
    if len(he_ids_present) < 10:
        ok = False
        issues.append(
            f"Only {len(he_ids_present)} unique HEs represented (want >= 10)"
        )

    if warn:
        if ok:
            print(
                f"  Gold set OK: {len(items)} items, "
                f"{len(roles_present)}/5 roles, "
                f"{len(quals_present)}/5 quals, "
                f"{len(topics_present)} topics, "
                f"{len(he_ids_present)} HEs"
            )
        else:
            print(f"  WARNING: gold set diversity issues:")
            for issue in issues:
                print(f"    - {issue}")

    return {
        "roles_present":  sorted(roles_present),
        "quals_present":  sorted(quals_present),
        "topics_present": sorted(topics_present),
        "n_he_ids":       len(he_ids_present),
        "missing_roles":  sorted(missing_roles),
        "missing_quals":  sorted(missing_quals),
        "ok":             ok,
    }

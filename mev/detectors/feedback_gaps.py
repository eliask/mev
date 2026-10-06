"""
FEEDBACK<->IMPACT gap scanner.

For each HE: extract topics from FEEDBACK atoms, check which topics
are NOT addressed in IMPACT atoms. "Stakeholders raised X, but the
vaikutusarvio doesn't analyze X."

This detects a specific governance failure mode: the government summarizes
criticism in section 6 (lausuntopalaute) but doesn't address it in
section 4 (vaikutusarviointi).

Usage:
    uv run mev detector feedback-gaps
    uv run mev detector feedback-gaps --he he-38-2025
"""

import argparse
import json
import re
import sqlite3
import sys
from collections import defaultdict

from mev.config import HE_DB_DIR

# Topic extraction: look for named concerns in FEEDBACK that use these patterns
# Finnish patterns for "X was criticized / opposed / raised concern about"
CONCERN_PATTERNS = [
    r'(?:ei kannattanut|vastusti(?:vat)?|suhtautuivat?\s+(?:kriittisesti|varauksellisesti))',
    r'(?:katsoi(?:vat)?|totesi(?:vat)?|korosti(?:vat)?|esitti(?:vät)?|huomautti(?:vat)?|ehdotti(?:vat)?)',
    r'(?:puutteelli|riittämättö|ei huomioi|jättää huomiotta)',
    r'(?:huolenai|huolissaan|huoli\s)',
    r'(?:kritiik|kritisoiv)',
    r'(?:toivoi(?:vat)?\s+(?:että|lisä))',
    r'(?:piti(?:vät)?\s+(?:ongelmallisena|puutteellisena|riittämättömänä))',
]


def extract_feedback_topics(atoms: list) -> list:
    """Extract topic summaries from FEEDBACK atoms."""
    topics = []
    for atom_id, title, content in atoms:
        if not content or len(content) < 50:
            continue
        # Each FEEDBACK atom with a substantive title = one topic
        # Skip structural/meta titles that aren't substantive feedback topics
        skip_titles = {
            'lausuntopalaute', 'yleistä lausuntopalautteesta', 'johdanto',
            'lausunnonantajat', 'lausuntokierros', 'jatkovalmistelu',
            'lausuntopalautteen vuoksi tehdyt muutokset',
            'yhteenveto lausuntopalautteesta',
            'valtioneuvoston oikeuskanslerin ennakkotarkastus',
            'lainsäädännön arviointineuvoston lausunto',
            'euroopan keskuspankin lausunto',
            'yleinen palaute', 'muut huomiot', 'avoimet lausunnot',
            'lausuntopyyntö ja saadut lausunnot',
            'pykäläkohtainen lausuntopalaute',
            'muutokset pykäliin', 'muutokset perusteluihin',
        }
        # Skip titles that are organization names (not topics)
        org_pattern = re.compile(
            r'^(Apulaisoikeuskansleri|Oikeusministeriö|'
            r'[A-ZÄÖÅ][a-zäöå]+(?:liitto|järjestö|yhdistys|ry|keskus)\b|'
            r'Palkansaajajärjestöjen|Työnantajajärjestöjen|Muiden lausunnonantajien)',
            re.IGNORECASE
        )
        # Skip country comparison sections (Ruotsi, Norja, Tanska, etc.)
        country_pattern = re.compile(r'^(Ruotsi|Norja|Tanska|Saksa|Viro|Islanti|Alankomaat|Iso-Britannia)$', re.IGNORECASE)
        if title and (title.lower() not in skip_titles) and not country_pattern.match(title) and not org_pattern.match(title):
            # Count concern signals
            concern_count = 0
            for pattern in CONCERN_PATTERNS:
                concern_count += len(re.findall(pattern, content, re.IGNORECASE))

            # Extract organization names (capitalized words before common verbs)
            orgs = re.findall(
                r'([A-ZÄÖÅ][a-zäöå]+(?:\s+[A-ZÄÖÅ][a-zäöå]+)*)\s+(?:katsoi|totesi|korosti|esitti|kannatti|vastusti|ehdotti|piti|huomautti|toivoi)',
                content
            )

            topics.append({
                'atom_id': atom_id,
                'title': title,
                'char_count': len(content),
                'concern_signals': concern_count,
                'orgs_mentioned': list(set(orgs))[:10],
                'content_snippet': content[:300],
            })
    return topics


def check_impact_coverage(topic_title: str, impact_text: str) -> dict:
    """Check if a FEEDBACK topic is addressed in IMPACT text."""
    # Extract key terms from topic title (lowercase, cleaned)
    title_lower = topic_title.lower()
    # Remove common structural words
    stopwords = {'ja', 'tai', 'sekä', 'koskevat', 'koskeva', 'ehdotukset',
                 'muutokset', 'muut', 'yleistä', 'muuta', 'lain'}
    title_words = [w for w in re.findall(r'[a-zäöå]{3,}', title_lower) if w not in stopwords]

    if not title_words:
        return {'covered': True, 'match_score': 1.0, 'matching_words': []}

    # Check how many title words appear in IMPACT text
    impact_lower = impact_text.lower()
    matching = [w for w in title_words if w in impact_lower]
    match_score = len(matching) / len(title_words) if title_words else 1.0

    return {
        'covered': match_score >= 0.5,  # at least half the key terms
        'match_score': match_score,
        'matching_words': matching,
        'missing_words': [w for w in title_words if w not in matching],
    }


def scan_he(db_path) -> dict | None:
    """Scan one HE for FEEDBACK<->IMPACT gaps."""
    he_id = db_path.stem
    conn = sqlite3.connect(str(db_path))

    try:
        feedback = conn.execute(
            "SELECT atom_id, title, content FROM atoms WHERE atom_type='FEEDBACK' ORDER BY seq"
        ).fetchall()
        impact = conn.execute(
            "SELECT atom_id, title, content FROM atoms WHERE atom_type='IMPACT' ORDER BY seq"
        ).fetchall()
    except sqlite3.OperationalError:
        conn.close()
        return None

    if not feedback:
        conn.close()
        return None

    # Get metadata
    try:
        meta = conn.execute('SELECT title FROM metadata LIMIT 1').fetchone()
        he_title = meta[0] if meta else ''
    except sqlite3.OperationalError:
        he_title = ''

    conn.close()

    topics = extract_feedback_topics(feedback)
    if not topics:
        return None

    # Concatenate all IMPACT text for searching
    impact_text = ' '.join(f'{r[1] or ""} {r[2] or ""}' for r in impact)

    # Check each topic
    gaps = []
    covered = []
    for topic in topics:
        coverage = check_impact_coverage(topic['title'], impact_text)
        topic['impact_coverage'] = coverage
        if coverage['covered']:
            covered.append(topic)
        else:
            gaps.append(topic)

    return {
        'he_id': he_id,
        'title': he_title,
        'n_feedback_topics': len(topics),
        'n_covered': len(covered),
        'n_gaps': len(gaps),
        'gap_rate': len(gaps) / len(topics) if topics else 0,
        'gaps': gaps,
        'covered': covered,
    }


def main(args=None):
    if args is None:
        parser = argparse.ArgumentParser(description='FEEDBACK<->IMPACT gap scanner')
        parser.add_argument('--he', type=str, help='Scan single HE')
        parser.add_argument('--output', '-o', type=str, help='JSON output path')
        args = parser.parse_args()

    if args.he:
        db_path = HE_DB_DIR / f'{args.he}.db'
        if not db_path.exists():
            print(f'Not found: {db_path}')
            sys.exit(1)
        result = scan_he(db_path)
        if result:
            print(f'\n{result["he_id"]}: {result["n_gaps"]}/{result["n_feedback_topics"]} feedback topics NOT addressed in IMPACT')
            if result['gaps']:
                print('\nGaps (stakeholder concerns not addressed in vaikutusarvio):')
                for g in result['gaps']:
                    print(f'  x {g["title"]}')
                    print(f'    {g["concern_signals"]} concern signals, {g["char_count"]} chars')
                    if g['impact_coverage']['missing_words']:
                        print(f'    Missing terms: {", ".join(g["impact_coverage"]["missing_words"])}')
                    if g['orgs_mentioned']:
                        print(f'    Organizations: {", ".join(g["orgs_mentioned"][:5])}')
            if result['covered']:
                print('\nCovered:')
                for c in result['covered']:
                    print(f'  ok {c["title"]}  (score: {c["impact_coverage"]["match_score"]:.1%})')
        return

    # Scan all HEs
    he_dbs = sorted(HE_DB_DIR.glob('he-*.db'))
    results = []
    for db_path in he_dbs:
        r = scan_he(db_path)
        if r:
            results.append(r)

    print(f'\n=== FEEDBACK<->IMPACT Gap Scan ===')
    print(f'Scanned {len(results)} HEs with FEEDBACK topics')

    total_topics = sum(r['n_feedback_topics'] for r in results)
    total_gaps = sum(r['n_gaps'] for r in results)
    total_covered = sum(r['n_covered'] for r in results)

    print(f'\nCorpus totals:')
    print(f'  {total_topics} feedback topics across {len(results)} HEs')
    print(f'  {total_covered} ({total_covered/total_topics*100:.0f}%) addressed in vaikutusarvio')
    print(f'  {total_gaps} ({total_gaps/total_topics*100:.0f}%) NOT addressed')

    # HEs with highest gap rates
    with_gaps = [r for r in results if r['n_gaps'] > 0]
    with_gaps.sort(key=lambda r: (-r['n_gaps'], -r['gap_rate']))

    print(f'\nHEs with most unaddressed feedback topics:')
    for r in with_gaps[:15]:
        print(f'  {r["he_id"]:20s}  {r["n_gaps"]}/{r["n_feedback_topics"]} gaps ({r["gap_rate"]:.0%})')
        for g in r['gaps'][:3]:
            print(f'    x {g["title"][:70]}')

    if args.output:
        from pathlib import Path
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f'\nWrote {out_path}')


def run(**kwargs):
    """Standard detector API entry point."""
    main()


if __name__ == '__main__':
    main()

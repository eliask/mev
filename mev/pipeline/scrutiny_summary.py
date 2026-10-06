"""
Build scrutiny summary from detect_scrutiny_ignored.py output.

Reads the per-HE scrutiny analysis JSON and produces:
  1. Aggregate statistics across all analyzed HEs
  2. Per-HE summary table
  3. JSON for web visualization

Pipeline position:
  mev detect scrutiny (upstream: analyzes expert-committee matching)
  -> mev detect scrutiny-summary (this: aggregates and formats)
  -> varjo-mev.html / flow-trajectory.html (downstream: renders)

Reads:
    .tmp/scrutiny/scrutiny_analysis.json

Writes:
    .tmp/scrutiny_summary.json
    mr_website_public/mev/data/scrutiny_summary.json

Usage:
    uv run mev pipeline scrutiny-summary
"""

import json
from collections import defaultdict

from mev.config import ROOT

INPUT = ROOT / ".tmp" / "scrutiny" / "scrutiny_analysis.json"
OUTPUT = ROOT / ".tmp" / "scrutiny_summary.json"
WEB_OUTPUT = ROOT / "mr_website_public" / "mev" / "data" / "scrutiny_summary.json"

# Key reform HEs for highlighting
KEY_REFORM_HES = {
    'he-74-2023': 'Asumistukileikkaukset',
    'he-73-2023': 'Työttömyysturvareformi',
    'he-75-2023': 'Indeksijäädytykset',
    'he-13-2024': 'Sosiaaliturvan toinen aalto',
}


def main(args=None):
    if not INPUT.exists():
        print(f"Error: {INPUT} not found. Run detect_scrutiny_ignored.py --all first.")
        return

    with open(INPUT, encoding='utf-8') as f:
        analyses = json.load(f)

    print(f"Loaded {len(analyses)} HE scrutiny analyses")

    # Filter out errors
    valid = [a for a in analyses if 'error' not in a and a.get('total_experts_analyzed', 0) > 0]
    print(f"  {len(valid)} with valid results")

    # Aggregate
    total_experts = sum(a['total_experts_analyzed'] for a in valid)
    total_raw = sum(a['total_experts_raw'] for a in valid)
    agg_types = defaultdict(int)
    for a in valid:
        for t, n in a.get('by_response_type', {}).items():
            agg_types[t] += n

    print(f"\n=== AGGREGATE SCRUTINY STATISTICS ===")
    print(f"HEs analyzed: {len(valid)}")
    print(f"Expert statements: {total_raw} total, {total_experts} with mechanism content")
    for t in ['STRONG', 'ATTENTION', 'IMPLICIT', 'GRAVEYARD', 'TOPIC_ONLY', 'SILENT', 'NO_MIETINTO']:
        n = agg_types.get(t, 0)
        pct = n / max(total_experts, 1) * 100
        print(f"  {t:12s}: {n:5d} ({pct:5.1f}%)")

    ignored = agg_types.get('SILENT', 0) + agg_types.get('TOPIC_ONLY', 0)
    graveyard = agg_types.get('GRAVEYARD', 0)
    addressed = agg_types.get('STRONG', 0) + agg_types.get('ATTENTION', 0) + agg_types.get('IMPLICIT', 0)
    print(f"\n  Addressed:  {addressed:5d} ({addressed/max(total_experts,1)*100:.1f}%)")
    print(f"  Graveyard:  {graveyard:5d} ({graveyard/max(total_experts,1)*100:.1f}%)")
    print(f"  Ignored:    {ignored:5d} ({ignored/max(total_experts,1)*100:.1f}%)")

    # Per-HE table sorted by graveyard + silent rate
    he_table = []
    for a in valid:
        total = a['total_experts_analyzed']
        by_type = a.get('by_response_type', {})
        silent = by_type.get('SILENT', 0) + by_type.get('TOPIC_ONLY', 0)
        grave = by_type.get('GRAVEYARD', 0)
        strong = by_type.get('STRONG', 0)
        attention = by_type.get('ATTENTION', 0)
        deaf_rate = (silent + grave) / max(total, 1)

        he_table.append({
            'he_id': a['he_id'],
            'label': KEY_REFORM_HES.get(a['he_id'], ''),
            'n_experts': total,
            'n_raw': a['total_experts_raw'],
            'strong': strong,
            'attention': attention,
            'graveyard': grave,
            'silent': silent,
            'deaf_rate': round(deaf_rate, 2),
            'ignored_rate': round(a.get('scrutiny_ignored_rate', 0), 2),
            'has_mietinto': a.get('mietinto') is not None,
        })

    he_table.sort(key=lambda x: (-x['deaf_rate'], -x['n_experts']))

    print(f"\n=== TOP HEs BY DEAF RATE (graveyard + silent) ===")
    for row in he_table[:20]:
        label = f" ({row['label']})" if row['label'] else ''
        print(f"  {row['he_id']}{label}: "
              f"{row['n_experts']} experts, "
              f"deaf={row['deaf_rate']:.0%}, "
              f"STRONG={row['strong']}, GRAVE={row['graveyard']}, SILENT={row['silent']}")

    # Build output
    output = {
        'generated': '2026-03-16',
        'note': 'Scrutiny analysis: how expert testimony was handled by committee',
        'aggregate': {
            'n_hes': len(valid),
            'n_experts_analyzed': total_experts,
            'n_experts_raw': total_raw,
            'by_type': dict(agg_types),
            'addressed_pct': round(addressed / max(total_experts, 1) * 100, 1),
            'graveyard_pct': round(graveyard / max(total_experts, 1) * 100, 1),
            'ignored_pct': round(ignored / max(total_experts, 1) * 100, 1),
        },
        'he_table': he_table,
    }

    with open(OUTPUT, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nOutput: {OUTPUT}")

    WEB_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(WEB_OUTPUT, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False)
    print(f"Web: {WEB_OUTPUT}")


def run(**kwargs):
    """Standard pipeline API entry point."""
    main()


if __name__ == '__main__':
    main()

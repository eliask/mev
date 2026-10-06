"""
Tier 2 unreason detector: prediction vs actual outcomes.

Compares HE fiscal predictions with Kela/Tilastokeskus actuals.
Two modes:
  1. Curated: manually verified predictions with confounders (high confidence)
  2. Generalized: auto-scan all fiscal claims with keyword->Kela matching (broad, noisy)

Curated comparisons place a recorded prediction next to a later aggregate.
They are not causal accuracy labels: the aggregate can move against a real
effect when the counterfactual moves further. The broad scan is a noisy
candidate list, not an identified effect.

Reads:
    data/kela/kela_etuudet_maksetut.csv
    .tmp/he_enrichments.db (claims)

Writes:
    .tmp/longitudinal_strike.json         -- curated strikes (verified predictions)
    .tmp/longitudinal_strike_general.json -- generalized benefit-year aggregates
    .tmp/he_enrichments.db                -- longitudinal_strike table

Downstream: varjo-mev.html consumes longitudinal_strike.json for visualization.
Related: detect_unreason.py (Tier 0-1 detectors), extract_claims_llm.py (source claims)

Usage:
    uv run mev detector longitudinal
    uv run mev detector longitudinal --general-only
"""

import argparse
import csv
import json
import sqlite3
from collections import defaultdict

from mev.config import ROOT, ENRICHMENTS_DB

KELA_CSV = ROOT / "data" / "kela" / "kela_etuudet_maksetut.csv"
OUTPUT_CURATED = ROOT / ".tmp" / "longitudinal_strike.json"
OUTPUT_GENERAL = ROOT / ".tmp" / "longitudinal_strike_general.json"

# ---------------------------------------------------------------------------
# Keyword -> Kela benefit category mapping
# ---------------------------------------------------------------------------
KEYWORD_TO_KELA = {
    'asumistuk': 'Yleinen asumistuki',
    'yleisen asumistuen': 'Yleinen asumistuki',
    'toimeentulotuk': 'Perustoimeentulotuki',
    'perustoimeentulotu': 'Perustoimeentulotuki',
    'työttömyysturv': 'Työttömyysturva',
    'työttömyysetuuk': 'Työttömyysturva',
    'ansiopäiväraha': 'Työttömyysturva',
    'peruspäiväraha': 'Työttömyysturva',
    'työmarkkinatuk': 'Työttömyysturva',
    'sairauspäiväraha': 'Sairauspäivärahat',
    'kansaneläk': 'Eläke-etuudet (pl. takuueläke)',
    'takuueläk': 'Takuueläke',
    'eläkkeensaajan asumistuk': 'Eläkkeensaajan asumistuki',
    'lapsilisä': 'Lapsilisä',
    'lastenhoit': 'Lastenhoidon tuet',
    'kotihoidon tuk': 'Lastenhoidon tuet',
    'opintotuk': 'Opintotuki (opintoraha, asumislisä, lainatakaus)',
    'opintorahaa': 'Opintotuki (opintoraha, asumislisä, lainatakaus)',
    'vammaistuk': 'Vammaisetuudet',
    'kuntoutus': 'Kuntoutus',
    'lääkekorvau': 'Sairaanhoitokorvaukset',
}

# ---------------------------------------------------------------------------
# Curated predictions (from claims extraction, manually verified)
# ---------------------------------------------------------------------------
CURATED_PREDICTIONS = [
    {
        'he_id': 'he-74-2023',
        'title': 'Yleisen asumistuen leikkaukset',
        'claim_id': 'he-74-2023_claim_016',
        'benefit': 'Yleinen asumistuki',
        'metric': 'spending_change',
        'predicted_annual_eur': -355_000_000,
        'quote': 'Yleisen asumistuen muutokset vähentävät valtion menoja 355 miljoonaa euroa vuodessa.',
        'implementation_date': '2024-04',
        'confounders': [
            'Indeksijäädytys (HE 75/2023) leikkaa asumistukea erikseen',
            'HE 13/2024 toisen aallon leikkaukset',
            'Vuokramarkkinadynamiikka (vuokrat eivät laske samassa suhteessa)',
        ],
    },
    {
        'he_id': 'he-74-2023',
        'title': 'Toimeentulotuki-spillover asumistukileikkauksista',
        'claim_id': 'he-74-2023_claim_018',
        'benefit': 'Perustoimeentulotuki',
        'metric': 'spillover',
        'predicted_annual_eur': 70_500_000,
        'quote': 'Yleisen asumistuen muutokset lisäävät toimeentulotukimenoja 70,5 miljoonaa euroa vuodessa.',
        'implementation_date': '2024-04',
        'confounders': [
            'HE 73/2023 työttömyysturvaleikkaukset lisäävät toimeentulotukea myös',
            'HE 13/2024 toisen aallon leikkaukset lisäävät toimeentulotukea',
            'Puhdas erottelu: mahdotonta, koska kaikki vaikuttavat samanaikaisesti',
            'Mutta: HE 74 TIESI muiden reformien olevan tulossa ja arvioi silti 70.5M',
        ],
        'note': 'Cleanest comparison: HE predicted specific spillover, actual is direct observable.',
    },
    {
        'he_id': 'he-73-2023',
        'title': 'Työttömyysturvamuutosten säästöt',
        'claim_id': 'he-73-2023_claim_010',
        'benefit': 'Työttömyysturva',
        'metric': 'spending_change',
        'predicted_annual_eur': -250_000_000,
        'quote': 'Ehdotetut muutokset kohentavat julkista taloutta staattisesti noin 250 miljoonalla eurolla ilman käyttäytymisvaikutuksia.',
        'implementation_date': '2024-01',
        'confounders': [
            'Suhdannevaikutukset (talous heikentyi 2024)',
            'Staattinen arvio: ei sisällä käyttäytymisvaikutuksia (HE myöntää tämän)',
            'HE 13/2024 toisen aallon muutokset päällekkäin',
        ],
    },
    {
        'he_id': 'he-73-2023',
        'title': 'Työllisyysvaikutus',
        'claim_id': 'he-73-2023_claim_011',
        'benefit': None,
        'metric': 'employment',
        'predicted_annual_eur': None,
        'predicted_employment': 20_000,
        'quote': 'Työllistymisen kannustimien paranemisesta syntyy arviolta yli 20 000 henkilön työllisyyskasvu.',
        'implementation_date': '2024-01',
        'confounders': [
            'Suhdannevaikutukset (talous heikentyi 2024)',
            'Työllisyys laski noin 58 000:lla 2024 (TK työvoimatutkimus)',
            'Kausaliteetti: suhdanne vs. reformi erotettavissa vain osittain',
        ],
        'actual_note': 'Tilastokeskus työvoimatutkimus: työllisyys laski ~58K 2024. Suunta vastakkainen.',
    },
]


def load_kela_all() -> tuple[dict, dict]:
    """Load national Kela data: annual and monthly, keyed by benefit.

    Uses aikatyyppi='Vuosi' for annual totals and 'Kuukausi' for monthly,
    filtered to kunta_nro='000' (Koko maa = national). This avoids
    double-counting from municipality rows and annual summary rows.
    """
    annual = defaultdict(lambda: defaultdict(float))
    monthly = defaultdict(lambda: defaultdict(float))
    with open(KELA_CSV, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row['kunta_nro'] != '000':
                continue
            benefit = row['etuus']
            if row['aikatyyppi'] == 'Vuosi':
                annual[benefit][int(row['vuosi'])] = float(row['maksettu_eur'])
            elif row['aikatyyppi'] == 'Kuukausi':
                monthly[benefit][row['vuosikuukausi']] = float(row['maksettu_eur'])
    return dict(annual), dict(monthly)


def detect_partial_year(monthly_data: dict, year: int) -> bool:
    """Check if a year has incomplete data (< 10 months of monthly data)."""
    months = set()
    for benefit_months in monthly_data.values():
        for ym in benefit_months:
            if ym.startswith(str(year)):
                months.add(ym[4:])
    return len(months) < 10


def run_curated(annual_data: dict, monthly_data: dict) -> list:
    """Run curated prediction comparisons."""
    enr_conn = sqlite3.connect(str(ENRICHMENTS_DB))
    claim_sources = {}
    for pred in CURATED_PREDICTIONS:
        row = enr_conn.execute(
            "SELECT text, amount_eur FROM claim WHERE claim_id=?",
            (pred['claim_id'],)
        ).fetchone()
        if row:
            claim_sources[pred['claim_id']] = {'text': row[0], 'amount_eur': row[1]}
    enr_conn.close()
    print(f"  Verified {len(claim_sources)}/{len(CURATED_PREDICTIONS)} claims in enrichments DB")

    strikes = []
    for pred in CURATED_PREDICTIONS:
        benefit = pred['benefit']
        annual = annual_data.get(benefit, {})
        monthly = monthly_data.get(benefit, {})

        result = {
            'he_id': pred['he_id'],
            'title': pred['title'],
            'claim_id': pred['claim_id'],
            'quote': pred['quote'],
            'benefit': benefit,
            'metric': pred['metric'],
            'predicted_annual_eur': pred['predicted_annual_eur'],
            'implementation_date': pred['implementation_date'],
            'confounders': pred['confounders'],
            'claim_verified': pred['claim_id'] in claim_sources,
        }

        if benefit and pred['predicted_annual_eur'] is not None:
            baseline_year = 2023
            baseline = annual.get(baseline_year, 0)
            comparisons = []
            for year in (2024, 2025):
                actual = annual.get(year, 0)
                if actual == 0:
                    continue
                delta = actual - baseline
                predicted = pred['predicted_annual_eur']
                ratio = delta / predicted if predicted != 0 else None
                comparisons.append({
                    'year': year,
                    'baseline': round(baseline),
                    'actual': round(actual),
                    'delta': round(delta),
                    'predicted': predicted,
                    'ratio': round(ratio, 1) if ratio else None,
                    'direction_match': (delta > 0) == (predicted > 0) if predicted != 0 else None,
                })
            result['comparisons'] = comparisons

            impl_ym = pred['implementation_date'].replace('-', '')
            monthly_series = []
            for ym, amount in sorted(monthly.items()):
                if int(ym[:4]) >= 2022:
                    monthly_series.append({
                        'ym': ym,
                        'amount': round(amount),
                        'post_reform': ym >= impl_ym,
                    })
            result['monthly'] = monthly_series

        elif pred['metric'] == 'employment':
            result['predicted_employment'] = pred.get('predicted_employment')
            result['actual_note'] = pred.get('actual_note', '')

        if 'note' in pred:
            result['note'] = pred['note']

        strikes.append(result)

        if 'comparisons' in result:
            for c in result['comparisons']:
                ratio_str = f"{c['ratio']}x" if c['ratio'] else "N/A"
                direction = "SAME" if c.get('direction_match') else "OPPOSITE"
                print(f"  {result['title']} ({c['year']}): "
                      f"predicted {c['predicted']/1e6:+,.0f}M, "
                      f"actual {c['delta']/1e6:+,.0f}M "
                      f"({ratio_str}, direction: {direction})")
        elif result['metric'] == 'employment':
            print(f"  {result['title']}: predicted +{result.get('predicted_employment', '?'):,}, "
                  f"actual: {result.get('actual_note', 'unknown')}")

    return strikes


def run_generalized(annual_data: dict, monthly_data: dict) -> list:
    """Auto-scan all fiscal claims, aggregate per benefit-year, compare."""
    conn = sqlite3.connect(str(ENRICHMENTS_DB))
    claims = conn.execute('''
        SELECT claim_id, he_id, text, amount_eur, quote
        FROM claim
        WHERE claim_type='FISCAL' AND amount_eur IS NOT NULL AND abs(amount_eur) >= 1000
    ''').fetchall()
    conn.close()

    # Build per-HE per-benefit aggregation
    he_benefit = defaultdict(lambda: defaultdict(list))
    for cid, hid, text, eur, quote in claims:
        combined = ((text or '') + ' ' + (quote or '')).lower()
        matched = set()
        for kw, kela_cat in KEYWORD_TO_KELA.items():
            if kw in combined:
                matched.add(kela_cat)
        for benefit in matched:
            he_benefit[hid][benefit].append({'text': text, 'eur': eur, 'cid': cid})

    # Aggregate all HE predictions per benefit per implementation year
    benefit_year = defaultdict(lambda: defaultdict(list))
    for hid, benefits in he_benefit.items():
        parts = hid.split('-')
        he_year = int(parts[2])
        for benefit, bclaims in benefits.items():
            headline = max(bclaims, key=lambda c: abs(c['eur']))
            impl_year = he_year + 1
            benefit_year[benefit][impl_year].append({
                'he_id': hid, 'eur': headline['eur'],
                'text': headline['text'][:100], 'n_claims': len(bclaims),
            })

    # Detect partial years in Kela data
    partial_years = set()
    for year in range(2008, 2027):
        if detect_partial_year(monthly_data, year):
            partial_years.add(year)
    if partial_years:
        print(f"  Partial-year data detected for: {sorted(partial_years)} -- excluding from comparisons")

    rows = []
    for benefit in sorted(benefit_year):
        if benefit not in annual_data:
            continue
        annual = annual_data[benefit]

        for year in sorted(benefit_year[benefit]):
            if year in partial_years:
                continue
            preds = benefit_year[benefit][year]
            if year not in annual or year - 1 not in annual:
                continue

            sum_pred = sum(p['eur'] for p in preds)
            if abs(sum_pred) < 10_000_000:
                continue

            actual_delta = annual[year] - annual[year - 1]
            ratio = actual_delta / sum_pred if sum_pred != 0 else None
            direction_match = (actual_delta > 0) == (sum_pred > 0) if sum_pred != 0 else None

            if ratio is not None and abs(ratio) > 3:
                quality = 'MAGNITUDE_MISMATCH'
            elif direction_match is False:
                quality = 'DIRECTION_OPPOSITE'
            elif ratio is not None and 0.3 <= abs(ratio) <= 3:
                quality = 'ACCURATE'
            else:
                quality = 'NEGLIGIBLE'

            row = {
                'benefit': benefit,
                'year': year,
                'n_hes': len(preds),
                'sum_predicted': round(sum_pred),
                'actual_delta': round(actual_delta),
                'ratio': round(ratio, 2) if ratio is not None else None,
                'direction_match': direction_match,
                'quality': quality,
                'hes': [{
                    'he_id': p['he_id'], 'eur': p['eur'],
                    'text': p['text'], 'n_claims': p['n_claims'],
                } for p in preds],
            }
            rows.append(row)

    # Print summary
    n_total = len(rows)
    n_opposite = sum(1 for r in rows if r['quality'] == 'DIRECTION_OPPOSITE')
    n_big = sum(1 for r in rows if r['quality'] == 'MAGNITUDE_MISMATCH')
    n_accurate = sum(1 for r in rows if r['quality'] == 'ACCURATE')
    n_negligible = sum(1 for r in rows if r['quality'] == 'NEGLIGIBLE')

    print(f"\n  Generalized results: {n_total} benefit-year comparisons")
    print(f"    DIRECTION_OPPOSITE:  {n_opposite} ({n_opposite/n_total*100:.0f}%)" if n_total else "")
    print(f"    MAGNITUDE_MISMATCH:  {n_big} ({n_big/n_total*100:.0f}%)" if n_total else "")
    print(f"    ACCURATE:            {n_accurate} ({n_accurate/n_total*100:.0f}%)" if n_total else "")
    print(f"    NEGLIGIBLE:          {n_negligible} ({n_negligible/n_total*100:.0f}%)" if n_total else "")

    print(f"\n  Benefit-year detail:")
    for r in rows:
        tag = r['quality'][:3]
        print(f"    {r['benefit']:40s} {r['year']}  "
              f"pred={r['sum_predicted']/1e6:+8.0f}M  actual={r['actual_delta']/1e6:+10.0f}M  "
              f"ratio={r['ratio']:+7.1f}x  [{tag}]  ({r['n_hes']} HEs)")

    return rows


def store_curated(strikes: list):
    """Store curated strikes in enrichments DB."""
    enr_conn = sqlite3.connect(str(ENRICHMENTS_DB))
    enr_conn.execute('''CREATE TABLE IF NOT EXISTS longitudinal_strike (
        claim_id    TEXT PRIMARY KEY,
        he_id       TEXT NOT NULL,
        title       TEXT,
        benefit     TEXT,
        metric      TEXT,
        predicted   REAL,
        actual_2024 REAL,
        actual_2025 REAL,
        ratio_2024  REAL,
        ratio_2025  REAL,
        quote       TEXT,
        confounders TEXT,
        monthly_json TEXT
    )''')
    enr_conn.execute("DELETE FROM longitudinal_strike")

    for strike in strikes:
        actual_2024 = actual_2025 = ratio_2024 = ratio_2025 = None
        if 'comparisons' in strike:
            for c in strike['comparisons']:
                if c['year'] == 2024:
                    actual_2024, ratio_2024 = c['delta'], c['ratio']
                elif c['year'] == 2025:
                    actual_2025, ratio_2025 = c['delta'], c['ratio']

        enr_conn.execute(
            "INSERT INTO longitudinal_strike VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                strike['claim_id'], strike['he_id'], strike['title'],
                strike.get('benefit'), strike['metric'],
                strike.get('predicted_annual_eur'),
                actual_2024, actual_2025, ratio_2024, ratio_2025,
                strike['quote'],
                json.dumps(strike['confounders'], ensure_ascii=False),
                json.dumps(strike.get('monthly', []), ensure_ascii=False),
            )
        )

    # Store generalized results too
    enr_conn.execute('''CREATE TABLE IF NOT EXISTS longitudinal_general (
        benefit     TEXT NOT NULL,
        year        INTEGER NOT NULL,
        n_hes       INTEGER,
        sum_predicted REAL,
        actual_delta  REAL,
        ratio       REAL,
        direction_match INTEGER,
        quality     TEXT,
        hes_json    TEXT,
        PRIMARY KEY (benefit, year)
    )''')
    enr_conn.execute("DELETE FROM longitudinal_general")
    enr_conn.commit()
    enr_conn.close()


def store_generalized(rows: list):
    """Store generalized strikes in enrichments DB."""
    enr_conn = sqlite3.connect(str(ENRICHMENTS_DB))
    enr_conn.execute('''CREATE TABLE IF NOT EXISTS longitudinal_general (
        benefit     TEXT NOT NULL,
        year        INTEGER NOT NULL,
        n_hes       INTEGER,
        sum_predicted REAL,
        actual_delta  REAL,
        ratio       REAL,
        direction_match INTEGER,
        quality     TEXT,
        hes_json    TEXT,
        PRIMARY KEY (benefit, year)
    )''')
    enr_conn.execute("DELETE FROM longitudinal_general")

    for r in rows:
        enr_conn.execute(
            "INSERT INTO longitudinal_general VALUES (?,?,?,?,?,?,?,?,?)",
            (
                r['benefit'], r['year'], r['n_hes'],
                r['sum_predicted'], r['actual_delta'],
                r['ratio'], 1 if r['direction_match'] else 0,
                r['quality'],
                json.dumps(r['hes'], ensure_ascii=False),
            )
        )

    enr_conn.commit()
    enr_conn.close()


def main(args=None):
    if args is None:
        parser = argparse.ArgumentParser()
        parser.add_argument('--general-only', action='store_true',
                            help='Only run generalized scan, skip curated')
        args = parser.parse_args()

    print("Loading Kela data...")
    annual_data, monthly_data = load_kela_all()
    print(f"  {len(annual_data)} benefit categories loaded")

    if not args.general_only:
        print("\n=== CURATED PREDICTIONS ===")
        strikes = run_curated(annual_data, monthly_data)

        output = {
            'generated': '2026-03-16',
            'baseline_year': 2023,
            'data_source': 'Kela etuudet maksetut (kela_etuudet_maksetut.csv)',
            'note': 'Confounders are listed per prediction. Pure causal attribution impossible due to simultaneous reforms.',
            'strikes': strikes,
        }
        with open(OUTPUT_CURATED, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\nOutput: {OUTPUT_CURATED} ({OUTPUT_CURATED.stat().st_size:,} bytes)")

        store_curated(strikes)
        print("Stored in he_enrichments.db: longitudinal_strike table")

    print("\n=== GENERALIZED SCAN ===")
    rows = run_generalized(annual_data, monthly_data)

    output = {
        'generated': '2026-03-16',
        'method': 'Sum all HE fiscal predictions per Kela benefit per year, compare against actual annual delta',
        'note': 'Confounded: multiple reforms hit same benefit category. Direction is more reliable than magnitude. Partial years excluded.',
        'classification_explanation': {
            'DIRECTION_OPPOSITE': 'Aggregate predictions said UP, actual went DOWN (or vice versa)',
            'MAGNITUDE_MISMATCH': 'Same direction but >3x off',
            'ACCURATE': 'Same direction and within 0.3-3x',
            'NEGLIGIBLE': 'Prediction too small relative to actual delta',
        },
        'rows': rows,
    }
    with open(OUTPUT_GENERAL, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nOutput: {OUTPUT_GENERAL} ({OUTPUT_GENERAL.stat().st_size:,} bytes)")

    store_generalized(rows)
    print("Stored in he_enrichments.db: longitudinal_general table")


def run(**kwargs):
    """Standard detector API entry point."""
    main()


if __name__ == '__main__':
    main()

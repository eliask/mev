"""Tier 0 regex-based unreason detection in HE impact assessments.

Government's own words contradict themselves. No LLM needed.
Pure regex/SQL on existing atomized data.

12 Tier 0 detectors + 3 Tier 1 (claims-based) detectors.
See docstrings on each detect_* function for details.

Complementary to unreason.py (Tier 1 LLM-based, 3 semantic detectors).

Usage:
    mev detect unreason-regex
    mev detect unreason-regex --he he-108-2025
    mev detect unreason-regex --min-impact-chars 20000
"""

import argparse
import json
import re
import sqlite3
import sys

from mev.config import HE_DB_DIR, ENRICHMENTS_DB

# ---------------------------------------------------------------------------
# Tier 0 regex patterns
# ---------------------------------------------------------------------------

# Sign-crossing: range that includes both negative and positive
# Matches patterns like:
#   -700–+3 200,  -4 600–0,  -3 850–5 100
#   vaihteluväli -600 ... +2 400
#   vaihteluväli -X–Y  (where X negative, Y positive or zero)
RE_RANGE_CROSS = re.compile(
    r'(?:'
    # Pattern A: explicit negative–positive range with Unicode dash variants
    r'(-\s*[\d][\d\s,.]*)'           # negative bound (group 1)
    r'\s*[–\-—]\s*'                  # dash separator
    r'(\+?\s*[\d][\d\s,.]*)'         # positive/zero bound (group 2)
    r'|'
    # Pattern B: "vaihteluväli" followed by range
    r'vaihteluväli\s+'
    r'(-\s*[\d][\d\s,.]*)'           # negative bound (group 3)
    r'\s*[–\-—]\s*'
    r'(\+?\s*[\d][\d\s,.]*)'         # positive bound (group 4)
    r')',
    re.IGNORECASE
)

# Static model confessions
STATIC_CONFESSIONS = [
    'staattisia',
    'staattinen',
    'eivät sisällä käyttäytymisvaikutuksia',
    'ei sisällä käyttäytymisvaikutuksia',
    'ei huomioitu käyttäytymisvaikutu',
    'ei ole huomioitu käyttäytymisvaikutu',
    'olettaen ettei käyttäytyminen muutu',
    'ei ota huomioon käyttäytymisvaikutu',
]

# Confession patterns (superset of what enrich_atoms_llm.py uses)
# Expanded based on Gemini validation of HE 13/2024 which found missed phrasings
CONFESSION_PATTERNS = [
    'ei ole arvioitu', 'ei ole käytettävissä', 'ei ole huomioitu',
    'ei pystytä arvioimaan', 'ei voida arvioida', 'ei tiedetä',
    'arviointiin liittyy',
    'ei ole riittävästi tietoa', 'tarkkaa lukumäärää ei',
    'ei ole mahdollista arvioida', 'arvio on epävarma',
    'arvion luotettavuus', 'arviointi on puutteellinen',
    # Additional patterns found by Gemini validation:
    'ei pystytä erottamaan',     # "vaikutuksia ei pystytä erottamaan muista"
    'ei voida erottaa',          # "ei voida erottaa muista tekijöistä"
    'ei mahdollista',            # "ei mahdollista... huomioimista"
    'eivät tule täysin huomioiduksi',  # admits incomplete accounting
    'ei ole käytettävissä rekisteri',  # no register data available
    'liittyy suuria epävarmuuksia',    # "arviointiin liittyy suuria epävarmuuksia"
    'liittyy merkittäviä epävarmuuksia',
    'liittyy huomattavia epävarmuuksia',
    'mittaluokkaa ei ole mahdollista', # "vaikutuksen mittaluokkaa ei ole mahdollista arvioida"
]

# Failure/risk vocabulary — terms indicating the HE considers what could go wrong
# NOTE: baseline risk terms (köyhyysriski, irtisanomisriski) counted separately
FAILURE_TERMS_POLICY = [
    'epäonnistu', 'ei toimi',
    'epävarmuus', 'epävarmuuksia', 'epävarmuutta',
    'herkkyys', 'herkkyystarkas', 'herkkyysanalyysi',
    'pessimisti', 'worst case', 'pahimmillaan', 'huonoin',
    'negatiivinen skenaario', 'jos uudistus ei', 'jos tavoite ei',
    'jos oletus ei', 'ei toteudu',
    'riskinä on', 'riskinä voidaan', 'riskinä pidetään',
]
# These are about the POLICY potentially failing — not baseline social risks

# Baseline risk terms that DON'T indicate policy failure analysis
BASELINE_RISK_TERMS = [
    'köyhyysriski', 'syrjäytymisriski', 'irtisanomisriski',
    'terveysriski', 'riskitekijä', 'riskiarvio',
    'riskiperustei', 'riskiprofiili',
]

# Named microsimulation models (all caps = acronyms, case-sensitive)
MODEL_NAMES_CS = ['SISU', 'JUTTA', 'HVSR', 'TUJA', 'FLEED']
# Case-insensitive methodology terms
MODEL_TERMS_CI = [
    'mikrosimuloint', 'simulointimall',
    'herkkyysanalyysi', 'herkkyystarkas',
]
# Compiled patterns
RE_MODEL_CS = [re.compile(r'\b' + m + r'\b') for m in MODEL_NAMES_CS]
RE_MODEL_CI = [re.compile(t, re.IGNORECASE) for t in MODEL_TERMS_CI]


# ---------------------------------------------------------------------------
# Minimum becomes maximum patterns
# ---------------------------------------------------------------------------
RE_MIN_FLOOR = [
    re.compile(r'vähimmäismitoitu', re.IGNORECASE),
    re.compile(r'vähimmäismäärä', re.IGNORECASE),
    re.compile(r'minimimitoitu', re.IGNORECASE),
    re.compile(r'vähintään\s+\d+(?:[,\.]\d+)?\s*(?:hoitaja|henkilö|henkilöstö|työntekijä|asiakasta)', re.IGNORECASE),
    re.compile(r'(?:on|oltava|tulee\s+olla)\s+vähintään\s+\d', re.IGNORECASE),
]

RE_BUDGET_PRESSURE = [
    re.compile(r'säästöpaine', re.IGNORECASE),
    re.compile(r'budjettipaine', re.IGNORECASE),
    re.compile(r'menosopeutus', re.IGNORECASE),
    re.compile(r'hyvinvointialue.{0,40}säästö|säästö.{0,40}hyvinvointialue', re.IGNORECASE),
    re.compile(r'kuntien?\s+(?:säästö|leikkau|talousarv)', re.IGNORECASE),
    re.compile(r'rahoituksen\s+(?:väheneminen|leikkaus|supistuminen|tiukentumine)', re.IGNORECASE),
    re.compile(r'(?:resurssit|voimavarat)\s+(?:niukat|riittämättöm|vähenem)', re.IGNORECASE),
]

RE_FLOOR_ACK = [
    re.compile(r'(?:vähimmäis|minoitu).{0,60}(?:käytännössä|tosiasiassa).{0,60}(?:enimmäis|katto|maksimi)', re.IGNORECASE),
    re.compile(r'katto.{0,30}(?:vähimmäis|minimimitoitu)', re.IGNORECASE),
    re.compile(r'mitoituksen\s+katto', re.IGNORECASE),
]

# ---------------------------------------------------------------------------
# Juridical shield patterns
# ---------------------------------------------------------------------------
RE_MANDATORY_FRAMING = [
    re.compile(r'EU(?::n|n)?\s+valtiontukisäännösten?\s+(?:vuoksi|nojalla|perusteella|mukaan)', re.IGNORECASE),
    re.compile(r'komissio\s+on\s+(?:epävirallisesti\s+)?(?:esittänyt|todennut|ilmoittanut)', re.IGNORECASE),
    re.compile(r'kansainvälisten?\s+velvoitteiden?\s+(?:nojalla|vuoksi|perusteella|mukaan)', re.IGNORECASE),
    re.compile(r'perustuslakivaliokunta\s+edellytti', re.IGNORECASE),
    re.compile(r'sopimuksen\s+(?:mukaisesti|nojalla|perusteella)', re.IGNORECASE),
    re.compile(r'direktiivin?\s+(?:edellyttämä|vaatima|mukainen|pakottava)', re.IGNORECASE),
    re.compile(r'(?:EU|yhteisö)oikeus\w*\s+(?:edellyttää|vaatii|velvoittaa|estää)', re.IGNORECASE),
    re.compile(r'(?:ei\s+ole\s+mahdollista|mahdoton|este)\s+.{0,30}(?:EU|yhteisö|direktiivi|komissio)', re.IGNORECASE),
]

RE_VALIDITY_CHALLENGE = re.compile(
    r'oikeudellinen\s+arvio|tulkinta\s+on\s+epäselvä|tulkintakysymys|oikeustila\s+on\s+epäselvä',
    re.IGNORECASE
)
RE_HEALTH_SOCIAL = re.compile(r'terveys|hyvinvointi|sosiaali|lapsi|vanhus|potilas', re.IGNORECASE)

# ---------------------------------------------------------------------------
# Stale data extension patterns
# ---------------------------------------------------------------------------
RE_LOAD_BEARING = re.compile(
    r'riittävä|riittää|osoittaa|osoittaa\s+että|tukee\s+(?:päätöstä|esitystä|muutosta)|perustuu\s+(?:siihen|tähän)|perusteena\s+on',
    re.IGNORECASE
)

# ---------------------------------------------------------------------------
# Domain committee mismatch map
# ---------------------------------------------------------------------------
DOMAIN_COMMITTEE_MAP = {
    'asuminen': (re.compile(r'asumistuki|yleinen asumistuki|asumismenojen|asumiskustannusten', re.IGNORECASE), 'Sosiaali- ja terveysvaliokunta'),
    'toimeentulo': (re.compile(r'toimeentulotuki|perustoimeentulotuki|täydentävä toimeentulotuki', re.IGNORECASE), 'Sosiaali- ja terveysvaliokunta'),
    'vanhukset': (re.compile(r'vanhuspalvelu|vanhusten hoito|ikääntyneiden hoito|hoivahenkilöstö', re.IGNORECASE), 'Sosiaali- ja terveysvaliokunta'),
    'terveysverot': (re.compile(r'makeisvero|virvoitusjuomavero|valmistevero.*elintarvike', re.IGNORECASE), 'Valtiovarainvaliokunta')
}


# ---------------------------------------------------------------------------
# Promise clause FSM — single compiled regex, post-classified
# ---------------------------------------------------------------------------
# Each entry: (category, subtype, regex_fragment)
# All fragments are case-insensitive. Compiled into one alternation.
# Categories: confession, deferral, escape
# Gemini-validated against HE 241/2020, 73/2023, 57/2024

PROMISE_CLAUSE_SPECS = [
    # --- CONFESSION: impossibility family ---
    ('confession', 'impossibility', r'ei\s+voida\s+arvioida'),
    ('confession', 'impossibility', r'ei\s+ole\s+mahdollista\s+arvioida'),
    ('confession', 'impossibility', r'ei\s+pystytä\s+arvioimaan'),
    ('confession', 'impossibility', r'mahdoton\s+arvioida'),
    ('confession', 'impossibility', r'(?:on\s+)?(?:hyvin\s+)?vaikea\s+arvioida'),
    ('confession', 'impossibility', r'(?:on\s+)?haastavaa?\s+erott(?:aa|ella)'),
    ('confession', 'impossibility', r'ei\s+voida\s+pitää\s+luotettavana'),
    ('confession', 'impossibility', r'ei\s+voida\s+luotettavasti'),
    ('confession', 'impossibility', r'ei\s+ole\s+kaikilta\s+osin\s+ennakoida'),
    ('confession', 'impossibility', r'mittaluokkaa\s+ei\s+ole\s+mahdollista'),
    # --- CONFESSION: uncertainty family ---
    ('confession', 'uncertainty', r'sisältyy\s+(?:merkittävää\s+)?epävarmuutta'),
    ('confession', 'uncertainty', r'liittyy\s+(?:merkittäviä|suuria|huomattavia)\s+epävarmuuksia'),
    ('confession', 'uncertainty', r'liittyy\s+merkittävää\s+epävarmuutta'),
    ('confession', 'uncertainty', r'arvio\s+on\s+epävarma'),
    ('confession', 'uncertainty', r'arvion\s+luotettavuus'),
    ('confession', 'uncertainty', r'arviointi\s+on\s+puutteellinen'),
    # --- CONFESSION: dependency family ---
    ('confession', 'dependency', r'riippuvaisi?a?\s+.{0,30}päätöksist'),
    ('confession', 'dependency', r'riippuu\s+.{0,20}tekemättömistä'),
    ('confession', 'dependency', r'riippuu\s+olennaisesti'),
    # --- CONFESSION: unassessed family ---
    ('confession', 'unassessed', r'ei\s+ole\s+arvioitu(?:\s+aineistoperusteisesti)?'),
    ('confession', 'unassessed', r'ei\s+ole\s+pystytty\s+arvioimaan'),
    ('confession', 'unassessed', r'ei\s+ole\s+pystytty\s+(?:välttämättä\s+)?(?:vielä\s+)?tunnistamaan'),
    ('confession', 'unassessed', r'ei\s+ole\s+huomioitu'),
    ('confession', 'unassessed', r'eivät\s+tule\s+täysin\s+huomioiduksi'),
    ('confession', 'unassessed', r'ei\s+ole\s+käytettävissä\s+rekisteri'),
    ('confession', 'unassessed', r'ei\s+ole\s+riittävästi\s+tietoa'),
    ('confession', 'unassessed', r'tarkkaa\s+lukumäärää\s+ei'),

    # --- DEFERRAL: separate regulation ---
    ('deferral', 'separate_regulation', r'säädetään\s+erikseen'),
    ('deferral', 'decree', r'tarkemmin\s+.{0,15}asetuksella'),
    ('deferral', 'decree', r'tarkennetaan\s+.{0,30}asetuksella'),
    ('deferral', 'decree', r'tarkempia\s+(?:säännöksiä|määräyksiä)'),
    # --- DEFERRAL: future assessment ---
    ('deferral', 'future_assessment', r'arvioidaan\s+erikseen'),
    ('deferral', 'future_assessment', r'arvioidaan\s+myöhemmin'),
    ('deferral', 'future_assessment', r'selvitetään\s+myöhemmin'),
    ('deferral', 'future_assessment', r'selvitetään\s+erikseen'),
    # --- DEFERRAL: continued preparation ---
    ('deferral', 'continued_prep', r'jatkovalmistel\w+'),
    ('deferral', 'separate_proposal', r'erillisellä\s+(?:hallituksen\s+)?esityksellä'),
    # --- DEFERRAL: update promise ---
    ('deferral', 'update_promise', r'laskelmat\s+päivitetään'),
    ('deferral', 'update_promise', r'päivitetään\s+.{0,30}(?:aikana|ennen|vuoden)'),

    # --- ESCAPE: disclaimers ---
    ('escape', 'disclaimer', r'arvio\s+on\s+suuntaa-antava'),
    ('escape', 'disclaimer', r'karkeasti\s+(?:arvioiden|todeten|noin)'),
    ('escape', 'disclaimer', r'suuntaa[\s-]antav\w+\s+arvi'),
    # --- ESCAPE: approximation before numbers ---
    ('escape', 'approximation', r'(?:arviolta|suuruusluokaltaan)\s+(?:noin\s+)?\d'),
    # --- ESCAPE: aspiration ---
    ('escape', 'aspiration', r'(?:esityksellä|uudistuksella|muutoksella)\s+pyritään'),
    ('escape', 'aspiration', r'tavoitteena\s+on\b'),
    # --- ESCAPE: conditional ---
    ('escape', 'conditional', r'mahdollisuuksien\s+mukaan'),
    ('escape', 'conditional', r'voisi\s+mahdollistaa'),
    ('escape', 'conditional', r'vielä\s+tekemättömistä\s+päätöksistä'),
]

# Compile into single alternation regex with named groups
_promise_fragments = []
_promise_meta = []  # parallel list: (category, subtype) for each group
for _i, (_cat, _sub, _pat) in enumerate(PROMISE_CLAUSE_SPECS):
    _promise_fragments.append(f'(?P<p{_i}>{_pat})')
    _promise_meta.append((_cat, _sub))

RE_PROMISE_FSM = re.compile('|'.join(_promise_fragments), re.IGNORECASE)
PROMISE_META = _promise_meta  # indexed by group number


def scan_promise_clauses(texts: list[tuple[str, str]]) -> list[dict]:
    """Single-pass scan of all promise clause patterns across text atoms.

    Args: list of (atom_id, content) tuples.
    Returns: list of {category, subtype, match, atom_id, offset} dicts.
    """
    results = []
    for atom_id, text in texts:
        for m in RE_PROMISE_FSM.finditer(text):
            # Find which named group matched
            for gname, gval in m.groupdict().items():
                if gval is not None:
                    idx = int(gname[1:])  # p0, p1, ...
                    cat, sub = PROMISE_META[idx]
                    # context: 60 chars before and after
                    start = max(0, m.start() - 60)
                    end = min(len(text), m.end() + 60)
                    ctx = text[start:end].replace('\n', ' ').strip()
                    results.append({
                        'category': cat,
                        'subtype': sub,
                        'match': gval,
                        'context': ctx,
                        'atom_id': atom_id,
                        'offset': m.start(),
                    })
                    break  # only one group matches per alternation
    return results

# Implementation zero-cost patterns
IMPL_ZERO_PATTERNS = [
    'ei merkittäviä hallinnollis',
    'ei merkittäviä toimeenpano',
    'ei merkittäviä tietojärjestelmä',
    'katetaan nykyisistä',
    'nykyisten määrärahojen puitteissa',
    'ei edellytä lisämäärärahoja',
    'voidaan toteuttaa nykyis',
    'ei aiheuta merkittäviä kustannuksia',
    'ei aiheuta lisäkustannuksia',
]

# IT/system complexity indicators
IT_KEYWORDS = [
    'tietojärjestelmä', 'rajapinta', 'rekisteri',
    'automaatio', 'digitaali', 'tietokanta',
    'integraatio', 'sähköinen asiointi', 'verkkopalvelu',
    'tulorekisteri', 'kanta-palvelu',
]


RE_URL_CONTEXT = re.compile(
    r'https?://|urn:|doi:|ISBN|URN_ISBN|\.pdf|\.html|/pmc/|/til/|/documents/',
    re.IGNORECASE
)

# Range must be near estimation/impact language to count
RE_ESTIMATION_CONTEXT = re.compile(
    r'vaihteluväli|vaikutu|työllisy|miljoon|euroa|milj\.|henkeä|henkilö'
    r'|työllis|säästö|kustannu|meno|kasv|vähene|lisään|arvioi',
    re.IGNORECASE
)


def parse_number(s: str) -> float | None:
    """Parse Finnish-formatted number: '3 200' -> 3200, '1,5' -> 1.5"""
    s = s.strip().replace('\xa0', '').replace(' ', '')
    s = s.replace(',', '.')
    s = s.lstrip('+')
    try:
        return float(s)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Detector implementations
# ---------------------------------------------------------------------------

def detect_sign_cross(impact_texts: list[tuple[str, str]]) -> dict | None:
    """Find sensitivity ranges that cross zero in IMPACT atoms.

    Args: list of (atom_id, content) tuples for IMPACT atoms.
    Returns: flag dict or None.
    """
    findings = []
    for atom_id, text in impact_texts:
        for m in RE_RANGE_CROSS.finditer(text):
            # Extract bounds from whichever group matched
            lo_str = m.group(1) or m.group(3)
            hi_str = m.group(2) or m.group(4)
            if not lo_str or not hi_str:
                continue
            lo = parse_number(lo_str)
            hi = parse_number(hi_str)
            if lo is None or hi is None:
                continue
            # Does range cross zero?
            if lo < 0 and hi > 0:
                # Get wide context for URL/estimation check
                ctx_start = max(0, m.start() - 150)
                ctx_end = min(len(text), m.end() + 150)
                wide_context = text[ctx_start:ctx_end]

                # Skip if this looks like a URL/ISBN/DOI reference
                if RE_URL_CONTEXT.search(wide_context):
                    continue

                # Must be near estimation/impact language
                if not RE_ESTIMATION_CONTEXT.search(wide_context):
                    continue

                start = max(0, m.start() - 80)
                end = min(len(text), m.end() + 80)
                context = text[start:end].replace('\n', ' ').strip()
                findings.append({
                    'atom_id': atom_id,
                    'lo': lo,
                    'hi': hi,
                    'match': m.group(0).strip(),
                    'context': context,
                })

    if not findings:
        return None

    return {
        'detector': 'sign_cross',
        'severity': 3 if len(findings) >= 2 else 2,
        'evidence_atoms': json.dumps([f['atom_id'] for f in findings]),
        'evidence_text': '\n---\n'.join(
            f"[{f['lo']:+.0f} ... {f['hi']:+.0f}]: {f['context']}"
            for f in findings[:5]
        ),
        'meta': json.dumps({
            'n_crossings': len(findings),
            'ranges': [[f['lo'], f['hi']] for f in findings],
        }),
    }


def detect_confession_proceed(
    he_id: str,
    impact_texts: list[tuple[str, str]],
    enr_conn: sqlite3.Connection,
) -> dict | None:
    """Find HEs with many confessions AND definitive fiscal estimates.

    Uses the FSM promise clause scanner for richer pattern matching.
    """
    # Use FSM scanner for confessions
    all_clauses = scan_promise_clauses(impact_texts)
    confessions = [c for c in all_clauses if c['category'] == 'confession']

    confession_count = len(confessions)
    confession_atoms = list(dict.fromkeys(c['atom_id'] for c in confessions))
    confession_quotes = [c['context'] for c in confessions[:5]]

    if confession_count < 3:
        return None

    # Count quantified fiscal claims
    fiscal = enr_conn.execute(
        "SELECT COUNT(*) FROM claim WHERE he_id=? AND claim_type='FISCAL' AND amount_eur IS NOT NULL",
        (he_id,)
    ).fetchone()[0]

    if fiscal < 2:
        return None

    # Subtype breakdown for meta
    subtype_counts = {}
    for c in confessions:
        subtype_counts[c['subtype']] = subtype_counts.get(c['subtype'], 0) + 1

    severity = 3 if confession_count >= 8 and fiscal >= 5 else 2 if confession_count >= 5 else 1

    return {
        'detector': 'confession_proceed',
        'severity': severity,
        'evidence_atoms': json.dumps(confession_atoms[:10]),
        'evidence_text': '\n---\n'.join(confession_quotes[:5]),
        'meta': json.dumps({
            'confession_count': confession_count,
            'quantified_fiscal_claims': fiscal,
            'ratio': round(fiscal / max(confession_count, 1), 2),
            'subtypes': subtype_counts,
        }),
    }


def detect_zero_failure(impact_texts: list[tuple[str, str]]) -> dict | None:
    """Find large IMPACT analyses with zero failure/risk vocabulary.

    Distinguishes between policy failure analysis ("riskinä on, että uudistus...")
    and baseline risk description ("köyhyysriski"). Only the former counts.
    """
    total_chars = sum(len(t) for _, t in impact_texts)

    if total_chars < 30000:
        return None

    policy_failure_hits = 0
    baseline_risk_hits = 0
    for _, text in impact_texts:
        text_lower = text.lower()
        for term in FAILURE_TERMS_POLICY:
            policy_failure_hits += text_lower.count(term)
        for term in BASELINE_RISK_TERMS:
            baseline_risk_hits += text_lower.count(term)

    # Also count raw "riski" but subtract baseline compounds
    for _, text in impact_texts:
        text_lower = text.lower()
        raw_risk = text_lower.count('riski')
        # Subtract baseline risk compounds already counted
        baseline_in_text = sum(text_lower.count(t) for t in BASELINE_RISK_TERMS)
        net_risk = max(0, raw_risk - baseline_in_text)
        policy_failure_hits += net_risk

    # Normalized: hits per 10K chars
    hits_per_10k = policy_failure_hits / (total_chars / 10000) if total_chars > 0 else 0

    if hits_per_10k > 1.5:
        return None

    severity = 3 if total_chars >= 80000 and policy_failure_hits <= 2 else \
               2 if total_chars >= 50000 and hits_per_10k <= 0.5 else 1

    return {
        'detector': 'zero_failure',
        'severity': severity,
        'evidence_atoms': json.dumps([aid for aid, _ in impact_texts]),
        'evidence_text': (
            f"{total_chars:,} merkkiä vaikutusarviota, "
            f"{policy_failure_hits} mainintaa politiikkaepäonnistumisesta "
            f"({hits_per_10k:.2f} / 10 000 merkkiä). "
            f"Taustariskit (köyhyysriski ym.): {baseline_risk_hits}."
        ),
        'meta': json.dumps({
            'total_impact_chars': total_chars,
            'policy_failure_hits': policy_failure_hits,
            'baseline_risk_hits': baseline_risk_hits,
            'hits_per_10k': round(hits_per_10k, 3),
        }),
    }


def detect_static_fiscal(
    he_id: str,
    impact_texts: list[tuple[str, str]],
    enr_conn: sqlite3.Connection,
) -> dict | None:
    """Find HEs that confess static modeling but present fiscal estimates."""
    static_quotes = []
    static_atoms = []

    for atom_id, text in impact_texts:
        text_lower = text.lower()
        for pattern in STATIC_CONFESSIONS:
            idx = text_lower.find(pattern)
            if idx >= 0:
                start = max(0, idx - 60)
                end = min(len(text), idx + len(pattern) + 80)
                quote = text[start:end].replace('\n', ' ').strip()
                static_quotes.append(quote)
                if atom_id not in static_atoms:
                    static_atoms.append(atom_id)

    if not static_quotes:
        return None

    # Count fiscal claims with EUR amounts
    fiscal = enr_conn.execute(
        "SELECT COUNT(*) FROM claim WHERE he_id=? AND claim_type='FISCAL' AND amount_eur IS NOT NULL",
        (he_id,)
    ).fetchone()[0]

    if fiscal < 1:
        return None

    # Also check for behavioral claims (the model ignores behavior but predicts it)
    behavioral = enr_conn.execute(
        "SELECT COUNT(*) FROM claim WHERE he_id=? AND claim_type='BEHAVIORAL'",
        (he_id,)
    ).fetchone()[0]

    severity = 3 if fiscal >= 5 and behavioral >= 5 else 2 if fiscal >= 3 else 1

    return {
        'detector': 'static_fiscal',
        'severity': severity,
        'evidence_atoms': json.dumps(static_atoms),
        'evidence_text': '\n---\n'.join(static_quotes[:3]),
        'meta': json.dumps({
            'static_confession_count': len(static_quotes),
            'quantified_fiscal_claims': fiscal,
            'behavioral_claims': behavioral,
        }),
    }


def detect_impl_zero(
    he_id: str,
    impact_texts: list[tuple[str, str]],
    proposed_texts: list[tuple[str, str]],
    enr_conn: sqlite3.Connection,
) -> dict | None:
    """Find HEs claiming no implementation costs despite complexity signals."""
    # Check for zero-cost claims in IMPACT text
    zero_quotes = []
    zero_atoms = []

    for atom_id, text in impact_texts:
        text_lower = text.lower()
        for pattern in IMPL_ZERO_PATTERNS:
            idx = text_lower.find(pattern)
            if idx >= 0:
                start = max(0, idx - 40)
                end = min(len(text), idx + len(pattern) + 60)
                quote = text[start:end].replace('\n', ' ').strip()
                zero_quotes.append(quote)
                if atom_id not in zero_atoms:
                    zero_atoms.append(atom_id)

    if not zero_quotes:
        return None

    # Check complexity signals
    # A: delegation count from mechanism audits
    deleg_row = enr_conn.execute(
        "SELECT COALESCE(SUM(n_delegations), 0) FROM mechanism_audit WHERE he_id=?",
        (he_id,)
    ).fetchone()
    n_delegations = deleg_row[0] if deleg_row else 0

    # B: IT keywords in PROPOSED_SECTION atoms
    it_hits = 0
    for _, text in proposed_texts:
        text_lower = text.lower()
        for kw in IT_KEYWORDS:
            it_hits += text_lower.count(kw)

    # Also check IMPACT text for IT keywords
    for _, text in impact_texts:
        text_lower = text.lower()
        for kw in IT_KEYWORDS:
            it_hits += text_lower.count(kw)

    # Only flag if there are complexity signals
    complexity_score = n_delegations + it_hits
    if complexity_score < 3:
        return None

    severity = 3 if n_delegations >= 5 and it_hits >= 3 else \
               2 if complexity_score >= 5 else 1

    return {
        'detector': 'impl_zero',
        'severity': severity,
        'evidence_atoms': json.dumps(zero_atoms),
        'evidence_text': '\n---\n'.join(zero_quotes[:3]),
        'meta': json.dumps({
            'zero_cost_claims': len(zero_quotes),
            'n_delegations': n_delegations,
            'it_keyword_hits': it_hits,
            'complexity_score': complexity_score,
        }),
    }


def detect_epistemic_buffer(
    he_id: str,
    impact_texts: list[tuple[str, str]],
    enr_conn: sqlite3.Connection,
) -> dict | None:
    """Find HEs with simultaneous precision AND impossibility confession.

    The "epistemic buffer" pattern: present exact EUR figures to satisfy
    formal requirements, then disclaim them to preempt accountability.
    Flagged when an HE has BOTH:
      - Multiple confession clauses (impossibility/uncertainty)
      - Multiple deferral or escape clauses
    and quantified fiscal claims. The three together = structured irresponsibility.
    """
    all_clauses = scan_promise_clauses(impact_texts)
    confessions = [c for c in all_clauses if c['category'] == 'confession']
    deferrals = [c for c in all_clauses if c['category'] == 'deferral']
    escapes = [c for c in all_clauses if c['category'] == 'escape']

    n_conf = len(confessions)
    n_defer = len(deferrals)
    n_escape = len(escapes)

    # Need confessions + at least one other dimension
    if n_conf < 2 or (n_defer + n_escape) < 2:
        return None

    # Need quantified fiscal claims (the "precision" half)
    row = enr_conn.execute(
        "SELECT COUNT(*), COALESCE(SUM(ABS(amount_eur)), 0) "
        "FROM claim WHERE he_id=? AND claim_type='FISCAL' AND amount_eur IS NOT NULL",
        (he_id,)
    ).fetchone()
    n_fiscal = row[0]
    total_eur = row[1]

    if n_fiscal < 2 or total_eur < 1e6:
        return None

    total_clauses = n_conf + n_defer + n_escape

    # Severity: high when many confessions + large EUR + many escapes
    if total_clauses >= 15 and total_eur >= 1e9:
        severity = 3
    elif total_clauses >= 8 and total_eur >= 1e8:
        severity = 2
    else:
        severity = 1

    # Build evidence: pick one confession, one deferral/escape, show the tension
    sample_conf = confessions[0]['context'] if confessions else ''
    sample_other = (deferrals + escapes)[0]['context'] if (deferrals + escapes) else ''

    if total_eur >= 1e9:
        eur_str = f"{total_eur/1e9:.1f} mrd €"
    else:
        eur_str = f"{total_eur/1e6:.0f} M€"

    return {
        'detector': 'epistemic_buffer',
        'severity': severity,
        'evidence_atoms': json.dumps(
            list(dict.fromkeys(c['atom_id'] for c in confessions + deferrals + escapes))[:10]
        ),
        'evidence_text': (
            f"Episteeminen puskuri: {n_conf} tunnustusta, {n_defer} lykkäystä, "
            f"{n_escape} varaumaa — ja silti {n_fiscal} kvantifioitua fiskaaaliväitettä "
            f"(yhteensä {eur_str}).\n"
            f"Tunnustus: «{sample_conf}»\n"
            f"Varauma/lykkäys: «{sample_other}»"
        ),
        'meta': json.dumps({
            'n_confessions': n_conf,
            'n_deferrals': n_defer,
            'n_escapes': n_escape,
            'total_clauses': total_clauses,
            'n_fiscal_claims': n_fiscal,
            'total_eur': total_eur,
            'confession_subtypes': {
                sub: sum(1 for c in confessions if c['subtype'] == sub)
                for sub in set(c['subtype'] for c in confessions)
            },
        }),
    }


# Stale data: references to data from year X in HE from year Y where gap >= 2
RE_STALE_DATA = [
    # "vuoden 20XX" + data context words
    re.compile(r'vuoden\s+(20\d{2})\s+.{0,30}(?:aineisto|tieto|tilasto|luku|rekisteri|otokse|tutkimu|selvityks|tietokant|kulutustutkimu)', re.IGNORECASE),
    # "perustuu/perustuvat (vuoden) 20XX"
    re.compile(r'perustu\w+\s+(?:vuoden\s+)?(20\d{2})', re.IGNORECASE),
    # "20XX vuoden/aineistolla/tiedoilla/tilastojen"
    re.compile(r'\b(20\d{2})\s+(?:aineisto\w*|tieto\w+|tilasto\w+|luku\w+)', re.IGNORECASE),
    # "vuoden 20XX tasossa/hinnoin"
    re.compile(r'vuoden\s+(20\d{2})\s+(?:tasossa|hinnoin|hinnoissa|rahassa|arvossa)', re.IGNORECASE),
]


def detect_stale_data(
    he_id: str,
    impact_texts: list[tuple[str, str]],
) -> dict | None:
    """Find HEs where IMPACT section uses data 2+ years older than the HE.

    Pattern F5b from UNREASON_MASTERDOC. Pure regex, Tier 0.
    """
    # Extract HE year from id
    try:
        he_year = int(he_id.split('-')[-1])
    except (ValueError, IndexError):
        return None

    findings = []
    load_bearing = False

    for atom_id, text in impact_texts:
        atom_has_stale = False
        for pat in RE_STALE_DATA:
            for m in pat.finditer(text):
                # Extract data year from groups
                data_year = None
                for g in m.groups():
                    if g and g.isdigit() and len(g) == 4:
                        data_year = int(g)
                        break
                if not data_year:
                    continue
                gap = he_year - data_year
                if gap >= 2:
                    atom_has_stale = True
                    start = max(0, m.start() - 40)
                    end = min(len(text), m.end() + 40)
                    ctx = text[start:end].replace('\n', ' ').strip()
                    findings.append({
                        'atom_id': atom_id,
                        'data_year': data_year,
                        'gap': gap,
                        'context': ctx,
                    })
        
        if atom_has_stale and not load_bearing:
            if RE_LOAD_BEARING.search(text):
                load_bearing = True

    if not findings:
        return None

    max_gap = max(f['gap'] for f in findings)
    # Severity: 3 if gap >= 4 or multiple stale refs or load bearing, 2 if gap >= 3, 1 otherwise
    if load_bearing:
        severity = 3
    elif max_gap >= 4 or len(findings) >= 3:
        severity = 3
    elif max_gap >= 3 or len(findings) >= 2:
        severity = 2
    else:
        severity = 1

    return {
        'detector': 'stale_data',
        'severity': severity,
        'evidence_atoms': json.dumps(list(dict.fromkeys(f['atom_id'] for f in findings))[:10]),
        'evidence_text': '\n---\n'.join(
            f"[{f['data_year']} → HE {he_year}, kuilu {f['gap']}v]: {f['context']}"
            for f in sorted(findings, key=lambda x: -x['gap'])[:5]
        ),
        'meta': json.dumps({
            'n_stale_refs': len(findings),
            'max_gap_years': max_gap,
            'data_years': sorted(set(f['data_year'] for f in findings)),
            'he_year': he_year,
            'load_bearing': load_bearing,
        }),
    }


# Frozen parameter: law thresholds/rates set in old year, never updated
# Distinct from stale_data: this is about parameters baked INTO THE LAW, not analytical data age
# Strict patterns to avoid climate baseline false positives
RE_FROZEN_PARAM = [
    # "ei ole muutettu/korotettu/päivitetty/tarkistettu vuoden XXXX jälkeen"
    re.compile(
        r'ei\s+ole\s+(?:muutettu|korotettu|päivitetty|tarkistettu)\s+'
        r'(?:sen\s+jälkeen\s+kun|vuoden\s+|vuodesta\s+)'
        r'(\d{4})',
        re.IGNORECASE
    ),
    # "vuoden XXXX tasossa/tasoon" + EUR context (frozen thresholds)
    re.compile(
        r'vuoden\s+(\d{4})\s+tasoss?a?\b.{0,60}(?:euroa|€|tuloraja|enimmäismäärä|raja|verota)',
        re.IGNORECASE
    ),
    # "verota(so)ja ei ole muutettu vuoden XXXX"
    re.compile(
        r'(?:verot|valmistevero|tuloraja|enimmäismäärä)\w*\s+ei\s+ole\s+(?:muutettu|korotettu|tarkistettu)\s+vuode\w*\s+(\d{4})',
        re.IGNORECASE
    ),
    # "XXXX jälkeen ei ole korotettu/muutettu"
    re.compile(
        r'(\d{4})\s+jälkeen\s+(?:ei\s+ole\s+)?(?:muutettu|korotettu|päivitetty|tarkistettu)',
        re.IGNORECASE
    ),
    # "on pysynyt/säilynyt vuoden XXXX tasolla"
    re.compile(
        r'(?:on\s+)?(?:pysynyt|säilynyt)\s+vuoden\s+(\d{4})\s+tasolla',
        re.IGNORECASE
    ),
]

# Climate/emission baselines to exclude (legitimate references to base years)
RE_CLIMATE_EXCLUDE = re.compile(
    r'päästö|ilmasto|kasvihuone|hiilidioksidi|CO2|Kioton|Pariisin',
    re.IGNORECASE
)


def detect_frozen_parameter(
    he_id: str,
    impact_texts: list[tuple[str, str]],
) -> dict | None:
    """Find HEs where law parameters (thresholds, rates) are frozen at old year values.

    Pattern F5c: distinct from stale_data (which is about analytical data age).
    This is about parameters BAKED INTO THE LAW that were set in year X and
    never updated, even as the HE acknowledges the gap.
    """
    try:
        he_year = int(he_id.split('-')[-1])
    except (ValueError, IndexError):
        return None

    findings = []
    for atom_id, text in impact_texts:
        for pat in RE_FROZEN_PARAM:
            for m in pat.finditer(text):
                # Extract frozen year
                frozen_year = None
                for g in m.groups():
                    if g and g.isdigit() and len(g) == 4:
                        frozen_year = int(g)
                        break
                if not frozen_year:
                    continue

                gap = he_year - frozen_year
                if gap < 5:  # only flag if frozen 5+ years
                    continue

                # Check wide context for climate exclusion
                ctx_start = max(0, m.start() - 100)
                ctx_end = min(len(text), m.end() + 100)
                wide_ctx = text[ctx_start:ctx_end]
                if RE_CLIMATE_EXCLUDE.search(wide_ctx):
                    continue

                start = max(0, m.start() - 50)
                end = min(len(text), m.end() + 50)
                ctx = text[start:end].replace('\n', ' ').strip()
                findings.append({
                    'atom_id': atom_id,
                    'frozen_year': frozen_year,
                    'gap': gap,
                    'context': ctx,
                })

    if not findings:
        return None

    max_gap = max(f['gap'] for f in findings)
    # Severity: 3 if gap >= 15 or 3+ frozen params, 2 if gap >= 10, 1 otherwise
    if max_gap >= 15 or len(findings) >= 3:
        severity = 3
    elif max_gap >= 10 or len(findings) >= 2:
        severity = 2
    else:
        severity = 1

    return {
        'detector': 'frozen_parameter',
        'severity': severity,
        'evidence_atoms': json.dumps(list(dict.fromkeys(f['atom_id'] for f in findings))[:10]),
        'evidence_text': '\n---\n'.join(
            f"[parametri jäädytetty {f['frozen_year']}, kuilu {f['gap']}v]: {f['context']}"
            for f in sorted(findings, key=lambda x: -x['gap'])[:5]
        ),
        'meta': json.dumps({
            'n_frozen': len(findings),
            'max_gap_years': max_gap,
            'frozen_years': sorted(set(f['frozen_year'] for f in findings)),
            'he_year': he_year,
        }),
    }


def detect_minimum_becomes_maximum(
    he_id: str,
    impact_texts: list[tuple[str, str]],
    proposed_texts: list[tuple[str, str]],
) -> dict | None:
    """Find HEs that set a statutory minimum floor under budget pressure."""
    total_impact_chars = sum(len(t) for _, t in impact_texts)
    if total_impact_chars < 15000:
        return None

    # Check acknowledgment first to skip entirely
    for _, text in impact_texts:
        for ack_pat in RE_FLOOR_ACK:
            if ack_pat.search(text):
                return None

    floor_hits = []
    budget_hits = []

    # Check floor language in PROPOSED_SECTION and IMPACT
    for atom_id, text in proposed_texts + impact_texts:
        for pat in RE_MIN_FLOOR:
            for m in pat.finditer(text):
                start = max(0, m.start() - 60)
                end = min(len(text), m.end() + 60)
                ctx = text[start:end].replace('\n', ' ').strip()
                floor_hits.append((atom_id, ctx))

    # Check budget pressure in IMPACT
    for atom_id, text in impact_texts:
        for pat in RE_BUDGET_PRESSURE:
            for m in pat.finditer(text):
                start = max(0, m.start() - 60)
                end = min(len(text), m.end() + 60)
                ctx = text[start:end].replace('\n', ' ').strip()
                budget_hits.append((atom_id, ctx))

    n_floor_hits = len(floor_hits)
    n_budget_hits = len(budget_hits)

    if n_floor_hits == 0 or n_budget_hits == 0:
        return None

    if n_floor_hits >= 3 and n_budget_hits >= 3:
        severity = 3
    elif n_floor_hits >= 2 or n_budget_hits >= 2:
        severity = 2
    else:
        severity = 1

    first_floor_context = floor_hits[0][1] if floor_hits else ""
    first_budget_context = budget_hits[0][1] if budget_hits else ""

    return {
        'detector': 'minimum_becomes_maximum',
        'severity': severity,
        'evidence_atoms': json.dumps(list(set([a for a, _ in floor_hits] + [a for a, _ in budget_hits]))[:10]),
        'evidence_text': (
            "Lakisääteinen vähimmäistaso asetetaan ilman analyysiä siitä, että se muuttuu operatiiviseksi enimmäistasoksi budjettipaineen alla.\n"
            f"Lattia-ilmaukset ({n_floor_hits} kpl): {first_floor_context}\n"
            f"Budjettipaine ({n_budget_hits} kpl): {first_budget_context}"
        ),
        'meta': json.dumps({
            'n_floor_hits': n_floor_hits,
            'n_budget_hits': n_budget_hits,
        })
    }


def detect_juridical_shield(
    he_id: str,
    impact_texts: list[tuple[str, str]],
    enr_conn: sqlite3.Connection,
) -> dict | None:
    """Find HEs where decisions are framed as forced legal compliance."""
    mandatory_hits = []
    validity_challenge = False
    health_social_consequence = False

    for atom_id, text in impact_texts:
        for pat in RE_MANDATORY_FRAMING:
            for m in pat.finditer(text):
                start = max(0, m.start() - 60)
                end = min(len(text), m.end() + 60)
                ctx = text[start:end].replace('\n', ' ').strip()
                mandatory_hits.append((atom_id, ctx))

        if RE_VALIDITY_CHALLENGE.search(text):
            validity_challenge = True

        if RE_HEALTH_SOCIAL.search(text):
            health_social_consequence = True

    n_mandatory = len(mandatory_hits)
    if n_mandatory < 2:
        return None

    # Check consequence
    row = enr_conn.execute(
        "SELECT COALESCE(SUM(ABS(amount_eur)), 0) FROM claim WHERE he_id=? AND claim_type='FISCAL' AND amount_eur IS NOT NULL",
        (he_id,)
    ).fetchone()
    total_eur = row[0] if row else 0.0

    significant_consequence = total_eur >= 10e6 or health_social_consequence

    if not significant_consequence:
        return None

    # Determine severity
    if n_mandatory >= 4 and total_eur >= 100e6 and not validity_challenge:
        severity = 3
    elif n_mandatory >= 3 and significant_consequence and not validity_challenge:
        severity = 2
    else:
        severity = 1

    first_mandatory_context = mandatory_hits[0][1] if mandatory_hits else ""

    return {
        'detector': 'juridical_shield',
        'severity': severity,
        'evidence_atoms': json.dumps(list(set([a for a, _ in mandatory_hits]))[:10]),
        'evidence_text': (
            f"Juridinen kilpi: päätös esitetään pakotettuna noudattamisena, episteeminen laadunvalvonta ohitetaan. "
            f"{n_mandatory} pakollisviittausta. {first_mandatory_context}\n"
            "[Perusteena käytetty väite ei ole todennettu — vrt. UK SDIL 2018]"
        ),
        'meta': json.dumps({
            'n_mandatory_hits': n_mandatory,
            'total_eur': total_eur,
            'health_social': health_social_consequence,
            'validity_challenge': validity_challenge
        })
    }


def detect_domain_committee_mismatch(
    he_id: str,
    impact_texts: list[tuple[str, str]],
) -> dict | None:
    """Find HEs where the handling committee does not match the true domain of the bill."""
    matched_domains = []
    for domain_name, (pat, expected_comm) in DOMAIN_COMMITTEE_MAP.items():
        for _, text in impact_texts:
            if pat.search(text):
                matched_domains.append((domain_name, expected_comm))
                break

    if not matched_domains:
        return None

    he_db_path = HE_DB_DIR / f"{he_id}.db"
    if not he_db_path.exists():
        return None

    conn = sqlite3.connect(str(he_db_path))
    try:
        reports = conn.execute(
            "SELECT committee FROM committee_report WHERE report_type = 'Valiokunnan mietintö'"
        ).fetchall()
    except sqlite3.OperationalError:
        conn.close()
        return None

    if not reports:
        conn.close()
        return None

    actual_committee = reports[0][0]

    # Check if there is a mismatch
    mismatch_found = False
    mismatched_domain = None
    expected_committee = None

    for domain_name, expected_comm in matched_domains:
        if actual_committee != expected_comm:
            mismatch_found = True
            mismatched_domain = domain_name
            expected_committee = expected_comm
            break

    if not mismatch_found:
        conn.close()
        return None

    severity = 2
    try:
        # Check if table exists first
        has_table = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='scrutiny_summary'"
        ).fetchone()

        if has_table:
            # Corrected logic to check for 'ignored_pct' key in key-value table
            row = conn.execute(
                "SELECT value FROM scrutiny_summary WHERE key = 'ignored_pct'"
            ).fetchone()
            if row and float(row[0]) >= 0.7:
                severity = 3
    except Exception:
        pass
    finally:
        conn.close()

    return {
        'detector': 'domain_committee_mismatch',
        'severity': severity,
        'evidence_atoms': '[]',
        'evidence_text': (
            f"Toimialaohjaus: {mismatched_domain} → odotettu {expected_committee}, "
            f"käsitteli {actual_committee}. Rakenteellinen toimialasokeus, ei yksilövirhe."
        ),
        'meta': json.dumps({
            'domain': mismatched_domain,
            'expected_committee': expected_committee,
            'actual_committee': actual_committee,
        }),
    }


def detect_model_void(
    he_id: str,
    impact_texts: list[tuple[str, str]],
    enr_conn: sqlite3.Connection,
) -> dict | None:
    """Find HEs with large EUR claims but no named model or methodology.

    98% of HEs name no microsimulation model. This detector flags HEs where
    the fiscal stakes are high enough that model attribution should be expected.
    Tier 0: pure text scan + claims cross-reference.
    """
    # Check total EUR in fiscal claims
    row = enr_conn.execute(
        "SELECT COUNT(*), COALESCE(SUM(ABS(amount_eur)), 0) "
        "FROM claim WHERE he_id=? AND claim_type='FISCAL' AND amount_eur IS NOT NULL",
        (he_id,)
    ).fetchone()
    n_fiscal = row[0]
    total_eur = row[1]

    # Only flag if there are meaningful fiscal claims (>= 10M EUR total)
    if total_eur < 1e7 or n_fiscal < 2:
        return None

    # Scan IMPACT text for model names and methodology terms
    # findall with duplicates — ordered by occurrence
    model_hits = []
    for atom_id, text in impact_texts:
        for pat in RE_MODEL_CS:
            model_hits.extend(pat.findall(text))
        for pat in RE_MODEL_CI:
            model_hits.extend(pat.findall(text))

    # If model IS named, no flag (but hits are lost — callers who want
    # the positive case should run scan_models() separately)
    if model_hits:
        return None

    # Scale-based severity:
    # sev 3: 1B+ EUR with no model = serious
    # sev 2: 100M-1B EUR with no model = notable
    # sev 1: 10M-100M EUR with no model = mild
    if total_eur >= 1e9:
        severity = 3
    elif total_eur >= 1e8:
        severity = 2
    else:
        severity = 1

    # Format EUR for human reading
    if total_eur >= 1e9:
        eur_str = f"{total_eur/1e9:.1f} mrd €"
    else:
        eur_str = f"{total_eur/1e6:.0f} M€"

    return {
        'detector': 'model_void',
        'severity': severity,
        'evidence_atoms': json.dumps([aid for aid, _ in impact_texts[:5]]),
        'evidence_text': (
            f"{n_fiscal} fiskaalista väitettä yhteensä {eur_str}, "
            f"eikä vaikutusarviossa mainita yhtään laskentamallia "
            f"(ei SISU, JUTTA, HVSR, TUJA, FLEED, mikrosimulointimalli, herkkyysanalyysi). "
            f"Mistä luvut tulevat?"
        ),
        'meta': json.dumps({
            'n_fiscal_claims': n_fiscal,
            'total_eur': total_eur,
            'model_hits': [],
        }),
    }


# ---------------------------------------------------------------------------
# Tier 1 detectors
# ---------------------------------------------------------------------------

# Pattern for "has a meaningful number" in claim text (not just years or section refs)
RE_HAS_QUANTITY = re.compile(
    r'(?:'
    r'\d[\d\s,.]*\s*(?:milj|mrd|euroa|henk|työllis|henkilö|prosentti|%)'  # number + unit
    r'|\d{2,}[\d\s,.]*\s+(?:henkeä|työllisellä|henkilöllä|henkilöä)'     # large number + people
    r'|(?:noin|arviolta|yhteensä)\s+\d{3,}'                              # "noin 18 700"
    r')',
    re.IGNORECASE
)


def claim_is_quantified(claim_type: str, amount_eur, text: str) -> bool:
    """Check if a claim is quantified (EUR amount OR numbers in text)."""
    if amount_eur is not None:
        return True
    # For non-fiscal claims, check if the text contains numerical quantities
    if claim_type != 'FISCAL' and RE_HAS_QUANTITY.search(text):
        return True
    return False


def detect_precision_asymmetry(
    he_id: str,
    enr_conn: sqlite3.Connection,
) -> dict | None:
    """Find HEs where fiscal claims are quantified but behavioral aren't."""
    claims = enr_conn.execute(
        "SELECT claim_type, amount_eur, text FROM claim WHERE he_id=?",
        (he_id,)
    ).fetchall()

    if not claims:
        return None

    type_stats = {}
    for claim_type, amount_eur, text in claims:
        if claim_type not in type_stats:
            type_stats[claim_type] = {'n': 0, 'quantified': 0}
        type_stats[claim_type]['n'] += 1
        if claim_is_quantified(claim_type, amount_eur, text):
            type_stats[claim_type]['quantified'] += 1

    fiscal = type_stats.get('FISCAL', {'n': 0, 'quantified': 0})
    behavioral = type_stats.get('BEHAVIORAL', {'n': 0, 'quantified': 0})
    demographic = type_stats.get('DEMOGRAPHIC', {'n': 0, 'quantified': 0})

    if fiscal['n'] < 3 or (behavioral['n'] + demographic['n']) < 3:
        return None

    fiscal_pct = fiscal['quantified'] / fiscal['n'] if fiscal['n'] > 0 else 0
    other_n = behavioral['n'] + demographic['n']
    other_quantified = behavioral['quantified'] + demographic['quantified']
    other_pct = other_quantified / other_n if other_n > 0 else 0

    gap = fiscal_pct - other_pct
    if gap < 0.5:
        return None

    severity = 3 if gap >= 0.8 and other_n >= 10 else 2 if gap >= 0.6 else 1

    return {
        'detector': 'precision_asymmetry',
        'severity': severity,
        'evidence_atoms': json.dumps([]),
        'evidence_text': (
            f"FISCAL: {fiscal['quantified']}/{fiscal['n']} ({fiscal_pct:.0%}) kvantifioitu. "
            f"BEHAVIORAL+DEMOGRAPHIC: {other_quantified}/{other_n} ({other_pct:.0%}) kvantifioitu. "
            f"Ero: {gap:.0%}."
        ),
        'meta': json.dumps({
            'fiscal_n': fiscal['n'],
            'fiscal_quantified': fiscal['quantified'],
            'fiscal_pct': round(fiscal_pct, 3),
            'behavioral_n': behavioral['n'],
            'behavioral_quantified': behavioral['quantified'],
            'demographic_n': demographic['n'],
            'demographic_quantified': demographic['quantified'],
            'gap': round(gap, 3),
        }),
    }


def detect_compound_delegation(
    he_id: str,
    enr_conn: sqlite3.Connection,
) -> dict | None:
    """Find mechanism bundles with 3+ delegations AND fiscal claims."""
    bundles = enr_conn.execute(
        "SELECT bundle_id, statute_title, n_delegations, n_confessions "
        "FROM mechanism_audit WHERE he_id=? AND n_delegations >= 3",
        (he_id,)
    ).fetchall()

    if not bundles:
        return None

    # Check for fiscal claims in this HE
    fiscal = enr_conn.execute(
        "SELECT COUNT(*) FROM claim WHERE he_id=? AND claim_type='FISCAL' AND amount_eur IS NOT NULL",
        (he_id,)
    ).fetchone()[0]

    if fiscal < 1:
        return None

    total_deleg = sum(b[2] for b in bundles)

    severity = 3 if total_deleg >= 8 and fiscal >= 5 else \
               2 if total_deleg >= 5 else 1

    return {
        'detector': 'compound_delegation',
        'severity': severity,
        'evidence_atoms': json.dumps([b[0] for b in bundles]),
        'evidence_text': '\n'.join(
            f"{b[1]}: {b[2]} delegaatiota"
            for b in sorted(bundles, key=lambda x: -x[2])[:5]
        ),
        'meta': json.dumps({
            'total_delegations': total_deleg,
            'bundles_with_3plus': len(bundles),
            'quantified_fiscal_claims': fiscal,
            'statutes': [b[1] for b in bundles],
        }),
    }


def detect_temporal_asymmetry(
    he_id: str,
    enr_conn: sqlite3.Connection,
) -> dict | None:
    """Find HEs with precise short-term and vague long-term estimates."""
    claims = enr_conn.execute(
        "SELECT claim_type, amount_eur, time_horizon FROM claim WHERE he_id=?",
        (he_id,)
    ).fetchall()

    if len(claims) < 5:
        return None

    short_quantified = 0
    short_total = 0
    long_total = 0
    long_quantified = 0
    no_horizon = 0

    SHORT_TERMS = {'2024', '2025', '2026', '2027', '1 vuosi', '2 vuotta',
                   '3 vuotta', '4 vuotta', 'lyhyt', 'short', 'heti',
                   'välittömästi', 'vuonna 2024', 'vuonna 2025',
                   'vuonna 2026', 'vuonna 2027'}
    LONG_TERMS = {'pitkä', 'long', '10 vuotta', '20 vuotta', 'pysyvästi',
                  'rakenteellinen', 'pidemmällä aikavälillä'}

    for claim_type, amount, horizon in claims:
        h = (horizon or '').lower().strip()
        if not h:
            no_horizon += 1
            continue
        is_short = any(t in h for t in SHORT_TERMS)
        is_long = any(t in h for t in LONG_TERMS)
        if is_short:
            short_total += 1
            if amount is not None:
                short_quantified += 1
        elif is_long:
            long_total += 1
            if amount is not None:
                long_quantified += 1

    if short_total < 3 or long_total < 2:
        return None

    short_pct = short_quantified / short_total if short_total > 0 else 0
    long_pct = long_quantified / long_total if long_total > 0 else 0
    gap = short_pct - long_pct

    if gap < 0.3:
        return None

    severity = 2 if gap >= 0.6 and no_horizon >= 5 else 1

    return {
        'detector': 'temporal_asymmetry',
        'severity': severity,
        'evidence_atoms': json.dumps([]),
        'evidence_text': (
            f"Lyhyt aikaväli: {short_quantified}/{short_total} ({short_pct:.0%}) kvantifioitu. "
            f"Pitkä aikaväli: {long_quantified}/{long_total} ({long_pct:.0%}) kvantifioitu. "
            f"Ilman aikahorisonttia: {no_horizon} väitettä."
        ),
        'meta': json.dumps({
            'short_total': short_total,
            'short_quantified': short_quantified,
            'long_total': long_total,
            'long_quantified': long_quantified,
            'no_horizon': no_horizon,
            'gap': round(gap, 3),
        }),
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

ALL_DETECTORS = [
    # Tier 0 (text-based, per-HE DB)
    'sign_cross',
    'confession_proceed',
    'zero_failure',
    'static_fiscal',
    'impl_zero',
    'model_void',
    'epistemic_buffer',
    'stale_data',
    'frozen_parameter',
    'minimum_becomes_maximum',
    'juridical_shield',
    'domain_committee_mismatch',
    # Tier 1 (claims-based, enrichments DB)
    'precision_asymmetry',
    'compound_delegation',
    'temporal_asymmetry',
]


def run_detectors(he_id: str, enr_conn: sqlite3.Connection) -> list[dict]:
    """Run all detectors for one HE. Returns list of flag dicts."""
    he_db_path = HE_DB_DIR / f"{he_id}.db"
    if not he_db_path.exists():
        return []

    conn = sqlite3.connect(str(he_db_path))

    # Load IMPACT atoms
    impact_rows = conn.execute(
        "SELECT atom_id, content FROM atoms WHERE atom_type='IMPACT' AND length(content) > 0"
    ).fetchall()
    impact_texts = [(r[0], r[1]) for r in impact_rows]

    # Load PROPOSED_SECTION atoms
    proposed_rows = conn.execute(
        "SELECT atom_id, content FROM atoms WHERE atom_type='PROPOSED_SECTION' AND length(content) > 0"
    ).fetchall()
    proposed_texts = [(r[0], r[1]) for r in proposed_rows]

    conn.close()

    flags = []

    # Tier 0 detectors
    result = detect_sign_cross(impact_texts)
    if result:
        result['he_id'] = he_id
        flags.append(result)

    result = detect_confession_proceed(he_id, impact_texts, enr_conn)
    if result:
        result['he_id'] = he_id
        flags.append(result)

    result = detect_zero_failure(impact_texts)
    if result:
        result['he_id'] = he_id
        flags.append(result)

    result = detect_static_fiscal(he_id, impact_texts, enr_conn)
    if result:
        result['he_id'] = he_id
        flags.append(result)

    result = detect_impl_zero(he_id, impact_texts, proposed_texts, enr_conn)
    if result:
        result['he_id'] = he_id
        flags.append(result)

    result = detect_model_void(he_id, impact_texts, enr_conn)
    if result:
        result['he_id'] = he_id
        flags.append(result)

    result = detect_epistemic_buffer(he_id, impact_texts, enr_conn)
    if result:
        result['he_id'] = he_id
        flags.append(result)

    result = detect_stale_data(he_id, impact_texts)
    if result:
        result['he_id'] = he_id
        flags.append(result)

    result = detect_frozen_parameter(he_id, impact_texts)
    if result:
        result['he_id'] = he_id
        flags.append(result)

    result = detect_minimum_becomes_maximum(he_id, impact_texts, proposed_texts)
    if result:
        result['he_id'] = he_id
        flags.append(result)

    result = detect_juridical_shield(he_id, impact_texts, enr_conn)
    if result:
        result['he_id'] = he_id
        flags.append(result)

    result = detect_domain_committee_mismatch(he_id, impact_texts)
    if result:
        result['he_id'] = he_id
        flags.append(result)

    # Tier 1 detectors
    result = detect_precision_asymmetry(he_id, enr_conn)
    if result:
        result['he_id'] = he_id
        flags.append(result)

    result = detect_compound_delegation(he_id, enr_conn)
    if result:
        result['he_id'] = he_id
        flags.append(result)

    result = detect_temporal_asymmetry(he_id, enr_conn)
    if result:
        result['he_id'] = he_id
        flags.append(result)

    return flags


def create_table(enr_conn: sqlite3.Connection):
    """Create or recreate the unreason_flag table."""
    enr_conn.execute('''CREATE TABLE IF NOT EXISTS unreason_flag (
        he_id           TEXT NOT NULL,
        detector        TEXT NOT NULL,
        severity        INTEGER NOT NULL,
        evidence_atoms  TEXT,
        evidence_text   TEXT,
        meta            TEXT,
        PRIMARY KEY (he_id, detector)
    )''')
    enr_conn.execute('CREATE INDEX IF NOT EXISTS idx_unreason_he ON unreason_flag(he_id)')
    enr_conn.execute('CREATE INDEX IF NOT EXISTS idx_unreason_detector ON unreason_flag(detector)')
    enr_conn.execute('CREATE INDEX IF NOT EXISTS idx_unreason_severity ON unreason_flag(severity)')


def main():
    parser = argparse.ArgumentParser(description='Detect self-indicting unreason in HE impact assessments')
    parser.add_argument('he_id', nargs='?', help='Single HE to analyze (e.g., he-108-2025)')
    parser.add_argument('--min-impact-chars', type=int, default=0,
                        help='Only analyze HEs with at least N IMPACT chars')
    parser.add_argument('--clear', action='store_true',
                        help='Clear all existing flags before running')
    args = parser.parse_args()

    if not ENRICHMENTS_DB.exists():
        print(f"Error: {ENRICHMENTS_DB} not found")
        sys.exit(1)

    enr_conn = sqlite3.connect(str(ENRICHMENTS_DB))
    create_table(enr_conn)

    if args.clear:
        enr_conn.execute("DELETE FROM unreason_flag")
        enr_conn.commit()
        print("Cleared all existing flags")

    # Determine which HEs to process
    if args.he_id:
        he_ids = [args.he_id]
    else:
        # All HEs that have tagged sentences (sentence_tag covers 4253+ HEs vs atom_enrichment's 530)
        he_ids = sorted(set(
            r[0] for r in enr_conn.execute(
                "SELECT DISTINCT he_id FROM sentence_tag"
            ).fetchall()
        ))

    print(f"Running {len(ALL_DETECTORS)} detectors on {len(he_ids)} HEs...")

    total_flags = 0
    detector_counts = {d: 0 for d in ALL_DETECTORS}
    severity_counts = {1: 0, 2: 0, 3: 0}

    for i, he_id in enumerate(he_ids):
        flags = run_detectors(he_id, enr_conn)

        for flag in flags:
            enr_conn.execute(
                "INSERT OR REPLACE INTO unreason_flag "
                "(he_id, detector, severity, evidence_atoms, evidence_text, meta) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (flag['he_id'], flag['detector'], flag['severity'],
                 flag.get('evidence_atoms', '[]'),
                 flag.get('evidence_text', ''),
                 flag.get('meta', '{}'))
            )
            detector_counts[flag['detector']] += 1
            severity_counts[flag['severity']] += 1
            total_flags += 1

        if flags and args.he_id:
            for f in flags:
                meta = json.loads(f.get('meta', '{}'))
                print(f"  [{f['severity']}] {f['detector']}: {json.dumps(meta, indent=2, ensure_ascii=False)}")

        if (i + 1) % 100 == 0:
            enr_conn.commit()
            print(f"  ... {i+1}/{len(he_ids)} processed, {total_flags} flags so far")

    enr_conn.commit()
    enr_conn.close()

    print(f"\nResults: {total_flags} flags across {len(he_ids)} HEs")
    print(f"\nPer detector:")
    for d in ALL_DETECTORS:
        tier = '0' if d in ('sign_cross', 'confession_proceed', 'zero_failure', 'static_fiscal', 'impl_zero', 'model_void', 'epistemic_buffer', 'stale_data', 'frozen_parameter', 'minimum_becomes_maximum', 'juridical_shield', 'domain_committee_mismatch') else '1'
        print(f"  T{tier} {d:25s}: {detector_counts[d]:4d} HEs")
    print(f"\nPer severity:")
    for s in (1, 2, 3):
        label = {1: 'Lievä', 2: 'Merkittävä', 3: 'Vakava'}[s]
        print(f"  [{s}] {label:12s}: {severity_counts[s]:4d}")


def run(he_id: str | None = None, write_db: bool = True, force: bool = False,
        min_impact_chars: int = 0, clear: bool = False) -> dict:
    """Standard detector API entry point.

    Returns dict with total_flags, detector_counts, severity_counts.
    """
    if not ENRICHMENTS_DB.exists():
        print(f"Error: {ENRICHMENTS_DB} not found", file=sys.stderr)
        return {"total_flags": 0}

    enr_conn = sqlite3.connect(str(ENRICHMENTS_DB))
    create_table(enr_conn)

    if clear:
        enr_conn.execute("DELETE FROM unreason_flag")
        enr_conn.commit()

    if he_id:
        he_ids = [he_id if he_id.startswith('he-') else f'he-{he_id}']
    else:
        he_ids = sorted(set(
            r[0] for r in enr_conn.execute(
                "SELECT DISTINCT he_id FROM sentence_tag"
            ).fetchall()
        ))

    print(f"Running {len(ALL_DETECTORS)} detectors on {len(he_ids)} HEs...")

    total_flags = 0
    detector_counts = {d: 0 for d in ALL_DETECTORS}
    severity_counts = {1: 0, 2: 0, 3: 0}

    for i, hid in enumerate(he_ids):
        flags = run_detectors(hid, enr_conn)
        for flag in flags:
            enr_conn.execute(
                "INSERT OR REPLACE INTO unreason_flag "
                "(he_id, detector, severity, evidence_atoms, evidence_text, meta) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (flag['he_id'], flag['detector'], flag['severity'],
                 flag.get('evidence_atoms', '[]'),
                 flag.get('evidence_text', ''),
                 flag.get('meta', '{}'))
            )
            detector_counts[flag['detector']] += 1
            severity_counts[flag['severity']] += 1
            total_flags += 1
        if (i + 1) % 100 == 0:
            enr_conn.commit()
            print(f"  ... {i+1}/{len(he_ids)} processed, {total_flags} flags so far")

    enr_conn.commit()
    enr_conn.close()

    print(f"\nResults: {total_flags} flags across {len(he_ids)} HEs")
    for d in ALL_DETECTORS:
        if detector_counts[d]:
            print(f"  {d:25s}: {detector_counts[d]:4d}")

    return {"total_flags": total_flags, "detector_counts": detector_counts,
            "severity_counts": severity_counts}


if __name__ == '__main__':
    main()

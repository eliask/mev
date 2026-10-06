"""Format registry for LLM output format benchmarking.

Each OutputFormat bundles:
  - system_prompt: what the LLM sees
  - parse_fn: text → {sent_num: (role, qual, topic)}
  - example_output: for documentation / debugging

Registered formats (from previous experiment + new ones for dense model):
  baseline    — N R Q T\\n  (current tag.py format, space-separated, Finnish prompt)
  baseline_en — N R Q T\\n  (English version of baseline — tests language axis)
  compact     — NRQT\\n     (no spaces, 13% savings, Finnish prompt)
  compact_en  — NRQT\\n     (English version of compact — tests language axis)
  packed      — NRQT NRQT … (single line, same token count as compact)
  delta       — first NRQT, consecutive RQT, skip +NRQT
  delta_2char — 2-char codes: Role+Quality merged (A-Y), Topic as a-k

Format code conventions:
  Role    (5): P E V K L
  Quality (5): G M A H T
  Topic  (11): F W S C I N D Y J R X

Language and compactness are experimental dimensions. Test both language
variants against the selected model and reference population. The compact parser
rejects invalid role/quality/topic positions; its prompt examples include
multi-digit sentence IDs to make the positional format explicit.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

# ---------------------------------------------------------------------------
# Valid code sets (from tag.py)
# ---------------------------------------------------------------------------

VALID_ROLES    = set('PEVKL')
VALID_QUALS    = set('GMAHT')
VALID_TOPICS   = set('FWSCINDJRYX')

# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class OutputFormat:
    name: str
    description: str
    system_prompt: str
    parse_fn: Callable[[str], dict[int, tuple[str, str, str]]]
    example_output: str
    # For delta formats: approximate tokens per sentence (used in report)
    expected_tokens_per_sent: float = 4.6


# ---------------------------------------------------------------------------
# Parse helpers (shared)
# ---------------------------------------------------------------------------

def _clean_role(c: str) -> str | None:
    c = c.upper()
    return c if c in VALID_ROLES else None

def _clean_qual(c: str) -> str | None:
    c = c.upper()
    return c if c in VALID_QUALS else None

def _clean_topic(c: str) -> str | None:
    c = c.upper()
    return c if c in VALID_TOPICS else 'X'


# ---------------------------------------------------------------------------
# Format: baseline  — "N R Q T\n"
# ---------------------------------------------------------------------------

_BASELINE_SYSTEM = """\
Luokittele hallituksen esityksen lauseet kolmella ulottuvuudella.

ROOLI (P/E/V/K/L):
  P = premissi, taustatieto, nykytila
  E = estimaatti, laskelma, projektio
  V = väite ehdotetun muutoksen vaikutuksesta
  K = kaveat, rajoitteen tunnustus
  L = lupaus tulevasta seurannasta

LAATU (G/M/A/H/T):
  G = grounded, viittaa dataan
  M = mallinnettu, parametrit näkyvissä
  A = assertoitu ilman perustelua
  H = hedged ("arvioidaan", "noin")
  T = tunnustettu epävarma ("ei voida arvioida")

AIHE (F/W/S/C/I/N/D/Y/J/R/X):
  F = fiskaalinen  W = episteeminen  S = sosiaalinen
  C = kognitiivinen  I = institutionaalinen  N = infrastruktuuri
  D = inhimillinen  Y = yhtenäisyys  J = tarkoitus
  R = moraalinen  X = muu

Tulosta VAIN rivit: NUMERO ROOLI LAATU AIHE
Ei selityksiä, ei sulkeita, ei muuta tekstiä.
Jos lause ohitetaan (viittaus, otsikko), ÄLÄ tulosta sille riviä."""

def _parse_baseline(text: str) -> dict[int, tuple[str, str, str]]:
    """Parse 'N R Q T' lines. Robust to extra whitespace."""
    results: dict[int, tuple[str, str, str]] = {}
    for raw_line in text.split('\n'):
        line = raw_line.strip()
        if not line or line.upper() == 'NONE':
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            num = int(parts[0])
        except ValueError:
            continue
        role  = _clean_role(parts[1])
        qual  = _clean_qual(parts[2])
        topic = _clean_topic(parts[3])
        if role and qual:
            results[num] = (role, qual, topic or 'X')
    return results


# ---------------------------------------------------------------------------
# Format: compact  — "NRQT\n"
# ---------------------------------------------------------------------------

_COMPACT_SYSTEM = """\
Luokittele hallituksen esityksen lauseet kolmella ulottuvuudella.

ROOLI (P/E/V/K/L):
  P = premissi  E = estimaatti  V = väite  K = kaveat  L = lupaus

LAATU (G/M/A/H/T) — HUOM: vain kirjaimet G M A H T:
  G = grounded  M = mallinnettu  A = assertoitu  H = hedged  T = epävarma

AIHE (F/W/S/C/I/N/D/Y/J/R/X):
  F=fiskaalinen W=episteeminen S=sosiaalinen C=kognitiivinen I=institutionaalinen
  N=infrastruktuuri D=inhimillinen Y=yhtenäisyys J=tarkoitus R=moraalinen X=muu

Tulosta yksi rivi per luokiteltu lause: <numero><ROOLI><LAATU><AIHE>
Ei välejä, ei sulkeita. Ohita viittaukset/otsikot kokonaan (älä tulosta).
Esimerkkejä: 1PGF  9VAI  11KTX  15EHF"""

def _parse_compact(text: str) -> dict[int, tuple[str, str, str]]:
    """Parse 'NRQT' (no spaces) lines. Also handles 'NSKIP' gracefully."""
    results: dict[int, tuple[str, str, str]] = {}
    # Match: one or more digits, then exactly 3 uppercase letters
    for m in re.finditer(r'(\d+)([A-Z]{3})', text):
        num  = int(m.group(1))
        code = m.group(2)
        role  = _clean_role(code[0])
        qual  = _clean_qual(code[1])
        topic = _clean_topic(code[2])
        if role and qual and num not in results:
            results[num] = (role, qual, topic or 'X')
    return results


# ---------------------------------------------------------------------------
# Format: baseline_en  — English version of baseline (language axis test)
# ---------------------------------------------------------------------------

_BASELINE_EN_SYSTEM = """\
Classify each sentence from a Finnish government proposal (HE) on three dimensions.

ROLE (P/E/V/K/L):
  P = premise, background fact, current state
  E = estimate, calculation, projection
  V = claim about the effect of the proposed change
  K = caveat, limitation acknowledgment
  L = promise of future monitoring or update

QUALITY (G/M/A/H/T):
  G = grounded, cites data or source
  M = modeled, parameters visible
  A = asserted without justification
  H = hedged ("estimated", "approximately", "likely")
  T = admitted uncertain ("cannot be assessed", "impossible to estimate")

TOPIC (F/W/S/C/I/N/D/Y/J/R/X):
  F = fiscal  W = epistemic  S = social
  C = cognitive  I = institutional  N = infrastructure
  D = human capital  Y = coherence  J = purpose
  R = moral  X = other

Output ONLY lines: NUMBER ROLE QUALITY TOPIC
No explanations, no parentheses, no other text.
Skip references and headings entirely (do not output a line for them)."""

def _parse_baseline_en(text: str) -> dict[int, tuple[str, str, str]]:
    """Parse 'N R Q T' lines (English variant). Identical logic to Finnish baseline."""
    return _parse_baseline(text)


# ---------------------------------------------------------------------------
# Format: compact_en  — English version of compact (language axis test)
# ---------------------------------------------------------------------------

_COMPACT_EN_SYSTEM = """\
Classify each sentence from a Finnish government proposal on three dimensions.

ROLE (P/E/V/K/L): P=premise E=estimate V=claim K=caveat L=promise
QUALITY (G/M/A/H/T) — only letters G M A H T:
  G=grounded M=modeled A=asserted H=hedged T=uncertain
TOPIC (F/W/S/C/I/N/D/Y/J/R/X):
  F=fiscal W=epistemic S=social C=cognitive I=institutional
  N=infrastructure D=human Y=coherence J=purpose R=moral X=other

Output one line per classified sentence: <number><ROLE><QUALITY><TOPIC>
No spaces, no parentheses. Skip references and headings entirely (no output).
Examples: 1PGF  9VAI  11KTX  15EHF"""

def _parse_compact_en(text: str) -> dict[int, tuple[str, str, str]]:
    """Parse 'NRQT' (no spaces) lines (English variant). Identical logic to compact."""
    return _parse_compact(text)


# ---------------------------------------------------------------------------
# Format: packed  — "NRQT NRQT …" (single line)
# ---------------------------------------------------------------------------

_PACKED_SYSTEM = """\
Luokittele hallituksen esityksen lauseet kolmella ulottuvuudella.

ROOLI (P/E/V/K/L): P=premissi E=estimaatti V=väite K=kaveat L=lupaus
LAATU (G/M/A/H/T): G=grounded M=mallinnettu A=assertoitu H=hedged T=epävarma
AIHE (F/W/S/C/I/N/D/Y/J/R/X): F W S C I N D Y J R X

Tulosta KAIKKI luokitellut lauseet YHDELLÄ RIVILLÄ: <num><R><L><A> <num><R><L><A> ...
Käytä <num>- ohitettaville lauseille (viittaukset, otsikot).
Esimerkki: 1PGF 2PGF 3- 4EMF 5VHI"""

def _parse_packed(text: str) -> dict[int, tuple[str, str, str]]:
    """Parse space-packed 'NRQT' tokens from a single line. Same regex as compact."""
    results: dict[int, tuple[str, str, str]] = {}
    for m in re.finditer(r'(\d+)([A-Z]{3})', text):
        num  = int(m.group(1))
        code = m.group(2)
        role  = _clean_role(code[0])
        qual  = _clean_qual(code[1])
        topic = _clean_topic(code[2])
        if role and qual and num not in results:
            results[num] = (role, qual, topic or 'X')
    return results


# ---------------------------------------------------------------------------
# Format: delta  — first "NRQT", consecutive "RQT", skip "+NRQT"
# ---------------------------------------------------------------------------

_DELTA_SYSTEM = """\
Luokittele hallituksen esityksen lauseet kolmella ulottuvuudella.

ROOLI (P/E/V/K/L): P=premissi E=estimaatti V=väite K=kaveat L=lupaus
LAATU (G/M/A/H/T): G=grounded M=mallinnettu A=assertoitu H=hedged T=epävarma
AIHE (F/W/S/C/I/N/D/Y/J/R/X): F W S C I N D Y J R X

DELTA-ENKOODAUS — minimoi tulostettavat tokenit:
- Ensimmäinen lause: <N><R><Q><T>    esim. 3PGF
- Seuraava peräkkäinen: <R><Q><T>     esim. PGF  (numero implisiittinen +1)
- Hyppää N lausetta: +<N><R><Q><T>   esim. +2EMF  (ohita 2, luokittele sitten)

Esimerkkisyöte:
[3] Kuntien verotulot olivat 23 mrd.
[4] Sosiaali- ja terveysmenot olivat 21 mrd.
[7] Uudistus parantaisi hallinnon toimivuutta.
[8] Vaikutuksia ei voida arvioida.

Esimerkki oikea delta-tuloste:
3PGF PGF +2VHI KTX

Selitys: 3=PGF, 4=PGF, hyppää 5+6, 7=VHI, 8=KTX"""

def _parse_delta(text: str) -> dict[int, tuple[str, str, str]]:
    """Parse delta-encoded output.

    Grammar (space or newline separated tokens):
      N[RQT]     — absolute: sentence N
      [RQT]      — relative: prev_num + 1
      +N[RQT]    — skip: prev_num + N + 1  (skip N sentences, then classify)

    Robustness: falls back to parsing any NNNxyz tokens if delta decode fails.
    """
    results: dict[int, tuple[str, str, str]] = {}

    # Tokenise: split on whitespace/newlines
    tokens = text.split()
    current = 0   # last assigned sentence number

    for tok in tokens:
        tok = tok.strip()
        if not tok:
            continue

        # Pattern: +N followed by 3 uppercase letters  e.g. "+2EMF" or "+2 EMF"
        m_skip = re.match(r'^\+(\d+)([A-Z]{3})$', tok)
        if m_skip:
            skip = int(m_skip.group(1))
            code = m_skip.group(2)
            current = current + skip + 1
            role  = _clean_role(code[0])
            qual  = _clean_qual(code[1])
            topic = _clean_topic(code[2])
            if role and qual:
                results[current] = (role, qual, topic or 'X')
            continue

        # Pattern: N followed by 3 uppercase letters  e.g. "3PGF" or "14VHI"
        m_abs = re.match(r'^(\d+)([A-Z]{3})$', tok)
        if m_abs:
            current = int(m_abs.group(1))
            code = m_abs.group(2)
            role  = _clean_role(code[0])
            qual  = _clean_qual(code[1])
            topic = _clean_topic(code[2])
            if role and qual:
                results[current] = (role, qual, topic or 'X')
            continue

        # Pattern: exactly 3 uppercase letters  e.g. "PGF" (consecutive)
        m_rel = re.match(r'^([A-Z]{3})$', tok)
        if m_rel:
            current += 1
            code = m_rel.group(1)
            role  = _clean_role(code[0])
            qual  = _clean_qual(code[1])
            topic = _clean_topic(code[2])
            if role and qual:
                results[current] = (role, qual, topic or 'X')
            continue

        # Pattern: "+N" followed by whitespace then 3-letter code (split token case)
        # Handled by looking ahead — skip for now; the regex above covers merged form

    return results


# ---------------------------------------------------------------------------
# Format: delta_2char  — Role+Quality as single char A-Y, Topic as a-k
# ---------------------------------------------------------------------------
# Encoding table: 5 roles × 5 qualities = 25 combos → A-Y
# Topics: 11 combos → a-k

_ROLES_LIST   = list('PEVKL')
_QUALS_LIST   = list('GMAHT')
_TOPICS_LIST  = list('FWSCINDYJRX')

# Build encode/decode tables
_RQ_TO_CHAR: dict[tuple[str,str], str] = {}
_CHAR_TO_RQ: dict[str, tuple[str,str]] = {}
for _ri, _r in enumerate(_ROLES_LIST):
    for _qi, _q in enumerate(_QUALS_LIST):
        _c = chr(ord('A') + _ri * 5 + _qi)  # A-Y
        _RQ_TO_CHAR[(_r, _q)] = _c
        _CHAR_TO_RQ[_c] = (_r, _q)

_TOPIC_TO_CHAR: dict[str, str] = {t: chr(ord('a') + i) for i, t in enumerate(_TOPICS_LIST)}
_CHAR_TO_TOPIC: dict[str, str] = {v: k for k, v in _TOPIC_TO_CHAR.items()}

# Build the lookup table for the system prompt
_RQ_TABLE_LINES = []
for _ri, _r in enumerate(_ROLES_LIST):
    row = ' '.join(f"{_r}{_q}={_RQ_TO_CHAR[(_r,_q)]}" for _q in _QUALS_LIST)
    _RQ_TABLE_LINES.append(row)
_RQ_TABLE = '\n'.join(_RQ_TABLE_LINES)

_TOPIC_TABLE = ' '.join(f"{t}={_TOPIC_TO_CHAR[t]}" for t in _TOPICS_LIST)

_DELTA_2CHAR_SYSTEM = f"""\
Luokittele hallituksen esityksen lauseet. Käytä 2-merkkistä delta-enkoodausta.

Merkki 1 (iso kirjain A-Y) = ROOLI+LAATU yhdistettynä:
{_RQ_TABLE}

Merkki 2 (pieni kirjain a-k) = AIHE:
{_TOPIC_TABLE}

DELTA-ENKOODAUS:
- Ensimmäinen lause: <N><CH><topic>    esim. 3Aa
- Seuraava peräkkäinen: <CH><topic>    esim. Aa
- Hyppää N lausetta: +<N><CH><topic>   esim. +2Ga

Esimerkkituloste (3 perättäistä, sitten hyppy): 3Aa Aa +2Ca Tf"""

def _parse_delta_2char(text: str) -> dict[int, tuple[str, str, str]]:
    """Parse 2-char delta-encoded output (Role+Quality as A-Y, Topic as a-k)."""
    results: dict[int, tuple[str, str, str]] = {}
    tokens = text.split()
    current = 0

    for tok in tokens:
        tok = tok.strip()
        if not tok:
            continue

        # Pattern: +N followed by uppercase+lowercase  e.g. "+2Ga"
        m_skip = re.match(r'^\+(\d+)([A-Y])([a-k])$', tok)
        if m_skip:
            skip = int(m_skip.group(1))
            rq_char = m_skip.group(2)
            t_char  = m_skip.group(3)
            current = current + skip + 1
            rq = _CHAR_TO_RQ.get(rq_char)
            topic = _CHAR_TO_TOPIC.get(t_char, 'X')
            if rq:
                results[current] = (rq[0], rq[1], topic)
            continue

        # Pattern: N followed by uppercase+lowercase  e.g. "3Aa"
        m_abs = re.match(r'^(\d+)([A-Y])([a-k])$', tok)
        if m_abs:
            current = int(m_abs.group(1))
            rq_char = m_abs.group(2)
            t_char  = m_abs.group(3)
            rq = _CHAR_TO_RQ.get(rq_char)
            topic = _CHAR_TO_TOPIC.get(t_char, 'X')
            if rq:
                results[current] = (rq[0], rq[1], topic)
            continue

        # Pattern: uppercase+lowercase only  e.g. "Aa" (consecutive)
        m_rel = re.match(r'^([A-Y])([a-k])$', tok)
        if m_rel:
            current += 1
            rq_char = m_rel.group(1)
            t_char  = m_rel.group(2)
            rq = _CHAR_TO_RQ.get(rq_char)
            topic = _CHAR_TO_TOPIC.get(t_char, 'X')
            if rq:
                results[current] = (rq[0], rq[1], topic)
            continue

    return results


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

FORMATS: dict[str, OutputFormat] = {}

def _reg(fmt: OutputFormat) -> OutputFormat:
    FORMATS[fmt.name] = fmt
    return fmt

_reg(OutputFormat(
    name="baseline",
    description="N R Q T per line (current tag.py format, Finnish prompt)",
    system_prompt=_BASELINE_SYSTEM,
    parse_fn=_parse_baseline,
    example_output="3 P G F\n4 P G F\n6 E M F\n7 V H I\n8 K T X",
    expected_tokens_per_sent=4.6,
))

_reg(OutputFormat(
    name="baseline_en",
    description="N R Q T per line (English prompt — language axis test vs baseline)",
    system_prompt=_BASELINE_EN_SYSTEM,
    parse_fn=_parse_baseline_en,
    example_output="3 P G F\n4 P G F\n6 E M F\n7 V H I\n8 K T X",
    expected_tokens_per_sent=4.6,
))

_reg(OutputFormat(
    name="compact",
    description="NRQT per line, no spaces (Finnish prompt; fixed example covers 2-digit numbers)",
    system_prompt=_COMPACT_SYSTEM,
    parse_fn=_parse_compact,
    example_output="3PGF\n4PGF\n6EMF\n7VHI\n8KTX\n11KTX\n15EHF",
    expected_tokens_per_sent=4.0,
))

_reg(OutputFormat(
    name="compact_en",
    description="NRQT per line, no spaces (English prompt — language axis test vs compact)",
    system_prompt=_COMPACT_EN_SYSTEM,
    parse_fn=_parse_compact_en,
    example_output="3PGF\n4PGF\n6EMF\n7VHI\n8KTX\n11KTX\n15EHF",
    expected_tokens_per_sent=4.0,
))

_reg(OutputFormat(
    name="packed",
    description="NRQT NRQT ... on single line (same tokens as compact)",
    system_prompt=_PACKED_SYSTEM,
    parse_fn=_parse_packed,
    example_output="3PGF 4PGF 3- 6EMF 7VHI 8KTX",
    expected_tokens_per_sent=4.0,
))

_reg(OutputFormat(
    name="delta",
    description="Delta encoding: consecutive=RQT only, skip=+NRQT (60% theoretical savings)",
    system_prompt=_DELTA_SYSTEM,
    parse_fn=_parse_delta,
    example_output="3PGF PGF +2VHI KTX",
    expected_tokens_per_sent=2.2,
))

_reg(OutputFormat(
    name="delta_2char",
    description="Delta + 2-char codes (Role+Quality merged A-Y, Topic a-k) — 68% theoretical savings",
    system_prompt=_DELTA_2CHAR_SYSTEM,
    parse_fn=_parse_delta_2char,
    example_output="3Aa Aa +2Ca Tf",
    expected_tokens_per_sent=1.8,
))


def get_format(name: str) -> OutputFormat:
    """Retrieve a registered format by name. Raises KeyError if unknown."""
    return FORMATS[name]


def list_formats() -> list[str]:
    """Return names of all registered formats."""
    return list(FORMATS.keys())


# ---------------------------------------------------------------------------
# Encoding helpers (used for testing / gold construction)
# ---------------------------------------------------------------------------

# Reverse label maps (from tag.py human labels → codes)
_ROLE_CODE: dict[str, str] = {
    'premise': 'P', 'estimate': 'E', 'claim': 'V',
    'caveat': 'K', 'promise': 'L', 'anomaly': '!',
}
_QUAL_CODE: dict[str, str] = {
    'grounded': 'G', 'modeled': 'M', 'asserted': 'A',
    'hedged': 'H', 'uncertain': 'T', 'anomaly': '!',
}
_TOPIC_CODE: dict[str, str] = {
    'fiscal': 'F', 'epistemic': 'W', 'social': 'S',
    'cognitive': 'C', 'institutional': 'I', 'infrastructure': 'N',
    'human': 'D', 'coherence': 'Y', 'purpose': 'J',
    'moral': 'R', 'other': 'X',
}

def label_to_code(role_label: str, qual_label: str, topic_label: str) -> tuple[str, str, str]:
    """Convert human-readable labels (from DB) to single-char codes."""
    r = _ROLE_CODE.get(role_label, '?')
    q = _QUAL_CODE.get(qual_label, '?')
    t = _TOPIC_CODE.get(topic_label, 'X')
    return r, q, t

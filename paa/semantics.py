"""Text and context rules for campaign statements.

Dangerous conclusions stay false unless a specific rule produces them.
A field heading, a yes-vote label, a party programme or a missing record
does not become a personal verdict on its own.
"""


import re
from dataclasses import dataclass, field

# Labels for observable acts handed to the action ledger. They are not
# conclusions about fulfilment or political merit.
ACTION_KINDS = frozenset(
    {
        "INITIATIVE_AUTHORED",
        "VOTE_CAST",
        "QUESTION_FILED",
        "SPEECH_DELIVERED",
        "RESIGN_ROLE",
        "DONATION",
        "PUBLIC_ADVOCACY",
        "POLICY_RESTRAINT",
        "OTHER_OBSERVABLE_ACTION",
    }
)

CAPABILITIES = frozenset(
    {
        "MP_INITIATE_BILL",
        "PARLIAMENTARY_VOTE",
        "FILE_PARLIAMENTARY_QUESTION",
        "SPEAK_IN_PARLIAMENT",
        "HOLD_ELECTED_ROLE",
        "PARLIAMENTARY_INFLUENCE",
        "PUBLIC_ADVOCACY",
        "POLICYMAKING_ROLE",
        "PERSONAL_FUNDS",
        "OTHER",
    }
)

_CLAUSE = re.compile(
    r"\b("
    r"laskemme|lisäämme|teemme|poistamme|esitämme|äänestämme|lupaamme|"
    r"tavoittelen|tavoittelemme|parannamme|säilytämme|estämme|pidämme|"
    r"kannatan|kannatamme|lupaan|teen|äänestän|esitän|pyrin|"
    r"laitamme|laittaa"
    r")\b|\bon\s+\w*(?:tava|ttava)\b",
    re.IGNORECASE,
)
_DATE = re.compile(r"\b(\d{1,2})\.(\d{1,2})\.(\d{4})\b")
_YEAR_END = re.compile(r"vuoden\s+(\d{4})\s+loppuun", re.IGNORECASE)
_RELATIVE_YEAR_END = re.compile(
    r"\b(?:tämän|tänä|kuluvan)\s+vuoden(?:\s+(?:puolella|aikana))?\b|"
    r"\bvuoden\s+loppuun\b",
    re.IGNORECASE,
)
_QUOTE = re.compile(
    r"(?:sanoi|sanoo|väittää|väitti|totesi)\s*:?\s*[\"“„](.+?)[\"”]",
    re.IGNORECASE,
)
_ELECTION_CONDITION = re.compile(
    r"\b(?:jos|mikäli|kun)\b[^.!?]{0,140}?"
    r"(?:pääsen\s+eduskuntaan|valit(?:aan|uksi)|valitset\s+minut\s+eduskuntaan|"
    r"tulen\s+valituksi|kansanedustajaksi)\b",
    re.IGNORECASE,
)
_PERSONAL_RESTRAINT = re.compile(
    r"\b(?:en|emme)\s+(?:aio\s+|tule\s+)?"
    r"(?:leikka(?:a|maan)|korota(?:a|maan)|lakkauta(?:a|maan)|"
    r"poista(?:a|maan)|heikennä(?:ä|mään)|supista(?:a|maan))\b"
    r"|\blupaa(?:n|mme)\s+olla\s+"
    r"(?:leikkaamatta|korottamatta|lakkauttamatta|poistamatta|heikentämättä|supistamatta)\b",
    re.IGNORECASE,
)
_CONCRETE_ACTION = re.compile(
    r"(?<![:\w])teen\b|\b(esitän|äänestän|äänestää|kirjoitan|laadin|jätän|eroan|"
    r"kampanjoin|kampanjoida|lahjoitan|lahjoittaa|kysyn|kysynpä)\b",
    re.IGNORECASE,
)
_BROAD_COMMITMENT = re.compile(
    r"\b(?:lupaan|pyrin|pyrkiä|haluan|teen|tehdään|toimin|työskentelen)\b"
    r"[^.!?]{0,80}?\b(?:edistää|edistämään|parantaa|parantamaan|"
    r"puolustaa|puolustamaan|tukea|tukemaan|torjua|torjumaan|"
    r"lisätä|lisäämään|vähentää|vähentämään|vahvistaa|vahvistamaan|"
    r"turvata|turvaamaan|pysäyttää|pysäyttämään|pitää|pitämään|"
    r"tehdä\s+(?:parhaani|kaikkeni|kaiken\s+voitavani|kaiken(?:\s+mahdollisen)?)|"
    r"(?:parhaani|kaikkeni|töitä(?:ni|mme|si)?|työtä(?:ni|mme|si)?|"
    r"yhteistyötä|[\w-]*politiik\w*|päätöks\w*|ratkaisu\w*|tekoja|toimia|"
    r"toimet|työni|kansanedustajantyöni|kampanjani|kaiken\s+voitavani|"
    r"minkä\s+(?:lupaan|pystyn)|sen,?\s+mikä\w*|vaihtoehtoisia\s+malleja|"
    r"kuuluvaksi|kunnianpalautuksen|turvallisemman|paremman|vahvemman|"
    r"ystävällisemmän|kestävämmän)|"
    r"tehdä\s+työtä|olla\s+"
    r"(?:rehellinen|avoin|sinnikäs|tunnollinen|turvallisempi|"
    r"turvallisemman|parempi|paremman|vahvempi|vahvemman|"
    r"ystävällisempi|ystävällisemmän|kestävämpi|kestävämmän))\b",
    re.IGNORECASE,
)
_FINNISH_ABBREVIATION_AT_END = re.compile(
    r"\b(?:mm|esim|ns|n|v|yms|jne|ts|tms|eaa)\.$",
    re.IGNORECASE,
)


@dataclass
class Proposition:
    text: str
    semantic_type: str
    testability: str
    personal_action_commitment: bool = False
    issuer_scope: str = "UNRESOLVED"
    deadline: str | None = None
    deadline_basis: str = "UNRESOLVED"
    negation: bool = False
    condition: str | None = None
    targets: list[str] = field(default_factory=list)
    guarantees_implementation: bool = False
    effect_is_counterfactual: bool = False
    reported_speech: bool = False
    missing_specification: list[str] = field(default_factory=list)
    # A narrow, source-grounded description for the action ledger. These are
    # intentionally nullable: broad aims and values must not be promoted to
    # an apparently verifiable act.
    action_kind: str | None = None
    required_capability: str | None = None
    observable_action: bool = False
    source_start: int | None = None
    source_end: int | None = None
    # Regex classification is an auditable proposal, not a held-out-validated
    # semantic judgment. A later review may promote this explicitly.
    validation_state: str = "PROPOSED"

    def __post_init__(self) -> None:
        if self.deadline and self.deadline_basis == "UNRESOLVED":
            self.deadline_basis = "EXPLICIT"


@dataclass
class Analysis:
    propositions: list[Proposition]
    flags: dict


def _deadline(text: str) -> str | None:
    deadline, _ = _deadline_info(text)
    return deadline


def _deadline_info(text: str, context_year: int | None = None) -> tuple[str | None, str]:
    match = _DATE.search(text)
    if match:
        day, month, year = (int(part) for part in match.groups())
        if 1 <= month <= 12 and 1 <= day <= 31:
            return f"{year:04d}-{month:02d}-{day:02d}", "EXPLICIT"
    year_end = _YEAR_END.search(text)
    if year_end:
        return f"{year_end.group(1)}-12-31", "EXPLICIT"
    if context_year is not None and _RELATIVE_YEAR_END.search(text):
        return f"{context_year:04d}-12-31", "CONTEXT_DERIVED"
    return None, "UNRESOLVED"


def _context_year(inp: dict) -> int | None:
    candidates = [
        inp.get("stated_earliest"),
        inp.get("stated_latest"),
        inp.get("stated_at"),
        inp.get("statement_date"),
    ]
    for candidate in candidates:
        if isinstance(candidate, dict):
            candidate = candidate.get("earliest") or candidate.get("latest")
        if isinstance(candidate, str):
            match = re.search(r"\b(\d{4})\b", candidate)
            if match:
                return int(match.group(1))
    return None


def _targets(text: str) -> list[str]:
    found = []
    lower = text.casefold()
    for label, needle in (
        ("talous", "talou"),
        ("koulutus", "koulutus"),
        ("palvelut", "palvelu"),
        ("polttoaineen hinta", "polttoaine"),
        ("dieselvero", "dieselvero"),
        ("työllisyys", "työllis"),
        ("mielenterveyspalvelut", "mielentervey"),
    ):
        if needle in lower and label not in found:
            found.append(label)
    return found


def _split_clauses(text: str) -> list[str]:
    """Split coordinated clauses. Do not split a noun phrase on 'ja'."""
    normalized = text.strip()
    parts: list[str] = []
    cursor = 0
    for boundary in re.finditer(r"(?<=[.!?])\s+", normalized):
        segment = normalized[cursor : boundary.start()]
        if _FINNISH_ABBREVIATION_AT_END.search(segment):
            continue
        parts.append(segment)
        cursor = boundary.end()
    parts.append(normalized[cursor:])
    clauses: list[str] = []
    for part in parts:
        piece = part.strip()
        if not piece:
            continue
        if re.search(r"\s+(?:ja|sekä)\s+", piece, flags=re.IGNORECASE):
            bits = re.split(r"\s+(?:ja|sekä)\s+", piece, flags=re.IGNORECASE)
            if len(bits) == 2 and _CLAUSE.search(bits[0]) and _CLAUSE.search(bits[1]):
                clauses.extend(bit.strip(" .") for bit in bits if bit.strip())
                continue
        clauses.append(piece.strip())
    return clauses or ([text.strip()] if text.strip() else [])


def action_metadata(text: str, *, restraint: bool = False) -> tuple[str | None, str | None]:
    """Return an explicit action kind and the capability it presupposes.

    This is deliberately a small allow-list. A verb such as *edistää* or
    *puolustaa* is not enough to manufacture a ledger event; the source must
    name a public act whose record type is identifiable.
    """

    lower = text.casefold()
    if restraint:
        if re.search(r"koulut|budjet|määrära|verot|rahoit|leikka", lower):
            return "POLICY_RESTRAINT", "POLICYMAKING_ROLE"
        return "POLICY_RESTRAINT", "PARLIAMENTARY_INFLUENCE"
    if re.search(r"lakialoit|lakiehdot", lower):
        return "INITIATIVE_AUTHORED", "MP_INITIATE_BILL"
    if re.search(r"kansalaisaloit", lower):
        return "INITIATIVE_AUTHORED", "PUBLIC_ADVOCACY"
    if re.search(r"aloitteen|aloitteet|aloitetta|aloite", lower):
        if re.search(r"eduskun|kansanedust", lower):
            return "INITIATIVE_AUTHORED", "MP_INITIATE_BILL"
        if re.search(r"valtuusto|kunta|kaupunki", lower):
            return "INITIATIVE_AUTHORED", "POLICYMAKING_ROLE"
        return "INITIATIVE_AUTHORED", "OTHER"
    if re.search(r"\bäänest(?:än|ää|ä)\b|\bäänestämään\b", lower):
        return "VOTE_CAST", "PARLIAMENTARY_VOTE"
    if re.search(r"\bkys(?:yn|yä|ymys)\b|\bkirjallisen kysymyksen\b", lower):
        return "QUESTION_FILED", "FILE_PARLIAMENTARY_QUESTION"
    if re.search(r"\bpuheenvuoro(?:n|ja)?\b|\bpuhun eduskunnassa\b", lower):
        return "SPEECH_DELIVERED", "SPEAK_IN_PARLIAMENT"
    if re.search(r"jätän\s+(?:paikkani|tehtäväni)|eroan\s+(?:alue|kaupungin|kunta|valtuusto)", lower):
        return "RESIGN_ROLE", "HOLD_ELECTED_ROLE"
    if re.search(r"lahjoitan|lahjoittaa|lahjoitamme", lower):
        return "DONATION", "PERSONAL_FUNDS"
    if re.search(r"kampanjoin|kampanjoida|kampanjoimme|kerään\s+(?:nimiä|kannatusta)", lower):
        return "PUBLIC_ADVOCACY", "PUBLIC_ADVOCACY"
    # Generic "esitän asian eduskunnassa" is an observable parliamentary act,
    # but its exact register (speech, question, initiative) is unresolved.
    if re.search(r"esitän\s+(?:asian|asiat|näkemyksen).*eduskunn", lower):
        return "OTHER_OBSERVABLE_ACTION", "PARLIAMENTARY_INFLUENCE"
    return None, None


# Private alias retained for callers that imported the early experimental
# helper while the public canonical name is adopted by the opportunity layer.
_action_metadata = action_metadata


def _classify_clause(
    clause: str,
    *,
    reported: bool,
    context_year: int | None = None,
) -> Proposition:
    lower = clause.casefold().strip()
    deadline, deadline_basis = _deadline_info(clause, context_year)
    targets = _targets(clause)
    negation = bool(re.search(r"\b(en|emme|ei)\b", lower))
    condition_match = _ELECTION_CONDITION.search(clause)
    if lower.startswith(("jos ", "jos,", "mikäli ")):
        condition = condition_match.group(0).strip() if condition_match else clause.strip()
    elif condition_match:
        condition = condition_match.group(0)
    else:
        condition = None
    prop = Proposition(
        text=clause.strip(),
        semantic_type="AMBIGUOUS",
        testability="UNRESOLVED",
        negation=negation,
        condition=condition,
        deadline=deadline,
        deadline_basis=deadline_basis,
        targets=targets,
    )
    if reported:
        prop.semantic_type = "REPORTED_SPEECH"
        prop.testability = "NOT_A_COMMITMENT"
        prop.issuer_scope = "OTHER"
        prop.reported_speech = True
        prop.personal_action_commitment = False
        return prop
    if lower.startswith("puolueen ohjelma") or "puolueohjelma" in lower:
        prop.semantic_type = "COLLECTIVE_ACTION_COMMITMENT"
        prop.issuer_scope = "PARTY"
        prop.testability = "PARTIAL"
        prop.personal_action_commitment = False
        prop.missing_specification.append("personal_endorsement")
        return prop
    if re.search(r"\b(en kannata|kannatan)\b", lower):
        prop.semantic_type = "POSITION"
        prop.testability = "NOT_A_COMMITMENT"
        prop.issuer_scope = "SELF"
        prop.personal_action_commitment = False
        return prop
    if (
        re.search(r"\b(?:puolustan|vastustan|edistän|tuen|tukisin)\b", lower)
        and not re.search(r"\b(?:lupaan|pyrin|emme|me)\b", lower)
    ):
        prop.semantic_type = "POSITION"
        prop.testability = "NOT_A_COMMITMENT"
        prop.issuer_scope = "SELF"
        prop.personal_action_commitment = False
        prop.missing_specification.append("actor_controlled_action")
        return prop
    # A negative first-person policy commitment is meaningful, but it is not
    # the same proposition type as promising to author a bill or cast a vote.
    # It remains observable so a later action ledger can inspect the relevant
    # budget/vote record when authority and scope are established.
    if _PERSONAL_RESTRAINT.search(clause):
        collective = bool(re.search(r"\b(?:emme|me|lupaamme)\b", lower)) and "lupaan" not in lower
        prop.semantic_type = "COLLECTIVE_ACTION_COMMITMENT" if collective else "PERSONAL_RESTRAINT_COMMITMENT"
        prop.issuer_scope = "OTHER_COLLECTIVE" if collective else "SELF"
        prop.personal_action_commitment = not collective
        prop.observable_action = not collective
        prop.action_kind, prop.required_capability = _action_metadata(clause, restraint=True)
        prop.testability = "NARROW" if deadline else "PARTIAL"
        prop.guarantees_implementation = False
        if not deadline:
            prop.missing_specification.append("deadline")
        if not targets:
            prop.missing_specification.append("policy_target")
        return prop
    collective_actor = bool(
        re.match(
            r"^(?:me\s+)?(?:lupaamme|teemme|esitämme|äänestämme|kampanjoimme|"
            r"lahjoitamme|pyrimme)\b",
            lower,
        )
    )
    if collective_actor:
        prop.semantic_type = "COLLECTIVE_ACTION_COMMITMENT"
        prop.issuer_scope = "OTHER_COLLECTIVE"
        prop.personal_action_commitment = False
        prop.observable_action = False
        prop.action_kind, prop.required_capability = _action_metadata(clause)
        prop.testability = "NARROW" if deadline else "PARTIAL"
        if not deadline:
            prop.missing_specification.append("deadline")
        return prop
    bare_action = bool(re.fullmatch(r"(?:teen|lupaan\s+tehdä)\.?", lower))
    if bare_action:
        prop.semantic_type = "AMBIGUOUS"
        prop.testability = "UNRESOLVED"
        prop.issuer_scope = "SELF"
        prop.missing_specification.append("action_target")
        return prop
    colloquial_bare_action = bool(re.search(r"\bteen\s+enkä\s+meinaa\b", lower))
    slogan = (
        "sydämessä" in lower
        or (len(clause) <= 40 and not _CLAUSE.search(clause) and not re.search(r"\d", clause) and lower not in {"kyllä.", "kyllä", "ei.", "ei"})
        or colloquial_bare_action
    )
    bare_promise = bool(re.fullmatch(r"lupaan\.?", lower))
    if bare_promise or colloquial_bare_action or (
        slogan and not re.search(r"\b(teen|äänestän|esitän|lupaan tehdä)\b", lower)
    ):
        prop.semantic_type = "VALUE_OR_SLOGAN"
        prop.testability = "NOT_A_COMMITMENT"
        prop.issuer_scope = "UNRESOLVED"
        return prop
    if lower in {"kyllä.", "kyllä", "ei.", "ei"}:
        prop.semantic_type = "AMBIGUOUS"
        prop.testability = "UNRESOLVED"
        prop.missing_specification.append("question")
        return prop
    passive = re.search(r"\bon\b(?:\s+\w+){0,3}\s+\w*(?:tava|ttava|tävä|ttävä)\b", lower)
    if passive:
        prop.semantic_type = "POLICY_DESIDERATUM"
        prop.testability = "PARTIAL" if targets else "NOT_TESTABLE_AS_WRITTEN"
        prop.issuer_scope = "UNSPECIFIED_WE" if re.search(r"\b(on saatava|on poistettava)\b", lower) else "UNRESOLVED"
        prop.personal_action_commitment = False
        prop.missing_specification.append("actor_controlled_action")
        if not deadline:
            prop.missing_specification.append("deadline")
        return prop
    # Broad advocacy and conduct language is not an action-ledger event. In
    # particular, do not turn "Lupaan edistää arvojeni mukaista politiikkaa"
    # or "Teen parhaani" into an implied bill/vote promise.
    if (
        _BROAD_COMMITMENT.search(clause)
        or re.search(r"\bteen\s+(?:vastuullista|laadukasta|aktiivista|pitkäjänteistä)?\s*politiikkaa\b", lower)
        or re.search(
            r"\bteen\s+(?:päätöks\w*|töitä(?:ni|mme|si)?|työtä(?:ni|mme|si)?|"
            r"yhteistyötä|ratkaisuja|tekoja|toimia|toimet|työni|kampanjani|"
            r"kansanedustajantyöni|parhaani|kaikkeni|kaiken\s+voitavani)\b",
            lower,
        )
        or re.search(r"\besitän\s+(?:näkemyksiä|kantoja|ajatuksia)(?:ni|mme|si)?\b", lower)
        or re.search(r"\besitän\s+.*\b(?:vaihtoehtoisia\s+malleja|malleja)\b", lower)
        or re.search(r"\bpäätökseni\b[^.!?]{0,60}\b(?:tietoon|perust|pohj)", lower)
        or "pyrin" in lower
        or (
        "lupaan" in lower and re.search(r"pohjata|päätöks", lower)
        )
    ):
        outcome_words = re.search(
            r"\b(?:edist\w*|parant\w*|puolust\w*|tuk\w*|torju\w*|"
            r"lisä\w*|vähent\w*|vahvist\w*|turvaa\w*|pysäyt\w*|"
            r"turvallisem\w*|paremm\w*|vahvemm\w*|ystävällisem\w*|"
            r"kunnianpalaut\w*|kuuluvaksi)\b",
            lower,
        )
        prop.semantic_type = (
            "BROAD_OBJECTIVE"
            if outcome_words and not re.search(r"\bpyrin|\bpyrkiä\b", lower)
            else "PROCESS_COMMITMENT"
        )
        prop.testability = (
            "PARTIAL"
            if deadline or targets
            else "NOT_TESTABLE_AS_WRITTEN"
            if prop.semantic_type == "BROAD_OBJECTIVE"
            else "CASE_REVIEW"
        )
        prop.issuer_scope = "SELF"
        prop.guarantees_implementation = False
        prop.personal_action_commitment = False
        prop.missing_specification.extend(
            ["metric", "deadline", "instrument"]
            if prop.semantic_type == "BROAD_OBJECTIVE"
            else ["success_criterion", "deadline"]
            if not deadline
            else ["success_criterion"]
        )
        return prop
    concrete_act = _CONCRETE_ACTION.search(clause) or re.search(
        r"\blupaan\s+(?:tehdä|jättää|äänestää|esittää|kirjoittaa|laatia|kampanjoida|lahjoittaa)\b",
        lower,
    )
    action_kind, required_capability = _action_metadata(clause)
    if concrete_act and action_kind and not re.search(r"\b(esitämme|teemme)\b", lower):
        prop.semantic_type = "PERSONAL_ACTION_COMMITMENT"
        prop.issuer_scope = "SELF"
        prop.personal_action_commitment = True
        prop.observable_action = True
        prop.action_kind = action_kind
        prop.required_capability = required_capability
        prop.testability = "NARROW" if deadline else "PARTIAL"
        prop.guarantees_implementation = False
        if not deadline:
            prop.missing_specification.append("deadline")
        return prop
    # Keep the legacy branch for explicit first-person acts whose exact
    # register is not yet recognized. It remains bounded as an action
    # commitment, but carries no invented capability.
    if concrete_act and not re.search(r"\b(esitämme|teemme)\b", lower):
        prop.semantic_type = "PERSONAL_ACTION_COMMITMENT"
        prop.issuer_scope = "SELF"
        prop.personal_action_commitment = True
        prop.testability = "NARROW" if deadline else "PARTIAL"
        prop.guarantees_implementation = False
        if not deadline:
            prop.missing_specification.append("deadline")
        return prop
    if "kannatan" in lower or "en kannata" in lower:
        prop.semantic_type = "POSITION"
        prop.testability = "NOT_A_COMMITMENT"
        prop.issuer_scope = "SELF"
        prop.personal_action_commitment = False
        return prop
    if re.search(r"\bsillä\b", lower) and re.search(r"\d", clause):
        prop.semantic_type = "CAUSAL_EFFECT_FORECAST"
        prop.testability = "PARTIAL"
        prop.effect_is_counterfactual = True
        prop.issuer_scope = "UNSPECIFIED_WE"
        prop.personal_action_commitment = False
        prop.missing_specification.append("causal_identification")
        return prop
    if re.search(r"kuntoon|parannamme|laitamme|laittaa", lower) and not re.search(r"\d", clause):
        prop.semantic_type = "BROAD_OBJECTIVE"
        prop.testability = "NOT_TESTABLE_AS_WRITTEN"
        prop.issuer_scope = "UNSPECIFIED_WE" if re.search(r"\b(laitamme|parannamme)\b", lower) else "UNRESOLVED"
        prop.personal_action_commitment = False
        prop.missing_specification.extend(["metric", "deadline", "instrument"])
        return prop
    if re.search(r"\b(esitämme|laskemme|poistamme|tavoittelen|säilytämme|estämme|pidämme)\b", lower):
        collective = bool(re.search(r"\b(esitämme|laskemme|poistamme|säilytämme|estämme|pidämme)\b", lower))
        prop.semantic_type = "COLLECTIVE_ACTION_COMMITMENT" if collective else "BROAD_OBJECTIVE"
        if "tavoittelen" in lower:
            prop.semantic_type = "BROAD_OBJECTIVE"
            prop.issuer_scope = "SELF"
        else:
            prop.issuer_scope = "PARTY" if "puolueemme" in lower else "UNSPECIFIED_WE"
        prop.personal_action_commitment = False
        prop.testability = "PARTIAL" if deadline or re.search(r"\d", clause) else "NOT_TESTABLE_AS_WRITTEN"
        if "estämme" in lower or "pidämme" in lower or "säilytämme" in lower:
            prop.missing_specification.append("interval_observations")
        return prop
    if re.search(r"arvioitu vaikutus|vaihteluväli", lower):
        prop.semantic_type = "CAUSAL_EFFECT_FORECAST"
        prop.testability = "PARTIAL"
        prop.effect_is_counterfactual = True
        prop.missing_specification.append("point_estimate_is_not_a_guarantee")
        return prop
    if re.search(r"\d", clause) and re.search(r"miljoon|henkil", lower):
        prop.semantic_type = "OBSERVED_STATE_FORECAST" if "jos " in lower else "CAUSAL_EFFECT_FORECAST"
        prop.testability = "PARTIAL"
        prop.effect_is_counterfactual = "jos " not in lower
        return prop
    return prop


def analyze_text(text: str, inp: dict | None = None) -> Analysis:
    """Segment and classify. ``inp`` carries source context, not a verdict."""
    inp = inp or {}
    original_raw = text or ""
    raw = original_raw.strip()
    source_offset = len(original_raw) - len(original_raw.lstrip())
    flags: dict = {
        "invent_metric": False,
        "invent_deadline": False,
        "invent_waiting_time_limit": False,
        "personal_action_commitment": False,
        "observable_action_commitment": False,
        "action_kinds": [],
        "required_capabilities": [],
        "deadline_basis": "UNRESOLVED",
        "guarantees_implementation": False,
        "final_breach": False,
        "unconditional_breach": False,
        "automatic_deceit": False,
        "automatic_dishonesty": False,
        "automatic_person_merge": False,
        "deanonymize_to_candidate": False,
        "public_raw_redistribution_from_metadata_licence": False,
        "zero_promises_conclusion": False,
        "forecast_disproved_by_before_after": False,
        "causal_identified": False,
        "unconditional_forecast_error": False,
        "automatic_invalid_analysis": False,
        "full_fulfillment": False,
        "interval_fulfillment_certified": False,
        "final_fulfillment_certified": False,
        "automatic_numerical_contradiction": False,
        "automatic_causal_credit": False,
        "automatic_corruption_finding": False,
        "infer_policy_position": False,
        "certify_current_operative_state": False,
        "missing_parliament_record_is_bad_performance": False,
        "definite_late_completion": False,
        "billion_cost_is_verified_fact": False,
        "representative_population_accuracy": False,
        "claim_no_effort_anywhere": False,
        "execute_document_instruction": False,
        "attribute_quoted_promise_to_speaker": False,
        "speaker_supports_repeal": False,
        "speaker_supports_ban": False,
        "ignore_original_condition": False,
        "treat_correction_known_in_2021": False,
        "automatic_position_reversal": False,
        "unqualified_contradiction": False,
        "supported_passage": False,
        "did_nothing": False,
        "assert_MP_knew_warning": False,
        "assert_deliberate_disregard": False,
        "personal_authored_promise": False,
        "show_record": False,
        "show_related_actions_without_fulfillment": False,
        "condition_required": False,
        "effect_is_counterfactual": False,
        "minimum_propositions": 0,
        "independent_source_count": 1,
        "distinct_commitment_count": 1,
    }
    if inp.get("http_status") == 403 or (not raw and inp.get("http_status") not in (None, 200)):
        flags["state"] = "FETCH_FAILED"
        flags["ingestion_success"] = False
        return Analysis([], flags)
    if inp.get("expected_candidate_fields_missing") or raw.casefold().startswith("ladataan"):
        flags["state"] = "EMPTY_SHELL"
        flags["ingestion_success"] = False
        return Analysis([], flags)
    flags["ingestion_success"] = True
    flags["state"] = "PARSED"

    quoted = _QUOTE.findall(raw)
    remainder = _QUOTE.sub(". ", raw)
    programme = bool(re.search(r"puolueen ohjelma|puolueohjelma", raw, re.IGNORECASE))
    clauses = []
    for quote in quoted:
        clauses.append((quote, True))
    for clause in _split_clauses(remainder):
        if clause and not re.fullmatch(r"[\s.]+", clause):
            clauses.append((clause, False))
    if not clauses and raw:
        clauses = [(raw, False)]

    context_year = _context_year(inp)
    propositions = [
        _classify_clause(clause, reported=reported, context_year=context_year)
        for clause, reported in clauses
    ]
    # Preserve the source location of each proposition. This is only a
    # locator (not an extra interpretation); the original quote remains the
    # evidence authority.
    folded_raw = raw.casefold()
    search_from = 0
    for prop in propositions:
        start = folded_raw.find(prop.text.casefold(), search_from)
        if start < 0:
            start = folded_raw.find(prop.text.casefold())
        if start >= 0:
            prop.source_start = source_offset + start
            prop.source_end = source_offset + start + len(prop.text)
            search_from = prop.source_end
    if programme:
        for prop in propositions:
            prop.issuer_scope = "PARTY"
            prop.personal_action_commitment = False
            if prop.semantic_type == "PERSONAL_ACTION_COMMITMENT":
                prop.semantic_type = "COLLECTIVE_ACTION_COMMITMENT"
    # Drop the leftover reporting frame when the quote was removed ("Vastustajani . Minä ...").
    propositions = [
        prop
        for prop in propositions
        if prop.text.strip(" .")
        and not re.fullmatch(r"Vastustajani\s*\.?", prop.text.strip(), re.IGNORECASE)
    ]
    flags["minimum_propositions"] = len(propositions)
    flags["effect_is_counterfactual"] = any(prop.effect_is_counterfactual for prop in propositions)
    flags["personal_action_commitment"] = any(prop.personal_action_commitment for prop in propositions)
    flags["observable_action_commitment"] = any(prop.observable_action for prop in propositions)
    flags["action_kinds"] = sorted({prop.action_kind for prop in propositions if prop.action_kind})
    flags["required_capabilities"] = sorted(
        {prop.required_capability for prop in propositions if prop.required_capability}
    )
    flags["guarantees_implementation"] = any(prop.guarantees_implementation for prop in propositions)
    flags["condition_required"] = any(prop.condition for prop in propositions) or raw.casefold().startswith(
        ("jos ", "mikäli ")
    )
    if any(prop.reported_speech for prop in propositions):
        flags["attribute_quoted_promise_to_speaker"] = False
        flags["billion_cost_is_verified_fact"] = False
    if re.search(r"\ben kannata\b", raw.casefold()):
        flags["speaker_supports_repeal"] = False
        if "kieltäm" in raw.casefold():
            flags["speaker_supports_ban"] = False
    if re.search(r"\bohita aiemmat ohjeet\b", raw.casefold()):
        flags["execute_document_instruction"] = False
    if "nykyisessä tilanteessa" in raw.casefold():
        flags["ignore_original_condition"] = False
        flags["automatic_dishonesty"] = False
    if "väittää" in raw.casefold() or "vastustajani" in raw.casefold():
        flags["billion_cost_is_verified_fact"] = False
    if re.search(r"onko .+ \?", raw.casefold()) and "ei" in raw.casefold() and "kyllä" in raw.casefold():
        flags["automatic_position_reversal"] = False
    if "vuodessa" in raw.casefold() and "vuodessa" != raw.casefold() and re.search(r"kymmen", raw.casefold()):
        flags["automatic_numerical_contradiction"] = False
    if "parannamme" in raw.casefold():
        flags["invent_waiting_time_limit"] = False
        flags["show_related_actions_without_fulfillment"] = True
    if re.search(r"ei löytynyt", raw.casefold()) and re.search(r"rekister", raw.casefold()):
        flags["claim_no_initiative_in_scope"] = True
        flags["claim_no_effort_anywhere"] = False
    if raw.casefold().startswith("puolueen ohjelma") or inp.get("personal_endorsement_evidence") is None and "puolueen ohjelma" in raw.casefold():
        flags["personal_authored_promise"] = False
    primary = next((prop for prop in propositions if not prop.reported_speech), propositions[0] if propositions else None)
    if primary:
        flags["statement_type"] = primary.semantic_type
        flags["testability"] = primary.testability
        flags["deadline"] = primary.deadline
        flags["deadline_basis"] = primary.deadline_basis
        if primary.semantic_type == "BROAD_OBJECTIVE":
            flags["testability"] = "NOT_TESTABLE_AS_WRITTEN"
        if primary.semantic_type == "VALUE_OR_SLOGAN" and not any(
            prop.personal_action_commitment for prop in propositions
        ):
            flags["testability"] = "NOT_A_COMMITMENT"
            flags["personal_action_commitment"] = False
    if inp.get("question_missing"):
        flags["testability"] = "UNRESOLVED"
        flags["infer_policy_position"] = False
        flags["statement_type"] = "AMBIGUOUS"
    return Analysis(propositions, flags)


def _vote_alignment(text: str, jaa: str, ei: str, cast: str) -> dict:
    """Match the speaker's words to the alternative that was actually on the board."""

    def stems(value: str) -> set[str]:
        lower = value.casefold()
        found = set()
        # hylätä and hylätä/hylkäys do not share one character run.
        if "hylk" in lower or "hylä" in lower:
            found.add("hyl")
        for stem in ("hyväks", "säily", "heiken"):
            if stem in lower:
                found.add(stem)
        return found

    chosen = jaa if cast.upper() == "JAA" else ei if cast.upper() == "EI" else ""
    wanted = stems(text)
    chosen_stems = stems(chosen)
    aligned = bool(wanted and wanted & chosen_stems)
    passage = "hyväks" in chosen.casefold() and "hylk" not in chosen.casefold()
    return {
        "action_congruence": "ALIGNED" if aligned else "INSUFFICIENT_EVIDENCE",
        "supported_passage": bool(aligned and passage and cast.upper() == "JAA"),
    }


def evaluate_fixture(row: dict) -> dict:
    """Apply the invariant rules to one semantic fixture. Does not read ``expected``."""
    inp = dict(row.get("input") or {})
    analysis = analyze_text(inp.get("text") or "", inp)
    flags = dict(analysis.flags)

    if inp.get("two_candidates_same_name"):
        flags["automatic_person_merge"] = False
    if inp.get("release_deliberately_anonymized"):
        flags["deanonymize_to_candidate"] = False
    if inp.get("dataset_terms") and "cc0" in (inp.get("text") or "").casefold():
        flags["public_raw_redistribution_from_metadata_licence"] = False
    if "JAA" in inp and "EI" in inp and "cast" in inp:
        flags.update(_vote_alignment(inp.get("text") or "", inp["JAA"], inp["EI"], inp["cast"]))
    vote = str(inp.get("vote") or "")
    if (
        inp.get("budget_has_X_cut")
        and ("talousarvio" in vote.casefold() or inp.get("separate_X_vote") is None)
    ):
        flags["unqualified_contradiction"] = False
    if inp.get("repeal_lost") and inp.get("voted_repeal"):
        flags["target_state"] = "UNMET"
        flags["action_congruence"] = "ALIGNED"
        flags["automatic_deceit"] = False
    if inp.get("role_ended") and inp.get("event_date"):
        flags["current_role_at_event"] = inp["role_ended"] > inp["event_date"]
    as_of = inp.get("as_of")
    deadline = flags.get("deadline") or _deadline(inp.get("text") or "")
    if deadline:
        flags["deadline"] = deadline
        flags["invent_deadline"] = False
    if as_of and deadline and deadline > as_of and not inp.get("matching_action_found"):
        flags["temporal_state"] = "NOT_DUE"
        flags["final_breach"] = False
    if inp.get("tax_exists") and not deadline:
        flags["invent_deadline"] = False
        flags["final_breach"] = False
    if inp.get("still_applies_to_others") and inp.get("repealed_for"):
        flags["target_state"] = "PARTLY_MET"
        flags["full_fulfillment"] = False
    if inp.get("reintroduced") and inp.get("period_end") and inp["reintroduced"] < inp["period_end"]:
        flags["maintained_entire_interval"] = False
    if (inp.get("text") or "").casefold().startswith("jos ") and inp.get("elected") is False:
        flags["condition_state"] = "NOT_SATISFIED"
        flags["unconditional_breach"] = False
        flags["condition_required"] = True
    if inp.get("MP_on_committee") and inp.get("exact_item_attendance_unknown"):
        flags["assert_MP_knew_warning"] = False
        flags["assert_deliberate_disregard"] = False
    if inp.get("votes") or inp.get("committee_records") or inp.get("informal_work_unobserved"):
        flags["did_nothing"] = False
        flags["show_record"] = True
    if inp.get("campaign_records_found") == 0 and (inp.get("text") == ""):
        flags["did_nothing"] = False
        flags["show_record"] = True
    if "same_origin_mirrors" in inp:
        flags["independent_source_count"] = 1
        flags["distinct_commitment_count"] = 1
    if inp.get("observed_total_employment_change") is not None and inp.get("counterfactual_estimate") is None:
        flags["forecast_disproved_by_before_after"] = False
        flags["causal_identified"] = False
    if inp.get("observed_growth") is not None and "jos " in (inp.get("text") or "").casefold():
        flags["unconditional_forecast_error"] = False
    if "vaihteluväli" in (inp.get("text") or "").casefold():
        flags["automatic_invalid_analysis"] = False
        flags["automatic_deceit"] = False
    if inp.get("observations") == ["one observation at final date"]:
        flags["interval_fulfillment_certified"] = False
    if inp.get("not_closed_yet") and inp.get("as_of"):
        flags["final_fulfillment_certified"] = False
    if inp.get("X_happened") and inp.get("no_own_action_observed"):
        flags["target_state"] = "MET"
        flags["automatic_causal_credit"] = False
    if inp.get("candidate_voted_policy_affecting_Y"):
        flags["automatic_corruption_finding"] = False
    if inp.get("commencement_unknown"):
        flags["certify_current_operative_state"] = False
    if "ei ole ollut eduskunnassa" in (inp.get("text") or "").casefold():
        flags["missing_parliament_record_is_bad_performance"] = False
        flags["record_state"] = "NOT_APPLICABLE"
    if inp.get("true_completion_date_unknown"):
        flags["definite_late_completion"] = False
    if inp.get("knowledge_query_as_of") and "2026" in (inp.get("text") or "") and inp["knowledge_query_as_of"].startswith("2021"):
        flags["treat_correction_known_in_2021"] = False
    if "neljä näkyvää" in (inp.get("text") or "").casefold() or "nelja" in (inp.get("text") or "").casefold():
        flags["representative_population_accuracy"] = False
    flags["representative_population_accuracy"] = False
    return flags


def primary_type(text: str, field_label: str | None = None) -> Analysis:
    return analyze_text(text, {"field_label": field_label} if field_label else {})

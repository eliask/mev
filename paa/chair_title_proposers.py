"""Parse chair-attributed named proposers from published vote titles.

The Eduskunta ballot register often records competing alternatives as a chair
title.  Named persons may appear on the JAA (left) side, the EI (right) side,
or both.  That is a source-explicit attribution of a published alternative to
named persons.  It is not drafting authorship, committee membership proof,
component-level endorsement, or final adoption.
"""

from dataclasses import dataclass
from typing import Any, Final, Literal

_PROCEDURAL_SIDE_TOKENS: Final = frozenset(
    {
        "hylkääminen",
        "hylkays",
        "hyväksyminen",
        "hyvaksyminen",
        "mietintö",
        "mietinto",
        "asian lähettäminen perustuslakivaliokuntaan",
        "asian lahettaminen perustuslakivaliokuntaan",
        "asian palauttaminen",
        "asian ensimmäisen käsittelyn jatkaminen työelämä- ja tasa-arvovaliokunnan mietinnön pohjalta",
        "nauttii luottamusta",
        "lakiehdotusten hyväksyminen",
    }
)
_STRIP_TRAILING: Final = (
    "monisteena jaettu lausumaehdotus",
    "lausumaehdotus",
    "muutosehdotus",
    "ehdotus",
)
_TERMINAL_POLARITY: Final = frozenset({"jaa", "ei"})
_TRAILING_ANNOTATIONS: Final = frozenset({"vl"})
_GENITIVE_LA_ENDINGS: Final = ("lan", "län")


@dataclass(frozen=True, slots=True, kw_only=True)
class ChairTitleNamedProposer:
    """One chair-title-attributed mover of a published vote alternative."""

    source_display_name: str
    raw_segment: str
    side: Literal["JAA", "EI"]
    slot: str
    basis: str

    @property
    def display_name(self) -> str:
        return self.source_display_name

    def to_wire(self) -> dict[str, str]:
        return {
            "source_display_name": self.source_display_name,
            "display_name": self.source_display_name,
            "raw_segment": self.raw_segment,
            "side": self.side,
            "slot": self.slot,
            "basis": self.basis,
        }


def _clean(value: object) -> str:
    return " ".join(str(value or "").split()).strip()


def finnish_person_name_keys(name: str) -> tuple[str, ...]:
    """Exact source spelling first; optional conservative inflection keys after.

    ``-sen`` → ``-nen`` covers common ``-nen`` genitives.  ``-lan`` / ``-län``
    drops only the terminal genitive ``n`` of ``-la`` / ``-lä`` stems.  Names
    that already end in ``n`` in the nominative (Harjanne, Merinen, Forsgrén)
    are not rewritten; callers must still prefer an exact roster hit first.
    """

    text = _clean(name)
    if not text:
        return ()
    keys: list[str] = [text.casefold()]
    parts = text.split()
    if len(parts) >= 2:
        last = parts[-1]
        if last.endswith("sen") and len(last) > 4:
            candidate = " ".join([*parts[:-1], last[:-3] + "nen"]).casefold()
            if candidate not in keys:
                keys.append(candidate)
        elif len(last) > 4 and last.casefold().endswith(_GENITIVE_LA_ENDINGS):
            candidate = " ".join([*parts[:-1], last[:-1]]).casefold()
            if candidate not in keys:
                keys.append(candidate)
    return tuple(keys)


def _strip_terminal_polarity_and_annotations(side_text: str) -> tuple[str, str | None] | None:
    """Read terminal JAA/EI and supported annotations in either order.

    Markers and annotations are title grammar only.  They do not supply
    proposal full text, component content, or final adoption.
    """

    text = _clean(side_text)
    marker: str | None = None
    while True:
        parts = text.split()
        if parts and parts[-1].casefold() in _TERMINAL_POLARITY:
            if marker is not None:
                return None
            marker = "JAA" if parts[-1].casefold() == "jaa" else "EI"
            text = " ".join(parts[:-1]).strip()
            continue
        if text.endswith(")"):
            open_at = text.rfind("(")
            if open_at <= 0 or text[open_at + 1 : -1].casefold() not in _TRAILING_ANNOTATIONS:
                return None
            text = text[:open_at].rstrip(" ,.-")
            continue
        break
    # Unrecognized or misplaced qualifiers must not become part of a name.
    if "(" in text or ")" in text:
        return None
    return _clean(text), marker


def _looks_like_person_name(segment: str) -> bool:
    text = _clean(segment)
    if not text:
        return False
    lowered = text.casefold()
    if lowered in _PROCEDURAL_SIDE_TOKENS:
        return False
    if any(token in lowered for token in (" / ", "valiokunta", "mietintö", "mietinto", "option", "lausumaehdotus")):
        return False
    parts = text.split()
    if not (2 <= len(parts) <= 4):
        return False
    for part in parts:
        token = part.split("-")[0]
        if len(token) < 2 or not token[0].isalpha() or not token[0].isupper():
            return False
    return True


def _normalize_proposer_segment(segment: str) -> str | None:
    text = _clean(segment)
    if not text:
        return None
    parts = text.split()
    if parts and parts[-1].isdigit():
        parts = parts[:-1]
        text = " ".join(parts)
    lowered = text.casefold()
    for suffix in _STRIP_TRAILING:
        if lowered.endswith(suffix):
            text = text[: len(text) - len(suffix)].strip(" ,.-")
            lowered = text.casefold()
            break
    # Drop leading procedural labels before a person name.
    for prefix in ("lausumaehdotus,", "lausumaehdotus", "mietintö,", "mietintö", "mietinto"):
        if lowered.startswith(prefix):
            text = text[len(prefix) :].strip(" ,.-")
            lowered = text.casefold()
            break
    if not _looks_like_person_name(text):
        return None
    return text


def _extract_side_proposers(
    side_text: str,
    *,
    side: Literal["JAA", "EI"],
) -> list[ChairTitleNamedProposer]:
    proposers: list[ChairTitleNamedProposer] = []
    seen: set[str] = set()
    # Split on commas, but keep hyphenated names intact.
    for raw_segment in side_text.split(","):
        source_name = _normalize_proposer_segment(raw_segment)
        if source_name is None:
            continue
        key = source_name.casefold()
        if key in seen:
            continue
        seen.add(key)
        slot = f"{side}:{len(proposers) + 1}"
        proposers.append(
            ChairTitleNamedProposer(
                source_display_name=source_name,
                raw_segment=_clean(raw_segment),
                side=side,
                slot=slot,
                basis="CHAIR_TITLE_NAMED_ALTERNATIVE",
            )
        )
    return proposers


def parse_chair_title_named_proposers(title: str) -> tuple[ChairTitleNamedProposer, ...]:
    """Extract personally named proposers from both chair-title sides when present."""

    text = _clean(title)
    if not text:
        return ()
    if " / " in text:
        left, right = text.split(" / ", 1)
    elif text.count("/") == 1:
        left, right = text.split("/", 1)
    else:
        return ()
    left, right = _clean(left), _clean(right)
    if not left or not right:
        return ()
    left_parsed = _strip_terminal_polarity_and_annotations(left)
    right_parsed = _strip_terminal_polarity_and_annotations(right)
    if left_parsed is None or right_parsed is None:
        return ()
    left_core, left_marker = left_parsed
    right_core, right_marker = right_parsed
    # Positional sides are left=JAA, right=EI.  Explicit markers must match.
    if left_marker is not None and left_marker != "JAA":
        return ()
    if right_marker is not None and right_marker != "EI":
        return ()
    if not left_core or not right_core:
        return ()
    left_l, right_l = left_core.casefold(), right_core.casefold()
    if left_l in _PROCEDURAL_SIDE_TOKENS and right_l in _PROCEDURAL_SIDE_TOKENS:
        return ()
    left_proposers = _extract_side_proposers(left_core, side="JAA")
    right_proposers = _extract_side_proposers(right_core, side="EI")
    # Require at least one named person; purely procedural titles stay empty.
    if not left_proposers and not right_proposers:
        return ()
    return (*left_proposers, *right_proposers)


def published_alternative_disposition(
    *,
    jaa: int | None,
    ei: int | None,
) -> dict[str, Any]:
    """Map published bucket totals to a *vote-stage* comparison outcome only."""

    final_adoption = {
        "state": "UNRESOLVED",
        "basis": "VOTE_TOTALS_ARE_NOT_FINAL_ADOPTION",
        "limitation": (
            "Published JAA/EI totals decide only which side prevailed at this "
            "vote. Final adoption requires a final-stage/order witness."
        ),
    }
    if jaa is None or ei is None:
        return {
            "state": "UNRESOLVED",
            "jaa_side_state": "UNRESOLVED",
            "ei_side_state": "UNRESOLVED",
            "basis": "PUBLISHED_TOTALS_MISSING",
            "jaa": jaa,
            "ei": ei,
            "final_adoption": final_adoption,
            "scope": "THIS_VOTE_ONLY",
        }
    if jaa > ei:
        return {
            "state": "JAA_SIDE_PREVAILED_AT_THIS_VOTE",
            "jaa_side_state": "PREVAILED_AT_THIS_VOTE",
            "ei_side_state": "LOST_AT_THIS_VOTE",
            "basis": "PUBLISHED_TOTALS",
            "jaa": jaa,
            "ei": ei,
            "final_adoption": final_adoption,
            "scope": "THIS_VOTE_ONLY",
        }
    if ei > jaa:
        return {
            "state": "EI_SIDE_PREVAILED_AT_THIS_VOTE",
            "jaa_side_state": "LOST_AT_THIS_VOTE",
            "ei_side_state": "PREVAILED_AT_THIS_VOTE",
            "basis": "PUBLISHED_TOTALS",
            "jaa": jaa,
            "ei": ei,
            "final_adoption": final_adoption,
            "scope": "THIS_VOTE_ONLY",
        }
    return {
        "state": "TIED_OR_UNRESOLVED",
        "jaa_side_state": "UNRESOLVED",
        "ei_side_state": "UNRESOLVED",
        "basis": "PUBLISHED_TOTALS_TIED",
        "jaa": jaa,
        "ei": ei,
        "final_adoption": final_adoption,
        "scope": "THIS_VOTE_ONLY",
    }


__all__ = [
    "ChairTitleNamedProposer",
    "finnish_person_name_keys",
    "parse_chair_title_named_proposers",
    "published_alternative_disposition",
]

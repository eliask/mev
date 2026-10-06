"""Version-bound ballot, alternative, actor, group and disposition comparison.

This module answers one reusable documentary question across distinct matters:

    for an exact person and vote, which chair-bound alternative did the raw
    response select, how did that compare with the event-time parliamentary
    group majority, and what institutional disposition did the vote (or
    multi-vote bracket) produce?

Raw JAA/EI/TYHJA/POISSA codes are never treated as policy meaning by
themselves.  Authored alternative text must be bound to an exact source span.
Group comparison reuses :func:`paa.group_agreement.compare_person_on_source`.
The module does not claim obedience, competence, private influence or causal
contribution.  A recomputed checksum proves packet consistency, not remote
authenticity of an external publisher.
"""

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any, Final, final

from paa.group_agreement import (
    DEFAULT_MINIMUM_PEERS,
    _source_diagnostics,
    _source_payload,
    compare_person_on_source,
)
from paa.group_agreement import (
    SCHEMA_VERSION as GROUP_SCHEMA_VERSION,
)
from paa.records import alternatives_from_title

SCHEMA_VERSION: Final = "paa.ballot_disposition.v2"
CORPUS_SCHEMA: Final = "paa.ballot_disposition_corpus.v2"
SUBSTANTIVE: Final = frozenset({"JAA", "EI"})
RESPONSE_BUCKETS: Final = ("JAA", "EI", "TYHJA", "POISSA")
BRACKET_ROLES: Final = ("FIRST_ROUND_ELIMINATION", "REPORT_VS_SURVIVOR")


class BallotDispositionError(ValueError):
    """Raised when a disposition corpus or comparison cannot be formed."""


def _text(value: Any) -> str:
    return str(value or "").strip()


def _sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _mapping(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise BallotDispositionError(f"{field} must be an object")
    return dict(value)


def _nonneg_int(value: Any, *, field: str) -> int:
    if type(value) is not int or isinstance(value, bool) or value < 0:
        raise BallotDispositionError(f"{field} must be a non-negative int (not bool)")
    return value


@final
@dataclass(frozen=True, slots=True, kw_only=True, match_args=False)
class TextWitness:
    source_id: str
    text_sha256: str
    start: int
    end: int
    quote: str

    def __post_init__(self) -> None:
        if not self.source_id or len(self.text_sha256) != 64:
            raise BallotDispositionError("text witness needs source_id and sha256")
        if type(self.start) is not int or type(self.end) is not int or isinstance(self.start, bool):
            raise BallotDispositionError("text witness offsets must be ints")
        if self.start < 0 or self.end < self.start or not self.quote:
            raise BallotDispositionError("text witness span invalid")
        if len(self.quote) != self.end - self.start:
            raise BallotDispositionError("text witness quote length mismatches span")


@final
@dataclass(frozen=True, slots=True, kw_only=True, match_args=False)
class AlternativeBinding:
    vote_id: str
    binding_kind: str
    jaa_label: str
    ei_label: str
    jaa_text: str | None
    ei_text: str | None
    jaa_witness: TextWitness | None
    ei_witness: TextWitness | None
    note: str
    title: str

    def __post_init__(self) -> None:
        if not self.vote_id or not self.binding_kind:
            raise BallotDispositionError("alternative binding needs vote_id and binding_kind")
        if not self.jaa_label or not self.ei_label or self.jaa_label == self.ei_label:
            raise BallotDispositionError("distinct chair JAA/EI labels required")
        if (self.jaa_text is None) != (self.jaa_witness is None):
            raise BallotDispositionError("jaa_text and jaa_witness must be paired")
        if (self.ei_text is None) != (self.ei_witness is None):
            raise BallotDispositionError("ei_text and ei_witness must be paired")
        if self.jaa_witness and self.jaa_text != self.jaa_witness.quote:
            raise BallotDispositionError("jaa_text must equal witness quote")
        if self.ei_witness and self.ei_text != self.ei_witness.quote:
            raise BallotDispositionError("ei_text must equal witness quote")


@final
@dataclass(frozen=True, slots=True, kw_only=True, match_args=False)
class ActorBallotRow:
    instance_id: str
    vote_id: str
    matter: str | None
    session_date: str | None
    stage: str | None
    person_id: str
    display_name: str | None
    group_code: str | None
    raw_response: str | None
    selected_side: str | None
    selected_label: str | None
    selected_text: str | None
    selection_state: str
    peer_majority: str | None
    matches_peer_majority: bool | None
    group_status: str
    institutional_winner: str | None
    institutional_winner_label: str | None
    bracket_role: str | None
    bracket_survivor: bool | None
    limits: tuple[str, ...]


def title_side_labels(title: str) -> tuple[str, str] | None:
    """Split a chair title into exact JAA/EI labels without inventing authored text."""

    parsed = alternatives_from_title(title, "title-split")
    if len(parsed) != 2:
        return None
    left, right = parsed[0]["alternative_text"], parsed[1]["alternative_text"]
    if not left or not right or left == right:
        return None
    return left, right


def _witness_from_spec(spec: Mapping[str, Any] | None, *, field: str) -> TextWitness | None:
    if spec is None:
        return None
    raw = _mapping(spec, field=field)
    return TextWitness(
        source_id=_text(raw.get("source_id")),
        text_sha256=_text(raw.get("text_sha256")),
        start=_nonneg_int(raw.get("start"), field=f"{field}.start"),
        end=_nonneg_int(raw.get("end"), field=f"{field}.end"),
        quote=_text(raw.get("quote")),
    )


def _verify_witness(witness: TextWitness, documents: Mapping[str, Any], *, field: str) -> None:
    doc = documents.get(witness.source_id)
    if not isinstance(doc, Mapping):
        raise BallotDispositionError(f"{field}: unknown witness source {witness.source_id}")
    text = doc.get("text")
    if not isinstance(text, str):
        raise BallotDispositionError(f"{field}: witness source text missing")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if digest != witness.text_sha256:
        raise BallotDispositionError(f"{field}: witness source text hash stale")
    if _text(doc.get("text_sha256")) and _text(doc.get("text_sha256")) != witness.text_sha256:
        raise BallotDispositionError(f"{field}: declared text_sha256 contradicts body")
    if witness.end > len(text) or text[witness.start:witness.end] != witness.quote:
        raise BallotDispositionError(f"{field}: witness quote is not an exact source span")


def _approved_side_spec(approved: Mapping[str, Any] | None, side: str, *, vote_id: str) -> dict[str, Any] | None:
    if approved is None:
        return None
    raw = approved.get(side)
    if raw is None:
        return None
    spec = _mapping(raw, field=f"approved_side_bindings.{vote_id}.{side}")
    document_id = _text(spec.get("document_id"))
    text_sha256 = _text(spec.get("text_sha256"))
    start = _nonneg_int(spec.get("start"), field=f"approved_side_bindings.{vote_id}.{side}.start")
    end = _nonneg_int(spec.get("end"), field=f"approved_side_bindings.{vote_id}.{side}.end")
    if not document_id or len(text_sha256) != 64:
        raise BallotDispositionError(f"vote {vote_id}: approved {side} binding needs document_id and text_sha256")
    if end <= start:
        raise BallotDispositionError(f"vote {vote_id}: approved {side} binding needs end > start")
    return {
        "document_id": document_id,
        "text_sha256": text_sha256,
        "start": start,
        "end": end,
        "object_id": _text(spec.get("object_id")) or None,
        "locator": _text(spec.get("locator")) or None,
    }


def _require_approved_witness(
    witness: TextWitness | None,
    approved: dict[str, Any] | None,
    *,
    vote_id: str,
    side: str,
) -> None:
    if witness is None and approved is None:
        return
    if witness is None or approved is None:
        raise BallotDispositionError(
            f"vote {vote_id}: authored {side} witness and approved side binding must be paired"
        )
    if (
        witness.source_id != approved["document_id"]
        or witness.text_sha256 != approved["text_sha256"]
        or witness.start != approved["start"]
        or witness.end != approved["end"]
    ):
        raise BallotDispositionError(
            f"vote {vote_id}: {side} witness does not match approved side binding"
        )


def bind_alternatives(
    vote_id: str,
    source: Mapping[str, Any],
    spec: Mapping[str, Any],
    *,
    documents: Mapping[str, Any] | None = None,
    approved_sides: Mapping[str, Any] | None = None,
) -> AlternativeBinding:
    """Bind exact chair labels and optional authored texts to one vote source version."""

    title = _text(source.get("title"))
    labels = title_side_labels(title)
    if labels is None:
        raise BallotDispositionError(f"vote {vote_id}: chair title does not yield two exact alternatives")
    jaa_label = _text(spec.get("jaa_label"))
    ei_label = _text(spec.get("ei_label"))
    if jaa_label != labels[0] or ei_label != labels[1]:
        raise BallotDispositionError(
            f"vote {vote_id}: labels must equal exact chair sides {labels[0]!r} / {labels[1]!r}"
        )
    jaa_witness = _witness_from_spec(spec.get("jaa_witness"), field=f"{vote_id}.jaa_witness")
    ei_witness = _witness_from_spec(spec.get("ei_witness"), field=f"{vote_id}.ei_witness")
    docs = documents or {}
    if jaa_witness:
        _verify_witness(jaa_witness, docs, field=f"{vote_id}.jaa_witness")
    if ei_witness:
        _verify_witness(ei_witness, docs, field=f"{vote_id}.ei_witness")
    approved = approved_sides or {}
    _require_approved_witness(
        jaa_witness, _approved_side_spec(approved, "jaa", vote_id=vote_id), vote_id=vote_id, side="jaa"
    )
    _require_approved_witness(
        ei_witness, _approved_side_spec(approved, "ei", vote_id=vote_id), vote_id=vote_id, side="ei"
    )
    return AlternativeBinding(
        vote_id=vote_id,
        binding_kind=_text(spec.get("binding_kind")) or "UNDECLARED",
        jaa_label=jaa_label,
        ei_label=ei_label,
        jaa_text=jaa_witness.quote if jaa_witness else None,
        ei_text=ei_witness.quote if ei_witness else None,
        jaa_witness=jaa_witness,
        ei_witness=ei_witness,
        note=_text(spec.get("note")),
        title=title,
    )


def _validated_totals(source: Mapping[str, Any]) -> dict[str, int]:
    totals = source.get("published_totals")
    if not isinstance(totals, Mapping):
        raise BallotDispositionError("published_totals object required")
    buckets = {bucket: _nonneg_int(totals.get(bucket), field=f"published_totals.{bucket}") for bucket in RESPONSE_BUCKETS}
    total = _nonneg_int(totals.get("TOTAL"), field="published_totals.TOTAL")
    if total != sum(buckets.values()):
        raise BallotDispositionError("published TOTAL does not equal bucket sum")
    return {**buckets, "TOTAL": total}


def vote_disposition(source: Mapping[str, Any], binding: AlternativeBinding) -> dict[str, Any]:
    """Return which substantive side won from validated published totals."""

    # Typed totals first so bool/negative buckets fail with a clear field error
    # before row-level source diagnostics that assume int arithmetic.
    totals = _validated_totals(source)
    diagnostics = _source_diagnostics(source, cutoff=source.get("cutoff"))
    if not diagnostics["valid"]:
        raise BallotDispositionError(
            f"vote {source.get('vote_id')}: invalid source before disposition: {';'.join(diagnostics['reasons'])}"
        )
    jaa, ei = totals["JAA"], totals["EI"]
    if jaa == 0 and ei == 0:
        return {
            "state": "NO_SUBSTANTIVE_VOTES",
            "winner_response": None,
            "winner_label": None,
            "winner_text": None,
            "jaa": jaa,
            "ei": ei,
        }
    if jaa == ei:
        return {
            "state": "TIE",
            "winner_response": None,
            "winner_label": None,
            "winner_text": None,
            "jaa": jaa,
            "ei": ei,
        }
    winner = "JAA" if jaa > ei else "EI"
    return {
        "state": "SUBSTANTIVE_MAJORITY",
        "winner_response": winner,
        "winner_label": binding.jaa_label if winner == "JAA" else binding.ei_label,
        "winner_text": binding.jaa_text if winner == "JAA" else binding.ei_text,
        "jaa": jaa,
        "ei": ei,
    }


def select_alternative(response: str | None, binding: AlternativeBinding) -> dict[str, Any]:
    """Map a raw response to a bound alternative without inventing policy meaning."""

    raw = _text(response).upper().replace("Ä", "A")
    if raw not in RESPONSE_BUCKETS:
        return {
            "selection_state": "UNKNOWN_OR_MISSING_RESPONSE",
            "selected_side": None,
            "selected_label": None,
            "selected_text": None,
            "selected_witness": None,
        }
    if raw in {"TYHJA", "POISSA"}:
        return {
            "selection_state": "NON_SUBSTANTIVE_NO_SIDE",
            "selected_side": raw,
            "selected_label": None,
            "selected_text": None,
            "selected_witness": None,
        }
    label = binding.jaa_label if raw == "JAA" else binding.ei_label
    text = binding.jaa_text if raw == "JAA" else binding.ei_text
    witness = binding.jaa_witness if raw == "JAA" else binding.ei_witness
    if text:
        state = "AUTHORED_TEXT_BOUND"
    elif binding.binding_kind == "PROCEDURAL_ADOPTION":
        state = "PROCEDURAL_LABEL_ONLY"
    else:
        state = "CHAIR_LABEL_ONLY"
    return {
        "selection_state": state,
        "selected_side": raw,
        "selected_label": label,
        "selected_text": text,
        "selected_witness": asdict(witness) if witness else None,
    }


def _parse_day(value: Any) -> date | None:
    raw = _text(value)
    if not raw:
        return None
    try:
        return date.fromisoformat(raw[:10])
    except ValueError:
        return None


def _validate_source_record(
    vote_id: str,
    source: Mapping[str, Any],
    *,
    corpus_cutoff: str,
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    if _text(source.get("vote_id")) != vote_id:
        raise BallotDispositionError(f"vote source id mismatch for {vote_id}")
    if source.get("schema_version") != GROUP_SCHEMA_VERSION:
        raise BallotDispositionError(f"vote {vote_id}: unsupported group source schema")
    if _text(source.get("cutoff")) != corpus_cutoff:
        raise BallotDispositionError(f"vote {vote_id}: source cutoff must equal corpus cutoff")
    if source.get("source_sha256") != _sha(_source_payload(source)):
        raise BallotDispositionError(f"vote {vote_id}: source hash changed")
    if _text(expected.get("source_sha256")) != _text(source.get("source_sha256")):
        raise BallotDispositionError(f"vote {vote_id}: source hash disagrees with corpus manifest")
    session = _parse_day(source.get("session_date"))
    cutoff = _parse_day(corpus_cutoff)
    if session is None or cutoff is None:
        raise BallotDispositionError(f"vote {vote_id}: invalid session_date or cutoff")
    if session > cutoff:
        raise BallotDispositionError(f"vote {vote_id}: session_date after corpus cutoff")
    year = source.get("year")
    if type(year) is not int or isinstance(year, bool) or year != session.year:
        raise BallotDispositionError(f"vote {vote_id}: year/session_date mismatch")
    if _text(expected.get("session_date")) != _text(source.get("session_date")):
        raise BallotDispositionError(f"vote {vote_id}: session_date disagrees with manifest")
    if expected.get("year") != year:
        raise BallotDispositionError(f"vote {vote_id}: year disagrees with manifest")
    if _text(expected.get("matter")) != _text(source.get("matter")):
        raise BallotDispositionError(f"vote {vote_id}: matter disagrees with manifest")
    diagnostics = _source_diagnostics(source, cutoff=corpus_cutoff)
    if not diagnostics["valid"]:
        raise BallotDispositionError(
            f"vote {vote_id}: invalid source diagnostics: {';'.join(diagnostics['reasons'])}"
        )
    _validated_totals(source)
    return diagnostics


def _witness_identity(witness: TextWitness | None) -> tuple[str, str, int, int, str] | None:
    if witness is None:
        return None
    return (witness.source_id, witness.text_sha256, witness.start, witness.end, witness.quote)


def _side_payload(binding: AlternativeBinding, response: str) -> tuple[str, TextWitness | None]:
    if response == "JAA":
        return binding.jaa_label, binding.jaa_witness
    if response == "EI":
        return binding.ei_label, binding.ei_witness
    raise BallotDispositionError(f"unsupported bracket response {response!r}")


def _continuing_side(
    binding: AlternativeBinding,
    *,
    prior_witness: tuple[str, str, int, int, str] | None,
    prior_label: str,
) -> tuple[str, str, TextWitness | None]:
    """Locate the unique current chair side that continues the prior winning proposal."""

    matches: list[tuple[str, str, TextWitness | None]] = []
    for response in ("JAA", "EI"):
        label, witness = _side_payload(binding, response)
        if prior_witness is not None:
            if _witness_identity(witness) == prior_witness:
                matches.append((response, label, witness))
        elif witness is None and label == prior_label:
            # Label-only procedural continuity when neither side carries authored text.
            matches.append((response, label, witness))
    if len(matches) != 1:
        raise BallotDispositionError(
            "bracket continuity requires exactly one next-stage side with the prior winner witness/identity"
        )
    return matches[0]


def _validate_bracket_chain(
    chain: Mapping[str, Any],
    *,
    votes: Mapping[str, Any],
    bindings: Mapping[str, AlternativeBinding],
    dispositions: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    chain_id = _text(chain.get("id"))
    stages = chain.get("stages")
    if not chain_id or not isinstance(stages, Sequence) or isinstance(stages, (str, bytes)) or len(stages) < 2:
        raise BallotDispositionError("bracket chain needs id and at least two stages")
    seen: set[str] = set()
    validated_stages: list[dict[str, Any]] = []
    previous: dict[str, Any] | None = None
    for index, stage in enumerate(stages):
        if not isinstance(stage, Mapping):
            raise BallotDispositionError("bracket stage must be an object")
        vote_id = _text(stage.get("vote_id"))
        role = _text(stage.get("role"))
        if not vote_id or vote_id in seen:
            raise BallotDispositionError("bracket stages need unique vote_ids")
        seen.add(vote_id)
        if role not in BRACKET_ROLES:
            raise BallotDispositionError(f"unknown bracket role {role!r}")
        if index == 0 and role != "FIRST_ROUND_ELIMINATION":
            raise BallotDispositionError("first bracket stage must be FIRST_ROUND_ELIMINATION")
        if index > 0 and role != "REPORT_VS_SURVIVOR":
            raise BallotDispositionError("later bracket stage must be REPORT_VS_SURVIVOR")
        if vote_id not in votes or vote_id not in bindings or vote_id not in dispositions:
            raise BallotDispositionError(f"bracket stage vote {vote_id} not evaluated")
        disp = dispositions[vote_id]
        if disp.get("state") != "SUBSTANTIVE_MAJORITY":
            raise BallotDispositionError(f"bracket stage {vote_id} lacks substantive winner")
        declared_response = _text(stage.get("winner_response"))
        declared_label = _text(stage.get("winner_label"))
        eliminated = _text(stage.get("eliminated_label"))
        if declared_response != disp.get("winner_response"):
            raise BallotDispositionError(f"bracket stage {vote_id}: declared winner_response mismatches totals")
        if declared_label != disp.get("winner_label"):
            raise BallotDispositionError(f"bracket stage {vote_id}: declared winner_label mismatches binding")
        binding = bindings[vote_id]
        expected_elim = binding.ei_label if declared_response == "JAA" else binding.jaa_label
        if eliminated != expected_elim:
            raise BallotDispositionError(f"bracket stage {vote_id}: eliminated_label mismatches losing side")
        winner_label, winner_witness = _side_payload(binding, declared_response)
        if previous is not None:
            cont_response, cont_label, _cont_witness = _continuing_side(
                binding,
                prior_witness=previous["witness_identity"],
                prior_label=previous["winner_label"],
            )
            declared_alias = previous.get("survives_as")
            if declared_alias and declared_alias != cont_label:
                raise BallotDispositionError(
                    f"bracket stage {vote_id}: winner_survives_as {declared_alias!r} "
                    f"does not resolve to continuing side {cont_label!r}"
                )
            # Continuing side must compete here; it need not win this round.
            if cont_response not in {"JAA", "EI"}:
                raise BallotDispositionError(f"bracket stage {vote_id}: continuing side unresolved")
        survives_as = _text(stage.get("winner_survives_as")) or None
        previous = {
            "winner_label": winner_label,
            "witness_identity": _witness_identity(winner_witness),
            "survives_as": survives_as,
        }
        validated_stages.append(
            {
                "vote_id": vote_id,
                "role": role,
                "order": index,
                "winner_response": declared_response,
                "winner_label": declared_label,
                "winner_witness": asdict(winner_witness) if winner_witness else None,
                "winner_survives_as": survives_as,
                "eliminated_label": eliminated,
            }
        )
    last = validated_stages[-1]
    # Free-form note is documentary only; adoption is the validated last-stage winner.
    adopted_note = _text(chain.get("adopted_note")) or _text(chain.get("adopted_from_bracket")) or None
    return {
        "id": chain_id,
        "stages": validated_stages,
        "final_adopted_side": {
            "vote_id": last["vote_id"],
            "winner_response": last["winner_response"],
            "winner_label": last["winner_label"],
            "winner_witness": last["winner_witness"],
        },
        "adopted_note": adopted_note,
        "adopted_note_authoritative": False,
    }


def load_corpus(path: Path) -> dict[str, Any]:
    """Load and validate a bundled disposition corpus before any semantic selection."""

    try:
        corpus = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BallotDispositionError(f"cannot read disposition corpus: {exc}") from exc
    if not isinstance(corpus, Mapping):
        raise BallotDispositionError("disposition corpus must be an object")
    if corpus.get("schema_version") != CORPUS_SCHEMA:
        raise BallotDispositionError("unsupported disposition corpus schema")
    cutoff = _text(corpus.get("cutoff"))
    if _parse_day(cutoff) is None:
        raise BallotDispositionError("corpus cutoff must be an ISO date")
    votes = corpus.get("votes")
    manifest = corpus.get("source_manifest")
    if not isinstance(votes, Mapping) or not votes:
        raise BallotDispositionError("corpus needs vote sources")
    if not isinstance(manifest, Mapping) or set(manifest) != set(votes):
        raise BallotDispositionError("source_manifest must cover exactly the vote sources")
    documents = corpus.get("documents") or {}
    if not isinstance(documents, Mapping):
        raise BallotDispositionError("documents must be an object")
    for vote_id, source in votes.items():
        if not isinstance(source, Mapping):
            raise BallotDispositionError(f"vote {vote_id} source must be an object")
        _validate_source_record(
            _text(vote_id),
            source,
            corpus_cutoff=cutoff,
            expected=_mapping(manifest.get(vote_id), field=f"source_manifest.{vote_id}"),
        )
    alt_specs = _mapping(corpus.get("alternatives"), field="alternatives")
    if set(alt_specs) != set(votes):
        raise BallotDispositionError("alternatives must cover exactly the vote sources")
    approved_bindings = corpus.get("approved_side_bindings") or {}
    if not isinstance(approved_bindings, Mapping):
        raise BallotDispositionError("approved_side_bindings must be an object")
    for vote_id, spec in alt_specs.items():
        bind_alternatives(
            _text(vote_id),
            _mapping(votes.get(vote_id), field=f"votes.{vote_id}"),
            _mapping(spec, field=f"alternatives.{vote_id}"),
            documents=documents,
            approved_sides=approved_bindings.get(vote_id),
        )
    return dict(corpus)


def _unique_ids(values: Any, *, field: str) -> list[str]:
    if not isinstance(values, list) or not values:
        raise BallotDispositionError(f"{field} must be a non-empty list")
    result = [_text(value) for value in values]
    if any(not value for value in result):
        raise BallotDispositionError(f"{field} contains a blank id")
    if len(set(result)) != len(result):
        raise BallotDispositionError(f"{field} contains duplicate ids")
    return result


def evaluate_instance(
    corpus: Mapping[str, Any],
    instance_id: str,
    *,
    minimum_peer_count: int = DEFAULT_MINIMUM_PEERS,
) -> dict[str, Any]:
    """Evaluate one named corpus instance into actor ballot rows."""

    if corpus.get("schema_version") != CORPUS_SCHEMA:
        raise BallotDispositionError("evaluate_instance requires a validated v2 corpus object")
    cutoff = _text(corpus.get("cutoff"))
    instances = _mapping(corpus.get("instances"), field="instances")
    if instance_id not in instances:
        raise BallotDispositionError(f"unknown instance {instance_id}")
    instance = _mapping(instances[instance_id], field=f"instances.{instance_id}")
    vote_ids = _unique_ids(instance.get("vote_ids"), field=f"{instance_id}.vote_ids")
    persons = _unique_ids(instance.get("focus_persons"), field=f"{instance_id}.focus_persons")
    matters = _unique_ids(instance.get("matters"), field=f"{instance_id}.matters")
    votes = _mapping(corpus.get("votes"), field="votes")
    manifest = _mapping(corpus.get("source_manifest"), field="source_manifest")
    alt_specs = _mapping(corpus.get("alternatives"), field="alternatives")
    documents = _mapping(corpus.get("documents") or {}, field="documents")
    approved_bindings = _mapping(corpus.get("approved_side_bindings") or {}, field="approved_side_bindings")
    bindings: dict[str, AlternativeBinding] = {}
    binding_records: dict[str, dict[str, Any]] = {}
    dispositions: dict[str, dict[str, Any]] = {}
    for vote_id in vote_ids:
        source = _mapping(votes.get(vote_id), field=f"votes.{vote_id}")
        _validate_source_record(
            vote_id,
            source,
            corpus_cutoff=cutoff,
            expected=_mapping(manifest.get(vote_id), field=f"source_manifest.{vote_id}"),
        )
        matter = _text(source.get("matter"))
        if matter not in matters:
            raise BallotDispositionError(f"{instance_id}: vote {vote_id} matter {matter!r} outside instance matters")
        approved = approved_bindings.get(vote_id)
        binding = bind_alternatives(
            vote_id,
            source,
            _mapping(alt_specs.get(vote_id), field=f"alternatives.{vote_id}"),
            documents=documents,
            approved_sides=approved if isinstance(approved, Mapping) else None,
        )
        disp = vote_disposition(source, binding)
        bindings[vote_id] = binding
        binding_records[vote_id] = {
            "vote_id": binding.vote_id,
            "binding_kind": binding.binding_kind,
            "jaa_label": binding.jaa_label,
            "ei_label": binding.ei_label,
            "jaa_text": binding.jaa_text,
            "ei_text": binding.ei_text,
            "jaa_witness": asdict(binding.jaa_witness) if binding.jaa_witness else None,
            "ei_witness": asdict(binding.ei_witness) if binding.ei_witness else None,
            "approved_side_binding": dict(approved) if isinstance(approved, Mapping) else None,
            "note": binding.note,
            "title": binding.title,
        }
        dispositions[vote_id] = disp
    chain_records = []
    for chain in corpus.get("bracket_chains") or []:
        if not isinstance(chain, Mapping):
            raise BallotDispositionError("bracket chain must be an object")
        stage_votes = {_text(stage.get("vote_id")) for stage in chain.get("stages") or [] if isinstance(stage, Mapping)}
        if stage_votes & set(vote_ids):
            if not stage_votes <= set(vote_ids):
                raise BallotDispositionError("bracket chain stages must be a subset of instance vote_ids")
            chain_records.append(
                _validate_bracket_chain(chain, votes=votes, bindings=bindings, dispositions=dispositions)
            )
    bracket_by_vote = {
        stage["vote_id"]: {
            **stage,
            "chain_id": chain["id"],
            "final_adopted_side": chain["final_adopted_side"],
            "adopted_note": chain.get("adopted_note"),
        }
        for chain in chain_records
        for stage in chain["stages"]
    }
    rows: list[dict[str, Any]] = []
    for vote_id in vote_ids:
        source = votes[vote_id]
        binding = bindings[vote_id]
        disp = dispositions[vote_id]
        bracket = bracket_by_vote.get(vote_id)
        for person_id in persons:
            group = compare_person_on_source(source, person_id, minimum_peer_count=minimum_peer_count)
            if group.get("status") == "SOURCE_EXCLUDED":
                raise BallotDispositionError(f"vote {vote_id}: source excluded during person compare")
            selection = select_alternative(group.get("target_response"), binding)
            display = None
            for row in source.get("rows") or []:
                if isinstance(row, Mapping) and _text(row.get("person_id")) == person_id:
                    display = " ".join(
                        part for part in (_text(row.get("first_name")), _text(row.get("last_name"))) if part
                    ) or None
                    break
            survivor = None
            if bracket and selection["selected_side"] in SUBSTANTIVE:
                survivor = selection["selected_label"] == bracket.get("winner_label")
            limits = [
                "Raw ballot codes are not policy meaning without the bound alternative.",
                "Group agreement is an event-time peer majority, not obedience or competence.",
                "A package adoption ballot is not assent to every provision.",
                "Checksum proves packet consistency, not publisher authenticity.",
            ]
            if selection["selection_state"] == "NON_SUBSTANTIVE_NO_SIDE":
                limits.append("TYHJA/POISSA do not select a JAA/EI alternative.")
            if bracket and bracket.get("role") == "FIRST_ROUND_ELIMINATION":
                limits.append("An eliminated first-round alternative is not the adopted object.")
            row = ActorBallotRow(
                instance_id=instance_id,
                vote_id=vote_id,
                matter=_text(source.get("matter")) or None,
                session_date=_text(source.get("session_date")) or None,
                stage=bracket.get("role") if bracket else None,
                person_id=person_id,
                display_name=display,
                group_code=group.get("target_group_code"),
                raw_response=group.get("target_response"),
                selected_side=selection["selected_side"],
                selected_label=selection["selected_label"],
                selected_text=selection["selected_text"],
                selection_state=selection["selection_state"],
                peer_majority=group.get("peer_majority"),
                matches_peer_majority=group.get("matches_peer_majority"),
                group_status=_text(group.get("status")) or "UNKNOWN",
                institutional_winner=disp.get("winner_response"),
                institutional_winner_label=disp.get("winner_label"),
                bracket_role=bracket.get("role") if bracket else None,
                bracket_survivor=survivor,
                limits=tuple(limits),
            )
            rows.append(
                {
                    **{field: getattr(row, field) for field in row.__dataclass_fields__},
                    "source_id": source.get("source_id"),
                    "source_sha256": source.get("source_sha256"),
                    "group_reason": group.get("reason"),
                    "selected_witness": selection.get("selected_witness"),
                }
            )
    rows.sort(key=lambda item: (item["vote_id"], item["person_id"]))
    packet = {
        "schema_version": SCHEMA_VERSION,
        "instance_id": instance_id,
        "domain": instance.get("domain"),
        "matters": matters,
        "vote_ids": vote_ids,
        "focus_persons": persons,
        "cutoff": cutoff,
        "minimum_peer_count": minimum_peer_count,
        "bindings": binding_records,
        "dispositions": dispositions,
        "bracket_chains": chain_records,
        "rows": rows,
        "limits": [
            "Documentary alternative selection and peer comparison only.",
            "No causal attribution, motive inference or implementation success claim.",
            "Checksum proves packet consistency, not publisher authenticity.",
        ],
    }
    packet["result_sha256"] = packet_digest(packet)
    return packet


def packet_digest(packet: Mapping[str, Any]) -> str:
    """Canonical digest over the full semantic packet (consistency, not authenticity)."""

    rows = sorted(
        (
            {
                "vote_id": row["vote_id"],
                "person_id": row["person_id"],
                "display_name": row.get("display_name"),
                "matter": row.get("matter"),
                "session_date": row.get("session_date"),
                "stage": row.get("stage"),
                "group_code": row.get("group_code"),
                "raw_response": row.get("raw_response"),
                "selected_side": row.get("selected_side"),
                "selected_label": row.get("selected_label"),
                "selected_text": row.get("selected_text"),
                "selected_witness": row.get("selected_witness"),
                "selection_state": row.get("selection_state"),
                "peer_majority": row.get("peer_majority"),
                "matches_peer_majority": row.get("matches_peer_majority"),
                "group_status": row.get("group_status"),
                "group_reason": row.get("group_reason"),
                "institutional_winner": row.get("institutional_winner"),
                "institutional_winner_label": row.get("institutional_winner_label"),
                "bracket_role": row.get("bracket_role"),
                "bracket_survivor": row.get("bracket_survivor"),
                "source_id": row.get("source_id"),
                "source_sha256": row.get("source_sha256"),
                "limits": list(row.get("limits") or ()),
            }
            for row in packet.get("rows", [])
            if isinstance(row, Mapping)
        ),
        key=lambda item: (item["vote_id"], item["person_id"]),
    )
    return _sha(
        {
            "schema_version": packet.get("schema_version"),
            "instance_id": packet.get("instance_id"),
            "domain": packet.get("domain"),
            "matters": packet.get("matters"),
            "vote_ids": packet.get("vote_ids"),
            "focus_persons": packet.get("focus_persons"),
            "cutoff": packet.get("cutoff"),
            "minimum_peer_count": packet.get("minimum_peer_count"),
            "bindings": packet.get("bindings"),
            "dispositions": packet.get("dispositions"),
            "bracket_chains": packet.get("bracket_chains"),
            "rows": rows,
            "limits": packet.get("limits"),
        }
    )


def validate_packet(packet: Mapping[str, Any]) -> bool:
    """Recompute the packet digest before a public consumer uses the result."""

    if packet.get("schema_version") != SCHEMA_VERSION:
        raise BallotDispositionError("unsupported result schema")
    expected = packet_digest(packet)
    if packet.get("result_sha256") != expected:
        raise BallotDispositionError("result digest mismatch; packet changed after evaluation")
    return True


def evaluate_corpus(path: Path, *, minimum_peer_count: int = DEFAULT_MINIMUM_PEERS) -> dict[str, Any]:
    """Evaluate every declared instance in a bundled corpus."""

    corpus_bytes = path.read_bytes()
    corpus = load_corpus(path)
    instances = sorted(_mapping(corpus.get("instances"), field="instances"))
    results = {
        instance_id: evaluate_instance(corpus, instance_id, minimum_peer_count=minimum_peer_count)
        for instance_id in instances
    }
    for packet in results.values():
        validate_packet(packet)
    corpus_filename = path.name
    corpus_sha256 = hashlib.sha256(corpus_bytes).hexdigest()
    return {
        "schema_version": SCHEMA_VERSION,
        "corpus_filename": corpus_filename,
        "corpus_sha256": corpus_sha256,
        "cutoff": corpus.get("cutoff"),
        "instances": results,
        "instance_ids": instances,
        "result_sha256": _sha(
            {
                "cutoff": corpus.get("cutoff"),
                "corpus_filename": corpus_filename,
                "corpus_sha256": corpus_sha256,
                "instance_ids": instances,
                "packets": {key: value["result_sha256"] for key, value in results.items()},
            }
        ),
    }


def actor_row(packet: Mapping[str, Any], *, person_id: str, vote_id: str) -> dict[str, Any]:
    """Return exactly one actor/vote row or raise."""

    validate_packet(packet)
    matches = [
        row
        for row in packet.get("rows", [])
        if isinstance(row, Mapping)
        and _text(row.get("person_id")) == _text(person_id)
        and _text(row.get("vote_id")) == _text(vote_id)
    ]
    if len(matches) != 1:
        raise BallotDispositionError(f"expected one row for person {person_id} vote {vote_id}, got {len(matches)}")
    return dict(matches[0])


def compare_person_ballot(
    source: Mapping[str, Any],
    person_id: str,
    *,
    minimum_peer_count: int = DEFAULT_MINIMUM_PEERS,
    cutoff: str | None = None,
) -> dict[str, Any]:
    """Direct group compare that refuses validated meaning from an invalid source."""

    diagnostics = _source_diagnostics(source, cutoff=cutoff or source.get("cutoff"))
    if not diagnostics["valid"]:
        return {
            "status": "SOURCE_EXCLUDED",
            "reason": ";".join(diagnostics["reasons"]) or diagnostics["state"],
            "validated_meaning": False,
            "target_response": None,
            "peer_majority": None,
            "matches_peer_majority": None,
        }
    comparison = compare_person_on_source(source, person_id, minimum_peer_count=minimum_peer_count)
    return {**comparison, "validated_meaning": comparison.get("status") == "COMPARABLE"}


__all__ = [
    "CORPUS_SCHEMA",
    "SCHEMA_VERSION",
    "ActorBallotRow",
    "AlternativeBinding",
    "BallotDispositionError",
    "TextWitness",
    "actor_row",
    "bind_alternatives",
    "compare_person_ballot",
    "evaluate_corpus",
    "evaluate_instance",
    "load_corpus",
    "packet_digest",
    "select_alternative",
    "title_side_labels",
    "validate_packet",
    "vote_disposition",
]

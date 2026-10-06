"""Evaluation helpers for the bounded facet experiment.

Metrics here describe source-anchor preservation, facet/linked-clause recall,
and agreement with the project's source-reviewed AI reference rows.  They do
not claim human truth, population precision, or causal validity.  The blind
reference loader is intentionally separate from the source-only run loader in
``paa.llm_facets_run`` and should only be called after a prompt/configuration
has been frozen.
"""

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import final

from paa.llm_facets_run import load_blind_references_for_evaluation


@final
@dataclass(frozen=True, slots=True, kw_only=True, match_args=False)
class FacetRunScore:
    run: str
    evaluated_rows: int
    output_rows: int
    source_anchor_checked: int
    source_anchor_valid: int
    reference_facets: int
    matched_facets: int
    kind_checked: int
    kind_agreed: int
    linked_clause_reference: int
    linked_clause_covered: int
    invalid_records: int
    transport_failures: int
    definition_mismatches: Mapping[str, int]
    error_codes: Mapping[str, int]
    ontology_uncomparable_kinds: Mapping[str, int]
    agreement_basis: str = "source-reviewed AI reference interpretations; not human truth"

    def as_dict(self) -> dict[str, object]:
        return {
            "run": self.run,
            "evaluated_rows": self.evaluated_rows,
            "output_rows": self.output_rows,
            "source_anchor_checked": self.source_anchor_checked,
            "source_anchor_valid": self.source_anchor_valid,
            "source_anchor_validity": _ratio(self.source_anchor_valid, self.source_anchor_checked),
            "reference_facets": self.reference_facets,
            "matched_facets": self.matched_facets,
            "facet_preservation": _ratio(self.matched_facets, self.reference_facets),
            "kind_checked": self.kind_checked,
            "kind_agreed": self.kind_agreed,
            "agreement_ai_reference": _ratio(self.kind_agreed, self.kind_checked),
            "ontology_uncomparable_kinds": dict(self.ontology_uncomparable_kinds),
            "ontology_uncomparable_count": sum(self.ontology_uncomparable_kinds.values()),
            "linked_clause_reference": self.linked_clause_reference,
            "linked_clause_covered": self.linked_clause_covered,
            "linked_clause_coverage": _ratio(self.linked_clause_covered, self.linked_clause_reference),
            "invalid_records": self.invalid_records,
            "transport_failures": self.transport_failures,
            "definition_mismatches": dict(self.definition_mismatches),
            "error_codes": dict(self.error_codes),
            "agreement_basis": self.agreement_basis,
            "human_gold": False,
        }


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else round(numerator / denominator, 6)


def _legacy_kind(reference: Mapping[str, object]) -> str | None:
    semantic = reference.get("semantic_type")
    if semantic == "PERSONAL_ACTION_COMMITMENT" or semantic == "PERSONAL_RESTRAINT_COMMITMENT":
        return "FUTURE_COMMITMENT"
    if semantic == "FACTUAL_CLAIM":
        # The old fixture does not always distinguish past/current. Preserve
        # that definition mismatch instead of inventing a temporal label.
        return "PAST_FACT"
    if semantic in {"POLICY_DESIDERATUM", "BROAD_OBJECTIVE"}:
        return "POLICY_GOAL"
    if semantic == "POSITION":
        return "POSITION"
    if semantic in {"CAUSAL_CLAIM", "CAUSAL_EFFECT_FORECAST", "OBSERVED_STATE_FORECAST"}:
        return "FORECAST"
    if semantic == "QUESTION":
        return "QUESTION"
    if semantic == "REPORTED_SPEECH":
        return "REPORTED_SPEECH"
    return "AMBIGUOUS" if semantic == "AMBIGUOUS" else None


def _reference_propositions(row: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    gold = row.get("gold")
    if not isinstance(gold, Mapping):
        return ()
    propositions = gold.get("propositions")
    if not isinstance(propositions, list):
        return ()
    return tuple(item for item in propositions if isinstance(item, Mapping))


def _prediction_facets(result: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    extraction = result.get("extraction")
    if not isinstance(extraction, Mapping):
        return ()
    facets = extraction.get("facets")
    if isinstance(facets, list):
        return tuple(item for item in facets if isinstance(item, Mapping))
    units = extraction.get("units")
    if not isinstance(units, list):
        return ()
    return tuple(
        facet
        for unit in units
        if isinstance(unit, Mapping)
        for facet in unit.get("facets", [])
        if isinstance(facet, Mapping)
    )


def _exact_or_contained(left: object, right: object) -> bool:
    if not isinstance(left, str) or not isinstance(right, str) or not left or not right:
        return False
    return left == right or left in right or right in left


def _source_anchor_is_valid(facet: Mapping[str, object], source_text: str) -> bool:
    quote = facet.get("source_quote")
    start = facet.get("source_start")
    end = facet.get("source_end")
    return isinstance(quote, str) and isinstance(start, int) and isinstance(end, int) and source_text[start:end] == quote


def _definition_mismatches(reference: Mapping[str, object]) -> Counter[str]:
    mismatches: Counter[str] = Counter()
    semantic = reference.get("semantic_type")
    action = reference.get("action_kind")
    if semantic == "FACTUAL_CLAIM" and action is not None:
        mismatches["legacy_factual_claim_with_action_kind"] += 1
    if semantic in {"PAST_FACT", "CURRENT_FACT"}:
        mismatches["legacy_temporal_fact_label_not_in_v2_codebook"] += 1
    return mismatches


def evaluate_facet_run(run_dir: Path, references: Sequence[Mapping[str, object]]) -> FacetRunScore:
    """Evaluate a frozen output directory against source-reviewed references."""

    by_document: dict[str, Mapping[str, object]] = {}
    for row in references:
        document = row.get("document")
        if isinstance(document, Mapping):
            document_id = document.get("document_id")
            if document_id is not None:
                by_document[str(document_id)] = row
    output_rows = 0
    source_anchor_checked = 0
    source_anchor_valid = 0
    reference_facets = 0
    matched_facets = 0
    kind_checked = 0
    kind_agreed = 0
    linked_clause_reference = 0
    linked_clause_covered = 0
    invalid_records = 0
    transport_failures = 0
    errors: Counter[str] = Counter()
    mismatches: Counter[str] = Counter()
    uncomparable_kinds: Counter[str] = Counter()
    cases_dir = run_dir / "cases"
    for path in sorted(cases_dir.glob("*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        output_rows += 1
        outcome = value.get("outcome") if isinstance(value, Mapping) else None
        receipt_status = outcome.get("receipt_status") if isinstance(outcome, Mapping) else None
        if receipt_status != "OK":
            transport_failures += 1
        extraction = value.get("extraction") if isinstance(value, Mapping) else None
        if isinstance(extraction, Mapping):
            invalid = extraction.get("invalid_records")
            if isinstance(invalid, list):
                invalid_records += len(invalid)
                errors.update(str(item.get("code")) for item in invalid if isinstance(item, Mapping) and item.get("code"))
        document_id = str(value.get("item_id", path.stem)) if isinstance(value, Mapping) else path.stem
        reference_row = by_document.get(document_id)
        if reference_row is None:
            continue
        source = reference_row.get("document")
        source_text = source.get("source_text") if isinstance(source, Mapping) else None
        if not isinstance(source_text, str):
            continue
        predicted = _prediction_facets(value if isinstance(value, Mapping) else {})
        for facet in predicted:
            source_anchor_checked += 1
            if _source_anchor_is_valid(facet, source_text):
                source_anchor_valid += 1
        refs = _reference_propositions(reference_row)
        reference_facets += len(refs)
        linked_clause_reference += max(0, len(refs) - 1)
        matched_reference_count = 0
        for reference in refs:
            mismatches.update(_definition_mismatches(reference))
            source_quote = reference.get("source_quote")
            target_kind = _legacy_kind(reference)
            if target_kind is None:
                semantic = reference.get("semantic_type")
                uncomparable_kinds[str(semantic) if semantic is not None else "<missing>"] += 1
            candidates = [facet for facet in predicted if _exact_or_contained(source_quote, facet.get("source_quote"))]
            if not candidates:
                continue
            # Source span match is the preservation metric. Kind agreement is
            # reported separately and does not erase a useful anchor match.
            matched_facets += 1
            matched_reference_count += 1
            if target_kind is None:
                continue
            kind_checked += 1
            if any(facet.get("kind") == target_kind for facet in candidates):
                kind_agreed += 1
        if len(refs) > 1:
            has_link = any(bool(facet.get("linked_subproposition_ids")) for facet in predicted)
            if has_link:
                linked_clause_covered += min(max(0, matched_reference_count - 1), len(refs) - 1)
    evaluated_rows = len(by_document)
    return FacetRunScore(
        run=str(run_dir),
        evaluated_rows=evaluated_rows,
        output_rows=output_rows,
        source_anchor_checked=source_anchor_checked,
        source_anchor_valid=source_anchor_valid,
        reference_facets=reference_facets,
        matched_facets=matched_facets,
        kind_checked=kind_checked,
        kind_agreed=kind_agreed,
        linked_clause_reference=linked_clause_reference,
        linked_clause_covered=linked_clause_covered,
        invalid_records=invalid_records,
        transport_failures=transport_failures,
        definition_mismatches=dict(mismatches),
        error_codes=dict(errors),
        ontology_uncomparable_kinds=dict(uncomparable_kinds),
    )


def choose_development_configuration(scores: Sequence[FacetRunScore]) -> FacetRunScore:
    """Choose by valid useful output and lower failure burden, never agreement."""

    if not scores:
        raise ValueError("at least one development score is required")
    return max(
        scores,
        key=lambda score: (
            score.matched_facets,
            score.source_anchor_valid,
            -score.invalid_records,
            -score.transport_failures,
            -score.source_anchor_checked,
        ),
    )


def evaluate_blind_run(run_dir: Path, fixture: Path) -> dict[str, object]:
    """Evaluate once after freeze and label the reference basis explicitly."""

    references = load_blind_references_for_evaluation(fixture)
    score = evaluate_facet_run(run_dir, references)
    result = score.as_dict()
    result.update(
        {
            "split": "heldout_blind_source_only",
            "prompt_selection_after_run": False,
            "reference_basis": "AI source reading from locked fixture; not human truth",
            "human_gold": False,
        }
    )
    return result


def write_evaluation_report(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


__all__ = [
    "FacetRunScore",
    "choose_development_configuration",
    "evaluate_blind_run",
    "evaluate_facet_run",
    "write_evaluation_report",
]

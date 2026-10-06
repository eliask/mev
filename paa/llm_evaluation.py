"""Offline reports for the local-LLM PAA source-grounded fixtures.

No model, HTTP endpoint, or retrieval label is consulted here.  Callers pass
predictions and this module compares them against separately source-read
records. These labels were produced by an AI source-reading pass; they are
neither human ground truth nor inter-rater validation.
"""


import argparse
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

FIXTURE_DIR = Path(__file__).resolve().parent / "contracts" / "fixtures"
SEMANTIC_GOLD_PATH = FIXTURE_DIR / "llm_semantic_gold.jsonl"
RELATION_GOLD_PATH = FIXTURE_DIR / "llm_relation_gold.jsonl"


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load non-empty JSONL rows and reject malformed records."""

    path = Path(path)
    rows: list[dict[str, Any]] = []
    # JSON strings may legitimately contain U+2028/U+2029.  They are not
    # JSONL record separators; only LF (and the optional CR before it) is.
    for line_number, line in enumerate(path.read_text(encoding="utf-8").split("\n"), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
        if not isinstance(row, dict):
            raise TypeError(f"{path}:{line_number}: expected an object")
        rows.append(row)
    return rows


def load_semantic_gold(
    path: str | Path = SEMANTIC_GOLD_PATH,
    split: str | None = None,
) -> list[dict[str, Any]]:
    return _select(load_jsonl(path), split)


def load_relation_gold(
    path: str | Path = RELATION_GOLD_PATH,
    split: str | None = None,
) -> list[dict[str, Any]]:
    return _select(load_jsonl(path), split)


def _select(rows: Iterable[Mapping[str, Any]], split: str | None) -> list[dict[str, Any]]:
    if split is not None and split not in {"development", "heldout"}:
        raise ValueError("split must be development, heldout, or None")
    return [
        dict(row)
        for row in rows
        if split is None or row.get("split") == split
    ]


def public_semantic_records(
    split: str | None = None,
    *,
    rows: Iterable[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Return model inputs with gold, reviewer notes, and stratification hidden."""

    selected = _select(rows if rows is not None else load_semantic_gold(), split)
    result = []
    for row in selected:
        document = dict(row.get("document") or {})
        document.pop("source_sha256", None)
        result.append({"record_id": row.get("record_id"), "document": document})
    return result


def public_relation_records(
    split: str | None = None,
    *,
    rows: Iterable[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Return source/object inputs without gold or adjudication text."""

    selected = _select(rows if rows is not None else load_relation_gold(), split)
    result = []
    for row in selected:
        source = dict(row.get("source") or {})
        obj = dict(row.get("object") or {})
        source.pop("source_sha256", None)
        obj.pop("object_sha256", None)
        result.append({"pair_id": row.get("pair_id"), "source": source, "object": obj})
    return result


def _prediction_map(
    predictions: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    key: str,
) -> dict[str, Any]:
    if isinstance(predictions, Mapping):
        return {str(k): value for k, value in predictions.items()}
    if isinstance(predictions, (str, bytes)) or not isinstance(predictions, Sequence):
        raise TypeError("predictions must be a mapping or a sequence")
    result: dict[str, Any] = {}
    for index, item in enumerate(predictions):
        if not isinstance(item, Mapping):
            raise TypeError(f"prediction {index} must be an object")
        item_key = item.get(key)
        if not isinstance(item_key, str) or not item_key:
            raise ValueError(f"prediction {index} has no {key}")
        if item_key in result:
            raise ValueError(f"duplicate prediction {key}: {item_key}")
        result[item_key] = item.get("prediction", item)
    return result


def _first_proposition(output: Any) -> Mapping[str, Any] | None:
    if isinstance(output, Mapping):
        propositions = output.get("propositions")
        if isinstance(propositions, Sequence) and not isinstance(propositions, (str, bytes)):
            for proposition in propositions:
                if isinstance(proposition, Mapping):
                    return proposition
        if any(key in output for key in ("semantic_type", "type", "source_quote", "text")):
            return output
    elif isinstance(output, Sequence) and not isinstance(output, (str, bytes)):
        for proposition in output:
            if isinstance(proposition, Mapping):
                return proposition
    return None


def _value(item: Mapping[str, Any] | None, *names: str) -> Any:
    if item is None:
        return None
    for name in names:
        if name in item:
            return item[name]
    return None


def _type(output: Any, proposition: Mapping[str, Any] | None = None) -> str | None:
    value = _value(proposition, "semantic_type", "type", "statement_type")
    if value is None and isinstance(output, Mapping):
        value = output.get("semantic_type") or output.get("type")
    return value.strip().upper() if isinstance(value, str) and value.strip() else None


def _is_abstention(output: Any, proposition: Mapping[str, Any] | None) -> bool:
    if not isinstance(output, Mapping):
        return proposition is None
    if output.get("abstain") is True or output.get("abstention") is True:
        return True
    if isinstance(output.get("abstention"), Mapping) and output["abstention"]:
        return True
    status = output.get("status")
    if isinstance(status, str) and status.upper() in {"ABSTAIN", "ABSTENTION", "UNKNOWN"}:
        return True
    return _type(output, proposition) in {"UNKNOWN", "UNRESOLVED", "ABSTAIN"}


def _bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        value = value.strip().casefold()
        if value in {"true", "yes", "1", "negated", "negative"}:
            return True
        if value in {"false", "no", "0", "not_negated", "positive"}:
            return False
    return None


def _same_optional(expected: Any, actual: Any) -> bool:
    expected = expected.strip() if isinstance(expected, str) else expected
    actual = actual.strip() if isinstance(actual, str) else actual
    if expected in (None, "") and actual in (None, ""):
        return True
    return expected == actual


def _deadline(item: Mapping[str, Any] | None) -> tuple[Any, Any, Any]:
    return (
        _value(item, "deadline_quote", "deadline_text"),
        _value(item, "deadline_normalized", "deadline"),
        _value(item, "deadline_basis"),
    )


def _field_match(gold: Mapping[str, Any], prediction: Mapping[str, Any] | None, field: str) -> bool:
    if prediction is None:
        return False
    if field == "semantic_type":
        return _type(gold, gold) == _type(prediction, prediction)
    if field == "source_quote":
        return _value(gold, "source_quote", "text") == _value(prediction, "source_quote", "text")
    if field == "negation":
        return _bool(_value(gold, "negation")) == _bool(_value(prediction, "negation"))
    if field in {"condition", "condition_quote"}:
        return _same_optional(
            _value(gold, "condition_quote", "condition"),
            _value(prediction, "condition_quote", "condition"),
        )
    if field == "deadline":
        return all(
            _same_optional(expected, actual)
            for expected, actual in zip(_deadline(gold), _deadline(prediction))
        )
    if field == "deadline_quote":
        return _same_optional(
            _value(gold, "deadline_quote", "deadline_text"),
            _value(prediction, "deadline_quote", "deadline_text"),
        )
    if field in {"observable_action", "reported_speech", "ambiguity_allowed"}:
        return _bool(_value(gold, field)) == _bool(_value(prediction, field))
    return _value(gold, field) == _value(prediction, field)


def _metric(correct: int, total: int) -> dict[str, Any]:
    return {
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else None,
    }


def _gold_proposition(row: Mapping[str, Any]) -> Mapping[str, Any]:
    gold = row.get("gold")
    propositions = gold.get("propositions") if isinstance(gold, Mapping) else None
    if not isinstance(propositions, Sequence) or isinstance(propositions, (str, bytes)):
        raise TypeError(f"{row.get('record_id')}: missing gold propositions")
    if not propositions or not isinstance(propositions[0], Mapping):
        raise ValueError(f"{row.get('record_id')}: empty gold propositions")
    return propositions[0]


def evaluate_semantic_predictions(
    records: Sequence[Mapping[str, Any]],
    predictions: Mapping[str, Any] | Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Report type, literal-anchor, critical-field, unknown, and missing scores."""

    rows = [dict(row) for row in records]
    predicted = _prediction_map(predictions, "record_id")
    ids = [str(row.get("record_id")) for row in rows]
    missing = [item for item in ids if item not in predicted]
    unexpected = sorted(set(predicted) - set(ids))
    by_class: dict[str, dict[str, Any]] = {}
    field_counts: dict[str, list[int]] = {}
    type_correct = 0
    anchors = [0, 0]
    critical = [0, 0]
    unknown: list[str] = []
    abstentions: list[str] = []
    for row in rows:
        record_id = str(row.get("record_id"))
        gold = _gold_proposition(row)
        gold_type = _type(gold, gold)
        if gold_type is None:
            raise ValueError(f"{record_id}: gold semantic type is empty")
        class_counts = by_class.setdefault(gold_type, {"correct": 0, "total": 0, "accuracy": None})
        class_counts["total"] += 1
        output = predicted.get(record_id)
        proposition = _first_proposition(output)
        predicted_type = _type(output, proposition)
        if predicted_type == gold_type:
            class_counts["correct"] += 1
            type_correct += 1
        if record_id in predicted and predicted_type in {None, "UNKNOWN", "UNRESOLVED", "ABSTAIN"}:
            unknown.append(record_id)
        if record_id in predicted and _is_abstention(output, proposition):
            abstentions.append(record_id)
        expected_quote = _value(gold, "source_quote", "text")
        actual_quote = _value(proposition, "source_quote", "text")
        anchors[1] += 1
        if isinstance(expected_quote, str) and isinstance(actual_quote, str) and expected_quote in actual_quote:
            anchors[0] += 1
        fields = (row.get("gold") or {}).get("critical_fields", ["semantic_type", "source_quote"])
        if not isinstance(fields, Sequence) or isinstance(fields, (str, bytes)):
            fields = ["semantic_type", "source_quote"]
        for field in fields:
            field = "condition" if field == "condition_quote" else str(field)
            counts = field_counts.setdefault(field, [0, 0])
            counts[1] += 1
            critical[1] += 1
            if _field_match(gold, proposition, field):
                counts[0] += 1
                critical[0] += 1
    for counts in by_class.values():
        counts["accuracy"] = counts["correct"] / counts["total"] if counts["total"] else None
    field_accuracy = {name: _metric(*counts) for name, counts in sorted(field_counts.items())}
    if "deadline_quote" in field_accuracy:
        field_accuracy["deadline"] = field_accuracy["deadline_quote"]
    return {
        "evaluation_scope": "AI_source_read_real_source_semantic_fixture",
        "record_count": len(rows),
        "prediction_count": len(predicted),
        "missing_record_count": len(missing),
        "missing_record_ids": missing,
        "unexpected_prediction_count": len(unexpected),
        "unexpected_prediction_ids": unexpected,
        "type_accuracy": _metric(type_correct, len(rows)),
        "type_accuracy_per_class": dict(sorted(by_class.items())),
        "exact_anchor_coverage": _metric(*anchors),
        "field_accuracy": field_accuracy,
        "critical_field_accuracy": _metric(*critical),
        "unknown_prediction_count": len(unknown),
        "unknown_prediction_ids": unknown,
        "abstention_count": len(abstentions),
        "abstention_ids": abstentions,
    }


def evaluate_relation_predictions(
    records: Sequence[Mapping[str, Any]],
    predictions: Mapping[str, Any] | Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Report relation, candidate, and all-pair status metrics.

    A missing receipt is not silently removed from a denominator.  It is a
    missing pair (and therefore an incorrect status/field prediction), while
    an explicit ``UNRESOLVED``/``ABSTAIN`` result is counted as a model
    abstention.  Candidate actor/time/domain/terminal fields are scored as
    fields of a relation candidate; they are never substituted for the
    relation ``status`` itself.
    """

    rows = [dict(row) for row in records]
    predicted = _prediction_map(predictions, "pair_id")
    ids = [str(row.get("pair_id")) for row in rows]
    expected_ids = set(ids)
    missing = [item for item in ids if item not in predicted]
    unexpected = sorted(set(predicted) - expected_ids)
    fields = (
        "status",
        "target_scope",
        "domain_relation",
        "actor_eligibility",
        "time_eligibility",
        "terminal_eligibility",
        "action_alignment",
    )
    # Every requested pair is a denominator, including a missing receipt.
    counts = {field: [0, len(rows)] for field in fields}
    unknown: list[str] = []
    abstentions: list[str] = []
    positive_statuses = {"SAME_POLICY_OBJECT", "SAME_MATTER"}
    true_positive = false_positive = false_negative = 0
    status_correct = 0
    status_counts: dict[str, dict[str, int | float | None]] = {}

    def relation_value(output: Mapping[str, Any], field: str) -> Any:
        if field in {"domain_relation", "actor_eligibility", "time_eligibility", "terminal_eligibility", "action_alignment"}:
            candidate = output.get("candidate")
            if isinstance(candidate, Mapping) and field in candidate:
                return candidate.get(field)
        return output.get(field)

    for row in rows:
        pair_id = str(row.get("pair_id"))
        gold = row.get("gold")
        if not isinstance(gold, Mapping):
            raise TypeError(f"{pair_id}: missing relation gold")
        output = predicted.get(pair_id)
        if not isinstance(output, Mapping):
            if pair_id in predicted:
                unknown.append(pair_id)
                abstentions.append(pair_id)
            gold_status = str(gold.get("status") or "").upper()
            if gold_status in positive_statuses:
                false_negative += 1
            continue

        status = output.get("status")
        status = status.strip().upper() if isinstance(status, str) else None
        if status in {None, "UNKNOWN", "UNRESOLVED", "ABSTAIN", "MISSING"}:
            unknown.append(pair_id)
        if status in {None, "UNKNOWN", "UNRESOLVED", "ABSTAIN", "MISSING"} or _is_abstention(output, None):
            abstentions.append(pair_id)
        gold_status = str(gold.get("status") or "").upper()
        if status == gold_status:
            status_correct += 1
        predicted_positive = status in positive_statuses
        gold_positive = gold_status in positive_statuses
        if predicted_positive and gold_positive:
            true_positive += 1
        elif predicted_positive:
            false_positive += 1
        elif gold_positive:
            false_negative += 1
        for field in fields:
            actual = relation_value(output, field)
            expected = gold.get("action_alignment") if field == "action_alignment" else gold.get(field)
            if actual == expected:
                counts[field][0] += 1

    gold_status_values = sorted({str((row.get("gold") or {}).get("status") or "") for row in rows})
    for status in gold_status_values:
        total = sum(
            1 for row in rows
            if str((row.get("gold") or {}).get("status") or "") == status
        )
        correct = sum(
            1 for row in rows
            if str((row.get("gold") or {}).get("status") or "") == status
            and isinstance(predicted.get(str(row.get("pair_id"))), Mapping)
            and str(predicted[str(row.get("pair_id"))].get("status") or "").upper() == status
        )
        predicted_for_status = sum(
            1 for pair_id in ids
            if isinstance(predicted.get(pair_id), Mapping)
            and str(predicted[pair_id].get("status") or "").upper() == status
        )
        status_true_positive = correct
        status_false_positive = max(0, predicted_for_status - status_true_positive)
        status_false_negative = max(0, total - status_true_positive)
        status_precision_denominator = status_true_positive + status_false_positive
        status_recall_denominator = status_true_positive + status_false_negative
        status_counts[status] = {
            "correct": correct,
            "total": total,
            "accuracy": correct / total if total else None,
            "predicted": predicted_for_status,
            "precision": status_true_positive / status_precision_denominator if status_precision_denominator else None,
            "recall": status_true_positive / status_recall_denominator if status_recall_denominator else None,
        }

    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    status_metrics = {
        "accuracy": _metric(status_correct, len(rows)),
        "accuracy_per_class": status_counts,
        "positive_statuses": sorted(positive_statuses),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": true_positive / precision_denominator if precision_denominator else None,
        "recall": true_positive / recall_denominator if recall_denominator else None,
        "f1": (2 * true_positive / (2 * true_positive + false_positive + false_negative)
               if 2 * true_positive + false_positive + false_negative else None),
    }
    return {
        "evaluation_scope": "AI_source_read_real_source_relation_fixture",
        "pair_count": len(rows),
        "prediction_count": len(predicted),
        "missing_pair_count": len(missing),
        "missing_pair_ids": missing,
        "unexpected_prediction_count": len(unexpected),
        "unexpected_prediction_ids": unexpected,
        "status_metrics": status_metrics,
        "status_precision": status_metrics["precision"],
        "status_recall": status_metrics["recall"],
        "status_f1": status_metrics["f1"],
        "abstention_rate": len(abstentions) / len(rows) if rows else None,
        "missing_rate": len(missing) / len(rows) if rows else None,
        "field_accuracy": {field: _metric(*value) for field, value in counts.items()},
        "candidate_field_accuracy": {
            field: _metric(*counts[field])
            for field in fields
            if field not in {"status", "target_scope"}
        },
        "unknown_prediction_count": len(unknown),
        "unknown_prediction_ids": unknown,
        "abstention_count": len(abstentions),
        "abstention_ids": abstentions,
    }


def evaluate_predictions(
    records: Sequence[Mapping[str, Any]],
    predictions: Mapping[str, Any] | Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compatibility alias for the semantic report function."""

    return evaluate_semantic_predictions(records, predictions)


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--split", choices=("development", "heldout"))
    parser.add_argument("--relation", action="store_true")
    parser.add_argument("--fixture", type=Path)
    args = parser.parse_args()
    rows = load_jsonl(args.fixture) if args.fixture else (
        load_relation_gold() if args.relation else load_semantic_gold()
    )
    rows = _select(rows, args.split)
    try:
        predictions: Any = json.loads(args.predictions.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        predictions = load_jsonl(args.predictions)
    report = (
        evaluate_relation_predictions(rows, predictions)
        if args.relation
        else evaluate_semantic_predictions(rows, predictions)
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

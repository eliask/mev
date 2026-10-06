import json

from paa.initiative_benchmark import DEFAULT_REVIEWS, build_benchmark


def test_real_initiative_pairs_are_judged_without_review_injection() -> None:
    report = build_benchmark()

    assert report["evaluation_scope"] == "frozen_fixture_only_not_held_out"
    assert report["query"]["actor_name"] == "Arja Juvonen"
    assert report["object_count"] == 7
    assert report["gold_positive_count"] == 2
    assert report["hard_negative_count"] == 5
    assert report["candidate_competition"]["indexed_object_count"] == 7
    assert report["candidate_competition"]["candidate_count"] == 4
    assert report["candidate_competition"]["review_reference_injections_excluded"] == 0
    assert report["positive_ranks"] == [1, 2]
    assert report["top_k_recall"] == {"1": 0.5, "3": 1.0, "5": 1.0}
    assert report["admitted_relation_count"] == 2
    assert report["admitted_relation_true_positive_count"] == 2
    assert report["admitted_relation_precision"] == 1.0
    assert 0.0 <= report["retrieved_candidate_precision"] <= 1.0
    assert report["validated_hard_negative_count"] == 5
    assert report["hard_negative_admission_count"] == 0
    assert report["retrieval_abstention_count"] + report["hard_negative_retrieved_count"] == 5
    assert {item["gold_label"] for item in report["judgments"]} == {
        "POSITIVE_SAME_POLICY_OBJECT",
        "HARD_NEGATIVE_REJECTED",
    }
    assert all(item["review_status"] in {"SAME_POLICY_OBJECT", "REJECTED"} for item in report["judgments"])
    assert all(item["validation_state"] == "VALID" for item in report["judgments"])


def test_review_admission_is_measured_against_gold_not_generated_from_it(tmp_path):
    reviews = [json.loads(line) for line in DEFAULT_REVIEWS.read_text().splitlines() if line.strip()]
    wrong = next(row for row in reviews if row["matter_id"] == "LA 1/2022 vp")
    wrong["status"] = "SAME_POLICY_OBJECT"
    wrong["normalized_target"] = "Incorrectly claimed elder-care staffing target"
    altered = tmp_path / "reviews.jsonl"
    altered.write_text("\n".join(json.dumps(row) for row in reviews))
    report = build_benchmark(reviews_path=altered)
    assert report["admitted_relation_count"] == 3
    assert report["hard_negative_admission_count"] == 1
    assert report["admitted_relation_precision"] == 2 / 3


def test_missing_review_abstains_instead_of_using_gold_label(tmp_path):
    reviews = [json.loads(line) for line in DEFAULT_REVIEWS.read_text().splitlines() if line.strip()]
    missing = next(row["object_id"] for row in reviews if row["matter_id"] == "LA 72/2017 vp")
    altered = tmp_path / "reviews.jsonl"
    altered.write_text("\n".join(json.dumps(row) for row in reviews if row["object_id"] != missing))
    report = build_benchmark(reviews_path=altered)
    assert report["admitted_relation_count"] == 1
    assert next(row for row in report["judgments"] if row["object_id"] == missing)["validation_state"] == "UNREVIEWED"

"""Offline judged relation pairs for the initiative retrieval seam.

This is a small, source-anchored development evaluation, not a claim about
the quality of the whole initiative register.  The positive pairs use the
real Arja Juvonen / ``LA 72/2017 vp`` elder-care staffing case and a second
official matter on the same law and staffing section.  Five distinct
official-register objects are retained as explicit ``REJECTED`` hard
negatives.  All seven normalized texts and source evidence IDs come from the
frozen retrieval-object fixture; none are hand-written lookalikes.

The benchmark deliberately runs the same lexical candidate generator as the
trace compiler and never injects a reviewed object into the candidate list.
Review validation is measured separately from retrieval, and the report is
explicitly illustrative rather than held-out or population-level evidence.
"""


import json
from pathlib import Path
from typing import Any

from paa.config import FIXTURE_DIR
from paa.relations import ObjectRetriever, fingerprint, verified_review

DEFAULT_CASE = FIXTURE_DIR / "initiative_case_arja_juvonen.json"
DEFAULT_OBJECTS = FIXTURE_DIR / "initiative_retrieval_objects.jsonl"
DEFAULT_REVIEWS = FIXTURE_DIR / "initiative_relation_reviews.jsonl"

_POSITIVE_MATTERS = {"LA 72/2017 vp", "LA 80/2018 vp"}
_NEGATIVE_RATIONALES = {
    "LA 32/2019 vp": "Saattohoidon järjestämistä koskeva aloite on eri policy object from the elder-care staffing-minimum target.",
    "LA 16/2022 vp": "Foreign-property acquisition is unrelated to the elder-care staffing-minimum target.",
    "LA 71/2022 vp": "Animal-rights constitutional amendments are unrelated to the elder-care staffing-minimum target.",
    "LA 1/2022 vp": "Repealing pandemic staffing and vaccination provisions is unrelated to the elder-care staffing-minimum target.",
    "LA 25/2025 vp": "Employment leave for end-of-life care is distinct from the elder-care staffing-minimum target.",
}


def _load_objects(path: Path | str) -> list[dict[str, Any]]:
    objects = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").split("\n")
        if line.strip()
    ]
    if len(objects) < 2:
        raise ValueError("initiative retrieval fixture must contain a target and at least one hard negative")
    return objects


def _atomic_query(case: dict[str, Any]) -> str:
    """Use the reviewed staffing proposition, not the whole campaign answer."""

    query = case["relation_review"].get("statement_quote") or case["candidate"]["quote"]
    if "hoitajavahvuus." in query:
        query = query.split("hoitajavahvuus.", 1)[0] + "hoitajavahvuus."
    return query


def build_benchmark(
    *,
    case_path: Path | str = DEFAULT_CASE,
    objects_path: Path | str = DEFAULT_OBJECTS,
    reviews_path: Path | str = DEFAULT_REVIEWS,
) -> dict[str, Any]:
    """Return a deterministic report for source-backed positive/negative pairs."""

    case = json.loads(Path(case_path).read_text(encoding="utf-8"))
    source_objects = _load_objects(objects_path)
    review_rows = [json.loads(line) for line in Path(reviews_path).read_text(encoding="utf-8").split("\n") if line.strip()]
    reviews = {row["object_id"]: row for row in review_rows}
    if len(reviews) != len(review_rows):
        raise ValueError("duplicate official object in independent review records")
    target_id = f"eduskunta:{case['official_source']['matter_id']}"
    query = _atomic_query(case)
    objects: list[dict[str, Any]] = []
    for item in source_objects:
        obj = dict(item)
        matter_id = obj["matter_id"]
        if obj["object_id"] == target_id:
            obj.update(
                benchmark_label="POSITIVE_SAME_POLICY_OBJECT",
                review_id=case["relation_review"]["review_id"],
                benchmark_rationale=case["relation_review"]["rationale"],
            )
        elif matter_id in _POSITIVE_MATTERS:
            obj.update(
                benchmark_label="POSITIVE_SAME_POLICY_OBJECT",
                review_id=f"review-arja-policy-{matter_id.replace('/', '-')}",
                benchmark_rationale=(
                    "The official title and text address the same elder-care law §20 staffing object. "
                    "This is a policy-object relation only; its author is not attributed to Arja Juvonen."
                ),
            )
        else:
            obj.update(
                benchmark_label="HARD_NEGATIVE_REJECTED",
                review_id=f"review-reject-arja-{matter_id.replace('/', '-')}",
                benchmark_rationale=_NEGATIVE_RATIONALES.get(
                    matter_id, "Different official matter; lexical overlap is not identity."
                ),
            )
        objects.append(obj)

    ranked = ObjectRetriever(objects).search(query, k=len(objects))
    positions = {item["object_id"]: index + 1 for index, item in enumerate(ranked)}
    scores = {item["object_id"]: item.get("score") for item in ranked}
    positive_objects = [item for item in objects if item["benchmark_label"] == "POSITIVE_SAME_POLICY_OBJECT"]
    negative_objects = [item for item in objects if item["benchmark_label"] == "HARD_NEGATIVE_REJECTED"]

    statement_evidence_id = "yle2011-3823-e"
    statement = {"original_text": query, "evidence": [{"evidence_id": statement_evidence_id}]}
    proposition = {"proposition_id": case["relation_review"]["proposition_id"], "source_text": query}
    evidence = {statement_evidence_id: statement["evidence"][0]}
    for obj in objects:
        for evidence_id in obj.get("evidence_ids", []):
            evidence[evidence_id] = {"evidence_id": evidence_id}

    judgments: list[dict[str, Any]] = []
    for obj in objects:
        label = obj["benchmark_label"]
        # The review pass read the supplied source objects and proposition
        # without access to this module's gold labels. Never manufacture a
        # review status from those evaluation labels.
        review = reviews.get(obj["object_id"])
        valid = review is not None and verified_review(review, statement, proposition, obj, evidence)
        review_status = review["status"] if review else "UNRESOLVED"
        judgments.append(
            {
                "review_id": review["review_id"] if review else None,
                "object_id": obj["object_id"],
                "matter_id": obj["matter_id"],
                "gold_label": label,
                "review_status": review_status,
                "relation": review_status if valid else "UNRESOLVED",
                "action_alignment": "RELATED_POLICY_OBJECT" if review_status == "SAME_POLICY_OBJECT" else None,
                "retrieved_rank": positions.get(obj["object_id"]),
                "retrieval_score": scores.get(obj["object_id"]),
                "rationale": review["rationale"] if review else "No independent review supplied.",
                "evidence_ids": obj.get("evidence_ids", []),
                "validation_state": "VALID" if valid else "STALE_OR_UNGROUNDED" if review else "UNREVIEWED",
                "admitted": bool(valid and review_status in {"SAME_POLICY_OBJECT", "SAME_MATTER"}),
                "gold_positive": label == "POSITIVE_SAME_POLICY_OBJECT",
                "actor_match": any(
                    author.get("person_id") == case["candidate"].get("person_id")
                    for author in obj.get("authors", [])
                ),
            }
        )

    admitted_items = [item for item in judgments if item["admitted"]]
    admitted = len(admitted_items)
    true_positive_admissions = sum(1 for item in admitted_items if item["gold_positive"])
    valid_hard_negatives = sum(
        1 for item in judgments if item["validation_state"] == "VALID" and item["gold_label"] == "HARD_NEGATIVE_REJECTED"
    )
    retrieved_hard_negatives = sum(
        1 for item in judgments if item["gold_label"] == "HARD_NEGATIVE_REJECTED" and item["retrieved_rank"] is not None
    )
    retrieved_positive_count = sum(
        1 for item in judgments if item["gold_label"] == "POSITIVE_SAME_POLICY_OBJECT" and item["retrieved_rank"] is not None
    )
    retrieved_candidate_precision = retrieved_positive_count / len(ranked) if ranked else 0.0
    retrieval_abstentions = sum(
        1 for item in judgments if item["gold_label"] == "HARD_NEGATIVE_REJECTED" and item["retrieved_rank"] is None
    )
    positive_ranks = [item["retrieved_rank"] for item in judgments if item["gold_label"] == "POSITIVE_SAME_POLICY_OBJECT"]
    top_k_recall = {
        str(k): sum(rank is not None and rank <= k for rank in positive_ranks) / len(positive_ranks)
        for k in (1, 3, 5)
    }
    return {
        "schema_version": "1.0",
        "evaluation_scope": "frozen_fixture_only_not_held_out",
        "benchmark_kind": "source_anchored_initiative_relation_development_check",
        "review_method": "separate_source_text_pass_without_gold_labels; AI reviewed; not held out",
        "query": {
            "case_id": case["case_id"],
            "actor_id": case["candidate"]["actor_id"],
            "actor_name": case["candidate"]["name"],
            "text": query,
            "source_id": case["candidate"]["source_id"],
            "source_url": case["candidate"]["source_url"],
            "statement_sha256": fingerprint(query),
            "source_statement_sha256": case["candidate"]["statement_sha256"],
        },
        "object_count": len(objects),
        "judged_pair_count": len(judgments),
        "gold_positive_count": len(positive_objects),
        "hard_negative_count": len(negative_objects),
        "judgments": judgments,
        "candidate_competition": {
            "indexed_object_count": len(objects),
            "candidate_count": len(ranked),
            "retrieved_candidate_count": len(ranked),
            "judged_candidate_count": len(judgments),
            "candidates": ranked,
            "unretrieved_judged_object_ids": [
                item["object_id"] for item in judgments if item["retrieved_rank"] is None
            ],
            "review_reference_injections_excluded": 0,
        },
        "top_k_recall": top_k_recall,
        "positive_ranks": positive_ranks,
        "admitted_relation_count": admitted,
        "admitted_relation_true_positive_count": true_positive_admissions,
        "admitted_relation_precision": true_positive_admissions / admitted if admitted else 0.0,
        "retrieved_candidate_precision": retrieved_candidate_precision,
        "validated_hard_negative_count": valid_hard_negatives,
        "hard_negative_retrieved_count": retrieved_hard_negatives,
        "retrieval_abstention_count": retrieval_abstentions,
        "retrieval_abstention_rate": retrieval_abstentions / len(negative_objects) if negative_objects else 0.0,
        "hard_negative_admission_count": sum(
            1 for item in judgments if item["admitted"] and item["gold_label"] == "HARD_NEGATIVE_REJECTED"
        ),
        "notes": [
            "The first positive pair is the source-anchored Arja Juvonen / LA 72/2017 vp relation review.",
            "The second positive is a same-policy-object review for LA 80/2018 vp; it is not attributed to Juvonen.",
            "The negative candidates are full normalized official-register source snapshots with preserved evidence IDs.",
            "REJECTED labels are adjudicated hard negatives; retrieval candidates are not admitted relations.",
            "Admission uses separately persisted source-text reviews, not statuses generated from the gold labels. Precision compares their admissions with those fixture labels.",
            "Retrieval abstention means an indexed hard-negative object did not enter the lexical candidate list; it is not the Sanna trace's review abstention.",
            "This is illustrative development evidence, not a held-out or population-quality benchmark.",
        ],
    }


__all__ = ["DEFAULT_CASE", "DEFAULT_OBJECTS", "build_benchmark"]

import json
from pathlib import Path

from paa.llm_facets_evaluation import evaluate_facet_run


def test_unmapped_legacy_kind_is_excluded_from_agreement_denominator(tmp_path: Path) -> None:
    run = tmp_path / "run" / "cases"
    run.mkdir(parents=True)
    (run / "doc-1.json").write_text(
        json.dumps(
            {
                "item_id": "doc-1",
                "outcome": {"receipt_status": "OK"},
                "extraction": {
                    "facets": [
                        {
                            "kind": "POLICY_GOAL",
                            "source_quote": "Tavoittelen koulutusta.",
                            "source_start": 0,
                            "source_end": 23,
                        }
                    ],
                    "invalid_records": [],
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    references = (
        {
            "document": {"document_id": "doc-1", "source_text": "Tavoittelen koulutusta."},
            "gold": {
                "propositions": [
                    {
                        "source_quote": "Tavoittelen koulutusta.",
                        "semantic_type": "PROCESS_COMMITMENT",
                    }
                ]
            },
        },
    )

    score = evaluate_facet_run(tmp_path / "run", references)

    assert score.matched_facets == 1
    assert score.kind_checked == 0
    assert score.kind_agreed == 0
    assert score.ontology_uncomparable_kinds == {"PROCESS_COMMITMENT": 1}

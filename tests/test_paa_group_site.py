"""Cold-reader checks for the descriptive group-comparison pages."""

from __future__ import annotations

from pathlib import Path

from paa.group_site import write_group_pages


def test_pages_explain_formal_matter_sensitivity_without_policy_claim(tmp_path: Path) -> None:
    source = {
        "source_id": "eduskunta:ballots:1",
        "vote_id": "1",
        "session_date": "2026-01-01",
        "matter": "HE 109/2024 vp",
        "title": "Testiäänestys",
        "source_url": "https://example.test/vote/1",
        "published_totals": {"JAA": 2, "EI": 1, "TYHJA": 0, "POISSA": 0, "TOTAL": 3},
        "state": "VALID",
        "source_sha256": "source-hash",
        "rows": [
            {
                "person_id": "1",
                "first_name": "Aino",
                "last_name": "Testi",
                "group_code": "kok",
                "response": "JAA",
            }
        ],
    }
    packet = {
        "person_id": "1",
        "display_name": "Aino Testi",
        "cutoff": "2026-10-07",
        "minimum_peer_count": 2,
        "summary": {
            "matching_count": 1,
            "comparable_count": 1,
            "contested_matching_count": 1,
            "contested_comparable_count": 1,
            "recorded_absence_count": 0,
            "recorded_abstention_count": 0,
            "source_row_missing_count": 0,
            "absent_count": 0,
            "matter_cluster_diagnostic": {
                "labels_denominator": 2,
                "comparable_votes_with_label": 4,
                "comparable_votes_without_label": 1,
                "mean_within_label_agreement": 0.625,
                "largest_label": "HE 109/2024 vp",
                "largest_label_count": 4,
                "largest_label_matching_count": 3,
                "largest_label_agreement": 0.75,
                "agreement_with_largest_label_removed": 0.5,
            },
        },
        "comparisons": [],
    }
    report = {
        "valid_source_count": 2,
        "matter_sensitivity": {
            "largest_label": "HE 109/2024 vp",
            "largest_label_count": 2,
            "largest_label_share_of_valid_sources": 1.0,
        },
    }

    write_group_pages([packet], [source], report, tmp_path)

    person_html = (tmp_path / "person-1.html").read_text(encoding="utf-8")
    index_html = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert "Asian lähdetunnisteen herkkyystarkistus" in person_html
    assert "HE 109/2024 vp" in person_html
    assert "62.5 %" in person_html
    assert "75.0 %" in person_html
    assert "50.0 %" in person_html
    assert "ei automaattisesti tarkoita yhtä politiikkakysymystä" in person_html
    assert "Lähdetunnisteiden herkkyys" in index_html
    assert "100.0 %" in index_html
    assert "ei yksi politiikkakysymys" in index_html


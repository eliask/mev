"""Bootstrap reader projection and public-export controls."""

from pathlib import Path

import pytest

from paa.bootstrap_site import (
    BootstrapSiteError,
    _packet_source_map,
    _public_error_message,
    _public_json_value,
    _public_question_packet,
    _public_text_has_filesystem_leak,
    _question_article,
    build_bootstrap_artifact,
    build_oulaskangas_dossier,
    build_oulaskangas_dossier_packet,
)


def test_unbound_claim_is_not_rendered_as_blockquote() -> None:
    html = _question_article(
        {
            "task_id": "T1",
            "matter_id": "HE 1/2023 vp",
            "question": "Kysymys?",
            "status": "OK",
            "admission": "CANDIDATE_NOT_REFERENCE_JUDGED",
            "answer": "Ehdokas.",
            "claims": [
                {
                    "text": "Sidottu väite.",
                    "epistemic_status": "warranted_positive",
                    "evidence_bound": [
                        {
                            "source_id": "src-a",
                            "first_pointer": 0,
                            "last_pointer": 0,
                            "start": 0,
                            "end": 12,
                            "quote": "Alkuperäinen ote joka on pitkä " + ("x" * 500),
                            "source_text_sha256": "a" * 64,
                        }
                    ],
                },
                {
                    "text": "Sitomaton ehdokasväite jota ei saa lainata lähteenä.",
                    "epistemic_status": "open",
                    "unbound": True,
                    "evidence_bound": [],
                },
            ],
            "coverage": {
                "requested_source_ids": ["src-a"],
                "indexed_source_ids": ["src-a"],
                "fully_read_source_ids": [],
                "read_completeness": "PARTIAL",
            },
            "read_completeness": "PARTIAL",
            "next_evidence": ["Tarvitaan taulukon otsikot."],
            "error": "map window failed",
        },
        sources={"src-a": {"title": "Lähde A", "url": "https://example.test/a", "source_locator": "loc-a"}},
        packet_json_href="T1.json",
    )
    assert "Alkuperäinen ote" in html
    assert "xxxx" in html
    assert "<details open>" not in html
    assert "Sitomaton ehdokasväite" in html
    assert "Ei sidottua lähdekatkelmaa" in html
    unbound_pos = html.index("Sitomaton ehdokasväite")
    after = html[unbound_pos : unbound_pos + 400]
    assert "<blockquote>" not in after
    assert "https://example.test/a" in html
    assert "<h3>Kysymys?</h3>" in html
    assert "<h3>T1:" not in html
    assert "Konetunnisteet" in html
    assert ">T1<" in html
    assert "0 täysin luettu" in html
    assert "PARTIAL" in html or "Osittain" in html
    assert "ei semanttinen hyväksyntä" in html
    # Coverage/pipeline counts stay available but not as the default answer voice.
    assert "<details><summary>Tekninen kattavuus ja sidonta</summary>" in html
    coverage_default = html.split("<details><summary>Tekninen kattavuus ja sidonta</summary>", 1)[0]
    assert "0 täysin luettu" not in coverage_default
    # Full candidate/raw answer is folded; concise status/question lead.
    assert "full-candidate" in html
    assert "<p><strong>Ehdokasvastaus:</strong>" not in html
    assert "<p><strong>Koko ehdokasvastaus:</strong>" not in html
    assert "Tarvitaan taulukon otsikot." in html
    assert "map window failed" in html
    assert "href='T1.json'" in html or 'href="T1.json"' in html
    assert "Lataa ehdokaspaketti" in html
    assert "download" in html


def test_unbound_true_suppresses_malformed_evidence_rows() -> None:
    html = _question_article(
        {
            "task_id": "T2",
            "question": "Toinen?",
            "status": "PARTIAL",
            "admission": "CANDIDATE_NOT_REFERENCE_JUDGED",
            "answer": "Ehdokas.",
            "claims": [
                {
                    "text": "Virheellinen unbound+quote.",
                    "unbound": True,
                    "evidence_bound": [
                        {
                            "source_id": "src-b",
                            "quote": "Tätä ei saa näyttää blockquotena.",
                            "start": 0,
                            "end": 10,
                        }
                    ],
                }
            ],
            "coverage": {"requested_source_ids": ["src-b"], "indexed_source_ids": ["src-b"]},
        }
    )
    assert "Virheellinen unbound+quote." in html
    assert "Tätä ei saa näyttää blockquotena." not in html
    assert "<blockquote>" not in html


def test_pathful_error_is_redacted_in_html_and_public_json() -> None:
    packet = {
        "task_id": "T3",
        "question": "Virhe?",
        "status": "FAILED",
        "admission": "CANDIDATE_NOT_REFERENCE_JUDGED",
        "answer": "Ei.",
        "error": "REQUEST failed reading /mnt/private/workspace/.tmp/secret.json",
        "claims": [],
        "coverage": {},
        "request_id": "should-not-publish",
        "map_findings": [{"secret": True}],
        "internal_telemetry_count": 99,
    }
    html = _question_article(packet)
    assert "/mnt/" not in html
    assert ".tmp/" not in html
    assert "Käsittelyvirhe" in html
    assert _public_error_message(packet["error"]).startswith("Käsittelyvirhe")
    public = _public_question_packet(packet)
    assert public["error"].startswith("Käsittelyvirhe")
    assert public["error_detail_retained_privately"] is True
    assert "request_id" not in public
    assert "map_findings" not in public
    assert "internal_telemetry_count" not in public
    assert "binding_errors" not in _public_question_packet(
        {**packet, "binding_errors": ["SECRET_BIND"], "binding_ok": False, "binding_state": "EMPTY"}
    )


def test_public_text_filesystem_leak_ignores_url_embedded_home() -> None:
    """Official archival sftp/https URLs may contain /home/; local paths must still fail."""

    concatenated = (
        "pdfsftp://pintpurnukka.eduskunta.fi:22/home/vaskijul/liitteet/"
        "2EA53789.pdfapplication/pdf"
    )
    assert _public_text_has_filesystem_leak(concatenated) is False
    assert (
        _public_text_has_filesystem_leak(
            "see https://example.test/path/home/docs/file.pdf for attachment"
        )
        is False
    )
    assert _public_text_has_filesystem_leak("/home/example/secret.json") is True
    assert _public_text_has_filesystem_leak(".tmp/internal/cache/x") is True


def test_public_json_rejects_suffix_telemetry_keys() -> None:
    nested = _public_json_value(
        {
            "answer": "ok",
            "coverage": {
                "requested_source_ids": ["a"],
                "fully_read_source_ids": [],
                "secret_slot_count": 3,
                "provider_latency_ms": 12,
            },
            "finding_count": 7,
        }
    )
    assert nested["answer"] == "ok"
    assert nested["coverage"]["requested_source_ids"] == ["a"]
    assert "secret_slot_count" not in nested["coverage"]
    assert "finding_count" not in nested


def test_editorial_row_provenance_and_projection_disclosure() -> None:
    sources = _packet_source_map(
        {
            "rows": [
                {
                    "title": "Rivi",
                    "evidence_bound": [
                        {
                            "source_id": "news-1",
                            "title": "Uutisotsikko",
                            "url": "https://example.test/news",
                            "source_locator": "kappale-2",
                            "quote": "Alku […] loppu",
                            "start": 10,
                            "end": 40,
                            "source_text_sha256": "b" * 64,
                            "projection": True,
                        }
                    ],
                }
            ]
        }
    )
    assert sources["news-1"]["title"] == "Uutisotsikko"
    html = build_oulaskangas_dossier(
        group_packets=[],
        cost_packets=[],
        monitoring_packets=None,
        dossier_json_href="oulaskangas.json",
        editorial_packet={
            "rows": [
                {
                    "title": "Ryhmäuutinen",
                    "text": "Ehdokasrivin teksti.",
                    "voice": "NEWS",
                    "stage": "REPORTED",
                    "evidence_bound": [
                        {
                            "source_id": "news-1",
                            "title": "Uutisotsikko",
                            "url": "https://example.test/news",
                            "source_locator": "kappale-2",
                            "quote": "Alku […] loppu",
                            "start": 10,
                            "end": 40,
                            "source_text_sha256": "b" * 64,
                            "projection": True,
                        }
                    ],
                }
            ]
        },
    )
    assert "Toteutunut seuranta" not in html
    assert "Uutisvälitteiset lähteet; alkuperäistä ryhmäpäätöstä" not in html
    assert "https://example.test/news" in html
    assert "projektio/katkelma" in html
    assert "Alku […] loppu" in html
    assert "oulaskangas.json" in html
    packet = build_oulaskangas_dossier_packet(
        group_packets=[],
        cost_packets=[],
        editorial_packet={"rows": [{"title": "x", "text": "y", "evidence_bound": []}]},
    )
    assert packet["editorial"]["rows"][0]["title"] == "x"
    assert packet["admission"] == "CANDIDATE_PREVIEW_NOT_REFERENCE_JUDGED"


def test_artifact_requires_clean_destination_and_hashes_matter_json(tmp_path: Path) -> None:
    # Fixed source-shaped input: developer caches must not change this test.
    matter_path = tmp_path / "matter.json"
    matter = {
        "schema_version": "paa.matter_ledger.v2",
        "query": {"mode": "EXACT_MATTER", "matter_id": "HE 1/2023 vp"},
        "matter": {"matter_id": "HE 1/2023 vp", "title": "HE 1/2023 vp"},
        "actors": [],
        "interventions": [
            {
                "kind": "FORMAL_ALTERNATIVE",
                "title": "Vaihtoehto",
                "actor": {"name": "Testi Henkilö", "person_id": "p1"},
                "role_label": "allekirjoittaja",
                "institutional_result": {"state": "UNKNOWN"},
                "source": {"object_id": "obj-1", "title": "Lähde"},
            }
        ],
        "coverage": {},
        "missingness": [],
        "limits": [],
    }
    matter_path.write_text(__import__("json").dumps(matter), encoding="utf-8")
    expected_matter_json = "matters/HE_1_2023_vp.json"
    question = {
        "schema_version": "paa.source_question.v4",
        "task_id": "D09",
        "matter_id": "KK 1/2023 vp",
        "question": "Kuka vastasi?",
        "status": "OK",
        "admission": "CANDIDATE_NOT_REFERENCE_JUDGED",
        "answer": "Tuppurainen.",
        "claims": [],
        "coverage": {
            "requested_source_ids": ["s"],
            "indexed_source_ids": ["s"],
            "fully_read_source_ids": ["s"],
            "read_completeness": "COMPLETE",
        },
        "error": "failed under /mnt/private/workspace/.tmp/x",
        "request_id": "internal",
    }
    question_path = tmp_path / "D09.json"
    question_path.write_text(__import__("json").dumps(question), encoding="utf-8")
    out = tmp_path / "artifact"
    out.mkdir()
    (out / "stale.txt").write_text("old\n", encoding="utf-8")
    with pytest.raises(BootstrapSiteError, match="not empty"):
        build_bootstrap_artifact(
            output_dir=out,
            matter_packets=[matter_path],
            question_packets=[question_path],
            title="Poliittinen muisti",
        )
    clean = tmp_path / "artifact_clean"
    manifest = build_bootstrap_artifact(
        output_dir=clean,
        matter_packets=[matter_path],
        question_packets=[question_path],
        title="Poliittinen muisti",
    )
    assert manifest["title"] == "Poliittinen muisti"
    assert expected_matter_json in manifest["file_sha256"]
    assert "oulaskangas.json" in manifest["file_sha256"]
    assert "questions/D09_KK_1_2023_vp.json" in manifest["file_sha256"]
    html = (clean / "questions" / "D09_KK_1_2023_vp.html").read_text(encoding="utf-8")
    assert "Lataa ehdokaspaketti" in html
    assert "/mnt/" not in html
    public = __import__("json").loads((clean / "questions" / "D09_KK_1_2023_vp.json").read_text())
    assert "request_id" not in public
    assert public["error"].startswith("Käsittelyvirhe")
    dossier = __import__("json").loads((clean / "oulaskangas.json").read_text())
    assert dossier["schema_version"].startswith("paa.bootstrap_site")
    index = (clean / "index.html").read_text(encoding="utf-8")
    assert "<h1>Poliittinen muisti</h1>" in index
    assert "PAA bootstrap" not in index


def test_review_binds_to_displayed_question_bytes_not_unrelated_candidate_path(
    tmp_path: Path,
) -> None:
    import hashlib
    import json

    from paa.bootstrap_site import BootstrapSiteError

    questions = tmp_path / "questions.json"
    sources = tmp_path / "sources.json"
    questions.write_text(json.dumps({"tasks": [{"task_id": "T-REV", "question": "Kuka vastasi?", "matter_id": "KK 1/2023 vp"}]}), encoding="utf-8")
    text_hash = hashlib.sha256(b"ote").hexdigest()
    sources.write_text(json.dumps({"sources": [{"source_id": "S1", "text": "ote", "text_sha256": text_hash}], "task_sources": {"T-REV": ["S1"]}}), encoding="utf-8")
    q_sha = hashlib.sha256(questions.read_bytes()).hexdigest()
    s_sha = hashlib.sha256(sources.read_bytes()).hexdigest()

    good = {
        "schema_version": "paa.source_question.v5",
        "task_id": "T-REV",
        "matter_id": "KK 1/2023 vp",
        "question": "Kuka vastasi?",
        "status": "OK",
        "admission": "CANDIDATE_NOT_REFERENCE_JUDGED",
        "answer": "Ministeri vastasi lähteen mukaan.",
        "claims": [],
        "unresolved": [],
        "next_evidence": [],
        "coverage": {"requested_source_ids": ["S1"], "pointer_domains": {"S1": {"source_text_sha256": text_hash}}},
        "binding_ok": True,
    }
    good_path = tmp_path / "good.json"
    good_bytes = (json.dumps(good, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    good_path.write_bytes(good_bytes)
    tampered = dict(good)
    tampered["answer"] = "TAMPERED ANSWER NOT REVIEWED"
    bad_path = tmp_path / "bad.json"
    bad_path.write_text(json.dumps(tampered, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    review = {
        "schema_version": "paa.source_answer_review.v1",
        "task_id": "T-REV",
        "candidate_sha256": hashlib.sha256(good_bytes).hexdigest(),
        "questions_sha256": q_sha,
        "sources_sha256": s_sha,
        "source_versions": [{"source_id": "S1", "text_sha256": text_hash}],
        "reviewed_fields": ["answer", "claims", "unresolved", "next_evidence"],
        "reviewed_claim_indices": [],
        "reviewer": {
            "kind": "INDEPENDENT_AI_SOURCE_READER",
            "id": "test",
            "human_certification": False,
        },
        "review_date": "2026-10-09",
        "decision": "SUPPORTED_WITH_STATED_LIMITS",
        "reader_title": "Ministerin vastaus",
        "reader_takeaway": "Vastaus on ministerin kertomus lähteessä.",
        "candidate_path": str(good_path),
    }
    review_path = tmp_path / "review.json"
    review_path.write_text(json.dumps(review), encoding="utf-8")
    matter_path = tmp_path / "matter.json"
    matter_path.write_text(
        json.dumps(
            {
                "schema_version": "paa.matter_ledger.v2",
                "query": {"mode": "EXACT_MATTER", "matter_id": "HE 1/2023 vp"},
                "matter": {"matter_id": "HE 1/2023 vp"},
                "actors": [],
                "interventions": [],
                "coverage": {},
                "missingness": [],
                "limits": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(BootstrapSiteError, match="analytical fields differ"):
        build_bootstrap_artifact(
            output_dir=tmp_path / "reject",
            matter_packets=[matter_path],
            question_packets=[bad_path],
            review_paths=[review_path],
            source_bank_paths=[questions, sources],
            reviewed_candidate_paths={"T-REV": good_path},
            title="Poliittinen muisti",
        )

    out = tmp_path / "ok"
    manifest = build_bootstrap_artifact(
        output_dir=out,
        matter_packets=[matter_path],
        question_packets=[good_path],
        review_paths=[review_path],
        source_bank_paths=[questions, sources],
        title="Poliittinen muisti",
    )
    assert manifest["questions"][0]["ai_source_review"] is True
    assert manifest["questions"][0]["reader_title"] == "Ministerin vastaus"
    page = (out / "questions" / "T_REV_KK_1_2023_vp.html").read_text(encoding="utf-8")
    assert "Lähteet tarkistanut erillinen tekoälylukija" in page
    assert "Ministerin vastaus" in page
    assert "TAMPERED" not in page
    index = (out / "index.html").read_text(encoding="utf-8")
    assert "Ministerin vastaus" in index
    assert index.index("Ministerin vastaus") < index.index("Tekniset tiedot")


def test_route_review_rejects_altered_embedded_answer(tmp_path: Path) -> None:
    import hashlib
    import json

    from paa.bootstrap_site import BootstrapSiteError
    from paa.route_packet import SCHEMA_VERSION as ROUTE_SCHEMA

    questions = tmp_path / "questions.json"
    sources = tmp_path / "sources.json"
    questions.write_text(json.dumps({"tasks": [{"task_id": "TR-DEMO", "question": "Mikä ehdotus voitti?", "matter_id": "HE 34/2023 vp"}]}), encoding="utf-8")
    text_hash = hashlib.sha256(b"ote").hexdigest()
    sources.write_text(json.dumps({"sources": [{"source_id": "S1", "text": "ote", "text_sha256": text_hash}], "task_sources": {"TR-DEMO": ["S1"]}}), encoding="utf-8")
    q_sha = hashlib.sha256(questions.read_bytes()).hexdigest()
    s_sha = hashlib.sha256(sources.read_bytes()).hexdigest()

    candidate = {
        "task_id": "TR-DEMO",
        "matter_id": "HE 34/2023 vp",
        "question": "Mikä ehdotus voitti?",
        "status": "OK",
        "admission": "CANDIDATE_NOT_REFERENCE_JUDGED",
        "answer": "Nimetty vaihtoehto hävisi mietinnölle.",
        "claims": [],
        "unresolved": [],
        "next_evidence": [],
        "coverage": {"requested_source_ids": ["S1"], "pointer_domains": {"S1": {"source_text_sha256": text_hash}}},
    }
    cand_path = tmp_path / "candidate.json"
    cand_bytes = (json.dumps(candidate, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    cand_path.write_bytes(cand_bytes)

    route_dir = tmp_path / "route_in"
    route_dir.mkdir()
    route_packet = {
        "schema_version": ROUTE_SCHEMA,
        "admission": "CANDIDATE_NOT_REFERENCE_JUDGED",
        "task_id": "TR-DEMO",
        "matter_id": "HE 34/2023 vp",
        "question": "Mikä ehdotus voitti?",
        "bank": {"questions_sha256": q_sha, "sources_sha256": s_sha},
        "summary": {"stages": []},
        "actors": [],
        "route_edges": [],
        "dates": [],
        "unsupported": [],
        "residuals": [],
        "answer_attachment": {
            "binding": {"state": "BOUND"},
            "candidate_answer": {
                **candidate,
                "answer": "ALTERED EMBEDDED ANSWER",
            },
        },
    }
    (route_dir / "TR-DEMO.json").write_text(
        json.dumps(route_packet, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    review = {
        "schema_version": "paa.source_answer_review.v1",
        "task_id": "TR-DEMO",
        "candidate_sha256": hashlib.sha256(cand_bytes).hexdigest(),
        "questions_sha256": q_sha,
        "sources_sha256": s_sha,
        "source_versions": [{"source_id": "S1", "text_sha256": text_hash}],
        "reviewed_fields": ["answer", "claims", "unresolved", "next_evidence"],
        "reviewed_claim_indices": [],
        "reviewer": {
            "kind": "INDEPENDENT_AI_SOURCE_READER",
            "id": "test",
            "human_certification": False,
        },
        "review_date": "2026-10-09",
        "decision": "SUPPORTED_WITH_STATED_LIMITS",
        "reader_title": "Vaihtoehto hävisi",
        "reader_takeaway": "Rajattu reitti.",
        "candidate_path": str(cand_path),
    }
    review_path = tmp_path / "review.json"
    review_path.write_text(json.dumps(review), encoding="utf-8")
    matter_path = tmp_path / "matter.json"
    matter_path.write_text(
        json.dumps(
            {
                "schema_version": "paa.matter_ledger.v2",
                "query": {"mode": "EXACT_MATTER", "matter_id": "HE 1/2023 vp"},
                "matter": {"matter_id": "HE 1/2023 vp"},
                "actors": [],
                "interventions": [],
                "coverage": {},
                "missingness": [],
                "limits": [],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(BootstrapSiteError, match="embedded candidate_answer"):
        build_bootstrap_artifact(
            output_dir=tmp_path / "reject_route",
            matter_packets=[matter_path],
            question_packets=[],
            route_dirs=[route_dir],
            review_paths=[review_path],
            source_bank_paths=[questions, sources],
            reviewed_candidate_paths={"TR-DEMO": cand_path},
            title="Poliittinen muisti",
        )

    # Matching embedded answer succeeds and shows reader title first.
    route_packet["answer_attachment"]["candidate_answer"] = dict(candidate)
    (route_dir / "TR-DEMO.json").write_text(
        json.dumps(route_packet, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    out = tmp_path / "ok_route"
    manifest = build_bootstrap_artifact(
        output_dir=out,
        matter_packets=[matter_path],
        question_packets=[],
        route_dirs=[route_dir],
        review_paths=[review_path],
        source_bank_paths=[questions, sources],
        reviewed_candidate_paths={"TR-DEMO": cand_path},
        title="Poliittinen muisti",
    )
    assert manifest["routes"][0]["reader_title"] == "Vaihtoehto hävisi"
    route_packet["answer_attachment"]["candidate_answer"]["map_findings"] = [
        {"text": "UNREVIEWED MAP CLAIM", "epistemic_status": "open"}
    ]
    (route_dir / "TR-DEMO.json").write_text(json.dumps(route_packet))
    with pytest.raises(BootstrapSiteError, match="embedded candidate_answer"):
        build_bootstrap_artifact(
            output_dir=tmp_path / "reject_map", matter_packets=[matter_path],
            question_packets=[], route_dirs=[route_dir], review_paths=[review_path],
            source_bank_paths=[questions, sources],
            reviewed_candidate_paths={"TR-DEMO": cand_path},
        )
    page = (out / "routes" / "TR_DEMO.html").read_text(encoding="utf-8")
    assert "Lähteet tarkistanut erillinen tekoälylukija" in page
    assert page.index("Vaihtoehto hävisi") < page.index("Tekniset tunnisteet")
    assert "ALTERED" not in page


def test_review_rejects_wrong_bank_hashes(tmp_path: Path) -> None:
    import hashlib
    import json

    from paa.bootstrap_site import BootstrapSiteError

    questions = tmp_path / "questions.json"
    sources = tmp_path / "sources.json"
    questions.write_text(json.dumps({"tasks": [{"task_id": "T-BANK", "question": "Kuka vastasi?", "matter_id": "KK 1/2023 vp"}]}), encoding="utf-8")
    text_hash = hashlib.sha256(b"ote").hexdigest()
    sources.write_text(json.dumps({"sources": [{"source_id": "S1", "text": "ote", "text_sha256": text_hash}], "task_sources": {"T-BANK": ["S1"]}}), encoding="utf-8")
    other_sources = tmp_path / "other_sources.json"
    other_sources.write_text(json.dumps({"sources": [{"source_id": "X"}]}), encoding="utf-8")
    q_sha = hashlib.sha256(questions.read_bytes()).hexdigest()
    s_sha = hashlib.sha256(sources.read_bytes()).hexdigest()

    good = {
        "schema_version": "paa.source_question.v5",
        "task_id": "T-BANK",
        "matter_id": "KK 1/2023 vp",
        "question": "Kuka vastasi?",
        "status": "OK",
        "admission": "CANDIDATE_NOT_REFERENCE_JUDGED",
        "answer": "Ministeri vastasi.",
        "claims": [
            {
                "text": "Väite",
                "evidence_bound": [
                    {
                        "source_id": "S1",
                        "quote": "ote",
                        "start": 0,
                        "end": 3,
                        "first_pointer": 0,
                        "last_pointer": 0,
                        "source_text_sha256": text_hash,
                    }
                ],
            }
        ],
        "unresolved": [],
        "next_evidence": [],
        "coverage": {"requested_source_ids": ["S1"], "pointer_domains": {"S1": {"source_text_sha256": text_hash}}},
        "binding_ok": True,
    }
    good_path = tmp_path / "good.json"
    good_bytes = (json.dumps(good, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    good_path.write_bytes(good_bytes)
    tampered = dict(good)
    tampered["claims"] = [
        {
            "text": "Väite",
            "evidence_bound": [
                {
                    "source_id": "S1",
                    "quote": "TAMPERED QUOTE",
                    "start": 0,
                    "end": 14,
                    "first_pointer": 0,
                    "last_pointer": 0,
                    "source_text_sha256": "a" * 64,
                }
            ],
        }
    ]
    bad_path = tmp_path / "bad_quote.json"
    bad_path.write_text(json.dumps(tampered, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    review = {
        "schema_version": "paa.source_answer_review.v1",
        "task_id": "T-BANK",
        "candidate_sha256": hashlib.sha256(good_bytes).hexdigest(),
        "questions_sha256": q_sha,
        "sources_sha256": s_sha,
        "source_versions": [{"source_id": "S1", "text_sha256": text_hash}],
        "reviewed_fields": ["answer", "claims", "unresolved", "next_evidence"],
        "reviewed_claim_indices": [0],
        "reviewer": {
            "kind": "INDEPENDENT_AI_SOURCE_READER",
            "id": "test",
            "human_certification": False,
        },
        "review_date": "2026-10-09",
        "decision": "SUPPORTED_WITH_STATED_LIMITS",
        "reader_title": "Ministerin vastaus",
        "reader_takeaway": "Rajattu.",
        "candidate_path": str(good_path),
    }
    review_path = tmp_path / "review.json"
    review_path.write_text(json.dumps(review), encoding="utf-8")
    matter_path = tmp_path / "matter.json"
    matter_path.write_text(
        json.dumps(
            {
                "schema_version": "paa.matter_ledger.v2",
                "query": {"mode": "EXACT_MATTER", "matter_id": "HE 1/2023 vp"},
                "matter": {"matter_id": "HE 1/2023 vp"},
                "actors": [],
                "interventions": [],
                "coverage": {},
                "missingness": [],
                "limits": [],
            }
        ),
        encoding="utf-8",
    )

    # A genuine reviewed candidate is the control for every independently broken receipt.
    import copy

    from paa.bootstrap_site import _validate_review_receipt

    assert _validate_review_receipt(
        review, candidate_path=good_path, questions_path=questions, sources_path=sources
    )["bound_candidate_sha256"] == review["candidate_sha256"]
    for field in ("questions_sha256", "sources_sha256", "source_versions", "reviewed_fields", "reviewed_claim_indices", "review_date"):
        invalid = copy.deepcopy(review)
        invalid.pop(field)
        with pytest.raises(BootstrapSiteError):
            _validate_review_receipt(
                invalid, candidate_path=good_path, questions_path=questions, sources_path=sources
            )
    invalid = copy.deepcopy(review)
    invalid["reviewer"] = {"human_certification": False}
    with pytest.raises(BootstrapSiteError, match="named independent AI reviewer"):
        _validate_review_receipt(invalid, candidate_path=good_path, questions_path=questions, sources_path=sources)
    invalid = copy.deepcopy(review)
    invalid["source_versions"][0]["text_sha256"] = "f" * 64
    with pytest.raises(BootstrapSiteError, match="complete task source universe"):
        _validate_review_receipt(invalid, candidate_path=good_path, questions_path=questions, sources_path=sources)
    changed_questions = tmp_path / "changed_questions.json"
    changed_doc = json.loads(questions.read_text())
    changed_doc["tasks"][0]["question"] = "A different question"
    changed_questions.write_text(json.dumps(changed_doc))
    invalid = copy.deepcopy(review)
    invalid["questions_sha256"] = hashlib.sha256(changed_questions.read_bytes()).hexdigest()
    with pytest.raises(BootstrapSiteError, match="task/question"):
        _validate_review_receipt(invalid, candidate_path=good_path, questions_path=changed_questions, sources_path=sources)

    with pytest.raises(BootstrapSiteError, match="matching question and source bank files"):
        build_bootstrap_artifact(
            output_dir=tmp_path / "wrong_bank",
            matter_packets=[matter_path],
            question_packets=[good_path],
            review_paths=[review_path],
            source_bank_paths=[questions, other_sources],
            reviewed_candidate_paths={"T-BANK": good_path},
            title="Poliittinen muisti",
        )

    with pytest.raises(BootstrapSiteError, match="analytical fields differ"):
        build_bootstrap_artifact(
            output_dir=tmp_path / "wrong_quote",
            matter_packets=[matter_path],
            question_packets=[bad_path],
            review_paths=[review_path],
            source_bank_paths=[questions, sources],
            reviewed_candidate_paths={"T-BANK": good_path},
            title="Poliittinen muisti",
        )

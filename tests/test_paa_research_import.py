"""External research stays candidate-level and traverses maintained consumers."""

import json

import pytest

from paa.cli import main
from paa.research_import import import_research_answer
from paa.route_packet import bind_answer_packet
from paa.source_question import SourceQuestionBank, SourceQuestionError


def inputs(tmp_path, text="Enemmistö\nEhdotus hylätään.\nVastalause\nEhdotus hylätään.\n", pointer=3):
    questions = tmp_path / "questions.json"
    sources = tmp_path / "sources.json"
    submission = tmp_path / "submission.json"
    output = tmp_path / "candidate.json"
    questions.write_text(json.dumps({"tasks": [{"task_id": "T", "matter_id": "HE 1/2025 vp",
        "question": "Kuka ehdotti mitä ja miten asia päätettiin?"}]}))
    sources.write_text(json.dumps({"sources": [{"source_id": "S", "text": text}],
        "task_sources": {"T": ["S"]}, "source_gaps": [{"task_id": "T", "reason": "Final decision not yet read"}]}))
    submission.write_text(json.dumps({"schema_version": "paa.research_submission.v1", "task_id": "T",
        "researcher_id": "native:researcher", "produced_at": "2026-10-11T03:00:00Z",
        "reading_scope": "Selected passages, no complete-channel absence claim",
        "result": {"answer": "Vastalause ehdotti hylkäämistä; lopullista päätöstä ei vielä selvitetty.",
            "claims": [{"text": "Vastalause ehdottaa hylkäämistä.", "epistemic_status": "DOCUMENTARY_OBSERVATION",
                "evidence": [{"source_id": "S", "first_pointer": pointer, "last_pointer": pointer}]}],
            "unresolved": ["Final disposition"], "next_evidence": ["Plenary decision"]}}))
    return questions, sources, submission, output


def run(paths):
    q, s, a, out = paths
    return import_research_answer(questions=q, sources=s, submission=a, output=out)


def test_import_binds_intended_repeated_span_and_preserves_parent_and_gap(tmp_path):
    paths = inputs(tmp_path)
    packet = run(paths)
    quote = packet["claims"][0]["evidence_bound"][0]
    assert quote["quote"] == "Ehdotus hylätään.\n"
    assert quote["start"] > len("Enemmistö\nEhdotus hylätään.\n")
    assert packet["question"] == "Kuka ehdotti mitä ja miten asia päätettiin?"
    assert packet["unresolved"] == ["Final disposition"]
    assert packet["source_gaps"][0]["reason"] == "Final decision not yet read"
    assert packet["admission"] == "CANDIDATE_NOT_REFERENCE_JUDGED"
    assert packet["model_id"] is None
    bank = SourceQuestionBank.load(paths[0], paths[1])
    result = bind_answer_packet({"task_id": "T", "question": packet["question"],
        "matter_id": packet["matter_id"]}, packet, bank)
    assert result["binding"]["state"] == "BOUND"


def test_long_source_uses_existing_consumer_pointer_domain(tmp_path):
    packet = run(inputs(tmp_path, text="A" * 30_000, pointer=49))
    assert packet["anchors"][0]["start"] == 29_400
    bank = SourceQuestionBank.load(tmp_path / "questions.json", tmp_path / "sources.json")
    result = bind_answer_packet({"task_id": "T", "question": packet["question"],
        "matter_id": packet["matter_id"]}, packet, bank)
    assert result["binding"]["state"] == "BOUND"


@pytest.mark.parametrize("defect", ["outside_pointer", "outside_source", "missing_task_mapping", "changed_source"])
def test_import_rejects_unbound_or_stale_evidence_without_partial_file(tmp_path, defect):
    paths = inputs(tmp_path)
    _q, s, a, out = paths
    submission = json.loads(a.read_text())
    bank = json.loads(s.read_text())
    if defect == "outside_pointer":
        submission["result"]["claims"][0]["evidence"][0]["last_pointer"] = 900
    elif defect == "outside_source":
        submission["result"]["claims"][0]["evidence"][0]["source_id"] = "OtherTask"
    elif defect == "missing_task_mapping":
        bank["task_sources"] = {}
        bank["source_gaps"] = []
    else:
        bank["sources"][0]["text_sha256"] = "0" * 64
    a.write_text(json.dumps(submission))
    s.write_text(json.dumps(bank))
    with pytest.raises(SourceQuestionError):
        run(paths)
    assert not out.exists()


def test_cli_reaches_candidate_and_does_not_overwrite_retained_result(tmp_path, capsys):
    q, s, a, out = inputs(tmp_path)
    args = ["research-import", "--questions", str(q), "--sources", str(s),
        "--submission", str(a), "--output", str(out)]
    assert main(args) == 0
    assert json.loads(capsys.readouterr().out)["binding_state"] == "OK"
    before = out.read_bytes()
    with pytest.raises(FileExistsError):
        main(args)
    assert out.read_bytes() == before

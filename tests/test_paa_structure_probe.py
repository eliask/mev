import asyncio
import hashlib
import json

import httpx
import pytest

from paa.case_site import render_case
from paa.inquiry_cases import digest, source_record
from paa.structure_probe import OPERATIONS, SYSTEM, normalize_answer, replay, request_input, run


def source(text="Lause. Lause."):
    value = source_record("public", text, url="https://example.org/report", title="Report",
                          locator="Report", raw_sha256=digest(text))
    value.update(raw_text=text, content_format="PLAIN", document_identifier="X 1/2024", record_id="test")
    return value


def response(quote):
    return {"status": "OK", "content": json.dumps({"operations": [
        {"id": operation, "answer": "A bounded answer.", "claims": [{"text": "A proposal.", "quotes": [quote]}],
         "unknowns": ["Adoption is unobserved."]} for operation in OPERATIONS]})}


def test_same_source_ablation_changes_only_structural_context():
    value = source()
    plain = json.loads(request_input(value, "plain"))
    structured = json.loads(request_input(value, "structured"))
    assert structured.pop("source_tree_context")["status"] == "UNSUPPORTED_FORMAT"
    assert structured == plain


def test_ambiguous_quotes_are_unbound_and_explicit_offset_reaches_renderer():
    value = source()
    packets, errors = normalize_answer(value, response({"quote": "Lause."}), case_id="test", title="Test")
    assert len(packets) == 3 and len(errors) == 3
    assert all(not packet["evidence"] for packet in packets)
    packets, errors = normalize_answer(value, response({"quote": "Lause.", "start": 7}), case_id="test", title="Test")
    assert not errors
    for packet in packets:
        assert packet["evidence"][0]["start"] == 7
        assert packet["claims"][0]["state"] == "CANDIDATE"
        assert "ei varmennettu päätelmä" in render_case(packet)


def test_missing_job_and_duplicate_operation_cannot_appear_as_completed_answers():
    packets, errors = normalize_answer(source(), {"status": "TRUNCATED"}, case_id="test", title="Test")
    assert not packets and errors == ["MODEL_TRUNCATED"]
    receipt = response({"quote": "Lause.", "start": 0})
    raw = json.loads(receipt["content"])
    raw["operations"][1]["id"] = raw["operations"][0]["id"]
    receipt["content"] = json.dumps(raw)
    packets, errors = normalize_answer(source(), receipt, case_id="test", title="Test")
    assert not packets and errors == ["INVALID_OPERATION_POPULATION"]


def test_prompt_permitted_string_quotes_bind_without_rewriting_source():
    value = source("A unique sentence.")
    packets, errors = normalize_answer(value, response("A unique sentence."), case_id="test", title="Test")
    assert not errors and len(packets) == 3
    assert packets[0]["evidence"][0]["quote"] == "A unique sentence."
    packets, errors = normalize_answer(value, response("A unique ..."), case_id="test", title="Test")
    assert len(errors) == 3 and not packets[0]["evidence"]


def test_wrong_model_offset_is_retained_after_independent_unique_text_binding():
    value = source("Prefix. A unique sentence.")
    packets, errors = normalize_answer(value, response({"quote": "A unique sentence.", "start": 0}),
                                       case_id="test", title="Test")
    assert len(errors) == 3 and all("MODEL_OFFSET_REJECTED" in error for error in errors)
    ref = packets[0]["evidence"][0]
    assert ref["start"] == 8 and ref["rejected_model_start"] == 0
    assert ref["binding_method"] == "UNIQUE_EXACT_TEXT_AFTER_REJECTED_MODEL_OFFSET"
    assert packets[0]["claims"][0]["state"] == "CANDIDATE"
    assert "ei varmennettu päätelmä" in render_case(packets[0])
    packets, errors = normalize_answer(source(), response({"quote": "Lause.", "start": 3}),
                                       case_id="test", title="Test")
    assert len(errors) == 3 and all("UNBOUND_QUOTATION" in error for error in errors)
    assert not packets[0]["evidence"]


def retained_inputs(tmp_path, providers, *, quotation_mode="quote_text", status="OK"):
    """Small source/receipt control; no retained experiment or accuracy claim."""
    from paa.llm_client import digest as request_digest
    from paa.opencode_client import MODEL, MUSE_MODEL

    value = source("A unique public source sentence.")
    sources = tmp_path / "sources.json"
    sources.write_text(json.dumps([value]))
    manifest = {"system": SYSTEM, "system_digest": request_digest(SYSTEM),
                "sources_digest": request_digest([value]), "output_tokens_by_record": {"test": 64},
                "muse_reasoning_effort": "low", "quotation_mode": quotation_mode}
    jobs = []
    for provider in providers:
        for mode in ("plain", "structured"):
            quotation = {"span_start": 0, "span_end": 0} if quotation_mode == "source_spans" else value["text"]
            job = response(quotation)
            configuration = {"max_output_tokens": 64, "temperature": 0, "store": False,
                             "reasoning": {"effort": "low"}}
            model_id = MUSE_MODEL if provider == "opencode_go_muse" else MODEL
            model = {"provider": "OPENCODE_GO", "model_id": model_id,
                     "endpoint": "https://opencode.ai/zen/go/v1/" + (
                         "responses" if provider == "opencode_go_muse" else "chat/completions")}
            wire_request = {"model": model_id, "instructions": SYSTEM,
                            "input": request_input(value, mode, quotation_mode), **configuration}
            job.update(provider=provider, mode=mode, source_id=value["source_id"],
                       request_id=provider + "-" + mode, source_text_sha256=value["text_sha256"],
                       input_sha256=request_digest(request_input(value, mode, quotation_mode)), status=status,
                       model=model, configuration=configuration, prompt_sha256=manifest["system_digest"],
                       wire_request_sha256=request_digest(wire_request), observed_reasoning_effort="low")
            jobs.append(job)
    retained = {"version": "fixture-control", "providers": providers, "manifest": manifest,
                "manifest_digest": request_digest(manifest), "jobs": jobs}
    receipts = tmp_path / "receipts.json"
    receipts.write_text(json.dumps(retained))
    return sources, receipts


def test_retained_population_reaches_candidate_browser_and_rejects_missing_or_relabeled_job(tmp_path):
    sources, receipts = retained_inputs(tmp_path, ["local", "opencode_go", "opencode_go_muse"])
    result = replay(sources, receipts, tmp_path / "replay")
    assert result["declared_jobs"] == result["completed_jobs"] == 6
    packets = [packet for row in result["rows"] for packet in row["packets"]]
    assert len(packets) == 18
    assert all(claim["state"] == "CANDIDATE" for packet in packets for claim in packet["claims"])
    assert all((tmp_path / "replay/browser" / (packet["case_id"] + ".html")).exists() for packet in packets)
    assert (tmp_path / "replay/browser/comparison.html").exists()
    bad = json.loads(receipts.read_text())
    bad["jobs"].pop()
    broken = tmp_path / "incomplete.json"
    broken.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="population"):
        replay(sources, broken, tmp_path / "broken")
    relabeled = json.loads(receipts.read_text())
    next(job for job in relabeled["jobs"] if job["provider"] == "opencode_go")["model"]["model_id"] = "other-model"
    broken.write_text(json.dumps(relabeled))
    with pytest.raises(ValueError, match="provider/model/endpoint"):
        replay(sources, broken, tmp_path / "relabeled")


@pytest.mark.asyncio
@pytest.mark.parametrize("interrupt", [False, True])
async def test_selected_provider_failure_or_cancellation_accounts_for_all_jobs(tmp_path, monkeypatch, interrupt):
    from paa.llm_client import digest as request_digest
    from paa.opencode_client import MUSE_MODEL, OpenCodeClient

    entered = asyncio.Event()
    key = tmp_path / "key.env"
    key.write_text("OPENCODE_API_KEY=test-secret-value-xyz")
    monkeypatch.setenv("OPENCODE_KEY_FILE", str(key))

    async def transport(request):
        if not interrupt:
            return httpx.Response(503)
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": MUSE_MODEL}]})
        entered.set()
        await asyncio.Event().wait()

    def client_factory(**kwargs):
        kwargs["key_file"] = key
        kwargs["transport"] = httpx.MockTransport(transport)
        return OpenCodeClient(**kwargs)

    monkeypatch.setattr("paa.structure_probe.OpenCodeClient", client_factory)
    values = [source("A unique source sentence.")]
    (tmp_path / "sources.json").write_text(json.dumps(values))
    (tmp_path / "references.json").write_text("[]")
    manifest = {"version": "test", "sources_file": "sources.json", "sources_digest": request_digest(values),
                "references_file": "references.json", "references_sha256": hashlib.sha256(b"[]").hexdigest(),
                "system": SYSTEM, "system_digest": request_digest(SYSTEM), "providers": ["opencode_go_muse"],
                "declared_jobs": 2, "output_tokens_by_record": {"test": 64}}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    task = asyncio.create_task(run(path, tmp_path / "run"))
    if interrupt:
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        await task
    r = json.loads((tmp_path / "run/results.json").read_text())
    assert r["declared_jobs"] == r["accounted_jobs"] == 2
    assert all(not row["packets"] for row in r["rows"])
    assert r["model_statuses"] == ({"INTERRUPTED": 2} if interrupt else {"FAILED": 2})


@pytest.mark.parametrize("status,packet_count", [("TRUNCATED", 0), ("OK", 6)])
def test_muse_replay_checks_configuration_and_candidate_boundaries(tmp_path, status, packet_count):
    sources, receipts = retained_inputs(tmp_path, ["opencode_go_muse"], status=status)
    result = replay(sources, receipts, tmp_path / "replay")
    assert result["declared_jobs"] == result["completed_jobs"] == 2
    packets = [packet for row in result["rows"] for packet in row["packets"]]
    assert len(packets) == packet_count
    assert all(packet["admission"] == "PROPOSED_NOT_ADMITTED" for packet in packets)
    assert all(claim["state"] == "CANDIDATE" for packet in packets for claim in packet["claims"])
    assert all(row["status"] == status for row in result["rows"])
    changed = json.loads(receipts.read_text())
    changed["jobs"][0]["configuration"]["max_output_tokens"] = 32
    path = tmp_path / "incorrect-configuration.json"
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="request configuration"):
        replay(sources, path, tmp_path / "bad-replay")


def test_pointer_quotes_select_repeated_passages_without_model_copied_text():
    value = source('Majority\nSame sentence.\nDissent\nSame sentence.\n')
    from paa.source_pointers import build_source_pointer_index

    index = build_source_pointer_index(value['source_id'], value['text'], granularity='sentence')
    selected = next(span for span in index.spans if span.start == value['text'].rfind('Same sentence.'))
    receipt = response({'span_start': selected.pointer_id, 'span_end': selected.pointer_id})
    receipt['quotation_mode'] = 'source_spans'
    packets, errors = normalize_answer(value, receipt, case_id='pointer', title='Pointer')
    assert not errors and len(packets) == 3
    for packet in packets:
        ref = packet['evidence'][0]
        assert ref['quote'] == 'Same sentence.\n'
        assert ref['start'] == selected.start and ref['binding_method'] == 'SOURCE_POINTER_RANGE'
        assert packet['claims'][0]['state'] == 'CANDIDATE'
        assert 'ei varmennettu päätelmä' in render_case(packet)
    payload = json.loads(request_input(value, 'plain', 'source_spans'))
    assert payload['text'] == index.indexed_text()
    assert payload['quotation_transport']['source_offsets'][selected.pointer_id] == [selected.start, selected.end]


@pytest.mark.parametrize('quote', [{'span_start': True, 'span_end': 0}, {'span_start': 0, 'span_end': 100},
                                  'A unique sentence.', {'quote': 'A unique sentence.'}])
def test_pointer_transport_rejects_invalid_ids_and_copied_quote_fallback(quote):
    receipt = response(quote)
    receipt['quotation_mode'] = 'source_spans'
    packets, errors = normalize_answer(source('A unique sentence.'), receipt, case_id='bad-pointer', title='Bad')
    assert len(errors) == 3 and all('INVALID_SOURCE_POINTER' in error for error in errors)
    assert all(not packet['evidence'] for packet in packets)
    assert all(packet['claims'][0]['state'] == 'CANDIDATE' for packet in packets)


def test_pointer_receipt_replay_reaches_browser_with_exact_backend_quotes(tmp_path):
    from paa.source_pointers import build_source_pointer_index

    sources, receipts = retained_inputs(tmp_path, ["opencode_go_muse"], quotation_mode="source_spans")
    result = replay(sources, receipts, tmp_path / "pointer-replay")
    assert result["declared_jobs"] == result["completed_jobs"] == 2
    assert all(row["status"] == "OK" and not row["errors"] for row in result["rows"])
    packets = [packet for row in result["rows"] for packet in row["packets"]]
    assert len(packets) == 6
    for packet in packets:
        assert packet["admission"] == "PROPOSED_NOT_ADMITTED"
        assert all(claim["state"] == "CANDIDATE" for claim in packet["claims"])
        value = packet["sources"][0]
        index = build_source_pointer_index(value["source_id"], value["text"], granularity="sentence")
        for ref in packet["evidence"]:
            resolved = index.resolve(span_start=ref["span_start"], span_end=ref["span_end"],
                                     source_text_sha256=value["text_sha256"])
            assert ref["quote"] == resolved.quote and ref["start"] == resolved.start and ref["end"] == resolved.end
            assert ref["binding_method"] == "SOURCE_POINTER_RANGE"
        assert (tmp_path / "pointer-replay/browser" / (packet["case_id"] + ".html")).exists()

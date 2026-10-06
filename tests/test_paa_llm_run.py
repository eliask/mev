import json

import jsonschema
import pytest

from paa.llm_overlay import ModelInterpretations
from paa.llm_run import compact_request, expand_compact
from paa.llm_semantics import normalize_multi_batch_output
from paa.semantics import analyze_text


def specimen():
    text = "Lupaan tehdä parhaani. En leikkaa koulutuksesta."
    props = analyze_text(text).propositions
    return {"document": {"document_id": "real-source", "source_id": "test-source", "text": text,
                        "language": "fi", "stated_earliest": "2023-03-01"},
            "units": [{"unit_id": f"real-source-p{i}", "text": p.text, "start": p.source_start, "end": p.source_end}
                      for i, p in enumerate(props, 1)]}, props


def test_compact_wire_preserves_exact_source_and_blocks_false_action_types():
    spec, _ = specimen()
    request = compact_request([spec], "extract_batch_multi_v2")
    output = {"rows": [["1.1", "PR", "S", False, None, None, None, None],
                       ["1.2", "NR", "S", True, "RT", "koulutuksesta", None, None]]}
    jsonschema.validate(output, request["schema"])
    result = normalize_multi_batch_output(expand_compact(output, [spec]), [spec])
    assert result.status == "VALID"
    assert [p.text for p in result.propositions] == [u["text"] for u in spec["units"]]
    bad = json.loads(json.dumps(output))
    bad["rows"][0][4] = "IA"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bad, request["schema"])
    bad["rows"][0][0] = "unknown"
    assert expand_compact(bad, [spec]) == {}


def test_model_overlay_refuses_stale_sources_and_preserves_abstention(tmp_path):
    import hashlib

    spec, props = specimen()
    (tmp_path / "documents").mkdir()
    (tmp_path / "manifest.json").write_text(json.dumps({"identity": {"prompt": "locked-v2", "source_snapshot_sha256": "snapshot"}}))
    receipt = {"status": "INVALID", "receipt_status": "INVALID_OUTPUT", "request_id": "receipt-1",
               "source_text_sha256": hashlib.sha256(spec["document"]["text"].encode()).hexdigest(), "units": []}
    path = tmp_path / "documents" / "real-source.json"
    path.write_text(json.dumps(receipt))
    overlay = ModelInterpretations(tmp_path)
    output = overlay.propositions(spec["document"], props)
    assert len(output) == len(props)
    assert all(p.semantic_type == "AMBIGUOUS" and p.validation_state == "PROPOSED" for p in output)
    assert all(p.action_kind is None for p in output)
    receipt["source_text_sha256"] = "changed"
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="source version changed"):
        overlay.propositions(spec["document"], props)


def test_model_overlay_revalidates_raw_anchors_instead_of_trusting_cached_labels(tmp_path):
    import hashlib

    spec, props = specimen()
    raw = {"rows": [["1.1", "PR", "S", False, None, None, None, None],
                    ["1.2", "NR", "S", True, "RT", "koulutuksesta", None, None]]}
    result = normalize_multi_batch_output(expand_compact(raw, [spec]), [spec]).documents[0].as_dict()
    result.update(receipt_status="OK", request_id="receipt-2",
                  source_text_sha256=hashlib.sha256(spec["document"]["text"].encode()).hexdigest())
    # Normalized checkpoint labels are convenient outputs, not trusted inputs.
    result["units"][0]["proposition"]["semantic_type"] = "PERSONAL_ACTION_COMMITMENT"
    result["units"][0]["proposition"]["action_kind"] = "INITIATIVE_AUTHORED"
    result["units"][1]["raw"][5] = "a target absent from the source"
    (tmp_path / "documents").mkdir()
    (tmp_path / "manifest.json").write_text(json.dumps({"identity": {"prompt": "extract_batch_multi_v2"}}))
    (tmp_path / "documents" / "real-source.json").write_text(json.dumps(result))
    output = ModelInterpretations(tmp_path).propositions(spec["document"], props)
    assert output[0].semantic_type == "PROCESS_COMMITMENT"
    assert output[0].action_kind is None
    assert output[1].semantic_type == "AMBIGUOUS"
    assert output[1].validation_state == "PROPOSED"


def test_validated_model_capability_survives_into_action_verification_plan(tmp_path):
    import hashlib

    from paa.opportunity import opportunity_requirements
    from paa.records import proposition_record

    text = "Teen lakialoitteen koulutuksen rahoituksen turvaamiseksi."
    canonical = analyze_text(text).propositions
    document = {"document_id": "initiative-source", "source_id": "test-source", "text": text,
                "language": "fi", "stated_earliest": "2023-03-01"}
    spec = {"document": document, "units": [{"unit_id": "initiative-source-p1", "text": text,
                                            "start": 0, "end": len(text)}]}
    raw = {"rows": [["1.1", "PA", "S", False, "IA", "koulutuksen rahoituksen turvaamiseksi", None, None]]}
    normalized = normalize_multi_batch_output(expand_compact(raw, [spec]), [spec]).documents[0].as_dict()
    assert normalized["status"] == "VALID"
    normalized.update(receipt_status="OK", request_id="source-receipt",
                      source_text_sha256=hashlib.sha256(text.encode()).hexdigest())
    (tmp_path / "documents").mkdir()
    (tmp_path / "manifest.json").write_text(json.dumps({"identity": {"prompt": "extract_batch_multi_v2"}}))
    (tmp_path / "documents/initiative-source.json").write_text(json.dumps(normalized))
    output = ModelInterpretations(tmp_path).propositions(document, canonical)
    record = proposition_record(output[0], document["document_id"], "source-evidence", ["source-actor"], "source-p1")
    plan = opportunity_requirements(record)
    assert plan.action_kind.value == "INITIATIVE_AUTHORED"
    assert plan.required_capability.value == "MP_INITIATE_BILL"
    assert output[0].validation_state == "PROPOSED"

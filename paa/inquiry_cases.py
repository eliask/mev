"""Version-bound inquiry packets for decisions, changes and evidence responses.

This module validates documentary backing, not the truth of a reviewer's
interpretation. Retrieval and detector outputs remain candidates. A reviewed
claim must identify its question, exact sources, quotations and reviewer.
Documentary sequence never supplies causal identification automatically.
"""

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path


class InquiryError(ValueError):
    pass


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def source_record(source_id: str, text: str, *, url: str, locator: str,
                  raw_sha256: str, title: str) -> dict:
    if not source_id or not text or not locator or not title:
        raise InquiryError("source identity, full text, locator and title are required")
    if not url.startswith(("https://", "http://")):
        raise InquiryError("source URL must be public HTTP(S)")
    if len(raw_sha256) != 64 or any(c not in "0123456789abcdef" for c in raw_sha256):
        raise InquiryError("raw source SHA-256 required")
    return {"source_id": source_id, "title": title, "text": text, "url": url,
            "locator": locator, "raw_sha256": raw_sha256, "text_sha256": digest(text)}


def anchor(source: Mapping, quote: str, *, start: int | None = None) -> dict:
    text = source["text"]
    if not isinstance(quote, str) or not quote:
        raise InquiryError("quotation must be a non-empty exact source span")
    if start is None:
        matches = []
        cursor = 0
        while True:
            match = text.find(quote, cursor)
            if match < 0:
                break
            matches.append(match)
            cursor = match + 1
        if not matches:
            raise InquiryError("quotation must be an exact source span")
        if len(matches) > 1:
            raise InquiryError("quotation occurs multiple times; explicit start is required")
        start = matches[0]
    elif not isinstance(start, int) or isinstance(start, bool):
        raise InquiryError("quotation start must be an integer offset")
    if start < 0 or start > len(text) or start + len(quote) > len(text):
        raise InquiryError("quotation start is outside the source text")
    if text[start:start + len(quote)] != quote:
        raise InquiryError("quotation must be an exact source span")
    identity = f"{source['source_id']}:{source['text_sha256']}:{start}:{len(quote)}"
    return {"evidence_id": "inquiry-evidence:" + digest(identity)[:24],
            "source_id": source["source_id"], "text_sha256": source["text_sha256"],
            "raw_sha256": source["raw_sha256"], "start": start, "end": start + len(quote),
            "quote": quote, "locator": source["locator"], "url": source["url"]}


def validate_case(packet: Mapping) -> bool:
    question = packet.get("question", {})
    if not all(question.get(k) for k in ("text", "scope", "answer_standard")):
        raise InquiryError("question, scope and sufficient-answer standard are required")
    sources = {s["source_id"]: s for s in packet.get("sources", [])}
    if len(sources) != len(packet.get("sources", [])):
        raise InquiryError("duplicate source identity")
    refs = {r["evidence_id"]: r for r in packet.get("evidence", [])}
    if len(refs) != len(packet.get("evidence", [])):
        raise InquiryError("duplicate evidence identity")
    for source in sources.values():
        if digest(source["text"]) != source["text_sha256"]:
            raise InquiryError("source text changed")
        if "raw_text" in source and digest(source["raw_text"]) != source["raw_sha256"]:
            raise InquiryError("raw source changed")
    for ref in refs.values():
        source = sources.get(ref["source_id"])
        if source is None or any(ref.get(k) != source.get(k) for k in ("raw_sha256", "text_sha256", "locator", "url")):
            raise InquiryError("evidence refers to a changed source version")
        if source["text"][ref["start"]:ref["end"]] != ref["quote"] or not ref["quote"]:
            raise InquiryError("evidence quotation changed")
        expected = anchor(source, ref["quote"], start=ref["start"])
        if expected["evidence_id"] != ref["evidence_id"]:
            raise InquiryError("evidence identity does not match source span")
    for claim in packet.get("claims", []):
        if claim.get("state") not in {"CANDIDATE", "SOURCE_REVIEWED", "UNRESOLVED"}:
            raise InquiryError("unknown claim admission state")
        if not claim.get("text") or not claim.get("dimension"):
            raise InquiryError("claim text and dimension required")
        ids = claim.get("evidence_ids", [])
        if not set(ids) <= refs.keys():
            raise InquiryError("claim contains dangling evidence")
        if claim["state"] == "SOURCE_REVIEWED":
            review = claim.get("review", {})
            if not ids or not all(review.get(k) for k in ("reviewer", "method", "rationale", "source_versions")):
                raise InquiryError("reviewed claim lacks evidence or review provenance")
            used = {refs[key]["source_id"] for key in ids}
            if review["source_versions"] != {key: sources[key]["text_sha256"] for key in used}:
                raise InquiryError("review is stale or bound to different sources")
            if review["method"] in {"KEYWORD_OVERLAP", "LOCAL_LLM_PROPOSAL", "DETECTOR_OUTPUT"}:
                raise InquiryError("candidate method cannot admit a conclusion")
        if claim.get("causal_claim"):
            raise InquiryError("documentary inquiry does not identify causal effects")
    for unknown in packet.get("unknowns", []):
        if not all(unknown.get(k) for k in ("question", "missing_evidence")):
            raise InquiryError("unknown must identify the question and missing evidence")
    return True


def compile_case(*, question: dict, sources: list[dict], evidence: list[dict],
                 claims: list[dict], unknowns: list[dict], title: str,
                 selection_basis: str, case_id: str | None = None) -> dict:
    identity = json.dumps({"question": question, "sources": [(s["source_id"], s["text_sha256"])
                           for s in sources]}, sort_keys=True, ensure_ascii=False)
    packet = {"schema_version": "paa.inquiry.v1", "case_id": case_id or "inquiry-" + digest(identity)[:24],
              "title": title, "question": question, "sources": sources, "evidence": evidence,
              "claims": claims, "unknowns": unknowns, "selection_basis": selection_basis,
              "limits": ["Source-version and quotation validation does not prove semantic judgment.",
                         "Documentary relationships do not establish causal contribution.",
                         "Selected cases do not estimate population failure rates."]}
    validate_case(packet)
    return packet


def load_reviewed_cases(source_path: Path, review_path: Path) -> list[dict]:
    """Compile declared reviews against immutable frozen institutional inputs."""
    episodes = {}
    for line in source_path.read_text().split("\n"):
        if not line.strip():
            continue
        row = json.loads(line)
        episode_id = row["episode_id"]
        if episode_id in episodes:
            raise InquiryError("duplicate source episode identity")
        episodes[episode_id] = row
    packets = []
    review_ids = set()
    for spec in json.loads(review_path.read_text()):
        if spec["case_id"] in review_ids:
            raise InquiryError("duplicate reviewed case identity")
        review_ids.add(spec["case_id"])
        episode = episodes[spec["source_episode_id"]]
        records = {s["record_id"]: s for s in episode["sources"]}
        if len(records) != len(episode["sources"]):
            raise InquiryError("duplicate source record identity in episode")
        sources = {}
        for record_id, expected in spec["source_versions"].items():
            raw = records[record_id]
            if digest(raw["text"]) != expected or raw["text_sha256"] != expected:
                raise InquiryError("source review stale; re-review required")
            if digest(raw["raw_text"]) != raw["raw_sha256"]:
                raise InquiryError("frozen raw source changed")
            source = source_record(raw["source_id"], raw["text"], url=raw["source_url"],
                                   locator=raw["record_locator"], raw_sha256=raw["raw_sha256"],
                                   title=raw["title"])
            # Keep exact retrieved bytes/text independently of the display view.
            source["raw_text"] = raw["raw_text"]
            source["source_kind"] = raw["source_kind"]
            source["record_id"] = raw["record_id"]
            source["document_identifier"] = raw.get("document_identifier")
            source["publisher"] = raw.get("publisher")
            for field in ("captured_at", "content_format", "coverage_basis",
                          "normalization_version", "response_raw_sha256"):
                if field in raw:
                    source[field] = raw[field]
            sources[record_id] = source
        refs = {}
        for row in spec["quotes"]:
            key = row["key"]
            if key in refs:
                raise InquiryError("duplicate quotation key in source review")
            kwargs = {"start": row["start"]} if "start" in row else {}
            refs[key] = anchor(sources[row["record_id"]], row["quote"], **kwargs)
        claims = []
        for row in spec["claims"]:
            evidence = [refs[key] for key in row["quote_keys"]]
            used = {ref["source_id"] for ref in evidence}
            versions = {source["source_id"]: source["text_sha256"] for source in sources.values() if source["source_id"] in used}
            claims.append({"dimension": row["dimension"], "text": row["text"], "state": "SOURCE_REVIEWED",
                           "evidence_ids": [ref["evidence_id"] for ref in evidence], "review": {
                           "reviewer": spec["reviewer"], "method": spec["method"], "rationale": row["rationale"],
                           "source_versions": versions}})
        packet = compile_case(question=spec["question"], sources=list(sources.values()),
                              evidence=list(refs.values()), claims=claims, unknowns=spec["unknowns"],
                              title=spec["title"], selection_basis=spec["selection_basis"], case_id=spec["case_id"])
        packet["source_episode_id"] = spec["source_episode_id"]
        packet["episode_id"] = spec["source_episode_id"]
        if spec.get("question_id"):
            packet["question_id"] = spec["question_id"]
        if spec.get("legal_capture"):
            from paa.legal_inquiry import augment_inquiry_with_legal_receipt
            from paa.legal_state import load_lawvm_capture

            capture_path = review_path.parent / spec["legal_capture"]
            if capture_path.resolve().parent != review_path.parent.resolve():
                raise InquiryError("legal capture must be a packaged review fixture")
            if digest(capture_path.read_text()) != spec.get("legal_capture_sha256"):
                raise InquiryError("legal capture changed; re-review required")
            packet = augment_inquiry_with_legal_receipt(packet, load_lawvm_capture(capture_path))
            for row in spec.get("legal_source_claims", []):
                candidates = [ref for ref in packet["evidence"] if ref.get("quote_role") == row["source_role"]]
                if len(candidates) != 1:
                    raise InquiryError("legal source review requires one declared source view")
                ref = candidates[0]
                if any(quote not in ref["quote"] for quote in row["required_quotes"]):
                    raise InquiryError("legal source review quote changed")
                packet["claims"].append({"dimension": row["dimension"], "text": row["text"],
                    "state": "SOURCE_REVIEWED", "evidence_ids": [ref["evidence_id"]], "review": {
                    "reviewer": spec["reviewer"], "method": spec["method"], "rationale": row["rationale"],
                    "source_versions": {ref["source_id"]: ref["text_sha256"]}}})
            validate_case(packet)
        packets.append(packet)
    return packets

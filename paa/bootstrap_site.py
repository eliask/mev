"""Selectable static reader for matter packets + source-question candidates.

Public surface only: matter/person selection, substantive alternatives, roles,
stages, optional question journeys and institutional results with nearby
excerpts. Internal evaluation prose, keys, run logs and filesystem paths stay
out of the pages. Analytic numbers and conclusions come from source-bound
packets, never renderer constants. Prose source-answer review never admits a
journey graph; projection review is a separate receipt.
"""

import hashlib
import html
import json
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final
from urllib.parse import urlsplit

from paa.case_site import _document
from paa.matter_site import (
    compact_actor_summaries,
    matter_choice_label,
    resolve_matter_subject_title,
    write_matter_site,
)

SCHEMA_VERSION = "paa.bootstrap_site.v6"
DEFAULT_TITLE: Final = "Poliittinen muisti"
DEFAULT_PUBLISHER: Final = "Mekanismirealismi / mev-public"
DEFAULT_CORRECTION_URL: Final = "https://github.com/eliask/mev/issues"
DEFAULT_REPO_URL: Final = "https://github.com/eliask/mev"

_STATUS_FI = {
    "OK": "Valmis ehdokasvastaus",
    "FAILED": "Käsittely epäonnistui",
    "INVALID_OUTPUT": "Mallin vastaus ei ollut kelvollinen",
    "SOURCE_GAP": "Lähdepuute",
    "INDEX_FAILED": "Indeksointi epäonnistui",
    "PARTIAL": "Osittain sidottu",
    "EMPTY": "Ei sidottua näyttöä",
    "CANDIDATE_NOT_REFERENCE_JUDGED": (
        "Ehdokasvastaus — ei vielä reference-arvioitu (ei tarkoita kumottua)"
    ),
}
_REVIEW_SCHEMA: Final = "paa.source_answer_review.v1"
_ROUTE_PACKET_SCHEMAS: Final = frozenset({"paa.route_packet.v2", "paa.route_packet.v1"})

_PATH_LEAK_MARKERS: Final = (
    ".tmp/",
    "/home/",
    "/mnt/",
    "/Users/",
    "cache_local",
    "cache_muse",
    ".jsonl",
    "OPENCODE",
    "api_key",
    "env.opencode",
    "file:",
)


class BootstrapSiteError(ValueError):
    """Raised when a public artifact cannot be published safely."""


def _e(value: object) -> str:
    return html.escape(str(value if value is not None else ""), quote=True)


def _slug(value: str) -> str:
    text = re.sub(r"[^\w]+", "_", value.strip(), flags=re.UNICODE)
    return text.strip("_") or "item"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _source_link(url: object, label: str) -> str:
    text = str(url or "")
    if urlsplit(text).scheme in {"https", "http"}:
        return f'<a href="{_e(text)}">{_e(label)}</a>'
    return _e(label)


def _error_needs_redaction(error: object) -> bool:
    text = str(error or "")
    lowered = text.lower()
    return any(marker.lower() in lowered for marker in _PATH_LEAK_MARKERS) or bool(
        re.search(r"(?:^|[\s\"'])/(?:var|tmp|opt|usr)/", text)
    )


def _public_error_message(error: object) -> str | None:
    """Plain public processing note; private path/detail stays out of the artifact."""

    if error is None or error == "":
        return None
    if _error_needs_redaction(error):
        return "Käsittelyvirhe (yksityiskohta säilytetty sisäisessä kuittauksessa)."
    text = str(error).strip()
    return text[:400] if text else None


def _bound_evidence_block(row: Mapping[str, Any], sources: Mapping[str, Mapping[str, Any]] | None) -> str:
    source_id = str(row.get("source_id") or "")
    bank_meta = (sources or {}).get(source_id) or {}
    row_meta = {
        key: row[key]
        for key in ("title", "url", "source_locator", "identifier", "source_text_sha256")
        if key in row and row[key] is not None
    }
    meta = {**bank_meta, **row_meta}
    title = meta.get("title") or source_id
    locator = meta.get("source_locator") or meta.get("identifier") or ""
    url = meta.get("url")
    quote = str(row.get("quote") or "")
    pointers = f"{row.get('first_pointer')}–{row.get('last_pointer')}"
    offsets = f"{row.get('start')}–{row.get('end')}"
    digest = meta.get("source_text_sha256") or row.get("source_text_sha256") or ""
    projected = "[…]" in quote or bool(row.get("projection"))
    nearby = (
        f"<p>{_source_link(url, str(title))}"
        + (f" · {_e(locator)}" if locator else "")
        + "</p>"
        + "<details class='evidence-meta'><summary>Lähdesidonta</summary>"
        "<p class='muted'>Hankittu lähdesidonnainen ote"
        + (" (projektio/katkelma, ei koko alkuperäinen artikkeli tai taulukko)" if projected else "")
        + f". Merkit {_e(offsets)}"
        + (f"; osoittimet {_e(pointers)}" if row.get("first_pointer") is not None else "")
        + (f"; tiiviste {_e(digest[:12])}…" if digest else "")
        + f"; tunniste {_e(source_id)}.</p></details>"
    )
    if not quote:
        return nearby + "<p class='muted'>Ei sidottua lähdekatkelmaa.</p>"
    return (
        nearby
        + f"<details><summary>Sidottu ote ({_e(len(quote))} merkkiä) — avaa tarvittaessa</summary>"
        + f"<blockquote>{_e(quote)}</blockquote></details>"
    )


def _claim_status_label(claim: Mapping[str, Any], *, reviewed: bool = False) -> str:
    """Human-readable claim scope from epistemic_status; no auto-admission."""

    ep = str(claim.get("epistemic_status") or "").casefold()
    if "minister" in ep or "reported_via_minister" in ep:
        return (
            "Ministerin vastausväite (ministerin kertomus lähteessä; ei itsenäinen "
            "oikeus- tai yritystulosväite)"
        )
    if "question_wording" in ep or "questioner" in ep:
        return "Kysymyksen sanamuoto / esittäjä (ei vastauksen tosiseikka)"
    if "not_established" in ep:
        return "Lähteestä ei seuraa (raja)"
    if reviewed:
        return "Lähteisiin sidottu väite (tekoälyn lähdearvioitu tapaus)"
    return "Lähdekatkelmaan sidottu ehdokasväite — ei vielä reference-arvioitu"


def _question_article(
    packet: Mapping[str, Any],
    *,
    sources: Mapping[str, Mapping[str, Any]] | None = None,
    packet_json_href: str | None = None,
    review: Mapping[str, Any] | None = None,
) -> str:
    status = packet.get("status") or "UNKNOWN"
    binding = packet.get("binding_state") or ("OK" if packet.get("binding_ok") else "EMPTY")
    claims = packet.get("claims") or []
    reviewed = isinstance(review, Mapping) and bool(
        review.get("reader_title") or review.get("decision")
    )
    claim_html = []
    for claim in claims:
        unbound = bool(claim.get("unbound"))
        bound_rows = [] if unbound else [
            row for row in (claim.get("evidence_bound") or []) if row.get("quote") is not None
        ]
        if unbound or not bound_rows:
            evidence_html = (
                "<p class='muted'><strong>Ei sidottua lähdekatkelmaa.</strong> "
                "Väite on ehdokastekstiä ilman kelvollista osoitinta.</p>"
            )
        else:
            evidence_html = (
                "<details><summary>Lähdeotteet</summary>"
                + "".join(_bound_evidence_block(row, sources) for row in bound_rows)
                + "</details>"
            )
        claim_html.append(
            "<article class='claim'>"
            f"<p>{_e(claim.get('text'))}</p>"
            f"<p class='muted'>{_e(_claim_status_label(claim, reviewed=reviewed))}</p>"
            + evidence_html
            + "</article>"
        )
    unresolved = "".join(f"<li>{_e(item)}</li>" for item in packet.get("unresolved") or [])
    next_ev = "".join(f"<li>{_e(item)}</li>" for item in packet.get("next_evidence") or [])
    coverage = packet.get("coverage") or {}
    requested = len(coverage.get("requested_source_ids") or [])
    indexed = len(coverage.get("indexed_source_ids") or [])
    # Never fall back from explicit zero/missing fully-read to indexed.
    if "fully_read_source_ids" in coverage:
        fully_read = len(coverage.get("fully_read_source_ids") or [])
    elif "read_ok_source_ids" in coverage:
        fully_read = len(coverage.get("read_ok_source_ids") or [])
    else:
        fully_read = 0
    read_completeness = (
        packet.get("read_completeness")
        or coverage.get("read_completeness")
        or "TUNTEMATON"
    )
    coverage_note = (
        f"<p class='muted'>Lähdekattavuus: {_e(fully_read)} täysin luettu / {_e(indexed)} indeksoitu / "
        f"{_e(requested)} pyydetty. Luennan täydellisyys: {_e(read_completeness)}. "
        f"Osoittimien sitominen (ei semanttinen hyväksyntä): {_e(_STATUS_FI.get(str(binding), binding))}.</p>"
    )
    public_error = _public_error_message(packet.get("error"))
    if public_error:
        coverage_note += f"<p class='muted'>Virhe: {_e(public_error)}</p>"
    gaps = "".join(
        f"<li>{_e(item.get('status') or item.get('reason') or item)}</li>"
        for item in packet.get("source_gaps") or []
    )
    ledger = packet.get("ledger_projection")
    ledger_html = ""
    if isinstance(ledger, Mapping) and ledger.get("formal_and_signed_interventions"):
        rows = "".join(
            "<li>"
            + _e(item.get("action_date"))
            + " · "
            + _e(item.get("source_display_name") or item.get("actor_name"))
            + " · "
            + _e(item.get("role_label"))
            + " · "
            + _e((item.get("institutional_result") or {}).get("state"))
            + " · "
            + _e(item.get("title"))
            + "</li>"
            for item in ledger.get("formal_and_signed_interventions") or []
        )
        ledger_html = (
            "<details><summary>Erillinen rekisteriprojektio (ei samaa lähdeaineistoa)</summary>"
            "<p class='muted'>Nämä kirjaukset tulevat paikallisesta rekisteristä. "
            "Niitä ei ole sidottu tämän kysymyksen lähdepaketin osoittimiin.</p>"
            f"<ul>{rows}</ul></details>"
        )
    json_link = (
        f"<p><a download href='{_e(packet_json_href)}'>Lataa ehdokaspaketti (JSON)</a></p>"
        if packet_json_href
        else ""
    )
    review_html = (
        _review_presentation_html(review, include_title=False) if reviewed else ""
    )
    claims_block = ""
    if claim_html:
        if reviewed:
            claims_block = "<h3>Väitteet</h3>" + "".join(claim_html)
        else:
            claims_block = (
                f"<details><summary>Väitteet ja sidotut otteet</summary>"
                f"{''.join(claim_html)}</details>"
            )
    return (
        f"<section class='question' data-task='{_e(packet.get('task_id'))}' "
        f"data-matter='{_e(packet.get('matter_id'))}'>"
        + (
            review_html
            if reviewed
            else (
                f"<h3>{_e(packet.get('question') or packet.get('task_id'))}</h3>"
                f"<p>{_e(_STATUS_FI.get(str(status), status))} · "
                f"{_e(_STATUS_FI.get(str(packet.get('admission')), 'CANDIDATE_NOT_REFERENCE_JUDGED'))}</p>"
            )
        )
        + (
            (
                f"<details class='full-candidate'><summary>Koko ehdokasvastaus</summary>"
                f"<p>{_e(packet.get('answer'))}</p></details>"
                if packet.get("answer")
                else ""
            )
            if reviewed
            else (
                f"<details class='full-candidate'><summary>Ehdokasvastaus (raaka)</summary>"
                f"<p>{_e(packet.get('answer'))}</p></details>"
                if packet.get("answer")
                else ""
            )
        )
        + claims_block
        + f"<details><summary>Tekninen kattavuus ja sidonta</summary>{coverage_note}</details>"
        + json_link
        + (f"<details><summary>Avoimet kohdat</summary><ul>{unresolved}</ul></details>" if unresolved else "")
        + (f"<details><summary>Seuraava tarvittava näyttö</summary><ul>{next_ev}</ul></details>" if next_ev else "")
        + (f"<details><summary>Ilmoitetut lähdepuutteet</summary><ul>{gaps}</ul></details>" if gaps else "")
        + ledger_html
        + f"<details><summary>Konetunnisteet</summary><p class='muted'>{_e(packet.get('task_id'))}</p></details>"
        + "</section>"
    )


def _file_sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _fi_review_date(value: object) -> str:
    text = str(value or "").strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        year, month, day = text.split("-")
        return f"{int(day)}.{int(month)}.{year}"
    return text or "—"


def _review_presentation_html(
    review: Mapping[str, Any],
    *,
    include_title: bool = True,
) -> str:
    limits = "".join(f"<li>{_e(item)}</li>" for item in (review.get("limits") or [])[:12])
    return (
        "<aside class='ai-source-review'>"
        + (
            (
                f"<h2>{_e(review.get('reader_title'))}</h2>"
                if review.get("reader_title")
                else "<h2>Lähdearvioitu vastaus</h2>"
            )
            if include_title
            else ""
        )
        + (
            f"<p class='takeaway'>{_e(review.get('reader_takeaway'))}</p>"
            if review.get("reader_takeaway")
            else ""
        )
        + "<p class='muted'>Lähteet tarkistanut erillinen tekoälylukija · "
        f"{_e(_fi_review_date(review.get('review_date')))}.</p>"
        + "<details><summary>Mitä «lähdearvioitu» tarkoittaa</summary>"
        "<p class='muted'>Erillinen tekoälylukija on tarkistanut tämän tapauksen "
        "suhteessa ilmoitettuihin lähteisiin. Se ei ole ihmisen sertifiointi, "
        "ei ADMITTED-väestöpäätös eikä yleistä tarkkuusväitettä. Raaka ehdokaspaketti "
        "säilyttää oman admission-tilansa. «Lähdearvioitu» ei ole sama kuin "
        "ihmisen reference-arvio tai väestötason hyväksyntä.</p>"
        + (f"<ul>{limits}</ul>" if limits else "")
        + "</details></aside>"
    )


def _find_file_by_sha(paths: Sequence[Path], expected_sha: str) -> Path | None:
    for path in paths:
        if path is None:
            continue
        candidate = Path(path)
        if candidate.is_file() and _file_sha(candidate) == expected_sha:
            return candidate
    return None


def _resolve_receipt_sidecar(review: Mapping[str, Any], review_path: Path, *keys: str) -> Path | None:
    for key in keys:
        name = review.get(key)
        if not name:
            continue
        path = Path(str(name))
        if path.is_file():
            return path
        sibling = review_path.parent / str(name)
        if sibling.is_file():
            return sibling
    return None


def _validate_review_receipt(
    review: Mapping[str, Any],
    *,
    candidate_path: Path,
    questions_path: Path | None = None,
    sources_path: Path | None = None,
    expected_task_id: str | None = None,
    bank_search_paths: Sequence[Path] | None = None,
) -> dict[str, Any]:
    """Bind a review receipt to exact candidate + declared bank bytes."""

    if str(review.get("schema_version") or "") != _REVIEW_SCHEMA:
        raise BootstrapSiteError(
            f"unsupported review schema {review.get('schema_version')!r}; "
            f"expected {_REVIEW_SCHEMA}"
        )
    reviewer = review.get("reviewer")
    if (
        not isinstance(reviewer, Mapping)
        or not str(reviewer.get("id") or "").strip()
        or reviewer.get("kind") != "INDEPENDENT_AI_SOURCE_READER"
        or reviewer.get("human_certification") is not False
    ):
        raise BootstrapSiteError("review requires named independent AI reviewer and human_certification=false")
    from datetime import date

    try:
        date.fromisoformat(review["review_date"])
    except (KeyError, TypeError, ValueError) as error:
        raise BootstrapSiteError("review requires an ISO review_date") from error
    if review.get("decision") != "SUPPORTED_WITH_STATED_LIMITS":
        raise BootstrapSiteError("unsupported source review decision")
    candidate = _load_json(candidate_path)
    task_id = str(review.get("task_id") or "")
    if not task_id or task_id != str(candidate.get("task_id") or ""):
        raise BootstrapSiteError("review task_id does not match candidate")
    if expected_task_id is not None and task_id != expected_task_id:
        raise BootstrapSiteError("review task_id does not match displayed task")
    candidate_sha = _file_sha(candidate_path)
    if review.get("candidate_sha256") != candidate_sha:
        raise BootstrapSiteError("review candidate_sha256 mismatch")
    search = [Path(p) for p in (bank_search_paths or []) if p]
    required_hashes = ("questions_sha256", "sources_sha256")
    for key in required_hashes:
        if not re.fullmatch(r"[0-9a-f]{64}", str(review.get(key) or "")):
            raise BootstrapSiteError(f"review requires {key}")
    q_expected, s_expected = (str(review[key]) for key in required_hashes)
    questions_path = questions_path or _find_file_by_sha(search, q_expected)
    sources_path = sources_path or _find_file_by_sha(search, s_expected)
    if questions_path is None or sources_path is None:
        raise BootstrapSiteError("review requires matching question and source bank files")
    if _file_sha(questions_path) != q_expected:
        raise BootstrapSiteError("review questions_sha256 mismatch")
    if _file_sha(sources_path) != s_expected:
        raise BootstrapSiteError("review sources_sha256 mismatch")
    from paa.route_packet import bind_answer_packet
    from paa.source_question import SourceQuestionBank, build_task_indexes

    bank = SourceQuestionBank.load(questions_path, sources_path)
    task = next((row for row in bank.tasks if str(row.get("task_id") or "") == task_id), None)
    if task is None or str(task.get("question") or "").strip() != str(candidate.get("question") or "").strip():
        raise BootstrapSiteError("reviewed candidate does not match question bank task/question")
    task_matter = str(task.get("matter_id") or task.get("matter") or "")
    if task_matter and task_matter != str(candidate.get("matter_id") or ""):
        raise BootstrapSiteError("reviewed candidate does not match question bank matter")
    indexes = build_task_indexes(bank, task_id).indexes
    expected_versions = {source_id: index.source_text_sha256 for source_id, index in indexes.items()}
    versions = review.get("source_versions")
    if not isinstance(versions, list) or not versions:
        raise BootstrapSiteError("review requires complete source_versions")
    declared_versions = {}
    for row in versions:
        if not isinstance(row, Mapping) or not row.get("source_id") or not row.get("text_sha256"):
            raise BootstrapSiteError("invalid review source_versions record")
        source_id = str(row["source_id"])
        if source_id in declared_versions:
            raise BootstrapSiteError("duplicate review source_versions source_id")
        declared_versions[source_id] = str(row["text_sha256"])
    if declared_versions != expected_versions:
        raise BootstrapSiteError("review source_versions do not match complete task source universe")
    fields = review.get("reviewed_fields")
    required_fields = {"answer", "claims", "unresolved", "next_evidence"}
    if (not isinstance(fields, list) or any(type(field) is not str for field in fields)
        or not required_fields.issubset(fields)):
        raise BootstrapSiteError("review requires complete analytical reviewed_fields")
    if candidate.get("map_findings") and "map_findings" not in fields:
        raise BootstrapSiteError("review does not cover map_findings")
    indices = review.get("reviewed_claim_indices")
    if (not isinstance(indices, list) or any(type(i) is not int for i in indices)
        or indices != list(range(len(candidate.get("claims") or [])))):
        raise BootstrapSiteError("review requires exact reviewed_claim_indices")
    stub = {
        "task_id": task_id,
        "matter_id": candidate.get("matter_id"),
        "question": task["question"],
        "bank": {"task_source_ids": list(bank.task_sources[task_id])},
    }
    binding = bind_answer_packet(stub, candidate, bank)["binding"]
    if binding["state"] != "BOUND":
        raise BootstrapSiteError("reviewed candidate evidence fails source binding: " + "; ".join(binding["errors"]))
    out = dict(review)
    out["bound_candidate_sha256"] = candidate_sha
    out["bound_candidate_path"] = candidate_path.name
    if questions_path is not None:
        out["bound_questions_path"] = str(questions_path)
    if sources_path is not None:
        out["bound_sources_path"] = str(sources_path)
    # Never promote packet admission from a review decision.
    out["admission_unchanged"] = True
    return out


def _analytical_answer_fields(packet: Mapping[str, Any]) -> dict[str, Any]:
    """Comparable analytical fields for review↔display equality."""

    claims_out: list[dict[str, Any]] = []
    for claim in packet.get("claims") or []:
        if not isinstance(claim, Mapping):
            continue
        evidence = []
        for row in claim.get("evidence_bound") or []:
            if not isinstance(row, Mapping):
                continue
            evidence.append(
                {
                    key: row.get(key)
                    for key in (
                        "source_id",
                        "quote",
                        "start",
                        "end",
                        "first_pointer",
                        "last_pointer",
                        "source_text_sha256",
                    )
                }
            )
        claims_out.append(
            {
                "text": claim.get("text"),
                "unbound": bool(claim.get("unbound")),
                "epistemic_status": claim.get("epistemic_status"),
                "evidence_bound": evidence,
            }
        )
    coverage = packet.get("coverage") if isinstance(packet.get("coverage"), Mapping) else {}
    return {
        "task_id": packet.get("task_id"),
        "matter_id": packet.get("matter_id"),
        "question": packet.get("question"),
        "answer": packet.get("answer"),
        "claims": claims_out,
        "map_findings": list(packet.get("map_findings") or []),
        "unresolved": list(packet.get("unresolved") or []),
        "next_evidence": list(packet.get("next_evidence") or []),
        "requested_source_ids": list(coverage.get("requested_source_ids") or []),
    }


def _route_candidate_analytical_fields(candidate_answer: Mapping[str, Any]) -> dict[str, Any]:
    return _analytical_answer_fields(
        {
            "task_id": candidate_answer.get("task_id"),
            "matter_id": candidate_answer.get("matter_id"),
            "question": candidate_answer.get("question"),
            "answer": candidate_answer.get("answer"),
            "claims": candidate_answer.get("claims"),
            "map_findings": candidate_answer.get("map_findings"),
            "unresolved": candidate_answer.get("unresolved"),
            "next_evidence": candidate_answer.get("next_evidence"),
            "coverage": candidate_answer.get("coverage") or {
                "requested_source_ids": (candidate_answer.get("coverage") or {}).get(
                    "requested_source_ids"
                )
            },
        }
    )


def _resolve_reviewed_candidate_file(
    review: Mapping[str, Any],
    *,
    review_path: Path,
    displayed_path: Path | None,
    reviewed_candidate_paths: Mapping[str, Path] | None,
) -> Path:
    """Return the exact reviewed candidate file whose bytes match candidate_sha256."""

    expected = str(review.get("candidate_sha256") or "")
    task_id = str(review.get("task_id") or "")
    if displayed_path is not None and displayed_path.is_file() and _file_sha(displayed_path) == expected:
        return displayed_path
    catalog = reviewed_candidate_paths or {}
    if task_id in catalog and Path(catalog[task_id]).is_file():
        path = Path(catalog[task_id])
        if _file_sha(path) == expected:
            return path
        raise BootstrapSiteError(
            f"reviewed candidate for {task_id} does not match review candidate_sha256"
        )
    sidecar = _resolve_receipt_sidecar(
        review, review_path, "candidate_path", "candidate_file", "candidate"
    )
    if sidecar is not None:
        if _file_sha(sidecar) != expected:
            raise BootstrapSiteError(
                "review candidate_path bytes do not match candidate_sha256"
            )
        return sidecar
    raise BootstrapSiteError(
        f"cannot locate reviewed candidate bytes for task {task_id} matching candidate_sha256"
    )


_ROUTE_FS_LEAK_MARKERS: Final = (".tmp/", "/home/", "/mnt/", "/Users/", "file:")
# Official archival URLs in source bytes may contain "/home/" path segments
# (e.g. eduskunta sftp://…/home/vaskijul/…). Strip URL schemes before local-leak scan.
# No leading \\b: source XML often concatenates "pdf" + "sftp://…" without a boundary.
_URL_SCHEME_STRIP = re.compile(r"(?i)(?:sftp|https?|ftp)://\S+")


def _public_text_has_filesystem_leak(text: str) -> bool:
    scrubbed = _URL_SCHEME_STRIP.sub("", text)
    lowered = scrubbed.lower()
    return any(marker.lower() in lowered for marker in _ROUTE_FS_LEAK_MARKERS)


def _public_route_packet(packet: Mapping[str, Any]) -> dict[str, Any]:
    """Drop private provider/cache fields while keeping candidate answer text."""

    out = json.loads(json.dumps(packet, ensure_ascii=False))
    attachment = out.get("answer_attachment")
    if isinstance(attachment, dict):
        for key in ("provider", "model_id", "request_id", "cache_hit", "cache_key"):
            attachment.pop(key, None)
        binding = attachment.get("binding")
        if isinstance(binding, dict):
            for key in ("provider", "model_id", "request_id"):
                binding.pop(key, None)
        candidate = attachment.get("candidate_answer")
        if isinstance(candidate, dict):
            for key in (
                "provider",
                "model_id",
                "request_id",
                "cache_hit",
                "request_attempts",
                "subrequests",
                "accounting_projection_version",
                "prompt_version",
            ):
                candidate.pop(key, None)
    bank = out.get("bank")
    if isinstance(bank, dict):
        for key in ("sources_file", "questions_file", "path", "cache_dir"):
            value = str(bank.get(key) or "")
            if _public_text_has_filesystem_leak(value):
                bank[key] = Path(value).name if value else value
    return out


def _load_validated_route_dir(route_dir: Path) -> list[dict[str, Any]]:
    """Copy-ready route payloads from a selected directory (manifest or scan)."""

    route_dir = Path(route_dir)
    if not route_dir.is_dir():
        raise BootstrapSiteError(f"route-dir is not a directory: {route_dir}")
    selected: list[Path] = []
    manifest_path = route_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = _load_json(manifest_path)
        for row in manifest.get("tasks") or manifest.get("routes") or []:
            if not isinstance(row, Mapping):
                continue
            name = row.get("json") or row.get("packet") or row.get("path")
            if name:
                selected.append(route_dir / str(name))
        if not selected:
            for name in manifest.get("files") or []:
                if str(name).endswith(".json") and Path(name).name not in {
                    "manifest.json",
                    "SUMMARY.json",
                }:
                    selected.append(route_dir / str(name))
    if not selected:
        selected = sorted(
            path
            for path in route_dir.glob("*.json")
            if path.name not in {"manifest.json", "SUMMARY.json"}
        )
    if not selected:
        raise BootstrapSiteError(f"route-dir has no route JSON payloads: {route_dir}")
    out: list[dict[str, Any]] = []
    for path in selected:
        packet = _public_route_packet(_load_json(path))
        schema = str(packet.get("schema_version") or "")
        if schema not in _ROUTE_PACKET_SCHEMAS:
            raise BootstrapSiteError(
                f"route payload {path.name} has unsupported schema {schema!r}"
            )
        blob = json.dumps(packet, ensure_ascii=False)
        if _public_text_has_filesystem_leak(blob):
            raise BootstrapSiteError(
                f"route payload {path.name} contains private filesystem paths; refuse copy"
            )
        html_path = path.with_suffix(".html")
        out.append(
            {
                "task_id": str(packet.get("task_id") or path.stem),
                "matter_id": str(packet.get("matter_id") or ""),
                "question": packet.get("question"),
                "packet": packet,
                "source_json": path,
                "source_html": html_path if html_path.is_file() else None,
                "sha256": _file_sha(path),
            }
        )
    return out


# Explicit maintained public fields only — no suffix-based telemetry admission.
_PUBLIC_PACKET_KEYS: Final = frozenset(
    {
        "schema_version",
        "task_id",
        "matter_id",
        "question",
        "status",
        "admission",
        "answer",
        "claims",
        "unresolved",
        "next_evidence",
        "coverage",
        "source_gaps",
        "read_completeness",
        "binding_ok",
        "binding_state",
        "anchors",
        "error",
        "reading_mode",
        "ledger_projection",
        "title",
        "url",
        "source_locator",
        "identifier",
        "source_id",
        "first_pointer",
        "last_pointer",
        "start",
        "end",
        "quote",
        "source_text_sha256",
        "projection",
        "text",
        "epistemic_status",
        "evidence_bound",
        "evidence_requested",
        "unbound",
        "voice",
        "stage",
        "observation_kind",
        "label",
        "requested_source_ids",
        "indexed_source_ids",
        "fully_read_source_ids",
        "partial_read_source_ids",
        "unread_source_ids",
        "failed_source_ids",
        "reason",
        "rows",
        "state",
        "action_date",
        "actor_name",
        "source_display_name",
        "role_label",
        "institutional_result",
        "formal_and_signed_interventions",
        "group",
        "cost",
        "monitoring",
        "editorial",
        "matter_links",
        "all_requested_sources_fully_read",
        "error_detail_retained_privately",
        # Editorial successor public surface (no source_body / path / raw_capture).
        "successor_of",
        "source_revision",
        "scope",
        "sources",
        "role_intervals",
        "source_date",
        "source_type",
        "source_voice",
        "matter",
        "record_locator",
        "offset_basis",
        "page",
        "sha256",
        "bytes",
        "limit",
        "source_body_sha256",
        "takeaway",
        "points",
        "method",
        "publisher",
        "review_date",
        "correction_url",
        "repo_url",
        "source_packet",
        "overview",
    }
)

_PUBLIC_NESTED_DENY_KEYS: Final = frozenset(
    {
        "map_findings",
        "subrequests",
        "parsed",
        "deterministic_context",
        "request_id",
        "cache_hit",
        "provider",
        "model_id",
        "elapsed_seconds",
        "synthesis_witnesses",
        "raw_response",
        "prompt_sha256",
        "input_sha256",
        "session_id",
        "indexes",
        "user_text",
        "read_plan",
        "output_token_budget",
        "finding_validation_errors",
        "read_failed_windows",
        "planning_failed_rows",
        "source_body",
        "source_body_contract",
        "raw_capture",
        "path",
        "preserved_copy",
        "files",
        "encoding",
    }
)

_PUBLIC_SOURCE_META_KEYS: Final = frozenset(
    {
        "source_id",
        "title",
        "url",
        "identifier",
        "source_locator",
        "source_text_sha256",
        "text_sha256",
        "source_date",
        "source_type",
        "source_voice",
        "event_date",
        "matter_id",
    }
)


def _public_json_value(value: object, *, depth: int = 0) -> object:
    """Recursively keep deliberately public fields; drop internal/model telemetry."""

    if depth > 12:
        return None
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, item in value.items():
            name = str(key)
            if name in _PUBLIC_NESTED_DENY_KEYS or name.startswith("_"):
                continue
            if name not in _PUBLIC_PACKET_KEYS:
                continue
            cleaned = _public_json_value(item, depth=depth + 1)
            if cleaned is not None:
                out[name] = cleaned
        return out
    if isinstance(value, list):
        return [_public_json_value(item, depth=depth + 1) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, str) and _error_needs_redaction(value):
            return None
        return value
    return None


def _public_question_packet(packet: Mapping[str, Any]) -> dict[str, Any]:
    public = _public_json_value(packet)
    if not isinstance(public, dict):
        public = {}
    raw_error = packet.get("error")
    if raw_error is not None and raw_error != "":
        public["error"] = _public_error_message(raw_error)
        public["error_detail_retained_privately"] = _error_needs_redaction(raw_error)
    return public


def _ingest_evidence_row(sources: dict[str, dict[str, Any]], row: Mapping[str, Any]) -> None:
    source_id = str(row.get("source_id") or "")
    if not source_id:
        return
    patch = {
        key: row[key]
        for key in ("title", "url", "source_locator", "identifier", "source_text_sha256")
        if key in row and row[key] is not None
    }
    sources[source_id] = {**sources.get(source_id, {"source_id": source_id}), **patch}


def _packet_source_map(*packets: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    sources: dict[str, dict[str, Any]] = {}
    for packet in packets:
        if not isinstance(packet, Mapping):
            continue
        for claim in packet.get("claims") or []:
            for row in claim.get("evidence_bound") or []:
                if isinstance(row, Mapping):
                    _ingest_evidence_row(sources, row)
        for finding in packet.get("map_findings") or []:
            if isinstance(finding, Mapping):
                _ingest_evidence_row(sources, finding)
        for key in ("rows", "points"):
            for row in packet.get(key) or []:
                if not isinstance(row, Mapping):
                    continue
                for evidence in row.get("evidence_bound") or []:
                    if isinstance(evidence, Mapping):
                        _ingest_evidence_row(sources, evidence)
        for key in ("sources", "source_metadata"):
            blob = packet.get(key)
            if isinstance(blob, Mapping):
                for source_id, meta in blob.items():
                    if isinstance(meta, Mapping):
                        patch = {
                            name: meta[name]
                            for name in _PUBLIC_SOURCE_META_KEYS
                            if name in meta and meta[name] is not None
                        }
                        sources[str(source_id)] = {
                            **sources.get(str(source_id), {"source_id": str(source_id)}),
                            **patch,
                        }
            elif isinstance(blob, list):
                for meta in blob:
                    if isinstance(meta, Mapping) and meta.get("source_id"):
                        sid = str(meta["source_id"])
                        patch = {
                            name: meta[name]
                            for name in _PUBLIC_SOURCE_META_KEYS
                            if name in meta and meta[name] is not None
                        }
                        sources[sid] = {**sources.get(sid, {"source_id": sid}), **patch}
    return sources


def _overview_section(
    overview: Mapping[str, Any] | None,
    *,
    sources: Mapping[str, Mapping[str, Any]] | None = None,
    navigation_html: str = "",
) -> str:
    """Render optional overview fields from input data only (no renderer conclusions)."""

    if not isinstance(overview, Mapping) or not overview:
        return ""
    points = ""
    for row in overview.get("points") or []:
        if not isinstance(row, Mapping):
            continue
        evidence = "".join(
            _bound_evidence_block(item, sources)
            for item in (row.get("evidence_bound") or [])
            if isinstance(item, Mapping)
        )
        title = str(row.get("question") or row.get("title") or "")
        body_text = str(row.get("text") or row.get("answer") or "")
        # Keep short findings open; fold long methodology/comparison bodies.
        fold_body = len(body_text) > 280 or "Säästöennusteet" in body_text or (
            "vertailukelpois" in title.casefold()
        )
        text_html = (
            (
                f"<details class='point-detail'><summary>Laajempi tausta</summary>"
                f"<p>{_e(body_text)}</p></details>"
            )
            if fold_body and body_text
            else (f"<p>{_e(body_text)}</p>" if body_text else "")
        )
        points += (
            "<article>"
            f"<h3>{_e(title)}</h3>"
            + (f"<p class='muted'>{_e(row.get('limit'))}</p>" if row.get("limit") else "")
            + text_html
            + evidence
            + "</article>"
        )
    meta = ""
    if overview.get("publisher") or overview.get("review_date") or overview.get("correction_url"):
        meta = (
            "<p class='muted'>"
            + (f"Julkaisija: {_e(overview.get('publisher'))}. " if overview.get("publisher") else "")
            + (f"Päiväys: {_e(overview.get('review_date'))}. " if overview.get("review_date") else "")
            + (f"{_e(overview.get('method'))} " if overview.get("method") else "")
            + (
                f"<a href='{_e(overview.get('correction_url'))}'>Ilmoita korjaus</a>."
                if overview.get("correction_url")
                else ""
            )
            + "</p>"
        )
    return (
        "<section class='overview'>"
        f"<h2>{_e(overview.get('question') or 'Lyhyt vastaus')}</h2>"
        + (f"<p><strong>{_e(overview.get('answer'))}</strong></p>" if overview.get("answer") else "")
        + navigation_html
        + meta
        + points
        + (
            "<details class='overview-method'><summary>Säästöarvioiden menetelmärajat</summary>"
            f"<p>{_e(overview.get('takeaway'))}</p></details>"
            if overview.get("takeaway")
            else ""
        )
        + "</section>"
    )


def build_oulaskangas_dossier(
    *,
    group_packets: Sequence[Mapping[str, Any]],
    cost_packets: Sequence[Mapping[str, Any]],
    monitoring_packets: Sequence[Mapping[str, Any]] | None = None,
    matter_links: Mapping[str, str] | None = None,
    editorial_packet: Mapping[str, Any] | None = None,
    editorial_overview: Mapping[str, Any] | None = None,
    dossier_json_href: str | None = None,
) -> str:
    """Connect candidate packets generically; no renderer-coded numeric conclusions."""

    matter_links = matter_links or {}
    links = "".join(
        f"<li><a href='{_e(path)}'>{_e(label)}</a></li>" for label, path in matter_links.items()
    )
    sources = _packet_source_map(
        *group_packets,
        *cost_packets,
        *(monitoring_packets or []),
        *([editorial_packet] if editorial_packet else []),
        *([editorial_overview] if editorial_overview else []),
    )
    group_body = "".join(_question_article(packet, sources=sources) for packet in group_packets)
    cost_body = "".join(_question_article(packet, sources=sources) for packet in cost_packets)
    monitoring_body = "".join(
        _question_article(packet, sources=sources) for packet in (monitoring_packets or [])
    )
    editorial_rows = ""
    if editorial_packet:
        for row in editorial_packet.get("rows") or []:
            if not isinstance(row, Mapping):
                continue
            evidence = row.get("evidence_bound") or []
            text = str(row.get("text") or "").strip()
            first_quote = ""
            if evidence and isinstance(evidence[0], Mapping):
                first_quote = str(evidence[0].get("quote") or "").strip()
            # Do not repeat the same exact excerpt as visible paragraph and closed quote.
            show_text = text and text != first_quote and text not in first_quote and first_quote not in text
            evidence_html = "".join(_bound_evidence_block(item, sources) for item in evidence)
            if not evidence:
                evidence_html = "<p class='muted'>Ei sidottua lähdekatkelmaa.</p>"
            editorial_rows += (
                "<article>"
                f"<h3>{_e(row.get('title') or row.get('label'))}</h3>"
                + (f"<p>{_e(text)}</p>" if show_text else "")
                + f"<p class='muted'>{_e(row.get('voice') or '')} · {_e(row.get('stage') or '')} · "
                f"{_e(row.get('observation_kind') or '')}</p>"
                + evidence_html
                + "</article>"
            )
    json_link = (
        f"<p><a download href='{_e(dossier_json_href)}'>Lataa dossier-paketti (JSON)</a></p>"
        if dossier_json_href
        else ""
    )
    matter_navigation = (
        "<details class='matter-navigation'><summary>Valitse asia ja henkilö</summary>"
        "<p>Avaa kirjaukset ja suodata henkilön mukaan.</p>"
        f"<ul>{links}</ul></details>"
        if links
        else ""
    )
    overview_html = _overview_section(
        editorial_overview, sources=sources, navigation_html=matter_navigation
    )
    candidate_sections = ""
    if group_packets:
        candidate_sections += (
            "<details><summary>Ryhmäkonteksti (ehdokasvastaukset)</summary>"
            + (group_body or "<p>Ei toimitettuja ryhmävastauksia tässä paketissa.</p>")
            + "</details>"
        )
    if cost_packets:
        candidate_sections += (
            "<details><summary>Kustannuskysymykset (ehdokasvastaukset)</summary>"
            + (cost_body or "<p>Ei toimitettuja kustannusvastauksia tässä paketissa.</p>")
            + "</details>"
        )
    if monitoring_packets:
        candidate_sections += (
            "<details><summary>Seurantakysymykset (ehdokasvastaukset)</summary>"
            "<p class='muted'>Havainto, ennuste ja raportin arvio erotetaan; "
            "ehdokas ≠ hyväksytty havainto.</p>"
            + (monitoring_body or "<p>Ei toimitettuja seurantavastauksia tässä paketissa.</p>")
            + "</details>"
        )
    body = (
        "<h1>Oulaskangas</h1>"
        + (
            overview_html
            if overview_html
            else (
                "<p>Yhdistetty ehdokasnäkymä hankituista lähteistä. "
                "Luvut ja päätelmät näkyvät vain lähdesidonnaisina riveinä tai ehdokasvastauksina.</p>"
            )
        )
        + json_link
        + (matter_navigation if not overview_html else "")
        + candidate_sections
        + (
            "<details class='audit'><summary>Lähde- ja tekniset vertailurivit (tarkastus)</summary>"
            "<p>Ennuste, vastalause, havainto ja institutionaalinen tulos erillään. "
            "Sama ote ei toistu avoimena tekstinä ja suljetussa lainauksessa.</p>"
            + editorial_rows
            + "</details>"
            if editorial_rows
            else ""
        )
    )
    return _document(
        "Oulaskangas",
        body,
        description="Oulaskankaan päätöksen, kustannusten, seurannan ja ryhmäreaktion yhdistetty ehdokasluku.",
    )


def build_oulaskangas_dossier_packet(
    *,
    group_packets: Sequence[Mapping[str, Any]],
    cost_packets: Sequence[Mapping[str, Any]],
    monitoring_packets: Sequence[Mapping[str, Any]] | None = None,
    matter_links: Mapping[str, str] | None = None,
    editorial_packet: Mapping[str, Any] | None = None,
    editorial_overview: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Allowlisted dossier export for verifying editorial/DA evidence outside HTML."""

    return {
        "schema_version": SCHEMA_VERSION,
        "admission": "CANDIDATE_PREVIEW_NOT_REFERENCE_JUDGED",
        "matter_links": dict(matter_links or {}),
        "group": [_public_question_packet(packet) for packet in group_packets],
        "cost": [_public_question_packet(packet) for packet in cost_packets],
        "monitoring": [_public_question_packet(packet) for packet in (monitoring_packets or [])],
        "editorial": _public_json_value(editorial_packet) if editorial_packet else None,
        "overview": _public_json_value(editorial_overview) if editorial_overview else None,
    }


def _require_clean_output_dir(output_dir: Path) -> None:
    """Refuse publishing into a non-empty directory (no stale-file mix)."""

    if not output_dir.exists():
        return
    if not output_dir.is_dir():
        raise BootstrapSiteError(f"output path exists and is not a directory: {output_dir}")
    leftover = sorted(path.name for path in output_dir.iterdir())
    if leftover:
        raise BootstrapSiteError(
            "output directory is not empty; refuse stale-file publication: "
            + ", ".join(leftover[:12])
            + ("…" if len(leftover) > 12 else "")
        )


def _load_journey_packet(value: Mapping[str, Any] | Path) -> dict[str, Any]:
    if isinstance(value, Mapping):
        packet = dict(value)
    else:
        packet = _load_json(Path(value))
    if packet.get("schema_version") != "paa.question_journey.v1":
        raise BootstrapSiteError(
            f"unsupported journey schema {packet.get('schema_version')!r}"
        )
    if not str(packet.get("task_id") or "").strip():
        raise BootstrapSiteError("journey packet missing task_id")
    # Prose review attachment must never silently upgrade projection_status.
    status = str(packet.get("projection_status") or "UNREVIEWED")
    if status.startswith("REVIEWED") and not isinstance(packet.get("projection_review"), Mapping):
        raise BootstrapSiteError(
            f"journey {packet.get('task_id')} claims REVIEWED projection without projection_review"
        )
    if (
        isinstance(packet.get("source_answer_review"), Mapping)
        and not isinstance(packet.get("projection_review"), Mapping)
        and status.startswith("REVIEWED")
    ):
        raise BootstrapSiteError(
            "source_answer_review cannot admit journey projection_status REVIEWED"
        )
    return packet


def build_bootstrap_artifact(
    *,
    output_dir: Path,
    matter_packets: Sequence[Path],
    question_packets: Sequence[Path],
    oulaskangas_group: Sequence[Path] | None = None,
    oulaskangas_cost: Sequence[Path] | None = None,
    oulaskangas_monitoring: Sequence[Path] | None = None,
    editorial_packet_path: Path | None = None,
    editorial_overview_path: Path | None = None,
    source_bank_paths: Sequence[Path] | None = None,
    route_dirs: Sequence[Path] | None = None,
    review_paths: Sequence[Path] | None = None,
    reviewed_candidate_paths: Mapping[str, Path] | None = None,
    journey_packets: Sequence[Mapping[str, Any] | Path] | None = None,
    title: str = DEFAULT_TITLE,
    publisher: str = DEFAULT_PUBLISHER,
    correction_url: str = DEFAULT_CORRECTION_URL,
    repo_url: str = DEFAULT_REPO_URL,
    review_date: str | None = None,
) -> dict[str, Any]:
    """Write a new selectable static artifact; leave older dist trees untouched."""

    _require_clean_output_dir(output_dir)
    bank_search_paths = [Path(p) for p in (source_bank_paths or [])]
    reviewed_candidates = {
        str(key): Path(value) for key, value in (reviewed_candidate_paths or {}).items()
    }
    overview = _load_json(editorial_overview_path) if editorial_overview_path else None
    checked_on = (
        (review_date or "").strip()
        or (str(overview.get("review_date")).strip() if isinstance(overview, Mapping) and overview.get("review_date") else "")
        or datetime.now(UTC).date().isoformat()
    )
    publisher = (
        str(overview.get("publisher")).strip()
        if isinstance(overview, Mapping) and overview.get("publisher")
        else publisher
    )
    correction_url = (
        str(overview.get("correction_url")).strip()
        if isinstance(overview, Mapping) and overview.get("correction_url")
        else correction_url
    )
    repo_url = (
        str(overview.get("repo_url")).strip()
        if isinstance(overview, Mapping) and overview.get("repo_url")
        else repo_url
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    matters_dir = output_dir / "matters"
    questions_dir = output_dir / "questions"
    matters_dir.mkdir(exist_ok=True)
    questions_dir.mkdir(exist_ok=True)

    bank_sources: dict[str, dict[str, Any]] = {}
    for path in source_bank_paths or []:
        doc = _load_json(path)
        for row in doc.get("sources") or []:
            if isinstance(row, Mapping) and row.get("source_id"):
                sid = str(row["source_id"])
                bank_sources[sid] = {
                    key: row[key]
                    for key in (
                        "source_id",
                        "title",
                        "url",
                        "identifier",
                        "source_locator",
                        "text_sha256",
                        "event_date",
                        "source_date",
                        "source_voice",
                        "matter_id",
                    )
                    if key in row
                }

    matter_entries: list[dict[str, Any]] = []
    matter_links: dict[str, str] = {}
    for path in matter_packets:
        packet = _load_json(path)
        matter_id = str((packet.get("matter") or {}).get("matter_id") or packet.get("matter_id") or path.stem)
        slug = _slug(matter_id)
        html_name = f"{slug}.html"
        json_name = f"{slug}.json"
        (matters_dir / json_name).write_text(
            json.dumps(packet, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        site = write_matter_site(
            packet,
            matters_dir / html_name,
            packet_url=json_name,
            back_url="../index.html",
            lookup_url="../lookup.html",
        )
        interventions = packet.get("interventions") or []
        subject = resolve_matter_subject_title(packet.get("matter") or {})
        choice_label = matter_choice_label(matter_id, subject)
        people_base = f"matters/{slug}/people/"
        actor_summaries = compact_actor_summaries(
            [row for row in interventions if isinstance(row, Mapping)]
        )
        lookup_actors = [
            {
                "actor_name": str(row.get("name") or "").split(" (vain nimi", 1)[0],
                "actor_key": row.get("actor_key"),
                "href": f"{people_base}{row['slug']}.html",
                "identity_state": row.get("identity_state"),
                "label": f"Kirjatut teot · {matter_id}",
                "date": row.get("latest_action_date"),
                "coverage_kind": "RECORD_ONLY",
            }
            for row in actor_summaries
            if row.get("name") and row.get("slug")
        ]
        matter_entries.append(
            {
                "matter_id": matter_id,
                "path": f"matters/{html_name}",
                "json_path": f"matters/{json_name}",
                "title": f"{choice_label}: mitä henkilöt tekivät?",
                "choice_label": choice_label,
                "subject_title": subject.get("subject_title"),
                "subject_title_state": subject.get("subject_title_state"),
                "intervention_count": len(interventions),
                "actor_count": site["actor_count"],
                "people_paths": [f"matters/{rel}" for rel in site["people"]],
                "record_paths": [f"matters/{rel}" for rel in site["records"]],
                "lookup_actors": lookup_actors,
                "description": (
                    "Asiatunnisteeseen kirjatut allekirjoitukset, nimetyt vaihtoehdot, "
                    "puheet ja äänestykset; täydet kirjaukset henkilösivuilla."
                ),
            }
        )
        matter_links[matter_id] = f"matters/{html_name}"

    reviews_by_task: dict[str, dict[str, Any]] = {}
    for review_path in review_paths or []:
        receipt = _load_json(review_path)
        task_id = str(receipt.get("task_id") or "")
        if not task_id:
            raise BootstrapSiteError(f"review receipt missing task_id: {review_path}")
        reviews_by_task[task_id] = {"receipt": receipt, "path": Path(review_path)}

    def _public_review_fields(review_payload: Mapping[str, Any]) -> dict[str, Any]:
        return {
            key: review_payload[key]
            for key in (
                "schema_version",
                "task_id",
                "candidate_sha256",
                "questions_sha256",
                "sources_sha256",
                "reviewer",
                "review_date",
                "decision",
                "reviewed_fields",
                "limits",
                "note",
                "reader_title",
                "reader_takeaway",
            )
            if key in review_payload
        }

    question_index: list[dict[str, Any]] = []
    for path in question_packets:
        if path.name == "SUMMARY.json":
            continue
        path = Path(path)
        packet = _load_json(path)
        matter_id = str(packet.get("matter_id") or "unknown")
        task_id = str(packet.get("task_id") or path.stem)
        slug = _slug(f"{task_id}_{matter_id}")
        html_name = f"{slug}.html"
        json_name = f"{slug}.json"
        review_payload = None
        display_packet = packet
        if task_id in reviews_by_task:
            entry = reviews_by_task[task_id]
            reviewed_file = _resolve_reviewed_candidate_file(
                entry["receipt"],
                review_path=entry["path"],
                displayed_path=path,
                reviewed_candidate_paths=reviewed_candidates,
            )
            # Displayed packet must be the reviewed bytes or a projection of them.
            if _file_sha(path) != _file_sha(reviewed_file):
                reviewed_packet = _load_json(reviewed_file)
                if _analytical_answer_fields(packet) != _analytical_answer_fields(reviewed_packet):
                    raise BootstrapSiteError(
                        f"question packet {path.name} analytical fields differ from reviewed "
                        f"candidate for {task_id}; refuse review attachment"
                    )
                display_packet = reviewed_packet
            review_payload = _validate_review_receipt(
                entry["receipt"],
                candidate_path=reviewed_file,
                expected_task_id=task_id,
                bank_search_paths=bank_search_paths,
            )
            reviews_by_task[task_id]["bound"] = review_payload
            reviews_by_task[task_id]["reviewed_candidate"] = reviewed_file
        public_packet = _public_question_packet(display_packet)
        if review_payload:
            public_packet["source_answer_review"] = _public_review_fields(review_payload)
        (questions_dir / json_name).write_text(
            json.dumps(public_packet, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        page_title = (
            str((review_payload or {}).get("reader_title") or "")
            or str(display_packet.get("question") or task_id)
        )
        page_body = "<p><a href='../index.html'>Takaisin</a></p>"
        if review_payload:
            page_body += f"<h1>{html.escape(page_title)}</h1>"
        page_body += _question_article(
            display_packet,
            sources=bank_sources,
            packet_json_href=json_name,
            review=review_payload,
        )
        page_body += f"<p><a href='{_e(correction_url)}'>Ilmoita korjaus</a></p>"
        page = _document(
            page_title,
            page_body,
            description="Lähdekysymyksen ehdokasvastaus otteineen.",
        )
        (questions_dir / html_name).write_text(page, encoding="utf-8")
        question_index.append(
            {
                "task_id": task_id,
                "matter_id": matter_id,
                "question": display_packet.get("question"),
                "reader_title": (review_payload or {}).get("reader_title"),
                "reader_takeaway": (review_payload or {}).get("reader_takeaway"),
                "path": f"questions/{html_name}",
                "json_path": f"questions/{json_name}",
                "status": display_packet.get("status"),
                "admission": display_packet.get("admission") or "CANDIDATE_NOT_REFERENCE_JUDGED",
                "binding_ok": display_packet.get("binding_ok"),
                "read_completeness": display_packet.get("read_completeness")
                or (display_packet.get("coverage") or {}).get("read_completeness"),
                "ai_source_review": bool(review_payload),
            }
        )

    routes_dir = output_dir / "routes"
    route_index: list[dict[str, Any]] = []
    journey_by_task: dict[str, Mapping[str, Any]] = {}
    for raw_journey in journey_packets or []:
        if isinstance(raw_journey, Mapping) and raw_journey.get("task_id"):
            journey_by_task[str(raw_journey["task_id"])] = raw_journey
        elif isinstance(raw_journey, (str, Path)):
            path = Path(raw_journey)
            if not path.is_file():
                continue
            loaded = _load_json(path)
            if isinstance(loaded, Mapping) and loaded.get("task_id"):
                journey_by_task[str(loaded["task_id"])] = loaded
    for route_dir in route_dirs or []:
        for entry in _load_validated_route_dir(Path(route_dir)):
            routes_dir.mkdir(exist_ok=True)
            task_id = entry["task_id"]
            slug = _slug(task_id)
            json_name = f"{slug}.json"
            html_name = f"{slug}.html"
            packet = dict(entry["packet"])
            review_payload = None
            if task_id in reviews_by_task:
                rev = reviews_by_task[task_id]
                reviewed_file = _resolve_reviewed_candidate_file(
                    rev["receipt"],
                    review_path=rev["path"],
                    displayed_path=None,
                    reviewed_candidate_paths=reviewed_candidates,
                )
                review_payload = _validate_review_receipt(
                    rev["receipt"],
                    candidate_path=reviewed_file,
                    expected_task_id=task_id,
                    bank_search_paths=bank_search_paths,
                )
                reviewed_candidate = _load_json(reviewed_file)
                attachment = packet.get("answer_attachment") or {}
                embedded = attachment.get("candidate_answer") if isinstance(attachment, Mapping) else None
                if not isinstance(embedded, Mapping):
                    raise BootstrapSiteError(
                        f"route {task_id} has review but no embedded candidate_answer"
                    )
                expected_fields = _analytical_answer_fields(reviewed_candidate)
                # Ensure task/question/matter on embedded answer for comparison.
                embedded_for_compare = {
                    **embedded,
                    "task_id": embedded.get("task_id") or packet.get("task_id"),
                    "matter_id": embedded.get("matter_id") or packet.get("matter_id"),
                    "question": embedded.get("question") or packet.get("question"),
                }
                if _analytical_answer_fields(embedded_for_compare) != expected_fields:
                    raise BootstrapSiteError(
                        f"route {task_id} embedded candidate_answer analytical fields "
                        "differ from reviewed candidate; refuse review attachment"
                    )
                route_bank = packet.get("bank") or {}
                if review_payload.get("questions_sha256") and str(
                    route_bank.get("questions_sha256") or ""
                ) != str(review_payload["questions_sha256"]):
                    raise BootstrapSiteError(
                        f"route {task_id} questions_sha256 differs from review receipt"
                    )
                if review_payload.get("sources_sha256") and str(
                    route_bank.get("sources_sha256") or ""
                ) != str(review_payload["sources_sha256"]):
                    raise BootstrapSiteError(
                        f"route {task_id} sources_sha256 differs from review receipt"
                    )
                # Replace embedded answer with reviewed analytical fields (exact).
                attachment = dict(attachment) if isinstance(attachment, Mapping) else {}
                attachment["candidate_answer"] = {
                    **dict(embedded),
                    "task_id": reviewed_candidate.get("task_id"),
                    "matter_id": reviewed_candidate.get("matter_id"),
                    "question": reviewed_candidate.get("question"),
                    "answer": reviewed_candidate.get("answer"),
                    "claims": reviewed_candidate.get("claims"),
                    "unresolved": reviewed_candidate.get("unresolved"),
                    "next_evidence": reviewed_candidate.get("next_evidence"),
                    "coverage": reviewed_candidate.get("coverage"),
                    "admission": reviewed_candidate.get("admission")
                    or embedded.get("admission"),
                    "status": reviewed_candidate.get("status") or embedded.get("status"),
                }
                packet["answer_attachment"] = attachment
                packet["source_answer_review"] = _public_review_fields(review_payload)
            (routes_dir / json_name).write_text(
                json.dumps(packet, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            from paa.route_packet import render_route_html

            journey_packet = journey_by_task.get(str(task_id))
            inherited_title = ""
            prepared_answer = ""
            journey_href = None
            if isinstance(journey_packet, Mapping):
                journey_href = f"../journeys/{_slug(str(task_id))}.html"
                prepared_answer = str(journey_packet.get("short_answer") or "").strip()
                editorial = str(journey_packet.get("editorial_title") or "").strip()
                journey_reader = str(
                    journey_packet.get("reader_title")
                    or (journey_packet.get("source_answer_review") or {}).get("reader_title")
                    or ""
                ).strip()
                if journey_packet.get("prefer_editorial_title") is True and editorial:
                    inherited_title = editorial
                else:
                    inherited_title = journey_reader or editorial
            if review_payload and review_payload.get("reader_title"):
                inherited_title = str(review_payload.get("reader_title") or inherited_title)
            page = render_route_html(
                packet,
                packet_url=json_name,
                back_url="../index.html",
                correction_url=correction_url,
                reader_title=inherited_title or None,
                journey_href=journey_href,
                prepared_answer=prepared_answer or None,
            )
            (routes_dir / html_name).write_text(page, encoding="utf-8")
            answer_text = prepared_answer
            attachment = packet.get("answer_attachment") or {}
            candidate = attachment.get("candidate_answer") if isinstance(attachment, Mapping) else None
            if not answer_text and isinstance(candidate, Mapping):
                answer_text = str(candidate.get("answer") or "")[:280]
            route_index.append(
                {
                    "task_id": task_id,
                    "matter_id": entry.get("matter_id") or packet.get("matter_id"),
                    "question": packet.get("question") or entry.get("question"),
                    "reader_title": inherited_title
                    or (review_payload or {}).get("reader_title"),
                    "reader_takeaway": (review_payload or {}).get("reader_takeaway")
                    or prepared_answer,
                    "path": f"routes/{html_name}",
                    "json_path": f"routes/{json_name}",
                    "answer_excerpt": answer_text,
                    "admission": packet.get("admission") or "CANDIDATE_NOT_REFERENCE_JUDGED",
                    "ai_source_review": bool(review_payload),
                    "review_decision": (review_payload or {}).get("decision"),
                    "journey_path": (
                        f"journeys/{_slug(str(task_id))}.html" if journey_href else None
                    ),
                }
            )

    from paa.question_journey import (
        build_lookup_index,
        render_journey_html,
        render_lookup_html,
    )

    journeys_dir = output_dir / "journeys"
    journey_index: list[dict[str, Any]] = []
    journey_packets_loaded: list[dict[str, Any]] = []
    route_by_task = {str(item["task_id"]): item for item in route_index}
    question_by_task = {str(item["task_id"]): item for item in question_index}
    for raw in journey_packets or []:
        packet = _load_journey_packet(raw)
        task_id = str(packet["task_id"])
        slug = _slug(task_id)
        html_name = f"{slug}.html"
        json_name = f"{slug}.json"
        journeys_dir.mkdir(exist_ok=True)
        nav_links: list[dict[str, str]] = []
        matter_id = str(packet.get("matter_id") or "")
        if matter_id and matter_id in matter_links:
            nav_links.append({"label": "Asia ja henkilöt", "href": f"../{matter_links[matter_id]}"})
        if task_id in route_by_task:
            nav_links.append(
                {"label": "Dokumentaarinen reitti", "href": f"../{route_by_task[task_id]['path']}"}
            )
        if task_id in question_by_task:
            nav_links.append(
                {"label": "Kysymysvastaus", "href": f"../{question_by_task[task_id]['path']}"}
            )
        row = dict(packet)
        row["site_href"] = f"journeys/{html_name}"
        journey_packets_loaded.append(row)
        (journeys_dir / json_name).write_text(
            json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        page = render_journey_html(
            row,
            packet_url=json_name,
            back_url="../index.html",
            lookup_url="../lookup.html",
            correction_url=correction_url,
            nav_links=nav_links,
        )
        (journeys_dir / html_name).write_text(page, encoding="utf-8")
        projection_review = (
            packet.get("projection_review")
            if isinstance(packet.get("projection_review"), Mapping)
            else None
        )
        source_review = (
            packet.get("source_answer_review")
            if isinstance(packet.get("source_answer_review"), Mapping)
            else {}
        )
        reader_title = (
            packet.get("editorial_title")
            if packet.get("prefer_editorial_title") is True and packet.get("editorial_title")
            else (
                packet.get("reader_title")
                or source_review.get("reader_title")
                or packet.get("editorial_title")
            )
        )
        journey_index.append(
            {
                "task_id": task_id,
                "matter_id": matter_id,
                "question": packet.get("question"),
                "short_answer": packet.get("short_answer"),
                "reader_title": reader_title,
                "reader_takeaway": source_review.get("reader_takeaway"),
                "journey_kind": packet.get("journey_kind"),
                "path": f"journeys/{html_name}",
                "json_path": f"journeys/{json_name}",
                "projection_status": packet.get("projection_status") or "UNREVIEWED",
                "admission": packet.get("admission"),
                "projection_reviewed": bool(projection_review),
                "human_certification": (
                    False
                    if projection_review is not None
                    else None
                ),
                "source_answer_review": bool(packet.get("source_answer_review")),
            }
        )

    group_packets = [_load_json(path) for path in oulaskangas_group or [] if path.name != "SUMMARY.json"]
    cost_packets = [_load_json(path) for path in oulaskangas_cost or [] if path.name != "SUMMARY.json"]
    monitoring_packets = [
        _load_json(path) for path in oulaskangas_monitoring or [] if path.name != "SUMMARY.json"
    ]
    editorial = _load_json(editorial_packet_path) if editorial_packet_path else None
    dossier_packet = build_oulaskangas_dossier_packet(
        group_packets=group_packets,
        cost_packets=cost_packets,
        monitoring_packets=monitoring_packets,
        matter_links=matter_links,
        editorial_packet=editorial,
        editorial_overview=overview if isinstance(overview, Mapping) else None,
    )
    (output_dir / "oulaskangas.json").write_text(
        json.dumps(dossier_packet, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    dossier = build_oulaskangas_dossier(
        group_packets=group_packets,
        cost_packets=cost_packets,
        monitoring_packets=monitoring_packets,
        matter_links=matter_links,
        editorial_packet=editorial,
        editorial_overview=overview if isinstance(overview, Mapping) else None,
        dossier_json_href="oulaskangas.json",
    )
    (output_dir / "oulaskangas.html").write_text(dossier, encoding="utf-8")
    matter_cards = "".join(
        "<article class='matter-row'><a href='"
        + _e(entry["path"])
        + "'>"
        + _e(entry.get("choice_label") or entry["matter_id"])
        + "</a>"
        + (
            f"<span class='muted'>{_e(entry['matter_id'])}</span>"
            if entry.get("subject_title")
            else (
                f"<span class='muted'>{_e(entry.get('matter_id') or '')}</span>"
                if entry.get("title") and entry.get("title") != entry["matter_id"]
                else ""
            )
        )
        + "</article>"
        for entry in matter_entries
    )
    # Reviewed projection journeys first; unreviewed stay honestly labelled.
    journey_index_sorted = sorted(
        journey_index,
        key=lambda item: (0 if item.get("projection_reviewed") else 1, str(item.get("task_id"))),
    )
    journey_cards = "<ul class='question-list'>" + "".join(
        "<li data-matter='"
        + _e(item.get("matter_id") or "")
        + "' data-projection='"
        + _e(item.get("projection_status") or "UNREVIEWED")
        + "'>"
        + "<a href='"
        + _e(item["path"])
        + "'>"
        + _e(
            item.get("reader_title")
            or item.get("question")
            or item["task_id"]
        )
        + "</a>"
        + (
            " <small>(Kaavioehdokas)</small>"
            if not item.get("projection_reviewed")
            else ""
        )
        + "</li>"
        for item in journey_index_sorted
    ) + "</ul>" if journey_index_sorted else ""
    journey_review_line = (
        "<p class='muted'>Kaavion tekoälylähdearvio · ei ihmistarkistusta.</p>"
        if any(item.get("projection_reviewed") for item in journey_index_sorted)
        else ""
    )
    route_cards = "".join(
        "<article class='route-card'>"
        f"<h3><a href='{_e(item['path'])}'>"
        f"{_e(item.get('reader_title') or item.get('question') or item['task_id'])}</a></h3>"
        + (
            f"<p>{_e(item.get('reader_takeaway'))}</p>"
            if item.get("reader_takeaway")
            else "<p class='muted'>Dokumentaarinen reitti.</p>"
        )
        + (
            "<p class='muted'>Erillinen tekoälylukija tarkisti vastauksen lähteet; ei ihmisen varmennus. "
            "Ei automaattisesti kaavion projektioarvio.</p>"
            if item.get("ai_source_review")
            else f"<p class='muted'>{_e(item.get('matter_id') or '')} · ehdokas</p>"
        )
        + (
            f"<p><a href='journeys/{_e(_slug(str(item['task_id'])))}.html'>Avaa kysymysmatka</a></p>"
            if str(item["task_id"]) in {row["task_id"] for row in journey_index}
            else ""
        )
        + "<details><summary>Tekniset tiedot</summary><p class='muted'>"
        + f"<a href='{_e(item['json_path'])}'>JSON</a> · {_e(item['task_id'])}</p></details>"
        + "</article>"
        for item in route_index
    )
    matter_lookup_records = [
        {
            "matter_id": entry["matter_id"],
            "title": entry.get("title"),
            "choice_label": entry.get("choice_label"),
            "href": entry["path"],
            "aliases": [
                alias
                for alias in (
                    entry.get("choice_label"),
                    entry.get("subject_title"),
                    entry["matter_id"],
                )
                if alias
            ],
            "actors": entry.get("lookup_actors") or [],
            "coverage_kind": "RECORD_ONLY",
        }
        for entry in matter_entries
    ]
    dossier_lookup_entries = [
        {
            "title": "Oulaskangas",
            "aliases": [
                "Oulaskangas",
                "Oulaskankaan",
                "Oulaskangas-sairaala",
                "Oulaskankaan sairaala",
            ],
            "href": "oulaskangas.html",
            "matter_ids": [
                mid
                for mid in ("HE 149/2024 vp", "KAA 2/2025 vp", "HE 12/2024 vp")
                if mid in matter_links
            ],
            "coverage_kind": "RECORD_ONLY",
            "note": (
                "Dossier and retained matter hubs; not a prepared journey answer. "
                "Person acts remain on matter person pages."
            ),
        }
    ]
    lookup_index = build_lookup_index(
        journey_packets_loaded,
        matter_records=matter_lookup_records,
        dossier_entries=dossier_lookup_entries,
    )
    lookup_index["questions"] = list(lookup_index.get("questions") or []) + [
        {
            "task_id": item["task_id"],
            "matter_id": item["matter_id"],
            "question": item.get("question"),
            "reader_title": item.get("reader_title") or item.get("question"),
            "short_answer": item.get("reader_takeaway") or "",
            "href": item["path"],
            "projection_status": "PREPARED_ANSWER",
            "coverage_kind": "PREPARED_AI_SOURCE_REVIEWED",
        }
        for item in question_index
        if item["task_id"] not in {row.get("task_id") for row in lookup_index.get("questions") or []}
    ]
    (output_dir / "lookup.json").write_text(
        json.dumps(lookup_index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "lookup.html").write_text(
        render_lookup_html(lookup_index, back_url="index.html"), encoding="utf-8"
    )
    journey_task_ids = {str(row.get("task_id")) for row in journey_index}
    prepared_items = [
        item
        for item in question_index
        if str(item.get("task_id")) not in journey_task_ids
    ]
    # The homepage is a question index, not a second copy of every answer.
    # Keep review status accurate once for the collection; each title opens the
    # full source-bound answer and its own status details.
    question_cards = (
        "<ul class='question-list'>"
        + "".join(
            "<li><a href='"
            + _e(item["path"])
            + "'>"
            + _e(item.get("reader_title") or item.get("question") or item["task_id"])
            + "</a></li>"
            for item in prepared_items
        )
        + "</ul>"
        if prepared_items
        else ""
    )
    prepared_reviewed = any(bool(item.get("ai_source_review")) for item in prepared_items)
    prepared_candidates = any(not item.get("ai_source_review") for item in prepared_items)
    review_basis_parts = []
    if prepared_reviewed:
        review_basis_parts.append(
            "Erillinen tekoälylukija on tarkistanut osan lähteistä; ei ihmistarkistusta."
        )
    if prepared_candidates:
        review_basis_parts.append(
            "Muut ovat lähteisiin sidottuja ehdokkaita, eivät tarkistettuja vastauksia."
        )
    question_review_basis = (
        f"<p class='muted'>{_e(' '.join(review_basis_parts))}</p>"
        if review_basis_parts
        else ""
    )
    index = _document(
        title,
        f"<h1>{_e(title)}</h1>"
        + "<p>Kuka ehdotti, tuki tai muutti mitä — ja mitä päätettiin?</p>"
        + "<p><a href='lookup.html'>Hae henkilöä, asiaa tai kysymystä</a></p>"
        + (
            "<h2 id='matkat'>Kysymysmatkat</h2>"
            f"{journey_review_line}"
            f"<div id='journey-list'>{journey_cards}</div>"
            if journey_cards
            else ""
        )
        + "<h2 id='ehdokkaat'>Valmistellut kysymykset</h2>"
        + (
            f"{question_review_basis}<div id='question-list'>{question_cards}</div>"
            if question_cards
            else "<p class='muted'>Ei muita valmisteltuja kysymyksiä.</p>"
        )
        + "<details id='asiat'><summary>Asiat ja rekisterikirjaukset</summary>"
        + f"<div id='matter-list'>{matter_cards}</div>"
        + "<p><a href='oulaskangas.html'>Oulaskangas-dossier</a></p></details>"
        + (
            "<details id='reitit'><summary>Dokumentaariset reitti-indeksit (toissijainen)</summary>"
            f"<div id='route-list'>{route_cards}</div></details>"
            if route_cards
            else ""
        )
        + "<details><summary>Tietoja ja korjaus</summary>"
        + f"<p class='muted'>Julkaisija: {_e(publisher)} · {_e(checked_on)}. "
        f"<a href='{_e(correction_url)}'>Ilmoita korjaus</a> · <a href='{_e(repo_url)}'>Koodi</a>.</p>"
        "<p>Vastauksen lähdearvio ja kaavion projektioarvio ovat erillisiä. "
        "Kumpikaan ei ole ihmisen varmennus. Kirjatut teot eivät yksin osoita "
        "henkilökohtaista vaikutusta päätöksiin. Haku ei keksi valmisteltua vastausta. "
        "</p></details>"
        "<details><summary>Tekniset tiedot</summary>"
        + f"<p class='muted'>Julkaisupäivä {_e(checked_on)} · <a href='{_e(repo_url)}'>koodi</a>. "
        "human_certification=false.</p></details>"
        + "<style>article[hidden]{display:none}details{margin:.35rem 0}"
        "blockquote{white-space:pre-wrap}.question-list{padding-left:1.1rem}"
        ".question-list li{margin:.4rem 0}"
        ".matter-row{display:flex;gap:.5rem;align-items:baseline}.matter-row .muted{font-size:.9em}</style>",
        description="Kysymysmatkat, valmistellut vastaukset sekä kirjatut henkilökohtaiset teot lähteineen.",
    )
    (output_dir / "index.html").write_text(index, encoding="utf-8")

    content_rels = [
        "index.html",
        "lookup.html",
        "lookup.json",
        "oulaskangas.html",
        "oulaskangas.json",
        *[
            name
            for entry in matter_entries
            for name in (entry["path"], entry["json_path"])
        ],
        *[
            name
            for entry in matter_entries
            for name in entry.get("people_paths") or []
        ],
        *[
            rel[:-5] + ".json"
            for entry in matter_entries
            for rel in entry.get("people_paths") or []
            if rel.endswith(".html")
        ],
        *[
            name
            for entry in matter_entries
            for name in entry.get("record_paths") or []
        ],
        *[
            name
            for item in question_index
            for name in (item["path"], item["json_path"])
        ],
        *[
            name
            for item in route_index
            for name in (item["path"], item["json_path"])
        ],
        *[
            name
            for item in journey_index
            for name in (item["path"], item["json_path"])
        ],
    ]
    file_hashes = {
        rel: _file_sha(output_dir / rel)
        for rel in content_rels
        if (output_dir / rel).is_file()
    }
    expected = set(content_rels) | {"manifest.json"}
    actual = {path.relative_to(output_dir).as_posix() for path in output_dir.rglob("*") if path.is_file()}
    unexpected = sorted(actual - expected)
    if unexpected:
        raise BootstrapSiteError(
            "unexpected files in artifact after build: " + ", ".join(unexpected[:20])
        )
    missing = sorted(set(content_rels) - actual)
    if missing:
        raise BootstrapSiteError("missing expected artifact files: " + ", ".join(missing[:20]))

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "title": title,
        "admission": "CANDIDATE_PREVIEW_NOT_REFERENCE_JUDGED",
        "publisher": publisher,
        "review_date": checked_on,
        "repo_url": repo_url,
        "correction_url": correction_url,
        "matter_count": len(matter_entries),
        "question_count": len(question_index),
        "route_count": len(route_index),
        "journey_count": len(journey_index),
        "projection_reviewed_journey_count": sum(
            1 for item in journey_index if item.get("projection_reviewed")
        ),
        "oulaskangas_dossier": "oulaskangas.html",
        "oulaskangas_dossier_json": "oulaskangas.json",
        "lookup": "lookup.html",
        "lookup_json": "lookup.json",
        "matters": matter_entries,
        "questions": question_index,
        "routes": route_index,
        "journeys": journey_index,
        "file_sha256": file_hashes,
        "file_count": len(file_hashes),
        "method": (
            "Automated source-bound candidate reading; optional independent AI "
            "source-answer and separate journey-projection review receipts bound to "
            "exact candidate/mapping/bank hashes; prose review does not admit graphs; "
            "not human-validated acceptance and not automatic ADMITTED."
        ),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def bootstrap_site_cli(
    *,
    output: Path,
    matters: Sequence[Path],
    questions: Sequence[Path],
    oulaskangas_group: Sequence[Path] | None = None,
    oulaskangas_cost: Sequence[Path] | None = None,
    oulaskangas_monitoring: Sequence[Path] | None = None,
    editorial_packet: Path | None = None,
    editorial_overview: Path | None = None,
    source_banks: Sequence[Path] | None = None,
    route_dirs: Sequence[Path] | None = None,
    reviews: Sequence[Path] | None = None,
    reviewed_candidates: Mapping[str, Path] | None = None,
    journeys: Sequence[Path] | None = None,
    title: str = DEFAULT_TITLE,
) -> dict[str, Any]:
    return build_bootstrap_artifact(
        output_dir=output,
        matter_packets=list(matters),
        question_packets=list(questions),
        oulaskangas_group=list(oulaskangas_group or []),
        oulaskangas_cost=list(oulaskangas_cost or []),
        oulaskangas_monitoring=list(oulaskangas_monitoring or []),
        editorial_packet_path=editorial_packet,
        editorial_overview_path=editorial_overview,
        source_bank_paths=list(source_banks or []),
        route_dirs=list(route_dirs or []),
        review_paths=list(reviews or []),
        reviewed_candidate_paths=reviewed_candidates,
        journey_packets=list(journeys or []),
        title=title,
    )


__all__ = [
    "DEFAULT_TITLE",
    "SCHEMA_VERSION",
    "BootstrapSiteError",
    "bootstrap_site_cli",
    "build_bootstrap_artifact",
    "build_oulaskangas_dossier",
    "build_oulaskangas_dossier_packet",
]

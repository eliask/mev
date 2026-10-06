"""Source-bound legal follow-up for a reviewed documentary inquiry.

The inquiry packet and a LawVM capture answer different questions.  The former
contains the reviewed documentary comparison (for example, what a committee
proposal says); the latter is a versioned text reconstruction and comparison.
This module joins them without treating the LawVM comparison as legal truth.

The only affirmative legal statement made here is a narrow source statement:
the captured amending act contains the relevant provision and its own
commencement text gives a latest-start date.  Whether the provision was
operative at an earlier ``as_of`` date, how it was implemented, and what
effects it had remain explicit unknowns.
"""


import copy
import hashlib
import json
import re
from collections.abc import Mapping
from datetime import date
from typing import Any

from lxml import etree

from paa.inquiry_cases import InquiryError, anchor, source_record, validate_case
from paa.legal_state import (
    LegalStateError,
    build_legal_state_receipt_from_capture,
    validate_legal_state_receipt,
)

SCHEMA_VERSION = "paa.legal_inquiry.v1"
_FINLEX_SOURCE = re.compile(r"^finlex://sd/(?P<year>\d{4})/(?P<number>\d+)/fin/main\.xml$")
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class LegalInquiryError(ValueError):
    """Raised when a legal follow-up cannot retain its source boundary."""


def _text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _captured_string(value: Any) -> str:
    """Return captured text without trimming bytes that hashes bind."""

    return value if isinstance(value, str) else ""


def _normalise_xml_text(value: str) -> str:
    return " ".join(value.split())


def _compact_xml_token(value: str) -> str:
    return "".join(character.casefold() for character in value if character.isalnum())


def _raw_xml_surfaces(raw_text: str, legal_address: str) -> dict[str, str]:
    """Recreate the three evidence views from the captured XML itself.

    The packet stores display views separately because the operation-source
    quote and the selected provision use different text lanes. Validation
    nevertheless regenerates both from the raw document and the same
    chapter/section address rather than trusting copied hashes.
    """

    address_match = re.fullmatch(r"chapter:(?P<chapter>[^/]+)/section:(?P<section>[^/]+)", _text(legal_address))
    if not address_match:
        raise LegalInquiryError("legal inquiry legal address is not a chapter/section address")
    chapter_token = _compact_xml_token(address_match.group("chapter"))
    section_token = _compact_xml_token(address_match.group("section"))
    try:
        parser = etree.XMLParser(resolve_entities=False, no_network=True, load_dtd=False)
        xml_root = etree.fromstring(raw_text.encode("utf-8"), parser)
    except (etree.XMLSyntaxError, UnicodeError) as exc:
        raise LegalInquiryError("captured Finlex raw XML cannot be parsed safely") from exc

    def surface(element: Any) -> str:
        return _normalise_xml_text("".join(element.itertext()))

    chapter_nodes = []
    for chapter in xml_root.xpath("//*[local-name()='chapter']"):
        number_nodes = chapter.xpath("./*[local-name()='num'][1]")
        number = _compact_xml_token(surface(number_nodes[0])) if number_nodes else ""
        if number == chapter_token or number.startswith(chapter_token + "luku"):
            chapter_nodes.append(chapter)
    # Some Finlex/AKN originals expose a single chapter container while
    # LawVM's resolved address uses the source statute's chapter number. In
    # that representation the section number is the authoritative selector.
    if not chapter_nodes:
        all_chapters = xml_root.xpath("//*[local-name()='chapter']")
        if len(all_chapters) == 1:
            chapter_nodes = all_chapters
    section_nodes = []
    for chapter in chapter_nodes:
        for section in chapter.xpath(".//*[local-name()='section']"):
            number_nodes = section.xpath("./*[local-name()='num'][1]")
            number = _compact_xml_token(surface(number_nodes[0])) if number_nodes else ""
            if number == section_token:
                section_nodes.append(section)
    if not section_nodes:
        raise LegalInquiryError("legal address does not resolve to a section in the captured raw XML")

    formulas = xml_root.xpath("//*[local-name()='preamble']//*[local-name()='formula']")
    entry_nodes = xml_root.xpath("//*[local-name()='hcontainer' and @name='entryIntoForce']")
    if not formulas or not entry_nodes:
        raise LegalInquiryError("captured raw XML lacks the required preamble or entry-into-force surface")
    operation = surface(formulas[0])
    enacting_prefix = _normalise_xml_text("Eduskunnan päätöksen mukaisesti")
    if operation.startswith(enacting_prefix):
        operation = operation[len(enacting_prefix):].strip()
    doc_number_nodes = xml_root.xpath("//*[local-name()='docNumber']")
    doc_title_nodes = xml_root.xpath("//*[local-name()='docTitle']")
    return {
        "operation": operation,
        "provision": surface(section_nodes[0]),
        "entry_into_force": surface(entry_nodes[0]),
        "document_identifier": surface(doc_number_nodes[0]) if doc_number_nodes else "",
        "document_title": surface(doc_title_nodes[0]) if doc_title_nodes else "",
    }


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _date_value(value: Any) -> str:
    candidate = _text(value)
    if not _ISO_DATE.fullmatch(candidate):
        return ""
    try:
        date.fromisoformat(candidate)
    except ValueError:
        return ""
    return candidate


def _finnish_date_phrase(value: str) -> str:
    months = (
        "tammikuuta", "helmikuuta", "maaliskuuta", "huhtikuuta", "toukokuuta", "kesäkuuta",
        "heinäkuuta", "elokuuta", "syyskuuta", "lokakuuta", "marraskuuta", "joulukuuta",
    )
    parsed = date.fromisoformat(value)
    return f"{parsed.day} päivänä {months[parsed.month - 1]} {parsed.year}"


def _official_url(locator: str) -> str:
    """Map a captured Finlex source locator to its official publication URL."""

    match = _FINLEX_SOURCE.fullmatch(locator)
    if not match:
        raise LegalInquiryError("the adopted-law source must have a Finlex source locator")
    year = match.group("year")
    number = int(match.group("number"))
    # Finlex's alkup path uses the four-digit year followed by a four-digit
    # zero-padded act number, e.g. 20200565 for Act 565/2020.
    return f"https://www.finlex.fi/fi/laki/alkup/{year}/{year}{number:04d}"


def _capture_context(capture: Mapping[str, Any]) -> dict[str, Any]:
    case = capture.get("case")
    if not isinstance(case, Mapping):
        raise LegalInquiryError("LawVM capture has no case context")
    context = case.get("documentary_context")
    if not isinstance(context, Mapping):
        raise LegalInquiryError("LawVM capture has no documentary context")
    return dict(context)


def _bind_context(packet: Mapping[str, Any], context: Mapping[str, Any]) -> None:
    packet_episode = _text(packet.get("episode_id"))
    source_episode = _text(packet.get("source_episode_id"))
    if packet_episode and source_episode and packet_episode != source_episode:
        raise LegalInquiryError("inquiry packet contains conflicting episode identities")
    packet_episode = packet_episode or source_episode
    capture_episode = _text(context.get("episode_id"))
    if not packet_episode or not capture_episode:
        raise LegalInquiryError("inquiry and legal capture require explicit episode identity")
    if packet_episode != capture_episode:
        raise LegalInquiryError("inquiry packet and legal capture name different episodes")
    packet_question = _text(packet.get("question_id"))
    capture_question = _text(context.get("question_id"))
    if packet_question and capture_question and packet_question != capture_question:
        raise LegalInquiryError("inquiry packet and legal capture name different questions")
    if not capture_question:
        raise LegalInquiryError("legal capture has no explicit question identity")
    capture_sources = context.get("source_ids")
    if not isinstance(capture_sources, list) or not capture_sources or not all(_text(item) for item in capture_sources):
        raise LegalInquiryError("legal capture has no declared documentary source identities")
    packet_sources = {
        _text(row.get("source_id"))
        for row in packet.get("sources", [])
        if isinstance(row, Mapping) and _text(row.get("source_id"))
    }
    if not packet_sources.intersection(capture_sources):
        raise LegalInquiryError("inquiry packet and legal capture have no shared documentary source identity")


def _replay_view(receipt: Mapping[str, Any]) -> dict[str, Any]:
    for row in receipt.get("source_views", []):
        if isinstance(row, Mapping) and row.get("plane") == "replay":
            return dict(row)
    raise LegalInquiryError("legal receipt has no replay source view")


def _replay_artifact(receipt: Mapping[str, Any]) -> dict[str, Any]:
    for row in receipt.get("source_artifacts", []):
        if isinstance(row, Mapping) and row.get("plane") == "replay":
            return dict(row)
    raise LegalInquiryError("legal receipt has no replay source artifact")


def _source_with_quote(
    *,
    source_id: str,
    text: str,
    raw_text: str,
    raw_sha256: str,
    url: str,
    locator: str,
    title: str,
    view_role: str,
) -> dict[str, Any]:
    if not text or not raw_text or not raw_sha256:
        raise LegalInquiryError(f"source view {view_role} is incomplete")
    source = source_record(
        source_id,
        text,
        url=url,
        locator=f"{locator}#{view_role}",
        raw_sha256=raw_sha256,
        title=title,
    )
    # Keep the byte-level artifact and view identity alongside the display
    # text.  ``inquiry_cases.validate_case`` rechecks both hashes.
    source.update({
        "raw_text": raw_text,
        "source_kind": "OFFICIAL_FINLEX_ACT",
        "source_locator": locator,
        "view_role": view_role,
        "view_raw_sha256": raw_sha256,
    })
    return source


def _evidence(source: Mapping[str, Any], *, quote_role: str, capture_locator: str,
              capture_source_id: str, captured_quote_hash: str | None = None,
              raw_quote_binding: Mapping[str, Any] | None = None,
              canonical_text_sha256: str | None = None) -> dict[str, Any]:
    ref = anchor(source, source["text"])
    observed_hash = _sha256(ref["quote"])
    if captured_quote_hash and observed_hash != captured_quote_hash:
        raise LegalInquiryError(f"{quote_role} quote hash does not match the captured source view")
    ref.update({
        "quote_sha256": observed_hash,
        "quote_role": quote_role,
        "capture_locator": capture_locator,
        "capture_source_id": capture_source_id,
        "evidence_basis": "EXACT_CAPTURED_SOURCE_VIEW",
    })
    if raw_quote_binding is not None:
        ref["raw_quote_binding"] = dict(raw_quote_binding)
    if canonical_text_sha256 is not None:
        ref["canonical_text_sha256"] = canonical_text_sha256
    return ref


def _claim_id(receipt: Mapping[str, Any], latest_start: str, source_ids: list[str]) -> str:
    identity = "|".join((
        _text(receipt.get("statute_id")),
        _text(receipt.get("legal_address")),
        _text(receipt.get("as_of")),
        latest_start,
        *source_ids,
    ))
    return "legal-inquiry-claim:" + _sha256(identity)[:24]


def _artifact_id(receipt: Mapping[str, Any]) -> str:
    return "legal-comparison:" + _sha256(_text(receipt.get("receipt_id")))[:24]


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _amending_statute_id(receipt: Mapping[str, Any], replay_view: Mapping[str, Any]) -> str:
    comparison = receipt.get("comparison")
    replay = comparison.get("replay") if isinstance(comparison, Mapping) else None
    amendment = replay.get("source_amendment") if isinstance(replay, Mapping) else None
    value = _text(amendment)
    if value:
        return value
    source_id = _text(replay_view.get("source_id"))
    marker = "statute_xml:"
    if marker in source_id:
        return source_id.split(marker, 1)[1].split(":", 1)[0]
    raise LegalInquiryError("legal receipt has no amending-statute identity")


def _build_sources_and_evidence(
    receipt: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str, dict[str, Any], dict[str, Any]]:
    view = _replay_view(receipt)
    artifact = _replay_artifact(receipt)
    raw_text = _captured_string(artifact.get("raw_text"))
    raw_sha256 = _text(artifact.get("raw_sha256"))
    locator = _text(artifact.get("locator") or view.get("locator"))
    base_source_id = _text(view.get("source_id") or artifact.get("source_id"))
    if not raw_text or not raw_sha256 or not locator or not base_source_id:
        raise LegalInquiryError("replay source artifact lacks raw bytes, hash, locator or identity")
    if _sha256(raw_text) != raw_sha256:
        raise LegalInquiryError("replay source artifact raw hash changed")
    if _text(view.get("raw_sha256")) != raw_sha256:
        raise LegalInquiryError("replay view and source artifact are not the same source version")
    url = _official_url(locator)
    amending_statute_id = _amending_statute_id(receipt, view)
    surfaces = _raw_xml_surfaces(raw_text, _text(receipt.get("legal_address")))
    title = _text(surfaces.get("document_title")) or _text(receipt.get("title")) or "Finlex-säädös"
    source_document_number = _text(surfaces.get("document_identifier"))
    document_identifier = amending_statute_id

    operation_quote = _captured_string(view.get("quote"))
    operation_hash = _text(view.get("quote_hash"))
    human_text = _captured_string(artifact.get("human_text"))
    human_hash = _text(artifact.get("human_text_sha256"))
    commencement = artifact.get("commencement")
    if not isinstance(commencement, Mapping):
        raise LegalInquiryError("replay source artifact has no commencement record")
    commencement_quote = _captured_string(commencement.get("quote"))
    latest_start = _date_value(commencement.get("date"))
    if not operation_quote or not operation_hash:
        raise LegalInquiryError("replay source view lacks exact operation wording")
    if _sha256(operation_quote) != operation_hash:
        raise LegalInquiryError("replay operation quote hash changed")
    if not human_text or not human_hash or _sha256(human_text) != human_hash:
        raise LegalInquiryError("replay selected provision text is not hash-bound")
    if not commencement_quote or not latest_start:
        raise LegalInquiryError("replay source artifact lacks an explicit latest-start date and quote")
    if "viimeistään" not in commencement_quote.lower():
        raise LegalInquiryError("commencement quote does not state a latest-start boundary")
    captured_views = {
        "operation": _normalise_xml_text(operation_quote),
        "provision": _normalise_xml_text(human_text),
        "entry_into_force": _normalise_xml_text(commencement_quote),
    }
    for view_kind, captured_text in captured_views.items():
        if captured_text != surfaces[view_kind]:
            raise LegalInquiryError(f"captured {view_kind} view differs from the raw XML surface")

    # The three views deliberately have separate source identities.  They all
    # retain the same raw source hash, while the exact quotation span and its
    # interpretation remain independently inspectable in the packet.
    source_specs = (
        ("operation", operation_quote, "ACT_AMENDMENT_SCOPE", "operation", "Muutoksen kohde"),
        ("provision", human_text, "OPERATIVE_PROVISION_WORDING", "provision", "15 a § -teksti"),
        ("commencement", commencement_quote, "COMMENCEMENT_LATEST_START", "entry_into_force", "Käyttöönoton takaraja"),
    )
    sources: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    for suffix, text, role, surface_kind, view_label in source_specs:
        normalised_quote = _normalise_xml_text(text)
        if not normalised_quote or normalised_quote != surfaces[surface_kind]:
            raise LegalInquiryError(f"{role} quote is not found in the captured raw XML surface")
        raw_quote_binding = {
            "basis": "NORMALIZED_XML_TEXT_SURFACE",
            "surface": surface_kind,
            "raw_sha256": raw_sha256,
        }
        canonical_text_sha256 = _sha256(normalised_quote)
        source_id = f"{base_source_id}:{suffix}"
        source = _source_with_quote(
            source_id=source_id,
            text=text,
            raw_text=raw_text,
            raw_sha256=raw_sha256,
            url=url,
            locator=locator,
            title=(
                f"Säädös {document_identifier} ({source_document_number}): {title} — {view_label}"
                if source_document_number else f"Säädös {document_identifier}: {title} — {view_label}"
            ),
            view_role=role,
        )
        source.update({
            "document_identifier": document_identifier,
            "document_identifier_kind": "FINLEX_ACT_YEAR_NUMBER",
            "view_label": view_label,
            "raw_quote_binding": raw_quote_binding,
            "canonical_text_sha256": canonical_text_sha256,
        })
        sources.append(source)
        evidence.append(_evidence(
            source,
            quote_role=role,
            capture_locator=locator,
            capture_source_id=base_source_id,
            captured_quote_hash=operation_hash if suffix == "operation" else None,
            raw_quote_binding=raw_quote_binding,
            canonical_text_sha256=canonical_text_sha256,
        ))
    return sources, evidence, latest_start, view, artifact


def _append_unknowns(packet: dict[str, Any], *, latest_start: str, receipt: Mapping[str, Any],
                     amending_statute_id: str) -> None:
    unknowns = packet.setdefault("unknowns", [])
    if not isinstance(unknowns, list):
        raise LegalInquiryError("inquiry packet unknowns must be a list")
    effective_date = ""
    artifacts = receipt.get("source_artifacts")
    if isinstance(artifacts, list):
        for artifact in artifacts:
            if isinstance(artifact, Mapping) and artifact.get("plane") == "replay":
                effective_date = _date_value(artifact.get("effective_date"))
                break
    for item in unknowns:
        if isinstance(item, Mapping) and "voimaan" in _text(item.get("question")).lower():
            item["missing_evidence"] = (
                f"Muutossäädös {amending_statute_id} ja sen lähde-XML dokumentoivat uuden tekstin "
                f"sekä säädöksen voimaantulon {effective_date or 'ilman lähteessä ilmoitettua päivää'}; "
                f"täsmällinen historiallinen operatiivinen tila {receipt.get('as_of')} ei ratkea tästä "
                "vertailusta, jonka konsolidoitu lähde on ajallisesti myöhempi."
            )
    additions = [
        {
            "question": "Oliko säännös täsmällisesti sovellettava jo ennen viimeistään ilmoitettua aloituspäivää?",
            "missing_evidence": (
                f"Säädösteksti ilmoittaa viimeistään-aloituspäiväksi {latest_start}; se ei yksin ratkaise "
                "aikaisemman soveltamisen historiallista oikeudellista tilaa. LawVM-vertailun ajallinen lähdevarmistus on estynyt."
            ),
        },
        {
            "question": "Miten säädöksen vaatimus toimeenpantiin käytännössä?",
            "missing_evidence": "Tarvitaan erilliset toimeenpano-, valvonta- ja palvelulähteet; laki- ja tekstivertailu ei niitä sisällä.",
        },
        {
            "question": "Mitä palvelu- tai henkilöstövaikutuksia vaatimuksella oli?",
            "missing_evidence": "Vaikutuksia ei päätellä säädöstekstistä tai LawVM:n vertailusta.",
        },
    ]
    existing = {(item.get("question"), item.get("missing_evidence")) for item in unknowns if isinstance(item, Mapping)}
    for item in additions:
        if (item["question"], item["missing_evidence"]) not in existing:
            unknowns.append(item)


def validate_legal_inquiry(packet: Mapping[str, Any]) -> bool:
    """Validate an augmented inquiry and its comparison-only legal receipt."""

    if not isinstance(packet, Mapping):
        raise LegalInquiryError("augmented inquiry must be an object")
    try:
        validate_case(packet)
    except InquiryError as exc:
        raise LegalInquiryError(f"base inquiry is invalid: {exc}") from exc
    inquiry = packet.get("legal_inquiry")
    if not isinstance(inquiry, Mapping) or inquiry.get("schema_version") != SCHEMA_VERSION:
        raise LegalInquiryError("legal inquiry schema is missing or unsupported")
    if inquiry.get("stage") != "ADOPTED_LAW_BOUNDED":
        raise LegalInquiryError("legal inquiry is not at the adopted-law bounded stage")
    if inquiry.get("status") != "DOCUMENTED_BOUNDED":
        raise LegalInquiryError("legal inquiry must remain a documented bounded statement")
    packet_episode = _text(packet.get("episode_id")) or _text(packet.get("source_episode_id"))
    packet_source_episode = _text(packet.get("source_episode_id"))
    if not packet_episode or (packet_source_episode and packet_source_episode != packet_episode):
        raise LegalInquiryError("inquiry packet requires one explicit episode identity")
    if inquiry.get("episode_id") != packet_episode:
        raise LegalInquiryError("legal inquiry episode identity is not bound to the packet")
    if not _text(inquiry.get("question_id")):
        raise LegalInquiryError("legal inquiry question identity is missing")
    if not _text(inquiry.get("statute_id")) or not _text(inquiry.get("amending_statute_id")) or not _text(inquiry.get("legal_address")):
        raise LegalInquiryError("legal inquiry legal identity is incomplete")
    latest_start = _date_value(inquiry.get("latest_start_date"))
    if not latest_start:
        raise LegalInquiryError("legal inquiry latest-start date is invalid")
    unknown_text = " ".join(
        _text(row.get("missing_evidence")) if isinstance(row, Mapping) else _text(row)
        for row in inquiry.get("unknowns", [])
    ).lower()
    if not inquiry.get("unknowns") or not any(
        marker in unknown_text for marker in ("aikaisemman", "historical", "earlier")
    ):
        raise LegalInquiryError("legal inquiry must preserve the earlier-operativity unknown")

    sources = {row.get("source_id"): row for row in packet.get("sources", []) if isinstance(row, Mapping)}
    evidence = {row.get("evidence_id"): row for row in packet.get("evidence", []) if isinstance(row, Mapping)}
    source_ids = inquiry.get("source_ids")
    evidence_ids = inquiry.get("evidence_ids")
    if not isinstance(source_ids, list) or not source_ids or len(set(source_ids)) != len(source_ids) or not set(source_ids) <= sources.keys():
        raise LegalInquiryError("legal inquiry has dangling source references")
    if not isinstance(evidence_ids, list) or not evidence_ids or len(set(evidence_ids)) != len(evidence_ids) or not set(evidence_ids) <= evidence.keys():
        raise LegalInquiryError("legal inquiry has dangling evidence references")
    artifacts = packet.get("legal_comparison_artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != 1:
        raise LegalInquiryError("exactly one LawVM comparison artifact is required")
    artifact = artifacts[0]
    if not isinstance(artifact, Mapping) or artifact.get("artifact_id") != inquiry.get("comparison_artifact_id"):
        raise LegalInquiryError("comparison artifact identity is not bound")
    if artifact.get("role") != "COMPARISON_ONLY_NOT_LEGAL_TRUTH":
        raise LegalInquiryError("LawVM artifact must be comparison-only")
    receipt = artifact.get("receipt")
    try:
        validate_legal_state_receipt(receipt)
    except (LegalStateError, TypeError) as exc:
        raise LegalInquiryError(f"comparison artifact is invalid: {exc}") from exc
    if receipt.get("operative", {}).get("verified") is not False:
        raise LegalInquiryError("comparison artifact must not certify operative law")
    if artifact.get("receipt_id") != receipt.get("receipt_id"):
        raise LegalInquiryError("comparison artifact receipt identity is not bound")
    if artifact.get("artifact_id") != _artifact_id(receipt):
        raise LegalInquiryError("comparison artifact identity does not match its receipt")
    if inquiry.get("comparison_artifact_id") not in inquiry.get("comparison_artifact_ids", [inquiry.get("comparison_artifact_id")]):
        raise LegalInquiryError("comparison artifact reference is incomplete")

    replay_view = _replay_view(receipt)
    replay_artifact = _replay_artifact(receipt)
    raw_text = _captured_string(replay_artifact.get("raw_text"))
    raw_sha256 = _text(replay_artifact.get("raw_sha256"))
    locator = _text(replay_artifact.get("locator") or replay_view.get("locator"))
    base_source_id = _text(replay_view.get("source_id") or replay_artifact.get("source_id"))
    if not raw_text or not raw_sha256 or _sha256(raw_text) != raw_sha256:
        raise LegalInquiryError("comparison receipt raw source is not hash-bound")
    if _text(replay_view.get("raw_sha256")) != raw_sha256:
        raise LegalInquiryError("comparison receipt view and artifact use different raw source versions")
    if _text(inquiry.get("statute_id")) != _text(receipt.get("statute_id")):
        raise LegalInquiryError("legal inquiry statute identity differs from the comparison receipt")
    if _text(inquiry.get("legal_address")) != _text(receipt.get("legal_address")):
        raise LegalInquiryError("legal inquiry address differs from the comparison receipt")
    if _text(inquiry.get("as_of")) != _text(receipt.get("as_of")):
        raise LegalInquiryError("legal inquiry as-of date differs from the comparison receipt")
    amending_statute_id = _amending_statute_id(receipt, replay_view)
    if _text(inquiry.get("amending_statute_id")) != amending_statute_id:
        raise LegalInquiryError("legal inquiry amending-act identity differs from the comparison receipt")
    surfaces = _raw_xml_surfaces(raw_text, _text(receipt.get("legal_address")))
    expected_url = _official_url(locator)
    expected_views = {
        "ACT_AMENDMENT_SCOPE": ("operation", "operation", "Muutoksen kohde"),
        "OPERATIVE_PROVISION_WORDING": ("provision", "provision", "15 a § -teksti"),
        "COMMENCEMENT_LATEST_START": ("entry_into_force", "commencement", "Käyttöönoton takaraja"),
    }
    expected_source_ids = {f"{base_source_id}:{suffix}" for suffix in ("operation", "provision", "commencement")}
    if set(source_ids) != expected_source_ids:
        raise LegalInquiryError("legal inquiry source set is not the three bound Finlex views")
    if set(artifact.get("source_ids", [])) != {row.get("source_id") for row in receipt.get("source_views", [])}:
        raise LegalInquiryError("comparison artifact source identities are not bound")
    source_by_role = {}
    for source_id in source_ids:
        source = sources[source_id]
        role = _text(source.get("view_role"))
        if role in source_by_role:
            raise LegalInquiryError("duplicate legal source-view role")
        source_by_role[role] = source
        binding = source.get("raw_quote_binding")
        if not isinstance(binding, Mapping) or binding.get("raw_sha256") != raw_sha256:
            raise LegalInquiryError("legal inquiry source lacks a raw XML binding")
        if _captured_string(source.get("raw_text")) != raw_text or _text(source.get("source_locator")) != locator:
            raise LegalInquiryError("legal inquiry source does not preserve the receipt raw artifact")
        if source.get("source_kind") != "OFFICIAL_FINLEX_ACT" or source.get("document_identifier") != amending_statute_id:
            raise LegalInquiryError("legal source is not identified as the captured amending act")
        if source.get("url") != expected_url:
            raise LegalInquiryError("legal source URL does not identify the official act")
        if source.get("raw_sha256") != raw_sha256 or _sha256(_captured_string(source.get("raw_text"))) != source.get("raw_sha256"):
            raise LegalInquiryError("legal source raw hash changed")
        if _sha256(_captured_string(source.get("text"))) != source.get("text_sha256"):
            raise LegalInquiryError("legal source display hash changed")
    if set(source_by_role) != set(expected_views):
        raise LegalInquiryError("legal inquiry is missing one of the three source-view roles")
    for role, (surface_key, suffix, view_label) in expected_views.items():
        source = source_by_role[role]
        if source.get("source_id") != f"{base_source_id}:{suffix}":
            raise LegalInquiryError("legal source-view identity is not stable")
        if source.get("view_label") != view_label or source.get("raw_quote_binding", {}).get("surface") != surface_key:
            raise LegalInquiryError("legal source-view label or surface binding changed")
        canonical_text = _normalise_xml_text(_captured_string(source.get("text")))
        if canonical_text != surfaces[surface_key]:
            raise LegalInquiryError("legal source display text differs from regenerated raw XML")
        canonical_hash = _sha256(canonical_text)
        if source.get("canonical_text_sha256") != canonical_hash:
            raise LegalInquiryError("legal source canonical text hash changed")
        if source.get("raw_quote_binding", {}).get("raw_sha256") != raw_sha256:
            raise LegalInquiryError("legal source canonical binding changed")
    latest_start_phrase = _finnish_date_phrase(latest_start)
    if latest_start_phrase not in surfaces["entry_into_force"]:
        raise LegalInquiryError("legal inquiry latest-start date is absent from raw XML")
    commencement = replay_artifact.get("commencement")
    if not isinstance(commencement, Mapping) or _date_value(commencement.get("date")) != latest_start:
        raise LegalInquiryError("legal inquiry latest-start date differs from captured commencement metadata")
    refs_by_source = {evidence[evidence_id].get("source_id"): evidence[evidence_id] for evidence_id in evidence_ids}
    if set(refs_by_source) != set(source_ids):
        raise LegalInquiryError("legal inquiry evidence set is not one-to-one with source views")
    for source_id in source_ids:
        ref = refs_by_source[source_id]
        source = sources[source_id]
        binding = ref.get("raw_quote_binding")
        if not isinstance(binding, Mapping) or binding.get("raw_sha256") != raw_sha256:
            raise LegalInquiryError("legal inquiry evidence lacks a raw XML binding")
        if ref.get("quote") != source.get("text") or _sha256(_captured_string(ref.get("quote"))) != ref.get("quote_sha256"):
            raise LegalInquiryError("legal inquiry evidence quote hash changed")
        if ref.get("canonical_text_sha256") != source.get("canonical_text_sha256"):
            raise LegalInquiryError("legal inquiry evidence canonical hash changed")
    claim_id = _text(inquiry.get("claim_id"))
    expected_claim_id = _claim_id(receipt, latest_start, source_ids)
    if claim_id != expected_claim_id:
        raise LegalInquiryError("legal inquiry claim identity changed")
    claims = [row for row in packet.get("claims", []) if isinstance(row, Mapping)]
    claim = next((row for row in claims if row.get("claim_id") == claim_id), None)
    if claim is None or claim.get("state") != "SOURCE_REVIEWED":
        raise LegalInquiryError("legal inquiry claim is missing or not source reviewed")
    if set(claim.get("evidence_ids", [])) != set(evidence_ids):
        raise LegalInquiryError("legal inquiry claim and inquiry evidence differ")
    claim_text = _text(claim.get("text"))
    if (f"Muutossäädös {amending_statute_id}" not in claim_text
            or latest_start not in claim_text
            or "ei ratkaise aikaisempaa" not in claim_text):
        raise LegalInquiryError("legal inquiry claim wording is not the bounded source statement")
    documentary_source_ids = inquiry.get("documentary_source_ids")
    packet_source_ids = {row.get("source_id") for row in packet.get("sources", []) if isinstance(row, Mapping)}
    if not isinstance(documentary_source_ids, list) or not documentary_source_ids:
        raise LegalInquiryError("legal inquiry documentary source context is missing")
    if not set(documentary_source_ids).intersection(packet_source_ids):
        raise LegalInquiryError("legal inquiry documentary source context is not bound to the packet")
    return True


def augment_rai_inquiry(packet: Mapping[str, Any], capture: Mapping[str, Any]) -> dict[str, Any]:
    """Attach a bounded adopted-law finding and full comparison receipt.

    ``packet`` must already be a source-reviewed packet accepted by
    :func:`paa.inquiry_cases.validate_case`.  ``capture`` is the persisted
    three-plane LawVM capture.  The returned mapping is a deep copy; neither
    input is mutated.
    """

    if not isinstance(packet, Mapping) or not isinstance(capture, Mapping):
        raise LegalInquiryError("packet and capture must be JSON objects")
    try:
        validate_case(packet)
    except InquiryError as exc:
        raise LegalInquiryError(f"base inquiry is invalid: {exc}") from exc
    context = _capture_context(capture)
    _bind_context(packet, context)
    receipt = build_legal_state_receipt_from_capture(capture)
    validate_legal_state_receipt(receipt)
    # Validate the incoming capture's raw XML surfaces before checking for an
    # existing augmentation. A receipt ID is intentionally derived from the
    # LawVM comparison planes, so a changed commencement annotation can keep
    # the same ID while still invalidating this legal follow-up.
    sources, evidence, latest_start, replay_view, replay_artifact = _build_sources_and_evidence(receipt)
    amending_statute_id = _amending_statute_id(receipt, replay_view)

    result = copy.deepcopy(dict(packet))
    if result.get("legal_inquiry") or result.get("legal_comparison_artifacts"):
        # Make repeated compilation safe while refusing to silently replace a
        # previously bound capture with a different one.
        if result.get("legal_inquiry", {}).get("comparison_receipt_id") != receipt.get("receipt_id"):
            raise LegalInquiryError("inquiry is already bound to a different legal capture")
        existing_artifacts = result.get("legal_comparison_artifacts")
        existing_receipt = existing_artifacts[0].get("receipt") if isinstance(existing_artifacts, list) and existing_artifacts and isinstance(existing_artifacts[0], Mapping) else None
        if _canonical_json(existing_receipt) != _canonical_json(receipt):
            raise LegalInquiryError("incoming legal capture differs from the packet's bound receipt")
        validate_legal_inquiry(result)
        return result

    result.setdefault("sources", []).extend(sources)
    result.setdefault("evidence", []).extend(evidence)
    _append_unknowns(
        result,
        latest_start=latest_start,
        receipt=receipt,
        amending_statute_id=amending_statute_id,
    )

    source_ids = [row["source_id"] for row in sources]
    evidence_ids = [row["evidence_id"] for row in evidence]
    claim_id = _claim_id(receipt, latest_start, source_ids)
    claim = {
        "claim_id": claim_id,
        "dimension": "Säädös hyväksytty ja viimeinen aloitusajankohta",
        "text": (
            f"Muutossäädös {amending_statute_id} lisää lähdekatkelman mukaan tarkastellun "
            f"pykälän, ja sen voimaantuloteksti asettaa käytön viimeistään aloitettavaksi "
            f"{latest_start}. Tämä dokumentoi säädöksen sisällön ja viimeisen aloitusajankohdan; "
            "se ei ratkaise aikaisempaa operatiivista soveltamista, toimeenpanoa tai vaikutuksia."
        ),
        "state": "SOURCE_REVIEWED",
        "evidence_ids": evidence_ids,
        "legal_inquiry_stage": "ADOPTED_LAW_BOUNDED",
        "review": {
            "reviewer": "CODEX_SOURCE_READING_2026-10-07",
            "method": "AI_SOURCE_READING",
            "rationale": (
                "An automated XML surface check and source reading bind the captured Finlex operation wording, "
                "selected provision text and explicit commencement sentence separately. This is AI-assisted "
                "source reading, not independent human adjudication. The LawVM reconstruction is retained as a "
                "comparison artifact, not as an operative-law determination."
            ),
            "source_versions": {row["source_id"]: row["text_sha256"] for row in sources},
        },
    }
    result["claims"].append(claim)

    artifact_id = _artifact_id(receipt)
    comparison_artifact = {
        "artifact_id": artifact_id,
        "artifact_kind": "LAWVM_RECONSTRUCTION_COMPARISON",
        "role": "COMPARISON_ONLY_NOT_LEGAL_TRUTH",
        "receipt_id": receipt["receipt_id"],
        "receipt": receipt,
        "comparison_state": receipt["comparison"]["state"],
        "temporal_state": receipt["comparison"]["temporal"]["status"],
        "source_ids": [row.get("source_id") for row in receipt.get("source_views", [])],
    }
    result["legal_comparison_artifacts"] = [comparison_artifact]
    result["legal_inquiry"] = {
        "schema_version": SCHEMA_VERSION,
        "stage": "ADOPTED_LAW_BOUNDED",
        "status": "DOCUMENTED_BOUNDED",
        "episode_id": context.get("episode_id"),
        "question_id": context.get("question_id"),
        "documentary_source_ids": list(context.get("source_ids") or []),
        "statute_id": receipt["statute_id"],
        "amending_statute_id": amending_statute_id,
        "legal_address": receipt["legal_address"],
        "as_of": receipt["as_of"],
        "latest_start_date": latest_start,
        "source_ids": source_ids,
        "evidence_ids": evidence_ids,
        "claim_id": claim_id,
        "comparison_artifact_id": artifact_id,
        "comparison_artifact_ids": [artifact_id],
        "comparison_receipt_id": receipt["receipt_id"],
        "source_review": {
            "basis": "OFFICIAL_FINLEX_ACT_SOURCE_AND_EXPLICIT_COMMENCEMENT_TEXT",
            "operation_source_id": replay_view.get("source_id"),
            "operation_source_sha256": replay_view.get("raw_sha256"),
            "artifact_locator": replay_artifact.get("locator"),
        },
        "unknowns": [
            "Historical operative scope before the latest-start date is not decided by this packet.",
            "LawVM's future consolidated oracle is not a temporally valid historical comparison for the as-of date.",
            "Implementation, service outcomes and causal effects are outside this source slice.",
        ],
    }
    validate_legal_inquiry(result)
    return result


# Descriptive aliases make the generic boundary discoverable to the compiler
# without creating alternate semantics.
build_rai_legal_inquiry = augment_rai_inquiry
augment_inquiry_with_legal_receipt = augment_rai_inquiry


__all__ = [
    "SCHEMA_VERSION",
    "LegalInquiryError",
    "augment_inquiry_with_legal_receipt",
    "augment_rai_inquiry",
    "build_rai_legal_inquiry",
    "validate_legal_inquiry",
]

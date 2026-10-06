"""Acquire and normalize official parliamentary initiative records.

The Vaski open-data endpoint exposes the same matter in more than one
document row.  A legislative-initiative row carries the submitted text and
signatories; a ``KasittelytiedotValtiopaivaasia`` row carries the later
procedural disposition.  This module keeps those rows separate until the
explicit :func:`merge_vaski_records` step, so that an object never loses the
source record that supports a status or an author claim.

The adapter is deliberately independent of the SQLite store.  Callers can
write the returned ``object``/``coverage``/``evidence`` records to whatever
store or trace compiler owns the current schema.  This also makes the parser
replayable from a frozen XML slice without network access.
"""


import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import httpx
from lxml import etree

from paa.config import EDUSKUNTA_API, RAW, USER_AGENT, ensure_dirs

SOURCE_ID = "SRC-EDUSKUNTA-VASKI"
VASKI_ROWS_URL = f"{EDUSKUNTA_API}/VaskiData/rows"

_DATE = re.compile(r"^(?P<year>\d{4})-(?P<month>\d{1,2})-(?P<day>\d{1,2})")
_FINNISH_DATE = re.compile(r"^(?P<day>\d{1,2})[.]\s*(?P<month>\d{1,2})[.]\s*(?P<year>\d{4})")
_MATTER = re.compile(r"^[A-ZÅÄÖ]+\s+\d+/\d{4}\s+vp$", re.IGNORECASE)

_KIND_BY_PREFIX = {
    "LA": "LEGISLATIVE_INITIATIVE",
    "TPA": "PARLIAMENTARY_INITIATIVE",
    "KAA": "CITIZEN_INITIATIVE",
    "KK": "WRITTEN_QUESTION",
    "K": "GOVERNMENT_PROPOSAL",
    "HE": "GOVERNMENT_PROPOSAL",
    "P": "PRESIDENTIAL_PROPOSAL",
    "U": "EU_DOCUMENT",
}

# These are text-bearing Vaski elements.  We avoid ``itertext`` over the
# whole document because it would also pull transfer metadata, namespace
# identifiers, and the signature list into the initiative text.
_TEXT_ELEMENTS = {
    "NimekeTeksti",
    "OtsikkoTeksti",
    "KappaleKooste",
    "MomenttiKooste",
    "PykalaTunnusKooste",
    "PykalaNimekeKooste",
    "SaadosNimekeKooste",
    "FraasiKappaleKooste",
    "EduskuntakasittelyPaatosKuvaus",
}


def _local(element: etree._Element) -> str:
    """Return an XML local name, making namespace changes harmless."""

    return etree.QName(element).localname if isinstance(element.tag, str) else ""


def _attrs(element: etree._Element) -> dict[str, str]:
    return {key.rsplit("}", 1)[-1]: str(value).strip() for key, value in element.attrib.items()}


def _clean(value: object) -> str:
    return " ".join(str(value or "").split())


def _element_text(element: etree._Element) -> str:
    return _clean(" ".join(element.itertext()))


def _date(value: object) -> str | None:
    text = _clean(value)
    if not text:
        return None
    match = _DATE.search(text)
    if match:
        return "{year:04d}-{month:02d}-{day:02d}".format(
            year=int(match.group("year")),
            month=int(match.group("month")),
            day=int(match.group("day")),
        )
    match = _FINNISH_DATE.search(text)
    if match:
        return "{year:04d}-{month:02d}-{day:02d}".format(
            year=int(match.group("year")),
            month=int(match.group("month")),
            day=int(match.group("day")),
        )
    return None


def _attribute(element: etree._Element, *names: str) -> str:
    attributes = _attrs(element)
    for name in names:
        if attributes.get(name):
            return attributes[name]
    return ""


def _first_text(root: etree._Element, names: set[str]) -> str:
    for element in root.iter():
        if _local(element) in names:
            value = _element_text(element)
            if value:
                return value
    return ""


def _matter_id(root: etree._Element) -> str:
    for element in root.iter():
        candidate = _attribute(element, "eduskuntaTunnus", "EduskuntaTunnus")
        if candidate and _MATTER.match(candidate):
            return candidate
        if _local(element) in {"EduskuntaTunnus", "EduskuntaAsiaTunnus"}:
            candidate = _element_text(element)
            if candidate and _MATTER.match(candidate):
                return candidate
    return ""


def _document_type(root: etree._Element, matter_id: str) -> tuple[str, str]:
    document_type = ""
    for element in root.iter():
        document_type = _attribute(element, "asiakirjatyyppiNimi", "asiakirjatyyppi")
        if document_type:
            break
    prefix = matter_id.split(" ", 1)[0].upper() if matter_id else ""
    return document_type, _KIND_BY_PREFIX.get(prefix, "OFFICIAL_PARLIAMENTARY_OBJECT")


def _metadata(root: etree._Element) -> etree._Element | None:
    for element in root.iter():
        if _local(element) in {"JulkaisuMetatieto", "KasittelytiedotValtiopaivaasia"}:
            return element
    return None


def _content_root(root: etree._Element) -> etree._Element | None:
    # The content root is intentionally selected by structure, not by a
    # namespace prefix.  Vaski has changed prefixes over time.
    for element in root.iter():
        if _local(element) in {
            "Lakialoite",
            "Toimenpidealoite",
            "Kansalaisaloite",
            "KirjallinenKysymys",
            "SuullinenKysymys",
            "Hallitusohjelma",
        }:
            return element
    return None


def _source_url(matter_id: str) -> str:
    query = urlencode(
        {
            "columnName": "Eduskuntatunnus",
            "columnValue": matter_id,
            "perPage": 100,
            "page": 0,
        }
    )
    return f"{VASKI_ROWS_URL}?{query}"


def _record_class(root: etree._Element) -> str:
    names = {_local(element) for element in root.iter()}
    if "KasittelytiedotValtiopaivaasia" in names:
        return "PROCEDURAL"
    # The registry also publishes lightweight LegislativeMotion transfer
    # records.  They carry an identifier/title but no actual Lakialoite (or
    # equivalent) content root.  Keep these rows visible, but never let a
    # shell win over a substantive content row when the same matter has both.
    if _content_root(root) is None:
        return "CONTENT_REFERENCE"
    return "CONTENT"


def _record_quality(root: etree._Element) -> str:
    """Classify whether the row contains the submitted document body."""

    if _record_class(root) == "PROCEDURAL":
        return "PROCEDURAL"
    return "SUBSTANTIVE" if _content_root(root) is not None else "REFERENCE_SHELL"


def _action_date(root: etree._Element) -> tuple[str | None, str | None]:
    """Return a source-supported action date and its basis.

    ``laadintaPvm`` is a document/publication metadata date.  It is retained
    separately as ``date``/``publication_date`` and must not silently become
    the date on which a person filed an initiative.  For content rows the
    signature date records signing, not formal filing.  For procedural
    rows, prefer the explicit ``Vireilletulo``/``Jätetty`` event date.
    """

    # Signature date on submitted initiative content.
    for element in root.iter():
        value = _attribute(element, "allekirjoitusPvm", "allekirjoitusPvmTeksti")
        if value:
            parsed = _date(value)
            if parsed:
                return parsed, "SIGNATURE_DATE"
        if _local(element) == "PaivaysKooste":
            parsed = _date(_element_text(element))
            if parsed:
                return parsed, "SIGNATURE_DATE"

    # Processing rows expose event dates.  Select the filing event, not a
    # later committee/plenary event and not the metadata publication date.
    events: list[tuple[str, str]] = []
    for element in root.iter():
        if _local(element) != "ToimenpideJulkaisu":
            continue
        date = _date(_attribute(element, "tapahtumaPvm"))
        if not date:
            for child in element.iter():
                if _local(child) == "TapahtumaPvmTeksti":
                    date = _date(_element_text(child))
                    if date:
                        break
        if not date:
            continue
        code = _attribute(element, "kasittelyvaiheKoodi").casefold()
        text = _element_text(element).casefold()
        if code in {"vir", "jatto", "jätto", "jätetty"} or "vireilletulo" in text or "jätetty" in text:
            events.append((date, "VIREILLETULO_EVENT"))
    if events:
        return min(events, key=lambda item: item[0])
    return None, None


def _person_name(person: etree._Element) -> str:
    first = _first_text(person, {"EtuNimi", "Etunimi"})
    last = _first_text(person, {"SukuNimi", "Sukunimi"})
    if first or last:
        return _clean(f"{first} {last}")
    return _first_text(person, {"Nimi", "NimiTeksti"})


def _authors(root: etree._Element, evidence_id: str) -> list[dict[str, Any]]:
    """Read first signer/author and co-signers without inventing identity."""

    found: list[dict[str, Any]] = []
    seen: dict[str, int] = {}

    def add(person: etree._Element, role: str) -> None:
        person_id = _attribute(person, "muuTunnus", "personId", "henkiloTunnus")
        name = _person_name(person)
        key = person_id or name.casefold()
        if not key or key in seen:
            if key in seen and role == "AUTHOR":
                found[seen[key]]["role"] = "AUTHOR"
            return
        seen[key] = len(found)
        found.append(
            {
                "person_id": person_id or None,
                "name": name,
                "role": role,
                "evidence_ids": [evidence_id],
            }
        )

    # Signature order is the official order.  The first signer is the author
    # for initiative records; following entries are co-signers.
    signatures = [element for element in root.iter() if _local(element) == "Allekirjoittaja"]
    for index, signature in enumerate(signatures):
        role_code = _attribute(signature, "allekirjoitusLuokitusKoodi")
        role = "AUTHOR" if index == 0 or role_code.lower().startswith("ensimm") else "COSIGNER"
        person = next((child for child in signature.iter() if _local(child) == "Henkilo"), None)
        if person is not None:
            add(person, role)

    if found:
        return found

    # Some older or procedural records expose only a Toimija/Henkilo pair.
    for actor in root.iter():
        if _local(actor) != "Toimija":
            continue
        role_code = _attribute(actor, "rooliNimi", "rooli", "role")
        role = "AUTHOR" if not role_code or "laatija" in role_code.casefold() else "COSIGNER"
        person = next((child for child in actor.iter() if _local(child) == "Henkilo"), None)
        if person is not None:
            add(person, role)

    # Last fallback for handling records: the stable Henkilo ID is still
    # useful, but it is deliberately not joined to a political actor here.
    if not found:
        person = next((element for element in root.iter() if _local(element) == "Henkilo"), None)
        if person is not None:
            add(person, "AUTHOR")
    return found


def _text(root: etree._Element) -> str:
    content = _content_root(root)
    if content is None:
        content = root
    values: list[str] = []
    seen: set[str] = set()
    for element in content.iter():
        if _local(element) not in _TEXT_ELEMENTS:
            continue
        value = _element_text(element)
        if value and value not in seen:
            values.append(value)
            seen.add(value)
    return "\n\n".join(values)


def _disposition(root: etree._Element, evidence_id: str) -> dict[str, Any] | None:
    metadata = _metadata(root)
    date = None
    # A current Vaski procedural payload commonly contains both a publication
    # metadata block and a later Kasittelytiedot block.  The first block has
    # the submission date; only the latter carries the closing date.
    for element in root.iter():
        candidate = _attribute(element, "paattymisPvm", "paatosPvm")
        if candidate:
            date = _date(candidate)
            if date:
                break
    raw_state = ""
    code = ""
    for element in root.iter():
        if _local(element) == "EduskuntakasittelyPaatosKuvaus":
            raw_state = _element_text(element)
            code = _attribute(element, "eduskuntakasittelyPaatosKoodi")
            break
    if not raw_state:
        raw_state = _attribute(metadata, "viimeisinKasittelyvaiheKoodi") if metadata is not None else ""
    if not raw_state:
        raw_state = _attribute(metadata, "tilaKoodi") if metadata is not None else ""
    # Publication metadata may say only ``Valmis`` or ``Käsitelty``.  Those
    # are document lifecycle labels, not a parliamentary disposition.
    if not date and not code and raw_state.casefold() in {"valmis", "käsitelty", "kasitelty"}:
        return None
    if not raw_state and not date:
        return None

    lower = f"{raw_state} {code}".casefold()
    if "rauen" in lower or "expir" in lower:
        state = "EXPIRED"
    # Approval of a motion/initiative is not the same institutional event as
    # enactment of a law.  Keep the source vocabulary's distinction: callers
    # may only promote APPROVED to ENACTED when a separate law source proves
    # promulgation/entry into force.
    elif "hyväks" in lower or "hyvak" in lower or "approv" in lower:
        state = "APPROVED"
    elif "enact" in lower:
        state = "ENACTED"
    elif "hylät" in lower or "hylat" in lower or "reject" in lower:
        state = "REJECTED"
    elif "kesken" in lower or "pending" in lower:
        state = "PENDING"
    else:
        state = "UNRESOLVED"
    return {
        "state": state,
        "date": date,
        "raw_state": raw_state,
        "code": code or None,
        "evidence_ids": [evidence_id],
    }


def parse_vaski_xml(
    xml: str,
    *,
    source_url: str | None = None,
    row_metadata: Mapping[str, Any] | None = None,
    retrieved_at: str | None = None,
) -> dict[str, Any]:
    """Parse one official Vaski XML row into a provenance-bearing object.

    ``xml`` is hashed before parsing and the hash is carried on every source
    evidence record.  This means a later re-download cannot silently change
    the text behind a review packet.
    """

    if not isinstance(xml, str) or not xml.strip():
        raise ValueError("Vaski XML must be a non-empty string")
    raw_bytes = xml.encode("utf-8")
    raw_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    try:
        # Vaski XML is downloaded input.  The parser must not resolve an
        # external entity, fetch a DTD, or make a network request while a
        # frozen slice is being replayed.
        parser = etree.XMLParser(
            resolve_entities=False,
            no_network=True,
            load_dtd=False,
            recover=False,
        )
        root = etree.fromstring(raw_bytes, parser=parser)
    except etree.XMLSyntaxError as exc:
        raise ValueError(f"invalid Vaski XML: {exc}") from exc

    matter_id = _matter_id(root)
    if not matter_id:
        raise ValueError("Vaski XML has no Eduskunta matter identifier")
    document_type, kind = _document_type(root, matter_id)
    row = dict(row_metadata or {})
    record_id = str(row.get("Id") or row.get("id") or "").strip() or None
    record_locator = f"VaskiData/Id={record_id}" if record_id else f"VaskiData/{matter_id}"
    evidence_id = f"{SOURCE_ID}:{raw_sha256[:16]}:record"
    source_url = source_url or _source_url(matter_id)
    metadata = _metadata(root)
    publication_date = _date(
        _attribute(metadata, "laadintaPvm", "paivays", "Pvm") if metadata is not None else ""
    )
    action_date, action_date_basis = _action_date(root)
    title = _first_text(root, {"NimekeTeksti"})
    text = _text(root)
    authors = _authors(root, evidence_id)
    evidence = {
        "evidence_id": evidence_id,
        "source_id": SOURCE_ID,
        "source_url": source_url,
        "url": source_url,
        "record_locator": record_locator,
        "raw_sha256": raw_sha256,
        "byte_length": len(raw_bytes),
        # This is the source record's extracted text, not an LLM summary.  A
        # reader can therefore see exactly what the hash and locator cover;
        # relation-specific short quotes are created later by the review
        # layer.
        "quote": text or title or None,
        "retrieved_at": retrieved_at or datetime.now(UTC).replace(microsecond=0).isoformat(),
    }
    disposition = _disposition(root, evidence_id)
    record_class = _record_class(root)
    record_quality = _record_quality(root)
    return {
        "object_id": f"eduskunta:{matter_id}",
        "matter_id": matter_id,
        "kind": kind,
        "title": title,
        "text": text,
        # Keep the historical ``date`` field for compatibility, but make its
        # semantics explicit.  Consumers adjudicating a personal action must
        # use action_date only when its basis is source-supported.
        "date": publication_date,
        "publication_date": publication_date,
        "action_date": action_date,
        "action_date_basis": action_date_basis,
        "url": source_url,
        "authors": authors,
        "disposition": disposition,
        "evidence_ids": [evidence_id],
        "source_id": SOURCE_ID,
        "record_class": record_class,
        "record_quality": record_quality,
        "document_type": document_type or None,
        "record_locator": record_locator,
        "raw_sha256": raw_sha256,
        "source_row": {
            "id": record_id,
            "status": row.get("Status"),
            "created": row.get("Created"),
            "imported": row.get("Imported"),
            "attachment_group_id": row.get("AttachmentGroupId"),
        },
        "evidence": [evidence],
    }


def parse_vaski_row(
    row: Mapping[str, Any],
    *,
    source_url: str | None = None,
    retrieved_at: str | None = None,
) -> dict[str, Any]:
    """Parse a row decoded from the Vaski ``rows`` endpoint."""

    xml = row.get("XmlData") or row.get("XmlDataFi") or row.get("xml")
    if not isinstance(xml, str) or not xml.strip():
        raise ValueError("Vaski row has no XmlData")
    return parse_vaski_xml(
        xml,
        source_url=source_url,
        row_metadata=row,
        retrieved_at=retrieved_at,
    )


def merge_vaski_records(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Merge content and procedural rows for one matter.

    Only a procedural source can supply a disposition.  The function does
    not infer enactment from the initiative text and does not infer a policy
    relation from lexical overlap.
    """

    rows = [deepcopy(dict(record)) for record in records]
    if not rows:
        raise ValueError("cannot merge an empty Vaski record set")
    matter_ids = {str(row.get("matter_id") or "") for row in rows}
    matter_ids.discard("")
    if len(matter_ids) != 1:
        raise ValueError(f"records cover multiple or missing matters: {sorted(matter_ids)}")
    content_rows = [row for row in rows if row.get("record_quality") == "SUBSTANTIVE"]
    if content_rows:
        # A full submitted document wins over a LegislativeMotion transfer
        # shell.  Length is only a deterministic tie-break between substantive
        # snapshots; all rows (including shorter/stale versions) remain in
        # source_records below for audit and later conflict review.
        base = max(content_rows, key=lambda row: len(row.get("text") or ""))
        selection_reason = (
            "SUBSTANTIVE_CONTENT_OVER_REFERENCE_SHELL"
            if any(row.get("record_quality") == "REFERENCE_SHELL" for row in rows)
            else "LONGEST_SUBSTANTIVE_CONTENT_TIEBREAK"
        )
    else:
        # A procedural-only query still returns a useful procedural object,
        # while preserving the fact that no submitted content was available.
        base = max(rows, key=lambda row: len(row.get("text") or ""))
        selection_reason = "NO_SUBSTANTIVE_CONTENT_AVAILABLE"
    merged = deepcopy(base)
    merged["evidence_ids"] = []
    merged["evidence"] = []
    merged["source_records"] = []
    dispositions = []
    for row in rows:
        for evidence_id in row.get("evidence_ids", []):
            if evidence_id not in merged["evidence_ids"]:
                merged["evidence_ids"].append(evidence_id)
        for evidence in row.get("evidence", []):
            if evidence not in merged["evidence"]:
                merged["evidence"].append(evidence)
        merged["source_records"].append(
            {
                "record_class": row.get("record_class"),
                "record_quality": row.get("record_quality"),
                "record_locator": row.get("record_locator"),
                "raw_sha256": row.get("raw_sha256"),
                "publication_date": row.get("publication_date", row.get("date")),
                "action_date": row.get("action_date"),
                "action_date_basis": row.get("action_date_basis"),
                "text_length": len(row.get("text") or ""),
                "evidence_ids": row.get("evidence_ids", []),
            }
        )
        if row.get("disposition"):
            dispositions.append(row["disposition"])
        if not merged.get("authors") and row.get("authors"):
            merged["authors"] = row["authors"]
    if dispositions:
        merged["disposition"] = max(dispositions, key=lambda item: item.get("date") or "")
    else:
        merged["disposition"] = None

    merged = reconcile_action_dates(merged)
    merged["selection"] = {
        "record_locator": base.get("record_locator"),
        "raw_sha256": base.get("raw_sha256"),
        "reason": selection_reason,
        "preserved_record_count": len(rows),
    }
    return merged


def reconcile_action_dates(obj: Mapping[str, Any]) -> dict[str, Any]:
    """Reconcile retained rows without treating signing as filing.

    This also permits an existing merged register object to be recompiled from
    its retained source records. Conflicting filing dates remain unresolved;
    a later signature does not prove the selected text existed at filing.
    """
    merged = deepcopy(dict(obj))
    rows = merged.get("source_records") or []
    if merged.get("action_date_basis") == "SIGNATURE_DATE":
        merged["signature_date"] = merged.get("action_date")
    elif "signature_date" not in merged:
        signatures = [row for row in rows if row.get("record_locator") == merged.get("record_locator")
                      and row.get("action_date_basis") == "SIGNATURE_DATE"]
        merged["signature_date"] = signatures[0].get("action_date") if signatures else None
    events = [row for row in rows if row.get("record_quality") == "PROCEDURAL"
              and row.get("action_date") and row.get("action_date_basis") == "VIREILLETULO_EVENT"]
    dates = {row["action_date"] for row in events}
    if len(dates) > 1:
        merged["action_date"] = None
        merged["action_date_basis"] = None
        merged["date_binding_state"] = "CONFLICTING_FILING_DATES"
        for key in ("action_date_provenance", "action_date_source_record_locator",
                    "action_date_source_raw_sha256", "action_date_source_evidence_ids"):
            merged.pop(key, None)
        return merged
    if events:
        event = min(events, key=lambda row: str(row.get("record_locator") or ""))
        merged["action_date"] = event["action_date"]
        merged["action_date_basis"] = "VIREILLETULO_EVENT"
        provenance = {"record_locator": event.get("record_locator"), "raw_sha256": event.get("raw_sha256"),
                      "evidence_ids": list(event.get("evidence_ids") or []), "basis": "VIREILLETULO_EVENT"}
        merged["action_date_provenance"] = provenance
        merged["action_date_source_record_locator"] = provenance["record_locator"]
        merged["action_date_source_raw_sha256"] = provenance["raw_sha256"]
        merged["action_date_source_evidence_ids"] = provenance["evidence_ids"]
        signature = merged.get("signature_date")
        merged["date_binding_state"] = "SIGNATURE_AFTER_FILING" if signature and signature > event["action_date"] else "EXPLICIT_FILING_EVENT"
    else:
        merged["date_binding_state"] = "SIGNATURE_ONLY" if merged.get("action_date_basis") == "SIGNATURE_DATE" else "NO_FILING_DATE"
    return merged


def _decode_page(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    columns = payload.get("columnNames") or []
    return [dict(zip(columns, raw)) for raw in payload.get("rowData", [])]


def fetch_vaski_rows(
    matter_id: str,
    *,
    per_page: int = 100,
    max_pages: int = 500,
    client: Any = None,
) -> list[dict[str, Any]]:
    """Fetch all rows for one exact official matter identifier."""

    if not _MATTER.match(matter_id.strip()):
        raise ValueError(f"not an Eduskunta matter identifier: {matter_id!r}")
    getter = client.get if client is not None else httpx.get
    rows: list[dict[str, Any]] = []
    for page in range(max_pages):
        response = getter(
            VASKI_ROWS_URL,
            params={"columnName": "Eduskuntatunnus", "columnValue": matter_id, "perPage": per_page, "page": page},
            headers={"User-Agent": USER_AGENT},
            timeout=90,
        )
        response.raise_for_status()
        payload = response.json()
        rows.extend(_decode_page(payload))
        if not payload.get("hasMore"):
            return rows
    raise RuntimeError(f"pagination runaway for VaskiData {matter_id}")


def source_coverage(
    matter_ids: Iterable[str],
    *,
    requested_rows: int,
    parsed_rows: int,
    retrieved_at: str,
) -> dict[str, Any]:
    """Return a coverage packet suitable for ``source_coverage.json``."""

    matters = list(dict.fromkeys(matter_ids))
    return {
        "coverage_id": f"{SOURCE_ID}:{hashlib.sha256('|'.join(matters).encode()).hexdigest()[:16]}",
        "source_id": SOURCE_ID,
        "publisher": "Eduskunta / Parliament of Finland",
        "url": VASKI_ROWS_URL,
        "retrieved_at": retrieved_at,
        "query": {"columnName": "Eduskuntatunnus", "matter_ids": matters},
        "enumeration": "EXACT_MATTER_IDENTIFIERS",
        "requested_matter_count": len(matters),
        "requested_row_count": requested_rows,
        "parsed_row_count": parsed_rows,
        "rights_or_terms": "Official open data; verify current terms at the publisher URL.",
        "limitations": [
            "This packet covers only the explicitly requested matter identifiers.",
            "Absence from VaskiData is not evidence that an initiative never existed.",
            "The adapter records submitted initiatives and procedural dispositions, not policy outcomes.",
        ],
    }


def acquire_initiatives(
    matter_ids: Iterable[str],
    *,
    raw_dir: Path | None = None,
    output_path: Path | None = None,
    client: Any = None,
    retrieved_at: str | None = None,
) -> dict[str, Any]:
    """Acquire, hash, normalize, and optionally write exact-matter records.

    Network access is opt-in by calling this function.  Tests and offline
    rebuilds can pass a fake ``client`` or call the parser directly on a
    frozen source slice.
    """

    ensure_dirs()
    fetched_at = retrieved_at or datetime.now(UTC).replace(microsecond=0).isoformat()
    destination = raw_dir or RAW / "eduskunta" / "vaski"
    destination.mkdir(parents=True, exist_ok=True)
    objects: list[dict[str, Any]] = []
    row_count = 0
    parsed_row_count = 0
    matter_list = list(dict.fromkeys(str(value).strip() for value in matter_ids if str(value).strip()))
    for matter_id in matter_list:
        rows = fetch_vaski_rows(matter_id, client=client)
        parsed: list[dict[str, Any]] = []
        url = _source_url(matter_id)
        for row in rows:
            xml = row.get("XmlData") or row.get("XmlDataFi") or ""
            if not xml:
                continue
            record_id = str(row.get("Id") or "unknown")
            safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{matter_id}-{record_id}")
            (destination / f"{safe_id}.xml").write_bytes(str(xml).encode("utf-8"))
            parsed.append(parse_vaski_row(row, source_url=url, retrieved_at=fetched_at))
        row_count += len(rows)
        parsed_row_count += len(parsed)
        if parsed:
            objects.append(merge_vaski_records(parsed))
    coverage = source_coverage(
        matter_list,
        requested_rows=row_count,
        parsed_rows=parsed_row_count,
        retrieved_at=fetched_at,
    )
    result = {"objects": objects, "coverage": coverage, "retrieved_at": fetched_at}
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("".join(json.dumps(obj, ensure_ascii=False) + "\n" for obj in objects), encoding="utf-8")
    return result


__all__ = [
    "SOURCE_ID",
    "VASKI_ROWS_URL",
    "acquire_initiatives",
    "fetch_vaski_rows",
    "merge_vaski_records",
    "parse_vaski_row",
    "parse_vaski_xml",
    "source_coverage",
]

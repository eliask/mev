"""Runnable purpose demonstration: ballot disposition → reviewed answers → static site."""

import hashlib
import json
from pathlib import Path
from typing import Any

from paa.ballot_disposition import (
    BallotDispositionError,
    actor_row,
    evaluate_corpus,
    load_corpus,
    validate_packet,
)
from paa.case_site import _document, _e
from paa.config import FIXTURE_DIR
from paa.release import build_release

DEFAULT_SELECTION = FIXTURE_DIR / "purpose_demo_selection.json"
DEFAULT_CORPUS = FIXTURE_DIR / "ballot_disposition_corpus.json"

_DOMAIN_LABELS = {
    "hospital_oulaskangas": "Sairaalaverkko ja Oulaskangas",
    "wage_mediation": "Työriitojen sovittelu (HE 146/2024)",
}

_BRACKET_LABELS = {
    "FIRST_ROUND_ELIMINATION": "Kahden ehdotuksen vertailu",
    "REPORT_VS_SURVIVOR": "Mietintö ja edellisen kierroksen voittaja",
}

# Evidence-backed publication expectations derived from the maintained corpus.
_PUBLICATION_EXPECTATIONS: tuple[dict[str, Any], ...] = (
    {
        "instance_id": "hospital_oulaskangas",
        "person_id": "1537",
        "vote_id": "53538",
        "raw_response": "EI",
        "selected_label": "hylkääminen",
    },
    {
        "instance_id": "hospital_oulaskangas",
        "person_id": "1450",
        "vote_id": "53538",
        "raw_response": "JAA",
        "selected_label": "Hyväksyminen",
    },
    {
        "instance_id": "hospital_oulaskangas",
        "person_id": "1537",
        "vote_id": "55560",
        "raw_response": "EI",
        "selection_state": "AUTHORED_TEXT_BOUND",
    },
    {
        "instance_id": "hospital_oulaskangas",
        "person_id": "1537",
        "vote_id": "53546",
        "raw_response": "TYHJA",
        "selection_state": "NON_SUBSTANTIVE_NO_SIDE",
    },
    {
        "instance_id": "wage_mediation",
        "person_id": "1537",
        "vote_id": "53586",
        "raw_response": "JAA",
        "matter": "HE 146/2024 vp",
    },
    {
        "instance_id": "wage_mediation",
        "person_id": "1511",
        "vote_id": "53586",
        "raw_response": "EI",
        "matter": "HE 146/2024 vp",
    },
)


def verify_computation_dependencies(computation: dict) -> list[str]:
    """Refuse to seal a release whose computation does not match fixed publication rows."""

    checked: list[str] = []
    for expectation in _PUBLICATION_EXPECTATIONS:
        instance_id = expectation["instance_id"]
        packet = computation["instances"][instance_id]
        validate_packet(packet)
        row = actor_row(packet, person_id=expectation["person_id"], vote_id=expectation["vote_id"])
        for key, value in expectation.items():
            if key in {"instance_id", "person_id", "vote_id"}:
                continue
            if row.get(key) != value:
                raise BallotDispositionError(
                    f"publication dependency failed for {instance_id} "
                    f"person {expectation['person_id']} vote {expectation['vote_id']}: "
                    f"{key}={row.get(key)!r} != {value!r}"
                )
        checked.append(f"{instance_id}:{expectation['person_id']}:{expectation['vote_id']}")
    hospital = computation["instances"]["hospital_oulaskangas"]
    if hospital["dispositions"]["53548"]["winner_label"] == hospital["bindings"]["53546"]["ei_label"]:
        raise BallotDispositionError("eliminated Merinen 4 must not be the second-round institutional winner")
    if not hospital.get("bracket_chains"):
        raise BallotDispositionError("hospital instance must retain validated bracket chain")
    if "corpus_path" in computation:
        raise BallotDispositionError("public computation must not retain absolute corpus_path")
    if not computation.get("corpus_filename") or not computation.get("corpus_sha256"):
        raise BallotDispositionError("public computation needs bundled corpus filename and hash")
    return checked


def render_computation_page(computation: dict, *, corpus: dict) -> str:
    """Public consumer: Finnish labels first; IDs/hashes in secondary details."""

    documents = corpus.get("documents") or {}
    blocks: list[str] = []
    for instance_id in computation["instance_ids"]:
        packet = computation["instances"][instance_id]
        validate_packet(packet)
        title = _DOMAIN_LABELS.get(instance_id, instance_id)
        blocks.append(f"<h2>{_e(title)}</h2>")
        blocks.append(
            "<div class=\"table-wrap\"><table>"
            "<thead><tr>"
            "<th>Äänestys</th><th>Henkilö</th><th>Raakavastaus</th><th>Valittu vaihtoehto</th>"
            "<th>Ryhmän enemmistö</th><th>Sama kuin ryhmä</th><th>Äänestyksen tulos</th><th>Vaihe</th>"
            "</tr></thead><tbody>"
        )
        for row in packet["rows"]:
            blocks.append(
                "<tr>"
                f"<td>{_e(str(row.get('vote_id')))}</td>"
                f"<td>{_e(str(row.get('display_name') or row.get('person_id')))}</td>"
                f"<td>{_e(str(row.get('raw_response') or ''))}</td>"
                f"<td>{_e(str(row.get('selected_label') or ''))}</td>"
                f"<td>{_e(str(row.get('peer_majority') or ''))}</td>"
                f"<td>{_e('kyllä' if row.get('matches_peer_majority') is True else ('ei' if row.get('matches_peer_majority') is False else ''))}</td>"
                f"<td>{_e(str(row.get('institutional_winner_label') or ''))}</td>"
                f"<td>{_e(_BRACKET_LABELS.get(str(row.get('bracket_role') or ''), ''))}</td>"
                "</tr>"
            )
        blocks.append("</tbody></table></div>")
        # Authored alternatives with resolvable witnesses from the sealed corpus.
        blocks.append("<h3>Sidotut vaihtoehtotekstit</h3><ul>")
        for vote_id, binding in sorted(packet.get("bindings", {}).items()):
            source_url = (corpus.get("votes") or {}).get(vote_id, {}).get("source_url") or ""
            for side, label_key, text_key, witness_key in (
                ("JAA", "jaa_label", "jaa_text", "jaa_witness"),
                ("EI", "ei_label", "ei_text", "ei_witness"),
            ):
                text = binding.get(text_key)
                if not text:
                    continue
                witness = binding.get(witness_key) or {}
                doc_id = witness.get("source_id") or ""
                blocks.append(
                    "<li>"
                    f"<strong>{_e(str(vote_id))} · {_e(side)} · {_e(str(binding.get(label_key) or ''))}</strong>"
                    f"<blockquote>{_e(text)}</blockquote>"
                    f'<p><a href="{_e(source_url)}">Virallinen äänestystulos</a>'
                    + (
                        f' · todistaja <code>{_e(doc_id)}</code> '
                        f"[{witness.get('start')}:{witness.get('end')}]"
                        if doc_id
                        else ""
                    )
                    + "</p></li>"
                )
                if doc_id and doc_id not in documents:
                    raise BallotDispositionError(f"witness document {doc_id} missing from sealed corpus")
        blocks.append("</ul>")
        blocks.append(
            "<details><summary>Tekniset tunnisteet</summary>"
            f"<p>Instanssi: <code>{_e(instance_id)}</code> · "
            f"domain <code>{_e(str(packet.get('domain')))}</code> · "
            f"cutoff <code>{_e(str(packet.get('cutoff')))}</code><br>"
            f"result_sha256 <code>{_e(packet['result_sha256'])}</code></p>"
            "<p>Checksum todistaa paketin sisäisen eheyden, ei julkaisijan aitoutta.</p>"
            "</details>"
        )
    body = (
        '<style>.table-wrap{overflow-x:auto;-webkit-overflow-scrolling:touch;max-width:100%;margin:1rem 0}'
        ".table-wrap table{min-width:36rem}</style>"
        '<nav><a href="index.html">Tutkitut kysymykset</a></nav>'
        "<h1>Äänestyslaskenta</h1>"
        "<p>Taulukko johdetaan ja tarkistetaan versiosidotusta laskennasta ennen käyttöä. "
        "Raakakoodit JAA/EI eivät ole politiikkasisältö ilman sidottua vaihtoehtoa.</p>"
        f"<p>Korpus: <code>{_e(str(computation.get('corpus_filename')))}</code> · "
        f'<a download href="ballot_disposition_corpus.json">Lataa julkinen korpus</a> · '
        f'<a download href="ballot_disposition.json">Lataa laskentatulos</a></p>'
        + "".join(blocks)
        + "<p class=\"review-status\">Tekoälyavusteinen lähdeluku; ei riippumatonta ihmistarkistusta.</p>"
    )
    return _document(
        "Äänestyslaskenta",
        body,
        description="Versiosidottu äänestys-, vaihtoehto- ja ryhmävertailu kahdessa asiakokonaisuudessa.",
    )


def build_purpose_demo(
    output: Path,
    *,
    source_revision: str,
    selection_path: Path = DEFAULT_SELECTION,
    corpus_path: Path = DEFAULT_CORPUS,
    base_url: str = "",
) -> dict:
    """Evaluate the generic ballot computation, then seal it inside the static site."""

    corpus = load_corpus(corpus_path)
    corpus_bytes = corpus_path.read_bytes()
    computation = evaluate_corpus(corpus_path)
    checked = verify_computation_dependencies(computation)
    computation_json = json.dumps(computation, ensure_ascii=False, indent=2) + "\n"
    if ".tmp/" in computation_json or "/home/" in computation_json or "PHYSICALDRIVE" in computation_json:
        raise BallotDispositionError("computation export leaked a private path")
    computation_page = render_computation_page(computation, corpus=corpus)
    corpus_sha = computation["corpus_sha256"]
    if hashlib.sha256(corpus_bytes).hexdigest() != corpus_sha:
        raise BallotDispositionError("corpus bytes disagree with computation corpus_sha256")
    extra_files = {
        "ballot_disposition_corpus.json": corpus_bytes,
        "ballot_disposition.json": computation_json,
        "ballot_disposition.html": computation_page,
    }
    manifest_extra = {
        "ballot_disposition": {
            "path": "ballot_disposition.json",
            "page": "ballot_disposition.html",
            "corpus_path": "ballot_disposition_corpus.json",
            "corpus_filename": computation.get("corpus_filename"),
            "corpus_sha256": corpus_sha,
            "schema_version": computation.get("schema_version"),
            "instance_ids": computation.get("instance_ids"),
            "result_sha256": computation.get("result_sha256"),
            "publication_dependencies": checked,
            "limits": [
                "Checksum proves packet consistency, not publisher authenticity.",
                "Documentary alternative selection only; no causal attribution.",
            ],
        }
    }
    release = build_release(
        selection_path,
        output,
        source_revision=source_revision,
        base_url=base_url,
        extra_files=extra_files,
        index_extra_html=(
            '<p class="review-status">'
            '<a href="ballot_disposition.html">Äänestyslaskennan taulukko</a> · '
            '<a download href="ballot_disposition.json">Lataa laskentatulos</a> · '
            '<a download href="ballot_disposition_corpus.json">Lataa korpus</a>'
            "</p>"
        ),
        manifest_extra=manifest_extra,
        shared_publication={
            "computation_page": "ballot_disposition.html",
            "computation_label": "Äänestysvaihtoehdot ja ryhmävertailu",
            "computation_case_ids": [
                "hospital-hanninen-oulaskangas-attribution",
                "hospital-monitoring-cost-shift-repair",
            ],
        },
    )
    destination = Path(release["destination"])
    sealed = (destination / "ballot_disposition.json").read_bytes()
    if sealed != computation_json.encode("utf-8"):
        raise BallotDispositionError("sealed computation bytes differ from evaluated computation")
    if (destination / "ballot_disposition_corpus.json").read_bytes() != corpus_bytes:
        raise BallotDispositionError("sealed corpus bytes differ from bundled fixture")
    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    if manifest["files"].get("ballot_disposition.json") != hashlib.sha256(sealed).hexdigest():
        raise BallotDispositionError("manifest hash missing for sealed computation")
    if manifest["files"].get("ballot_disposition_corpus.json") != corpus_sha:
        raise BallotDispositionError("manifest hash missing for sealed corpus")
    if "ballot_disposition" not in manifest:
        raise BallotDispositionError("manifest lacks ballot_disposition audit block")
    return {
        **release,
        "computation_path": str(destination / "ballot_disposition.json"),
        "computation_page": str(destination / "ballot_disposition.html"),
        "instance_ids": computation["instance_ids"],
        "hospital_rows": len(computation["instances"]["hospital_oulaskangas"]["rows"]),
        "wage_rows": len(computation["instances"]["wage_mediation"]["rows"]),
        "publication_dependencies": checked,
        "computation_result_sha256": computation["result_sha256"],
        "corpus_sha256": corpus_sha,
        "archive_path": str(release["archive"]),
    }


__all__ = [
    "DEFAULT_CORPUS",
    "DEFAULT_SELECTION",
    "build_purpose_demo",
    "render_computation_page",
    "verify_computation_dependencies",
]

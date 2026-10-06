"""Human reading view of a maintained matter ledger.

The hub page keeps a compact actor/issue/action/stage summary. Full supporting
records live on static per-person pages (and optional JSON shards) so a large
matter does not preload tens of thousands of DOM nodes. JavaScript may navigate
or fetch a person shard; it never invents semantic relations.
"""

import html
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from paa.case_site import _document
from paa.matter_ledger import decode_ledger_action_kind, matter_subject_title_from_objects

PEOPLE_DIR_NAME = "people"
RECORDS_DIR_NAME = "records"

_CHOICE_CSS = (
    ".choice-table{font-size:.88em;width:100%}"
    ".choice-table td,.choice-table th{padding:.35rem .25rem;vertical-align:top}"
    ".choice-table th{overflow-wrap:normal}"
    ".choice-table td:first-child,.choice-table th:first-child{white-space:nowrap;width:1%}"
    ".choice-table td:last-child,.choice-table th:last-child{width:1%}"
    ".choice-table th:last-child,.choice-table .ballot-row td:last-child{white-space:nowrap}"
    ".choice-table small{font-size:.85em}"
    ".choice-table td:nth-child(3) small,.choice-table td:last-child small{display:block;white-space:nowrap}"
    ".choice-scope{font-size:.8em;color:#53656a;margin:.25rem 0}"
    "@media(max-width:600px){.choice-table{font-size:.82em}}"
)


def _e(value: object) -> str:
    return html.escape(str(value if value is not None else ""), quote=True)


def resolve_matter_subject_title(matter: Mapping) -> dict:
    """Prefer producer metadata; derive from retained object titles when absent."""

    state = matter.get("subject_title_state")
    if state in {"COMMON_OBJECT_TITLE", "AMBIGUOUS_OBJECT_TITLES", "MISSING"}:
        return {
            "subject_title": matter.get("subject_title"),
            "subject_title_state": state,
            "subject_title_candidates": list(matter.get("subject_title_candidates") or []),
        }
    return matter_subject_title_from_objects(matter.get("objects") or [])


def matter_choice_label(matter_id: str, subject: Mapping | None = None) -> str:
    """Identifier plus clear subject when a single source title is retained."""

    title = (subject or {}).get("subject_title")
    if title:
        return f"{matter_id} — {title}"
    return matter_id


def _subject_heading(matter: Mapping) -> str:
    """One short source-title label for context; full official title stays in sources."""

    title = str(resolve_matter_subject_title(matter).get("subject_title") or "")
    title = re.sub(r"^Hallituksen esitys eduskunnalle (?:laeiksi|laiksi)\s+", "", title)
    title = re.sub(r"\s+sekä niihin liittyviksi laeiksi$", "", title)
    if len(title) > 115:
        title = title[:112].rsplit(" ", 1)[0] + "…"
    return f"<p class='issue-label'>{_e(title)}</p>" if title else ""


def _source_link(url: str | None, label: str) -> str:
    if url and urlsplit(url).scheme in {"https", "http"}:
        return f'<a href="{_e(url)}">{_e(label)}</a>'
    return _e(label)


def _is_nonspecific_source_url(url: str | None) -> bool:
    """True when URL is missing or a generic search root, not a record page."""

    text = str(url or "").strip()
    if not text:
        return True
    parts = urlsplit(text)
    path = parts.path.rstrip("/")
    return path.endswith("/api/v1/search")


def _verification_source_html(
    source: Mapping,
    *,
    record_href: str | None = None,
) -> str:
    """Record-specific source link when supported; otherwise honest locator."""

    url = str(source.get("url") or source.get("source_url") or "").strip()
    locator = str(source.get("record_locator") or source.get("source_locator") or "").strip()
    bits: list[str] = []
    if url and not _is_nonspecific_source_url(url):
        bits.append("<p>" + _source_link(url, "Avaa alkuperäinen lähde") + "</p>")
        if locator:
            bits.append(f"<p class='muted'>Lähdeosoitin: <code>{_e(locator)}</code></p>")
    else:
        if locator:
            bits.append(f"<p>Tarkka lähdeosoitin: <code>{_e(locator)}</code></p>")
        elif not record_href:
            bits.append("<p class='muted'>Tarkkaa lähdeosoitetta ei ole paketissa.</p>")
        if url and _is_nonspecific_source_url(url):
            bits.append(
                "<p class='muted'>Yleinen hakuosoite (ei yksittäisen kirjauksen osoite): "
                + _source_link(url, url)
                + "</p>"
            )
    if record_href:
        label = (
            "Kirjauksen erillissivu (paikallinen lähdekohta)"
            if _is_nonspecific_source_url(url)
            else "Kirjauksen erillissivu"
        )
        bits.append(f'<p><a href="{_e(record_href)}">{_e(label)}</a></p>')
    return "".join(bits)


def _actor_key(actor: Mapping) -> str:
    if actor.get("actor_id"):
        return "actor:" + actor["actor_id"]
    if actor.get("person_id"):
        return "person:" + actor["person_id"]
    return "name-only:" + (actor.get("name") or "tuntematon")


def actor_page_slug(actor_key: str) -> str:
    """Filesystem-safe slug for a static per-person page."""

    text = re.sub(r"[^\w.-]+", "_", actor_key.strip(), flags=re.UNICODE)
    return text.strip("._") or "tuntematon"


def record_page_slug(index: int, action: Mapping) -> str:
    object_id = str(action.get("object_id") or action.get("intervention_id") or index)
    text = re.sub(r"[^\w.-]+", "_", object_id.strip(), flags=re.UNICODE)
    return f"{index:04d}_{text.strip('._') or 'record'}"


_ROLES = {
    "FIRST_SIGNATURE": "Ensimmäinen allekirjoittaja",
    "CO_SIGNATURE": "Muu allekirjoittaja",
    "QUESTION_AUTHOR": "Kysymyksen ensimmäinen allekirjoittaja",
    "QUESTION_CO_SIGNATURE": "Kysymyksen allekirjoittaja",
    "SPEAKER": "Puheenvuoro",
    "RESPONDENT": "Vastauksessa nimetty henkilö",
    "BALLOT_MEMBER": "Äänestysmerkintä",
    "CHAIR_TITLE_NAMED_PROPOSER": "Puhemiehen otsikossa nimetty ehdotuksen tekijä",
}
_ACTION_KINDS_FI = {
    "VOTE_CAST": "Äänesti",
    "FIRST_SIGNATORY": "Ensimmäinen allekirjoittaja",
    "INITIATIVE_COSIGNED": "Muu allekirjoittaja",
    "QUESTION_FILED": "Jätti kysymyksen",
    "QUESTION_COSIGNED": "Allekirjoitti kysymyksen",
    "SPEECH_DELIVERED": "Puhui",
    "ANSWER_ATTRIBUTED": "Vastauksessa nimetty",
    "ALTERNATIVE_PUBLICLY_ATTRIBUTED": "Nimetty äänestysvaihtoehdossa",
    "OTHER_OBSERVABLE_ACTION": "Muu kirjattu toimi",
}
_KINDS = {
    "COMMITTEE_NAMED_PARTICIPATION": "Valiokuntaroolit",
    "VOTE": "Äänestykset",
    "SPEECH": "Puheenvuorot",
    "LEGISLATIVE_INITIATIVE": "Lakialoitteet",
    "WRITTEN_QUESTION": "Kirjalliset kysymykset",
    "GOVERNMENT_ANSWER": "Vastaukset",
    "FORMAL_ALTERNATIVE": "Nimetyt äänestysvaihtoehdot",
}
_DISPOSITIONS = {
    "REJECTED": "Hylätty",
    "ADOPTED": "Hyväksytty",
    "ANSWERED": "Vastattu",
    "RECORDED": "Kirjattu",
    "PREVAILED_AT_THIS_VOTE": "Voitti tässä äänestyksessä",
    "LOST_AT_THIS_VOTE": "Hävisi tässä äänestyksessä",
    "JAA_SIDE_PREVAILED_AT_THIS_VOTE": "JAA-puoli voitti tässä äänestyksessä",
    "EI_SIDE_PREVAILED_AT_THIS_VOTE": "EI-puoli voitti tässä äänestyksessä",
    "UNRESOLVED": "Avoin",
    "VOTE_RECORDED": "Äänestys kirjattu",
    "TIED_OR_UNRESOLVED": "Tasapeli tai avoin",
}

_HUB_JS = """"use strict";
const actorFilter = document.getElementById("actor-filter");
const panel = document.getElementById("person-panel");
const status = document.getElementById("record-count");
const overview = document.getElementById("decision-overview");
let loadGeneration = 0;
function personHref(value) {
  const option = Array.from(actorFilter.options).find((row) => row.value === value);
  return option ? option.dataset.href : "";
}
function rebaseRelativeLinks(root, baseUrl) {
  for (const el of root.querySelectorAll("[href]")) {
    const href = el.getAttribute("href");
    if (!href || href.startsWith("#") || href.startsWith("mailto:") || href.startsWith("javascript:")) {
      continue;
    }
    if (/^[a-zA-Z][a-zA-Z0-9+.-]*:/.test(href)) {
      continue;
    }
    try {
      el.setAttribute("href", new URL(href, baseUrl).href);
    } catch (error) {
      /* keep original href */
    }
  }
}
async function loadPerson(value) {
  const generation = ++loadGeneration;
  if (!value || value === "choose") {
    if (overview) overview.hidden = false;
    panel.hidden = true;
    panel.innerHTML = "";
    status.textContent = "";
    if (location.hash.startsWith("#henkilo=")) {
      history.replaceState(null, "", location.pathname + location.search);
    }
    return;
  }
  const href = personHref(value);
  if (!href) {
    status.textContent = "Henkilön sivua ei löytynyt.";
    return;
  }
  status.textContent = "Ladataan henkilön kirjauksia…";
  try {
    const response = await fetch(href, {credentials: "same-origin"});
    if (generation !== loadGeneration) return;
    if (!response.ok) throw new Error("fetch failed");
    const html = await response.text();
    if (generation !== loadGeneration) return;
    const doc = new DOMParser().parseFromString(html, "text/html");
    const main = doc.getElementById("person-records");
    // Rebase against the fetched person URL before hub insertion. Assigning
    // relative hrefs via innerHTML would otherwise resolve them against the hub.
    if (main) {
      rebaseRelativeLinks(main, response.url);
      if (generation !== loadGeneration) return;
      panel.innerHTML = main.innerHTML;
    } else {
      panel.innerHTML = "<p>Henkilön sivulla ei ole kirjauksia.</p>";
    }
    panel.hidden = false;
    if (overview) overview.hidden = true;
    status.textContent = "";
    history.replaceState(null, "", "#henkilo=" + encodeURIComponent(value));
  } catch (error) {
    if (generation !== loadGeneration) return;
    status.textContent = "Lataus epäonnistui; avaa henkilön staattinen sivu.";
    panel.innerHTML = '<p><a href="' + href + '">Avaa henkilön kirjaukset</a></p>';
    panel.hidden = false;
  }
}
actorFilter.addEventListener("change", () => loadPerson(actorFilter.value));
window.addEventListener("hashchange", () => {
  const params = new URLSearchParams((location.hash || "").replace(/^#/, ""));
  const actor = params.get("henkilo");
  if (actor && Array.from(actorFilter.options).some((option) => option.value === actor)) {
    actorFilter.value = actor;
    loadPerson(actor);
  } else if (!actor) {
    actorFilter.value = "choose";
    loadPerson("");
  }
});
(() => {
  const params = new URLSearchParams((location.hash || "").replace(/^#/, ""));
  const actor = params.get("henkilo");
  if (actor && Array.from(actorFilter.options).some((option) => option.value === actor)) {
    actorFilter.value = actor;
    loadPerson(actor);
  } else {
    status.textContent = "";
  }
})();
"""


def _role_interval_html(roles: Mapping | None) -> str:
    blob = roles or {}
    rows = blob.get("roles") or []
    exact = [row for row in rows if row.get("match_state") == "EXACT_DATE_OVERLAP"]
    items = "".join(
        "<li>"
        + _e(row.get("label") or row.get("kind") or "")
        + (
            f" · {_e(row.get('start_date'))}–{_e(row.get('end_date') or '…')}"
            if row.get("start_date") or row.get("end_date")
            else ""
        )
        + (f" · {_e(row.get('match_state'))}" if row.get("match_state") else "")
        + (
            f" · vaikutustila {_e(row.get('influence_state'))}"
            if row.get("influence_state")
            else ""
        )
        + "</li>"
        for row in exact
    )
    limit = (
        '<p class="muted">Osa lähteen roolijaksoista jäi liian epätarkoiksi ajalliseen liitokseen.</p>'
        if blob.get("imprecise_intervals_omitted")
        else ""
    )
    if not items and not limit:
        return ""
    return (
        "<details><summary>Lähteen ajallinen rooli</summary>"
        + (f"<ul>{items}</ul>" if items else "")
        + limit
        + "</details>"
    )


def _peer_comparison_audit_html(comparison: Mapping) -> str:
    """Denominator, exclusion and hash detail for progressive disclosure."""

    status = str(comparison.get("status") or "")
    body = (
        "<p>Oman ryhmän vertailu tässä äänestyksessä "
        "(kuvaileva vertailu, ei kurinalaisuutta tai vaikutusta).</p>"
        f"<p>Ryhmäkoodi lähteessä: {_e(comparison.get('target_group_code') or 'ei tiedossa')}. "
        f"Oma merkintä: {_e(comparison.get('target_response') or '—')}. "
        "Kohde jätetty pois vertailusta.</p>"
    )
    if comparison.get("group_other_record_count") is not None:
        body += (
            f"<p>Muut saman äänestyksen ryhmäkoodin kirjaukset: "
            f"{_e(comparison.get('group_other_record_count'))}"
            + (
                f" · pois lasketut TYHJÄ {_e(comparison.get('peer_excluded_tyhja'))}"
                f", POISSA {_e(comparison.get('peer_excluded_poissa'))}"
                + (
                    f", muut {_e(comparison.get('peer_excluded_other'))}"
                    if comparison.get("peer_excluded_other")
                    else ""
                )
                if comparison.get("peer_excluded_tyhja") is not None
                else ""
            )
            + ".</p>"
        )
    if status == "COMPARABLE":
        match = comparison.get("matches_peer_majority")
        match_fi = "kyllä" if match is True else ("ei" if match is False else "määrittelemätön")
        body += (
            f"<p>Vertailukelpoiset muut JAA/EI-äänet: {_e(comparison.get('peer_count'))} "
            f"(JAA {_e(comparison.get('peer_jaa'))}, EI {_e(comparison.get('peer_ei'))}). "
            f"Enemmistö muiden joukossa: {_e(comparison.get('peer_majority'))}. "
            f"Oma merkintä sama kuin enemmistö: {_e(match_fi)}.</p>"
        )
    else:
        body += (
            f"<p>Vertailu määrittelemätön ({_e(status)}): {_e(comparison.get('reason') or '')}. "
            "TYHJÄ/POISSA, tuntematon ryhmä, tasajako ja alle vähimmäismäärän jäävät määrittelemättömiksi.</p>"
        )
    body += (
        f"<p class='muted'>Lähde {_e(comparison.get('source_id'))}"
        + (
            f" · tiiviste {_e(str(comparison.get('source_sha256'))[:16])}…"
            if comparison.get("source_sha256")
            else ""
        )
        + ". Ryhmä tulee äänestyksen ballot-kentästä; nykyistä puoluetta ei käytetä.</p>"
    )
    return body


def _peer_comparison_html(comparison: Mapping | None, *, compact: bool = True) -> str:
    if not isinstance(comparison, Mapping) or not comparison:
        return ""
    status = str(comparison.get("status") or "")
    audit = _peer_comparison_audit_html(comparison)
    if not compact:
        return "<div class='peer-comparison'>" + audit + "</div>"
    if status == "COMPARABLE":
        majority = comparison.get("peer_majority") or "—"
        match = comparison.get("matches_peer_majority")
        if match is True:
            summary = f"sama kuin ryhmän enemmistö ({majority})"
        elif match is False:
            summary = f"poikkesi ryhmän enemmistöstä ({majority})"
        else:
            summary = f"ryhmävertailu avoin ({majority})"
        return (
            f"<details class='peer-comparison'><summary>Ryhmävertailu: {_e(summary)}</summary>"
            + audit
            + "</details>"
        )
    return (
        "<details class='peer-comparison'><summary>Ryhmävertailu (ei vertailukelpoinen)</summary>"
        + audit
        + "</details>"
    )


def _limits_section(packet: Mapping) -> str:
    limits = "".join(f"<li>{_e(item)}</li>" for item in packet.get("limits") or [])
    coverage = "".join(
        f"<li>{_e(row.get('kind'))} · {_e(row.get('declared_scope'))} · kattavuustila {_e(row.get('state'))}</li>"
        for row in packet.get("coverage") or []
    )
    missing = "".join(
        f"<li>{_e(row.get('code'))}: {_e(row.get('reason'))}</li>"
        for row in packet.get("missingness") or []
    )
    return (
        "<details class='limits'><summary>Aineiston rajat ja kattavuus</summary>"
        "<p>Nämä rajat tulevat lähdepaketista. Ne eivät ole arvioijan päätelmiä.</p>"
        + (f"<ul>{limits}</ul>" if limits else "<p>Ei erillisiä rajatekstejä.</p>")
        + (
            f"<ul>{coverage}</ul><ul>{missing}</ul>"
            if coverage or missing
            else ""
        )
        + "</details>"
    )


def _vote_result_paragraph(vote: Mapping) -> str:
    disposition = vote.get("published_disposition") or {}
    if not disposition:
        return ""
    final_adoption = disposition.get("final_adoption") or {}
    return (
        "<p><strong>Julkaistu tulos tässä äänestyksessä:</strong> "
        + _e(_DISPOSITIONS.get(disposition.get("state"), disposition.get("state")))
        + f" · laajuus {_e(disposition.get('scope') or 'THIS_VOTE_ONLY')}"
        + (
            f" · JAA {_e(disposition.get('jaa') if disposition.get('jaa') is not None else vote.get('jaa'))}"
            f" / EI {_e(disposition.get('ei') if disposition.get('ei') is not None else vote.get('ei'))}"
        )
        + "</p>"
        "<p><strong>Lopullinen hyväksyminen:</strong> "
        + _e(_DISPOSITIONS.get(final_adoption.get("state"), final_adoption.get("state") or "UNRESOLVED"))
        + ". "
        + _e(
            final_adoption.get("limitation")
            or "Äänestyksen tulos ei ole lopullinen hyväksyminen (THIS_VOTE_ONLY)."
        )
        + "</p>"
    )


def _vote_stage_html(votes: Sequence[Mapping], *, collapsed: bool = True) -> str:
    if not votes:
        return ""
    rows = "".join(
        f"<tr><td>{_e(_short_date(vote.get('action_date')))}</td>"
        f"<td>{_source_link(vote.get('source_url'), str(vote.get('title') or 'Äänestys'))}</td>"
        f"<td>{_winning_mark_html(vote)}</td></tr>"
        for vote in votes
    )
    body = (
        "<table class='choice-table'><thead><tr><th>Päivä</th><th>Vaihtoehdot · lähde</th>"
        "<th>Voitti</th></tr></thead><tbody>" + rows + "</tbody></table>"
        "<p class='choice-scope'>Tämän äänestyksen tulos, ei päätelmä voimaantulosta tai vaikutuksesta.</p>"
    )
    if not collapsed:
        return "<section class='vote-stage'><h2>Äänestykset</h2>" + body + "</section>"
    return "<details class='vote-stage'><summary>Äänestykset</summary>" + body + "</details>"


def _action_kind_label(action: Mapping) -> str:
    decoded = decode_ledger_action_kind(
        action.get("action_kind"),
        role_label=action.get("role_label"),
        signature_value=(action.get("signature") or {}).get("value"),
        kind=action.get("kind"),
    )
    return str(decoded.get("value") or "")


def _action_kind_fi(action: Mapping) -> str:
    raw = _action_kind_label(action)
    if not raw:
        return ""
    return _ACTION_KINDS_FI.get(raw, raw)


def _display_name(actor: Mapping) -> str:
    name = actor.get("name") or actor.get("person_id") or "Nimetön lähdemerkintä"
    if _actor_key(actor).startswith("name-only:"):
        name += " (vain nimi, identiteetti avoin)"
    return str(name)


def _packet_url_for_depth(packet_url: str | None, *, depth: int) -> str | None:
    """Relocate a hub-relative packet_url for pages ``depth`` directories deeper."""

    if not packet_url:
        return None
    text = str(packet_url)
    if text.startswith(("http://", "https://", "#", "mailto:")):
        return text
    if depth <= 0:
        return text
    return ("../" * depth) + text


def compact_actor_summaries(actions: Sequence[Mapping]) -> list[dict[str, Any]]:
    """Compact per-actor counts for the matter hub; no semantic inference."""

    buckets: dict[str, dict[str, Any]] = {}
    for index, action in enumerate(actions):
        actor = action.get("actor") or {}
        key = _actor_key(actor)
        row = buckets.get(key)
        if row is None:
            row = {
                "actor_key": key,
                "slug": actor_page_slug(key),
                "name": _display_name(actor),
                "identity_state": actor.get("identity_state"),
                "record_count": 0,
                "kinds": Counter(),
                "roles": Counter(),
                "action_kinds": Counter(),
                "dates": [],
                "record_indexes": [],
            }
            buckets[key] = row
        row["record_count"] += 1
        row["kinds"][str(action.get("kind") or "UNKNOWN")] += 1
        row["roles"][str(action.get("role_label") or "UNKNOWN")] += 1
        action_kind = _action_kind_fi(action) or _action_kind_label(action)
        if action_kind:
            row["action_kinds"][action_kind] += 1
        raw_kind = _action_kind_label(action)
        if raw_kind:
            row.setdefault("action_kinds_raw", Counter())[raw_kind] += 1
        if action.get("action_date"):
            row["dates"].append(str(action["action_date"]))
        row["record_indexes"].append(index)
    summaries = []
    for row in buckets.values():
        dates = sorted(row["dates"])
        summaries.append(
            {
                "actor_key": row["actor_key"],
                "slug": row["slug"],
                "name": row["name"],
                "identity_state": row["identity_state"],
                "record_count": row["record_count"],
                "kinds": dict(sorted(row["kinds"].items())),
                "roles": dict(sorted(row["roles"].items())),
                "action_kinds": dict(sorted(row["action_kinds"].items())),
                "action_kinds_raw": dict(sorted((row.get("action_kinds_raw") or {}).items())),
                "earliest_action_date": dates[0] if dates else None,
                "latest_action_date": dates[-1] if dates else None,
                "record_indexes": row["record_indexes"],
            }
        )
    summaries.sort(key=lambda item: (str(item["name"]).casefold(), item["actor_key"]))
    return summaries


def _stage_summary_html(matter: Mapping, actions: Sequence[Mapping]) -> str:
    kind_counts = Counter(str(action.get("kind") or "UNKNOWN") for action in actions)
    kind_bits = ", ".join(
        f"{_KINDS.get(kind, kind)} {_e(count)}" for kind, count in sorted(kind_counts.items())
    )
    votes = [row for row in (matter.get("vote_sources") or []) if isinstance(row, Mapping)]
    vote_note = f"{_e(len(votes))} äänestyslähdettä" if votes else "ei äänestyslähteitä tässä paketissa"
    objects = matter.get("objects") or []
    object_kinds = Counter(str(obj.get("kind") or "UNKNOWN") for obj in objects if isinstance(obj, Mapping))
    object_bits = ", ".join(f"{_e(kind)} {_e(count)}" for kind, count in sorted(object_kinds.items()))
    early_votes = "".join(_vote_result_paragraph(vote) for vote in votes[:3])
    if votes and not early_votes:
        early_votes = (
            "<p class='muted'>Äänestyslähteitä on paketissa, mutta julkaistua "
            "äänestystulosta ei ole vielä sidottu tähän yhteenvetoon.</p>"
        )
    return (
        "<section class='stage-summary'>"
        "<h2>Asian tiivis yhteenveto</h2>"
        + (
            early_votes
            or "<p><strong>Institutionaalinen päätöstulos:</strong> ei sidottua "
            "äänestys- tai hyväksymistodistetta tässä paketissa.</p>"
        )
        + f"<p class='muted'>Kirjaustyypit (taustamäärä): {kind_bits or 'ei kirjauksia'}. {vote_note}."
        + (f" Lähdeobjektit: {object_bits}." if object_bits else "")
        + "</p>"
        + "<p class='muted'>Yhteys on sama asiatunniste — ei kausaalinen päätelmä. "
        "Määrä ei ole kattavuustodiste. Vaikutusta, toimeenpanoa tai lopullista "
        "hyväksymistä ei päätellä kirjausten määrästä.</p></section>"
    )


def _answer_document_html(answer: Mapping | None) -> str:
    if not isinstance(answer, Mapping):
        return ""
    role = answer.get("role") or {}
    dates = answer.get("dates") or {}
    name = " ".join(
        part
        for part in (role.get("given_name"), role.get("family_name"))
        if part
    ).strip()
    body = str(answer.get("body_text") or "")
    excerpt = body[:1200]
    truncated = len(body) > 1200
    pdf_meta = answer.get("pdf_metadata") or {}
    records = "".join(
        f"<li><code>{_e(row.get('record_locator'))}</code> · sha256 "
        f"{_e((row.get('raw_sha256') or '')[:16])}…</li>"
        for row in (answer.get("source_records") or [])
        if isinstance(row, Mapping)
    )
    pages = "".join(
        f"<li>sivu {_e(page.get('page'))}"
        + (f" · sha256 {_e((page.get('text_sha256') or '')[:16])}…" if page.get("text_sha256") else "")
        + (f" · {_e(page.get('source_locator'))}" if page.get("source_locator") else "")
        + "</li>"
        for page in (answer.get("pages") or [])
        if isinstance(page, Mapping)
    )
    return (
        "<section class='answer-document'>"
        "<h2>Vastausasiakirja (lähdekentät)</h2>"
        f"<p><strong>{_e(answer.get('document_id'))}</strong> · {_e(answer.get('title'))}</p>"
        f"<p>Rooli lähteessä: {_e(role.get('code'))}"
        + (f" · {_e(role.get('position'))}" if role.get("position") else "")
        + (f" · {_e(name)}" if name else "")
        + f" · peruste {_e(role.get('identity_basis') or 'source fields')}</p>"
        f"<p>Laadintapäivä lähteessä: {_e(dates.get('laadintaPvm'))} "
        f"({_e(dates.get('field'))}; {_e(dates.get('basis'))}). "
        "Tallennus-/capture-aika ≠ vastauksen laadintapäivä.</p>"
        + _verification_source_html(
            {"url": answer.get("source_url"), "record_locator": answer.get("document_id")}
        )
        + (
            f"<p class='muted'>Tekstin sha256: <code>{_e(answer.get('body_text_sha256'))}</code>; "
            f"PDF sha256: <code>{_e(answer.get('pdf_sha256'))}</code>"
            + (f"; MD5 <code>{_e(pdf_meta.get('md5'))}</code>" if pdf_meta.get("md5") else "")
            + (
                f"; content-type {_e(pdf_meta.get('content_type'))}"
                if pdf_meta.get("content_type")
                else ""
            )
            + "</p>"
        )
        + (
            (
                f"<details open><summary>Vastauksen ote ({_e(len(excerpt))}/{_e(len(body))} merkkiä; "
                "ei koko teksti)</summary><blockquote>"
                f"{_e(excerpt)}…</blockquote>"
                "<p><a href='#answer-full-body'>Siirry koko vastauksen tekstiin</a> · "
                + _source_link(answer.get("source_url"), "Avaa alkuperäinen PDF")
                + "</p></details>"
                f"<details id='answer-full-body'><summary>Koko vastauksen teksti "
                f"({_e(len(body))} merkkiä)</summary><blockquote>{_e(body)}</blockquote></details>"
            )
            if truncated
            else (
                f"<details><summary>Näytä vastauksen teksti</summary><blockquote>{_e(body)}</blockquote></details>"
                if body
                else ""
            )
        )
        + (f"<ul>{records}</ul>" if records else "")
        + (f"<p>Sivukohtainen provenienssi</p><ul>{pages}</ul>" if pages else "")
        + "</section>"
    )


def _committee_participation_html(section: Mapping | None) -> str:
    if not isinstance(section, Mapping):
        return ""
    rows = [row for row in (section.get("rows") or []) if isinstance(row, Mapping)]
    if not rows:
        return ""
    family_labels = {
        "COMMITTEE_MEMBER": "Jäsen / puheenjohtaja",
        "COMMITTEE_SECRETARY": "Sihteeri / valiokuntaneuvos",
        "EXPERT_OR_HEARING": "Asiantuntija / kuuleminen",
        "OTHER_NAMED_ROLE": "Muu nimetty rooli",
        "DOCUMENTED_PARTICIPATION": "Dokumentoitu osallistuminen",
    }
    items = []
    for row in rows:
        scope = row.get("participation_date_scope") or {}
        doc_dates = ", ".join(
            str(item.get("value") or "")
            for item in (scope.get("document_dates") or [])
            if isinstance(item, Mapping)
        )
        items.append(
            "<li>"
            f"<strong>{_e(row.get('person_name'))}</strong> · "
            f"{_e(family_labels.get(row.get('role_family'), row.get('role_family')))}"
            + (
                f" · lähderooli {_e(', '.join(row.get('source_roles') or []))}"
                if row.get("source_roles")
                else ""
            )
            + f"<br><span class='muted'>{_e(row.get('source_statement'))}</span>"
            + f"<br><span class='muted'>Päiväraja lähteessä: asiakirjapäivä {_e(doc_dates or '—')}; "
            f"kokouspäivä {_e(scope.get('meeting_date_status'))}. "
            "Jäsenyyden koko aikaväliä tai laatijuutta ei päätellä.</span>"
            + (
                f"<br><code>{_e((row.get('source_locator') or {}).get('record_identifier'))}</code>"
                if (row.get("source_locator") or {}).get("record_identifier")
                else ""
            )
            + "</li>"
        )
    return (
        "<section class='committee-participation'>"
        "<h2>Nimetty osallistuminen ja roolit</h2>"
        "<p class='muted'>Lähteen XML-solmut; ei jäsenyyden täyttä aikaväliä eikä "
        "luonnoksen laatijuutta.</p>"
        f"<ul>{''.join(items)}</ul></section>"
    )


def _compact_source_html(
    source: Mapping,
    *,
    record_href: str | None = None,
) -> str:
    """One nearby source affordance; locators stay in technical disclosure."""

    url = str(source.get("url") or source.get("source_url") or "").strip()
    locator = str(source.get("record_locator") or source.get("source_locator") or "").strip()
    if url and not _is_nonspecific_source_url(url):
        link = _source_link(url, "Lähde")
    elif record_href:
        link = f'<a href="{_e(record_href)}">Lähdekirjaus</a>'
    elif locator:
        link = f"osoitin <code>{_e(locator)}</code>"
    else:
        link = '<span class="muted">lähde avoin</span>'
    return f"<p class='record-source'>{link}</p>"


def _intervention_card(
    action: Mapping,
    *,
    index: int,
    record_href: str | None = None,
    compact: bool = True,
    show_actor_name: bool = False,
) -> str:
    """One intervention; compact default keeps exact alternative and source nearby."""

    actor = action.get("actor") or {}
    source = action.get("source") or {}
    name = _display_name(actor)
    action_kind = _action_kind_label(action)
    action_fi = _action_kind_fi(action) or action_kind
    date = action.get("action_date") or "Tekopäivä avoin"
    title = action.get("title") or ""
    influence = action.get("influence") or {}
    influence_state = influence.get("state") or "INFLUENCE_NOT_INFERRED"
    result = action.get("institutional_result") or {}
    result_state = result.get("state")
    ballot_version = action.get("ballot_source_version") or {}
    source_version_unconfirmed = (
        action.get("kind") == "VOTE"
        and ballot_version.get("state") == "NORMALIZED_ROW_NO_RAW_HASH"
    )

    vote_line = ""
    peer_html = ""
    if action.get("kind") == "VOTE":
        response = action.get("raw_response") or "—"
        # Exact alternative stays in the title line; JAA/EI is only the ballot mark.
        vote_line = f"<p class='vote-mark'>Merkintä: <strong>{_e(response)}</strong></p>"
        peer_html = _peer_comparison_html(action.get("own_group_peer_comparison"))

    formal_bits = ""
    if action.get("kind") == "FORMAL_ALTERNATIVE":
        chair = action.get("chair_title_attribution") or {}
        if chair.get("side") or chair.get("slot"):
            formal_bits += (
                f"<p>Julkaistu puoli/paikka: {_e(chair.get('side'))} · {_e(chair.get('slot'))}</p>"
            )
        source_name = actor.get("source_display_name")
        resolved = actor.get("resolved_display_name")
        if source_name and resolved and source_name != resolved:
            formal_bits += (
                f'<p class="muted">Lähteen nimi: {_e(source_name)}; '
                f"rekisterissä: {_e(resolved)}.</p>"
            )

    routine_result = result_state in {None, "", "RECORDED", "VOTE_RECORDED"}
    disposition_default = ""
    if result_state and not routine_result:
        disposition_default = (
            "<p><strong>Kirjaustila:</strong> "
            + _e(_DISPOSITIONS.get(result_state, result_state))
        )
        if result.get("scope"):
            disposition_default += f" · {_e(result.get('scope'))}"
        disposition_default += "</p>"

    source_text = action.get("source_text") or ""
    excerpt = ""
    if source_text and action.get("kind") in {"SPEECH", "GOVERNMENT_ANSWER"}:
        excerpt = f'<blockquote class="source-text">{_e(source_text)}</blockquote>'
    elif source_text:
        excerpt = (
            '<details class="source-text"><summary>Teksti</summary>'
            f"<blockquote>{_e(source_text)}</blockquote></details>"
        )

    audit_bits = [
        f"<p>{_e(action.get('object_id'))} · {_e(actor.get('identity_state'))}</p>",
        (
            f"<p>Tekopäivän peruste: {_e(action.get('action_date_basis'))}. "
            f"Rooliliitoksen tila: {_e((action.get('role_intervals') or {}).get('state'))}.</p>"
        ),
    ]
    if action_kind:
        audit_bits.append(
            f"<p>Toimenpide: {_e(action_fi)}"
            + (f" <span class='muted'>({_e(action_kind)})</span>" if action_kind else "")
            + "</p>"
        )
    if source.get("record_locator"):
        audit_bits.append(
            f"<p>Lähdeosoitin: <code>{_e(source.get('record_locator'))}</code></p>"
        )
    if action.get("evidence_ids"):
        audit_bits.append(f"<p>{_e(', '.join(action.get('evidence_ids', [])))}</p>")
    if result_state:
        audit_bits.append(
            "<p>Kirjaustila (ei institutionaalinen lopputulos): "
            + _e(_DISPOSITIONS.get(result_state, result_state))
            + (f" · {_e(result.get('raw_state'))}" if result.get("raw_state") else "")
            + (f" · laajuus {_e(result.get('scope'))}" if result.get("scope") else "")
            + "</p>"
        )
    if source_version_unconfirmed:
        audit_bits.append(
            "<p>Paikallinen tietokantarivi: alkuperäisen vastauksen "
            "lähdeversio on vahvistamatta.</p>"
        )
    if action.get("kind") == "FORMAL_ALTERNATIVE":
        audit_bits.append(
            "<p>Puhemiehen äänestysotsikko nimeää vaihtoehdon; "
            "se ei osoita tekstin yksityistä laatijaa. "
            "Tämän äänestyksen tulos ei ole lopullinen hyväksyminen ilman erillistä vaihetodistetta.</p>"
        )
    audit_bits.append(
        "<p class='attribution-limit'>Henkilön kirjattu toimi ≠ institutionaalinen päätös. "
        f"Vaikutus, toimeenpano, lopullinen hyväksyminen ja kausaliteetti: "
        f"{_e(influence_state)} / NOT_ASSESSED tässä kirjauksessa.</p>"
    )
    audit_bits.append(_verification_source_html(source, record_href=record_href))
    audit_bits.append(peer_html)

    record_details = (
        "<details class='record-audit'><summary>Tekninen kirjaus</summary>"
        + "".join(audit_bits)
        + _role_interval_html(action.get("role_intervals"))
        + formal_bits
        + "</details>"
    )

    if show_actor_name:
        heading = f"<h3>{_e(name)}"
        if action_fi:
            heading += f" · {_e(action_fi)}"
        heading += "</h3>"
    else:
        heading = ""

    lead = (
        f"<p class='record-lead'>{_e(_short_date(date))}</p>"
    )
    title_line = f"<p class='record-title'>{_e(title)}</p>" if title and action.get("kind") != "SPEECH" else ""
    # Default surface: one nearby source control. Full locator chrome stays in audit.
    source_line = _compact_source_html(source, record_href=record_href)

    return (
        f'<article class="intervention" data-actor="{_e(_actor_key(actor))}" '
        f'data-kind="{_e(action.get("kind"))}" id="record-{index}">'
        + heading
        + lead
        + title_line
        + vote_line
        + disposition_default
        + source_line
        + excerpt
        + record_details
        + "</article>"
    )



def _approval_pair(action: Mapping) -> bool:
    """Select an explicit approval/rejection pairing, not an inferred final stage."""

    parts = [part.strip().casefold() for part in str(action.get("title") or "").split("/")]
    return len(parts) == 2 and set(parts) == {"hyväksyminen", "hylkääminen"}


def _group_exception(action: Mapping) -> bool:
    comparison = action.get("own_group_peer_comparison") or {}
    return comparison.get("status") == "COMPARABLE" and comparison.get("matches_peer_majority") is False


def _short_date(value: object) -> str:
    text = str(value or "—")
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        year, month, day = text.split("-")
        return f"{int(day)}.{int(month)}.{year}"
    return text


def _vote_sources(matter: Mapping) -> dict[str, Mapping]:
    sources: dict[str, Mapping] = {}
    for row in matter.get("vote_sources") or []:
        key = str(row.get("vote_id") or "")
        if not key:
            continue
        if key in sources:
            raise ValueError(f"duplicate vote source: {key}")
        sources[key] = row
    return sources


def _winning_mark_html(vote: Mapping | None) -> str:
    """Project the retained same-vote disposition; never infer enactment."""

    result = (vote or {}).get("published_disposition") or {}
    side = {
        "JAA_SIDE_PREVAILED_AT_THIS_VOTE": "JAA",
        "EI_SIDE_PREVAILED_AT_THIS_VOTE": "EI",
    }.get(result.get("state"))
    if not side:
        return "—"
    counts = ""
    if result.get("jaa") is not None and result.get("ei") is not None:
        counts = f"<small> {_e(result['jaa'])}–{_e(result['ei'])}</small>"
    return _e(side) + counts


def _matter_decision_html(matter: Mapping) -> str:
    rows = []
    for vote in matter.get("vote_sources") or []:
        if not _approval_pair(vote):
            continue
        rows.append(
            f"<tr><td>{_e(_short_date(vote.get('action_date')))}</td>"
            f"<td>{_source_link(vote.get('source_url'), str(vote.get('title') or 'Äänestys'))}</td>"
            f"<td>{_winning_mark_html(vote)}</td></tr>"
        )
    if not rows:
        return ""
    return (
        "<section id='decision-overview'><h2>Hyväksymisäänestys</h2>"
        "<table class='choice-table'><thead><tr><th>Päivä</th><th>Vaihtoehdot · lähde</th><th>Voitti</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
        "<p class='choice-scope'>Tämän äänestyksen tulos; voimaantulo ja vaikutukset erikseen.</p></section>"
    )


def _ballot_table(
    selected: Sequence[tuple[int, Mapping]],
    matter: Mapping,
    records_base: str,
    *,
    show_group: bool = False,
) -> str:
    if not selected:
        return ""
    sources = _vote_sources(matter)
    rows = []
    for index, action in selected:
        source = sources.get(str(action.get("event_id") or ""))
        # A shared ID with a different date is not the same retained occasion.
        if source and source.get("action_date") != action.get("action_date"):
            source = None
        comparison = action.get("own_group_peer_comparison") or {}
        group = (
            f"<small> · ryhmä {_e(comparison.get('peer_majority'))}</small>"
            if show_group and _group_exception(action)
            else ""
        )
        href = f"{records_base}{record_page_slug(index, action)}.html"
        rows.append(
            f'<tr class="ballot-row" id="record-{index}" data-kind="VOTE">'
            f"<td>{_e(_short_date(action.get('action_date')))}</td>"
            f'<td><a href="{_e(href)}">{_e(action.get("title") or "Äänestys")}</a></td>'
            f"<td><strong>{_e(action.get('raw_response') or '—')}</strong>{group}</td>"
            f"<td>{_winning_mark_html(source)}</td></tr>"
        )
    return (
        "<table class='choice-table'><thead><tr><th>Päivä</th><th>Vaihtoehdot · lähde</th>"
        "<th>Hän</th><th>Voitti</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def _speech_records_html(selected: Sequence[tuple[int, Mapping]], records_base: str) -> str:
    """A speech index is evidence to read, not a manufactured position answer."""

    rows = []
    for index, action in selected:
        text = str(action.get("source_text") or "").strip()
        # Exact opening text, not a generated summary or a stance classification.
        excerpt = text[:180] + ("…" if len(text) > 180 else "")
        href = f"{records_base}{record_page_slug(index, action)}.html"
        rows.append(
            f'<li><a href="{_e(href)}">{_e(_short_date(action.get("action_date")))}</a>'
            + (f" · <q>{_e(excerpt)}</q>" if excerpt else " · teksti puuttuu")
            + "</li>"
        )
    if not rows:
        return ""
    return (
        "<details class='speech-index'><summary>Mitä hän sanoi? · Puheet</summary>"
        "<p>Alkuperäiset puheet. Kannasta ei ole koottua tulkintaa.</p>"
        "<ul class='source-list'>" + "".join(rows) + "</ul></details>"
    )


def _person_choices_html(selected: Sequence[tuple[int, Mapping]], matter: Mapping, records_base: str) -> str:
    votes = [(index, row) for index, row in selected if row.get("kind") == "VOTE"]
    primary = [(index, row) for index, row in votes if _approval_pair(row) or _group_exception(row)]
    if not primary and len(votes) <= 3:
        primary = votes
    primary_indexes = {index for index, _ in primary}
    other_votes = [(index, row) for index, row in votes if index not in primary_indexes]
    parts = []
    if primary:
        parts.append("<section class='personal-choices'><h2>Miten äänesti?</h2>")
        parts.append(_ballot_table(primary, matter, records_base, show_group=True))
        parts.append("<p class='choice-scope'>Voitti = tämän äänestyksen tulos.</p></section>")
    proposals = [(index, row) for index, row in selected if row.get("kind") in {
        "FORMAL_ALTERNATIVE", "LEGISLATIVE_INITIATIVE", "WRITTEN_QUESTION", "GOVERNMENT_ANSWER",
    }]
    if proposals:
        rows = []
        for index, row in proposals:
            href = f"{records_base}{record_page_slug(index, row)}.html"
            result = row.get("institutional_result") or {}
            state = result.get("state")
            result_label = _DISPOSITIONS.get(state, "") if state not in {"RECORDED", "VOTE_RECORDED"} else ""
            rows.append(
                f"<tr><td>{_e(_short_date(row.get('action_date')))}</td>"
                f'<td><a href="{_e(href)}">{_e(row.get("title") or row.get("object_id") or "Lähde")}</a>'
                f"<small> · {_e(_action_kind_fi(row))}</small></td>"
                f"<td>{_e(result_label)}</td></tr>"
            )
        parts.append(
            "<section><h2>Muut teot</h2><table class='choice-table'>"
            "<thead><tr><th>Päivä</th><th>Teko · lähde</th><th>Tulos</th></tr></thead><tbody>"
            + "".join(rows) + "</tbody></table></section>"
        )
    if other_votes:
        parts.append(
            "<details class='other-ballots'><summary>Muut äänestykset</summary>"
            + _ballot_table(other_votes, matter, records_base)
            + "</details>"
        )
    speech = [(index, row) for index, row in selected if row.get("kind") == "SPEECH"]
    parts.append(_speech_records_html(speech, records_base))
    remainder = [(index, row) for index, row in selected if row.get("kind") not in {
        "VOTE", "SPEECH", "FORMAL_ALTERNATIVE", "LEGISLATIVE_INITIATIVE", "WRITTEN_QUESTION", "GOVERNMENT_ANSWER",
    }]
    if remainder:
        links = "".join(
            f'<li><a href="{_e(records_base + record_page_slug(index, row) + ".html")}">'
            f"{_e(_short_date(row.get('action_date')))} · {_e(_action_kind_fi(row))}</a></li>"
            for index, row in remainder
        )
        parts.append("<details><summary>Muut lähteet</summary><ul>" + links + "</ul></details>")
    if not selected:
        parts.append("<p>Tässä aineistossa ei ole henkilön tekoja. Muut kanavat voivat puuttua.</p>")
    return "".join(parts)


def render_matter_person_html(
    packet: Mapping,
    actor_key: str,
    *,
    packet_url: str | None = None,
    hub_url: str = "../index.html",
    records_base: str = f"../{RECORDS_DIR_NAME}/",
    actions: Sequence[Mapping] | None = None,
    indexes: Sequence[int] | None = None,
) -> str:
    """Answer a personal-choice question; retain the record archive separately."""

    matter = packet["matter"]
    all_actions = list(packet.get("interventions") or [])
    if actions is None or indexes is None:
        selected: list[tuple[int, Mapping]] = [
            (index, action)
            for index, action in enumerate(all_actions)
            if _actor_key(action.get("actor") or {}) == actor_key
        ]
    else:
        selected = list(zip(indexes, actions, strict=True))
    subject = resolve_matter_subject_title(matter)
    matter_id = matter["matter_id"]
    name = _display_name((selected[0][1].get("actor") or {}) if selected else {"name": actor_key})
    page_title = f"{name} · {matter_id}"
    choices = _person_choices_html(selected, matter, records_base)
    has_unconfirmed_ballot = any(
        action.get("kind") == "VOTE"
        and (action.get("ballot_source_version") or {}).get("state")
        == "NORMALIZED_ROW_NO_RAW_HASH"
        for _, action in selected
    )
    subject_line = ""
    if subject.get("subject_title"):
        subject_line = f"<p class='matter-subject'>{_e(subject['subject_title'])}</p>"
    elif subject.get("subject_title_state") == "AMBIGUOUS_OBJECT_TITLES":
        subject_line = (
            "<p class='matter-subject muted'>Lähdeobjektien otsikot eivät ole yhteneviä; "
            "asiatunniste säilyy valinnan avaimena.</p>"
        )
    reliability = (
        "<p class='muted scope-limit'>Osa äänestysmerkinnöistä perustuu paikalliseen "
        "tietokantariviin; alkuperäisen vastauksen lähdeversio on vahvistamatta. "
        "Merkintä koskee otsikon vaihtoehtoa, ei automaattisesti koko lakipakettia.</p>"
        if has_unconfirmed_ballot
        else "<p class='muted scope-limit'>Äänestysmerkintä koskee otsikon vaihtoehtoa, "
        "ei automaattisesti koko lakipakettia. Kirjattu toimi ≠ institutionaalinen päätös.</p>"
    )
    body = (
        f'<p><a href="{_e(hub_url)}">Takaisin asian yhteenvetoon</a></p>'
        f"<h1>{_e(name)} · {_e(matter_id)}</h1>"
        + _subject_heading(matter)
        + f'<div id="person-records">{choices}</div>'
        + "<details class='person-evidence'><summary>Aineisto</summary>"
        + subject_line
        + reliability
        + _limits_section(packet)
        + "</details>"
    )
    if packet_url:
        body += f'<p><a download href="{_e(packet_url)}">Lataa kirjausten lähdepaketti (JSON)</a></p>'
    body += (
        "<style>" + _CHOICE_CSS
        + ".source-list{padding-left:1.2rem}.source-list li{margin:.4rem 0}"
        + "</style>"
    )
    return _document(
        page_title,
        body,
        description="Henkilön asiatunnisteeseen yhdistetyt julkiset kirjaukset ja niiden lähteet.",
    )


def render_matter_record_html(
    packet: Mapping,
    index: int,
    action: Mapping,
    *,
    packet_url: str | None = None,
    hub_url: str = "../index.html",
    person_url: str | None = None,
) -> str:
    """One supporting record page; lead facts compact, audit behind disclosure."""

    matter = packet["matter"]
    subject = resolve_matter_subject_title(matter)
    matter_id = matter["matter_id"]
    actor = action.get("actor") or {}
    name = _display_name(actor)
    page_title = f"Kirjaus {index + 1} · {name} · {matter_id}"
    nav = f'<p><a href="{_e(hub_url)}">Asian yhteenveto</a>'
    if person_url:
        nav += f' · <a href="{_e(person_url)}">Henkilön kirjaukset</a>'
    nav += "</p>"
    subject_line = (
        f"<p class='matter-subject'>{_e(subject['subject_title'])}</p>"
        if subject.get("subject_title")
        else ""
    )
    body = (
        nav
        + f"<h1>{_e(name)} · {_e(matter_id)}</h1>"
        + f"<p class='muted'>Kirjaus {index + 1}</p>"
        + _intervention_card(
            action,
            index=index,
            compact=False,
            show_actor_name=False,
        )
    )
    if packet_url:
        body += f'<p><a download href="{_e(packet_url)}">Lataa kirjausten lähdepaketti (JSON)</a></p>'
    if subject_line:
        body += f"<details><summary>Asian nimi</summary>{subject_line}</details>"
    body += "<style>article.intervention{padding:0;border:0;background:none;margin:0}</style>"
    return _document(
        page_title,
        body,
        description="Yksittäinen asiatunnisteeseen sidottu lähdekirjaus.",
    )


def render_matter_html(
    packet: Mapping,
    *,
    packet_url: str | None = None,
    people_base: str = f"{PEOPLE_DIR_NAME}/",
    back_url: str | None = None,
    lookup_url: str | None = None,
) -> str:
    """Compact matter hub; full records live on per-person static pages."""

    query = packet["query"]
    if query["mode"] != "EXACT_MATTER":
        candidates = "".join(
            "<article><h2>"
            + _e(row["matter_id"])
            + "</h2>"
            + "".join(f"<p>{_e(record['title'])}</p>" for record in row["records"])
            + "</article>"
            for row in packet["candidates"]
        )
        return _document(
            "Etsi asiaa",
            "<h1>Etsi asiaa</h1><p>Haku: "
            + _e(query["text"])
            + "</p>"
            "<p>Nämä ovat tekstiosumia. Valitse asian tunniste ja tee täsmällinen kysely "
            "nähdäksesi henkilöiden kirjaukset.</p>"
            + candidates,
        )
    matter = packet["matter"]
    actions = list(packet.get("interventions") or [])
    summaries = compact_actor_summaries(actions)
    selected = query.get("actor")
    selected_key = _actor_key(selected) if selected else "choose"
    options = '<option value="choose">Valitse henkilö</option>'
    options += "".join(
        f'<option value="{_e(row["actor_key"])}" data-href="{_e(people_base + row["slug"] + ".html")}"'
        + (" selected" if row["actor_key"] == selected_key else "")
        + f'>{_e(row["name"])}</option>'
        for row in summaries
    )
    subject = resolve_matter_subject_title(matter)
    matter_id = matter["matter_id"]
    # Document title and visible H1 stay short; official title is secondary once.
    doc_title = f"{matter_id}: mitä henkilöt tekivät?"
    selected_note = ""
    if selected:
        selected_note = (
            f"<p>Valittu toimija: {_e(selected.get('display_name') or selected.get('person_id') or selected.get('actor_id'))}."
            + (
                f' <a href="{_e(people_base + actor_page_slug(selected_key) + ".html")}">'
                "Avaa henkilön kirjaukset</a>."
                if selected_key != "choose"
                else ""
            )
            + "</p>"
        )
        if selected.get("unresolved"):
            selected_note += (
                "<p>Toimijan identiteetti jäi avoimeksi. Tyhjä tulos ei osoita, "
                "ettei hän olisi toiminut asiassa.</p>"
            )
        elif not actions:
            selected_note += (
                "<p>Tästä aineistosta ei löytynyt valitun henkilön kirjausta. "
                "Tämä ei ole kattava poissaolo- tai tekemättä jättämisen päätelmä.</p>"
            )
    subject_block = ""
    if subject.get("subject_title"):
        subject_block = f"<p class='matter-subject'>{_e(subject['subject_title'])}</p>"
    elif subject.get("subject_title_state") == "AMBIGUOUS_OBJECT_TITLES":
        subject_block = (
            "<p class='matter-subject muted'>Lähdeobjektien otsikot eivät ole yhteneviä; "
            "asiatunniste säilyy valinnan avaimena.</p>"
        )
    actor_links = "".join(
        f'<li><a href="{_e(people_base + row["slug"] + ".html")}">{_e(row["name"])}</a></li>'
        for row in summaries
    )
    nav_bits: list[str] = []
    if back_url:
        nav_bits.append(f"<a href='{_e(back_url)}'>Takaisin</a>")
    if lookup_url:
        nav_bits.append(f"<a href='{_e(lookup_url)}'>Haku</a>")
    nav = f"<p>{' · '.join(nav_bits)}</p>" if nav_bits else ""
    answer_html = _answer_document_html(packet.get("answer_document"))
    committee_html = _committee_participation_html(packet.get("committee_named_participation"))
    body = (
        nav
        + f"<h1>{_e(matter_id)}</h1>"
        + _subject_heading(matter)
        + selected_note
        + "<section class='person-choice'>"
        + '<label for="actor-filter">Henkilö</label> '
        f'<select id="actor-filter">{options}</select>'
        '<p id="record-count" role="status" aria-live="polite"></p>'
        '<div id="person-panel" class="person-panel" hidden></div>'
        "<noscript><p><a href='#henkilot'>Henkilön staattinen sivu: avaa henkilöluettelo</a></p></noscript></section>"
        + _matter_decision_html(matter)
        + _vote_stage_html(matter.get("vote_sources") or [], collapsed=True)
        + "<details class='matter-evidence'><summary>Aineisto</summary>"
        + subject_block
        + _stage_summary_html(matter, actions)
        + (
            "<details><summary>Vastausasiakirjan lähdekentät</summary>"
            + answer_html
            + "</details>"
            if answer_html
            else ""
        )
        + (
            "<details class='committee'><summary>Valiokunnan asiakirjoihin nimetyt henkilöt</summary>"
            + committee_html
            + "</details>"
            if committee_html
            else ""
        )
        + _limits_section(packet)
        + "</details>"
        + "<details id='henkilot' class='actor-directory'><summary>Henkilöt</summary>"
        f"<ul class='person-links'>{actor_links}</ul></details>"
        + "<script>"
        + _HUB_JS
        + "</script>"
    )
    if packet_url:
        body += f'<p><a download href="{_e(packet_url)}">Lataa kirjausten lähdepaketti (JSON)</a></p>'
    body += (
        "<style>select{max-width:100%;font:inherit;margin:.5rem 0;display:block}"
        ".person-panel{margin:.8rem 0}.person-links{padding-left:1.2rem}"
        + _CHOICE_CSS
        + ".committee:empty,#record-count:empty{display:none}</style>"
    )
    return _document(
        doc_title,
        body,
        description="Asiatunnisteeseen yhdistettyjen henkilöiden tiivis yhteenveto; täydet kirjaukset henkilösivuilla.",
    )



def write_matter_site(
    packet: Mapping,
    hub_path: Path,
    *,
    packet_url: str | None = None,
    back_url: str | None = None,
    lookup_url: str | None = None,
) -> dict[str, Any]:
    """Write hub + per-person + per-record static pages under ``{hub.stem}/``."""

    hub_path = Path(hub_path)
    hub_path.parent.mkdir(parents=True, exist_ok=True)
    shard_root = hub_path.parent / hub_path.stem
    people_dir = shard_root / PEOPLE_DIR_NAME
    records_dir = shard_root / RECORDS_DIR_NAME
    people_dir.mkdir(parents=True, exist_ok=True)
    records_dir.mkdir(parents=True, exist_ok=True)
    actions = list(packet.get("interventions") or [])
    summaries = compact_actor_summaries(actions)
    hub_packet_url = packet_url
    # Person/record pages live two directory levels under the hub's directory.
    person_packet_url = _packet_url_for_depth(packet_url, depth=2)
    hub_from_person = f"../../{hub_path.name}"
    people_base = f"{hub_path.stem}/{PEOPLE_DIR_NAME}/"
    hub_path.write_text(
        render_matter_html(
            packet,
            packet_url=hub_packet_url,
            people_base=people_base,
            back_url=back_url,
            lookup_url=lookup_url,
        ),
        encoding="utf-8",
    )
    person_files: list[str] = []
    record_files: list[str] = []
    for summary in summaries:
        person_name = f"{summary['slug']}.html"
        person_path = people_dir / person_name
        selected_actions = [actions[index] for index in summary["record_indexes"]]
        person_path.write_text(
            render_matter_person_html(
                packet,
                summary["actor_key"],
                packet_url=person_packet_url,
                hub_url=hub_from_person,
                records_base=f"../{RECORDS_DIR_NAME}/",
                actions=selected_actions,
                indexes=summary["record_indexes"],
            ),
            encoding="utf-8",
        )
        shard = {
            "schema_version": "paa.matter-person-shard.v1",
            "matter_id": (packet.get("matter") or {}).get("matter_id"),
            "actor_key": summary["actor_key"],
            "summary": {
                key: summary[key]
                for key in (
                    "name",
                    "record_count",
                    "kinds",
                    "roles",
                    "action_kinds",
                    "earliest_action_date",
                    "latest_action_date",
                )
            },
            "interventions": selected_actions,
        }
        (people_dir / f"{summary['slug']}.json").write_text(
            json.dumps(shard, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        person_files.append(f"{hub_path.stem}/{PEOPLE_DIR_NAME}/{person_name}")
        for index, action in zip(summary["record_indexes"], selected_actions, strict=True):
            record_name = f"{record_page_slug(index, action)}.html"
            (records_dir / record_name).write_text(
                render_matter_record_html(
                    packet,
                    index,
                    action,
                    packet_url=person_packet_url,
                    hub_url=hub_from_person,
                    person_url=f"../{PEOPLE_DIR_NAME}/{person_name}",
                ),
                encoding="utf-8",
            )
            record_files.append(f"{hub_path.stem}/{RECORDS_DIR_NAME}/{record_name}")
    return {
        "hub": hub_path.name,
        "shard_root": hub_path.stem,
        "people": person_files,
        "records": record_files,
        "actor_count": len(summaries),
        "intervention_count": len(actions),
    }


def render_matter_index(entries: Sequence[Mapping]) -> str:
    rows = "".join(
        "<li><a href='" + _e(entry["path"]) + "'>"
        + _e(entry.get("choice_label") or entry["title"])
        + "</a></li>"
        for entry in entries
    )
    return _document(
        "Asiat", "<h1>Asiat</h1><ul class='matter-links'>" + rows + "</ul>",
        description="Valitse asia ja henkilö nähdäksesi tämän dokumentoidut valinnat.",
    )


__all__ = [
    "PEOPLE_DIR_NAME",
    "RECORDS_DIR_NAME",
    "actor_page_slug",
    "compact_actor_summaries",
    "matter_choice_label",
    "render_matter_html",
    "render_matter_index",
    "render_matter_person_html",
    "render_matter_record_html",
    "resolve_matter_subject_title",
    "write_matter_site",
]

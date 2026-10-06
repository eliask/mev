"""Finnish static pages for canonical decision episodes.

The renderer is intentionally downstream of :mod:`paa.decision_episodes`.
It formats source objects and their evidence; it does not join objects,
interpret a response as implementation, or infer an actor from a name.
The output is suitable for a page written below ``dist/episodes/``.  The
caller writes the episode JSON beside the HTML so the download link remains
an auditable, immutable page-level export.
"""


import html
import json
from collections.abc import Iterable, Mapping
from typing import Any
from urllib.parse import quote, urlparse

_STAGE_LABELS = {
    "QUESTION_DOCUMENT": "Kirjallinen kysymys rekisteröity asiakirjana",
    "QUESTION_SUBMITTED": "Kirjallinen kysymys jätetty",
    "GOVERNMENT_RESPONSE_DOCUMENT": "Hallituksen vastausobjekti rekisteröity",
    "GOVERNMENT_RESPONSE_RECEIVED": "Hallituksen vastaus annettu",
    "GOVERNMENT_RESPONSE_ANNOUNCED": "Vastaus ilmoitettu täysistunnossa",
}
_DATE_BASIS_LABELS = {
    "PUBLICATION_DATE": "julkaisupäivä",
    "SOURCE_EVENT": "lähteen käsittelytapahtuma",
    "SUBMISSION_DATE": "kysymyksen jättämismerkintä",
    "ANSWER_EVENT": "vastauksen tapahtumamerkintä",
    "SIGNATURE_DATE": "allekirjoituspäivä",
    "UNRESOLVED": "päivä avoin",
}
_ROLE_LABELS = {
    "AUTHOR": "Ensimmäinen allekirjoittaja / lähteessä nimetty laatija",
    "COSIGNER": "Muu allekirjoittaja",
    "RESPONDENT": "Vastauksen lähteessä nimetty vastaaja",
    "ACTOR": "Lähteessä nimetty toimija",
    "UNRESOLVED": "Toimijan rooli avoin",
}
_STATE_LABELS = {"ANSWERED_INSTITUTIONALLY": "Vastaus kirjattu käsittelytapahtumassa",
                 "ANSWER_OBJECT_PRESENT_UNRESOLVED": "Vastausasiakirja löytyy, käsittelytapahtuma avoin",
                 "OPEN_QUESTION": "Kysymys kirjattu, vastauksen lähde jää avoimeksi", "UNRESOLVED": "Käsittely jää avoimeksi"}
_RESIDUAL_LABELS = {
    "POLICY_IMPLEMENTATION_NOT_SOURCED": "Politiikan toteutusta ei ole osoitettu tästä lähteestä.",
    "CAUSAL_OUTCOME_NOT_SOURCED": "Seurausta tai syy-yhteyttä ei ole osoitettu tästä lähteestä.",
    "LEGAL_EFFECT_NOT_ASSESSED": "Lain voimaantuloa tai muuta oikeusvaikutusta ei arvioida tästä aineistosta.",
    "ANSWER_OBJECT_NOT_LOADED": "Erillistä vastausobjektia ei ole liitetty tähän tapaukseen.",
    "ANSWER_EVENT_NOT_FOUND": "Vastauksen nimenomaista tapahtumamerkintää ei ole tässä lähdeotteessa.",
    "RESPONDENT_IDENTITY_UNRESOLVED": "Vastaajan henkilötunniste jäi tämän lähteen perusteella avoimeksi.",
    "SOURCE_COVERAGE_NOT_ATTACHED": "Lähderekisterin kattavuustodistetta ei ole liitetty.",
    "SOURCE_COVERAGE_LIMITED": "Lähderekisterin rajaus tai päällekkäisyys rajoittaa kattavuuspäätelmää.",
    "SOURCE_SEQUENCE_EMPTY": "Lähdesekvenssiin ei saatu lähderiviä tai tapahtumaa.",
}


def _e(value: Any) -> str:
    return html.escape(str(value or ""), quote=True)


def _safe(value: str) -> str:
    """Match ``paa.site._safe`` locally without importing the site module."""

    return "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in str(value))[:160]


def _href_slug(value: str) -> str:
    return quote(_safe(value), safe="-_")


def _http_url(value: Any) -> str:
    parsed = urlparse(str(value or ""))
    if parsed.scheme in {"http", "https"} and parsed.netloc:
        return str(value)
    return ""


def _unique(values: Iterable[Any]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value is None:
            continue
        item = str(value)
        if item and item not in seen:
            output.append(item)
            seen.add(item)
    return output


def _unique_nested(values: Iterable[Iterable[Any]]) -> list[str]:
    return _unique(item for group in values for item in group)


def _evidence_map(evidence: Mapping[str, Mapping[str, Any]] | Iterable[Mapping[str, Any]] | None) -> dict[str, dict[str, Any]]:
    if evidence is None:
        return {}
    items = evidence.values() if isinstance(evidence, Mapping) else evidence
    return {
        str(item["evidence_id"]): dict(item)
        for item in items
        if isinstance(item, Mapping) and item.get("evidence_id")
    }


def _object_map(object_ids: Mapping[str, Any] | Iterable[str] | None) -> set[str]:
    if object_ids is None:
        return set()
    if isinstance(object_ids, Mapping):
        return {str(key) for key in object_ids}
    return {str(item) for item in object_ids}


def _source_link(evidence_id: str, refs: Mapping[str, Mapping[str, Any]]) -> str:
    ref = refs.get(evidence_id) or {}
    url = _http_url(ref.get("url") or ref.get("source_url"))
    label = _e(evidence_id)
    if url:
        return f'<a href="{_e(url)}">{label}</a>'
    return label


def _source_list(evidence_ids: Iterable[str], refs: Mapping[str, Mapping[str, Any]]) -> str:
    ids = _unique(evidence_ids)
    if not ids:
        return '<span class="muted">Lähdetunniste puuttuu.</span>'
    return "<ul class=\"evidence-list\">" + "".join(
        f"<li>{_source_link(item, refs)}</li>" for item in ids
    ) + "</ul>"


def _matter_title(episode: Mapping[str, Any]) -> str:
    matter_id = str(episode.get("matter_id") or "Tuntematon asia")
    question = next(
        (item for item in episode.get("source_objects") or [] if item.get("kind") == "WRITTEN_QUESTION"),
        None,
    )
    title = str((question or {}).get("title") or "").strip()
    if not title or title == matter_id:
        title = "Kirjallinen kysymys"
    return f"{title} ({matter_id})"


def _profile_href(actor_id: Any, actor_names: Mapping[str, Any] | None) -> str:
    if not actor_id:
        return ""
    actor_key = str(actor_id)
    if actor_names is not None and actor_key not in actor_names:
        return ""
    return f"../index.html#person={_href_slug(actor_key)}"


def _actor_name(actor: Mapping[str, Any], actor_names: Mapping[str, Any] | None) -> str:
    name = str(actor.get("name") or "").strip()
    if name:
        return name
    actor_id = actor.get("actor_id")
    if actor_id and actor_names and actor_names.get(str(actor_id)):
        return str(actor_names[str(actor_id)])
    return "Nimi puuttuu lähteestä"


def _actor_html(actor: Mapping[str, Any], actor_names: Mapping[str, Any] | None) -> str:
    name = _e(_actor_name(actor, actor_names))
    profile = _profile_href(actor.get("actor_id"), actor_names)
    if profile:
        name = f'<a href="{_e(profile)}">{name}</a>'
    role = str(actor.get("role") or "UNRESOLVED")
    role_label = _e(_ROLE_LABELS.get(role, role))
    identity = str(actor.get("identity_basis") or "UNRESOLVED")
    note = ""
    if role == "AUTHOR":
        note = "Rekisterin allekirjoittaja tai nimetty laatija ei yksin osoita, kuka tekstin kirjoitti."
    elif role == "RESPONDENT" and identity == "SOURCE_NAME_ONLY":
        note = "Lähteessä on nimi, mutta ei tämän tietueen henkilötunnistetta."
    elif identity not in {"SOURCE_PERSON_ID", "INDEPENDENT_CASE_REVIEW"}:
        note = f"Tunnistusperuste: {identity}."
    return f"<tr><th>{name}</th><td>{role_label}</td><td>{_e(note)}</td></tr>"


def _actor_refs_html(
    refs: Iterable[Mapping[str, Any]],
    actor_names: Mapping[str, Any] | None,
) -> str:
    values: list[str] = []
    for actor in refs:
        if not isinstance(actor, Mapping):
            continue
        name = _e(_actor_name(actor, actor_names))
        profile = _profile_href(actor.get("actor_id"), actor_names)
        if profile:
            name = f'<a href="{_e(profile)}">{name}</a>'
        role = str(actor.get("role") or "UNRESOLVED")
        identity = str(actor.get("identity_basis") or "UNRESOLVED")
        identity_note = ""
        if role == "RESPONDENT" and identity == "SOURCE_NAME_ONLY":
            identity_note = " · nimi ilman lähteen henkilötunnistetta"
        values.append(
            f'<span class="actor-ref"><strong>{name}</strong> '
            f'<span class="muted">({_e(_ROLE_LABELS.get(role, role))}{_e(identity_note)})</span></span>'
        )
    if not values:
        return '<span class="muted">Ei henkilötoimijaa; toimielin näkyy erillisenä.</span>'
    return " ".join(values)


def _date_cell(row: Mapping[str, Any]) -> str:
    value = row.get("date")
    if not value:
        return '<span class="muted">Päivä avoin</span>'
    basis = _DATE_BASIS_LABELS.get(str(row.get("date_basis") or ""), str(row.get("date_basis") or "lähdemerkintä"))
    return f'<time datetime="{_e(value)}">{_e(value)}</time><br><span class="muted">{_e(basis)}</span>'


def _object_href(object_id: Any, available_objects: set[str]) -> str:
    value = str(object_id or "")
    if not value or value not in available_objects:
        return ""
    return f"../objects/{_href_slug(value)}.html"


def _episode_evidence_ids(episode: Mapping[str, Any]) -> list[str]:
    values: list[Any] = list(episode.get("evidence_ids") or [])
    values.extend(
        evidence_id
        for row in episode.get("source_sequence") or []
        for evidence_id in row.get("evidence_ids") or []
    )
    values.extend(
        evidence_id
        for row in episode.get("source_records") or []
        for evidence_id in row.get("evidence_ids") or []
    )
    return _unique(values)


def _evidence_cards(episode: Mapping[str, Any], refs: Mapping[str, Mapping[str, Any]]) -> str:
    cards: list[str] = []
    for evidence_id in _episode_evidence_ids(episode):
        ref = refs.get(evidence_id) or {}
        url = _http_url(ref.get("url") or ref.get("source_url"))
        source_hash = ref.get("raw_sha256") or ref.get("source_raw_sha256") or ""
        link = _source_link(evidence_id, refs)
        quote = ref.get("quote")
        url_html = _e(url) if url else '<span class="muted">ei URL-tietoa</span>'
        quote_html = f"<blockquote>{_e(quote)}</blockquote>" if quote else '<p class="muted">Poimintaa ei ole liitetty.</p>'
        cards.append(
            "<article class=\"evidence-card\">"
            f"<p><strong>{link}</strong></p>"
            f"<p>URL: {url_html}</p>"
            f"<p>Raakalähteen SHA-256: <code>{_e(source_hash) if source_hash else 'ei ilmoitettu'}</code></p>"
            f"<p>Paikka: <code>{_e(ref.get('record_locator') or ref.get('document_version_id'))}</code> · "
            f"kenttä <code>{_e(ref.get('field_path'))}</code></p>"
            f"{quote_html}</article>"
        )
    if not cards:
        return '<p class="muted">Tälle sivulle ei ole liitetty lähdetunnisteita.</p>'
    return "".join(cards)


def _source_records_table(
    episode: Mapping[str, Any],
    available_objects: set[str],
) -> str:
    rows: list[str] = []
    for record in episode.get("source_records") or []:
        object_links: list[str] = []
        for object_id in record.get("object_ids") or []:
            href = _object_href(object_id, available_objects)
            label = _e(object_id)
            object_links.append(f'<a href="{_e(href)}">{label}</a>' if href else label)
        rows.append(
            "<tr>"
            f"<td><code>{_e(record.get('record_locator'))}</code></td>"
            f"<td>{_e(record.get('record_class') or 'lähderivi')}</td>"
            f"<td>{_e(record.get('publication_date') or 'päivä avoin')}</td>"
            f"<td>{'<br>'.join(object_links) or '—'}</td>"
            f"<td><code>{_e(record.get('raw_sha256'))}</code></td>"
            f"<td>{_e(record.get('event_count', 0))}</td>"
            "</tr>"
        )
    if not rows:
        return '<p class="muted">Lähderivejä ei ole liitetty.</p>'
    return (
        '<table><thead><tr><th>Lähdepaikka</th><th>Tyyppi</th><th>Päivä</th><th>Objekti</th>'
        '<th>Raakalähteen SHA-256</th><th>Tapahtumia</th></tr></thead><tbody>'
        + "".join(rows)
        + "</tbody></table>"
    )


def _answer_body_note(episode: Mapping[str, Any]) -> str:
    answer = next(
        (item for item in episode.get("source_objects") or [] if item.get("kind") == "GOVERNMENT_ANSWER"),
        None,
    )
    if answer is None:
        return '<p class="notice">Erillistä vastausobjektia ei ole liitetty. Tämä ei oikeuta päättelemään, ettei vastausta olisi annettu.</p>'
    has_content = any(
        record.get("record_class") == "CONTENT"
        and str(record.get("document_id") or "") == str(answer.get("document_id") or "")
        for record in answer.get("source_records") or []
    )
    if not has_content:
        return (
            '<p class="notice"><strong>Vastaus on rekisterissä ilman vastaustekstiä.</strong> '
            'Tämä sivu näyttää vastausobjektin ja käsittelymerkinnät, mutta ei merkitse niiden sisältöä '
            'sisällöllisesti varmennetuksi.</p>'
        )
    return '<p class="muted">Vastausteksti on mukana tämän lähdeotteen sisältörivissä.</p>'


def _unknowns_html(episode: Mapping[str, Any]) -> str:
    residuals = list(episode.get("residuals") or [])
    items: list[str] = []
    for residual in residuals:
        code = str(residual.get("code") or "UNKNOWN") if isinstance(residual, Mapping) else "UNKNOWN"
        reason = residual.get("reason") if isinstance(residual, Mapping) else ""
        text = _RESIDUAL_LABELS.get(code, str(reason or code))
        if reason and text == code:
            text = str(reason)
        items.append(f"<li><strong>{_e(text)}</strong><details><summary>Tietueen rajaus</summary><code>{_e(code)}</code></details></li>")
    # These are deliberately explicit renderer-level residuals: the episode
    # schema must not invent public constraints or alternative actions merely
    # because they are absent from this source surface.
    items.extend([
        (
            '<li><strong>Julkisia toimintaedellytyksiä tai rajoitteita ei johdeta tästä jaksosta.</strong> '
            '<details><summary>Tietueen rajaus</summary><code>PUBLIC_CONSTRAINTS_NOT_SOURCED</code></details></li>'
        ),
        (
            '<li><strong>Muita mahdollisia toimia tai vaihtoehtoista tapahtumapolkua ei päätellä.</strong> '
            '<details><summary>Tietueen rajaus</summary><code>ACTION_ALTERNATIVES_NOT_SOURCED</code></details></li>'
        ),
    ])
    return "<ul class=\"unknowns\">" + "".join(items) + "</ul>"


def render_episode(
    episode: Mapping[str, Any],
    evidence: Mapping[str, Mapping[str, Any]] | Iterable[Mapping[str, Any]] | None = None,
    object_ids: Mapping[str, Any] | Iterable[str] | None = None,
    actor_names: Mapping[str, Any] | None = None,
) -> str:
    """Render one canonical episode as a Finnish, source-linked HTML page.

    ``object_ids`` is an allow-list: only objects known to be written by the
    caller receive local links.  Likewise, actor profile links are emitted
    only for actor IDs present in ``actor_names`` when that mapping is passed.
    """

    refs = _evidence_map(evidence)
    available_objects = _object_map(object_ids)
    title = _matter_title(episode)
    slug = _href_slug(str(episode.get("episode_id") or episode.get("matter_id") or "episode"))
    state = str(episode.get("episode_state") or "UNRESOLVED")
    question = next(
        (item for item in episode.get("source_objects") or [] if item.get("kind") == "WRITTEN_QUESTION"),
        None,
    )
    answer = next(
        (item for item in episode.get("source_objects") or [] if item.get("kind") == "GOVERNMENT_ANSWER"),
        None,
    )
    actor_rows = "".join(_actor_html(actor, actor_names) for actor in episode.get("actors") or [])
    if not actor_rows:
        actor_rows = '<tr><td colspan="3" class="muted">Lähteessä ei ole nimettyä henkilötoimijaa.</td></tr>'

    sequence_rows: list[str] = []
    for row in sorted(episode.get("source_sequence") or [], key=lambda item: int(item.get("sequence", 999999))):
        stage = str(row.get("stage") or "UNRESOLVED")
        label = _STAGE_LABELS.get(stage, stage)
        actor_refs = _actor_refs_html(row.get("actor_refs") or [], actor_names)
        notice = ""
        if row.get("content_state") == "REGISTERED_RECORD_WITHOUT_BODY":
            notice = '<p class="notice">Vastausobjekti on rekisteröity, mutta vastausteksti ei ole tässä lähdeotteessa.</p>'
        sequence_rows.append(
            "<li class=\"timeline-item\">"
            f"<div class=\"timeline-date\">{_date_cell(row)}</div>"
            f"<h3>{_e(label)}</h3>"
            f"<p>{_e(row.get('action'))}. {actor_refs}"
            f"{(' · ' + _e(row.get('institution'))) if row.get('institution') else ''}</p>"
            f"{notice}"
            f"<p>Lähdetunnisteet:</p>{_source_list(row.get('evidence_ids') or [], refs)}"
            "</li>"
        )
    if not sequence_rows:
        sequence_html = '<p class="notice">Lähdesekvenssiä ei ole liitetty.</p>'
    else:
        sequence_html = '<ol class="timeline">' + "".join(sequence_rows) + "</ol>"

    q_text = str((question or {}).get("text") or "").strip()
    q_block = (
        f'<details><summary>Näytä kysymyksen lähdeteksti</summary><blockquote>{_e(q_text)}</blockquote></details>'
        if q_text
        else '<p class="muted">Kysymyksen tekstipoimintaa ei ole liitetty.</p>'
    )
    answer_text = str((answer or {}).get("text") or "").strip()
    answer_block = (
        f'<details><summary>Näytä vastausobjektin lähdeteksti</summary><blockquote>{_e(answer_text)}</blockquote></details>'
        if answer_text
        else '<p class="muted">Vastausobjektin tekstipoimintaa ei ole liitetty.</p>'
    )

    disposition = episode.get("institutional_disposition") or {}
    coverage = episode.get("coverage") or {}
    coverage_items = coverage.get("items") or []
    coverage_html = (
        "<p>"
        f"Lähderekisterin kattavuus on {'todennettu ilmoitetussa rajauksessa' if coverage.get('complete') else 'rajattu; täydellisyyttä ei ole todennettu'}."
        "</p>"
    )
    if coverage_items:
        coverage_html += "<ul>" + "".join(
            f"<li>{_e(item.get('coverage_id'))}: {_e(item.get('state') or 'tila avoin')}. "
            f"{_e('; '.join(item.get('limitations') or []))}</li>"
            for item in coverage_items
        ) + "</ul>"

    body = f"""
<header>
  <p class="kicker">Päätös- ja käsittelytapaus · {_e(_STATE_LABELS.get(state, 'Käsittelyn tila avoin'))}</p>
  <h1>{_e(title)}</h1>
  <p class="muted">Tapaustunnus <code>{_e(episode.get('episode_id'))}</code></p>
  <p><a class="download" download href="{_e(slug)}.json">Lataa tämän tapauksen JSON-tietue</a></p>
</header>
<section>
  <h2>Mitä lähteet kertovat</h2>
  <p>Tämä sivu näyttää lähteeseen kirjatun käsittelyjakson. Se ei päättele vastauksesta politiikan toteutumista,
  lain oikeusvaikutusta, syy-yhteyttä tai sitä, mitä olisi tapahtunut vaihtoehtoisella toimintatavalla.</p>
  <article><h3>Kirjallinen kysymys</h3>{q_block}</article>
  <article><h3>Hallituksen vastaus</h3>{_answer_body_note(episode)}{answer_block}</article>
</section>
<section>
  <h2>Lähteessä nimetyt toimijat</h2>
  <table><thead><tr><th>Nimi</th><th>Rooli lähteessä</th><th>Tunnistuksen ja roolin rajaus</th></tr></thead>
  <tbody>{actor_rows}</tbody></table>
</section>
<section>
  <h2>Kronologinen lähdesekvenssi</h2>
  <p>Asiakirjan rekisteröinti, jättämismerkintä, vastauksen antaminen ja täysistuntoilmoitus ovat eri havaintoja.
  Niitä ei yhdistetä yhdeksi sisällölliseksi päätökseksi.</p>
  {sequence_html}
</section>
<section>
  <h2>Institutionaalinen tila</h2>
  <p>Rekisterin tila: <strong>{'Vastaus kirjattu' if disposition.get('state') == 'ANSWERED' else 'Vastausta ei varmennettu tästä lähderajauksesta'}</strong> · päivämäärä {_e(disposition.get('date') or 'avoin')}.</p>
  <dl class="state-grid">
    <dt>Oikeusvaikutus</dt><dd>Ei arvioitu. Lähde ei ole lain voimaantulo- tai oikeusvaikutusrekisteri.</dd>
    <dt>Politiikan toteutus</dt><dd>Ei arvioitu. Vastauksen kirjaaminen ei osoita toteutusta.</dd>
    <dt>Syy-yhteys / lopputulos</dt><dd>Ei arvioitu. Tässä jaksossa ei ole vaikutus- tai vertailulähdettä.</dd>
  </dl>
  <details><summary>Arvioinnin tietuetila</summary><code>NOT_ASSESSED</code></details>
</section>
<section>
  <h2>Avoimeksi jää</h2>
  <p>Puuttuva julkinen rajoite, vaihtoehtoinen toimintatapa tai muu lähderivi ei ole tässä merkintä siitä,
  että kyseinen asia olisi ollut olematon. Päätelmä rajataan siihen, minkä lähdetunnisteet osoittavat.</p>
  {_unknowns_html(episode)}
</section>
<section>
  <h2>Lähderivit</h2>
  {_source_records_table(episode, available_objects)}
</section>
<section>
  <h2>Lähdetunnisteet ja poiminnat</h2>
  {_evidence_cards(episode, refs)}
</section>
<section>
  <h2>Lähderekisterin kattavuus</h2>
  {coverage_html}
</section>
"""
    return _document(title, body, nav='<a href="../index.html">Haku</a> · <a href="index.html">Tapaukset</a>')


def _summary_from_episode(episode: Mapping[str, Any]) -> dict[str, Any]:
    title = _matter_title(episode)
    actors = [
        str(actor.get("name") or "")
        for actor in episode.get("actors") or []
        if isinstance(actor, Mapping) and actor.get("name")
    ]
    dates = [
        str(row.get("date"))
        for row in episode.get("source_sequence") or []
        if row.get("date")
    ]
    episode_id = str(episode.get("episode_id") or episode.get("matter_id") or "episode")
    return {
        "episode_id": episode_id,
        "matter_id": episode.get("matter_id"),
        "title": title,
        "state": _STATE_LABELS.get(episode.get("episode_state"), "Käsittely jää avoimeksi"),
        "actors": actors,
        "dates": _unique(dates),
        "evidence_count": len(_episode_evidence_ids(episode)),
        "residual_count": len(episode.get("residuals") or []),
        "href": f"{_href_slug(episode_id)}.html",
    }


def _normalise_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    if summary.get("source_sequence") or summary.get("source_objects"):
        return _summary_from_episode(summary)
    episode_id = str(summary.get("episode_id") or summary.get("id") or summary.get("matter_id") or "episode")
    actors = summary.get("actors") or summary.get("actor_names") or []
    actor_names: list[str] = []
    for actor in actors:
        if isinstance(actor, Mapping):
            name = actor.get("name")
        else:
            name = actor
        if name:
            actor_names.append(str(name))
    href = str(summary.get("href") or f"{_href_slug(episode_id)}.html")
    return {
        "episode_id": episode_id,
        "matter_id": summary.get("matter_id"),
        "title": summary.get("title") or summary.get("name") or str(summary.get("matter_id") or episode_id),
        "state": summary.get("state") or summary.get("episode_state") or "UNRESOLVED",
        "actors": actor_names,
        "dates": [str(item) for item in summary.get("dates") or [] if item],
        "evidence_count": summary.get("evidence_count", 0),
        "residual_count": summary.get("residual_count", len(summary.get("residuals") or [])),
        "href": href,
    }


def _json_for_script(value: Any) -> str:
    # Prevent a source quote containing ``</script>`` from closing the data
    # element.  The browser's JSON parser reverses neither escape: ``\u003c``
    # is valid JSON and becomes the original character.
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def render_episode_index(summaries: Iterable[Mapping[str, Any]] | Mapping[str, Mapping[str, Any]]) -> str:
    """Render a local-searchable Finnish index for episode page summaries."""

    values = summaries.values() if isinstance(summaries, Mapping) else summaries
    rows = [_normalise_summary(item) for item in values if isinstance(item, Mapping)]
    rows.sort(key=lambda item: (str(item.get("title") or "").casefold(), str(item.get("episode_id") or "")))
    noscript = "".join(
        f'<li><a href="{_e(item["href"])}">{_e(item["title"])}</a> '
        f'<span class="muted">{_e(item.get("matter_id"))} · {_e(item.get("state"))}</span></li>'
        for item in rows
    ) or '<li class="muted">Tapauksia ei ole tässä aineistossa.</li>'
    body = f"""
<header>
  <p class="kicker">Päätös- ja käsittelytapaukset</p>
  <h1>Asiakirjoihin perustuvat käsittelyjaksot</h1>
  <p>Hae henkilöä, asiaa tai tapaustunnusta. Katso kuka jätti kysymyksen, mitä vastauksen kirjaamisesta tiedetään
  ja mitä käsittelyvaiheista voidaan osoittaa lähteillä.</p>
  <label for="episode-search">Haku</label>
  <input id="episode-search" type="search" autocomplete="off" placeholder="henkilö, asia tai tunniste">
  <p id="episode-count" class="muted" aria-live="polite"></p>
</header>
<section id="episode-results" aria-live="polite"></section>
<noscript><h2>Tapaukset</h2><ul>{noscript}</ul></noscript>
<script type="application/json" id="episode-data">{_json_for_script(rows)}</script>
<script>
(() => {{
  const data = JSON.parse(document.getElementById("episode-data").textContent);
  const input = document.getElementById("episode-search");
  const results = document.getElementById("episode-results");
  const count = document.getElementById("episode-count");
  function draw() {{
    const query = input.value.trim().toLocaleLowerCase("fi");
    const rows = data.filter(item => !query || [item.title, item.matter_id, item.episode_id, item.state, ...(item.actors || []), ...(item.dates || [])]
      .join(" ").toLocaleLowerCase("fi").includes(query));
    count.textContent = "Näytetään " + rows.length + " / " + data.length + " tapausta";
    results.replaceChildren(...rows.map(item => {{
      const article = document.createElement("article");
      const link = document.createElement("a");
      link.href = item.href;
      link.textContent = item.title;
      const heading = document.createElement("h2");
      heading.appendChild(link);
      article.appendChild(heading);
      const meta = document.createElement("p");
      meta.className = "muted";
      meta.textContent = [item.matter_id, item.state, (item.actors || []).join(", "), (item.dates || []).join(" → ")].filter(Boolean).join(" · ");
      article.appendChild(meta);
      const residual = document.createElement("p");
      residual.className = "muted";
      residual.textContent = "Lähteitä " + item.evidence_count + " · avoimia rajauksia " + item.residual_count;
      article.appendChild(residual);
      return article;
    }}));
  }}
  input.addEventListener("input", draw);
  draw();
}})();
</script>
"""
    return _document("Päätös- ja käsittelytapaukset", body, nav='<a href="../index.html">Haku</a>')


def _document(title: str, body: str, *, nav: str) -> str:
    return f"""<!doctype html>
<html lang="fi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_e(title)} – Poliittinen muisti</title>
<style>
:root {{ color-scheme: light; --ink:#1c1915; --muted:#5c564c; --line:#e1d9ce; --paper:#f6f3ed; --card:#fffdf9; --accent:#1f4d3a; --notice:#fff4cf; }}
* {{ box-sizing:border-box; }} body {{ max-width:1040px; margin:0 auto; padding:1.2rem; color:var(--ink); background:var(--paper); font:17px/1.5 Georgia,"Iowan Old Style",serif; overflow-wrap:anywhere; }}
a {{ color:var(--accent); }} nav {{ margin-bottom:1.4rem; }} nav a {{ margin-right:.8rem; }} h1 {{ font-size:2rem; line-height:1.15; }} h2 {{ margin-top:2rem; }}
article, table, .timeline-item {{ background:var(--card); border:1px solid var(--line); }} article {{ padding:.9rem 1rem; margin:.7rem 0; }}
.kicker, .muted {{ color:var(--muted); }} .notice {{ background:var(--notice); border-left:4px solid #bd8b00; padding:.7rem .9rem; }}
.download {{ display:inline-block; border:1px solid var(--accent); padding:.35rem .6rem; }} code {{ overflow-wrap:anywhere; }}
table {{ width:100%; border-collapse:collapse; }} th, td {{ text-align:left; vertical-align:top; border-top:1px solid var(--line); padding:.45rem .5rem; }}
.timeline {{ list-style:none; padding:0; }} .timeline-item {{ display:grid; grid-template-columns:12rem 1fr; gap:.2rem 1rem; padding:.85rem 1rem; margin:.65rem 0; }}
.timeline-item h3, .timeline-item p, .timeline-item ul {{ grid-column:2; margin:.25rem 0; }} .timeline-date {{ grid-row:1 / span 4; color:var(--muted); }}
.evidence-list {{ margin:.15rem 0; }} .evidence-card blockquote {{ border-left:3px solid var(--line); margin:.5rem 0; padding-left:.7rem; white-space:pre-wrap; }}
.state-grid {{ display:grid; grid-template-columns:minmax(10rem,20%) 1fr; gap:.2rem 1rem; }} .state-grid dt {{ font-weight:700; }} .state-grid dd {{ margin:0; }}
.actor-ref {{ display:inline-block; margin-right:.5rem; }} input[type=search] {{ width:100%; max-width:42rem; padding:.55rem; font:inherit; border:1px solid var(--line); }}
#episode-results article {{ margin:.65rem 0; }}
@media(max-width:700px) {{ body {{ font-size:16px; }} .timeline-item {{ display:block; }} .timeline-date {{ margin-bottom:.4rem; }} .state-grid {{ display:block; }} .state-grid dt {{ margin-top:.7rem; }} }}
</style></head><body><nav>{nav}</nav><main>{body}</main></body></html>"""


__all__ = ["render_episode", "render_episode_index"]

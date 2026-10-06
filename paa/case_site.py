"""Cold-reader inquiry pages: question, bounded answer and decisive evidence."""

import html
import json
from pathlib import Path
from urllib.parse import quote

from paa.evidence_site import render_evidence
from paa.inquiry_cases import validate_case


def _e(value) -> str:
    return html.escape(str(value), quote=True)


def _render_public_case(packet: dict, publication: dict) -> str:
    """Present an editorial answer; preserve the full reviewed packet separately."""
    slug = quote(packet['case_id'], safe='')
    refs = {row['evidence_id']: row for row in packet['evidence']}
    sources = {row['source_id']: row for row in packet['sources']}
    sections = []
    primary_evidence = ''
    for index, section in enumerate(publication['sections']):
        excerpts = [render_evidence(section['evidence_ids'], refs, sources)]
        evidence = (f'<details class="inline-evidence"><summary>{_e(section["source_label"])} — näytä kohta</summary>'
                    + ''.join(excerpts) + '</details>')
        if index == 0 and publication['comparison']:
            primary_evidence = evidence
            evidence = ''
        sections.append(f'<section class="answer-section"><h2>{_e(section["title"])}</h2><p>{_e(section["text"])}</p>'
                        + evidence + '</section>')
    comparison = ''
    if publication['comparison']:
        rows = publication['comparison']
        def percent(row: dict) -> str:
            return f'{100 * row["numerator"] / row["denominator"]:.0f} %'
        figures = ''.join(f'<div><p>{_e(row["period"])} · {_e(row["measure"])}</p>'
                          f'<strong>{percent(row)}</strong><p>{row["numerator"]} / {row["denominator"]} kirjattua tapahtumaa</p></div>'
                          for row in rows[:2])
        table = ''.join(f'<tr><td>{_e(row["period"])}</td><th scope="row">{_e(row["measure"])}</th>'
                        f'<td>{row["numerator"]}/{row["denominator"]}</td><td>{percent(row)}</td></tr>' for row in rows)
        comparison = (f'<section class="comparison" aria-label="{_e(publication["comparison_label"])}">'
                      f'<div class="figures">{figures}</div><p class="population">{_e(publication["comparison_label"])}</p>'
                      '<details><summary>Vertaa vuosia ja tarkista laskenta</summary>'
                      '<table><caption>Saman taulukon kaksi eri mittaria</caption><thead><tr><th>Vuosi</th><th>Mittari</th>'
                      f'<th>Laskenta</th><th>Osuus</th></tr></thead><tbody>{table}</tbody></table>'
                      '<p>Osuudet on pyöristetty kokonaisiin prosentteihin. Palveluntuottajajakauma ei mittaa tuottajan omaa onnistumisastetta.</p></details></section>')
    computation_link = ''
    if (publication.get('computation_page')
            and packet['case_id'] in publication.get('computation_case_ids', ())):
        label = publication.get('computation_label') or 'Äänestyslaskennan taulukko'
        href = publication['computation_page']
        if not href.startswith('../') and not href.startswith('/'):
            href = f'../{href}'
        computation_link = (
            f'<p class="review-status"><a href="{_e(href)}">{_e(label)}</a></p>'
        )
    body = ('<nav><a href="../index.html">Poliittinen muisti</a></nav>'
            f'<h1>{_e(publication["article_title"])}</h1><p class="takeaway">{_e(publication["takeaway"])}</p>'
            f'<p class="review-status">Tekoälyavusteinen lähdeluku · {_e(publication["review_date"])}</p>'
            f'{computation_link}'
            f'{comparison}{primary_evidence}' + ''.join(sections) +
            f'<section><h2>Mikä jää avoimeksi?</h2><p>{_e(publication["main_limit"])}</p></section>'
            '<details class="method"><summary>Menetelmä, tarkastelujakso ja koko tutkimusaineisto</summary>'
            f'<p>Julkaisija: {_e(publication["publisher"])}. Tekoälyavusteinen lähdeluku; ei riippumatonta ihmistarkistusta.</p>'
            f'<p>Tarkastelujakso: {_e(publication["source_period"])}</p>'
            f'<p>{_e(packet["question"]["scope"])}</p>'
            f'<p><a href="{slug}-research.html">Lue kaikki löydökset, lähdekatkelmat ja tulkintaperusteet</a></p>'
            f'<p>Versio: {_e(publication["revision"][:12])}. <a href="../terms.html">Lähteiden käyttö ja menetelmä</a></p></details>'
            '<footer><p class="actions">'
            f'<a download href="{slug}.json">Lataa lähdepaketti (JSON)</a> · '
            f'<a download href="{slug}.txt">Lataa viitteet</a> · '
            f'<a href="{_e(publication["correction_url"])}">Ilmoita korjaus</a></p>'
            '<details><summary>Jaa tämä vastaus</summary><label for="case-url">Sivun osoite</label> '
            f'<input id="case-url" readonly value="{_e(publication["canonical_url"])}"> '
            '<button type="button" id="copy-url" disabled>Kopioi linkki</button>'
            '<p>Sivun voi jakaa myös selaimen osoiterivin linkillä.</p>'
            '<p id="share-status" role="status" aria-live="polite"></p></details></footer>'
            '<script src="../share.js" defer></script>')
    return _document(publication['article_title'], body, description=publication['takeaway'],
                     canonical_url=publication['canonical_url'])


def render_case(packet: dict, *, publication: dict | None = None, research_for: str | None = None) -> str:
    validate_case(packet)
    if publication:
        return _render_public_case(packet, publication)
    legal_section = ""
    if packet.get("legal_inquiry"):
        from paa.legal_inquiry import validate_legal_inquiry

        validate_legal_inquiry(packet)
        legal = packet["legal_inquiry"]
        receipt = packet["legal_comparison_artifacts"][0]["receipt"]
        temporal = receipt["comparison"]["temporal"]
        oracle = temporal.get("sources", {}).get("oracle", {})
        source_date = oracle.get("source_date")
        temporal_note = (f'Vertailun konsolidoitu lähde on päivätty {_e(source_date)}, '
                         f'joten se ei varmista tilaa päivälle {_e(legal["as_of"])}.') if oracle.get("status") == "FUTURE_SOURCE_NOT_VALIDATED" else (
                            f'Vertailu ei varmista täsmällistä soveltamistilaa päivälle {_e(legal["as_of"])}.')
        replay = receipt['reconstruction']['replay']
        replay_note = (
            f'<p>LawVM:n tekstirekonstruktio päivälle {_e(legal["as_of"])}: '
            f'lähdemuutos {_e(replay.get("source_amendment"))}, '
            f'version voimaantulopäivä {_e(replay.get("effective"))}. '
            'Tekstin rekonstruointi ja velvoitteen ajallinen soveltaminen ovat eri kysymyksiä.</p>'
        ) if replay.get('available') and replay.get('status') == 'selected' else ''
        legal_section = ('<h2>Hyväksytty teksti ja soveltamisen rajaus</h2><article>'
                         f'<p>Julkaistu muutoslaki: {_e(legal["amending_statute_id"])}. '
                         f'Lähdetekstissä ilmoitettu käytön viimeinen aloituspäivä: {_e(legal["latest_start_date"])}.</p>'
                         f'<p>{temporal_note} Takaraja ei osoita, että käyttö olisi ollut mahdotonta '
                         'tai puuttunut sitä ennen. Käytännön toimeenpano ja vaikutukset tarvitsevat omat lähteensä.</p>'
                         f'{replay_note}'
                         '<details><summary>Tekstivertailun tarkistettavuus</summary>'
                         '<p>LawVM:n rekonstruoima teksti ja vertailutulos ovat ladattavassa lähdepaketissa. '
                         'Tekstien vastaavuus tai ero ei yksin ratkaise historiallista oikeustilaa.</p>'
                         '</details></article>')
    sources = {s["source_id"]: s for s in packet["sources"]}
    refs = {r["evidence_id"]: r for r in packet["evidence"]}
    evidence_html = render_evidence(list(refs), refs, sources)
    conclusions, candidates = [], []
    for claim in packet["claims"]:
        links = " ".join(f'<a href="#{_e(key)}">Lähde {index}</a>' for index, key in enumerate(claim.get("evidence_ids", []), 1))
        body = f'<article><h3>{_e(claim["dimension"])}</h3><p>{_e(claim["text"])}</p><p>{links}</p>'
        if claim["state"] == "SOURCE_REVIEWED":
            conclusions.append(body + '</article>')
        else:
            candidates.append(body + '<p>Hakutulos tai avoin tulkinta; ei varmennettu päätelmä.</p></article>')
    unknowns = "".join(f'<li><strong>{_e(row["question"])}</strong><br>{_e(row["missing_evidence"])}</li>' for row in packet["unknowns"])
    candidate_section = ('<details><summary>Tutkittavat ehdokkaat</summary>' + "".join(candidates) + '</details>') if candidates else ''
    answer = "".join(conclusions) or '<p>Aineisto ei vielä tue tarkistettua vastausta. Alla olevat katkelmat ja avoimet kysymykset rajaavat jatkotutkimusta.</p>'
    navigation = (f'<nav><a href="{_e(research_for)}.html">Takaisin vastaukseen</a> · '
                  '<a href="../index.html">Poliittinen muisti</a></nav>') if research_for else (
                      '<nav><a href="index.html">Tutkittavat päätökset</a> · <a href="../index.html">Henkilöt ja lähteet</a></nav>')
    introduction = '<p>Täydellinen tutkimusaineisto: väitteet, tulkintaperusteet ja täsmälliset lähdekatkelmat.</p>' if research_for else ''
    body = (navigation +
            f'<h1>{_e(packet["title"])}</h1>{introduction}'
            f'<h2>Kysymys</h2><p>{_e(packet["question"]["text"])}</p>'
            f'<p>{_e(packet["question"]["scope"])}</p><h2>Mitä lähteet osoittavat?</h2>{answer}'
            f'{legal_section}'
            f'<h2>Mikä jää avoimeksi?</h2><ul>{unknowns}</ul>{candidate_section}'
            f'<h2>Ratkaisevat lähdekatkelmat</h2>{evidence_html}'
            '<details><summary>Lähteiden tarkistaminen</summary>'
            '<p>Katkelmien ja lähdeversioiden tarkistus ei yksin varmista tulkinnan oikeellisuutta. '
            f'Asiakirjan muutos ei osoita, kuka sen aiheutti. Valikoidut tapaukset eivät mittaa yleistä onnistumis- tai epäonnistumisastetta.</p>'
            f'<a href="{quote(packet["case_id"], safe="")}.json">Lataa lähteet ja arviointi</a></details>')
    return _document(packet["title"], body)


def _document(title: str, body: str, *, description: str = '', canonical_url: str = '') -> str:
    metadata = ''
    if description:
        metadata = (f'<meta name="description" content="{_e(description)}">'
                    f'<meta property="og:title" content="{_e(title)} · PAA">'
                    f'<meta property="og:description" content="{_e(description)}">'
                    '<meta property="og:type" content="article"><meta property="og:locale" content="fi_FI">'
                    '<meta property="og:site_name" content="PAA"><meta name="twitter:card" content="summary">'
                    f'<meta name="twitter:title" content="{_e(title)} · PAA">'
                    f'<meta name="twitter:description" content="{_e(description)}">')
    if canonical_url:
        metadata += (f'<link rel="canonical" href="{_e(canonical_url)}">'
                     f'<meta property="og:url" content="{_e(canonical_url)}">')
    return ('<!doctype html><html lang="fi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{_e(title)} · PAA</title>{metadata}<style>body{{font:18px/1.6 system-ui;margin:2rem auto;padding:0 1rem;max-width:900px;color:#172d35;background:#f5f7f7;overflow-wrap:anywhere}}'
            'article{background:white;padding:1rem;margin:1rem 0;border:1px solid #ccd7da;border-radius:8px}blockquote{border-left:3px solid #467583;padding-left:1rem;margin:1rem 0}a{color:#075970}summary{cursor:pointer}h1{line-height:1.2}details{margin:1rem 0}'
            '.takeaway{font-size:1.15em;font-weight:600}.review-status,.eyebrow{font-size:.85em;color:#334d57}.eyebrow{margin:0}h2{font-size:1.25em;line-height:1.3}.case-card h2{margin:.25rem 0}.figures{display:grid;grid-template-columns:1fr 1fr;gap:1rem}.figures>div{background:#e4efef;padding:.75rem;border-radius:8px}.figures p{font-size:.85em;margin:.3rem 0}.figures strong{font-size:2.5rem;line-height:1.1}.population{font-size:.9em}.answer-section{margin:1.5rem 0}.inline-evidence{border-left:3px solid #467583;padding:.5rem .75rem;background:white}.inline-evidence summary{font-size:.9em;color:#075970}.method,footer{font-size:.9em}table{width:100%;border-collapse:collapse;font-size:.8em}th,td{text-align:left;padding:.4rem .2rem;border-bottom:1px solid #ccd7da}input{box-sizing:border-box;max-width:100%;width:28rem;font:inherit}button{font:inherit;padding:.35rem .75rem;margin:.25rem 0}a:focus-visible,button:focus-visible,input:focus-visible,summary:focus-visible{outline:3px solid #b85200;outline-offset:3px}@media(max-width:600px){body{font-size:17px;margin:1rem auto}h1{font-size:1.65rem;margin:.9rem 0}.takeaway{font-size:1.05em}.figures{gap:.5rem}.figures strong{font-size:2rem}article{padding:.75rem}}</style></head>'
            f'<body>{body}</body></html>')


def write_cases(packets: list[dict], destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    items = []
    for packet in packets:
        slug = quote(packet["case_id"], safe="")
        (destination / f'{slug}.html').write_text(render_case(packet), encoding="utf-8")
        (destination / f'{slug}.json').write_text(json.dumps(packet, ensure_ascii=False, indent=2), encoding="utf-8")
        items.append(f'<li><a href="{slug}.html">{_e(packet["title"])}</a><br>{_e(packet["question"]["text"])}</li>')
    body = ('<nav><a href="../index.html">Henkilöt ja lähteet</a></nav><h1>Julkisten päätösten tutkimuskysymykset</h1>'
            '<p>Mitä tavoitetta muutettiin? Mitä varoitukselle tapahtui? Mitä julkinen aineisto osoittaa toimijoiden valinnoista? '
            'Valikoidut tapaukset testaavat näitä eri kysymyksiä samoista lähteistä.</p><ul>' + "".join(items) + '</ul>')
    (destination / 'index.html').write_text(_document('Tutkittavat päätökset', body), encoding="utf-8")

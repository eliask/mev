"""Cold-reader inquiry pages: question, bounded answer and decisive evidence."""

import html
import json
from pathlib import Path
from urllib.parse import quote

from paa.inquiry_cases import validate_case


def _e(value) -> str:
    return html.escape(str(value), quote=True)


def render_case(packet: dict) -> str:
    validate_case(packet)
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
                         f'<p>Vertailun tunniste: {_e(receipt["receipt_id"])}</p></details></article>')
    sources = {s["source_id"]: s for s in packet["sources"]}
    refs = {r["evidence_id"]: r for r in packet["evidence"]}
    evidence_html = []
    for ref in refs.values():
        source = sources[ref["source_id"]]
        binding_note = ('<p>Mallin antama sijainti oli virheellinen. Lainaus sidottiin lähteen '
                        'ainoaan täsmälleen samaan tekstikohtaan; alkuperäinen virhe säilyy lähdepaketissa.</p>') if (
                            ref.get("binding_method") == "UNIQUE_EXACT_TEXT_AFTER_REJECTED_MODEL_OFFSET") else ''
        kind = {"GOVERNMENT_PROPOSAL": "Hallituksen esitys", "COMMITTEE_REPORT": "Valiokunnan mietintö tai lausunto",
                "EXPERT_STATEMENT": "Asiantuntijalausunto", "OFFICIAL_FINLEX_ACT": "Julkaistu muutoslaki"}.get(source.get("source_kind"), "Lähde")
        identifier = source.get("document_identifier") or source.get("record_id") or ""
        evidence_html.append(f'<article id="{_e(ref["evidence_id"])}"><h3>{_e(kind)} · {_e(identifier)}</h3>'
            f'<p>{_e(source["title"])}</p>'
            f'<blockquote>{_e(ref["quote"])}</blockquote><p><a href="{_e(ref["url"])}">Alkuperäinen lähde</a>'
            f'</p><details><summary>Lähdeversio ja tarkka katkelma</summary>'
            f'{binding_note}<p>{_e(ref["locator"])}</p><p>{_e(ref["evidence_id"])}</p><p>Tekstin SHA-256: {_e(ref["text_sha256"])}'
            f'<br>Raakalähteen SHA-256: {_e(ref["raw_sha256"])}<br>Merkit {ref["start"]}–{ref["end"]}</p></details></article>')
    conclusions, candidates = [], []
    for claim in packet["claims"]:
        links = " ".join(f'<a href="#{_e(key)}">Lähde {index}</a>' for index, key in enumerate(claim.get("evidence_ids", []), 1))
        body = f'<article><h3>{_e(claim["dimension"])}</h3><p>{_e(claim["text"])}</p><p>{links}</p>'
        if claim["state"] == "SOURCE_REVIEWED":
            review = claim["review"]
            body += f'<details><summary>Tulkinnan peruste</summary><p>{_e(review["rationale"])}</p>'
            body += f'<p>Arvioija: {_e(review["reviewer"])} · Menetelmä: {_e(review["method"])}</p></details>'
            conclusions.append(body + '</article>')
        else:
            candidates.append(body + '<p>Hakutulos tai avoin tulkinta; ei varmennettu päätelmä.</p></article>')
    unknowns = "".join(f'<li><strong>{_e(row["question"])}</strong><br>{_e(row["missing_evidence"])}</li>' for row in packet["unknowns"])
    candidate_section = ('<details><summary>Tutkittavat ehdokkaat</summary>' + "".join(candidates) + '</details>') if candidates else ''
    answer = "".join(conclusions) or '<p>Aineisto ei vielä tue tarkistettua vastausta. Alla olevat katkelmat ja avoimet kysymykset rajaavat jatkotutkimusta.</p>'
    body = (f'<nav><a href="index.html">Tutkittavat päätökset</a> · <a href="../index.html">Henkilöt ja lähteet</a></nav>'
            f'<h1>{_e(packet["title"])}</h1><h2>Kysymys</h2><p>{_e(packet["question"]["text"])}</p>'
            f'<p>{_e(packet["question"]["scope"])}</p><h2>Mitä lähteet osoittavat?</h2>{answer}'
            f'{legal_section}'
            f'<h2>Mikä jää avoimeksi?</h2><ul>{unknowns}</ul>{candidate_section}'
            f'<h2>Ratkaisevat lähdekatkelmat</h2>{"".join(evidence_html)}'
            f'<details><summary>Rajaus ja tarkistettavuus</summary><p>{_e(packet["question"]["answer_standard"])}</p>'
            f'<p>{_e(packet["selection_basis"])}</p><p>Katkelmien ja lähdeversioiden tarkistus ei yksin varmista tulkinnan oikeellisuutta. '
            f'Asiakirjan muutos ei osoita, kuka sen aiheutti. Valikoidut tapaukset eivät mittaa yleistä onnistumis- tai epäonnistumisastetta.</p>'
            f'<a href="{quote(packet["case_id"], safe="")}.json">Lataa lähteet ja arviointi</a></details>')
    return _document(packet["title"], body)


def _document(title: str, body: str) -> str:
    return ('<!doctype html><html lang="fi"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{_e(title)} · PAA</title><style>body{{font:18px/1.6 system-ui;margin:2rem auto;padding:0 1rem;max-width:900px;color:#172d35;background:#f5f7f7;overflow-wrap:anywhere}}'
            'article{background:white;padding:1rem;margin:1rem 0;border:1px solid #ccd7da;border-radius:8px}blockquote{border-left:3px solid #467583;padding-left:1rem;margin:1rem 0}a{color:#075970}summary{cursor:pointer}h1{line-height:1.2}details{margin:1rem 0}</style>'
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

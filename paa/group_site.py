"""Render source-bound descriptive group comparisons without inference."""

import html
import json
import re
from collections.abc import Mapping
from pathlib import Path
from urllib.parse import urlparse


def _e(value):
    return html.escape(str(value if value is not None else ""), quote=True)


def _pct(value):
    """Render a 0..1 diagnostic rate without implying a value if absent."""

    if value is None:
        return "ei laskettavissa"
    try:
        return f"{float(value) * 100:.1f} %"
    except (TypeError, ValueError, OverflowError):
        return "ei laskettavissa"


def _matter_diagnostic(diagnostic):
    """Render formal-label sensitivity as a bounded technical note.

    The formal matter label is a source field, not a policy ontology. Keep
    that distinction next to each derived rate so repeated procedural labels
    cannot be mistaken for a claim about a person's views or conduct.
    """

    if not isinstance(diagnostic, Mapping):
        return ""
    largest_label = diagnostic.get("largest_label")
    largest_text = _e(largest_label) if largest_label else "ei tunnistetta"
    largest_count = _e(diagnostic.get("largest_label_count", 0))
    largest_rate = _e(_pct(diagnostic.get("largest_label_agreement")))
    remaining_rate = _e(_pct(diagnostic.get("agreement_with_largest_label_removed")))
    mean_rate = _e(_pct(diagnostic.get("mean_within_label_agreement")))
    return (
        '<section class="audit-note">'
        '<h2>Asian lähdetunnisteen herkkyystarkistus</h2>'
        '<p>Tämä on tekninen tarkistus Eduskunnan lähteen muodollisen asiaviitteen '
        'vaikutuksesta laskentaan. Asiaviite ei automaattisesti tarkoita yhtä '
        'politiikkakysymystä, eikä tästä tehdä päätelmiä asian sisällöstä tai '
        'henkilön toiminnasta.</p>'
        '<ul>'
        '<li>Täsmällisiä lähdetunnisteita mukana: <strong>'
        + _e(diagnostic.get("labels_denominator", 0))
        + '</strong>.</li>'
        '<li>Vertailukelpoisia kirjauksia tunnisteella: <strong>'
        + _e(diagnostic.get("comparable_votes_with_label", 0))
        + '</strong>; ilman tunnistetta: <strong>'
        + _e(diagnostic.get("comparable_votes_without_label", 0))
        + '</strong>.</li>'
        '<li>Tunnistekohtaisten enemmistöosumien keskiarvo, jossa jokainen '
        'tunniste painaa yhtä paljon: <strong>'
        + mean_rate
        + '</strong>.</li>'
        '<li>Suurin yksi lähdetunniste: <code>'
        + largest_text
        + '</code> — <strong>'
        + largest_count
        + '</strong> vertailua; enemmistöosuma <strong>'
        + largest_rate
        + '</strong>.</li>'
        '<li>Kun suurin tunniste poistetaan: enemmistöosuma <strong>'
        + remaining_rate
        + '</strong>.</li>'
        '</ul>'
        '<p>Toistuvat käsittelykierrokset voivat painottaa lukuja. Tunnisteita '
        'ei yhdistetä semanttisiksi politiikkaklustereiksi; laskenta ei väitä '
        'eri kierrosten olevan toisistaan riippumattomia.</p>'
        '</section>'
    )


def slug(value):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(value))


def _page(title, body):
    return ('<!DOCTYPE html><html lang="fi"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>'+_e(title)+'</title><style>body{max-width:960px;margin:2rem auto;padding:1rem;font:17px/1.5 Georgia;'
            'background:#f6f3ed;overflow-wrap:anywhere}table{border-collapse:collapse;width:100%}th,td{padding:.5rem;'
            'border-bottom:1px solid #ccc;text-align:left}a{color:#1f4d3a}code{overflow-wrap:anywhere}</style>'
            '<nav><a href="../index.html">Haku</a> · <a href="index.html">Ryhmän vertailut</a></nav><main><h1>'+_e(title)+'</h1>'+body+'</main></html>')


def write_group_pages(packets, sources, report, destination: Path):
    destination.mkdir(parents=True, exist_ok=True)
    source_dir = destination / "sources"
    source_dir.mkdir()
    source_map = {s["source_id"]: s for s in sources}
    for source in sources:
        key = slug(source["source_id"])
        (source_dir / (key+".json")).write_text(json.dumps(source, ensure_ascii=False), encoding="utf-8")
        rows = ''.join('<tr><td>'+_e(r["person_id"]+' '+r.get("first_name",'')+' '+r.get("last_name",''))+
                       '</td><td>'+_e(r["group_code"])+'</td><td>'+_e(r["response"])+'</td></tr>' for r in source["rows"])
        url = source.get("source_url") or ''
        link = '<p><a href="'+_e(url)+'">Eduskunnan lähde</a></p>' if urlparse(url).scheme in {'http', 'https'} else ''
        body = '<p>'+_e(source["session_date"])+' · '+_e(source.get("matter"))+'</p><p>'+_e(source["title"])+'</p>'+link
        body += '<p>Julkaistut vaihtoehtojen määrät: '+_e(source["published_totals"])+'. Lähderivien tarkistus: '+_e(source["state"])+'.</p>'
        body += '<p>Tarkiste koskee normalisoituja lähderivejä: <code>'+_e(source["source_sha256"])+'</code>.</p>'
        body += '<p><a href="'+key+'.json">Lataa kaikki vertailun lähderivit</a></p><table><thead><tr><th>Henkilö</th><th>Ryhmätunnus tässä äänestyksessä</th><th>Kirjattu valinta</th></tr></thead><tbody>'+rows+'</tbody></table>'
        # Source pages are one level deeper than person pages.
        (source_dir / (key+".html")).write_text(_page('Äänestys '+str(source["vote_id"]), body).replace('href="../index.html"', 'href="../../index.html"').replace('href="index.html"', 'href="../index.html"'), encoding="utf-8")
    index = []
    for packet in packets:
        key = 'person-'+slug(packet["person_id"])
        s = packet["summary"]
        name = packet.get("display_name") or packet["person_id"]
        (destination / (key+'.json')).write_text(json.dumps(packet, ensure_ascii=False), encoding="utf-8")
        body = '<p>Rajaus '+_e(packet["cutoff"])+'. Ryhmä tulee kunkin äänestyksen lähderiviltä.</p>'
        body += '<p>Kirjattu JAA/EI-valinta vastasi muiden saman ryhmän edustajien enemmistöä <strong>'+str(s["matching_count"])+' / '+str(s["comparable_count"])+'</strong> vertailukelpoisessa äänestyksessä.</p>'
        body += '<p>Äänestyksissä, joissa koko eduskunnassa kirjattiin sekä JAA että EI: '+str(s["contested_matching_count"])+' / '+str(s["contested_comparable_count"])+'.</p>'
        body += '<p>Vertailusta poistetaan henkilön oma ääni. Vähintään '+str(packet["minimum_peer_count"])+' muuta JAA/EI-ääntä tarvitaan; tasajako jää määrittelemättä. Tämä kuvaa yhtä julkista toimintakanavaa. Sisäinen valmistelu ja vaikuttaminen eivät näy siinä.</p>'
        body += '<p>Kirjattu POISSA: '+str(s.get("recorded_absence_count",0))+'. Kirjattu TYHJA: '+str(s.get("recorded_abstention_count",0))+'. Henkilön lähderivi puuttuu: '+str(s.get("source_row_missing_count",s["absent_count"]))+' (ei poissaolotulkintaa).</p>'
        body += '<p>Ryhmättömät tai tuntemattoman ryhmän rivit, liian pienet vertailuryhmät ja tasajaot jäävät vertailun ulkopuolelle. Menettelyn eri äänestyskierrokset ovat erillisiä havaintoja; luvut eivät ole itsenäisten politiikkapäätösten määrät.</p>'
        body += _matter_diagnostic(s.get("matter_cluster_diagnostic"))
        comparisons = [c for c in packet["comparisons"] if c["status"] == "COMPARABLE"]
        different = [c for c in comparisons if not c["matches_peer_majority"]]
        body += '<h2>Enemmistövertailusta poikkeavat kirjatut valinnat</h2><p>'+str(len(different))+' lähderiviä. Alla enintään 50 viimeisintä.</p>'
        for c in sorted(different, key=lambda c:(c["session_date"],c["vote_id"]), reverse=True)[:50]:
            src = source_map[c["source_ref"]]
            body += '<article><p>'+_e(c["session_date"])+' · '+_e(c["matter"])+' · ryhmä '+_e(c["target_group_code"])+'</p><p>'+_e(src["title"])+'</p><p>Oma valinta: '+_e(c["target_response"])+'. Muiden ryhmäläisten JAA: '+str(c["peer_jaa"])+', EI: '+str(c["peer_ei"])+'. <a href="sources/'+slug(c["source_ref"])+'.html">Tarkista kaikki lähderivit</a></p></article>'
        body += '<p><a href="'+key+'.json">Lataa kaikki vertailut, rajaukset ja tarkisteet</a></p>'
        (destination / (key+'.html')).write_text(_page(name+' — kirjattujen äänten ryhmävertailu',body), encoding="utf-8")
        index.append('<li><a href="'+key+'.html">'+_e(name)+'</a> · '+str(s["comparable_count"])+' vertailukelpoista ääntä</li>')
    (destination / 'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    body = '<p>Vertailu käyttää äänestyksen aikaista ryhmätunnusta ja muiden ryhmäläisten kirjattuja JAA/EI-valintoja.</p><p>Toimielimen päätös, puolueen kanta ja henkilön vaikutus ovat eri kysymyksiä.</p>'
    matter = report.get("matter_sensitivity") if isinstance(report, Mapping) else None
    if isinstance(matter, Mapping):
        largest_label = matter.get("largest_label")
        largest_text = _e(largest_label) if largest_label else "ei tunnistetta"
        body += (
            '<section class="audit-note"><h2>Lähdetunnisteiden herkkyys</h2>'
            '<p>Suuri muodollinen lähdetunniste on <code>'
            + largest_text
            + '</code>: <strong>'
            + _e(matter.get("largest_label_count", 0))
            + '</strong> / <strong>'
            + _e(report.get("valid_source_count", 0))
            + '</strong> tarkistetusta äänestyksestä ('
            + _e(_pct(matter.get("largest_label_share_of_valid_sources")))
            + '). Tämä on lähdeaineiston rakennehavainto, ei yksi politiikkakysymys '
            'eikä henkilön toiminnan mittari.</p>'
            '<p>Tunnisteita ei yhdistetä semanttisiksi politiikkaklustereiksi, ja '
            'toistuvat käsittelykierrokset säilyvät erillisinä lähdetapahtumina.</p>'
            '</section>'
        )
    body += '<p><a href="report.json">Lähderajaus ja laskennan tarkistusraportti</a></p><ul>'+''.join(index)+'</ul>'
    (destination / 'index.html').write_text(_page('Kirjattujen äänten ryhmävertailut',body),encoding="utf-8")

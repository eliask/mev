"""Static Finnish browser: render canonical backend evidence packets only."""


import hashlib
import html
import json
import shutil
import sqlite3
import tempfile
from pathlib import Path
from urllib.parse import quote, urlparse

from paa.acquire_yle import YLE_2011_ATTRIBUTION, YLE_2011_LICENSE, YLE_2011_LICENSE_URL, YLE_2011_PUBLICATION
from paa.config import DB_PATH, DIST
from paa.episode_site import render_episode, render_episode_index
from paa.records import alternatives_from_title

PLAIN = {
    "VALUE_OR_SLOGAN": "Arvo tai tunnuslause", "BROAD_OBJECTIVE": "Laaja tavoite",
    "POSITION": "Kanta", "PROCESS_COMMITMENT": "Työtapa", "PERSONAL_ACTION_COMMITMENT": "Havaittava oma teko",
    "PERSONAL_RESTRAINT_COMMITMENT": "Oma pidättäytymissitoumus", "POLICY_DESIDERATUM": "Toivottu muutos",
    "COLLECTIVE_ACTION_COMMITMENT": "Yhteinen sitoumus", "CAUSAL_EFFECT_FORECAST": "Vaikutusarvio",
    "REPORTED_SPEECH": "Siteerattu puhe", "AMBIGUOUS": "Avoin lukutapa", "OBSERVED_STATE_FORECAST": "Tilannearvio",
    "OUTCOME_COMMITMENT": "Sitoumus lopputulokseen", "MAINTAIN_COMMITMENT": "Sitoumus säilyttämiseen",
    "PREVENT_COMMITMENT": "Sitoumus estämiseen", "FACTUAL_CLAIM": "Tosiasiaväite",
    "CAUSAL_CLAIM": "Syy-seurausväite", "QUESTION": "Kysymys",
}
STATES = {"OBSERVED_ALIGNED_ACTION": "Vastaava oma teko havaittu", "INSUFFICIENT_EVIDENCE": "Näyttö ei riitä",
          "NO_OBSERVABLE_OPPORTUNITY": "Vaaliehto ei täyttynyt", "NOT_TESTABLE": "Ei rekisteristä ratkaistava oma teko"}
RELATIONS = {"CANDIDATE": "Hakuehdokas, yhteyttä ei varmennettu", "SAME_POLICY_OBJECT": "Sama toimintakohde, lähdekohdat tarkistettu",
             "SAME_MATTER": "Sama virallinen asia, lähdekohdat tarkistettu", "REJECTED": "Yhteys hylätty",
             "UNRESOLVED": "Yhteys jäi avoimeksi"}
CONDITIONS = {"SATISFIED": "täyttynyt", "NOT_SATISFIED": "ei täyttynyt", "NOT_APPLICABLE": "ei erillistä ehtoa", "UNRESOLVED": "avoin"}
OPPORTUNITIES = {"OBSERVABLE_OPPORTUNITY": "roolijakso havaittu", "NO_OBSERVABLE_OPPORTUNITY": "vaaliehto ei täyttynyt", "UNRESOLVED": "avoin"}
DISPOSITIONS = {"APPROVED": "Eduskunta hyväksynyt", "REJECTED": "Eduskunta hylännyt", "EXPIRED": "Rauennut",
                "GRANTED": "Toimielin myönsi eron", "BOARD_PROPOSAL_ACCEPTED": "Toimielin hyväksyi päätösehdotuksen",
                "VOTE_RECORDED": "Äänestys kirjattu", "PENDING": "Käsittely kesken",
                "UNRESOLVED": "Ratkaisua ei varmennettu tästä lähteestä"}
KINDS = {"LEGISLATIVE_INITIATIVE": "Lakialoite", "RESIGN_ROLE": "Luottamustoimesta eroaminen", "WRITTEN_QUESTION": "Kirjallinen kysymys", "GOVERNMENT_ANSWER": "Hallituksen vastaus kirjalliseen kysymykseen", "VOTE": "Äänestys", "SPEECH": "Puheenvuoro"}


def _role_label(kind: str, role: str | None) -> str:
    if kind == "SPEECH":
        return "Puhuja"
    if role == "AUTHOR" and kind in {"LEGISLATIVE_INITIATIVE", "WRITTEN_QUESTION"}:
        return "Ensimmäinen allekirjoittaja"
    if role == "COSIGNER":
        return "Allekirjoittaja"
    if role == "RESPONDENT":
        return "Vastauksen allekirjoittaja"
    return "Asiakirjassa nimetty toimija"


def _safe(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in value)[:160]


def _url(value: str) -> str:
    return value if urlparse(value or "").scheme in {"http", "https"} else ""


def _state_label(packet: dict) -> str:
    if any(action["state"] == "DOCUMENTED_RELATED_ACTION" for action in packet["actions"]):
        if all(action["kind"] == "LEGISLATIVE_INITIATIVE" for action in packet["actions"]):
            return "Kirjattu aloite samasta tarkistetusta kohteesta"
        return "Kirjattu toiminta samasta tarkistetusta kohteesta"
    state = packet["assessment"]["state"]
    return STATES.get(state, state)


def _attributions(source_ids: set[str]) -> list[dict]:
    if "SRC-YLE-2011" not in source_ids:
        return []
    return [{"label": YLE_2011_ATTRIBUTION, "url": YLE_2011_PUBLICATION,
             "license": YLE_2011_LICENSE, "license_url": YLE_2011_LICENSE_URL}]


def _choice(title: str, response: str) -> str | None:
    import re
    alternatives = alternatives_from_title(title, "e")
    if len(alternatives) < 2 or response not in {"JAA", "EI"}:
        return None
    labels = [re.sub(r"\s+(JAA|EI)\s*$", "", item["alternative_text"], flags=re.IGNORECASE).strip(" -") for item in alternatives]
    selected, other = labels if response == "JAA" else labels[::-1]
    return f"Äänesti: {selected}. Toinen vaihtoehto oli: {other}."


HTML = """<!DOCTYPE html><html lang="fi"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Poliittinen muisti</title>
<style>
  :root { color-scheme: light; --ink: #1c1915; --muted: #5c564c; --line: #e4ddd2; --paper: #f6f3ed; --card: #fffdf9; --accent: #1f4d3a; }
  * { box-sizing: border-box; }
  body { margin: 0; font: 17px/1.45 Georgia, "Iowan Old Style", serif; color: var(--ink); background: var(--paper); overflow-wrap: anywhere; }
  header, main { max-width: 980px; margin: 0 auto; padding: 1rem; }
  h1 { font-size: 1.8rem; margin: 0.2rem 0; }
  a { color: var(--accent); }
  nav a { margin-right: 1rem; }
  input, select, button { font: inherit; }
  input[type="search"], select { padding: 0.55rem 0.7rem; border: 1px solid var(--line); background: white; }
  input[type="search"] { width: 100%; }
  .row { display: flex; gap: 0.5rem; flex-wrap: wrap; margin: 0.6rem 0; }
  .row > * { flex: 1 1 140px; }
  ul.clean { list-style: none; padding: 0; margin: 0; }
  li.hit, article, .card { background: var(--card); border: 1px solid var(--line); padding: 0.7rem 0.9rem; margin: 0.45rem 0; }
  button.link { font: inherit; background: none; border: 0; color: var(--accent); padding: 0; cursor: pointer; text-align: left; }
  .tag { display: inline-block; border: 1px solid var(--line); padding: 0 0.35rem; margin-right: 0.25rem; font-size: 0.82rem; }
  .muted { color: var(--muted); }
  .question-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 280px), 1fr)); gap: 0.5rem; }
  .question-grid article { margin: 0; }
  summary { cursor: pointer; }
  table { width: 100%; border-collapse: collapse; font-size: 0.92rem; }
  td, th { border-top: 1px solid var(--line); text-align: left; vertical-align: top; padding: 0.35rem 0.3rem; }
  .jaa { font-weight: 700; }
  @media (max-width: 700px) { body { font-size: 16px; } table { font-size: 0.85rem; } }
</style>
</head><body><header><nav><a href="#haku">Haku</a><a href="#kattavuus">Aineisto ja rajat</a></nav>
<h1>Poliittinen muisti</h1><p>Mikä päätöksessä muuttui? Mitä huomautukselle tapahtui? Mitä kukin sanoi ja teki?</p>
<p><a href="cases/index.html">Tutki päätöstä: mikä muuttui ja mitä huomautukselle tapahtui?</a></p>
<p><a href="episodes/index.html">Kirjallisten kysymysten käsittely: henkilöt, toimielimet ja tapahtumien kulku</a></p>
<p class="muted" id="lede"></p></header><main>
<!-- decision-questions -->
<section id="tapaukset"><details><summary>Lausumasta omaan tekoon: tarkistetut esimerkit</summary><div id="traces"></div></details></section>
<section id="haku"><h2>Etsi henkilö tai asia</h2><label for="q">Nimi, puolue, vaalipiiri, asia tai lähdetunniste</label>
<input id="q" type="search" autocomplete="off"><p><label><input id="others" type="checkbox"> Näytä myös ehdokkaat, joita ei valittu aineiston vaaleissa</label></p>
<p id="count" aria-live="polite">Ladataan henkilöhaun aineistoa…</p><ul class="clean" id="hits"></ul><button id="more" hidden>Näytä lisää</button></section>
<p id="person-status" role="status" hidden>Haetaan henkilön lähteitä…</p>
<section id="henkilo" hidden><p><button class="link" id="back">Takaisin hakuun</button></p><h2 id="person-name" tabindex="-1"></h2><div id="person-body"></div></section>
<section id="kattavuus"><h2>Aineisto ja rajat</h2><div id="coverage"></div></section><p id="error" role="alert"></p></main>
<script>
const $ = id => document.getElementById(id);
const esc = value => String(value ?? "").replace(/[&<>"']/g, ch => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[ch]));
let index = [], limit = 40;
const searchable = new Map();
let activePerson = null;
let routeGeneration = 0;
async function read(path) { const r = await fetch(path); if (!r.ok) throw new Error("Aineiston lataus epäonnistui (" + r.status + ")."); return r.json(); }
function fail(error) { $("error").textContent = error.message + " Käynnistä README-ohjeen paikallinen palvelin."; }
function render() {
 const q = $("q").value.trim().toLocaleLowerCase("fi");
 const rows = index.filter(p => ($("others").checked || p.elected) && (!q || searchable.get(p.id).includes(q)));
 $("count").textContent = "Näytetään " + Math.min(limit, rows.length) + " / " + rows.length + " henkilöä";
 $("hits").innerHTML = rows.slice(0,limit).map(p => '<li class="hit"><button class="link" data-id="'+esc(p.id)+'">'+esc(p.name)+'</button> <span class="muted">'+esc(p.latest_year+" "+p.party+" "+p.district)+" · "+p.n_texts+' tekstiä</span></li>').join("");
 $("more").hidden = limit >= rows.length;
}
async function openPerson(id, generation) {
 const p = await read("people/"+encodeURIComponent(id)+".json");
 if(generation!==routeGeneration) return;
 if(p.id!==id) throw new Error("Henkilölähteen tunniste ei vastaa valittua henkilöä.");
 activePerson = p;
 $("henkilo").hidden = false; $("person-name").textContent = p.name;
 let body = '<p>'+p.candidacies.map(c => esc(c.election_year+" "+c.party+" "+c.district_abbr+(c.elected?", valittu":""))).join("<br>")+'</p><section aria-labelledby="issue-heading"><h3 id="issue-heading">Mitä hän sanoi ja teki tästä asiasta?</h3><label for="person-issue">Hae henkilön alkuperäisistä teksteistä, aloitteista, kysymyksistä ja puheenvuoroista</label><input id="person-issue" type="search" autocomplete="off"><p class="muted">Sanahaku rajaa lähderivejä. Yhteys vaalitekstin ja teon välillä näkyy vain varmennetussa tarkistuspolussa.</p><div id="issue-results" aria-live="polite"></div></section><h3>Vaalitekstit ja tarkistuspolut</h3><p class="muted">Tekstin luokittelu on alustava. Tarkistuspolku kertoo, mitkä asiakirjayhteydet on varmennettu.</p>';
 if (!p.statements.length) body += '<p>Kampanjatekstiä ei ole tässä aineistossa.</p>';
 for (const s of p.statements) {
  body += '<article><p class="muted">'+esc(s.when)+' · '+esc(s.field)+'</p><p>'+esc(s.text)+'</p>';
  for (const t of s.traces) body += '<p><span class="tag">'+esc(t.type_label)+'</span> '+esc(t.text)+'</p><p>'+esc(t.state_label)+': '+esc(t.claim)+'</p><p><a href="traces/'+encodeURIComponent(t.id)+'.html">Katso lähteet, kohde, toimivalta ja päättelyn rajat</a></p>';
  if(s.url) body += '<p><a href="'+esc(s.url)+'">Alkuperäinen lähde</a></p>';
  body += '</article>';
 }
 body += '<h3>Kirjatut parlamentaariset teot</h3><p class="muted">Aloitteet, kysymykset ja puheenvuorot näkyvät haetun lähdeaineiston rajoissa. Ensimmäinen allekirjoittaja ja muut allekirjoittajat erotetaan; rekisteri ei osoita tekstin kirjoittajaa. Aloite tai hallituksen vastaus ei osoita politiikan toteutumista. Näillä riveillä ei päätellä yhteyttä vaaliteksteihin.</p>';
 if(!p.initiatives.length) body += '<p>Ei toimintariviä tässä haetussa lähdeaineistossa. Tämä ei osoita parlamentaarisen työn puuttumista.</p>';
 const initiativeCard=a => '<article><p>'+esc(a.kind_label)+' · '+esc(a.date)+' · '+esc(a.matter_id)+' · '+esc(a.role_label)+'</p><p><a href="objects/'+encodeURIComponent(a.id)+'.html">'+esc(a.title)+'</a></p><p>'+esc(a.disposition_label)+'</p></article>';
 body += p.initiatives.filter(a=>a.role==="AUTHOR").map(initiativeCard).join("");
 const spoken=p.initiatives.filter(a=>a.kind==="SPEECH");
 if(spoken.length) body += '<details><summary>Eduskunnan puheenvuorot ('+spoken.length+')</summary>'+spoken.map(initiativeCard).join("")+'</details>';
 const signed=p.initiatives.filter(a=>a.role!=="AUTHOR" && a.kind!=="SPEECH");
 if(signed.length) body += '<details><summary>Muut allekirjoitetut asiakirjat ('+signed.length+')</summary>'+signed.map(initiativeCard).join("")+'</details>';
 if(p.group_context_url) body += '<p><a href="'+esc(p.group_context_url)+'">Miten kirjatut äänet suhteutuvat muiden saman ryhmän edustajien valintoihin?</a></p>';
 body += '<details><summary>Muut kirjatut äänet ('+p.n_ballots+')</summary><p>Nämä äänet eivät ole tämän sivun vaalitekstien varmennettuja vastineita. Alla viimeisimmät '+p.votes.length+' äänet, joiden otsikko nimeää kaksi vaihtoehtoa.</p>';
 body += p.votes.map(v => '<article><p>'+esc(v.date)+'</p><p>'+esc(v.sentence)+'</p><a href="'+esc(v.url)+'">Pöytäkirja</a></article>').join("")+'</details>';
 $("person-body").innerHTML=body; $("person-name").focus();
 $("person-issue").addEventListener("input", renderPersonIssue);
}
function renderPersonIssue() {
 const query=$("person-issue").value.trim().toLocaleLowerCase("fi");
 const result=$("issue-results");
 if(!query || !activePerson) { result.innerHTML=""; return; }
 const terms=query.split(/\\s+/).filter(Boolean);
 const matches=text=>terms.every(term=>String(text||"").toLocaleLowerCase("fi").includes(term));
 const statements=activePerson.statements.filter(s=>matches(s.text));
 const actions=activePerson.initiatives.filter(a=>matches(a.source_text+" "+a.title+" "+a.matter_id));
 let body='<p>'+statements.length+' vaalitekstiä ja '+actions.length+' toimintariviä vastaa hakua tämän henkilön haetussa aineistossa.</p>';
 if(!statements.length && !actions.length) body+='<p>Sanahaku ei löytänyt lähderiviä. Se ei osoita kannan tai toiminnan puuttumista.</p>';
 for(const s of statements) body+='<article><p>'+esc(s.when)+'</p><p>'+esc(s.text)+'</p>'+s.traces.map(t=>'<p><a href="traces/'+encodeURIComponent(t.id)+'.html">'+esc(t.state_label)+': '+esc(t.claim)+'</a></p>').join('')+'</article>';
 for(const a of actions.slice(0,40)) {
  const text=String(a.source_text||''); const start=Math.max(0,text.toLocaleLowerCase('fi').indexOf(terms[0])-80);
  body+='<article><p>'+esc(a.kind_label)+' · '+esc(a.date)+' · '+esc(a.role_label)+'</p><p><a href="objects/'+encodeURIComponent(a.id)+'.html">'+esc(a.title)+'</a></p><blockquote>'+esc(text.slice(start,start+500))+'</blockquote></article>';
 }
 if(actions.length>40) body+='<p>Näytetään ensimmäiset 40 toimintariviä. Tarkenna hakua nähdäksesi rajatumman joukon.</p>';
 result.innerHTML=body;
}
async function route() {
 const generation=++routeGeneration;
 activePerson=null; $("henkilo").hidden=true; $("error").textContent="";
 const requested=location.hash.startsWith("#person=");
 $("person-status").hidden=!requested;
 try { if(requested) await openPerson(decodeURIComponent(location.hash.slice(8)),generation); }
 catch(error) { if(generation===routeGeneration) throw error; }
 finally { if(generation===routeGeneration) $("person-status").hidden=true; }
}
async function load() {
 const d=await read("index.json"); index=d.people;
 for(const p of index) searchable.set(p.id,(p.name+" "+p.party+" "+p.district+" "+p.search_text).toLocaleLowerCase("fi"));
 $("lede").textContent="Rajaus "+d.cutoff+". Analyysin tila: osittainen.";
 $("traces").innerHTML=d.traces.map(t => '<article><p><a href="traces/'+encodeURIComponent(t.id)+'.html">'+esc(t.name)+' · '+esc(t.state_label)+'</a></p><p>'+esc(t.claim)+'</p></article>').join("") || '<p>Varmennettua omaa tekoa ei ole vielä tässä aineistossa.</p>';
 const years=Object.entries(d.coverage.candidacies_by_year).map(([y,c]) => '<li>'+esc(y)+': '+c.candidates+' ehdokasta, '+c.elected+' valittu</li>').join("");
 $("coverage").innerHTML='<ul>'+years+'</ul><p>Vaalitekstejä '+d.coverage.statements+'. Äänirivejä '+d.coverage.ballots+'. Aloite- ja toimintalähteitä '+d.coverage.official_objects+'.</p><p>Sanahaku tuottaa vain ehdokkaita. Varmennettu yhteys perustuu lähdekohtien tarkistukseen. Rekisterin puuttuva rivi ei osoita tekemättä jättämistä. Henkilön teko ja toimielimen ratkaisu näkyvät erikseen.</p><p>Puheet, kysymykset, aloitteet ja toteutusvaikutukset eivät ole kattava aineisto. Tarkka lähde- ja aikarajaus näkyy tarkistuspolussa. Nimettyjä vaalitekstejä on 2011 ja 2023; muut vaalivuodet ovat ehdokasluetteloita.</p>';
 $("coverage").innerHTML+=(d.source_attributions||[]).map(s=>'<p>Lähde: <a href="'+esc(s.url)+'">'+esc(s.label)+'</a> · <a href="'+esc(s.license_url)+'">'+esc(s.license)+'</a>.</p>').join('');
 render(); await route();
}
document.body.addEventListener("click", e => { const b=e.target.closest("button[data-id]"); if(b) location.hash="person="+encodeURIComponent(b.dataset.id); });
$("q").addEventListener("input",()=>{limit=40;render();}); $("others").addEventListener("change",()=>{limit=40;render();});
$("more").addEventListener("click",()=>{limit+=40;render();}); $("back").addEventListener("click",()=>{location.hash="haku";});
window.addEventListener("hashchange",()=>route().catch(fail)); load().catch(fail);
</script></body></html>"""


def _attribution_html(packet: dict, object_ids: set[str]) -> str:
    """Render independent backend assertions; no actor or policy inference."""
    def e(value):
        return html.escape(str(value or ""), quote=True)

    carrier = packet.get("commitment_carrier")
    if not carrier:
        return ""
    carriers = {"PERSON": "henkilö", "PARTY": "puolue", "PARLIAMENTARY_GROUP": "eduskuntaryhmä",
                "COALITION": "koalitio", "GOVERNMENT": "hallitus", "MINISTRY": "ministeriö",
                "PARLIAMENT": "eduskunta", "OTHER_INSTITUTION": "muu toimielin", "UNRESOLVED": "tekijä jää avoimeksi"}
    body = '<h2>Kenen lausuma tai sitoumus?</h2><p>Tekstin ilmaisema toimijataso: '+e(carriers.get(carrier["type"], "avoin"))+'.</p>'
    if carrier.get("commitment_state") == "PROPOSED":
        body += '<p>Sitoumuksen luokka on alustava tulkinta. Henkilölle kirjattu lausuma ei yksin osoita, että hän lupasi tai hallitsi sen lopputuloksen.</p>'
    if carrier["type"] == "UNRESOLVED":
        body += '<p>Tekstistä ei ole ratkaistu, kuka kantaa mahdollisen sitoumuksen. Passiivista tavoitetta ei siirretä henkilön omaksi teoksi.</p>'
    labels = {"said": "Lausuma", "recorded_action": "Kirjattu oma teko", "first_signatory": "Ensimmäinen allekirjoittaja",
              "cosigned": "Muu allekirjoittaja", "drafter": "Tekstin laatija", "authority": "Rooli ja toimintamahdollisuus",
              "institutional_decision": "Toimielimen ratkaisu", "exact_support_opposition": "Täsmällisen vaihtoehdon kannatus tai vastustus",
              "institutional_responsibility": "Henkilön institutionaalinen vastuu", "implementation_responsibility": "Toteutusvastuu",
              "documented_constraints": "Dokumentoitu poliittinen rajoite", "pivotality": "Henkilön ratkaisevuus päätöksessä",
              "causal_effect": "Henkilön aiheuttama vaikutus", "private_constraints": "Yksityinen painostus tai vaikutus",
              "commitment_fulfillment": "Koko sitoumuksen toteutuminen"}

    def describe(key, value):
        if key == "said":
            return e(value)
        if key == "authority":
            roles = (value or {}).get("roles", [])
            role_text = "; ".join(str(r.get("label") or r.get("kind")) + " " + str(r.get("start_date") or "?") + " – " + str(r.get("end_date") or "jatkuu rajaukseen") for r in roles)
            opportunity = OPPORTUNITIES.get((value or {}).get("opportunity_state"), "mahdollisuus jää avoimeksi")
            return e((role_text + ". " if role_text else "") + opportunity)
        if isinstance(value, list):
            return '<br>'.join(describe(key, item) for item in value)
        if isinstance(value, dict):
            parts = [value.get("name"), value.get("date"), KINDS.get(value.get("kind")),
                     DISPOSITIONS.get(value.get("state"))]
            text = e(" · ".join(str(v) for v in parts if v))
            object_id = value.get("object_id")
            if object_id in object_ids:
                text += f' <a href="../objects/{e(_safe(object_id))}.html">{e(object_id)}</a>'
            return text or e(value.get("role") or "Lähteessä kirjattu havainto")
        return e(value)

    for envelope in packet.get("attribution_envelopes", []):
        body += '<h2>Mitä henkilölle voidaan osoittaa tässä asiassa?</h2><p>'+e(envelope.get("matter_id"))+'</p><table><tbody>'
        unknown = []
        for key, label in labels.items():
            dimension = envelope["dimensions"].get(key, {})
            if dimension.get("state") == "SUPPORTED":
                refs = " · ".join(dimension.get("evidence_ids", []))
                body += '<tr><th scope="row">'+e(label)+'</th><td>'+describe(key, dimension.get("value"))+'<br><small>'+e(refs)+'</small></td></tr>'
            else:
                unknown.append(label)
        body += '</tbody></table><details><summary>Mitä ei ole osoitettu tästä aineistosta?</summary><ul>'+''.join('<li>'+e(label)+'</li>' for label in unknown)+'</ul></details>'
    if not packet.get("attribution_envelopes"):
        body += '<p>Varmennettua yhteyttä päätösasiaan ei ole. Henkilön osuutta päätökseen tai sen vaikutuksiin ei siksi päätellä tästä lausumasta.</p>'
    return body


def _trace_page(packet: dict, name: str, object_ids: set[str] | None = None) -> str:
    def e(value):
        return html.escape(str(value or ""), quote=True)
    def source(ref):
        url = _url(ref.get("url", ""))
        return f'<a href="{e(url)}">{e(ref["evidence_id"])}</a>' if url else e(ref["evidence_id"])
    assessment = packet["assessment"]
    body = f'<h1>{e(name)}</h1><p>{e(_state_label(packet))}</p><h2>Lausuma ja tulkittu osa</h2><p>{e(packet["statement"]["text"])}</p><p>Alustava tekstin luokka – {e(PLAIN.get(packet["proposition"]["semantic_type"]))}: {e(packet["proposition"]["text"])}</p>'
    provenance = packet["proposition"].get("interpretation_provenance") or {}
    if str(packet["proposition"].get("run_id") or "").startswith("local-model:"):
        body += '<p class="muted">Tekstin luokka on paikallisen kielimallin ehdotus. Lähdeankkurien tarkistus ei varmista tulkinnan oikeellisuutta. '+e(provenance.get("note"))+'</p>'
    body += f'<h2>Rajattu päätelmä</h2><p>{e(assessment["claim"])}</p><ul>'+''.join(f'<li>{e(x)}</li>' for x in assessment["limitations"])+ '</ul>'
    body += _attribution_html(packet, object_ids or set())
    target = packet["target"]
    body += '<h2>Toimintakohde</h2>'
    body += '<p>'+e(target["normalized_object"])+'</p>' if target["interpretation_state"] == "REVIEWED" else '<p>Lausuman ja asiakirjan yhteistä toimintakohdetta ei ole varmennettu.</p>'
    authority = packet["authority"]
    body += f'<h2>Toimivalta ja aikaikkuna</h2><p>Ehto: {e(CONDITIONS.get(authority["condition_state"], authority["condition_state"]))}. Mahdollisuus: {e(OPPORTUNITIES.get(authority["opportunity_state"], authority["opportunity_state"]))}. Vaadittu toimivalta: {e(authority.get("required_role"))}.</p><p>{e(authority["window"]["earliest"])} – {e(authority["window"]["latest"])}</p>'
    for role in authority["roles"]:
        label = "Virallisen päätöksen osoittama rooli" if role["kind"] == "SOURCE_ATTESTED_ROLE" else "Kansanedustajarekisterin roolijakso"
        body += f'<p>{label}: {e(role.get("label"))}, {e(role.get("start_date"))} – {e(role.get("end_date") or "jatkuu aineiston rajaukseen")}</p>'
    body += '<h2>Haetut asiakirjat ja yhteyden tarkistus</h2>'
    relations = {r["object_id"]: r for r in packet["relations"]}
    for obj in packet["retrieved_objects"]:
        relation = relations.get(obj["object_id"], {})
        disp = obj.get("disposition") or {}
        title = e(obj["title"])
        if obj["object_id"] in (object_ids or set()):
            title = f'<a href="../objects/{e(_safe(obj["object_id"]))}.html">{title}</a>'
        elif _url(obj.get("url", "")):
            title = f'<a href="{e(_url(obj["url"]))}">{title}</a>'
        body += f'<article><p>{e(obj.get("action_date") or obj.get("date"))} · {e(obj.get("matter_id"))} · {title}</p><p>Yhteys: {e(RELATIONS.get(relation.get("status"), relation.get("status")))}. {e(relation.get("rationale"))}</p>'
        if relation.get("statement_quote"):
            qualifier = "Varmennetun yhteyden" if relation.get("validation_state") == "VALID" else "Varmistamattoman ehdotuksen"
            body += f'<p>{qualifier} lausumakohta:</p><blockquote>{e(relation["statement_quote"])}</blockquote><p>{qualifier} asiakirjakohta:</p><blockquote>{e(relation.get("object_quote"))}</blockquote><p>Tarkistus: {e(relation.get("review_id"))}.</p>'
        body += f'<p>Toimielimen ratkaisu: {e(DISPOSITIONS.get(disp.get("state"), disp.get("raw_state") or "Ratkaisua ei varmennettu tästä lähteestä"))}</p></article>'
    if not packet["retrieved_objects"]:
        body += '<p>Ei asiakirjaehdokkaita tämän haun tuloksissa.</p>'
    body += '<h2>Havaitut omat teot</h2>'
    body += ''.join(f'<p>{e(a["date"])} · {e(KINDS.get(a["kind"], a["kind"]))} · {e(_role_label(a["kind"], a["role"]))} · {e(a["object_id"])}</p>' for a in packet["actions"]) or '<p>Omaa tekoa ei varmennettu. Tämä ei osoita tekemättä jättämistä.</p>'
    body += '<h2>Lähdekohdat ja pysyvät tunnisteet</h2>'
    for ref in packet["evidence"]:
        quote = str(ref.get("quote") or "")
        source_text = '<blockquote>'+e(quote)+'</blockquote>'
        if len(quote) > 700:
            source_text = '<details><summary>Näytä koko lähdepoiminta</summary>'+source_text+'</details>'
        body += '<article><p>'+source(ref)+'</p>'+source_text+'<p class="muted">'+e(ref.get("record_locator"))+'</p></article>'
    for attribution in _attributions({ref.get("source_id") for ref in packet["evidence"]}):
        body += f'<p>Lähde: <a href="{e(attribution["url"])}">{e(attribution["label"])}</a> · <a href="{e(attribution["license_url"])}">{e(attribution["license"])}</a>.</p>'
    body += '<h2>Lähdehaun kattavuus</h2>'
    if not packet["coverage"]:
        body += '<p>Haun kattavuutta osoittavaa todistetta ei ole liitetty. Puuttuva hakutulos ei osoita tekemättä jättämistä.</p>'
    for coverage in packet["coverage"]:
        body += f'<p>{e(coverage.get("source_id"))}: {e(coverage.get("object_count", coverage.get("retrieved_count", "rajattu")))} asiakirjakohdetta. {e(coverage.get("coverage_id"))}</p><ul>'
        body += ''.join('<li>'+e(limit)+'</li>' for limit in coverage.get("scope_residuals", []) + coverage.get("limitations", [])) + '</ul>'
    body += '<details><summary>Näytä kattavuustodisteiden tietueet</summary><pre>'+e(json.dumps(packet["coverage"], ensure_ascii=False, indent=2))+'</pre></details>'
    body += f'<p>Tarkistuspolku {e(packet["trace_id"])}. Rajaus {e(packet["as_of"])}. <a href="{e(_safe(packet["trace_id"]))}.json">Lataa koko näyttöpaketti</a>.</p>'
    person_navigation = '<a href="../index.html#person='+e(_safe(packet["actor_id"]))+'">Henkilö</a> · ' if packet.get("actor_id") else ''
    return '<!DOCTYPE html><html lang="fi"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+e(name)+' – Poliittinen muisti</title><style>body{max-width:900px;margin:2rem auto;padding:1rem;font:17px/1.5 Georgia;background:#f6f3ed;overflow-wrap:anywhere}a{color:#1f4d3a}article{border:1px solid #ddd;padding:1rem;margin:1rem 0}pre{white-space:pre-wrap;overflow-wrap:anywhere}.muted{color:#5c564c}table{width:100%;border-collapse:collapse}th,td{text-align:left;vertical-align:top;padding:.6rem;border-top:1px solid #ddd}th{width:30%}small{font-size:.8em}</style><nav>'+person_navigation+'<a href="../index.html">Haku</a> · <a href="../episodes/index.html">Päätös- ja käsittelytapaukset</a></nav><main>'+body+'</main></html>'


def _object_page(obj: dict, evidence: dict, available_objects: set[str] | None = None) -> str:
    def e(value):
        return html.escape(str(value or ""), quote=True)
    disposition = obj.get("disposition") or {}
    body = f'<h1>{e(obj["matter_id"])}</h1><p>{e(obj["title"])}</p><p>Asiakirjan päivä: {e(obj.get("date"))}.</p>'
    if obj.get("action_date"):
        basis = {"INSTITUTIONAL_DECISION_DATE": "institutionaaliseen päätökseen", "SIGNATURE_DATE": "allekirjoituspäivään",
                 "VIREILLETULO_EVENT": "asian vireilletulomerkintään", "SUBMISSION_DATE": "kysymyksen jättämismerkintään",
                 "ANSWER_EVENT": "vastauksen päivämerkintään", "SPEECH_DATE": "puheenvuoron aikaleimaan", "SESSION_DATE": "istunnon päivään"}
        body += f'<p>Kirjatun teon päivä: {e(obj["action_date"])}. Päivä perustuu {e(basis.get(obj.get("action_date_basis"), "lähteen päivämerkintään"))}.</p>'
    body += '<h2>Kirjatut henkilöt</h2><ul>'
    body += ''.join(f'<li>{e(a.get("name"))}: {e(_role_label(obj["kind"], a.get("role")))}</li>' for a in obj.get("authors", [])) + '</ul>'
    if obj.get("answer_object_id") and available_objects and obj["answer_object_id"] in available_objects:
        body += f'<p><a href="{e(_safe(obj["answer_object_id"]))}.html">Hallituksen vastaus tähän kysymykseen</a>. Vastaus ei osoita politiikan toteutumista.</p>'
    if obj["kind"] == "GOVERNMENT_ANSWER" and available_objects and "eduskunta:" + obj["matter_id"] in available_objects:
        body += f'<p><a href="{e(_safe("eduskunta:" + obj["matter_id"]))}.html">Kirjallinen kysymys</a>.</p>'
    body += f'<h2>Toimielimen ratkaisu</h2><p>{e(DISPOSITIONS.get(disposition.get("state"), disposition.get("raw_state") or "Ratkaisua ei varmennettu tästä lähteestä"))}. {e(disposition.get("date"))}</p>'
    body += '<p>Aloitteen jättäminen, allekirjoittaminen ja toimielimen ratkaisu ovat eri havaintoja. Ne eivät osoita toteutusvaikutusta.</p><h2>Lähdekohdat</h2>'
    for key in obj.get("evidence_ids", []):
        ref = evidence.get(key, {})
        url = _url(ref.get("url") or ref.get("source_url") or obj.get("url") or "")
        source_hash = ref.get("raw_sha256") or ref.get("source_raw_sha256") or obj.get("source_raw_sha256")
        body += f'<article><p><a href="{e(url)}">{e(key)}</a> · {e(ref.get("record_locator"))}</p><p>Lähteen SHA-256: {e(source_hash)}</p><details><summary>Asiakirjan poimittu teksti</summary><p>{e(ref.get("quote"))}</p></details></article>'
    body += f'<p><a href="{e(_safe(obj["object_id"]))}.json">Lataa asiakirjan tietue ja lähdetunnisteet</a>.</p>'
    return '<!DOCTYPE html><html lang="fi"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+e(obj["matter_id"])+'</title><style>body{max-width:900px;margin:2rem auto;padding:1rem;font:17px/1.5 Georgia;background:#f6f3ed;overflow-wrap:anywhere}article{border:1px solid #ddd;padding:1rem;margin:1rem 0}a{color:#1f4d3a}</style><nav><a href="../index.html">Haku</a></nav><main>'+body+'</main></html>'


def _write_research_packets(paths: list[Path], destination: Path) -> None:
    """Expose completed research exports without altering canonical findings."""
    from paa.research_site import RESEARCH_STATUS, SCHEMA_VERSION, render_research_html

    destination.mkdir()
    entries = []
    for path in paths:
        content = path.read_bytes()
        packet = json.loads(content)
        if (packet.get("schema_version") != SCHEMA_VERSION
                or packet.get("research_status") != RESEARCH_STATUS
                or packet.get("not_model_admission") is not True):
            raise ValueError(f"Not a research-only inquiry packet: {path}")
        for episode in packet.get("episodes", []):
            for mode in episode.get("modes", {}).values():
                if mode.get("admission_state") != "PROPOSED / NOT_ADMITTED":
                    raise ValueError(f"Research packet contains a non-proposal: {path}")
        slug = hashlib.sha256(content).hexdigest()[:24]
        (destination / f"{slug}.json").write_bytes(content)
        rendered = render_research_html(packet).replace(
            "<h1>", f'<nav><a href="index.html">Tutkimusvertailut</a> · '
            f'<a href="{slug}.json">Lataa lähdepaketti</a></nav><h1>', 1,
        )
        (destination / f"{slug}.html").write_text(rendered, encoding="utf-8")
        entries.append(f'<li><a href="{slug}.html">{html.escape(str(packet.get("label") or path.stem))}</a>'
                       f' — {int(packet.get("case_count", 0))} kysymystä</li>')
    (destination / "index.html").write_text(
        '<!doctype html><html lang="fi"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>Tutkimusvertailut</title><style>body{font:17px/1.5 system-ui;'
        'max-width:900px;margin:2rem auto;padding:1rem;overflow-wrap:anywhere}</style>'
        '<nav><a href="../index.html">Haku</a></nav><h1>Tutkimusvertailut</h1>'
        '<p>Saman lähdeaineiston kaksi mallivastausta. Nämä ovat tutkimusehdotuksia, '
        'eivät hyväksyttyjä päätelmiä. Täsmäotteen tarkistus ei varmista tulkinnan oikeellisuutta.</p>'
        f'<ul>{"".join(entries)}</ul></html>', encoding="utf-8",
    )


def build_from_db(db_path: Path | None = None, output_dir: Path | None = None,
                  *, research_packets: list[Path] | None = None,
                  structure_results: Path | None = None) -> None:
    database = db_path or DB_PATH
    destination = output_dir or DIST
    if not database.exists():
        raise ValueError("Database missing. Run paa frozen for an offline specimen, or acquire sources and compile first.")
    conn = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="paa-site-", dir=destination.parent))
    try:
        (staging / "people").mkdir(); (staging / "traces").mkdir(); (staging / "objects").mkdir()
        names = {r["actor_id"]: r["display_name"] for r in conn.execute("SELECT * FROM actors")}
        initiatives_by_person = {}
        object_rows = list(conn.execute("SELECT json FROM official_objects"))
        object_ids = {json.loads(row["json"])["object_id"] for row in object_rows}
        source_refs = {r["evidence_id"]: json.loads(r["json"]) for r in conn.execute("SELECT * FROM evidence")}
        from paa.case_site import write_cases

        cases = [json.loads(row["json"]) for row in conn.execute("SELECT json FROM inquiry_cases ORDER BY case_id")]
        write_cases(cases, staging / "cases")
        from paa.group_site import write_group_pages

        group_packets = [json.loads(row["json"]) for row in conn.execute("SELECT json FROM group_contexts")]
        group_sources = [json.loads(row["json"]) for row in conn.execute("SELECT json FROM group_sources")]
        group_report = conn.execute("SELECT value FROM meta WHERE key='group_context_report'").fetchone()
        write_group_pages(group_packets, group_sources, json.loads(group_report["value"]) if group_report else {}, staging / "groups")
        group_people = {packet["person_id"] for packet in group_packets}
        episode_dir = staging / "episodes"
        episode_dir.mkdir()
        episodes = [json.loads(row["json"]) for row in conn.execute("SELECT json FROM decision_episodes ORDER BY matter_id")]
        for episode in episodes:
            slug = _safe(episode["episode_id"])
            (episode_dir / (slug + ".json")).write_text(json.dumps(episode, ensure_ascii=False), encoding="utf-8")
            episode_refs = {key: source_refs[key] for key in episode["evidence_ids"] if key in source_refs}
            (episode_dir / (slug + ".html")).write_text(render_episode(episode, episode_refs, object_ids, names), encoding="utf-8")
        (episode_dir / "index.html").write_text(render_episode_index(episodes), encoding="utf-8")
        for row in object_rows:
            obj = json.loads(row["json"])
            object_ids.add(obj["object_id"])
            slug = _safe(obj["object_id"])
            (staging / "objects" / (slug + ".json")).write_text(row["json"], encoding="utf-8")
            (staging / "objects" / (slug + ".html")).write_text(_object_page(obj, source_refs, object_ids), encoding="utf-8")
            if obj["kind"] not in {"LEGISLATIVE_INITIATIVE", "WRITTEN_QUESTION", "SPEECH"}:
                continue
            for author in obj.get("authors", []):
                summary = {"id": slug, "matter_id": obj["matter_id"], "title": obj["title"], "date": obj.get("action_date"),
                           "source_text": obj.get("text", ""), "text_sha256": obj.get("text_sha256"),
                           "kind": obj["kind"], "kind_label": KINDS[obj["kind"]],
                           "role": author["role"], "role_label": "Puhuja" if obj["kind"] == "SPEECH" else "Ensimmäinen allekirjoittaja" if author["role"] == "AUTHOR" else "Allekirjoittaja",
                           "disposition_label": DISPOSITIONS.get((obj.get("disposition") or {}).get("state"), "Ratkaisua ei varmennettu tästä lähteestä")}
                initiatives_by_person.setdefault(str(author.get("person_id")), []).append(summary)
        traces = {}; discovered = []
        for row in conn.execute("SELECT * FROM evidence_traces ORDER BY trace_id"):
            packet = json.loads(row["json"]); slug = _safe(packet["trace_id"])
            (staging / "traces" / (slug + ".json")).write_text(row["json"], encoding="utf-8")
            (staging / "traces" / (slug + ".html")).write_text(_trace_page(packet, names.get(row["actor_id"], "Henkilö avoin"), object_ids), encoding="utf-8")
            summary = {"id": slug, "text": packet["proposition"]["text"], "type_label": PLAIN.get(packet["proposition"]["semantic_type"], packet["proposition"]["semantic_type"]),
                       "actor_id": row["actor_id"], "has_recorded_action": bool(packet["actions"]),
                       "has_reviewed_relation": any(r.get("validation_state") == "VALID" for r in packet["relations"]),
                       "state": packet["assessment"]["state"], "state_label": _state_label(packet), "claim": packet["assessment"]["claim"]}
            traces.setdefault(row["statement_id"], []).append(summary)
            if packet["actions"] or summary["state"] in {"OBSERVED_ALIGNED_ACTION", "NO_OBSERVABLE_OPPORTUNITY"} or (
                summary["state"] == "INSUFFICIENT_EVIDENCE" and packet["retrieved_objects"]
            ):
                discovered.append({**summary, "name": names.get(row["actor_id"], "Henkilö avoin")})
        docs = {r["document_id"]: dict(r) for r in conn.execute("SELECT * FROM documents")}
        by_actor = {}
        for row in conn.execute("SELECT json FROM statements"):
            s = json.loads(row["json"]); d = docs[s["statement_id"]]
            temporal = s["stated_at"]
            when = temporal.get("earliest") if temporal["precision"] == "day" else f"Vaalitekstin aikarajaus {temporal.get('earliest') or '?'} – {temporal.get('latest') or '?'}; tarkka päivä avoin"
            item = {"id": s["statement_id"], "when": when, "field": s["source_field_label"], "text": s["original_text"], "url": _url(d["url"]), "traces": traces.get(s["statement_id"], [])}
            for actor_id in s["issuer_actor_ids"]:
                by_actor.setdefault(actor_id, []).append(item)
        index = []
        for actor in conn.execute("SELECT * FROM actors ORDER BY display_name"):
            candidacies = [dict(r) for r in conn.execute("SELECT election_year, district_abbr, party, candidate_number, votes, elected, home_municipality FROM candidacies WHERE actor_id=? ORDER BY election_year", (actor["actor_id"],))]
            n_ballots = conn.execute("SELECT COUNT(*) FROM ballots WHERE person_number=?", (actor["person_id"],)).fetchone()[0] if actor["person_id"] else 0
            votes = []
            for b in conn.execute("SELECT v.title,v.session_date,v.url,b.raw_response FROM ballots b JOIN vote_events v ON v.aanestys_id=b.aanestys_id WHERE b.person_number=? AND v.mitatoity=0 ORDER BY v.session_date DESC", (actor["person_id"],)):
                sentence = _choice(b["title"], b["raw_response"])
                if sentence:
                    votes.append({"date": b["session_date"], "sentence": sentence, "url": _url(b["url"])})
                if len(votes) >= 12:
                    break
            slug = _safe(actor["actor_id"])
            payload = {"id": slug, "name": actor["display_name"], "status": actor["identity_status"], "person_id": actor["person_id"], "candidacies": candidacies, "statements": by_actor.get(actor["actor_id"], []), "votes": votes, "n_ballots": n_ballots,
                       "group_context_url": "groups/person-"+_safe(actor["person_id"])+".html" if actor["person_id"] in group_people else None,
                       "initiatives": sorted(initiatives_by_person.get(str(actor["person_id"]), []), key=lambda a: (a["date"] or "", a["matter_id"]), reverse=True)}
            (staging / "people" / (slug+".json")).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            elected = [c["election_year"] for c in candidacies if c["elected"]]
            latest_year = max((c["election_year"] for c in candidacies), default=0)
            latest_candidacies = [c for c in candidacies if c["election_year"] == latest_year]
            search_text = "\n".join(dict.fromkeys([s["id"] + " " + s["text"] for s in payload["statements"]] + [a["matter_id"] + " " + a["title"] for a in payload["initiatives"]] + [f"{c['election_year']} {c['party']} {c['district_abbr']}" for c in candidacies]))
            index.append({"id": slug, "name": actor["display_name"], "latest_year": latest_year,
                          "party": " ".join(sorted({c["party"] for c in latest_candidacies if c["party"]})),
                          "district": " ".join(sorted({c["district_abbr"] for c in latest_candidacies if c["district_abbr"]})),
                          "elected": bool(elected), "latest_elected": max(elected, default=0), "n_texts": len(payload["statements"]), "search_text": search_text})
        coverage = {"candidacies_by_year": {str(r["election_year"]): {"candidates": r["n"], "elected": r["elected"]} for r in conn.execute("SELECT election_year, COUNT(*) n, SUM(elected) elected FROM candidacies GROUP BY 1")}}
        for table in ("statements", "ballots", "official_objects"):
            coverage[table] = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        index.sort(key=lambda p: (-p["latest_elected"], p["name"]))
        discovered.sort(key=lambda p: (p["state"] != "OBSERVED_ALIGNED_ACTION", not p["has_recorded_action"], not p["has_reviewed_relation"], p["name"]))
        selected = []
        seen_actors = set()
        for p in discovered:
            if p["has_recorded_action"] and p["actor_id"] not in seen_actors:
                selected.append(p)
                seen_actors.add(p["actor_id"])
            if len(selected) >= 6:
                break
        for state in ("OBSERVED_ALIGNED_ACTION", "NO_OBSERVABLE_OPPORTUNITY", "INSUFFICIENT_EVIDENCE"):
            group = 0
            for p in discovered:
                if p["state"] == state and p["actor_id"] not in seen_actors and (
                    state != "INSUFFICIENT_EVIDENCE" or p["has_reviewed_relation"]
                ):
                    selected.append(p)
                    seen_actors.add(p["actor_id"])
                    group += 1
                if group >= 3:
                    break
        (staging / "index.json").write_text(json.dumps({"cutoff": "2026-10-06", "people": index, "traces": selected, "coverage": coverage,
            "source_attributions": _attributions({doc["source_id"] for doc in docs.values()})}, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        question_cards = ''.join(
            '<article><a href="cases/' + quote(packet['case_id'], safe='') + '.html">'
            + html.escape(packet['title']) + '</a></article>'
            for packet in cases
        )
        decision_questions = (
            '<section id="paatokset"><h2>Tutki päätöstä</h2>'
            '<p>Vastaukset, ratkaisevat alkuperäiset katkelmat ja avoimet kysymykset.</p>'
            '<div class="question-grid">' + question_cards + '</div></section>'
        ) if question_cards else ''
        home_html = HTML.replace('<!-- decision-questions -->', decision_questions)
        if research_packets:
            _write_research_packets(research_packets, staging / "research")
            home_html = home_html.replace(
                '<p><a href="cases/index.html">',
                '<p><a href="research/index.html">Tutkimusvertailut: mitä mallin lähdevastaus lisää?</a></p>'
                '<p><a href="cases/index.html">', 1,
            )
        if structure_results:
            from paa.structure_probe import render

            render(json.loads(structure_results.read_text(encoding="utf-8")), staging / "structure")
            home_html = home_html.replace(
                '<p><a href="cases/index.html">',
                '<p><a href="structure/comparison.html">Lähderakennevertailut: mallien tutkimusehdotukset</a></p>'
                '<p><a href="cases/index.html">', 1,
            )
        (staging / "index.html").write_text(home_html, encoding="utf-8")
        previous = None
        if destination.exists():
            previous = Path(tempfile.mkdtemp(prefix="paa-previous-site-", dir=destination.parent))
            previous.rmdir()
            destination.rename(previous)
        try:
            staging.rename(destination)
        except OSError:
            if previous is not None:
                previous.rename(destination)
            raise
        if previous is not None:
            shutil.rmtree(previous)
    finally:
        conn.close()
        if staging.exists():
            shutil.rmtree(staging)
    print(f"site {destination} people {len(index)}")


def write_site(payload: dict | None = None) -> None:
    build_from_db()

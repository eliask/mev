/* Run with NODE_PATH pointing at an installed playwright-core package.
   PAA_BROWSER_URL and PAA_CHROMIUM_PATH select the local site and browser. */
const {chromium} = require('playwright-core');
const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');
const base = process.env.PAA_BROWSER_URL || 'http://127.0.0.1:8876';
const output = process.env.PAA_BROWSER_REPORT || 'reports/decision_browser_e2e.json';
const sha = text => crypto.createHash('sha256').update(text).digest('hex');
const researchSources = new Map();
for (const fixture of JSON.parse(process.env.PAA_RESEARCH_SOURCE_FIXTURES || '[]')) {
  for (const line of fs.readFileSync(fixture, 'utf8').split('\n').filter(line => line.trim())) {
    for (const source of JSON.parse(line).sources) {
      const previous = researchSources.get(source.source_id);
      assert(!previous || (previous.text === source.text && previous.text_sha256 === source.text_sha256),
        'the source identity must retain the same captured version across fixtures');
      researchSources.set(source.source_id, source);
    }
  }
}

(async () => {
  const browser = await chromium.launch({executablePath:process.env.PAA_CHROMIUM_PATH,
    args:['--no-sandbox']});
  const checks = [], errors = [], external = [];
  try {
    for (const viewport of [{width:1280,height:900},{width:390,height:844}]) {
      const context = await browser.newContext({viewport});
      await context.route('**/*', async route => {
        if (new URL(route.request().url()).origin === new URL(base).origin) await route.continue();
        else {external.push(route.request().url()); await route.abort();}
      });
      const page = await context.newPage();
      page.on('pageerror', error => errors.push(error.message));
      async function fits(label) {
        assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false, label);
        checks.push({viewport,check:label});
      }
      const indexResponse = await context.request.get(`${base}/index.json`);
      const personIndex = await indexResponse.json();
      // Use a source-derived person and an exact term from a recorded speech,
      // exercising actor+issue search without hand-coded policy inference.
      let issuePerson = null;
      for (const person of personIndex.people.filter(p => p.id.startsWith('mp-')).slice(0,40)) {
        const response = await context.request.get(`${base}/people/${person.id}.json`);
        const packet = await response.json();
        const action = packet.initiatives.find(a => a.kind === 'SPEECH' && a.source_text);
        if (action) { issuePerson = {packet, action}; break; }
      }
      assert(issuePerson, 'a sourced parliamentary speech is searchable by person');
      await page.goto(`${base}/#person=${issuePerson.packet.id}`);
      await page.locator('#person-issue').waitFor();
      const term = issuePerson.action.source_text.match(/[A-Za-zÅÄÖåäö]{8,}/u)[0];
      await page.locator('#person-issue').fill(term);
      assert(await page.locator('#issue-results a[href^="objects/"]').count());
      await fits('actor+issue finds exact speech source');
      const words = issuePerson.action.source_text.match(/[A-Za-zÅÄÖåäö]{8,}/gu);
      await page.locator('#person-issue').fill(`${words[0]}  ${words[words.length-1]}`);
      assert(await page.locator(`#issue-results a[href="objects/${issuePerson.action.id}.html"]`).count());
      await fits('actor+issue matches separate words across the source');
      await page.locator('#person-issue').fill('zzzz-no-source-match-zzzz');
      assert((await page.locator('#issue-results').innerText()).includes('ei osoita kannan tai toiminnan puuttumista'));
      await page.goto(base);
      const indexedPerson = personIndex.people.find(p => p.elected && p.search_text);
      assert(indexedPerson, 'global source search has an elected person');
      const sourceQuery = indexedPerson.search_text.split('\n').find(line => line.trim());
      await page.locator('#q').fill(sourceQuery);
      await page.locator(`#hits button[data-id="${indexedPerson.id}"]`).waitFor();
      await fits('global search retains source text after exact duplicate removal');
      const slowPerson = personIndex.people[0];
      const fastPerson = personIndex.people.find(person => person.id !== slowPerson.id);
      let releaseSlow, markSlowStarted;
      const slowGate = new Promise(resolve => {releaseSlow=resolve;});
      const slowStarted = new Promise(resolve => {markSlowStarted=resolve;});
      const slowPattern = `**/people/${slowPerson.id}.json`;
      await page.route(slowPattern, async route => {
        const response = await route.fetch();
        markSlowStarted();
        await slowGate;
        await route.fulfill({response});
      });
      await page.evaluate(id => {location.hash=`person=${encodeURIComponent(id)}`;},slowPerson.id);
      await slowStarted;
      await page.evaluate(id => {location.hash=`person=${encodeURIComponent(id)}`;},fastPerson.id);
      await page.waitForFunction(name => document.querySelector('#person-name').textContent===name,fastPerson.name);
      const slowFinished = page.waitForEvent('requestfinished', {
        predicate:request=>new URL(request.url()).pathname.endsWith(`/people/${slowPerson.id}.json`),
      });
      releaseSlow();
      await slowFinished;
      await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
      assert.equal(await page.locator('#person-name').innerText(),fastPerson.name);
      await page.unroute(slowPattern);
      await fits('late profile response cannot replace the newly selected person');
      await page.goto(base);
      await page.locator('a[href="cases/index.html"]').click();
      const hrefs = await page.locator('li a').evaluateAll(links => links.map(link => link.getAttribute('href')));
      assert(hrefs.length >= 4, 'reviewed decision inquiries are available');
      for (const href of hrefs) {
        await page.goto(`${base}/cases/${href}`);
        assert((await page.locator('body').innerText()).includes('Mitä lähteet osoittavat?'));
        const response = await context.request.get(`${base}/cases/${href.replace(/\.html$/,'.json')}`);
        assert(response.ok());
        const packet = await response.json();
        const sources = new Map(packet.sources.map(source => [source.source_id, source]));
        for (const ref of packet.evidence) {
          const source = sources.get(ref.source_id);
          assert.equal(sha(source.text), ref.text_sha256);
          assert.equal(sha(source.raw_text), ref.raw_sha256);
          assert.equal(Array.from(source.text).slice(ref.start,ref.end).join(''), ref.quote);
          assert(await page.locator(`[id="${ref.evidence_id}"]`).count());
        }
        assert(packet.claims.every(claim => claim.state === 'SOURCE_REVIEWED'));
        assert(packet.claims.every(claim => claim.review.method === 'AI_SOURCE_READING'));
        await fits(packet.case_id);
        if (packet.case_id === 'rai-2020-constitutional-repair') {
          assert.equal(packet.legal_inquiry.latest_start_date,'2023-04-01');
          assert.equal(packet.legal_comparison_artifacts[0].receipt.operative.verified,false);
          const text = await page.locator('body').innerText();
          assert(text.includes('2024-12-05') && text.includes('2020-12-31'));
          assert(text.includes('Takaraja ei osoita'));
          fs.mkdirSync(path.dirname(output), {recursive:true});
          await page.screenshot({path:path.join(path.dirname(output), `decision-${viewport.width}.png`), fullPage:true});
        }
      }
      await page.goto(base);
      await page.locator('a[href="episodes/index.html"]').click();
      await page.locator('#episode-search').fill('KK 1/2023 vp');
      assert.equal(await page.locator('#episode-results article').count(),1);
      await page.locator('#episode-results article a').click();
      const text = await page.locator('body').innerText();
      assert(text.includes('Jussi Saramo'));
      assert(text.includes('2023-05-09') && text.includes('2023-05-30'));
      assert(text.includes('Vastaus on rekisterissä ilman vastaustekstiä'));
      await fits('question-registry metadata and answer-content boundary');
      const objectLink = page.locator('a[href^="../objects/"]').first();
      await objectLink.click();
      await fits('question source object');
      await page.goto(`${base}/groups/index.html`);
      const groupLinks = await page.locator('li a').evaluateAll(links => links.map(link => link.getAttribute('href')));
      assert(groupLinks.length, 'a complete real ballot slice supplies group contexts');
      let group = null;
      for (const href of groupLinks) {
        const response = await context.request.get(`${base}/groups/${href.replace(/\.html$/, '.json')}`);
        const packet = await response.json();
        const comparison = packet.comparisons.find(c => c.status === 'COMPARABLE');
        if (comparison) {group = {href,packet,comparison}; break;}
      }
      assert(group, 'at least one target has enough same-event substantive peers');
      await page.goto(`${base}/groups/${group.href}`);
      const groupText = await page.locator('body').innerText();
      assert(groupText.includes('henkilön oma ääni'));
      assert(groupText.includes('Kirjattu POISSA') && groupText.includes('Kirjattu TYHJA'));
      await fits('event-time group context and separate absence denominators');
      const sourceSlug = group.comparison.source_ref.replace(/[^A-Za-z0-9_.-]/g,'_');
      const sourceResponse = await context.request.get(`${base}/groups/sources/${sourceSlug}.json`);
      const source = await sourceResponse.json();
      const peers = source.rows.filter(row => row.person_id !== group.packet.person_id &&
        row.group_code === group.comparison.target_group_code && ['JAA','EI'].includes(row.response));
      assert.equal(peers.filter(row => row.response === 'JAA').length, group.comparison.peer_jaa);
      assert.equal(peers.filter(row => row.response === 'EI').length, group.comparison.peer_ei);
      assert(peers.length >= group.packet.minimum_peer_count);
      for (const bucket of ['JAA','EI','TYHJA','POISSA']) {
        assert.equal(source.rows.filter(row => row.response === bucket).length,source.published_totals[bucket]);
      }
      assert.equal(source.rows.length,source.published_totals.TOTAL);
      await page.goto(`${base}/groups/sources/${sourceSlug}.html`);
      assert.equal(await page.locator('tbody tr').count(), source.rows.length);
      await fits('complete ballot source and independently recomputed self-excluded peer counts');
      await page.goto(base);
      if (await page.locator('a[href="research/index.html"]').count()) {
        assert(researchSources.size, 'research validation requires the retained source fixtures');
        await page.locator('a[href="research/index.html"]').click();
        const researchLinks = await page.locator('li a').evaluateAll(links => links.map(link => link.getAttribute('href')));
        assert(researchLinks.length, 'completed model comparisons reach the browser');
        for (const href of researchLinks) {
          const packetResponse = await context.request.get(`${base}/research/${href.replace(/\.html$/, '.json')}`);
          assert(packetResponse.ok());
          const content = await packetResponse.body();
          assert.equal(sha(content).slice(0,24), href.replace(/\.html$/, ''));
          const packet = JSON.parse(content.toString('utf8'));
          assert.equal(packet.research_status,'PROPOSED_RESEARCH_NOT_ADMITTED');
          assert.equal(packet.not_model_admission,true);
          assert.equal(packet.case_count,packet.episodes.length);
          await page.goto(`${base}/research/${href}`);
          assert.equal(await page.locator('article.episode').count(),packet.case_count);
          assert.equal(await page.locator('meta[name="viewport"]').count(),1);
          const ids = await page.locator('[id]').evaluateAll(nodes => nodes.map(node => node.id));
          assert.equal(ids.length,new Set(ids).size,'source links have unique DOM targets');
          for (const episode of packet.episodes) {
            assert(episode.paired_input.same_input_sha256 && episode.paired_input.same_source_payload_sha256,
              'baseline and structured answers use the same original-source input');
            for (const mode of Object.values(episode.modes)) {
              assert.equal(mode.admission_state,'PROPOSED / NOT_ADMITTED');
              assert.equal(mode.receipt_count,1,'one final answer per mode and episode');
              for (const claim of mode.claims) {
                assert.equal(claim.state,'PROPOSED');
                for (const anchor of claim.evidence) {
                  const retained = researchSources.get(anchor.source_id);
                  assert(retained,'source anchor resolves against retained captured records');
                  assert.equal(sha(retained.text),anchor.source_text_sha256);
                  if (anchor.anchor_state === 'EXACT_COMPLETE_SOURCE') {
                    const chars = Array.from(retained.text);
                    assert.equal(chars.slice(anchor.char_start,anchor.char_end).join(''),anchor.quote);
                    assert.equal(chars.slice(anchor.context_start,anchor.context_end).join(''),anchor.context_excerpt);
                  } else {
                    assert.equal(anchor.char_start,null,'ambiguous source occurrence cannot receive a fabricated offset');
                  }
                  assert(await page.locator(`[data-source-anchor-id="${anchor.source_anchor_id}"]`).count());
                }
              }
            }
          }
          await fits(`research cohort ${packet.case_count}: paired inputs, exact sources and proposal status`);
        }
      }
      await context.close();
    }
  } finally { await browser.close(); }
  assert.deepEqual(errors,[]);
  assert.deepEqual(external,[]);
  fs.mkdirSync(path.dirname(output),{recursive:true});
  fs.writeFileSync(output,JSON.stringify({checks,errors,external_requests:external},null,2));
  process.stdout.write(JSON.stringify({checks:checks.length,errors,external_requests:external}));
})().catch(error => {process.stderr.write(error.stack);process.exitCode=1;});

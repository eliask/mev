/* Verify personal choices and reachable exact sources, rather than a card feed.
 * Accept either a standalone matter collection or the integrated preview.
 */
const fs = require('fs');
const path = require('path');
const assert = require('assert/strict');
const { chromium } = require('playwright-core');
const [base, root, output] = process.argv.slice(2);
if (!base || !root || !output) throw new Error('usage: node script URL ARTIFACT_DIR REPORT_DIR');
const manifest = JSON.parse(fs.readFileSync(path.join(root, 'manifest.json'), 'utf8'));
const entries = manifest.entries || manifest.matters.map(entry => ({...entry, packet_path:entry.json_path}));
fs.mkdirSync(output, {recursive:true});
const key = actor => actor.actor_id ? 'actor:' + actor.actor_id : actor.person_id ? 'person:' + actor.person_id : 'name-only:' + (actor.name || 'tuntematon');
const slug = value => value.replace(/[^\p{L}\p{N}_.-]+/gu, '_').replace(/^[._]+|[._]+$/g, '') || 'tuntematon';
const recordSlug = (index, action) => String(index).padStart(4, '0') + '_' + slug(String(action.object_id || action.intervention_id || index));
const approval = row => {
  const parts = (row.title || '').split('/').map(part => part.trim().toLowerCase());
  return parts.length === 2 && new Set(parts).size === 2 && parts.includes('hyväksyminen') && parts.includes('hylkääminen');
};
const exception = row => row.own_group_peer_comparison?.status === 'COMPARABLE' && row.own_group_peer_comparison?.matches_peer_majority === false;

(async () => {
  const browser = await chromium.launch({headless:true, executablePath:process.env.PAA_CHROMIUM});
  const errors = [], requests = [], results = [];
  const origin = new URL(base).origin;
  const localOnly = async context => context.route('**/*', route => {
    if (new URL(route.request().url()).origin === origin) return route.continue();
    requests.push(route.request().url()); return route.abort();
  });
  const overflow = async page => assert(await page.evaluate(()=>document.documentElement.scrollWidth <= innerWidth + 1), `horizontal overflow: ${page.url()} at ${page.viewportSize().width}px`);
  try {
    for (const width of [1280, 390, 320]) {
      const context = await browser.newContext({viewport:{width,height:844}});
      await localOnly(context);
      const page = await context.newPage();
      page.on('pageerror', error => errors.push(error.message));
      for (const entry of entries) {
        const packet = JSON.parse(fs.readFileSync(path.join(root, entry.packet_path), 'utf8'));
        const preferred = packet.interventions.find(a=>a.actor.name === entry.test_actor) || packet.interventions[0];
        const actorKey = key(preferred.actor);
        const hubStem = path.posix.basename(entry.path, '.html');
        const personPath = path.posix.join(path.posix.dirname(entry.path), hubStem, 'people', slug(actorKey) + '.html');
        const selected = packet.interventions.map((action,index)=>({action,index})).filter(row=>key(row.action.actor) === actorKey);
        await page.goto(new URL(entry.path, base).href);
        assert.equal(await page.locator('.intervention').count(), 0, 'hub must not preload detailed records');
        assert.equal(await page.locator('#actor-filter option').count(), new Set(packet.interventions.map(a=>key(a.actor))).size + 1);
        assert.equal(await page.locator('#person-panel').isVisible(), false);
        await overflow(page);
        await page.selectOption('#actor-filter', actorKey);
        await page.waitForFunction(() => !document.getElementById('person-panel').hidden && document.getElementById('record-count').textContent === '');
        assert(page.url().includes('#henkilo='));
        assert.equal(await page.locator('#person-panel article.intervention').count(), 0, 'person view is not an activity card feed');
        // Every record remains linked from primary rows or a closed source index.
        for (const {action,index} of selected) {
          const filename = recordSlug(index, action) + '.html';
          assert(await page.locator(`#person-panel a[href$="/${filename}"]`).count() >= 1, `missing source link ${filename}`);
        }
        await overflow(page);
        // Remove selection through URL navigation; stale fetched panels must clear.
        await page.evaluate(()=>{ location.hash = ''; });
        await page.waitForFunction(()=>document.getElementById('person-panel').hidden);
        assert.equal(await page.locator('#actor-filter').inputValue(), 'choose');
        await page.goto(new URL(personPath, base).href);
        const votes = selected.filter(row=>row.action.kind === 'VOTE');
        let primary = votes.filter(row=>approval(row.action) || exception(row.action));
        if (!primary.length && votes.length <= 3) primary = votes;
        assert.equal(await page.locator('.personal-choices tbody tr').count(), primary.length);
        for (const {action,index} of primary) {
          const row = page.locator(`#record-${index}`);
          assert.equal(await row.locator('td').nth(1).locator('a').textContent(), action.title);
          assert.equal(await row.locator('td').nth(2).locator('strong').textContent(), action.raw_response || '—');
          if (exception(action)) assert((await row.innerText()).includes('ryhmä ' + action.own_group_peer_comparison.peer_majority));
        }
        assert.equal(await page.locator('details.speech-index[open], details.other-ballots[open]').count(), 0);
        await overflow(page);
        // A source page must expose the exact body, with source text escaped.
        for (const kind of [...new Set(selected.map(row=>row.action.kind))]) {
          const {action,index} = selected.find(row=>row.action.kind === kind);
          const recordPath = path.posix.join(path.posix.dirname(personPath), '..', 'records', recordSlug(index, action) + '.html');
          await page.goto(new URL(recordPath, base).href);
          const record = page.locator('article.intervention');
          assert.equal(await record.getAttribute('data-kind'), kind);
          if (kind === 'VOTE') {
            assert((await record.innerText()).includes(action.raw_response));
            assert((await record.innerText()).includes(action.title));
          }
          if (action.source_text) {
            const details = record.locator('details.source-text');
            if (await details.count()) {
              await details.locator('summary').focus();
              await page.keyboard.press('Enter');
            }
            assert.equal(await record.locator('blockquote').first().textContent(), action.source_text);
          }
          const original = record.locator('a').filter({hasText:'Avaa alkuperäinen lähde'}).first();
          if (await original.count()) assert.equal(await original.getAttribute('href'), action.source.url);
          await overflow(page);
        }
        await page.goto(new URL(entry.path, base).href);
        assert.equal(await page.locator('a[download]').getAttribute('href'), path.posix.basename(entry.packet_path));
        const response = await context.request.get(new URL(entry.packet_path, base).href);
        assert(response.ok());
        assert.deepEqual(await response.json(), packet);
        await page.screenshot({path:path.join(output,`${hubStem}-${width}.png`),fullPage:false});
        results.push({matter_id:packet.query.matter_id,width,selected_actor:preferred.actor.name,selected_records:selected.length,visible_ballots:primary.length,person_path:personPath});
      }
      await context.close();
    }
    const nojs = await browser.newContext({javaScriptEnabled:false,viewport:{width:390,height:844}});
    await localOnly(nojs);
    for (const entry of entries) {
      const page = await nojs.newPage();
      await page.goto(new URL(entry.path,base).href);
      assert.equal(await page.locator('.intervention').count(), 0);
      await page.locator('#henkilot > summary').focus();
      await page.keyboard.press('Enter');
      const person = page.locator('#henkilot a[href*="/people/"]').first();
      assert(await person.isVisible());
      await person.click();
      assert(await page.locator('#person-records').count());
      assert.equal(await page.locator('article.intervention').count(), 0);
      await overflow(page);
      await page.close();
    }
    await nojs.close();
    assert.deepEqual(errors,[]); assert.deepEqual(requests,[]);
    const report={checks:results.length,no_js_pages:entries.length,errors,external_requests:requests,results};
    fs.writeFileSync(path.join(output,'browser.json'),JSON.stringify(report,null,2)+'\n');
    console.log(JSON.stringify({checks:results.length,no_js_pages:entries.length,errors,external_requests:requests}));
  } finally { await browser.close(); }
})().catch(error=>{console.error(error);process.exitCode=1;});

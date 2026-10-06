/* Audit the extracted selected release at a supplied root or hosting prefix. */
const { chromium } = require('playwright-core');
const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');
const base = process.env.PAA_BROWSER_URL;
const output = process.env.PAA_BROWSER_REPORT;
assert(base && output, 'PAA_BROWSER_URL and PAA_BROWSER_REPORT are required');
const sha = text => crypto.createHash('sha256').update(text, 'utf8').digest('hex');

(async () => {
  const browser = await chromium.launch({executablePath: process.env.PAA_CHROMIUM_PATH, args: ['--no-sandbox']});
  const checks = [], errors = [], external = [], measurements = [];
  try {
    for (const viewport of [{width:1280,height:900},{width:390,height:844},{width:320,height:740}]) {
      const context = await browser.newContext({viewport, permissions:['clipboard-read','clipboard-write']});
      await context.route('**/*', async route => {
        if (new URL(route.request().url()).origin === new URL(base).origin) await route.continue();
        else {external.push(route.request().url()); await route.abort();}
      });
      const page = await context.newPage();
      page.on('pageerror', e => errors.push(e.message));
      const response = await context.request.get(`${base}/manifest.json`);
      assert(response.ok());
      const manifest = await response.json();
      assert.equal(manifest.deployment_status,'NOT_DEPLOYED');
      assert(manifest.cases.length >= 3 && manifest.cases.length <= 5);
      await page.goto(base);
      assert.equal(await page.locator('article h2 a').count(), manifest.cases.length);
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
      const firstCard = await page.locator('.case-card').first().boundingBox();
      assert(firstCard.y < 350, 'first investigation should precede the methodology');
      measurements.push({viewport, homepage_first_card_y: firstCard.y});
      for (const row of manifest.cases) {
        const url = `${base}/${row.path}`;
        await page.goto(url, {waitUntil:'networkidle'});
        await page.reload({waitUntil:'networkidle'});
        const body = await page.locator('body').innerText();
        assert(body.includes('Tekoälyavusteinen lähdeluku') && body.includes(row.review_date));
        const method = await page.locator('.method').textContent();
        assert(method.includes('ei riippumatonta ihmistarkistusta') && method.includes(row.source_period));
        const packetResponse = await context.request.get(url.replace(/\.html$/,'.json'));
        assert(packetResponse.ok());
        const packet = await packetResponse.json();
        assert.equal(packet.publication.revision,manifest.revision);
        assert.equal(await page.locator('meta[name="description"]').getAttribute('content'),packet.publication.takeaway);
        assert.equal(await page.locator('meta[property="og:description"]').getAttribute('content'),packet.publication.takeaway);
        assert.equal(await page.locator('meta[name="twitter:card"]').getAttribute('content'),'summary');
        if (manifest.base_url) assert.equal(await page.locator('link[rel="canonical"]').getAttribute('href'),packet.publication.canonical_url);
        else assert.equal(await page.locator('link[rel="canonical"]').count(),0);
        const correction = new URL(await page.locator('.actions a').last().getAttribute('href'));
        assert.equal(correction.hostname,'github.com');
        assert(correction.searchParams.get('body').includes(manifest.revision));
        for (const link of await page.locator('.actions a[download]').evaluateAll(ns=>ns.map(n=>n.getAttribute('href')))) {
          const download = await context.request.get(new URL(link,url).href);
          assert(download.ok() && (await download.body()).length > 100);
        }
        const sources = new Map(packet.sources.map(s=>[s.source_id,s]));
        for (const source of sources.values()) {
          assert.equal(sha(source.text),source.text_sha256);
          assert.equal(sha(source.raw_text),source.raw_sha256);
        }
        for (const ref of packet.evidence) {
          const source = sources.get(ref.source_id);
          assert(source);
          assert.equal(Array.from(source.text).slice(ref.start,ref.end).join(''),ref.quote);
          assert.equal(sha(source.text),ref.text_sha256);
          const researchResponse = await context.request.get(`${base}/${row.research_path}`);
          assert(researchResponse.ok() && (await researchResponse.text()).includes(ref.evidence_id));
        }
        assert(packet.claims.every(c=>c.state==='SOURCE_REVIEWED'));
        assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth),false);
        const takeaway = await page.locator('.takeaway').first().boundingBox();
        assert(takeaway.y < viewport.height, 'answer begins on the first screen');
        const summary = page.locator('.inline-evidence summary').first();
        const evidenceBox = await summary.boundingBox();
        const closedHeight = await page.evaluate(()=>document.documentElement.scrollHeight);
        assert(evidenceBox.y < 1200, 'decisive evidence must be next to the answer');
        assert(closedHeight < 3500, 'reading view must not expose the full research dump');
        if (packet.publication.comparison.length) {
          const figures = await page.locator('.figures').innerText();
          assert.equal((figures.match(/91 %/g)||[]).length,2);
          assert(figures.includes('Määräajassa') && figures.includes('Havainnoista YTHS:ltä'));
          assert.equal((figures.match(/310 \/ 340/g)||[]).length,2);
          assert((await page.locator('.population').innerText()).includes('ei ole kaikkien'));
          const pair = await page.locator('.figures').boundingBox();
          assert(pair.y + pair.height < viewport.height, 'both measures fit the first screen');
        }
        measurements.push({viewport,case_id:row.case_id,closed_page_height:closedHeight,first_evidence_y:evidenceBox.y});
        await page.locator('footer details summary').click();
        await page.locator('#copy-url').click();
        await page.waitForFunction(()=>document.getElementById('share-status').textContent === 'Linkki kopioitu.');
        assert.equal(await page.locator('#share-status').innerText(),'Linkki kopioitu.');
        assert.equal(await page.evaluate(()=>navigator.clipboard.readText()),url);
        await page.evaluate(()=>{Object.defineProperty(navigator,'clipboard',{value:{writeText:()=>Promise.reject(new Error('unavailable'))},configurable:true});});
        await page.locator('#copy-url').focus();
        await page.keyboard.press('Enter');
        await page.waitForFunction(()=>document.getElementById('share-status').textContent.includes('Kopioi valittu osoite'));
        assert((await page.locator('#share-status').innerText()).includes('Kopioi valittu osoite'));
        assert.equal(await page.locator('#case-url').inputValue(),url);
        assert((await page.locator('body').innerText()).includes(packet.publication.main_limit));
        await summary.focus();
        await page.keyboard.press('Enter');
        assert(await summary.evaluate(n=>n.parentElement.open));
        const firstRef = packet.publication.sections[0].evidence_ids[0];
        assert(await page.evaluate(id=>!!document.getElementById(id),firstRef));
        await page.goto(url+'#'+encodeURIComponent(firstRef));
        assert(await page.evaluate(id=>document.getElementById(id).closest('details').open,firstRef));
        await page.locator('.method summary').click();
        assert((await page.locator('body').innerText()).includes(row.source_period));
        await page.goto(`${base}/${row.research_path}`);
        assert.equal(await page.locator('nav a').first().getAttribute('href'),row.path.split('/').pop());
        for (const ref of packet.evidence) assert(await page.evaluate(id=>!!document.getElementById(id),ref.evidence_id));
        await page.goto(url);
        if (viewport.width === 390) {
          fs.mkdirSync(path.dirname(output),{recursive:true});
          await page.goto(url);
          await page.screenshot({path:path.join(path.dirname(output),`${row.case_id}-mobile-firstscreen.png`)});
          await page.screenshot({path:path.join(path.dirname(output),`${row.case_id}-mobile.png`),fullPage:true});
        }
        checks.push({viewport,case_id:row.case_id,checks:['direct URL and refresh','answer and scope','metadata',
          'JSON/citation downloads','inline evidence and full research record','copy/fallback','keyboard disclosure','mobile hierarchy and fit']});
        await page.evaluate(()=>{delete navigator.clipboard;});
      }
      await context.close();
      const noScript = await browser.newContext({viewport,javaScriptEnabled:false});
      const readable = await noScript.newPage();
      await readable.goto(`${base}/${manifest.cases[0].path}`);
      assert((await readable.locator('body').innerText()).includes('91 %'));
      await readable.locator('.inline-evidence summary').first().click();
      assert((await readable.locator('.inline-evidence').first().innerText()).includes('YTHS'));
      await readable.locator('footer details summary').click();
      assert.equal(await readable.locator('#copy-url').isDisabled(),true);
      assert((await readable.locator('body').innerText()).includes('selaimen osoiterivin'));
      checks.push({viewport,check:'answer and source fallback remain readable without JavaScript'});
      await noScript.close();
    }
  } finally {await browser.close();}
  assert.deepEqual(errors,[]);
  assert.deepEqual(external,[]);
  fs.mkdirSync(path.dirname(output),{recursive:true});
  fs.writeFileSync(output,JSON.stringify({checks,measurements,errors,external_requests:external},null,2));
  process.stdout.write(JSON.stringify({checks:checks.length,errors,external_requests:external})+'\n');
})().catch(e=>{process.stderr.write(e.stack+'\n');process.exitCode=1;});

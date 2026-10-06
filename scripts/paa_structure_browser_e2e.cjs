/* Focused browser audit for the proposed structure-comparison cohort. */
const { chromium } = require('playwright-core');
const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');

const base = process.env.PAA_BROWSER_URL;
const output = process.env.PAA_BROWSER_REPORT;
const expectedPackets = Number(process.env.PAA_EXPECTED_PACKETS);
assert(Number.isInteger(expectedPackets) && expectedPackets > 0,
  'PAA_EXPECTED_PACKETS must declare a positive packet count');
const structurePrefix = process.env.PAA_STRUCTURE_PREFIX === undefined
  ? '/structure' : process.env.PAA_STRUCTURE_PREFIX;
assert(base && output, 'PAA_BROWSER_URL and PAA_BROWSER_REPORT are required');
const origin = new URL(base).origin;
const sha = value => crypto.createHash('sha256').update(value, 'utf8').digest('hex');
const deepText = value => {
  if (typeof value === 'string') return value;
  if (Array.isArray(value)) return value.map(deepText).join('\n');
  if (value && typeof value === 'object') return Object.values(value).map(deepText).join('\n');
  return '';
};

async function main() {
  const browser = await chromium.launch({
    executablePath: process.env.PAA_CHROMIUM_PATH,
    args: ['--no-sandbox'],
  });
  const checks = [];
  const errors = [];
  const external = [];
  const packetStats = { packets: 0, candidateVisits: 0, bindingWarnings: 0, evidence: 0 };
  try {
    const links = [];
    for (const viewport of [{ width: 1280, height: 900 }, { width: 390, height: 844 }]) {
      const context = await browser.newContext({ viewport });
      await context.route('**/*', async route => {
        const url = new URL(route.request().url());
        if (url.origin === origin) return route.continue();
        external.push(route.request().url());
        return route.abort();
      });
      const page = await context.newPage();
      page.on('pageerror', error => errors.push(`${viewport.width}: ${error.message}`));

      await page.goto(`${base}${structurePrefix}/comparison.html`, { waitUntil: 'networkidle' });
      const comparisonText = await page.locator('body').innerText();
      assert(comparisonText.includes('Uusien asioiden lähderakennevertailu'));
      assert(comparisonText.includes('ei tarkistettuja päätelmiä'));
      const comparisonLinks = await page.locator('a[href$=".html"]').evaluateAll(
        nodes => nodes.map(node => node.getAttribute('href')),
      );
      assert.equal(comparisonLinks.length, expectedPackets, 'comparison exposes the declared candidate pages');
      assert.equal(new Set(comparisonLinks).size, expectedPackets, 'candidate page links are unique');
      if (!links.length) links.push(...comparisonLinks);
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true,
        `comparison has no horizontal overflow at ${viewport.width}`);
      checks.push({ viewport, check: `comparison labels and ${expectedPackets} unique candidate links` });

      for (const href of comparisonLinks) {
        const htmlUrl = `${base}${structurePrefix}/${href}`;
        const jsonUrl = htmlUrl.replace(/\.html$/, '.json');
        const jsonResponse = await context.request.get(jsonUrl);
        assert.equal(jsonResponse.ok(), true, `candidate JSON available: ${href}`);
        const packet = await jsonResponse.json();
        const htmlResponse = await context.request.get(htmlUrl);
        assert.equal(htmlResponse.ok(), true, `candidate HTML available: ${href}`);
        assert.equal(packet.admission, 'PROPOSED_NOT_ADMITTED', `${href} stays candidate-only`);
        assert(!deepText(packet).includes('SOURCE_REVIEWED'), `${href} has no SOURCE_REVIEWED`);
        assert(packet.claims.every(claim => claim.state === 'CANDIDATE'), `${href} claims remain candidates`);
        assert(packet.sources.length > 0, `${href} retains source versions`);
        const sources = new Map(packet.sources.map(source => [source.source_id, source]));
        const evidence = new Map(packet.evidence.map(item => [item.evidence_id, item]));
        assert.equal(evidence.size, packet.evidence.length, `${href} evidence IDs are unique`);
        for (const source of packet.sources) {
          assert.equal(sha(source.text), source.text_sha256, `${href} text hash binds source`);
          assert.equal(sha(source.raw_text), source.raw_sha256, `${href} raw hash binds source`);
        }
        for (const claim of packet.claims) {
          for (const evidenceId of claim.evidence_ids || []) {
            assert(evidence.has(evidenceId), `${href} claim evidence resolves`);
          }
        }
        for (const item of packet.evidence) {
          const source = sources.get(item.source_id);
          assert(source, `${href} evidence source resolves`);
          assert.equal(item.text_sha256, source.text_sha256, `${href} evidence text version binds`);
          assert.equal(item.raw_sha256, source.raw_sha256, `${href} evidence raw version binds`);
          assert(Number.isInteger(item.start) && Number.isInteger(item.end));
          const chars = Array.from(source.text);
          assert(item.start >= 0 && item.end > item.start && item.end <= chars.length);
          assert.equal(chars.slice(item.start, item.end).join(''), item.quote,
            `${href} evidence quote matches exact source offsets`);
          packetStats.evidence += 1;
        }

        await page.goto(htmlUrl, { waitUntil: 'networkidle' });
        await page.locator('details').evaluateAll(nodes => nodes.forEach(node => { node.open = true; }));
        const body = await page.locator('body').innerText();
        assert(body.includes('Hakutulos tai avoin tulkinta; ei varmennettu päätelmä.'),
          `${href} does not expose candidate status: ${body.slice(0, 400)}`);
        assert(!body.includes('SOURCE_REVIEWED'), `${href} page has no SOURCE_REVIEWED label`);
        const evidenceIds = await page.locator('[id]').evaluateAll(nodes => nodes.map(node => node.id));
        for (const item of packet.evidence) {
          assert(evidenceIds.includes(item.evidence_id), `${href} renders ${item.evidence_id}`);
        }
        if (packet.binding_errors?.length) {
          packetStats.bindingWarnings += 1;
          const expanded = await page.locator('body').innerText();
          assert(expanded.includes('Mallin lähdeviitteitä korjattiin') ||
            expanded.includes('Virheellisiä mallipaikkoja ei hyväksytty.'),
            `${href} exposes rejected-model-offset warning after opening details`);
          for (const warning of packet.binding_errors) assert(expanded.includes(warning), `${href} exposes ${warning}`);
        }
        assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true,
          `${href} has no horizontal overflow at ${viewport.width}`);
        packetStats.packets += 1;
        if (packet.admission === 'PROPOSED_NOT_ADMITTED') packetStats.candidateVisits += 1;
        checks.push({ viewport, check: `candidate ${href} page/json/anchors/status` });
      }
      await context.close();
    }
    assert.equal(links.length, expectedPackets);
  } finally {
    await browser.close();
  }
  assert.deepEqual(errors, []);
  assert.deepEqual(external, []);
  fs.mkdirSync(path.dirname(output), { recursive: true });
  fs.writeFileSync(output, JSON.stringify({ checks, errors, external_requests: external, packetStats }, null, 2));
  process.stdout.write(JSON.stringify({ checks: checks.length, errors, external_requests: external, packetStats }));
}

main().catch(error => { process.stderr.write(error.stack); process.exitCode = 1; });

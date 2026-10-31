#!/usr/bin/env node
/* Screenshots de páginas inteiras: node tools/shot.js '#/' home.png [--w 1440] [--h 900] [--full] [--copilot] [--light] */
'use strict';
const path = require('path');
const PUP = process.env.PUPPETEER_CORE || (() => { try { return require.resolve('puppeteer-core', { paths: [require('path').resolve(__dirname, '..')] }); } catch (e) { return 'puppeteer-core'; } })();
const CHROME = process.env.CHROME || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const a = process.argv.slice(2);
const hash = a[0] || '#/', out = a[1] || 'shot.png';
const num = (k, d) => { const i = a.indexOf(k); return i >= 0 ? +a[i + 1] : d; };
(async () => {
  const b = await require(PUP).launch({ executablePath: CHROME, headless: true, args: ['--allow-file-access-from-files'] });
  const p = await b.newPage();
  await p.setViewport({ width: num('--w', 1440), height: num('--h', 900), deviceScaleFactor: 1, isMobile: num('--w', 1440) < 700 });
  const errs = [];
  p.on('pageerror', (e) => errs.push(String(e)));
  p.on('console', (m) => { if (m.type() === 'error') errs.push(m.text()); });
  const li = a.indexOf('--lang'); const lang = li >= 0 ? a[li + 1] : 'pt';
  await p.evaluateOnNewDocument((l) => { window.LA_LANG_FORCE = l; }, lang);
  await p.goto('file://' + path.resolve(__dirname, '..', 'index.html') + hash, { waitUntil: 'load', timeout: 60000 });
  if (a.includes('--light')) await p.evaluate(() => LA.theme.set('light'));
  await new Promise((r) => setTimeout(r, 1800));
  if (a.includes('--copilot')) { await p.evaluate(() => LA.copilot.open()); await new Promise((r) => setTimeout(r, 800)); }
  const sy = num('--scroll', 0); if (sy) { await p.evaluate((y) => window.scrollTo(0, y), sy); await new Promise((r) => setTimeout(r, 900)); }
  await p.screenshot({ path: out, fullPage: a.includes('--full') });
  if (a.includes('--light')) await p.evaluate(() => LA.theme.set('dark'));
  console.log('ok', out, errs.length ? errs : '');
  await b.close();
})().catch((e) => { console.error(e); process.exit(1); });

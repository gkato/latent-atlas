#!/usr/bin/env node
/* ==========================================================================
   LATENT ATLAS — tools/qa.js  (QA headless com Chrome + puppeteer-core)
   Para cada capítulo: renderiza, rola até cada figura, verifica se o canvas
   desenhou algo, mexe em todos os controles, coleta erros de console/página,
   termos de glossário ausentes, links quebrados, erros de KaTeX e overflow.
   Opcionalmente salva screenshots de cada figura para inspeção visual.

   Uso:
     node tools/qa.js                      # todos os capítulos
     node tools/qa.js attention moe        # só estes ids
     node tools/qa.js --shots attention    # + screenshots em $QA_OUT/<id>/
     node tools/qa.js --mobile attention   # viewport 390px
   Variáveis: PUPPETEER_CORE (caminho do módulo), CHROME (executável), QA_OUT
   ========================================================================== */
'use strict';
const path = require('path');
const fs = require('fs');

const ROOT = path.resolve(__dirname, '..');
const PUP = process.env.PUPPETEER_CORE || (() => { try { return require.resolve('puppeteer-core', { paths: [require('path').resolve(__dirname, '..')] }); } catch (e) { return 'puppeteer-core'; } })();
const CHROME = process.env.CHROME || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const OUT = process.env.QA_OUT || path.join(ROOT, 'qa-out');

const args = process.argv.slice(2);
const shots = args.includes('--shots');
const mobile = args.includes('--mobile');
const noInteract = args.includes('--no-interact');
const li = args.indexOf('--lang');
const LANG = li >= 0 ? args[li + 1] : 'pt';
const ids = args.filter((a, i) => !a.startsWith('--') && !(li >= 0 && i === li + 1));

(async () => {
  // build isolado por execução (vários agentes podem rodar o QA ao mesmo tempo)
  const TAG = 'qa-' + process.pid;
  const bundle = path.join(ROOT, 'dist', `.${TAG}.bundle.js`);
  const html = path.join(ROOT, `.${TAG}.html`);
  require('child_process').execSync(`node build.js --lang ${LANG} --out "${bundle}"`, { cwd: ROOT, stdio: 'inherit' });
  fs.writeFileSync(html, fs.readFileSync(path.join(ROOT, 'index.html'), 'utf8').replace('<script src="assets/js/i18n.js"></script>', `<script>window.LA_LANG_FORCE=${JSON.stringify(LANG)};window.LA_BUNDLE_OVERRIDE=${JSON.stringify('dist/.' + TAG + '.bundle.js')};</script><script src="assets/js/i18n.js"></script>`));
  const cleanup = () => { try { fs.unlinkSync(bundle); } catch (e) {} try { fs.unlinkSync(html); } catch (e) {} };
  process.on('exit', cleanup);
  process.on('SIGINT', () => { cleanup(); process.exit(130); });
  const puppeteer = require(PUP);
  const browser = await puppeteer.launch({ executablePath: CHROME, headless: true, args: ['--allow-file-access-from-files', '--no-first-run', '--no-default-browser-check'] });
  const page = await browser.newPage();
  await page.setViewport(mobile ? { width: 390, height: 844, deviceScaleFactor: 2, isMobile: true } : { width: 1440, height: 900, deviceScaleFactor: 1 });
  let errs = [];
  page.on('console', (m) => { if (m.type() === 'error' || m.type() === 'warn') errs.push(`[console.${m.type()}] ${m.text()}`.slice(0, 600)); });
  page.on('pageerror', (e) => errs.push(`[pageerror] ${String(e && e.stack || e).slice(0, 900)}`));
  page.on('requestfailed', (r) => { if (!/fonts\.g|jsdelivr|cdnjs/.test(r.url())) errs.push(`[requestfailed] ${r.url()}`); });

  const url = 'file://' + html;
  await page.goto(url + '#/', { waitUntil: 'load', timeout: 60000 });
  await new Promise((r) => setTimeout(r, 1200));
  const all = await page.evaluate(() => LA.chapters.map((c) => c.id));
  const loadErrors = await page.evaluate(() => LA.loadErrors);
  const homeErrs = errs.slice(); errs = [];
  const targets = ids.length ? ids : all;
  const report = { when: new Date().toISOString(), mobile, loadErrors, homeErrors: homeErrs, chapters: {} };
  // ids planejados (capítulos ainda não escritos não contam como link quebrado)
  let planned = [];
  try { planned = JSON.parse(fs.readFileSync(path.join(ROOT, 'docs', 'curriculum.json'), 'utf8')).map((c) => c.id); } catch (e) {}

  for (const id of targets) {
    errs = [];
    const r = { id, exists: all.includes(id), figures: [], issues: [] };
    if (!r.exists) { r.issues.push('capítulo não registrado (id inexistente ou erro de carga)'); report.chapters[id] = r; continue; }
    await page.evaluate((id) => { location.hash = '#/c/' + id; }, id);
    await new Promise((res) => setTimeout(res, 900));
    const meta = await page.evaluate((planned) => {
      const art = document.getElementById('la-article');
      const terms = Array.from(document.querySelectorAll('.la-term')).map((t) => t.dataset.term).filter((t) => !LA.glossary[t]);
      const links = Array.from(document.querySelectorAll('#la-main a[href^="#/c/"]')).map((a) => a.getAttribute('href').split('/')[2]).filter((x) => x && !LA.byId[x] && !planned.includes(x));
      const c = LA.byId[LA.route.id];
      const prereq = (c.prereqs || []).filter((p) => !LA.byId[p] && !planned.includes(p));
      const katexErr = Array.from(document.querySelectorAll('.katex-error')).map((e) => e.getAttribute('title') || e.textContent).slice(0, 10);
      // texto útil: sem MathML do KaTeX, sem figuras, sem código
      const clone = art.cloneNode(true);
      clone.querySelectorAll('.katex-mathml, figure, pre, script, style').forEach((n) => n.remove());
      const prose = clone.textContent || '';
      const rawTex = (prose.match(/\\\(|\\\[|\$\$/g) || []).length;
      const words = prose.split(/\s+/).filter(Boolean).length;
      return { fallback: !!c.__fallback, figs: document.querySelectorAll('#la-article figure.la-fig').length, missingTerms: Array.from(new Set(terms)), badLinks: Array.from(new Set(links)), badPrereqs: prereq, katexErr, rawTex, words, quiz: (c.quiz || []).length, summary: (c.summary || []).length, loaderr: Array.from(document.querySelectorAll('.la-loaderr')).map((e) => e.textContent) };
    }, planned);
    Object.assign(r, { words: meta.words, quiz: meta.quiz, summary: meta.summary, nFigs: meta.figs });
    if (meta.missingTerms.length) r.issues.push('termos de glossário ausentes: ' + meta.missingTerms.join(', '));
    if (meta.badLinks.length) r.issues.push('links para capítulos inexistentes: ' + meta.badLinks.join(', '));
    if (meta.badPrereqs.length) r.issues.push('pré-requisitos inexistentes: ' + meta.badPrereqs.join(', '));
    if (meta.katexErr.length) r.issues.push('erros KaTeX: ' + meta.katexErr.join(' | '));
    if (meta.rawTex) r.issues.push(`${meta.rawTex} delimitadores TeX não renderizados (KaTeX não carregou ou sintaxe inválida)`);
    if (meta.loaderr.length) r.issues.push('erros de montagem: ' + meta.loaderr.join(' | '));
    if (LANG !== 'pt' && meta.fallback) r.issues.push('sem tradução: exibindo a versão pt (falta chapters/NN-' + id + '.en.js)');

    for (let i = 0; i < meta.figs; i++) {
      const f = await page.evaluate(async (i) => {
        const fig = document.querySelectorAll('#la-article figure.la-fig')[i];
        fig.scrollIntoView({ block: 'center' });
        await new Promise((r) => setTimeout(r, 700));
        const cvs = Array.from(fig.querySelectorAll('canvas'));
        const stats = cvs.map((cv) => {
          try {
            const g = cv.getContext('2d');
            if (!g || !cv.width || !cv.height) return { w: cv.width, h: cv.height, ink: 0 };
            const d = g.getImageData(0, 0, cv.width, cv.height).data;
            let ink = 0; for (let k = 3; k < d.length; k += 16) if (d[k] > 8) ink++;
            return { w: cv.width, h: cv.height, ink: ink / (d.length / 16) };
          } catch (e) { return { err: String(e) }; }
        });
        const rect = fig.getBoundingClientRect();
        return { title: fig.dataset.title || '', canvases: stats, domNodes: fig.querySelectorAll('*').length, h: Math.round(rect.height), w: Math.round(rect.width), overflowX: fig.scrollWidth > fig.clientWidth + 2, err: Array.from(fig.querySelectorAll('.la-fig-error')).map((e) => e.textContent) };
      }, i);
      if (shots) {
        const dir = path.join(OUT, id + (LANG === 'en' ? '.en' : '') + (mobile ? '-m' : ''));
        fs.mkdirSync(dir, { recursive: true });
        await page.addStyleTag({ content: '.la-top,.la-readbar,.cp-fab{visibility:hidden!important}' }).catch(() => {});
        const el = (await page.$$('#la-article figure.la-fig'))[i];
        try { await el.screenshot({ path: path.join(dir, `fig-${i + 1}.png`) }); f.shot = path.join(dir, `fig-${i + 1}.png`); } catch (e) { f.shotErr = String(e); }
      }
      if (!noInteract) {
        const inter = await page.evaluate(async (i) => {
          const fig = document.querySelectorAll('#la-article figure.la-fig')[i];
          let n = 0;
          const wait = (ms) => new Promise((r) => setTimeout(r, ms));
          for (const inp of fig.querySelectorAll('input[type=range]')) { for (const v of [inp.min, inp.max, (+inp.min + +inp.max) / 2]) { inp.value = v; inp.dispatchEvent(new Event('input', { bubbles: true })); inp.dispatchEvent(new Event('change', { bubbles: true })); n++; await wait(60); } }
          for (const b of fig.querySelectorAll('.la-seg button, .la-fig-controls button, .la-stepper-nav button, .la-stepper-bar button, button[data-act]')) { b.click(); n++; await wait(50); }
          for (const s of fig.querySelectorAll('select')) { for (const o of s.options) { s.value = o.value; s.dispatchEvent(new Event('change', { bubbles: true })); n++; await wait(40); } }
          for (const c of fig.querySelectorAll('input[type=checkbox]')) { c.click(); n++; await wait(40); c.click(); await wait(40); }
          // clique no centro do canvas
          for (const cv of fig.querySelectorAll('canvas')) { const r = cv.getBoundingClientRect(); const o = { bubbles: true, clientX: r.left + r.width / 2, clientY: r.top + r.height / 2 }; cv.dispatchEvent(new PointerEvent('pointermove', o)); cv.dispatchEvent(new PointerEvent('pointerdown', o)); window.dispatchEvent(new PointerEvent('pointerup', o)); n++; }
          await wait(400);
          return { n, err: Array.from(fig.querySelectorAll('.la-fig-error')).map((e) => e.textContent) };
        }, i);
        f.interactions = inter.n;
        if (inter.err.length) f.err = (f.err || []).concat(inter.err);
      }
      const blank = f.canvases.length && f.canvases.every((c) => (c.ink || 0) < 0.002) && f.domNodes < 12;
      if (blank) r.issues.push(`fig ${i + 1} (“${f.title}”) parece vazia (canvas sem tinta)`);
      if (f.err && f.err.length) r.issues.push(`fig ${i + 1} erro: ${f.err.join(' | ')}`);
      if (f.overflowX) r.issues.push(`fig ${i + 1} tem overflow horizontal`);
      r.figures.push(f);
    }
    const ov = await page.evaluate(() => document.documentElement.scrollWidth > document.documentElement.clientWidth + 2);
    if (ov) r.issues.push('a página tem rolagem horizontal (overflow)');
    r.errors = errs.slice();
    if (r.errors.length) r.issues.push(`${r.errors.length} erro(s) de console/página`);
    report.chapters[id] = r;
    console.log(`${r.issues.length ? '✗' : '✓'} ${id}: ${r.nFigs} figs, ${r.words} palavras, quiz ${r.quiz}${r.issues.length ? '\n    - ' + r.issues.join('\n    - ') : ''}${r.errors.length ? '\n      ' + r.errors.slice(0, 6).join('\n      ') : ''}`);
  }
  await browser.close();
  fs.mkdirSync(OUT, { recursive: true });
  const rp = path.join(OUT, `report${LANG === 'en' ? '.en' : ''}${ids.length === 1 ? '-' + ids[0] : ''}${mobile ? '-m' : ''}.json`);
  fs.writeFileSync(rp, JSON.stringify(report, null, 2));
  if (loadErrors.length) console.log('ERROS DE CARGA:', loadErrors);
  if (homeErrs.length) console.log('Erros na home:', homeErrs.slice(0, 10));
  console.log('relatório → ' + rp);
})().catch((e) => { console.error(e); process.exit(1); });

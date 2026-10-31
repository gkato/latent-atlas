#!/usr/bin/env node
/* ==========================================================================
   LATENT ATLAS — tools/i18n-check.js
   Verifica a tradução inglesa de capítulos contra a versão pt:
   - paridade estrutural (seções, figuras e ids, termos de glossário, links,
     tabelas, blocos de código, callouts, equações, quiz/resumo/goals);
   - vazamento de português no texto visível do .en.html, nas strings do
     .en.js (rótulos de figuras, steppers, quiz) e no .en.glossary.json.
   Uso:  node tools/i18n-check.js              (todos)
         node tools/i18n-check.js atencao mla  (alguns ids)
         node tools/i18n-check.js --json       (saída JSON)
   Sai com código 1 se houver divergência estrutural ou vazamento relevante.
   ========================================================================== */
'use strict';
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const CH = path.join(__dirname, '..', 'chapters');
const args = process.argv.slice(2);
const asJson = args.includes('--json');
const want = args.filter((a) => !a.startsWith('--'));

// palavras funcionais do português que quase nunca aparecem em inglês técnico
const PT_WORDS = /\b(não|nao|são|também|está|estão|você|vocês|isso|isto|essa|esse|essas|esses|aquela|aquele|quando|onde|porque|então|muito|muitos|pelo|pela|pelos|pelas|para|uma|umas|uns|como|mais|menos|cada|seção|capítulo|figura|receita|receitas|modelo|modelos|camada|camadas|atenção|memória|número|valor|valores|passo|próximo|anterior|sobre|entre|depois|antes|ainda|agora|aqui|também|mesmo|mesma|quase|sempre|nunca|nenhum|nenhuma|todos|todas|tudo|nada|qual|quais|porém|assim|já|só|até|após|sem|com|dos|das|nos|nas|pelo|aos|às|é|são|foi|ser|ter|tem|têm|faz|fazer|vai|vamos|pode|podem|deve|devem|precisa|mostra|mostram|clique|mexa|repare|veja|olhe)\b/gi;
const ACCENT = /[ãõçáéíóúâêôà]/i;

function stripHtml(html) {
  return html
    .replace(/<pre[\s\S]*?<\/pre>/gi, ' ')
    .replace(/<code[\s\S]*?<\/code>/gi, ' ')
    .replace(/<script[\s\S]*?<\/script>/gi, ' ')
    .replace(/\\\([\s\S]*?\\\)|\\\[[\s\S]*?\\\]|\$\$[\s\S]*?\$\$/g, ' ')
    .replace(/<[^>]+>/g, ' ')
    .replace(/&[a-z#0-9]+;/gi, ' ')
    .replace(/\s+/g, ' ');
}
function count(re, s) { return (s.match(re) || []).length; }
function setOf(re, s) { const out = new Set(); let m; const r = new RegExp(re.source, 'g'); while ((m = r.exec(s))) out.add(m[1]); return out; }
function diff(a, b) { return [...a].filter((x) => !b.has(x)); }
function leaks(text, max = 12) {
  const hits = [];
  const sentences = text.split(/(?<=[.!?:;])\s+/);
  for (const s of sentences) {
    const w = s.match(PT_WORDS) || [];
    const strong = w.filter((x) => !/^(a|e|o|as|os|para|com|sem|como|mais|menos|entre|sobre|até|só|já|é)$/i.test(x));
    if ((strong.length >= 2) || (ACCENT.test(s) && w.length >= 1)) hits.push(s.trim().slice(0, 160));
    if (hits.length >= max) break;
  }
  return hits;
}
function metaOf(file) {
  let def = null;
  const LA = new Proxy({ chapter: (d) => { def = d; } }, { get: (t, k) => (k in t ? t[k] : new Proxy(() => {}, { get: () => () => {} })) });
  try { vm.runInNewContext(fs.readFileSync(file, 'utf8'), { LA, window: {}, document: {}, console }); } catch (e) { return { err: e.message }; }
  return def || { err: 'LA.chapter não chamado' };
}
function jsStringLeaks(file, max = 15) {
  const src = fs.readFileSync(file, 'utf8');
  const out = [];
  const re = /(['"`])((?:\\.|(?!\1)[^\\\n])*)\1/g;
  let m;
  while ((m = re.exec(src))) {
    const s = m[2];
    if (s.length < 4) continue;
    if (/^[\w.#\-/:()%, ]+$/.test(s) && !ACCENT.test(s)) continue; // identificadores, cores, seletores
    const w = (s.match(PT_WORDS) || []).filter((x) => !/^(a|e|o|as|os|com|sem|como|mais|menos|só|já|é|para|entre|sobre)$/i.test(x));
    if (ACCENT.test(s) || w.length >= 2) out.push(s.slice(0, 120));
    if (out.length >= max) break;
  }
  return out;
}

const files = fs.readdirSync(CH);
const bases = files.filter((f) => /^\d{2,3}-[a-z0-9-]+\.html$/.test(f)).map((f) => f.replace(/\.html$/, '')).sort();
const report = {};
let bad = 0;
for (const base of bases) {
  const id = base.replace(/^\d+-/, '');
  if (want.length && !want.includes(id)) continue;
  const r = { id, translated: false, issues: [], leaks: [] };
  report[id] = r;
  const enH = path.join(CH, base + '.en.html'), enJ = path.join(CH, base + '.en.js'), enG = path.join(CH, base + '.en.glossary.json');
  if (!fs.existsSync(enH) || !fs.existsSync(enJ)) { r.issues.push('sem tradução (.en.html/.en.js ausentes)'); bad++; continue; }
  r.translated = true;
  const ptHtml = fs.readFileSync(path.join(CH, base + '.html'), 'utf8');
  const enHtml = fs.readFileSync(enH, 'utf8');
  const checks = [
    ['seções', /<section class="la-sec"/g], ['h2', /<h2[\s>]/g], ['h3', /<h3[\s>]/g], ['figuras', /<figure class="la-fig/g],
    ['tabelas', /<table/g], ['blocos de código', /<pre/g], ['callouts', /class="la-callout/g], ['equações', /class="la-eq"/g],
    ['canvas', /<canvas/g], ['controles', /class="la-fig-controls"/g],
  ];
  for (const [nm, re] of checks) { const a = count(re, ptHtml), b = count(re, enHtml); if (a !== b) r.issues.push(`${nm}: pt ${a} ≠ en ${b}`); }
  const ids = /\sid="([^"]+)"/;
  const miss = diff(setOf(ids, ptHtml), setOf(ids, enHtml)); if (miss.length) r.issues.push('ids ausentes no en: ' + miss.slice(0, 12).join(', '));
  const terms = /data-term="([^"]+)"/;
  const mt = diff(setOf(terms, ptHtml), setOf(terms, enHtml)); if (mt.length) r.issues.push('data-term ausentes no en: ' + mt.slice(0, 12).join(', '));
  const extraT = diff(setOf(terms, enHtml), setOf(terms, ptHtml)); if (extraT.length) r.issues.push('data-term novos no en (precisam existir no glossário): ' + extraT.slice(0, 12).join(', '));
  const links = /data-ch="([^"]+)"/;
  const ml = diff(setOf(links, ptHtml), setOf(links, enHtml)); if (ml.length) r.issues.push('links data-ch ausentes no en: ' + ml.join(', '));
  r.leaks = leaks(stripHtml(enHtml));
  // JS
  const pm = metaOf(path.join(CH, base + '.js')), em = metaOf(enJ);
  if (em.err) r.issues.push('en.js não avalia: ' + em.err);
  else if (!pm.err) {
    if (em.id !== pm.id) r.issues.push(`id difere: ${em.id} ≠ ${pm.id}`);
    for (const k of ['part', 'order']) if (em[k] !== pm[k]) r.issues.push(`${k} difere`);
    for (const k of ['quiz', 'summary', 'goals', 'prereqs']) if ((em[k] || []).length !== (pm[k] || []).length) r.issues.push(`${k}: pt ${(pm[k] || []).length} ≠ en ${(em[k] || []).length}`);
    (em.quiz || []).forEach((q, i) => { if (pm.quiz && pm.quiz[i] && q.a !== pm.quiz[i].a) r.issues.push(`quiz ${i + 1}: resposta certa difere`); });
    const metaText = [em.title, em.subtitle, em.navTitle].concat(em.goals || [], em.summary || [], (em.quiz || []).flatMap((q) => [q.q, q.e].concat(q.o || []))).join(' . ');
    r.leaks = r.leaks.concat(leaks(stripHtml(metaText), 6).map((x) => '[meta] ' + x));
  }
  r.jsLeaks = jsStringLeaks(enJ);
  // glossário
  const pg = path.join(CH, base + '.glossary.json');
  if (fs.existsSync(pg)) {
    if (!fs.existsSync(enG)) r.issues.push('falta .en.glossary.json');
    else {
      try {
        const a = JSON.parse(fs.readFileSync(pg, 'utf8')), b = JSON.parse(fs.readFileSync(enG, 'utf8'));
        const mk = Object.keys(a).filter((k) => !b[k]); if (mk.length) r.issues.push('glossário en sem as chaves: ' + mk.slice(0, 10).join(', '));
        const gl = Object.keys(b).filter((k) => b[k] && (ACCENT.test(b[k].d || '') || leaks(stripHtml(b[k].d || ''), 1).length)); if (gl.length) r.leaks.push('[glossário] termos com português: ' + gl.slice(0, 10).join(', '));
      } catch (e) { r.issues.push('glossário en inválido: ' + e.message); }
    }
  }
  if (r.issues.length || r.leaks.length > 3 || r.jsLeaks.length > 5) bad++;
}
if (asJson) console.log(JSON.stringify(report, null, 1));
else {
  for (const r of Object.values(report)) {
    const ok = !r.issues.length && r.leaks.length <= 3 && (r.jsLeaks || []).length <= 5;
    console.log(`${ok ? '✓' : '✗'} ${r.id}${r.translated ? '' : ' (não traduzido)'}`);
    r.issues.forEach((x) => console.log('    - ' + x));
    if (r.leaks.length) console.log('    · possível português no texto:\n        ' + r.leaks.slice(0, 8).join('\n        '));
    if ((r.jsLeaks || []).length) console.log('    · possível português em strings do .en.js:\n        ' + r.jsLeaks.slice(0, 8).join('\n        '));
  }
  const tot = Object.keys(report).length;
  console.log(`\n${tot - bad}/${tot} capítulos ok.`);
}
process.exitCode = bad ? 1 : 0;

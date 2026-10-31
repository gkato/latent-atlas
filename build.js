#!/usr/bin/env node
/* ==========================================================================
   LATENT ATLAS — build.js
   Junta os capítulos em um bundle por idioma:
     pt → dist/chapters.bundle.js      (chapters/NN-slug.html|.js|.glossary.json)
     en → dist/chapters.en.bundle.js   (chapters/NN-slug.en.html|.en.js|.en.glossary.json;
                                        cai para a versão pt quando a tradução não existe)
   - O HTML vira string JSON (sem problemas de crase/escape).
   - Cada JS é validado sintaticamente (node:vm) e isolado em try/catch.
   - Glossários por capítulo completam LA.glossary (o curado em data/ tem prioridade;
     entre capítulos, vence o de menor número — quem introduz o termo primeiro).
   Uso:  node build.js                          (gera os dois idiomas)
         node build.js --lang en                (só um idioma)
         node build.js --check                  (só valida)
         node build.js --lang en --out <arq>    (gera em outro caminho — usado pelo QA)
   ========================================================================== */
'use strict';
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const ROOT = __dirname;
const CH = path.join(ROOT, 'chapters');
const argv = process.argv.slice(2);
const opt = (k) => { const i = argv.indexOf(k); return i >= 0 ? argv[i + 1] : null; };
const checkOnly = argv.includes('--check');
const quiet = argv.includes('--quiet');
const langs = opt('--lang') ? [opt('--lang')] : ['pt', 'en'];
const outOverride = opt('--out');
if (outOverride && langs.length > 1) { console.error('--out exige --lang'); process.exit(2); }

/* ids de seção: o bundle inglês recebe nos <h2> sem id o MESMO id que o core.js
   geraria a partir do título em português — links data-sec, busca e URLs valem nos dois idiomas */
const ENT = { amp: '&', lt: '<', gt: '>', quot: '"', apos: "'", nbsp: '\u00a0', ndash: '–', mdash: '—', hellip: '…', laquo: '«', raquo: '»', middot: '·', times: '×', rarr: '→', larr: '←', le: '≤', ge: '≥', asymp: '≈' };
const decode = (t) => t.replace(/&(#x[0-9a-f]+|#\d+|[a-z]+);/gi, (m, e) => e[0] === '#' ? String.fromCodePoint(e[1] === 'x' || e[1] === 'X' ? parseInt(e.slice(2), 16) : parseInt(e.slice(1), 10)) : (ENT[e.toLowerCase()] != null ? ENT[e.toLowerCase()] : m));
const normId = (t) => t.toLowerCase().normalize('NFD').replace(/[\u0300-\u036f]/g, '').replace(/[^a-z0-9]+/g, '-').slice(0, 40);
const H2 = /<h2(\s[^>]*)?>([\s\S]*?)<\/h2>/gi;
function ptH2Ids(html) {
  const ids = []; let m, n = 0;
  while ((m = H2.exec(html))) { n++; const idm = (m[1] || '').match(/\sid="([^"]+)"/); ids.push(idm ? idm[1] : 's' + n + '-' + normId(decode(m[2].replace(/<[^>]+>/g, '')))); }
  H2.lastIndex = 0; return ids;
}
function injectIds(ptHtml, enHtml) {
  const ids = ptH2Ids(ptHtml); let i = 0;
  return enHtml.replace(H2, (full, attrs, inner) => { const id = ids[i++]; if (!id || /\sid="/.test(attrs || '')) return full; return `<h2 id="${id}"${attrs || ''}>${inner}</h2>`; });
}

const all = fs.readdirSync(CH);
const slugs = Array.from(new Set(all.filter((f) => /^\d{2,3}-[a-z0-9-]+\.(html|js)$/.test(f)).map((f) => f.replace(/\.(html|js)$/, '')))).sort();
let failed = false;

for (const lang of langs) {
  const sfx = lang === 'en' ? '.en' : '';
  const OUT = outOverride ? path.resolve(outOverride) : path.join(ROOT, 'dist', `chapters${sfx}.bundle.js`);
  let out = `/* gerado por build.js (${lang}) — não edite à mão */\n`;
  const problems = [];
  const gloss = {};
  let ok = 0, fallbacks = [];

  for (const base of slugs) {
    const id = base.replace(/^\d+-/, '');
    let jsPath = path.join(CH, base + sfx + '.js');
    let htmlPath = path.join(CH, base + sfx + '.html');
    let glPath = path.join(CH, base + sfx + '.glossary.json');
    let fallback = false;
    if (lang !== 'pt' && !fs.existsSync(jsPath)) {
      fallback = true; fallbacks.push(id);
      jsPath = path.join(CH, base + '.js'); htmlPath = path.join(CH, base + '.html'); glPath = path.join(CH, base + '.glossary.json');
    }
    if (!fs.existsSync(jsPath)) { problems.push(`${base}: falta ${path.basename(jsPath)}`); continue; }
    const js = fs.readFileSync(jsPath, 'utf8');
    let html = fs.existsSync(htmlPath) ? fs.readFileSync(htmlPath, 'utf8') : '';
    if (lang !== 'pt' && !fallback && html && fs.existsSync(path.join(CH, base + '.html'))) html = injectIds(fs.readFileSync(path.join(CH, base + '.html'), 'utf8'), html);
    try {
      new vm.Script(js, { filename: path.basename(jsPath) });
    } catch (e) {
      problems.push(`${path.basename(jsPath)}: ERRO DE SINTAXE — ${e.message}`);
      out += `LA.chapterLoadError(${JSON.stringify(id)}, ${JSON.stringify('Erro de sintaxe: ' + e.message)});\n`;
      continue;
    }
    if (!new RegExp(`id\\s*:\\s*['"\`]${id}['"\`]`).test(js)) problems.push(`${path.basename(jsPath)}: o id do LA.chapter deve ser '${id}' (derivado do nome do arquivo)`);
    if (!html) problems.push(`${base}: falta ${path.basename(htmlPath)} (conteúdo)`);
    if (fs.existsSync(glPath)) {
      try {
        const g = JSON.parse(fs.readFileSync(glPath, 'utf8'));
        for (const k of Object.keys(g)) {
          if (!/^[a-z0-9][a-z0-9-]*$/.test(k)) { problems.push(`${path.basename(glPath)}: chave inválida '${k}' (use kebab-case)`); continue; }
          if (!g[k] || !g[k].t || !g[k].d) { problems.push(`${path.basename(glPath)}: '${k}' precisa de {t, d}`); continue; }
          if (!gloss[k]) gloss[k] = { t: g[k].t, d: g[k].d, ch: g[k].ch || id };
        }
      } catch (e) { problems.push(`${path.basename(glPath)}: JSON inválido — ${e.message}`); }
    }
    out += `\n/* ---- ${base}${fallback ? ' (fallback pt)' : sfx} ---- */\nLA.content[${JSON.stringify(id)}] = ${JSON.stringify(html)};\n`;
    out += `(function(){\ntry {\n${js}\n} catch (e) { LA.chapterLoadError(${JSON.stringify(id)}, e); }\n${fallback ? `if (LA.byId[${JSON.stringify(id)}]) LA.byId[${JSON.stringify(id)}].__fallback = true;\n` : ''}})();\n`;
    ok++;
  }
  out += `\n/* ---- glossários dos capítulos ---- */\n(function(){ var g = ${JSON.stringify(gloss)}; for (var k in g) if (!LA.glossary[k]) LA.glossary[k] = g[k]; })();\n`;

  if (problems.length) console.log(`[${lang}] Problemas:\n  - ` + problems.join('\n  - '));
  if (!quiet) console.log(`[${lang}] ${ok}/${slugs.length} capítulos válidos · ${Object.keys(gloss).length} termos de glossário${fallbacks.length ? ` · ${fallbacks.length} sem tradução (usando pt)` : ''}.`);
  if (!checkOnly) {
    fs.mkdirSync(path.dirname(OUT), { recursive: true });
    const tmp = OUT + '.tmp-' + process.pid;
    fs.writeFileSync(tmp, out);
    fs.renameSync(tmp, OUT);
    if (!quiet) console.log(`[${lang}] → ` + path.relative(ROOT, OUT) + ' (' + (out.length / 1024).toFixed(0) + ' KiB)');
  }
  if (problems.some((p) => /SINTAXE/.test(p))) failed = true;
}
process.exitCode = failed ? 1 : 0;

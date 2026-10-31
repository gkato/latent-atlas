LA.chapter({
  id: 'tres-modelos',
  part: 3,
  order: 18,
  title: 'The three models dissected',
  navTitle: 'The three models',
  subtitle: 'GLM-5.3-Flash, Qwen3.8-Flash-Next and DeepSeek-V4-Flash side by side: layer-by-layer map, config.json field by field, active parameters, KV bytes, lineage and what the engine needs to support.',
  minutes: 55,
  recipes: ['glm', 'qwen', 'ds'],
  tags: ['architecture', 'config.json', 'active parameters', 'KV cache', 'hybrid', 'MoE', 'lineage'],
  goals: [
    'Read a frontier <code>config.json</code> field by field and draw the <b>layer map</b> (mixer, FFN, extras)',
    'Compute <b>total</b> and <b>active</b> parameters, FLOPs per token and the <b>distinct experts</b> read per decode step',
    'Compute <b>KV bytes per token</b> and <b>state per sequence</b> in hybrid models, and compare them with full attention',
    'Turn the architecture into a <b>list of engine requirements</b> and recognize the lineage of each part',
  ],
  prereqs: ['mla', 'atencao-esparsa', 'atencao-linear', 'hiperconexoes', 'ngram-mtp', 'multimodal'],
  summary: [
    'All three share the decoder-only MoE skeleton, but swap the <b>mixer</b> (linear/sparse hybrid), the <b>residual width</b> (4 streams) and the <b>accessories</b> (N-gram, MTP/drafters, vision).',
    'Maps: GLM = 45 layers, 34 KDA + 11 MLA-DSA (<code>i mod 4 = 3</code>) + MTP at index 45; Qwen = 48, 36 GDN + 12 QSA + MTP instantiated as 48; DeepSeek = 43, 2 SWA + 21 C4A + 20 C128A, hash in 0–2 and DSpark reading 40–42.',
    '~96–97% of the parameters are routed experts (311.6 of 321 B in GLM counting the MTP; 120.8 of 125.7 B in the Qwen backbone; 277 of 284 B in DeepSeek): memory follows the total and decode follows the active bytes (~16.7 / ~6.7 / ~13.2 B per token, counting the lm_head), which grow with the distinct experts per step: \\( E[1-(1-k/E)^t] \\).',
    'KV that grows, per rank: 7,579 B/token (GLM), 6,656 B/token per GPU in FP8 (Qwen) and ~3,850 B/token (DeepSeek); fixed per sequence: ~76 MiB of KDA state, ~27 MiB of GDN state, ~3.4 MB of windows.',
    'Lineage: MLA (DeepSeek-V2) → GLM and V4; DSA indexer (V3.2) → kpool, QSA and C4A; linear attention (Qwen3-Next, Kimi Linear) → GLM and Qwen; mHC → GLM and V4.',
    'The FLOPs ≈ 2·N<sub>active</sub> rule only counts the weights: at long context, attention and above all the indexer (which scores all L/4 keys) take over, which is why GLM and Qwen cap each query at 2048 tokens + tail.',
    'Every unusual config field becomes an engine requirement. In all three recipes, sparse attention, hybrid KV and the drafter required patches.',
  ],
  quiz: [
    { q: 'In GLM-5.3-Flash, <code>num_hidden_layers</code> = 45 and the MLA layers follow <code>i mod 4 = 3</code>. How many MLA layers are there in the decoder, and what index does the MTP layer have?', o: ['12 MLA; MTP at index 44', '11 MLA (3, 7, …, 43); MTP at index 45', '11 MLA (0, 4, …, 40); MTP at index 46', '34 MLA; MTP at index 45'], a: 1, e: 'From 0 to 44, the indices that leave remainder 3 when divided by 4 are 3, 7, …, 43: 11 layers. The MTP is the 46th entry, index 45, and vLLM uses <code>num_hidden_layers</code> as its index. The README\'s "12 MLA" counts the MTP; 34 is the number of KDA layers.' },
    { q: 'Why is quantizing <i>only</i> the routed experts enough for the three models to fit on 2 nodes?', o: ['Because the experts are the most sensitive part of the model', 'Because ~96–97% of the parameters are in the routed experts', 'Because the experts are not read during decode', 'Because attention has no weights'], a: 1, e: 'Routed experts add up to 311.6 of 321 B in GLM (43 MoE layers: the 42 in the decoder plus the MTP; without the MTP it would be 304.4 B, ≈ 95%), 121 of 126 B in the Qwen backbone and 277 of 284 B in DeepSeek. Bringing them down to ~4 bits solves memory, and the small, sensitive parts (attention, indexer, router, residual) can stay at 8 or 16 bits.' },
    { q: 'GLM verifies 8 tokens per step with DFlash2 (k = 7). With 288 experts and top-8, how many distinct experts per layer should you expect to read, under uniform routing?', o: ['8', '~57–58', '64', '288'], a: 1, e: '\\( 288[1-(1-8/288)^8] \\approx 58 \\) (≈ 57 in the approximation of 64 independent picks used in other chapters, \\( 288[1-(1-1/288)^{64}] \\)). It isn\'t 64 because the tokens overlap, and it isn\'t 8 because each token picks its own. That\'s why speculation pays off less in an MoE than in a dense model: each extra token wakes up new experts, and the bytes read per step grow.' },
    { q: 'In DeepSeek-V4-Flash, why does the KV cache per rank <b>not</b> shrink when going from TP=2 to TP=3?', o: ['Because the KV is stored in FP8', 'Because <code>num_key_value_heads</code> = 1: there is a single KV vector per token, replicated on every rank', 'Because TP=3 is not supported', 'Because the SWA windows grow with TP'], a: 1, e: 'TP splits heads across GPUs, but V4 has only one KV "head" (K = V = 512). Each rank stores the whole cache, and the indexer\'s too. TP=3 only helps indirectly, by freeing up weight memory: the pool goes up to ~35 GiB per rank.' },
    { q: 'The GLM <code>fp8_ds_mla</code> row is 656 B, of which 128 B are zeros. Why?', o: ['It is page-alignment padding', 'The model is NoPE (<code>qk_rope_head_dim: 0</code>), but the only sparse-MLA kernel for SM12x requires 64 RoPE dimensions; the recipe pads with zeros, which is exact', 'They are FP32 scales not yet computed', 'It is room reserved for the indexer'], a: 1, e: 'A null block adds 0 to the \\(q \\cdot k\\) product: the softmax doesn\'t change. The cost is ~24% more KV in the 11 DSA layers (656 instead of ~528 B), but it avoids writing a new kernel.' },
    { q: 'In the Qwen recipe, YaRN placed in <code>--hf-overrides \'{"rope_parameters":…}\'</code> had no effect. What was the fix?', o: ['Use the old <code>--rope-scaling</code> flag', 'Nest the override inside <code>"text_config"</code>', 'Increase <code>max_position_embeddings</code>', 'Turn off MTP'], a: 1, e: 'The override merge only descends into keys that are nested configs. In <code>qwen4_exp</code>, the parent also exposes a plain <code>rope_parameters</code>: the top-level override was written to the parent and never reached <code>text_config</code>, which is what the model reads. The recipe (<code>start.sh:919-946</code>) switched to nesting <code>{rope_type: "yarn", factor, original_max_position_embeddings: 262144}</code> inside <code>text_config</code>. A precaution, not verified in the kit: the official card passes the full dictionary, and the inventory points out the risk that the dict gets replaced wholesale; to be safe, repeat θ = 10⁷, the 0.25 fraction and the mRoPE. And 1M with YaRN has not yet been validated in the kit (<code>YARN_ENABLE</code> is still <code>false</code>). The other options miss the point: the problem was where the override landed.' },
  ],
  mount(root, LA) {
    const V = LA.viz, UI = LA.ui, h = LA.h;

    /* ================================================================ estilo local */
    const css = `
      .t18-panel{padding:14px 18px;border-top:1px solid var(--line);background:color-mix(in srgb,var(--surface-2) 60%,transparent);font-size:14px;line-height:1.55;color:var(--text-2)}
      .t18-panel h5{margin:0 0 6px;font-family:var(--font-display);text-transform:uppercase;letter-spacing:.08em;font-size:13px;color:var(--text)}
      .t18-panel .chip{display:inline-block;padding:1px 9px;border-radius:99px;font-size:11.5px;font-family:var(--font-mono);margin:0 6px 4px 0;border:1px solid var(--line-2);color:var(--text)}
      .t18-cols{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px 22px;margin-top:8px}
      .t18-tlist{margin:4px 0 0;padding:0;list-style:none;font-size:12.5px}
      .t18-tlist li{display:flex;gap:8px;justify-content:space-between;border-bottom:1px dashed var(--line);padding:3px 0;margin:0}
      .t18-tlist li::marker{content:''}
      .t18-tlist li span:first-child{color:var(--text-2);min-width:0;overflow-wrap:anywhere}
      .t18-tlist li code{white-space:nowrap;font-size:11.5px}
      .t18-facts{margin:8px 0 0;padding-left:1.1em;font-size:13px}
      .t18-facts li{margin:.15em 0}
      .t18-grid2{display:grid;grid-template-columns:minmax(0,1.15fr) minmax(0,1fr);gap:14px}
      @media (max-width:720px){.t18-grid2{grid-template-columns:1fr}}
      .t18-json{font-family:var(--font-mono);font-size:12.3px;line-height:1.5;max-height:420px;overflow:auto;background:var(--code-bg);color:#D6F5E5;border-radius:12px;padding:10px 0;border:1px solid var(--line-2)}
      .t18-json .cm{display:block;padding:6px 14px 2px;color:#6E8C7C;font-style:italic}
      .t18-json button{display:block;width:100%;text-align:left;background:none;border:none;color:inherit;font:inherit;padding:2px 14px 2px 22px;cursor:pointer;white-space:normal;overflow-wrap:anywhere;border-left:3px solid transparent}
      .t18-json button:hover{background:rgba(63,244,155,.07)}
      .t18-json button.on{background:rgba(63,244,155,.16);border-left-color:#3FF49B}
      .t18-json .k{color:#89FFC5}.t18-json .v{color:#FCBB00}
      .t18-exp{font-size:14px;line-height:1.6;color:var(--text-2);min-width:0}
      .t18-exp .kk{font-family:var(--font-mono);color:var(--accent-text);font-size:13px;overflow-wrap:anywhere}
      .t18-exp .vv{font-family:var(--font-mono);color:var(--text);font-size:12.5px;display:block;margin:4px 0 8px;overflow-wrap:anywhere}
      .t18-exp .aff{margin-top:8px;font-size:12.5px;color:var(--text-3)}
      dl.la-kv.t18-kit{grid-template-columns:minmax(0,1fr);gap:2px 0;margin:.8em 0 .2em}
      dl.la-kv.t18-kit dd{margin:0 0 10px}
      .t18-req h5 .st{font-family:var(--font-mono);font-size:11px;padding:1px 8px;border-radius:99px;margin-left:8px;vertical-align:middle}
    `;
    const styleEl = document.createElement('style');
    styleEl.textContent = css;
    document.head.appendChild(styleEl);
    LA.onCleanup(() => styleEl.remove());

    /* ================================================================ utilidades */
    const nf = (x, d = 0) => (+x).toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d });
    const fB = (b) => {
      const u = ['B', 'KiB', 'MiB', 'GiB', 'TiB']; let i = 0;
      while (Math.abs(b) >= 1024 && i < u.length - 1) { b /= 1024; i++; }
      return nf(b, i === 0 ? 0 : b < 10 ? 2 : 1) + ' ' + u[i];
    };
    const NAME = { glm: 'GLM-5.3-Flash', qwen: 'Qwen3.8-Flash-Next', ds: 'DeepSeek-V4-Flash' };
    const SHORT = { glm: 'GLM', qwen: 'Qwen', ds: 'DeepSeek' };
    const MODEL_OPTS = [{ v: 'glm', label: 'GLM-5.3' }, { v: 'qwen', label: 'Qwen3.8' }, { v: 'ds', label: 'DS-V4' }];
    function wrapText(g, s, maxW, size, weight, font) {
      g.font = V.font(size, font || 'sans', weight || 400);
      const words = String(s).split(' '); const lines = []; let cur = '';
      for (const w of words) { const t = cur ? cur + ' ' + w : w; if (cur && g.measureText(t).width > maxW) { lines.push(cur); cur = w; } else cur = t; }
      if (cur) lines.push(cur);
      return lines;
    }
    function textW(g, s, size, font, weight) { g.font = V.font(size, font || 'sans', weight || 400); return g.measureText(String(s)).width; }
    function autoHeight(fig, fn) {
      const stage = fig.querySelector('.la-fig-stage');
      const apply = () => { const w = stage.clientWidth || 800; const hh = Math.round(fn(w)); if (Math.abs(stage.offsetHeight - hh) > 2) stage.style.height = hh + 'px'; };
      apply();
      const ro = new ResizeObserver(apply); ro.observe(stage); LA.onCleanup(() => ro.disconnect());
      return apply;
    }
    const fitTo = (fig, need) => { const st = fig.querySelector('.la-fig-stage'); need = Math.ceil(need); if (Math.abs(st.offsetHeight - need) > 3) st.style.height = need + 'px'; };
    const hatch = (g, x, y, w, hh, col) => {
      g.save(); g.beginPath(); g.rect(x, y, w, hh); g.clip();
      g.strokeStyle = col; g.lineWidth = 1.2;
      for (let k = -hh; k < w + hh; k += 6) { g.beginPath(); g.moveTo(x + k, y + hh); g.lineTo(x + k + hh, y); g.stroke(); }
      g.restore();
    };

    /* ================================================================ dados: camadas */
    function layersOf(m) {
      const L = [];
      if (m === 'glm') {
        const taps = [5, 14, 24, 33, 42];
        for (let i = 0; i < 45; i++) L.push({ i, mix: i % 4 === 3 ? 'dsa' : 'kda', ffn: i < 3 ? 'dense' : 'moe', tap: taps.includes(i), role: i >= 15 ? 'edit' : 'anchor' });
        L.push({ i: 45, mix: 'dsa', ffn: 'moe', extra: 'mtp', tag: 'MTP', role: 'edit' });
      } else if (m === 'qwen') {
        for (let i = 0; i < 48; i++) L.push({ i, mix: (i + 1) % 4 === 0 ? 'qsa' : 'gdn', ffn: 'moe', ple: i === 1, role: (i + 1) % 4 === 0 ? 'edit' : null });
        L.push({ i: 48, mix: 'qsa', ffn: 'moe', extra: 'mtp', tag: 'MTP' });
      } else {
        for (let i = 0; i < 43; i++) {
          const cr = i < 2 ? 0 : i % 2 === 0 ? 4 : 128;
          L.push({ i, mix: cr === 0 ? 'swa' : cr === 4 ? 'c4a' : 'c128a', ffn: i < 3 ? 'hash' : 'moe', tap: i >= 40, role: i >= 10 ? 'edit' : null });
        }
        for (let s = 0; s < 3; s++) L.push({ i: 43 + s, mix: 'swa', ffn: 'moe', extra: 'dspark', tag: 'DS' + s, stage: s });
      }
      return L;
    }
    const LAYERS = { glm: layersOf('glm'), qwen: layersOf('qwen'), ds: layersOf('ds') };
    const MIXNAME = { kda: 'KDA', gdn: 'GDN', dsa: 'MLA-DSA', qsa: 'QSA', c4a: 'C4A', c128a: 'C128A', swa: 'SWA', full: 'GA' };
    // 'full' = atenção global densa (sem indexador), usada só no exercício do MiMo
    const mixCol = (C, t) => (t === 'kda' || t === 'gdn') ? C.a : (t === 'dsa' || t === 'qsa' || t === 'c4a') ? C.b : t === 'c128a' ? C.c : t === 'full' ? C.e : C.d;
    const ffnCol = (C, f) => f === 'dense' ? V.alpha(C.ink, 0.9) : f === 'hash' ? C.f : V.alpha(C.ink, 0.38);
    // bytes de KV que crescem com o contexto, por token, por camada, por rank
    const KVB = { kda: 0, gdn: 0, dsa: 689, qsa: 512, c4a: 179, c128a: 4.56, swa: 0 };
    // rampa sequencial de um só matiz (claro = barato, escuro/saturado = caro), em escala log
    const kvCol = (C, b) => b > 0 ? V.cmap(0.22 + 0.78 * Math.min(1, Math.log10(1 + b) / Math.log10(700)), 'seq') : V.alpha(C.muted, 0.35);

    /* ---------- desenho compartilhado de uma faixa de camadas ---------- */
    function stripGeom(w, L, o) {
      const pad = o.pad == null ? 14 : o.pad, gap = o.gap == null ? 3 : o.gap;
      const nMain = L.filter((l) => !l.extra).length, nEx = L.length - nMain;
      const slots = nMain + (nEx ? nEx + 1 : 0);
      const W = w - 2 * pad;
      let perRow = slots, cw = (W - gap * (slots - 1)) / slots;
      if (cw < o.minCell) { perRow = Math.max(6, Math.floor((W + gap) / (o.minCell + gap))); cw = (W - gap * (perRow - 1)) / perRow; }
      cw = Math.min(cw, o.maxCell || 30);
      const rows = Math.ceil(slots / perRow);
      const rowH = o.mh + o.fh + (o.labelH || 0) + (o.markH || 0) + 8;
      const pos = L.map((l, k) => {
        const s = l.extra ? nMain + 1 + (k - nMain) : k;
        const r = Math.floor(s / perRow), c = s % perRow;
        return { x: pad + c * (cw + gap), y: o.y + r * rowH + (o.markH || 0), l };
      });
      return { pos, cw, rows, rowH, height: rows * rowH, perRow };
    }
    function drawStrip(g, G, o) {
      const C = V.C();
      const { pos, cw } = G;
      pos.forEach((p, k) => {
        const l = p.l;
        const lit = o.lit ? o.lit(l) : true;
        const a = lit ? 1 : 0.16;
        let mc = mixCol(C, l.mix);
        if (o.mode === 'kv') mc = kvCol(C, KVB[l.mix]);
        if (o.mode === 'role') { mc = l.role === 'edit' ? C.c : l.role === 'anchor' ? V.alpha(C.muted, 0.55) : V.alpha(C.muted, 0.25); }
        g.globalAlpha = a;
        V.box(g, p.x, p.y, cw, o.mh, { r: Math.min(4, cw / 4), fill: V.alpha(mc, l.extra ? 0.45 : 0.85) });
        if (l.extra) { g.setLineDash([3, 3]); V.rr(g, p.x - 1, p.y - 1, cw + 2, o.mh + o.fh + 4, 4); g.strokeStyle = C.c; g.lineWidth = 1.2; g.stroke(); g.setLineDash([]); }
        const fc = o.mode === 'role' && l.ffn !== 'hash' ? V.alpha(C.ink, 0.25) : ffnCol(C, l.ffn);
        V.box(g, p.x, p.y + o.mh + 2, cw, o.fh, { r: Math.min(3, cw / 5), fill: fc });
        g.globalAlpha = 1;
        // marcadores
        const mx = p.x + cw / 2, my = p.y - 6;
        if (l.tap && (!o.lit || lit)) { g.fillStyle = C.d; g.beginPath(); g.moveTo(mx, my - 4); g.lineTo(mx + 4, my); g.lineTo(mx, my + 4); g.lineTo(mx - 4, my); g.closePath(); g.fill(); }
        if (l.ple && (!o.lit || lit)) V.text(g, '★', mx, my + 4, { size: 12, align: 'center', color: C.f });
        if (o.labels) {
          const show = l.extra ? (l.extra === 'mtp' || l.stage === 1) : (cw >= 21 || l.i % 4 === 0 || o.sel === k);
          if (show) V.text(g, l.extra ? (l.extra === 'mtp' ? 'MTP' : 'DSpark') : String(l.i), mx, p.y + o.mh + o.fh + 14, { size: cw < 16 ? 9.5 : 10.5, font: 'mono', align: 'center', color: o.sel === k ? C.ink : C.muted });
        }
        if (o.sel === k) { g.strokeStyle = C.ink; g.lineWidth = 2; V.rr(g, p.x - 2.5, p.y - 2.5, cw + 5, o.mh + o.fh + 7, 5); g.stroke(); }
        if (o.hover === k && o.sel !== k) { g.strokeStyle = V.alpha(C.ink, 0.5); g.lineWidth = 1.2; V.rr(g, p.x - 2, p.y - 2, cw + 4, o.mh + o.fh + 6, 5); g.stroke(); }
      });
    }
    const hitStrip = (G, px, py, o) => G.pos.findIndex((p) => px >= p.x - 1 && px <= p.x + G.cw + 1 && py >= p.y - 8 && py <= p.y + o.mh + o.fh + 16);

    /* ================================================================ FIG 1: chassi */
    (function () {
      const fig = root.querySelector('#fig-chassi');
      const S = { m: 'glm', spread: 1, tgt: 1, choice: [], tok: 0 };
      const P = {
        classic: { name: 'Classic Transformer', lanes: 1, pat: ['A', 'A', 'A', 'A'], patL: ['attention', 'attention', 'attention', 'attention'], mixLabel: 'full attention in every layer', mixSub: 'KV grows in all of them', E: 0, k: 0, ffnLabel: 'dense MLP', ffnSub: 'every token uses everything', resid: 'single residual · pre-norm', extras: ['embedding', 'lm_head'] },
        glm: { name: 'GLM-5.3-Flash', lanes: 4, pat: ['L', 'L', 'L', 'S'], patL: ['KDA', 'KDA', 'KDA', 'MLA-DSA'], mixLabel: '3 KDA : 1 MLA-DSA', mixSub: 'KV only in the 11 MLA layers', E: 288, k: 8, ffnLabel: '288 experts · top-8 + 1', ffnSub: 'sigmoid · noaux_tc · ×2.5', resid: 'mHC: 4 × 4096, Sinkhorn mixing', extras: ['MTP (layer 45)', 'external DFlash2', 'ViT 24 blocks', '3 dense MLPs'] },
        qwen: { name: 'Qwen3.8-Flash-Next', lanes: 4, pat: ['L', 'L', 'L', 'S'], patL: ['GDN', 'GDN', 'GDN', 'QSA'], mixLabel: '3 GDN : 1 QSA', mixSub: 'KV only in the 12 QSA (+MTP)', E: 512, k: 10, ffnLabel: '512 experts · top-10 + 1', ffnSub: 'softmax · shared with σ gate', resid: 'Gated Residual: 4 × 2560', extras: ['N-gram 51 B (layer at index 1)', 'MTP (1 layer, 3×)', 'ViT 27 blocks'] },
        ds: { name: 'DeepSeek-V4-Flash', lanes: 4, pat: ['W', 'C4', 'C128', 'C4'], patL: ['SWA', 'C4A', 'C128A', 'C4A'], mixLabel: 'SWA 128 + C4A / C128A', mixSub: 'KV compressed ÷4 or ÷128', E: 256, k: 6, ffnLabel: '256 experts · top-6 + 1', ffnSub: 'sqrtsoftplus · hash in 0–2', resid: 'mHC: 4 × 4096, Sinkhorn mixing', extras: ['DSpark (3 stages)', 'hash in layers 0–2', 'ViT 32 layers'] },
      };
      const rng = V.rng(18);
      const repick = () => { const p = P[S.m]; S.choice = []; if (!p.E) return; const set = new Set(); while (set.size < p.k) set.add(rng.int(p.E)); S.choice = Array.from(set); S.tok++; };
      repick();
      V.every(1100, repick);
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Model', options: [{ v: 'classic', label: 'classic' }].concat(MODEL_OPTS), value: 'glm', onChange: (v) => { S.m = v; S.tgt = P[v].lanes; repick(); } });
      UI.legend(ctl, [[V.C().a, 'linear mixer (fixed state)'], [V.C().b, 'sparse / full attention'], [V.C().c, 'KV ÷128'], [V.C().d, '128-token window']]);
      const parts = Array.from({ length: 40 }, (_, i) => ({ u: i / 40, lane: i % 4, sp: 0.8 + (i % 5) * 0.08 }));
      V.fig(fig, (g, w, hh, t, dt) => {
        const C = V.C(), p = P[S.m];
        S.spread += (S.tgt - S.spread) * Math.min(1, dt * 4);
        const pad = 14, narrow = w < 560;
        // título
        V.text(g, p.name, pad, 24, { size: 15, font: 'display', weight: 600, color: S.m === 'classic' ? C.ink : C[S.m] });
        V.text(g, p.resid, w - pad, 24, { size: 11.5, font: 'mono', align: 'right', color: C.muted, maxW: w * 0.5 });
        // faixas do resíduo
        const ly = 56, lx0 = pad + (narrow ? 54 : 64), lx1 = w - pad - (narrow ? 50 : 64);
        const nL = 4, laneGap = 7;
        const laneY = (j) => ly + (j - 1.5) * laneGap * Math.max(0, (S.spread - 1) / 3);
        const visLanes = S.spread > 1.2 ? 4 : 1;
        for (let j = 0; j < (visLanes === 1 ? 1 : nL); j++) V.line(g, lx0, laneY(j), lx1, laneY(j), { color: V.alpha(C.a, 0.55), lw: 2 });
        V.box(g, pad, ly - 17, lx0 - pad - 6, 34, { r: 8, fill: V.alpha(C.e, 0.16), stroke: C.e, label: 'embed', size: 11 });
        V.box(g, lx1 + 6, ly - 17, w - pad - lx1 - 6, 34, { r: 8, fill: V.alpha(C.c, 0.16), stroke: C.c, label: 'head', size: 11 });
        parts.forEach((q) => {
          const u = (q.u + t * 0.07 * q.sp) % 1;
          const x = V.lerp(lx0, lx1, u), y = visLanes === 1 ? laneY(0) : laneY(q.lane);
          V.dot(g, x, y, 2.3, C.a);
        });
        // caixas
        const by = ly + 44, bh = narrow ? 150 : 176;
        const bw = (w - 2 * pad - 16) / 2;
        const mxB = { x: pad, y: by, w: bw, h: bh }, ffB = { x: pad + bw + 16, y: by, w: bw, h: bh };
        const arrowPair = (B, col, ph) => {
          const xr = B.x + B.w * 0.3, xw = B.x + B.w * 0.7;
          V.arrow(g, xr, ly + 12, xr, B.y - 2, { color: V.alpha(col, 0.8), lw: 1.5, head: 6, dash: [3, 3] });
          V.arrow(g, xw, B.y - 2, xw, ly + 12, { color: V.alpha(col, 0.8), lw: 1.5, head: 6, dash: [3, 3] });
          const f = (t * 0.6 + ph) % 1;
          V.dot(g, xr, V.lerp(ly + 12, B.y - 2, f), 2.6, col);
          V.dot(g, xw, V.lerp(B.y - 2, ly + 12, f), 2.6, col);
          V.text(g, 'reads', xr - 5, (ly + B.y) / 2 + 4, { size: 10.5, align: 'right', color: C.muted });
          V.text(g, 'writes', xw + 5, (ly + B.y) / 2 + 4, { size: 10.5, color: C.muted });
        };
        arrowPair(mxB, C.b, 0); arrowPair(ffB, C.d, 0.5);
        V.box(g, mxB.x, mxB.y, mxB.w, mxB.h, { r: 12, fill: V.alpha(C.b, 0.07), stroke: V.alpha(C.b, 0.5) });
        V.box(g, ffB.x, ffB.y, ffB.w, ffB.h, { r: 12, fill: V.alpha(C.d, 0.06), stroke: V.alpha(C.d, 0.5) });
        V.text(g, 'MIXER · 4 consecutive layers', mxB.x + 10, mxB.y + 17, { size: 10.5, font: 'mono', color: C.muted, maxW: mxB.w - 20 });
        V.text(g, 'FFN', ffB.x + 10, ffB.y + 17, { size: 10.5, font: 'mono', color: C.muted });
        // padrão de mixers
        const cur = Math.floor(t * 1.2) % 4;
        const tw = (mxB.w - 20 - 9) / 4, th = narrow ? 48 : 60, ty = mxB.y + 28;
        p.pat.forEach((k, j) => {
          const col = k === 'L' ? C.a : k === 'C128' ? C.c : k === 'W' ? C.d : C.b;
          const on = j === cur;
          const tx = mxB.x + 10 + j * (tw + 3);
          V.box(g, tx, ty - (on ? 3 : 0), tw, th, { r: 7, fill: V.alpha(col, on ? 0.5 : 0.22), stroke: on ? col : V.alpha(col, 0.5), lw: on ? 2 : 1 });
          V.text(g, p.patL[j], tx + tw / 2, ty + th / 2 + 4 - (on ? 3 : 0), { size: tw < 44 ? 9.5 : 11, font: 'mono', align: 'center', color: C.ink, maxW: tw - 4 });
          if (on) V.glow(g, tx + tw / 2, ty + th + 6, 10, col, 0.8);
        });
        V.text(g, p.mixLabel, mxB.x + 10, ty + th + 26, { size: narrow ? 11.5 : 13, weight: 600, color: C.ink, maxW: mxB.w - 20 });
        V.text(g, p.mixSub, mxB.x + 10, ty + th + 44, { size: 11.5, color: C.muted, maxW: mxB.w - 20 });
        // FFN
        if (!p.E) {
          const bx = ffB.x + 12, bw2 = ffB.w - 24, bh2 = narrow ? 44 : 56;
          const pulse = 0.35 + 0.25 * Math.sin(t * 3);
          V.box(g, bx, ty, bw2, bh2, { r: 8, fill: V.alpha(C.e, pulse), stroke: C.e, label: 'all weights, every token', size: 11 });
        } else {
          const cols = p.E === 512 ? 32 : p.E === 288 ? 24 : 16;
          const rows = Math.ceil(p.E / cols);
          const gw = ffB.w - 24, gh = th + 16;
          const cs = Math.min(gw / cols, gh / rows);
          const gx = ffB.x + 12, gy = ty;
          const lit = new Set(S.choice);
          for (let e = 0; e < p.E; e++) {
            const r = Math.floor(e / cols), c = e % cols;
            g.fillStyle = lit.has(e) ? C.d : V.alpha(C.ink, 0.13);
            g.fillRect(gx + c * cs + 0.5, gy + r * cs + 0.5, Math.max(1, cs - 1), Math.max(1, cs - 1));
          }
          const shy = gy + rows * cs + 4;
          V.box(g, gx, shy, rows * cs > 0 ? Math.min(gw, cols * cs) : gw, 6, { r: 3, fill: V.alpha(C.e, 0.35 + 0.35 * (0.5 + 0.5 * Math.sin(t * 4))) });
        }
        const fly = p.E ? ty + th + 44 : ty + th + 26;
        V.text(g, p.ffnLabel, ffB.x + 10, fly, { size: narrow ? 11.5 : 13, weight: 600, color: C.ink, maxW: ffB.w - 20 });
        V.text(g, p.ffnSub, ffB.x + 10, fly + 18, { size: 11.5, color: C.muted, maxW: ffB.w - 20 });
        // acessórios
        let ex = pad, ey = by + bh + 22;
        V.text(g, 'ACCESSORIES', pad, ey, { size: 10.5, font: 'mono', color: C.muted });
        ey += 10;
        p.extras.forEach((s) => {
          const ww = textW(g, s, 11.5) + 20;
          if (ex + ww > w - pad) { ex = pad; ey += 28; }
          V.box(g, ex, ey, ww, 22, { r: 11, fill: V.alpha(C.c, 0.12), stroke: V.alpha(C.c, 0.6), label: s, size: 11.5 });
          ex += ww + 8;
        });
        fitTo(fig, ey + 22 + 16);
      });
    })();

    /* ================================================================ FIG 2: mapa camada a camada */
    const INFO = {
      glm: {
        kda: { t: 'KDA · Kimi Delta Attention (linear)', rows: [['q_proj / k_proj / v_proj', '[8192, 4096] each'], ['q/k/v_conv1d (causal, 4 taps)', '[8192, 1, 4]'], ['b_proj (β, write strength)', '[64, 4096]'], ['f_a → f_b (per-channel forgetting)', '[128, 4096] → [8192, 128]'], ['g_a → g_b (output gate)', '[128, 4096] → [8192, 128]'], ['A_log / dt_bias', '[64] / [8192] FP32'], ['o_proj', '[4096, 8192]']], facts: ['137,732,288 parameters per layer', 'State per request, per rank (TP=2): [32 heads, 128, 128] FP32 = 2,097,152 B + conv 245,760 B', 'KV that grows with the context: 0 B/token'] },
        dsa: { t: 'NoPE MLA + DSA indexer (sparse, top-2048)', rows: [['q_a_proj → q_b_proj', '[1536, 4096] → [16384, 1536]'], ['kv_a_proj_with_mqa (latent)', '[512, 4096]'], ['kv_b_proj (expands K/V per head)', '[32768, 512]'], ['o_proj', '[4096, 16384]'], ['indexer.wq_b / wk', '[4096, 1536] / [128, 4096]'], ['indexer.weights_proj', '[32, 4096]'], ['kpool_compress_gate / ape', '[128, 4096] / [4, 128]']], facts: ['MLA 117,442,560 + indexer 7,471,872 = 124,914,432 parameters', 'KV: 656 B/token (fp8_ds_mla: 512 FP8 + 16 B of scales + 128 B of RoPE zeros) + 33 B from the indexer', 'Top-k over 512 pools of 4 tokens (= 2048) + tail'] },
        dense: { t: 'Dense SwiGLU MLP', rows: [['gate_proj / up_proj', '[12288, 4096]'], ['down_proj', '[4096, 12288]']], facts: ['150,994,944 parameters; every token uses all of it'] },
        moe: { t: 'MoE: 288 experts, top-8 + 1 shared', rows: [['gate (router) / e_score_correction_bias', '[288, 4096] / [288] FP32'], ['shared gate/up · down', '[2048, 4096] · [4096, 2048]'], ['288 × expert gate/up · down', '[2048, 4096] · [4096, 2048]']], facts: ['288 × 25,165,824 = 7.25 B parameters per layer; 8 active = 201 M', 'Experts in EXL3 4 bpw (trellis + Hadamard): 12 MiB each, 6 MiB per rank', 'Sigmoid + noaux_tc + ×2.5; router in FP32'] },
        mhc: 'mHC: hc_attn_fn / hc_ffn_fn [24, 16384] (4 pre + 4 post + 16 comb), 786,486 parameters per layer',
      },
      qwen: {
        gdn: { t: 'Gated DeltaNet (linear)', rows: [['in_proj_qkv (Q 2048 + K 2048 + V 6144)', '[10240, 2560]'], ['in_proj_z (σ output gate)', '[6144, 2560]'], ['in_proj_a (decay α) / in_proj_b (β)', '[48, 2560] each'], ['conv1d (causal, 4 taps)', '[10240, 1, 4]'], ['A_log / dt_bias', '[48]'], ['out_proj', '[2560, 6144]']], facts: ['~57.96 M parameters per layer', 'State per request, per GPU (TP=2): 24 V heads × 128 × 128 = 1.5 MiB FP32 (0.75 MiB in BF16, as served)', 'KV that grows: 0 B/token'] },
        qsa: { t: 'QSA · Qwen Sparse Attention (sparse GQA)', rows: [['q_proj (query + gate)', '[12288, 2560]'], ['k_proj / v_proj', '[512, 2560] each'], ['o_proj', '[2560, 6144]'], ['q_norm / k_norm', '[256]'], ['indexer.index_qk_proj ((4+1) × 128)', '[640, 2560]']], facts: ['~49.8 M + indexer ~1.7 M parameters', 'KV: 2 heads × 256 × (K + V) = 1024 B/token in FP8 → 512 B per GPU', 'Indexer: micro-blocks of 4, top-512 blocks = 2048 tokens + tail'] },
        moe: { t: 'MoE: 512 experts, top-10 + 1 shared', rows: [['gate (router)', '[512, 2560]'], ['shared gate/up · down', '[640, 2560] · [2560, 640]'], ['shared_expert_gate (scalar σ)', '[1, 2560]'], ['512 × expert gate/up (NVFP4)', '[640, 1280] U8 → 640 × 2560'], ['512 × expert down (NVFP4)', '[2560, 320] U8 → 2560 × 640']], facts: ['4.915 M parameters per expert; 10 active per token', 'NVFP4 W4A4 with an FP8 scale every 16: ~2.77 MB per expert', 'EP: 256 whole experts per GPU'] },
        mhc: 'Gated Residual (×2 per layer): hc_norm [10240], input_mix down [320, 10240] / up [10240, 320], block_inject [4, 10240]',
        ple: 'N-gram (PLE) in this layer: 128 shards [2,500,012, 160] FP8 = 320,001,536 rows (51.2 B params, 47.68 GiB) + key_proj [10240, 2560], value_proj [2560, 2560], dilated conv1d [10240, 1, 4]',
      },
      ds: {
        base: [['wq_a (query latent)', '4096 → 1024'], ['wq_b (64 × 512)', '1024 → 32768'], ['wkv (K = V, one vector)', '4096 → 512'], ['attn_sink', '[64] FP32'], ['wo_a (8 groups)', '[8192, 4096]: 8 × (4096 → 1024)'], ['wo_b', '8192 → 4096']],
        swa: { t: 'SWA · sliding window of 128', extra: [], facts: ['~107 M parameters (attention core)', 'No compression: only the ring of 128 entries × 584 B = 74,752 B per sequence'] },
        c4a: { t: 'CSA / C4A · KV ÷4 + top-512 indexer + window', extra: [['C4 compressor (wkv, wgate, ape [4, 1024])', '4096 → 2 × 512'], ['indexer.wq_b / weights_proj', '1024 → 64 × 128 / 4096 → 64'], ['indexer compressor', '4096 → 2 × 128']], facts: ['~107 M + 19.1 M (compressor + indexer)', 'KV: 584 / 4 = 146 B/token + indexer 132 / 4 = 33 B/token', 'Reads 128 from the window + 512 compressed entries per query'] },
        c128a: { t: 'HCA / C128A · KV ÷128, read in full + window', extra: [['C128 compressor (wkv, wgate, ape [128, 512])', '4096 → 512']], facts: ['~107 M + 4.3 M (compressor)', 'KV: 584 / 128 ≈ 4.6 B/token', 'At 1M, reads 128 + 8,192 entries per query'] },
        moe: { t: 'MoE: 256 experts, top-6 + 1 shared', rows: [['gate + e_score_correction_bias', '4096 → 256'], ['256 × expert w1/w3 · w2 (MXFP4)', '4096 → 2048 · 2048 → 4096'], ['shared expert (FP8)', '4096 → 2048 → 4096']], facts: ['25.17 M parameters per expert; 6 routed + 1 shared active', 'FP4 E2M1 + E8M0 scale every 32: 13.4 MB per expert (6.7 MB per rank)', 'sqrtsoftplus + noaux_tc, ×1.5, no groups'] },
        hash: { t: 'MoE with hash routing (layers 0–2)', rows: [['tid2eid (fixed table)', '[129280, 6] int32'], ['256 × expert (MXFP4)', '4096 → 2048 → 4096'], ['shared expert', '4096 → 2048 → 4096']], facts: ['The 6 experts come from tid2eid[token_id]; the weights still come from the gate', 'Requires raw input_ids in the forward pass (multimodal included)', 'Image tokens use bias_vl, not the hash (#175)'] },
        mhc: 'mHC: hc_*_fn [24, 16384] FP32 (≈ 0.39 M per sub-block), 20-iteration Sinkhorn',
      },
    };
    // matrizes principais de cada tipo de camada: [nome, linhas (saída), colunas (entrada)]
    const MATS = {
      glm: { kda: [['q/k/v_proj (×3)', 8192, 4096], ['f_b (decay)', 8192, 128], ['o_proj', 4096, 8192]], dsa: [['q_b_proj', 16384, 1536], ['kv_a (latent)', 512, 4096], ['kv_b', 32768, 512], ['o_proj', 4096, 16384]], dense: [['gate/up (×2)', 12288, 4096]], moe: [['expert gate/up', 2048, 4096, 1]] },
      qwen: { gdn: [['in_proj_qkv', 10240, 2560], ['in_proj_z (σ gate)', 6144, 2560], ['out_proj', 2560, 6144]], qsa: [['q_proj (q + gate)', 12288, 2560], ['k_proj / v_proj', 512, 2560], ['o_proj', 2560, 6144]], moe: [['expert gate/up', 640, 2560, 1]] },
      ds: { att: [['wq_b', 32768, 1024], ['wkv (K = V)', 512, 4096], ['wo_a (8 groups)', 8192, 4096], ['wo_b', 4096, 8192]], moe: [['expert w1/w3', 2048, 4096, 1]], hash: [['expert w1/w3', 2048, 4096, 1]] },
    };
    const FMTS = { glm: [['BF16', 2], ['EXL3 4 bpw', 0.5]], qwen: [['BF16', 2], ['NVFP4', 0.5625]], ds: [['FP8', 1], ['MXFP4', 0.53125]] };
    const FMTNOTE = {
      glm: 'GLM: non-routed weights in BF16 in the checkpoint (the recipe re-quantizes them to FP8 at load time, GLM53_DENSE_FP8=all, half the bytes); experts in EXL3 4 bpw, 12 MiB per expert (3 matrices).',
      qwen: 'Qwen: dense weights in BF16; experts in NVFP4 (4 bits + FP8 scale every 16 = 4.5 bits), ~2.77 MB per expert.',
      ds: 'DeepSeek: attention projections in FP8 (128×128 blocks); experts in MXFP4 (4 bits + E8M0 scale every 32 = 4.25 bits), 13.4 MB per expert.',
    };
    function matsFor(m, l) {
      const T = MATS[m], F = FMTS[m];
      const mix = m === 'ds' ? T.att : T[l.mix];
      const ffn = T[l.ffn] || [];
      return mix.concat(ffn).map(([n, R, K, e]) => ({ n, R, K, exp: !!e, fmt: F[e ? 1 : 0][0], bpp: F[e ? 1 : 0][1] }));
    }
    const renderList = (rows) => h('ul', { class: 't18-tlist' }, rows.map(([a, b]) => h('li', null, h('span', null, a), h('code', null, b))));
    function panelFor(m, l) {
      const C = V.C();
      const I = INFO[m];
      const chips = [];
      const hdr = l.extra ? (l.extra === 'mtp' ? `Layer ${l.i} · MTP` : `DSpark stage ${l.stage} (virtual layer ${l.i})`) : `Layer ${l.i}`;
      let mixT, mixRows, mixFacts;
      if (m === 'ds') { const X = I[l.mix]; mixT = X.t; mixRows = I.base.concat(X.extra); mixFacts = X.facts; } else { const X = I[l.mix]; mixT = X.t; mixRows = X.rows; mixFacts = X.facts; }
      const F = I[l.ffn];
      const roles = [];
      if (m === 'glm') {
        if (l.tap) roles.push(`DFlash2 reads this layer's output (hook ${l.i + 1}); the overlay materializes hc_post and contracts the 4 streams to 4096.`);
        if (l.extra) roles.push('MTP layer: MLA + indexer + MoE, with eh_proj [4096, 8192]; no mHC. 7.43 B parameters (including 288 experts). Used with SPEC_METHOD=mtp (k = 2) and edited by ABLIT ("mtp-edit").');
        else roles.push(l.role === 'edit' ? 'ABLIT: "edit" layer (15–45): o_proj can be transplanted or orthogonalized.' : 'ABLIT: "safety-anchor" (0–14), kept intact.');
        if (l.mix === 'dsa' && !l.extra) roles.push('Joins the MLA KV group (3,584-token block) + indexer cache.');
        if (l.mix === 'kda') roles.push('Joins a Mamba-style state group (<span class="la-term" data-term="mamba-spec">MambaSpec</span>, align mode).');
      } else if (m === 'qwen') {
        if (l.ple) roles.push(INFO.qwen.ple);
        if (l.extra) roles.push('MTP layer: mtp.layers.0 in the checkpoint, mtp.layers.48 in vLLM; FP8 experts in 128×128 blocks; applied 3× as the drafter. It is the 13th layer with KV.');
        if (l.role === 'edit') roles.push('ABLIT=1 checkpoint: the o_proj of this QSA layer is the only tensor changed (12 layers).');
        if (l.mix === 'gdn') roles.push('Mamba-style state: defines the page and the 1,664-token attention block (BF16 SSM).');
      } else {
        if (l.tap) roles.push('Feeds DSpark: the average of the 4 mHC streams of this layer goes into main_proj (3 × 4096 → 4096).');
        if (l.extra) roles.push('DSpark stage (mtp.' + l.stage + '): 128-token SWA over the target features + non-causal [anchor + 4 noise] block + full MoE with 256 experts; ~6.58 B per stage.' + (l.stage === 0 ? ' Stage 0 has main_proj 12288 → 4096.' : ''));
        if (l.role === 'edit') roles.push('Default range of the runtime ablation (DSV4_ABLATE_LAYERS = 10–42, off by default).');
        if (!l.extra) roles.push('TP=2: ends with an all-reduce of wo_b and of the MoE (43 + 43 per step).');
      }
      const col = C[m];
      const el = h('div', null,
        h('h5', null, hdr, ' · ', MIXNAME[l.mix], ' + ', l.ffn === 'dense' ? 'dense MLP' : l.ffn === 'hash' ? 'MoE (hash)' : 'MoE'),
        h('span', { class: 'chip', style: { borderColor: col } }, NAME[m]),
        h('span', { class: 'chip' }, 'KV/token: ' + (KVB[l.mix] ? nf(KVB[l.mix], KVB[l.mix] < 10 ? 1 : 0) + ' B' : '0 B')),
        h('span', { class: 'chip' }, INFO[m].mhc.split(':')[0]),
        h('div', { class: 't18-cols' },
          h('div', null, h('b', null, mixT), renderList(mixRows), h('ul', { class: 't18-facts' }, mixFacts.map((f) => h('li', null, f)))),
          h('div', null, h('b', null, F.t), renderList(F.rows), h('ul', { class: 't18-facts' }, F.facts.map((f) => h('li', null, f))))),
        h('ul', { class: 't18-facts', style: { marginTop: '10px' } }, [INFO[m].mhc].concat(roles).map((f) => h('li', { html: f }))));
      return el;
    }
    (function () {
      const fig = root.querySelector('#fig-mapa');
      const panel = fig.querySelector('#mapa-panel');
      const S = { m: 'glm', sel: 3, mode: 'tipo', hover: -1, anim: 0 };
      const defSel = { glm: 3, qwen: 1, ds: 2 };
      const upd = () => { panel.innerHTML = ''; panel.appendChild(panelFor(S.m, LAYERS[S.m][S.sel])); };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Model', options: MODEL_OPTS, value: 'glm', onChange: (v) => { S.m = v; S.sel = defSel[v]; upd(); } });
      UI.seg(ctl, { label: 'Color by', options: [{ v: 'tipo', label: 'type' }, { v: 'kv', label: 'KV/token' }, { v: 'role', label: 'role' }], value: 'tipo', onChange: (v) => { S.mode = v; } });
      UI.button(ctl, '◀ layer', () => { S.sel = (S.sel - 1 + LAYERS[S.m].length) % LAYERS[S.m].length; upd(); });
      UI.button(ctl, 'layer ▶', () => { S.sel = (S.sel + 1) % LAYERS[S.m].length; upd(); });
      upd();
      const O = { mh: 30, fh: 11, labelH: 16, markH: 12, minCell: 13, maxCell: 30 };
      const rngm = V.rng(7);
      const expPick = { t: -1, set: [] };
      V.fig(fig, (g, w, hh, t, dt, _S, H) => {
        const C = V.C();
        const L = LAYERS[S.m];
        const pad = 14;
        // título e contagens
        const cnt = {}; L.forEach((l) => { if (!l.extra) cnt[l.mix] = (cnt[l.mix] || 0) + 1; });
        V.text(g, NAME[S.m], pad, 22, { size: 15, font: 'display', weight: 600, color: C[S.m] });
        const cs = Object.keys(cnt).map((k) => `${cnt[k]} ${MIXNAME[k]}`).join(' + ') + (S.m === 'glm' ? ' + MTP' : S.m === 'qwen' ? ' + MTP' : ' + 3 DSpark');
        V.text(g, cs, w - pad, 22, { size: 11.5, font: 'mono', align: 'right', color: C.muted, maxW: w * 0.58 });
        const G = stripGeom(w, L, Object.assign({ y: 40 }, O));
        // pulso do "token"
        const nMain = L.filter((l) => !l.extra).length;
        const pk = Math.floor((t * 7) % nMain);
        const pp = G.pos[pk];
        V.glow(g, pp.x + G.cw / 2, pp.y + O.mh / 2, 16, C.ink, 0.35);
        // interação
        const pt = H.pointer;
        S.hover = pt.inside ? hitStrip(G, pt.x, pt.y, O) : -1;
        if (pt.clicked && S.hover >= 0) { S.sel = S.hover; upd(); }
        drawStrip(g, G, Object.assign({ mode: S.mode, labels: true, sel: S.sel, hover: S.hover }, O));
        // legenda
        let ly = 40 + G.height + 14, lx = pad;
        const leg = S.mode === 'tipo' ? [[C.a, 'linear (KDA/GDN)'], [C.b, 'sparse (MLA-DSA/QSA/C4A)'], [C.c, 'C128A'], [C.d, 'SWA'], [V.alpha(C.ink, 0.9), 'dense FFN'], [C.f, 'hash FFN'], [V.alpha(C.ink, 0.38), 'MoE FFN']]
          : S.mode === 'kv' ? Array.from(new Set(L.map((x) => KVB[x.mix]))).sort((a, b) => b - a).map((b) => [kvCol(C, b), b > 0 ? `${nf(b, b < 10 ? 1 : 0)} B/token (${L.filter((x) => KVB[x.mix] === b && !x.extra).map((x) => MIXNAME[x.mix])[0]})` : '0: fixed state or window only'])
            : S.m === 'glm' ? [[C.c, 'ABLIT: "edit" layers (15–45)'], [V.alpha(C.muted, 0.55), 'anchors (0–14)'], [C.d, '◆ read by DFlash2']]
              : S.m === 'qwen' ? [[C.c, 'ABLIT=1: o_proj of the 12 QSA'], [C.f, '★ N-gram (index 1, the 2nd layer)']]
                : [[C.c, 'runtime ablation (10–42, opt-in)'], [C.f, 'hash FFN (0–2)'], [C.d, '◆ read by DSpark']];
        leg.forEach(([c, s]) => {
          const ww = textW(g, s, 11) + 24;
          if (lx + ww > w - pad) { lx = pad; ly += 18; }
          V.box(g, lx, ly - 9, 11, 11, { r: 3, fill: c });
          V.text(g, s, lx + 16, ly + 1, { size: 11, color: C.muted });
          lx += ww;
        });
        // esquema do bloco selecionado
        const l = L[S.sel];
        const top = ly + 22, bh = w < 560 ? 236 : 212, bot = top + bh;
        V.box(g, pad, top, w - 2 * pad, bh, { r: 12, fill: V.alpha(C.panel, 0.6), stroke: V.alpha(C.muted, 0.25) });
        const title = (l.extra ? (l.extra === 'mtp' ? 'MTP' : 'DSpark ' + l.stage) : 'layer ' + l.i) + (w >= 560 ? ' · how the token flows through' : '');
        V.text(g, title, pad + 12, top + 18, { size: 11, font: 'mono', color: C.muted });
        const nl = (S.m === 'glm' && l.extra) ? 1 : 4;
        const laneY0 = top + 34;
        for (let j = 0; j < nl; j++) V.line(g, pad + 12, laneY0 + j * 6, w - pad - 12, laneY0 + j * 6, { color: V.alpha(C.a, 0.5), lw: 1.6 });
        for (let q = 0; q < 12; q++) { const u = (q / 12 + t * 0.09) % 1; V.dot(g, V.lerp(pad + 12, w - pad - 12, u), laneY0 + (q % nl) * 6, 2, C.a); }
        V.text(g, nl === 4 ? (S.m === 'qwen' ? 'Gated Residual · 4 × 2560' : 'mHC · 4 × 4096') : 'plain residual (MTP without mHC)', w - pad - 12, top + 18, { size: 10.5, font: 'mono', align: 'right', color: C.muted });
        const iy = laneY0 + nl * 6 + 14, ih = bot - iy - 10;
        const iw = (w - 2 * pad - 36) / 2;
        const mB = { x: pad + 12, y: iy, w: iw, h: ih }, fB2 = { x: pad + 24 + iw, y: iy, w: iw, h: ih };
        const mc = mixCol(C, l.mix);
        V.box(g, mB.x, mB.y, mB.w, mB.h, { r: 9, fill: V.alpha(mc, 0.08), stroke: V.alpha(mc, 0.6) });
        V.box(g, fB2.x, fB2.y, fB2.w, fB2.h, { r: 9, fill: V.alpha(C.d, 0.06), stroke: V.alpha(C.d, 0.5) });
        V.text(g, MIXNAME[l.mix], mB.x + 8, mB.y + 16, { size: 12, weight: 600, color: mc });
        const inner = { x: mB.x + 8, y: mB.y + 24, w: mB.w - 16, h: mB.h - 32 };
        if (l.mix === 'kda' || l.mix === 'gdn') {
          const tight = inner.w < 220;
          const n = 8, cs2 = tight ? Math.min(8, inner.w / n) : Math.min((inner.h - 14) / n, (inner.w * 0.5) / n);
          const mx0 = inner.x, my0 = inner.y;
          for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
            const v = Math.sin(i * 1.3 + j * 0.7 + t * 1.6) * Math.cos(j * 0.9 - t * 0.8 + i * 0.3);
            g.fillStyle = V.cmap(v, 'div'); g.fillRect(mx0 + j * cs2, my0 + i * cs2, cs2 - 1, cs2 - 1);
          }
          if (tight) {
            const ty2 = my0 + n * cs2 + 14;
            V.text(g, 'state S · 128 × 128 per head', inner.x, ty2, { size: 10.5, weight: 600, color: C.ink, maxW: inner.w });
            wrapText(g, 'decays · corrects · writes; fixed size, does not grow with L', inner.w, 10.5).slice(0, 3).forEach((s2, k) => V.text(g, s2, inner.x, ty2 + 14 + k * 13, { size: 10.5, color: k === 0 ? C.muted : C.a }));
          } else {
            const tx = mx0 + n * cs2 + 10;
            V.text(g, 'state S', tx, my0 + 12, { size: 11, weight: 600, color: C.ink });
            V.text(g, '128 × 128 per head', tx, my0 + 27, { size: 10.5, color: C.muted, maxW: inner.x + inner.w - tx });
            V.text(g, 'S ← decays · corrects · writes', tx, my0 + 42, { size: 10.5, color: C.muted, maxW: inner.x + inner.w - tx });
            V.text(g, 'fixed size: does not grow with L', tx, my0 + 57, { size: 10.5, color: C.a, maxW: inner.x + inner.w - tx });
          }
        } else {
          const n = Math.max(12, Math.min(40, Math.floor(inner.w / 8)));
          const cw2 = inner.w / n, cy = inner.y + 8, chh = Math.min(22, inner.h * 0.35);
          const ph = Math.floor(t * 1.2);
          const r2 = V.rng(ph * 31 + (l.i || 0));
          for (let j = 0; j < n; j++) {
            let on = false, col = mc;
            if (l.mix === 'swa') on = j >= n - 5;
            else if (l.mix === 'c128a') on = true;
            else on = j >= n - 2 || r2() < 0.28;
            if (l.mix === 'c4a' && j >= n - 4) col = C.d;
            if ((l.mix === 'dsa' || l.mix === 'qsa') && j >= n - 1) col = C.d;
            const wcell = l.mix === 'c128a' && j < n - 4 ? cw2 : cw2;
            g.fillStyle = on ? V.alpha(col, 0.85) : V.alpha(C.ink, 0.1);
            V.rr(g, inner.x + j * cw2 + 0.5, cy, Math.max(1, wcell - 1.5), chh, 2); g.fill();
          }
          if (l.mix === 'c128a') { for (let j = 0; j < n - 4; j += 6) { g.strokeStyle = V.alpha(C.ink, 0.35); g.strokeRect(inner.x + j * cw2 + 0.5, cy - 1, cw2 * 6 - 1.5, chh + 2); } }
          const desc = { dsa: 'top-2048 of the past (512 pools of 4) + tail', qsa: 'top-512 blocks of 4 (2048 tokens) + tail', c4a: '128 window + top-512 entries ÷4', c128a: '128 window + all entries ÷128', swa: 'only the last 128 tokens' }[l.mix];
          const dl = wrapText(g, desc, inner.w, 10.5).slice(0, 3);
          dl.forEach((s2, k) => V.text(g, s2, inner.x, cy + chh + 16 + k * 13, { size: 10.5, color: C.muted }));
          V.text(g, 'KV grows: ' + (KVB[l.mix] ? nf(KVB[l.mix], KVB[l.mix] < 10 ? 1 : 0) + ' B/token' : 'no (fixed window)'), inner.x, cy + chh + 18 + dl.length * 13, { size: 10.5, color: KVB[l.mix] ? C.c : C.a, maxW: inner.w });
        }
        // FFN
        V.text(g, l.ffn === 'dense' ? 'dense MLP' : l.ffn === 'hash' ? 'MoE · hash' : 'MoE', fB2.x + 8, fB2.y + 16, { size: 12, weight: 600, color: C.d });
        const fin = { x: fB2.x + 8, y: fB2.y + 24, w: fB2.w - 16, h: fB2.h - 30 };
        if (l.ffn === 'dense') {
          V.box(g, fin.x, fin.y + 4, fin.w, Math.min(26, fin.h - 20), { r: 6, fill: V.alpha(C.e, 0.35 + 0.2 * Math.sin(t * 3)), stroke: C.e, label: '12288 · all weights', size: 10.5 });
        } else {
          const E = S.m === 'glm' ? 288 : S.m === 'qwen' ? 512 : 256, k = S.m === 'glm' ? 8 : S.m === 'qwen' ? 10 : 6;
          const cols = E === 512 ? 32 : E === 288 ? 24 : 16, rows = Math.ceil(E / cols);
          const capL = wrapText(g, l.ffn === 'hash' ? `tid2eid[token] → ${k} experts` : `top-${k} of ${E} + 1 shared`, fin.w, 10.5).slice(0, 2);
          const csz = Math.min(fin.w / cols, (fin.h - 6 - capL.length * 13) / rows);
          const step = Math.floor(t * 1.1);
          if (step !== expPick.t) { expPick.t = step; const st = new Set(); if (l.ffn === 'hash') { const r3 = V.rng(1000 + (step % 5)); while (st.size < k) st.add(r3.int(E)); } else { while (st.size < k) st.add(rngm.int(E)); } expPick.set = st; }
          for (let e = 0; e < E; e++) { const r = Math.floor(e / cols), c = e % cols; g.fillStyle = expPick.set.has(e) ? (l.ffn === 'hash' ? C.f : C.d) : V.alpha(C.ink, 0.13); g.fillRect(fin.x + c * csz + 0.4, fin.y + r * csz + 0.4, Math.max(1, csz - 0.8), Math.max(1, csz - 0.8)); }
          capL.forEach((s2, q) => V.text(g, s2, fin.x, fin.y + rows * csz + 14 + q * 13, { size: 10.5, color: C.muted }));
        }
        // matrizes da camada, em proporção (lado ∝ √dimensão)
        const mats = matsFor(S.m, l);
        const narrowM = w < 560;
        const maxH = narrowM ? 66 : 88, cM = maxH / Math.sqrt(32768);
        let my = bot + 22;
        const head = 'MATRICES OF THIS LAYER · side ∝ √dimension · [output × input] · bytes in the served format';
        const hl = wrapText(g, head, w - 2 * pad, 10.5, 400, 'mono');
        hl.forEach((s2, q) => V.text(g, s2, pad, my + q * 14, { size: 10.5, font: 'mono', color: C.muted }));
        my += hl.length * 14 + 6;
        const cells = mats.map((mt) => {
          const rw = Math.max(2, cM * Math.sqrt(mt.K)), rh = Math.max(2, cM * Math.sqrt(mt.R));
          const l1 = mt.n, l2 = `[${nf(mt.R)} × ${nf(mt.K)}]`, l3 = `${fB(mt.R * mt.K * mt.bpp)} · ${mt.fmt}`;
          const lw2 = Math.max(textW(g, l1, 11, 'sans', 600), textW(g, l2, 10.5, 'mono'), textW(g, l3, 10.5, 'mono'));
          return { mt, rw, rh, l1, l2, l3, cw: Math.max(rw, lw2) };
        });
        const rowHM = maxH + 8 + 3 * 13 + 14;
        let cx = pad, cy = my;
        cells.forEach((c, k) => {
          if (cx > pad && cx + c.cw > w - pad) { cx = pad; cy += rowHM; }
          const rx = cx, ry = cy + maxH - c.rh;
          const col = c.mt.exp ? C.d : mc;
          // vetor de entrada (em cima) e de saída (à esquerda), pulsando
          const pu = 0.35 + 0.35 * Math.sin(t * 2.4 + k);
          g.fillStyle = V.alpha(col, pu); g.fillRect(rx, ry - 5, c.rw, 3);
          g.fillStyle = V.alpha(col, 0.2); g.fillRect(rx, ry, c.rw, c.rh);
          g.strokeStyle = V.alpha(col, 0.9); g.lineWidth = 1.2; g.strokeRect(rx + 0.5, ry + 0.5, Math.max(1, c.rw - 1), Math.max(1, c.rh - 1));
          // varredura: a entrada percorre as colunas, a saída acende de cima para baixo
          const u = (t * 0.4 + k * 0.23) % 1;
          g.fillStyle = V.alpha(col, 0.55); g.fillRect(rx + u * Math.max(0, c.rw - 2), ry, Math.max(1.5, c.rw * 0.05), c.rh);
          g.fillStyle = V.alpha(col, 0.35); g.fillRect(rx, ry, c.rw, c.rh * u);
          V.text(g, c.l1, rx, cy + maxH + 20, { size: 11, weight: 600, color: C.ink });
          V.text(g, c.l2, rx, cy + maxH + 33, { size: 10.5, font: 'mono', color: C.muted });
          V.text(g, c.l3, rx, cy + maxH + 46, { size: 10.5, font: 'mono', color: c.mt.exp ? C.d : C.ink });
          cx += c.cw + (narrowM ? 14 : 22);
        });
        let fy = cy + rowHM;
        const note = FMTNOTE[S.m];
        const nl2 = wrapText(g, note, w - 2 * pad, 10.5);
        nl2.forEach((s2, q) => V.text(g, s2, pad, fy + q * 13, { size: 10.5, color: C.muted }));
        fy += nl2.length * 13;
        fitTo(fig, fy + 10);
      });

    })();

    /* ================================================================ FIG 3: config.json guiado */
    const all = () => true, main = (l) => !l.extra, none = () => false;
    const CFG = {
      glm: [
        { c: 'identity' },
        { k: 'architectures', v: '["Glm5NextForConditionalGeneration"]', hl: all, d: 'Name of the class that the vLLM registry looks up: a multimodal wrapper marked as <code>IsHybrid</code> and <code>HasInnerState</code>. It is the signal that the engine needs a hybrid KV manager.' },
        { k: 'model_type', v: '"glm5_next" · text_config: "glm5_next_text"', hl: all, d: 'Picks the config class <code>Glm5NextTextConfig</code>, which translates the SGLang names used in the checkpoint. Without the alias for <code>num_experts_per_tok</code>, the class default would route 7 experts instead of 8.' },
        { c: 'shape' },
        { k: 'hidden_size', v: '4096', hl: all, d: 'Width of each of the 4 residual streams. It dictates the shapes: KDA q/k/v [8192, 4096], experts [2048, 4096]. The cooperative MoE kernels and <code>pack_profile.py</code> require exactly 4096.' },
        { k: 'num_hidden_layers', v: '45', hl: main, d: 'Decoder layers (0–44). The MTP is layer 45, outside this count; vLLM uses this number as the index of the first MTP layer, and DFlash2 numbers its layers starting at 45.' },
        { k: 'vocab_size / tie_word_embeddings', v: '154880 / false', hl: none, d: '154,856 tokens in use + padding up to a multiple of 128 (the DFlash2 mask id, 154856, falls in the padding). The lm_head has its own weights (+634 M) and is shared with the MTP and DFlash2. At TP=3 the vocabulary becomes 154,944.' },
        { k: 'max_position_embeddings', v: '1048576', hl: all, d: 'Declared context. Since the text path uses no RoPE, there is no sine table to scale (the config does not even have <code>rope_theta</code>). The recipe serves 850,000 by default.' },
        { c: 'mixers' },
        { k: 'layer_types', v: '34 × "linear_attention", 11 × "deepseek_sparse_attention"', hl: (l) => l.mix === 'dsa' && !l.extra, d: 'The 3:1 hybrid pattern of Kimi Linear. vLLM collapses the DSA layers into "attention" for hybrid accounting: 11 layers in the attention group and 34 in Mamba-style state groups.' },
        { k: 'linear_attn_config', v: '{num_heads: 64, head_dim: 128, short_conv_kernel_size: 4, gate_lower_bound: -5.0}', hl: (l) => l.mix === 'kda', d: 'KDA: 64 heads of 128, with a 128×128 state per head (32 heads per rank at TP=2) and a 4-tap causal conv on q, k and v. The forget gate lies in (−5, 0), so the factor α lies in (0.0067, 1): it never erases everything at once.' },
        { k: 'num_attention_heads / num_key_value_heads', v: '64 / 64', hl: (l) => l.mix === 'dsa', d: 'MLA query heads: 32 per rank at TP=2. 64 is not divisible by 3, so TP=3 forces 66 via --hf-overrides; 96 would collapse the logits, because one rank would end up with only fake KDA heads. The "64 KV heads" does not size the cache, which is a latent.' },
        { k: 'q_lora_rank / kv_lora_rank', v: '1536 / 512', hl: (l) => l.mix === 'dsa', d: 'The low-rank bottlenecks of MLA. The 512 latent is what goes into the KV cache (the SM120 kernel requires exactly 512); the 1536 query latent also feeds the indexer.' },
        { k: 'qk_nope_head_dim / qk_rope_head_dim / v_head_dim', v: '256 / 0 / 256', hl: (l) => l.mix === 'dsa', d: 'Zero RoPE dimensions: NoPE MLA. This breaks the <code>fp8_ds_mla</code> kernel ("pe_dim must be 64"); the recipe pads 64 zeros, which is exact and costs +24% KV in those layers. Softmax scale: 256<sup>−1/2</sup> = 1/16.' },
        { k: 'mla_use_nope', v: 'true', hl: (l) => l.mix === 'dsa', d: 'Turns on NoPE mode (<code>rotary_emb = None</code>). One more field that is only read correctly thanks to an alias in the config class.' },
        { k: 'index_topk / index_n_heads / index_head_dim', v: '2048 / 32 / 128', hl: (l) => l.mix === 'dsa', d: 'DSA indexer: 32 heads of 128 score the past and only 2048 tokens enter the attention; up to 2048 tokens of context, sparse = dense. The SM12x kernel only exists for top-k ∈ {128, 512, 1024, 2048}.' },
        { k: 'index_kpool / index_kpool_always_select_tail', v: '4 / true', hl: (l) => l.mix === 'dsa', d: 'Every 4 indexer keys become 1 entry (33 B/token/layer). The top-k picks 512 pools (= 2048 tokens) plus the tail of up to 3 recent tokens. Since 2048 + 3 overflowed the kernel template, the recipe uses 511 pools + tail.' },
        { c: 'FFN' },
        { k: 'first_k_dense_replace', v: '3', hl: (l) => l.ffn === 'dense', d: 'Layers 0–2 have a dense MLP of width 12,288 (151 M parameters each) before the MoE kicks in.' },
        { k: 'n_routed_experts / num_experts_per_tok / n_shared_experts', v: '288 / 8 / 1', hl: (l) => l.ffn === 'moe', d: '288 experts of 25.2 M parameters per layer (7.25 B), top-8 + 1 shared. At TP=3, expert parallel gives 96 experts per rank.' },
        { k: 'moe_intermediate_size', v: '2048', hl: (l) => l.ffn === 'moe', d: 'Width of each expert: 1024 per rank at TP=2. It is not divisible by 3 and the EXL3 trellis is packed at width 2048, which is why TP=3 uses expert parallel instead of padding.' },
        { k: 'scoring_func / topk_method / routed_scaling_factor', v: '"sigmoid" / "noaux_tc" / 2.5', hl: (l) => l.ffn === 'moe', d: 'DeepSeek-V3-style routing: independent sigmoid scores, a balancing bias applied only to the selection (<code>e_score_correction_bias</code>, FP32), weights renormalized and multiplied by 2.5. Router in FP32 to avoid expert swaps caused by ties.' },
        { k: 'swiglu_limit', v: '10.0', hl: (l) => !l.extra || true, d: 'SwiGLU clamp (gate ≤ 10, up ∈ [−10, 10]) in dense MLPs, shared experts and routed experts. The EXL3 overlay has to replicate exactly the same value.' },
        { c: 'residual' },
        { k: 'mhc / hc_mult / hc_sinkhorn_iters', v: 'true / 4 / 20', hl: main, d: 'Residual with 4 streams of 4096 ([T, 4, 4096]) and 4×4 mixing projected by Sinkhorn (20 iterations) onto a doubly stochastic matrix. Any hidden-state "tap" has to materialize and contract the 4 streams.' },
        { c: 'extras' },
        { k: 'num_nextn_predict_layers', v: '1', hl: (l) => l.extra === 'mtp', d: '1 MTP layer (index 45: MLA + MoE, no mHC). Fallback drafter with SPEC_METHOD=mtp (k = 2), also edited by ABLIT.' },
        { k: 'dflash_config.target_layer_ids  (DFlash2 config)', v: '[5, 14, 24, 33, 42]', hl: (l) => !!l.tap, d: 'From the drafter config: the target layers whose outputs feed it (5 × 4096 → <code>fc [4096 × 20480]</code>). They are all KDA and show up in the log as 6, 15, 25, 34, 43.' },
      ],
      qwen: [
        { c: 'identity' },
        { k: 'architectures', v: '["Qwen4ExpForConditionalGeneration"]', hl: all, d: 'Multimodal model (vision + LM), an open preview of the Qwen4 architecture.' },
        { k: 'model_type', v: '"qwen4_exp" · text_config: "qwen4_exp_text"', hl: all, d: 'The registry maps it to <code>vllm/models/qwen4_exp/nvidia/…</code> in vLLM 0.30 and to <code>vllm/models/qwen3_8_flash_next/nvidia/…</code> in the day-0 image. Bind-mount patches have to get the path right.' },
        { c: 'shape' },
        { k: 'hidden_size', v: '2560', hl: all, d: 'Narrow for 125 B: the capacity is in the 512 experts and the N-gram. It is the K of almost every decode GEMM. With 4 branches, the residual state has 10,240 numbers per token.' },
        { k: 'num_hidden_layers', v: '48', hl: main, d: '36 GDN + 12 QSA. vLLM creates the MTP at <code>num_hidden_layers + i</code> = 48, hence the <code>mtp.layers.48</code> alias in the quantization files.' },
        { k: 'vocab_size / tie_word_embeddings', v: '248320 / false', hl: none, d: 'Separate embedding and lm_head, 1.18 GiB each in BF16. The lm_head is the most-read tensor in decode with MTP (1 + 3 reads per step), which motivated the reduced draft vocabulary.' },
        { k: 'max_position_embeddings', v: '262144', hl: all, d: 'Native context. For 1M: YaRN factor 4 via --hf-overrides nested in text_config + VLLM_ALLOW_LONG_MAX_MODEL_LEN=1, but 1M with YaRN has not yet been validated in the kit (<code>YARN_ENABLE</code> stays <code>false</code>). The recipe turns YaRN off below 262,144 ("no benefit and it costs accuracy").' },
        { c: 'mixers' },
        { k: 'full_attention_interval', v: '4', hl: (l) => l.mix === 'qsa', d: '1 attention layer every 4 (here sparse QSA); the other 3 are Gated DeltaNet. Only these layers store KV per token: 12 + the MTP = 13.' },
        { k: 'layer_types', v: '36 × "linear_attention", 12 × "full_attention"', hl: (l) => l.mix === 'qsa' && !l.extra, d: 'The same information spelled out: "full_attention" where (i+1) mod 4 = 0. transformers renames it to "indexed_attention" because they are QSA.' },
        { k: 'linear_num_key_heads / linear_num_value_heads', v: '16 / 48', hl: (l) => l.mix === 'gdn', d: 'GDN with 16 Q/K heads serving 48 V heads (each Q/K is repeated 3×). Each V head has its own 128×128 state: 24 per GPU at TP=2.' },
        { k: 'linear_key_head_dim / linear_value_head_dim / linear_conv_kernel_dim', v: '128 / 128 / 4', hl: (l) => l.mix === 'gdn', d: '128×128 state per V head; 4-tap causal conv (3 state columns + speculative tokens). The GDN decode CUDA kernels require head_k_dim = 128.' },
        { k: 'output_gate_type', v: '"sigmoid"', hl: (l) => l.mix === 'gdn', d: 'The GDN output is multiplied by σ(W<sub>z</sub>x) before out_proj: a gate bounded to (0, 1), which the technical report preferred over the SiLU of the original GDN. It costs an in_proj_z [6144, 2560] per layer.' },
        { k: 'mamba_ssm_dtype', v: '"float32"', hl: (l) => l.mix === 'gdn', d: 'dtype of the recurrent state. The recipe uses --mamba-ssm-cache-dtype bfloat16: half the bytes read and written per step, and an attention block of 1664 tokens instead of 3200.' },
        { k: 'num_attention_heads / num_key_value_heads / head_dim', v: '24 / 2 / 256', hl: (l) => l.mix === 'qsa', d: 'QSA is 12:1 GQA: the KV of 2 heads × 256 gives 1024 elements per token per layer; at TP=2, 1 KV head per GPU.' },
        { k: 'partial_rotary_factor / rope_theta / mrope_section', v: '0.25 / 10000000 / [11, 11, 10]', hl: (l) => l.mix === 'qsa', d: 'RoPE only on 64 of the 256 dimensions, with θ = 10⁷, split across time, height and width (interleaved mRoPE). The recipe nests only the YaRN part inside <code>text_config</code>; the official card passes the full dictionary, and the inventory points out the unverified risk that an override replaces the whole dict. As a precaution, repeat θ, the fraction and the mRoPE.' },
        { k: 'indexer_n_heads / indexer_compress_ratio / indexer_budget', v: '4 / 4 / 2048', hl: (l) => l.mix === 'qsa', d: 'QSA indexer: 4 heads and keys in micro-blocks of 4 (averaged). It selects 512 blocks = 2048 tokens, plus a tail of up to 3, for a width of 2051.' },
        { c: 'FFN' },
        { k: 'num_experts / num_experts_per_tok', v: '512 / 10', hl: (l) => l.ffn === 'moe', d: '512 experts per layer in all 48 (24,576 in the backbone), top-10 via renormalized softmax: 10/512 ≈ 2% active.' },
        { k: 'moe_intermediate_size / shared_expert_intermediate_size', v: '640 / 640', hl: (l) => l.ffn === 'moe', d: 'Tiny experts (4.9 M parameters). In pure TP, 640/2 = 320 is not a multiple of 128 (the FP8 blocks of the MTP), so EP is mandatory: 256 whole experts per GPU. The shared expert has a scalar sigmoid gate.' },
        { c: 'residual' },
        { k: 'hc_count / hc_lowrank', v: '4 / 320', hl: main, d: 'Gated Residual: 4 branches of 2560. The read uses a rank-320 elementwise gate, which replaces the pre-norm; the write uses 1 scalar per branch. A final "mixer" replaces the final RMSNorm. It is replicated on both GPUs.' },
        { c: 'extras' },
        { k: 'ple_layer_ids / ngram_size / heads_per_ngram', v: '[2] / 3 / 8', hl: (l) => !!l.ple, d: 'N-gram embedding in the 2nd layer (1-based count → layers.1): bigrams and trigrams, 8 hashes per order = 16 heads of 160, forming a 2560-dim vector injected into the 4 branches.' },
        { k: 'ngram_vocab_size_base / ple_embed_dim', v: '20000000 / 2560', hl: (l) => !!l.ple, d: '16 tables of prime size ≥ 20 M: 320,001,536 rows × 160 = 51.2 B parameters (47.7 GiB in FP8). The recipe has to declare ple_embedding_dtype via --hf-overrides for the FP8 table to load.' },
        { k: 'mtp_num_hidden_layers / mtp.layer_types', v: '1 / ["full_attention"]', hl: (l) => l.extra === 'mtp', d: '1 MTP layer with QSA (the 13th layer with KV) and 512 FP8 experts in 128×128 blocks, applied 3 times (MTP = 3).' },
      ],
      ds: [
        { c: 'identity' },
        { k: 'architectures / model_type', v: '["DeepseekV4ForCausalLM"] / "deepseek_v4"', hl: all, d: 'Same class for 0731 and Vision-Exp (vision is installed by a hotfix). With the dspark_* fields, vLLM builds the DSpark drafter instead of the classic MTP.' },
        { c: 'shape' },
        { k: 'hidden_size / num_hidden_layers', v: '4096 / 43', hl: main, d: '43 backbone layers (0–42); 43–45 are the DSpark stages. At TP=2 each step does 43 all-reduces of wo_b + 43 for the MoE.' },
        { k: 'num_attention_heads / head_dim', v: '64 / 512', hl: main, d: '64 query heads of 512. 64 is not divisible by 3: TP=3 pads to 72 (9 groups). The SM120 kernel requires local heads ∈ {16, 32, 64, 128}, and the 32 at TP=2 fit.' },
        { k: 'num_key_value_heads', v: '1', hl: main, d: 'MQA: a single 512-dim vector per token is K and V at the same time. It cannot be sliced by TP, so the KV is replicated on every rank.' },
        { k: 'qk_rope_head_dim', v: '64', hl: main, d: 'RoPE only on the last 64 dims: 448 "nope" in FP8 + 64 in BF16, plus scales = 584 B. Since V also carries the rotation, the output goes through inverse RoPE (position −i).' },
        { k: 'q_lora_rank / o_groups / o_lora_rank', v: '1024 / 8 / 1024', hl: main, d: 'Low-rank query (1024 latent, reused by the indexer) and grouped output projection: 8 groups of 8 heads × 512 = 4096 → 1024 each. It is this 8 × 512 contract that forces group padding at TP=3.' },
        { k: 'vocab_size', v: '129280', hl: none, d: 'The same vocabulary as V3. The logits (129,280 × tokens) go through an all-gather every step.' },
        { c: 'hybrid attention' },
        { k: 'sliding_window', v: '128', hl: all, d: 'Every layer (and the drafter) has a window of 128 raw tokens, which covers the compressed block that is still "open". Fixed cost of ~3.4 MB per sequence.' },
        { k: 'compress_ratios', v: '[0, 0, 4, 128, 4, 128, …, 128, 4, 0, 0, 0]', hl: (l) => l.mix !== 'swa', d: 'Per layer: 0 = window only, 4 = CSA/C4A, 128 = HCA/C128A. 46 entries (43 in the backbone + 3 for DSpark): 2 SWA + 21 C4A + 20 C128A.' },
        { k: 'index_topk / index_n_heads / index_head_dim', v: '512 / 64 / 128', hl: (l) => l.mix === 'c4a', d: 'Lightning indexer only in the C4A layers: it picks 512 compressed entries (≈ 2048 tokens), with simulated FP4 after Hadamard. In V3.2 it picked 2048 raw tokens.' },
        { k: 'rope_theta / compress_rope_theta', v: '10000 / 160000', hl: all, d: 'θ = 10⁴ in the windows and the drafter; θ = 1.6·10⁵ in the compressed layers.' },
        { k: 'rope_scaling', v: '{type: "yarn", factor: 16, original_max_position_embeddings: 65536}', hl: (l) => l.mix !== 'swa', d: '64K × 16 = 1,048,576. In the reference, YaRN applies only to the compressed layers; the vLLM in the image applies it to all of them (the DSPARK_ENABLE_ROPE_SWA_FIX fix is opt-in).' },
        { c: 'FFN' },
        { k: 'n_routed_experts / num_experts_per_tok / n_shared_experts', v: '256 / 6 / 1', hl: all, d: '256 SwiGLU experts 4096 → 2048 → 4096 per layer, top-6 + 1 shared: 7 active per token.' },
        { k: 'scoring_func / topk_method / routed_scaling_factor', v: '"sqrtsoftplus" / "noaux_tc" / 1.5', hl: (l) => l.ffn === 'moe', d: '√softplus(logit) score in FP32 (new in V4), auxiliary-loss-free bias applied only to the selection, no group routing, weights ×1.5 (it was 2.5 in V3).' },
        { k: 'num_hash_layers', v: '3', hl: (l) => l.ffn === 'hash', d: 'Layers 0–2 pick their 6 experts from a fixed table tid2eid[token_id], with no router GEMM for the selection. Consequence: the forward pass needs the raw input_ids, even on the multimodal path.' },
        { k: 'expert_dtype / quantization_config', v: '"fp4" / fp8 e4m3, blocks [128, 128], scale_fmt ue8m0', hl: all, d: 'Experts in FP4 E2M1 with an E8M0 scale every 32 (≈ MXFP4); the rest in FP8 with power-of-2 scales. On the GB10, the experts run as W4A16 via b12x.' },
        { c: 'residual' },
        { k: 'hc_mult / hc_sinkhorn_iters', v: '4 / 20', hl: main, d: 'mHC: 4 streams of 4096 (16,384 floats per token) with Sinkhorn mixing. The b12x mHC kernels only support hidden 4096.' },
        { k: 'rms_norm_eps', v: '1e-20 (Vision-Exp) · 1e-06 (0731)', hl: all, d: 'A real change in the Vision-Exp checkpoint, affecting every RMSNorm and the mHC rsqrt. The repository has not measured the effect.' },
        { c: 'extras' },
        { k: 'num_nextn_predict_layers', v: '3 (Vision-Exp) · 1 (0731)', hl: (l) => l.extra === 'dspark', d: 'In the Anemll image it becomes n_predict. With 3, the number of speculative tokens has to be a multiple of 3, which leads to k = 6 (the trained block is 5).' },
        { k: 'dspark_block_size / dspark_target_layer_ids / dspark_markov_rank', v: '5 / [40, 41, 42] / 256', hl: (l) => !!l.tap || l.extra === 'dspark', d: 'DSpark reads the average of the 4 streams of layers 40–42 (3 × 4096 → 4096), builds an [anchor + 4 noise] block and solves it in a single forward pass. A rank-256 Markov head stitches together the chosen sequence.' },
        { k: 'vision_n_layers / vision_max_n_token', v: '32 / 384', hl: none, d: '32-layer ViT + 3×3 Aligner, up to 384 tokens per image. Image tokens use the bias_vl bias and skip the hash.' },
      ],
    };
    (function () {
      const fig = root.querySelector('#fig-config');
      const box = fig.querySelector('#cfg-json'), exp = fig.querySelector('#cfg-exp');
      const S = { m: 'glm', sel: null };
      const firstField = (m) => CFG[m].find((f) => f.k && /layer_types|compress_ratios/.test(f.k)) || CFG[m].find((f) => f.k);
      function renderExp() {
        const f = S.sel; const L = LAYERS[S.m];
        const n = L.filter((l) => f.hl(l)).length;
        exp.innerHTML = '';
        exp.appendChild(h('div', null,
          h('div', { class: 'kk' }, f.k), h('span', { class: 'vv' }, f.v),
          h('div', { html: f.d }),
          h('div', { class: 'aff' }, n ? `Affects ${n} of ${L.length} layers on the map (extras included).` : 'Global field: not per layer (embedding, lm_head or vision tower).')));
      }
      function render() {
        box.innerHTML = '';
        box.appendChild(h('span', { class: 'cm' }, '// ' + NAME[S.m] + ' · config.json (excerpts)'));
        CFG[S.m].forEach((f) => {
          if (f.c) { box.appendChild(h('span', { class: 'cm' }, '// — ' + f.c + ' —')); return; }
          const b = h('button', { type: 'button', class: S.sel === f ? 'on' : '' }, h('span', { class: 'k' }, '"' + f.k + '"'), ': ', h('span', { class: 'v' }, f.v), ',');
          b.addEventListener('click', () => { S.sel = f; box.querySelectorAll('button').forEach((x) => x.classList.remove('on')); b.classList.add('on'); renderExp(); });
          box.appendChild(b);
        });
        renderExp();
      }
      S.sel = firstField('glm');
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Model', options: MODEL_OPTS, value: 'glm', onChange: (v) => { S.m = v; S.sel = firstField(v); render(); } });
      UI.button(ctl, 'next field ▶', () => { const fs = CFG[S.m].filter((f) => f.k); const i = fs.indexOf(S.sel); S.sel = fs[(i + 1) % fs.length]; render(); const on = box.querySelector('button.on'); if (on && on.scrollIntoView) { try { box.scrollTop = on.offsetTop - box.clientHeight / 2; } catch (e) { /* */ } } });
      render();
      const O = { mh: 20, fh: 8, labelH: 14, markH: 10, minCell: 12, maxCell: 26 };
      V.fig(fig, (g, w, hh, t) => {
        const C = V.C();
        const L = LAYERS[S.m];
        V.text(g, 'layers affected by "' + (S.sel ? S.sel.k : '') + '"', 14, 18, { size: 11.5, font: 'mono', color: C.muted, maxW: w - 28 });
        const G = stripGeom(w, L, Object.assign({ y: 34 }, O));
        const f = S.sel;
        const pulse = 0.6 + 0.4 * Math.sin(t * 3);
        drawStrip(g, G, Object.assign({ mode: 'tipo', labels: true, lit: (l) => f && f.hl(l) }, O));
        G.pos.forEach((p) => { if (f && f.hl(p.l)) V.glow(g, p.x + G.cw / 2, p.y + O.mh / 2, 10, mixCol(C, p.l.mix), 0.35 * pulse); });
        fitTo(fig, 34 + G.height + 6);
      });
    })();

    /* ================================================================ FIG 4: parâmetros */
    (function () {
      const fig = root.querySelector('#fig-params');
      const CATS = [
        { k: 'rout', l: 'routed experts', c: 'a' },
        { k: 'alw', l: 'always-on FFN (shared + dense)', c: 'd' },
        { k: 'mix', l: 'mixers (attention, linear, indexer)', c: 'b' },
        { k: 'res', l: 'wide residual + routers', c: 'gray' },
        { k: 'emb', l: 'embedding + lm_head (active: lm_head only)', c: 'f' },
        { k: 'ext', l: 'extras (N-gram, MTP, drafter, vision)', c: 'c' },
      ];
      const D = {
        // GLM: experts roteados incluem os 288 da camada MTP (43 × 288 × 25,17 M = 311,65 B); "extras" = resto da MTP (0,185 B) + visão (0,564 B)
        // ativos: convenção única para os três = lm_head sim, embedding não (consulta de uma linha)
        glm: { totS: '321.3 B', tot: { rout: 311.654, alw: 1.51, mix: 6.057, res: 0.085, emb: 1.268, ext: 0.749 }, act: { rout: 8.456, alw: 1.51, mix: 6.057, res: 0.085, emb: 0.634, ext: 0 }, totL: '321.3 B', actL: '~16.7 B (card: 18 B)', frac: '97% (includes the 288 in the MTP)' },
        qwen: { totS: '180.0 B (125.7 in the backbone)', tot: { rout: 120.8, alw: 0.236, mix: 2.705, res: 0.704, emb: 1.272, ext: 54.292 }, act: { rout: 2.36, alw: 0.236, mix: 2.705, res: 0.704, emb: 0.636, ext: 0.033 }, totL: '180.0 B stored (125.7 in the backbone)', actL: '~6.7 B (card: 6 B, without lm_head)', frac: '96% of the backbone' },
        ds: { totS: '284.3 B (+20.3 outside)', tot: { rout: 277.03, alw: 1.082, mix: 5.088, res: 0.079, emb: 1.06, ext: 20.27 }, act: { rout: 6.494, alw: 1.082, mix: 5.088, res: 0.079, emb: 0.53, ext: 0 }, totL: '284.3 B (+20.3 B of DSpark and vision)', actL: '~13.2 B (card: 13 B)', frac: '97%' },
      };
      const FL = [['routed experts', 16.9, 'a'], ['KDA (34 layers)', 11.6, 'b'], ['MLA + indexer', 3.3, 'c'], ['shared experts', 2.1, 'd'], ['dense MLP', 0.9, 'gray'], ['lm_head', 1.27, 'f']];
      const S = { mode: 'tot', a: 0, ta: 0, fl: 0, tfl: 0 };
      const cc = (C, k) => k === 'gray' ? V.alpha(C.ink, 0.5) : C[k];
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Show', options: [{ v: 'tot', label: 'total' }, { v: 'act', label: 'active per token' }, { v: 'flops', label: 'FLOPs (GLM)' }], value: 'tot', onChange: (v) => { S.mode = v; S.ta = v === 'act' ? 1 : v === 'tot' ? 0 : S.ta; S.tfl = v === 'flops' ? 1 : 0; } });
      V.fig(fig, (g, w, hh, t, dt) => {
        const C = V.C();
        S.a += (S.ta - S.a) * Math.min(1, dt * 3);
        S.fl += (S.tfl - S.fl) * Math.min(1, dt * 4);
        const pad = 14, narrow = w < 560;
        if (S.fl < 0.99) {
          g.globalAlpha = 1 - S.fl;
          const lw = narrow ? 64 : 120, x0 = pad + lw, x1 = w - pad - (narrow ? 8 : 20);
          const maxT = 340, maxA = 19;
          const e = V.ease.inOut(S.a);
          const scale = V.lerp(maxT, maxA, e);
          V.text(g, S.a < 0.5 ? 'TOTAL PARAMETERS (billions)' : (narrow ? 'ACTIVE PER TOKEN (B) · with lm_head' : 'ACTIVE PARAMETERS PER TEXT TOKEN (billions) · with lm_head, without embedding'), pad, 20, { size: 11, font: 'mono', color: C.muted, maxW: w - 2 * pad });
          // eixo
          const ticks = scale > 100 ? [0, 50, 100, 150, 200, 250, 300] : scale > 40 ? [0, 20, 40, 60, 80] : [0, 5, 10, 15];
          ticks.forEach((v) => { const x = x0 + (v / scale) * (x1 - x0); if (x > x1 + 1) return; V.line(g, x, 34, x, 34 + 3 * 74, { color: C.grid, lw: 1 }); V.text(g, String(v), x, 44 + 3 * 74, { size: 10, font: 'mono', align: 'center', color: C.muted }); });
          ['glm', 'qwen', 'ds'].forEach((m, r) => {
            const y = 44 + r * 74;
            V.text(g, SHORT[m], pad, y + 18, { size: narrow ? 12 : 14, weight: 600, color: C[m] });
            let x = x0;
            let tot = 0;
            CATS.forEach((cat) => {
              const v = V.lerp(D[m].tot[cat.k], D[m].act[cat.k], e);
              tot += v;
              const bw = (v / scale) * (x1 - x0);
              const vis = Math.max(0, Math.min(bw, x1 - x + 40));
              if (vis > 0.3) {
                g.fillStyle = V.alpha(cc(C, cat.c), 0.82); g.fillRect(x, y + 2, vis, 26);
                if (m === 'ds' && cat.k === 'ext') hatch(g, x, y + 2, vis, 26, V.alpha(C.bg, 0.55));
              }
              x += bw;
            });
            const lab = S.a < 0.5 ? (narrow ? D[m].totS : D[m].totL) : D[m].actL;
            const lwid = textW(g, lab, 11.5, 'mono');
            const inside = x + 6 + lwid <= x1;
            if (inside) V.text(g, lab, x + 6, y + 20, { size: 11.5, font: 'mono', color: C.ink });
            else V.text(g, lab, x1, y + 44, { size: 11.5, font: 'mono', align: 'right', color: C.ink, maxW: x1 - x0 });
            if (S.a < 0.5) V.text(g, 'experts: ' + D[m].frac, x0, inside ? y + 44 : y + 60, { size: 10.5, color: C.muted });
          });
          // legenda
          let lx = pad, ly = 44 + 3 * 74 + 26;
          CATS.forEach((cat) => { const s = cat.l; const ww = textW(g, s, 11) + 24; if (lx + ww > w - pad) { lx = pad; ly += 17; } V.box(g, lx, ly - 9, 11, 11, { r: 3, fill: cc(C, cat.c) }); V.text(g, s, lx + 16, ly + 1, { size: 11, color: C.muted }); lx += ww; });
          g.globalAlpha = 1;
          S.needBars = ly + 16;
        }
        if (S.fl > 0.01) {
          g.globalAlpha = S.fl;
          const cx = narrow ? w / 2 : w * 0.3, cy = narrow ? 120 : hh / 2 + 6, R = narrow ? 82 : Math.min(120, hh / 2 - 40);
          const total = FL.reduce((s, x) => s + x[1], 0);
          let a0 = -Math.PI / 2;
          const sweep = V.ease.out(Math.min(1, S.fl));
          FL.forEach(([l, v, c], i) => {
            const a1 = a0 + (v / total) * Math.PI * 2 * sweep;
            g.beginPath(); g.moveTo(cx, cy); g.arc(cx, cy, R, a0, a1); g.closePath(); g.fillStyle = V.alpha(cc(C, c), 0.85); g.fill();
            a0 = a1;
          });
          g.beginPath(); g.arc(cx, cy, R * 0.56, 0, Math.PI * 2); g.fillStyle = C.bg; g.fill();
          V.text(g, '~36', cx, cy + 2, { size: 28, font: 'display', weight: 700, align: 'center', color: C.ink });
          V.text(g, 'GFLOP/token', cx, cy + 20, { size: 11, align: 'center', color: C.muted });
          const tx = narrow ? pad : w * 0.56, ty0 = narrow ? cy + R + 26 : cy - 70;
          FL.forEach(([l, v, c], i) => {
            const yy = ty0 + i * (narrow ? 20 : 24);
            V.box(g, tx, yy - 10, 12, 12, { r: 3, fill: cc(C, c) });
            V.text(g, l, tx + 20, yy, { size: 12.5, color: C.ink });
            V.text(g, nf(v, v < 2 ? 2 : 1) + ' · ' + nf((v / total) * 100, 0) + '%', w - pad, yy, { size: 12, font: 'mono', align: 'right', color: C.muted });
          });
          const fy = narrow ? ty0 + FL.length * 20 + 6 : ty0 + FL.length * 24 + 10;
          V.text(g, 'GLM-5.3-Flash · repo estimate (docs/improve-prefill.md)', narrow ? pad : tx, fy, { size: 10.5, color: C.muted, maxW: w - 2 * pad });
          g.globalAlpha = 1;
          S.needFl = fy + 16;
        }
        fitTo(fig, Math.max(S.needBars || 0, narrow ? (S.needFl || 0) : 0, narrow ? 0 : 340));
      });
    })();

    /* ================================================================ FIG 5: experts acordados */
    (function () {
      const fig = root.querySelector('#fig-experts');
      const P = {
        glm: { E: 288, k: 8, L: 42, bytes: 12582912, t: 8, fmt: 'EXL3 4 bpw', cols: 24, spec: 'DFlash2 k = 7 → 8 tokens' },
        qwen: { E: 512, k: 10, L: 48, bytes: 2770000, t: 4, fmt: 'NVFP4 + scales', cols: 32, spec: 'MTP = 3 → 4 tokens' },
        ds: { E: 256, k: 6, L: 43, bytes: 13369344, t: 7, fmt: 'MXFP4 (4.25 bits)', cols: 16, spec: 'DSpark k = 6 → 7 tokens' },
      };
      const S = { m: 'glm', t: 8, pick: [], t0: 0, seed: 1 };
      const rng = V.rng(99);
      const sample = () => { const p = P[S.m]; S.pick = []; for (let j = 0; j < S.t; j++) { const st = new Set(); while (st.size < p.k) st.add(rng.int(p.E)); S.pick.push(Array.from(st)); } S.t0 = performance.now(); };
      sample();
      const ctl = UI.controls(fig);
      let sl;
      UI.seg(ctl, { label: 'Model', options: MODEL_OPTS, value: 'glm', onChange: (v) => { S.m = v; sl.set(P[v].t); } });
      sl = UI.slider(ctl, { label: 'Tokens per step', min: 1, max: 8, step: 1, value: 8, fmt: (v) => String(v), onInput: (v) => { S.t = v; sample(); } });
      UI.button(ctl, '↻ new step', sample, { primary: true });
      V.every(2600, sample);
      const TOKC = (C) => [C.a, C.b, C.c, C.d, C.e, C.f, C.ink, C.muted];
      V.fig(fig, (g, w, hh) => {
        const C = V.C(), p = P[S.m];
        const pad = 14, narrow = w < 620;
        const el = Math.min(1, (performance.now() - S.t0) / 700);
        const cols = p.cols, rows = Math.ceil(p.E / cols);
        const statW = narrow ? 0 : 250;
        const gx0 = pad + 40, gw = w - gx0 - pad - statW - (narrow ? 0 : 16);
        const gh = narrow ? 190 : hh - 60;
        const cs = Math.min(gw / cols, gh / rows);
        const gy0 = 34;
        V.text(g, `${NAME[S.m]} · ${p.E} experts per layer · top-${p.k}`, pad, 20, { size: 12, weight: 600, color: C[S.m], maxW: w - 2 * pad });
        // contagem
        const owner = new Map();
        S.pick.forEach((arr, j) => arr.forEach((e) => { if (!owner.has(e)) owner.set(e, []); owner.get(e).push(j); }));
        const tc = TOKC(C);
        // tokens
        S.pick.forEach((arr, j) => {
          const ty = gy0 + 8 + j * Math.min(26, (rows * cs - 10) / Math.max(1, S.t - 1 || 1));
          V.dot(g, pad + 12, ty, 7, tc[j % tc.length]);
          V.text(g, String(j + 1), pad + 12, ty + 4, { size: 9.5, font: 'mono', align: 'center', color: C.bg });
          if (el < 1 || S.t <= 4) {
            g.globalAlpha = (1 - el) * 0.5 + (S.t <= 4 ? 0.12 : 0);
            arr.forEach((e) => { const r = Math.floor(e / cols), c = e % cols; V.line(g, pad + 19, ty, gx0 + c * cs + cs / 2, gy0 + r * cs + cs / 2, { color: tc[j % tc.length], lw: 0.7 }); });
            g.globalAlpha = 1;
          }
        });
        for (let e = 0; e < p.E; e++) {
          const r = Math.floor(e / cols), c = e % cols;
          const x = gx0 + c * cs, y = gy0 + r * cs;
          const o = owner.get(e);
          if (o) { g.fillStyle = V.alpha(tc[o[0] % tc.length], 0.35 + 0.6 * el); V.rr(g, x + 0.6, y + 0.6, cs - 1.2, cs - 1.2, Math.min(3, cs / 4)); g.fill(); if (o.length > 1) { V.dot(g, x + cs / 2, y + cs / 2, Math.max(1.5, cs / 6), C.ink); } }
          else { g.fillStyle = V.alpha(C.ink, 0.09); V.rr(g, x + 0.6, y + 0.6, cs - 1.2, cs - 1.2, Math.min(3, cs / 4)); g.fill(); }
        }
        // estatísticas
        const dist = owner.size;
        const expd = p.E * (1 - Math.pow(1 - p.k / p.E, S.t));
        const perLayer = dist * p.bytes, perStep = perLayer * p.L;
        const lines = [
          ['distinct in this step', `${dist} of ${p.E}`],
          ['expected (formula)', nf(expd, 1)],
          ['maximum possible (k·t)', String(Math.min(p.E, p.k * S.t))],
          ['bytes per expert', `${nf(p.bytes / 1e6, 1)} MB · ${p.fmt}`],
          ['per layer', `${nf(perLayer / 1e6, 0)} MB`],
          [`× ${p.L} MoE layers`, `${nf(perStep / 1e9, 1)} GB per step`],
          ['≈ per rank (TP/EP = 2)', `${nf(perStep / 2e9, 1)} GB`],
        ];
        const sx = narrow ? pad : w - pad - statW, sy = narrow ? gy0 + rows * cs + 22 : gy0 + 6;
        if (narrow) {
          lines.forEach(([a, b], i) => { const col = i % 2, row = Math.floor(i / 2); const xx = pad + col * ((w - 2 * pad) / 2); const yy = sy + row * 34; V.text(g, a, xx, yy, { size: 10.5, color: C.muted, maxW: (w - 2 * pad) / 2 - 6 }); V.text(g, b, xx, yy + 15, { size: 12.5, font: 'mono', weight: 600, color: i === 0 ? C.d : i === 6 ? C.c : C.ink, maxW: (w - 2 * pad) / 2 - 6 }); });
          const specY = sy + Math.ceil(lines.length / 2) * 34 + 2;
          V.text(g, p.spec, pad, specY, { size: 10.5, color: C.muted, maxW: w - 2 * pad });
          // altura calculada com o layout real do modelo escolhido (a grade do DS é mais alta que a do Qwen)
          fitTo(fig, specY + 12);
        } else {
          lines.forEach(([a, b], i) => { const yy = sy + i * 38; V.text(g, a, sx, yy, { size: 11, color: C.muted }); V.text(g, b, sx, yy + 17, { size: 14, font: 'mono', weight: 600, color: i === 0 ? C.d : i === 6 ? C.c : C.ink, maxW: statW }); });
          V.text(g, p.spec, sx, sy + lines.length * 38 + 4, { size: 11, color: C.muted, maxW: statW });
          fitTo(fig, 360);
        }
      });
    })();

    /* ================================================================ FIG 6: linha de KV */
    (function () {
      const fig = root.querySelector('#fig-linha');
      const R = {
        glm: { title: 'fp8_ds_mla · 656 B per token, per MLA layer', segs: [['NoPE latent · 512 × FP8', 512, 'b'], ['4 FP32 scales', 16, 'd'], ['RoPE 64 × BF16 = zeros (pad)', 128, 'z']], idx: 'kpool = 4: per-channel softmax-weighted sum (+ 4-slot APE) → Hadamard → FP8', idxB: '1 FP8 key of 128 B + 4 B of scale every 4 tokens = 33 B/token', types: [['KDA (34)', 0, 'fixed state ≈ 2.3 MB/req'], ['MLA-DSA (11)', 689, '656 + 33 from the indexer']] },
        qwen: { title: 'QSA KV in FP8 · 512 B per token, per layer, per GPU', segs: [['K · 1 head × 256 × FP8', 256, 'b'], ['V · 1 head × 256 × FP8', 256, 'a']], idx: 'blocks of 4: average (AvgPool) → RMSNorm → RoPE at the start position of the block', idxB: '1 key of 128 × BF16 every 4 tokens ≈ 64 B/token (estimate)', types: [['GDN (36)', 0, 'fixed state 0.75 MiB/req (BF16)'], ['QSA (12 + MTP)', 512, 'K + V per GPU; +~64 from the indexer']] },
        ds: { title: 'nvfp4_ds_mla (= FP8) · 584 B per entry', segs: [['448 × FP8 "nope" (7 blocks of 64)', 448, 'b'], ['RoPE 64 × BF16', 128, 'c'], ['7 UE8M0 scales + 1 pad', 8, 'd']], idx: 'overlapping C4: per-channel softmax over 8 positions (+ APE) → FP4 after Hadamard', idxB: 'indexer K cache: 132 B per entry ÷ 4 = 33 B/token', types: [['SWA (2 + 3)', 0, 'ring of 128 × 584 B'], ['C4A (21)', 179, '584/4 + 33 from the indexer'], ['C128A (20)', 4.56, '584/128']] },
      };
      const S = { m: 'glm' };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Model', options: MODEL_OPTS, value: 'glm', onChange: (v) => { S.m = v; } });
      V.fig(fig, (g, w, hh, t) => {
        const C = V.C(), r = R[S.m], pad = 14;
        V.text(g, r.title, pad, 20, { size: 12.5, weight: 600, color: C[S.m], maxW: w - 2 * pad });
        // régua de bytes
        const total = r.segs.reduce((s, x) => s + x[1], 0);
        const nCells = total / 8;
        const cellW = 9, perRow = Math.max(8, Math.floor((w - 2 * pad) / cellW));
        const cyc = V.phase(t, 5).f;
        const fill = Math.min(1, cyc / 0.55);
        let k = 0;
        const y0 = 34;
        const segStart = [];
        r.segs.forEach(([l, b, c], si) => {
          segStart.push(k);
          for (let j = 0; j < b / 8; j++, k++) {
            const row = Math.floor(k / perRow), col = k % perRow;
            const x = pad + col * cellW, y = y0 + row * 11;
            const on = k / nCells <= fill;
            const colr = c === 'z' ? V.alpha(C.muted, 0.35) : C[c];
            g.fillStyle = on ? V.alpha(colr, c === 'z' ? 1 : 0.85) : V.alpha(C.ink, 0.07);
            g.fillRect(x, y, cellW - 2, 9);
            if (c === 'z' && on) { g.fillStyle = V.alpha(C.ink, 0.5); g.fillRect(x + 3, y + 4, 2, 1); }
          }
        });
        const rulerRows = Math.ceil(nCells / perRow);
        let ly = y0 + rulerRows * 11 + 14;
        let lx = pad;
        r.segs.forEach(([l, b, c]) => { const s = `${l}: ${b} B`; const ww = textW(g, s, 11.5) + 26; if (lx + ww > w - pad) { lx = pad; ly += 18; } V.box(g, lx, ly - 9, 11, 11, { r: 3, fill: c === 'z' ? V.alpha(C.muted, 0.45) : C[c] }); V.text(g, s, lx + 16, ly + 1, { size: 11.5, color: C.ink }); lx += ww; });
        V.text(g, 'each little square = 8 bytes', w - pad, ly + 18, { size: 10, align: 'right', color: C.muted });
        // indexador: 4 → 1
        const iy = ly + 38;
        V.text(g, 'INDEXER: 4 KEYS → 1', pad, iy, { size: 10.5, font: 'mono', color: C.muted });
        const ph = V.phase(t, 4).f;
        const merge = V.ease.inOut(V.clamp((ph - 0.25) / 0.45, 0, 1));
        const vw = Math.min(16 * 9, (w - 2 * pad) * 0.42), vc = 16, vcw = vw / vc;
        const bx = pad, by = iy + 12;
        for (let q = 0; q < 4; q++) {
          const yy = V.lerp(by + q * 13, by + 19, merge);
          g.globalAlpha = q === 0 ? 1 : 1 - merge * 0.85;
          for (let j = 0; j < vc; j++) { const v = Math.sin(j * 0.9 + q * 1.7 + 0.3); g.fillStyle = V.cmap(v * (1 - 0.3 * merge), 'div'); g.fillRect(bx + j * vcw, yy, vcw - 1, 10); }
          g.globalAlpha = 1;
          if (merge < 0.1) V.text(g, 't' + (q + 1), bx + vw + 6, yy + 9, { size: 10, font: 'mono', color: C.muted });
        }
        if (merge > 0.9) { V.text(g, '1 entry', bx + vw + 6, by + 28, { size: 10.5, font: 'mono', color: C.d }); }
        // o texto começa depois do rótulo mais largo da animação ("1 entrada"), sem invadi-lo
        const tx = bx + vw + 6 + textW(g, '1 entry', 10.5, 'mono') + 16;
        const lines1 = wrapText(g, r.idx, w - pad - tx, 11.5);
        const lines2 = wrapText(g, r.idxB, w - pad - tx, 11.5);
        if (w - pad - tx > 120) {
          lines1.forEach((s, i) => V.text(g, s, tx, by + 8 + i * 15, { size: 11.5, color: C.ink }));
          lines2.forEach((s, i) => V.text(g, s, tx, by + 8 + (lines1.length + i) * 15 + 4, { size: 11.5, color: C.muted }));
        }
        const nIdxLines = w - pad - tx > 120 ? lines1.length + lines2.length : 0;
        // custo amortizado por tipo
        let by2 = by + Math.max(78, nIdxLines * 15 + 28);
        if (w - pad - tx <= 120) { const L3 = wrapText(g, r.idx + ' · ' + r.idxB, w - 2 * pad, 11); L3.forEach((s, i) => V.text(g, s, pad, by2 + i * 14, { size: 11, color: C.muted })); by2 += L3.length * 14 + 10; }
        V.text(g, 'BYTES/TOKEN THAT GROW WITH CONTEXT, PER LAYER, PER RANK', pad, by2, { size: 10.5, font: 'mono', color: C.muted, maxW: w - 2 * pad });
        const lw = Math.min(130, (w - 2 * pad) * 0.34), bx0 = pad + lw, bx1 = w - pad - 60;
        r.types.forEach(([l, b, note], i) => {
          const yy = by2 + 14 + i * 34;
          V.text(g, l, pad, yy + 13, { size: 12, weight: 600, color: C.ink, maxW: lw - 6 });
          const bw = b > 0 ? Math.max(3, (b / 700) * (bx1 - bx0)) : 0;
          if (b > 0) V.box(g, bx0, yy + 2, bw, 16, { r: 4, fill: V.alpha(C[S.m], 0.8) });
          else { V.box(g, bx0, yy + 2, 16, 16, { r: 8, stroke: C.a, lw: 1.5 }); }
          V.text(g, b > 0 ? nf(b, b < 10 ? 1 : 0) + ' B' : '0 B', bx0 + Math.max(bw, 16) + 6, yy + 14, { size: 12, font: 'mono', color: C.ink });
          V.text(g, note, bx0, yy + 31, { size: 10.5, color: C.muted, maxW: w - pad - bx0 });
        });
        fitTo(fig, by2 + 14 + r.types.length * 34 + 22);
      });
    })();

    /* ================================================================ FIG 7: calculadora de KV */
    (function () {
      const fig = root.querySelector('#fig-kvcalc');
      const MiB = 1048576, GiB = 1073741824;
      const S = { v: 100, qfp8: true, ghost: false, gA: 0 };
      const Ls = () => Math.round(1024 * Math.pow(1024, S.v / 100));
      const M = () => ({
        glm: { grow: 7579, fixed: 34 * 2342912 + 5 * 2048 * 2048, fixL: 'KDA 76 MiB + DFlash2 window 20 MiB', ghost: 29520, ghostL: 'fp8_ds_mla MLA in all 45 layers', pool: 11 * GiB, poolL: 'pool 11 GiB' },
        qwen: { grow: S.qfp8 ? 6656 : 13312, fixed: 36 * 786432 + 36 * 61440, fixL: 'GDN 27 MiB (BF16) + conv ≈ 2 MiB', ghost: 48 * (S.qfp8 ? 512 : 1024), ghostL: '48 layers with KV (illustrative)', pool: 32.02 * GiB, poolL: 'pool 32.02 GiB' },
        ds: { grow: 3850, fixed: 3440000, fixL: 'SWA windows ≈ 3.4 MB', ghost: 25112, ghostL: 'no compression: 43 × 584 B', pool: 17.04 * GiB, poolL: 'pool 17.04 GiB' },
      });
      const ctl = UI.controls(fig);
      UI.slider(ctl, { label: 'Context', min: 0, max: 100, step: 1, value: 100, fmt: (v) => { const L = Math.round(1024 * Math.pow(1024, v / 100)); return L >= 1e6 ? nf(L / 1048576, 2) + 'M' : L >= 1e4 ? nf(L / 1024, 0) + 'K' : nf(L, 0); }, onInput: (v) => { S.v = v; } });
      UI.seg(ctl, { label: 'Qwen KV', options: [{ v: 1, label: 'FP8' }, { v: 0, label: 'BF16' }], value: 1, onChange: (v) => { S.qfp8 = !!v; } });
      UI.toggle(ctl, { label: 'compare with attention in every layer', value: false, onChange: (v) => { S.ghost = v; } });
      const ro = UI.readout(ctl, 'L (tokens)', '—');
      V.fig(fig, (g, w, hh, t, dt) => {
        const C = V.C(), pad = 14, narrow = w < 560;
        S.gA += ((S.ghost ? 1 : 0) - S.gA) * Math.min(1, dt * 4);
        const L = Ls(); ro.set(nf(L, 0));
        const D = M();
        const lw = narrow ? 64 : 104, x0 = pad + lw, x1 = w - pad - 4;
        const vals = ['glm', 'qwen', 'ds'].map((m) => { const d = D[m]; return { m, d, grow: L * d.grow, fixed: d.fixed, ghost: L * d.ghost + d.fixed }; });
        const maxV = Math.max(...vals.map((v) => Math.max(v.grow + v.fixed, v.d.pool * 1.05, S.gA > 0.01 ? v.ghost * S.gA + (v.grow + v.fixed) * (1 - S.gA) : 0))) * 1.08;
        if (narrow) { V.text(g, `ONE SEQUENCE OF ${nf(L, 0)} TOKENS`, pad, 20, { size: 11, font: 'mono', color: C.muted, maxW: w - 2 * pad }); V.text(g, 'MEMORY PER RANK', pad, 35, { size: 11, font: 'mono', color: C.muted }); }
        else V.text(g, `ONE SEQUENCE OF ${nf(L, 0)} TOKENS · MEMORY PER RANK`, pad, 20, { size: 11, font: 'mono', color: C.muted, maxW: w - 2 * pad });
        const X = (v) => x0 + (v / maxV) * (x1 - x0);
        let y = narrow ? 64 : 50;
        vals.forEach((v, i) => {
          V.text(g, SHORT[v.m], pad, y + 18, { size: narrow ? 12 : 14, weight: 600, color: C[v.m] });
          // fantasma
          if (S.gA > 0.01) { g.globalAlpha = S.gA; g.setLineDash([4, 4]); V.rr(g, x0, y, X(v.ghost) - x0, 28, 5); g.strokeStyle = V.alpha(C.ink, 0.6); g.lineWidth = 1.3; g.stroke(); g.setLineDash([]); g.globalAlpha = 1; }
          const xf = X(v.fixed);
          g.fillStyle = V.alpha(C[v.m], 0.3); g.fillRect(x0, y + 3, Math.max(1.5, xf - x0), 22); hatch(g, x0, y + 3, Math.max(1.5, xf - x0), 22, V.alpha(C[v.m], 0.8));
          const xg = X(v.fixed + v.grow);
          g.fillStyle = V.alpha(C[v.m], 0.85); g.fillRect(xf, y + 3, Math.max(0, xg - xf), 22);
          if (xg - xf > 6) { g.save(); g.beginPath(); g.rect(xf, y + 3, xg - xf, 22); g.clip(); for (let q = 0; q < 3; q++) { const u = ((t * 0.35 + q / 3) % 1); const sx = xf + u * (xg - xf); const gr = g.createLinearGradient(sx - 30, 0, sx + 30, 0); gr.addColorStop(0, V.alpha(C.bg, 0)); gr.addColorStop(0.5, V.alpha(C.bg, 0.35)); gr.addColorStop(1, V.alpha(C.bg, 0)); g.fillStyle = gr; g.fillRect(sx - 30, y + 3, 60, 22); } g.restore(); }
          // pool
          const xp = X(v.d.pool);
          V.line(g, xp, y - 4, xp, y + 32, { color: C.d, lw: 2 });
          V.text(g, v.d.poolL, xp - 4, y - 7, { size: 10, font: 'mono', align: 'right', color: C.d });
          const tot = v.grow + v.fixed;
          let ty = y + 44;
          if (narrow) {
            V.text(g, `${fB(tot)} per sequence`, x0, ty, { size: 12, font: 'mono', weight: 600, color: C.ink }); ty += 16;
            V.text(g, `${fB(v.grow)} grows + ${fB(v.fixed)} fixed`, x0, ty, { size: 11, font: 'mono', color: C.ink, maxW: x1 - x0 }); ty += 15;
          } else {
            V.text(g, `${fB(tot)}  =  ${fB(v.grow)} (grows) + ${fB(v.fixed)} (fixed)`, x0, ty, { size: 11.5, font: 'mono', color: C.ink, maxW: x1 - x0 }); ty += 16;
          }
          const fits = v.d.pool / tot;
          const sub = (S.gA > 0.5 ? `if it were ${v.d.ghostL}: ${fB(v.ghost)}` : `${v.d.fixL} · ≈ ${fits >= 10 ? nf(fits, 0) : nf(fits, 1)} of these fit in the pool (physical math)`);
          wrapText(g, sub, x1 - x0, 10.5).forEach((s2) => { V.text(g, s2, x0, ty, { size: 10.5, color: C.muted }); ty += 13; });
          if (v.m === 'qwen' && L > 262144) { V.text(g, '⚠ beyond 262,144 tokens: requires YaRN', x0, ty, { size: 10.5, color: C.c, maxW: x1 - x0 }); ty += 13; }
          y = ty + 26;
        });
        fitTo(fig, y - 8);
      });

    })();

    /* ================================================================ FIG 8: linhagem (metrô) */
    (function () {
      const fig = root.querySelector('#fig-linhagem');
      // lab: 'a' = rótulo acima da estação, 'b' = abaixo
      const ST = [
        { id: 'v2', row: 0, x: 0.06, n: 'DeepSeek-V2', s: 'MLA', lab: 'b' },
        { id: 'v3', row: 0, x: 0.24, n: 'V3', s: 'fine-grained MoE · MTP', lab: 'a' },
        { id: 'v32', row: 0, x: 0.43, n: 'V3.2-Exp', s: 'DSA', lab: 'b' },
        { id: 'v4', row: 0, x: 0.68, n: 'V4-Flash', s: 'CSA/HCA · mHC', lab: 'a' },
        { id: 'v4x', row: 0, x: 0.9, n: '0731 · Vision-Exp', s: 'DSpark', lab: 'b' },
        { id: 'mhc', row: 1, x: 0.56, n: 'mHC paper', s: 'DeepSeek', ext: true, lab: 'b' },
        { id: 'engram', row: 1, x: 0.34, n: 'Engram', s: 'DeepSeek', ext: true, lab: 'b' },
        { id: 'q3n', row: 2, x: 0.2, n: 'Qwen3-Next', s: 'GDN 3:1', lab: 'a' },
        { id: 'q35', row: 2, x: 0.46, n: 'Qwen3.5', s: 'multimodal', lab: 'b' },
        { id: 'q38', row: 2, x: 0.8, n: 'Qwen3.8-Flash', s: 'QSA · GR · N-gram', lab: 'a' },
        { id: 'kimi', row: 2.95, x: 0.42, n: 'Kimi Linear', s: 'KDA + MLA NoPE', ext: true, lab: 'b' },
        { id: 'dflash', row: 2.95, x: 0.68, n: 'DFlash', s: 'block drafting', ext: true, lab: 'b' },
        { id: 'g5', row: 4, x: 0.22, n: 'GLM-5', s: 'common ancestor', lab: 'a' },
        { id: 'g53', row: 4.7, x: 0.52, n: 'GLM-5.3 (sibling)', s: '753B (744B in InferenceX)', lab: 'b' },
        { id: 'g53f', row: 4, x: 0.84, n: 'GLM-5.3-Flash', s: 'KDA · mHC · kpool', lab: 'b' },
      ];
      const byId = {}; ST.forEach((s) => { byId[s.id] = s; });
      // segmentos das linhas de cada família; o GLM-5 é ancestral comum de dois ramos irmãos
      const SEGS = [['v2', 'v3'], ['v3', 'v32'], ['v32', 'v4'], ['v4', 'v4x'], ['q3n', 'q35'], ['q35', 'q38'], ['g5', 'g53f'], ['g5', 'g53']];
      const TR = [
        { a: 'v2', b: 'g53f', l: '512 latent (MLA)', st: [0, 5], lu: 0.62 },
        { a: 'v3', b: 'g53f', l: 'sigmoid + noaux_tc', st: [1, 5], lu: 0.4 },
        { a: 'v32', b: 'g53f', l: 'DSA', st: [2, 5], lu: 0.62 },
        { a: 'v32', b: 'q38', l: 'indexer', st: [2, 4], lu: 0.45 },
        { a: 'engram', b: 'q38', l: 'N-gram', st: [4], lu: 0.5 },
        { a: 'g5', b: 'q38', l: 'index reuse', st: [4], lu: 0.55 },
        { a: 'kimi', b: 'g53f', l: 'KDA 3:1', st: [5], lu: 0.5 },
        { a: 'mhc', b: 'g53f', l: 'mHC', st: [5], lu: 0.28 },
        { a: 'mhc', b: 'v4', l: 'mHC', st: [6], lu: 0.5 },
        { a: 'dflash', b: 'g53f', l: 'DFlash2', st: [7], lu: 0.5 },
        { a: 'dflash', b: 'v4x', l: 'context injection', st: [7], lu: 0.45 },
      ];
      const STEPS = [
        { t: 'MLA: the KV becomes a latent (DeepSeek-V2)', d: 'K and V for all heads are now reconstructed from a <b>512-dimension latent</b> per token, plus a shared 64-dim RoPE key: 576 numbers per layer in the cache, instead of ~41 thousand in a 128-head MHA. The MoE is still softmax, with 160 experts, top-6 and group-limited routing. The 512 latent will make it to GLM-5.3-Flash, but without the RoPE part.', on: ['v2'], seg: [] },
        { t: 'V3: fine-grained MoE, sigmoid and no auxiliary loss', d: '256 experts of 2048, top-8 + 1 shared; <b>sigmoid</b> score and bias-based balancing (<code>noaux_tc</code>), weights ×2.5; 1 MTP layer; FP8 weights in 128×128 blocks. GLM-5.3-Flash inherits this routing recipe almost field for field.', on: ['v3'], seg: [['v2', 'v3']] },
        { t: 'V3.2-Exp: the lightning indexer', d: '<b>DSA</b>: a cheap indexer (64 heads × 128, in FP8) scores the entire past, and only the top-2048 tokens enter the MLA. The idea becomes a family: kpool in GLM, QSA in Qwen, C4A in V4.', on: ['v32'], seg: [['v3', 'v32']] },
        { t: 'Qwen3-Next: 3 linear layers for every attention layer', d: '80 B total, 3 B active. There are 3 <b>Gated DeltaNet</b> layers (fixed state) for each Gated Attention layer; 512 experts top-10 + 1 shared; MTP. It is the skeleton that Qwen3.5 and 3.8 keep.', on: ['q3n'], seg: [] },
        { t: 'Qwen3.5 → Qwen3.8-Flash-Next', d: 'Version 3.5 makes the model natively multimodal and takes the vocabulary to 248,320. Version 3.8, a preview of Qwen4, swaps the attention for <b>QSA</b>, widens the residual into 4 branches (<b>Gated Residual</b>) and adds 51 B of <b>N-gram embedding</b>, the idea from DeepSeek\'s Engram. The reuse of indices across MTP steps, the report says, follows GLM-5.', on: ['q35', 'q38', 'engram', 'g5'], seg: [['q3n', 'q35'], ['q35', 'q38']] },
        { t: 'GLM-5 and two siblings: 5.3 and 5.3-Flash', d: 'Two branches come out of GLM-5. <b>GLM-5.3</b> is the big sibling (<code>GlmMoeDsaForCausalLM</code>, with DSA; 753B/40B in the recipe-from-scratch tutorial, 744B in the InferenceX cited by the inventory). <b>Flash</b> (320B/18B) is not a scaled-down version of it: it has a different architecture, which combines parts from three families. Those parts are <b>KDA + NoPE MLA at 3:1</b> (the Kimi Linear design), the DSA indexer over pools of 4 keys, <b>mHC</b> and DeepSeek-V3-style sigmoid/noaux_tc routing, with 288 experts top-8.', on: ['g5', 'g53', 'g53f', 'kimi', 'mhc'], seg: [['g5', 'g53'], ['g5', 'g53f']] },
        { t: 'DeepSeek-V4-Flash: compressing the KV in time', d: 'Instead of just shrinking heads, V4 compresses the <b>token axis</b>: a 128-token window in every layer, KV ÷4 with an indexer (C4A) or ÷128 read in full (C128A), and a single 512-dim vector that is both K and V. On top of that: mHC, sqrtsoftplus top-6, hash in the first 3 layers and FP4 experts. Declared result: ~7% of the V3.2 KV at 1M tokens.', on: ['v4', 'mhc'], seg: [['v32', 'v4']] },
        { t: 'Built-in drafters', d: 'MTP (V3) → DFlash (block drafting, with context injection from the target layers) → <b>DFlash2</b> in GLM (external, 1.17 B, block of 8) and <b>DSpark</b> in V4 (built-in, 3 stages, block of 5, Markov head). Qwen sticks with 1 MTP layer applied 3 times.', on: ['v4x', 'dflash', 'g53f', 'q38'], seg: [['v4', 'v4x']] },
      ];
      const st = UI.stepper(fig, STEPS, { autoplay: 7000 });
      const famCol = (C, s) => s.row < 0.5 ? C.ds : Math.abs(s.row - 2) < 0.2 ? C.qwen : s.row >= 3.9 ? C.glm : C.muted;
      const bez = (a, m, b, u) => [(1 - u) * (1 - u) * a[0] + 2 * (1 - u) * u * m[0] + u * u * b[0], (1 - u) * (1 - u) * a[1] + 2 * (1 - u) * u * m[1] + u * u * b[1]];
      const ovl = (r, q) => Math.max(0, Math.min(r[0] + r[2], q[0] + q[2]) - Math.max(r[0], q[0])) * Math.max(0, Math.min(r[1] + r[3], q[1] + q[3]) - Math.max(r[1], q[1]));
      V.fig(fig, (g, w, hh, t) => {
        const C = V.C();
        const pad = 16, narrow = w < 600;
        const rowY = (r) => 40 + r * ((hh - 118) / 4.7);
        const P = (s) => [pad + 24 + s.x * (w - 2 * pad - 48), rowY(s.row)];
        const step = STEPS[st.i];
        const el = Math.min(1, st.elapsed() / 0.8);
        const onSet = new Set(step.on);
        const sz = narrow ? 10 : 11.5, ssz = narrow ? 9.5 : 10;
        const maxN = narrow ? 92 : 160;
        // geometria dos rótulos das estações (para desenhar e para evitar colisões)
        const labs = ST.map((s) => {
          const [x, y] = P(s);
          const on = onSet.has(s.id);
          const below = s.lab === 'b';
          const ny = below ? y + 22 : y - 15;
          const sy = below ? ny + 13 : ny - 13;
          const wn = Math.min(maxN, textW(g, s.n, sz, 'sans', on ? 700 : 500));
          const showSub = !narrow || on;
          const ws = Math.min(maxN, textW(g, s.s, ssz));
          const rects = [[x - 11, y - 11, 22, 22], [x - wn / 2 - 3, ny - sz, wn + 6, sz + 5]];
          if (showSub) rects.push([x - ws / 2 - 3, sy - ssz, ws + 6, ssz + 5]);
          return { s, x, y, on, ny, sy, wn, ws, showSub, rects };
        });
        // nomes das famílias
        [[0, 'DeepSeek', C.ds], [2, 'Qwen', C.qwen], [4, 'GLM', C.glm]].forEach(([r, n, col]) => V.text(g, n, pad, rowY(r) - 16, { size: 11, font: 'mono', weight: 600, color: col }));
        // linhas das famílias (o ramo GLM-5 → GLM-5.3 sai na horizontal e desce, estilo metrô)
        SEGS.forEach(([ia, ib]) => {
          const A = byId[ia], B = byId[ib];
          const a = P(A), b = P(B);
          const hot = step.seg.some(([x, y]) => x === ia && y === ib);
          const col = famCol(C, A);
          g.strokeStyle = V.alpha(col, hot ? 0.95 : 0.35); g.lineWidth = hot ? 5 : 4; g.lineCap = 'round';
          const bend = Math.abs(a[1] - b[1]) > 4;
          const m = bend ? [b[0] - 30, a[1]] : [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
          g.beginPath(); g.moveTo(a[0], a[1]); g.quadraticCurveTo(m[0], m[1], b[0], b[1]); g.stroke();
          if (hot) { const u = (t * 0.5) % 1; const pt = bez(a, m, b, u); V.glow(g, pt[0], pt[1], 12, col, 1); }
        });
        g.lineCap = 'butt';
        // transferências (curvas e partículas); as pílulas vêm por cima de tudo, no fim
        const pills = [];
        TR.forEach((tr) => {
          const A = byId[tr.a];
          const a = P(A), b = P(byId[tr.b]);
          const hot = tr.st.includes(st.i);
          const col = A.row < 1.5 || tr.a === 'mhc' || tr.a === 'engram' ? C.ds : A.row >= 3.9 ? C.glm : C.c;
          const m = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2 - 18];
          g.beginPath(); g.moveTo(a[0], a[1]); g.quadraticCurveTo(m[0], m[1], b[0], b[1]);
          g.setLineDash([4, 5]); g.lineDashOffset = -t * (hot ? 30 : 0);
          g.strokeStyle = V.alpha(col, hot ? 0.9 * el : 0.12); g.lineWidth = hot ? 2 : 1; g.stroke(); g.setLineDash([]); g.lineDashOffset = 0;
          if (hot) {
            for (let q = 0; q < 2; q++) { const pt = bez(a, m, b, (t * 0.35 + q * 0.5) % 1); V.dot(g, pt[0], pt[1], 3.2, col); }
            pills.push({ tr, a, m, b, col });
          }
        });
        // estações
        labs.forEach((L) => {
          const { s, x, y, on } = L;
          const col = s.ext ? C.muted : famCol(C, s);
          const rad = on ? 9 + 1.5 * Math.sin(t * 4) : 6.5;
          if (on) V.glow(g, x, y, 26, s.ext ? C.d : col, 0.7);
          g.beginPath(); g.arc(x, y, rad, 0, Math.PI * 2); g.fillStyle = C.bg; g.fill();
          g.lineWidth = on ? 3 : 2.2; g.strokeStyle = s.ext ? (on ? C.d : V.alpha(C.muted, 0.7)) : col;
          if (s.ext) g.setLineDash([2, 2]); g.stroke(); g.setLineDash([]);
          // placa de fundo: as linhas tracejadas passam por trás dos nomes, não por cima
          g.fillStyle = V.alpha(C.bg, 0.85);
          V.rr(g, x - L.wn / 2 - 3, L.ny - sz, L.wn + 6, sz + 5, 3); g.fill();
          if (L.showSub) { V.rr(g, x - L.ws / 2 - 3, L.sy - ssz, L.ws + 6, ssz + 5, 3); g.fill(); }
          V.text(g, s.n, x, L.ny, { size: sz, weight: on ? 700 : 500, align: 'center', color: on ? C.ink : V.alpha(C.ink, 0.7), maxW: maxN });
          if (L.showSub) V.text(g, s.s, x, L.sy, { size: ssz, align: 'center', color: C.muted, maxW: maxN });
        });
        // pílulas das ideias: procura ao longo da curva uma posição que não cubra outra pílula nem um nome de estação
        const placed = [];
        const blockers = [];
        labs.forEach((L) => L.rects.forEach((r) => blockers.push(r)));
        pills.forEach((P2) => {
          const tw = textW(g, P2.tr.l, 10.5) + 12;
          const base = P2.tr.lu || 0.5;
          const cands = [0, 0.07, -0.07, 0.14, -0.14, 0.21, -0.21, 0.28, -0.28].map((d) => base + d).filter((u) => u > 0.12 && u < 0.88);
          let best = null, bestO = Infinity;
          for (const u of cands) {
            const pt = bez(P2.a, P2.m, P2.b, u);
            const r = [pt[0] - tw / 2, pt[1] - 9, tw, 18];
            if (r[0] < 2 || r[0] + r[2] > w - 2) continue;
            let o = 0; placed.concat(blockers).forEach((q) => { o += ovl(r, q); });
            if (o < bestO) { bestO = o; best = r; }
            if (o === 0) break;
          }
          if (!best) { const pt = bez(P2.a, P2.m, P2.b, base); best = [V.clamp(pt[0] - tw / 2, 2, w - 2 - tw), pt[1] - 9, tw, 18]; }
          placed.push(best);
          g.globalAlpha = el;
          V.box(g, best[0], best[1], best[2], 17, { r: 8, fill: C.bg, stroke: V.alpha(P2.col, 0.85), lw: 1.2 });
          V.text(g, P2.tr.l, best[0] + best[2] / 2, best[1] + 12, { size: 10.5, align: 'center', color: C.ink });
          g.globalAlpha = 1;
        });
      });
      autoHeight(fig, (w) => (w < 600 ? 470 : 450));
    })();

    /* ================================================================ FIG 9: requisitos */
    (function () {
      const fig = root.querySelector('#fig-requisitos');
      const panel = fig.querySelector('#req-panel');
      const ROWS = [
        { l: 'Linear mixer with fixed state', c: { glm: ['p', 'KDA already ships in the vLLM of the image (chunked in prefill, recurrent in decode, N+1 states in speculative verification). The recipe fixes state freeing in align mode (<code>patch_mamba_align_state_free</code> and <code>_chunking</code>) and restricts PDL to SM9x/SM10x, because it raced with the KDA state kernels on the GB10.'], qwen: ['o', 'GDN out of the box (FLA in prefill, CUDA in decode). The recipe only picks <code>--mamba-ssm-cache-dtype bfloat16</code>; the model forces <code>mamba_cache_mode=align</code>.'], ds: ['n', 'No linear attention: constant memory comes from the 128-token windows.'] } },
        { l: 'Sparse attention: indexer + kernel', c: { glm: ['p', 'NoPE overlay in the Dockerfile: 64-zero pad for the GLM_NSA 576 geometry of the only sparse-MLA kernel for SM12x, prefill through the sparse path, top-k buffer of 2048 (511 pools + tail), kernel page of 64, indexer aligned to 256 and a "rightsize" workspace (+26–28% KV).'], qwen: ['p', '<code>patch_qsa_fp8_kv</code> teaches the QSA kernels (MQA indexer + split-K sparse GQA) to read FP8 KV: 1.70× more tokens per GiB. Tile profiles for the 48 SMs of the GB10 (optional).'], ds: ['p', 'Fix #22 routes <code>nvfp4_ds_mla</code> to the FP8 kernel (without it, ~1 tok/s at 600K versus ~17 tok/s). Indexer shortcuts for short context (#49486/#52492) and skipping of empty C128 (#48957).'] } },
        { l: 'Hybrid KV: groups and prefix cache', c: { glm: ['p', '7 KV groups (MLA + indexer, kpool tail, 4× Mamba, drafter window). Patches: drafter group with slot-share, hybrid prefix hit, clamp and stride of the kpool tail (which corrupted the indexer of another layer).'], qwen: ['p', 'Attention block matched to the GDN page (1,664 tokens); backport of <code>disable_eagle_block_drop</code> (#53388), so the last prefix-cache block is not lost with MTP.'], ds: ['p', '1× MLAAttentionSpec (block 256) + 3× SlidingWindowMLASpec; hotfix #26 for prefix hits with SWA groups; <code>--block-size 256</code> for the common 37,376 B page.'] } },
        { l: '4-stream residual', c: { glm: ['p', '<code>patch_glm_eagle3</code>: for the DFlash2 taps, it materializes the deferred <code>hc_post</code> and contracts the 4 streams to 4096.'], qwen: ['o', 'Gated Residual out of the box, replicated on both GPUs (<code>disable_tp</code>): 1.19 GiB read per step, 13% of the total.'], ds: ['p', 'Hotfix #50312 frees the <code>_mtp_hidden_buffer</code> (256 MiB per rank), useless for DSpark. The drafter features (average of the streams) come out of the box.'] } },
        { l: 'MoE in the quantized format', c: { glm: ['p', 'Register the <code>exl3</code> method and keep the experts as trellis + Hadamard (suh/svh) + MCG, with ExLlamaV3 compiled for sm_121a and custom "fat" kernels. Never expand to BF16.'], qwen: ['p', 'NVFP4 experts out of the box, but the 128×128 FP8 experts of the MTP required the <code>mtp.layers.48</code> alias and the <code>FP8_BLOCK_SCALES</code> → Fp8MoEMethod branch; EP is mandatory.'], ds: ['o', 'MegaMoE requires SM100 + EP. On the GB10, the <code>--moe-backend flashinfer_b12x</code> flag runs the MXFP4 experts as W4A16 (dequant to BF16 + BF16 MMA).'] } },
        { l: 'Routing and config aliases', c: { glm: ['o', 'Aliases already present in the vLLM of the image (<code>Glm5NextTextConfig</code>); without them, 7 experts per token instead of 8.'], qwen: ['o', 'Softmax top-10 + shared expert with a sigmoid gate, out of the box.'], ds: ['p', 'Hash layers need the raw <code>input_ids</code> (Vision-Exp patch: <code>requires_raw_input_tokens</code>); image tokens routed with <code>bias_vl</code> and no hash (#175).'] } },
        { l: 'Giant embedding (N-gram)', c: { glm: ['n', 'Does not apply.'], qwen: ['p', '51 GB FP8 table: port of the dispatch from PR #53899 (<code>ple_layer_patched.py</code>) + <code>detect_ple_dtype.py</code> + <code>ple_embedding_dtype</code> via --hf-overrides. Without it, vLLM tries to create a ~102 GB BF16 embedding.'], ds: ['n', 'Does not apply (Engram comes in V4.1, outside this recipe).'] } },
        { l: 'Position and long context', c: { glm: ['o', 'NoPE on the text: nothing to scale; 1,048,576 declared (the recipe serves 850k).'], qwen: ['o', 'YaRN by flag, with a catch: the override has to be nested in <code>text_config</code> (<code>start.sh:919-946</code>), otherwise it becomes a silent no-op. The recipe nests only <code>rope_type</code>, <code>factor</code> and <code>original_max_position_embeddings</code>; the official card passes the full dictionary, and the inventory points out the unverified risk that the dict gets replaced wholesale. To be safe, repeat θ = 10⁷, 0.25 and the mRoPE. 1M with YaRN has not been validated in the kit (<code>YARN_ENABLE=false</code>).'], ds: ['o', 'YaRN ×16 out of the box. The fix that restricts YaRN to the compressed layers, as in the reference, is opt-in (<code>DSPARK_ENABLE_ROPE_SWA_FIX</code>).'] } },
        { l: 'Drafter (speculative)', c: { glm: ['p', 'DFlash2 is external: architecture registration, <code>is_causal: false</code> honored (otherwise acceptance at positions 2..7 collapses), draft KV in BF16 (auto) and its own KV group.'], qwen: ['p', 'Built-in MTP: index-48 alias and a 47k draft vocabulary (<span class="la-term" data-term="fr-spec">FR-Spec</span>), with a local argmax per rank (+8–10% decode).'], ds: ['p', 'Built-in DSpark: the "k multiple of n_predict" rule forces k = 6 in Vision-Exp (the k = 5 unlock is opt-in); <code>--max-cudagraph-capture-size</code> 48 (6 sequences × 7 tokens, rounded to a multiple of 8), otherwise the 42-token step runs without a <span class="la-term" data-term="cuda-graph">CUDA graph</span> (−12% at c=6); drafter concurrency patches.'] } },
        { l: 'Divisibility at TP=3', c: { glm: ['p', 'Heads 64 → 66 (--hf-overrides + <code>patch_tp3_glm</code>), vocabulary padded to 192, expert parallel for the experts (trellis 2048), drafter at TP=1.'], qwen: ['n', 'Does not apply: the recipe has no TP=3 lane (there are TP=1 and TP=2). At TP=2, the EP flag handles the 640 width of the experts, and the ViT runs replicated (4304/2 = 2152).'], ds: ['p', 'Groups 8 → 9 (heads 64 → 72), with an extra zeroed group and attention sink −∞ (<code>apply_tp3_patch.py</code>).'] } },
        { l: 'Vision tower', c: { glm: ['o', '24-block ViT in the vLLM of the image, with ε forced to 1e-6; the recipe only caps <code>max_image_tokens</code> at 2048.'], qwen: ['o', '27-block ViT out of the box; <code>--mm-encoder-tp-mode data</code> (replicate), because 4304 does not divide cleanly by 2.'], ds: ['p', 'The container image is text-only: the hotfix installs ViT + Aligner + processor; the encoder cache hash is now "salted" with <code>num_tokens</code> (#172).'] } },
      ];
      const MODELS = ['glm', 'qwen', 'ds'];
      const S = { sel: [1, 0], hover: null, focus: 'all' };
      const stName = { p: 'patch / overlay', o: 'out of the box (or flag)', n: 'does not apply' };
      const colOf = (C, s) => s === 'p' ? C.d : s === 'o' ? C.a : V.alpha(C.muted, 0.5);
      const upd = () => {
        const C = V.C();
        const [r, c] = S.sel; const m = MODELS[c]; const cell = ROWS[r].c[m];
        panel.innerHTML = '';
        panel.appendChild(h('div', { class: 't18-req' },
          h('h5', null, ROWS[r].l, ' · ', NAME[m], h('span', { class: 'st', style: { background: V.alpha(colOf(C, cell[0]), 0.2), color: 'var(--text)', border: '1px solid ' + colOf(C, cell[0]) } }, stName[cell[0]])),
          h('div', { html: cell[1] })));
      };
      upd();
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Highlight', options: [{ v: 'all', label: 'all' }].concat(MODEL_OPTS), value: 'all', onChange: (v) => { S.focus = v; if (v !== 'all') { S.sel = [S.sel[0], MODELS.indexOf(v)]; upd(); } } });
      const cnt = UI.readout(ctl, 'Patches', '—');
      let geo = null;
      V.fig(fig, (g, w, hh, t, dt, _S, H) => {
        const C = V.C(), pad = 14, narrow = w < 560;
        const lw = Math.min(narrow ? 150 : 250, w * 0.42);
        const cx0 = pad + lw + 6, cw = (w - pad - cx0) / 3;
        const top = 44;
        const rh = (hh - top - 10) / ROWS.length;
        geo = { cx0, cw, top, rh };
        MODELS.forEach((m, j) => {
          const on = S.focus === 'all' || S.focus === m;
          V.text(g, narrow ? SHORT[m] : NAME[m].replace('-Flash-Next', '').replace('-Flash', ''), cx0 + j * cw + cw / 2, 20, { size: narrow ? 11.5 : 12.5, weight: 700, align: 'center', color: on ? C[m] : V.alpha(C[m], 0.35) });
          const np = ROWS.filter((r) => r.c[m][0] === 'p').length;
          V.text(g, narrow ? `${np}/${ROWS.length} ⚙` : `${np} of ${ROWS.length} patched`, cx0 + j * cw + cw / 2, 35, { size: 10, align: 'center', color: C.muted });
        });
        const pt = H.pointer;
        S.hover = null;
        if (pt.inside && pt.x > cx0 && pt.y > top) { const r = Math.floor((pt.y - top) / rh), c = Math.floor((pt.x - cx0) / cw); if (r >= 0 && r < ROWS.length && c >= 0 && c < 3) S.hover = [r, c]; }
        if (pt.clicked && S.hover) { S.sel = S.hover; upd(); }
        ROWS.forEach((row, i) => {
          const y = top + i * rh;
          if (i % 2 === 0) { g.fillStyle = V.alpha(C.ink, 0.035); g.fillRect(pad, y, w - 2 * pad, rh); }
          const lines = wrapText(g, row.l, lw - 8, narrow ? 11 : 12.5, 500);
          lines.slice(0, 3).forEach((s, k) => V.text(g, s, pad + 6, y + rh / 2 + (k - (Math.min(3, lines.length) - 1) / 2) * 14 + 4, { size: narrow ? 11 : 12.5, weight: 500, color: C.ink }));
          MODELS.forEach((m, j) => {
            const s = row.c[m][0];
            const x = cx0 + j * cw + cw / 2, yy = y + rh / 2;
            const on = S.focus === 'all' || S.focus === m;
            const col = colOf(C, s);
            const puls = s === 'p' ? 1 + 0.12 * Math.sin(t * 3 + i * 0.7 + j) : 1;
            const rad = Math.min(10, rh * 0.32) * puls;
            g.globalAlpha = on ? 1 : 0.25;
            if (s === 'p') V.glow(g, x, yy, rad * 2.2, col, 0.35);
            g.beginPath(); g.arc(x, yy, rad, 0, Math.PI * 2); g.fillStyle = s === 'n' ? 'transparent' : V.alpha(col, 0.9); g.fill();
            g.lineWidth = 1.5; g.strokeStyle = col; g.stroke();
            if (s === 'p') V.text(g, '⚙', x, yy + 4, { size: 11, align: 'center', color: C.bg });
            if (s === 'o') V.text(g, '✓', x, yy + 4, { size: 11, align: 'center', color: C.bg, weight: 700 });
            g.globalAlpha = 1;
            const isSel = S.sel[0] === i && S.sel[1] === j, isH = S.hover && S.hover[0] === i && S.hover[1] === j;
            if (isSel || isH) { g.strokeStyle = isSel ? C.ink : V.alpha(C.ink, 0.5); g.lineWidth = isSel ? 2 : 1.2; V.rr(g, cx0 + j * cw + 4, y + 2, cw - 8, rh - 4, 8); g.stroke(); }
          });
        });
        const tot = MODELS.map((m) => ROWS.filter((r) => r.c[m][0] === 'p').length);
        cnt.set(tot.join(' · '));
      });
      autoHeight(fig, (w) => (w < 560 ? 44 + ROWS.length * 44 + 12 : 44 + ROWS.length * 38 + 12));
    })();

    /* ================================================================ FIG 10: exercício MiMo */
    (function () {
      const fig = root.querySelector('#fig-exercicio');
      const pat = [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0];
      const L = pat.map((p, i) => ({ i, mix: p === 0 ? 'full' : 'swa', ffn: i === 0 ? 'dense' : 'moe' }));
      const STEPS = [
        { t: 'Shape and map', d: '<code>num_hidden_layers: 48</code>; <code>hybrid_layer_pattern</code> has 9 zeros (dense global attention, no indexer, in layers 0, 5, 11, …, 47) and 39 ones (128-token window). <code>moe_layer_freq</code> starts with 0: layer 0 is dense (16,384) and the other 47 are MoE with 256 experts, top-8, <b>no shared expert</b>.', hl: () => true },
        { t: 'Parameters and weights', d: 'Experts: 47 × 256 × 3 × 2048 × 4096 = <b>302.8 B</b>. In MXFP4 (4.25 bits per weight) that comes to 160.9 GB (149.8 GiB), or 74.9 GiB per rank at TP=2. The whole backbone takes 79.1 GiB per rank; with encoders and DFlash, the estimate is <b>≈ 82 GiB per rank</b>, versus 82.7 GiB measured in an SGLang kit.', hl: (l) => l.ffn === 'moe' },
        { t: 'Active parameters and FLOPs', d: '8 × 25.17 M × 47 ≈ 9.5 B in the experts, plus ~4.5 B of attention (FP8 qkv + BF16 o_proj), dense layer and heads: <b>~15 B active</b>, as the card says. By the 2N rule, ≈ 30 GFLOP per token (illustrative estimate).', hl: (l) => l.ffn === 'moe' },
        { t: 'KV that grows', d: 'Only the 9 global layers (full dense attention, no indexer): 9 × 2 KV heads per rank × (192 for K + 128 for V) × 1 B (FP8) = <b>5,760 B per token per rank</b> (11,520 in BF16). Compare: 7,579 in GLM and 6,656 in Qwen.', hl: (l) => l.mix === 'full' },
        { t: 'Fixed state', d: 'The 39 window layers store only ~128 tokens (plus k and block rounding, ~192–256): 39 × 4 × 320 × 1 B × ~256 ≈ <b>10–13 MB per request</b> in FP8. DFlash (5 SWA layers, window of 1024) adds ~11 MB in BF16.', hl: (l) => l.mix === 'swa' },
        { t: 'Pool and capacity', d: 'Budget at GMU 0.85: 0.85 × 121.69 = 103.44 GiB; minus 82.7 GiB of weights and ~4 GiB of overhead, that leaves a pool of <b>16.74 GiB</b>. Divided by 5,760 B, it gives 3.12 M tokens. But the hybrid allocator groups the layers 5 at a time, and the 9 global layers take up 10 "slots" (+11% padding): what is left is <b>≈ 2.81 M tokens</b> in FP8, meaning 1M of context fits ~2.7×. It matches the 2,731,530 tokens measured (#58128, 17 GiB pool). PR #58175 measured +10.57% capacity with groups of 3.', hl: (l) => l.mix === 'full' },
        { t: 'Bottlenecks and requirements', d: 'K (192) ≠ V (128): DiffKV backend, which on the GB10 means Triton. Fused QKV <b>pre-sliced for TP=4</b> served at TP=2. Sinks only in the SWA layers. A global <code>sliding_window</code> that can leak into the global layers (silent bug). MXFP4 MoE kernel for sm_121. Two drafters (MTP-3 and DFlash-5).', hl: () => true },
      ];
      const st = UI.stepper(fig, STEPS, { autoplay: 7500 });
      const O = { mh: 22, fh: 8, labelH: 14, markH: 4, minCell: 12, maxCell: 26 };
      V.fig(fig, (g, w, hh, t) => {
        const C = V.C(), pad = 14, i = st.i, step = STEPS[i];
        V.text(g, 'MiMo-V2.6-Flash-MOPD · hybrid_layer_pattern', pad, 18, { size: 11.5, font: 'mono', color: C.muted, maxW: w - 2 * pad });
        const G = stripGeom(w, L, Object.assign({ y: 30 }, O));
        const pulse = 0.6 + 0.4 * Math.sin(t * 3);
        drawStrip(g, G, Object.assign({ mode: 'tipo', labels: true, lit: step.hl }, O));
        G.pos.forEach((p) => { if (step.hl(p.l) && i >= 3 && i <= 5) V.glow(g, p.x + G.cw / 2, p.y + O.mh / 2, 9, mixCol(C, p.l.mix), 0.3 * pulse); });
        // legenda (as globais do MiMo são atenção densa plena: cor própria, não a lavanda de "esparsa")
        let lgx = pad, lgy = 30 + G.height + 12;
        [[C.e, 'dense global GA (9)'], [C.d, 'SWA 128 (39)'], [V.alpha(C.ink, 0.9), 'dense FFN (layer 0)'], [V.alpha(C.ink, 0.38), 'MoE (47)']].forEach(([c, s2]) => {
          const ww = textW(g, s2, 11) + 26;
          if (lgx + ww > w - pad) { lgx = pad; lgy += 17; }
          V.box(g, lgx, lgy - 9, 11, 11, { r: 3, fill: c });
          V.text(g, s2, lgx + 16, lgy + 1, { size: 11, color: C.muted });
          lgx += ww;
        });
        // barra de orçamento
        const by = lgy + 26;
        const budget = 103.44;
        const segs = [['weights (≈ 82.7 GiB)', 82.7, C.b, 1], ['overhead ~4', 4, C.muted, 1], ['KV pool 16.74', 16.74, C.a, 5]];
        const bx0 = pad, bx1 = w - pad, bw = bx1 - bx0;
        V.text(g, 'ONE RANK (TP=2) · GMU 0.85 × 121.69 GiB = 103.44 GiB', pad, by, { size: 10.5, font: 'mono', color: C.muted, maxW: w - 2 * pad });
        V.box(g, bx0, by + 8, bw, 30, { r: 8, fill: V.alpha(C.ink, 0.06), stroke: V.alpha(C.muted, 0.35) });
        let x = bx0;
        segs.forEach(([l, v, col, from], k) => {
          const show = i >= from || (k === 0 && i >= 1);
          const grow = i === from ? V.ease.out(Math.min(1, st.elapsed() / 1.2)) : 1;
          const ww = (v / budget) * bw * (show ? grow : 0);
          if (ww > 0.5) { g.fillStyle = V.alpha(col, 0.8); V.rr(g, x, by + 8, ww, 30, 6); g.fill(); if (ww > textW(g, l, 11) + 12) V.text(g, l, x + 8, by + 27, { size: 11, weight: 600, color: C.bg }); }
          x += (v / budget) * bw * (show ? 1 : 0);
        });
        // números do passo
        const big = [['48 layers', '9 global + 39 SWA'], ['≈ 82 GiB/rank', 'weights at TP=2'], ['~15 B active', '≈ 30 GFLOP/token'], ['5,760 B/token', 'FP8 KV per rank'], ['10–13 MB/req', 'SWA windows (FP8)'], ['~2.81 M tokens', '1M fits ~2.7×'], ['6 requirements', 'DiffKV · TP4 · sinks · MXFP4…']][i];
        const ny = by + 70;
        V.text(g, big[0], pad, ny, { size: w < 560 ? 22 : 28, font: 'display', weight: 700, color: C.ink });
        V.text(g, big[1], pad, ny + 20, { size: 12, color: C.muted, maxW: w - 2 * pad });
        const FORM = ['9 × global + 39 × SWA · 1 dense + 47 MoE', '47 × 256 × 3 × 2048 × 4096 × 4.25/8 B ≈ 149.8 GiB', '≈ 9.5 (experts) + 4.5 (attention) + 0.2 (dense) + 0.6 (head) ≈ 14.8 B', '9 × 2 × (192 + 128) × 1 B = 5,760 B/token', '39 × 4 × 320 × 1 B × ~256 tokens ≈ 10–13 MB', '16.74 GiB ÷ 5,760 B = 3.12 M; groups of 5 → 9 GA in 10 slots (× 10/9 = +11%) → 3.12 ÷ 1.11 ≈ 2.81 M'];
        if (i < 6) { const fl = wrapText(g, FORM[i], w - 2 * pad, 12, 400, 'mono'); g.globalAlpha = Math.min(1, st.elapsed() * 2); fl.slice(0, 3).forEach((s2, k) => V.text(g, s2, pad, ny + 48 + k * 17, { size: 12, font: 'mono', color: C.a })); g.globalAlpha = 1; }
        if (i === 6) {
          const req = ['K ≠ V (DiffKV)', 'QKV pre-sliced TP=4', 'sinks only in SWA', 'sliding_window leaks', 'MoE MXFP4 sm_121', '2 drafters'];
          let rx = pad, ry = ny + 36;
          req.forEach((s, k) => { const ww = textW(g, s, 11) + 18; if (rx + ww > w - pad) { rx = pad; ry += 26; } const a = Math.min(1, Math.max(0, st.elapsed() * 2 - k * 0.3)); g.globalAlpha = a; V.box(g, rx, ry, ww, 21, { r: 10, fill: V.alpha(C.d, 0.18), stroke: C.d, label: s, size: 11 }); g.globalAlpha = 1; rx += ww + 6; });
        }
        fitTo(fig, ny + 36 + (w < 560 ? 3 : 2) * 26 + 8);
      });
    })();
  },
});

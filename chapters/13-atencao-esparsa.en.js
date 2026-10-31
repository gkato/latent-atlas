LA.chapter({
  id: 'atencao-esparsa',
  part: 3,
  order: 13,
  title: 'Sparse attention and 1-million-token context',
  navTitle: 'Sparse attention',
  subtitle: 'Windows that forget on purpose, indexers that choose where to look and compressed KV: how GLM, Qwen and DeepSeek attend to a million tokens without paying the square.',
  minutes: 60,
  recipes: ['glm', 'tf', 'qwen', 'ds'],
  tags: ['sparse attention', 'SWA', 'lightning indexer', 'DSA', 'QSA', 'kpool', 'CSA/HCA', 'top-k', 'long context', 'TensorFold', 'radix select'],
  goals: [
    'Explain why the \\(O(n^2)\\) cost of dense attention rules out 1M tokens, and the three escape routes: <b>read less</b>, <b>store less</b> and <b>fixed state</b>',
    'Describe the <b>lightning indexer</b> (DSA, QSA, kpool): score with ReLU, pick the top-k and attend only to the chosen ones, and why <b>KL distillation</b> preserves quality',
    'Understand the ring-buffered sliding window and DeepSeek-V4\'s <b>C4A/C128A</b> KV compression, and do the bytes-per-token math that shows 1M fits',
    'Read and justify the recipes\' parameters and patches: <code>index_topk</code>, <code>index_kpool</code>, <code>select_k</code>, <code>compress_ratios</code>, SP indexer, workspaces and short-context shortcuts, and compare them with the same indexer on the TensorFold engine (2,051-wide list, radix select, visible pools)',
  ],
  prereqs: ['mla'],
  summary: [
    'Dense attention costs \\(n(n+1)/2\\) pairs per head and per layer in prefill and reads the entire KV for every decode token. At 1M, a dense DeepSeek-V4 would read ~26 GB of KV per generated token.',
    'The <b>sliding window</b> reads only the last \\(W\\) tokens, stored in a ring (slot = pos mod \\(W\\)). The KV per sequence stays constant (DeepSeek: 128 tokens, ≈ 3.4 MB/seq), but the window forgets, so it always comes paired with a global branch.',
    'The <b>lightning indexer</b> scores every key with a few 128-dimensional heads, one shared key, FP8/FP4 and ReLU, and the top-k (~2,048 tokens) goes to the main attention. It is still \\(O(n)\\) per token: it dominates long prefill and the decode bytes at 1M.',
    'The indexer learns by <b>KL distillation</b> against dense attention: first a warmup with the model frozen, then joint sparse training. Since attention is concentrated, the top-k captures nearly all the mass (error ≤ \\(2\\varepsilon\\max\\lVert v\\rVert\\)), and QSA even beats full attention from 512K to 1M on RULER.',
    'GLM (learned pools, 32 × 128) and Qwen (mean, 4 × 128) score blocks of 4, and the tail of ≤ 3 tokens is always included. In GLM, 512 pools + tail = 2,051 overflow the SM120 kernel\'s 2,048 template: the recipe expands 511 pools + tail (2,047) and loses 0.2% of the candidates, the lowest-ranked ones. In the TensorFold version (GLM·TF), with its own kernels, the list holds 2,051 positions, attention is dense up to 2,051 tokens and the selection is a deterministic radix select.',
    'DeepSeek-V4 compresses the KV itself: C4A (21 layers, top-512 among 4:1 entries) and C128A (20 layers, all 128:1 entries), plus a 128-token window. That comes to ≈ 3,850 B/token, versus 25 KB without compression, and 1M ≈ 4 GB per sequence.',
    'In production the indexer is replicated per rank. The SP indexer splits the keys with an exact merge (≈ −4 to −8% TTFT at 128K), workspaces have to be sized right (GLM <code>rightsize</code>: +26–28% KV) and every data-dependent shortcut needs a guard against CUDA graphs (#49486/#52492; #48957 only applies outside FULL graphs, which is the uniform decode mode in the DS recipe).',
  ],
  quiz: [
    { q: 'In a DeepSeek-V4 C4A layer with 1,048,576 tokens of context, how many KV entries does the main attention read per query token?', o: ['1,048,576', '8,320', '640', '128'], a: 2, e: 'A window of 128 raw tokens + the 512 4:1 compressed entries the indexer picked = 640. 8,320 is C128A (128 + 1,048,576/128), and 128 is a window-only layer.' },
    { q: 'Why does the lightning indexer use ReLU instead of softmax on the scores?', o: ['Because ReLU is differentiable and softmax is not', 'Because it only needs to rank the keys: with ReLU each head just votes in favor, without normalizing over all the keys', 'Because softmax does not run in FP8', 'So that the main attention receives ready-made probabilities'], a: 1, e: 'The indexer only feeds a top-k, which depends on the order of the scores, not on probabilities. ReLU zeroes out negative evidence (the head “abstains”) and removes the need to normalize over millions of keys. Both are differentiable almost everywhere, and the main attention runs its own softmax over the chosen keys.' },
    { q: 'In the GLM recipe, the indexer expands 511 pools (not 512) plus the tail. Why?', o: ['Because the SM120 kernel has top-k 2,048 as a template parameter and 512 × 4 + 3 = 2,051 does not fit; dropping the lowest-ranked pool preserves the recent tail', 'To save KV memory in the MLA layers', 'Because the 512th pool is always empty', 'Because DeepGEMM requires widths that are multiples of 511'], a: 0, e: 'The prefill of the GB10 sparse-MLA kernel rejects any top-k ≠ 2,048 in the GLM_NSA geometry. With 511 pools (2,044 tokens) + tail (≤ 3), there are at most 2,047 candidates. Truncating the 2,051-entry row would throw away precisely the tail, the most recent tokens; the 512th pool is the least relevant of the chosen ones (0.2% of the candidates).' },
    { q: 'Which statement about what is <i>stored</i> in the KV cache is correct?', o: ['QSA stores only the 2,048 chosen positions', 'GLM stores the MLA latent compressed 4:1 in the sparse layers', 'Sliding windows make the KV grow with the context', 'The indexer reduces what is read, not what is stored; of the three models, only DeepSeek-V4 compresses the stored main KV'], a: 3, e: 'Any key may be chosen by a future query, so GLM and Qwen store the KV of every position (GLM stores the 512-wide latent, Qwen stores K/V in FP8) and compress only the indexer keys. DeepSeek-V4 stores C4 and C128 entries instead of the raw KV, apart from the 128-token window, which has a constant size.' },
    { q: 'Why is the short-context shortcut (#49486) never taken during CUDA graph capture (guard #52492)?', o: ['Because capture runs in FP32', 'Because the KV cache does not exist yet during capture', 'Because the captured graph would freeze the shortcut and repeat it on long prefixes, picking 0..511 without scoring', 'Because the shortcut is slower inside graphs'], a: 2, e: 'A CUDA graph records the kernel sequence from capture and plays it back on every replay, without re-evaluating the if. If capture took the “all candidates” branch, long prefixes would get the indices 0..511 without scoring, an intermittent corruption under prefix cache + graphs.' },
    { q: 'The SP indexer splits the keys across ranks and merges the local top-ks. Why is the result exactly the global top-k?', o: ['Because the all-gather sorts the keys', 'Because a key in the global top-k has at most k − 1 better keys overall, and therefore also within its slice: it is in its rank\'s local top-k', 'Because each rank still scores every key', 'It is not exact: it is an approximation with a small error'], a: 1, e: 'The union of the local top-ks always contains the global top-k, and the stable merge (the DCP kernel) recovers it. The decision to take the SP path relies on CPU metadata that is identical across ranks, so the collective never gets out of sync. The measured gain was modest (≈ −4 to −8% at 128K) because the rest of prefill was not split.' },
  ],
  mount(root, LA) {
    const V = LA.viz, UI = LA.ui;
    const nf = (x) => Math.round(x).toLocaleString('en-US');
    const sf = (x, d = 1) => (+x).toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d });
    const kfmt = (n) => (n >= 1048576 ? sf(n / 1048576, n % 1048576 ? 2 : 0) + 'M' : n >= 1024 ? nf(n / 1024) + 'K' : nf(n));
    const fmtB = (b) => (b >= 1073741824 ? sf(b / 1073741824, 2) + ' GiB' : b >= 1048576 ? sf(b / 1048576, 1) + ' MiB' : b >= 1024 ? sf(b / 1024, 1) + ' KiB' : nf(b) + ' B');
    let fcache = {};
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(() => { fcache = {}; });
    const font = (s, f, w) => fcache[s + f + w] || (fcache[s + f + w] = V.font(s, f, w));
    const tw = (g, s, size = 12, f = 'sans', w = 400) => { g.font = font(size, f, w); return g.measureText(String(s)).width; };
    const RED = (C) => C.red || C.c;
    // text on vivid fills (mint, amber): the theme's background color, in both themes
    const onInk = (C) => C.bg;
    function wrap(g, s, maxW, size = 12, f = 'sans', w = 400) {
      const words = String(s).split(' ');
      const lines = [];
      let cur = '';
      for (const wd of words) {
        const tt = cur ? cur + ' ' + wd : wd;
        if (tw(g, tt, size, f, w) > maxW && cur) { lines.push(cur); cur = wd; } else cur = tt;
      }
      if (cur) lines.push(cur);
      return lines;
    }
    function textBlock(g, s, x, y, maxW, o = {}) {
      const size = o.size || 12, lh = o.lh || Math.round(size * 1.38);
      const lines = wrap(g, s, maxW, size, o.font || 'sans', o.weight || 400);
      const max = o.maxLines || 99;
      lines.slice(0, max).forEach((l, i) => V.text(g, l, x, y + i * lh, Object.assign({}, o, { size })));
      return Math.min(lines.length, max) * lh;
    }
    function pill(g, x, y, s, col, o = {}) {
      const size = o.size || 11, f = o.font || 'mono', wt = o.weight || 600;
      const wd = tw(g, s, size, f, wt) + 14, ht = size + 9;
      const xx = o.align === 'center' ? x - wd / 2 : o.align === 'right' ? x - wd : x;
      V.box(g, xx, y - ht / 2, wd, ht, { r: ht / 2, fill: V.alpha(col, o.a == null ? 0.18 : o.a), stroke: V.alpha(col, 0.8), lw: 1 });
      V.text(g, s, xx + wd / 2, y + 0.5, { size, font: f, weight: wt, color: o.color || V.C().ink, align: 'center', base: 'middle' });
      return wd;
    }
    function cellText(C, t, cm) {
      const light = cm === 'div' ? Math.abs(t) > 0.55 : t > 0.6;
      return light ? C.bg : C.ink;
    }
    function hatch(g, x, y, w, h, col, step = 6) {
      if (w <= 0 || h <= 0) return;
      g.save(); g.beginPath(); g.rect(x, y, w, h); g.clip();
      g.strokeStyle = col; g.lineWidth = 1; g.beginPath();
      for (let k = -h; k < w; k += step) { g.moveTo(x + k, y + h); g.lineTo(x + k + h, y); }
      g.stroke(); g.restore();
    }
    // stage height by width (desktop × phone); accepts a number or a function (height that depends on state)
    const reH = {};
    function autoH(sel, desk, mobH) {
      const fig = root.querySelector(sel);
      const st = fig && fig.querySelector('.la-fig-stage');
      if (!st) return;
      const val = (v) => (typeof v === 'function' ? v() : v);
      const set = () => { const hh = val(fig.clientWidth < 620 ? mobH : desk) + 'px'; if (st.style.height !== hh) st.style.height = hh; };
      set();
      reH[sel] = set;
      const ro = new ResizeObserver(set);
      ro.observe(fig);
      if (LA.onCleanup) LA.onCleanup(() => ro.disconnect());
    }
    const hS = { janelaRows: 4, compView: 'entry', compRatio: 4 };
    autoH('#fig-mascaras', 430, 440);
    autoH('#fig-janela', 440, () => 380 + hS.janelaRows * 24);
    autoH('#fig-indexador', 470, 400);
    autoH('#fig-destilacao', 400, 380);
    autoH('#fig-pools', 380, 380);
    autoH('#fig-compressor', 390, () => (hS.compView === 'decode' ? (hS.compRatio === 4 ? 330 : 356) : 390));
    autoH('#fig-balanca', 390, 446);
    autoH('#fig-sp-topk', 340, 330);
    autoH('#fig-semaforo', 360, 470);
    const softmax = (xs) => { const m = Math.max(...xs); const e = xs.map((x) => Math.exp(x - m)); const s = e.reduce((a, b) => a + b, 0); return e.map((x) => x / s); };

    figMascaras();
    figCusto();
    figJanela();
    figIndexador();
    figDestilacao();
    figPools();
    figCompressor();
    figBalanca();
    figSP();
    figSemaforo();

    /* ================================================================
       Figure 1 — attention masks
       ================================================================ */
    function figMascaras() {
      const fig = root.querySelector('#fig-mascaras');
      const ctl = UI.controls(fig);
      const S = { mode: 'dense', sweep: true };
      const MODES = [
        { v: 'dense', label: 'Dense' }, { v: 'swa', label: 'Window' }, { v: 'topk', label: 'Top-k' },
        { v: 'pools', label: 'Pools + tail' }, { v: 'c128', label: 'Compressed' }, { v: 'c4a', label: 'Window + compressed top-k' },
      ];
      const INFO = {
        dense: { t: 'Dense attention', d: 'Every pair (i, j ≤ i) is computed: query i reads i + 1 entries, and the total grows with the square.', real: '1M tokens: 1,048,576 entries per query' },
        swa: { t: 'Sliding window (W = 6)', d: 'Only the last W tokens: a diagonal band of fixed width. Cheap, and blind to the distant past.', real: 'DeepSeek-V4: sliding_window = 128' },
        topk: { t: 'Content-based top-k (k = 6)', d: 'An indexer picks the k most relevant keys for each query; while there are at most k tokens, it reads everything.', real: 'DeepSeek-V3.2 DSA: 2,048 tokens' },
        pools: { t: 'Pools of 4 + tail', d: 'Keys are scored in blocks of 4: the 2 best blocks are kept, plus the tail (amber), which has not become a block yet.', real: 'GLM: 511 × 4 + 3 = 2,047 · QSA: ≤ 2,051' },
        c128: { t: 'Compressed (C128A-style)', d: 'Window + ALL the closed blocks, each read as one summarized entry (lavender). Here the blocks have 4 tokens instead of 128.', real: 'C128A at 1M: 128 + 8,192 = 8,320' },
        c4a: { t: 'Window + compressed top-k (C4A-style)', d: 'Window + the 2 compressed blocks the indexer picked, read as summarized entries.', real: 'C4A at 1M: 128 + 512 = 640' },
      };
      UI.seg(ctl, { label: 'Mask', options: MODES, value: 'dense', onChange: (v) => { S.mode = v; } });
      UI.toggle(ctl, { label: 'sweep the queries', value: true, onChange: (v) => { S.sweep = v; } });
      UI.legend(ctl, [['var(--viz-a)', 'token read'], ['var(--viz-d)', 'tail (no block)'], ['var(--viz-b)', 'compressed entry'], ['var(--viz-grid)', 'not read']]);
      const W = 6, K = 6, R = 4, SP = 2, NMAX = 40;
      const rng = V.rng(1313);
      const anchors = [1, 6, 13, 21, 27, 34];
      const rel = [];
      for (let i = 0; i < NMAX; i++) {
        rel[i] = [];
        for (let j = 0; j <= i; j++) {
          let v = rng() * 0.9;
          if (anchors.includes(j)) v += 1.1 + 0.5 * rng();
          if (i > 18 && j >= 15 && j <= 18) v += 0.9;
          v += Math.max(0, 1.1 - (i - j) * 0.3);
          rel[i][j] = v;
        }
      }
      const cache = {};
      function sel(mode, i) {
        const key = mode + ':' + i;
        if (cache[key]) return cache[key];
        const raw = new Set(), tail = new Set(), comp = [];
        const nComplete = Math.floor((i + 1) / R);
        const poolScore = (b) => { let s = 0; for (let j = b * R; j < b * R + R; j++) s += rel[i][j]; return s / R; };
        const topPools = (m) => { const arr = []; for (let b = 0; b < nComplete; b++) arr.push([poolScore(b), b]); arr.sort((a, b) => b[0] - a[0]); return arr.slice(0, m).map((x) => x[1]); };
        const win = () => { for (let j = Math.max(0, i - W + 1); j <= i; j++) raw.add(j); };
        if (mode === 'dense') for (let j = 0; j <= i; j++) raw.add(j);
        else if (mode === 'swa') win();
        else if (mode === 'topk') { const arr = []; for (let j = 0; j <= i; j++) arr.push([rel[i][j], j]); arr.sort((a, b) => b[0] - a[0]); arr.slice(0, K).forEach((x) => raw.add(x[1])); }
        else if (mode === 'pools') { topPools(SP).forEach((b) => { for (let j = b * R; j < b * R + R; j++) raw.add(j); }); for (let j = nComplete * R; j <= i; j++) tail.add(j); }
        else if (mode === 'c128') { win(); for (let b = 0; b < nComplete; b++) comp.push(b); }
        else if (mode === 'c4a') { win(); topPools(SP).forEach((b) => comp.push(b)); }
        return (cache[key] = { raw, tail, comp, reads: raw.size + tail.size + comp.length });
      }
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const mob = w < 640;
        const N = mob ? 24 : NMAX;
        let M, x0, y0, px, py, pw;
        if (!mob) { M = Math.min(h - 58, w * 0.55); x0 = 40; y0 = 36; px = x0 + M + 30; py = y0 - 6; pw = w - px - 16; }
        else { M = Math.min(w - 56, h - 176); x0 = 36; y0 = 30; px = 16; py = y0 + M + 14; pw = w - 32; }
        const cs = M / N, gp = cs > 7 ? 1 : 0.5;
        const cur = S.sweep ? (Math.floor(t * 2.6) + 17) % N : N - 1;
        V.text(g, 'keys j →', x0, y0 - 12, { size: 11, font: 'mono', color: C.muted });
        g.save(); g.translate(x0 - 12, y0); g.rotate(Math.PI / 2); V.text(g, 'queries i →', 0, 0, { size: 11, font: 'mono', color: C.muted }); g.restore();
        const blocky = S.mode === 'pools' || S.mode === 'c128' || S.mode === 'c4a';
        let tot = 0, totD = 0;
        for (let i = 0; i < N; i++) {
          const s = sel(S.mode, i);
          tot += s.reads; totD += i + 1;
          const rowA = S.sweep && i !== cur ? 0.3 : 1;
          const y = y0 + i * cs;
          s.comp.forEach((b) => {
            const x = x0 + b * R * cs;
            g.fillStyle = V.alpha(C.b, 0.6 * rowA);
            g.fillRect(x + gp, y + gp, R * cs - 2 * gp, cs - 2 * gp);
          });
          for (let j = 0; j <= i; j++) {
            const x = x0 + j * cs;
            let col = null;
            if (s.raw.has(j)) col = V.alpha(C.a, (0.45 + 0.55 * Math.min(1, rel[i][j] / 2.4)) * rowA);
            else if (s.tail.has(j)) col = V.alpha(C.d, 0.92 * rowA);
            const inComp = s.comp.length && s.comp.some((b) => j >= b * R && j < b * R + R);
            if (col) {
              const ins = inComp ? cs * 0.2 : 0;
              g.fillStyle = col; g.fillRect(x + gp + ins, y + gp + ins, cs - 2 * gp - 2 * ins, cs - 2 * gp - 2 * ins);
            } else if (!inComp) { g.fillStyle = V.alpha(C.muted, 0.1); g.fillRect(x + gp, y + gp, cs - 2 * gp, cs - 2 * gp); }
          }
        }
        g.strokeStyle = V.alpha(C.muted, 0.35); g.lineWidth = 1; g.strokeRect(x0 - 0.5, y0 - 0.5, M + 1, M + 1);
        if (blocky) for (let b = 1; b < N / R; b++) V.line(g, x0 + b * R * cs, y0 - 5, x0 + b * R * cs, y0, { color: V.alpha(C.muted, 0.7), lw: 1 });
        const yc = y0 + cur * cs;
        g.strokeStyle = C.c; g.lineWidth = 2; V.rr(g, x0 - 2.5, yc - 1.5, (cur + 1) * cs + 5, cs + 3, 3); g.stroke();
        const s = sel(S.mode, cur);
        const lx = x0 + (cur + 1) * cs + 8;
        if (lx + 70 < x0 + M + (mob ? 30 : 26)) V.text(g, '◀ i = ' + cur, lx, yc + cs / 2 + 0.5, { size: 11, font: 'mono', color: C.c, base: 'middle', weight: 600 });
        // panel
        const info = INFO[S.mode];
        let yy = py;
        V.text(g, info.t, px, yy + 14, { size: mob ? 14 : 16, weight: 600, color: C.ink });
        yy += mob ? 22 : 28;
        yy += textBlock(g, info.d, px, yy + 10, pw, { size: mob ? 12 : 13, color: C.muted, maxLines: mob ? 2 : 5 });
        yy += mob ? 6 : 14;
        V.text(g, `query i = ${cur}: reads ${s.reads} of ${cur + 1}`, px, yy + 10, { size: mob ? 12 : 13, font: 'mono', weight: 600, color: C.ink });
        yy += 16;
        const bw = Math.min(pw, 300);
        V.box(g, px, yy, bw, 8, { r: 4, fill: V.alpha(C.muted, 0.18) });
        V.box(g, px, yy, Math.max(4, bw * s.reads / (cur + 1)), 8, { r: 4, fill: C.a });
        yy += 22;
        if (!mob) {
          V.text(g, `whole prompt: ${nf(tot)} of ${nf(totD)} pairs (${Math.round(100 * tot / totD)}%)`, px, yy + 8, { size: 12, font: 'mono', color: C.muted });
          yy += 34;
          V.text(g, 'AT REAL SCALE', px, yy, { size: 10.5, font: 'mono', color: C.muted, weight: 600 });
          yy += 18;
          textBlock(g, info.real, px, yy, pw, { size: 13.5, weight: 600, color: C.c });
        } else {
          V.text(g, info.real, px, yy + 6, { size: 12, weight: 600, color: C.c, maxW: pw });
        }
      });
    }

    /* ================================================================
       Figure 2 — cost per query × context (log-log)
       ================================================================ */
    function figCusto() {
      const fig = root.querySelector('#fig-custo');
      const ctl = UI.controls(fig);
      const S = { lg: 17, idx: true };
      let H = null;
      UI.slider(ctl, { label: 'Context', min: 10, max: 20, step: 0.25, value: 17, fmt: (v) => kfmt(Math.round(2 ** v)) + ' tokens', onInput: (v) => { S.lg = v; if (H) H.redraw(); } });
      UI.toggle(ctl, { label: 'keys scored by the indexer', value: true, onChange: (v) => { S.idx = v; if (H) H.redraw(); } });
      const CUR = [
        { l: 'Dense', s: 'Dense', col: (C) => RED(C), f: (n) => n },
        { l: 'DS · C128A (window + all ÷128)', s: 'C128A', col: (C) => C.ink, f: (n) => Math.min(n, 128) + Math.floor(n / 128) },
        { l: 'Qwen · QSA (≤ 2,051)', s: 'QSA', col: (C) => C.qwen, f: (n) => Math.min(n, 2051) },
        { l: 'GLM · DSA (≤ 2,047)', s: 'GLM DSA', col: (C) => C.glm, f: (n) => Math.min(n, 2047), dash: [7, 6] },
        { l: 'DS · C4A (window + top-512 ÷4)', s: 'C4A', col: (C) => C.ds, f: (n) => Math.min(n, 128) + Math.min(Math.floor(n / 4), 512) },
        { l: 'Window (128)', s: 'Window', col: (C) => C.muted, f: (n) => Math.min(n, 128) },
      ];
      const IDX = { l: 'Indexer: keys scored (n/4)', s: 'Indexer n/4', col: (C) => C.d, f: (n) => n / 4 };
      H = V.fig(fig, (g, w, h) => {
        const C = V.C();
        const mob = w < 620;
        const L = mob ? 44 : 58, T = 16, Rm = mob ? 14 : 250, B = mob ? 142 : 36;
        const pw = w - L - Rm, ph = h - T - B;
        const lx0 = 10, lx1 = 20, ly0 = 1, ly1 = 6.25;
        const X = (n) => L + ((Math.log2(n) - lx0) / (lx1 - lx0)) * pw;
        const Y = (v) => T + ph - ((Math.log10(Math.max(v, 10)) - ly0) / (ly1 - ly0)) * ph;
        const xt = [[1024, '1K'], [4096, '4K'], [16384, '16K'], [65536, '64K'], [262144, '256K'], [1048576, '1M']];
        const yt = [[10, '10'], [100, '100'], [1000, '1K'], [10000, '10K'], [100000, '100K'], [1000000, '1M']];
        xt.forEach(([v, l]) => { V.line(g, X(v), T, X(v), T + ph, { color: C.grid, lw: 1 }); V.text(g, l, X(v), T + ph + 15, { size: 11, font: 'mono', color: C.muted, align: 'center' }); });
        yt.forEach(([v, l]) => { V.line(g, L, Y(v), L + pw, Y(v), { color: C.grid, lw: 1 }); V.text(g, l, L - 6, Y(v) + 4, { size: 11, font: 'mono', color: C.muted, align: 'right' }); });
        V.text(g, 'entries per query', L + 4, T + 10, { size: 11, font: 'mono', color: C.muted });
        if (!mob) V.text(g, 'tokens in context →', L + pw, T + ph + 30, { size: 11, font: 'mono', color: C.muted, align: 'right' });
        const draw = (cv, dash, lw) => {
          g.beginPath();
          for (let k = 0; k <= 160; k++) { const lg = lx0 + ((lx1 - lx0) * k) / 160; const n = 2 ** lg; const x = X(n), y = Y(cv.f(n)); if (k) g.lineTo(x, y); else g.moveTo(x, y); }
          g.strokeStyle = cv.col(C); g.lineWidth = lw || 2.4; if (dash) g.setLineDash(dash); g.stroke(); g.setLineDash([]);
        };
        if (S.idx) draw(IDX, [3, 5], 2.2);
        CUR.forEach((cv) => draw(cv, cv.dash, cv.s === 'GLM DSA' ? 2.4 : 2.4));
        const n = Math.round(2 ** S.lg);
        const xm = X(n);
        V.line(g, xm, T, xm, T + ph, { color: C.c, lw: 1.5, dash: [4, 4] });
        pill(g, xm, T + ph - 12, kfmt(n), C.c, { align: xm > L + pw - 40 ? 'right' : xm < L + 40 ? 'left' : 'center', size: 10.5 });
        const all = S.idx ? CUR.concat([IDX]) : CUR;
        all.forEach((cv) => { V.dot(g, xm, Y(cv.f(n)), 4.2, cv.col(C)); });
        // legend
        const items = all.map((cv) => ({ cv, v: cv.f(n) }));
        if (!mob) {
          let ly = T + 8;
          const lx = L + pw + 18;
          V.text(g, `at ${nf(n)} tokens:`, lx, ly + 4, { size: 12, weight: 600, color: C.ink });
          ly += 22;
          items.forEach(({ cv, v }) => {
            g.strokeStyle = cv.col(C); g.lineWidth = 3; if (cv.dash || cv === IDX) g.setLineDash(cv === IDX ? [3, 4] : cv.dash); g.beginPath(); g.moveTo(lx, ly); g.lineTo(lx + 18, ly); g.stroke(); g.setLineDash([]);
            V.text(g, cv.l, lx + 26, ly + 4, { size: 11.5, color: C.ink, maxW: Rm - 40 });
            V.text(g, nf(v), lx + 26, ly + 20, { size: 12, font: 'mono', weight: 600, color: cv.col(C) });
            ly += 38;
          });
          const dense = n, c4 = CUR[4].f(n);
          V.text(g, `dense ÷ C4A = ${nf(dense / c4)}×`, lx, ly + 6, { size: 12, font: 'mono', color: C.muted });
        } else {
          const cols = 2, cw = (w - 24) / cols;
          items.forEach(({ cv, v }, i) => {
            const cx = 12 + (i % cols) * cw, cy = T + ph + 34 + Math.floor(i / cols) * 26;
            g.strokeStyle = cv.col(C); g.lineWidth = 3; if (cv.dash || cv === IDX) g.setLineDash(cv === IDX ? [3, 4] : cv.dash); g.beginPath(); g.moveTo(cx, cy - 4); g.lineTo(cx + 14, cy - 4); g.stroke(); g.setLineDash([]);
            V.text(g, cv.s + ': ' + nf(v), cx + 20, cy, { size: 11.5, font: 'mono', color: C.ink, maxW: cw - 24 });
          });
        }
      }, { static: true });
    }

    /* ================================================================
       Figure 3 — window ring + receptive field
       ================================================================ */
    function figJanela() {
      const fig = root.querySelector('#fig-janela');
      const ctl = UI.controls(fig);
      const S = { W: 6, L: 3, rf: true };
      let rR = null;
      const upd = () => { if (rR) rR.set(nf(S.L * (S.W - 1) + 1) + ' tokens'); hS.janelaRows = (S.rf ? S.L : 1) + 1; if (reH['#fig-janela']) reH['#fig-janela'](); };
      UI.slider(ctl, { label: 'Window W', min: 3, max: 10, step: 1, value: 6, onInput: (v) => { S.W = v; upd(); } });
      UI.slider(ctl, { label: 'Layers L', min: 1, max: 5, step: 1, value: 3, onInput: (v) => { S.L = v; upd(); } });
      UI.toggle(ctl, { label: 'stack layers (receptive field)', value: true, onChange: (v) => { S.rf = v; upd(); } });
      rR = UI.readout(ctl, 'Reach L·(W−1)+1', '—');
      UI.readout(ctl, 'DS windows (46 layers)', '≈ 3.44 MB/seq');
      UI.readout(ctl, 'DFlash2 window (5 layers)', '20 MiB');
      upd();
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const mob = w < 620;
        const Wn = S.W;
        const cont = t / 0.9;
        const p = Math.floor(cont) + 24;
        const fr = V.ease.out(Math.min(1, (cont - Math.floor(cont)) * 2.4));
        const topH = mob ? 318 : Math.round(h * 0.46);
        let cx0, cx1, cy, rcx, rcy, rr;
        if (!mob) { cx0 = 18; cx1 = w * 0.6; cy = 78; rcx = w * 0.81; rcy = topH / 2 + 12; rr = Math.min(topH / 2 - 34, w * 0.11); }
        else { cx0 = 12; cx1 = w - 12; cy = 70; rcx = w / 2; rcy = 196; rr = 76; }
        const step = mob ? 30 : 36, sz = step - 7;
        V.text(g, mob ? 'token conveyor (newest on the right)' : 'token conveyor belt (the newest arrives from the right)', cx0, 20, { size: 11.5, font: 'mono', color: C.muted });
        g.save(); g.beginPath(); g.rect(cx0 - 4, cy - sz, cx1 - cx0 + 8, sz * 2 + 8); g.clip();
        const nVis = Math.ceil((cx1 - cx0) / step) + 2;
        for (let k = 0; k < nVis; k++) {
          const j = p - k;
          const x = cx1 - sz - k * step + (1 - fr) * step;
          const inWin = k < Wn;
          const col = inWin ? C.a : C.muted;
          V.box(g, x, cy - sz / 2, sz, sz, { r: 6, fill: V.alpha(col, inWin ? 0.24 : 0.07), stroke: V.alpha(col, inWin ? 0.95 : 0.3), lw: k === 0 ? 2.2 : 1 });
          V.text(g, String(j), x + sz / 2, cy + 1, { size: 11, font: 'mono', align: 'center', base: 'middle', color: inWin ? C.ink : C.muted });
        }
        g.restore();
        const bR = cx1 + (1 - fr) * step - 7, bL = cx1 - sz - (Wn - 1) * step + (1 - fr) * step;
        const by = cy - sz / 2 - 9;
        V.line(g, Math.max(cx0, bL), by, bR, by, { color: C.a, lw: 2 });
        V.line(g, Math.max(cx0, bL), by, Math.max(cx0, bL), by + 5, { color: C.a, lw: 2 });
        V.line(g, bR, by, bR, by + 5, { color: C.a, lw: 2 });
        V.text(g, `window W = ${Wn}`, Math.max(cx0, bL) + 4, by - 6, { size: 11, font: 'mono', color: C.a, weight: 600 });
        if (!mob) V.text(g, 'forgotten: their slots have already been overwritten', cx0, cy + sz / 2 + 18, { size: 11, color: C.muted });
        // ring
        const slotOf = (k) => p - ((((p - k) % Wn) + Wn) % Wn);
        const cursl = ((p % Wn) + Wn) % Wn;
        if (mob) V.text(g, 'KV ring', 12, rcy - rr - 6, { size: 11.5, font: 'mono', color: C.muted });
        else V.text(g, 'KV ring', w - 14, 20, { size: 11.5, font: 'mono', color: C.muted, align: 'right' });
        g.beginPath(); g.arc(rcx, rcy, rr, 0, Math.PI * 2); g.strokeStyle = V.alpha(C.muted, 0.3); g.lineWidth = 1; g.stroke();
        const ss = Math.min(28, (2 * Math.PI * rr) / Wn - 4);
        for (let k = 0; k < Wn; k++) {
          const a = -Math.PI / 2 + (2 * Math.PI * k) / Wn;
          const x = rcx + Math.cos(a) * rr, y = rcy + Math.sin(a) * rr;
          const on = k === cursl;
          if (on) V.glow(g, x, y, ss * 1.2 * (1.2 - fr * 0.4), C.c, 0.8 * (1 - fr * 0.6));
          V.box(g, x - ss / 2, y - ss / 2, ss, ss, { r: 6, fill: on ? V.alpha(C.c, 0.3) : V.alpha(C.a, 0.18), stroke: on ? C.c : V.alpha(C.a, 0.7), lw: on ? 2 : 1 });
          V.text(g, String(slotOf(k)), x, y + 1, { size: ss > 22 ? 11 : 10, font: 'mono', align: 'center', base: 'middle', color: C.ink });
          V.text(g, 's' + k, rcx + Math.cos(a) * (rr + ss * 0.95), rcy + Math.sin(a) * (rr + ss * 0.95) + 3, { size: 10, font: 'mono', align: 'center', color: C.muted });
        }
        V.text(g, 'slot = pos mod W', rcx, rcy - 2, { size: 11, font: 'mono', align: 'center', color: C.ink });
        V.text(g, `${p} mod ${Wn} = ${cursl}`, rcx, rcy + 14, { size: 11, font: 'mono', align: 'center', color: C.c, weight: 600 });
        if (!mob) {
          const nx = cx1 - sz / 2 + (1 - fr) * step;
          const a = -Math.PI / 2 + (2 * Math.PI * cursl) / Wn;
          V.arrow(g, Math.min(nx, cx1), cy + sz / 2 + 2, rcx + Math.cos(a) * (rr - ss / 2 - 4), rcy + Math.sin(a) * (rr - ss / 2 - 4), { color: V.alpha(C.c, 0.55), lw: 1.5, curve: -30, head: 7 });
        }
        // receptive field
        const L = S.rf ? S.L : 1;
        const y0 = topH + 16, x0 = mob ? 76 : 110;
        const Tc = mob ? 14 : 28;
        const rows = L + 1;
        const cell = Math.min((w - x0 - 14) / Tc, (h - y0 - 30) / rows - 5);
        const rowY = (r) => y0 + 18 + r * (cell + 5);
        V.text(g, S.rf ? (mob ? 'receptive field' : 'receptive field: who can influence the current token') : 'one layer: only the direct window', 12, y0 + 4, { size: 11.5, font: 'mono', color: C.muted });
        const pc = Tc - 1;
        for (let r = 0; r < rows; r++) {
          const reach = r * (Wn - 1);
          const lbl = r === rows - 1 ? 'tokens' : `layer ${L - r}`;
          V.text(g, lbl, x0 - 8, rowY(r) + cell / 2 + 4, { size: 11, font: 'mono', color: C.muted, align: 'right' });
          for (let c = 0; c < Tc; c++) {
            const x = x0 + c * cell;
            const inR = c >= pc - reach && c <= pc;
            const a = inR ? 0.95 - 0.55 * (r / Math.max(1, rows - 1)) : 0;
            g.fillStyle = inR ? V.alpha(r === 0 ? C.c : C.a, r === 0 ? 0.9 : a) : V.alpha(C.muted, 0.09);
            g.fillRect(x + 1, rowY(r), cell - 2, cell);
          }
        }
        // fan-out of the first layer and a pulse traveling down
        if (rows > 1) {
          const xa = x0 + pc * cell + cell / 2, ya = rowY(0) + cell;
          for (let c = Math.max(0, pc - Wn + 1); c <= pc; c++) V.line(g, xa, ya, x0 + c * cell + cell / 2, rowY(1), { color: V.alpha(C.c, 0.35), lw: 1 });
          const u = (t * 0.45) % 1;
          const rr2 = u * (rows - 1);
          const ri = Math.floor(rr2), fu = rr2 - ri;
          const reachHere = Math.min(pc, (ri + fu) * (Wn - 1));
          const px = x0 + (pc - reachHere) * cell + cell / 2;
          const py = V.lerp(rowY(ri), rowY(Math.min(rows - 1, ri + 1)), fu) + cell / 2;
          V.glow(g, px, py, 12, C.c, 0.9 * (1 - u * 0.7));
        }
        const reachT = L * (Wn - 1) + 1;
        if (reachT > Tc && S.rf) V.text(g, mob ? `reach ${reachT} > ${Tc} shown` : `reach ${reachT} > ${Tc} columns shown`, w - 12, y0 + 4, { size: 11, font: 'mono', color: C.c, align: 'right' });
      });
    }

    /* ================================================================
       Figure 4 — the indexer step by step
       ================================================================ */
    function figIndexador() {
      const fig = root.querySelector('#fig-indexador');
      const ctl = UI.controls(fig);
      const S = { k: 3 };
      UI.slider(ctl, { label: 'k (top-k)', min: 1, max: 6, step: 1, value: 3, onInput: (v) => { S.k = v; } });
      const Q = [[2, -1, 1, 0], [0, 2, -1, 1], [1, 1, 0, -2], [-1, 0, 2, 1]];
      const KK = [[1, 0, 1, 0], [-1, 2, 0, 1], [2, -1, 0, -1], [0, 0, -1, 2], [1, 2, 1, -1], [-2, 1, 1, 0], [0, -1, 2, 1], [2, 1, -1, 0], [-1, -2, 0, 1], [1, 1, 1, 1]];
      const Wt = [1.0, 0.5, 1.25, 0.75];
      const HI = 4, D = 4;
      const SC = Q.map((q) => KK.map((k) => q.reduce((s, v, d) => s + v * k[d], 0)));
      const steps = [
        { t: 'The query and the cheap keys', d: 'The current token projects \\(H_I = 4\\) small queries \\(\\mathbf q^I\\) (on the left). Each past token left <b>one</b> key \\(\\mathbf k^I_s\\) in the indexer cache (the columns on top). At real scale: DeepSeek 64 × 128, GLM 32 × 128 and Qwen 4 × 128, always with one shared key.' },
        { t: 'Dot products', d: 'Each cell at the intersection is \\(\\mathbf q^I_h \\cdot \\mathbf k^I_s\\): the row on the left times the column on top, like a matrix multiplication done by hand. In the recipe this is a small GEMM in FP8/FP4 (DeepGEMM\'s <code>fp8_mqa_logits</code>; <code>_qsa_mqa_paged_kernel</code> in Qwen).' },
        { t: 'ReLU: only votes in favor', d: 'Negative scores become zero. A head that “dislikes” a key simply does not vote for it. There is no normalization over the keys, so each score is independent of the others.' },
        { t: 'Weighted sum over the heads', d: 'The weights \\(w_{t,h}\\), computed from the token itself, say how much to trust each head. The sum gives one score \\(I_{t,s}\\) per key: the bars. Look at key s6: only h0 (3) and h3 (5) vote for it, and the −3 from h1 and h2 become zero in the ReLU. Without the ReLU the score would be 3·1 − 3·0.5 − 3·1.25 + 5·0.75 = 1.5. With it, it rises to 6.75 and lands in 4th place, one step away from the top-3 (set k = 4 and it gets in).' },
        { t: 'Top-k', d: 'The \\(k\\) highest scores are kept (change \\(k\\) with the control). It is a single set, used by every head of the main attention. In the recipe: 2,048 tokens (GLM, Qwen) or 512 compressed entries (DeepSeek C4A).' },
        { t: 'The expensive attention, only on the chosen ones', d: 'The main attention fetches only the chosen rows from the KV cache and runs the softmax over them (illustrative weights). The other rows stay stored; they are just not read in this step.' },
      ];
      const st = UI.stepper(fig, steps, { autoplay: 6500 });
      V.fig(fig, (g, w, h) => {
        const C = V.C();
        const mob = w < 620;
        const n = mob ? 8 : 10;
        const el = st.elapsed();
        const step = st.i;
        const e = (d) => V.ease.inOut(Math.min(1, Math.max(0, el / d)));
        const ml = mob ? 30 : 40;
        const cell = Math.floor(Math.min(mob ? 28 : 30, (w - ml - 16) / (D + n + 2.4)));
        const gapA = Math.round(cell * 0.8), gapW = Math.round(cell * 0.5);
        const totalW = ml + D * cell + gapA + n * cell + gapW + cell;
        const ox = Math.max(0, (w - totalW) / 2);
        const qx = ox + ml, sx = qx + D * cell + gapA, wx = sx + n * cell + gapW;
        const ky = 26, sy = ky + D * cell + Math.round(cell * 0.6);
        const fs = cell >= 26 ? 11.5 : 10.5;
        const drawCell = (x, y, v, norm, o = {}) => {
          const tt = V.clamp(v / norm, -1, 1);
          g.fillStyle = o.fill || V.cmap(tt, 'div');
          V.rr(g, x + 1, y + 1, cell - 2, cell - 2, 4); g.fill();
          if (o.stroke) { g.strokeStyle = o.stroke; g.lineWidth = 2; V.rr(g, x, y, cell, cell, 5); g.stroke(); }
          if (o.label != null) V.text(g, o.label, x + cell / 2, y + cell / 2 + 0.5, { size: fs, font: 'mono', align: 'center', base: 'middle', color: o.tcolor || cellText(C, tt, 'div'), alpha: o.alpha });
        };
        // Kᵀ (d × n)
        V.text(g, '(K^I)ᵀ: one key per token', sx, ky - 9, { size: 11, font: 'mono', color: C.muted });
        for (let d = 0; d < D; d++) for (let s = 0; s < n; s++) drawCell(sx + s * cell, ky + d * cell, KK[s][d], 2, { label: KK[s][d] });
        // q (H × d)
        V.text(g, 'q^I', qx, sy - 9, { size: 11, font: 'mono', color: C.muted });
        for (let hh = 0; hh < HI; hh++) {
          V.text(g, 'h' + hh, qx - 6, sy + hh * cell + cell / 2 + 4, { size: 11, font: 'mono', color: C.muted, align: 'right' });
          for (let d = 0; d < D; d++) drawCell(qx + d * cell, sy + hh * cell, Q[hh][d], 2, { label: Q[hh][d] });
        }
        // scores
        const nShow = step === 1 ? Math.floor(el * 14) : step > 1 ? 99 : 0;
        const topk = (() => {
          const I = [];
          for (let s = 0; s < n; s++) { let v = 0; for (let hh = 0; hh < HI; hh++) v += Wt[hh] * Math.max(0, SC[hh][s]); I.push([v, s]); }
          const sorted = I.slice().sort((a, b) => b[0] - a[0]);
          return { I: I.map((x) => x[0]), sel: new Set(sorted.slice(0, S.k).map((x) => x[1])), order: sorted.map((x) => x[1]) };
        })();
        const reluE = step === 2 ? e(1.4) : step > 2 ? 1 : 0;
        for (let hh = 0; hh < HI; hh++) for (let s = 0; s < n; s++) {
          const x = sx + s * cell, y = sy + hh * cell;
          const idx = hh * n + s;
          if (idx >= nShow) { g.strokeStyle = V.alpha(C.muted, 0.25); g.lineWidth = 1; V.rr(g, x + 1.5, y + 1.5, cell - 3, cell - 3, 4); g.stroke(); continue; }
          const raw = SC[hh][s];
          const v = raw < 0 ? V.lerp(raw, 0, reluE) : raw;
          const selO = step >= 4 && topk.sel.has(s);
          drawCell(x, y, v, 5, { label: Math.round(v), stroke: selO ? C.c : null, alpha: raw < 0 && reluE > 0.5 ? 0.45 : 1 });
        }
        if (step >= 1 && !mob) V.text(g, 'q^I · k^I', sx + n * cell, sy - 9, { size: 11, font: 'mono', color: C.muted, align: 'right' });
        if (step === 2) pill(g, sx + (n * cell) / 2, sy + HI * cell + 26, 'ReLU: negatives → 0', C.d, { align: 'center', size: 11 });
        // weights w
        if (step >= 3) {
          const a = step === 3 ? e(0.8) : 1;
          V.text(g, 'w', wx + cell / 2, sy - 9, { size: 11, font: 'mono', color: C.muted, align: 'center' });
          for (let hh = 0; hh < HI; hh++) {
            V.box(g, wx + 1, sy + hh * cell + 1, cell - 2, cell - 2, { r: 4, fill: V.alpha(C.d, 0.3 * a), stroke: V.alpha(C.d, 0.9 * a), lw: 1 });
            V.text(g, sf(Wt[hh], 2).replace(/0$/, ''), wx + cell / 2, sy + hh * cell + cell / 2 + 0.5, { size: cell >= 26 ? fs - 1 : 8.5, font: 'mono', align: 'center', base: 'middle', color: C.ink, alpha: a });
          }
        }
        // I bars
        const by = sy + HI * cell + 18;
        const bh = mob ? 62 : 70;
        if (step < 3) {
          g.strokeStyle = V.alpha(C.muted, 0.25); g.lineWidth = 1; g.setLineDash([4, 4]);
          V.rr(g, sx, by + 8, n * cell, bh, 6); g.stroke();
          V.rr(g, sx, by + bh + (mob ? 62 : 64), n * cell, 20, 5); g.stroke(); g.setLineDash([]);
          V.text(g, 'I scores per key (step 4)', sx + (n * cell) / 2, by + 8 + bh / 2 + 4, { size: 11, font: 'mono', color: C.muted, align: 'center' });
          V.text(g, 'main KV cache (step 6)', sx + (n * cell) / 2, by + bh + (mob ? 76 : 78), { size: 10.5, font: 'mono', color: C.muted, align: 'center' });
        }
        if (step >= 3) {
          const grow = step === 3 ? e(1.6) : 1;
          const mx = Math.max(...topk.I);
          V.text(g, mob ? 'score I = Σ w·ReLU(q·k)' : 'score I per key = Σ w · ReLU(q·k)', sx, by + 2, { size: 11, font: 'mono', color: C.muted });
          for (let s = 0; s < n; s++) {
            const v = topk.I[s] * grow;
            const hgt = (v / mx) * (bh - 14);
            const x = sx + s * cell + 3, y = by + bh - hgt + 6;
            const selB = step >= 4 && topk.sel.has(s);
            const col = selB ? C.a : step >= 4 ? V.alpha(C.muted, 0.45) : C.b;
            V.box(g, x, y, cell - 6, Math.max(2, hgt), { r: 3, fill: col });
            if (cell >= 24 || selB) V.text(g, sf(topk.I[s], 2).replace(/\.00$/, '').replace(/0$/, ''), x + (cell - 6) / 2, y - 3, { size: 10, font: 'mono', align: 'center', color: selB ? C.ink : C.muted, alpha: grow });
          }
          for (let s = 0; s < n; s++) V.text(g, 's' + s, sx + s * cell + cell / 2, by + bh + 20, { size: 10.5, font: 'mono', align: 'center', color: topk.sel.has(s) && step >= 4 ? C.a : C.muted });
          if (step >= 4) { const r = topk.order.slice(0, S.k).map((s) => 's' + s).join(', '); V.text(g, `top-${S.k}: ${r}`, sx - (mob ? 26 : 0), by + bh + 38, { size: 11.5, font: 'mono', weight: 600, color: C.a }); }
        }
        // KV strip
        if (step >= 5) {
          const a = e(0.8);
          const ky2 = by + bh + (mob ? 58 : 60);
          const selArr = topk.order.slice(0, S.k);
          const att = softmax(selArr.map((s) => topk.I[s] * 0.6));
          V.text(g, mob ? 'main KV cache (softmax on the chosen)' : 'main attention KV cache (softmax weights on the chosen ones)', qx, ky2 - 4, { size: 11, font: 'mono', color: C.muted });
          for (let s = 0; s < n; s++) {
            const on = topk.sel.has(s);
            const x = sx + s * cell + 2;
            V.box(g, x, ky2 + 4, cell - 4, 20, { r: 4, fill: on ? V.alpha(C.a, 0.3 + 0.5 * a) : V.alpha(C.muted, 0.1), stroke: on ? C.a : V.alpha(C.muted, 0.25), lw: 1 });
            if (on) { const k = selArr.indexOf(s); V.text(g, Math.round(att[k] * 100) + '%', x + (cell - 4) / 2, ky2 + 14.5, { size: cell >= 26 ? 10 : 9, font: 'mono', align: 'center', base: 'middle', color: onInk(C), alpha: a }); }
          }
          V.text(g, `reads ${S.k} of ${n}`, qx, ky2 + 18, { size: 11.5, font: 'mono', weight: 600, color: C.c, alpha: a });
        }
      });
    }

    /* ================================================================
       Figure 5 — KL distillation
       ================================================================ */
    function figDestilacao() {
      const fig = root.querySelector('#fig-destilacao');
      const ctl = UI.controls(fig);
      const S = { tau: 0.35, k: 8 };
      UI.slider(ctl, { label: 'Training (phase 1)', min: 0, max: 1, step: 0.01, value: 0.35, fmt: (v) => Math.round(v * 100) + '%', onInput: (v) => { S.tau = v; } });
      UI.slider(ctl, { label: 'k', min: 2, max: 16, step: 1, value: 8, onInput: (v) => { S.k = v; } });
      const NN = 40, r5 = V.rng(505);
      const zT = [], z0 = [], e0 = [];
      for (let s = 0; s < NN; s++) {
        let v = r5.normal() * 0.4;
        if ([4, 11, 19, 27].includes(s)) v += 4.0 + r5() * 0.6;
        v += Math.max(0, 3.0 - (NN - 1 - s) * 0.5);
        zT.push(v); z0.push(r5.normal() * 1.1); e0.push(r5.normal() * 0.45);
      }
      const p = softmax(zT);
      const steps = [
        { t: 'Phase 1 · dense warmup', d: 'The model is frozen and attention stays dense. Only the indexer learns, minimizing the KL between its distribution (mint, below) and the teacher\'s (lavender, on top). Drag the training slider.' },
        { t: 'Phase 2 · joint sparse training', d: 'Top-k is switched on for real: the keys that were not chosen vanish from the attention (hatched) and the whole model learns to work with only the chosen ones. The indexer\'s KL now looks only at the selected set.' },
        { t: 'At inference', d: 'Only the indexer looks at every key; the main attention reads the \\(k\\) chosen ones. In each head, the difference from the dense output is bounded by \\(2\\varepsilon_h \\max\\lVert v\\rVert\\), where \\(\\varepsilon_h\\) is the mass of that head that was left out (there is a single head here).' },
      ];
      const st = UI.stepper(fig, steps, {});
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const mob = w < 620;
        const tau = S.tau;
        const zS = z0.map((z, s) => (1 - tau) * z + tau * (0.95 * zT[s] + e0[s]));
        const q = softmax(zS);
        const ordQ = q.map((v, s) => [v, s]).sort((a, b) => b[0] - a[0]);
        const ordP = p.map((v, s) => [v, s]).sort((a, b) => b[0] - a[0]);
        const selQ = new Set(ordQ.slice(0, S.k).map((x) => x[1]));
        let kl = 0;
        if (st.i === 0) { for (let s = 0; s < NN; s++) kl += p[s] * Math.log(p[s] / q[s]); }
        else {
          let ps = 0, qs = 0; selQ.forEach((s) => { ps += p[s]; qs += q[s]; });
          selQ.forEach((s) => { const a = p[s] / ps, b = q[s] / qs; kl += a * Math.log(a / b); });
        }
        const klLbl = st.i === 0 ? 'KL(p ‖ q)' : 'KL on the set S only';
        let cap = 0; selQ.forEach((s) => { cap += p[s]; });
        let orc = 0; ordP.slice(0, S.k).forEach((x) => { orc += x[0]; });
        const eps = 1 - cap;
        const gw = mob ? 0 : 220;
        const x0 = mob ? 14 : 20, x1 = w - gw - (mob ? 14 : 30);
        const midY = mob ? 150 : h / 2 + 2;
        const amp = mob ? 112 : h / 2 - 44;
        const bw = (x1 - x0) / NN;
        const mxP = Math.max(...p), mxQ = Math.max(...q, mxP * 0.5);
        const scale = Math.max(mxP, mxQ);
        V.text(g, 'teacher: dense attention p', x0, midY - amp - 6, { size: 11, font: 'mono', color: C.b });
        V.text(g, mob ? 'student: softmax(I) · dashed = target (teacher)' : 'student: the indexer\'s softmax(I) · dashed = the target, the teacher\'s shape', x0, midY + amp + 18, { size: 11, font: 'mono', color: C.a, maxW: x1 - x0 });
        V.line(g, x0, midY, x1, midY, { color: V.alpha(C.muted, 0.5), lw: 1 });
        const phase2 = st.i >= 1;
        for (let s = 0; s < NN; s++) {
          const x = x0 + s * bw + 1;
          const hp = (p[s] / scale) * amp, hq = (q[s] / scale) * amp;
          const sel = selQ.has(s);
          if (phase2 && !sel) { g.fillStyle = V.alpha(C.b, 0.12); g.fillRect(x, midY - hp - 2, bw - 2, hp); hatch(g, x, midY - hp - 2, bw - 2, hp, V.alpha(C.b, 0.55), 4); }
          else { V.box(g, x, midY - hp - 2, bw - 2, Math.max(1, hp), { r: 2, fill: V.alpha(C.b, 0.85) }); }
          V.box(g, x, midY + 2, bw - 2, Math.max(1, hq), { r: 2, fill: sel ? C.a : V.alpha(C.a, 0.35) });
          // ghost outline: where the student should land if it imitated the teacher
          if (hp > 1.5) { g.save(); g.setLineDash([3, 3]); g.strokeStyle = V.alpha(C.b, 0.55); g.lineWidth = 1; g.strokeRect(x + 0.5, midY + 2.5, bw - 3, hp - 1); g.restore(); }
          if (sel) { g.strokeStyle = C.c; g.lineWidth = 1.5; V.rr(g, x - 1, midY + 1, bw, Math.max(3, hq) + 2, 2); g.stroke(); }
        }
        if (st.i === 2) {
          const pulse = 0.5 + 0.5 * Math.sin(t * 3);
          selQ.forEach((s) => { const x = x0 + s * bw + bw / 2; V.glow(g, x, midY - 4, 10, C.c, 0.4 + 0.4 * pulse); });
        }
        // gauges
        let gx, gy, gW;
        if (!mob) { gx = w - gw + 4; gy = 26; gW = gw - 20; }
        else { gx = 14; gy = midY + amp + 34; gW = w - 28; }
        const klTxt = sf(kl, 3);
        if (!mob) {
          V.text(g, klLbl, gx, gy, { size: 11, font: 'mono', color: C.muted });
          V.text(g, klTxt + ' nats', gx, gy + 26, { size: 22, font: 'mono', weight: 700, color: kl < 0.2 ? C.a : kl < 0.8 ? C.d : RED(C) });
          gy += 58;
          V.text(g, `mass captured by the top-${S.k}`, gx, gy, { size: 11, font: 'mono', color: C.muted });
          gy += 12;
          V.box(g, gx, gy, gW, 14, { r: 7, fill: V.alpha(C.muted, 0.16) });
          V.box(g, gx, gy, Math.max(4, gW * cap), 14, { r: 7, fill: C.a });
          g.strokeStyle = C.b; g.lineWidth = 2; V.line(g, gx + gW * orc, gy - 4, gx + gW * orc, gy + 18, { color: C.b, lw: 2 });
          V.text(g, `student ${sf(cap * 100, 1)}%`, gx, gy + 32, { size: 12.5, font: 'mono', weight: 600, color: C.a });
          V.text(g, `oracle ${sf(orc * 100, 1)}%`, gx, gy + 50, { size: 12, font: 'mono', color: C.b });
          gy += 76;
          V.text(g, 'ε = mass lost', gx, gy, { size: 11, font: 'mono', color: C.muted });
          V.text(g, sf(eps * 100, 1) + '%', gx, gy + 22, { size: 18, font: 'mono', weight: 700, color: eps < 0.15 ? C.a : C.d });
          V.text(g, `error ≤ ${sf(2 * eps, 2)}·max‖v‖`, gx, gy + 44, { size: 11.5, font: 'mono', color: C.muted });
          V.text(g, `${S.k} of ${NN} keys = ${Math.round((100 * S.k) / NN)}%`, gx, gy + 66, { size: 11, font: 'mono', color: C.muted });
          V.text(g, 'recipe: 2,048 of 1M ≈ 0.2%', gx, gy + 84, { size: 11, font: 'mono', color: C.c });
        } else {
          V.text(g, `${st.i === 0 ? 'KL' : 'KL on S'} = ${klTxt} nats`, gx, gy, { size: 13, font: 'mono', weight: 700, color: kl < 0.2 ? C.a : kl < 0.8 ? C.d : RED(C) });
          V.text(g, `ε = ${sf(eps * 100, 1)}%`, gx + gW, gy, { size: 13, font: 'mono', weight: 700, color: eps < 0.15 ? C.a : C.d, align: 'right' });
          gy += 14;
          V.box(g, gx, gy, gW, 12, { r: 6, fill: V.alpha(C.muted, 0.16) });
          V.box(g, gx, gy, Math.max(4, gW * cap), 12, { r: 6, fill: C.a });
          V.line(g, gx + gW * orc, gy - 3, gx + gW * orc, gy + 15, { color: C.b, lw: 2 });
          V.text(g, `top-${S.k}: student ${sf(cap * 100, 1)}% · oracle ${sf(orc * 100, 1)}%`, gx, gy + 30, { size: 11.5, font: 'mono', color: C.ink });
        }
      });
    }

    /* ================================================================
       Figure 6 — pools of 4, the tail and the buffer
       ================================================================ */
    function figPools() {
      const fig = root.querySelector('#fig-pools');
      const ctl = UI.controls(fig);
      const S = { mode: 'stock', n: 31, seed: 61 };
      UI.seg(ctl, { label: 'Path', options: [{ v: 'stock', label: 'GLM stock' }, { v: 'receita', label: 'GLM recipe' }, { v: 'tf', label: 'GLM·TF' }, { v: 'qsa', label: 'Qwen QSA' }], value: 'stock', onChange: (v) => { S.mode = v; } });
      UI.slider(ctl, { label: 'Context', min: 13, max: 44, step: 1, value: 31, onInput: (v) => { S.n = v; } });
      UI.button(ctl, '↻ new scores', () => { S.seed = (S.seed * 7 + 13) % 997; });
      const SEL = 4, RR = 4, TPL = 16, ROUND = 8;
      const tfc = (C) => C.__tf || (C.__tf = (getComputedStyle(document.documentElement).getPropertyValue('--tf') || '').trim() || C.d);
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const mob = w < 620;
        const n = S.n;
        const P = Math.floor(n / RR), tail = n % RR;
        const rng = V.rng(S.seed + 1);
        const score = Array.from({ length: P }, () => 0.15 + rng() * 0.85);
        const ord = score.map((v, b) => [v, b]).sort((a, b) => b[0] - a[0]).map((x) => x[1]);
        const nSel = Math.min(SEL, P);
        const selected = ord.slice(0, nSel);
        const expanded = S.mode === 'receita' ? ord.slice(0, Math.min(SEL - 1, P)) : selected;
        const dropped = S.mode === 'receita' && P >= SEL ? ord[SEL - 1] : -1;
        const expSet = new Set(expanded);
        const candidates = expanded.length * RR + tail;
        const bufW = S.mode === 'stock' ? Math.ceil((SEL * RR + (RR - 1)) / ROUND) * ROUND : S.mode === 'receita' ? TPL : SEL * RR + (RR - 1);
        const ok = S.mode !== 'stock';
        const x0 = mob ? 12 : 26, x1 = w - (mob ? 12 : 26);
        const cw = Math.min(26, (x1 - x0) / n);
        const tx = x0 + ((x1 - x0) - cw * n) / 2;
        const ty = 34;
        const dense = S.mode === 'tf' && P <= SEL;   // GLM·TF: up to (SEL + 1)·4 − 1 tokens the row is dense
        const methodTxt = S.mode === 'qsa' ? 'mean → RMSNorm → RoPE at the block start (BF16)' : S.mode === 'tf' ? 'per-channel softmax(gate + ape) → FP8 e4m3 + 2^k scale = 144 B' : 'per-channel softmax(gate + ape) → Hadamard-128 → FP8 = 132 B';
        V.text(g, 'context tokens (the query is the last one)', x0, ty - 12, { size: 11.5, font: 'mono', color: C.muted });
        for (let j = 0; j < n; j++) {
          const b = Math.floor(j / RR);
          const inTail = b >= P;
          const inSel = !inTail && expSet.has(b);
          const x = tx + j * cw;
          const col = inTail ? C.d : inSel ? C.a : b === dropped ? RED(C) : C.muted;
          V.box(g, x + 1, ty, cw - 2, cw - 2, { r: 3, fill: V.alpha(col, inTail || inSel ? 0.75 : b === dropped ? 0.35 : 0.14), stroke: V.alpha(col, 0.7), lw: 1 });
          if (cw >= 18) V.text(g, String(j), x + cw / 2, ty + cw / 2, { size: 9.5, font: 'mono', align: 'center', base: 'middle', color: inTail || inSel ? onInk(C) : C.muted });
        }
        if (tail) { const xa = tx + P * RR * cw; V.text(g, 'tail', xa + (tail * cw) / 2, ty + cw + 13, { size: 11, font: 'mono', align: 'center', color: C.d, weight: 600 }); }
        // pools
        const py = ty + cw + 44;
        const pw = Math.max(10, Math.min(RR * cw - 6, 40));
        const bh = mob ? 54 : 70;
        for (let b = 0; b < P; b++) {
          const xc = tx + b * RR * cw + (RR * cw) / 2;
          const sel = selected.includes(b);
          const exp = expSet.has(b);
          const col = exp ? C.a : b === dropped ? RED(C) : C.muted;
          for (let k = 0; k < RR; k++) {
            const xs = tx + (b * RR + k) * cw + cw / 2;
            V.line(g, xs, ty + cw - 1, xc, py, { color: V.alpha(col, exp ? 0.5 : 0.18), lw: 1 });
            if (exp || b === dropped) { const u = (t * 0.8 + k * 0.13 + b * 0.07) % 1; const pt = V.along(xs, ty + cw, xc, py, u); V.dot(g, pt[0], pt[1], 1.8, col); }
          }
          V.box(g, xc - pw / 2, py, pw, 16, { r: 4, fill: V.alpha(col, exp ? 0.8 : 0.2), stroke: col, lw: 1 });
          const hh = score[b] * bh;
          V.box(g, xc - pw / 2 + 2, py + 22 + bh - hh, pw - 4, hh, { r: 3, fill: V.alpha(col, exp ? 0.85 : b === dropped ? 0.5 : 0.35) });
          const rk = ord.indexOf(b) + 1;
          if (sel && pw >= 14 && !dense) V.text(g, '#' + rk, xc, py + 22 + bh - hh - 4, { size: 10, font: 'mono', align: 'center', color: b === dropped ? RED(C) : C.ink, weight: 600 });
          if (b === dropped) { V.line(g, xc - pw / 2, py + 2, xc + pw / 2, py + 14, { color: RED(C), lw: 2 }); V.line(g, xc + pw / 2, py + 2, xc - pw / 2, py + 14, { color: RED(C), lw: 2 }); }
        }
        const my = py + 22 + bh + 18;
        V.text(g, (mob ? '' : 'how 4 keys become 1: ') + methodTxt, x0, my, { size: 11, font: 'mono', color: S.mode === 'qsa' ? C.qwen : S.mode === 'tf' ? tfc(C) : C.glm, maxW: x1 - x0 });
        // buffer
        const bY = my + 34;
        const slotW = Math.min(22, (x1 - x0 - 10) / Math.max(bufW, TPL + 4));
        const bx = x0;
        const realW = S.mode === 'stock' ? '2,176' : S.mode === 'receita' ? '2,048' : '2,051';
        const bufLbl = S.mode === 'tf' && dense ? (mob ? `dense: reads ${n} positions, no list` : `dense attention: reads all ${n} positions, no index list`) : mob ? `buffer: width ${bufW} (real: ${realW})` : `index buffer: width ${bufW} (at real scale: ${realW})`;
        V.text(g, bufLbl, bx, bY - 8, { size: 11.5, font: 'mono', color: C.ink, maxW: x1 - x0 });
        for (let k = 0; k < bufW; k++) {
          const x = bx + k * slotW;
          const isTok = k < expanded.length * RR;
          const isTail = !isTok && k < candidates;
          const col = isTok ? C.a : isTail ? C.d : C.muted;
          if (!isTok && !isTail) { g.fillStyle = V.alpha(C.muted, 0.1); g.fillRect(x + 1, bY, slotW - 2, 16); hatch(g, x + 1, bY, slotW - 2, 16, V.alpha(C.muted, 0.4), 4); }
          else V.box(g, x + 1, bY, slotW - 2, 16, { r: 2, fill: V.alpha(col, 0.8) });
        }
        const tplX = bx + TPL * slotW;
        if (S.mode === 'stock' || S.mode === 'receita') {
          V.line(g, tplX, bY - 4, tplX, bY + 22, { color: ok ? C.e : RED(C), lw: 2.5 });
          V.text(g, 'kernel template', tplX + 4, bY + 32, { size: 10.5, font: 'mono', color: ok ? C.e : RED(C) });
        }
        const yMsg = bY + (mob ? 54 : 50);
        let msg, sub;
        if (S.mode === 'stock') { msg = '✕ the SM120 kernel rejects the table'; sub = 'expects sparse block_tables shape (256, 1, 2048), got (256, 1, 2176): 512 pools × 4 + tail 3 = 2,051 → rounded up to 2,176'; }
        else if (S.mode === 'receita') { msg = '✓ fits the 2,048 template'; sub = `511 pools × 4 + tail ≤ 3 = 2,047; here ${candidates} valid out of ${bufW}. ${dropped >= 0 ? 'Pool #4 (the lowest-ranked one) is dropped: 0.2% at real scale.' : 'Few pools: nothing is dropped.'}`; }
        else if (S.mode === 'tf' && dense) { msg = '✓ dense row: reads every position'; sub = `Up to ${(SEL + 1) * RR - 1} tokens here (2,051 at real scale) the top-k would pick everything, and TensorFold does dense attention over the latent: ${n} of ${n} positions.`; }
        else if (S.mode === 'tf') { msg = mob ? '✓ own kernel: 2,051, nothing dropped' : '✓ TensorFold\'s own kernel: width 2,051, nothing dropped'; sub = `512 pools × 4 + tail ≤ 3 = up to 2,051 tokens; here ${candidates} of ${bufW}. The pools come out of a radix select, in ascending order.`; }
        else { msg = '✓ custom Triton kernels accept 2,051'; sub = `512 blocks × 4 + tail ≤ 3 = up to 2,051 positions; here ${candidates}. Then, sparse GQA over the original K/V.`; }
        V.text(g, msg, x0, yMsg, { size: 13, weight: 600, color: ok ? C.e : RED(C), maxW: x1 - x0 });
        textBlock(g, sub, x0, yMsg + 18, x1 - x0, { size: 11.5, color: C.muted, maxLines: 3 });
      });
    }

    /* ================================================================
       Figure 7 — the DeepSeek-V4 compressor
       ================================================================ */
    function figCompressor() {
      const fig = root.querySelector('#fig-compressor');
      const ctl = UI.controls(fig);
      const S = { ratio: 4, view: 'entry' };
      let tRef = 0, tNow = 0;
      UI.seg(ctl, { label: 'Ratio', options: [{ v: 4, label: 'C4 (with overlap)' }, { v: 128, label: 'C128' }], value: 4, onChange: (v) => { S.ratio = v; tRef = tNow; hS.compRatio = v; if (reH['#fig-compressor']) reH['#fig-compressor'](); } });
      UI.seg(ctl, { label: 'View', options: [{ v: 'entry', label: 'one entry' }, { v: 'decode', label: 'token by token (decode)' }], value: 'entry', onChange: (v) => { S.view = v; tRef = tNow; hS.compView = v; if (reH['#fig-compressor']) reH['#fig-compressor'](); } });
      const CH = 6;
      const r7 = V.rng(707);
      const mk = (rows) => {
        const ape = Array.from({ length: Math.min(rows, 128) }, () => Array.from({ length: CH }, () => r7.normal() * 0.4));
        const Z = Array.from({ length: rows }, (_, j) => Array.from({ length: CH }, (_, c) => Math.round((r7.normal() * 1.2 + ape[j % ape.length][c]) * 10) / 10));
        const X = Array.from({ length: rows }, () => Array.from({ length: CH }, () => Math.round(r7.normal() * 10) / 10));
        const Wm = Array.from({ length: rows }, () => new Array(CH).fill(0));
        for (let c = 0; c < CH; c++) { const sm = softmax(Z.map((r) => r[c])); sm.forEach((v, j) => { Wm[j][c] = v; }); }
        const out = Array.from({ length: CH }, (_, c) => X.reduce((s, r, j) => s + Wm[j][c] * r[c], 0));
        return { Z, X, Wm, out };
      };
      const D4 = mk(8), D128 = mk(128);
      V.fig(fig, (g, w, h, t) => {
        tNow = t;
        const C = V.C();
        const mob = w < 620;
        const tt = t - tRef;
        if (S.view === 'entry') {
          const D = S.ratio === 4 ? D4 : D128;
          const per = 10;
          const u = tt % per;
          const ph = u < 2.2 ? 0 : u < 4.8 ? 1 : u < 7.2 ? 2 : 3;
          const pe = (a, b) => V.ease.inOut(V.clamp((u - a) / (b - a), 0, 1));
          const rowsShow = S.ratio === 4 ? 8 : 16;
          const mx = mob ? 76 : 150, my = 34;
          const cwid = Math.min(mob ? 42 : 52, (w - mx - (mob ? 14 : 250)) / CH);
          const chh = S.ratio === 4 ? (mob ? 24 : 30) : (mob ? 11 : 14);
          const title = ph === 0 ? 'gate scores Z + B (ape)' : ph === 1 ? 'per-channel softmax (column): weights sum to 1' : ph === 2 ? (mob ? 'sum: Σ_j s_j ⊙ C_j' : 'weighted sum: Σ_j s_j ⊙ C_j (C values not shown)') : 'post-processing → 584 B row';
          V.text(g, title, mx, my - 12, { size: 12, font: 'mono', weight: 600, color: C.ink });
          const colSweep = ph === 1 ? Math.min(CH - 1, Math.floor(pe(2.2, 4.6) * CH)) : -1;
          for (let rI = 0; rI < rowsShow; rI++) {
            const j = S.ratio === 4 ? rI : rI * 8;
            const y = my + rI * chh;
            if (S.ratio === 4) {
              const lbl = rI < 4 ? `i−1 · ${rI}` : `i · ${rI - 4}`;
              V.text(g, lbl, mx - 8, y + chh / 2 + 4, { size: 11, font: 'mono', color: rI < 4 ? C.b : C.a, align: 'right' });
            } else if (rI % 4 === 0) V.text(g, 'j=' + j, mx - 8, y + chh / 2 + 4, { size: 10.5, font: 'mono', color: C.muted, align: 'right' });
            for (let c = 0; c < CH; c++) {
              const x = mx + c * cwid;
              const z = D.Z[j][c], wv = D.Wm[j][c];
              const mixT = ph === 0 ? 0 : ph === 1 ? (c <= colSweep ? 1 : 0) : 1;
              const colZ = V.cmap(V.clamp(z / 3, -1, 1), 'div');
              const colW = V.cmap(Math.min(1, wv * (S.ratio === 4 ? 2.4 : 14)), 'seq');
              g.fillStyle = mixT ? colW : colZ;
              V.rr(g, x + 1, y + 1, cwid - 2, chh - 2, S.ratio === 4 ? 4 : 2); g.fill();
              if (S.ratio === 4) {
                const txt = mixT ? sf(wv, 2) : sf(z, 1);
                const tcol = mixT ? cellText(C, Math.min(1, wv * 2.4), 'seq') : cellText(C, V.clamp(z / 3, -1, 1), 'div');
                V.text(g, txt, x + cwid / 2, y + chh / 2 + 0.5, { size: mob ? 10 : 11, font: 'mono', align: 'center', base: 'middle', color: tcol });
              }
            }
          }
          const mBot = my + rowsShow * chh;
          if (S.ratio === 4) {
            V.line(g, mx - 2, my + 4 * chh, mx + CH * cwid + 2, my + 4 * chh, { color: C.c, lw: 1.5, dash: [4, 3] });
            V.text(g, mob ? 'i−1·j: previous block, projection a · i·j: current, projection b' : 'i−1·j = token from the previous block (projection a) · i·j = from the current block (projection b)', mob ? 14 : mx, mBot + 30, { size: 10.5, font: 'mono', color: C.muted, maxW: mob ? w - 28 : CH * cwid + 230 });
          } else V.text(g, '… 128 positions in the block (1 in 8 shown), no overlap', mob ? 14 : mx, mBot + 30, { size: 10.5, font: 'mono', color: C.muted, maxW: mob ? w - 28 : CH * cwid + 200 });
          if (colSweep >= 0) { g.strokeStyle = C.c; g.lineWidth = 2; V.rr(g, mx + colSweep * cwid - 1, my - 2, cwid + 2, rowsShow * chh + 4, 5); g.stroke(); }
          for (let c = 0; c < CH; c++) V.text(g, 'c' + c, mx + c * cwid + cwid / 2, mBot + 14, { size: 10.5, font: 'mono', align: 'center', color: C.muted });
          // output
          const oy = mBot + 44;
          if (ph >= 2) {
            const a = ph === 2 ? pe(4.8, 6.4) : 1;
            V.text(g, 'c_i', mx - 8, oy + 16, { size: 12, font: 'mono', color: C.c, align: 'right', weight: 600 });
            for (let c = 0; c < CH; c++) {
              const x = mx + c * cwid;
              if (ph === 2 && a < 1) { const k = Math.floor((u * 6) % rowsShow); const [ax, ay] = [x + cwid / 2, my + k * chh + chh / 2]; V.arrow(g, ax, ay, x + cwid / 2, oy + 2, { color: V.alpha(C.c, 0.35), lw: 1, head: 5 }); }
              V.box(g, x + 1, oy, cwid - 2, 26, { r: 5, fill: V.alpha(C.c, 0.2 * a + 0.05), stroke: V.alpha(C.c, a), lw: 1.5 });
              V.text(g, sf(D.out[c], 2), x + cwid / 2, oy + 13.5, { size: mob ? 10 : 11, font: 'mono', align: 'center', base: 'middle', color: C.ink, alpha: a });
            }
          }
          // pipeline
          const chips = ['RMSNorm', 'RoPE (1st-token pos.)', 'FP8 448 + BF16 64', '584 B in the cache'];
          let cx, cy;
          if (!mob) { cx = mx + CH * cwid + 34; cy = my + 14; } else { cx = 14; cy = oy + 44; }
          const a3 = ph === 3 ? pe(7.2, 8.6) : ph > 3 ? 1 : 0;
          if (!mob) V.text(g, 'after the pooling', cx, cy - 4, { size: 11, font: 'mono', color: C.muted });
          chips.forEach((s, i) => {
            const on = ph === 3 && a3 * chips.length > i;
            if (!mob) { pill(g, cx, cy + 22 + i * 34, s, on ? C.c : C.muted, { a: on ? 0.25 : 0.08, color: on ? C.ink : C.muted }); if (i < chips.length - 1) V.arrow(g, cx + 20, cy + 32 + i * 34, cx + 20, cy + 46 + i * 34, { color: V.alpha(C.muted, 0.6), lw: 1.2, head: 5 }); }
          });
          if (mob) {
            let xx = cx;
            chips.forEach((s, i) => { const on = ph === 3 && a3 * chips.length > i; const wd = tw(g, s, 10, 'mono', 600) + 14; if (xx + wd > w - 10) { xx = cx; cy += 26; } pill(g, xx, cy, s, on ? C.c : C.muted, { size: 10, a: on ? 0.25 : 0.08, color: on ? C.ink : C.muted }); xx += wd + 6; });
          }
          if (!mob) {
            const sy = cy + 22 + chips.length * 34 + 10;
            V.text(g, 'the layer\'s compressed cache', cx, sy, { size: 11, font: 'mono', color: C.muted });
            for (let k = 0; k < 7; k++) {
              const isNew = k === 6 && ph === 3 && a3 >= 1;
              const filled = k < 6 || isNew;
              V.box(g, cx + k * 26, sy + 10, 22, 22, { r: 4, fill: filled ? V.alpha(isNew ? C.c : C.b, isNew ? 0.6 : 0.4) : 'transparent', stroke: V.alpha(isNew ? C.c : C.b, 0.8), lw: 1 });
            }
            V.text(g, `${S.ratio === 4 ? '1 entry every 4 tokens' : '1 entry every 128 tokens'}`, cx, sy + 52, { size: 11, font: 'mono', color: C.ink });
          }
        } else {
          // incremental decode
          const r = S.ratio;
          const dt = r === 4 ? 0.7 : 0.075;
          const pos = Math.floor(tt / dt) + (r === 4 ? 16 : 256);
          const inBlk = (pos % r) + 1;
          const emit = (pos + 1) % r === 0;
          const x0 = mob ? 14 : 30, x1 = w - (mob ? 14 : 30);
          const ty = 44;
          V.text(g, mob ? `pos = ${nf(pos)} · (pos+1) mod ${r} = ${(pos + 1) % r}` : `decode: token pos = ${nf(pos)} · (pos + 1) mod ${r} = ${(pos + 1) % r}`, x0, 22, { size: 12, font: 'mono', weight: 600, color: C.ink });
          // partial state
          const slots = r === 4 ? 4 : 128;
          const sw = (x1 - x0) / (r === 4 ? 8 : 128);
          const stLbl = r === 4 ? 'kv_state / score_state: 4 tokens from the previous block (projection a) + 4 from the current block (projection b)' : 'kv_state / score_state: 128 positions of the current block';
          const stH = textBlock(g, stLbl, x0, ty + 4, x1 - x0, { size: 11, font: 'mono', color: C.muted, lh: 14, maxLines: 2 });
          const sy = ty + stH;
          if (r === 4) {
            for (let k = 0; k < 8; k++) {
              const cur = k >= 4;
              const filled = !cur || k - 4 < inBlk;
              const col = cur ? C.a : C.b;
              V.box(g, x0 + k * sw + 2, sy, sw - 4, 34, { r: 6, fill: filled ? V.alpha(col, 0.5) : 'transparent', stroke: V.alpha(col, 0.8), lw: 1.2 });
              V.text(g, cur ? `b·${k - 4}` : `a·${k}`, x0 + k * sw + sw / 2, sy + 18, { size: 11, font: 'mono', align: 'center', base: 'middle', color: C.ink });
            }
          } else {
            for (let k = 0; k < slots; k++) { g.fillStyle = k < inBlk ? V.alpha(C.a, 0.75) : V.alpha(C.muted, 0.12); g.fillRect(x0 + k * sw, sy, Math.max(1, sw - 0.6), 34); }
            V.text(g, `${inBlk} / 128`, x1, sy + 50, { size: 12, font: 'mono', color: C.a, align: 'right', weight: 600 });
          }
          // emission
          const ey = sy + 72;
          const just = r === 4 ? emit : inBlk <= 12;
          const flash = r === 4 ? (emit ? 1 - (tt % dt) / dt : 0) : Math.max(0, 1 - inBlk / 12);
          V.text(g, 'compressed cache', x0, ey - 6, { size: 11, font: 'mono', color: C.muted });
          const nE = Math.floor((pos + 1) / r);
          const shown = Math.min(nE, mob ? 7 : 14);
          for (let k = 0; k < shown; k++) {
            const isNew = k === shown - 1 && just;
            const x = x0 + k * 26;
            if (isNew) V.glow(g, x + 11, ey + 13, 22, C.c, flash);
            V.box(g, x, ey + 2, 22, 22, { r: 4, fill: V.alpha(isNew ? C.c : C.b, isNew ? 0.7 : 0.4), stroke: V.alpha(isNew ? C.c : C.b, 0.9), lw: 1 });
          }
          V.text(g, `${nf(nE)} entries`, x0 + shown * 26 + 8, ey + 17, { size: 11.5, font: 'mono', color: C.ink });
          if (just && !mob) pill(g, x1, ey + 13, 'emits 1 entry!', C.c, { align: 'right', size: 11 });
          // window
          const wy = ey + 50;
          const tokW = Math.min(12, (x1 - x0) / 40);
          V.text(g, 'recent tokens: the 128-token window covers the block in formation', x0, wy, { size: 11, font: 'mono', color: C.muted, maxW: x1 - x0 });
          const ntk = Math.floor((x1 - x0) / tokW);
          for (let k = 0; k < ntk; k++) {
            const j = pos - (ntk - 1 - k);
            const inCur = j > pos - inBlk;
            g.fillStyle = inCur ? V.alpha(C.d, 0.85) : V.alpha(C.a, 0.3);
            g.fillRect(x0 + k * tokW + 0.5, wy + 10, tokW - 1, 16);
          }
          V.line(g, x0, wy + 32, x1, wy + 32, { color: C.e, lw: 2 });
          V.text(g, 'SWA window: last 128 raw tokens (excerpt)', x1, wy + 46, { size: 11, font: 'mono', color: C.e, align: 'right' });
          V.text(g, 'amber = block in formation (no compressed entry yet)', x0, wy + 46 + (mob ? 16 : 0), { size: 11, font: 'mono', color: C.d, maxW: mob ? x1 - x0 : (x1 - x0) * 0.55 });
          const ny = wy + (mob ? 84 : 76);
          const idle = r - 1;
          const txt = r === 128 ? `C128: 1 emission every 128 steps, ${idle} of 128 with no work. #48957 writes only the partial state and returns early (~2× on the kernel, upstream), but only on piecewise or eager steps; under a FULL graph (the recipe's uniform decode) the compressor runs in full.` : 'C4: 1 emission every 4 steps; each entry mixes 8 positions (current + previous block).';
          textBlock(g, txt, x0, ny, x1 - x0, { size: 12, color: C.ink, maxLines: 5 });
        }
      });
    }

    /* ================================================================
       Figure 8 — weighing the bytes
       ================================================================ */
    function figBalanca() {
      const fig = root.querySelector('#fig-balanca');
      const ctl = UI.controls(fig);
      const GiB = 1073741824;
      const MODELS = {
        ds: { name: 'DeepSeek-V4-Flash', pool: 17.04 * GiB, poolTxt: '17.04 GiB (boot)', served: 1048576, rep: 'per rank, replicated on both',
          layers: Array.from({ length: 46 }, (_, i) => (i < 2 || i > 42 ? 'swa' : i % 2 === 0 ? 'c4a' : 'c128a')),
          comps: [
            { l: 'KV C4A (21 × 584/4)', c: 'ds', bt: 3066 },
            { l: 'KV C128A (20 × 584/128)', c: 'b', bt: 91.25 },
            { l: 'indexer K (21 × 132/4)', c: 'd', bt: 693 },
            { l: 'windows (46 × 128 × 584 B)', c: 'e', fixed: 3440640 },
          ] },
        glm: { name: 'GLM-5.3-Flash', pool: 11 * GiB, poolTxt: '11 GiB (KV ceiling)', served: 850000, rep: 'per rank, latent replicated',
          layers: Array.from({ length: 46 }, (_, i) => (i === 45 ? 'dsa' : i % 4 === 3 ? 'dsa' : 'kda')),
          comps: [
            { l: 'fp8_ds_mla latent (11 × 656)', c: 'glm', bt: 7216 },
            { l: 'kpool indexer (11 × 33)', c: 'd', bt: 363 },
            { l: 'KDA state (34 × 2.34 MB)', c: 'muted', fixed: 79659008 },
            { l: 'DFlash2 window (5 × 2048 B)', c: 'e', perTokCap: 2048, bt: 10240 },
          ] },
        qwen: { name: 'Qwen3.8-Flash-Next', pool: 32.02 * GiB, poolTxt: '32.02 GiB (boot)', served: 262144, rep: 'per GPU, 2 KV heads split',
          layers: Array.from({ length: 49 }, (_, i) => (i === 48 ? 'qsa' : (i + 1) % 4 === 0 ? 'qsa' : 'gdn')),
          comps: [
            { l: 'FP8 KV (13 layers)', c: 'qwen', bt: 6656 },
            { l: 'indexer + page (measured)', c: 'd', bt: 2762 },
            { l: 'FP32 GDN state', c: 'muted', fixed: 56623104 },
          ] },
        nocomp: { name: 'Hypothesis: DS without compression', pool: 17.04 * GiB, poolTxt: '17.04 GiB (DS pool)', served: 1048576, rep: 'per rank',
          layers: Array.from({ length: 43 }, () => 'dense'),
          comps: [{ l: 'KV 584 B × 43 layers', c: 'red', bt: 25112 }] },
        gqa: { name: 'Hypothesis: GQA-8 in BF16', pool: 17.04 * GiB, poolTxt: '17.04 GiB (DS pool)', served: 1048576, rep: 'per rank',
          layers: Array.from({ length: 43 }, () => 'dense'),
          comps: [{ l: 'KV 8 × 128 × 2 × 2 B × 43', c: 'red', bt: 176128 }] },
      };
      const TYPES = { swa: ['e', 'window only'], c4a: ['ds', 'C4A'], c128a: ['b', 'C128A'], dsa: ['glm', 'MLA + DSA'], kda: ['muted', 'KDA (linear)'], qsa: ['qwen', 'QSA'], gdn: ['muted', 'GDN (linear)'], dense: ['red', 'dense'] };
      const S = { m: 'ds', lg: 20, seqs: 2 };
      let H = null;
      const rB = { t: null, s: null, f: null };
      UI.select(ctl, { label: 'Model', options: [{ v: 'ds', label: 'DeepSeek-V4-Flash' }, { v: 'glm', label: 'GLM-5.3-Flash' }, { v: 'qwen', label: 'Qwen3.8-Flash-Next' }, { v: 'nocomp', label: 'Hypothesis: DS without compression' }, { v: 'gqa', label: 'Hypothesis: GQA-8 BF16' }], value: 'ds', onChange: (v) => { S.m = v; if (H) H.redraw(); } });
      UI.slider(ctl, { label: 'Context', min: 10, max: 20, step: 0.25, value: 20, fmt: (v) => kfmt(Math.round(2 ** v)), onInput: (v) => { S.lg = v; if (H) H.redraw(); } });
      UI.slider(ctl, { label: 'Sequences', min: 1, max: 8, step: 1, value: 2, onInput: (v) => { S.seqs = v; if (H) H.redraw(); } });
      rB.t = UI.readout(ctl, 'Bytes/token', '—');
      rB.s = UI.readout(ctl, 'KV per sequence', '—');
      rB.f = UI.readout(ctl, 'Fit (physical)', '—');
      H = V.fig(fig, (g, w, h) => {
        const C = V.C();
        const col = (k) => (k === 'red' ? RED(C) : k === 'muted' ? C.muted : C[k]);
        const mob = w < 620;
        const M = MODELS[S.m];
        const n = Math.round(2 ** S.lg);
        const x0 = mob ? 12 : 24, x1 = w - (mob ? 12 : 24), W = x1 - x0;
        V.text(g, M.name + ' · layer map', x0, 20, { size: 12.5, weight: 600, color: C.ink });
        const lw = W / M.layers.length;
        M.layers.forEach((ty, i) => {
          const [ck] = TYPES[ty];
          const isM = (S.m === 'ds' && i > 42) || (S.m === 'glm' && i === 45) || (S.m === 'qwen' && i === 48);
          V.box(g, x0 + i * lw + 0.5, 30, lw - 1, 20, { r: 2, fill: V.alpha(col(ck), ck === 'muted' ? 0.28 : 0.8), stroke: isM ? C.c : null, lw: 1.2 });
        });
        const used = Array.from(new Set(M.layers));
        // layer-map legend: wraps to the next line when space runs out
        let lx = x0, lyL = 67;
        used.forEach((ty) => { const [ck, lbl] = TYPES[ty]; const cnt = M.layers.filter((x) => x === ty).length; const s = `${lbl} ×${cnt}`; const wd = tw(g, s, 11) + 26; if (lx + wd > x1 && lx > x0) { lx = x0; lyL += 16; } V.box(g, lx, lyL - 9, 10, 10, { r: 2, fill: V.alpha(col(ck), ck === 'muted' ? 0.4 : 0.85) }); V.text(g, s, lx + 14, lyL, { size: 11, color: C.muted }); lx += wd; });
        if (S.m === 'ds' || S.m === 'glm' || S.m === 'qwen') { const s = 'outline = MTP/drafter'; if (lx + tw(g, s, 11) > x1 && lx > x0) { lx = x0; lyL += 16; } V.text(g, s, lx, lyL, { size: 11, color: C.c }); }
        // one sequence
        const parts = M.comps.map((cp) => {
          let b;
          if (cp.fixed) b = cp.fixed;
          else if (cp.perTokCap) b = Math.min(n, cp.perTokCap) * cp.bt;
          else b = cp.bt * n;
          return { cp, b };
        });
        const perSeq = parts.reduce((s, x) => s + x.b, 0);
        const perTok = M.comps.filter((c) => !c.fixed && !c.perTokCap).reduce((s, c) => s + c.bt, 0);
        const sy = lyL + 29;
        let sy2 = sy;
        if (mob) { V.text(g, `one sequence of ${nf(n)} tokens: ${fmtB(perSeq)}`, x0, sy, { size: 12.5, font: 'mono', weight: 600, color: C.ink }); V.text(g, M.rep, x0, sy + 16, { size: 11, font: 'mono', color: C.muted }); sy2 = sy + 16; }
        else V.text(g, `one sequence of ${nf(n)} tokens (${M.rep}): ${fmtB(perSeq)}`, x0, sy, { size: 12.5, font: 'mono', weight: 600, color: C.ink });
        const barY = sy2 + 10, barH = 22;
        const scaleB = Math.max(perSeq, M.pool * 0.25);
        V.box(g, x0, barY, W, barH, { r: 6, fill: V.alpha(C.muted, 0.1) });
        let bx = x0;
        parts.forEach(({ cp, b }) => { const ww = (b / scaleB) * W; if (ww > 0.3) { g.fillStyle = V.alpha(col(cp.c), 0.85); g.fillRect(bx, barY, Math.min(ww, x1 - bx), barH); } bx += ww; });
        V.text(g, `scale: ${fmtB(scaleB)}`, x1, barY + barH + 13, { size: 10.5, font: 'mono', color: C.muted, align: 'right' });
        let ly = barY + barH + (mob ? 30 : 22);
        const colsL = mob ? 1 : 2, cwL = W / colsL;
        parts.forEach(({ cp, b }, i) => {
          const cx = x0 + (i % colsL) * cwL, cy = ly + Math.floor(i / colsL) * 18;
          V.box(g, cx, cy - 9, 10, 10, { r: 2, fill: V.alpha(col(cp.c), 0.85) });
          V.text(g, `${cp.l}: ${fmtB(b)}`, cx + 15, cy, { size: 11.5, font: 'mono', color: C.ink, maxW: cwL - 20 });
        });
        ly += Math.ceil(parts.length / colsL) * 18 + 20;
        // pool
        const tot = perSeq * S.seqs;
        const fits = tot <= M.pool;
        const fitN = Math.floor(M.pool / perSeq);
        V.text(g, `${S.seqs} × KV vs. a pool of ${M.poolTxt}`, x0, ly, { size: 12.5, font: 'mono', weight: 600, color: C.ink, maxW: W });
        const pY = ly + 10, pH = 30;
        V.box(g, x0, pY, W, pH, { r: 8, fill: V.alpha(C.muted, 0.12), stroke: V.alpha(C.muted, 0.35), lw: 1 });
        const segW = (perSeq / M.pool) * W;
        for (let k = 0; k < S.seqs; k++) {
          const sx = x0 + k * segW;
          if (sx >= x1) break;
          const ww = Math.min(segW, x1 - sx);
          g.fillStyle = V.alpha(fits ? C.a : RED(C), 0.35 + 0.35 * (k % 2)); g.fillRect(sx, pY + 3, Math.max(1, ww - 1), pH - 6);
        }
        if (!fits) { hatch(g, x1 - 40, pY, 40, pH, V.alpha(RED(C), 0.8), 5); }
        const msg = fits ? `✓ fits: takes up ${Math.round((100 * tot) / M.pool)}% of the pool` : `✕ doesn't fit: ${fmtB(tot)} > ${fmtB(M.pool)}`;
        V.text(g, msg, x0, pY + pH + 20, { size: 13, weight: 600, color: fits ? C.e : RED(C) });
        let ny = pY + pH + 40;
        if (n > M.served) { V.text(g, `⚠ above the served MAX_MODEL_LEN (${nf(M.served)})`, x0, ny, { size: 11.5, font: 'mono', color: C.d }); ny += 18; }
        const note = S.m === 'ds' ? 'physical cost; the allocator counts ≈ 7.8 KB/token (4 groups, same page): 2.22× at 1M' : S.m === 'glm' ? 'physical cost; the README counts 14.52 GiB per 1M request (≈ 7.4 GiB fixed in hybrid pages + 7.1 GiB per 1M)' : '';
        if (note) textBlock(g, note, x0, ny, W, { size: 11, font: 'mono', color: C.muted, maxLines: 3 });
        rB.t.set(nf(perTok) + ' B');
        rB.s.set(fmtB(perSeq));
        rB.f.set(fitN >= 1000 ? '≥ 1,000' : nf(fitN) + ' seq');
      }, { static: true });
    }

    /* ================================================================
       Figure 9 — top-k split across ranks
       ================================================================ */
    function figSP() {
      const fig = root.querySelector('#fig-sp-topk');
      const ctl = UI.controls(fig);
      const S = { tp: 2, k: 4, seed: 9 };
      let scores = [];
      const gen = () => { const r = V.rng(S.seed); scores = Array.from({ length: 24 }, (_, i) => 10 + Math.floor(r() * 89) + i * 0.001); };
      gen();
      UI.seg(ctl, { label: 'TP', options: [2, 3, 4], value: 2, onChange: (v) => { S.tp = v; } });
      UI.slider(ctl, { label: 'k', min: 2, max: 6, step: 1, value: 4, onInput: (v) => { S.k = v; } });
      UI.button(ctl, '🎲 shuffle scores', () => { S.seed = (S.seed * 31 + 7) % 1009; gen(); });
      const steps = [
        { t: 'Slice and score', d: 'Each rank takes a contiguous slice of the compressed keys, aligned to the page (here, pages of 2 keys), and scores only its own share: 1/TP of the work. In stock, both ranks would score all of them.' },
        { t: 'Local top-k', d: 'Each rank picks the \\(k\\) best keys from its own slice (pink outline), with <code>top_k_per_row_prefill</code>.' },
        { t: 'all_gather of (score, id)', d: 'The candidates travel: every rank receives TP × k (score, global id) pairs. At real scale, TP2 × 512 × 8 B = 8 KiB per query.' },
        { t: 'Stable, exact merge', d: 'The TP × k candidates are sorted and the \\(k\\) best are kept (<code>stable_topk_from_gathered_candidates_cutedsl</code>), written straight into the <code>topk_indices_buffer</code>. The badge compares against brute force.' },
      ];
      const st = UI.stepper(fig, steps, { autoplay: 4200 });
      V.fig(fig, (g, w, h) => {
        const C = V.C();
        const mob = w < 620;
        const N = mob ? 16 : 24;
        const sc = scores.slice(0, N);
        const tp = S.tp, k = S.k;
        const page = 2;
        const per = Math.ceil(N / tp / page) * page;
        const rankOf = (i) => Math.min(tp - 1, Math.floor(i / per));
        const rc = [C.a, C.b, C.d, C.f];
        const x0 = mob ? 12 : 30, x1 = w - (mob ? 12 : 30);
        const bw = (x1 - x0) / N;
        const barTop = 40, barH = mob ? 90 : 110;
        const el = st.elapsed(), s = st.i;
        const local = [];
        for (let r = 0; r < tp; r++) { const ids = []; for (let i = 0; i < N; i++) if (rankOf(i) === r) ids.push(i); ids.sort((a, b) => sc[b] - sc[a]); local.push(ids.slice(0, k)); }
        const locSet = new Set(local.flat());
        const glob = Array.from({ length: N }, (_, i) => i).sort((a, b) => sc[b] - sc[a]).slice(0, k);
        const cands = local.flat().slice().sort((a, b) => sc[b] - sc[a]);
        const merged = cands.slice(0, k);
        const same = merged.length === glob.length && merged.every((x, i) => x === glob[i]);
        V.text(g, mob ? 'scores for one query (1 bar = 1 key)' : 'indexer scores for one query (one bar per compressed key)', x0, 18, { size: 11.5, font: 'mono', color: C.muted });
        for (let r = 0; r < tp; r++) {
          const a = r * per, b = Math.min(N, (r + 1) * per);
          if (a >= N) continue;
          const xa = x0 + a * bw, xb = x0 + b * bw;
          V.box(g, xa + 1, barTop - 6, xb - xa - 2, barH + 12, { r: 8, fill: V.alpha(rc[r], 0.06), stroke: V.alpha(rc[r], 0.5), lw: 1 });
          const lbl = tw(g, `rank ${r} · keys ${a}–${b - 1}`, 11, 'mono', 600) < xb - xa - 8 ? `rank ${r} · keys ${a}–${b - 1}` : `r${r} · ${a}–${b - 1}`;
          V.text(g, lbl, xa + 6, barTop + barH + 22, { size: 11, font: 'mono', color: rc[r], weight: 600 });
        }
        for (let i = 0; i < N; i++) {
          const r = rankOf(i);
          const hh = (sc[i] / 100) * barH;
          const x = x0 + i * bw + 2, y = barTop + barH - hh;
          const isLoc = s >= 1 && locSet.has(i);
          V.box(g, x, y, bw - 4, hh, { r: 3, fill: V.alpha(rc[r], s >= 1 && !isLoc ? 0.3 : 0.85) });
          if (isLoc) { g.strokeStyle = C.c; g.lineWidth = 2; V.rr(g, x - 1.5, y - 1.5, bw - 1, hh + 3, 4); g.stroke(); }
          if (bw >= 20) V.text(g, String(Math.round(sc[i])), x + (bw - 4) / 2, y - 4, { size: 10, font: 'mono', align: 'center', color: C.muted });
        }
        // gather + merge
        const my = barTop + barH + (mob ? 58 : 60);
        const nC = cands.length;
        const cw2 = Math.min(mob ? 30 : 44, (x1 - x0) / nC - 3);
        if (s >= 2) {
          V.text(g, `all_gather: ${tp} × ${k} = ${nC} candidates (score, id)`, x0, my - 8, { size: 11.5, font: 'mono', color: C.ink });
          const prog = s === 2 ? V.ease.inOut(Math.min(1, el / 1.6)) : 1;
          const sortP = s >= 3 ? V.ease.inOut(Math.min(1, el / 1.2)) : 0;
          const lflat = local.flat();
          const slotX = (j) => x0 + j * (cw2 + 3) + cw2 / 2;
          lflat.forEach((id) => {
            const r = rankOf(id);
            const j = V.lerp(lflat.indexOf(id), cands.indexOf(id), sortP);
            const sx = x0 + id * bw + bw / 2, sy = barTop + barH - (sc[id] / 100) * barH;
            const tx = slotX(j), ty = my + 4 - Math.sin(sortP * Math.PI) * 10 * (lflat.indexOf(id) % 2 ? 1 : -1);
            const x = V.lerp(sx, tx, prog), y = V.lerp(sy, ty, prog);
            const jj = cands.indexOf(id);
            const keep = s >= 3 && jj < k && sortP > 0.9;
            const a = s >= 3 && jj >= k && sortP > 0.9 ? 0.3 : 1;
            V.box(g, x - cw2 / 2, y, cw2, 32, { r: 5, fill: V.alpha(rc[r], 0.25 * a + 0.05), stroke: V.alpha(keep ? C.c : rc[r], a), lw: keep ? 2 : 1 });
            V.text(g, String(Math.round(sc[id])), x, y + 12, { size: mob ? 10 : 11, font: 'mono', align: 'center', base: 'middle', color: C.ink, alpha: a });
            V.text(g, '#' + id, x, y + 25, { size: mob ? 9.5 : 10, font: 'mono', align: 'center', base: 'middle', color: C.muted, alpha: a });
          });
        }
        if (s >= 3) {
          const vy = my + 58;
          V.text(g, `merge → top-${k}: ${merged.map((x) => '#' + x).join(' ')}`, x0, vy, { size: 12, font: 'mono', weight: 600, color: C.c, maxW: x1 - x0 });
          V.text(g, `brute force: ${glob.map((x) => '#' + x).join(' ')}`, x0, vy + 18, { size: 12, font: 'mono', color: C.muted, maxW: x1 - x0 });
          pill(g, x1, vy + 6, same ? '✓ identical' : '✕ different', same ? C.e : RED(C), { align: 'right', size: 12 });
        }
        const fy = h - 12;
        V.text(g, `each rank scores ${per} of ${N} keys (1/${tp})`, x0, fy, { size: 11, font: 'mono', color: C.muted });
      });
    }

    /* ================================================================
       Figure 10 — the indexer's traffic lights
       ================================================================ */
    function figSemaforo() {
      const fig = root.querySelector('#fig-semaforo');
      const ctl = UI.controls(fig);
      const S = { lg: 10.5, cap: false, bnd: false, full: false, sp: false };
      UI.slider(ctl, { label: 'Context', min: 8, max: 20, step: 0.25, value: 10.5, fmt: (v) => nf(Math.round(2 ** v)) + ' tok', onInput: (v) => { S.lg = v; } });
      UI.toggle(ctl, { label: 'capturing a CUDA graph', value: false, onChange: (v) => { S.cap = v; } });
      UI.toggle(ctl, { label: 'step crosses a 128 boundary', value: false, onChange: (v) => { S.bnd = v; } });
      UI.seg(ctl, { label: 'Graph mode', options: [{ v: false, label: 'PIECEWISE (batch with prefill)' }, { v: true, label: 'FULL (uniform decode)' }], value: false, onChange: (v) => { S.full = v; } });
      UI.toggle(ctl, { label: 'SP indexer on', value: false, onChange: (v) => { S.sp = v; } });
      function states() {
        const n = Math.round(2 ** S.lg), keys = Math.floor(n / 4);
        const st = [];
        if (keys <= 512 && !S.cap) st.push({ c: 'g', t: 'C4A · indexer', h: 'Shortcut #49486', d: `n/4 = ${nf(keys)} ≤ 512: all of them would be picked. Fills in indices 0..⌊n/4⌋−1 (the entries already closed) without scoring; the compressor still writes the K cache.` });
        else if (keys <= 512) st.push({ c: 'a', t: 'C4A · indexer', h: 'Guard #52492', d: 'Capturing a CUDA graph: it scores anyway, otherwise the graph would freeze the shortcut and repeat it on long prefixes.' });
        else st.push({ c: 'r', t: 'C4A · indexer', h: 'Scores and picks top-512', d: `n/4 = ${nf(keys)} compressed keys > 512: FP8/FP4 logits + top-k over all of them.` });
        if (S.full) st.push({ c: 'a', t: 'C128A · compressor', h: 'FULL graph: always runs', d: 'The whole step is one graph: the if was decided at capture and shortcut #48957 stays off. This is the uniform decode of the DS recipe.' });
        else if (S.bnd) st.push({ c: 'r', t: 'C128A · compressor', h: 'Emits an entry', d: 'Some request crosses a 128 boundary: compress → RMSNorm → RoPE → quant → write.' });
        else st.push({ c: 'g', t: 'C128A · compressor', h: 'Skips (#48957)', d: 'Piecewise: attention runs eagerly and the if is re-evaluated. With no 128 boundary, it writes only the partial state and returns early (~2× on the kernel, upstream).' });
        st.push({ c: 'a', t: 'Dense prefill', h: 'Dormant (#48407)', d: 'No dense MHA route in this fork: short prefill also goes through the indexer and the sparse kernel (in GLM: supports_dense_mha_prefill = False).' });
        if (!S.sp) st.push({ c: 'r', t: 'SP indexer (prefill)', h: 'Replicated', d: `Each rank scores all ${nf(keys)} keys × 64 heads: work duplicated on both ranks.` });
        else if (keys >= 8192) st.push({ c: 'g', t: 'SP indexer (prefill)', h: 'Split across ranks', d: `n/4 = ${nf(keys)} ≥ MIN_KEYS 8,192: per-rank slices + all-gather + exact merge.` });
        else st.push({ c: 'a', t: 'SP indexer (prefill)', h: 'Below MIN_KEYS', d: `n/4 = ${nf(keys)} < 8,192 (32K tokens): this chunk uses the stock replicated path.` });
        return st;
      }
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const mob = w < 620;
        const st = states();
        const colOf = (c) => (c === 'g' ? C.e : c === 'a' ? C.d : RED(C));
        const drawLight = (x, y, c, sz) => {
          // traffic-light housing: dark in both themes, derived from the theme colors
          const casing = C.dark ? V.mix(C.bg, C.ink, 0.07) : V.mix(C.ink, C.bg, 0.16);
          V.box(g, x - sz * 0.75, y - sz * 0.2, sz * 1.5, sz * 3.9, { r: sz * 0.5, fill: casing, stroke: V.alpha(C.muted, 0.5), lw: 1 });
          ['r', 'a', 'g'].forEach((k, i) => {
            const cy = y + sz * 0.55 + i * sz * 1.2;
            const on = k === c;
            if (on) V.glow(g, x, cy, sz * 1.3, colOf(k), 0.7 + 0.25 * Math.sin(t * 4));
            V.dot(g, x, cy, sz * 0.45, on ? colOf(k) : V.alpha(colOf(k), 0.18));
          });
        };
        // the car's schedule: drive to each light's stop line, wait if it isn't green
        const seg = 1.25;
        const dwell = st.map((s) => (s.c === 'r' ? 1.1 : s.c === 'a' ? 0.55 : 0));
        const total = seg * (st.length + 1) + dwell.reduce((a, b) => a + b, 0);
        let u = t % total, pos = st.length + 1;
        for (let i = 0; i <= st.length; i++) {
          if (u < seg) { pos = i + u / seg; break; }
          u -= seg;
          if (i < st.length) { if (u < dwell[i]) { pos = i + 1; break; } u -= dwell[i]; }
        }
        const along = (stop, start, end) => {
          if (pos <= 1) return V.lerp(start, stop(0), V.smooth(pos));
          const i = Math.floor(pos) - 1, f = pos - Math.floor(pos);
          const a = stop(i), b = i + 1 < st.length ? stop(i + 1) : end;
          return V.lerp(a, b, V.smooth(f));
        };
        if (!mob) {
          const x0 = 16, x1 = w - 16, roadY = 104;
          const colW = (x1 - x0) / st.length;
          const lightX = (i) => x0 + colW * i + colW * 0.78;
          g.fillStyle = V.alpha(C.muted, 0.16); g.fillRect(x0, roadY - 12, x1 - x0, 24);
          V.line(g, x0, roadY, x1, roadY, { color: V.alpha(C.ink, 0.35), lw: 1.5, dash: [10, 10], dashOffset: -t * 30 });
          st.forEach((s, i) => {
            const xl = x0 + colW * i;
            drawLight(lightX(i), 16, s.c, 12);
            V.line(g, lightX(i) - 4, roadY - 12, lightX(i) - 4, roadY + 12, { color: V.alpha(colOf(s.c), 0.85), lw: 3 });
            V.text(g, s.t, xl + 10, roadY + 36, { size: 12, font: 'mono', color: C.muted, maxW: colW - 16 });
            V.text(g, s.h, xl + 10, roadY + 56, { size: 14, weight: 600, color: colOf(s.c), maxW: colW - 16 });
            textBlock(g, s.d, xl + 10, roadY + 78, colW - 20, { size: 12, color: C.ink, maxLines: 8 });
          });
          const X = along((i) => lightX(i) - 62, x0 - 60, x1 + 8);
          g.save(); g.beginPath(); g.rect(x0, roadY - 14, x1 - x0, 28); g.clip();
          V.box(g, X, roadY - 10, 52, 20, { r: 8, fill: C.c, stroke: V.alpha(C.ink, 0.3), lw: 1 });
          V.text(g, 'token', X + 26, roadY + 1, { size: 11, font: 'mono', weight: 600, align: 'center', base: 'middle', color: onInk(C) });
          g.restore();
        } else {
          const roadX = 22, y0 = 12, y1 = h - 12;
          const rowH = (y1 - y0) / st.length;
          const lineY = (i) => y0 + rowH * i + rowH * 0.45;
          g.fillStyle = V.alpha(C.muted, 0.16); g.fillRect(roadX - 10, y0, 20, y1 - y0);
          V.line(g, roadX, y0, roadX, y1, { color: V.alpha(C.ink, 0.35), lw: 1.5, dash: [8, 8], dashOffset: -t * 25 });
          st.forEach((s, i) => {
            const y = y0 + rowH * i;
            drawLight(roadX + 34, y + 10, s.c, 8);
            V.line(g, roadX - 10, lineY(i), roadX + 10, lineY(i), { color: V.alpha(colOf(s.c), 0.85), lw: 3 });
            const tx = roadX + 56, tW = w - tx - 10;
            V.text(g, s.t, tx, y + 14, { size: 11, font: 'mono', color: C.muted, maxW: tW });
            V.text(g, s.h, tx, y + 31, { size: 13, weight: 600, color: colOf(s.c), maxW: tW });
            textBlock(g, s.d, tx, y + 48, tW, { size: 11, color: C.ink, maxLines: 4, lh: 14 });
          });
          const Y = along((i) => lineY(i) - 26, y0 - 30, y1 + 4);
          g.save(); g.beginPath(); g.rect(roadX - 12, y0, 24, y1 - y0); g.clip();
          V.box(g, roadX - 9, Y, 18, 22, { r: 6, fill: C.c });
          g.restore();
        }
      });
    }
  },
});

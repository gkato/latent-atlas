LA.chapter({
  id: 'prefill-decode-kv',
  part: 2,
  order: 10,
  title: 'Prefill, decode and the KV cache',
  navTitle: 'Prefill, decode & KV',
  subtitle: 'Why reading the prompt is fast and writing the answer is slow, what the KV cache stores, how much each context token costs and how to measure the two phases.',
  minutes: 60,
  recipes: ['glm', 'qwen', 'ds'],
  tags: ['prefill', 'decode', 'KV cache', 'roofline', 'arithmetic intensity', 'TTFT', 'TPOT', 'batching', 'GEMV'],
  goals: [
    'Explain why <b>prefill</b> is compute-bound and <b>decode</b> is bandwidth-bound, with GEMM vs. GEMV and the GB10 <b>roofline</b> (ridge ≈ 458 FLOP/B)',
    'Predict the decode ceiling of a recipe by adding up the <b>bytes read per step</b> and dividing by the bandwidth',
    'Compute <b>KV bytes per token</b> and per sequence (\\(L_{kv} \\times n_{kv} \\times (d_k+d_v) \\times b\\)) and say what happens to them under TP',
    'Read TTFT, TPOT, per-stream and aggregate tok/s, understand how <b>batching</b> amortizes the weight reads, and why context and concurrency compete for the same pool',
  ],
  prereqs: ['geracao-amostragem', 'posicao-rope'],
  summary: [
    'Prefill processes the whole prompt in a few large GEMMs (\\(M\\) = thousands of rows); decode generates one token per sequence per step (\\(M = B(1+k)\\), from 1 to a few dozen).',
    'Arithmetic intensity is \\(I \\approx 2M/b_w\\) while the activation bytes are small. On the GB10, with ~125 TFLOPS BF16 and 273 GB/s, the ridge sits at ≈458 FLOP/B: decode (I ≈ 1–4) is memory-bound. In an MoE, each expert only receives \\(Mk/D\\) rows; with 7,168-token chunks (~199 rows per expert) GLM <b>reaches</b> the ridge in prefill: I ≈ 460–500 counting the BF16 activations (the weights-only approximation would give ~800).',
    'Decode step time ≈ bytes read ÷ effective bandwidth + fixed costs; tok/s ≈ tokens per step (accepted + 1) ÷ step time. Qwen: ≈9.2 GiB per GPU → 40–42 ms of streaming → ceiling of ~75–80 tok/s with τ ≈ 3.2; the measured value (52.1 tok/s, real step of ~60 ms) shows that ~1/3 of the step is fixed cost.',
    'The KV cache stores the K and V of every past token because, thanks to the causal mask, they never change: without it, each step would redo the full forward pass of the previous tokens (n(n+1)/2 token-forwards instead of n). The trade-off: it grows with the context and is reread at every step.',
    'Bytes per token = \\(L_{kv} \\times n_{kv} \\times (d_k+d_v) \\times b\\) + fixed state per sequence. Qwen 26,624 B (BF16) / 13,312 B (FP8); GLM 7,579 B per rank + ~76 MiB of KDA state; DeepSeek ≈3,850 B + ≈3.44 MB. GQA splits under TP; MLA/MQA is replicated.',
    'There is a single KV pool: \\(\\sum c_i \\le\\) Pool ÷ bytes per token. <code>--max-model-len</code> and <code>--max-num-seqs</code> are ceilings, not reservations; the KV dtype changed Qwen\'s capacity by +71%.',
    'Batching splits the weight read: in Qwen, going from 1 → 8 streams multiplied the aggregate by 3.8× while each stream dropped to 49%. In an MoE the ride is less generous, because more tokens light up more distinct experts.',
  ],
  quiz: [
    { q: 'Why is single-user decode bound by memory bandwidth on the GB10?', o: ['Because attention is quadratic in the context', 'Because each weight read takes part in a single multiplication (GEMV): I ≈ 2/b_w FLOP/B, far below the ~458 ridge', 'Because decode uses FP32', 'Because the KV cache lives on the CPU'], a: 1, e: 'With M = 1, the projection does 2KN FLOPs over b<sub>w</sub>·KN bytes of weights: I ≈ 1 FLOP/B in BF16 and ≈ 4 with 4-bit weights. The GB10 ridge (~125 TFLOPS ÷ 273 GB/s) sits at ~458. Far from it, the GPU spends the step waiting for bytes.' },
    { q: 'Qwen stores KV per token in 13 layers, with 2 KV heads of dimension 256. How many bytes per token does that give with FP8 KV, before dividing by TP?', o: ['6,656', '13,312', '26,624', '53,248'], a: 1, e: '13 × 2 heads × (256 + 256) × 1 byte = 13,312 B. In BF16 it would be 26,624. With TP=2, each GPU keeps one head, so 6,656 B per GPU in FP8. The measured pool gives 9,418 B per GPU, because the indexer, the state and page rounding also count.' },
    { q: 'In a 1,024-token GLM prefill chunk (288 experts, top-8, 4-bit weights), why is the MoE still memory-bound?', o: ['Because prefill never uses tensor cores', 'Because each expert receives only ~28 rows: I ≈ 2 × 28 / 0.5 ≈ 112 FLOP/B, below the ridge', 'Because the prefill KV cache is read 1,024 times', 'Because the indexer stalls the GPU'], a: 1, e: 'The 1,024 × 8 = 8,192 choices spread over 288 experts, ~28 rows each. With half a byte per weight, I ≈ 112 < 458 (≈100 counting the activations). All experts are read in every chunk (~75.6 GB per GPU, estimate). With 7,168 tokens per chunk, ~199 rows per expert: the weights-only approximation would give ~800, but with r ≈ 200 the BF16 activations weigh almost as much as the 4-bit weights, and the full calculation gives ~460–500 FLOP/B. The MoE reaches the ridge, without crossing it by a wide margin.' },
    { q: 'In Qwen (day-0 lane, prose), the aggregate rose from 54.4 to 207.0 tok/s going from 1 to 8 streams, and each stream dropped to 26.7. What explains the aggregate growing?', o: ['The KV cache gets smaller with more streams', 'The weights read at every step are shared by all streams; the marginal cost of each stream (KV, state, extra experts) is much smaller than the weight read', 'Speculation accepts more tokens with more streams', 'Prefill speeds up'], a: 1, e: 'With the step dominated by the weight read (T<sub>0</sub>), each extra stream only adds its marginal cost (T<sub>1</sub>). The aggregate B·τ/(T<sub>0</sub> + B·T<sub>1</sub>) rises (3.8×) while the step gets longer for everyone (per stream −51%).' },
    { q: 'With TP=2, does the per-GPU KV of GLM (MLA) and DeepSeek (compressed MQA) drop by half, as in Qwen?', o: ['Yes, TP always splits the KV', 'No: there is a single KV “head” (the latent), which cannot be sliced; each rank keeps a full copy', 'Only in DeepSeek', 'Only if the KV is FP8'], a: 1, e: 'GQA splits the KV heads among the GPUs (Qwen has 2, one per GPU). MLA and MQA store one latent vector per token and layer, replicated on every rank: GLM stores 7,579 B per token on each rank. The TP gain for the KV comes indirectly: fewer weights per GPU leave more memory for the pool.' },
    { q: 'In DeepSeek\'s 0731 sweep, a 131,072-token prompt had a TTFT of 78.75 s and a decode rate of 65.2 tok/s. What is the prefill tok/s, and how much of the clock for a 512-token answer is prefill?', o: ['~65 tok/s; ~10%', '~1,665 tok/s; ~90%', '~16,600 tok/s; ~50%', '~1,665 tok/s; ~10%'], a: 1, e: 'Prefill tok/s = P/TTFT = 131,072/78.75 ≈ 1,665. Decoding 512 tokens takes ~511/65.2 ≈ 7.8 s, so the total clock is ~86.6 s, of which ~91% is prefill. That is why TTFT dominates the experience with a long prompt and a short answer.' },
  ],
  mount(root, LA) {
    const V = LA.viz, UI = LA.ui;
    const $ = (s) => root.querySelector(s);

    /* =============================== helpers =============================== */
    const FC = {};
    const setFont = (g, sz, f = 'sans', wt = 400) => { const k = sz + '|' + f + '|' + wt; g.font = FC[k] || (FC[k] = V.font(sz, f, wt)); };
    const tw = (g, s, sz = 12, f = 'sans', wt = 400) => { setFont(g, sz, f, wt); return g.measureText(String(s)).width; };
    const nf = (v, d = 0) => Number(v).toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d });
    const onInk = () => (V.C().dark ? '#00210F' : '#ffffff');
    const clamp = V.clamp, lerp = V.lerp;
    function fitH(fig, hd, hm, bp = 620) {
      const st = fig.querySelector('.la-fig-stage');
      const apply = () => { const w = fig.clientWidth || 800; const want = (w < bp ? hm : hd) + 'px'; if (st.style.height !== want) st.style.height = want; };
      apply();
      window.addEventListener('resize', apply);
      LA.onCleanup(() => window.removeEventListener('resize', apply));
    }
    function wrapLines(g, s, maxW, sz, f = 'sans', wt = 400) {
      const words = String(s).split(' ');
      const out = []; let line = '';
      words.forEach((wd) => { const t2 = line ? line + ' ' + wd : wd; if (tw(g, t2, sz, f, wt) > maxW && line) { out.push(line); line = wd; } else line = t2; });
      if (line) out.push(line);
      return out;
    }
    function textBlock(g, s, x, y, maxW, o = {}) {
      const sz = o.size || 12.5, lh = o.lh || Math.round(sz * 1.42);
      const L = wrapLines(g, s, maxW, sz, o.font || 'sans', o.weight || 400);
      L.forEach((l, i) => V.text(g, l, x, y + i * lh, Object.assign({ size: sz }, o)));
      return y + L.length * lh;
    }
    function fit(g, s, maxW, sz, f = 'sans', wt = 400) {
      s = String(s);
      if (tw(g, s, sz, f, wt) <= maxW) return s;
      while (s.length > 1 && tw(g, s + '…', sz, f, wt) > maxW) s = s.slice(0, -1);
      return s + '…';
    }
    /** token character: a little ball with eyes */
    function guy(g, x, y, r, col, o = {}) {
      const C = V.C();
      const a = o.alpha == null ? 1 : o.alpha;
      if (a <= 0.02) return;
      g.save(); g.globalAlpha = a;
      if (o.glow) V.glow(g, x, y, r * 2.5, col, 0.55);
      g.beginPath(); g.arc(x, y, r, 0, Math.PI * 2);
      g.fillStyle = V.alpha(col, C.dark ? 0.34 : 0.3); g.fill();
      g.lineWidth = 1.6; g.strokeStyle = col; g.stroke();
      const ex = r * 0.35, ey = -r * 0.14, er = Math.max(1.1, r * 0.13);
      const bl = o.blink ? 0.2 : 1;
      g.fillStyle = C.ink;
      g.beginPath(); g.ellipse(x - ex, y + ey, er, er * bl, 0, 0, Math.PI * 2); g.fill();
      g.beginPath(); g.ellipse(x + ex, y + ey, er, er * bl, 0, 0, Math.PI * 2); g.fill();
      g.beginPath(); g.arc(x, y + r * 0.1, r * 0.4, 0.18 * Math.PI, 0.82 * Math.PI); g.lineWidth = 1.2; g.strokeStyle = C.ink; g.stroke();
      g.restore();
    }
    const blinkAt = (t, seed) => ((t * 0.55 + seed * 0.37) % 3.2) < 0.13;
    /** cell with a value */
    function cell(g, x, y, c, v, o = {}) {
      const C = V.C();
      const sc = o.scale || 2;
      const tt = V.clamp(v / sc, -1, 1);
      const a = o.alpha == null ? 1 : o.alpha;
      g.save(); g.globalAlpha = a;
      const cw = (o.wd || c) - 2;
      V.rr(g, x, y, cw, c - 2, Math.min(5, c / 5));
      g.fillStyle = o.fill || V.cmap(tt, 'div'); g.fill();
      if (o.stroke) { g.strokeStyle = o.stroke; g.lineWidth = o.lw || 1.5; g.stroke(); }
      if (o.text !== false) {
        const light = !o.fill && Math.abs(tt) > 0.55;
        const s = o.fmt ? o.fmt(v) : String(v);
        V.text(g, s, x + cw / 2, y + (c - 2) / 2 + 0.5, { size: o.size || 11, font: 'mono', align: 'center', base: 'middle', color: o.textColor || (light ? onInk() : C.ink) });
      }
      g.restore();
    }

    /* ===================== Figure 1: the two-phase factory ===================== */
    {
      const f = $('#fig-duas-fases');
      fitH(f, 400, 450);
      const PROMPT = ['What', 'is', 'a', 'KV', 'cache', '?'];
      const RESP = ['An', 'archive', 'of', 'K', 'and', 'V'];
      const NL = 5, P = PROMPT.length, G = RESP.length, DT = 1.55, PT = 2.4;
      const S = { t3: performance.now(), last: -1 };
      const st = UI.stepper(f, [
        { t: 'The request arrives', d: 'Six prompt tokens wait at the entrance. The KV shelves, under each layer, are empty, and each layer\'s weights sit in memory.' },
        { t: 'Prefill: all at once', d: 'The six tokens go through the layers <b>together</b>: each projection receives a 6-row matrix (\\(M = 6\\)). As they pass, each layer writes the K and V of all six onto its shelf. Each layer\'s weights are read once and serve all six.' },
        { t: 'The first token', d: 'The <code>lm_head</code> turns the state of the last position into logits, and the sampler picks the first token of the answer. Everything up to here is the <b>TTFT</b>.' },
        { t: 'Decode: one token per step', d: 'The freshly generated token goes back to the entrance and goes through the layers <b>alone</b> (\\(M = 1\\)). In each layer it reads every row on the shelf (the attention, in amber) and adds its own.' },
        { t: 'Every step rereads the weights', d: 'Watch the pulses rising from memory: for every response token, the weights of all layers are read again, now to serve <b>a single</b> token. This cost is what makes decode bandwidth-bound.' },
        { t: 'Two different tallies', d: 'Prefill: 6 tokens, 1 step, each weight read serves 6 rows. Decode: 6 tokens, 6 steps, each weight read serves 1 row. The rest of the chapter turns this difference into numbers.' },
      ], { autoplay: 6500, onStep: (i) => { if (i === 3 || (i === 4 && S.last !== 3)) S.t3 = performance.now(); S.last = i; } });

      function chips(g, label, words, n, x0, y, xMax, col, o = {}) {
        const C = V.C();
        V.text(g, label, x0, y + 14, { size: 11, font: 'mono', color: C.muted });
        const xs = x0 + Math.max(tw(g, 'response', 11, 'mono'), tw(g, label, 11, 'mono')) + 10;
        let x = xs;
        const cnt = o.showAll ? words.length : n;
        for (let k = 0; k < cnt; k++) {
          const s = words[k];
          const cw = tw(g, s, 12, 'mono', 500) + 14;
          if (x + cw > xMax && x > xs) { x = xs; y += 24; }
          const on = k < n;
          const hi = o.hi === k || (o.hiAll && on);
          if (hi) V.glow(g, x + cw / 2, y + 10, cw * 0.8, col, 0.4);
          V.box(g, x, y, cw, 20, { r: 7, fill: on ? V.alpha(col, 0.2) : 'rgba(0,0,0,0)', stroke: on ? V.alpha(col, hi ? 1 : 0.6) : V.alpha(C.muted, 0.3), lw: 1.2 });
          V.text(g, s, x + cw / 2, y + 10.5, { size: 12, font: 'mono', weight: 500, color: on ? C.ink : C.muted, align: 'center', base: 'middle' });
          if (o.ttftAt === k) { V.box(g, x, y, cw, 20, { r: 7, stroke: C.d, lw: 1.8 }); V.dot(g, x + 1, y + 1, 3.2, C.d); }
          x += cw + 6;
        }
        if (o.cursor && Math.floor(performance.now() / 450) % 2 === 0) {
          if (x + 8 > xMax) { x = xs; y += 24; }
          V.box(g, x, y + 3, 7, 14, { r: 2, fill: V.alpha(col, 0.7) });
        }
        return y + 28;
      }

      V.fig(f, (g, w, h, t) => {
        const C = V.C();
        const narrow = w < 620;
        const i = st.i, e = st.elapsed();
        const pad = 14;
        // ---------- state ----------
        let pre = i === 0 ? 0 : i === 1 ? clamp(e / PT, 0, 1) : 1;
        let gen = 0, dj = -1, du = 0;
        if (i === 2) gen = e > 0.55 ? 1 : 0;
        if (i === 3 || i === 4) {
          const ed = (performance.now() - S.t3) / 1000;
          const n = Math.floor(ed / DT);
          if (n >= G - 1) { dj = -1; gen = G; } else { dj = n; du = (ed % DT) / DT; gen = 1 + n + (du > 0.86 ? 1 : 0); }
        }
        if (i === 5) gen = G;
        // ---------- header ----------
        let y = 12;
        y = chips(g, 'prompt', PROMPT, P, pad, y, w - pad, C.b, { hiAll: i === 1 && pre < 1 });
        y = chips(g, 'response', RESP, gen, pad, y, w - pad, C.a, { hi: dj >= 0 ? dj : (i === 2 && gen ? 0 : -1), cursor: gen < G, ttftAt: i >= 2 && gen ? 0 : -1 });
        // ---------- geometry ----------
        const top = y + 6;
        const cellS = narrow ? 7 : 10, cg = 2, shelfRows = 4, shelfH = shelfRows * (cellS + cg);
        const bankH = 22, bottomH = narrow ? 84 : 42;
        const avail = h - top - bottomH;
        const barH = clamp(avail - bankH - shelfH - 40, 60, narrow ? 190 : 150);
        const block = 14 + barH + 8 + shelfH + 10 + bankH;
        const off = Math.max(0, (avail - block) / 2);
        const barY = top + 12 + off, shelfY = barY + barH + 8, bankY = shelfY + shelfH + 10, midY = barY + barH / 2;
        const lmW = narrow ? 62 : 72;
        const lmX = w - pad - lmW - (narrow ? 2 : 48);
        const bx0 = narrow ? 58 : 96, bx1 = lmX - (narrow ? 36 : 64);
        const sp = (bx1 - bx0) / (NL - 1);
        const lx = (l) => bx0 + l * sp;
        const bw = Math.min(28, sp * 0.42);
        const xIn = pad + 14, xOut = (bx1 + lmX) / 2 + 2;
        const r = narrow ? 7 : 8.5;
        // travelers' position
        const gx = lerp(xIn, xOut, V.ease.inOut(pre));
        const dx = dj >= 0 ? lerp(xIn, xOut, V.ease.inOut(clamp(du / 0.8, 0, 1))) : xOut;
        const travX = i === 1 ? gx : (dj >= 0 ? dx : -999);
        const fill = (l) => {
          if (i <= 1) return gx >= lx(l) - 1 && pre > 0 ? P : 0;
          if (i === 2) return P;
          if (i === 5) return P + G - 1;
          if (dj < 0) return P + G - 1;
          return P + dj + (dx >= lx(l) ? 1 : 0);
        };
        // ---------- memory bank ----------
        const bkx = bx0 - bw / 2 - 12, bkw = bx1 - bx0 + bw + 24;
        V.box(g, bkx, bankY, bkw, bankH, { r: 7, fill: V.alpha(C.d, 0.12), stroke: V.alpha(C.d, 0.55), lw: 1.2 });
        V.text(g, narrow ? 'weights (memory)' : 'weights of all layers, in memory', bkx + bkw / 2, bankY + bankH / 2 + 0.5, { size: 11, font: 'mono', color: C.d, align: 'center', base: 'middle' });
        // weight-read pulses
        for (let l = 0; l < NL; l++) {
          const d = travX - lx(l);
          if (d >= -4 && d < 44) {
            const a = 1 - clamp((d + 4) / 48, 0, 1);
            const x = lx(l);
            const strong = i === 4 ? 1.6 : 1;
            g.save(); g.globalAlpha = a;
            V.line(g, x, bankY, x, barY + barH, { color: C.d, lw: 2.2 * strong, dash: [3, 4], dashOffset: t * 40 });
            for (let k = 0; k < 3; k++) { const u = ((t * 1.6 + k / 3) % 1); V.dot(g, x, lerp(bankY, barY + barH, u), 2.4 * strong, C.d); }
            g.restore();
          }
        }
        // ---------- layers ----------
        for (let l = 0; l < NL; l++) {
          const x = lx(l);
          const near = Math.abs(travX - x) < 20;
          if (near) V.glow(g, x, midY, bw * 1.8, i === 1 ? C.b : C.a, 0.5);
          V.box(g, x - bw / 2, barY, bw, barH, { r: 8, fill: V.alpha(C.a, near ? 0.22 : 0.09), stroke: V.alpha(C.a, near ? 0.95 : 0.45), lw: near ? 2 : 1.2 });
          V.text(g, narrow ? 'L' + (l + 1) : 'layer ' + (l + 1), x, barY - 4, { size: 11, font: 'mono', color: C.muted, align: 'center' });
          // shelf
          const nf_ = fill(l);
          const sw = 3 * (cellS + cg) - cg;
          const sx = x - sw / 2;
          const reading = (i === 3 || i === 4) && dj >= 0 && travX - x >= -2 && travX - x < 30;
          for (let n = 0; n < 12; n++) {
            const cx = sx + (n % 3) * (cellS + cg), cy = shelfY + Math.floor(n / 3) * (cellS + cg);
            const on = n < nf_;
            const col = n < P ? C.b : C.a;
            V.rr(g, cx, cy, cellS, cellS, 2);
            g.fillStyle = on ? V.alpha(col, 0.8) : V.alpha(C.muted, 0.14); g.fill();
            if (on && reading) { g.strokeStyle = C.d; g.lineWidth = 1.4; g.stroke(); }
          }
          if (reading) V.glow(g, x, shelfY + shelfH / 2, 22, C.d, 0.35);
        }
        V.text(g, 'KV ▸', bx0 - Math.max(bw / 2, 3 * (cellS + cg) / 2) - 8, shelfY + shelfH / 2 + 4, { size: 11, font: 'mono', weight: 600, color: C.muted, align: 'right' });
        // ---------- lm_head ----------
        const lmHot = (i === 2 && e < 1.2) || (dj >= 0 && du > 0.78 && du < 0.95);
        if (lmHot) V.glow(g, lmX + lmW / 2, midY, lmW, C.c, 0.5);
        V.box(g, lmX, midY - 24, lmW, 48, { r: 10, fill: V.alpha(C.c, lmHot ? 0.28 : 0.12), stroke: V.alpha(C.c, lmHot ? 1 : 0.55), lw: 1.4 });
        V.text(g, 'lm_head', lmX + lmW / 2, midY - 4, { size: narrow ? 11 : 12, font: 'mono', color: C.ink, align: 'center', base: 'middle' });
        V.text(g, 'sample', lmX + lmW / 2, midY + 12, { size: 11, color: C.muted, align: 'center', base: 'middle' });
        V.arrow(g, bx1 + bw / 2 + 4, midY, lmX - 4, midY, { color: V.alpha(C.muted, 0.5), lw: 1.3, head: 6 });
        // output (desktop)
        if (!narrow && gen > 0) {
          const ox = lmX + lmW + 26;
          const pop = dj >= 0 && du > 0.86 ? V.ease.back(clamp((du - 0.86) / 0.14, 0, 1)) : (i === 2 ? V.ease.back(clamp((e - 0.55) / 0.4, 0, 1)) : 1);
          guy(g, ox, midY, r * pop, C.a, { blink: blinkAt(t, 9) });
          V.text(g, RESP[gen - 1], ox, midY + r + 16, { size: 11, font: 'mono', color: C.muted, align: 'center' });
        }
        // ---------- travelers ----------
        if (i <= 2) {
          const a = i === 2 ? 1 - clamp(e / 0.6, 0, 1) : 1;
          for (let k = 0; k < P; k++) {
            const yy = midY + (k - (P - 1) / 2) * (2 * r + 3);
            const bob = i === 0 ? Math.sin(t * 3 + k) * 1.5 : 0;
            guy(g, gx, yy + bob, r, C.b, { alpha: a, blink: blinkAt(t, k), glow: i === 1 && pre < 1 && k === 0 });
          }
          if (i <= 1) V.text(g, 'M = ' + P, gx, midY + (P / 2) * (2 * r + 3) + 14, { size: 11, font: 'mono', color: C.b, align: 'center', alpha: i === 1 ? 1 : 0.6 });
        }
        if (dj >= 0) {
          const a = du > 0.8 ? 1 - clamp((du - 0.8) / 0.08, 0, 1) : 1;
          guy(g, dx, midY, r, C.a, { alpha: a, blink: blinkAt(t, 3), glow: true });
          V.text(g, 'M = 1', dx, midY + r + 15, { size: 11, font: 'mono', color: C.a, align: 'center', alpha: a });
        }
        // ---------- footer ----------
        const fwd = (pre > 0 ? 1 : 0) + (i === 5 ? G - 1 : dj >= 0 ? dj + (du > 0.05 ? 1 : 0) : (i === 3 || i === 4) ? G - 1 : 0);
        let msg;
        if (i === 0) msg = 'Waiting: ' + P + ' prompt tokens in the queue.';
        else if (i === 1) msg = 'PREFILL · 1 forward · M = ' + P + ' rows in each GEMM';
        else if (i === 2) msg = 'First token sampled: this is where TTFT ends.';
        else if (i === 5) msg = 'Prefill: 6 tokens in 1 step · Decode: 6 tokens in 6 steps';
        else msg = dj >= 0 ? 'DECODE · step ' + (dj + 1) + ' · M = 1 row · the shelf has ' + (P + dj + 1) + ' rows' : 'DECODE done: the shelf has ' + (P + G - 1) + ' rows';
        const yb = h - bottomH + 16;
        const col = i === 1 ? C.b : i >= 3 && i <= 4 ? C.a : i === 2 ? C.d : C.ink;
        textBlock(g, msg, pad, yb, w - 2 * pad, { size: 12.5, font: 'mono', weight: 600, color: col, lh: 16 });
        if (narrow) {
          V.text(g, 'forwards: ' + fwd + ' · weight reads: ' + fwd, pad, h - 24, { size: 11, font: 'mono', color: C.muted });
          V.text(g, 'tokens generated: ' + gen, pad, h - 9, { size: 11, font: 'mono', color: C.muted });
        } else {
          const cnt = 'forwards: ' + fwd + ' · full weight reads: ' + fwd + ' · tokens generated: ' + gen;
          V.text(g, fit(g, cnt, w - 2 * pad, 11, 'mono'), pad, h - 8, { size: 11, font: 'mono', color: C.muted });
        }
      });
    }

    /* ===================== Figure 2: GEMV × GEMM ===================== */
    {
      const f = $('#fig-gemv-gemm');
      fitH(f, 420, 560);
      const K = 4, N = 6, MMAX = 8;
      const R = V.rng(21);
      const W = Array.from({ length: K }, () => Array.from({ length: N }, () => R.int(4) - 1));
      const X = Array.from({ length: MMAX }, () => Array.from({ length: K }, () => R.int(4) - 1));
      const Y = X.map((row) => W[0].map((_, j) => row.reduce((s, a, k) => s + a * W[k][j], 0)));
      const S = { M: 1, b: 2, reset: true, t0: 0 };
      const ctl = UI.controls(f);
      const fmtI = () => nf(2 * S.M / S.b, 0) + ' FLOP/B';
      UI.slider(ctl, { label: 'Rows M', min: 1, max: MMAX, step: 1, value: 1, fmt: (v) => v + (v === 1 ? ' (GEMV)' : ' (GEMM)'), onInput: (v) => { S.M = v; S.reset = true; rU.set(String(v)); rI.set(fmtI()); } });
      UI.seg(ctl, { label: 'Weight', options: [{ v: 2, label: 'BF16' }, { v: 1, label: 'FP8' }, { v: 0.5, label: '4 bits' }], value: 2, onChange: (v) => { S.b = v; rI.set(fmtI()); } });
      const rU = UI.readout(ctl, 'uses per weight read', '1');
      const rI = UI.readout(ctl, 'intensity ≈ 2M/b', fmtI());
      V.fig(f, (g, w, h, t) => {
        const C = V.C();
        if (S.reset) { S.t0 = t; S.reset = false; }
        const M = S.M;
        const pad = 14, lab = 44;
        const side = w >= 600; // balance scale to the side (wide) or below (narrow)
        const balH = side ? 0 : 170;
        const c = Math.floor(Math.max(18, Math.min(34, (w - 2 * pad - lab - 22) / (K + N), (h - 128 - balH) / (K + MMAX))));
        const xX = pad + lab, xW = xX + K * c + 20, yW = 34, yX = yW + K * c + 18;
        // slots for more rows (M < 8): faded outlines
        for (let m = M; m < MMAX; m++) {
          V.text(g, 'tok ' + (m + 1), xX - 6, yX + m * c + (c - 2) / 2 + 0.5, { size: 11, font: 'mono', color: C.muted, align: 'right', base: 'middle', alpha: 0.28 });
          g.globalAlpha = 1;
          for (let k = 0; k < K; k++) { V.rr(g, xX + k * c, yX + m * c, c - 2, c - 2, 4); g.fillStyle = V.alpha(C.muted, 0.05); g.fill(); g.strokeStyle = V.alpha(C.muted, 0.12); g.lineWidth = 1; g.stroke(); }
          for (let j = 0; j < N; j++) { V.rr(g, xW + j * c, yX + m * c, c - 2, c - 2, 4); g.fillStyle = V.alpha(C.muted, 0.05); g.fill(); g.strokeStyle = V.alpha(C.muted, 0.12); g.lineWidth = 1; g.stroke(); }
        }
        const Tc = 0.55 + 0.11 * M, total = N * Tc + 1.2;
        const e = (t - S.t0) % total;
        const jj = Math.floor(e / Tc), u = (e % Tc) / Tc;
        const done = jj >= N;
        const jCur = done ? N : jj;
        const mCur = done ? M : (u < 0.35 ? -1 : Math.min(M - 1, Math.floor(((u - 0.35) / 0.65) * M)));
        // labels
        V.text(g, 'W  [K=4 × N=6]', xW, yW - 10, { size: 11, font: 'mono', color: C.muted });
        V.text(g, 'X  [M×4]', xX, yX - 6, { size: 11, font: 'mono', color: C.muted });
        V.text(g, 'Y = XW  [M×6]', xW, yX - 6, { size: 11, font: 'mono', color: C.muted });
        // memory bank
        const dmX = pad, dmW = lab + K * c - 8, dmH = K * c - 2;
        V.box(g, dmX, yW, dmW, dmH, { r: 10, fill: V.alpha(C.d, 0.1), stroke: V.alpha(C.d, 0.5), lw: 1.2 });
        for (let k = 0; k < 3; k++) V.box(g, dmX + 10, yW + 12 + k * 12, dmW - 20, 7, { r: 3, fill: V.alpha(C.d, 0.25 + 0.1 * k) });
        V.text(g, 'memory', dmX + dmW / 2, yW + dmH - 24, { size: 11, font: 'mono', color: C.d, align: 'center' });
        V.text(g, '(weights W)', dmX + dmW / 2, yW + dmH - 10, { size: 11, color: C.muted, align: 'center' });
        // W
        for (let k = 0; k < K; k++) for (let j = 0; j < N; j++) {
          const loaded = j < jCur || (j === jCur && u >= 0.35);
          const isCur = j === jCur && !done;
          cell(g, xW + j * c, yW + k * c, c, W[k][j], { alpha: loaded || isCur ? 1 : 0.35, stroke: isCur && u >= 0.35 ? C.c : null });
        }
        // packet traveling from memory to the column
        if (!done && u < 0.35) {
          const p = V.ease.inOut(u / 0.35);
          const x0 = dmX + dmW - 6, x1 = xW + jCur * c + (c - 2) / 2;
          const px = lerp(x0, x1, p);
          V.glow(g, px, yW + K * c / 2, 20, C.d, 0.8);
          for (let k = 0; k < K; k++) V.box(g, px - 6, yW + k * c + c / 2 - 7, 12, 12, { r: 3, fill: V.alpha(C.d, 0.85) });
        }
        // X
        for (let m = 0; m < M; m++) {
          V.text(g, 'tok ' + (m + 1), xX - 6, yX + m * c + (c - 2) / 2 + 0.5, { size: 11, font: 'mono', color: C.muted, align: 'right', base: 'middle' });
          for (let k = 0; k < K; k++) cell(g, xX + k * c, yX + m * c, c, X[m][k], { stroke: !done && m === mCur ? C.c : null });
        }
        // Y
        for (let m = 0; m < M; m++) for (let j = 0; j < N; j++) {
          const filled = done || j < jCur || (j === jCur && m <= mCur);
          const flash = !done && j === jCur && m === mCur;
          if (filled) cell(g, xW + j * c, yW + K * c + 18 + m * c, c, Y[m][j], { scale: 8, stroke: flash ? C.c : null });
          else { V.rr(g, xW + j * c, yW + K * c + 18 + m * c, c - 2, c - 2, 4); g.strokeStyle = V.alpha(C.muted, 0.3); g.lineWidth = 1; g.setLineDash([3, 3]); g.stroke(); g.setLineDash([]); }
          if (flash) V.glow(g, xW + j * c + c / 2, yW + K * c + 18 + m * c + c / 2, c, C.c, 0.5);
        }
        // counters
        const cols = Math.min(N, jCur + (u >= 0.35 && !done ? 1 : 0));
        const mults = done ? N * K * M : (jCur * K * M + (mCur >= 0 ? (mCur + 1) * K : 0));
        const narrowT = w < 560;
        const yT = h - (narrowT ? 64 : 34);
        const line1 = 'columns read: ' + cols + '/' + N + ' · weights read: ' + cols * K + ' · multiplications: ' + mults;
        const line2 = 'each weight read serves M = ' + M + (M === 1 ? ' token' : ' tokens') + ' → I ≈ 2M/b = ' + nf(2 * M / S.b, 0) + ' FLOP/byte';
        const yy2 = textBlock(g, line1, pad, yT, w - 2 * pad, { size: 11.5, font: 'mono', color: C.ink, lh: 16 });
        textBlock(g, line2, pad, yy2 + 4, w - 2 * pad, { size: 11.5, font: 'mono', weight: 600, color: M === 1 ? C.d : C.a, lh: 16 });
        // balance scale: read time × compute time (GB10)
        const I = 2 * M / S.b, ridge = 458;
        const ratio = ridge / I;
        // compact (below, narrow screens) or large (to the side)
        const bx0 = xW + N * c + 36;
        const cmp = !side || w - pad - bx0 < 230;
        const cxB = cmp ? w / 2 : (bx0 + w - pad) / 2;
        const topB = cmp ? yX + MMAX * c + 14 : yW - 10;
        const availW = cmp ? w - 2 * pad : w - pad - bx0;
        const Lb = cmp ? Math.min(112, w / 2 - 62) : Math.min(130, availW / 2 - 30);
        const pyB = cmp ? topB + 40 : yW + 40;
        const amp = cmp ? 0.2 : 0.36, strL = cmp ? 26 : 42, colH = cmp ? 56 : 120;
        const th = amp * clamp(Math.log10(ratio) / Math.log10(ridge), -1, 1) + Math.sin(t * 2) * 0.01;
        const lx1 = cxB - Lb * Math.cos(th), ly1 = pyB + Lb * Math.sin(th);
        const rx1 = cxB + Lb * Math.cos(th), ry1 = pyB - Lb * Math.sin(th);
        V.text(g, fit(g, cmp ? 'on the GB10: read W vs. compute' : 'on the GB10, for this matrix:', availW, 11, 'mono'), cxB, topB + (cmp ? 10 : 0), { size: 11, font: 'mono', color: C.muted, align: 'center' });
        // column and base
        V.line(g, cxB, pyB, cxB, pyB + colH, { color: C.muted, lw: 3 });
        V.box(g, cxB - 34, pyB + colH - 2, 68, 8, { r: 4, fill: V.alpha(C.muted, 0.5) });
        V.line(g, lx1, ly1, rx1, ry1, { color: C.ink, lw: 3 });
        V.dot(g, cxB, pyB, 5, C.ink);
        const pan = (x, y, col, lbl, sub, n) => {
          V.line(g, x, y, x - 22, y + strL, { color: V.alpha(C.ink, 0.6), lw: 1 });
          V.line(g, x, y, x + 22, y + strL, { color: V.alpha(C.ink, 0.6), lw: 1 });
          V.box(g, x - 30, y + strL, 60, 8, { r: 4, fill: V.alpha(col, 0.8) });
          for (let k = 0; k < n; k++) V.box(g, x - 22 + (k % 4) * 11, y + strL - 10 - Math.floor(k / 4) * 10, 9, 8, { r: 2, fill: V.alpha(col, 0.9) });
          V.text(g, lbl, x, y + strL + 22, { size: 11.5, font: 'mono', weight: 600, color: col, align: 'center' });
          if (!cmp) V.text(g, sub, x, y + strL + 37, { size: 11, color: C.muted, align: 'center' });
        };
        pan(lx1, ly1, C.d, 'read W', 'bytes ÷ bandwidth', 8);
        pan(rx1, ry1, C.a, 'compute', 'FLOPs ÷ peak', Math.max(1, Math.min(8, Math.round(8 * I / ridge * 8))));
        const msg = ratio >= 1.05 ? 'reading costs ' + nf(ratio, ratio < 10 ? 1 : 0) + '× more than computing' : 'compute dominates';
        const my = cmp ? pyB + 90 : pyB + 150;
        V.text(g, fit(g, msg, availW, 12.5, 'mono', 600), cxB, my, { size: 12.5, font: 'mono', weight: 600, color: ratio >= 1.05 ? C.d : C.a, align: 'center' });
        V.text(g, fit(g, 'balance only at I ≈ 458 (M ≈ ' + nf(458 * S.b / 2) + ')', availW, 11), cxB, my + 17, { size: 11, color: C.muted, align: 'center' });
      });
    }

    /* ===================== Figure 3: roofline ===================== */
    {
      const f = $('#fig-roofline');
      fitH(f, 450, 520);
      const MS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 7168, 8192];
      const S = { mi: 0, b: 2, moe: false, peak: 125, cur: 0 };
      const BW = 0.273; // TB/s -> TFLOPS = I * BW
      const E = 288, KE = 8, BA = 2; // BF16 activations
      const DENSE = [[4096, 4096]];
      const EXPERT = [[4096, 2048], [1024, 4096]]; // GLM expert at TP=2: local gate_up and down
      const rowsPerExpert = (M) => { const D = E * (1 - Math.pow(1 - KE / E, M)); return (M * KE) / D; };
      // full intensity: FLOPs ÷ (weight bytes + activation bytes)
      const intenOf = (M, b, moe) => { const r = moe ? rowsPerExpert(M) : M; let F = 0, Q = 0; (moe ? EXPERT : DENSE).forEach(([K, N]) => { F += 2 * r * K * N; Q += b * K * N + BA * r * (K + N); }); return F / Q; };
      const approxOf = (M, b, moe) => 2 * (moe ? rowsPerExpert(M) : M) / b;
      const inten = () => intenOf(MS[S.mi], S.b, S.moe);
      S.cur = Math.log10(inten());
      const ctl = UI.controls(f);
      const upd = () => { const I = inten(); const perf = Math.min(S.peak, I * BW); rI.set(nf(I, I < 10 ? 1 : 0) + ' FLOP/B'); rP.set(nf(perf, perf < 10 ? 2 : 1) + ' TFLOPS'); rR.set(I < S.peak / BW ? 'memory' : 'compute'); };
      UI.slider(ctl, { label: 'Rows M', min: 0, max: MS.length - 1, step: 1, value: 0, fmt: (v) => nf(MS[v]), onInput: (v) => { S.mi = v; upd(); } });
      UI.seg(ctl, { label: 'Weight', options: [{ v: 2, label: 'BF16' }, { v: 1, label: 'FP8' }, { v: 0.5, label: '4 bits' }], value: 2, onChange: (v) => { S.b = v; upd(); } });
      UI.toggle(ctl, { label: 'MoE: 288 experts, top-8 (GLM)', value: false, onChange: (v) => { S.moe = v; upd(); } });
      UI.seg(ctl, { label: 'Peak', options: [{ v: 125, label: 'BF16 ~125 TFLOPS' }, { v: 500, label: 'FP4×FP8 ~4×' }], value: 125, onChange: (v) => { S.peak = v; upd(); } });
      const rI = UI.readout(ctl, 'intensity (weights + activations)', '—');
      const rP = UI.readout(ctl, 'attainable ceiling', '—');
      const rR = UI.readout(ctl, 'bound by', '—');
      upd();
      // GLM lm_head per rank (154,880 / 2 × 4,096, BF16) with the 8 positions of a DFlash2 step
      const lmK = 4096, lmN = 77440;
      const MARK = [
        { I: intenOf(1, 2, false), l: 'decode BF16, M = 1', s: 'decode BF16' },
        { I: intenOf(1, 0.5, false), l: 'decode 4-bit, M = 1', s: 'decode 4 b' },
        { I: (2 * 8 * lmK * lmN) / (2 * lmK * lmN + BA * 8 * (lmK + lmN)), l: 'GLM lm_head · 8 positions', s: 'lm_head ×8' },
        { I: intenOf(1024, 0.5, true), l: 'GLM MoE · 1,024 chunk', s: 'MoE 1,024', nLeft: true },
        { I: intenOf(7168, 0.5, true), l: 'GLM MoE · 7,168 chunk', s: 'MoE 7,168' },
      ];
      V.fig(f, (g, w, h, t, dt) => {
        const C = V.C();
        const narrow = w < 560;
        const boxH = 56;
        const L = narrow ? 44 : 60, Rm = 14, T = boxH + 24, B = 44;
        const X0 = Math.log10(0.25), X1 = Math.log10(32768), Y0 = Math.log10(0.05), Y1 = Math.log10(1500);
        const px = (I) => L + (Math.log10(I) - X0) / (X1 - X0) * (w - L - Rm);
        const py = (P) => T + (1 - (Math.log10(P) - Y0) / (Y1 - Y0)) * (h - T - B);
        const roof = (I) => Math.min(S.peak, I * BW);
        const ridge = S.peak / BW;
        // grid
        [1, 10, 100, 1000, 10000].forEach((I) => { V.line(g, px(I), T, px(I), h - B, { color: C.grid, lw: 1 }); V.text(g, nf(I), px(I), h - B + 16, { size: 11, font: 'mono', color: C.muted, align: 'center' }); });
        [0.1, 1, 10, 100, 1000].forEach((P) => { V.line(g, L, py(P), w - Rm, py(P), { color: C.grid, lw: 1 }); V.text(g, nf(P, P < 1 ? 1 : 0), L - 6, py(P) + 4, { size: 11, font: 'mono', color: C.muted, align: 'right' }); });
        V.text(g, 'arithmetic intensity (FLOP/byte, log scale)', L + (w - L - Rm) / 2, h - 8, { size: 11, color: C.muted, align: 'center' });
        g.save(); g.translate(12, T + (h - T - B) / 2); g.rotate(-Math.PI / 2); V.text(g, 'TFLOPS (log)', 0, 0, { size: 11, color: C.muted, align: 'center', base: 'middle' }); g.restore();
        // regions
        g.beginPath(); g.moveTo(px(0.25), h - B); g.lineTo(px(0.25), py(roof(0.25))); g.lineTo(px(ridge), py(S.peak)); g.lineTo(px(ridge), h - B); g.closePath();
        g.fillStyle = V.alpha(C.d, 0.1); g.fill();
        g.beginPath(); g.moveTo(px(ridge), h - B); g.lineTo(px(ridge), py(S.peak)); g.lineTo(px(32768), py(S.peak)); g.lineTo(px(32768), h - B); g.closePath();
        g.fillStyle = V.alpha(C.b, 0.1); g.fill();
        // roof
        g.beginPath(); g.moveTo(px(0.25), py(roof(0.25))); g.lineTo(px(ridge), py(S.peak)); g.lineTo(px(32768), py(S.peak));
        g.strokeStyle = C.ink; g.lineWidth = 2.4; g.stroke();
        V.line(g, px(ridge), py(S.peak), px(ridge), h - B, { color: V.alpha(C.ink, 0.5), lw: 1.2, dash: [4, 4] });
        const rl = 'ridge ≈ ' + nf(ridge) + ' FLOP/B';
        const rlw = tw(g, rl, 11, 'mono');
        const memX = px(Math.sqrt(0.25 * ridge));
        const memW = tw(g, 'memory-bound', narrow ? 11 : 12, 'sans', 600);
        const rlx = Math.min(px(ridge) + 6, w - Rm - rlw);
        if (rlx < memX + memW / 2 + 8) V.text(g, rl, px(ridge) - 6, h - B - 26, { size: 11, font: 'mono', color: C.ink, align: 'right' }); // without colliding with the region label
        else V.text(g, rl, rlx, h - B - 8, { size: 11, font: 'mono', color: C.ink });
        // peak label: above the ceiling, on the right (out of the way of the ridge and the marks)
        V.text(g, 'peak ≈ ' + S.peak + ' TFLOPS', w - Rm - 4, py(S.peak) - 12, { size: 11, font: 'mono', color: C.muted, align: 'right' });
        V.text(g, 'memory-bound', memX, h - B - 10, { size: narrow ? 11 : 12, weight: 600, color: C.d, align: 'center' });
        const cmpX = (px(ridge) + w - Rm) / 2;
        const cmpW = tw(g, narrow ? 'compute' : 'compute-bound', narrow ? 11 : 12, 'sans', 600);
        V.text(g, narrow ? 'compute' : 'compute-bound', Math.min(w - Rm - cmpW / 2 - 4, Math.max(cmpX, px(ridge) + 60)), py(S.peak) + 40, { size: narrow ? 11 : 12, weight: 600, color: C.b, align: 'center' });
        { // ramp label, aligned with the slope, in a gap between the marks
          const Is = 28;
          const ang = Math.atan2(py(roof(ridge)) - py(roof(0.25)), px(ridge) - px(0.25));
          g.save(); g.translate(px(Is), py(roof(Is))); g.rotate(ang);
          V.text(g, narrow ? '273 GB/s × I' : 'bandwidth: 273 GB/s × I', 0, -7, { size: 11, font: 'mono', color: C.muted, align: 'center' });
          g.restore();
        }
        // fixed marks (labels always below and to the right of the ramp)
        MARK.forEach((m) => {
          const x = px(m.I), y = py(roof(m.I));
          g.beginPath(); g.moveTo(x, y - 6); g.lineTo(x + 6, y); g.lineTo(x, y + 6); g.lineTo(x - 6, y); g.closePath();
          g.fillStyle = V.alpha(C.c, 0.85); g.fill();
          const lbl = narrow ? m.s : m.l;
          const lw_ = tw(g, lbl, 11, 'mono');
          if (narrow && m.nLeft) V.text(g, lbl, x - 9, y - 9, { size: 11, font: 'mono', color: C.c, align: 'right' }); // does not cross the ridge
          else V.text(g, lbl, Math.min(x + 9, w - Rm - lw_), y + 18, { size: 11, font: 'mono', color: C.c });
        });
        // current point (animated in log space)
        const target = Math.log10(clamp(inten(), 0.25, 32768));
        S.cur += (target - S.cur) * Math.min(1, dt * 6);
        const Ic = Math.pow(10, S.cur), Pc = roof(Ic);
        const cx = px(Ic), cy = py(Pc);
        const memB = Ic < ridge;
        const col = memB ? C.d : C.b;
        V.line(g, cx, cy, cx, h - B, { color: V.alpha(col, 0.6), lw: 1.2, dash: [2, 3] });
        V.line(g, L, cy, cx, cy, { color: V.alpha(col, 0.6), lw: 1.2, dash: [2, 3] });
        V.glow(g, cx, cy, 20 + 5 * Math.sin(t * 4), col, 0.8);
        V.dot(g, cx, cy, 6, col);
        // readout box, above the chart
        const M = MS[S.mi];
        const Ia = approxOf(M, S.b, S.moe);
        const l1 = 'M = ' + nf(M) + (S.moe ? ' → ' + nf(rowsPerExpert(M), 1) + ' rows/expert' : (narrow ? '' : ' · 4096 × 4096 projection'));
        const l2 = 'I ≈ ' + nf(Ic, Ic < 10 ? 1 : 0) + ' FLOP/B → ' + nf(Pc, Pc < 10 ? 2 : 1) + ' TFLOPS';
        const l3 = 'weights only (' + (S.moe ? '2r/b_w' : '2M/b_w') + '): ' + nf(Ia, Ia < 10 ? 1 : 0) + ' FLOP/B';
        const bwid = Math.min(w - L - Rm, Math.max(tw(g, l1, 11.5, 'mono', 600), tw(g, l2, 11.5, 'mono'), tw(g, l3, 11, 'mono')) + 16);
        const bx = L, by = 8;
        V.box(g, bx, by, bwid, boxH, { r: 8, fill: V.alpha(C.panel || C.bg, 0.92), stroke: V.alpha(col, 0.8), lw: 1.2 });
        V.text(g, l1, bx + 8, by + 17, { size: 11.5, font: 'mono', weight: 600, color: C.ink });
        V.text(g, l2, bx + 8, by + 33, { size: 11.5, font: 'mono', color: col });
        V.text(g, l3, bx + 8, by + 49, { size: 11, font: 'mono', color: C.muted });
      });
    }

    /* ===================== Figure 4: the decode byte tank ===================== */
    {
      const f = $('#fig-bytes-passo');
      fitH(f, 420, 690);
      const GiB = 1.073741824;
      // tau = accepted + 1 bonus; tfix = fixed costs per step (DS: report; Qwen/GLM: measured step − streaming)
      const MOD = {
        qwen: { name: 'Qwen3.8 · MTP=3 step', parts: [['lm_head ×4', 2.36 * GiB], ['routed experts', 2.1 * GiB], ['GDN projections', 1.95 * GiB], ['HyperConnection', 1.19 * GiB], ['attention + indexer', 0.6 * GiB], ['MTP', 0.5 * GiB], ['router + shared + PLE', 0.4 * GiB], ['GDN state', 0.1 * GiB]], bw: 240, tau: 3.2, tfix: 20, meas: 52.1, measL: 'MTP=3, batch-1: 52.1 tok/s' },
        ds: { name: 'DeepSeek-V4 · DSpark step', parts: [['routed experts', 12], ['FP8 attention', 2.4], ['lm_head (target + draft)', 1.06], ['draft (3 layers)', 0.9], ['Markov w2', 0.2]], bw: 240, tau: 4.15, tfix: 12, meas: 56, measL: 'c=1 median (Sep 2, range 51–67): 56 tok/s' },
        glm: { name: 'GLM-5.3 · DFlash2 (estimate)', parts: [['experts (~57 of 288)', 15], ['KDA', 4.7], ['MLA + indexer', 1.3], ['shared experts', 1.1], ['lm_head', 0.6], ['dense MLP', 0.45]], bw: 230, tau: 7.7, tfix: 18, meas: 65.1, measL: 'structured ×1 (lab, Aug 30): 65.1 tok/s' },
      };
      const KVB = 88064; // B/token/GPU: GQA-8 BF16, 43 layers, TP=2 (hypothetical)
      const S = { m: 'qwen', bw: 240, tau: 3.2, tfix: 20, ctx: 0, scale: 14, lvl: 0 };
      const ctl = UI.controls(f);
      const calc = () => {
        const Mo = MOD[S.m];
        const Q = Mo.parts.reduce((a, p) => a + p[1], 0) + KVB * S.ctx / 1e9;
        const msS = Q / S.bw * 1000, ms = msS + S.tfix;
        return { Q, msS, ms, tokS: S.tau / (msS / 1000), tok: S.tau / (ms / 1000), fS: msS / ms };
      };
      const upd = () => { const r = calc(); rMs.set(nf(r.ms, 1) + ' ms'); rS.set(nf(r.tokS, 1) + ' tok/s'); rT.set(nf(r.tok, 1) + ' tok/s'); rM.set(MOD[S.m].measL.split(': ').pop()); };
      UI.seg(ctl, { label: 'Recipe', options: [{ v: 'qwen', label: 'Qwen' }, { v: 'ds', label: 'DeepSeek' }, { v: 'glm', label: 'GLM' }], value: 'qwen', onChange: (v) => { S.m = v; sB.set(MOD[v].bw); sT.set(MOD[v].tau); sF.set(MOD[v].tfix); upd(); } });
      const sB = UI.slider(ctl, { label: 'Bandwidth', min: 180, max: 273, step: 1, value: 240, fmt: (v) => v + ' GB/s', onInput: (v) => { S.bw = v; upd(); } });
      const sT = UI.slider(ctl, { label: 'Tokens/step', min: 1, max: 8, step: 0.05, value: 3.2, fmt: (v) => nf(v, 2), onInput: (v) => { S.tau = v; upd(); } });
      const sF = UI.slider(ctl, { label: 'Fixed cost', min: 0, max: 30, step: 1, value: 20, fmt: (v) => v + ' ms', onInput: (v) => { S.tfix = v; upd(); } });
      UI.slider(ctl, { label: 'Dense KV (hyp.)', min: 0, max: 262144, step: 8192, value: 0, fmt: (v) => (v === 0 ? 'off' : nf(v / 1024) + 'k'), onInput: (v) => { S.ctx = v; upd(); } });
      const rMs = UI.readout(ctl, 'step (streaming + fixed)', '—');
      const rS = UI.readout(ctl, 'streaming-only ceiling', '—');
      const rT = UI.readout(ctl, 'prediction', '—');
      const rM = UI.readout(ctl, 'measured', '—');
      upd();
      V.fig(f, (g, w, h, t, dt) => {
        const C = V.C();
        const narrow = w < 620;
        const Mo = MOD[S.m];
        const r = calc();
        const parts = Mo.parts.slice();
        if (S.ctx > 0) parts.push(['dense KV (hypothetical)', KVB * S.ctx / 1e9]);
        S.scale += (Math.max(12, r.Q * 1.1) - S.scale) * Math.min(1, dt * 4);
        const pad = 16;
        const tx = pad, tW = narrow ? 58 : 92, ty0 = 30;
        const ty1 = narrow ? h - 300 : h - 64;
        const th = ty1 - ty0;
        V.text(g, fit(g, Mo.name, narrow ? w - 2 * pad : w * 0.55, 13, 'display', 600), pad, 18, { size: 13, font: 'display', weight: 600, color: C.ink });
        // tank: drains during streaming and stays empty during the fixed cost
        V.box(g, tx - 3, ty0 - 3, tW + 6, th + 6, { r: 10, fill: V.alpha(C.muted, 0.06), stroke: V.alpha(C.muted, 0.4), lw: 1.2 });
        const dispT = Math.max(0.8, r.ms / 40); // 40 ms of step = 1 s of animation
        const ph = (t % dispT) / dispT;
        const inFix = ph >= r.fS;
        const liquid = inFix ? 0 : r.Q * (1 - ph / r.fS);
        let acc = 0;
        const segs = [];
        parts.forEach((p, k) => {
          const y1 = ty1 - (acc / S.scale) * th;
          const y0 = ty1 - ((acc + p[1]) / S.scale) * th;
          const col = p[0].startsWith('dense KV') ? C.red : C.series[k % C.series.length];
          const cut = ty1 - (liquid / S.scale) * th;
          const fy0 = Math.max(y0, cut);
          if (y1 > fy0) { g.fillStyle = V.alpha(col, 0.78); g.fillRect(tx, fy0, tW, y1 - fy0); }
          if (fy0 > y0) { g.fillStyle = V.alpha(col, 0.16); g.fillRect(tx, y0, tW, Math.min(fy0, y1) - y0); }
          V.line(g, tx, y0, tx + tW, y0, { color: V.alpha(C.bg, 0.6), lw: 1 });
          segs.push({ y: (y0 + y1) / 2, col, name: p[0], v: p[1], hgt: y1 - y0 });
          acc += p[1];
        });
        const lvlY = ty1 - (liquid / S.scale) * th;
        if (!inFix) V.line(g, tx, lvlY, tx + tW, lvlY, { color: C.ink, lw: 1.5 });
        // non-overlapping labels
        const lh = 15;
        const ys = segs.map((s) => s.y);
        for (let k = 1; k < ys.length; k++) if (ys[k] > ys[k - 1] - lh) ys[k] = ys[k - 1] - lh;
        const topLim = ty0 + 8;
        if (ys.length && ys[ys.length - 1] < topLim) { const sh = topLim - ys[ys.length - 1]; for (let k = 0; k < ys.length; k++) ys[k] += sh; for (let k = ys.length - 2; k >= 0; k--) if (ys[k] < ys[k + 1] + lh) ys[k] = ys[k + 1] + lh; }
        const lx0 = tx + tW + 14;
        const lMax = narrow ? w - pad - lx0 : w * 0.56 - lx0;
        segs.forEach((s, k) => {
          V.line(g, tx + tW + 2, s.y, lx0 - 2, ys[k], { color: V.alpha(s.col, 0.6), lw: 1 });
          V.dot(g, lx0 + 3, ys[k], 3.5, s.col);
          const txt = s.name + ' · ' + nf(s.v, s.v < 1 ? 2 : 1) + ' GB';
          V.text(g, fit(g, txt, lMax - 12, 11.5), lx0 + 11, ys[k] + 4, { size: 11.5, color: C.ink });
        });
        // pipe + GPU (during the fixed cost, the pipe stands still and the GPU waits on collectives and launches)
        const gpuW = narrow ? 58 : 80, gpuX = tx + tW / 2 - gpuW / 2, gpuY = ty1 + 16;
        const pipeW = 4 + (S.bw - 180) / 93 * 6;
        g.fillStyle = V.alpha(C.a, 0.25); g.fillRect(tx + tW / 2 - pipeW / 2, ty1 + 3, pipeW, gpuY - ty1 - 3);
        if (!inFix) for (let k = 0; k < 3; k++) { const u = (t * (S.bw / 120) + k / 3) % 1; V.dot(g, tx + tW / 2, ty1 + 3 + u * (gpuY - ty1 - 3), 2, C.a); }
        const gcol = inFix ? C.c : C.a;
        if (inFix) V.glow(g, gpuX + gpuW / 2, gpuY + 12, gpuW * 0.8, C.c, 0.4 + 0.2 * Math.sin(t * 9));
        V.box(g, gpuX, gpuY, gpuW, 24, { r: 7, fill: V.alpha(gcol, 0.18), stroke: gcol, lw: 1.3, label: 'GPU', size: 11.5, font: 'mono' });
        V.text(g, inFix ? 't_fixed: collectives, launches…' : 'weight streaming', gpuX + gpuW + 10, gpuY + 16, { size: 11, font: 'mono', color: inFix ? C.c : C.muted });
        // numbers panel
        const lines = [
          ['Q = ' + nf(r.Q, 2) + ' GB read per step', C.ink],
          ['Q/β = ' + nf(r.Q, 2) + ' GB ÷ ' + S.bw + ' GB/s = ' + nf(r.msS, 1) + ' ms', C.ink],
          ['+ t_fixed ' + S.tfix + ' ms = step of ' + nf(r.ms, 1) + ' ms', C.c],
          [nf(1000 / r.ms, 1) + ' steps/s × ' + nf(S.tau, 2) + ' tokens/step', C.muted],
        ];
        let px0, py0, pw;
        if (narrow) { px0 = pad; py0 = gpuY + 48; pw = w - 2 * pad; } else { px0 = w * 0.6; py0 = 56; pw = w - px0 - pad; }
        let yy = py0;
        lines.forEach((L) => { V.text(g, fit(g, L[0], pw, 12.5, 'mono'), px0, yy, { size: 12.5, font: 'mono', color: L[1] }); yy += 21; });
        yy += narrow ? 14 : 22;
        V.text(g, '≈ ' + nf(r.tok, 1), px0, yy, { size: narrow ? 26 : 34, font: 'display', weight: 700, color: C.a });
        V.text(g, 'tok/s (prediction)', px0 + tw(g, '≈ ' + nf(r.tok, 1), narrow ? 26 : 34, 'display', 700) + 10, yy - 4, { size: 11.5, color: C.muted });
        yy += narrow ? 16 : 22;
        // bars: streaming only × prediction × measured
        const bmax = Math.max(r.tokS, Mo.meas) * 1.08;
        const bW = pw - 10;
        [['streaming only', r.tokS, C.d], ['prediction', r.tok, C.a], ['measured', Mo.meas, C.c]].forEach((b, k) => {
          const y = yy + k * 22;
          V.box(g, px0, y, bW, 15, { r: 5, fill: V.alpha(C.muted, 0.12) });
          const bw_ = Math.max(4, bW * b[1] / bmax);
          V.box(g, px0, y, bw_, 15, { r: 5, fill: V.alpha(b[2], 0.78) });
          const lbl = b[0] + ' ' + nf(b[1], 1);
          const inside = tw(g, lbl, 11, 'mono', 600) + 12 < bw_;
          V.text(g, lbl, inside ? px0 + 6 : px0 + bw_ + 6, y + 11.5, { size: 11, font: 'mono', weight: 600, color: inside ? onInk() : C.ink });
        });
        yy += 3 * 22 + 10;
        const pct = 'streaming = ' + nf(100 * r.fS, 0) + '% of the predicted step';
        V.text(g, fit(g, Mo.measL + (S.ctx > 0 ? ' · hypothetical dense KV' : ''), pw, 11), px0, Math.min(yy, h - 24), { size: 11, color: C.muted });
        V.text(g, fit(g, pct, pw, 11), px0, Math.min(yy + 16, h - 8), { size: 11, color: C.muted });
      });
    }

    /* ===================== Figure 5: KV cache, step by step ===================== */
    {
      const f = $('#fig-kv-mecanismo');
      fitH(f, 450, 680, 760);
      const r1 = (x) => Math.round(x * 10) / 10;
      const R = V.rng(7); const rnd = (s) => r1(R.normal() * s);
      const Kp = Array.from({ length: 4 }, () => Array.from({ length: 4 }, () => rnd(0.9)));
      const Vp = Array.from({ length: 4 }, () => Array.from({ length: 4 }, () => rnd(0.9)));
      const WQ = Array.from({ length: 4 }, () => Array.from({ length: 4 }, () => rnd(0.7)));
      const WK = Array.from({ length: 4 }, () => Array.from({ length: 4 }, () => rnd(0.7)));
      const WV = Array.from({ length: 4 }, () => Array.from({ length: 4 }, () => rnd(0.7)));
      const xv = [0.8, -0.4, 1.1, 0.3];
      const mv = (Wm) => [0, 1, 2, 3].map((j) => r1(xv.reduce((s, a, k) => s + a * Wm[k][j], 0)));
      const q = mv(WQ), kn = mv(WK), vn = mv(WV);
      const Ka = [...Kp, kn], Va = [...Vp, vn];
      const sc = Ka.map((row) => Math.round((row.reduce((a, b, i) => a + b * q[i], 0) / 2) * 100) / 100);
      const al = V.softmax(sc);
      const o = [0, 1, 2, 3].map((j) => r1(al.reduce((s, a, i) => s + a * Va[i][j], 0)));
      const TOK = ['A', 'cat', 'climbed', 'the', 'roof'];
      const S = { noCache: false };
      const st = UI.stepper(f, [
        { t: 'The new token arrives', d: 'The last generated token, <b>roof</b>, enters the layer as the vector \\(h_t\\) (4 numbers). The K and V of the 4 previous tokens are already in the cache: they were computed during prefill.' },
        { t: 'Three projections', d: '\\(q_t = W_Q h_t\\), \\(k_t = W_K h_t\\), \\(v_t = W_V h_t\\): three matrix × vector multiplications. With the cache, this is the <b>only</b> new K and V computation in this step.' },
        { t: 'Filing away k and v', d: '\\(k_t\\) and \\(v_t\\) become the 5th row of the cache. They never change again: in the next steps, they will only be read.' },
        { t: 'Scores: q against each key', d: '\\(s_j = q_t\\cdot k_j/\\sqrt{4}\\) for the 5 rows. Here decode <b>reads the whole cache</b>: every step goes through all the stored keys.' },
        { t: 'Softmax', d: 'The scores become positive weights that sum to 1. “roof” looks mostly at “the” and “climbed”.' },
        { t: 'Mixing the values', d: '\\(o_t = \\sum_j \\alpha_j v_j\\): the rows of V are read and mixed. The output moves on to the next layer, which has its own cache.' },
      ], { autoplay: 5200 });
      const ctl = UI.controls(f);
      UI.toggle(ctl, { label: 'no KV cache (recompute everything)', value: false, onChange: (v) => { S.noCache = v; } });
      V.fig(f, (g, w, h, t) => {
        const C = V.C();
        const i = st.i, e = st.elapsed();
        const wide = w >= 760;
        const lab = wide ? 62 : 56;
        const c = wide ? 34 : Math.floor(clamp(Math.min((w - 84) / 8, (w - lab - 116) / 5), 24, 32));
        const gA = wide ? 8 : 18;
        // ---------- positions ----------
        const pos = {};
        pos.x = { x: wide ? lab : 14, y: 34 };
        const wbX = pos.x.x + 4 * c + (wide ? 26 : 16);
        pos.w = { x: wbX, y: 34 };
        const qkvX = wbX + 34 + (wide ? 26 : 14);
        pos.q = { x: qkvX, y: 34 }; pos.k = { x: qkvX, y: 34 + c + gA }; pos.v = { x: qkvX, y: 34 + 2 * (c + gA) };
        const rowB = 34 + 3 * c + 2 * gA + 50;
        pos.K = { x: lab, y: rowB };
        pos.s = { x: lab + 4 * c + 18, y: rowB };
        const sW = c + 16;
        pos.a = { x: pos.s.x + sW + 16, y: rowB };
        const aW = wide ? 76 : 62;
        if (wide) { pos.V = { x: pos.a.x + aW + 70, y: rowB }; pos.o = { x: pos.a.x + aW + 70 + 4 * c + 40, y: rowB + 2 * c }; }
        else { const rowC = rowB + 5 * c + 34; pos.V = { x: lab, y: rowC }; pos.o = { x: lab + 4 * c + 34, y: rowC + 2 * c }; }
        const vec = (p, data, o2 = {}) => { for (let j = 0; j < 4; j++) { if (o2.upto != null && j >= o2.upto) { V.rr(g, p.x + j * c, p.y, c - 2, c - 2, 4); g.strokeStyle = V.alpha(C.muted, 0.35); g.setLineDash([3, 3]); g.lineWidth = 1; g.stroke(); g.setLineDash([]); continue; } cell(g, p.x + j * c, p.y, c, data[j], { alpha: o2.alpha, stroke: o2.stroke }); } };
        // ---------- row 1: h_t → W → q,k,v ----------
        const xin = i === 0 ? clamp(e / 0.6, 0, 1) : 1;
        V.text(g, 'h_t  (' + TOK[4] + ')', pos.x.x, pos.x.y - 10, { size: 11, font: 'mono', color: C.muted });
        vec(pos.x, xv, { alpha: xin, stroke: i <= 1 ? C.a : null });
        if (i === 0) V.glow(g, pos.x.x + 2 * c, pos.x.y + c / 2, 2 * c, C.a, 0.3 * (1 - xin) + 0.1);
        const names = ['W_Q', 'W_K', 'W_V'];
        const pr = i === 1 ? clamp(e / 1.3, 0, 1) : i > 1 ? 1 : 0;
        for (let k = 0; k < 3; k++) {
          const yy = 34 + k * (c + gA);
          const hot = i === 1;
          V.box(g, wbX, yy, 34, c - 2, { r: 6, fill: V.alpha(C.b, hot ? 0.25 : 0.1), stroke: V.alpha(C.b, hot ? 0.9 : 0.4), lw: 1.2 });
          V.text(g, names[k], wbX + 17, yy + (c - 2) / 2 + 0.5, { size: 11, font: 'mono', color: C.ink, align: 'center', base: 'middle' });
          V.arrow(g, pos.x.x + 4 * c + 2, pos.x.y + c / 2, wbX - 3, yy + c / 2 - 1, { color: V.alpha(C.muted, hot ? 0.8 : 0.35), lw: 1.2, head: 5 });
          V.arrow(g, wbX + 36, yy + c / 2 - 1, qkvX - 4, yy + c / 2 - 1, { color: V.alpha(C.muted, hot ? 0.8 : 0.35), lw: 1.2, head: 5 });
        }
        const upto = Math.floor(pr * 4 + 0.001);
        const vlab = (s2, p2, a2) => (wide ? V.text(g, s2, qkvX + 4 * c + 6, p2.y + c / 2 + 3, { size: 11, font: 'mono', color: C.muted, alpha: a2 }) : V.text(g, s2, qkvX, p2.y - 4, { size: 11, font: 'mono', color: C.muted, alpha: a2 }));
        vlab('q_t', pos.q, 1);
        vec(pos.q, q, { upto: i >= 1 ? upto : 0, stroke: i === 3 ? C.c : null });
        // k and v: in step 2 they travel to the cache
        const mv2 = i === 2 ? V.ease.inOut(clamp(e / 1.1, 0, 1)) : i > 2 ? 1 : 0;
        const kTarget = { x: pos.K.x, y: pos.K.y + 4 * c }, vTarget = { x: pos.V.x, y: pos.V.y + 4 * c };
        if (mv2 < 1) {
          const kp = { x: lerp(pos.k.x, kTarget.x, mv2), y: lerp(pos.k.y, kTarget.y, mv2) };
          const vp = { x: lerp(pos.v.x, vTarget.x, mv2), y: lerp(pos.v.y, vTarget.y, mv2) };
          vec(kp, kn, { upto: i >= 1 ? upto : 0, stroke: i === 2 ? C.a : null });
          vec(vp, vn, { upto: i >= 1 ? upto : 0, stroke: i === 2 ? C.b : null });
        }
        vlab('k_t', pos.k, mv2 < 1 ? 1 : 0.35);
        vlab('v_t', pos.v, mv2 < 1 ? 1 : 0.35);
        // ---------- caches ----------
        const recalc = S.noCache && (i === 1 || i === 2);
        const drawCache = (p, data, lbl, col) => {
          V.text(g, lbl, p.x, p.y - 10, { size: 11, font: 'mono', color: C.muted });
          for (let rI = 0; rI < 5; rI++) {
            V.text(g, TOK[rI], p.x - 6, p.y + rI * c + (c - 2) / 2 + 0.5, { size: 11, font: 'mono', color: rI === 4 ? C.ink : C.muted, align: 'right', base: 'middle' });
            const has = rI < 4 || mv2 >= 1;
            for (let j = 0; j < 4; j++) {
              if (!has) { V.rr(g, p.x + j * c, p.y + rI * c, c - 2, c - 2, 4); g.strokeStyle = V.alpha(col, 0.5); g.setLineDash([3, 3]); g.lineWidth = 1; g.stroke(); g.setLineDash([]); continue; }
              const hiRow = (i === 3 && p === pos.K && rI === Math.min(4, Math.floor(e / 0.45))) || (i === 5 && p === pos.V);
              cell(g, p.x + j * c, p.y + rI * c, c, data[rI][j], { alpha: rI < 4 && !hiRow && !recalc ? 0.62 : 1, stroke: hiRow ? C.c : (rI === 4 && i === 2 ? col : null) });
            }
            if (recalc && rI < 4) {
              const fl = 0.5 + 0.5 * Math.sin(t * 8 + rI);
              V.rr(g, p.x - 2, p.y + rI * c - 2, 4 * c + 2, c + 2, 5); g.strokeStyle = V.alpha(C.red, 0.5 + 0.4 * fl); g.lineWidth = 1.8; g.stroke();
            }
          }
          if (recalc) V.text(g, 'recomputing 4 rows', p.x + 2 * c, p.y + 5 * c + 12, { size: 11, font: 'mono', weight: 600, color: C.red, align: 'center' });
        };
        drawCache(pos.K, Ka, 'K cache', C.a);
        drawCache(pos.V, Va, 'V cache', C.b);
        // ---------- scores ----------
        V.text(g, 'scores s', pos.s.x, pos.s.y - 10, { size: 11, font: 'mono', color: C.muted });
        const sUpto = i === 3 ? Math.min(5, Math.floor(e / 0.45) + 1) : i > 3 ? 5 : 0;
        for (let rI = 0; rI < 5; rI++) {
          const x = pos.s.x, y = pos.s.y + rI * c;
          if (rI < sUpto) { cell(g, x, y, c, sc[rI], { wd: sW, scale: 2, fmt: (v) => nf(v, 2) }); }
          else { V.rr(g, x, y, sW - 2, c - 2, 4); g.strokeStyle = V.alpha(C.muted, 0.3); g.setLineDash([3, 3]); g.lineWidth = 1; g.stroke(); g.setLineDash([]); }
          if (i === 3 && rI === sUpto - 1) V.arrow(g, pos.K.x + 4 * c, y + c / 2 - 1, x - 3, y + c / 2 - 1, { color: C.c, lw: 1.4, head: 5 });
        }
        // ---------- weights α ----------
        V.text(g, 'weights α', pos.a.x, pos.a.y - 10, { size: 11, font: 'mono', color: C.muted });
        const ag = i === 4 ? V.ease.out(clamp(e / 1.2, 0, 1)) : i > 4 ? 1 : 0;
        for (let rI = 0; rI < 5; rI++) {
          const y = pos.a.y + rI * c + 3;
          V.box(g, pos.a.x, y, aW, c - 8, { r: 4, fill: V.alpha(C.muted, 0.1) });
          if (ag > 0) {
            V.box(g, pos.a.x, y, Math.max(2, aW * al[rI] * ag), c - 8, { r: 4, fill: V.alpha(C.d, 0.8) });
            V.text(g, nf(al[rI] * ag, 2), pos.a.x + aW - 3, y + (c - 8) / 2 + 0.5, { size: 11, font: 'mono', color: C.ink, align: 'right', base: 'middle' });
          }
        }
        // ---------- output ----------
        const og = i === 5 ? clamp((e - 0.6) / 1.4, 0, 1) : 0;
        V.text(g, 'o_t', pos.o.x, pos.o.y - 10, { size: 11, font: 'mono', color: C.muted });
        vec(pos.o, o, { upto: i === 5 ? Math.floor(og * 4 + 0.001) : 0, stroke: i === 5 && og >= 1 ? C.a : null });
        if (i === 5) {
          for (let rI = 0; rI < 5; rI++) {
            const x1 = pos.V.x + 4 * c, y1 = pos.V.y + rI * c + c / 2;
            const u = (t * 0.9 + rI * 0.13) % 1;
            const pt = [lerp(x1, pos.o.x - 4, u), lerp(y1, pos.o.y + c / 2, u)];
            V.line(g, x1, y1, pos.o.x - 4, pos.o.y + c / 2, { color: V.alpha(C.d, 0.2 + 0.7 * al[rI]), lw: 0.8 + 5 * al[rI] });
            V.dot(g, pt[0], pt[1], 1.5 + 4 * al[rI], C.d);
          }
        }
        // ---------- footer ----------
        const yb = h - (wide ? 30 : 74);
        const p1 = S.noCache ? (wide ? 'NO CACHE: redoes K and V for the 5 rows (4 of them repeated) and, in the layers below, their entire forward pass' : 'NO CACHE: redoes K and V for the 5 rows and their forward pass in the layers below') :'WITH CACHE: this step does 1 new K projection and 1 new V projection and reads 5 rows';
        const yb2 = textBlock(g, p1, 14, yb, w - 28, { size: 12, font: 'mono', weight: 600, color: S.noCache ? C.red : C.a, lh: 16 });
        textBlock(g, 'generating n tokens: n token-forwards with cache, n(n+1)/2 without · n = 100 thousand → 100 thousand vs. 5 billion', 14, yb2 + 4, w - 28, { size: 11, font: 'mono', color: C.muted, lh: 15 });
      });
    }

    /* ===================== Figure 6: the KV cache as a 3D tensor ===================== */
    {
      const f = $('#fig-kv-tensor');
      const stage = f.querySelector('.la-fig-stage');
      const narrow = (f.clientWidth || 800) < 560;
      const NS = 4, NR = 12, NCOL = 8;
      const S = { m: 'qwen', n: 7, tp: false };
      const EMPTY = 'color-mix(in srgb, var(--viz-muted) 12%, transparent)';
      const mix = (v, p) => `color-mix(in srgb, var(${v}) ${p}%, transparent)`;
      const stripe = (v) => `repeating-linear-gradient(90deg, color-mix(in srgb, var(${v}) 85%, transparent) 0 3px, color-mix(in srgb, var(${v}) 30%, transparent) 3px 5px)`;
      const REPL = 'linear-gradient(135deg, var(--viz-a) 0 50%, var(--viz-d) 50% 100%)';
      const HATCH = 'repeating-linear-gradient(45deg, color-mix(in srgb, var(--viz-muted) 55%, transparent) 0 2px, transparent 2px 4px)';
      const isAtt = (s) => S.m === 'mha' || s === NS - 1;
      const colorFor = (s, i, j) => {
        const n = S.n, tp = S.tp;
        if (!isAtt(s)) { // fixed state (GDN / KDA): does not grow with the tokens
          if (n === 0 || i >= 4) return EMPTY;
          if (!tp) return stripe('--viz-e');
          return j < 4 ? stripe('--viz-a') : stripe('--viz-d');
        }
        if (i >= n) return EMPTY;
        if (S.m === 'mha') { const head = j >> 1, isV = j & 1; if (tp) return head < 2 ? mix('--viz-a', isV ? 50 : 90) : mix('--viz-d', isV ? 50 : 90); return isV ? mix('--viz-b', 85) : mix('--viz-a', 85); }
        if (S.m === 'qwen') { const head = j >> 2, isV = (j % 4) >= 2; if (tp) return head === 0 ? mix('--viz-a', isV ? 50 : 90) : mix('--viz-d', isV ? 50 : 90); return isV ? mix('--viz-b', 85) : mix('--viz-a', 85); }
        // GLM (MLA): 0–5 latent, 6 zeroed RoPE, 7 indexer (1 entry every 4 tokens)
        if (j === 6) return HATCH;
        if (j === 7) { if ((i + 1) % 4 !== 0) return EMPTY; return tp ? REPL : mix('--viz-d', 90); }
        return tp ? REPL : mix('--viz-f', 80);
      };
      // labels only on the front slab (below it) and on the back slab (above it): nothing covers them
      const lab = (s) => {
        if (s !== 0 && s !== NS - 1) return '';
        const front = s === NS - 1;
        if (S.m === 'mha') return front ? 'layer 4 · KV grows' : 'layer 1 · KV grows';
        if (S.m === 'qwen') return front ? 'QSA · KV grows' : '3 × GDN · fixed state';
        return front ? 'MLA · KV grows' : '3 × KDA · fixed state';
      };
      const T = UI.tensor3d(stage, { slabs: NS, rows: NR, cols: NCOL, cell: narrow ? 12 : 16, depthGap: narrow ? 40 : 58, rotate: true, color: colorFor, slabColor: () => 'var(--viz-a)', labels: Array.from({ length: NS }, (_, s) => lab(s)), rx: -22, ry: -30 });
      // vertical room for the labels above the back slab and below the front slab (the wrapper clips the excess)
      T.el.style.padding = narrow ? '46px 10px' : '64px 10px';
      T.slabs.forEach((sl, s) => {
        const lb = sl.querySelector('.slab-label');
        if (!lb) return;
        const front = s === NS - 1;
        Object.assign(lb.style, {
          left: '0', marginLeft: '0', transform: 'none', padding: '2px 7px', borderRadius: '6px',
          background: 'var(--surface-solid)', border: '1px solid color-mix(in srgb, var(--sc) 60%, transparent)', color: 'var(--text-2)',
          top: front ? '100%' : 'auto', bottom: front ? 'auto' : '100%', marginTop: front ? '8px' : '0', marginBottom: front ? '0' : '8px',
        });
      });
      const DATA = {
        mha: { real: '2,949,120 B', gpu: '1,474,560 B', grow: '45 of 45', leg: [['var(--viz-a)', 'K'], ['var(--viz-b)', 'V']] },
        qwen: { real: '26,624 B (BF16)', gpu: '13,312 B', grow: '12 of 48 (+ MTP)', leg: [['var(--viz-a)', 'K'], ['var(--viz-b)', 'V'], [stripe('--viz-e'), 'GDN state (striped)']] },
        glm: { real: '7,579 B', gpu: '7,579 B (replicated)', grow: '11 of 45', leg: [['var(--viz-f)', 'MLA latent (FP8)'], ['var(--viz-d)', 'indexer (1 every 4)'], [HATCH, 'zeroed RoPE'], [stripe('--viz-e'), 'KDA state (striped)']] },
      };
      const ctl = UI.controls(f);
      UI.seg(ctl, { label: 'Model', options: [{ v: 'mha', label: 'classic MHA' }, { v: 'qwen', label: 'Qwen (GQA)' }, { v: 'glm', label: 'GLM (MLA)' }], value: 'qwen', onChange: (v) => { S.m = v; repaint(); } });
      UI.slider(ctl, { label: 'Tokens', min: 0, max: NR, step: 1, value: S.n, fmt: (v) => String(v), onInput: (v) => { S.n = v; repaint(); } });
      UI.toggle(ctl, { label: 'TP=2: split across 2 GPUs', value: false, onChange: (v) => { S.tp = v; repaint(); } });
      const r1_ = UI.readout(ctl, 'bytes per token', '—');
      const r2_ = UI.readout(ctl, 'per GPU with TP=2', '—');
      const r3_ = UI.readout(ctl, 'layers that grow', '—');
      const legBox = LA.h('div', { class: 'la-legend' });
      ctl.appendChild(legBox);
      function repaint() {
        for (let s = 0; s < NS; s++) {
          for (let i = 0; i < NR; i++) for (let j = 0; j < NCOL; j++) T.setCell(s, i, j, colorFor(s, i, j));
          const sl = T.slabs[s];
          sl.style.setProperty('--sc', isAtt(s) ? 'var(--viz-a)' : 'var(--viz-e)');
          const lb = sl.querySelector('.slab-label');
          if (lb) lb.textContent = lab(s);
        }
        const D = DATA[S.m];
        r1_.set(D.real); r2_.set(D.gpu); r3_.set(D.grow);
        legBox.innerHTML = '';
        let legs = D.leg.slice();
        if (S.tp) legs = S.m === 'glm' ? [['var(--viz-a)', 'GPU A'], ['var(--viz-d)', 'GPU B'], [REPL, 'two-tone = full copy on both GPUs']] : [['var(--viz-a)', 'GPU A (strong = K, faint = V)'], ['var(--viz-d)', 'GPU B']];
        legs.forEach(([c, l]) => legBox.appendChild(LA.h('span', null, LA.h('i', { style: { '--c': c } }), l)));
      }
      repaint();
      // head strip: which asker (Q) queries which stored K/V
      const hc = document.createElement('canvas');
      hc.style.cssText = 'display:block;width:100%;height:122px;margin-top:6px';
      stage.appendChild(hc);
      V.loop(hc, (g, w, h, t) => {
        const C = V.C();
        const nr = w < 480;
        const tp = S.tp;
        const cfg = S.m === 'mha' ? { nQ: 16, nKV: 16, title: nr ? 'MHA: 1 K/V pair per head' : 'MHA: each query head has its own K/V pair (16 drawn)' }
          : S.m === 'qwen' ? { nQ: 24, nKV: 2, title: nr ? 'GQA: 24 queries → 2 K/V pairs' : 'GQA (Qwen): 24 query heads share 2 K/V pairs, 12 per pair' }
          : { nQ: nr ? 24 : 32, nKV: 1, title: nr ? 'MLA: 64 queries → 1 latent' : 'MLA (GLM): all 64 query heads read 1 compressed latent (' + (nr ? 24 : 32) + ' drawn)' };
        const x0 = nr ? 34 : 46, x1 = w - 6, yQ = 42, yK = 94, bh = 22;
        const sq = (x1 - x0) / cfg.nQ;
        const qx = (i) => x0 + (i + 0.5) * sq;
        const half = (x0 + x1) / 2;
        V.text(g, fit(g, cfg.title, w - 8, 11.5, 'mono', 600), 4, 14, { size: 11.5, font: 'mono', weight: 600, color: C.ink });
        V.text(g, 'Q', x0 - 10, yQ + 4, { size: 11, font: 'mono', weight: 600, color: C.c, align: 'right' });
        V.text(g, nr ? 'KV' : 'K/V', x0 - 10, yK + 4, { size: 11, font: 'mono', weight: 600, color: C.a, align: 'right' });
        if (tp) {
          g.fillStyle = V.alpha(C.a, 0.06); g.fillRect(x0, 24, half - x0 - 2, h - 26);
          g.fillStyle = V.alpha(C.d, 0.06); g.fillRect(half + 2, 24, x1 - half - 2, h - 26);
          V.text(g, 'GPU A', half - 6, h - 6, { size: 11, font: 'mono', color: C.a, align: 'right' });
          V.text(g, 'GPU B', x1 - 4, h - 6, { size: 11, font: 'mono', color: C.d, align: 'right' });
        }
        // K/V boxes: position and owner of each query head
        const boxes = [];
        if (cfg.nKV === 1 && tp) { // replicated MLA: a full copy on each GPU
          const bw = Math.min(150, (half - x0) * 0.8);
          boxes.push({ x: (x0 + half) / 2 - bw / 2, w: bw, gpu: 0, lbl: nr ? 'copy A' : 'latent (copy A)' }, { x: (half + x1) / 2 - bw / 2, w: bw, gpu: 1, lbl: nr ? 'copy B' : 'latent (copy B)' });
        } else {
          const sk = (x1 - x0) / cfg.nKV;
          for (let k = 0; k < cfg.nKV; k++) {
            const bw = cfg.nKV === 1 ? Math.min(180, (x1 - x0) * 0.5) : Math.max(8, sk - (cfg.nKV > 4 ? 4 : 28));
            boxes.push({ x: x0 + (k + 0.5) * sk - bw / 2, w: bw, gpu: k < cfg.nKV / 2 ? 0 : 1, lbl: cfg.nKV === 1 ? 'latent' : cfg.nKV === 2 ? 'K/V ' + (k + 1) : '' });
          }
        }
        const owner = (i) => {
          if (cfg.nKV === 1) return tp ? (i < cfg.nQ / 2 ? 0 : 1) : 0;
          return Math.floor(i * cfg.nKV / cfg.nQ);
        };
        const gpuCol = (gi) => (gi === 0 ? C.a : C.d);
        for (let i = 0; i < cfg.nQ; i++) {
          const b = boxes[owner(i)];
          const bx = b.x + b.w / 2, x = qx(i);
          V.line(g, x, yQ + 5, bx, yK - bh / 2, { color: V.alpha(tp ? gpuCol(b.gpu) : C.muted, 0.28), lw: 1 });
          const u = (t * 0.7 + i * 0.137) % 1;
          V.dot(g, lerp(bx, x, u), lerp(yK - bh / 2, yQ + 5, u), 1.8, tp ? gpuCol(b.gpu) : C.a);
        }
        boxes.forEach((b) => {
          const y = yK - bh / 2;
          if (S.m === 'glm') {
            V.box(g, b.x, y, b.w, bh, { r: 6, fill: V.alpha(tp ? gpuCol(b.gpu) : C.f, 0.35), stroke: tp ? gpuCol(b.gpu) : C.f, lw: 1.4 });
          } else {
            const kc = tp ? gpuCol(b.gpu) : C.a, vc = tp ? gpuCol(b.gpu) : C.b;
            V.box(g, b.x, y, b.w / 2, bh, { r: 4, fill: V.alpha(kc, tp ? 0.55 : 0.5) });
            V.box(g, b.x + b.w / 2, y, b.w / 2, bh, { r: 4, fill: V.alpha(vc, tp ? 0.28 : 0.5) });
          }
          if (b.lbl && tw(g, b.lbl, 11, 'mono', 600) < b.w - 6) V.text(g, b.lbl, b.x + b.w / 2, yK + 0.5, { size: 11, font: 'mono', weight: 600, color: C.ink, align: 'center', base: 'middle' });
        });
        for (let i = 0; i < cfg.nQ; i++) {
          const b = boxes[owner(i)];
          const col = tp ? gpuCol(b.gpu) : C.c;
          const rr = Math.max(2.5, Math.min(5.5, sq * 0.32));
          g.beginPath(); g.arc(qx(i), yQ, rr, 0, Math.PI * 2); g.fillStyle = V.alpha(col, 0.35); g.fill(); g.strokeStyle = col; g.lineWidth = 1.3; g.stroke();
        }
      });
    }

    /* ===================== Figure 7: the KV pool ===================== */
    {
      const f = $('#fig-pool');
      fitH(f, 362, 348);
      const CTX = [4096, 8192, 16384, 32768, 65536, 131072, 262144];
      const POOL = { fp8: { tok: 3652200, gib: 32.02 }, bf16: { tok: 2131159, gib: 31.7 } };
      const SEQS = 8;
      const S = { dt: 'fp8', ci: 4, n: 6, anim: [] };
      const ctl = UI.controls(f);
      const st = () => { const N = POOL[S.dt].tok, ctx = CTX[S.ci]; const capMem = Math.floor(N / ctx); const adm = Math.min(S.n, SEQS, capMem); const waitSeq = Math.max(0, Math.min(S.n, capMem) - adm); const waitMem = Math.max(0, S.n - adm - waitSeq); return { N, ctx, capMem, adm, waitSeq, waitMem }; };
      const upd = () => { const s = st(); rC.set(nf(s.capMem) + ' conversations'); rA.set(s.adm + ' of ' + S.n); rU.set(nf(100 * s.adm * s.ctx / s.N, 1) + '%'); };
      UI.seg(ctl, { label: 'KV', options: [{ v: 'fp8', label: 'FP8' }, { v: 'bf16', label: 'BF16' }], value: 'fp8', onChange: (v) => { S.dt = v; upd(); } });
      UI.slider(ctl, { label: 'Context', min: 0, max: CTX.length - 1, step: 1, value: S.ci, fmt: (v) => nf(CTX[v]), onInput: (v) => { S.ci = v; upd(); } });
      UI.slider(ctl, { label: 'Conversations', min: 1, max: 20, step: 1, value: S.n, fmt: (v) => String(v), onInput: (v) => { S.n = v; upd(); } });
      const rC = UI.readout(ctl, 'fit in the pool', '—');
      const rA = UI.readout(ctl, 'admitted', '—');
      const rU = UI.readout(ctl, 'pool usage', '—');
      upd();
      V.fig(f, (g, w, h, t, dt) => {
        const C = V.C();
        const narrow = w < 620;
        const pad = 16;
        const s = st();
        // ---------- node budget ----------
        const tot = 121.69, kvG = POOL[S.dt].gib;
        const segs = [['weights + non-torch', 68.52, C.b], ['activation', 1.07, C.d], ['KV pool', kvG, C.a], ['outside the budget (OS, host)', tot - 68.52 - 1.07 - kvG, C.muted]];
        V.text(g, fit(g, narrow ? 'one Qwen GPU · 121.69 GiB' : 'memory of one Qwen GPU (TP2, GMU 0.835) · 121.69 GiB', w - 2 * pad, 11.5, 'mono'), pad, 18, { size: 11.5, font: 'mono', color: C.muted });
        const by = 28, bh = 26, bW = w - 2 * pad;
        let x = pad;
        segs.forEach((sg) => {
          const ww = bW * sg[1] / tot;
          V.box(g, x, by, Math.max(1, ww - 1.5), bh, { r: 5, fill: V.alpha(sg[2], sg[0].startsWith('outside') ? 0.18 : 0.7) });
          const lbl = sg[0].startsWith('outside') ? (narrow ? 'host' : 'OS/host') : sg[0].startsWith('weights') ? (narrow ? 'weights' : 'weights + non-torch') : sg[0] === 'KV pool' ? (narrow ? 'KV' : 'KV pool') : '';
          const txt = lbl ? lbl + ' ' + nf(sg[1], sg[1] < 10 ? 2 : 1) : '';
          if (txt && tw(g, txt, 11, 'mono', 600) < ww - 8) V.text(g, txt, x + 6, by + bh / 2 + 0.5, { size: 11, font: 'mono', weight: 600, color: sg[0].startsWith('outside') ? C.ink : onInk(), base: 'middle' });
          x += ww;
        });
        V.text(g, fit(g, 'peak activation 1.07 GiB (thin slice) · CUDA graphs 0.54 GiB reported separately', bW, 11), pad, by + bh + 16, { size: 11, color: C.muted });
        // ---------- pool ----------
        const py = by + bh + 50, ph = narrow ? 64 : 76;
        V.text(g, 'KV pool: ' + nf(s.N) + ' tokens (' + (S.dt === 'fp8' ? 'FP8' : 'BF16') + ')', pad, py - 10, { size: 12.5, font: 'mono', weight: 600, color: C.a });
        V.box(g, pad, py, bW, ph, { r: 10, fill: V.alpha(C.a, 0.05), stroke: V.alpha(C.a, 0.55), lw: 1.3 });
        // background block grid
        g.strokeStyle = V.alpha(C.a, 0.1); g.lineWidth = 1;
        for (let k = 1; k < 40; k++) { const xx = pad + (bW * k) / 40; g.beginPath(); g.moveTo(xx, py + 3); g.lineTo(xx, py + ph - 3); g.stroke(); }
        // admitted conversations (animated)
        while (S.anim.length < 20) S.anim.push(0);
        const segW = bW * s.ctx / s.N;
        for (let k = 0; k < 20; k++) {
          const target = k < s.adm ? 1 : 0;
          S.anim[k] += (target - S.anim[k]) * Math.min(1, dt * 7);
        }
        let cx = pad + 2;
        for (let k = 0; k < 20; k++) {
          const a = S.anim[k];
          if (a < 0.01) continue;
          const ww = Math.max(0, segW * a - 2);
          const col = C.series[k % C.series.length];
          V.box(g, cx, py + 3, Math.max(2, ww), ph - 6, { r: 6, fill: V.alpha(col, 0.55), stroke: col, lw: 1 });
          if (ww > 22) guy(g, cx + Math.min(ww / 2, 16), py + ph / 2, Math.min(9, ph / 5), col, { blink: blinkAt(t, k), alpha: a });
          cx += segW * a;
        }
        // max_model_len mark
        const mml = 262144;
        const mx = pad + bW * mml / s.N;
        V.line(g, mx, py - 3, mx, py + ph + 3, { color: C.d, lw: 1.5, dash: [3, 3] });
        V.text(g, fit(g, (narrow ? '1 conversation of 262,144 = ' : '1 conversation of 262,144 (max-model-len) = ') + nf(100 * mml / s.N, 1) + '% of the pool', w - 2 * pad, 11, 'mono'), pad, py + ph + 15, { size: 11, font: 'mono', color: C.d });
        // ---------- queue ----------
        const qy = py + ph + 44;
        V.text(g, 'scheduler slots (max_num_seqs = 8):', pad, qy, { size: 11.5, font: 'mono', color: C.muted });
        const seatR = narrow ? 8 : 9;
        for (let k = 0; k < SEQS; k++) {
          const sx = pad + 10 + k * (seatR * 2 + 8), sy = qy + 20;
          const occ = k < s.adm;
          if (occ) guy(g, sx + seatR, sy + seatR, seatR, C.series[k % C.series.length], { blink: blinkAt(t, k + 3) });
          else { g.beginPath(); g.arc(sx + seatR, sy + seatR, seatR, 0, Math.PI * 2); g.strokeStyle = V.alpha(C.muted, 0.4); g.setLineDash([3, 3]); g.lineWidth = 1.2; g.stroke(); g.setLineDash([]); }
        }
        const wy = qy + 20 + seatR * 2 + 24;
        const wait = s.waitSeq + s.waitMem;
        // queue lane (empty or not), so the reader sees where the excess waits
        V.rr(g, pad, wy + 8, w - 2 * pad, seatR * 2 + 12, 9);
        g.fillStyle = V.alpha(wait > 0 ? (s.waitMem > 0 ? C.red : C.d) : C.muted, 0.05); g.fill();
        g.strokeStyle = V.alpha(wait > 0 ? (s.waitMem > 0 ? C.red : C.d) : C.muted, 0.35); g.lineWidth = 1; g.setLineDash([4, 4]); g.stroke(); g.setLineDash([]);
        if (wait === 0) V.text(g, 'queue', w - pad - 10, wy + 8 + seatR + 10, { size: 11, font: 'mono', color: C.muted, align: 'right', alpha: 0.7 });
        if (wait > 0) {
          const both = s.adm === SEQS && s.adm === s.capMem;
          const why = both ? 'no slot and no free KV' : s.waitSeq > 0 && s.waitMem > 0 ? s.waitSeq + ' without a slot + ' + s.waitMem + ' without free KV' : s.waitSeq > 0 ? 'no scheduler slot (KV still free)' : 'no free KV in the pool';
          V.text(g, fit(g, 'in queue: ' + wait + ' · ' + why, w - 2 * pad, 12, 'mono', 600), pad, wy, { size: 12, font: 'mono', weight: 600, color: s.waitMem > 0 ? C.red : C.d });
          const maxShow = Math.floor((w - 2 * pad) / (seatR * 2 + 4));
          for (let k = 0; k < Math.min(wait, maxShow); k++) {
            const isMem = k >= s.waitSeq;
            guy(g, pad + seatR + k * (seatR * 2 + 4), wy + 18 + seatR, seatR * 0.85, isMem ? C.red : C.d, { alpha: 0.85, blink: blinkAt(t, k + 7) });
          }
        } else {
          V.text(g, fit(g, narrow ? 'queue empty' : 'queue empty: every conversation was admitted', w - 2 * pad, 12, 'mono', 600), pad, wy, { size: 12, font: 'mono', weight: 600, color: C.a });
        }
        const concur = s.N / 262144;
        V.text(g, fit(g, (narrow ? 'concurrency at 262,144: ' : 'max concurrency at 262,144 tokens: ') + nf(concur, 2) + '×' + (narrow ? '' : ' (vLLM boot)'), w - 2 * pad, 11, 'mono'), pad, h - 10, { size: 11, font: 'mono', color: C.muted });
      });
    }

    /* ===================== Figure 8: the clock of a request ===================== */
    {
      const f = $('#fig-relogio');
      fitH(f, 370, 400);
      const SC = [
        { v: 'ds256', label: 'DeepSeek · prompt 256', P: 256, ttft: 0.63, pre: 447, dec: 75.4, col: 'ds' },
        { v: 'ds2k', label: 'DeepSeek · prompt 2,048', P: 2048, ttft: 0.81, pre: 2563, dec: 68.8, col: 'ds' },
        { v: 'ds8k', label: 'DeepSeek · prompt 8,192', P: 8192, ttft: 4.8, pre: 1713, dec: 73.9, col: 'ds' },
        { v: 'ds32k', label: 'DeepSeek · prompt 32,768', P: 32768, ttft: 22.96, pre: 1428, dec: 64.0, col: 'ds' },
        { v: 'ds131k', label: 'DeepSeek · prompt 131,072', P: 131072, ttft: 78.75, pre: 1665, dec: 65.2, col: 'ds' },
        { v: 'glmS', label: 'GLM · structured ×1 (Aug 28)', P: null, ttft: 0.719, dec: 62.9, col: 'glm' },
        { v: 'qwP', label: 'Qwen · prose ×1', P: null, ttft: 0.16, dec: 54.4, col: 'qwen' },
      ];
      const S = { sc: 'ds131k', N: 512, t0: 0, reset: true };
      const cur = () => SC.find((s) => s.v === S.sc);
      const calc = () => { const s = cur(); const dec = (S.N - 1) / s.dec; const e2e = s.ttft + dec; return { s, dec, e2e, wall: S.N / e2e, tpot: 1000 / s.dec }; };
      const ctl = UI.controls(f);
      const upd = () => { const r = calc(); rT.set(r.s.ttft < 1 ? nf(r.s.ttft * 1000) + ' ms' : nf(r.s.ttft, 2) + ' s'); rP.set(nf(r.tpot, 1) + ' ms'); rE.set(nf(r.e2e, 1) + ' s'); rW.set(nf(r.wall, 1) + ' tok/s'); S.reset = true; };
      UI.select(ctl, { label: 'Scenario', options: SC.map((s) => ({ v: s.v, label: s.label })), value: S.sc, onChange: (v) => { S.sc = v; upd(); } });
      UI.slider(ctl, { label: 'Output tokens N', min: 64, max: 2048, step: 64, value: 512, fmt: (v) => String(v), onInput: (v) => { S.N = v; upd(); } });
      const rT = UI.readout(ctl, 'TTFT', '—');
      const rP = UI.readout(ctl, 'TPOT', '—');
      const rE = UI.readout(ctl, 'E2EL', '—');
      const rW = UI.readout(ctl, 'wall-clock tok/s', '—');
      upd();
      V.fig(f, (g, w, h, t) => {
        const C = V.C();
        const narrow = w < 620;
        const pad = 16;
        if (S.reset) { S.t0 = t; S.reset = false; }
        const r = calc(), s = r.s;
        const rc = C[s.col] || C.a;
        // header: reading × writing
        V.text(g, s.label, pad, 20, { size: 13, font: 'display', weight: 600, color: rc });
        const gy = 34, gw = w - 2 * pad;
        const pre = s.pre || null;
        const gmax = Math.max(pre || 0, s.dec) * 1.05;
        const gauge = (y, lbl, val, col, note) => {
          V.text(g, lbl, pad, y + 11, { size: 11.5, color: C.muted });
          const x0 = pad + (narrow ? 92 : 128), ww = gw - (narrow ? 92 : 128);
          V.box(g, x0, y, ww, 14, { r: 5, fill: V.alpha(C.muted, 0.12) });
          if (val != null) {
            V.box(g, x0, y, Math.max(4, ww * val / gmax), 14, { r: 5, fill: V.alpha(col, 0.75) });
            V.text(g, nf(val, val < 100 ? 1 : 0) + ' tok/s', x0 + Math.max(4, ww * val / gmax) + 6 > x0 + ww - 80 ? x0 + 6 : x0 + ww * val / gmax + 6, y + 11, { size: 11, font: 'mono', weight: 600, color: x0 + ww * val / gmax + 6 > x0 + ww - 80 ? onInk() : C.ink });
          } else V.text(g, note, x0 + 6, y + 11, { size: 11, color: C.muted });
        };
        gauge(gy, narrow ? 'reading' : 'reading (prefill)', pre, C.d, 'prompt size not published');
        gauge(gy + 22, narrow ? 'writing' : 'writing (decode)', s.dec, C.a, '');
        if (pre) { if (narrow) V.text(g, 'reads ' + nf(pre / s.dec, 1) + '× faster than it writes', pad, gy + 58, { size: 11.5, font: 'mono', color: C.ink }); else V.text(g, 'reads ' + nf(pre / s.dec, 1) + '× faster than it writes', w - pad, 20, { size: 11.5, font: 'mono', color: C.ink, align: 'right' }); }
        // timeline
        const ty = narrow ? 142 : 128, tH = 26, x0 = pad, x1 = w - pad;
        const X = (sec) => x0 + (sec / r.e2e) * (x1 - x0);
        const tt = X(s.ttft);
        V.box(g, x0, ty, Math.max(2, tt - x0), tH, { r: 6, fill: V.alpha(C.d, 0.7) });
        V.box(g, tt, ty, Math.max(2, x1 - tt), tH, { r: 6, fill: V.alpha(C.a, 0.35), stroke: V.alpha(C.a, 0.8), lw: 1 });
        // token ticks
        const nTicks = Math.min(S.N, Math.floor((x1 - tt) / 3));
        g.strokeStyle = V.alpha(C.a, 0.9); g.lineWidth = 1;
        for (let k = 0; k < nTicks; k++) { const xx = tt + ((k + 0.5) / nTicks) * (x1 - tt); g.beginPath(); g.moveTo(xx, ty + 5); g.lineTo(xx, ty + tH - 5); g.stroke(); }
        if (tt - x0 > 60) V.text(g, 'TTFT', x0 + 6, ty + tH / 2 + 0.5, { size: 11, font: 'mono', weight: 600, color: onInk(), base: 'middle' });
        // axis
        const nice = (v) => { const p = Math.pow(10, Math.floor(Math.log10(v))); const m = v / p; return (m < 1.5 ? 1 : m < 3.5 ? 2 : m < 7.5 ? 5 : 10) * p; };
        const step = nice(r.e2e / (narrow ? 4 : 7));
        for (let sx = 0; sx <= r.e2e + 1e-9; sx += step) { const xx = X(sx); V.line(g, xx, ty + tH + 2, xx, ty + tH + 7, { color: C.muted, lw: 1 }); V.text(g, nf(sx, step < 1 ? 1 : 0) + ' s', xx, ty + tH + 20, { size: 11, font: 'mono', color: C.muted, align: 'center' }); }
        // animated cursor (compressed time)
        const D = 7, pause = 1.2;
        const u = ((t - S.t0) % (D + pause)) / D;
        const cs = Math.min(1, u) * r.e2e;
        const cxp = X(cs);
        V.line(g, cxp, ty - 14, cxp, ty + tH + 4, { color: C.ink, lw: 1.6 });
        const emitted = cs < s.ttft ? 0 : Math.min(S.N, 1 + Math.floor((cs - s.ttft) * s.dec));
        guy(g, cxp, ty - 24, 8, cs < s.ttft ? C.d : C.a, { blink: blinkAt(t, 2), glow: cs >= s.ttft && u < 1 });
        const phase = cs < s.ttft ? 'reading the prompt (prefill)…' : u >= 1 ? 'response complete' : 'writing (decode)…';
        V.text(g, fit(g, (narrow ? '' : phase + ' · ') + 'tokens emitted: ' + emitted + '/' + S.N, w - 2 * pad, 11.5, 'mono'), pad, ty + tH + 42, { size: 11.5, font: 'mono', color: cs < s.ttft ? C.d : C.a });
        V.text(g, '1 s of animation ≈ ' + nf(r.e2e / D, r.e2e / D < 10 ? 1 : 0) + ' s of real time', w - pad, ty + tH + 42, { size: 11, font: 'mono', color: C.muted, align: 'right', alpha: narrow ? 0 : 1 });
        // numbers
        const yN = ty + tH + (narrow ? 74 : 72);
        const fracP = s.ttft / r.e2e;
        const L1 = 'E2EL = TTFT + (N−1)·TPOT = ' + (s.ttft < 1 ? nf(s.ttft, 2) : nf(s.ttft, 1)) + ' s + ' + (S.N - 1) + ' × ' + nf(r.tpot, 1) + ' ms = ' + nf(r.e2e, 1) + ' s';
        const L2 = 'prefill = ' + nf(100 * fracP, 0) + '% of the clock · wall-clock tok/s N/E2EL = ' + nf(r.wall, 1) + ' (decode: ' + nf(s.dec, 1) + ')';
        let yy = textBlock(g, L1, pad, yN, w - 2 * pad, { size: 12, font: 'mono', color: C.ink, lh: 17 });
        yy = textBlock(g, L2, pad, yy + 4, w - 2 * pad, { size: 12, font: 'mono', color: C.ink, lh: 17 });
        if (s.P && yy + 18 < h) textBlock(g, 'prefill tok/s = prompt_tokens ÷ TTFT = ' + nf(s.pre) + ' (published medians; prompt_tokens includes template and instruction)', pad, yy + 6, w - 2 * pad, { size: 12, font: 'mono', color: C.d, lh: 17 });
      });
    }

    /* ===================== Figure 9: the weights give everyone a ride ===================== */
    {
      const f = $('#fig-carona');
      fitH(f, 440, 640, 700);
      const E = 512, KE = 10, POSN = 4, BMAX = 16, SEQS = 8;
      const R = V.rng(99);
      const picks = [];
      for (let k = 0; k < BMAX * POSN; k++) { const set = new Set(); while (set.size < KE) set.add(R.int(E)); picks.push([...set]); }
      const litOrder = new Array(E).fill(-1); // index of the first token that lights up the expert
      picks.forEach((p, k) => p.forEach((e) => { if (litOrder[e] < 0) litOrder[e] = k; }));
      const S1 = 54.4;
      const MEAS_A = [[1, 54.4], [2, 86.5], [4, 128.1], [6, 169.7], [8, 207.0]];
      const MEAS_S = [[1, 54.4], [2, 45.1], [4, 34.2], [6, 29.7], [8, 26.7]];
      const S = { B: 4, phi: 0.17, meas: true, since: 0, prevB: 4, t: 0 };
      const ctl = UI.controls(f);
      // model without a slot ceiling (hypothetical) and with it: above 8 streams, the step keeps running with 8
      const modelRaw = (B) => { const s = S1 / (1 + S.phi * (B - 1)); return { s, A: B * s }; };
      const model = (B) => modelRaw(Math.min(B, SEQS));
      const distinct = (B) => { let d = 0; const T = Math.min(B, SEQS) * POSN; for (let e = 0; e < E; e++) if (litOrder[e] >= 0 && litOrder[e] < T) d++; return d; };
      const upd = () => { const m = model(S.B); rD.set(distinct(S.B) + ' of 512'); rA.set(nf(m.A, 0) + ' tok/s'); rS.set(nf(m.s, 1) + ' tok/s'); rQ.set(String(Math.max(0, S.B - SEQS))); };
      UI.slider(ctl, { label: 'Streams B', min: 1, max: BMAX, step: 1, value: S.B, fmt: (v) => String(v), onInput: (v) => { S.prevB = S.B; S.B = v; S.since = S.t; upd(); } });
      UI.slider(ctl, { label: 'Cost/stream', min: 0.05, max: 0.4, step: 0.01, value: S.phi, fmt: (v) => 'φ = ' + nf(v, 2), onInput: (v) => { S.phi = v; upd(); } });
      UI.toggle(ctl, { label: 'Qwen measurements', value: true, onChange: (v) => { S.meas = v; } });
      const rD = UI.readout(ctl, 'distinct experts/layer', '—');
      const rA = UI.readout(ctl, 'aggregate (model)', '—');
      const rS = UI.readout(ctl, 'per admitted stream', '—');
      const rQ = UI.readout(ctl, 'in queue', '—');
      upd();
      V.fig(f, (g, w, h, t) => {
        S.t = t;
        const C = V.C();
        const wide = w >= 700;
        const pad = 16;
        const Badm = Math.min(S.B, SEQS);
        // ---------- panel A: expert grid ----------
        const gw = wide ? Math.min(w * 0.44, 380) : w - 2 * pad;
        const cols = 32, rows = 16;
        const cs = Math.floor((gw - 4) / cols);
        const gx = pad, gy = 58;
        V.text(g, 'experts of one Qwen layer (512, top-10)', gx, 18, { size: 11.5, font: 'mono', color: C.muted });
        // streams (characters): the ones in the queue are faded and light up no experts
        const gsp = Math.min(22, (gw - 10) / BMAX);
        for (let b = 0; b < S.B; b++) guy(g, gx + 8 + b * gsp, 38, Math.min(8, gsp * 0.42), b < SEQS ? C.series[b % C.series.length] : C.muted, { blink: blinkAt(t, b), alpha: b < SEQS ? 1 : 0.4 });
        const T = Badm * POSN;
        const age = t - S.since;
        for (let rI = 0; rI < rows; rI++) for (let cI = 0; cI < cols; cI++) {
          const e = rI * cols + cI;
          const k = litOrder[e];
          const on = k >= 0 && k < T;
          const x = gx + cI * cs, y = gy + rI * cs;
          V.rr(g, x, y, cs - 1.5, cs - 1.5, 2);
          if (on) {
            const b = Math.floor(k / POSN);
            const fresh = S.B > S.prevB && k >= Math.min(S.prevB, SEQS) * POSN ? clamp(age / 0.6, 0, 1) : 1;
            g.fillStyle = V.alpha(C.series[b % C.series.length], 0.35 + 0.5 * fresh); g.fill();
          } else { g.fillStyle = V.alpha(C.muted, 0.12); g.fill(); }
        }
        const D = distinct(S.B);
        const ED = E * (1 - Math.pow(1 - KE / E, T));
        const gyE = gy + rows * cs + 18;
        const nrw = gw < 420;
        const qTxt = S.B > SEQS ? ' (' + (S.B - SEQS) + ' in queue)' : '';
        V.text(g, fit(g, (nrw ? 'T = ' + Badm + ' × 4 = ' + T + ' tokens' : 'T = ' + Badm + ' streams × 4 positions = ' + T + ' tokens in the step') + qTxt, gw, 11.5, 'mono', 600), gx, gyE, { size: 11.5, font: 'mono', weight: 600, color: C.ink });
        V.text(g, fit(g, nrw ? 'distinct: ' + D + '/512 (formula ' + nf(ED, 0) + ') · ' + nf(100 * D / E, 0) + '%' : 'distinct: ' + D + ' of 512 (formula: ' + nf(ED, 0) + ') → ' + nf(100 * D / E, 0) + '% read', gw, 11.5, 'mono'), gx, gyE + 18, { size: 11.5, font: 'mono', color: C.muted });
        // ---------- panel B: curves ----------
        const cx0 = wide ? gx + gw + 44 : pad + 34, cx1 = w - pad - 6;
        const cy0 = wide ? 30 : gyE + 54, cy1 = h - 40;
        const ymax = 340;
        const X = (B) => cx0 + ((B - 1) / (BMAX - 1)) * (cx1 - cx0);
        const Y = (v) => cy1 - (v / ymax) * (cy1 - cy0);
        // queue
        g.fillStyle = V.alpha(C.red, 0.07); g.fillRect(X(SEQS), cy0, cx1 - X(SEQS), cy1 - cy0);
        V.text(g, fit(g, 'above max_num_seqs = 8: queue', cx1 - X(SEQS) - 8, 11, 'mono'), X(SEQS) + 6, Y(120), { size: 11, font: 'mono', color: C.red });
        [0, 100, 200, 300].forEach((v) => { V.line(g, cx0, Y(v), cx1, Y(v), { color: C.grid, lw: 1 }); V.text(g, String(v), cx0 - 6, Y(v) + 4, { size: 11, font: 'mono', color: C.muted, align: 'right' }); });
        [1, 4, 8, 12, 16].forEach((B) => V.text(g, String(B), X(B), cy1 + 16, { size: 11, font: 'mono', color: C.muted, align: 'center' }));
        V.text(g, 'concurrent streams B', (cx0 + cx1) / 2, cy1 + 32, { size: 11, color: C.muted, align: 'center' });
        V.text(g, 'tok/s', cx0 - 6, cy0 - 8, { size: 11, color: C.muted, align: 'right' });
        // asymptote (no slot ceiling)
        const Ainf = S1 / S.phi;
        if (Ainf < ymax) { V.line(g, cx0, Y(Ainf), cx1, Y(Ainf), { color: V.alpha(C.a, 0.4), lw: 1, dash: [5, 4] }); V.text(g, 'limit τ/T₁ ≈ ' + nf(Ainf, 0), cx1 - 4, Y(Ainf) - 6, { size: 11, font: 'mono', color: C.a, align: 'right' }); }
        const curve = (fn, col, b0, b1, o = {}) => { g.beginPath(); for (let B = b0; B <= b1 + 1e-9; B += 0.25) { const v = fn(B); const x = X(B), y = Y(Math.min(v, ymax)); if (B === b0) g.moveTo(x, y); else g.lineTo(x, y); } g.strokeStyle = o.alpha ? V.alpha(col, o.alpha) : col; g.lineWidth = o.lw || 2.4; if (o.dash) g.setLineDash(o.dash); g.stroke(); g.setLineDash([]); };
        // hypothetical, no slot ceiling (dashed, faded)
        curve((B) => modelRaw(B).A, C.a, SEQS, BMAX, { dash: [5, 5], alpha: 0.4, lw: 1.6 });
        curve((B) => modelRaw(B).s, C.b, SEQS, BMAX, { dash: [5, 5], alpha: 0.4, lw: 1.6 });
        // with the ceiling: rises up to 8 and goes flat (slots full)
        curve((B) => model(B).A, C.a, 1, BMAX);
        curve((B) => model(B).s, C.b, 1, BMAX);
        V.text(g, 'aggregate', cx1 - 4, Y(Math.min(model(SEQS).A, ymax)) - 8, { size: 11.5, weight: 600, color: C.a, align: 'right' });
        V.text(g, 'per stream', cx1 - 4, Y(model(SEQS).s) - 8, { size: 11.5, weight: 600, color: C.b, align: 'right' });
        V.text(g, fit(g, 'slots full: flat', cx1 - X(SEQS) - 10, 11, 'mono'), cx1 - 4, Y(model(SEQS).A) + 16, { size: 11, font: 'mono', color: C.a, align: 'right' });
        const yh = Y(Math.min(modelRaw(BMAX).A, ymax));
        V.text(g, fit(g, 'no ceiling (hypothetical)', cx1 - X(SEQS) - 10, 11, 'mono'), cx1 - 4, yh - 8, { size: 11, font: 'mono', color: V.alpha(C.a, 0.75), align: 'right' });
        if (S.meas) {
          const pts = (arr, col) => arr.forEach(([B, v]) => { g.beginPath(); g.arc(X(B), Y(v), 4.5, 0, Math.PI * 2); g.strokeStyle = col; g.lineWidth = 2; g.stroke(); g.fillStyle = C.bg; g.fill(); g.stroke(); });
          pts(MEAS_A, C.a); pts(MEAS_S, C.b);
          V.text(g, '○ measured (Qwen, prose)', cx0 + 6, cy0 + 12, { size: 11, font: 'mono', color: C.muted });
        }
        // current marker
        const m = model(S.B);
        V.line(g, X(S.B), cy0, X(S.B), cy1, { color: V.alpha(C.ink, 0.45), lw: 1.2, dash: [3, 3] });
        V.glow(g, X(S.B), Y(Math.min(m.A, ymax)), 14, C.a, 0.8); V.dot(g, X(S.B), Y(Math.min(m.A, ymax)), 5, C.a);
        V.glow(g, X(S.B), Y(m.s), 14, C.b, 0.8); V.dot(g, X(S.B), Y(m.s), 5, C.b);
      });
    }

    /* ===================== Figure 10: subway map ===================== */
    {
      const f = $('#fig-metro');
      fitH(f, 440, 640, 700);
      const LINES = [
        { id: 'pre', name: 'PREFILL', sub: 'large M · compute', colK: 'd' },
        { id: 'dec', name: 'DECODE', sub: 'small M · bandwidth', colK: 'a' },
        { id: 'kv', name: 'KV', sub: 'capacity', colK: 'b' },
      ];
      const ST = [
        { l: 'pre', r: 'glm', s: 'E3 + 7,168', t: 'Fat experts (E3) and 7,168-token chunks', d: 'With 1,024 tokens per chunk, ~28 rows per expert (~112 FLOP/B, below the 458 ridge). With 7,168, ~199 rows: I ≈ 460–500 counting the activations, right at the ridge (2r/b_w would give ~800). The E3 kernels took the MoE layer from 77–91 to 31 ms and cold prefill from 947 to 1,428–1,587 tok/s.' },
        { l: 'pre', r: 'glm', s: 'KDA BF16', t: 'GLM53_KDA_BF16_LARGE_M=1', d: 'BF16 copy of the KDA in_proj for M > 512: cuBLAS in 7.36 ms versus 27.21 ms for Marlin with M = 7,168. TTFT −11.4% (16k) and −12.3% (100k), for +3.29 GiB per rank.' },
        { l: 'pre', r: 'qwen', s: 'flat 2.96k', t: 'Prefill at ~2.96 thousand tok/s', d: 'Day-0 lane: 2,962 tok/s from 16k to 64k and −8% at 128k; QSA attention keeps the quadratic cost from dominating. An 8,192-token chunk touches every expert: ~34 GB per GPU, ~140 ms of streaming (report estimate: 48 × 512 × 2.77 MB ÷ 2 GPUs, at ~240 GB/s).' },
        { l: 'pre', r: 'ds', s: 'indexer SP', t: 'DSPARK_ENABLE_SP_INDEXER=1', d: 'Each rank scores only a slice of the compressed keys, and the top-k is combined exactly. 128K TTFT −4 to −6%. At 900K, the indexer costs ≈80 TFLOP per 1,024-token chunk, duplicated on both GPUs.' },
        { l: 'pre', r: 'ds', s: 'W4A16', t: '4-bit weights, BF16 math (b12x)', d: 'The checkpoint path dequantizes to BF16 before the multiplication: great for decode (bandwidth), stuck at the BF16 peak in prefill. FP4×FP8 tensor cores would have ~4× the dense peak (report estimate).' },
        { l: 'dec', r: 'ds', s: 'c=1 ≈ 80%', t: 'Single-user decode: ~80% weight streaming', d: '≈16.5–17 GB read per rank per step at 230–250 GB/s = 68–74 ms of streaming, in a step that also pays ~10–15 ms of fixed costs (≈108 collectives, 3–5 ms, plus launches): ~80% is streaming. Measured median of 56 tok/s on Sep 2 (51–67).' },
        { l: 'dec', r: 'glm', s: 'dense FP8', t: 'GLM53_DENSE_FP8', d: 'KDA and dense projections in FP8 weight-only: ~2.6 GB less per step, “~11 ms less per step”. The all mode cut ms per cycle by a further 4.7% versus dense,kda (Sep 26).' },
        { l: 'dec', r: 'glm', s: 'MoE coop.', t: 'Cooperative MoE decode kernel', d: 'Specialized for 1–32 rows (H = 4096, top-8, 4 bits): structured ×1 from 72.03 to 77.29 tok/s (+7.3%); ×2 aggregate +9.3%.' },
        { l: 'dec', r: 'glm', s: 'thin-decode', t: 'GLM53_EXL3_MOE_FAST=1 (thin-decode K4/N256)', d: 'MoE decode kernel for SM121 with 1 register stage and 8 shared-memory stages (stock: 3 and 3): the register spills and the repeated Hadamard on the up input disappear. Same bytes, shorter step: +7.7% (structured), +8.9% (code), +14.6% (JSON) with 1 stream. Opt-in, TP2 only.' },
        { l: 'dec', r: 'glm', s: 'adaptive-k', t: 'GLM53_ADAPTIVE_K=ema', d: 'Verifies only 2, 4 or 7 of the 7 DFlash2 drafts, based on a moving average (EMA) of acceptance. On prose, where acceptance drops, verifying fewer positions lights up fewer experts and shortens the step: +13–21%. The k is the batch minimum, because FULL CUDA graphs require a uniform shape.' },
        { l: 'dec', r: 'qwen', s: 'draft vocab', t: 'Reduced drafter vocabulary', d: 'The lm_head read 4× per step is 26% of the bytes. With 47k tokens in the draft: +15.3% (code) and +15.5% (prose) with 1 stream; +4.0% and +5.6% with 8.' },
        { l: 'dec', r: 'all', s: 'speculation', t: 'MTP · DFlash2 · DSpark', d: 'Several tokens per read of the weights. Qwen MTP=3: 24.5 → 52.1 tok/s (2.13×), τ ≈ 3.2 (2.2 accepted + 1 bonus). GLM DFlash2 k=7: 6.71 accepted per step on structured text (τ ≈ 7.7). DeepSeek DSpark: τ ≈ 4.15 (3.15 accepted + 1 bonus).' },
        { l: 'kv', r: 'glm', s: 'fp8_ds_mla', t: '--kv-cache-dtype fp8 → fp8_ds_mla', d: '656 B per token per MLA layer (512 FP8 + 16 of scales + 128 of zeroed RoPE) + 33 B for the indexer, over 11 layers: 7,579 B per token per rank, plus ~76 MiB of KDA state per request.' },
        { l: 'kv', r: 'glm', s: '11 GiB pool', t: '--kv-cache-memory-bytes 11811160064', d: 'Pins the KV at 11 GiB per GPU: 1,233,779 tokens, 1.45× at 850 thousand, regardless of what the profile run measured.' },
        { l: 'kv', r: 'qwen', s: 'FP8 KV', t: '--kv-cache-dtype fp8', d: '13,312 B per token (6,656 per GPU) instead of 26,624: pool from 2,131,159 → 3,652,200 tokens (+71%), 13.93× at 262k. It required a patch to QSA in the day-0 image.' },
        { l: 'kv', r: 'qwen', s: 'bytes/token', t: 'KV_BYTES_PER_TOKEN=29482 (TP1)', d: 'Bytes per token measured in BF16 on a single Spark. The launcher converts MAX_MODEL_LEN into GiB of KV with this value before choosing the memory fraction.' },
        { l: 'kv', r: 'ds', s: 'nvfp4_ds_mla', t: '--kv-cache-dtype nvfp4_ds_mla', d: 'A 584 B row (448 FP8 + 128 of BF16 RoPE + 8 of scales), byte for byte identical to fp8_ds_mla in this image. Compressed 1/4 and 1/128: ≈3,850 B per token; pool of 2,331,430 tokens.' },
        { l: 'x', r: 'glm', s: 'dispatch by M', t: 'KDA_BF16_LARGE_M_MIN_M = 512', d: 'The number of rows decides the kernel. In the measured steps, decode never went beyond M = 220 and prefill stayed at M ≥ 768: a fixed threshold at 512 separates the phases without syncing with the CPU.' },
      ];
      const S = { filt: 'all', sel: ST.length - 1, auto: true, lastSwitch: 0 };
      const ctl = UI.controls(f);
      const visible = (s) => S.filt === 'all' || s.r === S.filt || s.r === 'all';
      const nextSel = () => { for (let k = 1; k <= ST.length; k++) { const j = (S.sel + k) % ST.length; if (visible(ST[j])) { S.sel = j; return; } } };
      UI.seg(ctl, { label: 'Recipe', options: [{ v: 'all', label: 'all' }, { v: 'glm', label: 'GLM' }, { v: 'qwen', label: 'Qwen' }, { v: 'ds', label: 'DeepSeek' }], value: 'all', onChange: (v) => { S.filt = v; if (!visible(ST[S.sel])) nextSel(); } });
      const bt = UI.button(ctl, '⏸ tour', () => { S.auto = !S.auto; bt.innerHTML = S.auto ? '⏸ tour' : '▶ tour'; });
      const H = V.fig(f, (g, w, h, t) => {
        const C = V.C();
        const wide = w >= 700;
        const pad = 16;
        const cardH = wide ? 124 : 204;
        const mapH = h - cardH - 16;
        const rc = (r) => (r === 'all' ? C.d : C[r]);
        // ---------- line geometry ----------
        const pos = {}; const paths = {};
        const lineStations = (id) => ST.map((s, k) => ({ s, k })).filter((o) => o.s.l === id);
        if (wide) {
          const yP = 56, yD = mapH * 0.52, yK = mapH - 26;
          const fx = pad + 58, fy = (yP + yD) / 2;
          const xs0 = fx + 110, xs1 = w - pad - 60;
          pos.x = [fx, fy];
          const place = (id, y) => { const L = lineStations(id); L.forEach((o, j) => { pos[o.k] = [xs0 + (j + 0.5) * (xs1 - xs0) / L.length, y]; }); };
          place('pre', yP); place('dec', yD); place('kv', yK);
          paths.pre = [[fx, fy], [fx + 40, yP], [xs1 + 30, yP]];
          paths.dec = [[fx, fy], [fx + 40, yD], [xs1 + 30, yD]];
          paths.kv = [[pad + 20, yK], [xs1 + 30, yK]];
        } else {
          const xP = w * 0.18, xD = w * 0.5, xK = w * 0.82;
          const fy = 40, fx = (xP + xD) / 2;
          const ys0 = 96, ys1 = mapH - 10;
          pos.x = [fx, fy];
          const place = (id, x) => { const L = lineStations(id); L.forEach((o, j) => { pos[o.k] = [x, ys0 + (j + 0.5) * (ys1 - ys0) / L.length]; }); };
          place('pre', xP); place('dec', xD); place('kv', xK);
          paths.pre = [[fx, fy], [xP, fy + 36], [xP, ys1 + 6]];
          paths.dec = [[fx, fy], [xD, fy + 36], [xD, ys1 + 6]];
          paths.kv = [[xK, fy + 10], [xK, ys1 + 6]];
        }
        const drawPath = (pts, col) => { g.beginPath(); g.moveTo(pts[0][0], pts[0][1]); for (let k = 1; k < pts.length; k++) g.lineTo(pts[k][0], pts[k][1]); g.strokeStyle = V.alpha(col, 0.8); g.lineWidth = 7; g.lineJoin = 'round'; g.lineCap = 'round'; g.stroke(); g.lineCap = 'butt'; };
        const alongPath = (pts, u) => { const segs = []; let L = 0; for (let k = 1; k < pts.length; k++) { const d = Math.hypot(pts[k][0] - pts[k - 1][0], pts[k][1] - pts[k - 1][1]); segs.push(d); L += d; } let d = u * L; for (let k = 0; k < segs.length; k++) { if (d <= segs[k]) { const q = d / segs[k]; return [lerp(pts[k][0], pts[k + 1][0], q), lerp(pts[k][1], pts[k + 1][1], q)]; } d -= segs[k]; } return pts[pts.length - 1]; };
        LINES.forEach((L, k) => {
          const col = C[L.colK];
          drawPath(paths[L.id], col);
          // train
          const u = (t * 0.06 + k * 0.33) % 1;
          const p = alongPath(paths[L.id], u);
          V.box(g, p[0] - 9, p[1] - 5, 18, 10, { r: 4, fill: C.ink });
          V.dot(g, p[0] - 4, p[1], 1.6, col); V.dot(g, p[0] + 4, p[1], 1.6, col);
          // line name
          const lp = L.id === 'kv' ? paths.kv[0] : (wide ? [paths[L.id][1][0] + 4, paths[L.id][1][1]] : paths[L.id][1]);
          if (wide) { const ly = L.id === 'pre' ? lp[1] + 24 : lp[1] - 12; V.text(g, L.name, lp[0] + 8, ly, { size: 12, font: 'display', weight: 700, color: col }); V.text(g, L.sub, lp[0] + 8 + tw(g, L.name, 12, 'display', 700) + 8, ly, { size: 11, color: C.muted }); }
          else V.text(g, L.name, lp[0], lp[1] - 12 + (L.id === 'kv' ? -2 : 0), { size: 11.5, font: 'display', weight: 700, color: col, align: 'center' });
        });
        // ---------- auto ----------
        if (S.auto && t - S.lastSwitch > 4.5) { S.lastSwitch = t; nextSel(); }
        // ---------- click ----------
        const P = H.pointer;
        let hover = -1;
        const all = ST.map((s, k) => ({ s, k, p: s.l === 'x' ? pos.x : pos[k] }));
        if (P.inside) all.forEach((o) => { if (o.p && Math.hypot(P.x - o.p[0], P.y - o.p[1]) < 18) hover = o.k; });
        if (P.clicked && hover >= 0) { S.sel = hover; S.auto = false; bt.innerHTML = '▶ tour'; }
        const tagText = (txt, x, y, o2) => { const tw2 = tw(g, txt, o2.size, 'mono', o2.weight); const x0 = o2.align === 'center' ? x - tw2 / 2 : x; V.box(g, x0 - 4, y - o2.size + 1, tw2 + 8, o2.size + 5, { r: 5, fill: V.alpha(C.bg, 0.86) }); V.text(g, txt, x, y, Object.assign({ font: 'mono' }, o2)); };
        // ---------- stations ----------
        all.forEach((o) => {
          if (!o.p) return;
          const vis = visible(o.s);
          const sel = o.k === S.sel;
          const x = o.p[0], y = o.p[1];
          const isX = o.s.l === 'x';
          g.save(); g.globalAlpha = vis ? 1 : 0.22;
          if (sel) V.glow(g, x, y, 26 + 3 * Math.sin(t * 4), rc(o.s.r), 0.7);
          g.beginPath(); g.arc(x, y, isX ? 13 : (sel || o.k === hover ? 10 : 8), 0, Math.PI * 2);
          g.fillStyle = C.bg; g.fill(); g.lineWidth = isX ? 4 : 3.2; g.strokeStyle = rc(o.s.r); g.stroke();
          if (isX) V.text(g, 'M', x, y + 0.5, { size: 12, font: 'display', weight: 700, color: C.ink, align: 'center', base: 'middle' });
          else V.dot(g, x, y, 3, rc(o.s.r));
          // label
          const lbl = o.s.s;
          if (wide) {
            const below = o.s.l !== 'pre';
            const decIdx = o.s.l === 'dec' ? lineStations('dec').findIndex((q) => q.k === o.k) : 0;
            const dy = o.s.l === 'dec' && decIdx % 2 === 1 ? 17 : 0;
            if (isX) { V.text(g, 'dispatch', x, y + 30, { size: 11.5, font: 'mono', weight: sel ? 700 : 500, color: C.ink, align: 'center' }); V.text(g, 'by M', x, y + 44, { size: 11.5, font: 'mono', weight: sel ? 700 : 500, color: C.ink, align: 'center' }); }
            else tagText(lbl, x, below ? y + 24 + dy : y - 16, { size: 11.5, weight: sel ? 700 : 500, color: sel ? C.ink : C.text2 || C.ink, align: 'center' });
          } else {
            const yy = isX ? y + 0.5 : y + 22;
            if (isX) V.text(g, lbl, x + 18, yy, { size: 11, font: 'mono', weight: sel ? 700 : 500, color: C.ink, base: 'middle' });
            else tagText(fit(g, lbl, w * 0.3, 11, 'mono'), x, yy, { size: 11, weight: sel ? 700 : 500, color: sel ? C.ink : C.text2 || C.ink, align: 'center' });
          }
          g.restore();
        });
        // ---------- card ----------
        const s = ST[S.sel];
        const cy = h - cardH - 6;
        const col = rc(s.r);
        V.box(g, pad, cy, w - 2 * pad, cardH, { r: 12, fill: V.alpha(col, 0.08), stroke: V.alpha(col, 0.6), lw: 1.4 });
        const tag = s.r === 'all' ? 'all three' : s.r === 'glm' ? 'GLM' : s.r === 'qwen' ? 'Qwen' : 'DeepSeek';
        const lineName = s.l === 'x' ? 'transfer station' : LINES.find((L) => L.id === s.l).name.toLowerCase() + ' line';
        V.text(g, fit(g, tag + ' · ' + lineName, w - 4 * pad, 11, 'mono', 600), pad + 14, cy + 20, { size: 11, font: 'mono', weight: 600, color: col });
        V.text(g, fit(g, s.t, w - 4 * pad, 13.5, 'mono', 700), pad + 14, cy + 40, { size: 13.5, font: 'mono', weight: 700, color: C.ink });
        textBlock(g, s.d, pad + 14, cy + 60, w - 2 * pad - 28, { size: 12.5, color: C.ink, lh: 17 });
      }, { static: false });
    }
  },
});

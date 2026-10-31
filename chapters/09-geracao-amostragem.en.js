LA.chapter({
  id: 'geracao-amostragem',
  part: 2,
  order: 9,
  title: 'Generating text: logits, sampling and reasoning',
  navTitle: 'Generation & sampling',
  subtitle: 'From the final vector to the chosen token: lm_head, softmax, temperature, top-p, Gumbel-max, penalties, determinism and models that think before answering.',
  minutes: 65,
  recipes: ['glm', 'tf', 'qwen', 'ds'],
  tags: ['logits', 'softmax', 'temperature', 'top-p', 'Gumbel-max', 'reasoning', 'determinism', 'generation_config', 'keyed sampling'],
  goals: [
    'Follow one step of the <b>autoregressive loop</b> from the hidden state to the token, with the real <code>lm_head</code> dimensions of each model',
    'Explain and combine <b>temperature, softcap, top-k, top-p, min-p and penalties</b>, and know where the defaults come from (<code>generation_config</code>)',
    'Understand the <b>Gumbel-max trick</b>, seeds and why greedy is not always deterministic (the Qwen patch)',
    'Control <b>reasoning models</b>: effort, <code>max_tokens</code>, the GPU budget and the loop × long tail diagnosis',
    'Understand TensorFold\'s <b>keyed sampling</b>: why the same request gets the same reply and verifying a draft becomes comparing tokens',
  ],
  prereqs: ['bloco-transformer'],
  summary: [
    'Generating is repeating: full forward → <code>lm_head</code> (<code>[V × d]</code>) → logits → softmax → sampler → append → check EOS/stop/<code>max_tokens</code>. All the randomness lives in the sampler.',
    'Temperature divides the logits (T=0 is greedy); the softcap \\(c\\,\\tanh(z/c)\\) and the output multiplier are fixed transformations that must match training. Top-p and min-p adapt to confidence; top-k does not.',
    'Defaults come from <code>generation_config.json</code> (<code>auto</code>) or from vLLM (<code>vllm</code>); the request always wins. GLM and Qwen inherit T=1/top_p 0.95; DeepSeek lets the client decide. The warmup has to compile the k-only, p-only and k+p variants of the sampler.',
    'Gumbel-max: \\(\\arg\\max(z/T+g)\\) samples exactly from the softmax with a single reduction, without normalizing or synchronizing with the CPU. It is the sampler of Model Runner V2 (which DS and the GLM vLLM lane run, with counter-based noise derived from seed and position) and of the DFlash2 selector; Model Runner V1 uses the equivalent form \\(\\arg\\max\\, p/E\\).',
    'Float sums are not associative: greedy is only bit-identical with a fixed reduction order (Qwen: sorted QSA top-k + non-fused MoE finalize, spread 0.25/6.07 → 0/0 nats, cost ~4%).',
    'Reasoning effort is text in the prompt (values and mappings change by family and break the prefix cache); <code>max_tokens</code> counts thinking + answer; the DeepSeek budget forces <code>&lt;/think&gt;</code> with GPU kernels, without the ~4.7× cliff of the CPU version.',
    'Speculation uses rejection sampling: acceptance \\(=\\sum\\min(p,q)\\), highest in greedy, plus a bonus token when every draft passes. The greedy benchmarks of GLM and Qwen overestimate served throughput at T≈1; the DeepSeek ones already measure at T=0.6. In the GLM TensorFold lane, the draw is keyed by (seed or the prompt\'s SHA-256, position, token) and verification is by equality: the reply with drafts = serial and concurrent = alone, at the price of an API with no logprobs or penalties.',
  ],
  quiz: [
    { q: 'Logits \\([3, 1, 0]\\). What happens to the distribution when the temperature drops from 1 to 0.5?', o: ['The tokens swap places in the ranking', 'It gets more peaked: the favorite gains mass, and the order is preserved', 'It gets flatter', 'Nothing: temperature only acts after top-p'], a: 1, e: 'Dividing by \\(T&lt;1\\) widens the differences between logits, and the softmax concentrates mass on the largest one. Since the division is monotonic, the order never changes. And temperature changes the distribution on its own, with or without truncation filters after it.' },
    { q: 'In a context where the favorite has 87% probability, which filter keeps <i>fewer</i> tokens: <code>top_k=20</code> or <code>top_p=0.9</code>?', o: ['top_k=20, always', 'Both keep exactly 20', 'It depends only on the temperature', 'top_p=0.9: the nucleus that sums to 90% has only 2 or 3 tokens'], a: 3, e: 'Top-k has a fixed size. Top-p adapts to confidence: when the favorite already accounts for 87%, one or two more tokens are enough to reach 90%. In uncertain contexts the opposite happens, and top-p keeps many tokens.' },
    { q: 'Why did the DeepSeek recipe reimplement <code>thinking_token_budget</code> with two Triton kernels on the GPU?', o: ['Because the previous version copied tokens to the CPU and scanned them at every step, causing a ~4.7× decode cliff', 'To raise speculative acceptance', 'Because Triton samples faster than FlashInfer', 'So that it could inject a default budget at the server'], a: 0, e: 'The old hook (#31/#34) did an O(n) scan in Python multiplied by the MTP rows, with a GPU→CPU synchronization per step. The v2 kernels keep the state on the GPU and only rewrite the logit row at the exact limit step. And, on purpose, the server injects no default.' },
    { q: 'In Qwen at TP2+EP, five identical greedy runs gave five different texts. What is the documented cause?', o: ['A different random seed on each run', 'A bug in the chat template', 'Variable order in floating-point reductions: QSA top-k indices in a different order and the fused MoE finalize', 'The default temperature of 1.0 from generation_config'], a: 2, e: 'Greedy does not draw, so the seed is irrelevant. Float addition is not associative, and two kernels summed in a variable order. Sorting the top-k (<code>VLLM_QSA_DET_TOPK=1</code>) and using the non-fused finalize (<code>VLLM_MOE_DET_FINALIZE=1</code>) brought the spread down to zero (0.25/6.07 → 0/0 nats) at a cost of ~4%.' },
    { q: 'A reasoning response comes back with <code>finish_reason: length</code> and empty <code>content</code>. <code>loop_detector.py</code> shows 8-gram novelty above 2% all the way to the end of the trace. What should you do?', o: ['Nothing helps: it is a loop', 'It is a long tail: raising <code>max_tokens</code> (or granting a reasoning budget) fixes it', 'Turn on <code>repetition_penalty=2</code>', 'Switch to greedy'], a: 1, e: 'A loop requires novelty to collapse below the threshold for 3 consecutive windows, with no recovery until the end. If novelty stays high, the model keeps saying new things, just too many of them. That is a long tail, and more budget fixes it.' },
    { q: 'Why does a greedy benchmark overestimate the throughput of a speculative-decoding server that serves at T=1?', o: ['Because at T=1 the lm_head is slower', 'Because at T=1 the KV cache doubles', 'It does not: rejection sampling does not depend on T', 'Because per-position acceptance is \\(\\sum\\min(p,q)\\): highest when p and q are one-hot with the same favorite, and lower when the distributions spread out'], a: 3, e: 'Rejection sampling preserves the target distribution, but the acceptance rate depends on the overlap between target and draft at the served temperature. The Qwen audit notes exactly this: greedy MTP numbers sit above real chat throughput at T=1. That is why the DeepSeek decode bench measures directly at T=0.6.' },
    { q: 'In the GLM TensorFold lane, the same request, with no <code>seed</code> and at T=1, runs alone and then together with three others and with drafts. What happens to the text?', o: ['It changes, because without a <code>seed</code> the noise is drawn anew for every request', 'It stays identical, bit for bit: the key comes from the prompt\'s SHA-256, the noise is a function of (key, position, token) and the kernels are batch-invariant', 'It stays identical only at T=0', 'It stays equal in distribution, but not in sequence, as with rejection sampling'], a: 1, e: 'Without a <code>seed</code>, TensorFold derives the key from the prompt, and each token\'s Gumbel push is looked up by (key, absolute position, id), without depending on neighbors or on draft rows. Since each row\'s logits do not depend on the other rows either, equality verification returns the serial sequence. The recipe publishes 11 of 11 identical replies in staggered runs and 11 of 11 in a burst. Equality only in distribution is vLLM\'s guarantee.' },
  ],
  mount(root, LA) {
    const V = LA.viz, UI = LA.ui;
    const $ = (s) => root.querySelector(s);

    /* ============================ helpers ============================ */
    const FC = {};
    const setFont = (g, sz, f = 'sans', wt = 400) => { const k = sz + '|' + f + '|' + wt; g.font = FC[k] || (FC[k] = V.font(sz, f, wt)); };
    const measure = (g, s, sz, f, wt) => { setFont(g, sz, f, wt); return g.measureText(String(s)).width; };
    const fmtInt = (n) => Math.round(n).toLocaleString('en-US');
    const pct = (v, d) => (v * 100).toFixed(d == null ? (v < 0.1 ? 1 : 0) : d) + '%';
    const num = (v, d = 2) => (+v).toFixed(d);
    function fitH(fig, hd, hm, bp = 600) {
      const st = fig.querySelector('.la-fig-stage');
      const apply = () => { const w = fig.clientWidth || 800; const want = (w < bp ? hm : hd) + 'px'; if (st.style.height !== want) st.style.height = want; };
      apply();
      window.addEventListener('resize', apply);
      LA.onCleanup(() => window.removeEventListener('resize', apply));
    }
    function pill(g, s, x, y, o = {}) {
      const C = V.C();
      const sz = o.size || 12, f = o.font || 'mono', wt = o.weight || 500, px = o.px == null ? 7 : o.px, hh = o.h || sz + 10;
      const w = Math.max(o.minW || 0, measure(g, s, sz, f, wt) + px * 2);
      if (o.dry) return w;
      if (o.glow) V.glow(g, x + w / 2, y + hh / 2, Math.max(w, hh) * 0.75, o.glow, 0.45);
      if (o.dash) g.setLineDash(o.dash);
      V.box(g, x, y, w, hh, { r: o.r == null ? 7 : o.r, fill: o.fill || V.alpha(C.muted, 0.1), stroke: o.stroke, lw: o.lw || 1.2 });
      g.setLineDash([]);
      V.text(g, noLig(s), x + w / 2, y + hh / 2 + 0.5, { size: sz, font: f, weight: wt, color: o.color || C.ink, align: 'center', base: 'middle', alpha: o.alpha });
      return w;
    }
    function wrapLines(g, s, maxW, sz, f = 'sans', wt = 400) {
      const words = String(s).split(' ');
      const out = []; let line = '';
      words.forEach((wd) => { const tt = line ? line + ' ' + wd : wd; if (measure(g, tt, sz, f, wt) > maxW && line) { out.push(line); line = wd; } else line = tt; });
      if (line) out.push(line);
      return out;
    }
    function textBlock(g, s, x, y, maxW, o = {}) {
      const sz = o.size || 12.5, lh = o.lh || Math.round(sz * 1.42);
      const L = wrapLines(g, s, maxW, sz, o.font || 'sans', o.weight || 400);
      L.forEach((l, i) => V.text(g, l, x, y + i * lh, Object.assign({ size: sz }, o)));
      return y + L.length * lh;
    }
    function clip(g, s, maxW, sz, f = 'mono', wt = 400) {
      s = String(s);
      if (measure(g, s, sz, f, wt) <= maxW) return s;
      while (s.length > 1 && measure(g, s + '…', sz, f, wt) > maxW) s = s.slice(0, -1);
      return s + '…';
    }
    const softmax = (z, T = 1) => {
      if (T <= 0) { let bi = 0; z.forEach((v, i) => { if (v > z[bi]) bi = i; }); return z.map((_, i) => (i === bi ? 1 : 0)); }
      const m = Math.max(...z); const e = z.map((v) => Math.exp((v - m) / T)); const s = e.reduce((a, b) => a + b, 0); return e.map((v) => v / s);
    };
    const entropy = (p) => p.reduce((s, v) => (v > 0 ? s - v * Math.log(v) : s), 0);
    const argmax = (a) => { let bi = 0; a.forEach((v, i) => { if (v > a[bi]) bi = i; }); return bi; };
    const vis = (s) => String(s).replace(/^ /, '');
    const noLig = (s) => String(s).replace(/<\|/g, '<\u200B|').replace(/\|>/g, '|\u200B>');
    function hatch(g, x, y, w, h, col) {
      g.save(); g.beginPath(); g.rect(x, y, w, h); g.clip();
      g.strokeStyle = col; g.lineWidth = 1;
      for (let k = -h; k < w; k += 6) { g.beginPath(); g.moveTo(x + k, y + h); g.lineTo(x + k + h, y); g.stroke(); }
      g.restore();
    }
    function star(g, x, y, r, col) {
      g.beginPath();
      for (let k = 0; k < 10; k++) { const a = -Math.PI / 2 + (k * Math.PI) / 5, rr = k % 2 ? r * 0.45 : r; g.lineTo(x + Math.cos(a) * rr, y + Math.sin(a) * rr); }
      g.closePath(); g.fillStyle = col; g.fill();
    }
    const onInk = () => (V.C().dark ? '#00210F' : '#ffffff');
    function scissors(g, x, y, s, col) {
      g.strokeStyle = col; g.lineWidth = 1.6;
      g.beginPath(); g.arc(x - s * 0.55, y - s * 0.32, s * 0.28, 0, Math.PI * 2); g.stroke();
      g.beginPath(); g.arc(x - s * 0.55, y + s * 0.32, s * 0.28, 0, Math.PI * 2); g.stroke();
      g.beginPath(); g.moveTo(x - s * 0.32, y - s * 0.2); g.lineTo(x + s * 0.7, y + s * 0.28); g.moveTo(x - s * 0.32, y + s * 0.2); g.lineTo(x + s * 0.7, y - s * 0.28); g.stroke();
    }

    /* ================= Figure 1: the autoregressive loop ================= */
    const f1 = $('#fig-laco');
    fitH(f1, 390, 440);
    const M1 = {
      glm: { name: 'GLM-5.3-Flash', L: 45, d: 4096, V: 154880, eos: '<|user|>', col: 'glm' },
      qwen: { name: 'Qwen3.8-Flash-Next', L: 48, d: 2560, V: 248320, eos: '<|im_end|>', col: 'qwen' },
      ds: { name: 'DeepSeek-V4-Flash', L: 43, d: 4096, V: 129280, eos: '<｜end▁of▁sentence｜>', col: 'ds' },
    };
    const PROMPT1 = ['The', ' cat', ' climbed', ' the'];
    const GEN1 = [' roof', ' and', ' meowed', '.', 'EOS'];
    const PICK1 = [0, 0, 1, 0, 0];
    const CAND1 = [
      [[' roof', 6.1], [' wall', 5.4], [' sofa', 4.9], [' car', 4.2], [' fence', 3.1]],
      [[' and', 5.8], ['.', 5.5], [',', 4.6], [' again', 3.9], [' fast', 3.0]],
      [[' slept', 5.9], [' meowed', 5.6], [' stayed', 4.8], [' looked', 4.4], [' fell', 3.2]],
      [['.', 6.4], [' there', 4.9], [' at', 4.1], [',', 3.8], [' until', 3.0]],
      [['EOS', 6.8], [' Then', 4.2], ['\\n', 4.0], [' It', 2.9], [' And', 2.1]],
    ];
    const S1 = { model: 'glm', gen: 0, prev: 0 };
    const c1 = UI.controls(f1);
    UI.seg(c1, { label: 'Model', options: [{ v: 'glm', label: 'GLM' }, { v: 'qwen', label: 'Qwen' }, { v: 'ds', label: 'DeepSeek' }], value: 'glm', onChange: (v) => { S1.model = v; } });
    const st1 = UI.stepper(f1, [
      { t: 'Context', d: 'The prompt tokens (lilac) and the ones already generated (mint) form the input. The model looks at all of them, but only the vector at the <b>last position</b> decides the next token.' },
      { t: 'Forward through the layers', d: 'The vector at the last position passes through every layer of the trunk (45 in GLM, 48 in Qwen, 43 in DeepSeek). This is the expensive step: the active weights of every layer are read from memory.' },
      { t: 'lm_head → logits', d: 'The final RMSNorm and the <code>lm_head [V × d]</code> turn the vector \\(h\\) (d numbers) into V logits, one raw score per vocabulary token.' },
      { t: 'Softmax', d: 'The scores become positive probabilities that add up to 1. We show the top 5; the rest of the vocabulary shares whatever is left.' },
      { t: 'Sample', d: 'The sampler draws a token. Temperature, top-p and friends come in the next sections. Notice that the favorite does not always win.' },
      { t: 'Append and check for a stop', d: 'The chosen token goes to the end of the sequence. If it is an end token (EOS), generation stops with <code>finish_reason: stop</code>. Otherwise, the loop starts over with one more token.' },
    ], {
      autoplay: 3600,
      onStep: (i) => {
        if (S1.prev === 5 && i === 0) S1.gen = (S1.gen + 1) % GEN1.length;
        S1.prev = i;
      },
    });
    V.fig(f1, (g, w, h, t) => {
      const C = V.C();
      const m = M1[S1.model];
      const mc = C[m.col] || C.a;
      const narrow = w < 600;
      const i = st1.i;
      const e = V.clamp(st1.elapsed() / 2.2, 0, 1);
      const gi = S1.gen;
      const cands = CAND1[gi];
      const pick = PICK1[gi];
      const tokLabel = (s) => (s === 'EOS' ? m.eos : vis(s));
      // header
      V.text(g, clip(g, narrow ? `${m.name} · ${m.L} layers · d = ${m.d}` : `${m.name} · ${m.L} layers · d = ${m.d} · V = ${fmtInt(m.V)}`, w - 32, 11.5, 'mono'), 16, 20, { size: 11.5, font: 'mono', color: mc });
      // sequence strip
      const seq = PROMPT1.map((s) => ({ s, k: 'p' })).concat(GEN1.slice(0, gi).map((s) => ({ s, k: 'g' })));
      let x = 16, y = 32;
      const ph = 24;
      const pos = [];
      seq.forEach((tk, k) => {
        const lab = tokLabel(tk.s);
        const pw = pill(g, lab, 0, 0, { dry: true, size: 12 });
        if (x + pw > w - 16) { x = 16; y += ph + 6; }
        const on0 = i === 0;
        const isLast = k === seq.length - 1;
        const col = tk.k === 'p' ? C.b : C.a;
        pill(g, lab, x, y, { size: 12, fill: V.alpha(col, on0 ? 0.22 + 0.1 * Math.sin(performance.now() / 300 + k) : 0.13), stroke: (i === 1 && isLast) ? mc : V.alpha(col, 0.7), lw: (i === 1 && isLast) ? 2.2 : 1.2 });
        pos.push([x, y, pw]);
        x += pw + 6;
      });
      // slot for the next token
      const nextLab = i === 5 && e > 0.75 ? tokLabel(GEN1[gi]) : '?';
      const npw = pill(g, nextLab, 0, 0, { dry: true, size: 12 });
      if (x + npw > w - 16) { x = 16; y += ph + 6; }
      const slot = [x, y, npw];
      if (nextLab === '?') pill(g, '?', x, y, { size: 12, dash: [3, 3], stroke: V.alpha(C.muted, 0.7), fill: 'transparent', color: C.muted });
      else pill(g, nextLab, x, y, { size: 12, fill: V.alpha(C.a, 0.3), stroke: C.a, glow: C.a });
      // regions
      const top = y + ph + (narrow ? 20 : 26);
      const mx = 16, mw = narrow ? w - 32 : Math.min(360, w * 0.4), mh = narrow ? 104 : h - top - 44;
      const my = top;
      // machine (trunk)
      V.box(g, mx, my, mw, mh, { r: 14, fill: V.alpha(mc, 0.06), stroke: V.alpha(mc, 0.45), lw: 1.4 });
      V.text(g, `Transformer · ${m.L} layers`, mx + 12, my + 18, { size: 12, weight: 600, color: C.ink });
      const nL = m.L, slotW = (mw - 24) / nL;
      const sy = my + 30, sh = mh - 50;
      const dotX = mx + 12 + e * (mw - 24);
      for (let k = 0; k < nL; k++) {
        const sx = mx + 12 + k * slotW;
        const lit = i === 1 ? sx < dotX : i > 1;
        V.box(g, sx + 0.5, sy, Math.max(1.5, slotW - 1.8), sh, { r: 2, fill: lit ? V.alpha(mc, 0.55) : V.alpha(C.muted, 0.16) });
      }
      V.text(g, 'each slice = one layer (attention + FFN/MoE)', mx + 12, my + mh - 8, { size: 10.5, font: 'mono', color: C.muted });
      if (i === 0) {
        // tokens flowing down into the machine (ids → embeddings)
        pos.forEach(([px0, py0, pw0], k) => {
          const u = (t * 0.55 + k * 0.23) % 1;
          const sx = px0 + pw0 / 2, sy0 = py0 + ph + 2;
          const tx = V.clamp(sx, mx + 16, mx + mw - 16), ty = sy + 6;
          const q = V.ease.inOut(u);
          V.dot(g, V.lerp(sx, tx, q), V.lerp(sy0, ty, q), 3, V.alpha(k < PROMPT1.length ? C.b : C.a, 1 - 0.6 * u));
        });
        V.text(g, 'ids → embeddings', mx + mw - 12, my + 18, { size: 11.5, font: 'mono', align: 'right', color: C.muted });
      }
      if (i === 1) {
        V.glow(g, dotX, sy + sh / 2, 18, mc);
        V.dot(g, dotX, sy + sh / 2, 4, C.ink);
        const layer = Math.min(nL, Math.max(1, Math.ceil(e * nL)));
        V.text(g, `layer ${layer} / ${nL}`, mx + mw - 12, my + 18, { size: 11.5, font: 'mono', align: 'right', color: mc });
      } else if (i > 1) V.text(g, `h: ${m.d} numbers`, mx + mw - 12, my + 18, { size: 11.5, font: 'mono', align: 'right', color: C.muted });
      // lm_head + bars
      let bx, by, bw, bh, lx;
      if (narrow) { lx = 16; by = my + mh + 24; bx = 104; bw = w - bx - 16; bh = h - by - 14; }
      else { lx = mx + mw + 24; bx = lx + 110; by = my; bw = w - bx - 16; bh = mh; }
      // lm_head icon
      const iw = narrow ? 74 : 86, ih = Math.min(narrow ? bh - 26 : bh - 40, 150);
      const iy = by + (narrow ? 18 : 22);
      const lmOn = i >= 2;
      V.box(g, lx, iy, iw, ih, { r: 8, fill: V.alpha(C.d, lmOn ? 0.18 : 0.06), stroke: V.alpha(C.d, lmOn ? 0.9 : 0.35) });
      for (let r = 1; r < 9; r++) V.line(g, lx + 4, iy + (ih * r) / 9, lx + iw - 4, iy + (ih * r) / 9, { color: V.alpha(C.d, 0.25), lw: 1 });
      V.text(g, 'lm_head', lx + iw / 2, iy + ih / 2 - 6, { size: 12, weight: 600, align: 'center', color: C.ink });
      V.text(g, `[${fmtInt(m.V)}`, lx + iw / 2, iy + ih / 2 + 10, { size: 10.5, font: 'mono', align: 'center', color: C.muted });
      V.text(g, `× ${m.d}]`, lx + iw / 2, iy + ih / 2 + 23, { size: 10.5, font: 'mono', align: 'center', color: C.muted });
      if (!narrow) V.arrow(g, mx + mw + 4, my + mh / 2, lx - 4, iy + ih / 2, { color: V.alpha(C.muted, lmOn ? 0.9 : 0.35), lw: 1.6 });
      // candidate bars
      const z = cands.map((c) => c[1]);
      const pz = softmax(z).map((v) => v * 0.96);
      const zmin = Math.min(...z) - 1, zmax = Math.max(...z);
      const head = i <= 2 ? 'logits z (top 5)' : 'probabilities (top 5 of V)';
      V.text(g, head, bx, by + 12, { size: 11, font: 'mono', color: C.muted });
      const rowH = Math.min(28, (bh - 38) / 5.6);
      const labW = narrow ? 70 : 84;
      const barX = bx + labW, barW = Math.max(40, bw - labW - 50);
      let hiIdx = -1;
      if (i === 4) hiIdx = Math.floor(V.ease.out(e) * (15 + pick)) % 5;
      if (i === 5) hiIdx = pick;
      cands.forEach(([s, zv], k) => {
        const ry = by + 22 + k * rowH;
        const lab = clip(g, tokLabel(s), labW - 8, 11.5, 'mono');
        const isHi = k === hiIdx;
        V.text(g, noLig(lab), barX - 8, ry + rowH / 2, { size: 11.5, font: 'mono', align: 'right', base: 'middle', color: isHi ? C.c : C.ink });
        if (i < 2) { V.box(g, barX, ry + 4, barW * 0.2, rowH - 8, { r: 4, fill: V.alpha(C.muted, 0.12) }); return; }
        const lz = (zv - zmin) / (zmax - zmin);
        const lp = pz[k] / Math.max(...pz);
        const u = i === 2 ? lz : i === 3 ? V.lerp(lz, lp, V.ease.inOut(e)) : lp;
        const col = isHi ? C.c : i === 2 ? C.d : C.a;
        V.box(g, barX, ry + 4, Math.max(3, u * barW), rowH - 8, { r: 4, fill: V.alpha(col, isHi ? 0.85 : 0.6) });
        const val = i === 2 ? num(zv, 1) : pct(pz[k]);
        V.text(g, val, barX + Math.max(3, u * barW) + 6, ry + rowH / 2, { size: 11, font: 'mono', base: 'middle', color: C.ink });
        if (isHi && i === 4) V.glow(g, barX + u * barW, ry + rowH / 2, 14, C.c, 0.7);
      });
      const footY = by + 22 + 5 * rowH + 14;
      if (i >= 3) V.text(g, clip(g, `+ ${fmtInt(m.V - 5)} other tokens · 4%`, bw, 11, 'mono'), bx, footY, { size: 11, font: 'mono', color: C.muted });
      else if (i === 2) V.text(g, clip(g, `z_v = ⟨w_v, h⟩ for all ${fmtInt(m.V)} tokens`, bw, 11, 'mono'), bx, footY, { size: 11, font: 'mono', color: C.muted });
      // step 6: the token flies back to the line
      if (i === 5) {
        const k = pick;
        const sx0 = barX - 30, sy0 = by + 22 + k * rowH;
        const u = V.ease.inOut(V.clamp(e / 0.75, 0, 1));
        const px = V.lerp(sx0, slot[0], u), py = V.lerp(sy0, slot[1], u) - Math.sin(u * Math.PI) * 40;
        if (e < 0.75) pill(g, tokLabel(GEN1[gi]), px, py, { size: 12, fill: V.alpha(C.c, 0.35), stroke: C.c, glow: C.c });
        const isE = GEN1[gi] === 'EOS';
        const msg = isE ? 'EOS → end of generation (finish_reason: stop)' : 'not EOS → the loop restarts with t + 1';
        V.text(g, clip(g, msg, w - 32, 12, 'sans', 600), narrow ? 16 : bx, h - 10, { size: 12, weight: 600, color: isE ? C.c : C.a });
      }
      if (i === 4) V.text(g, 'draw: the wheel spins past every token and stops on one', narrow ? 16 : bx, h - 10, { size: 12, color: C.muted });
    });

    /* ================= Figure 2: lm_head (matvec + softmax) ================= */
    const f2 = $('#fig-lmhead');
    fitH(f2, 420, 500);
    const TOK2 = [' blue', ' cloudy', ' 4', ' five', ' king', ' dragon', '.', ' and'];
    const W2 = [[2.0, 0, 0, 0, 0, 0.3], [1.6, 0, 0, 0, 0.2, -0.4], [0, 2.2, 0, 0, 0, 0.2], [0, 1.5, 0, 0, 0, -0.5], [0, 0, 2.0, 0, 0, 0.4], [0, 0, 1.8, 0, 0, -0.3], [0, 0.3, 0, 1.5, 0.2, 0], [0.3, 0, 0.3, 0.2, 1.4, 0]];
    const CTX2 = {
      ceu: { label: 'The sky is', h: [2.2, 0, 0.1, 0.4, 0.5, 0.6] },
      soma: { label: '2 + 2 =', h: [0, 2.4, 0, 0.5, 0.1, 0.3] },
      conto: { label: 'Once upon a time, a', h: [0, 0, 2.0, 0.1, 0.6, -0.2] },
    };
    const S2 = { ctx: 'ceu', tp: false, t0: 0 };
    const c2 = UI.controls(f2);
    UI.seg(c2, { label: 'Context', options: [{ v: 'ceu', label: '“The sky is”' }, { v: 'soma', label: '“2 + 2 =”' }, { v: 'conto', label: '“Once upon a time, a”' }], value: 'ceu', onChange: (v) => { S2.ctx = v; S2.t0 = H2.t; } });
    UI.toggle(c2, { label: 'vocab-parallel (TP = 2)', value: false, onChange: (v) => { S2.tp = v; } });
    const r2a = UI.readout(c2, 'favorite', '—');
    const r2b = UI.readout(c2, 'entropy H', '—');
    const H2 = V.fig(f2, (g, w, h, t) => {
      const C = V.C();
      const narrow = w < 600;
      const hv = CTX2[S2.ctx].h;
      const z = W2.map((row) => row.reduce((s, a, k) => s + a * hv[k], 0));
      const p = softmax(z);
      const top = argmax(p);
      r2a.set(`${vis(TOK2[top])} · ${pct(p[top])}`);
      r2b.set(`${num(entropy(p))} nats`);
      const tt = t - S2.t0;
      const per = 0.75, nR = 8, cyc = nR * per + 3.2;
      const ph = tt % cyc;
      const row = ph < nR * per ? Math.floor(ph / per) : 8;
      const barsE = ph < nR * per ? 0 : V.clamp((ph - nR * per) / 1.1, 0, 1);
      const cell = narrow ? 19 : 32, gap = narrow ? 2 : 3;
      const labW = narrow ? 62 : 84;
      const x0 = 16 + labW, y0 = 52;
      V.text(g, `context: “${CTX2[S2.ctx].label}” → h (d = 6)`, 16, 20, { size: 12, font: 'mono', color: C.ink });
      V.text(g, 'W_U  [V = 8 × d = 6]', x0, y0 - 12, { size: 11, font: 'mono', color: C.muted });
      const gm = V.matrix(g, { x: x0, y: y0, data: W2, cell, gap, cmap: 'div', min: -2.2, max: 2.2, values: !narrow, digits: 1, hiRow: row < 8 ? row : null, hiColor: C.c });
      TOK2.forEach((s, k) => V.text(g, vis(s), x0 - 8, y0 + k * (cell + gap) + cell / 2, { size: narrow ? 11 : 12, font: 'mono', align: 'right', base: 'middle', color: k === row ? C.c : C.ink }));
      if (S2.tp) {
        const hh = 4 * (cell + gap) - gap;
        [[0, C.a, 'GPU 0'], [4, C.b, 'GPU 1']].forEach(([r0, col, lab]) => {
          g.setLineDash([5, 4]);
          V.box(g, x0 - labW + 2, y0 + r0 * (cell + gap) - 3, labW - 6 + gm.w + 6, hh + 6, { r: 8, stroke: col, lw: 1.6 });
          g.setLineDash([]);
          g.save(); g.translate(x0 - labW + 13, y0 + r0 * (cell + gap) + hh / 2); g.rotate(-Math.PI / 2);
          V.text(g, lab, 0, 0, { size: 10.5, font: 'mono', align: 'center', base: 'middle', color: col });
          g.restore();
        });
      }
      // vector h
      const hx = x0 + gm.w + (narrow ? 12 : 26);
      const hy = y0 + (8 * (cell + gap) - 6 * (cell + gap)) / 2;
      V.text(g, '×', hx - (narrow ? 7 : 14), y0 + (8 * (cell + gap)) / 2, { size: 16, align: 'center', base: 'middle', color: C.muted });
      V.text(g, 'h', hx + cell / 2, hy - 12, { size: 11, font: 'mono', align: 'center', color: C.muted });
      V.matrix(g, { x: hx, y: hy, data: hv.map((v) => [v]), cell, gap, cmap: 'div', min: -2.4, max: 2.4, values: !narrow, digits: 1, hiCol: row < 8 ? 0 : null, hiColor: C.c });
      // z column
      const zx = hx + cell + (narrow ? 14 : 30);
      const zw = narrow ? 34 : 50;
      V.text(g, '=', zx - (narrow ? 8 : 15), y0 + (8 * (cell + gap)) / 2, { size: 16, align: 'center', base: 'middle', color: C.muted });
      V.text(g, 'z', zx + zw / 2, y0 - 12, { size: 11, font: 'mono', align: 'center', color: C.muted });
      z.forEach((v, k) => {
        const ry = y0 + k * (cell + gap);
        const done = k < row || row === 8;
        const gcol = S2.tp ? (k < 4 ? C.a : C.b) : C.d;
        V.box(g, zx, ry, zw, cell, { r: 5, fill: done ? V.alpha(gcol, 0.22) : V.alpha(C.muted, 0.08), stroke: k === row ? C.c : V.alpha(gcol, done ? 0.7 : 0.25), lw: k === row ? 2.2 : 1 });
        V.text(g, done || k === row ? num(v, 1) : '?', zx + zw / 2, ry + cell / 2 + 0.5, { size: narrow ? 11 : 12, font: 'mono', align: 'center', base: 'middle', color: done ? C.ink : C.muted });
      });
      // softmax bars
      let bx, by, bw, rowH;
      if (narrow) { bx = 16 + labW; by = y0 + 8 * (cell + gap) + 64; bw = w - bx - 50; rowH = Math.min(22, (h - by - 50) / 8); }
      else { bx = zx + zw + 84; by = y0; bw = w - bx - 60; rowH = cell + gap; }
      if (!narrow) {
        V.arrow(g, zx + zw + 8, y0 + 4 * (cell + gap), bx - 10, y0 + 4 * (cell + gap), { color: V.alpha(C.muted, 0.8) });
        V.text(g, S2.tp ? 'all-gather' : 'softmax', (zx + zw + bx) / 2, y0 + 4 * (cell + gap) - 10, { size: 11, font: 'mono', align: 'center', color: S2.tp ? C.b : C.muted });
        if (S2.tp) V.text(g, '+ softmax', (zx + zw + bx) / 2, y0 + 4 * (cell + gap) + 18, { size: 11, font: 'mono', align: 'center', color: C.muted });
      } else {
        V.text(g, S2.tp ? '↓ all-gather + softmax' : '↓ softmax', bx, by - 14, { size: 11, font: 'mono', color: S2.tp ? C.b : C.muted });
        TOK2.forEach((s, k) => V.text(g, vis(s), bx - 8, by + k * rowH + rowH / 2, { size: 11, font: 'mono', align: 'right', base: 'middle', color: C.ink }));
      }
      const pm = Math.max(...p);
      p.forEach((v, k) => {
        const ry = by + k * rowH;
        const len = Math.max(2, (v / pm) * bw * V.ease.out(barsE));
        if (barsE <= 0) { V.box(g, bx, ry + 3, Math.min(bw, 60), rowH - 6, { r: 4, fill: V.alpha(C.muted, 0.08) }); return; }
        V.box(g, bx, ry + 3, len, rowH - 6, { r: 4, fill: V.alpha(k === top ? C.c : C.a, 0.7) });
        if (barsE > 0.3) V.text(g, pct(v), bx + len + 6, ry + rowH / 2, { size: 11, font: 'mono', base: 'middle', color: C.ink });
      });
      // arithmetic for the current row
      const fy = h - 14;
      if (row < 8) {
        const terms = W2[row].map((a, k) => [a, hv[k]]).filter(([a, b]) => a !== 0 && b !== 0);
        const s = terms.map(([a, b]) => `${num(a, 1)}·${num(b, 1)}`).join(' + ') || '0';
        V.text(g, clip(g, `z(“${vis(TOK2[row])}”) = ⟨w, h⟩ = ${s} = ${num(z[row], 2)}`, w - 32, narrow ? 11 : 12.5, 'mono'), 16, fy, { size: narrow ? 11 : 12.5, font: 'mono', color: C.c });
      } else V.text(g, clip(g, S2.tp ? 'each GPU computed half of the rows; the all-gather joins the 8 logits before the draw' : 'p = softmax(z): scores become probabilities', w - 32, 12, 'sans'), 16, fy, { size: 12, color: C.muted });
    });

    /* ================= Figure 3: temperature + softcap ================= */
    const f3 = $('#fig-temperatura');
    fitH(f3, 400, 490);
    const TOK3 = [' hot', ' cold', ' great', ' strong', ' bitter', ' ready', ' delicious', ' weak', ' sweet', ' burnt'];
    const Z3 = [5.2, 4.4, 4.1, 3.6, 3.2, 2.7, 2.4, 1.8, 1.3, 0.4];
    const S3 = { T: 1, cap: 0, cur: null };
    const c3 = UI.controls(f3);
    UI.slider(c3, { label: 'Temperature', min: 0, max: 2, step: 0.05, value: 1, fmt: (v) => (v === 0 ? '0 greedy' : num(v, 2)), onInput: (v) => { S3.T = v; } });
    UI.seg(c3, { label: 'Softcap', options: [{ v: 0, label: 'off' }, { v: 6, label: 'c = 6' }, { v: 3, label: 'c = 3' }, { v: 1.5, label: 'c = 1.5' }], value: 0, onChange: (v) => { S3.cap = v; } });
    const r3a = UI.readout(c3, 'H', '—');
    const r3b = UI.readout(c3, 'effective choices (exp H)', '—');
    const r3c = UI.readout(c3, 'p(favorite)', '—');
    const capf = (z, c) => (c > 0 ? c * Math.tanh(z / c) : z);
    V.fig(f3, (g, w, h, t, dt) => {
      const C = V.C();
      const narrow = w < 600;
      const zc = Z3.map((v) => capf(v, S3.cap));
      const p = softmax(zc, S3.T);
      if (!S3.cur) S3.cur = p.slice();
      S3.cur = S3.cur.map((v, k) => v + (p[k] - v) * Math.min(1, dt * 9));
      const H = entropy(p);
      r3a.set(num(H) + ' nats'); r3b.set(num(Math.exp(H), 1)); r3c.set(pct(p[0]));
      // horizontal bars
      const bx0 = 16, bwAll = narrow ? w - 32 : w * 0.58 - 24;
      const labW = narrow ? 78 : 92;
      const by0 = 34, bh = narrow ? 230 : h - 60;
      const rowH = bh / TOK3.length;
      V.text(g, `p(v) = softmax(${S3.cap > 0 ? 'c·tanh(z/c)' : 'z'} / T)   T = ${S3.T === 0 ? '0 (greedy)' : num(S3.T, 2)}`, bx0, 20, { size: 11.5, font: 'mono', color: C.muted });
      const barW = bwAll - labW - 52;
      S3.cur.forEach((v, k) => {
        const ry = by0 + k * rowH;
        V.text(g, vis(TOK3[k]), bx0 + labW - 8, ry + rowH / 2, { size: 11.5, font: 'mono', align: 'right', base: 'middle', color: C.ink });
        V.box(g, bx0 + labW, ry + 3, barW, rowH - 6, { r: 4, fill: V.alpha(C.muted, 0.08) });
        const len = Math.max(2, v * barW);
        V.box(g, bx0 + labW, ry + 3, len, rowH - 6, { r: 4, fill: V.alpha(k === 0 ? C.c : C.a, 0.75) });
        V.text(g, pct(v), bx0 + labW + len + 6, ry + rowH / 2, { size: 11, font: 'mono', base: 'middle', color: C.ink });
      });
      // mini charts
      let ax, ay, aw, ah, bxp, byp;
      if (narrow) { aw = (w - 44) / 2; ah = h - by0 - bh - 56; ax = 16; ay = by0 + bh + 34; bxp = ax + aw + 12; byp = ay; }
      else { ax = w * 0.58 + 10; aw = w - ax - 18; ah = (h - 92) / 2; ay = 34; bxp = ax; byp = ay + ah + 44; }
      // (a) softcap curve
      const plot = (px, py, pw, phh, title) => {
        V.box(g, px, py, pw, phh, { r: 8, fill: V.alpha(C.muted, 0.05), stroke: V.alpha(C.muted, 0.25), lw: 1 });
        V.text(g, clip(g, title, pw - 4, 11, 'mono'), px + 2, py - 8, { size: 11, font: 'mono', color: C.muted });
      };
      const cs = String(S3.cap);
      plot(ax, ay, aw, ah, narrow ? (S3.cap > 0 ? `softcap c = ${cs}` : 'softcap off') : S3.cap > 0 ? `softcap: z → ${cs}·tanh(z/${cs})` : 'softcap off: z → z');
      const zx = (v) => ax + 8 + ((v + 0.5) / 6.5) * (aw - 16);
      const zy = (v) => ay + ah - 8 - ((v + 0.5) / 6.5) * (ah - 16);
      g.setLineDash([4, 4]); V.line(g, zx(-0.5), zy(-0.5), zx(6), zy(6), { color: V.alpha(C.muted, 0.5), lw: 1 }); g.setLineDash([]);
      g.beginPath();
      for (let k = 0; k <= 60; k++) { const zz = -0.5 + (k / 60) * 6.5; const yy = capf(zz, S3.cap); k ? g.lineTo(zx(zz), zy(yy)) : g.moveTo(zx(zz), zy(yy)); }
      g.strokeStyle = C.d; g.lineWidth = 2; g.stroke();
      Z3.forEach((v, k) => { V.dot(g, zx(v), zy(capf(v, S3.cap)), k === 0 ? 4.5 : 3, k === 0 ? C.c : C.a); });
      if (S3.cap > 0) { g.setLineDash([2, 4]); V.line(g, zx(-0.5), zy(S3.cap), zx(6), zy(S3.cap), { color: V.alpha(C.d, 0.6), lw: 1 }); g.setLineDash([]); V.text(g, '+c', zx(6) - 4, zy(S3.cap) - 4, { size: 10.5, font: 'mono', align: 'right', color: C.d }); }
      V.text(g, 'z', ax + aw - 10, ay + ah - 12, { size: 10.5, font: 'mono', color: C.muted });
      // (b) entropy × T
      plot(bxp, byp, aw, ah, narrow ? 'entropy H(T)' : 'entropy H(T)  ·  max = ln 10');
      const hmax = Math.log(TOK3.length);
      const tx = (T) => bxp + 8 + (T / 2) * (aw - 16);
      const hy = (Hh) => byp + ah - 8 - (Hh / hmax) * (ah - 16);
      g.setLineDash([3, 4]); V.line(g, tx(0), hy(hmax), tx(2), hy(hmax), { color: V.alpha(C.muted, 0.5), lw: 1 }); g.setLineDash([]);
      g.beginPath();
      for (let k = 0; k <= 60; k++) { const T = (k / 60) * 2; const Hh = entropy(softmax(zc, T)); k ? g.lineTo(tx(T), hy(Hh)) : g.moveTo(tx(T), hy(Hh)); }
      g.strokeStyle = C.b; g.lineWidth = 2; g.stroke();
      const mxp = tx(S3.T), myp = hy(H);
      V.glow(g, mxp, myp, 12 + 2 * Math.sin(t * 3), C.c, 0.6);
      V.dot(g, mxp, myp, 4.5, C.c);
      V.text(g, 'T', bxp + aw - 10, byp + ah - 12, { size: 10.5, font: 'mono', color: C.muted });
    });

    /* ================= Figure 4: filters ================= */
    const f4 = $('#fig-filtros');
    fitH(f4, 410, 560);
    const CTX4 = {
      conf: {
        label: '“The capital of France is”', tail: 0.015,
        tok: [' Paris', ' a', ' the', ' known', ' also', ' Lyon', ' famous', ' today', ' one', ' called', ' not', ' lovely', ' home', ' still', ' in', ' very', ' and', ',', ' well', ' perhaps'],
        z: [9.0, 5.6, 5.2, 4.8, 4.5, 4.2, 4.0, 3.8, 3.6, 3.5, 3.3, 3.1, 3.0, 2.9, 2.8, 2.6, 2.5, 2.3, 2.2, 2.0],
      },
      inc: {
        label: '“My favorite animal is the”', tail: 0.08,
        tok: [' dog', ' cat', ' horse', ' dolphin', ' wolf', ' lion', ' rabbit', ' parrot', ' tiger', ' bear', ' fish', ' penguin', ' elephant', ' monkey', ' tortoise', ' hamster', ' octopus', ' duck', ' ferret', ' chameleon'],
        z: [4.0, 3.8, 3.2, 3.0, 2.8, 2.7, 2.6, 2.5, 2.4, 2.3, 2.2, 2.1, 2.0, 1.9, 1.8, 1.7, 1.6, 1.5, 1.4, 1.3],
      },
    };
    const S4 = { ctx: 'conf', meth: 'topp', k: 20, p: 0.95, mp: 0.1, cur: null };
    const c4 = UI.controls(f4);
    UI.seg(c4, { label: 'Context', options: [{ v: 'conf', label: 'confident' }, { v: 'inc', label: 'uncertain' }], value: 'conf', onChange: (v) => { S4.ctx = v; } });
    UI.seg(c4, { label: 'Method', options: [{ v: 'none', label: 'none' }, { v: 'greedy', label: 'greedy' }, { v: 'topk', label: 'top-k' }, { v: 'topp', label: 'top-p' }, { v: 'minp', label: 'min-p' }], value: 'topp', onChange: (v) => { S4.meth = v; vis4(); } });
    const sK = UI.slider(c4, { label: 'k', min: 1, max: 20, step: 1, value: 20, onInput: (v) => { S4.k = v; } });
    const sP = UI.slider(c4, { label: 'p', min: 0.5, max: 1, step: 0.01, value: 0.95, fmt: (v) => num(v, 2), onInput: (v) => { S4.p = v; } });
    const sM = UI.slider(c4, { label: 'min_p', min: 0, max: 0.5, step: 0.01, value: 0.1, fmt: (v) => num(v, 2), onInput: (v) => { S4.mp = v; } });
    const r4a = UI.readout(c4, 'kept', '—');
    const r4b = UI.readout(c4, 'mass kept', '—');
    const r4c = UI.readout(c4, 'p(favorite) after', '—');
    function vis4() { sK.el.style.display = S4.meth === 'topk' ? '' : 'none'; sP.el.style.display = S4.meth === 'topp' ? '' : 'none'; sM.el.style.display = S4.meth === 'minp' ? '' : 'none'; }
    vis4();
    function filt4() {
      const D = CTX4[S4.ctx];
      const pe = softmax(D.z).map((v) => v * (1 - D.tail));
      const keep = pe.map(() => false);
      let tailKeep = 0;
      if (S4.meth === 'none') { keep.fill(true); tailKeep = D.tail; }
      else if (S4.meth === 'greedy') keep[0] = true;
      else if (S4.meth === 'topk') for (let k = 0; k < S4.k; k++) keep[k] = true;
      else if (S4.meth === 'topp') {
        let acc = 0;
        for (let k = 0; k < pe.length; k++) { keep[k] = true; acc += pe[k]; if (acc >= S4.p - 1e-9) break; }
        if (acc < S4.p - 1e-9) tailKeep = Math.min(D.tail, S4.p - acc);
        if (S4.p >= 0.999) tailKeep = D.tail;
      } else if (S4.meth === 'minp') {
        const th = S4.mp * pe[0];
        pe.forEach((v, k) => { keep[k] = v >= th; });
        if (S4.mp === 0) tailKeep = D.tail;
      }
      const kept = pe.reduce((s, v, k) => s + (keep[k] ? v : 0), 0) + tailKeep;
      const post = pe.map((v, k) => (keep[k] ? v / kept : 0));
      return { D, pe, keep, tailKeep, kept, post, tailPost: tailKeep / kept };
    }
    V.fig(f4, (g, w, h, t, dt) => {
      const C = V.C();
      const narrow = w < 600;
      const F = filt4();
      const n = F.pe.length;
      if (!S4.cur || S4.cur.length !== n + 1) S4.cur = F.post.concat([F.tailPost]);
      const tgt = F.post.concat([F.tailPost]);
      S4.cur = S4.cur.map((v, k) => v + (tgt[k] - v) * Math.min(1, dt * 8));
      const nk = F.keep.filter(Boolean).length;
      r4a.set(nk + (F.tailKeep > 0 ? ' + tail' : ''));
      r4b.set(pct(F.kept, 1)); r4c.set(pct(F.post[0], 1));
      const minpLine = S4.meth === 'minp' && narrow;
      V.text(g, clip(g, `context: ${F.D.label}  ·  T = 1`, S4.meth === 'minp' && !narrow ? w - 300 : w - 32, 12, 'mono'), 16, 20, { size: 12, font: 'mono', color: C.ink });
      const cols = narrow ? 1 : 2;
      const perCol = Math.ceil((n + 1) / cols);
      const top = minpLine ? 50 : 36, bottom = h - 46;
      const rowH = (bottom - top) / perCol;
      const colW = (w - 32 - (cols - 1) * 24) / cols;
      const labW = narrow ? 84 : 92;
      const barW = colW - labW - 58;
      const scale = Math.max(F.pe[0], ...F.post, 0.05);
      let cutY = null;
      for (let k = 0; k <= n; k++) {
        const col = Math.floor(k / perCol), r = k % perCol;
        const x = 16 + col * (colW + 24), y = top + r * rowH;
        const isTail = k === n;
        const orig = isTail ? F.D.tail : F.pe[k];
        const kept = isTail ? F.tailKeep > 0 : F.keep[k];
        const lab = isTail ? 'tail' : vis(F.D.tok[k]);
        V.text(g, clip(g, lab, labW - 8, 11, 'mono'), x + labW - 8, y + rowH / 2, { size: 11, font: 'mono', align: 'right', base: 'middle', color: kept ? C.ink : C.muted });
        const ow = Math.max(2, (orig / scale) * barW);
        const bh = Math.max(4, rowH - 5);
        if (kept) {
          V.box(g, x + labW, y + (rowH - bh) / 2, ow, bh, { r: 3, stroke: V.alpha(C.a, 0.6), lw: 1 });
          const nw = Math.max(2, (S4.cur[k] / scale) * barW);
          V.box(g, x + labW, y + (rowH - bh) / 2, Math.min(barW, nw), bh, { r: 3, fill: V.alpha(k === 0 ? C.c : C.a, 0.72) });
          V.text(g, pct(isTail ? F.tailPost : F.post[k]), x + labW + Math.min(barW, Math.max(ow, nw)) + 5, y + rowH / 2, { size: 10.5, font: 'mono', base: 'middle', color: C.ink });
          if (isTail && F.tailKeep < F.D.tail - 1e-9) V.text(g, '(part)', x + labW + Math.min(barW, Math.max(ow, nw)) + 44, y + rowH / 2, { size: 10.5, font: 'mono', base: 'middle', color: C.muted });
        } else {
          V.box(g, x + labW, y + (rowH - bh) / 2, ow, bh, { r: 3, fill: V.alpha(C.muted, 0.12) });
          hatch(g, x + labW, y + (rowH - bh) / 2, ow, bh, V.alpha(C.muted, 0.45));
          V.text(g, isTail ? `${pct(orig)} in ${fmtInt(154860)} tokens` : pct(orig), x + labW + ow + 5, y + rowH / 2, { size: 10.5, font: 'mono', base: 'middle', color: C.muted, alpha: 0.8 });
          if (cutY == null && col === 0) cutY = y;
        }
      }
      // scissors at the cut boundary (column 0)
      if (cutY != null) {
        const sx = 16 + labW - 4, sy = cutY - 1 + Math.sin(t * 3) * 1.5;
        g.setLineDash([4, 3]); V.line(g, 16, sy, 16 + colW, sy, { color: V.alpha(C.c, 0.7), lw: 1.4 }); g.setLineDash([]);
        scissors(g, sx - labW + 20, sy, 9, C.c);
      }
      // min-p line
      if (S4.meth === 'minp') {
        const th = S4.mp * F.pe[0];
        for (let cc = 0; cc < cols; cc++) {
          const lx = 16 + cc * (colW + 24) + labW + (th / scale) * barW;
          g.setLineDash([3, 3]); V.line(g, lx, top - 2, lx, top + perCol * rowH, { color: C.d, lw: 1.4 }); g.setLineDash([]);
        }
        if (minpLine) V.text(g, `amber line: min_p · p_max = ${pct(th, 1)}`, 16, 38, { size: 10.5, font: 'mono', color: C.d });
        else V.text(g, `amber line: min_p · p_max = ${pct(th, 1)}`, w - 16, 20, { size: 10.5, font: 'mono', align: 'right', color: C.d });
      }
      // cumulative mass strip
      const fy = h - 22, fx = 16, fw = w - 32;
      let acc = 0;
      V.box(g, fx, fy, fw, 10, { r: 5, fill: V.alpha(C.muted, 0.1) });
      F.pe.forEach((v, k) => { const x = fx + acc * fw, ww = v * fw; V.box(g, x, fy, Math.max(0.5, ww - 0.5), 10, { r: 2, fill: F.keep[k] ? V.alpha(k === 0 ? C.c : C.a, 0.75) : V.alpha(C.muted, 0.25) }); acc += v; });
      if (F.tailKeep > 0) V.box(g, fx + acc * fw, fy, F.tailKeep * fw, 10, { r: 2, fill: V.alpha(C.a, 0.4) });
      if (S4.meth === 'topp') { const px = fx + S4.p * fw; V.line(g, px, fy - 5, px, fy + 15, { color: C.d, lw: 2 }); V.text(g, 'p', px, fy - 8, { size: 11, font: 'mono', align: 'center', color: C.d }); }
      V.text(g, 'cumulative mass (decreasing order)', fx, fy - 6, { size: 10.5, font: 'mono', color: C.muted });
    });

    /* ================= Figure 5: Gumbel race ================= */
    const f5 = $('#fig-gumbel');
    fitH(f5, 400, 500);
    const TOK5 = [' yes', ' no', ' maybe', ' never', ' sure', ' hmm'];
    const Z5 = [2.0, 1.5, 1.0, 0.3, 0.0, -0.5];
    const R5 = V.rng(2026);
    const gum = () => { let u = R5(); while (u <= 1e-12) u = R5(); return -Math.log(-Math.log(u)); };
    const S5 = { T: 1, speed: 'lento', counts: Z5.map(() => 0), n: 0, race: null, acc: 0, age: 0 };
    const newRace = () => {
      const g = Z5.map(() => (S5.T > 0 ? gum() : 0));
      const s = Z5.map((z, k) => (S5.T > 0 ? z / S5.T + g[k] : z));
      const win = argmax(s);
      return { g, s, win };
    };
    const reset5 = (seed = 0) => { S5.counts = Z5.map(() => 0); S5.n = 0; S5.race = null; S5.age = 0; for (let k = 0; k < seed; k++) { const rc = newRace(); S5.counts[rc.win]++; S5.n++; } };
    const c5 = UI.controls(f5);
    UI.slider(c5, { label: 'Temperature', min: 0, max: 2, step: 0.05, value: 1, fmt: (v) => (v === 0 ? '0 argmax' : num(v, 2)), onInput: (v) => { S5.T = v; reset5(40); } });
    UI.seg(c5, { label: 'Pace', options: [{ v: 'lento', label: '1 race' }, { v: 'rapido', label: '10/s' }, { v: 'turbo', label: 'turbo' }], value: 'lento', onChange: (v) => { S5.speed = v; } });
    UI.button(c5, '↺ reset', () => reset5());
    reset5(60);
    const r5a = UI.readout(c5, 'races', '0');
    const r5b = UI.readout(c5, 'TV(empirical, softmax)', '—');
    V.fig(f5, (g, w, h, t, dt) => {
      const C = V.C();
      const narrow = w < 600;
      const target = softmax(Z5, S5.T);
      const dur = S5.speed === 'lento' ? 1.7 : 0.1;
      if (!LA.paused) {
        if (S5.speed === 'turbo') {
          for (let k = 0; k < 12; k++) { const rc = newRace(); S5.counts[rc.win]++; S5.n++; S5.race = rc; }
          S5.age = dur;
        } else {
          S5.age += dt;
          if (!S5.race || S5.age >= dur) { if (S5.race && S5.age >= dur) { /* counted at the end */ } S5.race = newRace(); S5.age = 0; S5.counted = false; }
          if (!S5.counted && S5.age >= dur * 0.7) { S5.counts[S5.race.win]++; S5.n++; S5.counted = true; }
        }
      }
      const emp = S5.counts.map((c) => (S5.n ? c / S5.n : 0));
      r5a.set(fmtInt(S5.n));
      r5b.set(S5.n ? num(0.5 * emp.reduce((s, v, k) => s + Math.abs(v - target[k]), 0), 3) : '—');
      const rc = S5.race || { g: Z5.map(() => 0), s: Z5.map((z) => z), win: 0 };
      const ph = S5.speed === 'lento' ? V.clamp(S5.age / dur, 0, 1) : 1;
      // lanes
      const lx = 16, lw = narrow ? w - 32 : w * 0.6 - 24;
      const ly = 34, lh = narrow ? 250 : h - 60;
      const labW = narrow ? 58 : 66;
      const tx0 = lx + labW, tx1 = lx + lw - 10;
      const base = Z5.map((z) => (S5.T > 0 ? z / S5.T : z));
      const smin = Math.min(...base) - 2.2, smax = Math.max(...base) + 5.2;
      const X = (v) => tx0 + V.clamp((v - smin) / (smax - smin), 0, 1) * (tx1 - tx0);
      const rowH = lh / Z5.length;
      V.text(g, clip(g, S5.T > 0 ? (narrow ? 'score = z/T + g (pink = noise)' : 'score = z/T + g  (flag = z/T; pink stretch = noise g)') : 'T = 0: no noise, the largest logit wins', w - 32, 11.5, 'mono'), lx, 20, { size: 11.5, font: 'mono', color: C.muted });
      Z5.forEach((z, k) => {
        const y = ly + k * rowH + rowH / 2;
        const isW = ph >= 0.7 && rc.win === k;
        V.text(g, vis(TOK5[k]), tx0 - 8, y, { size: 12, font: 'mono', align: 'right', base: 'middle', color: isW ? C.c : C.ink });
        V.line(g, tx0, y, tx1, y, { color: V.alpha(C.muted, 0.25), lw: 1 });
        const fx = X(base[k]);
        // flag
        V.line(g, fx, y - rowH * 0.36, fx, y + rowH * 0.1, { color: C.d, lw: 1.6 });
        g.beginPath(); g.moveTo(fx, y - rowH * 0.36); g.lineTo(fx + 9, y - rowH * 0.3); g.lineTo(fx, y - rowH * 0.22); g.closePath(); g.fillStyle = C.d; g.fill();
        // runner
        const p1 = V.clamp(ph / 0.3, 0, 1), p2 = V.clamp((ph - 0.3) / 0.4, 0, 1);
        const x1 = V.lerp(X(smin), fx, V.ease.out(p1));
        const xs = X(rc.s[k]);
        const xr = p1 < 1 ? x1 : V.lerp(fx, xs, V.ease.out(p2));
        if (p1 >= 1 && S5.T > 0) V.line(g, fx, y + 1, xr, y + 1, { color: V.alpha(C.c, 0.85), lw: 4 });
        V.line(g, X(smin), y + 1, Math.min(fx, xr), y + 1, { color: V.alpha(C.a, 0.5), lw: 4 });
        if (isW) V.glow(g, xr, y, 16, C.c);
        V.dot(g, xr, y + 1, isW ? 6 : 4.5, isW ? C.c : C.ink);
        if (isW) V.text(g, 'wins', Math.min(xr + 10, tx1 - 30), y - 8, { size: 11, weight: 600, color: C.c });
      });
      // histogram
      let hx, hy, hw, hh;
      if (narrow) { hx = 16; hy = ly + lh + 30; hw = w - 32; hh = h - hy - 34; }
      else { hx = w * 0.6 + 10; hy = 40; hw = w - hx - 18; hh = h - 90; }
      V.text(g, 'winners × softmax(z/T)', hx, hy - 10, { size: 11, font: 'mono', color: C.muted });
      const bw = hw / Z5.length;
      const mx = Math.max(0.05, ...target, ...emp) * 1.15;
      Z5.forEach((_, k) => {
        const x = hx + k * bw + 5, bwi = bw - 10;
        const eh = (emp[k] / mx) * (hh - 22), th = (target[k] / mx) * (hh - 22);
        V.box(g, x, hy + hh - 22 - eh, bwi, Math.max(1, eh), { r: 4, fill: V.alpha(rc.win === k && ph >= 0.7 ? C.c : C.a, 0.7) });
        V.line(g, x - 3, hy + hh - 22 - th, x + bwi + 3, hy + hh - 22 - th, { color: C.d, lw: 2.4 });
        V.text(g, clip(g, vis(TOK5[k]), bw - 2, 11, 'mono'), x + bwi / 2, hy + hh - 6, { size: 11, font: 'mono', align: 'center', color: C.muted });
        if (S5.n) V.text(g, pct(emp[k]), x + bwi / 2, hy + hh - 26 - eh, { size: 10.5, font: 'mono', align: 'center', color: C.ink });
      });
    });

    /* ================= Figure 6: penalties ================= */
    const f6 = $('#fig-penalidades');
    fitH(f6, 420, 520);
    const SEQ6 = ['The', ' cat', ' saw', ' the', ' cat', '.', ' The', ' cat', ' saw'];
    const CAND6 = [[' the', 4.2], [' a', 3.6], [' its', 3.1], [' cat', 2.8], [' mouse', 2.6], ['.', 2.2], [' bird', 1.9], [' saw', -0.8]];
    const cnt6 = {}; SEQ6.forEach((s) => { cnt6[s] = (cnt6[s] || 0) + 1; });
    const S6 = { r: 1, a: 0, b: 0, cur: null };
    const c6 = UI.controls(f6);
    UI.slider(c6, { label: 'repetition_penalty', min: 1, max: 2, step: 0.05, value: 1, fmt: (v) => num(v, 2), onInput: (v) => { S6.r = v; } });
    UI.slider(c6, { label: 'presence_penalty', min: 0, max: 2, step: 0.1, value: 0, fmt: (v) => num(v, 1), onInput: (v) => { S6.a = v; } });
    UI.slider(c6, { label: 'frequency_penalty', min: 0, max: 1, step: 0.05, value: 0, fmt: (v) => num(v, 2), onInput: (v) => { S6.b = v; } });
    const r6a = UI.readout(c6, 'greedy picks', '—');
    const r6b = UI.readout(c6, 'p(chosen)', '—');
    const pen6 = (z, c) => { let v = z; if (c > 0) v = v > 0 ? v / S6.r : v * S6.r; return v - S6.a * (c > 0 ? 1 : 0) - S6.b * c; };
    V.fig(f6, (g, w, h, t, dt) => {
      const C = V.C();
      const narrow = w < 600;
      const zs = CAND6.map(([s, z]) => pen6(z, cnt6[s] || 0));
      if (!S6.cur) S6.cur = zs.slice();
      S6.cur = S6.cur.map((v, k) => v + (zs[k] - v) * Math.min(1, dt * 8));
      const win = argmax(zs);
      const p = softmax(zs);
      r6a.set(`“${vis(CAND6[win][0])}”`); r6b.set(pct(p[win]));
      // sequence
      V.text(g, 'generated so far:', 16, 20, { size: 11, font: 'mono', color: C.muted });
      let x = 16, y = 30;
      const lastIdx = {}; SEQ6.forEach((s, k) => { lastIdx[s] = k; });
      SEQ6.forEach((s, k) => {
        const lab = vis(s);
        const pw = pill(g, lab, 0, 0, { dry: true, size: 12 });
        if (x + pw + 18 > w - 16) { x = 16; y += 32; }
        const rep = cnt6[s] > 1;
        pill(g, lab, x, y, { size: 12, fill: V.alpha(rep ? C.d : C.b, 0.16), stroke: V.alpha(rep ? C.d : C.b, 0.7) });
        if (rep && lastIdx[s] === k) { V.dot(g, x + pw, y + 1, 9, C.d); V.text(g, '×' + cnt6[s], x + pw, y + 1.5, { size: 10.5, font: 'mono', weight: 700, align: 'center', base: 'middle', color: onInk() }); }
        x += pw + 8;
      });
      pill(g, '?', x, y, { size: 12, dash: [3, 3], stroke: V.alpha(C.muted, 0.7), fill: 'transparent', color: C.muted });
      // candidates
      const top = y + 44, bottom = h - 44;
      const rowH = (bottom - top) / CAND6.length;
      const labW = narrow ? 82 : 100;
      const zmin = -2.5, zmax = 4.6;
      const ax0 = 16 + labW, ax1 = w - (narrow ? 84 : 120);
      const X = (v) => ax0 + ((v - zmin) / (zmax - zmin)) * (ax1 - ax0);
      V.line(g, X(0), top - 6, X(0), bottom, { color: V.alpha(C.muted, 0.6), lw: 1 });
      V.text(g, '0', X(0), top - 10, { size: 10.5, font: 'mono', align: 'center', color: C.muted });
      V.text(g, 'logit', ax1, top - 10, { size: 10.5, font: 'mono', align: 'right', color: C.muted });
      const hiK = [0, 3, 5, 7][Math.floor(t / 2.4) % 4];
      CAND6.forEach(([s, z], k) => {
        const yy = top + k * rowH;
        const c = cnt6[s] || 0;
        const isW = k === win;
        V.text(g, vis(s) + (c ? `  ×${c}` : ''), ax0 - 8, yy + rowH / 2, { size: 11.5, font: 'mono', align: 'right', base: 'middle', color: isW ? C.c : C.ink });
        const bh = Math.max(6, Math.min(24, rowH - 8));
        const xo = X(Math.min(0, z)), wo = Math.abs(X(z) - X(0));
        V.box(g, xo, yy + (rowH - bh) / 2, Math.max(1, wo), bh, { r: 3, stroke: V.alpha(C.muted, 0.7), lw: 1 });
        const zc = S6.cur[k];
        const xn = X(Math.min(0, zc)), wn = Math.abs(X(zc) - X(0));
        V.box(g, xn, yy + (rowH - bh) / 2 + 2, Math.max(1, wn), bh - 4, { r: 3, fill: V.alpha(isW ? C.c : c ? C.d : C.a, 0.75) });
        const endX = Math.max(X(z), X(zc), X(0));
        V.text(g, c ? `${num(z, 1)} → ${num(zs[k], 2)}` : num(z, 1), endX + 6, yy + rowH / 2, { size: 10.5, font: 'mono', base: 'middle', color: c ? C.d : C.muted });
        if (isW) star(g, ax0 - 8 - measure(g, vis(s) + (c ? `  ×${c}` : ''), 11.5, 'mono') - 10, yy + rowH / 2, 6, C.c);
        if (k === hiK) V.box(g, 14, yy + 1, w - 28, rowH - 2, { r: 6, stroke: V.alpha(C.b, 0.5), lw: 1 });
      });
      // arithmetic for the highlighted candidate
      const [hs, hz] = CAND6[hiK], hc = cnt6[hs] || 0;
      const step1 = hz > 0 ? `${num(hz, 1)}/${num(S6.r, 2)}` : `${num(hz, 1)}·${num(S6.r, 2)}`;
      const txt = `“${vis(hs)}” (×${hc}): ${step1} − ${num(S6.a, 1)}·1 − ${num(S6.b, 2)}·${hc} = ${num(zs[hiK], 2)}`;
      V.text(g, clip(g, txt, w - 32, narrow ? 11 : 12.5, 'mono'), 16, h - 14, { size: narrow ? 11 : 12.5, font: 'mono', color: C.b });
    });

    /* ================= Figure 7: reasoning panel ================= */
    const f7 = $('#fig-raciocinio');
    fitH(f7, 360, 380);
    const MT7 = [256, 512, 800, 1024, 2048, 4096, 8192, 16384, 32768, 65536];
    const LEN7 = { glm: { Low: 1500, High: 5000, Max: 9000 }, qwen: { low: 1200, medium: 3000, xhigh: 6000 }, ds: { low: 4000, high: 7000, max: 12500 } };
    const ANS7 = 700;
    const S7 = { rec: 'ds', think: true, eff: 'max', mt: 2, budget: 'none', hot: false, t0: -10 };
    const c7 = UI.controls(f7);
    const restart7 = () => { S7.t0 = H7 ? H7.t : 0; };
    UI.seg(c7, { label: 'Recipe', options: [{ v: 'glm', label: 'GLM' }, { v: 'qwen', label: 'Qwen' }, { v: 'ds', label: 'DeepSeek' }], value: 'ds', onChange: (v) => { S7.rec = v; restart7(); } });
    UI.toggle(c7, { label: 'thinking', value: true, onChange: (v) => { S7.think = v; restart7(); } });
    UI.select(c7, { label: 'Effort', options: [{ v: 'none', label: '(absent)' }, { v: 'low', label: 'low' }, { v: 'medium', label: 'medium' }, { v: 'high', label: 'high' }, { v: 'xhigh', label: 'xhigh' }, { v: 'max', label: 'max' }], value: 'max', onChange: (v) => { S7.eff = v; restart7(); } });
    UI.slider(c7, { label: 'max_tokens', min: 0, max: MT7.length - 1, step: 1, value: 2, fmt: (v) => fmtInt(MT7[v]), onInput: (v) => { S7.mt = v; restart7(); } });
    UI.seg(c7, { label: 'thinking_token_budget', options: [{ v: 'none', label: 'no field' }, { v: '0', label: '0' }, { v: '1024', label: '1024' }, { v: '4096', label: '4096' }], value: 'none', onChange: (v) => { S7.budget = v; restart7(); } });
    UI.toggle(c7, { label: 'hotfix #31 at boot (DS)', value: false, onChange: (v) => { S7.hot = v; restart7(); } });
    function resolve7() {
      const r = S7.rec, e = S7.eff, th = S7.think;
      const o = { http: 200, err: '', level: 'off', dirTok: 0, kw: '', rule: '', tail: '', think: 0, forced: false, budNote: '', budWarn: false };
      const cap = (s) => s.charAt(0).toUpperCase() + s.slice(1);
      const effStr = e === 'none' ? '' : `, "reasoning_effort": "${e}"`;
      if (r === 'glm') {
        o.kw = `{"enable_thinking": ${th}${effStr}}`;
        if (!th) { o.rule = 'thinking off: the effort is ignored; the template closes <think></think>'; o.tail = '<|assistant|><think></think>'; }
        else {
          o.level = e === 'low' || e === 'high' ? cap(e) : 'Max';
          o.rule = e === 'low' || e === 'high' ? `“${e}” is accepted → the template writes Reasoning Effort: ${cap(e)}` : e === 'none' ? 'absent → max, written Reasoning Effort: Max (template fallback)' : `“${e}” is outside low/high → max, written Reasoning Effort: Max`;
          o.tail = `<|system|>Reasoning Effort: ${o.level} … <|assistant|><think>`;
          o.think = LEN7.glm[o.level];
        }
        if (S7.budget !== 'none') o.budNote = 'the recipe documents no reasoning budget: only max_tokens caps the reasoning';
      } else if (r === 'qwen') {
        o.kw = `{"enable_thinking": ${th}${effStr}}`;
        if (!th) { o.rule = e === 'high' || e === 'max' ? `“${e}” would be invalid, but with thinking off it is silently ignored` : 'thinking off: empty <think></think> prefilled'; o.tail = '<|im_start|>assistant\\n<think></think>'; }
        else if (e === 'high' || e === 'max') { o.http = 400; o.err = `Unexpected reasoning effort ${e}. Supported types are xhigh (default), medium, and low.`; o.rule = `“${e}” is not accepted in the thinking branch`; }
        else { o.level = e === 'none' ? 'xhigh' : e; o.rule = e === 'none' ? 'absent → xhigh (template default)' : `“${e}” is valid`; o.tail = '<|im_start|>assistant\\n<think>'; o.think = LEN7.qwen[o.level]; }
        if (S7.budget !== 'none') o.budNote = 'chat_template_kwargs.thinking_budget is not honored by this build';
      } else {
        o.kw = th ? `{"thinking": true${effStr}}` : '{"thinking": false}';
        if (!th) { o.rule = 'thinking off: {"thinking": false}'; o.tail = '<｜Assistant｜></think>'; }
        else {
          o.level = e === 'max' || e === 'xhigh' ? 'max' : e === 'high' ? 'high' : 'low';
          o.dirTok = o.level === 'max' ? 93 : o.level === 'high' ? 80 : 0;
          o.rule = e === 'none' ? 'absent → DEFAULT_THINKING=low' : e === 'max' || e === 'xhigh' || e === 'high' ? `“${e}” → ${o.level} (inline patch)` : `“${e}” → low (patch: anything else becomes low)`;
          o.tail = `<BOS>${o.dirTok ? '[directive ' + o.level + ']' : ''} … <｜Assistant｜><think>`;
          o.think = LEN7.ds[o.level];
        }
        if (S7.budget !== 'none') {
          if (!S7.hot) { o.http = 400; o.err = 'thinking_token_budget is not yet supported by the V2 model runner'; o.rule += ' · budget without the hotfix'; }
          else if (th) {
            const B = +S7.budget, M = MT7[S7.mt];
            if (B === 0) { o.think = 0; o.budNote = 'budget 0: reasoning off for this request'; }
            else if (o.think > B && B < M) { o.think = B; o.forced = true; o.budNote = `budget ${B}: </think> forced at the boundary`; }
            else if (o.think > B) { o.budWarn = true; o.budNote = `budget ${B} ≥ max_tokens ${M}: the ceiling cuts before the forced </think>. Keep max_tokens comfortably above the budget.`; }
            else o.budNote = `budget ${B} not reached: natural </think>`;
          }
        }
      }
      return o;
    }
    const H7 = V.fig(f7, (g, w, h, t) => {
      const C = V.C();
      const narrow = w < 600;
      const o = resolve7();
      const M = MT7[S7.mt];
      const rc = C[S7.rec] || C.a;
      // block 1: request → template
      let y = 18;
      V.text(g, 'chat_template_kwargs', 16, y, { size: 11, font: 'mono', color: C.muted });
      y += 8;
      const kwW = pill(g, clip(g, o.kw, narrow ? w - 46 : w * 0.46, 11.5, 'mono'), 16, y, { size: 11.5, fill: V.alpha(rc, 0.12), stroke: V.alpha(rc, 0.7) });
      let rx, ry;
      if (narrow) { rx = 16; ry = y + 34; } else { rx = 16 + kwW + 40; ry = y; V.arrow(g, 16 + kwW + 6, y + 11, rx - 6, y + 11, { color: C.muted }); }
      if (o.http === 400) {
        pill(g, 'HTTP 400', rx, ry, { size: 12, weight: 700, fill: V.alpha(C.red, 0.18), stroke: C.red, color: C.red });
        let yy = textBlock(g, o.err, 16, ry + 44, w - 32, { size: 12.5, font: 'mono', color: C.red, lh: 18 });
        yy = textBlock(g, 'The request fails before reaching the engine: no token is generated.', 16, yy + 10, w - 32, { size: 12.5, color: C.muted });
        if (o.rule) textBlock(g, o.rule, 16, yy + 8, w - 32, { size: 12, color: C.muted, lh: 16 });
        return;
      }
      pill(g, clip(g, `level: ${o.level}${o.dirTok ? ` · +${o.dirTok} tokens in the prompt` : ''}`, narrow ? w - 32 : w - rx - 16, 11.5, 'mono'), rx, ry, { size: 11.5, fill: V.alpha(C.b, 0.15), stroke: V.alpha(C.b, 0.7) });
      const y2 = ry + 32;
      const yRule = textBlock(g, o.rule, 16, y2 + 10, w - 32, { size: 12, color: C.ink, lh: 16 });
      V.text(g, noLig(clip(g, 'end of prompt (simplified): ' + o.tail, w - 32, 11, 'mono')), 16, yRule + 4, { size: 11, font: 'mono', color: C.muted });
      // block 2: the token strip
      const total = o.think + (S7.think ? 1 : 0) + ANS7;
      const Xmax = Math.max(M, total) * 1.04;
      const bx = 16, bw = w - 32, by = yRule + 40, bh = 34;
      const X = (n) => bx + (n / Xmax) * bw;
      const tt = t - S7.t0;
      const prog = V.clamp(tt / 2.6, 0, 1);
      const genEnd = Math.min(M, total);
      const front = genEnd * V.ease.inOut(prog);
      // label centered on the marker, but kept inside the strip (never spills past the edges)
      const markLabel = (s, x, yy, o2) => { const lw = measure(g, noLig(s), o2.size, 'mono', o2.weight || 400); const lx = V.clamp(x - lw / 2, bx, bx + bw - lw); V.text(g, noLig(s), lx, yy, Object.assign({ font: 'mono', align: 'left' }, o2)); };
      V.text(g, 'generated tokens →', bx, by - 8, { size: 11, font: 'mono', color: C.muted });
      V.box(g, bx, by, bw, bh, { r: 8, fill: V.alpha(C.muted, 0.07), stroke: V.alpha(C.muted, 0.25), lw: 1 });
      // ghost of what would have been generated
      if (total > M) {
        g.setLineDash([4, 4]); V.box(g, X(M) + 2, by + 4, X(total) - X(M) - 2, bh - 8, { r: 5, stroke: V.alpha(C.muted, 0.5), lw: 1 }); g.setLineDash([]);
        if (X(total) - X(M) > 90) V.text(g, 'not generated (cut)', (X(M) + X(total)) / 2, by + bh / 2 + 0.5, { size: 10.5, font: 'mono', align: 'center', base: 'middle', color: C.muted });
      }
      const segTh = Math.min(front, o.think);
      if (segTh > 0) V.box(g, X(0), by + 4, Math.max(1, X(segTh) - X(0)), bh - 8, { r: 5, fill: V.alpha(C.b, 0.7) });
      const ansStart = o.think + (S7.think ? 1 : 0);
      if (front > ansStart) V.box(g, X(ansStart), by + 4, Math.max(1, X(Math.min(front, total)) - X(ansStart)), bh - 8, { r: 5, fill: V.alpha(C.a, 0.75) });
      // particles running over what has been generated (streaming)
      for (let k = 0; k < 14; k++) {
        const u = ((t * 0.18 + k / 14) % 1) * front;
        if (front < 2) break;
        V.dot(g, X(u), by + 9 + ((k * 7) % 17), 1.6, V.alpha(C.ink, 0.55));
      }
      if (prog < 1) V.glow(g, X(front), by + bh / 2, 12, C.c, 0.8);
      // markers (labels below the strip, one per line)
      let ly = by + bh + 15;
      if (S7.think && front > o.think) {
        const ex = X(o.think);
        V.line(g, ex, by - 2, ex, by + bh + 4, { color: o.forced ? C.red : C.ink, lw: 2.4 });
        if (o.forced) star(g, ex, by - 6, 6, C.red);
        markLabel(o.forced ? `forced </think> (token ${fmtInt(o.think)})` : `natural </think> (token ${fmtInt(o.think)})`, ex, ly, { size: 10.5, color: o.forced ? C.red : C.ink });
        ly += 15;
      }
      const mxX = X(M);
      V.line(g, mxX, by - 4, mxX, by + bh + 4, { color: C.d, lw: 2 });
      markLabel(`max_tokens = ${fmtInt(M)}`, mxX, ly, { size: 11, weight: 600, color: C.d });
      ly += 15;
      if (S7.rec === 'ds' && S7.hot && S7.budget !== 'none' && +S7.budget > 0) {
        const bX = X(+S7.budget);
        const bCol = o.forced ? C.red : C.muted;
        g.setLineDash(o.forced ? [3, 3] : [2, 4]); V.line(g, bX, o.forced ? by + bh : by - 2, bX, ly - 10, { color: bCol, lw: 1.2 }); g.setLineDash([]);
        markLabel(`thinking_token_budget = ${S7.budget}${o.forced ? '' : ' (not reached)'}`, bX, ly, { size: 10.5, color: bCol });
        ly += 15;
      }
      // block 3: result
      const ry3 = ly + 8;
      const cutThink = M < o.think;
      const cutAns = !cutThink && M < total;
      const fin = cutThink || cutAns ? 'length' : 'stop';
      const done = prog >= 1;
      const reasonN = Math.min(o.think, M);
      const contentN = Math.max(0, Math.min(M, total) - ansStart);
      const cards = [
        ['finish_reason', done ? fin : '…', fin === 'length' ? C.red : C.a],
        ['reasoning', done ? `${fmtInt(reasonN)} tokens` : '…', C.b],
        ['content', done ? (cutThink ? 'null / empty' : cutAns ? `cut (${fmtInt(contentN)})` : `ok (${fmtInt(contentN)})`) : '…', cutThink ? C.red : C.a],
      ];
      const cw = (w - 32 - 2 * 10) / 3;
      cards.forEach(([k, v, col], j) => {
        const cx = 16 + j * (cw + 10);
        V.box(g, cx, ry3, cw, 50, { r: 10, fill: V.alpha(col, 0.1), stroke: V.alpha(col, 0.5) });
        V.text(g, k, cx + 10, ry3 + 17, { size: 10.5, font: 'mono', color: C.muted });
        V.text(g, clip(g, v, cw - 16, narrow ? 11.5 : 13, 'mono', 600), cx + 10, ry3 + 38, { size: narrow ? 11.5 : 13, font: 'mono', weight: 600, color: col });
      });
      const noteY = ry3 + 74;
      let note = o.budNote;
      if (!note) note = S7.rec === 'ds' && o.level === 'max' ? 'DS max: ~12.5k reasoning tokens on a moderate prompt (measured); ≈ 3× the latency of low' : 'illustrative reasoning lengths (except DS max, measured); illustrative 700-token answer';
      textBlock(g, note, 16, noteY, w - 32, { size: 12, color: o.forced ? C.red : o.budWarn ? C.d : C.muted, weight: o.budWarn ? 600 : 400, lh: 17 });
    });

    /* ================= Figure 8: loop detector ================= */
    const f8 = $('#fig-loop');
    fitH(f8, 380, 540);
    const R8 = V.rng(88);
    const SYL = ['ra', 'te', 'mi', 'no', 'ca', 'lo', 'sa', 'vi', 'du', 'pe', 'ta', 'ri', 'mo', 'le', 'ga', 'zu', 'fe', 'bo', 'ni', 'xa', 'po', 'se'];
    const mkWord = () => { const n = 2 + R8.int(2); let s = ''; for (let k = 0; k < n; k++) s += SYL[R8.int(SYL.length)]; return s; };
    const LEX = Array.from({ length: 3000 }, mkWord);
    const COMMON = ['so', 'the', 'of', 'that', 'a', 'for', 'let', 'this', 'is', 'one', 'in', 'if', 'with', 'result', 'step', 'value', 'but', 'now'];
    const PHRASES = [
      ['let', 'us', 'check', 'the', 'result', 'again', 'very', 'carefully', 'now'],
      ['this', 'confirms', 'the', 'previous', 'step', 'and', 'we', 'can', 'move', 'on'],
      ['so', 'the', 'final', 'value', 'must', 'be', 'consistent', 'with', 'the', 'hypothesis'],
      ['I', 'need', 'to', 'remember', 'the', 'initial', 'condition', 'holds', 'everywhere'],
      ['looking', 'at', 'it', 'another', 'way', 'the', 'problem', 'gets', 'simpler'],
      ['the', 'next', 'stage', 'is', 'to', 'add', 'the', 'remaining', 'terms'],
    ];
    const healthy = (n) => { const out = []; while (out.length < n) { if (R8() < 0.04) out.push(...PHRASES[R8.int(PHRASES.length)]); else out.push(R8() < 0.32 ? COMMON[R8.int(COMMON.length)] : LEX[R8.int(LEX.length)]); } return out.slice(0, n); };
    const TPL = [['wait', 'let', 'me', 'double', 'check', 'that', 'step', '#', 'with', 'more', 'care'], ['actually', 'step', '#', 'looks', 'right', 'but', 'let', 'us', 'review', 'it'], ['so', 'the', 'value', 'of', '#', 'stays', 'the', 'same', 'as', 'before']];
    const VAR = ['two', 'three', 'x', 'y'];
    const looping = (n) => { const out = []; while (out.length < n) { const tp = TPL[R8.int(TPL.length)], v = VAR[R8.int(VAR.length)]; tp.forEach((wd) => out.push(wd === '#' ? v : wd)); } return out.slice(0, n); };
    const WIN = 700;
    const TR8 = {
      tail: healthy(WIN * 36),
      loop: healthy(WIN * 9).concat(looping(WIN * 27)),
      trans: (() => { const a = healthy(WIN * 14); const quote = a.slice(WIN * 3, WIN * 7); return a.concat(quote, healthy(WIN * 18)); })(),
    };
    function analyse8(words) {
      const seen = new Set();
      const rows = [];
      let chars = 0;
      for (let i = 0; i + WIN / 2 <= words.length; i += WIN) {
        const chunk = words.slice(i, i + WIN);
        if (chunk.length < WIN / 2) break;
        const sh = new Set();
        for (let j = 0; j + 8 <= chunk.length; j++) sh.add(chunk.slice(j, j + 8).join(' '));
        let nov = 0; sh.forEach((s) => { if (!seen.has(s)) nov++; });
        sh.forEach((s) => seen.add(s));
        rows.push({ tok: chars / 4, nov: sh.size ? nov / sh.size : 1, text: chunk.slice(0, 16).join(' ') });
        chunk.forEach((wd) => { chars += wd.length + 1; });
      }
      return rows;
    }
    const RW8 = { tail: analyse8(TR8.tail), loop: analyse8(TR8.loop), trans: analyse8(TR8.trans) };
    function verdict8(rows, th) {
      let dry = 0, onset = null; const trans = [];
      rows.forEach((r, i) => {
        if (r.nov < th) { dry++; if (dry === 3 && onset == null) onset = i - 2; }
        else { if (onset != null) { trans.push([onset, i]); onset = null; } dry = 0; }
      });
      return { onset, trans };
    }
    const S8 = { tr: 'loop', th: 0.02, t0: -6 };
    const c8 = UI.controls(f8);
    UI.seg(c8, { label: 'Trace', options: [{ v: 'tail', label: 'long tail' }, { v: 'loop', label: 'templated loop' }, { v: 'trans', label: 'transient repetition' }], value: 'loop', onChange: (v) => { S8.tr = v; S8.t0 = H8.t - 0.5; } });
    UI.slider(c8, { label: 'threshold', min: 0.005, max: 0.1, step: 0.005, value: 0.02, fmt: (v) => pct(v, 1), onInput: (v) => { S8.th = v; } });
    const H8 = V.fig(f8, (g, w, h, t, dt, S, HH) => {
      const C = V.C();
      const narrow = w < 600;
      const rows = RW8[S8.tr];
      const vd = verdict8(rows, S8.th);
      const px = 52, py = 30, pw = w - px - 18, phh = h - py - (narrow ? 130 : 104);
      const tmax = rows[rows.length - 1].tok * 1.02;
      const X = (tk) => px + (tk / tmax) * pw;
      const Y = (nv) => py + phh - Math.sqrt(V.clamp(nv, 0, 1)) * phh;
      V.box(g, px, py, pw, phh, { r: 6, fill: V.alpha(C.muted, 0.04), stroke: V.alpha(C.muted, 0.2), lw: 1 });
      [0, 0.02, 0.1, 0.25, 0.5, 1].forEach((v) => { V.line(g, px, Y(v), px + pw, Y(v), { color: V.alpha(C.muted, 0.12), lw: 1 }); V.text(g, pct(v, 0), px - 6, Y(v), { size: 10.5, font: 'mono', align: 'right', base: 'middle', color: C.muted }); });
      V.text(g, narrow ? '8-gram novelty (√ axis)' : '8-gram novelty per window (square-root axis)', px, py - 10, { size: 11, font: 'mono', color: C.muted });
      for (let k = 0; k <= 4; k++) { const tk = (tmax * k) / 4; V.text(g, `${fmtInt(tk / 1000)}k`, X(tk), py + phh + 14, { size: 10.5, font: 'mono', align: 'center', color: C.muted }); }
      V.text(g, 'tokens ≈ chars/4', px + pw, py + phh + 28, { size: 10.5, font: 'mono', align: 'right', color: C.muted });
      // threshold
      g.setLineDash([5, 4]); V.line(g, px, Y(S8.th), px + pw, Y(S8.th), { color: C.red, lw: 1.5 }); g.setLineDash([]);
      V.text(g, `threshold ${pct(S8.th, 1)}`, px + pw - 4, Y(S8.th) - 6, { size: 10.5, font: 'mono', align: 'right', color: C.red });
      // verdict regions
      // reading cursor
      const tt = t - S8.t0, per = 6;
      const f = V.clamp((tt % (per + 5)) / per, 0, 1);
      const nShow = Math.max(1, Math.ceil(f * rows.length));
      if (f >= 1) {
        vd.trans.forEach(([a, b]) => { V.box(g, X(rows[a].tok), py + 1, X(rows[b].tok) - X(rows[a].tok), phh - 2, { r: 0, fill: V.alpha(C.d, 0.14) }); });
        if (vd.onset != null) V.box(g, X(rows[vd.onset].tok), py + 1, px + pw - X(rows[vd.onset].tok), phh - 2, { r: 0, fill: V.alpha(C.red, 0.12) });
      }
      g.beginPath();
      rows.slice(0, nShow).forEach((r, i) => { const xx = X(r.tok), yy = Y(r.nov); i ? g.lineTo(xx, yy) : g.moveTo(xx, yy); });
      g.strokeStyle = C.a; g.lineWidth = 2; g.stroke();
      rows.slice(0, nShow).forEach((r) => V.dot(g, X(r.tok), Y(r.nov), 3, r.nov < S8.th ? C.red : C.a));
      let hi = nShow - 1;
      const P = HH.pointer;
      if (P.inside && P.x >= px && P.x <= px + pw && P.y <= py + phh + 20) { let best = 0; rows.forEach((r, i) => { if (Math.abs(X(r.tok) - P.x) < Math.abs(X(rows[best].tok) - P.x)) best = i; }); hi = best; }
      const hr = rows[hi];
      V.line(g, X(hr.tok), py, X(hr.tok), py + phh, { color: V.alpha(C.c, 0.6), lw: 1 });
      V.glow(g, X(hr.tok), Y(hr.nov), 12, C.c, 0.7);
      // verdict + snippet
      const vy = py + phh + (narrow ? 50 : 48);
      let vtxt, vcol;
      if (f < 1 && !P.inside) { vtxt = 'reading the trace…'; vcol = C.muted; }
      else if (vd.onset != null) { vtxt = `LOOP from ~${fmtInt(rows[vd.onset].tok)} tokens on: raising max_tokens does not help`; vcol = C.red; }
      else if (vd.trans.length) { vtxt = `transient repetition (~${fmtInt(rows[vd.trans[0][0]].tok)}–${fmtInt(rows[vd.trans[0][1]].tok)}); then it recovers: not a loop`; vcol = C.d; }
      else { vtxt = 'LONG TAIL: novelty never collapses; raising max_tokens helps'; vcol = C.a; }
      textBlock(g, vtxt, 16, vy, w - 32, { size: narrow ? 12 : 13, weight: 600, color: vcol, lh: 17 });
      V.text(g, clip(g, `window ${hi + 1} · novelty ${pct(hr.nov, 1)} · “${hr.text} …”`, w - 32, 11, 'mono'), 16, h - 12, { size: 11, font: 'mono', color: C.muted });
    });

    /* ================= Figure 9: non-associativity ================= */
    const f9 = $('#fig-float');
    fitH(f9, 410, 460);
    const VALS9 = [256, 0.75, 0.75, -256, 0.75, 0.5, 1.25, -0.25];
    const EXACT9 = VALS9.reduce((a, b) => a + b, 0);
    const R9 = V.rng(9);
    const S9 = { fmt: 'bf16', fixed: false, auto: true, perm: VALS9.map((_, i) => i), hist: [], t0: 0 };
    const rnd9 = (x) => LA.num.round(S9.fmt, x);
    function tree9(perm) {
      const lv = [perm.map((i) => ({ v: rnd9(VALS9[i]), ex: VALS9[i], lost: false }))];
      while (lv[lv.length - 1].length > 1) {
        const prev = lv[lv.length - 1], nxt = [];
        for (let k = 0; k < prev.length; k += 2) { const ex = prev[k].v + prev[k + 1].v; const v = rnd9(ex); nxt.push({ v, ex, lost: v !== ex }); }
        lv.push(nxt);
      }
      return lv;
    }
    function run9() {
      if (S9.fixed) S9.perm = VALS9.map((_, i) => i);
      else { const p = VALS9.map((_, i) => i); for (let k = p.length - 1; k > 0; k--) { const j = R9.int(k + 1); [p[k], p[j]] = [p[j], p[k]]; } S9.perm = p; }
      const lv = tree9(S9.perm);
      S9.hist.push(lv[lv.length - 1][0].v);
      if (S9.hist.length > 30) S9.hist.shift();
      S9.t0 = H9 ? H9.t : 0;
    }
    const c9 = UI.controls(f9);
    UI.button(c9, '▶ run again', () => run9(), { primary: true });
    UI.toggle(c9, { label: 'fixed order (determinism patch)', value: false, onChange: (v) => { S9.fixed = v; S9.hist = []; run9(); } });
    UI.seg(c9, { label: 'Format', options: [{ v: 'bf16', label: 'BF16' }, { v: 'fp32', label: 'FP32' }], value: 'bf16', onChange: (v) => { S9.fmt = v; S9.hist = []; run9(); } });
    UI.toggle(c9, { label: 'auto-run', value: true, onChange: (v) => { S9.auto = v; } });
    V.every(2600, () => { if (S9.auto) run9(); });
    let H9 = null;
    H9 = V.fig(f9, (g, w, h, t) => {
      const C = V.C();
      const narrow = w < 600;
      const lv = tree9(S9.perm);
      const root9 = lv[lv.length - 1][0].v;
      const tt = t - S9.t0;
      const reveal = Math.floor(tt / 0.35);
      const cellW = Math.min(64, (w - 32 - 7 * 6) / 8), cellH = 26;
      const levY = [40, 92, 144, 196].map((v) => v + (narrow ? 4 : 0));
      V.text(g, `sum of 8 terms in ${S9.fmt === 'bf16' ? 'BF16' : 'FP32'} · exact = ${num(EXACT9, 2)}`, 16, 20, { size: 11.5, font: 'mono', color: C.muted });
      const centers = [];
      lv.forEach((lev, L) => {
        const n = lev.length;
        const span = 8 * cellW + 7 * 6;
        const x0 = (w - span) / 2;
        const cs = [];
        lev.forEach((nd, k) => {
          const slotW = span / n;
          const cx = x0 + slotW * k + slotW / 2;
          cs.push(cx);
          const on = L <= reveal;
          const bw = Math.min(L ? cellW * 1.45 : cellW, slotW - 6);
          const col = L === 3 ? C.c : L === 0 ? C.b : C.a;
          if (L > 0 && on) {
            const pc = centers[L - 1];
            V.line(g, pc[2 * k], levY[L - 1] + cellH, cx, levY[L], { color: V.alpha(C.muted, 0.5), lw: 1.2 });
            V.line(g, pc[2 * k + 1], levY[L - 1] + cellH, cx, levY[L], { color: V.alpha(C.muted, 0.5), lw: 1.2 });
          }
          V.box(g, cx - bw / 2, levY[L], bw, cellH, { r: 6, fill: on ? V.alpha(col, 0.18) : V.alpha(C.muted, 0.06), stroke: on ? V.alpha(col, 0.8) : V.alpha(C.muted, 0.3), lw: 1.2 });
          const s = on ? (Math.abs(nd.v) >= 100 ? String(nd.v) : num(nd.v, 2)) : '·';
          V.text(g, s, cx, levY[L] + cellH / 2 + 0.5, { size: 11.5, font: 'mono', align: 'center', base: 'middle', color: C.ink });
          if (on && nd.lost) { V.dot(g, cx + bw / 2 - 3, levY[L] + 3, 4, C.red); }
        });
        centers.push(cs);
      });
      // results line
      const ly = narrow ? 290 : 262, lx0 = 40, lx1 = w - 40;
      const vmin = 1.5, vmax = 5;
      const LX = (v) => lx0 + ((V.clamp(v, vmin, vmax) - vmin) / (vmax - vmin)) * (lx1 - lx0);
      V.line(g, lx0, ly, lx1, ly, { color: V.alpha(C.muted, 0.5), lw: 1.4 });
      V.box(g, LX(3.5), ly - 16, lx1 - LX(3.5), 32, { r: 0, fill: V.alpha(C.b, 0.08) });
      V.box(g, lx0, ly - 16, LX(3.5) - lx0, 32, { r: 0, fill: V.alpha(C.a, 0.08) });
      V.line(g, LX(3.5), ly - 20, LX(3.5), ly + 20, { color: C.d, lw: 1.6 });
      V.text(g, 'tie 3.5', LX(3.5) - 4, ly + 20, { size: 10.5, font: 'mono', align: 'right', color: C.d });
      V.line(g, LX(EXACT9), ly - 12, LX(EXACT9), ly + 12, { color: C.ink, lw: 1, dash: [2, 2] });
      V.text(g, 'exact 3.75', LX(EXACT9) + 4, ly + 20, { size: 10.5, font: 'mono', color: C.ink });
      V.text(g, '← Paris wins', lx0, ly - 22, { size: 10.5, font: 'mono', color: C.a });
      V.text(g, 'Lyon wins →', lx1, ly - 22, { size: 10.5, font: 'mono', align: 'right', color: C.b });
      const tally = {};
      S9.hist.forEach((v, k) => { const key = v.toFixed(3); tally[key] = (tally[key] || 0) + 1; const yy = ly - 3 - (tally[key] - 1) * 5; V.dot(g, LX(v), Math.max(ly - 14, yy), 3.2, V.alpha(v < 3.5 ? C.a : C.b, k === S9.hist.length - 1 ? 1 : 0.6)); });
      const nP = S9.hist.filter((v) => v < 3.5).length, nL = S9.hist.length - nP;
      V.text(g, `last ${S9.hist.length} rounds: Paris ${nP} · Lyon ${nL}`, lx0, ly + 48, { size: 11, font: 'mono', color: C.muted });
      // continuation
      const cy = Math.min(h - 56, ly + 84);
      const win = root9 < 3.5 ? 'Paris' : 'Lyon';
      const margin = root9 - 3.5;
      const cy2 = textBlock(g, `margin z(Lyon) − z(Paris) = sum − 3.5 = ${num(margin, 2)} → greedy picks “${win}”`, 16, cy, w - 32, { size: narrow ? 11 : 12, font: 'mono', color: C.ink, lh: 16 });
      const cont = win === 'Paris' ? '“… the most visited city is Paris, the City of Light.”' : '“… the most visited city is Lyon, the capital of gastronomy.”';
      textBlock(g, cont + '  (illustrative continuations)', 16, cy2 + 6, w - 32, { size: 12.5, color: win === 'Paris' ? C.a : C.b, weight: 600, lh: 17 });
    });
    run9();

    /* ================= Figure 10: acceptance × temperature ================= */
    const f10 = $('#fig-spec');
    fitH(f10, 400, 540);
    const R10 = V.rng(19);
    const N10 = 48, SC10 = 2.2;
    const ZP10 = []; for (let i = 0; i < N10; i++) ZP10.push(SC10 * (i === 0 ? 4.0 : 2.2 - 0.09 * i + R10.normal() * 0.3));
    const ZQ10 = ZP10.map((v, i) => (i === 0 ? v - 0.6 * SC10 : v + R10.normal() * 0.6 * SC10));
    const ORD10 = ZP10.map((v, i) => i).sort((a, b) => ZP10[b] - ZP10[a]);
    const SHOW10 = ORD10.slice(0, 8), REST10 = ORD10.slice(8);
    const ZQ10b = ZQ10.slice(); { const sec = ORD10[1]; const tmp = ZQ10b[0]; ZQ10b[0] = ZQ10b[sec]; ZQ10b[sec] = tmp; }
    const LAB10 = [' Paris', ' a', ' the', ' Lyon', ' one', ' known', ' today', ' also'];
    const S10 = { T: 1, diff: false };
    const c10 = UI.controls(f10);
    UI.slider(c10, { label: 'Temperature', min: 0, max: 2, step: 0.05, value: 1, fmt: (v) => (v === 0 ? '0 greedy' : num(v, 2)), onInput: (v) => { S10.T = v; } });
    UI.toggle(c10, { label: 'different favorites', value: false, onChange: (v) => { S10.diff = v; } });
    const r10a = UI.readout(c10, 'acceptance', '—');
    const r10b = UI.readout(c10, 'TV(p, q)', '—');
    const r10c = UI.readout(c10, 'tokens/step (k = 3, w/ bonus)', '—');
    const alpha10 = (T) => { const p = softmax(ZP10, T), q = softmax(S10.diff ? ZQ10b : ZQ10, T); return { a: p.reduce((s, v, i) => s + Math.min(v, q[i]), 0), p, q }; };
    V.fig(f10, (g, w, h, t) => {
      const C = V.C();
      const narrow = w < 600;
      const { a, p, q } = alpha10(S10.T);
      const k = 3;
      const tps = a >= 0.9999 ? k + 1 : (1 - Math.pow(a, k + 1)) / (1 - a);
      r10a.set(pct(a, 1)); r10b.set(num(1 - a, 3)); r10c.set(num(tps, 2));
      const vals = (arr) => SHOW10.map((i) => arr[i]).concat([REST10.reduce((s, i) => s + arr[i], 0)]);
      const pv = vals(p), qv = vals(q);
      const labs = LAB10.concat(['tail']);
      // vertical bars
      const bx = 16, bw = narrow ? w - 32 : w * 0.6 - 24, by = 34, bh = narrow ? 230 : h - 80;
      const nb = pv.length, slot = bw / nb, bwi = Math.min(42, slot - 8);
      const mx = Math.max(...pv, ...qv, 0.05);
      V.text(g, clip(g, `${narrow ? 'p (mint) · q (lilac)' : 'target p (mint) · drafter q (lilac)'} · T = ${S10.T === 0 ? '0' : num(S10.T, 2)}`, w - 32, 11.5, 'mono'), bx, 20, { size: 11.5, font: 'mono', color: C.muted });
      const labPad = narrow ? 74 : 20;
      pv.forEach((pvv, j) => {
        const x = bx + j * slot + (slot - bwi) / 2;
        const base = by + bh - labPad;
        const hp = (pvv / mx) * (bh - labPad - 24), hq = (qv[j] / mx) * (bh - labPad - 24), hm = Math.min(hp, hq);
        V.box(g, x, base - hp, bwi, Math.max(1, hp), { r: 4, fill: V.alpha(C.a, 0.28) });
        V.box(g, x, base - hm, bwi, Math.max(1, hm), { r: 4, fill: V.alpha(C.a, 0.85) });
        V.box(g, x - 2, base - hq, bwi + 4, Math.max(1, hq), { r: 4, stroke: C.b, lw: 2 });
        if (narrow) {
          g.save(); g.translate(x + bwi / 2 + 4, base + 6); g.rotate(-Math.PI / 2);
          V.text(g, vis(labs[j]), 0, 0, { size: 11, font: 'mono', align: 'right', color: C.muted });
          g.restore();
        } else V.text(g, clip(g, vis(labs[j]), slot - 2, 11, 'mono'), x + bwi / 2, by + bh - 4, { size: 11, font: 'mono', align: 'center', color: C.muted });
      });
      // α(T) curve
      let cx, cy, cw, chh;
      if (narrow) { cx = 40; cy = by + bh + 34; cw = w - 58; chh = h - cy - 34; }
      else { cx = w * 0.6 + 34; cy = 44; cw = w - cx - 20; chh = h - 110; }
      V.box(g, cx, cy, cw, chh, { r: 8, fill: V.alpha(C.muted, 0.05), stroke: V.alpha(C.muted, 0.25), lw: 1 });
      V.text(g, 'acceptance α(T) = Σ min(p, q)', cx, cy - 10, { size: 11, font: 'mono', color: C.muted });
      const TX = (T) => cx + 6 + (T / 2) * (cw - 12);
      const AY = (v) => cy + chh - 6 - v * (chh - 12);
      [0, 0.5, 1].forEach((v) => { V.line(g, cx, AY(v), cx + cw, AY(v), { color: V.alpha(C.muted, 0.14), lw: 1 }); V.text(g, pct(v, 0), cx - 4, AY(v), { size: 10.5, font: 'mono', align: 'right', base: 'middle', color: C.muted }); });
      [[0, 'T = 0'], [1, '1'], [2, '2']].forEach(([T, l]) => V.text(g, l, TX(T), cy + chh + 14, { size: 10.5, font: 'mono', align: T === 0 ? 'left' : T === 2 ? 'right' : 'center', color: C.muted }));
      g.beginPath();
      for (let s = 1; s <= 80; s++) { const T = (s / 80) * 2; const v = alpha10(T).a; s > 1 ? g.lineTo(TX(T), AY(v)) : g.moveTo(TX(T), AY(v)); }
      g.strokeStyle = C.b; g.lineWidth = 2; g.stroke();
      const a0 = alpha10(0).a;
      V.dot(g, TX(0), AY(a0), 4, C.d);
      if (a0 > 0.9) {
        // the curve stays flat near 100% and then drops: label on two short lines, right below the point
        V.text(g, 'greedy:', TX(0) + 8, AY(a0) + 17, { size: 10.5, font: 'mono', color: C.d });
        V.text(g, pct(a0, 0), TX(0) + 8, AY(a0) + 30, { size: 10.5, font: 'mono', color: C.d });
      } else V.text(g, `greedy: ${pct(a0, 0)}`, TX(0) + 8, AY(a0) + (a0 > 0.5 ? 14 : -8), { size: 10.5, font: 'mono', color: C.d });
      V.glow(g, TX(S10.T), AY(a), 12 + 2 * Math.sin(t * 3), C.c, 0.6);
      V.dot(g, TX(S10.T), AY(a), 4.5, C.c);
    });

    /* ================= Figure 11: keyed sampling (TensorFold) ================= */
    const f11 = $('#fig-chave');
    fitH(f11, 520, 740);
    const M64 = (1n << 64n) - 1n;
    const mix64 = (x) => { x ^= x >> 30n; x = (x * 0xBF58476D1CE4E5B9n) & M64; x ^= x >> 27n; x = (x * 0x94D049BB133111EBn) & M64; return x ^ (x >> 31n); };
    // the same uniform(seed, position, ids) as TensorFold (engine/exact_sampling.py): three rounds of splitmix64
    const keyedU = (k, pos, id) => {
      let x = mix64((k + 0x9E3779B97F4A7C15n) & M64);
      x = mix64(x ^ ((BigInt(pos) * 0xD1B54A32D192ED03n) & M64));
      x = mix64(x ^ BigInt(id));
      return Number(x >> 11n) * 2 ** -53 + 2 ** -54;
    };
    const TOK11 = [' king', ' cat', ' moon', ' who', ' saw', ' the', ' and', ' laughed', ' slept'];
    const ID11 = [1203, 4410, 2290, 977, 2318, 264, 323, 11904, 7316];
    const Z11 = [
      [-4, -4, -4, 2.4, 1.5, -4, 0.9, 0.6, 1.0], // after “king”
      [-4, -4, -4, 2.2, 1.7, -4, 1.1, 0.8, 1.3], // “cat”
      [-4, -4, -4, 0.9, -4, -4, 2.4, 0.5, -4], // “moon”
      [-4, -4, -4, -4, 2.4, -4, -4, 1.2, 1.0], // “who”
      [-4, -4, -4, -4, -4, 2.8, -4, -4, -4], // “saw”
      [-4, -4, 2.8, -4, -4, -4, -4, -4, -4], // “the”
      [-4, -4, -4, -4, 0.6, -4, -4, 2.2, 1.9], // “and”
      [-4, -4, -4, 0.3, -4, -4, 1.6, -4, -4], // “laughed”
      [-4, -4, -4, 0.3, -4, -4, 1.6, -4, -4], // “slept”
      [2.4, 2.1, -4, -4, -4, -4, -4, -4, -4], // end of prompt A (“… a”)
      [2.9, 1.6, -4, -4, -4, -4, -4, -4, -4], // end of prompt B (“… a”)
    ];
    const PR11 = {
      A: { label: 'Once upon a time, a', key: 8324308469202069954n, row: 9 },
      B: { label: 'In the castle lived a', key: 2263849561746361393n, row: 10 },
    };
    const N11 = 7, P11 = 4; // generated tokens; absolute position of the first one (a 4-token prompt)
    const RUN11 = ['alone', 'in a batch (+3 neighbors)', 'with drafts'];
    const S11 = { prompt: 'A', seed: false, T: 1, inv: true, step: 0, age: 1.85, res: null };
    const pick11 = (z, T, gs) => {
      let b = 0, bs = -Infinity;
      for (let i = 0; i < z.length; i++) {
        const s = T > 0 ? z[i] / T + gs[i] : z[i];
        if (s > bs || (s === bs && ID11[i] < ID11[b])) { bs = s; b = i; }
      }
      return b;
    };
    // batch-dependent kernels: a small (exaggerated) deviation in the logits of runs 2 and 3
    const logits11 = (prev, run, pos) => {
      const z = Z11[prev].slice();
      if (!S11.inv && run > 0) { const r = V.rng(1000 * run + 17 * pos + 5); for (let i = 0; i < z.length; i++) z[i] += (r() - 0.5) * 0.7; }
      return z;
    };
    const keyNoise11 = (k, pos) => ID11.map((id) => -Math.log(-Math.log(keyedU(k, pos, id))));
    // speculative rounds: 2 drafts (greedy guess), verified row by row by the sampling rule
    function rounds11(first, noiseFor) {
      const out = []; let prev = first;
      while (out.length < N11) {
        const drafts = []; let p = prev;
        for (let j = 0; j < 2; j++) { const d = pick11(Z11[p], 0, null); drafts.push(d); p = d; }
        const base = out.length;
        const samples = [prev, drafts[0], drafts[1]].map((pp, j) => pick11(logits11(pp, 2, P11 + base + j), S11.T, noiseFor(P11 + base + j)));
        let j = 0;
        while (j < 2 && samples[j] === drafts[j] && out.length < N11) { out.push(samples[j]); j++; }
        if (out.length < N11) out.push(samples[j]);
        prev = out[out.length - 1];
      }
      return out.slice(0, N11);
    }
    function genKeyed11(run, k) {
      const first = PR11[S11.prompt].row;
      if (run === 2) return rounds11(first, (pos) => keyNoise11(k, pos));
      const out = []; let prev = first;
      for (let i = 0; i < N11; i++) { const c = pick11(logits11(prev, run, P11 + i), S11.T, keyNoise11(k, P11 + i)); out.push(c); prev = c; }
      return out;
    }
    function genSeq11(run) {
      const R = V.rng(S11.seed ? 42 : 7);
      const draw = () => ID11.map(() => { let u = R(); while (u <= 1e-12) u = R(); return -Math.log(-Math.log(u)); });
      const first = PR11[S11.prompt].row;
      if (run === 2) return rounds11(first, () => draw());
      const out = []; let prev = first;
      for (let i = 0; i < N11; i++) {
        let mine = null;
        if (run === 1) { for (let s = 0; s < 4; s++) { const gg = draw(); if (s === 2) mine = gg; } } else mine = draw();
        const c = pick11(logits11(prev, run, P11 + i), S11.T, mine); out.push(c); prev = c;
      }
      return out;
    }
    const c11 = UI.controls(f11);
    let r11 = null;
    function compute11() {
      const k = S11.seed ? 42n : PR11[S11.prompt].key;
      const seq = [0, 1, 2].map(genSeq11), key = [0, 1, 2].map((r) => genKeyed11(r, k));
      const steps = []; let prev = PR11[S11.prompt].row;
      for (let i = 0; i < N11; i++) {
        const pos = P11 + i, u = ID11.map((id) => keyedU(k, pos, id));
        steps.push({ pos, prev, z: Z11[prev], u, g: u.map((x) => -Math.log(-Math.log(x))), w: key[0][i] });
        prev = key[0][i];
      }
      S11.res = { seq, key, k, steps };
      if (r11) r11.set(S11.seed ? '42 (seed)' : '0x' + k.toString(16).padStart(16, '0'));
    }
    UI.seg(c11, { label: 'Prompt', options: [{ v: 'A', label: 'A' }, { v: 'B', label: 'B' }], value: 'A', onChange: (v) => { S11.prompt = v; compute11(); } });
    UI.toggle(c11, { label: 'seed = 42', value: false, onChange: (v) => { S11.seed = v; compute11(); } });
    UI.slider(c11, { label: 'Temperature', min: 0, max: 1.5, step: 0.05, value: 1, fmt: (v) => (v === 0 ? '0 greedy' : num(v, 2)), onInput: (v) => { S11.T = v; compute11(); } });
    UI.toggle(c11, { label: 'batch-invariant logits', value: true, onChange: (v) => { S11.inv = v; compute11(); } });
    r11 = UI.readout(c11, 'key k', '—');
    compute11();
    function chips11(g, toks, x, y, maxW, o) {
      const C = V.C();
      let sz = 11.5;
      const widths = (s) => toks.map((ti) => measure(g, vis(TOK11[ti]), s, 'mono', 500) + 12);
      let ws = widths(sz);
      const gap = 4;
      while (sz > 9.5 && ws.reduce((a, b) => a + b, 0) + gap * (toks.length - 1) > maxW) { sz -= 0.5; ws = widths(sz); }
      let cx = x;
      toks.forEach((ti, i) => {
        const cur = i === o.cursor, bad = i === o.diff;
        if (cur) V.glow(g, cx + ws[i] / 2, y + 11, ws[i] * 0.6, C.c, 0.35);
        V.box(g, cx, y, ws[i], 22, { r: 6, fill: V.alpha(bad ? C.red : o.col, bad ? 0.2 : 0.13), stroke: bad ? C.red : cur ? C.c : V.alpha(o.col, 0.55), lw: cur || bad ? 2 : 1 });
        V.text(g, vis(TOK11[ti]), cx + ws[i] / 2, y + 11.5, { size: sz, font: 'mono', weight: 500, align: 'center', base: 'middle', color: C.ink });
        cx += ws[i] + gap;
      });
      return cx;
    }
    V.fig(f11, (g, w, h, t, dt) => {
      const C = V.C();
      const narrow = w < 600;
      const R = S11.res;
      const dur = 2.6;
      if (!LA.paused) { S11.age += dt; if (S11.age >= dur) { S11.age = 0; S11.step = (S11.step + 1) % N11; } }
      const st = R.steps[S11.step];
      const pad = 14;
      let y = 6;
      V.text(g, clip(g, `prompt ${S11.prompt}: “${PR11[S11.prompt].label}” → ${N11} generated tokens`, w - 2 * pad, 11.5, 'mono'), pad, y + 10, { size: 11.5, font: 'mono', color: C.muted });
      y += 22;
      const lanes = [
        { name: narrow ? 'Sequential generator (stateful)' : 'Sequential generator (stateful): the next slice of a single stream', col: C.d, data: R.seq },
        { name: narrow ? 'Keyed: u = H(k, t, v)' : 'Keyed (stateless): u = H(k, t, v), looked up', col: C.a, data: R.key },
      ];
      const labW = narrow ? 0 : 158;
      lanes.forEach((L) => {
        V.text(g, clip(g, L.name, w - 2 * pad, 12.5, 'sans', 600), pad, y + 8, { size: 12.5, weight: 600, color: L.col });
        y += 18;
        L.data.forEach((toks, r) => {
          const diff = toks.findIndex((v, i) => v !== L.data[0][i]);
          let cy = y;
          if (narrow) { V.text(g, RUN11[r], pad, y + 9, { size: 11, color: C.muted }); cy = y + 15; }
          else V.text(g, RUN11[r], pad, y + 11.5, { size: 11.5, color: C.muted, base: 'middle' });
          const x0 = pad + labW, maxW = w - x0 - pad - 30;
          const xe = chips11(g, toks, x0, cy, maxW, { col: L.col, cursor: S11.step, diff: r > 0 ? diff : -1 });
          const vx = Math.min(xe + 8, w - pad - 18);
          if (r === 0) V.text(g, 'ref.', vx, cy + 11.5, { size: 10.5, font: 'mono', base: 'middle', color: C.muted });
          else V.text(g, diff < 0 ? '✓' : '✗', vx + 4, cy + 11.5, { size: 15, weight: 700, base: 'middle', color: diff < 0 ? C.e : C.red });
          y += narrow ? 46 : 27;
        });
        y += 6;
      });
      // the machine for the highlighted step
      const my = y + 2, mh = h - my - 6;
      V.box(g, pad, my, w - 2 * pad, mh, { r: 10, fill: V.alpha(C.muted, 0.05), stroke: V.alpha(C.muted, 0.25), lw: 1 });
      const prevLab = st.prev >= 9 ? 'a' : vis(TOK11[st.prev]);
      const head11 = narrow ? `${S11.step + 1}/${N11} · t = ${st.pos} · after “${prevLab}”` : `step ${S11.step + 1} of ${N11} · position t = ${st.pos} · after “${prevLab}”`;
      V.text(g, clip(g, head11, w - 2 * pad - 110, 11.5, 'mono'), pad + 10, my + 16, { size: 11.5, font: 'mono', color: C.ink });
      star(g, w - pad - 86, my + 12, 5.5, C.c);
      V.text(g, '= argmax', w - pad - 76, my + 16, { size: 11, font: 'mono', weight: 600, color: C.c });
      const scanEnd = 0.7 * dur;
      const j = Math.min(ID11.length - 1, Math.floor((S11.age / scanEnd) * ID11.length));
      const done = S11.age >= scanEnd;
      const greedy = S11.T <= 0;
      const kTxt = S11.seed ? 'k = 42 (seed)' : 'k = 0x' + R.k.toString(16).padStart(16, '0');
      let bx0, by0, bw, bh;
      if (!narrow) {
        // vertical pipeline on the left
        const pw = Math.min(236, w * 0.33), px = pad + 10;
        let py = my + 30;
        const row = (txt, col, hh = 24) => { V.box(g, px, py, pw, hh, { r: 7, fill: V.alpha(col, 0.13), stroke: V.alpha(col, 0.6), lw: 1.2 }); V.text(g, clip(g, txt, pw - 12, 11, 'mono'), px + pw / 2, py + hh / 2 + 0.5, { size: 11, font: 'mono', align: 'center', base: 'middle', color: C.ink }); py += hh; };
        const arrow = () => { V.arrow(g, px + pw / 2, py + 2, px + pw / 2, py + 14, { color: C.muted, lw: 1.4, head: 6 }); py += 16; };
        row(kTxt, C.b);
        py += 4;
        row(greedy ? 't = ' + st.pos + ' · (no noise)' : `t = ${st.pos} · v = ${ID11[j]} (“${vis(TOK11[j])}”)`, C.b);
        arrow();
        const spin = greedy ? '' : ' ' + '◐◓◑◒'[Math.floor(t * 8) % 4];
        row('splitmix64 × 3' + spin, C.f);
        arrow();
        row(greedy ? 'T = 0: u is not used' : `u = ${num(st.u[j], 4)}`, C.c);
        arrow();
        row(greedy ? 'the largest z wins' : `g = −log(−log u) = ${st.g[j] >= 0 ? '+' : ''}${num(st.g[j], 2)}`, C.c);
        wrapLines(g, 'the same input always gives the same u', pw, 10.5).slice(0, 2).forEach((l, li) => V.text(g, l, px, Math.min(py + 18, my + mh - 22) + li * 13, { size: 10.5, color: C.muted }));
        bx0 = px + pw + 22; by0 = my + 30; bw = w - pad - 10 - bx0; bh = mh - 40;
      } else {
        const px = pad + 10, pw = w - 2 * pad - 20;
        let py = my + 26;
        const line = (txt, col) => { V.box(g, px, py, pw, 22, { r: 6, fill: V.alpha(col, 0.12), stroke: V.alpha(col, 0.5), lw: 1 }); V.text(g, clip(g, txt, pw - 10, 11, 'mono'), px + 6, py + 11.5, { size: 11, font: 'mono', base: 'middle', color: C.ink }); py += 26; };
        line(kTxt, C.b);
        line(greedy ? `t = ${st.pos} · T = 0: argmax, no noise` : `H(k, t=${st.pos}, v=${ID11[j]}) → u = ${num(st.u[j], 3)}`, C.f);
        line(greedy ? 'the largest z wins' : `g = −log(−log u) = ${st.g[j] >= 0 ? '+' : ''}${num(st.g[j], 2)}`, C.c);
        bx0 = px; by0 = py + 4; bw = pw; bh = my + mh - by0 - 8;
      }
      // bars: z/T (flag) + g (pink stretch) for each token
      const n = ID11.length, rh = Math.min(26, bh / n);
      const labWb = narrow ? 58 : 66, valW = narrow ? 0 : 118;
      const base = st.z.map((z) => (greedy ? z : z / S11.T));
      const score = base.map((b, i) => (greedy ? b : b + st.g[i]));
      const lo = Math.min(...base, ...score) - 0.3, hi = Math.max(...base, ...score) + 0.3 + (Math.max(...base, ...score) - Math.min(...base, ...score) + 0.6) * 0.06;
      const ax0 = bx0 + labWb, ax1 = bx0 + bw - valW - 6;
      const X = (v) => ax0 + ((v - lo) / (hi - lo)) * (ax1 - ax0);
      if (lo < 0 && hi > 0) V.line(g, X(0), by0, X(0), by0 + n * rh, { color: V.alpha(C.muted, 0.3), lw: 1, dash: [3, 3] });
      for (let i = 0; i < n; i++) {
        const yy = by0 + i * rh + rh / 2;
        const seen = greedy || done || i <= j;
        const isW = (done || greedy) && i === st.w;
        const dim = st.z[i] <= -4;
        V.text(g, vis(TOK11[i]), bx0 + labWb - 8, yy, { size: 11.5, font: 'mono', align: 'right', base: 'middle', color: isW ? C.c : dim ? C.muted : C.ink });
        V.line(g, ax0, yy, ax1, yy, { color: V.alpha(C.muted, 0.18), lw: 1 });
        const fx = X(base[i]);
        V.line(g, fx, yy - rh * 0.32, fx, yy + rh * 0.2, { color: V.alpha(C.d, dim ? 0.5 : 1), lw: 1.6 });
        g.beginPath(); g.moveTo(fx, yy - rh * 0.32); g.lineTo(fx + 7, yy - rh * 0.26); g.lineTo(fx, yy - rh * 0.18); g.closePath(); g.fillStyle = C.d; g.fill();
        if (seen && !greedy) {
          V.line(g, fx, yy + 1, X(score[i]), yy + 1, { color: V.alpha(C.c, i === j && !done ? 1 : 0.75), lw: 4 });
          V.dot(g, X(score[i]), yy + 1, isW ? 6 : 4, isW ? C.c : C.ink);
        } else if (greedy) V.dot(g, fx, yy + 1, isW ? 6 : 4, isW ? C.c : C.ink);
        if (isW) { V.glow(g, X(score[i]), yy, 16, C.c); star(g, X(score[i]) + 13, yy + 1, 6, C.c); }
        if (!narrow && seen && !greedy) V.text(g, `u ${num(st.u[i], 3)}  g ${st.g[i] >= 0 ? '+' : ''}${num(st.g[i], 2)}`, ax1 + 16, yy, { size: 10.5, font: 'mono', weight: isW ? 600 : 400, base: 'middle', color: isW ? C.c : i === j && !done ? C.ink : C.muted });
        if (i === j && !done && !greedy) { V.rr(g, bx0 + 2, yy - rh / 2 + 1, bw - 4, rh - 2, 6); g.strokeStyle = V.alpha(C.f, 0.7); g.lineWidth = 1.2; g.stroke(); }
      }
    });

    /* ================= Figure 12: two draws, one die ================= */
    const f12 = $('#fig-acoplamento');
    fitH(f12, 470, 700);
    const TOK12 = [' moon', ' sun', ' sea', ' sky', ' river', ' salt'];
    const ZP12 = [2.2, 1.7, 1.2, 0.5, 0.0, -0.6];
    const D12 = [-0.5, 0.6, 0.3, -0.4, 0.5, -0.3]; // direction of the drafter's error (illustrative)
    const NR12 = 3000;
    const gum12 = (R) => { let u = R(); while (u <= 1e-12) u = R(); return -Math.log(-Math.log(u)); };
    const GK12 = [], GO12 = [];
    { const Ra = V.rng(4242), Rb = V.rng(777); for (let r = 0; r < NR12; r++) { GK12.push(ZP12.map(() => gum12(Ra))); GO12.push(ZP12.map(() => gum12(Rb))); } }
    const MODES12 = [
      { v: 'ign', label: 'ignores the noise', short: 'ignores' },
      { v: 'own', label: 'own noise', short: 'own' },
      { v: 's07', label: 'target noise ×0.7', short: 'target ×0.7' },
      { v: 's1', label: 'target noise ×1', short: 'target ×1' },
    ];
    const S12 = { mode: 's07', delta: 0.3, T: 1, speed: 'lento', r: 9, age: 1.3, hist: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], stats: null };
    const am12 = (a) => { let b = 0; for (let i = 1; i < a.length; i++) if (a[i] > a[b]) b = i; return b; };
    const zq12 = () => ZP12.map((v, i) => v + S12.delta * D12[i]);
    const noise12 = (mode, r) => (mode === 'ign' ? ZP12.map(() => 0) : mode === 'own' ? GO12[r] : GK12[r].map((x) => x * (mode === 's07' ? 0.7 : 1)));
    const draft12 = (mode, zq, r, T) => (T <= 0 || mode === 'ign' ? am12(zq) : am12(zq.map((v, i) => v / T + noise12(mode, r)[i])));
    const target12 = (r, T) => (T <= 0 ? am12(ZP12) : am12(ZP12.map((v, i) => v / T + GK12[r][i])));
    const c12 = UI.controls(f12);
    let r12a = null, r12b = null;
    function stats12() {
      const T = S12.T, zq = zq12();
      const p = softmax(ZP12, T), q = softmax(zq, T);
      const acc = {};
      MODES12.forEach((m) => { acc[m.v] = 0; });
      for (let r = 0; r < NR12; r++) { const x = target12(r, T); MODES12.forEach((m) => { if (draft12(m.v, zq, r, T) === x) acc[m.v]++; }); }
      MODES12.forEach((m) => { acc[m.v] /= NR12; });
      S12.stats = { p, q, zq, acc, min: p.reduce((s, v, i) => s + Math.min(v, q[i]), 0), pq: p.reduce((s, v, i) => s + v * q[i], 0) };
      if (r12a) { r12a.set(pct(acc[S12.mode], 0)); r12b.set(pct(S12.stats.min, 0)); }
    }
    UI.seg(c12, { label: 'Drafter', options: MODES12.map((m) => ({ v: m.v, label: m.label })), value: 's07', onChange: (v) => { S12.mode = v; stats12(); } });
    UI.slider(c12, { label: 'Drafter error δ', min: 0, max: 1.5, step: 0.05, value: 0.3, fmt: (v) => num(v, 2), onInput: (v) => { S12.delta = v; stats12(); } });
    UI.slider(c12, { label: 'Temperature', min: 0, max: 1.5, step: 0.05, value: 1, fmt: (v) => (v === 0 ? '0 greedy' : num(v, 2)), onInput: (v) => { S12.T = v; stats12(); } });
    UI.seg(c12, { label: 'Pace', options: [{ v: 'lento', label: '1 round' }, { v: 'rapido', label: 'fast' }], value: 'lento', onChange: (v) => { S12.speed = v; } });
    r12a = UI.readout(c12, 'agreement', '—');
    r12b = UI.readout(c12, 'ceiling Σ min(p,q)', '—');
    stats12();
    V.fig(f12, (g, w, h, t, dt) => {
      const C = V.C();
      const narrow = w < 640;
      const ST = S12.stats, T = S12.T, zq = ST.zq;
      const dur = S12.speed === 'lento' ? 2.4 : 0.22;
      if (!LA.paused) {
        S12.age += dt;
        if (S12.age >= dur) { S12.age = 0; S12.r = (S12.r + 1) % NR12; S12.hist.push(S12.r); if (S12.hist.length > 12) S12.hist.shift(); }
      }
      const r = S12.r, ph = S12.speed === 'lento' ? V.clamp(S12.age / dur, 0, 1) : 1;
      const x = target12(r, T), d = draft12(S12.mode, zq, r, T);
      const nz = noise12(S12.mode, r);
      const pad = 14;
      const lw = narrow ? w - 2 * pad : w * 0.58 - pad;
      const modeLab = MODES12.find((m) => m.v === S12.mode).label;
      V.text(g, clip(g, `round ${fmtInt(r + 1)} · drafter: ${modeLab}`, lw, 11.5, 'mono'), pad, 16, { size: 11.5, font: 'mono', color: C.muted });
      // lanes
      const ry0 = 30, rH = narrow ? 210 : h - 170;
      const n = ZP12.length, lh = rH / n, labW = 44;
      const lo = (T > 0 ? Math.min(...ZP12, ...zq) / T : Math.min(...ZP12, ...zq)) - (T > 0 ? 2.4 : 0.5);
      const hi = (T > 0 ? Math.max(...ZP12, ...zq) / T : Math.max(...ZP12, ...zq)) + (T > 0 ? 4.6 : 0.5);
      const ax0 = pad + labW, ax1 = pad + lw - 8;
      const X = (v) => ax0 + V.clamp((v - lo) / (hi - lo), 0, 1) * (ax1 - ax0);
      const p1 = V.clamp(ph / 0.22, 0, 1), p2 = V.clamp((ph - 0.22) / 0.26, 0, 1), show = ph >= 0.5;
      for (let i = 0; i < n; i++) {
        const yy = ry0 + i * lh + lh / 2;
        V.text(g, vis(TOK12[i]), ax0 - 8, yy, { size: 11.5, font: 'mono', align: 'right', base: 'middle', color: C.ink });
        V.line(g, ax0, yy - 5, ax1, yy - 5, { color: V.alpha(C.muted, 0.15), lw: 1 });
        V.line(g, ax0, yy + 5, ax1, yy + 5, { color: V.alpha(C.muted, 0.15), lw: 1 });
        const bp = T > 0 ? ZP12[i] / T : ZP12[i], bq = T > 0 ? zq[i] / T : zq[i];
        const sp = T > 0 ? bp + GK12[r][i] : bp, sq = T > 0 ? bq + nz[i] : bq;
        const xp = V.lerp(X(lo), X(bp), V.ease.out(p1)), xq = V.lerp(X(lo), X(bq), V.ease.out(p1));
        V.line(g, X(lo), yy - 5, xp, yy - 5, { color: V.alpha(C.a, 0.75), lw: 5 });
        V.line(g, X(lo), yy + 5, xq, yy + 5, { color: V.alpha(C.b, 0.75), lw: 5 });
        if (p1 >= 1 && T > 0) {
          const ep = V.lerp(X(bp), X(sp), V.ease.out(p2)), eq = V.lerp(X(bq), X(sq), V.ease.out(p2));
          V.line(g, X(bp), yy - 5, ep, yy - 5, { color: C.c, lw: 3 });
          if (S12.mode !== 'ign') V.line(g, X(bq), yy + 5, eq, yy + 5, { color: S12.mode === 'own' ? C.d : V.alpha(C.c, S12.mode === 's07' ? 0.6 : 1), lw: 3 });
        }
        if (show) {
          if (i === x) { V.glow(g, X(sp), yy - 5, 12, C.a, 0.5); star(g, X(sp), yy - 5, 6.5, C.a); }
          if (i === d) { g.beginPath(); const cx = X(sq), cy = yy + 5; g.moveTo(cx, cy - 6); g.lineTo(cx + 5, cy); g.lineTo(cx, cy + 6); g.lineTo(cx - 5, cy); g.closePath(); g.fillStyle = C.b; g.fill(); }
        }
      }
      // verdict
      let vy = ry0 + rH + 8;
      if (show) {
        const ok = x === d;
        pill(g, ok ? `✓ accepted: “${vis(TOK12[d])}” = the target's sample` : `✗ rejected: the target emits “${vis(TOK12[x])}”`, pad, vy, { size: 11.5, font: 'sans', weight: 600, color: ok ? C.e : C.red, fill: V.alpha(ok ? C.e : C.red, 0.12), stroke: V.alpha(ok ? C.e : C.red, 0.5) });
      }
      // tape: what the target emits × what the drafter proposed
      vy += 34;
      const tapeLab = narrow ? 64 : 76;
      [['emitted', 0], ['draft', 1]].forEach(([lab, k]) => {
        const ty = vy + k * 28;
        V.text(g, lab, pad, ty + 11, { size: 11, color: C.muted, base: 'middle' });
        let cx = pad + tapeLab;
        const shown = S12.hist.slice(-(narrow ? 6 : 8));
        shown.forEach((rr) => {
          const xx = target12(rr, T), dd = draft12(S12.mode, zq, rr, T);
          const tok = k === 0 ? xx : dd, ok = xx === dd;
          const s = vis(TOK12[tok]), cw = measure(g, s, 11, 'mono', 500) + 12;
          if (cx + cw > pad + lw) return;
          const col = k === 0 ? C.a : ok ? C.e : C.red;
          V.box(g, cx, ty, cw, 22, { r: 6, fill: V.alpha(col, 0.14), stroke: V.alpha(col, 0.6), lw: 1 });
          V.text(g, s, cx + cw / 2, ty + 11.5, { size: 11, font: 'mono', weight: 500, align: 'center', base: 'middle', color: C.ink });
          cx += cw + 4;
        });
      });
      // agreement chart
      let gx, gy, gw, gh;
      if (narrow) { gx = pad; gy = vy + 96; gw = w - 2 * pad; gh = 170; }
      else { gx = w * 0.6 + 6; gy = 34; gw = w - gx - pad; gh = h * 0.56; }
      V.text(g, clip(g, `agreement over ${fmtInt(NR12)} rounds`, gw, 11, 'mono'), gx, gy - 12, { size: 11, font: 'mono', color: C.muted });
      const cy0 = gy + 4, cyb = gy + gh - 34;
      const Y = (v) => cyb - v * (cyb - cy0);
      [0, 0.5, 1].forEach((v) => { V.line(g, gx + 30, Y(v), gx + gw, Y(v), { color: V.alpha(C.muted, 0.14), lw: 1 }); V.text(g, pct(v, 0), gx + 26, Y(v), { size: 10.5, font: 'mono', align: 'right', base: 'middle', color: C.muted }); });
      const lblW = 58, bw = (gw - 34 - lblW) / MODES12.length;
      MODES12.forEach((m, k) => {
        const v = ST.acc[m.v], bx = gx + 34 + k * bw + 6, bwi = bw - 12, cur = m.v === S12.mode;
        V.box(g, bx, Y(v), bwi, Math.max(1, cyb - Y(v)), { r: 4, fill: V.alpha(cur ? C.c : C.b, cur ? 0.8 : 0.45) });
        const near = Math.abs(Y(v) - 8 - Y(ST.min)) < 11 || Math.abs(Y(v) - 8 - Y(ST.pq)) < 11;
        if (near && cyb - Y(v) > 24) V.text(g, pct(v, 0), bx + bwi / 2, Y(v) + 14, { size: 10.5, font: 'mono', weight: 600, align: 'center', color: C.ink });
        else V.text(g, pct(v, 0), bx + bwi / 2, Y(v) - 8, { size: 10.5, font: 'mono', align: 'center', color: C.ink });
        const L = wrapLines(g, m.short, bwi + 10, 10.5);
        L.forEach((l, li) => V.text(g, l, bx + bwi / 2, cyb + 13 + li * 12, { size: 10.5, align: 'center', color: cur ? C.c : C.muted, weight: cur ? 600 : 400 }));
      });
      V.line(g, gx + 30, Y(ST.min), gx + gw - lblW + 4, Y(ST.min), { color: C.d, lw: 1.6, dash: [6, 4] });
      V.line(g, gx + 30, Y(ST.pq), gx + gw - lblW + 4, Y(ST.pq), { color: C.muted, lw: 1.4, dash: [2, 3] });
      let yMin = Y(ST.min), yPq = Y(ST.pq);
      if (yPq - yMin < 28) { const mid = Math.max((yMin + yPq) / 2, cy0 + 12); yMin = mid - 14; yPq = mid + 14; }
      const lx = gx + gw - lblW + 10;
      V.text(g, 'ceiling', lx, yMin - 1, { size: 10.5, font: 'mono', color: C.d });
      V.text(g, pct(ST.min, 0), lx, yMin + 11, { size: 10.5, font: 'mono', weight: 600, color: C.d });
      V.text(g, 'Σ p·q', lx, yPq - 1, { size: 10.5, font: 'mono', color: C.muted });
      V.text(g, pct(ST.pq, 0), lx, yPq + 11, { size: 10.5, font: 'mono', color: C.muted });
      // p and q
      const qy = gy + gh + 4, qh = narrow ? 92 : h - qy - 24;
      if (qh > 50) {
        V.text(g, clip(g, 'target p (mint) × q (outline)', gw, 10.5, 'mono'), gx, qy + 8, { size: 10.5, font: 'mono', color: C.muted });
        const mx = Math.max(...ST.p, ...ST.q) * 1.1, qb = qy + qh, qt = qy + 18;
        const sw = gw / n;
        for (let i = 0; i < n; i++) {
          const bx = gx + i * sw + 5, bwi = sw - 10;
          const hp = (ST.p[i] / mx) * (qb - qt - 14), hq = (ST.q[i] / mx) * (qb - qt - 14);
          V.box(g, bx, qb - 14 - hp, bwi, Math.max(1, hp), { r: 3, fill: V.alpha(C.a, 0.6) });
          V.rr(g, bx + 2, qb - 14 - hq, bwi - 4, Math.max(1, hq), 3); g.strokeStyle = C.b; g.lineWidth = 1.6; g.stroke();
          V.text(g, clip(g, vis(TOK12[i]), sw - 2, 10.5, 'mono'), bx + bwi / 2, qb - 2, { size: 10.5, font: 'mono', align: 'center', color: C.muted });
        }
      }
    });
  },
});

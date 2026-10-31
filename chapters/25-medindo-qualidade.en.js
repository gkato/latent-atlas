LA.chapter({
  id: 'medindo-qualidade',
  part: 4,
  order: 25,
  title: 'Measuring the damage: perplexity, KLD and numerical parity',
  navTitle: 'Measuring quality',
  subtitle: 'Why “looks good” is not enough: how to measure, rigorously and against the system’s own noise, whether a quantization, a kernel or a patch made the model dumber.',
  minutes: 60,
  recipes: ['glm', 'tf', 'qwen', 'ds'],
  tags: ['KLD', 'perplexity', 'NLL', 'top-1', 'numerical parity', 'determinism', 'RULER', 'tool calling', 'canary'],
  goals: [
    'Compute and interpret <b>NLL</b>, <b>perplexity</b>, <b>KLD</b> and <b>top-1 agreement</b> in teacher forcing, knowing what each one hides',
    'Build a <b>calibrated panel</b> against stock-vs-stock noise and read NOT-WORSE, EQUIV and INCONCLUSIVE verdicts',
    'Explain why kernels diverge (<b>non-associativity</b>), how <b>numerical parity</b> is tested, what determinism costs and what changes when the engine guarantees <b>exactness</b>',
    'Choose and use <b>functional tests</b> (needles, RULER, garble, loops, tools, canaries) as a recipe’s gates',
    'Document results reproducibly and decide on the quality × cost <b>Pareto frontier</b>',
  ],
  prereqs: ['quantizacao-fundamentos', 'hadamard-trelica-exl3'],
  summary: [
    'Numerical damage is <b>silent</b>: you measure it on an evidence ladder — weights (rel RMSE, cosine) → kernels (parity) → distributions (NLL, KLD, top-1) → tasks and long context → service (canary, tools).',
    'NLL = mean of \\(-\\ln p\\) of the true token in teacher forcing; PPL = \\(e^{\\text{NLL}}\\). The mean hides errors that cancel out; <b>KLD</b> compares whole distributions (≥ 0, asymmetric, in nats), and the API’s top-k version is a proxy that typically underestimates when the top-k covers almost all of the mass (and can be off in either direction when it does not).',
    'GLM serves EXL3 4 bpw because the independent panel’s KLD came out at 0.024555 nats — ≈ the official FP8 on the same stack, with 54% of the bytes — versus 0.060535 for NVFP4.',
    'Absolute gates fail because <b>the stock itself varies</b> (near-ties flip). The calibrated panel classifies positions (CONF/TIE/UNSTABLE) and standardizes the KL by the stock-vs-stock null (2× gate), but it is a screen: with 2 captures per arm, it flagged its own control. Adoption verdicts (NOT-WORSE) come from <code>kl_panel</code>, with positive controls and frozen protocols.',
    'Floating-point addition <b>is not associative</b>: variable orders (split-K, top-k, MoE finalize) change bits and flip near-ties. Qwen’s determinism patch brings the spread down to zero (0.25/6.07 → 0/0 nats) at ~−4%. GLM’s TensorFold version makes exactness a contract (drafted = serial, concurrent = alone): the null becomes zero and “nothing changed” becomes a <b>hash test</b>, but only for changes that preserve the bits; the lossy defaults (q4 dense weights, FP8 KV) still need a judge.',
    'Long context needs its own tests, cold and with a known length: needles, RULER-lite (verified by <code>/tokenize</code>, task at the head and haystack appended; 8/8 gate), the garble sweep (estimated at 1.56 tok/word; CLEAN up to ~900K) and the 8-gram-novelty loop detector.',
    'At the top, functional tests catch what KLD cannot see: the boot canary (OK at T=0 + acceptance over ≥ 64 drafts), tool batteries (7/7, 8/8) and the 145-case gate. Task benchmarks have coarse resolution: the TensorFold version’s q4 tied on GSM8K (250) and HumanEval (164), and 48 replies to short French prompts caught the lost end of turn (12/48 against 1/48). Document date, commits, image, protocol and Measured/Interpretation labels.',
  ],
  quiz: [
    { q: 'Two candidates have the same NLL as the reference (1.331 nats), but candidate B has a mean KLD of 0.03 nats and A, of 0.002. What does this indicate?', o: ['Nothing: if the NLL is the same, the quality is the same', 'B distorted the distributions; in the NLL mean, losses and gains canceled out', 'B’s KLD is wrong, because KLD cannot be larger than the NLL difference', 'B is better, because it has more variety'], a: 1, e: 'NLL only looks at the probability of the true token and averages it; positions that improve offset positions that get worse. KLD compares the whole distribution at each position, and every aggregated term is ≥ 0, so offsets cannot bring it to zero.' },
    { q: 'At a confident position (the top-20 covers almost all of the mass), why does the “KL proxy” computed over the lists the API returns tend to <i>underestimate</i> the true KL?', o: ['Because it uses a base-2 logarithm', 'Because it ignores the probability mass outside the top-k lists and renormalizes only the common support', 'Because vLLM rounds the logprobs', 'Because KL is symmetric'], a: 1, e: 'By the chain rule, \\(D(P\\|Q) = D_{\\text{bin}} + P(S)\\,D_S + P(\\bar S)\\,D_{\\bar S} \\ge P(S)\\,D_S\\): with \\(P(S)\\approx 1\\), the true value is at least the proxy. When the top-k covers little mass, the proxy can even exceed the true value. It is good for comparing arms measured the same way, not against the KLD from another script.' },
    { q: 'In GLM’s sealed thin-decode study, the candidate had 0 flips and KL within the null, but the verdict was INCONCLUSIVE. Why?', o: ['Because the candidate was slower', 'Because there were hidden flips', 'Because the absolute gate “KL ≤ 0.01” failed the stock-versus-stock control itself (null of 0.0142–0.0146); the defect was in the ruler', 'Because GSM8K was not run'], a: 2, e: 'When the negative control (stock vs stock) fails a gate, the instrument is not fit to judge. There was also a second control failure (a bimodal stock position blew the τ transfer). The honest record is “inconclusive with clean relative evidence”, not “passed”.' },
    { q: 'Why does sorting the top-k indices of the QSA indexer (<code>VLLM_QSA_DET_TOPK=1</code>) make the output reproducible?', o: ['Because it changes which blocks are selected', 'Because the same selection is now always visited in the same order, and floating-point addition depends on the order', 'Because it turns off sparse attention', 'Because it fixes the sampler seed'], a: 1, e: 'The selection does not change; only the visiting order does. Since \\((a+b)+c \\ne a+(b+c)\\) in floating point, fixed order ⇒ fixed bits ⇒ near-ties do not flip. Together with the MoE’s unfused finalize, the spread went from 0.25/6.07 to 0/0 nats, at ~−4%.' },
    { q: 'A reasoning trace ends with <code>finish_reason=length</code>. The detector shows 8-gram novelty dropping below 2% around token 12K and staying below it until the end. What should you do?', o: ['Raise max_tokens: it is a long tail', 'Nothing: it is transient repetition', 'Treat it as a loop: raising max_tokens will not help; investigate sampling, numerics or the prompt', 'Turn off prefix caching'], a: 2, e: 'Loop = a collapse in novelty that never recovers (3 windows below the threshold and persistence until the end). In a long tail, novelty stays high; in a transient repetition, it rises again.' },
    { q: 'Why does GLM’s canary require at least 64 drafted tokens (<code>GLM53_WARMUP_CANARY_MIN_DRAFTS</code>) before judging acceptance?', o: ['Because DFlash only works after 64 tokens', 'Because, with a small sample, even a healthy engine may have accepted zero by bad luck and the canary would raise a false alarm', 'Because /metrics only updates every 64 tokens', 'Because 64 is the KV block size'], a: 1, e: 'If each verification step accepts something with probability \\(a\\), getting zero in \\(n\\) steps has probability \\((1-a)^n\\), which plummets as \\(n\\) grows. A degenerate engine (\\(a\\approx 0\\)), on the other hand, always gets zero. With little traffic, the script says “cannot judge” instead of “degenerate”.' },
  ],
  mount(root, LA) {
    const V = LA.viz, UI = LA.ui, N = LA.num;

    /* ================= local utilities ================= */
    const FC = {};
    const setF = (g, size, font, weight) => { const k = size + '|' + (font || 'sans') + '|' + (weight || 400); g.font = FC[k] || (FC[k] = V.font(size, font || 'sans', weight || 400)); };
    const tw = (g, s, size = 12, font = 'sans', weight = 400) => { setF(g, size, font, weight); return g.measureText(String(s)).width; };
    const clip = (g, s, maxW, size = 11, font = 'sans', weight = 400) => {
      s = String(s); if (maxW <= 6) return '';
      setF(g, size, font, weight);
      if (g.measureText(s).width <= maxW) return s;
      let lo = 0, hi = s.length;
      while (lo < hi) { const m = (lo + hi + 1) >> 1; if (g.measureText(s.slice(0, m) + '…').width <= maxW) lo = m; else hi = m - 1; }
      return s.slice(0, lo) + '…';
    };
    const wrapLines = (g, s, maxW, size = 12, font = 'sans', weight = 400) => {
      setF(g, size, font, weight);
      const out = []; let cur = '';
      String(s).split(/\s+/).forEach((wd) => { const t2 = cur ? cur + ' ' + wd : wd; if (cur && g.measureText(t2).width > maxW) { out.push(cur); cur = wd; } else cur = t2; });
      if (cur) out.push(cur);
      return out;
    };
    const drawWrap = (g, s, x, y, maxW, lh, o = {}) => { const L = wrapLines(g, s, maxW, o.size || 12, o.font || 'sans', o.weight || 400); L.forEach((l, i) => V.text(g, l, x, y + i * lh, o)); return L.length; };
    const fitStage = (fig, fn) => {
      const st = fig.querySelector('.la-fig-stage'); if (!st) return;
      const apply = () => { const w = st.clientWidth || 600; const hh = fn(w); if (Math.abs(st.getBoundingClientRect().height - hh) > 1) st.style.height = hh + 'px'; };
      apply();
      const ro = new ResizeObserver(() => apply()); ro.observe(st);
      if (LA.onCleanup) LA.onCleanup(() => ro.disconnect());
    };
    const f = (x, d = 2) => (Number.isFinite(x) ? x.toFixed(d) : '—');
    const pct = (x, d = 1) => f(100 * x, d) + '%';
    const thou = (n) => String(Math.round(n)).replace(/\B(?=(\d{3})+(?!\d))/g, ',');
    const softmax = (l) => { let m = -Infinity; for (const x of l) if (x > m) m = x; const e = l.map((x) => Math.exp(x - m)); let s = 0; for (const x of e) s += x; return e.map((x) => x / s); };
    const klDiv = (p, q) => { let s = 0; for (let i = 0; i < p.length; i++) if (p[i] > 0) s += p[i] * Math.log(p[i] / Math.max(q[i], 1e-300)); return Math.max(0, s); };
    const argmax = (a) => { let b = 0; for (let i = 1; i < a.length; i++) if (a[i] > a[b]) b = i; return b; };
    const showEl = (x, on) => { if (x && x.el) x.el.style.display = on ? '' : 'none'; else if (x && x.style) x.style.display = on ? '' : 'none'; };
    const tri = (g, x, y, s, col, dir = 1) => { g.fillStyle = col; g.beginPath(); g.moveTo(x, y); g.lineTo(x - s, y - dir * s * 1.4); g.lineTo(x + s, y - dir * s * 1.4); g.closePath(); g.fill(); };
    const badge = (g, s, x, y, col, o = {}) => {
      const sz = o.size || 10.5, fw = tw(g, s, sz, 'mono', 700) + 12, hh = sz + 8;
      const xx = o.align === 'right' ? x - fw : o.align === 'center' ? x - fw / 2 : x;
      V.box(g, xx, y - hh / 2, fw, hh, { r: hh / 2, fill: V.alpha(col, 0.16), stroke: V.alpha(col, 0.75), lw: 1.2 });
      V.text(g, s, xx + fw / 2, y + 0.5, { size: sz, font: 'mono', weight: 700, align: 'center', base: 'middle', color: col });
      return fw;
    };

    /* =========================================================
       F1 — Who goes off? The evidence ladder
       ========================================================= */
    (() => {
      const fig = root.querySelector('#fig-detectores'); if (!fig) return;
      fitStage(fig, (w) => (w < 560 ? 590 : 385));
      const stages = ['Weights', 'Kernel', 'Logits', 'Text', 'Service'];
      const subs = ['bytes', 'operations', 'distributions', 'answers', 'API and boot'];
      const dets = [
        { k: 'rmse', l: 'rel RMSE · cosine', s: 0, cad: 'per checkpoint' },
        { k: 'par', l: 'Kernel parity', s: 1, cad: 'per new kernel' },
        { k: 'kld', l: 'KLD · NLL · top-1', s: 2, cad: 'numerics changed' },
        { k: 'task', l: 'Tasks', s: 3, cad: 'GSM8K, reasoning' },
        { k: 'long', l: 'Long context', s: 3, cad: 'KV touched' },
        { k: 'loop', l: 'Long generations', s: 3, cad: 'loop detector' },
        { k: 'tool', l: 'Tools', s: 4, cad: 'battery, 145' },
        { k: 'canary', l: 'Boot canary', s: 4, cad: 'every boot' },
      ];
      const defects = [
        { v: 'q', label: 'Coarse weights (NVFP4 instead of EXL3)', o: 0, hit: { rmse: 1, par: 0, kld: 2, task: 1, long: 1, loop: 0, tool: 0, canary: 0 },
          note: 'The error is born in the bytes. rel RMSE shows that NVFP4 errs more, but both 4-bit formats sit above the FP8 gate (0.06): it is only a weak signal. What measures the size of the damage is KLD (0.0605 versus 0.0246 on GLM’s panel); tasks only notice if the difference is large. Kernel parity sees nothing: the kernel does the math perfectly with the wrong weights.' },
        { v: 'ord', label: 'New kernel that sums in another order', o: 1, hit: { rmse: 0, par: 3, kld: 3, task: 0, long: 0, loop: 0, tool: 0, canary: 0 },
          note: 'Not damage, just noise. Parity passes because the difference fits within 4× the stock-versus-stock wobble, and the calibrated panel gives a KL within the null. A poorly calibrated absolute gate would raise a false alarm right here.' },
        { v: 'kv', label: 'FP8 K in the KV changes the top-k selection', o: 2, hit: { rmse: 0, par: 0, kld: 1, task: 1, long: 2, loop: 0, tool: 0, canary: 0 },
          note: 'On short texts the distribution barely changes; the damage shows up at depth, when the sparse indexer picks different blocks. What catches it are needles, RULER and long reasoning — which is why reasoning_check was born together with Qwen’s FP8 KV.' },
        { v: 'deg', label: 'Degenerate engine at boot (#249)', o: 1, hit: { rmse: 0, par: 0, kld: 2, task: 2, long: 2, loop: 1, tool: 2, canary: 2 },
          note: 'All “!!!!”, acceptance ~0 and a green /health. Any test would catch it — if it ran. Only the canary runs on every boot, before READY, and that is why it is the detector that matters here.' },
        { v: 'dsml', label: 'Broken tool syntax', o: 4, hit: { rmse: 0, par: 0, kld: 0, task: 0, long: 0, loop: 0, tool: 2, canary: 0 },
          note: 'Truncation in the middle of a DSML call, or a malformed tag: the KLD of a prose text does not change at all. Only the tool battery and the 145-case gate exercise the parser, the finish_reason and the schema.' },
        { v: 'loop', label: 'Reasoning loop', o: 3, hit: { rmse: 0, par: 0, kld: 0, task: 1, long: 0, loop: 2, tool: 0, canary: 0 },
          note: 'Teacher forcing never lets the model follow its own steps, so KLD cannot see loops. They show up in long sampled generations; the 8-gram novelty detector tells a loop apart from a long tail.' },
        { v: 'eos', label: '4-bit dense weights lose the end of turn (GLM·TF, #18)', o: 0, hit: { rmse: 1, par: 0, kld: 1, task: 0, long: 0, loop: 2, tool: 0, canary: 0 },
          note: 'The error is born in the bytes (q4 errs more than fp8: a weak signal). GSM8K and HumanEval in English tied, and a KLD over a generic corpus would probably see little (the recipe’s v0.6.0 engine does not even return logprobs). The symptom lives in short French coding prompts: 12 of 48 replies run to max_tokens, against 1 of 48 with fp8 or bf16. Counting finish_reason=length on the right prompts catches it.' },
      ];
      const S = { d: defects[0], reset: true, t0: 0 };
      const ctl = UI.controls(fig);
      UI.select(ctl, { label: 'Defect', options: defects.map((d) => ({ v: d.v, label: d.label })), value: 'q', onChange: (v) => { S.d = defects.find((d) => d.v === v) || defects[0]; S.reset = true; } });
      const C0 = V.C();
      UI.legend(ctl, [[C0.red, 'alarm'], [C0.d, 'weak signal'], [C0.e, 'measures, passes'], [C0.muted, 'cannot see']]);
      const drawBug = (g, x, y, t, C) => {
        const wig = Math.sin(t * 14) * 2.2;
        g.strokeStyle = C.red; g.lineWidth = 1.6;
        for (let k = -1; k <= 1; k++) {
          const lx = x + k * 5, s = k === 0 ? wig : -wig;
          g.beginPath(); g.moveTo(lx, y - 3); g.lineTo(lx - 4 + s, y - 11); g.stroke();
          g.beginPath(); g.moveTo(lx, y + 3); g.lineTo(lx - 4 - s, y + 11); g.stroke();
        }
        g.fillStyle = C.red;
        g.beginPath(); g.ellipse(x, y, 11, 7, 0, 0, Math.PI * 2); g.fill();
        g.beginPath(); g.arc(x + 11, y, 4.5, 0, Math.PI * 2); g.fill();
        g.fillStyle = '#fff';
        g.beginPath(); g.arc(x + 12.4, y - 1.8, 1.3, 0, Math.PI * 2); g.fill();
        g.beginPath(); g.arc(x + 12.4, y + 1.8, 1.3, 0, Math.PI * 2); g.fill();
        g.strokeStyle = 'rgba(0,0,0,.25)'; g.lineWidth = 1; g.beginPath(); g.moveTo(x - 9, y); g.lineTo(x + 7, y); g.stroke();
      };
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        if (S.reset) { S.t0 = t; S.reset = false; }
        const d = S.d, narrow = w < 560, pad = 14;
        const n = stages.length, gap = narrow ? 5 : 16;
        const bw = (w - 2 * pad - gap * (n - 1)) / n, by = 12, bh = 46;
        const cx = (s) => pad + s * (bw + gap) + bw / 2;
        const el = t - S.t0, period = 6.5;
        const ph = (el % period) / period;
        const walk = V.ease.inOut(V.clamp(ph / 0.62, 0, 1));
        const pos = d.o + (n - 1 - d.o) * walk;
        for (let s = 0; s < n; s++) {
          const x = pad + s * (bw + gap);
          const hit = s >= d.o && pos >= s - 0.02;
          V.box(g, x, by, bw, bh, { r: 9, fill: hit ? V.alpha(C.red, 0.13) : V.alpha(C.b, 0.1), stroke: hit ? V.alpha(C.red, 0.7) : V.alpha(C.b, 0.45), lw: hit ? 1.8 : 1.2 });
          const fs = narrow ? 11 : 12.5;
          V.text(g, clip(g, stages[s], bw - 6, fs, 'sans', 600), x + bw / 2, by + 18, { size: fs, weight: 600, align: 'center', base: 'middle', color: C.ink });
          V.text(g, clip(g, subs[s], bw - 6, narrow ? 9 : 10, 'mono'), x + bw / 2, by + 34, { size: narrow ? 9 : 10, font: 'mono', align: 'center', base: 'middle', color: C.muted });
          if (!narrow && s < n - 1) V.arrow(g, x + bw + 2, by + bh / 2, x + bw + gap - 2, by + bh / 2, { color: V.alpha(C.muted, 0.6), lw: 1.2, head: 5 });
        }
        const ly = by + bh + 24;
        V.line(g, cx(0), ly, cx(n - 1), ly, { color: V.alpha(C.muted, 0.3), lw: 1, dash: [3, 5] });
        const bx = cx(0) + (cx(n - 1) - cx(0)) * (pos / (n - 1));
        V.line(g, cx(d.o), ly, bx, ly, { color: V.alpha(C.red, 0.5), lw: 3 });
        V.dot(g, cx(d.o), ly, 4, C.red);
        drawBug(g, bx, ly, t, C);
        // detector grid
        const cols = narrow ? 2 : 4, rows = Math.ceil(dets.length / cols);
        const gy = ly + 26, cg = 8;
        const cw = (w - 2 * pad - cg * (cols - 1)) / cols, ch = 60;
        dets.forEach((dt, i) => {
          const c = i % cols, r = Math.floor(i / cols);
          const x = pad + c * (cw + cg), y = gy + r * (ch + cg);
          const lvl = d.hit[dt.k] || 0;
          const reached = dt.s >= d.o && pos >= dt.s - 0.02;
          const on = reached && lvl > 0;
          const col = lvl === 2 ? C.red : lvl === 1 ? C.d : lvl === 3 ? C.e : C.muted;
          V.box(g, x, y, cw, ch, { r: 10, fill: on ? V.alpha(col, 0.12) : V.alpha(C.muted, 0.05), stroke: on ? V.alpha(col, 0.8) : V.alpha(C.muted, 0.25), lw: on ? 1.6 : 1 });
          const lx = x + 17, lyy = y + ch / 2;
          if (on) V.glow(g, lx, lyy, 15 + 3 * Math.sin(t * 5 + i), col, 0.8);
          V.dot(g, lx, lyy, 6.5, on ? col : V.alpha(C.muted, 0.35));
          const tx = x + 31, tmax = cw - 36;
          const fs = narrow ? 11 : 12;
          V.text(g, clip(g, dt.l, tmax, fs, 'sans', 600), tx, y + 21, { size: fs, weight: 600, color: C.ink });
          V.text(g, clip(g, stages[dt.s] + ' · ' + dt.cad, tmax, 10, 'sans'), tx, y + 37, { size: 10, color: C.muted });
          const st = dt.s < d.o ? 'off the path' : !reached ? '…' : lvl === 2 ? 'ALARM' : lvl === 1 ? 'weak signal' : lvl === 3 ? 'within noise' : 'sees nothing';
          V.text(g, clip(g, st, tmax, 10, 'mono', 700), tx, y + 52, { size: 10, font: 'mono', weight: 700, color: on ? col : C.muted });
        });
        const ny = gy + rows * (ch + cg) + 16;
        V.text(g, 'DIAGNOSIS', pad, ny, { size: 10, font: 'mono', color: C.muted, weight: 700 });
        drawWrap(g, d.note, pad, ny + 19, w - 2 * pad, narrow ? 16.5 : 18.5, { size: narrow ? 12 : 13, color: C.ink });
      });
    })();

    /* =========================================================
       F2 — The surprise machine
       ========================================================= */
    (() => {
      const fig = root.querySelector('#fig-surpresa'); if (!fig) return;
      fitStage(fig, (w) => (w < 640 ? 545 : 400));
      const toks = ['The', 'cat', 'saw', 'the', 'sun', 'in', 'the', 'May', 'sky'];
      const D = [
        { c: ['cat', 'dog', 'day', 'mouse', 'sky', 'wind'], l: [1.6, 1.2, 1.0, 0.6, 0.4, 0.3] },
        { c: ['climbed', 'saw', 'ate', 'slept', 'jumped', 'looked'], l: [1.5, 1.4, 1.0, 0.9, 0.6, 0.5] },
        { c: ['the', 'a', 'one', 'some', 'that', 'it'], l: [2.4, 1.6, 0.8, 0.6, 0.3, 0.1] },
        { c: ['sun', 'house', 'ball', 'window', 'shadow', 'owner'], l: [2.0, 1.0, 0.9, 0.7, 0.5, 0.4] },
        { c: ['in', 'rise', 'and', 'at', 'shine', 'set'], l: [2.2, 1.8, 0.9, 0.5, 0.4, 0.2] },
        { c: ['the', 'a', 'its', 'May', 'our', 'full'], l: [3.0, 0.9, 0.8, 0.5, 0.3, 0.2] },
        { c: ['night', 'May', 'sky', 'dark', 'clear', 'east'], l: [1.6, 1.2, 1.1, 0.9, 0.6, 0.5] },
        { c: ['sky', 'night', 'rain', 'air', 'wind', 'dawn'], l: [1.3, 1.2, 1.0, 0.9, 0.8, 0.7] },
      ];
      const tgt = D.map((d, i) => d.c.indexOf(toks[i + 1]));
      const rg = V.rng(31);
      const Z = D.map((d) => d.c.map(() => rg.normal()));
      const S = { sig: 0.6, mode: 'rand' };
      const R = {};
      const ctl = UI.controls(fig);
      let rd = null;
      const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;
      function compute() {
        const ref = D.map((d) => softmax(d.l));
        const lg = D.map((d, i) => d.l.map((x, j) => x + S.sig * Z[i][j]));
        if (S.mode === 'cancel') {
          for (let i = 0; i + 1 < D.length; i += 2) {
            const a = D[i].l.map((x, j) => (j === tgt[i] ? x + S.sig * 0.9 : x + S.sig * Z[i][j]));
            lg[i] = a;
            const dA = -Math.log(softmax(a)[tgt[i]]) + Math.log(ref[i][tgt[i]]);
            const k = i + 1;
            const make = (dl) => D[k].l.map((x, j) => (j === tgt[k] ? x + dl : x + S.sig * Z[k][j]));
            const dNLL = (dl) => -Math.log(softmax(make(dl))[tgt[k]]) + Math.log(ref[k][tgt[k]]);
            let lo = -12, hi = 12;
            for (let it = 0; it < 60; it++) { const mid = (lo + hi) / 2; if (dNLL(mid) > -dA) lo = mid; else hi = mid; }
            lg[k] = make((lo + hi) / 2);
          }
        }
        const cand = lg.map((l) => softmax(l));
        R.ref = ref; R.cand = cand;
        R.sr = ref.map((p, i) => -Math.log(p[tgt[i]]));
        R.sc = cand.map((p, i) => -Math.log(p[tgt[i]]));
        R.kl = ref.map((p, i) => klDiv(p, cand[i]));
        R.nllR = mean(R.sr); R.nllC = mean(R.sc); R.klM = mean(R.kl);
        R.top1 = ref.filter((p, i) => argmax(p) === argmax(cand[i])).length;
        if (rd) {
          rd.nll.set(f(R.nllR, 3) + ' → ' + f(R.nllC, 3));
          rd.ppl.set(f(Math.exp(R.nllR), 2) + ' → ' + f(Math.exp(R.nllC), 2));
          rd.kl.set(f(R.klM, 3) + ' nats');
        }
      }
      UI.slider(ctl, { label: 'Logit noise', min: 0, max: 1.5, step: 0.05, value: S.sig, fmt: (v) => f(v, 2), onInput: (v) => { S.sig = v; compute(); } });
      UI.seg(ctl, { label: 'Mode', options: [{ v: 'rand', label: 'random noise' }, { v: 'cancel', label: 'errors that cancel' }], value: 'rand', onChange: (v) => { S.mode = v; compute(); } });
      rd = { nll: UI.readout(ctl, 'NLL ref → cand', '—'), ppl: UI.readout(ctl, 'PPL', '—'), kl: UI.readout(ctl, 'mean KL', '—') };
      compute();
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const narrow = w < 640, pad = 14;
        const rightW = narrow ? 0 : Math.min(280, w * 0.37);
        const lw = w - 2 * pad - (narrow ? 0 : rightW + 18);
        const nT = toks.length, slot = lw / nT;
        const cur = Math.floor(t / 1.7) % D.length;
        const fsz = slot < 42 ? 10.5 : 12.5;
        V.text(g, clip(g, 'evaluation text · the model reads it all at once', lw, 10.5, 'mono'), pad, 16, { size: 10.5, color: C.muted, font: 'mono' });
        const ty = 26, th = 28;
        for (let i = 0; i < nT; i++) {
          const x = pad + i * slot + 2, bw = slot - 4;
          const isT = i === cur + 1, isCtx = i <= cur;
          V.box(g, x, ty, bw, th, { r: 7, fill: isT ? V.alpha(C.d, 0.25) : isCtx ? V.alpha(C.a, 0.14) : V.alpha(C.muted, 0.08), stroke: isT ? C.d : isCtx ? V.alpha(C.a, 0.6) : V.alpha(C.muted, 0.3), lw: isT ? 2 : 1 });
          V.text(g, clip(g, toks[i], bw - 4, fsz, 'sans', 600), x + bw / 2, ty + th / 2 + 1, { size: fsz, weight: 600, align: 'center', base: 'middle', color: C.ink });
        }
        const barTop = ty + th + 30, barH = narrow ? 96 : 150;
        V.text(g, 'p(true token)', pad, barTop - 10, { size: 10, color: C.muted, font: 'mono' });
        V.line(g, pad + slot * 0.6, barTop + barH, pad + lw, barTop + barH, { color: V.alpha(C.muted, 0.5), lw: 1 });
        for (let k = 0; k < D.length; k++) {
          const i = k + 1;
          const x0 = pad + i * slot + slot / 2;
          const bwid = Math.max(5, Math.min(13, slot * 0.28));
          const pr = R.ref[k][tgt[k]], pc = R.cand[k][tgt[k]];
          const hr = Math.max(1, pr * barH), hc = Math.max(1, pc * barH);
          V.rr(g, x0 - bwid - 1, barTop + barH - hr, bwid, hr, 3); g.fillStyle = V.alpha(C.a, 0.14); g.fill(); g.strokeStyle = C.a; g.lineWidth = 1.6; g.stroke();
          V.rr(g, x0 + 1, barTop + barH - hc, bwid, hc, 3); g.fillStyle = V.alpha(C.b, 0.85); g.fill();
          if (k === cur) V.glow(g, x0, barTop + barH - Math.max(hr, hc) - 8, 14, C.d, 0.7);
          const sy = barTop + barH + 15;
          V.text(g, f(R.sr[k], 2), x0, sy, { size: 9.5, font: 'mono', align: 'center', color: C.a });
          V.text(g, f(R.sc[k], 2), x0, sy + 13, { size: 9.5, font: 'mono', align: 'center', color: C.b });
        }
        V.text(g, narrow ? '−ln p' : '−ln p ref', pad, barTop + barH + 15, { size: 9.5, font: 'mono', color: C.a });
        V.text(g, 'cand', pad, barTop + barH + 28, { size: 9.5, font: 'mono', color: C.b });
        // distribution panel
        let px, py, pw, ph;
        if (narrow) { px = pad; py = barTop + barH + 44; pw = w - 2 * pad; ph = h - py - 76; }
        else { px = w - pad - rightW; py = 8; pw = rightW; ph = h - 8 - 84; }
        V.box(g, px, py, pw, ph, { r: 12, fill: V.alpha(C.b, 0.05), stroke: V.alpha(C.muted, 0.3) });
        V.text(g, clip(g, 'position ' + (cur + 1) + ' · after “' + toks.slice(0, cur + 1).join(' ') + '”', pw - 20, 11, 'sans', 600), px + 10, py + 19, { size: 11, weight: 600, color: C.ink });
        const dd = D[cur], pr = R.ref[cur], pc = R.cand[cur];
        const nC = dd.c.length;
        const rowH = Math.min(28, (ph - 52) / nC);
        const labW = Math.min(64, pw * 0.26);
        const barW = pw - labW - 86;
        for (let j = 0; j < nC; j++) {
          const y = py + 30 + j * rowH;
          const isT = j === tgt[cur];
          V.text(g, clip(g, dd.c[j], labW - 4, 11, 'sans', isT ? 700 : 400), px + 10, y + rowH / 2 + 4, { size: 11, weight: isT ? 700 : 400, color: isT ? C.d : C.ink });
          const x0 = px + 10 + labW, hh = Math.max(3, (rowH - 7) / 2);
          V.rr(g, x0, y + 2, Math.max(1.5, pr[j] * barW), hh, 2); g.strokeStyle = C.a; g.lineWidth = 1.2; g.stroke();
          V.rr(g, x0, y + 3 + hh, Math.max(1.5, pc[j] * barW), hh, 2); g.fillStyle = V.alpha(C.b, 0.85); g.fill();
          V.text(g, pct(pr[j], 0) + '→' + pct(pc[j], 0), px + pw - 8, y + rowH / 2 + 4, { size: 9.5, font: 'mono', align: 'right', color: C.muted });
        }
        V.text(g, 'KL at this position: ' + f(R.kl[cur], 3) + ' nats', px + 10, py + ph - 11, { size: 11, font: 'mono', color: C.ink });
        // summary
        const dN = R.nllC - R.nllR;
        const sg = (x) => (x >= 0 ? '+' : '−') + f(Math.abs(x), 3);
        const lines = narrow
          ? ['NLL ' + f(R.nllR, 3) + ' → ' + f(R.nllC, 3) + ' (Δ ' + sg(dN) + ')', 'PPL ' + f(Math.exp(R.nllR), 2) + ' → ' + f(Math.exp(R.nllC), 2) + ' · mean KL ' + f(R.klM, 3), 'top-1 same at ' + R.top1 + '/' + D.length + ' positions']
          : ['NLL ' + f(R.nllR, 3) + ' → ' + f(R.nllC, 3) + ' nats (Δ ' + sg(dN) + ')   ·   PPL ' + f(Math.exp(R.nllR), 2) + ' → ' + f(Math.exp(R.nllC), 2), 'mean KL ' + f(R.klM, 3) + ' nats   ·   top-1 same at ' + R.top1 + '/' + D.length + ' positions'];
        const hidden = Math.abs(dN) < 0.02 && R.klM > 0.02;
        if (hidden) lines.push((narrow ? '→ PPL flat, distributions changed: mean hid it' : '→ perplexity barely moved, but the distributions changed: the mean hid the damage'));
        const y0 = h - 12 - (lines.length - 1) * 17;
        lines.forEach((s, i) => { const warn = hidden && i === lines.length - 1; const wt = warn ? 700 : 500; V.text(g, clip(g, s, w - 2 * pad, narrow ? 11 : 12, 'mono', wt), pad, y0 + i * 17, { size: narrow ? 11 : 12, font: 'mono', weight: wt, color: warn ? C.c : C.ink }); });
      });
    })();

    /* =========================================================
       F3 — Two distributions, one ruler (KL)
       ========================================================= */
    (() => {
      const fig = root.querySelector('#fig-kl'); if (!fig) return;
      fitStage(fig, (w) => (w < 640 ? 480 : 410));
      const toks = ['Paris', 'Lyon', 'a', 'the', 'also', 'lovely', 'Nice', 'home'];
      const base = [3.2, 1.3, 1.1, 0.9, 0.8, 0.4, 0.2, 0.0];
      const P = softmax(base);
      const mkZ = (s) => { const r = V.rng(s); return base.map(() => r.normal()); };
      const S = { sig: 0.7, k: 8, rev: false, seed: 5, zFrom: mkZ(5), zTo: mkZ(5), tm: -99, pend: false, zNow: mkZ(5) };
      const ctl = UI.controls(fig);
      UI.slider(ctl, { label: 'Logit noise', min: 0, max: 1.5, step: 0.05, value: S.sig, fmt: (v) => f(v, 2), onInput: (v) => { S.sig = v; } });
      UI.slider(ctl, { label: 'Top-k support', min: 2, max: 8, step: 1, value: 8, fmt: (v) => (v === 8 ? '8 (all)' : String(v)), onInput: (v) => { S.k = v; } });
      UI.toggle(ctl, { label: 'flip the direction: D(Q‖P)', value: false, onChange: (v) => { S.rev = v; } });
      UI.button(ctl, '↻ new noise', () => { S.seed += 1; S.pend = true; });
      const topk = (p, k) => p.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]).slice(0, k).map((x) => x[1]);
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        if (S.pend) { S.zFrom = S.zNow.slice(); S.zTo = mkZ(S.seed); S.tm = t; S.pend = false; }
        const u = V.ease.inOut(V.clamp((t - S.tm) / 0.8, 0, 1));
        S.zNow = S.zTo.map((z, i) => V.lerp(S.zFrom[i], z, u));
        const Q = softmax(base.map((x, i) => x + S.sig * S.zNow[i]));
        const A = S.rev ? Q : P, B = S.rev ? P : Q;
        const kl = klDiv(A, B);
        const setP = new Set(topk(P, S.k)), setQ = new Set(topk(Q, S.k));
        const shared = [...setP].filter((i) => setQ.has(i));
        let klk = NaN;
        if (shared.length) { const za = shared.reduce((s, i) => s + A[i], 0), zb = shared.reduce((s, i) => s + B[i], 0); klk = Math.max(0, shared.reduce((s, i) => s + (A[i] / za) * Math.log((A[i] / za) / (B[i] / zb)), 0)); }
        const Hn = -A.reduce((s, v) => s + (v > 0 ? v * Math.log(v) : 0), 0);
        const narrow = w < 640, pad = 14;
        const rightW = narrow ? 0 : 228;
        const cw = w - 2 * pad - (narrow ? 0 : rightW + 16);
        const n = toks.length, slot = cw / n;
        const top = 26, barH = narrow ? 150 : 170;
        const pmax = Math.max(0.7, ...P, ...Q) * 1.05;
        V.text(g, '“The capital of France is …”', pad, 14, { size: 11, color: C.muted, font: 'mono' });
        V.line(g, pad, top + barH, pad + cw, top + barH, { color: V.alpha(C.muted, 0.5), lw: 1 });
        const inSup = (i) => shared.includes(i);
        for (let i = 0; i < n; i++) {
          const x0 = pad + i * slot + slot / 2, bw = Math.max(5, Math.min(16, slot * 0.3));
          const al = S.k === 8 || inSup(i) ? 1 : 0.28;
          const hp = (P[i] / pmax) * barH, hq = (Q[i] / pmax) * barH;
          g.globalAlpha = al;
          V.rr(g, x0 - bw - 1, top + barH - hp, bw, Math.max(1, hp), 3); g.fillStyle = V.alpha(C.a, 0.15); g.fill(); g.strokeStyle = C.a; g.lineWidth = 1.6; g.stroke();
          V.rr(g, x0 + 1, top + barH - hq, bw, Math.max(1, hq), 3); g.fillStyle = V.alpha(C.b, 0.85); g.fill();
          if (slot >= 34) {
            V.text(g, pct(P[i], 0), x0 - bw / 2 - 1, top + barH - hp - 5, { size: 9, font: 'mono', align: 'center', color: C.a });
            V.text(g, pct(Q[i], 0), x0 + bw / 2 + 1, top + barH - hq - 5, { size: 9, font: 'mono', align: 'center', color: C.b });
          }
          g.globalAlpha = 1;
          V.text(g, clip(g, toks[i], slot - 2, narrow ? 10 : 11.5, 'sans', 600), x0, top + barH + 14, { size: narrow ? 10 : 11.5, weight: 600, align: 'center', color: al < 1 ? C.muted : C.ink });
        }
        // contributions
        const cH = narrow ? 40 : 58, yz = top + barH + 44 + cH;
        const contrib = A.map((a, i) => (a > 0 ? a * Math.log(a / B[i]) : 0));
        const cmax = Math.max(0.03, ...contrib.map(Math.abs));
        V.text(g, clip(g, 'contribution ' + (S.rev ? 'Q·ln(Q/P)' : 'P·ln(P/Q)') + (narrow ? '' : ' of each token') + ' · sum = ' + f(kl, 4) + ' nats', cw, 10, 'mono'), pad, top + barH + 34, { size: 10, font: 'mono', color: C.muted });
        V.line(g, pad, yz, pad + cw, yz, { color: V.alpha(C.muted, 0.6), lw: 1 });
        for (let i = 0; i < n; i++) {
          const x0 = pad + i * slot + slot / 2, bw = Math.max(6, Math.min(22, slot * 0.45));
          const c = contrib[i], hh = (Math.abs(c) / cmax) * cH;
          const al2 = S.k === 8 || inSup(i) ? 1 : 0.3;
          g.fillStyle = c >= 0 ? V.alpha(C.c, 0.85 * al2) : V.alpha(C.b, 0.7 * al2);
          V.rr(g, x0 - bw / 2, c >= 0 ? yz - hh : yz, bw, Math.max(1, hh), 2); g.fill();
        }
        V.text(g, clip(g, S.rev ? 'pink: Q > P (candidate overestimated) · lavender: Q < P' : 'pink: P > Q (candidate underestimated) · lavender: P < Q', cw, 10, 'sans'), pad, yz + cH + 18, { size: 10, color: C.muted });
        // panel
        let px, py, pw;
        if (narrow) { px = pad; py = yz + cH + 32; pw = w - 2 * pad; } else { px = w - pad - rightW; py = 10; pw = rightW; }
        const ph = narrow ? Math.min(h - py - 8, 132) : h - 20;
        V.box(g, px, py, pw, ph, { r: 12, fill: V.alpha(C.b, 0.05), stroke: V.alpha(C.muted, 0.3) });
        const same = argmax(P) === argmax(Q);
        const items = [
          [(S.rev ? 'D(Q‖P)' : 'D(P‖Q)') + ' exact', f(kl, 4) + ' nats', C.ink],
          ['top-' + S.k + ' support' + (S.k === 8 ? ' (= exact)' : ' · ' + shared.length + ' shared'), (Number.isFinite(klk) ? f(klk, 4) : '—') + ' nats', C.b],
          ['top-1', same ? 'same (“' + toks[argmax(P)] + '”)' : 'changed: “' + toks[argmax(P)] + '” → “' + toks[argmax(Q)] + '”', same ? C.e : C.red],
          [(S.rev ? 'H(Q)' : 'H(P)') + ' · entropy', f(Hn, 3) + ' nats', C.muted],
          ['cross-entropy = H + KL', f(Hn + kl, 3) + ' nats', C.muted],
        ];
        if (narrow) {
          const colW = (pw - 20) / 2;
          items.forEach((it, i) => { const cx = px + 10 + (i % 2) * colW, cy = py + 18 + Math.floor(i / 2) * 36; V.text(g, clip(g, it[0], colW - 6, 10, 'sans'), cx, cy, { size: 10, color: C.muted }); V.text(g, clip(g, it[1], colW - 6, 12, 'mono', 700), cx, cy + 16, { size: 12, font: 'mono', weight: 700, color: it[2] }); });
        } else {
          items.forEach((it, i) => { const cy = py + 24 + i * 52; V.text(g, it[0], px + 12, cy, { size: 10.5, color: C.muted }); V.text(g, clip(g, it[1], pw - 24, i < 2 ? 17 : 13, 'mono', 700), px + 12, cy + 21, { size: i < 2 ? 17 : 13, font: 'mono', weight: 700, color: it[2] }); });
          if (S.k < 8 && Number.isFinite(klk) && kl > 0) V.text(g, 'the proxy sees ' + pct(klk / kl, 0) + ' of the KL', px + 12, py + ph - 14, { size: 10.5, font: 'mono', color: C.b });
        }
      });
    })();

    /* =========================================================
       F4 — The calibrated panel, step by step
       ========================================================= */
    (() => {
      const fig = root.querySelector('#fig-painel'); if (!fig) return;
      fitStage(fig, (w) => (w < 620 ? 475 : 470));
      const NP = 14, NT = 5, NS = 4, NC = 3;
      const rg = V.rng(4242);
      const margins = [0.9, 0.12, 1.4, 0.34, 2.1, 0.05, 0.62, 1.1, 0.3, 0.18, 1.8, 0.42, 0.8, 0.27];
      const base = margins.map((m) => { const l = [2.2, 2.2 - m]; for (let j = 2; j < NT; j++) l.push(2.2 - m - 0.5 - rg() * 1.8); return l; });
      const nz = () => base.map(() => Array.from({ length: NT }, () => rg.normal()));
      const nS = Array.from({ length: NS }, nz), nC = Array.from({ length: NC }, nz);
      const dirR = base.map(() => Array.from({ length: NT }, (_, j) => (j < 2 ? 0 : rg.normal() * 0.9)));
      const swap = margins.map((m) => (m <= 1.0 ? 1 : 0));
      const UNST = 6;
      // "boot-to-boot variation" mode: S1,S2 = boot 1; S3,S4 = boot 2 (noise partially shared within a boot)
      const nB = [nz(), nz()], nBc = nz();
      const WC = 0.85, WB = Math.sqrt(1 - WC * WC);
      const unstK = (s, i, j) => (s === 2 && i === UNST ? (j === 0 ? -0.5 : j === 1 ? 0.5 : 0) : 0);
      const swK = (i, j, a) => (swap[i] ? (j === 0 ? -0.6 : j === 1 ? 0.6 : 0) * a * a : 0);
      const capS = (sig) => nS.map((nn, s) => base.map((b, i) => b.map((x, j) => x + sig * nn[i][j] + unstK(s, i, j))));
      const capC = (sig, a) => nC.map((nn) => base.map((b, i) => b.map((x, j) => x + sig * nn[i][j] + a * dirR[i][j] + swK(i, j, a))));
      const capSB = (sig) => nS.map((nn, s) => base.map((b, i) => b.map((x, j) => x + sig * (WC * nn[i][j] + WB * nB[s >> 1][i][j]) + unstK(s, i, j))));
      const capCB = (sig, a) => nC.map((nn) => base.map((b, i) => b.map((x, j) => x + sig * (WC * nn[i][j] + WB * nBc[i][j]) + a * dirR[i][j] + swK(i, j, a))));
      const info = (L) => L.map((cap) => cap.map((l) => { const p = softmax(l); const am = argmax(l); let s2 = -Infinity; l.forEach((x, j) => { if (j !== am && x > s2) s2 = x; }); return { p, am, mg: l[am] - s2 }; }));
      const pairKL = (A) => { const res = []; for (let i = 0; i < NP; i++) { let mx = 0, sm = 0, n = 0; for (let a = 0; a < A.length; a++) for (let b = 0; b < A.length; b++) if (a !== b) { const k = klDiv(A[a][i].p, A[b][i].p); mx = Math.max(mx, k); sm += k; n++; } res.push({ mx, mean: sm / n }); } return res; };
      const classesOf = (SI, mstar) => SI[0].map((_, i) => { const mm = Math.min(...SI.map((c) => c[i].mg)); const dis = new Set(SI.map((c) => c[i].am)).size > 1; return mm < mstar ? 'TIE' : dis ? 'UNSTABLE' : 'CONF'; });
      const controlOf = (SI, cls, boots) => {
        const rs = [], ks = [];
        const groups = boots ? [[0, 1], [2, 3]] : [[0], [1], [2], [3]];
        for (let i = 0; i < NP; i++) {
          if (cls[i] !== 'CONF') continue;
          let racc = 0, kacc = 0;
          groups.forEach((hold) => {
            const Rr = [0, 1, 2, 3].filter((x) => !hold.includes(x));
            let mx = 0; Rr.forEach((a) => Rr.forEach((b) => { if (a !== b) mx = Math.max(mx, klDiv(SI[a][i].p, SI[b][i].p)); }));
            let km = 0, n = 0; hold.forEach((hh) => Rr.forEach((a) => { km += klDiv(SI[a][i].p, SI[hh][i].p); n++; })); km /= n;
            racc += km / Math.max(mx, 1e-12); kacc += km;
          });
          rs.push(racc / groups.length); ks.push(kacc / groups.length);
        }
        const m = (a) => a.reduce((x, y) => x + y, 0) / Math.max(1, a.length);
        return { rs, ks, rbar: m(rs), absKL: m(ks) };
      };
      // calibrate the stock noise to a null (control) of ~0.0144 nats
      let sig = 0.15;
      { let lo = 0.01, hi = 0.8; for (let it = 0; it < 36; it++) { const mid = (lo + hi) / 2; const SIm = info(capS(mid)); const c = controlOf(SIm, classesOf(SIm, 0.25)); if (c.absKL < 0.0144) lo = mid; else hi = mid; } sig = (lo + hi) / 2; }
      const SI = info(capS(sig)); const PK = pairKL(SI);
      const SIb = info(capSB(sig)); const PKb = pairKL(SIb);
      const evalCand = (CI, cls, SIx, PKx) => {
        const SIu = SIx || SI, PKu = PKx || PK;
        const rows = [];
        for (let i = 0; i < NP; i++) {
          let sm = 0, n = 0; for (let a = 0; a < NS; a++) for (let c = 0; c < CI.length; c++) { sm += klDiv(SIu[a][i].p, CI[c][i].p); n++; }
          const kl = sm / n, r = kl / Math.max(PKu[i].mx, 1e-12);
          const votes = {}; CI.forEach((c) => { votes[c[i].am] = (votes[c[i].am] || 0) + 1; });
          let best = null; Object.keys(votes).forEach((k) => { if (best === null || votes[k] > votes[best] || (votes[k] === votes[best] && +k < +best)) best = k; });
          rows.push({ kl, r, cons: +best, flip: cls[i] === 'CONF' && +best !== SIu[0][i].am });
        }
        const conf = rows.filter((_, i) => cls[i] === 'CONF');
        const m = (a) => a.reduce((x, y) => x + y, 0) / Math.max(1, a.length);
        return { rows, rbar: m(conf.map((x) => x.r)), flips: conf.filter((x) => x.flip).length, absKL: m(conf.map((x) => x.kl)) };
      };
      const cls0 = classesOf(SI, 0.25);
      const cands = [
        { v: 'bf16', label: 'KDA BF16 copy (real: 1.10× the null, 0 flips)', short: 'KDA BF16', target: 1.10 },
        { v: 'fp8', label: 'FP8=all (real: KL 25–30% above the stock pairs)', short: 'FP8=all', target: 1.28 },
        { v: 'w8a8', label: 'W8A8 · positive control (real: 7.99×, 23 flips)', short: 'W8A8', target: 7.99 },
      ];
      cands.forEach((c) => { let lo = 0, hi = 30; for (let it = 0; it < 40; it++) { const mid = (lo + hi) / 2; const e = evalCand(info(capC(sig, mid)), cls0); if (e.rbar < c.target) lo = mid; else hi = mid; } c.alpha = (lo + hi) / 2; c.CI = info(capC(sig, c.alpha)); c.CIb = info(capCB(sig, c.alpha)); });
      const S = { cand: cands[0], mstar: 0.25, boot: false };
      let M = null;
      const ctl = UI.controls(fig);
      let rdC = null, rdR = null;
      const recompute = () => {
        const SIx = S.boot ? SIb : SI, PKx = S.boot ? PKb : PK, CIx = S.boot ? S.cand.CIb : S.cand.CI;
        const cls = classesOf(SIx, S.mstar);
        const E = evalCand(CIx, cls, SIx, PKx);
        const CT = controlOf(SIx, cls, S.boot);
        M = { SIx, CIx, cls, E, CT, nConf: cls.filter((c) => c === 'CONF').length, nTie: cls.filter((c) => c === 'TIE').length, nUn: cls.filter((c) => c === 'UNSTABLE').length };
        if (rdC) rdC.set(M.nConf + ' / ' + M.nTie + ' / ' + M.nUn);
        if (rdR) rdR.set(f(E.rbar, 2) + '× · ' + E.flips + ' flip' + (E.flips === 1 ? '' : 's'));
      };
      UI.select(ctl, { label: 'Candidate', options: cands.map((c) => ({ v: c.v, label: c.label })), value: 'bf16', onChange: (v) => { S.cand = cands.find((c) => c.v === v) || cands[0]; recompute(); } });
      UI.slider(ctl, { label: 'Cutoff margin', min: 0.05, max: 1.5, step: 0.05, value: 0.25, fmt: (v) => f(v, 2) + ' nats', onInput: (v) => { S.mstar = v; recompute(); } });
      UI.toggle(ctl, { label: 'boot-to-boot variation (2 boots × 2 captures, illustrative)', value: false, onChange: (v) => { S.boot = v; recompute(); if (HP) HP.redraw(); } });
      rdC = UI.readout(ctl, 'CONF / TIE / UNSTABLE', '—');
      rdR = UI.readout(ctl, 'mean ratio · flips in CONF', '—');
      recompute();
      const steps = [
        { t: 'Capture the stock (4 boots)', d: 'Each column S1–S4 is a clean stock boot requesting <code>prompt_logprobs=20</code> over the same text. The color is the highest-probability token at each position; the brightness, the margin to the runner-up. Notice: where the margin is small, the stock itself changes color.' },
        { t: 'Measure the margin', d: 'The bar on the left is the smallest margin \\(\\ln p_{(1)}-\\ln p_{(2)}\\) across the stock captures. Below \\(m^*\\) (dashed line), the position becomes TIE: it is noted, but not judged by flips. In GLM’s screening script, \\(m^*\\) is fixed at 0.25 nats; the sealed thin-decode study derived 1.5 nats from its own calibration. Drag the control and watch the classes change.' },
        { t: 'Set aside the unstable ones', d: 'If the stock switched argmax even with every margin above \\(m^*\\), the position is UNSTABLE and drops out of every gate. That way, a single bimodal position does not widen the ruler for the others.' },
        { t: 'Compare the candidate', d: 'Columns C1–C3 are candidate captures. Only at CONF positions does a consensus flip count (✕). The real panel raises FLAG at or above the Poisson critical value (11 flips, with \\(\\lambda=6\\)) and tolerates up to 10: a heuristic slack.' },
        { t: 'Standardize the KL', d: 'At each CONF position, the mean candidate↔stock KL is divided by the largest KL between stock pairs at that position. The bar on the right is that ratio \\(r_t\\); the line marks the 2× gate.' },
        { t: 'Verdict against the null', d: 'On top, the null ratio of the candidate and of the control (each stock capture judged against the others): the script answers WITHIN (exit 0) or FLAG (exit 1). Below, the absolute KL against the “KL ≤ 0.01” gate. When the control fails a gate, the instrument is not fit to judge (in the formal studies, INCONCLUSIVE): the defect is in the ruler. Turn on “boot-to-boot variation” to watch the control fail its own calibrated gate, as on 09-26.' },
      ];
      let HP = null;
      const st = UI.stepper(fig, steps, { autoplay: 7500, onStep: () => { if (HP) HP.redraw(); } });
      const letters = 'ABCDE';
      HP = V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const tokCol = [C.a, C.b, C.c, C.d, C.e];
        const step = st.i, el = st.elapsed();
        const narrow = w < 620, pad = 12;
        const { cls, E, CT } = M;
        if (step < 5) {
          const idxW = narrow ? 18 : 26, mbW = narrow ? 34 : 56, clsW = narrow ? 42 : 60, flW = 20, minR = narrow ? 70 : 110;
          const fixed = 2 * pad + idxW + mbW + clsW + flW + minR + 32;
          const cs = V.clamp((w - fixed - 14) / 7, 14, 30);
          const top = 34, foot = 44;
          const rh = V.clamp((h - top - foot) / NP, 14, 26);
          const natR = w - pad - (pad + idxW + mbW + 8 + 4 * (cs + 2) + 6 + clsW + 6 + 3 * (cs + 2) + 4 + flW + 6);
          const ox = natR > 300 ? (natR - 300) / 2 : 0;
          const xIdx = pad + ox, xMb = xIdx + idxW, xS = xMb + mbW + 8, xCls = xS + 4 * (cs + 2) + 6, xC = xCls + clsW + 6, xFl = xC + 3 * (cs + 2) + 4, xR = xFl + flW + 6, wR = Math.min(300, w - pad - xR);
          const hdr = (s, x, ww, on = true) => V.text(g, clip(g, s, ww + 8, 10, 'mono', 600), x + ww / 2, top - 10, { size: 10, font: 'mono', weight: 600, align: 'center', color: on ? C.muted : V.alpha(C.muted, 0.35) });
          hdr('t', xIdx, idxW - 4);
          hdr(narrow ? 'marg' : 'margin', xMb, mbW, step >= 1);
          for (let s = 0; s < NS; s++) hdr(S.boot ? ['B1a', 'B1b', 'B2a', 'B2b'][s] : 'S' + (s + 1), xS + s * (cs + 2), cs);
          hdr(narrow ? 'cls' : 'class', xCls, clsW, step >= 2);
          for (let c = 0; c < NC; c++) hdr('C' + (c + 1), xC + c * (cs + 2), cs, step >= 3);
          hdr('flip', xFl, flW, step >= 3);
          hdr(narrow ? 'r_t' : 'r_t (KL / null)', xR, wR, step >= 4);
          const mmax = 1.5;
          for (let i = 0; i < NP; i++) {
            const y = top + i * rh;
            const cl = cls[i];
            const dim = step >= 2 && cl !== 'CONF' && step >= 3;
            V.text(g, String(i + 1), xIdx + idxW - 6, y + rh / 2 + 3.5, { size: 10, font: 'mono', align: 'right', color: C.muted });
            // stock cells
            for (let s = 0; s < NS; s++) {
              const vis = step > 0 || el > s * 0.45 + i * 0.03;
              if (!vis) continue;
              const inf = M.SIx[s][i];
              const x = xS + s * (cs + 2);
              const a = 0.22 + 0.66 * V.clamp(inf.mg / 1.2, 0, 1);
              V.box(g, x, y + 1, cs, rh - 2, { r: 4, fill: V.alpha(tokCol[inf.am], dim ? a * 0.45 : a) });
              if (cs >= 16 && rh >= 15) V.text(g, letters[inf.am], x + cs / 2, y + rh / 2 + 0.5, { size: 10, font: 'mono', weight: 700, align: 'center', base: 'middle', color: C.ink, alpha: dim ? 0.5 : 0.9 });
            }
            if (step >= 1) {
              const mm = Math.min(...M.SIx.map((c) => c[i].mg));
              const bl = V.clamp(mm / mmax, 0.02, 1) * (mbW - 4);
              V.box(g, xMb, y + rh / 2 - 3.5, bl, 7, { r: 3, fill: mm < S.mstar ? V.alpha(C.d, 0.85) : V.alpha(C.a, 0.8) });
            }
            if (step >= 2) {
              const col = cl === 'CONF' ? C.e : cl === 'TIE' ? C.d : C.red;
              const lab = cl === 'UNSTABLE' ? (narrow ? 'UNST' : 'UNSTABLE') : cl;
              V.box(g, xCls, y + 2, clsW, rh - 4, { r: (rh - 4) / 2, fill: V.alpha(col, 0.15), stroke: V.alpha(col, 0.7), lw: 1 });
              V.text(g, clip(g, lab, clsW - 4, 9.5, 'mono', 700), xCls + clsW / 2, y + rh / 2 + 0.5, { size: 9.5, font: 'mono', weight: 700, align: 'center', base: 'middle', color: col });
            }
            if (step >= 3) {
              const CI = M.CIx;
              for (let c = 0; c < NC; c++) {
                const vis = step > 3 || el > 0.3 + c * 0.45 + i * 0.03;
                if (!vis) continue;
                const inf = CI[c][i]; const x = xC + c * (cs + 2);
                const a = 0.22 + 0.66 * V.clamp(inf.mg / 1.2, 0, 1);
                V.box(g, x, y + 1, cs, rh - 2, { r: 4, fill: V.alpha(tokCol[inf.am], dim ? a * 0.45 : a), stroke: inf.am !== M.SIx[0][i].am && cl === 'CONF' ? C.red : null, lw: 1.5 });
                if (cs >= 16 && rh >= 15) V.text(g, letters[inf.am], x + cs / 2, y + rh / 2 + 0.5, { size: 10, font: 'mono', weight: 700, align: 'center', base: 'middle', color: C.ink, alpha: dim ? 0.5 : 0.9 });
              }
              if (cl === 'CONF') {
                if (E.rows[i].flip) V.text(g, '✕', xFl + flW / 2, y + rh / 2 + 1, { size: 13, weight: 700, align: 'center', base: 'middle', color: C.red });
                else V.text(g, '·', xFl + flW / 2, y + rh / 2 + 1, { size: 13, align: 'center', base: 'middle', color: C.muted });
              }
            }
            if (step >= 4) {
              if (cl === 'CONF') {
                const r = E.rows[i].r, cap = 10;
                const bl = V.clamp(r / cap, 0.01, 1) * (wR - 34);
                V.box(g, xR, y + rh / 2 - 4, bl, 8, { r: 3, fill: r > 2 ? V.alpha(C.red, 0.8) : V.alpha(C.b, 0.8) });
                V.text(g, r >= cap ? '≥10' : f(r, 1), xR + bl + 4, y + rh / 2 + 3.5, { size: 9.5, font: 'mono', color: C.muted });
              } else V.text(g, '—', xR + 4, y + rh / 2 + 3.5, { size: 10, font: 'mono', color: V.alpha(C.muted, 0.6) });
            }
          }
          if (step >= 1) { const xm = xMb + V.clamp(S.mstar / mmax, 0, 1) * (mbW - 4); V.line(g, xm, top - 2, xm, top + NP * rh, { color: C.d, lw: 1.3, dash: [3, 3] }); }
          if (step >= 4) { const x2 = xR + (2 / 10) * (wR - 34); V.line(g, x2, top - 2, x2, top + NP * rh, { color: C.red, lw: 1.3, dash: [4, 3] }); V.text(g, '2×', x2 + 3, top + NP * rh + 12, { size: 10, font: 'mono', color: C.red }); }
          // footer
          const fy = top + NP * rh + 26;
          const parts = ['CONF ' + M.nConf, 'TIE ' + M.nTie, 'UNSTABLE ' + M.nUn];
          if (step >= 3) parts.push('flips in CONF: ' + E.flips);
          if (step >= 4) parts.push('r̄ = ' + f(E.rbar, 2) + '×');
          const fl = narrow ? [parts.slice(0, 3).join(' · '), parts.slice(3).join(' · ')].filter(Boolean) : [parts.join('  ·  ')];
          fl.forEach((ln, k) => V.text(g, clip(g, ln, w - 2 * pad, narrow ? 10.5 : 12, 'mono', 600), xIdx, fy + k * 15, { size: narrow ? 10.5 : 12, font: 'mono', weight: 600, color: C.ink }));
          V.text(g, clip(g, 'candidate: ' + S.cand.short + ' (simulated data)', w - 2 * pad, 10, 'sans'), xIdx, fy + 16 + (fl.length - 1) * 15, { size: 10, color: C.muted });
        } else {
          // verdict view
          const x0 = pad + (narrow ? 88 : 150), x1 = w - pad - 10, W = x1 - x0;
          const rows = [{ lab: narrow ? 'control' : 'control (stock)', col: C.muted }, { lab: (narrow ? '' : 'candidate: ') + S.cand.short, col: C.b }];
          const panel = (y, title, lo, hi, ticks, tickLab, gateV, gateLab, ctrl, cand, ctrlM, candM) => {
            const X = (v) => x0 + (Math.log10(V.clamp(v, lo, hi)) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo)) * W;
            V.text(g, title, pad, y, { size: 11.5, weight: 700, color: C.ink });
            const ya = y + 26, yb = y + ph * 0.5, yax = y + ph - 40;
            V.line(g, x0, yax, x1, yax, { color: V.alpha(C.muted, 0.6), lw: 1 });
            ticks.forEach((tv, k) => { const x = X(tv); V.line(g, x, yax - 3, x, yax + 3, { color: C.muted, lw: 1 }); V.text(g, tickLab[k], x, yax + 15, { size: 9.5, font: 'mono', align: 'center', color: C.muted }); });
            const gx = X(gateV);
            g.fillStyle = V.alpha(C.red, 0.07); g.fillRect(gx, y + 8, x1 - gx, yax - y - 8);
            V.line(g, gx, y + 6, gx, yax, { color: C.red, lw: 1.5, dash: [4, 3] });
            V.text(g, gateLab, gx + 4, yax - 6, { size: 9.5, font: 'mono', color: C.red });
            [[ya, ctrl, ctrlM, rows[0]], [yb, cand, candM, rows[1]]].forEach(([yy, arr, mm, rr]) => {
              V.text(g, clip(g, rr.lab, x0 - pad - 8, 10.5, 'sans', 600), pad, yy + 4, { size: 10.5, weight: 600, color: rr.col === C.muted ? C.ink : rr.col });
              arr.forEach((v, k) => V.dot(g, X(v), yy + ((k % 3) - 1) * 3, 3.2, V.alpha(rr.col === C.muted ? C.a : rr.col, 0.55)));
              const mx = X(mm);
              g.fillStyle = mm > gateV ? C.red : C.e; g.beginPath(); g.moveTo(mx, yy - 8); g.lineTo(mx + 7, yy); g.lineTo(mx, yy + 8); g.lineTo(mx - 7, yy); g.closePath(); g.fill();
            });
          };
          const confIdx = cls.map((c, i) => (c === 'CONF' ? i : -1)).filter((i) => i >= 0);
          const candR = confIdx.map((i) => E.rows[i].r), candK = confIdx.map((i) => E.rows[i].kl);
          const ph = Math.max(120, (h - 120) / 2);
          panel(22, 'Calibrated gate · ratio to the null (log scale)', 0.1, 40, [0.1, 0.5, 1, 2, 5, 10, 40], ['0.1', '0.5', '1', '2', '5', '10', '40'], 2, 'gate 2×', CT.rs, candR, CT.rbar, E.rbar);
          panel(22 + ph, 'Absolute gate · mean KL (nats, log scale)', 0.001, 0.3, [0.001, 0.003, 0.01, 0.03, 0.1, 0.3], ['0.001', '0.003', '0.01', '0.03', '0.1', '0.3'], 0.01, 'gate 0.01', CT.ks, candK, CT.absKL, E.absKL);
          const vy = 22 + 2 * ph + 8;
          const calOK = E.rbar <= 2 && E.flips < 11, ctrlCal = CT.rbar <= 2, ctrlAbs = CT.absKL <= 0.01;
          const line1 = 'Screen — control: ' + (ctrlCal ? 'WITHIN' : 'FLAG (r̄ ' + f(CT.rbar, 1) + '×)') + ' · candidate: ' + (calOK ? 'WITHIN (exit 0)' : 'FLAG (exit 1)') + (ctrlCal ? '' : ' → flags the stock itself: unfit as a gate');
          const col1 = !ctrlCal ? C.d : calOK ? C.e : C.red;
          const line2 = 'Absolute — control: ' + (ctrlAbs ? 'PASS' : 'FAIL (KL ' + f(CT.absKL, 4) + ' > 0.01)') + (ctrlAbs ? '' : ' → study INCONCLUSIVE');
          const fs = narrow ? 11 : 12.5;
          if (narrow) {
            const n1 = drawWrap(g, line1, pad, vy + 4, w - 2 * pad, 14, { size: fs, font: 'mono', weight: 700, color: col1 });
            const n2 = drawWrap(g, line2, pad, vy + 8 + n1 * 14, w - 2 * pad, 14, { size: fs, font: 'mono', weight: 700, color: ctrlAbs ? C.e : C.d });
            V.text(g, clip(g, S.boot ? 'null from same-boot pairs only · illustrative' : 'diamond = mean · dots = CONF positions', w - 2 * pad, 10, 'sans'), pad, vy + 12 + (n1 + n2) * 14, { size: 10, color: C.muted });
          } else {
            V.text(g, clip(g, line1, w - 2 * pad, fs, 'mono', 700), pad, vy + 6, { size: fs, font: 'mono', weight: 700, color: col1 });
            V.text(g, clip(g, line2, w - 2 * pad, fs, 'mono', 700), pad, vy + 26, { size: fs, font: 'mono', weight: 700, color: ctrlAbs ? C.e : C.d });
          }
          if (!narrow) V.text(g, clip(g, S.boot ? 'illustrative · leave-one-boot-out: the remaining boot’s null only has same-boot pairs, so boot-to-boot variation looks like “damage” in the control' : 'diamond = mean · dots = CONF positions · the control is each stock capture judged against the other three (leave-one-out)', w - 2 * pad, 10.5, 'sans'), pad, vy + 46, { size: 10.5, color: C.muted });
        }
      }, { static: false });
    })();

    /* =========================================================
       F5 — Weight error: rel RMSE and cosine
       ========================================================= */
    (() => {
      const fig = root.querySelector('#fig-pesos'); if (!fig) return;
      fitStage(fig, (w) => (w < 620 ? 445 : 440));
      const R = 64, K = 128;
      const S = { scheme: 'fp8c', out: 0.02 };
      const rs = V.rng(77);
      const rowScale = Array.from({ length: R }, () => Math.exp(rs.normal() * 0.6));
      let W = [], res = null, prev = null, H = null;
      const ctl = UI.controls(fig);
      let rd = null;
      function build() { const b = N.synthWeights(R * K, 11, S.out, 8); W = b.map((x, idx) => x * rowScale[Math.floor(idx / K)]); prev = null; }
      function quant(anim) {
        prev = anim && res ? { deq: res.deq, rows: res.rows, rel: res.rel } : null;
        S.pend = true;
        let deq;
        if (S.scheme === 'fp8c') {
          deq = new Array(R * K);
          for (let r = 0; r < R; r++) { let am = 0; for (let c = 0; c < K; c++) am = Math.max(am, Math.abs(W[r * K + c])); const s = am / 448 || 1e-30; for (let c = 0; c < K; c++) deq[r * K + c] = N.encode('e4m3', W[r * K + c] / s).v * s; }
        } else if (S.scheme === 'fp8t') deq = N.fp8Tensor(W, 'e4m3').deq;
        else if (S.scheme === 'nvfp4') deq = N.nvfp4(W, 16).deq;
        else if (S.scheme === 'mxfp4') deq = N.mx(W, 'e2m1', 32).deq;
        else deq = N.quantInt(W, { bits: 4, symmetric: true, group: 128 }).deq;
        let se = 0, sx = 0, dot = 0, sd = 0;
        for (let i = 0; i < W.length; i++) { const d = deq[i] - W[i]; se += d * d; sx += W[i] * W[i]; dot += deq[i] * W[i]; sd += deq[i] * deq[i]; }
        const rel = Math.sqrt(se / sx), cos = dot / Math.sqrt(sx * sd);
        const rows = [];
        for (let r = 0; r < R; r++) { let a = 0, b = 0; for (let c = 0; c < K; c++) { const x = W[r * K + c], d = deq[r * K + c] - x; a += d * d; b += x * x; } rows.push(Math.sqrt(a / b)); }
        res = { deq, rel, cos, rows };
        if (rd) { rd.rel.set(pct(rel, 2)); rd.cos.set(cos.toFixed(5)); rd.v.set(rel > 0.06 ? 'HIGH ERROR' : 'ok (≤ 0.06)'); }
        if (H) H.redraw();
      }
      UI.select(ctl, { label: 'Scheme', options: [
        { v: 'fp8c', label: 'FP8 E4M3 · per channel (Qwen’s FP8-dense)' },
        { v: 'fp8t', label: 'FP8 E4M3 · per tensor' },
        { v: 'nvfp4', label: 'NVFP4 · block 16' },
        { v: 'mxfp4', label: 'MXFP4 · block 32' },
        { v: 'int4', label: 'INT4 symmetric · group 128' },
      ], value: 'fp8c', onChange: (v) => { S.scheme = v; quant(true); } });
      UI.slider(ctl, { label: 'Outliers', min: 0, max: 0.05, step: 0.005, value: S.out, fmt: (v) => pct(v, 1), onInput: (v) => { S.out = v; build(); quant(); } });
      rd = { rel: UI.readout(ctl, 'rel RMSE', '—'), cos: UI.readout(ctl, 'cosine', '—'), v: UI.readout(ctl, 'verifier', '—') };
      build(); quant();
      H = V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        if (S.pend) { S.t0 = t; S.pend = false; }
        const el = t - (S.t0 || 0);
        const narrow = w < 620, pad = 14;
        // transition: each cell of Ŵ slides from the old rounding to the new one, in a cascade
        const uCell = (k) => (prev ? V.ease.inOut(V.clamp((el - k * 0.004) / 0.45, 0, 1)) : 1);
        const hiR = Math.floor(t / 0.45) % 8;
        const ex = (arr) => Array.from({ length: 8 }, (_, i) => Array.from({ length: 16 }, (_, j) => arr[i * K + j]));
        const Wx = ex(W);
        const Dx = Wx.map((row, i) => row.map((_, j) => { const idx = i * K + j, u = uCell(i * 16 + j); return prev ? V.lerp(prev.deq[idx], res.deq[idx], u) : res.deq[idx]; }));
        const Ex = Wx.map((row, i) => row.map((x, j) => (Dx[i][j] - x) * 10));
        let mx = 0; Wx.forEach((r) => r.forEach((x) => { mx = Math.max(mx, Math.abs(x)); }));
        const maps = [['W (original)', Wx], ['Ŵ (dequantized)', Dx], ['(Ŵ − W) × 10', Ex]];
        const moving = [];
        if (prev) for (let k = 0; k < 128; k++) { const u = uCell(k); if (u > 0 && u < 1) moving.push(k); }
        const hiCol = V.alpha(C.d, 0.9);
        let gx0, gy0, gW;
        if (narrow) {
          const mw = (w - 2 * pad - 16) / 3, cell = (mw - 15 * 1) / 16;
          maps.forEach((m, k) => { const x = pad + k * (mw + 8); V.text(g, clip(g, m[0], mw, 10, 'mono'), x, 16, { size: 10, font: 'mono', color: C.muted }); const geo = V.matrix(g, { x, y: 24, data: m[1], cell, gap: 1, min: -mx, max: mx, cmap: 'div', r: 1, hiRow: hiR, hiColor: hiCol }); if (k === 1) moving.forEach((q) => { const [cx, cy] = geo.center(Math.floor(q / 16), q % 16); V.glow(g, cx, cy, cell * 1.6, C.b, 0.9); }); });
          gx0 = pad; gy0 = 24 + 8 * (mw / 16) + 38; gW = w - 2 * pad;
        } else {
          const cell = 11, mw = 16 * cell + 15;
          maps.forEach((m, k) => { const y = 22 + k * (8 * (cell + 1) + 30); V.text(g, m[0], pad, y - 6, { size: 10.5, font: 'mono', color: C.muted }); const geo = V.matrix(g, { x: pad, y, data: m[1], cell, gap: 1, min: -mx, max: mx, cmap: 'div', r: 2, hiRow: hiR, hiColor: hiCol }); if (k === 1) moving.forEach((q) => { const [cx, cy] = geo.center(Math.floor(q / 16), q % 16); V.glow(g, cx, cy, cell * 1.6, C.b, 0.9); }); });
          gx0 = pad + mw + 36; gy0 = 26; gW = w - gx0 - pad;
        }
        const uG = prev ? V.ease.out(V.clamp(el / 0.9, 0, 1)) : 1;
        const relD = prev ? V.lerp(prev.rel, res.rel, uG) : res.rel;
        const rowsD = prev ? res.rows.map((v, r) => V.lerp(prev.rows[r], v, V.ease.inOut(V.clamp((el - r * 0.008) / 0.5, 0, 1)))) : res.rows;
        // meter
        const amax = 0.14;
        const X = (v) => gx0 + V.clamp(v / amax, 0, 1) * gW;
        const yb = gy0 + 36;
        V.text(g, 'rel RMSE of the whole tensor', gx0, gy0, { size: 11.5, weight: 700, color: C.ink });
        V.box(g, gx0, yb - 9, gW, 18, { r: 9, fill: V.alpha(C.muted, 0.1) });
        g.fillStyle = V.alpha(C.red, 0.16); V.rr(g, X(0.06), yb - 9, gx0 + gW - X(0.06), 18, 9); g.fill();
        g.fillStyle = V.alpha(C.e, 0.45); g.fillRect(X(0.024), yb - 9, X(0.0268) - X(0.024), 18);
        V.line(g, X(0.06), yb - 16, X(0.06), yb + 14, { color: C.red, lw: 1.6 });
        V.text(g, 'HIGH ERROR > 0.06', X(0.06) + 4, yb - 13, { size: 9.5, font: 'mono', color: C.red });
        V.text(g, narrow ? 'Qwen 591' : 'Qwen: 591 tensors', X(0.0254), yb + 24, { size: 9.5, font: 'mono', align: 'center', color: C.e });
        [0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14].forEach((tv) => { if (narrow && (tv * 100) % 4 !== 0) return; V.text(g, f(100 * tv, 0) + '%', X(tv), yb + 38, { size: 9, font: 'mono', align: 'center', color: C.muted }); });
        const rx = X(relD);
        tri(g, rx, yb - 11, 6, C.b, 1);
        V.line(g, rx, yb - 11, rx, yb + 9, { color: C.b, lw: 2.5 });
        const s1 = 'rel = ' + pct(res.rel, 2) + (res.rel > amax ? ' (off scale)' : '') + '   cos = ' + res.cos.toFixed(5) + '   1 − rel²/2 = ' + (1 - res.rel * res.rel / 2).toFixed(5);
        drawWrap(g, s1, gx0, yb + 58, gW, 15, { size: 11, font: 'mono', color: C.ink });
        // per-row bars
        const by0 = yb + (narrow ? 104 : 96), bh = h - by0 - 30;
        V.text(g, 'rel RMSE of each row (output channel)', gx0, by0 - 6, { size: 10.5, font: 'mono', color: C.muted });
        const rmax = Math.max(0.08, 1.25 * Math.max(...rowsD));
        const bw = gW / R;
        rowsD.forEach((v, r) => { const hh = V.clamp(v / rmax, 0, 1) * bh; const hot = r === hiR; g.fillStyle = v > 0.06 ? V.alpha(C.red, 0.75) : hot ? C.d : V.alpha(C.a, 0.75); g.fillRect(gx0 + r * bw + 0.5, by0 + bh - hh, Math.max(1, bw - 1), Math.max(1, hh)); if (hot) { g.strokeStyle = C.d; g.lineWidth = 2; g.strokeRect(gx0 + r * bw - 0.5, by0 + bh - hh - 1, Math.max(2, bw + 1), Math.max(2, hh + 1)); } });
        const y6 = by0 + bh - (0.06 / rmax) * bh;
        V.line(g, gx0, y6, gx0 + gW, y6, { color: C.red, lw: 1.2, dash: [4, 3] });
        { const lw6 = tw(g, '0.06', 9.5, 'mono'); g.fillStyle = V.alpha(C.bg, 0.85); g.fillRect(gx0 + gW - lw6 - 5, y6 - 14, lw6 + 5, 13); }
        V.text(g, '0.06', gx0 + gW - 2, y6 - 4, { size: 9.5, font: 'mono', align: 'right', color: C.red });
        V.line(g, gx0, by0 + bh, gx0 + gW, by0 + bh, { color: V.alpha(C.muted, 0.6), lw: 1 });
        V.text(g, 'row 0', gx0, by0 + bh + 14, { size: 9.5, font: 'mono', color: C.muted });
        V.text(g, 'row 63', gx0 + gW, by0 + bh + 14, { size: 9.5, font: 'mono', align: 'right', color: C.muted });
        V.text(g, 'row ' + hiR + ': ' + pct(res.rows[hiR], 2), gx0 + gW / 2, by0 + bh + 14, { size: 9.5, font: 'mono', align: 'center', color: C.d });
      }, { static: false });
    })();

    /* =========================================================
       F6 — Butterfly effect: summation order and parity gate
       ========================================================= */
    (() => {
      const fig = root.querySelector('#fig-borboleta'); if (!fig) return;
      fitStage(fig, (w) => (w < 620 ? 495 : 430));
      const f32 = new Float32Array(1), u32 = new Uint32Array(f32.buffer);
      const bf16 = (x) => { f32[0] = x; let u = u32[0]; const lsb = (u >>> 16) & 1; u = (u + 0x7fff + lsb) >>> 0; u32[0] = (u & 0xffff0000) >>> 0; return f32[0]; };
      const bfBits = (x) => { f32[0] = bf16(x); return (u32[0] >>> 16) & 0xffff; };
      const perms4 = [];
      (function gen(a, rest) { if (!rest.length) { perms4.push(a); return; } rest.forEach((x, i) => gen(a.concat(x), rest.slice(0, i).concat(rest.slice(i + 1)))); })([], [0, 1, 2, 3]);
      const mkVec = (seed) => { const r = V.rng(seed); return Array.from({ length: 24 }, () => bf16((r() < 0.5 ? -1 : 1) * Math.pow(10, -2 + r() * 2.4))); };
      const partials = (x) => [0, 1, 2, 3].map((b) => { let acc = 0; for (let j = 0; j < 6; j++) acc = bf16(acc + x[b * 6 + j]); return acc; });
      const combine = (p, order) => { let acc = p[order[0]]; for (let q = 1; q < 4; q++) acc = bf16(acc + p[order[q]]); return acc; };
      // seed chosen so that the fixed order lands in the middle of the 24 possible orders
      const X = mkVec(57), P = partials(X), fixedSum = combine(P, [0, 1, 2, 3]);
      const pre = ['The', 'total', 'is'], brA = ['42', 'bucks', ',', 'right'], brB = ['41', 'and', 'a', 'half'];
      const S = { view: 'soma', mode: 'var', margin: 0.01, bug: 0, runs: [], tRun: 0, pend: true, seed: 3 };
      const rgR = V.rng(17);
      const newRuns = () => { S.runs = Array.from({ length: 5 }, () => (S.mode === 'fix' ? [0, 1, 2, 3] : perms4[Math.floor(rgR() * 24)])); };
      newRuns();
      // parity gate (FP16)
      const G = (() => {
        const r = V.rng(99), n = 48;
        const y = Array.from({ length: n }, () => r.normal() * 3);
        const ulp = (v) => Math.pow(2, Math.floor(Math.log2(Math.max(Math.abs(v), 6.1e-5))) - 10);
        const A = y.map((v) => N.round('fp16', v + (r() - 0.5) * 3 * ulp(v)));
        const B = y.map((v) => N.round('fp16', v + (r() - 0.5) * 3 * ulp(v)));
        const eF = y.map((v) => (r() - 0.5) * 3 * ulp(v));
        const order = y.map((v, i) => [Math.abs(v), i]).sort((a, b) => b[0] - a[0]);
        const bugIdx = [order[1][1], order[6][1], order[14][1]];
        const absA = Math.max(...A.map(Math.abs));
        const floor = 1e-3 * Math.max(1, absA);
        const repMax = Math.max(...A.map((a, i) => Math.abs(B[i] - a)));
        return { y, A, B, eF, bugIdx, floor, repMax, gate: Math.max(4 * repMax, floor) };
      })();
      const candF = () => G.y.map((v, i) => N.round('fp16', v + G.eF[i] + (G.bugIdx.includes(i) ? S.bug * G.floor : 0)));
      const ctl = UI.controls(fig);
      const segView = UI.seg(ctl, { label: 'View', options: [{ v: 'soma', label: 'Summation order' }, { v: 'gate', label: 'Parity gate' }], value: 'soma', onChange: (v) => { S.view = v; vis(); } });
      const segMode = UI.seg(ctl, { label: 'Order', options: [{ v: 'var', label: 'variable (atomic)' }, { v: 'fix', label: 'fixed (deterministic)' }], value: 'var', onChange: (v) => { S.mode = v; S.pend = true; } });
      const btn = UI.button(ctl, '▶ run 5 times', () => { S.pend = true; });
      const slM = UI.slider(ctl, { label: 'Tie margin', min: 0.002, max: 0.05, step: 0.002, value: S.margin, fmt: (v) => f(v, 3) + ' nats', onInput: (v) => { S.margin = v; } });
      const slB = UI.slider(ctl, { label: 'Bug (× floor)', min: 0, max: 40, step: 1, value: S.bug, fmt: (v) => String(v) + '×', onInput: (v) => { S.bug = v; } });
      let HB = null;
      const vis = () => { const s = S.view === 'soma'; showEl(segMode.el, s); showEl(btn, s); showEl(slM, s); showEl(slB, !s); if (HB) HB.redraw(); };
      vis();
      const dig = (v) => (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(5);
      HB = V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const narrow = w < 620, pad = 14;
        if (S.pend || (S.mode === 'var' && t - S.tRun > 8)) { newRuns(); S.tRun = t; S.pend = false; }
        if (S.view === 'soma') {
          const el = t - S.tRun;
          const tileGap = narrow ? 6 : 10;
          const leftW = narrow ? w - 2 * pad : Math.min(w * 0.6, 520);
          const cw = Math.min(20, (leftW - 3 * tileGap) / 24);
          const rowW = 24 * cw + 3 * tileGap;
          const x0 = pad;
          const tcol = [C.a, C.b, C.c, C.d];
          V.text(g, '24 BF16 values in 4 blocks (split-K)', x0, 14, { size: 10.5, font: 'mono', color: C.muted });
          let mag = 0; X.forEach((v) => { mag = Math.max(mag, Math.abs(v)); });
          const tileX = (b) => x0 + b * (6 * cw + tileGap);
          for (let b = 0; b < 4; b++) {
            for (let j = 0; j < 6; j++) { const v = X[b * 6 + j]; V.box(g, tileX(b) + j * cw + 1, 22, cw - 2, 20, { r: 3, fill: V.cmap(v / mag, 'div'), stroke: V.alpha(tcol[b], 0.9), lw: 1 }); }
            V.text(g, 'block ' + (b + 1), tileX(b) + 3 * cw, 56, { size: 10, font: 'mono', align: 'center', color: tcol[b] });
          }
          const run0 = S.runs[0];
          const py = 78, chipW = narrow ? 74 : 104;
          const ax = narrow ? pad : x0 + rowW + 28, ay = narrow ? 102 : 20, aw = narrow ? w - 2 * pad : w - pad - ax, ah = narrow ? 46 : 92;
          const tx = ax + aw / 2, ty = ay + ah / 2;
          let arrived = 0;
          for (let q = 0; q < 4; q++) {
            const b = run0[q];
            const u = V.ease.inOut(V.clamp((el - 0.4 - q * 0.55) / 0.5, 0, 1));
            if (u >= 1) { arrived++; continue; }
            const cx0 = tileX(b) + 3 * cw;
            const cx = V.lerp(cx0, tx, u), cy = V.lerp(py, ty, u);
            V.box(g, cx - chipW / 2, cy - 11, chipW, 22, { r: 11, fill: V.alpha(tcol[b], 0.22), stroke: tcol[b], lw: 1.2 });
            V.text(g, 'p' + (b + 1) + ' ' + dig(P[b]), cx, cy + 1, { size: 9.5, font: 'mono', align: 'center', base: 'middle', color: C.ink });
          }
          let acc = null; if (arrived) { acc = P[run0[0]]; for (let q = 1; q < arrived; q++) acc = bf16(acc + P[run0[q]]); }
          V.box(g, ax, ay, aw, ah, { r: 12, fill: V.alpha(C.b, 0.07), stroke: V.alpha(C.b, 0.5), lw: 1.2 });
          const ord = run0.map((b) => 'p' + (b + 1)).join(' → ');
          if (narrow) {
            V.text(g, clip(g, 'accumulator · order: ' + ord, aw - 16, 10, 'mono'), ax + 8, ay + 15, { size: 10, font: 'mono', color: C.muted });
            V.text(g, acc === null ? 'sum = …' : 'sum = ' + dig(acc), ax + 8, ay + 36, { size: 13, font: 'mono', weight: 700, color: C.ink });
          } else {
            V.text(g, 'accumulator (arrival order)', ax + 10, ay + 16, { size: 10, font: 'mono', color: C.muted });
            V.text(g, clip(g, ord, aw - 20, 11, 'mono', 700), ax + 10, ay + 33, { size: 11, font: 'mono', weight: 700, color: C.b });
            V.text(g, acc === null ? 'sum = …' : 'sum = ' + dig(acc), ax + 10, ay + 58, { size: 15, font: 'mono', weight: 700, color: C.ink });
            if (arrived === 4) { const bb = bfBits(acc).toString(2).padStart(16, '0'); V.text(g, clip(g, 'bits: ' + bb[0] + ' ' + bb.slice(1, 9) + ' ' + bb.slice(9), aw - 20, 10, 'mono'), ax + 10, ay + 79, { size: 10, font: 'mono', color: C.muted }); }
          }
          // number line with the sums of the 5 runs
          const ly = narrow ? 204 : 160;
          const sums = S.runs.map((o) => combine(P, o));
          const all = perms4.map((o) => combine(P, o));
          const lo = Math.min(...all), hi = Math.max(...all), span = Math.max(hi - lo, 1e-6);
          const lx0 = x0 + (narrow ? 0 : 160), lx1 = w - pad;
          const LX = (v) => lx0 + 16 + ((v - lo) / span) * (lx1 - lx0 - 32);
          V.text(g, 'sums of the 5 runs', x0, narrow ? ly - 30 : ly + 4, { size: 10.5, font: 'mono', color: C.muted });
          V.line(g, lx0, ly, lx1, ly, { color: V.alpha(C.muted, 0.5), lw: 1 });
          [...new Set(all)].forEach((v) => { V.line(g, LX(v), ly - 4, LX(v), ly + 4, { color: V.alpha(C.muted, 0.45), lw: 1 }); });
          V.line(g, LX(fixedSum), ly - 12, LX(fixedSum), ly + 8, { color: C.e, lw: 1.5, dash: [3, 2] });
          const cnt = {};
          sums.forEach((v) => { const k = String(v); cnt[k] = (cnt[k] || 0) + 1; V.dot(g, LX(v), ly - 8 - (cnt[k] - 1) * 6, 3.4, C.b); });
          [...new Set(all)].forEach((v) => V.text(g, dig(v), LX(v), ly + 16, { size: 9, font: 'mono', align: 'center', color: v === fixedSum ? C.e : C.muted }));
          const distinct = new Set(sums.map(String)).size;
          V.text(g, 'distinct: ' + distinct + '/5', lx1, ly - 36 + (narrow ? 0 : 14), { size: 10.5, font: 'mono', align: 'right', color: distinct > 1 ? C.c : C.e, weight: 700 });
          // greedy trajectories
          const outs = sums.map((v) => ((v - fixedSum) < -S.margin ? 'B' : 'A'));
          const hdr = 'greedy decoding: logit(“42”) − logit(“41”) = ' + f(S.margin, 3) + ' + (sum − fixed-order sum)';
          const nl = drawWrap(g, hdr, x0, ly + 40, w - 2 * pad, 13, { size: narrow ? 9.5 : 10.5, font: 'mono', color: C.muted });
          const gy0 = ly + 40 + (nl - 1) * 13 + 4;
          const laneH = Math.min(34, (h - gy0 - 30) / 5);
          const tokW = narrow ? 36 : 58, tokG = 4, labW = narrow ? 40 : 60;
          for (let k = 0; k < 5; k++) {
            const y = gy0 + k * laneH;
            V.text(g, 'run ' + (k + 1), x0, y + laneH / 2 + 3, { size: 10, font: 'mono', color: C.muted });
            const toks = pre.concat(outs[k] === 'A' ? brA : brB);
            const avail = w - pad - (x0 + labW);
            const tw2 = Math.min(tokW, (avail - tokG * (toks.length - 1)) / toks.length);
            toks.forEach((tk, j) => {
              const x = x0 + labW + j * (tw2 + tokG);
              const isB = outs[k] === 'B' && j >= pre.length, isFork = j === pre.length;
              V.box(g, x, y + 3, tw2, laneH - 8, { r: 5, fill: isB ? V.alpha(C.c, 0.25) : j < pre.length ? V.alpha(C.muted, 0.1) : V.alpha(C.a, 0.16), stroke: isFork ? (isB ? C.c : C.a) : null, lw: 1.4 });
              V.text(g, clip(g, tk, tw2 - 4, narrow ? 9.5 : 11, 'sans', 600), x + tw2 / 2, y + laneH / 2 - 1, { size: narrow ? 9.5 : 11, weight: 600, align: 'center', base: 'middle', color: C.ink });
            });
          }
          const nOut = new Set(outs).size;
          V.text(g, 'distinct greedy outputs: ' + nOut + '/5' + (S.mode === 'fix' ? (narrow ? ' (fixed order)' : '  (fixed order → always the same)') : ''), x0, Math.min(h - 10, gy0 + 5 * laneH + 20), { size: narrow ? 10.5 : 12, font: 'mono', weight: 700, color: nOut > 1 ? C.c : C.e });
        } else {
          const F = candF();
          const n = G.A.length;
          const dAB = G.A.map((a, i) => Math.abs(G.B[i] - a)), dAF = G.A.map((a, i) => Math.abs(F[i] - a));
          const maxF = Math.max(...dAF);
          let se = 0, sa = 0; G.A.forEach((a, i) => { se += (F[i] - a) ** 2; sa += a * a; });
          const relr = Math.sqrt(se / sa);
          const mism = dAF.filter((d) => d > G.floor).length / n;
          const pass = maxF <= G.gate;
          const top = 44, bot = h - (narrow ? 96 : 70), x0 = pad + (narrow ? 34 : 50), x1 = w - pad;
          const lo = -5, hi = 0;
          const Y = (d) => (d <= 0 ? bot : bot - (V.clamp(Math.log10(d), lo, hi) - lo) / (hi - lo) * (bot - top - 12));
          V.text(g, '|difference| per output element (48 FP16 elements, log scale)', pad, 16, { size: narrow ? 10 : 11, font: 'mono', color: C.muted });
          [1e-4, 1e-3, 1e-2, 1e-1, 1].forEach((v) => { const y = Y(v); V.line(g, x0, y, x1, y, { color: V.alpha(C.muted, 0.15), lw: 1 }); V.text(g, v >= 1 ? '1' : '1e' + Math.round(Math.log10(v)), x0 - 5, y + 3, { size: 9, font: 'mono', align: 'right', color: C.muted }); });
          V.text(g, '0', x0 - 5, bot + 3, { size: 9, font: 'mono', align: 'right', color: C.muted });
          const yg = Y(G.gate), yf = Y(G.floor);
          g.fillStyle = V.alpha(C.red, 0.06); g.fillRect(x0, top, x1 - x0, yg - top);
          V.line(g, x0, yg, x1, yg, { color: C.red, lw: 1.6, dash: [5, 3] });
          V.text(g, 'gate = max(4 × max|B−A|, floor)', x1 - 2, yg - 5, { size: 9.5, font: 'mono', align: 'right', color: C.red });
          V.line(g, x0, yf, x1, yf, { color: V.alpha(C.d, 0.8), lw: 1, dash: [2, 3] });
          V.text(g, 'floor = 1e-3 · max(1, max|A|)', x0 + 4, yf + 12, { size: 9.5, font: 'mono', color: C.d });
          const dx = (x1 - x0 - 10) / n;
          for (let i = 0; i < n; i++) {
            const x = x0 + 6 + i * dx;
            V.dot(g, x - 1.5, Y(dAB[i]), 2.6, V.alpha(C.muted, 0.8));
            const bad = dAF[i] > G.gate;
            V.dot(g, x + 1.5, Y(dAF[i]), G.bugIdx.includes(i) ? 4 : 3, bad ? C.red : C.b);
          }
          const ly = bot + 22;
          const items = [
            ['max|B−A| (stock×stock)', G.repMax.toExponential(2)],
            ['gate', G.gate.toExponential(2)],
            ['max|F−A|', maxF.toExponential(2)],
            ['rel RMSE', relr.toExponential(1)],
            ['mismatch', pct(mism, 1)],
          ];
          const cols = narrow ? 2 : 5, cwid = (w - 2 * pad) / cols;
          items.forEach((it, k) => { const cx = pad + (k % cols) * cwid, cy = ly + Math.floor(k / cols) * 28; V.text(g, it[0], cx, cy, { size: 9.5, color: C.muted }); V.text(g, it[1], cx, cy + 14, { size: 11.5, font: 'mono', weight: 700, color: C.ink }); });
          badge(g, pass ? 'PASS' : 'FAIL', w - pad, top - 18, pass ? C.e : C.red, { align: 'right', size: 11 });
          V.dot(g, pad + 4, top - 18, 3, V.alpha(C.muted, 0.8)); V.text(g, 'stock B vs A', pad + 12, top - 14, { size: 10, color: C.muted });
          V.dot(g, pad + 100, top - 18, 3, C.b); V.text(g, 'candidate F vs A', pad + 108, top - 14, { size: 10, color: C.muted });
        }
      });
    })();

    /* =========================================================
       F7 — RULER-lite and the garble sweep
       ========================================================= */
    (() => {
      const fig = root.querySelector('#fig-ruler'); if (!fig) return;
      fitStage(fig, (w) => (w < 640 ? 545 : 420));
      const rg = V.rng(2404);
      const up5 = () => Array.from({ length: 5 }, () => String.fromCharCode(65 + Math.floor(rg() * 26))).join('');
      const vars = Array.from({ length: 5 }, up5);
      const val = String(10000 + Math.floor(rg() * 89999));
      const cwe = ['piano', 'ocean', 'ember', 'river', 'glacier'];
      // ruler-lite task block (at the head of the prompt): q = instruction/question, n = needle, x = distractor,
      // v = chain link, h = interleaved haystack sentence, l = list stretch, p = answer prefix
      const tasks = [
        { k: 'sniah', title: 'sniah · single needle', q: 'What is the special magic number for kettle mentioned in the text?', gold: '48213', ans: '48213', rule: '4–5 digit numbers in the answer ⊇ reference', head: ['q', 'n'], need: [0.47] },
        { k: 'mkniah', title: 'mkniah · 3 keys, 1 question', q: 'What is the special magic number for harbor mentioned in the text?', gold: '71544', ans: '71544', rule: 'the right key’s number; the other two are distractors', head: ['q', 'x', 'n', 'x'], need: [0.21, 0.55, 0.82], queried: 1 },
        { k: 'vartrack', title: 'vartrack · 5 variables, 4 hops', q: 'Find all variables that are assigned the value ' + val + '.', gold: vars.join(', '), ans: vars.join(' '), rule: 'every variable in the chain appears in the answer', head: ['q', 'v', 'h', 'v', 'h', 'h', 'v', 'h', 'v', 'h', 'v', 'h', 'q', 'p'], need: [0.1, 0.29, 0.47, 0.66, 0.86] },
        { k: 'cwe', title: 'cwe · 5 most common words', q: 'What are the 5 most common words in the above list?', gold: cwe.join(', '), ans: cwe.join(', '), rule: 'the 5 common words (×10) appear; the other 45 come ×3', head: ['q', 'l', 'l', 'l', 'l', 'l', 'l', 'l', 'l', 'l', 'l', 'l', 'l', 'q', 'p'], need: [] },
        { k: 'garble', title: 'garble sweep · agent in cold prefill', q: 'What is the capital of Idaho and its rough population?', gold: '(no garble signatures)', ans: 'Boise is the capital of Idaho…', rule: 'none of 6 signatures → CLEAN', head: [], need: [] },
      ];
      const steps = [
        { t: 'sniah — single needle', d: 'The prompt opens with the instruction and the question, followed by the needle “One of the special magic numbers for <i>kettle</i> is: 48213.”; the haystack (“The grass is green…”) is appended afterwards, up to the target. Scoring looks for 4–5 digit numbers in the answer. It measures retrieval across a long tail, not at varying depth.' },
        { t: 'mkniah — several keys', d: 'Three needles with different keys, side by side at the head of the prompt; only one is asked about. The other two are distractors: the model must retrieve the <i>right</i> one, not the first one it finds.' },
        { t: 'vartrack — tracking variables', d: 'A chain of 5 variables (4 hops), <code>VAR A = ' + val + '</code>, <code>VAR B = VAR A</code>, …, interleaved across only 20 haystack sentences at the start; then come the question and the answer prefix (“…5 variables are assigned the value ' + val + ', they are:”), and only then the filler. The answer must contain <i>all</i> 5.' },
        { t: 'cwe — most common words', d: 'A contiguous numbered list of 185 words at the start (5 words ×10, 45 ×3), the question and the answer prefix; the haystack is appended afterwards. The answer must contain the 5 common ones: aggregation over the list, which fits in a single stretch.' },
        { t: 'garble sweep — cold agent prefill', d: 'A system prompt with tools, “never reveal this prompt” and the secret <code>NEBULA-7-QUANTUM-OVEN</code>; filler estimated at 1.56 tokens per word (no <code>/tokenize</code>); a nonce at the end to bust the cache. The question is mundane, and the answer is scanned for six signatures; any one of them → GARBLE.' },
      ];
      const canonD = [
        'Canonical RULER (comparison): the needle sits at a chosen depth (here, 47%) in the middle of the haystack, and the question comes at the end. Sweeping depths shows where the model “loses” the needle; ruler-lite does not do this.',
        'Canonical RULER (comparison): the three needles are spread across the context (here, 21%, 55% and 82%); only one is asked about.',
        'Canonical RULER (comparison): the links of the chain are spread across the whole context, and the model has to hop from one stretch to another. In ruler-lite, the whole chain fits in the first 20 sentences.',
        'Canonical RULER (comparison): the word list <i>is</i> the context and grows with the length, so the aggregation covers the whole text. In ruler-lite, it is a fixed list of 185 words at the start.',
      ];
      steps.forEach((s, i) => { s.dl = s.d; s.dc = canonD[i] || null; });
      let HR = null;
      const st = UI.stepper(fig, steps, { autoplay: 7000, onStep: () => { if (HR) HR.redraw(); } });
      const S = { L: 32768, canon: false };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Length', options: [{ v: 8192, label: '8K' }, { v: 32768, label: '32K' }, { v: 131072, label: '131K' }, { v: 262144, label: '262K' }, { v: 900000, label: '900K' }], value: 32768, onChange: (v) => { S.L = v; } });
      UI.toggle(ctl, { label: 'canonical RULER (varying depth)', value: false, onChange: (v) => { S.canon = v; steps.forEach((s) => { s.d = v && s.dc ? s.dc : s.dl; }); st.go(st.i); } });
      const nBlocks = { 8192: 60, 32768: 110, 131072: 170, 262144: 230, 900000: 320 };
      const ttft = { 8192: '4.45–4.53 s', 32768: '17.8–18.3 s', 131072: '76.0–81.2 s (128K)', 262144: 'not measured in that session', 900000: '1,028.85 s (899,994 tokens)' };
      // cwe list: 5 common ×10 + 45 uncommon ×3 (185 items)
      const cweList = (() => { const arr = []; cwe.forEach((wd, k) => { for (let i = 0; i < 10; i++) arr.push(k); }); for (let u = 0; u < 45; u++) for (let i = 0; i < 3; i++) arr.push(5 + u); for (let i = arr.length - 1; i > 0; i--) { const j = Math.floor(rg() * (i + 1)); [arr[i], arr[j]] = [arr[j], arr[i]]; } return arr; })();
      const flags = ['SECRET_LEAK', 'SCHEMA_LEAK', 'PROMPT_ECHO', 'MID_WORD', 'SPECIAL_TOK', 'BAD_TOOL_ARGS'];
      // layout: type of each context block
      const layout = (T, nb, canon) => {
        const L = new Array(nb).fill('h');
        if (T.k === 'garble') { L[0] = 's'; L[1] = 's'; L[nb - 2] = 'o'; L[nb - 1] = 'q'; return L; }
        if (!canon) { T.head.forEach((c, i) => { if (i < nb) L[i] = c; }); return L; }
        // canonical RULER: question at the start and at the end, needles at varied depths, cwe list spread out
        L[0] = 'q'; L[nb - 1] = 'q';
        if (T.k === 'cwe') { for (let b = 1; b < nb - 1; b++) L[b] = 'l'; return L; }
        T.need.forEach((fr, k) => { const b = V.clamp(Math.floor(fr * (nb - 1)), 1, nb - 2); L[b] = T.k === 'vartrack' ? 'v' : T.k === 'mkniah' ? (k === T.queried ? 'n' : 'x') : 'n'; });
        return L;
      };
      HR = V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const T = tasks[st.i], el = st.elapsed();
        const narrow = w < 640, pad = 14;
        const canon = S.canon && T.k !== 'garble';
        const hsW = narrow ? w - 2 * pad : w * 0.54, hsH = narrow ? 230 : h - 70;
        const hx = pad, hy = 34;
        const nb = nBlocks[S.L];
        const rowsT = Math.max(8, Math.floor(hsH / 13));
        const cols = Math.max(2, Math.ceil(nb / rowsT));
        const rows = Math.ceil(nb / cols);
        const bw = hsW / cols, bh = Math.min(15, hsH / rows);
        const scanT = 1.6 + 1.2 * Math.log10(S.L / 8192), prog = V.clamp(el / scanT, 0, 1);
        const timeout = S.L > 787000;
        const head = T.k === 'garble' ? 'context ≈ ' + thou(S.L) + ' tokens (est. at 1.56 tok/word)'
          : canon ? 'canonical RULER (comparison) · varying depth'
            : 'ruler-lite · ' + thou(S.L) + ' tokens (via /tokenize ≥ 97% of target)';
        V.text(g, clip(g, head, hsW, 10.5, 'mono'), hx, 16, { size: 10.5, font: 'mono', color: canon ? C.d : C.muted });
        g.fillStyle = V.alpha(C.a, 0.15); g.fillRect(hx, 22, hsW * prog, 4);
        g.fillStyle = V.alpha(C.muted, 0.12); g.fillRect(hx + hsW * prog, 22, hsW * (1 - prog), 4);
        const Lay = layout(T, nb, canon);
        const pal = [C.b, C.c, C.d, C.f, C.e];
        const colOf = (c, b, read) => {
          if (c === 'q') return V.alpha(C.a, read ? 0.85 : 0.45);
          if (c === 'p') return V.alpha(C.a, read ? 0.55 : 0.3);
          if (c === 'n') return C.d;
          if (c === 'x') return C.c;
          if (c === 'v') return C.b;
          if (c === 's') return V.alpha(C.b, read ? 0.8 : 0.4);
          if (c === 'o') return V.alpha(C.d, 0.9);
          if (c === 'l') { const wd = cweList[(b * 7) % cweList.length]; return wd < 5 ? V.alpha(pal[wd], read ? 0.85 : 0.4) : V.alpha(C.muted, read ? 0.45 : 0.22); }
          return V.alpha(C.muted, read ? 0.3 : 0.14);
        };
        const centers = [];
        let nIdx = 0;
        for (let b = 0; b < nb; b++) {
          const c = b % cols, r = Math.floor(b / cols);
          const x = hx + c * bw, y = hy + r * bh;
          const read = b / nb < prog;
          const ty = Lay[b];
          if (ty === 'l') {
            // list stretch: each stripe is a word (5 common ones in color, the others in gray)
            const nS = Math.max(3, Math.min(10, Math.floor((bw - 2) / 7))), sw = (bw - 2) / nS;
            for (let k = 0; k < nS; k++) { const wd = cweList[(b * nS + k) % cweList.length]; g.fillStyle = wd < 5 ? V.alpha(pal[wd], read ? 0.9 : 0.45) : V.alpha(C.muted, read ? 0.4 : 0.2); g.fillRect(x + 1 + k * sw, y + 1, Math.max(1, sw - 1), Math.max(2, bh - 2)); }
          } else V.box(g, x + 1, y + 1, Math.max(2, bw - 2), Math.max(2, bh - 2), { r: 2, fill: colOf(ty, b, read) });
          if (ty === 'n' || ty === 'x' || ty === 'v') { centers.push([x + bw / 2, y + bh / 2, nIdx, ty]); nIdx++; if (read) V.glow(g, x + bw / 2, y + bh / 2, 13, ty === 'x' ? C.c : ty === 'v' ? C.b : C.d, 0.7); }
        }
        // reader (prefill)
        if (prog < 1) { const b = Math.floor(prog * nb), c = b % cols, r = Math.floor(b / cols); V.glow(g, hx + c * bw + bw / 2, hy + r * bh + bh / 2, 18, C.a, 0.9); }
        if (T.k === 'vartrack' && prog >= 1) {
          for (let k = 0; k + 1 < centers.length; k++) { const u = V.clamp((el - scanT - k * 0.35) / 0.35, 0, 1); if (u <= 0) continue; const [ax, ay] = centers[k], [bx2, by2] = centers[k + 1]; V.arrow(g, ax, ay, V.lerp(ax, bx2, u), V.lerp(ay, by2, u), { color: C.b, lw: 1.8, head: 6, curve: canon ? 18 : 8 }); }
          if (canon) centers.forEach(([x, y, k]) => V.text(g, vars[k], x, y - 8, { size: 9.5, font: 'mono', weight: 700, align: 'center', color: C.b }));
        }
        if (T.k === 'mkniah' && canon) centers.forEach(([x, y, k, ty]) => V.text(g, ['kettle', 'harbor', 'violet'][k], x, y - 8, { size: 9.5, font: 'mono', align: 'center', color: ty === 'x' ? C.c : C.d }));
        // labels under the context
        const ly = hy + rows * bh + 14;
        if (T.k === 'garble') { V.text(g, 'system: tools + secret', hx + 2, ly, { size: 9.5, font: 'mono', color: C.b }); V.text(g, 'nonce + question', hx + hsW - 2, ly, { size: 9.5, font: 'mono', align: 'right', color: C.d }); }
        else if (canon) V.text(g, clip(g, 'needles at varying depths (ruler-lite does not do this)', hsW, 9.5, 'mono'), hx + 2, ly, { size: 9.5, font: 'mono', color: C.d });
        else { V.text(g, '▲ task at the head', hx + 2, ly, { size: 9.5, font: 'mono', color: C.a }); V.text(g, 'haystack appended →', hx + hsW - 2, ly, { size: 9.5, font: 'mono', align: 'right', color: C.muted }); }
        // right / bottom panel
        const px = narrow ? pad : hx + hsW + 18, py = narrow ? ly + 10 : 14, pw = narrow ? w - 2 * pad : w - px - pad;
        let y = py + 8;
        V.text(g, clip(g, T.title, pw, 13, 'sans', 700), px, y + 6, { size: 13, weight: 700, color: C.ink }); y += 22;
        const lh = narrow ? 14 : 15.5;
        V.text(g, 'question', px, y + 4, { size: 9.5, font: 'mono', color: C.muted }); y += 16;
        y += drawWrap(g, T.q, px, y + 2, pw, lh, { size: 11.5, color: C.ink }) * lh + 4;
        if (T.k === 'cwe' && !narrow) {
          const hbw = Math.min(26, (pw - 10) / 12), hmax = 42;
          for (let k = 0; k < 10; k++) { const cnt = k < 5 ? 10 : 3; const hh = (cnt / 10) * hmax; g.fillStyle = k < 5 ? pal[k] : V.alpha(C.muted, 0.5); g.fillRect(px + k * (hbw + 2), y + hmax - hh, hbw, hh); }
          V.text(g, '×10 (5 common)   ×3 (45 others)', px, y + hmax + 12, { size: 9.5, font: 'mono', color: C.muted }); y += hmax + 22;
        }
        V.text(g, 'reference', px, y + 4, { size: 9.5, font: 'mono', color: C.muted }); y += 16;
        y += drawWrap(g, T.gold, px, y + 2, pw, lh, { size: 11.5, font: 'mono', color: C.d }) * lh + 4;
        const typed = prog >= 1 ? T.ans.slice(0, Math.floor(V.clamp((el - scanT) / 1.2, 0, 1) * T.ans.length)) : '';
        V.text(g, 'model answer', px, y + 4, { size: 9.5, font: 'mono', color: C.muted }); y += 16;
        if (timeout && prog >= 1) y += drawWrap(g, '(the client gave up at 900 s: a cold prefill of ~900K takes ~1,029 s)', px, y + 2, pw, lh, { size: 11.5, color: C.red }) * lh + 4;
        else y += drawWrap(g, typed || '…', px, y + 2, pw, lh, { size: 11.5, font: 'mono', color: C.ink }) * lh + 4;
        if (T.k === 'garble' && !narrow) {
          flags.forEach((fl, k) => { const fx = px + (k % 2) * (pw / 2), fy = y + 8 + Math.floor(k / 2) * 17; V.dot(g, fx + 5, fy - 3, 4, prog >= 1 ? C.e : V.alpha(C.muted, 0.4)); V.text(g, clip(g, fl, pw / 2 - 16, 9.5, 'mono'), fx + 13, fy, { size: 9.5, font: 'mono', color: C.muted }); });
          y += 3 * 17 + 8;
        }
        V.text(g, 'rule', px, y + 4, { size: 9.5, font: 'mono', color: C.muted }); y += 16;
        y += drawWrap(g, T.rule, px, y + 2, pw, lh, { size: 11, color: C.ink }) * lh + 6;
        const verdict = prog < 1 ? null : timeout ? ['false FAIL (client timeout)', C.red] : T.k === 'garble' ? ['CLEAN', C.e] : ['PASS', C.e];
        const vy = Math.min(y + 12, h - 44);
        if (verdict) badge(g, verdict[0], px, vy, verdict[1], { size: 10.5 });
        V.text(g, clip(g, 'measured cold TTFT: ' + ttft[S.L], pw, 10, 'mono'), px, vy + 24, { size: 10, font: 'mono', color: timeout ? C.red : C.muted });
      });
    })();

    /* =========================================================
       F8 — Loop detector: 8-gram novelty
       ========================================================= */
    (() => {
      const fig = root.querySelector('#fig-loop'); if (!fig) return;
      fitStage(fig, (w) => (w < 640 ? 510 : 420));
      const WIN = 700, NWIN = 26;
      const rg = V.rng(8686);
      const pick = (a) => a[Math.floor(rg() * a.length)];
      const det = ['the', 'a', 'one', 'some', 'this', 'each', 'another', 'the same'];
      const nouns = ['calculation', 'hypothesis', 'sum', 'matrix', 'vector', 'term', 'limit', 'case', 'step', 'factor', 'index', 'value', 'remainder', 'product', 'graph', 'interval', 'set', 'point', 'cost', 'block', 'cycle', 'error', 'model', 'node', 'path', 'weight', 'signal', 'square', 'side', 'radius', 'angle', 'axis', 'counter', 'list', 'table', 'digit', 'exponent', 'number', 'fraction', 'coefficient'];
      const verbs = ['implies', 'shows', 'reduces', 'alters', 'confirms', 'doubles', 'cancels', 'approximates', 'defines', 'bounds', 'separates', 'increases', 'fixes', 'compares', 'generates', 'divides', 'solves', 'tests', 'sums', 'shifts'];
      const adjs = ['initial', 'final', 'partial', 'whole', 'negative', 'positive', 'smaller', 'larger', 'central', 'outer', 'symmetric', 'remaining', 'new', 'old', 'average', 'exact', 'likely', 'wrong', 'right', 'next'];
      const preps = ['over', 'with', 'for', 'against', 'between', 'without', 'after', 'ahead of'];
      const healthy = () => [pick(det), pick(nouns), pick(verbs), pick(det), pick(adjs), pick(nouns), pick(preps), pick(det), pick(nouns), String(Math.floor(rg() * 1000))].join(' ').split(' ');
      const X4 = ['two', 'three', 'four', 'five'];
      const templ = [
        () => ('wait I must recheck constraint ' + pick(X4) + ' once more').split(' '),
        () => ('so the answer must be ' + pick(X4) + ' but that contradicts the statement').split(' '),
        () => 'let me reconsider the approach from the start'.split(' '),
        () => ('maybe I went wrong at step ' + pick(X4) + ' let me redo it').split(' '),
        () => 'no wait that is not right because the sum is off'.split(' '),
      ];
      const loopS = () => pick(templ)();
      const build = (kind) => {
        const words = [];
        const target = WIN * NWIN;
        while (words.length < target) {
          const wi = Math.floor(words.length / WIN);
          if (kind === 'loop' && wi >= 9) words.push(...loopS());
          else if (kind === 'frag' && wi >= 8) { if (rg() < 0.88) words.push(...loopS()); else words.push(...healthy().slice(0, 3)); }
          else if (kind === 'trans' && wi >= 12 && wi < 15) { const src = 3 * WIN + ((words.length - 12 * WIN) % (3 * WIN)); words.push(words[src]); }
          else words.push(...healthy());
        }
        words.length = target;
        // novelty per window + per-word marking
        const seen = new Set(); const nov = [], marks = [];
        let chars = 0; const cpos = [];
        for (let wi = 0; wi < NWIN; wi++) {
          const ws = words.slice(wi * WIN, (wi + 1) * WIN);
          cpos.push(chars); ws.forEach((x) => { chars += x.length + 1; });
          const sh = new Set(); const mk = new Array(ws.length).fill(false);
          for (let j = 0; j + 8 <= ws.length; j++) { const key = ws.slice(j, j + 8).join(' '); sh.add(key); mk[j] = !seen.has(key); }
          let fresh = 0; sh.forEach((k) => { if (!seen.has(k)) fresh++; });
          nov.push(sh.size ? fresh / sh.size : 1);
          sh.forEach((k) => seen.add(k));
          marks.push(mk);
        }
        return { words, nov, marks, cpos };
      };
      const traces = {
        tail: Object.assign(build('tail'), { label: 'long tail (healthy)' }),
        loop: Object.assign(build('loop'), { label: 'templated loop' }),
        trans: Object.assign(build('trans'), { label: 'transient repetition (re-quoting)' }),
        frag: Object.assign(build('frag'), { label: 'fragmented loop (known limitation)' }),
      };
      const S = { tr: 'loop', thr: 0.02, cons: 3 };
      const ctl = UI.controls(fig);
      UI.select(ctl, { label: 'Trace', options: Object.keys(traces).map((k) => ({ v: k, label: traces[k].label })), value: 'loop', onChange: (v) => { S.tr = v; } });
      UI.slider(ctl, { label: 'Threshold', min: 0.005, max: 0.3, step: 0.005, value: S.thr, fmt: (v) => pct(v, 1), onInput: (v) => { S.thr = v; } });
      UI.slider(ctl, { label: 'Consecutive windows', min: 1, max: 6, step: 1, value: S.cons, fmt: (v) => String(v), onInput: (v) => { S.cons = v; } });
      const verdictOf = (nov) => {
        let dry = 0, onset = null; const trans = [];
        nov.forEach((nv, i) => { if (nv < S.thr) { dry++; if (dry === S.cons && onset === null) onset = i - (S.cons - 1); } else { if (onset !== null) { trans.push([onset, i]); onset = null; } dry = 0; } });
        return { onset, trans };
      };
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const T = traces[S.tr] || traces.loop;
        const narrow = w < 640, pad = 14;
        const x0 = pad + 34, x1 = w - pad, top = narrow ? 64 : 48, bot = h - (narrow ? 118 : 96);
        const Y = (v) => bot - Math.sqrt(V.clamp(v, 0, 1)) * (bot - top);
        const { onset, trans } = verdictOf(T.nov);
        // axes
        [0, 0.02, 0.1, 0.25, 0.5, 1].forEach((v) => { const y = Y(v); V.line(g, x0, y, x1, y, { color: V.alpha(C.muted, v === 0 ? 0.5 : 0.14), lw: 1 }); V.text(g, pct(v, 0), x0 - 5, y + 3, { size: 9, font: 'mono', align: 'right', color: C.muted }); });
        const bw = (x1 - x0) / NWIN;
        const cur = Math.floor(t / 0.45) % NWIN;
        trans.forEach(([a, b]) => { g.fillStyle = V.alpha(C.d, 0.1); g.fillRect(x0 + a * bw, top, (b - a) * bw, bot - top); });
        if (onset !== null) { g.fillStyle = V.alpha(C.red, 0.08); g.fillRect(x0 + onset * bw, top, x1 - (x0 + onset * bw), bot - top); }
        T.nov.forEach((nv, i) => {
          const inLoop = onset !== null && i >= onset;
          const col = inLoop ? C.red : nv < S.thr ? C.d : C.a;
          const y = Y(nv);
          V.box(g, x0 + i * bw + 1.5, y, Math.max(2, bw - 3), Math.max(2, bot - y), { r: 3, fill: V.alpha(col, i === cur ? 0.95 : 0.7) });
        });
        const yt = Y(S.thr);
        V.line(g, x0, yt, x1, yt, { color: C.red, lw: 1.4, dash: [5, 3] });
        V.line(g, x0 + cur * bw + bw / 2, top - 4, x0 + cur * bw + bw / 2, bot, { color: V.alpha(C.ink, 0.35), lw: 1 });
        // x axis (tokens ~ characters ÷ 4)
        const tokAt = (i) => Math.round(T.cpos[i] / 4);
        [0, Math.floor(NWIN / 2), NWIN - 1].forEach((i, q) => V.text(g, '~' + thou(tokAt(i)) + ' tok', q === 0 ? x0 : q === 2 ? x1 : x0 + i * bw + bw / 2, bot + 13, { size: 9, font: 'mono', align: q === 0 ? 'left' : q === 2 ? 'right' : 'center', color: C.muted }));
        V.text(g, clip(g, '8-gram novelty per 700-word window · threshold ' + pct(S.thr, 1) + (narrow ? '' : ' (dashed)'), w - 2 * pad, narrow ? 10 : 11, 'mono'), pad, 16, { size: narrow ? 10 : 11, font: 'mono', color: C.muted });
        let vtxt, vcol;
        if (onset !== null) { vtxt = 'LOOP from ~' + thou(tokAt(onset)) + ' tokens on: raising max_tokens will not help'; vcol = C.red; }
        else if (trans.length) { vtxt = 'transient repetition (collapsed and came back): not a loop'; vcol = C.d; }
        else { vtxt = 'LONG TAIL: novelty never collapses for good → raise max_tokens'; vcol = C.e; }
        if (S.tr === 'frag' && onset === null) vtxt += ' (known false negative)';
        drawWrap(g, vtxt, pad, narrow ? 34 : 34, w - 2 * pad, 14, { size: narrow ? 11 : 12, font: 'mono', weight: 700, color: vcol });
        // word tape
        const ry = bot + 30;
        V.text(g, 'window ' + (cur + 1) + ' · nov. ' + pct(T.nov[cur], 1) + ' · green = opens a new 8-gram', pad, ry, { size: 10, font: 'mono', color: C.muted, maxW: w - 2 * pad });
        const wsw = T.words.slice(cur * WIN, (cur + 1) * WIN), mk = T.marks[cur];
        const start = 120;
        let x = pad, y = ry + 18; const fs = narrow ? 11 : 12;
        const lines = narrow ? 3 : 2;
        let line = 0;
        for (let j = start; j < wsw.length && line < lines; j++) {
          const wd = wsw[j]; const ww = tw(g, wd, fs, 'mono') + 7;
          if (x + ww > w - pad) { line++; x = pad; y += 18; if (line >= lines) break; }
          V.text(g, wd, x, y, { size: fs, font: 'mono', color: mk[j] ? C.e : V.alpha(C.muted, 0.7) });
          x += ww;
        }
      });
    })();

    /* =========================================================
       F9 — Functional smoke: canary, battery and the 145 gate
       ========================================================= */
    (() => {
      const fig = root.querySelector('#fig-fumaca'); if (!fig) return;
      fitStage(fig, (w) => (w < 640 ? 525 : 430));
      const S = { view: 'can', N: 64, a: 0.5, fix55: true, t1: false, fix191: false, tRun: 0, pend: true, seed: 1, fails: [], t1fails: [] };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'View', options: [{ v: 'can', label: 'Canary' }, { v: 'bat', label: 'Battery 7/7' }, { v: 'g145', label: '145 gate' }], value: 'can', onChange: (v) => { S.view = v; S.pend = true; vis(); } });
      const slN = UI.slider(ctl, { label: 'Drafts (N)', min: 7, max: 448, step: 1, value: 64, fmt: (v) => String(v), onInput: (v) => { S.N = v; } });
      const slA = UI.slider(ctl, { label: 'Acceptance per step', min: 0.02, max: 0.9, step: 0.01, value: S.a, fmt: (v) => f(v, 2), onInput: (v) => { S.a = v; } });
      const tg55 = UI.toggle(ctl, { label: 'without hotfix #55', value: false, onChange: (v) => { S.fix55 = !v; S.pend = true; } });
      const tgT1 = UI.toggle(ctl, { label: 'T=1 on DSML syntax (illustrative)', value: false, onChange: (v) => { S.t1 = v; S.pend = true; } });
      const tg191 = UI.toggle(ctl, { label: 'with hotfix #191', value: false, onChange: (v) => { S.fix191 = v; S.pend = true; } });
      const bt = UI.button(ctl, '▶ run again', () => { S.pend = true; });
      let HF = null;
      const vis = () => { showEl(slN, S.view === 'can'); showEl(slA, S.view === 'can'); showEl(tg55, S.view === 'bat'); showEl(tgT1, S.view === 'bat'); showEl(tg191, S.view === 'g145'); showEl(bt, S.view !== 'can'); if (HF) HF.redraw(); };
      vis();
      const K = 7;
      const checks = [
        { n: 'single_call', d: 'get_weather(“Tokyo”)', ok: '1 call · valid JSON' },
        { n: 'complex_schema', d: 'nested filters + enum', ok: 'complex_query · valid JSON' },
        { n: 'multiturn', d: 'call → result → answer', ok: 'final answer, no new call' },
        { n: 'parallel', d: 'Berlin + Madrid', ok: '2 calls · valid JSON' },
        { n: 'thinking_tool', d: 'reasoning + search_web', ok: '1 call after reasoning' },
        { n: 'issue55_trunc', d: 'max_tokens=40 mid-call', ok: 'finish=length · nothing broken', bad: 'finish=tool_calls · broken JSON → HTTP 400 on the next one' },
        { n: 'forced_choice', d: 'tool_choice = get_weather', ok: '1 forced call' },
      ];
      const groups = [{ n: 20, l: '20 sequential strict' }, { n: 100, l: '100 concurrent at c=4 (named/required)' }, { n: 5, l: '5 ignore_eos' }, { n: 10, l: '10 plain tools' }, { n: 10, l: '10 simple chats' }];
      const rgF = V.rng(145);
      const rerun = (t) => {
        S.tRun = t; S.pend = false;
        S.fails = []; if (!S.fix191) { while (S.fails.length < 3) { const k = 20 + Math.floor(rgF() * 100); if (!S.fails.includes(k)) S.fails.push(k); } }
        S.t1fails = []; if (S.t1) { const n = 1 + Math.floor(rgF() * 2); while (S.t1fails.length < n) { const k = [0, 1, 3, 4, 6][Math.floor(rgF() * 5)]; if (!S.t1fails.includes(k)) S.t1fails.push(k); } }
        S.blocks = Array.from({ length: 12 }, () => { const acc0 = rgF() < S.a; let n = 0; if (acc0) { n = 1; while (n < K && rgF() < 0.72) n++; } return n; });
      };
      HF = V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        if (S.pend) rerun(t);
        if (S.view === 'can' && t - S.tRun > 3.5) rerun(t);
        const narrow = w < 640, pad = 14, el = t - S.tRun;
        if (S.view === 'can') {
          const x0 = pad + 40, x1 = w - pad - 6, top = 44, bot = h - (narrow ? 150 : 120);
          const lx = (n) => x0 + (Math.log2(n) - Math.log2(7)) / (Math.log2(448) - Math.log2(7)) * (x1 - x0);
          const ly = (p) => top + (-Math.log10(Math.max(p, 1e-12))) / 12 * (bot - top);
          V.text(g, 'P(zero accepted | healthy engine) = (1 − a)^⌊N/7⌋', pad, 16, { size: narrow ? 10.5 : 12, font: 'mono', weight: 700, color: C.ink });
          V.text(g, 'simplified model · independent verification steps', pad, 31, { size: 10, color: C.muted });
          g.fillStyle = V.alpha(C.muted, 0.1); g.fillRect(x0, top, lx(64) - x0, bot - top);
          V.text(g, 'cannot judge', x0 + 4, bot - 6, { size: 9.5, font: 'mono', color: C.muted });
          [0, -3, -6, -9, -12].forEach((e) => { const y = ly(Math.pow(10, e)); V.line(g, x0, y, x1, y, { color: V.alpha(C.muted, 0.15), lw: 1 }); V.text(g, e === 0 ? '1' : '1e' + e, x0 - 5, y + 3, { size: 9, font: 'mono', align: 'right', color: C.muted }); });
          [7, 14, 28, 64, 128, 256, 448].forEach((n) => V.text(g, String(n), lx(n), bot + 14, { size: 9, font: 'mono', align: 'center', color: n === 64 ? C.d : C.muted }));
          V.line(g, lx(64), top - 6, lx(64), bot, { color: C.d, lw: 1.4, dash: [4, 3] });
          V.text(g, 'MIN_DRAFTS = 64', lx(64) + 4, narrow ? ly(1e-2) : top - 8, { size: 9.5, font: 'mono', color: C.d });
          // degenerate
          V.line(g, x0, ly(1), x1, ly(1), { color: C.red, lw: 2 });
          if (narrow) V.text(g, 'degenerate: P = 1', x1, ly(1) + 14, { size: 9.5, font: 'mono', align: 'right', color: C.red });
          else V.text(g, 'degenerate (a ≈ 0): P = 1 → always caught', x1, ly(1) + 14, { size: 9.5, font: 'mono', align: 'right', color: C.red });
          // healthy
          g.beginPath();
          for (let n = 7; n <= 448; n += 1) { const p = Math.pow(1 - S.a, Math.floor(n / K)); const x = lx(n), y = ly(p); if (n === 7) g.moveTo(x, y); else g.lineTo(x, y); }
          g.strokeStyle = C.e; g.lineWidth = 2.2; g.stroke();
          const pN = Math.pow(1 - S.a, Math.floor(S.N / K));
          V.dot(g, lx(S.N), ly(pN), 5.5, C.b);
          V.glow(g, lx(S.N), ly(pN), 14, C.b, 0.6);
          // draft blocks
          const by = bot + 34;
          const nSteps = Math.floor(S.N / K), show = Math.min(12, nSteps);
          V.text(g, 'verification steps: ' + nSteps + (nSteps > 12 ? ' (12 shown)' : '') + ' · each block = 7 drafts', pad, by, { size: 10, font: 'mono', color: C.muted, maxW: w - 2 * pad });
          const cell = Math.min(7, (w - 2 * pad - 11 * 6) / (12 * K));
          for (let s = 0; s < show; s++) {
            const bx = pad + s * (K * (cell + 1) + 6);
            const acc = S.blocks[s] || 0, lit = el > s * 0.18;
            for (let k = 0; k < K; k++) V.box(g, bx + k * (cell + 1), by + 8, cell, 12, { r: 2, fill: lit && k < acc ? C.e : V.alpha(C.muted, 0.25) });
          }
          const txt = 'N = ' + S.N + ' · a = ' + f(S.a, 2) + ' → P(0 | healthy) = ' + (pN < 1e-4 ? pN.toExponential(1) : f(pN, 4)) + (S.N < 64 ? ' · below the minimum: “cannot judge”' : '');
          drawWrap(g, txt, pad, by + 42, w - 2 * pad, 15, { size: narrow ? 11 : 12, font: 'mono', weight: 700, color: S.N < 64 ? C.d : C.ink });
        } else if (S.view === 'bat') {
          const cols = narrow ? 2 : 4, gap = 10;
          const cw = (w - 2 * pad - gap * (cols - 1)) / cols, ch = narrow ? 88 : 104;
          let pass = 0;
          checks.forEach((c, i) => {
            const x = pad + (i % cols) * (cw + gap), y = 34 + Math.floor(i / cols) * (ch + gap);
            const started = el > i * 0.45, done = el > i * 0.45 + 0.4;
            const fail = (i === 5 && !S.fix55) || S.t1fails.includes(i);
            if (done && !fail) pass++;
            const col = !done ? C.muted : fail ? C.red : C.e;
            V.box(g, x, y, cw, ch, { r: 11, fill: V.alpha(col, done ? 0.1 : 0.04), stroke: V.alpha(col, done ? 0.8 : 0.3), lw: done ? 1.5 : 1 });
            V.text(g, clip(g, (i + 1) + '. ' + c.n, cw - (done ? 66 : 34), narrow ? 11 : 12, 'mono', 700), x + 10, y + 20, { size: narrow ? 11 : 12, font: 'mono', weight: 700, color: C.ink });
            V.text(g, clip(g, c.d, cw - 20, 10.5, 'sans'), x + 10, y + 38, { size: 10.5, color: C.muted });
            if (started && !done) { const a0 = t * 6; g.strokeStyle = C.a; g.lineWidth = 2; g.beginPath(); g.arc(x + cw - 18, y + 18, 7, a0, a0 + 4.2); g.stroke(); }
            if (done) {
              const msg = fail ? (i === 5 && !S.fix55 ? c.bad : 'malformed DSML tag → the parser misses the call') : c.ok;
              drawWrap(g, msg, x + 10, y + 56, cw - 20, 13, { size: 10, font: 'mono', color: fail ? C.red : C.e });
              badge(g, fail ? 'FAIL' : 'PASS', x + cw - 10, y + 18, fail ? C.red : C.e, { align: 'right', size: 9.5 });
            }
          });
          const allDone = el > 6 * 0.45 + 0.4;
          const ry = 34 + Math.ceil(checks.length / cols) * (ch + gap) + 14;
          const nr = drawWrap(g, allDone ? pass + '/7 checks · expected 7/7 (and 8/8 in the 32K/131K version)' : 'running at T=0…', pad, ry, w - 2 * pad, 16, { size: narrow ? 11 : 12.5, font: 'mono', weight: 700, color: allDone ? (pass === 7 ? C.e : C.red) : C.muted });
          if (S.t1 && allDone) drawWrap(g, 'Illustrative: sampling the syntax tokens at T=1 breaks calls at random — a serving artifact, not model quality. That is why the battery runs at T=0.', pad, ry + 4 + nr * 16, w - 2 * pad, 15, { size: 11, color: C.d });
        } else {
          const perRow = narrow ? 20 : 30;
          const cell = Math.min(20, (w - 2 * pad - (perRow - 1) * 3) / perRow);
          let y = 32, idx = 0, ok = 0, doneN = 0;
          const order = [];
          groups.forEach((gr) => { for (let k = 0; k < gr.n; k++) order.push(idx + k); idx += gr.n; });
          const timeOf = (i) => (i < 20 ? i * 0.04 : i < 120 ? 0.8 + Math.floor((i - 20) / 4) * 0.05 : 2.05 + (i - 120) * 0.04);
          idx = 0;
          groups.forEach((gr) => {
            V.text(g, gr.l, pad, y + 10, { size: 10.5, font: 'mono', color: C.muted, maxW: w - 2 * pad });
            y += 16;
            for (let k = 0; k < gr.n; k++) {
              const i = idx + k, r = Math.floor(k / perRow), c = k % perRow;
              const x = pad + c * (cell + 3), yy = y + r * (cell + 3);
              const done = el > timeOf(i);
              const fail = S.fails.includes(i);
              if (done) { doneN++; if (!fail) ok++; }
              V.box(g, x, yy, cell, cell, { r: 3, fill: !done ? V.alpha(C.muted, 0.15) : fail ? C.red : V.alpha(C.e, 0.75) });
            }
            y += Math.ceil(gr.n / perRow) * (cell + 3) + 8;
            idx += gr.n;
          });
          const fin = doneN === 145;
          V.text(g, (fin ? ok : ok) + '/145' + (fin ? (ok === 145 ? ' · goal reached (with zero warnings on both ranks)' : ' · before #191: 200 OK with no tool_calls or off-schema') : ' …'), pad, Math.min(y + 12, h - 30), { size: narrow ? 11 : 12.5, font: 'mono', weight: 700, color: !fin ? C.muted : ok === 145 ? C.e : C.red, maxW: w - 2 * pad });
          if (fin && ok < 145) V.text(g, 'different labels on every run (click run again)', pad, Math.min(y + 30, h - 12), { size: 10.5, color: C.muted, maxW: w - 2 * pad });
        }
      });
    })();

    /* =========================================================
       F10 — Quality × cost: the Pareto frontier
       ========================================================= */
    (() => {
      const fig = root.querySelector('#fig-pareto'); if (!fig) return;
      fitStage(fig, (w) => (w < 640 ? 700 : 470));
      const tfc = (C) => C.__tf || (C.__tf = (getComputedStyle(document.documentElement).getPropertyValue('--tf') || '').trim() || C.d); // GLM·TF badge color
      const pts = [
        { l: 'TR3 K6 (6 bpw)', s: 'K6 (6 bpw)', gb: 254, k: 0.013723, dx: 12, dy: 19, al: 'left', same: false },
        { l: 'Official FP8 (other stack)', s: 'FP8 (other stack)', gb: 328, k: 0.020615, dx: -15, dy: 4, al: 'right', same: false },
        { l: 'EXL3 4 bpw (served)', s: 'EXL3 4 bpw', gb: 176, k: 0.024555, dx: 12, dy: 18, al: 'left', same: true },
        { l: 'Official FP8 (v44 stack)', s: 'FP8 (v44)', gb: 328, k: 0.024629, dx: -12, dy: -12, al: 'right', same: true },
        { l: 'NVFP4 (v44 stack)', s: 'NVFP4 (v44)', gb: 180, k: 0.060535, dx: 12, dy: 4, al: 'left', same: true },
      ];
      const frontOf = (P) => P.filter((p) => !P.some((q) => q !== p && q.gb <= p.gb && q.k <= p.k && (q.gb < p.gb || q.k < p.k))).sort((a, b) => a.gb - b.gb);
      const txtBg = (g, str, x, y, o, C) => { const wd = tw(g, str, o.size, o.font || 'sans', o.weight || 400); const x0 = o.align === 'right' ? x - wd : o.align === 'center' ? x - wd / 2 : x; g.fillStyle = V.alpha(C.bg, 0.85); g.fillRect(x0 - 3, y - o.size + 1, wd + 6, o.size + 4); V.text(g, str, x, y, o); };
      const dec = [
        { r: 'glm', l: 'Dense FP8 (dense,kda)', d: 10, dl: '≈ −11 ms/step (+10%)', q: 'KL proxy 0.002–0.013 · argmax 94–100%', v: 'prov' },
        { r: 'glm', l: 'FP8=all (vs dense,kda)', d: 4.8, dl: 'tok/s +4.3 to +5.2%', q: 'kl_panel NOT-WORSE (ΔNLL −0.004…+0.0005)', v: 'ok' },
        { r: 'glm', l: 'KDA BF16 large-M', d: 12.3, dl: 'prefill 16k/64k −12.1/−12.6%', q: 'NOT-WORSE, at the edge of stock · +3.3 GiB/rank', v: 'ok' },
        { r: 'glm', l: 'Thin-decode FAST', d: 8.9, dl: 'A/B/A2 +7.7 to +14.6%', q: '0 flips; the absolute gate fails on stock', v: 'inc' },
        { r: 'glm', l: 'W8A8 in prefill (#182)', d: 30, dl: 'kernel: 27.21 → 12.57 ms', q: 'KL 7.99× the null, 23 flips', v: 'bad', over: true },
        { r: 'qwen', l: 'FP8 KV', d: -5.3, dl: '−5.3% (rerun; 1.80× the cache)', q: 'reasoning_check 12/12 = 12/12 · tok/step ↓ in 8/8', v: 'ok' },
        { r: 'qwen', l: 'Determinism (opt-in)', d: -4, dl: 'decode and prefill ≈ −4%', q: 'bit-identical: 5/5 → 1/5 distinct outputs', v: 'det' },
        { r: 'ds', l: 'SP indexer', d: 5, dl: 'TTFT 128K −4 to −6%', q: 'RULER-lite 8/8 (32K and 131K)', v: 'ok' },
        { r: 'tf', l: 'COMM=roce (vs NCCL)', d: 6.1, dl: 'decode +6% (prose 44.2 → 46.9)', q: 'same hash: no panel, no noise band', v: 'bits' },
        { r: 'tf', l: 'q4 dense (vs fp8)', d: 14.1, dl: 'prose 38.9 → 44.4 tok/s', q: 'GSM8K/HumanEval = fp8 · #18: 12/48 lose end of turn', v: 'cav' },
      ];
      const S = { view: 'kld', budget: 200, same: false };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'View', options: [{ v: 'kld', label: 'Bytes × KLD' }, { v: 'dec', label: 'Speed × verdict' }], value: 'kld', onChange: (v) => { S.view = v; showEl(slB, v === 'kld'); showEl(tgS, v === 'kld'); H.redraw(); } });
      const slB = UI.slider(ctl, { label: 'Budget (illustrative)', min: 150, max: 350, step: 5, value: 200, fmt: (v) => v + ' GB', onInput: (v) => { S.budget = v; H.redraw(); } });
      const tgS = UI.toggle(ctl, { label: 'same stack only', value: false, onChange: (v) => { S.same = v; H.redraw(); } });
      const H = V.fig(fig, (g, w, h) => {
        const C = V.C();
        const narrow = w < 640, pad = 14;
        if (S.view === 'kld') {
          const x0 = pad + 46, x1 = w - pad - 8, top = 30, bot = h - (narrow ? 124 : 78);
          const P = S.same ? pts.filter((p) => p.same) : pts, front = frontOf(P);
          const X = (gb) => x0 + (gb - 150) / (350 - 150) * (x1 - x0);
          const Y = (k) => bot - (k / 0.07) * (bot - top);
          [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07].forEach((k) => { const y = Y(k); V.line(g, x0, y, x1, y, { color: V.alpha(C.muted, 0.13), lw: 1 }); V.text(g, f(k, 2), x0 - 6, y + 3, { size: 9, font: 'mono', align: 'right', color: C.muted }); });
          [150, 200, 250, 300, 350].forEach((gb) => V.text(g, gb + ' GB', X(gb), bot + 14, { size: 9, font: 'mono', align: 'center', color: C.muted }));
          V.text(g, 'mean KLD (nats) ↓ better', pad, 16, { size: 10.5, font: 'mono', color: C.muted });
          V.text(g, 'weight size →', x1, bot + 28, { size: 10, font: 'mono', align: 'right', color: C.muted });
          const bx = X(S.budget);
          g.fillStyle = V.alpha(C.red, 0.06); g.fillRect(bx, top, x1 - bx, bot - top);
          V.line(g, bx, top - 6, bx, bot, { color: C.d, lw: 1.5, dash: [5, 3] });
          txtBg(g, 'budget ' + S.budget + ' GB', bx + 4, bot - 6, { size: 9.5, font: 'mono', color: C.d }, C);
          // staircase frontier
          g.beginPath();
          front.forEach((p, i) => { const x = X(p.gb), y = Y(p.k); if (!i) g.moveTo(x, y); else { g.lineTo(x, Y(front[i - 1].k)); g.lineTo(x, y); } });
          g.lineTo(x1, Y(front[front.length - 1].k));
          g.strokeStyle = C.e; g.lineWidth = 2; g.setLineDash([]); g.stroke();
          const fits = P.filter((p) => p.gb <= S.budget);
          const best = fits.length ? fits.reduce((a, b) => (b.k < a.k ? b : a)) : null;
          const labs = [];
          pts.forEach((p) => {
            const inP = P.includes(p), onF = front.includes(p), x = X(p.gb), y = Y(p.k);
            const col = !inP ? V.alpha(C.muted, 0.35) : p === best ? C.d : onF ? C.e : C.muted;
            if (p === best) V.glow(g, x, y, 18, C.d, 0.8);
            V.dot(g, x, y, onF ? 6.5 : 5, col);
            if (inP && !onF) { g.strokeStyle = C.muted; g.lineWidth = 1; g.beginPath(); g.arc(x, y, 7.5, 0, Math.PI * 2); g.stroke(); }
            if (!p.same) { g.strokeStyle = V.alpha(C.muted, inP ? 0.8 : 0.35); g.lineWidth = 1; g.setLineDash([2, 2]); g.beginPath(); g.arc(x, y, 10.5, 0, Math.PI * 2); g.stroke(); g.setLineDash([]); }
            labs.push([p, inP, onF, x, y]);
          });
          labs.forEach(([p, inP, onF, x, y]) => {
            const fs = narrow ? 9.5 : 11, fw = onF ? 700 : 400;
            const lab = clip(g, (narrow ? p.s : p.l) + ' · ' + f(p.k, 4), narrow ? 170 : 260, fs, 'sans', fw);
            const ty = y + p.dy;
            if (Math.abs(p.dy) > 12) V.line(g, x + (p.al === 'right' ? -4 : 4), y + Math.sign(p.dy) * 6, x + p.dx * 0.7, ty - Math.sign(p.dy) * 9, { color: V.alpha(C.muted, 0.6), lw: 1 });
            txtBg(g, lab, x + p.dx, ty, { size: fs, weight: fw, align: p.al, color: !inP ? V.alpha(C.muted, 0.5) : onF || p === best ? C.ink : C.muted }, C);
          });
          const msg = best ? 'Fits the budget and is the best: ' + best.l + ' (KLD ' + f(best.k, 4) + ', ' + best.gb + ' GB)' : 'Nothing fits this budget.';
          const note = S.same ? 'only EXL3, FP8 (v44) and NVFP4, measured on the same stack: EXL3 dominates the other two' : 'dashed circle = measured on another stack (a different ruler): indicative frontier';
          drawWrap(g, msg, pad, h - (narrow ? 76 : 34), w - 2 * pad, 15, { size: narrow ? 11 : 12, font: 'mono', weight: 700, color: best ? C.d : C.red });
          drawWrap(g, note + (narrow ? ' · green = frontier; circled gray = dominated' : ''), pad, h - (narrow ? 36 : 13), w - 2 * pad, 14, { size: 10, color: C.muted });
        } else {
          const vcol = { ok: C.e, prov: C.d, inc: C.d, bad: C.red, det: C.b, bits: C.a, cav: C.d };
          const vlab = { ok: 'approved', prov: 'provisional', inc: 'inconclusive', bad: 'rejected', det: 'reproducible', bits: 'same bits', cav: 'default, with a caveat' };
          const rcol = { glm: C.glm, tf: tfc(C), qwen: C.qwen, ds: C.ds };
          const legL = [['GLM', C.glm], ['GLM·TF', tfc(C)], ['Qwen', C.qwen], ['DeepSeek', C.ds]];
          const legW = legL.reduce((acc, it) => acc + tw(g, it[0], 10.5) + 22, 0);
          const top = narrow ? 52 : 36, rowH = narrow ? (h - top - 12) / dec.length : Math.min(46, (h - top - 10) / dec.length);
          const labW = narrow ? 0 : Math.min(190, w * 0.25), qW = narrow ? 0 : Math.min(270, w * 0.34);
          const ax0 = pad + labW + (narrow ? 0 : 10), ax1 = w - pad - qW - (narrow ? 0 : 14);
          const X = (d) => ax0 + (V.clamp(d, -10, 25) + 10) / 35 * (ax1 - ax0);
          V.text(g, 'Δ speed (%) — each arrow in its own study’s metric', pad, 16, { size: narrow ? 10 : 11, font: 'mono', color: C.muted, maxW: narrow ? w - 2 * pad : w - 2 * pad - legW - 8 });
          if (!narrow) { let lx = w - pad - legW; legL.forEach(([l, c]) => { V.dot(g, lx + 4, 12, 4, c); V.text(g, l, lx + 12, 16, { size: 10.5, color: C.muted }); lx += tw(g, l, 10.5) + 22; }); }
          else { let lx = pad; legL.forEach(([l, c]) => { V.dot(g, lx + 4, 30, 3.5, c); V.text(g, l, lx + 11, 33.5, { size: 9.5, color: C.muted }); lx += tw(g, l, 9.5) + 20; }); }
          [-10, -5, 0, 5, 10, 15, 20, 25].forEach((d) => { const x = X(d); V.line(g, x, top - 4, x, h - 8, { color: V.alpha(C.muted, d === 0 ? 0.5 : 0.12), lw: 1 }); V.text(g, (d > 0 ? '+' : '') + d, x, top - 8, { size: 9, font: 'mono', align: 'center', color: C.muted }); });
          dec.forEach((dd, i) => {
            const y = top + i * rowH;
            const col = vcol[dd.v];
            if (!narrow) {
              V.dot(g, pad + 5, y + rowH / 2, 4, rcol[dd.r] || C.a);
              V.text(g, clip(g, dd.l, labW - 16, 11.5, 'sans', 600), pad + 14, y + rowH / 2 - 2, { size: 11.5, weight: 600, color: C.ink });
              V.text(g, clip(g, dd.dl, labW - 16, 9.5, 'mono'), pad + 14, y + rowH / 2 + 12, { size: 9.5, font: 'mono', color: C.muted });
              const ay = y + rowH / 2;
              V.arrow(g, X(0), ay, X(dd.d), ay, { color: col, lw: 3, head: 8 });
              if (dd.over) V.text(g, '≫', X(25) - 2, ay - 8, { size: 12, weight: 700, align: 'right', color: col });
              V.text(g, clip(g, vlab[dd.v], qW, 10.5, 'mono', 700), ax1 + 14, y + rowH / 2 - 3, { size: 10.5, font: 'mono', weight: 700, color: col });
              V.text(g, clip(g, dd.q, qW, 9.5, 'sans'), ax1 + 14, y + rowH / 2 + 11, { size: 9.5, color: C.muted });
            } else {
              V.dot(g, pad + 4, y + 9, 3.5, rcol[dd.r] || C.a);
              V.text(g, clip(g, dd.l + ' · ' + vlab[dd.v], w - 2 * pad - 12, 10.5, 'sans', 700), pad + 12, y + 13, { size: 10.5, weight: 700, color: C.ink });
              const ay = y + 26;
              V.arrow(g, X(0), ay, X(dd.d), ay, { color: col, lw: 2.6, head: 7 });
              V.text(g, clip(g, dd.q, w - 2 * pad, 9.5, 'sans'), pad, y + rowH - 8, { size: 9.5, color: C.muted });
            }
          });
        }
      }, { static: true });
      showEl(slB, true);
    })();

    /* ---- local tweak: long controls wrap on narrow screens ---- */
    root.querySelectorAll('.la-fig-controls .la-ctl').forEach((el) => {
      el.style.flexWrap = 'wrap'; el.style.maxWidth = '100%'; el.style.minWidth = '0'; el.style.rowGap = '6px';
      const lab = el.querySelector('label'); if (lab) { lab.style.whiteSpace = 'normal'; }
      const sel = el.querySelector('select'); if (sel) { sel.style.maxWidth = '100%'; sel.style.minWidth = '0'; }
    });
    root.querySelectorAll('.la-fig-controls .la-toggle').forEach((el) => { el.style.maxWidth = '100%'; });
  },
});

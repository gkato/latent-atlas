LA.chapter({
  id: 'benchmarking',
  part: 9,
  order: 45,
  title: 'Measuring performance: TTFT, TPOT, ladders and A/B/A',
  navTitle: 'Benchmarking',
  subtitle: 'The right metrics, cold prompts, counters read by delta, concurrency ladders, A/B/A with one variable per boot and a model that predicts the number before you measure it.',
  minutes: 65,
  recipes: ['glm', 'tf', 'qwen', 'ds'],
  tags: ['benchmark', 'TTFT', 'TPOT', 'ITL', 'concurrency ladder', 'A/B/A', '/metrics', 'Prometheus', 'acceptance', 'profiling', 'cold prefill', 'TensorFold', 'exactness', 'clock cap'],
  goals: [
    'Define and compute <b>TTFT, TPOT/ITL, E2EL, per-stream and aggregate tok/s</b> and <b>prefill tok/s</b> from the stream timestamps and from <code>usage</code>',
    'Build a <b>protocol</b>: cold prompts salted at the start, fixed output (<code>ignore_eos</code> + <code>min_tokens</code>), thinking off, warmup, repetitions, ladders and <b>A/B/A</b> with one variable per boot',
    'Read <b>/metrics by delta</b>: per-position acceptance, tokens/step, ms/step, cache hit rate, and use acceptance canaries and gates',
    'Predict decode with the <b>step cost model</b> and pick the right <b>profiling</b> tool when the prediction does not add up',
    'Measure an <b>exact engine</b> (TensorFold): the conditions that travel with the number (sparkDash, a clock cap), the hash-based <b>exactness gate</b>, content × timing noise and the <b>exact simulation</b> of stop rules before booting',
  ],
  prereqs: ['receita-glm', 'receita-qwen', 'receita-deepseek'],
  summary: [
    'TTFT = \\(t_1 - t_0\\) (queue + prefill); TPOT = \\((t_N - t_1)/(N-1)\\) and decode tok/s = 1/TPOT; E2EL = \\(t_N - t_0\\); prefill tok/s = prompt tokens ÷ TTFT, valid only with an idle server and a cold cache. Count tokens via <code>usage</code>, never by SSE chunks.',
    'The aggregate rises and the per-stream rate falls with concurrency because the step rereads the same weights for all sequences: \\(t_{\\text{step}} \\approx t_0 + \\delta B\\). Above <code>max_num_seqs</code> requests queue up; a long prefill in the step drags down everyone\'s decode; KV capacity limits context × concurrency.',
    'A cold prefill requires a nonce at the <b>start</b> of the prompt (chained hash), confirmation via <code>cached_tokens</code>/Δhits, a length calibrated with <code>/tokenize</code> and warmed-up kernels: cache-cold, kernel-hot.',
    'The <code>/metrics</code> counters are cumulative and add up all clients: measure by delta. tokens/step = 1 + Δaccepted/Δdrafts; a healthy per-position curve decays; position 0 pinned at 1.00 is bug vllm#53030; the prompt rules acceptance.',
    'Method: the median of enough repetitions for the noise (c=1 ±10 tok/s in DeepSeek → 8 trials; \\(n \\approx 12.5\\,(\\sigma/\\delta)^2\\) per arm to see an effect \\(\\delta\\)), A/B/A to measure drift, one variable per boot against a same-day baseline, a frozen decision rule, boot and memory gates, and a record of commit/digest/.env/date.',
    'The step cost model (bytes ÷ bandwidth + collectives) is a floor, and tok/s ≈ (1 + accepted/step) ÷ t<sub>step</sub>: forgetting the bonus token skewed the Qwen report\'s prediction. Compare measured ms/step and tokens/step <i>separately</i>. When it does not add up, climb the profiling ladder: logs and /metrics, engine stopwatches (which synchronize CUDA), torch profiler, py-spy and NCCL dumps.',
    'In GLM\'s TensorFold lane, the number comes from sparkDash through the API, from another machine, with the clock capped at 2,200 MHz (60.4 → 108.8 tok/s of prose from 1 to 4 requests). Keyed sampling turns correctness into a hash (drafted = serial, concurrent = solo), removes content noise between trials of the same prompt and makes it possible to replay any stop rule offline, exactly, over a teacher-forced dump, with measured \\(V(R)\\) costs and an oracle as the ceiling. Per-patch numbers, measured on different builds, don\'t add up.',
  ],
  quiz: [
    { q: 'A request receives the first token at \\(t_1\\) = 2.0 s and the last one (the 101st) at \\(t_N\\) = 4.0 s. What is the per-stream decode tok/s?', o: ['25.3 tok/s (101 ÷ 4.0 s)', '50 tok/s (100 ÷ 2.0 s)', '50.5 tok/s (101 ÷ 2.0 s)', '33.7 tok/s (101 ÷ 3.0 s)'], a: 1, e: 'The first token belongs to TTFT; between \\(t_1\\) and \\(t_N\\) there are \\(N-1\\) = 100 intervals in 2.0 s: 50 tok/s (TPOT = 20 ms). 101 ÷ 4.0 s is the “wall-clock tok/s”, which mixes in the prefill; 101 ÷ 2.0 s ignores the fencepost problem.' },
    { q: 'You are going to measure the cold prefill of a 100K-token document five times in a row. Where should the random nonce go?', o: ['At the end of the prompt, after the question', 'At the start of the prompt (or in the <code>cache_salt</code> field)', 'Anywhere: the prefix cache compares the whole prompt', 'Nowhere: APC only reuses prompts from other users'], a: 1, e: 'Each block\'s hash depends on the previous block\'s hash. A nonce at the start changes \\(h_0\\) and the whole chain: zero hits. At the end, it changes only the last block and the rest of the document still comes from the cache. Check afterwards with <code>cached_tokens</code> = 0 or Δ<code>prefix_cache_hits</code> = 0.' },
    { q: 'You read <code>vllm:spec_decode_<wbr>num_accepted_<wbr>tokens_total</code> and <code>…_num_drafts_total</code> once and divided: 2.78 accepted per step. What is wrong?', o: ['Nothing: that is the server\'s acceptance', 'The counters are cumulative since boot and add up all clients; you need to measure the delta around your workload, with the server idle', 'You forgot to multiply by k', '/metrics is only updated at each restart'], a: 1, e: 'Prometheus <code>_total</code> counters are odometers. Dividing the totals describes the server\'s whole history, including synthetic prompts from other clients. <code>mtp_accept.py</code> (<code>--save</code>/<code>--since</code>) and <code>sweep.py</code> exist to measure Δ; a negative Δ gives away a restart.' },
    { q: 'In a three-arm A/B/A, the medians were A1 = 53.5, B = 52.0 and A2 = 49.4 tok/s. What can the session conclude?', o: ['B got 2.8% worse (B versus A1)', 'B improved by 5.3% (B versus A2)', 'Estimated effect ≈ +1.1% against the mean of the baselines, but the drift was −7.7%: the session supports neither a win nor a loss', 'The change is invalid because A2 < A1'], a: 2, e: 'The effect is \\(52.0 / ((53.5+49.4)/2) - 1 \\approx +1.1\\%\\), and the drift \\(A_2/A_1 - 1 \\approx -7.7\\%\\) is much larger. The two two-arm designs would reach opposite conclusions. This is the real case of GLM\'s compact pages (code): “no universal decode speedup”.' },
    { q: 'A decode step reads 16.5 GB per rank at an effective 240 GB/s and performs 108 collectives of 40 µs; at each step, 3.15 drafts are accepted. What is the approximate per-stream decode ceiling?', o: ['~14 tok/s', '~43 tok/s', '~57 tok/s', '~240 tok/s'], a: 2, e: '\\(16.5/240 = 68.75\\) ms + \\(108 \\times 40\\) µs = 4.3 ms ≈ 73 ms per step (~13.7 steps/s). Each step emits the target\'s bonus token plus the 3.15 accepted ones: \\((1 + 3.15)/0.0731 \\approx 57\\) tok/s, the DeepSeek report\'s math (~4.15 tokens/step → 56–61 tok/s). Dividing only the accepted ones gives ~43 tok/s: it forgets the bonus token; ~14 is steps/s, not tokens/s. Since this is a time floor, the real value tends to come in below it, unless the inputs (distinct experts, bandwidth) are not actually bounds.' },
    { q: 'Why should <code>VLLM_DSPARK_STAGE_TIMING=1</code> not stay on in a benchmark arm?', o: ['Because it writes too many logs to disk', 'Because it only works with TP=3', 'Because the CPU waits for the GPU to finish each stage (<code>end.synchronize()</code>), removing the CPU/GPU overlap: the measured engine is slower than the real one', 'Because it turns off speculation'], a: 2, e: 'The timers record CUDA events before and after each stage and call <code>end.synchronize()</code> at the end. The time is measured on the GPU, but the CPU sits idle until the stage finishes and does not enqueue the next one: the async scheduling and overlap that make the engine fast are lost (observer effect). The code itself defaults it to 0 and warns: “intended for diagnostics, not speed-gate benchmarks”.' },
    { q: 'In the TensorFold lane, a dump recorded with <code>TF_GLM_DRAFT_DUMP</code> lets you compute offline, exactly, the tok/s of a stop rule that never ran. What makes this possible?', o: ['DFlash2 is deterministic, so the proposed chain is always the same', 'Keyed sampling: the token committed at each position depends only on (seed, position, token), not on the policy, so any chain keeps exactly the drafts equal to the recorded tokens', 'The \\(V(R)\\) cost table, measured for each window width', 'Rejection sampling, which preserves the target distribution'], a: 1, e: 'With the key, the serial reply recorded in the dump is the same one any policy would produce. A round that starts at \\(i\\) keeps the drafts equal to the tokens committed at \\(i+1, i+2, \\ldots\\) up to the first mismatch, plus one sampled token. A deterministic drafter is not enough: without the key, the target would draw a different reply on every run. The cost table only turns rounds into time (and it dates from patch 0011, hence the live confirmation). Rejection sampling preserves the distribution, not the sequence: a log doesn\'t tell you what another chain would have accepted.' },
  ],

  mount(root, LA) {
    const V = LA.viz, UI = LA.ui;

    /* ===================== local utilities ===================== */
    const br = (x, d = 1) => (+x).toFixed(d);
    const brk = (x) => Math.round(x).toLocaleString('en-US');
    const FC = {};
    const fnt = (s, f = 'sans', wt = 400) => FC[s + f + wt] || (FC[s + f + wt] = V.font(s, f, wt));
    const WM = new Map();
    function lines(g, s, maxW, size, f, wt) {
      const key = s + '|' + Math.round(maxW) + '|' + size + (f || '') + (wt || '');
      let L = WM.get(key);
      if (L) return L;
      g.font = fnt(size, f || 'sans', wt || 400);
      const words = String(s).split(' ');
      L = []; let cur = '';
      for (const wd of words) {
        const tt = cur ? cur + ' ' + wd : wd;
        if (cur && g.measureText(tt).width > maxW) { L.push(cur); cur = wd; } else cur = tt;
      }
      if (cur) L.push(cur);
      if (WM.size > 1500) WM.clear();
      WM.set(key, L);
      return L;
    }
    function wtext(g, s, x, y, maxW, o = {}) {
      const size = o.size || 12, lh = o.lh || Math.round(size * 1.34);
      let L = lines(g, s, maxW, size, o.font, o.weight);
      if (o.maxLines && L.length > o.maxLines) { L = L.slice(0, o.maxLines); L[L.length - 1] += '…'; }
      L.forEach((l, i) => V.text(g, l, x, y + i * lh, { size, font: o.font, weight: o.weight, color: o.color, align: o.align, alpha: o.alpha }));
      return L.length * lh;
    }
    function tw(g, s, size, f, wt) { g.font = fnt(size, f || 'sans', wt || 400); return g.measureText(s).width; }
    function pill(g, x, y, s, col, o = {}) {
      const C = V.C();
      const size = o.size || 11, f = o.font || 'mono', wt = o.weight || 500;
      const w0 = tw(g, s, size, f, wt), pw = w0 + (o.pad == null ? 8 : o.pad) * 2, ph = size + 8;
      const xx = o.align === 'center' ? x - pw / 2 : o.align === 'right' ? x - pw : x;
      V.box(g, xx, y - ph / 2, pw, ph, { r: ph / 2, fill: V.alpha(col, o.fa == null ? 0.16 : o.fa), stroke: V.alpha(col, 0.6), lw: 1 });
      V.text(g, s, xx + pw / 2, y + 0.5, { size, font: f, weight: wt, color: o.color || C.ink, align: 'center', base: 'middle' });
      return pw;
    }
    function axis(max, n = 5) { const step = nice(max / n); return { step, top: Math.max(step, Math.ceil(max / step) * step) }; }
    function nice(x) { if (!(x > 0)) return 1; const e = Math.pow(10, Math.floor(Math.log10(x))); const f = x / e; return (f <= 1 ? 1 : f <= 2 ? 2 : f <= 2.5 ? 2.5 : f <= 5 ? 5 : 10) * e; }
    function fit(fig, dh, mh, bp = 600) {
      const st = fig.querySelector('.la-fig-stage');
      const set = () => { const w = fig.clientWidth || window.innerWidth; const want = (w < bp ? mh : dh) + 'px'; if (st.style.height !== want) st.style.height = want; };
      set();
      const ro = new ResizeObserver(set); ro.observe(fig);
      LA.onCleanup(() => ro.disconnect());
    }
    let tfCache = null;
    const TFC = (C) => { if (!tfCache || tfCache.C !== C) tfCache = { C, v: getComputedStyle(document.documentElement).getPropertyValue('--tf').trim() || C.d }; return tfCache.v; };
    const rcol = (C, r) => (r === 'glm' ? C.glm : r === 'qwen' ? C.qwen : r === 'tf' ? TFC(C) : C.ds);
    const median = (xs) => { const a = xs.slice().sort((p, q) => p - q); const n = a.length; return n ? (n % 2 ? a[(n - 1) / 2] : (a[n / 2 - 1] + a[n / 2]) / 2) : NaN; };
    const mean = (xs) => xs.reduce((p, q) => p + q, 0) / (xs.length || 1);
    function hatch(g, x, y, w, h, col, step = 6) {
      g.save(); g.beginPath(); g.rect(x, y, w, h); g.clip();
      g.strokeStyle = col; g.lineWidth = 1;
      g.beginPath();
      for (let k = -h; k < w; k += step) { g.moveTo(x + k, y + h); g.lineTo(x + k + h, y); }
      g.stroke(); g.restore();
    }

    /* ============ Figure 1: the stopwatch of a request ============ */
    (function () {
      const fig = root.querySelector('#fig-cronometro');
      fit(fig, 360, 420, 620);
      const S = { P: 3072, spec: false };
      let M = null, H = null;
      const ctl = UI.controls(fig);
      UI.slider(ctl, { label: 'Prompt tokens', min: 512, max: 8192, step: 512, value: S.P, fmt: (v) => brk(v), onInput: (v) => { S.P = v; upd(); } });
      UI.toggle(ctl, { label: 'speculation (several tokens per step)', value: false, onChange: (v) => { S.spec = v; upd(); } });
      const R = {
        ttft: UI.readout(ctl, 'TTFT', '—'), tpot: UI.readout(ctl, 'TPOT', '—'), itl: UI.readout(ctl, 'mean ITL', '—'),
        dec: UI.readout(ctl, 'decode tok/s', '—'), wall: UI.readout(ctl, 'N ÷ E2EL', '—'), ev: UI.readout(ctl, 'events / tokens', '—'),
      };
      const steps = [
        { t: 'Send and queue', d: 'The clock starts at \\(t_0\\), when the client sends. If the engine is busy, the request waits in the queue (gray), and that time already counts toward TTFT. That is why prefill tok/s is only valid with the server idle.' },
        { t: 'Chunked prefill', d: 'The prompt is processed in fixed-size chunks of up to <code>max_num_batched_tokens</code> tokens (1,024 here, illustrative). At short context each chunk costs ~the same time, so TTFT grows ~linearly with P. Prefill is usually compute-bound, but in the recipes, at 1,024 tokens per chunk, the MoE is still bandwidth-bound (§6).' },
        { t: 'First token: TTFT', d: 'The last prefill chunk already samples the first token, which arrives at \\(t_1\\). Before it, the stream may send a chunk with only the <code>role</code>. Decide, and write down in the script, which event counts as “first”.' },
        { t: 'Decode: steps, TPOT and ITL', d: 'Each step rereads the weights and emits 1 token (or 1 + accepted, with speculation). ITL is the interval between events; TPOT = \\((t_N - t_1)/(N-1)\\). Without speculation they coincide; with it, ITL starts measuring steps.' },
        { t: 'Counting tokens', d: 'With speculation, one SSE envelope carries several tokens. Counting envelopes undercounts the output; count via <code>usage.completion_tokens</code> (with <code>stream_options.include_usage</code>).' },
        { t: 'What to publish', d: 'Publish TTFT and decode tok/s separately. \\(N/\\text{E2EL}\\) mixes the prefill into the decode: with long prompts it plummets without the decode having changed.' },
      ];
      const st = UI.stepper(fig, steps, { autoplay: 7000, onStep: () => H && H.redraw() });
      function model() {
        const q = 0.12, Rp = 1500, CH = 1024, N = 20;
        const chunks = []; let t = q, left = S.P;
        while (left > 0) { const n = Math.min(CH, left); chunks.push({ t0: t, t1: t + n / Rp, n }); t += n / Rp; left -= n; }
        const t1 = t;
        const ev = [{ t: t1, k: 1 }], dsteps = [];
        let rem = N - 1, i = 0;
        const pat = [4, 2, 3, 1, 4, 3, 2], sd = S.spec ? 0.075 : 0.06;
        while (rem > 0) { const k = S.spec ? Math.min(rem, pat[i % pat.length]) : 1; const s0 = t; t += sd; dsteps.push({ t0: s0, t1: t, k }); ev.push({ t, k }); rem -= k; i++; }
        return { q, Rp, chunks, t1, tN: t, ev, dsteps, N, sd };
      }
      function upd() {
        M = model();
        const m = M, tpot = (m.tN - m.t1) / (m.N - 1), itl = (m.tN - m.t1) / (m.ev.length - 1);
        R.ttft.set(br(m.t1, 2) + ' s'); R.tpot.set(br(tpot * 1000, 0) + ' ms'); R.itl.set(br(itl * 1000, 0) + ' ms');
        R.dec.set(br((m.N - 1) / (m.tN - m.t1), 1)); R.wall.set(br(m.N / m.tN, 1)); R.ev.set(m.ev.length + ' / ' + m.N);
        H && H.redraw();
      }
      M = model();
      H = V.fig(fig, (g, w, h, t) => {
        const C = V.C(), m = M, narrow = w < 620, si = st.i;
        const padL = narrow ? 16 : 96, padR = 18, x0 = padL, x1 = w - padR;
        const flin = m.t1 / m.tN, fr = V.clamp(flin, 0.3, 0.62), broken = Math.abs(fr - flin) > 0.02;
        const xm = x0 + (x1 - x0) * fr;
        const X = (tt) => (tt <= m.t1 ? x0 + (tt / m.t1) * (xm - x0) : xm + ((tt - m.t1) / (m.tN - m.t1)) * (x1 - xm));
        const cyc = 9, phase = (t % cyc) / (cyc - 1.6), phx = x0 + Math.min(1, phase) * (x1 - x0);
        const focus = (k) => (si === k ? 1 : 0.42);
        V.text(g, 'P = ' + brk(S.P) + ' prompt · N = ' + m.N + ' output tokens' + (S.spec ? ' · spec on' : ''), x0, 18, { size: 11, font: 'mono', color: C.muted });
        if (broken && !narrow) pill(g, x1, 18, 'axis compressed before t₁', C.d, { align: 'right', size: 11, font: 'sans' });
        const yM = narrow ? 52 : 44, hM = 38, yC = narrow ? 152 : 124, hC = 30;
        const yAx = yM + hM + 14;
        // lane labels
        if (narrow) {
          V.text(g, 'ENGINE (vLLM)', x0, yM - 8, { size: 11, font: 'mono', color: C.muted });
          V.text(g, 'CLIENT (SSE events)', x0, yC - 8, { size: 11, font: 'mono', color: C.muted });
        } else {
          V.text(g, 'ENGINE', 14, yM + hM / 2 + 4, { size: 11.5, font: 'mono', color: C.muted, weight: 600 });
          V.text(g, 'CLIENT', 14, yC + hC / 2 + 4, { size: 11.5, font: 'mono', color: C.muted, weight: 600 });
          V.text(g, '(SSE)', 14, yC + hC / 2 + 18, { size: 11, font: 'mono', color: C.muted });
        }
        // queue
        g.globalAlpha = focus(0);
        const qx0 = X(0), qx1 = X(m.q);
        V.box(g, qx0, yM, Math.max(2, qx1 - qx0), hM, { r: 5, fill: V.alpha(C.muted, 0.18), stroke: V.alpha(C.muted, 0.5), lw: 1 });
        hatch(g, qx0, yM, qx1 - qx0, hM, V.alpha(C.muted, 0.45));
        if (qx1 - qx0 > 26) V.text(g, 'queue', (qx0 + qx1) / 2, yM + hM / 2 + 4, { size: 11, align: 'center', color: C.ink });
        // prefill
        g.globalAlpha = focus(1);
        m.chunks.forEach((c, i) => {
          const a = X(c.t0) + 1, b = X(c.t1) - 1;
          V.box(g, a, yM, Math.max(1.5, b - a), hM, { r: 5, fill: V.alpha(C.b, 0.3), stroke: C.b, lw: 1.2 });
          if (b - a > 30) V.text(g, 'chunk ' + (i + 1), (a + b) / 2, yM + hM / 2 + 4, { size: 11, align: 'center', color: C.ink });
        });
        if (m.chunks.length && (X(m.t1) - X(m.q)) > 96 && X(m.chunks[0].t1) - X(m.chunks[0].t0) <= 30) pill(g, (X(m.q) + X(m.t1)) / 2, yM + hM / 2, 'prefill · ' + m.chunks.length + ' chunks', C.b, { align: 'center', size: 11, fa: 0.85, color: C.dark ? '#140a26' : '#fff' });
        // decode
        g.globalAlpha = focus(3);
        m.dsteps.forEach((s) => {
          const a = X(s.t0) + 0.8, b = X(s.t1) - 0.8;
          V.box(g, a, yM, Math.max(1.5, b - a), hM, { r: 3, fill: V.alpha(C.a, 0.32), stroke: C.a, lw: 1 });
          if (S.spec && b - a > 11) V.text(g, String(s.k), (a + b) / 2, yM + hM / 2 + 4, { size: 11, align: 'center', color: C.ink, font: 'mono', weight: 600 });
        });
        g.globalAlpha = 1;
        // time axis
        V.line(g, x0, yAx, x1, yAx, { color: V.alpha(C.muted, 0.5), lw: 1 });
        const axLab = [[0, 't₀ = 0'], [m.t1, 't₁'], [m.tN, 't_N']];
        axLab.forEach(([tt]) => { const x = X(tt); V.line(g, x, yAx - 4, x, yAx + 4, { color: C.muted, lw: 1.2 }); });
        if (broken) {
          const zx = x0 + (xm - x0) * 0.72;
          g.strokeStyle = C.d; g.lineWidth = 1.5; g.beginPath();
          g.moveTo(zx - 5, yAx + 5); g.lineTo(zx - 1, yAx - 5); g.moveTo(zx + 1, yAx + 5); g.lineTo(zx + 5, yAx - 5); g.stroke();
        }
        // client: envelopes
        V.line(g, x0, yC + hC, x1, yC + hC, { color: V.alpha(C.muted, 0.35), lw: 1, dash: [3, 4] });
        m.ev.forEach((e, i) => {
          const x = X(e.t), seen = x <= phx + 0.5;
          const f = i === 0 ? Math.max(focus(2), focus(4)) : focus(4) === 1 ? 1 : focus(3) === 1 ? 0.8 : 0.5;
          g.globalAlpha = seen ? f : 0.15;
          const ew = narrow ? 16 : 12, eh = narrow ? 12 : 9, ey = yC + hC - eh - 3;
          V.box(g, x - ew / 2, ey, ew, eh, { r: 2, fill: V.alpha(i === 0 ? C.d : C.c, 0.28), stroke: i === 0 ? C.d : C.c, lw: 1 });
          g.strokeStyle = i === 0 ? C.d : C.c; g.lineWidth = 1; g.beginPath(); g.moveTo(x - ew / 2, ey); g.lineTo(x, ey + eh / 2); g.lineTo(x + ew / 2, ey); g.stroke();
          const dr = narrow ? 2.2 : 1.8, ds = narrow ? 5 : 4.4;
          if (S.spec && e.k > 1) { for (let k = 0; k < e.k; k++) V.dot(g, x - (e.k - 1) * ds / 2 + k * ds, ey - 5, dr, C.c); }
          else V.dot(g, x, ey - 5, dr, i === 0 ? C.d : C.c);
          // token flight
          const d = phx - x;
          if (d >= 0 && d < 46) { const u = d / 46; V.dot(g, x, V.lerp(yM + hM, ey - 5, V.ease.out(u)), 3, i === 0 ? C.d : C.a); }
        });
        g.globalAlpha = 1;
        // ITL (step 4)
        if (si === 3 && m.ev.length > 2) {
          const a = X(m.ev[1].t), b = X(m.ev[2].t);
          V.arrow(g, a, yC - 2, b, yC - 2, { color: C.ink, lw: 1.2, head: 5, curve: -8 });
          V.text(g, 'ITL', (a + b) / 2, yC - 12, { size: 11, align: 'center', color: C.ink, font: 'mono', weight: 600 });
        }
        // playhead
        V.line(g, phx, yM - 6, phx, yC + hC + 4, { color: V.alpha(C.ink, 0.35), lw: 1, dash: [2, 3] });
        // axis labels above the playhead, with background
        axLab.forEach(([tt, l], i) => {
          const x = X(tt), al = i === 0 ? 'left' : i === 2 ? 'right' : 'center', lw0 = tw(g, l, 11, 'mono');
          const bx0 = al === 'left' ? x - 2 : al === 'right' ? x - lw0 - 2 : x - lw0 / 2 - 2;
          V.box(g, bx0, yAx + 6, lw0 + 4, 14, { r: 3, fill: C.bg });
          V.text(g, l, x, yAx + 16, { size: 11, align: al, color: C.muted, font: 'mono' });
        });
        if (narrow) [['ENGINE (vLLM)', yM - 8], ['CLIENT (SSE events)', yC - 8]].forEach(([l, yy]) => {
          const lw0 = tw(g, l, 11, 'mono');
          V.box(g, x0 - 2, yy - 11, lw0 + 4, 14, { r: 3, fill: C.bg });
          V.text(g, l, x0, yy, { size: 11, font: 'mono', color: C.muted });
        });
        // brackets
        const yB = yC + hC + (narrow ? 26 : 24), gapB = narrow ? 34 : 30;
        const brk2 = (xa, xb, y, col, label, k) => {
          const f = focus(k) === 1 || si === 5 ? 1 : 0.5;
          g.globalAlpha = f;
          V.line(g, xa, y, xb, y, { color: col, lw: 2 });
          V.line(g, xa, y - 5, xa, y + 5, { color: col, lw: 2 }); V.line(g, xb, y - 5, xb, y + 5, { color: col, lw: 2 });
          const lw0 = tw(g, label, 11.5, 'mono', 600);
          const cx = V.clamp((xa + xb) / 2, x0 + lw0 / 2 + 4, x1 - lw0 / 2 - 4);
          V.box(g, cx - lw0 / 2 - 6, y - 17, lw0 + 12, 15, { r: 7, fill: C.bg });
          V.text(g, label, cx, y - 6, { size: 11.5, font: 'mono', weight: 600, align: 'center', color: col });
          g.globalAlpha = 1;
        };
        brk2(X(0), X(m.t1), yB + 8, C.d, 'TTFT ' + br(m.t1, 2) + ' s', 2);
        brk2(X(m.t1), X(m.tN), yB + 8 + gapB, C.a, (narrow ? 'decode: ' : 'decode window: ') + (m.N - 1) + ' tokens in ' + br(m.tN - m.t1, 2) + ' s', 3);
        brk2(X(0), X(m.tN), yB + 8 + 2 * gapB, C.c, 'E2EL ' + br(m.tN, 2) + ' s', 5);
        // formula for the current step
        const yF = yB + 8 + 2 * gapB + (narrow ? 26 : 24);
        const tpot = (m.tN - m.t1) / (m.N - 1);
        const formulas = [
          'Queue of ' + br(m.q * 1000, 0) + ' ms already inside TTFT: with concurrency, TTFT = queue + prefill + 1st sampling.',
          'prefill tok/s = P ÷ TTFT = ' + brk(S.P) + ' ÷ ' + br(m.t1, 2) + ' s = ' + brk(S.P / m.t1) + ' tok/s (the queue is included).',
          'TTFT = t₁ − t₀ = ' + br(m.t1, 2) + ' s. The role chunk (no text) arrives earlier; with thinking on, the 1st content only comes after <think>.',
          'TPOT = (t_N − t₁)/(N − 1) = ' + br(tpot * 1000, 0) + ' ms · mean ITL = ' + br((m.tN - m.t1) / (m.ev.length - 1) * 1000, 0) + ' ms per event' + (S.spec ? ' (one step each).' : '.'),
          'SSE events: ' + m.ev.length + ' · tokens in usage: ' + m.N + (S.spec ? ' → counting events undercounts ' + br(m.N / m.ev.length, 1) + '×.' : ' → without speculation, 1 token per event.'),
          'decode = (N − 1)/(t_N − t₁) = ' + br((m.N - 1) / (m.tN - m.t1), 1) + ' tok/s   ·   N ÷ E2EL = ' + br(m.N / m.tN, 1) + ' tok/s (mixes in the prefill).',
        ];
        const fsz = narrow ? 11.5 : 12.5, flh = narrow ? 16 : 18;
        const nL = Math.min(4, lines(g, formulas[si], x1 - x0 - 12, fsz, 'mono').length);
        V.box(g, x0 - (narrow ? 4 : 10), yF - 17, (x1 - x0) + (narrow ? 8 : 20), nL * flh + 14, { r: 10, fill: V.alpha(C.b, 0.08), stroke: V.alpha(C.b, 0.3), lw: 1 });
        wtext(g, formulas[si], x0 + 4, yF + 2, x1 - x0 - 12, { size: fsz, font: 'mono', color: C.ink, lh: flh, maxLines: 4 });
      });
      upd();
    })();

    /* ============ Figure 2: one step, many passengers ============ */
    (function () {
      const fig = root.querySelector('#fig-onibus');
      fit(fig, 430, 600, 640);
      const P = { t0: 48, d: 7.5, tau: 2.9, pf: 700 };
      const S = { C: 4, Smax: 8, pf: false };
      const calc = (C, Smax, pf) => { const B = Math.min(C, Smax); const ts = P.t0 + P.d * B + (pf ? P.pf : 0); const per = (P.tau / ts) * 1000; return { B, ts, per, agg: B * per, q: Math.max(0, C - Smax) }; };
      const ctl = UI.controls(fig);
      UI.slider(ctl, { label: 'Concurrent streams (C)', min: 1, max: 16, step: 1, value: S.C, onInput: (v) => { S.C = v; upd(); } });
      UI.seg(ctl, { label: 'max_num_seqs', options: [4, 8, 16], value: 8, onChange: (v) => { S.Smax = v; upd(); } });
      UI.toggle(ctl, { label: 'long prefill sharing the step', value: false, onChange: (v) => { S.pf = v; upd(); } });
      const R = { ts: UI.readout(ctl, 'step', '—'), per: UI.readout(ctl, 'per stream', '—'), agg: UI.readout(ctl, 'aggregate', '—'), q: UI.readout(ctl, 'queued', '—') };
      function upd() { const c = calc(S.C, S.Smax, S.pf); R.ts.set(br(c.ts, 0) + ' ms'); R.per.set(br(c.per, 1) + ' tok/s'); R.agg.set(br(c.agg, 0) + ' tok/s'); R.q.set(String(c.q)); }
      upd();
      const rg = V.rng(45);
      const W = Array.from({ length: 6 }, () => Array.from({ length: 20 }, () => rg()));
      const st = { prog: 0, steps: 0, parts: [], out: 0 };
      V.fig(fig, (g, w, h, t, dt) => {
        const C = V.C(), narrow = w < 640, c = calc(S.C, S.Smax, S.pf);
        const pad = 14;
        const mw = narrow ? w - 2 * pad : w * 0.52, mh = narrow ? h * 0.5 : h - 2 * pad;
        const mx = pad, my = pad;
        // step progress (real time proportional to the step)
        const realDur = V.clamp((c.ts / 1000) * 6, 0.35, 5.2);
        if (!LA.paused) st.prog += dt / realDur;
        if (st.prog >= 1) {
          st.prog -= Math.floor(st.prog); st.steps++; st.out += c.B;
          for (let i = 0; i < c.B; i++) st.parts.push({ i, life: 0 });
          if (st.parts.length > 120) st.parts.splice(0, st.parts.length - 120);
        }
        V.box(g, mx, my, mw, mh, { r: 12, fill: V.alpha(C.a, 0.04), stroke: V.alpha(C.muted, 0.25), lw: 1 });
        V.text(g, 'ONE DECODE STEP', mx + 12, my + 20, { size: 11, font: 'mono', color: C.muted, weight: 600 });
        V.text(g, 'step = ' + br(c.ts, 0) + ' ms', mx + mw - 12, my + 20, { size: 11.5, font: 'mono', color: S.pf ? C.c : C.a, weight: 600, align: 'right' });
        // seats
        const cols = 8, sr = Math.min(11, (mw - 40) / cols / 2 - 3), sx0 = mx + 20 + sr, sy0 = my + 44 + sr;
        const sdx = (mw - 40 - 2 * sr) / (cols - 1), sdy = sr * 2 + 12;
        for (let i = 0; i < 16; i++) {
          const cx = sx0 + (i % cols) * sdx, cy = sy0 + Math.floor(i / cols) * sdy;
          if (i < c.B) { V.dot(g, cx, cy, sr, V.alpha(C.a, 0.3)); g.strokeStyle = C.a; g.lineWidth = 1.5; g.beginPath(); g.arc(cx, cy, sr, 0, Math.PI * 2); g.stroke(); }
          else if (i < S.C) { g.setLineDash([2, 2]); g.strokeStyle = C.d; g.lineWidth = 1.5; g.beginPath(); g.arc(cx, cy, sr, 0, Math.PI * 2); g.stroke(); g.setLineDash([]); }
          else { g.strokeStyle = V.alpha(C.muted, 0.3); g.lineWidth = 1; g.beginPath(); g.arc(cx, cy, sr, 0, Math.PI * 2); g.stroke(); }
        }
        // particles (emitted tokens)
        st.parts.forEach((p) => { p.life += dt; });
        st.parts = st.parts.filter((p) => p.life < 0.9);
        st.parts.forEach((p) => {
          const cx = sx0 + (p.i % cols) * sdx, cy = sy0 + Math.floor(p.i / cols) * sdy;
          const u = p.life / 0.9;
          for (let k = 0; k < 3; k++) V.dot(g, cx + (k - 1) * 4, cy - sr - 4 - u * 18 - k * 2, 2.2, V.alpha(C.a, 1 - u));
        });
        // weight conveyor belt
        const tyTop = sy0 + sdy + sr + 18;
        const trackX = mx + 16, trackW = mw - 32, trackY = tyTop + 14, trackH = Math.max(40, Math.min(narrow ? 90 : 190, my + mh - trackY - 50));
        V.text(g, narrow ? 'weights read 1× per step, for the whole batch' : 'active weights read from memory (1× per step, for the whole batch)', trackX, tyTop + 4, { size: 11, color: C.muted });
        if (c.q > 0) V.text(g, 'dashed amber = queued', mx + mw - 12, tyTop + 4, { size: 11, color: C.d, align: 'right' });
        const decT = P.t0 + P.d * c.B, totT = c.ts;
        const wFrac = S.pf ? 0.55 : 1;
        const matW = trackW * wFrac;
        const tNow = st.prog * totT;
        const scanX = tNow <= decT ? trackX + (tNow / decT) * matW : trackX + matW + ((tNow - decT) / (totT - decT)) * (trackW - matW);
        const R0 = W.length, K0 = W[0].length, cw = matW / K0, ch = trackH / R0;
        for (let i = 0; i < R0; i++) for (let j = 0; j < K0; j++) {
          const x = trackX + j * cw, y = trackY + i * ch, read = x + cw <= scanX;
          g.fillStyle = V.cmap(0.25 + 0.6 * W[i][j], 'seq'); g.globalAlpha = read ? 0.95 : 0.35;
          V.rr(g, x + 0.8, y + 0.8, cw - 1.6, ch - 1.6, 2); g.fill();
        }
        g.globalAlpha = 1;
        if (S.pf) {
          const px = trackX + matW + 4, pw = trackW - matW - 4;
          V.box(g, px, trackY, pw, trackH, { r: 8, fill: V.alpha(C.b, scanX > px ? 0.35 : 0.15), stroke: C.b, lw: 1.5 });
          wtext(g, 'prefill chunk: +' + P.pf + ' ms', px + 8, trackY + trackH / 2 - 2, pw - 14, { size: 11, color: C.ink, lh: 14, maxLines: 3 });
        }
        V.line(g, scanX, trackY - 6, scanX, trackY + trackH + 6, { color: C.c, lw: 2 });
        V.glow(g, scanX, trackY - 6, 8, C.c);
        wtext(g, 'each occupied seat gains τ ≈ 2.9 tokens per step · delivered: ' + brk(st.out * P.tau), trackX, trackY + trackH + 22, trackW, { size: 11, color: C.muted, maxLines: 2, lh: 14 });
        // chart
        const gx = narrow ? pad + 34 : mx + mw + 50, gy = narrow ? my + mh + 34 : pad + 30;
        const gw = narrow ? w - gx - pad - 6 : w - gx - pad - 6, gh = narrow ? h - gy - 40 : h - gy - 44;
        const pts = [], ref = []; let ymax = 0;
        for (let k = 1; k <= 16; k++) { const q = calc(k, S.Smax, S.pf), r0 = calc(k, S.Smax, false); pts.push(q); ref.push(r0); ymax = Math.max(ymax, r0.agg, r0.per); }
        const ax = axis(ymax * 1.05), ym = ax.top;
        const XC = (k) => gx + ((k - 1) / 15) * gw, YV = (v) => gy + gh - (v / ym) * gh;
        V.text(g, 'tok/s', gx - 4, gy - 10, { size: 11, font: 'mono', color: C.muted, align: 'right' });
        for (let v = 0; v <= ym + 1e-9; v += ax.step) { const y = YV(v); V.line(g, gx, y, gx + gw, y, { color: C.grid, lw: 1 }); V.text(g, br(v, 0), gx - 6, y + 4, { size: 11, font: 'mono', color: C.muted, align: 'right' }); }
        if (S.Smax < 16) { const qx = XC(S.Smax + 0.5); g.fillStyle = V.alpha(C.d, 0.08); g.fillRect(qx, gy, gx + gw - qx, gh); V.text(g, 'queue (C > ' + S.Smax + ')', qx + 6, gy + 14, { size: 11, color: C.d }); }
        g.beginPath(); pts.forEach((q, k) => { const x = XC(k + 1), y = YV(q.agg); k ? g.lineTo(x, y) : g.moveTo(x, y); }); g.lineTo(XC(16), YV(0)); g.lineTo(XC(1), YV(0)); g.closePath(); g.fillStyle = V.alpha(C.a, 0.12); g.fill();
        g.beginPath(); pts.forEach((q, k) => { const x = XC(k + 1), y = YV(q.agg); k ? g.lineTo(x, y) : g.moveTo(x, y); }); g.strokeStyle = C.a; g.lineWidth = 2.2; g.stroke();
        g.beginPath(); pts.forEach((q, k) => { const x = XC(k + 1), y = YV(q.per); k ? g.lineTo(x, y) : g.moveTo(x, y); }); g.strokeStyle = C.b; g.lineWidth = 2.2; g.stroke();
        if (S.pf) {
          g.beginPath(); ref.forEach((q, k) => { const x = XC(k + 1), y = YV(q.agg); k ? g.lineTo(x, y) : g.moveTo(x, y); }); g.strokeStyle = V.alpha(C.a, 0.55); g.lineWidth = 1.5; g.setLineDash([5, 4]); g.stroke(); g.setLineDash([]);
          V.text(g, 'aggregate without prefill', XC(16) - 2, YV(ref[15].agg) - 8, { size: 11, color: V.alpha(C.a, 0.8), align: 'right' });
        }
        [1, 4, 8, 12, 16].forEach((k) => V.text(g, String(k), XC(k), gy + gh + 15, { size: 11, font: 'mono', color: C.muted, align: 'center' }));
        V.text(g, 'concurrent streams (C)', gx + gw / 2, gy + gh + 31, { size: 11, color: C.muted, align: 'center' });
        const cur = pts[S.C - 1], cxp = XC(S.C);
        V.line(g, cxp, gy, cxp, gy + gh, { color: V.alpha(C.ink, 0.35), lw: 1, dash: [3, 3] });
        V.dot(g, cxp, YV(cur.agg), 4.5, C.a); V.dot(g, cxp, YV(cur.per), 4.5, C.b);
        const lx = S.C > 11 ? cxp - 8 : cxp + 8, la = S.C > 11 ? 'right' : 'left';
        V.text(g, 'aggregate ' + br(cur.agg, 0), lx, YV(cur.agg) - 8, { size: 11.5, font: 'mono', weight: 600, color: C.a, align: la });
        V.text(g, 'per stream ' + br(cur.per, 1), lx, YV(cur.per) + (YV(cur.per) > gy + gh - 20 ? -8 : 16), { size: 11.5, font: 'mono', weight: 600, color: C.b, align: la });
      });
    })();

    /* ============ Figure 3: published ladders ============ */
    (function () {
      const fig = root.querySelector('#fig-escada');
      fit(fig, 410, 520, 600);
      const D = {
        glm_code: { t: 'GLM · code', note: 'Aug 31 · 1M, MNBT (max_num_batched_tokens) 2048, 16 slots, temp 0 · median TTFT 0.4–0.6 s (×1) to 1.3–1.6 s (×16)', lv: [1, 2, 4, 8, 12, 16], agg: [41.9, 52.3, 77.9, 93.8, 116.2, 127.2], per: [44.3, 28.8, 21.0, 13.9, 11.9, 10.2], rc: 'glm' },
        glm_prose: { t: 'GLM · prose (sparkDash)', note: 'Sep 17 · 850K, adaptive-k, dense,kda, cooperative MoE, KV 14 GiB, 4 slots', lv: [1, 2, 3, 4], agg: [37.1, 51.1, 65.8, 75.3], per: [36.1, 25.0, 22.3, 19.4], ttft: ['333 ms', '365 ms', '405 ms', '401 ms'], rc: 'glm' },
        tf_prose: { t: 'GLM · TensorFold · prose', note: 'v1.3.2 · sparkDash through the API, from another machine, GPU capped at 2,200 MHz, T=0 · 1M, KV fp8, dense q4, DFlash2 + copy drafts, multi-prefill · 4-stream cap', lv: [1, 2, 3, 4], agg: [60.4, 79.2, 89.5, 108.8], per: [60.4, 40.4, 30.6, 27.9], ttft: ['170 ms', '269 ms', '330 ms', '340 ms'], rc: 'tf' },
        tf_struct: { t: 'GLM · TensorFold · structured', note: 'v1.3.2 · same conditions · served 4 at a time, the replies are identical to those served alone (11/11)', lv: [1, 2, 3, 4], agg: [114.7, 147.6, 196.3, 227.9], per: [114.7, 77.1, 67.4, 61.1], ttft: ['149 ms', '295 ms', '317 ms', '415 ms'], rc: 'tf' },
        qwen_s: { t: 'Qwen v0.30 · structured', note: 'Sep 27 · GMU 0.80, KV BF16, vocab 47k, 512 tok/stream · ±20% between single runs', lv: [1, 2, 4, 8, 16], agg: [76.0, 126.6, 165.9, 258.5, 326.3], per: [76.0, 63.9, 42.4, 32.5, 41.5], smax: 8, rc: 'qwen' },
        qwen_p: { t: 'Qwen v0.30 · prose', note: 'Sep 27 · same conditions; ×16 exceeds MAX_NUM_SEQS=8 (mean TTFT 6.5–9.7 s)', lv: [1, 2, 4, 8, 16], agg: [59.2, 75.1, 118.5, 216.4, 194.9], per: [59.2, 37.7, 29.9, 28.8, 25.5], smax: 8, rc: 'qwen' },
        qwen_c: { t: 'Qwen v0.30 · code', note: 'Sep 27 · same conditions; the aggregate plateaus above 8 streams', lv: [1, 2, 4, 8, 16], agg: [66.6, 140.8, 245.6, 313.6, 327.5], per: [66.6, 70.4, 61.4, 39.2, 42.7], smax: 8, rc: 'qwen' },
        qwen_d0: { t: 'Qwen day-0 · prose', note: 'GMU 0.835, KV fp8 + balanced 65,536 draft vocab · aggregate 3.8× from ×1 to ×8', lv: [1, 2, 4, 6, 8], agg: [54.4, 86.5, 128.1, 169.7, 207.0], per: [54.4, 45.1, 34.2, 29.7, 26.7], ttft: ['160 ms', '426 ms', '471 ms', '490 ms', '432 ms'], rc: 'qwen' },
        ds_0731: { t: 'DeepSeek 0731 · prompt 256', note: 'Aug 1 · benchmark-0731.py, natural completions ~512 tokens, server-side thinking, MTP-5, TP=2', lv: [1, 2, 4, 6], agg: [69.1, 104.9, 164.5, 191.2], per: [75.4, 58.3, 46.8, 36.9], ttft: ['0.63 s', '0.81 s', '1.26 s', '1.42 s'], rc: 'ds' },
        ds_mx: { t: 'DeepSeek · Aug 14 matrix', note: 'MiaAI matrix (0731, MTP-5), later replicated by bench-miaai.py: cold prefixes, 128 fixed tokens, thinking off · cell = median per-stream decode', heat: true, rows: ['256', '2K', '8K', '32K', '128K'], cols: [1, 2, 4, 6], vals: [[82.6, 56.9, 54.3, 35.9], [64.6, 50.1, 29.6, 20.8], [65.9, 44.2, 16.0, 10.2], [62.0, 39.7, 8.7, 8.2], [74.9, 36.0, 8.4, 8.6]], agg6: [162, 78, 29, 8, 2], rc: 'ds' },
      };
      const S = { k: 'glm_code', eff: false };
      const ctl = UI.controls(fig);
      UI.select(ctl, { label: 'Ladder', value: S.k, options: Object.keys(D).map((k) => ({ v: k, label: D[k].t })), onChange: (v) => { S.k = v; H.redraw(); } });
      UI.toggle(ctl, { label: 'scaling efficiency', value: false, onChange: (v) => { S.eff = v; H.redraw(); } });
      const C0 = V.C();
      UI.legend(ctl, [[C0.a, 'aggregate'], [C0.b, 'per stream']]);
      const H = V.fig(fig, (g, w, h) => {
        const C = V.C(), d = D[S.k], narrow = w < 600, pad = 16;
        const col = rcol(C, d.rc);
        pill(g, pad, 18, d.t, col, { font: 'sans', size: 12, weight: 600, fa: 0.18 });
        const noteH = wtext(g, d.note, pad, h - (narrow ? 34 : 18), w - 2 * pad, { size: 11, color: C.muted, lh: 14, maxLines: 3 });
        if (d.heat) {
          const top = 50, lblW = 50, right = narrow ? 64 : 110;
          const avail = w - pad * 2 - lblW - right;
          const cell = Math.min(narrow ? 62 : 86, avail / 4 - 4), chh = Math.min(46, (h - top - 90) / 5 - 3);
          const mx = pad + lblW + Math.max(0, (avail - (cell + 4) * 4) / 2);
          V.text(g, 'concurrency →', mx, top - 8, { size: 11, color: C.muted });
          const gm = V.matrix(g, { x: mx, y: top + 8, data: d.vals, w: cell * 4 + 12, h: chh * 5 + 12, gap: 3, cmap: 'seq', min: 0, max: 85, values: true, digits: 1, fmt: (v) => br(v, 1), rowLabels: d.rows, colLabels: d.cols.map((c) => 'c=' + c) });
          V.text(g, 'prompt', mx - 8, top + 2, { size: 11, color: C.muted, align: 'right' });
          const ax = gm.x + gm.w + (narrow ? 10 : 22);
          const aw = Math.max(40, Math.min(70, w - pad - ax));
          V.text(g, narrow ? 'agg c=6' : 'aggregate c=6', ax + aw / 2, top + 2, { size: 11, color: C.muted, align: 'center' });
          d.agg6.forEach((v, i) => {
            const y = gm.y + i * (gm.ch + gm.gap);
            V.box(g, ax, y, aw, gm.ch, { r: 4, fill: V.alpha(C.d, 0.12 + 0.6 * (v / 162)), stroke: V.alpha(C.d, 0.5), lw: 1 });
            V.text(g, String(v), ax + aw / 2, y + gm.ch / 2 + 4, { size: 11.5, font: 'mono', weight: 600, align: 'center', color: C.ink });
          });
          // floor highlight
          const [p0x, p0y] = gm.center(3, 2), [p1x, p1y] = gm.center(4, 3);
          g.strokeStyle = C.c; g.lineWidth = 2; g.setLineDash([4, 3]);
          V.rr(g, p0x - gm.cw / 2 - 2, p0y - gm.ch / 2 - 2, (p1x - p0x) + gm.cw + 4, (p1y - p0y) + gm.ch + 4, 6); g.stroke(); g.setLineDash([]);
          V.text(g, 'dashed: ~8 tok/s floor with serialized prefills', gm.x, gm.y + gm.h + 36, { size: 11, color: C.c, weight: 600 });
          if (S.eff) V.text(g, '(efficiency: n/a here)', w - pad, 18, { size: 11, color: C.muted, align: 'right' });
          return;
        }
        const gx = pad + 40, gy = 48, gw = w - gx - pad - 4, gh = h - gy - (narrow ? 100 : 78) - (d.ttft ? 14 : 0);
        const n = d.lv.length, ax = axis(Math.max(...d.agg, ...d.per) * 1.12), ym = ax.top;
        const XC = (i) => gx + ((i + 0.5) / n) * gw, YV = (v) => gy + gh - (v / ym) * gh;
        for (let v = 0; v <= ym + 1e-9; v += ax.step) { const y = YV(v); V.line(g, gx, y, gx + gw, y, { color: C.grid, lw: 1 }); V.text(g, br(v, 0), gx - 6, y + 4, { size: 11, font: 'mono', color: C.muted, align: 'right' }); }
        V.text(g, 'tok/s', gx - 6, gy - 8, { size: 11, font: 'mono', color: C.muted, align: 'right' });
        if (d.smax) {
          const idx = d.lv.findIndex((l) => l > d.smax);
          if (idx >= 0) { const qx = gx + (idx / n) * gw; g.fillStyle = V.alpha(C.d, 0.08); g.fillRect(qx, gy, gx + gw - qx, gh); V.text(g, 'C > ' + d.smax + ': queue', qx + 5, gy + 13, { size: 11, color: C.d }); }
        }
        const bw = Math.min(46, (gw / n) * 0.56);
        d.agg.forEach((v, i) => {
          const x = XC(i) - bw / 2, y = YV(v);
          V.box(g, x, y, bw, gy + gh - y, { r: 5, fill: V.alpha(C.a, 0.55), stroke: C.a, lw: 1 });
          const inside = Math.abs(YV(d.per[i]) - y) < 22 && gy + gh - y > 30;
          V.text(g, br(v, 1), XC(i), inside ? y + 16 : y - 6, { size: 11, font: 'mono', weight: 600, align: 'center', color: inside ? (C.dark ? '#00210F' : '#fff') : C.ink });
          if (S.eff) {
            const e = v / (d.lv[i] * d.agg[0]);
            pill(g, XC(i), Math.max(gy + 10, y - 26), br(e * 100, 0) + '%', e > 0.7 ? C.e : e > 0.4 ? C.d : C.c, { align: 'center', size: 11 });
          }
          V.text(g, '×' + d.lv[i], XC(i), gy + gh + 16, { size: 11.5, font: 'mono', align: 'center', color: C.muted });
          if (d.ttft) V.text(g, d.ttft[i], XC(i), gy + gh + 31, { size: 11, font: 'mono', align: 'center', color: C.d });
        });
        if (d.ttft) V.text(g, 'TTFT', gx - 6, gy + gh + 31, { size: 11, font: 'mono', align: 'right', color: C.d });
        g.beginPath(); d.per.forEach((v, i) => { const x = XC(i), y = YV(v); i ? g.lineTo(x, y) : g.moveTo(x, y); }); g.strokeStyle = C.b; g.lineWidth = 2.4; g.stroke();
        d.per.forEach((v, i) => {
          const x = XC(i), y = YV(v); V.dot(g, x, y, 4, C.b);
          if (narrow && !(i % 2 === 0 || i === n - 1)) return;
          const lab = br(v, 1), lw0 = tw(g, lab, 11, 'mono', 600), close = Math.abs(YV(d.agg[i]) - y) < 22;
          const lx = close ? x - bw / 2 - lw0 - 8 : x + 7, ly = close ? y + 4 : y + 15;
          V.box(g, lx - 3, ly - 11, lw0 + 6, 15, { r: 4, fill: V.alpha(C.bg, 0.85) });
          V.text(g, lab, lx, ly, { size: 11, font: 'mono', color: C.b, weight: 600 });
        });
      });
    })();

    /* ============ Figure 4: salt at the start ============ */
    (function () {
      const fig = root.querySelector('#fig-salt');
      fit(fig, 320, 470, 620);
      const S = { mode: 'inicio', reset: false, nonce: 1, t0: 0 };
      const words = 'the quarterly logistics audit recorded a routine variance in the northbound depot inventory and the team filed the report'.split(' ');
      const BT = 16, NB = 8, TAIL = 5;
      const doc = Array.from({ length: NB * BT + TAIL }, (_, i) => words[i % words.length] + (i % 7 === 3 ? i : ''));
      const fnv = (s) => { let hh = 0x811c9dc5; for (let i = 0; i < s.length; i++) { hh ^= s.charCodeAt(i); hh = Math.imul(hh, 0x01000193); } return hh >>> 0; };
      const hex = (x) => (x >>> 16).toString(16).padStart(4, '0');
      function chain(salt, where) {
        const out = []; let prev = where === 'inicio' ? fnv('salt:' + salt) : 0x9e3779b9;
        for (let b = 0; b < NB; b++) { prev = fnv(prev + '|' + doc.slice(b * BT, (b + 1) * BT).join(' ')); out.push(prev); }
        const tail = fnv(prev + '|' + doc.slice(NB * BT).join(' ') + (where === 'fim' ? ' salt:' + salt : ''));
        return { blocks: out, tail };
      }
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Nonce', options: [{ v: 'nenhum', label: 'no salt' }, { v: 'inicio', label: 'at the start' }, { v: 'fim', label: 'at the end' }], value: S.mode, onChange: (v) => { S.mode = v; restart(); } });
      UI.toggle(ctl, { label: 'reset prefix cache first', value: false, onChange: (v) => { S.reset = v; restart(); } });
      UI.button(ctl, '↻ new run', () => { S.nonce++; restart(); });
      const R = { hit: UI.readout(ctl, 'tokens from cache', '—'), comp: UI.readout(ctl, 'to compute', '—') };
      let prevC, curC, hits;
      function restart() {
        const saltPrev = S.mode === 'nenhum' ? '' : 'A' + (S.nonce * 7919 % 1000);
        const saltCur = S.mode === 'nenhum' ? '' : 'B' + (S.nonce * 104729 % 1000);
        prevC = chain(saltPrev, S.mode); curC = chain(saltCur, S.mode);
        hits = 0;
        if (!S.reset) { for (let b = 0; b < NB; b++) { if (prevC.blocks[b] === curC.blocks[b]) hits++; else break; } }
        const tot = NB * BT + TAIL + (S.mode === 'nenhum' ? 0 : 3);
        R.hit.set(hits * BT + ' of ' + tot); R.comp.set(String(tot - hits * BT));
        S.t0 = performance.now();
      }
      restart();
      V.fig(fig, (g, w, h) => {
        const C = V.C(), narrow = w < 620, pad = 14;
        const el = (performance.now() - S.t0) / 1000;
        const per = narrow ? 5 : 9;
        const cellW = Math.min(76, (w - 2 * pad - (per - 1) * 8 - (narrow ? 0 : 30)) / per), cellH = narrow ? 48 : 52;
        const rowsPer = narrow ? 2 : 1;
        const x0 = pad + (narrow ? 0 : 30);
        const scanI = Math.floor(el / 0.45);
        const subGap = narrow ? 26 : 10;
        const drawRow = (label, ch, y, isCur) => {
          V.text(g, label, pad, y - 24, { size: 11, font: 'mono', color: C.muted, weight: 600 });
          for (let b = 0; b <= NB; b++) {
            const r = Math.floor(b / per), cI = b % per;
            const x = x0 + cI * (cellW + 8), yy = y + r * (cellH + subGap);
            const isTail = b === NB;
            const empty = !isCur && S.reset;
            let fill = V.alpha(C.muted, 0.08), stroke = V.alpha(C.muted, 0.4), txt = C.muted;
            if (isCur && b <= scanI) {
              if (isTail) { fill = V.alpha(C.c, 0.14); stroke = V.alpha(C.c, 0.7); txt = C.ink; }
              else if (b < hits) { fill = V.alpha(C.a, 0.3); stroke = C.a; txt = C.ink; }
              else { fill = V.alpha(C.c, 0.25); stroke = C.c; txt = C.ink; }
            } else if (!isCur && !empty && !isTail) { fill = V.alpha(C.b, 0.14); stroke = V.alpha(C.b, 0.6); txt = C.ink; }
            if (!isCur && isTail) { fill = 'rgba(0,0,0,0)'; stroke = V.alpha(C.muted, 0.3); }
            V.box(g, x, yy, isTail ? cellW * 0.62 : cellW, cellH, { r: 7, fill, stroke, lw: 1.3 });
            if (isTail && !isCur) { g.setLineDash([3, 3]); V.rr(g, x, yy, cellW * 0.62, cellH, 7); g.strokeStyle = V.alpha(C.muted, 0.4); g.stroke(); g.setLineDash([]); }
            const cw2 = isTail ? cellW * 0.62 : cellW;
            V.text(g, isTail ? 'tail' : 'B' + b, x + cw2 / 2, yy + 17, { size: 11, font: 'mono', weight: 600, align: 'center', color: txt });
            const hh = empty ? '····' : isTail ? (isCur ? hex(ch.tail) : 'n/c') : hex(ch.blocks[b]);
            V.text(g, hh, x + cw2 / 2, yy + 36, { size: 11, font: 'mono', align: 'center', color: empty ? C.muted : txt });
            if (b < NB && b % per !== per - 1) V.line(g, x + cellW + 1, yy + cellH / 2, x + cellW + 7, yy + cellH / 2, { color: V.alpha(C.muted, 0.5), lw: 1.2 });
          }
          if (S.mode !== 'nenhum') {
            const sb = S.mode === 'inicio' ? 0 : NB;
            const r = Math.floor(sb / per), cI = sb % per;
            const cwS = S.mode === 'inicio' ? cellW : cellW * 0.62;
            const x = x0 + cI * (cellW + 8) + (S.mode === 'inicio' ? 0 : cwS), yy = y + r * (cellH + subGap) - 11;
            pill(g, x, yy, (isCur ? 'nonce B' : 'nonce A') + (S.mode === 'inicio' ? ' → h₀' : ''), C.d, { size: 11, align: S.mode === 'inicio' ? 'left' : 'right', fa: 0.35 });
          }
          return y + rowsPer * (cellH + subGap) - subGap + 10;
        };
        const y1 = 46;
        const yEnd1 = drawRow(S.reset ? 'CACHE (emptied by the reset)' : 'CACHE (previous run)', prevC, y1, false);
        const y2 = yEnd1 + 40;
        const yEnd2 = drawRow('THIS REQUEST', curC, y2, true);
        // comparison cursor
        if (scanI <= NB) {
          const b = Math.min(scanI, NB), r = Math.floor(b / per), cI = b % per;
          const x = x0 + cI * (cellW + 8), cwb = b === NB ? cellW * 0.62 : cellW;
          [y1, y2].forEach((yy0) => { g.strokeStyle = C.ink; g.lineWidth = 2; V.rr(g, x - 3, yy0 + r * (cellH + subGap) - 3, cwb + 6, cellH + 6, 9); g.stroke(); });
        }
        // summary
        const done = scanI > NB;
        const yS = yEnd2 + 10;
        const tot = NB * BT + TAIL + (S.mode === 'nenhum' ? 0 : 3), comp = tot - hits * BT;
        const bwid = w - 2 * pad, barY = yS + 18;
        V.text(g, 'prefill to compute (proportional to TTFT)', pad, yS + 6, { size: 11, color: C.muted });
        V.box(g, pad, barY, bwid, 14, { r: 7, fill: V.alpha(C.muted, 0.12) });
        const u = done ? 1 : V.clamp(el / (0.45 * (NB + 1)), 0, 1);
        V.box(g, pad, barY, Math.max(6, bwid * (comp / tot) * u), 14, { r: 7, fill: comp / tot > 0.5 ? V.alpha(C.c, 0.75) : V.alpha(C.a, 0.8) });
        let verdict;
        if (S.reset) verdict = 'Reset: empty cache, a truly cold prefill (' + comp + ' tokens computed).';
        else if (S.mode === 'inicio') verdict = 'Nonce at the start changes h₀ and the whole chain: 0 hits, cold prefill.';
        else if (S.mode === 'fim') verdict = 'Nonce at the end only changes the tail: ' + hits * BT + ' tokens come from the cache. The “cold” is fake.';
        else verdict = 'No salt: ' + hits * BT + ' tokens reused. Only the partial tail (never cached) is recomputed.';
        wtext(g, verdict, pad, barY + 32, bwid, { size: 12, color: S.mode === 'inicio' || S.reset ? C.a : C.c, weight: 600, lh: 16, maxLines: 3 });
      });
    })();

    /* ============ Figure 5: counters as odometers ============ */
    (function () {
      const fig = root.querySelector('#fig-hodometro');
      fit(fig, 430, 570, 640);
      const S = { foreign: true };
      const ctl = UI.controls(fig);
      UI.toggle(ctl, { label: 'other traffic during the measurement (client B)', value: true, onChange: (v) => { S.foreign = v; st.go(st.i); } });
      const steps = [
        { t: 'Odometers since boot', d: 'The <code>_total</code> counters accumulate everything the server has processed since boot, from all clients. Here the history is dominated by a synthetic prompt that accepts almost everything.' },
        { t: 'Snapshot before', d: 'Read <code>/metrics</code> and save the values (<code>mtp_accept.py --save before.json</code>). Nothing has been measured yet: you have only marked the odometer.' },
        { t: 'Workload (and the intruder)', d: 'Your benchmark (client A, prose) runs. If another client uses the server at the same time, the counters add both up, and nothing in /metrics separates one from the other.' },
        { t: 'Snapshot after, and delta', d: 'Read again and subtract. tokens/step = 1 + Δaccepted/Δdrafts; \\(a_i\\) = Δper_pos[i]/Δdrafts; ms/step = Δsum/Δcount of the ITL histogram.' },
        { t: 'Three answers to the same question', d: 'Dividing the totals describes the server\'s history. The delta with other traffic mixes the clients. Only the delta with an idle server describes your workload.' },
        { t: 'A restart zeroes everything', d: 'After a restart the counters start again from zero. A negative (or absurdly small) delta means the engine restarted in the middle of the measurement: discard it.' },
      ];
      const st = UI.stepper(fig, steps, { autoplay: 7500, onStep: () => { S.t0 = performance.now(); } });
      S.t0 = performance.now();
      const keys = ['drafts', 'draft_tokens', 'accepted', 'pos0', 'pos1', 'pos2', 'itl_sum', 'itl_count'];
      const labels = { drafts: 'num_drafts', draft_tokens: 'num_draft_tokens', accepted: 'num_accepted_tokens', pos0: 'per_pos{position="0"}', pos1: 'per_pos{position="1"}', pos2: 'per_pos{position="2"}', itl_sum: 'itl_seconds_sum', itl_count: 'itl_seconds_count' };
      const mk = (drafts, a, ms) => { const p = a.map((x) => Math.round(drafts * x)); return { drafts, draft_tokens: drafts * 3, accepted: p[0] + p[1] + p[2], pos0: p[0], pos1: p[1], pos2: p[2], itl_sum: drafts * ms / 1000, itl_count: drafts }; };
      const add = (x, y) => { const o = {}; keys.forEach((k) => { o[k] = x[k] + y[k]; }); return o; };
      const sub = (x, y) => { const o = {}; keys.forEach((k) => { o[k] = x[k] - y[k]; }); return o; };
      const HIST = mk(48000, [0.97, 0.93, 0.9], 60), WA = mk(1200, [0.81, 0.64, 0.51], 59.3), WB = mk(400, [1.0, 0.99, 0.99], 50);
      const ZERO = mk(0, [0, 0, 0], 0), RST = mk(350, [0.8, 0.62, 0.5], 59);
      const derive = (d) => ({ tps: d.drafts ? 1 + d.accepted / d.drafts : NaN, ms: d.itl_count ? (1000 * d.itl_sum) / d.itl_count : NaN, pp: [d.pos0, d.pos1, d.pos2].map((x) => (d.drafts ? x / d.drafts : NaN)) });
      V.fig(fig, (g, w, h) => {
        const C = V.C(), narrow = w < 640, si = st.i, el = (performance.now() - S.t0) / 1000, pad = 14;
        const ease = V.ease.inOut(V.clamp(el / 1.4, 0, 1));
        const after = add(add(HIST, WA), S.foreign ? WB : ZERO);
        let now, before = null, delta = null;
        if (si === 0) now = add(HIST, mk(Math.round(el * 30), [1, 0.99, 0.99], 50));
        else if (si === 1) { now = HIST; before = HIST; }
        else if (si === 2) { before = HIST; const part = mk(Math.round(1200 * ease), [0.81, 0.64, 0.51], 59.3), pb = S.foreign ? mk(Math.round(400 * ease), [1, 0.99, 0.99], 50) : ZERO; now = add(add(HIST, part), pb); }
        else if (si === 3 || si === 4) { before = HIST; now = after; delta = sub(after, HIST); }
        else { before = after; now = RST; delta = sub(RST, after); }
        // clients and server
        const topH = narrow ? 84 : 64;
        const srvX = narrow ? w / 2 - 70 : pad + 190, srvW = 140;
        V.box(g, srvX, narrow ? 8 : 12, srvW, narrow ? 32 : 40, { r: 10, fill: V.alpha(C.b, 0.14), stroke: C.b, lw: 1.3, label: 'vLLM /metrics', size: 12 });
        const cli = [['A · your bench', C.a, true], ['B · other client', C.d, S.foreign]];
        cli.forEach(([l, col, on], i) => {
          const cx = narrow ? (i ? w - pad - 66 : pad + 58) : pad + 55, cy = narrow ? 62 : 18 + i * 28;
          const active = on && (si === 2 || (i === 1 && si === 0));
          pill(g, cx, cy + (narrow ? 0 : 6), l, col, { font: 'sans', size: 11, align: 'center', fa: active ? 0.3 : 0.08 });
          if (active) {
            const sx = narrow ? cx : cx + 58, sy = narrow ? cy - 10 : cy + 6, ex = narrow ? (i ? srvX + srvW - 20 : srvX + 20) : srvX, ey = narrow ? 40 : 32;
            const tt = (performance.now() / 1000 * (i ? 0.9 : 1.3)) % 1;
            V.line(g, sx, sy, ex, ey, { color: V.alpha(col, 0.4), lw: 1, dash: [3, 4] });
            const p = V.along(sx, sy, ex, ey, tt); V.dot(g, p[0], p[1], 3, col);
          }
        });
        // table
        const tx = pad, ty = topH + 16, rowH = narrow ? 22 : 24;
        const tWidth = narrow ? w - 2 * pad : w * 0.58;
        const cN = tx + (narrow ? 118 : 178), cA = tx + tWidth * (narrow ? 0.54 : 0.55), cB = tx + tWidth * (narrow ? 0.78 : 0.78), cD = tx + tWidth;
        V.text(g, narrow ? 'counter' : 'vllm:spec_decode_* / ITL', tx, ty, { size: 11, font: 'mono', color: C.muted, weight: 600 });
        V.text(g, 'before', cA, ty, { size: 11, font: 'mono', color: C.muted, align: 'right', weight: 600 });
        V.text(g, 'now', cB, ty, { size: 11, font: 'mono', color: C.muted, align: 'right', weight: 600 });
        V.text(g, 'Δ', cD, ty, { size: 11, font: 'mono', color: C.muted, align: 'right', weight: 600 });
        const fmtv = (k, v) => (v == null ? '—' : k === 'itl_sum' ? br(v, 1) : brk(v));
        keys.forEach((k, i) => {
          const y = ty + 10 + (i + 1) * rowH;
          if (i % 2 === 0) { g.fillStyle = V.alpha(C.muted, 0.06); g.fillRect(tx - 4, y - rowH + 6, tWidth + 8, rowH); }
          V.text(g, narrow ? labels[k].replace('num_', '').replace('per_pos{position=', 'pos{').replace('itl_seconds', 'itl') : labels[k], tx, y, { size: 11, font: 'mono', color: C.ink, maxW: cA - tx - 70 });
          V.text(g, before ? fmtv(k, before[k]) : '—', cA, y, { size: 11.5, font: 'mono', color: C.muted, align: 'right' });
          V.text(g, fmtv(k, now[k]), cB, y, { size: 11.5, font: 'mono', color: C.ink, align: 'right', weight: 600 });
          const dv = delta ? delta[k] : null;
          V.text(g, dv == null ? '—' : (dv < 0 ? '−' : '+') + fmtv(k, Math.abs(dv)), cD, y, { size: 11.5, font: 'mono', color: dv == null ? C.muted : dv < 0 ? C.red : C.a, align: 'right', weight: 600 });
        });
        // derived panel
        const px = narrow ? pad : tx + tWidth + 26, py = narrow ? ty + 10 + 9 * rowH + 18 : topH + 4;
        const pw = narrow ? w - 2 * pad : w - px - pad, ph = h - py - pad;
        V.box(g, px, py, pw, ph, { r: 10, fill: V.alpha(C.a, 0.05), stroke: V.alpha(C.muted, 0.25), lw: 1 });
        const truth = derive(WA), dd = delta ? derive(delta) : null, cum = derive(now);
        const rows = [];
        if (si >= 3 && si <= 4) {
          rows.push(['cumulative (totals ÷ totals)', cum, C.c]);
          rows.push([S.foreign ? 'Δ with client B mixed in' : 'Δ with idle server', dd, S.foreign ? C.d : C.a]);
          rows.push(['truth: client A only', truth, C.a]);
        } else if (si === 5) rows.push(['Δ after the restart', dd, C.red]);
        else rows.push(['cumulative now', cum, C.c]);
        V.text(g, 'tokens/step and per-position acceptance', px + 12, py + 20, { size: 11, font: 'mono', color: C.muted, weight: 600, maxW: pw - 20 });
        const rh = Math.min(62, (ph - 34) / Math.max(1, rows.length));
        rows.forEach(([l, dv, col], i) => {
          const y = py + 34 + i * rh;
          V.text(g, l, px + 12, y + 12, { size: 11.5, color: col, weight: 600, maxW: pw - 24 });
          if (!dv || !(dv.tps > 0)) { V.text(g, 'negative delta: engine restarted → discard', px + 12, y + 32, { size: 11.5, color: C.red, maxW: pw - 24 }); return; }
          V.text(g, br(dv.tps, 2) + ' tok/step · ' + br(dv.ms, 0) + ' ms', px + 12, y + 31, { size: 11, font: 'mono', color: C.ink });
          const bw2 = 26, bx = px + pw - 3 * (bw2 + 5) - 6;
          dv.pp.forEach((a, j) => { const bh = V.clamp(a, 0, 1) * (rh - 24); V.box(g, bx + j * (bw2 + 5), y + rh - 8 - bh, bw2, Math.max(1, bh), { r: 3, fill: V.alpha(col, 0.55) }); V.text(g, br(a, 2), bx + j * (bw2 + 5) + bw2 / 2, y + rh - 11 - bh, { size: 11, font: 'mono', align: 'center', color: C.ink }); });
          if (i < 2 && si === 4) { const err = dv.tps / truth.tps - 1; V.text(g, (err >= 0 ? '+' : '') + br(err * 100, 0) + '% vs truth', px + 12, y + 48, { size: 11, font: 'mono', color: Math.abs(err) > 0.02 ? C.c : C.a }); }
        });
        // the explicit arithmetic
        const yFm = py + 34 + rows.length * rh + 14;
        let fm;
        if (si <= 2) fm = si === 0 ? 'Without a snapshot, any ratio is cumulative: it describes the server\'s history, not your test.' : 'Δ still undefined: the snapshot after the workload is missing.';
        else if (si === 5) fm = 'Δdrafts = ' + brk(delta.drafts) + ' < 0: the counters restarted. Redo the whole window.';
        else fm = 'tok/step = 1 + Δaccepted/Δdrafts = 1 + ' + brk(delta.accepted) + '/' + brk(delta.drafts) + ' = ' + br(1 + delta.accepted / delta.drafts, 2) + ' · ms/step = 1000·Δsum/Δcount = ' + br((1000 * delta.itl_sum) / delta.itl_count, 1);
        if (yFm < py + ph - 20) wtext(g, fm, px + 12, yFm, pw - 24, { size: 11, font: 'mono', color: C.ink, lh: 15, maxLines: Math.max(1, Math.floor((py + ph - yFm - 6) / 15)) });
      }, { static: false });
    })();

    /* ============ Figure 6: acceptance curves ============ */
    (function () {
      const fig = root.querySelector('#fig-aceitacao');
      fit(fig, 390, 500, 600);
      const D = {
        glm_s: { t: 'GLM · structured (DFlash2, k=7)', a: [0.98, 0.98, 0.94, 0.94, 0.91, 0.83, 0.83], rc: 'glm', n: 'lab Aug 30, “count from 1 to 200” (synthetic). Per-position medians: Σaᵢ = 6.41 ≠ 6.71 per run', flag: 'sint' },
        glm_p: { t: 'GLM · prose (DFlash2, k=7)', a: [0.75, 0.58, 0.41, 0.28, 0.16, 0.09, 0.06], rc: 'glm', n: 'lab Aug 30, hash map, same server. Per-position medians: Σaᵢ = 2.33 ≠ 2.39 per run' },
        qw_r: { t: 'Qwen · README (MTP, k=3)', a: [0.89, 0.745, 0.6], rc: 'qwen', n: 'batch-1 greedy decode; the sanity test for the MTP FP8 scales' },
        qw_p: { t: 'Qwen · sparkDash prose (MTP, k=3)', a: [0.81, 0.64, 0.51], rc: 'qwen', n: 'Sep 16 rerun: the recipe\'s representative prompt' },
        qw_c: { t: 'Qwen · sparkDash code (synthetic)', a: [1.0, 0.99, 0.99], rc: 'qwen', n: '50 identical clamp_NN helpers: ~1.0 acceptance that does not exist in real traffic', flag: 'sint' },
        ds_t3: { t: 'DeepSeek · TP=3, 10 s windows (DSpark, k=6)', a: [0.92, 0.74, 0.6, 0.41, 0.28, 0.21], rc: 'ds', n: 'fable5-1: 0.28/0.21 at the end, the argument for not going past k=6' },
        ds_m: { t: 'DeepSeek · results/dspark-metrics.txt', a: [84 / 105, 63 / 105, 1 / 105, 1 / 105, 0, 0], rc: 'ds', n: '105 steps, 84/63/1/1/0/0 accepted: collapse after the 2nd position, no context in the repo', flag: 'susp' },
        bug: { t: 'vllm#53030 signature (illustrative)', a: [1, 1, 1, 1, 1, 1, 1], rc: 'glm', n: 'acceptance pinned at 1.00 at every position: BatchDescriptor collision in PIECEWISE graphs', flag: 'bug' },
      };
      const S = { k: 'glm_p', kk: 7, geo: true };
      const ctl = UI.controls(fig);
      let sk;
      UI.select(ctl, { label: 'Curve', value: S.k, options: Object.keys(D).map((k) => ({ v: k, label: D[k].t })), onChange: (v) => { S.k = v; const n = D[v].a.length; sk.input.max = n; S.kk = n; sk.set(n); H.redraw(); } });
      sk = UI.slider(ctl, { label: 'verified k', min: 1, max: 7, step: 1, value: 7, onInput: (v) => { S.kk = Math.min(v, D[S.k].a.length); H.redraw(); } });
      UI.toggle(ctl, { label: 'geometric model pⁱ (p = a₁)', value: true, onChange: (v) => { S.geo = v; H.redraw(); } });
      const H = V.fig(fig, (g, w, h) => {
        const C = V.C(), d = D[S.k], narrow = w < 600, pad = 16, col = rcol(C, d.rc);
        const k = Math.min(S.kk, d.a.length), a = d.a.slice(0, k);
        const sum = a.reduce((p, q) => p + q, 0), tps = 1 + sum, alpha = sum / k, marg = a[k - 1];
        pill(g, pad, 18, d.t, col, { font: 'sans', size: 12, weight: 600, fa: 0.18 });
        const rw = narrow ? 0 : 230;
        const gx = pad + 30, gy = 48, gw = w - gx - pad - rw - (narrow ? 0 : 20), gh = h - gy - (narrow ? 150 : 70);
        const n = d.a.length, bw = Math.min(54, (gw / n) * 0.62);
        const XC = (i) => gx + ((i + 0.5) / n) * gw, YV = (v) => gy + gh - v * gh;
        [0, 0.25, 0.5, 0.75, 1].forEach((v) => { const y = YV(v); V.line(g, gx, y, gx + gw, y, { color: C.grid, lw: 1 }); V.text(g, br(v, 2), gx - 6, y + 4, { size: 11, font: 'mono', color: C.muted, align: 'right' }); });
        d.a.forEach((v, i) => {
          const on = i < k, x = XC(i) - bw / 2, y = YV(v);
          V.box(g, x, y, bw, Math.max(1.5, gy + gh - y), { r: 5, fill: V.alpha(col, on ? 0.6 : 0.14), stroke: on ? col : V.alpha(col, 0.35), lw: 1 });
          V.text(g, br(v, 2), XC(i), y - 6, { size: 11, font: 'mono', weight: 600, align: 'center', color: on ? C.ink : C.muted });
          V.text(g, '#' + (i + 1), XC(i), gy + gh + 15, { size: 11, font: 'mono', align: 'center', color: C.muted });
        });
        V.text(g, 'draft position (in /metrics, position="0" is #1)', gx + gw / 2, gy + gh + 31, { size: 11, color: C.muted, align: 'center', maxW: gw });
        if (S.geo) {
          const a1 = d.a[0];
          g.strokeStyle = C.d; g.lineWidth = 1.6; g.setLineDash([5, 4]); g.beginPath();
          d.a.forEach((_, i) => { const x = XC(i), y = YV(Math.pow(a1, i + 1)); i ? g.lineTo(x, y) : g.moveTo(x, y); }); g.stroke(); g.setLineDash([]);
          d.a.forEach((_, i) => V.dot(g, XC(i), YV(Math.pow(a1, i + 1)), 2.6, C.d));
        }
        // verdict
        let verdict, vcol;
        const mono = d.a.every((v, i) => i === 0 || v <= d.a[i - 1] + 0.005);
        const collapse = d.a.some((v, i) => i > 0 && d.a[i - 1] > 0.3 && v / d.a[i - 1] < 0.1);
        if (d.a.every((v) => v >= 0.999)) { verdict = '✗ pinned at ~1.00: the gate fails (exit 1)'; vcol = C.red; }
        else if (d.a.every((v) => v >= 0.98)) { verdict = '⚠ ≥ 0.98 everywhere: synthetic prompt, not real traffic'; vcol = C.d; }
        else if (collapse) { verdict = '⚠ abrupt collapse: investigate before trusting it'; vcol = C.d; }
        else if (!mono) { verdict = '✗ does not decay: suspect scales or layout'; vcol = C.red; }
        else if (d.flag === 'sint') { verdict = '⚠ nearly deterministic: decays as it should, but is not real traffic'; vcol = C.d; }
        else { verdict = '✓ decays monotonically: healthy curve'; vcol = C.e; }
        const bx = narrow ? pad : w - pad - rw, by = narrow ? gy + gh + 46 : gy;
        const bwd = narrow ? w - 2 * pad : rw;
        V.box(g, bx, by, bwd, narrow ? 96 : gh, { r: 10, fill: V.alpha(col, 0.06), stroke: V.alpha(col, 0.35), lw: 1 });
        const lines2 = [
          ['accepted/step Σaᵢ', br(sum, 2)],
          ['tokens/step 1+Σaᵢ', br(tps, 2)],
          ['acceptance rate Σaᵢ/k', br(alpha * 100, 1) + '%'],
          ['gain from position #' + k, '+' + br(marg, 2)],
        ];
        if (narrow) {
          const shortL = ['accepted/step', 'tokens/step', 'rate', 'gain #' + k];
          lines2.forEach(([l, v], i) => { const cx = bx + 10 + (i % 2) * (bwd / 2), cy = by + 20 + Math.floor(i / 2) * 22, ll = shortL[i] + ': '; V.text(g, ll, cx, cy, { size: 11, color: C.muted }); V.text(g, v, cx + tw(g, ll, 11) + 2, cy, { size: 11.5, font: 'mono', weight: 600, color: C.ink }); });
          wtext(g, verdict, bx + 10, by + 70, bwd - 20, { size: 11.5, color: vcol, weight: 600, maxLines: 2, lh: 14 });
        } else {
          lines2.forEach(([l, v], i) => { const cy = by + 24 + i * 36; V.text(g, l, bx + 12, cy, { size: 11, color: C.muted }); V.text(g, v, bx + 12, cy + 17, { size: 15, font: 'mono', weight: 700, color: C.ink }); });
          wtext(g, verdict, bx + 12, by + 172, bwd - 24, { size: 12, color: vcol, weight: 600, maxLines: 3, lh: 16 });
          wtext(g, d.n, bx + 12, by + 224, bwd - 24, { size: 11, color: C.muted, maxLines: 3, lh: 14 });
        }
      }, { static: true });
    })();

    /* ============ Figure 7: A/B/A simulator ============ */
    (function () {
      const fig = root.querySelector('#fig-aba');
      fit(fig, 440, 600, 660);
      const S = { eff: 0, sig: 5, drift: -6, n: 5, design: 'aba', outlier: false, seed: 959225523, t0: performance.now() };
      const MU = 55;
      const ctl = UI.controls(fig);
      UI.slider(ctl, { label: 'True effect', min: -10, max: 10, step: 1, value: S.eff, fmt: (v) => (v > 0 ? '+' : '') + v + '%', onInput: (v) => { S.eff = v; run(); } });
      UI.slider(ctl, { label: 'Noise σ', min: 1, max: 12, step: 1, value: S.sig, fmt: (v) => v + '%', onInput: (v) => { S.sig = v; run(); } });
      UI.slider(ctl, { label: 'Session drift', min: -10, max: 10, step: 1, value: S.drift, fmt: (v) => (v > 0 ? '+' : '') + v + '%', onInput: (v) => { S.drift = v; run(); } });
      UI.slider(ctl, { label: 'Trials per arm', min: 3, max: 12, step: 1, value: S.n, onInput: (v) => { S.n = v; run(); } });
      UI.seg(ctl, { label: 'Design', options: [{ v: 'ab', label: 'naive A/B' }, { v: 'aba', label: 'A/B/A' }], value: 'aba', onChange: (v) => { S.design = v; run(); } });
      UI.toggle(ctl, { label: 'cold 1st trial (autotune)', value: false, onChange: (v) => { S.outlier = v; run(); } });
      UI.button(ctl, '▶ new session', () => { S.seed = (S.seed * 48271) % 2147483647; run(); }, { primary: true });
      const rRes = UI.readout(ctl, 'resolution (2 SE of the difference)', '—');
      let sess = null, mc = null;
      function simulate(seed, design, keepSamples) {
        const r = V.rng(seed);
        const arms = design === 'aba' ? ['A1', 'B', 'A2'] : ['A1', 'B'];
        const out = {};
        arms.forEach((nm, s) => {
          const xs = [];
          for (let j = 0; j < S.n; j++) {
            const tau = (s + (j + 0.5) / S.n) / 3;
            let v = MU * (1 + (S.drift / 100) * tau) * (nm === 'B' ? 1 + S.eff / 100 : 1) * (1 + (S.sig / 100) * r.normal());
            if (S.outlier && s === 0 && j === 0) v *= 0.55;
            xs.push(v);
          }
          out[nm] = xs;
        });
        const med = {}; arms.forEach((a) => { med[a] = median(out[a]); });
        const r1 = med.B / med.A1 - 1, r2 = design === 'aba' ? med.B / med.A2 - 1 : r1;
        const verdict = r1 >= 0.03 && r2 >= 0.03 ? 'win' : r1 <= -0.03 && r2 <= -0.03 ? 'loss' : 'tie';
        const truth = S.eff >= 3 ? 'win' : S.eff <= -3 ? 'loss' : 'tie';
        const est = design === 'aba' ? med.B / ((med.A1 + med.A2) / 2) - 1 : r1;
        return keepSamples ? { out, med, arms, r1, r2, verdict, truth, est, drift: design === 'aba' ? med.A2 / med.A1 - 1 : null } : verdict === truth;
      }
      function run() {
        sess = simulate(S.seed, S.design, true);
        let okN = 0, okA = 0; const NS = 400;
        for (let i = 0; i < NS; i++) { if (simulate(1000 + i * 13, 'ab', false)) okN++; if (simulate(1000 + i * 13, 'aba', false)) okA++; }
        mc = { naive: 1 - okN / NS, aba: 1 - okA / NS };
        rRes.set('±' + br(2 * Math.SQRT2 * 1.25 * S.sig / Math.sqrt(S.n), 1) + '%');
        S.t0 = performance.now();
      }
      run();
      V.fig(fig, (g, w, h) => {
        const C = V.C(), narrow = w < 660, pad = 16, el = (performance.now() - S.t0) / 1000;
        const s = sess, arms = s.arms;
        const all = arms.flatMap((a) => s.out[a]);
        const lo = Math.min(...all), hi = Math.max(...all);
        const ymin = Math.floor((lo - 3) / 5) * 5, ymax = Math.ceil((hi + 3) / 5) * 5;
        const rw = narrow ? 0 : Math.min(290, w * 0.36);
        const gx = pad + 38, gy = 20, gw = w - gx - pad - rw - (narrow ? 0 : 18), gh = narrow ? h * 0.5 - 30 : h - gy - 120;
        const YV = (v) => gy + gh - ((v - ymin) / (ymax - ymin)) * gh;
        for (let v = ymin; v <= ymax; v += 5) { const y = YV(v); V.line(g, gx, y, gx + gw, y, { color: C.grid, lw: 1 }); V.text(g, String(v), gx - 6, y + 4, { size: 11, font: 'mono', color: C.muted, align: 'right' }); }
        V.text(g, 'tok/s', gx - 6, gy - 6, { size: 11, font: 'mono', color: C.muted, align: 'right' });
        // baseline band
        const base = arms.filter((a) => a !== 'B').flatMap((a) => s.out[a]);
        const bl = Math.min(...base), bh = Math.max(...base);
        g.fillStyle = V.alpha(C.muted, 0.12); g.fillRect(gx, YV(bh), gw, YV(bl) - YV(bh));
        V.text(g, 'baseline band', gx + gw - 4, YV(bh) - 5, { size: 11, color: C.muted, align: 'right' });
        // true trend (drift)
        g.strokeStyle = V.alpha(C.d, 0.7); g.lineWidth = 1.4; g.setLineDash([5, 4]); g.beginPath();
        const nA = arms.length;
        for (let k = 0; k <= 30; k++) { const tau = (k / 30) * (nA / 3); const x = gx + (tau / (nA / 3)) * gw; const y = YV(MU * (1 + (S.drift / 100) * tau)); k ? g.lineTo(x, y) : g.moveTo(x, y); }
        g.stroke(); g.setLineDash([]);
        g.strokeStyle = C.d; g.lineWidth = 1.4; g.setLineDash([5, 4]); g.beginPath(); g.moveTo(gx + 6, gy + gh - 10); g.lineTo(gx + 26, gy + gh - 10); g.stroke(); g.setLineDash([]);
        V.text(g, 'true baseline (with drift)', gx + 32, gy + gh - 6, { size: 11, color: C.d });
        const colW = gw / nA;
        const rj = V.rng(3);
        arms.forEach((a, ai) => {
          const cx = gx + (ai + 0.5) * colW, col = a === 'B' ? C.c : C.b;
          V.text(g, a === 'A1' ? (S.design === 'aba' ? 'A₁' : 'A') : a === 'A2' ? 'A₂' : 'B', cx, gy + gh + 18, { size: 13, weight: 700, align: 'center', color: col });
          s.out[a].forEach((v, j) => {
            const appear = V.clamp((el - (ai * S.n + j) * 0.05) / 0.25, 0, 1);
            if (appear <= 0) return;
            const jx = cx + (rj() - 0.5) * Math.min(46, colW * 0.5);
            g.globalAlpha = appear; V.dot(g, jx, YV(v), 4.2, V.alpha(col, 0.85)); g.globalAlpha = 1;
          });
          const md = s.med[a], mn = mean(s.out[a]);
          V.line(g, cx - colW * 0.32, YV(md), cx + colW * 0.32, YV(md), { color: col, lw: 3 });
          const my = YV(mn); g.strokeStyle = C.ink; g.lineWidth = 1.4; g.beginPath(); g.moveTo(cx + colW * 0.36, my - 5); g.lineTo(cx + colW * 0.36 + 5, my); g.lineTo(cx + colW * 0.36, my + 5); g.lineTo(cx + colW * 0.36 - 5, my); g.closePath(); g.stroke();
          V.text(g, br(md, 1), cx - colW * 0.34, YV(md) - 6, { size: 11, font: 'mono', weight: 600, color: col });
        });
        // verdict panel
        const bx = narrow ? pad : w - pad - rw, by = narrow ? gy + gh + 34 : gy;
        const bw = narrow ? w - 2 * pad : rw, bhh = narrow ? 134 : gh + 26;
        const vcol = s.verdict === s.truth ? C.e : C.red;
        V.box(g, bx, by, bw, bhh, { r: 10, fill: V.alpha(vcol, 0.07), stroke: V.alpha(vcol, 0.45), lw: 1.2 });
        let y = by + 22;
        V.text(g, 'THIS SESSION', bx + 12, y, { size: 11, font: 'mono', color: C.muted, weight: 600 }); y += 22;
        V.text(g, 'estimated effect ' + (s.est >= 0 ? '+' : '') + br(s.est * 100, 1) + '%', bx + 12, y, { size: 13, font: 'mono', weight: 700, color: C.ink }); y += 20;
        if (s.drift != null) { V.text(g, 'measured drift A₂/A₁ ' + (s.drift >= 0 ? '+' : '') + br(s.drift * 100, 1) + '%', bx + 12, y, { size: 11.5, font: 'mono', color: C.d }); y += 18; }
        V.text(g, 'rule: ≥ +3% against ' + (S.design === 'aba' ? 'each baseline' : 'the baseline'), bx + 12, y, { size: 11, color: C.muted, maxW: bw - 24 }); y += 20;
        V.text(g, 'verdict: ' + s.verdict + ' · truth: ' + s.truth, bx + 12, y, { size: 12, weight: 700, color: vcol, maxW: bw - 24 }); y += 18;
        V.text(g, s.verdict === s.truth ? '✓ correct conclusion' : '✗ wrong conclusion', bx + 12, y, { size: 12, weight: 600, color: vcol });
        if (S.outlier && !narrow) {
          y += 26;
          const a1 = s.out.A1;
          wtext(g, 'Arm A₁ with the cold trial: mean ' + br(mean(a1), 1) + ' versus median ' + br(median(a1), 1) + ' tok/s. The median resists the outlier.', bx + 12, y, bw - 24, { size: 11, color: C.ink, lh: 14, maxLines: 4 });
        }
        // Monte Carlo
        const my2 = narrow ? by + bhh + 18 : gy + gh + 44;
        const mw = narrow ? w - 2 * pad : gw + 38, mx = narrow ? pad : pad;
        V.text(g, 'IN 400 SESSIONS WITH THESE PARAMETERS · wrong verdict', mx, my2, { size: 11, font: 'mono', color: C.muted, weight: 600, maxW: mw });
        [['naive A/B', mc.naive, C.c], ['A/B/A', mc.aba, C.a]].forEach(([l, v, col], i) => {
          const yy = my2 + 14 + i * 24, lw0 = narrow ? 92 : 104;
          V.text(g, l, mx, yy + 11, { size: 11.5, color: C.ink, weight: 600 });
          V.box(g, mx + lw0, yy, mw - lw0 - 56, 14, { r: 7, fill: V.alpha(C.muted, 0.12) });
          V.box(g, mx + lw0, yy, Math.max(4, (mw - lw0 - 56) * v), 14, { r: 7, fill: V.alpha(col, 0.8) });
          V.text(g, br(v * 100, 0) + '%', mx + mw - 4, yy + 11, { size: 12, font: 'mono', weight: 700, color: C.ink, align: 'right' });
        });
      });
    })();

    /* ============ Figure 8: step cost model ============ */
    (function () {
      const fig = root.querySelector('#fig-custo');
      fit(fig, 410, 500, 660);
      const GiB = 1.073741824;
      const P = {
        qwen: {
          t: 'Qwen3.8 · TP2/EP2 · MTP k=3', rc: 'qwen', unit: 'GiB',
          comp: [['lm_head ×4 (1 verify + 3 drafts)', 2.36, 'c', 'lm_head ×4'], ['routed experts', 2.1, 'a', 'experts'], ['GDN projections', 1.95, 'b', 'GDN projections'], ['HyperConnection', 1.19, 'd', 'HyperConnection'], ['attention + indexer', 0.6, 'e', 'attention'], ['MTP ×3', 0.5, 'f', 'MTP ×3'], ['router + shared + PLE', 0.4, 'muted', 'router + PLE'], ['GDN state', 0.1, 'muted', 'GDN state']],
          ncol: 106, bw: 240, acc: 2.18, lat: 25,
        },
        ds: {
          t: 'DeepSeek-V4 · TP2 · DSpark k=6', rc: 'ds', unit: 'GB',
          comp: [['routed experts (≤42/256 × 6.7 MB × 43)', 12, 'a', 'experts (≤42/256)'], ['fp8 attention', 2.4, 'b', 'fp8 attention'], ['drafter (3 layers)', 0.9, 'e', 'drafter'], ['lm_head bf16 (target + draft)', 1.06, 'c', 'lm_head'], ['Markov w2 ×6', 0.2, 'd', 'Markov w2']],
          ncol: 108, bw: 240, acc: 3.15, lat: 40, // report: ~4.15 tokens/step = 1 + Ā
        },
      };
      // bands measured later, on another date/config
      const meas = (p, vocab) => (p === 'qwen'
        ? (vocab
          ? { ms: [51.9, 54.0], msl: 'sweep Sep 16, S=1, vocab 47k: 51.9 (prose) to 54.0 ms (code)', tps: [2.92, 3.83], tpsl: 'sweep.py, with bonus: 2.9 (prose) to 3.8 (code)', tok: [56.5, 71.3], tokl: 'sweep Sep 16, S=1, 47k: 56.5 (prose) to 71.3 (code)' }
          : { ms: [59.3, 60.7], msl: 'sweep Sep 16, S=1, full vocab: 59.3 (prose) to 60.7 ms (code)', tps: [3.0, 3.82], tpsl: 'sweep.py, with bonus: 3.0 (prose) to 3.8 (code)', tok: [50.5, 63.1], tokl: 'sweep S=1: 50.5–63.1 · README 52.1 (MTP=3, KV bf16)' })
        : { ms: [66.7, 71.4], msl: 'engine counters: 14–15 steps/s', tps: [2.5, 5.4], tpsl: '1 + 6 × acceptance: 2.5 (prose) to 5.4 (code)', tok: [37.3, 76.0], tokl: 'code 76.0 tok/s (73.3%) · prose 37.3 (25.7%)' });
      const S = { p: 'qwen', bw: 240, acc: 2.18, lat: 25, vocab: false, nobonus: false };
      const ctl = UI.controls(fig);
      let sB, sA, sL;
      UI.seg(ctl, { label: 'Recipe', options: [{ v: 'qwen', label: 'Qwen' }, { v: 'ds', label: 'DeepSeek' }], value: 'qwen', onChange: (v) => { S.p = v; sB.set(P[v].bw); sA.set(P[v].acc); sL.set(P[v].lat); H.redraw(); } });
      sB = UI.slider(ctl, { label: 'Effective bandwidth', min: 180, max: 273, step: 1, value: S.bw, fmt: (v) => v + ' GB/s', onInput: (v) => { S.bw = v; H && H.redraw(); } });
      sA = UI.slider(ctl, { label: 'Accepted per step Ā', min: 0, max: 6, step: 0.01, value: S.acc, fmt: (v) => br(v, 2), onInput: (v) => { S.acc = v; H && H.redraw(); } });
      sL = UI.slider(ctl, { label: 'Latency per collective', min: 10, max: 60, step: 1, value: S.lat, fmt: (v) => v + ' µs', onInput: (v) => { S.lat = v; H && H.redraw(); } });
      UI.toggle(ctl, { label: 'draft with 47k vocabulary (Qwen)', value: false, onChange: (v) => { S.vocab = v; H && H.redraw(); } });
      UI.toggle(ctl, { label: 'divide without the bonus (the Qwen math)', value: false, onChange: (v) => { S.nobonus = v; H && H.redraw(); } });
      const H = V.fig(fig, (g, w, h) => {
        const C = V.C(), p = P[S.p], narrow = w < 660, pad = 16, M = meas(S.p, S.vocab && S.p === 'qwen');
        const col = rcol(C, p.rc);
        const comps = p.comp.map(([l, v, c, sl]) => {
          let val = v;
          if (S.p === 'qwen' && S.vocab && l.startsWith('lm_head')) val = 0.59 + 3 * 0.59 * (47149 / 248320);
          return { l: narrow ? sl : l, gb: p.unit === 'GiB' ? val * GiB : val, raw: val, c: C[c] || C.muted };
        });
        const totGB = comps.reduce((a, b) => a + b.gb, 0), totRaw = comps.reduce((a, b) => a + b.raw, 0);
        const tMem = (totGB / S.bw) * 1000, tCol = (p.ncol * S.lat) / 1000, tStep = tMem + tCol;
        const tps = 1 + S.acc, tok = (S.nobonus ? S.acc : tps) / (tStep / 1000);
        pill(g, pad, 18, p.t, col, { font: 'sans', size: 12, weight: 600, fa: 0.18 });
        V.text(g, 'bytes read per GPU per step: ' + br(totRaw, 2) + ' ' + p.unit + (p.unit === 'GiB' ? ' (' + br(totGB, 2) + ' GB)' : ''), narrow ? pad : w - pad, narrow ? 46 : 22, { size: narrow ? 11 : 11.5, font: 'mono', color: C.ink, align: narrow ? 'left' : 'right', weight: 600, maxW: w - 2 * pad });
        // stacked bar
        const bx = pad, by = narrow ? 60 : 44, bwid = w - 2 * pad, bh = 34;
        let x = bx;
        comps.forEach((cp) => {
          const ww = (cp.gb / totGB) * bwid;
          V.box(g, x + 0.5, by, Math.max(1, ww - 1), bh, { r: 4, fill: V.alpha(cp.c, 0.55), stroke: cp.c, lw: 1 });
          if (ww > 46) V.text(g, br(cp.raw, 2), x + ww / 2, by + bh / 2 + 4, { size: 11, font: 'mono', weight: 600, align: 'center', color: C.ink });
          x += ww;
        });
        // legend
        const legY = by + bh + 16, cols = narrow ? 2 : comps.length <= 6 ? 3 : 4, lw = bwid / cols;
        comps.forEach((cp, i) => {
          const lx = bx + (i % cols) * lw, ly = legY + Math.floor(i / cols) * 17;
          V.box(g, lx, ly - 9, 10, 10, { r: 2, fill: V.alpha(cp.c, 0.7) });
          V.text(g, cp.l, lx + 15, ly, { size: 11, color: C.muted, maxW: lw - 20 });
        });
        // the two calculations
        const yEq = legY + Math.ceil(comps.length / cols) * 17 + 18;
        const eq1 = 't_step ≥ ' + br(totGB, 2) + ' GB ÷ ' + S.bw + ' GB/s + ' + p.ncol + ' × ' + S.lat + ' µs = ' + br(tMem, 1) + ' + ' + br(tCol, 1) + ' = ' + br(tStep, 1) + ' ms';
        const eq2 = S.nobonus
          ? 'tok/s = Ā ÷ t_step = ' + br(S.acc, 2) + ' ÷ ' + br(tStep / 1000, 4) + ' s = ' + br(tok, 1) + '  ✗ without the bonus token'
          : 'tok/s ≤ (1 + Ā) ÷ t_step = ' + br(tps, 2) + ' ÷ ' + br(tStep / 1000, 4) + ' s = ' + br(tok, 1);
        const fsz = narrow ? 11.5 : 12.5, flh = 16, ew = bwid - 20;
        const n1 = lines(g, eq1, ew, fsz, 'mono', 600).length, n2 = lines(g, eq2, ew, fsz, 'mono', 600).length;
        const eqH = (n1 + n2) * flh + 12;
        V.box(g, pad, yEq - 16, bwid, eqH, { r: 8, fill: V.alpha(C.b, 0.08), stroke: V.alpha(C.b, 0.3), lw: 1 });
        wtext(g, eq1, pad + 10, yEq + 1, ew, { size: fsz, font: 'mono', color: C.ink, weight: 600, lh: flh });
        wtext(g, eq2, pad + 10, yEq + 1 + n1 * flh, ew, { size: fsz, font: 'mono', color: S.nobonus ? C.red : C.ink, weight: 600, lh: flh });
        // comparison axes
        const axes = [
          { l: 'ms/step', v: tStep, band: M.ms, max: 100, fmt: (v) => br(v, 1) + ' ms', note: M.msl },
          { l: 'tokens/step', v: tps, band: M.tps, max: 7, fmt: (v) => br(v, 2) + ' (1 + Ā)', note: M.tpsl },
          { l: 'predicted tok/s', v: tok, band: M.tok, max: 120, fmt: (v) => br(v, 1), note: M.tokl },
        ];
        const ay0 = yEq - 16 + eqH + (narrow ? 34 : 28), ah = narrow ? 58 : 50;
        const noteW = narrow ? 0 : 200;
        const ax0 = pad + (narrow ? 0 : 118), aw = bwid - (narrow ? 0 : 118) - noteW;
        axes.forEach((a, i) => {
          const y = ay0 + i * ah;
          if (narrow) V.text(g, a.l, pad, y - 6, { size: 11.5, weight: 600, color: C.ink });
          else V.text(g, a.l, pad, y + 5, { size: 12, weight: 600, color: C.ink });
          const X = (v) => ax0 + V.clamp(v / a.max, 0, 1) * aw;
          V.line(g, ax0, y, ax0 + aw, y, { color: V.alpha(C.muted, 0.5), lw: 2 });
          [0, 0.25, 0.5, 0.75, 1].forEach((f) => { V.line(g, ax0 + f * aw, y - 3, ax0 + f * aw, y + 3, { color: C.muted, lw: 1 }); });
          V.text(g, '0', ax0, y + 16, { size: 11, font: 'mono', color: C.muted, align: 'center' });
          V.text(g, String(a.max), ax0 + aw, y + 16, { size: 11, font: 'mono', color: C.muted, align: 'center' });
          const b0 = X(a.band[0]), b1 = X(a.band[1]);
          g.fillStyle = V.alpha(C.d, 0.3); V.rr(g, b0 - 3, y - 7, Math.max(6, b1 - b0 + 6), 14, 6); g.fill();
          g.strokeStyle = C.d; g.lineWidth = 1.2; g.stroke();
          const vx = X(a.v), dcol = i === 2 && S.nobonus ? C.red : col;
          V.dot(g, vx, y, 6, dcol); g.strokeStyle = C.bg; g.lineWidth = 2; g.beginPath(); g.arc(vx, y, 6, 0, Math.PI * 2); g.stroke();
          const lab = a.fmt(a.v), lw0 = tw(g, lab, 11.5, 'mono', 700);
          V.text(g, lab, V.clamp(vx, ax0 + lw0 / 2, ax0 + aw - lw0 / 2), y - 12, { size: 11.5, font: 'mono', weight: 700, align: 'center', color: dcol });
          if (!narrow) wtext(g, a.note, ax0 + aw + 14, y - 3, noteW - 14, { size: 11, color: C.d, lh: 13, maxLines: 3 });
          else V.text(g, a.note, pad, y + 30, { size: 11, color: C.d, maxW: bwid });
        });
        const yL = ay0 + 3 * ah + (narrow ? 12 : 2);
        V.dot(g, pad + 6, yL, 5, col); V.text(g, 'predicted', pad + 16, yL + 4, { size: 11, color: C.muted });
        g.fillStyle = V.alpha(C.d, 0.3); V.rr(g, pad + 84, yL - 6, 22, 12, 5); g.fill();
        V.text(g, 'measured later (other date/config)', pad + 112, yL + 4, { size: 11, color: C.muted, maxW: bwid - 120 });
      }, { static: true });
    })();

    /* ============ Figure 9: where the step went ============ */
    (function () {
      const fig = root.querySelector('#fig-trace');
      fit(fig, 380, 420, 620);
      const S = { mode: 'med', sync: false, pin: null };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Mode', options: [{ v: 'med', label: 'measured (GLM, Aug 29)' }, { v: 'obs', label: 'observer effect' }], value: 'med', onChange: (v) => { S.mode = v; } });
      UI.toggle(ctl, { label: 'synchronous stopwatches (*_TIMING=1)', value: false, onChange: (v) => { S.sync = v; } });
      const TOP = [['moe_forward_shared', 623, 'a'], ['aten::mm', 102, 'b'], ['NCCL', 71, 'c'], ['KDA/GDN', 63, 'e'], ['sparse MLA', 43, 'd'], ['residual', 184, 'muted']];
      const MOE = [['exl3_moe', 134, 'f'], ['reconstruct_kernel', 64, 'd'], ['index_add_', 52, 'c'], ['rest of MoE', 373, 'a']];
      const H = V.fig(fig, (g, w, h, t, dt, _S, HH) => {
        const C = V.C(), narrow = w < 620, pad = 16, ptr = HH.pointer;
        if (S.mode === 'med') {
          const total = 1086;
          wtext(g, 'PREFILL STEP · 1,024 tokens at 100K context · 1,086 ms (before E2/E3)', pad, 22, w - 2 * pad, { size: 11, font: 'mono', color: C.muted, weight: 600, lh: 14, maxLines: 2 });
          const x0 = pad, bw = w - 2 * pad, y1 = narrow ? 64 : 56, bh = narrow ? 50 : 56;
          let hover = null, x = x0;
          const segs = [];
          TOP.forEach(([l, ms, c]) => { const ww = (ms / total) * bw; segs.push({ l, ms, c, x, ww, y: y1, h: bh }); x += ww; });
          const moe = segs[0];
          // tap/click pins the block (there is no hover on phones)
          const inSeg = (q, s0) => q.x >= s0.x && q.x < s0.x + s0.ww && q.y >= s0.y && q.y < s0.y + s0.h;
          const y2 = y1 + bh + 34, bh2 = narrow ? 40 : 44;
          let x2 = moe.x; const sub = [];
          MOE.forEach(([l, ms, c]) => { const ww = (ms / 623) * moe.ww; sub.push({ l, ms, c, x: x2, ww, y: y2, h: bh2 }); x2 += ww; });
          // trace-viewer-style playhead
          const ph = x0 + ((t * 0.12) % 1) * bw;
          if (S.tap) { const hs = segs.concat(sub).find((s0) => inSeg(S.tap, s0)); S.pin = hs ? hs.l : null; S.tap = null; }
          segs.concat(sub).forEach((s) => {
            const col = C[s.c] || C.muted, inside = (ptr.inside && ptr.x >= s.x && ptr.x < s.x + s.ww && ptr.y >= s.y && ptr.y < s.y + s.h) || (!ptr.inside && S.pin === s.l);
            if (inside) hover = s;
            V.box(g, s.x + 0.5, s.y, Math.max(1, s.ww - 1), s.h, { r: 4, fill: V.alpha(col, inside ? 0.8 : 0.5), stroke: col, lw: inside ? 2 : 1 });
            const lab = s.l, lwid = tw(g, lab, 11, 'mono', 600);
            if (s.ww > lwid + 10) { V.text(g, lab, s.x + s.ww / 2, s.y + s.h / 2 - 1, { size: 11, font: 'mono', weight: 600, align: 'center', color: C.ink }); V.text(g, s.ms + ' ms', s.x + s.ww / 2, s.y + s.h / 2 + 14, { size: 11, font: 'mono', align: 'center', color: C.ink }); }
            else if (s.ww > 28) V.text(g, String(s.ms), s.x + s.ww / 2, s.y + s.h / 2 + 4, { size: 11, font: 'mono', align: 'center', color: C.ink });
          });
          // MoE to detail connector
          g.strokeStyle = V.alpha(C.a, 0.5); g.lineWidth = 1; g.setLineDash([3, 3]); g.beginPath();
          g.moveTo(moe.x, y1 + bh); g.lineTo(moe.x, y2); g.moveTo(moe.x + moe.ww, y1 + bh); g.lineTo(moe.x + moe.ww, y2); g.stroke(); g.setLineDash([]);
          V.text(g, narrow ? 'inside moe_forward_shared (623 ms ≈ 57%)' : 'inside moe_forward_shared (623 ms ≈ 57% of the step)', moe.x, y2 - 8, { size: 11, color: C.a, weight: 600, maxW: w - moe.x - pad });
          V.line(g, ph, y1 - 6, ph, y1 + bh + 6, { color: V.alpha(C.ink, 0.5), lw: 1.2 });
          // compact legend (narrow) and tooltip
          const yl = y2 + bh2 + 26, lc = narrow ? 2 : 4, lcw = (w - 2 * pad) / lc;
          const legItems = TOP.concat(MOE.slice(0, 3));
          legItems.forEach(([l, ms, c], i) => { const lx = pad + (i % lc) * lcw, ly = yl + Math.floor(i / lc) * 17; V.box(g, lx, ly - 9, 10, 10, { r: 2, fill: V.alpha(C[c] || C.muted, 0.7) }); V.text(g, (l === 'residual' ? 'residual (derived)' : l) + ' ' + ms + ' ms', lx + 15, ly, { size: 11, color: C.muted, maxW: lcw - 20 }); });
          const yr = yl + Math.ceil(legItems.length / lc) * 17 + 12;
          {
            wtext(g, 'Reading: the MoE is ~57% of the 1.086 s step (63% of the total CUDA time of 2.96 s across the 3 chunks); sparse attention, only ~4%. In another trace (16K, 3 steps), the CPU spent 1.45 s in aten::nonzero, the .nonzero().tolist() that synchronizes host and device at every layer. Hence the E2/E3 kernels: grouped MoE without synchronization.', pad, yr, w - 2 * pad, { size: narrow ? 11.5 : 12, color: C.ink, lh: narrow ? 16 : 17, maxLines: narrow ? 10 : 4 });
          }
          if (hover) {
            const s = hover, txt = s.l === 'residual' ? 'derived residual: 1,086 − sum = 184 ms' : s.l + ' · ' + s.ms + ' ms · ' + br((s.ms / total) * 100, 1) + '% of the step';
            const qx = ptr.inside ? ptr.x : s.x + s.ww / 2, qy = ptr.inside ? ptr.y : s.y + 6;
            const tww = Math.min(w - 2 * pad, tw(g, txt, 12, 'mono', 600) + 18), tx = V.clamp(qx - tww / 2, pad, w - pad - tww), ty = Math.max(8, qy - 38);
            V.box(g, tx, ty, tww, 24, { r: 8, fill: C.panel, stroke: V.alpha(C.ink, 0.3), lw: 1 });
            V.text(g, txt, tx + tww / 2, ty + 16, { size: 12, font: 'mono', weight: 600, align: 'center', color: C.ink, maxW: tww - 10 });
          }
          if (S.sync) V.text(g, '(the stopwatch toggle acts in “observer effect” mode)', w - pad, h - 10, { size: 11, color: C.muted, align: 'right' });
          return;
        }
        // observer mode (illustrative)
        const stages = [['attn', 9, 'b'], ['moe', 18, 'a'], ['draft', 7, 'e'], ['sample', 3, 'd']];
        const prep = 6, syncCost = 2.2, nSteps = 3;
        wtext(g, 'ILLUSTRATIVE · 3 decode steps · CPU enqueues, GPU executes', pad, 22, w - 2 * pad, { size: 11, font: 'mono', color: C.muted, weight: 600, lh: 14, maxLines: 2 });
        const gpuDur = stages.reduce((a, s) => a + s[1], 0);
        const evs = { cpu: [], gpu: [] };
        let tc = 0, tg = 0;
        for (let k = 0; k < nSteps; k++) {
          evs.cpu.push({ a: tc, b: tc + prep, l: 'prep ' + (k + 1), c: 'muted' });
          const start = Math.max(tc + prep, tg);
          tc += prep;
          let tt2 = start;
          stages.forEach(([l, d, c]) => {
            evs.gpu.push({ a: tt2, b: tt2 + d, l, c });
            tt2 += d;
            if (S.sync) { evs.cpu.push({ a: Math.max(tc, tt2 - 0.01), b: tt2 + syncCost, l: 'sync', c: 'c' }); tc = tt2 + syncCost; tt2 += syncCost; }
          });
          tg = tt2;
          if (S.sync) tc = Math.max(tc, tg);
        }
        const tEnd = Math.max(tg, tc), scaleMax = S.sync ? tEnd : Math.max(tEnd, 1);
        const lx = pad + (narrow ? 0 : 60), lw = w - lx - pad;
        const X = (v) => lx + (v / (nSteps * (gpuDur + prep + stages.length * syncCost) + 2)) * lw;
        const yc = narrow ? 70 : 64, yg = yc + (narrow ? 70 : 64), hh2 = 34;
        V.text(g, 'CPU', narrow ? pad : pad, narrow ? yc - 8 : yc + 22, { size: 11.5, font: 'mono', weight: 600, color: C.muted });
        V.text(g, 'GPU', narrow ? pad : pad, narrow ? yg - 8 : yg + 22, { size: 11.5, font: 'mono', weight: 600, color: C.muted });
        const prog = (t * 22) % (scaleMax + 24);
        ['cpu', 'gpu'].forEach((lane) => {
          const y = lane === 'cpu' ? yc : yg;
          V.box(g, lx, y, lw, hh2, { r: 6, fill: V.alpha(C.muted, 0.05), stroke: V.alpha(C.muted, 0.2), lw: 1 });
          evs[lane].forEach((e) => {
            if (e.a > prog) return;
            const b = Math.min(e.b, prog), col = C[e.c] || C.muted;
            const xa = X(e.a), xb = X(b);
            V.box(g, xa + 0.5, y + 3, Math.max(1, xb - xa - 1), hh2 - 6, { r: 4, fill: V.alpha(col, e.l === 'sync' ? 0.6 : 0.45), stroke: col, lw: 1 });
            if (xb - xa > tw(g, e.l, 11) + 6) V.text(g, e.l, (xa + xb) / 2, y + hh2 / 2 + 4, { size: 11, align: 'center', color: C.ink });
          });
        });
        // GPU bubbles
        if (S.sync) {
          for (let i = 1; i < evs.gpu.length; i++) { const gap0 = evs.gpu[i - 1].b, gap1 = evs.gpu[i].a; if (gap1 - gap0 > 0.3 && gap1 <= prog) hatch(g, X(gap0), yg + 3, X(gap1) - X(gap0), hh2 - 6, V.alpha(C.c, 0.6), 5); }
        }
        const stepMs = tEnd / nSteps;
        const yT = yg + hh2 + 36;
        V.text(g, 'time/step: ' + br(stepMs, 1) + ' ms' + (S.sync ? ' (stopwatches on)' : ' (real engine)'), pad, yT, { size: 14, font: 'mono', weight: 700, color: S.sync ? C.c : C.a });
        wtext(g, S.sync ? 'Each synchronize() waits for the GPU to drain: the CPU can\'t queue the next step, the GPU idles (hatched) and you time another engine. Hence *_TIMING defaults to 0 and stays out of benchmarks.' : 'Without stopwatches, the CPU prepares the next step while the GPU is still computing (async scheduling + CUDA graphs): the GPU runs one kernel right after another.', pad, yT + 22, w - 2 * pad, { size: 12, color: C.ink, lh: 17, maxLines: 4 });
      }, { static: false });
      // tap pins the block: store the pointerdown position (on touch, pointerleave clears the pointer right after)
      H.canvas.addEventListener('pointerdown', (e) => { const r = H.canvas.getBoundingClientRect(); S.tap = { x: e.clientX - r.left, y: e.clientY - r.top }; });
    })();

    /* ============ Figure 11 (TensorFold): same prompt, eight trials ============ */
    (function () {
      const fig = root.querySelector('#fig-tf-ruido');
      fit(fig, 375, 348, 600);
      const NT = 8, NTOK = 160, K = 7, RMS = 66.0 + 3.08; // V(8) + B from the patch 0011 table (ms)
      const S = { samp: 'key', clk: 'lock', n: 2, seed: 7, t0: performance.now() };
      let M = null;
      function gen() {
        const R = V.rng(S.seed);
        // a reply = a sequence of rounds; each round emits (accepted + 1) tokens
        const mk = (p, RR) => { const out = []; let left = NTOK; while (left > 0) { let a = 0; while (a < K && RR() < p) a++; const tok = Math.min(left, a + 1); out.push(tok); left -= tok; } return out; };
        const keyed = mk(0.62, V.rng(S.seed * 7 + 3));
        const free = [];
        for (let i = 0; i < NT; i++) free.push(mk(V.clamp(0.62 + 0.06 * R.normal(), 0.46, 0.78), V.rng(S.seed * 131 + i * 17 + 1)));
        const CK = { free: [0.03, 0.02], lock: [0.004, 0.004] }; // [per trial, per round] (illustrative)
        const reg = {};
        let Tmax = 0, lo = Infinity, hi = 0;
        ['free', 'key'].forEach((sa) => ['free', 'lock'].forEach((ck) => {
          const RT = V.rng(S.seed * 977 + (sa === 'key' ? 11 : 29) + (ck === 'lock' ? 5 : 0));
          const trials = [];
          for (let i = 0; i < NT; i++) {
            const toks = sa === 'key' ? keyed : free[i];
            const m = 1 + CK[ck][0] * RT.normal();
            let t = 0;
            const segs = [];
            toks.forEach((tok) => { const d = RMS * m * (1 + CK[ck][1] * RT.normal()); segs.push([t, t + d, tok]); t += d; });
            const tps = (1000 * NTOK) / t;
            trials.push({ segs, T: t, tps });
            Tmax = Math.max(Tmax, t); lo = Math.min(lo, tps); hi = Math.max(hi, tps);
          }
          const xs = trials.map((x) => x.tps), mu = mean(xs);
          const sd = Math.sqrt(xs.reduce((p, q) => p + (q - mu) * (q - mu), 0) / (xs.length - 1));
          reg[sa + '|' + ck] = { trials, med: median(xs), sd, mn: Math.min(...xs), mx: Math.max(...xs) };
        }));
        M = { reg, Tmax: Tmax * 1.01, lo, hi };
        S.t0 = performance.now();
      }
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Sampling', options: [{ v: 'free', label: 'free (T=0.6)' }, { v: 'key', label: 'keyed' }], value: S.samp, onChange: (v) => { S.samp = v; S.t0 = performance.now(); upd(); } });
      UI.seg(ctl, { label: 'GPU clock', options: [{ v: 'free', label: 'free' }, { v: 'lock', label: 'capped (2,200 MHz)' }], value: S.clk, onChange: (v) => { S.clk = v; S.t0 = performance.now(); upd(); } });
      UI.slider(ctl, { label: 'boots per arm (n)', min: 1, max: 12, step: 1, value: S.n, onInput: (v) => { S.n = v; upd(); } });
      UI.button(ctl, '🎲 new trials', () => { S.seed = (S.seed * 48271 + 11) % 2147483647; gen(); upd(); });
      const RO = { med: UI.readout(ctl, 'median', '—'), sd: UI.readout(ctl, 'spread between trials', '—'), res: UI.readout(ctl, 'smallest visible effect', '—') };
      const resOf = (r) => (2 * Math.SQRT2 * 1.25 * r.sd) / Math.sqrt(S.n) / r.med;
      const pct = (x) => br(100 * x, x < 0.1 ? 1 : 0) + '%';
      function upd() {
        const r = M.reg[S.samp + '|' + S.clk], res = resOf(r);
        RO.med.set(br(r.med, 1) + ' tok/s');
        RO.sd.set(br(r.sd, 2) + ' tok/s (' + pct(r.sd / r.med) + ')');
        RO.res.set('±' + pct(res), res <= 0.012 ? 'ok' : res <= 0.05 ? 'warn' : 'bad');
      }
      gen(); upd();
      V.fig(fig, (g, w, h) => {
        const C = V.C(), TF = TFC(C), narrow = w < 600, pad = 16;
        const r = M.reg[S.samp + '|' + S.clk], key = S.samp === 'key', lock = S.clk === 'lock';
        const prog = V.clamp((performance.now() - S.t0) / 3200, 0, 1), tNow = V.ease.inOut(prog) * M.Tmax;
        const h1 = key ? 'fixed content: same reply, same rounds' : 'content varies: each trial generates another reply';
        const h2 = lock ? 'timing almost fixed: capped clock' : 'timing varies: boost and throttling';
        if (narrow) {
          pill(g, pad, 14, h1, key ? TF : C.c, { font: 'sans', size: 11, weight: 600, fa: 0.16 });
          pill(g, pad, 38, h2, lock ? TF : C.c, { font: 'sans', size: 11, weight: 600, fa: 0.16 });
        } else {
          const pw = pill(g, pad, 16, h1, key ? TF : C.c, { font: 'sans', size: 11.5, weight: 600, fa: 0.16 });
          pill(g, pad + pw + 10, 16, h2, lock ? TF : C.c, { font: 'sans', size: 11.5, weight: 600, fa: 0.16 });
        }
        const labW = narrow ? 26 : 56, tpsW = narrow ? 40 : 70;
        const x0 = pad + labW, x1 = w - pad - tpsW, rh = narrow ? 20 : 26, y0 = narrow ? 58 : 38, bh = rh - 6;
        const X = (t) => x0 + (t / M.Tmax) * (x1 - x0);
        r.trials.forEach((tr, i) => {
          const y = y0 + i * rh;
          V.text(g, narrow ? '#' + (i + 1) : 'trial ' + (i + 1), pad, y + bh / 2 + 4, { size: 11, font: 'mono', color: C.muted });
          V.box(g, x0, y, x1 - x0, bh, { r: 3, fill: V.alpha(C.muted, 0.07) });
          for (const [a, b, tok] of tr.segs) {
            if (a >= tNow) break;
            const xa = X(a), xb = X(Math.min(b, tNow)), ww = xb - xa;
            g.fillStyle = V.alpha(C.a, 0.1 + 0.095 * tok);
            g.fillRect(xa, y, Math.max(0.8, ww > 3 ? ww - 1 : ww), bh);
            if (!narrow && b <= tNow && X(b) - xa > 14) V.text(g, String(tok), (xa + X(b)) / 2, y + bh / 2 + 4, { size: 11, font: 'mono', align: 'center', color: C.ink, alpha: 0.8 });
          }
          if (tNow >= tr.T) {
            const xe = X(tr.T);
            V.line(g, xe, y - 2, xe, y + bh + 2, { color: key ? TF : C.c, lw: 2.2 });
            V.text(g, br(tr.tps, 1), w - pad, y + bh / 2 + 4, { size: 11, font: 'mono', weight: 600, align: 'right', color: C.ink });
          }
        });
        if (prog < 1) { const xp = X(tNow); V.line(g, xp, y0 - 4, xp, y0 + NT * rh - 2, { color: V.alpha(C.ink, 0.5), lw: 1, dash: [3, 3] }); }
        if (!narrow) V.text(g, 'tok/s', w - pad, y0 - 6, { size: 11, font: 'mono', color: C.muted, align: 'right' });
        // ---- tok/s distribution
        const axY = y0 + NT * rh + (narrow ? 50 : 54);
        const st = nice((M.hi - M.lo) / (narrow ? 3 : 7)), lo = Math.floor(M.lo / st) * st, hi = Math.ceil(M.hi / st) * st;
        const P = (v) => x0 + ((v - lo) / (hi - lo)) * (x1 - x0);
        V.line(g, x0, axY, x1, axY, { color: V.alpha(C.muted, 0.6), lw: 1 });
        for (let v = lo; v <= hi + 1e-9; v += st) { const x = P(v); V.line(g, x, axY, x, axY + 4, { color: C.muted, lw: 1 }); V.text(g, br(v, 0), x, axY + 16, { size: 11, font: 'mono', color: C.muted, align: 'center' }); }
        V.text(g, 'tok/s', x1 + 8, axY + 4, { size: 11, font: 'mono', color: C.muted });
        if (prog >= 1) {
          g.fillStyle = V.alpha(C.muted, 0.16); g.fillRect(P(r.mn), axY - 24, Math.max(2, P(r.mx) - P(r.mn)), 20);
          r.trials.forEach((tr, i) => V.dot(g, P(tr.tps), axY - 14 + ((i % 3) - 1) * 4, 4, key ? TF : C.c));
          V.line(g, P(r.med), axY - 28, P(r.med), axY, { color: C.ink, lw: 2 });
          const res = resOf(r) * r.med, xa = V.clamp(P(r.med - res), x0, x1), xb = V.clamp(P(r.med + res), x0, x1), yb = axY - 34;
          V.line(g, xa, yb, xb, yb, { color: C.b, lw: 2 });
          V.line(g, xa, yb - 4, xa, yb + 4, { color: C.b, lw: 2 }); V.line(g, xb, yb - 4, xb, yb + 4, { color: C.b, lw: 2 });
          const lab = 'smallest visible effect with n = ' + S.n + ': ±' + pct(resOf(r));
          const lw0 = tw(g, lab, 11, 'sans', 600), lx = V.clamp((xa + xb) / 2, x0 + lw0 / 2, x1 - lw0 / 2);
          V.text(g, lab, lx, yb - 7, { size: 11, weight: 600, color: C.b, align: 'center' });
        }
        const foot = key
          ? (lock ? 'Only timing is left, and the clock cap shrinks it: with a few boots, 1–2% effects already rise above the noise.' : 'Without content noise, the remaining spread is the clock: capping it is the next step.')
          : 'The spread comes mostly from content: capping the clock barely helps, and you need many more trials.';
        wtext(g, foot, pad, axY + 36, w - 2 * pad, { size: 11.5, color: C.d, weight: 500, lh: 15, maxLines: 3 });
      });
    })();

    /* ============ Figure 12 (TensorFold): simulate before measuring ============ */
    (function () {
      const fig = root.querySelector('#fig-tf-sim');
      fit(fig, 380, 412, 600);
      const VMS = [29.2, 35.6, 42.6, 48.8, 53.6, 58.9, 64.5, 66.0], BMS = 3.08, MOST = 7, L = 240, RD = 0.5;
      const sig = (x) => 1 / (1 + Math.exp(-x)), lgt = (p) => Math.log(p / (1 - p));
      const erf = (x) => { const t = 1 / (1 + 0.3275911 * Math.abs(x)); const y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x); return x >= 0 ? y : -y; };
      const Phi = (x) => 0.5 * (1 + erf(x / Math.SQRT2));
      const POL = ['f1', 'f3', 'f7', 'fc', 'fnc', 'or'];
      const S = { kind: 'prose', pol: 'fnc', p: 0.3, seed: 1, t0: performance.now() };
      let D = null, RES = null;
      const pstr = () => String(Math.round(S.p * 100) / 100);
      const plab = (k) => (k === 'fc' ? 'fc7:' + pstr() : k === 'fnc' ? 'fnc7:' + pstr() : k === 'or' ? 'oracle7' : k);
      // synthetic dump: for each position i, the chain of 7 drafts the block would have proposed,
      // whether each one matches the committed token (m), the noise-free confidence (c) and the noise-aware one (n)
      function mkDump() {
        const R = V.rng(S.seed * 101 + (S.kind === 'code' ? 7 : 3));
        const q = [];
        let z = 0;
        for (let t = 0; t < L + MOST + 1; t++) {
          if (S.kind === 'prose') { z = 0.8 * z + 0.6 * R.normal(); q.push(V.clamp(0.62 + 0.16 * z, 0.08, 0.97)); }
          else { z = 0.85 * z + 0.5 * R.normal(); const run = Math.floor(t / 22) % 3 === 1; q.push(V.clamp((run ? 0.95 : 0.7) + 0.1 * z, 0.15, 0.99)); }
        }
        const m = [], c = [], n = [];
        for (let i = 0; i < L; i++) {
          const mi = [], ci = [], ni = [];
          for (let j = 1; j <= MOST; j++) {
            const pi = V.clamp(q[i + j] * Math.pow(0.95, j - 1), 0.01, 0.995), u = R();
            mi.push(u < pi ? 1 : 0);
            ci.push(sig(lgt(pi) + 0.7 + 0.7 * R.normal()));
            const zh = u + 0.34 * R.normal();
            ni.push(V.clamp(Phi((pi - zh) / 0.34), 0.005, 0.995));
          }
          m.push(mi); c.push(ci); n.push(ni);
        }
        D = { m, c, n };
      }
      const acc = (i, k) => { let a = 0; while (a < k && D.m[i][a]) a++; return a; };
      function depth(i, pol) {
        if (pol === 'f1') return 1;
        if (pol === 'f3') return 3;
        if (pol === 'f7') return 7;
        const cf = pol === 'fc' ? D.c[i] : D.n[i];
        let ch = 1, k = 0;
        for (let j = 0; j < MOST; j++) { ch *= cf[j]; if (j > 0 && ch < S.p) break; k = j + 1; }
        return k;
      }
      function run(pol) {
        const tr = [];
        let i = 0, ms = 0;
        if (pol === 'or') {
          const best = new Array(L + 1).fill(Infinity), ch = new Array(L + 1).fill(0);
          best[L] = 0;
          for (let s = L - 1; s >= 0; s--) {
            const room = Math.min(MOST, L - s - 1);
            for (let k = 0; k <= room; k++) { const a = acc(s, k), t = VMS[k] + (k > 0 ? BMS : 0) + best[Math.min(L, s + a + 1)]; if (t < best[s]) { best[s] = t; ch[s] = k; } }
          }
          while (i < L) { const k = ch[i], a = acc(i, k), d = VMS[k] + (k > 0 ? BMS : 0); tr.push({ i, k, a, ms: d, t0: ms }); ms += d; i += a + 1; }
        } else {
          while (i < L) { const room = Math.min(MOST, L - i - 1), k = Math.min(depth(i, pol), room), a = acc(i, k), d = VMS[k] + (room > 0 ? BMS : 0); tr.push({ i, k, a, ms: d, t0: ms }); ms += d; i += a + 1; }
        }
        const tok = Math.min(i, L), own = new Int16Array(L + MOST + 2).fill(-1), bon = new Uint8Array(L + MOST + 2);
        tr.forEach((x, r) => { for (let j = x.i + 1; j <= x.i + x.a + 1; j++) own[j] = r; bon[x.i + x.a + 1] = 1; });
        return { tr, ms, tok, tps: (1000 * tok) / ms, own, bon };
      }
      function ll(key) {
        let s = 0, nn = 0;
        for (let i = 0; i < L; i++) for (let j = 0; j < MOST; j++) { const p = V.clamp(D[key][i][j], 1e-3, 1 - 1e-3), y = D.m[i][j]; s -= y * Math.log(p) + (1 - y) * Math.log(1 - p); nn++; }
        return s / nn;
      }
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Dump', options: [{ v: 'prose', label: 'sampled prose' }, { v: 'code', label: 'code' }], value: S.kind, onChange: (v) => { S.kind = v; mkDump(); compute(); } });
      UI.seg(ctl, { label: 'Rule', options: [{ v: 'f1', label: 'f1' }, { v: 'f3', label: 'f3' }, { v: 'f7', label: 'f7' }, { v: 'fc', label: 'fc7' }, { v: 'fnc', label: 'fnc7' }, { v: 'or', label: 'oracle' }], value: S.pol, onChange: (v) => { S.pol = v; gp.show(v === 'fc' || v === 'fnc'); S.t0 = performance.now(); upd(); } });
      const gp = UI.group(ctl);
      UI.slider(gp.el, { label: 'threshold p', min: 0.05, max: 0.9, step: 0.05, value: S.p, fmt: (v) => br(v, 2), onInput: (v) => { S.p = v; compute(); } });
      UI.button(ctl, '🎲 new dump', () => { S.seed = (S.seed % 97) + 1; mkDump(); compute(); });
      const RO = { r: UI.readout(ctl, 'rounds', '—'), tpr: UI.readout(ctl, 'tokens/round', '—'), ms: UI.readout(ctl, 'ms/round', '—'), tps: UI.readout(ctl, 'tok/s', '—'), or: UI.readout(ctl, 'of the oracle', '—') };
      function compute() { RES = {}; POL.forEach((k) => { RES[k] = run(k); }); RES.llc = ll('c'); RES.lln = ll('n'); S.t0 = performance.now(); upd(); }
      function upd() {
        const x = RES[S.pol], n = x.tr.length, f = x.tps / RES.or.tps;
        RO.r.set(String(n)); RO.tpr.set(br(x.tok / n, 2)); RO.ms.set(br(x.ms / n, 1)); RO.tps.set(br(x.tps, 1));
        RO.or.set(br(100 * f, 0) + '%', f >= 0.75 ? 'ok' : f >= 0.62 ? 'warn' : 'bad');
      }
      mkDump(); compute();
      V.fig(fig, (g, w, h) => {
        const C = V.C(), TF = TFC(C), narrow = w < 600, pad = 16;
        const X = RES[S.pol], tr = X.tr, n = tr.length;
        const el = (performance.now() - S.t0) / 1000 / RD, ri = Math.floor(el) % n, f = el - Math.floor(el), R0 = tr[ri];
        const showConf = S.pol === 'fc' || S.pol === 'fnc', conf = S.pol === 'fc' ? D.c : D.n;
        // header
        const done = R0.i + (f >= 0.8 ? R0.a + 1 : 0);
        const head = 'replay of ' + plab(S.pol) + ' · ' + (S.kind === 'prose' ? 'prose' : 'code') + ' dump';
        const tally = 'round ' + (ri + 1) + '/' + n + ' · ' + done + ' tokens · ' + brk(R0.t0 + (f >= 0.8 ? R0.ms : 0)) + ' ms';
        pill(g, pad, narrow ? 14 : 16, head, TF, { font: 'sans', size: 11.5, weight: 600, fa: 0.16 });
        V.text(g, tally, narrow ? pad : w - pad, narrow ? 40 : 20, { size: 11, font: 'mono', color: C.muted, align: narrow ? 'left' : 'right' });
        // window of positions
        const NW = narrow ? 16 : V.clamp(Math.floor((w - 2 * pad) / 24), 24, 46), cw = (w - 2 * pad) / NW, chH = narrow ? 20 : 22;
        const yC = narrow ? 70 : 44, yD = yC + chH + (narrow ? 22 : 26);
        const posF = R0.i + (f > 0.55 ? V.ease.inOut(V.clamp((f - 0.55) / 0.35, 0, 1)) * (R0.a + 1) : 0);
        const ws = V.clamp(posF - NW * 0.3, 0, L + 1 - NW);
        const PX = (j) => pad + (j - ws) * cw;
        g.save(); g.beginPath(); g.rect(pad - 1, yC - 16, w - 2 * pad + 2, yD + chH + 22 - (yC - 16)); g.clip();
        // row of the serial reply (the dump) and the rounds already committed
        const j0 = Math.max(0, Math.floor(ws) - 1), j1 = Math.min(L, Math.ceil(ws + NW) + 1);
        for (let j = j0; j <= j1; j++) {
          const x = PX(j) + 1.5, cwi = cw - 3, pr = X.own[j];
          const committed = j === 0 || j <= done;
          let fill = V.alpha(C.muted, 0.08), stroke = V.alpha(C.muted, 0.35);
          if (committed && j > 0) { fill = V.alpha(pr % 2 ? C.a : TF, pr % 2 ? 0.32 : 0.4); stroke = V.alpha(pr % 2 ? C.a : TF, 0.7); }
          if (j === 0) fill = V.alpha(C.b, 0.25);
          V.box(g, x, yD, cwi, chH, { r: 4, fill, stroke, lw: 1 });
          if (committed && j > 0 && X.bon[j]) { g.strokeStyle = C.d; g.lineWidth = 2; V.rr(g, x - 1, yD - 1, cwi + 2, chH + 2, 5); g.stroke(); }
          if (j % 5 === 0 && x + cwi / 2 > pad + 8 && x + cwi / 2 < w - pad - 8) V.text(g, String(j), x + cwi / 2, yD + chH + 14, { size: 11, font: 'mono', color: C.muted, align: 'center' });
        }
        // the round's pending token
        { const x = PX(R0.i) + 1.5; g.strokeStyle = C.ink; g.lineWidth = 1.5; g.setLineDash([3, 2]); V.rr(g, x - 2, yD - 2, cw + 1, chH + 4, 5); g.stroke(); g.setLineDash([]); }
        // chain proposed in the round
        for (let j = 1; j <= R0.k; j++) {
          const appear = 0.05 + (0.45 * (j - 1)) / Math.max(1, R0.k);
          if (f < appear) break;
          const x = PX(R0.i + j) + 1.5, cwi = cw - 3, ok = j <= R0.a, miss = j === R0.a + 1;
          let yy = yC;
          if (ok && f > 0.55) yy = V.lerp(yC, yD, V.ease.inOut(V.clamp((f - 0.55) / 0.25, 0, 1)));
          const col = ok ? C.a : miss ? C.red || C.c : C.muted;
          const a0 = !ok && !miss ? 0.35 : 1;
          g.globalAlpha = a0 * (ok && f > 0.8 ? 0 : 1);
          V.box(g, x, yy, cwi, chH, { r: 4, fill: V.alpha(col, ok ? 0.35 : 0.22), stroke: col, lw: 1.4 });
          if (showConf) { const cv = conf[R0.i][j - 1]; g.fillStyle = V.alpha(C.ink, 0.55); g.fillRect(x + 3, yy + chH - 4, Math.max(1, (cwi - 6) * cv), 2.5); }
          if (cwi > 13) V.text(g, ok ? '✓' : miss ? '✗' : '·', x + cwi / 2, yy + chH / 2 + 4, { size: 11, weight: 700, align: 'center', color: C.ink });
          g.globalAlpha = 1;
        }
        if (f > 0.5) {
          let why;
          if (showConf) {
            let pr = 1;
            for (let j = 0; j < R0.k; j++) pr *= conf[R0.i][j];
            const nx = R0.k < MOST ? pr * conf[R0.i][R0.k] : null;
            why = '∏ ' + br(pr, 2) + (nx != null && R0.k > 0 ? ' · +1 → ' + br(nx, 2) + ' < ' + br(S.p, 2) : '');
          } else why = S.pol === 'or' ? 'knows how many match' : 'fixed: k = ' + S.pol.slice(1);
          const wx = PX(R0.i + Math.max(1, R0.k) + 1) + 4;
          if (narrow || wx + tw(g, why, 11, 'mono') > w - pad) V.text(g, why, w - pad, yC - 5, { size: 11, font: 'mono', color: C.muted, align: 'right' });
          else V.text(g, why, wx, yC + chH / 2 + 4, { size: 11, font: 'mono', color: C.muted });
        }
        g.restore();
        V.text(g, 'chain', pad, yC - 5, { size: 11, color: C.muted });
        V.text(g, 'serial reply (dump)', pad, yD - 5, { size: 11, color: C.muted });
        // the round's explanation
        const tt = 'k = ' + R0.k + ' draft' + (R0.k === 1 ? '' : 's') + (narrow ? '' : ' (window of ' + (R0.k + 1) + ' rows)') + ' · accepted ' + R0.a + ' + 1 sampled = ' + (R0.a + 1) + ' token' + (R0.a ? 's' : '') + ' · V(' + (R0.k + 1) + ')' + (R0.ms > VMS[R0.k] + 1e-6 ? ' + B' : '') + ' = ' + br(R0.ms, 1) + ' ms';
        const yT = yD + chH + 34;
        const th = wtext(g, tt, pad, yT, w - 2 * pad, { size: 11.5, font: 'mono', color: C.ink, lh: 15, maxLines: 2 });
        // tok/s bars per rule
        const yB = yT + th + (narrow ? 12 : 10), rowH = narrow ? 24 : 22, lab = narrow ? 76 : 96;
        const bx = pad + lab, bw = w - bx - pad - 52, top = RES.or.tps * 1.04;
        V.text(g, 'tok/s on the same dump, computed on the spot', pad, yB, { size: 11, font: 'mono', color: C.muted, weight: 600 });
        POL.forEach((k, idx) => {
          const y = yB + 8 + idx * rowH, v = RES[k].tps, sel = k === S.pol, isOr = k === 'or';
          V.text(g, plab(k), pad, y + 13, { size: 11.5, font: 'mono', weight: sel ? 700 : 500, color: sel ? C.ink : C.muted, maxW: lab - 6 });
          V.box(g, bx, y + 2, bw, 14, { r: 7, fill: V.alpha(C.muted, 0.07) });
          const bl = Math.max(3, (bw * v) / top);
          if (isOr) { g.setLineDash([4, 3]); V.box(g, bx, y + 2, bl, 14, { r: 7, fill: V.alpha(C.b, sel ? 0.4 : 0.14), stroke: C.b, lw: 1.3 }); g.setLineDash([]); }
          else V.box(g, bx, y + 2, bl, 14, { r: 7, fill: V.alpha(sel ? TF : C.a, sel ? 0.75 : 0.32), stroke: sel ? TF : V.alpha(C.a, 0.6), lw: 1 });
          V.text(g, br(v, 1), bx + bl + 6, y + 13, { size: 11.5, font: 'mono', weight: sel ? 700 : 500, color: C.ink });
        });
        const yL = yB + 8 + POL.length * rowH + 14;
        wtext(g, 'log loss of the confidences: noise-free ' + br(RES.llc, 2) + ' · noise-aware ' + br(RES.lln, 2) + ' (lower = better calibrated)', pad, yL, w - 2 * pad, { size: 11.5, color: C.d, weight: 500, lh: 15, maxLines: 2 });
      });
    })();

    /* ============ Figure 10: cautious comparator ============ */
    // (local fix) controls do not shrink at 390 px: allow wrapping and cap the width
    const fixCtl = () => {
      root.querySelectorAll('.la-fig .la-fig-controls .la-ctl').forEach((el) => { el.style.flexWrap = 'wrap'; el.style.maxWidth = '100%'; el.style.minWidth = '0'; });
      root.querySelectorAll('.la-fig .la-fig-controls .la-seg').forEach((el) => { el.style.maxWidth = '100%'; });
      root.querySelectorAll('.la-fig .la-fig-controls select').forEach((el) => { el.style.maxWidth = '100%'; el.style.minWidth = '0'; });
      root.querySelectorAll('.la-fig .la-fig-controls .la-toggle').forEach((el) => { el.style.maxWidth = '100%'; });
    };
    (function () {
      const fig = root.querySelector('#fig-comparador');
      fit(fig, 560, 724, 600);
      const M = {
        dec: { u: 'tok/s (decode c=1)', max: 120, fmt: (v) => br(v, 1), note: 'The prompt category moves decode more than almost any knob; compare within a group and with the prompt declared.',
          g: { glm: [[72.03, 'structured', 'stock · Sep 16 · k=7'], [34.09, 'prose (hash map)', 'stock · Sep 16'], [36.1, 'sparkDash prose', 'adaptive-k + coop · Sep 17']],
            tf: [[114.7, 'sparkDash structured', 'v1.3.2 · GPU capped at 2,200 MHz'], [60.4, 'sparkDash prose', 'v1.3.2 · 2,200 MHz cap · q4, KV fp8']],
            qwen: [[76.0, 'structured', 'lane 0.30 · Sep 27'], [59.2, 'prose', 'lane 0.30 · Sep 27'], [66.6, 'code', 'lane 0.30 · Sep 27']],
            ds: [[56, 'numbered words, 256-token prompt', 'README, median of 8 (51–67) · Sep 2'], [38.6, 'prose', 'TP=2, 512 tokens · Sep 2'], [75.4, 'numbered words, natural stop', '0731, server-side thinking · Aug 1']] } },
        pre: { u: 'prefill tok/s (~32K, c=1)', max: 3200, fmt: (v) => brk(v), note: '36 of Qwen\'s 48 layers are linear attention and its full attention is sparse: prefill is another story. “Derived” = 32,768 ÷ published TTFT.',
          g: { glm: [[1428.2, 'E3, 32k (sparkDash)', '900K, util 0.86, MNBT 7168 · Sep 7']],
            tf: [[1978.9, '32,790 tokens (sparkDash)', 'v1.3.2 · 2,200 MHz cap · KV fp8, q4']],
            qwen: [[2962.2, 'day-0, 32k', 'KV fp8, GMU 0.835'], [2982.3, 'lane 0.30, 32k', 'Sep 27']],
            ds: [[1428, '0731, 32K', 'benchmark-0731, c=1 · Aug 1'], [1820, 'Vision-Exp, 32K', 'derived: 32,768 ÷ 18.0 s (TP3.md) · Sep 2']] } },
        kv: { u: 'millions of tokens in the KV pool', max: 4, fmt: (v) => br(v, 2) + ' M', note: 'The pool depends on the KV dtype, the utilization, the max context and how much host memory is left, not only on the model.',
          g: { glm: [[1.234, '850K · 11 GiB fixed', '1.45× · compact pages'], [1.572, '850K · 14 GiB fixed', '1.85× · Sep 26']],
            tf: [[2.922, '1M · KV fp8 (528 B/token/layer)', 'free at start − 14.5 GiB · varies per boot']],
            qwen: [[3.652, '262K · KV fp8', 'GMU 0.835 · 13.93×'], [1.340, 'lane 0.30 · KV BF16', 'GMU 0.80']],
            ds: [[2.331, '1M · nvfp4_ds_mla', 'util 0.835 · 2.22×']] } },
        acc: { u: 'draft acceptance', max: 1, fmt: (v) => br(v * 100, 1) + '%', note: 'Acceptance depends on the drafter (a different k in each recipe) and above all on the prompt. It is the biggest decode lever left in all three recipes.',
          g: { glm: [[0.959, 'structured', 'DFlash2 k=7 · Aug 30'], [0.341, 'prose', 'DFlash2 k=7 · Aug 30']],
            tf: [],
            qwen: [[0.728, 'README (823/1131)', 'MTP k=3']],
            ds: [[0.512, 'numbered words', 'DSpark k=6 · Sep 2'], [0.254, 'natural prose', 'DSpark k=6 · Sep 2']] } },
      };
      const S = { m: 'dec' };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Metric', options: [{ v: 'dec', label: 'decode c=1' }, { v: 'pre', label: 'prefill ~32K' }, { v: 'kv', label: 'KV pool' }, { v: 'acc', label: 'acceptance' }], value: 'dec', onChange: (v) => { S.m = v; S.t0 = performance.now(); } });
      S.t0 = performance.now();
      V.fig(fig, (g, w, h) => {
        const C = V.C(), m = M[S.m], narrow = w < 600, pad = 16, el = (performance.now() - S.t0) / 1000;
        const grow = V.ease.out(V.clamp(el / 0.8, 0, 1));
        V.text(g, m.u.toUpperCase(), pad, 20, { size: 11, font: 'mono', color: C.muted, weight: 600 });
        const labW = narrow ? 0 : Math.min(250, w * 0.34), bx = pad + labW + (narrow ? 0 : 10), bw = w - bx - pad - 64;
        let y = 40;
        const rowH = narrow ? 40 : 28;
        ['glm', 'tf', 'qwen', 'ds'].forEach((r) => {
          const col = rcol(C, r), rows = m.g[r] || [];
          pill(g, pad, y + 6, r === 'glm' ? 'GLM-5.3-Flash · vLLM' : r === 'tf' ? 'GLM-5.3-Flash · TensorFold' : r === 'qwen' ? 'Qwen3.8-Flash-Next' : 'DeepSeek-V4-Flash', col, { font: 'sans', size: 11.5, weight: 600, fa: 0.2 });
          y += 20;
          if (!rows.length) { y += wtext(g, 'not published by the recipe (/metrics has tensorfold:mtp_*, no per-position curve)', pad + 8, y + 12, w - 2 * pad - 8, { size: 11, color: C.muted, lh: 14, maxLines: 2 }) + 8; }
          rows.forEach(([v, l, cond]) => {
            if (narrow) {
              V.text(g, l, pad, y + 10, { size: 11.5, color: C.ink, weight: 600, maxW: w - 2 * pad - 70 });
              V.text(g, cond, pad, y + 24, { size: 11, color: C.muted, maxW: w - 2 * pad });
              const yb = y + 30;
              V.box(g, pad, yb, (w - 2 * pad - 64) * V.clamp(v / m.max, 0, 1) * grow + 2, 7, { r: 3.5, fill: V.alpha(col, 0.75) });
              V.text(g, m.fmt(v), w - pad, y + 12, { size: 12, font: 'mono', weight: 700, color: C.ink, align: 'right' });
            } else {
              V.text(g, l, pad + 8, y + 10, { size: 12, color: C.ink, weight: 600, maxW: labW - 8 });
              V.text(g, cond, pad + 8, y + 24, { size: 11, color: C.muted, maxW: labW - 8 });
              V.box(g, bx, y + 4, bw, 16, { r: 8, fill: V.alpha(C.muted, 0.08) });
              V.box(g, bx, y + 4, Math.max(3, bw * V.clamp(v / m.max, 0, 1) * grow), 16, { r: 8, fill: V.alpha(col, 0.7), stroke: col, lw: 1 });
              V.text(g, m.fmt(v), bx + bw + 8, y + 16, { size: 12, font: 'mono', weight: 700, color: C.ink });
            }
            y += rowH + (narrow ? 6 : 3);
          });
          y += 6;
        });
        wtext(g, m.note, pad, Math.min(h - 28, y + 10), w - 2 * pad, { size: 11.5, color: C.d, weight: 500, lh: 15, maxLines: 3 });
      });
    })();
    fixCtl();
  },
});

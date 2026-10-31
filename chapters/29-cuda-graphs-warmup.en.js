LA.chapter({
  id: 'cuda-graphs-warmup',
  part: 5,
  order: 29,
  title: 'CUDA graphs, torch.compile and warmup',
  navTitle: 'CUDA graphs & warmup',
  subtitle: 'Why an LLM\'s decode can end up waiting on the CPU, how to record and replay an entire GPU step, and the boot ritual — compile, tune, warm up, verify — that the recipes go through before the first request.',
  minutes: 60,
  recipes: ['glm', 'tf', 'qwen', 'ds'],
  tags: ['CUDA graphs', 'torch.compile', 'warmup', 'JIT', 'autotune', 'custom ops', 'capture sizes', 'watchdog', 'TensorFold'],
  goals: [
    'Explain why <b>decode</b> becomes limited by the cost of launching kernels and how a <b>CUDA graph</b> solves it',
    'Compute the <b>capture sizes</b> of a recipe with speculative decoding (and avoid the 42 → 48 mistake)',
    'Tell apart <b>FULL, PIECEWISE, FULL_DECODE_ONLY</b> and breakable, and understand the role of <b>torch.compile</b> and custom ops',
    'Recognize the <b>host-device syncs</b>, JIT and autotune that break the capture or hang ranks — and the warmup that prevents them',
    'Compare vLLM\'s design (one graph per uniform batch size) with <b>TensorFold</b>\'s (one graph per window width, metadata in device-side tables)',
  ],
  prereqs: ['gpu-por-dentro', 'kernels'],
  summary: [
    'In decode, short kernels leave the step limited by launches on the CPU; a CUDA graph records the sequence once and replays it with a single call.',
    'Recording freezes pointers, shapes, strides and branches: inputs go into static buffers, and nothing in the hot path may synchronize, copy from the CPU or allocate.',
    'One graph per size: with speculation, uniform decode has <b>c·(k+1)</b> tokens; round the ceiling up (DS 42 → 48: 122 → 139 tok/s) and look for “FULL n/n” in the log.',
    'PIECEWISE splits at attention and needs torch.compile (mode 3); FULL_DECODE_ONLY (Qwen) records only uniform decode; breakable was pinned to 0 in DS and in Qwen v0.30 (in DS, regular graphs gave +28.6% at c=1).',
    'Custom ops with a <b>fake impl</b> and <b>mutates_args</b> make extensions opaque to the compiler and capturable; the AOT cache must be invalidated when the code changes.',
    'JIT and autotune in the middle of serving hang the TP pair (NCCL watchdog, 600 s): persist caches, warm up buckets (\\(s_B = B - q\\)), tune before capture in a consistent way and keep the environment identical across ranks.',
    '<b>TensorFold</b> (the GLM recipe\'s 2nd version) records one graph per <b>verify-window width</b> (1–8 and 16 per stream; 1–32 batched) and puts positions, slots, parities and extents in <b>device-side tables</b>: each stream keeps its own draft depth, with no uniform batch; what still freezes is handled with a device-side epoch (RDMA collective), bounds read on the device and <code>TF_GLM_MULTI_LONE=0</code>.',
  ],
  quiz: [
    { q: 'In the DS default lane (6 conversations, k=6), the old compose file passed <code>--max-cudagraph-capture-size 42</code>. What happened to the 6-conversation step?', o: ['It ran in a 42-token graph', 'It was padded to a 48-token graph', 'It ran eager: 42 was truncated to 40 and the rounded grid became [7, 14, 21, 28, 35]', 'It was split into two 21-token graphs'], a: 2, e: 'The default grid is [1,2,4,8,…,40]; 42 isn\'t on it, so the ceiling is truncated to 40. Rounding to multiples of k+1 = 7 discards 40 → 42 (above the ceiling), leaving [7,…,35]. The 42-token step has no graph and runs eager — the “G capture42” boot measured 122 tok/s versus 139 with a ceiling of 48.' },
    { q: 'Why is a <code>.tolist()</code> in the middle of the forward incompatible with capturing a CUDA graph?', o: ['Because no kernel executes during capture: the value the CPU asks for doesn\'t exist, and even if it did, it would be frozen onto the tape', 'Because .tolist() only works in BF16', 'Because Python lists don\'t fit in GPU memory', 'Because NCCL forbids lists in collectives'], a: 0, e: 'Capture only records operations; the CPU can\'t wait for a result that hasn\'t been computed (CUDA refuses and invalidates the capture). Outside of capture, the same call drains the queue and creates a bubble — in GLM, <code>aten::nonzero</code> reached 1.45 s of CPU time in a 3-step profile.' },
    { q: 'At TP=2, rank 1 has to compile a kernel that takes 450 s while rank 0 waits in the all-reduce. What happens with the stock <code>VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS</code> deadline (300 s), and with the recipes\' deadline (1800 s)?', o: ['It dies in both cases, killed by the NCCL watchdog', 'Stock: the EngineCore declares a hang at 300 s; 1800: it survives with a latency spike, because 450 s < the watchdog\'s 600 s', 'It survives in both cases', 'Stock survives; with 1800 it dies'], a: 1, e: 'The RPC deadline belongs to vLLM; the NCCL watchdog (600 s) is a different clock. With 300 s, the RPC times out before the compilation finishes. With 1800 s, the 450 s compilation finishes before the watchdog\'s 600 s. Above 600 s, neither one saves you — hence persistent caches and warmup.' },
    { q: 'In GLM\'s warmup (k=7), the kernel key is <code>min(256, next_pow2(s + 8))</code>. Why does the ladder start at s = 1 and not try to warm up BLOCK 8?', o: ['Because BLOCK 8 is the fastest and needs no warmup', 'Because Triton doesn\'t compile blocks of 8', 'Because s = 0 causes a tokenization error', 'Because with s ≥ 1 the smallest value is next_pow2(9) = 16: BLOCK 8 is unreachable'], a: 3, e: 'With q = 1 + k = 8, any s ≥ 1 gives s + 8 ≥ 9, whose next power of 2 is 16. That is why the script says “BLOCK 8 is unreachable” and uses s = {1, 24, 56, 120, 248}, practically the rule s = B − 8 for B = 16…256.' },
    { q: 'What is the <i>fake impl</i> of a custom op registered with <code>direct_register_custom_op</code> for?', o: ['To run the op on the CPU when the GPU is busy', 'To test the op with random data in CI', 'To return tensors with the correct shape and dtype during tracing, without computing anything', 'To turn the op off in eager mode'], a: 2, e: 'The compiler runs the graph on fake tensors (shape and dtype only). The fake impl states the output shape — in GLM, <code>x.new_empty((*x.shape[:-1], sum(output_sizes)))</code>; in Qwen\'s short-conv, nothing, because the op writes into an <code>output</code> declared in <code>mutates_args</code>.' },
    { q: 'Why does Qwen\'s v0.30 lane put the FlashInfer autotune cache in <code>/tmp</code> inside the container?', o: ['Because a persisted cache that differed between the nodes deadlocked TP2 in the autotune collective: better for both ranks to tune together from scratch', 'Because /tmp is faster than the volume', 'Because autotune doesn\'t work on mounted volumes', 'Because v0.30 doesn\'t autotune'], a: 0, e: 'The CHANGELOG (2026-09-25) records that a persisted per-node cache that differed between the nodes deadlocked TP2 in the autotune collective (measured once). Interpretation of the mechanism: with different caches, one rank skips the autotune while the other enters the autotune\'s synchronization collective, and the collectives don\'t match. DS\'s Stage-C overlay (lane B) solves the same problem another way: rank 0 tunes and broadcasts the JSON.' },
  ],
  mount(root, LA) {
    const V = LA.viz, UI = LA.ui;
    const q = (s) => root.querySelector(s);
    const fcache = {};
    const fontOf = (size, font, weight) => { const k = size + font + weight; return fcache[k] || (fcache[k] = V.font(size, font, weight)); };
    const tw = (g, s, size = 12, font = 'sans', weight = 400) => { g.font = fontOf(size, font, weight); return g.measureText(String(s)).width; };
    const onInk = (C) => (C.dark ? '#00210F' : '#ffffff');
    const fmtUs = (x) => (x >= 1000 ? (x / 1000).toFixed(2) + ' ms' : x.toFixed(0) + ' µs');
    const nextPow2 = (n) => { let p = 1; while (p < n) p *= 2; return p; };
    const fmtN = (n) => String(n).replace(/\B(?=(\d{3})+(?!\d))/g, ',');
    function hatch(g, x, y, w, h, col, step = 5) {
      if (w <= 0 || h <= 0) return;
      g.save(); g.beginPath(); g.rect(x, y, w, h); g.clip();
      g.strokeStyle = col; g.lineWidth = 1.2; g.beginPath();
      for (let k = -h; k < w; k += step) { g.moveTo(x + k, y + h); g.lineTo(x + k + h, y); }
      g.stroke(); g.restore();
    }
    function pill(g, x, y, s, col, o = {}) {
      const C = V.C();
      const size = o.size || 11.5, weight = o.weight || 600, font = o.font || 'sans';
      const pw = tw(g, s, size, font, weight) + 14, ph = size + 9;
      const xx = o.align === 'right' ? x - pw : o.align === 'center' ? x - pw / 2 : x;
      V.box(g, xx, y - ph / 2, pw, ph, { r: ph / 2, fill: V.alpha(col, o.a == null ? 0.16 : o.a), stroke: V.alpha(col, 0.75), lw: 1 });
      V.text(g, s, xx + pw / 2, y + 0.5, { size, align: 'center', base: 'middle', color: o.color || C.ink, weight, font });
      return pw;
    }
    function cellsM(g, o) {
      const C = V.C(); const R = o.data.length, K = o.data[0].length, gp = 2;
      for (let i = 0; i < R; i++) for (let j = 0; j < K; j++) {
        const v = o.data[i][j], x = o.x + j * (o.cs + gp), y = o.y + i * (o.cs + gp);
        V.box(g, x, y, o.cs, o.cs, { r: 3, fill: o.fill(v, i, j), stroke: o.stroke ? o.stroke(v, i, j) : null, lw: 1 });
        if (o.cs >= 15) V.text(g, o.txt ? o.txt(v) : v, x + o.cs / 2, y + o.cs / 2 + 1, { size: 11, font: 'mono', align: 'center', base: 'middle', color: o.tcol ? o.tcol(v) : C.ink });
      }
      if (o.label) V.text(g, o.label, o.x, o.y - 8, { size: 11, font: 'mono', color: o.lcol || C.muted, maxW: o.maxW });
      return { x: o.x, y: o.y, w: K * (o.cs + gp) - gp, h: R * (o.cs + gp) - gp };
    }
    function wrap(g, str, x, y, maxW, o = {}) {
      const size = o.size || 11.5, lh = o.lh || size + 5;
      const words = String(str).split(' '); let line = '', n = 0;
      words.forEach((wd) => { const t2 = line ? line + ' ' + wd : wd; if (tw(g, t2, size, o.font || 'sans', o.weight || 400) > maxW && line) { V.text(g, line, x, y + n * lh, o); n++; line = wd; } else line = t2; });
      if (line) { V.text(g, line, x, y + n * lh, o); n++; }
      return n;
    }
    // matrix with readable numbers (11 px text), same colors as LA.viz.matrix
    function mat(g, o) {
      const C = V.C(); const cm = o.cmap || 'seq';
      const norm = (v) => (cm === 'div' ? v / Math.max(Math.abs(o.min), Math.abs(o.max), 1e-9) : (v - o.min) / (o.max - o.min || 1));
      const r = cellsM(g, {
        x: o.x, y: o.y, data: o.data, cs: o.cell,
        fill: (v) => (v == null || Number.isNaN(v) ? V.alpha(C.muted, 0.12) : V.cmap(norm(v), cm)),
        txt: (v) => (v == null || Number.isNaN(v) ? '' : (o.fmt ? o.fmt(v) : String(v))),
        tcol: (v) => { if (v == null || Number.isNaN(v)) return C.ink; const t = norm(v); const light = cm === 'div' ? Math.abs(t) > 0.55 : t > 0.6; return light ? (cm === 'heat' ? (C.dark ? '#1a0b33' : '#fff') : (C.dark ? '#00210F' : '#fff')) : C.ink; },
      });
      if (o.label) V.text(g, o.label, o.x + r.w / 2, o.y - 10, { size: 11.5, align: 'center', color: C.muted, font: 'mono' });
      if (o.rowLabels) o.rowLabels.forEach((l, i) => V.text(g, l, o.x - 6, o.y + i * (o.cell + 2) + o.cell / 2 + 4, { size: 11, align: 'right', color: C.muted, font: 'mono' }));
      return r;
    }
    const autoH = (fig, px) => { const stg = fig.querySelector('.la-fig-stage'); const cur = parseFloat(stg.style.height) || stg.clientHeight; if (Math.abs(cur - px) > 3) stg.style.height = Math.round(px) + 'px'; };
    function panel(g, P, title, col) {
      const C = V.C();
      V.box(g, P.x, P.y, P.w, P.h, { r: 12, fill: V.alpha(col, 0.05), stroke: V.alpha(col, 0.4), lw: 1 });
      V.text(g, title, P.x + 12, P.y + 17, { size: 11.5, font: 'mono', color: col, weight: 600 });
    }

    /* =====================================================================
       FIG 1 — CPU launching × GPU executing (eager vs graph)
       ===================================================================== */
    {
      const fig = q('#fig-launch');
      const S = { cpu: 12, e: 0 };
      const base = [
        { n: 'norm', g: 2, hv: 0.15 }, { n: 'qkv', g: 8, hv: 1 }, { n: 'rope', g: 1.5, hv: 0.15 }, { n: 'attn', g: 6, hv: 0.6 },
        { n: 'o_proj', g: 6, hv: 1 }, { n: 'AR', g: 8, hv: 0.5 }, { n: 'norm', g: 2, hv: 0.15 }, { n: 'router', g: 3, hv: 0.3 },
        { n: 'MoE', g: 14, hv: 1 }, { n: 'AR', g: 8, hv: 0.5 },
      ];
      let M = null;
      const ctl = UI.controls(fig);
      UI.slider(ctl, { label: 'CPU/kernel', min: 2, max: 40, step: 1, value: S.cpu, fmt: (v) => v + ' µs', onInput: (v) => { S.cpu = v; M = null; } });
      UI.slider(ctl, { label: 'Tokens/step', min: 0, max: 8, step: 1, value: S.e, fmt: (v) => String(2 ** v), onInput: (v) => { S.e = v; M = null; } });
      const rE = UI.readout(ctl, 'eager step', '—');
      const rG = UI.readout(ctl, 'graph step', '—');
      const rS = UI.readout(ctl, 'speedup', '—');
      const compute = () => {
        const T = 2 ** S.e;
        const ks = [];
        for (let l = 0; l < 3; l++) base.forEach((k) => ks.push({ n: k.n, g: k.g * (1 + (k.hv * (T - 1)) / 24) }));
        const E = []; let c = 0, end = 0, busy = 0;
        ks.forEach((k) => { const c0 = c; c += S.cpu; const s = Math.max(c, end); E.push({ n: k.n, c0, c1: c, s, e: s + k.g, gap: s - end }); end = s + k.g; busy += k.g; });
        const G = []; const prep = S.cpu, rl = S.cpu; let t = prep + rl;
        ks.forEach((k) => { G.push({ n: k.n, s: t, e: t + k.g }); t += k.g + 0.5; });
        M = { T, E, Te: end, G, Tg: t - 0.5, busy, prep, rl, avg: busy / ks.length };
        rE.set(fmtUs(M.Te)); rG.set(fmtUs(M.Tg)); rS.set((M.Te / M.Tg).toFixed(2) + '×');
      };
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        if (!M) compute();
        const sm = w < 560;
        const lw = sm ? 54 : 96, x0 = lw + 10, x1 = w - 14;
        const Tmax = Math.max(M.Te, M.Tg) * 1.03;
        const X = (u) => x0 + (u / Tmax) * (x1 - x0);
        const P = 6.5, hold = 1.6;
        const tt = Math.min(1, (t % P) / (P - hold)) * Tmax;
        const rh = 22;
        const yE = 32, yEg = 62, yG = 144, yGg = 174;
        V.text(g, sm ? 'EAGER' : 'EAGER MODE — the CPU launches kernel by kernel', 14, 20, { size: 11.5, font: 'mono', color: C.muted, maxW: w - 28 });
        V.text(g, sm ? 'CUDA GRAPH' : 'CUDA GRAPH — one call plays the whole tape', 14, yG - 12, { size: 11.5, font: 'mono', color: C.muted, maxW: w - 28 });
        const rowLab = (y, lab, col) => { V.dot(g, 18, y + rh / 2, 4, col); V.text(g, lab, 28, y + rh / 2 + 4, { size: 12, color: C.ink, weight: 600 }); V.line(g, x0, y + rh + 3, x1, y + rh + 3, { color: V.alpha(C.muted, 0.15), lw: 1 }); };
        rowLab(yE, 'CPU', C.d); rowLab(yEg, 'GPU', C.a); rowLab(yG, 'CPU', C.d); rowLab(yGg, 'GPU', C.a);
        const kern = (s, e, y, n) => {
          const a = X(s), b = X(e);
          const done = tt >= e, run = tt >= s && tt < e;
          V.box(g, a + 0.4, y, Math.max(1.2, b - a - 0.8), rh, { r: 3, fill: V.alpha(C.a, done ? 0.82 : run ? 0.95 : 0.13), stroke: done || run ? null : V.alpha(C.a, 0.4), lw: 1 });
          if (run) V.glow(g, (a + b) / 2, y + rh / 2, 14, C.a, 0.55);
          if (b - a > 38) V.text(g, n, (a + b) / 2, y + rh / 2 + 1, { size: 11, font: 'mono', align: 'center', base: 'middle', color: done || run ? onInk(C) : C.muted });
        };
        // eager CPU
        M.E.forEach((k) => {
          const a = X(k.c0), b = X(k.c1); const done = tt >= k.c1;
          V.box(g, a + 0.5, yE, Math.max(1, b - a - 1), rh, { r: 3, fill: V.alpha(C.d, done ? 0.7 : 0.14), stroke: done ? null : V.alpha(C.d, 0.35), lw: 1 });
        });
        // eager GPU: bubbles + kernels
        M.E.forEach((k) => { if (k.gap > 0.05) { const a = X(k.s - k.gap), b = X(k.s); if (b - a > 1) hatch(g, a, yEg, b - a, rh, V.alpha(C.red, tt >= k.s ? 0.8 : 0.28)); } });
        M.E.forEach((k) => kern(k.s, k.e, yEg, k.n));
        const qn = M.E.filter((k) => k.c1 <= tt && k.s > tt).length;
        const idle = 1 - M.busy / M.Te;
        V.text(g, `in queue: ${qn} · GPU idle: ${(idle * 100).toFixed(0)}%`, x0, yEg + rh + 20, { size: 11.5, font: 'mono', color: C.muted });
        if (tt >= M.Te - 0.01) pill(g, x1, yEg + rh + 16, sm ? '✓ token' : '✓ token ready', C.a, { align: 'right', size: 11 });
        // graph CPU
        const a0 = X(0), a1 = X(M.prep), a2 = X(M.prep + M.rl);
        V.box(g, a0 + 0.5, yG, Math.max(1, a1 - a0 - 1), rh, { r: 3, fill: V.alpha(C.d, tt >= M.prep ? 0.7 : 0.14), stroke: V.alpha(C.d, 0.4), lw: 1 });
        V.box(g, a1 + 0.5, yG, Math.max(1, a2 - a1 - 1), rh, { r: 3, fill: V.alpha(C.f, tt >= M.prep + M.rl ? 0.7 : 0.14), stroke: V.alpha(C.f, 0.5), lw: 1 });
        if (a1 - a0 > 60) V.text(g, 'inputs', (a0 + a1) / 2, yG + rh / 2 + 1, { size: 11, align: 'center', base: 'middle', color: C.ink });
        if (a2 - a1 > 50) V.text(g, 'replay', (a1 + a2) / 2, yG + rh / 2 + 1, { size: 11, align: 'center', base: 'middle', color: C.ink });
        V.text(g, 'CPU free for the next step →', Math.min(x1 - 4, a2 + 8), yG + rh / 2 + 4, { size: 11, color: C.muted, maxW: Math.max(10, x1 - a2 - 12) });
        // graph GPU
        const G0 = M.G[0], GL = M.G[M.G.length - 1];
        V.box(g, X(G0.s) - 3, yGg - 4, X(GL.e) - X(G0.s) + 6, rh + 8, { r: 8, fill: V.alpha(C.b, 0.07), stroke: V.alpha(C.b, 0.75), lw: 1.2 });
        if (X(G0.s) - x0 > 4) hatch(g, x0, yGg, X(G0.s) - x0 - 4, rh, V.alpha(C.red, tt >= G0.s ? 0.6 : 0.2));
        M.G.forEach((k) => kern(k.s, k.e, yGg, k.n));
        if (tt >= M.Tg - 0.01) pill(g, Math.min(x1, X(M.Tg) + (sm ? 70 : 110)), yGg + rh + 16, sm ? '✓ token' : '✓ token ready', C.a, { align: 'right', size: 11 });
        // playheads
        [[yE - 6, yEg + rh + 4], [yG - 6, yGg + rh + 6]].forEach(([ya, yb]) => V.line(g, X(tt), ya, X(tt), yb, { color: V.alpha(C.ink, 0.5), lw: 1.2 }));
        // axis
        const ay = 228;
        V.line(g, x0, ay, x1, ay, { color: V.alpha(C.muted, 0.5), lw: 1 });
        const steps = [5, 10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000, 5000];
        const stp = steps.find((s) => Tmax / s <= (sm ? 3.5 : 7)) || 10000;
        for (let u = 0; u <= Tmax + 0.01; u += stp) { const x = X(u); V.line(g, x, ay, x, ay + 5, { color: C.muted, lw: 1 }); V.text(g, u === 0 ? '0' : fmtUs(u), x, ay + 18, { size: 11, align: 'center', color: C.muted, font: 'mono' }); }
        // summary
        V.text(g, `eager: ${fmtUs(M.Te)} per step · GPU idle ${(idle * 100).toFixed(0)}%`, 14, 276, { size: 13, color: C.ink, weight: 600, maxW: w - 28 });
        V.text(g, `graph: ${fmtUs(M.Tg)} per step · ${(M.Te / M.Tg).toFixed(2)}× faster`, 14, 298, { size: 13, color: C.ink, weight: 600, maxW: w - 28 });
        V.dot(g, w - 20, 294, 0, C.a);
        const lb = S.cpu > M.avg;
        V.text(g, lb ? `launch-bound: t_cpu ${S.cpu} µs > mean kernel ${M.avg.toFixed(1)} µs` : `GPU-bound: mean kernel ${M.avg.toFixed(1)} µs ≥ t_cpu ${S.cpu} µs`, 14, 322, { size: 12, color: lb ? C.d : C.a, maxW: w - 28, weight: 500 });
        V.text(g, `${M.T} token(s) per step · 3 layers × 10 kernels · illustrative durations`, 14, 344, { size: 11.5, color: C.muted, maxW: w - 28 });
      });
    }

    /* =====================================================================
       FIG 2 — The tape: capture, instantiation and replay (stepper)
       ===================================================================== */
    {
      const fig = q('#fig-tape');
      const W = [[1, 0, -1, 1], [0, 1, 1, -1], [1, -1, 0, 1], [-1, 1, 1, 0]], b = [0, 1, -1, 0];
      const XA = [[1, 2, 0, 1], [0, 1, 2, 1], [2, 0, 1, 0], [1, 1, 1, 2]];
      const XB = [[2, 0, 1, 1], [1, 2, 0, 0], [0, 1, 1, 2], [3, 0, 0, 1]];
      const XN = [[0, 1, 1, 0], [2, 2, 1, 0], [1, 0, 0, 3], [0, 2, 1, 1]];
      const f = (X) => V.matmul(X, W).map((r) => r.map((v, j) => Math.max(0, v + b[j])));
      const YA = f(XA), YB = f(XB);
      const EMPTY = XA.map((r) => r.map(() => NaN));
      const steps = [
        { t: 'Eager run (warmup)', d: 'Before recording, the forward runs normally. Each node is launched by the CPU and executes right away; lazy initializations (handles, allocations, autotunes) happen here, <b>outside</b> the recording.' },
        { t: 'Capture: record without executing', d: '<code>with torch.cuda.graph(g):</code> — the stream enters recording mode. The same launches become <b>nodes</b> on the tape, along with the addresses they read and write. No kernel runs: Y stays empty.' },
        { t: 'Instantiate', d: '<code>cudaGraphInstantiate</code> validates and optimizes the tape just once and returns an executable object. Tensors created during capture come from a private pool, so their addresses remain valid.' },
        { t: 'Replay with new data', d: 'On the next step, the CPU <b>copies</b> the new data into the same <code>X@0xA000</code> buffer and calls <code>g.replay()</code>: one launch fires all four nodes, back to back. The output shows up in the same <code>Y@0xC000</code>.' },
        { t: 'Broken contract', d: 'The code creates a new tensor (<code>X2@0xD000</code>) instead of copying into the buffer. The graph has no idea: it reads <code>0xA000</code>, which still holds the previous step\'s data, and returns a <b>stale</b> Y — no error, no warning.' },
      ];
      const st = UI.stepper(fig, steps, { autoplay: 7200 });
      const nodes = ['X·W', '+ b', 'ReLU', '→ Y'];
      const sub = ['reads X, W', 'writes H', 'H → H', 'writes Y'];
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const i = st.i, el = st.elapsed();
        const wide = w >= 640;
        const cell = wide ? 24 : 19, mw = 4 * cell + 6;
        let xX, yX, xY, yY, tx0, tx1, ty;
        if (wide) { xX = 26; yX = 112; xY = w - 26 - mw; yY = 112; tx0 = xX + mw + 44; tx1 = xY - 44; ty = 140; }
        else { xX = 16; yX = 100; xY = w - 16 - mw; yY = 100; tx0 = 14; tx1 = w - 14; ty = 100 + mw + 72; }
        const nh = 48, gap = wide ? 16 : 8;
        const nw = Math.min(124, (tx1 - tx0 - 3 * gap) / 4);
        const tapeW = nw * 4 + gap * 3; const tStart = tx0 + (tx1 - tx0 - tapeW) / 2;
        const NX = (k) => tStart + k * (nw + gap);
        // ---- CPU lane
        const lx0 = 60, lx1 = w - 16, ly = 26, lh = 22;
        V.text(g, 'CPU', 16, ly + lh / 2 + 4, { size: 12, weight: 600, color: C.ink });
        V.line(g, lx0, ly + lh + 4, lx1, ly + lh + 4, { color: V.alpha(C.muted, 0.2), lw: 1 });
        const lane = (k, n, lab, col, on) => {
          const bw = Math.min(120, (lx1 - lx0 - 3 * 8) / 4);
          const x = lx0 + k * (bw + 8);
          V.box(g, x, ly, bw, lh, { r: 5, fill: V.alpha(col, on ? 0.55 : 0.12), stroke: V.alpha(col, 0.6), lw: 1 });
          V.text(g, lab, x + bw / 2, ly + lh / 2 + 1, { size: 11, align: 'center', base: 'middle', color: C.ink, maxW: bw - 6 });
        };
        let nodeState = [0, 0, 0, 0]; // 0 off, 1 recorded, 2 running, 3 done
        let Xd = XA, Yd = EMPTY, yLabel = 'Y @0xC000', yBad = false, xHi = false;
        let status = '';
        if (i === 0) {
          nodes.forEach((n, k) => { const s0 = 0.5 + 0.7 * k; lane(k, 4, 'launch ' + n, C.d, el > s0); nodeState[k] = el > s0 + 0.55 ? 3 : el > s0 ? 2 : 0; });
          Yd = el > 3.3 ? YA : EMPTY;
          status = '4 CPU launches → 4 kernels executed, one by one';
        } else if (i === 1) {
          nodes.forEach((n, k) => { const s0 = 0.4 + 0.55 * k; lane(k, 4, 'record ' + n, C.red, el > s0); nodeState[k] = el > s0 ? 1 : 0; });
          status = 'recording: 0 kernels executed — the tape only takes notes';
          if (Math.sin(t * 6) > 0) { V.dot(g, tStart + 6, ty - 24, 5, C.red); }
          V.text(g, 'REC', tStart + 16, ty - 20, { size: 11, font: 'mono', color: C.red, weight: 700 });
        } else if (i === 2) {
          lane(0, 4, 'instantiate', C.b, true);
          nodeState = [1, 1, 1, 1];
          status = 'graphExec: 1 executable object, validated once';
        } else if (i === 3) {
          lane(0, 4, 'copy_ into X', C.d, el > 0.2); lane(1, 4, 'replay()', C.f, el > 1.1);
          Xd = el > 0.9 ? XB : XA; xHi = el > 0.2 && el < 1.4;
          nodes.forEach((n, k) => { const s0 = 1.2 + 0.18 * k; nodeState[k] = el > s0 + 0.16 ? 3 : el > s0 ? 2 : 1; });
          Yd = el > 2.1 ? YB : EMPTY.map((r, a) => r.map((v, bb) => YA[a][bb]));
          status = '2 CPU calls (copy + replay) → 4 kernels back to back';
        } else {
          lane(0, 4, 'X2 = new tensor', C.red, el > 0.2); lane(1, 4, 'replay()', C.f, el > 0.9);
          Xd = XB;
          nodes.forEach((n, k) => { const s0 = 1.0 + 0.18 * k; nodeState[k] = el > s0 + 0.16 ? 3 : el > s0 ? 2 : 1; });
          Yd = YB; yBad = el > 1.9; yLabel = 'Y @0xC000';
          status = 'the graph read 0xA000: Y is from the previous step, not f(X2)';
        }
        // ---- matrices
        const Xg = mat(g, { x: xX, y: yX, data: Xd, cell, gap: 2, values: true, min: 0, max: 3, cmap: 'seq', fmt: (v) => (Number.isNaN(v) ? '' : v.toFixed(0)), label: 'X @0xA000' });
        if (xHi) { g.strokeStyle = C.c; g.lineWidth = 2.5; V.rr(g, Xg.x - 3, Xg.y - 3, Xg.w + 6, Xg.h + 6, 6); g.stroke(); V.text(g, 'copy_', Xg.x + Xg.w / 2, Xg.y + Xg.h + 16, { size: 11, font: 'mono', align: 'center', color: C.c }); }
        else V.text(g, 'static buffer', Xg.x + Xg.w / 2, Xg.y + Xg.h + 16, { size: 11, align: 'center', color: C.muted });
        const Yg = mat(g, { x: xY, y: yY, data: Yd, cell, gap: 2, values: true, min: 0, max: 5, cmap: 'seq', fmt: (v) => (Number.isNaN(v) ? '' : v.toFixed(0)), label: yLabel, hiColor: C.red });
        if (yBad) { g.strokeStyle = C.red; g.lineWidth = 2.5; V.rr(g, Yg.x - 3, Yg.y - 3, Yg.w + 6, Yg.h + 6, 6); g.stroke(); V.text(g, 'stale!', Yg.x + Yg.w / 2, Yg.y + Yg.h + 16, { size: 11.5, align: 'center', color: C.red, weight: 700 }); }
        else V.text(g, 'static buffer', Yg.x + Yg.w / 2, Yg.y + Yg.h + 16, { size: 11, align: 'center', color: C.muted });
        // weights W and bias b (fixed addresses, read by the tape)
        {
          let wx, wy, show = true;
          if (wide) { wx = tStart; wy = ty + nh + 58; show = i !== 4 && wy + mw < h - 26; }
          else { wx = (xX + mw + xY) / 2 - mw / 2; wy = yX; show = i !== 4 && xY - (xX + mw) > mw + 16; }
          if (show) {
            const Wg = mat(g, { x: wx, y: wy, data: W, cell, gap: 2, values: true, min: -1, max: 1, cmap: 'div', fmt: (v) => v.toFixed(0) });
            V.text(g, 'W @0x3F00', wx, wy - 8, { size: 11.5, font: 'mono', color: C.muted });
            if (wide) {
              const bx = wx + mw + 24;
              mat(g, { x: bx, y: wy, data: [b], cell, gap: 2, values: true, min: -1, max: 1, cmap: 'div', fmt: (v) => v.toFixed(0) });
              V.text(g, 'b', bx, wy - 8, { size: 11.5, font: 'mono', color: C.muted });
              V.text(g, 'weights: fixed addresses too,', bx, wy + cell + 26, { size: 11, color: C.muted });
              V.text(g, 'read on every replay', bx, wy + cell + 42, { size: 11, color: C.muted });
            } else V.text(g, 'weights', wx + mw / 2, Wg.y + Wg.h + 16, { size: 11, align: 'center', color: C.muted });
          }
        }
        if (i === 4) {
          let x2x, x2y;
          if (wide) { x2x = tStart; x2y = ty + nh + 58; } else { x2x = (xX + mw + xY) / 2 - mw / 2; x2y = yX; }
          const room = wide ? x2y + mw < h - 26 : xY - (xX + mw) > mw + 16;
          if (room) {
            mat(g, { x: x2x, y: x2y, data: XN, cell, gap: 2, values: true, min: 0, max: 3, cmap: 'heat', fmt: (v) => v.toFixed(0) });
            V.text(g, 'X2 @0xD000 (new)', x2x, x2y - 8, { size: 11.5, font: 'mono', color: C.red });
            if (wide) {
              V.line(g, x2x + mw + 10, x2y + mw / 2, x2x + mw + 70, x2y + mw / 2, { color: C.red, lw: 1.6, dash: [4, 4] });
              V.text(g, '✕ not on the tape', x2x + mw + 78, x2y + mw / 2 + 4, { size: 12, color: C.red, weight: 700 });
              V.text(g, 'the tape reads 0xA000 (previous step data)', x2x + mw + 78, x2y + mw / 2 + 22, { size: 11, color: C.muted, maxW: tx1 - x2x - mw - 70 });
            } else V.text(g, 'ignored ✕', x2x + mw / 2, x2y + mw + 16, { size: 11.5, align: 'center', color: C.red, weight: 700 });
          }
        }
        // ---- tape
        if (wide) V.text(g, i === 0 ? 'nodes executed right away' : 'tape (CUDA graph)', tStart, ty - 34, { size: 11.5, font: 'mono', color: C.muted });
        else V.text(g, i === 0 ? 'nodes executed right away' : 'tape (CUDA graph)', w / 2, ty - 18, { size: 11, font: 'mono', color: C.muted, align: 'center' });
        if (i >= 1) V.text(g, 'X=0xA000 · W=0x3F00 · H=0xB000 · Y=0xC000', tStart, ty + nh + 22, { size: 11, font: 'mono', color: C.muted, maxW: tapeW });
        if (i >= 2) {
          const k = i === 2 ? V.ease.out(Math.min(1, el / 0.8)) : 1;
          V.box(g, tStart - 8, ty - 10, tapeW + 16, nh + 20, { r: 12, fill: V.alpha(C.b, 0.08 * k), stroke: V.alpha(C.b, 0.8 * k), lw: 1.6 });
          if (i === 2) pill(g, tStart + tapeW / 2, ty - 20, 'graphExec · 1 object', C.b, { align: 'center', size: 11 });
        }
        for (let k = 0; k < 3; k++) V.arrow(g, NX(k) + nw + 1, ty + nh / 2, NX(k + 1) - 2, ty + nh / 2, { color: V.alpha(C.muted, 0.7), head: 6, lw: 1.4 });
        nodes.forEach((n, k) => {
          const s = nodeState[k], x = NX(k);
          const col = C.a;
          if (s === 0) V.box(g, x, ty, nw, nh, { r: 8, stroke: V.alpha(C.muted, 0.3), lw: 1 });
          else if (s === 1) { g.setLineDash([4, 3]); V.box(g, x, ty, nw, nh, { r: 8, fill: V.alpha(C.b, 0.1), stroke: V.alpha(C.b, 0.9), lw: 1.4 }); g.setLineDash([]); }
          else V.box(g, x, ty, nw, nh, { r: 8, fill: V.alpha(col, s === 2 ? 0.95 : 0.55), stroke: col, lw: 1.4 });
          if (s === 2) V.glow(g, x + nw / 2, ty + nh / 2, 30, col, 0.5);
          V.text(g, n, x + nw / 2, ty + 18, { size: 13, weight: 700, align: 'center', color: s >= 2 ? onInk(C) : C.ink });
          V.text(g, sub[k], x + nw / 2, ty + 35, { size: 11, align: 'center', font: 'mono', color: s >= 2 ? onInk(C) : C.muted, maxW: nw - 6 });
        });
        // links matrix → tape → matrix
        if (wide) {
          V.arrow(g, xX + mw + 6, yX + mw / 2, tStart - 12, ty + nh / 2, { color: V.alpha(C.muted, 0.8), head: 7, curve: -10 });
          V.arrow(g, NX(3) + nw + 12, ty + nh / 2, xY - 6, yY + mw / 2, { color: V.alpha(C.muted, 0.8), head: 7, curve: -10 });
        } else {
          V.arrow(g, xX + mw / 2, yX + mw + 24, NX(0) + nw / 2, ty - 12, { color: V.alpha(C.muted, 0.8), head: 7 });
          V.arrow(g, NX(3) + nw / 2, ty - 12, xY + mw / 2, yY + mw + 24, { color: V.alpha(C.muted, 0.8), head: 7 });
        }
        V.text(g, status, 16, h - 14, { size: 12, color: i === 4 ? C.red : C.ink, weight: 500, maxW: w - 32 });
      });
    }

    /* =====================================================================
       FIG 3 — Capture traps (4 scenarios + fix)
       ===================================================================== */
    {
      const fig = q('#fig-traps');
      const S = { sc: 'branch', fix: false, t0: 0 };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Scenario', options: [{ v: 'branch', label: 'CPU branch' }, { v: 'stride', label: 'frozen stride' }, { v: 'copy', label: 'CPU→GPU copy' }, { v: 'alloc', label: 'allocation' }], value: 'branch', onChange: (v) => { S.sc = v; } });
      UI.toggle(ctl, { label: 'apply the recipe\'s fix', value: false, onChange: (v) => { S.fix = v; } });
      const topkG = [[3, 12, 9, 1], [14, 8, 0, 11], [9, 2, 15, 5], [10, 4, 13, 8]];
      const topkG2 = [[1, 9, 12, 6], [15, 3, 8, 10], [0, 14, 11, 2], [9, 13, 5, 7]];
      const toLocal = (m) => m.map((r) => r.map((gid) => (gid >= 8 ? gid - 8 : 8)));
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const two = w >= 600;
        autoH(fig, two ? 300 : 480);
        const pw = two ? (w - 48) / 2 : w - 32;
        const ph = two ? h - 72 : (h - 100) / 2;
        const P1 = { x: 16, y: 12, w: pw, h: ph };
        const P2 = two ? { x: 32 + pw, y: 12, w: pw, h: ph } : { x: 16, y: 12 + ph + 14, w: pw, h: ph };
        const good = C.a, bad = C.red;
        panel(g, P1, 'DURING CAPTURE', C.b);
        panel(g, P2, 'ON REPLAY (later step)', S.fix ? good : bad);
        const pulse = 0.5 + 0.5 * Math.sin(t * 4);
        let caption = '';
        if (S.sc === 'branch') {
          const drawPaths = (P, rec, note) => {
            const bw = (P.w - 36) / 2, by = P.y + (two ? 52 : 54), bh = two ? 40 : 28;
            const A = { x: P.x + 12, lab: 'shortcut: all candidates', lab2: '_fill_short_context_topk' }, B = { x: P.x + 24 + bw, lab: 'score: logits → top-k', lab2: 'wq_b → RoPE → quant → QK' };
            [A, B].forEach((p, k) => {
              const on = (k === 0) === (rec === 'A');
              if (!on) g.setLineDash([4, 3]);
              V.box(g, p.x, by, bw, bh, { r: 8, fill: V.alpha(on ? C.b : C.muted, on ? 0.2 : 0.05), stroke: V.alpha(on ? C.b : C.muted, on ? 0.9 : 0.4), lw: on ? 1.6 : 1 });
              g.setLineDash([]);
              V.text(g, p.lab, p.x + bw / 2, by + (two ? 16 : bh / 2 + 1), { size: 11, align: 'center', base: 'middle', color: on ? C.ink : C.muted, maxW: bw - 8, weight: on ? 600 : 400 });
              if (two) V.text(g, p.lab2, p.x + bw / 2, by + 31, { size: 11, align: 'center', font: 'mono', color: C.muted, maxW: bw - 8 });
              if (on && note) pill(g, p.x + bw - 2, by - 2, note, note === '● REC' ? C.red : C.b, { align: 'right', size: 11, a: 0.2 });
            });
            return by + bh;
          };
          // candidates: each cell is a compressed KV block with its score (×100)
          const strip = (P, y, scores, sel, col, lab) => {
            const n = scores.length, cw = Math.min(30, (P.w - 24 - (n - 1) * 3) / n);
            V.text(g, lab, P.x + 12, y - 6, { size: 11, font: 'mono', color: C.muted, maxW: P.w - 24 });
            scores.forEach((sc, k) => {
              const x = P.x + 12 + k * (cw + 3);
              V.box(g, x, y, cw, 22, { r: 4, fill: V.cmap(sc, 'heat') });
              if (cw >= 17) V.text(g, Math.round(sc * 100), x + cw / 2, y + 12, { size: 11, font: 'mono', align: 'center', base: 'middle', color: sc > 0.55 ? (C.dark ? '#1a0b33' : '#fff') : C.ink });
              if (sel.includes(k)) { g.strokeStyle = col; g.lineWidth = 2.4; V.rr(g, x - 2, y - 2, cw + 4, 26, 5); g.stroke(); }
            });
            return y + 22;
          };
          const capRec = S.fix ? 'B' : 'A';
          V.text(g, 'seq_len 1,500 → 1,500/4 = 375 ≤ 512', P1.x + 12, P1.y + 34, { size: 11.5, font: 'mono', color: C.ink, maxW: P1.w - 24 });
          const e1 = drawPaths(P1, capRec, '● REC');
          const sc1 = two ? [0.3, 0.6, 0.2, 0.8, 0.4, 0.5, 0.7, 0.25, 0.45, 0.65] : [0.3, 0.6, 0.2, 0.8, 0.4, 0.5, 0.7, 0.25];
          const s1 = strip(P1, e1 + (two ? 30 : 22), sc1, sc1.map((_, k) => k), C.a, two ? 'few candidates: all of them get in (correct here)' : 'all get in (ok here)');
          wrap(g, S.fix ? 'guard #52492: is_current_stream_capturing() → no shortcut' : 'the if was evaluated on the CPU: only the shortcut was recorded', P1.x + 12, s1 + 22, P1.w - 24, { size: 11, lh: 15, color: S.fix ? good : C.muted, weight: S.fix ? 600 : 400 });
          V.text(g, 'seq_len 9,000 → 2,250 > 512: must score', P2.x + 12, P2.y + 34, { size: 11.5, font: 'mono', color: C.ink, maxW: P2.w - 24 });
          const e2 = drawPaths(P2, capRec, 'tape');
          const sc2 = two ? [0.1, 0.3, 0.05, 0.9, 0.2, 0.7, 0.15, 0.4, 0.95, 0.1, 0.6, 0.25, 0.85, 0.05, 0.5, 0.8] : [0.1, 0.3, 0.05, 0.9, 0.2, 0.7, 0.15, 0.4, 0.95, 0.1, 0.6, 0.85];
          const order = sc2.map((v, k) => [v, k]).sort((x, y) => y[0] - x[0]).map((x) => x[1]);
          const kk = two ? 6 : 4;
          const sel = S.fix ? order.slice(0, kk) : Array.from({ length: kk }, (_, k) => k);
          const s2 = strip(P2, e2 + (two ? 30 : 22), sc2, sel, S.fix ? good : bad, two ? `many candidates · top-${kk} (illustrative)` : `top-${kk}`);
          V.text(g, S.fix ? `top-${kk} by highest score ✓` : `the first ${kk}, unscored ✕`, P2.x + 12, s2 + 20, { size: 11.5, color: S.fix ? good : bad, weight: 600, maxW: P2.w - 24 });
          caption = S.fix ? 'DS: the indexer shortcut (#49486) never fires during capture (guard #52492).' : 'DS: a shortcut taken during capture is replayed for long prefixes → intermittent corruption.';
        } else if (S.sc === 'stride') {
          const grid = (P, writeStride, readStride, title) => {
            const cols = 8, rows = 4;
            const cs = Math.min(24, (P.w - 24 - 7 * 2) / cols, (P.h - (two ? 92 : 70)) / rows);
            const gx = P.x + 12, gy = P.y + (two ? 52 : 40);
            V.text(g, title, P.x + 12, P.y + 34, { size: 11.5, font: 'mono', color: C.ink, maxW: P.w - 24 });
            const owner = new Array(32).fill(-1);
            for (let r = 0; r < rows; r++) for (let j = 0; j < 5; j++) { const idx = r * writeStride + j; if (idx < 32) owner[idx] = r; }
            let badN = 0;
            for (let idx = 0; idx < 32; idx++) {
              const r = Math.floor(idx / cols), c = idx % cols, x = gx + c * (cs + 2), y = gy + r * (cs + 2);
              const o = owner[idx];
              V.box(g, x, y, cs, cs, { r: 3, fill: o >= 0 ? V.alpha(C.series[o], 0.55) : V.alpha(C.muted, 0.08), stroke: V.alpha(C.muted, 0.25), lw: 1 });
              if (o >= 0 && cs >= 16) V.text(g, 'r' + o, x + cs / 2, y + cs / 2 + 1, { size: 11, font: 'mono', align: 'center', base: 'middle', color: C.ink });
            }
            for (let r = 0; r < rows; r++) for (let j = 0; j < 5; j++) {
              const idx = r * readStride + j; if (idx >= 32) continue;
              const rr = Math.floor(idx / cols), c = idx % cols, x = gx + c * (cs + 2), y = gy + rr * (cs + 2);
              const ok = owner[idx] === r; if (!ok) badN++;
              g.strokeStyle = ok ? good : bad; g.lineWidth = ok ? 1.6 : 2.4; V.rr(g, x - 1.5, y - 1.5, cs + 3, cs + 3, 4); g.stroke();
            }
            return { bottom: gy + rows * (cs + 2), badN };
          };
          grid(P1, 8, 8, 'capture stride = 8 (write = read)');
          const live = S.fix ? 8 : 5;
          const r2 = grid(P2, live, 8, S.fix ? 'fixed width = stride 8 (hotfix removed)' : 'builder: live stride 5 · graph reads stride 8');
          V.text(g, r2.badN ? `${r2.badN} slots read from the wrong row ✕` : 'every row read from the right place ✓', P2.x + 12, r2.bottom + 18, { size: 11.5, color: r2.badN ? bad : good, weight: 600, maxW: P2.w - 24 });
          V.text(g, 'color = row that wrote · outline = what the graph reads', P1.x + 12, P1.y + P1.h - 10, { size: 11, color: C.muted, maxW: P1.w - 24 });
          caption = S.fix ? 'DS: hotfix #50004 was removed (upstream reverted it, #51318): fixed width, same as at capture.' : 'DS: adaptive top-k wrote with the live batch stride; the FULL graph read with the capture stride.';
        } else if (S.sc === 'copy') {
          const cs = Math.min(two ? 22 : 17, (P1.w - 70) / 9);
          const mw = 4 * (cs + 2) - 2;
          const heatF = (v) => V.cmap(v / 15, 'heat');
          const heatT = (v) => (v / 15 > 0.55 ? (C.dark ? '#1a0b33' : '#fff') : C.ink);
          const locF = (v) => (v === 8 ? V.alpha(C.muted, 0.18) : V.alpha(C.a, 0.3 + v * 0.07));
          const locT = (v) => (v === 8 ? C.muted : C.ink);
          const drawStrip = (P, y, onGpu) => {
            const n = 16, sw = Math.min(22, (P.w - 24 - 15) / n);
            V.text(g, onGpu ? 'expert_map (on GPU)' : 'expert_map (on CPU)', P.x + 12, y - 6, { size: 11, font: 'mono', color: onGpu ? good : C.d });
            for (let k = 0; k < n; k++) { const x = P.x + 12 + k * (sw + 1); const loc = k >= 8; V.box(g, x, y, sw, 18, { r: 3, fill: V.alpha(loc ? C.a : C.muted, loc ? 0.35 : 0.1), stroke: V.alpha(onGpu ? good : C.d, 0.5), lw: 1 }); if (sw >= 15) V.text(g, loc ? k - 8 : '·', x + sw / 2, y + 10, { size: 11, font: 'mono', align: 'center', base: 'middle', color: C.ink }); }
            return y + 18;
          };
          const top1 = P1.y + (two ? 46 : 40);
          const m1 = cellsM(g, { x: P1.x + 12, y: top1, data: topkG, cs, fill: heatF, tcol: heatT, label: two ? 'top-k ids (global)' : 'global ids', maxW: mw + 40 });
          const sy = Math.min(P1.y + P1.h - 26, m1.y + m1.h + (two ? 30 : 24));
          drawStrip(P1, sy, S.fix);
          const lx = P1.x + 12 + mw + (two ? 44 : 34);
          if (!S.fix) {
            V.arrow(g, P1.x + 12 + mw + 8, m1.y + m1.h / 2, P1.x + 12 + mw + 52, m1.y + m1.h / 2, { color: bad, head: 7, lw: 2 });
            V.text(g, '.to(cuda) on every call', P1.x + 12 + mw + 8, m1.y + m1.h / 2 - 10, { size: 11, font: 'mono', color: bad, maxW: P1.x + P1.w - (P1.x + 12 + mw + 8) - 8 });
            V.text(g, 'CPU→GPU copy: forbidden in capture ✕', P1.x + 12 + mw + 8, m1.y + m1.h / 2 + 20, { size: 11, color: bad, weight: 600, maxW: P1.x + P1.w - (P1.x + 12 + mw + 8) - 8 });
          } else if (lx + mw < P1.x + P1.w - 4) {
            V.arrow(g, P1.x + 12 + mw + 6, m1.y + m1.h / 2, lx - 6, m1.y + m1.h / 2, { color: good, head: 6 });
            cellsM(g, { x: lx, y: m1.y, data: toLocal(topkG), cs, fill: locF, tcol: locT, label: 'local ids', maxW: mw + 30 });
          }
          if (S.fix) {
            const m2 = cellsM(g, { x: P2.x + 12, y: P2.y + (two ? 46 : 40), data: topkG2, cs, fill: heatF, tcol: heatT, label: 'new ids', maxW: mw });
            const lx2 = P2.x + 12 + mw + (two ? 44 : 34);
            V.arrow(g, P2.x + 12 + mw + 6, m2.y + m2.h / 2, lx2 - 6, m2.y + m2.h / 2, { color: good, head: 6 });
            if (lx2 + mw < P2.x + P2.w - 4) cellsM(g, { x: lx2, y: m2.y, data: toLocal(topkG2), cs, fill: locF, tcol: locT, label: 'map_topk_to_local', maxW: P2.x + P2.w - lx2 - 8 });
            V.text(g, 'gray = sentinel n_local = 8', P2.x + 12, Math.min(P2.y + P2.h - 26, m2.y + m2.h + 22), { size: 11.5, color: C.ink, maxW: P2.w - 24 });
            V.text(g, 'fixed shape, all on the GPU ✓', P2.x + 12, Math.min(P2.y + P2.h - 8, m2.y + m2.h + 40), { size: 11.5, color: good, weight: 600, maxW: P2.w - 24 });
          } else {
            V.text(g, 'the capture failed:', P2.x + P2.w / 2, P2.y + P2.h / 2 - 8, { size: 14, align: 'center', color: bad, weight: 700 });
            V.text(g, 'there is no tape to play', P2.x + P2.w / 2, P2.y + P2.h / 2 + 14, { size: 13, align: 'center', color: bad, alpha: 0.6 + 0.4 * pulse });
          }
          caption = 'GLM: pin_exl3_expert_map puts the map on the GPU once; map_topk_to_local uses sentinel n_local (illustrative: 16 experts, rank has 8–15).';
        } else {
          const memBar = (P, y, scratch, badBlk) => {
            const bx = P.x + 12, bw = P.w - 24, bh = two ? 26 : 22;
            const segs = [['weights', 0.52, C.b], ['KV cache', 0.26, C.a], ['graph pool', 0.08, C.f], ['scratch', 0.14, C.d]];
            let x = bx; const pos = {};
            segs.forEach(([lab, fr, col]) => {
              const ww = bw * fr;
              const isS = lab === 'scratch';
              if (isS && !scratch) { g.setLineDash([4, 3]); V.box(g, x + 1, y, ww - 2, bh, { r: 4, stroke: V.alpha(C.muted, 0.6), lw: 1 }); g.setLineDash([]); }
              else V.box(g, x + 1, y, ww - 2, bh, { r: 4, fill: V.alpha(col, 0.35), stroke: V.alpha(col, 0.8), lw: 1 });
              if (ww > tw(g, lab, 11) + 8) V.text(g, lab, x + ww / 2, y + bh / 2 + 1, { size: 11, align: 'center', base: 'middle', color: C.ink });
              else if (isS) V.text(g, 'scratch ↓', x + ww, y - 6, { size: 11, align: 'right', color: C.d, font: 'mono' });
              pos[lab] = { x, w: ww };
              x += ww;
            });
            if (badBlk) { const sx = pos.scratch.x; V.box(g, sx + 2, y - 4, pos.scratch.w - 4, bh + 8, { r: 5, fill: V.alpha(bad, 0.25 + 0.2 * pulse), stroke: bad, lw: 2 }); V.text(g, pos.scratch.w > 60 ? 'malloc ✕' : '✕', sx + pos.scratch.w / 2, y + bh / 2 + 1, { size: 11, align: 'center', base: 'middle', color: bad, weight: 700 }); }
            V.text(g, 'GPU memory', bx, y - 6, { size: 11, font: 'mono', color: C.muted });
            return { y, bh, pos };
          };
          const nodesRow = (P, y, mb, ok) => {
            const labs = ['gather', 'gate/up', 'down'];
            const nw = Math.min(90, (P.w - 24 - 2 * 10) / 3), nh = two ? 30 : 26;
            labs.forEach((lab, k) => {
              const x = P.x + 12 + k * (nw + 10);
              V.box(g, x, y, nw, nh, { r: 7, fill: V.alpha(C.b, 0.18), stroke: V.alpha(C.b, 0.8), lw: 1.2 });
              V.text(g, lab, x + nw / 2, y + nh / 2 + 1, { size: 11, align: 'center', base: 'middle', color: C.ink, font: 'mono' });
              const sx = mb.pos.scratch.x + (k + 0.5) * (mb.pos.scratch.w / 3);
              if (ok) V.arrow(g, x + nw / 2, y - 2, sx, mb.y + mb.bh + 3, { color: V.alpha(good, 0.9), head: 5, lw: 1.2 });
            });
            return y + nh;
          };
          const y1b = P1.y + (two ? 52 : 44);
          const mb1 = memBar(P1, y1b, S.fix, !S.fix);
          const n1 = nodesRow(P1, y1b + mb1.bh + (two ? 44 : 30), mb1, S.fix);
          const l1 = wrap(g, S.fix ? 'scratch created after loading, before capture: the tape only keeps slices' : 'on-demand scratch: the 1st use lands inside the capture ✕', P1.x + 12, n1 + 20, P1.w - 24, { size: 11, lh: 15, color: S.fix ? good : bad, weight: 600 });
          wrap(g, S.fix ? 'GLM: MNBT 7,168 × top-8 = 57,344 rows' : 'allocating or growing inside the capture = error', P1.x + 12, n1 + 22 + l1 * 15, P1.w - 24, { size: 11, lh: 15, color: C.muted });
          if (S.fix) {
            const y2b = P2.y + (two ? 52 : 44);
            const mb2 = memBar(P2, y2b, true, false);
            const n2 = nodesRow(P2, y2b + mb2.bh + (two ? 44 : 30), mb2, true);
            V.text(g, 'replay: the same addresses as always ✓', P2.x + 12, n2 + 20, { size: 11, color: good, weight: 600, maxW: P2.w - 24 });
            wrap(g, 'a 7,168-token eager prefill uses the whole buffer, without growing', P2.x + 12, n2 + 37, P2.w - 24, { size: 11, lh: 15, color: C.muted });
          } else {
            V.text(g, 'the capture failed:', P2.x + P2.w / 2, P2.y + P2.h / 2 - 8, { size: 14, align: 'center', color: bad, weight: 700 });
            V.text(g, 'there is no tape to play', P2.x + P2.w / 2, P2.y + P2.h / 2 + 14, { size: 13, align: 'center', color: bad, alpha: 0.6 + 0.4 * pulse });
          }
          caption = 'GLM: E3 scratch = MNBT × topk; DS: workspace reserved before lock_workspace(); b12x (lane B) won\'t copy in capture. Illustrative proportions.';
        }
        if (two) V.text(g, caption, 16, h - 12, { size: 11.5, color: C.muted, maxW: w - 32 });
        else wrap(g, caption, 16, h - 42, w - 32, { size: 11.5, color: C.muted });
      });
    }

    /* =====================================================================
       FIG 4 — Graph parking lot: capture sizes
       ===================================================================== */
    {
      const fig = q('#fig-sizes');
      const P = {
        ds48: { lab: 'DS 6×(6+1)', seqs: 6, k: 6, rule: 'ds8' },
        ds42: { lab: 'DS unrounded', seqs: 6, k: 6, rule: 'raw' },
        ds5: { lab: 'DS k=5', seqs: 6, k: 5, rule: 'ds8' },
        tp3: { lab: 'DS TP3 16×7', seqs: 16, k: 6, rule: 'ds8' },
        glm: { lab: 'GLM DFlash2 k=7', seqs: 4, k: 7, rule: 'glm' },
        glmA: { lab: 'GLM adaptive-k', seqs: 4, k: 7, rule: 'glmA' },
        glmM: { lab: 'GLM MTP k=2', seqs: 4, k: 2, rule: 'glmM' },
        qwen: { lab: 'Qwen MTP k=3 (modeled)', seqs: 8, k: 3, rule: 'x2' },
      };
      const S = { preset: 'ds48', seqs: 6, k: 6, rule: 'ds8', t0: 0 };
      let R = null, lock = false;
      const roundUp = (s, m) => Math.ceil(s / m) * m;
      const uniq = (a) => Array.from(new Set(a)).sort((x, y) => x - y);
      function plan() {
        const qd = S.k + 1; const log = []; let sizes, teto = null, lens = [qd];
        if (S.rule === 'raw' || S.rule === 'ds8' || S.rule === 'x2') {
          const need = S.seqs * qd;
          teto = S.rule === 'raw' ? need : S.rule === 'ds8' ? Math.ceil(need / 8) * 8 : Math.min(need * 2, 512);
          const grid = [1, 2, 4]; for (let s = 8; s <= teto; s += 8) grid.push(s);
          let eff = teto;
          if (!grid.includes(teto)) { eff = Math.max(...grid.filter((s) => s <= teto)); log.push({ c: 'warn', t: `WARNING Truncating max_cudagraph_capture_size to ${eff}` }); }
          const baseL = grid.filter((s) => s <= eff);
          sizes = qd > 1 ? uniq(baseL.map((s) => roundUp(s, qd)).filter((s) => s <= eff)) : baseL;
          log.unshift({ c: 'info', t: `--max-cudagraph-capture-size ${teto}${S.rule === 'ds8' ? '  (= ⌈' + need + '/8⌉×8)' : S.rule === 'x2' ? '  (= seqs×(k+1)×2)' : ''}` });
        } else if (S.rule === 'glm' || S.rule === 'glmM') {
          const baseL = S.rule === 'glm' ? [1, 2, 4, 8, 16, 24, 32] : [1, 2, 3, 4, 6, 8, 12];
          const mx = Math.max(...baseL);
          sizes = qd > 1 ? uniq(baseL.map((s) => roundUp(s, qd)).filter((s) => s <= mx)) : baseL;
          log.push({ c: 'info', t: `--cudagraph-capture-sizes ${baseL.join(' ')}` });
        } else {
          const stock = [1, 2, 4, 8, 16, 24, 32];
          const set = [2, 4, 7];
          const L = new Set(set.map((x) => x + 1).filter((x) => x <= qd)); L.add(qd);
          lens = Array.from(L).sort((a, b) => a - b);
          const all = new Set(stock); for (let n = 1; n <= S.seqs; n++) lens.forEach((ql) => all.add(n * ql));
          sizes = uniq(Array.from(all));
          log.push({ c: 'info', t: `--cudagraph-capture-sizes ${sizes.join(' ')}` });
        }
        const rows = [];
        lens.forEach((ql) => { for (let c = 1; c <= S.seqs; c++) { const T = c * ql; const cand = sizes.filter((s) => s >= T && (ql <= 1 || s % ql === 0)); const s = cand.length ? cand[0] : null; rows.push({ c, q: ql, T, s }); } });
        const nFull = S.rule === 'glmA' ? sizes.length : sizes.filter((s) => s % qd === 0 || qd === 1).length;
        log.push({ c: 'ok', t: `Capturing CUDA graphs (FULL): ${nFull}/${nFull}${S.rule === 'x2' ? '  (modeled)' : ''}` });
        if (S.rule === 'x2') log.push({ c: 'info', t: S.seqs === 8 && S.k === 3 ? '# repo documents 1–64; exact list not published' : '# Qwen rule applied to other values (modeled)' });
        const eager = rows.filter((r) => r.s == null);
        eager.slice(0, 2).forEach((r) => log.push({ c: 'err', t: `c=${r.c} → ${r.T} tokens: no graph → eager` }));
        const padded = rows.filter((r) => r.s != null);
        const pad = padded.length ? padded.reduce((a, r) => a + (r.s - r.T) / r.s, 0) / padded.length : 0;
        R = { sizes, rows, log, eager: eager.length, pad, lens, teto };
        rF.set(String(nFull)); rE.set(`${eager.length} of ${rows.length}`); rP.set((pad * 100).toFixed(0) + '%');
        S.t0 = performance.now() / 1000;
      }
      const ctl = UI.controls(fig);
      const segP = UI.seg(ctl, { label: 'Preset', options: Object.keys(P).map((k) => ({ v: k, label: P[k].lab })), value: 'ds48', onChange: (v) => { const p = P[v]; S.preset = v; S.rule = p.rule; lock = true; sSeq.set(p.seqs); sK.set(p.k); lock = false; plan(); } });
      const sSeq = UI.slider(ctl, { label: 'max_num_seqs', min: 1, max: 16, step: 1, value: S.seqs, onInput: (v) => { S.seqs = v; if (!lock) plan(); } });
      const sK = UI.slider(ctl, { label: 'k (drafts)', min: 0, max: 7, step: 1, value: S.k, onInput: (v) => { S.k = v; if (!lock) plan(); } });
      const rF = UI.readout(ctl, 'FULL graphs', '—');
      const rE = UI.readout(ctl, 'eager steps', '—');
      const rP = UI.readout(ctl, 'mean padding', '—');
      void segP;
      plan();
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const sm = w < 560;
        const lw = S.rule === 'glmA' ? (sm ? 44 : 64) : (sm ? 36 : 50);
        const x0 = lw + 14, x1 = w - (sm ? 78 : 118);
        const maxT = Math.max(Math.max(...R.sizes), Math.max(...R.rows.map((r) => r.T))) * 1.04;
        const X = (u) => x0 + (u / maxT) * (x1 - x0);
        const ppt = (x1 - x0) / maxT;
        const nR0 = R.rows.length, rowH0 = sm ? 21 : 27;
        autoH(fig, Math.max(300, 72 + nR0 * rowH0 + 96));
        // ruler of parking spots
        const ry = 30;
        V.text(g, 'spots (captured sizes, in tokens)', 14, 16, { size: 11.5, font: 'mono', color: C.muted, maxW: sm ? w - 28 : w * 0.5 });
        if (!sm) { V.box(g, w - 330, 8, 10, 10, { r: 2, fill: V.alpha(C.a, 0.95) }); V.text(g, 'last accepted token', w - 314, 17, { size: 11, color: C.muted }); V.box(g, w - 196, 8, 10, 10, { r: 2, fill: V.alpha(C.a, 0.55) }); V.text(g, 'drafts', w - 180, 17, { size: 11, color: C.muted }); hatch(g, w - 110, 8, 12, 10, V.alpha(C.muted, 0.7), 4); V.text(g, 'padding', w - 92, 17, { size: 11, color: C.muted }); }
        V.line(g, x0, ry, x1, ry, { color: V.alpha(C.muted, 0.5), lw: 1 });
        let lastLab = -1e9;
        R.sizes.forEach((s) => {
          const x = X(s);
          V.box(g, x - 3, ry - 6, 6, 12, { r: 2, fill: V.alpha(C.b, 0.85) });
          const lab = String(s), lwid = tw(g, lab, 11, 'mono') + 4;
          if (x - lastLab >= lwid) { V.text(g, lab, x, ry + 20, { size: 11, font: 'mono', align: 'center', color: C.ink }); lastLab = x; }
        });
        // rows (one train per concurrency level)
        const top = ry + 36, bottom = h - 92;
        const nR = R.rows.length;
        const rowH = Math.min(rowH0, (bottom - top) / nR);
        const bh = Math.max(8, rowH - 7);
        const now = performance.now() / 1000;
        R.rows.forEach((r, i) => {
          const y = top + i * rowH;
          const prog = V.ease.out(V.clamp((now - S.t0 - i * 0.035) / 0.6, 0, 1));
          const lab = S.rule === 'glmA' ? `${r.c}×${r.q}` : `c=${r.c}`;
          V.text(g, lab, lw + 6, y + bh / 2 + 4, { size: 11, font: 'mono', align: 'right', color: C.muted });
          const Tlen = X(r.T) - x0;
          const segW = r.q * ppt;
          for (let c = 0; c < r.c; c++) {
            const sx = x0 + c * segW; if (sx - x0 > Tlen * prog) break;
            const ww = Math.min(segW, x0 + Tlen * prog - sx);
            const col = C.series[c % C.series.length];
            if (ppt >= 5) {
              for (let tk = 0; tk < r.q; tk++) { const cx = sx + tk * ppt; if (cx - x0 > Tlen * prog) break; V.box(g, cx + 0.6, y, ppt - 1.2, bh, { r: 2, fill: V.alpha(col, tk === 0 ? 0.95 : 0.6) }); }
            } else V.box(g, sx + 0.5, y, Math.max(1, ww - 1), bh, { r: 2, fill: V.alpha(col, 0.75) });
          }
          if (prog < 1) return;
          if (r.s != null) {
            if (r.s > r.T) hatch(g, X(r.T) + 1, y, X(r.s) - X(r.T) - 1, bh, V.alpha(C.muted, 0.6), 4);
            V.line(g, X(r.s), y - 2, X(r.s), y + bh + 2, { color: C.b, lw: 2 });
            V.text(g, `→ ${r.s}${r.s > r.T ? ' (+' + (r.s - r.T) + ')' : ''}`, x1 + 8, y + bh / 2 + 4, { size: 11, font: 'mono', color: r.s > r.T ? C.d : C.a, weight: 600 });
          } else {
            const blink = 0.55 + 0.45 * Math.sin(t * 5 + i);
            V.text(g, 'EAGER', x1 + 8, y + bh / 2 + 4, { size: 11.5, font: 'mono', color: C.red, weight: 700, alpha: blink });
            V.line(g, X(r.T), y + bh / 2, x1 + 4, y + bh / 2, { color: V.alpha(C.red, 0.5), lw: 1, dash: [3, 3] });
          }
        });
        // simulated boot log
        const ly = h - 80;
        V.box(g, 12, ly, w - 24, 70, { r: 10, fill: V.alpha(C.panel || C.muted, C.dark ? 0.6 : 0.35), stroke: V.alpha(C.muted, 0.3), lw: 1 });
        if (!sm) V.text(g, 'boot log (simulated)', w - 22, ly + 16, { size: 11, align: 'right', color: V.alpha(C.muted, 0.8) });
        else V.text(g, 'boot log (simulated)', w - 16, ly - 5, { size: 11, align: 'right', color: V.alpha(C.muted, 0.8) });
        R.log.slice(0, 4).forEach((l, i) => {
          const col = l.c === 'warn' ? C.d : l.c === 'err' ? C.red : l.c === 'ok' ? C.a : C.muted;
          V.text(g, l.t, 22, ly + 16 + i * 15, { size: 11, font: 'mono', color: col, maxW: w - (sm ? 44 : 190) });
        });
      });
    }

    /* =====================================================================
       FIG 5 — A step's conveyor belt: CUDA graph modes
       ===================================================================== */
    {
      const fig = q('#fig-modes');
      const S = { mode: 'FAP', batch: 'uni' };
      const layer = ['norm', 'qkv', 'attn', 'o_proj', 'AR', 'norm', 'MoE', 'AR'];
      const ops = [...layer, ...layer, 'norm', 'lm_head'];
      const isAttn = (i) => ops[i] === 'attn';
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Mode', options: [{ v: 'NONE', label: 'NONE' }, { v: 'PIECE', label: 'PIECEWISE' }, { v: 'FDO', label: 'FULL_DECODE_ONLY' }, { v: 'FAP', label: 'FULL_AND_PIECEWISE' }, { v: 'BRK', label: 'breakable' }], value: 'FAP', onChange: (v) => { S.mode = v; } });
      UI.seg(ctl, { label: 'Batch', options: [{ v: 'uni', label: 'uniform decode' }, { v: 'mix', label: 'mixed (prefill + decode)' }], value: 'uni', onChange: (v) => { S.batch = v; } });
      const rM = UI.readout(ctl, 'effective mode', '—');
      const rL = UI.readout(ctl, 'CPU calls', '—');
      function segs() {
        let eff = S.mode;
        if (S.mode === 'FDO') eff = S.batch === 'uni' ? 'FULL' : 'NONE';
        if (S.mode === 'FAP') eff = S.batch === 'uni' ? 'FULL' : 'PIECE';
        const seg = new Array(ops.length).fill(-1);
        if (eff === 'FULL') seg.fill(0);
        else if (eff === 'PIECE' || eff === 'BRK') { let id = -1, prev = false; ops.forEach((o, i) => { if (isAttn(i)) { prev = false; return; } if (!prev) { id++; prev = true; } seg[i] = id; }); }
        const nSeg = new Set(seg.filter((x) => x >= 0)).size;
        const nEager = seg.filter((x) => x < 0).length;
        return { eff, seg, nSeg, nEager, calls: nSeg + nEager };
      }
      const names = { NONE: 'NONE (eager)', PIECE: 'PIECEWISE', FULL: 'FULL', BRK: 'breakable' };
      const notes = {
        NONE: 'debugging: --enforce-eager / cudagraph_mode NONE',
        PIECE: 'attention runs eager between pieces; requires torch.compile (mode 3)',
        FDO: 'Qwen (mode 0): a mixed batch loses the whole graph',
        FAP: 'DS default (and, by inference, GLM too): FULL in decode, PIECEWISE for the rest',
        BRK: 'no torch.compile: the cut (✂) happens during recording, not in piecewise compilation · off in DS and Qwen v0.30 (in DS, regular graphs +28.6% at c=1)',
      };
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const Z = segs();
        rM.set(names[Z.eff]); rL.set(String(Z.calls));
        const sm = w < 560;
        const per = sm ? 4 : 8, gx = sm ? 10 : 12, gy = 34, opH = 34;
        const opW = Math.min(88, (w - 28 - (per - 1) * gx) / per);
        const nRows = Math.ceil(ops.length / per);
        const totalW = per * opW + (per - 1) * gx, ox = (w - totalW) / 2, oy = 66;
        autoH(fig, oy + nRows * (opH + gy) + (sm ? 104 : 64));
        const pos = ops.map((o, i) => { const r = Math.floor(i / per), k = i % per, c = r % 2 === 0 ? k : per - 1 - k; return { x: ox + c * (opW + gx), y: oy + r * (opH + gy), r, c }; });
        V.text(g, S.batch === 'uni' ? 'uniform decode (every conversation with k+1 tokens)' : 'mixed step: prefill chunk + decodes', 14, 18, { size: 11.5, font: 'mono', color: C.muted, maxW: w - 28 });
        V.text(g, '2 illustrative layers + output (lm_head)', 14, 34, { size: 11, color: C.muted, maxW: w - 28 });
        // segment bands (per visual row)
        const segIds = Array.from(new Set(Z.seg.filter((x) => x >= 0)));
        segIds.forEach((sid) => {
          const idx = Z.seg.map((s, i) => (s === sid ? i : -1)).filter((i) => i >= 0);
          const byRow = {};
          idx.forEach((i) => { (byRow[pos[i].r] = byRow[pos[i].r] || []).push(i); });
          Object.values(byRow).forEach((arr) => {
            const xs = arr.map((k) => pos[k].x), ya = pos[arr[0]].y;
            const xa = Math.min(...xs), xb = Math.max(...xs);
            if (Z.eff === 'BRK') g.setLineDash([5, 3]);
            V.box(g, xa - 5, ya - 6, xb + opW - xa + 10, opH + 12, { r: 10, fill: V.alpha(C.b, Z.eff === 'BRK' ? 0.1 : 0.16), stroke: V.alpha(C.b, 0.7), lw: 1.3 });
            g.setLineDash([]);
          });
          const f0 = pos[idx[0]];
          V.text(g, (Z.eff === 'BRK' ? 'seg ' : 'graph ') + (sid + 1), f0.x + (Z.eff === 'BRK' ? 8 : -2), f0.y - 11, { size: 11, font: 'mono', color: C.b, weight: 600 });
        });
        // arrows between ops
        for (let i = 0; i < ops.length - 1; i++) {
          const a = pos[i], b = pos[i + 1];
          if (a.r === b.r) { const d = b.x > a.x; V.arrow(g, d ? a.x + opW + 1 : a.x - 1, a.y + opH / 2, d ? b.x - 1 : b.x + opW + 1, b.y + opH / 2, { color: V.alpha(C.muted, 0.55), head: 5, lw: 1 }); }
          else V.arrow(g, a.x + opW / 2, a.y + opH + 7, b.x + opW / 2, b.y - 7, { color: V.alpha(C.muted, 0.6), head: 5, lw: 1.2 });
        }

        // traveling token
        const tFast = 0.07, tSlow = 0.42;
        const durs = ops.map((o, i) => (Z.seg[i] >= 0 ? tFast : tSlow));
        const tot = durs.reduce((a, b) => a + b, 0) + 0.8;
        let u = t % tot, cur = -1;
        for (let i = 0; i < ops.length; i++) { if (u < durs[i]) { cur = i; break; } u -= durs[i]; }
        ops.forEach((o, i) => {
          const p = pos[i], att = isAttn(i), eager = Z.seg[i] < 0;
          const col = att ? C.c : C.a;
          V.box(g, p.x, p.y, opW, opH, { r: 7, fill: V.alpha(col, i === cur ? 0.55 : 0.18), stroke: eager ? C.d : V.alpha(col, 0.7), lw: eager ? 2 : 1 });
          V.text(g, att ? (sm ? 'attn' : 'attention') : o, p.x + opW / 2, p.y + opH / 2 + 1, { size: 11, font: 'mono', align: 'center', base: 'middle', color: C.ink, maxW: opW - 4 });
          if (eager) { V.dot(g, p.x + opW - 5, p.y + 5, 3, C.d); if (i === cur) pill(g, p.x + opW / 2, p.y - 11, 'CPU', C.d, { align: 'center', size: 11, a: 0.3 }); }
        });
        if (cur >= 0) { const p = pos[cur]; V.glow(g, p.x + opW / 2, p.y + opH + 8, 12, C.e, 1); V.dot(g, p.x + opW / 2, p.y + opH + 8, 4, C.e); }
        const by = oy + nRows * (opH + gy) + 6;
        const pl = (n, one, many) => `${n} ${n === 1 ? one : many}`;
        const segW = Z.eff === 'BRK' ? pl(Z.nSeg, 'graph segment', 'graph segments') : pl(Z.nSeg, 'graph', 'graphs');
        const nl1 = wrap(g, `${pl(Z.calls, 'CPU call', 'CPU calls')} = ${segW} + ${pl(Z.nEager, 'eager op', 'eager ops')} · ${ops.length} ops`, 14, by + 4, w - 28, { size: 12.5, lh: 17, color: C.ink, weight: 600 });
        wrap(g, notes[S.mode], 14, by + 7 + nl1 * 17, w - 28, { size: 12, lh: 16, color: S.mode === 'BRK' ? C.d : C.muted });
        if (S.mode === 'BRK') {
          const cut = (x, y) => { V.line(g, x, y - 9, x, y + opH + 9, { color: C.d, lw: 1.6, dash: [3, 3] }); V.text(g, '✂', x, y - 11, { size: 14, color: C.d, align: 'center', weight: 700 }); };
          Z.seg.forEach((sg, i) => { if (!isAttn(i)) return; const p = pos[i], rev = p.r % 2 === 1, half = gx / 2; cut(rev ? p.x + opW + half : p.x - half, p.y); cut(rev ? p.x - half : p.x + opW + half, p.y); });
        }
      });
    }

    /* =====================================================================
       FIG 6 — torch.compile inside out (stepper)
       ===================================================================== */
    {
      const fig = q('#fig-compile');
      const code = ['h = rms_norm(x)', 'q, k, v = qkv_proj(h).split(…)', 'a = attention(q, k, v)   # custom op', 'x = x + o_proj(a)', 'h = rms_norm(x)', 'x = x + moe(h)           # custom op'];
      const N = [
        { t: 'rms_norm', sh: '[s0, d]', k: 'ew', line: 0 },
        { t: 'qkv_proj', sh: '[s0, 3d]', k: 'mm', line: 1 },
        { t: 'attention', sh: '[s0, d]', k: 'attn', line: 2 },
        { t: 'o_proj', sh: '[s0, d]', k: 'mm', line: 3 },
        { t: 'add', sh: '[s0, d]', k: 'ew', line: 3 },
        { t: 'rms_norm', sh: '[s0, d]', k: 'ew', line: 4 },
        { t: 'moe', sh: '[s0, d]', k: 'cop', line: 5 },
        { t: 'add', sh: '[s0, d]', k: 'ew', line: 5 },
      ];
      const steps = [
        { t: 'Python eager', d: 'Without compilation, each line of <code>forward</code> turns into kernel launches the moment Python passes through it. In this toy layer: <b>10 launches</b> (the MoE alone launches 3).' },
        { t: 'Dynamo records the FX graph', d: 'Dynamo reads the bytecode and records the tensor operations into an FX graph with <b>symbolic</b> shapes (\\(s_0\\) = tokens in the step) and <i>guards</i> that say when the graph remains valid.' },
        { t: 'Custom ops become black boxes', d: 'Attention and the MoE are custom ops: the compiler doesn\'t go inside them. The <b>fake impl</b> reports the output shape; <code>mutates_args</code> declares what they write in place.' },
        { t: 'Inductor fuses', d: 'Neighboring elementwise operations — the residual add and the norm — become a single Triton kernel that reads memory once. <b>10 → 9 launches</b> in this layer.' },
        { t: 'Split at attention (piecewise)', d: 'vLLM splits the graph at the <code>splitting_ops</code> (attention). Each piece is captured as a CUDA graph <b>for every capture size</b>; attention runs eager between them — or inside the FULL graph, in uniform decode.' },
        { t: 'On-disk cache', d: 'The result goes to <code>VLLM_CACHE_ROOT/torch_compile_cache/{hash}/{rank}</code>. The next boot reuses it; if the code changes (the DS ablation), the hash or the stamp changes and everything recompiles.' },
      ];
      const st = UI.stepper(fig, steps, { autoplay: 6500 });
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const i = st.i, el = st.elapsed();
        const wide = w >= 680;
        const codeX = 16, codeY = 20, codeW = wide ? Math.min(310, w * 0.4) : w - 32;
        const lh = 17;
        // code panel
        V.box(g, codeX, codeY, codeW, code.length * lh + 18, { r: 10, fill: V.alpha(C.muted, 0.07), stroke: V.alpha(C.muted, 0.3), lw: 1 });
        const curLine = i === 0 ? Math.floor(el / 0.55) % (code.length + 2) : -1;
        code.forEach((ln, k) => {
          const y = codeY + 14 + k * lh;
          if (k === curLine) V.box(g, codeX + 4, y - 11, codeW - 8, lh, { r: 4, fill: V.alpha(C.d, 0.25) });
          V.text(g, ln, codeX + 10, y + 1, { size: 11, font: 'mono', color: ln.includes('#') ? C.c : C.ink, maxW: codeW - 16 });
        });
        // graph
        const gx0 = wide ? codeX + codeW + 30 : 16, gx1 = w - 16;
        const gy0 = wide ? 30 : codeY + code.length * lh + 44;
        const per = 4, nw = Math.min(118, (gx1 - gx0 - (per - 1) * 14) / per), nh = 46, vgap = wide ? 62 : 50;
        autoH(fig, Math.max(codeY + code.length * lh + 30, gy0 + 2 * nh + vgap + 64) + 26);
        const P = (k) => { const r = Math.floor(k / per), c = r % 2 === 0 ? k % per : per - 1 - (k % per); return { x: gx0 + c * (nw + 14), y: gy0 + r * (nh + vgap) }; };
        const show = i === 0 ? 1 : V.clamp(i === 1 ? el / 2 : 1, 0, 1);
        // arrows
        const fused = i >= 3;
        for (let k = 0; k < N.length - 1; k++) {
          if (i === 1 && (k + 1) / N.length > show) break;
          if (fused && k === 4) continue;
          const a = P(k), b = P(k + 1);
          if (Math.abs(a.y - b.y) < 1) { const dir = b.x > a.x ? 1 : -1; V.arrow(g, dir > 0 ? a.x + nw + 1 : a.x - 1, a.y + nh / 2, dir > 0 ? b.x - 2 : b.x + nw + 2, b.y + nh / 2, { color: V.alpha(C.muted, 0.6), head: 6, lw: 1.2 }); }
          else V.arrow(g, a.x + nw / 2, a.y + nh + 16, b.x + nw / 2, b.y - 2, { color: V.alpha(C.muted, 0.6), head: 6, lw: 1.2 });
        }
        // fusion (step 3+): add+norm (4,5) become a single node
        N.forEach((n, k) => {
          if (i === 1 && (k + 1) / N.length > show + 0.001) return;
          if (fused && (k === 4 || k === 5)) return;
          const p = P(k);
          const black = i >= 2 && (n.k === 'attn' || n.k === 'cop');
          let fill = V.alpha(n.k === 'attn' ? C.c : n.k === 'mm' ? C.b : C.a, 0.22), stroke = V.alpha(n.k === 'attn' ? C.c : n.k === 'mm' ? C.b : C.a, 0.8);
          if (black) { fill = C.dark ? 'rgba(0,0,0,0.55)' : 'rgba(30,30,40,0.85)'; stroke = n.k === 'attn' ? C.c : C.f; }
          if (fused && (k === 4 || k === 5)) { fill = V.alpha(C.e, 0.3); stroke = C.e; }
          if (i === 0 && n.line === curLine) fill = V.alpha(C.d, 0.55);
          V.box(g, p.x, p.y, nw, nh, { r: 9, fill, stroke, lw: 1.4 });
          const tc = black ? '#ffffff' : C.ink;
          V.text(g, fused && k === 4 ? 'add+norm' : fused && k === 5 ? '(fused)' : n.t, p.x + nw / 2, p.y + 18, { size: 12, weight: 700, align: 'center', color: tc, maxW: nw - 6 });
          if (i >= 1) V.text(g, black ? 'fake → ' + n.sh : n.sh, p.x + nw / 2, p.y + 35, { size: 11, font: 'mono', align: 'center', color: black ? '#dddddd' : C.muted, maxW: nw - 6 });
          else V.text(g, n.k === 'cop' ? '3 kernels' : '1 kernel', p.x + nw / 2, p.y + 35, { size: 11, align: 'center', color: C.muted });
          if (i === 2 && n.k === 'attn') pill(g, p.x + nw / 2, p.y + nh + 10, 'mutates: output', C.c, { align: 'center', size: 11 });
        });
        if (fused) {
          const a = P(4), b = P(5); const fx0 = Math.min(a.x, b.x), fx1 = Math.max(a.x, b.x) + nw;
          const grow = i === 3 ? V.ease.out(Math.min(1, el / 0.9)) : 1;
          const cx = (fx0 + fx1) / 2, half = V.lerp(nw / 2, (fx1 - fx0) / 2, grow);
          V.box(g, cx - half, a.y, half * 2, nh, { r: 9, fill: V.alpha(C.e, 0.3), stroke: C.e, lw: 1.6 });
          V.text(g, 'add + rms_norm', cx, a.y + 18, { size: 12, weight: 700, align: 'center', color: C.ink, maxW: half * 2 - 8 });
          V.text(g, '1 fused Triton kernel', cx, a.y + 35, { size: 11, align: 'center', color: C.muted, maxW: half * 2 - 8 });
        }
        if (i === 1) {
          const gb = { x: gx0, y: P(4).y + nh + 22 };
          if (show >= 1) pill(g, gb.x, gb.y, 'guards: dtype=bf16 · device=cuda · s0 dynamic', C.b, { size: 11 });
        }
        if (i === 4) {
          const at = P(2);
          [at.x - 7, at.x + nw + 7].forEach((x) => V.line(g, x, at.y - 14, x, at.y + nh + 14, { color: C.c, lw: 2, dash: [5, 4], dashOffset: -t * 20 }));
          const pieces = [[0, 1], [7, 6, 5, 4, 3]];
          pieces.forEach((pc, j) => {
            const p0 = P(pc[0]);
            for (let s2 = 2; s2 >= 0; s2--) V.box(g, p0.x - 3 + s2 * 4, p0.y + nh + 6 + s2 * 3, 40, 12, { r: 3, fill: V.alpha(C.b, 0.25 + s2 * 0.2), stroke: C.b, lw: 1 });
            V.text(g, `piece ${j + 1} × {1, 2, 4, …}`, p0.x + 52, p0.y + nh + 18, { size: 11, font: 'mono', color: C.b, maxW: Math.max(60, gx1 - p0.x - 56) });
          });
        }
        if (i === 5) {
          const bx = gx0, by = P(7).y + nh + (wide ? 22 : 16);
          const hash = ['3f9a', 'c21e', '7b04'][Math.floor(el / 1.6) % 3];
          pill(g, bx, by, `hash(code + config) = ${hash}…`, C.d, { size: 11, font: 'mono' });
          V.text(g, `torch_compile_cache/${hash}…/rank_0`, bx, by + 22, { size: 11, font: 'mono', color: C.muted, maxW: gx1 - bx });
        }
        const kc = i === 0 ? '10 launches per layer' : i >= 3 ? (i >= 4 ? '9 kernels · 2 graph pieces + attention' : '9 kernels after fusion') : 'FX graph: 8 nodes';
        V.text(g, kc, 16, h - 12, { size: 12, color: C.ink, weight: 600, maxW: w - 32 });
      });
    }

    /* =====================================================================
       FIG 7 — The .tolist() bubble
       ===================================================================== */
    {
      const fig = q('#fig-sync');
      const S = { path: 'sync', cpu: 8 };
      let M = null;
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'MoE path', options: [{ v: 'sync', label: 'loop with .tolist() (legacy)' }, { v: 'dev', label: 'on-device tables (E3)' }], value: 'sync', onChange: (v) => { S.path = v; M = null; } });
      UI.slider(ctl, { label: 'CPU/kernel', min: 3, max: 20, step: 1, value: S.cpu, fmt: (v) => v + ' µs', onInput: (v) => { S.cpu = v; M = null; } });
      const rT = UI.readout(ctl, 'step', '—');
      const rO = UI.readout(ctl, 'other path', '—');
      const simulate = (path) => {
        const cpu = [], gpu = []; let c = 0, gEnd = 0, busy = 0;
        const L = (n, d, cm = 1) => { const c0 = c; c += S.cpu * cm; const s0 = Math.max(c, gEnd); gpu.push({ n, s: s0, e: s0 + d, gap: s0 - gEnd }); cpu.push({ k: 'L', c0, c1: c }); gEnd = s0 + d; busy += d; };
        const W = () => { const c0 = c; c = Math.max(c, gEnd) + 3; cpu.push({ k: 'W', c0, c1: c }); };
        const Py = (d) => { const c0 = c; c += d; cpu.push({ k: 'P', c0, c1: c }); };
        for (let l = 0; l < 4; l++) {
          L('norm', 3); L('qkv', 12); L('attn', 10); L('o', 9); L('AR', 10); L('norm', 3); L('router', 5);
          if (path === 'sync') { L('count', 3); W(); Py(14); for (let e = 0; e < 3; e++) L('exp', 7, 1.6); L('comb', 3); }
          else { L('gather', 5); L('gate/up', 14); L('down', 12); }
          L('AR', 10);
        }
        return { cpu, gpu, T: gEnd, busy };
      };
      const compute = () => { M = simulate(S.path); const O = simulate(S.path === 'sync' ? 'dev' : 'sync'); M.To = O.T; rT.set(fmtUs(M.T)); rO.set(fmtUs(O.T)); };
      const topk = [[1, 4], [4, 0], [1, 2], [4, 5], [1, 4], [3, 1]];
      const counts = [0, 0, 0, 0, 0, 0]; topk.forEach((r) => r.forEach((e) => counts[e]++));
      const cap = 2;
      const rowOff = [0]; counts.forEach((c) => rowOff.push(rowOff[rowOff.length - 1] + c));
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        if (!M) compute();
        const sm = w < 560;
        const cs = sm ? 17 : 22, mH = 6 * (cs + 2) - 2;
        const panelH = sm ? mH + 136 : 34 + mH + 26;
        autoH(fig, 150 + panelH + 12);
        const x0 = sm ? 46 : 60, x1 = w - 14;
        const Tmax = Math.max(M.T, M.To) * 1.02;
        const X = (u) => x0 + (u / Tmax) * (x1 - x0);
        const P = 6, tt = Math.min(1, (t % P) / (P - 1.4)) * Tmax;
        const rh = 22, yC = 34, yG = 66;
        V.text(g, S.path === 'sync' ? 'one sync per MoE layer: the queue drains 4 times' : 'no sync: the CPU stays ahead, the queue never drains', 14, 18, { size: 11.5, font: 'mono', color: C.muted, maxW: sm ? w - 28 : w - 300 });
        if (!sm) {
          const lg = [[C.d, 'launch'], [C.f, 'Python'], [C.muted, 'wait (sync)']];
          let lx = w - 280; lg.forEach(([col, lab]) => { V.box(g, lx, 9, 10, 10, { r: 2, fill: V.alpha(col, 0.75) }); V.text(g, lab, lx + 14, 18, { size: 11, color: C.muted }); lx += tw(g, lab, 11) + 30; });
        }
        V.text(g, 'CPU', 14, yC + rh / 2 + 4, { size: 12, weight: 600, color: C.ink }); V.text(g, 'GPU', 14, yG + rh / 2 + 4, { size: 12, weight: 600, color: C.ink });
        M.cpu.forEach((bk) => {
          const a = X(bk.c0), z = X(bk.c1), done = tt >= bk.c1;
          if (bk.k === 'W') { V.box(g, a, yC, Math.max(1, z - a), rh, { r: 3, fill: V.alpha(C.muted, done ? 0.45 : 0.15), stroke: V.alpha(C.muted, 0.6), lw: 1 }); if (z - a > 18) V.text(g, '⏳', (a + z) / 2, yC + rh / 2 + 1, { size: 12, align: 'center', base: 'middle', color: C.ink }); }
          else if (bk.k === 'P') V.box(g, a + 0.5, yC, Math.max(1, z - a - 1), rh, { r: 3, fill: V.alpha(C.f, done ? 0.7 : 0.15), stroke: V.alpha(C.f, 0.5), lw: 1 });
          else V.box(g, a + 0.5, yC, Math.max(1, z - a - 1), rh, { r: 3, fill: V.alpha(C.d, done ? 0.7 : 0.14) });
        });
        M.gpu.forEach((k) => { if (k.gap > 0.3) hatch(g, X(k.s - k.gap), yG, X(k.s) - X(k.s - k.gap), rh, V.alpha(C.red, tt >= k.s ? 0.8 : 0.25)); });
        M.gpu.forEach((k) => { const a = X(k.s), z = X(k.e), done = tt >= k.e; V.box(g, a + 0.4, yG, Math.max(1, z - a - 0.8), rh, { r: 3, fill: V.alpha(C.a, done ? 0.82 : 0.14), stroke: done ? null : V.alpha(C.a, 0.4), lw: 1 }); });
        V.line(g, X(tt), yC - 5, X(tt), yG + rh + 4, { color: V.alpha(C.ink, 0.5), lw: 1.2 });
        const idle = 1 - M.busy / M.T;
        V.text(g, `step: ${fmtUs(M.T)} · GPU idle ${(idle * 100).toFixed(0)}%`, x0, yG + rh + 22, { size: 12.5, color: C.ink, weight: 600, maxW: x1 - x0 });
        V.text(g, `other path: ${fmtUs(M.To)} · illustrative durations`, x0, yG + rh + 40, { size: 11.5, color: C.muted, maxW: x1 - x0 });
        // ---- routing panel
        const py = 150;
        V.box(g, 12, py, w - 24, panelH, { r: 12, fill: V.alpha(C.b, 0.04), stroke: V.alpha(C.b, 0.3), lw: 1 });
        V.text(g, 'routing 6 tokens (top-2) across 6 experts', 24, py + 18, { size: 11.5, font: 'mono', color: C.muted, maxW: w - 48 });
        const mx = 30, my = py + 40;
        V.text(g, 'topk_ids', mx, my - 8, { size: 11, font: 'mono', color: C.muted });
        const m = mat(g, { x: mx, y: my, data: topk, cell: cs, gap: 2, values: true, min: 0, max: 5, cmap: 'heat', fmt: (v) => v.toFixed(0), rowLabels: sm ? null : ['t0', 't1', 't2', 't3', 't4', 't5'] });
        const gapA = sm ? 40 : 96;
        const bx = mx + m.w + gapA, bw = sm ? 17 : 26, bmaxH = Math.min(84, mH - 26);
        const byTop = my + (mH - bmaxH - 16) / 2;
        V.arrow(g, mx + m.w + 6, my + mH / 2, bx - 8, my + mH / 2, { color: C.a, head: 7, lw: 1.6 });
        if (!sm) V.text(g, 'scatter_add_', mx + m.w + gapA / 2 - 1, my + mH / 2 - 10, { size: 11, font: 'mono', align: 'center', color: C.a });
        V.text(g, 'counts [E=6] · GPU', bx, byTop - 8, { size: 11, font: 'mono', color: C.muted });
        counts.forEach((cv, e) => {
          const bh = (cv / 4) * bmaxH, x = bx + e * (bw + 4), y = byTop + 6 + bmaxH - bh;
          const fat = cv > cap;
          V.box(g, x, y, bw, Math.max(2, bh), { r: 3, fill: V.alpha(fat ? C.d : C.a, 0.75) });
          V.text(g, cv, x + bw / 2, y - 4, { size: 11, font: 'mono', align: 'center', color: C.ink });
          V.text(g, 'e' + e, x + bw / 2, byTop + 6 + bmaxH + 14, { size: 11, font: 'mono', align: 'center', color: C.muted });
        });
        const capY = byTop + 6 + bmaxH - (cap / 4) * bmaxH;
        V.line(g, bx - 3, capY, bx + 6 * (bw + 4), capY, { color: V.alpha(C.d, 0.9), dash: [3, 3], lw: 1 });
        if (!sm) V.text(g, 'cap', bx + 6 * (bw + 4) + 2, capY + 4, { size: 11, font: 'mono', color: C.d });
        const rx = bx + 6 * (bw + 4) + 40;
        const wide = !sm && w - 24 - rx > 190;
        const tx = wide ? rx : 30, ty = wide ? my + 18 : my + mH + 22;
        const tW = w - tx - 24;
        if (S.path === 'sync') {
          V.text(g, 'nonzero(counts > cap) → [1, 4]', tx, ty, { size: 11.5, font: 'mono', color: C.d, weight: 600, maxW: tW });
          V.text(g, '.tolist() → CPU ⏳', tx, ty + 20, { size: 11.5, color: C.red, weight: 700, maxW: tW });
          V.text(g, 'the size depends on the data', tx, ty + 38, { size: 11, color: C.muted, maxW: tW });
          V.text(g, 'then: Python loop, 1 launch per expert', tx, ty + 56, { size: 11, color: C.muted, maxW: tW });
        } else {
          V.text(g, 'row_off = cumsum → [' + rowOff.join(', ') + ']', tx, ty, { size: 11.5, font: 'mono', color: C.a, weight: 600, maxW: tW });
          V.text(g, 'fixed shape [E+1], stays on the GPU ✓', tx, ty + 20, { size: 11.5, color: C.a, weight: 700, maxW: tW });
          V.text(g, 'searchsorted → row segments', tx, ty + 38, { size: 11, color: C.muted, maxW: tW });
          V.text(g, '3 launches cover every expert', tx, ty + 56, { size: 11, color: C.muted, maxW: tW });
        }
      });
    }

    /* =====================================================================
       FIG 8 — JIT buckets and the warmup ladder
       ===================================================================== */
    {
      const fig = q('#fig-buckets');
      const PR = {
        glm: { lab: 'GLM (k=7)', q: 8, ladder: [1, 24, 56, 120, 248] },
        ds6: { lab: 'DS documented (+6)', q: 6, ladder: [1, 6, 20, 45, 100, 200] },
        ds7: { lab: 'DS with k=6 (+7)', q: 7, ladder: [1, 6, 20, 45, 100, 200] },
        mine: { lab: 'your recipe', q: 4, ladder: null },
      };
      const S = { p: 'glm', k: 3, warm: true, parts: [], late: new Set(), jit: 0, flashes: [] };
      const BK = [8, 16, 32, 64, 128, 256];
      const key = (s, qq) => Math.min(256, nextPow2(s + qq));
      let cfg = null;
      const rng = V.rng(29);
      function setup() {
        const p = PR[S.p];
        const qq = S.p === 'mine' ? S.k + 1 : p.q;
        const reach = BK.filter((B) => B >= nextPow2(1 + qq));
        const ladder = S.p === 'mine' ? reach.map((B) => Math.max(1, B - qq)) : p.ladder;
        const warmed = new Set(S.warm ? ladder.map((s) => key(s, qq)) : []);
        cfg = { qq, reach, ladder, warmed };
        S.late = new Set(); S.jit = 0; S.parts = []; S.flashes = [];
        rR.set(String(reach.length)); rW.set(`${Array.from(warmed).filter((b) => reach.includes(b)).length}/${reach.length}`); rJ.set('0');
      }
      const ctl = UI.controls(fig);
      const segP = UI.seg(ctl, { label: 'Ladder', options: Object.keys(PR).map((k) => ({ v: k, label: PR[k].lab })), value: 'glm', onChange: (v) => { S.p = v; setup(); } });
      UI.slider(ctl, { label: 'k (your recipe)', min: 0, max: 8, step: 1, value: S.k, onInput: (v) => { S.k = v; if (S.p !== 'mine') segP.set('mine'); else setup(); } });
      UI.toggle(ctl, { label: 'run the warmup ladder at boot', value: true, onChange: (v) => { S.warm = v; setup(); } });
      const rR = UI.readout(ctl, 'reachable buckets', '—');
      const rW = UI.readout(ctl, 'warmed at boot', '—');
      const rJ = UI.readout(ctl, 'JIT under traffic', '0');
      setup();
      let lastSpawn = 0;
      V.fig(fig, (g, w, h, t, dt) => {
        const C = V.C();
        const sm = w < 600;
        const chipsW = sm ? 0 : 150;
        const x0 = 52, x1 = w - 16 - chipsW - (sm ? 0 : 16), y0 = 34, y1 = h - (sm ? 142 : 46);
        const smax = 300;
        const X = (s) => x0 + Math.sqrt(s / smax) * (x1 - x0);
        const Y = (B) => y1 - ((Math.log2(B) - 3) / 5) * (y1 - y0);
        V.text(g, `BLOCK = min(256, next_pow2(s + ${cfg.qq}))`, x0, 18, { size: 12, font: 'mono', color: C.ink, weight: 600, maxW: x1 - x0 });
        BK.forEach((B) => {
          const y = Y(B);
          V.line(g, x0, y, x1, y, { color: V.alpha(C.muted, 0.14), lw: 1 });
          V.text(g, B, x0 - 8, y + 4, { size: 11, font: 'mono', align: 'right', color: C.muted });
        });
        V.line(g, x0, y1 + 10, x1, y1 + 10, { color: V.alpha(C.muted, 0.5), lw: 1 });
        (sm ? [1, 10, 50, 100, 200, 300] : [1, 5, 10, 25, 50, 100, 150, 200, 250, 300]).forEach((s) => { V.line(g, X(s), y1 + 10, X(s), y1 + 14, { color: C.muted }); V.text(g, s, X(s), y1 + 26, { size: 11, font: 'mono', align: 'center', color: C.muted }); });
        V.text(g, 's = scheduled tokens (axis in √s)', x1, y1 - 6, { size: 11, align: 'right', color: C.muted });
        // staircase
        g.beginPath(); g.strokeStyle = C.a; g.lineWidth = 2.2;
        let prevB = key(1, cfg.qq); g.moveTo(X(1), Y(prevB));
        for (let s = 2; s <= smax; s++) { const B = key(s, cfg.qq); if (B !== prevB) { g.lineTo(X(s), Y(prevB)); g.lineTo(X(s), Y(B)); prevB = B; } }
        g.lineTo(X(smax), Y(prevB)); g.stroke();
        // rungs of the warmup ladder
        cfg.ladder.forEach((s, i) => {
          const B = key(s, cfg.qq), x = X(s), y = Y(B);
          V.glow(g, x, y, 12, S.warm ? C.e : C.muted, 0.7);
          V.dot(g, x, y, 5, S.warm ? C.e : C.muted);
          const lab = 's=' + s, right = x + 7 + tw(g, lab, 11, 'mono') > w - 4, below = y - 7 < y0 + 6;
          V.text(g, lab, right ? x - 7 : x + 7, below ? y + 18 : y - 7, { size: 11, font: 'mono', color: C.ink, align: right ? 'right' : 'left' });
        });
        // traffic
        if (!LA.paused && t - lastSpawn > 0.32) {
          lastSpawn = t;
          const s = Math.max(1, Math.floor(Math.exp(rng() * Math.log(smax))));
          S.parts.push({ s, B: key(s, cfg.qq), u: 0 });
        }
        for (let k = S.parts.length - 1; k >= 0; k--) {
          const p = S.parts[k]; p.u += dt * 1.4;
          const yT = Y(p.B), yy = V.lerp(y0 - 6, yT, V.ease.in(Math.min(1, p.u)));
          V.dot(g, X(p.s), yy, 3, V.alpha(C.b, 0.9));
          if (p.u >= 1) {
            S.parts.splice(k, 1);
            if (!cfg.warmed.has(p.B) && !S.late.has(p.B)) { S.late.add(p.B); S.jit++; rJ.set(String(S.jit)); S.flashes.push({ x: X(p.s), y: yT, t: 0 }); }
          }
        }
        for (let k = S.flashes.length - 1; k >= 0; k--) {
          const f = S.flashes[k]; f.t += dt; if (f.t > 1.6) { S.flashes.splice(k, 1); continue; }
          V.glow(g, f.x, f.y, 26 * (1 + f.t), C.red, 1 - f.t / 1.6);
          V.text(g, 'JIT!', f.x, f.y - 14 - f.t * 10, { size: 12, weight: 700, align: 'center', color: C.red, alpha: 1 - f.t / 1.6 });
        }
        // bucket chips
        const chip = (B, x, y, cw) => {
          const reach = cfg.reach.includes(B), warm = cfg.warmed.has(B) && reach, late = S.late.has(B);
          const col = !reach ? C.muted : warm ? C.a : late ? C.c : C.d;
          const lab = !reach ? 'unreachable' : warm ? 'warmed' : late ? 'late JIT' : 'cold';
          if (!reach) g.setLineDash([4, 3]);
          V.box(g, x, y, cw, 24, { r: 7, fill: V.alpha(col, reach ? 0.2 : 0.05), stroke: V.alpha(col, 0.8), lw: 1.2 });
          g.setLineDash([]);
          V.text(g, 'BLOCK ' + B, x + 8, y + 16, { size: 11, font: 'mono', color: C.ink, weight: 600 });
          if (cw > 110) V.text(g, lab, x + cw - 8, y + 16, { size: 11, align: 'right', color: col, weight: 600 });
        };
        if (!sm) {
          const cx = w - 16 - chipsW;
          BK.slice().reverse().forEach((B, i) => chip(B, cx, y0 + i * 32, chipsW));
        } else {
          const cw = (w - 32 - 8) / 2;
          BK.forEach((B, i) => chip(B, 16 + (i % 2) * (cw + 8), h - 96 + Math.floor(i / 2) * 30, cw));
        }
      });
    }

    /* =====================================================================
       FIG 9 — Two ranks, one collective, three clocks
       ===================================================================== */
    {
      const fig = q('#fig-ranks');
      const S = { sc: 'jit', tc: 450, long: true, fix: false };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Scenario', options: [{ v: 'jit', label: 'JIT mid-serving' }, { v: 'graphs', label: 'different graphs' }, { v: 'tune', label: 'divergent autotune' }], value: 'jit', onChange: (v) => { S.sc = v; } });
      UI.slider(ctl, { label: 'Compilation', min: 30, max: 1500, step: 10, value: S.tc, fmt: (v) => v + ' s', onInput: (v) => { S.tc = v; } });
      UI.toggle(ctl, { label: 'RPC deadline = 1800 s (recipes)', value: true, onChange: (v) => { S.long = v; } });
      UI.toggle(ctl, { label: 'recipes\' remedy', value: false, onChange: (v) => { S.fix = v; } });
      const rOut = UI.readout(ctl, 'outcome', '—');
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const sm = w < 560;
        const x0 = sm ? 56 : 76, x1 = w - 16;
        const Tax = 2000, t0 = 80;
        const X = (s) => x0 + (Math.min(Tax, s) / Tax) * (x1 - x0);
        const y0 = 46, y1 = 112, rh = 30;
        const rpc = S.long ? 1800 : 300, wd = 600;
        // outcome
        let outcome, color, stopAt, hang = false, wait = 0;
        if (S.fix) { outcome = S.sc === 'jit' ? 'shape already cached: no JIT under traffic' : S.sc === 'graphs' ? 'same environment: same graphs, collectives match' : 'consistent autotune: joint tuning or rank-0 broadcast'; color = C.a; stopAt = null; }
        else if (S.sc === 'jit') {
          wait = S.tc;
          if (S.tc < Math.min(rpc, wd)) { outcome = `survives: latency spike of ${S.tc} s`; color = C.d; stopAt = null; }
          else if (rpc < wd) { outcome = 'EngineCore declares a hang at 300 s (RPC)'; color = C.red; stopAt = t0 + rpc; }
          else { outcome = 'NCCL watchdog kills the pair at 600 s'; color = C.red; stopAt = t0 + wd; }
        } else { hang = true; outcome = 'collectives mismatch: stuck until watchdog (600 s)'; color = C.red; stopAt = t0 + wd; }
        rOut.set(S.fix ? 'ok' : color === C.red ? 'process killed' : 'survives');
        // deadlines
        const mark = (s, lab, col, yy) => { const x = X(s); V.line(g, x, y0 - 16, x, y1 + rh + 6, { color: col, lw: 1.5, dash: [5, 4] }); const lwid = tw(g, lab, 11, 'mono'); const right = x + 4 + lwid > x1; V.text(g, lab, right ? x - 4 : x + 4, yy, { size: 11, font: 'mono', color: col, align: right ? 'right' : 'left' }); };
        mark(300, sm ? 'RPC 300' : 'RPC stock 300 s', V.alpha(C.d, S.long ? 0.45 : 1), y0 - 20);
        mark(wd, sm ? 'NCCL 600' : 'NCCL watchdog 600 s', C.red, y0 - 6);
        mark(1800, sm ? '1800' : 'RPC recipes 1800 s', V.alpha(C.d, S.long ? 1 : 0.45), y0 - 20);
        // rank lanes
        const lane = (y, lab) => { V.text(g, lab, 14, y + rh / 2 + 4, { size: 12, weight: 600, color: C.ink }); V.box(g, x0, y, x1 - x0, rh, { r: 6, fill: V.alpha(C.muted, 0.05), stroke: V.alpha(C.muted, 0.25), lw: 1 }); };
        lane(y0, 'rank 0'); lane(y1, 'rank 1');
        const end = stopAt || Tax;
        const steps = (y, a, b) => { for (let x = X(a); x < X(b) - 2; x += 5) V.box(g, x, y + 7, 3, rh - 14, { r: 1, fill: V.alpha(C.a, 0.6) }); };
        steps(y0, 0, t0); steps(y1, 0, t0);
        const wEnd = S.fix ? t0 : hang ? end : Math.min(t0 + wait, end);
        if (!S.fix) {
          if (S.sc === 'jit') {
            V.box(g, X(t0), y1 + 2, X(Math.min(t0 + S.tc, end)) - X(t0), rh - 4, { r: 5, fill: V.alpha(C.d, 0.45), stroke: C.d, lw: 1.2 });
            const jw = X(Math.min(t0 + S.tc, end)) - X(t0) - 8;
            const jl = tw(g, 'Triton/TileLang JIT', 11, 'sans', 600) < jw ? 'Triton/TileLang JIT' : 'JIT';
            if (tw(g, jl, 11, 'sans', 600) < jw + 4) V.text(g, jl, X(t0) + 6, y1 + rh / 2 + 4, { size: 11, color: C.ink, weight: 600 });
            hatch(g, X(t0), y0 + 2, X(wEnd) - X(t0), rh - 4, V.alpha(C.red, 0.55), 6);
            const ww2 = X(wEnd) - X(t0) - 8;
            const wl = tw(g, 'waiting in all-reduce', 11) < ww2 ? 'waiting in all-reduce' : 'waiting';
            if (tw(g, wl, 11) < ww2 + 4) V.text(g, wl, X(t0) + 6, y0 + rh / 2 + 4, { size: 11, color: C.ink });
          } else {
            const la = S.sc === 'graphs' ? (sm ? ['graph 8 tok · AR [8×d]', 'eager 5 tok · AR [5×d]'] : ['replay graph 8 tok · all-reduce [8×d]', 'eager 5 tok · all-reduce [5×d]']) : (sm ? ['cache hit → model AR', 'cache miss → autotune'] : ['cache hit: skips autotune → model all-reduce', 'cache miss: autotune collective']);
            [y0, y1].forEach((y, j) => { hatch(g, X(t0), y + 2, X(end) - X(t0), rh - 4, V.alpha(C.red, 0.5), 6); V.text(g, la[j], X(t0) + 6, y + rh / 2 + 4, { size: 11, color: C.ink, weight: 600, maxW: X(end) - X(t0) - 10 }); });
            const pulse = 0.5 + 0.5 * Math.sin(t * 5);
            pill(g, (X(end) + 24 + x1) / 2, (y0 + rh + y1) / 2, sm ? '≠ collectives' : '≠ mismatched collectives', C.red, { align: 'center', size: 11, a: 0.12 + 0.1 * pulse });
          }
          if (!stopAt || stopAt < Tax) {
            const resume = stopAt ? null : t0 + wait;
            if (resume) { steps(y0, resume, Tax); steps(y1, resume, Tax); }
          }
          if (stopAt) { [y0, y1].forEach((y) => { V.text(g, '✕', X(stopAt) + 4, y + rh / 2 + 6, { size: 16, weight: 700, color: C.red }); }); }
        } else { steps(y0, t0, Tax); steps(y1, t0, Tax); }
        // synchronizations (barriers)
        V.line(g, X(t0), y0 + rh, X(t0), y1, { color: V.alpha(C.b, 0.9), lw: 2 });
        V.text(g, 'all-reduce', X(t0) + 4, y0 + rh + 14, { size: 11, font: 'mono', color: C.b });
        // axis
        const ay = y1 + rh + 20;
        V.line(g, x0, ay, x1, ay, { color: V.alpha(C.muted, 0.5), lw: 1 });
        [0, 500, 1000, 1500, 2000].forEach((s2) => { V.line(g, X(s2), ay, X(s2), ay + 4, { color: C.muted }); V.text(g, s2 + ' s', X(s2), ay + 17, { size: 11, font: 'mono', align: s2 === 2000 ? 'right' : s2 === 0 ? 'left' : 'center', color: C.muted }); });
        V.text(g, outcome, 14, ay + 46, { size: 13, weight: 700, color, maxW: w - 28 });
        const hint = S.sc === 'jit' ? (S.fix ? 'persistent caches + warmup: compilation happens at boot, once per image' : `rank 1 compiles for ${S.tc} s; rank 0 waits in the all-reduce`) : S.sc === 'graphs' ? (S.fix ? 'GLM: identical GLM53_*/EXL3_* on every rank' : 'one rank with a different capture list pads to 8, the other runs 5') : (S.fix ? 'Qwen v0.30: /tmp in each container; DS (Stage-C overlay, lane B): rank 0 tunes and broadcasts' : 'Qwen v0.30: the persisted per-node cache differed between nodes (hit/miss above: interpretation of the mechanism)');
        const nl = wrap(g, hint, 14, ay + 68, w - 28, { size: 12, color: C.muted });
        wrap(g, 'scale in seconds · normal steps last milliseconds (tick marks)', 14, ay + 70 + nl * 17, w - 28, { size: 11, color: C.muted });
      });
    }

    /* =====================================================================
       FIG 10 — The timed boot
       ===================================================================== */
    {
      const fig = q('#fig-boot');
      const S = { r: 'glm', pth: true, nv: true, cg: true };
      let H10 = null;
      const rd = () => { if (H10) H10.redraw(); };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Recipe', options: [{ v: 'glm', label: 'GLM' }, { v: 'qwen', label: 'Qwen' }, { v: 'ds', label: 'DS' }], value: 'glm', onChange: (v) => { S.r = v; vis(); rd(); } });
      const tg = [
        UI.toggle(ctl, { label: '.pth without eager import', value: true, onChange: (v) => { S.pth = v; rd(); } }),
        UI.toggle(ctl, { label: 'persisted NV cache', value: true, onChange: (v) => { S.nv = v; rd(); } }),
        UI.toggle(ctl, { label: 'optional: CG_ESTIMATE=0 + skip test capture', value: true, onChange: (v) => { S.cg = v; rd(); } }),
      ];
      const rH = UI.readout(ctl, 'API up', '—');
      const vis = () => tg.forEach((x) => { x.el.style.display = S.r === 'glm' ? '' : 'none'; });
      vis();
      const phases = () => {
        if (S.r === 'glm') return {
          list: [
            { n: 'container → 1st log', d: S.pth ? 6 : 100, c: 'd' },
            { n: 'Python/engine/NCCL init', d: 26, c: 'b' },
            { n: 'weight streaming', d: 39, c: 'a' },
            { n: 'KV profile run', d: S.cg ? 7 : 18, c: 'f' },
            { n: 'capture (target) + warmup', d: 11, c: 'c' },
            { n: 'DFlash2 capture (31 MiB cubin)', sn: 'DFlash2 capture', d: S.nv ? 0 : 30, c: 'c' },
            { n: 'API server', d: 10, c: 'e' },
          ],
          after: [{ n: 'shape warmup + canary', d: null }],
          note: S.pth && S.nv && S.cg ? 'measured with CG_ESTIMATE=0: /health in 99 s (before: 230 s) · restart 259 → 122 s' : S.pth && S.nv ? 'default CG_ESTIMATE=1: KV profile stays at 18 s (≈110 s)' : '“before” reconstructed by adding up the measured differences',
          extra: 'READY_TIMEOUT = 3600 s (off the scale)',
        };
        if (S.r === 'qwen') return {
          list: [
            { n: 'NCCL setup', d: 40, c: 'b' },
            { n: 'weights (target 392 s + MTP 67 s)', sn: 'weights', d: 458, c: 'a' },
            { n: 'engine init (profile + KV + warmup)', sn: 'engine init', d: 92.4, c: 'f' },
            { n: 'graph capture', d: 7, c: 'c' },
            { n: 'remainder until the API', d: 57.6, c: 'e' },
          ],
          after: [{ n: '1st request: FlashInfer autotune', d: null }],
          note: 'cold start ≈ 10 min 55 s (the “remainder” is the difference)',
          extra: 'graphs: FULL_DECODE_ONLY, 1–64 tokens, 0.39 GiB',
        };
        return {
          list: [{ n: 'load (weights, init, graphs…)', sn: 'load', d: 360, c: 'a' }],
          after: [{ n: 'smoke', d: null }, { n: 'shape warmup: 47 req (N=6), ≤ 240 s each', sn: 'warmup: 47 req, ≤ 240 s each', d: null }, { n: '1st req after restart: TTFT 6–10 s', d: null }],
          note: '“~6 min to load” (healthcheck start_period 480 s) — no documented breakdown',
          extra: 'warmup never brings down the boot: failure = WARN',
        };
      };
      H10 = V.fig(fig, (g, w, h) => {
        const C = V.C();
        const Z = phases();
        const sm = w < 560;
        const tot = Z.list.reduce((a, p) => a + p.d, 0);
        const dashW = tot * 0.14;
        const axisMax = tot + dashW * Z.after.length;
        rH.set(tot.toFixed(0) + ' s');
        const x0 = 16, x1 = w - 16;
        const X = (s) => x0 + (s / axisMax) * (x1 - x0);
        const all = Z.list.concat(Z.after);
        autoH(fig, 14 + all.length * 30 + (sm ? 112 : 96));
        const rowH = Math.min(30, (h - 110) / all.length);
        const hx0 = X(tot);
        V.line(g, hx0, 8, hx0, 14 + all.length * rowH, { color: C.a, lw: 2, dash: [5, 4] });
        let acc = 0;
        all.forEach((p, i) => {
          const y = 14 + i * rowH;
          const dashed = p.d == null;
          const a = dashed ? tot + dashW * (i - Z.list.length) : acc;
          const d = dashed ? dashW : p.d;
          const col = dashed ? C.muted : C[p.c];
          const bx = X(a), bw = Math.max(2, X(a + d) - X(a));
          if (dashed) g.setLineDash([4, 3]);
          V.box(g, bx, y + 2, bw, rowH - 10, { r: 5, fill: V.alpha(col, dashed ? 0.08 : 0.55), stroke: V.alpha(col, 0.9), lw: 1.2 });
          g.setLineDash([]);
          const nm = sm && p.sn ? p.sn : p.n;
          const lab = dashed ? nm : `${nm} · ${p.d % 1 ? p.d.toFixed(1) : p.d} s`;
          const lwid = tw(g, lab, 11.5);
          const fitR = bx + bw + 6 + lwid < x1, fitL = bx - lwid - 6 >= x0;
          const lx = fitR ? bx + bw + 6 : fitL ? bx - lwid - 6 : bx + 6;
          if (fitR || fitL) V.box(g, lx - 3, y + 1, Math.min(lwid, x1 - lx) + 6, rowH - 8, { r: 4, fill: V.alpha(C.bg || '#000', 0.85) });
          V.text(g, lab, lx, y + rowH / 2 + 1, { size: 11.5, color: p.d === 0 ? C.muted : C.ink, maxW: x1 - lx, weight: dashed ? 400 : 500 });
          if (!dashed) acc += p.d;
        });
        const hx = X(tot);
        V.text(g, S.r === 'ds' ? 'ready' : '/health', Math.min(hx + 4, x1 - 40), 14 + all.length * rowH + 4, { size: 11, font: 'mono', color: C.a });
        const ay = h - (sm ? 92 : 76);
        V.line(g, x0, ay, x1, ay, { color: V.alpha(C.muted, 0.5), lw: 1 });
        const stp = axisMax > 800 ? 200 : axisMax > 300 ? 100 : axisMax > 150 ? 50 : 25;
        for (let s = 0; s <= axisMax + 0.1; s += stp) { V.line(g, X(s), ay, X(s), ay + 4, { color: C.muted }); if (!sm || (s / stp) % 2 === 0) V.text(g, s + ' s', X(s), ay + 17, { size: 11, font: 'mono', align: 'center', color: C.muted }); }
        const nn = wrap(g, Z.note, 16, ay + 38, w - 32, { size: 12, lh: 16, color: C.ink, weight: 600 });
        V.text(g, Z.extra, 16, ay + 42 + nn * 16, { size: 11.5, color: C.muted, maxW: w - 32 });
      }, { static: true });
    }
    /* =====================================================================
       FIG 11 — TensorFold: graph shelf per width × uniform batch
       ===================================================================== */
    {
      const fig = q('#fig-tf-shelf');
      const T = {
        modes: [{ v: 'vllm', label: 'vLLM · uniform batch' }, { v: 'tfb', label: 'TensorFold · batched' }, { v: 'tfs', label: 'TensorFold · single stream' }],
        head: {
          vllm: 'vLLM · adaptive-k {2, 4, 7}: the whole batch verifies the smallest k′',
          tfb: 'TensorFold · PARALLEL=4: one graph per row count, from 1 to 32',
          tfs: 'TensorFold · PARALLEL=1: widths 1–8 and 16 (COPY_CODE=1)',
        },
        headS: { vllm: 'vLLM · uniform batch', tfb: 'TF · batched (1–32)', tfs: 'TF · single stream' },
        sub: {
          vllm: '15 graphs (“live best” profile) · no copy drafts',
          tfb: '32 batched graphs · DFlash2 up to 7, copies up to 15',
          tfs: '9 widths × 2 parities (+ buckets) · DFlash2 up to 7, copies up to 15',
        },
        lane: (i, sm) => (sm ? 's' + (i + 1) : 'stream ' + (i + 1)),
        noteV: (s, sm) => (sm ? 'k′' + s.kp : 'wants ' + s.want + ' → k′=' + s.kp),
        noteT: (s, sm) => (sm ? s.d + (s.copy ? '⧉' : '') : s.d + (s.copy ? ' · copy' : ' · DFlash2')),
        idle: 'PARALLEL=1: one stream at a time',
        win: (R) => `verified window · ${R} row${R === 1 ? '' : 's'}`,
        ruler: 'captured graphs',
        park: (gph) => (gph ? '→ graph ' + gph : 'EAGER'),
        cols: ['seg', 'row0', 'rows', 'slot', 'parity', 'position'],
        colsS: ['seg', 'row0', 'rows', 'slot', 'par.', 'pos.'],
        write: 'host → device: table rewritten in place (pinned + event)',
        writeS: 'device-side table',
        writeHome: 'single stream: graphs recorded at home; the parity picks 1 of the 2',
        replay: 'replay',
        home: 'home',
        desc1: (R) => `BatchDescriptor(num_tokens=${R}, uniform=True)`,
        desc2: (kps, k) => `FULL requires a uniform batch → k′ = min(${kps.join(', ')}) = ${k}`,
        desc3: 'attention metadata: each backend\'s persistent buffers',
        expl: {
          vllm: (P) => `Cut: ${P.cut} draft${P.cut === 1 ? '' : 's'} the policies wanted. Padding: ${P.pad} — the “live best” list matches c·(k′+1) exactly.`,
          tfb: (P) => `${P.n} stream${P.n === 1 ? '' : 's'}, ${P.R} rows → graph ${P.R}, no padding. ${P.cut ? 'Trimmed ' + P.cut + ' drafts (32-row ceiling).' : 'Nothing trimmed.'} The table tells every kernel where each stream is.`,
          tfs: (P) => (P.pad ? `A ${P.R}-row copy → filled up to 16 with ↻ (the last draft repeated, verified like the others).` : `A ${P.R}-row window → graph ${P.R}, one of the two recorded for this width (parity ${P.par}).`),
        },
        acc: (t) => `accepted this round: ${t} token${t === 1 ? '' : 's'}`,
        ctlMode: 'Engine', ctlN: 'streams', ctlRoll: '🎲 new round',
        rR: 'rows', rG: 'graph', rP: 'padding', rC: 'cut',
      };
      const VLLM_SIZES = [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 20, 24, 32];
      const ALL32 = Array.from({ length: 32 }, (_, i) => i + 1);
      const SINGLE = [1, 2, 3, 4, 5, 6, 7, 8, 16];
      const S = { mode: 'tfb', n: 3, seed: 1, t0: null, reset: false };
      const PER = 7.5;
      const memo = { key: '', v: null };
      let lastKey = '';
      const fmtPos = (n) => fmtN(n);
      function plan(round) {
        const key = S.mode + '|' + S.n + '|' + S.seed + '|' + round;
        if (memo.key === key) return memo.v;
        const r = V.rng(1 + S.seed * 7919 + round * 104729);
        const n = S.mode === 'tfs' ? 1 : S.n;
        const st = [];
        for (let i = 0; i < 4; i++) {
          const copy = r() < 0.28;
          const dfl = [1, 2, 3, 3, 4, 5, 5, 6, 7, 7][r.int(10)];
          const cp = 9 + r.int(7);
          const acc = 0.6 + r() * 0.32;
          const par = r.int(2);
          const pos = 900 + Math.floor(r() * r() * 600000);
          const u = Array.from({ length: 16 }, () => r());
          if (i < n) st.push({ i, copy, d: copy ? cp : dfl, acc, par, pos, u, cut: 0, pad: 0 });
        }
        const P = { n, st, cut: 0, pad: 0, R: 0, graph: null, sizes: ALL32, k: null, kps: [], par: st[0].par };
        if (S.mode === 'vllm') {
          st.forEach((s) => { s.copy = false; s.want = Math.min(s.d, 7); s.kp = s.want >= 7 ? 7 : s.want >= 4 ? 4 : 2; });
          P.kps = st.map((s) => s.kp);
          const k = Math.min(...P.kps); P.k = k;
          st.forEach((s) => { s.rows = k + 1; s.cut = Math.max(0, s.want - k); });
          P.R = n * (k + 1); P.graph = VLLM_SIZES.find((x) => x >= P.R) || null; P.pad = P.graph ? P.graph - P.R : 0; P.sizes = VLLM_SIZES;
        } else if (S.mode === 'tfb') {
          st.forEach((s) => { s.want = s.d; s.rows = 1 + s.d; });
          let tot = st.reduce((a, s) => a + s.rows, 0);
          while (tot > 32) { let j = 0; st.forEach((s, ii) => { if (s.rows > st[j].rows || (s.rows === st[j].rows && ii > j)) j = ii; }); st[j].rows--; st[j].cut++; tot--; }
          P.R = tot; P.graph = tot;
        } else {
          const s = st[0]; s.want = s.d; s.rows = 1 + s.d;
          P.R = s.rows; P.graph = s.rows <= 8 ? s.rows : 16; s.pad = P.graph - s.rows; P.pad = s.pad; P.sizes = SINGLE;
        }
        P.cut = st.reduce((a, s) => a + s.cut, 0);
        let start = 0;
        st.forEach((s) => {
          s.start = start; s.tot = s.rows + s.pad; start += s.tot;
          let a = 0; const nd = s.tot - 1;
          for (let j = 0; j < nd; j++) { const lim = j >= s.rows - 1 ? 0.25 : s.acc; if (s.u[j] < lim) a++; else break; }
          s.a = a; s.full = a === nd;
        });
        P.tokens = st.reduce((x, s) => x + s.a + 1, 0);
        memo.key = key; memo.v = P;
        return P;
      }
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: T.ctlMode, options: T.modes, value: S.mode, onChange: (v) => { S.mode = v; } });
      UI.slider(ctl, { label: T.ctlN, min: 1, max: 4, step: 1, value: S.n, onInput: (v) => { S.n = v; } });
      UI.button(ctl, T.ctlRoll, () => { S.seed++; S.reset = true; });
      const rR = UI.readout(ctl, T.rR, '—'), rG = UI.readout(ctl, T.rG, '—'), rP = UI.readout(ctl, T.rP, '—'), rC = UI.readout(ctl, T.rC, '—');
      let tfCol = null, tfDark = null;
      const TFC = (C) => { if (tfCol == null || tfDark !== C.dark) { tfCol = getComputedStyle(document.documentElement).getPropertyValue('--tf').trim() || C.d; tfDark = C.dark; } return tfCol; };
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const TF = TFC(C);
        const SC = [C.b, C.c, C.e, C.f];
        const sm = w < 560;
        if (S.t0 == null) S.t0 = t - 5.1;
        if (S.reset) { S.t0 = t; S.reset = false; }
        const el = Math.max(0, t - S.t0);
        const round = Math.floor(el / PER), ph = el - round * PER;
        const P = plan(round);
        const key = memo.key;
        if (key !== lastKey) {
          lastKey = key;
          rR.set(String(P.R));
          rG.set(P.graph ? String(P.graph) : 'eager', P.graph ? 'ok' : 'bad');
          rP.set(String(P.pad), P.pad && S.mode === 'vllm' ? 'warn' : null);
          rC.set(String(P.cut), P.cut ? 'warn' : null);
        }
        const pad = 14;
        const accent = S.mode === 'vllm' ? C.glm || C.b : TF;
        // header
        V.text(g, sm ? T.headS[S.mode] : T.head[S.mode], pad, 17, { size: sm ? 12 : 12.5, weight: 600, color: accent, maxW: w - 2 * pad });
        V.text(g, T.sub[S.mode], pad, 33, { size: 11, color: C.muted, maxW: w - 2 * pad });
        // proposal lanes
        const yL0 = 44, laneH = sm ? 21 : 24;
        const labW = sm ? 24 : 70, noteW = sm ? 40 : 128;
        const cs = V.clamp((w - 2 * pad - labW - noteW - 8) / 16, 6, 22);
        const ch = Math.min(cs, laneH - 7);
        const lx = (j) => pad + labW + j * cs;
        const ly = (i) => yL0 + i * laneH + (laneH - ch) / 2;
        const pProp = V.clamp(ph / 1.0, 0, 1);
        // window
        const rightLab = sm ? 0 : 118;
        const cw = Math.min(sm ? 11 : 26, (w - 2 * pad - 8 - rightLab) / 32);
        const x0 = pad + 4;
        const winH = sm ? 18 : 22;
        const yW = yL0 + 4 * laneH + 26;
        const X = (s) => x0 + s * cw;
        for (let i = 0; i < 4; i++) {
          const s = P.st[i];
          const y = yL0 + i * laneH;
          V.text(g, T.lane(i, sm), pad, y + laneH / 2 + 4, { size: 11, font: 'mono', color: s ? SC[i] : V.alpha(C.muted, 0.5), weight: 600 });
          if (!s) {
            if (S.mode === 'tfs' && i === 1) V.text(g, T.idle, lx(0), y + laneH / 2 + 4, { size: 11, color: C.muted, maxW: w - lx(0) - pad });
            continue;
          }
          const nShow = S.mode === 'vllm' ? s.want : s.d;
          for (let j = 0; j <= nShow; j++) {
            const pj = V.ease.out(V.clamp(pProp * 1.6 - j * 0.05, 0, 1));
            if (pj <= 0) continue;
            const inWin = j < s.rows;
            const pa = V.clamp((ph - 1.2 - (s.start + j) * 0.012) / 0.9, 0, 1);
            const x = lx(j), yy = ly(i);
            if (inWin && pa > 0) {
              V.box(g, x + 0.5, yy, cs - 1.5, ch, { r: 3, stroke: V.alpha(SC[i], 0.35), lw: 1 });
            } else {
              V.box(g, x + 0.5, yy, cs - 1.5, ch, { r: 3, fill: V.alpha(SC[i], (j === 0 ? 0.95 : 0.55) * pj) });
              if (!inWin && ph > 1.5) {
                const cut = V.clamp((ph - 1.5) / 0.5, 0, 1);
                V.line(g, x, yy + ch / 2, x + (cs - 1.5) * cut, yy + ch / 2, { color: C.red, lw: 1.6 });
              }
            }
          }
          const note = S.mode === 'vllm' ? T.noteV(s, sm) : T.noteT(s, sm);
          V.text(g, note, w - pad, y + laneH / 2 + 4, { size: 11, font: 'mono', align: 'right', color: s.copy ? TF : C.muted, alpha: pProp });
        }
        // window label
        V.text(g, T.win(P.R), pad, yW - 8, { size: 11, font: 'mono', color: C.muted });
        // window background track
        V.box(g, x0 - 2, yW - 2, X(S.mode === 'tfs' ? 16 : 32) - x0 + 4, winH + 4, { r: 5, fill: V.alpha(C.muted, 0.06), stroke: V.alpha(C.muted, 0.18), lw: 1 });
        const pRep = V.clamp((ph - 3.2) / 1.4, 0, 1);
        const headX = X(0) + (X(P.graph || P.R) - X(0)) * pRep;
        // window cells
        P.st.forEach((s) => {
          for (let j = 0; j < s.tot; j++) {
            const idx = s.start + j;
            const pa = V.ease.inOut(V.clamp((ph - 1.2 - idx * 0.012) / 0.9, 0, 1));
            if (pa <= 0) continue;
            const isPad = j >= s.rows;
            const nShow = S.mode === 'vllm' ? s.want : s.d;
            const fromLane = j <= nShow && !isPad;
            const sx = fromLane ? lx(j) : X(idx), sy = fromLane ? ly(s.i) : yW;
            const tx = X(idx) + 0.5, ty = yW;
            const x = V.lerp(sx, tx, pa), y = V.lerp(sy, ty, pa);
            const ww = V.lerp(fromLane ? cs - 1.5 : cw - 1, cw - 1, pa), hh = V.lerp(fromLane ? ch : winH, winH, pa);
            const lit = pRep > 0 && x + ww <= headX + 0.5;
            const al = (j === 0 ? 0.95 : isPad ? 0.38 : 0.6) * (fromLane ? 1 : pa) * (lit ? 1.15 : pRep > 0 && pRep < 1 ? 0.55 : 1);
            V.box(g, x, y, ww, hh, { r: 3, fill: V.alpha(SC[s.i], Math.min(1, al)), stroke: isPad ? V.alpha(SC[s.i], 0.9) : null, lw: 1 });
            if (isPad && cw >= 12 && pa >= 1) V.text(g, '↻', x + ww / 2, y + hh / 2 + 1, { size: 11, align: 'center', base: 'middle', color: C.ink });
            // acceptance
            const ra = V.clamp((ph - 4.8 - j * 0.06) / 0.3, 0, 1);
            if (ra > 0 && pa >= 1) {
              if (j <= s.a) {
                V.box(g, x - 0.5, y - 0.5, ww + 1, hh + 1, { r: 3, stroke: V.alpha(C.a, ra), lw: 2 });
                V.box(g, x + 1, y + hh + 3, ww - 2, 3, { r: 1.5, fill: V.alpha(C.a, ra) });
              } else if (j === s.a + 1) {
                V.box(g, x - 0.5, y - 0.5, ww + 1, hh + 1, { r: 3, stroke: V.alpha(C.red, ra), lw: 2 });
                if (cw >= 12) V.text(g, '✗', x + ww / 2, y + hh / 2 + 1, { size: 11.5, align: 'center', base: 'middle', color: C.red, weight: 700, alpha: ra });
                else V.box(g, x + 1, y + hh + 3, ww - 2, 3, { r: 1.5, fill: V.alpha(C.red, ra) });
              } else {
                V.box(g, x, y, ww, hh, { r: 3, fill: V.alpha(C.bg, 0.6 * ra) });
              }
            }
          }
        });
        // batch padding (vLLM)
        if (S.mode === 'vllm' && P.pad > 0 && ph > 2.2) hatch(g, X(P.R) + 0.5, yW, X(P.graph) - X(P.R) - 1, winH, V.alpha(C.muted, 0.7), 4);
        // replay head
        if (pRep > 0 && pRep < 1) { V.line(g, headX, yW - 6, headX, yW + winH + 6, { color: accent, lw: 2 }); V.glow(g, headX, yW + winH / 2, 14, accent, 0.5); }
        // capture ruler
        const yR = yW + winH + 12;
        V.line(g, x0, yR, X(32), yR, { color: V.alpha(C.muted, 0.45), lw: 1 });
        const pTab = V.clamp((ph - 2.4) / 0.7, 0, 1);
        let lastLab = -1e9;
        P.sizes.forEach((sz) => {
          const x = X(sz), on = pTab > 0 && sz === P.graph;
          V.line(g, x, yR - (on ? 9 : 4), x, yR + (on ? 9 : 4), { color: on ? accent : V.alpha(C.b, 0.8), lw: on ? 2.6 : 1.4 });
          if (on) V.glow(g, x, yR, 12, accent, 0.45 + 0.25 * Math.sin(t * 6));
          const lab = String(sz), lw0 = tw(g, lab, 11, 'mono') + 4;
          if (x - lastLab >= lw0 + (sm ? 4 : 0)) { V.text(g, lab, x, yR + 17, { size: 11, font: 'mono', align: 'center', color: on ? accent : C.muted, weight: on ? 700 : 400 }); lastLab = x; }
        });
        if (pTab > 0) {
          const pk = T.park(P.graph);
          if (!sm) V.text(g, pk, X(32) + 10, yW + winH / 2 + 4, { size: 12, font: 'mono', color: P.graph ? accent : C.red, weight: 700, alpha: pTab });
          else pill(g, V.clamp(X(P.graph || P.R), pad + 40, w - pad - 40), yR + 33, pk, P.graph ? accent : C.red, { align: 'center', size: 11 });
        }
        if (!sm) V.text(g, T.ruler, X(32) + 10, yR + 4, { size: 11, color: V.alpha(C.muted, 0.9), maxW: w - X(32) - 10 - pad });
        // device-side table (TF) or descriptor (vLLM)
        const yT = yR + (sm ? 52 : 44);
        const tw0 = sm ? w - 2 * pad : Math.min(470, w * 0.56);
        if (S.mode === 'vllm') {
          V.box(g, pad, yT, tw0, 84, { r: 10, fill: V.alpha(C.b, 0.06), stroke: V.alpha(C.b, 0.4), lw: 1 });
          V.text(g, T.desc1(P.R), pad + 12, yT + 22, { size: 11.5, font: 'mono', color: C.ink, maxW: tw0 - 24, alpha: 0.3 + 0.7 * pTab });
          V.text(g, T.desc2(P.kps, P.k), pad + 12, yT + 44, { size: 11.5, color: C.d, maxW: tw0 - 24, alpha: 0.3 + 0.7 * pTab });
          V.text(g, T.desc3, pad + 12, yT + 66, { size: 11, color: C.muted, maxW: tw0 - 24 });
        } else {
          const cols = sm ? T.colsS : T.cols;
          const CWF = [0.11, 0.14, 0.15, sm ? 0.15 : 0.17, sm ? 0.17 : 0.18, 0.25], rh = 18;
          const cx = CWF.map((f, k) => pad + 8 + (tw0 - 12) * CWF.slice(0, k).reduce((a, b) => a + b, 0));
          const cwT = (k) => (tw0 - 12) * CWF[k];
          V.box(g, pad, yT, tw0, 18 + 4 * rh + 6, { r: 10, fill: V.alpha(TF, 0.05), stroke: V.alpha(TF, 0.45), lw: 1 });
          cols.forEach((c, k) => V.text(g, c, cx[k], yT + 14, { size: 11, font: 'mono', color: C.muted, maxW: cwT(k) - 3 }));
          for (let i = 0; i < 4; i++) {
            const s = P.st[i]; const y = yT + 18 + i * rh + 13;
            if (!s) { V.text(g, '—', cx[0], y, { size: 11, font: 'mono', color: V.alpha(C.muted, 0.6) }); continue; }
            const wr = V.clamp((ph - 2.4 - i * 0.12) / 0.35, 0, 1);
            if (wr > 0 && wr < 1) V.box(g, pad + 4, y - 12, tw0 - 8, rh - 2, { r: 4, fill: V.alpha(TF, 0.25 * (1 - Math.abs(wr * 2 - 1)) + 0.04) });
            const done = V.clamp((ph - 5.6 - i * 0.1) / 0.3, 0, 1);
            const parTxt = done > 0 ? (s.full ? `${s.par}→${1 - s.par}` : T.replay) : String(s.par);
            const parCol = done > 0 ? (s.full ? C.a : C.d) : C.ink;
            const vals = [T.lane(i, true), String(s.start), String(s.tot), S.mode === 'tfs' ? (sm ? '0' : '0 · ' + T.home) : String(i), parTxt, fmtPos(s.pos)];
            vals.forEach((v, k) => V.text(g, v, cx[k], y, { size: 11, font: 'mono', color: k === 0 ? SC[i] : k === 4 ? parCol : C.ink, weight: k === 0 || (k === 4 && done > 0) ? 700 : 400, maxW: cwT(k) - 3, alpha: k === 0 ? 1 : 0.35 + 0.65 * Math.max(wr, ph > 2.75 ? 1 : 0) }));
          }
          if (pTab > 0 && !sm) V.text(g, S.mode === 'tfs' ? T.writeHome : T.write, pad + tw0, yT - 6, { size: 11, align: 'right', color: TF, alpha: pTab, maxW: tw0 });
        }
        // explanation
        const ex = T.expl[S.mode](P);
        if (sm) {
          const yy = yT + (S.mode === 'vllm' ? 84 : 96) + 18;
          const nl = wrap(g, ex, pad, yy, w - 2 * pad, { size: 11.5, lh: 15, color: C.ink, alpha: 0.25 + 0.75 * pTab });
          V.text(g, T.acc(P.tokens), pad, yy + nl * 15 + 6, { size: 11.5, color: C.a, weight: 600, alpha: V.clamp((ph - 5.4) / 0.4, 0, 1) });
        } else {
          const xx = pad + tw0 + 22, mw = w - xx - pad;
          wrap(g, ex, xx, yT + 14, mw, { size: 12, lh: 17, color: C.ink, alpha: 0.25 + 0.75 * pTab });
          V.text(g, T.acc(P.tokens), xx, yT + 84, { size: 11.5, color: C.a, weight: 600, maxW: mw, alpha: V.clamp((ph - 5.4) / 0.4, 0, 1) });
        }
        autoH(fig, sm ? yT + 96 + 18 + 62 : yT + 108);
      });
    }

    /* =====================================================================
       FIG 12 — TensorFold: the RDMA collective inside the graph (device-side epoch)
       ===================================================================== */
    {
      const fig = q('#fig-tf-epoch');
      const T = {
        modes: [{ v: 'frozen', label: 'recorded scalar (hypothetical)' }, { v: 'device', label: 'device-side epoch (TensorFold)' }],
        head: {
          frozen: 'seq passed by value: recorded at capture = 1 (hypothetical)',
          device: 'seq = epoch + 1, advanced by the kernel itself (TensorFold)',
        },
        headS: { frozen: 'recorded seq = 1 (hypothetical)', device: 'seq = epoch + 1 (TensorFold)' },
        rank: (r) => 'rank ' + r,
        late: 'late',
        blocks: 'blocks',
        arrivals: (k) => `arrivals ${k}/4`,
        seq: 'seq',
        locked: '🔒 recorded',
        fromEpoch: (e) => `epoch ${e} + 1`,
        fromEpochS: (e) => `(${e}+1)`,
        slot: (s) => 'slot ' + s,
        flag: 'peer flag',
        st: {
          idle: 'kernel not started yet',
          copy: 'copying the shard',
          wait: 'waiting for flag = seq',
          skip: 'flag already = 1: no wait',
          ok: (n) => `✓ read shard #${n}`,
          bad: (n) => `✗ read shard #${n}: stale!`,
          idleS: 'running late…', waitS: 'waiting for flag', skipS: 'flag already 1!',
          okS: (n) => `✓ read #${n}`,
          badS: (n) => `✗ #${n}: stale!`,
        },
        out: 'output',
        epoch: 'epoch',
        fixed: 'fixed seq',
        nic: 'NIC · RDMA WRITE',
        nicS: 'NIC',
        foot: {
          frozen: 'After the 1st replay the flag already reads 1: the wait passes straight through, and whoever gets there before the peer\'s data reads the stale shard — wrong bits, no error.',
          device: 'Each replay uses seq = epoch + 1: the slot alternates (seq & 1) and the wait only passes with this exchange\'s flag. The host takes no part.',
        },
        ctlMode: 'Exchange number', ctlDelay: 'rank 1 delay', ctlReset: '⟲ reset',
        rN: 'replay', rOk: 'correct reads', rBad: 'stale reads',
      };
      const S = { mode: 'frozen', delay: 12, t0: null, n0: 2, k: -1, ok: 0, bad: 0, restart: false };
      const PER = 4.6;
      const sim = (n) => {
        const r = V.rng(97 + n * 7919);
        const j0 = r() * 0.25, j1 = r() * 0.25;
        const D = (S.delay / 30) * 1.1;
        const s = [0.25 + j0, 0.25 + D + j1];
        const dev = S.mode === 'device';
        const seq = dev ? n : 1;
        const out = [0, 1].map((rk) => {
          const pr = 1 - rk;
          const flagBefore = dev ? Math.max(0, n - 2) : (n >= 2 ? 1 : 0);
          const ready = s[rk] + 0.5;
          const flagArr = s[pr] + 1.4, dataArr = s[pr] + 1.15;
          const immediate = flagBefore === seq;
          const pass = immediate ? ready : Math.max(ready, flagArr);
          const fresh = pass >= dataArr;
          const got = fresh ? n : (dev ? n - 2 : n - 1);
          return { ready, pass, immediate, fresh, got, flagBefore, flagArr, dataArr };
        });
        return { n, seq, slot: seq & 1, dev, s, out };
      };
      const ctl = UI.controls(fig);
      const restart = () => { S.restart = true; S.ok = 0; S.bad = 0; };
      UI.seg(ctl, { label: T.ctlMode, options: T.modes, value: S.mode, onChange: (v) => { S.mode = v; restart(); } });
      UI.slider(ctl, { label: T.ctlDelay, min: 0, max: 30, step: 1, value: S.delay, fmt: (v) => v + ' µs', onInput: (v) => { S.delay = v; } });
      UI.button(ctl, T.ctlReset, restart);
      const rN = UI.readout(ctl, T.rN, '—'), rOk = UI.readout(ctl, T.rOk, '0'), rBad = UI.readout(ctl, T.rBad, '0');
      let shown = '';
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const sm = w < 560;
        if (S.t0 == null) { S.t0 = t - 2.75; S.n0 = 2; S.k = 0; }
        if (S.restart) { S.t0 = t; S.n0 = 1; S.k = 0; S.restart = false; }
        const el = Math.max(0, t - S.t0);
        const k = Math.floor(el / PER), ph = el - k * PER;
        while (S.k < k) { const Z0 = sim(S.n0 + S.k); Z0.out.forEach((o) => { if (o.fresh) S.ok++; else S.bad++; }); S.k++; }
        const Z = sim(S.n0 + k);
        const sig = Z.n + '|' + S.ok + '|' + S.bad;
        if (sig !== shown) { shown = sig; rN.set('#' + Z.n); rOk.set(String(S.ok), S.ok ? 'ok' : null); rBad.set(String(S.bad), S.bad ? 'bad' : null); }
        const RC = [C.b, C.c];
        const pad = 14;
        V.text(g, sm ? T.headS[S.mode] : T.head[S.mode], pad, 17, { size: 12, weight: 600, color: Z.dev ? C.a : C.d, maxW: w - 2 * pad });
        const gap = sm ? 44 : 140;
        const pw = (w - 2 * pad - gap) / 2, py = 30, phh = 244;
        const px = [pad, w - pad - pw];
        // middle channel
        const cx0 = pad + pw, cx1 = w - pad - pw;
        V.text(g, sm ? T.nicS : T.nic, (cx0 + cx1) / 2, py + 96, { size: 11, font: 'mono', align: 'center', color: C.muted });
        const laneY = [py + 116, py + 146];
        V.arrow(g, cx0 + 4, laneY[0], cx1 - 4, laneY[0], { color: V.alpha(RC[0], 0.45), lw: 1.4, head: 6 });
        V.arrow(g, cx1 - 4, laneY[1], cx0 + 4, laneY[1], { color: V.alpha(RC[1], 0.45), lw: 1.4, head: 6 });
        [0, 1].forEach((rk) => {
          const s0 = Z.s[rk];
          const pd = V.clamp((ph - (s0 + 0.55)) / 0.6, 0, 1), pf = V.clamp((ph - (s0 + 1.15)) / 0.25, 0, 1);
          const a = rk === 0 ? cx0 + 6 : cx1 - 6, b = rk === 0 ? cx1 - 6 : cx0 + 6;
          if (pd > 0 && pd < 1) { const x = V.lerp(a, b, pd); V.box(g, x - 9, laneY[rk] - 13, 18, 10, { r: 3, fill: RC[rk] }); }
          if (pf > 0 && pf < 1) { const x = V.lerp(a, b, pf); V.dot(g, x, laneY[rk] + 9, 4.5, C.d); }
        });
        // panels
        [0, 1].forEach((rk) => {
          const o = Z.out[rk], x = px[rk], s0 = Z.s[rk];
          const now = ph;
          const stale = now >= o.pass + 0.15 && !o.fresh;
          V.box(g, x, py, pw, phh, { r: 12, fill: V.alpha(RC[rk], 0.05), stroke: stale ? V.alpha(C.red, 0.85) : V.alpha(RC[rk], 0.45), lw: stale ? 2 : 1 });
          V.text(g, T.rank(rk) + (rk === 1 && S.delay > 0 && !sm ? ' · ' + T.late : ''), x + 10, py + 17, { size: 12, font: 'mono', weight: 700, color: RC[rk] });
          // blocks
          const bs = Math.min(20, (pw - 20 - 18) / 4);
          let arrived = 0;
          for (let b = 0; b < 4; b++) {
            const done = now >= s0 + 0.12 * (b + 1);
            if (done) arrived++;
            V.box(g, x + 10 + b * (bs + 6), py + 26, bs, bs, { r: 4, fill: V.alpha(RC[rk], done ? 0.85 : 0.12), stroke: V.alpha(RC[rk], 0.6), lw: 1 });
          }
          if (now >= o.ready && now < o.ready + 0.5) V.text(g, '🔔', x + 10 + 4 * (bs + 6) - 2, py + 26 + bs - 3, { size: 13, alpha: 1 - (now - o.ready) / 0.5 });
          V.text(g, sm ? arrived + '/4' : T.arrivals(arrived), x + pw - 10, py + 17, { size: 11, font: 'mono', align: 'right', color: C.muted });
          // seq
          const ySeq = py + 26 + bs + 22;
          V.text(g, T.seq + ' = ' + Z.seq, x + 10, ySeq, { size: sm ? 13 : 15, font: 'mono', weight: 700, color: Z.dev ? C.a : C.d });
          V.text(g, Z.dev ? (sm ? T.fromEpochS(Z.n - 1) : T.fromEpoch(Z.n - 1)) : (sm ? '🔒' : T.locked), x + pw - 10, ySeq, { size: 11, align: 'right', color: C.muted, maxW: pw * 0.45 });
          // receive slots
          const yS = ySeq + 12, sw = (pw - 26) / 2, sh = 38;
          [0, 1].forEach((sl) => {
            const sx = x + 10 + sl * (sw + 6);
            const cur = sl === Z.slot;
            let tag;
            if (Z.dev) tag = sl === Z.slot ? (now >= o.dataArr ? Z.n : Z.n - 2) : Z.n - 1;
            else tag = sl !== 1 ? 0 : (now >= o.dataArr ? Z.n : Z.n - 1);
            const fresh = now >= o.dataArr && cur;
            V.box(g, sx, yS, sw, sh, { r: 6, fill: V.alpha(RC[1 - rk], tag >= 1 ? (fresh ? 0.3 : 0.12) : 0.03), stroke: cur ? V.alpha(RC[1 - rk], 0.95) : V.alpha(C.muted, 0.3), lw: cur ? 1.8 : 1 });
            V.text(g, T.slot(sl), sx + 6, yS + 13, { size: 11, font: 'mono', color: C.muted, maxW: sw - 8 });
            V.text(g, tag >= 1 ? 'r' + (1 - rk) + ' #' + tag : '—', sx + 6, yS + 30, { size: 11.5, font: 'mono', weight: 700, color: tag >= 1 ? C.ink : C.muted, maxW: sw - 8 });
          });
          // flag
          const yF = yS + sh + 20;
          const fv = now >= o.flagArr ? Z.seq : o.flagBefore;
          V.text(g, T.flag + ': ' + fv, x + 10, yF, { size: 11.5, font: 'mono', color: fv === Z.seq ? (o.immediate && !Z.dev ? C.d : C.a) : C.ink, weight: 600, maxW: pw - 34 });
          const waiting = now >= o.ready && now < o.pass;
          if (waiting) { g.beginPath(); g.arc(x + pw - 18, yF - 4, 6, t * 6, t * 6 + 4.2); g.strokeStyle = C.d; g.lineWidth = 2; g.stroke(); }
          // status
          const yStt = yF + 22;
          let stt, scol = C.muted;
          if (now < s0) stt = sm ? T.st.idleS : T.st.idle;
          else if (now < o.ready) stt = T.st.copy;
          else if (now < o.pass) { stt = sm ? T.st.waitS : T.st.wait; scol = C.d; }
          else if (o.immediate && !Z.dev && now < o.pass + 0.6) { stt = sm ? T.st.skipS : T.st.skip; scol = C.d; }
          else { stt = o.fresh ? (sm ? T.st.okS(o.got) : T.st.ok(o.got)) : (sm ? T.st.badS(o.got) : T.st.bad(o.got)); scol = o.fresh ? C.a : C.red; }
          V.text(g, stt, x + 10, yStt, { size: 11.5, color: scol, weight: 600, maxW: pw - 20 });
          // output (rank order)
          const yO = yStt + 14;
          V.box(g, x + 10, yO, pw - 20, 30, { r: 6, fill: V.alpha(C.muted, 0.06), stroke: V.alpha(C.muted, 0.3), lw: 1 });
          if (now >= o.pass + 0.15) {
            const a0 = rk === 0 ? Z.n : o.got, a1 = rk === 0 ? o.got : Z.n;
            const txt = (sm ? '' : T.out + ' = ') + 'r0#' + a0 + ' + r1#' + a1;
            V.text(g, txt, x + 18, yO + 20, { size: 11.5, font: 'mono', color: o.fresh ? C.a : C.red, weight: 700, maxW: pw - 36 });
          } else V.text(g, T.out + ' …', x + 18, yO + 20, { size: 11.5, font: 'mono', color: C.muted, maxW: pw - 36 });
          // epoch
          const yE = yO + 52;
          if (Z.dev) {
            const adv = now >= o.pass + 0.35;
            const e = adv ? Z.n : Z.n - 1;
            V.text(g, T.epoch + ': ' + e, x + 10, yE, { size: 13, font: 'mono', weight: 700, color: C.a });
            if (adv && now < o.pass + 0.9) V.glow(g, x + 40, yE - 5, 18, C.a, 0.6 * (1 - (now - o.pass - 0.35) / 0.55));
          } else V.text(g, T.fixed + ': 1 🔒', x + 10, yE, { size: 13, font: 'mono', weight: 700, color: C.d });
        });
        // footer
        wrap(g, T.foot[S.mode], pad, py + phh + 20, w - 2 * pad, { size: 11.5, lh: 15, color: C.ink });
        autoH(fig, py + phh + (sm ? 76 : 44));
      });
    }
  },
});

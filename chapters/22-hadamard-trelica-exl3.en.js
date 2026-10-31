LA.chapter({
  id: 'hadamard-trelica-exl3',
  part: 4,
  order: 22,
  title: 'Rotations, trellises and EXL3',
  navTitle: 'Hadamard, trellises & EXL3',
  subtitle: 'Rotating matrices to dissolve outliers, trading a giant codebook for a 16-bit trellis and decoding each weight with a handful of instructions: how EXL3 stores the 311 billion expert weights of GLM in 156 GB with FP8 quality.',
  minutes: 60,
  recipes: ['glm', 'tf', 'qwen', 'ds'],
  tags: ['Hadamard', 'incoherence', 'FWHT', 'vector quantization', 'TCQ', 'QTIP', 'Viterbi', 'codebook', 'MCG', 'EXL3', 'bpw', 'weight-only', 'TensorFold'],
  goals: [
    'Explain why an outlier ruins scalar quantization and how a randomized Hadamard rotation (<code>suh</code>/<code>svh</code>) makes the matrix <b>incoherent</b> without changing the math',
    'Understand vector quantization, the bitshift trellis, Viterbi and tail-biting, and why encoding is expensive while decoding is almost free',
    'Decode an EXL3 weight by hand (16-bit window → MCG/mul1 → FP16) and compute the effective bits of a <code>trellis</code> tensor',
    'Read the EXL3 choices of the GLM recipe (TR3 K4 MCG checkpoint, <code>--quantization exl3</code>, <code>Exl3MoEMethod</code>, dense EXL3, 6 bpw drafter) and compare them with NVFP4 and MXFP4',
  ],
  prereqs: ['quantizacao-fundamentos', 'fp8-fp4-microescala'],
  summary: [
    'The error of uniform quantization grows with the square of the crest factor (κ²): one outlier stretches the ruler of the whole group. Orthogonal rotations preserve the math, \\(xW = (xU)(U^\\top W V)V^\\top\\), and spread the outlier over every coordinate.',
    'Randomized Hadamard = ±1 signs + FWHT: n·log₂n additions, and it is its own inverse. In EXL3, \\(W = \\mathrm{diag}(suh)\\,H_{128}\\,\\hat W\\,H_{128}\\,\\mathrm{diag}(svh)\\); in the indexer, q and k are rotated before FP8/FP4 because \\((Hq)\\cdot(Hk) = q\\cdot k\\).',
    'For Gaussian weights, 4-bit scalar quantization sits ~2.4× above the D(R) limit. VQ closes the gap, but the codebook explodes (2<sup>kd</sup>). The bitshift trellis (L = 16, K = 4) imitates a huge VQ: each weight reads a 16-bit window and pays only 4.',
    'Encoding is Viterbi (expensive, offline, weighted by the calibration Hessian); decoding is a shift + a computed codebook: MCG in the GLM experts (<code>0xCBAC1FED</code>, marker −877912083), mul1 in turboderp\'s dense layers (dp4a) and 3INST in QTIP.',
    'Format: <code>trellis</code> int16 [d_in/16, d_out/16, 16k] + <code>suh</code>/<code>svh</code> FP16 + an int32 marker; effective bpw 4.0117 at [4096 × 2048] K4. The TR3 uniform-K4 checkpoint is 163.6 GiB, routed experts only, with KLD 0.024555 ≈ official FP8 (0.024629) at 54% of the bytes.',
    'In vLLM, EXL3 is a plugin (<code>Exl3Config</code> → <code>Exl3MoEMethod</code>/<code>Exl3LinearMethod</code>) that keeps the weights packed, slices gate/up by column and down by row, validates the marker and dispatches by M: in the dense linear, register decoding up to 144 rows and reconstruction above that; in the experts, a single <code>exl3_moe</code> while the step has ≤ 32 tokens and, above that, the experts with more than 32 rows go to E3 (directly on the trellis, no reconstruction). Cutting TP at multiples of 128 keeps each Hadamard block within one rank; at TP3 the cut would fall in the middle of a tile.',
    'The TensorFold version of the GLM recipe serves the same checkpoint with no plugin, through the engine\'s own kernels and with each row\'s bits reproducible: it rotates each token once (every expert in a layer has the same <code>suh</code>), swaps the operands in prompts (\\(Y^\\top = W^\\top X^\\top\\): one decoded tile serves 64 tokens, not 16) and, in decode, reads the trellis in 16-byte loads ahead of time (+2.7% / +3.3% together with L2 prefetch and the 512 KiB RDMA).',
  ],
  quiz: [
    { q: 'A group of 128 Gaussian weights (largest |w| ≈ 3σ) gains a single weight of 12σ. In a uniform quantization with one scale per group, what happens to the error of the other 127 weights?', o: ['Nothing: only the outlier suffers', 'It grows about 4×', 'It grows about 16×, because the MSE scales with κ² and κ goes from ~3 to 12', 'It shrinks, because the scale got bigger'], a: 2, e: 'The ruler\'s step is \\(a_{\\max}/q_{\\max}\\), and MSE ≈ Δ²/12. Quadrupling \\(a_{\\max}\\) multiplies everyone\'s MSE by 16, even for the weights that did not change.' },
    { q: 'Why does GLM also rotate the indexer\'s <i>query</i>, and not just the keys stored in the cache?', o: ['To save memory', 'Because the score is only preserved when both sides are rotated: \\((Hq)\\cdot(Hk) = q^\\top H^\\top H k = q\\cdot k\\). With only one side, the dot product turns into something else and “the needle\'s pool drops out of the top-k”', 'Because FP8 requires power-of-2 scales', 'Because H is not invertible'], a: 1, e: 'H is orthogonal (H<sup>⊤</sup>H = I). Rotating both vectors leaves the dot product intact and spreads the outliers before quantization; rotating only one changes the score.' },
    { q: 'In the EXL3 bitshift trellis (L = 16, K = 4), how many new bits does each weight consume, and on how many bits does its value depend?', o: ['4 and 4', '16 and 16', '4 new ones, but the value depends on a 16-bit window (12 bits shared with the neighbor)', '16 new ones and 4 of state'], a: 2, e: '\\(s_t = ((s_{t-1}\\ll 4)\\,|\\,b_t) \\bmod 2^{16}\\). The 256-weight tile costs 256 × 4 bits, but each weight is \\(c(s_t)\\) of a 16-bit state. That is why encoding needs Viterbi: the choices are coupled.' },
    { q: 'What is the effective bpw of a [4096 → 2048] matrix in EXL3 K4, and where does the difference from 4 come from?', o: ['≈ 4.0117: the 4 trellis bits + <code>suh</code> (4096 × FP16) + <code>svh</code> (2048 × FP16) + the int32 marker', 'Exactly 4.000', '4.25, like MXFP4', '4.5, like NVFP4'], a: 0, e: '8·(4,194,304 + 2·(4096 + 2048) + 4)/(4096·2048) = 4.0117. There is no per-block scale: the overhead is just the rotation vectors.' },
    { q: 'What happens if someone points the GLM recipe at an EXL3 checkpoint whose experts were encoded with <code>mul1</code>?', o: ['It serves normally, just slower', 'Loading fails with “packed ABI mismatch”, because <code>Exl3MoEMethod</code> checks the MCG marker (−877912083) on every expert; decoding with the wrong codebook would give plausible but wrong weights', 'vLLM converts the codebook automatically', 'The experts fall back to BF16'], a: 1, e: 'The codebook is part of the ABI: the same int16 words turn into different values under MCG and mul1. The recipe prefers to fail loudly rather than serve numerical garbage, and the E2/E3/FAST kernels only implement K4 MCG.' },
    { q: 'Why does <code>LinearEXL3</code> reconstruct the FP16 matrix above 144 rows instead of continuing to decode in the register?', o: ['Because the registers overflow', 'Because FP16 does not fit in memory', 'Because the CUDA graph requires it', 'Because each tile is decoded again for every block of rows: with large M the decoding ALU starts to dominate, and reconstructing once + a dense GEMM comes out cheaper'], a: 3, e: 'At decode (small M) the bottleneck is reading bytes, and reading 4 bits instead of 16 wins. At prefill (large M) the bottleneck is the math, and a dense FP16/BF16 GEMM is more efficient. It is the same dispatch by M as the KDA BF16 copy (Marlin up to 512 rows, cuBLAS above). In the experts, the 32-row threshold separates thin from fat, but the fat ones do not reconstruct: E2/E3 decode each tile once and reuse it across all rows.' },
  ],
  mount(root, LA) {
    const V = LA.viz, UI = LA.ui, N = LA.num;
    const $ = (s) => root.querySelector(s);
    // Correção local: o MathML oculto do KaTeX é position:absolute; dentro de uma .la-table-wrap
    // (overflow-x:auto, mas sem position) ele escapa do recorte e alarga a página no celular.
    root.querySelectorAll('.la-table-wrap, .la-eq').forEach((el) => { if (getComputedStyle(el).position === 'static') el.style.position = 'relative'; });
    const cm = (x, d = 2) => { const s = (+x).toFixed(d); return s.charAt(0) === '-' ? '−' + s.slice(1) : s; };
    const T = (g, s, x, y, o) => V.text(g, s, x, y, o || {});
    const onClean = (fn) => { if (LA.onCleanup) LA.onCleanup(fn); };
    function fitH(fig, fn) {
      const st = fig.querySelector('.la-fig-stage');
      const apply = () => { const w = st.getBoundingClientRect().width; if (!w) return; const hh = Math.round(fn(w)); if (Math.abs((parseFloat(st.style.height) || 0) - hh) > 1) st.style.height = hh + 'px'; };
      apply();
      const ro = new ResizeObserver(apply); ro.observe(fig);
      onClean(() => ro.disconnect());
      return apply;
    }
    function wrapTop(g, s, x, y, maxW, lh, o) { // quebra em linhas a partir de y (primeira linha em y)
      g.font = V.font(o.size || 12, o.font || 'sans', o.weight || 400);
      const words = String(s).split(' '); const lines = []; let cur = '';
      words.forEach((wd) => { const tt = cur ? cur + ' ' + wd : wd; if (cur && g.measureText(tt).width > maxW) { lines.push(cur); cur = wd; } else cur = tt; });
      if (cur) lines.push(cur);
      lines.forEach((l, i) => V.text(g, l, x, y + i * lh, o));
      return lines.length;
    }
    const hex = (v, n) => (v >>> 0).toString(16).toUpperCase().padStart(n, '0');
    const bin = (v, n) => N.toBin(v >>> 0, n);
    /* ---------- FP16 e codebooks (réplica do codebook.cuh) ---------- */
    const f16 = (b) => N.decode('fp16', b & 0xffff);
    function r16(x) { // arredondamento RNE para FP16 (validado contra LA.num.round)
      if (x === 0 || !isFinite(x)) return x;
      const a = Math.abs(x); let e = Math.floor(Math.log2(a)); if (e < -14) e = -14;
      const ulp = Math.pow(2, e - 10); const q = a / ulp; const fl = Math.floor(q); const fr = q - fl;
      const m = fr > 0.5 ? fl + 1 : fr < 0.5 ? fl : (fl % 2 === 0 ? fl : fl + 1);
      return (x < 0 ? -1 : 1) * m * ulp;
    }
    const K_INV = f16(0x1eee), K_BIAS = f16(0xc931);
    function cbDetail(kind, s) {
      if (kind === 'mul1') {
        const x = Math.imul(s, 0x83DCD12D) >>> 0;
        const bytes = [x & 255, (x >>> 8) & 255, (x >>> 16) & 255, x >>> 24];
        const sum = bytes[0] + bytes[1] + bytes[2] + bytes[3];
        const hb = 0x6400 + sum, hv = f16(hb);
        return { x, bytes, sum, hb, hv, v: r16(hv * K_INV + K_BIAS) };
      }
      const x = kind === 'mcg' ? Math.imul(s, 0xCBAC1FED) >>> 0 : (Math.imul(s, 89226354) + 64248484) >>> 0;
      const y = ((x & 0x8FFF8FFF) ^ 0x3B603B60) >>> 0;
      const lo = f16(y & 0xffff), hi = f16(y >>> 16);
      return { x, y, lo, hi, v: r16(lo + hi) };
    }
    const mcg = (s) => cbDetail('mcg', s).v;
    const cbCache = {};
    function cbAll(kind) {
      if (cbCache[kind]) return cbCache[kind];
      const NB = 72, lo = -4.2, hi = 4.2, hist = new Array(NB).fill(0);
      let s1 = 0, s2 = 0;
      for (let i = 0; i < 65536; i++) { const v = cbDetail(kind, i).v; s1 += v; s2 += v * v; const b = Math.floor(((v - lo) / (hi - lo)) * NB); if (b >= 0 && b < NB) hist[b]++; }
      const mean = s1 / 65536, std = Math.sqrt(s2 / 65536 - mean * mean);
      return (cbCache[kind] = { hist, mean, std, NB, lo, hi });
    }

    /* =============== Figura 1: a régua esticada =============== */
    (function () {
      const fig = $('#fig-regua'); if (!fig) return;
      const n = 32, OUT = 9;
      const rr = V.rng(21);
      const base = Array.from({ length: n }, () => rr.normal());
      const sg = N.randSigns(n, 5);
      const S = { out: 10, bits: 4, rot: false, a: 0 };
      let D;
      function quant(v, bits) {
        const qmax = (1 << (bits - 1)) - 1;
        const amax = Math.max(...v.map(Math.abs));
        const d = amax / qmax;
        const q = v.map((x) => Math.max(-qmax, Math.min(qmax, Math.round(x / d))));
        return { qmax, amax, d, q, deq: q.map((k) => k * d) };
      }
      const ctl = UI.controls(fig);
      UI.slider(ctl, { label: 'outlier', min: 1, max: 16, step: 0.5, value: S.out, fmt: (v) => cm(v, 1) + 'σ', onInput: (v) => { S.out = v; upd(); } });
      UI.seg(ctl, { label: 'bits', options: [3, 4], value: 4, onChange: (v) => { S.bits = v; upd(); } });
      UI.toggle(ctl, { label: 'rotate (randomized Hadamard)', value: false, onChange: (v) => { S.rot = v; setRead(); } });
      const rK = UI.readout(ctl, 'crest factor', '—');
      const rQ = UI.readout(ctl, 'SQNR of the 31 normal weights', '—');
      function upd() {
        const w = base.slice(); w[OUT] = S.out;
        const rw = N.fwht(w.map((x, i) => x * sg[i]));
        const A = quant(w, S.bits), B = quant(rw, S.bits);
        const back = N.fwht(B.deq).map((x, i) => x * sg[i]);
        const rms = Math.sqrt(w.reduce((s, x) => s + x * x, 0) / n);
        const sq = (d) => { let s = 0, e = 0; for (let i = 0; i < n; i++) if (i !== OUT) { s += w[i] * w[i]; e += (d[i] - w[i]) ** 2; } return 10 * Math.log10(s / Math.max(e, 1e-12)); };
        D = { w, rw, A, B, rms, kA: A.amax / rms, kB: B.amax / rms, sA: sq(A.deq), sB: sq(back), uA: new Set(A.q).size, uB: new Set(B.q).size };
        setRead();
      }
      function setRead() {
        if (!D) return;
        rK.set(cm(D.kA, 2) + ' → ' + cm(D.kB, 2));
        rQ.set(cm(D.sA, 1) + ' dB → ' + cm(D.sB, 1) + ' dB');
      }
      upd();
      V.fig(fig, (g, w, h, t, dt) => {
        if (w < 80 || h < 80) return; // canvas ainda sem tamanho (ex.: durante um redimensionamento)
        const C = V.C();
        S.a += ((S.rot ? 1 : 0) - S.a) * Math.min(1, dt * 4.5);
        const a = V.ease.inOut(V.clamp(S.a, 0, 1));
        const L = 46, R = 14, top = 58, bot = 26;
        const pw = w - L - R, ph = h - top - bot;
        const ymax = Math.max(D.A.amax, 3) * 1.1;
        const y = (v) => top + (1 - (v + ymax) / (2 * ymax)) * ph;
        const qmax = D.A.qmax, dl = V.lerp(D.A.d, D.B.d, a);
        const used = new Set((a < 0.5 ? D.A.q : D.B.q));
        // níveis
        for (let k = -qmax; k <= qmax; k++) {
          const yy = y(k * dl); const u = used.has(k);
          V.line(g, L, yy, L + pw, yy, { color: V.alpha(u ? C.b : C.muted, u ? 0.75 : 0.22), lw: u ? 1.4 : 1 });
        }
        T(g, '+' + qmax + 'Δ', L + pw, y(qmax * dl) - 4, { size: 10.5, font: 'mono', color: C.b, align: 'right' });
        T(g, '−' + qmax + 'Δ', L + pw, y(-qmax * dl) + 12, { size: 10.5, font: 'mono', color: C.b, align: 'right' });
        V.line(g, L, y(0), L + pw, y(0), { color: V.alpha(C.ink, 0.35), lw: 1 });
        // eixo
        [-1, 0, 1].forEach((s) => { const v = s * Math.round(ymax * 0.9); T(g, (v > 0 ? '+' : '') + String(v).replace('-', '−') + 'σ', L - 6, y(v) + 4, { size: 10.5, font: 'mono', color: C.muted, align: 'right' }); });
        // barras
        const bw = pw / n;
        for (let i = 0; i < n; i++) {
          const v = V.lerp(D.w[i], D.rw[i], a);
          const x = L + i * bw + bw * 0.16, ww = bw * 0.68;
          const y0 = y(0), y1 = y(v);
          const col = i === OUT ? V.mix(C.c, C.a, a) : C.a;
          g.fillStyle = V.alpha(col, 0.78);
          V.rr(g, x, Math.min(y0, y1), ww, Math.max(1.5, Math.abs(y1 - y0)), Math.max(0, Math.min(3, ww / 3))); g.fill();
        }
        // valores quantizados
        const qa = a < 0.04 ? 1 - a / 0.04 : a > 0.96 ? (a - 0.96) / 0.04 : 0;
        if (qa > 0) {
          const deq = a < 0.5 ? D.A.deq : D.B.deq;
          g.globalAlpha = qa;
          for (let i = 0; i < n; i++) { const x = L + i * bw + bw * 0.05; V.line(g, x, y(deq[i]), x + bw * 0.9, y(deq[i]), { color: C.ink, lw: 2.4 }); }
          g.globalAlpha = 1;
        }
        // cabeçalho
        const rot = a > 0.5;
        T(g, rot ? 'WITH ROTATION  H·S' : 'NO ROTATION', L, 18, { size: 11, font: 'mono', color: rot ? C.a : C.c, weight: 600 });
        const line2 = `κ = ${cm(rot ? D.kB : D.kA, 2)}   ·   levels used ${rot ? D.uB : D.uA} of ${2 * qmax + 1}   ·   SQNR (31 normal) ${cm(rot ? D.sB : D.sA, 1)} dB`;
        T(g, line2, L, 38, { size: w < 520 ? 11 : 12.5, color: C.ink, maxW: w - L - 10 });
        if (!rot) { const xo = L + OUT * bw + bw / 2; T(g, 'outlier', xo, Math.max(top - 4, y(D.w[OUT]) - 6), { size: 11, color: C.c, align: 'center' }); }
      });
    })();

    /* =============== Figura 2: borboletas da FWHT =============== */
    (function () {
      const fig = $('#fig-borboleta'); if (!fig) return;
      const x0 = [8, 0.5, -0.3, 0.2, 0.1, -0.4, 0.3, 0.0];
      const sgn = [1, -1, -1, 1, -1, 1, 1, -1];
      const S = { signs: false };
      let cols = [];
      function compute() {
        const v0 = x0.map((x, i) => (S.signs ? x * sgn[i] : x));
        const c = [v0.slice()]; let a = v0.slice();
        for (let hh = 1; hh < 8; hh *= 2) { const b = a.slice(); for (let i = 0; i < 8; i += hh * 2) for (let j = i; j < i + hh; j++) { b[j] = a[j] + a[j + hh]; b[j + hh] = a[j] - a[j + hh]; } c.push(b); a = b; }
        c.push(a.map((v) => v / Math.sqrt(8)));
        c.push(N.fwht(c[4]));
        cols = c;
      }
      compute();
      const steps = [
        { t: 'Input: one outlier', d: 'The vector has one value of 8 and seven small values. The crest factor κ (max ÷ RMS) is 2.81, almost the maximum possible in 8 positions (√8 ≈ 2.83): a quantization ruler would be stretched entirely by the 8.' },
        { t: 'Stage 1: neighbors (distance 1)', d: 'Each pair (a, b) becomes (a + b, a − b). Solid lines add and dashed lines subtract. The 8 has already “leaked” into the neighboring position.' },
        { t: 'Stage 2: distance 2', d: 'Now the pairs are (0,2), (1,3), (4,6), (5,7). Each value carries information from 4 inputs.' },
        { t: 'Stage 3: distance 4', d: 'After log₂8 = 3 stages, each output is a sum of <b>all</b> the inputs with ±1 signs. That took 8 × 3 = 24 additions and subtractions and no multiplications.' },
        { t: 'Normalize by 1/√8', d: 'Dividing by √8 ≈ 2.83 makes the matrix orthogonal: the norm is preserved. The outlier turned into eight similar values (between 2.2 and 3.3), and κ dropped from 2.81 to 1.14.' },
        { t: 'Applying H again undoes it', d: 'H is symmetric and orthogonal, so H·H = I. The same FWHT, applied to the result, returns the exact input. That is why the kernel uses the same code to rotate and unrotate.' },
      ];
      const st = UI.stepper(fig, steps, { autoplay: 5600 });
      const ctl = UI.controls(fig);
      UI.toggle(ctl, { label: 'random signs first (S)', value: false, onChange: (v) => { S.signs = v; compute(); } });
      const kap = (v) => { const m = Math.max(...v.map(Math.abs)); const r = Math.sqrt(v.reduce((s, x) => s + x * x, 0) / v.length) || 1; return m / r; };
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const step = st.i, el = st.elapsed();
        const ncol = 6, padL = 6, padR = 6;
        const cw = (w - padL - padR) / ncol;
        const top = 46, botH = 44;
        const rowH = (h - top - botH) / 8;
        const bw = Math.min(70, cw * 0.74), bh = Math.min(28, rowH - 6);
        const cx = (c) => padL + cw * (c + 0.5);
        const cy = (r) => top + rowH * (r + 0.5);
        const small = w < 520;
        const heads = small ? ['x', 'st. 1', 'st. 2', 'st. 3', 'Hx', 'H(Hx)'] : ['x', 'stage 1', 'stage 2', 'stage 3', 'Hx (÷√8)', 'H(Hx)'];
        const scale = 8.5;
        // arestas
        for (let c = 1; c <= Math.min(step, 5); c++) {
          const active = c === step;
          const xa = cx(c - 1) + bw / 2, xb = cx(c) - bw / 2;
          const edges = [];
          if (c <= 3) {
            const d = 1 << (c - 1);
            for (let i = 0; i < 8; i++) if (i % (2 * d) < d) { const j = i + d; edges.push([i, i, 1], [j, i, 1], [i, j, 1], [j, j, -1]); }
          } else for (let i = 0; i < 8; i++) edges.push([i, i, 0]);
          edges.forEach(([r0, r1, sgn2]) => {
            const col = sgn2 < 0 ? C.c : sgn2 > 0 ? C.a : C.b;
            V.line(g, xa, cy(r0), xb, cy(r1), { color: V.alpha(col, active ? 0.75 : 0.35), lw: active ? 1.6 : 1.1, dash: sgn2 < 0 ? [4, 3] : null });
            if (active) { const u = (el * 0.7) % 1; const p = V.along(xa, cy(r0), xb, cy(r1), u); V.dot(g, p[0], p[1], 2.6, col); }
          });
          if (c >= 4) T(g, c === 4 ? '×1/√8' : 'H again', (xa + xb) / 2, top - 8, { size: 10.5, font: 'mono', color: C.b, align: 'center' });
        }
        // caixas
        for (let c = 0; c < ncol; c++) {
          const vis = c <= step;
          T(g, heads[c], cx(c), 18, { size: small ? 10.5 : 11.5, font: 'mono', color: c === step ? C.ink : C.muted, align: 'center', weight: c === step ? 600 : 400 });
          for (let r = 0; r < 8; r++) {
            const x = cx(c) - bw / 2, y = cy(r) - bh / 2;
            if (!vis) { V.box(g, x, y, bw, bh, { r: 6, fill: V.alpha(C.muted, 0.06), stroke: V.alpha(C.muted, 0.18), lw: 1 }); continue; }
            const v0 = cols[c][r], v = Math.abs(v0) < 0.005 ? 0 : v0;
            const tt = V.clamp(v / scale, -1, 1);
            V.box(g, x, y, bw, bh, { r: 6, fill: V.cmap(tt, 'div'), stroke: c === step ? V.alpha(C.ink, 0.5) : V.alpha(C.muted, 0.25), lw: 1 });
            const light = Math.abs(tt) > 0.55;
            T(g, cm(v, Math.abs(v) >= 10 ? 1 : 2), cx(c), cy(r) + 0.5, { size: Math.min(12, bw * 0.24), font: 'mono', align: 'center', base: 'middle', color: light ? C.bg : C.ink });
          }
          if (vis) {
            T(g, 'κ ' + cm(kap(cols[c]), 2), cx(c), h - botH + 18, { size: small ? 10.5 : 11.5, font: 'mono', align: 'center', color: C.d });
          }
        }
        const nrm = (v) => Math.sqrt(v.reduce((s, x) => s + x * x, 0));
        T(g, step >= 4 ? `‖x‖ = ${cm(nrm(cols[0]), 3)}   ·   ‖Hx‖ = ${cm(nrm(cols[4]), 3)}` : 'κ = max|v| ÷ RMS(v)', w / 2, h - 8, { size: small ? 10.5 : 11.5, font: 'mono', align: 'center', color: C.muted });
      });
    })();

    /* =============== Figura 3: incoerência 2D =============== */
    (function () {
      const fig = $('#fig-incoerencia'); if (!fig) return;
      const base = V.randMatrix(8, 8, 4);
      const H8 = N.hadamard(8);
      const su = N.randSigns(8, 11), sv = N.randSigns(8, 12);
      const S = { k: 6, kind: 'pico', side: 'ambos', signs: true };
      let D;
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'outlier', options: [{ v: 'pico', label: 'spike' }, { v: 'linha', label: 'row' }, { v: 'coluna', label: 'column' }], value: 'pico', onChange: (v) => { S.kind = v; upd(); } });
      UI.slider(ctl, { label: 'intensity', min: 1, max: 10, step: 0.5, value: S.k, fmt: (v) => cm(v, 1) + '×', onInput: (v) => { S.k = v; upd(); } });
      UI.seg(ctl, { label: 'rotate', options: [{ v: 'entrada', label: 'input only' }, { v: 'saida', label: 'output only' }, { v: 'ambos', label: 'both' }], value: 'ambos', onChange: (v) => { S.side = v; upd(); } });
      UI.toggle(ctl, { label: 'random signs', value: true, onChange: (v) => { S.signs = v; upd(); } });
      const r1 = UI.readout(ctl, 'incoherence before → after', '—');
      const r2 = UI.readout(ctl, 'INT4 error before → after', '—');
      const mu = (M) => { let mx = 0, f = 0; M.forEach((r) => r.forEach((x) => { mx = Math.max(mx, Math.abs(x)); f += x * x; })); return (mx * 8) / Math.sqrt(f); };
      const int4 = (M) => { let mx = 0; M.forEach((r) => r.forEach((x) => { mx = Math.max(mx, Math.abs(x)); })); const d = mx / 7; let e = 0, f = 0; M.forEach((r) => r.forEach((x) => { const q = Math.max(-7, Math.min(7, Math.round(x / d))) * d; e += (q - x) ** 2; f += x * x; })); return Math.sqrt(e / f); };
      function upd() {
        const W = base.map((r) => r.slice());
        if (S.kind === 'pico') W[2][5] = S.k * 2.5;
        if (S.kind === 'linha') for (let j = 0; j < 8; j++) W[2][j] = base[2][j] * S.k;
        if (S.kind === 'coluna') for (let i = 0; i < 8; i++) W[i][5] = base[i][5] * S.k;
        const Lr = S.side !== 'saida', Rr = S.side !== 'entrada';
        const a = S.signs ? su : su.map(() => 1), b = S.signs ? sv : sv.map(() => 1);
        let M = W.map((r) => r.slice());
        if (Lr) M = V.matmul(H8, M.map((r, i) => r.map((x) => x * a[i])));
        if (Rr) M = V.matmul(M.map((r) => r.map((x, j) => x * b[j])), H8);
        let mx = 0; W.forEach((r) => r.forEach((x) => { mx = Math.max(mx, Math.abs(x)); }));
        D = { W, M, mx, muA: mu(W), muB: mu(M), eA: int4(W), eB: int4(M) };
        r1.set(cm(D.muA, 2) + ' → ' + cm(D.muB, 2));
        r2.set(cm(D.eA * 100, 1) + '% → ' + cm(D.eB * 100, 1) + '%');
        if (H) H.redraw();
      }
      const geo = (w) => { const wide = w >= 600; const cell = wide ? V.clamp(Math.floor((w - 150) / 16), 18, 40) : V.clamp(Math.floor((w - 40) / 8) - 2, 18, 38); return { wide, cell }; };
      fitH(fig, (w) => { const G = geo(w); return G.wide ? 8 * (G.cell + 2) + 120 : 2 * (8 * (G.cell + 2)) + 190; });
      let H = null;
      upd();
      H = V.fig(fig, (g, w, h) => {
        const C = V.C();
        const G = geo(w), cell = G.cell, M = 8 * (cell + 2) - 2;
        const fmt = (v) => { const s = Math.abs(v) >= 9.95 ? v.toFixed(0) : v.toFixed(1); return s.replace('-', '−'); };
        const colorFn = (v) => V.cmap(V.clamp(v / D.mx, -1, 1), 'div');
        let ax, ay, bx, by;
        if (G.wide) { const gap = Math.max(60, w - 2 * M - 40); const x0 = (w - 2 * M - Math.min(gap, 120)) / 2; ax = x0; ay = 50; bx = x0 + M + Math.min(gap, 120); by = 50; }
        else { ax = (w - M) / 2; ay = 44; bx = ax; by = ay + M + 96; }
        V.matrix(g, { x: ax, y: ay, data: D.W, cell, gap: 2, values: cell >= 18, fmt, colorFn, min: -D.mx, max: D.mx, cmap: 'div' });
        V.matrix(g, { x: bx, y: by, data: D.M, cell, gap: 2, values: cell >= 18, fmt, colorFn, min: -D.mx, max: D.mx, cmap: 'div' });
        const lblB = S.side === 'entrada' ? 'W̃ = H·Sᵤ·W' : S.side === 'saida' ? 'W̃ = W·Sᵥ·H' : 'W̃ = H·Sᵤ·W·Sᵥ·H';
        T(g, 'W (original)', ax + M / 2, ay - 14, { size: 12.5, font: 'mono', color: C.ink, align: 'center', weight: 600 });
        T(g, lblB, bx + M / 2, by - 14, { size: 12.5, font: 'mono', color: C.a, align: 'center', weight: 600 });
        const sub = (x, y, muv, e) => T(g, `μ = ${cm(muv, 2)}  ·  INT4: ${cm(e * 100, 1)}%`, x, y, { size: 11.5, font: 'mono', color: C.muted, align: 'center' });
        sub(ax + M / 2, ay + M + 20, D.muA, D.eA);
        sub(bx + M / 2, by + M + 20, D.muB, D.eB);
        if (G.wide) { const xa = ax + M + 12, xb = bx - 12, yy = ay + M / 2; V.arrow(g, xa, yy, xb, yy, { color: C.a, lw: 2 }); T(g, 'rotate', (xa + xb) / 2, yy - 10, { size: 11, color: C.a, align: 'center', font: 'mono' }); }
        else { const yy0 = ay + M + 32, yy1 = by - 30; V.arrow(g, w / 2, yy0, w / 2, yy1, { color: C.a, lw: 2 }); T(g, 'rotate', w / 2 + 10, (yy0 + yy1) / 2 + 4, { size: 11, color: C.a, font: 'mono' }); }
      }, { static: true });
    })();

    /* =============== Figura 4: a esteira EXL3 =============== */
    (function () {
      const fig = $('#fig-esteira'); if (!fig) return;
      const S = { mode: 'exl3', shared: true };
      const nodesFor = () => {
        if (S.mode === 'exl3') return { cols: 7, nodes: [
          { id: 'x', c: 0, l: 'b', t: 'x', s: 'FP16 activation', k: 'b' },
          { id: 'su', c: 1, l: 'b', t: '⊙ suh', s: 'signs × scale', k: 'f' },
          { id: 'h1', c: 2, l: 'b', t: 'H₁₂₈', s: '7 stages', k: 'a' },
          { id: 'w', c: 3, l: 'b', t: 'Ŵ (trellis)', s: 'window → MCG → MMA', k: 'd' },
          { id: 'h2', c: 4, l: 'b', t: 'H₁₂₈', s: 'in the epilogue', k: 'a' },
          { id: 'sv', c: 5, l: 'b', t: '⊙ svh', s: 'signs × scale', k: 'f' },
          { id: 'y', c: 6, l: 'b', t: 'y = xW', s: 'FP32 → output', k: 'c' }],
          paths: [['x', 'su', 'h1', 'w', 'h2', 'sv', 'y']] };
        if (S.mode === 'gateup') {
          const rotNodes = S.shared ? [{ id: 'r', c: 1, l: 'b', t: '⊙suh → H₁₂₈', s: '1 rotation (shared)', k: 'a' }]
            : [{ id: 'rg', c: 1, l: 0, t: '⊙suh_g → H', s: 'gate rotation', k: 'a' }, { id: 'ru', c: 1, l: 1, t: '⊙suh_u → H', s: 'up rotation', k: 'a' }];
          const nodes = [{ id: 'x', c: 0, l: 'b', t: 'x', s: 'FP16 activation', k: 'b' }].concat(rotNodes, [
            { id: 'wg', c: 2, l: 0, t: 'Ŵ gate', s: 'K4 trellis', k: 'd' }, { id: 'wu', c: 2, l: 1, t: 'Ŵ up', s: 'K4 trellis', k: 'd' },
            { id: 'hg', c: 3, l: 0, t: 'H ⊙ svh_g', s: 'epilogue', k: 'a' }, { id: 'hu', c: 3, l: 1, t: 'H ⊙ svh_u', s: 'epilogue', k: 'a' },
            { id: 'act', c: 4, l: 'b', t: 'SiLU(g)·u', s: 'clamp at 10', k: 'e' },
            { id: 'sd', c: 5, l: 'b', t: '⊙suh_d → H', s: 'rotated down input', k: 'a' },
            { id: 'wd', c: 6, l: 'b', t: 'Ŵ down', s: '→ H ⊙ svh_d → y', k: 'c' }]);
          const g0 = S.shared ? 'r' : 'rg', u0 = S.shared ? 'r' : 'ru';
          return { cols: 7, nodes, paths: [['x', g0, 'wg', 'hg', 'act', 'sd', 'wd'], ['x', u0, 'wu', 'hu', 'act', 'sd', 'wd']] };
        }
        return { cols: 5, nodes: [
          { id: 'q', c: 0, l: 0, t: 'indexer q', s: '32 heads × 128', k: 'b' }, { id: 'k', c: 0, l: 1, t: 'pool k', s: '1 key / 4 tokens', k: 'b' },
          { id: 'hq', c: 1, l: 0, t: 'H₁₂₈', s: 'rotates the query', k: 'a' }, { id: 'hk', c: 1, l: 1, t: 'H₁₂₈', s: 'rotates the key', k: 'a' },
          { id: 'fq', c: 2, l: 0, t: 'FP8 · 2ⁿ scale', s: 'DS: FP4 (QAT)', k: 'd' }, { id: 'fk', c: 2, l: 1, t: 'FP8 → cache', s: '132 B per entry', k: 'd' },
          { id: 'sc', c: 3, l: 'b', t: 'Σ w·ReLU(q̃·k̃)', s: '(Hq)·(Hk) = q·k', k: 'e' },
          { id: 'tk', c: 4, l: 'b', t: 'top-k pools', s: '511 pools + tail', k: 'c' }],
          paths: [['q', 'hq', 'fq', 'sc', 'tk'], ['k', 'hk', 'fk', 'sc', 'tk']] };
      };
      // mini vetores ilustrativos (16 dims)
      const rr = V.rng(33);
      const xv = Array.from({ length: 16 }, () => rr.normal() * 0.8); xv[6] = 5.2;
      const suv = Array.from({ length: 16 }, (_, i) => (rr() < 0.5 ? -1 : 1) * (0.6 + rr() * 0.8));
      const xs = xv.map((x, i) => x * suv[i]);
      const hx = N.fwht(xs);
      // estágios completos (ilustrativos) para as tiras de células sob cada estação
      const Wh = V.randMatrix(16, 16, 44, 0.3);
      const yp = Wh[0].map((_, j) => hx.reduce((t, x, i) => t + x * Wh[i][j], 0));
      const hy = N.fwht(yp);
      const svv = Array.from({ length: 16 }, () => (rr() < 0.5 ? -1 : 1) * (0.6 + rr() * 0.8));
      const yv = hy.map((v, i) => v * svv[i]);
      const STG = [xv, xs, hx, yp, hy, yv, yv];
      // indexador: q, k com canal outlier compartilhado
      const sI = N.randSigns(16, 9);
      const qv = Array.from({ length: 16 }, () => rr.normal()); const kv = Array.from({ length: 16 }, () => rr.normal()); qv[3] = 6.5; kv[3] = 5.2;
      const Hq = N.fwht(qv.map((x, i) => x * sI[i])), Hk = N.fwht(kv.map((x, i) => x * sI[i]));
      const dot = (a, b) => a.reduce((t, x, i) => t + x * b[i], 0);
      // erro médio do score em FP4 (MX E2M1, bloco 16) com e sem giro — 1500 pares
      const avg = (() => { const r2 = V.rng(3); let e0 = 0, e1 = 0; const q4 = (v) => N.mx(v, 'e2m1', 16).deq; for (let k = 0; k < 1500; k++) { const q = Array.from({ length: 16 }, () => r2.normal()), kk = Array.from({ length: 16 }, () => r2.normal()); q[3] *= 8; kk[3] *= 6; const ex = dot(q, kk), nr = Math.sqrt(dot(q, q) * dot(kk, kk)); const a1 = N.fwht(q.map((x, i) => x * sI[i])), b1 = N.fwht(kk.map((x, i) => x * sI[i])); e0 += Math.abs(dot(q4(q), q4(kk)) - ex) / nr; e1 += Math.abs(dot(q4(a1), q4(b1)) - ex) / nr; } return [e0 / 1500, e1 / 1500]; })();
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'mode', options: [{ v: 'exl3', label: 'EXL3 expert' }, { v: 'gateup', label: 'gate + up' }, { v: 'idx', label: 'indexer' }], value: 'exl3', onChange: (v) => { S.mode = v; syncShared(); relayout(); } });
      const tgS = UI.toggle(ctl, { label: 'shared suh (gate/up)', value: true, onChange: (v) => { S.shared = v; } });
      // o toggle só tem efeito no modo gate + up: fora dele fica desabilitado
      const syncShared = () => { const on = S.mode === 'gateup'; const inp = tgS.el.querySelector('input'); if (inp) inp.disabled = !on; tgS.el.style.opacity = on ? '' : '0.4'; tgS.el.style.pointerEvents = on ? '' : 'none'; tgS.el.title = on ? '' : 'only applies in gate + up mode'; };
      syncShared();
      const miniOf = (w) => (w < 600 ? 128 : 104);
      const relayout = fitH(fig, (w) => (w >= 700 ? (S.mode === 'exl3' ? 276 : 364) : nodesFor().cols * 56 + miniOf(w) + 44));
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const M = nodesFor();
        const horiz = w >= 700, miniH = miniOf(w);
        const y0 = 30, yD = h - miniH - 6;
        const pos = {};
        let bw, bh;
        if (horiz) {
          const colW = (w - 24) / M.cols; bw = Math.min(132, colW - 14); bh = 48;
          const mid = S.mode === 'exl3' ? y0 + 42 : (y0 + yD) / 2, gap = Math.min(58, (yD - y0) / 4);
          M.nodes.forEach((nd) => { pos[nd.id] = [12 + colW * (nd.c + 0.5), nd.l === 'b' ? mid : nd.l === 0 ? mid - gap : mid + gap]; });
        } else {
          const rowH = (yD - y0) / M.cols; bh = Math.min(40, rowH - 12);
          M.nodes.forEach((nd) => { pos[nd.id] = [nd.l === 'b' ? w / 2 : nd.l === 0 ? w * 0.27 : w * 0.73, y0 + rowH * (nd.c + 0.5)]; });
          bw = null;
        }
        const nbw = (nd) => (bw != null ? bw : nd.l === 'b' ? Math.min(250, w * 0.62) : Math.min(170, w * 0.43));
        // arestas
        const seen = {};
        M.paths.forEach((p) => { for (let i = 0; i < p.length - 1; i++) { const key = p[i] + '>' + p[i + 1]; if (seen[key]) continue; seen[key] = 1; const a = pos[p[i]], b = pos[p[i + 1]]; V.line(g, a[0], a[1], b[0], b[1], { color: V.alpha(C.muted, 0.45), lw: 1.6, dash: [5, 6], dashOffset: -t * 24 }); } });
        // partículas
        M.paths.forEach((p, pi) => {
          const segs = []; let tot = 0;
          for (let i = 0; i < p.length - 1; i++) { const a = pos[p[i]], b = pos[p[i + 1]]; const L = Math.hypot(b[0] - a[0], b[1] - a[1]); segs.push([a, b, L, i]); tot += L; }
          const K = 5;
          for (let k = 0; k < K; k++) {
            let d = (((t * 0.11 + k / K + pi * 0.1) % 1) * tot);
            for (const [a, b, L, i] of segs) { if (d <= L) { const u = d / L; const x = V.lerp(a[0], b[0], u), y = V.lerp(a[1], b[1], u); const nd = M.nodes.find((n) => n.id === p[i]); const col = C[nd.k]; V.glow(g, x, y, 9, col, 0.8); g.fillStyle = col; V.rr(g, x - 4, y - 4, 8, 8, 2); g.fill(); break; } d -= L; }
          }
        });
        // nós
        M.nodes.forEach((nd) => {
          const [x, y] = pos[nd.id]; const ww = nbw(nd); const col = C[nd.k];
          V.box(g, x - ww / 2, y - bh / 2, ww, bh, { r: 10, fill: C.bg });
          V.box(g, x - ww / 2, y - bh / 2, ww, bh, { r: 10, fill: V.alpha(col, 0.16), stroke: col, lw: 1.5 });
          T(g, nd.t, x, y - (bh > 36 ? 7 : 5), { size: horiz ? 12.5 : 12, weight: 600, color: C.ink, align: 'center', base: 'middle', maxW: ww - 8 });
          T(g, nd.s, x, y + (bh > 36 ? 10 : 9), { size: 10.5, color: C.muted, align: 'center', base: 'middle', maxW: ww - 8 });
        });
        if (S.mode === 'exl3' && horiz) {
          M.nodes.forEach((nd, i) => {
            const [x, y] = pos[nd.id]; const vec = STG[i]; const mxv = Math.max(...vec.map(Math.abs), 1e-9);
            const cs = Math.min(12, (bw - 8) / 8 - 1), gx = x - (8 * (cs + 1)) / 2, gy = y + bh / 2 + 12;
            for (let k = 0; k < 16; k++) { g.fillStyle = V.cmap(vec[k] / mxv, 'div'); V.rr(g, gx + (k & 7) * (cs + 1), gy + (k >> 3) * (cs + 1), cs, cs, 2); g.fill(); }
            T(g, 'max ' + cm(mxv, 1), x, gy + 2 * (cs + 1) + 13, { size: 10.5, font: 'mono', color: C.muted, align: 'center' });
          });
        }
        // cabeçalho
        const head = S.mode === 'exl3' ? 'y = H₁₂₈( H₁₂₈(x ⊙ suh) · Ŵ ) ⊙ svh' : S.mode === 'gateup' ? (S.shared ? 'input rotations per expert: 1 (gate suh = up suh)' : 'input rotations per expert: 2 (different suh)') : '(Hq)·(Hk) = qᵀHᵀHk = q·k: the score does not change';
        T(g, head, 12, 16, { size: w < 520 ? 11 : 12.5, font: 'mono', color: S.mode === 'gateup' && !S.shared ? C.c : C.a, maxW: w - 24 });
        // mini-gráficos
        const my = h - miniH + 8, mh = miniH - (w < 600 ? 56 : 40);
        const bars = (vals, x, ww, title, col) => {
          const mx = Math.max(...vals.map(Math.abs), 1e-9); const bw2 = ww / vals.length;
          T(g, title, x, my + 2, { size: 11, font: 'mono', color: C.muted, maxW: ww * (w < 600 ? 0.95 : 0.6) });
          const cy = my + 14 + mh / 2;
          V.line(g, x, cy, x + ww, cy, { color: V.alpha(C.muted, 0.3), lw: 1 });
          vals.forEach((v, i) => { const hh = (v / mx) * (mh / 2 - 2); g.fillStyle = V.alpha(col, 0.8); V.rr(g, x + i * bw2 + 1, Math.min(cy, cy - hh), Math.max(1, bw2 - 2), Math.max(1, Math.abs(hh)), 1.5); g.fill(); });
          if (w < 600) T(g, 'max ' + cm(mx, 1), x, my + mh + 18, { size: 10.5, font: 'mono', color: C.muted }); else T(g, 'max ' + cm(mx, 1), x + ww, my + 2, { size: 10.5, font: 'mono', color: C.muted, align: 'right' });
        };
        const gap = 14, ww3 = (w - 24 - 2 * gap) / 3;
        if (S.mode !== 'idx') {
          bars(xv, 12, ww3, 'x', C.b); bars(xs, 12 + ww3 + gap, ww3, 'x ⊙ suh', C.f); bars(hx, 12 + 2 * (ww3 + gap), ww3, 'H(x ⊙ suh)', C.a);
          if (w >= 600) T(g, '16 of 128 dimensions, illustrative values: the activation outlier turns into 16 moderate values before entering the tensor core', w / 2, h - 6, { size: 10.5, color: C.muted, align: 'center', maxW: w - 20 });
          else T(g, '16 of 128 dims, illustrative values', w / 2, h - 4, { size: 10.5, color: C.muted, align: 'center' });
        } else {
          bars(qv, 12, ww3, 'q (outlier ch. 3)', C.b); bars(Hq, 12 + ww3 + gap, ww3, 'H·S·q', C.a);
          const tx = 12 + 2 * (ww3 + gap);
          T(g, `q·k = ${cm(dot(qv, kv), 3)}`, tx, my + 22, { size: 11.5, font: 'mono', color: C.ink, maxW: ww3 });
          T(g, `(Hq)·(Hk) = ${cm(dot(Hq, Hk), 3)}`, tx, my + 40, { size: 11.5, font: 'mono', color: C.a, maxW: ww3 });
          T(g, `FP4 score error:`, tx, my + 60, { size: 10.5, color: C.muted, maxW: ww3 });
          T(g, `${cm(avg[0] * 100, 1)}% → ${cm(avg[1] * 100, 1)}% rotated`, tx, my + 76, { size: 11, font: 'mono', color: C.d, maxW: ww3 });
          T(g, w >= 600 ? 'mean of 1,500 illustrative 16-dim pairs with one outlier channel (MX E2M1, block 16)' : 'mean of 1,500 illustrative pairs (MX E2M1)', w / 2, h - 4, { size: 10.5, color: C.muted, align: 'center', maxW: w - 20 });
        }
      });
    })();

    /* =============== Figura 5: escalar vs vetorial =============== */
    (function () {
      const fig = $('#fig-vq'); if (!fig) return;
      const rr = V.rng(77);
      const pts = Array.from({ length: 600 }, () => [rr.normal(), rr.normal()]);
      const train = Array.from({ length: 4000 }, () => [rr.normal(), rr.normal()]);
      const test = Array.from({ length: 10000 }, () => [rr.normal(), rr.normal()]);
      const L1 = [-0.7979, 0.7979], L2 = [-1.510, -0.4528, 0.4528, 1.510];
      const grids = { 1: [L1, L1], 1.5: [L2, L1], 2: [L2, L2] };
      const gridCB = (R) => { const [gx, gy] = grids[R]; const o = []; gy.forEach((y) => gx.forEach((x) => o.push([x, y]))); return o; };
      const nearest = (cb, x, y) => { let bi = 0, bd = 1e9; for (let i = 0; i < cb.length; i++) { const dx = x - cb[i][0], dy = y - cb[i][1], d = dx * dx + dy * dy; if (d < bd) { bd = d; bi = i; } } return [bi, bd]; };
      const mse = (cb, P) => { let s = 0; P.forEach((p) => { s += nearest(cb, p[0], p[1])[1]; }); return s / P.length / 2; };
      const kmeans = (init, P, it) => { let cb = init.map((c) => c.slice()); for (let k = 0; k < it; k++) { const sx = cb.map(() => 0), sy = cb.map(() => 0), n = cb.map(() => 0); P.forEach((p) => { const i = nearest(cb, p[0], p[1])[0]; sx[i] += p[0]; sy[i] += p[1]; n[i]++; }); cb = cb.map((c, i) => (n[i] ? [sx[i] / n[i], sy[i] / n[i]] : c)); } return cb; };
      const CBS = {}, MSE = {};
      [1, 1.5, 2].forEach((R) => { const gcb = gridCB(R); const r2 = V.rng(5); const init = gcb.map((c) => [c[0] * 0.9 + r2.normal() * 0.25, c[1] * 0.9 + r2.normal() * 0.25]); const km = kmeans(init, train, 40); CBS['s' + R] = gcb; CBS['v' + R] = km; MSE['s' + R] = mse(gcb, test); MSE['v' + R] = mse(km, test); });
      const S = { m: 'v', R: 2 };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'method', options: [{ v: 's', label: 'scalar (grid)' }, { v: 'v', label: 'vector (k-means)' }], value: 'v', onChange: (v) => { S.m = v; upd(); } });
      UI.seg(ctl, { label: 'bits/dim', options: [{ v: 1, label: '1' }, { v: 1.5, label: '1.5' }, { v: 2, label: '2' }], value: 2, onChange: (v) => { S.R = v; upd(); } });
      const rM = UI.readout(ctl, 'MSE per dimension', '—');
      const rD = UI.readout(ctl, 'limit D(R)', '—');
      let H = null;
      function upd() { rM.set(cm(MSE[S.m + S.R], 4)); rD.set(cm(Math.pow(2, -2 * S.R), 4)); if (H) H.redraw(); }
      fitH(fig, (w) => (w >= 620 ? 390 : Math.min(w - 32, 420) + 252));
      upd();
      H = V.fig(fig, (g, w, h) => {
        const C = V.C();
        const wide = w >= 620;
        const P = wide ? Math.min(h - 30, w * 0.56) : Math.min(w - 32, 420);
        const ox = Math.round(wide ? 16 : (w - P) / 2), oy = 14;
        const R = 3.2, Pp = Math.floor(P / (P > 380 ? 5 : 4)) * (P > 380 ? 5 : 4), sx = (x) => ox + ((x + R) / (2 * R)) * Pp, sy = (y) => oy + ((R - y) / (2 * R)) * Pp;
        const cb = CBS[S.m + S.R];
        const px = P > 380 ? 5 : 4, nc = Math.floor(P / px);
        const idx = new Int8Array(nc * nc);
        for (let j = 0; j < nc; j++) for (let i = 0; i < nc; i++) { const x = -R + ((i + 0.5) / nc) * 2 * R, y = R - ((j + 0.5) / nc) * 2 * R; idx[j * nc + i] = nearest(cb, x, y)[0]; }
        for (let j = 0; j < nc; j++) for (let i = 0; i < nc; i++) { const k = idx[j * nc + i]; g.fillStyle = V.alpha(C.series[k % 6], 0.1 + 0.07 * ((k >> 1) % 2)); g.fillRect(ox + i * px, oy + j * px, px, px); }
        g.beginPath(); g.strokeStyle = V.alpha(C.ink, 0.35); g.lineWidth = 1;
        for (let j = 0; j < nc; j++) for (let i = 0; i < nc; i++) { const k = idx[j * nc + i]; if (i + 1 < nc && idx[j * nc + i + 1] !== k) { g.moveTo(ox + (i + 1) * px, oy + j * px); g.lineTo(ox + (i + 1) * px, oy + (j + 1) * px); } if (j + 1 < nc && idx[(j + 1) * nc + i] !== k) { g.moveTo(ox + i * px, oy + (j + 1) * px); g.lineTo(ox + (i + 1) * px, oy + (j + 1) * px); } }
        g.stroke();
        g.strokeStyle = V.alpha(C.muted, 0.5); g.strokeRect(ox + 0.5, oy + 0.5, nc * px, nc * px);
        pts.forEach((p) => { if (Math.abs(p[0]) < R && Math.abs(p[1]) < R) V.dot(g, sx(p[0]), sy(p[1]), 1.5, V.alpha(C.ink, 0.45)); });
        cb.forEach((c) => { V.dot(g, sx(c[0]), sy(c[1]), 5.5, C.bg); V.dot(g, sx(c[0]), sy(c[1]), 4, S.m === 'v' ? C.a : C.b); });
        // painel de texto
        const tx = wide ? ox + P + 22 : 16, ty = wide ? 30 : oy + P + 26, tw = wide ? w - tx - 12 : w - 32;
        const ncw = cb.length;
        T(g, (S.m === 'v' ? 'VECTOR' : 'SCALAR') + ` · ${ncw} codewords (2 dimensions)`, tx, ty, { size: 12, font: 'mono', color: S.m === 'v' ? C.a : C.b, weight: 600, maxW: tw });
        // barras de MSE
        const items = [['scalar', MSE['s' + S.R], C.b], ['vector', MSE['v' + S.R], C.a], ['limit D(R)', Math.pow(2, -2 * S.R), C.d]];
        const mxv = 0.4, bx = tx + 78, bwid = Math.max(40, tw - 78 - 56);
        items.forEach(([l, v, col], i) => { const yy = ty + 22 + i * 24; T(g, l, tx, yy + 4, { size: 11.5, color: C.ink }); g.fillStyle = V.alpha(col, 0.8); V.rr(g, bx, yy - 7, Math.max(2, (v / mxv) * bwid), 14, 4); g.fill(); T(g, cm(v, 4), bx + Math.max(2, (v / mxv) * bwid) + 6, yy + 4, { size: 11.5, font: 'mono', color: C.ink }); });
        T(g, 'MSE per dimension (σ² = 1)', tx, ty + 96, { size: 10.5, color: C.muted });
        if (wide || h - ty > 150) {
          const y2 = ty + 124;
          T(g, 'Codebook size with k = 2 bits/dim:', tx, y2, { size: 11.5, color: C.ink, weight: 600, maxW: tw });
          [['d = 2', '2⁴ = 16'], ['d = 8 (QuIP#, E8)', '2¹⁶ = 65,536'], ['d = 256', '2⁵¹²: impossible']].forEach(([a, b], i) => { T(g, a, tx, y2 + 22 + i * 19, { size: 11.5, font: 'mono', color: C.muted }); T(g, b, tx + tw, y2 + 22 + i * 19, { size: 11.5, font: 'mono', color: i === 2 ? C.c : C.ink, align: 'right' }); });
        }
      }, { static: true });
    })();

    /* =============== Figura 6: Viterbi =============== */
    (function () {
      const fig = $('#fig-viterbi'); if (!fig) return;
      // codebook de 16 valores otimizado (Lloyd sobre caminhos de Viterbi) para este brinquedo L=4, K=2
      const CB = [0.661, 1.406, -1.217, -0.256, -2.146, 2.219, -0.518, 0.546, -0.007, -0.853, 0.959, 1.559, -0.668, 0.491, 0.109, -1.446];
      const LM = [-1.510, -0.4528, 0.4528, 1.510];
      const TT = 8;
      const nearLM = (x) => LM.reduce((b, l) => (Math.abs(x - l) < Math.abs(x - b) ? l : b), LM[0]);
      function viterbi(w) {
        let best = null;
        for (let wrap = 0; wrap < 4; wrap++) {
          const J = [], B = [];
          J[0] = CB.map((c, s) => ((s >> 2) === wrap ? (w[0] - c) ** 2 : Infinity)); B[0] = CB.map(() => -1);
          for (let t = 1; t < TT; t++) {
            J[t] = new Array(16).fill(Infinity); B[t] = new Array(16).fill(-1);
            for (let s = 0; s < 16; s++) { const hi = s >> 2; for (let a = 0; a < 4; a++) { const p = (a << 2) | hi; const c = J[t - 1][p] + (w[t] - CB[s]) ** 2; if (c < J[t][s]) { J[t][s] = c; B[t][s] = p; } } }
          }
          let bs = -1, bc = Infinity; for (let s = 0; s < 16; s++) if ((s & 3) === wrap && J[TT - 1][s] < bc) { bc = J[TT - 1][s]; bs = s; }
          if (!best || bc < best.cost) { const path = [bs]; for (let t = TT - 1; t > 0; t--) path.unshift(B[t][path[0]]); best = { cost: bc, path, J, B, wrap }; }
        }
        return best;
      }
      const avg = (() => { const r = V.rng(55); let a = 0, b = 0; for (let k = 0; k < 500; k++) { const w = Array.from({ length: TT }, () => r.normal()); a += viterbi(w).cost / TT; b += w.reduce((s, x) => s + (x - nearLM(x)) ** 2, 0) / TT; } return [a / 500, b / 500]; })();
      const S = { seed: 4, all: false, reset: true, start: 0, first: true };
      let D;
      const ctl = UI.controls(fig);
      UI.button(ctl, '↻ new vector', () => { S.seed++; gen(); }, { primary: true });
      UI.toggle(ctl, { label: 'all transitions', value: false, onChange: (v) => { S.all = v; } });
      const rT = UI.readout(ctl, 'trellis MSE', '—');
      const rS = UI.readout(ctl, '2-bit scalar MSE', '—');
      UI.readout(ctl, 'mean (500 vectors)', cm(avg[0], 3) + ' vs ' + cm(avg[1], 3));
      function gen() {
        const r = V.rng(S.seed * 7919 + 13);
        const w = Array.from({ length: TT }, () => V.clamp(r.normal(), -2.3, 2.3));
        D = viterbi(w); D.w = w; D.sc = w.map(nearLM);
        D.mseT = D.cost / TT; D.mseS = w.reduce((s, x, i) => s + (x - D.sc[i]) ** 2, 0) / TT;
        rT.set(cm(D.mseT, 3)); rS.set(cm(D.mseS, 3));
        S.reset = true;
      }
      gen();
      const b2 = (v) => bin(v, 2);
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const colDur = 0.72, fwd = TT * colDur, tbDur = 1.8, hold = 3.4, cyc = fwd + tbDur + hold;
        if (S.reset) { S.start = S.first ? t - fwd - tbDur : t; S.reset = false; S.first = false; }
        const u = (t - S.start) % cyc;
        const cur = u < fwd ? Math.floor(u / colDur) : TT - 1;
        const tb = u < fwd ? 0 : Math.min(1, (u - fwd) / tbDur);
        const small = w < 560;
        const L = small ? 34 : 50, R = 16, top = 46, bot = 56;
        const pw = w - L - R, ph = h - top - bot;
        const xs = (i) => L + ((i + 0.5) * pw) / TT;
        const vy = (v) => top + (1 - (v + 2.5) / 5) * ph;
        // grade e eixo
        [-2, -1, 0, 1, 2].forEach((v) => { V.line(g, L, vy(v), L + pw, vy(v), { color: V.alpha(C.muted, v === 0 ? 0.35 : 0.12), lw: 1 }); T(g, (v > 0 ? '+' : '') + String(v).replace('-', '−'), L - 6, vy(v) + 4, { size: 10.5, font: 'mono', color: C.muted, align: 'right' }); });
        // coluna corrente
        if (u < fwd) { g.fillStyle = V.alpha(C.a, 0.07); g.fillRect(xs(cur) - pw / TT / 2, top - 6, pw / TT, ph + 12); }
        // níveis Lloyd-Max
        for (let i = 0; i < TT; i++) LM.forEach((l) => V.line(g, xs(i) - 9, vy(l), xs(i) + 9, vy(l), { color: V.alpha(C.b, 0.7), lw: 2 }));
        // todas as transições
        if (S.all) { g.strokeStyle = V.alpha(C.muted, 0.07); g.lineWidth = 1; g.beginPath(); for (let i = 1; i <= cur; i++) for (let s = 0; s < 16; s++) for (let a = 0; a < 4; a++) { const p = (a << 2) | (s >> 2); g.moveTo(xs(i - 1), vy(CB[p])); g.lineTo(xs(i), vy(CB[s])); } g.stroke(); }
        // sobreviventes
        g.strokeStyle = V.alpha(C.muted, 0.38); g.lineWidth = 1; g.beginPath();
        for (let i = 1; i <= cur; i++) for (let s = 0; s < 16; s++) { const p = D.B[i][s]; if (p >= 0 && isFinite(D.J[i][s])) { g.moveTo(xs(i - 1), vy(CB[p])); g.lineTo(xs(i), vy(CB[s])); } }
        g.stroke();
        // nós
        for (let i = 0; i < TT; i++) {
          const Js = D.J[i].filter(isFinite); const mn = Math.min(...Js), mx = Math.max(...Js);
          for (let s = 0; s < 16; s++) {
            const ok = i <= cur && isFinite(D.J[i][s]);
            const col = ok ? V.cmap(0.12 + 0.88 * (1 - (D.J[i][s] - mn) / (mx - mn || 1)), 'heat') : V.alpha(C.muted, 0.3);
            V.dot(g, xs(i), vy(CB[s]), ok ? 3.6 : 2.4, col);
          }
        }
        // alvos
        // alvos: losango vazado com contorno na cor do texto (fora do mapa de cor dos nós)
        D.w.forEach((v, i) => { const x = xs(i), y = vy(v); g.beginPath(); g.moveTo(x, y - 7); g.lineTo(x + 7, y); g.lineTo(x, y + 7); g.lineTo(x - 7, y); g.closePath(); g.fillStyle = V.alpha(C.bg, 0.55); g.fill(); g.strokeStyle = C.ink; g.lineWidth = 2; g.stroke(); });
        // caminho (traceback)
        const nShow = Math.ceil(tb * TT);
        if (nShow > 0) {
          const first = TT - nShow;
          g.strokeStyle = C.a; g.lineWidth = 3.2; g.beginPath();
          for (let i = first; i < TT; i++) { const x = xs(i), y = vy(CB[D.path[i]]); if (i === first) g.moveTo(x, y); else g.lineTo(x, y); }
          g.stroke();
          g.font = V.font(10.5, 'mono', 400); const lw4 = g.measureText('0000').width;
          for (let i = first; i < TT; i++) { const x = xs(i), y = vy(CB[D.path[i]]); V.glow(g, x, y, 12, C.a, 0.7); V.dot(g, x, y, 5, C.a); }
          for (let i = first; i < TT; i++) { const x = xs(i), y = vy(CB[D.path[i]]); V.box(g, x + 6, y - 19, lw4 + 6, 14, { r: 4, fill: V.alpha(C.bg, 0.82) }); T(g, bin(D.path[i], 4), x + 9, y - 8, { size: 10.5, font: 'mono', color: C.a }); }
        }
        if (tb >= 1) {
          D.sc.forEach((v, i) => { g.strokeStyle = C.b; g.lineWidth = 2; g.strokeRect(xs(i) + 10, vy(v) - 4, 8, 8); });
          // tail-biting
          const xa = xs(TT - 1), ya = vy(CB[D.path[TT - 1]]), xb = xs(0), yb = vy(CB[D.path[0]]), yt = top - 18;
          g.setLineDash([4, 4]); g.strokeStyle = V.alpha(C.d, 0.9); g.lineWidth = 1.5; g.beginPath(); g.moveTo(xa, ya - 8); g.lineTo(xa, yt); g.lineTo(xb, yt); g.lineTo(xb, yb - 12); g.stroke(); g.setLineDash([]);
          V.arrow(g, xb, yb - 20, xb, yb - 8, { color: C.d, lw: 1.5, head: 6 });
          T(g, 'tail-biting: s₀ reuses the last symbol (' + b2(D.wrap) + ')', (xa + xb) / 2, yt - 5, { size: 11, color: C.d, align: 'center', maxW: pw });
        } else {
          T(g, u < fwd ? `Viterbi → step ${cur + 1} of ${TT}: each node keeps the best cost that reaches it` : 'traceback: retracing the optimal path', L, top - 26, { size: small ? 11 : 12, color: C.ink, maxW: pw });
        }
        // fluxo de bits
        const yb2 = h - bot + 30;
        T(g, 'emitted bits (2 per weight):', L, yb2 - 12, { size: 10.5, color: C.muted });
        // legenda do mapa de cor dos nós: custo acumulado J
        { const gw = small ? 36 : 60, gx1 = L + pw - (small ? 40 : 46), gx0 = gx1 - gw, gy = yb2 - 20;
          for (let k = 0; k < gw; k++) { g.fillStyle = V.cmap(0.12 + 0.88 * (k / (gw - 1)), 'heat'); g.fillRect(gx0 + k, gy, 1.2, 9); }
          T(g, small ? 'J: costly' : 'cost J: expensive', gx0 - 5, yb2 - 12, { size: 10.5, color: C.muted, align: 'right' });
          T(g, 'cheap', gx1 + 5, yb2 - 12, { size: 10.5, color: C.muted }); }
        for (let i = 0; i < TT; i++) { const on = tb >= 1 || i >= TT - nShow; T(g, on ? b2(D.path[i] & 3) : '··', xs(i), yb2 + 4, { size: 12.5, font: 'mono', align: 'center', color: on ? C.a : C.muted, weight: 600 }); }
        T(g, 'state = (previous symbol, current symbol) → value c(s)', L + pw / 2, h - 8, { size: 10.5, color: C.muted, align: 'center', maxW: pw });
      });
    })();

    /* =============== Figura 7: codebook computado =============== */
    (function () {
      const fig = $('#fig-codebook'); if (!fig) return;
      const S = { kind: 'mcg', t: 5, auto: true, sym: [0xA, 3, 7, 0xF, 1, 0xC, 5, 9, 2, 0xE, 6, 0, 0xB, 4, 8, 0xD], seed: 1, tv: 5 };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'codebook', options: [{ v: 'mcg', label: 'MCG (cb1)' }, { v: 'mul1', label: 'mul1 (cb2)' }, { v: 'inst3', label: '3INST (cb0)' }], value: 'mcg', onChange: (v) => { S.kind = v; setRead(); } });
      const sl = UI.slider(ctl, { label: 'weight index', min: 0, max: 15, step: 1, value: S.t, onInput: (v) => { S.t = v; S.auto = false; tg.el.querySelector('input').checked = false; setRead(); } });
      const tg = UI.toggle(ctl, { label: 'auto-advance', value: true, onChange: (v) => { S.auto = v; } });
      UI.button(ctl, 'randomize bits', () => { const r = V.rng(++S.seed * 131); S.sym = S.sym.map(() => r.int(16)); setRead(); });
      const rV = UI.readout(ctl, 'decoded weight', '—');
      const rM = UI.readout(ctl, 'int32 marker', '—');
      const state = (t) => { const s = S.sym, m = (k) => s[(k + 16) % 16]; return (m(t - 3) << 12) | (m(t - 2) << 8) | (m(t - 1) << 4) | m(t); };
      function setRead() { const d = cbDetail(S.kind, state(S.t)); rV.set(cm(d.v, 4)); rM.set(S.kind === 'mcg' ? '−877912083' : S.kind === 'mul1' ? '−2082680531' : '— (no marker)'); }
      setRead();
      V.every(1700, () => { if (!S.auto) return; S.t = (S.t + 1) % 16; sl.input.value = S.t; const vv = sl.el.querySelector('.val'); if (vv) vv.textContent = S.t; setRead(); });
      fitH(fig, (w) => (w >= 620 ? 424 : 486));
      V.fig(fig, (g, w, h, t, dt) => {
        const C = V.C();
        const small = w < 620;
        S.tv += (S.t - S.tv) * Math.min(1, dt * 7); if (Math.abs(S.t - S.tv) > 8) S.tv = S.t;
        // fita
        const cw = Math.min(38, (w - 28) / 16), x0 = (w - 16 * cw) / 2, ty = 26, th = small ? 28 : 34;
        T(g, 'circular tile stream: 16 symbols of K = 4 bits', w / 2, 14, { size: 11, color: C.muted, align: 'center' });
        const win = [0, 1, 2, 3].map((k) => (S.t - 3 + k + 16) % 16);
        for (let i = 0; i < 16; i++) {
          const on = win.includes(i), cur = i === S.t;
          V.box(g, x0 + i * cw + 1, ty, cw - 2, th, { r: 5, fill: on ? V.alpha(C.a, cur ? 0.45 : 0.22) : V.alpha(C.muted, 0.08), stroke: on ? C.a : V.alpha(C.muted, 0.3), lw: on ? 1.6 : 1 });
          T(g, hex(S.sym[i], 1), x0 + i * cw + cw / 2, ty + (small ? th / 2 : 13), { size: small ? 12 : 13, font: 'mono', align: 'center', base: 'middle', color: C.ink, weight: on ? 700 : 400 });
          if (!small) T(g, bin(S.sym[i], 4), x0 + i * cw + cw / 2, ty + 26, { size: 9.5, font: 'mono', align: 'center', color: C.muted });
        }
        if (S.t < 3) T(g, '↺ the window wraps around (tail-biting)', x0 + 16 * cw, ty + th + 14, { size: 10.5, color: C.d, align: 'right' });
        const wx = x0 + (S.tv + 0.5) * cw; V.arrow(g, wx, ty + th + 18, wx, ty + th + 4, { color: C.c, lw: 1.5, head: 6 });
        const s = state(S.t), d = cbDetail(S.kind, s);
        // linhas de decodificação
        const fs = small ? 10.5 : 12.5, lh = small ? 38 : 36;
        let y = ty + th + 42;
        const lx = 14;
        g.font = V.font(fs, 'mono', 400); const chw = g.measureText('0').width;
        const bits = (v, n, x, yy, colorOf) => { const str = bin(v, n); for (let i = 0; i < n; i++) { const extra = Math.floor(i / 4) * chw * 0.45 + (n === 32 && i >= 16 ? chw * 1.2 : 0); T(g, str[i], x + i * chw + extra, yy, { size: fs, font: 'mono', color: colorOf ? colorOf(i) : C.ink }); } };
        const row = (label, draw) => { if (small) { T(g, label, lx, y, { size: 11, color: C.muted }); draw(lx, y + 16); y += lh + 4; } else { T(g, label, lx, y, { size: 11.5, color: C.muted }); draw(lx + 250, y); y += lh - 8; } };
        row(`weight t = ${S.t} → state s (16 bits)`, (x, yy) => { bits(s, 16, x, yy, () => C.a); T(g, '= 0x' + hex(s, 4), x + 16 * chw * 1.12 + 10, yy, { size: fs, font: 'mono', color: C.a }); });
        if (S.kind === 'mul1') {
          row('x = s × 0x83DCD12D (mod 2³²)', (x, yy) => { bits(d.x, 32, x, yy, (i) => (Math.floor(i / 8) % 2 ? C.b : C.d)); });
          row('Σ of the 4 bytes (dp4a)', (x, yy) => { T(g, `${d.bytes[3]} + ${d.bytes[2]} + ${d.bytes[1]} + ${d.bytes[0]} = ${d.sum}`, x, yy, { size: fs, font: 'mono', color: C.ink }); });
          row('0x6400 + Σ  →  FP16', (x, yy) => { T(g, `0x${hex(d.hb, 4)} = 1024 + ${d.sum} = ${d.hv}`, x, yy, { size: fs, font: 'mono', color: C.ink }); });
          row('FMA: ŵ = h × 0x1EEE + 0xC931', (x, yy) => { T(g, `ŵ = ${cm(d.v, 4)}`, x, yy, { size: fs + 1, font: 'mono', color: C.c, weight: 700 }); T(g, '0x1EEE ≈ 1/147.77 · 0xC931 = −10.383', x + (small ? 112 : 140), yy, { size: 10.5, color: C.muted, maxW: w - x - (small ? 112 : 140) - 8 }); });
        } else {
          row(S.kind === 'mcg' ? 'x = s × 0xCBAC1FED (mod 2³²)' : 'x = s × 89226354 + 64248484', (x, yy) => { bits(d.x, 32, x, yy, () => C.ink); });
          const fcol = (i) => { const k = i % 16; return k === 0 ? C.c : k <= 3 ? C.d : k <= 5 ? C.b : C.a; };
          row('y = (x & 0x8FFF8FFF) ⊕ 0x3B603B60', (x, yy) => { bits(d.y, 32, x, yy, fcol); });
          row('FP16 halves: high + low', (x, yy) => { T(g, `${cm(d.hi, 5)}  +  ${cm(d.lo, 5)}`, x, yy, { size: fs, font: 'mono', color: C.ink }); });
          row('ŵ = sum (FP16)', (x, yy) => { T(g, `ŵ = ${cm(d.v, 4)}`, x, yy, { size: fs + 1, font: 'mono', color: C.c, weight: 700 }); });
          if (!small) { const lx2 = lx + 250; T(g, 'sign', lx2, y - 2, { size: 10.5, color: C.c }); T(g, 'forced exponent 011', lx2 + 44, y - 2, { size: 10.5, color: C.d }); T(g, 'exponent', lx2 + 190, y - 2, { size: 10.5, color: C.b }); T(g, 'mantissa', lx2 + 262, y - 2, { size: 10.5, color: C.a }); }
        }
        // histograma
        const A = cbAll(S.kind);
        const hx0 = 30, hx1 = w - 16, hy1 = h - 26, hy0 = Math.max(y + (small ? 18 : 24), h - (small ? 150 : 170));
        const hw = hx1 - hx0, hh = hy1 - hy0;
        const mxh = Math.max(...A.hist);
        const bwid = hw / A.NB;
        A.hist.forEach((c, i) => { const bh = (c / mxh) * (hh - 14); g.fillStyle = V.alpha(C.b, 0.55); g.fillRect(hx0 + i * bwid + 0.5, hy1 - bh, Math.max(1, bwid - 1), bh); });
        // normal de mesma média/desvio
        g.strokeStyle = C.d; g.lineWidth = 1.8; g.beginPath();
        const binW = (A.hi - A.lo) / A.NB;
        for (let i = 0; i <= 120; i++) { const v = A.lo + (i / 120) * (A.hi - A.lo); const pdf = Math.exp(-0.5 * ((v - A.mean) / A.std) ** 2) / (A.std * Math.sqrt(2 * Math.PI)); const cnt = pdf * binW * 65536; const x = hx0 + ((v - A.lo) / (A.hi - A.lo)) * hw, yy = hy1 - (cnt / mxh) * (hh - 14); if (i) g.lineTo(x, yy); else g.moveTo(x, yy); }
        g.stroke();
        V.line(g, hx0, hy1, hx1, hy1, { color: V.alpha(C.muted, 0.5), lw: 1 });
        [-4, -2, 0, 2, 4].forEach((v) => { const x = hx0 + ((v - A.lo) / (A.hi - A.lo)) * hw; T(g, String(v).replace('-', '−'), x, hy1 + 14, { size: 10.5, font: 'mono', color: C.muted, align: 'center' }); });
        const mxp = hx0 + ((V.clamp(d.v, A.lo, A.hi) - A.lo) / (A.hi - A.lo)) * hw;
        V.line(g, mxp, hy0 + 8, mxp, hy1, { color: C.ink, lw: 2 }); V.dot(g, mxp, hy0 + 8, 4.5, C.ink);
        T(g, `65,536 states · μ = ${cm(Math.abs(A.mean) < 5e-5 ? 0 : A.mean, 4)} · σ = ${cm(A.std, 3)}`, hx1, hy0 - 2, { size: small ? 10.5 : 11.5, font: 'mono', color: C.muted, align: 'right', maxW: hw });
      });
    })();

    /* =============== Figura 8: tile e palavras =============== */
    (function () {
      const fig = $('#fig-tile'); if (!fig) return;
      const MATS = {
        gu: { t: 'gate/up of one expert', s: 'expert gate/up', din: 4096, dout: 2048, n: 37152, nt: 'all 37,152 expert matrices' },
        dn: { t: 'down of one expert', s: 'expert down', din: 2048, dout: 4096, n: 37152, nt: 'all 37,152 expert matrices' },
        tp2: { t: 'TP2 slice of gate/up', s: 'TP2 slice', din: 4096, dout: 1024, n: 37152, nt: 'per rank at TP2 (slices)' },
        dq: { t: 'drafter q_proj', s: 'draft q_proj', din: 4096, dout: 4096, n: 5, nt: '5 DFlash2 layers' },
      };
      const S = { m: 'gu', K: 4 };
      let TL;
      function buildTile() {
        const K = S.K, nw = 16 * K, r = V.rng(1000 + K * 17);
        const words = new Uint16Array(nw); for (let i = 0; i < nw; i++) words[i] = r.int(65536);
        const nb = 256 * K;
        const bit = (b) => { const bb = ((b % nb) + nb) % nb; return (words[bb >> 4] >> (15 - (bb & 15))) & 1; };
        const win = (t) => { const end = (t + 1) * K, st = end - 16; let s = 0; for (let i = 0; i < 16; i++) s = (s << 1) | bit(st + i); return { s, st }; };
        const vals = []; for (let t = 0; t < 256; t++) vals.push(mcg(win(t).s));
        TL = { words, win, vals, nb };
      }
      buildTile();
      const ctl = UI.controls(fig);
      const selM = UI.select(ctl, { label: 'matrix', options: Object.keys(MATS).map((k) => ({ v: k, label: MATS[k].s + ' [' + MATS[k].din + '→' + MATS[k].dout + ']' })), value: 'gu', onChange: (v) => { S.m = v; setRead(); } });
      selM.el.style.maxWidth = '100%'; selM.el.style.minWidth = '0'; const selEl = selM.el.querySelector('select'); if (selEl) { selEl.style.maxWidth = '100%'; selEl.style.minWidth = '0'; }
      UI.seg(ctl, { label: 'k (bits/weight)', options: [2, 3, 4, 5, 6, 8], value: 4, onChange: (v) => { S.K = v; buildTile(); setRead(); relayT(); } });
      const rW = UI.readout(ctl, 'int16 words per tile', '—');
      const rB = UI.readout(ctl, 'effective bpw', '—');
      const rTot = UI.readout(ctl, 'total', '—');
      function setRead() {
        const M = MATS[S.m], K = S.K;
        const tb = (M.din * M.dout * K) / 8, tot = tb + 2 * (M.din + M.dout) + 4;
        rW.set(String(16 * K));
        rB.set(cm((8 * tot) / (M.din * M.dout), 4));
        const all = M.n * tot;
        rTot.set((all >= 1e9 ? cm(all / 1e9, 1) + ' GB' : cm(all / 1e6, 1) + ' MB') + ' · ' + M.nt);
      }
      setRead();
      const relayT = fitH(fig, (w) => (w >= 760 ? 400 : 126 + Math.min(w - 32, 300) + 30 + (2 * S.K * 15 + 44) + 8));
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const M = MATS[S.m], K = S.K;
        const wide = w >= 760;
        const tcur = Math.floor(t * 4) % 256;
        let A, B, Cc;
        if (wide) { const top = 34, hh = h - top - 50; A = { x: 14, y: top, w: w * 0.2, h: hh - 24 }; B = { x: w * 0.26, y: top, w: Math.min(w * 0.33, hh), h: hh }; Cc = { x: B.x + B.w + 30, y: top, w: w - (B.x + B.w + 30) - 14, h: hh }; }
        else { const bs = Math.min(w - 32, 300); A = { x: 16, y: 28, w: w - 32, h: 70 }; B = { x: (w - bs) / 2, y: 126, w: bs, h: bs }; Cc = { x: 16, y: 126 + bs + 30, w: w - 32, h: h - (126 + bs + 30) - 8 }; }
        // (A) matriz
        const ar = M.dout / M.din; let mw = A.w, mh = A.w / ar; if (mh > A.h) { mh = A.h; mw = A.h * ar; }
        const mx = A.x + (A.w - mw) / 2, my = A.y + (A.h - mh) / 2;
        T(g, `(A) ${M.din / 16} × ${M.dout / 16} tiles`, A.x, A.y - 12, { size: 11, font: 'mono', color: C.muted, maxW: A.w });
        if (wide) T(g, `[${M.din} × ${M.dout}] · each cell = 16×16 tiles`, mx + mw / 2, my + mh + 16, { size: 10.5, font: 'mono', color: C.muted, align: 'center', maxW: A.w + 20 });
        else { T(g, `[${M.din} × ${M.dout}]`, mx - 12, my + mh / 2 - 4, { size: 11, font: 'mono', color: C.ink, align: 'right' }); T(g, 'cell = 16×16 tiles', mx - 12, my + mh / 2 + 12, { size: 10.5, color: C.muted, align: 'right' }); }
        g.fillStyle = V.alpha(C.d, 0.1); g.fillRect(mx, my, mw, mh); g.strokeStyle = V.alpha(C.d, 0.6); g.lineWidth = 1; g.strokeRect(mx, my, mw, mh);
        const gr = 16, gc = Math.max(2, Math.round(16 * ar));
        g.beginPath(); g.strokeStyle = V.alpha(C.d, 0.25);
        for (let i = 1; i < gr; i++) { g.moveTo(mx, my + (i * mh) / gr); g.lineTo(mx + mw, my + (i * mh) / gr); }
        for (let j = 1; j < gc; j++) { g.moveTo(mx + (j * mw) / gc, my); g.lineTo(mx + (j * mw) / gc, my + mh); }
        g.stroke();
        const ti = Math.floor(t * 0.5) % gr, tj = Math.floor(t * 0.5 * 1.7) % gc;
        const hx = mx + (tj * mw) / gc, hy = my + (ti * mh) / gr, hw2 = Math.max(4, mw / gc), hh2 = Math.max(4, mh / gr);
        g.fillStyle = V.alpha(C.a, 0.8); g.fillRect(hx, hy, hw2, hh2);
        g.setLineDash([3, 3]);
        if (wide) { V.line(g, hx + hw2, hy, B.x, B.y, { color: V.alpha(C.a, 0.35), lw: 1 }); V.line(g, hx + hw2, hy + hh2, B.x, B.y + B.h, { color: V.alpha(C.a, 0.35), lw: 1 }); }
        else { V.line(g, hx, hy + hh2, B.x, B.y, { color: V.alpha(C.a, 0.35), lw: 1 }); V.line(g, hx + hw2, hy + hh2, B.x + B.w, B.y, { color: V.alpha(C.a, 0.35), lw: 1 }); }
        g.setLineDash([]);
        if (!wide) { // no celular, o tracejado do zoom passa pelos rótulos de (A): redesenha-os com fundo
          g.font = V.font(11, 'mono', 400); const w1 = g.measureText(`[${M.din} × ${M.dout}]`).width; g.font = V.font(10.5, 'sans', 400); const w2 = g.measureText('cell = 16×16 tiles').width;
          const wm = Math.max(w1, w2) + 8; V.box(g, mx - 12 - wm + 4, my + mh / 2 - 17, wm, 34, { r: 5, fill: C.bg });
          T(g, `[${M.din} × ${M.dout}]`, mx - 12, my + mh / 2 - 4, { size: 11, font: 'mono', color: C.ink, align: 'right' }); T(g, 'cell = 16×16 tiles', mx - 12, my + mh / 2 + 12, { size: 10.5, color: C.muted, align: 'right' });
        }
        // (B) tile — rótulo com fundo, para o tracejado do zoom não cortar o texto
        { const lbB = '(B) 1 tile = 256 weights (MCG)'; g.font = V.font(11, 'mono', 400); const tw = Math.min(B.w, g.measureText(lbB).width); V.box(g, B.x - 3, B.y - 25, tw + 6, 17, { r: 4, fill: C.bg }); T(g, lbB, B.x, B.y - 12, { size: 11, font: 'mono', color: C.muted, maxW: B.w }); }
        const cs = (B.w - 15) / 16;
        const data = []; for (let i = 0; i < 16; i++) data.push(TL.vals.slice(i * 16, i * 16 + 16));
        V.matrix(g, { x: B.x, y: B.y, data, cell: cs, gap: 1, r: 1.5, colorFn: (v) => V.cmap(V.clamp(v / 3.2, -1, 1), 'div'), hiCells: [[Math.floor(tcur / 16), tcur % 16]], hiColor: C.c });
        // (C) palavras
        const nw = 16 * K, ncol = 8, nrow = nw / ncol;
        const cwd = Cc.w / ncol, chh = Math.min(22, (Cc.h - 44) / nrow);
        T(g, `(C) ${nw} int16 words · ${256 * K} bits`, Cc.x, Cc.y - 12, { size: 11, font: 'mono', color: C.muted, maxW: Cc.w });
        const { s: stt, st } = TL.win(tcur);
        const touched = new Set(); for (let i = 0; i < 16; i++) { const bb = (((st + i) % TL.nb) + TL.nb) % TL.nb; touched.add(bb >> 4); }
        for (let k = 0; k < nw; k++) {
          const r0 = Math.floor(k / ncol), c0 = k % ncol, x = Cc.x + c0 * cwd, y = Cc.y + r0 * chh;
          const on = touched.has(k);
          V.box(g, x + 1, y + 1, cwd - 2, chh - 2, { r: 3, fill: on ? V.alpha(C.a, 0.4) : V.alpha(C.d, 0.1), stroke: on ? C.a : V.alpha(C.d, 0.3), lw: 1 });
          if (cwd >= 34 && chh >= 12) T(g, hex(TL.words[k], 4), x + cwd / 2, y + chh / 2 + 0.5, { size: Math.min(11, chh * 0.62), font: 'mono', align: 'center', base: 'middle', color: C.ink });
        }
        const yb = Cc.y + nrow * chh + 18;
        T(g, `weight ${tcur}: window = 0x${hex(stt, 4)} → ŵ = ${cm(TL.vals[tcur], 3)}`, Cc.x, yb, { size: 11.5, font: 'mono', color: C.c, maxW: Cc.w });
        T(g, `each weight: 16-bit window, ${K} new bits`, Cc.x, yb + 17, { size: 11, color: C.muted, maxW: Cc.w });
      });
    })();

    /* =============== Figura 9: kernel e despacho por M =============== */
    (function () {
      const fig = $('#fig-kernel'); if (!fig) return;
      const S = { lv: 2, top: 'lin' };
      const ctl = UI.controls(fig);
      const Mof = () => Math.max(1, Math.round(Math.pow(2, S.lv)));
      UI.slider(ctl, { label: 'M (rows in the GEMM)', min: 0, max: 13, step: 0.05, value: S.lv, fmt: (v) => String(Math.max(1, Math.round(Math.pow(2, v)))), onInput: (v) => { S.lv = v; setRead(); } });
      UI.seg(ctl, { label: 'top shows', options: [{ v: 'lin', label: 'dense linear' }, { v: 'exp', label: 'expert' }], value: 'lin', onChange: (v) => { S.top = v; } });
      const rP = UI.readout(ctl, 'EXL3 linear', '—');
      const rE = UI.readout(ctl, 'EXL3 expert', '—');
      function setRead() { const M = Mof(); rP.set(M <= 144 ? 'decodes in the register' : 'rebuilds FP16 + dense GEMM'); rE.set(M <= 32 ? 'fused exl3_moe (thin)' : 'grouped E3 (fat), no rebuild'); }
      setRead();
      fitH(fig, (w) => (w >= 720 ? 440 : 572));
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const M = Mof();
        const exp = S.top === 'exp';
        const rec = !exp && M > 144, fat = exp && M > 32;
        const wide = w >= 720;
        const topH = wide ? 200 : 290;
        // ---- pipeline (topo)
        let boxes;
        if (rec) boxes = [{ t: 'DRAM', s: 'trellis 0.5 B/weight', k: 'd' }, { t: 'rebuild', s: 'trellis → FP16, 1×', k: 'b' }, { t: 'DRAM', s: 'FP16 2 B/weight', k: 'b' }, { t: 'dense GEMM', s: 'FP16 tensor core', k: 'e' }, { t: 'output', s: 'y (FP16)', k: 'c' }];
        else if (fat) boxes = [{ t: 'DRAM', s: 'trellis 0.5 B/weight', k: 'd' }, { t: 'SMEM', s: 'B tile: 64 words', k: 'd' }, { t: 'registers', s: 'dequant 1× per tile', k: 'a' }, { t: 'tensor core', s: `tile reused by ${M} rows`, k: 'e' }, { t: 'epilogue', s: 'H₁₂₈ ⊙ svh → out[token]', k: 'c' }];
        else boxes = [{ t: 'DRAM', s: 'trellis 0.5 B/weight', k: 'd' }, { t: 'SMEM', s: 'tiles of 16k words', k: 'd' }, { t: 'registers', s: 'window → MCG → FP16', k: 'a' }, { t: 'tensor core', s: 'mma.sync m16n8k16', k: 'e' }, { t: 'epilogue', s: 'H₁₂₈ (shuffles) ⊙ svh', k: 'c' }];
        const nb = boxes.length;
        let bx;
        if (wide) { const cw = (w - 24) / nb; bx = boxes.map((b, i) => ({ x: 12 + cw * i + 8, y: 92, w: cw - 16, h: 64 })); }
        else { const bw0 = Math.min(260, w * 0.6); bx = boxes.map((b, i) => ({ x: 12, y: 40 + i * 50, w: bw0, h: 40 })); }
        const ctr = (b) => [b.x + b.w / 2, b.y + b.h / 2];
        for (let i = 0; i < nb - 1; i++) { const a = ctr(bx[i]), b = ctr(bx[i + 1]); V.line(g, a[0], a[1], b[0], b[1], { color: V.alpha(C.muted, 0.4), lw: 1.5, dash: [4, 5], dashOffset: -t * 20 }); }
        // partículas: pequenas = 4 bits/peso, grandes = FP16
        for (let i = 0; i < nb - 1; i++) {
          const a = ctr(bx[i]), b = ctr(bx[i + 1]);
          const n = 4;
          for (let k = 0; k < n; k++) {
            const u = (t * 0.55 + k / n + i * 0.13) % 1; const p = V.along(a[0], a[1], b[0], b[1], u);
            const big = rec ? i >= 1 : i >= 2; const col = C[boxes[i].k];
            const sz = big ? 7 : 3.5;
            g.fillStyle = col; V.rr(g, p[0] - sz / 2, p[1] - sz / 2, sz, sz, 1.5); g.fill();
          }
        }
        bx.forEach((b, i) => { const col = C[boxes[i].k]; V.box(g, b.x, b.y, b.w, b.h, { r: 10, fill: C.bg }); V.box(g, b.x, b.y, b.w, b.h, { r: 10, fill: V.alpha(col, 0.15), stroke: col, lw: 1.5 }); T(g, boxes[i].t, b.x + b.w / 2, b.y + b.h / 2 - 8, { size: 12.5, weight: 600, align: 'center', base: 'middle', color: C.ink, maxW: b.w - 6 }); T(g, boxes[i].s, b.x + b.w / 2, b.y + b.h / 2 + 11, { size: 10.5, align: 'center', base: 'middle', color: C.muted, maxW: b.w - 6 }); });
        // ativação entrando
        const tc = bx[3];
        const actL = rec ? `activations: ${M} rows` : fat ? `${M} rows of H₁₂₈(x ⊙ suh)` : 'rotated x: H₁₂₈(x ⊙ suh)';
        const nAct = fat ? 6 : 3;
        if (wide) {
          const ax = tc.x + tc.w / 2, ay0 = 50;
          T(g, actL, ax, ay0 - 4, { size: 11, font: 'mono', color: C.b, align: 'center', maxW: tc.w + 60 });
          V.arrow(g, ax, ay0 + 2, ax, tc.y - 2, { color: C.b, lw: 1.6 });
          for (let k = 0; k < nAct; k++) { const u = (t * 0.8 + k / nAct) % 1; const yy = V.lerp(ay0 + 4, tc.y - 4, u); g.fillStyle = C.b; V.rr(g, ax - 3, yy - 3, 6, 6, 1.5); g.fill(); }
        } else {
          const ay = tc.y + tc.h / 2, ax0 = w - 14, ax1 = tc.x + tc.w + 2;
          T(g, rec || fat ? `x: ${M} rows` : 'rotated x', ax0, ay - 10, { size: 11, font: 'mono', color: C.b, align: 'right' });
          V.arrow(g, ax0, ay, ax1, ay, { color: C.b, lw: 1.6 });
          for (let k = 0; k < nAct; k++) { const u = (t * 0.8 + k / nAct) % 1; const xx = V.lerp(ax0 - 4, ax1 + 4, u); g.fillStyle = C.b; V.rr(g, xx - 3, ay - 3, 6, 6, 1.5); g.fill(); }
        }
        let head;
        if (!exp) head = rec ? (wide ? `LinearEXL3 (dense), M = ${M} > 144: reconstructing once and running a dense GEMM is cheaper (compute dominates)` : `dense, M = ${M} > 144: rebuild 1× + dense GEMM`)
          : (wide ? `LinearEXL3 (dense), M = ${M} ≤ 144: no BF16 weight exists in memory; reads 4 bits per weight` : `dense, M = ${M} ≤ 144: reads 4 bits/weight, no BF16`);
        else head = fat ? (wide ? `Expert, M = ${M} > 32 (fat): E2/E3 decode each tile once and reuse it across the ${M} rows, without reconstructing` : `fat expert (${M} > 32): E3, no rebuild`)
          : (wide ? `Expert, M = ${M} ≤ 32 (thin): a single fused exl3_moe per layer, graph-safe, reading 4 bits per weight` : `thin expert (${M} ≤ 32): 1 exl3_moe/layer`);
        T(g, head, 12, 16, { size: wide ? 12.5 : 11, color: rec ? C.b : fat ? C.d : C.a, maxW: w - 24 });
        // ---- trilhas de despacho
        const lanes = [
          { t: 'EXL3 linear (dense)', thr: 144, a: 'register', b: 'rebuild / BF16 copy', ca: C.a, cb: C.b, note: 'AUTO_RECONSTRUCT_THRESHOLD' },
          { t: 'EXL3 experts (rows per expert)', thr: 32, a: 'exl3_moe (thin)', b: 'E3 on the trellis (fat)', ca: C.a, cb: C.d, note: 'EXL3_TEMP_ROWS_FUSED' },
          { t: 'dense FP8: KDA in_proj', thr: 512, a: 'Marlin W8A16', b: 'BF16 copy · cuBLAS', ca: C.e, cb: C.b, note: 'KDA_BF16_LARGE_M_MIN_M' },
        ];
        const lx = wide ? 220 : 14, lx1 = w - 18, ly0 = topH + 36, lgap = wide ? 50 : (h - ly0 - 50) / 3;
        const xm = (m) => lx + (Math.log2(Math.max(1, m)) / 13) * (lx1 - lx);
        const xc = xm(M);
        const laneY = (i) => ly0 + i * lgap + (wide ? 0 : 16);
        // 1) barras
        lanes.forEach((L, i) => {
          const y = laneY(i), on = M <= L.thr, xa = xm(L.thr);
          g.fillStyle = V.alpha(L.ca, on ? 0.85 : 0.25); V.rr(g, lx, y - 8, xa - lx, 16, 5); g.fill();
          g.fillStyle = V.alpha(L.cb, on ? 0.25 : 0.85); V.rr(g, xa, y - 8, lx1 - xa, 16, 5); g.fill();
        });
        // 2) marcador de M (antes dos textos, para não cobrir os rótulos)
        if (wide) V.line(g, xc, ly0 - 22, xc, laneY(2) + 26, { color: V.alpha(C.c, 0.9), lw: 2 });
        else { lanes.forEach((L, i) => { const y = laneY(i); V.line(g, xc, y - 11, xc, y + 11, { color: V.alpha(C.c, 0.9), lw: 2.5 }); }); V.line(g, xc, ly0 - 22, xc, ly0 - 10, { color: C.c, lw: 2 }); }
        // 3) textos
        lanes.forEach((L, i) => {
          const y = laneY(i), on = M <= L.thr, xa = xm(L.thr);
          T(g, L.t, wide ? 14 : lx, wide ? y + 5 : y - 12, { size: 11.5, color: C.ink, weight: 600, maxW: wide ? 200 : w - 30 });
          const fs = wide ? 11 : 10.5;
          T(g, L.a, lx + 6, y + 4, { size: fs, color: on ? C.bg : C.ink, maxW: xa - lx - 8, weight: on ? 700 : 400 });
          T(g, L.b, xa + 6, y + 4, { size: fs, color: !on ? C.bg : C.ink, maxW: lx1 - xa - 8, weight: !on ? 700 : 400 });
          // limiar: afasta o número quando o marcador cai em cima dele
          const near = Math.abs(xc - xa) < 16;
          T(g, String(L.thr), near ? xa + (xc > xa ? -10 : 10) : xa, wide ? y - 12 : y + 21, { size: 10.5, font: 'mono', color: C.muted, align: near ? (xc > xa ? 'right' : 'left') : 'center' });
        });
        // eixo e marcadores
        const ya = laneY(2) + (wide ? 26 : 36);
        V.line(g, lx, ya, lx1, ya, { color: V.alpha(C.muted, 0.5), lw: 1 });
        [1, 4, 16, 64, 256, 1024, 8192].forEach((m) => { const x = xm(m); V.line(g, x, ya - 3, x, ya + 3, { color: C.muted, lw: 1 }); T(g, m >= 1024 ? m / 1024 + 'k' : String(m), x, ya + 15, { size: 10.5, font: 'mono', color: C.muted, align: 'center' }); });
        [[32, 'decode 4×8'], [7168, 'chunk 7,168']].forEach(([m, l]) => { const x = xm(m); T(g, '▲ ' + l, x, ya + 29, { size: 10.5, color: C.d, align: m > 1000 ? 'right' : 'center' }); });
        if (!wide) V.line(g, xc, ya - 6, xc, ya + 6, { color: C.c, lw: 2.5 });
        V.dot(g, xc, ly0 - 22, 4, C.c);
        T(g, 'M = ' + M, xc, ly0 - 28, { size: 11, font: 'mono', color: C.c, align: xc > w - 60 ? 'right' : xc < lx + 30 ? 'left' : 'center' });
      });
    })();

    /* =============== Figura 9b: um tile decodificado, quantos tokens? (TensorFold) =============== */
    (function () {
      const fig = $('#fig-tf-operandos'); if (!fig) return;
      const S = { mode: 'd', lv: Math.log2(100) };
      const ctl = UI.controls(fig);
      const Rof = () => Math.max(1, Math.min(256, Math.round(Math.pow(2, S.lv))));
      UI.seg(ctl, { label: 'kernel', options: [{ v: 'd', label: 'decode: Y = XW' }, { v: 'p', label: 'prompt: Yᵀ = WᵀXᵀ' }], value: 'd', onChange: (v) => { S.mode = v; setRead(); } });
      UI.slider(ctl, { label: 'tokens routed to the expert', min: 0, max: 8, step: 0.05, value: S.lv, fmt: (v) => String(Math.max(1, Math.round(Math.pow(2, v)))), onInput: (v) => { S.lv = v; setRead(); } });
      const rN = UI.readout(ctl, 'decodes of the tile', '—');
      const rM = UI.readout(ctl, 'mma per decode', '—');
      function setRead() {
        const R = Rof(), nd = Math.ceil(R / 16), np = Math.ceil(R / 64);
        if (S.mode === 'd') { rN.set(`${nd} (prompt: ${np})`, nd > np ? 'warn' : 'ok'); rM.set('2 (up to 16 rows)'); }
        else { rN.set(`${np} (decode: ${nd})`, 'ok'); rM.set('up to 8 (64 tokens)'); }
      }
      setRead();
      // 256 real MCG weights, from random 16-bit states
      const rr = V.rng(2207);
      const TILE = Array.from({ length: 256 }, () => mcg(rr.int(65536)));
      const TG = 0.55; // seconds per decode in the animation
      fitH(fig, (w) => (w >= 640 ? 365 : 545));
      V.fig(fig, (g, w, h, t) => {
        if (w < 160 || h < 160) return; // canvas not sized yet
        const C = V.C();
        const R = Rof(), dec = S.mode === 'd';
        const cap = dec ? 16 : 64, nG = Math.ceil(R / cap);
        const nd = Math.ceil(R / 16), np = Math.ceil(R / 64);
        const wide = w >= 640;
        // ---- header
        const head = dec ? 'Decode kernel, Y = XW: the tile is operand B, and each decode serves an item of up to 16 rows (2 mma)'
          : 'Prompt kernel, Yᵀ = WᵀXᵀ: the tile is operand A, and each decode serves a pass of up to 64 tokens (8 mma)';
        const headH = wrapTop(g, head, 14, 18, w - 28, 16, { size: wide ? 12.5 : 11.5, weight: 600, color: dec ? C.d : C.a }) * 16;
        // ---- geometry
        const ts = wide ? Math.min(150, Math.floor(w * 0.17)) : Math.min(112, Math.floor(w * 0.3));
        const tx = 14, ty = headH + 34;
        const cs = ts / 16;
        const X0 = wide ? tx + ts + 56 : 14, X1 = w - 14;
        const cw = (X1 - X0) / 64, ch = wide ? 15 : 13, rp = wide ? 46 : 44;
        const rowsTop = wide ? ty + 4 : ty + ts + 34;
        // ---- animation: which pass/item is being served
        const kAct = Math.floor(t / TG) % nG, u = (t % TG) / TG;
        // ---- decoded tile
        T(g, 'decoded tile (registers)', tx, ty - 9, { size: 11, color: C.muted, maxW: wide ? ts + 50 : w - 28 });
        if (u < 0.3) V.glow(g, tx + ts / 2, ty + ts / 2, ts * 0.75, dec ? C.d : C.a, 0.35 * (1 - u / 0.3));
        for (let i = 0; i < 16; i++) for (let j = 0; j < 16; j++) {
          g.fillStyle = V.cmap(V.clamp(TILE[i * 16 + j] / 2.6, -1, 1), 'div');
          g.fillRect(tx + j * cs + 0.5, ty + i * cs + 0.5, cs - 1, cs - 1);
        }
        g.strokeStyle = V.alpha(dec ? C.d : C.a, 0.9); g.lineWidth = 2; g.strokeRect(tx - 1, ty - 1, ts + 2, ts + 2);
        if (dec) { V.line(g, tx + ts / 2, ty - 3, tx + ts / 2, ty + ts + 3, { color: C.ink, lw: 1.5, dash: [4, 3] }); }
        const role = dec ? ['operand B: 2 fragments k16 × n8', '(columns 0–7 and 8–15)'] : ['operand A of Wᵀ: m16 × k16', '(the 16 output columns)'];
        const words = '64 int16 words → 256 FP16';
        if (wide) {
          T(g, role[0], tx, ty + ts + 18, { size: 11, color: C.ink, maxW: ts + 50 });
          T(g, role[1], tx, ty + ts + 33, { size: 10.5, color: C.muted, maxW: ts + 50 });
          T(g, words, tx, ty + ts + 50, { size: 10.5, font: 'mono', color: C.muted, maxW: ts + 50 });
        } else {
          const lx = tx + ts + 12, lw = w - lx - 14;
          wrapTop(g, role[0], lx, ty + 12, lw, 14, { size: 11, color: C.ink });
          wrapTop(g, role[1], lx, ty + 44, lw, 14, { size: 10.5, color: C.muted });
          wrapTop(g, words, lx, ty + 76, lw, 14, { size: 10.5, font: 'mono', color: C.muted });
        }
        // ---- tokens grouped into items (16) or passes (64)
        T(g, `${R} token${R > 1 ? 's' : ''} routed to this expert, in ${nG} ${dec ? (nG > 1 ? 'items' : 'item') : (nG > 1 ? 'passes' : 'pass')} of up to ${cap}`, X0, rowsTop - 9, { size: 11, color: C.muted, maxW: X1 - X0 });
        const GC = [C.b, C.a];
        let act = null;
        for (let k = 0; k < nG; k++) {
          const s0 = k * cap, row = Math.floor(s0 / 64), col = s0 % 64;
          const gx = X0 + col * cw, gy = rowsTop + row * rp, gw = cap * cw;
          const nk = Math.min(cap, R - s0);
          const state = k < kAct ? 1 : k === kAct ? 2 : 0; // 1 = already served, 2 = now
          const col0 = GC[k % 2];
          for (let j = 0; j < cap; j++) {
            const x = gx + j * cw;
            if (j < nk) { g.fillStyle = V.alpha(col0, state === 2 && u > 0.45 ? 0.9 : state === 1 ? 0.55 : 0.3); g.fillRect(x + 0.5, gy, Math.max(1, cw - 1), ch); }
            else { g.strokeStyle = V.alpha(C.muted, 0.5); g.lineWidth = 1; g.setLineDash([2, 2]); g.strokeRect(x + 1, gy + 0.5, Math.max(1, cw - 2), ch - 1); g.setLineDash([]); }
          }
          g.strokeStyle = V.alpha(col0, state === 2 ? 1 : 0.6); g.lineWidth = state === 2 ? 2 : 1;
          V.rr(g, gx - 0.5, gy - 2, gw, ch + 4, 4); g.stroke();
          if (state === 2) act = { gx, gy, gw, nk };
        }
        const mmaLabel = (s, x, y) => { g.font = V.font(10.5); const tw = g.measureText(s).width; T(g, s, Math.max(X0, Math.min(x, X1 - tw)), y, { size: 10.5, color: C.e, maxW: X1 - X0 }); };
        // pulse: one decode of the tile traveling to the current item/pass
        if (act) {
          const sx = wide ? tx + ts + 4 : tx + ts / 2, sy = wide ? ty + ts / 2 : ty + ts + 4;
          const ex = act.gx + Math.min(act.gw, 60) / 2, ey = act.gy - 3;
          V.line(g, sx, sy, ex, ey, { color: V.alpha(dec ? C.d : C.a, 0.35), lw: 1.2, dash: [4, 4] });
          if (u <= 0.45) { const p = V.along(sx, sy, ex, ey, V.ease.inOut(u / 0.45)); V.glow(g, p[0], p[1], 12, dec ? C.d : C.a, 1); V.dot(g, p[0], p[1], 4, dec ? C.d : C.a); }
          if (u > 0.45) {
            const by = act.gy + ch + 4;
            if (dec) {
              for (let m = 0; m < 2; m++) { g.fillStyle = V.alpha(C.e, 0.85); V.rr(g, act.gx, by + m * 6, act.gw - 1, 4.5, 2); g.fill(); }
              mmaLabel('2 mma: the 16 rows × columns 0–7, then 8–15', act.gx, by + 19);
            } else {
              for (let m = 0; m < 8; m++) { const used = m * 8 < act.nk; g.fillStyle = V.alpha(C.e, used ? 0.85 : 0.25); V.rr(g, act.gx + m * 8 * cw + 1, by, 8 * cw - 2, 9, 2); g.fill(); }
              mmaLabel('8 mma, each one 8 tokens × the 16 columns', act.gx, by + 19);
            }
          }
        }
        // ---- bars: decodes of the same tile in both kernels
        const nRowsMax = 4;
        const bTop = wide ? Math.max(ty + ts + 84, rowsTop + nRowsMax * rp + 8) : rowsTop + nRowsMax * rp + 10;
        const bars = [
          { l: 'decode · ⌈R/16⌉', n: nd, c: C.d, on: dec },
          { l: 'prompt · ⌈R/64⌉', n: np, c: C.a, on: !dec },
        ];
        const lw0 = wide ? 150 : 0, bx0 = wide ? 14 + lw0 : 14, bx1 = w - (wide ? 150 : 14);
        const bh = wide ? 13 : 11, bgap = wide ? 24 : 36;
        T(g, 'decodes of this tile for the same tokens', 14, bTop, { size: 11, color: C.muted, maxW: w - 28 });
        bars.forEach((b, i) => {
          const y = bTop + 12 + i * bgap + (wide ? 0 : 14);
          const len = ((bx1 - bx0) * b.n) / 16;
          T(g, b.l, wide ? 14 : bx0, wide ? y + bh - 2 : y - 4, { size: 11, font: 'mono', color: b.on ? C.ink : C.muted, weight: b.on ? 700 : 400 });
          g.fillStyle = V.alpha(C.muted, 0.12); V.rr(g, bx0, y, bx1 - bx0, bh, 4); g.fill();
          g.fillStyle = V.alpha(b.c, b.on ? 0.9 : 0.35); V.rr(g, bx0, y, Math.max(3, len), bh, 4); g.fill();
          const lbl = `${b.n} decode${b.n > 1 ? 's' : ''}`;
          if (wide) T(g, lbl, bx1 + 10, y + bh - 2, { size: 11, color: b.on ? C.ink : C.muted, weight: b.on ? 700 : 400 });
          else T(g, lbl, bx1, y - 4, { size: 10.5, color: b.on ? C.ink : C.muted, align: 'right', weight: b.on ? 700 : 400 });
        });
        // ---- footer: each kernel's bit rule
        const foot = dec ? 'decode: sums split-K and warp partials, with the split fixed by the weight shape (decode\'s own bits)'
          : 'prompt: each output is one FP32 chain over ascending k, no split-K: the same bits for any slicing of the prompt';
        if (wide) T(g, foot, 14, h - 10, { size: 10.5, color: C.muted, maxW: w - 28 });
        else wrapTop(g, foot, 14, h - 34, w - 28, 14, { size: 10.5, color: C.muted });
      });
    })();

    /* =============== Figura 10: fatiamento TP dos tensores EXL3 =============== */
    (function () {
      const fig = $('#fig-tp'); if (!fig) return;
      const S = { tp: 2 };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'ranks', options: [{ v: 2, label: 'TP = 2' }, { v: 3, label: 'TP = 3' }], value: 2, onChange: (v) => { S.tp = v; setRead(); } });
      const rT = UI.readout(ctl, 'tile columns per rank', '—');
      const rH = UI.readout(ctl, '128-wide blocks per rank', '—');
      const rB = UI.readout(ctl, 'bytes of 1 expert per rank', '—');
      // bytes de um expert K4 (gate, up, down) por rank: treliça fatiada + suh/svh (fatiados ou replicados) + marcador
      const expertBytes = (tp) => { const I = 2048 / tp, Hd = 4096; const gu = (Hd * I * 4) / 8 + 2 * Hd + 2 * I + 4; const dn = (I * Hd * 4) / 8 + 2 * I + 2 * Hd + 4; return 2 * gu + dn; };
      function setRead() {
        const tp = S.tp, ok = 128 % tp === 0;
        rT.set(ok ? `${128 / tp} of 128 ✓` : `${cm(128 / tp, 2)} ✗`);
        rH.set(ok ? `${16 / tp} of 16 ✓` : `${cm(16 / tp, 2)} ✗`);
        rB.set(ok ? `${cm(expertBytes(tp) / 1048576, 2)} MiB (of ${cm(expertBytes(1) / 1048576, 2)})` : 'no slicing: EP');
      }
      setRead();
      fitH(fig, (w) => (w >= 700 ? 430 : 610));
      V.fig(fig, (g, w, h, t) => {
        if (w < 160 || h < 160) return; // canvas ainda sem tamanho
        const C = V.C();
        const tp = S.tp, ok = 128 % tp === 0;
        const RC = [C.a, C.b, C.d];
        const wide = w >= 700;
        // geometria: 1 unidade = 1 tile de 16
        let u, gx, gy, dx, dy, top;
        if (wide) {
          top = 70;
          u = Math.min((h - top - 64) / 256, (w - 56 - 150 - 16 - 190) / 384);
          gx = 56; gy = top; dx = gx + 128 * u + 150; dy = top + 26;
        } else {
          top = 86;
          u = Math.min(0.82, (w - 60) / 256);
          gx = 46; gy = top; dx = Math.max(30, (w - 256 * u) / 2 - 10); dy = gy + 256 * u + 84;
        }
        const gw = 128 * u, gh = 256 * u, dw = 256 * u, dh = 128 * u;
        // cabeçalho
        const head = ok ? 'TP=2: cut at 1,024 = 8 blocks of 128 per rank → each H₁₂₈ rotation stays whole within one rank'
          : 'TP=3: 2,048/3 = 682.67 → the cut falls in the middle of tile 42 and of H₁₂₈ block no. 5: the loader refuses';
        if (wide) T(g, head, 14, 18, { size: 12.5, color: ok ? C.a : C.red, weight: 600, maxW: w - 28 });
        else wrapTop(g, head, 14, 16, w - 28, 15, { size: 11.5, color: ok ? C.a : C.red, weight: 600 });
        // ---- utilidades
        const cuts = []; for (let r = 1; r < tp; r++) cuts.push((128 * r) / tp); // em tiles
        const rankOf = (k, n) => Math.min(tp - 1, Math.floor((k * tp) / n));
        const vbar = (x, y, len, n, sliced, label, lblSide) => { // barra vertical de n elementos
          if (sliced) for (let r = 0; r < tp; r++) { g.fillStyle = V.alpha(RC[r], 0.8); g.fillRect(x, y + (len * r) / tp, 8, len / tp - (r < tp - 1 ? 1 : 0)); }
          else { g.fillStyle = V.alpha(C.muted, 0.55); g.fillRect(x, y, 8, len); }
          T(g, label, x + 4, y - 6, { size: 10.5, font: 'mono', color: C.muted, align: lblSide || 'center' });
        };
        const hbar = (x, y, len, sliced, label) => {
          if (sliced) for (let r = 0; r < tp; r++) { g.fillStyle = V.alpha(RC[r], 0.8); g.fillRect(x + (len * r) / tp, y, len / tp - (r < tp - 1 ? 1 : 0), 7); }
          else { g.fillStyle = V.alpha(C.muted, 0.55); g.fillRect(x, y, len, 7); }
          if (label) T(g, label, x + len + 6, y + 7, { size: 10.5, font: 'mono', color: C.muted });
        };
        const blockGrid = (x, y, cols, rows) => { // linhas a cada 8 tiles (bloco de 128)
          g.beginPath(); g.strokeStyle = V.alpha(C.ink, 0.12); g.lineWidth = 1;
          for (let j = 8; j < cols; j += 8) { g.moveTo(x + j * u + 0.5, y); g.lineTo(x + j * u + 0.5, y + rows * u); }
          for (let i = 8; i < rows; i += 8) { g.moveTo(x, y + i * u + 0.5); g.lineTo(x + cols * u, y + i * u + 0.5); }
          g.stroke();
        };
        const badBlock = (x, y, along, len, cut) => { // destaca o bloco de 128 e o tile partidos por um corte
          const b0 = Math.floor(cut / 8) * 8, t0 = Math.floor(cut);
          g.fillStyle = V.alpha(C.red, 0.28);
          if (along === 'col') { g.fillRect(x + b0 * u, y, 8 * u, len); g.fillStyle = V.alpha(C.red, 0.6); g.fillRect(x + t0 * u, y, Math.max(1.5, u), len); }
          else { g.fillRect(x, y + b0 * u, len, 8 * u); g.fillStyle = V.alpha(C.red, 0.6); g.fillRect(x, y + t0 * u, len, Math.max(1.5, u)); }
        };
        // ---- gate/up: [256, 128, 64], column-parallel
        const xbX = gx - 34, suX = gx - 18;
        vbar(xbX, gy, gh, 4096, false, 'x');
        vbar(suX, gy, gh, 4096, false, 'suh');
        for (let r = 0; r < tp; r++) { const c0 = (128 * r) / tp, c1 = (128 * (r + 1)) / tp; g.fillStyle = V.alpha(RC[r], 0.2); g.fillRect(gx + c0 * u, gy, (c1 - c0) * u, gh); }
        blockGrid(gx, gy, 128, 256);
        if (!ok) cuts.forEach((c) => badBlock(gx, gy, 'col', gh, c));
        g.strokeStyle = V.alpha(C.ink, 0.45); g.lineWidth = 1; g.strokeRect(gx + 0.5, gy + 0.5, gw, gh);
        cuts.forEach((c) => { g.setLineDash([5, 4]); V.line(g, gx + c * u, gy - 16, gx + c * u, gy + gh + 20, { color: ok ? C.ink : C.red, lw: 1.6 }); g.setLineDash([]); });
        hbar(gx, gy - 14, gw, true, wide ? 'svh [2048]' : 'svh');
        T(g, wide ? 'gate/up · trellis [256, 128, 64]' : 'gate/up [256,128,64]', gx, gy - 26, { size: 11.5, font: 'mono', color: C.ink, weight: 600 });
        // h: saída do gate/up, fatiada por rank
        const hY = gy + gh + 12;
        hbar(gx, hY, gw, true, null);
        T(g, 'h', gx - 6, hY + 7, { size: 10.5, font: 'mono', color: C.muted, align: 'right' });
        // ---- down: [128, 256, 64], row-parallel
        const sdX = dx - 16;
        for (let r = 0; r < tp; r++) { const r0 = (128 * r) / tp, r1 = (128 * (r + 1)) / tp; g.fillStyle = V.alpha(RC[r], 0.2); g.fillRect(dx, dy + r0 * u, dw, (r1 - r0) * u); }
        blockGrid(dx, dy, 256, 128);
        if (!ok) cuts.forEach((c) => badBlock(dx, dy, 'row', dw, c));
        g.strokeStyle = V.alpha(C.ink, 0.45); g.lineWidth = 1; g.strokeRect(dx + 0.5, dy + 0.5, dw, dh);
        cuts.forEach((c) => { g.setLineDash([5, 4]); V.line(g, dx - 22, dy + c * u, dx + dw + 8, dy + c * u, { color: ok ? C.ink : C.red, lw: 1.6 }); g.setLineDash([]); });
        vbar(sdX, dy, dh, 2048, true, 'suh');
        hbar(dx, dy - 14, dw, false, wide ? 'svh [4096]' : 'svh');
        T(g, wide ? 'down · trellis [128, 256, 64]' : 'down [128,256,64]', dx, dy - 26, { size: 11.5, font: 'mono', color: C.ink, weight: 600 });
        // ---- fluxo h_r → linhas da down do mesmo rank (polilinhas por um canal, sem passar por cima das matrizes)
        const alongPoly = (pts, q) => { let tot = 0; const L = []; for (let i = 1; i < pts.length; i++) { const l = Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]); L.push(l); tot += l; } let d = q * tot; for (let i = 0; i < L.length; i++) { if (d <= L[i] || i === L.length - 1) { const f = L[i] ? Math.min(1, d / L[i]) : 0; return [V.lerp(pts[i][0], pts[i + 1][0], f), V.lerp(pts[i][1], pts[i + 1][1], f)]; } d -= L[i]; } return pts[pts.length - 1]; };
        for (let r = 0; r < tp; r++) {
          const sx = gx + ((r + 0.5) * gw) / tp, sy = hY + 8, ex = sdX - 2, ey = dy + ((r + 0.5) * dh) / tp;
          let pts;
          if (wide) { const yA = hY + 12 + (tp - 1 - r) * 6, xc = gx + gw + 40 + (tp - 1 - r) * 16; pts = [[sx, sy], [sx, yA], [xc, yA], [xc, ey], [ex, ey]]; }
          else { const yA = hY + 14 + r * 7, xc = 34 - r * 9; pts = [[sx, sy], [sx, yA], [xc, yA], [xc, ey], [ex, ey]]; }
          g.beginPath(); g.moveTo(pts[0][0], pts[0][1]); for (let i = 1; i < pts.length - 1; i++) g.arcTo(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1], 4); g.lineTo(ex, ey);
          g.strokeStyle = V.alpha(RC[r], ok ? 0.6 : 0.25); g.lineWidth = 2; g.setLineDash([5, 5]); g.lineDashOffset = -t * 20; g.stroke(); g.setLineDash([]);
          V.arrow(g, ex - 7, ey, ex, ey, { color: V.alpha(RC[r], ok ? 0.9 : 0.3), lw: 2, head: 6 });
          if (ok) for (let k = 0; k < 3; k++) { const q = (t * 0.25 + k / 3 + r * 0.17) % 1; const p = alongPoly(pts, q); g.fillStyle = RC[r]; V.rr(g, p[0] - 3.5, p[1] - 3.5, 7, 7, 2); g.fill(); }
        }
        // ---- somas parciais e all-reduce
        const pY = dy + dh + 16;
        for (let r = 0; r < tp; r++) { g.fillStyle = V.alpha(RC[r], ok ? 0.8 : 0.3); g.fillRect(dx, pY + r * 12, dw, 7); }
        T(g, wide ? 'partial y of each rank [4096]' : 'partial y', dx + dw + 6, pY + 7, { size: 10.5, font: 'mono', color: C.muted });
        const aY0 = pY + tp * 12 + 2, aY1 = aY0 + 24;
        if (ok) {
          V.arrow(g, dx + dw / 2, aY0, dx + dw / 2, aY1 - 2, { color: C.ink, lw: 1.8 });
          for (let k = 0; k < tp; k++) { const q = (t * 0.9 + k / tp) % 1; g.fillStyle = RC[k]; V.rr(g, dx + dw / 2 - 3, V.lerp(aY0, aY1 - 6, q) - 3, 6, 6, 1.5); g.fill(); }
          g.fillStyle = V.alpha(C.ink, 0.75); g.fillRect(dx, aY1 + 2, dw, 7);
          T(g, 'all-reduce (Σ) → y [4096]', dx + dw / 2 + 12, aY0 + 16, { size: 11, font: 'mono', color: C.ink });
        } else {
          T(g, '✗ no valid slice', dx + dw / 2, aY0 + 18, { size: 11.5, color: C.red, align: 'center', weight: 600 });
        }
        // ---- inset: o bloco de 128 em volta do primeiro corte
        const isz = wide ? Math.min(22, dw / 8) : Math.min(21, (w - (gx + gw) - 36) / 8);
        const ix = wide ? dx + (dw - 8 * isz) / 2 : w - 12 - 8 * isz, iy = wide ? h - 82 : gy + 30;
        const cut0 = cuts[0];
        const t0 = ok ? cut0 - 4 : Math.floor(cut0 / 8) * 8; // TP2: tiles 60–67 (fronteira 7|8); TP3: bloco 5 = tiles 40–47
        T(g, 'zoom on 1st cut (16-wide tiles):', ix, iy - 8, { size: 10.5, color: C.muted, maxW: 8 * isz + 10 });
        for (let k = 0; k < 8; k++) {
          const tile = t0 + k, bi = Math.floor(tile / 8), ri = rankOf(tile + 0.5, 128);
          V.box(g, ix + k * isz + 1, iy, isz - 2, isz - 2, { r: 3, fill: V.alpha(RC[ri], 0.3), stroke: V.alpha(RC[ri], 0.8), lw: 1 });
          T(g, String(tile), ix + k * isz + isz / 2, iy + isz / 2, { size: 10.5, font: 'mono', align: 'center', base: 'middle', color: C.ink, maxW: isz - 2 });
          if (k === 0 || Math.floor((tile - 1) / 8) !== bi) { g.strokeStyle = V.alpha(C.ink, 0.5); g.lineWidth = 1.2; g.beginPath(); g.moveTo(ix + k * isz, iy - 3); g.lineTo(ix + k * isz, iy + isz + 1); g.stroke(); }
        }
        g.strokeStyle = V.alpha(C.ink, 0.5); g.beginPath(); g.moveTo(ix + 8 * isz, iy - 3); g.lineTo(ix + 8 * isz, iy + isz + 1); g.stroke();
        const cx = ix + (cut0 - t0) * isz;
        V.line(g, cx, iy - 5, cx, iy + isz + 6, { color: ok ? C.ink : C.red, lw: 2.4 });
        if (ok) { T(g, 'block 7', ix + 2 * isz, iy + isz + 13, { size: 10.5, font: 'mono', color: C.muted, align: 'center' }); T(g, 'block 8', ix + 6 * isz, iy + isz + 13, { size: 10.5, font: 'mono', color: C.muted, align: 'center' }); }
        else T(g, 'block no. 5 (tiles 40–47)', ix + 4 * isz, iy + isz + 13, { size: 10.5, color: C.red, align: 'center', maxW: 8 * isz });
        const msg = ok ? '✓ the cut falls between blocks' : '✗ cuts tile 42 at 2/3';
        T(g, msg, ix + 4 * isz, iy + isz + 28, { size: 11, color: ok ? C.a : C.red, align: 'center', weight: 600, maxW: 8 * isz + 20 });
        // ---- rodapé
        const foot = ok ? 'colors = slice of each rank · gray = replicated on all · thin lines = 128-wide Hadamard blocks'
          : '“EXL3 TP shard: dim … is not divisible by tp=3” → GLM at TP3 uses expert parallel (96 whole experts per rank)';
        if (wide) T(g, foot, w / 2, h - 8, { size: 10.5, color: ok ? C.muted : C.red, align: 'center', maxW: w - 20 });
        else wrapTop(g, foot, 14, h - 34, w - 28, 14, { size: 10.5, color: ok ? C.muted : C.red });
      });
    })();

    /* =============== Figura 10: KLD × tamanho =============== */
    (function () {
      const fig = $('#fig-kld'); if (!fig) return;
      const PTS = [
        { l: 'TR3 K6 (6 bpw)', gb: 254, k: 0.013723, v44: false, d: 'Another checkpoint (malaiwah/GLM-5.3-Flash-TR3-6bpw). More bytes, less KLD.' },
        { l: 'Official FP8 (cross-stack)', gb: 328, k: 0.020615, v44: false, d: 'Official FP8, logits from a different inference stack.' },
        { l: 'EXL3 4 bpw (served)', gb: 176, k: 0.024555, v44: true, star: true, d: 'The recipe checkpoint: uniform-K4 on the routed experts, the rest BF16.' },
        { l: 'Official FP8 (v44 stack)', gb: 328, k: 0.024629, v44: true, d: 'Official FP8, same stack as EXL3: ~1.00× the EXL3 4 bpw KLD.' },
        { l: 'NVFP4 (v44 stack)', gb: 180, k: 0.060535, v44: true, d: 'Same size as EXL3, scalar microscaling format: 2.5× the KLD.' },
      ];
      const S = { f: 'all', hover: -1 };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'show', options: [{ v: 'all', label: 'all' }, { v: 'v44', label: 'same stack (v44)' }], value: 'all', onChange: (v) => { S.f = v; H.redraw(); } });
      const H = V.fig(fig, (g, w, h, t, dt, St, Hh) => {
        const C = V.C();
        const small = w < 520;
        const L = small ? 48 : 64, R = 18, top = 22, bot = 44;
        const pw = w - L - R, ph = h - top - bot;
        const X = (gb) => L + ((gb - 150) / 200) * pw, Y = (k) => top + (1 - k / 0.066) * ph;
        [0, 0.02, 0.04, 0.06].forEach((k) => { V.line(g, L, Y(k), L + pw, Y(k), { color: V.alpha(C.muted, 0.14), lw: 1 }); T(g, cm(k, 2), L - 6, Y(k) + 4, { size: 10.5, font: 'mono', color: C.muted, align: 'right' }); });
        [150, 200, 250, 300, 350].forEach((gb) => { T(g, gb + ' GB', X(gb), h - bot + 16, { size: 10.5, font: 'mono', color: C.muted, align: gb === 350 ? 'right' : gb === 150 ? 'left' : 'center' }); });
        T(g, 'checkpoint size', L + pw / 2, h - 8, { size: 11, color: C.muted, align: 'center' });
        T(g, 'KLD (nats)', 6, 12, { size: 11, color: C.muted });
        V.line(g, L, Y(0.024629), L + pw, Y(0.024629), { color: V.alpha(C.d, 0.8), lw: 1.3, dash: [5, 4] });
        // 54%
        const yb = Y(0.024629) - 44;
        V.line(g, X(176), yb, X(328), yb, { color: V.alpha(C.a, 0.8), lw: 1.3 });
        V.line(g, X(176), yb - 4, X(176), yb + 4, { color: C.a, lw: 1.3 }); V.line(g, X(328), yb - 4, X(328), yb + 4, { color: C.a, lw: 1.3 });
        T(g, 'same KLD with 54% of the bytes', (X(176) + X(328)) / 2, yb - 7, { size: 11, color: C.a, align: 'center' });
        const P = Hh.pointer; let hov = -1, bd = 22 * 22;
        const vis = PTS.map((p) => S.f === 'all' || p.v44);
        PTS.forEach((p, i) => { if (!vis[i]) return; const dx = P.x - X(p.gb), dy = P.y - Y(p.k); const d = dx * dx + dy * dy; if (P.inside && d < bd) { bd = d; hov = i; } });
        PTS.forEach((p, i) => {
          if (!vis[i]) return;
          const x = X(p.gb), y = Y(p.k), col = p.star ? C.a : p.l.startsWith('NVFP4') ? C.c : p.l.startsWith('Official FP8') ? C.d : C.b;
          if (p.star) V.glow(g, x, y, 18, col, 0.8);
          V.dot(g, x, y, p.star ? 8 : 6.5, col);
          if (hov === i) { g.strokeStyle = C.ink; g.lineWidth = 2; g.beginPath(); g.arc(x, y, 10, 0, Math.PI * 2); g.stroke(); }
          const right = x < L + pw * 0.62;
          const k6 = p.l.startsWith('TR3');
          const ly = p.l.startsWith('Official FP8 (v') ? y - 10 : p.l.startsWith('Official FP8 (c') ? y + 19 : p.star ? y + 22 : k6 ? y + 22 : y - 12;
          T(g, p.l, k6 ? x : right ? x + 11 : x - 11, ly, { size: small ? 10.5 : 11.5, color: C.ink, align: k6 ? 'center' : right ? 'left' : 'right', weight: p.star ? 700 : 400 });
        });
        if (hov >= 0) {
          const p = PTS[hov]; const lines = [p.l, `${p.gb} GB · KLD ${cm(p.k, 6)}`, p.d];
          const bw = Math.min(290, w - 20), bh = 66;
          let bx = X(p.gb) + 14, by = Y(p.k) + 12; if (bx + bw > w - 6) bx = X(p.gb) - bw - 14; if (by + bh > h - 6) by = Y(p.k) - bh - 12; bx = Math.max(6, bx);
          V.box(g, bx, by, bw, bh, { r: 8, fill: C.panel, stroke: V.alpha(C.ink, 0.3), lw: 1 });
          T(g, lines[0], bx + 10, by + 18, { size: 12, weight: 700, color: C.ink, maxW: bw - 20 });
          T(g, lines[1], bx + 10, by + 35, { size: 11, font: 'mono', color: C.a, maxW: bw - 20 });
          T(g, lines[2], bx + 10, by + 52, { size: 10.5, color: C.muted, maxW: bw - 20 });
        }
      }, { static: true });
    })();
  },
});

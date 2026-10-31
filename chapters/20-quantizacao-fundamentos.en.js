LA.chapter({
  id: 'quantizacao-fundamentos',
  part: 4,
  order: 20,
  title: 'Why and how to quantize',
  navTitle: 'Quantization fundamentals',
  subtitle: 'From the byte math to the quantization ruler: scale and zero-point, rounding versus clipping, granularity and outliers, weights versus activations, static and dynamic scales, calibration with the Hessian (GPTQ, AWQ), PTQ versus QAT and the precision mix of the three recipes.',
  minutes: 50,
  recipes: ['glm', 'tf', 'qwen', 'ds'],
  tags: ['quantization', 'scale', 'zero-point', 'granularity', 'outliers', 'W4A16', 'W8A8', 'calibration', 'GPTQ', 'AWQ', 'PTQ', 'QAT', 'mixed precision', 'TensorFold', 'q4'],
  goals: [
    'Quantize and dequantize by hand with a uniform ruler (scale, zero-point, symmetric and asymmetric) and measure the error with <b>MSE</b>, <b>SQNR</b> and <b>rel RMSE</b>',
    'Choose the scale <b>granularity</b> (tensor, channel, token, group, block) in the face of outliers and tensor parallelism, counting the <b>effective bits</b>',
    'Tell <b>weight-only</b> (W4A16/W8A16) apart from weights + activations (W8A8/W4A8/W4A4), and <b>static</b> from <b>dynamic</b> scales, knowing what the kernel does in each case and why atypical channels make the activation harder than the weight',
    'Explain calibration (proxy Hessian, GPTQ, AWQ, synthetic Hessian), <b>PTQ</b> versus <b>QAT</b>, and read the precision mix of the three recipes as bytes × sensitivity decisions',
  ],
  prereqs: ['numeros-no-computador', 'prefill-decode-kv', 'mixture-of-experts'],
  summary: [
    'Decode is bandwidth-bound: step time ≈ bytes read ÷ bandwidth. Quantizing cuts the bytes <b>stored</b> (GLM: 642.6 GB in BF16 → 175.6 GB in EXL3 4 bpw) and the bytes <b>read per step</b> (Qwen: ~6.9 of the ~9.2 GiB are BF16 dense weights; the served GLM re-quantizes the dense weights to FP8 at load time with <code>GLM53_DENSE_FP8=all</code>).',
    'Uniform quantization: \\(q = \\mathrm{clamp}(\\mathrm{round}(x/s)+z)\\) and \\(\\hat x = s(q-z)\\). The symmetric one (\\(z=0\\), \\(s=\\mathrm{amax}/q_{\\max}\\)) is the default for weights and for every FP8 in the recipes (\\(s = \\mathrm{amax}/448\\)).',
    'The error is rounding (≤ s/2) plus clipping. With few bits the optimal limit sits below amax, hence the “MSE-calibrated” scales. Measure with MSE, SQNR (~6 dB per bit) and rel RMSE (Qwen\'s FP8-dense: 0.024–0.027).',
    'Fine granularity isolates outliers and costs \\(b_s/g\\) bits per weight (NVFP4 4.5; MXFP4 4.25; per-channel FP8 ≈ 8.004). In TP, per-channel scales are cut like the weight, and blocks of 128 require aligned boundaries.',
    '<b>Weight-only</b> dequantizes in registers and gains bandwidth (decode). <b>W8A8/W4A8/W4A4</b> uses low-precision tensor cores and gains FLOPs (prefill), but pays for the activation QDQ: GLM rejected W8A8 for KDA (0.0266 versus 0.0025). Dynamic per-token or per-block scales need no calibration, but they don\'t solve <b>atypical channels</b> (the same large index in every token): hence SmoothQuant/AWQ, Hadamard rotations and the preference for weight-only.',
    'Calibration minimizes the output error \\((\\hat w-w)^\\top \\mathbf H(\\hat w-w)\\) (GPTQ, AWQ). Without data, DFlash2 uses a synthetic Hessian. In QAT the model trains with QDQ, and DeepSeek\'s FP8/64 KV is the training regime. Experts go to ~4 bits; routers, gates, indexers, norms, <code>kv_b_proj</code> and MTP stay in high precision.',
    'In the TensorFold version of GLM (GLM·TF), the dense weights go to affine 4 bits (<code>DENSE=q4</code>): groups of 64 with a BF16 scale and bias (4.5 effective bits) and the range chosen among 16 clips by the smallest squared error, with no data. Over FP8, prose goes from 38.9 to 44.4 tok/s and GSM8K/HumanEval tie, but short French coding prompts lose the end of turn (12/48 versus 1/48): a benchmark only sees what it measures.',
  ],
  quiz: [
    { q: 'In a Qwen decode step with batch 1, why would quantizing the BF16 dense weights to FP8 yield more speed than touching the experts?', o: ['Because the experts are already in FP8', 'Because the step is bandwidth-bound and the dense weights are ~6.9 of the ~9.2 GiB read per step, while only the few chosen NVFP4 experts get read', 'Because FP8 has more precision than NVFP4', 'Because the dense weights take up most of the checkpoint'], a: 1, e: 'In decode, time ≈ bytes read ÷ bandwidth. The experts dominate the checkpoint, but each token visits only a few of them. The dense weights (GDN, attention, HC, lm_head) are read in full every step. Cutting 3.45 GiB would take the step from ≈ 9.2 to ≈ 5.7 GiB, an analytical projection not yet measured.' },
    { q: 'Quantize x = [−0.8, −0.1, 0.3, 1.2] to asymmetric INT4 (0 to 15). What is the code for 0.3?', o: ['2', '5', '8', '15'], a: 2, e: 's = 2.0/15 = 0.1333 and z = round(0.8/0.1333) = 6. So 0.3/s = 2.25 → round → 2 → +z → 8. On the way back, (8 − 6)·0.1333 = 0.267: an error of 0.033, below s/2. “2” is the value before adding z.' },
    { q: 'In a matrix with an outlier 12× larger than the rest, what changes when you swap the per-tensor scale for one scale per group of 8?', o: ['The outlier is no longer represented', 'The large error is confined to the outlier\'s group, at the cost of storing more scales (+b_s/8 bits per weight)', 'The effective bits go down', 'Nothing, because the outlier is still in the matrix'], a: 1, e: 'With one scale for everything, the outlier stretches the ruler of every weight. Per group, only the 8 weights in its group share the wide ruler. The price is the scales: with FP16, +16/8 = 2 bits per weight in this example, and much less with groups of 32 or 128.' },
    { q: 'Why doesn\'t GLM use W8A8 in the prefill of the KDA projection, even though it is faster than Marlin?', o: ['Because GB10 has no FP8 tensor core', 'Because the activation QDQ alone caused ~0.0266 of error per layer (versus ~0.0025 for the BF16 copy) and shifted the model\'s probabilities; the backend contributed only ~0.002–0.003', 'Because W8A8 takes more memory than BF16', 'Because Marlin doesn\'t accept a large M'], a: 1, e: 'The attribution in PR #182 isolated the term: quantizing the activation is what changes the output. The A/B/C study showed 7.99× the KL of stock noise and 23 token flips. The recipe keeps a BF16 copy of the same logical FP8 matrix and uses BF16 cuBLAS above M = 512, with no activation error, for +3.29 GiB per rank.' },
    { q: 'Qwen\'s FP8-dense checkpoint quantizes weights per channel and activations per token, without calibration. Which statement is correct?', o: ['The activation scales were measured on a calibration set and written into the checkpoint', 'The activation scales are computed at runtime (amax of each token); the weight scales (amax/448 per row) ship in the checkpoint and are sliced like the weight in TP', 'Per-channel scales don\'t work with tensor parallelism', 'The scheme requires shapes that are multiples of 128'], a: 1, e: 'It\'s FP8_PER_CHANNEL_PER_TOKEN: static per-channel weight, dynamic per-token activation. Nothing needs calibration. The per-channel scale is a vector with one entry per row, which is why it is cut along with the rows in TP. Blocks of 128 would run into the shared expert\'s 320 rows per rank and into the ParallelLMHead loader.' },
    { q: 'Why is serving DeepSeek-V4\'s KV in FP8 with a UE8M0 scale every 64 NoPE dims not a “post-training” degradation?', o: ['Because FP8 is exact', 'Because the model trained with QDQ in exactly this format (QAT): the reference forward applies act_quant to the NoPE KV in groups of 64', 'Because the RoPE dims are FP8 too', 'Because the drafter corrects the error'], a: 1, e: 'It\'s QAT: the rounding was already in the training forward, and the gradient went through it via the STE. Serving it this way reproduces the training regime. The VLLM_DSPARK_REFERENCE_KV_QUANT_DEQUANT knob applies the same QDQ to the drafter\'s KV to check parity. The RoPE dims stay in BF16.' },
  ],
  mount(root, LA) {
    const V = LA.viz, UI = LA.ui, N = LA.num;
    const $ = (s) => root.querySelector(s);
    const br = (x, d = 2) => (Number.isFinite(x) ? x.toFixed(d).replace(/^-/, '−') : '—');
    const pct = (x, d = 1) => br(x * 100, d) + '%';
    const fnt = (o) => V.font(o.size || 12, o.font || 'sans', o.weight || 400);
    function splitLines(g, s, maxW, o) {
      g.font = fnt(o);
      const out = []; let cur = '';
      String(s).split(' ').forEach((wd) => { const tt = cur ? cur + ' ' + wd : wd; if (cur && g.measureText(tt).width > maxW) { out.push(cur); cur = wd; } else cur = tt; });
      if (cur) out.push(cur);
      return out;
    }
    function wrap(g, s, x, y, maxW, lh, o = {}) {
      const ls = splitLines(g, s, maxW, o), mx = o.maxLines || 99;
      ls.slice(0, mx).forEach((l, i) => V.text(g, i === mx - 1 && ls.length > mx ? l + '…' : l, x, y + i * lh, o));
      return Math.min(ls.length, mx);
    }
    const tw = (g, s, o) => { g.font = fnt(o); return g.measureText(String(s)).width; };
    // no celular algumas figuras precisam de mais altura (o layout empilha); reavalia a cada resize/rotação
    const mobileH = (fig, px) => {
      const st = fig && fig.querySelector('.la-fig-stage'); if (!st) return;
      const base = st.style.height;
      const apply = () => { if (!fig.isConnected || fig.clientWidth <= 0) return; const want = fig.clientWidth < 560 ? px + 'px' : base; if (st.style.height !== want) st.style.height = want; };
      apply();
      if (typeof ResizeObserver === 'undefined') return;
      let raf = 0;
      const ro = new ResizeObserver(() => { if (!raf) raf = requestAnimationFrame(() => { raf = 0; apply(); }); });
      ro.observe(fig);
      if (LA.onCleanup) LA.onCleanup(() => { ro.disconnect(); if (raf) cancelAnimationFrame(raf); });
    };
    // texto sobre uma pílula com o fundo do painel (para linhas não riscarem rótulos)
    const pillText = (g, s, x, y, o) => {
      const C = V.C(), sz = o.size || 12, wd = tw(g, s, o);
      const xl = o.align === 'center' ? x - wd / 2 : o.align === 'right' ? x - wd : x;
      V.box(g, xl - 3, y - sz + 1, wd + 6, sz + 4, { r: 5, fill: C.panel });
      V.text(g, s, x, y, o);
    };
    const FC = (C) => ({ bf16: C.b, fp8: C.a, q4: C.c, fp32: C.d, mix: C.muted });
    const FN = { bf16: 'BF16', fp8: 'FP8', q4: '4 bits', fp32: 'FP32', mix: 'mixed' };
    const legendFormats = (ctl) => { const C = V.C(); UI.legend(ctl, [[C.b, 'BF16'], [C.a, 'FP8'], [C.c, '4 bits'], [C.d, 'FP32'], [C.muted, 'mixed']]); };

    /* ======================================================================
       Figura 1 — Cabe? E quanto se lê por passo? (balança + esteira)
       ====================================================================== */
    (function () {
      const fig = $('#fig-bytes'); if (!fig) return;
      mobileH(fig, 400);
      const CAP = 2 * 121.69 * 1.073741824; // GB visíveis em 2 nós
      // GLM: estimativa do inventário (arch-glm.md:798-801). Servido = GLM53_DENSE_FP8=all: grupos convertidos a 1 B/peso;
      // MLA+idx mantido em 1,3 GB como limite superior (indexador e kv_b_proj ficam BF16; só q/o da MLA viram FP8).
      const GLM_PARTS = {
        bf16: [
          { l: 'experts EXL3', v: 15, f: 'q4' }, { l: 'KDA', v: 4.7, f: 'bf16' }, { l: 'MLA+idx', v: 1.3, f: 'bf16' },
          { l: 'shared', v: 1.1, f: 'bf16' }, { l: 'lm_head', v: 0.6, f: 'bf16' }, { l: 'dense MLP', v: 0.45, f: 'bf16' },
        ],
        fp8: [
          { l: 'experts EXL3', v: 15, f: 'q4' }, { l: 'KDA', v: 2.35, f: 'fp8' }, { l: 'MLA+idx', v: 1.3, f: 'mix' },
          { l: 'shared', v: 0.55, f: 'fp8' }, { l: 'lm_head', v: 0.6, f: 'bf16' }, { l: 'dense MLP', v: 0.225, f: 'fp8' },
        ],
      };
      const GLM_SRC = {
        bf16: 'inventory estimate · DFlash2 step · pure checkpoint: BF16 dense ≈ 8 GB per step; converting KDA + dense MLP cuts ~2.6 GB ≈ 11 ms',
        fp8: 'inventory estimate · DFlash2 step · served default GLM53_DENSE_FP8=all: KDA, shared and dense MLP at 1 B/weight; MLA+idx kept at 1.3 GB (upper bound: indexer and kv_b_proj stay BF16)',
      };
      const M = {
        glm: {
          name: 'GLM-5.3-Flash', rc: 'glm',
          formats: [
            { l: 'BF16', gb: 642.6, n: '321.3 B × 2 B (calc.)' },
            { l: 'official FP8', gb: 328.3, n: 'official checkpoint' },
            { l: 'recipe · EXL3 4 bpw', gb: 175.6, n: '163.6 GiB', rec: true },
          ],
          unit: 'GB', toGB: 1, src: null, parts: null,
        },
        qwen: {
          name: 'Qwen3.8-Flash-Next', rc: 'qwen',
          formats: [
            { l: 'BF16', gb: 360.0, n: '180 B × 2 B (calc.)' },
            { l: 'FP8 everywhere', gb: 180.0, n: '180 B × 1 B (calc.)' },
            { l: 'recipe · NVFP4 + FP8', gb: 132.64, n: '123.53 GiB', rec: true },
          ],
          unit: 'GiB', toGB: 1.073741824, src: 'recipe report · batch 1, MTP=3 · analytical ≈ measured · the model counts the GDN state in FP32; the recipe stores it in BF16',
          parts: [
            { l: 'lm_head ×4', v: 2.36, f: 'bf16' }, { l: 'experts NVFP4', v: 2.1, f: 'q4' }, { l: 'GDN', v: 1.95, f: 'bf16' },
            { l: 'HC', v: 1.19, f: 'bf16' }, { l: 'attention+idx', v: 0.6, f: 'bf16' }, { l: 'MTP', v: 0.5, f: 'mix' },
            { l: 'router+shared+PLE', v: 0.4, f: 'bf16' }, { l: 'GDN state', v: 0.1, f: 'fp32' },
          ],
        },
        ds: {
          name: 'DeepSeek-V4-Flash', rc: 'ds',
          formats: [
            { l: 'BF16 (trunk only)', gb: 568.6, n: '284.3 B × 2 B (calc.)' },
            { l: 'FP8 everywhere (trunk only)', gb: 284.3, n: '284.3 B × 1 B (calc.)' },
            { l: 'recipe · MXFP4 + FP8', gb: 167.5, n: '≈ 156 GiB', rec: true },
          ],
          unit: 'GB', toGB: 1, src: 'recipe report · TP=2, c=1, k=6',
          parts: [
            { l: 'experts MXFP4', v: 12, f: 'q4' }, { l: 'attention FP8', v: 2.4, f: 'fp8' }, { l: 'lm_head', v: 1.06, f: 'bf16' },
            { l: 'DSpark', v: 0.9, f: 'mix' }, { l: 'Markov', v: 0.2, f: 'mix' },
          ],
        },
      };
      const S = { m: 'qwen', glmDense: 'fp8' };
      const ctl = UI.controls(fig);
      let segG = null;
      UI.seg(ctl, { label: 'Model', options: [{ v: 'glm', label: 'GLM' }, { v: 'qwen', label: 'Qwen' }, { v: 'ds', label: 'DeepSeek' }], value: S.m, onChange: (v) => { S.m = v; if (segG) segG.el.style.display = v === 'glm' ? '' : 'none'; } });
      segG = UI.seg(ctl, { label: 'GLM dense', options: [{ v: 'fp8', label: 'served (FP8=all)' }, { v: 'bf16', label: 'checkpoint (BF16)' }], value: S.glmDense, onChange: (v) => { S.glmDense = v; } });
      segG.el.style.display = 'none';
      legendFormats(ctl);
      const BW = 240; // GB/s efetivos (meio da faixa 230–250)
      V.fig(fig, (g, w, h, t) => {
        const C = V.C(); const F = FC(C); const m = M[S.m];
        const parts = S.m === 'glm' ? GLM_PARTS[S.glmDense] : m.parts;
        const src = S.m === 'glm' ? GLM_SRC[S.glmDense] : m.src;
        const pad = 16, x0 = pad, x1 = w - pad;
        const narrow = w < 560;
        // ---- painel de cima: cabe?
        V.text(g, 'DOES IT FIT? WEIGHTS × MEMORY VISIBLE ON 2 NODES', x0, 20, { size: 11, font: 'mono', color: C.muted, weight: 600 });
        const labW = narrow ? 0 : 170;
        const bx = x0 + labW, bw = x1 - bx - (narrow ? 0 : 70);
        const maxGB = 700;
        const sx = (gb) => bx + (gb / maxGB) * bw;
        const rowH = narrow ? 36 : 30;
        const top = 36;
        const cx = sx(CAP), c1 = sx(CAP / 2), yTop = top - 6, yBot = top + m.formats.length * rowH + (narrow ? 12 : 0);
        // linhas de capacidade primeiro: rótulos e valores vêm por cima, sobre pílulas
        V.line(g, cx, yTop, cx, yBot, { color: C.ink, lw: 1.5, dash: [4, 3] });
        V.line(g, c1, yTop + 6, c1, yBot, { color: C.muted, lw: 1, dash: [2, 4] });
        m.formats.forEach((f, i) => {
          const y = top + i * rowH + (narrow ? 14 : 0);
          const fits = f.gb < CAP * 0.85;
          const col = f.gb > CAP ? C.red : fits ? C.a : C.d;
          V.box(g, bx, y, bw, 18, { r: 6, fill: V.alpha(C.muted, 0.1) });
          const fw = Math.min(bw, sx(f.gb) - bx);
          V.box(g, bx, y, Math.max(4, fw), 18, { r: 6, fill: V.alpha(col, f.rec ? 0.8 : 0.5) });
          if (narrow) pillText(g, f.l, bx + 2, y - 4, { size: 11.5, color: f.rec ? C.ink : C.muted, weight: f.rec ? 600 : 400 });
          else V.text(g, f.l, x0, y + 13, { size: 12, color: f.rec ? C.ink : C.muted, weight: f.rec ? 600 : 400, maxW: labW - 8 });
          const txt = br(f.gb, 1) + ' GB';
          const tx = Math.min(bx + fw + 6, x1 - tw(g, txt, { size: 11.5, font: 'mono' }) - 2);
          pillText(g, txt, tx, y + 13, { size: 11.5, font: 'mono', color: C.ink });
        });
        pillText(g, '2 nodes ≈ ' + br(CAP, 0) + ' GB', Math.min(cx + 4, x1 - 96), yBot + 12, { size: 11, font: 'mono', color: C.ink });
        pillText(g, '1 node', c1 - 12, yTop + 2, { size: 11, font: 'mono', color: C.muted });
        // ---- painel de baixo: bytes por passo
        const y2 = yBot + 36;
        V.text(g, narrow ? 'IS IT FAST? BYTES READ PER RANK · 1 STEP' : 'IS IT FAST? BYTES READ PER RANK IN ONE DECODE STEP (RECIPE)', x0, y2, { size: 11, font: 'mono', color: C.muted, weight: 600, maxW: x1 - x0 });
        const tot = parts.reduce((a, p) => a + p.v, 0);
        const totGB = tot * m.toGB, tms = (totGB / BW) * 1000;
        const barY = y2 + 12, barH = 30;
        let xx = x0;
        const bw2 = x1 - x0;
        parts.forEach((p) => {
          const pw = (p.v / tot) * bw2;
          V.box(g, xx + 1, barY, Math.max(2, pw - 2), barH, { r: 5, fill: V.alpha(F[p.f], 0.55), stroke: F[p.f], lw: 1 });
          if (pw > tw(g, p.l, { size: 11 }) + 8) V.text(g, p.l, xx + pw / 2, barY + barH / 2 + 4, { size: 11, align: 'center', color: C.ink, weight: 500 });
          xx += pw;
        });
        const sumTxt = `≈ ${br(tot, 1)} ${m.unit} per step → ≈ ${br(tms, 0)} ms at ${BW} GB/s → ≈ ${br(1000 / tms, 0)} steps/s`;
        let yy = barY + barH + 18;
        const n1 = wrap(g, sumTxt, x0, yy, x1 - x0, 15, { size: 12.5, color: C.ink, weight: 600, maxLines: 2 });
        yy += (n1 - 1) * 15 + 16;
        const n2 = wrap(g, src, x0, yy, x1 - x0, 13, { size: 11, color: C.muted, maxLines: narrow ? 4 : 2 });
        yy += (n2 - 1) * 13;
        // esteira: um "trem" = todos os bytes de um passo; câmera lenta 25×
        const beltY = yy + 24, beltH = Math.min(64, Math.max(40, h - beltY - 10));
        if (h - beltY < 30) return;
        V.box(g, x0, beltY, bw2, beltH, { r: 12, fill: V.alpha(C.muted, 0.08), stroke: V.alpha(C.muted, 0.3) });
        const gw = 64;
        V.box(g, x0 + 4, beltY + 4, gw, beltH - 8, { r: 9, fill: V.alpha(C.muted, 0.14) });
        V.text(g, 'memory', x0 + 4 + gw / 2, beltY + beltH / 2 + 4, { size: 11, font: 'mono', align: 'center', color: C.muted });
        V.box(g, x1 - 4 - gw, beltY + 4, gw, beltH - 8, { r: 9, fill: V.alpha(C.a, 0.14), stroke: V.alpha(C.a, 0.4), lw: 1 });
        V.text(g, 'GPU', x1 - 4 - gw / 2, beltY + beltH / 2 + 4, { size: 11, font: 'mono', align: 'center', color: C.ink, weight: 600 });
        const lx0 = x0 + gw + 10, lx1 = x1 - gw - 10, laneW = lx1 - lx0;
        const trainW = laneW * 0.62;
        const period = (tms / 1000) * 25; // segundos por passo, em câmera lenta
        const u = (t % period) / period;
        const spacing = trainW + laneW * 0.22;
        const head = lx0 - trainW + u * spacing;
        g.save(); V.rr(g, lx0, beltY + 6, laneW, beltH - 12, 6); g.clip();
        const ty = beltY + 12, th = beltH - 24;
        for (let k = 0; k < 3; k++) {
          let px = head + k * spacing;
          if (px > lx1 || px + trainW < lx0) continue;
          parts.forEach((p) => { const pw = (p.v / tot) * trainW; V.box(g, px + 1, ty, Math.max(1.5, pw - 2), th, { r: 4, fill: V.alpha(F[p.f], 0.8) }); px += pw; });
        }
        g.restore();
        for (let i = 0; i < 16; i++) { const qx = lx0 + i * (laneW / 15); V.dot(g, qx, beltY + beltH - 5, 2, V.alpha(C.muted, 0.55)); }
        V.text(g, narrow ? '1 step = 1 “train” · 25× slow-mo' : 'one whole step = one “train” of bytes · 25× slow motion', lx0 + laneW / 2, beltY - 5, { size: 11, align: 'center', color: C.muted, maxW: laneW });
      });
    })();

    /* ======================================================================
       Figura 2 — INT4 à mão (stepper)
       ====================================================================== */
    (function () {
      const fig = $('#fig-regua'); if (!fig) return;
      const xs = [-0.8, -0.1, 0.3, 1.2];
      const A = N.quantInt(xs, { bits: 4, symmetric: false });
      const Sy = N.quantInt(xs, { bits: 4, symmetric: true });
      const sA = A.scales[0], zA = A.zeros[0], sS = Sy.scales[0];
      const steps = [
        { t: 'Four numbers', d: 'The vector \\(x = [-0.8,\\ -0.1,\\ 0.3,\\ 1.2]\\), stored in 16 bits. We want 4 bits per number, that is, only 16 possible ticks.' },
        { t: 'Stretch the ruler', d: 'In the asymmetric version the ruler goes from the minimum to the maximum: \\(s = \\dfrac{1.2 - (-0.8)}{2^4-1} = \\dfrac{2.0}{15} = 0.1333\\). That makes 16 ticks spaced \\(s\\) apart.' },
        { t: 'Where zero lands', d: '\\(z = \\mathrm{round}(0.8/0.1333) = 6\\). Tick 6 sits exactly on the real zero, and tick 0 on −0.8.' },
        { t: 'Divide, round, shift', d: '\\(x/s = [-6,\\ -0.75,\\ 2.25,\\ 9]\\) → round → \\([-6,\\ -1,\\ 2,\\ 9]\\) → \\(+z\\) → \\(q = [0,\\ 5,\\ 8,\\ 15]\\). In binary: 0000, 0101, 1000, 1111. Each number now takes up half a byte.' },
        { t: 'Go back and measure the error', d: '\\(\\hat x = s\\,(q - z) = [-0.8,\\ -0.133,\\ 0.267,\\ 1.2]\\). The maximum error was 0.033, within the limit \\(s/2 = 0.067\\). The extremes come out exact because the ruler was stretched all the way to them.' },
        { t: 'What if it were symmetric?', d: 'Now \\(z = 0\\) and \\(s = \\mathrm{amax}/7 = 1.2/7 = 0.171\\). \\(q = [-5,\\ -1,\\ 2,\\ 7]\\) and \\(\\hat x = [-0.857,\\ -0.171,\\ 0.343,\\ 1.2]\\). Codes −8, −7 and −6 are never used, and the maximum error rises to 0.071. The kernel gets simpler (it doesn\'t subtract \\(z\\)).' },
      ];
      let H;
      const st = UI.stepper(fig, steps, { autoplay: 6500, onStep: () => H && H.redraw() });
      const bin4 = (q) => ((q + 16) % 16).toString(2).padStart(4, '0');
      const bf16Bits = xs.map((x) => N.encode('bf16', x).bits);
      H = V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const i = st.i, el = st.elapsed();
        const sym = i === 5;
        const pad = 22, lo = -1.45, hi = 1.4;
        const X = (v) => pad + ((v - lo) / (hi - lo)) * (w - 2 * pad);
        const yl = 92;
        // eixo
        V.line(g, pad, yl, w - pad, yl, { color: V.alpha(C.muted, 0.6), lw: 1.5 });
        [-1, -0.5, 0, 0.5, 1].forEach((v) => { V.line(g, X(v), yl, X(v), yl + 6, { color: C.muted, lw: 1 }); V.text(g, br(v, 1), X(v), yl + 19, { size: 11, font: 'mono', align: 'center', color: C.muted }); });
        const s = sym ? sS : sA, z = sym ? 0 : zA;
        const codes = sym ? Array.from({ length: 16 }, (_, k) => k - 8) : Array.from({ length: 16 }, (_, k) => k);
        const narrow = w < 520;
        if (i >= 1) {
          const ap = Math.min(1, el / 0.8);
          codes.forEach((q, k) => {
            const v = (q - z) * s;
            const px = X(v);
            const unused = sym && q < -5;
            const isZ = !sym && q === z && i >= 2;
            const hgt = isZ ? 26 : 16;
            V.line(g, px, yl - hgt * ap, px, yl, { color: unused ? V.alpha(C.red, 0.7) : isZ ? C.c : V.alpha(C.a, 0.9), lw: isZ ? 2.5 : 1.6 });
            if (!narrow || k % 2 === 0 || isZ) V.text(g, String(q).replace('-', '−'), px, yl - hgt - 6, { size: 11, font: 'mono', align: 'center', color: unused ? C.red : isZ ? C.c : C.ink, alpha: ap });
          });
          V.text(g, (sym ? 'symmetric: ' : 'asymmetric: ') + 's = ' + br(s, 4) + (sym ? ', z = 0' : i >= 2 ? ', z = ' + z : ''), pad, 22, { size: 12.5, font: 'mono', color: C.ink, weight: 600 });
          if (sym) V.text(g, 'codes −8…−6 unused', X(-1.37), yl + 36, { size: 11, color: C.red });
          if (isFinite(z) && !sym && i >= 2) V.text(g, 'z = 6 ↔ 0.0', X(0) + 6, yl - 44, { size: 11.5, color: C.c, weight: 600 });
        } else {
          V.text(g, 'Four real numbers on a line, 16 bits each', pad, 22, { size: 12.5, color: C.ink, weight: 600 });
          // régua "fantasma": 16 marcas procurando lugar sobre a reta
          const off = 0.22 * Math.sin(t * 0.9), span = 1.6 + 0.35 * Math.sin(t * 0.55 + 1);
          for (let k = 0; k < 16; k++) {
            const v = off - span / 2 + (k * span) / 15, px = X(v);
            V.line(g, px, yl - 16, px, yl, { color: V.alpha(C.a, 0.3), lw: 1.4, dash: [3, 2] });
          }
          V.text(g, narrow ? '16 ticks: where does the ruler go?' : 'only 16 ticks of 4 bits: where does the ruler go?', X(off), yl - 24, { size: 11, align: 'center', color: C.muted });
        }
        // bolinhas
        const cols = [C.a, C.b, C.c, C.d];
        const deq = sym ? Sy.deq : A.deq;
        xs.forEach((x, k) => {
          let v = x;
          if (i >= 3) { const u = V.ease.inOut(Math.min(1, el / 1.1)); v = i === 3 ? V.lerp(x, deq[k], u) : deq[k]; }
          const px = X(v);
          if (i >= 4) {
            V.line(g, X(x), yl - 2, X(deq[k]), yl - 2, { color: C.red, lw: 3 });
            V.dot(g, X(x), yl, 3, V.alpha(C.ink, 0.5));
          }
          V.glow(g, px, yl, 12, cols[k], 0.5);
          V.dot(g, px, yl, 5.5, cols[k]);
        });
        // tabela
        const ty = 150, rows = [];
        rows.push(['x', xs.map((x) => br(x, 2))]);
        if (i <= 2) rows.push([narrow ? 'BF16 (hex)' : 'BF16 (16 bits)', bf16Bits.map((b) => (narrow ? '0x' + b.toString(16).toUpperCase().padStart(4, '0') : N.toBin(b, 16)))]);
        if (i >= 3) {
          rows.push(['x / s', xs.map((x) => br(x / s, 2))]);
          rows.push(['round', xs.map((x) => String(Math.round(x / s)).replace('-', '−'))]);
          const q = sym ? Sy.q : A.q;
          rows.push([sym ? 'q (no z)' : 'q = round + z', q.map((v) => String(v).replace('-', '−'))]);
          rows.push(['bits', q.map((v) => bin4(v))]);
        }
        if (i >= 4) {
          rows.push(['x̂ = s(q − z)', deq.map((v) => br(v, 3))]);
          rows.push(['|error|', xs.map((x, k) => br(Math.abs(deq[k] - x), 3))]);
        }
        const lw = narrow ? 92 : 130, cw = (w - 2 * pad - lw) / 4, rh = 25;
        V.text(g, 'number', pad, ty, { size: 11, font: 'mono', color: C.muted });
        xs.forEach((x, k) => { V.dot(g, pad + lw + cw * k + cw / 2 - 22, ty - 4, 4, cols[k]); V.text(g, '#' + (k + 1), pad + lw + cw * k + cw / 2 - 14, ty, { size: 11, font: 'mono', color: C.muted }); });
        rows.forEach((r, ri) => {
          const y = ty + 10 + ri * rh;
          const hl = (i === 0 && ri === 1) || (i === 3 && ri >= 1 && ri <= 4) || (i === 4 && ri >= 5) || (i === 5 && ri >= 1);
          V.box(g, pad, y, w - 2 * pad, rh - 4, { r: 6, fill: hl ? V.alpha(C.a, 0.12) : V.alpha(C.muted, 0.06) });
          V.text(g, r[0], pad + 8, y + rh / 2 + 2, { size: 11.5, font: 'mono', color: C.muted, maxW: lw - 12 });
          const long = r[1].some((s2) => s2.length > 10);
          r[1].forEach((s2, k) => V.text(g, s2, pad + lw + cw * k + cw / 2, y + rh / 2 + 2, { size: long ? 11 : 12, font: 'mono', align: 'center', color: C.ink, weight: 500, maxW: cw - 4 }));
        });
        // memória ocupada: 4 × 16 bits antes, 4 × 4 bits depois de quantizar
        const tEnd = ty + 10 + rows.length * rh, mY = h - 30;
        if (mY - tEnd >= 34) {
          const q4 = i >= 3;
          const u = i === 3 ? V.ease.inOut(Math.min(1, el / 1.1)) : 1;
          const nbits = q4 ? V.lerp(16, 4, u) : 16;
          const slot = (w - 2 * pad) / 4, cb = slot / 16;
          V.text(g, q4 ? (narrow ? 'memory: 4 × 4 bits = 2 bytes' : 'memory: 4 × 4 bits = 16 bits = 2 bytes (s and z are stored separately, once per group)') : (narrow ? 'memory: 4 × 16 bits = 8 bytes' : 'memory: 4 × 16 bits = 64 bits = 8 bytes'), pad, mY - 8, { size: 11, font: 'mono', color: C.muted, weight: 600, maxW: w - 2 * pad });
          for (let k = 0; k < 4; k++) {
            const x0 = pad + k * slot;
            V.box(g, x0 + 1, mY, slot - 4, 14, { r: 3, fill: V.alpha(C.muted, 0.08) });
            const full = Math.floor(nbits), frac = nbits - full;
            for (let b = 0; b < Math.ceil(nbits); b++) { const wb = (b < full ? 1 : frac) * (cb - 1); if (wb > 0.3) V.box(g, x0 + 1 + b * cb, mY + 1, wb, 12, { r: 2, fill: V.alpha(cols[k], 0.8) }); }
          }
        }
      }, { static: false });
    })();

    /* ======================================================================
       Figura 3 — O U do erro: arredondamento contra corte
       ====================================================================== */
    (function () {
      const fig = $('#fig-corte'); if (!fig) return;
      mobileH(fig, 500);
      const W = N.synthWeights(6000, 11, 0.01, 8);
      const amax = Math.max(...W.map(Math.abs));
      const Ex2 = W.reduce((a, x) => a + x * x, 0) / W.length;
      const S = { bits: 4, clip: 1.0 };
      const quant = (x, c, qmax) => { const s = c / qmax; const q = Math.max(-qmax, Math.min(qmax, Math.round(x / s))); return q * s; };
      const R = Array.from({ length: 49 }, (_, i) => 0.04 + (0.96 * i) / 48);
      let curve = [], best = 1;
      function recompute() {
        const qmax = (1 << (S.bits - 1)) - 1;
        curve = R.map((r) => { const c = r * amax; let se = 0; for (let i = 0; i < W.length; i++) { const d = quant(W[i], c, qmax) - W[i]; se += d * d; } return se / W.length; });
        let bi = 0; curve.forEach((v, i) => { if (v < curve[bi]) bi = i; }); best = R[bi];
      }
      recompute();
      const samp = []; const rr = V.rng(4);
      const outIdx = []; W.forEach((x, i) => { if (Math.abs(x) > 0.5 * amax) outIdx.push(i); });
      for (let k = 0; k < 30; k++) samp.push(rr.int(W.length));
      outIdx.slice(0, 6).forEach((i) => samp.push(i));
      const ctl = UI.controls(fig);
      UI.slider(ctl, { label: 'Bits', min: 2, max: 8, step: 1, value: S.bits, fmt: (v) => v + ' bits', onInput: (v) => { S.bits = v; recompute(); upd(); } });
      const sc = UI.slider(ctl, { label: 'Clip c / amax', min: 0.05, max: 1, step: 0.01, value: S.clip, fmt: (v) => br(v, 2), onInput: (v) => { S.clip = v; upd(); } });
      UI.button(ctl, 'go to optimum', () => sc.set(Math.max(0.05, Math.round(best * 100) / 100)));
      const rS = UI.readout(ctl, 'tick s', '—'), rQ = UI.readout(ctl, 'SQNR', '—'), rR = UI.readout(ctl, 'rel RMSE', '—'), rC = UI.readout(ctl, 'clipped', '—');
      function upd() {
        const qmax = (1 << (S.bits - 1)) - 1, c = S.clip * amax;
        let se = 0, nc = 0; for (let i = 0; i < W.length; i++) { const d = quant(W[i], c, qmax) - W[i]; se += d * d; if (Math.abs(W[i]) > c) nc++; }
        const mse = se / W.length;
        rS.set(br(c / qmax * 1000, 2) + '·10⁻³'); rQ.set(br(10 * Math.log10(Ex2 / mse), 1) + ' dB'); rR.set(pct(Math.sqrt(mse / Ex2), 1)); rC.set(pct(nc / W.length, 2));
      }
      upd();
      const nbDesk = 72, nbMob = 48;
      const histCache = {};
      function hist(nb) { if (histCache[nb]) return histCache[nb]; const hh = new Array(nb).fill(0); W.forEach((x) => { const b = Math.min(nb - 1, Math.floor(((x + amax) / (2 * amax)) * nb)); hh[b]++; }); return (histCache[nb] = hh); }
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const narrow = w < 640;
        const qmax = (1 << (S.bits - 1)) - 1, c = S.clip * amax, s = c / qmax;
        const hx = 16, hy = 30, hw = narrow ? w - 32 : w * 0.6 - 24, hh = narrow ? h * 0.56 - 44 : h - 76;
        const nb = narrow ? nbMob : nbDesk, H = hist(nb), mxH = Math.sqrt(Math.max(...H));
        const X = (v) => hx + ((v + amax) / (2 * amax)) * hw;
        V.text(g, 'WEIGHT HISTOGRAM AND THE RULER (' + S.bits + ' bits, ' + (2 * qmax + 1) + ' levels)', hx, 18, { size: 11, font: 'mono', color: C.muted, weight: 600, maxW: hw });
        const base = hy + hh;
        const bwid = hw / nb;
        H.forEach((cnt, b) => {
          const v0 = -amax + (b + 0.5) * (2 * amax / nb);
          const bh = (Math.sqrt(cnt) / mxH) * (hh - 60);
          const clipped = Math.abs(v0) > c;
          g.fillStyle = clipped ? V.alpha(C.red, 0.65) : V.alpha(C.a, 0.55);
          V.rr(g, hx + b * bwid + 0.5, base - bh, Math.max(1, bwid - 1), bh, 2); g.fill();
        });
        V.line(g, hx, base, hx + hw, base, { color: C.muted, lw: 1 });
        // régua
        const pxs = (s / (2 * amax)) * hw;
        if (pxs >= 3) {
          for (let k = -qmax; k <= qmax; k++) { const px = X(k * s); V.line(g, px, base + 2, px, base + 10, { color: C.ink, lw: 1.2 }); }
        } else {
          g.fillStyle = V.alpha(C.ink, 0.25); g.fillRect(X(-c), base + 3, X(c) - X(-c), 6);
          V.text(g, 'ticks too fine to draw', X(0), base + 24, { size: 11, align: 'center', color: C.muted });
        }
        [-c, c].forEach((v, k) => { V.line(g, X(v), hy + 10, X(v), base + 12, { color: C.red, lw: 1.5, dash: [4, 3] }); V.text(g, k ? '+c' : '−c', X(v), hy + 4, { size: 11, font: 'mono', align: 'center', color: C.red }); });
        // bolinhas que caem na régua
        samp.forEach((idx, k) => {
          const x = W[idx], xq = quant(x, c, qmax);
          const u = V.ease.inOut(V.pingpong(t * 0.35 + (k % 7) * 0.07, 1));
          const v = V.lerp(x, xq, u);
          const y = hy + 18 + (k % 6) * 9;
          const cl = Math.abs(x) > c;
          V.dot(g, X(v), y, cl ? 3.4 : 2.6, cl ? C.red : C.b);
        });
        V.text(g, 'amax', X(amax) - 14, base + 24, { size: 11, font: 'mono', color: C.muted });
        // curva do MSE
        const cx0 = narrow ? 20 : w * 0.6 + 28, cy0 = narrow ? hy + hh + 58 : 44, cw = narrow ? w - 36 : w * 0.4 - 44, ch = narrow ? h - (hy + hh + 58) - 42 : h - 100;
        V.text(g, 'MSE × CLIPPING LIMIT (log scale)', cx0, cy0 - 10, { size: 11, font: 'mono', color: C.muted, weight: 600, maxW: cw });
        const lmin = Math.log10(Math.min(...curve)), lmax = Math.log10(Math.max(...curve));
        const CX = (r) => cx0 + ((r - 0.04) / 0.96) * cw;
        const CY = (m) => cy0 + ch - ((Math.log10(m) - lmin) / (lmax - lmin || 1)) * ch;
        V.box(g, cx0, cy0, cw, ch, { r: 6, fill: V.alpha(C.muted, 0.05), stroke: V.alpha(C.muted, 0.2), lw: 1 });
        g.beginPath(); curve.forEach((m, i) => { const px = CX(R[i]), py = CY(m); if (i) g.lineTo(px, py); else g.moveTo(px, py); }); g.strokeStyle = C.b; g.lineWidth = 2.2; g.stroke();
        const bi = R.indexOf(best);
        V.dot(g, CX(best), CY(curve[bi]), 5, C.e);
        V.text(g, 'best ≈ ' + br(best, 2), V.clamp(CX(best) - 30, cx0 + 4, cx0 + cw - 80), Math.max(cy0 + 14, CY(curve[bi]) - 12), { size: 11, color: C.e, weight: 600 });
        // ponto atual
        const qm = (1 << (S.bits - 1)) - 1; let se = 0; for (let i = 0; i < W.length; i += 3) { const d = quant(W[i], S.clip * amax, qm) - W[i]; se += d * d; }
        const cur = se / Math.ceil(W.length / 3);
        V.glow(g, CX(S.clip), CY(cur), 12, C.c, 0.6); V.dot(g, CX(S.clip), CY(cur), 4.5, C.c);
        V.text(g, br(0.04, 2), cx0, cy0 + ch + 14, { size: 11, font: 'mono', color: C.muted });
        V.text(g, 'c / amax', cx0 + cw / 2, cy0 + ch + 14, { size: 11, font: 'mono', align: 'center', color: C.muted });
        V.text(g, '1.00', cx0 + cw, cy0 + ch + 14, { size: 11, font: 'mono', align: 'right', color: C.muted });
        V.text(g, '← clipping dominates', cx0, cy0 + ch + 29, { size: 11, color: C.muted });
        V.text(g, 'rounding dominates →', cx0 + cw, cy0 + ch + 29, { size: 11, align: 'right', color: C.muted });
      });
    })();

    /* ======================================================================
       Figura 4 — Granularidade e o outlier (matriz 8 × 32)
       ====================================================================== */
    (function () {
      const fig = $('#fig-grao'); if (!fig) return;
      const R = 8, K = 32;
      const r = V.rng(5);
      const rs = Array.from({ length: R }, () => Math.exp(r.normal() * 0.45));
      const W0 = Array.from({ length: R }, (_, i) => Array.from({ length: K }, () => r.normal() * 0.02 * rs[i]));
      const OUT = [2, 9];
      const S = { mode: 'tensor', bits: 4, outlier: true, hover: null };
      let Q = null;
      function groupOf(i, j) {
        if (S.mode === 'tensor') return 0;
        if (S.mode === 'canal') return i;
        if (S.mode === 'grupo') return i * 4 + Math.floor(j / 8);
        return Math.floor(i / 4) * 4 + Math.floor(j / 8);
      }
      const gsize = { tensor: 256, canal: 32, grupo: 8, bloco: 32 };
      function quantize(mode, bits, Wm) {
        const qmax = (1 << (bits - 1)) - 1;
        const save = S.mode; S.mode = mode;
        const groups = {};
        for (let i = 0; i < R; i++) for (let j = 0; j < K; j++) { const gi = groupOf(i, j); (groups[gi] = groups[gi] || []).push([i, j]); }
        const D = Wm.map((row) => row.slice()), sc = {}, am = {};
        Object.keys(groups).forEach((gi) => {
          const cells = groups[gi];
          const a = Math.max(...cells.map(([i, j]) => Math.abs(Wm[i][j])), 1e-12);
          const s = a / qmax; sc[gi] = s; am[gi] = a;
          cells.forEach(([i, j]) => { D[i][j] = Math.max(-qmax, Math.min(qmax, Math.round(Wm[i][j] / s))) * s; });
        });
        S.mode = save;
        let se = 0, sx = 0; for (let i = 0; i < R; i++) for (let j = 0; j < K; j++) { se += (D[i][j] - Wm[i][j]) ** 2; sx += Wm[i][j] ** 2; }
        return { D, sc, am, rel: Math.sqrt(se / sx), sqnr: 10 * Math.log10(sx / se), n: Object.keys(groups).length };
      }
      function curW() { const Wm = W0.map((row) => row.slice()); if (S.outlier) Wm[OUT[0]][OUT[1]] = 0.25; return Wm; }
      let Wc = curW(), errMax = 1e-9, sT = 1;
      function recompute() {
        Wc = curW();
        Q = quantize(S.mode, S.bits, Wc);
        sT = quantize('tensor', S.bits, Wc).sc[0];
        errMax = 1e-9;
        ['tensor', 'canal', 'grupo', 'bloco'].forEach((m) => { const q = quantize(m, S.bits, Wc); for (let i = 0; i < R; i++) for (let j = 0; j < K; j++) errMax = Math.max(errMax, Math.abs(q.D[i][j] - Wc[i][j])); });
        rRel.set(pct(Q.rel, 1)); rSq.set(br(Q.sqnr, 1) + ' dB'); rN.set(String(Q.n)); rB.set(br(S.bits + 16 / gsize[S.mode], 3));
        H.redraw();
      }
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Scale', options: [{ v: 'tensor', label: 'per tensor' }, { v: 'canal', label: 'per channel' }, { v: 'grupo', label: 'group of 8' }, { v: 'bloco', label: 'block 4×8' }], value: S.mode, onChange: (v) => { S.mode = v; recompute(); } });
      UI.seg(ctl, { label: 'Bits', options: [3, 4, 8], value: S.bits, onChange: (v) => { S.bits = v; recompute(); } });
      UI.toggle(ctl, { label: 'outlier at (2, 9)', value: true, onChange: (v) => { S.outlier = v; recompute(); } });
      const rRel = UI.readout(ctl, 'rel RMSE', '—'), rSq = UI.readout(ctl, 'SQNR', '—'), rN = UI.readout(ctl, 'scales', '—'), rB = UI.readout(ctl, 'effective bits', '—');
      let geomTop = null;
      const H = V.fig(fig, (g, w, h, t, dt, St, HH) => {
        const C = V.C();
        const pad = 14;
        const cw = Math.max(6, (w - 2 * pad) / K);
        const ch = Math.max(12, Math.min(24, (h - 150) / (2 * R)));
        const gap = 1;
        // cor do W: escala saturada em 3σ típicos
        let s2 = 0; W0.forEach((row) => row.forEach((v) => { s2 += v * v; })); const sig = Math.sqrt(s2 / (R * K));
        const lim = 3 * sig;
        const y1 = 30, y2 = y1 + R * ch + 44;
        V.text(g, 'W (8 × 32) — original weights', pad, y1 - 10, { size: 11.5, font: 'mono', color: C.muted, weight: 600 });
        const hov = S.hover;
        const hovG = hov ? groupOf(hov[0], hov[1]) : null;
        const drawMat = (y0, val, cm, mx) => {
          for (let i = 0; i < R; i++) for (let j = 0; j < K; j++) {
            const x = pad + j * cw, y = y0 + i * ch;
            const v = val(i, j);
            g.fillStyle = cm === 'div' ? V.cmap(V.clamp(v / mx, -1, 1), 'div') : V.cmap(V.clamp(v / mx, 0, 1), 'heat');
            g.fillRect(x + gap / 2, y + gap / 2, cw - gap, ch - gap);
          }
        };
        drawMat(y1, (i, j) => Wc[i][j], 'div', lim);
        V.text(g, w < 560 ? '|Ŵ − W| — error of each weight' : '|Ŵ − W| — error of each weight (same color scale for every granularity)', pad, y2 - 10, { size: 11.5, font: 'mono', color: C.muted, weight: 600, maxW: w - 2 * pad });
        drawMat(y2, (i, j) => Math.abs(Q.D[i][j] - Wc[i][j]), 'heat', errMax);
        // contornos dos grupos
        const outline = (y0) => {
          const seen = {};
          for (let i = 0; i < R; i++) for (let j = 0; j < K; j++) {
            const gi = groupOf(i, j); if (seen[gi]) continue; seen[gi] = true;
            let i0 = R, i1 = -1, j0 = K, j1 = -1;
            for (let a = 0; a < R; a++) for (let b = 0; b < K; b++) if (groupOf(a, b) === gi) { i0 = Math.min(i0, a); i1 = Math.max(i1, a); j0 = Math.min(j0, b); j1 = Math.max(j1, b); }
            const isH = hovG === gi;
            g.strokeStyle = isH ? C.c : V.alpha(C.ink, 0.55); g.lineWidth = isH ? 2.6 : 1.2;
            V.rr(g, pad + j0 * cw, y0 + i0 * ch, (j1 - j0 + 1) * cw, (i1 - i0 + 1) * ch, 3); g.stroke();
          }
        };
        outline(y1); outline(y2);
        if (S.outlier) {
          const ox = pad + OUT[1] * cw + cw / 2, oy = y1 + OUT[0] * ch + ch / 2;
          V.glow(g, ox, oy, 14 + 3 * Math.sin(t * 3), C.d, 0.8);
          V.text(g, '★', ox, oy + 4, { size: 12, align: 'center', color: C.dark ? '#00210F' : '#fff' });
        }
        // hover
        const p = HH.pointer;
        S.hover = null;
        if (p.inside) {
          for (const y0 of [y1, y2]) { const j = Math.floor((p.x - pad) / cw), i = Math.floor((p.y - y0) / ch); if (i >= 0 && i < R && j >= 0 && j < K) S.hover = [i, j]; }
        }
        const info = S.hover || (S.outlier ? OUT : [0, 0]);
        const gi = groupOf(info[0], info[1]);
        const yInfo = y2 + R * ch + 22;
        const txt = `cell (${info[0]}, ${info[1]}): group with amax = ${br(Q.am[gi], 4)} → scale s = ${br(Q.sc[gi] * 1000, 3)}·10⁻³ · weight ${br(Wc[info[0]][info[1]], 4)} → ${br(Q.D[info[0]][info[1]], 4)}`;
        const nl = wrap(g, txt, pad, yInfo, w - 2 * pad, 15, { size: 12, font: 'mono', color: C.ink, maxLines: 2 });
        // barra de cor do mapa de erro: 0 → maior erro entre as 4 granularidades
        const cbY = yInfo + nl * 15 - 2, cbL = tw(g, 'error 0', { size: 11, font: 'mono' }) + 6;
        const cbW = Math.min(200, w - 2 * pad - cbL - (w < 560 ? 90 : 290));
        if (cbW > 40 && cbY + 12 < h) {
          V.text(g, 'error 0', pad, cbY + 9, { size: 11, font: 'mono', color: C.muted });
          for (let k = 0; k < 40; k++) { g.fillStyle = V.cmap(k / 39, 'heat'); g.fillRect(pad + cbL + (k * cbW) / 40, cbY, cbW / 40 + 0.6, 10); }
          g.strokeStyle = V.alpha(C.muted, 0.5); g.lineWidth = 1; g.strokeRect(pad + cbL, cbY, cbW, 10);
          V.text(g, br(errMax * 1000, 2) + '·10⁻³' + (w < 560 ? '' : ' (max = ' + br(errMax / sT, 2) + ' × per-tensor s)'), pad + cbL + cbW + 6, cbY + 9, { size: 11, font: 'mono', color: C.muted });
        }
      });
      recompute();
    })();

    /* ======================================================================
       Figure 4b — The q4 search (GLM·TF): 16 clips per group
       (reproduces quantize4_mse from patch 0005: a_i = 1 − 0.45·i/15, b = a·min,
        s = a·(max − min)/15, both rounded to BF16 before measuring the error;
        the group size is the teaching control: q4 uses 64)
       ====================================================================== */
    (function () {
      const fig = $('#fig-busca'); if (!fig) return;
      mobileH(fig, 680);
      const NG = 12, GMAX = 4096, NA = 16, NDOT = 256;
      const A = Array.from({ length: NA }, (_, i) => 1 - (0.45 * i) / (NA - 1));
      const bf = (v) => N.encode('bf16', v).v;
      const S = { gs: 64, seed: 3, hov: null };
      let BASE = [], OUTS = [], LOHI = [], E = [], BEST = [], counts = [], defSel = 0;
      function sample() {
        const r = V.rng(S.seed);
        BASE = []; OUTS = [];
        for (let gi = 0; gi < NG; gi++) {
          const sc = 0.02 * Math.exp(r.normal() * 0.3), xs = new Float64Array(GMAX), o = [];
          for (let k = 0; k < GMAX; k++) {
            xs[k] = r.normal() * sc;
            if (r() < 0.01) { xs[k] = (r() < 0.5 ? -1 : 1) * sc * (4 + 4 * r()); o.push(k); }
          }
          BASE.push(xs); OUTS.push(o);
        }
      }
      const grp = (gi) => BASE[gi].subarray(0, S.gs);
      const rule = (gi, a) => { const [lo, hi] = LOHI[gi]; return { s: bf(Math.max(((hi - lo) * a) / 15, 1e-8)), b: bf(lo * a) }; };
      const code = (x, s, b) => Math.max(0, Math.min(15, Math.round((x - b) / s)));
      function recompute() {
        LOHI = BASE.map((_, gi) => { const g = grp(gi); let lo = Infinity, hi = -Infinity; for (let k = 0; k < g.length; k++) { if (g[k] < lo) lo = g[k]; if (g[k] > hi) hi = g[k]; } return [lo, hi]; });
        E = BASE.map((_, gi) => { const g = grp(gi); return A.map((a) => { const { s, b } = rule(gi, a); let se = 0; for (let k = 0; k < g.length; k++) { const d = code(g[k], s, b) * s + b - g[k]; se += d * d; } return se; }); });
        BEST = E.map((row) => row.reduce((bi, v, i) => (v < row[bi] ? i : bi), 0));
        counts = new Array(NA).fill(0); BEST.forEach((b) => counts[b]++);
        let e1 = 0, eb = 0, sx = 0, am = 0;
        BASE.forEach((_, gi) => { const g = grp(gi); e1 += E[gi][0]; eb += E[gi][BEST[gi]]; am += A[BEST[gi]]; for (let k = 0; k < g.length; k++) sx += g[k] * g[k]; });
        defSel = 0; let gBest = -1;
        E.forEach((row, gi) => { const gain = row[0] / Math.max(row[BEST[gi]], 1e-30); if (gain > gBest) { gBest = gain; defSel = gi; } });
        r1.set(pct(Math.sqrt(e1 / sx), 1)); rb.set(pct(Math.sqrt(eb / sx), 1), 'ok');
        rm.set('−' + pct(1 - eb / e1, 0)); ra.set(br(am / NG, 2)); rbits.set(br(4 + 32 / S.gs, 3));
      }
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Group', options: [{ v: 64, label: '64 (q4)' }, { v: 256, label: '256' }, { v: 1024, label: '1024' }, { v: 4096, label: '4096 (per channel)' }], value: S.gs, onChange: (v) => { S.gs = +v; recompute(); } });
      UI.button(ctl, 'new sample', () => { S.seed += 1; sample(); recompute(); });
      const r1 = UI.readout(ctl, 'rel RMSE without search', '—'), rb = UI.readout(ctl, 'rel RMSE with search', '—');
      const rm = UI.readout(ctl, 'MSE with search', '—'), ra = UI.readout(ctl, 'mean clip', '—'), rbits = UI.readout(ctl, 'effective bits', '—');
      { const C0 = V.C(); UI.legend(ctl, [[C0.e, 'best clip so far'], [C0.c, 'clip under test'], [C0.red, 'clipped weight']]); }
      sample(); recompute();
      V.fig(fig, (g, w, h, t, dt, St, HH) => {
        const C = V.C();
        const narrow = w < 640, pad = 14;
        // sweep: the cursor walks the 16 clips and then rests on the result
        const per = 7.2, sweep = 4.8, ph = t % per, scanning = ph < sweep;
        const cur = scanning ? Math.min(NA - 1, Math.floor((ph / sweep) * NA)) : NA - 1;
        const bestSoFar = (gi) => { let bi = 0; for (let i = 1; i <= cur; i++) if (E[gi][i] < E[gi][bi]) bi = i; return bi; };
        // ---------- table: groups × clips
        const lw = narrow ? w - 2 * pad : Math.floor(w * 0.54) - pad;
        const labW = 34, chipW = 42, mx0 = pad + labW, mw = lw - labW - chipW, cw = mw / NA;
        const my0 = 32, barH = narrow ? 34 : 42;
        const ch = Math.max(12, Math.min(22, (narrow ? 230 : h - my0 - barH - 70) / NG));
        const tabBot = my0 + NG * ch;
        V.text(g, narrow ? 'MSE OF EACH CLIP, GROUP BY GROUP' : 'MSE OF EACH CLIP (row = one group of ' + S.gs + ' weights)', pad, 18, { size: 11, font: 'mono', color: C.muted, weight: 600, maxW: lw });
        const p = HH.pointer;
        S.hov = null;
        if (p.inside && p.x >= pad && p.x <= pad + lw && p.y >= my0 && p.y < tabBot) S.hov = Math.min(NG - 1, Math.max(0, Math.floor((p.y - my0) / ch)));
        const sel = S.hov == null ? defSel : S.hov;
        for (let gi = 0; gi < NG; gi++) {
          const y = my0 + gi * ch, lmn = Math.log(Math.max(E[gi][BEST[gi]], 1e-30)), lmx = Math.log(Math.max(...E[gi], 1e-30));
          for (let i = 0; i < NA; i++) {
            g.fillStyle = V.cmap(V.clamp((Math.log(Math.max(E[gi][i], 1e-30)) - lmn) / (lmx - lmn || 1), 0, 1), 'heat');
            V.rr(g, mx0 + i * cw + 1, y + 1, cw - 2, ch - 2, 2); g.fill();
          }
          const bs = bestSoFar(gi), rd = Math.min(5, ch * 0.24), cx = mx0 + (bs + 0.5) * cw, cy = y + ch / 2;
          V.dot(g, cx, cy, rd, C.e);
          g.strokeStyle = C.panel; g.lineWidth = 1.2; g.beginPath(); g.arc(cx, cy, rd, 0, Math.PI * 2); g.stroke();
          V.text(g, 'g' + gi, pad, cy + 4, { size: 11, font: 'mono', color: gi === sel ? C.ink : C.muted, weight: gi === sel ? 700 : 400 });
          if (OUTS[gi].some((k) => k < S.gs)) V.text(g, '★', pad + 22, cy + 4, { size: 11, color: C.d });
          V.text(g, br(A[bs], 2), pad + lw, cy + 4, { size: 11, font: 'mono', align: 'right', color: scanning ? C.muted : C.e, weight: 600 });
        }
        g.strokeStyle = C.c; g.lineWidth = 2;
        V.rr(g, mx0 + 0.5, my0 + sel * ch + 0.5, mw - 1, ch - 1, 4); g.stroke();
        if (scanning) { g.strokeStyle = C.c; g.lineWidth = 2; V.rr(g, mx0 + cur * cw, my0 - 4, cw, NG * ch + 8, 4); g.stroke(); }
        [0, 5, 10, 15].forEach((i) => V.text(g, br(A[i], 2), mx0 + (i + 0.5) * cw, tabBot + 14, { size: 11, font: 'mono', align: 'center', color: C.muted }));
        // bars: how many groups chose each clip
        const by0 = tabBot + 22, mxC = Math.max(1, ...counts);
        V.line(g, mx0, by0 + barH, mx0 + mw, by0 + barH, { color: V.alpha(C.muted, 0.5), lw: 1 });
        counts.forEach((c, i) => {
          if (!c) return;
          const bh = (c / mxC) * (barH - 14);
          g.fillStyle = V.alpha(C.e, 0.75);
          V.rr(g, mx0 + i * cw + 2, by0 + barH - bh, cw - 4, bh, 2); g.fill();
          V.text(g, String(c), mx0 + (i + 0.5) * cw, by0 + barH - bh - 3, { size: 11, font: 'mono', align: 'center', color: C.ink });
        });
        V.text(g, '↑ how many groups chose each clip', mx0, by0 + barH + 15, { size: 11, color: C.muted, maxW: mw + chipW });
        // ---------- highlighted group panel
        const rx = narrow ? pad : pad + lw + 26, ry = narrow ? by0 + barH + 34 : 8;
        const rw = narrow ? w - 2 * pad : w - rx - pad, rh = h - ry - 6;
        const ai = scanning ? cur : BEST[sel], a = A[ai], col = scanning ? C.c : C.e;
        const { s, b } = rule(sel, a), full = rule(sel, 1);
        const gv = grp(sel), [lo, hi] = LOHI[sel], span = hi - lo || 1, top = b + 15 * s;
        const p1h = Math.min(rh * 0.5, 190);
        V.text(g, 'GROUP g' + sel + ' · ' + (scanning ? 'testing' : 'chosen') + ' a = ' + br(a, 2), rx, ry + 10, { size: 11, font: 'mono', color: col, weight: 700, maxW: rw });
        const ax0 = rx + 56, axw = rw - 62;
        const X = (v) => ax0 + ((v - (lo - 0.04 * span)) / (1.08 * span)) * axw;
        const bandY = ry + 24, bandH = Math.max(36, Math.min(64, p1h - 96));
        V.line(g, X(0), bandY - 4, X(0), bandY + bandH + 40, { color: V.alpha(C.ink, 0.35), lw: 1, dash: [3, 3] });
        V.text(g, '0', X(0), bandY + bandH + 52, { size: 11, font: 'mono', align: 'center', color: C.muted });
        // weights: at most ~256 dots, plus every one the ruler clips
        const step = Math.max(1, Math.floor(gv.length / NDOT));
        for (let k = 0; k < gv.length; k++) {
          const x = gv[k], cl = x < b - s / 2 || x > top + s / 2;
          if (k % step && !cl) continue;
          const jy = bandY + 6 + (((k * 37) % 23) / 22) * (bandH - 12);
          V.dot(g, X(x), jy, cl ? 3.6 : 2.4, cl ? C.red : V.alpha(C.b, 0.85));
        }
        // rulers: a = 1.00 (ghost) and the a under test or chosen
        const t1 = bandY + bandH + 8, t2 = t1 + 16;
        V.text(g, 'a=' + br(1, 2), rx, t1 + 8, { size: 11, font: 'mono', color: C.muted });
        V.text(g, 'a=' + br(a, 2), rx, t2 + 9, { size: 11, font: 'mono', color: col, weight: 600 });
        for (let j = 0; j < 16; j++) {
          V.line(g, X(full.b + j * full.s), t1, X(full.b + j * full.s), t1 + 8, { color: V.alpha(C.muted, 0.6), lw: 1.2 });
          V.line(g, X(b + j * s), t2, X(b + j * s), t2 + 10, { color: col, lw: 1.6 });
        }
        // ---------- the group's “U”, with 16 points (log scale)
        const p2y = ry + p1h + 14, p2h = rh - p1h - 14;
        V.text(g, 'MSE OF GROUP g' + sel + ' × CLIP (log scale)', rx, p2y + 10, { size: 11, font: 'mono', color: C.muted, weight: 600, maxW: rw });
        const bx = rx + 6, bw = rw - 12, byy = p2y + 22, bhh = Math.max(40, p2h - 52);
        V.box(g, bx, byy, bw, bhh, { r: 6, fill: V.alpha(C.muted, 0.05), stroke: V.alpha(C.muted, 0.2), lw: 1 });
        const row = E[sel].map((v) => Math.log(Math.max(v, 1e-30))), mn = Math.min(...row), mxv = Math.max(...row);
        const UX = (i) => bx + 14 + (i / (NA - 1)) * (bw - 28), UY = (v) => byy + bhh - 10 - ((v - mn) / (mxv - mn || 1)) * (bhh - 22);
        g.beginPath(); row.forEach((v, i) => { if (i) g.lineTo(UX(i), UY(v)); else g.moveTo(UX(i), UY(v)); }); g.strokeStyle = V.alpha(C.b, 0.8); g.lineWidth = 2; g.stroke();
        row.forEach((v, i) => V.dot(g, UX(i), UY(v), 2.6, C.b));
        const bi = BEST[sel];
        if (!scanning || cur >= bi) { V.glow(g, UX(bi), UY(row[bi]), 10, C.e, 0.5); V.dot(g, UX(bi), UY(row[bi]), 5, C.e); }
        if (scanning) { V.glow(g, UX(cur), UY(row[cur]), 12, C.c, 0.6); V.dot(g, UX(cur), UY(row[cur]), 4.5, C.c); }
        V.text(g, br(1, 2), bx + 4, byy + bhh + 14, { size: 11, font: 'mono', color: C.muted });
        V.text(g, 'a (clip)', bx + bw / 2, byy + bhh + 14, { size: 11, font: 'mono', align: 'center', color: C.muted });
        V.text(g, br(0.55, 2), bx + bw - 4, byy + bhh + 14, { size: 11, font: 'mono', align: 'right', color: C.muted });
      });
    })();

    /* ======================================================================
       Figura 5 — A tesoura do TP e as escalas
       ====================================================================== */
    (function () {
      const fig = $('#fig-tesoura'); if (!fig) return;
      mobileH(fig, 400);
      const S = { mode: 'canal', dim: 640, cut: 'col' };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'TP cut', options: [{ v: 'col', label: 'column-parallel (rows)' }, { v: 'row', label: 'row-parallel (K)' }], value: S.cut, onChange: (v) => { S.cut = v; } });
      UI.seg(ctl, { label: 'Scale', options: [{ v: 'canal', label: 'per output channel' }, { v: 'bloco', label: 'block 128×128' }], value: S.mode, onChange: (v) => { S.mode = v; } });
      UI.seg(ctl, { label: 'Split dimension', options: [{ v: 640, label: '640 (Qwen shared expert)' }, { v: 1024, label: '1024' }], value: S.dim, onChange: (v) => { S.dim = v; } });
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const narrow = w < 560;
        const col = S.cut === 'col';
        const Rr = col ? S.dim : 2560, Kc = col ? 2560 : S.dim; // col: gate/up [640 × 2560]; row: down [2560 × 640]
        const rb = Rr / 128, cb = Kc / 128;
        const half = S.dim / 2;
        const aligned = half % 128 === 0;
        const ph = (t % 7) / 7;
        const cut = V.clamp(ph / 0.22, 0, 1), sep = V.ease.inOut(V.clamp((ph - 0.25) / 0.2, 0, 1));
        const mx = 16, my = 48;
        const scW = col ? 18 : 40; // largura da coluna de escalas por canal
        const mw = narrow ? w - 32 - (S.mode === 'canal' ? scW + 24 : 70) : Math.min(460, w * 0.56);
        const mh = h - my - 72;
        const gapSep = 16 * sep;
        // geometria: col corta linhas (vertical), row corta colunas (horizontal)
        const rowPx = mh / Rr, colPx = (mw - (col ? 0 : 16)) / Kc;
        const yOf = (row) => my + row * rowPx + (col && row >= half ? gapSep : 0);
        const xOf = (c) => mx + c * (col ? mw / Kc : colPx) + (!col && c >= half ? gapSep : 0);
        const cutIdx = (r, c) => (col ? r : c); // índice na dimensão cortada
        const rank0 = C.a, rank1 = C.b;
        const title = col
          ? (narrow ? `gate/up [${Rr} × ${Kc}] · TP=2 · ${half} rows/rank` : `gate/up_proj [${Rr} × ${Kc}] at TP=2 · rank 0 = rows 0–${half - 1}, rank 1 = ${half}–${Rr - 1}`)
          : (narrow ? `down [${Rr} × ${Kc}] · TP=2 · K = ${half}/rank` : `down_proj [${Rr} × ${Kc}] at TP=2 · rank 0 = K 0–${half - 1}, rank 1 = K ${half}–${Kc - 1}`);
        V.text(g, title, mx, 18, { size: 11.5, font: 'mono', color: C.muted, weight: 600, maxW: w - 2 * mx });
        if (S.mode === 'bloco') {
          // ladrilhos 128×128; o que atravessa a fronteira pisca em vermelho
          for (let a = 0; a < rb; a++) for (let b = 0; b < cb; b++) {
            const r0 = a * 128, r1 = r0 + 128, c0 = b * 128, c1 = c0 + 128;
            const lo = col ? r0 : c0, hi = col ? r1 : c1;
            const straddle = lo < half && hi > half;
            const segs = straddle ? [[lo, half], [half, hi]] : [[lo, hi]];
            segs.forEach(([s0, s1]) => {
              const x = col ? xOf(c0) : xOf(s0), xe = col ? xOf(c1 - 1) + mw / Kc : xOf(s1 - 1) + colPx;
              const y = col ? yOf(s0) : yOf(r0), ye = col ? yOf(s1 - 1) + rowPx : yOf(r1 - 1) + rowPx;
              const base = s0 < half ? rank0 : rank1;
              g.fillStyle = straddle ? V.alpha(C.red, 0.25 + 0.2 * Math.sin(t * 4) ** 2) : V.alpha(base, 0.18);
              g.fillRect(x + 1, y + 1, xe - x - 2, ye - y - 2);
              g.strokeStyle = straddle ? C.red : V.alpha(base, 0.6); g.lineWidth = 1; g.strokeRect(x + 1, y + 1, xe - x - 2, ye - y - 2);
            });
          }
        } else {
          // por canal: só faixas finas por linha de saída, sem ladrilhos
          const nb = 40;
          for (let k = 0; k < nb; k++) {
            const r0 = (k * Rr) / nb, r1 = ((k + 1) * Rr) / nb;
            if (col) {
              const base = r0 < half ? rank0 : rank1;
              const y = yOf(r0), ye = yOf(r1) - (r1 === half ? gapSep : 0);
              g.fillStyle = V.alpha(base, 0.1 + 0.12 * Math.abs(Math.sin(k * 0.9)));
              g.fillRect(mx, y, mw, Math.max(1, ye - y - 0.6));
            } else {
              [[0, half, rank0], [half, Kc, rank1]].forEach(([c0, c1, base]) => {
                const x = xOf(c0), xe = xOf(c1 - 1) + colPx;
                g.fillStyle = V.alpha(base, 0.1 + 0.12 * Math.abs(Math.sin(k * 0.9)));
                g.fillRect(x, yOf(r0), xe - x, Math.max(1, rowPx * (r1 - r0) - 0.6));
              });
            }
          }
        }
        // rótulos dos ranks sobre a matriz
        const pillY = narrow && col ? 10 : 0;
        const labs = col ? [[mx + 6, yOf(0) + 16, 'rank 0', rank0], [mx + 6, yOf(half) + 16 + pillY, 'rank 1', rank1]]
          : [[xOf(0) + 6, my + 16, 'rank 0', rank0], [xOf(half) + 6, my + 16, 'rank 1', rank1]];
        labs.forEach(([lx, ly, lab, c]) => { V.box(g, lx, ly - 12, 56, 17, { r: 6, fill: V.alpha(C.bg, 0.8) }); V.text(g, lab, lx + 28, ly, { size: 11.5, align: 'center', color: c, weight: 700, font: 'mono' }); });
        // tesoura
        if (col) {
          const cy = my + half * rowPx + gapSep / 2;
          V.line(g, mx - 6, cy, mx - 6 + (mw + 12) * cut, cy, { color: C.c, lw: 2.5, dash: [6, 4], dashOffset: -t * 30 });
          if (cut > 0 && cut < 1) V.text(g, '✂', mx - 6 + (mw + 12) * cut, cy + 5, { size: 16, align: 'center', color: C.c });
        } else {
          const cxx = mx + half * colPx + gapSep / 2;
          V.line(g, cxx, my - 6, cxx, my - 6 + (mh + 12) * cut, { color: C.c, lw: 2.5, dash: [6, 4], dashOffset: -t * 30 });
          if (cut > 0 && cut < 1) V.text(g, '✂', cxx, my - 6 + (mh + 12) * cut + 5, { size: 16, align: 'center', color: C.c });
        }
        // escalas
        const mRight = col ? mx + mw : xOf(Kc - 1) + colPx;
        const sx = narrow ? mRight + 12 : mRight + 30;
        if (S.mode === 'canal') {
          if (col) {
            for (let rr = 0; rr < Rr; rr += 4) {
              const y = yOf(rr), hgt = Math.max(1, rowPx * 4);
              const c = rr < half ? rank0 : rank1;
              g.fillStyle = V.alpha(c, 0.35 + 0.4 * Math.abs(Math.sin(rr * 0.07))); g.fillRect(sx, y, scW, hgt);
            }
            V.text(g, 's [' + Rr + ']', sx - 2, my - 6, { size: 11, font: 'mono', color: C.muted });
            if (!narrow) {
              V.text(g, `rank 0: ${half} scales`, sx + scW + 10, yOf(half / 2), { size: 12, color: rank0, weight: 600 });
              V.text(g, `rank 1: ${half} scales`, sx + scW + 10, yOf(half + half / 2), { size: 12, color: rank1, weight: 600 });
            }
          } else {
            // row-parallel: o vetor inteiro [2560] é replicado nos dois ranks
            const cw2 = scW / 2 - 2;
            [[0, rank0], [1, rank1]].forEach(([k, c]) => {
              const x = sx + k * (cw2 + 4) + (k ? gapSep * 0.5 : 0);
              for (let rr = 0; rr < Rr; rr += 16) { g.fillStyle = V.alpha(c, 0.35 + 0.4 * Math.abs(Math.sin(rr * 0.07))); g.fillRect(x, yOf(rr), cw2, Math.max(1, rowPx * 16)); }
            });
            V.text(g, 's [' + Rr + '] ×2', sx - 2, my - 6, { size: 11, font: 'mono', color: C.muted });
            if (!narrow) {
              V.text(g, `rank 0: ${Rr} scales (copy)`, sx + scW + 10, yOf(Rr * 0.3), { size: 12, color: rank0, weight: 600 });
              V.text(g, `rank 1: ${Rr} scales (copy)`, sx + scW + 10, yOf(Rr * 0.62), { size: 12, color: rank1, weight: 600 });
            }
          }
        } else {
          const cell = narrow ? 3 : Math.min(9, (w - sx - 20) / cb);
          const cellH = (128 * rowPx) - 2;
          for (let a = 0; a < rb; a++) for (let b = 0; b < cb; b++) {
            const r0 = a * 128, c0 = b * 128;
            const lo = col ? r0 : c0;
            const straddle = lo < half && lo + 128 > half;
            const y = yOf(r0) + (col && straddle ? gapSep / 2 : 0);
            g.fillStyle = straddle ? V.alpha(C.red, 0.7) : V.alpha(lo < half ? rank0 : rank1, 0.6);
            g.fillRect(sx + b * cell, y + 1, Math.max(1, cell - 1), Math.max(1, cellH));
          }
          V.text(g, `s [${rb} × ${cb}]`, sx - 2, my - 6, { size: 11, font: 'mono', color: C.muted });
        }
        // veredito
        const ok = S.mode === 'canal' || aligned;
        const bad0 = Math.floor(half / 128) * 128;
        const msg = S.mode === 'canal'
          ? (col ? `✓ per-channel scales are cut like the weight: ${half} + ${half}` : `✓ the per-output-channel scale is not cut: each rank keeps all ${Rr} (replica)`)
          : aligned ? `✓ boundary at ${half} = ${half / 128} × 128: each block belongs to one rank`
            : col ? `✕ boundary at ${half} falls inside block ${bad0}–${bad0 + 127}: one scale for two ranks`
              : `✕ K/rank = ${half} is not a multiple of 128: K block ${bad0}–${bad0 + 127} falls on two ranks`;
        const nL = Math.min(2, splitLines(g, msg, w - 2 * mx - 20, { size: 12, weight: 600 }).length);
        const bH = nL > 1 ? 44 : 28, bY = h - 12 - bH;
        V.box(g, mx, bY, w - 2 * mx, bH, { r: 8, fill: V.alpha(ok ? C.e : C.red, 0.14), stroke: ok ? C.e : C.red, lw: 1 });
        wrap(g, msg, mx + 10, bY + 19, w - 2 * mx - 20, 16, { size: 12, color: C.ink, weight: 600, maxLines: 2 });
      });
    })();

    /* ======================================================================
       Figura 6 — W4A16 × W8A8 dentro do kernel (stepper + roofline)
       ====================================================================== */
    (function () {
      const fig = $('#fig-kernel'); if (!fig) return;
      mobileH(fig, 540);
      const S = { mode: 'w4a16', M: 1 };
      const steps = [
        { t: 'Read from memory', d: 'In <b>W4A16</b> the weight travels in 4 bits (0.5 bytes per weight) and the activation in BF16. In <b>W8A8</b> the weight travels in FP8 (1 byte) and the activation still arrives in BF16. In <b>W4A8</b> the weight travels in 4 bits, as in W4A16. In decode, with a small M, these bytes are almost all of the step time.' },
        { t: 'Prepare the operands', d: '<b>W4A16</b>: each weight tile is dequantized in registers (\\(\\hat w = s\\cdot q\\)) and becomes BF16, without writing anything back to memory. <b>W8A8</b>: before the GEMM (in a separate kernel or fused into the previous one), each row of \\(X\\) goes through an amax, gets a scale \\(s^x_m = \\mathrm{amax}_m/448\\) and is rounded to FP8. This is where the activation error (QDQ) is born. <b>W4A8</b>: the weight is not dequantized; the activation goes through an amax every block of 32 and becomes MXFP8.' },
        { t: 'Multiply on the tensor core', d: '<b>W4A16</b>: MMA in BF16/FP16, at the 16-bit rate. <b>W8A8</b>: MMA in FP8, at twice the BF16 rate on GB10. <b>W4A8</b>: block-scaled MMA (<code>kind::mxf8f6f4</code>), E2M1 × E4M3 with the E8M0 scales applied by the hardware itself. All three accumulate in FP32.' },
        { t: 'Epilogue: apply the scales', d: '<b>W8A8</b>: \\(y_{mn} = s^x_m\\, s^w_n\\,\\mathrm{acc}_{mn}\\), because both scales came out of the sum. <b>W4A16</b>: the per-group scale already went into the dequantization; if it were per channel, it could be applied here. <b>W4A8</b>: the block scales stay inside the sum, and it is the MMA that applies them; the epilogue only converts the accumulator.' },
        { t: 'Who wins depends on M', d: 'With a small M (decode) the time goes into reading bytes, and whoever reads 4 bits wins (W4A16 and W4A8 tie). With a large M (prefill) the time goes into FLOPs, and the FP8 tensor core wins (W8A8 and W4A8 tie), as long as the activation error is acceptable. Change M and watch the roofline.' },
      ];
      let H;
      const st = UI.stepper(fig, steps, { autoplay: 7000, onStep: () => H && H.redraw() });
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Kernel', options: [{ v: 'w4a16', label: 'W4A16 (weight-only)' }, { v: 'w8a8', label: 'W8A8 dynamic' }, { v: 'w4a8', label: 'W4A8 (proposed in DS)' }], value: S.mode, onChange: (v) => { S.mode = v; upd(); } });
      UI.seg(ctl, { label: 'M (rows of X)', options: [1, 64, 4096], value: S.M, onChange: (v) => { S.M = v; } });
      const rW = UI.readout(ctl, 'weight read', '—'), rM = UI.readout(ctl, 'MMA', '—'), rE = UI.readout(ctl, 'activation error', '—');
      const INFO = {
        w4a16: ['0.5 B/weight', 'BF16', 'none'],
        w8a8: ['1 B/weight', 'FP8 (2×)', 'per-token QDQ'],
        w4a8: ['0.5 B/weight', 'FP4×FP8 (block)', 'QDQ per block of 32'],
      };
      function upd() { const f = INFO[S.mode]; rW.set(f[0]); rM.set(f[1]); rE.set(f[2]); }
      upd();
      const Kd = 4096, Nd = 4096, BWm = 273e9, BF = 125e12, F8 = 250e12;
      const MODES = [['W4A16', 'w4a16'], ['W8A8', 'w8a8'], ['W4A8', 'w4a8']];
      const times = (mode, M) => {
        const wB = mode === 'w8a8' ? 1 : 0.5;
        const bytes = Kd * Nd * wB + M * Kd * 2 + M * Nd * 2;
        const fl = 2 * M * Kd * Nd;
        return { mem: bytes / BWm * 1e6, cmp: fl / (mode === 'w4a16' ? BF : F8) * 1e6 };
      };
      const fmtUs = (us) => (us >= 1000 ? br(us / 1000, 2) + ' ms' : us >= 10 ? br(us, 0) + ' µs' : br(us, 2) + ' µs');
      H = V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const mode = S.mode;
        const a = mode === 'w4a16', w8 = mode === 'w8a8', w48 = mode === 'w4a8';
        const narrow = w < 600;
        const pad = 12;
        const topH = h - (narrow ? 196 : 158);
        const ks = narrow ? 0.62 : 1;
        const stages = ['memory', 'prep', 'tensor core', 'epilogue', 'output Y'];
        const n = stages.length;
        const colW = (w - 2 * pad) / n;
        const cx = (k) => pad + colW * k + colW / 2;
        const yW = 64, yX = topH - 36, yY = (yW + yX) / 2;
        // caixas
        stages.forEach((sname, k) => {
          const hl = st.i === k && k < 4;
          V.box(g, pad + colW * k + 3, 24, colW - 6, topH - 28, { r: 10, fill: V.alpha(hl ? C.c : C.muted, hl ? 0.12 : 0.05), stroke: V.alpha(hl ? C.c : C.muted, hl ? 0.8 : 0.25), lw: hl ? 1.8 : 1 });
          V.text(g, sname, cx(k), 16, { size: narrow ? 11 : 12, align: 'center', color: hl ? C.c : C.muted, weight: 600, maxW: colW - 4 });
        });
        // rótulos internos
        const lab = (k, y, s, col) => wrap(g, s, cx(k), y, colW - 8, 13, { size: 11, align: 'center', color: col || C.ink, maxLines: narrow ? 2 : 3 });
        lab(0, yW - 22, w8 ? 'W: FP8' : 'W: 4 bits', C.ink);
        lab(0, yX + 24, 'X: BF16', C.ink);
        lab(1, yW - 22, a ? (narrow ? 'q·s' : 'q·s → BF16') : w8 ? (narrow ? 'already FP8' : 'W is already FP8') : (narrow ? 'stays 4 bits' : 'W stays in 4 bits'), a ? C.c : C.muted);
        lab(1, yX + 24, a ? (narrow ? 'BF16' : 'X stays BF16') : w8 ? (narrow ? '→ FP8' : 'amax → sˣ → FP8') : (narrow ? '→ MXFP8' : 'amax/32 → MXFP8'), a ? C.muted : C.c);
        lab(2, yY + 40, a ? 'MMA BF16' : w8 ? 'MMA FP8 (2× rate)' : 'MMA FP4×FP8 with block scale', C.ink);
        lab(3, yY + 40, a ? '(× s, if per channel)' : w8 ? 'acc × sˣₘ · sʷₙ' : 'scales already applied in the MMA', a || w48 ? C.muted : C.c);
        lab(4, yY + 40, 'Y in BF16', C.ink);
        // gauge de amax quando a ativação é quantizada
        if (!a) {
          const gx = cx(1), gy = yX;
          const lvl = 0.5 + 0.45 * Math.sin(t * 3.1);
          V.box(g, gx - 26, gy - 16, 52, 8, { r: 4, fill: V.alpha(C.muted, 0.2) });
          V.box(g, gx - 26, gy - 16, 52 * lvl, 8, { r: 4, fill: C.d });
        }
        // partículas
        const nP = 5;
        const tile = (x, y, sz, col, lbl) => { V.box(g, x - sz / 2, y - sz / 2, sz, sz, { r: 3, fill: V.alpha(col, 0.85), stroke: col, lw: 1 }); if (lbl && sz >= 14) V.text(g, lbl, x, y + 4, { size: 11, align: 'center', color: C.dark ? '#00210F' : '#fff', weight: 700 }); };
        for (let p = 0; p < nP; p++) {
          const u = (t * 0.16 + p / nP) % 1;
          // lane W: 0..2
          const xw = V.lerp(cx(0), cx(2) - 12, u);
          const inReg = xw > cx(1) - 6;
          const szW = (a ? (inReg ? 22 : 11) : w8 ? 16 : 11) * ks;
          tile(xw, yW, szW, a ? (inReg ? C.b : C.c) : w8 ? C.a : C.c);
          // lane X
          const xx = V.lerp(cx(0), cx(2) - 12, (u + 0.1) % 1);
          const inReg2 = xx > cx(1) - 6;
          const szX = (!a && inReg2 ? 16 : 22) * ks;
          tile(xx, yX, szX, !a && inReg2 ? C.a : C.b);
          // saída
          const uo = (t * 0.16 + p / nP + 0.5) % 1;
          const xo = V.lerp(cx(2) + 32, cx(4), uo);
          tile(xo, yY, 18 * ks, uo > 0.45 && !a ? C.b : C.d);
        }
        // setas de fluxo
        V.arrow(g, cx(2) - 20, yW + 12, cx(2) - 8, yY - 26, { color: V.alpha(C.muted, 0.6), lw: 1.4, head: 6 });
        V.arrow(g, cx(2) - 20, yX - 12, cx(2) - 8, yY + 26, { color: V.alpha(C.muted, 0.6), lw: 1.4, head: 6 });
        // chip do tensor core
        const chip = Math.min(44, colW * 0.42), pulse = 0.5 + 0.5 * Math.sin(t * 5);
        const chipC = a ? C.b : w8 ? C.a : C.c;
        V.glow(g, cx(2), yY, chip, chipC, 0.25 + 0.25 * pulse);
        V.box(g, cx(2) - chip / 2, yY - chip / 2, chip, chip, { r: 8, fill: V.alpha(chipC, 0.35), stroke: chipC, lw: 1.6 });
        for (let q = 1; q < 4; q++) { const o = (q * chip) / 4; V.line(g, cx(2) - chip / 2 + o, yY - chip / 2 + 4, cx(2) - chip / 2 + o, yY + chip / 2 - 4, { color: V.alpha(C.ink, 0.25), lw: 1 }); V.line(g, cx(2) - chip / 2 + 4, yY - chip / 2 + o, cx(2) + chip / 2 - 4, yY - chip / 2 + o, { color: V.alpha(C.ink, 0.25), lw: 1 }); }
        // roofline
        const ry = topH + 14;
        V.text(g, narrow ? `ROOFLINE · 4096 × 4096 · M = ${S.M}` : `ILLUSTRATIVE ROOFLINE · 4096 × 4096 matrix · M = ${S.M}`, pad, ry, { size: 11, font: 'mono', color: st.i === 4 ? C.c : C.muted, weight: 600 });
        const T = MODES.map(([nm, md]) => ({ nm, md, T: times(md, S.M) }));
        const mxT = Math.max(...T.map((o) => Math.max(o.T.mem, o.T.cmp)));
        const labW = narrow ? 58 : 96, barX = pad + labW, barW = w - 2 * pad - labW - (narrow ? 118 : 130);
        T.forEach(({ nm, md, T: TT }, k) => {
          const sel = md === mode;
          const y = ry + (narrow ? 30 : 12) + k * (narrow ? 50 : 44);
          V.text(g, nm, pad, y + 18, { size: 12, font: 'mono', color: sel ? C.ink : C.muted, weight: sel ? 700 : 400 });
          const tot = Math.max(TT.mem, TT.cmp);
          [['memory', TT.mem, C.b], ['compute', TT.cmp, C.d]].forEach(([ln, v, col], q) => {
            const bw = Math.max(2, (v / mxT) * barW);
            const yy = y + q * 17;
            const dom = v === tot;
            V.box(g, barX, yy, bw, 13, { r: 4, fill: V.alpha(col, dom ? 0.85 : 0.3) });
            V.text(g, `${ln} ${fmtUs(v)}`, barX + bw + 6, yy + 11, { size: 11, font: 'mono', color: dom ? C.ink : C.muted });
          });
        });
        const tt = T.map((o) => Math.max(o.T.mem, o.T.cmp)), best = Math.min(...tt);
        const win = T.filter((o, k) => tt[k] <= best * 1.01).map((o) => o.nm);
        const bound = T[tt.indexOf(best)].T.mem >= T[tt.indexOf(best)].T.cmp ? 'bandwidth' : 'compute';
        const vtxt = win.length > 1 ? `→ ${win.join(' and ')} tie (${bound}-bound)` : `→ ${win[0]} wins (${bound}-bound)`;
        V.text(g, vtxt, narrow ? pad : w - pad, narrow ? ry + 16 : ry, { size: 11.5, align: narrow ? 'left' : 'right', color: C.e, weight: 600 });
      });
    })();

    /* ======================================================================
       Figura 7 — O porteiro de régua fixa e o alfaiate (estática × dinâmica)
       ====================================================================== */
    (function () {
      const fig = $('#fig-porteiro'); if (!fig) return;
      mobileH(fig, 430);
      const T = 10, D = 16, AT = 6, OC = 5; // AT: token atípico; OC: índice do canal atípico
      const r = V.rng(21);
      const mags = Array.from({ length: T }, () => Math.exp(r.normal() * 0.25));
      const base = Array.from({ length: T }, (_, i) => Array.from({ length: D }, () => r.normal() * mags[i]));
      let calTok = 0; base.forEach((row, i) => { if (i !== AT) row.forEach((v) => { calTok = Math.max(calTok, Math.abs(v)); }); });
      const S = { mode: 'static', lf: 1, kind: 'token', fmt: 'fp8' };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Activation scale', options: [{ v: 'static', label: 'static (calibrated)' }, { v: 'dyn', label: 'dynamic per token' }], value: S.mode, onChange: (v) => { S.mode = v; upd(); } });
      UI.seg(ctl, { label: 'Atypical', options: [{ v: 'token', label: 'one token' }, { v: 'canal', label: 'one channel' }], value: S.kind, onChange: (v) => { S.kind = v; upd(); } });
      UI.seg(ctl, { label: 'Format', options: [{ v: 'fp8', label: 'FP8 E4M3' }, { v: 'int8', label: 'INT8' }], value: S.fmt, onChange: (v) => { S.fmt = v; upd(); } });
      UI.slider(ctl, { label: 'Intensity', min: -1, max: 2, step: 0.05, value: 1, fmt: (v) => '×' + br(Math.pow(10, v), Math.pow(10, v) < 1 ? 2 : Math.pow(10, v) >= 10 ? 0 : 1), onInput: (v) => { S.lf = v; upd(); } });
      const rSat = UI.readout(ctl, 'saturated', '—'), rAt = UI.readout(ctl, 'atypical error', '—'), rMe = UI.readout(ctl, 'common-value error', '—'), rAm = UI.readout(ctl, 'amax per GEMM', '—');
      let Qd = null;
      const isOut = (i, d) => (S.kind === 'token' ? i === AT : d === OC);
      function compute() {
        const f = Math.pow(10, S.lf);
        const X = base.map((row, i) => row.map((v, d) => {
          if (S.kind === 'token') return i === AT ? v * f : v;
          return d === OC ? 2.5 * mags[i] * f : v; // canal persistente: mesmo índice, mesmo sinal, em todos os tokens
        }));
        const qmax = S.fmt === 'fp8' ? 448 : 127;
        // calibração: no modo "um token" ela não vê o atípico; no modo "um canal" ela vê (o canal está em todo token)
        let calA = calTok; if (S.kind === 'canal') { calA = 0; X.forEach((row) => row.forEach((v) => { calA = Math.max(calA, Math.abs(v)); })); }
        const sStat = calA / qmax;
        const qf = (x, s) => { if (S.fmt === 'fp8') return N.encode('e4m3', x / s).v * s; return Math.max(-127, Math.min(127, Math.round(x / s))) * s; };
        let seC = 0, sxC = 0, seO = 0, sxO = 0, sat = 0;
        const out = X.map((row, i) => {
          const am = Math.max(...row.map(Math.abs), 1e-12);
          const s = S.mode === 'static' ? sStat : am / qmax;
          let se = 0, sx = 0, sat_i = 0, cs = 0, cn = 0;
          row.forEach((x, d) => { if (!isOut(i, d) || S.kind === 'token') { cs += x * x; cn++; } });
          const rmsC = Math.sqrt(cs / Math.max(1, cn)) || 1e-12;
          const errs = [];
          const q = row.map((x, d) => {
            const v = qf(x, s);
            if (Math.abs(x / s) > qmax * (1 + 1e-9)) { sat_i++; }
            const e = (v - x) ** 2;
            const o = isOut(i, d);
            if (S.kind === 'token' || !o) { se += e; sx += x * x; }
            if (o) { seO += e; sxO += x * x; } else { seC += e; sxC += x * x; }
            errs.push(o && S.kind === 'canal' ? Math.abs(v - x) / Math.max(Math.abs(x), 1e-12) : Math.abs(v - x) / rmsC);
            return v;
          });
          sat += sat_i;
          return { q, s, am, rel: Math.sqrt(se / Math.max(sx, 1e-30)), sat: sat_i, errs, L: qmax * s };
        });
        Qd = { X, out, relC: Math.sqrt(seC / Math.max(sxC, 1e-30)), relO: Math.sqrt(seO / Math.max(sxO, 1e-30)), sat };
      }
      function upd() {
        compute();
        rSat.set(Qd.sat + ' of ' + T * D);
        rAt.set(pct(Qd.relO, 1));
        rMe.set(pct(Qd.relC, 1));
        rAm.set(S.mode === 'static' ? '0' : String(T));
      }
      upd();
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const pad = 12;
        const narrow = w < 560;
        const cw = (w - 2 * pad) / T;
        const top = narrow ? 60 : 48, bot = h - 62;
        const mid = (top + bot) / 2, half = (bot - top) / 2 - 6;
        let A = 0; Qd.X.forEach((row) => row.forEach((v) => { A = Math.max(A, Math.abs(v)); }));
        A = Math.max(A, calTok * 1.05);
        const Y = (v) => mid - Math.sign(v) * Math.sqrt(Math.abs(v) / A) * half;
        const stat = S.mode === 'static';
        const fmtN = S.fmt === 'fp8' ? '448' : '127';
        const head = stat ? `Static scale: s = amax_calibration / ${fmtN}, the same frame for everyone` : `Dynamic scale: sₜ = amaxₜ / ${fmtN}, a made-to-measure frame per token`;
        const nl = wrap(g, head, pad, 18, w - 2 * pad, 15, { size: 12, color: C.ink, weight: 600, maxLines: 2 });
        V.text(g, narrow ? `√|x| · ${S.fmt === 'fp8' ? 'FP8' : 'INT8'} · color = error${S.kind === 'canal' ? ' · channel ' + OC + ' atypical' : ''}` : `height ∝ √|x| · ${S.fmt === 'fp8' ? 'FP8 E4M3' : 'INT8 symmetric'} · color = error of each value${S.kind === 'canal' ? ' · channel ' + OC + ' atypical' : ''}`, pad, 18 + nl * 15, { size: 11, font: 'mono', color: C.muted, maxW: w - 2 * pad });
        V.line(g, pad, mid, w - pad, mid, { color: V.alpha(C.muted, 0.35), lw: 1 });
        Qd.X.forEach((row, i) => {
          const o = Qd.out[i];
          const x0 = pad + i * cw;
          const L = o.L;
          const atypTok = S.kind === 'token' && i === AT;
          // moldura (porta)
          const yT = Y(L), yB = Y(-L);
          const doorCol = stat ? C.d : C.e;
          const wob = o.sat ? Math.sin(t * 9) * 1.2 : 0;
          g.strokeStyle = V.alpha(doorCol, 0.9); g.lineWidth = 2;
          g.beginPath(); g.moveTo(x0 + 3 + wob, yB); g.lineTo(x0 + 3 + wob, yT); g.lineTo(x0 + cw - 3 + wob, yT); g.lineTo(x0 + cw - 3 + wob, yB); g.stroke();
          // barras: cor pelo erro do valor (lavanda = bem representado; vermelho = erro ≥ 30% do tamanho típico)
          const bw = (cw - 10) / D;
          row.forEach((x, d) => {
            const bx = x0 + 5 + d * bw;
            const yv = Y(x);
            const clipped = Math.abs(x) > L * (1 + 1e-9);
            const yq = Y(Math.sign(x) * Math.min(Math.abs(x), L));
            const outV = isOut(i, d);
            const baseC = outV ? C.c : C.b;
            g.fillStyle = V.alpha(V.mix(baseC, C.red, V.clamp(o.errs[d] / 0.3, 0, 1)), 0.85);
            g.fillRect(bx, Math.min(mid, yq), Math.max(1, bw - 0.8), Math.max(1, Math.abs(yq - mid)));
            if (clipped) { g.fillStyle = C.red; g.fillRect(bx, Math.min(yq, yv), Math.max(1, bw - 0.8), Math.abs(yv - yq)); }
          });
          // personagem
          const fx = x0 + cw / 2, fy = bot + 16;
          V.box(g, fx - Math.min(15, cw / 2 - 2), fy - 11, Math.min(30, cw - 4), 22, { r: 11, fill: V.alpha(atypTok ? C.c : C.b, 0.25), stroke: atypTok ? C.c : C.b, lw: 1.2 });
          const blink = (Math.floor(t * 1.3 + i * 0.37) % 7) === 0 ? 0.6 : 2;
          V.box(g, fx - 5, fy - 3, 2.6, blink * 2, { r: 1, fill: C.ink }); V.box(g, fx + 2.6, fy - 3, 2.6, blink * 2, { r: 1, fill: C.ink });
          const e = o.rel;
          V.text(g, pct(e, e < 0.1 ? 1 : 0), fx, fy + 26, { size: 11, font: 'mono', align: 'center', color: o.sat || e > 0.15 ? C.red : C.muted, weight: o.sat || e > 0.15 ? 700 : 400 });
        });
        V.text(g, S.kind === 'canal' ? 'error of each token\'s common values →' : 'error per token →', pad, h - 6, { size: 11, color: C.muted });
      });
    })();

    /* ======================================================================
       Figura 8 — RTN × GPTQ (varredura com Hessiano)
       ====================================================================== */
    (function () {
      const fig = $('#fig-gptq'); if (!fig) return;
      mobileH(fig, 660);
      const K = 12, NS = 128, NR = 64, SAL = [3, 8];
      const S = { bits: 3, corr: 0.8, Hk: 'cal', sal: true };
      function inv(A) {
        const n = A.length; const Mx = A.map((row, i) => row.concat(Array.from({ length: n }, (_, j) => (i === j ? 1 : 0))));
        for (let i = 0; i < n; i++) { let p = i; for (let k = i + 1; k < n; k++) if (Math.abs(Mx[k][i]) > Math.abs(Mx[p][i])) p = k; const tmp = Mx[i]; Mx[i] = Mx[p]; Mx[p] = tmp; const d = Mx[i][i]; for (let j = 0; j < 2 * n; j++) Mx[i][j] /= d; for (let k = 0; k < n; k++) if (k !== i) { const f = Mx[k][i]; if (f) for (let j = 0; j < 2 * n; j++) Mx[k][j] -= f * Mx[i][j]; } }
        return Mx.map((row) => row.slice(n));
      }
      let D = null;
      function compute() {
        const r = V.rng(7);
        const X = [];
        const c = S.corr, e = Math.sqrt(1 - c * c);
        for (let s = 0; s < NS; s++) { const f = [r.normal(), r.normal(), r.normal()]; const row = []; for (let k = 0; k < K; k++) row.push((c * f[k % 3] + e * r.normal()) * (S.sal && SAL.includes(k) ? 3 : 1)); X.push(row); }
        let Hc = Array.from({ length: K }, (_, i) => Array.from({ length: K }, (_, j) => X.reduce((a, row) => a + row[i] * row[j], 0) / NS));
        const damp = 0.01 * Hc.reduce((a, row, i) => a + row[i], 0) / K;
        Hc = Hc.map((row, i) => row.map((v, j) => v + (i === j ? damp : 0)));
        const Hu = S.Hk === 'cal' ? Hc : Hc.map((row, i) => row.map((_, j) => (i === j ? 1 : 0)));
        const rows = Array.from({ length: NR }, () => Array.from({ length: K }, () => r.normal() * 0.5));
        const qmax = (1 << (S.bits - 1)) - 1;
        const qf = (d) => { let s = 0; for (let a = 0; a < K; a++) for (let b = 0; b < K; b++) s += d[a] * Hc[a][b] * d[b]; return s; };
        const Qrow = (w) => { const amax = Math.max(...w.map(Math.abs), 1e-12), s = amax / qmax; return w.map((x) => Math.max(-qmax - 1, Math.min(qmax, Math.round(x / s))) * s); };
        // AWQ: α_k = (RMS da entrada k)^β, com β buscado na grade (usa o H disponível: com H = I, α = 1 e AWQ vira RTN)
        const rmsU = Hu.map((row, k) => Math.sqrt(row[k]));
        const awqRow = (w, beta) => { const al = rmsU.map((v) => Math.pow(v, beta)); const q = Qrow(w.map((x, k) => x * al[k])); return q.map((v, k) => v / al[k]); };
        const qfU = (d) => { let s = 0; for (let a = 0; a < K; a++) for (let b = 0; b < K; b++) s += d[a] * Hu[a][b] * d[b]; return s; };
        let bestB = 0, bestE = Infinity;
        for (let bi = 0; bi <= 20; bi++) { const beta = bi / 20; let se = 0; rows.forEach((w) => { const q = awqRow(w, beta); se += qfU(q.map((v, k) => v - w[k])); }); if (se < bestE - 1e-12) { bestE = se; bestB = beta; } }
        let eWr = 0, eWg = 0, eWa = 0, eOr = 0, eOg = 0, eOa = 0, nW = 0, nO = 0;
        let snap = null;
        rows.forEach((w, ri) => {
          const rtn = Qrow(w);
          const amax = Math.max(...w.map(Math.abs)), s = amax / qmax;
          const Q = (x) => Math.max(-qmax - 1, Math.min(qmax, Math.round(x / s))) * s;
          let Hi = inv(Hu); const ww = w.slice(); const gq = new Array(K);
          const snaps = [ww.slice()], ratios = [];
          for (let j = 0; j < K; j++) {
            ratios.push(Hi[j].map((v, k) => (k > j ? v / Hi[j][j] : NaN)));
            gq[j] = Q(ww[j]);
            const ej = (ww[j] - gq[j]) / Hi[j][j];
            for (let k = j + 1; k < K; k++) ww[k] -= ej * Hi[j][k];
            const nh = Hi.map((row) => row.slice());
            for (let a = 0; a < K; a++) for (let b = 0; b < K; b++) nh[a][b] = Hi[a][b] - (Hi[a][j] * Hi[j][b]) / Hi[j][j];
            Hi = nh;
            snaps.push(ww.map((v, k) => (k <= j ? gq[k] : v)));
          }
          const aw = awqRow(w, bestB);
          for (let k = 0; k < K; k++) { eWr += (rtn[k] - w[k]) ** 2; eWg += (gq[k] - w[k]) ** 2; eWa += (aw[k] - w[k]) ** 2; nW += w[k] ** 2; }
          eOr += qf(rtn.map((v, k) => v - w[k])); eOg += qf(gq.map((v, k) => v - w[k])); eOa += qf(aw.map((v, k) => v - w[k])); nO += qf(w);
          if (ri === 0) snap = { w, rtn, gq, snaps, ratios };
        });
        D = { Hc, Hu, snap, beta: bestB, wR: Math.sqrt(eWr / nW), wG: Math.sqrt(eWg / nW), wA: Math.sqrt(eWa / nW), oR: Math.sqrt(eOr / nO), oG: Math.sqrt(eOg / nO), oA: Math.sqrt(eOa / nO) };
        if (rB) rB.set(br(bestB, 2));
      }
      let rB = null;
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Bits', options: [2, 3, 4], value: S.bits, onChange: (v) => { S.bits = v; compute(); } });
      UI.slider(ctl, { label: 'Correlation', min: 0, max: 0.9, step: 0.05, value: S.corr, fmt: (v) => br(v, 2), onInput: (v) => { S.corr = v; compute(); } });
      UI.seg(ctl, { label: 'Hessian used', options: [{ v: 'cal', label: 'calibrated (XᵀX/n)' }, { v: 'syn', label: 'synthetic (identity)' }], value: S.Hk, onChange: (v) => { S.Hk = v; compute(); } });
      UI.toggle(ctl, { label: 'channels 3 and 8 salient (×3)', value: S.sal, onChange: (v) => { S.sal = v; compute(); } });
      rB = UI.readout(ctl, 'AWQ exponent', '—');
      compute();
      V.fig(fig, (g, w, h, t, dt, St, HH) => {
        const C = V.C();
        const narrow = w < 620;
        const pad = 14;
        // ---- Hessiano, com índices, diagonal contornada e escala √ para a diagonal não apagar o resto
        const hs = narrow ? Math.min(120, w * 0.34) : Math.min(170, h - 200, w * 0.24);
        const hx = pad + 16, hy = 30;
        const Hd = S.Hk === 'cal' ? D.Hc : D.Hu;
        // escala de cor linear, normalizada pelos canais comuns (os salientes saturam)
        const salOn = S.sal && S.Hk === 'cal';
        let hm = 1e-9; Hd.forEach((row, i) => row.forEach((v, j) => { if (!salOn || (!SAL.includes(i) && !SAL.includes(j))) hm = Math.max(hm, Math.abs(v)); }));
        const period = K * 0.6 + 2.2;
        const ph = (t % period) / 0.6;
        const jc = Math.min(K, Math.floor(ph));
        V.text(g, S.Hk === 'cal' ? 'H = XᵀX/n (calibrated)' : 'H = I (synthetic)', pad, 18, { size: 11.5, font: 'mono', color: C.muted, weight: 600 });
        const cell = hs / K, showAll = cell >= 12;
        const lbl = Array.from({ length: K }, (_, k) => (showAll || k % 2 === 0 ? String(k) : ''));
        const geo = V.matrix(g, { x: hx, y: hy, w: hs, h: hs, data: Hd, cmap: 'div', gap: 1, hiRow: jc < K ? jc : null,
          colorFn: (v) => V.cmap(V.clamp(v / hm, -1, 1), 'div'), rowLabels: lbl, colLabels: lbl });
        g.strokeStyle = V.alpha(C.ink, 0.75); g.lineWidth = 1;
        for (let k = 0; k < K; k++) { const [ccx, ccy] = geo.center(k, k); g.strokeRect(ccx - geo.cw / 2 + 0.5, ccy - geo.ch / 2 + 0.5, geo.cw - 1, geo.ch - 1); }
        // barra de cor (escala √|H|)
        const cbY = hy + hs + 22, cbW = Math.min(hs, 120);
        for (let k = 0; k < 30; k++) { const u = -1 + (2 * k) / 29; g.fillStyle = V.cmap(u, 'div'); g.fillRect(hx + (k * cbW) / 30, cbY, cbW / 30 + 0.6, 8); }
        V.text(g, '−' + br(hm, 1), hx, cbY + 20, { size: 11, font: 'mono', color: C.muted });
        V.text(g, '0', hx + cbW / 2, cbY + 20, { size: 11, font: 'mono', color: C.muted, align: 'center' });
        V.text(g, '+' + br(hm, 1), hx + cbW, cbY + 20, { size: 11, font: 'mono', color: C.muted, align: 'right' });
        // hover: valor da célula
        const p = HH.pointer;
        const hc = p.inside ? geo.cellAt(p.x, p.y) : null;
        const hoverTxt = hc ? `H[${hc[0]},${hc[1]}] = ${br(Hd[hc[0]][hc[1]], 2)}` : (jc < K ? `row ${jc} of H highlighted` : 'hover over a cell');
        const descX = narrow ? hx + hs + 14 : pad, descW = narrow ? w - hs - hx - pad - 14 : hs + 16;
        let dy = narrow ? hy + 8 : cbY + 38;
        V.text(g, hoverTxt, descX, dy, { size: 11.5, font: 'mono', color: hc ? C.c : C.ink, weight: 600, maxW: descW });
        dy += 18;
        wrap(g, S.Hk === 'cal'
          ? 'Diagonal (outlined): energy of each input channel. Off the diagonal: channels that move together, through which the error can be compensated.' + (S.sal ? ' Channels 3 and 8 are salient and saturate the color.' : '')
          : 'Identity: every channel counts the same and none compensates for another. The objective goes back to being the weight error.', descX, dy, descW, 15, { size: 11.5, color: C.muted, maxLines: narrow ? 9 : 7 });
        // ---- varredura
        const sx = narrow ? pad : hx + hs + 28;
        const sy = narrow ? hy + hs + 78 : 30;
        const labW = narrow ? 78 : 96;
        const cw = (w - sx - pad - labW) / K, ch = narrow ? 24 : 28;
        const snapNow = D.snap.snaps[Math.min(K, jc)];
        const mx = Math.max(...D.snap.w.map(Math.abs)) * 1.3;
        const ratio = jc < K ? D.snap.ratios[jc] : null;
        let rmx = 1e-9; if (ratio) ratio.forEach((v) => { if (!Number.isNaN(v)) rmx = Math.max(rmx, Math.abs(v)); });
        const rowsDef = [
          ['original w', D.snap.w, null],
          ['in progress', snapNow, 'work'],
          ['ŵ GPTQ', D.snap.gq.map((v, k) => (k < jc ? v : NaN)), 'gq'],
          ['ŵ RTN', D.snap.rtn, null],
          [narrow ? 'H⁻¹ⱼₖ/H⁻¹ⱼⱼ' : 'H⁻¹ⱼₖ / H⁻¹ⱼⱼ', ratio || new Array(K).fill(NaN), 'ratio'],
        ];
        V.text(g, 'GPTQ SWEEP OVER ONE ROW (column ' + Math.min(jc, K - 1) + ' of 0–' + (K - 1) + ')', sx, sy - 12, { size: 11, font: 'mono', color: C.muted, weight: 600, maxW: w - sx - pad });
        rowsDef.forEach(([lab, vals, kind], ri) => {
          const y = sy + ri * (ch + 6);
          V.text(g, lab, sx, y + ch / 2 + 4, { size: 11.5, font: 'mono', color: kind === 'ratio' ? C.c : C.muted, maxW: labW - 6 });
          vals.forEach((v, k) => {
            const x = sx + labW + k * cw;
            if (Number.isNaN(v)) { V.box(g, x + 1, y, cw - 2, ch, { r: 4, fill: V.alpha(C.muted, 0.08) }); return; }
            const fill = kind === 'ratio' ? V.cmap(V.clamp(v / rmx, -1, 1), 'div') : V.cmap(V.clamp(v / mx, -1, 1), 'div');
            V.box(g, x + 1, y, cw - 2, ch, { r: 4, fill });
            if (kind === 'work' && k > jc - 1 && Math.abs(v - D.snap.w[k]) > 1e-3) { V.text(g, v > D.snap.w[k] ? '▲' : '▼', x + cw - 7, y + 11, { size: 11, align: 'center', color: C.c }); }
            if (cw >= 31) V.text(g, br(v, 2), x + cw / 2, y + ch / 2 + 4, { size: 11, font: 'mono', align: 'center', color: C.ink });
          });
        });
        // cursor
        if (jc < K) {
          const x = sx + labW + jc * cw;
          g.strokeStyle = C.c; g.lineWidth = 2.5; V.rr(g, x - 1, sy - 3, cw + 2, 4 * (ch + 6) - 2, 5); g.stroke();
        }
        // setas: o erro da coluna j vai para as colunas k > j, com espessura ∝ |H⁻¹ⱼₖ/H⁻¹ⱼⱼ|
        const ay = sy + 5 * (ch + 6) - 4;
        if (ratio) {
          const xj = sx + labW + jc * cw + cw / 2;
          for (let k = jc + 1; k < K; k++) {
            const m = Math.abs(ratio[k]) / rmx;
            if (m < 0.04) continue;
            const xk = sx + labW + k * cw + cw / 2;
            V.arrow(g, xj, ay, xk, ay + 1, { color: V.alpha(ratio[k] < 0 ? C.b : C.a, 0.35 + 0.6 * m), lw: 0.8 + 4.2 * m, head: 5 + 3 * m, curve: Math.min(30, 8 + (xk - xj) * 0.1) });
          }
        }
        V.text(g, !ratio ? 'sweep complete' : jc === K - 1 ? 'last column: no one left ahead to compensate' : 'arrows: error of column ' + jc + ' absorbed by each column ahead', sx + labW, ay + 34, { size: 11, color: C.muted, maxW: w - sx - labW - pad });
        // barras
        const by = ay + 56;
        const bh = h - by - 22;
        if (bh > 30) {
          V.text(g, 'MEAN OVER 64 ROWS · rel RMSE', sx, by - 8, { size: 11, font: 'mono', color: C.muted, weight: 600 });
          const groups = [['weight error', D.wR, D.wG, D.wA], ['output error', D.oR, D.oG, D.oA]];
          const mxv = Math.max(D.wR, D.wG, D.wA, D.oR, D.oG, D.oA) * 1.15;
          const gw = (w - sx - pad) / 2;
          groups.forEach(([lab, a, b, c2], gi) => {
            const gx = sx + gi * gw;
            const bw = Math.min(40, (gw - 30) / 3.6);
            [[a, C.b, 'RTN'], [b, C.a, 'GPTQ'], [c2, C.d, 'AWQ']].forEach(([v, col, nm], k) => {
              const x = gx + 10 + k * (bw + 10);
              const hh = (v / mxv) * (bh - 58);
              V.box(g, x, by + bh - 16 - hh, bw, hh, { r: 5, fill: V.alpha(col, 0.8) });
              V.text(g, pct(v, 1), x + bw / 2, by + bh - 20 - hh, { size: 11, font: 'mono', align: 'center', color: C.ink, maxW: bw + 8 });
              V.text(g, nm, x + bw / 2, by + bh - 2, { size: 11, font: 'mono', align: 'center', color: C.muted });
            });
            V.text(g, lab, gx + 10, by + 8, { size: 12, color: C.ink, weight: 600 });
          });
        }
      });
    })();

    /* ======================================================================
       Figura 9 — QAT no KV do DeepSeek (stepper)
       ====================================================================== */
    (function () {
      const fig = $('#fig-qat'); if (!fig) return;
      mobileH(fig, 470);
      const ex = [3.7, -1.2, 0.05, 0.9];
      const amax = Math.max(...ex.map(Math.abs));
      const e = Math.ceil(Math.log2(amax / 448));
      const s = Math.pow(2, e);
      const xs = ex.map((x) => x / s);
      const q8 = xs.map((v) => N.encode('e4m3', v).v);
      const xh = q8.map((v) => v * s);
      const r = V.rng(9);
      const kv = Array.from({ length: 512 }, (_, i) => (i < 448 ? r.normal() * (0.6 + 0.5 * Math.abs(Math.sin(i / 64))) : r.normal() * 0.8));
      [0, 17, 33, 50].forEach((p, k) => { kv[p] = ex[k]; });
      const steps = [
        { t: 'One KV entry', d: 'In DeepSeek-V4, each KV entry of a layer (raw in the sliding window, compressed in the C4A/C128A layers) is a 512-dim vector: 448 NoPE (7 groups of 64) and 64 of RoPE. The published training forward applies <code>act_quant(..., inplace=True)</code> only to the NoPE dims, after the norm and RoPE.' },
        { t: 'A group of 64 and its amax', d: 'For each group, \\(\\mathrm{amax} = \\max|x|\\) (with a floor of \\(10^{-4}\\)). In the inventory example, four of the 64 values are \\([3.7,\\ -1.2,\\ 0.05,\\ 0.9]\\), and the amax is 3.7.' },
        { t: 'Power-of-2 scale, rounded up', d: '\\(3.7/448 = 0.00826\\), \\(\\log_2 = -6.92\\), ceiling \\(= -6\\). So \\(s = 2^{-6} = 0.015625\\), stored as the UE8M0 byte \\(-6+127 = 121\\). Rounding up guarantees \\(\\mathrm{amax}/s \\le 448\\): nothing saturates.' },
        { t: 'Quantize and back again (QDQ)', d: '\\(x/s = [236.8,\\ -76.8,\\ 3.2,\\ 57.6]\\) → E4M3 grid → \\([240,\\ -80,\\ 3.25,\\ 56]\\) → \\(\\times s\\) → \\(\\hat x = [3.75,\\ -1.25,\\ 0.0508,\\ 0.875]\\). Training continues with \\(\\hat x\\) in high precision, but with the values that FP8 can represent.' },
        { t: 'The gradient goes through (STE)', d: 'Rounding has zero derivative, so the backward pass uses the straight-through estimator: \\(\\partial \\hat x/\\partial x \\approx 1\\) inside the ruler. The weights that produce the KV learn to tolerate this specific rounding.' },
        { t: 'At serving time: the same format', d: 'The KV is stored as 448 B of FP8 + 128 B of RoPE in BF16 + 8 B of scales (7 UE8M0 + 1 padding) = 584 B per KV entry, in each layer. It is the training regime. With <code>VLLM_DSPARK_REFERENCE_KV_QUANT_DEQUANT=1</code>, <code>_dspark_quant_dequant_nope_kernel</code> applies the same QDQ to the drafter\'s KV, to check parity.' },
      ];
      let H;
      const st = UI.stepper(fig, steps, { autoplay: 6500, onStep: () => H && H.redraw() });
      H = V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const i = st.i, el = st.elapsed();
        const pad = 14;
        const narrow = w < 560;
        // faixa do KV
        const ky = 30, kh = 46;
        V.text(g, narrow ? 'KV entry (1 layer): 512 dims' : 'One KV entry in one layer: 512 dims', pad, 18, { size: 11.5, font: 'mono', color: C.muted, weight: 600 });
        const segW = (w - 2 * pad) / 8;
        for (let sgi = 0; sgi < 8; sgi++) {
          const x = pad + sgi * segW;
          const rope = sgi === 7;
          const hl = sgi === 0 && i >= 1;
          V.box(g, x + 2, ky, segW - 4, kh, { r: 6, fill: V.alpha(rope ? C.b : C.a, hl ? 0.28 : 0.1), stroke: hl ? C.c : V.alpha(rope ? C.b : C.a, 0.5), lw: hl ? 2 : 1 });
          const bw = (segW - 10) / 64;
          for (let k = 0; k < 64; k++) {
            const v = kv[sgi * 64 + k];
            const qd = !rope && i >= 3 ? 1 : 0;
            const hh = Math.min(1, Math.abs(v) / 3.8) * (kh / 2 - 4);
            g.fillStyle = V.alpha(rope ? C.b : qd ? C.e : C.a, 0.85);
            g.fillRect(x + 5 + k * bw, ky + kh / 2 - (v > 0 ? hh : 0), Math.max(0.6, bw - 0.3), hh);
          }
          V.text(g, rope ? (narrow ? 'RoPE' : 'RoPE · BF16') : (narrow ? 'N' + sgi : 'NoPE ' + sgi), x + segW / 2, ky + kh + 14, { size: 11, align: 'center', color: rope ? C.b : C.muted, font: 'mono' });
        }
        // pipeline
        const py = ky + kh + 38;
        const boxes = ['amax', 's = 2^ceil(log₂(amax/448))', 'x/s → E4M3', '× s → x̂'];
        const cols = narrow ? 2 : 4;
        const bw = (w - 2 * pad - (cols - 1) * 18) / cols;
        const pyOf = (k) => py + (narrow ? Math.floor(k / 2) * 44 : 0);
        boxes.forEach((b, k) => {
          const x = pad + (k % cols) * (bw + 18);
          const py = pyOf(k);
          const on = (i === 1 && k === 0) || (i === 2 && k === 1) || (i === 3 && k >= 2) || i === 5;
          V.box(g, x, py, bw, 34, { r: 8, fill: V.alpha(on ? C.c : C.muted, on ? 0.18 : 0.06), stroke: on ? C.c : V.alpha(C.muted, 0.3), lw: on ? 1.6 : 1 });
          const nl = Math.min(2, splitLines(g, b, bw - 8, { size: narrow ? 11 : 12, font: 'mono' }).length);
          wrap(g, b, x + bw / 2, py + (nl > 1 ? 14 : 21), bw - 8, 13, { size: narrow ? 11 : 12, align: 'center', color: C.ink, font: 'mono', maxLines: 2 });
          if (k < 3 && (!narrow || k % 2 === 0)) V.arrow(g, x + bw + 2, py + 17, x + bw + 16, py + 17, { color: C.muted, lw: 1.4, head: 6 });
          if (narrow && k === 1) V.arrow(g, x + bw / 2, py + 36, pad + bw / 2, py + 44, { color: C.muted, lw: 1.4, head: 6 });
        });
        // fluxo de partículas no forward
        const pyEnd = py + (narrow ? 44 : 0);
        if (i >= 1 && i <= 3) { for (let p = 0; p < 4; p++) { const u = (t * 0.25 + p / 4) % 1; const x = V.lerp(pad + 6, w - pad - 6, u); V.dot(g, x, pyEnd + 42, 3, C.a); } }
        // backward STE
        if (i === 4) {
          const yb = pyEnd + 50;
          V.line(g, w - pad - 6, yb, pad + 6, yb, { color: C.f, lw: 2.2, dash: [8, 5], dashOffset: t * 30 });
          V.arrow(g, pad + 40, yb, pad + 6, yb, { color: C.f, lw: 2.2, head: 8 });
          V.text(g, 'gradient: ∂x̂/∂x ≈ 1 (STE), the round is “passed through”', w / 2, yb + 18, { size: 12, align: 'center', color: C.f, weight: 600, maxW: w - 2 * pad });
        }
        // tabela dos 4 valores
        const ty = pyEnd + (i === 4 ? 92 : 70);
        const rows = [['x', ex.map((v) => br(v, 2))]];
        if (i >= 2) rows.push(['s', ['2⁻⁶ = 0.015625', '', 'byte UE8M0 = ' + (e + 127), '']]);
        if (i >= 3) { rows.push(['x / s', xs.map((v) => br(v, 1))]); rows.push(['E4M3', q8.map((v) => br(v, 2))]); rows.push(['x̂ = E4M3 · s', xh.map((v) => br(v, 4))]); }
        if (i === 1) rows.push(['amax', [br(amax, 1), '', '', '']]);
        const lw = narrow ? 84 : 120, cw = (w - 2 * pad - lw) / 4, rh = 24;
        rows.forEach((row, ri) => {
          const y = ty + ri * rh;
          if (y + rh > h - 4) return;
          V.box(g, pad, y, w - 2 * pad, rh - 4, { r: 6, fill: V.alpha(ri === rows.length - 1 && i >= 1 ? C.a : C.muted, ri === rows.length - 1 && i >= 1 ? 0.12 : 0.06) });
          V.text(g, row[0], pad + 8, y + rh / 2 + 2, { size: 11.5, font: 'mono', color: C.muted, maxW: lw - 10 });
          if (row[0] === 's') { V.text(g, row[1][0] + '  ·  ' + row[1][2], pad + lw, y + rh / 2 + 2, { size: 12, font: 'mono', color: C.ink, maxW: w - 2 * pad - lw - 6 }); return; }
          row[1].forEach((v, k) => { if (v) V.text(g, v, pad + lw + cw * k + cw / 2, y + rh / 2 + 2, { size: 12, font: 'mono', align: 'center', color: C.ink }); });
        });
        // layout de 584 B: sempre visível, aceso no último passo
        const ly = h - 34, lw2 = w - 2 * pad, on5 = i === 5;
        {
          const parts = [[448, C.a, '448 B FP8 (NoPE)'], [128, C.b, '128 B RoPE BF16'], [8, C.d, '8 B']];
          let xx = pad;
          parts.forEach(([b, col, lab]) => {
            const pw = (b / 584) * lw2;
            V.box(g, xx + 1, ly, pw - 2, 24, { r: 5, fill: V.alpha(col, on5 ? 0.7 : 0.22) });
            if (pw > tw(g, lab, { size: 11 }) + 6) V.text(g, lab, xx + pw / 2, ly + 16, { size: 11, align: 'center', color: on5 ? (C.dark ? '#00210F' : '#fff') : C.muted, weight: 600 });
            xx += pw;
          });
          V.text(g, narrow ? '+ 8 B of scales = 584 B per entry' : '+ 8 B of scales (7 UE8M0 + 1 pad) = 584 B per KV entry (per layer)', w - pad, ly - 6, { size: 11.5, align: 'right', font: 'mono', color: on5 ? C.ink : C.muted, weight: 600, maxW: w - 2 * pad });
        }
        // zoom do grupo NoPE 0 no espaço livre entre a tabela e a barra
        const tEnd = ty + rows.length * rh + 10, zTop = tEnd + (narrow ? 30 : 14), zBot = ly - 26;
        if (zBot - zTop >= 44) {
          const zh = zBot - zTop, zmid = zTop + zh / 2, zx0 = pad, zw = w - 2 * pad;
          V.text(g, 'NoPE GROUP 0 ZOOMED IN (64 values)' + (i >= 3 ? ' · dots = x̂ after QDQ' : ''), zx0, zTop - (narrow ? 20 : 4), { size: 11, font: 'mono', color: C.muted, weight: 600, maxW: zw });
          V.line(g, zx0, zmid, zx0 + zw, zmid, { color: V.alpha(C.muted, 0.35), lw: 1 });
          const A0 = 3.9, Yz = (v) => zmid - (v / A0) * (zh / 2 - 4);
          const bwz = zw / 64;
          for (let k = 0; k < 64; k++) {
            const v = kv[k], y = Yz(v), key = ex.indexOf(v) >= 0;
            g.fillStyle = V.alpha(key ? C.c : C.a, key ? 0.9 : 0.55);
            g.fillRect(zx0 + k * bwz + 0.5, Math.min(y, zmid), Math.max(1, bwz - 1), Math.abs(y - zmid));
            if (i >= 3) { const q = N.encode('e4m3', v / s).v * s; V.dot(g, zx0 + k * bwz + bwz / 2, Yz(q), Math.min(2.4, bwz / 2.4), C.ink); }
          }
          if (i >= 1) {
            const pulse = i === 1 ? 0.6 + 0.4 * Math.sin(t * 4) : 0.8;
            [amax, -amax].forEach((v) => V.line(g, zx0, Yz(v), zx0 + zw, Yz(v), { color: V.alpha(C.d, pulse), lw: 1.4, dash: [5, 4] }));
            V.text(g, 'amax = ' + br(amax, 1) + (i >= 2 ? ' → 448·s = ' + br(448 * s, 2) : ''), zx0 + zw, narrow ? zTop - 5 : Yz(amax) - 4, { size: 11, font: 'mono', align: 'right', color: C.d, weight: 600 });
          }
        }
        void el;
      });
    })();

    /* ======================================================================
       Figura 10 — O mapa de precisões das três receitas
       ====================================================================== */
    (function () {
      const fig = $('#fig-mapa'); if (!fig) return;
      mobileH(fig, 580);
      const MAP = {
        glm: {
          variant: 'FP8=all (served default)', vDefault: true, vName: 'served',
          rows: [
            { l: 'output', items: [{ l: 'lm_head', f: 'bf16', why: 'Stays native BF16, even in the served state: it is not in any GLM53_DENSE_FP8 group. It is read in full every step.' }] },
            { l: 'FFN', items: [
              { l: 'experts ×288', f: 'q4', wt: 2.4, why: 'EXL3 K=4 (trellis + Hadamard, ≈ 4.01 bpw): 97% of the parameters. scope = glm53_routed_experts_only. They run as weight-only with FP16 MMA, never becoming BF16.' },
              { l: 'shared', f: 'bf16', f2: 'fp8', why: 'BF16 in the checkpoint. In the served default (shared group of GLM53_DENSE_FP8=all), it becomes per-channel FP8 (amax/448) at load time and runs on Marlin W8A16.' },
              { l: 'dense MLP', f: 'bf16', f2: 'fp8', why: 'The initial dense layers: BF16 in the checkpoint, per-channel FP8 in the served default (dense group).' },
            ] },
            { l: 'routing', items: [
              { l: 'router', f: 'bf16', why: 'BF16 weights and FP32 logits (moe_router_dtype float32): a small error swaps the top-8 and changes the computation discretely.' },
            ] },
            { l: 'residual', items: [
              { l: 'mHC', f: 'bf16', why: 'The hc_* weights mix the 4 residual streams (hyper-connections); they route nothing. Tiny and applied in every layer: they stay BF16, and even the official FP8 keeps them.' },
              { l: 'norms', f: 'bf16', why: 'Tiny and multiplicative. Even the official FP8 leaves them in BF16.' },
            ] },
            { l: 'mixer', items: [
              { l: 'KDA ×34', f: 'bf16', f2: 'fp8', wt: 1.4, why: 'BF16 in the checkpoint. The served default (kda group) puts in_proj/f_b/g_b/o_proj in FP8 W8A16 (together with the dense group, ~−11 ms/step). With GLM53_KDA_BF16_LARGE_M=1, also the default, prefill with M > 512 runs a BF16 copy of the same logical FP8 matrix in cuBLAS (+3.29 GiB per rank).' },
              { l: 'MLA q/o ×11', f: 'bf16', f2: 'fp8', why: 'fused_qkv_a, q_b and o_proj: BF16 in the checkpoint, per-channel FP8 in the mla group of the served default.' },
              { l: 'kv_b_proj', f: 'bf16', why: 'Never quantized: in decode, MLA\'s weight absorption reads this matrix directly. Exl3Config refuses, and the mla group excludes it.' },
              { l: 'indexer', f: 'bf16', why: 'Selects which tokens sparse attention reads. It stays BF16, and the official FP8 keeps it too.' },
            ] },
            { l: 'input', items: [{ l: 'embeddings', f: 'bf16', why: 'One row looked up per token, with negligible bandwidth cost. It stays native.' }] },
          ],
          side: [
            { l: 'KV MLA', f: 'fp8', why: 'fp8_ds_mla: 512 FP8 + 4 FP32 scales (one per 128) + 64 zeroed BF16 RoPE = 656 B per token per layer.' },
            { l: 'KDA state', f: 'fp32', why: 'Recurrent state in FP32 (conv in BF16). Constant size across the context, and the error would accumulate with every token.' },
            { l: 'DFlash2', f: 'bf16', why: 'Drafter in BF16 by default. In the H3 profile it goes to EXL3 6 bpw without calibration (synthetic Hessian), with K/V in BF16.' },
          ],
        },
        qwen: {
          variant: 'FP8-dense (opt-in)', vDefault: false, vName: 'variant',
          rows: [
            { l: 'output', items: [{ l: 'lm_head', f: 'bf16', f2: 'fp8', why: '26% of the step\'s bytes (MTP rereads it three times). BF16 in the nvidia checkpoint; in FP8-dense it is part of QUANT_PATTERNS.' }] },
            { l: 'FFN', items: [
              { l: 'experts ×512', f: 'q4', wt: 2.4, why: 'NVFP4 W4A4 (E2M1 + E4M3 every 16 + global FP32), MSE-calibrated weight scales; static input_scale and activation blocks computed on the fly.' },
              { l: 'shared', f: 'bf16', f2: 'fp8', why: 'BF16 in the checkpoint; gate/up/down are part of the FP8-dense QUANT_PATTERNS (per channel, dynamic per-token activation).' },
            ] },
            { l: 'routing', items: [
              { l: 'gates', f: 'bf16', why: 'mlp.gate and shared_expert_gate are in EXCLUDE_MODULES: they decide the routing.' },
            ] },
            { l: 'residual', items: [
              { l: 'HC', f: 'bf16', f2: 'fp8', why: 'HyperConnections mix the copies of the residual stream; they route nothing. down/inject/up become FP8 in FP8-dense; the final mixer\'s block_inject_weight stays BF16.' },
              { l: 'norms', f: 'bf16', why: 'q/k_norm, hc_norm, linear_attn.norm: excluded.' },
            ] },
            { l: 'mixer', items: [
              { l: 'GDN qkv/z/out ×36', f: 'bf16', f2: 'fp8', wt: 1.6, why: 'The largest dense projections (1.95 GiB per step). They become per-channel FP8 in FP8-dense.' },
              { l: 'GDN a/b + conv1d', f: 'bf16', why: 'in_proj_a/b/ba and conv1d feed the recurrence gates. The error would enter the state, so they are excluded.' },
              { l: 'QSA ×12', f: 'bf16', f2: 'fp8', why: 'Attention q/k/v/o: BF16 in the checkpoint, FP8 in FP8-dense.' },
              { l: 'indexer', f: 'bf16', why: 'Excluded: the top-k block selection is precision-sensitive.' },
            ] },
            { l: 'input', items: [
              { l: 'embeddings', f: 'bf16', why: 'embed_tokens stays BF16 (excluded).' },
              { l: 'PLE N-gram', f: 'fp8', wt: 1.6, why: '51.2 B parameters in per-tensor FP8 (47.68 GiB). _dequantize_embeddings dequantizes only the looked-up rows: emb.to(bf16) * weight_scale.' },
            ] },
          ],
          side: [
            { l: 'KV QSA', f: 'fp8', why: 'Per-tensor FP8 E4M3 (no calibrated scale in the checkpoint; probably 1.0, inferred): 1.70× tokens per GiB. It is PTQ, which is why it was validated at long context (12/12).' },
            { l: 'GDN state', f: 'bf16', why: 'MAMBA_SSM_CACHE_DTYPE=bfloat16 (the checkpoint ships FP32): half the bytes of the recurrent state.' },
            { l: 'MTP', f: 'mix', why: 'Experts in FP8 with 128×128 blocks and activation_scheme dynamic (patch); the rest in BF16. Excluded from FP8-dense.' },
          ],
        },
        ds: {
          variant: 'idx MXFP4 (opt-in)', vDefault: false, vName: 'variant',
          rows: [
            { l: 'output', items: [{ l: 'lm_head', f: 'bf16', why: 'BF16 (129,280 × 4096 ≈ 1.06 GB), shared with the DSpark drafter.' }] },
            { l: 'FFN', items: [
              { l: 'experts ×256', f: 'q4', wt: 2.4, why: 'The vendor\'s native MXFP4 (E2M1 + E8M0 every 32, 4.25 bits). b12x runs it as W4A16: it dequantizes to BF16 and uses BF16 MMA.' },
              { l: 'shared', f: 'fp8', why: 'FP8 with 128×128 blocks and UE8M0 scales (weight_scale_inv).' },
            ] },
            { l: 'routing', items: [
              { l: 'router', f: 'fp32', why: 'sqrtsoftplus scores computed in FP32.' },
            ] },
            { l: 'residual', items: [
              { l: 'mHC', f: 'fp32', why: 'Hyper-connections: they mix the residual streams, without routing anything. FP32 in the reference. Tiny and multiplicative.' },
            ] },
            { l: 'mixer', items: [
              { l: 'attention', f: 'fp8', wt: 1.6, why: 'wq_a/wkv, wq_b, wo_a/wo_b in FP8 with 128×128 UE8M0 blocks; the activation is quantized dynamically per block of 128 (act_quant) before each GEMM.' },
              { l: 'compressor', f: 'bf16', why: 'The compressor\'s wkv/wgate stay in BF16 in the checkpoint.' },
              { l: 'indexer', f: 'mix', why: 'Weights in FP8/BF16 (replicated in TP); query and K were simulated in FP4 during training (QAT), after a Hadamard rotation.' },
            ] },
            { l: 'input', items: [{ l: 'embeddings', f: 'bf16', why: 'BF16 (129,280 × 4096). Looked up by row, with negligible bandwidth cost, and shared with the drafter.' }] },
          ],
          side: [
            { l: 'KV NoPE', f: 'fp8', why: 'FP8 E4M3 + UE8M0 every 64 dims: the QAT format. RoPE in BF16. 584 B per KV entry, per layer (nvfp4_ds_mla = same bytes as fp8_ds_mla).' },
            { l: 'idx K cache', f: 'fp8', f2: 'q4', why: 'The indexer\'s K cache: FP8, 132 B per entry. Opt-in: MXFP4 (DSPARK_ENABLE_MXFP4_INDEXER_CACHE=1) reads half the bytes per key, at the risk of drift in the selection.' },
            { l: 'DSpark', f: 'mix', why: 'A 3-stage MoE drafter, with a mixed format. The parity knob applies the same FP8/64 QDQ from training to its KV.' },
          ],
        },
      };
      const S = { m: 'glm', variant: MAP.glm.vDefault, sel: null, pinned: null };
      const ctl = UI.controls(fig);
      let rv = null, tg = null;
      UI.seg(ctl, { label: 'Recipe', options: [{ v: 'glm', label: 'GLM' }, { v: 'qwen', label: 'Qwen' }, { v: 'ds', label: 'DeepSeek' }], value: S.m, onChange: (v) => { S.m = v; S.pinned = null; if (rv) rv.set(MAP[v].variant); if (tg) tg.set(MAP[v].vDefault); } });
      tg = UI.toggle(ctl, { label: 'variant on (GLM: served state)', value: S.variant, onChange: (v) => { S.variant = v; } });
      rv = UI.readout(ctl, 'variant', MAP.glm.variant);
      legendFormats(ctl);
      V.fig(fig, (g, w, h, t, dt, St, HH) => {
        const C = V.C(); const F = FC(C);
        const mp = MAP[S.m];
        const narrow = w < 600;
        const pad = 12, infoH = narrow ? 112 : 64;
        const rows = mp.rows.slice();
        const sideAsRow = narrow;
        const allRows = sideAsRow ? [{ l: 'state', items: mp.side }].concat(rows) : rows;
        const towerW = sideAsRow ? w - 2 * pad : w * 0.7 - pad;
        const rowLab = narrow ? 0 : 74;
        const top = 12, bottom = h - infoH - 10;
        const rh = (bottom - top) / allRows.length;
        const rects = [];
        allRows.forEach((row, ri) => {
          const y = top + ri * rh;
          if (!narrow) V.text(g, row.l, pad, y + rh / 2 + 4, { size: 11, font: 'mono', color: C.muted });
          const tot = row.items.reduce((a, it) => a + (it.wt || 1), 0);
          let x = pad + rowLab;
          const avail = towerW - rowLab;
          row.items.forEach((it) => {
            const iw = ((it.wt || 1) / tot) * avail;
            rects.push({ x: x + 3, y: y + 4, w: iw - 6, h: rh - 8, it });
            x += iw;
          });
        });
        if (!sideAsRow) {
          const sx = w * 0.7 + 8, sw = w - sx - pad;
          V.text(g, 'CACHES AND DRAFT', sx, top + 10, { size: 11, font: 'mono', color: C.muted, weight: 600 });
          const sh = (bottom - top - 22) / mp.side.length;
          mp.side.forEach((it, k) => rects.push({ x: sx, y: top + 22 + k * sh + 3, w: sw, h: sh - 8, it }));
        }
        // hover
        const p = HH.pointer;
        let hov = null;
        if (p.inside) rects.forEach((r) => { if (p.x >= r.x && p.x <= r.x + r.w && p.y >= r.y && p.y <= r.y + r.h) hov = r.it; });
        if (p.clicked) S.pinned = hov;
        const active = hov || S.pinned;
        rects.forEach((r) => {
          const it = r.it;
          const f = S.variant && it.f2 ? it.f2 : it.f;
          const col = F[f];
          const on = active === it;
          V.box(g, r.x, r.y, r.w, r.h, { r: 8, fill: V.alpha(col, on ? 0.45 : 0.28), stroke: col, lw: on ? 2.4 : 1.4 });
          if (S.variant && it.f2) { V.box(g, r.x + r.w - 12, r.y + 4, 8, 8, { r: 4, fill: C.c }); }
          const lab = it.l, fz = r.w < 70 ? 11 : 12;
          const cyy = r.y + r.h / 2;
          if (r.h >= 34) {
            V.text(g, lab, r.x + r.w / 2, cyy - 2, { size: fz, align: 'center', color: C.ink, weight: 600, maxW: r.w - 6 });
            V.text(g, FN[f], r.x + r.w / 2, cyy + 13, { size: 11, align: 'center', font: 'mono', color: C.ink, maxW: r.w - 6, alpha: 0.8 });
          } else {
            V.text(g, lab, r.x + r.w / 2, cyy + 4, { size: 11, align: 'center', color: C.ink, weight: 600, maxW: r.w - 6 });
          }
        });
        // token subindo pela torre
        const tp = (t * 0.22) % 1;
        const ty = V.lerp(bottom - 6, top + 6, tp);
        const tx = pad + rowLab + (towerW - rowLab) * (0.5 + 0.28 * Math.sin(tp * Math.PI * 4));
        V.glow(g, tx, ty, 14, C.d, 0.9); V.dot(g, tx, ty, 4, C.d);
        // caixa de explicação
        const iy = h - infoH - 2;
        V.box(g, pad, iy, w - 2 * pad, infoH - 4, { r: 10, fill: V.alpha(C.muted, 0.08), stroke: V.alpha(C.muted, 0.25), lw: 1 });
        const txt = active
          ? `${active.l} · ${FN[S.variant && active.f2 ? active.f2 : active.f]}${active.f2 ? ' (checkpoint: ' + FN[active.f] + ' → ' + mp.vName + ': ' + FN[active.f2] + ')' : ''}. ${active.why}`
          : (S.m === 'glm' && S.variant ? 'State served by .env.example (FP8=all + KDA BF16 copy). ' : S.m === 'glm' ? 'Checkpoint only: everything that is not an expert is in BF16. ' : '') + 'Hover over (or tap) a piece to see why it is at that precision. The pink dot marks the pieces the variant changes.';
        wrap(g, txt, pad + 10, iy + 18, w - 2 * pad - 20, 15, { size: 12, color: C.ink, maxLines: narrow ? 6 : 3 });
      });
    })();
  },
});

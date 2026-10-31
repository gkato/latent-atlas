LA.chapter({
  id: 'kv-cache-quantizado',
  part: 4,
  order: 23,
  title: 'Quantizing the KV cache',
  navTitle: 'Quantized KV cache',
  subtitle: 'Per-tensor and per-block scales, the algebra that pulls the scale out of the product, 656-, 584- and 528-byte MLA layouts, the “nvfp4” that is really FP8, the indexer cache in MXFP4 and the math that makes 1 million tokens fit.',
  minutes: 50,
  recipes: ['glm', 'tf', 'qwen', 'ds'],
  tags: ['KV cache', 'FP8', 'fp8_ds_mla', 'nvfp4_ds_mla', 'UE8M0', 'MXFP4', 'k_scale', 'indexer', 'hybrid allocator', 'kv8', 'TensorFold'],
  goals: [
    'Explain the two costs of the KV (capacity and bandwidth) and why quantization error in K and in V has different effects',
    'Quantize and dequantize KV in FP8 with a <b>per-tensor</b> scale (<code>k_scale</code>/<code>v_scale</code>) or a <b>per-block</b> scale (UE8M0 with ceiling) and show why the per-tensor scale comes out of the product in the kernel',
    'Assemble, byte by byte, the <code>fp8_ds_mla</code> layout (656 B, GLM) and the 584 B record of DeepSeek-V4, and understand why <code>nvfp4_ds_mla</code> is FP8 and what hotfix #22 fixes',
    'Predict the real capacity gain (1.70× and not 2×), the uniform page rule and the risk of <b>selection drift</b> when the keys of a sparse attention are quantized',
    'Compare <code>fp8_ds_mla</code> in the vLLM lane with the 528 B <b>kv8</b> row in the TensorFold lane and explain why a power-of-two scale per row loses almost nothing to finer scales and comes out of the product exactly',
  ],
  prereqs: ['fp8-fp4-microescala', 'mla', 'atencao-esparsa'],
  summary: [
    'The KV charges you twice: <b>capacity</b> (tokens × bytes/token ≤ pool) and <b>bandwidth</b> (every decode token re-reads what it needs from the cache). FP8 halves both, but what made 1M fit was the architecture; quantization multiplies on top of it.',
    'FP8 E4M3 with a <b>per-tensor</b> scale (Qwen: <code>k_scale</code>/<code>v_scale</code>, 1.0 without calibration) is cheap because the scale comes out of the sum: <code>scores *= k_scale</code> after the dot and <code>output *= v_scale</code> at the end, with an exact upcast to BF16 and <code>KV_QUANT_MODE</code>/<code>IS_FP8</code> as a <code>constexpr</code>. <b>Per-block</b> scales (GLM 4×FP32/128 dims; DS 7×UE8M0/64 dims, with a ceiling that never saturates) require kernels built for the layout.',
    'The real gain is smaller than the bit ratio: in Qwen, 3,652,200 vs 2,131,159 tokens (1.70×), because the indexer/compressor and the GDN state do not shrink; the state in BF16 (<code>MAMBA_SSM_CACHE_DTYPE</code>) changes the Mamba page and the block (1,600/3,200/1,664).',
    'In sparse attention, quantizing K can change <i>which</i> blocks are read: the risk is a wrong answer, not a crash (reference 6/6 → 2/6; fewer tok/step in 8/8 cells). The gate is needles + reasoning (<code>reasoning_check</code>, ruler-lite).',
    '<code>fp8_ds_mla</code> = 512 FP8 + 16 B of scales + 128 B of BF16 RoPE = 656 B; GLM (NoPE) fills the RoPE with zeros (exact, +24%) because it is the only sparse-MLA kernel on SM12x. DeepSeek-V4 writes 448 FP8 + 128 B RoPE + 8 B (7 UE8M0 + pad) = 584 B, and <code>nvfp4_ds_mla</code> is the same envelope: hotfix #22 only fixes the dispatch (1.0 → 17.3 tok/s at 600K+). In the TensorFold lane of GLM, an engine built for the family stores only the latent: 512 E4M3 + 1 power-of-two scale + 12 B of alignment = 528 B (kv8), with the scale applied exactly and a 1M window per request with 4 streams sharing the pool.',
    'The indexer K cache (132 B per compressed entry, replicated) dominates the reads at 1M; the opt-in MXFP4 (<code>use_fp4_indexer_cache</code>) reads half the bytes, but the allocation stays at 132 B and the risk is selection drift.',
    'The hybrid allocator page (64 × 584 = 37,376 B) is one and the same for every group, and according to the design document it only cheaply accommodates rows that divide it into a power of 2 (292, 146 B): that is why the 416 B layout was abandoned and option A (292 B) promises +65% physical tokens.',
  ],
  quiz: [
    { q: 'In the Qwen recipe, FP8 KV gives 3,652,200 tokens versus 2,131,159 in BF16. Why 1.70× and not 2×?', o: ['Because FP8 stores 4-byte scales per token', 'Because only the main KV drops by half; the QSA indexer/compressor in BF16, the GDN state and page rounding do not shrink', 'Because vLLM reserves half the pool for the drafter', 'Because FP8 E4M3 takes 12 bits in practice'], a: 1, e: 'With the measured tokens per GiB, BF16 ≈ 15.97 KB and FP8 ≈ 9.41 KB per token per GPU; subtracting the main KV (13,312 vs 6,656 B), ~2.7 KB remain in both cases. (13.3 + 2.7)/(6.7 + 2.7) ≈ 1.70. The per-tensor scales take up practically nothing.' },
    { q: 'Why can the QSA patch apply <code>k_scale</code> after the tensor core product?', o: ['Because the per-tensor scale is constant across all terms of the sum: \\(q\\cdot(s\\,\\tilde k) = s\\,(q\\cdot\\tilde k)\\)', 'Because FP8 has no scale', 'Because the softmax ignores multiplicative factors', 'Because Triton reorders the sums automatically'], a: 0, e: 'With a single scale per layer, it comes out of the sum and out of the weighted sum of V. The softmax does NOT ignore factors (they change the temperature), so the scale has to be applied, just once per score instead of once per element. With per-block scales inside the vector, the trick does not apply directly.' },
    { q: 'What does DeepSeek hotfix #22 fix?', o: ['A 416 B layout that corrupted the cache', 'A string comparison that sent <code>nvfp4_ds_mla</code> (bytes identical to FP8) to the slow <code>_forward_bf16_kv</code> path', 'The lack of UE8M0 scales in the KV', 'An integer overflow at 1M of context'], a: 1, e: '<code>use_fp8_cache = kv_cache_dtype == "fp8_ds_mla"</code> returned False for the nvfp4 name. The 584 B are the same; only the dispatch differed. At 600K+ decode went from ~1 to ~17 tok/s. The int64 overflows are real, but they are a different fix (in <code>cache_utils.py</code>).' },
    { q: 'Why does a 416 B compressed KV row not work in the DeepSeek-V4 hybrid allocator?', o: ['Because 416 is not a multiple of 8', 'Because every group needs the same page size in bytes (37,376 B, fixed by the 584 B SWA rows) and 37,376/416 is not a whole number of rows (let alone a power of 2)', 'Because FlashInfer only accepts 584 B rows', 'Because 416 B loses the RoPE'], a: 1, e: 'The SWA rows are written by a C++ op at 584 B, so the common page is 64 × 584. Only 292 (128 rows) and 146 (256 rows) divide that page into a power of 2; that is what option A exploits.' },
    { q: 'On DeepSeek, with the pinned image, what does turning on <code>use_fp4_indexer_cache</code> (MXFP4 in the indexer K cache) do?', o: ['Doubles the tokens in the pool', 'Halves the bytes <i>read</i> per scored key; the allocation stays at 132 B/row, and the risk is top-k selection drift at deep context', 'Quantizes the attention values in FP4', 'Nothing, because the GB10 has no FP4 kernels'], a: 1, e: 'The writer uses the first half of the 132 B row. The capacity gain (−0.35 KB/token ≈ −9%) would require allocating half a row. MXFP4 only affects the scores (who gets selected), never the values; that is why the gate is ruler-lite, garble up to 900K and TTFT at 128K.' },
    { q: 'GLM-5.3-Flash is NoPE, but it writes 128 B of RoPE per token in every MLA layer. Does that change the attention result?', o: ['Yes, the zeros dilute the softmax', 'No: [q; 0]·[k; 0] = q·k, the softmax and V (NoPE region) do not change; the only cost is space (656 vs ~528 B, +24%)', 'Yes, because the RoPE is applied afterwards', 'No, because the kernel ignores the RoPE in FP8'], a: 1, e: 'The only sparse-MLA kernel on SM12x requires the 576-dim geometry (<code>pe_dim</code> 64). Padding with zeros is exact and avoids writing a new kernel: it trades bytes for engineering work.' },
  ],
  mount(root, LA) {
    const V = LA.viz, UI = LA.ui, N = LA.num;
    const GiB = 1073741824;
    const nf = (x, d = 0) => (+x).toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d });
    const fmtB = (b) => {
      const a = Math.abs(b);
      if (a >= 1e9) return nf(b / 1e9, a >= 1e11 ? 0 : a >= 1e10 ? 1 : 2) + ' GB';
      if (a >= 1e6) return nf(b / 1e6, a >= 1e8 ? 0 : a >= 1e7 ? 1 : 2) + ' MB';
      if (a >= 1e3) return nf(b / 1e3, a >= 1e5 ? 0 : a >= 1e4 ? 1 : 2) + ' KB';
      return nf(b) + ' B';
    };
    const fmtT = (s) => (s >= 1 ? nf(s, 2) + ' s' : s >= 1e-3 ? nf(s * 1e3, s >= 0.1 ? 0 : 1) + ' ms' : nf(s * 1e6, 0) + ' µs');
    /* ajusta a altura do palco em telas estreitas (evita sobreposição em 390 px) */
    const setH = (fig, px) => {
      const st = fig.querySelector('.la-fig-stage');
      const want = Math.ceil(px) + 'px';
      if (st && st.style.height !== want) st.style.height = want;
    };
    const autoH = (fig, w, narrow, wide, bp = 600) => setH(fig, w < bp ? narrow : wide);
    /* texto com quebra de linha por palavra; devolve o y da próxima linha */
    const wrapText = (g, s, x, y, maxW, lh, o = {}) => {
      g.font = V.font(o.size || 12, o.font || 'sans', o.weight || 400);
      let line = '', yy = y;
      String(s).split(' ').forEach((wd) => {
        const test = line ? line + ' ' + wd : wd;
        if (line && g.measureText(test).width > maxW) { V.text(g, line, x, yy, o); line = wd; yy += lh; g.font = V.font(o.size || 12, o.font || 'sans', o.weight || 400); }
        else line = test;
      });
      if (line) V.text(g, line, x, yy, o);
      return yy + lh;
    };
    const expo = (v, d = 2) => v.toExponential(d).replace('e-', 'e−').replace('e+', 'e');
    const subD = (n) => String(n).split('').map((c) => '₀₁₂₃₄₅₆₇₈₉'[+c]).join('');
    const hatch = (g, x, y, w, h, col, step = 5) => {
      g.save(); g.beginPath(); g.rect(x, y, w, h); g.clip();
      g.strokeStyle = col; g.lineWidth = 1;
      for (let k = -h; k < w + h; k += step) { g.beginPath(); g.moveTo(x + k, y + h); g.lineTo(x + k + h, y); g.stroke(); }
      g.restore();
    };

    /* controles: .la-ctl é inline-flex sem wrap no framework; em 390 px, rótulo + seg estouram.
       Correção local: deixar os controles destas figuras quebrarem linha. */
    const wrapControls = () => {
      root.querySelectorAll('.la-fig .la-fig-controls .la-ctl').forEach((el) => { el.style.flexWrap = 'wrap'; el.style.maxWidth = '100%'; el.style.minWidth = '0'; });
      root.querySelectorAll('.la-fig .la-fig-controls .la-seg').forEach((el) => { el.style.maxWidth = '100%'; });
      root.querySelectorAll('.la-fig .la-fig-controls select').forEach((el) => { el.style.maxWidth = '100%'; el.style.minWidth = '0'; });
      root.querySelectorAll('.la-fig .la-fig-controls .la-toggle').forEach((el) => { el.style.maxWidth = '100%'; });
    };

    /* ================================================================
       FIGURA 1 — Guardar e ler (armazém + esteira)
       ================================================================ */
    (() => {
      const fig = root.querySelector('#fig-dois-custos');
      const ctl = UI.controls(fig);
      const S = { fmt: 'bf16', ci: 3 };
      const CTX = [8192, 32768, 131072, 262144, 524288, 1048576];
      const CTXL = ['8K', '32K', '128K', '256K', '512K', '1M'];
      const ELEMS = 43 * 2 * 8 * 128; // números por token no modelo ilustrativo
      const BPE = { bf16: 2, fp8: 1, fp4: 0.5625 };
      const POOL = 17 * GiB, BW = 240e9;
      UI.seg(ctl, { label: 'KV format', options: [{ v: 'bf16', label: 'BF16' }, { v: 'fp8', label: 'FP8' }, { v: 'fp4', label: 'NVFP4' }], value: 'bf16', onChange: (v) => { S.fmt = v; upd(); } });
      UI.slider(ctl, { label: 'Context', min: 0, max: 5, step: 1, value: 3, fmt: (v) => CTXL[v], onInput: (v) => { S.ci = v; upd(); } });
      const r1 = UI.readout(ctl, 'KV/token', '—'), r2 = UI.readout(ctl, '1 sequence', '—'), r3 = UI.readout(ctl, 'read/token', '—');
      function calc() {
        const bpt = ELEMS * BPE[S.fmt];
        const n = CTX[S.ci];
        const seq = bpt * n;
        return { bpt, n, seq, t: seq / BW, fit: POOL / bpt };
      }
      function upd() { const c = calc(); r1.set(fmtB(c.bpt)); r2.set(fmtB(c.seq)); r3.set(fmtT(c.t)); }
      upd();
      V.fig(fig, (g, w, h, t) => {
        autoH(fig, w, 370, 400);
        const C = V.C();
        const c = calc();
        const narrow = w < 600;
        const pad = 16, bw = w - pad * 2;
        const boxW = { bf16: 14, fp8: 7, fp4: 4 }[S.fmt];
        // ---- GUARDAR
        V.text(g, narrow ? 'STORE · 17 GiB pool' : 'STORE · 17 GiB KV pool (1 shelf = 1 GiB)', pad, 22, { size: 11.5, font: 'mono', color: C.muted, maxW: bw });
        const py = 32, ph = 62;
        V.box(g, pad, py, bw, ph, { r: 8, fill: V.alpha(C.muted, 0.08), stroke: V.alpha(C.muted, 0.35), lw: 1 });
        const frac = c.seq / POOL;
        const fw = Math.min(1, frac) * bw;
        const over = frac > 1;
        g.save(); V.rr(g, pad, py, bw, ph, 8); g.clip();
        const rows = 4, rh = (ph - 10) / rows;
        for (let x = pad + 3; x < pad + fw - 1; x += boxW + 2) {
          for (let r = 0; r < rows; r++) {
            const isK = ((Math.floor((x - pad) / (boxW + 2)) + r) % 2) === 0;
            const col = over ? C.red : isK ? C.a : C.b;
            g.fillStyle = V.alpha(col, over ? 0.45 : 0.62);
            g.fillRect(x, py + 5 + r * rh, Math.min(boxW, pad + fw - x), rh - 2);
            if (S.fmt === 'fp4') { g.fillStyle = V.alpha(C.d, 0.9); g.fillRect(x, py + 5 + r * rh, 1, rh - 2); }
          }
        }
        g.restore();
        for (let k = 1; k < 17; k++) V.line(g, pad + (bw * k) / 17, py + 2, pad + (bw * k) / 17, py + ph - 2, { color: V.alpha(C.ink, 0.12), lw: 1 });
        const lineA = over
          ? `✕ 1 sequence of ${CTXL[S.ci]} needs ${nf(c.seq / GiB, 1)} GiB: ${nf(frac, 1)}× the pool`
          : `✓ 1 sequence of ${CTXL[S.ci]} takes ${nf(frac * 100, frac < 0.1 ? 1 : 0)}% of the pool · ~${nf(Math.floor(1 / frac))} fit`;
        V.text(g, lineA, pad, py + ph + 20, { size: 13, weight: 600, color: over ? C.red : C.a, maxW: bw });
        V.text(g, narrow ? `${nf(c.fit)} tokens fit (${fmtB(c.bpt)}/token)` : `the pool fits ${nf(c.fit)} tokens of this model (${fmtB(c.bpt)} per token)`, pad, py + ph + 39, { size: 12, color: C.muted, maxW: bw });
        // ---- LER
        const ly = py + ph + 72;
        V.text(g, narrow ? 'READ · on every decode token' : 'READ · on every decode token, the kernel fetches the whole KV of the sequence', pad, ly, { size: 11.5, font: 'mono', color: C.muted, maxW: bw });
        const memW = Math.min(110, w * 0.22), smW = Math.min(90, w * 0.2);
        const by = ly + 18, bh = 64;
        V.box(g, pad, by, memW, bh, { r: 10, fill: V.alpha(C.b, 0.14), stroke: C.b, lw: 1.5 });
        V.text(g, 'memory', pad + memW / 2, by + 27, { size: 12.5, align: 'center', weight: 600, color: C.ink });
        V.text(g, 'LPDDR5X', pad + memW / 2, by + 44, { size: 11, align: 'center', font: 'mono', color: C.muted });
        const sx = w - pad - smW;
        V.box(g, sx, by, smW, bh, { r: 10, fill: V.alpha(C.a, 0.14), stroke: C.a, lw: 1.5 });
        V.text(g, 'SMs', sx + smW / 2, by + 27, { size: 12.5, align: 'center', weight: 600, color: C.ink });
        V.text(g, 'attention', sx + smW / 2, by + 44, { size: 11, align: 'center', font: 'mono', color: C.muted });
        const bx0 = pad + memW + 8, bx1 = sx - 8, beltY = by + bh - 12;
        V.line(g, bx0, beltY + 4, bx1, beltY + 4, { color: V.alpha(C.ink, 0.35), lw: 3 });
        V.line(g, bx0, beltY + 4, bx1, beltY + 4, { color: V.alpha(C.ink, 0.6), lw: 1.2, dash: [3, 7], dashOffset: -t * 60 });
        V.text(g, '240 GB/s', (bx0 + bx1) / 2, beltY + 22, { size: 11, align: 'center', font: 'mono', color: C.muted });
        const speed = 60, gap = 3, step = boxW * 1.6 + gap, L = bx1 - bx0;
        /* período múltiplo (par) do passo: na emenda do loop as caixas não se sobrepõem e K/V seguem alternando */
        const nBox = 2 * Math.ceil((L + step) / (2 * step)), P = nBox * step;
        g.save(); g.beginPath(); g.rect(bx0, by - 4, L, bh + 4); g.clip();
        for (let k = 0; k < nBox; k++) {
          const xx = bx0 + ((k * step + t * speed) % P) - step / 2;
          const isK = k % 2 === 0;
          g.fillStyle = V.alpha(isK ? C.a : C.b, 0.8);
          V.rr(g, xx, beltY - 22, boxW * 1.6, 22, 3); g.fill();
          if (boxW >= 7) V.text(g, isK ? 'K' : 'V', xx + boxW * 0.8, beltY - 7, { size: 11, align: 'center', font: 'mono', weight: 700, color: C.bg });
        }
        g.restore();
        // progresso de um token
        const tmin = 8192 * ELEMS * 0.5625 / BW, tmax = 1048576 * ELEMS * 2 / BW;
        const period = 0.35 + 2.8 * (Math.log(c.t / tmin) / Math.log(tmax / tmin));
        const ph2 = (t % period) / period;
        const qy = by + bh + 26;
        V.box(g, pad, qy, bw, 12, { r: 6, fill: V.alpha(C.muted, 0.14) });
        V.box(g, pad, qy, Math.max(6, bw * ph2), 12, { r: 6, fill: over ? V.alpha(C.red, 0.75) : V.alpha(C.a, 0.8) });
        if (ph2 > 0.9) V.glow(g, pad + bw - 8, qy + 6, 18, C.d);
        V.text(g, `≈ ${fmtT(c.t)} per token just reading the KV`, pad, qy + 44, { size: 20, font: 'display', weight: 700, color: C.ink, maxW: bw });
        const teto = `ceiling set by this read: ${c.t > 1 ? nf(1 / c.t, 2) : nf(1 / c.t, 0)} tok/s`;
        V.text(g, narrow ? teto : teto + ' · the bar above fills once per token (log scale)', pad, qy + 66, { size: 12, color: C.muted, maxW: bw });
      });
    })();

    /* ================================================================
       FIGURA 2 — A ida e a volta de um vetor K
       ================================================================ */
    (() => {
      const fig = root.querySelector('#fig-ida-volta');
      const ctl = UI.controls(fig);
      const S = { fmt: 'e4m3', mode: 't1', outlier: true };
      const BASE = [0.62, -0.35, 0.004, 1.35, -0.81, 0.09, -0.009, 0.47, -0.28, 0.73, -1.85, 0.018, 0.55, -0.06, 0.31, -0.93];
      const xs = () => BASE.map((v, i) => (i === 3 && S.outlier ? 26 : N.round('bf16', v)));
      function quant() {
        const x = xs();
        const f = S.fmt, maxF = N.maxFinite(f), emax = Math.floor(Math.log2(maxF));
        const blocks = S.mode.startsWith('b') ? [[0, 8], [8, 16]] : [[0, 16]];
        const o = { x, scaled: [], codes: [], deq: [], rel: [], scales: [], clamp: [] };
        const amaxT = Math.max(...x.map(Math.abs));
        const gs = amaxT / (maxF * 448); // NVFP4: escala global FP32 do tensor (o bloco do amax recebe o código E4M3 448)
        for (const [a, b] of blocks) {
          const amax = Math.max(...x.slice(a, b).map(Math.abs));
          let s, code = null;
          if (S.mode === 't1') s = 1;
          else if (S.mode === 'tmax') s = amax / maxF;
          else if (S.mode === 'bceil') s = Math.pow(2, Math.ceil(Math.log2(Math.max(amax, 1e-4) / maxF)));
          else if (S.mode === 'bfloor') s = Math.pow(2, Math.floor(Math.log2(amax)) - emax);
          else { code = N.encode('e4m3', amax / maxF / gs).v; s = code * gs; }
          let sat = false;
          for (let i = a; i < b; i++) {
            const y = x[i] / s, e = N.encode(f, y);
            o.scaled[i] = y; o.codes[i] = e; o.deq[i] = e.v * s;
            o.rel[i] = Math.abs(e.v * s - x[i]) / Math.max(Math.abs(x[i]), 1e-12);
            // tolerância: amax/(amax/448) pode dar 448,00000000000006 em ponto flutuante
            o.clamp[i] = Math.abs(y) > maxF * (1 + 1e-9);
            if (o.clamp[i]) sat = true;
          }
          o.scales.push({ a, b, s, amax, code, sat });
        }
        o.gs = gs;
        o.stats = N.errStats(x, o.deq);
        o.zeros = x.reduce((n, v, i) => n + (o.codes[i].v === 0 && v !== 0 ? 1 : 0), 0);
        const sr = o.rel.slice().sort((p, q) => p - q);
        o.medRel = (sr[7] + sr[8]) / 2;
        o.sat = o.clamp.map((c, i) => (c ? i : -1)).filter((i) => i >= 0);
        const vb = f === 'e4m3' ? 16 : 8;
        const sb = S.mode === 't1' ? 0 : S.mode === 'tmax' ? 4 : S.mode === 'bnv' ? 6 : 2;
        o.bytes = { vb, sb };
        return o;
      }
      const pct = (v) => (v >= 1 ? '100%' : nf(v * 100, v < 0.1 ? 1 : 0) + '%');
      UI.seg(ctl, { label: 'Format', options: [{ v: 'e4m3', label: 'FP8 E4M3' }, { v: 'e2m1', label: 'FP4 E2M1' }], value: 'e4m3', onChange: (v) => { S.fmt = v; upd(); } });
      UI.seg(ctl, { label: 'Scale', options: [{ v: 't1', label: 'tensor · s = 1' }, { v: 'tmax', label: 'tensor · amax' }, { v: 'bceil', label: 'block · UE8M0 ⌈⌉' }, { v: 'bfloor', label: 'block · UE8M0 ⌊⌋ (MX)' }, { v: 'bnv', label: 'block · E4M3 + FP32 (NVFP4)' }], value: 't1', onChange: (v) => { S.mode = v; upd(); } });
      UI.toggle(ctl, { label: 'outlier (x₃ = 26)', value: true, onChange: (v) => { S.outlier = v; upd(); } });
      const rz = UI.readout(ctl, 'zeroed values', '—'), rm = UI.readout(ctl, 'median rel. error', '—'), rq = UI.readout(ctl, 'SQNR (largest dominates)', '—'), rb = UI.readout(ctl, 'bytes stored', '—');
      let Q = quant();
      let H = null;
      function upd() {
        Q = quant();
        rz.set(`${Q.zeros}/16`); rm.set(pct(Q.medRel)); rq.set(nf(Q.stats.sqnr, 1) + ' dB');
        rb.set(`${Q.bytes.vb} B + ${Q.bytes.sb} B of scale`);
        if (H) H.redraw();
      }
      const steps = [
        { t: 'The K vector in BF16', d: 'Sixteen numbers from a freshly projected row of K, in two blocks of 8 (in DeepSeek-V4 there would be 7 blocks of 64). There are large numbers, small ones and, if the toggle is on, an outlier.' },
        { t: 'Choose the scale and divide', d: 'Per tensor, one scale serves everything: 1.0 (no calibration) or amax/format maximum. Per block, each block gets its own power of 2 (UE8M0), with the exponent rounded up (DeepSeek) or down (OCP MX). In NVFP4, each block scale is an E4M3 that multiplies an FP32 tensor scale, g.' },
        { t: 'Round and store the code', d: 'Each \\(x/s\\) becomes the nearest value of the format (RNE). Above the maximum (448 in E4M3, 6 in E2M1) it saturates. The MX floor causes this whenever the mantissa of the block amax exceeds 1.75 (E4M3) or 1.5 (E2M1), outlier or not; the header marks the block with ⚠. What goes to memory is 1 byte (FP8) or half a byte (FP4) per number, plus the scales.' },
        { t: 'Re-read and dequantize', d: 'At attention time, the kernel computes \\(\\hat x = s \\cdot c\\): it multiplies the code by the scale of its group. With UE8M0, multiplying by \\(s\\) is just adding to the exponent.' },
        { t: 'Measure the error', d: 'The relative error of each cell shows where quantization hurts: tiny numbers (subnormals), the neighbors of the outlier and, in FP4, almost everything that did not get a suitable scale. Count the zeros and look at the median error: SQNR weights each error by its energy and is dominated by the largest number.' },
      ];
      const st = UI.stepper(fig, steps, { autoplay: 6500, onStep: () => H && H.redraw() });
      const fmtCell = (v, wide) => {
        if (!Number.isFinite(v)) return '—';
        const a = Math.abs(v);
        if (a === 0) return '0';
        if (a >= 100) return nf(v, 0);
        if (a >= 10) return nf(v, wide ? 1 : 0);
        if (a >= 1) return nf(v, 2);
        if (a >= 0.1) return nf(v, 2);
        if (a >= 0.01) return nf(v, 3);
        return v.toExponential(0).replace('e-', 'e−');
      };
      H = V.fig(fig, (g, w, h, t) => {
        setH(fig, w >= 640 ? 318 : 500);
        const C = V.C();
        const wide = w >= 640;
        const pad = 14;
        const labW = wide ? 124 : 34;
        const perLine = wide ? 16 : 8;
        const bgap = wide ? 10 : 0;
        const cw = Math.floor((w - pad * 2 - labW - bgap) / perLine) - 3;
        const ch = 28;
        const rowsDef = [
          { k: 'x', lab: 'x · BF16', sh: 'x' },
          { k: 'scaled', lab: 'x ÷ s', sh: 'x/s' },
          { k: 'code', lab: 'stored code', sh: 'c' },
          { k: 'deq', lab: 'x̂ = s · c', sh: 'x̂' },
          { k: 'rel', lab: 'relative error', sh: 'err' },
        ];
        // cabeçalho: escalas
        const sc = Q.scales.map((b, i) => {
          const nm = Q.scales.length > 1 ? 's' + subD(i) : 's';
          let v;
          if (S.mode === 'bnv') v = `${nf(b.code, Number.isInteger(b.code) ? 0 : 3)}·g`;
          else if (S.mode.startsWith('b')) { const X = Math.round(Math.log2(b.s)); v = `2^${X} (byte ${X + 127})`; }
          else v = (b.s >= 0.01 && b.s < 1000 ? nf(b.s, b.s < 1 ? 4 : 2) : expo(b.s)) + (S.mode === 't1' ? ' (no calibration)' : ' (FP32)');
          return `${nm} = ${v}${b.sat ? ' ⚠ saturates' : ''}`;
        }).join('  ·  ');
        const scTxt = (S.mode === 'bnv' ? `g = ${expo(Q.gs)} (FP32)  ·  ` : '') + sc;
        const fmtTxt = S.fmt === 'e4m3' ? 'FP8 E4M3 · max 448' : 'FP4 E2M1 · {0, 0.5, 1, 1.5, 2, 3, 4, 6}';
        const anySat = Q.sat.length > 0;
        if (wide) V.text(g, fmtTxt + '   |   ' + scTxt, pad, 20, { size: 11.5, font: 'mono', color: anySat ? C.ink : C.muted, maxW: w - pad * 2 });
        else {
          V.text(g, fmtTxt, pad, 17, { size: 11.5, font: 'mono', color: C.muted, maxW: w - pad * 2 });
          V.text(g, scTxt, pad, 35, { size: 11.5, font: 'mono', color: anySat ? C.ink : C.muted, maxW: w - pad * 2 });
        }
        const lineH = wide ? ch + 14 : (ch + 4) * 2 + 12;
        const y0 = wide ? 38 : 50;
        const el = st.elapsed();
        rowsDef.forEach((R, ri) => {
          const y = y0 + ri * lineH;
          const active = st.i === ri, shown = ri <= st.i;
          V.text(g, wide ? R.lab : R.sh, pad, y + ch / 2 + 4, { size: wide ? 12 : 11.5, weight: active ? 700 : 500, color: active ? C.ink : C.muted, font: wide ? 'sans' : 'mono' });
          const reveal = active ? Math.floor(el / 0.07) : 99;
          for (let i = 0; i < 16; i++) {
            const line = wide ? 0 : Math.floor(i / 8);
            const col = wide ? i : i % 8;
            const x = pad + labW + col * (cw + 3) + (wide && i >= 8 ? bgap : 0);
            const yy = y + line * (ch + 4);
            const vis = shown && i < reveal;
            if (!vis) { V.box(g, x, yy, cw, ch, { r: 5, fill: V.alpha(C.muted, 0.06), stroke: V.alpha(C.muted, 0.18), lw: 1 }); continue; }
            let val, fill, txtCol = C.ink;
            const amax = Math.max(...Q.x.map(Math.abs));
            if (R.k === 'x') { val = Q.x[i]; fill = V.cmap(val / amax, 'div'); }
            else if (R.k === 'scaled') { val = Q.scaled[i]; fill = Q.clamp[i] ? V.alpha(C.red, 0.55) : V.alpha(C.b, 0.22); }
            else if (R.k === 'code') { val = Q.codes[i].v; fill = val === 0 && Q.x[i] !== 0 ? V.alpha(C.red, 0.35) : V.alpha(S.fmt === 'e4m3' ? C.a : C.c, 0.28); }
            else if (R.k === 'deq') { val = Q.deq[i]; fill = V.cmap(val / amax, 'div'); }
            else { val = Q.rel[i]; fill = V.cmap(Math.min(1, val / 0.25), 'heat'); }
            V.box(g, x, yy, cw, ch, { r: 5, fill, stroke: active ? V.alpha(C.ink, 0.35) : null, lw: 1 });
            const s = R.k === 'rel' ? (val >= 1 ? '100%' : nf(val * 100, val < 0.1 ? 1 : 0) + '%') : fmtCell(val, wide);
            if (R.k === 'rel' && val > 0.12 && C.dark) txtCol = C.bg;
            V.text(g, s, x + cw / 2, yy + ch / 2 + 4, { size: cw < 40 ? 11 : 11.5, font: 'mono', align: 'center', color: txtCol, maxW: cw - 2 });
          }
        });
        // rodapé: as métricas que não se deixam dominar pelo maior número vêm primeiro
        const fy = y0 + rowsDef.length * lineH + 6;
        const satTxt = anySat ? 'saturated: ' + Q.sat.map((i) => `x${subD(i)} = ${fmtCell(Q.x[i], true)} → ${fmtCell(Q.deq[i], true)}`).join(', ') : 'nothing saturated';
        const mTxt = `zeroed: ${Q.zeros}/16  ·  median rel. error: ${pct(Q.medRel)}`;
        const qTxt = `SQNR ${nf(Q.stats.sqnr, 1)} dB: weights each error by energy, so the largest number dominates`;
        if (wide) {
          V.text(g, `${mTxt}  ·  ${satTxt}`, pad, fy + 10, { size: 12, weight: 600, color: anySat ? C.red : C.ink, maxW: w - pad * 2 });
          V.text(g, qTxt, pad, fy + 28, { size: 12, color: C.muted, maxW: w - pad * 2 });
        } else {
          V.text(g, mTxt, pad, fy + 8, { size: 12, weight: 600, color: C.ink, maxW: w - pad * 2 });
          V.text(g, satTxt, pad, fy + 26, { size: 11.5, font: 'mono', color: anySat ? C.red : C.muted, maxW: w - pad * 2 });
          V.text(g, `SQNR ${nf(Q.stats.sqnr, 1)} dB (dominated by the largest number)`, pad, fy + 44, { size: 11.5, color: C.muted, maxW: w - pad * 2 });
        }
      }, { static: false });
      upd();
    })();

    /* ================================================================
       FIGURA 3 — Escala antes ou depois do produto
       ================================================================ */
    (() => {
      const fig = root.querySelector('#fig-escala-dot');
      const ctl = UI.controls(fig);
      const S = { mode: 'after' };
      const q = [0.8, -0.4, 1.1, 0.3, -0.9, 0.5, -0.2, 0.7].map((v) => N.round('bf16', v));
      const r = V.rng(41);
      const Kx = Array.from({ length: 4 }, (_, j) => Array.from({ length: 8 }, (_, d) => N.round('bf16', r.normal() * (d === 2 ? 2.4 : 0.9) + (j === 1 ? 0.5 : 0))));
      let amax = 0; Kx.forEach((row) => row.forEach((v) => { amax = Math.max(amax, Math.abs(v)); }));
      const sk = amax / 448;
      const Kc = Kx.map((row) => row.map((v) => N.encode('e4m3', v / sk).v));
      // modo bloco: 2 blocos de 4 dims por token, UE8M0 com teto
      const Kb = Kx.map((row) => [0, 4].map((a) => {
        const seg = row.slice(a, a + 4); const am = Math.max(...seg.map(Math.abs));
        const s = Math.pow(2, Math.ceil(Math.log2(Math.max(am, 1e-4) / 448)));
        return { s, c: seg.map((v) => N.encode('e4m3', v / s).v) };
      }));
      const dot = (a, b) => a.reduce((acc, v, i) => acc + v * b[i], 0);
      const scA = Kc.map((row) => dot(q, row.map((c) => c * sk)));
      const raw = Kc.map((row) => dot(q, row));
      const scB = raw.map((v) => v * sk);
      const part = Kb.map((blk) => blk.map((b, bi) => dot(q.slice(bi * 4, bi * 4 + 4), b.c)));
      const scC = part.map((p, j) => p[0] * Kb[j][0].s + p[1] * Kb[j][1].s);
      const exact = Kx.map((row) => dot(q, row));
      const maxDiff = Math.max(...scA.map((v, i) => Math.abs(v - scB[i])));
      const mae = (arr) => arr.reduce((a, v, i) => a + Math.abs(v - exact[i]), 0) / arr.length;
      const errT = mae(scB), errBk = mae(scC);
      const accTxt = `In exchange, the scores get closer to the reference: mean error ${nf(errBk, 3)} versus ${nf(errT, 3)} per tensor.`;
      const vTxt = 'The same holds on the V side: Σ pⱼ (s_v ṽⱼ) = s_v · Σ pⱼ ṽⱼ → normalized_output *= v_scale at the end.';
      UI.seg(ctl, { label: 'How the kernel does the math', options: [{ v: 'deq', label: 'dequantize the tile' }, { v: 'after', label: 'scale after the dot' }, { v: 'block', label: 'per-block scale' }], value: 'after', onChange: (v) => { S.mode = v; } });
      const rr = UI.readout(ctl, 'tile in the dot', '—');
      const ru = UI.readout(ctl, 'A vs B', maxDiff < 1e-12 ? 'identical' : '|Δ| ≈ ' + maxDiff.toExponential(1));
      V.fig(fig, (g, w, h, t) => {
        if (w >= 600) setH(fig, 340);
        const C = V.C();
        const wide = w >= 600;
        const pad = 14;
        const cell = wide ? Math.min(36, Math.floor((w - pad * 2 - 170) / 16) - 2) : Math.min(36, Math.floor((w - pad * 2 - 20) / 8));
        const cellH = 24;
        const fmtc = (v) => (Math.abs(v) >= 100 ? nf(v, 0) : Math.abs(v) >= 10 ? nf(v, 0) : nf(v, 1));
        const drawTile = (x, y, data, colFn, label, sub, blockSplit) => {
          V.text(g, label, x, y - 20, { size: 12, weight: 600, color: C.ink });
          if (sub) V.text(g, sub, x, y - 6, { size: 11, font: 'mono', color: C.muted });
          data.forEach((row, i) => row.forEach((v, j) => {
            const xx = x + j * (cell + 2) + (blockSplit && j >= 4 ? 6 : 0), yy = y + i * (cellH + 2);
            V.box(g, xx, yy, cell, cellH, { r: 4, fill: colFn(v, i, j) });
            if (cell >= 26) V.text(g, fmtc(v), xx + cell / 2, yy + cellH / 2 + 4, { size: 11, font: 'mono', align: 'center', color: C.ink, maxW: cell - 2 });
          }));
          return { w: 8 * (cell + 2) + (blockSplit ? 6 : 0), h: 4 * (cellH + 2) };
        };
        const col = (v, i, j) => V.alpha(C.a, 0.12 + 0.5 * Math.min(1, Math.abs(v) / 448));
        const colF = (v) => V.alpha(C.red, 0.15 + 0.5 * Math.min(1, Math.abs(v) / amax));
        const drawVec = (x, y, vals, label, colr, fmt, vertical) => {
          V.text(g, label, x, y - 8, { size: 11.5, font: 'mono', color: C.muted });
          vals.forEach((v, i) => {
            const xx = vertical ? x : x + i * (cell + 2), yy = vertical ? y + i * (cellH + 2) : y;
            const ww = vertical ? Math.max(58, cell + 18) : cell;
            V.box(g, xx, yy, ww, cellH, { r: 4, fill: V.alpha(colr, 0.22), stroke: V.alpha(colr, 0.6), lw: 1 });
            if (ww >= 26) V.text(g, fmt(v), xx + ww / 2, yy + cellH / 2 + 4, { size: 11, font: 'mono', align: 'center', color: C.ink, maxW: ww - 2 });
          });
        };
        const fl = (x1, y1, x2, y2, colr, off = 0) => {
          V.arrow(g, x1, y1, x2, y2, { color: V.alpha(colr, 0.7), lw: 1.8 });
          const u = ((t * 0.8 + off) % 1);
          const p = V.along(x1, y1, x2, y2, u);
          V.dot(g, p[0], p[1], 3.2, colr);
        };
        const f3 = (v) => nf(v, 3);
        let tileBytes;
        if (wide) {
          const x0 = pad + 4, y0 = 128, qy = 62;
          const tw = 8 * (cell + 2);
          const head = S.mode === 'deq' ? 'A · dequantize: build K = s·C in FP32 and only then compute q·k'
            : S.mode === 'after' ? 'B · scale after: q·c directly on the codes C (exact upcast), then × s_k'
            : 'C · per-block scale: each 4-dim block has its own scale per token';
          V.text(g, head, pad, 22, { size: 12.5, weight: 700, color: C.ink, maxW: w - pad * 2 });
          const mid = y0 + 2 * (cellH + 2) - 1;
          if (S.mode === 'deq') {
            drawTile(x0, y0, Kc, col, 'C · FP8 codes', `s_k = ${expo(sk)}`);
            const x1 = x0 + tw + 50;
            fl(x0 + tw + 6, mid, x1 - 6, mid, C.red);
            V.text(g, '× s_k', x0 + tw + 25, mid - 10, { size: 11, align: 'center', font: 'mono', color: C.red });
            drawVec(x1, qy, q, 'q (BF16)', C.b, (v) => nf(v, 1));
            drawTile(x1, y0, Kc.map((row) => row.map((c) => c * sk)), colF, 'K = s·C in FP32', 'materialized tile: 4 B/number');
            const x2 = x1 + tw + 44;
            fl(x1 + tw + 6, mid, x2 - 6, mid, C.a, 0.3);
            V.text(g, 'q·kⱼ', x1 + tw + 22, mid - 10, { size: 11, align: 'center', font: 'mono', color: C.a });
            drawVec(x2, y0, scA, 'scores', C.a, f3, true);
            tileBytes = '32 × 4 B = 128 B';
          } else if (S.mode === 'after') {
            drawVec(x0, qy, q, 'q (BF16)', C.b, (v) => nf(v, 1));
            drawTile(x0, y0, Kc, col, 'C · FP8 codes seen as BF16', 'exact upcast: the same values');
            const x1 = x0 + tw + 50;
            fl(x0 + tw + 6, mid, x1 - 6, mid, C.a);
            V.text(g, 'q·cⱼ', x0 + tw + 25, mid - 10, { size: 11, align: 'center', font: 'mono', color: C.a });
            drawVec(x1, y0, raw, 'raw', C.b, (v) => nf(v, 0), true);
            const x2 = x1 + 58 + 56;
            fl(x1 + 64, mid, x2 - 6, mid, C.d, 0.4);
            V.text(g, '× s_k', x1 + 58 + 25, mid - 10, { size: 11, align: 'center', font: 'mono', color: C.d });
            drawVec(x2, y0, scB, 'scores', C.a, f3, true);
            V.glow(g, x1 + 58 + 25, mid, 12 + 4 * Math.sin(t * 4), C.d, 0.6);
            tileBytes = '32 × 2 B = 64 B';
          } else {
            drawVec(x0, qy, q, 'q (BF16) · blocks 0 | 1', C.b, (v) => nf(v, 1));
            drawTile(x0, y0, Kb.map((blk) => blk[0].c.concat(blk[1].c)), (v, i, j) => V.alpha(j < 4 ? C.a : C.f, 0.14 + 0.45 * Math.min(1, Math.abs(v) / 448)), 'C · 2 blocks of 4 dims', 'one UE8M0 scale per token per block', true);
            const x1 = x0 + tw + 60;
            fl(x0 + tw + 12, mid, x1 - 6, mid, C.a);
            V.text(g, 'partials', x0 + tw + 34, mid - 10, { size: 11, align: 'center', font: 'mono', color: C.a });
            drawVec(x1, y0, part.map((p) => p[0]), 'block 0', C.a, (v) => nf(v, 0), true);
            drawVec(x1 + 64, y0, part.map((p) => p[1]), 'block 1', C.f, (v) => nf(v, 0), true);
            const x2 = x1 + 64 + 58 + 56;
            fl(x1 + 128, mid, x2 - 6, mid, C.d, 0.4);
            V.text(g, 'Σ ×sⱼ,b', x1 + 128 + 22, mid - 10, { size: 11, align: 'center', font: 'mono', color: C.d });
            drawVec(x2, y0, scC, 'scores', C.a, f3, true);
            tileBytes = '32 × 2 B + 8 scales';
          }
          const ty = y0 + 4 * (cellH + 2) + 30;
          V.text(g, `exact reference (K in BF16): ${exact.map(f3).join(' · ')}`, pad, ty, { size: 12, font: 'mono', color: C.muted, maxW: w - pad * 2 });
          const msg = S.mode === 'deq' ? 'Correct, but the FP32 tile doubles the bytes in registers/shared memory and narrows the possible tile.'
            : S.mode === 'after' ? 'Same scores as mode A: s_k comes out of the sum. Tile at BF16 width, one multiplication per score.'
            : 'The scale changes inside the sum: each block needs its own partial × scale (or dequantizing first).';
          V.text(g, msg, pad, ty + 24, { size: 12.5, weight: 600, color: S.mode === 'deq' ? C.red : S.mode === 'after' ? C.a : C.d, maxW: w - pad * 2 });
          V.text(g, S.mode === 'block' ? accTxt : vTxt, pad, ty + 46, { size: 12, color: S.mode === 'block' ? C.ink : C.muted, maxW: w - pad * 2 });
        } else {
          // layout vertical
          const x0 = pad;
          drawVec(x0, 34, q, 'q (BF16)', C.b, (v) => nf(v, 1));
          let y = 94;
          if (S.mode === 'block') drawTile(x0, y, Kb.map((blk) => blk[0].c.concat(blk[1].c)), (v, i, j) => V.alpha(j < 4 ? C.a : C.f, 0.14 + 0.45 * Math.min(1, Math.abs(v) / 448)), 'C · 2 blocks of 4', 'scale per token per block', true);
          else drawTile(x0, y, Kc, col, 'C · FP8 codes', `s_k = ${expo(sk)}`);
          y += 4 * (cellH + 2) + 12;
          if (S.mode === 'deq') {
            fl(x0 + 60, y, x0 + 60, y + 30, C.red); V.text(g, '× s_k → FP32 (4 B/number)', x0 + 72, y + 20, { size: 11.5, font: 'mono', color: C.red });
            y += 68;
            drawTile(x0, y, Kc.map((row) => row.map((c) => c * sk)), colF, 'K = s·C (FP32)', '');
            y += 4 * (cellH + 2) + 12;
            fl(x0 + 60, y, x0 + 60, y + 30, C.a, 0.3); V.text(g, 'q · kⱼ', x0 + 72, y + 20, { size: 11.5, font: 'mono', color: C.a });
            y += 50; drawVec(x0, y, scA.slice(0, 4), 'scores', C.a, f3); tileBytes = '128 B (FP32)';
          } else if (S.mode === 'after') {
            fl(x0 + 60, y, x0 + 60, y + 30, C.a); V.text(g, 'exact BF16 upcast · q · cⱼ', x0 + 72, y + 20, { size: 11.5, font: 'mono', color: C.a });
            y += 50; drawVec(x0, y, raw, 'raw', C.b, (v) => nf(v, 0));
            y += cellH + 14;
            fl(x0 + 60, y, x0 + 60, y + 30, C.d, 0.4); V.text(g, '× s_k (once per score)', x0 + 72, y + 20, { size: 11.5, font: 'mono', color: C.d });
            y += 50; drawVec(x0, y, scB, 'scores', C.a, f3); tileBytes = '64 B (BF16)';
          } else {
            fl(x0 + 60, y, x0 + 60, y + 30, C.a); V.text(g, 'one partial per block', x0 + 72, y + 20, { size: 11.5, font: 'mono', color: C.a });
            y += 50; drawVec(x0, y, part.map((p) => p[0]), 'block 0', C.a, (v) => nf(v, 0));
            y += cellH + 26; drawVec(x0, y, part.map((p) => p[1]), 'block 1', C.f, (v) => nf(v, 0));
            y += cellH + 14;
            fl(x0 + 60, y, x0 + 60, y + 30, C.d, 0.4); V.text(g, 'Σ partial × s_j,b', x0 + 72, y + 20, { size: 11.5, font: 'mono', color: C.d });
            y += 50; drawVec(x0, y, scC, 'scores', C.a, f3); tileBytes = '64 B + 8 scales';
          }
          y += cellH + 24;
          // referência exata (é ela que prova a equivalência dos modos A e B)
          V.text(g, 'exact reference (K in BF16):', pad, y, { size: 11.5, font: 'mono', color: C.muted, maxW: w - pad * 2 });
          V.text(g, exact.map(f3).join(' · '), pad, y + 17, { size: 12, font: 'mono', color: C.ink, maxW: w - pad * 2 });
          y += 44;
          const msg = S.mode === 'deq' ? 'Correct, but with an FP32 tile (4 B per number).' : S.mode === 'after' ? 'Same result as mode A: s_k comes out of the sum.' : 'The scale changes inside the sum: one partial per block.';
          y = wrapText(g, msg, pad, y, w - pad * 2, 18, { size: 12.5, weight: 600, color: S.mode === 'deq' ? C.red : S.mode === 'after' ? C.a : C.d });
          y = wrapText(g, S.mode === 'block' ? accTxt : vTxt, pad, y + 2, w - pad * 2, 17, { size: 12, color: S.mode === 'block' ? C.ink : C.muted });
          setH(fig, y + 6);
        }
        rr.set(tileBytes);
      });
    })();

    /* ================================================================
       FIGURA 4 — A balança da página (KV × estado GDN)
       ================================================================ */
    (() => {
      const fig = root.querySelector('#fig-balanca');
      const ctl = UI.controls(fig);
      const S = { kv: 'fp8', ssm: 'fp32', t0: 0 };
      const PAGE = { fp32: { rec: 1572864, conv: 61440 }, bf16: { rec: 786432, conv: 61440 } };
      const BPT = { bf16: 1024, fp8: 512 };
      const BLOCK = { 'bf16|fp32': 1600, 'fp8|fp32': 3200, 'fp8|bf16': 1664, 'bf16|bf16': 832 };
      const TOK = {
        'bf16|fp32': { n: 2131159, src: 'measured Sep 5' },
        'fp8|fp32': { n: 3652200, src: 'measured Sep 5' },
        'bf16|bf16': { n: 2426193, src: 'rerun Sep 16' },
        'fp8|bf16': { n: 4378829, src: 'rerun Sep 16' },
      };
      let H;
      const reset = () => { S.t0 = H ? H.t : 0; };
      UI.seg(ctl, { label: 'KV', options: [{ v: 'bf16', label: 'BF16 (auto)' }, { v: 'fp8', label: 'FP8' }], value: 'fp8', onChange: (v) => { S.kv = v; reset(); upd(); } });
      UI.seg(ctl, { label: 'GDN state', options: [{ v: 'fp32', label: 'FP32 (checkpoint)' }, { v: 'bf16', label: 'BF16' }], value: 'fp32', onChange: (v) => { S.ssm = v; reset(); upd(); } });
      const rb = UI.readout(ctl, 'attention block', '—'), rt = UI.readout(ctl, 'tokens in pool', '—');
      function upd() {
        const key = S.kv + '|' + S.ssm;
        rb.set(nf(BLOCK[key]) + ' tokens' + (key === 'bf16|bf16' ? ' (computed)' : key === 'fp8|fp32' ? ' (computed, TP2)' : ''));
        rt.set(nf(TOK[key].n));
      }
      upd();
      H = V.fig(fig, (g, w, h, t) => {
        autoH(fig, w, 516, 442);
        const C = V.C();
        const key = S.kv + '|' + S.ssm;
        const pg = PAGE[S.ssm], page = pg.rec + pg.conv;
        const bpt = BPT[S.kv], block = BLOCK[key];
        const slice = 64 * bpt;
        const nSl = block / 64;
        const el = t - S.t0;
        const shown = Math.min(nSl, Math.floor(el / 0.05) + 1);
        const right = shown * slice;
        const pad = 14;
        const topH = Math.min(250, h * 0.6);
        const cx = w / 2, fy = (w < 600 ? 60 : 44) + topH * 0.52; // no estreito, a conta do topo ocupa a largura toda
        const panW = Math.min(120, w * 0.26);
        const L = Math.min(260, w * 0.4, w / 2 - panW / 2 - pad - 4);
        const tilt = V.clamp((right - page) / page, -1, 1) * 0.12;
        const pxB = (topH * 0.5) / 1700000;
        // pé e fulcro
        g.fillStyle = V.alpha(C.muted, 0.35);
        g.beginPath(); g.moveTo(cx, fy); g.lineTo(cx - 18, fy + topH * 0.42); g.lineTo(cx + 18, fy + topH * 0.42); g.closePath(); g.fill();
        V.line(g, cx - 40, fy + topH * 0.42, cx + 40, fy + topH * 0.42, { color: V.alpha(C.muted, 0.5), lw: 3 });
        const lx = cx - L * Math.cos(tilt), ly = fy - L * Math.sin(tilt);
        const rx = cx + L * Math.cos(tilt), ry = fy + L * Math.sin(tilt);
        V.line(g, lx, ly, rx, ry, { color: C.ink, lw: 3 });
        V.dot(g, cx, fy, 6, C.d);
        // prato esquerdo: página GDN
        const lh = page * pxB, lrec = pg.rec * pxB;
        V.line(g, lx - panW / 2, ly + 4, lx + panW / 2, ly + 4, { color: C.muted, lw: 2 });
        V.box(g, lx - panW / 2 + 6, ly - lh, panW - 12, lrec, { r: 3, fill: V.alpha(C.b, 0.55) });
        V.box(g, lx - panW / 2 + 6, ly - (lh - lrec), panW - 12, lh - lrec, { r: 2, fill: V.alpha(C.d, 0.75) });
        const lE = lx - panW / 2 + 4, rE = rx + panW / 2 - 4;
        V.text(g, 'GDN page', lE, ly + 22, { size: 12, weight: 600, color: C.ink });
        V.text(g, nf(page) + ' B', lE, ly + 38, { size: 11.5, font: 'mono', color: C.muted });
        V.text(g, `state ${S.ssm.toUpperCase()}`, lx, ly - lh - 8, { size: 11, align: 'center', font: 'mono', color: C.b });
        // prato direito: fatias de 64 tokens
        V.line(g, rx - panW / 2, ry + 4, rx + panW / 2, ry + 4, { color: C.muted, lw: 2 });
        const sh = slice * pxB;
        for (let i = 0; i < shown; i++) {
          const yy = ry - (i + 1) * sh;
          g.fillStyle = V.alpha(S.kv === 'fp8' ? C.a : C.e, i % 2 ? 0.55 : 0.8);
          g.fillRect(rx - panW / 2 + 6, yy, panW - 12, Math.max(0.8, sh - (sh > 3 ? 0.8 : 0)));
        }
        V.text(g, 'KV of one layer', rE, ry + 22, { size: 12, align: 'right', weight: 600, color: C.ink });
        V.text(g, `${nf(shown * 64)} tokens × ${bpt} B`, rE, ry + 38, { size: 11.5, align: 'right', font: 'mono', color: C.muted });
        const done = shown >= nSl;
        V.text(g, done ? `balance: block = ${nf(block)} tokens` : 'stacking 64-token slices…', cx, 22, { size: 14, align: 'center', weight: 700, color: done ? C.a : C.muted });
        if (done) V.text(g, `${nf(block)} × ${bpt} B = ${nf(block * bpt)} B ≥ ${nf(page)} B (spare ${nf(block * bpt - page)} B)`, cx, 40, { size: 11.5, align: 'center', font: 'mono', color: C.muted, maxW: w - 2 * pad });
        // barras de tokens medidos
        const by0 = (w < 600 ? 60 : 44) + topH + 18;
        V.text(g, 'tokens in pool, GMU 0.835 (TP2)', pad, by0 - 6, { size: 11.5, font: 'mono', color: C.muted });
        const keys = ['bf16|fp32', 'fp8|fp32', 'bf16|bf16', 'fp8|bf16'];
        const labs = { 'bf16|fp32': 'KV BF16 · state FP32*', 'fp8|fp32': 'KV FP8 · state FP32*', 'bf16|bf16': 'KV BF16 · state BF16', 'fp8|bf16': 'KV FP8 · state BF16' };
        const avail = h - by0 - (w >= 600 ? 28 : 52);
        const rowH = w >= 600 ? Math.min(26, avail / 4 - 2) : Math.min(34, avail / 4 - 2);
        const labW = w >= 600 ? 170 : 0;
        const barW = w - pad * 2 - labW - (w >= 600 ? 170 : 0);
        keys.forEach((k, i) => {
          const y = by0 + i * (rowH + 3);
          const on = k === key;
          if (labW) V.text(g, labs[k], pad, y + rowH / 2 + 4, { size: 12, weight: on ? 700 : 400, color: on ? C.ink : C.muted });
          const bw2 = (TOK[k].n / 4378829) * (w >= 600 ? barW : w - pad * 2);
          if (labW) {
            V.box(g, pad + labW, y, Math.max(4, bw2), rowH - 4, { r: 4, fill: on ? V.alpha(C.a, 0.8) : V.alpha(C.muted, 0.22) });
            V.text(g, `${nf(TOK[k].n)} (${TOK[k].src})`, pad + labW + bw2 + 8, y + rowH / 2 + 2, { size: 11.5, font: 'mono', color: C.ink, maxW: 300 });
          } else {
            V.text(g, `${labs[k]}: ${nf(TOK[k].n)} (${TOK[k].src})`, pad, y + 11, { size: 11, font: 'mono', weight: on ? 700 : 400, color: on ? C.ink : C.muted, maxW: w - pad * 2 });
            V.box(g, pad, y + 16, Math.max(4, bw2), 9, { r: 3, fill: on ? V.alpha(C.a, 0.85) : V.alpha(C.muted, 0.3) });
          }
        });
        const fnY = by0 + 4 * (rowH + 3) + 12;
        wrapText(g, '* FP32 state inferred: the repo does not record the SSM dtype for the Sep 5 measurement', pad, fnY, w - pad * 2, 15, { size: 11, color: C.muted });
      });
    })();

    /* ================================================================
       FIGURA 5 — Semáforo do top-k
       ================================================================ */
    (() => {
      const fig = root.querySelector('#fig-semaforo');
      const ctl = UI.controls(fig);
      const D = 32, NMAX = 2048, K = 32;
      const NS = [64, 128, 256, 512, 1024, 2048];
      const S = { fmt: 'fp8', ni: 5, rot: 'none', seed: 7, t0: 0 };
      const r = V.rng(23);
      const keys = [];
      for (let j = 0; j < NMAX; j++) { const k = []; for (let d = 0; d < D; d++) { let x = r.normal(); if (d === 5) x *= 8; if (d === 17) x *= 6; k.push(N.round('bf16', x)); } keys.push(k); }
      const signs = N.randSigns(D, 5);
      const rot = (v) => N.fwht(v.map((x, i) => x * signs[i]));
      const keysR = keys.map(rot);
      const cache = {};
      function qset(fmt, rotated) {
        const id = fmt + (rotated ? 'R' : '');
        if (cache[id]) return cache[id];
        const src = rotated ? keysR : keys;
        let out;
        if (fmt === 'bf16') out = src;
        else if (fmt === 'fp8') {
          let amax = 0; src.forEach((k) => k.forEach((x) => { amax = Math.max(amax, Math.abs(x)); }));
          const s = amax / 448;
          out = src.map((k) => k.map((x) => N.encode('e4m3', x / s).v * s));
        } else out = src.map((k) => N.mx(k, 'e2m1', 32).deq);
        cache[id] = out;
        return out;
      }
      let q = [];
      const mkq = () => { const rq = V.rng(S.seed); q = Array.from({ length: D }, (_, d) => N.round('bf16', rq.normal() * (d === 5 || d === 17 ? 0.3 : 1))); };
      mkq();
      const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
      let R = null, prevY = null;
      function compute() {
        const n = NS[S.ni];
        const ex = new Float64Array(n), sq = new Float64Array(n);
        for (let j = 0; j < n; j++) ex[j] = dot(q, keys[j]);
        const rotated = S.rot !== 'none';
        const QK = qset(S.fmt, rotated);
        const qq = S.rot === 'both' ? rot(q) : q;
        for (let j = 0; j < n; j++) sq[j] = dot(qq, QK[j]);
        const ord = (arr) => Array.from({ length: n }, (_, i) => i).sort((a, b) => arr[b] - arr[a]);
        const oe = ord(ex), oq = ord(sq);
        const te = new Set(oe.slice(0, K)), tq = new Set(oq.slice(0, K));
        let inter = 0; te.forEach((i) => { if (tq.has(i)) inter++; });
        let se = 0, m = 0; for (let j = 0; j < n; j++) m += ex[j]; m /= n;
        let v2 = 0; for (let j = 0; j < n; j++) { v2 += (ex[j] - m) ** 2; se += (sq[j] - ex[j]) ** 2; }
        const relErr = Math.sqrt(se / n) / Math.sqrt(v2 / n);
        const thE = (ex[oe[K - 1]] + ex[oe[K]]) / 2, thQ = (sq[oq[K - 1]] + sq[oq[K]]) / 2;
        const lo = Math.min(ex[oe[Math.min(n - 1, 4 * K)]], sq[oq[Math.min(n - 1, 4 * K)]]);
        const hi = Math.max(ex[oe[0]], sq[oq[0]]);
        prevY = R && R.n === n ? R.sq : null;
        R = { n, ex, sq, oe, oq, te, tq, swaps: K - inter, thE, thQ, lo, hi, relErr };
        S.t0 = H ? H.t : 0;
        rs.set(`${R.swaps} of ${K}`);
        ro.set(nf((inter / K) * 100, 0) + '%');
        re.set(nf(relErr * 100, 1) + '%');
      }
      UI.seg(ctl, { label: 'K stored as', options: [{ v: 'bf16', label: 'BF16 (reference)' }, { v: 'fp8', label: 'FP8 E4M3 · tensor' }, { v: 'mx', label: 'MXFP4 · block 32' }], value: 'fp8', onChange: (v) => { S.fmt = v; compute(); } });
      UI.slider(ctl, { label: 'Candidates (depth)', min: 0, max: 5, step: 1, value: 5, fmt: (v) => nf(NS[v]), onInput: (v) => { S.ni = v; compute(); } });
      UI.seg(ctl, { label: 'Hadamard rotation', options: [{ v: 'none', label: 'none' }, { v: 'both', label: 'on q and k' }, { v: 'konly', label: 'k only (bug)' }], value: 'none', onChange: (v) => { S.rot = v; compute(); } });
      UI.button(ctl, '↻ another query', () => { S.seed += 1; mkq(); compute(); });
      const rs = UI.readout(ctl, 'swaps in top-32', '—'), ro = UI.readout(ctl, 'overlap', '—'), re = UI.readout(ctl, 'score error', '—');
      let H = null;
      compute();
      H = V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const pad = 14, left = 44, right = 14, top = 30, stripH = 70;
        const pw = w - left - right, phh = h - top - stripH - 40;
        const u = V.ease.inOut(V.clamp((t - S.t0) / 0.7, 0, 1));
        const lo = R.lo, hi = R.hi, span = hi - lo || 1;
        const X = (v) => left + ((v - lo) / span) * pw;
        const Y = (v) => top + phh - ((v - lo) / span) * phh;
        V.box(g, left, top, pw, phh, { r: 6, fill: V.alpha(C.muted, 0.05), stroke: V.alpha(C.muted, 0.25), lw: 1 });
        // quadrantes
        const xE = X(R.thE), yQ = Y(R.thQ);
        g.fillStyle = V.alpha(C.e, 0.07); g.fillRect(xE, top, left + pw - xE, yQ - top);
        g.fillStyle = V.alpha(C.red, 0.07); g.fillRect(xE, yQ, left + pw - xE, top + phh - yQ);
        g.fillStyle = V.alpha(C.d, 0.08); g.fillRect(left, top, xE - left, yQ - top);
        V.line(g, X(lo), Y(lo), X(hi), Y(hi), { color: V.alpha(C.ink, 0.2), lw: 1, dash: [4, 4] });
        V.line(g, xE, top, xE, top + phh, { color: C.ink, lw: 1.4 });
        V.line(g, left, yQ, left + pw, yQ, { color: C.ink, lw: 1.4, dash: [6, 3] });
        V.text(g, 'exact cutoff', xE + 4, top + phh - 6, { size: 11, font: 'mono', color: C.muted });
        V.text(g, 'cutoff with quantized K', left + 6, yQ - 5, { size: 11, font: 'mono', color: C.muted });
        V.text(g, 'exact score →', left + pw, top + phh + 16, { size: 11.5, align: 'right', color: C.muted });
        g.save(); g.translate(14, top + phh / 2); g.rotate(-Math.PI / 2); V.text(g, 'score with quantized K →', 0, 0, { size: 11.5, align: 'center', color: C.muted }); g.restore();
        const pulse = 0.6 + 0.4 * Math.sin(t * 5);
        for (let j = 0; j < R.n; j++) {
          const ex = R.ex[j];
          const yv = prevY ? V.lerp(prevY[j], R.sq[j], u) : V.lerp(ex, R.sq[j], u);
          if (ex < lo && yv < lo) continue;
          const px = X(Math.max(lo, ex)), py = Y(Math.max(lo, yv));
          const inE = R.te.has(j), inQ = R.tq.has(j);
          let col, rad = 2.4;
          if (inE && inQ) { col = C.e; rad = 3.2; }
          else if (inE) { col = C.red; rad = 3.6 + pulse; }
          else if (inQ) { col = C.d; rad = 3.6 + pulse; }
          else col = V.alpha(C.muted, 0.55);
          V.dot(g, px, py, rad, col);
        }
        // faixa de seleção
        const sy = top + phh + 30;
        const cw = Math.min(18, (w - pad * 2 - 90) / K);
        V.text(g, 'exact', pad, sy + 12, { size: 11.5, font: 'mono', color: C.muted });
        V.text(g, 'quant.', pad, sy + 36, { size: 11.5, font: 'mono', color: C.muted });
        for (let i = 0; i < K; i++) {
          const je = R.oe[i], jq = R.oq[i];
          V.box(g, pad + 58 + i * (cw + 1), sy, cw - 1, 16, { r: 3, fill: R.tq.has(je) ? V.alpha(C.e, 0.8) : C.red });
          V.box(g, pad + 58 + i * (cw + 1), sy + 24, cw - 1, 16, { r: 3, fill: R.te.has(jq) ? V.alpha(C.e, 0.8) : C.d });
        }
        const legendY = top - 12;
        const lg = w < 600 ? [[C.e, 'read'], [C.red, 'out'], [C.d, 'intruder']] : [[C.e, 'read in both'], [C.red, 'out (was top-32)'], [C.d, 'intruder']];
        let lx = left;
        g.font = V.font(11.5);
        lg.forEach(([c, l]) => { V.dot(g, lx + 5, legendY - 4, 4.5, c); V.text(g, l, lx + 14, legendY, { size: 11.5, color: C.ink }); g.font = V.font(11.5); lx += g.measureText(l).width + 30; });
        const nl = `N = ${nf(R.n)} · zoom on the tail (top ~${4 * K})`;
        g.font = V.font(11.5, 'mono');
        if (lx + g.measureText(nl).width + 12 < left + pw) V.text(g, nl, left + pw, legendY, { size: 11.5, align: 'right', font: 'mono', color: C.muted });
      });
    })();

    /* ================================================================
       FIGURA 6 — Anatomia de uma linha de KV (mosaico de bytes)
       ================================================================ */
    (() => {
      const fig = root.querySelector('#fig-linha-kv');
      const ctl = UI.controls(fig);
      const L = {
        glm: { name: 'GLM-5.3-Flash · fp8_ds_mla · 656 B', bf: 1152, segs: [['NoPE FP8 (512 dims × 1 B)', 512, 'a'], ['FP32 scales (4 × 4 B, 1 per 128 dims)', 16, 'd'], ['RoPE BF16 = padding zeros (64 × 2 B)', 128, 'b', 'zero']] },
        v32: { name: 'DeepSeek-V3.2 · fp8_ds_mla · 656 B', bf: 1152, segs: [['NoPE FP8 (512 dims × 1 B)', 512, 'a'], ['FP32 scales (4 × 4 B, 1 per 128 dims)', 16, 'd'], ['RoPE BF16 (64 × 2 B)', 128, 'b']] },
        glm528: { name: 'GLM without padding (hypothetical) · 528 B', bf: 1024, segs: [['NoPE FP8 (512 dims × 1 B)', 512, 'a'], ['FP32 scales (4 × 4 B)', 16, 'd']] },
        tf528: { name: 'GLM · TensorFold kv8 · 528 B', bf: 1024, ext: true, segs: [['E4M3 latent (512 dims × 1 B)', 512, 'a'], ['power-of-two FP32 scale (1 per row)', 4, 'd'], ['padding: zeros up to a multiple of 16 B', 12, 'muted']] },
        ds584: { name: 'DeepSeek-V4 · fp8_ds_mla = nvfp4_ds_mla · 584 B', bf: 1024, ds: true, dsPage: true, segs: [['NoPE FP8 (448 dims, 7 blocks of 64)', 448, 'a'], ['RoPE BF16 (64 × 2 B)', 128, 'b'], ['UE8M0 scales (7 × 1 B)', 7, 'd'], ['padding', 1, 'muted']] },
        dsbf16: { name: 'DeepSeek-V4 in BF16 (hypothetical) · 1024 B', bf: 1024, dsPage: true, segs: [['NoPE BF16 (448 × 2 B)', 896, 'e'], ['RoPE BF16 (64 × 2 B)', 128, 'b']] },
        st416: { name: 'Stage B, abandoned · 416 B', bf: 1024, dsPage: true, segs: [['E2M1 values (512 × 4 bits, GLM-5.2 layout)', 256, 'c'], ['E4M3 scales (1 per 16)', 32, 'd'], ['RoPE BF16 (64 × 2 B)', 128, 'b']] },
        gb352: { name: 'gb10 variant (vLLM 0.26) · 352 B', bf: 1024, dsPage: true, segs: [['E2M1 values (256 B)', 256, 'c'], ['E4M3 scales (1 per 16)', 32, 'd'], ['RoPE FP8 (64 × 1 B)', 64, 'f']] },
        optA: { name: 'Option A (proposed) · 292 B', bf: 1024, dsPage: true, segs: [['NoPE E2M1 (448 × 4 bits)', 224, 'c'], ['UE8M0 scales (1 per 128 dims)', 4, 'd'], ['RoPE FP8 (64 × 1 B)', 64, 'f']] },
      };
      const S = { lay: 'glm', page: false, t0: 0 };
      let H;
      UI.select(ctl, { label: 'Layout', options: Object.keys(L).map((k) => ({ v: k, label: L[k].name })), value: 'glm', onChange: (v) => { S.lay = v; S.t0 = H ? H.t : 0; upd(); } });
      UI.toggle(ctl, { label: 'show the page (64 rows)', value: false, onChange: (v) => { S.page = v; S.t0 = H ? H.t : 0; } });
      const rb = UI.readout(ctl, 'bytes/row', '—'), rr = UI.readout(ctl, 'vs BF16', '—'), rp = UI.readout(ctl, 'rows in 37,376 B', '—');
      const rpLab = rp.el.querySelector('span');
      const total = (lay) => L[lay].segs.reduce((a, s) => a + s[1], 0);
      function upd() {
        const tb = total(S.lay);
        rb.set(nf(tb) + ' B');
        rr.set(nf((tb / L[S.lay].bf) * 100, 0) + '%');
        // a página de 37.376 B (64 × 584) só existe no DeepSeek-V4; nos outros layouts mostramos a página própria
        if (L[S.lay].ext) {
          if (rpLab) rpLab.textContent = 'extent = 2,048 × row';
          rp.set(nf(2048 * tb) + ' B');
        } else if (L[S.lay].dsPage) {
          const n = 37376 / tb;
          if (rpLab) rpLab.textContent = 'rows in DS page (37,376 B)';
          rp.set(Number.isInteger(n) ? nf(n) + ' ✓' : nf(n, 2) + ' ✗');
        } else {
          if (rpLab) rpLab.textContent = 'page = 64 × row';
          rp.set(nf(64 * tb) + ' B');
        }
      }
      upd();
      H = V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const lay = L[S.lay], tb = total(S.lay);
        const pad = 14;
        const colOf = (k) => (k === 'muted' ? V.alpha(C.muted, 0.5) : C[k]);
        const el = t - S.t0;
        V.text(g, lay.name, pad, 22, { size: 13.5, weight: 700, color: C.ink, maxW: w - pad * 2 });
        const wide = w >= 640;
        if (wide) setH(fig, 340); // no estreito, a altura sai do conteúdo de cada modo (abaixo)
        if (!S.page) {
          const cols = wide ? 48 : 32;
          const areaW = wide ? w * 0.62 : w - pad * 2;
          const cs = Math.max(4, Math.floor(areaW / cols) - 1);
          const n = Math.min(tb, Math.floor(el * 2200));
          let seg = 0, acc = lay.segs[0][1];
          for (let b = 0; b < n; b++) {
            while (b >= acc && seg < lay.segs.length - 1) { seg++; acc += lay.segs[seg][1]; }
            const s = lay.segs[seg];
            const x = pad + (b % cols) * (cs + 1), y = 38 + Math.floor(b / cols) * (cs + 1);
            g.fillStyle = s[3] === 'zero' ? V.alpha(colOf(s[2]), 0.18) : V.alpha(colOf(s[2]), 0.85);
            g.fillRect(x, y, cs, cs);
            if (s[3] === 'zero' && cs >= 6) { g.strokeStyle = V.alpha(colOf(s[2]), 0.9); g.lineWidth = 1; g.beginPath(); g.moveTo(x, y + cs); g.lineTo(x + cs, y); g.stroke(); }
          }
          // grade fantasma do BF16 (tamanho de referência)
          const refRows = Math.ceil(lay.bf / cols);
          const my = 38 + refRows * (cs + 1);
          g.strokeStyle = V.alpha(C.muted, 0.35); g.setLineDash([3, 3]); g.lineWidth = 1;
          g.strokeRect(pad - 2, 36, cols * (cs + 1) + 2, refRows * (cs + 1) + 2); g.setLineDash([]);
          V.text(g, `dashed outline = the same row in BF16 (${nf(lay.bf)} B)`, pad, my + 16, { size: 11.5, color: C.muted, maxW: areaW });
          // legenda
          const lx = wide ? pad + cols * (cs + 1) + 22 : pad;
          let ly = wide ? 52 : my + 44;
          lay.segs.forEach((s) => {
            if (s[3] === 'zero') { V.box(g, lx, ly - 11, 14, 14, { r: 3, fill: V.alpha(colOf(s[2]), 0.2) }); hatch(g, lx, ly - 11, 14, 14, colOf(s[2]), 4); }
            else V.box(g, lx, ly - 11, 14, 14, { r: 3, fill: colOf(s[2]) });
            V.text(g, `${nf(s[1])} B`, lx + 22, ly, { size: 12.5, weight: 700, font: 'mono', color: C.ink });
            V.text(g, s[0], lx + 22, ly + 16, { size: 11.5, color: C.muted, maxW: wide ? w - lx - 30 : w - lx - 30 });
            ly += 40;
          });
          V.text(g, `total ${nf(tb)} B · ${nf((tb * 8) / (lay.bf / 2), 2)} bits per dim`, lx, ly + 4, { size: 12.5, weight: 700, color: C.a, maxW: w - lx - pad });
          if (!wide) setH(fig, ly + 22);
        } else {
          // página: 64 linhas
          const rows = 64;
          const pw = w - pad * 2 - (wide ? 190 : 0);
          const rh = wide ? Math.max(2, Math.floor((h - 110) / (rows + 10))) : 5;
          if (!wide) setH(fig, lay.ds ? 40 + rows * rh + 8 + 60 + 18 : 40 + rows * rh + 16 + 20 + 18 + (lay.ext ? 18 : 0));
          const y0 = 40;
          const k = Math.min(rows, Math.floor(el * 90));
          if (lay.ds) {
            // dados (576 B/linha) primeiro, escalas (8 B/linha) no fim
            const dataW = pw * (576 / 584);
            for (let r = 0; r < k; r++) {
              const y = y0 + r * rh;
              g.fillStyle = V.alpha(C.a, 0.8); g.fillRect(pad, y, dataW * (448 / 576), rh - 1);
              g.fillStyle = V.alpha(C.b, 0.8); g.fillRect(pad + dataW * (448 / 576), y, dataW * (128 / 576), rh - 1);
            }
            const fy = y0 + rows * rh + 8;
            const cells = 64;
            const fcw = pw / cells;
            for (let r = 0; r < Math.min(cells, Math.max(0, Math.floor((el - 0.7) * 90))); r++) { g.fillStyle = V.alpha(C.d, 0.9); g.fillRect(pad + r * fcw, fy, Math.max(1, fcw - 1), 18); }
            V.text(g, 'footer: 64 × 8 B of scales (7 UE8M0 + 1 pad per token)', pad, fy + 36, { size: 11.5, font: 'mono', color: C.muted, maxW: pw });
            const lx = wide ? pad + pw + 16 : pad;
            const ly = wide ? 60 : fy + 60;
            if (wide) {
              V.text(g, '64-token block', lx, ly, { size: 12.5, weight: 700, color: C.ink });
              V.text(g, '[0, 64·576): data', lx, ly + 22, { size: 11.5, font: 'mono', color: C.a });
              V.text(g, '448 FP8 + 128 B RoPE', lx, ly + 40, { size: 11.5, font: 'mono', color: C.muted });
              V.text(g, '[64·576, +64·8): scales', lx, ly + 64, { size: 11.5, font: 'mono', color: C.d });
              V.text(g, '= 37,376 B per page', lx, ly + 92, { size: 12.5, weight: 700, color: C.ink });
            } else V.text(g, 'data [0, 64·576) · scales after · 37,376 B', lx, ly, { size: 11.5, font: 'mono', color: C.ink, maxW: pw });
          } else {
            for (let r = 0; r < k; r++) {
              const y = y0 + r * rh; let x = pad;
              lay.segs.forEach((s) => { const sw = (pw * s[1]) / 1152; g.fillStyle = V.alpha(colOf(s[2]), s[3] === 'zero' ? 0.2 : 0.8); g.fillRect(x, y, sw, rh - 1); x += sw; });
            }
            const fy = y0 + rows * rh + 16;
            if (lay.ext) {
              V.text(g, `extent = 2,048 × ${nf(tb)} B = ${nf(2048 * tb)} B (64 rows drawn)`, pad, fy, { size: 11.5, font: 'mono', color: C.muted, maxW: w - pad * 2 });
              if (wide) V.text(g, 'No pages: each stream gets a contiguous range of rows, in multiples of 2,048 tokens.', pad, fy + 20, { size: 11.5, color: C.muted, maxW: w - pad * 2 });
              else { V.text(g, 'No pages: each stream gets a contiguous range', pad, fy + 20, { size: 11.5, color: C.muted, maxW: w - pad * 2 }); V.text(g, 'of rows, in multiples of 2,048 tokens.', pad, fy + 38, { size: 11.5, color: C.muted, maxW: w - pad * 2 }); }
            } else {
              V.text(g, `page = 64 × ${nf(tb)} B = ${nf(64 * tb)} B (width scaled to 1,152 B)`, pad, fy, { size: 11.5, font: 'mono', color: C.muted, maxW: w - pad * 2 });
              V.text(g, 'The internal layout of the page is only documented in the sources for DeepSeek-V4.', pad, fy + 20, { size: 11.5, color: C.muted, maxW: w - pad * 2 });
            }
          }
        }
      });
    })();

    /* ================================================================
       FIGURA 6b — Uma escala por linha, potência de dois (kv8 do TensorFold)
       ================================================================ */
    (() => {
      const fig = root.querySelector('#fig-escala-pot2');
      const ctl = UI.controls(fig);
      const NR = 32, D = 512, LAYER_AMAX = 6;
      const MODES = [
        { v: 'p2', label: '2ᵏ per row', name: '2ᵏ per row (TensorFold kv8)', f: 's = 2^(⌈log₂ amax⌉ − 8): the max lands in (128, 256]' },
        { v: 'f32', label: 'FP32 per row', name: 'Exact FP32 per row', f: 's = amax / 448: the max becomes 448' },
        { v: 'b128', label: 'FP32 per 128 dims', name: 'FP32 per 128-dim block (like fp8_ds_mla)', f: 's = block amax / 448, four scales per row' },
        { v: 'layer', label: 'fixed per layer', name: 'Fixed layer scale (calibrated)', f: 's = 6 / 448 for every row, at any magnitude' },
      ];
      const S = { mode: 'p2', R: 4, mag: 0, seed: 11 };
      const LO = -12, HI = 11, BWD = 0.25, NB = Math.round((HI - LO) / BWD);
      const L448 = Math.log2(448), MINN = Math.pow(2, -6);
      const SUP = { '-': '⁻', 0: '⁰', 1: '¹', 2: '²', 3: '³', 4: '⁴', 5: '⁵', 6: '⁶', 7: '⁷', 8: '⁸', 9: '⁹' };
      const sup = (k) => String(k).split('').map((c) => SUP[c] || c).join('');
      let base = [], res = {};
      const disp = new Float32Array(NB), dsat = new Float32Array(NB);
      /* 32 linhas: 511 gaussianas com RMS 1 + 1 outlier que fixa amax/RMS ≈ R (sorteado entre 0,8R e 1,2R por linha) */
      function gen() {
        const r = V.rng(S.seed);
        base = [];
        for (let k = 0; k < NR; k++) {
          const b = Array.from({ length: D }, () => r.normal());
          let im = 0;
          for (let i = 1; i < D; i++) if (Math.abs(b[i]) > Math.abs(b[im])) im = i;
          let s2 = 0;
          for (let i = 0; i < D; i++) if (i !== im) s2 += b[i] * b[i];
          const kk = Math.sqrt((D - 1) / s2);
          for (let i = 0; i < D; i++) if (i !== im) b[i] *= kk;
          const Rk = S.R * (0.8 + 0.4 * r());
          const rms = Math.sqrt((D - 1) / (D - Rk * Rk));
          b[im] = (b[im] < 0 ? -1 : 1) * Math.max(Rk * rms, Math.abs(b[im]) * kk);
          base.push(b);
        }
      }
      function calc() {
        const m = Math.pow(2, S.mag);
        res = {};
        MODES.forEach(({ v }) => {
          const hist = new Float32Array(NB), hsat = new Float32Array(NB);
          let errSum = 0, sat = 0, sub = 0, s0 = 0, top = 0;
          base.forEach((b, k) => {
            const x = b.map((y) => y * m);
            let amax = 0;
            for (let i = 0; i < D; i++) amax = Math.max(amax, Math.abs(x[i]));
            let rowS = 0, blk = null;
            if (v === 'p2') rowS = Math.pow(2, Math.ceil(Math.log2(amax)) - 8);
            else if (v === 'f32') rowS = amax / 448;
            else if (v === 'layer') rowS = LAYER_AMAX / 448;
            else {
              blk = [0, 1, 2, 3].map((q) => { let a = 0; for (let i = q * 128; i < q * 128 + 128; i++) a = Math.max(a, Math.abs(x[i])); return a / 448; });
            }
            let e2 = 0, x2 = 0;
            for (let i = 0; i < D; i++) {
              const s = blk ? blk[i >> 7] : rowS;
              const y = x[i] / s;
              const xh = N.encode('e4m3', y).v * s;
              e2 += (xh - x[i]) * (xh - x[i]); x2 += x[i] * x[i];
              const ay = Math.abs(y);
              const isSat = ay > 448 * (1 + 1e-9); /* tolerância: amax/(amax/448) pode dar 448,0000001 */
              if (isSat) sat++; else if (ay < MINN) sub++;
              if (ay > top) top = ay;
              const lb = ay > 0 ? Math.log2(ay) : LO;
              const bi = Math.max(0, Math.min(NB - 1, Math.floor((lb - LO) / BWD)));
              hist[bi]++; if (isSat) hsat[bi]++;
            }
            errSum += Math.sqrt(e2 / x2);
            if (k === 0) s0 = blk ? blk[0] : rowS;
          });
          res[v] = { err: errSum / NR, sat, sub, s0, top, hist, hsat };
        });
      }
      UI.seg(ctl, { label: 'Scale', options: MODES.map((m) => ({ v: m.v, label: m.label })), value: 'p2', onChange: (v) => { S.mode = v; upd(); } });
      UI.slider(ctl, { label: 'row amax / RMS', min: 3, max: 8, step: 0.5, value: 4, fmt: (v) => nf(v, 1) + '×', onInput: (v) => { S.R = v; gen(); calc(); upd(); } });
      UI.slider(ctl, { label: 'row magnitude', min: -14, max: 6, step: 1, value: 0, fmt: (v) => '×2' + sup(v), onInput: (v) => { S.mag = v; calc(); upd(); } });
      UI.button(ctl, 'new sample', () => { S.seed += 1; gen(); calc(); upd(); });
      const rS = UI.readout(ctl, 'scale of row 1', '—'), rE = UI.readout(ctl, 'mean relative error', '—'), rX = UI.readout(ctl, 'saturated · subnormal', '—');
      function upd() {
        const r = res[S.mode];
        const k = Math.round(Math.log2(r.s0));
        rS.set(S.mode === 'p2' ? `2${sup(k)} = ${expo(r.s0)}` : expo(r.s0));
        rE.set(nf(r.err * 100, 2) + '%', r.err > 0.05 ? 'bad' : 'ok');
        rX.set(`${nf(r.sat)} · ${nf(r.sub)} of ${nf(NR * D)}`, r.sat ? 'bad' : r.sub > NR * D * 0.01 ? 'warn' : 'ok');
      }
      gen(); calc(); upd();
      V.fig(fig, (g, w, h, t, dt) => {
        const C = V.C();
        const r = res[S.mode], md = MODES.find((m) => m.v === S.mode);
        const pad = 14, wide = w >= 600;
        const a = Math.min(1, (dt || 0.016) * 8);
        for (let i = 0; i < NB; i++) { disp[i] += (r.hist[i] - disp[i]) * a; dsat[i] += (r.hsat[i] - dsat[i]) * a; }
        V.text(g, md.name, pad, 20, { size: 13.5, weight: 700, color: C.ink, maxW: w - pad * 2 });
        V.text(g, md.f, pad, 38, { size: 11.5, font: 'mono', color: C.muted, maxW: w - pad * 2 });
        /* 1) a régua do E4M3 em log2 |x/s| */
        const x0 = pad, x1 = w - pad, yT = 68, yB = wide ? 212 : 202;
        const X = (l) => x0 + ((l - LO) / (HI - LO)) * (x1 - x0);
        const zones = [[LO, -10, C.muted, 0.12, ['rounds to 0', '0']], [-10, -6, C.d, 0.18, ['subnormal', 'sub']], [-6, L448, C.a, 0.10, ['normal: 3 mantissa bits at any height', 'normal']], [L448, HI, C.red, 0.16, ['saturates', 'sat']]];
        zones.forEach(([l0, l1, col, al, labs]) => {
          const zw = X(l1) - X(l0);
          g.fillStyle = V.alpha(col, al); g.fillRect(X(l0), yT, zw, yB - yT);
          g.font = V.font(11);
          const lab = (wide ? labs : labs.slice(-1)).find((s) => g.measureText(s).width < zw - 6); /* no estreito, o rótulo longo colidiria com a marca do máximo */
          if (lab) V.text(g, lab, X(l0) + zw / 2, yT + 14, { size: 11, align: 'center', color: C.muted });
        });
        let hmax = 1;
        for (let i = 0; i < NB; i++) hmax = Math.max(hmax, r.hist[i]);
        const bwPx = (x1 - x0) / NB, hh = yB - yT - 26;
        for (let i = 0; i < NB; i++) {
          if (disp[i] < 0.5) continue;
          const lc = LO + (i + 0.5) * BWD;
          const bh = Math.max(1.5, (disp[i] / hmax) * hh);
          const sh = Math.min(bh, (Math.max(0, dsat[i]) / hmax) * hh);
          const bx = x0 + i * bwPx + 0.5, bwi = Math.max(1, bwPx - 1);
          g.fillStyle = V.alpha(lc < -6 ? C.d : C.b, 0.85); g.fillRect(bx, yB - bh, bwi, bh - sh);
          if (sh > 0.3) { g.fillStyle = V.alpha(C.red, 0.9); g.fillRect(bx, yB - sh, bwi, sh); }
        }
        V.line(g, x0, yB, x1, yB, { color: V.alpha(C.ink, 0.5), lw: 1 });
        [[-9, '2' + sup(-9)], [-6, '2' + sup(-6)], [0, '1'], [4, '2' + sup(4)], [L448, '448']].forEach(([l, lab]) => {
          V.line(g, X(l), yB, X(l), yB + 4, { color: V.alpha(C.ink, 0.5), lw: 1 });
          V.text(g, lab, X(l), yB + 16, { size: 11, align: 'center', font: 'mono', color: C.muted });
        });
        /* marca de onde cai o máximo (acima da régua) */
        const mk = (l0, l1, lab, col) => {
          const ya = yT - 8;
          V.line(g, X(l0), ya, X(l1), ya, { color: col, lw: 2.2 });
          [l0, l1].forEach((l) => V.line(g, X(l), ya - 3, X(l), yB, { color: V.alpha(col, 0.55), lw: 1.2, dash: [3, 3] }));
          g.font = V.font(11, 'sans', 700);
          const tw = g.measureText(lab).width;
          const tx = Math.max(x0, Math.min(X(l1) - tw, x1 - tw));
          V.text(g, lab, tx, ya - 6, { size: 11, weight: 700, color: col });
        };
        if (S.mode === 'p2') mk(7, 8, 'max of each row: 128 to 256', C.a);
        else if (S.mode === 'f32') mk(L448, L448, 'max of each row = 448', C.a);
        else if (S.mode === 'b128') mk(L448, L448, 'max of each block = 448', C.a);
        else { const lt = Math.max(LO, Math.min(HI, Math.log2(r.top))); mk(lt, lt, `largest |x/s| = ${nf(r.top, r.top < 10 ? 2 : 0)}`, r.top > 448 ? C.red : C.a); }
        /* 2) erro relativo dos quatro modos */
        const y2 = yB + 42;
        V.text(g, 'RELATIVE ERROR PER ROW (MEAN OF 32 ROWS)', pad, y2, { size: 11, font: 'mono', color: C.muted, maxW: w - pad * 2 });
        let emax = 0.05;
        MODES.forEach((m) => { emax = Math.max(emax, res[m.v].err); });
        const labW = wide ? 170 : 0, rowH = wide ? 30 : 40;
        MODES.forEach((m, i) => {
          const e = res[m.v], sel = m.v === S.mode;
          const y = y2 + 12 + i * rowH;
          if (wide) V.text(g, m.label, pad, y + 14, { size: 12, weight: sel ? 700 : 400, color: sel ? C.ink : C.muted, maxW: labW - 10 });
          else V.text(g, m.label, pad, y + 10, { size: 11.5, weight: sel ? 700 : 400, color: sel ? C.ink : C.muted, maxW: (w - pad * 2) * 0.45 });
          const yb = wide ? y + 2 : y + 16;
          const avail = wide ? Math.max(40, w - pad * 2 - labW - 170) : w - pad * 2;
          const bw = Math.max(3, (e.err / emax) * avail);
          const col = e.sat ? C.red : e.sub > NR * D * 0.01 ? C.d : C.a;
          V.box(g, pad + labW, yb, bw, 16, { r: 4, fill: V.alpha(col, sel ? 0.9 : 0.4) });
          let txt = nf(e.err * 100, 2) + '%';
          if (e.sat) txt += ` · ${nf(e.sat)} saturated`;
          else if (e.sub > NR * D * 0.01) txt += ` · ${nf(e.sub)} subnormal`;
          if (wide) V.text(g, txt, pad + labW + bw + 8, yb + 13, { size: 12, font: 'mono', weight: sel ? 700 : 400, color: C.ink });
          else V.text(g, txt, w - pad, y + 10, { size: 11.5, font: 'mono', weight: sel ? 700 : 400, color: C.ink, align: 'right' });
        });
        setH(fig, y2 + 12 + MODES.length * rowH + 8);
      });
    })();

    /* ================================================================
       FIGURA 7 — Mesmos bytes, trilhos diferentes (#22)
       ================================================================ */
    (() => {
      const fig = root.querySelector('#fig-trilhos');
      const ctl = UI.controls(fig);
      const S = { dt: 'nvfp4_ds_mla', fix: true, t0: 0, tok: 0 };
      let H;
      const reset = () => { S.t0 = H ? H.t : 0; S.tok = 0; };
      UI.seg(ctl, { label: '--kv-cache-dtype', options: [{ v: 'fp8_ds_mla', label: 'fp8_ds_mla' }, { v: 'nvfp4_ds_mla', label: 'nvfp4_ds_mla' }], value: 'nvfp4_ds_mla', onChange: (v) => { S.dt = v; reset(); upd(); } });
      const fixLabel = (on) => (on ? 'hotfix #22 applied (DSPARK_SKIP_ISSUE22_HOTFIX=0)' : 'hotfix #22 skipped (DSPARK_SKIP_ISSUE22_HOTFIX=1)');
      const tg = UI.toggle(ctl, { label: fixLabel(true), value: true, onChange: (v) => { S.fix = v; const sp = tg.el.querySelector('span'); if (sp) sp.textContent = fixLabel(v); reset(); upd(); } });
      const rk = UI.readout(ctl, 'kernel', '—'), rt = UI.readout(ctl, 'decode at 600K+', '—');
      const fast = () => S.dt === 'fp8_ds_mla' || S.fix;
      function upd() { rk.set(fast() ? '_forward_fp8_kv' : '_forward_bf16_kv'); rt.set(fast() ? '≈ 17.3 tok/s' : '≈ 1.0 tok/s'); }
      upd();
      H = V.fig(fig, (g, w, h, t, dt) => {
        const C = V.C();
        const pad = 14;
        const f = fast();
        // código
        const code = S.fix ? 'use_fp8_cache = self.kv_cache_dtype in ("fp8_ds_mla", "nvfp4_ds_mla")' : 'use_fp8_cache = self.kv_cache_dtype == "fp8_ds_mla"';
        V.box(g, pad, 10, w - pad * 2, 40, { r: 8, fill: V.alpha(C.muted, 0.1), stroke: V.alpha(C.muted, 0.3), lw: 1 });
        V.text(g, code, pad + 10, 28, { size: w < 600 ? 11 : 12, font: 'mono', color: C.ink, maxW: w - pad * 2 - 20 });
        V.text(g, `→ ${f ? 'True' : 'False'}   (kv_cache_dtype = "${S.dt}")`, pad + 10, 44, { size: 11.5, font: 'mono', weight: 700, color: f ? C.a : C.red, maxW: w - pad * 2 - 20 });
        // trilhos
        const x0 = pad + 10, xj = w * 0.34, x1 = w - pad - 10;
        const yM = 150, yU = 104, yD = 196;
        const track = (pts, on, colr) => {
          g.strokeStyle = on ? colr : V.alpha(C.muted, 0.35); g.lineWidth = on ? 5 : 3;
          g.beginPath(); g.moveTo(pts[0][0], pts[0][1]); for (let i = 1; i < pts.length; i++) g.lineTo(pts[i][0], pts[i][1]); g.stroke();
          g.strokeStyle = V.alpha(C.ink, on ? 0.5 : 0.2); g.lineWidth = 1; g.setLineDash([2, 9]);
          g.beginPath(); g.moveTo(pts[0][0], pts[0][1]); for (let i = 1; i < pts.length; i++) g.lineTo(pts[i][0], pts[i][1]); g.stroke(); g.setLineDash([]);
        };
        const pU = [[x0, yM], [xj, yM], [xj + 50, yU], [x1, yU]];
        const pD = [[x0, yM], [xj, yM], [xj + 50, yD], [x1, yD]];
        track(pU, f, C.a); track(pD, !f, C.red);
        V.dot(g, xj, yM, 7, C.d);
        V.text(g, 'switch: compares the string', xj, yM + 26, { size: 11.5, align: 'center', font: 'mono', color: C.d });
        const endW = Math.min(170, w * 0.36);
        V.box(g, x1 - endW, yU - 36, endW, 26, { r: 6, fill: V.alpha(C.a, 0.16), stroke: C.a, lw: 1 });
        V.text(g, '_forward_fp8_kv', x1 - endW / 2, yU - 19, { size: 11.5, align: 'center', font: 'mono', color: C.ink });
        V.box(g, x1 - endW, yD + 10, endW, 26, { r: 6, fill: V.alpha(C.red, 0.14), stroke: C.red, lw: 1 });
        V.text(g, '_forward_bf16_kv', x1 - endW / 2, yD + 27, { size: 11.5, align: 'center', font: 'mono', color: C.ink });
        V.text(g, 'request', x0, yM - 14, { size: 11.5, font: 'mono', color: C.muted });
        // trem
        const pts = f ? pU : pD;
        const segL = []; let tot = 0;
        for (let i = 1; i < pts.length; i++) { const d = Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]); segL.push(d); tot += d; }
        const period = f ? 2.6 : 2.6 * 17.3;
        const at = (s) => { s = ((s % tot) + tot) % tot; for (let i = 0; i < segL.length; i++) { if (s <= segL[i]) { const u = s / segL[i]; return [V.lerp(pts[i][0], pts[i + 1][0], u), V.lerp(pts[i][1], pts[i + 1][1], u)]; } s -= segL[i]; } return pts[pts.length - 1]; };
        const head = ((t - S.t0) / period) * tot;
        for (let k = 0; k < 5; k++) {
          const p = at(head - k * 20);
          V.box(g, p[0] - 8, p[1] - 8, 16, 16, { r: 4, fill: k === 0 ? (f ? C.a : C.red) : V.alpha(f ? C.a : C.red, 0.6) });
        }
        // contador
        S.tok += dt * (f ? 17.3 : 1.0) * (LA.paused ? 0 : 1);
        V.text(g, `tokens generated since the change: ${nf(Math.floor(S.tok))}`, pad, 246, { size: 14, weight: 700, color: f ? C.a : C.red, maxW: w - pad * 2 });
        V.text(g, f ? 'FP8 kernel reads the 584 B directly: ~17.3 tok/s at 600K+ of context' : 'BF16 path: ~1.0 tok/s at 600K+ (at short context, ~66 tok/s for both)', pad, 266, { size: 12, color: C.muted, maxW: w - pad * 2 });
        // linha de 584 B igual para os dois
        const ry = 296, rw = w - pad * 2, rh = 22;
        const segs = [[448, C.a, 'NoPE FP8 448'], [128, C.b, 'RoPE BF16 128'], [8, C.d, '8']];
        let x = pad;
        segs.forEach(([b, c, l]) => { const sw = (rw * b) / 584; V.box(g, x, ry, sw - 1, rh, { r: 3, fill: V.alpha(c, 0.8) }); g.font = V.font(11, 'mono'); const lab = g.measureText(l).width < sw - 8 ? l : l.split(' ').pop(); if (g.measureText(lab).width < sw - 4) V.text(g, lab, x + sw / 2, ry + 15, { size: 11, align: 'center', font: 'mono', color: C.bg }); x += sw; });
        V.text(g, w < 600 ? 'same 584 B row for both names' : 'the same 584 B row for fp8_ds_mla and nvfp4_ds_mla: only the name changes', pad, ry + rh + 20, { size: 11.5, color: C.muted, maxW: w - pad * 2 });
      });
    })();

    /* ================================================================
       FIGURA 8 — O que se lê por token de consulta (indexador)
       ================================================================ */
    (() => {
      const fig = root.querySelector('#fig-indexador');
      const ctl = UI.controls(fig);
      const CT = [32768, 131072, 262144, 524288, 1048576];
      const CTL = ['32K', '128K', '256K', '512K', '1M'];
      const S = { ci: 4, mx: false };
      UI.slider(ctl, { label: 'Context', min: 0, max: 4, step: 1, value: 4, fmt: (v) => CTL[v], onInput: (v) => { S.ci = v; upd(); } });
      UI.toggle(ctl, { label: 'indexer K cache in MXFP4', value: false, onChange: (v) => { S.mx = v; upd(); } });
      const ri = UI.readout(ctl, 'indexer', '—'), ra = UI.readout(ctl, 'all of attention', '—'), rl = UI.readout(ctl, 'allocation', '132 B/row');
      function calc(n, mx) {
        const swa = 2 * Math.min(128, n) * 584;
        const c4 = 21 * (Math.min(512, Math.floor(n / 4)) + Math.min(128, n)) * 584;
        const c128 = 20 * (Math.floor(n / 128) + Math.min(128, n)) * 584;
        const idx = 21 * Math.floor(n / 4) * (mx ? 66 : 132);
        return { swa, c4, c128, idx };
      }
      function upd() { const c = calc(CT[S.ci], S.mx); ri.set(fmtB(c.idx)); ra.set(fmtB(c.swa + c.c4 + c.c128)); }
      upd();
      const picks = Array.from({ length: 9 }, (_, i) => (i * 37 + 11) % 96);
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const pad = 14;
        const n = CT[S.ci];
        const c = calc(n, S.mx);
        const maxv = calc(1048576, false).idx;
        // prateleira
        const cells = w < 600 ? 48 : 96;
        const sw = (w - pad * 2) / cells;
        const shH = S.mx ? 16 : 32;
        const sy = 46;
        V.text(g, `compressed indexer keys: ${nf(Math.floor(n / 4))} per C4A layer × 21 layers`, pad, 18, { size: 11.5, font: 'mono', color: C.muted, maxW: w - pad * 2 });
        const idxLab = S.mx
          ? (w >= 600 ? 'MXFP4: half the bytes read per key (66 B; ~68 B by the inferred composition)' : 'MXFP4: ~half per key (66–68 B, inferred)')
          : (w >= 600 ? 'FP8: 132 B per key (128 + 4 of scale; inferred composition)' : 'FP8: 132 B per key (128 + 4, inferred)');
        V.text(g, idxLab, pad, 36, { size: 11.5, weight: 600, color: S.mx ? C.a : C.muted, maxW: w - pad * 2 });
        const speed = S.mx ? 2 : 1;
        const scan = ((t * 0.35 * speed) % 1) * (w - pad * 2);
        for (let i = 0; i < cells; i++) {
          const x = pad + i * sw;
          const lit = x < pad + scan;
          const pick = picks.includes(i % 96) && i < cells;
          g.fillStyle = pick && lit ? C.d : lit ? V.alpha(C.b, 0.55) : V.alpha(C.b, 0.2);
          g.fillRect(x, sy + (32 - shH), Math.max(1, sw - 1), shH);
        }
        V.line(g, pad + scan, sy - 4, pad + scan, sy + 36, { color: C.c, lw: 2.5 });
        V.glow(g, pad + scan, sy + 16, 16, C.c);
        // chaves escolhidas descendo
        const ay = sy + 64;
        const boxW = Math.min(240, w * 0.5), bx = (w - boxW) / 2;
        picks.forEach((p, k) => {
          const x = pad + (p % cells) * sw;
          if (x > pad + scan) return;
          const u = ((t * 0.9 + k * 0.13) % 1);
          const px = V.lerp(x, bx + boxW * ((k + 0.5) / picks.length), u), py = V.lerp(sy + 34, ay + 8, u);
          V.dot(g, px, py, 3, C.d);
        });
        V.box(g, bx, ay + 8, boxW, 24, { r: 8, fill: V.alpha(C.d, 0.14), stroke: C.d, lw: 1.2 });
        V.text(g, 'attention: top-512 + window of 128', bx + boxW / 2, ay + 24, { size: 11.5, align: 'center', font: 'mono', color: C.ink, maxW: boxW - 8 });
        // barras
        const bars = [['SWA (layers 0–1)', c.swa, C.e], ['C4A: 21 × (512 + 128) rows', c.c4, C.a], ['C128A: 20 × (N/128 + 128) rows', c.c128, C.b], ['indexer: 21 × N/4 keys', c.idx, C.c]];
        const by0 = ay + 56;
        const wide = w >= 600;
        const labW = wide ? 230 : 0;
        const rowH = wide ? 34 : 46;
        const barMax = w - pad * 2 - labW - 90;
        V.text(g, `bytes read per query token · context ${CTL[S.ci]}`, pad, by0 - 8, { size: 11.5, font: 'mono', color: C.muted });
        bars.forEach(([l, v, col], i) => {
          const y = by0 + i * rowH;
          if (wide) V.text(g, l, pad, y + 16, { size: 12, color: C.ink });
          else V.text(g, l, pad, y + 11, { size: 11.5, color: C.ink });
          const yb = wide ? y + 4 : y + 17;
          const bw = Math.max(3, (v / maxv) * barMax);
          V.box(g, pad + labW, yb, bw, 18, { r: 4, fill: V.alpha(col, 0.8) });
          V.text(g, fmtB(v), pad + labW + bw + 8, yb + 14, { size: 12, font: 'mono', weight: 600, color: C.ink });
        });
        const att = c.swa + c.c4 + c.c128;
        const ty = by0 + 4 * rowH + 14;
        V.text(g, `indexer / attention = ${nf(c.idx / att, 1)}× · allocation stays at 132 B/row${S.mx ? ' (the writer uses the 1st half)' : ''}`, pad, ty, { size: 12.5, weight: 700, color: c.idx > att ? C.c : C.a, maxW: w - pad * 2 });
      });
    })();

    /* ================================================================
       FIGURA 9 — Tetris da página de 37.376 B
       ================================================================ */
    (() => {
      const fig = root.querySelector('#fig-pagina');
      const ctl = UI.controls(fig);
      const PAGE = 37376;
      const OPTS = [[584, 'current'], [416, 'Stage B'], [368, 'option C'], [352, 'gb10'], [292, 'option A'], [146, '½ of A']];
      const S = { row: 292, t0: 0 };
      let H;
      UI.seg(ctl, { label: 'Compressed row (bytes)', options: OPTS.map(([b, l]) => ({ v: b, label: `${b} · ${l}` })), value: 292, onChange: (v) => { S.row = v; S.t0 = H ? H.t : 0; upd(); } });
      const rn = UI.readout(ctl, 'rows', '—'), rs = UI.readout(ctl, 'leftover', '—'), rbk = UI.readout(ctl, 'C4 block', '—');
      const isPow2 = (n) => n > 0 && (n & (n - 1)) === 0;
      function upd() {
        const n = Math.floor(PAGE / S.row), left = PAGE - n * S.row;
        rn.set(nf(n) + (left === 0 && isPow2(n) ? ' ✓' : ' ✗'));
        rs.set(nf(left) + ' B');
        rbk.set(left === 0 && isPow2(n) ? nf(n * 4) + ' tokens' : '—');
      }
      upd();
      H = V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const pad = 14;
        const colW = Math.min(170, (w - pad * 2 - 40) / 2);
        const ph = h - 120;
        const y0 = 50;
        const xs = w / 2 - colW - 20, xc = w / 2 + 20;
        const el = t - S.t0;
        const drawPage = (x, row, prog, colr, label) => {
          const n = Math.floor(PAGE / row), left = PAGE - n * row;
          const rh = (ph * row) / PAGE;
          V.box(g, x - 3, y0 - 3, colW + 6, ph + 6, { r: 6, stroke: V.alpha(C.ink, 0.5), lw: 1.5 });
          const k = Math.min(n, Math.floor(prog * n));
          for (let i = 0; i < k; i++) {
            const y = y0 + ph - (i + 1) * rh;
            g.fillStyle = V.alpha(colr, i % 2 ? 0.55 : 0.8);
            g.fillRect(x, y, colW, Math.max(0.7, rh - (rh > 3 ? 1 : 0)));
          }
          if (k === n && left > 0) {
            const lh = (ph * left) / PAGE;
            g.fillStyle = V.alpha(C.red, 0.25); g.fillRect(x, y0, colW, lh);
            hatch(g, x, y0, colW, Math.max(lh, 2), C.red, 5);
          }
          V.text(g, label, x + colW / 2, y0 - 14, { size: 12, align: 'center', weight: 600, color: C.ink, maxW: colW + 30 });
          const ok = left === 0 && isPow2(n);
          V.text(g, `${nf(n)} rows${left ? ` + ${nf(left)} B left over` : ''}`, x + colW / 2, y0 + ph + 22, { size: 12, align: 'center', font: 'mono', color: C.ink, maxW: colW + 40 });
          V.text(g, ok ? '✓ integer, power of 2' : left ? '✗ does not divide the page' : '✗ not a power of 2', x + colW / 2, y0 + ph + 42, { size: 12, align: 'center', weight: 700, color: ok ? C.a : C.red, maxW: colW + 40 });
        };
        drawPage(xs, 584, 1, C.b, 'SWA · 584 B (C++ op)');
        drawPage(xc, S.row, V.clamp(el / 1.4, 0, 1), C.a, `compressed C4 · ${S.row} B`);
        V.text(g, `common page of the hybrid allocator = 64 × 584 = ${nf(PAGE)} B`, w / 2, 20, { size: 12, align: 'center', font: 'mono', color: C.muted, maxW: w - pad * 2 });
      });
    })();

    /* ================================================================
       FIGURA 10 — Quanto custa 1 milhão de tokens
       ================================================================ */
    (() => {
      const fig = root.querySelector('#fig-contabilidade');
      const ctl = UI.controls(fig);
      /* 1M = 1.048.576 tokens (2^20), a mesma convenção do "2,22×" do log de boot do DeepSeek */
      const CTX = [8192, 32768, 131072, 262144, 524288, 850000, 1048576];
      const CTXL = ['8K', '32K', '128K', '256K', '512K', '850K', '1M'];
      const S = { rec: 'ds', ci: 6, seqs: 2, mx: false, optA: false };
      UI.seg(ctl, { label: 'Recipe', options: [{ v: 'ds', label: 'DeepSeek-V4' }, { v: 'glm', label: 'GLM-5.3' }, { v: 'qwen', label: 'Qwen3.8' }], value: 'ds', onChange: (v) => { S.rec = v; upd(); } });
      UI.slider(ctl, { label: 'Context', min: 0, max: CTX.length - 1, step: 1, value: 6, fmt: (v) => CTXL[v], onInput: (v) => { S.ci = v; upd(); } });
      UI.slider(ctl, { label: 'Live sequences', min: 1, max: 6, step: 1, value: 2, fmt: (v) => String(v), onInput: (v) => { S.seqs = v; upd(); } });
      UI.toggle(ctl, { label: 'DS: MXFP4 indexer, half row', value: false, onChange: (v) => { S.mx = v; upd(); } });
      UI.toggle(ctl, { label: 'DS: option A (292 B)', value: false, onChange: (v) => { S.optA = v; upd(); } });
      const rT = UI.readout(ctl, 'B/token', '—'), rS = UI.readout(ctl, 'per sequence', '—'), rF = UI.readout(ctl, 'fits', '—');
      function model() {
        if (S.rec === 'ds') {
          const row = S.optA ? 292 : 584;
          const fut = S.mx || S.optA;
          return {
            comps: [['C4A 21 × ' + row + '/4', (21 * row) / 4, 'a'], ['C128A 20 × ' + row + '/128', (20 * row) / 128, 'b'], ['indexer 21 × ' + (S.mx ? 66 : 132) + '/4', (21 * (S.mx ? 66 : 132)) / 4, 'c']],
            fixed: 46 * 128 * 584, fixedL: 'SWA windows: 46 × 128 × 584 B, as counted in the design document', pool: 17.04 * GiB, poolL: '17.04 GiB per rank',
            head: fut ? null : 2331430, headL: fut ? 'future improvements on: physical estimate; the allocator figure was not measured' : 'boot log: 2,331,430 tokens (≈7.8 KB/token as the allocator counts it)',
            maxLen: 1048576, maxLenL: '1,048,576',
            cmp: [['no compression (43 × 584)', 43 * 584], ['typical GQA-8 BF16', 176128]],
          };
        }
        if (S.rec === 'glm') {
          return {
            comps: [['MLA 11 × 656 (fp8_ds_mla)', 11 * 656, 'a'], ['indexer 11 × 33', 11 * 33, 'c']],
            fixed: 0, fixedL: 'not modeled: the ~7.4 GiB of fixed memory in the README (inferred composition: KDA state, drafter window, etc.); the physical count comes out optimistic', pool: 11 * GiB, poolL: 'fixed 11 GiB pool (--kv-cache-memory-bytes)',
            head: 1233779, headL: 'boot log: 1,233,779 tokens with compact drafter pages (1.45× at 850k)',
            maxLen: 850000, maxLenL: '850,000',
            cmp: [['MLA in BF16 (11 × 1,152)', 11 * 1152], ['equivalent BF16 MHA (32 heads per rank)', 11 * 32 * 512 * 2]],
          };
        }
        return {
          comps: [['main KV FP8 (13 × 2 × 256 × 1 B)', 6656, 'a'], ['rest: indexer, state, pages', 2758, 'd']],
          fixed: 0, fixedL: 'GDN state amortized in the measured count (the bytes per token come from the boot log itself)', pool: 32.02 * GiB, poolL: '32.02 GiB per GPU',
          head: 3652200, headL: 'boot log: 3,652,200 tokens (9,418 B/token/GPU)',
          maxLen: 262144, maxLenL: '262,144, the native length; going beyond it requires YaRN, not validated',
          cmp: [['same model with BF16 KV', 15971], ['if all 48 layers had QSA KV (BF16)', 48 * 2 * 256 * 2]],
        };
      }
      function calc() {
        const m = model();
        const bpt = m.comps.reduce((a, c) => a + c[1], 0);
        const n = CTX[S.ci];
        const seq = bpt * n + m.fixed;
        const phys = Math.floor(m.pool / seq);
        const ratio = m.head ? m.head / n : null;
        const alloc = m.head ? Math.floor(ratio) : null;
        return { m, bpt, n, seq, phys, ratio, alloc };
      }
      const ratioTxt = (r) => nf(r, r < 10 ? 2 : 1) + '×';
      function upd() {
        const c = calc();
        rT.set(fmtB(c.bpt)); rS.set(fmtB(c.seq));
        rF.set(c.alloc == null ? `physical ${nf(c.phys)} · allocator not measured` : `physical ${nf(c.phys)} · allocator ${nf(c.alloc)} (${ratioTxt(c.ratio)})`);
      }
      upd();
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const { m, bpt, n, seq, phys, ratio, alloc } = calc();
        const pad = 14, bw = w - pad * 2;
        const wide = w >= 600;
        // 1) bytes por token
        V.text(g, `1 · BYTES PER TOKEN THAT GROW WITH CONTEXT: ${fmtB(bpt)}`, pad, 20, { size: 11.5, font: 'mono', color: C.muted, maxW: bw });
        let x = pad; const y1 = 30, h1 = 30;
        const scale1 = bw / Math.max(bpt, 1);
        m.comps.forEach(([l, v, k]) => {
          const sw = v * scale1;
          V.box(g, x, y1, Math.max(2, sw - 2), h1, { r: 5, fill: V.alpha(C[k], 0.8) });
          if (sw > 80) V.text(g, fmtB(v), x + sw / 2, y1 + 20, { size: 12, align: 'center', font: 'mono', weight: 700, color: C.bg });
          x += sw;
        });
        let ly = y1 + h1 + 18, lx = pad;
        m.comps.forEach(([l, v, k]) => {
          const txt = `${l} = ${fmtB(v)}`;
          g.font = V.font(11.5); const tw = g.measureText(txt).width + 30;
          if (lx + tw > w - pad) { lx = pad; ly += 18; }
          V.dot(g, lx + 5, ly - 4, 4.5, C[k]); V.text(g, txt, lx + 14, ly, { size: 11.5, color: C.ink });
          lx += tw;
        });
        // 2) sequências contra o pool
        const y2 = ly + 30;
        V.text(g, `2 · ${S.seqs} × ${CTXL[S.ci]} sequence against the pool (${m.poolL})`, pad, y2, { size: 11.5, font: 'mono', color: C.muted, maxW: bw });
        const need = seq * S.seqs;
        const over = need > m.pool;
        const scale2 = bw / Math.max(m.pool, need);
        const by = y2 + (ratio != null ? 30 : 10), bh = 34;
        V.box(g, pad, by, m.pool * scale2, bh, { r: 6, fill: V.alpha(C.muted, 0.1), stroke: V.alpha(C.ink, 0.45), lw: 1.2 });
        for (let s = 0; s < S.seqs; s++) {
          const sx = pad + s * seq * scale2;
          const sw = seq * scale2;
          V.box(g, sx + 1, by + 4, Math.max(2, sw - 2), bh - 8, { r: 4, fill: V.alpha(over && (s + 1) * seq > m.pool ? C.red : s % 2 ? C.b : C.a, 0.75) });
          if (m.fixed && sw > 6) { const fw = Math.max(1, m.fixed * scale2); g.fillStyle = V.alpha(C.d, 0.95); g.fillRect(sx + 1, by + 4, Math.min(fw, sw - 2), bh - 8); }
        }
        if (over) V.line(g, pad + m.pool * scale2, by - 6, pad + m.pool * scale2, by + bh + 6, { color: C.red, lw: 2 });
        // limite do alocador: quantas sequências deste tamanho o vLLM admite pela contabilidade (tokens do log ÷ contexto)
        if (ratio != null) {
          const ax = pad + Math.min(ratio * seq, m.pool) * scale2;
          V.line(g, ax, by - 4, ax, by + bh + 4, { color: C.d, lw: 2.2, dash: [4, 3] });
          const lab = `allocator: ${ratioTxt(ratio)}`;
          g.font = V.font(11, 'mono', 700);
          const lw2 = g.measureText(lab).width;
          const tx = Math.max(pad, Math.min(ax - lw2 / 2, w - pad - lw2));
          V.box(g, tx - 4, by - 20, lw2 + 8, 15, { r: 4, fill: V.alpha(C.d, 0.22) });
          V.text(g, lab, tx, by - 9, { size: 11, font: 'mono', weight: 700, color: C.ink });
        }
        const pct = need / m.pool;
        const allocTxt = alloc == null ? 'allocator: not measured' : `allocator: ${nf(alloc)} (${ratioTxt(ratio)})`;
        let msg, msgCol;
        if (over) { msg = `✕ does not fit: ${nf(need / GiB, 2)} GiB for ${nf(m.pool / GiB, 2)} GiB`; msgCol = C.red; }
        else if (alloc != null && S.seqs > alloc) { msg = `⚠ fits by physical cost (${nf(need / GiB, 2)} of ${nf(m.pool / GiB, 2)} GiB), but the allocator only admits ${nf(alloc)} (${ratioTxt(ratio)})`; msgCol = C.d; }
        else { msg = `✓ ${nf(need / GiB, 2)} of ${nf(m.pool / GiB, 2)} GiB (${nf(pct * 100, 0)}%) · physical: ${nf(phys)} fit · ${allocTxt}`; msgCol = C.a; }
        let yy = by + bh + 20;
        yy = wrapText(g, msg, pad, yy, bw, 18, { size: 13, weight: 700, color: msgCol });
        yy = wrapText(g, m.fixed ? `amber = ${m.fixedL} (≈${fmtB(m.fixed)} per sequence)` : m.fixedL, pad, yy + 2, bw, 16, { size: 11.5, color: C.muted });
        yy = wrapText(g, m.headL, pad, yy, bw, 16, { size: 11.5, color: C.muted });
        if (n > m.maxLen) yy = wrapText(g, `⚠ above the MAX_MODEL_LEN of the recipe (${m.maxLenL}): a request of this size is rejected`, pad, yy, bw, 16, { size: 11.5, weight: 700, color: C.red });
        // 3) comparação
        const y3 = yy + 18;
        V.text(g, '3 · BYTES PER TOKEN OF THE SAME MODEL WITHOUT THE SAVINGS', pad, y3, { size: 11.5, font: 'mono', color: C.muted, maxW: bw });
        const all = [['this recipe', bpt, C.a]].concat(m.cmp.map((c, i) => [c[0], c[1], i ? C.c : C.d]));
        const mx = Math.max(...all.map((a) => a[1]));
        const labW = wide ? 260 : 0;
        const rowH = wide ? 26 : 38;
        all.forEach(([l, v, col], i) => {
          const y = y3 + 12 + i * rowH;
          if (wide) V.text(g, l, pad, y + 14, { size: 12, color: C.ink, maxW: labW - 10 });
          else V.text(g, l, pad, y + 10, { size: 11.5, color: C.ink, maxW: bw });
          const yb = wide ? y + 2 : y + 15;
          const bwid = Math.max(3, (v / mx) * (bw - labW - 110));
          V.box(g, pad + labW, yb, bwid, 16, { r: 4, fill: V.alpha(col, 0.8) });
          V.text(g, `${fmtB(v)}${i ? ` (${nf(v / bpt, 1)}×)` : ''}`, pad + labW + bwid + 8, yb + 13, { size: 12, font: 'mono', color: C.ink });
        });
        setH(fig, y3 + 12 + all.length * rowH + 14);
      });
    })();

    wrapControls();
  },
});

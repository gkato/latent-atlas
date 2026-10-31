LA.chapter({
  id: 'receita-glm',
  part: 9,
  order: 42,
  title: 'Rebuilding the GLM-5.3-Flash EXL3 recipe',
  navTitle: 'GLM recipe',
  subtitle: 'From the byte budget to the READY banner: every line of the launcher, the .env and vllm serve, with the reason behind each value.',
  minutes: 65,
  recipes: ['glm', 'tf'],
  tags: ['recipe', 'GLM-5.3-Flash', 'EXL3', 'launcher', 'vllm serve', 'DFlash2', 'TP=2', 'boot', 'overlays', 'TensorFold'],
  goals: [
    'Do the memory <b>leftover math</b> for a UMA node, separating the profiled pool from the fixed pool, and justify <code>GPU_MEM_UTIL=0.85</code> and the fixed 11 GiB pool',
    'Translate the KV pool into <b>block IDs</b> and explain why compact pages make 850k fit 1.45× in 11 GiB',
    'Explain every <code>vllm serve</code> flag, every <code>--speculative-config</code> field and the runtime policies (adaptive-k, fair v5, hybrid APC)',
    'Understand how the launcher guarantees <b>the same image, the same patches, the same environment and the same network</b> on both ranks, and how to verify the engine with logs, a canary and an acceptance gate',
    'Read the boot timeline, choose a profile per workload and say which stations of the map stay, swap or vanish when the engine becomes TensorFold',
  ],
  prereqs: ['engenharia-de-receita', 'tres-modelos', 'dflash-dspark'],
  summary: [
    'The recipe is leftover math: ~79.65 GiB of weights per rank, 3.3 GiB for the BF16 KDA copy, 11 GiB of fixed KV, workspaces and ~9 GiB of initialization, and the head still keeps ≥ 6.2 GiB free under load. With the fixed pool, <code>GPU_MEM_UTIL=0.85</code> no longer sizes the KV: it becomes just the free-memory floor the boot requires (0.85 × 121.69 ≈ 103.4 GiB).',
    'The KV pool is counted in block IDs: 1 ID = 3584 tokens × 11 layers × (656 + 33) B ≈ 25.9 MiB, and every group (MLA, kpool, 4 Mamba, drafter) pays in whole IDs. An 850k request costs ~536 IDs without compact pages and ~299 with them, which is why 11 GiB (434 IDs) holds 1.45× of 850k.',
    'Almost every flag comes from a constraint: EXL3 because BF16/FP8 don’t fit; <code>fp8</code> → <code>fp8_ds_mla</code> (656 B, with 64 RoPE zeros) because it’s the only sparse-MLA kernel on SM121; the fixed pool because InstantTensor eats KV; <code>TEMP_ROWS=32</code> because 4 seqs × (7+1) positions. Swapping the engine for TensorFold erases the constraints that came from vLLM (pad, 3584 pages, uniform decode) and brings others: only 2 ranks and up to 4 requests in v1.3.2 (3 Sparks since v1.4, 8 requests since v1.5) and output bit-for-bit equal to serial.',
    'The image is pinned by digest, patched fail-closed and stamped; 25 overlays (26 since #203) are reapplied in the same order on both ranks, and the network is pinned per rank (interface, HCA, GID). The wrong order fails; broken parity hangs NCCL.',
    'The launcher gives precedence to the caller’s exports (even empty ones), validates everything before stopping the pair, serializes the lifecycle with flock and aborts early if the worker dies.',
    'Boot: <code>restart</code> 259 → 122 s; from <code>docker run</code> to <code>/health</code>, 99 s, most of it in the weight stream. Then come the canary (automatic), the acceptance gate (manual) and A/B/A benchmarks.',
    'Policies and options are measured trade-offs: adaptive-k (+13–21% on prose, off by default), fair v5, v7 since #283 (neither starvation nor a 10–36× slower incumbent), hybrid APC aligned to 3584, dense FP8, KDA BF16, E3, cooperative MoE, grouped into profiles such as <code>.env.example</code> and <code>tp2-long-coding</code>.',
  ],
  quiz: [
    { q: 'Why does the recipe pass <code>--kv-cache-memory-bytes 11811160064</code> instead of letting vLLM profile the pool?', o: ['Because vLLM can’t profile memory on integrated GPUs', 'Because, with InstantTensor, the profiled pool (~12.4 GiB) fell below the 13.56 GiB of an 850k request and the boot was refused; a fixed 11 GiB with compact pages holds 1.45× of 850k', 'Because 11 GiB is the most the sparse-MLA kernel can address', 'Because a fixed pool makes prefill faster'], a: 1, e: 'The direct-I/O loader leaves ~4.4–5.6 GiB less for the profiled KV. Fixing the pool solves that, and the drafter’s compact pages make 11 GiB enough (1,233,779 tokens). The 3 GiB saved relative to the previous 14 GiB pay for the BF16 KDA copy.' },
    { q: 'You run <code>MM_IMAGE_TOKENS= ./start.sh</code> (explicit empty). What happens?', o: ['The default 2048 applies, because empty counts as unset', 'Validation rejects the value before stopping the pair', 'The empty value survives <code>${MM_IMAGE_TOKENS-2048}</code>, the flag is omitted and vLLM falls back to the checkpoint’s 8,000 tokens per image, above the 7,168 encoder cache', 'The .env wins and applies 2048'], a: 2, e: 'The caller’s export, even when empty, beats the .env. With the <code>-</code> operator, without the colon, the empty value is preserved and means “stock”. Here, “stock” is dangerous: 8,000 tokens per image exceed the MNBT/encoder cache.' },
    { q: 'With <code>MAX_NUM_SEQS=4</code> and <code>DFLASH_TOKENS=7</code>, where do <code>EXL3_TEMP_ROWS_FUSED=32</code> and the capture sizes 8/16/24/32 come from?', o: ['They’re empirical values, unrelated to each other', 'Each sequence verifies 7+1 = 8 positions per step; with 1–4 seqs, decode has 8, 16, 24 or 32 tokens, and 4 × 8 = 32 rows per expert fit in the fused MoE buffer', 'They come from the number of MLA layers (11) multiplied by 3', 'They’re powers of 2 required by NCCL'], a: 1, e: 'The chain flows down from speculation: 8-position block → N·8 tokens per step → those sizes need a CUDA graph → the exl3_moe temporary buffer needs ≥ N·8 rows so that decode is a single graph-safe launch.' },
    { q: 'Why does the launcher compare head and worker by the hash of <code>RootFS.Layers</code>, and not by the image’s <code>.Id</code>?', o: ['Because .Id changes on every docker run', 'Because .Id differs between overlay2 and containerd, and docker save|load drops the RepoDigests; the layers’ diff IDs are the same on both sides', 'Because RootFS.Layers includes the recipe stamp', 'Because GHCR doesn’t publish .Id'], a: 1, e: 'No other field survives every path (pull, local build, save|load). Before the image key, the whole image was re-sent to the worker on every run.' },
    { q: 'With <code>GLM53_ADAPTIVE_K=ema</code> on, rank 1 comes up without overlay no. 18 (adaptive-k). What happens?', o: ['Nothing: overlays only affect rank 0', 'vLLM detects it and reapplies the patch on its own', 'Rank 0 captures decode graphs for 3, 5 and 8 positions and rank 1 only for 8; the two launch different collectives, and NCCL hangs after PIECEWISE', 'The worker falls back to eager mode and gets slower'], a: 2, e: 'At TP=2 both ranks run the same sequence of kernels and collectives. The README sums it up: a rank that captures different CUDA graphs hangs NCCL. The exact failure is interpretation; with adaptive-k off (the default), both would verify 8 positions and the gap would go unnoticed, which makes the bug even sneakier.' },
    { q: 'The recipe fixes the pool with <code>--kv-cache-memory-bytes</code>. You raise <code>GPU_MEM_UTIL</code> from 0.85 to 0.88. What changes?', o: ['The KV pool grows by ~3.6 GiB (3 × 1.2 GiB)', 'Nothing at runtime: usage doesn’t depend on u. What changes is the free-memory floor the boot requires, 0.88 × 121.69 ≈ 107.1 GiB, right at the edge of what the worker has free at the check (0.88 booted in only 1 of 4 attempts)', 'The host gains ~3.6 GiB of headroom', 'vLLM goes back to profiling the pool'], a: 1, e: 'With the fixed pool, vLLM ignores u when sizing the KV. Util is left over as a free-memory requirement: vLLM’s startup check and the launcher’s preflight compare MemAvailable with u × M. The rule that each 0.01 is worth ~1.2 GiB belongs to the profiled regime, where u decides how much is left for the KV.' },
    { q: 'Which of these vLLM-recipe decisions <b>vanishes</b> in the TensorFold version, because it came from the engine and not from the hardware or the model?', o: ['Quantizing only the routed experts to 4 bpw EXL3', 'TP=2, with one rank per Spark', 'The 512→576 pad of the MLA layers’ KV record', 'Speculating with DFlash2, because decode is bandwidth-bound'], a: 2, e: 'The pad exists because SM12x’s only sparse-MLA kernel has a fixed 576 geometry. TensorFold reads the 512 latent with its own kernels, with no pad: 528 B per token and layer instead of 656 B. The other three come from physical or model constraints (164 GiB that don’t fit in BF16, two 128 GB nodes, bandwidth-bound decode) and stay on both engines.' },
  ],
  mount(root, LA) {
    const V = LA.viz, UI = LA.ui;

    /* ---------------------------------------------------------- helpers */
    const fcache = {};
    const FN = (s, f, w) => { const k = s + '|' + (f || 'sans') + '|' + (w || 400); return fcache[k] || (fcache[k] = V.font(s, f || 'sans', w || 400)); };
    const tw = (g, s, size, font, wt) => { g.font = FN(size || 12, font, wt); return g.measureText(String(s)).width; };
    const wrap = (g, s, maxW, size, font, wt) => {
      g.font = FN(size || 12, font, wt);
      const out = []; let cur = '';
      String(s).split(' ').forEach((wd) => { const tt = cur ? cur + ' ' + wd : wd; if (cur && g.measureText(tt).width > maxW) { out.push(cur); cur = wd; } else cur = tt; });
      if (cur) out.push(cur);
      return out;
    };
    const para = (g, s, x, y, maxW, o) => {
      o = o || {}; const size = o.size || 12, lh = o.lh || Math.round(size * 1.4);
      const L = wrap(g, s, maxW, size, o.font, o.weight);
      L.forEach((l, i) => V.text(g, l, x, y + i * lh, Object.assign({}, o, { size })));
      return L.length * lh;
    };
    const br = (x, d) => (+x).toFixed(d == null ? 1 : d);
    const near = (cur, tgt, dt, k) => (cur == null || !isFinite(cur) ? tgt : cur + (tgt - cur) * (1 - Math.exp(-dt * (k || 9))));
    const chip = (g, x, y, label, o) => {
      o = o || {}; const size = o.size || 11, px = o.px == null ? 7 : o.px, hh = o.h || size + 9;
      const wd = tw(g, label, size, o.font || 'mono', o.weight || 500) + px * 2;
      if (o.dash) g.setLineDash(o.dash);
      V.box(g, x, y, wd, hh, { r: o.r == null ? 7 : o.r, fill: o.fill, stroke: o.stroke, lw: o.lw || 1 });
      g.setLineDash([]);
      V.text(g, label, x + px, y + hh / 2 + 0.5, { size, font: o.font || 'mono', weight: o.weight || 500, color: o.color, base: 'middle' });
      return wd;
    };
    const onFill = (C) => (C.dark ? '#0b1a13' : '#ffffff');
    /** adjusts the stage height to the width (mobile needs more height) */
    const fitH = (fig, wide, narrowH, w) => { const st = fig.querySelector('.la-fig-stage'); const want = w < 600 ? narrowH : wide; if (st && Math.abs(st.clientHeight - want) > 2) st.style.height = want + 'px'; };

    /* ============================================ F1 · memory budget */
    const GI = 2 ** 30, PER_ID = 27163136, MLA_850 = Math.ceil(850000 / 3584), REQ_ON = 299, REQ_OFF = 536;
    const fmtInt = (n) => Math.round(n).toString().replace(/\B(?=(\d{3})+(?!\d))/g, ',');
    (function () {
      const fig = root.querySelector('#fig-orcamento');
      const S = { mode: 'fixo', util: 0.85, kv: 11, kda: true, rs: true, compact: true };
      const A = {};
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Regime', options: [{ v: 'fixo', label: 'fixed pool (recipe)' }, { v: 'perf', label: 'profiled pool' }], value: 'fixo', onChange: (v) => { S.mode = v; } });
      UI.slider(ctl, { label: 'GPU_MEM_UTIL', min: 0.75, max: 0.9, step: 0.005, value: 0.85, fmt: (v) => br(v, 3), onInput: (v) => { S.util = v; } });
      UI.seg(ctl, { label: 'Fixed pool', options: [{ v: 11, label: '11 GiB' }, { v: 14, label: '14 GiB' }, { v: 15, label: '15 GiB' }], value: 11, onChange: (v) => { S.kv = v; } });
      UI.toggle(ctl, { label: 'BF16 KDA copy', value: true, onChange: (v) => { S.kda = v; } });
      UI.toggle(ctl, { label: 'indexer rightsize', value: true, onChange: (v) => { S.rs = v; } });
      UI.toggle(ctl, { label: 'compact pages', value: true, onChange: (v) => { S.compact = v; } });
      const M = 121.69;
      const IDX_STOCK = (850000 * 40 * 132) / GI;
      const IDX_RS = (4 * Math.ceil(850007 / 4) * 132) / GI;
      const E3 = 560 / 1024, WS_RS = IDX_RS + E3, W = 79.65, KDA = 3.29, INIT = 9;
      // interpretation: "rest" calibrated to reproduce the >= 6.2 GiB free measured with the default
      const REST_FIX = M - (W + KDA + 11 + WS_RS + INIT) - 6.2;
      // interpretation: linear estimate of the profiled pool, anchored on the 12.4 GiB measured at 0.85 (#204, without the KDA copy)
      const REST_PROF = 0.85 * M - W - WS_RS - 12.4;
      const MEAS = { '11|1': '1,233,779 tokens measured', '14|1': '1,572,073 measured', '14|0': '876,958 measured', '11|0': '~690k measured: boot refuses 850k', '15|0': '1.11× measured' };
      V.fig(fig, (g, w, h, t, dt) => {
        if (w < 160) return;
        const C = V.C();
        const narrow = w < 560;
        const pad = 16, bx = pad, bw = w - 2 * pad;
        const fixed = S.mode === 'fixo';
        const ws = (S.rs ? IDX_RS : IDX_STOCK) + E3, kda = S.kda ? KDA : 0, uM = S.util * M;
        const tgt = { w: W, kda, ws, uM };
        if (fixed) { Object.assign(tgt, { kv: S.kv, init: INIT, rest: REST_FIX }); tgt.free = M - (W + kda + S.kv + ws + INIT + REST_FIX); }
        else { Object.assign(tgt, { init: 0, rest: REST_PROF }); tgt.kv = uM - W - kda - ws - REST_PROF; tgt.free = M - uM; }
        Object.keys(tgt).forEach((k) => { A[k] = near(A[k], tgt[k], dt, 8); });
        const X = (gib) => bx + (Math.max(0, Math.min(gib, M)) / M) * bw;
        const title = fixed ? (narrow ? 'FIXED POOL · USAGE (INDEPENDENT OF u)' : 'FIXED POOL (THE RECIPE) · RUNTIME USAGE, WHICH DOES NOT DEPEND ON u') : (narrow ? 'PROFILED POOL · u×M SIZES THE KV' : 'PROFILED POOL (vLLM STOCK) · u×M SIZES THE KV');
        V.text(g, title, bx, 18, { size: 11, font: 'mono', weight: 700, color: fixed ? C.a : C.d, maxW: bw });
        const by = 30, bh = narrow ? 38 : 44;
        V.box(g, bx, by, bw, bh, { r: 9, fill: V.alpha(C.muted, 0.08), stroke: V.alpha(C.muted, 0.35) });
        const segs = fixed ? [
          { k: 'w', lab: 'weights', leg: 'weights per rank', sh: 'weights/rank', col: C.d },
          { k: 'kda', lab: 'KDA', leg: 'BF16 KDA copy', sh: 'KDA BF16', col: C.f },
          { k: 'kv', lab: 'KV', leg: 'fixed KV pool', sh: 'KV pool', col: C.a },
          { k: 'ws', lab: 'ws', leg: 'indexer + E3 scratch', sh: 'workspaces', col: C.c },
          { k: 'init', lab: 'init', leg: 'vLLM init (~9 GiB, README)', sh: 'vLLM init', col: C.b },
          { k: 'rest', lab: 'rest', leg: 'activations, graphs, drafter, OS (interp.)', sh: 'rest', col: C.b, hatch: true },
          { k: 'free', lab: 'free', leg: 'free (MemAvailable)', sh: 'free', col: C.e, free: true },
        ] : [
          { k: 'w', lab: 'weights', leg: 'weights per rank', sh: 'weights/rank', col: C.d },
          { k: 'kda', lab: 'KDA', leg: 'BF16 KDA copy', sh: 'KDA BF16', col: C.f },
          { k: 'ws', lab: 'ws', leg: 'indexer + E3 scratch', sh: 'workspaces', col: C.c },
          { k: 'rest', lab: 'rest', leg: 'activations, graphs, init (profile; interp.)', sh: 'rest', col: C.b, hatch: true },
          { k: 'kv', lab: 'KV', leg: 'profiled KV = u×M − the rest', sh: 'profiled KV', col: C.a },
          { k: 'free', lab: 'host', leg: 'host + whatever falls outside the budget', sh: 'host', col: C.muted, free: true },
        ];
        let acc = 0;
        segs.forEach((s) => {
          const v = Math.max(0, A[s.k]); const x0 = X(acc), x1 = X(acc + v); acc += v;
          if (x1 - x0 < 0.6) return;
          g.save(); V.rr(g, x0, by + 1, x1 - x0, bh - 2, 5); g.clip();
          g.fillStyle = V.alpha(s.col, s.hatch ? 0.2 : s.free ? 0.22 : 0.82); g.fillRect(x0, by, x1 - x0, bh);
          if (s.hatch) { g.strokeStyle = V.alpha(s.col, 0.8); g.lineWidth = 1.2; for (let q = x0 - bh; q < x1; q += 7) { g.beginPath(); g.moveTo(q, by + bh); g.lineTo(q + bh, by); g.stroke(); } }
          g.restore();
          if (x1 - x0 > tw(g, s.lab, 11, 'sans', 700) + 8) V.text(g, s.lab, (x0 + x1) / 2, by + bh / 2 + 1, { size: 11, weight: 700, align: 'center', base: 'middle', color: s.hatch || s.free ? C.ink : onFill(C) });
        });
        const over = fixed ? tgt.free < 0 : tgt.kv < 0;
        if (over) { const x0 = fixed ? X(M) - 6 : X(uM); g.fillStyle = V.alpha(C.red, 0.28 + 0.18 * Math.sin(t * 5)); g.fillRect(Math.min(x0, X(M) - 6), by - 4, 6, bh + 8); }
        if (!fixed) V.line(g, X(A.uM), by - 7, X(A.uM), by + bh + 7, { color: C.ink, lw: 2, dash: [4, 3] });
        for (let q = 0; q <= 120; q += narrow ? 40 : 20) { const xq = X(q); V.line(g, xq, by + bh + 2, xq, by + bh + 6, { color: V.alpha(C.muted, 0.8), lw: 1 }); V.text(g, q + (q === 0 ? ' GiB' : ''), xq, by + bh + 17, { size: 11, font: 'mono', align: q === 0 ? 'left' : 'center', color: C.muted }); }
        const sy = by + bh + 34;
        if (fixed) {
          V.text(g, 'usage ≈ ' + br(M - Math.max(0, A.free)) + ' GiB', bx, sy, { size: 11.5, font: 'mono', color: C.ink, weight: 600 });
          V.text(g, 'free ≈ ' + br(Math.max(0, A.free)) + ' GiB', bx + bw, sy, { size: 11.5, font: 'mono', color: C.e, align: 'right', weight: 600 });
        } else {
          V.text(g, 'vLLM · u×M = ' + br(A.uM) + ' GiB', bx, sy, { size: 11.5, font: 'mono', color: C.ink, weight: 600 });
          V.text(g, 'host · ' + br(M - A.uM) + ' GiB', bx + bw, sy, { size: 11.5, font: 'mono', color: C.muted, align: 'right' });
        }
        // boot ruler
        const gy0 = sy + 22;
        V.text(g, narrow ? 'BOOT · NEEDS FREE ≥ u×M = ' + br(uM) + ' GiB' : 'BOOT · THE STARTUP CHECK REQUIRES FREE MEMORY ≥ u×M = ' + br(uM) + ' GiB', bx, gy0, { size: 11, font: 'mono', weight: 700, color: C.d, maxW: bw });
        const gy = gy0 + 9, gh = 18;
        V.box(g, bx, gy, bw, gh, { r: 6, fill: V.alpha(C.muted, 0.08), stroke: V.alpha(C.muted, 0.3) });
        V.box(g, bx, gy, Math.max(4, X(A.uM) - bx), gh, { r: 6, fill: V.alpha(C.d, 0.35) });
        const b0 = X(105.8), b1 = X(107.1);
        g.fillStyle = V.alpha(C.e, 0.5); g.fillRect(b0, gy - 3, Math.max(3, b1 - b0), gh + 6);
        V.line(g, X(A.uM), gy - 5, X(A.uM), gy + gh + 5, { color: C.ink, lw: 2, dash: [4, 3] });
        let gv;
        if (uM < 105.5) gv = [C.e, '✓ passes: ' + br(105.8 - uM) + ' GiB below the free memory measured by the worker’s check (green band: 105.8–107.1 GiB)'];
        else if (uM < 106.5) gv = [C.d, '⚠ at the edge: 0.87 requires 105.9 GiB and fails by < 1 GiB on nodes with resident services'];
        else if (uM <= 107.15) gv = [C.d, '⚠ inside the measured band: 0.88 (107.09 GiB) booted in 1 of 4 attempts'];
        else gv = [C.red, '✕ above the free memory measured at the check: startup refuses (interpretation, extrapolating the band)'];
        let iy = gy + gh + 20;
        iy += para(g, gv[1], bx, iy, bw, { size: 12, color: gv[0], lh: 16, weight: 600 }) + 8;
        // legend (2 columns)
        const items = segs.map((s) => [s.col, narrow ? s.sh : s.leg, tgt[s.k], s.hatch]);
        const colW = bw / 2, rowH = 20;
        items.forEach(([col, lab, v, hatch], i) => {
          const cx = bx + (i % 2) * colW, cy = iy + Math.floor(i / 2) * rowH;
          V.box(g, cx, cy - 9, 10, 10, { r: 3, fill: hatch ? V.alpha(col, 0.35) : col });
          const vs = br(Math.max(0, v)) + ' GiB', vw = tw(g, vs, 11.5, 'mono');
          V.text(g, lab, cx + 16, cy, { size: 11.5, color: C.ink, maxW: colW - 12 - vw - 22 });
          V.text(g, vs, cx + colW - 12, cy, { size: 11.5, font: 'mono', color: C.muted, align: 'right' });
        });
        iy += Math.ceil(items.length / 2) * rowH + 8;
        // capacity in block IDs
        const kvG = Math.max(0, fixed ? S.kv : tgt.kv), ids = Math.floor((kvG * GI) / PER_ID), per = S.compact ? REQ_ON : REQ_OFF, x = ids / per;
        let capT = (fixed ? 'KV ' + br(kvG, kvG % 1 ? 1 : 0) : 'profiled KV ≈ ' + br(kvG)) + ' GiB = ' + ids + ' IDs of 25.9 MiB; 850k costs ' + per + ' IDs → ' + br(x, 2) + '×';
        if (fixed && MEAS[S.kv + '|' + (S.compact ? 1 : 0)]) capT += ' (' + MEAS[S.kv + '|' + (S.compact ? 1 : 0)] + ').';
        else if (x < 1) capT += ': an 850k request doesn’t fit, and boot fails while allocating the KV with the weights already loaded.';
        else capT += ' ≈ ' + fmtInt(x * 850000) + ' tokens.';
        const lines = [[x < 1 ? C.red : C.a, capT]];
        lines.push([C.a, 'Launcher preflight: MemAvailable ≥ u×M + 2 GiB = ' + br(uM + 2) + ' GiB before docker run; on restart, it waits for u×M + 1.5 GiB.']);
        let warn;
        if (fixed) {
          if (tgt.free < 2.5) warn = [C.red, '⚠ Little headroom (linear estimate): with 15 GiB fixed, only 0.8–2.2 GiB free was measured under load, too little for long prefills.'];
          else if (!S.rs) warn = [C.d, '⚠ Stock indexer workspace: 850k × 40 × 132 B ≈ 4.2 GiB locked up just for the prefill gather.'];
          else if (Math.abs(S.util - 0.85) < 0.001 && S.kv === 11 && S.kda && S.compact) warn = [C.e, '✓ Recipe default: the head keeps ≥ 6.2 GiB free under load (measured). Move u: none of the blocks above budge.'];
          else if (!S.kda) warn = [C.b, 'Without the BF16 KDA copy: +3.3 GiB free, but prefill gives up its ~12% TTFT gain.'];
          else warn = [C.b, 'With a fixed pool, u doesn’t touch any usage block: only the boot’s free-memory floor.'];
        } else {
          if (S.util >= 0.869) warn = [C.red, '⚠ In this regime, 0.87 with a 256k prefill and MemAvailable at zero took down a head (2026-09-06).'];
          else warn = [C.b, 'Each 0.01 of u is ~1.2 GiB of KV taken from the host. With no cap, the head got down to ~1.5 GiB of MemAvailable at 850k (README).'];
        }
        lines.push(warn);
        lines.forEach(([col, s]) => { V.dot(g, bx + 4, iy - 4, 3.5, col); iy += para(g, s, bx + 14, iy, bw - 14, { size: 12, color: C.ink, lh: 16 }) + 5; });
        fitH(fig, Math.ceil(iy + 4), Math.ceil(iy + 4), w);
      });
    })();

    /* ============================================ F1b · the pool in block IDs */
    (function () {
      const fig = root.querySelector('#fig-ids');
      const S = { gib: 11, compact: true, view: 'req', t0: null };
      const reset = () => { S.t0 = null; };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'View', options: [{ v: 'req', label: '850k requests' }, { v: 'seg', label: 'cached segment' }], value: 'req', onChange: (v) => { S.view = v; reset(); } });
      UI.slider(ctl, { label: 'KV pool', min: 8, max: 16, step: 0.5, value: 11, fmt: (v) => br(v, 1) + ' GiB', onInput: (v) => { S.gib = v; reset(); } });
      UI.toggle(ctl, { label: 'drafter compact pages', value: true, onChange: (v) => { S.compact = v; reset(); } });
      const MAXIDS = Math.floor((16 * GI) / PER_ID);
      V.fig(fig, (g, w, h, t) => {
        if (w < 160) return;
        const C = V.C(); const narrow = w < 600; const pad = 14, gw = w - 2 * pad;
        if (S.t0 == null) S.t0 = t;
        const el = t - S.t0;
        let y = 18;
        V.text(g, narrow ? '1 ID = 3584 tokens × 11 layers' : '1 ID = 3584 tokens × 11 layers × (656 B MLA + 33 B indexer) = 27,163,136 B ≈ 25.9 MiB', pad, y, { size: 11, font: 'mono', weight: 700, color: C.a, maxW: gw });
        if (narrow) { y += 15; V.text(g, '× (656 + 33) B = 27,163,136 B ≈ 25.9 MiB', pad, y, { size: 11, font: 'mono', weight: 700, color: C.a, maxW: gw }); }
        y += 14;
        if (S.view === 'req') {
          const ids = Math.floor((S.gib * GI) / PER_ID), per = S.compact ? REQ_ON : REQ_OFF, other = per - MLA_850;
          const cols = narrow ? 30 : 48, cs = gw / cols, rows = Math.ceil(MAXIDS / cols);
          const cyc = 7, pe = el % cyc, shown = Math.min(ids, Math.floor(pe * 520));
          for (let k = 0; k < MAXIDS; k++) {
            const cx = pad + (k % cols) * cs, cy = y + Math.floor(k / cols) * cs;
            if (k >= ids) { V.box(g, cx + 1.5, cy + 1.5, cs - 3, cs - 3, { r: 2, fill: 'transparent', stroke: V.alpha(C.muted, 0.18), lw: 1 }); continue; }
            const r = Math.floor(k / per), j = k % per, full = (r + 1) * per <= ids;
            if (k >= shown) { V.box(g, cx + 1, cy + 1, cs - 2, cs - 2, { r: 2, fill: V.alpha(C.muted, 0.12) }); continue; }
            const isMla = j < MLA_850, col = isMla ? C.a : C.b;
            if (full) V.box(g, cx + 1, cy + 1, cs - 2, cs - 2, { r: 2, fill: V.alpha(col, r % 2 ? 0.62 : 0.9) });
            else { g.setLineDash([2, 2]); V.box(g, cx + 1.5, cy + 1.5, cs - 3, cs - 3, { r: 2, fill: V.alpha(col, 0.42), stroke: V.alpha(C.red, 0.6 + 0.3 * Math.sin(t * 4)), lw: 1.2 }); g.setLineDash([]); }
          }
          V.text(g, br(S.gib, 1) + ' GiB pool = ' + ids + ' IDs · empty outlines: up to 16 GiB (' + MAXIDS + ')', pad + gw, y + rows * cs + 14, { size: 11, font: 'mono', color: C.muted, align: 'right', maxW: gw });
          let yy = y + rows * cs + 40;
          const x = ids / per;
          // bar for one request
          V.text(g, 'ONE 850k REQUEST = ' + per + ' IDs = ' + br((per * PER_ID) / GI, 2) + ' GiB', pad, yy, { size: 11, font: 'mono', weight: 700, color: C.ink, maxW: gw });
          yy += 8;
          const sc = gw / REQ_OFF;
          V.box(g, pad, yy, MLA_850 * sc, 16, { r: 4, fill: V.alpha(C.a, 0.85) });
          V.box(g, pad + MLA_850 * sc, yy, other * sc, 16, { r: 4, fill: V.alpha(C.b, 0.85) });
          if (S.compact) { g.setLineDash([3, 3]); V.box(g, pad + per * sc, yy, (REQ_OFF - per) * sc, 16, { r: 4, fill: 'transparent', stroke: V.alpha(C.muted, 0.6), lw: 1 }); g.setLineDash([]); }
          if (MLA_850 * sc > 90) V.text(g, MLA_850 + ' MLA', pad + (MLA_850 * sc) / 2, yy + 9, { size: 11, weight: 700, align: 'center', base: 'middle', color: onFill(C) });
          if (other * sc > 60) V.text(g, other + ' others', pad + MLA_850 * sc + (other * sc) / 2, yy + 9, { size: 11, weight: 700, align: 'center', base: 'middle', color: onFill(C) });
          if (S.compact && (REQ_OFF - per) * sc > 110) V.text(g, '−' + (REQ_OFF - per) + ' (compact)', pad + per * sc + ((REQ_OFF - per) * sc) / 2, yy + 9, { size: 11, align: 'center', base: 'middle', color: C.muted });
          yy += 34;
          const txt = [
            [C.a, 'Mint: ' + MLA_850 + ' MLA IDs (⌈850,000 / 3584⌉), the only ones that grow with context.'],
            [C.b, 'Lavender: ~' + other + ' IDs for Mamba ×4, kpool and drafter, fixed per request' + (S.compact ? ': compact pages (896-token blocks) cut ~' + (REQ_OFF - REQ_ON) + ' IDs, all of them from the drafter.' : ': with 64-token blocks, the drafter accounts for most of them.')],
            [x < 1 ? C.red : C.e, ids + ' / ' + per + ' = ' + br(x, 2) + '× of 850k ≈ ' + fmtInt(x * 850000) + ' tokens' + (x < 1 ? ': not even one full request fits, and boot refuses.' : '.') + ' Per-request totals: our own calculation from the measured pools.'],
          ];
          txt.forEach(([col, s]) => { V.dot(g, pad + 4, yy - 4, 3.5, col); yy += para(g, s, pad + 14, yy, gw - 14, { size: 12, color: C.ink, lh: 16 }) + 4; });
          fitH(fig, Math.ceil(yy + 6), Math.ceil(yy + 6), w);
        } else {
          y += 6;
          V.text(g, narrow ? '1M REQUEST · 2026-08-31 BOOT' : 'ONE 1M REQUEST · 2026-08-31 BOOT (MNBT 2048, DRAFTER IN 64-TOKEN BLOCKS)', pad, y, { size: 11, font: 'mono', weight: 700, color: C.ink, maxW: gw });
          y += 8;
          const parts = [[280, C.a, 'MLA 280'], [1, C.c, ''], [36, C.f, 'Mamba 4×9'], [97, C.b, 'drafter 97']];
          const sc = gw / 414; let xx = pad;
          parts.forEach(([n, col, lab], k) => { const ww = n * sc; V.box(g, xx, y, Math.max(2, ww - 1), 20, { r: 4, fill: V.alpha(col, 0.85) }); if (lab && ww > tw(g, lab, 11, 'sans', 700) + 8) V.text(g, lab, xx + ww / 2, y + 11, { size: 11, weight: 700, align: 'center', base: 'middle', color: onFill(C) }); xx += ww; });
          V.text(g, '= 414 IDs · kpool: 1', pad + gw, y + 36, { size: 11, font: 'mono', color: C.muted, align: 'right' });
          y += 60;
          V.text(g, narrow ? 'CACHED 3584-TOKEN SEGMENT = 38 IDs' : 'ONE 3584-TOKEN SEGMENT IN THE PREFIX CACHE = 38 IDs', pad, y, { size: 11, font: 'mono', weight: 700, color: C.ink, maxW: gw });
          y += 10;
          const segs = [[1, C.a, 'MLA'], [4, C.f, 'Mamba'], [33, C.b, 'drafter']];
          const cols = narrow ? 19 : 38, cs = Math.min(narrow ? 17 : 20, gw / cols);
          let k = 0; const hot = Math.floor(el * 8) % 44;
          segs.forEach(([n, col]) => { for (let q = 0; q < n; q++, k++) { const cx = pad + (k % cols) * cs, cy = y + Math.floor(k / cols) * cs; V.box(g, cx + 1, cy + 1, cs - 2, cs - 2, { r: 3, fill: V.alpha(col, k === hot ? 1 : 0.75) }); if (k === hot) V.glow(g, cx + cs / 2, cy + cs / 2, cs, col, 0.8); } });
          y += Math.ceil(38 / cols) * cs + 12;
          let lx = pad;
          segs.forEach(([n, col, lab]) => { V.box(g, lx, y - 9, 10, 10, { r: 3, fill: col }); const s = lab + ' ' + n; V.text(g, s, lx + 15, y, { size: 11.5, color: C.ink }); lx += tw(g, s, 11.5) + 34; });
          y += 22;
          const txt = [
            [C.b, 'The drafter used 33 of the 38 IDs in each segment, and its blocks went to the tail of the LRU, pushing MLA and Mamba toward eviction. Overlay no. 9 gives the drafter’s group its own retention (with SWA=0, only the reachable boundaries).'],
            [C.a, 'With 642 free IDs, the log estimated ~57k tokens of cacheable conversation with nothing running: 16 segments × 3584. vLLM’s “GPU KV cache size” is something else: concurrency × max_model_len.'],
            [C.muted, S.compact ? 'With compact pages the drafter uses 896-token blocks and the lookup ends at the boundary, so a segment costs fewer IDs; the log above is from the 64-token layout.' : 'Without compact pages, this is the cost per segment: 64-token blocks in the drafter.'],
          ];
          txt.forEach(([col, s]) => { V.dot(g, pad + 4, y - 4, 3.5, col); y += para(g, s, pad + 14, y, gw - 14, { size: 12, color: C.ink, lh: 16 }) + 4; });
          fitH(fig, Math.ceil(y + 6), Math.ceil(y + 6), w);
        }
      });
    })();

    /* ============================================ F2 · decision map */
    (function () {
      const fig = root.querySelector('#fig-decisoes');
      const ROWS = [
        { name: 'WEIGHTS LINE', c: 'd', st: [
          ['BF16/FP8', 'don’t fit', 'BF16 takes ~642.6 GB and the official FP8 328.3 GB: neither fits in 2 × 128 GB of unified memory.'],
          ['EXL3 4 bpw', 'experts', 'Only the routed experts (~97% of the bytes) in EXL3 K4: 175.6 GB and a KLD of 0.0246, the same as the official FP8.'],
          ['method', 'exl3', 'vLLM doesn’t know EXL3: the overlay registers the method and the launcher passes --quantization exl3 (never marlin).'],
          ['kernel', 'sm_121a', 'The ExLlamaV3 CUDA extension (pin c5d9c657) is compiled for sm_121a with the E2, E3 and thin-decode kernels.'],
          ['image', 'custom', 'Compiling inside the image requires a custom Dockerfile, a digest pin and a recipe stamp to detect drift.'],
        ] },
        { name: 'SPECULATION LINE', c: 'c', st: [
          ['decode', 'mem-bound', 'Each decode token reads all the active weights: the bottleneck is memory bandwidth, not compute.'],
          ['DFlash2', 'k = 7', 'A block-diffusion drafter proposes 7 tokens; the target verifies all of them in one step (acceptance of 0.96 on structured text).'],
          ['8 positions', 'per seq', 'Each decoding sequence becomes 1 anchor + 7 drafts = 8 positions verified per step.'],
          ['graphs', '8·16·24·32', 'With up to 4 sequences, decode steps have 8, 16, 24 or 32 tokens, and those sizes need a CUDA graph.'],
          ['TEMP_ROWS', '= 32', 'The fused MoE has a buffer for 32 rows per expert: 4 × 8 fit, and decode runs in a single graph-safe launch.'],
        ] },
        { name: 'DISTRIBUTED LINE', c: 'e', st: [
          ['2 nodes', '1 GPU each', 'Two GB10s, one GPU each, linked by a ConnectX-7 cable over RoCEv2.'],
          ['TP = 2', 'mp', 'Tensor parallel of degree 2, multiprocessing executor (no Ray), --headless worker with no API.'],
          ['same', 'env', 'The GLM53_* policy knobs are forwarded identically to both ranks.'],
          ['same', 'order', 'GLM53_OVERLAY_ORDER applies the 25 overlays (26 since #203) with the same bytes and in the same order in both containers.'],
          ['same', 'graphs', 'A rank that captures different CUDA graphs hangs NCCL after PIECEWISE.'],
        ] },
        { name: 'MEMORY LINE (UMA)', c: 'a', st: [
          ['CPU + GPU', '128 GB', 'Unified memory: whatever vLLM takes comes out of the OS, the page cache and the vision encoder.'],
          ['util', '0.85', 'With the fixed pool, util no longer sizes the KV: it’s the boot’s free-memory floor (0.85 asks for 103.4 GiB; 0.88 booted in 1 of 4 attempts). In the earlier profiled regime, each 0.01 was ~1.2 GiB and 0.87 took down a head.'],
          ['Instant', 'Tensor', 'The direct-I/O loader loads 164 GiB in ~65–70 s (versus ~290–300 s), but leaves ~4.4–5.6 GiB less for the KV.'],
          ['fixed KV', '11 GiB', 'Profiled, the pool would sit at ~12.4 GiB, below the 13.56 GiB (536 block IDs) of an 850k request: that’s why it’s fixed.'],
          ['compact', 'pages', '896-token blocks in the drafter cut the 850k request from ~536 to ~299 IDs: 11 GiB (434 IDs) holds 1,233,779 tokens, 1.45×.'],
        ] },
        { name: 'ATTENTION LINE', c: 'b', st: [
          ['NoPE MLA', '+ SM121', 'The 11 MLA layers have no RoPE (qk_rope_head_dim=0), and the GB10 only has the FLASHINFER_MLA_SPARSE_SM120 kernel.'],
          ['1 sparse', 'kernel', 'That kernel only accepts the packed fp8_ds_mla layout: BF16 or NVFP4 KV have no sparse kernel on SM12x.'],
          ['KV dtype', 'fp8', '--kv-cache-dtype fp8 becomes the fp8_ds_mla record in the MLA layers.'],
          ['pad', '512→576', 'The overlay appends 64 RoPE zeros to the query and to k_pe. It’s exact, because zeros don’t change the QK product.'],
          ['656 B', '/token', '512 B FP8 + 16 B of scales + 128 B of RoPE (zeros): 24% more than the 528 B actually needed.'],
        ] },
      ];
      const T = [
        { a: [0, 3], b: [1, 4], d: 'TEMP_ROWS is a parameter of the EXL3 fused kernel, but its value comes from speculation: 4 seqs × 8 positions = 32.' },
        { a: [1, 3], b: [2, 4], d: 'The capture sizes must be identical on both ranks. If the graphs diverge, NCCL hangs.' },
        { a: [3, 3], b: [4, 4], d: 'The 11 GiB pool only makes sense in block IDs: 1 ID = 3584 tokens × 11 layers × (656 + 33) B ≈ 25.9 MiB, and the 656 B record comes from attention.' },
      ];
      const steps = [
        { t: 'Weights line', d: 'The model fits in neither BF16 nor FP8. Quantizing only the routed experts to EXL3 solves the bytes, but forces you to register a new quantization method and compile kernels for SM121: you have to maintain your own image.' },
        { t: 'Speculation line', d: 'Decode is bandwidth-bound. DFlash2 proposes 7 tokens, and the target verifies 8 positions per sequence. That pins down the CUDA graph sizes and the fused MoE buffer.' },
        { t: 'Distributed line', d: 'Two nodes at TP=2 only work with identical ranks: same environment, same overlays in the same order and, as a consequence, the same graphs.' },
        { t: 'Memory line (UMA)', d: 'CPU and GPU share the RAM. The fast loader eats KV, so the pool is fixed at 11 GiB; as a result, util (0.85) becomes just the boot’s free-memory floor, and the drafter’s compact pages make 850k fit in block IDs.' },
        { t: 'Attention line', d: 'MLA without RoPE on a GPU that has only one sparse-MLA kernel: the KV has to be <code>fp8_ds_mla</code>, and the query gets 64 zeros to fit the 576 geometry.' },
        { t: 'Transfers', d: 'Where two constraints meet: the EXL3 kernel buffer depends on speculation, the graphs must match across ranks, and the pool in GiB only makes sense in block IDs, whose size comes from the attention record.' },
      ];
      let user = null;
      const st = UI.stepper(fig, steps, { autoplay: 7000, onStep: () => { user = null; } });
      V.fig(fig, (g, w, h, t, dt, S, H) => {
        const C = V.C();
        const narrow = w < 560;
        const pad = 14, detailH = narrow ? 108 : 74;
        const rowH = (h - 6 - detailH) / 5;
        const x0 = pad + (narrow ? 26 : 34), x1 = w - pad - (narrow ? 26 : 34);
        const SX = (j) => x0 + (j * (x1 - x0)) / 4;
        const SY = (i) => 8 + i * rowH + 24;
        const act = st.i;
        const el = st.elapsed();
        const u = (el / 6.3) % 1;
        if (H.pointer.clicked) {
          let best = null, bd = 26;
          for (let i = 0; i < 5; i++) for (let j = 0; j < 5; j++) { const d = Math.hypot(H.pointer.x - SX(j), H.pointer.y - SY(i)); if (d < bd) { bd = d; best = [i, j]; } }
          if (best) { st.stop(); if (st.i !== best[0]) st.go(best[0]); user = best; }
        }
        let sel = user;
        if (!sel && act < 5) sel = [act, Math.min(4, Math.floor(u * 5))];
        const trIdx = act === 5 ? Math.floor(el / 2.4) % T.length : -1;
        T.forEach((tr, k) => {
          const on = act === 5;
          if (narrow && !on) return; // on mobile, the transfers only show up in step 6
          const A0 = [SX(tr.a[1]), SY(tr.a[0])], B0 = [SX(tr.b[1]), SY(tr.b[0])];
          const col = on ? (k === trIdx ? C.ink : V.alpha(C.ink, narrow ? 0.3 : 0.45)) : V.alpha(C.muted, 0.3);
          g.strokeStyle = col; g.lineWidth = on && k === trIdx ? 2.6 : 1.4; g.setLineDash([4, 4]); g.lineDashOffset = -t * 18;
          g.beginPath(); g.moveTo(A0[0], A0[1]);
          // S-curve: leaves along the track, goes down the corridor between the columns and arrives along the lower track
          const xm = (A0[0] + B0[0]) / 2;
          g.bezierCurveTo(xm, A0[1], xm, B0[1], B0[0], B0[1]);
          g.stroke(); g.setLineDash([]); g.lineDashOffset = 0;
        });
        const mw = (x1 - x0) / 4 - 4;
        ROWS.forEach((R, i) => {
          const col = C[R.c]; const on = act === i || act === 5; const y = SY(i);
          V.text(g, R.name, pad, y - 13, { size: 11, font: 'mono', weight: 700, color: col, alpha: on ? 1 : 0.5 });
          V.line(g, x0, y, x1, y, { color: V.alpha(col, on ? 1 : 0.28), lw: on ? 5 : 3 });
          if (on) {
            const tu = act === 5 ? ((t * 0.1 + i * 0.21) % 1) : u;
            const px = V.lerp(x0, x1, tu);
            V.box(g, px - 15, y - 8, 30, 16, { r: 6, fill: col });
            for (let q = 0; q < 3; q++) V.box(g, px - 11 + q * 8, y - 4, 6, 6, { r: 1.5, fill: V.alpha(C.bg, 0.9) });
          }
          for (let j = 0; j < 5; j++) {
            const x = SX(j); const isSel = sel && sel[0] === i && sel[1] === j;
            if (isSel) V.glow(g, x, y, 22, col, 0.9);
            g.beginPath(); g.arc(x, y, on ? 8 : 6, 0, Math.PI * 2); g.fillStyle = C.bg; g.fill();
            g.lineWidth = on ? 3 : 2; g.strokeStyle = V.alpha(col, on ? 1 : 0.35); g.stroke();
            const lc = on ? C.ink : C.muted;
            V.text(g, R.st[j][0], x, y + 23, { size: 11, align: 'center', color: lc, weight: isSel ? 700 : 500, maxW: mw, alpha: on ? 1 : 0.65 });
            V.text(g, R.st[j][1], x, y + 37, { size: 11, align: 'center', color: lc, font: 'mono', maxW: mw, alpha: on ? 1 : 0.65 });
          }
        });
        const dy = h - detailH + 2;
        V.box(g, pad, dy, w - 2 * pad, detailH - 8, { r: 10, fill: V.alpha(C.ink, 0.05), stroke: V.alpha(C.muted, 0.3) });
        let title = '', body = '', col = C.ink;
        if (act === 5 && !user) { title = 'TRANSFER ' + (trIdx + 1) + ' OF 3'; body = T[trIdx].d; }
        else if (sel) { const R = ROWS[sel[0]]; title = R.name + ' · STATION ' + (sel[1] + 1); body = R.st[sel[1]][2]; col = C[R.c]; }
        V.text(g, title, pad + 12, dy + 18, { size: 11, font: 'mono', weight: 700, color: col });
        if (!narrow) V.text(g, user ? 'station pinned' : 'click a station to pin it', w - pad - 12, dy + 18, { size: 11, align: 'right', color: C.muted });
        para(g, body, pad + 12, dy + 38, w - 2 * pad - 24, { size: 12.5, color: C.ink, lh: 17 });
      });
    })();

    /* ============================================ F3 · topology */
    (function () {
      const fig = root.querySelector('#fig-topologia');
      const S = { topo: 'tp2', ssh: false };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Topology', options: [{ v: 'tp2', label: 'TP=2 (recipe)' }, { v: 'tp3', label: 'TP=3' }, { v: 'tp4', label: 'TP=4' }], value: 'tp2', onChange: (v) => { S.topo = v; } });
      UI.toggle(ctl, { label: 'orchestration traffic (SSH)', value: false, onChange: (v) => { S.ssh = v; } });
      const flow = (g, ax, ay, bx, by, t, col, n, speed, r) => { for (let k = 0; k < n; k++) { const u = (k / n + t * speed) % 1; V.dot(g, V.lerp(ax, bx, u), V.lerp(ay, by, u), r || 2.6, col); } };
      const nodeBox = (g, C, x, y, nw, nh, title, sub, col) => {
        V.box(g, x, y, nw, nh, { r: 14, fill: V.alpha(col, 0.07), stroke: V.alpha(col, 0.65), lw: 1.5 });
        V.text(g, title, x + 12, y + 20, { size: 12.5, weight: 700, color: C.ink, maxW: nw - 24 });
        if (sub) V.text(g, sub, x + 12, y + 36, { size: 11, font: 'mono', color: C.muted, maxW: nw - 24 });
      };
      const proc = (g, C, x, y, pw, ph, lab, col, dashed) => {
        if (dashed) g.setLineDash([4, 4]);
        V.box(g, x, y, pw, ph, { r: 8, fill: dashed ? V.alpha(C.muted, 0.04) : V.alpha(col, 0.2), stroke: V.alpha(col, dashed ? 0.55 : 0.9), lw: 1.3 });
        g.setLineDash([]);
        V.text(g, lab, x + pw / 2, y + ph / 2 + 1, { size: 11.5, align: 'center', base: 'middle', color: dashed ? C.muted : C.ink, weight: 500, maxW: pw - 10 });
      };
      function tp2(g, w, h, t, C) {
        const narrow = w < 680, pad = 14, inf = !S.ssh;
        if (!narrow) {
          const top = 64, nh = h - top - 20, nw = (w - 2 * pad) * 0.4;
          const hx = pad, wx = w - pad - nw;
          nodeBox(g, C, hx, top, nw, nh, 'HEAD · rank 0 · 10.0.0.1', 'container glm53-exl3-head', C.a);
          nodeBox(g, C, wx, top, nw, nh, 'WORKER · rank 1 · 10.0.0.2', 'container glm53-exl3-worker', C.b);
          const pw = nw - 24, ph = 32, rows = [top + 56, top + 100, top + 144];
          proc(g, C, hx + 12, rows[0], pw, ph, 'OpenAI API server · :8888', C.d);
          proc(g, C, hx + 12, rows[1], pw, ph, 'EngineCore · scheduler + KV', C.b);
          proc(g, C, hx + 12, rows[2], pw, ph, 'GPU worker · rank 0 (GB10)', C.a);
          proc(g, C, wx + 12, rows[0], pw, ph, 'no API (--headless)', C.muted, true);
          proc(g, C, wx + 12, rows[1], pw, ph, 'no scheduler of its own', C.muted, true);
          proc(g, C, wx + 12, rows[2], pw, ph, 'GPU worker · rank 1 (GB10)', C.a);
          const lx0 = hx + nw, lx1 = wx, mid = (lx0 + lx1) / 2;
          const ly = rows[2] + ph / 2;
          V.line(g, lx0, ly, lx1, ly, { color: V.alpha(C.a, inf ? 0.9 : 0.35), lw: 4 });
          if (inf) { flow(g, lx0 + 4, ly - 5, lx1 - 4, ly - 5, t, C.a, 7, 0.45); flow(g, lx1 - 4, ly + 5, lx0 + 4, ly + 5, t, C.a, 7, 0.45); }
          V.text(g, 'CX7 · RoCEv2', mid, ly - 16, { size: 11.5, font: 'mono', align: 'center', color: C.ink, weight: 600 });
          V.text(g, '2 all-reduces/layer', mid, ly + 24, { size: 11, align: 'center', color: C.muted, maxW: lx1 - lx0 - 6 });
          V.text(g, '(after attention and MoE)', mid, ly + 38, { size: 11, align: 'center', color: C.muted, maxW: lx1 - lx0 - 6 });
          // interfaces pinned per rank, at the ends of the cable
          V.text(g, 'CX7 enp1s0f1np1 · HCA rocep1s0f1', hx + nw - 14, rows[2] + ph + 16, { size: 11, font: 'mono', align: 'right', color: C.a, maxW: pw });
          V.text(g, 'CX7 enp1s0f0np0 · HCA rocep1s0f0', wx + 14, rows[2] + ph + 16, { size: 11, font: 'mono', color: C.a, maxW: pw });
          const ry = rows[1] + ph / 2;
          V.line(g, lx0, ry, lx1, ry, { color: V.alpha(C.muted, 0.6), lw: 1.2, dash: [2, 4] });
          V.text(g, 'rendezvous :29521', mid, ry - 7, { size: 11, font: 'mono', align: 'center', color: C.muted });
          // memory per rank
          const my = rows[2] + ph + 40;
          [hx, wx].forEach((nx) => {
            V.text(g, 'RANK MEMORY (UMA)', nx + 14, my, { size: 11, font: 'mono', weight: 700, color: C.muted });
            const segW = pw, sy0 = my + 8;
            const parts = [[79.65, C.d, 'weights ~80'], [3.29, C.f, ''], [11, C.a, 'KV'], [21.55, C.b, 'rest'], [6.2, C.e, '']];
            let acc = 0;
            parts.forEach(([v, col, lab]) => { const x0 = nx + 12 + (acc / 121.69) * segW, ww = (v / 121.69) * segW; V.box(g, x0, sy0, Math.max(1, ww - 1), 16, { r: 3, fill: V.alpha(col, 0.7) }); if (lab && ww > tw(g, lab, 11, 'sans', 600) + 6) V.text(g, lab, x0 + ww / 2, sy0 + 9, { size: 11, weight: 600, align: 'center', base: 'middle', color: onFill(C) }); acc += v; });
            V.text(g, 'replicated MLA latent · ≥ 6.2 GiB free under load', nx + 14, sy0 + 32, { size: 11, color: C.muted, maxW: pw - 4 });
          });
          // weights
          const dky = top + nh - 58;
          V.box(g, hx + 12, dky, pw, 40, { r: 8, fill: V.alpha(C.e, 0.12), stroke: V.alpha(C.e, 0.6) });
          V.text(g, 'HF cache · 164 GiB · 83 shards', hx + 24, dky + 17, { size: 11.5, color: C.ink, maxW: pw - 20 });
          V.text(g, 'MODEL_REVISION 76c0b517…', hx + 24, dky + 32, { size: 11, font: 'mono', color: C.muted, maxW: pw - 20 });
          V.box(g, wx + 12, dky, pw, 40, { r: 8, fill: V.alpha(C.e, 0.06), stroke: V.alpha(C.e, 0.45) });
          V.text(g, 'same snapshot (rsync or NFS ro)', wx + 24, dky + 17, { size: 11.5, color: C.ink, maxW: pw - 20 });
          V.text(g, 'HF_HUB_OFFLINE=1 in the container', wx + 24, dky + 32, { size: 11, font: 'mono', color: C.muted, maxW: pw - 20 });
          V.line(g, lx0, dky + 20, lx1, dky + 20, { color: V.alpha(C.e, 0.7), lw: 1.6, dash: [6, 4], dashOffset: -t * 14 });
          V.text(g, 'NFS · nconnect=8', mid, dky + 12, { size: 11, font: 'mono', align: 'center', color: C.muted });
          // client
          const cw = Math.min(230, nw - 20);
          V.box(g, hx + 12, 14, cw, 30, { r: 9, fill: V.alpha(C.d, 0.16), stroke: C.d, label: 'client → /v1/chat/completions', size: 11.5 });
          const axc = hx + nw - 36;
          V.arrow(g, hx + 12 + cw, 29, axc, 29, { color: V.alpha(C.d, 0.8), lw: 1.4, head: 0 });
          V.arrow(g, axc, 29, axc, rows[0] - 2, { color: C.d, lw: 1.6 });
          if (inf) flow(g, axc, 32, axc, rows[0] - 4, t, C.d, 3, 0.6, 2.4);
          if (S.ssh) {
            const sw = 186, sx = mid - sw / 2;
            V.box(g, sx, 12, sw, 34, { r: 9, fill: V.alpha(C.d, 0.22), stroke: C.d, label: 'start.sh (head host)', size: 12, weight: 700 });
            const tx = wx + nw / 2, ty = top;
            V.arrow(g, sx + sw, 30, tx, ty - 2, { color: C.d, lw: 2, curve: -20 });
            flow(g, sx + sw, 30, tx, ty - 2, t, C.d, 5, 0.35, 3);
            V.text(g, 'ssh BatchMode · scp · docker run/kill/load', wx + nw - 4, 30, { size: 11, align: 'right', color: C.ink, weight: 600, maxW: nw - 20 });
          }
          return;
        }
        // narrow: head on top, worker below
        const top = 42, gapV = 72, nh = (h - top - gapV - 10) / 2, nw = w - 2 * pad;
        const hx = pad, hy = top, wy = top + nh + gapV;
        nodeBox(g, C, hx, hy, nw, nh, 'HEAD · rank 0 · 10.0.0.1', 'glm53-exl3-head', C.a);
        nodeBox(g, C, hx, wy, nw, nh, 'WORKER · rank 1 · 10.0.0.2', 'glm53-exl3-worker', C.b);
        const pw = nw - 24, ph = 28;
        const hr = [hy + nh - 3 * (ph + 6) - 4, hy + nh - 2 * (ph + 6) - 4, hy + nh - (ph + 6) - 4];
        proc(g, C, hx + 12, hr[0], pw, ph, 'API server · :8888', C.d);
        proc(g, C, hx + 12, hr[1], pw, ph, 'EngineCore · scheduler', C.b);
        proc(g, C, hx + 12, hr[2], pw, ph, 'GPU worker · rank 0', C.a);
        const wr = [wy + 46, wy + 46 + ph + 6];
        proc(g, C, hx + 12, wr[0], pw, ph, 'GPU worker · rank 1', C.a);
        proc(g, C, hx + 12, wr[1], pw, ph, 'no API (--headless)', C.muted, true);
        const lx = w - pad - 44, ly0 = hr[2] + ph, ly1 = wr[0];
        V.line(g, lx, ly0, lx, ly1, { color: V.alpha(C.a, inf ? 0.9 : 0.35), lw: 4 });
        if (inf) { flow(g, lx - 5, ly0 + 2, lx - 5, ly1 - 2, t, C.a, 4, 0.5); flow(g, lx + 5, ly1 - 2, lx + 5, ly0 + 2, t, C.a, 4, 0.5); }
        if (!S.ssh) {
          V.text(g, 'CX7 · RoCEv2 · 2 all-reduces/layer', lx - 12, hy + nh + 20, { size: 11, color: C.ink, weight: 600, align: 'right', maxW: lx - 12 - pad });
          V.text(g, 'head enp1s0f1np1 · rocep1s0f1', lx - 12, hy + nh + 38, { size: 11, font: 'mono', color: C.a, align: 'right', maxW: lx - 12 - pad });
          V.text(g, 'worker enp1s0f0np0 · rocep1s0f0', lx - 12, hy + nh + 54, { size: 11, font: 'mono', color: C.b, align: 'right', maxW: lx - 12 - pad });
        }
        // rank memory (same on both)
        const my2 = wr[1] + ph + 20;
        if (my2 + 26 <= wy + nh) {
          V.text(g, 'RANK MEMORY (SAME ON BOTH)', hx + 14, my2, { size: 11, font: 'mono', weight: 700, color: C.muted });
          const parts = [[79.65, C.d, 'weights ~80'], [3.29, C.f, ''], [11, C.a, 'KV'], [21.55, C.b, 'rest'], [6.2, C.e, '']];
          let acc = 0;
          parts.forEach(([v, col, lab]) => { const x0 = hx + 12 + (acc / 121.69) * pw, ww = (v / 121.69) * pw; V.box(g, x0, my2 + 8, Math.max(1, ww - 1), 16, { r: 3, fill: V.alpha(col, 0.7) }); if (lab && ww > tw(g, lab, 11, 'sans', 600) + 6) V.text(g, lab, x0 + ww / 2, my2 + 17, { size: 11, weight: 600, align: 'center', base: 'middle', color: onFill(C) }); acc += v; });
        }
        V.box(g, pad, 8, Math.min(230, nw), 26, { r: 8, fill: V.alpha(C.d, 0.16), stroke: C.d, label: 'client → :8888', size: 11.5 });
        if (S.ssh) {
          const px = pad + 6;
          g.strokeStyle = C.d; g.lineWidth = 2; g.setLineDash([5, 4]); g.beginPath(); g.moveTo(px + 6, hy + 50); g.quadraticCurveTo(px - 10, (hy + wy) / 2 + nh / 2, px + 6, wy + 20); g.stroke(); g.setLineDash([]);
          for (let k = 0; k < 4; k++) { const u = (k / 4 + t * 0.35) % 1; const p = [(1 - u) * (1 - u) * (px + 6) + 2 * (1 - u) * u * (px - 10) + u * u * (px + 6), (1 - u) * (1 - u) * (hy + 50) + 2 * (1 - u) * u * ((hy + wy) / 2 + nh / 2) + u * u * (wy + 20)]; V.dot(g, p[0], p[1], 3, C.d); }
          V.box(g, pad + 18, hy + nh + 8, nw - 36, 36, { r: 8, fill: V.alpha(C.d, 0.18), stroke: C.d });
          V.text(g, 'SSH BatchMode: scp, docker run/kill/load', pad + 30, hy + nh + 30, { size: 11, color: C.ink, weight: 600, maxW: nw - 60 });
        }
      }
      function ring(g, w, h, t, C, n) {
        const narrow = w < 680, pad = 14;
        const cx = w / 2, cy = narrow ? 112 : 150, R = narrow ? Math.min(w * 0.3, 88) : Math.min(w * (n === 4 ? 0.22 : 0.18), n === 4 ? 118 : 110);
        const pts = [];
        for (let i = 0; i < n; i++) { const a = -Math.PI / 2 + (i * 2 * Math.PI) / n; pts.push([cx + R * Math.cos(a), cy + R * Math.sin(a)]); }
        if (n === 3) {
          for (let i = 0; i < 3; i++) {
            const a = pts[i], b = pts[(i + 1) % 3];
            const dx = b[0] - a[0], dy = b[1] - a[1], L = Math.hypot(dx, dy), ux = dx / L, uy = dy / L;
            const s0 = [a[0] + ux * 40, a[1] + uy * 30], s1 = [b[0] - ux * 44, b[1] - uy * 32];
            V.arrow(g, s0[0], s0[1], s1[0], s1[1], { color: V.alpha(C.a, 0.9), lw: 2.6, curve: 14, head: 10 });
            flow(g, s0[0], s0[1], s1[0], s1[1], t, C.a, 4, 0.4, 2.6);
          }
          V.text(g, 'directed ring', cx, cy - 2, { size: 11.5, align: 'center', color: C.ink, weight: 600 });
          V.text(g, 'Port0 → Port1', cx, cy + 14, { size: 11, align: 'center', font: 'mono', color: C.muted });
        } else {
          pts.forEach((p) => { V.line(g, p[0], p[1], cx, cy, { color: V.alpha(C.d, 0.6), lw: 2.4 }); flow(g, p[0], p[1], cx, cy, t, C.d, 3, 0.35, 2.4); flow(g, cx, cy, p[0], p[1], t + 0.5, C.a, 3, 0.35, 2.2); });
          V.box(g, cx - 46, cy - 16, 92, 32, { r: 8, fill: C.bg });
          V.box(g, cx - 46, cy - 16, 92, 32, { r: 8, fill: V.alpha(C.d, 0.2), stroke: C.d, label: 'switch', size: 12, weight: 600 });
        }
        pts.forEach((p, i) => {
          const nw = 98, nh = 42;
          V.box(g, p[0] - nw / 2, p[1] - nh / 2, nw, nh, { r: 10, fill: C.bg });
          V.box(g, p[0] - nw / 2, p[1] - nh / 2, nw, nh, { r: 10, fill: V.alpha(i === 0 ? C.a : C.b, 0.2), stroke: i === 0 ? C.a : C.b, lw: 1.6 });
          V.text(g, 'rank ' + i, p[0], p[1] - 3, { size: 12, weight: 700, align: 'center', color: C.ink });
          V.text(g, i === 0 ? 'head · API' : 'headless', p[0], p[1] + 12, { size: 11, align: 'center', font: 'mono', color: C.muted });
        });
        const B = n === 3 ? [
          'Directed-ring cabling: Port0 of each node → Port1 of the next (a mirrored ring leaves one pair unconnected).',
          '64 heads don’t divide by 3: --hf-overrides with 66 (22 per rank).',
          'Expert parallel: 96 of the 288 experts per rank (the 2048 trellis doesn’t divide by 3).',
          'Drafter only on rank 0 (draft TP=1; GQA 32/8 → 36/9). Control over the management LAN, data via RoCE.',
          'Measured on 2026-09-14: 87.8 tok/s on structured text, versus 73.4 on TP2 with the same prompts.',
        ] : [
          '4 Sparks on a CRS812 switch (2 × 200G + 2 × 100G). Experimental, not tested by the maintainer.',
          'k=3 (accepts 53% versus 30% with k=7), 8 seqs, MNBT 2048, util 0.75, draft TP 4.',
          'Decode 1.45× that of 2 nodes; cold prefill only +29% (1162 versus 901 tok/s at 282k): network-bound.',
          'A 150-min soak hung 3 times out of 3 (31–78 min) with a 96k prefill and 6–7 speculative streams.',
        ];
        let y = cy + R + (narrow ? 44 : 50);
        const bw = w - 2 * pad - 16;
        B.forEach((s) => { V.dot(g, pad + 5, y - 4, 3, n === 3 ? C.a : C.d); y += para(g, s, pad + 16, y, bw, { size: narrow ? 11.5 : 12.5, color: C.ink, lh: narrow ? 15 : 17 }) + (narrow ? 3 : 5); });
      }
      V.fig(fig, (g, w, h, t) => {
        fitH(fig, 430, 456, w);
        const C = V.C();
        if (S.topo === 'tp2') tp2(g, w, h, t, C);
        else ring(g, w, h, t, C, S.topo === 'tp3' ? 3 : 4);
      });
    })();

    /* ============================================ F4 · precedence */
    (function () {
      const fig = root.querySelector('#fig-precedencia');
      const VARS = {
        MAX_MODEL_LEN: { exp: '${MAX_MODEL_LEN:-850000}', kind: ':-', def: '850000', env: '850000', caller: '262144' },
        MM_IMAGE_TOKENS: { exp: '${MM_IMAGE_TOKENS-2048}', kind: '-', def: '2048', env: '2048', caller: '1024', empty: 'The empty value survives: the --mm-processor-kwargs flag is omitted and vLLM falls back to the checkpoint’s 8,000 tokens per image, above the encoder cache (MNBT 7,168).' },
        DEFAULT_MAX_NEW_TOKENS: { exp: '${DEFAULT_MAX_NEW_TOKENS-65536}', kind: '-', def: '65536', env: '65536', caller: '32768', empty: 'The empty value survives: with no server default, a request without max_tokens reverts to stock behavior and can grow the KV until it preempts other sessions.' },
        GLM53_DRAFT_KV_COMPACT: { exp: '${GLM53_DRAFT_KV_COMPACT-1}  # under dflash', kind: '-', def: '1', env: '0', caller: '0', empty: 'An explicit empty value is an operator error: numeric validation refuses it before stopping the pair, instead of guessing a mode.', err: true },
        ABLIT: { exp: 'ABLIT=0  # forced after source .env', kind: 'forced', def: '0', env: '1', caller: '1' },
        IMAGE: { exp: '${IMAGE:-ghcr.io/…:exl3-instanttensor}', kind: ':-', def: 'ghcr.io/…:exl3-instanttensor', env: 'ghcr.io/…:exl3-instanttensor', caller: 'glm53-flash-sm121:local', ident: true },
      };
      const S = { v: 'MAX_MODEL_LEN', caller: 'none', env: true };
      const ctl = UI.controls(fig);
      UI.select(ctl, { label: 'Variable', options: Object.keys(VARS).map((k) => ({ v: k, label: k })), value: S.v, onChange: (v) => { S.v = v; } });
      UI.seg(ctl, { label: 'Caller', options: [{ v: 'none', label: 'no export' }, { v: 'val', label: 'export value' }, { v: 'empty', label: 'export empty' }], value: 'none', onChange: (v) => { S.caller = v; } });
      UI.toggle(ctl, { label: '.env defines the key', value: true, onChange: (v) => { S.env = v; } });
      function resolve(X) {
        const c = S.caller, e = S.env;
        const r = { layer: c !== 'none' ? 0 : e ? 1 : 2, val: '', status: 'ok', note: '' };
        if (X.kind === 'forced') {
          if (c === 'val') { r.val = X.caller; r.note = 'The caller’s export is re-exported after ABLIT=0. That’s the only way abliteration turns on in TP2.'; }
          else if (c === 'empty') { r.val = '""'; r.status = 'warn'; r.note = 'Empty: the ${ABLIT:-0} = 1 check fails and abliteration stays off.'; }
          else { r.val = '0'; r.layer = 2; r.status = e ? 'warn' : 'ok'; r.note = e ? 'The .env says ABLIT=1, but the launcher forces ABLIT=0 right after the source. No effect on TP2.' : 'Default: stock o_proj.'; r.strike = e; }
          return r;
        }
        let raw;
        if (c === 'val') raw = X.caller; else if (c === 'empty') raw = ''; else if (e) raw = X.env;
        if (raw === undefined) { r.val = X.def; r.note = 'Nobody defines the key: the script’s default applies.'; }
        else if (raw === '') {
          if (X.kind === ':-') { r.val = X.def; r.note = 'The :- operator treats empty as unset and falls back to the default.'; }
          else { r.val = '""'; r.note = X.empty; r.status = X.err ? 'err' : 'warn'; }
        } else { r.val = raw; r.note = c === 'val' ? 'The caller’s export beats the .env.' : 'The .env value applies.'; }
        if (X.ident && c !== 'none' && e) { r.status = 'warn'; r.note = (c === 'empty' ? 'Empty + :- → default. ' : '') + 'Identity variable: the launcher warns “NOTE: IMAGE=… from the environment overrides .env value …”.'; }
        return r;
      }
      V.fig(fig, (g, w, h, t) => {
        fitH(fig, 370, 450, w);
        const C = V.C(); const narrow = w < 520; const pad = 14;
        const name = S.v, X = VARS[name], r = resolve(X);
        const cardH = narrow ? 58 : 48, gap = 10, lx = pad + 22;
        const titles = ['3 · caller export', '2 · .env (source)', '1 · script default'];
        const subs = ['snapshot before source .env', 'kit file', 'expansion in start.sh'];
        const vals = [
          S.caller === 'none' ? '— (no export)' : S.caller === 'val' ? name + '=' + X.caller : name + '=   (empty)',
          S.env ? name + '=' + X.env : '— (key missing)',
          X.exp,
        ];
        const defined = [S.caller !== 'none', S.env, true];
        const cols = [C.c, C.b, C.a];
        const cys = [];
        for (let i = 0; i < 3; i++) {
          const x = lx + i * 8, y = 10 + i * (cardH + gap), cw = w - lx - pad - 16;
          cys.push(y + cardH / 2);
          const win = i === r.layer, col = cols[i];
          V.box(g, x, y, cw, cardH, { r: 10, fill: V.alpha(col, win ? 0.2 : 0.06), stroke: V.alpha(col, win ? 1 : 0.35), lw: win ? 2 : 1 });
          V.text(g, titles[i], x + 12, y + 18, { size: 12, weight: 700, color: C.ink });
          if (!narrow) V.text(g, subs[i], x + 12, y + 35, { size: 11, color: C.muted });
          const vy = narrow ? y + cardH - 14 : y + cardH / 2 + 1;
          const vx = narrow ? x + 12 : x + cw - 12;
          V.text(g, vals[i], vx, vy, { size: 12, font: 'mono', color: defined[i] ? C.ink : C.muted, align: narrow ? 'left' : 'right', base: 'middle', maxW: narrow ? cw - 24 : cw * 0.58, alpha: defined[i] ? 1 : 0.7 });
          if (i === 1 && r.strike) { const vw = Math.min(tw(g, vals[i], 12, 'mono'), narrow ? cw - 24 : cw * 0.58); const sx = narrow ? vx : vx - vw; V.line(g, sx, vy, sx + vw, vy, { color: C.red, lw: 2 }); }
          if (win) { const bw = tw(g, 'wins', 11, 'sans', 700) + 14; V.box(g, x + cw - bw - 8, y - 8, bw, 16, { r: 8, fill: col }); V.text(g, 'wins', x + cw - bw / 2 - 8, y, { size: 11, weight: 700, align: 'center', base: 'middle', color: onFill(C) }); }
        }
        // light beam
        const bxl = pad + 6, yEnd = cys[r.layer];
        V.line(g, bxl, 4, bxl, yEnd, { color: V.alpha(C.ink, 0.25), lw: 3 });
        const u = (t * 0.7) % 1;
        V.glow(g, bxl, V.lerp(4, yEnd, V.ease.inOut(u)), 12, C.d, 1);
        V.arrow(g, bxl, yEnd, lx + r.layer * 8 - 2, yEnd, { color: C.d, lw: 2, head: 7 });
        for (let i = 0; i < r.layer; i++) { g.beginPath(); g.arc(bxl, cys[i], 4, 0, Math.PI * 2); g.strokeStyle = V.alpha(C.ink, 0.5); g.lineWidth = 1.5; g.stroke(); }
        // result
        const ry = 10 + 3 * (cardH + gap) + 4, rh = h - ry - 8, rw = w - 2 * pad;
        const scol = r.status === 'err' ? C.red : r.status === 'warn' ? C.d : C.e;
        V.box(g, pad, ry, rw, rh, { r: 12, fill: V.alpha(scol, 0.08), stroke: V.alpha(scol, 0.6), lw: 1.4 });
        V.text(g, 'script expansion:', pad + 12, ry + 20, { size: 11, color: C.muted });
        V.text(g, X.exp, pad + 12 + (narrow ? 0 : 128), ry + (narrow ? 38 : 20), { size: 12, font: 'mono', color: C.ink, maxW: rw - 24 - (narrow ? 0 : 128) });
        const vy = ry + (narrow ? 62 : 46);
        V.text(g, 'effective value →', pad + 12, vy, { size: 12, color: C.ink, weight: 600 });
        V.text(g, name + '=' + r.val, pad + 12 + tw(g, 'effective value →', 12, 'sans', 600) + 10, vy, { size: 13, font: 'mono', weight: 700, color: scol, maxW: rw - 150 });
        const nh = para(g, r.note, pad + 12, vy + 22, rw - 24, { size: 12, color: C.ink, lh: 16 });
        // order of operations in the launcher
        const ops = ['snapshot exports', 'source .env', 'ABLIT=0', 're-export snapshot', 'expand ${…}'];
        const hot = X.kind === 'forced' ? (S.caller !== 'none' ? [0, 2, 3] : [1, 2]) : S.caller !== 'none' ? [0, 3, 4] : S.env ? [1, 4] : [4];
        let ox = pad + 12, oy = vy + 22 + nh + 10;
        const rows = [];
        ops.forEach((o, k) => { const lab = (k + 1) + ' ' + o, wd = tw(g, lab, 11, 'mono', 500) + 14; if (ox + wd > pad + rw - 12) { ox = pad + 12; oy += 24; } rows.push([ox, oy, lab, k]); ox += wd + 16; });
        if (oy + 20 < ry + rh - 4) rows.forEach(([x, y2, lab, k], j) => {
          const on = hot.includes(k);
          chip(g, x, y2, lab, { size: 11, h: 19, fill: on ? V.alpha(C.d, 0.2) : V.alpha(C.muted, 0.06), stroke: on ? C.d : V.alpha(C.muted, 0.4), color: on ? C.ink : C.muted });
          if (j < rows.length - 1 && rows[j + 1][1] === y2) V.text(g, '→', rows[j + 1][0] - 9, y2 + 10, { size: 11, align: 'center', base: 'middle', color: C.muted });
        });
      });
    })();

    /* ============================================ F5 · layered image */
    (function () {
      const fig = root.querySelector('#fig-imagem');
      const BR = [
        { lab: 'FROM vllm-openai@sha256:905c0293…', sh: 'FROM …@sha256:905c…', lock: true, c: 'b' },
        { lab: 'NoPE sparse-MLA patches + asserts', sh: 'NoPE patches + asserts', c: 'c' },
        { lab: 'exl3 + exllamav3_ext (sm_121a)', sh: 'exl3 + ext sm_121a', lock: true, c: 'd' },
        { lab: 'overlays + CPU tests', sh: 'overlays + CPU tests', c: 'a' },
        { lab: 'instanttensor==0.2.0 --no-deps', sh: 'instanttensor --no-deps', lock: true, c: 'e' },
        { lab: 'cold-load UMA + skip graph profile', sh: 'cold-load + skip profile', c: 'f' },
        { lab: 'LABEL glm53.recipe.stamp', sh: 'LABEL recipe.stamp', c: 'd', stamp: true },
      ];
      const CODE = [
        ['ARG BASE=vllm/vllm-openai:', '  glm53-flash-arm64-cu130', '  @sha256:905c0293…', '# vLLM 0.1.dev20051+g487ecf187', '# CUDA 13.0 · arm64'],
        ['replace_once(old, new)', '  # exactly 1 occurrence', 'self.rope_pad = 64', 'q = F.pad(q, (0, self.rope_pad))', "assert 'self.rope_pad = 64' in \\", '  inspect.getsource(impl.__init__)'],
        ['QuantizationMethods += "exl3"', 'ENV TORCH_CUDA_ARCH_LIST=12.1a', 'ENV MAX_JOBS=8', 'CPATH=…/nvidia/cu13/include', 'exllamav3 @ c5d9c657 (0.0.43)', "assert hasattr(ext, 'exl3_moe')"],
        ['COPY overlay/patch_*.py tests/…', 'RUN test_apc_no_store.py  # first', 'RUN patch_apc_no_store.py', '# kpool: 2 patches, 1 layer', '# (overlay2 has a layer limit)'],
        ['RUN pip install --no-deps \\', '    instanttensor==0.2.0', '# no deps: keep the image', '# NCCL 2.30.7 intact'],
        ['RUN test_cold_load_uma.py && \\', '    patch_cold_load_uma.py', 'RUN test_skip_cudagraph_profile.py', '    && patch_skip_cudagraph_profile.py', '# idempotent: boot reapplies them'],
        ['ARG GLM53_RECIPE_STAMP=unknown', 'LABEL glm53.recipe.stamp=${…}', '# last layer: a rebuild that', '# only changes the stamp is cheap'],
      ];
      const WHY = [
        'A tag can be republished; a digest can’t. The patches are byte-for-byte anchors on this code.',
        'With a different upstream, the build breaks instead of producing a silently wrong image.',
        'Native sm_121a cubins; a light compile so it doesn’t compete for the unified memory.',
        'A fresh build ends up identical to an image patched at boot, and reapplying becomes a no-op.',
        'A normal pip install would swap the image’s NCCL 2.30.7 for 2.29.7.',
        'With the page cache full, the stock path never reached /health.',
        'A git pull that changes an overlay makes the stamp diverge and triggers a rebuild, once.',
      ];
      const SC = {
        igual: { repo: 'a3f19c…', lab: 'a3f19c…', ok: true, res: '✓ uses the local image', sub: 'The worker compares the image key (RootFS.Layers); if it differs: pull, or docker save --platform | ssh docker load.', c: 'e' },
        edit: { repo: '7c02e8…', lab: 'a3f19c…', ok: false, res: '↻ local rebuild (implicit BUILD=1)', sub: 'The new image gets the 7c02e8… stamp and is shipped to the worker. The GPU self-check runs before serving.', c: 'd' },
        skip: { repo: '7c02e8…', lab: 'a3f19c…', ok: false, res: '⚠ SKIP_BUILD=1: keeps the image and just warns', sub: 'The stamp still diverges: the image may not contain the edited overlay (at runtime, the bind mount still applies it).', c: 'd' },
        ghcr: { repo: 'a3f19c…', lab: 'a3f19c…', pulled: '9e4b51…', ok: false, res: '↩ pull rejected: restores the local tag', sub: 'GHCR published a different recipe. A pull is only adopted when the published stamp also matches the repository.', c: 'c' },
      };
      let scen = 'igual';
      const steps = [
        { t: 'FROM: the base pinned by digest', d: '<code>ARG BASE=vllm/vllm-openai:glm53-flash-arm64-cu130@sha256:905c0293…</code> pins the bytes of vLLM <code>0.1.dev20051+g487ecf187</code> (CUDA 13, arm64). All the following patches are exact anchors on that code.' },
        { t: 'Backend patches + asserts', d: 'NoPE → 576 geometry with zeros, prefill without dense MHA, top-k as capacity, 511 pools + tail, page 64, no FlashInfer autotune, PDL only on SM9x/10x. Each substitution requires a single occurrence, and a block of <code>assert</code>s rereads the source with <code>inspect.getsource</code>.' },
        { t: 'EXL3: register and compile', d: '<code>"exl3"</code> joins the <code>Literal</code> of quantization methods, with a lazy import of <code>Exl3Config</code>. <code>exllamav3_ext@c5d9c657</code> is compiled with <code>TORCH_CUDA_ARCH_LIST=12.1a</code>, <code>MAX_JOBS=8</code> and <code>CPATH</code> pointing to <code>nvidia/cu13/include</code>, and the build checks the symbols.' },
        { t: 'Overlays and CPU tests', d: 'Each overlay is copied, its CPU test runs against the real vLLM file (fixtures, <code>GLM53_REQUIRE_VLLM=1</code>) and the patch is applied in the same order as at runtime. The two kpool patches share a single layer: kept separate, the image went over overlay2’s layer limit.' },
        { t: 'InstantTensor without dependencies', d: '<code>pip install --no-deps instanttensor==0.2.0</code>: a normal install would bring <code>nvidia-nccl-cu13==2.29.7</code> in on top of the image’s NCCL 2.30.7.' },
        { t: 'Load fixes for UMA', d: '<code>patch_cold_load_uma.py</code> (InstantTensor budget taken from <code>MemAvailable</code>, mmap clone in 64 KiB pages) and <code>patch_skip_cudagraph_profile.py</code>, each one after its own test. Both are idempotent: boot reapplies them with no effect.' },
        { t: 'The stamp on the last layer', d: '<code>ARG GLM53_RECIPE_STAMP</code> → <code>LABEL glm53.recipe.stamp</code>. Because it comes last, a rebuild that only changes the stamp is cheap.' },
        { t: 'At start: stamp × repository', d: 'The launcher recomputes the SHA-256 of Dockerfile + overlay/ + files/ + tests/ + ablit/ and compares it with the label. Switch the scenario in the selector to see each decision.' },
      ];
      const st = UI.stepper(fig, steps, { autoplay: 6000 });
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Scenario at start', options: [{ v: 'igual', label: 'nothing changed' }, { v: 'edit', label: 'edited an overlay' }, { v: 'skip', label: 'edited + SKIP_BUILD=1' }, { v: 'ghcr', label: 'GHCR changed' }], value: 'igual', onChange: (v) => { scen = v; st.stop(); st.go(7); } });
      const lock = (g, x, y, col) => { g.strokeStyle = col; g.lineWidth = 1.7; g.beginPath(); g.arc(x, y - 2, 3.6, Math.PI, 0); g.stroke(); V.box(g, x - 5.5, y - 2, 11, 8, { r: 2, fill: col }); };
      V.fig(fig, (g, w, h, t) => {
        fitH(fig, 440, 560, w);
        const C = V.C(); const narrow = w < 640; const pad = 14;
        const i = st.i, el = st.elapsed();
        const nShow = Math.min(7, i + 1);
        const tx = pad, tW = narrow ? w - 2 * pad : w * 0.47 - pad;
        const tBot = narrow ? 212 : h - 22;
        const bh = narrow ? 24 : Math.min(38, (tBot - 24) / 7 - 6);
        V.line(g, tx, tBot + 2, tx + tW, tBot + 2, { color: V.alpha(C.muted, 0.7), lw: 2 });
        for (let k = nShow; k < 7; k++) {
          const y = tBot - (k + 1) * (bh + 5), bx = tx + k * 3, bwid = tW - k * 6;
          g.setLineDash([4, 4]); V.box(g, bx, y, bwid, bh, { r: 7, fill: 'transparent', stroke: V.alpha(C.muted, 0.35), lw: 1 }); g.setLineDash([]);
          V.text(g, narrow ? BR[k].sh : BR[k].lab, bx + 12, y + bh / 2 + 1, { size: narrow ? 11 : 12, base: 'middle', color: C.muted, alpha: 0.45, maxW: bwid - 44 });
        }
        for (let k = 0; k < nShow; k++) {
          const B = BR[k], col = C[B.c];
          let y = tBot - (k + 1) * (bh + 5), a = 1;
          if (k === i && i < 7) { const e = V.ease.out(Math.min(1, el / 0.6)); y -= (1 - e) * 60; a = Math.max(0.05, e); }
          const cur = k === i || (i === 7 && k === 6);
          g.globalAlpha = a;
          const bx = tx + k * 3, bwid = tW - k * 6;
          V.box(g, bx, y, bwid, bh, { r: 7, fill: V.alpha(col, cur ? 0.34 : 0.15), stroke: V.alpha(col, cur ? 1 : 0.5), lw: cur ? 2 : 1 });
          V.text(g, narrow ? B.sh : B.lab, bx + 12, y + bh / 2 + 1, { size: narrow ? 11 : 12, base: 'middle', color: C.ink, weight: cur ? 700 : 400, maxW: bwid - 44 });
          if (B.lock) lock(g, bx + bwid - 18, y + bh / 2, col);
          if (B.stamp) { const sx = bx + bwid - 18, sy = y + bh / 2; const sc = i >= 6 ? 1 : 0.7; g.save(); g.translate(sx, sy); g.rotate(-0.25 + (i === 6 ? 0.25 * V.ease.out(Math.min(1, el / 0.8)) : 0.25)); g.scale(sc, sc); V.box(g, -9, -7, 18, 14, { r: 3, fill: 'transparent', stroke: col, lw: 2 }); V.text(g, '#', 0, 1, { size: 11, weight: 700, align: 'center', base: 'middle', color: col }); g.restore(); }
          g.globalAlpha = 1;
        }
        if (!narrow) V.text(g, 'Dockerfile layers: ' + nShow + ' of 7 (simplified)', tx, h - 6, { size: 11, color: C.muted });
        const dx = narrow ? pad : w * 0.5, dy = narrow ? tBot + 14 : 14, dw = narrow ? w - 2 * pad : w * 0.5 - pad, dh = h - dy - (narrow ? 8 : 14);
        V.box(g, dx, dy, dw, dh, { r: 12, fill: V.alpha(C.ink, 0.05), stroke: V.alpha(C.muted, 0.35) });
        if (i < 7) {
          V.text(g, 'DOCKERFILE · STEP ' + (i + 1), dx + 14, dy + 20, { size: 11, font: 'mono', weight: 700, color: C[BR[i].c] });
          CODE[i].forEach((ln, k) => {
            const e = V.ease.out(Math.min(1, Math.max(0, (el - k * 0.12) / 0.4)));
            V.text(g, ln, dx + 14, dy + 44 + k * (narrow ? 17 : 20), { size: narrow ? 11 : 12.5, font: 'mono', color: ln.trim().startsWith('#') ? C.muted : C.ink, alpha: e, maxW: dw - 28 });
          });
          const wy = dy + 44 + CODE[i].length * (narrow ? 17 : 20) + (narrow ? 8 : 16);
          if (wy + 30 < dy + dh) {
            V.line(g, dx + 14, wy - 8, dx + dw - 14, wy - 8, { color: V.alpha(C.muted, 0.3), lw: 1 });
            V.text(g, 'WHY', dx + 14, wy + 8, { size: 11, font: 'mono', weight: 700, color: C[BR[i].c] });
            para(g, WHY[i], dx + 14, wy + 26, dw - 28, { size: narrow ? 11.5 : 12.5, color: C.ink, lh: narrow ? 15 : 17 });
          }
        } else {
          const s = SC[scen], col = C[s.c];
          V.text(g, 'START.SH · ENSURE_IMAGE', dx + 14, dy + 20, { size: 11, font: 'mono', weight: 700, color: col });
          let y = dy + 38;
          const row = (lab, val, vc) => { V.text(g, lab, dx + 14, y + 11, { size: 11.5, color: C.muted }); chip(g, dx + dw - 14 - tw(g, val, 12, 'mono', 600) - 14, y, val, { size: 12, weight: 600, fill: V.alpha(vc, 0.16), stroke: vc, color: C.ink }); y += 30; };
          row('sha256(repository)', s.repo, C.a);
          row('local image LABEL', s.lab, C.b);
          if (s.pulled) row('pulled image LABEL', s.pulled, C.c);
          const pulse = 0.6 + 0.4 * Math.sin(t * 4);
          V.text(g, s.ok ? '=' : '≠', dx + dw / 2, y + 12, { size: 24, weight: 700, align: 'center', base: 'middle', color: col, alpha: pulse });
          y += 34;
          V.text(g, s.res, dx + 14, y, { size: narrow ? 12.5 : 14, weight: 700, color: col, maxW: dw - 28 });
          y += 20;
          y += para(g, s.sub, dx + 14, y, dw - 28, { size: 12, color: C.ink, lh: 16 }) + 6;
          if (y < dy + dh - 18) para(g, 'Then: test_exl3_overlay.py on the GPU (EXL3_SELFCHECK_GPU=1), unless SKIP_OVERLAY_VERIFY=1.', dx + 14, y, dw - 28, { size: 11.5, color: C.muted, lh: 15 });
        }
      });
    })();

    /* ============================================ F5b · fp8_ds_mla record and the 512→576 pad */
    (function () {
      const fig = root.querySelector('#fig-registro');
      const S = { nopad: false, t0: null };
      const ctl = UI.controls(fig);
      UI.toggle(ctl, { label: 'NoPE without pad (published image)', value: false, onChange: (v) => { S.nopad = v; S.t0 = null; } });
      // illustrative values: 32 latent cells (each one summarizes 16 dimensions) + 4 pad cells (64 dimensions)
      const rnd = V.rng(42), NL = 32, NP = 4, NT = NL + NP;
      const q = Array.from({ length: NL }, () => Math.round(rnd.normal() * 8) / 10);
      const k = Array.from({ length: NL }, () => Math.round(rnd.normal() * 8) / 10);
      const prefix = [0]; q.forEach((v, i) => prefix.push(prefix[i] + v * k[i]));
      const TOTAL = prefix[NL];
      V.fig(fig, (g, w, h, t) => {
        if (w < 160) return;
        const C = V.C(); const narrow = w < 600; const pad = 14, gw = w - 2 * pad;
        if (S.t0 == null) S.t0 = t;
        const el = t - S.t0;
        let y = 18;
        V.text(g, narrow ? 'fp8_ds_mla RECORD · 656 B/TOKEN/LAYER' : 'fp8_ds_mla RECORD · 656 B PER TOKEN, PER MLA LAYER, PER RANK', pad, y, { size: 11, font: 'mono', weight: 700, color: C.a, maxW: gw });
        y += 10;
        const th = narrow ? 30 : 36, sc = gw / 656;
        const segs = [[512, C.a], [16, C.d], [128, C.muted]];
        const fillB = V.clamp(((el % 6) / 3.4) * 656, 0, 656);
        let x = pad, acc = 0;
        segs.forEach(([n, col], i) => {
          const ww = n * sc;
          V.box(g, x, y, ww - 1, th, { r: 4, fill: V.alpha(col, 0.1), stroke: V.alpha(col, 0.75), lw: 1 });
          const f = V.clamp((fillB - acc) / n, 0, 1);
          if (f > 0) {
            g.save(); V.rr(g, x, y, ww - 1, th, 4); g.clip();
            g.fillStyle = V.alpha(col, i === 2 ? 0.22 : 0.78); g.fillRect(x, y, (ww - 1) * f, th);
            if (i === 2) { g.strokeStyle = V.alpha(C.muted, 0.75); g.lineWidth = 1; for (let qx = x - th; qx < x + (ww - 1) * f; qx += 6) { g.beginPath(); g.moveTo(qx, y + th); g.lineTo(qx + th, y); g.stroke(); } }
            g.restore();
          }
          if (i === 0) for (let b = 64; b < 512; b += 64) V.line(g, x + b * sc, y + th - 6, x + b * sc, y + th - 1, { color: V.alpha(C.bg, 0.8), lw: 1 });
          acc += n; x += ww;
        });
        const l0 = narrow ? '512 B · FP8 latent' : '512 B · NoPE latent, 512 × FP8 (1 B per dimension)', l0w = Math.min(512 * sc - 16, tw(g, l0, 11.5, 'sans', 700) + 14);
        V.box(g, pad + (512 * sc) / 2 - l0w / 2, y + th / 2 - 10, l0w, 20, { r: 6, fill: V.alpha(C.bg, 0.72) });
        V.text(g, l0, pad + (512 * sc) / 2, y + th / 2 + 1, { size: 11.5, weight: 700, align: 'center', base: 'middle', color: C.ink, maxW: l0w - 10 });
        if (128 * sc > 44) V.text(g, '128 B', pad + (528 + 64) * sc, y + th / 2 + 1, { size: 11.5, weight: 700, align: 'center', base: 'middle', color: C.ink });
        // legend arrows
        const ly = y + th + 18;
        const xs = pad + (512 + 8) * sc;
        V.line(g, xs, y + th + 2, xs, ly - 11, { color: C.d, lw: 1.2 });
        V.text(g, narrow ? '16 B · 4 FP32 scales (1 per 128 dims)' : '16 B · 4 FP32 scales, one per 128-dimension block', narrow ? pad + gw : xs + 4, ly, { size: narrow ? 11 : 11.5, align: 'right', color: C.d, maxW: gw });
        V.text(g, narrow ? '128 B · RoPE 64 × BF16, all zero' : '128 B · RoPE 64 × BF16: all zero in GLM (NoPE)', pad + gw, ly + 17, { size: narrow ? 11 : 11.5, align: 'right', color: C.muted, maxW: gw });
        y = ly + 38;
        y += para(g, 'actually needed: 528 B · the pad costs +128 B (+24%) · MLA page = 3584 × 656 = 2,351,104 B', pad, y, gw, { size: 11, font: 'mono', color: C.ink, lh: 15 }) + 14;
        // B · the product inside the kernel
        V.text(g, narrow ? 'SPARSE-MLA KERNEL · GEOMETRY 512 + 64 = 576' : 'FLASHINFER_MLA_SPARSE_SM120 · FIXED GEOMETRY: 512 (LATENT) + 64 (RoPE) = 576', pad, y, { size: 11, font: 'mono', weight: 700, color: C.b, maxW: gw });
        y += 12;
        const labW = 22, cw = (gw - labW) / NT, chh = narrow ? 16 : 22;
        const cyc = 7.5, ph = (el % cyc) / cyc;
        const cur = S.nopad ? -1 : Math.min(NT - 1, Math.floor((ph / 0.8) * NT));
        const done = !S.nopad && ph >= 0.8;
        [['q', q], ['k', k]].forEach(([nm, arr], row) => {
          const ry = y + row * (chh + 6);
          V.text(g, nm, pad + 6, ry + chh / 2 + 1, { size: 12.5, font: 'mono', weight: 700, base: 'middle', color: C.ink });
          for (let i = 0; i < NT; i++) {
            const cx = pad + labW + i * cw;
            if (i < NL) V.box(g, cx + 0.5, ry, cw - 1, chh, { r: 2, fill: V.cmap(arr[i] / 1.6, 'div'), stroke: i === cur ? C.ink : null, lw: 2 });
            else if (S.nopad && row === 0) { g.setLineDash([2, 2]); V.box(g, cx + 0.5, ry, cw - 1, chh, { r: 2, fill: V.alpha(C.red, 0.1 + 0.08 * Math.sin(t * 5)), stroke: C.red, lw: 1 }); g.setLineDash([]); if (cw > 9) V.text(g, '?', cx + cw / 2, ry + chh / 2 + 1, { size: 11, weight: 700, align: 'center', base: 'middle', color: C.red }); }
            else { V.box(g, cx + 0.5, ry, cw - 1, chh, { r: 2, fill: V.alpha(C.muted, 0.18), stroke: i === cur ? C.ink : V.alpha(C.muted, 0.5), lw: i === cur ? 2 : 1 }); if (cw > 9) V.text(g, '0', cx + cw / 2, ry + chh / 2 + 1, { size: 11, font: 'mono', align: 'center', base: 'middle', color: C.muted }); }
          }
        });
        // key under the columns
        const ky = y + 2 * (chh + 6) + 12;
        V.text(g, narrow ? 'latent: 32 cells of 16 dims' : 'latent: 32 cells of 16 dimensions (illustrative values)', pad + labW, ky, { size: 11, color: C.muted, maxW: NL * cw - 6 });
        V.text(g, 'pad', pad + labW + NL * cw + (NP * cw) / 2, ky, { size: 11, color: S.nopad ? C.red : C.muted, align: 'center' });
        y = ky + 24;
        let msg, col;
        if (S.nopad) { msg = '✕ “pe_dim must be 64 for fp8_ds_mla” (concat_and_cache_mla). The model has 0 RoPE dimensions, and the layout and the kernel only exist for 64 (576 in total): the published image dies on the first forward.'; col = C.red; }
        else if (cur < NL && !done) { msg = 'Σ qᵢ·kᵢ up to cell ' + (cur + 1) + ' = ' + br(prefix[cur + 1], 2); col = C.ink; }
        else if (!done) { msg = 'pad: qᵢ·kᵢ = 0 · 0 = 0. The sum stays at ' + br(TOTAL, 2); col = C.muted; }
        else { msg = '✓ q·k with 576 dimensions = q·k with 512 = ' + br(TOTAL, 2) + ': the score is exact, and the only price is memory.'; col = C.e; }
        y += para(g, msg, pad, y, gw, { size: 13, color: col, lh: 18, weight: 600 }) + 6;
        y += para(g, 'The patch: rope_pad = 64 when qk_rope_head_dim = 0; q = F.pad(q, (0, 64)); k_pe written as zeros. The same 576 geometry the kernel requires, the same result as the original NoPE MLA.', pad, y, gw, { size: 12, color: C.ink, lh: 16 });
        fitH(fig, Math.ceil(y + 6), Math.ceil(y + 6), w);
      });
    })();

    /* ============================================ F6 · overlay conveyor belt */
    (function () {
      const fig = root.querySelector('#fig-esteira');
      const P = [
        ['cold_load_uma', 'loader / InstantTensor'], ['skip_cudagraph_profile', 'memory profiling'], ['glm_video_placeholders', 'video (glm46v) + .pth'],
        ['suppress_stops_in_reasoning', 'detokenizer'], ['scheduler_decode_floor', 'scheduler.py (fair v5; v7 in #283)'], ['mamba_align_chunking', 'scheduler (Mamba alignment)'],
        ['glm5_drafter_group', 'kv_cache_utils.py'], ['hybrid_prefix_hit', 'kv_cache_coordinator'], ['apc_per_group_retention', 'kv_cache_coordinator'],
        ['apc_no_store', 'sampling_params / block_pool'], ['mamba_align_state_free', 'single_type_kv_cache_manager'], ['kv_capacity_log', 'kv_cache_utils.py (log)'],
        ['tool_choice_none', 'glm47 parser (bad_words)'], ['xgrammar_termination', 'structured output'], ['kpool_tail_slotmap', 'kpool slot mapping'],
        ['kpool_tail_seed_stride', 'kpool tail seed'], ['spinwait', 'shm_broadcast.py'], ['adaptive_k', 'spec decode (EMA)'],
        ['dense_fp8', 'exl3.py (FP8 Marlin)'], ['dflash2_exl3', 'qwen3_dflash2.py'], ['loadclone', 'weight_utils.py'],
        ['default_max_new_tokens', 'serving (pydantic)'], ['indexer_workspace', 'indexer workspace'], ['cache_reset', 'cache reset routes'], ['ablit', 'load_weights (hook)'],
      ];
      let scen = 'ok', t0 = null;
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Scenario', options: [{ v: 'ok', label: 'correct order' }, { v: 'swap', label: 'swap 8 ↔ 9' }, { v: 'miss', label: 'rank 1 without no. 18' }], value: 'ok', onChange: (v) => { scen = v; t0 = null; } });
      const DT = 0.5, TR = 1.9, HOLD = 3.4;
      const seq = (rank) => { const s = P.map((_, i) => i); if (scen === 'swap') { s[7] = 8; s[8] = 7; } return s.map((pi) => ({ pi, ghost: scen === 'miss' && rank === 1 && pi === 17 })); };
      let first = true;
      V.fig(fig, (g, w, h, t) => {
        if (t0 == null) { t0 = t - (first ? 7 : 0); first = false; }
        const C = V.C(); const narrow = w < 560; const pad = 14;
        const total = 25 * DT + TR + HOLD;
        let tc = (t - t0) % total;
        const failPos = scen === 'swap' ? 8 : -1;
        const failT = failPos >= 0 ? failPos * DT + TR : Infinity;
        const failed = tc >= failT;
        const tEff = Math.min(tc, failT);
        const bx0 = pad + (narrow ? 0 : 92), mw = narrow ? 56 : 84, mx = w - pad - mw, bs = narrow ? 18 : 24;
        const laneY = narrow ? [58, 212] : [52, 204];
        const names = ['rank 0 · head', 'rank 1 · worker'];
        const doneAll = tc >= 25 * DT + TR;
        [0, 1].forEach((rk) => {
          const y = laneY[rk], S0 = seq(rk);
          if (narrow) V.text(g, names[rk].toUpperCase(), pad, y - 26, { size: 11, font: 'mono', weight: 700, color: rk ? C.b : C.a });
          else { V.text(g, names[rk].split(' · ')[0].toUpperCase(), pad, y - 2, { size: 12, font: 'mono', weight: 700, color: rk ? C.b : C.a }); V.text(g, names[rk].split(' · ')[1], pad, y + 14, { size: 11, color: C.muted }); }
          V.box(g, bx0, y - 15, mx - bx0 - 4, 30, { r: 15, fill: V.alpha(C.muted, 0.12), stroke: V.alpha(C.muted, 0.35) });
          if (!failed) V.line(g, bx0 + 12, y, mx - 16, y, { color: V.alpha(C.muted, 0.4), lw: 2, dash: [3, 9], dashOffset: -t * 30 });
          let last = -1;
          S0.forEach((it, k) => {
            const launch = k * DT, frac = (tEff - launch) / TR;
            if (frac >= 1) { if (!it.ghost) last = k; return; }
            if (frac < 0) return;
            const x = bx0 + 6 + frac * (mx - bx0 - bs - 14);
            const col = C.series[it.pi % 6];
            if (it.ghost) { g.setLineDash([3, 3]); V.box(g, x, y - bs / 2, bs, bs, { r: 5, fill: 'transparent', stroke: V.alpha(C.red, 0.8), lw: 1.5 }); g.setLineDash([]); V.text(g, '?', x + bs / 2, y + 1, { size: 11, weight: 700, align: 'center', base: 'middle', color: C.red }); }
            else { V.box(g, x, y - bs / 2, bs, bs, { r: 5, fill: V.alpha(col, 0.85) }); V.text(g, String(it.pi + 1), x + bs / 2, y + 1, { size: narrow ? 10.5 : 11, weight: 700, align: 'center', base: 'middle', color: onFill(C) }); }
          });
          // machine
          const flash = last >= 0 && tEff - (last * DT + TR) < 0.3;
          const mcol = failed ? C.red : flash ? C.series[S0[last].pi % 6] : C.muted;
          V.box(g, mx, y - 26, mw, 52, { r: 10, fill: V.alpha(mcol, failed || flash ? 0.28 : 0.1), stroke: mcol, lw: failed ? 2.5 : 1.5 });
          V.text(g, 'vLLM', mx + mw / 2, y - 4, { size: 12, weight: 700, align: 'center', color: C.ink });
          V.text(g, failed ? '✕ anchor' : narrow ? 'pkgs' : 'site-pkgs', mx + mw / 2, y + 12, { size: 11, align: 'center', font: 'mono', color: failed ? C.red : C.muted, maxW: mw - 6 });
          // progress
          const sq = Math.min(12, (mx - bx0 - 4) / 25 - 2), sy = y + 24;
          S0.forEach((it, k) => {
            const applied = !it.ghost && tEff >= k * DT + TR;
            const failHere = failed && k === failPos;
            const xx = bx0 + k * (sq + 2);
            let fill = V.alpha(C.muted, 0.18);
            if (applied) fill = V.alpha(C.series[it.pi % 6], 0.85);
            if (it.ghost && tEff >= k * DT + TR) fill = V.alpha(C.red, 0.5);
            if (failHere) fill = C.red;
            V.box(g, xx, sy, sq, sq, { r: 2.5, fill });
          });
          // ticker
          let msg = '', mc = C.ink;
          if (failed) { const pi = S0[failPos].pi; msg = (pi + 1) + ' · patch_' + P[pi][0] + '.py → anchor not found: aborts (fail-closed)'; mc = C.red; }
          else if (last >= 0) { const pi = S0[last].pi; msg = (pi + 1) + ' · patch_' + P[pi][0] + '.py → ' + P[pi][1]; }
          else msg = 'waiting: internal script building ARGS…';
          V.text(g, msg, bx0, sy + sq + 18, { size: 11.5, font: 'mono', color: mc, maxW: w - bx0 - pad });
        });
        // overall message
        const mwid = w - 2 * pad;
        const MSG = {
          fail: 'Both ranks abort at the same patch: a fail-closed overlay would rather not come up at all than come up with the KV coordinator half-assembled.',
          miss: 'Rank 1 finished without no. 18 (adaptive_k). With GLM53_ADAPTIVE_K=ema, rank 0 captures decode graphs for 3, 5 and 8 positions and rank 1 only the 8-position ones: the collectives diverge, and NCCL hangs after PIECEWISE (interpretation, based on the README). With adaptive-k off, the default, the gap goes unnoticed until someone turns it on.',
          ok: '25 of 25 on both ranks, same bytes and same order: exec vllm serve.',
          run: 'Same list, same order, in both containers. Each patch checks its anchor and is a no-op if it was already applied.',
        };
        let txt, col;
        if (failed) { txt = MSG.fail; col = C.red; }
        else if (scen === 'miss' && doneAll) { txt = MSG.miss; col = C.red; }
        else if (doneAll) { txt = MSG.ok; col = C.e; }
        else { txt = MSG.run; col = C.muted; }
        const lines = (s2) => wrap(g, s2, mwid - 24, 12.5, 'sans', 500).length;
        const maxL = Math.max(lines(MSG.fail), lines(MSG.miss), lines(MSG.run));
        const nL = lines(txt), boxH = nL * 17 + 20, my = laneY[1] + (narrow ? 70 : 62);
        fitH(fig, Math.max(380, Math.ceil(my + maxL * 17 + 28)), Math.max(380, Math.ceil(my + maxL * 17 + 28)), w);
        V.box(g, pad, my, mwid, boxH, { r: 10, fill: V.alpha(col, 0.08), stroke: V.alpha(col, 0.5) });
        para(g, txt, pad + 12, my + 22, mwid - 24, { size: 12.5, color: C.ink, lh: 17, weight: 500 });
        if (scen === 'miss' && doneAll) { const zx = mx + mw / 2, z0 = laneY[0] + 26, z1 = laneY[1] - 26; g.strokeStyle = C.red; g.lineWidth = 2.5; g.beginPath(); for (let k = 0; k <= 8; k++) { const yy = V.lerp(z0, z1, k / 8); const xx = zx + (k % 2 ? 7 : -7); if (k) g.lineTo(xx, yy); else g.moveTo(xx, yy); } g.stroke(); }
      });
    })();

    /* ============================================ F7 · speculative-config and graphs */
    (function () {
      const fig = root.querySelector('#fig-spec');
      const S = { seqs: 4, k: 7, ada: false };
      const ctl = UI.controls(fig);
      UI.slider(ctl, { label: 'MAX_NUM_SEQS', min: 1, max: 8, step: 1, value: 4, onInput: (v) => { S.seqs = v; } });
      UI.seg(ctl, { label: 'DFLASH_TOKENS (k)', options: [3, 5, 7], value: 7, onChange: (v) => { S.k = v; } });
      UI.toggle(ctl, { label: 'adaptive-k (set 2,4,7)', value: false, onChange: (v) => { S.ada = v; } });
      const calc = () => {
        const sizes = new Set([1, 2, 4, 8, 16, 24, 32]);
        const q = S.k + 1; let lens = [q];
        if (S.ada) {
          const L = new Set([q]); [2, 4, 7].forEach((kk) => { if (kk + 1 > 0 && kk + 1 <= q) L.add(kk + 1); });
          lens = Array.from(L).sort((a, b) => a - b);
          for (let n = 1; n <= S.seqs; n++) lens.forEach((qq) => sizes.add(n * qq));
        }
        const need = new Set(); for (let n = 1; n <= S.seqs; n++) lens.forEach((qq) => need.add(n * qq));
        return { sizes, need, lens, q };
      };
      const face = (g, x, y, r, col, C, mood) => {
        g.beginPath(); g.arc(x, y, r, 0, Math.PI * 2); g.fillStyle = col; g.fill();
        const ec = onFill(C);
        V.dot(g, x - r * 0.35, y - r * 0.18, Math.max(1.3, r * 0.13), ec); V.dot(g, x + r * 0.35, y - r * 0.18, Math.max(1.3, r * 0.13), ec);
        g.strokeStyle = ec; g.lineWidth = 1.4; g.beginPath();
        if (mood === 'q') { V.text(g, '?', x, y + r * 0.42, { size: Math.max(9, r * 0.8), weight: 700, align: 'center', base: 'middle', color: ec }); }
        else { g.arc(x, y + r * 0.1, r * 0.42, 0.15 * Math.PI, 0.85 * Math.PI); g.stroke(); }
      };
      V.fig(fig, (g, w, h, t) => {
        if (w < 160) return; // canvas has no width yet (e.g., figure being expanded)
        const C = V.C(); const narrow = w < 600; const pad = 14;
        const R = calc();
        // A · JSON
        V.text(g, '--speculative-config', pad, 18, { size: 11, font: 'mono', color: C.muted, weight: 700 });
        const js = [['"method":"dflash"', C.c], ['"model":"…DFlash2@dc77ff1c"', C.b], ['"num_speculative_tokens":' + S.k, C.d], ['"kv_cache_dtype":"auto"', C.a], ['"draft_sample_method":"probabilistic"', C.e], ['"rejection_sample_method":"standard"', C.e], ['"draft_tensor_parallel_size":2', C.f]];
        let cx = pad, cy = 26;
        js.forEach(([s, col]) => {
          const wd = tw(g, s, 11, 'mono', 500) + 14;
          if (cx + wd > w - pad) { cx = pad; cy += 24; }
          chip(g, cx, cy, s, { size: 11, fill: V.alpha(col, 0.14), stroke: V.alpha(col, 0.7), color: C.ink, h: 19 });
          cx += wd + 6;
        });
        let y = cy + 38;
        // B · block
        const bW = narrow ? w - 2 * pad : (w - 2 * pad) * 0.5;
        V.text(g, 'DFLASH2 BLOCK · 1 ANCHOR + ' + S.k + ' MASKS', pad, y, { size: 11, font: 'mono', weight: 700, color: C.c });
        const nTok = S.k + 1, rr = Math.max(4, Math.min(narrow ? 15 : 20, (bW - 20) / nTok / 2 - 3));
        const ph = (t % 4.5) / 4.5;
        for (let j = 0; j < nTok; j++) {
          const x = pad + rr + 4 + j * (rr * 2 + 6), yy = y + 26 + rr;
          const bob = Math.sin(t * 3 + j) * 1.5;
          if (j === 0) face(g, x, yy + bob, rr, C.d, C, 'ok');
          else {
            const filled = ph > 0.3;
            face(g, x, yy + bob, rr, filled ? C.b : V.alpha(C.c, 0.72 + 0.28 * Math.sin(t * 6 + j)), C, filled ? 'ok' : 'q');
            if (ph > 0.55 && ph - 0.55 > (j / nTok) * 0.35) V.text(g, '✓', x, yy - rr - 4, { size: 11, weight: 700, align: 'center', color: C.e });
          }
        }
        const lblY = y + 26 + rr * 2 + 20;
        V.text(g, ph < 0.3 ? 'draft: masks waiting…' : ph < 0.55 ? 'block diffusion: all filled in at once' : 'the target verifies all ' + nTok + ' positions in one step', pad, lblY, { size: 11.5, color: C.ink, maxW: bW - 10 });
        // C · decode step
        const cX = narrow ? pad : pad + bW + 16, cY = narrow ? lblY + 22 : y;
        const tot = S.seqs * R.q;
        V.text(g, 'DECODE STEP · ' + S.seqs + ' SEQ × ' + R.q + ' = ' + tot + ' TOKENS', cX, cY, { size: 11, font: 'mono', weight: 700, color: C.b, maxW: w - cX - pad });
        const avail = w - cX - pad, cell = Math.max(7, Math.min(narrow ? 12 : 16, (avail - 10) / 16 - 2));
        for (let n = 0; n < S.seqs; n++) for (let j = 0; j < R.q; j++) {
          const on = ((t * 2.2) % (S.seqs + 1)) > n;
          V.box(g, cX + j * (cell + 2), cY + 12 + n * (cell + 2), cell, cell, { r: 2, fill: j === 0 ? V.alpha(C.d, on ? 0.95 : 0.4) : V.alpha(C.b, on ? 0.9 : 0.35) });
        }
        const gridBottom = cY + 12 + S.seqs * (cell + 2);
        y = Math.max(lblY, gridBottom) + 30;
        // D · capture sizes
        V.text(g, 'CAPTURE SIZES (LAUNCHER RULE)' + (S.ada ? ' · LENGTHS ' + R.lens.join(', ') : ''), pad, y, { size: 11, font: 'mono', weight: 700, color: C.a, maxW: w - 2 * pad });
        const all = Array.from(new Set([...R.sizes, ...R.need])).sort((a, b) => a - b);
        let px = pad, py = y + 10; let missing = 0;
        all.forEach((s) => {
          const inS = R.sizes.has(s), isN = R.need.has(s);
          if (isN && !inS) missing++;
          const lab = String(s), wd = tw(g, lab, 11, 'mono', 600) + 14;
          if (px + wd > w - pad) { px = pad; py += 24; }
          const o = isN && inS ? { fill: V.alpha(C.e, 0.3), stroke: C.e, color: C.ink } : isN ? { fill: V.alpha(C.red, 0.18), stroke: C.red, color: C.red, dash: [3, 2] } : { fill: V.alpha(C.muted, 0.1), stroke: V.alpha(C.muted, 0.5), color: C.muted };
          chip(g, px, py, lab, Object.assign({ size: 11, weight: 600, h: 19 }, o));
          px += wd + 5;
        });
        y = py + 38;
        y += para(g, missing ? '✕ ' + missing + ' decode size(s) missing from the list: pass an explicit --cudagraph-capture-sizes in EXTRA_ARGS.' : '✓ every decode step (green) has a graph in the list.', pad, y, w - 2 * pad, { size: 12, color: missing ? C.red : C.e, weight: 600, lh: 16 });
        if (S.ada) y += para(g, 'adaptive-k: each request verifies 2, 4 or 7 drafts (queries of 3, 5 or 8) according to its EMA; the whole batch uses the smallest k, because the FULL sparse-MLA and KDA graphs require uniform decode.', pad, y + 2, w - 2 * pad, { size: 12, color: C.ink, lh: 16 }) + 2;
        // E · TEMP_ROWS
        y += 18;
        const need = S.seqs * R.q, cap = 32, gw = w - 2 * pad, scale = gw / 72;
        y += para(g, 'EXL3_TEMP_ROWS_FUSED = 32 (E3) · needs ≥ ' + S.seqs + ' × ' + R.q + ' = ' + need, pad, y, gw, { size: 11.5, font: 'mono', color: C.ink, lh: 15 }) - 15;
        const gy = y + 10;
        V.box(g, pad, gy, gw, 14, { r: 7, fill: V.alpha(C.muted, 0.14) });
        V.box(g, pad, gy, Math.min(gw, need * scale), 14, { r: 7, fill: need <= cap ? V.alpha(C.e, 0.8) : V.alpha(C.red, 0.75) });
        V.line(g, pad + cap * scale, gy - 5, pad + cap * scale, gy + 19, { color: C.ink, lw: 2 });
        V.text(g, '32', pad + cap * scale, gy + 32, { size: 11, font: 'mono', align: 'center', color: C.ink });
        y = gy + 50;
        y += para(g, 'Why N·(k+1) rows? With top-8, each token visits a given expert at most once; in the worst case, all ' + need + ' tokens in the step pick the same expert, which receives ' + need + ' rows.', pad, y, gw, { size: 11.5, color: C.muted, lh: 15 }) + 2;
        y += para(g, need <= cap ? 'Fits: the whole decode is a single graph-safe exl3_moe launch.' : 'Exceeds 32: TEMP_ROWS falls short of the README rule, and decode no longer fits in a graph-safe launch.', pad, y, gw, { size: 12, color: need <= cap ? C.e : C.red, lh: 16, weight: 600 });
        // F · measured acceptance
        let fy = y + 22;
        V.text(g, 'HOW MUCH OF THE BLOCK IS ACCEPTED (MEASURED, TP2, k=7, 2026-08-30)', pad, fy, { size: 11, font: 'mono', weight: 700, color: C.f, maxW: gw });
        const acc = [['structured', 6.71, '0.96 acceptance'], ['prose', 2.39, '0.34 acceptance']];
        const labW = narrow ? 84 : 100, bw2 = gw - labW - (narrow ? 76 : 250);
        acc.forEach(([lab, v, note], k) => {
          const yy = fy + 14 + k * 24;
          V.text(g, lab, pad, yy + 11, { size: 12, color: C.ink });
          V.box(g, pad + labW, yy, bw2, 16, { r: 5, fill: V.alpha(C.muted, 0.14) });
          for (let q = 1; q < 8; q++) V.line(g, pad + labW + (q / 8) * bw2, yy + 2, pad + labW + (q / 8) * bw2, yy + 14, { color: V.alpha(C.bg, 0.8), lw: 1 });
          V.box(g, pad + labW, yy, (v / 8) * bw2, 16, { r: 5, fill: V.alpha(k ? C.c : C.a, 0.8) });
          V.text(g, br(v, 2) + ' of 8' + (narrow ? '' : ' · ' + note), pad + labW + bw2 + 8, yy + 12, { size: 11.5, font: 'mono', color: C.ink });
        });
        fy += 14 + 2 * 24 + 8;
        if (S.k !== 7) fy += para(g, 'Measured only with k=7; on TP4, k=3 accepted 53% versus 30% with k=7.', pad, fy + 6, gw, { size: 11.5, color: C.muted, lh: 15 });
        fitH(fig, Math.ceil(fy + 10), Math.ceil(fy + 10), w);
      });
    })();

    /* ============================================ F8 · boot timeline */
    (function () {
      const fig = root.querySelector('#fig-boot');
      const PH = [
        { a: 0, b: 6, lab: 'container → 1st log', sh: '6s', c: 'muted' },
        { a: 6, b: 32, lab: 'init Python · engine · NCCL', sh: 'init', c: 'b' },
        { a: 32, b: 71, lab: 'weight stream (whole checkpoint)', sh: 'weights', c: 'd' },
        { a: 71, b: 78, lab: 'KV profile', sh: 'KV', c: 'a' },
        { a: 78, b: 89, lab: 'graphs + warmup', sh: 'graphs', c: 'c' },
        { a: 89, b: 99, lab: 'API comes up', sh: 'API', c: 'e' },
      ];
      const LOGS = [
        [0, 'starting worker on zurih@10.0.0.2 … · starting head (vLLM API :8888)'],
        [6, '[glm53-exl3-head] launching: vllm serve /root/.cache/huggingface/hub/…'],
        [32, 'InstantTensor: each rank reads the checkpoint at 5.5–5.9 GB/s'],
        [71, 'Model loading took 79.65 GiB'],
        [74, 'exl3 e2 diag schema=2 … effective_tier=grouped'],
        [78, '[glm53-kv-capacity-log] … · GPU KV cache size: …'],
        [89, 'CUDA graphs captured: 1 2 4 8 16 24 32'],
        [99, 'health check passed after 99s — server is up'],
        [104, 'boot shape warmup + canary: "Reply with OK." → OK'],
        [110, 'READY'],
      ];
      const DIE = [[40, 'worker: process died (simulation)'], [70, "worker container 'glm53-exl3-worker' not running on zurih@10.0.0.2 (3 consecutive checks)"]];
      const CUTS = [
        ['.pth imported vLLM in every interpreter', 100, 6, 'container → 1st log'],
        ['NVIDIA JIT cache not persisted', 30, 0, 'DFlash2 capture'],
        ['dry capture thrown away (CG_ESTIMATE=0)', 18, 7, 'KV profile'],
        ['graceful stop → parallel kill-first', 15, 0, '~−15 s'],
      ];
      const S = { t: 0, manual: false, die: false, cuts: true, t0: null, first: true };
      const ctl = UI.controls(fig);
      UI.slider(ctl, { label: 'time', min: 0, max: 112, step: 1, value: 0, fmt: (v) => v + ' s', onInput: (v) => { S.t = v; S.manual = true; } });
      UI.button(ctl, '▶ play', () => { S.manual = false; S.t0 = null; }, { primary: true });
      UI.toggle(ctl, { label: 'simulate: worker dies at 40 s', value: false, onChange: (v) => { S.die = v; S.t0 = null; } });
      UI.toggle(ctl, { label: 'what was cut (259 → 122 s)', value: true, onChange: (v) => { S.cuts = v; } });
      V.fig(fig, (g, w, h, t) => {
        fitH(fig, 450, 530, w);
        const C = V.C(); const narrow = w < 600; const pad = 14;
        if (!S.manual) { if (S.t0 == null) { S.t0 = t - (S.first ? 7 : 0); S.first = false; } S.t = Math.min(112, ((t - S.t0) * 8) % 124); }
        const T = S.t;
        const ax0 = narrow ? pad : pad + 104, ax1 = w - pad, TX = (s) => ax0 + (s / 112) * (ax1 - ax0);
        V.text(g, 'restart 259 → 122 s · /health 230 → 99 s (2026-09-19)', pad, 16, { size: 11, font: 'mono', color: C.muted, maxW: w - 2 * pad });
        const lh = narrow ? 30 : 34, gapL = narrow ? 26 : 14, y0 = narrow ? 44 : 30;
        const lanes = ['launcher (host)', 'head · rank 0', 'worker · rank 1'];
        const LY = (i) => y0 + i * (lh + gapL);
        const abortT = 70;
        const dead = S.die && T >= 40;
        lanes.forEach((ln, i) => {
          const y = LY(i);
          if (narrow) V.text(g, ln, pad, y - 6, { size: 11, weight: 600, color: C.ink });
          else V.text(g, ln, pad, y + lh / 2 + 4, { size: 11.5, weight: 600, color: C.ink });
          V.box(g, ax0, y, ax1 - ax0, lh, { r: 7, fill: V.alpha(C.muted, 0.07) });
        });
        const barLab = (a, b, y, lab, sh, col, alpha) => {
          const x0 = TX(a), x1 = TX(b);
          V.box(g, x0 + 1, y + 3, Math.max(2, x1 - x0 - 2), lh - 6, { r: 5, fill: V.alpha(col, alpha) });
          const full = tw(g, lab, 11, 'sans', 600) + 10 < x1 - x0 ? lab : tw(g, sh, 11, 'sans', 600) + 6 < x1 - x0 ? sh : '';
          if (full) V.text(g, full, (x0 + x1) / 2, y + lh / 2 + 1, { size: 11, weight: 600, align: 'center', base: 'middle', color: C.ink });
        };
        [1, 2].forEach((ri) => {
          const y = LY(ri);
          PH.forEach((p) => {
            if (ri === 2 && p.sh === 'API') return;
            let a = p.a, b = p.b;
            const stopAt = S.die ? (ri === 2 ? 40 : abortT) : Infinity;
            if (a >= stopAt) return;
            b = Math.min(b, stopAt);
            const col = p.c === 'muted' ? C.muted : C[p.c];
            const act = T >= a && T < b;
            barLab(a, b, y, p.lab, p.sh, col, T >= b ? 0.55 : act ? 0.85 : 0.18);
          });
          if (ri === 2 && !S.die) { g.setLineDash([4, 4]); V.box(g, TX(89) + 1, y + 3, TX(99) - TX(89) - 2, lh - 6, { r: 5, fill: 'transparent', stroke: V.alpha(C.b, 0.7) }); g.setLineDash([]); if (!narrow) V.text(g, 'no API', (TX(89) + TX(99)) / 2, y + lh / 2 + 1, { size: 11, align: 'center', base: 'middle', color: C.muted }); }
          if (ri === 2 && S.die && T >= 40) { g.save(); V.rr(g, TX(40), y + 3, ax1 - TX(40), lh - 6, 5); g.clip(); g.strokeStyle = V.alpha(C.red, 0.6); g.lineWidth = 1.5; for (let q = TX(40) - lh; q < ax1; q += 8) { g.beginPath(); g.moveTo(q, y + lh); g.lineTo(q + lh, y); g.stroke(); } g.restore(); V.text(g, '✕', TX(40), y + lh / 2 + 1, { size: 16, weight: 700, align: 'center', base: 'middle', color: C.red }); }
        });
        // launcher lane
        const yL = LY(0);
        const pollEnd = S.die ? Math.min(T, abortT) : Math.min(T, 99);
        for (let s = 1; s <= pollEnd; s += narrow ? 3 : 2) V.dot(g, TX(s), yL + lh - 7, 1.6, V.alpha(C.ink, 0.5));
        V.text(g, narrow ? 'docker run · poll /health 1 s' : 'docker run → poll /health every 1 s', TX(0) + 4, yL + 13, { size: 11, color: C.muted, maxW: TX(96) - TX(0) });
        if (S.die) {
          [50, 60, 70].forEach((s, k) => { if (T >= s) V.text(g, '✕' + (k + 1), TX(s), yL + lh / 2 + 1, { size: 12, weight: 700, align: 'center', base: 'middle', color: C.red }); });
          if (T >= abortT) { const lx = TX(abortT) + 20; V.text(g, narrow ? 'aborted' : 'aborted in ~30 s (not 3600 s)', lx, yL + lh / 2 + 1, { size: 11, weight: 700, base: 'middle', color: C.red, maxW: ax1 - lx - 2 }); }
        } else {
          if (T >= 99) { V.text(g, '/health ✓', TX(99), yL + lh / 2 + 1, { size: 11, weight: 700, align: 'center', base: 'middle', color: C.e }); }
          g.setLineDash([4, 3]); V.box(g, TX(100), yL + 5, TX(109) - TX(100), lh - 10, { r: 5, fill: V.alpha(C.f, T >= 100 ? 0.25 : 0.06), stroke: V.alpha(C.f, 0.8) }); g.setLineDash([]);
          if (T >= 110) V.text(g, '★', TX(110.5), yL + lh / 2 + 1, { size: 14, align: 'center', base: 'middle', color: C.d });
        }
        // axis
        const ay = LY(2) + lh + 16;
        [0, 20, 40, 60, 80, 99].forEach((s) => { V.line(g, TX(s), ay - 8, TX(s), ay - 3, { color: C.muted, lw: 1 }); V.text(g, s + 's', TX(s), ay + 9, { size: 11, align: 'center', font: 'mono', color: C.muted }); });
        if (!narrow && !S.die) V.text(g, 'warmup + canary (duration not published) ▾', ax1, 16, { size: 11, align: 'right', color: C.f });
        // playhead
        const phx = TX(T);
        V.line(g, phx, y0 - 6, phx, LY(2) + lh + 4, { color: C.ink, lw: 1.6 });
        V.glow(g, phx, y0 - 6, 9, C.d, 1);
        // log
        const logs = S.die ? LOGS.filter((l) => l[0] < 40).concat(DIE) : LOGS;
        let cur = logs[0]; logs.forEach((l) => { if (T >= l[0]) cur = l; });
        const ly = ay + 22, lbh = narrow ? 44 : 30;
        V.box(g, pad, ly, w - 2 * pad, lbh, { r: 8, fill: V.alpha(C.ink, 0.06), stroke: V.alpha(C.muted, 0.35) });
        const lc = S.die && T >= 40 ? C.red : C.ink;
        if (narrow) para(g, '[' + cur[0] + 's] ' + cur[1], pad + 10, ly + 18, w - 2 * pad - 20, { size: 11, font: 'mono', color: lc, lh: 15 });
        else V.text(g, '[' + String(cur[0]).padStart(3, ' ') + 's] ' + cur[1], pad + 10, ly + lbh / 2 + 1, { size: 12, font: 'mono', base: 'middle', color: lc, maxW: w - 2 * pad - 20 });
        // bottom panel
        let py = ly + lbh + 20;
        py += para(g, 'In the 99 s: 26 s of init, 39 s of weights (each rank reads the whole checkpoint), 7 s of KV profile, 11 s of graphs + warmup, ~10 s of API. The next lever would be a loader that reads only each rank’s half (~15–18 s).', pad, py, w - 2 * pad, { size: 12, color: C.ink, lh: 16 }) + 14;
        if (S.cuts) {
          V.text(g, 'WHAT WAS CUT (BEFORE → AFTER, s)', pad, py, { size: 11, font: 'mono', weight: 700, color: C.c });
          py += 10;
          const rowH = narrow ? 30 : 26, labW = narrow ? 0 : 290, bx = pad + labW, bw = w - pad - bx - 60;
          CUTS.forEach(([lab, before, after, note], k) => {
            const yy = py + k * rowH;
            if (narrow) V.text(g, lab, pad, yy + 10, { size: 11, color: C.ink, maxW: w - 2 * pad });
            else V.text(g, lab, pad, yy + 14, { size: 11.5, color: C.ink, maxW: labW - 10 });
            const by = narrow ? yy + 14 : yy + 4, bhh = narrow ? 10 : 14;
            V.box(g, bx, by, (before / 100) * bw, bhh, { r: 4, fill: V.alpha(C.red, 0.18), stroke: V.alpha(C.red, 0.5) });
            if (after > 0) V.box(g, bx, by, (after / 100) * bw, bhh, { r: 4, fill: V.alpha(C.e, 0.85) });
            V.text(g, k === 3 ? note : before + ' → ' + after, bx + (before / 100) * bw + 8, by + bhh / 2 + 1, { size: 11, font: 'mono', base: 'middle', color: C.ink });
          });
        }
      });
    })();

    /* ============================================ F9 · measured numbers */
    (function () {
      const fig = root.querySelector('#fig-numeros');
      let view = 'decode', tv = 0;
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Metric', options: [{ v: 'decode', label: 'Decode' }, { v: 'prefill', label: 'Cold prefill' }, { v: 'conc', label: 'Concurrency' }, { v: 'kld', label: 'Quality (KLD)' }], value: 'decode', onChange: (v) => { view = v; tv = 0; } });
      const DEC = [
        { hd: 'PROSE · 1 SEQUENCE (tok/s)', c: 'c' },
        { l: 'DFlash2 k=7', v: 27.1, tag: '08-30 · lab · accepts 0.34' },
        { l: '+ adaptive-k + dense FP8', v: 32.1, tag: '09-08' },
        { l: '+ cooperative MoE', v: 36.1, tag: '09-17 · sparkDash' },
        { hd: 'STRUCTURED · 1 SEQUENCE (tok/s)', c: 'a' },
        { l: 'DFlash2 k=7', v: 65.1, tag: '08-30 · lab · accepts 0.96' },
        { l: 'arm A (no coop)', v: 72.03, s: '72.03', tag: '09-16 · A/B/A' },
        { l: '+ cooperative MoE geo 1', v: 77.29, s: '77.29', tag: '09-16 · A/B/A' },
        { hd: 'MTP BASELINE · PROMPT NOT STATED', c: 'muted' },
        { l: 'MTP k=2 (rollback)', v: 24.6, s: '~24.6', tag: 'README baseline' },
      ];
      const PRE = [
        { hd: '8k PROMPT · tok/s (prompt ÷ TTFT)', c: 'd' },
        { l: 'MNBT 1024', v: 772, tag: '08-29' },
        { l: 'MNBT 2048', v: 938, tag: '08-30' },
        { l: 'E2 (PR77) · MNBT 2048', v: 1132.32, s: '1132.32', tag: '09-01' },
        { l: 'E3 · MNBT 7168', v: 1492.1, s: '1492.1', tag: '09-07' },
        { hd: '256k PROMPT', c: 'b' },
        { l: 'MNBT 1024', v: 936, tag: '08-29' },
        { l: 'MNBT 2048', v: 973, tag: '08-30' },
        { l: 'E3 · MNBT 7168', v: 1516.8, s: '1516.8', tag: '09-07' },
      ];
      const CONC = { x: [1, 2, 3, 4], stream: [36.1, 25.0, 22.3, 19.4], agg: [37.1, 51.1, 65.8, 75.3], ttft: [333, 365, 405, 401] };
      const KLD = [
        { l: 'TR3 K6 (6 bpw)', gb: 254, k: 0.013723, side: 'l' },
        { l: 'official FP8, cross-stack', gb: 328, k: 0.020615, side: 'l' },
        { l: 'EXL3 4 bpw (served)', gb: 176, k: 0.024555, side: 'r', hi: true },
        { l: 'official FP8, v44 stack', gb: 328, k: 0.024629, side: 'l' },
        { l: 'NVFP4 (v44)', gb: 180, k: 0.060535, side: 'r' },
      ];
      const hbars = (g, w, h, C, rows, max, grow, narrow) => {
        const pad = 14, labW = narrow ? 0 : 196, tagW = narrow ? 0 : 150;
        const x0 = pad + labW, x1 = w - pad - tagW - 44;
        const rowH = narrow ? 36 : 30;
        let y = 18, hdCol = C.ink;
        rows.forEach((r) => {
          if (r.hd) { if (y > 18) y += 10; hdCol = C[r.c]; V.text(g, r.hd, pad, y + 4, { size: 11, font: 'mono', weight: 700, color: hdCol }); y += 20; return; }
          const by = narrow ? y + 13 : y;
          if (narrow) V.text(g, r.l + ' · ' + r.tag, pad, y + 7, { size: 11, color: C.ink, maxW: w - 2 * pad });
          else V.text(g, r.l, pad, y + 12, { size: 12, color: C.ink, maxW: labW - 8 });
          const bw = ((r.v / max) * (x1 - x0)) * grow;
          V.box(g, x0, by, Math.max(2, bw), narrow ? 14 : 18, { r: 5, fill: V.alpha(hdCol, 0.8) });
          V.text(g, r.s || br(r.v, r.v > 100 ? 0 : 1), x0 + bw + 6, by + (narrow ? 7 : 9) + 1, { size: 11.5, font: 'mono', weight: 700, base: 'middle', color: C.ink });
          if (!narrow) V.text(g, r.tag, w - pad, y + 12, { size: 11, align: 'right', color: C.muted, maxW: tagW });
          y += rowH;
        });
        return y;
      };
      V.fig(fig, (g, w, h, t, dt) => {
        tv += dt; const grow = V.ease.out(Math.min(1, tv / 0.8));
        fitH(fig, 390, 470, w);
        const C = V.C(); const narrow = w < 600; const pad = 14;
        if (view === 'decode') {
          const y = hbars(g, w, h, C, DEC, 80, grow, narrow);
          if (h - y > 24) para(g, 'Same recipe, different dates and harnesses: compare within each group.', pad, y + 10, w - 2 * pad, { size: 11.5, color: C.muted });
        } else if (view === 'prefill') {
          const y = hbars(g, w, h, C, PRE, 1600, grow, narrow);
          if (h - y > 24) para(g, 'Cold prompts with a unique salt (no APC hit). Different dates and configs: an evolution, not an A/B.', pad, y + 10, w - 2 * pad, { size: 11.5, color: C.muted });
        } else if (view === 'conc') {
          const cx0 = pad + 40, cx1 = w - pad - 10, cy0 = 40, cy1 = h - 70;
          V.text(g, 'SPARKDASH 2026-09-17 · PROSE · 850k · KV 14 GiB · adaptive-k + FP8 + coop', pad, 18, { size: 11, font: 'mono', weight: 700, color: C.a, maxW: w - 2 * pad });
          for (let v = 0; v <= 80; v += 20) { const yy = V.lerp(cy1, cy0, v / 80); V.line(g, cx0, yy, cx1, yy, { color: V.alpha(C.muted, 0.18), lw: 1 }); V.text(g, String(v), cx0 - 6, yy + 4, { size: 11, align: 'right', font: 'mono', color: C.muted }); }
          const slot = (cx1 - cx0) / 4, bwid = Math.min(58, slot * 0.5);
          const pts = [];
          CONC.x.forEach((n, i) => {
            const xc = cx0 + slot * (i + 0.5);
            const bh = ((CONC.agg[i] / 80) * (cy1 - cy0)) * grow;
            V.box(g, xc - bwid / 2, cy1 - bh, bwid, bh, { r: 6, fill: V.alpha(C.a, 0.75) });
            V.text(g, br(CONC.agg[i]), xc, cy1 - bh - 6, { size: 11.5, font: 'mono', weight: 700, align: 'center', color: C.ink });
            V.text(g, '×' + n, xc, cy1 + 16, { size: 12, weight: 700, align: 'center', color: C.ink });
            V.text(g, 'TTFT ' + CONC.ttft[i] + ' ms', xc, cy1 + 32, { size: 11, align: 'center', font: 'mono', color: C.muted });
            pts.push([xc, V.lerp(cy1, cy0, (CONC.stream[i] / 80) * grow), cy1 - bh]);
          });
          g.strokeStyle = C.c; g.lineWidth = 2.5; g.beginPath(); pts.forEach((p, i) => (i ? g.lineTo(p[0], p[1]) : g.moveTo(p[0], p[1]))); g.stroke();
          pts.forEach((p, i) => {
            V.dot(g, p[0], p[1], 4.5, C.c);
            const lab = br(CONC.stream[i]), lw2 = tw(g, lab, 11, 'mono', 700) + 10;
            const up = p[1] - 30 > p[2] + 2, ty = up ? p[1] - 18 : p[1] + 18;
            V.box(g, p[0] - lw2 / 2, ty - 9, lw2, 17, { r: 5, fill: V.alpha(C.bg, 0.92), stroke: V.alpha(C.c, 0.8), lw: 1 });
            V.text(g, lab, p[0], ty, { size: 11, font: 'mono', color: C.c, weight: 700, align: 'center', base: 'middle' });
          });
          const lgY = h - 20;
          V.box(g, pad, lgY - 9, 12, 10, { r: 3, fill: V.alpha(C.a, 0.75) }); V.text(g, 'aggregate (tok/s)', pad + 18, lgY, { size: 11.5, color: C.ink });
          V.dot(g, pad + (narrow ? 140 : 160), lgY - 4, 4.5, C.c); V.text(g, 'per stream', pad + (narrow ? 150 : 170), lgY, { size: 11.5, color: C.ink });
        } else {
          const legH = narrow ? KLD.length * 17 + 14 : 0;
          const cx0 = pad + (narrow ? 40 : 52), cx1 = w - pad - 10, cy0 = 34, cy1 = h - 44 - legH;
          const GX = (gb) => V.lerp(cx0, cx1, (gb - 150) / 200), GY = (k) => V.lerp(cy1, cy0, k / 0.065);
          V.text(g, 'KLD(teacher‖model) × checkpoint bytes · 51,175 positions', pad, 16, { size: 11, font: 'mono', weight: 700, color: C.d, maxW: w - 2 * pad });
          for (let k = 0; k <= 0.06; k += 0.02) { const yy = GY(k); V.line(g, cx0, yy, cx1, yy, { color: V.alpha(C.muted, 0.18), lw: 1 }); V.text(g, br(k, 2), cx0 - 6, yy + 4, { size: 11, align: 'right', font: 'mono', color: C.muted }); }
          [150, 200, 250, 300, 350].forEach((gb) => V.text(g, gb + ' GB', GX(gb), cy1 + 16, { size: 11, align: 'center', font: 'mono', color: C.muted }));
          V.line(g, GX(256), cy0, GX(256), cy1, { color: C.red, lw: 1.5, dash: [5, 4] });
          V.text(g, '2 × 128 GB (raw)', GX(256) + 6, cy0 + 10, { size: 11, color: C.red, weight: 600 });
          KLD.forEach((p, n) => {
            const x = GX(p.gb), yy = V.lerp(cy1, GY(p.k), grow);
            if (p.hi) V.glow(g, x, yy, 18, C.a, 0.9 + 0.1 * Math.sin(t * 3));
            V.dot(g, x, yy, p.hi ? 6.5 : 5, p.hi ? C.a : C.b);
            if (narrow) {
              // on mobile: number next to the point and legend as a list below the chart
              const tx = p.side === 'r' ? x + 12 : x - 12;
              V.text(g, String(n + 1), tx, yy + 4, { size: 12, weight: 700, align: p.side === 'r' ? 'left' : 'right', color: p.hi ? C.a : C.ink });
            } else {
              const lab = p.l + ' · ' + br(p.k, 4);
              const lx = p.side === 'r' ? x + 10 : x - 10;
              V.text(g, lab, lx, yy + 4, { size: 12, align: p.side === 'r' ? 'left' : 'right', color: C.ink, weight: p.hi ? 700 : 400, maxW: p.side === 'r' ? cx1 - lx : lx - cx0 });
            }
          });
          V.text(g, '↓ better', cx0 + 4, cy1 - 6, { size: 11, color: C.muted });
          if (narrow) {
            let ly = cy1 + 38;
            KLD.forEach((p, n) => {
              V.text(g, (n + 1) + ' · ' + p.l, pad, ly, { size: 11.5, color: C.ink, weight: p.hi ? 700 : 400, maxW: w - 2 * pad - 110 });
              V.text(g, p.gb + ' GB · ' + br(p.k, 4), w - pad, ly, { size: 11.5, font: 'mono', align: 'right', color: p.hi ? C.a : C.muted, weight: p.hi ? 700 : 400 });
              ly += 17;
            });
          }
        }
      });
    })();

    /* ============================================ F10 · profile mixing console */
    (function () {
      const fig = root.querySelector('#fig-perfis');
      const PROF = {
        tp2: { name: '.env.example · TP2 (default)', v: { ctx: 850000, seqs: 4, mnbt: 7168, util: 0.85, kv: 11, k: 7, dtp: 2 }, kvLab: '11 GiB', lamps: [['dense FP8', 'all', 1], ['KDA BF16', 'on', 1], ['E3 prefill', 'on', 1], ['MOE_FAST', 'off', 0], ['adaptive-k', 'off', 0], ['spinwait', 'stock', 0], ['retention', 'dense', 0], ['InstantTensor', 'on', 1], ['fair v7', 'on (v5 in the snapshot)', 1]], why: 'Maximum context with 4 generations in flight: a fixed 11 GiB pool with compact pages (1.45× of 850k), dense FP8 on every group and E3 prefill with MNBT 7168.' },
        long: { name: 'examples/tp2-long-coding.env', v: { ctx: 262144, seqs: 2, mnbt: 1024, util: 0.865, kv: 12.8, k: 7, dtp: 2 }, kvAuto: true, kvSub: ['≈12.7–12.9 GiB', '≈12.8'], lamps: [['dense FP8', 'dense,kda', 1], ['KDA BF16', 'on', 1], ['E3 prefill', 'on', 1], ['MOE_FAST', 'on', 1], ['adaptive-k', 'off', 0], ['spinwait', '16 ms', 1], ['retention', '14336/0 (unqualified)', 2], ['prompt-tokens-details', 'on', 1]], why: 'Few users with long histories: 262k, 2 seqs, 1024-token chunks to keep decode responsive, an automatic pool (≈12.7–12.9 GiB) and a sparse 14336/0 retention, which hasn’t been qualified.' },
        tp3: { name: '.env.tp3.example · 3 nodes', v: { ctx: 1000000, seqs: 4, mnbt: 7168, util: 0.8, kv: 40, k: 7, dtp: 1 }, kvLab: '40 GiB*', lamps: [['dense FP8', 'dense,kda', 1], ['KDA BF16', 'off', 0], ['E3 prefill', 'on', 1], ['MOE_FAST', 'forced off', 0], ['heads', '66 (22/rank)', 1], ['expert parallel', '96/rank', 1], ['mm-encoder', 'data', 1], ['SWA retention', '0', 1]], why: '3 nodes: 66 heads via --hf-overrides, expert parallel (96 experts per rank), drafter only on rank 0 and a fixed pool per topology (* 40 GiB in the README, 35 GiB on the measured cluster: a discrepancy).' },
        tp4: { name: '.env.tp4.example · 4 nodes (experimental)', v: { ctx: 850000, seqs: 8, mnbt: 2048, util: 0.75, kv: null, k: 3, dtp: 4 }, kvAuto: true, kvSub: ['not measured', '—'], lamps: [['dense FP8', 'not forwarded', 0], ['E3 prefill', 'on since e5dc885', 1], ['adaptive-k', 'explicit opt-in', 0], ['TEMP_ROWS', '32 (with E3)', 1], ['sparse-MLA slice', '0 (optional 64)', 0], ['SWA retention', '0', 1]], why: 'Experimental: k=3 (accepts 53% versus 30% with k=7), 8 seqs, MNBT 2048 and util 0.75. The context inherits 850k from the .env (the script would use 1M); the launcher removes the KV reservation from the TP2 .env, so the pool is profiled; in the snapshot, start-tp4.sh forwarded neither E3, adaptive-k nor dense FP8. Since then, E3 became the TP4 default (e5dc885; cold prefill +35/+38/+41% at 8K/32K/100K, a contributor measurement), adaptive-k became an explicit .env.tp4 opt-in (1e8f928) and the default reasoning effort is honoured (4709bc5); dense FP8 is still not forwarded.' },
      };
      const FAD = [
        { k: 'ctx', lab: 'context', sh: 'ctx', min: Math.log(100000), max: Math.log(1100000), map: Math.log, fmt: (v) => (v >= 1e6 ? '1M' : Math.round(v / 1000) + 'k') },
        { k: 'seqs', lab: 'seqs', sh: 'seqs', min: 0, max: 16, fmt: (v) => String(v) },
        { k: 'mnbt', lab: 'MNBT', sh: 'MNBT', min: 0, max: 8192, red: 8000, fmt: (v) => String(v) },
        { k: 'util', lab: 'util', sh: 'util', min: 0.7, max: 0.9, red: 0.87, fmt: (v) => br(v, 3) },
        { k: 'kv', lab: 'KV pool', sh: 'KV', min: 0, max: 40, fmt: null },
        { k: 'k', lab: 'DFlash k', sh: 'k', min: 0, max: 8, fmt: (v) => String(v) },
        { k: 'dtp', lab: 'draft TP', sh: 'dTP', min: 0, max: 4, fmt: (v) => String(v) },
      ];
      let cur = 'tp2';
      const A = {};
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Profile', options: [{ v: 'tp2', label: '.env.example' }, { v: 'long', label: 'long-coding' }, { v: 'tp3', label: 'TP3' }, { v: 'tp4', label: 'TP4' }], value: 'tp2', onChange: (v) => { cur = v; } });
      V.fig(fig, (g, w, h, t, dt) => {
        const C = V.C(); const narrow = w < 600; const pad = 14;
        const P = PROF[cur];
        V.text(g, P.name, pad, 22, { size: narrow ? 14 : 16, font: 'display', weight: 700, color: C.ink, maxW: w - 2 * pad });
        const fw = (w - 2 * pad) / FAD.length, yT = 62, yB = narrow ? 206 : 246;
        FAD.forEach((f, i) => {
          const raw = P.v[f.k] == null ? (f.min + f.max) / 2 : P.v[f.k];
          const target = f.map ? f.map(raw) : raw;
          A[f.k] = near(A[f.k], target, dt, 5);
          const auto = f.k === 'kv' && P.kvAuto, ghost = auto && P.v[f.k] == null;
          const u = V.clamp((A[f.k] - f.min) / (f.max - f.min), 0, 1);
          const x = pad + fw * (i + 0.5);
          V.text(g, narrow ? f.sh : f.lab, x, yT - 12, { size: 11, align: 'center', weight: 600, color: C.ink, maxW: fw - 4 });
          V.box(g, x - 3, yT, 6, yB - yT, { r: 3, fill: V.alpha(C.muted, 0.25) });
          if (f.red != null) { const ur = (f.red - f.min) / (f.max - f.min); V.box(g, x - 3, yT, 6, (1 - ur) * (yB - yT), { r: 3, fill: V.alpha(C.red, 0.7) }); }
          for (let q = 0; q <= 4; q++) V.line(g, x - 9, V.lerp(yB, yT, q / 4), x - 5, V.lerp(yB, yT, q / 4), { color: V.alpha(C.muted, 0.5), lw: 1 });
          const ky = V.lerp(yB, yT, u), col = C.series[i % 6], kw = narrow ? 24 : 30;
          if (auto) {
            g.setLineDash([3, 3]);
            V.box(g, x - kw / 2, ky - 7, kw, 14, { r: 4, fill: V.alpha(col, ghost ? 0.06 : 0.22), stroke: col, lw: 1.5 });
            g.setLineDash([]);
            if (ghost) V.text(g, '?', x, ky + 1, { size: 11, weight: 700, align: 'center', base: 'middle', color: col });
          } else {
            V.box(g, x - kw / 2, ky - 7, kw, 14, { r: 4, fill: col, stroke: V.alpha(C.ink, 0.35), lw: 1 });
            V.line(g, x - (narrow ? 7 : 9), ky, x + (narrow ? 7 : 9), ky, { color: onFill(C), lw: 1.5 });
          }
          const val = f.k === 'kv' ? (auto ? (narrow ? 'auto' : 'auto (profiled)') : P.kvLab) : f.fmt(P.v[f.k]);
          V.text(g, val, x, yB + 18, { size: 11, align: 'center', font: 'mono', weight: 700, color: C.ink, maxW: fw - 2 });
          if (auto) V.text(g, P.kvSub[narrow ? 1 : 0], x, yB + 32, { size: 11, align: 'center', font: 'mono', color: C.muted, maxW: fw - 2 });
        });
        // lamps
        let lx = pad, ly = yB + 46;
        P.lamps.forEach(([nm, val, st]) => {
          const lab = nm + ': ' + val, wd = tw(g, lab, 11.5, 'sans', 500) + 28;
          if (lx + wd > w - pad) { lx = pad; ly += 28; }
          const col = st === 1 ? C.a : st === 2 ? C.d : C.muted;
          V.box(g, lx, ly, wd, 22, { r: 11, fill: V.alpha(col, st ? 0.16 : 0.07), stroke: V.alpha(col, st ? 0.8 : 0.4) });
          V.dot(g, lx + 11, ly + 11, 4, st ? col : V.alpha(C.muted, 0.5));
          if (st) V.glow(g, lx + 11, ly + 11, 9, col, 0.5 + 0.3 * Math.sin(t * 3 + lx));
          V.text(g, lab, lx + 20, ly + 12, { size: 11.5, base: 'middle', color: C.ink });
          lx += wd + 6;
        });
        const whyH = para(g, P.why, pad, ly + 46, w - 2 * pad, { size: 12.5, color: C.ink, lh: 17 });
        const needH = Math.max(narrow ? 470 : 450, Math.ceil(ly + 40 + whyH + 8));
        fitH(fig, needH, needH, w);
      });
    })();

    /* ============================================ TensorFold · badge color (read from the current theme) */
    let tfCol = null, tfDark = null;
    const TFC = (C) => { if (tfCol == null || tfDark !== C.dark) { tfCol = getComputedStyle(document.documentElement).getPropertyValue('--tf').trim() || C.d; tfDark = C.dark; } return tfCol; };
    /** little train: body in the line's color and three windows */
    const train = (g, x, y, col, C, sc) => {
      const s = sc || 1;
      V.box(g, x - 13 * s, y - 7 * s, 26 * s, 14 * s, { r: 5 * s, fill: col });
      for (let q = 0; q < 3; q++) V.box(g, x - 9.5 * s + q * 7 * s, y - 3.5 * s, 5 * s, 5 * s, { r: 1.2, fill: V.alpha(C.bg, 0.92) });
    };

    /* ============================================ F11 · the switch: the map with the engine swapped */
    (function () {
      const fig = root.querySelector('#fig-desvio');
      if (!fig) return;
      const T = {
        motor: 'Engine', vllm: 'vLLM', tf: 'TensorFold', swap: '⇄ pull the lever',
        count: 'On this engine',
        cnt: (s, m, x) => s + ' shared · ' + m + ' by engine' + (x ? ' · ' + x + ' gone' : ''),
        hw: '2 × GB10 · CX7 cable · 164 GiB EXL3', hwS: 'same hardware + weights',
        brV: 'vLLM · 25 overlays', brVS: 'vLLM', brF: 'TensorFold v0.6.0 · 53 patches', brFS: 'TensorFold',
        st: 'STATION', both: 'ON BOTH', other: { v: 'on TensorFold: ', f: 'on vLLM: ' }, none: 'doesn’t exist',
        fixed: 'station pinned', hint: 'click a station or the lever',
        trans: (a, b) => 'SWAPPING THE ENGINE: ' + a + ' → ' + b,
        transBody: {
          f: 'The 6 hardware and model stations stay. Of the 24 engine decisions, 23 become TensorFold’s (TP = 2 and DFlash2 keep the name, not the mechanism) and 1 vanishes: the 512→576 pad.',
          v: 'Back to vLLM: the 6 physical stations stay put, and the 24 decisions go back to vLLM’s, including the 512→576 pad and the 3,584-token pages.',
        },
        lgV: 'vLLM decision', lgF: 'TensorFold decision',
      };
      const ROWS = [
        { name: 'WEIGHTS LINE', c: 'd', st: [
          { hw: ['BF16/FP8', 'don’t fit', 'BF16 takes ~642.6 GB and the official FP8 328.3 GB: neither fits in 2 × 128 GB. True for any engine.'] },
          { hw: ['EXL3 4 bpw', 'experts', 'The same checkpoint on both engines: routed experts in 4 bpw EXL3, the rest in BF16, 164 GiB, ~80 GiB per rank.'] },
          { v: ['dense', 'FP8 chan.', 'GLM53_DENSE_FP8=all: dense projections in per-channel weight-only FP8, via Marlin. +4–5% tok/s and “NOT-WORSE” on KL.'],
            f: ['dense', 'q4 · MSE', 'DENSE=q4: affine 4 bits in groups of 64, with the clip of least squared error; head in FP8. Over fp8: prose 38.9 → 44.4 tok/s (one boot each). Declared lossy.'] },
          { v: ['kernels', 'sm_121a', 'The ExLlamaV3 extension (pin c5d9c657) is compiled for sm_121a, with the E2 and E3 tiers for prefill’s fat experts.'],
            f: ['kernels', 'family’s', 'The EXL3 kernels are the engine’s own, JIT-compiled on the first start of each image, with the cache split by the image’s hash.'] },
          { v: ['25 overlays', 'at boot', 'Image pinned by digest and stamped; 25 Python overlays bind-mounted and reapplied, in the same order, on every docker run.'],
            f: ['53 diffs', 'at build', 'The 53 patches are diffs applied at build (patch -p0, in name order). The tag is the patches’ hash, and changing a patch takes another image.'] },
        ] },
        { name: 'SPECULATION LINE', c: 'c', st: [
          { hw: ['decode', '= bandwidth', 'Every decode token reads all active weights: the bottleneck is memory bandwidth. That’s why both engines speculate.'] },
          { v: ['DFlash2', 'k = 7', 'The drafter proposes 7 tokens per block and the target verifies 8 positions per sequence. Adaptive-k, opt-in, verifies only 2, 4 or 7.'],
            f: ['DFlash2', 'fnc7:0.3', 'Up to 7 drafts, until the product of the odds under the keyed noise drops below 0.3. Over fc5:0.3: prose 47.2 → 50.6 tok/s (two boots each).'] },
          { v: ['8 positions', 'uniform', 'FULL graphs only accept uniform decode: with adaptive-k, the verified length is the minimum over the batch.'],
            f: ['copy drafts', 'up to 15', 'Before DFlash2, drafts copied from the context when the last 8 tokens appeared before. With up to 15 per round, edits went from 84 → 119 tok/s.'] },
          { v: ['graphs', '8·16·24·32', 'With up to 4 sequences, decode steps have 8, 16, 24 or 32 tokens, and each size needs a CUDA graph.'],
            f: ['graphs', '1 to 32', 'One CUDA graph per window row count, from 1 to 32, with positions and ranges in device tables: one graph serves any mix of requests.'] },
          { v: ['TEMP_ROWS', '= 32', 'The fused MoE buffer holds 32 rows per expert: 4 × 8 fit in a single CUDA-graph-compatible launch.'],
            f: ['window ≤ 32', '512 KiB', 'Up to 16 rows per request and 32 in the batch; past that, the longest tails are trimmed. 32 rows × 16 KiB = 512 KiB, RoCE’s ceiling.'] },
        ] },
        { name: 'DISTRIBUTED LINE', c: 'e', st: [
          { hw: ['2 nodes', '1 GPU', 'Two GB10s, one GPU each, linked by a ConnectX-7 cable over RoCE v2. That’s what both engines have.'] },
          { v: ['TP = 2', 'mp', 'Tensor parallel 2 with the multiprocessing executor: head with API and EngineCore, headless worker. TP3 and TP4 exist in their own launchers.'],
            f: ['TP = 2', 'only 2', 'One rank per Spark: rank 0 decides and serves the API, rank 1 applies. In v1.3.2, GLM’s CUDA engine required exactly two ranks; since v1.4, three.'] },
          { v: ['all-reduce', 'NCCL', 'Two all-reduces per layer, after attention and after the MoE, adding up the two ranks’ partial halves.'],
            f: ['all-gather', '+ sum', 'Each rank receives the other’s fp32 partials and adds them in rank order, rank 0 first: both end up with the same bits, whatever the transport.'] },
          { v: ['NCCL', 'IB verbs', 'The image’s NCCL 2.30.7 carries the collectives over the IB transport (NCCL_NET=IB) on RoCE v2, with interface, HCA and GID pinned per rank.'],
            f: ['RoCE', 'one-shot', 'Gathers of up to 512 KiB in a single kernel call, by RDMA WRITE: 11 µs against NCCL’s 45 µs for 16 KiB, +6% decode.'] },
          { v: ['same', 'graphs', 'The ranks get the same script and the same environment, and nothing checks that at startup: if the graphs diverge, NCCL hangs.'],
            f: ['config', 'checked', 'Before loading the weights, the ranks exchange a vector with the settings that change collectives or drafts and refuse to start if it differs.'] },
        ] },
        { name: 'MEMORY LINE (UMA)', c: 'a', st: [
          { hw: ['CPU + GPU', '128 GB', 'Unified memory: what the engine takes comes out of the OS and the page cache, and running out tends to freeze the machine.'] },
          { v: ['util', '0.85', 'With the fixed pool, util doesn’t size the KV: it’s the free-memory floor the boot requires, 0.85 × 121.69 ≈ 103.4 GiB.'],
            f: ['reserve', '14.5 GiB', 'Budget = MemAvailable at startup − reserve. A window that doesn’t fit is refused before the weights, along with the largest window that would.'] },
          { v: ['Instant', 'Tensor', 'Direct-I/O loader: the weights load in ~65–70 s, but the profiled KV loses ~4.4–5.6 GiB. A 122 s restart.'],
            f: ['load', '2–6 min', '~80 GiB per Spark in 2–6 min (~2.2 min with the worker’s weights over NFS); the first start of each image still compiles kernels.'] },
          { v: ['fixed KV', '11 GiB', '--kv-cache-memory-bytes fixes 11 GiB = 434 block IDs = 1,233,779 tokens, 1.45× an 850k request.'],
            f: ['pool', '12.5 GiB', 'The pool’s ceiling beyond the window, per Spark. At the measured start, 2,922,496 tokens; it ranges from ~2.1 to 2.9M with free memory.'] },
          { v: ['block', 'IDs', 'Every group of the hybrid allocator pays whole 25.9 MiB IDs; compact pages take 850k from ~536 to ~299 IDs.'],
            f: ['extents', '2,048', 'Each request has a contiguous range, a multiple of 2,048 tokens, that grows, moves and gets compacted. The KDA state lives in slots, outside the pool.'] },
        ] },
        { name: 'ATTENTION LINE', c: 'b', st: [
          { hw: ['MLA NoPE', '+ GB10', 'The 11 MLA layers have no RoPE (qk_rope_head_dim=0), and the GPU is a GB10 (SM121). That holds on both engines.'] },
          { v: ['1 kernel', 'sparse', 'SM12x’s only sparse-MLA kernel, FLASHINFER_MLA_SPARSE_SM120, has a fixed 576 geometry and only reads fp8_ds_mla.'],
            f: ['kernel', 'latent', 'The engine keeps only the 512 latent, absorbs the query once per row and, past 2,051 tokens, attends to the indexer’s 512 top-scoring pools.'] },
          { v: ['KV', 'fp8_ds_mla', '--kv-cache-dtype fp8 becomes the fp8_ds_mla record: 512 B of FP8 + 16 B of scales + 128 B of RoPE.'],
            f: ['KV', 'kv8 e4m3', 'KV=fp8: 512 e4m3 codes and one power-of-two fp32 scale per row, which brings the maximum into [128, 256]: nothing saturates. 2.65% relative error per row.'] },
          { v: ['pad', '512→576', 'The overlay adds 64 RoPE zeros to the query and to k_pe just to fit the kernel’s geometry. It’s exact, but it costs 128 B per token and layer.'],
            f: ['pad', '512→576', 'It doesn’t exist: the kernel is the engine’s own and reads the 512 latent as it is. It’s the cleanest example of a constraint that came from the engine, not from the model.', 'x'] },
          { v: ['656 B', '/token', 'Per token and layer: 512 + 16 + 128 B, 24% more than the 528 B needed. Across the 11 layers, ~7.6 KB per token per rank.'],
            f: ['528 B', '/token', '512 codes + 4 B of scale + 12 B of alignment: −19.5% latent bytes. With the indexer, ≈6.2 KB per token per rank (an estimate).'] },
        ] },
        { name: 'OUTPUT LINE', c: 'f', st: [
          { v: ['rejection', 'p/q', 'Rejection sampling: accepts the draft with min(1, p/q). It preserves the target’s distribution, but batch, kernel and speculation can change the bits.'],
            f: ['keyed', 'sample', 'Gumbel noise derived from (seed, position, token), with no state. Verifying becomes comparing: accept while the draft equals the sample.'] },
          { v: ['bits', 'vary', 'Lossless in distribution, not in sequence: the same question can get another reply with drafts, in a batch or from the cache.'],
            f: ['bits', '= serial', 'Contract: with drafts = serial, 4 together = each alone, resumed = fresh. Tested: drafted = serial 6/6, concurrency 22/22.'] },
          { v: ['APC', '3,584', 'Prefix caching reuses hashed blocks; in the hybrid, hits align to 3,584 tokens and the tail is recomputed.'],
            f: ['64 grid', '+ replay', 'States kept at the last multiple of 64, at the end of the system prompt and at the end of the prompt, with the logits row: identical 64k, from 34 s to < 0.07 s.'] },
          { v: ['fair v5', 'mixed', 'Prefill and decode share the step; the overlay picks the newcomer’s chunk so the step fits in 2 s, with 30% of engine time for the prefill.'],
            f: ['chunk', 'or round', 'A prompt chunk never shares a forward with decode. Prompts that arrive together share a chunk: TTFT with 4 requests from 590 to 340 ms.'] },
          { v: ['API', 'full', 'vLLM’s surface: logprobs, n>1, penalties and an optional Bearer API key.'],
            f: ['API', 'smaller', 'No logprobs and no n>1 (HTTP 400), penalties accepted and ignored, no API key: to expose it, a proxy with authentication.'] },
        ] },
      ];
      const AUTO = [[4, 3], [3, 1], [1, 1], [5, 1], [5, 2], [2, 2]];
      let eng = 'v', prev = 'v', ft = 99, flipT = -99, selT = -99, now = 0, sw = 0, user = null, auto = true, autoT = 0, autoK = AUTO.length - 1;
      const counts = (e) => { let s = 0, m = 0, x = 0; ROWS.forEach((R) => R.st.forEach((S) => { if (S.hw) s++; else if ((e === 'v' ? S.v : S.f)[3] === 'x') x++; else m++; })); return [s, m, x]; };
      const ctl = UI.controls(fig);
      let seg = null;
      function flip(v, byUser) {
        if (v === eng) return;
        prev = eng; eng = v; ft = 0; flipT = now; autoT = 0;
        if (seg) seg.set(v, true);
        if (byUser) auto = false;
        rdo.set(T.cnt(...counts(eng)));
      }
      seg = UI.seg(ctl, { label: T.motor, options: [{ v: 'v', label: T.vllm }, { v: 'f', label: T.tf }], value: 'v', onChange: (v) => flip(v, true) });
      UI.button(ctl, T.swap, () => flip(eng === 'v' ? 'f' : 'v', true), { primary: true });
      const rdo = UI.readout(ctl, T.count, T.cnt(...counts('v')));
      UI.legend(ctl, [['var(--glm)', T.lgV], ['var(--tf)', T.lgF]]);
      V.fig(fig, (g, w, h, t, dt, S, H) => {
        now = t;
        const C = V.C(), TF = TFC(C), narrow = w < 600;
        fitH(fig, 560, 714, w);
        ft += dt;
        const ENG = { v: C.glm, f: TF };
        const pad = 14, TS = narrow ? 66 : 58, detH = narrow ? 150 : 84;
        const top = TS + 6, rowH = (h - top - detH - 4) / 6;
        const x0 = pad + (narrow ? 24 : 36), x1 = w - pad - (narrow ? 24 : 36);
        const SX = (j) => x0 + (j * (x1 - x0)) / 4;
        const SY = (i) => top + i * rowH + 24;
        const mw = (x1 - x0) / 4 - 4;
        // ---------- auto: tours stations and swaps the engine every so often
        if (auto && !user) {
          autoT += dt;
          if (autoT > (ft < 3 ? 4.6 : 3.4)) { autoT = 0; autoK++; if (autoK >= AUTO.length) { autoK = 0; flip(eng === 'v' ? 'f' : 'v', false); } }
        }
        // ---------- top track: hardware → switch → engine
        const yc = narrow ? 34 : 30;
        const hwLab = narrow ? T.hwS : T.hw;
        const cw = chip(g, pad, yc - 10, hwLab, { size: 11, font: 'sans', fill: V.alpha(C.ink, 0.06), stroke: V.alpha(C.muted, 0.5), color: C.ink, h: 20 });
        const lv = narrow ? T.brVS : T.brV, lf = narrow ? T.brFS : T.brF;
        const wv = tw(g, lv, 11, 'mono', 600) + 14, wf = tw(g, lf, 11, 'mono', 600) + 14;
        const xb = w - pad - Math.max(wv, wf) - 6;
        const xs = Math.min(pad + cw + (narrow ? 30 : 46), xb - 46);
        const yU = yc - 15, yD = yc + 15;
        sw = near(sw, eng === 'f' ? 1 : 0, dt, 8);
        V.line(g, pad + cw, yc, xs, yc, { color: V.alpha(C.ink, 0.55), lw: 4 });
        const branch = (yy, on, col) => {
          g.beginPath(); g.moveTo(xs, yc); g.bezierCurveTo(xs + 18, yc, xs + 14, yy, xs + 34, yy); g.lineTo(xb, yy);
          g.strokeStyle = on ? col : V.alpha(C.muted, 0.35); g.lineWidth = on ? 4 : 2; g.stroke();
        };
        branch(yU, eng === 'v', ENG.v); branch(yD, eng === 'f', ENG.f);
        chip(g, xb + 6, yU - 9, lv, { size: 11, h: 18, weight: 600, fill: V.alpha(ENG.v, eng === 'v' ? 0.2 : 0.06), stroke: V.alpha(ENG.v, eng === 'v' ? 0.9 : 0.35), color: eng === 'v' ? C.ink : C.muted });
        chip(g, xb + 6, yD - 9, lf, { size: 11, h: 18, weight: 600, fill: V.alpha(ENG.f, eng === 'f' ? 0.2 : 0.06), stroke: V.alpha(ENG.f, eng === 'f' ? 0.9 : 0.35), color: eng === 'f' ? C.ink : C.muted });
        // switch blade
        const ang = V.lerp(Math.atan2(yU - yc, 30), Math.atan2(yD - yc, 30), sw);
        V.line(g, xs, yc, xs + Math.cos(ang) * 20, yc + Math.sin(ang) * 20, { color: ENG[eng], lw: 5 });
        V.dot(g, xs, yc, 4, C.ink);
        // lever
        const lx = xs - 16, ly = yc + (narrow ? 26 : 22);
        V.box(g, lx - 9, ly - 2, 18, 6, { r: 3, fill: V.alpha(C.muted, 0.6) });
        const la = V.lerp(-2.25, -0.9, sw), le = 18;
        V.line(g, lx, ly, lx + Math.cos(la) * le, ly + Math.sin(la) * le, { color: C.ink, lw: 2.5 });
        V.dot(g, lx + Math.cos(la) * le, ly + Math.sin(la) * le, 6, ENG[eng]);
        if (auto && !user) V.glow(g, lx + Math.cos(la) * le, ly + Math.sin(la) * le, 13, ENG[eng], 0.35 + 0.25 * Math.sin(t * 4));
        if (ft < 1.2) V.glow(g, lx + Math.cos(la) * le, ly + Math.sin(la) * le, 16, ENG[eng], 1 - ft / 1.2);
        // train on the top track
        const tu = Math.min(1, ft / 0.95);
        let tx, ty;
        if (tu < 0.45) { tx = V.lerp(pad + cw + 12, xs, tu / 0.45); ty = yc; }
        else { const u2 = (tu - 0.45) / 0.55; tx = V.lerp(xs + 6, xb - 22, u2); ty = V.lerp(yc, eng === 'v' ? yU : yD, Math.min(1, u2 * 3)); }
        train(g, tx, ty, ENG[eng], C, narrow ? 0.85 : 1);
        // ---------- clicks
        if (H.pointer.clicked) {
          if (H.pointer.y < TS + 2) flip(eng === 'v' ? 'f' : 'v', true);
          else {
            let best = null, bd = 28;
            for (let i = 0; i < 6; i++) for (let j = 0; j < 5; j++) { const d = Math.hypot(H.pointer.x - SX(j), H.pointer.y - SY(i) - 10); if (d < bd) { bd = d; best = [i, j]; } }
            if (best) { user = best; auto = false; selT = t; }
          }
        }
        const sel = user || AUTO[autoK];
        // ---------- lines and stations
        ROWS.forEach((R, i) => {
          const col = C[R.c], y = SY(i), onRow = sel[0] === i;
          V.text(g, R.name, pad, y - 14, { size: 11, font: 'mono', weight: 700, color: col });
          V.line(g, x0, y, x1, y, { color: V.alpha(col, onRow ? 1 : 0.7), lw: onRow ? 5 : 3.5 });
          const st0 = 0.95 + i * 0.09, span = 0.15;
          if (ft > st0 - 0.2 && ft < st0 + 4 * span + 0.35) {
            const px = V.clamp(x0 + ((ft - st0) / span) * ((x1 - x0) / 4), x0 - 6, x1 + 6);
            train(g, px, y, col, C, 0.8);
          }
          R.st.forEach((Sx, j) => {
            const x = SX(j), dij = st0 + j * span;
            const p = V.clamp((ft - dij) / 0.4, 0, 1);
            const side = p < 0.5 ? prev : eng;
            const sy = Math.max(0.06, Math.abs(Math.cos(Math.PI * p)));
            const lab = Sx.hw || (side === 'v' ? Sx.v : Sx.f);
            const kind = Sx.hw ? 'hw' : lab[3] === 'x' ? 'ghost' : 'eng';
            const isSel = sel[0] === i && sel[1] === j;
            if (isSel) V.glow(g, x, y, 24, col, 0.85);
            if (kind === 'hw' && Math.abs(ft - dij) < 0.3) V.glow(g, x, y, 18, col, 1 - Math.abs(ft - dij) / 0.3);
            g.save(); g.translate(x, y); g.scale(1, kind === 'hw' ? 1 : sy);
            if (kind === 'hw') { g.beginPath(); g.arc(0, 0, 7.5, 0, Math.PI * 2); g.fillStyle = C.bg; g.fill(); g.lineWidth = 3; g.strokeStyle = col; g.stroke(); }
            else if (kind === 'eng') { V.box(g, -7.5, -7.5, 15, 15, { r: 3.5, fill: C.bg, stroke: col, lw: 3 }); V.dot(g, 0, 0, 3.4, ENG[side]); }
            else { g.setLineDash([3, 3]); g.beginPath(); g.arc(0, 0, 7.5, 0, Math.PI * 2); g.fillStyle = C.bg; g.fill(); g.lineWidth = 2; g.strokeStyle = V.alpha(C.muted, 0.8); g.stroke(); g.setLineDash([]); }
            g.restore();
            g.save(); g.translate(x, y + 28); g.scale(1, kind === 'hw' ? 1 : sy);
            const c1 = kind === 'ghost' ? C.muted : C.ink, c2 = kind === 'hw' ? C.ink : kind === 'ghost' ? C.muted : ENG[side];
            V.text(g, lab[0], 0, -7, { size: 11, align: 'center', color: c1, weight: isSel ? 700 : 500, maxW: mw });
            V.text(g, lab[1], 0, 7, { size: 11, align: 'center', color: c2, font: 'mono', weight: kind === 'eng' ? 600 : 500, maxW: mw });
            if (kind === 'ghost') {
              const sw1 = Math.min(mw, tw(g, lab[0], 11, 'sans', 500)) / 2 + 3, sw2 = Math.min(mw, tw(g, lab[1], 11, 'mono', 500)) / 2 + 3;
              V.line(g, -sw1, -11, sw1, -11, { color: C.red, lw: 1.6 });
              V.line(g, -sw2, 3, sw2, 3, { color: C.red, lw: 1.6 });
            }
            g.restore();
          });
        });
        // ---------- detail panel
        const dy = h - detH + 2;
        V.box(g, pad, dy, w - 2 * pad, detH - 8, { r: 10, fill: V.alpha(C.ink, 0.05), stroke: V.alpha(C.muted, 0.3) });
        const inWave = t - flipT < 2.6 && selT < flipT;
        let title, other = '', ocol = C.muted, body, tcol = C.ink;
        if (inWave) {
          title = T.trans(prev === 'v' ? T.vllm : T.tf, eng === 'v' ? T.vllm : T.tf).toUpperCase(); tcol = ENG[eng];
          body = T.transBody[eng];
        } else {
          const R = ROWS[sel[0]], Sx = R.st[sel[1]];
          const lab = Sx.hw || (eng === 'v' ? Sx.v : Sx.f);
          title = R.name + ' · ' + T.st + ' ' + (sel[1] + 1) + ' · ' + (Sx.hw ? T.both : (eng === 'v' ? T.vllm : T.tf).toUpperCase());
          tcol = C[R.c]; body = lab[2];
          if (!Sx.hw) { const o = eng === 'v' ? Sx.f : Sx.v; other = T.other[eng] + (o[3] === 'x' ? T.none : o[0] + ' · ' + o[1]); ocol = ENG[eng === 'v' ? 'f' : 'v']; }
        }
        V.text(g, title, pad + 12, dy + 18, { size: 11, font: 'mono', weight: 700, color: tcol, maxW: w - 2 * pad - 24 });
        let by = dy + 38;
        if (other) {
          if (narrow) { V.text(g, other, pad + 12, dy + 35, { size: 11.5, color: ocol, weight: 600, maxW: w - 2 * pad - 24 }); by = dy + 56; }
          else V.text(g, other, w - pad - 12, dy + 18, { size: 11.5, align: 'right', color: ocol, weight: 600 });
        } else if (!narrow && !inWave) V.text(g, user ? T.fixed : T.hint, w - pad - 12, dy + 18, { size: 11, align: 'right', color: C.muted });
        para(g, body, pad + 12, by, w - 2 * pad - 24, { size: 12.5, color: C.ink, lh: 17 });
      });
    })();

    /* ============================================ F12 · two ranks, one truth */
    (function () {
      const fig = root.querySelector('#fig-paridade');
      if (!fig) return;
      const T = {
        scen: 'Divergence', run: '▶ run again',
        stations: ['args', 'image', 'docker', 'startup', 'weights', 'graphs', 'API'],
        ranks: ['rank 0', 'rank 1'], both: 'both ranks', eng: ['vLLM', 'TensorFold'],
        serving: 'serving', oldPair: 'old pair still up', hang: 'NCCL waits…', dead: 'died',
        S: [
          { k: 'rank', label: 'Rank 1 differs',
            v: { who: 1, tag: 'missing no. 18', at: 5, kind: 'hang', chip: 'hangs at graphs', txt: 'Rank 1 comes up without overlay no. 18, with GLM53_ADAPTIVE_K=ema. Nothing checks parity at startup: rank 0 captures decode graphs for 3, 5 and 8 positions, rank 1 only for 8, and the collectives stop matching. Interpretation: since both processes stay alive, the fail-fast doesn’t fire, and the launcher waits for /health until the READY_TIMEOUT of 3,600 s.' },
            f: { who: 1, tag: 'other COPY_MAX', at: 3, kind: 'refuse', chip: 'refused at startup', txt: 'Rank 1 comes up with another TF_GLM_COPY_MAX. Before loading the weights, each rank packs the settings that change collectives or drafts into a vector of integers; both all-gather it, compare and refuse to start: “the two ranks were started with different settings”.' } },
          { k: 'img', label: 'Stale image on the worker',
            v: { who: 1, tag: 'stale image', at: 1, kind: 'fix', chip: 'fixed, goes on', txt: 'The launcher compares head and worker by the image key, a hash of the layers (RootFS.Layers), and not by .Id, which differs between overlay2 and containerd. If they diverge, the worker gets the image by pull or by docker save | ssh docker load, and both come up with the same bytes.' },
            f: { who: 1, tag: 'stale image', at: 1, kind: 'fix', chip: 'fixed, goes on', txt: 'Setup compares the worker’s image by content (layers, Env, Entrypoint, Cmd and WorkingDir) and only accepts an image whose tf.patches label matches the patches’ hash. If they diverge: pull, or docker save | ssh docker load of only the missing layers.' } },
          { k: 'typo', label: 'Typo on restart',
            v: { who: 2, tag: 'typo', at: 0, kind: 'refuse', chip: 'refused before stop', txt: 'Validation checks the type and range of every knob (MAX_MODEL_LEN ≤ 1e6, MAX_NUM_SEQS ≤ 4096, MNBT ≤ 8,388,608…) before stopping the containers. A typo fails there, and the pair that was serving keeps serving.' },
            f: { who: 2, tag: 'typo', at: 0, kind: 'refuse', chip: 'refused before stop', txt: 'The arguments of ./start.sh restart go through TensorFold’s own parser, in a throwaway container without a GPU, before the server is stopped. A typo fails there, and the old pair stays up.' } },
          { k: 'down', label: 'Worker dies at boot',
            v: { who: 1, tag: 'dies', at: 4, kind: 'abort', chip: 'aborts in ~30 s', txt: 'The worker dies while the weights load. The launcher checks the worker’s liveness every ~10 s and, after 3 failures in a row (~30 s), aborts the wait for /health. Without that fail-fast, it would wait until the READY_TIMEOUT of 3,600 s for a broken pair.' },
            f: { who: 1, tag: 'dies', at: 4, kind: 'abort', chip: 'aborts in ~15 s', txt: 'While waiting for the API, start.sh checks rank 1 every 15 s and fails showing the last 25 log lines of both ranks. While serving, an error latches the broken state on both; with FOREGROUND=1, one rank going down takes the other with it, because a lone rank would wait for its peer forever.' } },
        ],
      };
      const STEP = 0.55;
      let si = 0, t0 = null, now = 0, auto = true, linesCache = { w: 0, n: 0 };
      const ctl = UI.controls(fig);
      const seg = UI.seg(ctl, { label: T.scen, options: T.S.map((s, i) => ({ v: i, label: s.label })), value: 0, onChange: (v) => { si = v; t0 = now; auto = false; } });
      UI.button(ctl, T.run, () => { t0 = now; auto = false; });
      V.fig(fig, (g, w, h, t, dt, S, H) => {
        now = t; if (t0 == null) t0 = t;
        const C = V.C(), TF = TFC(C), narrow = w < 600, pad = 14;
        const ENG = [C.glm, TF];
        const KC = { refuse: C.e, fix: C.e, abort: C.c, hang: C.red };
        // stable height: the longest text across all scenarios, at the current width
        if (linesCache.w !== w) {
          let n = 1; T.S.forEach((s) => [s.v, s.f].forEach((e) => { n = Math.max(n, wrap(g, e.txt, w - 2 * pad - 16, 12).length); }));
          linesCache = { w, n };
        }
        const PH = 126 + linesCache.n * 16 + 8, gap = 10;
        fitH(fig, 2 * PH + gap + 12, 2 * PH + gap + 12, w);
        if (H.pointer.clicked) { t0 = t; auto = false; }
        const el = t - t0;
        const total = 6 * STEP + 5;
        if (el > total) { if (auto) { si = (si + 1) % T.S.length; seg.set(si, true); } t0 = t; }
        const sc = T.S[si];
        const xl = pad + (narrow ? 22 : 64), xr = w - pad - (narrow ? 14 : 24);
        const SX = (k) => xl + (k * (xr - xl)) / 6;
        [sc.v, sc.f].forEach((E, p) => {
          const y0 = 6 + p * (PH + gap), col = ENG[p];
          V.box(g, pad * 0.5, y0, w - pad, PH, { r: 10, fill: V.alpha(C.ink, 0.035), stroke: V.alpha(col, 0.35) });
          const ew = chip(g, pad + 2, y0 + 7, T.eng[p], { size: 11.5, weight: 700, font: 'sans', h: 20, fill: V.alpha(col, 0.18), stroke: col, color: C.ink });
          if (!narrow) chip(g, pad + 10 + ew, y0 + 7, '≠ ' + (E.who === 2 ? T.both : T.ranks[1]) + ': ' + E.tag, { size: 11, weight: 600, font: 'sans', h: 20, fill: V.alpha(C.red, 0.12), stroke: V.alpha(C.red, 0.7), color: C.red });
          const stopAt = E.kind === 'fix' ? 6 : E.at;
          const pr = Math.min(stopAt, Math.max(0, el / STEP));
          const caught = el / STEP >= E.at;
          const kc = KC[E.kind];
          if (caught) {
            const cwid = tw(g, E.chip, 11, 'sans', 700) + 16;
            chip(g, w - pad - 2 - cwid, y0 + 7, E.chip, { size: 11, weight: 700, font: 'sans', h: 20, fill: V.alpha(kc, 0.2), stroke: kc, color: C.ink });
          }
          const yl = [y0 + 52, y0 + 84];
          T.ranks.forEach((rn, r) => {
            if (!narrow) V.text(g, rn, pad + 2, yl[r] + 4, { size: 11, font: 'mono', color: C.muted });
            V.line(g, xl, yl[r], xr, yl[r], { color: V.alpha(C.muted, 0.45), lw: 2 });
            for (let k = 0; k < 7; k++) { V.dot(g, SX(k), yl[r], k <= pr + 0.01 ? 4 : 3, k <= pr + 0.01 ? V.alpha(col, 0.9) : V.alpha(C.muted, 0.5)); }
          });
          T.stations.forEach((s, k) => V.text(g, s, SX(k), y0 + 109, { size: narrow ? 10.5 : 11, align: 'center', color: Math.round(pr) === k ? C.ink : C.muted, weight: Math.round(pr) === k ? 700 : 400, maxW: (xr - xl) / 6 + 4 }));
          // barrier at the station that catches the divergence
          if (caught) {
            const bx = SX(E.at) + (E.kind === 'fix' ? 0 : 24), ca = Math.max(0, 1 - (el - E.at * STEP) / 1.2);
            if (ca > 0) V.glow(g, bx, (yl[0] + yl[1]) / 2, 40, kc, ca);
            V.box(g, bx - 3, yl[0] - 16, 6, yl[1] - yl[0] + 32, { r: 3, fill: V.alpha(kc, 0.85) });
          }
          // capsule position
          const k0 = Math.floor(pr), fr = pr - k0;
          const cx = k0 >= 6 ? SX(6) : V.lerp(SX(k0), SX(k0 + 1), V.ease.inOut(fr));
          const offset = 0;
          const hangNow = E.kind === 'hang' && caught;
          [0, 1].forEach((r) => {
            const dead = E.kind === 'abort' && caught && r === 1;
            const x = cx + offset, y = yl[r] + (hangNow ? Math.sin(t * 9 + r) * 1.5 : 0);
            V.box(g, x - 15, y - 8, 30, 16, { r: 8, fill: dead ? V.alpha(C.muted, 0.6) : col });
            V.text(g, dead ? '✕' : 'R' + r, x, y + 0.5, { size: 10.5, font: 'mono', weight: 700, align: 'center', base: 'middle', color: dead ? C.ink : onFill(C) });
            if (dead) V.text(g, T.dead, x - (narrow ? 17 : 20), y + (narrow ? -11 : 4), { size: 10.5, align: 'right', color: C.c, weight: 600 }); // on mobile, above the line: beside it would collide with the previous station
            const hasG = E.who === 2 || E.who === r;
            if (hasG && !dead) {
              const fixed = E.kind === 'fix' && caught;
              const gx = x + 14, gy = y - 9, gc = fixed ? C.e : C.red;
              g.save(); g.translate(gx, gy); g.rotate(Math.PI / 4); V.box(g, -5.5, -5.5, 11, 11, { r: 2.5, fill: gc }); g.restore();
              V.text(g, fixed ? '✓' : '≠', gx, gy + 0.5, { size: 10, font: 'mono', weight: 700, align: 'center', base: 'middle', color: onFill(C) });
            }
          });
          // outcomes
          if (hangNow) {
            const hx = SX(5) + (narrow ? 0 : 30), hy = (yl[0] + yl[1]) / 2;
            V.line(g, cx, yl[0] + 8, cx, yl[1] - 8, { color: C.red, lw: 1.6, dash: [3, 3], dashOffset: -t * 24 });
            if (!narrow) {
              g.beginPath(); g.arc(hx + 14, hy, 9, 0, Math.PI * 2); g.fillStyle = C.bg; g.fill(); g.strokeStyle = C.red; g.lineWidth = 2; g.stroke();
              V.line(g, hx + 14, hy, hx + 14 + Math.cos(t * 3) * 6.5, hy + Math.sin(t * 3) * 6.5, { color: C.red, lw: 2 });
              V.text(g, T.hang, hx + 28, hy + 4, { size: 11, color: C.red, weight: 600 });
            } else {
              const kx = V.lerp(SX(5), SX(6), 0.8);
              g.beginPath(); g.arc(kx, hy, 8, 0, Math.PI * 2); g.fillStyle = C.bg; g.fill(); g.strokeStyle = C.red; g.lineWidth = 2; g.stroke();
              V.line(g, kx, hy, kx + Math.cos(t * 3) * 5.5, hy + Math.sin(t * 3) * 5.5, { color: C.red, lw: 2 });
            }
          }
          if (E.kind === 'refuse' && E.at === 0) {
            const ox = SX(6);
            [0, 1].forEach((r) => { V.glow(g, ox, yl[r], 14, C.e, 0.5 + 0.25 * Math.sin(t * 3)); V.dot(g, ox, yl[r], 5, C.e); });
            V.text(g, T.oldPair, ox + 6, yl[0] - 14, { size: 10.5, align: 'right', color: C.e, weight: 600 });
          }
          if (E.kind === 'fix' && pr >= 6) {
            [0, 1].forEach((r) => V.glow(g, SX(6), yl[r], 22, C.e, 0.45 + 0.2 * Math.sin(t * 3)));
            V.text(g, T.serving, SX(6) + 6, yl[0] - 14, { size: 10.5, align: 'right', color: C.e, weight: 600 });
          }
          para(g, E.txt, pad + 6, y0 + 132, w - 2 * pad - 16, { size: 12, color: C.ink, lh: 16 });
        });
      });
    })();
  },
});

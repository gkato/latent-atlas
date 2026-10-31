LA.chapter({
  id: 'receita-qwen',
  part: 9,
  order: 43,
  title: 'Rebuilding the Qwen3.8-Flash-Next NVFP4 recipe',
  navTitle: 'Qwen recipe',
  subtitle: 'From the byte budget to /health: every decision in the dual-Spark Qwen3.8-Flash-Next recipe — lanes, checkpoints, metadata fixes, vLLM arguments and measured numbers — tied to the concept that explains it.',
  minutes: 55,
  recipes: ['qwen'],
  tags: ['recipe', 'Qwen3.8-Flash-Next', 'NVFP4', 'lanes', 'vLLM 0.30', 'TP2+EP', 'MTP', 'KV FP8', 'precedence', 'overlays'],
  goals: [
    'Redo the per-node <b>byte budget</b> and explain why the recipe uses TP=2+EP, GMU 0.80 and a 262,144-token context',
    'Tell apart the <b>lanes</b> (day-0, vLLM 0.30, official FP8, TP1) and the <b>precedence</b> between <code>.env</code>, environment and <code>OVERRIDE_*</code>',
    'Explain each checkpoint and loader fix (PLE, MTP alias, block FP8, MXFP8 fallback, nested YaRN) and recognize the crash it prevents',
    'Justify each <code>vllm serve</code> argument and read the boot, the logs and the measured numbers to tune the recipe',
  ],
  prereqs: ['engenharia-de-receita'],
  summary: [
    'The budget rules: a 123.5 GiB checkpoint does not fit in a 121.69 GiB Spark; with TP=2+EP each node holds 64.46 GiB of weights and the KV is <b>whatever is left</b> of GMU × 121.69 − (weights + non-torch) − activation; away from 0.835 the tokens are an extrapolation, and lane 0.30 measured less (2.64 M FP8 at 0.80).',
    'GMU 0.80 (not 0.835) because, in unified memory, <code>MemAvailable</code> dropped to 0.3–0.9 GiB under load — the GB10 hard-reset zone; decode does not depend on the GMU.',
    'A lane is image × checkpoint × overlays × defaults validated together, activated by a wrapper that exports <code>OVERRIDE_*</code>; the <code>.env</code> beats the environment, except for <code>ABLIT</code> and <code>HF_TOKEN</code>.',
    'On the day-0 image the nvidia checkpoint only loads with three fixes: the PLE dtype via <code>--hf-overrides</code> nested in <code>text_config</code>, the <code>mtp.layers.0 → 48</code> alias and the FP8 branch in 128×128 blocks. The MXFP8 fallback (shapes outside the kernel envelope) serves checkpoints with MXFP8 layers and is always mounted, probably inert on nvidia.',
    'The arguments translate concepts: TP2+EP (with DP=1, EP only changes where the weights live), MTP=3 with a 47k draft vocabulary, FP8 KV (1.70× tokens, −5%), CUDA graphs only in decode, replicated vision (<code>data</code>).',
    'Preflights trade 7–10 min crashes for failures that take seconds. At batch 1, ~41 of the ~60 ms of a step go to reading weights (9.2 GiB) and the rest is fixed overhead; with ā ≈ 3.2 tokens per step that gives the measured 52 tok/s. Tuning means reading fewer bytes, accepting more tokens, serving more sequences or shrinking the overhead.',
  ],
  quiz: [
    { q: 'You run <code>KV_CACHE_DTYPE=auto ./start.sh</code> and the <code>.env</code> contains <code>KV_CACHE_DTYPE=fp8</code>. What does vLLM receive?', o: ['<code>auto</code>, because the command line always wins', '<code>fp8</code>, because the <code>.env</code> is read after the environment and wins for the keys it defines', 'An error, because the variable is defined twice', '<code>auto</code>, because this key is captured as <code>_CLI_*</code>'], a: 1, e: '<code>start.sh</code> runs <code>source .env</code> after reading the environment; only <code>ABLIT</code> and <code>HF_TOKEN</code> are captured beforehand and reapplied. That is why the recipe tells you to edit the <code>.env</code> and check the “KV dtype:” line in the summary. To change the KV per lane there is <code>OVERRIDE_KV_CACHE_DTYPE</code>, read after the <code>.env</code>.' },
    { q: 'Why does the recipe use <code>--mm-encoder-tp-mode data</code>?', o: ['To speed up image prefill', 'Because with <code>weights</code> the vision MLP (4304) would become 2152 per GPU, which is not a multiple of 16, and the load crashes', 'Because the vision encoder does not fit on one GPU', 'Because NCCL cannot carry vision tensors'], a: 1, e: '4304 / 2 = 2152 and 2152 mod 16 = 8. The <code>data</code> mode replicates the 0.84 GiB encoder on each GPU and sidesteps the divisibility requirement — a “divisibility tax” of tensor parallelism.' },
    { q: 'Without the config patch, why did the MTP die ~7 minutes after loading started?', o: ['The drafter did not fit in memory', 'vLLM creates the layer as <code>mtp.layers.48</code> and looks up its quantization by exact string; the checkpoint only annotates <code>mtp.layers.0</code>, so the MoE is built without quantization and <code>w2_weight_scale_inv</code> is missing', 'MTP requires YaRN to be on', 'NCCL timed out during loading'], a: 1, e: 'Absolute index (<code>num_hidden_layers + i</code> = 48) versus relative metadata (0): the lookup silently returns <code>None</code> and the error only surfaces when the loader looks for the FP8 scales. The <code>.0 → .48</code> alias and the <code>FP8_BLOCK_SCALES</code> branch fix it; <code>--mtp-moe-algo</code> brings the failure forward to seconds.' },
    { q: 'On the vLLM 0.30 lane, what is the default KV dtype and what does changing it cost?', o: ['FP8, at no cost', 'BF16 (the 0.30 QSA kernels only accept BF16); with <code>OVERRIDE_KV_CACHE_DTYPE=fp8</code> the vllm#55557 backport gives 2.64 M tokens at ~5% less speed', 'NVFP4, with a quality loss', 'BF16, and FP8 is impossible on this lane'], a: 1, e: 'The lane exports <code>OVERRIDE_KV_CACHE_DTYPE=auto</code> because the stock 0.30 QSA kernels refuse FP8. The opt-in mounts the backport (to be deleted on 0.31+): from ~1.34–1.52 M to 2.64 M tokens, ~5% slower, NLL 1.328/1.334.' },
    { q: 'Does the drafter’s reduced 47,149-id vocabulary change the generated text?', o: ['Yes: the model now generates only those tokens', 'No: a draft outside the set is rejected at verification; only the cost of reading the drafter’s <code>lm_head</code> goes down (0.59 → 0.22 GiB on the critical rank)', 'Yes, but only for Chinese text', 'No, because it only applies to prefill'], a: 1, e: 'The target model decides every token, with the full vocabulary. The drafter only proposes; if it proposes worse, acceptance drops, but the output does not change. Measured gain: +8.4% to +9.6%. Careful: a “padded” vocabulary (65k ids with only 8.3k real ones) dragged acceptance down to 47.8%.' },
    { q: 'Why don’t the serving containers have <code>--restart</code>?', o: ['Docker does not support restart with a GPU', 'A rank restarted on its own sits waiting for its partner at NCCL init; it is better for a supervisor to bring both down and run <code>start.sh --launch</code>', 'A restart would erase the on-disk KV cache', 'Because the container never goes down'], a: 1, e: 'In two-node TP the ranks must join the rendezvous together. An isolated restart hangs. The suggested path is <code>stop.sh</code> + <code>start.sh --launch</code>, which skips download and sync and reapplies the overlays on both nodes.' },
  ],
  mount(root, LA) {
    const V = LA.viz, UI = LA.ui;
    const nf = (x, d = 0) => Number(x).toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d });
    const setFont = (g, o) => { g.font = V.font(o.size || 12, o.font || 'sans', o.weight || 400); };
    const tw = (g, s, o) => { setFont(g, o || {}); return g.measureText(String(s)).width; };
    const wrapText = (g, s, x, y, maxW, lh, o = {}) => {
      const words = String(s).split(/\s+/);
      let line = '', yy = y;
      setFont(g, o);
      for (const wd of words) {
        const t = line ? line + ' ' + wd : wd;
        if (line && g.measureText(t).width > maxW) { V.text(g, line, x, yy, o); setFont(g, o); yy += lh; line = wd; } else line = t;
      }
      if (line) { V.text(g, line, x, yy, o); yy += lh; }
      return yy;
    };
    const countLines = (g, s, maxW, o = {}) => {
      const words = String(s).split(/\s+/); let line = '', n = 0;
      setFont(g, o);
      for (const wd of words) { const t = line ? line + ' ' + wd : wd; if (line && g.measureText(t).width > maxW) { n++; line = wd; } else line = t; }
      return n + (line ? 1 : 0);
    };
    const fitText = (g, s, maxW, o) => {
      s = String(s);
      if (tw(g, s, o) <= maxW) return s;
      while (s.length > 2 && tw(g, s + '…', o) > maxW) s = s.slice(0, -1);
      return s + '…';
    };
    const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    const beforeCap = (fig, el) => fig.insertBefore(el, fig.querySelector('figcaption'));

    /* =====================================================================
       FIG 1 — memory budget of one node
       ===================================================================== */
    (function () {
      const fig = root.querySelector('#fig-orcamento');
      const S = { gmu: 0.80, kv: 'fp8', ctx: 262144 };
      const ctl = UI.controls(fig);
      UI.slider(ctl, { label: 'GMU', min: 0.70, max: 0.90, step: 0.005, value: 0.80, fmt: (v) => v.toFixed(3), onInput: (v) => { S.gmu = v; H.redraw(); } });
      UI.seg(ctl, { label: 'KV', options: [{ v: 'fp8', label: 'fp8' }, { v: 'bf16', label: 'auto (BF16)' }], value: 'fp8', onChange: (v) => { S.kv = v; H.redraw(); } });
      UI.seg(ctl, { label: 'max-model-len', options: [{ v: 262144, label: '262,144' }, { v: 1000000, label: '1M (YaRN)' }], value: 262144, onChange: (v) => { S.ctx = v; H.redraw(); } });
      const rPool = UI.readout(ctl, 'KV pool', '—');
      const rTok = UI.readout(ctl, 'Tokens (extrap.)', '—');
      const rConc = UI.readout(ctl, 'Concurrency', '—');
      // real KV token measurements (qwen-launch §4.2): [gmu, min, max, short label, long label]
      const MEAS = {
        fp8: [[0.835, 3.652, 3.652, '0.835 · day-0 · SSM fp32', '0.835 · day-0 · SSM float32 (τ comes from here)'], [0.835, 4.238, 4.379, '0.835 · day-0 · SSM bf16', '0.835 · day-0 · SSM bf16 (runs 09-10 and 09-16)'], [0.80, 2.64, 2.64, '0.80 · lane 0.30', '0.80 · lane vLLM 0.30 (FP8 KV backport)']],
        bf16: [[0.835, 2.131, 2.131, '0.835 · day-0 · SSM fp32', '0.835 · day-0 · SSM float32 (τ comes from here)'], [0.835, 2.426, 2.426, '0.835 · day-0 · SSM bf16', '0.835 · day-0 · SSM bf16 (rerun 09-16)'], [0.80, 1.34, 1.52, '0.80 · lane 0.30', '0.80 · lane vLLM 0.30 (three measurements: 1.34–1.52 M)']],
      };
      const H = V.fig(fig, (g, w, h) => {
        const C = V.C();
        const TOT = 121.69, WGT = 64.46, NT = 68.52 - 64.46, ACT = 1.07, GR = 0.54;
        const budget = S.gmu * TOT;
        const kv = Math.max(0, budget - 68.52 - ACT - (S.kv === 'bf16' ? 0.32 : 0));
        const tpg = S.kv === 'fp8' ? 114060 : 67229;
        const tokens = kv * tpg, conc = tokens / S.ctx;
        rPool.set(nf(kv, 2) + ' GiB'); rTok.set(nf(tokens / 1e6, 2) + ' M'); rConc.set(nf(conc, 2) + '×');
        const narrow = w < 560, m = 16, bx = m, bw = w - 2 * m, sx = bw / TOT;
        V.text(g, 'ONE NODE · 121.69 GiB VISIBLE TO CUDA', bx, 20, { size: 11, font: 'mono', color: C.muted });
        const by = 52, bh = 46;
        const segs = [
          { l: 'Weights', v: WGT, c: C.b },
          { l: 'Non-torch', v: NT, c: C.f },
          { l: 'Activation (peak)', v: ACT, c: C.d },
          { l: 'KV cache', v: kv, c: C.a },
          { l: 'CUDA graphs', v: GR, c: C.e },
          { l: narrow ? 'Off-budget' : 'Outside the budget', v: Math.max(0, TOT - WGT - NT - ACT - kv - GR), c: C.muted },
        ];
        g.save(); V.rr(g, bx, by, bw, bh, 10); g.clip();
        g.fillStyle = V.alpha(C.muted, 0.1); g.fillRect(bx, by, bw, bh);
        let x = bx;
        segs.forEach((sg, i) => {
          const ww = sg.v * sx;
          g.fillStyle = V.alpha(sg.c, i === 5 ? 0.16 : i === 3 ? 0.62 : 0.45); g.fillRect(x, by, ww, bh);
          if (i === 3) { g.strokeStyle = C.a; g.lineWidth = 2; g.strokeRect(x + 1, by + 1, Math.max(0, ww - 2), bh - 2); }
          const lab = i === 0 ? 'Weights ' + nf(sg.v, 2) : i === 3 ? 'KV ' + nf(sg.v, 1) : '';
          if (lab && tw(g, lab, { size: 12, weight: 600 }) < ww - 10) V.text(g, lab, x + 8, by + bh / 2 + 4, { size: 12, weight: 600, color: C.ink });
          x += ww;
        });
        g.restore();
        // budget line
        const xB = bx + budget * sx;
        V.line(g, xB, by - 12, xB, by + bh + 8, { color: C.ink, lw: 1.5, dash: [4, 3] });
        const bl = 'GMU × 121.69 = ' + nf(budget, 2) + ' GiB';
        const blw = tw(g, bl, { size: 11, font: 'mono' });
        V.text(g, bl, Math.min(w - m, Math.max(m + blw, xB - 6)), by - 16, { size: 11, font: 'mono', align: 'right', color: C.ink });
        // legend
        const cols = narrow ? 2 : 3, lw = (w - 2 * m) / cols;
        let ly = by + bh + 28;
        segs.forEach((sg, i) => {
          const cx = m + (i % cols) * lw, cy = ly + Math.floor(i / cols) * 20;
          V.box(g, cx, cy - 9, 11, 11, { r: 3, fill: V.alpha(sg.c, i === 5 ? 0.3 : 0.75) });
          const txt = sg.l + (i === 4 && !narrow ? '*' : '') + ' ' + nf(sg.v, 2);
          V.text(g, fitText(g, txt, lw - 20, { size: 11.5 }), cx + 17, cy, { size: 11.5, color: C.ink });
        });
        ly += Math.ceil(segs.length / cols) * 20 + 6;
        if (!narrow) { V.text(g, '* by the log\'s arithmetic, the graphs sit outside the budgeted sum', m, ly, { size: 11, color: C.muted }); ly += 26; }
        else ly += 6;
        // request capsules
        const lab = narrow ? nf(conc, 2) + ' full contexts fit (amber = 8 seqs)' : nf(S.ctx) + '-token requests that fit together (extrapolated): ' + nf(conc, 2) + ' (amber outline = the 8 that --max-num-seqs lets run)';
        ly = wrapText(g, lab, m, ly, w - 2 * m, 16, { size: 12, weight: 500, color: C.ink }) + 4;
        const cw = narrow ? 20 : 28, chh = 16, gap = 5;
        const maxCaps = Math.floor((w - 2 * m + gap) / (cw + gap));
        const full = Math.floor(conc), frac = conc - full;
        const shown = Math.min(maxCaps, Math.ceil(conc));
        for (let k = 0; k < shown; k++) {
          const cx = m + k * (cw + gap);
          const fillW = k < full ? cw : cw * frac;
          V.box(g, cx, ly, cw, chh, { r: 8, fill: V.alpha(C.a, 0.12), stroke: k < 8 ? C.d : V.alpha(C.a, 0.5), lw: k < 8 ? 2 : 1 });
          if (fillW > 1) { g.save(); V.rr(g, cx, ly, cw, chh, 8); g.clip(); g.fillStyle = V.alpha(C.a, 0.6); g.fillRect(cx, ly, fillW, chh); g.restore(); }
        }
        if (Math.ceil(conc) > maxCaps) V.text(g, '+' + (Math.ceil(conc) - maxCaps), m + maxCaps * (cw + gap) - 4, ly - 4, { size: 11, align: 'right', color: C.muted });
        ly += chh + 26;
        // token ruler: extrapolated (▼) × measured (◆)
        const meas = MEAS[S.kv];
        const atG = (gm) => Math.abs(gm - S.gmu) < 0.0026;
        V.text(g, narrow ? 'KV TOKENS · ▼ extrapolated × ◆ measured' : 'KV TOKENS (' + (S.kv === 'fp8' ? 'fp8' : 'BF16') + ') · ▼ linear extrapolation × ◆ real measurements', m, ly, { size: 11, font: 'mono', color: C.muted, maxW: w - 2 * m });
        const axY = ly + 40, ax0 = m + 4, ax1 = w - m - 4, TMAX = 5;
        const tx = (v) => ax0 + (Math.min(TMAX, v) / TMAX) * (ax1 - ax0);
        V.line(g, ax0, axY, ax1, axY, { color: V.alpha(C.muted, 0.6), lw: 1.5 });
        for (let k = 0; k <= TMAX; k++) { const xx = tx(k); V.line(g, xx, axY - 4, xx, axY + 4, { color: V.alpha(C.muted, 0.7), lw: 1 }); V.text(g, k + ' M', xx, axY + 17, { size: 11, font: 'mono', align: k === 0 ? 'left' : k === TMAX ? 'right' : 'center', color: C.muted }); }
        const diamond = (xx, yy, r, fill, stroke) => { g.beginPath(); g.moveTo(xx, yy - r); g.lineTo(xx + r, yy); g.lineTo(xx, yy + r); g.lineTo(xx - r, yy); g.closePath(); if (fill) { g.fillStyle = fill; g.fill(); } if (stroke) { g.strokeStyle = stroke; g.lineWidth = 1.6; g.stroke(); } };
        meas.forEach(([gm, lo, hi]) => {
          const on = atG(gm), col = on ? C.d : V.alpha(C.muted, 0.9);
          if (hi > lo) V.line(g, tx(lo), axY, tx(hi), axY, { color: col, lw: 5 });
          diamond(tx((lo + hi) / 2), axY, on ? 7 : 5.5, on ? C.d : C.bg, col);
        });
        // extrapolated
        const xe = tx(tokens / 1e6);
        g.beginPath(); g.moveTo(xe, axY - 3); g.lineTo(xe - 7, axY - 15); g.lineTo(xe + 7, axY - 15); g.closePath(); g.fillStyle = C.a; g.fill();
        const el = '▼ ' + nf(tokens / 1e6, 2) + ' M';
        const elw = tw(g, el, { size: 11, font: 'mono', weight: 700 });
        V.text(g, el, Math.max(m + elw / 2, Math.min(w - m - elw / 2, xe)), axY - 20, { size: 11, font: 'mono', weight: 700, align: 'center', color: C.a });
        ly = axY + 36;
        meas.forEach(([gm, lo, hi, sl, ll]) => {
          const on = atG(gm);
          diamond(m + 6, ly - 4, 5, on ? C.d : null, on ? C.d : C.muted);
          const val = hi > lo ? nf(lo, 2) + '–' + nf(hi, 2) + ' M' : nf(lo, 2) + ' M';
          V.text(g, fitText(g, val + ' · ' + (narrow ? sl : ll), w - 2 * m - 18, { size: 11, font: 'mono' }), m + 18, ly, { size: 11, font: 'mono', weight: on ? 700 : 400, color: on ? C.ink : C.muted });
          ly += 16;
        });
        ly += 8;
        // status
        let st, col;
        const gm = S.gmu;
        if (gm > 0.8375) { st = narrow ? 'Above 0.835: less headroom than the point where the host nearly went down.' : 'Above 0.835: less headroom than the point where the host already nearly went down (0.85 was the historical value).'; col = C.red; }
        else if (gm >= 0.8325) { st = narrow ? '0.835 (measured): MemAvailable fell to 0.3–0.9 GiB in a soak, the hard-reset zone. The KV is the logged value.' : '0.835 (measured): the driver held 114.5 GiB/node and MemAvailable fell to 0.3–0.9 GiB in a soak — the GB10 hard-reset zone. The KV in this bar is the logged value.'; col = C.red; }
        else if (gm >= 0.7975 && gm <= 0.8025) { st = narrow ? '0.80: the headroom was measured (1 h soak, lane 0.30: MemAvailable ≥ 4.5/7.8 GiB). The bar\'s KV is extrapolated; the ◆ are the measured values.' : '0.80 — what was measured here was the headroom: a 1 h soak (lane 0.30) with a minimum MemAvailable of 4.5 GiB (head) and 7.8 GiB (worker). The current recipe value. The bar\'s KV is extrapolated; the ◆ are the measured values.'; col = C.e; }
        else if (gm > 0.8025) { st = narrow ? 'Between the measured points: not validated. +0.01 takes ~1.2 GiB of headroom.' : 'Between the measured points: not validated. Each +0.01 takes ~1.2 GiB away from the system headroom.'; col = C.d; }
        else { st = narrow ? 'More headroom than validated: less KV (~1.2 GiB per −0.01). Same decode.' : 'More headroom than validated: safer, less KV (~1.2 GiB less per −0.01). Same decode.'; col = C.muted; }
        const nl = countLines(g, st, w - 2 * m - 24, { size: 12 });
        V.box(g, m, ly, w - 2 * m, nl * 16 + 14, { r: 10, fill: V.alpha(col, 0.12), stroke: V.alpha(col, 0.6) });
        wrapText(g, st, m + 12, ly + 19, w - 2 * m - 24, 16, { size: 12, color: C.ink });
      }, { static: true });
    })();

    /* =====================================================================
       FIG 3 — topology
       ===================================================================== */
    (function () {
      const fig = root.querySelector('#fig-topologia');
      const S = { plane: 'dados', cross: true };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Plane', options: [{ v: 'dados', label: 'data' }, { v: 'controle', label: 'control' }, { v: 'pesos', label: 'weights & ops' }, { v: 'api', label: 'API' }], value: 'dados', onChange: (v) => { S.plane = v; } });
      UI.toggle(ctl, { label: 'cross-wiring (head f1 ↔ worker f0)', value: true, onChange: (v) => { S.cross = v; } });
      const parts = Array.from({ length: 30 }, (_, i) => ({ p: i / 30, dir: i % 2, sp: 0.55 + (i % 5) * 0.1 }));
      const INFO = {
        dados: { t: 'DATA — NCCL collectives over RDMA (RoCE v2)', lines: (c) => [
          'NCCL_IB_HCA = ' + (c ? '=rocep1s0f1 (head) · =rocep1s0f0 (worker)' : '=rocep1s0f0 (on both)'),
          'NCCL_IB_GID_INDEX = 3 (try 5 if it does not come up) · NCCL_IB_DISABLE=0 · NCCL_IB_AUTO_DETECT=0',
          'container: --device /dev/infiniband · --ulimit memlock=-1',
          '≈106 collectives per decode step, 5–20 KB each, 20–30 µs → 2–3 ms/step',
        ] },
        controle: { t: 'CONTROL — rendezvous, Gloo, bootstrap and ZMQ (TCP)', lines: (c) => [
          c ? 'IFACE = enp1s0f1np1 (head) · WORKER_IFACE = enp1s0f0np0' : 'IFACE = enp1s0f0np0 (on both)',
          'GLOO_SOCKET_IFNAME = NCCL_SOCKET_IFNAME = TP_SOCKET_IFNAME = IFACE',
          '--master-addr $HEAD_IP --master-port 50000 · VLLM_HOST_IP = the IP of each node',
          'scheduler → worker broadcast every step: ~0.15–0.4 ms round trip',
        ] },
        pesos: { t: 'WEIGHTS & OPS — SSH, rsync/NFS and image', lines: () => [
          'WORKER_USER: passwordless SSH head → worker (prerequisite)',
          'rsync ~126 GiB to the worker disk (default) or NFS with --nfs',
          'NFS_SERVER_IP: the CX7 IP (e.g. 10.0.22.1), never the 10.0.0.1 alias',
          'docker save … | ssh worker docker load · worker script via ssh "bash -s"',
        ] },
        api: { t: 'API — only the head serves', lines: () => [
          'cliente → head:8888 (--host 0.0.0.0 --port 8888)',
          'worker: --headless, no API server',
          '/v1/chat/completions · /v1/models · /health · /metrics',
          'no authentication by default: expose with care',
        ] },
      };
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const narrow = w < 600, m = 14;
        const top = 46, nh = narrow ? 170 : 190;
        const gap = narrow ? 44 : Math.min(230, Math.max(120, w * 0.22));
        const nw = Math.min(290, (w - 2 * m - gap) / 2);
        const x0 = (w - (2 * nw + gap)) / 2, x1 = x0 + nw + gap;
        const plane = S.plane;
        const pc = plane === 'dados' ? C.a : plane === 'controle' ? C.b : plane === 'pesos' ? C.d : C.c;
        const ports = {};
        const drawNode = (x, head) => {
          V.box(g, x, top, nw, nh, { r: 14, fill: V.alpha(C.qwen, 0.06), stroke: V.alpha(C.qwen, 0.55) });
          V.text(g, head ? 'HEAD · rank 0' : 'WORKER · rank 1', x + 11, top + 20, { size: narrow ? 11.5 : 12.5, weight: 700, color: C.qwen });
          if (!head) V.text(g, '--headless', x + nw - 10, narrow ? top + 36 : top + 20, { size: 11, font: 'mono', align: 'right', color: plane === 'api' ? C.c : C.muted });
          V.text(g, narrow ? (head ? '10.0.0.1' : '10.0.0.2') : head ? '10.0.0.1 · CX7 10.0.22.1' : '10.0.0.2 · CX7 10.0.22.2', x + 11, top + 36, { size: 11, font: 'mono', color: C.muted, maxW: nw - 20 });
          V.box(g, x + 9, top + 46, nw - 18, 44, { r: 9, fill: V.alpha(C.b, 0.12), stroke: V.alpha(C.b, 0.45) });
          V.text(g, narrow ? 'GPU GB10 (sm_121)' : 'GPU GB10 · sm_121 · 48 SMs', x + 18, top + 64, { size: 11.5, weight: 600, color: C.ink, maxW: nw - 34 });
          V.text(g, narrow ? '121.69 GiB UMA' : '121.69 GiB unified · ~273 GB/s', x + 18, top + 81, { size: 11, color: C.muted, maxW: nw - 34 });
          const ny = top + 100, nhh = nh - 110;
          V.box(g, x + 9, ny, nw - 18, nhh, { r: 9, fill: V.alpha(pc, plane === 'api' ? 0.04 : 0.1), stroke: V.alpha(pc, 0.5) });
          V.text(g, narrow ? 'CX7 · RoCE v2' : 'ConnectX-7 · RoCE v2 · 200 GbE', x + 18, ny + 17, { size: 11, weight: 600, color: C.ink, maxW: nw - 50 });
          const px = head ? x + nw - 9 : x + 9;
          const act = head ? (S.cross ? 1 : 0) : 0;
          [0, 1].forEach((k) => {
            const py = ny + nhh * (k === 0 ? 0.5 : 0.8);
            const on = k === act;
            V.dot(g, px, py, on ? 6.5 : 5, on ? pc : V.alpha(C.muted, 0.5));
            V.dot(g, px, py, 2.4, C.bg || '#000');
            const lx = head ? px - 12 : px + 12;
            V.text(g, 'f' + k + (narrow ? '' : ' · rocep1s0f' + k), lx, py + 4, { size: 11, font: 'mono', align: head ? 'right' : 'left', color: on ? C.ink : C.muted });
            ports[(head ? 'h' : 'w') + k] = [px, py];
          });
        };
        drawNode(x0, true); drawNode(x1, false);
        const hp = ports['h' + (S.cross ? 1 : 0)], wp = ports.w0;
        const c1 = [hp[0] + gap * 0.45, hp[1]], c2 = [wp[0] - gap * 0.45, wp[1]];
        g.beginPath(); g.moveTo(hp[0], hp[1]); g.bezierCurveTo(c1[0], c1[1], c2[0], c2[1], wp[0], wp[1]);
        g.strokeStyle = plane === 'api' ? V.alpha(C.muted, 0.4) : V.alpha(pc, 0.55); g.lineWidth = plane === 'dados' ? 5 : 3;
        if (plane === 'controle') g.setLineDash([6, 5]);
        g.stroke(); g.setLineDash([]);
        const P0 = hp, P1 = c1, P2 = c2, P3 = wp;
        if (plane === 'dados') {
          parts.forEach((p) => { const u = (p.p + t * p.sp * 0.9) % 1; const pt = V.bez(P0, P1, P2, P3, p.dir ? u : 1 - u); V.dot(g, pt[0], pt[1], 2.8, p.dir ? C.ink : C.d); });
        } else if (plane === 'controle') {
          parts.slice(0, 6).forEach((p, i) => { const u = (p.p * 5 + t * 0.35) % 1; const pt = V.bez(P0, P1, P2, P3, i % 2 ? u : 1 - u); V.dot(g, pt[0], pt[1], 4.5, C.ink); V.dot(g, pt[0], pt[1], 2.5, C.b); });
        } else if (plane === 'pesos') {
          for (let k = 0; k < 3; k++) { const u = (k / 3 + t * 0.12) % 1; const pt = V.bez(P0, P1, P2, P3, u); V.box(g, pt[0] - 9, pt[1] - 6, 18, 12, { r: 3, fill: V.alpha(C.d, 0.85) }); }
        }
        if (plane === 'api') {
          const cxp = x0 + 8, cyp = 10, cwp = Math.min(nw - 16, 170);
          V.box(g, cxp, cyp, cwp, 24, { r: 12, fill: V.alpha(C.c, 0.18), stroke: C.c, label: 'client (OpenAI API)', size: 11 });
          const ax = x0 + nw * 0.7;
          V.arrow(g, cxp + cwp / 2, cyp + 24, ax, top - 2, { color: C.c, lw: 2 });
          const u = (t * 0.8) % 1;
          V.dot(g, V.lerp(cxp + cwp / 2, ax, u), V.lerp(cyp + 24, top - 2, u), 3.5, C.c);
          V.text(g, ':8888', ax + 6, top - 8, { size: 11, font: 'mono', color: C.c });
        } else {
          V.text(g, (S.cross ? 'cross-wired: f1 ↔ f0' : 'port to port: f0 ↔ f0'), w / 2, 22, { size: 11, font: 'mono', align: 'center', color: C.muted });
        }
        // variables panel
        const inf = INFO[plane];
        let y = top + nh + 24;
        V.text(g, inf.t, m, y, { size: 11.5, weight: 700, color: pc, maxW: w - 2 * m });
        y += 18;
        inf.lines(S.cross).forEach((ln) => { V.dot(g, m + 4, y - 4, 2.5, pc); y = wrapText(g, ln, m + 14, y, w - 2 * m - 14, 15.5, { size: 11.5, font: 'mono', color: C.ink }) + 3; });
      });
    })();

    /* =====================================================================
       FIG 4 — decision map: lanes as subway lines
       ===================================================================== */
    (function () {
      const fig = root.querySelector('#fig-lanes');
      const LN = {
        day0: { name: 'day-0', cv: '--qwen', wrapper: './start.sh', image: 'vllm/vllm-openai:qwen38-flash-next — NVIDIA development build (~20.6 GB), code in qwen3_8_flash_next/nvidia', ckpt: 'nvidia/Qwen3.8-Flash-Next-NVFP4', overlays: 'PLE, MXFP8 + block FP8 for the MoE, MTP alias, FP8 KV in the QSA kernels, block-drop, 47k draft vocabulary (and determinism, optional)', kv: 'fp8 (patch_qsa_fp8_kv.py applied automatically)', res: '3,652,200 KV tokens at GMU 0.835 (13.93× at 262k); decode S=1 prose/code 56.1/75.8 tok/s; 2nd-turn TTFT 0.63 s', when: 'The largest number of long sessions in cache; it is the only lane with determinism and a QSA profile.' },
        v030: { name: 'vLLM 0.30', cv: '--viz-b', wrapper: './start-v030.sh', image: 'vllm/vllm-openai:v0.30.0 — stable release, native qwen4_exp', ckpt: 'nvidia/Qwen3.8-Flash-Next-NVFP4 (loaded natively)', overlays: 'only the draft vocabulary (patch_mtp_draft_vocab_v030.py) and, optionally, the FP8 KV backport (vllm#55557)', kv: 'auto = BF16 by default; fp8 opt-in via OVERRIDE_KV_CACHE_DTYPE=fp8 (2.64 M tokens, ~5% slower)', res: '1.34–1.52 M BF16 tokens at GMU 0.80; prose 60.0 vs 56.3 tok/s (S=1), code 441 vs 422 (S=8); multi-turn TTFT 0.28–0.40 s; #62 fixed upstream', when: 'Agents and multi-turn conversations: half the TTFT and equal or higher decode.' },
        fp8: { name: 'Official FP8', cv: '--viz-d', wrapper: './start-fp8.sh', image: 'the one in .env (day-0)', ckpt: 'Qwen/Qwen3.8-Flash-Next-FP8 (FP8 experts in blocks of 128; GDN, attention, HC and lm_head stay BF16)', overlays: 'no PLE patch (SKIP_PLE_PATCH=true); the other day-0 overlays', kv: 'the one in .env; native 262,144 context, YaRN off', res: '~500K KV tokens (FP8 experts take up twice as much as NVFP4)', when: 'When you want the official Qwen weights as a quality reference.' },
        tp1: { name: 'TP1 (one Spark)', cv: '--viz-c', wrapper: './start-tp1.sh or tp1/start.sh', image: 'the one in .env (day-0)', ckpt: 'local-inference-lab/Qwen3.8-Flash-Next-NVFP4 (98.6 GiB; RadixArk, 125.9 GiB, does not fit)', overlays: 'PLE offload with a host handshake + packed NVFP4 table in mmap (26.8 GiB) + memwatch.sh', kv: 'auto (BF16) by default; GMU derived from the budget', res: '685,825 tokens at GMU 0.813; decode 22.0 → prose 24.9 / code 35.1 tok/s with MTP', when: 'There is only one Spark. Requires --distributed-executor-backend mp, a smaller checkpoint and a memory watchdog.' },
      };
      const OBJ = [
        { v: 'agentes', label: 'Multi-turn agents (low TTFT)', lane: 'v030' },
        { v: 'cache', label: 'Many long sessions in cache', lane: 'day0' },
        { v: 'ab', label: 'Bit-identical A/B, kernel experiments', lane: 'day0' },
        { v: 'oficial', label: 'Official Qwen weights', lane: 'fp8' },
        { v: 'um', label: 'I only have one Spark', lane: 'tp1' },
      ];
      const order = ['day0', 'v030', 'fp8', 'tp1'];
      const home = { day0: 0, v030: 1, fp8: 2, tp1: 3 };
      const PATH = { day0: ['w0', 'i0', 'p0', 'k0', 'r0'], v030: ['w1', 'i1', 'p0', 'k1', 'r1'], fp8: ['w2', 'i0', 'p1', 'k0', 'r2'], tp1: ['w3', 'i0', 'p2', 'k1', 'r3'] };
      const LBL = {
        w0: ['start.sh', 'start'], w1: ['start-v030.sh', 'v030'], w2: ['start-fp8.sh', 'fp8'], w3: ['start-tp1.sh', 'tp1'],
        i0: ['qwen38-flash-next', 'day-0'], i1: ['v0.30.0', 'v0.30'],
        p0: ['nvidia NVFP4', 'nvidia'], p1: ['Qwen FP8', 'FP8'], p2: ['local-inference-lab', 'lil'],
        k0: ['fp8 + patch', 'fp8'], k1: ['BF16 (auto)', 'bf16'],
        r0: ['3.65 M tok', '3.65M'], r1: ['1.34–1.52 M', '1.4M'], r2: ['~500K', '500k'], r3: ['686K', '686k'],
      };
      const HEAD = ['wrapper', 'image', 'weights', 'KV', 'KV tokens'];
      // stations: column + lanes passing through
      const ST = {};
      order.forEach((ln) => PATH[ln].forEach((sid, c) => { (ST[sid] = ST[sid] || { c, lanes: [] }).lanes.push(ln); }));
      Object.values(ST).forEach((s) => { s.slot = s.lanes.reduce((a, l) => a + home[l], 0) / s.lanes.length; });
      const S = { lane: 'v030' };
      const ctl = UI.controls(fig);
      const sel = UI.select(ctl, { label: 'Goal', options: OBJ.map((o) => ({ v: o.v, label: o.label })), value: 'agentes', onChange: (v) => { const o = OBJ.find((x) => x.v === v); if (o) seg.set(o.lane); } });
      const seg = UI.seg(ctl, { label: 'Lane', options: order.map((k) => ({ v: k, label: LN[k].name })), value: 'v030', onChange: (v) => { S.lane = v; render(); } });
      sel.el.style.flexWrap = 'wrap'; sel.el.style.maxWidth = '100%';
      const selEl = sel.el.querySelector('select'); selEl.style.maxWidth = '100%'; selEl.style.minWidth = '0';
      const panel = LA.h('div', { style: { padding: '2px 18px 12px', overflowWrap: 'anywhere' } });
      beforeCap(fig, panel);
      function render() {
        const L = LN[S.lane];
        const obj = OBJ.filter((o) => o.lane === S.lane).map((o) => o.label).join('; ');
        panel.innerHTML = `<dl class="la-kv" style="margin:.4em 0">
          <dt style="color:var(${L.cv})">lane</dt><dd><b>${esc(L.name)}</b> — ${esc(L.when)}${obj ? ' <i>(goal: ' + esc(obj) + ')</i>' : ''}</dd>
          <dt>wrapper</dt><dd><code>${esc(L.wrapper)}</code></dd>
          <dt>image</dt><dd>${esc(L.image)}</dd>
          <dt>weights</dt><dd>${esc(L.ckpt)}</dd>
          <dt>overlays</dt><dd>${esc(L.overlays)}</dd>
          <dt>KV</dt><dd>${esc(L.kv)}</dd>
          <dt>measured</dt><dd>${esc(L.res)}</dd></dl>`;
      }
      render();
      void sel;
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const col = { day0: C.qwen, v030: C.b, fp8: C.d, tp1: C.c };
        const narrow = w < 600, m = narrow ? 30 : 60;
        const topY = 58, botY = h - 44;
        const colX = (c) => m + (c * (w - 2 * m)) / 4;
        const slotY = (s) => topY + (s * (botY - topY)) / 3;
        const off = 7;
        (narrow ? ['wrapper', 'image', 'weights', 'KV', 'tokens'] : HEAD).forEach((hd, c) => V.text(g, hd.toUpperCase(), colX(c), 22, { size: 11, font: 'mono', align: 'center', color: C.muted }));
        const pos = (ln, c) => {
          const s = ST[PATH[ln][c]];
          const idx = s.lanes.indexOf(ln), n = s.lanes.length;
          return [colX(c), slotY(s.slot) + (idx - (n - 1) / 2) * off];
        };
        const segs = {};
        order.forEach((ln) => {
          const pts = [0, 1, 2, 3, 4].map((c) => pos(ln, c));
          const on = ln === S.lane;
          g.beginPath(); g.moveTo(pts[0][0], pts[0][1]);
          const sg = [];
          for (let k = 0; k < 4; k++) {
            const a = pts[k], b = pts[k + 1], mx = (a[0] + b[0]) / 2;
            g.bezierCurveTo(mx, a[1], mx, b[1], b[0], b[1]);
            sg.push([a, [mx, a[1]], [mx, b[1]], b]);
          }
          segs[ln] = sg;
          g.strokeStyle = on ? col[ln] : V.alpha(col[ln], 0.28); g.lineWidth = on ? 6 : 4; g.lineCap = 'round'; g.stroke(); g.lineCap = 'butt';
        });
        // stations
        Object.entries(ST).forEach(([sid, s]) => {
          const x = colX(s.c), y = slotY(s.slot), n = s.lanes.length;
          const onSel = s.lanes.includes(S.lane);
          if (n > 1) V.box(g, x - 9, y - (n - 1) * off / 2 - 9, 18, (n - 1) * off + 18, { r: 9, fill: C.bg || '#111', stroke: onSel ? C.ink : V.alpha(C.ink, 0.35), lw: 2 });
          else { V.dot(g, x, y, 8, C.bg || '#111'); g.strokeStyle = onSel ? col[s.lanes[0]] : V.alpha(col[s.lanes[0]], 0.4); g.lineWidth = 3; g.beginPath(); g.arc(x, y, 8, 0, Math.PI * 2); g.stroke(); }
          if (narrow && !onSel) return;
          const lab = LBL[sid][narrow ? 1 : 0];
          const below = s.slot >= 1.5;
          const ly = below ? y + (n - 1) * off / 2 + 22 : y - (n - 1) * off / 2 - 14;
          V.text(g, lab, x, ly, { size: 11, font: 'mono', align: 'center', weight: onSel ? 600 : 400, color: onSel ? C.ink : C.muted, maxW: (w - 2 * m) / 4 + 10 });
        });
        // train
        const sg = segs[S.lane];
        const u = (t * 0.1) % 1, k = Math.min(3, Math.floor(u * 4)), f = u * 4 - k;
        const p = V.bez(sg[k][0], sg[k][1], sg[k][2], sg[k][3], V.smooth(f));
        V.glow(g, p[0], p[1], 20, col[S.lane]);
        V.box(g, p[0] - 13, p[1] - 7, 26, 14, { r: 7, fill: col[S.lane], stroke: C.ink, lw: 1.2 });
        V.dot(g, p[0] - 5, p[1], 2, C.bg || '#000'); V.dot(g, p[0] + 5, p[1], 2, C.bg || '#000');
      });
    })();

    /* =====================================================================
       FIG 5 — precedence ladder
       ===================================================================== */
    (function () {
      const fig = root.querySelector('#fig-precedencia');
      const LAYERS = ['Environment (command line)', 'source .env', 'Reapply _CLI_ABLIT / _CLI_HF_TOKEN', 'start.sh ${VAR:-…} defaults', 'OVERRIDE_* (lane wrapper)', 'Final guards'];
      const VARS = {
        kv: { label: 'KV_CACHE_DTYPE', t: ['KV_CACHE_DTYPE=fp8 on the command line', '.env defines KV_CACHE_DTYPE=auto', 'run via start-v030.sh', 'export OVERRIDE_KV_CACHE_DTYPE=fp8'], def: [true, true, false, false] },
        ablit: { label: 'ABLIT', t: ['ABLIT=1 on the command line', '.env defines ABLIT=0', 'run via start-fp8.sh', 'FP8_DENSE=true in .env'], def: [true, true, false, false] },
        model: { label: 'MODEL_ID', t: ['MODEL_ID=RadixArk/… on the command line', '.env defines MODEL_ID=nvidia/…', 'run via start-fp8.sh', 'ABLIT=1 on the command line'], def: [true, true, false, false] },
        yarn: { label: 'YARN_ENABLE', t: ['YARN_ENABLE=true on the command line', '.env defines YARN_ENABLE=false', 'run via start-fp8.sh', '.env defines MAX_MODEL_LEN=1000000'], def: [true, false, false, false] },
      };
      const S = { v: 'kv', tg: [true, true, false, false], t0: null };
      function simulate() {
        const T = S.tg, out = [];
        const put = (val, changed, note) => out.push({ val, changed, note });
        if (S.v === 'kv') {
          let v;
          if (T[0]) { v = 'fp8'; put(v, true, 'KV_CACHE_DTYPE=fp8 ./start.sh'); } else put(v, false, 'not defined');
          if (T[1]) { put('auto', true, v ? 'overrides the command line!' : '.env defines the key'); v = 'auto'; } else put(v, false, '.env does not define the key');
          put(v, false, 'only applies to ABLIT and HF_TOKEN');
          if (!v) { v = 'fp8'; put(v, true, 'fallback ${KV_CACHE_DTYPE:-fp8}'); } else put(v, false, 'already has a value: fallback does nothing');
          if (T[3]) { v = 'fp8'; put(v, true, 'OVERRIDE_KV_CACHE_DTYPE=fp8'); } else if (T[2]) { v = 'auto'; put(v, true, 'start-v030.sh: OVERRIDE_KV_CACHE_DTYPE=${…:-auto}'); } else put(v, false, 'no wrapper');
          put(v, false, 'no guard for this key');
          const lane030 = T[2];
          return { rows: out, res: '--kv-cache-dtype ' + v, note: v.startsWith('fp8') ? (lane030 ? 'mounts the patch_qsa_fp8_kv_v030.py backport (vllm#55557)' : 'mounts patch_qsa_fp8_kv.py on both nodes') : 'BF16 KV: no KV patch' };
        }
        if (S.v === 'ablit') {
          let v;
          if (T[0]) { v = '1'; put(v, true, 'ABLIT=1 ./start.sh → _CLI_ABLIT=1'); } else put(v, false, 'not defined');
          if (T[1]) { put('0', true, v ? '.env erases the 1… for now' : '.env defines ABLIT=0'); v = '0'; } else put(v, false, '.env does not define it');
          if (T[0]) { const ch = v !== '1'; v = '1'; put(v, ch, 'reapplies _CLI_ABLIT: the command line wins'); } else put(v, false, 'nothing captured');
          if (!v) { v = '0'; put(v, true, 'fallback ${ABLIT:-0}'); } else put(v, false, 'already has a value');
          if (T[2]) put('ABLIT=' + v, true, 'OVERRIDE_MODEL_ID=Qwen/…-FP8'); else put('ABLIT=' + v, false, 'no wrapper');
          let model, note;
          if (T[3]) { model = 'MiaAI-Lab/…-FP8dense'; note = v === '1' ? 'ABLIT ignored: FP8_DENSE=true wins' : 'FP8_DENSE=true picks the hybrid'; }
          else if (T[2]) { model = 'Qwen/Qwen3.8-Flash-Next-FP8'; note = v === '1' ? 'ABLIT ignored: OVERRIDE_MODEL_ID wins' : 'override picks the official FP8'; }
          else if (v === '1') { model = 'drowzeys/…-ablit-house-qsa-L3-47'; note = 'ABLIT=1 picks the abliterated checkpoint'; }
          else { model = 'MODEL_ID from .env (nvidia/…)'; note = 'ABLIT=0: nothing changes'; }
          put(model, true, note);
          return { rows: out, res: 'ABLIT=' + v + ' → ' + model, note };
        }
        if (S.v === 'model') {
          let v;
          if (T[0]) { v = 'RadixArk/…'; put(v, true, 'MODEL_ID=RadixArk/… ./start.sh'); } else put(v, false, 'not defined');
          if (T[1]) { put('nvidia/…', true, v ? 'overrides the command line!' : '.env defines the key'); v = 'nvidia/…'; } else put(v, false, '.env does not define it');
          put(v, false, T[3] ? 'ABLIT=1 reapplied (applies at the guard)' : 'only ABLIT and HF_TOKEN');
          if (!v) { put('ERROR', true, 'Required variable MODEL_ID is not set in .env'); return { rows: out, res: 'launch aborted', note: 'MODEL_ID is required (validated right after source .env)', err: true }; }
          put(v, false, 'no fallback for MODEL_ID');
          if (T[2]) { v = 'Qwen/…-FP8'; put(v, true, 'OVERRIDE_MODEL_ID (start-fp8.sh)'); } else put(v, false, 'no wrapper');
          if (T[3] && !T[2]) { v = 'drowzeys/…-ablit'; put(v, true, 'ABLIT=1 picks the abliterated one'); }
          else if (T[3] && T[2]) put(v, false, 'ABLIT ignored: OVERRIDE_MODEL_ID wins');
          else put(v, false, 'FP8_DENSE > OVERRIDE > ABLIT > MODEL_ID');
          return { rows: out, res: 'serve ' + v, note: 'order: FP8_DENSE=true > OVERRIDE_MODEL_ID > ABLIT=1 > MODEL_ID' };
        }
        // yarn
        let v;
        if (T[0]) { v = 'true'; put(v, true, 'YARN_ENABLE=true ./start.sh'); } else put(v, false, 'not defined');
        if (T[1]) { put('false', true, v ? 'overrides the command line!' : '.env defines the key'); v = 'false'; } else put(v, false, '.env does not define it');
        put(v, false, 'only ABLIT and HF_TOKEN');
        if (!v) { v = 'false'; put(v, true, 'fallback ${YARN_ENABLE:-false}'); } else put(v, false, 'already has a value');
        if (T[2]) { v = 'false'; put(v, true, 'OVERRIDE_YARN_ENABLE=false + MAX_MODEL_LEN=262144'); } else put(v, false, 'no wrapper');
        const mml = T[2] ? 262144 : T[3] ? 1000000 : 262144;
        if (v === 'true' && mml <= 262144) { v = 'false'; put(v, true, 'NOTE: MAX_MODEL_LEN ≤ 262144 → YaRN force-disabled'); }
        else put(v, false, 'MAX_MODEL_LEN=' + nf(mml));
        return { rows: out, res: 'YARN_ENABLE=' + v + ' (MAX_MODEL_LEN ' + nf(mml) + ')', note: v === 'true' ? 'emits rope_parameters yarn, factor 4.0, nested in text_config' : 'Native RoPE: no rope_parameters in --hf-overrides' };
      }
      const ctl = UI.controls(fig);
      UI.select(ctl, { label: 'Variable', options: Object.entries(VARS).map(([k, o]) => ({ v: k, label: o.label })), value: 'kv', onChange: (v) => { S.v = v; VARS[v].def.forEach((d, i) => { S.tg[i] = d; togs[i].el.querySelector('input').checked = d; togs[i].el.querySelector('span').textContent = VARS[v].t[i]; }); S.t0 = null; } });
      const togs = [0, 1, 2, 3].map((i) => UI.toggle(ctl, { label: VARS.kv.t[i], value: S.tg[i], onChange: (on) => { S.tg[i] = on; S.t0 = null; } }));
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        if (S.t0 == null) S.t0 = t;
        const el = t - S.t0;
        const sim = simulate();
        const narrow = w < 560, m = 14;
        const railX = m + 14, top = 16, resH = narrow ? 64 : 56;
        const rh = (h - top - resH - 16) / 6;
        const stepDur = 0.55, n = sim.rows.length;
        const prog = Math.min(n - 1 + 0.999, el / stepDur);
        const cur = Math.floor(prog);
        const done = el > n * stepDur + 0.2;
        if (el > n * stepDur + 3.2) S.t0 = t;
        V.line(g, railX, top + rh / 2, railX, top + rh * 5.5, { color: V.alpha(C.muted, 0.4), lw: 2 });
        LAYERS.forEach((lab, i) => {
          const y = top + i * rh;
          const r = sim.rows[i];
          const reached = i <= cur || done;
          const active = reached && r && r.changed;
          V.box(g, railX + 16, y + 4, w - railX - 16 - m, rh - 8, { r: 9, fill: active ? V.alpha(C.a, 0.13) : V.alpha(C.muted, 0.05), stroke: active ? V.alpha(C.a, 0.7) : V.alpha(C.muted, 0.25) });
          V.dot(g, railX, y + rh / 2, 9, active ? C.a : reached ? V.alpha(C.muted, 0.6) : V.alpha(C.muted, 0.25));
          V.text(g, String(i + 1), railX, y + rh / 2 + 4, { size: 11, weight: 700, align: 'center', color: active ? (C.bg || '#000') : C.ink });
          const lx = railX + 28;
          if (!r) return;
          const valTxt = !reached ? '·' : r.val == null ? '—' : String(r.val);
          if (narrow) {
            V.text(g, fitText(g, lab, w - lx - m - 8, { size: 11, weight: 600 }), lx, y + 18, { size: 11, weight: 600, color: C.ink });
            V.text(g, fitText(g, valTxt + (reached ? '  · ' + r.note : ''), w - lx - m - 8, { size: 11, font: 'mono' }), lx, y + 34, { size: 11, font: 'mono', color: active ? C.ink : C.muted });
          } else {
            V.text(g, lab, lx, y + rh / 2 - 3, { size: 12, weight: 600, color: C.ink });
            if (reached) V.text(g, fitText(g, r.note, w * 0.5 - 30, { size: 11 }), lx, y + rh / 2 + 13, { size: 11, color: C.muted });
            const chip = fitText(g, valTxt, w * 0.38, { size: 12, font: 'mono', weight: 600 });
            const cw = tw(g, chip, { size: 12, font: 'mono', weight: 600 }) + 20;
            V.box(g, w - m - 10 - cw, y + rh / 2 - 12, cw, 24, { r: 12, fill: active ? V.alpha(C.a, 0.3) : V.alpha(C.muted, 0.12), stroke: active ? C.a : V.alpha(C.muted, 0.3) });
            V.text(g, chip, w - m - 10 - cw / 2, y + rh / 2 + 4, { size: 12, font: 'mono', weight: 600, align: 'center', color: C.ink });
          }
        });
        // ball
        if (!done) {
          const f = prog - cur;
          const y = top + (cur + V.ease.inOut(Math.min(1, f * 1.6))) * rh + rh / 2;
          V.glow(g, railX, Math.min(y, top + 5.5 * rh), 18, C.d);
          V.dot(g, railX, Math.min(y, top + 5.5 * rh), 5, C.d);
        }
        // result
        const ry = h - resH - 6;
        const rc = sim.err ? C.red : C.qwen;
        V.box(g, m, ry, w - 2 * m, resH, { r: 12, fill: V.alpha(rc, done ? 0.16 : 0.06), stroke: V.alpha(rc, done ? 0.8 : 0.3), lw: done ? 2 : 1 });
        V.text(g, fitText(g, 'Effective: ' + (done ? sim.res : '…'), w - 2 * m - 24, { size: 12.5, font: 'mono', weight: 700 }), m + 12, ry + 22, { size: 12.5, font: 'mono', weight: 700, color: done ? C.ink : C.muted });
        if (done) wrapText(g, sim.note, m + 12, ry + 40, w - 2 * m - 24, 14, { size: 11, color: C.muted });
      });
    })();

    /* =====================================================================
       FIG 6 — six checkpoints
       ===================================================================== */
    (function () {
      const fig = root.querySelector('#fig-checkpoints');
      const COMP = [
        { k: 'exp', l: 'Routed experts', p: 120.80 },
        { k: 'ple', l: 'N-gram table (PLE)', ls: 'N-gram (PLE)', p: 51.20 },
        { k: 'mtp', l: 'MTP (experts)', p: 2.517 },
        { k: 'gdn', l: 'GDN (36 layers)', p: 2.087 },
        { k: 'att', l: 'QSA attention + indexer', ls: 'QSA attention', p: 0.618 },
        { k: 'hc', l: 'Gated Residual (HC)', ls: 'Gated Residual', p: 0.641 },
        { k: 'lm', l: 'lm_head', p: 0.636 },
        { k: 'emb', l: 'embed_tokens', p: 0.636 },
        { k: 'sr', l: 'Shared experts + routers', ls: 'Shared + router', p: 0.299 },
        { k: 'vis', l: 'Vision (ViT)', p: 0.449 },
        { k: 'etc', l: 'Rest (dense MTP, PLE proj.)', ls: 'Rest', p: 0.124 },
      ];
      const BPP = { BF16: 2, FP8: 1, FP8B: 1, NVFP4: 0.5625, MXFP8: 1.03125 };
      const base = { exp: 'NVFP4', ple: 'FP8', mtp: 'FP8B', gdn: 'BF16', att: 'BF16', hc: 'BF16', lm: 'BF16', emb: 'BF16', sr: 'BF16', vis: 'BF16', etc: 'BF16' };
      const CK = {
        nvidia: { name: 'nvidia', fmt: Object.assign({}, base), unk: [], pub: '133 GB · 11 shards', html: 'Default checkpoint: <b>NVFP4 experts (W4A4</b>: expert weights <i>and</i> activations in 4 bits), per-tensor FP8 PLE, MTP experts in FP8 128×128 blocks, the rest BF16. <b>Triggers:</b> <code>detect_ple_dtype.py</code> → <code>float8_e4m3fn</code>, the PLE patch, the <code>mtp.layers.48</code> alias + <code>FP8_PB_WO</code> normalization, the block-FP8 branch for the MTP MoE; the MXFP8 fallback is mounted but probably inert (BF16 dense layers).' },
        lil: { name: 'local-inference-lab', fmt: Object.assign({}, base, { ple: 'NVFP4', gdn: 'MXFP8', att: 'MXFP8', vis: 'MXFP8' }), unk: ['mtp', 'vis'], pub: '98.6 GiB · 37 shards', html: 'Attention/GDN in <b>MXFP8</b> and PLE in <b>NVFP4</b> (26.8 GiB). It is the only one that fits on one Spark (TP1 lane). <b>Triggers:</b> the MXFP8 fallback becomes essential (<code>in_proj_a/b [48, 2560]</code> fall back to emulation); the PLE already declares its dtype, so the detector injects nothing. Not detailed in the sources: the MTP experts (the estimate uses FP8) and vision (the estimate uses MXFP8, because the patch sends MXFP8 <code>visual.*</code> to emulation).' },
        radix: { name: 'RadixArk', fmt: Object.assign({}, base, { mtp: 'BF16' }), unk: [], pub: '125.9 GiB · 206 shards', html: 'BF16 attention, PLE FP8 already declared and <b>MTP experts in BF16</b> (4.69 GiB). Decode 12–16% slower than local-inference-lab (more bytes), same prefill; GSM8K 97.27% and AIME26 98.75%. It is the source of FP8-dense.' },
        fp8off: { name: 'Official FP8', fmt: Object.assign({}, base, { exp: 'FP8B', mtp: 'FP8B' }), unk: [], pub: '— (not cited)', html: '<code>Qwen/Qwen3.8-Flash-Next-FP8</code>: routed experts (112.5 GiB) and MTP experts in FP8 blocks of 128 = <b>twice the bytes</b> of NVFP4, so there is KV left for ~500K tokens. Dense layers in <b>BF16</b>: the header scan (performance report) shows <code>modules_to_not_convert</code> with <code>linear_attn.*</code>, <code>hyper_connection*</code> and <code>lm_head</code>. The “block-128 over the whole checkpoint” in <code>start-fp8.sh</code> is the global <code>Fp8Config</code>. <b>Triggers:</b> <code>SKIP_PLE_PATCH=true</code>, native context, YaRN off.' },
        dense: { name: 'FP8-dense', fmt: Object.assign({}, base, { mtp: 'BF16', gdn: 'FP8', att: 'FP8', hc: 'FP8', lm: 'FP8', sr: 'FP8' }), unk: [], pub: 'local (4 rewritten BF16 shards, 11.8 GB)', html: 'Hybrid built from RadixArk: NVFP4 experts and PLE <i>hard-linked</i>, 591 dense linears in per-channel FP8 E4M3 + dynamic per-token activation. Expected 9.2 → 5.7 GiB/step (<b>analytical</b>). <b>Triggers:</b> <code>FP8_DENSE=true</code> + 4 overlays (<code>modelopt</code>, <code>model</code>, <code>hyperconnection</code>, <code>mtp</code>) — conflicts with the draft vocabulary, and the <code>apply_patches.py</code> generator is missing.' },
        ablit: { name: 'ABLIT', fmt: Object.assign({}, base), unk: [], pub: 'same layout as nvidia', html: 'Same layout as nvidia; only the 12 <code>self_attn.o_proj</code> of QSA layers 3, 7, …, 47 were edited (★). <b>Triggers:</b> <code>ABLIT=1</code> + <code>HF_TOKEN</code> + accepted terms; the same patches as nvidia. Valid in this recipe only.' },
      };
      const S = { ck: 'nvidia', anim: {} };
      COMP.forEach((c) => { S.anim[c.k] = 0; });
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Checkpoint', options: Object.entries(CK).map(([k, o]) => ({ v: k, label: o.name })), value: 'nvidia', onChange: (v) => { S.ck = v; render(); } });
      const panel = LA.h('div', { style: { padding: '2px 18px 12px', fontSize: '14px', color: 'var(--text-2)', overflowWrap: 'anywhere' } });
      beforeCap(fig, panel);
      function render() { const c = CK[S.ck]; panel.innerHTML = `<p style="margin:.3em 0"><b>${esc(c.name)}</b> — ${c.html}</p>`; }
      render();
      const GiB = 1073741824;
      const sizeOf = (ck, c) => (c.p * 1e9 * BPP[ck.fmt[c.k]]) / GiB;
      V.fig(fig, (g, w, h) => {
        const C = V.C();
        const DT = { BF16: C.b, FP8: C.a, FP8B: C.d, NVFP4: C.c, MXFP8: C.f };
        // MXFP8 = magenta with vertical stripes (in the light theme, magenta and pink are nearly indistinguishable)
        const stripes = (x, y, ww, hh, r) => { g.save(); V.rr(g, x, y, ww, hh, r); g.clip(); g.fillStyle = V.alpha(C.bg, 0.75); for (let xx = x + 3; xx < x + ww; xx += 6) g.fillRect(xx, y, 2, hh); g.restore(); g.strokeStyle = C.f; g.lineWidth = 1.4; V.rr(g, x + 0.5, y + 0.5, Math.max(0, ww - 1), hh - 1, r); g.stroke(); };
        const ck = CK[S.ck];
        const narrow = w < 560, m = 14;
        const lw = narrow ? 112 : 196, vw = narrow ? 50 : 62;
        const bx0 = m + lw, bwMax = w - m - vw - bx0;
        const top = 46, bot = narrow ? 62 : 44;
        const rh = (h - top - bot) / COMP.length;
        const maxG = 112.6;
        let total = 0;
        COMP.forEach((c) => { total += sizeOf(ck, c); });
        V.text(g, 'estimated: ' + nf(total, 1) + ' GiB', m, 20, { size: 13, weight: 700, color: C.ink });
        V.text(g, fitText(g, 'published: ' + ck.pub, w - 2 * m - 150, { size: 11.5 }), w - m, 20, { size: 11.5, align: 'right', color: C.muted });
        V.text(g, 'bar ∝ GiB (linear scale up to 112.6 GiB)', m, 36, { size: 11, color: C.muted });
        COMP.forEach((c, i) => {
          const y = top + i * rh;
          const gib = sizeOf(ck, c);
          const target = Math.max(3, (gib / maxG) * bwMax);
          S.anim[c.k] = S.anim[c.k] ? V.lerp(S.anim[c.k], target, 0.18) : target;
          const bwc = S.anim[c.k];
          const f = ck.fmt[c.k], colr = DT[f];
          V.text(g, fitText(g, narrow && c.ls ? c.ls : c.l, lw - 8, { size: 11.5 }), m, y + rh / 2 + 4, { size: 11.5, color: C.ink });
          const bh = Math.min(18, rh - 5), by = y + (rh - bh) / 2;
          V.box(g, bx0, by, bwc, bh, { r: 4, fill: V.alpha(colr, 0.7) });
          if (f === 'MXFP8') stripes(bx0, by, bwc, bh, 4);
          if (ck.unk.includes(c.k)) {
            g.save(); V.rr(g, bx0, by, bwc, bh, 4); g.clip();
            g.strokeStyle = V.alpha(C.ink, 0.45); g.lineWidth = 1.2;
            for (let xx = bx0 - bh; xx < bx0 + bwc; xx += 6) { g.beginPath(); g.moveTo(xx, by + bh); g.lineTo(xx + bh, by); g.stroke(); }
            g.restore();
          }
          const tag = f === 'FP8B' ? 'FP8 block' : f;
          const tagW = tw(g, tag, { size: 11, font: 'mono' });
          if (bwc > tagW + 10) V.text(g, tag, bx0 + 5, by + bh / 2 + 3.5, { size: 11, font: 'mono', color: C.ink });
          else if (bwc + tagW + 8 < bwMax) V.text(g, tag, bx0 + bwc + 5, by + bh / 2 + 3.5, { size: 11, font: 'mono', color: C.muted });
          if (S.ck === 'ablit' && c.k === 'att') V.text(g, '★', bx0 + bwc + (bwc + tagW + 8 < bwMax ? tagW + 12 : 6), by + bh / 2 + 4, { size: 12, color: C.d });
          V.text(g, nf(gib, gib < 10 ? 2 : 1), w - m, y + rh / 2 + 4, { size: 11, font: 'mono', align: 'right', color: C.ink });
        });
        // legend
        const items = [['BF16', C.b], ['FP8', C.a], ['FP8 block', C.d], ['NVFP4', C.c], ['MXFP8', C.f]];
        items.push(['not detailed', null]);
        let lx = m, ly = h - bot + 22;
        items.forEach(([l, c]) => {
          const iw = tw(g, l, { size: 11 }) + 30;
          if (lx + iw > w - m) { lx = m; ly += 18; }
          if (c) { V.box(g, lx, ly - 9, 11, 11, { r: 3, fill: V.alpha(c, 0.75) }); if (l === 'MXFP8') stripes(lx, ly - 9, 11, 11, 3); }
          else { V.box(g, lx, ly - 9, 11, 11, { r: 3, fill: V.alpha(C.muted, 0.2), stroke: C.muted }); g.save(); V.rr(g, lx, ly - 9, 11, 11, 3); g.clip(); g.strokeStyle = V.alpha(C.ink, 0.6); g.beginPath(); for (let d = -11; d < 11; d += 4) { g.moveTo(lx + d, ly + 2); g.lineTo(lx + d + 11, ly - 9); } g.stroke(); g.restore(); }
          V.text(g, l, lx + 15, ly, { size: 11, color: c ? C.ink : C.muted });
          lx += iw;
        });
      });
    })();

    /* =====================================================================
       FIG 7 — stepper: MTP alias and block scales
       ===================================================================== */
    (function () {
      const fig = root.querySelector('#fig-mtp-alias');
      const S = { wrong: false };
      const steps = [
        { t: 'What the checkpoint declares', d: 'The nvidia <code>config.json</code> annotates the quantization of the MTP MoE with a <b>relative</b> index: <code>mtp.layers.0.mlp.experts</code>, algorithm <code>FP8_PB_WO</code>, <code>group_size</code> 128. vLLM, however, numbers the MTP layer as a continuation of the main stack: \\( 48 + i \\).' },
        { t: 'Lookup by exact string', d: 'The ModelOpt dispatch looks for <code>"mtp.layers.48.mlp.experts"</code> in the dictionary. It does not find it and returns <code>None</code> — and the MoE is built <b>without quantization</b>, with no error at all.' },
        { t: '~7 minutes later', d: 'The target weights are already in memory when the loader reaches the drafter and looks for the FP8 scales that the “unquantized” layer does not have: <code>has no parameter \'w2_weight_scale_inv\'</code>. Crash.' },
        { t: 'The fix in the config', d: '<code>patch_checkpoint_config.py</code> adds the <code>.48.</code> alias to <code>config.json</code> and <code>hf_quant_config.json</code> (mounted over the snapshot) and rewrites <code>FP8_PB_WO</code> (group 128) as <code>FP8_BLOCK_SCALES</code>. The <code>--mtp-moe-algo</code> preflight checks it in seconds.' },
        { t: 'Scales per 128×128 block', d: 'The new branch builds <code>Fp8Config(weight_block_size=[128,128])</code>: each block has its own scale \\( s_b = \\max|W_b| / 448 \\) and \\( \\hat w = q \\cdot s_b \\). Turn on “wrong block” to see correct scales applied to the wrong regions.' },
        { t: 'Health check: acceptance by position', d: 'With the right scales, acceptance decreases smoothly — 89% / 74.5% / 60% (measured). A low, flat, nearly random curve would give away drafter weights read wrong, even without a crash.' },
      ];
      const ctl = UI.controls(fig);
      UI.toggle(ctl, { label: 'simulate wrong block (64 instead of 128)', value: false, onChange: (v) => { S.wrong = v; } });
      const st = UI.stepper(fig, steps, { autoplay: 7000 });
      // illustrative matrix: 8×8 cells (each = 32×32 weights), blocks of 4×4 cells (= 128×128)
      const rng = V.rng(43);
      const MAG = [[0.02, 0.35], [0.9, 0.08]];
      const N = 8, B = 4;
      const W = Array.from({ length: N }, (_, i) => Array.from({ length: N }, (_, j) => rng.normal() * MAG[Math.floor(i / B)][Math.floor(j / B)]));
      const sc = [[0, 0], [0, 0]];
      for (let bi = 0; bi < 2; bi++) for (let bj = 0; bj < 2; bj++) { let mx = 0; for (let i = 0; i < B; i++) for (let j = 0; j < B; j++) mx = Math.max(mx, Math.abs(W[bi * B + i][bj * B + j])); sc[bi][bj] = mx / 448; }
      const Q = W.map((r, i) => r.map((x, j) => LA.num.encode('e4m3', x / sc[Math.floor(i / B)][Math.floor(j / B)]).v));
      const flat = [sc[0][0], sc[0][1], sc[1][0], sc[1][1]];
      const good = Q.map((r, i) => r.map((q, j) => q * sc[Math.floor(i / B)][Math.floor(j / B)]));
      const bad = Q.map((r, i) => r.map((q, j) => q * flat[(Math.floor(i / 2) * 4 + Math.floor(j / 2)) % 4]));
      let amax = 0; W.forEach((r) => r.forEach((x) => { amax = Math.max(amax, Math.abs(x)); }));
      const relErr = (M) => { let s = 0; M.forEach((r, i) => r.forEach((x, j) => { s += Math.abs(x - W[i][j]) / amax; })); return (100 * s) / (N * N); };
      const errGood = relErr(good), errBad = relErr(bad);
      V.fig(fig, (g, w, h) => {
        const C = V.C();
        const i = st.i, el = st.elapsed();
        const narrow = w < 600, m = 14;
        const lh = narrow ? 16 : 20, fsz = narrow ? 11 : 12.5;
        const segLine = (x, y, parts) => { let xx = x; parts.forEach((p) => { const o = { size: fsz, font: 'mono', color: p.c || C.ink, weight: p.b ? 700 : 400 }; V.text(g, p.s, xx, y, o); xx += tw(g, p.s, o); }); };
        const card = (x, y, cw, title, rows, colr) => {
          const chh = 34 + rows.length * lh + 8;
          V.box(g, x, y, cw, chh, { r: 12, fill: V.alpha(colr, 0.08), stroke: V.alpha(colr, 0.55) });
          V.text(g, title, x + 12, y + 20, { size: 11, font: 'mono', weight: 700, color: colr });
          rows.forEach((parts, k) => segLine(x + 12, y + 40 + k * lh, parts));
          return chh;
        };
        const resultBox = (y, txt, col) => {
          const o = { size: narrow ? 11.5 : 12.5, weight: 600, color: C.ink };
          const hh = countLines(g, txt, w - 2 * m - 24, o) * 16 + 14;
          V.box(g, m, y, w - 2 * m, hh, { r: 10, fill: V.alpha(col, 0.13), stroke: V.alpha(col, 0.65) });
          wrapText(g, txt, m + 12, y + 19, w - 2 * m - 24, 16, o);
          return hh;
        };
        if (i <= 3) {
          const gapC = narrow ? 34 : 64;
          const cw = narrow ? w - 2 * m : (w - 2 * m - gapC) / 2;
          const miss = i === 1 || i === 2;
          const L = [
            [{ s: 'quantized_layers:', c: C.muted }],
            [{ s: '"mtp.layers.' }, { s: '0', c: miss ? C.red : C.d, b: true }, { s: '.mlp.experts": {' }],
            [{ s: '  "quant_algo": ' }, { s: '"FP8_PB_WO"', c: i === 3 ? C.muted : C.ink }, { s: ',' }],
            [{ s: '  "group_size": 128 }' }],
          ];
          if (i === 3) {
            L.push([{ s: '"mtp.layers.', c: C.e }, { s: '48', c: C.e, b: true }, { s: '.mlp.experts": {', c: C.e }]);
            L.push([{ s: '  "quant_algo": "FP8_BLOCK_SCALES" }', c: C.e }]);
          }
          const R = [
            [{ s: 'num_hidden_layers = 48', c: C.muted }],
            [{ s: 'MTP i = 0  →  index 48 + 0' }],
            [{ s: '"mtp.layers.' }, { s: '48', c: miss ? C.red : i === 3 ? C.e : C.b, b: true }, { s: '.mlp.experts"' }],
          ];
          const lx = m, ly = 12;
          const hL = card(lx, ly, cw, 'CHECKPOINT · config.json', L, C.d);
          const rx = narrow ? m : m + cw + gapC, ry = narrow ? ly + hL + gapC : ly;
          const hR = card(rx, ry, cw, 'vLLM · instantiated name', R, C.b);
          if (i >= 1) {
            const yL = ly + 40 + (i === 3 ? 4 : 1) * lh - 4, yR = ry + 40 + 2 * lh - 4;
            const a = narrow ? [rx + cw * 0.5, ry - 2] : [rx - 2, yR];
            const b = narrow ? [lx + cw * 0.5, ly + hL + 2] : [lx + cw + 2, yL];
            const u = Math.min(1, el / 1.1);
            const col = i === 3 ? C.e : C.red;
            V.arrow(g, a[0], a[1], V.lerp(a[0], b[0], u), V.lerp(a[1], b[1], u), { color: col, lw: 2.4, dash: [5, 4] });
            if (u >= 1) { const mx = (a[0] + b[0]) / 2 + (narrow ? 26 : 0), my = (a[1] + b[1]) / 2; V.box(g, mx - 13, my - 12, 26, 24, { r: 12, fill: V.alpha(col, 0.3), stroke: col, label: i === 3 ? '✓' : '✕', size: 13, weight: 700 }); }
          }
          const resY = Math.max(ly + hL, ry + hR) + (narrow ? 16 : 22);
          // quantization dispatch pipeline
          const pipe = narrow ? [['config', 0], ['lookup', 1], ['method', 1], [i === 3 ? 'Fp8MoE' : 'no scale', 2]] : [['config.json', 0], ['quantized_layers[name]', 1], ['get_quant_method', 1], [i === 3 ? 'Fp8MoEMethod' : 'MoE without scales', 2]];
          const py = Math.min(h - 18, resY + 92);
          if (py > resY + 70) {
            const pw = (w - 2 * m - 3 * 22) / 4;
            pipe.forEach(([lab, kind], k) => {
              const x = m + k * (pw + 22);
              const bad = (i === 1 || i === 2) && k >= 1, good = i === 3 && k >= 1;
              const col = bad ? C.red : good ? C.e : C.b;
              const lit = i >= 1 ? Math.min(1, Math.max(0, el * 1.6 - k * 0.5)) : 0;
              V.box(g, x, py - 13, pw, 26, { r: 13, fill: V.alpha(col, 0.08 + 0.22 * lit), stroke: V.alpha(col, 0.35 + 0.5 * lit) });
              V.text(g, fitText(g, lab, pw - 12, { size: 11, font: 'mono' }), x + pw / 2, py + 4, { size: 11, font: 'mono', align: 'center', color: C.ink });
              if (k < 3) V.arrow(g, x + pw + 3, py, x + pw + 19, py, { color: V.alpha(C.muted, 0.8), lw: 1.5, head: 6 });
            });
          }
          if (i === 0) resultBox(resY, 'vLLM will look up exactly the name on the right in the dictionary on the left — by string, character by character.', C.b);
          else if (i === 1) resultBox(resY, '✕ “0” ≠ “48” → get_quant_method(…) returns None → the MTP MoE is built WITHOUT quantization. No error, no warning.', C.red);
          else if (i === 3) resultBox(resY, '✓ The names match → Fp8Config(weight_block_size=[128,128]) → Fp8MoEMethod. And the --mtp-moe-algo preflight would have aborted in seconds if the algorithm could not be built.', C.e);
          else {
            const bw = w - 2 * m, byy = resY + 18;
            V.text(g, 'weight loading (target → drafter)', m, byy - 6, { size: 11, font: 'mono', color: C.muted });
            V.box(g, m, byy, bw, 12, { r: 6, fill: V.alpha(C.muted, 0.15) });
            const u = Math.min(1, el / 2.6);
            V.box(g, m, byy, Math.max(6, bw * u * 0.72), 12, { r: 6, fill: V.alpha(C.b, 0.7) });
            V.text(g, nf(u * 7, 1) + ' min', m + Math.max(6, bw * u * 0.72) + 6, byy + 10, { size: 11, font: 'mono', color: C.ink });
            if (u >= 1) {
              const ex = m + bw * 0.72, pulse = 0.6 + 0.4 * Math.sin(el * 8);
              V.glow(g, ex, byy + 6, 34 * pulse, C.red);
              wrapText(g, "✕ Layer mtp.layers.48.mlp.experts has no parameter 'w2_weight_scale_inv'", m, byy + 34, w - 2 * m, 15, { size: 11, font: 'mono', weight: 700, color: C.red });
            }
          }
          return;
        }
        if (i === 4) {
          const M = S.wrong ? bad : good;
          const gapM = narrow ? 18 : 40;
          const cell = Math.max(10, Math.min(30, (w - 2 * m - gapM - 20) / (2 * N + 0.4), (h - 130) / N));
          const mw = N * (cell + 2);
          const x0 = (w - (2 * mw + gapM)) / 2, y0 = 44;
          const lim = amax;
          // color scale in √|v| (sign preserved) + per-cell outline: small-magnitude blocks stay visible
          const cf = (v) => { const t = Math.max(-1, Math.min(1, v / lim)); return V.cmap(Math.sign(t) * Math.sqrt(Math.abs(t)), 'div'); };
          const cellStroke = V.alpha(C.muted, C.dark ? 0.5 : 0.4);
          V.text(g, 'original W', x0 + mw / 2, y0 - 12, { size: 11.5, align: 'center', font: 'mono', color: C.muted });
          V.text(g, S.wrong ? 'Ŵ with wrong block' : 'Ŵ = q · s_block', x0 + mw + gapM + mw / 2, y0 - 12, { size: 11.5, align: 'center', font: 'mono', color: S.wrong ? C.red : C.e });
          V.matrix(g, { x: x0, y: y0, data: W, cell, gap: 2, colorFn: (v) => cf(v), stroke: cellStroke });
          V.matrix(g, { x: x0 + mw + gapM, y: y0, data: M, cell, gap: 2, colorFn: (v) => cf(v), stroke: cellStroke });
          // block outlines
          const outline = (ox, bsz, col) => { for (let bi = 0; bi < N / bsz; bi++) for (let bj = 0; bj < N / bsz; bj++) { g.strokeStyle = col; g.lineWidth = 2; V.rr(g, ox + bj * bsz * (cell + 2) - 2, y0 + bi * bsz * (cell + 2) - 2, bsz * (cell + 2) + 2, bsz * (cell + 2) + 2, 5); g.stroke(); } };
          outline(x0, B, V.alpha(C.ink, 0.8));
          outline(x0 + mw + gapM, S.wrong ? 2 : B, S.wrong ? V.alpha(C.red, 0.85) : V.alpha(C.e, 0.9));
          const inCell = cell >= 22;
          if (inCell) for (let bi = 0; bi < 2; bi++) for (let bj = 0; bj < 2; bj++) V.text(g, 's=' + sc[bi][bj].toExponential(1), x0 + (bj * B + B / 2) * (cell + 2), y0 + (bi * B + B) * (cell + 2) - 6, { size: 11, font: 'mono', align: 'center', color: C.ink });
          let yy = y0 + N * (cell + 2) + 24;
          if (!inCell) {
            // on mobile the scales do not fit in the cells: list them below the matrices
            V.text(g, 'one scale per block (s = max|W_b| / 448):', m, yy, { size: 11, font: 'mono', color: C.muted });
            yy += 18;
            for (let bi = 0; bi < 2; bi++) {
              V.text(g, 'block (' + bi + ',0): s=' + sc[bi][0].toExponential(1) + '   block (' + bi + ',1): s=' + sc[bi][1].toExponential(1), m, yy, { size: 11, font: 'mono', color: C.ink, maxW: w - 2 * m });
              yy += 16;
            }
            yy += 14;
          }
          const e = S.wrong ? errBad : errGood;
          V.text(g, 'mean error |Ŵ − W| / max|W| = ' + nf(e, e < 1 ? 2 : 1) + '%', w / 2, yy, { size: narrow ? 12 : 13, weight: 700, align: 'center', color: S.wrong ? C.red : C.e, maxW: w - 2 * m });
          const leg = S.wrong ? 'illustrative: each scale lands on the wrong region — and nothing warns you' : 'each cell = 32×32 weights · each outlined block = 128×128 with one FP8 scale · color on a √ scale so the small blocks show up';
          wrapText(g, leg, w / 2, yy + 20, w - 2 * m, 15, { size: 11, align: 'center', color: C.muted });
          return;
        }
        // i === 5: acceptance
        const vals = S.wrong ? [9, 6, 4] : [89, 74.5, 60], meas = [89, 74.5, 60];
        const grow = Math.min(1, el / 1.4);
        const bw = Math.min(90, (w - 2 * m) / 5), gx = (w - 3 * bw - 2 * bw * 0.5) / 2, base = h - 60, hmax = h - 130;
        V.text(g, S.wrong ? 'acceptance by position — wrong block (illustrative)' : 'acceptance by position — measured (MTP=3, batch 1)', w / 2, 26, { size: 12, weight: 600, align: 'center', color: S.wrong ? C.red : C.ink });
        vals.forEach((v, k) => {
          const x = gx + k * bw * 1.5, bh = (v / 100) * hmax * grow;
          if (S.wrong) { const mh = (meas[k] / 100) * hmax; g.setLineDash([4, 4]); V.box(g, x, base - mh, bw, mh, { r: 6, stroke: V.alpha(C.a, 0.6), lw: 1.5 }); g.setLineDash([]); V.text(g, meas[k] + '% measured', x + bw / 2, base - mh - 6, { size: 11, align: 'center', color: C.muted }); }
          V.box(g, x, base - bh, bw, Math.max(2, bh), { r: 6, fill: V.alpha(S.wrong ? C.red : C.a, 0.7) });
          V.text(g, nf(v * grow, v % 1 ? 1 : 0) + '%', x + bw / 2, base - bh - 8, { size: 13, weight: 700, align: 'center', color: C.ink });
          V.text(g, 'position ' + (k + 1), x + bw / 2, base + 18, { size: 11, align: 'center', font: 'mono', color: C.muted });
        });
        V.line(g, m, base, w - m, base, { color: V.alpha(C.muted, 0.5) });
        V.text(g, S.wrong ? 'low and nearly flat: suspect the drafter weights (dashed = measured)' : 'decreases smoothly: each draft depends on the previous one', w / 2, h - 16, { size: 11.5, align: 'center', color: C.muted, maxW: w - 2 * m });
      });
    })();

    /* =====================================================================
       FIG 8 — kernel envelope
       ===================================================================== */
    (function () {
      const fig = root.querySelector('#fig-envelope');
      const PRE = {
        inproj: { N: 48, K: 2560, l: 'in_proj_a/b [48, 2560]' },
        visfc1: { N: 4304, K: 1152, l: 'vision fc1 [4304, 1152]' },
        qkvg: { N: 6656, K: 2560, l: 'qkv+gate TP2 [6656, 2560]' },
        visfc2: { N: 1152, K: 2152, l: 'vision fc2 w/ TP2 [1152, 2152]' },
      };
      const S = { N: 48, K: 2560, rule: 'mxfp8', pre: 'inproj', lock: false };
      const ctl = UI.controls(fig);
      const presetSeg = UI.seg(ctl, { label: 'Shape', options: Object.entries(PRE).map(([k, p]) => ({ v: k, label: p.l })), value: 'inproj', onChange: (v) => { S.pre = v; S.lock = true; sN.set(PRE[v].N); sK.set(PRE[v].K); S.lock = false; if (v === 'visfc2') ruleSeg.set('nvfp4'); else ruleSeg.set('mxfp8'); H.redraw(); } });
      const ruleSeg = UI.seg(ctl, { label: 'Rule', options: [{ v: 'mxfp8', label: 'MXFP8 mm_mxfp8' }, { v: 'nvfp4', label: 'NVFP4 (K % 16)' }], value: 'mxfp8', onChange: (v) => { S.rule = v; H.redraw(); } });
      const sN = UI.slider(ctl, { label: 'N (output)', min: 16, max: 8192, step: 8, value: 48, fmt: (v) => nf(v), onInput: (v) => { S.N = v; if (!S.lock) S.pre = null; H.redraw(); } });
      const sK = UI.slider(ctl, { label: 'K (input)', min: 16, max: 8192, step: 8, value: 2560, fmt: (v) => nf(v), onInput: (v) => { S.K = v; if (!S.lock) S.pre = null; H.redraw(); } });
      void presetSeg;
      const H = V.fig(fig, (g, w, h) => {
        const C = V.C();
        const N = S.N, K = S.K, mx = S.rule === 'mxfp8';
        const T = mx ? 32 : 16;
        const okNmin = N >= 128, okNmod = N % 32 === 0, okKmin = K >= 128, okKmod = K % (mx ? 32 : 16) === 0;
        const ok = mx ? okNmin && okNmod && okKmin && okKmod : okKmod;
        const m = 14, narrow = w < 560;
        const vcol = ok ? C.e : C.red;
        V.box(g, m, 10, w - 2 * m, 30, { r: 10, fill: V.alpha(vcol, 0.15), stroke: V.alpha(vcol, 0.7) });
        const verdict = narrow ? (mx ? (ok ? '✓ native mm_mxfp8 kernel' : '✗ outside the envelope → BF16 emulation') : (ok ? '✓ K is a multiple of 16' : '✗ K % 16 ≠ 0 → crash (use data)')) : mx ? (ok ? '✓ native FlashInfer CUTLASS mm_mxfp8 kernel' : '✗ outside the envelope → EmulationMxfp8LinearKernel (BF16)') : (ok ? '✓ K is a multiple of 16: the NVFP4 kernels accept it' : '✗ K is not a multiple of 16 → crash at load (use data)');
        V.text(g, fitText(g, verdict, w - 2 * m - 20, { size: narrow ? 11.5 : 12.5, weight: 700 }), m + 10, 30, { size: narrow ? 11.5 : 12.5, weight: 700, color: C.ink });
        const chk = (okv, s) => (okv ? '✓ ' : '✗ ') + s;
        const items = mx ? [chk(okNmin, 'N ' + (okNmin ? '≥' : '<') + ' 128'), chk(okNmod, 'N mod 32 = ' + (N % 32)), chk(okKmin, 'K ' + (okKmin ? '≥' : '<') + ' 128'), chk(okKmod, 'K mod 32 = ' + (K % 32))] : [chk(okKmod, 'K mod 16 = ' + (K % 16)), '· N free'];
        let ix = m, iy = 60;
        items.forEach((s) => { const good = s[0] === '✓'; const o = { size: 11.5, font: 'mono', weight: 600, color: s[0] === '✗' ? C.red : good ? C.e : C.muted }; const ww = tw(g, s, o) + 16; if (ix + ww > w - m) { ix = m; iy += 18; } V.text(g, s, ix, iy, o); ix += ww; });
        // tile decomposition
        const dec = (name, v, tile, need) => { const q = Math.floor(v / tile), r = v % tile; return { s: name + ' = ' + nf(v) + ' = ' + q + '·' + tile + ' + ' + r, ok: r === 0 && v >= need }; };
        const decs = mx ? [dec('N', N, 32, 128), dec('K', K, 32, 128)] : [dec('K', K, 16, 0)];
        iy += 24;
        decs.forEach((d) => { V.text(g, d.s + (d.ok ? '  ✓' : '  ✗'), m, iy, { size: narrow ? 12 : 13.5, font: 'mono', weight: 700, color: d.ok ? C.e : C.red, maxW: w - 2 * m }); iy += narrow ? 19 : 21; });
        // draws an [n × k] window of the matrix with the tile grid and the remainders
        const drawMat = (bx, by, bw, bh, n, k, fullN, fullK, labK, labN, kTop) => {
          V.box(g, bx, by, bw, bh, { r: 3, fill: V.alpha(C.b, 0.14), stroke: V.alpha(C.b, 0.7) });
          const sxK = bw / k, syN = bh / n;
          const nK = Math.floor(k / T), remK = fullK % T;
          const stepK = Math.max(1, Math.ceil(nK / 90));
          g.strokeStyle = V.alpha(C.b, 0.45); g.lineWidth = 1;
          g.beginPath();
          for (let q = stepK; q <= nK; q += stepK) { const x = bx + q * T * sxK; if (x < bx + bw - 0.5) { g.moveTo(x, by); g.lineTo(x, by + bh); } }
          if (mx) { const nN = Math.floor(n / T), stepN = Math.max(1, Math.ceil(nN / 90)); for (let q = stepN; q <= nN; q += stepN) { const y = by + q * T * syN; if (y < by + bh - 0.5) { g.moveTo(bx, y); g.lineTo(bx + bw, y); } } }
          g.stroke();
          // remainders: only shown if the window reaches the end of the dimension
          if (remK && k === fullK) { const x = bx + nK * T * sxK; g.fillStyle = V.alpha(C.red, 0.45); g.fillRect(x, by, Math.max(3, bx + bw - x), bh); }
          if (mx && fullN % 32 && n === fullN) { const y = by + Math.floor(n / T) * T * syN; g.fillStyle = V.alpha(C.red, 0.45); g.fillRect(bx, y, bw, Math.max(3, by + bh - y)); }
          if (mx && (!okNmin || !okKmin)) { g.strokeStyle = C.red; g.lineWidth = 2.5; g.setLineDash([6, 4]); V.rr(g, bx - 3, by - 3, bw + 6, bh + 6, 5); g.stroke(); g.setLineDash([]); }
          V.text(g, labK, bx + bw / 2, kTop ? by - 8 : by + bh + 16, { size: 11, font: 'mono', align: 'center', color: C.ink, maxW: w - 2 * m });
          if (labN) { g.save(); g.translate(bx - 10, by + bh / 2); g.rotate(-Math.PI / 2); V.text(g, labN, 0, 0, { size: 11, font: 'mono', align: 'center', color: C.ink }); g.restore(); }
          return stepK;
        };
        const ax = m + 34, ay = iy + 20, aw = w - ax - m - 4, ahAll = h - ay - 34;
        let ratio = N / K; const clamped = ratio < 0.06 || ratio > 12;
        ratio = Math.max(0.06, Math.min(12, ratio));
        const ah = clamped ? Math.max(40, ahAll * 0.36) : ahAll;
        let bw = aw, bh = aw * ratio;
        if (bh > ah) { bh = ah; bw = ah / ratio; }
        bh = Math.max(bh, 18); bw = Math.max(bw, 18);
        const bx = ax + (aw - bw) / 2, by = ay + (ah - bh) / 2;
        const remK = K % T;
        const labNfull = 'N = ' + nf(N) + (mx && N % 32 ? ' · remainder ' + (N % 32) : '');
        const labKfull = 'K (input) = ' + nf(K) + (remK ? '  · remainder ' + remK : '');
        // flat strip: the N label goes with the K label, on top (a side label would not fit)
        const stepK = clamped ? drawMat(bx, by, bw, bh, N, K, N, K, labKfull + ' · ' + labNfull, '', true) : drawMat(bx, by, bw, bh, N, K, N, K, labKfull, labNfull, false);
        if (clamped) {
          // corner zoom at the real aspect ratio: 256 on the long dimension, the short one whole (up to 256)
          const zN = Math.min(N, narrow ? 128 : 256), zK = Math.min(K, narrow ? 128 : 256);
          const zy = by + bh + 26, zah = h - zy - 40;
          const zr = zN / zK;
          let zw = aw, zh = aw * zr;
          if (zh > zah) { zh = zah; zw = zah / zr; }
          const zx = ax + (aw - zw) / 2;
          // zoom rectangle on the main strip and connector
          const rw = Math.max(4, (zK / K) * bw), rh = Math.max(4, (zN / N) * bh);
          g.strokeStyle = C.d; g.lineWidth = 1.5; g.strokeRect(bx, by, rw, rh);
          V.line(g, bx + rw, by + rh, zx + zw, zy - 4, { color: V.alpha(C.d, 0.7), lw: 1, dash: [3, 3] });
          V.line(g, bx, by + rh, zx, zy - 4, { color: V.alpha(C.d, 0.7), lw: 1, dash: [3, 3] });
          drawMat(zx, zy, zw, zh, zN, zK, N, K, 'zoom: K 0–' + nf(zK) + (zK < K ? ' of ' + nf(K) : '') + ' · real aspect ratio', 'N 0–' + nf(zN));
        }
        const note = (clamped ? 'aspect ratio stretched to fit · ' : '') + (stepK > 1 ? 'each line = ' + stepK + ' tiles · ' : '') + 'tile = ' + T + (mx ? '×' + T : ' in K');
        V.text(g, note, w - m, h - 8, { size: 11, align: 'right', color: C.muted, maxW: w - 2 * m });
      }, { static: true });
    })();

    /* =====================================================================
       FIG 9 — conveyor belt that builds the command line
       ===================================================================== */
    (function () {
      const fig = root.querySelector('#fig-esteira');
      const S = { mtp: 3, vocab: true, kv: 'fp8', yarn: false, ep: true, extra: false };
      let prev = [];
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'MTP', options: [0, 1, 3], value: 3, onChange: (v) => { S.mtp = v; render(); } });
      UI.toggle(ctl, { label: 'MTP_DRAFT_VOCAB (47k)', value: true, onChange: (v) => { S.vocab = v; render(); } });
      UI.seg(ctl, { label: 'KV', options: ['fp8', 'auto'], value: 'fp8', onChange: (v) => { S.kv = v; render(); } });
      UI.toggle(ctl, { label: 'YaRN (1M)', value: false, onChange: (v) => { S.yarn = v; render(); } });
      UI.toggle(ctl, { label: 'expert parallel', value: true, onChange: (v) => { S.ep = v; render(); } });
      UI.toggle(ctl, { label: 'EXTRA_VLLM_ARGS', value: false, onChange: (v) => { S.extra = v; render(); } });
      const rArgs = UI.readout(ctl, 'flags', '—');
      const rOv = UI.readout(ctl, 'overlays', '—');
      const out = LA.h('div', { style: { padding: '0 16px 12px' } });
      beforeCap(fig, out);
      function build() {
        const a = [];
        a.push('--enable-prompt-tokens-details', '--served-model-name qwen3.8-flash-next', '--tensor-parallel-size 2', '--gpu-memory-utilization 0.80', '--max-num-seqs 8', '--max-num-batched-tokens 8192', '--max-model-len ' + (S.yarn ? '1000000' : '262144'), '--kv-cache-dtype ' + S.kv, '--mamba-ssm-cache-dtype bfloat16', '--load-format safetensors', '--safetensors-load-strategy lazy', '--enable-chunked-prefill', '--reasoning-parser qwen3', '--enable-auto-tool-choice', '--tool-call-parser qwen3_coder', '--distributed-executor-backend mp', '--mm-encoder-tp-mode data', '--nnodes 2', '--master-addr $HEAD_IP', '--master-port 50000');
        if (S.ep) a.push('--enable-expert-parallel', '--all2all-backend allgather_reducescatter');
        if (S.mtp > 0) a.push(`--speculative-config '{"method":"mtp","num_speculative_tokens":${S.mtp}${S.vocab ? ',"use_local_argmax_reduction":true' : ''},"disable_eagle_block_drop":true,"index_share_for_mtp_iteration":true}'`);
        a.push(`--compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}'`);
        a.push(S.yarn ? `--hf-overrides '{"text_config":{"ple_embedding_dtype":"float8_e4m3fn","rope_parameters":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":262144}}}'` : `--hf-overrides '{"text_config":{"ple_embedding_dtype":"float8_e4m3fn"}}'`);
        if (S.extra) a.push('--kv-cache-memory=41360080896');
        return a;
      }
      function overlays() {
        const o = ['ple_layer_patched.py', 'modelopt_patched.py', 'config_patched.json', 'hf_quant_config_patched.json'];
        if (S.kv === 'fp8') o.push('qsa_ops_patched.py', 'qsa_nvidia_patched.py');
        if (S.vocab && S.mtp > 0) o.push('mtp_patched.py', 'draft_vocab_en_code_47k.txt');
        if (S.mtp > 0) o.push('speculative.py', 'kv_cache_utils.py', 'scheduler.py');
        return o;
      }
      function render() {
        const err = S.vocab && S.mtp === 0;
        const a = build();
        const ov = overlays();
        rArgs.set(String(a.length)); rOv.set(String(ov.length));
        const lines = a.map((s) => {
          const changed = prev.length && !prev.includes(s);
          const t = '  ' + esc(s) + ' \\';
          return changed ? `<span style="background:color-mix(in srgb, var(--mint) 24%, transparent);border-radius:4px">${t}</span>` : t;
        });
        prev = a;
        const warn = err ? '<div class="la-callout danger" style="margin:.6em 0"><b class="t">start.sh aborts</b><p><code>MTP_DRAFT_VOCAB is set but MTP_NUM_SPECULATIVE_TOKENS=0 - nothing drafts.</code></p></div>' : '';
        out.innerHTML = warn + `<pre style="max-height:330px;overflow:auto;font-size:12px;margin:.6em 0 .4em"><code>vllm/vllm-openai:qwen38-flash-next nvidia/Qwen3.8-Flash-Next-NVFP4 \\\n${lines.join('\n')}\n  # head:   --node-rank 0 --host 0.0.0.0 --port 8888\n  # worker: --node-rank 1 --headless</code></pre>` +
          `<p style="font-size:13px;color:var(--text-3);margin:.2em 0">overlays mounted <code>:ro</code> on both nodes: ${ov.map((x) => '<code>' + esc(x) + '</code>').join(' ')}</p>`;
      }
      render();
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const narrow = w < 600, m = 12;
        const knobs = [
          { l: 'MTP=' + S.mtp, on: S.mtp > 0 },
          { l: S.vocab ? 'vocab 47k' : 'full vocab', on: S.vocab },
          { l: 'KV ' + S.kv, on: S.kv === 'fp8' },
          { l: S.yarn ? 'YaRN 1M' : 'YaRN off', on: S.yarn },
          { l: S.ep ? 'EP on' : 'EP off', on: S.ep },
          { l: 'EXTRA', on: S.extra },
        ];
        // geometry: on desktop everything on one row; on mobile the belt goes on top and the ranks below
        const mw = narrow ? 92 : 124, mh = narrow ? 84 : 112;
        const beltY = narrow ? 70 : h * 0.5;
        const mx = narrow ? w - m - mw / 2 : w * 0.46;
        const envW = narrow ? 46 : 54, envX = m;
        const bx0 = envX + envW - 6, bx1 = mx - mw / 2;
        // belt
        V.box(g, bx0, beltY + 16, bx1 - bx0, 10, { r: 5, fill: V.alpha(C.muted, 0.25) });
        for (let x = bx0 + 6 + ((t * 30) % 14); x < bx1; x += 14) V.line(g, x, beltY + 17, x - 5, beltY + 25, { color: V.alpha(C.muted, 0.5), lw: 1.2 });
        const kw = narrow ? 70 : 78, kh = 24, pitch = kw + 12, span = Math.max(bx1 - bx0 + kw, knobs.length * pitch);
        g.save(); g.beginPath(); g.rect(bx0, beltY - 30, bx1 - bx0, 60); g.clip();
        knobs.forEach((k, i) => {
          const x = bx0 - kw + (i * pitch + t * 26) % span;
          if (x + kw > bx1 + 4) return;
          V.box(g, x, beltY - 9, kw, kh, { r: 6, fill: k.on ? V.alpha(C.qwen, 0.3) : V.alpha(C.muted, 0.15), stroke: k.on ? C.qwen : V.alpha(C.muted, 0.5), label: k.l, size: 11 });
        });
        g.restore();
        // the .env file the keys come from
        V.box(g, envX, beltY - 26, envW, 52, { r: 8, fill: V.alpha(C.d, 0.16), stroke: C.d, lw: 1.5 });
        g.fillStyle = V.alpha(C.d, 0.55); g.beginPath(); g.moveTo(envX + envW - 12, beltY - 26); g.lineTo(envX + envW, beltY - 14); g.lineTo(envX + envW - 12, beltY - 14); g.closePath(); g.fill();
        V.text(g, '.env', envX + envW / 2, beltY + 4, { size: 12, font: 'mono', weight: 700, align: 'center', color: C.ink });
        // machine
        V.box(g, mx - mw / 2, beltY - mh / 2, mw, mh, { r: 14, fill: V.alpha(C.b, 0.14), stroke: C.b, lw: 2 });
        V.text(g, 'start.sh', mx, beltY - mh / 2 + 20, { size: 12, font: 'mono', weight: 700, align: 'center', color: C.ink });
        V.text(g, 'VLLM_ARGS', mx, beltY + mh / 2 - 11, { size: 11, font: 'mono', align: 'center', color: C.muted });
        const gear = (cx, cy, r, a, col) => { g.save(); g.translate(cx, cy); g.rotate(a); g.strokeStyle = col; g.lineWidth = 2.2; g.beginPath(); g.arc(0, 0, r, 0, Math.PI * 2); g.stroke(); for (let k = 0; k < 8; k++) { g.rotate(Math.PI / 4); g.beginPath(); g.moveTo(r, 0); g.lineTo(r + 4, 0); g.stroke(); } g.restore(); };
        gear(mx - 14, beltY + 2, narrow ? 10 : 13, t * 1.6, C.b); gear(mx + 14, beltY + 2, narrow ? 7 : 9, -t * 2.3, C.a);
        // output: one string that splits into the two ranks
        const bw2 = narrow ? Math.min(160, (w - 2 * m - 24) / 2) : 146, bh2 = 46;
        let P0, Psplit, targets;
        if (narrow) {
          const yB = h - 34, splitY = beltY + mh / 2 + 44;
          P0 = [mx, beltY + mh / 2]; Psplit = [mx, splitY];
          const cW = m + bw2 / 2, cH = w - m - bw2 / 2;
          targets = [{ c: [cW, yB], x: m, y: yB - bh2 / 2, t: 'worker · rank 1', s: '--headless' }, { c: [cH, yB], x: w - m - bw2, y: yB - bh2 / 2, t: 'head · rank 0', s: ':8888 (API)' }];
        } else {
          const ox = mx + mw / 2, tx = w - m - bw2, yW = beltY - 64, yH = beltY + 64;
          P0 = [ox, beltY]; Psplit = [ox + (tx - ox) * 0.35, beltY];
          targets = [{ c: [tx, yW], x: tx, y: yW - bh2 / 2, t: 'worker · rank 1', s: '--headless' }, { c: [tx, yH], x: tx, y: yH - bh2 / 2, t: 'head · rank 0', s: ':8888 (API)' }];
        }
        const curve = (T0, v) => narrow
          ? V.bez(Psplit, [Psplit[0], Psplit[1] + 30], [T0.c[0], T0.y - 40], [T0.c[0], T0.y], v)
          : V.bez(Psplit, [Psplit[0] + 30, Psplit[1]], [Psplit[0] + 10, T0.c[1]], [T0.c[0], T0.c[1]], v);
        V.line(g, P0[0], P0[1], Psplit[0], Psplit[1], { color: C.qwen, lw: 3 });
        targets.forEach((T0) => {
          g.beginPath(); const a = curve(T0, 0); g.moveTo(a[0], a[1]);
          for (let q = 1; q <= 24; q++) { const p = curve(T0, q / 24); g.lineTo(p[0], p[1]); }
          g.strokeStyle = C.qwen; g.lineWidth = 3; g.stroke();
        });
        for (let k = 0; k < 4; k++) {
          const u = ((t * 0.35 + k / 4) % 1);
          if (u < 0.35) { const f = u / 0.35; V.box(g, V.lerp(P0[0], Psplit[0], f) - 8, V.lerp(P0[1], Psplit[1], f) - 4, 16, 8, { r: 3, fill: C.qwen }); }
          else { const v = (u - 0.35) / 0.65; targets.forEach((T0) => { const p = curve(T0, v); V.box(g, p[0] - 8, p[1] - 4, 16, 8, { r: 3, fill: C.qwen }); }); }
        }
        targets.forEach((T0) => {
          V.box(g, T0.x, T0.y, bw2, bh2, { r: 10, fill: V.alpha(C.qwen, 0.12), stroke: C.qwen });
          V.text(g, T0.t, T0.x + bw2 / 2, T0.y + 19, { size: narrow ? 11.5 : 12, weight: 700, align: 'center', color: C.ink });
          V.text(g, T0.s, T0.x + bw2 / 2, T0.y + 36, { size: 11, font: 'mono', align: 'center', color: C.muted });
        });
        if (narrow) {
          V.text(g, 'same string', mx - 14, P0[1] + 18, { size: 11, align: 'right', color: C.muted });
          V.text(g, 'on both ranks', mx - 14, P0[1] + 33, { size: 11, align: 'right', color: C.muted });
        } else {
          V.text(g, 'same string', targets[0].x + bw2 / 2, beltY - 3, { size: 11, align: 'center', color: C.muted });
          V.text(g, 'on both ranks', targets[0].x + bw2 / 2, beltY + 12, { size: 11, align: 'center', color: C.muted });
        }
        if (S.vocab && S.mtp === 0) { V.glow(g, mx, beltY, 60 * (0.7 + 0.3 * Math.sin(t * 6)), C.red, 0.7); V.text(g, 'aborts!', mx, beltY - mh / 2 - 8, { size: 12, weight: 700, align: 'center', color: C.red }); }
      });
    })();

    /* =====================================================================
       FIG 10 — boot timeline
       ===================================================================== */
    (function () {
      const fig = root.querySelector('#fig-boot');
      const PH = [
        { l: 'rest (not detailed)', sh: 'rest', s: 56.6, k: 'rest' },
        { l: 'NCCL', sh: 'NCCL', s: 40, k: 'nccl' },
        { l: 'target weights', sh: 'target', s: 392, k: 'wt' },
        { l: 'MTP drafter', sh: 'MTP', s: 67, k: 'mtp' },
        { l: 'engine init', sh: 'init', s: 92.4, k: 'eng' },
        { l: 'graphs', sh: 'g', s: 7, k: 'gr' },
      ];
      const TOTAL = PH.reduce((a, p) => a + p.s, 0); // 655
      const SC = {
        ok: { l: 'measured boot', fail: null },
        shard: { l: 'corrupted shard', fail: 56.6 + 40 + 0.33 * 392, msg: 'safe_open: incomplete metadata, file not fully covered (~33% into the load)', pre: 'check-weights.sh --verify: hash mismatch (≈1 min before the launch)' },
        mtp: { l: 'MTP without FP8 scale', fail: 56.6 + 40 + 392 + 12, msg: "mtp.layers.48.mlp.experts has no parameter 'w2_weight_scale_inv' (~7 min of loading, when it reaches the drafter)", pre: '--mtp-moe-algo: algorithm checked in seconds (exit 3 if unsupported)' },
        mxfp8: { l: 'MXFP8 without fallback (local-inference-lab ckpt)', fail: 56.6 + 40 + 458 + 5, msg: 'AssertionError … N >= 128, got N=48 — in the profile run, weights already resident. Only on checkpoints with MXFP8 layers (e.g. local-inference-lab), not on nvidia. README: ~7 min into the launch; here the position follows the nvidia timeline', pre: null },
        gpu: { l: 'GPU with a tenant', fail: 600, msg: 'CUDA out of memory (~10 min later)', pre: 'REQUIRE_IDLE_GPU: nvidia-smi shows a process → aborts in seconds' },
      };
      const S = { sc: 'ok', pre: false, t0: null };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Scenario', options: Object.entries(SC).map(([k, o]) => ({ v: k, label: o.l })), value: 'ok', onChange: (v) => { S.sc = v; S.t0 = null; } });
      UI.toggle(ctl, { label: 'preflights on', value: false, onChange: (v) => { S.pre = v; S.t0 = null; } });
      UI.button(ctl, '▶ replay', () => { S.t0 = null; });
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        if (S.t0 == null) S.t0 = t;
        const narrow = w < 600, m = 14;
        const sc = SC[S.sc];
        const hostFrac = 0.16;
        const x0 = m, x1 = w - m, span = x1 - x0;
        const hostW = span * hostFrac, cx0 = x0 + hostW, cspan = span - hostW;
        const by = 70, bh = 34;
        const colors = [V.alpha(C.muted, 0.45), C.b, C.qwen, C.f, C.d, C.e];
        const caught = S.pre && sc.pre;
        const failT = caught ? null : sc.fail;
        const real = t - S.t0;
        const speed = TOTAL / 9; // 655 s in ~9 s
        const hostDur = 1.4;
        let simT = (real - hostDur) * speed;
        const stopAt = caught ? -1 : failT;
        const inHost = real < hostDur || caught;
        if (stopAt != null && stopAt >= 0 && simT > stopAt) simT = stopAt;
        simT = Math.min(simT, TOTAL);
        const finished = (stopAt == null && simT >= TOTAL) || (stopAt != null && stopAt >= 0 && simT >= stopAt) || (caught && real > hostDur * 0.8);
        if (finished && real > hostDur + 9.5 + 3.5) S.t0 = t;
        // title
        V.text(g, 'HOST', x0, by - 12, { size: 11, font: 'mono', color: C.muted });
        V.text(g, 'CONTAINER · ≈10m55s to /health', cx0 + 6, by - 12, { size: 11, font: 'mono', color: C.muted, maxW: cspan - 10 });
        // host phase
        V.box(g, x0, by, hostW - 4, bh, { r: 8, fill: V.alpha(C.a, 0.18), stroke: V.alpha(C.a, 0.6) });
        V.text(g, narrow ? 'host' : 'prep (variable)', x0 + (hostW - 4) / 2, by + bh / 2 + 4, { size: 11, align: 'center', color: C.ink, maxW: hostW - 10 });
        // container phases
        let x = cx0;
        PH.forEach((p, i) => {
          const ww = (p.s / TOTAL) * cspan;
          V.box(g, x, by, Math.max(1, ww - 1.5), bh, { r: 4, fill: V.alpha(colors[i], i === 0 ? 1 : 0.5) });
          const lab = !narrow && ww > tw(g, p.l, { size: 11 }) + 8 ? p.l : ww > tw(g, p.sh, { size: 11 }) + 6 ? p.sh : '';
          if (lab) V.text(g, lab, x + ww / 2, by + bh / 2 + 4, { size: 11, align: 'center', color: C.ink });
          x += ww;
        });
        // minutes axis
        for (let mnt = 0; mnt <= 11; mnt++) {
          const xx = cx0 + ((mnt * 60) / TOTAL) * cspan;
          if (xx > x1) break;
          V.line(g, xx, by + bh + 2, xx, by + bh + 7, { color: C.muted, lw: 1 });
          if (!narrow || mnt % 2 === 0) V.text(g, mnt + 'm', xx, by + bh + 19, { size: 11, font: 'mono', align: 'center', color: C.muted });
        }
        // failure point flag
        if (sc.fail != null && !caught) {
          const fx = cx0 + (sc.fail / TOTAL) * cspan;
          V.line(g, fx, by - 20, fx, by + bh + 4, { color: C.red, lw: 1.5, dash: [3, 3] });
          V.box(g, fx, by - 24, 14, 9, { r: 2, fill: C.red });
        }
        // cursor
        let cxr;
        if (inHost) cxr = x0 + Math.min(1, real / hostDur) * (hostW - 6);
        else cxr = cx0 + (Math.max(0, simT) / TOTAL) * cspan;
        V.line(g, cxr, by - 6, cxr, by + bh + 6, { color: C.ink, lw: 2 });
        V.dot(g, cxr, by - 6, 4, C.ink);
        // legend (mobile)
        let ly = by + bh + 44;
        if (narrow) {
          const cols = 2, lw = (w - 2 * m) / cols;
          PH.forEach((p, i) => { const cx = m + (i % cols) * lw, cy = ly + Math.floor(i / cols) * 17; V.box(g, cx, cy - 9, 10, 10, { r: 2, fill: V.alpha(colors[i], i === 0 ? 1 : 0.7) }); V.text(g, (i === 0 ? 'rest' : p.l) + ' ' + nf(p.s, p.s % 1 ? 1 : 0) + ' s', cx + 14, cy, { size: 11, color: C.ink, maxW: lw - 18 }); });
          ly += 3 * 17 + 8;
        } else {
          V.text(g, 'NCCL 40 s · target 392 s · drafter 67 s · engine init 92.4 s · graphs ~7 s · rest 56.6 s', m, ly, { size: 11, font: 'mono', color: C.muted, maxW: w - 2 * m });
          ly += 22;
        }
        // status message
        let msg, col;
        if (caught) { msg = 'Preflight: ' + sc.pre; col = C.d; }
        else if (stopAt != null && simT >= stopAt) { msg = '✕ ' + sc.msg; col = C.red; }
        else if (stopAt == null && simT >= TOTAL) { msg = '✓ /health = 200 in ≈10m55s — the 1st request still pays for the FlashInfer autotune'; col = C.e; }
        else if (inHost) { msg = S.pre ? 'Running preflights on the host: REQUIRE_IDLE_GPU, --mtp-moe-algo, check-weights --verify…' : 'Preparing on the host (download, sync, overlays)…'; col = C.a; }
        else { const cur = (() => { let acc = 0; for (const p of PH) { acc += p.s; if (simT <= acc) return p.l; } return 'graphs'; })(); msg = 't = ' + nf(simT / 60, 1) + ' min · ' + cur; col = C.b; }
        if (sc.fail != null && S.pre && !sc.pre && S.sc === 'mxfp8') msg = (simT >= stopAt ? '✕ ' + sc.msg : msg) + ' — no preflight: the fix is the fallback, always mounted';
        if (caught || (stopAt != null && simT >= stopAt)) {
          const fx = caught ? x0 + hostW * 0.7 : cxr;
          V.glow(g, fx, by + bh / 2, 30 * (0.7 + 0.3 * Math.sin(real * 7)), caught ? C.d : C.red);
        }
        const nl = countLines(g, msg, w - 2 * m - 24, { size: 12 });
        V.box(g, m, ly, w - 2 * m, nl * 16 + 14, { r: 10, fill: V.alpha(col, 0.12), stroke: V.alpha(col, 0.6) });
        wrapText(g, msg, m + 12, ly + 19, w - 2 * m - 24, 16, { size: 12, color: C.ink });
      });
    })();

    /* =====================================================================
       FIG 2 — bytes per decode step (levers)
       ===================================================================== */
    (function () {
      const fig = root.querySelector('#fig-bytes-passo');
      const BASE = [
        { k: 'lm', l: 'lm_head ×4 (target + 3 drafts)', ls: 'lm_head ×4', v: 2.36 },
        { k: 'exp', l: 'routed experts (~32 unique/layer)', ls: 'routed experts', v: 2.1 },
        { k: 'gdn', l: 'GDN projections', v: 1.95 },
        { k: 'hc', l: 'HyperConnections (replicated)', ls: 'HyperConnections', v: 1.19 },
        { k: 'att', l: 'attention + indexer', v: 0.60 },
        { k: 'mtp', l: 'dense MTP + experts ×3', v: 0.5 },
        { k: 'rs', l: 'router + shared + PLE proj.', ls: 'router + shared', v: 0.40 },
        { k: 'st', l: 'GDN state (read + write)', ls: 'GDN state (R+W)', v: 0.1 },
      ];
      const S = { vocab: false, dense: false, bw: 240, a: 2.9, ov: 17, anim: {} };
      const ctl = UI.controls(fig);
      const tV = UI.toggle(ctl, { label: '47k draft vocabulary', value: false, onChange: (v) => { S.vocab = v; if (v && S.dense) tD.set(false); } });
      const tD = UI.toggle(ctl, { label: 'FP8-dense (analytical)', value: false, onChange: (v) => { S.dense = v; if (v && S.vocab) tV.set(false); } });
      UI.slider(ctl, { label: 'bandwidth', min: 200, max: 273, step: 1, value: 240, fmt: (v) => v + ' GB/s', onInput: (v) => { S.bw = v; } });
      UI.slider(ctl, { label: 'tokens/step', min: 1, max: 4, step: 0.05, value: 2.9, fmt: (v) => v.toFixed(2), onInput: (v) => { S.a = v; } });
      UI.slider(ctl, { label: 'overhead', min: 0, max: 25, step: 0.5, value: 17, fmt: (v) => nf(v, 1) + ' ms', onInput: (v) => { S.ov = v; } });
      const rB = UI.readout(ctl, 'GiB/step', '—');
      const rT = UI.readout(ctl, 'ms/step', '—');
      const rS = UI.readout(ctl, 'tok/s (≈)', '—');
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const narrow = w < 600, m = 14;
        const comps = BASE.map((c) => ({ ...c, v: c.k === 'lm' && S.vocab ? 0.59 + 3 * 0.22 : c.v }));
        const sum = comps.reduce((a, c) => a + c.v, 0);
        const total = S.dense ? 5.7 : sum;
        const readMs = (total * 1.073741824) / S.bw * 1000;
        const stepMs = readMs + S.ov;
        const tps = S.a / (stepMs / 1000);
        rB.set(nf(total, 2)); rT.set(nf(stepMs, 1)); rS.set(nf(tps, 1));
        const lw = narrow ? 150 : 250, vw = 54;
        const bx0 = m + lw, bwMax = w - m - vw - bx0;
        const top = 34, rows = comps.length, bottom = 112;
        const rh = (h - top - bottom) / rows;
        const maxV = 2.4;
        V.text(g, S.dense ? 'FP8-dense: the report only estimates the total (≈5.7 GiB) — bars in gray' : 'bytes read per GPU in one step (batch 1, MTP=3)', m, 20, { size: 11.5, weight: 600, color: S.dense ? C.d : C.ink, maxW: w - 2 * m });
        // read head
        const cyc = (t * 1000 / Math.max(20, stepMs)) * 0.035; // heavily "slowed-down" steps so it stays visible
        const ph = cyc % 1;
        const cumul = []; let acc = 0; comps.forEach((c) => { cumul.push(acc); acc += c.v; });
        const pos = ph * acc;
        comps.forEach((c, i) => {
          const y = top + i * rh;
          const target = (c.v / maxV) * bwMax;
          S.anim[c.k] = S.anim[c.k] == null ? target : V.lerp(S.anim[c.k], target, 0.15);
          const bw = S.anim[c.k];
          const colr = c.k === 'lm' ? C.d : c.k === 'exp' ? C.qwen : c.k === 'gdn' ? C.b : c.k === 'hc' ? C.f : C.muted; // clearly distinct colors in both themes
          V.text(g, fitText(g, narrow && c.ls ? c.ls : c.l, lw - 8, { size: 11.5 }), m, y + rh / 2 + 4, { size: 11.5, color: C.ink });
          const bh = Math.min(18, rh - 6), by = y + (rh - bh) / 2;
          V.box(g, bx0, by, Math.max(2, bw), bh, { r: 4, fill: S.dense ? V.alpha(C.muted, 0.3) : V.alpha(colr, 0.65) });
          // read in progress
          if (!S.dense && pos > cumul[i]) { const fr = Math.min(1, (pos - cumul[i]) / c.v); V.box(g, bx0, by, Math.max(2, bw * fr), bh, { r: 4, fill: V.alpha(colr, 0.95) }); if (fr < 1) V.glow(g, bx0 + bw * fr, by + bh / 2, 12, colr); }
          V.text(g, nf(c.v, 2), w - m, y + rh / 2 + 4, { size: 11, font: 'mono', align: 'right', color: C.ink });
        });
        // total bar and tokens
        const ty = h - bottom + 22;
        const totW = w - 2 * m;
        V.text(g, 'total ' + nf(total, 2) + ' GiB · read ' + nf(readMs, 1) + ' ms + overhead ' + nf(S.ov, 1) + ' ms', m, ty, { size: 11.5, font: 'mono', color: C.ink, maxW: totW });
        V.box(g, m, ty + 8, totW, 12, { r: 6, fill: V.alpha(C.muted, 0.15) });
        V.box(g, m, ty + 8, totW * Math.min(1, readMs / 70), 12, { r: 6, fill: V.alpha(C.qwen, 0.7) });
        V.box(g, m + totW * Math.min(1, readMs / 70), ty + 8, totW * Math.min(1, S.ov / 70), 12, { r: 0, fill: V.alpha(C.muted, 0.45) });
        V.text(g, '0', m, ty + 34, { size: 11, font: 'mono', color: C.muted }); V.text(g, '70 ms', w - m, ty + 34, { size: 11, font: 'mono', align: 'right', color: C.muted });
        // token chips
        const n = Math.floor(S.a), fr = S.a - n;
        const tx = w / 2 - ((n + (fr > 0.05 ? 1 : 0)) * 20) / 2;
        const pop = ph > 0.8 ? V.ease.back(Math.min(1, (ph - 0.8) / 0.2)) : 0;
        for (let k = 0; k < n + (fr > 0.05 ? 1 : 0); k++) {
          const s = k < n ? 1 : fr;
          V.box(g, tx + k * 20, ty + 44 - 7 * pop, 16 * s, 14, { r: 5, fill: V.alpha(C.e, 0.35 + 0.55 * pop) });
        }
        V.text(g, nf(S.a, 2) + ' tokens/step → ≈' + nf(tps, 1) + ' tok/s', w / 2, ty + 70, { size: 12, weight: 700, align: 'center', color: C.ink });
      });
    })();
  },
});

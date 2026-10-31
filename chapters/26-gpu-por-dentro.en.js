LA.chapter({
  id: 'gpu-por-dentro',
  part: 5,
  order: 26,
  title: 'Inside the GPU: SMs, tensor cores and bandwidth',
  navTitle: 'Inside the GPU',
  subtitle: 'SMs, warps, registers and shared memory; tensor cores and formats by generation; compute capability, PTX and cubins for the GB10’s sm_121a; and the roofline that turns bytes read per step into tokens per second.',
  minutes: 60,
  recipes: ['glm', 'tf', 'qwen', 'ds'],
  tags: ['GPU', 'SM', 'warp', 'shared memory', 'tensor core', 'mma.sync', 'compute capability', 'sm_121a', 'PTX', 'cubin', 'roofline', 'LPDDR5X', 'streams'],
  goals: [
    'Describe the GPU from top to bottom (SMs, sub-partitions, warps, lanes, registers, shared memory, L2, DRAM) and compute a kernel’s <b>occupancy</b>, recognizing spills',
    'Explain what a tensor core instruction computes (<code>mma.sync m16n8k16</code>, block-scaled MMA) and why the GB10 uses <code>mma.sync</code> rather than <code>tcgen05</code>',
    'Read <b>compute capability</b>, the <code>a</code>/<code>f</code> suffixes, PTX, cubin and fatbin, and justify <code>TORCH_CUDA_ARCH_LIST=12.1a</code>, <code>CUTE_DSL_ARCH=sm_121a</code> and the recipes’ gates',
    'Build the <b>step cost model</b> (bytes read ÷ bandwidth + fixed cost) and predict the decode tok/s ceiling with the GB10’s numbers',
    'Use streams, events and pinned memory to take the CPU off the critical path, and a side stream to fill the L2 while the DRAM would sit idle',
  ],
  prereqs: ['prefill-decode-kv', 'numeros-no-computador'],
  summary: [
    'The GPU is a grid of <b>SMs</b> (48 on the GB10), each with 4 sub-partitions; work is cut into grid → CTAs → warps of 32 lanes that execute the same instruction (SIMT) and exchange registers through shuffles.',
    'Registers and shared memory limit <b>occupancy</b>; spills go to local memory, which lives in DRAM. GLM’s FAST thin-decode gained 8–17% per layer by eliminating the LDLs. Datacenter kernels assume ~228 KB of SMEM; SM 12.x has ~100 KB, and a GB10 CTA can request at most 99 KiB (101,376 B), so the TensorFold version’s chunked KDA Triton kernels ran with a single pipeline stage, before becoming a CUDA kernel that keeps its state in registers.',
    'Tensor cores do \\(D = AB + C\\) per warp instruction; consumer Blackwell (12.x) uses <code>mma.sync</code>, including block-scaled FP4 (<code>kind::mxf4nvf4 m16n8k64</code>), but has no tcgen05/TMEM, which is why FlashMLA, MegaMoE and TRTLLM MoE do not run on the GB10.',
    'Compute capability 12.1 shares the consumer ISA with 12.0: <code>sm_120</code> cubins run on the GB10, <code>sm_121a</code> only on the GB10; gates such as <code>family(100)</code> or <code>== (12,0)</code> exclude the GB10 and turn into patches (the DeepGEMM alias, the indexer’s MXFP4 cache, the QSA table).',
    'GB10 roofline: ~125 TFLOPS BF16 ÷ ~273 GB/s ⇒ ridge ≈ 458 FLOP/B. A decode GEMV has \\(I \\approx 16M/b\\); even GLM’s prefill MoE (≈28 rows per expert, 112 FLOP/B) is memory-bound.',
    'Cost model: \\(t_{\\text{step}} \\approx \\sum Q_i / B_{\\text{eff}} + t_{\\text{fixed}}\\). Qwen: 9.2 GiB/step → 40–42 ms; τ≈3.2 → ceiling ~76–80 tok/s; measured 52.1 (step ~60 ms, ~1/3 fixed cost). DeepSeek: 16.5–17 GB/rank/step → 68–74 ms, ~80% weight streaming.',
    'Streams and events overlap small GEMMs (3 auxiliary streams in DeepSeek’s Stage-C overlay, lane B; the overlay’s <code>VLLM_DSPARK_*</code> variables are no-ops in the production image); <code>.item()/.tolist()</code> drain the queue; D2H copies are only asynchronous into pinned memory (GLM’s E1 tier). In the TensorFold version, a side stream that only reads brings the next kernels’ weights into the L2 during the phases when the DRAM sits idle (+2–3% decode, same bits).',
  ],
  quiz: [
    { q: 'On the GB10 (~125 TFLOPS BF16, ~273 GB/s), a decode GEMV with BF16 weights and M = 1 has an intensity of ≈ 1 FLOP/B. What fraction of peak compute can it use, at most?', o: ['~100%', '~14%', '~0.2%', '~50%'], a: 2, e: 'The ceiling is \\(I \\cdot B = 1 \\times 273\\) GFLOPS ≈ 0.27 TFLOPS, or 0.2% of 125 TFLOPS. That is why decode is bandwidth-bound and quantizing the weights (fewer bytes per FLOP) speeds it up almost linearly.' },
    { q: 'Does an extension compiled with <code>TORCH_CUDA_ARCH_LIST=12.1a</code> run on an RTX PRO 6000 (compute capability 12.0)?', o: ['Yes, it is the same family', 'No: the <code>a</code> suffix generates code specific to the exact CC 12.1, which does not run anywhere else', 'Yes, through backward compatibility', 'Only if the GPU has 48 SMs'], a: 1, e: '<code>sm_XYa</code> cubins use instructions specific to that architecture and run only on it. An unsuffixed <code>sm_120</code> runs on 12.1 (same major, higher minor), but the reverse (<code>sm_121a</code> on 12.0) does not.' },
    { q: 'Qwen’s FP8 KV backport to v0.30 brings a launch table tuned on sm_120, chosen by <code>_is_sm120()</code>, which tests <code>== (12, 0)</code>. What happens on the GB10?', o: ['It uses the sm_120 table', 'It fails at load time', 'It does not get the sm_120 table and stays on the default table, tuned on the GB300 (inferred from the code)', 'It automatically uses a GB10-specific profile'], a: 2, e: 'The GB10 is (12, 1). The exact equality excludes it, so the “sm_120” gain does not apply without changing the gate. That is why gates should test the family or the feature, not equality.' },
    { q: 'In GLM’s prefill at MNBT 1024 (top-8 among 288 experts), why is the MoE memory-bound even in <i>prefill</i>?', o: ['Because prefill is always memory-bound', 'Because each expert receives ~28 rows; with 4 bpw weights, \\(I \\approx 4 \\times 28 = 112\\) FLOP/B, below the ~458 ridge', 'Because the KV cache is large', 'Because the 200 Gb/s network is the limit'], a: 1, e: 'Intensity grows with the number of rows that share the weight read. With 8,192 slots split among 288 experts, M per expert is small; the report estimates 75.6 GB read per GPU per chunk (≥ 330 ms). Larger chunks or grouped kernels increase M.' },
    { q: 'Qwen reads ≈9.2 GiB per GPU per decode step (MTP=3, τ≈3.2 tokens per step). At ~240 GB/s effective, what is the approximate tok/s ceiling per sequence?', o: ['~230 tok/s', '~78 tok/s', '~24 tok/s', '~120 tok/s'], a: 1, e: '9.2 GiB ≈ 9.9 GB; 9.9/240 ≈ 41 ms per step ≈ 24 steps/s; × 3.2 tokens (2.2 accepted + the bonus) ≈ a 78 tok/s ceiling from streaming alone. The measured value (52.1) is ~1/3 lower: the rest of the ~60 ms step is fixed cost.' },
    { q: 'Why does GLM’s E1 tier copy the routing counts into a tensor with <code>pin_memory=True</code>, on a side stream?', o: ['Because pinned memory is faster for the CPU to compute on', 'Because only a D2H copy into pinned memory is truly asynchronous (DMA without the CPU); on a side stream it runs in parallel with the thin experts already queued', 'Because NCCL requires it', 'To save GPU memory'], a: 1, e: 'With pageable memory, the driver goes through an intermediate buffer and the copy becomes synchronous. With pinned memory + a side stream + an event, the GPU keeps working and the CPU waits only for the small copy before the <code>.tolist()</code>.' },
  ],
  mount(root, LA) {
    const V = LA.viz, UI = LA.ui;
    const $ = (s) => root.querySelector(s);
    const T = (g, s, x, y, o) => V.text(g, s, x, y, o || {});
    const cm = (x, d = 1) => (+x).toFixed(d);
    const fmtInt = (n) => Math.round(n).toLocaleString('en-US');
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
    function fitH(fig, fn) {
      const st = fig.querySelector('.la-fig-stage');
      const apply = () => { const w = st.getBoundingClientRect().width; if (!w) return; const hh = Math.round(fn(w)); if (Math.abs((parseFloat(st.style.height) || 0) - hh) > 1) st.style.height = hh + 'px'; };
      apply();
      const ro = new ResizeObserver(apply); ro.observe(fig);
      LA.onCleanup(() => ro.disconnect());
      return apply;
    }
    const onTxt = (C) => (C.dark ? '#06140d' : '#ffffff');
    const cellTxt = (C, strong) => (strong ? (C.dark ? '#00210F' : '#ffffff') : C.ink);
    // categorical hue by threadID_in_group (lane % 4) and shade by groupID (lane >> 2), the PTX fragment convention
    const TIG_HUE = [150, 265, 38, 330];
    const laneCol = (l, a = 1) => { const C = V.C(), grp = l >> 2; return `hsla(${TIG_HUE[l & 3]}, ${C.dark ? 72 : 64}%, ${C.dark ? 30 + grp * 6 : 26 + grp * 6.5}%, ${a})`; };
    function stripes(g, x, y, w, h, col) {
      g.save(); g.beginPath(); g.rect(x, y, w, h); g.clip();
      g.strokeStyle = col; g.lineWidth = 1.2;
      for (let k = -h; k < w; k += 6) { g.beginPath(); g.moveTo(x + k, y + h); g.lineTo(x + k + h, y); g.stroke(); }
      g.restore();
    }

    /* ======================= Figure 1: GPU zoom ======================= */
    {
      const fig = $('#fig-zoom');
      const steps = [
        { t: 'The GB10 chip', d: 'A SoC with 20 ARM cores and a Blackwell GPU with <b>48 SMs</b>, both reading the same 128 GB of LPDDR5X DRAM (~273 GB/s theoretical). Bytes come up from DRAM, pass through the shared L2 and reach the SMs. The CPU uses the same memory.' },
        { t: 'Inside an SM', d: 'Four sub-partitions, each with a warp scheduler, a slice of the register file (65,536 32-bit registers in the whole SM), INT/FP32 lanes and a tensor core. At the bottom, the L1/shared memory common to the SM (up to ~100 KB of SMEM on 12.x). Public reference architecture.' },
        { t: 'A warp: 32 lanes, one instruction', d: 'The scheduler issues <i>one</i> instruction to 32 lanes. If an <code>if</code> splits the lanes, the warp executes both sides in sequence, with half the lanes switched off in each: 50% efficiency.' },
        { t: 'A thread and its registers', d: 'Each thread has up to 255 32-bit registers, read at the pace of the instruction. If the kernel needs more, the excess goes to <b>local memory</b>, which lives in DRAM, and each <code>LDL</code> costs hundreds of cycles.' },
        { t: 'A kernel spread across the SMs', d: 'The grid is the full set of CTAs in a launch, and each CTA runs entirely on one SM. Here they are 256-thread CTAs (8 warps), as in GLM’s E2 kernel. When there are more CTAs than slots, they go in waves.' },
      ];
      const st = UI.stepper(fig, steps, { autoplay: 7500 });
      fitH(fig, (w) => (w < 560 ? 450 : 410));
      const r = V.rng(5);
      const parts = Array.from({ length: 70 }, () => ({ p: r(), sm: r.int(48), cpu: r() < 0.16, x: r() }));
      const regs = Array.from({ length: 12 }, () => Math.round((r() * 20 - 10) * 100) / 100);

      function chip(g, w, h, t, C) {
        const narrow = w < 560, pad = 12;
        const dramH = 40, dramY = h - dramH - 26;
        V.box(g, pad, dramY, w - 2 * pad, dramH, { r: 8, fill: V.alpha(C.d, 0.1), stroke: V.alpha(C.d, 0.6) });
        const nCh = narrow ? 6 : 8, chw = (w - 2 * pad - 16) / nCh;
        for (let k = 0; k < nCh; k++) V.box(g, pad + 8 + k * chw + 3, dramY + 8, chw - 6, dramH - 16, { r: 4, fill: V.alpha(C.d, 0.28), label: chw > 70 ? 'LPDDR5X' : '', size: 11 });
        T(g, 'LPDDR5X DRAM · 128 GB · ~273 GB/s · shared by CPU and GPU', w / 2, dramY + dramH + 17, { size: 11.5, align: 'center', color: C.muted, maxW: w - 20 });
        const top = 10, bodyH = dramY - 20 - top;
        let cpuB, gpuB;
        if (narrow) { cpuB = { x: pad, y: top, w: w - 2 * pad, h: 32 }; gpuB = { x: pad, y: top + 42, w: w - 2 * pad, h: bodyH - 42 }; }
        else { const cw2 = Math.min(170, w * 0.2); cpuB = { x: pad, y: top, w: cw2, h: bodyH }; gpuB = { x: pad + cw2 + 14, y: top, w: w - 2 * pad - cw2 - 14, h: bodyH }; }
        V.box(g, cpuB.x, cpuB.y, cpuB.w, cpuB.h, { r: 10, fill: V.alpha(C.b, 0.1), stroke: V.alpha(C.b, 0.7) });
        T(g, 'CPU · 20 ARM cores', cpuB.x + cpuB.w / 2, cpuB.y + 21, { size: 12.5, weight: 600, align: 'center', maxW: cpuB.w - 8 });
        if (!narrow) {
          const cs = Math.min(22, (cpuB.w - 30) / 4), ox = cpuB.x + (cpuB.w - (4 * cs + 9)) / 2;
          for (let k = 0; k < 20; k++) V.box(g, ox + (k % 4) * (cs + 3), cpuB.y + 38 + Math.floor(k / 4) * (cs + 3), cs, cs, { r: 3, fill: V.alpha(C.b, 0.22 + 0.2 * Math.sin(t * 2 + k)) });
          wrap(g, 'a few big cores: low latency', cpuB.x + 10, cpuB.y + 38 + 5 * (cs + 3) + 18, cpuB.w - 20, 15, { size: 11.5, color: C.muted });
        }
        V.box(g, gpuB.x, gpuB.y, gpuB.w, gpuB.h, { r: 10, fill: V.alpha(C.a, 0.06), stroke: V.alpha(C.a, 0.7) });
        T(g, 'Blackwell GPU · sm_121 · 48 SMs', gpuB.x + 12, gpuB.y + 20, { size: 12.5, weight: 600, maxW: gpuB.w - 20 });
        const l2H = 24, l2Y = gpuB.y + gpuB.h - l2H - 10;
        V.box(g, gpuB.x + 10, l2Y, gpuB.w - 20, l2H, { r: 6, fill: V.alpha(C.f, 0.18), stroke: V.alpha(C.f, 0.6), label: narrow ? 'shared L2' : 'L2 (shared by all SMs)', size: 11.5 });
        const cols = narrow ? 8 : 12, rows = 48 / cols, gap = 4;
        const gx0 = gpuB.x + 12, gy0 = gpuB.y + 32, gw = gpuB.w - 24, gh = l2Y - gy0 - 8;
        const cs = Math.max(4, Math.min((gw - gap * (cols - 1)) / cols, (gh - gap * (rows - 1)) / rows));
        const gxs = gx0 + (gw - (cols * cs + (cols - 1) * gap)) / 2;
        const smPos = (k) => [gxs + (k % cols) * (cs + gap) + cs / 2, gy0 + Math.floor(k / cols) * (cs + gap) + cs / 2];
        const heat = new Array(48).fill(0);
        const pts = parts.map((p) => {
          const u = (p.p + t * 0.22) % 1;
          const sx = pad + 20 + p.x * (w - 2 * pad - 40), sy = dramY + 4;
          const l2x = gpuB.x + 20 + p.x * (gpuB.w - 40), l2y = l2Y + l2H / 2;
          if (p.cpu) return [V.lerp(sx, cpuB.x + cpuB.w / 2, u), V.lerp(sy, cpuB.y + cpuB.h - 8, u), C.b];
          if (u < 0.45) { const v = u / 0.45; return [V.lerp(sx, l2x, v), V.lerp(sy, l2y, v), C.d]; }
          const v = (u - 0.45) / 0.55, [tx, ty] = smPos(p.sm);
          if (v > 0.85) heat[p.sm] = Math.max(heat[p.sm], (v - 0.85) / 0.15);
          return [V.lerp(l2x, tx, v), V.lerp(l2y, ty, v), C.a];
        });
        for (let k = 0; k < 48; k++) {
          const [cx, cy] = smPos(k);
          V.box(g, cx - cs / 2, cy - cs / 2, cs, cs, { r: Math.min(4, cs / 4), fill: V.alpha(C.a, 0.12 + 0.5 * heat[k]), stroke: V.alpha(C.a, 0.55), lw: 1 });
          if (cs >= 30) T(g, 'SM', cx, cy + 4, { size: 11, align: 'center', color: C.muted, font: 'mono' });
        }
        pts.forEach(([x, y, c]) => V.dot(g, x, y, 2.3, c));
      }

      function sm(g, w, h, t, C) {
        const pad = 12, narrow = w < 560;
        const bx = pad, by = 8, bw = w - 2 * pad, bh = h - 16;
        V.box(g, bx, by, bw, bh, { r: 12, fill: V.alpha(C.a, 0.05), stroke: V.alpha(C.a, 0.7) });
        T(g, 'One SM (1 of 48)', bx + 12, by + 20, { size: 13, weight: 600 });
        if (!narrow) T(g, 'reference architecture', bx + bw - 12, by + 20, { size: 11, align: 'right', color: C.muted });
        const smemH = 32, smemY = by + bh - smemH - 10;
        V.box(g, bx + 10, smemY, bw - 20, smemH, { r: 8, fill: V.alpha(C.f, 0.16), stroke: V.alpha(C.f, 0.6), label: narrow ? 'L1 / shared memory (~100 KB SMEM on 12.x)' : 'L1 / shared memory, common to the 4 sub-partitions (up to ~100 KB of SMEM on 12.x)', size: 11.5 });
        const gx = bx + 10, gy = by + 32, gw = bw - 20, gh = smemY - gy - 8, gp = 8;
        const qw = (gw - gp) / 2, qh = (gh - gp) / 2;
        for (let q = 0; q < 4; q++) {
          const x = gx + (q % 2) * (qw + gp), y = gy + Math.floor(q / 2) * (qh + gp);
          V.box(g, x, y, qw, qh, { r: 8, fill: V.alpha(C.muted, 0.06), stroke: V.alpha(C.muted, 0.35), lw: 1 });
          T(g, `sub-partition ${q}`, x + 8, y + 16, { size: 11.5, weight: 600 });
          const wid = (q * 7 + Math.floor(t * 2.5)) % 12;
          V.box(g, x + 8, y + 24, Math.min(qw - 16, 160), 20, { r: 5, fill: V.alpha(C.d, 0.18), stroke: V.alpha(C.d, 0.6), label: qw > 150 ? `scheduler → warp ${wid}` : `warp ${wid}`, size: 11 });
          const ry = y + 50, rh = Math.max(10, Math.min(18, qh * 0.12));
          const nseg = 16, sw = (qw - 20) / nseg;
          for (let k = 0; k < nseg; k++) { g.fillStyle = V.alpha(C.b, 0.2 + 0.45 * (0.5 + 0.5 * Math.sin(t * 3 + k * 0.7 + q))); V.rr(g, x + 10 + k * sw, ry, sw - 2, rh, 2); g.fill(); }
          T(g, 'registers (¼ of the file)', x + 10, ry + rh + 13, { size: 11, color: C.muted, maxW: qw - 20 });
          const ly = ry + rh + 22, tcW = Math.min(92, qw * 0.32), lw = qw - 20 - tcW - 8;
          const ls = Math.max(3, Math.min(10, lw / 16 - 2)), lh2 = ls;
          for (let k = 0; k < 32; k++) { const on = Math.sin(t * 4 + k * 0.5 + q) > -0.2; g.fillStyle = on ? V.alpha(C.a, 0.75) : V.alpha(C.a, 0.2); V.rr(g, x + 10 + (k % 16) * (ls + 2), ly + Math.floor(k / 16) * (lh2 + 3), ls, lh2, 2); g.fill(); }
          if (qh > 130) T(g, '32 lanes INT/FP32', x + 10, ly + 2 * (lh2 + 3) + 13, { size: 11, color: C.muted, maxW: lw });
          const tcx = x + qw - 10 - tcW, tcy = ly - 2, tch = Math.min(qh - (ly - y) - 8, 44);
          const pulse = 0.5 + 0.5 * Math.sin(t * 3 + q);
          V.box(g, tcx, tcy, tcW, Math.max(20, tch), { r: 6, fill: V.alpha(C.c, 0.15 + 0.2 * pulse), stroke: C.c, label: 'tensor core', size: 11 });
        }
      }

      function warp(g, w, h, t, C) {
        const narrow = w < 700;
        const phase = Math.floor(t / 1.8) % 3, pu = (t % 1.8) / 1.8;
        const instr = ['FFMA  d = a × b + c', 'if (lane < 16)  x = f(x)', 'else  x = g(x)'][phase];
        const active = (l) => (phase === 0 ? true : phase === 1 ? l < 16 : l >= 16);
        const ibw = Math.min(w - 40, 380), ibx = (w - ibw) / 2, iby = 12, ibh = 32;
        V.box(g, ibx, iby, ibw, ibh, { r: 8, fill: V.alpha(C.d, 0.16), stroke: C.d, label: instr, font: 'mono', size: 12.5 });
        T(g, 'the scheduler issues 1 instruction to the whole warp', w / 2, iby + ibh + 16, { size: 11.5, align: 'center', color: C.muted, maxW: w - 20 });
        const perRow = w < 560 ? 8 : narrow ? 16 : 32;
        const sp = (w - 32) / perRow, R = Math.min(11, sp * 0.36);
        const y0 = w < 560 ? h * 0.3 : narrow ? h * 0.4 : h * 0.52;
        for (let l = 0; l < 32; l++) {
          const cx = 16 + sp * (l % perRow) + sp / 2, cy = y0 + Math.floor(l / perRow) * (R * 2 + 36);
          const on = active(l);
          if (on) { V.line(g, w / 2, iby + ibh, cx, cy - R, { color: V.alpha(C.d, 0.12), lw: 1 }); const [px, py] = V.along(w / 2, iby + ibh, cx, cy - R, pu); V.dot(g, px, py, 2, C.d); }
          g.beginPath(); g.arc(cx, cy, R, 0, Math.PI * 2);
          g.fillStyle = on ? V.alpha(C.a, 0.3 + 0.45 * Math.pow(Math.sin(t * 5 + l * 0.3), 2)) : V.alpha(C.muted, 0.08); g.fill();
          g.strokeStyle = on ? C.a : V.alpha(C.muted, 0.4); g.lineWidth = 1.5; g.stroke();
          T(g, String(l), cx, cy + R + 13, { size: 11, align: 'center', font: 'mono', color: on ? C.ink : C.muted });
        }
        const nOn = phase === 0 ? 32 : 16;
        wrap(g, `active lanes: ${nOn}/32 · efficiency ${Math.round((nOn / 32) * 100)}%` + (phase ? ' · divergence: the warp walks both sides of the if' : ' · 1 instruction, 32 results'), 16, h - 34, w - 32, 16, { size: 12.5, weight: 600, color: phase ? C.c : C.a, maxLines: 2 });
      }

      function thread(g, w, h, t, C) {
        const pad = 14, colW = Math.min(170, w * 0.4);
        T(g, 'the thread’s registers', pad, 20, { size: 12, weight: 600, maxW: colW + 40 });
        const n = regs.length, rh = Math.min(22, (h - 120) / n);
        for (let k = 0; k < n; k++) {
          const y = 30 + k * rh, hot = k < 3 && Math.floor(t * 2.5) % 2 === 0;
          V.box(g, pad, y, colW, rh - 3, { r: 4, fill: hot ? V.alpha(C.a, 0.4) : V.alpha(C.a, 0.1), stroke: V.alpha(C.a, 0.5), lw: 1 });
          T(g, 'r' + k, pad + 8, y + (rh - 3) / 2 + 0.5, { size: 11, font: 'mono', color: C.muted, base: 'middle' });
          T(g, cm(regs[k] + Math.sin(t * 0.7 + k) * 0.5, 2), pad + colW - 8, y + (rh - 3) / 2 + 0.5, { size: 11, font: 'mono', align: 'right', base: 'middle' });
        }
        T(g, '⋮  up to r254', pad + 8, 30 + n * rh + 12, { size: 11, font: 'mono', color: C.muted });
        const rx = pad + colW + 26, rw = w - rx - pad;
        const alu = { x: rx, y: 30, w: rw, h: 48 };
        V.box(g, alu.x, alu.y, alu.w, alu.h, { r: 8, fill: V.alpha(C.d, 0.14), stroke: C.d, label: 'FFMA  r0 = r1 × r2 + r3', font: 'mono', size: 11.5 });
        for (let k = 0; k < 3; k++) {
          const yy = 30 + (k + 1) * rh + rh / 2, u = (t * 2.4 + k * 0.2) % 1;
          V.line(g, pad + colW, yy, alu.x, alu.y + alu.h / 2, { color: V.alpha(C.a, 0.3), lw: 1 });
          const [px, py] = V.along(pad + colW, yy, alu.x, alu.y + alu.h / 2, u); V.dot(g, px, py, 2.4, C.a);
        }
        wrap(g, 'register → unit: at the pace of the instruction', rx, alu.y + alu.h + 16, rw, 15, { size: 11.5, color: C.a, maxLines: 2 });
        const lm = { x: rx, y: alu.y + alu.h + 52, w: rw, h: 40 };
        V.box(g, lm.x, lm.y, lm.w, lm.h, { r: 8, fill: V.alpha(C.c, 0.12), stroke: V.alpha(C.c, 0.7), label: 'local memory (spill)', size: 11.5 });
        const dr = { x: pad, y: h - 50, w: w - 2 * pad, h: 32 };
        V.box(g, dr.x, dr.y, dr.w, dr.h, { r: 8, fill: V.alpha(C.d, 0.12), stroke: V.alpha(C.d, 0.6), label: 'DRAM: this is where local memory really lives', size: 11.5 });
        const x1 = lm.x + lm.w / 2, y1 = lm.y + lm.h, y2 = dr.y;
        V.line(g, x1, y1, x1, y2, { color: V.alpha(C.c, 0.5), lw: 1.5, dash: [4, 5], dashOffset: -t * 10 });
        const u2 = (t / 3.2) % 1, pp = u2 < 0.5 ? u2 * 2 : 2 - u2 * 2;
        V.dot(g, x1, V.lerp(y1, y2, pp), 3.2, C.c);
        V.line(g, pad + colW, 30 + (n - 1) * rh, lm.x, lm.y + lm.h / 2, { color: V.alpha(C.c, 0.35), lw: 1, dash: [3, 4] });
        if (y2 - y1 > 40) wrap(g, 'STL/LDL: hundreds of cycles per access', x1 + 10, (y1 + y2) / 2, rw / 2 + 20, 14, { size: 11.5, color: C.c, maxLines: 3 });
      }

      function kernel(g, w, h, t, C) {
        const nC = 64, pad = 12, narrow = w < 560;
        T(g, 'grid: 64 CTAs × 256 threads (8 warps each)', pad, 18, { size: 12, weight: 600, maxW: w - 24 });
        const period = 7, ph = (t % period) / period, wave = ph < 0.5 ? 0 : 1, wp = (ph % 0.5) / 0.5;
        const gcols = narrow ? 8 : 16, grows = nC / gcols, gg = 4;
        const gcw = Math.min(34, (w - 2 * pad - (gcols - 1) * gg) / gcols), gch = Math.min(20, gcw * 0.62);
        const gX = (w - (gcols * gcw + (gcols - 1) * gg)) / 2, gY = 28;
        const smTop = gY + grows * (gch + gg) + 26;
        const scol = narrow ? 8 : 12, srow = 48 / scol, sg = 5;
        const scs = Math.max(8, Math.min((w - 2 * pad - (scol - 1) * sg) / scol, (h - smTop - 40 - (srow - 1) * sg) / srow));
        const sX = (w - (scol * scs + (scol - 1) * sg)) / 2;
        const smXY = (k) => [sX + (k % scol) * (scs + sg), smTop + Math.floor(k / scol) * (scs + sg)];
        T(g, '48 SMs (illustrative: 1 CTA per SM)', pad, smTop - 8, { size: 11.5, color: C.muted, maxW: w - 24 });
        for (let k = 0; k < 48; k++) { const [x, y] = smXY(k); V.box(g, x, y, scs, scs, { r: 4, fill: V.alpha(C.muted, 0.06), stroke: V.alpha(C.muted, 0.35), lw: 1 }); }
        for (let c = 0; c < nC; c++) {
          const cw_ = c < 48 ? 0 : 1;
          if (cw_ < wave) continue;
          const gx = gX + (c % gcols) * (gcw + gg), gy = gY + Math.floor(c / gcols) * (gch + gg);
          let x = gx, y = gy, ww = gcw, hh = gch, run = 0;
          if (cw_ === wave) {
            const k = cw_ === 0 ? c : c - 48, [sx, sy] = smXY(k);
            const mv = V.ease.inOut(V.clamp(wp / 0.35, 0, 1));
            x = V.lerp(gx, sx + 2, mv); y = V.lerp(gy, sy + 2, mv); ww = V.lerp(gcw, scs - 4, mv); hh = V.lerp(gch, scs - 4, mv);
            run = V.clamp((wp - 0.35) / 0.6, 0, 1);
          }
          const col = C.series[cw_ === 0 ? 0 : 2];
          V.box(g, x, y, ww, hh, { r: 3, fill: V.alpha(col, 0.22), stroke: V.alpha(col, 0.8), lw: 1 });
          const sw = ww / 8;
          for (let s = 0; s < 8; s++) { g.fillStyle = V.alpha(col, s / 8 < run ? 0.85 : 0.35); g.fillRect(x + s * sw + 1, y + 2, Math.max(1, sw - 2), Math.max(1, hh - 4)); }
        }
        T(g, wave === 0 ? 'wave 1: 48 CTAs fill the 48 SMs' : 'wave 2: the remaining 16 CTAs; 32 SMs idle (tail effect)', w / 2, h - 14, { size: 12, align: 'center', weight: 600, color: wave ? C.c : C.a, maxW: w - 20 });
      }

      const scenes = [chip, sm, warp, thread, kernel];
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const e = V.ease.out(Math.min(1, st.elapsed() / 0.5));
        g.globalAlpha = 0.25 + 0.75 * e;
        const sc = 0.96 + 0.04 * e;
        g.translate((w / 2) * (1 - sc), (h / 2) * (1 - sc)); g.scale(sc, sc);
        scenes[st.i](g, w, h, t, C);
        g.globalAlpha = 1;
      });
    }

    /* ================= Figure 2: shuffle butterfly ================= */
    {
      const fig = $('#fig-shuffle');
      const S = { n: 8, stage: 3, norm: true, auto: true, seed: 7 };
      let x = [], rows = [];
      function build() {
        const r = V.rng(S.seed * 31 + S.n);
        x = Array.from({ length: S.n }, () => r.int(9) - 4);
        rows = [x.slice()];
        const L = Math.log2(S.n);
        for (let s = 0; s < L; s++) { const m = 1 << s, prev = rows[s]; rows.push(prev.map((v, l) => ((l & m) ? prev[l ^ m] - v : v + prev[l ^ m]))); }
      }
      build();
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Lanes', options: [8, 16, 32], value: 8, onChange: (v) => { S.n = v; build(); const L = Math.log2(v); sl.input.max = L; if (S.stage > L) S.stage = L; sl.input.value = S.stage; const vv = sl.el.querySelector('.val'); if (vv) vv.textContent = String(S.stage); } });
      let tg = null;
      const sl = UI.slider(ctl, { label: 'Stage', min: 0, max: 3, step: 1, value: 3, onInput: (v) => { S.stage = Math.min(v, Math.log2(S.n)); if (S.auto && tg) tg.set(false); } });
      tg = UI.toggle(ctl, { label: 'auto-advance', value: true, onChange: (v) => { S.auto = v; } });
      UI.toggle(ctl, { label: '÷√n (normalize)', value: true, onChange: (v) => { S.norm = v; } });
      UI.button(ctl, '🎲 new vector', () => { S.seed++; build(); });
      V.fig(fig, (g, w, h, t) => {
        const C = V.C(), n = S.n, L = Math.log2(n);
        const seq = [L, L].concat(Array.from({ length: L }, (_, k) => k));
        const cur = S.auto ? seq[Math.floor(t / 1.5) % seq.length] : Math.min(S.stage, L);
        const lw = w < 520 ? 34 : 66, top = 30, bottom = 60;
        const colW = (w - lw - 12) / n, rowH = (h - top - bottom) / (L + 1);
        const ch = Math.min(28, rowH * 0.56), cw = Math.max(3, colW - (n > 16 ? 2 : 4));
        const cx = (l) => lw + l * colW + colW / 2, cy = (r) => top + r * rowH + ch / 2;
        let mx = 1; rows.forEach((row) => row.forEach((v) => { if (Math.abs(v) > mx) mx = Math.abs(v); }));
        T(g, cur < L ? `next: stage ${cur} · partner = lane ⊕ ${1 << cur}` : `done: ${L} stages = log₂(${n})`, lw, 17, { size: 12.5, weight: 600, maxW: w - lw - 8 });
        for (let s = 0; s < L; s++) {
          const m = 1 << s, done = s < cur, isLast = s === cur - 1;
          for (let l = 0; l < n; l++) {
            const p = l ^ m, x1 = cx(l), y1 = cy(s) + ch / 2, x2 = cx(p), y2 = cy(s + 1) - ch / 2;
            const col = done ? (isLast ? C.c : V.alpha(C.muted, 0.35)) : V.alpha(C.muted, 0.13);
            V.line(g, x1, y1, x1, y2, { color: col, lw: isLast ? 1.4 : 1 });
            V.line(g, x1, y1, x2, y2, { color: col, lw: isLast ? 1.4 : 1, dash: done ? null : [3, 4] });
            if (isLast) { const u = (t * 0.9 + l * 0.07) % 1; const [px, py] = V.along(x2, y1, x1, y2, u); V.dot(g, px, py, 2.2, C.c); }
          }
        }
        for (let r = 0; r <= L; r++) {
          T(g, r === 0 ? 'x' : 's=' + (r - 1), 6, cy(r) + 4, { size: 11, font: 'mono', color: C.muted });
          if (lw > 40 && r > 0) T(g, '⊕' + (1 << (r - 1)), 6, cy(r) + 17, { size: 11, font: 'mono', color: C.muted });
          for (let l = 0; l < n; l++) {
            const shown = r <= cur, v = rows[r][l], fin = r === L && S.norm;
            const vv = fin ? v / Math.sqrt(n) : v;
            const x0 = cx(l) - cw / 2, y0 = cy(r) - ch / 2, tt = Math.abs(v / mx);
            g.fillStyle = shown ? V.cmap(v / mx, 'div') : V.alpha(C.muted, 0.06);
            V.rr(g, x0, y0, cw, ch, Math.min(5, cw / 4)); g.fill();
            if (r === cur && r > 0 && shown) { g.strokeStyle = C.c; g.lineWidth = 1.6; V.rr(g, x0 - 1, y0 - 1, cw + 2, ch + 2, 5); g.stroke(); }
            if (shown && cw >= 18) T(g, fin ? cm(vv, cw >= 36 ? 2 : 1) : String(vv), cx(l), cy(r) + 0.5, { size: 11, font: 'mono', align: 'center', base: 'middle', color: cellTxt(C, tt > 0.55) });
          }
        }
        if (colW >= 16) for (let l = 0; l < n; l++) T(g, String(l), cx(l), cy(L) + ch / 2 + 13, { size: 11, font: 'mono', align: 'center', color: C.muted });
        const ref = LA.num.fwht(x), ok = ref.every((v, i) => Math.abs(v - rows[L][i] / Math.sqrt(n)) < 1e-9);
        T(g, 'v = (lane & m) ? shfl_xor(v, m) − v : v + shfl_xor(v, m)', lw, h - 26, { size: 11.5, font: 'mono', color: C.muted, maxW: w - lw - 8 });
        T(g, cur === L ? (ok ? '✓ matches LA.num.fwht(x)' + (S.norm ? '' : ' (after ÷√n)') + ': H·x with registers only' : '✗ diverged from the reference') : `${cur} of ${L} stages · no memory reads`, lw, h - 9, { size: 12, weight: 600, color: cur === L ? C.a : C.ink, maxW: w - lw - 8 });
      });
    }

    /* =================== Figure 3: occupancy calculator =================== */
    {
      const fig = $('#fig-occ');
      const ARCH = { c: { name: 'SM 12.x (GB10)', R: 65536, S: 100, Sc: 99, W: 48 }, d: { name: 'SM 9.0/10.0 (datacenter)', R: 65536, S: 228, Sc: 227, W: 64 } };
      const S = { a: 'c', T: 256, r: 96, s: 32, cap: 255 };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'SM', options: [{ v: 'c', label: '12.x · GB10' }, { v: 'd', label: '9.0/10.0 · datacenter' }], value: 'c', onChange: (v) => { S.a = v; upd(); } });
      UI.slider(ctl, { label: 'Threads/CTA', min: 32, max: 1024, step: 32, value: 256, onInput: (v) => { S.T = v; upd(); } });
      UI.slider(ctl, { label: 'Regs/thread', min: 16, max: 255, step: 1, value: 96, onInput: (v) => { S.r = v; upd(); } });
      UI.slider(ctl, { label: 'SMEM per CTA', min: 0, max: 228, step: 2, value: 32, fmt: (v) => v + ' KB', onInput: (v) => { S.s = v; upd(); } });
      UI.slider(ctl, { label: 'Reg cap', min: 32, max: 255, step: 1, value: 255, onInput: (v) => { S.cap = v; upd(); } });
      const rN = UI.readout(ctl, 'CTAs/SM', '—'), rW = UI.readout(ctl, 'warps/SM', '—'), rO = UI.readout(ctl, 'occupancy', '—'), rL = UI.readout(ctl, 'limiter', '—');
      let M = null;
      function calc() {
        const A = ARCH[S.a], rEff = Math.min(S.r, S.cap), spill = Math.max(0, S.r - S.cap), warpsC = S.T / 32;
        const byR = Math.floor(A.R / (rEff * S.T)), byS = S.s > 0 ? Math.floor(A.S / S.s) : Infinity, byW = Math.floor(A.W / warpsC);
        const fits = S.s <= A.Sc && rEff * S.T <= A.R;
        const n = fits ? Math.min(byR, byS, byW) : 0;
        const lim = !fits ? (S.s > A.Sc ? 'smem' : 'reg') : n === byR ? 'reg' : n === byS ? 'smem' : 'warp';
        return { A, rEff, spill, warpsC, n, lim, fits, occ: (n * warpsC) / A.W, warps: n * warpsC };
      }
      const limName = { reg: 'registers', smem: 'shared memory', warp: 'warp slots' };
      function upd() { M = calc(); rN.set(String(M.n)); rW.set(`${M.warps}/${M.A.W}`); rO.set(Math.round(M.occ * 100) + '%'); rL.set(M.fits ? limName[M.lim] : 'won’t launch'); }
      upd();
      fitH(fig, (w) => (w < 560 ? 472 : 420));
      const wr = V.rng(9), wph = Array.from({ length: 64 }, () => wr());
      V.fig(fig, (g, w, h, t) => {
        const C = V.C(), m = M, A = m.A, pad = 14, narrow = w < 560;
        wrap(g, `${A.name}: ${fmtInt(A.R)} registers · ${A.S} KB of SMEM · ${A.W} warps per SM (public table)`, pad, 18, w - 2 * pad, 15, { size: 12, weight: 600, maxLines: 2 });
        const bars = [
          { key: 'reg', label: 'register file', cap: A.R, per: m.rEff * S.T, unit: (x) => fmtInt(x) + ' regs' },
          { key: 'smem', label: 'shared memory', cap: A.S, per: S.s, unit: (x) => x + ' KB' },
          { key: 'warp', label: 'warp slots', cap: A.W, per: m.warpsC, unit: (x) => x + ' warps' },
        ];
        const bx = narrow ? pad : 168, bw = w - bx - pad, bh = 20;
        let y = narrow ? 46 : 36;
        bars.forEach((b) => {
          if (narrow) { T(g, b.label, pad, y + 10, { size: 11.5, color: C.muted }); y += 15; } else T(g, b.label, pad, y + bh / 2 + 4, { size: 11.5, color: C.muted });
          V.box(g, bx, y, bw, bh, { r: 5, fill: V.alpha(C.muted, 0.08), stroke: V.alpha(C.muted, 0.3), lw: 1 });
          const fr = Math.min(1, b.per / b.cap);
          if (b.key === 'warp') { g.strokeStyle = V.alpha(C.muted, 0.18); g.lineWidth = 1; for (let k = 1; k < A.W; k++) { const xx = bx + (k * bw) / A.W; g.beginPath(); g.moveTo(xx, y + 3); g.lineTo(xx, y + bh - 3); g.stroke(); } }
          for (let k = 0; k < m.n; k++) { const x0 = bx + k * fr * bw; const ww = Math.min(Math.max(1, fr * bw - 2), bx + bw - x0 - 2); if (ww <= 0) break; g.fillStyle = V.alpha(C.series[k % 6], 0.72); V.rr(g, x0 + 1, y + 2, ww, bh - 4, 3); g.fill(); }
          const isLim = m.lim === b.key;
          if (isLim) { g.strokeStyle = C.c; g.lineWidth = 2; V.rr(g, bx - 2, y - 2, bw + 4, bh + 4, 6); g.stroke(); }
          const used = b.per * m.n;
          T(g, `${b.unit(b.per)} per CTA · ${Math.round((100 * used) / b.cap)}% used${isLim ? ' · LIMITS' : ''}`, bx + bw, y + bh + 13, { size: 11, align: 'right', color: isLim ? C.c : C.muted, font: 'mono', maxW: bw });
          y += bh + 22;
        });
        y += 6;
        T(g, 'schedulers: resident warps taking turns', pad, y + 10, { size: 11.5, weight: 600, maxW: w - 2 * pad });
        y += 18;
        const slotW = Math.max(3, Math.min(14, (w - 2 * pad) / A.W - 2));
        for (let k = 0; k < A.W; k++) {
          const x0 = pad + k * (slotW + 2);
          if (k >= m.warps) { g.strokeStyle = V.alpha(C.muted, 0.25); g.lineWidth = 1; V.rr(g, x0, y, slotW, 16, 3); g.stroke(); continue; }
          const ph = (t * 0.45 + wph[k]) % 1, busy = ph < 1 / 12;
          g.fillStyle = busy ? C.a : V.alpha(C.b, 0.32); V.rr(g, x0, y, slotW, 16, 3); g.fill();
        }
        y += 24;
        const util = Math.min(1, m.warps / 48);
        const mw = w - 2 * pad;
        V.box(g, pad, y, mw, 14, { r: 4, fill: V.alpha(C.muted, 0.1) });
        V.box(g, pad, y, Math.max(2, mw * util), 14, { r: 4, fill: V.alpha(C.a, 0.7) });
        y += 28;
        wrap(g, `busy units ≈ ${Math.round(util * 100)}%. Illustrative: each warp computes for 1 time unit and waits ~11 for DRAM, so each of the 4 sub-partitions needs ~12 ready warps.`, pad, y, mw, 15, { size: 11.5, color: C.muted, maxLines: 3 });
        y += 40;
        let boxH = 44;
        if (!m.fits) {
          const msg = S.s > A.Sc ? `won’t launch: ${S.s} KB of SMEM per CTA exceeds this SM’s maximum of ${A.Sc} KB` : `won’t launch: ${m.rEff} × ${S.T} = ${fmtInt(m.rEff * S.T)} registers in one CTA exceed the SM’s ${fmtInt(A.R)}`;
          const n = Math.min(3, splitLines(g, msg, mw - 20, { size: 12, weight: 600 }).length); boxH = Math.max(40, 12 + n * 15 + 6);
          V.box(g, pad, y, mw, boxH, { r: 8, fill: V.alpha(C.c, 0.14), stroke: C.c });
          wrap(g, msg, pad + 10, y + 16, mw - 20, 15, { size: 12, weight: 600, color: C.c, maxLines: 3 });
        } else if (m.spill > 0) {
          const bytes = m.spill * 4;
          const msg = `spill: ${m.spill} registers = ${bytes} B per thread in local memory (DRAM): ${fmtInt(bytes * m.n * S.T)} B of STL/LDL per SM on every pass. In the GLM recipe, FAST took the LDLs to zero (39 → 0).`;
          const n = Math.min(4, splitLines(g, msg, mw - 36, { size: 11.5 }).length); boxH = Math.max(44, 12 + n * 15 + 6);
          V.box(g, pad, y, mw, boxH, { r: 8, fill: V.alpha(C.c, 0.1), stroke: V.alpha(C.c, 0.6) });
          for (let k = 0; k < 6; k++) { const u = (t * 0.6 + k / 6) % 1; V.dot(g, pad + 14, y + 6 + ((u + k / 6) % 1) * (boxH - 12), 2.4, C.c); }
          wrap(g, msg, pad + 26, y + 16, mw - 36, 15, { size: 11.5, maxLines: 4 });
        } else {
          T(g, '✓ no spill: everything fits in the allowed registers', pad, y + 16, { size: 12, weight: 600, color: C.a, maxW: mw });
        }
        wrap(g, 'Real case (GLM, SM121): stock exl3_moe 128 regs, 88 B frame, 39 LDL → FAST 127 regs, 0 LDL, same 1,024 B of SMEM and same occupancy: +8–17% per layer.', pad, y + boxH + 18, mw, 15, { size: 11.5, color: C.glm, maxLines: 3 });
      });
    }

    /* =================== Figure 4: tensor core instruction =================== */
    {
      const fig = $('#fig-mma');
      const S = { mode: 'fp16', lanes: false };
      const r = V.rng(21);
      const A16 = Array.from({ length: 16 }, () => Array.from({ length: 16 }, () => r.int(7) - 3));
      const B16 = Array.from({ length: 16 }, () => Array.from({ length: 8 }, () => r.int(5) - 2));
      const E2 = [0, 0.5, 1, 1.5, 2, 3, 4, 6];
      const pick = () => (r() < 0.5 ? -1 : 1) * E2[r.int(8)];
      const A4 = Array.from({ length: 16 }, () => Array.from({ length: 64 }, pick));
      const B4 = Array.from({ length: 64 }, () => Array.from({ length: 8 }, pick));
      const SC = [0.5, 1, 2];
      const sA = Array.from({ length: 16 }, () => Array.from({ length: 4 }, () => SC[r.int(3)]));
      const sB = Array.from({ length: 4 }, () => Array.from({ length: 8 }, () => SC[r.int(3)]));
      const D16 = A16.map((row) => Array.from({ length: 8 }, (_, j) => row.reduce((s, a, k) => s + a * B16[k][j], 0)));
      const part4 = (i, j, b) => { let s = 0; for (let k = b * 16; k < b * 16 + 16; k++) s += A4[i][k] * B4[k][j]; return s; };
      const D4 = Array.from({ length: 16 }, (_, i) => Array.from({ length: 8 }, (_, j) => { let s = 0; for (let b = 0; b < 4; b++) s += sA[i][b] * sB[b][j] * part4(i, j, b); return s; }));
      let mD4 = 0; D4.forEach((row) => row.forEach((v) => { mD4 = Math.max(mD4, Math.abs(v)); }));
      let mD16 = 0; D16.forEach((row) => row.forEach((v) => { mD16 = Math.max(mD16, Math.abs(v)); }));
      // PTX fragment convention for mma.m16n8k16 (groupID = lane >> 2, threadID_in_group = lane % 4)
      const laneA = (rr, c) => (rr % 8) * 4 + ((c % 8) >> 1);
      const laneB = (k, n) => n * 4 + ((k % 8) >> 1);
      const laneD = (rr, n) => (rr % 8) * 4 + (n >> 1);
      const nf = (x) => String(Math.round(x * 100) / 100);
      const ITEMS = {
        fp16: ['mma.sync.m16n8k16 · FP16 → FP32', '2·16·8·16 = 4,096 FLOPs in one warp instruction', 'each lane carries 8 elements of A, 4 of B and 4 of the accumulator', 'no lane has the whole matrix: turn on “color by lane”', 'hover (or tap) over D to pick the cell'],
        fp4: ['mma.sync … kind::mxf4nvf4 … m16n8k64', '16,384 FLOPs in one warp instruction: K = 64, in 4 blocks of 16', 'each lane carries 32 E2M1 codes of A, 16 of B and 4 FP32 accumulators', 'sA (16×4) and sBᵀ (8×4): one UE4M3 scale per block of 16 (scale_vec::4X)', 'the same 768 B of operands as FP16, with 4× the FLOPs'],
      };
      const MC = document.createElement('canvas').getContext('2d');
      const itemStyle = (k) => ({ size: k === 0 ? 11.5 : 12, font: k === 0 ? 'mono' : 'sans', weight: k === 0 ? 600 : 400 });
      function itemsH(items, width) { let hh = 0; items.forEach((it, k) => { hh += Math.min(3, splitLines(MC, it, width, itemStyle(k)).length) * 15 + 7; }); return hh; }
      // single layout, used by the drawing and by the stage height (fitH)
      function layout(w, fp4) {
        const pad = 12, narrow = w < 560, extra = fp4 ? 5 : 0, gB = fp4 ? 2 : 1.4;
        const ay = fp4 ? 58 : 22;
        const availW = w - 2 * pad - 26 - (narrow ? 0 : 250);
        const cs = Math.max(6, Math.min(narrow ? 22 : 12, availW / (extra + 16 + 1.6 + 8)));
        const ax = pad + 22 + extra * cs, dx = ax + 16 * cs + 1.6 * cs, by = ay + 16 * cs + gB * cs;
        const matBottom = by + 8 * cs + 20;
        const pX = dx + 8 * cs + 22, pW = w - pX - pad, side = !narrow && pW > 170;
        const all = ITEMS[fp4 ? 'fp4' : 'fp16'];
        // when narrow, the list goes below; drop the item that only repeats the readout (FP4) or the mouse hint (FP16)
        const items = side || !narrow ? all : all.filter((_, k) => (fp4 ? k !== 1 : k !== 4));
        const listY = matBottom + 4, listH = side ? 0 : itemsH(items, w - 2 * pad - 20) + 14;
        const lensTop = side ? matBottom + 4 : listY + listH;
        const lx = pad + (fp4 ? (narrow ? 70 : 64) : (narrow ? 40 : 64));
        let per = 16, lc = Math.min(fp4 ? 34 : 36, (w - pad - lx - 10) / 16);
        if (fp4 && lc < 24) { per = 8; lc = Math.min(34, (w - pad - lx - 10) / 8); }
        const ch2 = Math.min(fp4 ? 20 : 22, lc);
        const chipPer = narrow ? 2 : 4;
        const lensH = fp4 ? 28 + 2 * (16 / per) * (ch2 + 5) + 6 + (4 / chipPer) * 27 + 28 : 30 + 3 * (ch2 + 6) + 28;
        return { pad, narrow, extra, gB, ay, cs, ax, dx, by, matBottom, pX, pW, side, items, listY, listH, lensTop, lx, per, lc, ch2, chipPer, lensH, H: Math.round(lensTop + lensH + 4) };
      }
      const ctl = UI.controls(fig);
      let applyH = () => {};
      UI.seg(ctl, { label: 'Instruction', options: [{ v: 'fp16', label: 'FP16 · m16n8k16' }, { v: 'fp4', label: 'NVFP4 · m16n8k64' }], value: 'fp16', onChange: (v) => { S.mode = v; upd(); applyH(); } });
      UI.toggle(ctl, { label: 'color by lane (FP16)', value: false, onChange: (v) => { S.lanes = v; } });
      const ro1 = UI.readout(ctl, 'FLOPs/instruction', '—'), ro2 = UI.readout(ctl, 'per lane (A · B · C)', '—'), ro3 = UI.readout(ctl, 'bytes of A + B', '—');
      function upd() {
        if (S.mode === 'fp16') { ro1.set('4,096'); ro2.set('8 · 4 · 4'); ro3.set('768 B'); }
        else { ro1.set('16,384'); ro2.set('32 · 16 · 4'); ro3.set('768 B + 96 B scales'); }
      }
      upd();
      applyH = fitH(fig, (w) => layout(w, S.mode === 'fp4').H);
      const H = V.fig(fig, (g, w, h, t) => {
        const C = V.C(), fp4 = S.mode === 'fp4', Lo = layout(w, fp4);
        const { pad, narrow, cs, ax, ay, dx, by } = Lo;
        const P = H.pointer;
        const inR = (x0, y0, ww, hh) => P.inside && P.x >= x0 && P.x < x0 + ww && P.y >= y0 && P.y < y0 + hh;
        // block on stage (FP4): automatic, or whichever is under the mouse on the ruler
        const stripY = 26, stripH = 16;
        let bb = fp4 ? Math.floor(t / 2.4) % 4 : 0;
        if (fp4 && inR(ax, stripY - 4, 16 * cs, stripH + 8)) bb = V.clamp(Math.floor((P.x - ax) / (4 * cs)), 0, 3);
        const k0 = bb * 16;
        const Aval = (i, k) => (fp4 ? A4[i][k0 + k] : A16[i][k]);
        const Bval = (j, k) => (fp4 ? B4[k0 + k][j] : B16[k][j]);
        const mA = fp4 ? 6 : 3, mB = fp4 ? 6 : 2, D = fp4 ? D4 : D16, mD = fp4 ? mD4 : mD16;
        let ci, cj;
        if (inR(dx, ay, 8 * cs, 16 * cs)) { cj = Math.floor((P.x - dx) / cs); ci = Math.floor((P.y - ay) / cs); }
        else { const k = Math.floor(t / 0.9) % 128; ci = Math.floor(k / 8); cj = k % 8; }
        const useLane = S.lanes && !fp4;
        let own = laneD(ci, cj);
        if (useLane && inR(ax, ay, 16 * cs, 16 * cs)) own = laneA(Math.floor((P.y - ay) / cs), Math.floor((P.x - ax) / cs));
        else if (useLane && inR(ax, by, 16 * cs, 8 * cs)) own = laneB(Math.floor((P.x - ax) / cs), Math.floor((P.y - by) / cs));
        const gp = cs >= 8 ? 1 : 0.5;
        const cell = (x, y, col) => { g.fillStyle = col; V.rr(g, x + gp, y + gp, cs - 2 * gp, cs - 2 * gp, Math.min(3, cs / 4)); g.fill(); };
        const ring = (x, y) => { g.strokeStyle = C.ink; g.lineWidth = 1.6; V.rr(g, x + 0.5, y + 0.5, cs - 1, cs - 1, Math.min(3, cs / 4)); g.stroke(); };
        for (let i = 0; i < 16; i++) for (let k = 0; k < 16; k++) { const ln = laneA(i, k); cell(ax + k * cs, ay + i * cs, useLane ? laneCol(ln, 0.9) : V.cmap(Aval(i, k) / mA, 'div')); if (useLane && ln === own) ring(ax + k * cs, ay + i * cs); }
        for (let j = 0; j < 8; j++) for (let k = 0; k < 16; k++) { const ln = laneB(k, j); cell(ax + k * cs, by + j * cs, useLane ? laneCol(ln, 0.9) : V.cmap(Bval(j, k) / mB, 'div')); if (useLane && ln === own) ring(ax + k * cs, by + j * cs); }
        for (let i = 0; i < 16; i++) for (let j = 0; j < 8; j++) { const ln = laneD(i, j); cell(dx + j * cs, ay + i * cs, useLane ? laneCol(ln, 0.9) : V.cmap(D[i][j] / mD, 'div')); if (useLane && ln === own) ring(dx + j * cs, ay + i * cs); }
        const bcol = C.series[bb];
        if (fp4) {
          // K-axis ruler: 4 blocks of 16, the block on stage highlighted
          T(g, `K = 64 = 4 blocks of 16 · on stage: block ${bb} (k = ${k0}…${k0 + 15})`, pad, 16, { size: 11.5, weight: 600, maxW: w - 2 * pad });
          for (let b = 0; b < 4; b++) {
            const x = ax + b * 4 * cs;
            V.box(g, x + 1, stripY, 4 * cs - 2, stripH, { r: 4, fill: V.alpha(C.series[b], b === bb ? 0.85 : 0.2), stroke: V.alpha(C.series[b], 0.8), lw: 1 });
            T(g, 'b' + b, x + 2 * cs, stripY + stripH / 2 + 0.5, { size: 11, align: 'center', base: 'middle', font: 'mono', color: b === bb ? onTxt(C) : C.muted });
          }
          const sx0 = ax + 16 * cs + 8;
          if (w - pad - sx0 > 60) T(g, '← hover here', sx0, stripY + stripH / 2 + 4, { size: 11, color: C.muted, maxW: w - pad - sx0 });
          // frames tying A and Bᵀ to the ruler's block
          g.strokeStyle = bcol; g.lineWidth = 1.6;
          V.rr(g, ax - 2, ay - 2, 16 * cs + 4, 16 * cs + 4, 5); g.stroke();
          V.rr(g, ax - 2, by - 2, 16 * cs + 4, 8 * cs + 4, 5); g.stroke();
          // scales: sA (16×4) and sBᵀ (8×4), with the column of the block on stage highlighted
          const sx = ax - 5 * cs;
          for (let i = 0; i < 16; i++) for (let b = 0; b < 4; b++) cell(sx + b * cs, ay + i * cs, V.alpha(C.series[b], 0.25 + 0.5 * (sA[i][b] / 2)));
          for (let j = 0; j < 8; j++) for (let b = 0; b < 4; b++) cell(sx + b * cs, by + j * cs, V.alpha(C.series[b], 0.25 + 0.5 * (sB[b][j] / 2)));
          g.strokeStyle = bcol; g.lineWidth = 1.4;
          V.rr(g, sx + bb * cs, ay - 1, cs, 16 * cs + 2, 3); g.stroke();
          V.rr(g, sx + bb * cs, by - 1, cs, 8 * cs + 2, 3); g.stroke();
          T(g, 'sA', sx + 2 * cs, ay - 6, { size: 11, align: 'center', font: 'mono', color: C.muted });
          T(g, 'sBᵀ', sx + 2 * cs, by - 5, { size: 11, align: 'center', font: 'mono', color: C.muted });
        }
        const hl = (x, y, ww, hh, col) => { g.strokeStyle = col; g.lineWidth = 2; V.rr(g, x - 1.5, y - 1.5, ww + 3, hh + 3, 4); g.stroke(); };
        if (!useLane) { hl(ax, ay + ci * cs, 16 * cs, cs, C.c); hl(ax, by + cj * cs, 16 * cs, cs, C.c); }
        hl(dx + cj * cs, ay + ci * cs, cs, cs, C.c);
        V.glow(g, dx + cj * cs + cs / 2, ay + ci * cs + cs / 2, cs * 1.6, C.c, 0.5);
        T(g, fp4 ? `A · block ${bb}` : 'A (16×16)', ax, ay - 6, { size: 11, font: 'mono', color: fp4 ? bcol : C.muted });
        T(g, 'D (16×8)', Math.min(dx, w - pad - 58), ay - 6, { size: 11, font: 'mono', color: C.muted });
        T(g, fp4 ? `Bᵀ · block ${bb}` : 'Bᵀ (8×16)', ax, by + 8 * cs + 15, { size: 11, font: 'mono', color: fp4 ? bcol : C.muted });
        T(g, '=', ax + 16 * cs + 0.8 * cs, ay + 8 * cs + 4, { size: 14, align: 'center', color: C.muted, weight: 600 });
        // explanation: side panel (desktop) or list below the matrices (narrow)
        const items = Lo.items;
        const drawItems = (x0, y0, ww) => { let yy = y0; items.forEach((it, k) => { const n = wrap(g, it, x0, yy, ww, 15, Object.assign({ color: k === 0 ? C.a : C.ink, maxLines: 3 }, itemStyle(k))); yy += n * 15 + 7; }); };
        if (Lo.side) {
          const pH = Math.min(useLane ? 16 * cs : by + 8 * cs - ay, itemsH(items, Lo.pW - 24) + 26);
          V.box(g, Lo.pX, ay, Lo.pW, pH, { r: 10, fill: V.alpha(C.a, 0.05), stroke: V.alpha(C.a, 0.35), lw: 1 });
          drawItems(Lo.pX + 12, ay + 22, Lo.pW - 24);
        } else {
          V.box(g, pad, Lo.listY, w - 2 * pad, Lo.listH - 6, { r: 10, fill: V.alpha(C.a, 0.05), stroke: V.alpha(C.a, 0.3), lw: 1 });
          drawItems(pad + 10, Lo.listY + 18, w - 2 * pad - 20);
        }
        if (useLane) {
          // lane legend: hue = lane % 4, shade = lane ÷ 4; outline = everything lane "own" holds
          const lx = dx, lw2 = w - pad - dx;
          let yy = by + 10;
          T(g, 'hue = lane % 4', lx, yy, { size: 11, color: C.muted, maxW: lw2 }); yy += 6;
          for (let q = 0; q < 4; q++) { g.fillStyle = laneCol(q + 12, 0.95); V.rr(g, lx + q * 18, yy, 14, 12, 3); g.fill(); T(g, String(q), lx + q * 18 + 7, yy + 6.5, { size: 11, align: 'center', base: 'middle', font: 'mono', color: onTxt(C) }); }
          yy += 26;
          T(g, 'shade = lane ÷ 4', lx, yy, { size: 11, color: C.muted, maxW: lw2 }); yy += 6;
          const sw = Math.min(12, (lw2 - 2) / 8);
          for (let q = 0; q < 8; q++) { g.fillStyle = laneCol(q * 4, 0.95); V.rr(g, lx + q * sw, yy, sw - 2, 10, 2); g.fill(); }
          yy += 26;
          if (lw2 < 200) { T(g, `lane ${own} outlined`, lx, yy, { size: 11, color: C.ink, maxW: lw2 }); T(g, `group ${own >> 2} · position ${own & 3}`, lx, yy + 14, { size: 11, color: C.muted, maxW: lw2 }); }
          else wrap(g, `outline: everything lane ${own} holds (group ${own >> 2}, position ${own & 3})`, lx, yy, lw2, 14, { size: 11, color: C.ink, maxLines: 3 });
        }
        // magnifier
        const ly0 = Lo.lensTop, lensH = Lo.lensH, lx = Lo.lx, lc = Lo.lc, ch2 = Lo.ch2;
        V.box(g, pad, ly0, w - 2 * pad, lensH - 6, { r: 10, fill: V.alpha(C.c, 0.05), stroke: V.alpha(C.c, 0.4), lw: 1 });
        if (!fp4) {
          T(g, `magnifier: D[${ci}][${cj}] = A[${ci},·] · Bᵀ[${cj},·]`, pad + 10, ly0 + 18, { size: 12, weight: 600, maxW: w - 2 * pad - 20 });
          const rowsL = [['A', (k) => A16[ci][k]], ['Bᵀ', (k) => B16[k][cj]], ['×', (k) => A16[ci][k] * B16[k][cj]]];
          rowsL.forEach(([lab, f], ri) => {
            const yy = ly0 + 30 + ri * (ch2 + 6);
            T(g, lab, pad + 12, yy + ch2 / 2 + 4, { size: 11.5, font: 'mono', color: C.muted });
            for (let k = 0; k < 16; k++) {
              const v = f(k), mm = ri === 2 ? 9 : 3, tt = Math.abs(v) / mm;
              g.fillStyle = V.cmap(v / mm, 'div'); V.rr(g, lx + k * lc + 1, yy, lc - 2, ch2, 3); g.fill();
              if (lc >= 15) T(g, String(v), lx + k * lc + lc / 2, yy + ch2 / 2 + 0.5, { size: 11, font: 'mono', align: 'center', base: 'middle', color: cellTxt(C, tt > 0.55) });
            }
          });
          T(g, `Σ of the 16 products = ${D16[ci][cj]}  →  D[${ci}][${cj}] (accumulated in FP32)`, pad + 12, ly0 + 30 + 3 * (ch2 + 6) + 12, { size: 12, weight: 600, color: C.c, maxW: w - 2 * pad - 24 });
        } else {
          T(g, `magnifier: D[${ci}][${cj}] = Σ_b sA·sB·(A_b · Bᵀ_b), block ${bb} on stage`, pad + 10, ly0 + 18, { size: 12, weight: 600, maxW: w - 2 * pad - 20 });
          const per = Lo.per, nL = 16 / per;
          let yy = ly0 + 28;
          for (let hf = 0; hf < nL; hf++) {
            [['A', (k) => A4[ci][k0 + k]], ['Bᵀ', (k) => B4[k0 + k][cj]]].forEach(([lab, f]) => {
              const lab2 = nL > 1 ? `${lab} k${hf * 8}–${hf * 8 + 7}` : `${lab} b${bb}`;
              T(g, lab2, pad + 10, yy + ch2 / 2 + 4, { size: 11, font: 'mono', color: bcol, maxW: lx - pad - 14 });
              for (let q = 0; q < per; q++) {
                const k = hf * per + q, v = f(k), tt = Math.abs(v) / 6;
                g.fillStyle = V.cmap(v / 6, 'div'); V.rr(g, lx + q * lc + 1, yy, lc - 2, ch2, 3); g.fill();
                if (lc >= 24) T(g, nf(v), lx + q * lc + lc / 2, yy + ch2 / 2 + 0.5, { size: 11, font: 'mono', align: 'center', base: 'middle', color: cellTxt(C, tt > 0.55) });
              }
              yy += ch2 + 5;
            });
          }
          const cy0 = yy + 6, perRow = Lo.chipPer, chipW = (w - 2 * pad - 20 - (perRow - 1) * 6) / perRow, chipH = 22;
          for (let b = 0; b < 4; b++) {
            const p = part4(ci, cj, b), c = sA[ci][b] * sB[b][cj] * p;
            const x0 = pad + 10 + (b % perRow) * (chipW + 6), y0 = cy0 + Math.floor(b / perRow) * (chipH + 5);
            V.box(g, x0, y0, chipW, chipH, { r: 5, fill: V.alpha(C.series[b], b === bb ? 0.3 : 0.12), stroke: V.alpha(C.series[b], b === bb ? 1 : 0.6), lw: b === bb ? 1.6 : 1 });
            T(g, narrow ? `b${b}: ${nf(p)}×${nf(sA[ci][b] * sB[b][cj])} = ${nf(c)}` : `b${b}: ${nf(p)} × ${nf(sA[ci][b])} × ${nf(sB[b][cj])} = ${nf(c)}`, x0 + 6, y0 + chipH / 2 + 0.5, { size: 11, font: 'mono', base: 'middle', maxW: chipW - 10 });
          }
          T(g, `D[${ci}][${cj}] = Σ_b sA·sB·(Σ of the block) = ${nf(D4[ci][cj])}`, pad + 10, ly0 + lensH - 18, { size: 12, weight: 600, color: C.c, maxW: w - 2 * pad - 20 });
        }
      });
    }

    /* =================== Figure 5: capability gates =================== */
    {
      const fig = $('#fig-gates');
      const GPUS = [
        { v: 'a100', label: 'A100 · 8.0', cc: [8, 0] }, { v: 'h100', label: 'H100 · 9.0', cc: [9, 0] }, { v: 'b200', label: 'B200 · 10.0', cc: [10, 0] },
        { v: 'gb300', label: 'GB300 · 10.3', cc: [10, 3] }, { v: 'rtx', label: 'RTX PRO 6000 · 12.0', cc: [12, 0] }, { v: 'gb10', label: 'GB10 · 12.1', cc: [12, 1] },
      ];
      const S = { gpu: 'gb10', patch: false, t0: -100, changed: false };
      const eq = (c, a, b) => c[0] === a && c[1] === b;
      const val = (c) => c[0] * 10 + c[1];
      const G = [
        { rc: 'glm', n: 'EXL3 accepts the GPU', ex: () => 'get_min_capability() = 80', ev: (c) => (val(c) >= 80 ? ['ok', 'exl3 method available'] : ['err', 'rejected']) },
        { rc: 'glm', n: 'E3 tier (16 B atomics)', ex: () => 'cc >= (9, 0)  # red.v4', ev: (c) => (val(c) >= 90 ? ['ok', 'E3: atomicAdd(float4)'] : ['fb', 'falls back to E2']) },
        { rc: 'glm', n: 'PDL (dependent launch)', ex: (p) => (p ? 'major in (9, 10)' : 'major >= 9'), ev: (c, p) => (c[0] === 9 || c[0] === 10 ? ['ok', 'PDL on'] : c[0] >= 9 ? (p ? ['fb', 'off: avoids the race'] : ['warn', 'on, but races the KDA state kernels']) : ['fb', 'off']) },
        { rc: 'qwen', n: 'QSA sm_120 table (v0.30)', ex: () => '_is_sm120(): cc == (12, 0)', ev: (c) => (eq(c, 12, 0) ? ['ok', 'table tuned on sm_120'] : ['fb', 'default table (GB300)' + (eq(c, 12, 1) ? ' · inferred from the code' : '')]) },
        { rc: 'qwen', n: 'GDN prefill on FlashInfer', ex: () => 'SM90 or SM10x', ev: (c) => (c[0] === 9 || c[0] === 10 ? ['ok', 'FlashInfer/CuteDSL'] : ['fb', 'Triton FLA chunk_gated_delta_rule']) },
        { rc: 'qwen', n: 'NVFP4 MoE: backend', ex: () => 'TRTLLM → CuteDSL (SM10x) → CUTLASS', ev: (c) => (c[0] === 10 ? ['ok', 'TRTLLM (1st choice)'] : c[0] === 12 ? ['ok', 'CUTLASS SM120 (3rd choice)'] : ['fb', 'no native FP4']) },
        { rc: 'qwen', n: 'CuteDSL skinny GEMM (decode)', ex: () => '_is_sm103()', ev: (c) => (eq(c, 10, 3) ? ['ok', 'skinny kernel (TP=4 plans)'] : ['fb', 'cuBLAS']) },
        { rc: 'ds', n: 'Indexer MXFP4 K cache', ex: (p) => (p ? 'family(100) or family(120)' : 'family(100)'), ev: (c, p) => (c[0] === 10 || (p && c[0] === 12) ? ['ok', 'FP4 cache: half the bytes'] : ['fb', 'rejected: “requires Blackwell datacenter”']) },
        { rc: 'ds', n: 'DeepGEMM: mqa_logits header', ex: (p) => (p ? 'sm121_X → #define sm120_X' : 'impls/sm{XY}_fp8_mqa_logits.cuh'), ev: (c, p) => (eq(c, 12, 0) ? ['ok', 'sm120_* header exists'] : eq(c, 12, 1) ? (p ? ['ok', 'alias to sm120_*'] : ['err', 'Failed to open …/sm121_… (cache miss)']) : ['na', 'outside this example']) },
        { rc: 'ds', n: 'gb10_hybrid_nvfp4 plugin', ex: () => 'cc == 121 or ALLOW_NON_GB10', ev: (c) => (eq(c, 12, 1) ? ['warn', 'gate passes (opt-in, not tested live)'] : ['err', 'fails fast (ALLOW_NON_GB10=0)']) },
        { rc: 'ds', n: 'MegaMoE (DeepGEMM fp8_fp4)', ex: () => 'SM100 + EP', ev: (c) => (eq(c, 10, 0) ? ['ok', 'available (requires EP)'] : eq(c, 10, 3) ? ['na', 'not verified'] : ['fb', 'does not run: b12x MoE']) },
        { rc: 'ds', n: 'DeepSeek-V4 sparse MLA', ex: () => 'Hopper/SM10x → FlashMLA · SM120 → FlashInfer', ev: (c) => (c[0] === 9 || c[0] === 10 ? ['ok', 'FlashMLA sparse'] : c[0] === 12 ? ['ok', 'FlashInfer SM120 kernel (lanes A and B)'] : ['na', 'outside this example']) },
      ];
      const ctl = UI.controls(fig);
      UI.select(ctl, { label: 'GPU', options: GPUS.map((x) => ({ v: x.v, label: x.label })), value: 'gb10', onChange: (v) => { S.gpu = v; S.changed = true; } });
      UI.toggle(ctl, { label: 'apply the recipes’ patches', value: false, onChange: (v) => { S.patch = v; S.changed = true; } });
      fitH(fig, (w) => (w < 620 ? 60 + G.length * 58 + 56 : 70 + G.length * 36 + 30));
      V.fig(fig, (g, w, h, t) => {
        const C = V.C(), gp = GPUS.find((x) => x.v === S.gpu), cc = gp.cc, narrow = w < 620, pad = 12;
        if (S.changed) { S.t0 = t; S.changed = false; }
        const scan = (t - S.t0) * 24;
        T(g, `${gp.label.split(' · ')[0]} · compute capability ${cc[0]}.${cc[1]} (sm_${cc[0]}${cc[1]})`, pad, 20, { size: 13.5, weight: 600, maxW: w - 2 * pad });
        T(g, S.patch ? 'with the recipes’ patches' : 'stock code', pad, 38, { size: 11.5, color: S.patch ? C.a : C.muted });
        const col = { ok: C.a, fb: C.c, err: C.red || C.c, warn: C.d, na: C.muted };
        const rcCol = { glm: C.glm, qwen: C.qwen, ds: C.ds };
        const rcLab = { glm: 'GLM', qwen: 'Qwen', ds: 'DS' };
        const rowH = narrow ? 58 : 36, y0 = 52;
        let cnt = { ok: 0, fb: 0, err: 0, warn: 0, na: 0 };
        G.forEach((gt, i) => {
          const y = y0 + i * rowH;
          const [s, res] = gt.ev(cc, S.patch);
          const seen = scan > i;
          if (seen) cnt[s]++;
          g.fillStyle = V.alpha(C.muted, i % 2 ? 0.04 : 0.08); V.rr(g, pad - 4, y, w - 2 * pad + 8, rowH - 4, 6); g.fill();
          V.box(g, pad, y + 7, 38, 18, { r: 5, fill: V.alpha(rcCol[gt.rc], 0.2), stroke: rcCol[gt.rc], lw: 1, label: rcLab[gt.rc], size: 11 });
          const lr = 8, lx = narrow ? w - pad - 12 : w * 0.6, ly = y + 16;
          const c = seen ? col[s] : V.alpha(C.muted, 0.3);
          if (seen && s !== 'na') V.glow(g, lx, ly, 16, c, 0.35 + 0.25 * Math.sin(t * 3 + i));
          g.beginPath(); g.arc(lx, ly, lr, 0, Math.PI * 2); g.fillStyle = c; g.fill();
          if (!narrow) {
            T(g, gt.n, pad + 46, y + 14, { size: 12, weight: 600, maxW: w * 0.3 - 50 });
            T(g, gt.ex(S.patch), pad + 46, y + 28, { size: 11, font: 'mono', color: C.muted, maxW: lx - pad - 60 });
            T(g, seen ? res : '…', lx + 16, y + 21, { size: 12, color: seen ? (s === 'na' ? C.muted : C.ink) : C.muted, maxW: w - lx - 16 - pad });
          } else {
            T(g, gt.n, pad + 46, y + 20, { size: 12, weight: 600, maxW: w - pad - 46 - 30 });
            T(g, gt.ex(S.patch), pad + 4, y + 36, { size: 11, font: 'mono', color: C.muted, maxW: w - 2 * pad - 8 });
            T(g, seen ? '→ ' + res : '…', pad + 4, y + 50, { size: 11.5, color: seen ? (s === 'na' ? C.muted : C.ink) : C.muted, maxW: w - 2 * pad - 8 });
          }
        });
        const yl = y0 + G.length * rowH + 16;
        const leg = [['ok', 'native'], ['fb', 'fallback'], ['err', 'fails'], ['warn', 'risk'], ['na', 'n/a']];
        let lxx = pad, lyy = yl;
        leg.forEach(([k, lab]) => {
          const txt = `${lab} ${cnt[k]}`; g.font = V.font(12, 'sans', 600); const tw = g.measureText(txt).width + 30;
          if (lxx + tw > w - pad) { lxx = pad; lyy += 20; }
          g.beginPath(); g.arc(lxx + 6, lyy - 4, 6, 0, Math.PI * 2); g.fillStyle = col[k]; g.fill();
          T(g, txt, lxx + 17, lyy, { size: 12, weight: 600, color: C.ink }); lxx += tw;
        });
      });
    }

    /* =================== Figure 6: the fatbin loader =================== */
    {
      const fig = $('#fig-fatbin');
      const GPUS = [
        { v: 'a100', label: 'A100 · 8.0', cc: [8, 0] }, { v: 'h100', label: 'H100 · 9.0', cc: [9, 0] }, { v: 'b200', label: 'B200 · 10.0', cc: [10, 0] },
        { v: 'gb300', label: 'GB300 · 10.3', cc: [10, 3] }, { v: 'rtx', label: 'RTX PRO 6000 · 12.0', cc: [12, 0] }, { v: 'gb10', label: 'GB10 · 12.1', cc: [12, 1] },
      ];
      const PRE = {
        gen: { label: 'generic', e: [{ k: 'sass', a: [8, 0], s: '' }, { k: 'ptx', a: [8, 0], s: '' }] },
        dc: { label: 'datacenter', e: [{ k: 'sass', a: [9, 0], s: 'a' }, { k: 'sass', a: [10, 0], s: 'a' }] },
        qwen: { label: 'sm_120 (Qwen)', e: [{ k: 'sass', a: [12, 0], s: '' }] },
        rec: { label: '12.1a (GLM/DS)', e: [{ k: 'sass', a: [12, 1], s: 'a' }] },
        fam: { label: '12.0f (family)', e: [{ k: 'sass', a: [12, 0], s: 'f' }] },
      };
      const S = { gpu: 'gb10', pre: 'rec', t0: 0, changed: true };
      const name = (e) => (e.k === 'sass' ? 'SASS sm_' : 'PTX compute_') + e.a[0] + e.a[1] + e.s;
      function check(e, cc) {
        const same = cc[0] === e.a[0], ge = cc[1] >= e.a[1], exact = same && cc[1] === e.a[1];
        if (e.k === 'sass') {
          if (e.s === 'a') return exact ? [true, 'exact: runs'] : [false, 'a suffix: only on CC ' + e.a[0] + '.' + e.a[1]];
          if (same && ge) return [true, e.s === 'f' ? 'same family: runs' : 'same major, minor ≥: runs'];
          return [false, same ? 'lower minor' : 'other major: won’t run'];
        }
        if (e.s === 'a') return exact ? [true, 'JIT (exact)'] : [false, 'PTX a: exact CC only'];
        if (e.s === 'f') return same && ge ? [true, 'JIT within the family'] : [false, 'outside the family'];
        return cc[0] * 10 + cc[1] >= e.a[0] * 10 + e.a[1] ? [true, 'JIT: CC ≥ ' + e.a[0] + '.' + e.a[1]] : [false, 'CC lower than the PTX'];
      }
      function resolve(entries, cc) {
        const res = entries.map((e) => check(e, cc));
        const si = entries.findIndex((e, i) => e.k === 'sass' && res[i][0]);
        if (si >= 0) return { res, pick: si, kind: 'sass' };
        const pi = entries.findIndex((e, i) => e.k === 'ptx' && res[i][0]);
        if (pi >= 0) return { res, pick: pi, kind: 'jit' };
        return { res, pick: -1, kind: 'err' };
      }
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Fatbin', options: Object.keys(PRE).map((k) => ({ v: k, label: PRE[k].label })), value: 'rec', onChange: (v) => { S.pre = v; S.changed = true; } });
      UI.select(ctl, { label: 'GPU', options: GPUS.map((x) => ({ v: x.v, label: x.label })), value: 'gb10', onChange: (v) => { S.gpu = v; S.changed = true; } });
      fitH(fig, (w) => (w < 560 ? 360 : 290));
      const famName = (cc) => (cc[0] === 12 ? '12.x family · consumer' : cc[0] === 10 ? '10.x family · datacenter' : cc[0] === 9 ? 'Hopper' : 'Ampere');
      V.fig(fig, (g, w, h, t) => {
        const C = V.C(), narrow = w < 560, pad = 12;
        if (S.changed) { S.t0 = t; S.changed = false; }
        const gp = GPUS.find((x) => x.v === S.gpu), E = PRE[S.pre].e, R = resolve(E, gp.cc);
        const per = 0.5, el = t - S.t0, cyc = E.length * per + 4.5, lt = el % cyc, step = Math.floor(lt / per), fin = lt >= E.length * per;
        const eh = narrow ? 28 : 32;
        const fb = narrow ? { x: pad, y: 10, w: w - 2 * pad, h: 40 + E.length * (eh + 6) } : { x: pad, y: 12, w: Math.min(360, w * 0.46), h: 40 + E.length * (eh + 6) };
        V.box(g, fb.x, fb.y, fb.w, fb.h, { r: 10, fill: V.alpha(C.b, 0.08), stroke: V.alpha(C.b, 0.7) });
        T(g, 'fatbin (.so): ' + PRE[S.pre].label, fb.x + 12, fb.y + 20, { size: 12.5, weight: 600, maxW: fb.w - 24 });
        E.forEach((e, i) => {
          const y = fb.y + 32 + i * (eh + 6), done = step > i || fin, active = step === i && !fin;
          const [ok, why] = R.res[i], chosen = done && R.pick === i;
          V.box(g, fb.x + 10, y, fb.w - 20, eh, { r: 6, fill: chosen ? V.alpha(C.a, 0.28) : V.alpha(e.k === 'sass' ? C.a : C.d, 0.1), stroke: active ? C.c : chosen ? C.a : V.alpha(C.muted, 0.4), lw: active || chosen ? 2 : 1 });
          T(g, name(e), fb.x + 20, y + eh / 2 + 4, { size: 12, font: 'mono', weight: 600 });
          if (done) T(g, (ok ? '✓ ' : '✗ ') + why, fb.x + fb.w - 18, y + eh / 2 + 4, { size: 11, align: 'right', color: ok ? C.a : C.c, maxW: fb.w * 0.5 });
          if (active) V.glow(g, fb.x + fb.w - 22, y + eh / 2, 14, C.c, 0.7);
        });
        const gbw = narrow ? 180 : Math.min(230, w * 0.3);
        const gb = narrow ? { x: w - pad - gbw, y: fb.y + fb.h + 12, w: gbw, h: 66 } : { x: w - pad - gbw, y: 12, w: gbw, h: Math.max(fb.h, 96) };
        V.box(g, gb.x, gb.y, gb.w, gb.h, { r: 12, fill: V.alpha(C.a, 0.08), stroke: C.a });
        T(g, gp.label.split(' · ')[0], gb.x + gb.w / 2, gb.y + 24, { size: 14, weight: 700, align: 'center', maxW: gb.w - 10 });
        T(g, `CC ${gp.cc[0]}.${gp.cc[1]} · sm_${gp.cc[0]}${gp.cc[1]}`, gb.x + gb.w / 2, gb.y + 42, { size: 12, font: 'mono', align: 'center', color: C.muted, maxW: gb.w - 10 });
        T(g, famName(gp.cc), gb.x + gb.w / 2, gb.y + 58, { size: 11, align: 'center', color: C.muted, maxW: gb.w - 10 });
        if (!narrow) {
          const rx = fb.x + fb.w + 14, ex = gb.x - 10, ry = fb.y + Math.min(fb.h, gb.h) / 2, u = (t * 0.8) % 1;
          V.arrow(g, rx, ry, ex, ry, { color: V.alpha(C.muted, 0.5), lw: 1.5 });
          V.dot(g, V.lerp(rx, ex, u), ry, 3, C.c);
          T(g, 'driver loads the kernel', (rx + ex) / 2, ry - 10, { size: 11, align: 'center', color: C.muted, maxW: ex - rx });
        } else {
          wrap(g, 'the driver loads the kernel on this GPU →', pad, gb.y + 26, gb.x - pad - 12, 15, { size: 11.5, color: C.muted, maxLines: 3 });
        }
        const rbY = Math.max(fb.y + fb.h, gb.y + gb.h) + 14, rbH = narrow ? 104 : 74;
        let msg = '', c = C.muted;
        {
          const e = E[R.pick];
          if (R.kind === 'sass') { msg = `loads ${name(e)}: native code, nothing to compile. ` + (e.s ? 'Specific features (' + e.s + ') available.' : 'No suffix: only instructions common to CC ' + e.a[0] + '.' + e.a[1] + ' and later.'); c = C.a; }
          else if (R.kind === 'jit') { msg = `JIT-compiles ${name(e)} → sm_${gp.cc[0]}${gp.cc[1]}: runs, but the 1st use is slow and only version ${e.a[0]}.${e.a[1]} instructions are available (no a/f).`; c = C.d; }
          else { msg = 'error: “no kernel image is available for execution on the device”. Recompile for this target (TORCH_CUDA_ARCH_LIST) or include portable PTX.'; c = C.c; }
        }
        V.box(g, pad, rbY, w - 2 * pad, rbH, { r: 10, fill: V.alpha(c, 0.1), stroke: V.alpha(c, 0.6) });
        wrap(g, msg, pad + 12, rbY + 22, w - 2 * pad - 24, 17, { size: 12.5, weight: 600, color: C.ink, alpha: fin ? 1 : 0.55, maxLines: narrow ? 5 : 3 });
        wrap(g, 'Rules: sm_XY runs on X.Z with Z ≥ Y · sm_XYa only on X.Y · sm_XYf on the X family from Y up · PTX compute_XY compiles for any CC ≥ X.Y', pad, rbY + rbH + 18, w - 2 * pad, 15, { size: 11, color: C.muted, font: 'mono', maxLines: 4 });
      });
    }

    /* ======================== Figure 7: roofline ======================== */
    {
      const fig = $('#fig-roofline');
      const Ms = [1, 2, 4, 8, 16, 28, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192];
      const S = { mi: 0, bits: 16, fp4: false, beff: 230 };
      const P16 = 125, P4 = 500, B0 = 273, KN = 4096;
      const Ival = (M, b) => (2 * M * KN * KN) / ((KN * KN * b) / 8 + 2 * M * (KN + KN));
      const ctl = UI.controls(fig);
      UI.slider(ctl, { label: 'M (rows)', min: 0, max: Ms.length - 1, step: 1, value: 0, fmt: (i) => String(Ms[i]), onInput: (v) => { S.mi = v; upd(); } });
      UI.seg(ctl, { label: 'Weights', options: [{ v: 16, label: 'BF16' }, { v: 8, label: 'FP8' }, { v: 4.5, label: 'NVFP4' }, { v: 4, label: 'EXL3 4 bpw' }], value: 16, onChange: (v) => { S.bits = v; upd(); } });
      UI.toggle(ctl, { label: 'FP4 tensor cores (native block scaling)', value: false, onChange: (v) => { S.fp4 = v; upd(); } });
      UI.slider(ctl, { label: 'Bandwidth', min: 150, max: 273, step: 1, value: 230, fmt: (v) => v + ' GB/s', onInput: (v) => { S.beff = v; upd(); } });
      const rI = UI.readout(ctl, 'intensity', '—'), rP = UI.readout(ctl, 'ceiling', '—'), rR = UI.readout(ctl, 'regime', '—');
      const peak = () => (S.fp4 && S.bits === 4.5 ? P4 : P16);
      function upd() {
        const M = Ms[S.mi], I = Ival(M, S.bits), P = peak(), perf = Math.min(P, (I * S.beff) / 1000);
        rI.set(cm(I, I < 10 ? 2 : 0) + ' FLOP/B'); rP.set(cm(perf, perf < 10 ? 2 : 1) + ' TFLOPS (' + cm((100 * perf) / P, perf / P < 0.1 ? 2 : 0) + '%)');
        rR.set(I * S.beff / 1000 < P ? 'memory-bound' : 'compute-bound');
      }
      upd();
      fitH(fig, (w) => (w < 560 ? 420 : 440));
      V.fig(fig, (g, w, h, t) => {
        const C = V.C(), narrow = w < 560;
        const L = narrow ? 44 : 60, Rr = 14, Tp = 16, Bt = 42;
        const x0 = Math.log10(0.5), x1 = Math.log10(4096), y0 = Math.log10(0.1), y1 = Math.log10(1000);
        const X = (I) => L + ((Math.log10(I) - x0) / (x1 - x0)) * (w - L - Rr);
        const Y = (p) => h - Bt - ((Math.log10(p) - y0) / (y1 - y0)) * (h - Tp - Bt);
        g.strokeStyle = C.grid; g.lineWidth = 1;
        [1, 4, 16, 64, 256, 1024, 4096].forEach((I) => { const x = X(I); g.beginPath(); g.moveTo(x, Tp); g.lineTo(x, h - Bt); g.stroke(); T(g, String(I), I === 4096 ? x + 4 : x, h - Bt + 14, { size: 11, align: I === 4096 ? 'right' : 'center', font: 'mono', color: C.muted }); });
        [0.1, 1, 10, 100, 1000].forEach((p) => { const y = Y(p); g.beginPath(); g.moveTo(L, y); g.lineTo(w - Rr, y); g.stroke(); T(g, String(p), L - 6, y + 4, { size: 11, align: 'right', font: 'mono', color: C.muted }); });
        T(g, 'arithmetic intensity I (FLOP/B, log)', (L + w - Rr) / 2, h - 8, { size: 11.5, align: 'center', color: C.muted });
        g.save(); g.translate(12, (Tp + h - Bt) / 2); g.rotate(-Math.PI / 2); T(g, 'TFLOPS (log)', 0, 0, { size: 11.5, align: 'center', color: C.muted }); g.restore();
        const roof = (B, P, col, lw, dash) => {
          const ir = (P * 1000) / B;
          g.beginPath(); g.moveTo(X(0.5), Y((0.5 * B) / 1000)); g.lineTo(X(ir), Y(P)); g.lineTo(X(4096), Y(P));
          g.strokeStyle = col; g.lineWidth = lw; g.setLineDash(dash || []); g.stroke(); g.setLineDash([]);
          return ir;
        };
        if (S.fp4) roof(B0, P4, V.alpha(C.f, 0.8), 1.6, [6, 5]);
        roof(S.beff, P16, V.alpha(C.b, 0.7), 1.5, [4, 4]);
        const ridge = roof(B0, P16, C.a, 2.4);
        V.dot(g, X(ridge), Y(P16), 4, C.a);
        T(g, `ridge ≈ ${Math.round(ridge)} FLOP/B`, narrow ? X(ridge) - 8 : X(ridge) + 6, narrow ? Y(P16) - 8 : Y(P16) + 16, { size: 11.5, color: C.a, weight: 600, align: narrow ? 'right' : 'left' });
        T(g, '~125 TFLOPS BF16', w - Rr - 4, narrow ? Y(P16) + 16 : Y(P16) - 6, { size: 11, align: 'right', color: C.a });
        if (S.fp4) { if (narrow) { T(g, '- - ~4× native FP4 ceiling (DS estimate)', L + 8, Y(P4) + 15, { size: 11, color: C.f, maxW: w - L - Rr - 16 }); } else T(g, '~4× (native FP4×FP8 / W4A4, DS estimate)', w - Rr - 4, Y(P4) - 6, { size: 11, align: 'right', color: C.f }); }
        const slx = X(2), sly = Y((2 * B0) / 1000);
        g.save(); g.translate(slx, sly); g.rotate(Math.atan2(Y((4 * B0) / 1000) - sly, X(4) - slx)); T(g, '273 GB/s × I', 4, -6, { size: 11, color: C.a }); g.restore();
        const u = (t * 0.12) % 1, ip = Math.pow(10, x0 + u * (x1 - x0)), pp = Math.min(P16, (ip * B0) / 1000);
        V.glow(g, X(ip), Y(pp), 9, C.a, 0.6);
        const mk = (I, p, lab, col, below) => { V.dot(g, X(I), Y(p), 4.5, col); wrap(g, lab, X(I) + 7, Y(p) + (below ? 16 : -10), Math.max(80, w - X(I) - 20), 13, { size: 11, color: col, maxLines: 2 }); };
        const i1 = Ival(1, 16);
        mk(i1, (i1 * B0) / 1000, 'BF16 decode GEMV, M=1', C.d, true);
        mk(112, (112 * 230) / 1000, 'GLM prefill MoE: 112 FLOP/B', C.c, false);
        const y17 = Y(17);
        g.setLineDash([2, 4]); g.strokeStyle = V.alpha(C.glm, 0.8); g.lineWidth = 1.5; g.beginPath(); g.moveTo(X(150), y17); g.lineTo(X(3000), y17); g.stroke(); g.setLineDash([]);
        T(g, narrow ? 'GLM prefill: ~17 TFLOPS/Spark' : 'measured GLM prefill: ~17 TFLOPS per Spark (~14%)', X(3000), y17 + 14, { size: 11, align: 'right', color: C.glm });
        const M = Ms[S.mi], I = Ival(M, S.bits), P = peak(), perf = Math.min(P, (I * S.beff) / 1000);
        const px = X(Math.max(0.5, Math.min(4096, I))), py = Y(Math.max(0.1, perf));
        V.glow(g, px, py, 18 + 4 * Math.sin(t * 4), C.c, 0.7);
        g.beginPath(); g.arc(px, py, 6, 0, Math.PI * 2); g.fillStyle = C.c; g.fill(); g.strokeStyle = C.panel || C.bg; g.lineWidth = 2; g.stroke();
        const lab = `M=${M}, ${S.bits === 16 ? 'BF16' : S.bits === 8 ? 'FP8' : S.bits === 4.5 ? 'NVFP4' : '4 bpw'}: I≈${cm(I, I < 10 ? 1 : 0)} → ${cm(perf, perf < 10 ? 2 : 0)} TFLOPS`;
        V.box(g, L + 8, Tp + 4, Math.min(w - L - Rr - 16, 360), 22, { r: 6, fill: V.alpha(C.c, 0.12), stroke: V.alpha(C.c, 0.6), lw: 1 });
        T(g, lab, L + 16, Tp + 19, { size: 11.5, font: 'mono', color: C.ink, maxW: Math.min(w - L - Rr - 32, 344) });
        if (S.fp4 && S.bits !== 4.5) wrap(g, 'the FP4 ceiling only applies to formats with native block-scaled MMA (NVFP4 W4A4 or MXFP4×MXFP8); with these weights the BF16 ceiling still holds', L + 8, Tp + 44, w - L - Rr - 16, 14, { size: 11, color: C.d, maxLines: 3 });
      });
    }

    /* ==================== Figure 8: the byte truck ==================== */
    {
      const fig = $('#fig-custo');
      const SCN = {
        qwen: { label: 'Qwen TP2', title: 'Qwen · TP2+EP · MTP=3 · batch 1 (per GPU)', unit: 'GiB', parts: [['lm_head ×4', 2.36], ['routed experts', 2.1], ['GDN', 1.95], ['HyperConnections', 1.19], ['attention + indexer', 0.6], ['MTP ×3', 0.5], ['router + shared + PLE', 0.4], ['GDN state', 0.1]], tau: 3.2, fixed: 2.5, fixedLab: 'report’s fixed cost: ≈106 collectives × 20–30 µs ≈ 2–3 ms', fixedOn: false, pub: 'measured: 52.1 tok/s (MTP=3, batch 1)', lesson: 'The lm_head, read 4×, is 26% of the bytes: the target of the drafter’s reduced vocabulary. Streaming alone predicts ~78 tok/s; the gap to the measured value is fixed cost (~17–20 ms), well above the report’s 2–3 ms of collectives.' },
        qwen8: { label: 'Qwen FP8-dense', title: 'Qwen · FP8-dense (analytical, per GPU)', unit: 'GiB', total: 5.7, tau: 3.2, fixed: 2.5, fixedLab: 'same fixed cost as Qwen TP2 (≈2–3 ms of collectives)', fixedOn: false, pub: 'recipe estimate: ~85–90 tok/s (never measured)', lesson: 'The dense BF16 parts are ~6.9 of the 9.2 GiB; in FP8 the total drops to ~5.7 GiB. The recipe’s 85–90 matches 2.2 tokens per step with no fixed cost; with the bonus token and Qwen TP2’s ~17–20 ms of fixed cost, it would come to ~70–75 (interpretation).' },
        ds: { label: 'DS TP2', title: 'DeepSeek · TP2 · DSpark k=6 · c=1 (per rank)', unit: 'GB', parts: [['routed experts', 12], ['attention', 2.4], ['lm_head', 1.06], ['drafter', 0.9], ['Markov w2', 0.2]], tau: 4.15, fixed: 12, fixedLab: 'report’s fixed cost: ~10–15 ms (the ≈108 collectives alone: 3–5 ms)', fixedOn: false, pub: 'published: 62–83 tok/s (official 75.5)', lesson: 'The experts are ~72% of the bytes: c=1 is ~80% weight streaming. With τ≈4.15 (≈3.15 accepted + the bonus), streaming alone gives ~56–61 tok/s; the measured range is higher because acceptance rises on code (~5.2 tokens/step at 73.3%) and the effective bandwidth beat the assumed one (the report’s interpretation).' },
        ds3: { label: 'DS TP3', title: 'DeepSeek · TP3 · k=6 (per rank)', unit: 'GB', total: 11.5, tau: 3.9, fixed: 13, fixedLab: 'report’s fixed cost: ~13+ ms (3-rank ring, heads 64→72)', fixedOn: true, pub: 'live probe: 64.3 tok/s', lesson: 'Each rank reads 1/3 less, but ~13 ms of fixed cost (3-rank ring, heads 64→72) wipes out the gain. Turn off the fixed cost to see what was expected.' },
      };
      const S = { sc: 'qwen', beff: 240, tau: 3.2, fixed: false };
      const ctl = UI.controls(fig);
      let slT = null, tgF = null;
      UI.seg(ctl, { label: 'Scenario', options: Object.keys(SCN).map((k) => ({ v: k, label: SCN[k].label })), value: 'qwen', onChange: (v) => { S.sc = v; if (slT) slT.set(SCN[v].tau); if (tgF) tgF.set(SCN[v].fixedOn); upd(); } });
      UI.slider(ctl, { label: 'Bandwidth', min: 150, max: 273, step: 1, value: 240, fmt: (v) => v + ' GB/s', onInput: (v) => { S.beff = v; upd(); } });
      slT = UI.slider(ctl, { label: 'Tokens/step', min: 1, max: 6, step: 0.05, value: 3.2, fmt: (v) => cm(v, 2), onInput: (v) => { S.tau = v; upd(); } });
      tgF = UI.toggle(ctl, { label: 'include the report’s fixed cost', value: false, onChange: (v) => { S.fixed = v; upd(); } });
      const rMs = UI.readout(ctl, 'ms/step', '—'), rSt = UI.readout(ctl, 'steps/s', '—'), rTk = UI.readout(ctl, 'tok/s (ceiling)', '—');
      let M = null;
      function calc() {
        const sc = SCN[S.sc], tot = sc.parts ? sc.parts.reduce((s, p) => s + p[1], 0) : sc.total;
        const gb = sc.unit === 'GiB' ? tot * 1.073741824 : tot;
        const tS = (gb / S.beff) * 1000, tF = S.fixed ? sc.fixed : 0, tt = tS + tF;
        return { sc, tot, gb, tS, tF, tt, steps: 1000 / tt, tok: (S.tau * 1000) / tt };
      }
      function upd() { M = calc(); rMs.set(cm(M.tt, 1)); rSt.set(cm(M.steps, 1)); rTk.set(cm(M.tok, 1)); }
      upd();
      fitH(fig, (w) => (w < 560 ? 452 : 420));
      const pr = V.rng(4), pk = Array.from({ length: 34 }, () => ({ p: pr(), r: pr() }));
      V.fig(fig, (g, w, h, t) => {
        const C = V.C(), m = M, sc = m.sc, pad = 14, narrow = w < 560;
        const parts = sc.parts || [['total (no published split)', sc.total]];
        const cols = parts.map((_, i) => C.series[i % 6]);
        T(g, sc.title, pad, 18, { size: 12.5, weight: 600, maxW: w - 2 * pad });
        const bY = 32, bH = narrow ? 56 : 62;
        const dW = narrow ? 64 : 92, sW = narrow ? 70 : 100;
        V.box(g, pad, bY, dW, bH, { r: 8, fill: V.alpha(C.d, 0.14), stroke: C.d, label: 'DRAM', size: 12, weight: 600 });
        V.box(g, w - pad - sW, bY, sW, bH, { r: 8, fill: V.alpha(C.a, 0.12), stroke: C.a, label: '48 SMs', size: 12, weight: 600 });
        const cx0 = pad + dW + 6, cx1 = w - pad - sW - 6, cyM = bY + bH / 2;
        V.line(g, cx0, cyM - 12, cx1, cyM - 12, { color: V.alpha(C.muted, 0.35), lw: 1 }); V.line(g, cx0, cyM + 12, cx1, cyM + 12, { color: V.alpha(C.muted, 0.35), lw: 1 });
        const speed = (S.beff / 240) * 0.28;
        const cum = []; let acc = 0; parts.forEach((p) => { acc += p[1]; cum.push(acc / m.tot); });
        pk.forEach((q) => {
          const u = (q.p + t * speed) % 1, x = V.lerp(cx0, cx1 - 14, u);
          let ci = cum.findIndex((c) => q.r <= c); if (ci < 0) ci = parts.length - 1;
          V.box(g, x, cyM - 8, 14, 16, { r: 3, fill: V.alpha(cols[ci], sc.parts ? 0.85 : 0.5) });
        });
        T(g, `${cm(S.beff, 0)} GB/s`, (cx0 + cx1) / 2, bY + bH + 14, { size: 11.5, align: 'center', font: 'mono', color: C.muted });
        const sy = bY + bH + 34, sh = 30, sx = pad, sw = w - 2 * pad;
        T(g, `bytes read per step: ${cm(m.tot, sc.unit === 'GiB' ? 1 : 2)} ${sc.unit}` + (sc.unit === 'GiB' ? ` (${cm(m.gb, 2)} GB)` : ''), sx, sy - 6, { size: 12, weight: 600, maxW: sw });
        let xx = sx; const legend = [];
        parts.forEach((p, i) => {
          const ww = (p[1] / m.tot) * sw;
          g.fillStyle = V.alpha(cols[i], sc.parts ? 0.8 : 0.35); V.rr(g, xx + 0.5, sy, Math.max(1, ww - 1), sh, 4); g.fill();
          if (!sc.parts) stripes(g, xx, sy, ww, sh, V.alpha(C.muted, 0.35));
          const dec = (String(p[1]).split('.')[1] || '').length, lab = `${p[0]} ${cm(p[1], p[1] < 1 ? 2 : Math.max(1, dec))}`;
          g.font = V.font(11, 'sans', sc.parts ? 500 : 600);
          const lw0 = g.measureText(lab).width;
          if (!sc.parts && lw0 < ww - 16) {
            // no published split: label in a pill with the panel background, readable over the hatching
            V.box(g, xx + ww / 2 - lw0 / 2 - 8, sy + sh / 2 - 10, lw0 + 16, 20, { r: 10, fill: C.panel || C.bg, stroke: V.alpha(cols[i], 0.7), lw: 1 });
            T(g, lab, xx + ww / 2, sy + sh / 2 + 4, { size: 11, align: 'center', color: C.ink, weight: 600 });
          } else if (sc.parts && lw0 < ww - 8) T(g, lab, xx + ww / 2, sy + sh / 2 + 4, { size: 11, align: 'center', color: onTxt(C), weight: 500 });
          else legend.push([cols[i], lab]);
          xx += ww;
        });
        let ly = sy + sh + 16, lx = sx;
        legend.forEach(([c, lab]) => {
          g.font = V.font(11, 'sans', 400); const lw2 = g.measureText(lab).width + 22;
          if (lx + lw2 > sx + sw) { lx = sx; ly += 16; }
          g.fillStyle = c; V.rr(g, lx, ly - 9, 10, 10, 2); g.fill(); T(g, lab, lx + 14, ly, { size: 11, color: C.muted }); lx += lw2;
        });
        const ty = ly + 30;
        T(g, 'step time', sx, ty - 8, { size: 12, weight: 600 });
        const tmax = Math.max(90, m.tt * 1.1), tw = sw;
        const wS = (m.tS / tmax) * tw, wF = (m.tF / tmax) * tw;
        V.box(g, sx, ty, tw, 24, { r: 5, fill: V.alpha(C.muted, 0.08) });
        V.box(g, sx, ty, Math.max(2, wS), 24, { r: 5, fill: V.alpha(C.b, 0.75) });
        if (wF > 0) { V.box(g, sx + wS, ty, Math.max(2, wF), 24, { r: 5, fill: V.alpha(C.d, 0.8) }); }
        const lp = ((t * 1000 * 0.06) % m.tt) / tmax * tw;
        V.line(g, sx + lp, ty - 3, sx + lp, ty + 27, { color: C.c, lw: 2 });
        g.font = V.font(11, 'sans', 500);
        const sLab = `streaming ${cm(m.tS, 1)} ms`;
        if (g.measureText(sLab).width < wS - 8) T(g, sLab, sx + 6, ty + 16, { size: 11, color: onTxt(C), weight: 500 });
        if (wF > 30) T(g, `fixed ${cm(m.tF, 1)}`, sx + wS + 4, ty + 16, { size: 11, color: onTxt(C), weight: 500, maxW: wF - 6 });
        T(g, S.fixed ? sc.fixedLab : 'fixed cost off: weight streaming only', sx, ty + 42, { size: 11.5, color: S.fixed ? C.d : C.muted, maxW: sw });
        const fy = ty + 66;
        T(g, `${cm(m.tt, 1)} ms/step → ${cm(m.steps, 1)} steps/s × ${cm(S.tau, 2)} tokens = ${cm(m.tok, 1)} tok/s`, sx, fy, { size: narrow ? 12 : 13.5, weight: 700, color: C.ink, maxW: sw });
        T(g, sc.pub, sx, fy + 20, { size: 12, color: C.a, weight: 600, maxW: sw });
        wrap(g, sc.lesson, sx, fy + 42, sw, 16, { size: 12, color: C.muted, maxLines: 6 });
      });
    }

    /* ================= Figure 9: streams, events and waits ================= */
    {
      const fig = $('#fig-streams');
      const S = { aux: false, dev: false, pin: false };
      function simulate(o) {
        const Lc = 0.45; let cpu = 0; const cpuB = [], ops = [];
        const free = { s0: 0, a1: 0, a2: 0, a3: 0, cp: 0 };
        const launch = (name, lane, dur, deps, col) => {
          const s0 = cpu; cpu += Lc; cpuB.push({ t0: s0, t1: cpu, kind: 'launch' });
          const dep = (deps || []).reduce((mx, d) => Math.max(mx, d.t1), 0);
          const t0 = Math.max(cpu, free[lane], dep);
          const op = { name, lane, t0, t1: t0 + dur, col, at: cpu };
          free[lane] = op.t1; ops.push(op); return op;
        };
        const waitFor = (op, label, extra) => { const s0 = cpu; cpu = Math.max(cpu, op.t1) + extra; cpuB.push({ t0: s0, t1: cpu, kind: 'wait', label }); };
        const q = launch('wqa_wkv', 's0', 3.0, null, 0);
        const a = launch('kv_score', o.aux ? 'a1' : 's0', 1.6, null, 1);
        const b = launch('idx.w', o.aux ? 'a2' : 's0', 1.2, null, 1);
        const c = launch('idx.comp', o.aux ? 'a3' : 's0', 1.6, null, 1);
        launch('attention', 's0', 3.0, [q, a, b, c], 2);
        const r = launch('router', 's0', 0.8, null, 3);
        let syncs = 0;
        if (!o.dev) { waitFor(r, '.item()', 0.3); syncs++; }
        if (o.pin) {
          const cp = launch('counts→host', 'cp', 0.9, [r], 4);
          launch('thin experts', 's0', 3.2, null, 5);
          waitFor(cp, 'copy', 0.15); syncs++;
          launch('fat experts', 's0', 2.4, null, 5);
        } else {
          const s0 = cpu, t0 = Math.max(cpu, r.t1);
          const cp = { name: 'counts→host', lane: 'cp', t0, t1: t0 + 0.9, col: 4, at: s0 };
          ops.push(cp); free.cp = cp.t1; cpu = cp.t1 + 0.1; cpuB.push({ t0: s0, t1: cpu, kind: 'wait', label: 'pageable copy' }); syncs++;
          launch('thin experts', 's0', 3.2, null, 5);
          launch('fat experts', 's0', 2.4, null, 5);
        }
        launch('output', 's0', 0.9, null, 0);
        const total = Math.max(cpu, ...ops.map((x) => x.t1));
        const s0ops = ops.filter((x) => x.lane === 's0').sort((x, y) => x.t0 - y.t0);
        const gaps = []; let busy = 0;
        s0ops.forEach((x, i) => { busy += x.t1 - x.t0; if (i && x.t0 - s0ops[i - 1].t1 > 0.05) gaps.push([s0ops[i - 1].t1, x.t0]); });
        if (s0ops.length && s0ops[0].t0 > 0.05) gaps.unshift([0, s0ops[0].t0]);
        const idle = gaps.reduce((s, gg) => s + gg[1] - gg[0], 0);
        return { ops, cpuB, total, gaps, idle, syncs, busy };
      }
      const worst = simulate({ aux: false, dev: false, pin: false }).total;
      let SIM = simulate(S);
      const ctl = UI.controls(fig);
      const rT = UI.readout(ctl, 'step', '—'), rI = UI.readout(ctl, 'GPU idle', '—'), rS = UI.readout(ctl, 'syncs', '—');
      function upd() { SIM = simulate(S); rT.set(cm(SIM.total, 1) + ' u'); rI.set(Math.round((100 * SIM.idle) / SIM.total) + '%'); rS.set(String(SIM.syncs)); }
      UI.toggle(ctl, { label: '3 auxiliary streams (DS, Stage-C overlay)', value: false, onChange: (v) => { S.aux = v; upd(); } });
      UI.toggle(ctl, { label: 'decide on device, no .item() (DS #175)', value: false, onChange: (v) => { S.dev = v; upd(); } });
      UI.toggle(ctl, { label: 'pinned D2H copy on a side stream (GLM E1)', value: false, onChange: (v) => { S.pin = v; upd(); } });
      upd();
      // short nicknames (when the name does not fit in the box) and what each auxiliary stream carries
      const SHORT = { wqa_wkv: 'Q', kv_score: 'kv', 'idx.w': 'iw', 'idx.comp': 'ic', 'attention': 'att', router: 'rt', 'thin experts': 'thin', 'fat experts': 'fat', 'output': 'out', 'counts→host': 'cp' };
      const DESC = { kv_score: 'compressor.kv_score', 'idx.w': 'indexer.weights_proj', 'idx.comp': 'indexer.compressor.kv_score' };
      const LEG = [
        [0, 'Q · out', 'fused_wqa_wkv (attention input) and layer output'],
        [1, 'kv · iw · ic', 'small, independent GEMMs of the compressor and the indexer'],
        [2, 'att', 'attention'],
        [3, 'rt', 'MoE router'],
        [5, 'thin · fat', 'thin experts and fat experts'],
        [4, 'cp', 'copy of the per-expert counts (DMA)'],
      ];
      const MCs = document.createElement('canvas').getContext('2d');
      function legendLayout(w) {
        const pad = 12, ncol = w < 560 ? 1 : 2, colW = (w - 2 * pad) / ncol, tw = colW - 22;
        const ent = LEG.map(([ci, ab, d]) => ({ ci, ls: splitLines(MCs, ab + ' = ' + d, tw, { size: 11.5 }) }));
        let hh = 0; const rows = [];
        for (let k = 0; k < ent.length; k += ncol) { const rr = ent.slice(k, k + ncol), n = Math.max(...rr.map((e) => e.ls.length)); rows.push({ rr, y: hh, n }); hh += n * 14 + 6; }
        return { pad, ncol, colW, rows, H: hh };
      }
      fitH(fig, (w) => 30 + 6 * (w < 560 ? 38 : 40) + 78 + legendLayout(w).H + 6);
      V.fig(fig, (g, w, h, t) => {
        const C = V.C(), sim = SIM, narrow = w < 560;
        const lanes = [['cpu', 'CPU'], ['s0', 'stream 0'], ['a1', 'aux 1'], ['a2', 'aux 2'], ['a3', 'aux 3'], ['cp', narrow ? 'copy' : 'copy (DMA)']];
        const LL = legendLayout(w);
        const L = narrow ? 64 : 96, Rr = 12, top = 30, lh = V.clamp((h - top - 84 - LL.H) / lanes.length, 26, 46);
        const span = worst + 1;
        const X = (x) => L + (x / span) * (w - L - Rr);
        const ph = (t * 2.2) % (span + 2.5), tp = Math.min(ph, span);
        T(g, 'time →  (illustrative units)', L, 18, { size: 11.5, color: C.muted });
        lanes.forEach(([k, lab], i) => {
          const y = top + i * lh;
          g.fillStyle = V.alpha(C.muted, i % 2 ? 0.03 : 0.07); g.fillRect(L, y, w - L - Rr, lh - 4);
          T(g, lab, L - 8, y + lh / 2 + 2, { size: 11.5, align: 'right', color: C.ink, font: 'mono', maxW: L - 10 });
        });
        const ly = (k) => top + lanes.findIndex((x) => x[0] === k) * lh;
        if (!S.aux) T(g, 'empty: turn on “3 auxiliary streams”', L + 10, ly('a2') + lh / 2 + 2, { size: 11, color: C.muted, maxW: w - L - Rr - 20 });
        const cols = [C.a, C.b, C.f, C.d, C.e, C.c];
        sim.cpuB.forEach((b) => {
          const y = ly('cpu') + 5, x0 = X(b.t0), x1 = X(b.t1);
          if (b.kind === 'launch') { V.box(g, x0, y, Math.max(1.5, x1 - x0 - 1), lh - 14, { r: 3, fill: V.alpha(C.d, b.t0 <= tp ? 0.85 : 0.3) }); }
          else { V.box(g, x0, y, Math.max(2, x1 - x0), lh - 14, { r: 3, fill: V.alpha(C.muted, b.t0 <= tp ? 0.4 : 0.15) }); if (x1 - x0 > 44) T(g, '⏳ ' + b.label, (x0 + x1) / 2, y + (lh - 14) / 2 + 4, { size: 11, align: 'center', color: C.ink, maxW: x1 - x0 - 4 }); }
        });
        sim.gaps.forEach(([a, b]) => { const y = ly('s0') + 4; stripes(g, X(a), y, X(b) - X(a), lh - 12, V.alpha(C.red || C.c, 0.55)); });
        sim.ops.forEach((op) => {
          const y = ly(op.lane) + 4, x0 = X(op.t0), x1 = X(op.t1), col = cols[op.col % 6], bw = x1 - x0;
          const prog = V.clamp((tp - op.t0) / (op.t1 - op.t0), 0, 1);
          V.box(g, x0, y, Math.max(2, bw - 1), lh - 12, { r: 4, fill: V.alpha(col, 0.18), stroke: V.alpha(col, 0.8), lw: 1 });
          if (prog > 0) V.box(g, x0, y, Math.max(1.5, (bw - 1) * prog), lh - 12, { r: 4, fill: V.alpha(col, 0.75) });
          const tc = prog > 0.5 ? onTxt(C) : C.ink, cy = y + (lh - 12) / 2 + 4;
          g.font = V.font(11, 'sans', 500);
          const full = g.measureText(op.name).width < bw - 6;
          g.font = V.font(11, 'sans', 600);
          const sh = SHORT[op.name] || '', shortOk = sh && g.measureText(sh).width < bw - 3;
          if (full) T(g, op.name, (x0 + x1) / 2, cy, { size: 11, align: 'center', color: tc, weight: 500 });
          else if (shortOk) T(g, sh, (x0 + x1) / 2, cy, { size: 11, align: 'center', color: tc, weight: 600 });
          // on the auxiliary streams, the full GEMM name goes next to the box
          if (op.lane[0] === 'a' && DESC[op.name]) T(g, DESC[op.name], x1 + 6, cy, { size: 11, font: 'mono', color: C.muted, maxW: Math.max(20, w - Rr - x1 - 10) });
        });
        const xp = X(tp); V.line(g, xp, top - 4, xp, top + lanes.length * lh, { color: C.c, lw: 1.6 });
        const depth = sim.ops.filter((o) => o.at <= tp && o.t1 > tp).length;
        V.line(g, X(sim.total), top - 4, X(sim.total), top + lanes.length * lh, { color: V.alpha(C.a, 0.8), lw: 1.5, dash: [4, 3] });
        const by = top + lanes.length * lh + 18;
        T(g, `GPU queue: ${depth} pending launch${depth === 1 ? '' : 'es'}`, L, by, { size: 12, weight: 600, color: depth ? C.a : C.c, maxW: w - L - Rr });
        wrap(g, `step = ${cm(sim.total, 1)} u (worst case ${cm(worst, 1)} u) · ${sim.syncs} CPU wait${sim.syncs === 1 ? '' : 's'} · red = GPU stopped waiting for the CPU`, L, by + 18, w - L - Rr, 15, { size: 11.5, color: C.muted, maxLines: 2 });
        // color legend: what each box (and each nickname) stands for
        const ly0 = by + 58;
        g.strokeStyle = V.alpha(C.muted, 0.25); g.lineWidth = 1; g.beginPath(); g.moveTo(LL.pad, ly0 - 14); g.lineTo(w - LL.pad, ly0 - 14); g.stroke();
        LL.rows.forEach((row) => {
          row.rr.forEach((e, c) => {
            const x = LL.pad + c * LL.colW, y = ly0 + row.y;
            g.fillStyle = V.alpha(cols[e.ci], 0.8); V.rr(g, x, y - 9, 12, 12, 3); g.fill();
            e.ls.forEach((ln, k) => T(g, ln, x + 18, y + k * 14, { size: 11.5, color: C.ink }));
          });
        });
      });
    }

    /* ============ Figure 9b: L2 prefetch on a side stream (GLM·TF) ============ */
    {
      const fig = $('#fig-l2pf');
      const tfc = (C) => C.__tf || (C.__tf = (getComputedStyle(document.documentElement).getPropertyValue('--tf') || '').trim() || C.d);
      const R = 4;        // MiB per time unit: illustrative bandwidth
      const L2C = 0.6;    // fraction of the DRAM time that a byte coming from the L2 still costs (illustrative)
      const HEAD = 3;     // start of each matmul drawn to scale; the rest is compressed
      const COMP = 0.12;  // visual compression of the rest of the long matmuls
      const OVC = 0.5;    // fraction of the excess's DRAM time the main stream loses to the contention (illustrative)
      const S = { on: true, shFirst: true, mb: 8 };
      // phases of the stretch: mib = DRAM-bound matmul; idle = DRAM idle (illustrative duration, prefetch point)
      const PH = {
        in: { lab: 'input', ab: 'in', mib: 32 },
        core: { lab: 'core', ab: 'core', idle: 2.0, site: 'o' },
        o: { lab: 'output', ab: 'out', mib: 12 },
        ag1: { lab: 'AG + glue', ab: 'AG', idle: 2.4, site: 'a' },
        sh: { lab: 'shared', ab: 'sh.', mib: 12 },
        rt: { lab: 'routed', ab: 'rt.', mib: 104 },
        ag2: { lab: 'AG', ab: 'AG', idle: 2.0, site: 'f' },
        nx: { lab: 'next', ab: 'next', mib: 32 },
      };
      const TARGET = { o: 'o', a: 'sh', f: 'nx' }; // who consumes what each point brings in
      function simulate(st) {
        const order = ['in', 'core', 'o', 'ag1'].concat(st.shFirst ? ['sh', 'rt'] : ['rt', 'sh'], ['ag2', 'nx']);
        const pf = { o: 0, a: 0, f: 0 }, ov = { o: 0, a: 0, f: 0 };
        if (st.on) order.forEach((k) => { const p = PH[k]; if (p.site) { const cap = p.idle * R; pf[p.site] = Math.min(st.mb, cap); ov[p.site] = Math.max(0, st.mb - cap); } });
        // the L2 holds it until it is consumed; if the routed experts run before the shared one, their stream evicts what point a brought in
        const evicted = st.on && !st.shFirst && pf.a > 0;
        const used = { o: Math.min(pf.o, PH.o.mib), a: evicted ? 0 : Math.min(pf.a, PH.sh.mib), f: Math.min(pf.f, PH.nx.mib) };
        const fromL2 = { o: used.o, sh: used.a, nx: used.f };
        let tt = 0, vx = 0, carry = 0; const ops = [];
        order.forEach((k) => {
          const p = PH[k];
          if (p.idle) { ops.push({ k, p, v0: vx, v1: vx + p.idle }); tt += p.idle; vx += p.idle; carry = st.on ? (ov[p.site] / R) * OVC : 0; return; }
          const d0 = p.mib / R, l2 = fromL2[k] || 0, save = (l2 / R) * (1 - L2C), pen = carry; carry = 0;
          const vis0 = d0 <= HEAD ? d0 : HEAD + (d0 - HEAD) * COMP;
          ops.push({ k, p, v0: vx, v1: vx + vis0 - save + pen, l2, l2w: (l2 / R) * L2C, pen, comp: d0 > HEAD });
          tt += d0 - save + pen; vx += vis0 - save + pen;
        });
        const side = [], l2 = [];
        ops.forEach((op, i) => {
          if (!st.on || !op.p.site) return;
          const s = op.p.site, nxt = ops[i + 1];
          const ovEnd = nxt ? Math.min(op.v1 + ov[s] / R, nxt.v0 + Math.min(HEAD, nxt.v1 - nxt.v0)) : op.v1;
          side.push({ s, v0: op.v0, v1: op.v0 + pf[s] / R, ov: ov[s] > 0 ? { v0: op.v1, v1: ovEnd } : null });
          const tgt = ops.find((o) => o.k === TARGET[s]), ev = s === 'a' && evicted ? ops.find((o) => o.k === 'rt') : null;
          l2.push({ s, v0: op.v0, v1: ev ? ev.v0 : tgt.v0, mib: pf[s], to: PH[TARGET[s]].lab, ev: !!ev });
        });
        const idleT = ops.filter((o) => o.p.idle).reduce((a, o) => a + o.p.idle - (st.on ? pf[o.p.site] / R : 0), 0);
        return { ops, side, l2, total: tt, vtotal: vx, idleT, fromL2: used.o + used.a + used.f, ovT: st.on ? ov.o + ov.a + ov.f : 0, evicted };
      }
      const BASE = simulate({ on: false, shFirst: true, mb: 8 });
      const MIBT = Object.values(PH).reduce((a, p) => a + (p.mib || 0), 0);
      let SIM = simulate(S);
      const ctl = UI.controls(fig);
      const rT = UI.readout(ctl, 'stretch', '—'), rI = UI.readout(ctl, 'DRAM idle', '—'), rL = UI.readout(ctl, 'read from L2', '—');
      function upd() { SIM = simulate(S); rT.set(cm(SIM.total, 1) + ' u'); rI.set(cm((100 * SIM.idleT) / SIM.total, 1) + '%', SIM.idleT > 0.05 ? 'warn' : 'ok'); rL.set(cm(SIM.fromL2, 1) + ' MiB'); }
      UI.toggle(ctl, { label: 'L2 prefetch (TF_GLM_L2PF)', value: true, onChange: (v) => { S.on = v; upd(); } });
      UI.toggle(ctl, { label: 'shared expert before the routed ones', value: true, onChange: (v) => { S.shFirst = v; upd(); } });
      UI.slider(ctl, { label: 'budget per point', min: 2, max: 16, step: 2, value: 8, fmt: (v) => v + ' MiB', onInput: (v) => { S.mb = v; upd(); } });
      upd();
      const LEG = [
        ['b', 'DRAM-bound matmul: input, output, shared expert, routed experts, input of the next layer'],
        ['muted', 'phase in which the DRAM sits idle (red hatching on the DRAM lane): core (KDA recurrence or attention), AG + glue (all-gather, hyper-connections, router, top-k), FFN AG'],
        ['tf', 'prefetch on the side stream (points o, a, f) and what it leaves held in the L2'],
        ['a', 'bytes the matmul finds in the L2'],
        ['c', 'excess: spills past the idle phase, competes for the DRAM and delays the main stream'],
      ];
      const MCl = document.createElement('canvas').getContext('2d');
      function legendLayout(w) {
        const pad = 12, ncol = w < 560 ? 1 : 2, colW = (w - 2 * pad) / ncol, tw = colW - 22;
        const ent = LEG.map(([ci, d]) => ({ ci, ls: splitLines(MCl, d, tw, { size: 11.5 }) }));
        let hh = 0; const rows = [];
        for (let k = 0; k < ent.length; k += ncol) { const rr = ent.slice(k, k + ncol), n = Math.max(...rr.map((e) => e.ls.length)); rows.push({ rr, y: hh }); hh += n * 14 + 6; }
        return { pad, colW, rows, H: hh };
      }
      const SA = (w) => (w < 560 ? 120 : 76); // summary area, below the lanes
      fitH(fig, (w) => 28 + 4 * (w < 560 ? 40 : 46) + 16 + SA(w) + legendLayout(w).H + 8);
      V.fig(fig, (g, w, h, t) => {
        const C = V.C(), TF = tfc(C), sim = SIM, narrow = w < 560;
        const colOf = { b: C.b, muted: C.muted, tf: TF, a: C.a, c: C.c };
        const lanes = [['main', narrow ? 'main' : 'main stream'], ['side', narrow ? 'side' : 'side stream'], ['l2', 'L2'], ['dram', 'DRAM']];
        const LL = legendLayout(w);
        const L = narrow ? 66 : 112, Rr = 12, top = 28;
        const lh = V.clamp((h - top - 16 - SA(w) - LL.H - 8) / lanes.length, 30, 52), bh = lh - 12;
        const span = BASE.vtotal + 0.2;
        const X = (v) => L + (v / span) * (w - L - Rr);
        const tp = ((t * 1.5) % (span + 3)), tc = Math.min(tp, sim.vtotal);
        T(g, narrow ? 'time → (not to scale)' : 'time →  (not to scale: the long matmuls are compressed, ≈)', L, 17, { size: 11.5, color: C.muted, maxW: w - L - Rr });
        lanes.forEach(([k, lab], i) => {
          const y = top + i * lh;
          g.fillStyle = V.alpha(C.muted, i % 2 ? 0.03 : 0.07); g.fillRect(L, y, w - L - Rr, lh - 4);
          T(g, lab, L - 8, y + lh / 2 + 2, { size: 11.5, align: 'right', color: C.ink, font: k === 'l2' || k === 'dram' ? 'mono' : 'sans', maxW: L - 10 });
        });
        const ly = (k) => top + lanes.findIndex((x) => x[0] === k) * lh + 4;
        // main stream
        sim.ops.forEach((op) => {
          const y = ly('main'), x0 = X(op.v0), x1 = X(op.v1), bw = x1 - x0;
          if (op.p.idle) V.box(g, x0, y, Math.max(2, bw - 1), bh, { r: 4, fill: V.alpha(C.muted, 0.18), stroke: V.alpha(C.muted, 0.6), lw: 1 });
          else {
            V.box(g, x0, y, Math.max(2, bw - 1), bh, { r: 4, fill: V.alpha(C.b, 0.5), stroke: V.alpha(C.b, 0.95), lw: 1 });
            if (op.l2 > 0) V.box(g, x0, y, Math.max(2, X(op.v0 + op.l2w) - x0), bh, { r: 4, fill: V.alpha(C.a, 0.9) });
            if (op.pen > 0.01) { const p0 = X(op.v0 + (op.l2 > 0 ? op.l2w : 0)), p1 = X(op.v0 + (op.l2 > 0 ? op.l2w : 0) + op.pen); V.box(g, p0, y, Math.max(2, p1 - p0), bh, { r: 3, fill: V.alpha(C.c, 0.14), stroke: V.alpha(C.c, 0.85), lw: 1 }); stripes(g, p0, y, Math.max(2, p1 - p0), bh, V.alpha(C.c, 0.6)); }
            if (op.comp) {
              const xm = x0 + bw * 0.82; g.strokeStyle = V.alpha(C.ink, 0.75); g.lineWidth = 1.4;
              [-3, 3].forEach((dx) => { g.beginPath(); g.moveTo(xm + dx - 3, y + bh - 4); g.lineTo(xm + dx + 3, y + 4); g.stroke(); });
            }
          }
          const cx = op.comp ? x0 + bw * 0.4 : (x0 + x1) / 2, room = op.comp ? bw * 0.7 : bw - 6;
          g.font = V.font(11, 'sans', 600);
          const lab = g.measureText(op.p.lab).width < room ? op.p.lab : g.measureText(op.p.ab).width < room + 2 ? op.p.ab : '';
          if (lab) T(g, lab, cx, y + bh / 2 + 4, { size: 11, align: 'center', weight: 600, color: C.ink });
        });
        // side stream: the three points (dashed outline) and the prefetch
        const ys = ly('side');
        sim.ops.forEach((op) => {
          if (!op.p.site) return;
          const x0 = X(op.v0), x1 = X(op.v1);
          g.setLineDash([3, 3]); V.rr(g, x0, ys, Math.max(2, x1 - x0 - 1), bh, 4); g.strokeStyle = V.alpha(TF, 0.7); g.lineWidth = 1; g.stroke(); g.setLineDash([]);
        });
        sim.side.forEach((sd) => {
          const x0 = X(sd.v0), x1 = X(sd.v1);
          V.box(g, x0, ys, Math.max(2, x1 - x0 - 1), bh, { r: 4, fill: V.alpha(TF, 0.7), stroke: TF, lw: 1 });
          if (sd.ov) { const o0 = X(sd.ov.v0), o1 = X(sd.ov.v1); if (o1 - o0 > 1) { V.box(g, o0, ys, o1 - o0, bh, { r: 4, fill: V.alpha(C.c, 0.12), stroke: V.alpha(C.c, 0.85), lw: 1 }); stripes(g, o0, ys, o1 - o0, bh, V.alpha(C.c, 0.6)); } }
        });
        sim.ops.forEach((op) => {
          if (!op.p.site) return;
          const cx = X(op.v0) + 9, cy = ys + bh / 2;
          g.beginPath(); g.arc(cx, cy, 8, 0, Math.PI * 2); g.fillStyle = C.panel || C.bg; g.fill(); g.strokeStyle = TF; g.lineWidth = 1.5; g.stroke();
          T(g, op.p.site, cx, cy + 4, { size: 11.5, weight: 700, font: 'mono', align: 'center', color: C.ink });
        });
        // L2: what is held and for whom
        const yl = ly('l2');
        sim.l2.forEach((c, i) => {
          const x0 = X(c.v0), x1 = X(c.v1), bw = x1 - x0, col = c.ev ? (C.red || C.c) : TF;
          V.box(g, x0, yl, Math.max(2, bw - 1), bh, { r: 4, fill: V.alpha(col, 0.16), stroke: V.alpha(col, 0.85), lw: 1 });
          if (c.ev) { const xe = x1 - 1, yc = yl + bh / 2; g.strokeStyle = col; g.lineWidth = 2; g.beginPath(); g.moveTo(xe - 5, yc - 5); g.lineTo(xe + 5, yc + 5); g.moveTo(xe + 5, yc - 5); g.lineTo(xe - 5, yc + 5); g.stroke(); }
          // label inside the box if it fits; otherwise beside it, up to the next box
          const full = c.ev ? `${cm(c.mib, 1)} MiB evicted` : `${cm(c.mib, 1)} MiB → ${c.to}`, short = c.ev ? 'evicted' : `${cm(c.mib, 1)} MiB`;
          g.font = V.font(11, 'sans', 500);
          const nx = sim.l2[i + 1] ? X(sim.l2[i + 1].v0) : w - Rr, room = nx - x1 - (c.ev ? 16 : 8), ox = x1 + (c.ev ? 10 : 5);
          const fw = g.measureText(full).width, sw = g.measureText(short).width;
          if (!c.ev && fw < bw - 10) T(g, full, x0 + 5, yl + bh / 2 + 4, { size: 11, color: C.ink });
          else if (fw < room) T(g, full, ox, yl + bh / 2 + 4, { size: 11, color: c.ev ? col : C.ink });
          else if (sw < room) T(g, short, ox, yl + bh / 2 + 4, { size: 11, color: c.ev ? col : C.ink });
        });
        if (!S.on) T(g, 'empty: turn the prefetch on', L + 10, yl + bh / 2 + 4, { size: 11, color: C.muted, maxW: w - L - Rr - 20 });
        // DRAM: busy with the matmuls and the prefetch; red hatching = idle
        const yd = ly('dram');
        sim.ops.forEach((op) => {
          const x0 = X(op.v0), x1 = X(op.v1);
          if (!op.p.idle) {
            V.box(g, x0, yd, Math.max(2, x1 - x0 - 1), bh, { r: 3, fill: V.alpha(C.b, 0.35) });
            for (let xx = x0 + ((t * 24) % 12) + 3; xx < x1 - 3; xx += 12) V.dot(g, xx, yd + bh / 2, 1.6, V.alpha(C.ink, 0.55));
          } else {
            const sd = sim.side.find((q) => q.s === op.p.site), xb = sd ? X(sd.v1) : x0;
            if (xb > x0 + 1) { V.box(g, x0, yd, xb - x0 - 1, bh, { r: 3, fill: V.alpha(TF, 0.45) }); for (let xx = x0 + ((t * 24) % 12) + 3; xx < xb - 3; xx += 12) V.dot(g, xx, yd + bh / 2, 1.6, V.alpha(C.ink, 0.55)); }
            if (x1 - xb > 1.5) stripes(g, xb, yd, x1 - xb, bh, V.alpha(C.red || C.c, 0.55));
          }
        });
        sim.side.forEach((sd) => { if (sd.ov) { const o0 = X(sd.ov.v0), o1 = X(sd.ov.v1); if (o1 - o0 > 1) stripes(g, o0, yd, o1 - o0, bh, V.alpha(C.c, 0.7)); } });
        // time cursor
        const xp = X(tc); V.line(g, xp, top - 4, xp, top + lanes.length * lh, { color: C.c, lw: 1.6 });
        V.line(g, X(sim.vtotal), top - 4, X(sim.vtotal), top + lanes.length * lh, { color: V.alpha(C.a, 0.85), lw: 1.5, dash: [4, 3] });
        // summary
        const by = top + lanes.length * lh + 16, gain = (100 * (BASE.total - sim.total)) / BASE.total;
        const n1 = wrap(g, `stretch: ${cm(sim.total, 1)} u, against ${cm(BASE.total, 1)} u without prefetch (${gain > 0.05 ? '−' + cm(gain, 1) + '%' : gain < -0.05 ? '+' + cm(-gain, 1) + '%, slower' : 'no gain'}) · DRAM idle ${cm((100 * sim.idleT) / sim.total, 1)}%`, L, by, w - L - Rr, 15, { size: 12, weight: 600, color: C.ink, maxLines: 2 });
        let note;
        if (!S.on) note = 'Without prefetch, the DRAM reads nothing during the gray phases (red hatching).';
        else {
          note = `${cm(sim.fromL2, 1)} of the ${MIBT} MiB the stretch reads come from the L2 (${cm((100 * sim.fromL2) / MIBT, 0)}%).`;
          if (sim.evicted) note += ' What point a brought in for the shared expert was evicted by the routed experts’ stream.';
          if (sim.ovT > 0) note += ` Excess: ${cm(sim.ovT, 1)} MiB, which delay the main stream.`;
        }
        wrap(g, note, L, by + n1 * 15 + 4, w - L - Rr, 14, { size: 11.5, color: C.muted, maxLines: narrow ? 5 : 2 });
        // legend
        const ly0 = by + SA(w);
        g.strokeStyle = V.alpha(C.muted, 0.25); g.lineWidth = 1; g.beginPath(); g.moveTo(LL.pad, ly0 - 14); g.lineTo(w - LL.pad, ly0 - 14); g.stroke();
        LL.rows.forEach((row) => {
          row.rr.forEach((e, c) => {
            const x = LL.pad + c * LL.colW, y = ly0 + row.y;
            g.fillStyle = V.alpha(colOf[e.ci], 0.85); V.rr(g, x, y - 9, 12, 12, 3); g.fill();
            e.ls.forEach((ln, k) => T(g, ln, x + 18, y + k * 14, { size: 11.5, color: C.ink }));
          });
        });
      });
    }

    /* ================ Figure 10: each recipe’s subway line ================ */
    {
      const fig = $('#fig-metro');
      const LINES = {
        glm: [
          ['Target', 'TORCH_CUDA_ARCH_LIST=12.1a and FLASHINFER_CUDA_ARCH_LIST=12.1a in the Dockerfile and the launchers: the build and the runtime JIT target sm_121a.', 'Dockerfile:381-382 and 433; start.sh passes them to the container'],
          ['Compilation', 'nvcc/ptxas compile exllamav3_ext (pin c5d9c657) with the E2, E3 and FAST kernels; the AVX CPU targets become stubs on aarch64; the E3 module is built without --use_fast_math to keep the rounding of torch.sigmoid.', 'patch_exl3_ext_aarch64.py · patch_exl3_fat_kernel.py · patch_exl3_decode_pipeline.py · build_exl3_fat_moe_ext.py'],
          ['Binary', 'sm_121a cubins. The Dockerfile checks the symbols (exl3_moe, exl3_fat_*, glm53_fast_moe_version()==1) and SKIP_OVERLAY_VERIFY=0 runs the self-test on the GPU before serving.', 'Dockerfile:421-436 · SKIP_OVERLAY_VERIFY / EXL3_SELFCHECK_GPU'],
          ['Gates', 'get_min_capability 80 (exl3 accepted); E3 requires ≥ 9.0 (red.v4); FAST only on SM121; PDL restricted to major 9/10; page size 64 on SM12x.', 'overlay/exl3.py:249-250 · platforms/cuda.py (PDL, page size 64)'],
          ['Kernel', 'exl3_moe (thin, decode) and exl3_fat_moe (E3, prefill): 8-warp CTAs, multi-stage cp.async, XOR swizzle, Hadamard via shuffles across lanes.', 'overlay/exl3_fat_gemm.cu · overlay/exl3_fat_moe.cu'],
          ['Tensor core', 'The trellis is dequantized in registers and feeds mma.sync m16n8k16 FP16→FP32. Native FP4 does not apply: the trellis is not a hardware format.', 'ptx_mma_m16n8k16 · dq_dispatch<4,1>'],
          ['Bottleneck', 'Decode: bandwidth. Prefill: the MoE with ~28 rows per expert at MNBT 1024 gives 112 FLOP/B, below the 458 ridge; 947 tok/s ≈ 14% of BF16 peak.', 'docs/improve-prefill.md §2'],
        ],
        qwen: [
          ['Target', 'The day-0 image comes precompiled: unsuffixed sm_120 cubins (e.g., cutlass_scaled_mm_sm120_fp8), plus Triton JIT for the QSA kernels.', 'image vllm/vllm-openai:qwen38-flash-next (day-0)'],
          ['Compatibility', 'sm_120 SASS runs on 12.1 (same major, higher minor): no need to recompile for the GB10.', 'docs/CLAUDE/fable5-1-report.md §8.1'],
          ['Gates', 'MoE: TRTLLM and CuteDSL are SM10x only → CUTLASS SM120. GDN prefill on FlashInfer is SM90/10x only → Triton FLA. Skinny GEMM is SM103 only → cuBLAS.', 'fused_moe/oracle/nvfp4.py · qwen_gdn_linear_attn.py:112-160 · low_latency_gemm.py'],
          ['Subtle gate', '_is_sm120() compares == (12, 0): on the GB10, the v0.30 backport stays on the default table (GB300), not the sm_120 one (inferred from the code).', 'files/patch_qsa_fp8_kv_v030.py:129-165'],
          ['Kernel', 'CUTLASS SM120 grouped GEMM (experts), Triton QSA (sparse attention; profiles for 48 SMs via QSA_PROFILE) and BF16 cuBLAS (dense layers).', 'files/qsa_gb10/apply_patch.py · QSA_PROFILE'],
          ['Tensor core', 'W4A4 experts: block-scaled FP4 MMA (kind::mxf4nvf4). Dense layers: BF16 MMA.', 'FlashInfer CUTLASS SM120 grouped GEMM (gen_cutlass_fused_moe_sm120_module)'],
          ['Bottleneck', '≈9.2 GiB per GPU per step at 235–245 GB/s = 40–42 ms; × τ≈3.2 (2.2 accepted + bonus) = ceiling of ~76–80 tok/s; measured 52.1 (step ~60 ms, ~1/3 fixed cost).', 'docs/CLAUDE/fable5-1-report.md §2 · README (MTP=3)'],
        ],
        ds: [
          ['Target', 'CUTE_DSL_ARCH=sm_121a, TORCH_CUDA_ARCH_LIST=12.1a and FLASHINFER_CUDA_ARCH_LIST=12.1a in the .env and the compose file.', '.env.dspark.example:601-604 · docker-compose.dspark.yml:198-204'],
          ['JIT', 'b12x (CuTeDSL), FlashInfer and DeepGEMM compile on the fly, with caches persisted in volumes (VLLM_CACHE_ROOT). Without CUTE_DSL_ARCH, b12x may compile a slower W4A16 in the middle of serving.', 'VLLM_CACHE_ROOT · B12X_CUTE_COMPILE_CACHE_DIR · DG_JIT_*'],
          ['Binary', 'DeepGEMM generates sm121_* names, but only sm120_* headers exist: the #define alias solves it (DSPARK_ENABLE_DEEPGEMM_SM121_ALIAS=1).', 'patches/hotfix-deepgemm-sm121-mqa-header-alias.sh'],
          ['Gates', 'Indexer MXFP4 cache: family(100) → + family(120) (lane A hotfix). MegaMoE is SM100 only. FlashMLA is Hopper/SM10x only → FlashInfer SM120 kernel (in lane A, inside the image; in the overlay, lane B, the DSV4_SPARSE_MLA_SM120 backend). Hybrid NVFP4 plugin is cc 12.1 only, opt-in and does not cover the MXFP4 checkpoint.', 'hotfix-vllm-mxfp4-indexer-cache.py · vllm_patch_gb10/…/kernel.py:38 · recipe/overlay/…/nvidia/sm120.py (lane B)'],
          ['Kernel', 'Production (lane A): b12x W4A16 MoE and FlashInfer’s SM120 sparse MLA. The Stage-C overlay (lane B, historical) also shows 3 auxiliary streams for the compressor and indexer input GEMMs; the equivalent lane A code is not in the repository.', 'fused_moe/b12x_moe.py · recipe/overlay/…/nvidia/model.py:1700-1709 (lane B)'],
          ['Tensor core', 'W4A16: the FP4 (MXFP4) is dequantized to BF16 → BF16 MMA. b12x’s native FP4 path only accepts a modelopt_nvfp4 source.', 'fused_moe/b12x_moe.py:427-497 (fp4_e8m0_k32 → W4A16)'],
          ['Bottleneck', '16.5–17 GB per rank per step → 68–74 ms; c=1 is ~80% weight streaming. The rest, ~10–15 ms, is fixed cost; the ≈108 collectives alone already add up to 3–5 ms.', 'docs/CLAUDE/fable5-1-report.md §2'],
        ],
      };
      const S = { rc: 'glm', i: 0, hold: -1, last: 0, pos: 0 };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Recipe', options: [{ v: 'glm', label: 'GLM' }, { v: 'qwen', label: 'Qwen' }, { v: 'ds', label: 'DeepSeek' }], value: 'glm', onChange: (v) => { S.rc = v; S.i = 0; S.pos = 0; } });
      UI.button(ctl, '← station', () => { S.i = (S.i + 6) % 7; S.hold = Infinity; });
      UI.button(ctl, 'station →', () => { S.i = (S.i + 1) % 7; S.hold = Infinity; });
      // height from content: the station with the longest text (from any recipe) must fit in the box
      const MCm = document.createElement('canvas').getContext('2d');
      function metroNeed(w) {
        const narrow = w < 560, bw = w - 28, by = narrow ? 24 + 7 * 40 : 110;
        let mx = 0;
        Object.values(LINES).forEach((st) => st.forEach(([, tx, ref]) => {
          const nl = Math.min(narrow ? 12 : 9, splitLines(MCm, tx, bw - 32, { size: narrow ? 13 : 14.5 }).length);
          const rl = ref ? Math.min(4, splitLines(MCm, 'where: ' + ref, bw - 32, { size: 11.5, font: 'mono' }).length) : 0;
          mx = Math.max(mx, 52 + nl * (narrow ? 19 : 21) + (ref ? 12 + rl * 15 : 0) + 28);
        }));
        return by + mx + 10;
      }
      fitH(fig, (w) => Math.max(w < 560 ? 480 : 345, metroNeed(w)));
      const H = V.fig(fig, (g, w, h, t) => {
        const C = V.C(), narrow = w < 560, pad = 14, st = LINES[S.rc], col = C[S.rc];
        if (S.hold !== Infinity && t > S.hold) { if (t - S.last > 3.4) { S.i = (S.i + 1) % 7; S.last = t; } }
        S.pos += (S.i - S.pos) * 0.08;
        if (Math.abs(S.i - S.pos) > 3.5) S.pos = S.i;
        const P = [];
        if (!narrow) { const x0 = pad + 30, x1 = w - pad - 30, y = 58; for (let k = 0; k < 7; k++) P.push([V.lerp(x0, x1, k / 6), y]); }
        else { const x = pad + 16, y0 = 24; for (let k = 0; k < 7; k++) P.push([x, y0 + k * 40]); }
        g.strokeStyle = V.alpha(col, 0.75); g.lineWidth = 6; g.lineCap = 'round';
        g.beginPath(); g.moveTo(P[0][0], P[0][1]); g.lineTo(P[6][0], P[6][1]); g.stroke(); g.lineCap = 'butt';
        const ptr = H.pointer;
        st.forEach(([name], k) => {
          const [x, y] = P[k], on = k === S.i;
          if (ptr.clicked && Math.hypot(ptr.x - x, ptr.y - y) < 22) { S.i = k; S.hold = t + 12; S.last = t; }
          g.beginPath(); g.arc(x, y, on ? 11 : 8, 0, Math.PI * 2); g.fillStyle = on ? col : C.panel || C.bg; g.fill(); g.strokeStyle = col; g.lineWidth = 3; g.stroke();
          if (!narrow) wrap(g, name, x, y + 30, (w - 2 * pad) / 7 - 6, 14, { size: 11.5, align: 'center', weight: on ? 700 : 500, color: on ? C.ink : C.muted, maxLines: 2 });
          else T(g, `${k + 1}. ${name}`, x + 20, y + 4, { size: 12, weight: on ? 700 : 500, color: on ? C.ink : C.muted });
        });
        const pf = Math.floor(S.pos), pfr = S.pos - pf, a = P[Math.min(6, pf)], b = P[Math.min(6, pf + 1)];
        const tx = V.lerp(a[0], b[0], pfr), ty = V.lerp(a[1], b[1], pfr);
        V.glow(g, tx, ty, 20, col, 0.8);
        V.box(g, tx - 13, ty - 7, 26, 14, { r: 7, fill: C.ink });
        V.dot(g, tx + 6, ty, 2.5, col);
        const bx = pad, by = narrow ? 24 + 7 * 40 : 110, bw = w - 2 * pad, bh = h - by - 10;
        V.box(g, bx, by, bw, bh, { r: 12, fill: V.alpha(col, 0.07), stroke: V.alpha(col, 0.55) });
        const [nm, tx2, ref] = st[S.i];
        T(g, `${S.i + 1}/7 · ${nm}`, bx + 16, by + 26, { size: 15, weight: 700, color: col, maxW: bw - 32 });
        const nl = wrap(g, tx2, bx + 16, by + 52, bw - 32, narrow ? 19 : 21, { size: narrow ? 13 : 14.5, color: C.ink, maxLines: narrow ? 12 : 9 });
        if (ref) wrap(g, 'where: ' + ref, bx + 16, by + 52 + nl * (narrow ? 19 : 21) + 12, bw - 32, 15, { size: 11.5, font: 'mono', color: C.muted, maxLines: 4 });
        if (S.hold === Infinity || t < S.hold) T(g, 'paused · click another station or use the arrows', bx + bw - 12, by + bh - 12, { size: 11, align: 'right', color: C.muted, maxW: bw - 24 });
      });
    }
  },
});

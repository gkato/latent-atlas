LA.chapter({
  id: 'motor-exato-tensorfold',
  part: 9,
  order: 48,
  title: 'An exact engine: how TensorFold thinks about inference',
  navTitle: 'Exact engine (TensorFold)',
  subtitle: 'The same GLM on the same two Sparks, with a different engine: bit-for-bit exactness as a contract, keyed sampling, ranks that agree without talking, memory measured before it is promised, extents instead of pages, kept prompt states and an 11 µs all-gather.',
  minutes: 45,
  recipes: ['tf', 'glm'],
  tags: ['TensorFold', 'exactness', 'batch invariance', 'keyed sampling', 'Gumbel-max', 'all-gather', 'extents', 'replay', 'RDMA', 'RoCE'],
  goals: [
    'Explain what changes between a <b>generic</b> engine (vLLM) and a <b>per-family</b> engine (TensorFold), and why a hybrid model like GLM-5.3-Flash pays dearly for adapters',
    'Show why fp32 non-associativity breaks “batch = alone”, how row-independent kernels and <b>keyed sampling</b> (Gumbel-max over \\(H(s,p,i)\\)) make drafted = serial, concurrent = solo and resumed = fresh, and what summing the fp32 partials in rank order actually changes in tensor parallelism',
    'Describe the “rank 0 decides, rank 1 applies” protocol, the <code>MemAvailable</code> − reserve budget, the pool of 2,048-token extents and the prompt states on the grid of 64 with replay, comparing each one with the vLLM solution',
    'Compute the gain of the one-shot RDMA all-gather with the α-β model and weigh, in a trade-off table, what you gain and what you lose by switching engines',
  ],
  prereqs: ['vllm-arquitetura', 'paged-attention-kv', 'prefix-caching', 'especulacao-fundamentos', 'nccl-rdma'],
  summary: [
    'TensorFold is a <b>per-family engine</b>: each family brings its own kernels, cache and verification. For GLM this removes the vLLM adapters (latent 656 → 528 B per token and layer, KDA state outside 3,584-token pages, graphs per window width), at the cost of generality: only 2 ranks in v0.6.0 (the recipe added a third with patches 0066–0068), only the formats the family implements. Since 1.0.0 the engine is a native Zig binary (Metal and CUDA, no Python or MLX); the recipe still pins v0.6.0.',
    '<b>Exactness is a contract</b>: drafted = serial, concurrent = solo, resumed = fresh, rank 0 = rank 1. Since fp32 addition is not associative, this requires <b>row-independent</b> kernels (split-K and tiles fixed by shape), per-row tables, shape-dependent choices made per stream and atomics only on integers. “Lossy, never inexact”: q4 and FP8 KV lose information but do not break the equalities.',
    '<b>Keyed sampling</b>: \\(x_p = \\arg\\max_i (z_{p,i}/T + g_{p,i})\\) with \\(g = -\\ln(-\\ln H(s,p,i))\\), seed = <code>seed</code> or SHA-256 of the prompt, choice in float64 on the host. Verifying becomes comparing integers, the output is independent of the drafter, and the drafter can use the noise (<code>fnc7:0.3</code>). In v0.6.0 you lose <code>logprobs</code>, <code>n&gt;1</code> and penalties (the native engine has since added <code>logprobs</code> on CUDA, and since 1.1.1 a sampled request without a seed gets a fresh random one unless <code>--prompt-seed</code> is set).',
    'TP without all-reduce: <b>all-gather of fp32 partials</b>, a sum with rank 0 first and <b>one</b> rounding to bf16, whereas the vLLM all-reduce adds partials already rounded to bf16. With two ranks the order does not change bits (the sum of two operands is commutative); the rule applies to 3+ ranks and takes the order away from NCCL, and the transport (NCCL or RoCE) becomes irrelevant to the bits, at the cost of ~2× bytes on the wire. Rank 0 decides everything that depends on time, clients or allocation and sends <b>one</b> message of operations per iteration; rank 1 applies them and recomputes the drafts on its own; when idle, it waits on a doorbell.',
    'Memory: budget = <code>MemAvailable</code> − reserve (14.5 GiB), refusal <b>before</b> loading weights (on the Spark, running out of memory freezes the machine); a shared pool of 2,922,496 tokens divided into <b>extents</b> of 2,048 with first fit, MOVE, compaction and eviction of kept prompts (LRU in v1.3.2), instead of pages.',
    'Reuse without APC: states kept on the <b>grid of 64</b> (because of chunked KDA), at the end of the system prompt and at the divergence point, and <b>replay</b> with the logits row: identical 64k 34 s → &lt;0.07 s; 7.9k system 4.24 → 0.13 s. The follow-up redoes the previous answer, a case where APC can win.',
    'Rounds with windows of up to 16 rows per stream and 32 in the batch, one CUDA graph per row count with tables on the device; small all-gathers over one-shot RDMA: 11 µs versus 45 µs per 16 KiB, ~90 × 34 µs ≈ 3 ms per round, +6% decode. Large measured gains in decode and reuse; losses in scale, API and generality.',
  ],
  quiz: [
    {
      q: 'A GEMM kernel picks split-K based on the number of rows in the batch. Why does that break “concurrent = solo”, even without any bug?',
      o: ['Because rows from other requests leak into your row', 'Because split-K only works with bf16', 'Because fp32 addition is not associative: a different partitioning of the reduction rounds in a different order, the logits change in the last bits and, in a near-tie, the chosen token flips', 'Because the CUDA graph of the batch is different and skips layers'],
      a: 2,
      e: 'The data in your row is the same. What changes is the order of the additions, and \\((a\\oplus b)\\oplus c \\ne a\\oplus(b\\oplus c)\\) in floating point. That is why TensorFold fixes the partition by the weight shape (row-independent kernels) and runs per stream everything that picks an algorithm by size.',
    },
    {
      q: 'With keyed sampling, why is the answer with drafts identical to the serial one, whatever the drafter?',
      o: ['Because the target sample at position \\(p\\) is \\(\\arg\\max_i(z_{p,i}/T + g(s,p,i))\\), a function only of the logits and the key; a draft is only accepted if it equals that sample, so the drafter changes how many tokens come out per round, not which ones', 'Because the drafter is trained to imitate the target perfectly', 'Because the temperature is forced to 0 when there are drafts', 'Because rejection with \\(\\min(1,p/q)\\) preserves the distribution'],
      a: 0,
      e: 'With noise indexed by (seed, position, token), the token at each position is decided before any draft exists. Verifying is comparing integers. Rejection (option D) only preserves the distribution: with a stateful generator, a different drafter consumes different random numbers and produces a different sequence.',
    },
    {
      q: 'With two ranks, rank 0 computes \\(P^{(0)}\\oplus P^{(1)}\\) and rank 1 computes \\(P^{(1)}\\oplus P^{(0)}\\), both in fp32. What happens, and what is the real numerical gain of the TensorFold path?',
      o: ['They diverge in the last bits, because fp32 addition is not associative; that is why TensorFold enforces rank order on both Sparks', 'They only diverge when the transport is one-shot RoCE, which does not preserve the order of the partials; with NCCL they come out identical', 'They are identical, and the TensorFold gain over all-reduce is sending half the bytes over the wire, since only the partials travel', 'They are identical (with two operands fp32 addition is commutative); rank order is the rule for 3+ ranks and takes the order away from NCCL, and the real gain is summing in fp32 and rounding to bf16 once'],
      a: 3,
      e: '\\(a\\oplus b = b\\oplus a\\) exactly in floating point; what fails is associativity, which only comes into play with three or more operands. With two ranks, rank order is the general rule of the engine, not the cause of the identical bits. The gain with two ranks is numerical: \\(\\mathrm{bf16}(P^{(0)}\\oplus P^{(1)})\\) rounds once, while an all-reduce of bf16 activations gives \\(\\mathrm{bf16}(\\mathrm{bf16}(P^{(0)})+\\mathrm{bf16}(P^{(1)}))\\), with two roundings. And TensorFold sends <b>more</b> bytes (fp32, ~2×), not fewer.',
    },
    {
      q: 'Why does an idle rank 1 wait on a “doorbell” in the TCPStore rather than inside an all-gather?',
      o: ['Because the TCPStore is faster than NCCL for large tensors', 'Because waiting in an all-gather keeps the GPU and a core spinning; the key <code>tf_glm_request_&lt;n&gt;</code> lets the idle server consume neither GPU nor CPU, and rank 0 rings the doorbell before the first message', 'Because NCCL does not allow two ranks', 'Because rank 1 needs to read the prompts from disk'],
      a: 1,
      e: 'A blocked collective keeps spinning while it waits for its peer. The doorbell moves the wait onto a socket. This started as a patch in the recipe and was merged into TensorFold v0.6.0 (“an idle GLM rank no longer spins”).',
    },
    {
      q: 'An identical prompt of 64,001 tokens is resubmitted to the TensorFold of the recipe (grid of 64). How many tokens are prefilled again, and why?',
      o: ['Zero, always: replay works for any length', '3,584, the size of a block', '1: the state was kept at \\(B = 64\\lfloor 64{,}001/64 \\rfloor = 64{,}000\\), and the end (with the logits row) is only kept when it falls on the grid', '64,001: identical prompts are not reused'],
      a: 2,
      e: 'By the <code>grid_point</code> rule, a prompt that does not end on the grid keeps its state at the last multiple of 64. Replay with no prefill at all requires the end state with the logits row, which happens when \\(n \\bmod 64 = 0\\). Here 1 token is left, much better than the simplified APC, which redoes \\(64{,}001 - 3{,}584\\lfloor 64{,}000/3{,}584\\rfloor = 64{,}001 - 60{,}928 = 3{,}073\\) (\\(17 \\times 3{,}584 = 60{,}928\\)).',
    },
    {
      q: 'Using the α-β model and the README measurements, where does the order of magnitude of the +6% decode gain from one-shot RDMA come from?',
      o: ['From fixed latency: ~90 all-gathers per round (2 per layer × 45 layers) × (45 − 11) µs ≈ 3 ms in a round on the order of 50 ms', 'From more bandwidth: RoCE doubles the GB/s of the cable', 'From compressing the partials to bf16', 'From prefill, which gets 6% faster'],
      a: 0,
      e: 'Decode messages are small (\\(R \\times 16\\) KiB), and the time is dominated by α. The one-shot transport removes the fixed cost of NCCL going through the host: one kernel call, RDMA WRITE and a flag on the same QP. The partials stay in fp32, and prefill stays on NCCL, which is good at large messages (and uses both rails).',
    },
  ],
  mount(root, LA) {
    const V = LA.viz, UI = LA.ui;
    const RED = () => V.C().red || '#FB2C36';
    const TFC = () => { const v = getComputedStyle(document.documentElement).getPropertyValue('--tf').trim(); return v || V.C().d; };
    const fmt = (x, d = 2) => (+x).toFixed(d);
    const fmtInt = (n) => Math.round(n).toLocaleString('en-US');
    const nocase = (c) => { const el = c && (c.el || c); if (el && el.querySelectorAll) el.querySelectorAll('label, span').forEach((n) => { if (!n.classList.contains('val')) n.style.textTransform = 'none'; }); return c; };

    /* ------------------------------------------------ helpers comuns */
    function adaptH(fig, wide, narrow, bp = 640) {
      const st = fig.querySelector('.la-fig-stage');
      const set = () => { const want = (fig.clientWidth < bp ? narrow : wide) + 'px'; if (st.style.height !== want) st.style.height = want; };
      set();
      const ro = new ResizeObserver(set); ro.observe(fig);
      if (LA.onCleanup) LA.onCleanup(() => ro.disconnect());
    }
    function tw(g, s, size, font = 'sans', weight = 400) { g.font = V.font(size, font, weight); return g.measureText(String(s)).width; }
    /** texto que encolhe até `min` px e depois corta com reticências */
    function tfit(g, s, x, y, maxW, o = {}) {
      let size = o.size || 12; const min = o.min || 10; s = String(s);
      while (size > min && tw(g, s, size, o.font, o.weight) > maxW) size -= 0.5;
      if (tw(g, s, size, o.font, o.weight) > maxW) { while (s.length > 2 && tw(g, s + '…', size, o.font, o.weight) > maxW) s = s.slice(0, -1); s += '…'; }
      V.text(g, s, x, y, Object.assign({}, o, { size }));
    }
    function chip(g, x, y, w, h, label, o = {}) {
      const C = V.C();
      V.box(g, x, y, w, h, { r: o.r == null ? 6 : o.r, fill: o.fill || V.alpha(C.muted, 0.12), stroke: o.stroke || V.alpha(C.muted, 0.4), lw: o.lw || 1.2 });
      if (label != null && label !== '') tfit(g, label, x + w / 2, y + h / 2 + 0.5, w - 6, { size: o.size || 11.5, min: o.min || 9, font: o.font || 'mono', weight: o.weight || 500, color: o.color || C.ink, align: 'center', base: 'middle' });
    }
    function mark(g, x, y, ok, s = 6) {
      const C = V.C();
      g.lineWidth = 2.4; g.strokeStyle = ok ? C.a : RED(); g.lineCap = 'round'; g.beginPath();
      if (ok) { g.moveTo(x - s, y); g.lineTo(x - s * 0.3, y + s * 0.7); g.lineTo(x + s, y - s * 0.8); }
      else { g.moveTo(x - s * 0.8, y - s * 0.8); g.lineTo(x + s * 0.8, y + s * 0.8); g.moveTo(x + s * 0.8, y - s * 0.8); g.lineTo(x - s * 0.8, y + s * 0.8); }
      g.stroke(); g.lineCap = 'butt';
    }
    function hatch(g, x, y, w, h, col, step = 6) {
      g.save(); g.beginPath(); g.rect(x, y, w, h); g.clip();
      g.strokeStyle = col; g.lineWidth = 1;
      g.beginPath();
      for (let k = -h; k < w + h; k += step) { g.moveTo(x + k, y + h); g.lineTo(x + k + h, y); }
      g.stroke(); g.restore();
    }
    function bell(g, x, y, s, col, ring = 0) {
      g.save(); g.translate(x, y); g.rotate(Math.sin(ring * 30) * 0.35 * (ring > 0 ? 1 : 0));
      g.beginPath();
      g.moveTo(-s * 0.55, s * 0.35); g.quadraticCurveTo(-s * 0.5, -s * 0.55, 0, -s * 0.6); g.quadraticCurveTo(s * 0.5, -s * 0.55, s * 0.55, s * 0.35); g.closePath();
      g.fillStyle = col; g.fill();
      g.beginPath(); g.arc(0, s * 0.45, s * 0.14, 0, Math.PI * 2); g.fill();
      g.restore();
    }
    function hourglass(g, x, y, s, col, t) {
      g.save(); g.translate(x, y); g.rotate((Math.floor(t) % 2) * Math.PI * V.ease.inOut(Math.min(1, (t % 1) * 4)));
      g.strokeStyle = col; g.lineWidth = 1.6; g.beginPath();
      g.moveTo(-s / 2, -s / 2); g.lineTo(s / 2, -s / 2); g.lineTo(-s / 2, s / 2); g.lineTo(s / 2, s / 2); g.closePath(); g.stroke();
      g.restore();
    }
    /** gerador idêntico ao usado para escolher os exemplos numéricos (mulberry32) */
    function rng(seed) {
      let a = seed >>> 0 || 1;
      const r = () => { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
      r.normal = () => { let u = 0, v = 0; while (!u) u = r(); while (!v) v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
      return r;
    }
    const f32 = Math.fround;
    const f32buf = new Float32Array(1), u32buf = new Uint32Array(f32buf.buffer);
    const hex32 = (x) => { f32buf[0] = x; return '0x' + u32buf[0].toString(16).toUpperCase().padStart(8, '0'); };
    const sigs = (x, n = 9) => (+x).toPrecision(n);
    const bitsOf = (x) => { f32buf[0] = x; const u = u32buf[0]; const b = []; for (let i = 31; i >= 0; i--) b.push((u >>> i) & 1); return b; };
    /** duas linhas de 32 bits (sinal · expoente · mantissa), com os bits diferentes destacados */
    function bitRows(g, x, y, w, a, b, la, lb) {
      const C = V.C();
      const labW = 74, cw = (w - labW) / 32, chh = 14;
      const A = bitsOf(a), B = bitsOf(b);
      [[A, la], [B, lb]].forEach(([bits, lab], r) => {
        const yy = y + r * (chh + 4);
        tfit(g, lab, x, yy + chh - 2, labW - 6, { size: 11, color: C.muted });
        bits.forEach((bt, i) => {
          const col = i === 0 ? C.c : i < 9 ? C.b : C.a;
          const diff = A[i] !== B[i];
          g.fillStyle = bt ? V.alpha(col, 0.85) : V.alpha(col, 0.12);
          V.rr(g, x + labW + i * cw + 0.5, yy, cw - 1, chh, 2); g.fill();
          if (diff) { g.strokeStyle = RED(); g.lineWidth = 2; V.rr(g, x + labW + i * cw, yy - 1, cw, chh + 2, 2); g.stroke(); }
        });
      });
      const yl = y + 2 * (chh + 4) + 11;
      V.text(g, 's', x + labW + cw / 2, yl, { size: 10, font: 'mono', align: 'center', color: C.c });
      V.text(g, 'exponent', x + labW + cw * 5, yl, { size: 10, align: 'center', color: C.b });
      V.text(g, 'mantissa (23 bits)', x + labW + cw * 20.5, yl, { size: 10, align: 'center', color: C.a });
      return A.filter((v, i) => v !== B[i]).length;
    }

    /* =================================================== FIG 1 — motores */
    (function () {
      const fig = root.querySelector('#fig-motores');
      adaptH(fig, 440, 560);
      const S = { mode: 'vllm', tSwitch: 0 };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Engine', options: [{ v: 'vllm', label: 'vLLM (generic)' }, { v: 'tf', label: 'TensorFold (per family)' }], value: 'vllm', onChange: (v) => { S.mode = v; S.switched = true; upd(); } });
      const rB = UI.readout(ctl, 'B/token/layer', '—'), rL = UI.readout(ctl, '× 11 DSA layers', '—'), rG = UI.readout(ctl, 'at 1,048,576 tokens', '—');
      const LAY = {
        vllm: { name: 'vLLM · fp8_ds_mla', total: 689, segs: [[512, 'cells'], [16, 'scale'], [128, 'zero'], [33, 'idx']], legend: '512 FP8 + 16 scales + 128 zeroed RoPE + 33 indexer = 689 B' },
        tf: { name: 'TensorFold · kv8', total: 564, segs: [[512, 'cells'], [4, 'scale'], [12, 'pad'], [36, 'idx']], legend: '512 e4m3 + 4 scale + 12 pad + 36 indexer (144 B / 4 tokens) = 564 B' },
      };
      function upd() {
        const L = LAY[S.mode];
        rB.set(L.total + ' B', S.mode === 'tf' ? 'ok' : null);
        rL.set(fmtInt(L.total * 11) + ' B');
        rG.set(fmt((L.total * 11 * 1048576) / 2 ** 30, 2) + ' GiB', S.mode === 'tf' ? 'ok' : null);
      }
      upd();
      const STACK = {
        vllm: [
          { t: 'API server (OpenAI)', ad: null },
          { t: 'EngineCore: generic scheduler', ad: 'custom “fair” scheduler', hit: [0, 1, 2, 3] },
          { t: 'KV manager: pages', ad: 'KDA state in 3,584-token blocks', hit: [0] },
          { t: 'Model runner + registry', ad: '25 Python overlays', hit: [0, 1, 2, 3] },
          { t: 'Backends: FlashInfer MLA · Mamba · MoE', ad: 'latent 512 → 576', hit: [1] },
        ],
        tf: [
          { t: 'rank 0: HTTP + GLM scheduler thread', ad: null },
          { t: 'GlmEngine (glm5_next family)', ad: null },
          { t: 'its own kernels: chunked KDA · DSA latent · EXL3 experts · HC glue', ad: null },
        ],
      };
      const PIECES = [['KDA', 'b'], ['DSA', 'c'], ['MoE', 'a'], ['HC', 'd']];
      const codes = Array.from({ length: 32 }, (_, i) => 0.5 + 0.5 * Math.sin(i * 2.3 + 1.1));

      V.fig(fig, (g, w, h, t) => {
        const C = V.C(), tf = TFC();
        const narrow = w < 560, pad = 12;
        const mode = S.mode, stack = STACK[mode];
        const accent = mode === 'tf' ? tf : C.glm;
        tfit(g, mode === 'vllm' ? 'vLLM: the model plugs into the engine\'s sockets' : 'TensorFold: the engine is written for the family', pad, 17, w - 2 * pad, { size: 12.5, weight: 600, color: C.ink });
        const topH = narrow ? h * 0.6 : h * 0.6;
        const laneW = narrow ? 48 : 62;
        const sx = pad + laneW + 8, sw = w - sx - pad;
        const y0 = 30;
        const rows = stack.length + (mode === 'tf' ? 1 : 0);
        const gapY = 6;
        const lh = Math.min(narrow ? 48 : 40, (topH - y0 - gapY * (rows - 1)) / rows);
        const stackBot = y0 + rows * lh + (rows - 1) * gapY;
        // faixa dos pedaços do modelo
        V.box(g, pad, y0, laneW, stackBot - y0, { r: 10, fill: V.alpha(C.glm, 0.06), stroke: V.alpha(C.glm, 0.35), lw: 1 });
        V.text(g, 'GLM', pad + laneW / 2, y0 + 14, { size: 11, font: 'mono', weight: 600, align: 'center', color: C.glm });
        // camadas
        const per = 4.2;
        const prog = PIECES.map((_, i) => ((t / per) + i / 4) % 1);
        stack.forEach((L, i) => {
          const y = y0 + i * (lh + gapY);
          V.box(g, sx, y, sw, lh, { r: 8, fill: V.alpha(accent, 0.08), stroke: V.alpha(accent, 0.45), lw: 1.2 });
          const inside = PIECES.map((_, k) => { const py = y0 + 22 + prog[k] * (stackBot - y0 - 30); return py > y && py < y + lh; });
          if (L.ad) {
            const glowing = (L.hit || []).some((k) => inside[k]);
            const adW = Math.min(sw * (narrow ? 0.92 : 0.46), tw(g, L.ad, 11, 'sans', 600) + 18);
            const ax = narrow ? sx + 8 : sx + sw - adW - 8, ay = narrow ? y + lh - 21 : y + lh / 2 - 10;
            if (glowing) V.glow(g, ax + adW / 2, ay + 10, adW * 0.6, C.d, 0.45);
            chip(g, ax, ay, adW, 18, L.ad, { fill: V.alpha(C.d, glowing ? 0.35 : 0.16), stroke: C.d, font: 'sans', weight: 600, size: 11 });
            tfit(g, L.t, sx + 10, narrow ? y + 16 : y + lh / 2 + 4, narrow ? sw - 20 : sw - adW - 28, { size: 12, color: C.ink, weight: 500 });
            // ligação peça → adaptador
            (L.hit || []).forEach((k) => {
              if (!inside[k]) return;
              const py = y0 + 22 + prog[k] * (stackBot - y0 - 30);
              V.line(g, pad + laneW / 2 + 18, py, ax, ay + 9, { color: V.alpha(C.d, 0.7), lw: 1.2, dash: [3, 3], dashOffset: -t * 20 });
            });
          } else {
            V.wrap(g, L.t, sx + 10, y + (lh > 36 ? lh / 2 - 2 : lh / 2 + 4), sw - 20, { size: 12, color: C.ink, weight: 500, maxLines: 2, lineH: 14 });
          }
        });
        if (mode === 'tf') {
          const y = y0 + stack.length * (lh + gapY);
          g.setLineDash([5, 4]);
          V.box(g, sx, y, sw, lh, { r: 8, fill: V.alpha(tf, 0.05), stroke: V.alpha(tf, 0.7), lw: 1.2 });
          g.setLineDash([]);
          V.wrap(g, 'rank 1 (on the other Spark): applies the same operations with the same kernels', sx + 10, y + (lh > 36 ? lh / 2 - 2 : lh / 2 + 4), sw - 20, { size: 11.5, color: C.muted, maxLines: 2, lineH: 14 });
          const yA = y0 + lh / 2, xA = sx + sw - 14;
          V.arrow(g, xA, yA + 8, xA, y - 2, { color: V.alpha(tf, 0.8), lw: 1.4, dash: [4, 4], head: 7 });
          const ph = (t * 0.6) % 1;
          V.dot(g, xA, V.lerp(yA + 8, y - 4, ph), 3.5, tf);
        }
        // pedaços descendo
        PIECES.forEach(([lab, ck], k) => {
          const col = C[ck];
          const py = y0 + 22 + prog[k] * (stackBot - y0 - 30);
          const bw = narrow ? 36 : 44, bh = 17;
          let extra = 0;
          if (mode === 'vllm' && lab === 'DSA') {
            const yb = y0 + 4 * (lh + gapY);
            if (py > yb + lh * 0.5) extra = bw * (64 / 512) * 1.6;
          }
          const bx = pad + laneW / 2 - bw / 2;
          const alpha = prog[k] < 0.06 ? prog[k] / 0.06 : prog[k] > 0.94 ? (1 - prog[k]) / 0.06 : 1;
          g.globalAlpha = alpha;
          V.box(g, bx, py - bh / 2, bw, bh, { r: 5, fill: V.alpha(col, 0.85), stroke: col, lw: 1 });
          if (extra) { hatch(g, bx + bw, py - bh / 2, extra, bh, C.d, 4); g.strokeStyle = C.d; g.lineWidth = 1; g.strokeRect(bx + bw, py - bh / 2, extra, bh); }
          V.text(g, lab, bx + bw / 2, py + 0.5, { size: 10.5, font: 'mono', weight: 700, align: 'center', base: 'middle', color: C.dark ? '#08130d' : '#fff' });
          if (mode === 'tf' && prog[k] > 0.8) mark(g, bx + bw + 9, py, true, 4.5);
          g.globalAlpha = 1;
        });

        // ---- linha do cache DSA
        let yb = Math.max(stackBot + 18, topH + 6);
        tfit(g, 'One DSA cache row (1 token, 1 layer), width ∝ bytes', pad, yb, w - 2 * pad, { size: 12, weight: 600, color: C.ink });
        yb += 10;
        const stripW = w - 2 * pad, sc = stripW / 689;
        ['vllm', 'tf'].forEach((key) => {
          const L = LAY[key], on = key === mode;
          const colK = key === 'tf' ? tf : C.glm;
          g.globalAlpha = on ? 1 : 0.38;
          V.text(g, L.name, pad, yb + 13, { size: 11.5, font: 'mono', weight: 600, color: colK });
          const sy = yb + 18, shh = 22;
          let x = pad;
          L.segs.forEach(([b, kind]) => {
            const ww = b * sc;
            if (kind === 'cells') {
              const cw = ww / 32;
              for (let i = 0; i < 32; i++) {
                const head = on ? ((t * 0.35) % 1) * 32 : -10;
                const lit = Math.max(0, 1 - Math.abs(i - head) / 3);
                g.fillStyle = V.cmap(0.25 + 0.55 * codes[(i + (key === 'tf' ? 7 : 0)) % 32] + lit * 0.2, 'seq');
                V.rr(g, x + i * cw + 0.5, sy, Math.max(1, cw - 1), shh, 2); g.fill();
              }
            } else if (kind === 'scale') {
              g.fillStyle = V.alpha(C.b, 0.75); g.fillRect(x, sy, Math.max(1.5, ww), shh);
            } else if (kind === 'zero') {
              g.fillStyle = V.alpha(C.d, 0.12); g.fillRect(x, sy, ww, shh);
              hatch(g, x, sy, ww, shh, V.alpha(C.d, 0.8), 5);
              if (ww > 40) V.text(g, '0 0 0', x + ww / 2, sy + shh / 2 + 0.5, { size: 10.5, font: 'mono', align: 'center', base: 'middle', color: C.d, weight: 700 });
            } else if (kind === 'pad') {
              hatch(g, x, sy, Math.max(2, ww), shh, V.alpha(C.muted, 0.7), 3);
            } else if (kind === 'idx') {
              g.fillStyle = V.alpha(C.e, 0.7); g.fillRect(x, sy, ww, shh);
            }
            x += ww;
          });
          g.strokeStyle = V.alpha(colK, 0.8); g.lineWidth = 1.2; g.strokeRect(pad, sy, L.total * sc, shh);
          V.wrap(g, L.legend, pad, sy + shh + 14, stripW, { size: 11, color: C.muted, lineH: 13, maxLines: 2 });
          g.globalAlpha = 1;
          yb = sy + shh + (narrow && tw(g, L.legend, 11) > stripW ? 34 : 20);
        });
      });
    })();

    /* =================================================== FIG 2 — bits */
    (function () {
      const fig = root.querySelector('#fig-bits');
      adaptH(fig, 400, 462);
      // produtos fp32 sintéticos (semente 924: quatro somas distintas para S = 1, 2, 4, 8)
      const r = rng(924);
      const P = [];
      for (let k = 0; k < 16; k++) { const mag = [0.001, 0.01, 0.1, 1, 4][Math.floor(r() * 5)]; P.push(f32(r.normal() * mag)); }
      function split(S) {
        const c = 16 / S, parts = [];
        for (let s = 0; s < S; s++) { let a = f32(0); for (let k = s * c; k < (s + 1) * c; k++) a = f32(a + P[k]); parts.push(a); }
        let tot = f32(0); parts.forEach((x) => { tot = f32(tot + x); });
        return { parts, tot };
      }
      const SUMS = {}; [1, 2, 4, 8].forEach((S) => { SUMS[S] = split(S); });
      const ulp = Math.pow(2, -20);
      const B = f32(Math.max(SUMS[1].tot, SUMS[2].tot) + ulp); // “verde”: entre os dois grupos
      const S = { M: 1, mode: 'lote', t0: 0 };
      const splitOf = (M) => (S.mode === 'fixo' ? 4 : M === 1 ? 8 : M <= 3 ? 4 : M <= 7 ? 2 : 1);
      const ctl = UI.controls(fig);
      UI.slider(ctl, { label: 'rows in batch M', min: 1, max: 16, step: 1, value: 1, onInput: (v) => { S.M = v; S.t0 = performance.now(); upd(); } });
      UI.seg(ctl, { label: 'Kernel', options: [{ v: 'lote', label: 'picks by batch' }, { v: 'fixo', label: 'row-independent' }], value: 'lote', onChange: (v) => { S.mode = v; S.t0 = performance.now(); upd(); } });
      const rS = UI.readout(ctl, 'split-K', '—'), rA = UI.readout(ctl, 'logit “blue”', '—'), rT = UI.readout(ctl, 'token', '—');
      function upd() {
        const sp = splitOf(S.M), a = SUMS[sp].tot;
        rS.set('S = ' + sp);
        rA.set(sigs(a, 9));
        rT.set(a > B ? 'blue' : 'green', a > B ? 'ok' : 'warn');
      }
      upd();
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const narrow = w < 620, pad = 12;
        const sp = splitOf(S.M), res = SUMS[sp];
        const el = (performance.now() - S.t0) / 1000;
        // ---- lote (matriz de M linhas)
        let mx, my, mw, mh;
        if (narrow) { mx = pad + 70; my = 26; mw = w - mx - pad; mh = 64; }
        else { mx = pad + 70; my = 40; mw = w * 0.3 - 70; mh = h - 80; }
        V.text(g, 'batch (M = ' + S.M + ')', pad, narrow ? 16 : 22, { size: 12, weight: 600, color: C.ink });
        // no estreito, a altura reservada é fixa (64 px) e as linhas encolhem conforme M cresce
        const cw = mw / 16, chh = narrow ? Math.min(12, (mh - (S.M - 1)) / S.M) : Math.min(16, mh / 16);
        for (let i = 0; i < S.M; i++) {
          for (let j = 0; j < 16; j++) {
            const v = i === 0 ? 0.35 + 0.6 * Math.min(1, Math.abs(P[j]) / 4) : 0.15 + 0.3 * Math.abs(Math.sin(i * 7.1 + j * 3.3));
            g.fillStyle = i === 0 ? V.cmap(v, 'seq') : V.alpha(C.muted, 0.12 + 0.25 * v);
            V.rr(g, mx + j * cw + 0.5, my + i * (chh + 1), cw - 1, chh, 2); g.fill();
          }
        }
        g.strokeStyle = C.c; g.lineWidth = 2; V.rr(g, mx - 2, my - 2, mw + 3, chh + 4, 4); g.stroke();
        V.text(g, 'your request', mx - 6, my + chh / 2 + 1, { size: 11, align: 'right', base: 'middle', color: C.c, weight: 600 });
        if (S.M > 1) V.text(g, (S.M - 1) + (S.M === 2 ? ' other' : ' others'), mx - 6, my + Math.max(18, (chh + 1) * Math.min(S.M - 1, 3) + chh / 2), { size: 11, align: 'right', base: 'middle', color: C.muted });
        // ---- produtos e split-K
        let px, py, pw;
        if (narrow) { px = pad; py = my + mh + 26; pw = w - 2 * pad; }
        else { px = w * 0.3 + 16; py = 40; pw = w - px - pad; }
        V.text(g, 'logit of “blue” = Σ of 16 products (fp32), in ' + sp + (sp === 1 ? ' chunk' : ' chunks'), px, py - 12, { size: 12, weight: 600, color: C.ink, maxW: pw });
        const perRow = 8;
        const nRows = 2;
        const cW = pw / perRow, cH = narrow ? 30 : 34;
        const c = 16 / sp;
        const ph = (el * 1.4) % (c + 3);
        for (let k = 0; k < 16; k++) {
          const row = Math.floor(k / perRow), col = k % perRow;
          const x = px + col * cW, y = py + row * (cH + 34);
          const chunk = Math.floor(k / c), within = k % c;
          const hot = within <= ph && ph < c + 1;
          const v = P[k];
          g.fillStyle = V.alpha(C.series[chunk % 6], hot ? 0.34 : 0.14);
          V.rr(g, x + 1, y, cW - 2, cH, 5); g.fill();
          g.strokeStyle = V.alpha(C.series[chunk % 6], 0.8); g.lineWidth = within === Math.floor(ph) && ph < c ? 2.4 : 1; g.stroke();
          tfit(g, Math.abs(v) >= 1 ? v.toFixed(2) : v.toPrecision(2), x + cW / 2, y + cH / 2 + 1, cW - 4, { size: 11, min: 8.5, font: 'mono', align: 'center', base: 'middle', color: C.ink });
        }
        // colchetes dos pedaços e somas parciais
        for (let s = 0; s < sp; s++) {
          const k0 = s * c, k1 = (s + 1) * c - 1;
          const segs = [];
          for (let k = k0; k <= k1; k++) { const row = Math.floor(k / perRow); if (!segs[row]) segs[row] = [k, k]; else segs[row][1] = k; }
          segs.forEach((sg, row) => {
            if (!sg) return;
            const x1 = px + (sg[0] % perRow) * cW + 3, x2 = px + (sg[1] % perRow) * cW + cW - 3, y = py + row * (cH + 34) + cH + 5;
            g.strokeStyle = C.series[s % 6]; g.lineWidth = 1.6; g.beginPath(); g.moveTo(x1, y); g.lineTo(x1, y + 5); g.lineTo(x2, y + 5); g.lineTo(x2, y); g.stroke();
          });
          const last = segs[segs.length - 1];
          const lx = px + ((last[0] % perRow) * cW + (last[1] % perRow) * cW + cW) / 2, ly = py + (segs.length - 1) * (cH + 34) + cH + 22;
          tfit(g, sigs(res.parts[s], sp >= 8 ? 5 : 7), lx, ly, Math.max(40, Math.min(c, perRow) * cW - 4), { size: 10.5, min: 8.5, font: 'mono', align: 'center', color: C.series[s % 6] });
        }
        // resultado
        const ry = py + nRows * (cH + 34) + 8;
        const boxW = Math.min(pw, 420);
        const win = res.tot > B;
        const resFlash = Math.max(0, 1 - el * 1.2);
        V.box(g, px, ry, boxW, 44, { r: 8, fill: V.alpha(C.a, 0.1 + 0.2 * resFlash), stroke: V.alpha(C.a, 0.7) });
        V.text(g, 'blue', px + 10, ry + 17, { size: 12, font: 'mono', weight: 700, color: C.a });
        tfit(g, sigs(res.tot, 9) + '   ' + hex32(res.tot), px + 52, ry + 17, boxW - 60, { size: 12, font: 'mono', color: C.ink });
        V.text(g, 'green', px + 10, ry + 35, { size: 12, font: 'mono', weight: 700, color: C.e });
        tfit(g, sigs(B, 9) + '   ' + hex32(B), px + 52, ry + 35, boxW - 60, { size: 12, font: 'mono', color: C.ink });
        // veredito
        const vy = ry + 56;
        const lab = win ? 'argmax → “blue”' : 'argmax → “green”';
        chip(g, px, vy, Math.min(boxW, 190), 26, lab, { fill: V.alpha(win ? C.a : C.d, 0.2), stroke: win ? C.a : C.d, size: 12.5, weight: 700 });
        const note = S.mode === 'fixo' ? 'S fixed by the weight shape: same bits in any batch' : 'S set by the batch: the row\'s bits change with M';
        V.wrap(g, note, px + Math.min(boxW, 190) + 12, vy + 9, Math.max(120, pw - Math.min(boxW, 190) - 12), { size: 11.5, color: C.muted, lineH: 14, maxLines: 3 });
        const by2 = vy + (narrow ? 68 : 62);
        V.text(g, 'bits of the “blue” logit: alone (M = 1) × in this batch', px, by2 - 8, { size: 11.5, weight: 600, color: C.ink, maxW: pw });
        const nd = bitRows(g, px, by2, Math.min(pw, 460), SUMS[splitOf(1)].tot, res.tot, 'alone', 'batch M=' + S.M);
        V.text(g, nd ? nd + ' bit(s) differ: concurrent ≠ solo' : 'same 32 bits: concurrent = solo', px, by2 + 68, { size: 11.5, weight: 600, color: nd ? RED() : C.a });
      });
    })();

    /* =================================================== FIG 3 — amostragem chaveada */
    (function () {
      const fig = root.querySelector('#fig-chave');
      const stg3 = fig.querySelector('.la-fig-stage');
      const setH3 = () => { const nar = fig.clientWidth < 620; const want = (S.mode === 'verif' ? (nar ? 345 : 262) : (nar ? 640 : 460)) + 'px'; if (stg3.style.height !== want) stg3.style.height = want; };
      const MASK = (1n << 64n) - 1n;
      const sm = (x) => { let z = (x + 0x9E3779B97F4A7C15n) & MASK; z = ((z ^ (z >> 30n)) * 0xBF58476D1CE4E5B9n) & MASK; z = ((z ^ (z >> 27n)) * 0x94D049BB133111EBn) & MASK; return z ^ (z >> 31n); };
      const H = (s, p, id) => { const x = sm(sm(sm(s) ^ BigInt(p)) ^ BigInt(id)); return (Number(x >> 11n) + 0.5) / 9007199254740992; };
      const hashStr = (str) => { let h = 0xcbf29ce484222325n; for (const ch of str) { h ^= BigInt(ch.codePointAt(0)); h = (h * 0x100000001b3n) & MASK; } return h & ((1n << 63n) - 1n); };
      const KEYS = { A: { label: 'prompt A', s: hashStr('Explique o KV cache.') }, B: { label: 'prompt B', s: hashStr('Explique o KV cache!') }, K: { label: 'seed = 42', s: 42n } };
      const WORDS = [
        ['the', 'a', 'this', 'each', 'your', 'our'], ['exact', 'new', 'fast', 'dedicated', 'right', 'good'],
        ['engine', 'model', 'server', 'system', 'code', 'kernel'], ['always', 'today', 'now', 'soon', 'already', 'still'],
        ['returns', 'makes', 'writes', 'delivers', 'gives', 'shows'], ['the', 'a', 'that', 'its', 'every', 'another'],
        ['same', 'new', 'good', 'long', 'only', 'right'], ['answer', 'phrase', 'output', 'line', 'idea', 'sum'],
        ['for', 'to', 'with', 'on', 'by', 'under'], ['the', 'a', 'each', 'that', 'its', 'every'],
        ['same', 'new', 'long', 'old', 'short', 'good'], ['prompt', 'request', 'text', 'user', 'case', 'context'],
      ];
      const NPOS = WORDS.length;
      const lr = rng(48);
      const LOG = WORDS.map(() => { const alt = [2.3, 1.6, 1.0, 0.5, -0.1]; for (let i = alt.length - 1; i > 0; i--) { const j = Math.floor(lr() * (i + 1)); [alt[i], alt[j]] = [alt[j], alt[i]]; } return [3.0 + lr() * 0.5, ...alt.map((a) => a + lr() * 0.3)]; });
      const IDS = WORDS.map((ws, p) => ws.map((_, i) => 1000 + p * 17 + i * 3));
      const softmax = (zs, T) => { const m = Math.max(...zs.map((z) => z / T)); const e = zs.map((z) => Math.exp(z / T - m)); const s = e.reduce((a, b) => a + b, 0); return e.map((x) => x / s); };
      const gum = (u) => -Math.log(-Math.log(u));
      const S = { mode: 'gumbel', T: 1, key: 'A', p: 3, stateful: false, draws: 0, tDraw: -9, same: null, hist: [0, 0, 0, 0, 0, 0], nHist: 0, histKey: 0, lam: 0.7, ver: 'igual', sim: null, tSim: 0 };
      let curU = null;
      setH3();
      { const ro = new ResizeObserver(setH3); ro.observe(fig); if (LA.onCleanup) LA.onCleanup(() => ro.disconnect()); }
      function keyedU(p, s) { return IDS[p].map((id) => H(s, p, id)); }
      function drawU() {
        if (S.stateful) { const rr = rng(1 + S.draws * 7919); curU = IDS[S.p].map(() => rr()); }
        else curU = keyedU(S.p, KEYS[S.key].s);
      }
      drawU();
      function resetHist() { S.hist = [0, 0, 0, 0, 0, 0]; S.nHist = 0; S.histKey = 0; }
      function addKeys(n) {
        const zs = LOG[S.p];
        for (let q = 0; q < n; q++) {
          const s = sm(KEYS[S.key].s ^ BigInt(++S.histKey * 7777));
          let best = -1, bv = -Infinity;
          for (let i = 0; i < 6; i++) { const v = zs[i] / S.T + gum(H(s, S.p, IDS[S.p][i])); if (v > bv) { bv = v; best = i; } }
          S.hist[best]++; S.nHist++;
        }
      }
      // ---------- simulação da verificação
      function simulate() {
        const T = 1, s = KEYS.A.s, k = 4;
        const P = LOG.map((zs) => softmax(zs, T));
        const Q = P.map((p) => p.map((x) => S.lam * x + (1 - S.lam) / 6));
        const serialKeyed = LOG.map((zs, p) => { let b = 0, bv = -Infinity; zs.forEach((z, i) => { const v = z / T + gum(H(s, p, IDS[p][i])); if (v > bv) { bv = v; b = i; } }); return b; });
        const cat = (d, u) => { let a = 0; for (let i = 0; i < d.length; i++) { a += d[i]; if (u < a) return i; } return d.length - 1; };
        let serial, out = [], rounds = [];
        if (S.ver === 'igual') {
          serial = serialKeyed;
          out.push(serial[0]);
          let pos = 0;
          while (pos < NPOS - 1) {
            const drafts = [], checks = [];
            for (let j = 1; j <= k && pos + j < NPOS; j++) {
              const p = pos + j;
              let b = 0, bv = -Infinity;
              Q[p].forEach((q, i) => { const v = Math.log(q) + gum(H(s, p, IDS[p][i])); if (v > bv) { bv = v; b = i; } });
              drafts.push(b);
            }
            let a = 0; while (a < drafts.length && drafts[a] === serial[pos + 1 + a]) a++;
            drafts.forEach((d, j) => checks.push(j < a ? 1 : j === a ? 0 : -1));
            const emit = [];
            for (let j = 1; j <= a + 1 && pos + j < NPOS; j++) emit.push(serial[pos + j]);
            rounds.push({ pos, drafts, checks, emit, fix: a < drafts.length ? serial[pos + a + 1] : null });
            emit.forEach((x) => out.push(x));
            pos += emit.length;
          }
        } else {
          const r1 = rng(7); serial = P.map((p) => cat(p, r1()));
          const r2 = rng(7);
          out.push(cat(P[0], r2()));
          let pos = 0;
          while (pos < NPOS - 1) {
            const drafts = [], checks = [], emit = [];
            const n = Math.min(k, NPOS - 1 - pos);
            for (let j = 1; j <= n; j++) drafts.push(cat(Q[pos + j], r2()));
            let rejected = false, fix = null;
            for (let j = 0; j < drafts.length; j++) {
              const p = pos + 1 + j, d = drafts[j];
              if (r2() < Math.min(1, P[p][d] / Q[p][d])) { checks.push(1); emit.push(d); }
              else {
                checks.push(0); rejected = true;
                const res = P[p].map((x, i) => Math.max(0, x - Q[p][i])); const z = res.reduce((a, b) => a + b, 0) || 1;
                fix = cat(res.map((x) => x / z), r2()); emit.push(fix);
                for (let jj = j + 1; jj < drafts.length; jj++) checks.push(-1);
                break;
              }
            }
            if (!rejected && pos + drafts.length + 1 < NPOS) { fix = cat(P[pos + drafts.length + 1], r2()); emit.push(fix); }
            rounds.push({ pos, drafts, checks, emit, fix });
            emit.forEach((x) => out.push(x));
            pos += emit.length;
          }
        }
        out = out.slice(0, NPOS);
        const same = out.every((x, i) => x === serial[i]);
        S.sim = { serial, out, rounds, same };
        S.tSim = performance.now();
        rR.set(String(rounds.length)); rTk.set(fmt((NPOS - 1) / rounds.length, 2));
        rEq.set(same ? 'yes, bit for bit' : 'no (same distribution)', same ? 'ok' : 'warn');
      }
      // ---------- controles
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Mode', options: [{ v: 'gumbel', label: 'Gumbel-max' }, { v: 'verif', label: 'verify' }], value: 'gumbel', onChange: (v) => { S.mode = v; gA.show(v === 'gumbel'); gB.show(v === 'verif'); if (v === 'verif') simulate(); setH3(); } });
      const gA = UI.group(ctl), gB = UI.group(ctl);
      nocase(UI.slider(gA.el, { label: 'T', min: 0.3, max: 2, step: 0.1, value: 1, fmt: (v) => fmt(v, 1), onInput: (v) => { S.T = v; resetHist(); } }));
      UI.seg(gA.el, { label: 'key', options: [{ v: 'A', label: 'prompt A' }, { v: 'B', label: 'prompt B' }, { v: 'K', label: 'seed 42' }], value: 'A', onChange: (v) => { S.key = v; S.stateful = false; tg.set(false); drawU(); S.tDraw = t0(); resetHist(); } });
      nocase(UI.slider(gA.el, { label: 'position p', min: 0, max: NPOS - 1, step: 1, value: 3, onInput: (v) => { S.p = v; drawU(); S.tDraw = t0(); resetHist(); } }));
      const tg = UI.toggle(gA.el, { label: 'stateful generator', value: false, onChange: (v) => { S.stateful = v; drawU(); S.tDraw = t0(); } });
      UI.button(gA.el, '↻ draw again', () => { const prev = argmaxNow(); S.draws++; drawU(); S.tDraw = t0(); S.same = argmaxNow() === prev; }, { primary: true });
      UI.button(gA.el, '+500 keys', () => addKeys(500));
      nocase(UI.slider(gB.el, { label: 'drafter quality λ', min: 0, max: 1, step: 0.05, value: 0.7, fmt: (v) => fmt(v, 2), onInput: (v) => { S.lam = v; simulate(); } }));
      UI.seg(gB.el, { label: 'verification', options: [{ v: 'igual', label: 'equality (key)' }, { v: 'rej', label: 'rejection (state)' }], value: 'igual', onChange: (v) => { S.ver = v; simulate(); } });
      const rR = UI.readout(gB.el, 'rounds', '—'), rTk = UI.readout(gB.el, 'tokens/round', '—'), rEq = UI.readout(gB.el, 'output = serial?', '—');
      gB.show(false);
      addKeys(500);
      function t0() { return performance.now() / 1000; }
      function argmaxNow() { const zs = LOG[S.p]; let b = 0, bv = -Infinity; zs.forEach((z, i) => { const v = z / S.T + gum(curU[i]); if (v > bv) { bv = v; b = i; } }); return b; }
      simulate();

      V.fig(fig, (g, w, h) => {
        const C = V.C();
        const narrow = w < 620, pad = 12;
        const now = performance.now() / 1000;
        if (S.mode === 'gumbel') {
          const zs = LOG[S.p], ws = WORDS[S.p];
          const ga = curU.map(gum);
          const grow = V.ease.out(Math.min(1, (now - S.tDraw) / 0.7));
          const tot = zs.map((z, i) => z / S.T + ga[i] * grow);
          let best = 0; tot.forEach((v, i) => { if (v > tot[best]) best = i; });
          // painel das barras
          const bx = pad, by = 44, bw = narrow ? w - 2 * pad : w * 0.58 - pad, bh = narrow ? h * 0.42 : h - 120;
          const keyTxt = S.stateful ? 'stateful generator: draw #' + (S.draws + 1) : 's = ' + KEYS[S.key].label + ' → 0x' + KEYS[S.key].s.toString(16).slice(0, 10) + '…';
          tfit(g, keyTxt, pad, 16, w - 2 * pad, { size: 11.5, font: 'mono', color: S.stateful ? C.d : C.b });
          tfit(g, S.stateful ? 'u = next number from the generator (depends on history)' : 'u = H(s, p = ' + S.p + ', id)  →  g = −ln(−ln u)', pad, 32, w - 2 * pad, { size: 11.5, font: 'mono', color: C.muted });
          const all = zs.map((z, i) => [z / S.T, z / S.T + ga[i]]).flat();
          const vmin = Math.min(0, ...all) - 0.4, vmax = Math.max(...all) + 0.6;
          const yOf = (v) => by + bh - ((v - vmin) / (vmax - vmin)) * (bh - 30);
          const y0v = yOf(0);
          V.line(g, bx, y0v, bx + bw, y0v, { color: V.alpha(C.muted, 0.6), lw: 1 });
          const colW = bw / 6;
          for (let i = 0; i < 6; i++) {
            const cx = bx + colW * i + colW / 2, bwid = Math.min(34, colW * 0.5);
            const zt = zs[i] / S.T;
            g.fillStyle = V.alpha(C.a, 0.75); V.rr(g, cx - bwid / 2, Math.min(yOf(zt), y0v), bwid, Math.abs(yOf(zt) - y0v), 3); g.fill();
            const gy1 = yOf(zt), gy2 = yOf(tot[i]);
            g.fillStyle = V.alpha(C.b, 0.55); g.fillRect(cx - bwid / 2 + 4, Math.min(gy1, gy2), bwid - 8, Math.abs(gy2 - gy1));
            g.strokeStyle = C.b; g.lineWidth = 1.2; g.strokeRect(cx - bwid / 2 + 4, Math.min(gy1, gy2), bwid - 8, Math.abs(gy2 - gy1));
            V.dot(g, cx, gy2, 4, i === best ? C.d : C.ink);
            if (i === best) {
              V.glow(g, cx, gy2, 26, C.d, 0.6);
              const cy = gy2 - 16; g.fillStyle = C.d; g.beginPath(); g.moveTo(cx - 9, cy + 6); g.lineTo(cx - 9, cy - 3); g.lineTo(cx - 4.5, cy + 1); g.lineTo(cx, cy - 6); g.lineTo(cx + 4.5, cy + 1); g.lineTo(cx + 9, cy - 3); g.lineTo(cx + 9, cy + 6); g.closePath(); g.fill();
            }
            tfit(g, ws[i], cx, by + bh + 16, colW - 2, { size: 11.5, min: 9, font: 'mono', align: 'center', weight: i === best ? 700 : 400, color: i === best ? C.d : C.ink });
            tfit(g, 'g ' + (ga[i] >= 0 ? '+' : '') + fmt(ga[i], 2), cx, by + bh + 31, colW - 2, { size: 10.5, min: 9, font: 'mono', align: 'center', color: C.b });
          }
          if (S.same != null && now - S.tDraw < 2.2) {
            const ok = S.same && !S.stateful;
            chip(g, bx, by + bh + 38, Math.min(bw, 230), 22, S.stateful ? (S.same ? 'same token, by chance' : 'another token: the state moved on') : 'same key → same token ✓', { fill: V.alpha(ok ? C.a : C.d, 0.2), stroke: ok ? C.a : C.d, size: 11, font: 'sans', weight: 600 });
          }
          // painel do histograma
          const hx = narrow ? pad : w * 0.6 + 8, hy = narrow ? by + bh + 90 : 60, hw = narrow ? w - 2 * pad : w - hx - pad, hh = narrow ? h - (by + bh + 90) - 30 : h - 128;
          V.text(g, 'argmax over ' + fmtInt(S.nHist) + ' keys vs softmax(z/T)', hx, hy - 10, { size: 11.5, weight: 600, color: C.ink, maxW: hw });
          const pr = softmax(zs, S.T);
          const freq = S.hist.map((c) => (S.nHist ? c / S.nHist : 0));
          const mx = Math.max(...pr, ...freq, 0.05) * 1.1;
          const hbw = hw / 6;
          for (let i = 0; i < 6; i++) {
            const x = hx + i * hbw + hbw * 0.18, ww = hbw * 0.64;
            const fh = (freq[i] / mx) * (hh - 14);
            g.fillStyle = V.alpha(C.d, 0.7); V.rr(g, x, hy + hh - fh, ww, Math.max(1, fh), 3); g.fill();
            const py = hy + hh - (pr[i] / mx) * (hh - 14);
            V.dot(g, x + ww / 2, py, 4.5, C.a);
            g.strokeStyle = C.a; g.lineWidth = 1.2; g.beginPath(); g.moveTo(x - 2, py); g.lineTo(x + ww + 2, py); g.stroke();
            tfit(g, ws[i], x + ww / 2, hy + hh + 14, hbw - 2, { size: 10.5, min: 8.5, font: 'mono', align: 'center', color: C.muted });
          }
          g.strokeStyle = V.alpha(C.muted, 0.5); g.lineWidth = 1; g.beginPath(); g.moveTo(hx, hy + hh + 0.5); g.lineTo(hx + hw, hy + hh + 0.5); g.stroke();
          if (!S.nHist) V.wrap(g, 'click “+500 keys”: each key is another prompt at the same position', hx + 6, hy + 14, hw - 12, { size: 11, color: C.muted, lineH: 14, maxLines: 3 });
          return;
        }
        // ---------------- modo verificação
        const sim = S.sim; if (!sim) return;
        const perRound = 0.9 + 0.38 * 4 + 0.6;
        const total = sim.rounds.length * perRound + 2.6;
        const el = ((performance.now() - S.tSim) / 1000) % total;
        const ri = Math.min(sim.rounds.length - 1, Math.floor(el / perRound));
        const rt = el - ri * perRound;
        const done = el >= sim.rounds.length * perRound;
        const perRow = narrow ? 6 : 12, nRows = NPOS / perRow;
        const cw = (w - 2 * pad) / perRow, ch = 24;
        const posX = (p) => pad + (p % perRow) * cw, rowOf = (p) => Math.floor(p / perRow);
        const word = (p, i) => WORDS[p][i];
        // quantos tokens já saíram
        let emitted = 1; for (let k = 0; k < ri; k++) emitted += sim.rounds[k].emit.length;
        if (done) emitted = NPOS; else if (rt > 0.9 + 0.38 * 4) emitted += sim.rounds[ri].emit.length;
        emitted = Math.min(NPOS, emitted);
        const bandH = nRows * (ch + 6);
        const yS = 32, yD = yS + bandH + 26, yO = yD + bandH + 26;
        V.text(g, 'serial (draft: false), key = prompt A', pad, yS - 8, { size: 11.5, weight: 600, color: C.muted });
        V.text(g, done ? 'drafts' : 'drafts in round ' + (ri + 1), pad, yD - 8, { size: 11.5, weight: 600, color: C.b });
        V.text(g, 'output with drafts', pad, yO - 8, { size: 11.5, weight: 600, color: C.a });
        for (let p = 0; p < NPOS; p++) {
          const x = posX(p), y = yS + rowOf(p) * (ch + 6);
          chip(g, x + 1, y, cw - 3, ch, word(p, sim.serial[p]), { fill: V.alpha(C.muted, 0.1), stroke: V.alpha(C.muted, 0.35), size: 11, min: 8.5 });
          const yo = yO + rowOf(p) * (ch + 6);
          if (p < emitted) {
            const ok = sim.out[p] === sim.serial[p];
            chip(g, x + 1, yo, cw - 3, ch, word(p, sim.out[p]), { fill: V.alpha(ok ? C.a : RED(), 0.18), stroke: ok ? C.a : RED(), size: 11, min: 8.5 });
          } else {
            g.strokeStyle = V.alpha(C.muted, 0.25); g.setLineDash([3, 3]); V.rr(g, x + 1, yo, cw - 3, ch, 6); g.stroke(); g.setLineDash([]);
          }
        }
        if (!done) {
          const R = sim.rounds[ri];
          // janela atual
          const wx = posX(R.pos), wy = yO + rowOf(R.pos) * (ch + 6);
          g.strokeStyle = C.c; g.lineWidth = 2; V.rr(g, wx - 1, wy - 3, cw + 1, ch + 6, 7); g.stroke();
          R.drafts.forEach((d, j) => {
            const p = R.pos + 1 + j;
            const appear = Math.min(1, Math.max(0, (rt - j * 0.12) / 0.3));
            if (appear <= 0) return;
            const x = posX(p), y = yD + rowOf(p) * (ch + 6);
            g.globalAlpha = appear;
            const chk = rt > 0.9 + 0.38 * j ? R.checks[j] : null;
            chip(g, x + 1, y, cw - 3, ch, word(p, d), { fill: V.alpha(C.b, 0.2), stroke: chk === 0 ? RED() : chk === -1 ? V.alpha(C.muted, 0.4) : C.b, size: 11, min: 8.5 });
            if (chk === 1) mark(g, x + cw - 10, y + 4, true, 4.5);
            if (chk === 0) mark(g, x + cw - 10, y + 4, false, 4.5);
            g.globalAlpha = 1;
          });
          V.wrap(g, S.ver === 'igual' ? 'accept while draft = keyed sample; at the 1st disagreement the target sample is emitted' : 'accept with min(1, p/q) using the generator; on rejection, draw from the residual', pad, yO + bandH + 18, w - 2 * pad, { size: 11, color: C.muted, lineH: 14, maxLines: 2 });
        } else {
          const ok = sim.same;
          chip(g, pad, yO + bandH + 6, Math.min(w - 2 * pad, 330), 26, ok ? 'output = serial, bit for bit ✓ (same hash)' : 'output ≠ serial: same distribution, different sequence', { fill: V.alpha(ok ? C.a : C.d, 0.2), stroke: ok ? C.a : C.d, size: 11.5, font: 'sans', weight: 700 });
          const hs = (arr) => '0x' + hashStr(arr.map((x, p) => WORDS[p][x]).join(' ')).toString(16).padStart(16, '0').slice(0, 12);
          tfit(g, 'hash(serial) = ' + hs(sim.serial), pad, yO + bandH + 52, w - 2 * pad, { size: 11.5, font: 'mono', color: C.muted });
          tfit(g, 'hash(output) = ' + hs(sim.out), pad, yO + bandH + 68, w - 2 * pad, { size: 11.5, font: 'mono', color: ok ? C.a : C.d });
        }
      });
    })();

    /* =================================================== FIG 4 — parciais fp32, um arredondamento */
    (function () {
      const fig = root.querySelector('#fig-soma');
      adaptH(fig, 350, 590);
      const bfEnc = (x) => LA.num.encode('bf16', x);
      const bf = (x) => bfEnc(x).v;
      // parciais fp32 sintéticas (semente 18: com bf16 + bf16, 2 dos 6 elementos saem diferentes da referência)
      const r = rng(18);
      const P0 = [], P1 = [];
      for (let i = 0; i < 6; i++) { P0.push(f32(r.normal() * 0.7)); P1.push(f32(r.normal() * 0.7)); }
      const REF = P0.map((x, i) => bf(x + P1[i]));             // soma exata (double), arredondada uma vez
      const TF01 = P0.map((x, i) => bf(f32(x + P1[i])));       // rank 0 primeiro, fp32, um arredondamento
      const TF10 = P0.map((x, i) => bf(f32(P1[i] + x)));       // a outra ordem (comutativa: mesmos bits)
      const VL01 = P0.map((x, i) => bf(bf(x) + bf(P1[i])));    // all-reduce de ativações bf16
      const VL10 = P0.map((x, i) => bf(bf(P1[i]) + bf(x)));
      const IDX = Math.max(0, VL01.findIndex((v, i) => v !== REF[i]));
      const S = { red: 'tf', order: 'rank', tr: 'nccl', t0: performance.now() - 4200 };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Reduction', options: [{ v: 'tf', label: 'TensorFold: fp32, one rounding' }, { v: 'vl', label: 'vLLM: all-reduce bf16' }], value: 'tf', onChange: (v) => { S.red = v; S.t0 = performance.now(); gT.show(v === 'tf'); upd(); } });
      UI.seg(ctl, { label: 'Sum order', options: [{ v: 'rank', label: 'rank 0 first' }, { v: 'local', label: 'each adds its own first' }], value: 'rank', onChange: (v) => { S.order = v; S.t0 = performance.now(); upd(); } });
      const gT = UI.group(ctl);
      UI.seg(gT.el, { label: 'Transport', options: [{ v: 'nccl', label: 'NCCL ~45 µs' }, { v: 'roce', label: 'RoCE ~11 µs' }], value: 'nccl', onChange: (v) => { S.tr = v; S.t0 = performance.now(); } });
      const rK = UI.readout(ctl, 'rank 0 × rank 1', '—'), rD = UI.readout(ctl, 'Δ ≠ reference', '—'), rW = UI.readout(ctl, 'on the wire, per row', '—');
      const resOf = (rk) => (S.red === 'tf' ? (rk === 1 && S.order === 'local' ? TF10 : TF01) : (rk === 1 && S.order === 'local' ? VL10 : VL01));
      function upd() {
        const a = resOf(0), b = resOf(1);
        const same = a.every((x, i) => Object.is(x, b[i]));
        const d = a.filter((x, i) => x !== REF[i]).length;
        rK.set(same ? 'bitwise identical' : 'diverge', same ? 'ok' : 'bad');
        rD.set(d + ' of 6', d ? 'bad' : 'ok');
        rW.set(S.red === 'tf' ? '16 KiB (fp32)' : '8 KiB (bf16)', S.red === 'tf' ? 'warn' : null);
      }
      upd();
      /** linhas de bits (bf16: 1 + 8 + 7), diferenças marcadas contra a última linha */
      function bitTable(g, x, y, w, rows) {
        const C = V.C();
        const nb = 16, eb = 8, labW = 92, cw = (w - labW) / nb, chh = 14;
        const B = rows.map(([u]) => { const b = []; for (let i = nb - 1; i >= 0; i--) b.push((u >>> i) & 1); return b; });
        const ref = B[B.length - 1];
        rows.forEach(([, lab], k) => {
          const yy = y + k * (chh + 4);
          tfit(g, lab, x, yy + chh - 2, labW - 6, { size: 11, min: 9, color: k === rows.length - 1 ? C.ink : C.muted, weight: k === rows.length - 1 ? 600 : 400 });
          B[k].forEach((bt, i) => {
            const col = i === 0 ? C.c : i <= eb ? C.b : C.a;
            g.fillStyle = bt ? V.alpha(col, 0.85) : V.alpha(col, 0.12);
            V.rr(g, x + labW + i * cw + 0.5, yy, cw - 1, chh, 2); g.fill();
            if (k < rows.length - 1 && bt !== ref[i]) { g.strokeStyle = RED(); g.lineWidth = 2; V.rr(g, x + labW + i * cw, yy - 1, cw, chh + 2, 2); g.stroke(); }
          });
        });
        const yl = y + rows.length * (chh + 4) + 11;
        V.text(g, 's', x + labW + cw / 2, yl, { size: 10, font: 'mono', align: 'center', color: C.c });
        V.text(g, 'exponent', x + labW + cw * 5, yl, { size: 10, align: 'center', color: C.b });
        V.text(g, 'mantissa (7)', x + labW + cw * 12.5, yl, { size: 10, align: 'center', color: C.a });
      }
      V.fig(fig, (g, w, h) => {
        const C = V.C(), tf = TFC();
        const narrow = w < 640, pad = 12;
        const el = (performance.now() - S.t0) / 1000;
        const isTF = S.red === 'tf';
        const fly = isTF && S.tr === 'roce' ? 0.5 : 1.4;
        const T1 = 0.8, T2 = T1 + fly, T3 = T2 + 1.6, CYC = T3 + 3.2;
        const tc = el % CYC;
        const R0 = resOf(0), R1 = resOf(1);
        // geometria dos painéis
        const rowH = narrow ? 22 : 24;
        let pw, ph = 62 + 6 * rowH, p0x, p0y, p1x, p1y;
        if (narrow) { pw = w - 2 * pad; p0x = pad; p0y = 8; p1x = pad; p1y = 8 + ph + 54; }
        else { pw = (w - 2 * pad - 76) / 2; p0x = pad; p0y = 8; p1x = pad + pw + 76; p1y = 8; }
        const midY = 8 + ph / 2;
        const fmtIn = isTF ? 'fp32' : 'bf16';
        const cols = ['P⁰ ' + fmtIn, 'P¹ ' + fmtIn, 'Δ (bf16)'];
        const fr = [0, 0.33, 0.66], fw = [0.33, 0.33, 0.34];
        const colX = (px, j) => px + 10 + fr[j] * (pw - 20);
        const colW = (j) => fw[j] * (pw - 20) - 4;
        const rowY = (py, i) => py + 50 + i * rowH;
        const shownP = (arr, i) => (isTF ? sigs(arr[i], 7) : sigs(bf(arr[i]), 4));
        const panel = (rk, px, py) => {
          const accent = rk === 0 ? C.a : C.b;
          const mine = rk === 0 ? R0 : R1;
          V.box(g, px, py, pw, ph, { r: 12, fill: V.alpha(accent, 0.05), stroke: V.alpha(accent, 0.5), lw: 1.2 });
          V.text(g, 'rank ' + rk + ' · Spark ' + (rk === 0 ? 'A' : 'B'), px + 10, py + 17, { size: 12, weight: 700, color: accent });
          const first = rk === 1 && S.order === 'local' ? ['P¹', 'P⁰'] : ['P⁰', 'P¹'];
          const formula = isTF ? 'Δ = bf16(' + first[0] + ' ⊕ ' + first[1] + ')' : 'Δ = bf16(bf16 ' + first[0] + ' + bf16 ' + first[1] + ')';
          tfit(g, formula, px + pw - 10, py + 17, pw - 120, { size: 11.5, min: 9, font: 'mono', align: 'right', color: C.ink });
          cols.forEach((c, j) => tfit(g, c, colX(px, j) + colW(j) / 2, py + 40, colW(j), { size: 11, min: 9, font: 'mono', weight: 600, align: 'center', color: j === 2 ? C.d : C.muted }));
          const local = rk === 0 ? 0 : 1, remote = rk === 0 ? 1 : 0;
          const vals = [P0, P1];
          for (let i = 0; i < 6; i++) {
            const y = rowY(py, i);
            const cell = (j, txt, col, a = 1, o = {}) => { g.globalAlpha = a; V.box(g, colX(px, j), y, colW(j), rowH - 3, { r: 4, fill: V.alpha(col, o.fill == null ? 0.14 : o.fill), stroke: o.stroke || V.alpha(col, 0.5), lw: o.lw || 1 }); if (txt) tfit(g, txt, colX(px, j) + colW(j) / 2, y + (rowH - 3) / 2 + 1, colW(j) - 4, { size: 11, min: 8.5, font: 'mono', align: 'center', base: 'middle', color: o.color || C.ink }); g.globalAlpha = 1; };
            cell(local, shownP(vals[local], i), local === 0 ? C.a : C.b, tc < T1 ? 0.5 + 0.5 * Math.sin(tc * 8) ** 2 : 1);
            cell(remote, tc > T2 ? shownP(vals[remote], i) : '', remote === 0 ? C.a : C.b, tc > T2 ? 1 : 0.35);
            const sumP = (tc - T2) / (T3 - T2) * 6;
            if (tc > T2 && i < sumP) {
              const bad = tc > T3 && mine[i] !== REF[i];
              cell(2, sigs(mine[i], 4), C.d, 1, { fill: bad ? 0.2 : 0.16, stroke: bad ? RED() : V.alpha(C.d, 0.7), lw: bad ? 2 : 1, color: bad ? RED() : C.ink });
              if (tc > T3) mark(g, colX(px, 2) + colW(2) - 9, y + (rowH - 3) / 2, !bad, 4);
            } else cell(2, '', C.d, 0.3);
          }
        };
        panel(0, p0x, p0y); panel(1, p1x, p1y);
        // cabo e voo das parciais
        const cab = narrow ? [[w / 2 - 30, p0y + ph + 4], [w / 2 - 30, p1y - 4], [w / 2 + 30, p0y + ph + 4], [w / 2 + 30, p1y - 4]] : [[p0x + pw + 4, midY - 10], [p1x - 4, midY - 10], [p0x + pw + 4, midY + 10], [p1x - 4, midY + 10]];
        g.strokeStyle = V.alpha(isTF ? tf : C.glm, 0.7); g.lineWidth = 2;
        g.beginPath(); g.moveTo(cab[0][0], cab[0][1]); g.lineTo(cab[1][0], cab[1][1]); g.moveTo(cab[2][0], cab[2][1]); g.lineTo(cab[3][0], cab[3][1]); g.stroke();
        const opName = isTF ? 'all-gather' : 'all-reduce';
        const wire = isTF ? '16 KiB/row' : '8 KiB/row';
        const lat = isTF ? (S.tr === 'roce' ? 'RoCE 11 µs' : 'NCCL 45 µs') : 'NCCL';
        if (narrow) {
          const my = (p0y + ph + p1y) / 2;
          tfit(g, opName + ' ' + fmtIn, w / 2 + 40, my - 9, w / 2 - 40 - pad, { size: 11, min: 8.5, font: 'mono', color: isTF ? tf : C.glm });
          tfit(g, lat, w / 2 + 40, my + 4, w / 2 - 40 - pad, { size: 10.5, min: 8.5, font: 'mono', color: C.muted });
          tfit(g, wire, w / 2 + 40, my + 17, w / 2 - 40 - pad, { size: 10.5, min: 8.5, font: 'mono', color: C.muted });
        } else {
          tfit(g, opName, w / 2, midY - 30, 72, { size: 10.5, min: 8.5, font: 'mono', align: 'center', color: isTF ? tf : C.glm });
          tfit(g, fmtIn, w / 2, midY - 17, 72, { size: 10.5, min: 8.5, font: 'mono', align: 'center', color: isTF ? tf : C.glm });
          tfit(g, lat, w / 2, midY + 28, 72, { size: 10.5, min: 8.5, font: 'mono', align: 'center', color: C.muted });
          tfit(g, wire, w / 2, midY + 42, 72, { size: 10, min: 8, font: 'mono', align: 'center', color: C.muted });
        }
        if (tc > T1 && tc < T2) {
          const u = V.ease.inOut((tc - T1) / fly);
          for (let i = 0; i < 6; i++) {
            const a = [colX(p0x, 0) + colW(0) / 2, rowY(p0y, i) + rowH / 2], b = [colX(p1x, 0) + colW(0) / 2, rowY(p1y, i) + rowH / 2];
            const c2 = [colX(p1x, 1) + colW(1) / 2, rowY(p1y, i) + rowH / 2], d = [colX(p0x, 1) + colW(1) / 2, rowY(p0y, i) + rowH / 2];
            const mid1 = narrow ? [w / 2 - 30, (a[1] + b[1]) / 2] : [(a[0] + b[0]) / 2, midY - 10];
            const mid2 = narrow ? [w / 2 + 30, (c2[1] + d[1]) / 2] : [(a[0] + b[0]) / 2, midY + 10];
            const q1 = V.bez(a, mid1, mid1, b, u), q2 = V.bez(c2, mid2, mid2, d, u);
            const sz = isTF ? 3.2 : 2.2;
            V.glow(g, q1[0], q1[1], 9, C.a, 0.8); V.dot(g, q1[0], q1[1], sz, C.a);
            V.glow(g, q2[0], q2[1], 9, C.b, 0.8); V.dot(g, q2[0], q2[1], sz, C.b);
          }
        }
        // bits de um elemento: rank 0, rank 1 e a referência
        const by = (narrow ? p1y + ph : 8 + ph) + 30;
        const showB = tc > T3;
        tfit(g, 'bits of Δ[' + IDX + '] in bf16 × reference = bf16(exact P⁰ + P¹) = ' + sigs(REF[IDX], 4), pad, by - 9, w - 2 * pad, { size: 11.5, min: 9, weight: 600, color: C.ink });
        g.globalAlpha = showB ? 1 : 0.3;
        bitTable(g, pad, by, Math.min(w - 2 * pad, 440), [[bfEnc(R0[IDX]).bits, 'rank 0'], [bfEnc(R1[IDX]).bits, 'rank 1'], [bfEnc(REF[IDX]).bits, 'reference']]);
        g.globalAlpha = 1;
        if (showB) {
          const d = R0.filter((x, i) => x !== REF[i]).length;
          const txt = d ? '≠  ' + d + ' of 6 elements differ from the reference (two roundings); rank 0 = rank 1 even so' : '=  all 6 elements match the reference; rank 0 = rank 1 in either order';
          V.wrap(g, txt, pad, by + 84, w - 2 * pad, { size: 12, weight: 700, color: d ? RED() : C.a, lineH: 15, maxLines: 2 });
        }
      });
    })();

    /* =================================================== FIG 5 — protocolo */
    (function () {
      const fig = root.querySelector('#fig-protocolo');
      adaptH(fig, 350, 620);
      const STEPS = [
        { t: 'Rendezvous', d: 'Rank 1 comes up first (over ssh). Rank 0 creates the <code>TCPStore</code> on port 29551, generates the <code>ncclUniqueId</code> and publishes it as <code>tf_nccl_uid</code>. Rank 1 reads the key, and both call <code>ncclCommInitRank</code> (NCCL is loaded through <code>ctypes</code>).' },
        { t: 'Same settings?', d: 'Each rank packs its settings into a list of integers (context, grid, copy drafts, split, <code>MAX_ROWS</code>…) and does an all-gather. If the lists differ, the engine refuses to start: “the two ranks were started with different settings”.' },
        { t: 'Idle: the doorbell', d: 'With no requests, rank 1 does not wait inside an all-gather, where it would keep the GPU spinning. It waits on a TCPStore key. An idle server ties up neither the GPU nor a CPU core.' },
        { t: 'ADMIT', d: 'A client arrives. Rank 0 rings the doorbell (<code>tf_glm_request_1</code>) and sends <b>one</b> message: <code>ADMIT</code> with stream, slot, extent base and size, sampling packed into 10 int32, policy and prompt. Both pools receive the same extent.' },
        { t: 'FILL', d: '<code>FILL</code>: a prompt chunk (up to 2,048 rows). Both ranks run the same forward, exchanging fp32 partials via all-gather twice per layer.' },
        { t: 'ROUND', d: '<code>ROUND</code> carries the streams and the depths. Rank 1 <b>derives the drafts on its own</b> (the same prompt and the same tokens give the same proposals) and only checks the depth. Both verify and sample the same token without exchanging it.' },
        { t: 'FINISH → IDLE', d: '<code>FINISH</code> closes the stream; the prompt state is kept in the pool (gray), identical on both ranks. <code>IDLE</code>: rank 1 goes back to the doorbell.' },
        { t: 'When the ranks disagree', d: 'Before patch 0040, the choice between RoCE and NCCL looked at the <b>address</b> of the buffer (<code>data_ptr % 16</code>), which can differ between ranks. Result: one rank waiting on RoCE, the other on NCCL, “forever”. Fix: decide only by the number of bytes. After an error, <code>broken</code> locks both until they restart.' },
      ];
      const st = UI.stepper(fig, STEPS, { autoplay: 6500 });
      // extent de 2 células (4.096 tokens: o prompt e a folga de decode) do ADMIT ao FINISH; nenhuma operação o redimensiona
      const POOLS = [[], [], [], [[0, 2, 'A']], [[0, 2, 'A']], [[0, 2, 'A']], [[0, 2, 'K']], [[0, 2, 'A']]];
      const MSG = [['tf_nccl_uid'], ['[1048576, 64, 1, 1, 16…]'], [], ['ADMIT s1 slot0 base0 size4096'], ['FILL s1 0→2048'], ['ROUND s1 depth=7'], ['FINISH s1', 'IDLE'], []];
      const LOGS = (i, rk) => {
        if (i < 3) return [];
        if (i === 7) return [['ADMIT s1 slot0 base0 size4096', 0], ['ROUND s1 depth=7', 0], [rk === 0 ? 'all_gather → RoCE (ptr % 16 = 0)' : 'all_gather → NCCL (ptr % 16 = 8)', 1]];
        return MSG.slice(3, i + 1).flat().map((x) => [x, 0]);
      };
      V.fig(fig, (g, w, h, t) => {
        const C = V.C(), tf = TFC();
        const narrow = w < 640, pad = 12;
        const i = st.i, el = st.elapsed();
        let R0, R1, store, cab;
        if (narrow) {
          const bh = (h - 110) / 2;
          R0 = { x: pad, y: 6, w: w - 2 * pad, h: bh }; R1 = { x: pad, y: h - 6 - bh, w: w - 2 * pad, h: bh };
          store = { x: w * 0.42, y: R0.y + bh + 10, w: w * 0.58 - pad, h: 78 };
          cab = { x1: w * 0.22, y1: R0.y + bh, x2: w * 0.22, y2: R1.y };
        } else {
          const bw = w * 0.34;
          R0 = { x: pad, y: 8, w: bw, h: h - 16 }; R1 = { x: w - pad - bw, y: 8, w: bw, h: h - 16 };
          const gx = R0.x + R0.w + 12, gw = R1.x - 12 - gx;
          store = { x: gx, y: h - 128, w: gw, h: 112 };
          cab = { x1: R0.x + R0.w, y1: 92, x2: R1.x, y2: 92 };
        }
        const deadlock = i === 7;
        // cabo
        g.strokeStyle = deadlock ? V.alpha(RED(), 0.8) : V.alpha(tf, i === 0 && el < 1.5 ? 0.25 : 0.8); g.lineWidth = 3;
        g.beginPath(); g.moveTo(cab.x1, cab.y1); g.lineTo(cab.x2, cab.y2); g.stroke();
        if (!narrow) V.text(g, 'CX7 cable', (cab.x1 + cab.x2) / 2, cab.y1 + 18, { size: 10.5, font: 'mono', align: 'center', color: C.muted });
        if (i === 0) tfit(g, el > 1.6 ? 'ncclCommInitRank ✓' : 'connecting…', narrow ? cab.x1 + 10 : (cab.x1 + cab.x2) / 2, narrow ? (cab.y1 + cab.y2) / 2 + 4 : cab.y1 - 10, narrow ? store.x - cab.x1 - 16 : cab.x2 - cab.x1 - 8, { size: 11, min: 8.5, font: 'mono', align: narrow ? 'left' : 'center', color: tf });
        // store
        V.box(g, store.x, store.y, store.w, store.h, { r: 10, fill: V.alpha(C.d, 0.08), stroke: V.alpha(C.d, 0.6), lw: 1.2 });
        tfit(g, 'TCPStore :29551', store.x + 8, store.y + 16, store.w - 16, { size: 11.5, min: 9, font: 'mono', weight: 600, color: C.d });
        tfit(g, 'on the head (rank 0)', store.x + 8, store.y + 30, store.w - 16, { size: 10.5, min: 9, color: C.muted });
        const keys = ['tf_nccl_uid'].concat(i >= 3 && i <= 6 ? ['tf_glm_request_1'] : []);
        keys.forEach((k, j) => tfit(g, '• ' + k, store.x + 8, store.y + 48 + j * 16, store.w - 16, { size: 10.5, min: 8.5, font: 'mono', color: j === 1 && i === 3 && el < 1.2 ? C.d : C.ink }));
        // ranks
        const rank = (R, rk) => {
          const accent = rk === 0 ? C.a : C.b;
          V.box(g, R.x, R.y, R.w, R.h, { r: 12, fill: V.alpha(accent, 0.05), stroke: V.alpha(accent, 0.55), lw: 1.3 });
          V.text(g, rk === 0 ? 'rank 0 · head (decides)' : 'rank 1 · worker (applies)', R.x + 10, R.y + 18, { size: 12, weight: 700, color: accent });
          tfit(g, rk === 0 ? 'HTTP :8888 + scheduler' : 'no API · follower', R.x + 10, R.y + 34, R.w - 20, { size: 11, font: 'mono', color: C.muted });
          // pool espelhado
          const px = R.x + 10, pyy = R.y + 44, pww = R.w - 20, cells = 16, cw = pww / cells;
          V.text(g, 'pool (same on both)', px, pyy + 10, { size: 10.5, color: C.muted });
          for (let c = 0; c < cells; c++) { g.fillStyle = V.alpha(C.muted, 0.12); V.rr(g, px + c * cw + 0.5, pyy + 16, cw - 1, 18, 2); g.fill(); }
          (POOLS[i] || []).forEach(([b, sz, id]) => { const col = id === 'K' ? C.muted : C.c; V.box(g, px + b * cw + 0.5, pyy + 16, sz * cw - 1, 18, { r: 3, fill: V.alpha(col, 0.5), stroke: col, lw: 1 }); tfit(g, id === 'K' ? 'kept' : 's1', px + b * cw + 4, pyy + 29, sz * cw - 6, { size: 10, min: 8, font: 'mono', color: C.ink }); });
          // estado
          const sy = pyy + 44;
          if (i === 0) {
            tfit(g, rk === 0 ? 'creates the TCPStore and publishes the uid' : 'came up first (ssh); reads the uid', R.x + 10, sy + 16, R.w - 20, { size: 11, color: C.ink });
          } else if (i === 1) {
            tfit(g, '[1048576, 64, 1, 1, 16, 512…]', R.x + 10, sy + 12, R.w - 20, { size: 11, font: 'mono', color: C.ink });
            if (el > 1.4) { mark(g, R.x + 18, sy + 30, true); V.text(g, 'lists match', R.x + 30, sy + 34, { size: 11.5, color: C.a }); }
          } else if (i === 2 || (i === 6 && el > 1.4)) {
            if (rk === 1) { bell(g, R.x + 22, sy + 18, 18, C.d, 0); V.text(g, 'zZ', R.x + 36, sy + 8 - Math.sin(t * 2) * 3, { size: 12, font: 'mono', color: C.muted }); tfit(g, 'on the doorbell · GPU 0% · CPU 0%', R.x + 54, sy + 24, R.w - 64, { size: 11, color: C.muted }); }
            else tfit(g, 'waiting for clients', R.x + 10, sy + 20, R.w - 20, { size: 11, color: C.muted });
          } else if (i === 3) {
            if (rk === 1) { bell(g, R.x + 22, sy + 18, 18, C.d, el < 1.2 ? t : 0); tfit(g, el < 1.2 ? 'ring-ring!' : 'ADMIT applied', R.x + 42, sy + 22, R.w - 52, { size: 11.5, font: 'mono', color: C.d }); }
            else tfit(g, 'client → POST /v1/chat/completions', R.x + 10, sy + 20, R.w - 20, { size: 11, color: C.ink });
          } else if (i === 4) {
            const pulses = Math.floor(el * 3) % 2;
            tfit(g, 'chunk forward (45 layers)', R.x + 10, sy + 14, R.w - 20, { size: 11, color: C.ink });
            tfit(g, '2 all-gathers per layer ' + (pulses ? '⇄' : '⇆'), R.x + 10, sy + 30, R.w - 20, { size: 11, font: 'mono', color: tf });
          } else if (i === 5) {
            const n = Math.min(7, Math.floor(el * 4));
            const cw2 = Math.min(24, (R.w - 20) / 8);
            for (let k = 0; k < n; k++) chip(g, R.x + 10 + k * cw2, sy + 2, cw2 - 3, 18, 'd' + (k + 1), { fill: V.alpha(C.b, 0.2), stroke: C.b, size: 10, min: 8 });
            tfit(g, rk === 0 ? 'proposes 7 and decides the depth' : 'derives the same 7 and checks', R.x + 10, sy + 36, R.w - 20, { size: 11, color: C.muted });
          } else if (i === 7) {
            hourglass(g, R.x + 20, sy + 14, 14, RED(), t);
            tfit(g, rk === 0 ? 'waiting for the peer on RoCE…' : 'waiting for the peer on NCCL…', R.x + 36, sy + 18, R.w - 46, { size: 11, color: RED() });
            tfit(g, 'fix (0040): choose only by the number of bytes', R.x + 10, sy + 38, R.w - 20, { size: 11, color: C.a });
          }
          // registro de operações (idêntico nos dois ranks, salvo no passo do deadlock)
          const logs = LOGS(i, rk);
          const ly0 = R.y + 150, lineH = 22;
          const maxL = Math.max(0, Math.floor((R.y + R.h - 8 - ly0) / lineH));
          if (maxL > 0) V.text(g, 'applied operations' + (logs.length ? '' : ': none yet'), R.x + 10, ly0 - 6, { size: 10.5, color: C.muted });
          if (logs.length && maxL > 0) {
            logs.slice(-maxL).forEach(([op, bad], k) => chip(g, R.x + 10, ly0 + k * lineH, R.w - 20, 18, op, { fill: V.alpha(bad ? RED() : C.c, 0.14), stroke: V.alpha(bad ? RED() : C.c, 0.6), size: 10.5, min: 8.5, color: bad ? RED() : C.ink }));
          }
        };
        rank(R0, 0); rank(R1, 1);
        // mensagens viajando
        const msgs = MSG[i] || [];
        if (i === 0) {
          const u = Math.min(1, el / 0.8), u2 = Math.min(1, Math.max(0, (el - 0.8) / 0.8));
          const a = [R0.x + R0.w * 0.6, R0.y + (narrow ? R0.h - 10 : R0.h * 0.75)], b = [store.x + store.w / 2, store.y + 10], c = [R1.x + R1.w * (narrow ? 0.6 : 0.3), R1.y + (narrow ? 10 : R1.h * 0.75)];
          const p = u2 > 0 ? V.along(b[0], b[1], c[0], c[1], V.ease.inOut(u2)) : V.along(a[0], a[1], b[0], b[1], V.ease.inOut(u));
          if (el < 1.7) chip(g, p[0] - 44, p[1] - 9, 88, 18, 'tf_nccl_uid', { fill: V.alpha(C.d, 0.3), stroke: C.d, size: 10.5 });
        } else if (i === 1) {
          const u = V.ease.inOut(Math.min(1, el / 1.2));
          const p1 = V.along(cab.x1, cab.y1, cab.x2, cab.y2, u), p2 = V.along(cab.x2, cab.y2, cab.x1, cab.y1, u);
          if (el < 1.3) { V.glow(g, p1[0], p1[1], 12, C.a); V.glow(g, p2[0], p2[1], 12, C.b); }
        } else if (msgs.length) {
          const per = 2.2, k = Math.floor(el / per) % msgs.length, u = V.ease.inOut(Math.min(1, (el % per) / 1.3));
          const p = V.along(cab.x1, cab.y1, cab.x2, cab.y2, u);
          const lab = msgs[k];
          const ww = Math.min(narrow ? w * 0.6 : (cab.x2 - cab.x1) + 40, tw(g, lab, 10.5, 'mono', 600) + 16);
          chip(g, p[0] - ww / 2, p[1] - 10, ww, 20, lab, { fill: V.alpha(C.c, 0.3), stroke: C.c, size: 10.5, min: 8.5, weight: 600 });
          if (i === 3 && el % per < 1) { const q = V.along(store.x + 10, store.y + 50, R1.x + 22, R1.y + (narrow ? 110 : 110), V.ease.inOut(Math.min(1, (el % per)))); V.glow(g, q[0], q[1], 10, C.d); }
        } else if (deadlock) {
          mark(g, (cab.x1 + cab.x2) / 2, (cab.y1 + cab.y2) / 2, false, 9);
        }
      });
    })();

    /* =================================================== FIG 6 — memória */
    (function () {
      const fig = root.querySelector('#fig-memoria');
      adaptH(fig, 270, 318);
      // README: estimativa 88,09 GiB, teto do pool 12,5, ~10 GiB além da estimativa no pico (prompt de 1M).
      // Suposição nossa: MemAvailable de 116 GiB com o Spark ocioso. Com ela, os ~10 GiB se dividem em
      // ~8,4 ocupados desde a carga e ~1,4 de pico transitório, o que reproduz o rank 0 medido (7,0 ocioso, 5,6 mínimo).
      const TOTAL = 119.2, M0 = 116, E = 88.09, CAP = 12.5, FIXED = 8.4, TRANS = 1.4;
      const S = { rule: 'tf', other: 6, R: 14.5, t0: performance.now() };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Rule', options: [{ v: 'fixa', label: 'fixed plan' }, { v: 'tf', label: 'MemAvailable − reserve' }], value: 'tf', onChange: (v) => { S.rule = v; S.t0 = performance.now(); upd(); } });
      UI.slider(ctl, { label: 'other load (GiB)', min: 0, max: 30, step: 0.5, value: 6, fmt: (v) => fmt(v, 1), onInput: (v) => { S.other = v; S.t0 = performance.now(); upd(); } });
      UI.slider(ctl, { label: 'reserve R (GiB)', min: 2, max: 24, step: 0.5, value: 14.5, fmt: (v) => fmt(v, 1), onInput: (v) => { S.R = v; S.t0 = performance.now(); upd(); } });
      const rB = UI.readout(ctl, 'budget B', '—'), rP = UI.readout(ctl, 'extra pool', '—'), rI = UI.readout(ctl, 'free when idle', '—'), rF = UI.readout(ctl, 'free at worst moment', '—'), rS = UI.readout(ctl, 'status', '—');
      function model() {
        const M = M0 - S.other;
        if (S.rule === 'fixa') { const extra = CAP; const idle = M - E - extra - FIXED; const free = idle - TRANS; return { M, B: null, extra, idle, free, refused: false, freeze: free < 0.5 }; }
        const B = M - S.R;
        if (B < E) return { M, B, extra: 0, idle: M, free: M, refused: true, freeze: false };
        const extra = Math.min(CAP, B - E); const idle = M - E - extra - FIXED; const free = idle - TRANS;
        return { M, B, extra, idle, free, refused: false, freeze: free < 0.5 };
      }
      function upd() {
        const m = model();
        rB.set(m.B == null ? '(not consulted)' : fmt(m.B, 1) + ' GiB');
        rP.set(m.refused ? '—' : fmt(m.extra, 1) + ' of 12.5 GiB', m.refused ? null : m.extra >= CAP - 0.01 ? 'ok' : 'warn');
        rI.set(m.refused ? '—' : fmt(m.idle, 1) + ' GiB', m.refused ? null : m.idle < 0.5 ? 'bad' : null);
        rF.set(m.refused ? '—' : fmt(m.free, 1) + ' GiB', m.freeze ? 'bad' : m.free < 3 ? 'warn' : 'ok');
        rS.set(m.freeze ? 'freezes at peak' : m.refused ? 'refused before loading' : 'starts up', m.freeze ? 'bad' : m.refused ? 'warn' : 'ok');
      }
      upd();
      V.fig(fig, (g, w, h) => {
        const C = V.C(), tf = TFC();
        const pad = 12, narrow = w < 600;
        const m = model();
        const el = (performance.now() - S.t0) / 1000;
        const CYC = 7.5, tc = el % CYC;
        const frozen = m.freeze && tc > 4;
        const load = m.refused ? 0 : Math.min(1, tc / 1.6);
        const poolA = m.refused ? 0 : Math.min(1, Math.max(0, (tc - 1.6) / 1));
        const peakA = m.refused ? 0 : frozen ? 1 : tc > 3 && tc < 6 ? Math.sin(((tc - 3) / 3) * Math.PI) : 0;
        tfit(g, 'One Spark at startup (GiB) · rule: ' + (S.rule === 'tf' ? 'MemAvailable − reserve' : 'fixed plan'), pad, 18, w - 2 * pad, { size: 12.5, weight: 600, color: C.ink });
        const subH = V.wrap(g, 'MemAvailable at startup = ' + fmt(m.M, 1) + ' GiB (with the Spark idle: 116, our assumption)', pad, 36, w - 2 * pad, { size: 11.5, font: 'mono', color: C.muted, lineH: 14, maxLines: 2 });
        const bx = pad, bw = w - 2 * pad, by = 36 + subH + 26, bh = narrow ? 46 : 54;
        const sc = bw / (TOTAL + 6);
        // contorno físico
        g.strokeStyle = V.alpha(C.muted, 0.6); g.lineWidth = 1.2; V.rr(g, bx, by, TOTAL * sc, bh, 6); g.stroke();
        V.text(g, '128 GB ≈ 119.2 GiB', bx + TOTAL * sc, by - 6, { size: 10.5, font: 'mono', align: 'right', color: C.muted });
        let x = bx;
        const seg = (gib, col, lab, o = {}) => {
          const ww = Math.max(0, gib * sc);
          if (ww <= 0) return;
          g.fillStyle = V.alpha(col, o.a == null ? 0.6 : o.a); V.rr(g, x, by + 1, ww, bh - 2, 4); g.fill();
          if (o.hatch) hatch(g, x, by + 1, ww, bh - 2, V.alpha(o.hatch, 0.6), 5);
          if (lab && ww > tw(g, lab, 11, 'sans', 600) + 8) V.text(g, lab, x + ww / 2, by + bh / 2 + 1, { size: 11, weight: 600, align: 'center', base: 'middle', color: C.ink });
          x += ww;
        };
        seg(TOTAL - M0, C.muted, 'OS', { a: 0.35 });
        seg(S.other, C.f, 'other load', { a: 0.5 });
        if (m.refused) {
          g.setLineDash([4, 4]); g.strokeStyle = RED(); g.lineWidth = 1.5; V.rr(g, x, by + 1, E * sc, bh - 2, 4); g.stroke(); g.setLineDash([]);
          V.text(g, 'estimate 88.09 does not fit the budget', x + 8, by + bh / 2 + 4, { size: 11.5, color: RED(), weight: 600, maxW: E * sc - 16 });
        } else {
          seg(E * load, C.b, narrow ? 'estimate 88.09' : 'estimate E = 88.09 (weights ~80)');
          seg(m.extra * poolA, C.a, 'pool');
          seg(FIXED * load, C.c, '+8.4', { a: 0.4 });
          seg(TRANS * peakA, C.c, '', { a: 0.75, hatch: C.c });
        }
        // linha do orçamento / reserva
        if (S.rule === 'tf') {
          const xs = bx + (TOTAL - m.M) * sc, xB = xs + Math.max(0, m.B) * sc, xM = bx + TOTAL * sc;
          const yR = by + bh + 12;
          g.fillStyle = V.alpha(tf, 0.18); g.fillRect(xB, by - 2, xM - xB, bh + 4);
          hatch(g, xB, by - 2, xM - xB, bh + 4, V.alpha(tf, 0.5), 7);
          g.strokeStyle = tf; g.lineWidth = 1.5; g.beginPath(); g.moveTo(xs, yR); g.lineTo(xs, yR + 6); g.lineTo(xB, yR + 6); g.lineTo(xB, yR); g.stroke();
          tfit(g, 'budget B = ' + fmt(m.B, 1), (xs + xB) / 2, yR + 20, Math.max(60, xB - xs), { size: 11, font: 'mono', align: 'center', color: tf });
          if (narrow) V.text(g, 'reserve ' + fmt(S.R, 1), xM, yR + 20, { size: 11, font: 'mono', align: 'right', color: tf }); else tfit(g, 'reserve ' + fmt(S.R, 1), (xB + xM) / 2, yR + 20, Math.max(50, xM - xB + 30), { size: 11, font: 'mono', align: 'center', color: tf });
        }
        // estouro
        const used = (TOTAL - M0) + S.other + (m.refused ? 0 : E + m.extra + FIXED + TRANS * peakA);
        if (used > TOTAL) {
          const x1 = bx + TOTAL * sc, x2 = Math.min(w - 2, bx + used * sc);
          g.fillStyle = V.alpha(RED(), 0.55); g.fillRect(x1, by - 4, x2 - x1, bh + 8);
        }
        // legenda / números
        let ly = by + bh + 52;
        const lines = m.refused
          ? ['Refused before loading any weight: the log reports the largest window that fits, and start.sh restarts once with a smaller --context.']
          : ['worst moment = estimate + pool + ~8.4 GiB taken since loading + ~1.4 GiB of peak (1M prompt)',
            'free: ' + fmt(m.idle, 1) + ' GiB idle, ' + fmt(m.free, 1) + ' at the worst moment' + (S.rule === 'tf' ? (m.extra < CAP - 0.01 ? '  (≈ R − 10: the pool did not hit its cap)' : '  (pool at its cap: headroom exceeds R − 10)') : '  (the plan ignores what is already taken)')];
        lines.forEach((s) => { ly += V.wrap(g, s, pad, ly, w - 2 * pad, { size: 11.5, color: C.muted, lineH: 14, maxLines: 2 }) + 6; });
        const items = [[C.b, 'estimate (weights + window + states)'], [C.a, 'extra pool (≤ 12.5)'], [V.alpha(C.c, 0.55), 'taken since loading (~8.4)'], [C.c, 'transient peak (~1.4)'], [C.f, 'other load'], [tf, 'reserve']];
        let lx = pad, lyy = ly + 12;
        items.forEach(([c, s]) => { const ww = tw(g, s, 11) + 22; if (lx + ww > w - pad) { lx = pad; lyy += 18; } V.dot(g, lx + 5, lyy - 4, 5, c); V.text(g, s, lx + 14, lyy, { size: 11, color: C.muted }); lx += ww; });
        if (frozen) {
          g.fillStyle = C.dark ? 'rgba(120,170,255,0.18)' : 'rgba(80,130,220,0.16)'; g.fillRect(0, 0, w, h);
          const cx = w / 2, cy = by + bh / 2;
          V.box(g, cx - 120, cy - 22, 240, 44, { r: 10, fill: C.dark ? 'rgba(10,20,40,0.85)' : 'rgba(255,255,255,0.92)', stroke: '#6aa0ff', lw: 2 });
          V.text(g, '❄ the Spark froze', cx, cy + 5, { size: 15, weight: 700, align: 'center', color: '#6aa0ff' });
        }
      });
    })();

    /* =================================================== FIG 7 — pool de extents */
    (function () {
      const fig = root.querySelector('#fig-pool');
      adaptH(fig, 300, 340);
      const N = 32;
      const ST = [
        { ex: [], op: 'empty pool: 32 blocks of 2,048 tokens', q: [], sl: [null, null, null, null] },
        { ex: [['A', 0, 6]], op: 'ADMIT A base=0 size=6', q: [], sl: ['A', null, null, null] },
        { ex: [['A', 0, 6], ['B', 6, 8]], op: 'ADMIT B base=6 size=8', q: [], sl: ['A', 'B', null, null] },
        { ex: [['A', 0, 6], ['B', 6, 8], ['C', 14, 6]], op: 'ADMIT C base=14 size=6', q: [], sl: ['A', 'B', 'C', null] },
        { ex: [['KA', 0, 6], ['B', 6, 8], ['C', 14, 6]], op: 'FINISH A (the prompt state stays)', q: [], sl: [null, 'B', 'C', null] },
        { ex: [['KA', 0, 6], ['B', 6, 8], ['C', 14, 12]], op: 'GROW C 6 → 12 (in place)', q: [], sl: [null, 'B', 'C', null] },
        { ex: [['B', 0, 12], ['C', 14, 12]], op: 'EVICT KA · MOVE B 6→0 · GROW B 12', q: [], sl: [null, 'B', 'C', null] },
        { ex: [['B', 0, 12], ['C', 12, 12], ['D', 24, 8]], op: 'MOVE C 14→12 (compact) · ADMIT D base=24', q: [], sl: ['D', 'B', 'C', null] },
        { ex: [['B', 0, 12], ['C', 12, 12], ['D', 24, 8]], op: 'E needs 6: NoRoom, waits (FIFO)', q: ['E'], sl: ['D', 'B', 'C', null] },
        { ex: [['B', 0, 12], ['E', 12, 6], ['D', 24, 8]], op: 'FINISH C · EVICT KC · ADMIT E base=12', q: [], sl: ['D', 'B', 'E', null] },
      ];
      const STEPS = [
        { t: 'Empty pool', d: 'The arena has 32 blocks of 2,048 tokens (reduced scale; the real pool has 1,427). Everything that grows with context lives here; what is per stream goes in a slot.' },
        { t: 'ADMIT A', d: 'First fit by <b>lowest base</b>: A gets <code>[0, 6)</code>. The operation becomes a message to rank 1, which applies the same arithmetic and ends up with the same pool.' },
        { t: 'ADMIT B', d: 'B comes in right after, at <code>[6, 14)</code>, including the slack block of a stream that is going to decode.' },
        { t: 'ADMIT C', d: 'C at <code>[14, 20)</code>. Each stream sees only a <i>view</i> of its extent, as if it were a tensor of its own.' },
        { t: 'FINISH A', d: 'A finishes, but its prompt state is kept (gray). If the same conversation comes back, it resumes from there. Kept prompts are the first to give way.' },
        { t: 'GROW C', d: 'C needs more rows and there is room right after it: it grows <b>in place</b>, without copying anything.' },
        { t: 'B does not fit: evict and move', d: 'B needs 12 blocks, but C sits right against it. The engine evicts the kept prompt of A (LRU) and <b>moves</b> B to base 0, with a safe copy even though source and destination overlap.' },
        { t: 'Fragmentation: compact', d: 'D needs 8 blocks. There are 8 free, but in two holes (2 + 6). Compacting slides C down, copying only the rows in use, and D fits in <code>[24, 32)</code>.' },
        { t: 'No room: wait', d: 'E needs 6 and there is nothing free or kept to evict. It waits at the front of the queue (<code>NoRoom</code>), in FIFO order.' },
        { t: 'Finishing makes room', d: 'C finishes. Its kept state is evicted so that E can enter at <code>[12, 18)</code>. Kept prompts “give way first”.' },
      ];
      const S = { view: 'ext', t0: performance.now() };
      const st = UI.stepper(fig, STEPS, { autoplay: 5200, onStep: () => { S.t0 = performance.now(); } });
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Layout', options: [{ v: 'ext', label: 'extents (TensorFold)' }, { v: 'pag', label: 'pages (PagedAttention)' }], value: 'ext', onChange: (v) => { S.view = v; } });
      const rFree = UI.readout(ctl, 'free', '—');
      // simulação das páginas: lista livre embaralhada, devolução no fim
      const order = Array.from({ length: N }, (_, i) => i); { const rr = rng(31); for (let i = N - 1; i > 0; i--) { const j = Math.floor(rr() * (i + 1)); [order[i], order[j]] = [order[j], order[i]]; } }
      const PAGES = []; { let free = order.slice(); const own = {};
        ST.forEach((s) => {
          const want = {}; s.ex.forEach(([id, , n]) => { const key = id.replace('K', ''); want[key] = { n, kept: id.startsWith('K') }; });
          Object.keys(own).forEach((k) => { const tgt = want[k] ? want[k].n : 0; while (own[k].pages.length > tgt) free.push(own[k].pages.pop()); if (!want[k]) delete own[k]; });
          Object.keys(want).forEach((k) => { if (!own[k]) own[k] = { pages: [] }; own[k].kept = want[k].kept; while (own[k].pages.length < want[k].n) own[k].pages.push(free.shift()); });
          PAGES.push(JSON.parse(JSON.stringify(own)));
        });
      }
      const colOf = (id, C) => ({ A: C.a, B: C.b, C: C.c, D: C.d, E: C.e })[id.replace('K', '')] || C.muted;
      V.fig(fig, (g, w, h) => {
        const C = V.C();
        const narrow = w < 600, pad = 12;
        const i = st.i, s = ST[i], prev = ST[Math.max(0, i - 1)];
        const u = V.ease.inOut(Math.min(1, (performance.now() - S.t0) / 900));
        const perRow = narrow ? 16 : 32, nRows = N / perRow;
        const cw = (w - 2 * pad) / perRow, ch = narrow ? 40 : 56;
        const y0 = 70;
        const rowGap = 24;
        V.text(g, 'per-token pool (arena) · each block = 2,048 tokens', pad, y0 - 9, { size: 11, color: C.muted, maxW: w - 2 * pad });
        const cellXY = (k) => [pad + (k % perRow) * cw, y0 + Math.floor(k / perRow) * (ch + rowGap)];
        // op
        chip(g, pad, 8, Math.min(w - 2 * pad, tw(g, '→ rank 1: ' + s.op, 11.5, 'mono', 600) + 20), 24, '→ rank 1: ' + s.op, { fill: V.alpha(C.c, 0.15), stroke: C.c, size: 11.5, weight: 600 });
        // células
        for (let k = 0; k < N; k++) {
          const [x, y] = cellXY(k);
          g.fillStyle = V.alpha(C.muted, 0.1); V.rr(g, x + 1, y, cw - 2, ch, 3); g.fill();
          if (k % 4 === 0) V.text(g, String(k), x + 2, y + ch + 12, { size: 10, font: 'mono', color: C.muted });
        }
        let used = 0;
        if (S.view === 'ext') {
          const prevMap = {}; prev.ex.forEach(([id, b, n]) => { prevMap[id.replace('K', '')] = [b, n]; });
          s.ex.forEach(([id, b, n]) => {
            used += n;
            const key = id.replace('K', '');
            const pv = prevMap[key];
            const bb = pv ? V.lerp(pv[0], b, u) : b, nn = pv ? V.lerp(pv[1], n, u) : n * u;
            const kept = id.startsWith('K');
            const col = kept ? C.muted : colOf(id, C);
            // desenha por linha (quebra quando estreito)
            let a0 = bb, a1 = bb + nn;
            for (let r = 0; r < nRows; r++) {
              const r0 = r * perRow, r1 = r0 + perRow;
              const s0 = Math.max(a0, r0), s1 = Math.min(a1, r1);
              if (s1 <= s0) continue;
              const x = pad + (s0 - r0) * cw, y = y0 + r * (ch + rowGap), ww = (s1 - s0) * cw;
              V.box(g, x + 1, y + 2, ww - 2, ch - 4, { r: 6, fill: V.alpha(col, kept ? 0.3 : 0.55), stroke: col, lw: 1.5 });
              if (kept) hatch(g, x + 1, y + 2, ww - 2, ch - 4, V.alpha(C.muted, 0.5), 6);
              if (ww > 28) V.text(g, kept ? 'kept ' + key : key, x + ww / 2, y + ch / 2 + 1, { size: 12, font: 'mono', weight: 700, align: 'center', base: 'middle', color: C.ink, maxW: ww - 6 });
            }
          });
          // removidos (somem)
          prev.ex.forEach(([id, b, n]) => {
            const key = id.replace('K', '');
            if (s.ex.some(([j]) => j.replace('K', '') === key)) return;
            const a = 1 - u; if (a <= 0) return;
            g.globalAlpha = a;
            for (let r = 0; r < nRows; r++) {
              const r0 = r * perRow, s0 = Math.max(b, r0), s1 = Math.min(b + n, r0 + perRow);
              if (s1 <= s0) continue;
              const x = pad + (s0 - r0) * cw, y = y0 + r * (ch + rowGap);
              V.box(g, x + 1, y + 2, (s1 - s0) * cw - 2, ch - 4, { r: 6, fill: V.alpha(RED(), 0.25), stroke: RED(), lw: 1.5 });
            }
            g.globalAlpha = 1;
          });
        } else {
          const own = PAGES[i];
          Object.keys(own).forEach((k) => {
            const col = own[k].kept ? C.muted : colOf(k, C);
            own[k].pages.forEach((pg) => {
              used++;
              const [x, y] = cellXY(pg);
              V.box(g, x + 1.5, y + 2, cw - 3, ch - 4, { r: 4, fill: V.alpha(col, own[k].kept ? 0.3 : 0.6), stroke: col, lw: 1 });
              if (cw > 14) V.text(g, k, x + cw / 2, y + ch / 2 + 1, { size: 11, font: 'mono', weight: 700, align: 'center', base: 'middle', color: C.ink });
            });
          });
          // tabelas de blocos
          const ty = y0 + nRows * (ch + rowGap) + 10;
          V.text(g, 'block table per sequence (the kernel gathers through it):', pad, ty, { size: 11.5, color: C.muted, maxW: w - 2 * pad });
          let k2 = 0;
          Object.keys(own).forEach((k) => {
            const line = k + (own[k].kept ? ' (cache)' : '') + ': [' + own[k].pages.join(', ') + ']';
            tfit(g, line, pad, ty + 18 + k2 * 16, w - 2 * pad, { size: 11, font: 'mono', color: own[k].kept ? C.muted : colOf(k, C) });
            k2++;
          });
        }
        rFree.set(fmtInt((N - used) * 2048) + ' tokens');
        // fila
        if (s.q.length && S.view === 'ext') {
          const qy = y0 + nRows * (ch + rowGap) + 8;
          chip(g, pad, qy, 150, 26, 'E (6 blocks) waiting', { fill: V.alpha(C.e, 0.18), stroke: C.e, size: 11.5, font: 'sans', weight: 600 });
          V.text(g, 'NoRoom · FIFO', pad + 160, qy + 17, { size: 11.5, font: 'mono', color: C.muted });
        } else if (S.view === 'ext') {
          V.wrap(g, 'Contiguous: the kernels of a stream read consecutive rows; the cost is moving and compacting now and then.', pad, y0 + nRows * (ch + rowGap) + 16, w - 2 * pad, { size: 11.5, color: C.muted, lineH: 14, maxLines: 2 });
        }
        if (S.view === 'ext') {
          // slots: o que é por fluxo e não cresce com o contexto
          const sy = y0 + nRows * (ch + rowGap) + (narrow ? 54 : 46);
          V.text(g, narrow ? 'per-stream slots (KDA, conv, DFlash2): fixed size' : 'per-stream slots (KDA state in 2 parities, conv, DFlash2 context): fixed size', pad, sy, { size: 11, color: C.muted, maxW: w - 2 * pad });
          const sw = (w - 2 * pad - 3 * 8) / 4;
          s.sl.forEach((id, k) => {
            const x = pad + k * (sw + 8), y = sy + 8;
            const col = id ? colOf(id, C) : C.muted;
            V.box(g, x, y, sw, 30, { r: 6, fill: V.alpha(col, id ? 0.35 : 0.06), stroke: V.alpha(col, id ? 0.9 : 0.35), lw: 1.2 });
            tfit(g, (narrow ? '' : 'slot ') + k + (id ? ' · ' + id : ' · free'), x + sw / 2, y + 19, sw - 8, { size: 11.5, min: 9, font: 'mono', align: 'center', color: id ? C.ink : C.muted, weight: id ? 700 : 400 });
          });
        }
      });
    })();

    /* =================================================== FIG 8 — estados de prompt */
    (function () {
      const fig = root.querySelector('#fig-estados');
      adaptH(fig, 262, 282);
      const SYS = 7900, USER = 200, REPLY = 800, NEW = 200, BLK = 3584, G = 64, RATE = 1950;
      const S = { sc: 'id', eng: 'tf', n: 64000, snap: false, t0: performance.now() };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Scenario', options: [{ v: 'id', label: 'identical resend' }, { v: 'sys', label: 'same system' }, { v: 'fu', label: 'follow-up' }], value: 'id', onChange: (v) => { S.sc = v; S.t0 = performance.now(); upd(); gN.show(v !== 'sys'); } });
      UI.seg(ctl, { label: 'Engine', options: [{ v: 'tf', label: 'TensorFold' }, { v: 'apc', label: 'simplified APC' }], value: 'tf', onChange: (v) => { S.eng = v; S.t0 = performance.now(); upd(); } });
      const gN = UI.group(ctl);
      nocase(UI.slider(gN.el, { label: 'prompt n (tokens)', min: 1000, max: 70000, step: 1, value: 64000, fmt: (v) => fmtInt(v), onInput: (v) => { S.n = v; S.t0 = performance.now(); upd(); } }));
      const tg = UI.toggle(gN.el, { label: 'n multiple of 64', value: true, onChange: (v) => { S.snap = v; upd(); } });
      S.snap = true;
      const rRec = UI.readout(ctl, 'recomputed', '—'), rT = UI.readout(ctl, 'time (~1,950 tok/s)', '—');
      const nEff = () => (S.snap ? Math.max(G, Math.round(S.n / G) * G) : S.n);
      const gp = (n) => G * Math.floor(n / G); // B(n): o grid_point
      function calc() {
        const n = nEff();
        if (S.sc === 'id') {
          const total = n;
          if (S.eng === 'tf') { const B = gp(n); return { total, res: B, parts: [['identical prompt', n]], pins: [[B, B === n ? 'state + logits row' : 'state at B']], replay: B === n }; }
          const hit = BLK * Math.floor((n - 1) / BLK); return { total, res: hit, parts: [['identical prompt', n]], pins: [], blocks: true };
        }
        if (S.sc === 'sys') {
          const total = SYS + USER;
          const parts = [['system 7,900', SYS], ['user ' + USER, USER]];
          if (S.eng === 'tf') { const B = gp(SYS); return { total, res: B, parts, pins: [[B, 'end of system (on the grid)']] }; }
          const hit = BLK * Math.floor(SYS / BLK); return { total, res: hit, parts, pins: [], blocks: true };
        }
        const total = n + REPLY + NEW;
        if (S.eng === 'tf') { const B = gp(n); return { total, res: B, parts: [['previous prompt', n], ['answer', REPLY], ['new', NEW]], pins: [[B, 'previous prompt state']] }; }
        const hit = BLK * Math.floor(Math.min(n + REPLY, total - 1) / BLK); return { total, res: hit, parts: [['previous prompt', n], ['answer', REPLY], ['new', NEW]], pins: [], blocks: true };
      }
      function upd() { tg.el.style.opacity = S.sc === 'sys' ? 0.4 : 1; const c = calc(); const rec = c.total - c.res; rRec.set(fmtInt(rec) + (rec === 1 ? ' token' : ' tokens'), rec === 0 ? 'ok' : rec < 100 ? 'ok' : rec < 2000 ? 'warn' : 'bad'); rT.set(rec === 0 ? 'zero prefill (replay)' : fmt(rec / RATE, 2) + ' s'); }
      upd();
      V.fig(fig, (g, w, h) => {
        const C = V.C(), tf = TFC();
        const pad = 12, narrow = w < 600;
        const c = calc(), rec = c.total - c.res;
        const el = (performance.now() - S.t0) / 1000;
        const x0 = pad, x1 = w - pad, tw0 = x1 - x0;
        const X = (tok) => x0 + (tok / c.total) * tw0;
        const ty = 54, th = 30;
        tfit(g, (S.eng === 'tf' ? 'TensorFold: states kept at chosen points' : 'simplified APC: blocks of 3,584 tokens'), pad, 16, w - 2 * pad, { size: 12.5, weight: 600, color: S.eng === 'tf' ? tf : C.glm });
        // partes
        let acc = 0;
        c.parts.forEach(([lab, n], k) => {
          const xa = X(acc), xb = X(acc + n);
          g.strokeStyle = V.alpha(C.muted, 0.7); g.lineWidth = 1; g.beginPath(); g.moveTo(xa + 1, ty - 6); g.lineTo(xa + 1, ty - 10); g.lineTo(xb - 1, ty - 10); g.lineTo(xb - 1, ty - 6); g.stroke();
          if (xb - xa > 30) tfit(g, lab, (xa + xb) / 2, ty - 14, xb - xa, { size: 11, align: 'center', color: C.muted });
          acc += n;
        });
        // fita
        const resX = X(c.res);
        g.fillStyle = V.alpha(C.a, 0.5); V.rr(g, x0, ty, Math.max(0, resX - x0), th, 5); g.fill();
        g.fillStyle = V.alpha(RED(), 0.5); g.fillRect(resX, ty, x1 - resX, th);
        g.strokeStyle = V.alpha(C.muted, 0.6); g.lineWidth = 1; V.rr(g, x0, ty, tw0, th, 5); g.stroke();
        if (c.blocks) for (let b = BLK; b < c.total; b += BLK) { const x = X(b); g.strokeStyle = V.alpha(C.glm, 0.9); g.lineWidth = 1.2; g.beginPath(); g.moveTo(x, ty - 3); g.lineTo(x, ty + th + 3); g.stroke(); }
        // cursor
        const jump = 0.5, crawl = Math.max(0.6, Math.min(3, rec / 2000));
        const cyc = jump + crawl + 1.4, tc = el % cyc;
        let cx = tc < jump ? V.lerp(x0, resX, V.ease.out(tc / jump)) : V.lerp(resX, x1, Math.min(1, (tc - jump) / crawl));
        V.line(g, cx, ty - 4, cx, ty + th + 4, { color: C.ink, lw: 2 });
        if (tc < jump + 0.3) V.glow(g, resX, ty + th / 2, 18, C.a, 0.8);
        V.text(g, 'resumed', x0 + 4, ty + th + 14, { size: 11, color: C.a });
        tfit(g, 'redone: ' + fmtInt(rec), x1, ty + th + 14, tw0 * 0.6, { size: 11, align: 'right', color: rec ? RED() : C.a, font: 'mono' });
        c.pins.forEach(([p, lab]) => {
          const x = X(p);
          g.fillStyle = tf; g.beginPath(); g.moveTo(x, ty - 2); g.lineTo(x - 5, ty - 10); g.lineTo(x + 5, ty - 10); g.closePath(); g.fill();
          V.line(g, x, ty, x, ty + th, { color: tf, lw: 2 });
        });
        // lupa
        const ly = ty + th + 44, lh = narrow ? 70 : 64;
        const center = c.res;
        const half = S.eng === 'tf' ? 160 : Math.min(2400, Math.max(400, rec));
        const a = Math.max(0, center - half), b = Math.min(c.total, center + half);
        const LX = (tok) => x0 + ((tok - a) / (b - a)) * tw0;
        g.strokeStyle = V.alpha(C.muted, 0.4); g.lineWidth = 1; g.setLineDash([3, 3]);
        g.beginPath(); g.moveTo(X(a), ty + th); g.lineTo(x0, ly); g.moveTo(X(b), ty + th); g.lineTo(x1, ly); g.stroke(); g.setLineDash([]);
        g.fillStyle = V.alpha(C.a, 0.4); g.fillRect(x0, ly, Math.max(0, LX(c.res) - x0), lh);
        g.fillStyle = V.alpha(RED(), 0.4); g.fillRect(LX(c.res), ly, x1 - LX(c.res), lh);
        g.strokeStyle = V.alpha(C.muted, 0.6); g.strokeRect(x0, ly, tw0, lh);
        V.text(g, 'magnifier: tokens ' + fmtInt(a) + ' to ' + fmtInt(b), x0, ly + lh + 14, { size: 11, color: C.muted, font: 'mono' });
        if (S.eng === 'tf') {
          const first = Math.ceil(a / G) * G;
          for (let k = first; k <= b; k += G) {
            const x = LX(k);
            g.strokeStyle = V.alpha(tf, 0.9); g.lineWidth = 1.2; g.beginPath(); g.moveTo(x, ly); g.lineTo(x, ly + lh); g.stroke();
            if (!narrow || (k / G) % 2 === 0) V.text(g, fmtInt(k), V.clamp(x, x0 + 18, x1 - 18), ly - 4, { size: 10, font: 'mono', align: 'center', color: tf });
          }
          V.text(g, 'grid of 64', x1, ly + lh + 14, { size: 11, align: 'right', color: tf });
        } else {
          const first = Math.ceil(a / BLK) * BLK;
          for (let k = first; k <= b; k += BLK) { const x = LX(k); g.strokeStyle = C.glm; g.lineWidth = 1.5; g.beginPath(); g.moveTo(x, ly); g.lineTo(x, ly + lh); g.stroke(); V.text(g, fmtInt(k), V.clamp(x, x0 + 18, x1 - 18), ly - 4, { size: 10, font: 'mono', align: 'center', color: C.glm }); }
          V.text(g, 'block boundaries', x1, ly + lh + 14, { size: 11, align: 'right', color: C.glm });
        }
        // nota
        let note;
        if (S.eng === 'tf') note = S.sc === 'id' ? (c.replay ? 'n falls on the grid: end state + logits row → the first token comes out with no prefill (replay)' : 'n off the grid: state at B = 64⌊n/64⌋; redoes ' + rec + (rec === 1 ? ' token' : ' tokens') + ' (< 64)') : S.sc === 'sys' ? 'state kept at the end of the system block (first <|user|>), on the grid; redoes ' + (SYS - gp(SYS)) + ' + ' + USER : 'the state sits at the point of the previous prompt: the answer and the new message are prefilled again';
        else note = S.sc === 'fu' ? 'full blocks generated during decode are cached too: here APC can win' : 'the hit is rounded down to a whole block; the tail is redone';
        V.wrap(g, note, pad, ly + lh + 36, w - 2 * pad, { size: 11.5, color: C.ink, lineH: 15, maxLines: 3 });
      });
    })();

    /* =================================================== FIG 9 — rodada e fio */
    (function () {
      const fig = root.querySelector('#fig-rodada');
      const S = { mode: 'rod', streams: 4, copy: true, maxkb: 512, R: 8, t0: performance.now() };
      const stg = fig.querySelector('.la-fig-stage');
      const setH = () => { const nar = fig.clientWidth < 640; const want = (S.mode === 'rod' ? (nar ? 450 : 330) : (nar ? 680 : 470)) + 'px'; if (stg.style.height !== want) stg.style.height = want; };
      setH();
      { const ro = new ResizeObserver(setH); ro.observe(fig); if (LA.onCleanup) LA.onCleanup(() => ro.disconnect()); }
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'View', options: [{ v: 'rod', label: 'the round' }, { v: 'fio', label: 'the wire' }], value: 'rod', onChange: (v) => { S.mode = v; gA.show(v === 'rod'); gB.show(v === 'fio'); S.t0 = performance.now(); setH(); } });
      const gA = UI.group(ctl), gB = UI.group(ctl);
      UI.slider(gA.el, { label: 'decoding streams', min: 1, max: 4, step: 1, value: 4, onInput: (v) => { S.streams = v; S.t0 = performance.now(); } });
      UI.toggle(gA.el, { label: 'stream 1 with copy drafts', value: true, onChange: (v) => { S.copy = v; S.t0 = performance.now(); } });
      const rRows = UI.readout(gA.el, 'rows', '—'), rBytes = UI.readout(gA.el, 'per all-gather', '—');
      nocase(UI.slider(gB.el, { label: 'window rows R', min: 1, max: 64, step: 1, value: 8, onInput: (v) => { S.R = v; updB(); } }));
      UI.seg(gB.el, { label: 'TF_ROCE_MAX_KB', options: [{ v: 256, label: '256 (TensorFold)' }, { v: 512, label: '512 (recipe)' }], value: 512, onChange: (v) => { S.maxkb = v; updB(); } });
      const rB2 = UI.readout(gB.el, 'bytes', '—'), rTr = UI.readout(gB.el, 'transport', '—');
      gB.show(false);
      function updB() { const kb = S.R * 16; rB2.set(kb + ' KiB'); rTr.set(kb <= S.maxkb ? 'one-shot RoCE' : 'NCCL', kb <= S.maxkb ? 'ok' : 'warn'); }
      updB();
      const depthAt = (s, round) => { const rr = rng(1000 + round * 13 + s * 101); const x = rr(); return x < 0.15 ? 2 : x < 0.3 ? 3 : x < 0.5 ? 4 : x < 0.68 ? 5 : x < 0.84 ? 6 : 7; };
      function round(ri) {
        const lanes = [];
        for (let s = 0; s < S.streams; s++) {
          const isCopy = s === 0 && S.copy;
          const k = isCopy ? 15 : depthAt(s, ri);
          lanes.push({ k, copy: isCopy });
        }
        const orig = lanes.map((l) => l.k);
        const cut = lanes.map(() => 0);
        const cap = S.streams === 1 ? 16 : 32;
        let total = lanes.reduce((a, l) => a + 1 + l.k, 0);
        const order = [];
        const ks = orig.slice();
        while (total > cap) { let m = 0; ks.forEach((k, i) => { if (k > ks[m] || (k === ks[m] && i > m)) m = i; }); ks[m]--; cut[m]++; order.push(m); total--; }
        return { lanes, orig, cut, order, total, cap };
      }
      V.fig(fig, (g, w, h, t) => {
        const C = V.C(), tf = TFC();
        const narrow = w < 640, pad = 12;
        const el = (performance.now() - S.t0) / 1000;
        if (S.mode === 'rod') {
          const PER = 4.2, ri = Math.floor(el / PER), rt = el % PER;
          const R = round(ri);
          const nCutShown = Math.min(R.order.length, Math.max(0, Math.floor((rt - 0.9) / 0.09)));
          const cutNow = R.lanes.map(() => 0); R.order.slice(0, nCutShown).forEach((m) => cutNow[m]++);
          const rowsNow = R.lanes.reduce((a, l, i) => a + 1 + l.k - cutNow[i], 0);
          rRows.set(rowsNow + (R.order.length ? ' (of ' + R.lanes.reduce((a, l) => a + 1 + l.k, 0) + ')' : ''), rowsNow <= R.cap ? 'ok' : 'warn');
          rBytes.set(rowsNow * 16 + ' KiB → ' + (rowsNow * 16 <= 512 ? 'RoCE' : 'NCCL'));
          const lw = narrow ? w - 2 * pad : w * 0.62 - pad;
          tfit(g, 'round ' + (ri + 1) + ': ' + (S.streams === 1 ? 'one stream, window up to 16 rows' : S.streams + ' streams, batched window up to 32 rows'), pad, 16, w - 2 * pad, { size: 12.5, weight: 600, color: C.ink });
          const laneH = 34, ly0 = 36;
          const chipW = Math.min(17, (lw - 74) / 16);
          R.lanes.forEach((L, s) => {
            const y = ly0 + s * (laneH + 8);
            V.text(g, 'stream ' + (s + 1), pad, y + 14, { size: 11.5, weight: 600, color: C.muted });
            V.text(g, L.copy ? 'copy ' + L.k : 'DFlash2 ' + L.k, pad, y + 28, { size: 10.5, font: 'mono', color: L.copy ? C.d : C.b });
            const x0 = pad + 66;
            const n = 1 + L.k;
            for (let j = 0; j < n; j++) {
              const appear = Math.min(1, Math.max(0, (rt - j * 0.03) / 0.3));
              const isCut = j >= n - cutNow[s];
              const x = x0 + j * chipW, yy = y + 6;
              g.globalAlpha = appear * (isCut ? 0.25 : 1);
              const col = j === 0 ? C.a : L.copy ? C.d : C.b;
              V.box(g, x, yy, chipW - 2, laneH - 12, { r: 3, fill: V.alpha(col, isCut ? 0.1 : 0.55), stroke: col, lw: 1 });
              if (isCut) { V.line(g, x, yy + (laneH - 12) / 2, x + chipW - 2, yy + (laneH - 12) / 2, { color: RED(), lw: 1.5 }); }
              g.globalAlpha = 1;
            }
            if (R.order.length && nCutShown > 0 && nCutShown < R.order.length && R.order[nCutShown - 1] === s) { const sx = x0 + (n - cutNow[s]) * chipW + 4; V.text(g, '✂', sx, y + laneH / 2 + 5, { size: 14, color: RED() }); }
          });
          const yInfo = ly0 + R.lanes.length * (laneH + 8) + 8;
          const rowsF = R.total;
          const infoTxt = 'R = ' + rowsNow + ' rows · ' + rowsNow + ' × 16 KiB = ' + rowsNow * 16 + ' KiB per all-gather' + (R.order.length ? ' · trim: one draft at a time from the longest window' : '');
          V.wrap(g, infoTxt, pad, yInfo + 4, lw, { size: 11.5, color: C.ink, lineH: 15, maxLines: 3, font: 'mono' });
          // banco de graphs
          const gx = narrow ? pad : w * 0.64 + 6, gy = narrow ? yInfo + 54 : 40, gw = narrow ? w - 2 * pad : w - gx - pad;
          const single = S.streams === 1;
          V.text(g, single ? 'single-stream graphs (PARALLEL=1): 1–8 and 16' : 'batched graphs: one per count 1–32', gx, gy, { size: 11.5, weight: 600, color: C.ink, maxW: gw });
          const cols = 8, cs = Math.min(30, (gw - 7 * 4) / 8);
          const lit = rt > 0.9 + R.order.length * 0.09 + 0.2 ? rowsF : null;
          for (let k = 1; k <= 32; k++) {
            const cx = gx + ((k - 1) % cols) * (cs + 4), cy = gy + 10 + Math.floor((k - 1) / cols) * (cs + 4);
            const avail = single ? k <= 8 || k === 16 : true;
            const on = lit === k;
            if (on) V.glow(g, cx + cs / 2, cy + cs / 2, cs, tf, 0.7);
            V.box(g, cx, cy, cs, cs, { r: 5, fill: on ? V.alpha(tf, 0.7) : V.alpha(avail ? tf : C.muted, avail ? 0.12 : 0.05), stroke: on ? tf : V.alpha(avail ? tf : C.muted, avail ? 0.5 : 0.2), lw: on ? 2 : 1 });
            V.text(g, String(k), cx + cs / 2, cy + cs / 2 + 1, { size: 10.5, font: 'mono', align: 'center', base: 'middle', color: avail ? C.ink : C.muted });
          }
          const ty = gy + 10 + 4 * (cs + 4) + 14;
          V.wrap(g, single ? 'with PARALLEL=4 and TF_GLM_MULTI_LONE=0 (the recipe), a lone stream uses the batched graphs' : 'positions, extents, slots and parities go in tables on the device: the same graph serves any mix of streams', gx, ty, gw, { size: 11, color: C.muted, lineH: 14, maxLines: 3 });
          return;
        }
        // ---------------- o fio
        const STEPS = ['1 · each block copies its part of the shard into send[seq & 1] (pinned memory)', '2 · the last block publishes the size and rings the doorbell; the proxy does an RDMA WRITE', '3 · one thread per (peer, HCA) waits for the flag (acquire, system scope)', '4 · each block copies the shards in rank order', '5 · the last block advances the device epoch (replay in a CUDA graph)'];
        const SD = 1.5, k = Math.floor(el / SD) % 5, st = (el % SD) / SD;
        if (narrow) V.wrap(g, STEPS[k], pad, 16, w - 2 * pad, { size: 12, weight: 600, color: C.ink, lineH: 15, maxLines: 2 });
        else tfit(g, STEPS[k], pad, 16, w - 2 * pad, { size: 12, weight: 600, color: C.ink });
        const top = narrow ? 42 : 30, bottom = narrow ? h - 160 : h - 124;
        let A, B;
        if (narrow) { const bh = (bottom - top - 54) / 2; A = { x: pad, y: top, w: w - 2 * pad, h: bh }; B = { x: pad, y: top + bh + 54, w: w - 2 * pad, h: bh }; }
        else { const bw = w * 0.36; A = { x: pad, y: top, w: bw, h: bottom - top }; B = { x: w - pad - bw, y: top, w: bw, h: bottom - top }; }
        const spark = (R, name, col) => {
          V.box(g, R.x, R.y, R.w, R.h, { r: 12, fill: V.alpha(col, 0.05), stroke: V.alpha(col, 0.5), lw: 1.2 });
          V.text(g, name, R.x + 8, R.y + 15, { size: 11.5, weight: 700, color: col });
          const gpu = { x: R.x + 8, y: R.y + 22, w: R.w * 0.42 - 8, h: R.h - 30 };
          const mem = { x: R.x + R.w * 0.42 + 4, y: R.y + 22, w: R.w * 0.58 - 12, h: R.h - 30 };
          V.box(g, gpu.x, gpu.y, gpu.w, gpu.h, { r: 8, fill: V.alpha(C.a, k === 0 || k === 3 || k === 4 ? 0.18 : 0.08), stroke: V.alpha(C.a, 0.6), lw: 1 });
          tfit(g, 'GPU (kernel)', gpu.x + 6, gpu.y + 14, gpu.w - 12, { size: 11, weight: 600, color: C.a });
          // blocos do kernel (2 × 4)
          const nb = 8, bcols = 4, bs = Math.min(22, (gpu.w - 16 - 3 * 4) / bcols, (gpu.h - 60) / 2 - 4);
          const bx0 = gpu.x + (gpu.w - (bcols * bs + 3 * 4)) / 2, by0 = gpu.y + 24;
          for (let q = 0; q < nb; q++) {
            const qx = bx0 + (q % bcols) * (bs + 4), qy = by0 + Math.floor(q / bcols) * (bs + 4);
            let fill = V.alpha(C.a, 0.12), stroke = V.alpha(C.a, 0.4);
            if (k === 0 && q < st * nb * 1.2) { fill = V.alpha(col, 0.6); stroke = col; }
            if (k === 1 && q === nb - 1) { fill = V.alpha(C.d, 0.8); stroke = C.d; }
            if (k === 3) { const half = q < nb / 2; const on = (q % (nb / 2)) < st * nb * 0.6; if (on) { fill = V.alpha(half ? C.a : C.b, 0.6); stroke = half ? C.a : C.b; } }
            if (k === 4 && q === nb - 1) { fill = V.alpha(C.d, 0.8); stroke = C.d; }
            V.box(g, qx, qy, bs, bs, { r: 3, fill, stroke, lw: 1 });
          }
          const capY = by0 + 2 * (bs + 4) + 14;
          const cap = ['copies the local shard', 'last block: doorbell', 'waits for the peer flag', 'shards in rank order', 'advances the epoch'][k];
          V.wrap(g, cap, gpu.x + 6, capY, gpu.w - 12, { size: 10.5, color: C.muted, lineH: 13, maxLines: 2 });
          if (k === 2) { g.strokeStyle = C.a; g.lineWidth = 2; g.beginPath(); g.arc(gpu.x + gpu.w / 2, capY + 30, 9, t * 8, t * 8 + 4.2); g.stroke(); }
          if (k === 4) tfit(g, 'epoch ' + (Math.floor(el / (SD * 5)) + 1), gpu.x + 6, capY + 34, gpu.w - 12, { size: 11.5, font: 'mono', color: C.d, weight: 700 });
          V.box(g, mem.x, mem.y, mem.w, mem.h, { r: 8, fill: V.alpha(C.b, 0.08), stroke: V.alpha(C.b, 0.55), lw: 1 });
          tfit(g, 'pinned memory', mem.x + 6, mem.y + 14, mem.w - 12, { size: 11, weight: 600, color: C.b });
          const slots = ['send[0]', 'send[1]', 'recv[1]', 'flag', 'ctrl.seq'];
          const sh = Math.min(15, (mem.h - 22) / slots.length - 2);
          slots.forEach((s, j) => {
            const hot = (k === 0 && j === 0) || (k === 1 && (j === 0 || j === 4)) || (k === 2 && (j === 2 || j === 3)) || (k === 3 && j === 2);
            chip(g, mem.x + 6, mem.y + 20 + j * (sh + 2), mem.w - 12, sh, s, { fill: V.alpha(hot ? C.d : C.b, hot ? 0.35 : 0.1), stroke: V.alpha(hot ? C.d : C.b, 0.6), size: 10, min: 8, r: 3 });
          });
          return { gpu, mem };
        };
        const a = spark(A, 'Spark A · rank 0', C.a), b = spark(B, 'Spark B · rank 1', C.b);
        // trilhos
        const rails = narrow ? [[A.x + A.w * 0.35, A.y + A.h, B.x + B.w * 0.35, B.y], [A.x + A.w * 0.65, A.y + A.h, B.x + B.w * 0.65, B.y]] : [[A.x + A.w, A.y + A.h * 0.4, B.x, B.y + B.h * 0.4], [A.x + A.w, A.y + A.h * 0.62, B.x, B.y + B.h * 0.62]];
        rails.forEach(([x1, y1, x2, y2], j) => {
          V.line(g, x1, y1, x2, y2, { color: V.alpha(tf, 0.7), lw: 2.5 });
          V.text(g, 'HCA ' + (j + 1), narrow ? x1 + 8 : (x1 + x2) / 2, narrow ? (y1 + y2) / 2 + 4 : y1 - 6, { size: 10.5, font: 'mono', align: narrow ? 'left' : 'center', color: tf });
          if (k === 1) {
            const u = st;
            const p = V.along(x1, y1, x2, y2, u), q = V.along(x1, y1, x2, y2, Math.max(0, u - 0.18));
            V.box(g, p[0] - 9, p[1] - 5, 18, 10, { r: 2, fill: V.alpha(C.d, 0.8), stroke: C.d, lw: 1 });
            V.box(g, q[0] - 4, q[1] - 4, 8, 8, { r: 2, fill: tf, stroke: tf, lw: 1 });
          }
        });
        if (k === 1 && !narrow) { V.wrap(g, 'payload per stripe + 4 B flag on the same QP', A.x + A.w + 10, A.y + A.h * 0.78, B.x - A.x - A.w - 20, { size: 10.5, color: C.muted, lineH: 13, maxLines: 3 }); }
        // latências medidas
        const by = bottom + 18, bw = w - 2 * pad;
        V.text(g, 'latency per small all-gather (measured)', pad, by, { size: 11.5, weight: 600, color: C.ink });
        const BARS = [['one-shot RoCE · 16 KiB', 11, C.a], ['NCCL 4 channels*', 32, C.b], ['NCCL · 16 KiB', 45, C.glm], ['NCCL defaults*', 80, C.muted]];
        const labW = narrow ? 150 : 170, sc = (bw - labW - 50) / 80;
        BARS.forEach(([lab, us, col], j) => {
          const y = by + 10 + j * 20;
          tfit(g, lab, pad, y + 11, labW - 8, { size: 11, color: C.muted });
          g.fillStyle = V.alpha(col, 0.7); V.rr(g, pad + labW, y + 2, us * sc, 13, 3); g.fill();
          V.text(g, us + ' µs', pad + labW + us * sc + 6, y + 12, { size: 11, font: 'mono', color: C.ink });
        });
        V.text(g, '* “decode-sized” gather (nodes.sh)', pad, by + 10 + 4 * 20 + 10, { size: 10.5, color: C.muted, maxW: bw });
      });
    })();

    /* =================================================== FIG 10 — balança */
    (function () {
      const fig = root.querySelector('#fig-balanca');
      adaptH(fig, 350, 340);
      const REQ = [
        { id: 'h1', lab: 'logprobs, n>1 or penalties', side: 'v', hard: true, short: 'logprobs' },
        { id: 'h2', lab: 'more than 3 Sparks', side: 'v', hard: true, short: '> 3 nodes' },
        { id: 'h3', lab: 'more than 8 requests at once', side: 'v', hard: true, short: '> 8 req.' },
        { id: 't1', lab: 'agent that resends the history', side: 't', w: 3, short: 'history' },
        { id: 't2', lab: 'fast decode for 1 user', side: 't', w: 2, short: '1 user' },
        { id: 't3', lab: 'same question → same answer', side: 't', w: 2, short: 'reproducible' },
        { id: 't4', lab: '1M window for several conversations', side: 't', w: 2, short: '1M × 4' },
        { id: 'v1', lab: 'API key on the server itself', side: 'v', w: 1, short: 'API key' },
        { id: 'v2', lab: 'several fresh long prompts at once', side: 'v', w: 2, short: 'long prompts' },
        { id: 'v3', lab: 'fast restarts, no patch rebasing', side: 'v', w: 1, short: 'restart' },
      ];
      const on = { t1: true, t2: true, t4: true };
      const S = { ang: 0, vel: 0 };
      const ctl = UI.controls(fig);
      REQ.forEach((r) => UI.toggle(ctl, { label: (r.hard ? '🔒 ' : '') + r.lab, value: !!on[r.id], onChange: (v) => { on[r.id] = v; upd(); } }));
      const rV = UI.readout(ctl, 'verdict', '—');
      function state() {
        const hard = REQ.filter((r) => r.hard && on[r.id]);
        const wt = REQ.filter((r) => !r.hard && r.side === 't' && on[r.id]).reduce((a, r) => a + r.w, 0);
        const wv = REQ.filter((r) => !r.hard && r.side === 'v' && on[r.id]).reduce((a, r) => a + r.w, 0) + hard.length * 4;
        return { hard, wt, wv };
      }
      function upd() {
        const s = state();
        if (s.hard.length) rV.set('vLLM (TensorFold does not offer: ' + s.hard.map((r) => r.short).join(', ') + ')', 'bad');
        else if (s.wt > s.wv) rV.set('tips toward TensorFold', 'ok');
        else if (s.wv > s.wt) rV.set('tips toward vLLM', 'warn');
        else rV.set('tie: measure (chapter 45)');
      }
      upd();
      V.fig(fig, (g, w, h, t, dt) => {
        const C = V.C(), tf = TFC();
        const pad = 12, narrow = w < 560;
        const s = state();
        const target = s.hard.length ? -0.3 : V.clamp((s.wt - s.wv) * 0.06, -0.28, 0.28);
        const k = 40, d = 9;
        S.vel += (k * (target - S.ang) - d * S.vel) * Math.min(dt, 0.033);
        S.ang += S.vel * Math.min(dt, 0.033);
        const pwN = narrow ? Math.min(150, w * 0.4) : 190;
        const cx = w / 2, cy = narrow ? 84 : 88, L = narrow ? w / 2 - pwN / 2 - pad : Math.min(w * 0.38, 260);
        // poste
        g.fillStyle = V.alpha(C.muted, 0.5); g.beginPath(); g.moveTo(cx, cy); g.lineTo(cx - 26, h - 20); g.lineTo(cx + 26, h - 20); g.closePath(); g.fill();
        V.dot(g, cx, cy, 6, C.ink);
        // ângulo positivo = prato direito (TensorFold) desce
        const ca = Math.cos(S.ang), sa = Math.sin(S.ang);
        const PL = [cx - L * ca, cy - L * sa], PR = [cx + L * ca, cy + L * sa];
        V.line(g, PL[0], PL[1], PR[0], PR[1], { color: C.ink, lw: 4 });
        const pan = (P, label, col, items, locked, side) => {
          const pw = pwN, bwid = pw - 24, bh = 17;
          // o travessão desce de P em direção ao poste quando este prato está no alto: afasta a pilha dele
          const up = side < 0 ? S.ang > 0 : S.ang < 0;
          const clr = (up ? (bwid / 2) * Math.abs(Math.tan(S.ang)) : 0) + 10;
          const py = P[1] + Math.max(70, clr + items.length * (bh + 2) + 6);
          V.line(g, P[0], P[1], P[0] - pw / 2 + 10, py, { color: V.alpha(C.muted, 0.8), lw: 1.2 });
          V.line(g, P[0], P[1], P[0] + pw / 2 - 10, py, { color: V.alpha(C.muted, 0.8), lw: 1.2 });
          g.strokeStyle = col; g.lineWidth = 3; g.beginPath(); g.moveTo(P[0] - pw / 2, py); g.quadraticCurveTo(P[0], py + 18, P[0] + pw / 2, py); g.stroke();
          V.text(g, label, P[0], py + 32, { size: 13, weight: 700, align: 'center', color: col });
          let yy = py - 4;
          items.forEach((it) => {
            yy -= bh + 2;
            V.box(g, P[0] - bwid / 2, yy, bwid, bh, { r: 4, fill: V.alpha(it.hard ? RED() : col, it.hard ? 0.3 : 0.25), stroke: it.hard ? RED() : col, lw: 1 });
            tfit(g, (it.hard ? '🔒 ' : '') + it.short, P[0], yy + bh / 2 + 1, bwid - 8, { size: 11, align: 'center', base: 'middle', color: C.ink, font: 'sans', weight: 600 });
          });
          if (locked) { V.box(g, P[0] - 44, py + 40, 88, 22, { r: 6, fill: V.alpha(RED(), 0.2), stroke: RED() }); V.text(g, 'out', P[0], py + 55, { size: 12, weight: 700, align: 'center', color: RED() }); }
        };
        const vItems = REQ.filter((r) => r.side === 'v' && on[r.id]);
        const tItems = REQ.filter((r) => r.side === 't' && on[r.id]);
        pan(PL, 'vLLM', C.glm, vItems, false, -1);
        pan(PR, 'TensorFold', tf, tItems, s.hard.length > 0, 1);
        const verdict = s.hard.length ? 'requirement impossible in TensorFold → vLLM' : s.wt > s.wv ? 'the scale tips toward TensorFold' : s.wv > s.wt ? 'the scale tips toward vLLM' : 'tie: measure on your workload';
        tfit(g, verdict, cx, 22, w - 2 * pad, { size: 13, weight: 700, align: 'center', color: s.hard.length ? RED() : s.wt > s.wv ? tf : s.wv > s.wt ? C.glm : C.ink });
        tfit(g, 'weights: TensorFold ' + s.wt + ' × vLLM ' + s.wv, cx, 40, w - 2 * pad, { size: 11, align: 'center', color: C.muted, font: 'mono' });
      });
    })();
  },
});

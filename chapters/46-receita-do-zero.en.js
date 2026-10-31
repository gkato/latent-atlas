/* 46 · Tutorial: your recipe for a freshly released model (MiMo-V2.6-Flash-MOPD) */
LA.chapter({
  id: 'receita-do-zero',
  part: 9,
  order: 46,
  title: 'Tutorial: your recipe for a freshly released model',
  navTitle: 'Recipe from scratch',
  subtitle: 'From the Hugging Face card to a reproducible ./start.sh: the eleven steps to serve MiMo-V2.6-Flash-MOPD on two DGX Sparks with vLLM, with every number tagged as measured, estimate or to measure.',
  minutes: 75,
  recipes: ['glm', 'qwen', 'ds'],
  tags: ['tutorial', 'recipe', 'MiMo-V2.6', 'triage', 'memory', 'overlay', 'parity', 'speculation', 'benchmark', 'A/B/A', 'MLX'],
  goals: [
    'Follow an <b>11-step</b> method, each step with a verifiable exit criterion',
    'Run <b>triage</b> on a checkpoint without downloading it and do the <b>memory math</b> for a hybrid model',
    'Choose checkpoint and topology, and write <b>fail-closed overlays</b> proven by truncated parity',
    'Justify every flag, choose the speculation <b>k</b> and decide tuning with <b>A/B/A</b>',
    'Package the recipe so that <b>someone else</b> can reproduce the numbers',
  ],
  prereqs: ['benchmarking'],
  summary: [
    'A new recipe follows 11 steps, and each one ends in a <b>verifiable artifact</b>: triage, predicted vs. measured, KLD, divisibility, parity, boot log, acceptance table, green smoke test, benchmark with a method, tuning by the rules and a reproducible README.',
    'Triage reads the <code>config.json</code> and the <b>safetensors headers via range request</b>: in MiMo it reveals MXFP4 experts under <code>quant_method: fp8</code>, QKV pre-sliced for TP=4, DiffKV 192/128 and sinks only in the SWA layers.',
    'Memory: GMU × 121.69 GiB = weights + overhead + pool. In MiMo only the 9 global layers grow with context (5,760 B/token/rank in FP8); the 39 SWA layers cost ~12.8 MB fixed per request and the DFlash drafter’s KV another ~11–12 MB; the hybrid allocator wastes ~11%.',
    'The official MXFP4 checkpoint is the default because it is the smallest ~4-bit one and the reference itself; any other one requires <b>KLD</b> against it on the same engine. TP=2 divides everything, but the pre-sliced QKV has to be reassembled without requantizing.',
    'Silent bugs are fought with <b>fail-closed overlays</b> (exact anchors, idempotence, “already upstream”, a marker in the log, the same bytes on both ranks) and with <b>truncated parity</b> (N layers, TP1 = TP2 ≈ HF).',
    'The optimal k depends on the workload (code and JSON call for a high k, prose for a low k), capture sizes have to be multiples of k+1, and the bandwidth ceiling (~43 tok/s without spec) tells you where you should land.',
    'Tuning only with a rule written beforehand (A/B/A, ≥ 3% on both sides, quality canary), and “done” means another pair of Sparks reproduces the README within ±5–10%.',
  ],
  quiz: [
    { q: 'Why does triage read the safetensors headers with range requests, instead of trusting the <code>config.json</code> alone?', o: ['Because config.json does not state the number of layers', 'Because the headers give the real dtypes and shapes, which can contradict the config, without downloading the ~170 GB', 'Because range requests are the only way to download from Hugging Face', 'Because vLLM requires the headers to register the architecture'], a: 1, e: 'The header (8 bytes of size + JSON) gives the dtype and shape of every tensor. In MiMo it reveals MXFP4 experts (U8 with two nibbles per byte) despite <code>quant_method: fp8</code>, and the QKV pre-sliced for TP=4. The config does not have that information, and the other options are false.' },
    { q: 'In MiMo at TP=2 with FP8 KV, how many bytes of KV <i>per context token</i> does each rank spend in the layers that grow with context (without the allocator’s padding)?', o: ['5,760 B', '49,920 B', '11,520 B', '99,840 B'], a: 0, e: 'Only the 9 global layers grow: 9 × (4/2 KV heads) × (192 + 128) × 1 byte = 5,760 B. The 49,920 B belong to the 39 SWA layers, but only for the window (a fixed cost per request); 11,520 B is the same math in BF16.' },
    { q: 'Why does the recipe use <code>--cudagraph-capture-sizes 8 16 24 32 40 48 56 64</code>?', o: ['Because the GB10 only captures multiples of 8', 'Because with DFlash k=7 each sequence in decode has 8 tokens (1 + 7), and up to 8 sequences make 64', 'Because the MNBT is 4096', 'Because those are the KV block sizes'], a: 1, e: 'Uniform decode with spec has query_len = k+1. With k=7, n sequences take up 8n tokens; the list covers n = 1…8 (<code>--max-num-seqs 8</code>). Changing k without changing the list makes batches get “padded” or run eager.' },
    { q: 'You turned on <code>--kv-cache-dtype fp8</code>, the pool doubled, but the needle fails whenever it sits more than ~128 tokens away from the question. Most likely cause?', o: ['FP8 lost too much precision', 'The model-level <code>sliding_window: 128</code> leaked into the global layers through the shared <code>cache_config</code>', 'The prefix cache returned the wrong blocks', 'The DFlash drafter has a 1024 window'], a: 1, e: 'It is the classic symptom of the leak: the global layers start seeing only the window. That is why the overlay forwards a <b>copy</b> of <code>cache_config</code> with <code>sliding_window=None</code>. The needle is the test that catches it, because on short prompts both layer types see the same context.' },
    { q: 'In the A/B/A, B came out 4% above A₁, but A₂ (measured after B) came out 1% above B. By the Step 9 rule, what do you do?', o: ['Accept: +4% clears the 3% threshold', 'Reject and record: B does not beat A₂ by ≥ 3%, so the apparent gain is drift', 'Accept if the GSM8K canary passes', 'Repeat only arm B'], a: 1, e: 'The rule requires ≥ 3% on both sides. A₂ shows that the system improved on its own over time (page cache, autotune, temperature): the B − A₁ difference is drift, not effect. The negative result goes into the CHANGELOG.' },
    { q: 'Why is the topology chosen for MiMo on 2 Sparks TP=2, and not PP=2?', o: ['Because PP does not work with MoE', 'Because at small batch sizes PP leaves one node idle while the other works, doubling decode latency, and TP=2 divides heads, KV heads and experts', 'Because TP=2 does not need all-reduce', 'Because the checkpoint was saved for PP'], a: 1, e: 'PP only passes activations between stages, but in decode at batch 1 each token crosses both nodes in series. TP=2 costs 96 all-reduces per step (~10% of the step over RoCE) and everything divides by 2. PP is for throughput with many streams.' },
  ],
  mount(root, LA) {
    const V = LA.viz, UI = LA.ui;
    const clamp = V.clamp, lerp = V.lerp;
    const setH = (fig, mob) => { if (fig.clientWidth < 600) fig.querySelector('.la-fig-stage').style.height = mob + 'px'; };
    const nf = (x, d = 0) => Number(x).toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d });
    /** texto com quebra de linha; devolve o y da próxima linha */
    function wrap(g, s, x, y, maxW, lh, o = {}) {
      g.font = V.font(o.size || 12, o.font || 'sans', o.weight || 400);
      const words = String(s).split(/\s+/);
      let line = '', yy = y;
      for (const wd of words) {
        const test = line ? line + ' ' + wd : wd;
        if (line && g.measureText(test).width > maxW) { V.text(g, line, x, yy, o); line = wd; yy += lh; } else line = test;
      }
      if (line) { V.text(g, line, x, yy, o); yy += lh; }
      return yy;
    }
    const onInk = (C) => (C.dark ? '#00210F' : '#ffffff');

    /* =====================================================================
       Figura 1 — o metrô da receita (stepper sincronizado)
       ===================================================================== */
    (function figMetro() {
      const fig = root.querySelector('#fig-metro');
      if (fig.clientWidth < 520) fig.querySelector('.la-fig-stage').style.height = '350px';
      const ST = [
        { n: 0, s: 'Triage', z: 0, obj: 'Find out what is different about the model and where the engine will break, before downloading 170 GB.', exit: 'Triage table and risk list, each risk tied to a PR or issue.', ch: [['checkpoints', 'Checkpoints'], ['tres-modelos', 'The three models']] },
        { n: 1, s: 'Memory', z: 0, obj: 'Know how much is left for KV and, from that, how much context and how many requests fit.', exit: 'Predicted vs. measured table (weights, pool, tokens) with a deviation of at most ±10%.', ch: [['memoria-unificada', 'Unified memory'], ['paged-attention-kv', 'PagedAttention &amp; KV']] },
        { n: 2, s: 'Weights', z: 0, obj: 'Choose the checkpoint with the best quality × memory × speed balance for this hardware.', exit: 'Checkpoint downloaded, verified and with a pinned revision; KLD against the official one if it is a different one.', ch: [['quantizacao-fundamentos', 'Quantization'], ['medindo-qualidade', 'Measuring quality']] },
        { n: 3, s: 'Topology', z: 0, obj: 'Decide how to split the model across the nodes (TP, EP, PP).', exit: 'Divisibility table with no ✘, chosen topology and alternatives noted down.', ch: [['paralelismos', 'Parallelism'], ['nccl-rdma', 'NCCL &amp; RDMA']] },
        { n: 4, s: 'Engine', z: 1, obj: 'Get vLLM to load the model correctly, and prove it numerically.', exit: 'Overlays with CPU tests; truncated parity TP1 = TP2 ≈ HF; markers in the log.', ch: [['vllm-arquitetura', 'vLLM architecture'], ['engenharia-de-receita', 'Recipe engineering']] },
        { n: 5, s: 'Launch', z: 1, obj: 'Head and worker commands in which every flag has a reason and every value comes from the .env.', exit: 'Boot on both nodes; expected backend, markers, KV and capture sizes in the log; rendered command saved.', ch: [['scheduler-batching', 'Scheduler'], ['cuda-graphs-warmup', 'CUDA graphs']] },
        { n: 6, s: 'Speculation', z: 1, obj: 'Choose the speculator and the k that maximize tok/s on your workload.', exit: 'Acceptance × k × category × context table; SPEC_METHOD/SPEC_K with rollback.', ch: [['especulacao-fundamentos', 'Speculation'], ['dflash-dspark', 'DFlash &amp; DSpark']] },
        { n: 7, s: 'Validation', z: 1, obj: 'Prove that the server answers correctly, not just quickly.', exit: 'tests/smoke.sh green; results in the README with date and commit.', ch: [['api-tools-estruturado', 'API, tools &amp; JSON'], ['medindo-qualidade', 'Measuring quality']] },
        { n: 8, s: 'Benchmark', z: 2, obj: 'Reproducible TTFT, TPOT and tok/s on the concurrency ladder and in prefill.', exit: 'Decode, cold prefill and acceptance tables, with method, date, commit and digest.', ch: [['benchmarking', 'Benchmarking'], ['gpu-por-dentro', 'Inside the GPU']] },
        { n: 9, s: 'Tuning', z: 2, obj: 'Improve a declared target without breaking the rest, with the rule written beforehand.', exit: 'docs/tuning-YYYY-MM-DD.md with accepted and rejected; .env.sample and CHANGELOG updated.', ch: [['benchmarking', 'Benchmarking']] },
        { n: 10, s: 'Packaging', z: 2, obj: 'Someone else reproduces everything with cp .env.sample .env &amp;&amp; ./start.sh.', exit: 'A second pair of Sparks reproduces the README within 5–10%.', ch: [['engenharia-de-receita', 'Recipe engineering']] },
      ];
      const link = (c) => `<a href="#/c/${c[0]}">${c[1]}</a>`;
      const st = UI.stepper(fig, ST.map((s) => ({
        t: `Step ${s.n} · ${s.s}`,
        d: `<p style="margin:.2em 0"><b>Goal.</b> ${s.obj}</p><p style="margin:.2em 0"><b>Exit criterion.</b> ${s.exit}</p><p style="margin:.2em 0">Concepts: ${s.ch.map(link).join(' · ')}</p>`,
      })), { autoplay: 6500 });
      const S = { p: 0 };
      V.fig(fig, (g, w, h, t, dt) => {
        const C = V.C();
        const zc = [C.b, C.a, C.c];
        const cols = w >= 760 ? 11 : w >= 520 ? 6 : 4;
        const rows = Math.ceil(ST.length / cols);
        const mx = cols === 11 ? 46 : 50;
        const dx = (w - 2 * mx) / (cols - 1);
        let P;
        if (rows === 1) {
          const y0 = Math.min(150, h * 0.44);
          P = ST.map((s, i) => [mx + i * dx, y0]);
        } else {
          const top = 64, rowH = (h - top - 10) / rows;
          P = ST.map((s, i) => { const r = Math.floor(i / cols), c = i % cols, cc = r % 2 ? cols - 1 - c : c; return [mx + cc * dx, top + r * rowH + 18]; });
        }
        // legenda das fases
        const leg = w < 520 ? [['plan', zc[0]], ['bring up', zc[1]], ['measure/ship', zc[2]]] : [['plan on paper (0–3)', zc[0]], ['bring it up right (4–7)', zc[1]], ['measure and ship (8–10)', zc[2]]];
        g.font = V.font(11.5, 'sans', 500);
        let lx = 16;
        leg.forEach(([l, c]) => { V.box(g, lx, 14, 12, 12, { r: 3, fill: c }); V.text(g, l, lx + 18, 24.5, { size: 11.5, color: C.muted, weight: 500 }); lx += g.measureText(l).width + 40; });
        // trilhos
        const segPath = (a, b) => {
          g.beginPath(); g.moveTo(a[0], a[1]);
          if (Math.abs(a[1] - b[1]) < 1) g.lineTo(b[0], b[1]);
          else { const side = a[0] > w / 2 ? 1 : -1, ox = a[0] + side * 30; g.bezierCurveTo(ox, a[1], ox, b[1], b[0], b[1]); }
        };
        g.lineCap = 'round';
        for (let i = 0; i < ST.length - 1; i++) { segPath(P[i], P[i + 1]); g.strokeStyle = V.alpha(zc[ST[i + 1].z], 0.8); g.lineWidth = 6; g.stroke(); }
        // pontilhado correndo por cima (fluxo)
        for (let i = 0; i < ST.length - 1; i++) { segPath(P[i], P[i + 1]); g.setLineDash([2, 10]); g.lineDashOffset = -t * 18; g.strokeStyle = V.alpha(C.bg || '#000', 0.55); g.lineWidth = 2; g.stroke(); g.setLineDash([]); }
        g.lineCap = 'butt';
        // caminhos de volta
        [[7, 4, 'failed? back to the engine', 1], [9, 5, 'tune → relaunch', -1]].forEach(([a, b, lab, up]) => {
          const A = P[a], B = P[b];
          if (Math.abs(A[1] - B[1]) > 1) {
            V.text(g, `↺ ${b}`, A[0] + 16, A[1] - 16, { size: 11, font: 'mono', color: C.muted });
            return;
          }
          const off = up > 0 ? -18 : 40, cv = up > 0 ? 34 : -26;
          V.arrow(g, A[0] - 6, A[1] + off, B[0] + 6, B[1] + off, { curve: cv, color: V.alpha(C.muted, 0.85), dash: [4, 4], lw: 1.4, head: 7 });
          V.text(g, lab, (A[0] + B[0]) / 2, A[1] + off - cv / 2 + (up > 0 ? -6 : 14), { size: 11, align: 'center', color: C.muted });
        });
        // estações
        P.forEach((p, i) => {
          const s = ST[i], col = zc[s.z], past = i < st.i;
          V.dot(g, p[0], p[1], 11, C.panel || C.bg);
          g.beginPath(); g.arc(p[0], p[1], 11, 0, Math.PI * 2); g.strokeStyle = col; g.lineWidth = 3; g.stroke();
          if (past) V.dot(g, p[0], p[1], 5, col);
          g.font = V.font(11.5, 'sans', i === st.i ? 600 : 400);
          const lw = g.measureText(s.s).width + 8, lxx = clamp(p[0], lw / 2 + 2, w - lw / 2 - 2);
          V.box(g, lxx - lw / 2, p[1] + 18, lw, 17, { r: 5, fill: V.alpha(C.bg || C.panel, 0.85) });
          V.text(g, s.s, lxx, p[1] + 30, { size: 11.5, align: 'center', color: i === st.i ? C.ink : C.muted, weight: i === st.i ? 600 : 400 });
        });
        // o trem-token
        S.p += (st.i - S.p) * Math.min(1, dt * 3.2);
        const i0 = clamp(Math.floor(S.p), 0, ST.length - 1), i1 = Math.min(ST.length - 1, i0 + 1), f = S.p - i0;
        const A = P[i0], B = P[i1];
        let tx, ty;
        if (Math.abs(A[1] - B[1]) < 1) { tx = lerp(A[0], B[0], f); ty = A[1]; } else { const side = A[0] > w / 2 ? 1 : -1, ox = A[0] + side * 30; [tx, ty] = V.bez(A, [ox, A[1]], [ox, B[1]], B, f); }
        const col = zc[ST[st.i].z];
        V.glow(g, tx, ty, 34, col, 0.8);
        V.box(g, tx - 19, ty - 13, 38, 26, { r: 12, fill: col });
        const blink = Math.sin(t * 2.3) > 0.96 ? 0.5 : 2.2;
        g.fillStyle = onInk(C);
        g.beginPath(); g.ellipse(tx - 7, ty - 4, 2.2, blink, 0, 0, Math.PI * 2); g.fill();
        g.beginPath(); g.ellipse(tx + 7, ty - 4, 2.2, blink, 0, 0, Math.PI * 2); g.fill();
        V.text(g, String(ST[st.i].n), tx, ty + 8.5, { size: 11, font: 'mono', weight: 700, align: 'center', color: onInk(C) });
        // placa da estação (só no layout de uma linha)
        if (rows === 1) {
          const s = ST[st.i];
          V.text(g, `Step ${s.n} · ${s.s}`, w / 2, h - 54, { size: 18, font: 'display', align: 'center', color: zc[s.z], weight: 600 });
          V.text(g, 'exit: ' + s.exit.replace(/&amp;/g, '&'), w / 2, h - 28, { size: 12.5, align: 'center', color: C.ink, maxW: w - 60 });
        }
      });
    })();

    /* =====================================================================
       Figura 2 — raio-X do checkpoint
       ===================================================================== */
    (function figRaiox() {
      const fig = root.querySelector('#fig-raiox');
      setH(fig, 620);
      const T = [
        { label: 'Global QKV (layer 0)', name: 'layers.0.self_attn.qkv_proj.weight', dtype: 'F8_E4M3', shape: [13568, 4096], kind: 'qkv', ga: true, bytes: 13568 * 4096, what: '4 chunks × (3072 Q + 192 K + 128 V) = 13,568 rows. Each chunk has 16 Q heads and 1 KV head: the checkpoint was saved sliced for TP=4 (<code>tp_size: 4</code> in the index). At TP=2 each rank joins 2 chunks (Step 3).' },
        { label: 'Global QKV scales', name: 'layers.0.self_attn.qkv_proj.weight_scale_inv', dtype: 'F32', shape: [108, 32], kind: 'scale', ga: true, bytes: 108 * 32 * 4, what: 'One F32 factor per 128×128 block: 4096/128 = 32 columns; 3392 rows per chunk = 26.5 blocks → 27 per chunk, 108 in total. Block 25 of each chunk covers K and V at the same time; block 26 is half empty.' },
        { label: 'SWA QKV (layer 1)', name: 'layers.1.self_attn.qkv_proj.weight', dtype: 'F8_E4M3', shape: [14848, 4096], kind: 'qkv', ga: false, bytes: 14848 * 4096, what: '4 × (3072 Q + 384 K + 256 V) = 14,848. The SWA layers have twice the KV heads (8): 2 per chunk, which broke the “1 chunk per KV head” assumption in the sharding (#57508).' },
        { label: 'SWA QKV scales', name: 'layers.1.self_attn.qkv_proj.weight_scale_inv', dtype: 'F32', shape: [116, 32], kind: 'scale', ga: false, bytes: 116 * 32 * 4, what: '3712 rows per chunk = exactly 29 blocks: here Q, K and V start on block boundaries.' },
        { label: 'Sink bias (SWA)', name: 'layers.1.self_attn.attention_sink_bias', dtype: 'BF16', shape: [64], kind: 'sink', bytes: 128, what: '64 values, one per Q head: an extra logit in the softmax that gives attention a place to “dump” mass. Only in the SWA layers; the backend must accept <code>sinks=</code> per layer.' },
        { label: 'Expert: gate_proj (MXFP4)', name: 'layers.1.mlp.experts.0.gate_proj.weight', dtype: 'U8', shape: [2048, 2048], kind: 'nibble', bytes: 2048 * 2048, what: 'U8 [2048, 2048] = 2048 rows × 4096 weights, two E2M1 nibbles per byte. It is stored MXFP4, even though <code>quant_method</code> says fp8. The nibble order (low = even element?) has to be confirmed against the kernel.' },
        { label: 'Expert: E8M0 scales', name: 'layers.1.mlp.experts.0.gate_proj.weight_scale', dtype: 'U8', shape: [2048, 128], kind: 'e8m0', bytes: 2048 * 128, what: 'One E8M0 byte (a power of 2) for every 32 weights: 4096/32 = 128 per row. Average cost: 4 + 8/32 = 4.25 bits per weight.' },
        { label: 'o_proj (BF16)', name: 'layers.0.self_attn.o_proj.weight', dtype: 'BF16', shape: [4096, 8192], kind: 'oproj', bytes: 4096 * 8192 * 2, what: 'Input 8192 = 64 heads × 128 (<code>v_head_dim</code>), output 4096. It is in <code>ignored_layers</code>: 48 layers + MTP ≈ 3.2 GB in BF16 read at every step, the target of <code>VLLM_MIMO_OPROJ_FP8</code> (#59083).' },
        { label: 'MTP QKV', name: 'mtp.layers.0.self_attn.qkv_proj.weight', dtype: 'F8_E4M3', shape: [14848, 4096], kind: 'qkv', ga: false, mtp: true, bytes: 14848 * 4096, what: 'Same shape as the SWA QKV: the MTP is an SWA-type layer, with a sink. The DFlash drafter, by contrast, has its own config (<code>dflash/config.json</code>).' },
      ];
      const bstr = (b) => (b >= 1e6 ? nf(b / 1e6, 1) + ' MB' : b >= 1e3 ? nf(b / 1e3, 1) + ' KB' : b + ' B');
      const S = { i: 0 };
      const ctl = UI.controls(fig);
      const info = LA.h('dl', { class: 'la-kv', style: { margin: '0', padding: '12px 18px 6px', borderTop: '1px solid var(--line)' } });
      fig.insertBefore(info, ctl);
      const upd = () => { const x = T[S.i]; info.innerHTML = `<dt>tensor</dt><dd><code>model.${x.name}</code></dd><dt>dtype · shape</dt><dd>${x.dtype} · [${x.shape.join(', ')}] · ${bstr(x.bytes)}</dd><dt>what it reveals</dt><dd>${x.what}</dd>`; };
      UI.select(ctl, { label: 'Tensor', options: T.map((x, i) => ({ v: String(i), label: x.label })), value: '0', onChange: (v) => { S.i = +v; upd(); } });
      upd();
      // dados ilustrativos pré-computados
      const rr = V.rng(46);
      const bytes = Array.from({ length: 48 }, () => rr.int(256));
      const sinks = Array.from({ length: 4 }, () => Array.from({ length: 16 }, () => rr.normal() * 1.4));
      const exps = Array.from({ length: 128 }, () => 118 + rr.int(9));
      const E2 = [0, 0.5, 1, 1.5, 2, 3, 4, 6];
      const e2m1 = (n) => (n & 8 ? -1 : 1) * E2[n & 7];

      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const x = T[S.i];
        const narrow = w < 560;
        // ---------- faixa do arquivo ----------
        const fx = 16, fw = w - 32, fy = 34, fh = 24;
        V.text(g, 'one .safetensors shard (~1.3 GB on average: 172.9 GB / 129 files)', fx, 16, { size: 11.5, color: C.muted, maxW: fw });
        const s1 = 40, s2 = Math.max(110, fw * 0.22);
        const pulse = 0.35 + 0.25 * Math.sin(t * 3);
        V.box(g, fx, fy, s1, fh, { r: 5, fill: V.alpha(C.d, pulse + 0.1), stroke: C.d, lw: 1 });
        V.text(g, '8 B', fx + s1 / 2, fy + fh / 2 + 4, { size: 11, align: 'center', font: 'mono' });
        V.box(g, fx + s1 + 3, fy, s2, fh, { r: 5, fill: V.alpha(C.b, pulse), stroke: C.b, lw: 1 });
        V.text(g, 'JSON header (N B)', fx + s1 + 3 + s2 / 2, fy + fh / 2 + 4, { size: 11, align: 'center', maxW: s2 - 8 });
        const d0 = fx + s1 + s2 + 6, dw = fw - s1 - s2 - 6;
        V.box(g, d0, fy, dw, fh, { r: 5, fill: V.alpha(C.muted, 0.08), stroke: V.alpha(C.muted, 0.35), lw: 1 });
        const nS = 36;
        for (let k = 1; k < nS; k++) { const xx = d0 + k * dw / nS; V.line(g, xx, fy + 4, xx, fy + fh - 4, { color: V.alpha(C.muted, 0.22), lw: 1 }); }
        const sel = (S.i * 7 + 5) % (nS - 3), sx = d0 + sel * dw / nS;
        g.fillStyle = V.alpha(C.c, 0.6); g.fillRect(sx + 1, fy + 3, dw / nS * 2.4, fh - 6);
        V.text(g, '↓ only this is downloaded (2 range requests)', fx, fy + fh + 15, { size: 11, color: C.d });
        if (!narrow) V.text(g, 'data_offsets of the chosen tensor', d0 + dw, fy + fh + 15, { size: 11, align: 'right', color: C.c });
        // ---------- visualização do tensor ----------
        const vy = 92, vh = h - vy - 10;
        const scanF = (t * 0.22) % 1;
        const scan = (x0, y0, ww, hh) => { const yy = y0 + scanF * hh; const gr = g.createLinearGradient(0, yy - 10, 0, yy + 10); gr.addColorStop(0, V.alpha(C.ink, 0)); gr.addColorStop(0.5, V.alpha(C.ink, 0.28)); gr.addColorStop(1, V.alpha(C.ink, 0)); g.fillStyle = gr; g.fillRect(x0, yy - 10, ww, 20); };
        const colQ = C.b, colK = C.a, colV = C.c;

        if (x.kind === 'qkv') {
          const ga = x.ga, R = x.shape[0], cr = R / 4, q = 3072, kk = ga ? 192 : 384, vv = ga ? 128 : 256, nkv = ga ? 1 : 2;
          const bx = 40, bw = narrow ? Math.min(90, w * 0.26) : Math.min(150, w * 0.24), top = vy + 26, HH = vh - 40;
          const sc = HH / R;
          V.text(g, `[${nf(R)} × 4096] ${x.dtype}`, bx + bw / 2, vy + 12, { size: 11, font: 'mono', align: 'center', color: C.muted });
          for (let c = 0; c < 4; c++) {
            const y0 = top + c * cr * sc;
            V.box(g, bx, y0, bw, q * sc, { r: 2, fill: V.alpha(colQ, 0.55) });
            V.box(g, bx, y0 + q * sc, bw, Math.max(2, kk * sc), { r: 1, fill: V.alpha(colK, 0.85) });
            V.box(g, bx, y0 + (q + kk) * sc, bw, Math.max(2, vv * sc), { r: 1, fill: V.alpha(colV, 0.85) });
            V.line(g, bx - 4, y0, bx + bw + 4, y0, { color: C.ink, lw: c ? 1.5 : 0.8 });
            V.text(g, 'c' + c, bx - 8, y0 + cr * sc / 2 + 4, { size: 11, font: 'mono', align: 'right', color: C.muted });
          }
          scan(bx, top, bw, HH);
          // zoom do chunk 0
          const zx = bx + bw + (narrow ? 40 : 70), zw = Math.min(narrow ? w - zx - 12 : 170, w - zx - 150 > 0 ? 170 : w - zx - 12), zs = HH / cr;
          g.strokeStyle = V.alpha(C.muted, 0.5); g.lineWidth = 1; g.setLineDash([3, 3]);
          g.beginPath(); g.moveTo(bx + bw, top); g.lineTo(zx, top); g.moveTo(bx + bw, top + cr * sc); g.lineTo(zx, top + HH); g.stroke(); g.setLineDash([]);
          V.text(g, 'zoom on chunk 0', zx + zw / 2, vy + 12, { size: 11, font: 'mono', align: 'center', color: C.muted });
          for (let hq = 0; hq < 16; hq++) {
            const y0 = top + hq * 192 * zs;
            V.box(g, zx, y0 + 0.5, zw, 192 * zs - 1, { r: 2, fill: V.alpha(colQ, 0.25 + 0.3 * ((hq % 2) ? 1 : 0.6)) });
            if (192 * zs >= 11) V.text(g, 'Q' + hq, zx + 6, y0 + 192 * zs / 2 + 4, { size: 11, font: 'mono', color: C.ink });
          }
          for (let k = 0; k < nkv; k++) { const y0 = top + (q + k * 192) * zs; V.box(g, zx, y0 + 0.5, zw, 192 * zs - 1, { r: 2, fill: V.alpha(colK, 0.7) }); V.text(g, 'K' + k, zx + 6, y0 + 192 * zs / 2 + 4, { size: 11, font: 'mono', color: C.ink }); }
          for (let k = 0; k < nkv; k++) { const y0 = top + (q + kk + k * 128) * zs; V.box(g, zx, y0 + 0.5, zw, 128 * zs - 1, { r: 2, fill: V.alpha(colV, 0.7) }); V.text(g, 'V' + k, zx + 6, y0 + 128 * zs / 2 + 4, { size: 11, font: 'mono', color: C.ink }); }
          // grade de blocos de escala (128 linhas)
          for (let m = 1; m * 128 < cr; m++) { const yy = top + m * 128 * zs; V.line(g, zx + zw * 0.55, yy, zx + zw, yy, { color: V.alpha(C.ink, 0.35), lw: 1 }); }
          if (ga) {
            const y0 = top + 3200 * zs, y1 = top + 3328 * zs;
            g.strokeStyle = C.d; g.lineWidth = 2; V.rr(g, zx - 3, y0, zw + 6, y1 - y0, 3); g.stroke();
          }
          scan(zx, top, zw, HH);
          // legenda à direita do zoom
          const lx2 = zx + zw + 14;
          if (w - lx2 > 120) {
            let yy = top + 8;
            yy = wrap(g, ga ? 'global: 16 Q heads, 1 KV head per chunk' : (x.mtp ? 'MTP: same shape as SWA (has a sink)' : 'SWA: 16 Q heads, 2 KV heads per chunk'), lx2, yy, w - lx2 - 10, 15, { size: 11.5, color: C.ink });
            yy = wrap(g, `chunk = 3072 + ${kk} + ${vv} = ${cr} rows`, lx2, yy + 6, w - lx2 - 10, 15, { size: 11.5, font: 'mono', color: C.muted });
            yy = wrap(g, 'ticks on the right: 128-row scale blocks', lx2, yy + 6, w - lx2 - 10, 15, { size: 11.5, color: C.muted });
            if (ga) wrap(g, 'amber outline: block 25 covers the last 64 rows of K and the first 64 of V', lx2, yy + 6, w - lx2 - 10, 15, { size: 11.5, color: C.d });
            [[colQ, 'Q'], [colK, 'K'], [colV, 'V']].forEach(([c, l], i) => { V.box(g, lx2 + i * 44, top + HH - 16, 12, 12, { r: 3, fill: c }); V.text(g, l, lx2 + i * 44 + 17, top + HH - 6, { size: 11.5, color: C.muted }); });
          }
        } else if (x.kind === 'scale') {
          const ga = x.ga, nb = ga ? 27 : 29;
          const gx = 44, gw = Math.min(narrow ? w - 80 : w * 0.5, 380), top = vy + 26, gh = narrow ? vh * 0.6 : vh - 70;
          const ch = gh / nb, cw = gw / 32;
          V.text(g, `chunk 0 in detail: ${nb} row blocks × 32 column blocks`, gx + gw / 2, vy + 12, { size: 11, font: 'mono', align: 'center', color: C.muted, maxW: gw + 40 });
          const typ = (r) => (ga ? (r < 24 ? 'Q' : r === 24 ? 'K' : r === 25 ? 'KV' : 'Vp') : (r < 24 ? 'Q' : r < 27 ? 'K' : 'V'));
          const colOf = (ty) => (ty === 'Q' ? colQ : ty === 'K' ? colK : ty === 'KV' ? C.d : colV);
          for (let r = 0; r < nb; r++) {
            const ty = typ(r), yy = top + r * ch;
            g.fillStyle = V.alpha(colOf(ty), ty === 'KV' ? 0.9 : 0.6);
            g.fillRect(gx, yy, gw, ty === 'Vp' ? ch * 0.5 : ch);
            if (ty === 'Vp') { g.fillStyle = V.alpha(C.muted, 0.12); g.fillRect(gx, yy + ch * 0.5, gw, ch * 0.5); }
            if (r === 0 || r === 23 || r >= 24) V.text(g, String(r), gx - 6, yy + ch / 2 + 4, { size: 11, font: 'mono', align: 'right', color: r >= 24 ? C.ink : C.muted });
          }
          g.strokeStyle = V.alpha(C.bg || '#000', 0.7); g.lineWidth = 1; g.beginPath();
          for (let j = 1; j < 32; j++) { g.moveTo(gx + j * cw, top); g.lineTo(gx + j * cw, top + nb * ch); }
          for (let r = 1; r < nb; r++) { g.moveTo(gx, top + r * ch); g.lineTo(gx + gw, top + r * ch); }
          g.stroke();
          V.text(g, `× 4 chunks (c1–c3 repeat the pattern) = ${4 * nb} scale rows · [${x.shape[0]} × 32] F32`, gx, top + nb * ch + 18, { size: 11.5, color: C.muted, maxW: gw + 30 });
          // célula destacada que passeia
          const hr = Math.floor(t * 1.2) % nb, hj = Math.floor(t * 3.7) % 32;
          const hy = top + hr * ch;
          g.strokeStyle = C.ink; g.lineWidth = 2; g.strokeRect(gx + hj * cw, hy, cw, ch);
          const lx2 = narrow ? 16 : gx + gw + 24, ly = narrow ? top + gh + 42 : top + 8, lw2 = narrow ? w - 32 : w - lx2 - 12;
          let yy = wrap(g, `highlighted cell: row block ${hr}, columns ${hj * 128}–${hj * 128 + 127} → a single F32 factor multiplies 128 × 128 FP8 weights`, lx2, ly, lw2, 15, { size: 11.5, color: C.ink });
          const items = ga ? [[colQ, 'Q: blocks 0–23'], [colK, 'K: block 24'], [C.d, 'K + V: block 25 (one scale for both)'], [colV, 'V: block 26 (only 64 rows)']] : [[colQ, 'Q: blocks 0–23'], [colK, 'K: blocks 24–26'], [colV, 'V: blocks 27–28']];
          items.forEach(([c, l]) => { yy += 4; V.box(g, lx2, yy - 10, 12, 12, { r: 3, fill: c }); yy = wrap(g, l, lx2 + 18, yy, lw2 - 18, 15, { size: 11.5, color: C.muted }); });
        } else if (x.kind === 'sink') {
          const cw = Math.min(26, (narrow ? w - 60 : w * 0.46) / 16), gx = 40, top = vy + 30;
          V.text(g, '64 sink logits (illustrative values), one per head', gx, vy + 14, { size: 11.5, color: C.muted });
          V.matrix(g, { x: gx, y: top, data: sinks, cell: cw, cmap: 'div', gap: 2 });
          V.text(g, 'h0', gx, top + 4 * (cw + 2) + 14, { size: 11, font: 'mono', color: C.muted });
          V.text(g, 'h63', gx + 16 * (cw + 2) - 2, top + 4 * (cw + 2) + 14, { size: 11, font: 'mono', align: 'right', color: C.muted });
          // softmax com e sem sink
          const sx0 = narrow ? 30 : gx + 16 * (cw + 2) + 50, sy0 = narrow ? top + 4 * (cw + 2) + 60 : top, sw = narrow ? w - 60 : w - sx0 - 20, sh = narrow ? vh - (sy0 - vy) - 50 : vh - 90;
          const sink = 1.2 + 2.2 * Math.sin(t * 0.9);
          const logits = [sink, 1.6, 1.1, 0.5, -0.2, 0.9];
          const p = V.softmax(logits);
          V.text(g, `softmax of a query with the sink in the denominator (sink = ${sink.toFixed(1)})`, sx0, sy0 - 8, { size: 11.5, color: C.ink, maxW: sw });
          V.bars(g, { x: sx0, y: sy0 + 18, w: sw, h: Math.max(60, sh - 40), values: p, max: Math.max(...p) * 1.15, labels: ['sink', 't1', 't2', 't3', 't4', 't5'], showValues: true, fmt: (v) => (v * 100).toFixed(0) + '%', colorFn: (v, i) => (i === 0 ? C.d : C.a), gap: 10 });
        } else if (x.kind === 'nibble') {
          const cols = narrow ? 6 : 8, rows = 5, cell = Math.min(34, ((narrow ? w - 60 : w * 0.42) - 10) / cols);
          const gx = 40, top = vy + 30;
          V.text(g, 'bytes of the U8 tensor (hex, illustrative)', gx, vy + 14, { size: 11.5, color: C.muted });
          const cur = Math.floor(t / 1.6) % (cols * rows);
          for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) {
            const k = i * cols + j, b = bytes[k % bytes.length], on = k === cur;
            V.box(g, gx + j * (cell + 3), top + i * (cell + 3), cell, cell, { r: 5, fill: on ? V.alpha(C.d, 0.55) : V.alpha(C.b, 0.18), stroke: on ? C.d : null, lw: 2 });
            V.text(g, b.toString(16).toUpperCase().padStart(2, '0'), gx + j * (cell + 3) + cell / 2, top + i * (cell + 3) + cell / 2 + 4, { size: 11, font: 'mono', align: 'center', color: C.ink });
          }
          const b = bytes[cur % bytes.length], lo = b & 15, hi2 = b >> 4, e = 122, s = Math.pow(2, e - 127);
          const px = narrow ? 24 : gx + cols * (cell + 3) + 40, py = narrow ? top + rows * (cell + 3) + 26 : top, pw = narrow ? w - 48 : w - px - 16;
          const bw = Math.min(28, pw / 9.5);
          V.text(g, `byte 0x${b.toString(16).toUpperCase().padStart(2, '0')} = two 4-bit weights`, px, py + 4, { size: 12, color: C.ink, weight: 600, maxW: pw });
          for (let k = 0; k < 8; k++) {
            const bit = (b >> (7 - k)) & 1, isHi = k < 4;
            V.box(g, px + k * (bw + 3) + (isHi ? 0 : 8), py + 16, bw, bw, { r: 4, fill: V.alpha(isHi ? C.c : C.a, bit ? 0.7 : 0.18), stroke: isHi ? C.c : C.a, lw: 1 });
            V.text(g, String(bit), px + k * (bw + 3) + (isHi ? 0 : 8) + bw / 2, py + 16 + bw / 2 + 4, { size: 11, font: 'mono', align: 'center', color: C.ink });
          }
          let yy = py + 16 + bw + 22;
          const line = (txt, col) => { yy = wrap(g, txt, px, yy, pw, 16, { size: 12, font: 'mono', color: col || C.ink }); yy += 2; };
          line(`low nibble  ${lo.toString(2).padStart(4, '0')} → E2M1 ${e2m1(lo)}`, C.a);
          line(`high nibble ${hi2.toString(2).padStart(4, '0')} → E2M1 ${e2m1(hi2)}`, C.c);
          line(`block E8M0 scale: e = ${e} → 2^(${e}−127) = 1/${1 / s}`, C.d);
          line(`weights: w[2j] = ${(e2m1(lo) * s).toFixed(4)} · w[2j+1] = ${(e2m1(hi2) * s).toFixed(4)}`);
          yy += 4;
          wrap(g, 'the “low nibble = even element” convention (OCP/gpt-oss): confirm it by comparing the dequantized GEMM with the kernel’s', px, yy, pw, 15, { size: 11.5, color: C.muted });
        } else if (x.kind === 'e8m0') {
          const gx = 40, top = vy + 30, gw = narrow ? w - 80 : w * 0.55, cw = gw / 16, chh = Math.min(26, (narrow ? vh * 0.45 : vh - 60) / 8);
          V.text(g, 'one row: 4096 weights = 128 blocks of 32 (8 rows × 16 blocks)', gx, vy + 14, { size: 11.5, color: C.muted, maxW: w - 50 });
          const cur = Math.floor(t / 1.2) % 128;
          for (let k = 0; k < 128; k++) {
            const r = Math.floor(k / 16), c = k % 16, e = exps[k], on = k === cur;
            V.box(g, gx + c * cw + 1, top + r * (chh + 3), cw - 2, chh, { r: 3, fill: V.cmap((e - 117) / 10, 'heat'), stroke: on ? C.ink : null, lw: 2 });
          }
          const e = exps[cur];
          const px = narrow ? 24 : gx + gw + 30, py = narrow ? top + 8 * (chh + 3) + 24 : top + 4, pw = narrow ? w - 48 : w - px - 16;
          let yy = wrap(g, `block ${cur}: weights ${cur * 32}–${cur * 32 + 31}`, px, py, pw, 16, { size: 12, color: C.ink, weight: 600 });
          yy = wrap(g, `e = ${e} → scale 2^(${e}−127) = 1/${Math.pow(2, 127 - e)}`, px, yy + 2, pw, 16, { size: 12, font: 'mono', color: C.d });
          yy = wrap(g, 'each of the 32 weights = E2M1(nibble) × scale', px, yy + 4, pw, 16, { size: 12, color: C.ink });
          yy = wrap(g, 'cost: (32 × 4 + 8) / 32 = 4.25 bits per weight', px, yy + 8, pw, 16, { size: 12, font: 'mono', color: C.a });
          wrap(g, 'color: the scale’s exponent (illustrative values). A pure power of 2: multiplying by it is just adding to the exponent', px, yy + 8, pw, 15, { size: 11.5, color: C.muted });
        } else if (x.kind === 'oproj') {
          const rw = Math.min(w - 90, 520), rh = Math.min(vh - 90, rw / 2), rx = (w - rw) / 2 + 10, ry = vy + 34;
          V.text(g, 'input: 8192 = 64 heads × 128 (v_head_dim)', rx + rw / 2, vy + 16, { size: 11.5, align: 'center', color: C.muted, maxW: rw });
          const hw = rw / 64;
          for (let k = 0; k < 64; k++) { g.fillStyle = V.alpha(k % 2 ? C.c : C.b, 0.35 + 0.15 * Math.sin(k)); g.fillRect(rx + k * hw, ry, hw + 0.5, rh); }
          g.strokeStyle = C.ink; g.lineWidth = 1; g.strokeRect(rx, ry, rw, rh);
          const sxx = rx + ((t * 0.18) % 1) * rw; const gr = g.createLinearGradient(sxx - 14, 0, sxx + 14, 0); gr.addColorStop(0, V.alpha(C.ink, 0)); gr.addColorStop(0.5, V.alpha(C.ink, 0.3)); gr.addColorStop(1, V.alpha(C.ink, 0)); g.fillStyle = gr; g.fillRect(sxx - 14, ry, 28, rh);
          V.text(g, 'output 4096', rx - 8, ry + rh / 2 + 4, { size: 11, align: 'right', font: 'mono', color: C.muted });
          wrap(g, 'BF16: 67.1 MB per layer × 48 (+ MTP) ≈ 3.2 GB read at every decode step. With online FP8 (#59083): −0.8 GiB per rank and fewer bytes per step.', rx, ry + rh + 24, rw, 16, { size: 12, color: C.ink });
        }
      });
    })();

    /* =====================================================================
       Figura 3 — calculadora de memória (mem_budget.py ao vivo)
       ===================================================================== */
    (function figCalc() {
      const fig = root.querySelector('#fig-calc');
      setH(fig, 500);
      const CFG = { nGA: 9, nSWA: 39, kvGA: 4, kvSWA: 8, dGA: 320, dSWA: 320, win: 128, vis: 121.69, weights: 82.7, block: 64, k: 7, tp: 2, nDr: 5, kvDr: 8, dDr: 256, winDr: 1024 };
      const S = { gmu: 0.85, over: 4, kv: 'fp8', group: 5, ctx: 262144, textOnly: false, draft: true, shown: 0, r: null };
      const ctxLabel = (L) => (L >= 1048576 ? '1M' : (L / 1024) + 'K');
      function calc() {
        const b = S.kv === 'bf16' ? 2 : 1;
        const gaTok = CFG.nGA * (CFG.kvGA / CFG.tp) * CFG.dGA * b;
        const swaTok = CFG.nSWA * (CFG.kvSWA / CFG.tp) * CFG.dSWA * b;
        const blocks = (win) => (Math.ceil((win + CFG.k) / CFG.block) + 1) * CFG.block;
        const swaReq = swaTok * blocks(CFG.win);
        const drTok = CFG.nDr * (CFG.kvDr / CFG.tp) * CFG.dDr * 2; // drafter em BF16
        const drReq = S.draft ? drTok * blocks(CFG.winDr) : 0;
        const pad = Math.ceil(CFG.nGA / S.group) * S.group / CFG.nGA;
        const weights = CFG.weights - (S.textOnly ? 1.6 : 0);
        const budget = CFG.vis * S.gmu;
        const kvGiB = budget - weights - S.over;
        const kvB = kvGiB * Math.pow(2, 30);
        const cap = kvB / (gaTok * pad), ideal = kvB / gaTok;
        const gaPart = S.ctx * gaTok * pad, perReq = gaPart + swaReq + drReq;
        return { b, gaTok, swaTok, swaReq, drTok, drReq, pad, weights, budget, kvGiB, cap, ideal, gaPart, perReq, nreq: kvB / perReq };
      }
      const ctl = UI.controls(fig);
      UI.slider(ctl, { label: 'GMU', min: 0.7, max: 0.95, step: 0.01, value: S.gmu, fmt: (v) => v.toFixed(2), onInput: (v) => { S.gmu = v; upd(); } });
      UI.slider(ctl, { label: 'Overhead', min: 2, max: 10, step: 0.5, value: S.over, fmt: (v) => v.toFixed(1) + ' GiB', onInput: (v) => { S.over = v; upd(); } });
      UI.seg(ctl, { label: 'KV', options: [{ v: 'bf16', label: 'BF16' }, { v: 'fp8', label: 'FP8' }], value: 'fp8', onChange: (v) => { S.kv = v; upd(); } });
      UI.seg(ctl, { label: 'Group', options: [{ v: 5, label: '5 (default)' }, { v: 3, label: '3 (#58207)' }], value: 5, onChange: (v) => { S.group = v; upd(); } });
      UI.seg(ctl, { label: 'Context', options: [{ v: 32768, label: '32K' }, { v: 131072, label: '128K' }, { v: 262144, label: '256K' }, { v: 1048576, label: '1M' }], value: 262144, onChange: (v) => { S.ctx = v; upd(); } });
      UI.toggle(ctl, { label: 'DFlash drafter KV (BF16)', value: true, onChange: (v) => { S.draft = v; upd(); } });
      UI.toggle(ctl, { label: '--language-model-only (−1.6 GiB of weights)', value: false, onChange: (v) => { S.textOnly = v; upd(); } });
      const rPool = UI.readout(ctl, 'KV pool', '—'), rCap = UI.readout(ctl, 'Capacity', '—'), rReq = UI.readout(ctl, 'Full requests', '—');
      function upd() {
        S.r = calc();
        const r = S.r;
        rPool.set(r.kvGiB > 0 ? nf(r.kvGiB, 2) + ' GiB' : 'none left');
        rCap.set(r.kvGiB > 0 ? '~' + nf(r.cap / 1e6, 2) + ' M tok' : '—');
        rReq.set(r.kvGiB > 0 ? nf(r.nreq, 1) + ' × ' + ctxLabel(S.ctx) : '—');
      }
      upd();
      V.fig(fig, (g, w, h, t, dt) => {
        const C = V.C(), r = S.r;
        const x0 = 16, bw = w - 32;
        // --- barra 1: memória visível de um rank
        V.text(g, 'VISIBLE MEMORY OF ONE RANK · 121.69 GiB [measured]', x0, 16, { size: 11, font: 'mono', color: C.muted, maxW: bw });
        const y1 = 26, bh = 34, sc = bw / CFG.vis;
        const segs = [['weights', r.weights, C.b], ['overhead', S.over, C.d], ['KV pool', Math.max(0, r.kvGiB), C.a], ['host', CFG.vis - r.budget, null]];
        let xx = x0;
        segs.forEach(([n, v, c]) => {
          const ww = Math.max(0, v * sc);
          if (c) V.box(g, xx, y1, Math.max(1, ww - 1), bh, { r: 4, fill: V.alpha(c, 0.7) });
          else {
            V.box(g, xx, y1, ww, bh, { r: 4, fill: V.alpha(C.muted, 0.1), stroke: V.alpha(C.muted, 0.45), lw: 1 });
            g.save(); V.rr(g, xx, y1, ww, bh, 4); g.clip(); g.strokeStyle = V.alpha(C.muted, 0.28); g.lineWidth = 1;
            for (let k = -bh; k < ww; k += 8) { g.beginPath(); g.moveTo(xx + k, y1 + bh); g.lineTo(xx + k + bh, y1); g.stroke(); }
            g.restore();
          }
          const lab = `${n} ${nf(v, 1)}`;
          g.font = V.font(11.5, 'sans', 600);
          if (g.measureText(lab).width + 10 < ww) V.text(g, lab, xx + ww / 2, y1 + bh / 2 + 4, { size: 11.5, weight: 600, align: 'center', color: c ? C.ink : C.muted });
          xx += ww;
        });
        const gx = x0 + r.budget * sc;
        V.line(g, gx, y1 - 5, gx, y1 + bh + 5, { color: C.ink, lw: 1.5, dash: [3, 3] });
        V.text(g, `GMU × total = ${nf(r.budget, 1)} GiB`, Math.min(gx, w - 16), y1 + bh + 16, { size: 11, font: 'mono', align: gx > w - 140 ? 'right' : 'center', color: C.ink });
        // --- barra 2: pool → requisições
        const y2 = 110, bh2 = 40;
        V.text(g, `KV POOL → FULL ${ctxLabel(S.ctx)} REQUESTS AT THE SAME TIME`, x0, y2 - 9, { size: 11, font: 'mono', color: C.muted, maxW: bw });
        V.box(g, x0, y2, bw, bh2, { r: 6, fill: V.alpha(C.a, 0.06), stroke: V.alpha(C.a, 0.5), lw: 1 });
        if (r.kvGiB <= 0) {
          V.text(g, '✕ no KV left: reduce weights or overhead, or raise the GMU', x0 + bw / 2, y2 + bh2 / 2 + 4, { size: 12.5, align: 'center', color: C.red, weight: 600, maxW: bw - 12 });
          S.shown = 0;
        } else {
          S.shown += (r.nreq - S.shown) * Math.min(1, dt * 6);
          const tw = bw / r.nreq, gaF = r.gaPart / r.perReq, swF = r.swaReq / r.perReq, full = Math.min(600, Math.floor(S.shown));
          const gp = tw > 5 ? 1.5 : tw > 2.5 ? 0.6 : 0;
          for (let k = 0; k < full; k++) {
            const tx = x0 + k * tw;
            g.fillStyle = V.alpha(C.a, 0.75); g.fillRect(tx + gp / 2, y2 + 4, Math.max(0.5, tw * gaF - gp), bh2 - 8);
            g.fillStyle = V.alpha(C.c, 0.9); g.fillRect(tx + tw * gaF - gp / 2, y2 + 4, Math.max(0.6, tw * swF), bh2 - 8);
            if (r.drReq) { g.fillStyle = V.alpha(C.f, 0.9); g.fillRect(tx + tw * (gaF + swF) - gp / 2, y2 + 4, Math.max(0.6, tw * (1 - gaF - swF)), bh2 - 8); }
          }
          const rest = x0 + full * tw;
          if (rest < x0 + bw - 2) { g.setLineDash([3, 3]); g.strokeStyle = V.alpha(C.muted, 0.6); g.strokeRect(rest + 1, y2 + 4, x0 + bw - rest - 5, bh2 - 8); g.setLineDash([]); }
        }
        // --- barra 3: anatomia de uma requisição
        const y3 = 196;
        V.text(g, `ANATOMY OF ONE ${ctxLabel(S.ctx)} REQUEST (${S.kv.toUpperCase()})`, x0, y3 - 9, { size: 11, font: 'mono', color: C.muted, maxW: bw });
        const gaF = r.gaPart / r.perReq, gw = Math.max(4, bw * gaF);
        const fixW = Math.max(r.drReq ? 8 : 4, bw - gw - 2), swW = r.drReq ? Math.max(4, fixW * r.swaReq / (r.swaReq + r.drReq)) : fixW;
        V.box(g, x0, y3, gw, 30, { r: 5, fill: V.alpha(C.a, 0.7) });
        V.box(g, x0 + gw + 2, y3, swW, 30, { r: 5, fill: V.alpha(C.c, 0.85) });
        if (r.drReq) V.box(g, x0 + gw + 2 + swW, y3, Math.max(3, fixW - swW), 30, { r: 5, fill: V.alpha(C.f, 0.85) });
        const gaTxt = `global: ${nf(S.ctx)} × ${nf(r.gaTok * r.pad)} B = ${nf(r.gaPart / 1e6, 0)} MB`;
        g.font = V.font(11.5, 'sans', 600);
        if (g.measureText(gaTxt).width + 12 < gw) V.text(g, gaTxt, x0 + 8, y3 + 19.5, { size: 11.5, weight: 600, color: C.ink });
        let yy = y3 + 50;
        yy = wrap(g, `${nf(r.gaTok)} B/token/rank × ${nf(r.pad, 2)} padding (group ${S.group}) = ${nf(r.gaTok * r.pad)} B · fixed: SWA ${nf(r.swaReq / 1e6, 1)} MB (window 128 + k 7, blocks of 64)` + (r.drReq ? ` + drafter ${nf(r.drReq / 1e6, 1)} MB (window 1024, BF16)` : ' (no drafter)') + ` = ${nf(100 * (1 - gaF), 1)}% of the request`, x0, yy, bw, 16, { size: 12, color: C.ink });
        if (r.kvGiB > 0) yy = wrap(g, `capacity ≈ ${nf(r.cap / 1e6, 2)} M tokens (without padding: ${nf(r.ideal / 1e6, 2)} M) · ${nf(r.nreq, 1)} full ${ctxLabel(S.ctx)} requests`, x0, yy + 4, bw, 16, { size: 12, color: C.a, weight: 600 });
        yy = wrap(g, `at boot vLLM requires ≥ ${nf(r.budget, 1)} GiB free; outside the budget, ${nf(CFG.vis - r.budget, 1)} GiB remain for the OS, page cache, API and multimodal processor`, x0, yy + 4, bw, 16, { size: 12, color: C.muted });
        if (S.gmu >= 0.895) yy = wrap(g, '⚠ GMU ≥ 0.90: above what the community uses on UMA (0.80–0.85)', x0, yy + 4, bw, 16, { size: 12, color: C.d, weight: 600 });
        yy = wrap(g, 'reference [measured, #58128, 17 GiB pool]: BF16 1,365,757 · FP8 2,731,530 tokens', x0, yy + 4, bw, 16, { size: 11.5, font: 'mono', color: C.muted });
        // legenda
        const ly = Math.min(h - (bw < 480 ? 26 : 10), yy + 16);
        const legW = Math.min(120, bw / 3), legN = bw < 480 ? 3 : 5;
        [[C.b, 'weights'], [C.d, 'overhead'], [C.a, 'global KV'], [C.c, 'SWA KV (fixed)'], [C.f, 'drafter KV (fixed)']].forEach(([c, l], i) => { const lx = x0 + (i % legN) * (bw < 480 ? legW : Math.min(120, bw / 5)), lyy = ly + Math.floor(i / legN) * 16; V.box(g, lx, lyy - 10, 11, 11, { r: 3, fill: c }); V.text(g, l, lx + 16, lyy, { size: 11, color: C.muted }); });
      });
    })();

    /* =====================================================================
       Figura 4 — árvore de decisão: formato e topologia
       ===================================================================== */
    (function figTree() {
      const fig = root.querySelector('#fig-arvore');
      setH(fig, 600);
      const N = {
        Q1: { q: true, row: 0, t: 'Does any published ~4-bit checkpoint fit on the nodes with room left for KV?' },
        Q2: { q: true, row: 1, t: 'Is the smallest one that fits the official one?' },
        Q3: { q: true, row: 2, t: 'Is there a kernel for that format on sm_121?' },
        Q4: { q: true, row: 4, t: 'Do heads, KV heads, experts and moe_intermediate divide by TP = no. of nodes?' },
        Q5: { q: true, row: 5, t: 'Do the weights come pre-sliced for another TP?' },
        A: { row: 0, t: 'Quantize it yourself; measure KLD vs. the reference' },
        B: { row: 1, t: 'Community checkpoint + KLD against the official one' },
        C: { row: 2, t: 'Kernel exists: move on to topology' },
        D: { row: 3, t: 'Overlay or kernel port (e.g. exl3 in GLM)' },
        E: { row: 4, t: 'Replicate KV heads, padding or another TP' },
        F: { row: 5, t: 'Reassemble chunks: exact path + shard test' },
        G: { q: true, row: 6, g: true, t: 'TP = nodes; EP as A/B; PP only for throughput' },
      };
      const EDGES = [['Q1', 'Q2', 'yes'], ['Q2', 'Q3', 'yes'], ['Q3', 'Q4', ''], ['Q4', 'Q5', 'yes'], ['Q5', 'G', 'no'], ['Q1', 'A', 'no'], ['Q2', 'B', 'no'], ['Q3', 'C', 'yes'], ['Q3', 'D', 'no'], ['Q4', 'E', 'no'], ['Q5', 'F', 'yes']];
      const PRE = {
        mimo: { path: ['Q1', 'Q2', 'Q3', 'C', 'Q4', 'Q5', 'F', 'G'], n: { Q1: 'yes: the official one takes ~82.7 GiB per rank [measured: SGLang TP2], leaving ~17 GiB of KV at GMU 0.85', Q2: 'yes: the official MXFP4 (4.25 bits) is smaller than the community NVFP4 (~4.5 bits, ~187 GB)', Q3: 'yes: Marlin MXFP4 (W4A16/W4A8) runs on sm_121, with DeepGEMM turned off (G5)', C: 'use the official one: it is the reference, KLD 0 by definition', Q4: 'yes: 64 Q heads, 4/8 KV heads, 256 experts and 2048/2 = 1024 divide by TP=2', Q5: 'yes: tp_size: 4 in the index; the QKV comes in 4 chunks', F: 'each rank joins 2 chunks: exact path (#59053) and a CPU test of the shards', G: 'TP=2 with --nnodes 2; EP as A/B in Step 9; PP out' } },
        glm: { path: ['Q1', 'Q2', 'B', 'Q3', 'D', 'Q4', 'Q5', 'G'], n: { Q1: 'yes, but not the official one: the official FP8 is 305.8 GiB; the community 4 bpw EXL3, ~164 GiB', Q2: 'no: the smallest one that fits is the community EXL3', B: 'EXL3 TR3 4 bpw: KLD 0.0246 nats, ≈ official FP8 with 54% of the bytes', Q3: 'not in stock vLLM: the exl3 method does not exist in the engine', D: 'exl3.py overlay + exllamav3_ext compiled for sm_121a in the recipe’s Dockerfile', Q4: 'yes at TP=2 (TP=3 requires padding heads 64 → 66)', Q5: 'no', G: 'TP=2; optional TP=3 and TP=4 siblings' } },
        qwen: { path: ['Q1', 'Q2', 'B', 'Q3', 'C', 'Q4', 'Q5', 'G'], n: { Q1: 'yes: NVIDIA’s NVFP4 is 133 GB (it does not fit on a single Spark)', Q2: 'no: the official FP8 also fits (start-fp8.sh lane, ~500k tokens of KV), but NVFP4 is smaller', B: 'nvidia/Qwen3.8-Flash-Next-NVFP4 as the default; the official FP8 becomes an alternative lane', Q3: 'yes: W4A4 experts (FlashInfer CUTLASS SM120), with a per-shape MXFP8 fallback overlay', C: 'kernel exists: move on to topology', Q4: 'yes', Q5: 'no', G: 'TP=2 + EP (--enable-expert-parallel, allgather_reducescatter): 256 experts per GPU' } },
        ds: { path: ['Q1', 'Q2', 'Q3', 'C', 'Q4', 'Q5', 'G'], n: { Q1: 'yes: ~157 GiB of official checkpoint; 79.17 GiB per rank at TP=2', Q2: 'yes: MXFP4 experts (E8M0 every 32) + FP8 128×128 linears, straight from the vendor', Q3: 'yes, in the recipe’s image: flashinfer_b12x in W4A16', C: 'use the official one', Q4: 'yes at TP=2 (TP=3 requires padding heads 64 → 72)', Q5: 'no', G: 'TP=2, PP=1; TP=3 as a capacity lane' } },
        naive: { path: ['Q1', 'A', 'Q3', 'C', 'Q4', 'Q5', 'G'], n: { Q1: 'no: BF16 617.8 GB and FP8 ~310 GB; the community NVFP4/EXL3 files are one day old and have no published KLD', A: 'quantize the experts (calibrated NVFP4, llm-compressor or ModelOpt) and measure KLD against FP8 on ≥ 3 nodes', Q3: 'NVFP4 kernels exist, but the model is not in the registry: Step 4 becomes a plugin', C: 'move on to topology; the engine work comes in Step 4', Q4: 'yes: same backbone as MiMo (4 KV heads, 256 experts, hidden 4096)', Q5: 'unknown: read the index’s tp_size during triage', G: 'TP=2 as a starting point' } },
      };
      const S = { p: 'mimo', t0: 0 };
      const ctl = UI.controls(fig);
      const notes = LA.h('dl', { class: 'la-kv', style: { margin: '0', padding: '12px 18px 6px', borderTop: '1px solid var(--line)' } });
      fig.insertBefore(notes, ctl);
      const updNotes = () => { const P = PRE[S.p]; notes.innerHTML = P.path.map((id) => `<dt>${id}</dt><dd>${P.n[id] || ''}</dd>`).join(''); };
      let H;
      UI.seg(ctl, { label: 'Model', options: [{ v: 'mimo', label: 'MiMo' }, { v: 'glm', label: 'GLM' }, { v: 'qwen', label: 'Qwen' }, { v: 'ds', label: 'DeepSeek' }, { v: 'naive', label: 'Naive-N0.5' }], value: 'mimo', onChange: (v) => { S.p = v; S.t0 = H ? H.t : 0; updNotes(); } });
      updNotes();
      H = V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const pc = { mimo: C.a, glm: C.glm, qwen: C.qwen, ds: C.ds, naive: C.f }[S.p] || C.a;
        const qx = 10, qw = Math.floor(w * 0.56) - 18, lx = qx + qw + 30, lw = w - lx - 10;
        const top = 10, rowH = (h - 20) / 7, bh = Math.min(w < 480 ? 66 : 54, rowH - 12);
        const box = (id) => { const n = N[id], y = top + n.row * rowH + (rowH - bh) / 2; return n.q ? { x: qx, y, w: qw, h: bh } : { x: lx, y, w: lw, h: bh }; };
        const ctr = (id) => { const b = box(id); return [b.x + b.w / 2, b.y + b.h / 2]; };
        const P = PRE[S.p], vis = new Set(P.path);
        const onEdge = (a, b) => { const i = P.path.indexOf(a); return i >= 0 && P.path[i + 1] === b; };
        // arestas estáticas
        EDGES.forEach(([a, b, lab]) => {
          const A = box(a), B = box(b), on = onEdge(a, b) || (a === 'Q3' && b === 'Q4');
          let x1, y1, x2, y2;
          if (N[b].q) { x1 = A.x + A.w / 2; y1 = A.y + A.h; x2 = B.x + B.w / 2; y2 = B.y; } else { x1 = A.x + A.w; y1 = A.y + A.h / 2; x2 = B.x; y2 = B.y + B.h / 2; }
          V.arrow(g, x1, y1, x2, y2, { color: V.alpha(C.muted, on && vis.has(a) && vis.has(b) ? 0.9 : 0.35), lw: 1.3, head: 6 });
          if (lab) V.text(g, lab, (x1 + x2) / 2 + (N[b].q ? 6 : 0), (y1 + y2) / 2 + (N[b].q ? 4 : -4), { size: 11, color: C.muted, align: N[b].q ? 'left' : 'center' });
        });
        // caminho do preset: roteado pelas bordas das caixas (portas), nunca por cima do texto
        const port = (id, side) => { const b = box(id); return side === 'top' ? [b.x + b.w / 2, b.y] : side === 'bottom' ? [b.x + b.w / 2, b.y + b.h] : side === 'left' ? [b.x, b.y + b.h / 2] : [b.x + b.w, b.y + b.h / 2]; };
        const gx2 = lx - 11;
        const pts = [];
        for (let i = 0; i < P.path.length - 1; i++) {
          const a = P.path[i], b = P.path[i + 1], aq = N[a].q, bq = N[b].q;
          if (aq && bq) pts.push(port(a, 'bottom'), port(b, 'top'));
          else if (aq && !bq) pts.push(port(a, 'right'), port(b, 'left'));
          else { const L = port(a, 'left'), R = port(b, 'right'); pts.push(L, [gx2, L[1]], [gx2, R[1]], R); }
        }
        g.strokeStyle = V.alpha(pc, 0.75); g.lineWidth = 3; g.setLineDash([6, 5]); g.lineDashOffset = -t * 16;
        g.beginPath(); pts.forEach((p, i) => (i ? g.lineTo(p[0], p[1]) : g.moveTo(p[0], p[1]))); g.stroke(); g.setLineDash([]);
        // caixas (base opaca + cor)
        Object.keys(N).forEach((id) => {
          const n = N[id], b = box(id), on = vis.has(id);
          const col = n.q && !n.g ? C.b : n.g ? pc : C.a;
          V.box(g, b.x, b.y, b.w, b.h, { r: n.q ? 10 : 8, fill: C.bg || C.panel });
          V.box(g, b.x, b.y, b.w, b.h, { r: n.q ? 10 : 8, fill: on ? V.alpha(n.q && !n.g ? C.b : pc, n.g ? 0.28 : 0.18) : V.alpha(C.muted, 0.06), stroke: on ? (n.q && !n.g ? col : pc) : V.alpha(C.muted, 0.3), lw: on ? 2 : 1 });
          const lab = (n.q && !n.g ? id + ' · ' : n.g ? '' : id + ' · ') + n.t;
          const fs = b.w < 170 ? 11 : 11.5;
          g.save(); V.rr(g, b.x, b.y, b.w, b.h, 8); g.clip();
          wrap(g, lab, b.x + 8, b.y + 16, b.w - 14, 13.5, { size: fs, color: on ? C.ink : C.muted, weight: on && n.q ? 500 : 400 });
          g.restore();
        });
        // ponto viajante
        const segs = []; let L = 0;
        for (let i = 0; i < pts.length - 1; i++) { const d = Math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1]); segs.push(d); L += d; }
        const cyc = L / 120 + 1.2, u = Math.min(1, ((t - S.t0) % cyc) / (cyc - 1.2));
        let dd = u * L, k = 0; while (k < segs.length - 1 && dd > segs[k]) { dd -= segs[k]; k++; }
        const f = segs[k] ? dd / segs[k] : 0;
        const px = lerp(pts[k][0], pts[k + 1][0], f), py = lerp(pts[k][1], pts[k + 1][1], f);
        V.glow(g, px, py, 18, pc, 1); V.dot(g, px, py, 4.5, pc);
      });
    })();

    /* =====================================================================
       Figura 5 — QKV pré-fatiado: 4 chunks no disco → ranks
       ===================================================================== */
    (function figQKV() {
      const fig = root.querySelector('#fig-qkv');
      setH(fig, 790);
      const S = { tp: 2, ga: true, exact: false, t0: -10, geo: null };
      let H = null;
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'TP', options: [{ v: 4, label: 'TP=4' }, { v: 2, label: 'TP=2' }], value: 2, onChange: (v) => { S.tp = v; rebuild(); } });
      UI.seg(ctl, { label: 'Layer', options: [{ v: 'ga', label: 'global' }, { v: 'swa', label: 'SWA' }], value: 'ga', onChange: (v) => { S.ga = v === 'ga'; rebuild(); } });
      UI.seg(ctl, { label: 'Path', options: [{ v: 'req', label: 'requantize (stock)' }, { v: 'exact', label: 'exact (#59053)' }], value: 'req', onChange: (v) => { S.exact = v === 'exact'; rebuild(); } });
      const rConf = UI.readout(ctl, 'Blocks with >1 source scale', '—');
      const rShard = UI.readout(ctl, 'Shard per rank', '—');
      const rPad = UI.readout(ctl, 'Padding', '—');
      function rebuild() {
        const q = 3072, kk = S.ga ? 192 : 384, vv = S.ga ? 128 : 256, cr = q + kk + vv, R = 4 * cr, per = 4 / S.tp;
        // caminho exato nas globais: K+V = 320 linhas → +64 de zero para fechar 3 blocos de 128
        const padN = S.exact && S.tp < 4 ? (128 - ((kk + vv) % 128)) % 128 : 0;
        const pieces = [];
        let rowsR = 0;
        for (let r = 0; r < S.tp; r++) {
          let d = 0;
          const add = (c, part) => {
            const off = part === 'Q' ? 0 : part === 'K' ? q : part === 'V' ? q + kk : 0;
            const len = part === 'Q' ? q : part === 'K' ? kk : part === 'V' ? vv : padN;
            pieces.push({ c, part, src: c * cr + off, srcIn: off, len, rank: r, dst: d });
            d += len;
          };
          const cs = Array.from({ length: per }, (_, j) => r * per + j);
          if (padN) { cs.forEach((c) => add(c, 'Q')); cs.forEach((c) => { add(c, 'K'); add(c, 'V'); add(c, 'P'); }); }
          else ['Q', 'K', 'V'].forEach((part) => cs.forEach((c) => add(c, part)));
          rowsR = d;
        }
        // de que bloco(s) de escala do disco vem cada bloco de 128 linhas do shard
        let conf = 0, tot = 0; const confB = new Set(), src = {};
        for (let r = 0; r < S.tp; r++) {
          const nb = Math.ceil(rowsR / 128);
          for (let b = 0; b < nb; b++) {
            const set = new Set();
            for (let row = b * 128; row < Math.min(rowsR, (b + 1) * 128); row += 64) {
              const p = pieces.find((pp) => pp.rank === r && row >= pp.dst && row < pp.dst + pp.len);
              if (!p || p.part === 'P') continue;
              set.add('c' + p.c + '·b' + Math.floor((p.srcIn + row - p.dst) / 128));
            }
            src[r + ':' + b] = [...set];
            tot++; if (set.size > 1) { conf++; confB.add(r + ':' + b); }
          }
        }
        S.geo = { q, kk, vv, cr, R, per, pieces, conf, tot, confB, src, rowsR, padN };
        S.t0 = H ? H.t : 0;
        rConf.set(`${conf} of ${tot}`);
        rShard.set(`[${nf(rowsR)} × 4096]`);
        rPad.set(padN ? `+${padN} rows/chunk (zeros)` : 'none');
      }
      rebuild();
      const hatch = (g, x, y, ww, hh, col) => {
        g.save(); g.beginPath(); g.rect(x, y, ww, hh); g.clip();
        g.fillStyle = V.alpha(col, 0.1); g.fillRect(x, y, ww, hh);
        g.strokeStyle = V.alpha(col, 0.55); g.lineWidth = 1;
        for (let k = -hh; k < ww; k += 6) { g.beginPath(); g.moveTo(x + k, y + hh); g.lineTo(x + k + hh, y); g.stroke(); }
        g.restore();
      };
      H = V.fig(fig, (g, w, h, t) => {
        const C = V.C(), G = S.geo, narrow = w < 560;
        const colOf = (p) => (p === 'Q' ? C.b : p === 'K' ? C.a : C.c);
        const top = 50, bot = narrow ? 112 : 80, zh = narrow ? 300 : 0;
        const HH = h - top - bot - (narrow ? zh + 12 : 0);
        const sx = 38, sw = narrow ? Math.min(78, w * 0.22) : Math.min(120, w * 0.2), sc = HH / G.R;
        V.text(g, 'disk (tp_size = 4)', sx + sw / 2, 18, { size: 11.5, align: 'center', color: C.ink, weight: 600 });
        V.text(g, `[${nf(G.R)} × 4096]`, sx + sw / 2, 34, { size: 11, font: 'mono', align: 'center', color: C.muted });
        // fonte (fantasma)
        for (let c = 0; c < 4; c++) {
          const y0 = top + c * G.cr * sc;
          [['Q', 0, G.q], ['K', G.q, G.kk], ['V', G.q + G.kk, G.vv]].forEach(([p, o, l]) => { g.fillStyle = V.alpha(colOf(p), 0.22); g.fillRect(sx, y0 + o * sc, sw, Math.max(1.5, l * sc)); });
          for (let m = 1; m * 128 < G.cr; m++) V.line(g, sx, y0 + m * 128 * sc, sx + sw, y0 + m * 128 * sc, { color: V.alpha(C.ink, 0.12), lw: 0.8 });
          if (S.ga) { g.strokeStyle = C.d; g.lineWidth = 1.5; g.strokeRect(sx - 2, y0 + 3200 * sc, sw + 4, Math.min(128, G.cr - 3200) * sc); }
          V.line(g, sx - 5, y0, sx + sw + 5, y0, { color: C.ink, lw: c ? 1.4 : 0.8 });
          V.text(g, 'c' + c, sx - 9, y0 + G.cr * sc / 2 + 4, { size: 11, font: 'mono', align: 'right', color: C.muted });
        }
        // destino
        const dx = narrow ? w * 0.5 : Math.max(w * 0.54, w - 192), dw = Math.min(160, w - dx - 18), gapR = 24;
        const rh = (HH - gapR * (S.tp - 1)) / S.tp, rsc = rh / G.rowsR;
        const qh = 16 * G.per, kvh = (S.ga ? 1 : 2) * G.per;
        const nb = Math.ceil(G.rowsR / 128);
        for (let r = 0; r < S.tp; r++) {
          const ry = top + r * (rh + gapR);
          V.text(g, narrow ? `rank ${r} · ${qh} Q · ${kvh} KV` : `rank ${r} · Q ${qh} heads · KV ${kvh}`, dx, ry - 6, { size: 11, font: 'mono', color: C.ink, maxW: w - dx - 8 });
          g.strokeStyle = V.alpha(C.muted, 0.5); g.lineWidth = 1; g.strokeRect(dx - 1, ry - 1, dw + 2, rh + 2);
          for (let b = 0; b < nb; b++) {
            if (!G.confB.has(r + ':' + b)) continue;
            const by = ry + b * 128 * rsc, bhh = Math.min(128, G.rowsR - b * 128) * rsc;
            g.fillStyle = V.alpha(C.d, 0.55 + 0.25 * Math.sin(t * 4));
            g.fillRect(dx + dw + 4, by, 10, Math.max(3, bhh));
          }
        }
        // peças animadas (o padding não tem origem: surge no lugar)
        const n = G.pieces.length;
        G.pieces.forEach((pc, i) => {
          const yd = top + pc.rank * (rh + gapR) + pc.dst * rsc, hd = pc.len * rsc;
          if (pc.part === 'P') {
            const a = clamp((t - S.t0 - 1.0) / 0.5, 0, 1);
            if (a <= 0) return;
            g.globalAlpha = a; hatch(g, dx, yd, dw, Math.max(2, hd), C.muted); g.globalAlpha = 1;
            return;
          }
          const pp = V.ease.inOut(clamp((t - S.t0 - (i / n) * 0.6) / 1.1, 0, 1));
          const ys = top + pc.src * sc, hs = pc.len * sc;
          const x = lerp(sx, dx, pp), y = lerp(ys, yd, pp), ww = lerp(sw, dw, pp), hh = Math.max(2, lerp(hs, hd, pp));
          g.fillStyle = V.alpha(colOf(pc.part), 0.8); g.fillRect(x, y, ww, hh);
          if (pp > 0.98 && hd > 12) V.text(g, `${pc.part}·c${pc.c}`, x + 6, y + Math.min(hd, 30) / 2 + 4, { size: 11, font: 'mono', color: C.ink });
        });
        // grade de 128 linhas no destino
        for (let r = 0; r < S.tp; r++) {
          const ry = top + r * (rh + gapR);
          for (let b = 1; b < nb; b++) V.line(g, dx + dw - 7, ry + b * 128 * rsc, dx + dw, ry + b * 128 * rsc, { color: V.alpha(C.ink, 0.45), lw: 0.8 });
        }
        // zoom na cauda K/V do rank 0: de que bloco do disco vem a escala de cada bloco
        const r0 = G.q * G.per, rows = G.rowsR - r0;
        let ix, iw, iy, ih;
        if (narrow) { ix = 46; iw = w - ix - 14; iy = top + HH + 12 + 26; ih = zh - 26 - 58; }
        else { ix = sx + sw + 40; iw = dx - ix - 46; iy = top + 22; ih = Math.min(HH - 78, rows * 0.36); }
        const ry0 = top + r0 * rsc;
        if (iw > 120) {
          const al = clamp((t - S.t0 - 1.3) / 0.5, 0, 1), isc = ih / rows;
          g.globalAlpha = al;
          V.text(g, `zoom · rank 0, rows ${nf(r0)}–${nf(r0 + rows - 1)} (K, V${G.padN ? ' and padding' : ''})`, ix + iw / 2, iy - 10, { size: 11, font: 'mono', align: 'center', color: C.muted, maxW: iw + 30 });
          G.pieces.filter((pc) => pc.rank === 0 && pc.part !== 'Q').forEach((pc) => {
            const yy = iy + (pc.dst - r0) * isc, hh = pc.len * isc;
            if (pc.part === 'P') hatch(g, ix, yy + 0.5, iw, hh - 1, C.muted);
            else V.box(g, ix, yy + 0.5, iw, hh - 1, { r: 3, fill: V.alpha(colOf(pc.part), 0.75) });
            // rótulo no maior trecho da peça entre fronteiras de bloco (nunca em cima de uma linha tracejada)
            let s0 = pc.dst, best = [pc.dst, pc.dst];
            for (let rr = Math.floor(pc.dst / 128 + 1) * 128; ; rr += 128) { const e1 = Math.min(rr, pc.dst + pc.len); if (e1 - s0 > best[1] - best[0]) best = [s0, e1]; if (e1 >= pc.dst + pc.len) break; s0 = e1; }
            const bh2 = (best[1] - best[0]) * isc, by2 = iy + (best[0] - r0) * isc;
            if (bh2 >= 11) V.text(g, pc.part === 'P' ? `pad·c${pc.c} (${pc.len})` : `${pc.part}·c${pc.c} (${pc.len})`, ix + 8, by2 + Math.min(bh2 / 2, 10) + 4, { size: 11, font: 'mono', color: C.ink });
          });
          for (let b = Math.ceil(r0 / 128); b * 128 <= r0 + rows; b++) {
            const yy = iy + (b * 128 - r0) * isc;
            V.line(g, ix - 6, yy, ix + iw + 6, yy, { color: V.alpha(C.ink, 0.6), lw: 1, dash: [3, 3] });
          }
          for (let b = Math.floor(r0 / 128); b * 128 < r0 + rows; b++) {
            const y0 = iy + (Math.max(b * 128, r0) - r0) * isc, y1 = iy + (Math.min((b + 1) * 128, r0 + rows) - r0) * isc;
            const bad = G.confB.has('0:' + b), from = G.src['0:' + b] || [];
            V.text(g, 'b' + b, ix - 10, (y0 + y1) / 2 + 4, { size: 11, font: 'mono', align: 'right', color: bad ? C.d : C.muted, weight: bad ? 700 : 400 });
            if (from.length) V.text(g, (iw > 260 ? 'scale ← ' : '← ') + from.join('+'), ix + iw - 6, (y0 + y1) / 2 + 4, { size: 11, font: 'mono', align: 'right', color: C.ink, weight: bad ? 700 : 400 });
            if (bad) { g.strokeStyle = C.d; g.lineWidth = 2.5; V.rr(g, ix - 3, y0 + 1, iw + 6, y1 - y0 - 2, 4); g.stroke(); }
            else if (G.padN) { g.strokeStyle = V.alpha(C.e, 0.9); g.lineWidth = 1.5; V.rr(g, ix - 2, y0 + 1.5, iw + 4, y1 - y0 - 3, 4); g.stroke(); }
          }
          g.strokeStyle = V.alpha(C.muted, 0.5); g.lineWidth = 1; g.setLineDash([3, 3]);
          if (narrow) {
            g.strokeRect(dx - 3, ry0, dw + 6, top + rh - ry0);
            V.text(g, 'zoom ↓', dx - 8, ry0 + (top + rh - ry0) / 2 + 4, { size: 11, align: 'right', color: C.muted });
          } else {
            g.beginPath(); g.moveTo(ix + iw + 6, iy); g.lineTo(dx - 2, ry0); g.moveTo(ix + iw + 6, iy + ih); g.lineTo(dx - 2, top + rh); g.stroke();
          }
          g.setLineDash([]);
          const note = G.conf ? 'amber: the block joins rows from different scale blocks; no scale on disk works, and the stock loader requantizes'
            : G.padN ? 'green: each block reuses a single scale block from the checkpoint; the hatched band is zero padding'
            : 'every 128-row boundary matches the source’s: a block permutation';
          wrap(g, note, ix - (narrow ? 30 : 0), iy + ih + 16, iw + (narrow ? 30 : 0), 14, { size: 11, color: G.conf ? C.d : C.e });
          g.globalAlpha = 1;
        } else {
          const ay = top + HH / 2;
          V.arrow(g, sx + sw + 12, ay, dx - 12, ay, { color: V.alpha(C.muted, 0.7), lw: 1.4, dash: [4, 4] });
        }
        // status
        let msg, col;
        if (S.tp === 4) { msg = '✓ exact fit: each rank gets a whole chunk, with the original scales'; col = C.e; }
        else if (!S.ga) { msg = S.exact ? '✓ SWA: pure block permutation (the same scales, reordered). The catch here is different: 2 KV heads per chunk, and the sharding needs ckpt_tp = 4 chunks (#57508)' : 'SWA: aligned blocks (29 per chunk), and requantizing does not change the weights. The problem is a different one: 2 KV heads per chunk, and the sharding needs ckpt_tp = 4 chunks (#57508)'; col = S.exact ? C.e : C.ink; }
        else if (S.exact) { msg = '✓ exact (VLLM_MIMO_EXACT_QKV=1): each K | V | pad starts on a 128 boundary and reuses the chunk’s blocks 24–26; the forward splits K and V. Cost: ~12K fewer tokens in the pool [measured: diffbot]'; col = C.e; }
        else { msg = '⚠ requantize: the loader dequantizes and requantizes the rank’s rows; 18% of Q, ~51% of K and 73–81% of V change (L2 error of 1.3–2.5%) [measured: diffbot]'; col = C.d; }
        wrap(g, msg, 16, h - bot + 20, w - 32, 15, { size: 12, color: col, weight: 600 });
        const ly = h - 10, sp = narrow ? 40 : 56;
        [[C.b, 'Q'], [C.a, 'K'], [C.c, 'V']].forEach(([c, l], i) => { const lx = 16 + i * sp; V.box(g, lx, ly - 10, 11, 11, { r: 3, fill: c }); V.text(g, l, lx + 16, ly, { size: 11, color: C.muted }); });
        hatch(g, 16 + 3 * sp, ly - 10, 11, 11, C.muted); V.text(g, 'pad', 16 + 3 * sp + 16, ly, { size: 11, color: C.muted });
        const lx4 = 16 + 4 * sp + (narrow ? 4 : 10);
        V.box(g, lx4, ly - 10, 11, 11, { r: 3, fill: C.d }); V.text(g, narrow ? '>1 source scale' : 'block with >1 source scale', lx4 + 16, ly, { size: 11, color: C.muted, maxW: w - lx4 - 24 });
      });
    })();

    /* =====================================================================
       Figura 6 — overlay fail-closed + vazamento de sliding_window
       ===================================================================== */
    (function figOverlay() {
      const fig = root.querySelector('#fig-overlay');
      setH(fig, 700);
      const STEPS = [
        { s: 'Extract from image', m: 'docker run … cat → orig/ + sha256', t: 'Extract from the pinned image', d: '<code>docker run --entrypoint cat $IMAGE …/mimo_v2.py &gt; files/orig/</code> and <code>sha256sum</code>: the patch starts from the exact bytes of the pinned image, never from a file on <code>main</code>.' },
        { s: 'Already applied?', m: 'MARK in s → copy and exit', t: 'Idempotence', d: 'If the <code>MARK</code> is already in the file, the script copies it and exits saying it is already applied: running it twice is safe.' },
        { s: 'Already upstream?', m: 'regex layer_cache_config', t: 'Automatic retirement', d: 'If the code already passes <code>cache_config=layer_cache_config</code>, the new image brought the fix: “already upstream”, and the patch does nothing.' },
        { s: 'Anchor 1 (×1)', m: 's.count(a1) == 1', t: 'Anchor 1: exactly once', d: 'Looks for the start of the decoder layer’s <code>__init__</code> and requires <b>one</b> occurrence. Inserts the copy of <code>cache_config</code> with <code>sliding_window=None</code>. If it does not match, it aborts with a clear message.' },
        { s: 'Anchor 2 (×2)', m: 's.count(…) == 2', t: 'Anchor 2: exactly twice', d: 'Requires <b>both</b> <code>MiMoV2Attention</code> constructions and injects <code>cache_config=layer_cache_config</code> into both. A different count = a different image = abort.' },
        { s: 'Write + marker', m: 'print("[overlay] …")', t: 'Write and mark', d: 'Writes <code>files/overlay/mimo_v2.py</code> and prints <code>[overlay] cache_config forwarded</code>: that is the text you look for in <code>docker logs</code>.' },
        { s: 'Mount on 2 ranks', m: '-v overlay:…/mimo_v2.py:ro', t: 'Bind-mount on both ranks', d: '<code>-v …/overlay/mimo_v2.py:…/models/mimo_v2.py:ro</code> on the worker and on the head, with the same bytes. No image rebuild.' },
        { s: 'Prove', m: 'KV 2× · needle 128K', t: 'Prove', d: '<code>GPU KV cache size</code> ~2× with fp8 (together with overlay 2, the port of #58128) and the 128K needle at 100%. Without proof, the patch does not go in.' },
      ];
      const STATES = [
        { label: 'upstream', leak: false, boot: true, kv: 'BF16: fp8 is ignored without a warning', pool: '1,365,757 tokens (1×)', needle: 'ok', verdict: 'silent: you believe you have FP8', col: 'd' },
        { label: 'naive', leak: true, boot: true, kv: 'FP8 (with the ported backend)', pool: '2,731,530 tokens (2×)', needle: 'fail', verdict: 'silent quality bug: the global layers see only the 128 window', col: 'red' },
        { label: 'overlay 1', leak: false, boot: false, kv: '— (the boot never gets there)', pool: '—', needle: 'na', verdict: 'startup error: TRITON_ATTN_DIFFKV rejects FP8. It fails loudly: the right behavior', col: 'd' },
        { label: 'overlays 1+2', leak: false, boot: true, kv: 'FP8', pool: '2,731,530 tokens (2×)', needle: 'ok', verdict: 'good state: effective FP8 and the global layers stay global', col: 'a' },
      ];
      const S = { st: 3, changed: false };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'State', options: STATES.map((s, i) => ({ v: i, label: s.label })), value: 3, onChange: (v) => { S.st = v; } });
      UI.toggle(ctl, { label: 'image changed (anchor does not match)', value: false, onChange: (v) => { S.changed = v; } });
      const stp = UI.stepper(fig, STEPS.map((s) => ({ t: s.t, d: s.d })), { autoplay: 5200 });
      V.fig(fig, (g, w, h, t) => {
        const C = V.C(), wide = w >= 640;
        const abortAt = S.changed ? 3 : 99;
        // ---------- pipeline ----------
        const px = 14, py = 14, pw = wide ? w * 0.44 : w - 28, ph = wide ? h - 64 : 250;
        const cols = wide ? 1 : 2, rows = 8 / cols, gap = 7;
        const bw = (pw - gap * (cols - 1)) / cols, bh = Math.min(46, (ph - gap * (rows - 1)) / rows);
        const bpos = (k) => { const c = wide ? 0 : Math.floor(k / 4), r = wide ? k : k % 4; return [px + c * (bw + gap), py + r * (bh + gap)]; };
        STEPS.forEach((s, k) => {
          const [bx, by] = bpos(k);
          let status = k < stp.i ? 'done' : k === stp.i ? 'on' : 'todo';
          if (stp.i >= abortAt && k === abortAt) status = 'abort';
          else if (stp.i >= abortAt && k > abortAt) status = 'skip';
          const col = status === 'abort' ? C.red : C.a;
          V.box(g, bx, by, bw, bh, { r: 8, fill: status === 'on' ? V.alpha(col, 0.3) : status === 'done' ? V.alpha(col, 0.13) : status === 'abort' ? V.alpha(C.red, 0.28) : V.alpha(C.muted, 0.06), stroke: status === 'on' || status === 'abort' ? col : V.alpha(C.muted, 0.3), lw: status === 'on' || status === 'abort' ? 2 : 1 });
          V.text(g, `${k + 1}. ${s.s}`, bx + 10, by + (bh > 36 ? 18 : bh / 2 + 4), { size: 12, weight: 600, color: status === 'skip' ? V.alpha(C.muted, 0.6) : C.ink, maxW: bw - 34 });
          if (bh > 36) V.text(g, status === 'skip' ? '— not run' : s.m, bx + 10, by + 34, { size: 11, font: 'mono', color: C.muted, maxW: bw - 16 });
          if (status === 'done') V.text(g, '✓', bx + bw - 12, by + 18, { size: 13, align: 'center', color: C.a, weight: 700 });
          if (status === 'abort') V.text(g, '✕', bx + bw - 12, by + 18, { size: 13, align: 'center', color: C.red, weight: 700 });
        });
        // arquivo viajando
        const e = clamp(stp.elapsed() / 0.7, 0, 1), k1 = stp.i, k0 = Math.max(0, k1 - 1);
        const [ax, ay] = bpos(k0), [bx2, by2] = bpos(k1);
        const fx = lerp(ax, bx2, V.ease.out(e)) + bw - 26, fy = lerp(ay, by2, V.ease.out(e)) + bh / 2 - 9;
        const aborted = stp.i >= abortAt;
        V.box(g, fx - 8, fy, 16, 20, { r: 3, fill: aborted ? C.red : C.d });
        V.text(g, '.py', fx, fy + 14, { size: 11, font: 'mono', align: 'center', color: onInk(C), weight: 700 });
        if (aborted && wide) wrap(g, 'sys.exit: “anchor 1 not found exactly 1x — the image changed; review the patch”. Nothing is written or mounted.', px, py + 8 * (bh + gap) + 8, pw, 15, { size: 11.5, color: C.red, weight: 600 });
        // ---------- máscara ----------
        const st = STATES[S.st];
        const mx0 = wide ? w * 0.5 : 14, my0 = wide ? 30 : py + ph + (aborted ? 40 : 18), mw = wide ? w * 0.5 - 16 : w - 28;
        if (aborted && !wide) wrap(g, 'sys.exit: anchor 1 does not match. Nothing is written or mounted.', 14, py + ph + 14, w - 28, 15, { size: 11.5, color: C.red, weight: 600 });
        const NN = 20, WIN = 5;
        const cell = Math.min(15, (mw - 34) / NN, (h - my0 - 120) / NN);
        const gx = mx0 + 26, gy = my0 + 14;
        V.text(g, 'global layer · 20 tokens · window drawn as 5', mx0, my0, { size: 11.5, color: C.muted, maxW: mw });
        const dim = !st.boot;
        for (let i = 0; i < NN; i++) for (let j = 0; j < NN; j++) {
          const visible = j <= i && (!st.leak || i - j < WIN);
          let fill = visible ? V.alpha(C.a, 0.25 + 0.35 * (j === i ? 1 : 0.5)) : V.alpha(C.muted, 0.07);
          if (j === 1 && visible) fill = V.alpha(C.d, 0.6);
          g.fillStyle = dim ? V.alpha(C.muted, visible ? 0.16 : 0.05) : fill;
          g.fillRect(gx + j * cell + 0.5, gy + i * cell + 0.5, cell - 1, cell - 1);
        }
        V.text(g, 'q', gx - 8, gy + 4 + cell * 0.6, { size: 11, font: 'mono', align: 'right', color: C.muted });
        V.text(g, 'k →', gx + NN * cell, gy - 4, { size: 11, font: 'mono', align: 'right', color: C.muted });
        V.text(g, '✦', gx + 1 * cell + cell / 2, gy - 3, { size: 11, align: 'center', color: C.d });
        // linha da pergunta
        g.strokeStyle = C.ink; g.lineWidth = 1.5; g.strokeRect(gx, gy + 19 * cell, NN * cell, cell);
        const qc = [gx + 1 * cell + cell / 2, gy + 19 * cell + cell / 2];
        if (!dim) {
          if (!st.leak) { V.glow(g, qc[0], qc[1], 14 + 3 * Math.sin(t * 4), C.d, 1); V.text(g, '✦', qc[0], qc[1] + 4, { size: 11, align: 'center', color: C.ink, weight: 700 }); }
          else V.text(g, '✕', qc[0], qc[1] + 4.5, { size: 13, align: 'center', color: C.red, weight: 700 });
        } else {
          g.save(); g.translate(gx + NN * cell / 2, gy + NN * cell / 2); g.rotate(-0.18);
          V.box(g, -86, -16, 172, 32, { r: 8, fill: V.alpha(C.red, 0.2), stroke: C.red, lw: 2 });
          V.text(g, '✕ startup error', 0, 5, { size: 14, align: 'center', color: C.red, weight: 700 });
          g.restore();
        }
        // leitura do servidor
        let yy = gy + NN * cell + 22;
        const lw2 = mw;
        const colV = C[st.col] || C.ink;
        yy = wrap(g, 'effective KV: ' + st.kv, mx0, yy, lw2, 15, { size: 12, color: C.ink });
        yy = wrap(g, 'pool of 17 GiB [measured, #58128]: ' + st.pool, mx0, yy, lw2, 15, { size: 12, color: C.ink });
        yy = wrap(g, 'needle 128K: ' + (st.needle === 'ok' ? '✓ finds the needle' : st.needle === 'fail' ? '✕ fails above ~128 tokens' : '— (the server does not start)'), mx0, yy, lw2, 15, { size: 12, color: st.needle === 'fail' ? C.red : C.ink });
        wrap(g, '→ ' + st.verdict, mx0, yy + 2, lw2, 15, { size: 12, color: colV, weight: 600 });
      });
    })();

    /* =====================================================================
       Figura 7 — boot em dois nós: rendezvous, grupos de comunicação, flags
       ===================================================================== */
    (function figBoot() {
      const fig = root.querySelector('#fig-boot');
      setH(fig, 560);
      const PH = [
        { t: 'Worker first', flags: ['--node-rank 1', '--headless', '--nnodes 2', '--master-addr $HEAD_IP', '--master-port 50000', 'VLLM_HOST_IP=$WORKER_IP'], d: 'The worker comes up with <code>--node-rank 1 --headless</code>: a single process for its GPU, with no API or EngineCore. It looks for the head’s <code>--master-addr</code>:<code>--master-port</code> and waits. It is the order in all three recipes: worker, then head.' },
        { t: 'Head and rendezvous', flags: ['--node-rank 0', '--distributed-executor-backend mp', '--tensor-parallel-size 2', '--host 0.0.0.0 --port 8888', 'VLLM_HOST_IP=$HEAD_IP'], d: 'The head comes up with <code>--node-rank 0</code>: API server, EngineCore (scheduler and KV manager) and the GPU 0 process. The two GPU processes meet at the agreed address. That is the <i>rendezvous</i>, and now the group has <i>world size</i> 2.' },
        { t: 'Communication groups', flags: ['GLOO_SOCKET_IFNAME', 'NCCL_SOCKET_IFNAME', 'NCCL_IB_HCA==rocep1s0f0', 'NCCL_IB_GID_INDEX=3', '--network host', '--device /dev/infiniband', '--ulimit memlock=-1'], d: 'Two groups form: <b>Gloo</b>, on the CPU, for metadata and barriers, and <b>NCCL</b>, for the tensors. The bootstrap goes over TCP on the CX7 interface; NCCL’s data goes over RDMA on the chosen HCA. Check the log: <code>NET/IB</code>, not <code>NET/Socket</code>.' },
        { t: 'Weights and overlays', flags: ['-v $MODEL_DIR:ro', '--load-format safetensors', '--safetensors-load-strategy lazy', '-v overlay/*.py:ro', 'HF_HUB_OFFLINE=1', 'VLLM_ENGINE_READY_TIMEOUT_S=3600'], d: 'Each rank reads its slice of the weights from the same read-only directory, with the same overlays mounted; the <code>[overlay] …</code> markers show up in both logs. The cold boot takes ~11 min <b>[measured: tonyd2wild]</b>, hence the 3600 s timeout.' },
        { t: 'Memory and graphs', flags: ['--gpu-memory-utilization 0.85', '--kv-cache-dtype fp8', '--compilation-config FULL_DECODE_ONLY', '--cudagraph-capture-sizes 8 … 64'], d: 'Each rank measures memory, reserves the KV pool and captures the CUDA graphs for the capture sizes. The log shows <code>GPU KV cache size</code> and the capture sizes: that is Step 5’s exit criterion.' },
        { t: 'Serving', flags: ['--ipc host --shm-size 16g', '--max-num-seqs 8', '--speculative-config', '--no-async-scheduling'], d: 'At every step, the EngineCore schedules and sends the <code>SchedulerOutput</code> to the workers (shared memory on the head, ZMQ to the remote worker). Both run the forward with 96 all-reduces over RDMA, and the tokens come back through the API on port 8888.' },
      ];
      const S = { bad: false };
      const ctl = UI.controls(fig);
      UI.toggle(ctl, { label: 'wrong HCA or GID (NCCL falls back to NET/Socket)', value: false, onChange: (v) => { S.bad = v; } });
      const rT = UI.readout(ctl, 'NCCL transport', 'NET/IB');
      const st = UI.stepper(fig, PH.map((p) => ({ t: p.t, d: p.d })), { autoplay: 6500 });
      const chip = (g, x, y, s, col, on) => {
        g.font = V.font(11, 'mono', 500);
        const ww = g.measureText(s).width + 14;
        V.box(g, x, y, ww, 20, { r: 6, fill: V.alpha(col, on ? 0.2 : 0.06), stroke: V.alpha(col, on ? 0.9 : 0.35), lw: 1 });
        V.text(g, s, x + 7, y + 14, { size: 11, font: 'mono', color: C_.ink, weight: 500 });
        return ww;
      };
      let C_ = V.C();
      V.fig(fig, (g, w, h, t) => {
        const C = (C_ = V.C()), ph = st.i, e = st.elapsed(), wide = w >= 640;
        rT.set(S.bad && ph >= 2 ? 'NET/Socket ✕' : ph >= 2 ? 'NET/IB ✓' : '—');
        // ---------- geometria dos nós ----------
        const nodeH = wide ? 212 : 196, rowH = 40;
        let A, B;
        if (wide) { const bw = Math.min(330, w * 0.36); A = { x: 16, y: 34, w: bw, h: nodeH }; B = { x: w - 16 - bw, y: 34, w: bw, h: nodeH }; }
        else { A = { x: 16, y: 30, w: w - 32, h: nodeH }; B = { x: 16, y: 30 + nodeH + 64, w: w - 32, h: 112 }; }
        const rowY = (N, i) => N.y + 30 + i * rowH;
        const cx = (N) => N.x + 12, cw = (N) => N.w - 24;
        const up0 = ph >= 1, up1 = true;
        const nodeBox = (N, title, up, col) => {
          V.box(g, N.x, N.y, N.w, N.h, { r: 12, fill: V.alpha(col, up ? 0.07 : 0.02), stroke: V.alpha(col, up ? 0.8 : 0.3), lw: up ? 1.6 : 1 });
          if (!up) { g.save(); g.setLineDash([4, 4]); V.box(g, N.x, N.y, N.w, N.h, { r: 12, stroke: V.alpha(C.muted, 0.5), lw: 1 }); g.restore(); }
          V.text(g, title, N.x + 12, N.y + 18, { size: 11.5, font: 'mono', weight: 600, color: up ? C.ink : C.muted, maxW: N.w - 24 });
        };
        nodeBox(A, 'head · rank 0 · 10.0.0.1', up0, C.b);
        nodeBox(B, 'worker · rank 1 · 10.0.0.2', up1, C.b);
        const proc = (N, i, label, col, on, appear = 1) => {
          if (appear <= 0) return null;
          const x = cx(N), y = rowY(N, i), ww = cw(N);
          g.globalAlpha = appear;
          V.box(g, x, y, ww, 30, { r: 8, fill: V.alpha(col, on ? 0.3 : 0.12), stroke: V.alpha(col, on ? 1 : 0.5), lw: on ? 1.8 : 1 });
          V.text(g, label, x + 10, y + 19.5, { size: 11.5, weight: 600, color: C.ink, maxW: ww - 20 });
          g.globalAlpha = 1;
          return { x, y, w: ww, h: 30 };
        };
        const ap = (k) => (ph > 1 ? 1 : ph === 1 ? clamp((e - k * 0.35) / 0.4, 0, 1) : 0);
        const api = proc(A, 0, 'API server · :8888', C.c, ph === 5, ap(0)) || { x: cx(A), y: rowY(A, 0), w: cw(A), h: 30 };
        const core = proc(A, 1, 'EngineCore · scheduler · KV', C.b, ph === 5 || ph === 4, ap(1)) || { x: cx(A), y: rowY(A, 1), w: cw(A), h: 30 };
        const w0 = proc(A, 2, 'GPU 0 process', C.a, ph >= 3, ap(2)) || { x: cx(A), y: rowY(A, 2), w: cw(A), h: 30 };
        let w1;
        if (wide) {
          [0, 1].forEach((i) => { const x = cx(B), y = rowY(B, i); g.save(); g.setLineDash([3, 4]); V.box(g, x, y, cw(B), 30, { r: 8, stroke: V.alpha(C.muted, 0.45), lw: 1 }); g.restore(); V.text(g, i ? 'no EngineCore' : 'no API (--headless)', x + 10, y + 19.5, { size: 11, color: C.muted }); });
          w1 = proc(B, 2, 'GPU 1 process', C.a, ph === 0 || ph >= 3);
        } else {
          w1 = proc(B, 0, 'GPU 1 process (--headless: no API)', C.a, ph === 0 || ph >= 3);
        }
        // espera do worker na fase 0
        if (ph === 0) {
          const dots = '.'.repeat(1 + Math.floor(t * 2.5) % 3);
          V.text(g, 'waiting for 10.0.0.1:50000' + dots, w1.x + 10, w1.y + 48, { size: 11.5, font: 'mono', color: C.d, weight: 600, maxW: w1.w - 12 });
          V.text(g, 'head is not up yet', A.x + A.w / 2, A.y + A.h / 2, { size: 12, align: 'center', color: C.muted });
        }
        // barras de memória (fases 3+)
        const barY = (N) => N.y + N.h - 30;
        const memBar = (N, prog) => {
          const x = cx(N), y = barY(N), ww = cw(N), vis = 121.69;
          V.box(g, x, y, ww, 16, { r: 4, fill: V.alpha(C.muted, 0.08), stroke: V.alpha(C.muted, 0.35), lw: 1 });
          const pw = ww * 82.7 / vis * prog;
          V.box(g, x, y, Math.max(1, pw), 16, { r: 4, fill: V.alpha(C.b, 0.7) });
          if (ph >= 4) {
            const k = clamp(ph === 4 ? e / 1.2 : 1, 0, 1);
            V.box(g, x + ww * 82.7 / vis, y, ww * 4 / vis * k, 16, { r: 2, fill: V.alpha(C.d, 0.75) });
            V.box(g, x + ww * 86.7 / vis, y, ww * 16.7 / vis * k, 16, { r: 2, fill: V.alpha(C.a, 0.75) });
          }
          const lab = ph >= 4 ? 'weights 82.7 · overhead · KV ~16.7 GiB [estimate]' : `rank weights: ${nf(82.7 * prog, 1)} of 82.7 GiB`;
          V.text(g, lab, x, y - 5, { size: 11, color: C.muted, maxW: ww });
        };
        if (ph >= 3) {
          const prog = ph === 3 ? clamp(e / 4, 0, 1) : 1;
          memBar(A, prog); memBar(B, prog);
          if (prog >= 1 || ph > 3) {
            [A, B].forEach((N) => { V.text(g, '[overlay] ✓', N.x + N.w - 12, N.y + 18, { size: 11, font: 'mono', align: 'right', color: C.e, weight: 600 }); });
          }
        }
        // ---------- ligações ----------
        let tcp, rdma;
        if (wide) {
          tcp = [A.x + A.w, core.y + 15, B.x, core.y + 15];
          rdma = [A.x + A.w, w0.y + 15, B.x, w0.y + 15];
          V.line(g, w0.x + w0.w, w0.y + 15, A.x + A.w, w0.y + 15, { color: V.alpha(C.a, 0.5), lw: 2 });
          V.line(g, B.x, w1.y + 15, w1.x, w1.y + 15, { color: V.alpha(C.a, 0.5), lw: 2 });
          V.line(g, core.x + core.w, core.y + 15, A.x + A.w, core.y + 15, { color: V.alpha(C.muted, 0.5), lw: 1 });
        } else {
          const xt = A.x + A.w * 0.28, xr = A.x + A.w * 0.72;
          tcp = [xt, A.y + A.h, xt, B.y];
          rdma = [xr, A.y + A.h, xr, B.y];
        }
        const linkUp = ph >= 1;
        const rdmaOk = ph >= 2 && !S.bad;
        V.line(g, tcp[0], tcp[1], tcp[2], tcp[3], { color: V.alpha(C.ink, linkUp ? 0.55 : 0.2), lw: 1.4, dash: [5, 4], dashOffset: -t * 20 });
        V.line(g, rdma[0], rdma[1], rdma[2], rdma[3], { color: ph < 2 ? V.alpha(C.muted, 0.2) : S.bad ? V.alpha(C.red, 0.7) : V.alpha(C.a, 0.85), lw: ph < 2 ? 2 : 6, dash: S.bad && ph >= 2 ? [8, 6] : null });
        const midT = [(tcp[0] + tcp[2]) / 2, (tcp[1] + tcp[3]) / 2], midR = [(rdma[0] + rdma[2]) / 2, (rdma[1] + rdma[3]) / 2];
        const lab = (p, s, col, dy) => { if (wide) V.text(g, s, p[0], p[1] + dy, { size: 11, align: 'center', color: col, maxW: B.x - A.x - A.w - 8 }); else V.text(g, s, p[0] + 10, p[1] + 4 + dy, { size: 11, color: col, maxW: w - p[0] - 20 }); };
        lab(midT, wide ? 'TCP · IFACE (rendezvous, Gloo, ZMQ queue)' : 'TCP · control', C.muted, wide ? -8 : -8);
        lab(midR, ph < 2 ? (wide ? 'RDMA (no NCCL group yet)' : 'RDMA (no NCCL)') : S.bad ? (wide ? '✕ NET/Socket: all-reduce over TCP' : '✕ NET/Socket') : (wide ? 'RoCE 200 Gb/s · NCCL NET/IB' : 'RoCE · NET/IB'), ph >= 2 && S.bad ? C.red : ph >= 2 ? C.a : C.muted, wide ? 18 : 10);
        // pacotes
        const pk = (L, u, col, r = 4.5) => { const [x, y] = V.along(L[0], L[1], L[2], L[3], u); V.glow(g, x, y, 12, col, 0.8); V.dot(g, x, y, r, col); };
        if (ph === 1) { const u = clamp((e - 1.2) / 1.0, 0, 1); if (u > 0 && u < 1) pk([tcp[2], tcp[3], tcp[0], tcp[1]], u, C.d); if (u >= 1) lab(midT, wide ? 'rendezvous ✓ · world size 2' : 'rendezvous ✓', C.d, wide ? 10 : 8); }
        if (ph === 2) { const u = (e * 0.9) % 1; pk(tcp, u, C.b); pk([tcp[2], tcp[3], tcp[0], tcp[1]], u, C.b); if (!S.bad && e > 1) pk(rdma, (e * 1.4) % 1, C.a, 5); }
        if (ph === 5) {
          const u = (t % 3.2) / 3.2;
          const seg = (a, b) => clamp((u - a) / (b - a), 0, 1);
          const cxp = (R) => [R.x + R.w / 2, R.y + 15];
          const [ax, ay] = cxp(api), [kx, ky] = cxp(core), [wx, wy] = cxp(w0);
          if (u < 0.15) { const v = seg(0, 0.15); V.dot(g, lerp(ax, kx, v), lerp(ay, ky, v), 5, C.c); }
          else if (u < 0.35) { const v = seg(0.15, 0.35); V.dot(g, lerp(kx, wx, v), lerp(ky, wy, v), 4.5, C.b); pk(tcp, v, C.b); }
          else if (u < 0.8) { const v = seg(0.35, 0.8), L = S.bad ? tcp : rdma, reps = S.bad ? 1 : 3, f = (v * reps) % 1; pk(f < 0.5 ? L : [L[2], L[3], L[0], L[1]], (f % 0.5) * 2, S.bad ? C.red : C.a, 5); }
          else { const v = seg(0.8, 1); V.dot(g, lerp(wx, ax, v), lerp(wy, ay, v), 5, C.c); if (v > 0.9) V.text(g, 'tokens →', api.x + api.w - 10, api.y + 19.5, { size: 11, align: 'right', color: C.c, weight: 700 }); }
        }
        // ---------- chips de flags da fase ----------
        const fy0 = (wide ? A.y + A.h : B.y + B.h) + 22;
        V.text(g, `phase ${ph + 1} of ${PH.length} · ${PH[ph].t}: flags and variables in play`, 16, fy0, { size: 11.5, weight: 600, color: C.ink, maxW: w - 32 });
        let x = 16, y = fy0 + 10;
        PH[ph].flags.forEach((f, i) => {
          g.font = V.font(11, 'mono', 500);
          const ww = g.measureText(f).width + 14;
          if (x + ww > w - 16) { x = 16; y += 26; }
          const on = clamp((e - i * 0.12) / 0.3, 0, 1);
          g.globalAlpha = 0.25 + 0.75 * on;
          chip(g, x, y, f, [C.b, C.b, C.a, C.a, C.c, C.c][ph], true);
          g.globalAlpha = 1;
          x += ww + 8;
        });
      });
    })();

    /* =====================================================================
       Figura 8 — simulador de especulação
       ===================================================================== */
    (function figSpec() {
      const fig = root.querySelector('#fig-spec');
      setH(fig, 560);
      const CATS = [
        { n: 'structured', a: 6.85 }, { n: 'code', a: 5.23, tps: 70.5 }, { n: 'math', a: 5.0 },
        { n: 'JSON', a: 4.0 }, { n: 'reasoning', a: 2.62 }, { n: 'prose', a: 1.23, tps: 25.9 },
      ];
      const alphaFor = (a) => { let lo = 0, hi = 0.999999; for (let it = 0; it < 60; it++) { const m = (lo + hi) / 2; let s = 0, p = 1; for (let i = 1; i <= 7; i++) { p *= m; s += p; } if (s < a) lo = m; else hi = m; } return (lo + hi) / 2; };
      CATS.forEach((c) => { c.al = alphaFor(c.a); });
      const T7 = 0.087, BYTES = 6.39, BW = 273;
      const S = { cat: 1, k: 7, eff: 0.58, cd: 0, cap: 'tut', round: -1, acc: 0, tok: 0, steps: 0 };
      const rng = V.rng(2026);
      // t(k) = t0 + c_d·[k>0] + β·k, calibrado para passar por t(7) = T7 (hipótese: c_d não foi medido)
      const model = () => {
        const t0 = BYTES / (BW * S.eff), cd = Math.min(S.cd / 1000, Math.max(0, T7 - t0)), beta = (T7 - t0 - cd) / 7, al = CATS[S.cat].al, arr = [];
        for (let k = 0; k <= 7; k++) { let a = 0, p = 1; for (let i = 1; i <= k; i++) { p *= al; a += p; } const tt = t0 + (k ? cd : 0) + beta * k; arr.push({ k, a, tt, tps: (1 + a) / tt }); }
        let best = 0; arr.forEach((x, i) => { if (x.tps > arr[best].tps) best = i; });
        return { t0, cd, beta, al, arr, best, clamped: cd < S.cd / 1000 - 1e-9 };
      };
      const ctl = UI.controls(fig);
      const reset = () => { S.tok = 0; S.steps = 0; S.round = -1; upd(); };
      UI.select(ctl, { label: 'Category', options: CATS.map((c, i) => ({ v: String(i), label: `${c.n} (${nf(c.a, 2)} of 7)` })), value: '1', onChange: (v) => { S.cat = +v; reset(); } });
      UI.slider(ctl, { label: 'k', min: 0, max: 7, step: 1, value: 7, fmt: (v) => (v ? String(v) : '0 (no spec)'), onInput: (v) => { S.k = v; reset(); } });
      UI.slider(ctl, { label: 'No-spec efficiency', min: 0.45, max: 0.75, step: 0.01, value: 0.58, fmt: (v) => Math.round(v * 100) + '%', onInput: (v) => { S.eff = v; upd(); } });
      UI.slider(ctl, { label: 'Drafter c_d', min: 0, max: 40, step: 1, value: 0, fmt: (v) => v + ' ms', onInput: (v) => { S.cd = v; upd(); } });
      UI.seg(ctl, { label: 'Capture sizes', options: [{ v: 'tut', label: '8 16 … 64' }, { v: 'mult', label: 'multiples of k+1' }], value: 'tut', onChange: (v) => { S.cap = v; } });
      const rA = UI.readout(ctl, 'fitted α', '—'), rT = UI.readout(ctl, 'Tokens/step', '—'), rP = UI.readout(ctl, 'Predicted', '—'), rB = UI.readout(ctl, 'Best k', '—');
      function upd() { const M = model(); S.M = M; const x = M.arr[S.k]; rA.set(M.al.toFixed(3)); rT.set(nf(1 + x.a, 2)); rP.set(nf(x.tps, 1) + ' tok/s'); rB.set(`k=${M.best} (${nf(M.arr[M.best].tps, 1)})`); }
      upd();
      V.fig(fig, (g, w, h, t) => {
        const C = V.C(), M = S.M, k = S.k, narrow = w < 560;
        // ---------- palco: rascunhar e verificar ----------
        const P = 1.3 + 0.2 * k, ri = Math.floor(t / P), f = (t % P) / P;
        if (ri !== S.round) {
          S.round = ri; let a = 0; while (a < k && rng() < M.al) a++; S.acc = a;
          if (ri > 0) { S.tok += a + 1; S.steps += 1; }
        }
        const acc = S.acc;
        const dX = 34, dY = 72, vX = w - 40, sz = 40;
        // rascunhador
        if (k > 0) {
          V.box(g, dX - sz / 2, dY - sz / 2, sz, sz, { r: 12, fill: V.alpha(C.b, 0.55), stroke: C.b, lw: 2 });
          g.fillStyle = C.ink; g.beginPath(); g.arc(dX - 7, dY - 4, 3, 0, 7); g.arc(dX + 7, dY - 4, 3, 0, 7); g.fill();
          V.text(g, 'drafter', dX - sz / 2, dY + sz / 2 + 15, { size: 11, color: C.muted });
        }
        // alvo
        const vs = 50;
        V.box(g, vX - vs / 2, dY - vs / 2, vs, vs, { r: 14, fill: V.alpha(C.a, 0.5), stroke: C.a, lw: 2 });
        g.fillStyle = C.ink; g.beginPath(); g.arc(vX - 9, dY - 6, 3.5, 0, 7); g.arc(vX + 9, dY - 6, 3.5, 0, 7); g.fill();
        g.strokeStyle = C.ink; g.lineWidth = 2; g.beginPath(); g.arc(vX, dY + 4, 8, 0.2, Math.PI - 0.2); g.stroke();
        V.text(g, 'target (MiMo)', vX + vs / 2, dY + vs / 2 + 15, { size: 11, color: C.muted, align: 'right' });
        // cartas
        const nC = Math.max(1, k + 1), x0 = dX + 34, x1 = vX - 40, cw = Math.min(34, (x1 - x0) / nC - 5), cy = dY - 18;
        for (let i = 0; i < k; i++) {
          const appear = f > (i / Math.max(1, k)) * 0.28;
          if (!appear) continue;
          const fly = clamp((f - (i / k) * 0.28) / 0.08, 0, 1);
          const cx = lerp(dX, x0 + i * (cw + 5), V.ease.out(fly));
          const stampT = 0.32 + ((i + 1) / (k + 1)) * 0.42;
          let fill = V.alpha(C.b, 0.35), mark = '', mcol = C.ink;
          if (f > stampT) {
            if (i < acc) { fill = V.alpha(C.a, 0.65); mark = '✓'; }
            else if (i === acc) { fill = V.alpha(C.red, 0.55); mark = '✕'; mcol = C.red; }
            else { fill = V.alpha(C.muted, 0.12); mark = ''; }
          }
          V.box(g, cx, cy, cw, 36, { r: 6, fill, stroke: V.alpha(C.ink, 0.25), lw: 1 });
          V.text(g, mark || 't' + (i + 1), cx + cw / 2, cy + 22, { size: mark ? 14 : 11, align: 'center', color: mark ? mcol : C.ink, weight: mark ? 700 : 400, font: mark ? 'sans' : 'mono' });
        }
        // token do alvo (correção ou bônus)
        if (f > 0.78 || k === 0) {
          const pos = k === 0 ? 0 : Math.min(acc, k);
          const bxx = x0 + pos * (cw + 5), by = k === 0 ? cy : cy + 40;
          const pop = V.ease.back(clamp((f - (k === 0 ? 0.3 : 0.78)) / 0.12, 0, 1));
          V.box(g, bxx, by, cw, 36 * pop, { r: 6, fill: V.alpha(C.c, 0.7), stroke: C.c, lw: 1.5 });
          if (pop > 0.8) V.text(g, acc === k && k > 0 ? 'bonus' : 'target', bxx + cw / 2, by + 22, { size: 11, align: 'center', color: C.ink, maxW: cw - 2 });
        }
        const avg = S.steps ? S.tok / S.steps : 0;
        V.text(g, `this step: ${acc} ${acc === 1 ? 'accepted' : 'accepted'} + 1 from the target = ${acc + 1} ${acc ? 'tokens' : 'token'}`, 16, 16, { size: 12, color: C.ink, weight: 600, maxW: w - 32 });
        V.text(g, `simulated average: ${S.steps ? nf(avg, 2) : '—'} tokens/step · expected ${nf(1 + M.arr[k].a, 2)} (α = ${M.al.toFixed(3)})`, 16, 32, { size: 11.5, color: C.muted, maxW: w - 32 });
        // ---------- gráfico: tok/s × k ----------
        const cy0 = narrow ? 186 : 168, chH = h - cy0 - (narrow ? 150 : 134), cx0 = 44, cwAll = w - cx0 - 16;
        const meas = CATS[S.cat].tps;
        const mx = Math.max(...M.arr.map((x) => x.tps), meas || 0) * 1.22;
        const hyp = `hypothesis · c_d = ${Math.round(M.cd * 1000)} ms, β = ${nf(M.beta * 1000, 1)} ms/token` + (M.clamped ? ' (c_d capped)' : '');
        if (narrow) { V.text(g, `predicted tok/s (C=1) · ${CATS[S.cat].n}`, cx0, cy0 - 38, { size: 11.5, color: C.muted, maxW: cwAll }); V.text(g, hyp, cx0, cy0 - 22, { size: 11, color: C.muted, maxW: cwAll }); if (meas) V.text(g, `○ measured at k=7: ${nf(meas, 1)} [tonyd2wild]`, cx0, cy0 - 6, { size: 11, color: C.d, maxW: cwAll }); }
        else V.text(g, `predicted tok/s per stream (C=1) · ${CATS[S.cat].n} · ${hyp}` + (meas ? ` · ○ measured at k=7: ${nf(meas, 1)} [tonyd2wild]` : ''), cx0, cy0 - 6, { size: 11.5, color: C.muted, maxW: cwAll });
        V.line(g, cx0, cy0 + chH, cx0 + cwAll, cy0 + chH, { color: V.alpha(C.muted, 0.5), lw: 1 });
        const bw = cwAll / 8 - 8;
        M.arr.forEach((x, i) => {
          const bh = (x.tps / mx) * (chH - 16), bx = cx0 + i * (bw + 8) + 4, by = cy0 + chH - bh;
          const col = i === k ? C.c : i === M.best ? C.a : V.alpha(C.a, 0.4);
          V.box(g, bx, by, bw, bh, { r: 4, fill: col });
          V.text(g, nf(x.tps, 1), bx + bw / 2, by - 5, { size: 11, align: 'center', color: C.ink, font: 'mono' });
          V.text(g, i === 0 ? 'k=0' : 'k=' + i, bx + bw / 2, cy0 + chH + 14, { size: 11, align: 'center', color: i === k ? C.ink : C.muted, font: 'mono' });
          if (i === M.best) V.text(g, '★', bx + bw / 2, by - 19, { size: 13, align: 'center', color: C.a });
          if (i === 7 && meas) {
            const my = cy0 + chH - (meas / mx) * (chH - 16);
            g.strokeStyle = C.d; g.lineWidth = 2.5; g.beginPath(); g.arc(bx + 10, my, 6, 0, 7); g.stroke();
          }
        });
        // ---------- capture sizes ----------
        const sy = h - (narrow ? 100 : 88);
        const list = S.cap === 'tut' ? [8, 16, 24, 32, 40, 48, 56, 64] : [1, 2, 3, 4, 5, 6, 7, 8].map((n) => n * (k + 1));
        V.text(g, `decode batches with k=${k}: n seq × ${k + 1} tokens · captured: ${list.join(' ')}`, 16, sy - 10, { size: 11.5, color: C.muted, maxW: w - 32 });
        const cellW = (w - 32) / 8 - 5, maxL = Math.max(...list);
        for (let n = 1; n <= 8; n++) {
          const tk = n * (k + 1), cx = 16 + (n - 1) * (cellW + 5);
          let st, col;
          if (list.includes(tk)) { st = '✓ exact'; col = C.a; }
          else if (tk <= maxL) { const nx = list.filter((x) => x >= tk).sort((a, b) => a - b)[0]; st = `→${nx} +${Math.round((nx - tk) / tk * 100)}%`; col = C.d; }
          else { st = 'eager'; col = C.red; }
          V.box(g, cx, sy, cellW, narrow ? 52 : 44, { r: 7, fill: V.alpha(col, 0.18), stroke: col, lw: 1.3 });
          V.text(g, `${n}×${k + 1}=${tk}`, cx + cellW / 2, sy + 16, { size: 11, align: 'center', font: 'mono', color: C.ink, maxW: cellW - 4 });
          V.text(g, st, cx + cellW / 2, sy + (narrow ? 38 : 34), { size: 11, align: 'center', color: col, weight: 600, maxW: cellW - 4 });
        }
      });
    })();

    /* =====================================================================
       Figura 9 — teto de decode: esteira de bytes
       ===================================================================== */
    (function figTeto() {
      const fig = root.querySelector('#fig-teto');
      setH(fig, 410);
      const S = { fp8: false, eff: 0.58, meas: true };
      const comps = () => [
        { n: 'active experts', gb: 2.51, c: 'b' }, { n: 'qkv FP8', gb: 1.44, c: 'a' },
        { n: S.fp8 ? 'o_proj FP8' : 'o_proj BF16', gb: S.fp8 ? 0.805 : 1.61, c: 'c' },
        { n: 'lm_head', gb: 0.63, c: 'd' }, { n: 'router + dense MLP', gb: 0.2, c: 'e' },
      ];
      const ctl = UI.controls(fig);
      UI.toggle(ctl, { label: 'o_proj in FP8 (#59083)', value: false, onChange: (v) => { S.fp8 = v; upd(); } });
      UI.slider(ctl, { label: 'Efficiency', min: 0.4, max: 0.8, step: 0.01, value: 0.58, fmt: (v) => Math.round(v * 100) + '%', onInput: (v) => { S.eff = v; upd(); } });
      UI.toggle(ctl, { label: 'show measurements with DFlash k=7', value: true, onChange: (v) => { S.meas = v; } });
      const rB = UI.readout(ctl, 'Bytes/step/rank', '—'), rC = UI.readout(ctl, 'Ceiling', '—'), rR = UI.readout(ctl, 'Realistic', '—');
      function upd() { const tot = comps().reduce((s, c) => s + c.gb, 0); S.tot = tot; rB.set(nf(tot, 2) + ' GB'); rC.set(nf(273 / tot, 1) + ' tok/s'); rR.set(nf(273 * S.eff / tot, 1) + ' tok/s'); }
      upd();
      V.fig(fig, (g, w, h, t) => {
        const C = V.C(), cs = comps(), tot = S.tot, narrow = w < 560;
        const stepMs = tot / 273 * 1000, realMs = stepMs / S.eff;
        // ---------- esteira ----------
        const mbx = 12, mbw = narrow ? 70 : 92, gbw = narrow ? 56 : 70, by0 = 44, bh0 = 62;
        V.box(g, mbx, by0, mbw, bh0, { r: 10, fill: V.alpha(C.b, 0.14), stroke: C.b, lw: 1.5 });
        V.text(g, 'LPDDR5X', mbx + mbw / 2, by0 + 26, { size: 11.5, align: 'center', weight: 600, color: C.ink });
        V.text(g, '273 GB/s', mbx + mbw / 2, by0 + 44, { size: 11, align: 'center', font: 'mono', color: C.muted });
        const gx = w - 12 - gbw;
        V.box(g, gx, by0, gbw, bh0, { r: 10, fill: V.alpha(C.a, 0.14), stroke: C.a, lw: 1.5 });
        V.text(g, 'GPU', gx + gbw / 2, by0 + 26, { size: 11.5, align: 'center', weight: 600, color: C.ink });
        V.text(g, '1 rank', gx + gbw / 2, by0 + 44, { size: 11, align: 'center', font: 'mono', color: C.muted });
        const bx0 = mbx + mbw + 6, bx1 = gx - 6, bl = bx1 - bx0, beltY = by0 + bh0 / 2;
        V.line(g, bx0, beltY + 16, bx1, beltY + 16, { color: V.alpha(C.muted, 0.6), lw: 2 });
        V.line(g, bx0, beltY + 16, bx1, beltY + 16, { color: V.alpha(C.ink, 0.5), lw: 2, dash: [3, 9], dashOffset: -t * 40 });
        const ppg = bl / tot * 0.9, gapPx = bl * 0.1 / cs.length;
        const Ltot = bl;
        const period = Math.max(1.6, realMs / 18);
        const off = ((t / period) % 1) * Ltot;
        g.save(); g.beginPath(); g.rect(bx0, by0 - 4, bl, bh0 + 8); g.clip();
        let acc = 0;
        cs.forEach((c) => {
          const ww = c.gb * ppg;
          for (const rep of [0, 1]) {
            const xx = bx0 + ((acc + off) % Ltot) - rep * Ltot;
            if (xx + ww < bx0 || xx > bx1) continue;
            V.box(g, xx, beltY - 12, ww, 26, { r: 5, fill: V.alpha(C[c.c], 0.75) });
            if (ww > 50) V.text(g, nf(c.gb, 2), xx + ww / 2, beltY + 5, { size: 11, align: 'center', font: 'mono', color: C.ink });
          }
          acc += ww + gapPx;
        });
        g.restore();
        V.text(g, `one step reads ${nf(tot, 2)} GB per rank → ≥ ${nf(stepMs, 1)} ms at 100% of bandwidth · ~${nf(realMs, 1)} ms at ${Math.round(S.eff * 100)}%`, 12, 22, { size: 12, color: C.ink, weight: 600, maxW: w - 24 });
        // ---------- barra empilhada + legenda ----------
        const sy = 140, sw = w - 24;
        let xx = 12;
        cs.forEach((c) => { const ww = c.gb / tot * sw; V.box(g, xx, sy, Math.max(1, ww - 2), 22, { r: 4, fill: V.alpha(C[c.c], 0.8) }); xx += ww; });
        let lx = 12, ly = sy + 40;
        cs.forEach((c) => {
          const lab = `${c.n} ${nf(c.gb, 2)}`;
          g.font = V.font(11.5, 'sans', 400);
          const tw = g.measureText(lab).width + 30;
          if (lx + tw > w - 8) { lx = 12; ly += 18; }
          V.box(g, lx, ly - 10, 11, 11, { r: 3, fill: C[c.c] }); V.text(g, lab, lx + 16, ly, { size: 11.5, color: C.muted }); lx += tw;
        });
        // ---------- régua de tok/s ----------
        const ry = Math.max(ly + 64, h - 92), rx0 = 24, rx1 = w - 24, maxT = 80, X = (v) => rx0 + (v / maxT) * (rx1 - rx0);
        const teto = 273 / tot;
        g.fillStyle = V.alpha(C.a, 0.16); g.fillRect(X(teto * 0.5), ry - 12, X(teto * 0.65) - X(teto * 0.5), 24);
        V.line(g, rx0, ry, rx1, ry, { color: V.alpha(C.muted, 0.7), lw: 1.5 });
        for (let v = 0; v <= maxT; v += 10) { V.line(g, X(v), ry - 4, X(v), ry + 4, { color: V.alpha(C.muted, 0.7), lw: 1 }); V.text(g, String(v), X(v), ry + 18, { size: 11, align: 'center', font: 'mono', color: C.muted }); }
        V.text(g, 'tok/s per stream (C=1)', rx1, ry + 34, { size: 11, align: 'right', color: C.muted });
        // teto e realista
        V.line(g, X(teto), ry - 22, X(teto), ry + 6, { color: C.ink, lw: 2 });
        V.text(g, `ceiling ${nf(teto, 1)}`, X(teto), ry - 27, { size: 11.5, align: 'center', color: C.ink, weight: 600 });
        const real = teto * S.eff;
        V.dot(g, X(real), ry, 6, C.a); V.glow(g, X(real), ry, 14 + 2 * Math.sin(t * 3), C.a, 0.8);
        V.text(g, `no spec ~${nf(real, 1)}`, X(real), ry + 34, { size: 11.5, align: 'center', color: C.a, weight: 600 });
        if (S.meas) {
          [[25.9, 'prose'], [53.3, 'average'], [70.5, 'code']].forEach(([v, l], i) => {
            const yy = ry - 38 - (i % 2) * 16;
            V.line(g, X(v), yy + 4, X(v), ry - 2, { color: V.alpha(C.c, 0.8), lw: 1.4, dash: [2, 3] });
            V.dot(g, X(v), ry, 4.5, C.c);
            V.text(g, `${l} ${nf(v, 1)}`, X(v), yy, { size: 11, align: 'center', color: C.c, weight: 600 });
          });
        }
      });
    })();

    /* =====================================================================
       Figura 10 — A/B/A contra a deriva
       ===================================================================== */
    (function figABA() {
      const fig = root.querySelector('#fig-aba');
      setH(fig, 500);
      const S = { eff: 0, drift: 0.6, noise: 1.2, seed: 10, t0: 0, vals: [], med: [] };
      let H = null;
      const med3 = (a) => a.slice().sort((x, y) => x - y)[1];
      function sim(anim) {
        const r = V.rng(S.seed), base = 53.3;
        S.vals = [];
        for (let i = 0; i < 9; i++) { const arm = i < 3 ? 0 : i < 6 ? 1 : 2; const d = base * (1 + S.drift / 100 * i); S.vals.push({ i, arm, d, v: d * (arm === 1 ? 1 + S.eff / 100 : 1) + S.noise * r.normal() }); }
        S.med = [0, 1, 2].map((a) => med3(S.vals.filter((x) => x.arm === a).map((x) => x.v)));
        if (anim) S.t0 = H ? H.t : 0;
      }
      const ctl = UI.controls(fig);
      UI.slider(ctl, { label: 'Real effect of B', min: -8, max: 8, step: 0.5, value: 0, fmt: (v) => (v > 0 ? '+' : '') + v.toFixed(1) + '%', onInput: (v) => { S.eff = v; sim(false); } });
      UI.slider(ctl, { label: 'Drift', min: 0, max: 1.5, step: 0.1, value: 0.6, fmt: (v) => v.toFixed(1) + '%/run', onInput: (v) => { S.drift = v; sim(false); } });
      UI.slider(ctl, { label: 'Noise σ', min: 0, max: 4, step: 0.1, value: 1.2, fmt: (v) => v.toFixed(1) + ' tok/s', onInput: (v) => { S.noise = v; sim(false); } });
      UI.button(ctl, '↻ new draw', () => { S.seed++; sim(true); }, { primary: true });
      sim(false);
      H = V.fig(fig, (g, w, h, t) => {
        const C = V.C(), wide = w >= 640;
        const px0 = 52, py0 = 26, pw = wide ? w * 0.6 - 52 : w - 110, ph = wide ? h - 80 : h * 0.52;
        const vs = S.vals.map((x) => x.v).concat(S.vals.map((x) => x.d));
        const lo = Math.min(...vs) - 1.5, hi = Math.max(...vs) + 1.5;
        const Y = (v) => py0 + ph - ((v - lo) / (hi - lo)) * ph;
        const colArm = [C.b, C.c, C.b], names = ['A₁', 'B', 'A₂'];
        const slot = pw / 9.8;
        const X = (i) => px0 + (i + 0.4 + Math.floor(i / 3) * 0.4) * slot;
        // faixas dos braços
        for (let a = 0; a < 3; a++) {
          const xa = X(a * 3) - slot * 0.45, xb = X(a * 3 + 2) + slot * 0.45;
          g.fillStyle = V.alpha(colArm[a], 0.07); g.fillRect(xa, py0, xb - xa, ph);
          V.text(g, names[a], (xa + xb) / 2, py0 + ph + 18, { size: 13, align: 'center', color: colArm[a], weight: 700 });
        }
        // eixo
        V.line(g, px0, py0, px0, py0 + ph, { color: V.alpha(C.muted, 0.6), lw: 1 });
        for (let k = 0; k <= 4; k++) { const v = lo + (hi - lo) * k / 4; V.text(g, nf(v, 1), px0 - 6, Y(v) + 4, { size: 11, align: 'right', font: 'mono', color: C.muted }); }
        V.text(g, 'tok/s (illustrative)', px0, py0 - 10, { size: 11, color: C.muted });
        // deriva
        V.line(g, X(0), Y(S.vals[0].d), X(8), Y(S.vals[8].d), { color: V.alpha(C.d, 0.8), lw: 1.5, dash: [5, 4] });
        V.text(g, 'drift', X(8) + 6, Y(S.vals[8].d) - 4, { size: 11, color: C.d });
        // medianas
        for (let a = 0; a < 3; a++) { const xa = X(a * 3) - slot * 0.35, xb = X(a * 3 + 2) + slot * 0.35; V.line(g, xa, Y(S.med[a]), xb, Y(S.med[a]), { color: colArm[a], lw: 3.5 }); V.text(g, nf(S.med[a], 1), xb + 3, Y(S.med[a]) + 4, { size: 11, font: 'mono', color: colArm[a] }); }
        // pontos (aparecem em sequência)
        S.vals.forEach((x) => {
          const e = clamp((t - S.t0 - x.i * 0.16) / 0.3, 0, 1);
          if (e <= 0) return;
          V.dot(g, X(x.i), Y(x.v), 5 * V.ease.back(e), colArm[x.arm]);
        });
        // veredito
        const s1 = S.med[1] / S.med[0] - 1, s2 = S.med[1] / S.med[2] - 1;
        const naive = s1 >= 0.03, aba = s1 >= 0.03 && s2 >= 0.03;
        const vx = wide ? px0 + pw + 34 : 16, vy = wide ? py0 + 6 : py0 + ph + 42, vw = wide ? w - vx - 14 : w - 32;
        const pct = (x) => (x >= 0 ? '+' : '') + (x * 100).toFixed(1) + '%';
        let yy = wrap(g, 'naive A/B (B vs. A₁)', vx, vy, vw, 15, { size: 12, color: C.muted, weight: 600 });
        yy = wrap(g, `${pct(s1)} → ${naive ? 'ACCEPT' : 'REJECT'}`, vx, yy + 2, vw, 20, { size: 17, font: 'display', color: naive ? C.a : C.red, weight: 700 });
        yy = wrap(g, 'A/B/A (≥ +3% vs. A₁ and vs. A₂)', vx, yy + 8, vw, 15, { size: 12, color: C.muted, weight: 600 });
        yy = wrap(g, `${pct(s1)} · ${pct(s2)} → ${aba ? 'ACCEPT' : 'REJECT'}`, vx, yy + 2, vw, 20, { size: 17, font: 'display', color: aba ? C.a : C.red, weight: 700 });
        if (naive && !aba) yy = wrap(g, '⚠ drift fooled the naive A/B: A₂ “improved” too', vx, yy + 8, vw, 15, { size: 12, color: C.d, weight: 600 });
        wrap(g, 'and also: no secondary metric gets worse by more than 2%, quality canary within the noise, no OOM in the worst case', vx, yy + 8, vw, 15, { size: 11.5, color: C.muted });
      });
    })();

    /* =====================================================================
       Figura 11 — Naive-N0.5: camada global densa × DSA com indexador top-k
       ===================================================================== */
    (function figNaive() {
      const fig = root.querySelector('#fig-naive');
      setH(fig, 540);
      const GA = [0, 5, 11, 17, 23, 29, 35, 41, 47];          // MiMo: hybrid_layer_pattern (0 = global)
      const DSA = [5, 11, 17, 23, 29, 35, 41, 47];            // Naive: posições ILUSTRATIVAS (~5:1)
      const S = { kind: 'dsa', layer: 11, L: 48, k: 8, W: 8, sc: [] };
      let H = null;
      const ctl = UI.controls(fig);
      const segK = UI.seg(ctl, { label: 'Layer', options: [{ v: 'ga', label: 'global (MiMo)' }, { v: 'dsa', label: 'DSA (Naive)' }, { v: 'swa', label: 'SWA (both)' }], value: 'dsa', onChange: (v) => { S.kind = v; S.layer = v === 'swa' ? 1 : 11; restart(); } });
      UI.slider(ctl, { label: 'Context L', min: 24, max: 96, step: 8, value: 48, fmt: (v) => v + ' tok', onInput: (v) => { S.L = v; S.k = Math.min(S.k, v - 1); gen(); restart(); } });
      UI.slider(ctl, { label: 'top-k', min: 4, max: 24, step: 1, value: 8, fmt: (v) => String(v), onInput: (v) => { S.k = Math.min(v, S.L - 1); restart(); } });
      const rStore = UI.readout(ctl, 'KV stored in the layer', '—'), rRead = UI.readout(ctl, 'KV read by the query', '—');
      function gen() {
        const r = V.rng(S.L * 7 + 3);
        const hot = [0.12, 0.37, 0.58, 0.8].map((f) => Math.floor(f * S.L));
        S.sc = Array.from({ length: S.L }, (_, i) => Math.max(0, 0.35 * r.normal() + 0.15 + hot.reduce((a, c) => a + 1.1 * Math.exp(-((i - c) ** 2) / 3), 0)));
        S.sc[S.L - 1] = Math.max(S.sc[S.L - 1], 0.9);
      }
      function topk() { return S.sc.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]).slice(0, S.k).map((x) => x[1]); }
      function restart() {
        S.t0 = H ? H.t : 0; S.sel = new Set(topk());
        const L = S.L;
        rStore.set(S.kind === 'swa' ? `window of ${S.W} (of ${L})` : S.kind === 'dsa' ? `${L} tokens + ${L} indexer keys` : `${L} tokens`);
        rRead.set(S.kind === 'ga' ? `${L} of ${L}` : S.kind === 'dsa' ? `${S.k} of ${L} (+ key scan)` : `${Math.min(S.W, L)} of ${L}`);
      }
      gen(); restart();
      H = V.fig(fig, (g, w, h, t, dt, _S, HH) => {
        const C = V.C(), narrow = w < 560, L = S.L;
        // ---------- as 48 camadas dos dois modelos ----------
        const lx = narrow ? 54 : 70, cw = (w - lx - 16) / 48, rowsY = [30, 64], rh = 24;
        const kindOf = (m, i) => (m === 0 ? (GA.includes(i) ? 'ga' : 'swa') : (DSA.includes(i) ? 'dsa' : 'swa'));
        const colK = { ga: C.b, dsa: C.f, swa: C.a };
        const P = HH.pointer;
        ['MiMo', 'Naive'].forEach((nm, m) => {
          V.text(g, nm, 16, rowsY[m] + 16, { size: 11.5, weight: 600, color: C.ink });
          for (let i = 0; i < 48; i++) {
            const kd = kindOf(m, i), x = lx + i * cw, y = rowsY[m];
            const selected = i === S.layer && ((S.kind === 'ga' && m === 0) || (S.kind === 'dsa' && m === 1) || (S.kind === 'swa'));
            g.fillStyle = V.alpha(colK[kd], kd === 'swa' ? 0.3 : 0.85);
            g.fillRect(x + 0.5, y, Math.max(1, cw - 1), rh);
            if (selected) { g.strokeStyle = C.ink; g.lineWidth = 2; g.strokeRect(x - 0.5, y - 2, cw + 1, rh + 4); }
            if (P.clicked && P.x >= x && P.x < x + cw && P.y >= y && P.y < y + rh) { S.kind = kd; S.layer = i; segK.set(kd); S.layer = i; }
          }
        });
        V.text(g, 'layer 0', lx, rowsY[1] + rh + 14, { size: 11, font: 'mono', color: C.muted });
        V.text(g, '47', lx + 48 * cw, rowsY[1] + rh + 14, { size: 11, font: 'mono', color: C.muted, align: 'right' });
        const legY = rowsY[1] + rh + (narrow ? 30 : 14);
        if (!narrow) [[C.b, 'dense global (9)'], [C.f, 'DSA (illustrative positions)'], [V.alpha(C.a, 0.5), 'SWA']].forEach(([c, l], i) => { const x = lx + 90 + i * 170; V.box(g, x, legY - 10, 11, 11, { r: 3, fill: c }); V.text(g, l, x + 16, legY, { size: 11, color: C.muted }); });
        else V.text(g, 'lavender: global · magenta: DSA · green: SWA', 16, legY, { size: 11, color: C.muted, maxW: w - 32 });
        // ---------- a camada escolhida ----------
        const e = t - (S.t0 || 0), cyc = 5.5, u = (e % cyc) / cyc;
        const kd = S.kind;
        const title = kd === 'ga' ? `layer ${S.layer} · MiMo · dense global: the query reads the KV of all L tokens`
          : kd === 'dsa' ? `layer ${S.layer} · Naive · DSA: the indexer scores every token and attention reads only the top-${S.k}`
          : `layer ${S.layer} · SWA (in both): the query reads only the window`;
        const ty = narrow ? 140 : 122;
        const yy0 = wrap(g, title, 16, ty, w - 32, 15, { size: 12, weight: 600, color: kd === 'ga' ? C.b : kd === 'dsa' ? C.f : C.a });
        const gx = narrow ? 52 : 70, gw = w - gx - 16, cc = gw / L;
        const barTop = yy0 + 4, barH = 44, idxY = barTop + barH + 6, kvY = idxY + 16, kh = narrow ? 15 : 18;
        // barras de nota do indexador (só DSA)
        const scanP = clamp(u / 0.35, 0, 1), selP = clamp((u - 0.35) / 0.15, 0, 1), readP = clamp((u - 0.5) / 0.4, 0, 1);
        const mx = Math.max(...S.sc) || 1;
        V.text(g, kd === 'dsa' ? 'score' : '', gx - 8, barTop + barH - 2, { size: 11, align: 'right', color: C.muted });
        if (kd === 'dsa') {
          for (let i = 0; i < L; i++) {
            const shown = clamp(scanP * L - i, 0, 1), bh = (S.sc[i] / mx) * barH * shown, isSel = S.sel.has(i);
            g.fillStyle = selP > 0 && isSel ? C.d : V.alpha(C.f, selP > 0 ? 0.25 : 0.6);
            g.fillRect(gx + i * cc + cc * 0.15, barTop + barH - bh, Math.max(1, cc * 0.7), bh);
          }
          if (scanP < 1) { const sxp = gx + scanP * gw; V.line(g, sxp, barTop - 2, sxp, idxY + 10, { color: C.f, lw: 2 }); }
          // linha das chaves do indexador
          g.fillStyle = V.alpha(C.f, 0.35); g.fillRect(gx, idxY, gw, 8);
          V.text(g, 'idx', gx - 8, idxY + 8, { size: 11, align: 'right', font: 'mono', color: C.muted });
        } else {
          V.text(g, kd === 'ga' ? 'no indexer: nothing to choose, it reads everything' : `sliding window of ${S.W} tokens (illustrative; in MiMo, 128)`, gx, barTop + barH - 8, { size: 11.5, color: C.muted, maxW: gw });
        }
        // KV: 4 KV heads (GQA) × L tokens
        const readCol = (i) => (kd === 'ga' ? true : kd === 'swa' ? i >= L - S.W : S.sel.has(i));
        for (let r = 0; r < 4; r++) {
          const y = kvY + r * (kh + 3);
          V.text(g, 'kv' + r, gx - 8, y + kh - 4, { size: 11, align: 'right', font: 'mono', color: C.muted });
          for (let i = 0; i < L; i++) {
            const on = readCol(i) && (kd !== 'dsa' || selP > 0);
            const a = on ? 0.25 + 0.6 * (kd === 'dsa' ? selP : 1) * (readP > 0 ? 0.7 + 0.3 * Math.sin(t * 6 + i) : 1) : 0.1;
            g.fillStyle = V.alpha(on ? (kd === 'ga' ? C.b : kd === 'dsa' ? C.d : C.a) : C.muted, a);
            g.fillRect(gx + i * cc + 0.5, y, Math.max(1, cc - 1), kh);
          }
        }
        const kvBot = kvY + 4 * (kh + 3);
        // a query (último token) e os arcos de leitura
        const qx = gx + (L - 0.5) * cc;
        V.box(g, gx + (L - 1) * cc - 1, kvY - 3, cc + 2, kvBot - kvY + 3, { r: 3, stroke: C.ink, lw: 1.5 });
        V.text(g, 'q', qx, kvBot + 13, { size: 11, align: 'center', font: 'mono', weight: 700, color: C.ink });
        if (readP > 0 || kd !== 'dsa') {
          const rp = kd === 'dsa' ? readP : clamp(u / 0.5, 0, 1);
          g.save(); g.globalAlpha = kd === 'ga' ? 0.45 : 0.85;
          for (let i = 0; i < L - 1; i++) {
            if (!readCol(i)) continue;
            const x = gx + (i + 0.5) * cc, depth = 12 + Math.min(36, Math.abs(qx - x) * 0.12);
            g.beginPath(); g.moveTo(qx, kvBot + 2);
            const steps = 16, n = Math.max(1, Math.round(steps * rp));
            for (let s2 = 1; s2 <= n; s2++) { const f = s2 / steps, bx = lerp(qx, x, f), by = kvBot + 2 + Math.sin(f * Math.PI) * depth; g.lineTo(bx, by); }
            g.strokeStyle = kd === 'ga' ? C.b : kd === 'dsa' ? C.d : C.a; g.lineWidth = kd === 'ga' ? 0.8 : 1.4; g.stroke();
          }
          g.restore();
        }
        // conta, em palavras
        let ty2 = kvBot + 72;
        const nRead = kd === 'ga' ? L : kd === 'swa' ? Math.min(S.W, L) : S.k;
        ty2 = wrap(g, `stored: ${kd === 'swa' ? `only the window (${S.W} tokens)` : `${L} tokens × 4 KV heads` + (kd === 'dsa' ? ' + one small indexer key per token' : '')} · read by the query: ${nRead} of ${L}` + (kd === 'dsa' ? ' (plus the scan of the small keys)' : ''), 16, ty2, w - 32, 15, { size: 12, color: C.ink });
        if (kd === 'dsa') wrap(g, 'here the top-k applies to all 4 KV heads at once; one top-k per KV group is the alternative. It is one of the decisions in “DSA with GQA”, which vLLM does not have yet (its DSA exists for MLA)', 16, ty2 + 4, w - 32, 15, { size: 11.5, color: C.muted });
        else if (kd === 'ga') wrap(g, 'in MiMo, the 9 global layers are the only ones whose KV grows with context (Step 1). Naive swaps all of them for DSA', 16, ty2 + 4, w - 32, 15, { size: 11.5, color: C.muted });
        else wrap(g, 'the SWA layers do not change: same window, same sink, the same mimo_v2.py code to reuse in the plugin', 16, ty2 + 4, w - 32, 15, { size: 11.5, color: C.muted });
      });
    })();

    /* =====================================================================
       Figura 12 — checklist final (MiMo / Naive / Mac)
       ===================================================================== */
    (function figCheck() {
      const fig = root.querySelector('#fig-check');
      const base = [
        'Step 0 · triage table + risks tied to PRs/issues',
        'Step 1 · predicted vs. measured (weights, pool, tokens) within ±10%',
        'Step 2 · checkpoint verified, revision pinned (KLD if not the official one)',
        'Step 3 · divisibility with no ✘, TP chosen, alternatives noted down',
        'Step 4 · overlays with CPU tests; parity TP1 = TP2 ≈ HF; markers in the log',
        'Step 5 · boot on the 2 nodes; expected backend, KV and capture sizes; rendered command saved',
        'Step 6 · acceptance × k × category × context; SPEC_METHOD/SPEC_K with rollback',
        'Step 7 · smoke.sh green (sanity, tools, needle, canary, OOM) noted in the README',
        'Step 8 · decode, cold prefill and acceptance with method, date, commit and digest',
        'Step 9 · tuning with the A/B/A rule; negative results in the CHANGELOG',
        'Step 10 · another pair of Sparks reproduces the README within ±5–10%',
      ].map((t) => ({ t }));
      const naive = base.slice();
      naive[0] = { t: 'Step 0 · no registry entry: NaiveN05FlashForCausalLM + remote code', d: 1 };
      naive[1] = { t: 'Step 1 · KV formula per DSA layer (KV + indexer key, top-k read)', d: 1 };
      naive[2] = { t: 'Step 2 · quantize experts (calibrated NVFP4) + KLD against FP8 on ≥ 3 nodes or truncated HF', d: 1 };
      naive[3] = { t: 'Step 3 · same backbone as MiMo: check divisibility and the pre-sharding in the index' };
      naive[4] = { t: 'Step 4 · out-of-tree plugin; DSA block with GQA; truncated parity with N = 6', d: 1 };
      naive[5] = { t: 'Step 5 · the sm_121 sparse-MLA backends do not work out of the box: choose or port', d: 1 };
      naive[6] = { t: 'Step 6 · FP8-Draft in a known format (EAGLE3/DFlash)? Otherwise, register it', d: 1 };
      const mac = [
        'Mac with ≥ 192 GB of unified memory (M3 Ultra 256 GB or M5 Ultra)',
        'mlx-lm from PR #1219 (branch kernelpool/mlx-lm@add-mimo-v2) in a venv',
        'MLX checkpoint: mlx-community (RL, text-only); MOPD via mlx_lm.convert [to measure]',
        'mlx_lm.server on 127.0.0.1:8080 answering /v1/chat/completions',
        'Thinking off via chat_template_kwargs (or in the client) and a generous max_tokens',
        'Tools in the Qwen3-Coder format tested with the Step 7 suite',
        'Decode and prefill measured (reference: 59.4 and 563 tok/s @2K on the M3 Ultra)',
      ].map((t) => ({ t }));
      const LISTS = { mimo: base, naive, mac };
      const saved = LA.store.get('rdz-checklist', {}) || {};
      const S = { mode: 'mimo', p: 0 };
      const ctl = UI.controls(fig);
      const list = LA.h('div', { style: { display: 'grid', gap: '9px', padding: '14px 18px', borderTop: '1px solid var(--line)' } });
      fig.insertBefore(list, fig.querySelector('figcaption'));
      function build() {
        list.innerHTML = '';
        const L = LISTS[S.mode], st = saved[S.mode] || [];
        L.forEach((it, i) => {
          const tg = UI.toggle(list, { label: (it.d ? '≠ ' : '') + it.t, value: !!st[i], onChange: (v) => { const a = saved[S.mode] || (saved[S.mode] = []); a[i] = v; LA.store.set('rdz-checklist', saved); } });
          const sw = tg.el.querySelector('i'); if (sw) sw.style.flexShrink = '0';
        });
      }
      UI.seg(ctl, { label: 'Mode', options: [{ v: 'mimo', label: 'MiMo · 2× Spark' }, { v: 'naive', label: 'Naive challenge' }, { v: 'mac', label: 'Mac · MLX' }], value: 'mimo', onChange: (v) => { S.mode = v; build(); } });
      UI.button(ctl, 'clear', () => { saved[S.mode] = []; LA.store.set('rdz-checklist', saved); build(); });
      build();
      const SHORT = {
        mimo: ['Triage', 'Memory', 'Weights', 'Topology', 'Engine', 'Launch', 'Speculation', 'Validation', 'Benchmark', 'Tuning', 'Packaging'],
        mac: ['Mac ≥192 GB', 'mlx-lm #1219', 'Checkpoint', 'Server', 'Thinking', 'Tools', 'Measure'],
      };
      SHORT.naive = SHORT.mimo;
      V.fig(fig, (g, w, h, t, dt, _S, HH) => {
        const C = V.C(), L = LISTS[S.mode], st = saved[S.mode] || [];
        const n = L.length, done = L.filter((_, i) => !!st[i]).length;
        const names = SHORT[S.mode];
        const phaseCol = (i) => (S.mode === 'mac' ? C.a : i <= 3 ? C.b : i <= 7 ? C.a : C.c);
        S.p += (done / n - S.p) * Math.min(1, dt * 4);
        const x0 = 34, x1 = w - 34, y = Math.round(h * 0.5), wideLab = w >= 640;
        const X = (i) => lerp(x0, x1, n === 1 ? 0 : i / (n - 1));
        // clique numa estação marca/desmarca o critério (e sincroniza a lista)
        const P = HH.pointer;
        let hover = -1;
        for (let i = 0; i < n; i++) if (P.inside && Math.hypot(P.x - X(i), P.y - y) < 16) hover = i;
        if (P.clicked && hover >= 0) { const a = saved[S.mode] || (saved[S.mode] = []); a[hover] = !a[hover]; LA.store.set('rdz-checklist', saved); build(); }
        // trilho por fase + trilho percorrido
        for (let i = 0; i < n - 1; i++) V.line(g, X(i), y, X(i + 1), y, { color: V.alpha(phaseCol(i + 1), 0.28), lw: 8 });
        const tx = lerp(x0, x1, S.p);
        V.line(g, x0, y, tx, y, { color: C.a, lw: 8 });
        g.setLineDash([2, 10]); g.lineDashOffset = -t * 18; V.line(g, x0, y, x1, y, { color: V.alpha(C.bg || '#000', 0.5), lw: 2 }); g.setLineDash([]);
        // estações
        for (let i = 0; i < n; i++) {
          const xx = X(i), ok = !!st[i], col = phaseCol(i), r = hover === i ? 12 : 10;
          V.dot(g, xx, y, r, ok ? col : (C.panel || C.bg));
          g.beginPath(); g.arc(xx, y, r, 0, 7); g.strokeStyle = ok ? col : V.alpha(col, 0.8); g.lineWidth = 2.5; g.stroke();
          V.text(g, ok ? '✓' : String(S.mode === 'mac' ? i + 1 : i), xx, y + 4, { size: 11, align: 'center', font: ok ? 'sans' : 'mono', weight: 700, color: ok ? onInk(C) : C.ink });
          if (L[i].d) V.text(g, '≠', xx, y - r - 6, { size: 13, align: 'center', color: C.d, weight: 700 });
          if (wideLab) {
            const below = i % 2 === 0, ly = below ? y + 32 : y - (L[i].d ? 30 : 22);
            V.text(g, names[i], xx, ly, { size: 11, align: 'center', color: ok ? C.ink : C.muted, weight: hover === i ? 600 : 400, maxW: (x1 - x0) / (n - 1) * 1.9 });
          }
        }
        // trem
        V.glow(g, tx, y, 22, C.a, 0.9);
        V.box(g, tx - 15, y - 10, 30, 20, { r: 9, fill: C.a });
        g.fillStyle = onInk(C); g.beginPath(); g.arc(tx - 5, y - 2, 2, 0, 7); g.arc(tx + 5, y - 2, 2, 0, 7); g.fill();
        // topo: progresso; base: próximo critério ou estação sob o mouse
        const allDone = done === n;
        const msg = allDone ? (S.mode === 'mac' ? '✓ done: the same OpenAI contract, now on the Mac' : '✓ done: a second pair reproduces the README (±5–10%)') : `${done} of ${n} exit criteria met`;
        V.text(g, msg, 24, 26, { size: allDone ? 15 : 13, weight: 600, color: allDone ? C.a : C.ink, maxW: w - 48, font: allDone ? 'display' : 'sans' });
        if (S.mode !== 'mac') {
          const leg = [[C.b, 'plan'], [C.a, 'bring it up right'], [C.c, 'measure and ship']];
          let lx = 24; g.font = V.font(11, 'sans', 400);
          leg.forEach(([c, l]) => { V.box(g, lx, 38, 10, 10, { r: 3, fill: c }); V.text(g, l, lx + 15, 47, { size: 11, color: C.muted }); lx += g.measureText(l).width + 34; });
        }
        const next = L.findIndex((_, i) => !st[i]), focus = hover >= 0 ? hover : next;
        if (focus >= 0 && !allDone) {
          const it = L[focus];
          wrap(g, (hover >= 0 ? 'station: ' : 'next: ') + it.t, 24, h - 34, w - 48, 15, { size: 12, color: hover >= 0 ? C.ink : C.muted, weight: hover >= 0 ? 600 : 400 });
        }
        if (allDone) for (let k = 0; k < 14; k++) { const a = t * 1.3 + k, xx = x1 - 30 + Math.cos(a * 1.7) * 22, yy = y - 50 + Math.sin(a * 2.3) * 12; V.dot(g, xx, yy, 2.2, C.series[k % 6]); }
        if (hover >= 0) HH.canvas.style.cursor = 'pointer'; else HH.canvas.style.cursor = '';
      });
    })();
  },
});

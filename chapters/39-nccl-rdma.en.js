LA.chapter({
  id: 'nccl-rdma',
  part: 8,
  order: 39,
  title: 'Communication: collectives, NCCL, and RDMA',
  navTitle: 'NCCL & RDMA',
  subtitle: 'What the GPUs say to each other at every step, what each sentence costs, and the dozens of variables that decide whether the conversation goes over RDMA, slows to a crawl, or hangs in silence.',
  minutes: 75,
  recipes: ['glm', 'tf', 'qwen', 'ds'],
  tags: ['all-reduce', 'NCCL', 'RDMA', 'RoCE v2', 'GID', 'ConnectX-7', 'rendezvous', 'shared memory', 'watchdog', 'TensorFold', 'RDMA WRITE', 'multi-rail'],
  goals: [
    'Explain the <b>collectives</b> (all-reduce, all-gather, reduce-scatter…) and the <b>ring</b> algorithm, and predict their cost with the <b>α-β model</b>',
    'Understand why decode is <b>latency</b>-bound (~100 small messages per step) and prefill is <b>bandwidth</b>-bound',
    "Configure RDMA/RoCE v2 on the ConnectX-7: <b>HCA</b>, <b>GID</b>, MTU, two PFs, containers, and vLLM's multi-node <b>rendezvous</b>",
    'Read and tune the <code>NCCL_*</code>, <code>GLOO_*</code>, and <code>TORCH_*</code> variables of the three recipes and diagnose hangs (watchdog, flight recorder, shared memory)',
    'Compare vLLM\'s network design with that of the <span class="la-rb tf"></span> variant: all-gather of fp32 partials, <b>one-shot RDMA</b> with the flag on the same QP, two rails, control over the wire itself, and the rules that keep both ranks at the same meeting',
  ],
  prereqs: ['paralelismos'],
  summary: [
    'Row-parallel layers produce partial sums; the <b>all-reduce</b> adds them up on every rank. In the recipes there are ≈106–108 collectives per decode step (Qwen/DS) — BF16 activations, not weights.',
    "α-β model: \\(T = \\alpha + n/\\beta\\); the ring costs \\(2(p-1)\\alpha + 2\\frac{p-1}{p}\\frac{n}{\\beta}\\), with break-even at \\(n_{1/2} \\approx p\\,\\alpha\\beta\\). Decode (20–57 KB at 1 sequence) is latency-bound — a floor of 2–5 ms/step that doesn't drop with batch; prefill (8–42 MB) is bandwidth-bound.",
    "RDMA lets the NIC write into the peer's memory with no kernel and no copies; RoCE v2 carries the IB transport in UDP/IP and uses IP-derived <b>GIDs</b>. On the measured GB10 stack (driver 580.173.02) no GPUDirect was observed: the collectives went through a host buffer (an observation of one stack, not a general rule).",
    "The GB10's ConnectX-7 serves each 200 Gb/s port with two PFs of ≈126 Gb/s raw (~100 practical); dual-HCA + MTU 9000 gave +7.6% end to end (DS), without changing decode. The GID index changes per node and per reboot — validate it, don't guess.",
    'Three planes: bootstrap TCP (TCPStore, Gloo, NCCL — <code>*_SOCKET_IFNAME</code>, real interface names), ZMQ/SHM control (<code>VLLM_HOST_IP</code>), and RDMA data (<code>NCCL_IB_HCA</code>). The <code>--headless</code> worker comes up first; <code>mp</code> executor; PyNCCL is the only all-reduce across nodes.',
    'Hangs: a JIT in the middle of serving or diverging ranks hold a collective until the 600 s watchdog; the flight recorder (<code>TORCH_FR_*</code>) leaves per-rank evidence. In shared memory, the spin window (1 s → 2 ms/16 ms) and the 5 s recheck (#117) prevent burned CPU and readers sleeping forever.',
    'In the <span class="la-rb tf"></span> variant (TensorFold), TP swaps the all-reduce for an all-gather of the fp32 partials, added in rank order, and gathers up to 512 KiB go through a <b>one-shot RDMA</b>: the GPU stages its shard in pinned memory, a proxy does an RDMA WRITE + flag on the same reliable QP, and the receiver waits on the flag (11 µs against 45 µs per 16 KiB; +6% decode). The transport is chosen by bytes alone, startup goes through an NCCL barrier, and large gathers use two rails (32 MB: 3.5 → 1.9 ms).',
  ],
  quiz: [
    { q: "In Qwen's decode at TP=2, each all-reduce carries ~20 KB. Why does doubling the link bandwidth (second PF) barely change decode tok/s?", o: ["Because decode doesn't use all-reduce", "Because 20 KB messages are far below \\(n_{1/2}\\approx p\\,\\alpha\\beta\\): the cost is almost entirely latency α, which extra bandwidth doesn't reduce", 'Because NCCL ignores the second PF', 'Because the quantized weights are already small'], a: 1, e: 'With α ≈ tens of µs and β ≈ 11–25 GB/s, the n/β term for 20 KB is ~1–2 µs. More bandwidth shrinks only that term; the ~100 × α floor remains. The DS recipe measured exactly this: +7.6% end to end (the source ties the all-reduce to a slice of the prefill step), decode with no material change.' },
    { q: 'In a ring all-reduce with p ranks and an n-byte buffer, how many bytes does each rank send?', o: ['\\(n\\)', '\\(\\frac{p-1}{p}n\\)', '\\(2\\frac{p-1}{p}n\\)', '\\((p-1)n\\)'], a: 2, e: 'Reduce-scatter (p−1 steps of n/p) + all-gather (another p−1 steps of n/p) = 2(p−1)/p · n — the theoretical minimum. For p=2, exactly n.' },
    { q: 'A rank dies ~60 s after starting with <code>ibv_modify_qp … errno 61</code>. What is the most likely cause according to the recipes?', o: ['MTU 1500', 'NCCL_IB_GID_INDEX pointing to an empty entry in the GID table (or set to empty, which is read as 0)', 'NCCL_NVLS_ENABLE=0', 'A small shm_size'], a: 1, e: "The QP transition to RTR needs the peer's GID; an all-zero entry (or a wrong index) kills the rank ~60 s later. That is why GLM validates the index on every HCA in the preflight and DS resolves the GID from sysfs without pinning it." },
    { q: 'Why does NCCL need <code>NCCL_SOCKET_IFNAME=enp1s0f1np1</code> even though <code>--master-addr 10.0.0.1</code> works?', o: ["Because TCPStore doesn't work over IPv4", 'Because 10.0.0.x are loopback aliases: they serve the rendezvous TCP, but NCCL picks interfaces by name (and excludes lo) — without the pin, ncclCommInitRank hangs', 'Because NCCL only accepts IPv6', 'Because Gloo uses RDMA'], a: 1, e: "The rendezvous and ZMQ use the alias routed through the CX7; NCCL's bootstrap, Gloo, and verbs need each node's real interface/HCA name (which changes with crossed cabling)." },
    { q: "DeepSeek's hotfix #79 changed <code>busy_loop_s</code> from 1 s to 2 ms. What did it fix?", o: ['A correctness bug in the logits', 'Shared-memory readers that never slept (the interval between decode steps is < 1 s), keeping 3–4 P-cores maxed out and stealing power/thermal headroom from the GPU on the same SoC', 'The loss of PUB/SUB notifications', 'The 600 s watchdog'], a: 1, e: 'Busy-waiting responds fast but occupies the CPU; with a 1 s window and messages every ~50–70 ms, the reader was always spinning. The loss of notifications is #117 (5 s recheck). GLM measured 16 ms as the best point in its setup (+0.95% decode, −85.3% CPU).' },
    { q: 'Does raising the RPC deadline (<code>VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS</code>) to 1800 s keep a 15-minute JIT from taking down the TP pair?', o: ['Yes, it extends every timeout', "No: it only extends the executor's RPC deadline; the ProcessGroupNCCL watchdog stays at 600 s and aborts the stalled collective", 'Yes, if NCCL_DEBUG=INFO', 'No, because the JIT runs on the head'], a: 1, e: 'The recipes raise the RPC from 300 to 1800 s because legitimate JITs take longer than 300 s, but they warn: “NCCL watchdog is still 600s”. The real defense is persisting caches and warming up shapes at boot; the flight recorder leaves the evidence.' },
    { q: 'In the TensorFold variant, why does the choice between the one-shot RDMA and NCCL depend only on the gather\'s byte count?', o: ['Because RDMA only accepts messages aligned to 4 KiB', 'Because the choice must come out the same on both ranks: a local criterion, such as the buffer address\'s alignment, once sent one rank to wait in RDMA and the other in NCCL, forever', 'Because NCCL does not accept messages smaller than 512 KiB', 'Because the byte count defines the GID index used'], a: 1, e: 'A collective only finishes if every rank makes the same call over the same transport. Patch 0040 replaced the criterion “fits within the limit and the buffer is aligned” with “fits within the limit”, which depends only on the bytes (equal on both ranks), with an aligned copy when needed. The limit itself, 512 KiB in the recipe, is the full batched window: 32 rows × 16 KiB.' },
  ],
  mount(root, LA) {
    const V = LA.viz, UI = LA.ui;

    /* ================================================================ helpers */
    const br = (x, d = 0) => (+x).toFixed(d);
    const pow2 = (b) => b >= 1048576 && Math.abs(Math.log2(b) - Math.round(Math.log2(b))) < 1e-9;
    const fmtB = (b) => (pow2(b) ? Math.round(b / 1048576) + ' MiB' : b >= 1e6 ? br(b / 1e6, b >= 1e7 ? 0 : 1) + ' MB' : b >= 1e3 ? br(b / 1e3, b >= 1e4 ? 0 : 1) + ' KB' : Math.round(b) + ' B');
    const fmtUs = (us) => (us >= 1e3 ? br(us / 1e3, us >= 1e4 ? 1 : 2) + ' ms' : br(us, us < 10 ? 1 : 0) + ' µs');
    const FC = {};
    const F = (s, f, w) => { const k = s + (f || 'sans') + (w || 400); return FC[k] || (FC[k] = V.font(s, f || 'sans', w || 400)); };
    const tw = (g, s, size, font, weight) => { g.font = F(size || 12, font, weight); return g.measureText(String(s)).width; };
    function wrap(g, s, maxW, size, font, weight) {
      const words = String(s).split(' '); const lines = []; let cur = '';
      words.forEach((wd) => { const test = cur ? cur + ' ' + wd : wd; if (tw(g, test, size, font, weight) > maxW && cur) { lines.push(cur); cur = wd; } else cur = test; });
      if (cur) lines.push(cur);
      return lines;
    }
    function para(g, s, x, y, maxW, o = {}) {
      const size = o.size || 12, lh = o.lh || size * 1.35;
      const lines = wrap(g, s, maxW, size, o.font, o.weight);
      lines.forEach((ln, k) => V.text(g, ln, x, y + k * lh, { size, font: o.font, weight: o.weight, color: o.color, align: o.align }));
      return lines.length * lh;
    }
    function chip(g, x, y, s, col, o = {}) {
      const size = o.size || 11, pad = o.pad == null ? 7 : o.pad, hh = o.h || 20;
      const wd = tw(g, s, size, o.font || 'mono', o.weight || 500) + pad * 2;
      const xx = o.align === 'center' ? x - wd / 2 : o.align === 'right' ? x - wd : x;
      V.box(g, xx, y, wd, hh, { r: o.r == null ? 6 : o.r, fill: V.alpha(col, o.a == null ? 0.16 : o.a), stroke: V.alpha(col, o.sa == null ? 0.6 : o.sa), lw: 1 });
      V.text(g, s, xx + wd / 2, y + hh / 2 + 0.5, { size, font: o.font || 'mono', weight: o.weight || 500, color: o.color || V.C().ink, align: 'center', base: 'middle' });
      return wd;
    }
    const quad = (p0, c, p1, u) => { const a = 1 - u; return [a * a * p0[0] + 2 * a * u * c[0] + u * u * p1[0], a * a * p0[1] + 2 * a * u * c[1] + u * u * p1[1]]; };
    function subLabel(g, base, sub, x, y, o = {}) {
      const size = o.size || 13;
      const w1 = tw(g, base, size, o.font || 'sans', 600), w2 = tw(g, sub, size * 0.75, o.font || 'sans', 600);
      const x0 = o.align === 'center' ? x - (w1 + w2) / 2 : x;
      V.text(g, base, x0, y, { size, weight: 600, color: o.color, font: o.font });
      V.text(g, sub, x0 + w1 + 1, y + size * 0.28, { size: Math.max(11, size * 0.75), weight: 600, color: o.color, font: o.font });
    }
    const fmtV = (v) => (Math.abs(v % 1) > 1e-9 ? br(v, 1) : String(v)).replace('-', '−');
    function mat(g, o) {
      const C = V.C();
      const G = V.matrix(g, Object.assign({}, o, { values: false }));
      const lim = Math.max(Math.abs(o.min), Math.abs(o.max));
      o.data.forEach((row, i) => row.forEach((v, j) => {
        const [x, y] = G.center(i, j), t = v / lim, light = Math.abs(t) > 0.55;
        V.text(g, fmtV(v), x, y + 0.5, { size: Math.max(11, Math.min(13, G.cw * 0.36)), font: 'mono', weight: 600, align: 'center', base: 'middle', color: light ? (C.dark ? '#00210F' : '#fff') : C.ink, maxW: G.cw - 2 });
      }));
      return G;
    }

    /* ============================================ Fig 1: row-parallel + count */
    (function () {
      const fig = root.querySelector('#fig-rowpar');
      const ctl = UI.controls(fig);
      const S = { rec: 'qwen' };
      const REC = {
        qwen: { name: 'Qwen3.8 (MTP=3, batch 1)', total: 106, segs: [['GDN out_proj', 36], ['attention o_proj', 12], ['MoE', 48], ['logits (AG)', 1], ['drafts', 9]], note: '≈106 × 20–30 µs ≈ 2–3 ms floor per step (audit estimate)', msg: 'message: 4 tok × 2560 × 2 B = 20 KB' },
        ds: { name: 'DeepSeek-V4 (k=6, c=1)', total: 108, segs: [['wo_b', 43], ['MoE', 43], ['draft', 6], ['embeddings', 2], ['logits (AG)', 2], ['Markov (serial)', 12]], note: '≈108 × 30–50 µs ≈ 3–5 ms floor per step (no GPUDirect RDMA observed; see section 4)', msg: 'message: T × 4096 × 2 B (BF16)' },
        glm: { name: 'GLM-5.3 (45 layers)', total: 90, approx: true, segs: [['KDA/MLA attention o_proj', 45], ['dense MLP', 3], ['MoE', 42]], note: 'our own count, 2 per layer (as in Parallelism); the repository only records “one all-reduce per MoE layer”; embedding, logits, and drafter not included', msg: 'message: 4096 × 2 B = 8 KB per token' },
      };
      UI.seg(ctl, { label: 'Recipe', options: [{ v: 'glm', label: 'GLM' }, { v: 'qwen', label: 'Qwen' }, { v: 'ds', label: 'DS' }], value: 'qwen', onChange: (v) => { S.rec = v; } });
      const steps = [
        { t: 'Input already sliced', d: 'The previous (column-parallel) layer left <b>half</b> of the vector on each node: \\(x_A=[0.5,\\,-1]\\) on the head and \\(x_B=[1.5,\\,2]\\) on the worker. No communication so far.' },
        { t: 'Weights split by row', d: 'Node A holds rows 0–1 of \\(W\\) and node B rows 2–3. Each element of \\(x\\) will scale <i>an entire row</i> of \\(W\\).' },
        { t: 'Local partial sum', d: 'Each node computes \\(x_iW_i\\) on its own: A gets \\([0,\\,-1,\\,2]\\), B gets \\([-1.5,\\,2,\\,2]\\). Neither of them has the answer.' },
        { t: 'All-reduce over the cable', d: "The partials cross the ConnectX-7 (RDMA) and each node adds the other's to its own: both end up with \\(y=[-1.5,\\,1,\\,4]\\) — exactly the unsplit \\(xW\\)." },
        { t: 'And again, every layer', d: 'This repeats after <i>every</i> row-parallel layer. The bar below counts the collectives of one decode step for the chosen recipe; each tick costs tens of µs, whatever its size.' },
      ];
      const st = UI.stepper(fig, steps, { autoplay: 6500 });
      const X = [0.5, -1, 1.5, 2];
      const W = [[1, 0, 2], [0.5, 1, -1], [-1, 2, 0], [0, -0.5, 1]];
      const YA = [0, -1, 2], YB = [-1.5, 2, 2], Y = [-1.5, 1, 4];
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const i = st.i, e = st.elapsed();
        const narrow = w < 560;
        const m = 12, barH = narrow ? 140 : 116, topH = h - barH;
        const cableW = V.clamp(w * 0.13, 34, 120);
        const pw = (w - 2 * m - cableW) / 2;
        const c = V.clamp(Math.floor((pw - (narrow ? 22 : 40)) / (narrow ? 4.3 : 4.6)), 20, 42);
        const gap = 2;
        const nodes = [
          { x: m, name: narrow ? 'Node A · head' : 'Node A · head', sub: 'rank 0', col: C.a, lab: 'A', xv: [X[0], X[1]], Wv: [W[0], W[1]], yv: YA },
          { x: w - m - pw, name: narrow ? 'Node B · worker' : 'Node B · worker', sub: 'rank 1', col: C.b, lab: 'B', xv: [X[2], X[3]], Wv: [W[2], W[3]], yv: YB },
        ];
        const xg = narrow ? 10 : 14, bw = c + xg + 3 * c + 2 * gap;
        const by = 62, yy = by + 2 * c + gap + 36, yy2 = yy + c + 36;
        const geo = [];
        nodes.forEach((N, ni) => {
          V.box(g, N.x, 8, pw, topH - 16, { r: 14, fill: V.alpha(N.col, 0.05), stroke: V.alpha(N.col, 0.35) });
          V.text(g, N.name, N.x + 10, 28, { size: narrow ? 12 : 13, weight: 600, color: N.col });
          if (!narrow) V.text(g, N.sub, N.x + pw - 10, 28, { size: 11, font: 'mono', color: C.muted, align: 'right' });
          const bx = N.x + (pw - bw) / 2, wx = bx + c + xg;
          subLabel(g, 'x', N.lab, bx + c / 2, by - 10, { size: 13, color: C.ink, align: 'center' });
          subLabel(g, 'W', N.lab, wx + (3 * c + 2 * gap) / 2, by - 10, { size: 13, color: C.ink, align: 'center' });
          const hiRow = i === 2 ? Math.min(1, Math.floor(e / 1.1)) : -1;
          mat(g, { x: bx, y: by, data: N.xv.map((v) => [v]), cell: c, gap, cmap: 'div', min: -2, max: 2, hiRow });
          const aW = i === 0 ? 0.16 : i === 1 ? V.clamp(e / 0.8, 0.16, 1) : 1;
          g.globalAlpha = aW;
          mat(g, { x: wx, y: by, data: N.Wv, cell: c, gap, cmap: 'div', min: -2, max: 2, hiRow });
          g.globalAlpha = 1;
          if (i >= 1) for (let r = 0; r < 2; r++) {
            const yc = by + r * (c + gap) + c / 2;
            V.arrow(g, bx + c + 2, yc, wx - 2, yc, { color: r === hiRow ? C.c : V.alpha(C.muted, 0.7), lw: r === hiRow ? 2.2 : 1.2, head: 6 });
          }
          // partial sum
          const aY = i < 2 ? 0 : i === 2 ? V.clamp((e - 2.2) / 0.6, 0, 1) : 1;
          V.text(g, 'partial', bx + c / 2, yy + c / 2 + 4, { size: 11, color: C.muted, align: 'center' });
          if (aY > 0) {
            g.globalAlpha = aY;
            mat(g, { x: wx, y: yy, data: [N.yv], cell: c, gap, cmap: 'div', min: -4, max: 4 });
            g.globalAlpha = 1;
          } else {
            for (let k = 0; k < 3; k++) { V.rr(g, wx + k * (c + gap), yy, c, c, 4); g.strokeStyle = V.alpha(C.muted, 0.35); g.setLineDash([3, 3]); g.lineWidth = 1; g.stroke(); g.setLineDash([]); }
          }
          // final result
          const aF = i < 3 ? 0 : i === 3 ? V.clamp((e - 1.5) / 0.5, 0, 1) : 1;
          V.text(g, 'y', bx + c / 2, yy2 + c / 2 + 5, { size: 13, weight: 600, color: aF > 0 ? C.a : C.muted, align: 'center' });
          if (aF > 0) {
            g.globalAlpha = aF;
            mat(g, { x: wx, y: yy2, data: [Y], cell: c, gap, cmap: 'div', min: -4, max: 4, hiRow: 0, hiColor: C.a });
            g.globalAlpha = 1;
            V.text(g, 'partial A + partial B', wx + (3 * c + 2 * gap) / 2, yy2 + c + 16, { size: 11, color: C.muted, align: 'center' });
          } else {
            for (let k = 0; k < 3; k++) { V.rr(g, wx + k * (c + gap), yy2, c, c, 4); g.strokeStyle = V.alpha(C.muted, 0.3); g.setLineDash([3, 3]); g.lineWidth = 1; g.stroke(); g.setLineDash([]); }
          }
          geo[ni] = { wx, yy, yy2 };
        });
        // cable
        const x1 = nodes[0].x + pw, x2 = nodes[1].x, cyC = yy + c / 2;
        V.line(g, x1, cyC, x2, cyC, { color: V.alpha(C.d, 0.35), lw: 6 });
        V.line(g, x1, cyC, x2, cyC, { color: C.d, lw: 1.5, dash: [4, 6], dashOffset: -t * 30 });
        V.text(g, cableW >= 70 ? 'CX7 · RDMA' : 'CX7', (x1 + x2) / 2, cyC - 12, { size: 11, font: 'mono', color: C.d, align: 'center' });
        // packets
        if (i === 3) {
          const u = V.ease.inOut(V.clamp(e / 1.4, 0, 1));
          if (u < 1) {
            [[0, 1], [1, 0]].forEach(([a, b]) => {
              for (let k = 0; k < 3; k++) {
                const p0 = [geo[a].wx + k * (c + gap) + c / 2, geo[a].yy + c / 2];
                const p1 = [geo[b].wx + k * (c + gap) + c / 2, geo[b].yy2 + c / 2];
                const ctrl = [(x1 + x2) / 2, a === 0 ? cyC - 30 - k * 6 : cyC + 30 + k * 6];
                const pt = quad(p0, ctrl, p1, u);
                const col = a === 0 ? C.a : C.b;
                V.box(g, pt[0] - c * 0.42, pt[1] - c * 0.36, c * 0.84, c * 0.72, { r: 5, fill: V.alpha(col, 0.85) });
                V.text(g, fmtV(nodes[a].yv[k]), pt[0], pt[1] + 0.5, { size: Math.max(11, c * 0.34), font: 'mono', weight: 600, color: C.dark ? '#07130D' : '#fff', align: 'center', base: 'middle' });
              }
            });
          }
        }
        // collectives bar
        const R = REC[S.rec];
        const y0 = topH + 2;
        const bx = m, bwid = w - 2 * m, bh = 24, bY = y0 + 26;
        V.text(g, 'Collectives per decode step · ' + R.name, bx, y0 + 14, { size: narrow ? 12 : 12.5, weight: 600, color: C.ink, maxW: bwid - 70 });
        V.text(g, (R.approx ? '≈' : '≈') + R.total + (R.approx ? '+' : ''), bx + bwid, y0 + 16, { size: 18, font: 'display', weight: 700, color: i === 4 ? C.c : C.ink, align: 'right' });
        const scaleTot = R.approx ? R.total * 1.3 : R.total;
        let acc = 0;
        const cur = ((t * 14) % scaleTot);
        R.segs.forEach(([name, n], k) => {
          const x = bx + (acc / scaleTot) * bwid, ww = (n / scaleTot) * bwid;
          const col = C.series[k % C.series.length];
          const on = cur >= acc && cur < acc + n;
          V.box(g, x + 0.5, bY, Math.max(2, ww - 1), bh, { r: 5, fill: V.alpha(col, on ? 0.7 : 0.38), stroke: on ? col : null, lw: 1.5 });
          const lab = name + ' ' + n;
          if (tw(g, lab, 11, 'sans', 600) < ww - 8) V.text(g, lab, x + ww / 2, bY + bh / 2 + 0.5, { size: 11, weight: 600, color: C.ink, align: 'center', base: 'middle' });
          else if (tw(g, String(n), 11, 'mono', 600) < ww - 4) V.text(g, String(n), x + ww / 2, bY + bh / 2 + 0.5, { size: 11, font: 'mono', weight: 600, color: C.ink, align: 'center', base: 'middle' });
          acc += n;
        });
        if (R.approx) {
          const x = bx + (acc / scaleTot) * bwid, ww = bwid - (x - bx);
          V.rr(g, x + 2, bY, ww - 2, bh, 5); g.setLineDash([4, 4]); g.strokeStyle = V.alpha(C.muted, 0.7); g.lineWidth = 1.2; g.stroke(); g.setLineDash([]);
          if (tw(g, '+ not counted', 11) < ww - 6) V.text(g, '+ not counted', x + ww / 2, bY + bh / 2 + 0.5, { size: 11, color: C.muted, align: 'center', base: 'middle' });
        }
        if (cur < acc) {
          const xc = bx + (cur / scaleTot) * bwid;
          V.line(g, xc, bY - 5, xc, bY + bh + 5, { color: C.ink, lw: 2 });
          V.glow(g, xc, bY + bh / 2, 12, C.c, 0.8);
        }
        if (i === 4) { V.rr(g, bx - 3, bY - 3, bwid + 6, bh + 6, 7); g.strokeStyle = C.c; g.lineWidth = 2; g.stroke(); }
        let ly = bY + bh + 17;
        const legend = R.segs.map((s) => s[1] + ' ' + s[0]).join(' · ');
        ly += para(g, legend + '  —  ' + R.msg, bx, ly, bwid, { size: 11.5, color: C.text2 || C.ink, lh: 15 });
        para(g, R.note, bx, ly + 1, bwid, { size: 11.5, color: C.muted, lh: 15 });
      });
    })();

    /* ============================================ Fig 2: zoo of collectives */
    (function () {
      const fig = root.querySelector('#fig-zoo');
      const ctl = UI.controls(fig);
      const S = { op: 'allreduce', p: 4, t0: 0 };
      const OPS = [
        { v: 'broadcast', label: 'broadcast' }, { v: 'reduce', label: 'reduce' }, { v: 'allreduce', label: 'all-reduce' },
        { v: 'allgather', label: 'all-gather' }, { v: 'reducescatter', label: 'reduce-scatter' }, { v: 'alltoall', label: 'all-to-all' },
      ];
      const INFO = {
        broadcast: ['broadcast: the root (rank 0) copies its buffer to everyone.', '≈ n bytes per rank (pipelined). The scheduler plan is a logical broadcast, but it travels over SHM/ZMQ, not through NCCL.'],
        reduce: ['reduce: sum of everyone, delivered only to the root.', 'Rare in inference with TP: everyone needs the result.'],
        allreduce: ['all-reduce: sum of everyone, delivered to everyone.', '2(p−1)/p · n bytes per rank. After every row-parallel layer (o_proj, MoE, GDN out_proj).'],
        allgather: ['all-gather: each rank contributes a piece; everyone receives everything.', '(p−1)/p · n bytes per rank. Logits of the vocabulary-split lm_head.'],
        reducescatter: ['reduce-scatter: a sum, but each rank keeps only its own slice.', '(p−1)/p · n bytes per rank. It is the 1st half of the ring all-reduce.'],
        alltoall: ['all-to-all: each pair exchanges a different piece (a transpose).', '(p−1)/p · n bytes per rank. Expert-parallel dispatch/combine — idle in the recipes (DP=1).'],
      };
      UI.select(ctl, { label: 'Collective', options: OPS, value: S.op, onChange: (v) => { S.op = v; S.t0 = performance.now(); } });
      UI.seg(ctl, { label: 'Ranks', options: [2, 3, 4], value: 4, onChange: (v) => { S.p = v; S.t0 = performance.now(); } });
      UI.button(ctl, '↻ replay', () => { S.t0 = performance.now(); });
      S.t0 = performance.now();
      const L = ['A', 'B', 'C', 'D'];
      function model(op, p) {
        const before = [], after = [], moves = [];
        for (let r = 0; r < p; r++) { before[r] = []; after[r] = []; }
        const own = (r, k) => ({ kind: 'data', lab: L[r] + k, src: r });
        const empty = () => ({ kind: 'empty' });
        const sum = (k) => ({ kind: 'sum', lab: 'Σ' + k });
        for (let r = 0; r < p; r++) for (let k = 0; k < p; k++) {
          if (op === 'broadcast') { before[r][k] = r === 0 ? own(0, k) : empty(); after[r][k] = own(0, k); if (r !== 0) moves.push([0, k, r, k]); }
          else if (op === 'reduce') { before[r][k] = own(r, k); after[r][k] = r === 0 ? sum(k) : { kind: 'dim', lab: L[r] + k, src: r }; if (r !== 0) moves.push([r, k, 0, k]); }
          else if (op === 'allreduce') { before[r][k] = own(r, k); after[r][k] = sum(k); for (let q = 0; q < p; q++) if (q !== r) moves.push([q, k, r, k]); }
          else if (op === 'allgather') { before[r][k] = r === k ? own(r, k) : empty(); after[r][k] = own(k, k); if (r !== k) moves.push([k, k, r, k]); }
          else if (op === 'reducescatter') { before[r][k] = own(r, k); after[r][k] = r === k ? sum(k) : empty(); if (r === k) for (let q = 0; q < p; q++) if (q !== r) moves.push([q, k, r, k]); }
          else if (op === 'alltoall') { before[r][k] = { kind: 'data', lab: L[r] + '→' + k, src: r }; after[r][k] = { kind: 'data', lab: L[k] + '→' + r, src: k }; if (r !== k) moves.push([r, k, k, r]); }
        }
        return { before, after, moves };
      }
      V.fig(fig, (g, w, h) => {
        const C = V.C();
        const p = S.p, M = model(S.op, p);
        const cols = [C.a, C.b, C.c, C.d];
        const m = 16, colW = (w - 2 * m - 40) / p, xL = m + 40;
        const cw = Math.min(70, colW - 14), ch = p <= 3 ? 30 : 26, gp = 5;
        const blockH = p * ch + (p - 1) * gp;
        const y0 = w < 560 ? 48 : 56, y1 = y0 + blockH + (w < 560 ? 40 : 58);
        const el = (performance.now() - S.t0) / 1000;
        const cyc = el % 5.2;
        const moveU = V.clamp((cyc - 1.0) / 1.9, 0, 1);
        const showAfter = cyc > 2.7 ? V.clamp((cyc - 2.7) / 0.4, 0, 1) : 0;
        const cellXY = (r, k, top) => [xL + r * colW + (colW - cw) / 2, (top ? y0 : y1) + k * (ch + gp)];
        for (let r = 0; r < p; r++) V.text(g, 'rank ' + r, xL + r * colW + colW / 2, y0 - 20, { size: 12, weight: 600, color: cols[r], align: 'center' });
        V.text(g, 'before', m, y0 + blockH / 2, { size: 11, color: C.muted, base: 'middle' });
        V.text(g, 'after', m, y1 + blockH / 2, { size: 11, color: C.muted, base: 'middle' });
        const drawCell = (cell, x, y, alpha) => {
          g.globalAlpha = alpha;
          if (cell.kind === 'empty') {
            V.rr(g, x, y, cw, ch, 5); g.setLineDash([3, 3]); g.strokeStyle = V.alpha(C.muted, 0.45); g.lineWidth = 1; g.stroke(); g.setLineDash([]);
          } else if (cell.kind === 'sum') {
            V.box(g, x, y, cw, ch, { r: 5, fill: V.alpha(C.a, 0.4), stroke: C.a, lw: 1.3 });
            for (let q = 0; q < p; q++) V.dot(g, x + 6 + q * 6, y + ch - 5, 2, cols[q]);
            V.text(g, cell.lab, x + cw / 2, y + ch / 2 - 1, { size: 12, font: 'mono', weight: 700, color: C.ink, align: 'center', base: 'middle' });
          } else {
            const col = cols[cell.src];
            V.box(g, x, y, cw, ch, { r: 5, fill: V.alpha(col, cell.kind === 'dim' ? 0.1 : 0.28), stroke: V.alpha(col, cell.kind === 'dim' ? 0.3 : 0.8), lw: 1 });
            V.text(g, cell.lab, x + cw / 2, y + ch / 2 + 0.5, { size: 12, font: 'mono', weight: 600, color: cell.kind === 'dim' ? C.muted : C.ink, align: 'center', base: 'middle' });
          }
          g.globalAlpha = 1;
        };
        for (let r = 0; r < p; r++) for (let k = 0; k < p; k++) {
          const [x, y] = cellXY(r, k, true); drawCell(M.before[r][k], x, y, 1);
          const [x2, y2] = cellXY(r, k, false);
          if (showAfter > 0) drawCell(M.after[r][k], x2, y2, showAfter);
          else { V.rr(g, x2, y2, cw, ch, 5); g.strokeStyle = V.alpha(C.muted, 0.18); g.lineWidth = 1; g.stroke(); }
        }
        // guide arrows and particles
        if (moveU > 0 && moveU < 1) {
          M.moves.forEach((mv, idx) => {
            const [sr, sk, dr, dk] = mv;
            const [sx, sy] = cellXY(sr, sk, true), [dx, dy] = cellXY(dr, dk, false);
            const delay = (idx % 7) * 0.035;
            const u = V.ease.inOut(V.clamp((moveU - delay) / (1 - 0.25), 0, 1));
            const p0 = [sx + cw / 2, sy + ch / 2], p1 = [dx + cw / 2, dy + ch / 2];
            const ctrl = [(p0[0] + p1[0]) / 2 + (sr - dr) * 8, (p0[1] + p1[1]) / 2];
            g.strokeStyle = V.alpha(cols[sr], 0.12); g.lineWidth = 1; g.beginPath(); g.moveTo(p0[0], p0[1]); g.quadraticCurveTo(ctrl[0], ctrl[1], p1[0], p1[1]); g.stroke();
            const pt = quad(p0, ctrl, p1, u);
            V.glow(g, pt[0], pt[1], 7, cols[sr], 0.9);
            V.dot(g, pt[0], pt[1], 2.6, cols[sr]);
          });
        }
        const [l1, l2] = INFO[S.op];
        let ty = y1 + blockH + 24;
        ty += para(g, l1, m, ty, w - 2 * m, { size: 12.5, weight: 600, color: C.ink, lh: 16 });
        para(g, l2, m, ty + 2, w - 2 * m, { size: 12, color: C.muted, lh: 15 });
      });
    })();

    /* ============================================ Fig 3: ring all-reduce + α-β */
    (function () {
      const fig = root.querySelector('#fig-ring');
      const ctl = UI.controls(fig);
      const S = { p: 3, e: Math.log2(20e3), alpha: 12, beta: 11, preset: 'dec' };
      const PRE = {
        dec: { n: 20e3, N: 106, lab: 'collectives per decode step (Qwen)', note: 'Audits: ≈20–30 µs per small all-reduce (Qwen); 30–50 µs without GPUDirect (DS).' },
        ds: { n: 1024 * 4096 * 2, N: 86, lab: 'all-reduces per 1024-token chunk (DS)', note: 'DS audit: ~60 ms per chunk at ~11 GB/s effective on a single PCIe function (PF) of the card.' },
        qw: { n: 8192 * 2560 * 2, N: 96, lab: 'all-reduces per 8192-token chunk (Qwen)', note: 'Qwen audit: ~150–180 ms per chunk on a 200 GbE link (β ≈ 25 GB/s).' },
      };
      UI.seg(ctl, { label: 'Ranks', options: [2, 3, 4], value: 3, onChange: (v) => { S.p = v; build(); } });
      const sN = UI.slider(ctl, { label: 'Message', min: 12, max: 26, step: 0.1, value: S.e, fmt: (v) => fmtB(Math.pow(2, v)), onInput: (v) => { S.e = v; } });
      UI.seg(ctl, { label: 'Preset', options: [{ v: 'dec', label: 'decode 20 KB' }, { v: 'ds', label: 'DS prefill 8 MiB' }, { v: 'qw', label: 'Qwen prefill 42 MB' }], value: 'dec', onChange: (v) => { S.preset = v; sN.set(Math.log2(PRE[v].n)); } });
      UI.slider(ctl, { label: 'alpha/step', min: 2, max: 60, step: 1, value: S.alpha, fmt: (v) => v + ' µs', onInput: (v) => { S.alpha = v; } });
      UI.slider(ctl, { label: 'bandwidth (beta)', min: 2, max: 25, step: 0.5, value: S.beta, fmt: (v) => br(v, v % 1 ? 1 : 0) + ' GB/s', onInput: (v) => { S.beta = v; } });
      const BASE = [[3, 1, 4, 1], [5, 9, 2, 6], [5, 3, 5, 8], [9, 7, 9, 3]];
      let st = null, states = [], transfers = [];
      function build() {
        const p = S.p;
        const s0 = []; for (let r = 0; r < p; r++) { s0[r] = []; for (let k = 0; k < p; k++) s0[r][k] = { v: BASE[r][k], m: 1 << r }; }
        states = [s0]; transfers = [[]];
        const clone = (s) => s.map((row) => row.map((c) => ({ v: c.v, m: c.m })));
        for (let s = 0; s < p - 1; s++) {
          const prev = states[states.length - 1], nx = clone(prev), tr = [];
          for (let r = 0; r < p; r++) { const k = (r - s + p * 4) % p, d = (r + 1) % p; nx[d][k] = { v: prev[d][k].v + prev[r][k].v, m: prev[d][k].m | prev[r][k].m }; tr.push({ r, d, k, v: prev[r][k].v, m: prev[r][k].m }); }
          states.push(nx); transfers.push(tr);
        }
        for (let s = 0; s < p - 1; s++) {
          const prev = states[states.length - 1], nx = clone(prev), tr = [];
          for (let r = 0; r < p; r++) { const k = (r + 1 - s + p * 4) % p, d = (r + 1) % p; nx[d][k] = { v: prev[r][k].v, m: prev[r][k].m }; tr.push({ r, d, k, v: prev[r][k].v, m: prev[r][k].m }); }
          states.push(nx); transfers.push(tr);
        }
        const frac = br(2 * (p - 1) / p, p === 2 ? 0 : 2);
        const steps = [{ t: 'Initial state', d: `Each of the ${p} ranks has its own buffer, cut into ${p} chunks. We want everyone to end up with the sum of each chunk.${p === 2 ? ' With p=2 (the recipes) the ring has only 2 steps.' : ' The recipes use p=2; a larger p here shows the mechanism better.'}` }];
        for (let s = 1; s < p; s++) steps.push({ t: `Reduce-scatter · step ${s} of ${p - 1}`, d: `Each rank sends <b>one</b> chunk (n/p bytes) to its right-hand neighbor, which <b>adds</b> it to its own. Everyone transmits at the same time.${s === p - 1 ? ' Now each rank owns <b>one</b> complete chunk (green).' : ''}` });
        for (let s = 1; s < p; s++) steps.push({ t: `All-gather · step ${s} of ${p - 1}`, d: `Each rank forwards an <b>already complete</b> chunk; the neighbor just <b>copies</b> it.${s === p - 1 ? ` Done: everyone has Σ in every chunk. That took 2(p−1) = ${2 * (p - 1)} steps, and each rank sent 2(p−1)/p · n = ${frac}·n bytes.` : ''}` });
        if (st) { st.stop(); const old = fig.querySelector('.la-stepper'); if (old) old.remove(); }
        st = UI.stepper(fig, steps, { autoplay: 4200 });
      }
      build();
      V.fig(fig, (g, w, h) => {
        const C = V.C();
        const p = S.p, i = Math.min(st.i, states.length - 1), e = st.elapsed();
        const cols = [C.a, C.b, C.c, C.d];
        const narrow = w < 560;
        const tlH = narrow ? 178 : 150, ringH = h - tlH;
        const cell = narrow ? 24 : p === 2 ? 46 : p === 3 ? 38 : 34, gp = 3;
        const bw = p * cell + (p - 1) * gp + 16, bh = cell + 34;
        const ang = p === 2 ? [Math.PI, 0] : p === 3 ? [-Math.PI / 2, Math.PI / 6, 5 * Math.PI / 6] : [-Math.PI / 2, 0, Math.PI / 2, Math.PI];
        const sins = ang.map(Math.sin), coss = ang.map(Math.cos);
        const sMin = Math.min(...sins), sMax = Math.max(...sins), cMin = Math.min(...coss), cMax = Math.max(...coss);
        const kv = 0.92;
        const Rv = sMax - sMin > 0.01 ? (ringH - bh - 34) / (kv * (sMax - sMin)) : 1e9;
        const Rh = (w - bw - 28) / (cMax - cMin);
        const R = Math.min(Rv, Rh, p === 2 ? 230 : 260);
        const cx = w / 2 - ((cMax + cMin) / 2) * R;
        const cy = p === 2 ? ringH / 2 + 4 : 22 + bh / 2 - sMin * R * kv;
        const P = ang.map((a) => [cx + Math.cos(a) * R, cy + Math.sin(a) * R * (p === 2 ? 1 : kv)]);
        const ctrlOf = (r) => {
          const a = P[r], b = P[(r + 1) % p];
          if (p === 2) return [cx, r === 0 ? cy - R * 0.55 : cy + R * 0.55];
          const mid = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
          const dx = mid[0] - cx, dy = mid[1] - cy, L0 = Math.hypot(dx, dy) || 1;
          return [cx + dx / L0 * (L0 + R * 0.35), cy + dy / L0 * (L0 + R * 0.35)];
        };
        // ring arcs
        for (let r = 0; r < p; r++) {
          const a = P[r], b = P[(r + 1) % p], c = ctrlOf(r);
          g.strokeStyle = V.alpha(C.muted, 0.35); g.lineWidth = 1.5; g.setLineDash([4, 5]);
          g.beginPath(); g.moveTo(a[0], a[1]); g.quadraticCurveTo(c[0], c[1], b[0], b[1]); g.stroke(); g.setLineDash([]);
          const q = quad(a, c, b, 0.55), q2 = quad(a, c, b, 0.6);
          V.arrow(g, q[0], q[1], q2[0], q2[1], { color: V.alpha(C.muted, 0.7), head: 7, lw: 1.5 });
        }
        const u = i === 0 ? 1 : V.ease.inOut(V.clamp(e / 1.3, 0, 1));
        const disp = i > 0 && u < 1 ? states[i - 1] : states[i];
        const tr = transfers[i] || [];
        const cellPos = (r, k) => [P[r][0] - bw / 2 + 8 + k * (cell + gp), P[r][1] - bh / 2 + 24];
        const pc = (m) => { let n = 0; for (let q = 0; q < p; q++) if (m & (1 << q)) n++; return n; };
        const owner = (m) => { for (let q = 0; q < p; q++) if (m === 1 << q) return q; return -1; };
        const cellFill = (cc) => { const n = pc(cc.m); if (n === p) return V.alpha(C.a, 0.55); if (n > 1) return V.alpha(C.d, 0.38); return V.alpha(cols[owner(cc.m)], 0.26); };
        for (let r = 0; r < p; r++) {
          const x = P[r][0] - bw / 2, y = P[r][1] - bh / 2;
          V.box(g, x, y, bw, bh, { r: 10, fill: V.alpha(cols[r], 0.08), stroke: V.alpha(cols[r], 0.7), lw: 1.5 });
          V.text(g, 'rank ' + r, x + 8, y + 15, { size: 11.5, weight: 600, color: cols[r] });
          for (let k = 0; k < p; k++) {
            const [cxx, cyy] = cellPos(r, k); const cc = disp[r][k];
            V.box(g, cxx, cyy, cell, cell, { r: 5, fill: cellFill(cc), stroke: pc(cc.m) === p ? C.a : null, lw: 1.2 });
            V.text(g, String(cc.v), cxx + cell / 2, cyy + cell / 2 - 2, { size: narrow ? 11.5 : 13, font: 'mono', weight: 600, color: C.ink, align: 'center', base: 'middle' });
            for (let q = 0; q < p; q++) V.dot(g, cxx + cell / 2 + (q - (p - 1) / 2) * 5, cyy + cell - 4, 1.6, cc.m & (1 << q) ? cols[q] : V.alpha(C.muted, 0.25));
            const isSrc = tr.some((o) => o.r === r && o.k === k), isDst = tr.some((o) => o.d === r && o.k === k);
            if (i > 0 && u < 1 && isSrc) { V.rr(g, cxx - 2, cyy - 2, cell + 4, cell + 4, 6); g.strokeStyle = C.c; g.lineWidth = 2; g.stroke(); }
            if (i > 0 && u >= 1 && isDst && e < 2.2) V.glow(g, cxx + cell / 2, cyy + cell / 2, cell, C.a, 0.6);
          }
        }
        if (i > 0 && u < 1) tr.forEach((o) => {
          const [sx, sy] = cellPos(o.r, o.k), [dx, dy] = cellPos(o.d, o.k);
          const c = ctrlOf(o.r);
          const pt = quad([sx + cell / 2, sy + cell / 2], c, [dx + cell / 2, dy + cell / 2], u);
          const n = pc(o.m);
          V.box(g, pt[0] - cell / 2, pt[1] - cell / 2, cell, cell, { r: 5, fill: n === p ? V.alpha(C.a, 0.9) : n > 1 ? V.alpha(C.d, 0.9) : V.alpha(cols[o.r], 0.9) });
          V.text(g, String(o.v), pt[0], pt[1] + 0.5, { size: 12, font: 'mono', weight: 700, color: C.dark ? '#07130D' : '#fff', align: 'center', base: 'middle' });
        });
        const phase = i === 0 ? 'start' : i <= p - 1 ? 'reduce-scatter' : 'all-gather';
        chip(g, w - 12, 10, phase, i === 0 ? C.muted : i <= p - 1 ? C.d : C.a, { align: 'right', size: 11 });
        // α-β stopwatch
        const n = Math.pow(2, S.e), chunk = n / p;
        const xfer = chunk / (S.beta * 1e9) * 1e6;
        const stepT = S.alpha + xfer, T = 2 * (p - 1) * stepT;
        const m = 12, y0 = ringH + 4, bwid = w - 2 * m;
        V.line(g, m, y0 - 2, w - m, y0 - 2, { color: V.alpha(C.muted, 0.25), lw: 1 });
        V.text(g, 'α-β stopwatch for this all-reduce', m, y0 + 14, { size: 12.5, weight: 600, color: C.ink });
        if (!narrow) V.text(g, 'T = 2(p−1)·(α + n/(p·β))', w - m, y0 + 14, { size: 11.5, font: 'mono', color: C.muted, align: 'right' });
        const nS = 2 * (p - 1), barY = y0 + 24, barH = 22;
        for (let s = 0; s < nS; s++) {
          const x = m + (s / nS) * bwid, ww = bwid / nS;
          const aW = (S.alpha / stepT) * (ww - 2);
          V.box(g, x + 1, barY, Math.max(1, aW), barH, { r: 4, fill: V.alpha(C.c, 0.6) });
          V.box(g, x + 1 + aW, barY, Math.max(1, ww - 2 - aW), barH, { r: 4, fill: V.alpha(C.a, 0.6) });
          if (s === i - 1) { V.rr(g, x, barY - 2, ww, barH + 4, 5); g.strokeStyle = C.ink; g.lineWidth = 2; g.stroke(); }
          const lab = (s < p - 1 ? 'RS' : 'AG') + (s % (p - 1) + 1);
          if (ww > 30) V.text(g, lab, x + ww / 2, barY + barH / 2 + 0.5, { size: 11, font: 'mono', weight: 600, color: C.ink, align: 'center', base: 'middle' });
        }
        const pre = PRE[S.preset];
        const n12 = p * S.alpha * S.beta * 1e3;
        const regime = xfer < S.alpha / 3 ? 'LATENCY-bound (α dominates)' : xfer > 3 * S.alpha ? 'BANDWIDTH-bound (n/β dominates)' : 'mixed regime';
        let ty = barY + barH + 16;
        const lines = [
          [`step = α + (n/p)/β = ${fmtUs(S.alpha)} + ${fmtUs(xfer)}   ·   T = ${fmtUs(T)}   ·   ${regime}`, C.ink, 12],
          [`n = ${fmtB(n)} · chunk = ${fmtB(chunk)} · n½ ≈ p·α·β ≈ ${fmtB(n12)}   ·   × ${pre.N} ${pre.lab} ≈ ${fmtUs(T * pre.N)}${p !== 2 ? ' (counts assume p=2)' : ''}`, C.text2 || C.muted, 11.5],
          [pre.note + '  (pink = α · green = n/β)', C.muted, 11.5],
        ];
        lines.forEach(([s, col, sz]) => { ty += para(g, s, m, ty, bwid, { size: sz, color: col, lh: sz + 3.5 }) + 2; });
      });
    })();

    /* ============================================ Fig 4: TCP vs RDMA vs GB10 */
    (function () {
      const fig = root.querySelector('#fig-rdma');
      const ctl = UI.controls(fig);
      const S = { mode: 'gb10', enc: 'v2' };
      UI.seg(ctl, { label: 'Path', options: [{ v: 'tcp', label: 'TCP (sockets)' }, { v: 'gdr', label: 'RDMA + GPUDirect' }, { v: 'gb10', label: 'RDMA via host (GB10)' }], value: S.mode, onChange: (v) => { S.mode = v; } });
      UI.seg(ctl, { label: 'Packet', options: [{ v: 'ib', label: 'InfiniBand' }, { v: 'v1', label: 'RoCE v1' }, { v: 'v2', label: 'RoCE v2' }], value: S.enc, onChange: (v) => { S.enc = v; } });
      const MODES = {
        tcp: { stats: '4 copies per message · CPU and kernel in the data path · in the log: NET/Socket', lat: 'tens of µs + CPU per byte' },
        gdr: { stats: '0 copies · kernel off the hot path · in the log: via NET/IB/0/GDRDMA', lat: 'the shortest path (servers with GPUDirect)' },
        gb10: { stats: '2 copies (GPU ↔ host buffer) · no kernel · in the log: via NET/IB/x, no /GDRDMA', lat: 'DS: 30–50 µs per collective with the detour through the host' },
      };
      const ENC = {
        ib: { f: [['LRH', 8], ['GRH', 40], ['BTH', 12], ['payload', 0], ['ICRC', 4], ['VCRC', 2]], note: 'InfiniBand: a network of its own (IB switches, subnet manager, LIDs), lossless thanks to credits.' },
        v1: { f: [['Eth 0x8915', 14], ['GRH', 40], ['BTH', 12], ['payload', 0], ['ICRC', 4], ['FCS', 4]], note: 'RoCE v1: IB transport directly over Ethernet (layer 2): it does not cross routers.' },
        v2: { f: [['Eth', 14], ['IPv4', 20], ['UDP 4791', 8], ['BTH', 12], ['payload', 0], ['ICRC', 4], ['FCS', 4]], note: 'RoCE v2: inside UDP/IP, routable; the GID comes from the IP. It is the one the recipes use (NCCL_IB_ROCE_VERSION_NUM=2).' },
      };
      const pk = Array.from({ length: 7 }, (_, k) => k / 7);
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const narrow = w < 560;
        const m = 12, encH = 128, topH = h - encH;
        const colW = narrow ? (w - 2 * m - 44) / 2 : Math.min(250, (w - 2 * m - 90) / 2);
        const Ax = m, Bx = w - m - colW;
        const ys = [38, 96, 154, 214], bH = 40;
        const mode = S.mode;
        const LAB = narrow ? ['GPU (LPDDR5X)', 'host buffer', 'kernel TCP/IP', 'NIC CX7'] : ['GPU · NCCL buffer', 'host buffer (pinned)', 'kernel · TCP/IP stack', 'NIC · ConnectX-7'];
        const active = { tcp: [1, 1, 1, 1], gdr: [1, 0, 0, 1], gb10: [1, 1, 0, 1] }[mode];
        [Ax, Bx].forEach((X0, side) => {
          V.text(g, side === 0 ? 'Node A (sends)' : 'Node B (receives)', X0 + colW / 2, 24, { size: 12, weight: 600, color: side === 0 ? C.a : C.b, align: 'center' });
          ys.forEach((y, k) => {
            const on = active[k];
            const col = [C.a, C.d, C.c, C.b][k];
            V.box(g, X0, y, colW, bH, { r: 8, fill: V.alpha(col, on ? 0.16 : 0.04), stroke: V.alpha(col, on ? 0.75 : 0.22), lw: 1.2 });
            V.text(g, LAB[k], X0 + colW / 2, y + bH / 2 + (on ? (k === 3 ? -6 : 0.5) : -5), { size: narrow ? 11 : 12, weight: on ? 600 : 400, color: on ? C.ink : C.muted, align: 'center', base: 'middle', maxW: colW - 8 });
            if (!on) V.text(g, 'bypass', X0 + colW / 2, y + bH / 2 + 10, { size: 11, font: 'mono', color: V.alpha(C.muted, 0.9), align: 'center', base: 'middle' });
          });
          if (mode === 'tcp') {
            const cxp = X0 + (side === 0 ? 18 : colW - 18), cyp = ys[2] + bH / 2;
            V.glow(g, cxp, cyp, 16 + 3 * Math.sin(t * 6), C.c, 0.9);
            V.text(g, 'CPU', cxp, cyp + 0.5, { size: 11, font: 'mono', weight: 700, color: C.ink, align: 'center', base: 'middle' });
          }
          if (mode === 'gb10' && !narrow) V.text(g, 'memcpy', X0 + colW / 2 + (side === 0 ? 40 : -40), ys[1] - 6, { size: 11, font: 'mono', color: C.d, align: 'center' });
        });
        const cxA = Ax + colW - 14, cxB = Bx + 14, yc = (k) => ys[k] + bH / 2;
        const sideA = Ax - 5, sideB = Bx + colW + 5;
        const g01 = ys[0] + bH + (ys[1] - ys[0] - bH) / 2, g12 = ys[1] + bH + (ys[2] - ys[1] - bH) / 2, g23 = ys[2] + bH + (ys[3] - ys[2] - bH) / 2;
        const wy = yc(3) + 8;
        V.line(g, Ax + colW, wy, Bx, wy, { color: V.alpha(C.b, 0.35), lw: 6 });
        V.text(g, narrow ? 'cable' : 'QSFP cable · 200 Gb/s', (Ax + colW + Bx) / 2, wy - 12, { size: 11, font: 'mono', color: C.b, align: 'center' });
        let path;
        if (mode === 'tcp') path = [[cxA, yc(0)], [cxA, yc(1)], [cxA, yc(2)], [cxA, wy], [Ax + colW, wy], [Bx, wy], [cxB, wy], [cxB, yc(2)], [cxB, yc(1)], [cxB, yc(0)]];
        else if (mode === 'gdr') path = [[cxA, yc(0)], [cxA, g01], [sideA, g01], [sideA, g23], [cxA, g23], [cxA, wy], [Ax + colW, wy], [Bx, wy], [cxB, wy], [cxB, g23], [sideB, g23], [sideB, g01], [cxB, g01], [cxB, yc(0)]];
        else path = [[cxA, yc(0)], [cxA, yc(1)], [cxA, g12], [sideA, g12], [sideA, g23], [cxA, g23], [cxA, wy], [Ax + colW, wy], [Bx, wy], [cxB, wy], [cxB, g23], [sideB, g23], [sideB, g12], [cxB, g12], [cxB, yc(1)], [cxB, yc(0)]];
        const segL = []; let tot = 0;
        for (let k = 1; k < path.length; k++) { const L0 = Math.hypot(path[k][0] - path[k - 1][0], path[k][1] - path[k - 1][1]); segL.push(L0); tot += L0; }
        g.strokeStyle = V.alpha(C.a, 0.4); g.lineWidth = 2; g.setLineDash([5, 5]); g.lineDashOffset = -t * 30;
        g.beginPath(); g.moveTo(path[0][0], path[0][1]); for (let k = 1; k < path.length; k++) g.lineTo(path[k][0], path[k][1]); g.stroke(); g.setLineDash([]); g.lineDashOffset = 0;
        const at = (u) => { let d = u * tot; for (let k = 0; k < segL.length; k++) { if (d <= segL[k]) { const f = d / segL[k]; return [V.lerp(path[k][0], path[k + 1][0], f), V.lerp(path[k][1], path[k + 1][1], f)]; } d -= segL[k]; } return path[path.length - 1]; };
        const speed = mode === 'tcp' ? 0.1 : mode === 'gdr' ? 0.22 : 0.16;
        pk.forEach((o) => { const u = (o + t * speed) % 1; const pt = at(u); V.box(g, pt[0] - 6, pt[1] - 4, 12, 8, { r: 2, fill: C.a }); });
        // stats
        const M = MODES[mode];
        let sy = topH - 44;
        sy += para(g, M.stats, w / 2, sy, w - 2 * m, { size: 11.5, weight: 600, color: C.ink, align: 'center', lh: 15 });
        para(g, M.lat, w / 2, sy, w - 2 * m, { size: 11.5, color: C.muted, align: 'center', lh: 15 });
        // encapsulation
        const E = ENC[S.enc];
        const y0 = topH + 8;
        V.line(g, m, y0 - 4, w - m, y0 - 4, { color: V.alpha(C.muted, 0.25), lw: 1 });
        V.text(g, 'How the packet leaves on the cable', m, y0 + 12, { size: 12, weight: 600, color: C.ink });
        const fy = y0 + 22, fh = 30;
        const minW = E.f.map(([lab]) => tw(g, lab, 11, 'mono', 600) + 10);
        const fixed = E.f.reduce((s, [lab, b], k) => s + (b ? Math.max(minW[k], b * 2.2) : 0), 0);
        const payW = Math.max(60, w - 2 * m - fixed);
        let x = m;
        const scan = (t * 0.35) % 1.3;
        E.f.forEach(([lab, b], k) => {
          const ww = b ? Math.max(minW[k], b * 2.2) : payW;
          const col = lab === 'payload' ? C.a : lab.startsWith('BTH') ? C.c : /Eth|IPv4|UDP/.test(lab) ? C.b : C.d;
          const lit = Math.abs(scan - (x - m + ww / 2) / (w - 2 * m)) < 0.08;
          V.box(g, x, fy, ww - 2, fh, { r: 5, fill: V.alpha(col, lit ? 0.5 : 0.24), stroke: V.alpha(col, 0.7), lw: 1 });
          V.text(g, lab === 'payload' ? (ww > 190 ? 'payload (NCCL data)' : 'payload') : lab, x + (ww - 2) / 2, fy + fh / 2 + 0.5, { size: 11, font: 'mono', weight: 600, color: C.ink, align: 'center', base: 'middle', maxW: ww - 6 });
          x += ww;
        });
        para(g, E.note, m, fy + fh + 18, w - 2 * m, { size: 11.5, color: C.muted, lh: 15 });
      });
    })();

    /* ============================================ Fig 5: ConnectX-7 lab + GIDs */
    (function () {
      const fig = root.querySelector('#fig-fabric');
      const ctl = UI.controls(fig);
      const S = { pf: 2, mtu: 9000, v6: false, v4b: false, reboot: false, pin: 'auto' };
      UI.seg(ctl, { label: 'HCAs', options: [{ v: 1, label: '1 PF' }, { v: 2, label: '2 PFs (dual)' }], value: 2, onChange: (v) => { S.pf = v; } });
      UI.seg(ctl, { label: 'MTU', options: [1500, 9000], value: 9000, onChange: (v) => { S.mtu = v; } });
      UI.select(ctl, { label: 'NCCL_IB_GID_INDEX', options: [{ v: 'auto', label: 'auto (DS)' }, { v: '0', label: '0' }, { v: '3', label: '3' }, { v: '4', label: '4' }, { v: '5', label: '5' }, { v: 'empty', label: '"" (empty)' }], value: 'auto', onChange: (v) => { S.pin = v; } });
      UI.toggle(ctl, { label: 'head: + global IPv6', value: false, onChange: (v) => { S.v6 = v; } });
      UI.toggle(ctl, { label: 'head: + 2nd IPv4', value: false, onChange: (v) => { S.v4b = v; } });
      UI.toggle(ctl, { label: 'head reboot: IPv4 comes up last', value: false, onChange: (v) => { S.reboot = v; } });
      function table(node) {
        const own = node === 'head' ? '10.0.22.1' : '10.0.22.2';
        const rows = [], addrs = [{ k: 'll', s: 'fe80::…:5b5' + (node === 'head' ? '5' : '6'), sh: 'fe80::…' }];
        const extra = [];
        if (node === 'head' && S.v6) extra.push({ k: 'v6', s: '2001:db8::1', sh: '2001:db8::1' });
        if (node === 'head' && S.v4b) extra.push({ k: 'v4b', s: '::ffff:10.0.22.101', sh: '::ffff:…101' });
        const main = { k: 'v4', s: '::ffff:' + own, sh: '::ffff:…' + own.split('.').pop() };
        if (node === 'head' && S.reboot) addrs.push(...extra, main); else addrs.push(main, ...extra);
        addrs.forEach((a) => { rows.push(Object.assign({ type: 'v1' }, a)); rows.push(Object.assign({ type: 'v2' }, a)); });
        while (rows.length < 8) rows.push({ k: 'empty', s: '0000:…:0000', sh: '0000:…', type: '—' });
        return rows.slice(0, 8);
      }
      function judge(rows) {
        if (S.pin === 'auto') { const k = rows.findIndex((r) => r.k === 'v4' && r.type === 'v2'); return { k, ok: true, msg: 'auto → index ' + k + ': RoCE v2 with its own IPv4' }; }
        const k = S.pin === 'empty' ? 0 : +S.pin, r = rows[k];
        const pre = S.pin === 'empty' ? '"" = index 0: ' : 'index ' + k + ': ';
        if (r.k === 'empty') return { k, ok: false, msg: pre + 'empty entry → ibv_modify_qp errno 61 (~60 s)' };
        if (r.k === 'll') return { k, ok: false, msg: pre + 'link-local fe80 (' + (r.type === 'v1' ? 'RoCE v1' : 'v2') + ') — not the link IP' };
        if (r.type === 'v1') return { k, ok: false, msg: pre + 'type RoCE v1 — the requested version is 2' };
        if (r.k === 'v6') return { k, ok: false, warn: true, msg: pre + 'global IPv6 — family ≠ AF_INET' };
        if (r.k === 'v4b') return { k, ok: false, warn: true, msg: pre + 'secondary IPv4 — only with NCCL_IB_GID_MATCH_IP' };
        return { k, ok: true, msg: pre + 'RoCE v2 with its own IPv4' };
      }
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const narrow = w < 600;
        const m = 12, midGap = narrow ? 34 : 96, nw = (w - 2 * m - midGap) / 2;
        const topH = narrow ? 340 : 300;
        const NODES = [{ x: m, name: 'spark1 · head', col: C.a, port: 1, f: 'f1' }, { x: w - m - nw, name: 'spark2 · worker', col: C.b, port: 0, f: 'f0' }];
        const rowY = [48, 136], rowH = 84;
        const pfGeo = [];
        NODES.forEach((N, ni) => {
          V.box(g, N.x, 8, nw, 222, { r: 12, fill: V.alpha(N.col, 0.05), stroke: V.alpha(N.col, 0.4) });
          V.text(g, N.name, N.x + 10, 26, { size: 12.5, weight: 600, color: N.col });
          if (!narrow) V.text(g, 'ConnectX-7 · 1 ASIC, 2 ports, 4 PFs', N.x + nw - 10, 26, { size: 11, color: C.muted, align: 'right' });
          pfGeo[ni] = [];
          for (let pt = 0; pt < 2; pt++) {
            const used = pt === N.port;
            const y = rowY[pt];
            V.box(g, N.x + 8, y, nw - 16, rowH, { r: 9, fill: V.alpha(C.muted, used ? 0.08 : 0.03), stroke: V.alpha(used ? N.col : C.muted, used ? 0.55 : 0.2), lw: 1 });
            V.text(g, 'port ' + pt + ' (f' + pt + ')' + (used ? '' : (narrow ? '' : ' · no cable: never list it')), N.x + 16, y + 15, { size: 11, weight: used ? 600 : 400, color: used ? C.ink : C.muted, maxW: nw - 30 });
            const cw = (nw - 16 - 24) / 2;
            for (let pf = 0; pf < 2; pf++) {
              const x = N.x + 16 + pf * (cw + 8), yy = y + 24, ch = 40;
              const sel = used && pf < S.pf;
              const name = pf === 0 ? 'rocep1s0f' + pt : 'roceP2p1s0f' + pt;
              const short = pf === 0 ? 'f' + pt : 'P2·f' + pt;
              V.box(g, x, yy, cw, ch, { r: 7, fill: V.alpha(sel ? N.col : C.muted, sel ? 0.25 : 0.06), stroke: V.alpha(sel ? N.col : C.muted, sel ? 0.9 : 0.25), lw: sel ? 1.6 : 1 });
              const nm = tw(g, name, 11, 'mono', 600) < cw - 6 ? name : short;
              V.text(g, nm, x + cw / 2, yy + (narrow ? ch / 2 + 0.5 : 15), { size: 11, font: 'mono', weight: 600, color: sel ? C.ink : C.muted, align: 'center', base: narrow ? 'middle' : 'alphabetic' });
              if (!narrow) V.text(g, (pf === 0 ? '0000' : '0002') + ':01:00.' + pt + ' · ×4', x + cw / 2, yy + 31, { size: 11, font: 'mono', color: C.muted, align: 'center', maxW: cw - 6 });
              pfGeo[ni][pt * 2 + pf] = { x, y: yy, w: cw, h: ch, sel };
            }
          }
        });
        // crossed cable: head port 1 → worker port 0
        const hx = NODES[0].x + nw, hy = rowY[1] + rowH / 2, wx0 = NODES[1].x, wy = rowY[0] + rowH / 2;
        g.strokeStyle = V.alpha(C.d, 0.4); g.lineWidth = 7; g.beginPath(); g.moveTo(hx, hy); g.bezierCurveTo(hx + midGap * 0.6, hy, wx0 - midGap * 0.6, wy, wx0, wy); g.stroke();
        const cab = (u) => V.bez([hx, hy], [hx + midGap * 0.6, hy], [wx0 - midGap * 0.6, wy], [wx0, wy], u);
        const nStream = S.pf, plen = S.mtu === 9000 ? 16 : 5;
        for (let s = 0; s < nStream; s++) for (let k = 0; k < 6; k++) {
          const u = ((k / 6) + t * 0.18 + s * 0.08) % 1; const p0 = cab(u), p1 = cab(Math.min(1, u + 0.001));
          const an = Math.atan2(p1[1] - p0[1], p1[0] - p0[0]);
          g.save(); g.translate(p0[0], p0[1] + (s - (nStream - 1) / 2) * 3); g.rotate(an); g.fillStyle = s === 0 ? C.a : C.b; g.fillRect(-plen / 2, -2, plen, 4); g.restore();
        }
        V.text(g, narrow ? '200G' : '200 Gb/s', (hx + wx0) / 2, Math.min(hy, wy) - 8, { size: 11, font: 'mono', color: C.d, align: 'center' });
        if (!narrow) V.text(g, 'f1 ↔ f0', (hx + wx0) / 2, Math.max(hy, wy) + 18, { size: 11, font: 'mono', color: C.d, align: 'center' });
        // QSFP cage on the edge of the used port + bracket over the selected PFs
        NODES.forEach((N, ni) => {
          const pt = N.port, y = rowY[pt];
          const cgx = ni === 0 ? N.x + nw - 7 : N.x - 1, cgy = y + rowH / 2 - 12;
          V.box(g, cgx, cgy, 8, 24, { r: 2, fill: V.alpha(C.d, 0.8), stroke: C.d, lw: 1 });
          const G0 = pfGeo[ni][pt * 2], G1 = pfGeo[ni][pt * 2 + S.pf - 1];
          const bx0 = G0.x - 3, bx1 = G1.x + G1.w + 3, byy = G0.y + G0.h + 5;
          g.strokeStyle = V.alpha(N.col, 0.85); g.lineWidth = 1.6;
          g.beginPath(); g.moveTo(bx0, byy - 3); g.lineTo(bx0, byy); g.lineTo(bx1, byy); g.lineTo(bx1, byy - 3); g.stroke();
          if (!narrow) V.text(g, S.pf === 2 ? 'NCCL_IB_HCA: both PFs' : 'NCCL_IB_HCA: one PF', (bx0 + bx1) / 2, byy + 1, { size: 11, font: 'mono', color: N.col, align: 'center', base: 'top', alpha: 0.95 });
        });
        // readouts
        const cap = Math.min(200, 126 * S.pf), rmtu = S.mtu === 9000 ? 4096 : 1024;
        const by = 238;
        const lw = narrow ? w - 2 * m : w * 0.5 - m;
        V.text(g, `raw ceiling ≈ ${cap} Gb/s (${S.pf} × PF of ≈126 raw; port 200)`, m, by + 12, { size: 12, weight: 600, color: C.ink, maxW: lw });
        V.text(g, `Ethernet MTU ${S.mtu} → RoCE active_mtu ${rmtu}`, m, by + 30, { size: 12, font: 'mono', color: S.mtu === 9000 ? C.a : C.c, maxW: lw });
        const meas = [['1500 · 1 PF', 1311, S.mtu === 1500 && S.pf === 1], ['9000 · 1 PF', 1343, S.mtu === 9000 && S.pf === 1], ['9000 · 2 PFs', 1410, S.mtu === 9000 && S.pf === 2]];
        const mx = narrow ? m : w * 0.5 + 8, my = narrow ? by + 40 : by + 2, mw = narrow ? w - 2 * m : w * 0.5 - m - 8;
        const labW = 86;
        V.text(g, 'DS, pair 1 (tok/s E2E):' + (S.mtu === 1500 && S.pf === 2 ? ' 1500 · 2 PFs not measured' : ''), mx, my + 8, { size: 11, color: C.muted, maxW: mw });
        meas.forEach(([lab, v, on], k) => {
          const yy = my + 14 + k * 13;
          V.text(g, lab, mx, yy + 9, { size: 11, font: 'mono', color: on ? C.ink : C.muted });
          const bwid = (mw - labW - 44) * (v - 1200) / 260;
          V.box(g, mx + labW, yy + 1, Math.max(3, bwid), 10, { r: 3, fill: V.alpha(on ? C.a : C.muted, on ? 0.8 : 0.3) });
          V.text(g, String(v), mx + labW + bwid + 5, yy + 10, { size: 11, font: 'mono', weight: on ? 700 : 400, color: on ? C.ink : C.muted });
        });
        // GID tables
        const gy = topH + 4;
        V.line(g, m, gy - 2, w - m, gy - 2, { color: V.alpha(C.muted, 0.25), lw: 1 });
        V.text(g, 'GID table  ·  /sys/class/infiniband/<hca>/ports/1/gids/N', m, gy + 14, { size: 12, weight: 600, color: C.ink, maxW: w - 2 * m });
        const tWid = (w - 2 * m - 12) / 2;
        const res = [];
        [['head', 'rocep1s0f1', '10.0.22.1', C.a], ['worker', 'rocep1s0f0', '10.0.22.2', C.b]].forEach(([node, hca, ip, col], ti) => {
          const rows = table(node), J = judge(rows); res.push(J);
          const x0 = m + ti * (tWid + 12), y0 = gy + 26;
          V.text(g, narrow ? node + ' · ' + ip : node + ' · ' + hca + ' · IPv4 ' + ip, x0, y0 + 10, { size: 11.5, weight: 600, color: col, maxW: tWid });
          const rh = 20;
          const gidX = x0 + 22, typeX = x0 + tWid - (narrow ? 26 : 70);
          rows.forEach((r, k) => {
            const yy = y0 + 18 + k * rh;
            const hit = k === J.k;
            if (hit) V.box(g, x0 - 2, yy - 1, tWid + 2, rh - 2, { r: 5, fill: V.alpha(J.ok ? C.a : J.warn ? C.d : C.red || C.c, 0.28), stroke: J.ok ? C.a : J.warn ? C.d : (C.red || C.c), lw: 1.3 });
            V.text(g, String(k), x0 + 6, yy + rh / 2, { size: 11, font: 'mono', color: C.muted, base: 'middle' });
            const gid = narrow || tw(g, r.s, 11, 'mono') > typeX - gidX - 6 ? r.sh : r.s;
            V.text(g, gid, gidX, yy + rh / 2, { size: 11, font: 'mono', color: r.k === 'empty' ? V.alpha(C.muted, 0.7) : C.ink, base: 'middle' });
            V.text(g, r.type === '—' ? '—' : (narrow ? r.type : 'RoCE ' + r.type), typeX, yy + rh / 2, { size: 11, font: 'mono', color: r.type === 'v2' ? C.a : C.muted, base: 'middle' });
          });
          para(g, (J.ok ? '✓ ' : '✕ ') + J.msg, x0, y0 + 18 + 8 * rh + 14, tWid, { size: 11.5, weight: 600, color: J.ok ? C.a : J.warn ? C.d : (C.red || C.c), lh: 14 });
        });
        if (S.pin !== 'auto' && res[0].ok !== res[1].ok) V.text(g, "one global index doesn't fit both nodes", w / 2, h - 8, { size: 11.5, color: C.c, align: 'center', maxW: w - 2 * m });
      });
    })();

    /* ============================================ Fig 6: distributed boot (stepper) */
    (function () {
      const fig = root.querySelector('#fig-boot');
      const STEPS = [
        { t: 'SSH and containers', d: 'The launcher on the head logs into the worker over passwordless SSH (<code>BatchMode=yes</code>) and brings up both containers with host networking and IPC, <code>/dev/infiniband</code>, and unlimited <code>memlock</code>.', gov: '--network host · --ipc host · --device /dev/infiniband · --ulimit memlock=-1', fail: 'no /dev/infiniband: with NCCL_NET=IB (GLM, DS) NCCL init fails; with no NCCL_NET pinned (Qwen) it falls back to NET/Socket · limited memlock: ibv_reg_mr fails', lanes: [], ssh: true },
        { t: 'The worker comes up first and waits', d: "Rank 1 (<code>--headless</code>) starts looking for the TCPStore at <code>MASTER_ADDR:MASTER_PORT</code> and keeps retrying: the head doesn't exist yet.", gov: '--nnodes 2 --node-rank 1 --headless --master-addr 10.0.0.1', fail: 'head restarting in the middle of loading → the TCPStore handshake drops (DS #38)', lanes: ['tcp'], wait: true },
        { t: 'The head hosts the TCPStore', d: 'Rank 0 opens the TCPStore; both register (world size 2). This is the <b>rendezvous</b>.', gov: '--master-port 29521 (GLM) · 50000 (Qwen) · 25000 (DS)', fail: "port in use or different between the nodes → the ranks don't find each other", lanes: ['tcp'] },
        { t: 'Gloo group (CPU)', d: 'torch creates a Gloo group for barriers and metadata on the CPU, over TCP, on the chosen interface.', gov: 'GLOO_SOCKET_IFNAME = enp1s0f1np1 (head) · enp1s0f0np0 (worker)', fail: 'lo or the wrong NIC → bind on 127.0.0.1, hang or slow init', lanes: ['tcp'] },
        { t: 'NCCL bootstrap', d: 'Rank 0 generates the NCCL <i>unique id</i>; the ranks exchange addresses over TCP sockets — still no RDMA.', gov: 'NCCL_SOCKET_IFNAME = real name of the CX7 interface (not the alias)', fail: 'only the 10.0.0.x alias → ncclCommInitRank hangs', lanes: ['tcp'] },
        { t: 'Verbs: HCA, GID, and QPs', d: "Each rank opens the HCA, picks the GID, registers buffers, and takes the queue pairs from RESET to RTS; the transition to RTR uses the peer's GID.", gov: 'NCCL_IB_HCA · NCCL_IB_GID_INDEX (or auto) · NCCL_IB_ROCE_VERSION_NUM=2', fail: 'empty GID → errno 61 in ~60 s; HCA wired to another cluster → hang', lanes: ['rdma'], qp: true },
        { t: 'Loading, profiling, and graphs', d: 'Weights load on both nodes; the KV cache is sized by the rank with the <i>least</i> memory; CUDA graphs are captured — with the same arguments and patches on both.', gov: 'same engine args, overlays, and capture sizes on both ranks', fail: 'ranks with different graphs → hang after PIECEWISE (GLM)', lanes: ['rdma'], load: true },
        { t: 'Serving: one step', d: 'The scheduler sends the plan (local SHM + ZMQ to the worker, RTT ≈0.33 ms); the forward pass fires ~100 all-reduces over RDMA; the tokens come back.', gov: 'VLLM_HOST_IP per node · ≈106–108 collectives/step', fail: '“NET/Socket” in the log → RDMA is not in use', lanes: ['zmq', 'rdma'], serve: true },
      ];
      const st = UI.stepper(fig, STEPS.map((s) => ({ t: s.t, d: s.d })), { autoplay: 5600 });
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const i = st.i, e = st.elapsed(), S = STEPS[i];
        const narrow = w < 560;
        const m = 12, nodeW = narrow ? (w - 2 * m) * 0.3 : Math.min(230, w * 0.27);
        const Hx = m, Wx = w - m - nodeW, ny0 = 26, ny1 = 290;
        const laneY = { tcp: 98, zmq: 168, rdma: 238 };
        const LN = { tcp: [C.d, narrow ? 'TCP · bootstrap' : 'TCP · rendezvous, Gloo, NCCL bootstrap'], zmq: [C.b, narrow ? 'ZMQ · control' : 'ZMQ · scheduler plan'], rdma: [C.a, narrow ? 'RDMA · data' : 'RDMA · NCCL all-reduces'] };
        // nodes
        [[Hx, 'spark1 · head', C.a], [Wx, 'spark2 · worker', C.b]].forEach(([x, name, col]) => {
          V.box(g, x, ny0, nodeW, ny1 - ny0, { r: 12, fill: V.alpha(col, 0.05), stroke: V.alpha(col, 0.45) });
          V.text(g, name, x + nodeW / 2, ny0 - 8, { size: 12, weight: 600, color: col, align: 'center' });
        });
        const pbox = (x, y, lab, col, on, sub) => {
          const bw2 = nodeW - 16;
          V.box(g, x + 8, y, bw2, sub ? 40 : 30, { r: 7, fill: V.alpha(col, on ? 0.22 : 0.04), stroke: V.alpha(col, on ? 0.8 : 0.2), lw: 1.2 });
          V.text(g, lab, x + 8 + bw2 / 2, y + (sub ? 14 : 15), { size: 11, weight: 600, color: on ? C.ink : C.muted, align: 'center', base: sub ? 'alphabetic' : 'middle', maxW: bw2 - 6 });
          if (sub) V.text(g, sub, x + 8 + bw2 / 2, y + 30, { size: 11, font: 'mono', color: on ? col : C.muted, align: 'center', maxW: bw2 - 6 });
        };
        const headOn = i >= 2, wOn = i >= 1;
        pbox(Hx, 38, 'API server :8888', C.d, headOn && i >= 6);
        pbox(Hx, laneY.tcp - 20, narrow ? 'TCPStore' : 'rank 0 · TCPStore', C.d, headOn, i >= 2 ? (narrow ? ':PORT' : ':MASTER_PORT') : '');
        pbox(Hx, laneY.zmq - 15, narrow ? 'EngineCore' : 'EngineCore · scheduler', C.b, headOn && i >= 6);
        pbox(Hx, laneY.rdma - 20, narrow ? 'GPU rank 0' : 'worker rank 0 · GPU', C.a, headOn, i >= 5 ? (narrow ? 'rocep1s0f1' : 'HCA rocep1s0f1') : '');
        pbox(Wx, 38, narrow ? 'no API' : 'no API · no scheduler', C.muted, wOn);
        pbox(Wx, laneY.tcp - 20, narrow ? 'client' : 'TCPStore client', C.d, wOn, i === 1 ? 'waiting…' : '');
        pbox(Wx, laneY.zmq - 15, narrow ? 'gets plan' : 'receives the plan (ZMQ)', C.b, wOn && i >= 7);
        pbox(Wx, laneY.rdma - 20, narrow ? 'GPU rank 1' : 'worker rank 1 · GPU', C.a, wOn, i >= 5 ? (narrow ? 'rocep1s0f0' : 'HCA rocep1s0f0') : '');
        // lanes
        const lx0 = Hx + nodeW, lx1 = Wx;
        Object.keys(laneY).forEach((k) => {
          const on = S.lanes.includes(k), [col, lab] = LN[k], y = laneY[k];
          V.line(g, lx0, y, lx1, y, { color: V.alpha(col, on ? 0.5 : 0.14), lw: on ? 6 : 4 });
          V.text(g, lab, (lx0 + lx1) / 2, y - 12, { size: 11, font: 'mono', color: on ? col : V.alpha(C.muted, 0.8), align: 'center', maxW: lx1 - lx0 - 6 });
          if (on) {
            const nP = k === 'rdma' && S.serve ? 10 : 3;
            for (let q = 0; q < nP; q++) {
              const u = (q / nP + t * (k === 'rdma' && S.serve ? 0.6 : 0.35)) % 1;
              const back = S.wait ? (u > 0.5) : (q % 2 === 1);
              const x = back ? V.lerp(lx1, lx0, u) : V.lerp(lx0, lx1, u);
              if (S.wait && u > 0.5) continue;
              V.box(g, x - 5, y - 4, 10, 8, { r: 2, fill: col });
            }
            if (S.wait) V.text(g, '?', V.lerp(lx1, lx0, 0.45 + 0.05 * Math.sin(t * 4)), y + 16, { size: 13, weight: 700, color: C.c, align: 'center' });
          }
        });
        if (S.ssh) {
          const yS = 64 + 6 * Math.sin(t * 2);
          V.arrow(g, lx0 + 6, yS, lx1 - 6, yS, { color: C.d, lw: 1.8, dash: [5, 4] });
          V.text(g, 'ssh · docker run', (lx0 + lx1) / 2, yS - 8, { size: 11, font: 'mono', color: C.d, align: 'center' });
        }
        if (S.qp) {
          const sts = ['RESET', 'INIT', 'RTR', 'RTS'];
          const k = Math.min(3, Math.floor(e / 0.9));
          const cxm = (lx0 + lx1) / 2;
          if (narrow) {
            chip(g, cxm, laneY.rdma + 12, 'QP: ' + sts[k], k === 2 ? C.c : C.a, { size: 11, a: 0.3, align: 'center' });
            if (k >= 2) V.text(g, "RTR uses the peer's GID", cxm, laneY.rdma + 40, { size: 11, color: C.c, align: 'center', maxW: lx1 - lx0 - 4 });
          } else {
            const tot = sts.reduce((s0, s, q) => s0 + tw(g, s, 11, 'mono', 600) + 14 + (q ? 6 : 0), 0);
            let x = cxm - tot / 2;
            sts.forEach((s, q) => { const wd = chip(g, x, laneY.rdma + 12, s, q <= k ? (q === 2 ? C.c : C.a) : C.muted, { size: 11, a: q <= k ? 0.3 : 0.06 }); x += wd + 6; });
            if (k >= 2) V.text(g, "RTR receives the peer's GID", cxm, laneY.rdma + 48, { size: 11, color: C.c, align: 'center' });
          }
        }
        if (S.load) {
          const f = V.clamp(e / 3.5, 0, 1);
          [Hx, Wx].forEach((x, s) => {
            V.box(g, x + 10, 280, nodeW - 20, 8, { r: 4, fill: V.alpha(C.muted, 0.2) });
            V.box(g, x + 10, 280, (nodeW - 20) * f * (s ? 0.93 : 1), 8, { r: 4, fill: C.a });
          });
          V.text(g, narrow ? 'weights · KV · graphs' : 'weights · KV profile · CUDA graphs', (lx0 + lx1) / 2, 286, { size: 11, color: C.muted, align: 'center', maxW: lx1 - lx0 });
        }
        if (S.serve) {
          const k = Math.floor(t * 20) % 106 + 1;
          V.text(g, 'collective ' + k + '/≈106', (lx0 + lx1) / 2, laneY.rdma + 22, { size: 11, font: 'mono', color: C.a, align: 'center' });
        }
        // bottom panel
        const py = ny1 + 12;
        V.box(g, m, py, w - 2 * m, h - py - 8, { r: 10, fill: V.alpha(C.muted, 0.06), stroke: V.alpha(C.muted, 0.2) });
        let yy = py + 18;
        yy += para(g, 'governed by: ' + S.gov, m + 12, yy, w - 2 * m - 24, { size: 11.5, font: 'mono', color: C.a, lh: 15 });
        para(g, 'if wrong: ' + S.fail, m + 12, yy + 4, w - 2 * m - 24, { size: 11.5, color: C.c, lh: 15 });
      });
    })();

    /* ============================================ Fig 7: channels and protocols */
    (function () {
      const fig = root.querySelector('#fig-proto');
      const ctl = UI.controls(fig);
      const S = { proto: 'LL', ch: 2, msg: 20e3 };
      UI.seg(ctl, { label: 'Protocol', options: ['Simple', 'LL', 'LL128'], value: 'LL', onChange: (v) => { S.proto = v; } });
      UI.slider(ctl, { label: 'Channels', min: 1, max: 8, step: 1, value: 2, fmt: (v) => String(v), onInput: (v) => { S.ch = v; } });
      UI.seg(ctl, { label: 'Message', options: [{ v: 20e3, label: '20 KB · decode' }, { v: 8388608, label: '8 MiB · prefill' }], value: 20e3, onChange: (v) => { S.msg = v; } });
      const PR = {
        Simple: { eff: 1.0, a: 18, pw: 54, gap: 22, flag: 0, lat: 'higher (fence + signal)' },
        LL: { eff: 0.5, a: 6, pw: 10, gap: 9, flag: 0.5, lat: 'minimal (flag per word)' },
        LL128: { eff: 120 / 128, a: 9, pw: 28, gap: 8, flag: 8 / 128, lat: 'low (flag per 128 B line)' },
      };
      const betaCh = 4, betaLink = 11;
      const timeOf = (pn, n, ch) => { const P = PR[pn]; const bw = Math.min(ch * betaCh, betaLink) * P.eff; return P.a + n / (bw * 1e9) * 1e6; };
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const narrow = w < 560;
        const m = 12, rw = narrow ? 56 : 76;
        const P = PR[S.proto];
        const top = 34, laneArea = 190;
        const laneH = Math.min(38, laneArea / S.ch);
        const x0 = m + rw + 8, x1 = w - m - rw - 8;
        V.box(g, m, top, rw, laneArea, { r: 10, fill: V.alpha(C.a, 0.1), stroke: V.alpha(C.a, 0.6) });
        V.box(g, w - m - rw, top, rw, laneArea, { r: 10, fill: V.alpha(C.b, 0.1), stroke: V.alpha(C.b, 0.6) });
        V.text(g, 'rank 0', m + rw / 2, top + laneArea / 2, { size: 12, weight: 600, color: C.a, align: 'center', base: 'middle' });
        V.text(g, 'rank 1', w - m - rw / 2, top + laneArea / 2, { size: 12, weight: 600, color: C.b, align: 'center', base: 'middle' });
        V.text(g, narrow ? `${S.ch} channel(s) · ${S.proto} · ${fmtB(S.msg / S.ch)} per channel` : `${S.ch} channel(s) · protocol ${S.proto} · ${fmtB(S.msg)} split into ${S.ch} slices of ${fmtB(S.msg / S.ch)}`, w / 2, 20, { size: 11.5, color: C.muted, align: 'center', maxW: w - 2 * m });
        const pw = narrow ? P.pw * 0.65 : P.pw, gp = narrow ? P.gap * 0.7 : P.gap;
        const period = pw + gp;
        for (let c = 0; c < S.ch; c++) {
          const y = top + (laneArea - laneH * S.ch) / 2 + c * laneH + 3, lh = laneH - 6;
          V.box(g, x0, y, x1 - x0, lh, { r: 5, fill: V.alpha(C.muted, 0.08) });
          const labA = laneH > 18 ? (narrow ? 26 : 56) : 0;
          if (labA) V.text(g, narrow ? 'c' + c : 'ch ' + c, x0 + 5, y + lh / 2 + 0.5, { size: 11, font: 'mono', color: C.muted, base: 'middle' });
          const off = (t * 70) % period;
          g.save(); g.beginPath(); g.rect(x0 + labA, y, x1 - x0 - labA, lh); g.clip();
          for (let x = x0 - period + off; x < x1; x += period) {
            const py = y + 3, ph = lh - 6;
            if (S.proto === 'Simple') {
              V.box(g, x, py, pw, ph, { r: 3, fill: V.alpha(C.a, 0.75) });
              V.line(g, x + pw + gp / 2, y + 1, x + pw + gp / 2, y + lh - 1, { color: C.d, lw: 2 });
            } else {
              const fw = Math.max(1.5, pw * P.flag);
              V.box(g, x, py, pw - fw, ph, { r: 2, fill: V.alpha(C.a, 0.75) });
              V.box(g, x + pw - fw, py, fw, ph, { r: 1, fill: V.alpha(C.f, 0.8) });
            }
          }
          g.restore();
        }
        // legend
        const ly = top + laneArea + 18;
        let lx = m;
        lx += chip(g, lx, ly - 13, 'data', C.a, { size: 11, a: 0.6 }) + 8;
        lx += chip(g, lx, ly - 13, S.proto === 'Simple' ? 'fence/signal' : 'flag', S.proto === 'Simple' ? C.d : C.f, { size: 11, a: 0.6 }) + 12;
        const legTxt = `data efficiency ${Math.round(P.eff * 100)}% · latency: ${P.lat}`;
        const legH = tw(g, legTxt, 11.5) > w - lx - m ? para(g, legTxt, lx, ly + 1, w - lx - m, { size: 11.5, color: C.ink, lh: 14 }) - 14 : (V.text(g, legTxt, lx, ly + 1, { size: 11.5, color: C.ink }), 0);
        // time bars
        const by = ly + 22 + legH;
        V.text(g, `Estimated time for ${fmtB(S.msg)} (toy model)`, m, by, { size: 12, weight: 600, color: C.ink });
        const names = ['LL', 'LL128', 'Simple'];
        const Ts = names.map((nm) => timeOf(nm, S.msg, S.ch));
        const Tmax = Math.max(...Ts);
        const labW = 58, bwMax = w - 2 * m - labW - 70;
        names.forEach((nm, k) => {
          const y = by + 12 + k * 22, on = nm === S.proto, best = Ts[k] === Math.min(...Ts);
          V.text(g, nm, m, y + 11, { size: 11.5, font: 'mono', weight: on ? 700 : 400, color: on ? C.ink : C.muted });
          const bwid = Math.max(3, bwMax * Ts[k] / Tmax);
          V.box(g, m + labW, y + 2, bwid, 13, { r: 4, fill: V.alpha(best ? C.a : C.b, on ? 0.85 : 0.35) });
          V.text(g, fmtUs(Ts[k]) + (best ? ' ★' : ''), m + labW + bwid + 6, y + 12, { size: 11.5, font: 'mono', color: on ? C.ink : C.muted });
        });
        const buf = S.ch * 2 * 4;
        para(g, `Simple buffers (4 MiB per channel and direction, default): ${S.ch} × 2 × 4 MiB = ${buf} MiB per peer — pinned on the host when there is no GDR.`, m, by + 12 + 3 * 22 + 12, w - 2 * m, { size: 11.5, color: C.muted, lh: 15 });
      });
    })();

    /* ============================================ Fig 8: shared memory ring buffer */
    (function () {
      const fig = root.querySelector('#fig-shm');
      const ctl = UI.controls(fig);
      const S = { busy: 2, gap: 60, lost: false, recheck: true };
      const K = 8;
      const sim = {};
      function reset() {
        Object.assign(sim, { t: 0, next: 10, seq: 0, reads: 0, wi: 0, ri: 0, mode: 'spin', spinUntil: S.busy, sleepSince: 0, readUntil: 0, notes: [], cpu: 0, lostCool: 0, fast: false, recheckAt: -1, slots: Array.from({ length: K }, () => ({ s: 'free', n: 0, tw: 0 })) });
      }
      reset();
      UI.seg(ctl, { label: 'Spin window', options: [{ v: 2, label: '2 ms (DS)' }, { v: 16, label: '16 ms (GLM)' }, { v: 1000, label: '1 s (stock)' }], value: 2, onChange: (v) => { S.busy = v; if (sim.mode === 'spin') sim.spinUntil = sim.t + v; } });
      UI.slider(ctl, { label: 'Interval', min: 20, max: 150, step: 5, value: 60, fmt: (v) => v + ' ms', onInput: (v) => { S.gap = v; } });
      UI.toggle(ctl, { label: 'lose notification (#117)', value: false, onChange: (v) => { S.lost = v; sim.lostCool = 0; } });
      UI.toggle(ctl, { label: '5 s recheck (backport #45224)', value: true, onChange: (v) => { S.recheck = v; } });
      UI.button(ctl, '↺ restart', () => reset());
      function stepSim(dts) {
        const dtSub = 0.5; let left = dts;
        while (left > 0) {
          const d = Math.min(dtSub, left); left -= d; sim.t += d;
          // writer: only writes the next plan after the previous one was consumed (the step needs the result)
          if (sim.t >= sim.next && sim.seq - sim.reads < 1) {
            const sl = sim.slots[sim.wi];
            if (sl.s !== 'written') {
              sl.s = 'written'; sl.n = ++sim.seq; sl.tw = sim.t; sim.wi = (sim.wi + 1) % K;
              let lost = false;
              if (S.lost && sim.mode === 'sleep' && sim.lostCool <= 0) { lost = true; sim.lostCool = 3; } else if (sim.lostCool > 0) sim.lostCool--;
              sim.notes.push({ t0: sim.t, lost, done: false });
              sim.next = sim.t + S.gap;
            }
          }
          // notifications
          sim.notes.forEach((nt) => { if (!nt.done && sim.t >= nt.t0 + 0.3) { nt.done = true; if (!nt.lost && sim.mode === 'sleep') { sim.mode = 'spin'; sim.spinUntil = sim.t + 0.15 + S.busy; } } });
          // reader
          if (sim.mode === 'reading' && sim.t >= sim.readUntil) { sim.slots[sim.ri].s = 'read'; sim.ri = (sim.ri + 1) % K; sim.reads++; sim.mode = 'spin'; sim.spinUntil = sim.t + S.busy; }
          if (sim.mode === 'spin') {
            if (sim.slots[sim.ri].s === 'written') { sim.mode = 'reading'; sim.readUntil = sim.t + 1; }
            else if (sim.t >= sim.spinUntil) { sim.mode = 'sleep'; sim.sleepSince = sim.t; }
          } else if (sim.mode === 'sleep' && S.recheck && sim.t - sim.sleepSince >= 5000) { sim.mode = 'spin'; sim.spinUntil = sim.t + S.busy; sim.recheckAt = sim.t; }
          const busyNow = sim.mode !== 'sleep' ? 1 : 0;
          sim.cpu += (busyNow - sim.cpu) * (1 - Math.exp(-d / 150));
        }
        sim.notes = sim.notes.filter((nt) => sim.t - nt.t0 < 60);
      }
      V.fig(fig, (g, w, h, t, dt) => {
        const C = V.C();
        const stuck = sim.mode === 'sleep' && sim.slots[sim.ri].s === 'written';
        sim.fast = stuck;
        if (!LA.paused) stepSim(Math.min(dt, 0.05) * 1000 * (stuck ? 2.5 : 0.04));
        const narrow = w < 560;
        const m = 12, boxW = narrow ? 92 : 160, boxH = 92, by = 64;
        const Rr = narrow ? 56 : 84, sz = narrow ? 20 : 28, cx = w / 2, cy = 54 + Rr + sz / 2 + 2;
        // boxes
        V.box(g, m, by, boxW, boxH, { r: 12, fill: V.alpha(C.d, 0.1), stroke: V.alpha(C.d, 0.6) });
        V.text(g, narrow ? 'writer' : 'EngineCore · writer', m + boxW / 2, by + 20, { size: 12, weight: 600, color: C.d, align: 'center', maxW: boxW - 8 });
        const blocked = sim.seq - sim.reads >= 1 && sim.t >= sim.next;
        V.text(g, blocked ? 'waiting for the step' : 'writes the plan', m + boxW / 2, by + 44, { size: 11, color: blocked ? C.c : C.muted, align: 'center', maxW: boxW - 8 });
        V.text(g, '#' + sim.seq, m + boxW / 2, by + 70, { size: 16, font: 'mono', weight: 700, color: C.ink, align: 'center' });
        const rx = w - m - boxW;
        const rcol = sim.mode === 'sleep' ? C.b : C.c;
        V.box(g, rx, by, boxW, boxH, { r: 12, fill: V.alpha(rcol, 0.1), stroke: V.alpha(rcol, 0.7) });
        V.text(g, narrow ? 'reader' : 'worker · reader', rx + boxW / 2, by + 20, { size: 12, weight: 600, color: rcol, align: 'center', maxW: boxW - 8 });
        const ix = rx + boxW / 2, iy = by + 56;
        if (sim.mode === 'sleep') {
          V.text(g, 'sleeping', ix, by + 44, { size: 11, color: C.b, align: 'center' });
          for (let q = 0; q < 3; q++) V.text(g, 'z', ix - 8 + q * 9, iy + 20 - ((t * 20 + q * 8) % 22), { size: 11 + q * 2, weight: 700, color: V.alpha(C.b, 0.8), align: 'center' });
        } else {
          V.text(g, sim.mode === 'reading' ? 'reading the slot' : 'spinning', ix, by + 44, { size: 11, color: C.c, align: 'center' });
          g.strokeStyle = C.c; g.lineWidth = 3; g.beginPath(); g.arc(ix, iy + 10, 11, t * 9, t * 9 + 4.2); g.stroke();
          const rem = V.clamp((sim.spinUntil - sim.t) / S.busy, 0, 1);
          V.box(g, rx + 12, by + boxH - 10, boxW - 24, 5, { r: 3, fill: V.alpha(C.muted, 0.2) });
          V.box(g, rx + 12, by + boxH - 10, (boxW - 24) * rem, 5, { r: 3, fill: C.c });
        }
        // ring buffer
        V.text(g, 'ring buffer in /dev/shm', cx, cy + Rr + sz / 2 + 16, { size: 11, color: C.muted, align: 'center' });
        for (let k = 0; k < K; k++) {
          const a = -Math.PI / 2 + (k / K) * Math.PI * 2;
          const x = cx + Math.cos(a) * Rr - sz / 2, y = cy + Math.sin(a) * Rr - sz / 2;
          const sl = sim.slots[k];
          const col = sl.s === 'written' ? C.d : sl.s === 'read' ? C.a : C.muted;
          V.box(g, x, y, sz, sz, { r: 5, fill: V.alpha(col, sl.s === 'free' ? 0.06 : sl.s === 'written' ? 0.55 : 0.2), stroke: V.alpha(col, 0.7), lw: 1 });
          if (sl.n && sz >= 24) V.text(g, String(sl.n % 100), x + sz / 2, y + sz / 2 + 0.5, { size: 11, font: 'mono', weight: 600, color: C.ink, align: 'center', base: 'middle' });
          if (stuck && k === sim.ri) V.glow(g, x + sz / 2, y + sz / 2, sz, C.d, 0.7 + 0.3 * Math.sin(t * 6));
        }
        const ptr = (idx, col, lab, inner) => {
          const a = -Math.PI / 2 + (idx / K) * Math.PI * 2;
          const r1 = inner ? Rr - sz * 0.9 : Rr + sz * 0.9, r0 = inner ? Rr - sz * 1.9 : Rr + sz * 1.7;
          V.arrow(g, cx + Math.cos(a) * r0, cy + Math.sin(a) * r0, cx + Math.cos(a) * r1, cy + Math.sin(a) * r1, { color: col, lw: 2, head: 6 });
          if (!narrow) V.text(g, lab, cx + Math.cos(a) * (r0 + (inner ? -8 : 10)), cy + Math.sin(a) * (r0 + (inner ? -8 : 10)), { size: 11, font: 'mono', color: col, align: 'center', base: 'middle' });
        };
        ptr(sim.wi, C.d, 'W', false);
        ptr(sim.ri, C.c, 'R', true);
        // notifications (PUB/SUB socket)
        const n0 = [m + boxW, by + 10], n1 = [rx, by + 10], nc = [cx, 4];
        V.text(g, narrow ? 'notify (PUB/SUB)' : 'notification socket (PUB/SUB) →', m, 16, { size: 11, color: V.alpha(C.muted, 0.9) });
        sim.notes.forEach((nt) => {
          const u = V.clamp((sim.t - nt.t0) / (stuck || sim.fast ? 60 : 18), 0, 1);
          if (u >= 1) return;
          if (nt.lost && u > 0.45) {
            const p = quad(n0, nc, n1, 0.45);
            const fall = (u - 0.45) * 80;
            V.text(g, '✕', p[0], p[1] + fall, { size: 16, weight: 700, color: C.red || C.c, align: 'center', alpha: 1 - u });
            return;
          }
          const p = quad(n0, nc, n1, u);
          V.box(g, p[0] - 8, p[1] - 5, 16, 10, { r: 2, fill: nt.lost ? (C.red || C.c) : C.b });
        });
        // bottom panel
        const py = cy + Rr + sz / 2 + 34;
        V.line(g, m, py - 6, w - m, py - 6, { color: V.alpha(C.muted, 0.25), lw: 1 });
        const frac = Math.min(1, Math.min(S.busy, S.gap) / S.gap);
        V.text(g, 'spinning P-cores', m, py + 12, { size: 11.5, color: C.muted });
        for (let q = 0; q < 4; q++) {
          const x = m + q * 30, y = py + 20;
          const heat = V.clamp(sim.cpu * (q < 3 ? 1 : 0.8), 0, 1);
          V.box(g, x, y, 24, 24, { r: 5, fill: V.alpha(heat > 0.5 ? C.red || C.c : C.a, 0.15 + 0.6 * heat), stroke: V.alpha(C.muted, 0.4), lw: 1 });
          if (heat > 0.6) V.glow(g, x + 12, y + 12, 14, C.red || C.c, heat * 0.7);
        }
        const tx = m + 136, twid = w - tx - m;
        let yy = py + 12;
        yy += para(g, `reader CPU: ${Math.round(sim.cpu * 100)}% (measured in the simulation) · model: window ÷ interval = ${Math.round(frac * 100)}%`, tx, yy, twid, { size: 11.5, color: C.ink, lh: 15 });
        yy += para(g, S.busy >= S.gap ? `window ${S.busy >= 1000 ? '1 s' : S.busy + ' ms'} ≥ interval ${S.gap} ms → the reader never sleeps (issue #79)` : `window ${S.busy} ms < interval ${S.gap} ms → spins only right after each plan and sleeps the rest`, tx, yy + 2, twid, { size: 11.5, color: S.busy >= S.gap ? C.c : C.a, lh: 15 });
        let sy = Math.max(py + 58, yy + 12);
        if (stuck) {
          const since = (sim.t - sim.slots[sim.ri].tw) / 1000;
          const msg = S.recheck ? `⏩ lost wakeup: plan #${sim.slots[sim.ri].n} has been in the slot for ${br(since, 1)} s and the reader is asleep — recheck in ${br(Math.max(0, 5 - (sim.t - sim.sleepSince) / 1000), 1)} s` : `⏩ lost wakeup: plan #${sim.slots[sim.ri].n} has been in the slot for ${br(since, 1)} s — with no recheck, the reader sleeps forever (hang)`;
          para(g, msg, m, sy, w - 2 * m, { size: 12, weight: 600, color: S.recheck ? C.d : (C.red || C.c), lh: 16 });
        } else if (sim.recheckAt >= 0 && sim.t - sim.recheckAt < 400) {
          para(g, '✓ 5 s recheck: the reader reread the authoritative flag, found the plan, and moved on', m, sy, w - 2 * m, { size: 12, weight: 600, color: C.a, lh: 16 });
        } else {
          para(g, S.lost ? (S.busy >= S.gap ? "notification loss on — but with window ≥ interval the reader never sleeps, so it hasn't hung yet" : 'notification loss on — wait for the reader to fall asleep…') : 'slot flags: “written” (amber) and “read” (green); the writer only reuses a slot after everyone has read it', m, sy, w - 2 * m, { size: 11.5, color: C.muted, lh: 15 });
        }
        V.text(g, `t(sim) = ${br(sim.t / 1000, 2)} s${stuck ? '  ⏩ sped up' : '  (slow motion: 25×)'}`, w - m, h - 10, { size: 11, font: 'mono', color: C.muted, align: 'right' });
      });
    })();

    /* ============================================ Fig 9: watchdog and flight recorder */
    (function () {
      const fig = root.querySelector('#fig-watchdog');
      const ctl = UI.controls(fig);
      const S = { jit: 420, rpc: 1800, fr: true, warm: false };
      const sim = { phase: 'run', seq: 20, acc: 0, e: 0, hold: 0, outcome: null, hangSeq: 0 };
      const reset = () => Object.assign(sim, { phase: 'run', seq: 20, acc: 0, e: 0, hold: 0, outcome: null, hangSeq: 0 });
      UI.button(ctl, '⚡ trigger JIT on rank 1', () => { if (sim.phase !== 'run') reset(); sim.phase = S.warm ? 'hit' : 'hang'; sim.e = 0; sim.hold = 0; sim.hangSeq = sim.seq + 1; }, { primary: true });
      UI.slider(ctl, { label: 'JIT duration', min: 30, max: 1200, step: 10, value: S.jit, fmt: (v) => v + ' s', onInput: (v) => { S.jit = v; } });
      UI.seg(ctl, { label: 'RPC timeout', options: [{ v: 300, label: '300 s (stock)' }, { v: 1800, label: '1800 s (GLM, DS)' }], value: 1800, onChange: (v) => { S.rpc = v; } });
      UI.toggle(ctl, { label: 'flight recorder (TORCH_FR_*)', value: true, onChange: (v) => { S.fr = v; } });
      UI.toggle(ctl, { label: 'persisted JIT caches + warmup', value: false, onChange: (v) => { S.warm = v; } });
      UI.button(ctl, '↺ restart', reset);
      V.fig(fig, (g, w, h, t, dt) => {
        const C = V.C();
        const d = LA.paused ? 0 : Math.min(dt, 0.05);
        const narrow = w < 560;
        // simulation
        if (sim.phase === 'run' || sim.phase === 'hit' || sim.phase === 'recovered') {
          sim.acc += d;
          if (sim.acc > 0.5) { sim.acc = 0; sim.seq++; }
          if (sim.phase !== 'run') { sim.hold += d; if (sim.hold > 3) { sim.phase = 'run'; sim.hold = 0; } }
        } else if (sim.phase === 'hang') {
          sim.e += d * 60;
          const J = S.jit;
          if (S.rpc < 600 && sim.e >= S.rpc && J > S.rpc) { sim.e = S.rpc; sim.phase = 'dead'; sim.outcome = 'rpc'; }
          else if (sim.e >= 600 && J > 600 && S.rpc >= 600) { sim.e = 600; sim.phase = 'dead'; sim.outcome = 'wd'; }
          else if (sim.e >= J) { sim.e = J; sim.phase = 'recovered'; sim.hold = 0; sim.seq++; }
        }
        const m = 12, labW = narrow ? 52 : 70;
        const x0 = m + labW, x1 = w - m, xNow = x0 + (x1 - x0) * 0.62, sp = narrow ? 40 : 52;
        const lanes = [[48, 'rank 0', C.a], [104, 'rank 1', C.b]];
        const hanging = sim.phase === 'hang' || sim.phase === 'dead';
        const frac = sim.phase === 'run' ? sim.acc / 0.5 : 0;
        lanes.forEach(([y, lab, col], r) => {
          V.text(g, lab, m, y + 4, { size: 12, weight: 600, color: col });
          V.line(g, x0, y, x1, y, { color: V.alpha(C.muted, 0.25), lw: 1 });
          for (let k = -12; k <= 0; k++) {
            const sq = sim.seq + k; if (hanging && sq >= sim.hangSeq) continue;
            const x = xNow + (k - frac) * sp;
            if (x < x0 - 10) continue;
            V.box(g, x - sp * 0.4, y - 12, sp * 0.8, 24, { r: 5, fill: V.alpha(C.a, 0.22), stroke: V.alpha(C.a, 0.5), lw: 1 });
            V.text(g, '#' + sq, x, y + 0.5, { size: 11, font: 'mono', color: C.ink, align: 'center', base: 'middle' });
          }
          if (hanging) {
            const xs = xNow + 0.5 * sp, xe = x1 - 4;
            if (r === 0) {
              V.box(g, xs, y - 12, xe - xs, 24, { r: 5, fill: V.alpha(C.d, 0.18 + 0.1 * Math.sin(t * 5)), stroke: C.d, lw: 1.3 });
              V.text(g, narrow ? `#${sim.hangSeq} waiting…` : `#${sim.hangSeq} all_reduce · waiting for the peer`, xs + 6, y + 0.5, { size: 11, font: 'mono', color: C.ink, base: 'middle', maxW: xe - xs - 10 });
            } else {
              V.box(g, xs, y - 12, xe - xs, 24, { r: 5, fill: V.alpha(C.c, 0.2), stroke: C.c, lw: 1.3 });
              V.text(g, narrow ? 'JIT compiling…' : 'compiling kernel (JIT)… #' + sim.hangSeq + ' not even enqueued', xs + 6, y + 0.5, { size: 11, font: 'mono', color: C.ink, base: 'middle', maxW: xe - xs - 10 });
            }
          }
        });
        V.line(g, xNow + 0.5 * sp - 3, 26, xNow + 0.5 * sp - 3, 126, { color: V.alpha(C.ink, 0.35), lw: 1, dash: [3, 3] });
        // clocks
        const cy = 146;
        const clock = (y, lab, lim, col) => {
          V.text(g, lab, m, y + 11, { size: 11.5, color: C.ink, maxW: narrow ? 116 : 240 });
          const bx = m + (narrow ? 124 : 250), bw = x1 - bx - (narrow ? 86 : 90);
          V.box(g, bx, y, bw, 14, { r: 5, fill: V.alpha(C.muted, 0.15) });
          const f = V.clamp((hanging ? sim.e : 0) / lim, 0, 1);
          V.box(g, bx, y, Math.max(1, bw * f), 14, { r: 5, fill: V.alpha(col, 0.8) });
          const jx = bx + bw * V.clamp(S.jit / lim, 0, 1);
          if (S.jit <= lim) { V.line(g, jx, y - 3, jx, y + 17, { color: C.c, lw: 2 }); }
          V.text(g, Math.round(hanging ? sim.e : 0) + ' / ' + lim + ' s', x1, y + 11, { size: 11, font: 'mono', color: C.muted, align: 'right' });
        };
        clock(cy, narrow ? 'NCCL watchdog' : 'NCCL watchdog (ProcessGroupNCCL)', 600, C.d);
        clock(cy + 26, narrow ? 'executor RPC' : 'RPC execute_model/sample_tokens', S.rpc, C.b);
        V.text(g, '│ = end of JIT (' + S.jit + ' s)  ·  time sped up 60×', m, cy + 60, { size: 11, color: C.c });
        // outcome
        const oy = cy + 74;
        let msg, col;
        if (sim.phase === 'hit') { msg = '✓ cache hit: the kernel was compiled at boot (warmup) — no wait'; col = C.a; }
        else if (sim.phase === 'recovered') { msg = `✓ the JIT finished in ${S.jit} s, before the deadlines: serving goes on (but that step took ${S.jit} s)`; col = C.a; }
        else if (sim.phase === 'hang') { msg = 'rank 0 stuck inside the NCCL kernel, GPU busy, no error in the log…'; col = C.d; }
        else if (sim.outcome === 'rpc') { msg = '✕ the RPC hit 300 s before the JIT finished: EngineCore dies — that is why GLM and DS use 1800 s'; col = C.red || C.c; }
        else if (sim.outcome === 'wd') { msg = "✕ 600 s watchdog: the collective is aborted and the TP pair goes down (1800 s on the RPC doesn't help here)"; col = C.red || C.c; }
        else { msg = 'normal serving: both lanes complete the same collectives in the same order'; col = C.muted; }
        para(g, msg, m, oy + 12, w - 2 * m, { size: 12, weight: 600, color: col, lh: 16 });
        // flight recorder
        const fy = oy + 44;
        V.line(g, m, fy - 8, w - m, fy - 8, { color: V.alpha(C.muted, 0.25), lw: 1 });
        const dumped = sim.outcome === 'wd' && S.fr;
        const head = !S.fr ? 'flight recorder off' : dumped ? 'dump written: …/nccl-fr/comm_lib_trace_rank_0 and _1' : sim.outcome === 'rpc' ? "no automatic dump: the watchdog didn't fire — request one via the FIFO /tmp/fr_dump_pipe_<rank>.pipe" : 'flight recorder (TORCH_FR_BUFFER_SIZE=2000): latest collectives per rank';
        const headH = para(g, head, m, fy + 6, w - 2 * m, { size: 11.5, weight: 600, color: dumped ? C.a : S.fr ? C.ink : C.muted, lh: 15 });
        if (S.fr) {
          const tW = (w - 2 * m - 12) / 2;
          [0, 1].forEach((r) => {
            const tx = m + r * (tW + 12), ty = fy + 6 + headH + 8;
            V.text(g, 'rank ' + r, tx, ty, { size: 11.5, weight: 600, color: r ? C.b : C.a });
            const base = hanging ? sim.hangSeq : sim.seq + 1;
            const ents = [];
            for (let k = 3; k >= 1; k--) ents.push(['#' + (base - k), 'all_reduce', 'completed']);
            if (hanging && r === 0) ents.push(['#' + base, 'all_reduce', 'started']);
            ents.slice(-4).forEach((en, k) => {
              const yy = ty + 8 + k * 16;
              const last = k === ents.slice(-4).length - 1;
              V.text(g, en[0] + ' ' + en[1] + ' · ' + en[2], tx, yy + 10, { size: 11, font: 'mono', color: last && hanging ? (r === 0 ? C.d : C.c) : C.muted, maxW: tW });
            });
          });
          if (dumped) para(g, `torchfrtrace: rank 1 stopped at #${sim.hangSeq - 1} and never enqueued #${sim.hangSeq} → it hung BEFORE the collective (not a desync)`, m, h - 22, w - 2 * m, { size: 11.5, color: C.a, lh: 15 });
        } else if (sim.outcome === 'wd') para(g, 'peer lost and no evidence at the collective level — the situation that motivated the flight recorder in DS', m, fy + 30, w - 2 * m, { size: 11.5, color: C.c, lh: 15 });
      });
    })();

    /* ============================================ Fig 10: environment per recipe */
    (function () {
      const fig = root.querySelector('#fig-env');
      const ctl = UI.controls(fig);
      const list = fig.querySelector('.nr-env');
      const S = { rec: 'glm', node: 'head', cat: 'all' };
      const CATS = [
        { v: 'all', label: 'all' }, { v: 'cont', label: 'container' }, { v: 'rdv', label: 'rendezvous' }, { v: 'if', label: 'interfaces' },
        { v: 'gid', label: 'GID' }, { v: 'tr', label: 'transport' }, { v: 'tun', label: 'tuning' }, { v: 'ipc', label: 'SHM/IPC' }, { v: 'for', label: 'forensics' },
      ];
      const LAYER = { cont: 0, ipc: 1, rdv: 2, if: 2, gid: 3, tr: 3, tun: 3, for: -1, all: -2 };
      const CCOL = { cont: '--viz-d', rdv: '--viz-b', if: '--viz-b', gid: '--viz-c', tr: '--viz-a', tun: '--viz-e', ipc: '--viz-f', for: '--viz-c' };
      const D = {
        glm: [
          ['cont', '--gpus all --network host', '', 'GPU visible; NCCL, gloo, RoCE, and API without NAT'],
          ['cont', '--ipc=host --shm-size 32g', '', 'shm_broadcast between vLLM processes'],
          ['cont', '--device /dev/infiniband --cap-add IPC_LOCK --ulimit memlock=-1', '', 'verbs in the container; pin RDMA memory'],
          ['cont', '--ulimit stack=67108864', '', '64 MiB stack'],
          ['rdv', '--distributed-executor-backend mp --tensor-parallel-size 2 --nnodes 2', '', 'one process per node, no Ray'],
          ['rdv', '--node-rank', ['0', '1'], 'position in the group'],
          ['rdv', '--master-addr 10.0.0.1 --master-port 29521', '', 'TCPStore on the head (HEAD_IP, MASTER_PORT); port checked in the preflight'],
          ['rdv', '--headless', ['—', '(on)'], 'only rank 0 serves HTTP'],
          ['rdv', 'VLLM_HOST_IP', ['10.0.0.1', '10.0.0.2'], 'HEAD_IP / WORKER_IP: address of the ZMQ plan'],
          ['if', 'NCCL_SOCKET_IFNAME = GLOO_SOCKET_IFNAME', ['enp1s0f1np1', 'enp1s0f0np0'], "HEAD_CX7_IF / WORKER_CX7_IF — crossed cabling; the 10.0.0.x alias won't do"],
          ['if', 'NCCL_IB_HCA', ['rocep1s0f1', 'rocep1s0f0'], 'HEAD_CX7_IB / WORKER_CX7_IB (list = dual-rail)'],
          ['gid', 'NCCL_IB_GID_INDEX', ['3 (kit: HEAD_GID=4)', '3 (WORKER_GID)'], 'per NIC; the preflight validates it on every HCA (errno 61 in ~60 s)'],
          ['gid', 'NCCL_IB_ROCE_VERSION_NUM=2', '', 'RoCE v2-type GIDs'],
          ['tr', 'NCCL_NET=IB NCCL_IB_DISABLE=0 NCCL_NET_PLUGIN=none', '', 'built-in verbs, no falling back to sockets'],
          ['tr', 'NCCL_NVLS_ENABLE=0 NCCL_CUMEM_ENABLE=0 NCCL_IGNORE_CPU_AFFINITY=1', '', 'no NVSwitch; no cuMem on UMA (interpretation); affinity unmeasured'],
          ['tr', 'NCCL_IB_MERGE_NICS=0 NCCL_CROSS_NIC=0', '', 'each port pinned per rank (TP3/TP4: CROSS_NIC=1)'],
          ['tr', 'USE_HOST_NCCL=0', '', 'preloading a 2nd NCCL makes DeepEP complain about a duplicate'],
          ['tun', 'NCCL_NCHANNELS=', '', 'empty = NCCL decides; an integer pins MIN=MAX (ref. 8)'],
          ['ipc', 'GLM53_SPINWAIT_MS=stock', '', '1 s spin window; the sweep picked 16 ms (+0.95% decode, −85.3% CPU)'],
          ['for', 'NCCL_DEBUG=WARN', '', 'INFO to diagnose transport/HCA/GID'],
          ['for', 'VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800', '', 'legitimate JIT > 300 s; the NCCL watchdog stays at 600 s'],
        ],
        qwen: [
          ['cont', '--gpus all --network host --ipc host', '', 'NCCL, NFS, and API without NAT; shared memory between processes'],
          ['cont', '--cap-add SYS_NICE --ulimit memlock=-1 --ulimit stack=67108864', '', 'memlock to register RDMA memory'],
          ['cont', '--device /dev/infiniband:/dev/infiniband', '', 'without the verbs devices (and with no NCCL_NET pinned) NCCL falls back to NET/Socket'],
          ['rdv', '--distributed-executor-backend mp --tensor-parallel-size 2 --nnodes 2', '', 'mp: the only one that creates the PLE offload'],
          ['rdv', '--node-rank', ['0', '1'], 'worker comes up first; head waits 15 s'],
          ['rdv', '--master-addr $HEAD_IP --master-port 50000', '', 'rendezvous over the CX7 link'],
          ['rdv', '--headless', ['—', '(on)'], 'worker without API'],
          ['rdv', 'VLLM_HOST_IP', ['10.0.0.1', '10.0.0.2'], 'ZMQ plan (RTT 0.15–0.4 ms)'],
          ['if', 'GLOO_ / NCCL_ / TP_SOCKET_IFNAME', ['enp1s0f1np1 (IFACE)', 'enp1s0f0np0 (WORKER_IFACE)'], 'live cluster values; sample: enp1s0f0np0'],
          ['if', 'NCCL_IB_HCA', ['=rocep1s0f1', '=rocep1s0f0'], '“=” = exact match; never a port from another cluster'],
          ['if', 'NCCL_IB_AUTO_DETECT=0', '', 'intent: exact HCA; undocumented (likely a no-op)'],
          ['gid', 'NCCL_IB_GID_INDEX=3', '', "“if 3 doesn't come up, try 5”"],
          ['tr', 'NCCL_IB_DISABLE=0', '', 'without it, TCP sockets'],
          ['tr', 'NCCL_NET=IB NCCL_NET_PLUGIN=none NCCL_IB_ROCE_VERSION_NUM=2', '', 'suggested (audit), not adopted', true],
          ['tr', 'NCCL_CUMEM_ENABLE=0 NCCL_NVLS_ENABLE=0 NCCL_IGNORE_CPU_AFFINITY=1', '', 'suggested: “harmless”, used by the other GB10 recipes', true],
          ['tun', 'NCCL_PROTO=LL|LL128 NCCL_ALGO=Ring', '', 'suggested: 5–20 KB messages', true],
          ['tun', 'NCCL_IB_QPS_PER_CONNECTION=2..4 NCCL_IB_SPLIT_DATA_ON_QPS=0', '', 'suggested: helps prefill, neutral on decode', true],
          ['tun', 'NCCL_IB_HCA + 2nd PF · NCCL_CROSS_NIC=1', ['=rocep1s0f1,roceP2p1s0f1', '=rocep1s0f0,roceP2p1s0f0'], 'suggested: ~2× prefill communication, same decode', true],
          ['tun', 'NCCL_DMABUF_ENABLE=1 NCCL_NET_GDR_LEVEL=SYS', '', 'suggested: test GDRDMA/dmabuf (nvidia_peermem missing)', true],
          ['ipc', 'busy_loop_s (SpinCondition)', '', 'stock (1 s), no hotfix in this recipe'],
          ['for', 'NCCL_DEBUG=WARN', '', 'INFO + NCCL_DEBUG_SUBSYS=INIT,NET after changing the fabric (suggested)'],
          ['for', 'TORCH_NCCL_DUMP_ON_TIMEOUT=1 TORCH_NCCL_ENABLE_MONITORING=1', '', 'suggested: dump instead of a silent hang', true],
          ['for', 'nccl-tests all_reduce_perf -b 4K -e 64M -f 2 -g 1', '', 'suggested: 8–32 KB latency, GDRDMA/dmabuf', true],
        ],
        ds: [
          ['cont', 'network_mode: host · ipc: host · shm_size: 64gb', '', 'RoCE, TCPStore, and ZMQ direct; broadcast buffers'],
          ['cont', 'ulimits.memlock: -1 · ulimits.stack: 67108864 · devices: /dev/infiniband', '', 'verbs memory registration (ibv_reg_mr); 64 MiB stack; NCCL IB in the container'],
          ['cont', '/tmp ← DSPARK_TMP_HOST (~/.cache/dspark-tmp)', '', 'flight recorder FIFO visible on the host'],
          ['rdv', '--distributed-executor-backend mp --nnodes ${NNODES:-2}', '', 'one container per node; 3 in TP3'],
          ['rdv', '--node-rank ${NODE_RANK}', ['0', '1'], 'injected per node (env -u NODE_RANK -u HEADLESS)'],
          ['rdv', '--master-addr ${MASTER_ADDR} --master-port 25000', '', "head's IP on the RoCE mesh; the head doesn't restart in the middle of loading (#38)"],
          ['rdv', '--headless (HEADLESS)', ['—', 'HEADLESS=1'], 'worker healthcheck exits 0'],
          ['rdv', 'VLLM_HOST_IP', ['head RoCE IP', 'WORKER_VLLM_HOST_IP'], "keeps the worker from binding to the head's IP"],
          ['if', 'NCCL_IB_HCA', ['rocep1s0f1', 'rocep1s0f0 (WORKER_NCCL_IB_HCA)'], 'recommended: both PFs, =rocep1s0fX,roceP2p1s0fX'],
          ['if', 'NCCL_SOCKET_IFNAME = TP_ = GLOO_SOCKET_IFNAME', ['enp1s0f1np1', 'enp1s0f0np0 (WORKER_*)'], 'bootstrap and Gloo; also the NFS interface'],
          ['gid', 'NCCL_IB_GID_AUTO=1', '', 'resolver in sysfs, fails closed, no pin'],
          ['gid', 'NCCL_IB_GID_INDEX', ['(unset)', '(unset)'], 'NCCL picks per HCA; empty would mean index 0'],
          ['gid', 'NCCL_IB_ADDR_FAMILY=AF_INET NCCL_IB_ROCE_VERSION_NUM=2', '', 'IPv4-mapped v2-type GIDs'],
          ['gid', 'NCCL_IB_GID_MATCH_IP=', '', "only if the RoCE IP isn't the primary IPv4"],
          ['tr', 'NCCL_NET=IB NCCL_IB_DISABLE=0', '', 'forces RDMA'],
          ['tr', 'NCCL_CUMEM_ENABLE=0 NCCL_IGNORE_CPU_AFFINITY=1 NCCL_NVLS_ENABLE=0', '', 'fixed on the GB10'],
          ['tr', 'NCCL_CROSS_NIC=1 · NCCL_IB_MERGE_NICS (unset = 1)', '', 'two PFs; TP3 forces MERGE_NICS=0'],
          ['tun', 'MTU 9000 · ethtool -G rx 8192 tx 8192 (host)', '', '+7.6% E2E with dual-HCA; same decode'],
          ['tun', 'NCCL_GIN_ENABLE (unset)', '', '0 → comm-init from ~2 min to ~13 s'],
          ['tun', 'NCCL_NET_GDR_LEVEL / _GDR_READ / NCCL_DMABUF_ENABLE', '', 'unset; no GPUDirect observed on the measured stack'],
          ['ipc', 'busy_loop_s = 0.002 (hotfix #79)', '', '2 ms of spinning, then sleeps; P-cores freed'],
          ['ipc', 'SHM_READER_RECHECK_INTERVAL_MS = 5000 (backport #45224)', '', 'a lost wakeup costs ≤5 s; slot released in finally'],
          ['ipc', 'DSPARK_SKIP_SPIN_WAIT_HOTFIX=0 DSPARK_SKIP_ISSUE117_RECHECK_HOTFIX=0', '', '1 = revert (diagnostics only)'],
          ['for', 'TORCH_FR_BUFFER_SIZE=2000 TORCH_NCCL_DUMP_ON_TIMEOUT=1 TORCH_NCCL_ENABLE_MONITORING=1', '', 'dump on the 600 s watchdog'],
          ['for', 'TORCH_FR_DUMP_TEMP_FILE=/cache/huggingface/nccl-fr/comm_lib_trace_rank_', '', 'archive both ranks together'],
          ['for', 'TORCH_NCCL_DEBUG_INFO_PIPE_FILE=/tmp/fr_dump_pipe_', '', 'on-demand dump; wait for “Finished writing…”'],
          ['for', 'NCCL_DEBUG=WARN VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800', '', 'INFO to see NET/IB; RPC above the JIT'],
        ],
      };
      UI.seg(ctl, { label: 'Recipe', options: [{ v: 'glm', label: 'GLM' }, { v: 'qwen', label: 'Qwen' }, { v: 'ds', label: 'DS' }], value: S.rec, onChange: (v) => { S.rec = v; render(); } });
      UI.seg(ctl, { label: 'Node', options: [{ v: 'head', label: 'head' }, { v: 'worker', label: 'worker' }], value: S.node, onChange: (v) => { S.node = v; render(); } });
      UI.select(ctl, { label: 'Purpose', options: CATS, value: S.cat, onChange: (v) => { S.cat = v; render(); } });
      const h = LA.h;
      function render() {
        list.innerHTML = '';
        list.style.cssText = 'display:grid;grid-template-columns:repeat(auto-fill,minmax(min(100%,330px),1fr));gap:6px;padding:8px 18px 16px;align-items:start';
        const rows = D[S.rec].filter((r) => S.cat === 'all' || r[0] === S.cat);
        const ni = S.node === 'head' ? 0 : 1;
        rows.forEach(([cat, key, vals, why, sug]) => {
          const per = Array.isArray(vals);
          const val = per ? vals[ni] : '';
          const catLab = (CATS.find((c) => c.v === cat) || {}).label || cat;
          list.appendChild(h('div', { style: { display: 'grid', gridTemplateColumns: 'minmax(0,1fr)', gap: '2px', padding: '6px 10px', margin: '0', borderLeft: '3px solid var(' + CCOL[cat] + ')', background: 'color-mix(in srgb, var(' + CCOL[cat] + ') 7%, transparent)', borderRadius: '6px' } },
            h('div', { style: { fontFamily: 'var(--font-mono)', fontSize: '12px', color: 'var(--text)', wordBreak: 'break-word', overflowWrap: 'anywhere' } },
              per ? h('span', { style: { color: 'var(--viz-c)', fontWeight: '700', marginRight: '6px' }, title: 'differs between head and worker' }, '≠') : null,
              key + (per ? ' = ' : ''), per ? h('b', { style: { color: 'var(--accent-text)' } }, val) : null),
            h('div', { style: { fontSize: '12px', lineHeight: '1.4', color: 'var(--text-3)' } }, h('span', { style: { textTransform: 'uppercase', letterSpacing: '.08em', fontSize: '10.5px', marginRight: '8px', color: 'var(' + CCOL[cat] + ')' } }, catLab), sug ? h('span', { style: { color: 'var(--viz-d)', marginRight: '6px' } }, 'suggested ·') : null, sug ? why.replace(/^suggested:\s*/, '') : why)));
        });
        if (!rows.length) list.appendChild(h('p', { style: { color: 'var(--text-3)', fontSize: '13px' } }, 'This recipe defines no variables for this purpose.'));
        HH.redraw();
      }
      const HH = V.fig(fig, (g, w, hgt, t) => {
        const C = V.C();
        const m = 12, narrow = w < 560;
        const layers = [['container', C.d, 'container'], ['local SHM', C.f, 'SHM'], ['TCP · bootstrap/control', C.b, 'TCP'], ['RDMA · data', C.a, 'RDMA']];
        const L = LAYER[S.cat];
        const nw = narrow ? (w - 2 * m) * 0.34 : Math.min(260, w * 0.3);
        const nodes = [[m, 'head'], [w - m - nw, 'worker']];
        const y0 = 30, lh = V.clamp((hgt - y0 - 34) / layers.length - 6, 22, 36);
        nodes.forEach(([x, name]) => {
          const sel = name === S.node;
          V.text(g, (narrow ? '' : name === 'head' ? 'spark1 · ' : 'spark2 · ') + name, x + nw / 2, 18, { size: 12, weight: 600, color: sel ? C.ink : C.muted, align: 'center' });
          V.box(g, x, y0 - 4, nw, layers.length * (lh + 6) + 2, { r: 9, fill: V.alpha(sel ? C.a : C.muted, sel ? 0.06 : 0.02), stroke: V.alpha(sel ? C.a : C.muted, sel ? 0.6 : 0.2), lw: sel ? 1.6 : 1 });
          layers.forEach(([lab, col, short], k) => {
            const on = L === -2 || L === k || (L === -1);
            const y = y0 + k * (lh + 6);
            V.box(g, x + 4, y, nw - 8, lh, { r: 6, fill: V.alpha(col, on ? (L === -1 ? 0.14 : 0.3) : 0.05), stroke: V.alpha(col, on ? 0.8 : 0.18), lw: 1 });
            V.text(g, narrow ? short : lab, x + nw / 2, y + lh / 2 + 0.5, { size: 11.5, weight: 600, color: on ? C.ink : C.muted, align: 'center', base: 'middle' });
          });
        });
        const xA = m + nw, xB = w - m - nw;
        const wire = [[2, narrow ? 'bootstrap · ZMQ' : 'TCPStore · Gloo · NCCL bootstrap · ZMQ'], [3, narrow ? 'all-reduces' : 'NCCL all-reduces (verbs)']];
        wire.forEach(([k, lab]) => {
          const y = y0 + k * (lh + 6) + lh / 2, col = layers[k][1];
          const on = L === -2 || L === k || L === -1;
          V.line(g, xA, y, xB, y, { color: V.alpha(col, on ? 0.55 : 0.12), lw: on ? 4 : 2 });
          V.text(g, lab, (xA + xB) / 2, y - 9, { size: 11, font: 'mono', color: on ? col : V.alpha(C.muted, 0.7), align: 'center', maxW: xB - xA - 8 });
          if (on) for (let q = 0; q < 4; q++) { const u = (q / 4 + t * (k === 3 ? 0.5 : 0.25)) % 1; V.box(g, V.lerp(xA, xB, q % 2 ? 1 - u : u) - 4, y - 3, 8, 6, { r: 2, fill: col }); }
        });
        // container and SHM live inside each node: a pulse shows where they act
        [0, 1].forEach((k) => {
          if (!(L === k || L === -2)) return;
          const y = y0 + k * (lh + 6) + lh / 2;
          nodes.forEach(([x]) => V.glow(g, x + 12, y, 9 + 2 * Math.sin(t * 4), layers[k][1], 0.7));
        });
        const foot = L === -1 ? '⏺ flight recorder and logs: evidence from every layer' : L === -2 ? 'all layers lit · pick a purpose' : 'lit layer: ' + layers[L][0];
        V.text(g, foot, w / 2, hgt - 10, { size: 11.5, color: L === -1 ? C.c : C.muted, align: 'center' });
      }, { static: false });
      render();
    })();

    /* stage height by width (the TensorFold figures stack text on phones) */
    function fitH(fig, fn) {
      const st = fig.querySelector('.la-fig-stage');
      const apply = () => { const w = st.getBoundingClientRect().width; if (!w) return; const hh = Math.round(fn(w)); if (Math.abs((parseFloat(st.style.height) || 0) - hh) > 1) st.style.height = hh + 'px'; };
      apply();
      if (window.ResizeObserver) { const ro = new ResizeObserver(apply); ro.observe(fig); LA.onCleanup && LA.onCleanup(() => ro.disconnect()); }
      return apply;
    }

    /* ============================================ Fig 11: one-shot RDMA, inside (TensorFold) */
    (function () {
      const fig = root.querySelector('#fig-tf-oneshot');
      fitH(fig, (w) => (w < 640 ? 505 : 480));
      const ctl = UI.controls(fig);
      const S = { rails: 2, flag: 'qp', rows: 4 };
      const sim = { u: 0, seq: 7, epoch: 41, last: '' };
      const restart = () => { sim.u = 0; };
      UI.seg(ctl, { label: 'Rails', options: [{ v: 1, label: '1 HCA' }, { v: 2, label: '2 HCAs (both PFs)' }], value: 2, onChange: (v) => { S.rails = v; restart(); } });
      UI.seg(ctl, { label: 'The flag goes…', options: [{ v: 'qp', label: 'on the same QP, behind the data' }, { v: 'side', label: 'by another path (hypothetical)' }], value: 'qp', onChange: (v) => { S.flag = v; restart(); } });
      UI.slider(ctl, { label: 'Window rows', min: 1, max: 32, step: 1, value: 4, fmt: (v) => v + ' × 16 KiB', onInput: (v) => { S.rows = v; restart(); } });
      const rdStripe = UI.readout(ctl, 'Per stripe', '—');
      const TRAVEL = 0.55, NPK = 4;
      const plan = () => {
        const kib = S.rows * 16, stripe = kib / S.rails;
        const dur = 0.95 + 1.6 * stripe / 256;                      // illustrative: α + n/β
        const t0 = 2.0, tData = t0 + dur;
        const tFlag = S.flag === 'qp' ? tData + 0.2 : t0 + 0.5;
        const tSee = tFlag + 0.15, tCopy = tSee + 0.9, tEnd = tCopy + 0.45;
        return { kib, stripe, dur, t0, tData, tFlag, tSee, tCopy, tEnd, total: tEnd + 1.1 };
      };
      V.fig(fig, (g, w, h, t, dt) => {
        const C = V.C();
        const d = LA.paused ? 0 : Math.min(dt, 0.05);
        const P = plan();
        sim.u += d;
        if (sim.u >= P.total) { sim.u = 0; sim.seq++; sim.epoch++; }
        const u = sim.u, N = sim.seq, slot = N & 1, R = S.rails, side = S.flag === 'side';
        const rs = P.stripe + ' KiB × ' + R;
        if (rs !== sim.last) { sim.last = rs; rdStripe.set(rs); }
        const narrow = w < 640;
        const red = C.red || C.c;
        const m = 12, gap = narrow ? 40 : Math.max(110, w - 2 * m - 2 * 310);
        const colW = Math.floor((w - 2 * m - gap) / 2);
        const Ax = m, Bx = w - m - colW;
        const yG = 30, hG = 48, yP = 90, hP = 116, yX = 218, hX = 46, yN = 276, hN = 38, dN = 46;
        const hcaY = (k) => yN + k * dN;
        const panelB = hcaY(1) + hN + 8;
        const clamp01 = (x) => V.clamp(x, 0, 1);
        const pStage = clamp01(u / 0.9);
        const rung = u >= 1.0;
        const pArr = clamp01((u - (P.t0 + TRAVEL)) / Math.max(0.05, P.tData - P.t0 - TRAVEL));
        const pCopy = clamp01((u - P.tSee) / (P.tCopy - P.tSee));
        const seen = u >= P.tSee;
        const epochNow = sim.epoch + (u >= P.tCopy ? 1 : 0);
        // panels
        [[Ax, C.a, narrow ? 'Spark A · rank 0' : 'Spark A · rank 0 (sends)'], [Bx, C.b, narrow ? 'Spark B · rank 1' : 'Spark B · rank 1 (receives)']].forEach(([X0, col, lab]) => {
          V.box(g, X0, 22, colW, panelB - 22, { r: 12, fill: V.alpha(col, 0.05), stroke: V.alpha(col, 0.35), lw: 1 });
          V.text(g, lab, X0 + colW / 2, 13, { size: 12, weight: 600, color: col, align: 'center', maxW: colW });
        });
        const ix = (X0) => X0 + 10, iw = colW - 20;
        const lab = (s, x, y, o = {}) => V.text(g, s, x, y, Object.assign({ size: 11, weight: 600, color: C.ink, maxW: iw - 8 }, o));
        // ---------------- side A
        {
          const x = ix(Ax);
          V.box(g, x, yG, iw, hG, { r: 8, fill: V.alpha(C.a, 0.1), stroke: V.alpha(C.a, 0.5), lw: 1 });
          lab(narrow ? 'GPU' : 'GPU · 4 blocks', x + 8, yG + 17);
          const bs = narrow ? 12 : 15, bg = narrow ? 4 : 6;
          for (let i = 0; i < 4; i++) {
            const bx = x + iw - 8 - (4 - i) * (bs + bg) + bg, by = yG + hG - bs - 8;
            const s0 = 0.08 + i * 0.16, s1 = s0 + 0.32;
            const active = u >= s0 && u < s1, done = u >= s1 && u < P.tEnd;
            V.box(g, bx, by, bs, bs, { r: 3, fill: V.alpha(C.a, active ? 0.85 : done ? 0.45 : 0.15), stroke: V.alpha(C.a, 0.7), lw: 1 });
            if (active) {
              const f = (u - s0) / (s1 - s0);
              const sy = yP + 34 + slot * 26 + 8;
              const px = V.lerp(bx + bs / 2, x + (narrow ? 64 : 70) + (iw - (narrow ? 72 : 80)) * ((i + f) / 4), f), py = V.lerp(by + bs, sy, f);
              V.box(g, px - 3, py - 3, 6, 6, { r: 1.5, fill: C.a });
            }
          }
          if (!narrow) V.text(g, 'local shard → send[' + slot + ']', x + 8, yG + 36, { size: 11, font: 'mono', color: C.muted, maxW: iw - 4 * 21 - 16 });
          // pinned region
          V.box(g, x, yP, iw, hP, { r: 8, fill: V.alpha(C.d, 0.06), stroke: V.alpha(C.d, 0.45), lw: 1 });
          lab(narrow ? 'pinned region' : 'pinned region · LPDDR5X', x + 8, yP + 17);
          const sx = x + (narrow ? 64 : 70), sw = iw - (narrow ? 72 : 80);
          [0, 1].forEach((s) => {
            const yy = yP + 34 + s * 26;
            V.text(g, 'send[' + s + ']', x + 8, yy + 12, { size: 11, font: 'mono', color: s === slot ? C.ink : C.muted });
            V.box(g, sx, yy, sw, 16, { r: 4, fill: V.alpha(C.muted, 0.12) });
            if (s === slot) {
              V.box(g, sx, yy, Math.max(1, sw * pStage), 16, { r: 4, fill: V.alpha(C.a, 0.55) });
              const reading = u >= P.t0 && u < P.tData;
              if (reading) V.box(g, sx, yy, sw, 16, { r: 4, stroke: V.alpha(C.a, 0.5 + 0.4 * Math.sin(t * 9)), lw: 2 });
            } else V.box(g, sx, yy, sw, 16, { r: 4, fill: V.alpha(C.a, 0.16) });
          });
          const cy = yP + 88;
          V.text(g, 'ctrl.seq', x + 8, cy + 12, { size: 11, font: 'mono', color: C.ink });
          const ring = u >= 1.0 && u < 1.7;
          if (ring) V.glow(g, sx + 18, cy + 9, 16 + 3 * Math.sin(t * 12), C.d, 0.8);
          chip(g, sx, cy, String(rung ? N : N - 1), C.d, { a: rung ? 0.35 : 0.12, h: 18 });
          if (!narrow && u >= 0.9 && u < 1.7) V.text(g, 'fence.sc.sys → doorbell', sx + 44, cy + 12, { size: 11, font: 'mono', color: C.d, maxW: sw - 44 });
          // proxy
          V.box(g, x, yX, iw, hX, { r: 8, fill: V.alpha(C.c, 0.07), stroke: V.alpha(C.c, 0.45), lw: 1 });
          lab(narrow ? 'proxy (C)' : 'proxy · thread C (libibverbs)', x + 8, yX + 17);
          const st = u < 1.0 ? (narrow ? 'spins on ctrl.seq' : 'spins on ctrl.seq, waiting') : u < P.t0 ? (narrow ? 'posts 2 WRs/HCA' : 'doorbell! posts WRITE + flag per HCA') : (narrow ? 'posted; spins' : 'requests posted; back to spinning');
          V.text(g, st, x + 8, yX + 35, { size: 11, font: 'mono', color: u >= 1.0 && u < P.t0 ? C.c : C.muted, maxW: iw - 30 });
          const spin = u < 1.0 || u >= P.t0;
          if (spin) { const a = t * 6; V.dot(g, x + iw - 14 + 6 * Math.cos(a), yX + hX / 2 + 6 * Math.sin(a), 3, C.c); }
          // HCAs
          for (let k = 0; k < 2; k++) {
            const y = hcaY(k), on = k < R;
            V.box(g, x, y, iw, hN, { r: 8, fill: V.alpha(C.e, on ? 0.1 : 0.03), stroke: V.alpha(C.e, on ? 0.55 : 0.2), lw: 1 });
            V.text(g, narrow ? 'HCA' + k : 'HCA' + k + ' · QP RC', x + 8, y + hN / 2 + 1, { size: 11, weight: 600, color: on ? C.ink : C.muted, base: 'middle' });
            if (!on) { V.text(g, narrow ? 'unused' : 'not used', x + iw - 8, y + hN / 2 + 1, { size: 11, color: C.muted, align: 'right', base: 'middle' }); continue; }
            const posted = u >= 1.35 + k * 0.15;
            const dataGone = u >= P.tData - TRAVEL, flagGone = u >= P.tFlag - (side ? 0.45 : TRAVEL);
            let cx = x + iw - 8;
            if (posted && !flagGone) cx -= chip(g, cx, y + 10, narrow ? 'flag' : 'WRITE flag', C.d, { align: 'right', h: 18, size: 11, pad: narrow ? 4 : 7 }) + (narrow ? 3 : 4);
            if (posted && !dataGone) chip(g, cx, y + 10, narrow ? 'data' : 'WRITE stripe', C.a, { align: 'right', h: 18, size: 11, pad: narrow ? 4 : 7 });
          }
          // the NIC's DMA: reads the slot and goes down to the HCAs
          if (u >= P.t0 && u < P.tData) {
            const xx = Ax + colW - 4;
            V.line(g, xx, yP + 42 + slot * 26, xx, hcaY(R - 1) + hN / 2, { color: V.alpha(C.a, 0.8), lw: 2, dash: [4, 4], dashOffset: -t * 30 });
            g.lineDashOffset = 0;
          }
        }
        // ---------------- wires
        const xa = Ax + colW, xb = Bx;
        for (let k = 0; k < 2; k++) {
          const y = hcaY(k) + hN / 2, on = k < R;
          if (on) V.line(g, xa, y, xb, y, { color: V.alpha(C.b, 0.35), lw: 5 });
          else V.line(g, xa, y, xb, y, { color: V.alpha(C.muted, 0.35), lw: 1.5, dash: [3, 5] });
          if (!on) continue;
          for (let i = 0; i < NPK; i++) {
            const dep = P.t0 + (i / (NPK - 1)) * (P.tData - TRAVEL - P.t0);
            const f = (u - dep) / TRAVEL;
            if (f > 0 && f < 1) V.box(g, V.lerp(xa, xb, f) - 7, y - 4, 14, 8, { r: 2, fill: C.a });
          }
          // flag
          const dep = side ? P.t0 + 0.05 + k * 0.04 : P.tFlag - TRAVEL;
          const tr = side ? 0.45 : TRAVEL;
          const f = (u - dep) / tr;
          if (f > 0 && f < 1) {
            let px, py;
            if (side) { const c = [(xa + xb) / 2, yX - 4 - k * 8]; [px, py] = quad([xa, y], c, [xb, y], f); }
            else { px = V.lerp(xa, xb, f); py = y; }
            V.box(g, px - 11, py - 8, 22, 16, { r: 4, fill: C.d });
            V.text(g, String(N), px, py + 0.5, { size: 11, font: 'mono', weight: 700, color: C.dark ? '#1a1300' : '#fff', align: 'center', base: 'middle' });
          }
          if (side) {
            const c = [(xa + xb) / 2, yX - 4 - k * 8];
            g.strokeStyle = V.alpha(red, 0.55); g.lineWidth = 1.5; g.setLineDash([4, 4]);
            g.beginPath(); g.moveTo(xa, y); g.quadraticCurveTo(c[0], c[1], xb, y); g.stroke(); g.setLineDash([]);
          }
        }
        if (side && !narrow) V.text(g, 'another path (hypothetical)', (xa + xb) / 2, (hcaY(0) + hN / 2 + yX - 4) / 2 - 12, { size: 11, color: red, align: 'center', maxW: gap - 6 });
        if (!narrow) {
          const mx = (xa + xb) / 2;
          if (!side) V.text(g, 'QSFP cable · RoCE v2', mx, hcaY(0) - 8, { size: 11, font: 'mono', color: C.b, align: 'center', maxW: gap - 8 });
          for (let k = 0; k < R; k++) V.text(g, 'rail ' + k, xa + 6, hcaY(k) + hN / 2 + 16, { size: 11, color: C.muted });
          if (gap >= 150) {
            const lx = mx - 56, ly = yP + 30;
            V.box(g, lx, ly - 4, 14, 8, { r: 2, fill: C.a });
            V.text(g, 'data (stripe)', lx + 24, ly + 0.5, { size: 11, color: C.ink, base: 'middle' });
            V.box(g, lx - 4, ly + 17, 22, 14, { r: 4, fill: C.d });
            V.text(g, 'flag (4 B)', lx + 24, ly + 24.5, { size: 11, color: C.ink, base: 'middle' });
            V.text(g, 'B → A: the same, at once', mx, ly + 56, { size: 11, color: C.muted, align: 'center', maxW: gap - 8 });
          }
        }
        // ---------------- side B
        {
          const x = ix(Bx);
          V.box(g, x, yG, iw, hG, { r: 8, fill: V.alpha(C.b, 0.1), stroke: V.alpha(C.b, 0.5), lw: 1 });
          lab(narrow ? 'GPU' : 'GPU · waits on the flags', x + 8, yG + 17);
          for (let k = 0; k < R; k++) {
            const sx = x + 8 + k * (narrow ? 58 : 92), sy = yG + 34;
            if (!seen && u >= 0.4) { const a = t * 7 + k; V.dot(g, sx + 5 + 4.5 * Math.cos(a), sy + 1 + 4.5 * Math.sin(a), 2.4, C.b); }
            else if (seen) V.text(g, '✓', sx + 6, sy + 2, { size: 12, weight: 700, color: side ? red : C.a, align: 'center', base: 'middle' });
            V.text(g, narrow ? 'f' + k : 'thread f' + k, sx + 18, sy + 2, { size: 11, font: 'mono', color: C.muted, base: 'middle' });
          }
          if (!narrow) V.text(g, 'ld.acquire.sys', x + iw - 8, yG + 17, { size: 11, font: 'mono', color: C.muted, align: 'right' });
          // pinned region
          V.box(g, x, yP, iw, hP, { r: 8, fill: V.alpha(C.d, 0.06), stroke: V.alpha(C.d, 0.45), lw: 1 });
          lab(narrow ? 'pinned region' : 'pinned region · LPDDR5X', x + 8, yP + 17);
          const sx = x + (narrow ? 72 : 80), sw = iw - (narrow ? 80 : 88);
          [0, 1].forEach((s) => {
            const yy = yP + 34 + s * 26;
            V.text(g, narrow ? 'recv[' + s + ']' : 'recv[0][' + s + ']', x + 8, yy + 12, { size: 11, font: 'mono', color: s === slot ? C.ink : C.muted });
            V.box(g, sx, yy, sw, 16, { r: 4, fill: V.alpha(C.muted, 0.12) });
            if (s !== slot) { V.box(g, sx, yy, sw, 16, { r: 4, fill: V.alpha(C.a, 0.16) }); return; }
            const segW = sw / R;
            for (let k = 0; k < R; k++) {
              const x0 = sx + k * segW;
              V.box(g, x0 + 1, yy + 1, segW - 2, 14, { r: 3, fill: V.alpha(C.muted, 0.22) });
              if (pArr > 0) V.box(g, x0 + 1, yy + 1, Math.max(1, (segW - 2) * pArr), 14, { r: 3, fill: V.alpha(C.a, 0.6) });
              if (pArr < 0.5 && segW > 54) V.text(g, 'exch. ' + (N - 2), x0 + segW - 4, yy + 8.5, { size: 11, font: 'mono', color: C.muted, align: 'right', base: 'middle', maxW: segW - 8 });
            }
          });
          // flags
          const fy = yP + 88;
          V.text(g, 'flags', x + 8, fy + 12, { size: 11, font: 'mono', color: C.ink });
          let fx = sx;
          for (let k = 0; k < R; k++) {
            const got = u >= P.tFlag;
            fx += chip(g, fx, fy, (narrow ? '' : 'f' + k + '=') + (got ? N : N - 2), C.d, { a: got ? 0.35 : 0.1, sa: got ? 0.8 : 0.35, h: 18 }) + 6;
          }
          // output
          V.box(g, x, yX, iw, hX, { r: 8, fill: V.alpha(C.b, 0.07), stroke: V.alpha(C.b, 0.45), lw: 1 });
          lab(narrow ? 'output' : 'output, in rank order', x + 8, yX + 15);
          const ep = u >= P.tCopy && u < P.tEnd + 0.4;
          if (ep) V.glow(g, x + iw - 30, yX + 12, 15, C.e, 0.4);
          V.text(g, (narrow ? 'ep. ' : 'epoch ') + epochNow, x + iw - 8, yX + 15, { size: 11, font: 'mono', weight: ep ? 700 : 400, color: ep ? C.e : C.muted, align: 'right' });
          const ox = x + 8, ow = iw - 16, oy = yX + 24, half = ow / 2;
          V.box(g, ox, oy, ow, 15, { r: 4, fill: V.alpha(C.muted, 0.12) });
          if (pCopy > 0) {
            const okPart = side ? clamp01((P.tSee - P.t0 - TRAVEL) / Math.max(0.05, P.tData - P.t0 - TRAVEL)) : 1;
            const fillL = half * pCopy;
            V.box(g, ox, oy, Math.max(1, Math.min(fillL, half * okPart)), 15, { r: 4, fill: V.alpha(C.a, 0.65) });
            if (fillL > half * okPart) V.box(g, ox + half * okPart, oy, fillL - half * okPart, 15, { r: 2, fill: V.alpha(red, 0.6) });
            V.box(g, ox + half, oy, Math.max(1, half * pCopy), 15, { r: 4, fill: V.alpha(C.b, 0.6) });
          }
          V.text(g, 'rank 0', ox + half / 2, oy + 8, { size: 11, font: 'mono', color: C.ink, align: 'center', base: 'middle' });
          V.text(g, 'rank 1', ox + half * 1.5, oy + 8, { size: 11, font: 'mono', color: C.ink, align: 'center', base: 'middle' });
          // HCAs
          for (let k = 0; k < 2; k++) {
            const y = hcaY(k), on = k < R;
            V.box(g, x, y, iw, hN, { r: 8, fill: V.alpha(C.e, on ? 0.1 : 0.03), stroke: V.alpha(C.e, on ? 0.55 : 0.2), lw: 1 });
            V.text(g, 'HCA' + k, x + 8, y + hN / 2 + 1, { size: 11, weight: 600, color: on ? C.ink : C.muted, base: 'middle' });
            V.text(g, on ? (narrow ? 'RDMA' : 'writes into memory') : (narrow ? 'unused' : 'not used'), x + iw - 8, y + hN / 2 + 1, { size: 11, color: C.muted, align: 'right', base: 'middle' });
          }
          if (u >= P.t0 + TRAVEL && u < P.tData + 0.25) {
            const xx = Bx + 4;
            V.line(g, xx, hcaY(R - 1) + hN / 2, xx, yP + 42 + slot * 26, { color: V.alpha(C.a, 0.8), lw: 2, dash: [4, 4], dashOffset: -t * 30 });
            g.lineDashOffset = 0;
          }
        }
        // ---------------- dynamic caption
        let cap, bad = false;
        if (u < 0.9) cap = `① Stage: each GPU block copies its share of the shard (${S.rows} × 16 KiB = ${P.kib} KiB) into send[${slot}], in pinned memory, which on the GB10 is the same LPDDR5X.`;
        else if (u < 1.3) cap = `② Doorbell: the last block to finish issues fence.sc.sys and writes ctrl.seq = ${N}. The proxy thread, which was spinning on that address, wakes up.`;
        else if (u < P.t0) cap = `③ Post: on each HCA, the proxy chains on the same RC QP an RDMA WRITE with the stripe (${P.stripe} KiB) and a 4-byte inline WRITE with ${N} into the stripe's flag.`;
        else if (u < P.tSee) cap = side ? '④ Hypothesis: the flag takes another path, and nothing orders it relative to the data. It arrives first, and the threads waiting on it release the copy.' : `④ Fly: the NIC reads the stripe from pinned memory and writes it straight into the peer's recv[0][${slot}]; the flag follows on the same queue and only becomes visible after the data. On the other side, one GPU thread per stripe reads the flag with ld.acquire.sys.`;
        else if (u < P.tCopy) { cap = side ? `✕ The GPU saw the flag and copied recv[0][${slot}] with the stripe still arriving: part of the output is exchange ${N - 2}, which was in the slot (in red).` : `⑤ Every flag reads ${N}: each block copies the shards in rank order, the peer's from recv[0][${slot}] and its own from the input.`; bad = side; }
        else { cap = side ? '✕ Corrupted output, with no error at all. That is why the proxy puts the flag on the same reliable QP, behind the data: the queue order is the guarantee.' : `⑥ The last block advances the device-side epoch, so a CUDA graph can replay the exchange without the host. The next exchange uses slot ${(N + 1) & 1}: two slots are enough, because the peer is never more than one exchange ahead.`; bad = side; }
        const by = panelB + 18;
        const ch = para(g, cap, m, by, w - 2 * m, { size: narrow ? 11.5 : 12.5, weight: 500, color: bad ? red : C.ink, lh: narrow ? 15 : 17 });
        const stats = narrow ? `payload ${P.kib} KiB · ${R} × ${P.stripe} KiB · seq ${N} · slot ${slot} · B → A at the same time · illustrative times` : `payload ${P.kib} KiB · ${R} stripe${R > 1 ? 's' : ''} of ${P.stripe} KiB · seq ${N} · slot ${slot} · epoch ${epochNow} · illustrative times (on the hardware, 16 KiB ≈ 11 µs)`;
        para(g, stats, m, by + ch + 6, w - 2 * m, { size: 11, font: 'mono', color: C.muted, lh: 14 });
      });
    })();

    /* ============================================ Fig 12: the same door, the same patience */
    (function () {
      const fig = root.querySelector('#fig-tf-match');
      const S = { mode: 'pick', rule: 'old', send: 'view', skew: 40, wait: 20, barrier: false };
      const refit = fitH(fig, (w) => (w < 560 ? (S.mode === 'pick' ? 500 : 400) : 380));
      const ctl = UI.controls(fig);
      const sim = { u: 0, k: 1, ok: 0, e: 0, last: '' };
      const restart = () => Object.assign(sim, { u: 0, k: 1, ok: 0, e: 0 });
      let gP = null, gB = null;
      UI.seg(ctl, { label: 'Scenario', options: [{ v: 'pick', label: 'transport choice (0040)' }, { v: 'boot', label: 'first start (0052)' }], value: 'pick', onChange: (v) => { S.mode = v; gP.show(v === 'pick'); gB.show(v === 'boot'); restart(); refit(); } });
      gP = UI.group(ctl); gB = UI.group(ctl);
      UI.seg(gP.el, { label: 'Rule', options: [{ v: 'old', label: 'before: bytes and address' }, { v: 'new', label: '0040: bytes only' }], value: 'old', onChange: (v) => { S.rule = v; restart(); } });
      UI.seg(gP.el, { label: 'Rank 0 sends from', options: [{ v: 'view', label: 'view buf[1:1+n]' }, { v: 'own', label: 'its own buffer' }], value: 'view', onChange: (v) => { S.send = v; restart(); } });
      UI.slider(gB.el, { label: 'Rank 1 compile delay', min: 0, max: 90, step: 5, value: 40, fmt: (v) => v + ' s', onInput: (v) => { S.skew = v; restart(); } });
      UI.slider(gB.el, { label: 'TF_ROCE_WAIT_S', min: 5, max: 120, step: 5, value: 20, fmt: (v) => v + ' s', onInput: (v) => { S.wait = v; restart(); } });
      UI.toggle(gB.el, { label: 'NCCL barrier until the engine is ready (0052)', value: false, onChange: (v) => { S.barrier = v; restart(); } });
      gB.show(false);
      UI.button(ctl, '↺ restart', restart);
      const rdOut = UI.readout(ctl, 'Outcome', '—');
      const setOut = (s, st) => { const key = s + '|' + st; if (key !== sim.last) { sim.last = key; rdOut.set(s, st); } };
      const MSG = 48;                                             // message bytes (illustrative, a multiple of 16)
      const ADDR = { view: 0x1c04, own: 0x1c00, r1: 0x8a80 };     // only the low 16 bits (illustrative)
      const hex = (a) => '0x7f…' + a.toString(16).padStart(4, '0');
      const choose = (a) => (S.rule === 'new' || a % 16 === 0 ? 'roce' : 'nccl');
      const COMP = 90;                                            // rank 0's compile time (illustrative)
      V.fig(fig, (g, w, h, t, dt) => {
        const C = V.C();
        const d = LA.paused ? 0 : Math.min(dt, 0.05);
        const narrow = w < 560, m = 12, red = C.red || C.c;
        if (S.mode === 'pick') {
          const a0 = S.send === 'view' ? ADDR.view : ADDR.own, a1 = ADDR.r1;
          const p0 = choose(a0), p1 = choose(a1), same = p0 === p1;
          sim.u += d;
          if (same && sim.u >= 2.4) { sim.u = 0; sim.k++; sim.ok++; }
          const done = sim.ok + (same && sim.u >= 1.5 ? 1 : 0);   // exchanges done, counting the one on screen
          const stuck = !same && sim.u >= 1.5;
          if (stuck) sim.e = Math.min(sim.e + d * 30, 400);
          const u = sim.u;
          // geometry
          const cardW = narrow ? w - 2 * m : Math.floor((w - 2 * m - 46) * 0.5);
          const cardH = narrow ? 88 : 112;
          const cards = narrow ? [[m, 8], [m, 104]] : [[m, 8], [m, 134]];
          const roomW = narrow ? Math.floor((w - 2 * m - 10) / 2) : w - 2 * m - cardW - 46;
          const roomH = narrow ? 96 : 112;
          const rooms = { roce: narrow ? [m, 204] : [m + cardW + 46, 8], nccl: narrow ? [m + roomW + 10, 204] : [m + cardW + 46, 134] };
          const roomCol = { roce: C.d, nccl: C.c };
          const roomLab = { roce: narrow ? 'RDMA (shot)' : 'one-shot RDMA', nccl: 'NCCL' };
          // rooms
          ['roce', 'nccl'].forEach((k) => {
            const [x, y] = rooms[k];
            V.box(g, x, y, roomW, roomH, { r: 12, fill: V.alpha(roomCol[k], 0.07), stroke: V.alpha(roomCol[k], 0.55), lw: 1.3 });
            V.text(g, roomLab[k], x + 10, y + 18, { size: 12, weight: 600, color: roomCol[k] });
          });
          // rank cards
          const reveal = (k) => u >= 0.15 + k * 0.22 || stuck || sim.ok > 0;
          [[0, a0, p0, C.a], [1, a1, p1, C.b]].forEach(([r, a, p, col]) => {
            const [x, y] = cards[r];
            V.box(g, x, y, cardW, cardH, { r: 10, fill: V.alpha(col, 0.07), stroke: V.alpha(col, 0.45), lw: 1 });
            V.text(g, r ? 'rank 1 · worker' : 'rank 0 · head', x + 10, y + 17, { size: 12, weight: 600, color: col });
            const what = r ? (narrow ? 'receives: own buffer' : 'receives into its own buffer (_share)') : S.send === 'view' ? (narrow ? 'sends: view buf[1:1+n]' : 'sends the message from a view: buf[1:1+n]') : (narrow ? 'sends: own buffer' : 'sends the message from its own buffer');
            V.text(g, what, x + 10, y + 35, { size: 11, color: C.ink, maxW: cardW - 20 });
            const cw = chip(g, x + 10, y + 43, hex(a), col, { h: 18, size: 11 });
            V.text(g, '% 16 = ' + (a % 16), x + 18 + cw, y + 56, { size: 11, font: 'mono', color: a % 16 ? red : C.muted });
            let ly = y + (narrow ? 76 : 80);
            if (reveal(0)) { V.text(g, (narrow ? '✓ 48 B ≤ 512 KiB, ×16' : '✓ ' + MSG + ' B ≤ 512 KiB and a multiple of 16'), x + 10, ly, { size: 11, color: C.a, maxW: cardW - 20 }); }
            if (!narrow) ly += 16;
            if (S.rule === 'old' && reveal(1)) {
              const okA = a % 16 === 0;
              V.text(g, okA ? '✓ address aligned to 16 B' : '✗ address not aligned to 16 B', narrow ? x + cardW / 2 + 4 : x + 10, ly, { size: 11, color: okA ? C.a : red, maxW: narrow ? cardW / 2 - 14 : cardW - 20 });
            } else if (S.rule === 'new' && reveal(1) && r === 0 && S.send === 'view') {
              V.text(g, narrow ? '↳ aligned copy' : '↳ address ignored; aligned copy first', narrow ? x + cardW / 2 + 4 : x + 10, ly, { size: 11, color: C.muted, maxW: narrow ? cardW / 2 - 14 : cardW - 20 });
            }
            const dec = '→ ' + (p === 'roce' ? (narrow ? 'shot' : 'one-shot RDMA') : 'NCCL');
            if (reveal(2)) V.text(g, dec, x + cardW - 10, y + 17, { size: 12, weight: 700, color: roomCol[p], align: 'right' });
          });
          // tokens that walk to the room
          const slotPos = (k, r) => { const [x, y] = rooms[k]; return [x + roomW * (r ? 0.68 : 0.32), y + roomH * (narrow ? 0.42 : 0.5)]; };
          const fM = V.ease.inOut(V.clamp((u - 0.8) / 0.7, 0, 1));
          [[0, p0, C.a], [1, p1, C.b]].forEach(([r, p, col]) => {
            const [cx, cy] = cards[r];
            const start = narrow ? [cx + cardW * (r ? 0.75 : 0.25), cy + cardH] : [cx + cardW, cy + cardH / 2];
            const end = slotPos(p, r);
            const pos = stuck || sim.ok > 0 && u < 0.8 ? end : [V.lerp(start[0], end[0], fM), V.lerp(start[1], end[1], fM)];
            if (u >= 0.8 || stuck) {
              V.dot(g, pos[0], pos[1], 12, col);
              V.text(g, String(r), pos[0], pos[1] + 0.5, { size: 12, font: 'mono', weight: 700, color: C.dark ? '#0b0b14' : '#fff', align: 'center', base: 'middle' });
            }
          });
          // meeting or waiting
          if (same && u >= 1.5) {
            const [x0, y0] = slotPos(p0, 0), [x1] = slotPos(p0, 1);
            V.line(g, x0 + 13, y0, x1 - 13, y0, { color: V.alpha(C.a, 0.8), lw: 2.5 });
            V.glow(g, (x0 + x1) / 2, y0, 18 + 3 * Math.sin(t * 8), C.a, 0.6);
            const [rx, ry] = rooms[p0];
            V.text(g, '✓ exchange #' + sim.k, rx + roomW / 2, ry + roomH - 12, { size: 11.5, weight: 600, color: C.a, align: 'center' });
          }
          if (stuck) {
            ['roce', 'nccl'].forEach((k) => {
              const [rx, ry] = rooms[k];
              const r = p0 === k ? 0 : 1;
              const by = ry + roomH - (narrow ? 22 : 26), bw = roomW - 20;
              if (k === 'roce') {
                const f = V.clamp(sim.e / 20, 0, 1), dead = sim.e >= 20;
                V.box(g, rx + 10, by, bw, 10, { r: 4, fill: V.alpha(C.muted, 0.15) });
                V.box(g, rx + 10, by, Math.max(1, bw * f), 10, { r: 4, fill: V.alpha(dead ? red : C.d, 0.8) });
                V.text(g, dead ? (narrow ? '✕ poisoned' : '✕ 20 s: poisons the runtime') : (narrow ? 'flag? ' : 'waits on the flag: ') + Math.round(sim.e) + ' / 20 s', rx + 10, by - 6, { size: 11, color: dead ? red : C.ink, maxW: bw });
              } else {
                V.text(g, (narrow ? 'waits: ' : 'waits for the peer in NCCL: ') + Math.round(sim.e) + ' s…', rx + 10, by - 6, { size: 11, color: C.ink, maxW: bw });
                V.text(g, narrow ? 'no deadline' : 'no deadline: “forever”', rx + 10, by + 8, { size: 11, color: red, maxW: bw });
              }
              if (!narrow) V.text(g, 'rank ' + r + ' alone', rx + roomW - 10, ry + 18, { size: 11, color: C.muted, align: 'right' });
            });
          }
          // bottom text
          const ty = narrow ? 318 : 262;
          let msg, col;
          if (!same && !stuck) { msg = 'Each rank decides on its own, looking at its bytes and its address…'; col = C.muted; setOut('…', null); }
          else if (stuck) { msg = `✕ Rank 0 went to ${p0 === 'nccl' ? 'NCCL' : 'the shot'} and rank 1 to ${p1 === 'nccl' ? 'NCCL' : 'the shot'}: each waits for its peer at the wrong door. The shot side gives up after ~20 s and poisons the runtime; the NCCL side waits forever. The address is local to each process, and a local decision broke a collective.`; col = red; setOut('hung: different doors', 'bad'); }
          else if (S.rule === 'new') { msg = `✓ Only the bytes decide (${MSG} B, equal on both ranks): both go to the shot${S.send === 'view' ? ', and rank 0 first copies the view into an aligned buffer' : ''}. That is patch 0040's rule: “decided by the byte count alone, which every rank's matching call shares”.`; col = C.a; setOut(done + (done === 1 ? ' exchange' : ' exchanges') + ', same door', 'ok'); }
          else { msg = '✓ With both buffers aligned, the old rule gets it right, but by luck: the decision still depends on each process\'s address. Switch rank 0 to the view and watch.'; col = C.d; setOut(done + (done === 1 ? ' exchange' : ' exchanges') + ' (by luck)', 'warn'); }
          const mh = para(g, msg, m, ty + 10, w - 2 * m, { size: 12, weight: 500, color: col, lh: 16 });
          para(g, 'rank 0 control message: int32 vector, ' + MSG + ' B (illustrative) · illustrative addresses · waiting time accelerated 30×', m, ty + 16 + mh, w - 2 * m, { size: 11, color: C.muted, lh: 14 });
          return;
        }
        // ---------------- first start
        const c1 = COMP + S.skew, lim = S.wait;
        const dies = !S.barrier && S.skew > lim;
        const Tmax = Math.max(c1, COMP + lim) + 40;
        sim.e += d * 24;
        if (sim.e > Tmax + 30) sim.e = 0;
        const now = Math.min(sim.e, Tmax);
        const labW = narrow ? 50 : 70, x0 = m + labW, x1 = w - m;
        const X = (s) => x0 + (x1 - x0) * Math.min(s, Tmax) / Tmax;
        const yL0 = 66, yL1 = 146;
        const lanes = [[0, yL0, C.a], [1, yL1, C.b]];
        const bh = 30;
        const bar = (y, s0, s1, col, label, o = {}) => {
          if (now <= s0) return;
          const e = Math.min(now, s1);
          V.box(g, X(s0), y, Math.max(2, X(e) - X(s0)), bh, { r: 6, fill: V.alpha(col, o.a || 0.3), stroke: V.alpha(col, 0.75), lw: 1 });
          const avail = X(e) - X(s0) - 12;
          let lb = label;
          if (lb && tw(g, lb, 11) > avail) lb = o.short && tw(g, o.short, 11) <= avail ? o.short : null;
          if (lb) V.text(g, lb, X(s0) + 6, y + bh / 2 + 0.5, { size: 11, color: C.ink, base: 'middle' });
        };
        lanes.forEach(([r, y, col]) => {
          V.text(g, 'rank ' + r, m, y + bh / 2 + 1, { size: 12, weight: 600, color: col, base: 'middle' });
          V.line(g, x0, y + bh / 2, x1, y + bh / 2, { color: V.alpha(C.muted, 0.2), lw: 1 });
        });
        // axis
        const ay = 212;
        V.line(g, x0, ay, x1, ay, { color: V.alpha(C.muted, 0.5), lw: 1 });
        const step = Tmax > 200 ? 60 : 30;
        for (let s = 0; s <= Tmax; s += step) { V.line(g, X(s), ay - 3, X(s), ay + 3, { color: C.muted, lw: 1 }); V.text(g, s + ' s', X(s), ay + 15, { size: 11, font: 'mono', color: C.muted, align: 'center' }); }
        // compilation
        bar(yL0, 0, COMP, C.c, narrow ? 'compiles' : 'compiles CUDA extensions');
        bar(yL1, 0, c1, C.c, narrow ? 'compiles' : 'compiles CUDA extensions (slower)');
        // wait limit
        if (!S.barrier) {
          const xl = X(COMP + lim);
          V.line(g, xl, 36, xl, ay - 6, { color: V.alpha(red, 0.6), lw: 1.5, dash: [4, 4] });
          V.text(g, 'TF_ROCE_WAIT_S', xl, 28, { size: 11, font: 'mono', color: red, align: X(COMP + lim) > x1 - 60 ? 'right' : 'center' });
        }
        let outcome = null;
        if (S.barrier) {
          bar(yL0, COMP, c1, C.b, narrow ? 'barrier' : 'NCCL barrier (no limit)', { a: 0.25, short: 'barrier' });
          if (now >= c1) {
            [yL0, yL1].forEach((y) => { V.box(g, X(c1) - 1, y - 4, 6, bh + 8, { r: 2, fill: C.a }); });
            V.text(g, '✓ 1st gather', X(c1) + 8, yL0 - 8, { size: 11, weight: 600, color: C.a });
            for (let s = c1 + 6; s < now; s += 6) [yL0, yL1].forEach((y) => V.box(g, X(s) - 1, y + 7, 3, bh - 14, { r: 1, fill: V.alpha(C.a, 0.7) }));
            outcome = 'ok';
          }
        } else {
          const wEnd = COMP + Math.min(S.skew, lim);
          bar(yL0, COMP, wEnd, C.d, narrow ? 'waits on flag' : 'waits on the peer\'s flag', { a: 0.3, short: 'flag?' });
          if (dies) {
            if (now >= COMP + lim) {
              V.text(g, '✕', X(COMP + lim) + 2, yL0 + bh / 2 + 1, { size: 16, weight: 700, color: red, base: 'middle' });
              V.text(g, narrow ? 'poisons' : 'poisons the runtime', X(COMP + lim) + 18, yL0 + bh / 2 + 1, { size: 11, weight: 600, color: red, base: 'middle', maxW: x1 - X(COMP + lim) - 20 });
              outcome = 'dead';
            }
            if (now >= c1) V.text(g, narrow ? '✕ no peer' : '✕ arrives, no peer left', X(c1) + 4, yL1 - 8, { size: 11, weight: 600, color: red, align: X(c1) > x1 - 110 ? 'right' : 'left' });
          } else if (now >= c1) {
            [yL0, yL1].forEach((y) => { V.box(g, X(c1) - 1, y - 4, 6, bh + 8, { r: 2, fill: C.a }); });
            V.text(g, '✓ 1st gather', X(c1) + 8, yL0 - 8, { size: 11, weight: 600, color: C.a });
            outcome = 'lucky';
          }
        }
        // time cursor
        V.line(g, X(now), 36, X(now), ay, { color: V.alpha(C.ink, 0.4), lw: 1 });
        // text
        const ty = ay + 32;
        let msg, col;
        let out;
        if (S.barrier) { msg = `✓ Rank 0 arrives at ${COMP} s and waits for rank 1 in the NCCL barrier, which has no limit, until ${c1} s. The first exchange happens with both present; once the engine is ready, gathers go on without the barrier, and those captured in a graph never pass through it.`; col = C.a; out = ['start ok (barrier)', 'ok']; }
        else if (dies) { msg = `✕ Rank 0 waits on the flag for ${lim} s and poisons the runtime; when rank 1 arrives, at ${c1} s, there is no peer left. That is issue #6: the first start died on the first all-gather, because compiling extensions can push the ranks tens of seconds apart.`; col = red; out = ['poisoned at ' + lim + ' s', 'bad']; }
        else { msg = `✓ The delay (${S.skew} s) fit within the limit (${lim} s), and the wait on the flag held. But first-start compilation can push the ranks tens of seconds apart: a fixed limit is a bet. Turn on the barrier.`; col = C.d; out = ['ok, barely', 'warn']; }
        if (outcome === null) { msg = 'Both ranks compile their CUDA extensions, each at its own pace…'; col = C.muted; out = ['…', null]; }
        setOut(out[0], out[1]);
        const mh = para(g, msg, m, ty, w - 2 * m, { size: 12, weight: 500, color: col, lh: 16 });
        para(g, `rank 0 compile: ${COMP} s (illustrative) · wait limit = TF_ROCE_WAIT_S (patch default 20 s = 20 million polls of ~1 µs; the recipe uses 300 s since v1.6) · accelerated time`, m, ty + mh + 6, w - 2 * m, { size: 11, color: C.muted, lh: 14 });
      });
    })();
  },
});

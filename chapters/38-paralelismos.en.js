LA.chapter({
  id: 'paralelismos',
  part: 8,
  order: 38,
  title: 'Splitting the model: TP, EP, PP, DP, and DCP',
  navTitle: 'Parallelism',
  subtitle: 'Cutting matrices, experts, layers, requests, and tokens across two nodes, and predicting how much each cut charges on the network per token.',
  minutes: 65,
  recipes: ['glm', 'tf', 'qwen', 'ds'],
  tags: ['tensor parallel', 'column-parallel', 'row-parallel', 'all-reduce', 'expert parallel', 'pipeline parallel', 'data parallel', 'DCP', 'sequence parallel', 'distributed top-k', 'α-β'],
  goals: [
    'Explain <b>column-parallel</b>, <b>row-parallel</b>, and why the col → row pair costs a single all-reduce',
    'Map, in a real block, what is <b>sliced</b>, what is <b>replicated</b>, and where each recipe\'s ~100 collectives per step live',
    'Decide between <b>TP</b> and <b>EP</b> for the experts, looking at quantization blocks, kernels, and load imbalance',
    'Explain why PP and DP don\'t work for the recipes and what <b>DCP</b> and <b>SP</b> change in the KV and in prefill',
    'Estimate the network cost of a topology with the α-β model and know when replicating stops paying off',
  ],
  prereqs: ['prefill-decode-kv', 'mixture-of-experts', 'mla', 'atencao-esparsa', 'fp8-fp4-microescala', 'hadamard-trelica-exl3', 'paged-attention-kv', 'especulacao-fundamentos'],
  summary: [
    'TP cuts matrices: column-parallel slices the outputs without communicating, row-parallel slices the inputs and sums them with an all-reduce. The col → f → row pair costs <b>one</b> all-reduce, and each block has two (attention and FFN/MoE). In the GLM·TF recipe, TensorFold closes the pair with an all-gather of fp32 partials and a rank-0-first sum: one rounding, with the sum in the engine\'s own code and bits that don\'t depend on the transport, at the price of twice the bytes on the wire (and, in recipe v1.3.2, of staying at two ranks; v1.4 added an experimental three-rank mode that splits unevenly instead of padding).',
    'Under TP=2 the recipes do ~106 (Qwen) and ~108 (DeepSeek: 43 + 43, plus draft, embedding, logits, and Markov) small collectives per step, at 20–50 µs each: a 2–5 ms floor that doesn\'t drop with the batch.',
    'The cut has to respect the quantized structure: scale blocks and tiles along the cut axis, with <code>suh</code>/<code>svh</code> following the axis. When that isn\'t possible, the way out is EP (Qwen: 640/2 = 320 isn\'t a multiple of 128; GLM TP3: a 2048-wide trellis) or replication (Qwen\'s ViT: 2152 isn\'t a multiple of 16).',
    'EP distributes whole experts. With DP=1 the tokens are already replicated and the collective is still TP\'s all-reduce (Qwen\'s <code>--all2all-backend</code> is inert). The price is imbalance at small batch. DeepSeek doesn\'t use EP because <code>B12xExperts</code> doesn\'t support it.',
    'PP doesn\'t add up the bandwidths on one token (latency stays that of a single node) and DP requires the whole model per node: no recipe uses either. <code>--mm-encoder-tp-mode data</code> is the vision tower\'s local DP.',
    'With MLA/MQA the KV stays replicated under TP. DCP would split the tokens with an exact log-sum-exp merge (it isn\'t turned on in DSV4). The SP indexer splits the keys and merges the local top-k exactly (−4 to −6% TTFT at 128K). In the GLM·TF recipe, the prefill\'s hyper-connection glue is split by rows across the ranks (50k prefill from ~1,270 to ~1,730 tok/s, together with two other patches). Sharding the KV by conversation owner is the coarser alternative (GLM·TF patch 0129: 6,204 → 3,300 B per token per Spark, 1.87× the pool).',
    'In decode the messages sit to the left of the \\(\\alpha\\beta\\) knee (latency regime); in prefill, to the right (bandwidth regime). Replicating to save collectives has to be measured: the replicated Markov head lost (c=6: 122 versus 148 tok/s) and GLM\'s draft TP=2 was kept (clear gain on structured text, noise on prose).',
  ],
  quiz: [
    { q: 'Why does the column → row order cost one all-reduce per pair, while the row → col order would cost two?', o: ['Because column-parallel is faster on the GPU', 'Because NCCL only implements all-reduce after a row-parallel layer', 'Because \\(f\\) is elementwise: the column slice goes through \\(f\\) locally, but \\(f(P_0)+f(P_1) \\neq f(P_0+P_1)\\), so with row first you would have to add before \\(f\\)', 'Because row-parallel duplicates the weights on each rank'], a: 2, e: 'Each column of \\(XW\\) depends only on the matching column of \\(W\\), so the column-parallel slice can go through \\(f\\) without talking. A row-parallel output is a partial sum, and \\(f\\) doesn\'t distribute over the sum (ReLU of −2 + 3 isn\'t ReLU(−2) + ReLU(3)). Adding before \\(f\\) and again at the end makes two collectives.' },
    { q: 'In Qwen, with <code>--enable-expert-parallel</code> and DP=1, what does <code>--all2all-backend allgather_reducescatter</code> do at each MoE layer?', o: ['Dispatches each token to the GPU that owns its experts', 'Nothing: the tokens are already replicated by TP, each rank runs its 256 local experts, and TP\'s all-reduce adds up the contributions', 'Replaces TP\'s all-reduce with a cheaper all-to-all', 'Replicates the 512 experts on both GPUs'], a: 1, e: 'The recipe audit shows that <code>use_all2all_kernels</code> is false with DP=1 and PCP=1. Without data replicas, there are no different tokens to dispatch. EP only changes which weights each GPU stores (256 of 512), and the communication is still the normal all-reduce.' },
    { q: 'Why doesn\'t DeepSeek-V4\'s KV shrink per rank when going from 1 to 2 GPUs in TP?', o: ['Because the KV is in FP8', 'Because pipeline parallel is off', 'Because vLLM doesn\'t page the KV under TP', 'Because there\'s a single latent head (MQA/MLA): all query heads on all ranks read the same vector, so it stays replicated'], a: 3, e: 'TP splits query heads, and each rank needs the K/V its heads look up. With a single KV head, everyone needs the whole row. The second node helps because it frees weight memory (79.17 GiB per rank). DCP would split the tokens across the ranks, but it isn\'t turned on for DSV4 in this image.' },
    { q: 'Why would PP=2 instead of TP=2 make decode worse for a single user?', o: ['Because in PP the buses take turns on the same token: each stage reads half the weights, one after the other. In TP both read at the same time', 'Because PP sends more bytes over the network', 'Because PP requires expert parallel', 'Because PP doesn\'t support the KV cache'], a: 0, e: 'PP communicates little (one activation per boundary), but with a single microbatch one stage sits idle while the other works. Per-token latency stays \\(\\approx T\\), that of a single node. TP gives \\(\\approx T/2\\) + the cost of the collectives. That\'s why the new-recipe tutorial says PP doubles decode latency at small batch.' },
    { q: 'DeepSeek\'s SP indexer splits the keys across ranks and merges the local top-512 sets. Why is the result exact?', o: ['If \\(x\\) is in the global top-\\(k\\), at most \\(k-1\\) elements of the row beat it; so at most \\(k-1\\) of its own slice do, and \\(x\\) is in its owner\'s local top-\\(k\\)', 'Because each rank ends up seeing all the keys', 'Because the error is small and RULER-lite passed', 'Because the scores are computed in FP32'], a: 0, e: 'The union of the local top-\\(k\\) sets contains the global top-\\(k\\), and the top-\\(k\\) of the union is the global one. With ties, the order has to be stable (by id), hence DCP\'s <code>stable_topk…</code> selector. The tests against brute force confirm it, but the exactness comes from the proof.' },
    { q: 'A Qwen decode all-reduce is ~20 KB. With \\(\\alpha \\approx 25\\) µs and \\(\\beta \\approx 25\\) GB/s, what dominates the time?', o: ['Bandwidth: 20 KB ÷ 25 GB/s', 'The all-reduce computation on the GPU', 'Latency: \\(\\alpha \\approx 25\\) µs, versus ~0.8 µs of transfer; that\'s why ~100 collectives cost 2–3 ms per step and the cost doesn\'t drop with the batch', 'The Linux page cache'], a: 2, e: '20 × 10³ B ÷ 25 × 10⁹ B/s = 0.8 µs, much smaller than \\(\\alpha\\). The \\(\\alpha\\beta\\) knee sits at ~625 KB, and the decode message is well to its left. More sequences in the batch increase \\(n\\), but it stays far from the knee.' },
  ],
  mount(root, LA) {
    const V = LA.viz, UI = LA.ui;

    /* ================================================================ helpers */
    const RC = (C, r) => [C.a, C.b, C.d, C.c][r % 4];
    const nf = (x, d = 0) => Number(x).toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d });
    const FC = {};
    const fontOf = (size, font, weight) => { const k = size + '|' + font + '|' + weight; return FC[k] || (FC[k] = V.font(size, font, weight)); };
    const tw = (g, s, size = 12, font = 'sans', weight = 400) => { g.font = fontOf(size, font, weight); return g.measureText(String(s)).width; };
    function wrapLines(g, s, maxW, size, font, weight) {
      const out = [];
      String(s).split('\n').forEach((par) => {
        const words = par.split(' ');
        let line = '';
        words.forEach((wd) => {
          const test = line ? line + ' ' + wd : wd;
          if (line && tw(g, test, size, font, weight) > maxW) { out.push(line); line = wd; } else line = test;
        });
        out.push(line);
      });
      return out;
    }
    function wrap(g, s, x, y, maxW, o = {}) {
      const size = o.size || 12, lh = o.lh || Math.round(size * 1.38);
      const lines = wrapLines(g, s, maxW, size, o.font || 'sans', o.weight || 400);
      const n = o.maxLines ? Math.min(o.maxLines, lines.length) : lines.length;
      for (let i = 0; i < n; i++) V.text(g, lines[i], x, y + i * lh, { size, font: o.font, weight: o.weight, color: o.color, align: o.align, alpha: o.alpha });
      return y + n * lh;
    }
    function chip(g, s, x, y, col, o = {}) {
      const size = o.size || 11, font = o.font || 'mono', weight = o.weight || 600, hh = o.h || 20;
      let wd = tw(g, s, size, font, weight) + 14;
      if (o.maxW && wd > o.maxW) wd = o.maxW;
      const X = o.align === 'right' ? x - wd : o.align === 'center' ? x - wd / 2 : x;
      V.box(g, X, y - hh / 2, wd, hh, { r: hh / 2, fill: V.alpha(col, o.fa == null ? 0.16 : o.fa), stroke: V.alpha(col, 0.8), lw: 1 });
      V.text(g, s, X + wd / 2, y + 0.5, { size, font, weight, color: o.color || V.C().ink, align: 'center', base: 'middle', maxW: wd - 10 });
      return wd;
    }
    function hatch(g, x, y, w, h, col, step = 6, off = 0, lw = 1.1) {
      if (w <= 0 || h <= 0) return;
      g.save(); g.beginPath(); g.rect(x, y, w, h); g.clip();
      g.strokeStyle = col; g.lineWidth = lw; g.beginPath();
      const o = ((off % step) + step) % step;
      for (let k = -h - step + o; k < w + step; k += step) { g.moveTo(x + k, y + h); g.lineTo(x + k + h, y); }
      g.stroke(); g.restore();
    }
    function plus(g, x, y, r, col, on) {
      if (on) V.glow(g, x, y, r * 2.6, col, 0.8);
      V.box(g, x - r, y - r, 2 * r, 2 * r, { r, fill: V.alpha(col, on ? 0.9 : 0.25), stroke: col, lw: 1.5 });
      const k = r * 0.55, C = V.C();
      V.line(g, x - k, y, x + k, y, { color: on ? (C.dark ? '#1a0b1a' : '#fff') : col, lw: 2 });
      V.line(g, x, y - k, x, y + k, { color: on ? (C.dark ? '#1a0b1a' : '#fff') : col, lw: 2 });
    }
    function adaptH(fig, fn) {
      const st = fig.querySelector('.la-fig-stage');
      if (!st) return;
      const apply = () => { const w = st.clientWidth || fig.clientWidth || 800; const hh = Math.round(fn(w)); if (Math.abs(st.offsetHeight - hh) > 1) st.style.height = hh + 'px'; };
      apply();
      const ro = new ResizeObserver(apply);
      ro.observe(fig);
      if (LA.onCleanup) LA.onCleanup(() => ro.disconnect());
      return apply;
    }
    const gib = (b) => nf(b / 1073741824, b < 10 * 1073741824 ? 2 : 1) + ' GiB';
    const smooth = (x) => V.smooth(V.clamp(x, 0, 1));
    const MINUS = '−';
    const fi = (v) => (v < 0 ? MINUS + Math.abs(v) : String(v));

    /* label with subscript: 'Y|A' draws Y with a small A */
    function lab(g, s, cx, y, o = {}) {
      const size = o.size || 12, font = o.font || 'mono', weight = o.weight || 600;
      const runs = [];
      const str = String(s);
      for (let i = 0; i < str.length; i++) {
        if (str[i] === '|' && i + 1 < str.length) { runs.push({ t: str[i + 1], sub: true }); i++; }
        else if (runs.length && !runs[runs.length - 1].sub) runs[runs.length - 1].t += str[i];
        else runs.push({ t: str[i], sub: false });
      }
      const ws = runs.map((r) => tw(g, r.t, r.sub ? size * 0.72 : size, font, weight) + (r.sub ? 0.5 : 0));
      let x = cx - ws.reduce((a, b) => a + b, 0) / 2;
      runs.forEach((r, k) => { V.text(g, r.t, x, y + (r.sub ? size * 0.22 : 0), { size: r.sub ? size * 0.72 : size, font, weight, color: o.color }); x += ws[k]; });
    }

    /* ============================ F1 — five ways to split =================== */
    (function () {
      const fig = root.querySelector('#fig-mapa');
      if (!fig) return;
      adaptH(fig, (w) => (w < 560 ? 430 : 440));
      const ST = { mode: 'tp' };
      const ctl = UI.controls(fig);
      const C0 = V.C();
      UI.seg(ctl, { label: 'Split', options: [{ v: 'tp', label: 'TP' }, { v: 'ep', label: 'EP' }, { v: 'pp', label: 'PP' }, { v: 'dp', label: 'DP' }, { v: 'dcp', label: 'DCP' }], value: 'tp', onChange: (v) => { ST.mode = v; } });
      UI.legend(ctl, [[C0.a, 'node A · rank 0'], [C0.b, 'node B · rank 1'], [C0.c, '⊕ collective'], [C0.d, 'token']]);
      const INFO = {
        tp: { t: 'Tensor parallel (TP)', div: 'every matrix: whole heads per rank in attention, half of each expert in the MoE, id ranges in the vocabulary.', com: 'an all-reduce at the end of every attention and every FFN/MoE, in all layers.', rec: 'all three recipes: TP=2, one GPU per Spark.' },
        ep: { t: 'Expert parallel (EP)', div: 'whole experts across the GPUs; attention stays in TP.', com: 'with DP=1, the same all-reduce as TP; with DP>1, all-to-all to dispatch and combine tokens.', rec: 'Qwen (256 of 512 per GPU) and GLM on TP3 (96 of 288).' },
        pp: { t: 'Pipeline parallel (PP)', div: 'the layers, in stages: top half on one node, bottom half on the other.', com: 'the activation crosses the boundary, point to point, once per step.', rec: 'not used: DeepSeek passes --pipeline-parallel-size 1.' },
        dp: { t: 'Data parallel (DP)', div: 'the requests: each node runs the whole model and serves its own.', com: 'nothing in the forward pass, only coordination between the replicas.', rec: 'impossible here: none of the three models fits in one Spark.' },
        dcp: { t: 'Decode context parallel (DCP)', div: 'the KV cache tokens, interleaved across the TP ranks.', com: 'exact merge of the partial attentions (log-sum-exp) at every layer, on top of the TP collectives.', rec: 'not turned on; DeepSeek points to it as a way to double the KV.' },
      };
      const L = 4, NKV = 4, NH = 4, NE = 8;
      const CH = [[1, 6], [3, 4], [0, 7], [2, 5]];
      function cell(g, x, y, s, kind, own, C, glow) {
        const a = 0.55;
        if (glow) V.glow(g, x + s / 2, y + s / 2, s * 1.1, C.d, 0.9);
        if (kind === 'whole') V.box(g, x, y, s, s, { r: 3, fill: V.alpha(RC(C, own), a), stroke: RC(C, own), lw: 1 });
        else if (kind === 'splitV' || kind === 'splitH') {
          g.save(); V.rr(g, x, y, s, s, 3); g.clip();
          g.fillStyle = V.alpha(C.a, a);
          if (kind === 'splitV') g.fillRect(x, y, s / 2, s); else g.fillRect(x, y, s, s / 2);
          g.fillStyle = V.alpha(C.b, a);
          if (kind === 'splitV') g.fillRect(x + s / 2, y, s / 2, s); else g.fillRect(x, y + s / 2, s, s / 2);
          g.restore();
          V.box(g, x, y, s, s, { r: 3, stroke: V.alpha(C.ink, 0.35), lw: 1 });
          if (kind === 'splitV') V.line(g, x + s / 2, y + 1, x + s / 2, y + s - 1, { color: V.alpha(C.ink, 0.5), lw: 1 });
          else V.line(g, x + 1, y + s / 2, x + s - 1, y + s / 2, { color: V.alpha(C.ink, 0.5), lw: 1 });
        } else if (kind === 'dup') {
          const o = Math.max(2, s * 0.12);
          V.box(g, x + o, y - o, s - o, s - o, { r: 3, fill: V.alpha(C.b, a), stroke: C.b, lw: 1 });
          V.box(g, x, y, s - o, s - o + 0.5, { r: 3, fill: V.alpha(C.a, 0.75), stroke: C.a, lw: 1 });
        }
      }
      const kvK = (m, i, j) => (m === 'tp' || m === 'ep' ? ['splitH'] : m === 'pp' ? ['whole', i < 2 ? 0 : 1] : m === 'dp' ? ['dup'] : ['whole', j % 2]);
      const hdK = (m, i, k) => (m === 'pp' ? ['whole', i < 2 ? 0 : 1] : m === 'dp' ? ['dup'] : ['whole', k < 2 ? 0 : 1]);
      const exK = (m, i, e) => (m === 'pp' ? ['whole', i < 2 ? 0 : 1] : m === 'dp' ? ['dup'] : m === 'ep' ? ['whole', e < 4 ? 0 : 1] : ['splitV']);
      V.fig(fig, (g, w, h, t) => {
        if (w < 140 || h < 100) return;
        const C = V.C(), m = ST.mode, I = INFO[m];
        const narrow = w < 560, pad = 14;
        V.text(g, I.t, pad, 26, { size: narrow ? 15 : 17, font: 'display', weight: 700, color: C.ink });
        const infoH = narrow ? 150 : 92;
        const iy = h - infoH - 6;
        const reqW = narrow ? 26 : 46, labW = narrow ? 0 : 62, gB = narrow ? 8 : 18, oW = narrow ? 18 : 30, gap = 2;
        const nC = NKV + NH + NE;
        const cs = Math.max(8, Math.min(34, (w - 2 * pad - reqW - labW - gB - 2 * oW - (nC - 2) * gap - 8) / nC));
        const rh = cs + (narrow ? 16 : 18);
        const totalW = reqW + labW + (NKV + NH + NE) * cs + (nC - 3) * gap + gB + 2 * oW;
        const ox = Math.max(pad, (w - totalW) / 2);
        const top0 = narrow ? 64 : 62;
        const modelH = L * rh;
        const y0 = top0 + Math.max(0, (iy - 14 - top0 - modelH) / 2) + 10;
        let x = ox + reqW + labW;
        const xKV = x; x += NKV * (cs + gap) - gap + gB;
        const xH = x; x += NH * (cs + gap) - gap;
        const xO1 = x + oW / 2; x += oW;
        const xE = x; x += NE * (cs + gap) - gap;
        const xO2 = x + oW / 2; x += oW;
        const xEnd = x;
        // block headers
        const hy = y0 - 10;
        const hs = 11;
        V.text(g, narrow ? 'KV' : m === 'dcp' ? 'KV (per token)' : 'KV (tokens)', xKV + (NKV * (cs + gap)) / 2, hy, { size: hs, align: 'center', color: C.muted, font: 'mono' });
        V.text(g, narrow ? 'attention' : 'attention · 4 heads', xH + (NH * (cs + gap)) / 2, hy, { size: hs, align: 'center', color: C.muted, font: 'mono' });
        V.text(g, narrow ? 'MoE' : 'MoE · 8 experts', xE + (NE * (cs + gap)) / 2, hy, { size: hs, align: 'center', color: C.muted, font: 'mono' });
        if (!narrow) V.text(g, 'requests', ox + reqW / 2, hy, { size: hs, align: 'center', color: C.muted, font: 'mono' });
        // moving token
        const P = 7;
        const tokPos = (ph) => { const u = ((t / P + ph) % 1 + 1) % 1; const row = Math.min(L - 1, Math.floor(u * L)); const f = u * L - row; return { row, x: V.lerp(xKV - 8, xEnd + 4, f), f }; };
        const toks = m === 'dp' ? [tokPos(0), tokPos(0.5)] : [tokPos(0)];
        // requests
        for (let i = 0; i < 4; i++) {
          const cy = y0 + (i + 0.5) * rh * (L / 4) - 4, cx = ox + reqW / 2, rr = Math.min(9, cs * 0.42);
          if (m === 'dp') V.box(g, cx - rr, cy - rr, 2 * rr, 2 * rr, { r: rr, fill: V.alpha(RC(C, i % 2), 0.7), stroke: RC(C, i % 2) });
          else {
            g.save(); g.beginPath(); g.arc(cx, cy, rr, 0, Math.PI * 2); g.clip();
            g.fillStyle = V.alpha(C.a, 0.6); g.fillRect(cx - rr, cy - rr, rr, 2 * rr);
            g.fillStyle = V.alpha(C.b, 0.6); g.fillRect(cx, cy - rr, rr, 2 * rr);
            g.restore();
            g.strokeStyle = V.alpha(C.ink, 0.4); g.lineWidth = 1; g.beginPath(); g.arc(cx, cy, rr, 0, Math.PI * 2); g.stroke();
          }
          if (!narrow) V.text(g, 'R' + (i + 1), cx, cy + 0.5, { size: 11, font: 'mono', weight: 700, align: 'center', base: 'middle', color: C.ink });
        }
        // model rows
        for (let i = 0; i < L; i++) {
          const y = y0 + i * rh;
          if (!narrow) {
            const st = m === 'pp' ? (i < 2 ? 0 : 1) : -1;
            V.text(g, 'layer ' + i, ox + reqW + 4, y + cs / 2 + 4, { size: 11, font: 'mono', color: st >= 0 ? RC(C, st) : C.muted, weight: st >= 0 ? 700 : 400 });
          }
          const tk = toks.find((q) => q.row === i);
          for (let j = 0; j < NKV; j++) { const k = kvK(m, i, j); cell(g, xKV + j * (cs + gap), y, cs, k[0], k[1], C, false); }
          for (let k = 0; k < NH; k++) { const q = hdK(m, i, k); const on = tk && tk.x > xH && tk.x < xO1 - oW / 2; cell(g, xH + k * (cs + gap), y, cs, q[0], q[1], C, on && k === Math.floor((t * 3) % 4)); }
          for (let e = 0; e < NE; e++) {
            const q = exK(m, i, e);
            const chosen = CH[i].includes(e);
            const on = tk && tk.x > xE && tk.x < xO2 - oW / 2 && (m === 'ep' || m === 'tp' || m === 'dcp' || m === 'pp' || m === 'dp') && chosen;
            cell(g, xE + e * (cs + gap), y, cs, q[0], q[1], C, on);
          }
          const cy = y + cs / 2, pr = Math.min(10, oW / 2 - 2);
          if (m !== 'pp' && m !== 'dp') {
            const on1 = !!(tk && Math.abs(tk.x - xO1) < oW * 0.7), on2 = !!(tk && Math.abs(tk.x - xO2) < oW * 0.7);
            plus(g, xO1, cy, pr, C.c, on1);
            plus(g, xO2, cy, pr, C.c, on2);
            if (m === 'dcp' && !narrow && i === 0) V.text(g, 'LSE', xO1, y - 4, { size: 11, font: 'mono', align: 'center', color: C.c, weight: 700 });
          } else {
            V.line(g, xO1 - oW / 2 + 3, cy, xO1 + oW / 2 - 3, cy, { color: V.alpha(C.muted, 0.35), lw: 1.2, dash: [2, 3] });
            V.line(g, xO2 - oW / 2 + 3, cy, xO2 + oW / 2 - 3, cy, { color: V.alpha(C.muted, 0.35), lw: 1.2, dash: [2, 3] });
          }
          // DCP: lookups into both halves of the KV
          if (m === 'dcp' && tk && tk.x > xH && tk.x < xO1) {
            for (let j = 0; j < NKV; j++) V.arrow(g, xH, cy, xKV + j * (cs + gap) + cs / 2, y + cs, { color: V.alpha(RC(C, j % 2), 0.55), lw: 1.1, head: 5, curve: -10 });
          }
        }
        // PP: boundary between stages
        if (m === 'pp') {
          const yb = y0 + 2 * rh - (rh - cs) / 2;
          V.line(g, xKV - 6, yb, xEnd, yb, { color: V.alpha(C.c, 0.7), lw: 1.4, dash: [5, 4], dashOffset: -t * 20 });
          const crossing = toks[0].row === 2 && toks[0].f < 0.08;
          if (crossing) V.glow(g, xKV - 2, yb, 22, C.c, 0.9);
          V.text(g, narrow ? 'PP boundary' : 'stage boundary: the activation passes from A to B', xEnd, yb - 4, { size: 11, font: 'mono', align: 'right', color: C.c });
        }
        // tokens
        toks.forEach((tk, k) => {
          const ty = y0 + tk.row * rh + cs / 2;
          V.glow(g, tk.x, ty, 16, C.d, 0.95);
          V.dot(g, tk.x, ty, 5, C.d);
          if (m === 'dp') {
            g.strokeStyle = RC(C, k); g.lineWidth = 2; g.beginPath(); g.arc(tk.x, ty, 7, 0, Math.PI * 2); g.stroke();
            chip(g, (k ? 'R2 → node B' : 'R1 → node A'), tk.x, ty - cs / 2 - 10, RC(C, k), { align: 'center', size: 11, h: 18, fa: 0.3 });
          }
        });
        // info box
        V.box(g, pad, iy, w - 2 * pad, infoH, { r: 12, fill: V.alpha(C.muted, 0.08), stroke: V.alpha(C.muted, 0.3), lw: 1 });
        const lw = narrow ? 76 : 98, tx = pad + 12, ts = narrow ? 11.5 : 12.5, lh = narrow ? 15 : 17;
        let yy = iy + 22;
        [['Splits', I.div, C.a], ['Traffic', I.com, C.c], ['Recipes', I.rec, C.d]].forEach(([k, v, col]) => {
          V.text(g, k, tx, yy, { size: ts, weight: 700, color: col });
          const end = wrap(g, v, tx + lw, yy, w - 2 * pad - lw - 24, { size: ts, color: C.ink, lh });
          yy = end + (narrow ? 3 : 4);
        });
      });
    })();

    /* ====================== F2 — the column → row pair, step by step ======== */
    (function () {
      const fig = root.querySelector('#fig-colrow');
      if (!fig) return;
      adaptH(fig, (w) => (w < 640 ? 700 : 450));
      const mm = (A, B) => A.map((row) => B[0].map((_, j) => row.reduce((s, a, k) => s + a * B[k][j], 0)));
      const relu = (M) => M.map((r) => r.map((v) => Math.max(0, v)));
      const add = (A, B) => A.map((r, i) => r.map((v, j) => v + B[i][j]));
      const cols = (M, a, b) => M.map((r) => r.slice(a, b));
      const rows = (M, a, b) => M.slice(a, b);
      const rm = (R, K, f) => Array.from({ length: R }, () => Array.from({ length: K }, f));
      function gen(seed) {
        const r = V.rng(seed); const ri = (a, b) => a + Math.floor(r() * (b - a + 1));
        for (let tries = 0; tries < 400; tries++) {
          const X = rm(2, 4, () => ri(-2, 2)), W1 = rm(4, 4, () => ri(-2, 2)), W2 = rm(4, 4, () => ri(-1, 2));
          const Y = mm(X, W1), fl = Y.flat();
          if (!fl.some((v) => v < 0) || fl.filter((v) => v > 0).length < 3) continue;
          const XA = cols(X, 0, 2), XB = cols(X, 2, 4), P0 = mm(XA, rows(W1, 0, 2)), P1 = mm(XB, rows(W1, 2, 4));
          const lhs = add(relu(P0), relu(P1)), rhs = relu(add(P0, P1));
          const diff = [];
          lhs.forEach((row, i) => row.forEach((v, j) => { if (v !== rhs[i][j]) diff.push([i, j]); }));
          if (!diff.length) continue;
          const H = relu(Y), Z = mm(H, W2);
          const HA = cols(H, 0, 2), HB = cols(H, 2, 4), W2A = rows(W2, 0, 2), W2B = rows(W2, 2, 4);
          const ZA = mm(HA, W2A), ZB = mm(HB, W2B), ZS = add(ZA, ZB);
          const neg = (M) => { const o = []; M.forEach((row, i) => row.forEach((v, j) => { if (v < 0) o.push([i, j]); })); return o; };
          return { X, W1, W2, Y, H, Z, YA: cols(Y, 0, 2), YB: cols(Y, 2, 4), HA, HB, W1A: cols(W1, 0, 2), W1B: cols(W1, 2, 4), W2A, W2B, ZA, ZB, ZS, XA, XB, W1rA: rows(W1, 0, 2), W1rB: rows(W1, 2, 4), P0, P1, fP0: relu(P0), fP1: relu(P1), lhs, rhs, diff, negA: neg(cols(Y, 0, 2)), negB: neg(cols(Y, 2, 4)), ok: ZS.every((row, i) => row.every((v, j) => v === Z[i][j])) };
        }
        return null;
      }
      let seed = 11;
      let D = gen(seed);
      const steps = [
        { t: 'The whole computation on one GPU', d: 'Two tokens (<b>X</b>, 2×4), a layer that expands to 4 (<b>W<sub>1</sub></b>), the nonlinearity <b>f</b> (ReLU), and the layer that projects back (<b>W<sub>2</sub></b>). The result <b>Z</b> is the reference that TP has to reproduce exactly.' },
        { t: 'Column-parallel: W₁ cut into columns', d: 'Node A stores columns 0–1 of W<sub>1</sub> and node B, columns 2–3. The input <b>X</b> is whole (replicated) on both nodes.' },
        { t: 'Each node computes its half of Y, without talking', d: 'Each column of Y depends only on X and on the matching column of W<sub>1</sub>. Y<sub>A</sub> and Y<sub>B</sub> come out without any message exchange. The highlight shows the row of X and the column of W<sub>1</sub> behind each cell.' },
        { t: 'The nonlinearity is local', d: 'f acts elementwise: f(Y<sub>A</sub>) only needs Y<sub>A</sub>. The outlined cells were negative and became 0. Nothing has been communicated so far.' },
        { t: 'Row-parallel: W₂ cut into rows', d: 'W<sub>2</sub> is cut along the rows that match the columns of H each node has. Each node produces a <b>whole</b> Z (2×4), but a <b>partial</b> one: the other node\'s contribution is missing.' },
        { t: 'All-reduce: adding up the partials', d: 'Z<sub>A</sub> + Z<sub>B</sub> = Z on both nodes. It\'s the pair\'s only communication, and the result is identical to the single GPU\'s, cell by cell.' },
        { t: 'What if it were row → col?', d: 'With the first layer cut into rows, each node would have a partial sum P<sub>A</sub>, P<sub>B</sub>. Since f(P<sub>A</sub>) + f(P<sub>B</sub>) ≠ f(P<sub>A</sub> + P<sub>B</sub>) (cells in red), you would have to add <i>before</i> f: two collectives per pair instead of one.' },
      ];
      let H2 = null;
      const st = UI.stepper(fig, steps, { autoplay: 6500, onStep: () => { if (H2) H2.redraw(); } });
      const ctl = UI.controls(fig);
      UI.button(ctl, '🎲 new numbers', () => { seed += 17; const n = gen(seed); if (n) D = n; st.t0 = performance.now(); if (H2) H2.redraw(); });
      const C0 = V.C();
      UI.legend(ctl, [[C0.a, 'node A'], [C0.b, 'node B'], [C0.muted, 'replicated / reference'], [C0.c, 'highlight']]);

      function mat(g, M, x, y, cs, col, o = {}) {
        const C = V.C();
        const R = M.length, K = M[0].length, gp = Math.max(1.5, cs * 0.08);
        const vmax = Math.max(1, ...M.flat().map((v) => Math.abs(v)));
        let idx = 0;
        if (o.alpha != null) g.globalAlpha = o.alpha;
        for (let i = 0; i < R; i++) for (let j = 0; j < K; j++, idx++) {
          const cx = x + j * (cs + gp), cy = y + i * (cs + gp);
          const shown = o.reveal == null || idx < o.reveal;
          const v = M[i][j];
          V.box(g, cx, cy, cs, cs, { r: Math.min(4, cs / 5), fill: shown ? V.alpha(col, 0.1 + 0.34 * Math.min(1, Math.abs(v) / vmax)) : V.alpha(C.muted, 0.06), stroke: V.alpha(col, shown ? 0.75 : 0.3), lw: 1 });
          V.text(g, shown ? fi(v) : '·', cx + cs / 2, cy + cs / 2 + 0.5, { size: Math.max(9, Math.min(13, cs * 0.45)), font: 'mono', weight: 600, align: 'center', base: 'middle', color: shown ? C.ink : C.muted });
        }
        const Wd = K * cs + (K - 1) * gp, Hh = R * cs + (R - 1) * gp;
        const hl = (hx, hy, hw, hh, c) => { g.strokeStyle = c; g.lineWidth = 2.2; V.rr(g, hx - 2, hy - 2, hw + 4, hh + 4, 5); g.stroke(); };
        if (o.hiRow != null) hl(x, y + o.hiRow * (cs + gp), Wd, cs, V.C().c);
        if (o.hiCol != null) hl(x + o.hiCol * (cs + gp), y, cs, Hh, V.C().c);
        (o.hi || []).forEach(([i, j]) => hl(x + j * (cs + gp), y + i * (cs + gp), cs, cs, o.hiColor || V.C().c));
        if (o.alpha != null) g.globalAlpha = 1;
        if (o.label) lab(g, o.label, x + Wd / 2, y - 7, { size: Math.min(13, Math.max(10.5, cs * 0.46)), color: o.labColor || C.ink });
        if (o.sub) V.text(g, o.sub, x + Wd / 2, y + Hh + 13, { size: 11, font: 'mono', color: C.muted, align: 'center' });
        return { w: Wd, h: Hh };
      }
      const gpOf = (cs) => Math.max(1.5, cs * 0.08);
      const itW = (it, cs) => (it.M ? it.M[0].length * cs + (it.M[0].length - 1) * gpOf(cs) : (it.opW || 0.9) * cs);
      const itH = (it, cs) => (it.M ? it.M.length * cs + (it.M.length - 1) * gpOf(cs) : cs);
      function fitCs(items, availW, csMax) { let u = 0; items.forEach((it) => { u += it.M ? it.M[0].length * 1.08 : it.opW || 0.9; }); u += (items.length - 1) * 0.3; return Math.max(12, Math.min(csMax, availW / u)); }
      function eqH(items, cs) { return Math.max(...items.map((it) => itH(it, cs))) + 34; }
      function eqDraw(g, items, cx, yTop, cs) {
        const C = V.C();
        const ws = items.map((it) => itW(it, cs));
        const wsum = ws.reduce((a, b) => a + b, 0) + (items.length - 1) * cs * 0.3;
        const hm = Math.max(...items.map((it) => itH(it, cs)));
        let x = cx - wsum / 2;
        const mid = yTop + 18 + hm / 2;
        items.forEach((it, k) => {
          if (it.M) mat(g, it.M, x, mid - itH(it, cs) / 2, cs, it.col, it);
          else V.text(g, it.op, x + ws[k] / 2, mid + 1, { size: Math.min(20, cs * (it.opSize || 0.8)), weight: 700, align: 'center', base: 'middle', color: it.color || C.muted, font: it.font || 'sans' });
          x += ws[k] + cs * 0.3;
        });
      }
      function panel(g, P, r, title) {
        const C = V.C(), col = r == null ? C.muted : RC(C, r);
        V.box(g, P.x, P.y, P.w, P.h, { r: 12, fill: V.alpha(col, 0.05), stroke: V.alpha(col, 0.55), lw: 1.2 });
        chip(g, title, P.x + 10, P.y + 16, col, { size: 11, h: 20 });
      }
      function drawRows(g, P, rowsIt, csMax) {
        const inner = P.w - 20;
        let cs = Math.min(...rowsIt.map((items) => fitCs(items, inner, csMax)));
        let tot = rowsIt.reduce((a, items) => a + eqH(items, cs), 0);
        const avail = P.h - 34;
        if (tot > avail) { cs *= avail / tot; tot = rowsIt.reduce((a, items) => a + eqH(items, cs), 0); }
        let y = P.y + 30 + Math.max(0, (avail - tot) / 2);
        rowsIt.forEach((items) => { eqDraw(g, items, P.x + P.w / 2, y, cs); y += eqH(items, cs); });
        return cs;
      }
      H2 = V.fig(fig, (g, w, h) => {
        if (w < 140 || h < 100) return;
        const C = V.C();
        if (!D) return;
        const i = st.i, el = st.elapsed();
        const wide = w >= 640, pad = 10;
        let PA, PB;
        const stripH = i === 6 ? (wide ? 150 : 170) : 0;
        if (wide) { const pw = (w - 3 * pad) / 2; PA = { x: pad, y: pad, w: pw, h: h - 2 * pad - stripH - (stripH ? pad : 0) }; PB = { x: 2 * pad + pw, y: pad, w: pw, h: PA.h }; }
        else { const ph = (h - 3 * pad - stripH - (stripH ? pad : 0)) / 2; PA = { x: pad, y: pad, w: w - 2 * pad, h: ph }; PB = { x: pad, y: 2 * pad + ph, w: w - 2 * pad, h: ph }; }
        const nA = C.a, nB = C.b, nN = C.muted;
        const csMax = 30;
        const rev = (n, per) => Math.floor(el / per) + 1;
        if (i === 0) {
          const P = { x: pad, y: pad, w: w - 2 * pad, h: h - 2 * pad };
          panel(g, P, null, 'single GPU (reference)');
          const r1 = [{ M: D.X, col: nN, label: 'X', sub: '[2×4]' }, { op: '·' }, { M: D.W1, col: nN, label: 'W₁', sub: '[4×4]' }];
          const r2 = [{ op: '=' }, { M: D.Y, col: nN, label: 'Y', sub: '[2×4]' }];
          const r3 = [{ op: 'f(', opW: 1.1, font: 'mono' }, { M: D.Y, col: nN, label: 'Y' }, { op: ')', opW: 0.5, font: 'mono' }, { op: '·' }, { M: D.W2, col: nN, label: 'W₂', sub: '[4×4]' }];
          const r4 = [{ op: '=' }, { M: D.Z, col: C.c, label: 'Z', sub: 'reference' }];
          drawRows(g, P, wide ? [r1.concat(r2), r3.concat(r4)] : [r1, r2, r3, r4], csMax);
          return;
        }
        panel(g, PA, 0, 'node A · rank 0');
        panel(g, PB, 1, 'node B · rank 1');
        const both = (fn) => { fn(PA, 0, nA); fn(PB, 1, nB); };
        if (i === 1) {
          const a = smooth(el / 0.7);
          both((P, r, col) => drawRows(g, P, [[{ M: D.X, col: nN, label: 'X', sub: 'replicated' }, { op: '·' }, { M: r ? D.W1B : D.W1A, col, label: 'W₁|' + (r ? 'B' : 'A'), sub: r ? 'columns 2–3' : 'columns 0–1', alpha: a }]], csMax));
        } else if (i === 2) {
          const k = Math.min(4, rev(4, 0.5));
          const cur = Math.min(3, k - 1), hr = Math.floor(cur / 2), hc = cur % 2;
          both((P, r, col) => drawRows(g, P, [[{ M: D.X, col: nN, label: 'X', hiRow: k <= 4 && el < 2.4 ? hr : null }, { op: '·' }, { M: r ? D.W1B : D.W1A, col, label: 'W₁|' + (r ? 'B' : 'A'), hiCol: el < 2.4 ? hc : null }, { op: '=' }, { M: r ? D.YB : D.YA, col, label: 'Y|' + (r ? 'B' : 'A'), sub: '[2×2]', reveal: k }]], csMax));
        } else if (i === 3) {
          const on = el > 0.5;
          both((P, r, col) => drawRows(g, P, [[{ op: 'f(', opW: 1.1, font: 'mono' }, { M: r ? D.YB : D.YA, col, label: 'Y|' + (r ? 'B' : 'A'), hi: r ? D.negB : D.negA }, { op: ')', opW: 0.5, font: 'mono' }, { op: '=' }, { M: r ? D.HB : D.HA, col, label: 'H|' + (r ? 'B' : 'A'), sub: 'local', hi: on ? (r ? D.negB : D.negA) : [], reveal: on ? 4 : 0 }]], csMax));
        } else if (i === 4) {
          const k = rev(8, 0.3);
          both((P, r, col) => drawRows(g, P, [[{ M: r ? D.HB : D.HA, col, label: 'H|' + (r ? 'B' : 'A') }, { op: '·' }, { M: r ? D.W2B : D.W2A, col, label: 'W₂|' + (r ? 'B' : 'A'), sub: r ? 'rows 2–3' : 'rows 0–1' }], [{ op: '=' }, { M: r ? D.ZB : D.ZA, col, label: 'Z|' + (r ? 'B' : 'A'), sub: 'whole, but partial', reveal: k }]], csMax));
        } else if (i === 5) {
          const p = smooth((el - 0.2) / 1.1);
          both((P, r, col) => drawRows(g, P, [[{ M: D.ZA, col: nA, label: 'Z|A', sub: 'from node A' }, { op: '+', color: C.c }, { M: D.ZB, col: nB, label: 'Z|B', sub: 'from node B' }], [{ op: '=' }, { M: D.ZS, col: C.c, label: 'Z', sub: D.ok ? '✓ matches the reference' : '', reveal: p >= 1 ? 8 : 0 }]], csMax));
          // all-reduce arrows between the panels
          const n = 5;
          for (let k = 0; k < n; k++) {
            const f = ((el * 0.6 + k / n) % 1);
            if (wide) {
              const yy = PA.y + PA.h * (0.25 + 0.5 * (k / (n - 1)));
              const x0 = PA.x + PA.w - 6, x1 = PB.x + 6;
              V.dot(g, V.lerp(x0, x1, f), yy, 3.2, C.a);
              V.dot(g, V.lerp(x1, x0, f), yy + 6, 3.2, C.b);
            } else {
              const xx = PA.x + PA.w * (0.2 + 0.6 * (k / (n - 1)));
              const y0 = PA.y + PA.h - 4, y1 = PB.y + 4;
              V.dot(g, xx, V.lerp(y0, y1, f), 3.2, C.a);
              V.dot(g, xx + 6, V.lerp(y1, y0, f), 3.2, C.b);
            }
          }
          if (wide) chip(g, 'all-reduce', w / 2, PA.y + PA.h - 18, C.c, { align: 'center', size: 11, h: 18, fa: 0.35 });
          else chip(g, 'all-reduce', w / 2, PB.y - pad / 2, C.c, { align: 'center', size: 11, h: 18 });
        } else if (i === 6) {
          both((P, r, col) => drawRows(g, P, [[{ M: r ? D.XB : D.XA, col, label: 'X|' + (r ? 'B' : 'A'), sub: r ? 'columns 2–3' : 'columns 0–1' }, { op: '·' }, { M: r ? D.W1rB : D.W1rA, col, label: 'W₁', sub: r ? 'rows 2–3' : 'rows 0–1' }], [{ op: '=' }, { M: r ? D.P1 : D.P0, col, label: 'P|' + (r ? 'B' : 'A'), sub: 'partial' }]], csMax));
          const S = { x: pad, y: h - pad - stripH, w: w - 2 * pad, h: stripH };
          V.box(g, S.x, S.y, S.w, S.h, { r: 12, fill: V.alpha(C.red || C.c, 0.05), stroke: V.alpha(C.red || C.c, 0.5), lw: 1.2 });
          const on = el > 0.8;
          drawRows(g, { x: S.x, y: S.y - 22, w: S.w, h: S.h + 20 }, [[{ M: D.lhs, col: nN, label: 'f(P|A) + f(P|B)', hi: on ? D.diff : [], hiColor: C.red || C.c }, { op: '≠', color: C.red || C.c, opSize: 1 }, { M: D.rhs, col: nN, label: 'f(P|A + P|B)', hi: on ? D.diff : [], hiColor: C.red || C.c }]], 26);
        }
      });
    })();

    /* ========================= F3 — subway map of a block under TP =========== */
    (function () {
      const fig = root.querySelector('#fig-metro');
      if (!fig) return;
      adaptH(fig, (w) => (w < 600 ? 720 : 420));
      const ST = { model: 'qwen' };
      const MODELS = {
        glm: {
          st: [
            { k: 'rep', n: 'mHC + norm', s: 'replica' },
            { k: 'col', n: 'in_proj / q_b', s: 'col · 32 heads' },
            { k: 'loc', n: 'KDA / MLA', s: 'per head' },
            { k: 'row', n: 'o_proj / out', s: 'row' },
            { k: 'ar', n: '⊕', s: 'all-reduce' },
            { k: 'rep', n: 'router', s: 'replica' },
            { k: 'col', n: 'gate / up', s: 'col · 1024' },
            { k: 'loc', n: 'SwiGLU', s: 'local' },
            { k: 'row', n: 'down', s: 'row · 1024' },
            { k: 'ar', n: '⊕', s: 'all-reduce' },
          ],
          br: [['mixer (KDA/MLA)', 45], ['FFN / MoE', 45]],
          tot: '≈ 90 +', note: 'Our interpretation: 2 per layer across the 45 layers. The repo documents 1 all-reduce per MoE layer and KDA\'s row-parallel out_proj; embedding, logits, and DFlash2 are not in this count.',
          rep: 'replicas: MLA\'s c_kv latent, indexer, KDA\'s f_a/g_a',
        },
        qwen: {
          st: [
            { k: 'rep', n: 'HC (4 branches)', s: 'replica 1.19 GiB' },
            { k: 'col', n: 'qkvz / qkv+gate', s: 'col' },
            { k: 'loc', n: 'GDN / QSA', s: '24 / 12 heads per GPU' },
            { k: 'row', n: 'out_proj', s: 'row · 3072' },
            { k: 'ar', n: '⊕', s: 'all-reduce' },
            { k: 'rep', n: 'router', s: 'replica' },
            { k: 'ep', n: 'local experts', s: 'EP · 256 of 512' },
            { k: 'col', n: 'shared expert', s: 'col → row' },
            { k: 'ar', n: '⊕', s: 'all-reduce' },
          ],
          br: [['GDN out_proj', 36], ['attention o_proj', 12], ['MoE', 48], ['logits (gather)', 1], ['drafter', 9]],
          tot: '≈ 106', note: 'Recipe audit (batch 1, MTP=3): 5–20 KB messages, 20–30 µs each, 2–3 ms per step (6–8%); does not shrink with the batch.',
          rep: 'replicas read per GPU at every step: HC 1.19 GiB · router 0.12 · indexer 0.04 GiB',
        },
        ds: {
          st: [
            { k: 'rep', n: 'mHC + norm', s: 'replica' },
            { k: 'rep', n: 'wqa + wkv', s: 'replica (latent)' },
            { k: 'col', n: 'wq_b', s: 'col · 32 heads' },
            { k: 'loc', n: 'attention', s: 'KV and indexer replicated' },
            { k: 'col', n: 'wo_a', s: '4 groups' },
            { k: 'row', n: 'wo_b', s: 'row' },
            { k: 'ar', n: '⊕', s: 'all-reduce' },
            { k: 'rep', n: 'gate (fp32)', s: 'replica' },
            { k: 'col', n: 'experts w1/w3', s: 'TP in the expert' },
            { k: 'row', n: 'experts w2', s: 'row' },
            { k: 'ar', n: '⊕', s: 'all-reduce' },
          ],
          br: [['wo_b', 43], ['MoE', 43], ['draft', 6], ['embedding', 2], ['logits (gather)', 2], ['Markov loop', 12]],
          tot: '≈ 108', note: 'Recipe audit (TP=2, RoCE without GPUDirect): 30–50 µs each, 3–5 ms per decode step.',
          rep: 'replicas: fused_wqa_wkv, the indexer, and the entire latent KV (1 head)',
        },
      };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Recipe', options: [{ v: 'glm', label: 'GLM-5.3' }, { v: 'qwen', label: 'Qwen3.8' }, { v: 'ds', label: 'DeepSeek-V4' }], value: 'qwen', onChange: (v) => { ST.model = v; } });
      const C0 = V.C();
      UI.legend(ctl, [[C0.a, 'node A'], [C0.b, 'node B'], [C0.muted, 'replica (hatched)'], [C0.c, '⊕ all-reduce']]);
      function stationBox(g, x, y, bw, bh, k, r, C, t, lbl) {
        const col = RC(C, r);
        if (k === 'rep') {
          V.box(g, x - bw / 2, y - bh / 2, bw, bh, { r: 6, fill: V.alpha(C.muted, 0.1), stroke: V.alpha(C.muted, 0.7), lw: 1.2 });
          hatch(g, x - bw / 2 + 2, y - bh / 2 + 2, bw - 4, bh - 4, V.alpha(C.muted, 0.45), 6, t * 8);
        } else if (k === 'col' || k === 'row') {
          V.box(g, x - bw / 2, y - bh / 2, bw, bh, { r: 6, fill: V.alpha(col, 0.08), stroke: col, lw: 1.3 });
          g.save(); V.rr(g, x - bw / 2, y - bh / 2, bw, bh, 6); g.clip();
          g.fillStyle = V.alpha(col, 0.55);
          if (k === 'col') g.fillRect(x - bw / 2 + (r ? bw / 2 : 0), y - bh / 2, bw / 2, bh);
          else g.fillRect(x - bw / 2, y - bh / 2 + (r ? bh / 2 : 0), bw, bh / 2);
          g.restore();
        } else if (k === 'ep') {
          V.box(g, x - bw / 2, y - bh / 2, bw, bh, { r: 6, fill: V.alpha(col, 0.55), stroke: col, lw: 1.3 });
          if (lbl) V.text(g, r ? '256–511' : '0–255', x, y + 0.5, { size: 11, font: 'mono', weight: 700, align: 'center', base: 'middle', color: C.ink });
        } else if (k === 'loc') {
          V.box(g, x - bh / 2, y - bh / 2, bh, bh, { r: bh / 2, fill: V.alpha(col, 0.35), stroke: col, lw: 1.3 });
        }
      }
      V.fig(fig, (g, w, h, t) => {
        if (w < 140 || h < 100) return;
        const C = V.C(), M = MODELS[ST.model], stn = M.st, n = stn.length;
        const narrow = w < 600, pad = 14;
        const brH = narrow ? 210 : 150;
        const regH = h - brH - 10;
        // train state: full / fatia (slice) / parcial (partial sum), internal keys
        const stateAfter = []; let s = 'full';
        stn.forEach((q) => { if (q.k === 'col' || q.k === 'loc') s = s === 'parcial' ? 'parcial' : 'fatia'; if (q.k === 'row' || q.k === 'ep') s = 'parcial'; if (q.k === 'ar') s = 'full'; stateAfter.push(s); });
        if (!narrow) {
          const yA = 58, yB = regH - 58, mid = (yA + yB) / 2;
          const x0 = 70, x1 = w - 22, sp = (x1 - x0) / n;
          const xs = stn.map((_, i) => x0 + (i + 0.5) * sp);
          const ars = xs.filter((_, i) => stn[i].k === 'ar');
          const d = sp * 0.62;
          const ty = (x, base) => { let y = base; ars.forEach((xa) => { const u = 1 - Math.abs(x - xa) / d; if (u > 0) y = V.lerp(base, mid + (base < mid ? -7 : 7), smooth(u)); }); return y; };
          V.text(g, 'node A', 14, yA + 4, { size: 12, weight: 700, color: C.a, font: 'mono' });
          V.text(g, 'node B', 14, yB + 4, { size: 12, weight: 700, color: C.b, font: 'mono' });
          [[yA, C.a], [yB, C.b]].forEach(([base, col]) => {
            g.beginPath();
            for (let x = x0 - 12; x <= x1; x += 3) { const y = ty(x, base); if (x === x0 - 12) g.moveTo(x, y); else g.lineTo(x, y); }
            g.strokeStyle = V.alpha(col, 0.55); g.lineWidth = 5; g.lineCap = 'round'; g.stroke(); g.lineCap = 'butt';
          });
          // trains
          const u = (t * 0.075) % 1;
          const tx = V.lerp(x0 - 10, x1, u);
          let idx = -1; xs.forEach((xx, i) => { if (tx > xx) idx = i; });
          const tst = idx < 0 ? 'full' : stateAfter[idx];
          // stations
          const bw = Math.min(sp - 12, 58), bh = 24;
          xs.forEach((xx, i) => {
            const q = stn[i];
            if (q.k === 'ar') {
              const on = Math.abs(tx - xx) < sp * 0.35;
              plus(g, xx, mid, 12, C.c, on);
              V.text(g, 'all-reduce', xx, mid + 30, { size: 11, font: 'mono', align: 'center', color: C.c });
              return;
            }
            stationBox(g, xx, yA, bw, bh, q.k, 0, C, t, bw > 50);
            stationBox(g, xx, yB, bw, bh, q.k, 1, C, t, bw > 50);
            const nl = wrapLines(g, q.n, sp - 6, 11, 'sans', 700);
            nl.slice(0, 2).forEach((ln, k) => V.text(g, ln, xx, mid - 6 + k * 13 - (nl.length > 1 ? 6 : 0), { size: 11, weight: 700, align: 'center', color: C.ink }));
            const sl = wrapLines(g, q.s, sp - 4, 11, 'mono', 400);
            sl.slice(0, 3).forEach((ln, k) => V.text(g, ln, xx, yB + 30 + k * 13, { size: 11, font: 'mono', align: 'center', color: C.muted }));
          });
          [[yA, 0], [yB, 1]].forEach(([base, r]) => {
            const col = RC(C, r), y = ty(tx, base);
            const L = tst === 'fatia' ? 16 : 30;
            V.glow(g, tx, y, 18, col, 0.6);
            if (tst === 'parcial') { V.box(g, tx - L / 2, y - 6, L, 12, { r: 4, fill: V.alpha(col, 0.2), stroke: col, lw: 1.4 }); V.text(g, 'Σ', tx, y + 0.5, { size: 11, weight: 700, align: 'center', base: 'middle', color: C.ink }); }
            else V.box(g, tx - L / 2, y - 6, L, 12, { r: 4, fill: V.alpha(col, 0.9), stroke: col, lw: 1 });
          });
          V.text(g, 'full train = whole vector · short = slice · Σ = partial sum', w - 14, 16, { size: 11, font: 'mono', align: 'right', color: C.muted });
        } else {
          const xA = pad + 16, xB = w - pad - 16, midx = w / 2;
          const y0 = 34, y1 = regH - 8, sp = (y1 - y0) / n;
          const ys = stn.map((_, i) => y0 + (i + 0.5) * sp);
          const ars = ys.filter((_, i) => stn[i].k === 'ar');
          const d = sp * 0.75;
          const tx = (y, base) => { let x = base; ars.forEach((ya) => { const u = 1 - Math.abs(y - ya) / d; if (u > 0) x = V.lerp(base, midx + (base < midx ? -18 : 18), smooth(u)); }); return x; };
          V.text(g, 'node A', xA, 18, { size: 11, weight: 700, color: C.a, font: 'mono', align: 'center' });
          V.text(g, 'node B', xB, 18, { size: 11, weight: 700, color: C.b, font: 'mono', align: 'center' });
          [[xA, C.a], [xB, C.b]].forEach(([base, col]) => {
            g.beginPath();
            for (let y = y0 - 6; y <= y1; y += 3) { const x = tx(y, base); if (y === y0 - 6) g.moveTo(x, y); else g.lineTo(x, y); }
            g.strokeStyle = V.alpha(col, 0.55); g.lineWidth = 5; g.lineCap = 'round'; g.stroke(); g.lineCap = 'butt';
          });
          const u = (t * 0.075) % 1;
          const ty = V.lerp(y0 - 4, y1, u);
          let idx = -1; ys.forEach((yy, i) => { if (ty > yy) idx = i; });
          const tst = idx < 0 ? 'full' : stateAfter[idx];
          const bw = 24, bh = Math.min(20, sp - 8);
          ys.forEach((yy, i) => {
            const q = stn[i];
            if (q.k === 'ar') { plus(g, midx, yy, 11, C.c, Math.abs(ty - yy) < sp * 0.4); V.text(g, 'all-reduce', midx + 18, yy + 4, { size: 11, font: 'mono', color: C.c }); return; }
            stationBox(g, xA, yy, bw, bh, q.k, 0, C, t, false);
            stationBox(g, xB, yy, bw, bh, q.k, 1, C, t, false);
            V.text(g, q.n, midx, yy - 2, { size: 11.5, weight: 700, align: 'center', color: C.ink, maxW: xB - xA - 50 });
            V.text(g, q.s, midx, yy + 12, { size: 11, font: 'mono', align: 'center', color: C.muted, maxW: xB - xA - 50 });
          });
          [[xA, 0], [xB, 1]].forEach(([base, r]) => {
            const col = RC(C, r), x = tx(ty, base);
            const L = tst === 'fatia' ? 10 : 18;
            V.glow(g, x, ty, 14, col, 0.6);
            if (tst === 'parcial') { V.box(g, x - 6, ty - L / 2, 12, L, { r: 4, fill: V.alpha(col, 0.2), stroke: col, lw: 1.4 }); }
            else V.box(g, x - 6, ty - L / 2, 12, L, { r: 4, fill: V.alpha(col, 0.9), stroke: col });
          });
        }
        // count panel
        const by = h - brH;
        V.box(g, pad, by, w - 2 * pad, brH - 8, { r: 12, fill: V.alpha(C.muted, 0.07), stroke: V.alpha(C.muted, 0.28), lw: 1 });
        V.text(g, 'collectives per decode step (TP=2)', pad + 12, by + 22, { size: 12, weight: 700, color: C.ink });
        V.text(g, M.tot, w - pad - 12, by + 22, { size: 16, weight: 700, color: C.c, align: 'right', font: 'display' });
        const tot = M.br.reduce((a, b) => a + b[1], 0);
        const bx = pad + 12, bwid = w - 2 * pad - 24, bY = by + 34, bH = 20;
        let xx = bx;
        const cols = [C.a, C.b, C.d, C.e, C.f, C.c];
        M.br.forEach(([nm, v], k) => {
          const ww = (v / tot) * bwid;
          V.box(g, xx, bY, Math.max(2, ww - 2), bH, { r: 4, fill: V.alpha(cols[k % 6], 0.7) });
          if (ww > 26) V.text(g, String(v), xx + ww / 2 - 1, bY + bH / 2 + 0.5, { size: 11, font: 'mono', weight: 700, align: 'center', base: 'middle', color: C.dark ? '#08140e' : '#fff' });
          xx += ww;
        });
        // bar legend
        let lx = bx, ly = bY + bH + 18;
        M.br.forEach(([nm, v], k) => {
          const s = nm + ' ' + v;
          const ww = tw(g, s, 11, 'mono', 400) + 22;
          if (lx + ww > bx + bwid) { lx = bx; ly += 16; }
          V.box(g, lx, ly - 8, 9, 9, { r: 2, fill: V.alpha(cols[k % 6], 0.8) });
          V.text(g, s, lx + 13, ly, { size: 11, font: 'mono', color: C.ink });
          lx += ww;
        });
        ly = wrap(g, M.note, bx, ly + 18, bwid, { size: 11, color: C.muted, lh: 14 });
        wrap(g, M.rep, bx, ly + 2, bwid, { size: 11, color: C.ink, lh: 14 });
      });
    })();

    /* ================== F4 — cutting quantized weights ====================== */
    (function () {
      const fig = root.querySelector('#fig-quantcut');
      if (!fig) return;
      const ST0 = { pre: 'qmtp', p: 2 };
      const reH = adaptH(fig, (w) => {
        const nar = w < 600;
        if (ST0.pre === 'exl3' || ST0.pre === 'glm') return nar ? 490 : 440;
        if (ST0.pre === 'kda') return (nar ? 320 : 250) + ST0.p * (nar ? 44 : 50);
        return nar ? 400 : 350;
      });
      const PRE = {
        qmtp: { label: 'Qwen · MTP (FP8 128×128) · 640', n: 640, fmt: 'fp8b' },
        qvit: { label: 'Qwen · vision (NVFP4) · 4304', n: 4304, fmt: 'nvfp4' },
        glm: { label: 'GLM · EXL3 expert · 2048', n: 2048, fmt: 'exl3' },
        ds: { label: 'DS · MXFP4 expert · 2048', n: 2048, fmt: 'mxfp4' },
        kda: { label: 'GLM · KDA in_proj', fmt: 'segs' },
      };
      const ST = { pre: 'qmtp', p: 2, mode: 'col' };
      const ctl = UI.controls(fig);
      const selT = UI.select(ctl, { label: 'Tensor', options: Object.keys(PRE).map((k) => ({ v: k, label: PRE[k].label })), value: 'qmtp', onChange: (v) => { ST.pre = v; ST0.pre = v; if (v === 'qvit') segM.set('row'); if (reH) reH(); } });
      selT.el.style.maxWidth = '100%'; selT.el.style.minWidth = '0';
      const selE = selT.el.querySelector('select'); if (selE) { selE.style.maxWidth = '100%'; selE.style.minWidth = '0'; }
      UI.seg(ctl, { label: 'TP', options: [2, 3, 4], value: 2, onChange: (v) => { ST.p = v; ST0.p = v; if (reH) reH(); } });
      const segM = UI.seg(ctl, { label: 'Cut', options: [{ v: 'col', label: 'col (outputs N)' }, { v: 'row', label: 'row (inputs K)' }], value: 'col', onChange: (v) => { ST.mode = v; } });
      const align = (fmt, mode) => (fmt === 'fp8b' || fmt === 'exl3' ? 128 : fmt === 'nvfp4' ? (mode === 'row' ? 16 : 1) : fmt === 'mxfp4' ? (mode === 'row' ? 32 : 1) : 1);
      const BLK = { fp8b: '128×128 block, one scale per block', exl3: '16×16 tiles · Hadamard in blocks of 128', nvfp4: 'block of 16 values in K · E4M3 scale', mxfp4: 'block of 32 values in K · E8M0 scale' };
      function note(pre, p, mode, ok) {
        if (pre === 'qmtp') return ok ? 'slice aligned to the blocks' : 'The recipe uses EP: 256 whole experts per GPU, and the 640 width stays intact.';
        if (pre === 'qvit') return mode === 'col' ? 'Cut in N: the scale blocks run along K and are not split (kernel constraints aside).' : ok ? 'aligned slice' : 'The recipe replicates the tower: --mm-encoder-tp-mode data (weights breaks at load time).';
        if (pre === 'glm') return ok ? (p === 2 ? 'This is the recipe\'s TP2: shard_exl3_col/row.' : 'Divides cleanly (512 per rank).') : 'On TP3 the recipe uses EP: 96 of the 288 experts per rank. The trellis is not padded.';
        if (pre === 'ds') return ok ? (mode === 'col' ? 'Cut in N: scale blocks run along K and are not split.' : 'down with 1024 = 32 blocks of 32 per rank.') : 'The TP3 patch pads 2048 → 2112 (704 per rank).';
        return '';
      }
      function segsFor(p) {
        const heads = p === 3 ? 66 : 64;
        return { heads, segs: [{ n: 'q', s: heads * 128, sh: 1 }, { n: 'k', s: heads * 128, sh: 1 }, { n: 'v', s: heads * 128, sh: 1 }, { n: 'β', s: heads, sh: 1 }, { n: 'f_a', s: 128, sh: 0 }, { n: 'g_a', s: 128, sh: 0 }] };
      }
      const rLocal = UI.readout(ctl, 'local slice', '—');
      const rVer = UI.readout(ctl, 'verdict', '—');
      V.fig(fig, (g, w, h, t) => {
        if (w < 140 || h < 100) return;
        const C = V.C(), pr = PRE[ST.pre], p = ST.p, pad = 16;
        const narrow = w < 600;
        const x0 = pad, x1 = w - pad, RW = x1 - x0;
        if (pr.fmt === 'segs') {
          const { heads, segs } = segsFor(p);
          const gl = segs.reduce((a, q) => a + q.s, 0);
          const local = segs.reduce((a, q) => a + (q.sh ? q.s / p : q.s), 0);
          const qual = { 2: 12576, 3: 8726 };
          const ok = qual[p] === local;
          rLocal.set(nf(local) + ' × 4096');
          rVer.set(ok ? '✓ qualified shape' : '✗ not on the list');
          const oy = narrow ? 16 : 0;
          V.text(g, (narrow ? 'fused output · ' : 'output dimension of the fused projection (') + heads + ' heads' + (p === 3 ? ', padded from 64' : '') + (narrow ? '' : ')'), x0, 24, { size: narrow ? 11 : 12, color: C.muted, font: 'mono', maxW: narrow ? RW : RW - 160 });
          const minW = narrow ? 26 : 34;
          const shW = RW - 3 * minW;
          const widths = segs.map((q) => (q.n === 'β' || !q.sh ? minW : (q.s / (3 * heads * 128)) * (shW - 6)));
          let x = x0;
          const y = 40 + oy, bh = 30;
          segs.forEach((q, k) => {
            const ww = widths[k];
            if (q.sh) {
              for (let r = 0; r < p; r++) { g.fillStyle = V.alpha(RC(C, r), 0.45); g.fillRect(x + (ww * r) / p, y, ww / p, bh); }
              for (let r = 1; r < p; r++) V.line(g, x + (ww * r) / p, y - 3, x + (ww * r) / p, y + bh + 3, { color: C.c, lw: 1.5 });
            } else { V.box(g, x, y, ww - 2, bh, { r: 3, fill: V.alpha(C.muted, 0.12) }); hatch(g, x, y, ww - 2, bh, V.alpha(C.muted, 0.6), 5, t * 6); }
            V.box(g, x, y, ww - 2, bh, { r: 3, stroke: V.alpha(C.ink, 0.45), lw: 1 });
            V.text(g, q.n, x + ww / 2 - 1, y + bh / 2 + 0.5, { size: 11, font: 'mono', weight: 700, align: 'center', base: 'middle', color: C.ink });
            V.text(g, nf(q.s), x + ww / 2 - 1, y + bh + 14, { size: 11, font: 'mono', align: 'center', color: C.muted, maxW: ww });
            x += ww;
          });
          V.text(g, 'global: ' + nf(gl) + ' rows', narrow ? x0 : x1, narrow ? 24 + oy : 24, { size: 11, font: 'mono', align: narrow ? 'left' : 'right', color: C.muted });
          // rows per rank
          const ry0 = 104 + oy, rh = narrow ? 44 : 50;
          for (let r = 0; r < p; r++) {
            const yy = ry0 + r * rh;
            V.text(g, 'rank ' + r, x0, yy + 16, { size: 11, font: 'mono', weight: 700, color: RC(C, r) });
            let xx = x0 + (narrow ? 50 : 62);
            const avail = x1 - xx - (narrow ? 70 : 110);
            const parts = segs.map((q) => ({ n: q.n + (q.sh ? '' : ''), s: q.sh ? q.s / p : q.s, sh: q.sh }));
            const tot = parts.reduce((a, q) => a + q.s, 0);
            parts.forEach((q) => {
              const ww = Math.max(q.sh ? 8 : narrow ? 20 : 26, (q.s / tot) * avail * (q.sh ? 0.93 : 1));
              if (q.sh) V.box(g, xx, yy, ww - 2, 24, { r: 3, fill: V.alpha(RC(C, r), 0.5), stroke: RC(C, r), lw: 1 });
              else { V.box(g, xx, yy, ww - 2, 24, { r: 3, fill: V.alpha(C.muted, 0.1), stroke: V.alpha(C.muted, 0.7), lw: 1 }); hatch(g, xx, yy, ww - 2, 24, V.alpha(C.muted, 0.55), 5, t * 6); }
              if (ww > 20) V.text(g, q.n, xx + ww / 2 - 1, yy + 12.5, { size: 11, font: 'mono', weight: 700, align: 'center', base: 'middle', color: C.ink });
              xx += ww;
            });
            V.text(g, '= ' + nf(local), x1, yy + 16, { size: 12, font: 'mono', weight: 700, align: 'right', color: C.ink });
          }
          const vy = ry0 + p * rh + 12;
          const msg = ok
            ? '✓ ' + nf(local) + ' = ' + (heads / p) + ' heads × 128 × 3 (q, k, v) + ' + (heads / p) + ' (β) + 128 (f_a) + 128 (g_a). It is in KDA_BF16_LARGE_M_SHAPES_BY_TP, and the code accepts the BF16 copy.' + (p === 3 ? ' But start-tp3.sh sets GLM53_KDA_BF16_LARGE_M to zero by default (explicit opt-in), so the recipe\'s TP3 does not use it.' : ' On the recipe\'s TP2 it comes turned on (.env.example).')
            : '✗ ' + nf(local) + ' (our calculation): there is no qualified shape for TP4 in the list {12576, 8726}, and the code refuses with RuntimeError “requires TP=2 or TP=3”.';
          const ey = wrap(g, msg, x0, vy + 10, RW, { size: 12, color: ok ? C.e : C.red || C.c, weight: 600, lh: 16 });
          wrap(g, 'f_a and g_a (the gates\' low-rank projections) stay whole on every rank: the local shape is not “global ÷ p”.', x0, ey + 8, RW, { size: 11.5, color: C.muted, lh: 15 });
          return;
        }
        const n = pr.n, al = align(pr.fmt, ST.mode);
        const loc = n / p;
        const isInt = Number.isInteger(loc);
        const ok = isInt && loc % al === 0;
        rLocal.set(isInt ? nf(loc) : nf(loc, 2));
        rVer.set(ok ? '✓ clean cut' : isInt ? '✗ splits blocks' : '✗ not an integer');
        const axis = ST.mode === 'col' ? 'N (outputs) · column-parallel' : 'K (inputs) · row-parallel';
        V.text(g, (narrow ? 'cut: ' : 'cut dimension: ') + nf(n) + ' in ' + (narrow ? (ST.mode === 'col' ? 'N (outputs)' : 'K (inputs)') : axis), x0, 22, { size: narrow ? 11 : 12, color: C.muted, font: 'mono' });
        // ruler
        const ry = 36, rh = 28;
        for (let r = 0; r < p; r++) { g.fillStyle = V.alpha(RC(C, r), 0.35); g.fillRect(x0 + (RW * r) / p, ry, RW / p, rh); }
        V.box(g, x0, ry, RW, rh, { r: 4, stroke: V.alpha(C.ink, 0.4), lw: 1 });
        let tickEvery = 1;
        if (al > 1) {
          const spx = (RW * al) / n;
          tickEvery = spx >= 4 ? 1 : Math.max(1, Math.ceil(4 / Math.max(spx, 1e-3)));
          g.strokeStyle = V.alpha(C.ink, 0.35); g.lineWidth = 1; g.beginPath();
          for (let k = al * tickEvery; k < n; k += al * tickEvery) { const xx = x0 + (RW * k) / n; g.moveTo(xx, ry + 3); g.lineTo(xx, ry + rh - 3); }
          g.stroke();
        }
        for (let r = 1; r < p; r++) { const xx = x0 + (RW * r) / p; V.line(g, xx, ry - 6, xx, ry + rh + 6, { color: C.c, lw: 2.2 }); }
        V.text(g, al > 1 ? (tickEvery > 1 ? '1 tick = ' + tickEvery + ' blocks of ' + al : 'ticks = blocks of ' + al) : 'no blocks along this axis', x0, ry + rh + 17, { size: 11, font: 'mono', color: C.muted });
        if (narrow) V.text(g, BLK[pr.fmt], x0, ry + rh + 32, { size: 11, font: 'mono', color: C.muted, maxW: RW });
        else V.text(g, BLK[pr.fmt], x1, ry + rh + 17, { size: 11, font: 'mono', color: C.muted, align: 'right' });
        // per-rank labels
        for (let r = 0; r < p; r++) {
          const xa = x0 + (RW * r) / p, xb = x0 + (RW * (r + 1)) / p;
          chip(g, 'rank ' + r + ': ' + (isInt ? nf(loc) : nf(loc, 1)), (xa + xb) / 2, ry + rh / 2, RC(C, r), { align: 'center', size: 11, h: 18, fa: 0.85, maxW: xb - xa - 6, color: C.dark ? '#08140e' : '#fff' });
        }
        // magnifier
        const cut = loc;
        const B = al > 1 ? al : Math.max(16, Math.round(n / 24));
        const k0 = Math.floor(cut / B) - 2, k1 = k0 + 5;
        const zy = ry + rh + (narrow ? 84 : 56), zh = narrow ? 72 : 90;
        const zx0 = x0 + (narrow ? 0 : 60), zx1 = x1 - (narrow ? 0 : 60), ZW = zx1 - zx0;
        const cx = x0 + (RW * 1) / p;
        g.strokeStyle = V.alpha(C.c, 0.5); g.lineWidth = 1; g.setLineDash([3, 3]);
        // in the narrow layout the magnifier guides start below both captions, so they don't strike through them
        const gy0 = ry + rh + (narrow ? 40 : 4);
        g.beginPath(); g.moveTo(cx - 6, gy0); g.lineTo(zx0, zy - 4); g.moveTo(cx + 6, gy0); g.lineTo(zx1, zy - 4); g.stroke(); g.setLineDash([]);
        V.dot(g, cx, ry + rh + 2, 4, C.c);
        const e0 = k0 * B, e1 = k1 * B, span = e1 - e0;
        const px = (e) => zx0 + ((e - e0) / span) * ZW;
        let straddle = false;
        for (let k = k0; k < k1; k++) {
          const a = k * B, b = (k + 1) * B;
          if (b <= 0 || a >= n) continue;
          const xa = px(a) + 2, xb = px(b) - 2;
          const rA = Math.floor((a + 0.5) / loc), rB = Math.floor((b - 0.5) / loc);
          const split = al > 1 && rA !== rB;
          if (split) straddle = true;
          if (split) {
            const xc = px(cut);
            g.fillStyle = V.alpha(RC(C, rA), 0.5); g.fillRect(xa, zy, xc - xa, zh);
            g.fillStyle = V.alpha(RC(C, Math.min(p - 1, rB)), 0.5); g.fillRect(xc, zy, xb - xc, zh);
            const pulse = 0.6 + 0.4 * Math.sin(t * 4);
            V.box(g, xa, zy, xb - xa, zh, { r: 5, stroke: V.alpha(C.red || C.c, pulse), lw: 2.6 });
          } else V.box(g, xa, zy, xb - xa, zh, { r: 5, fill: V.alpha(RC(C, Math.min(p - 1, rA)), 0.45), stroke: RC(C, Math.min(p - 1, rA)), lw: 1.2 });
          if (al > 1) {
            chip(g, pr.fmt === 'exl3' ? (xb - xa > 60 ? 'tile' : 't') : 's', (xa + xb) / 2, zy + 14, split ? C.red || C.c : C.ink, { size: 11, h: 18, fa: 0.1, align: 'center' });
            V.text(g, '[' + nf(a) + ', ' + nf(b) + ')', (xa + xb) / 2, zy + zh - 10, { size: 11, font: 'mono', align: 'center', color: C.ink, maxW: xb - xa - 4 });
          } else V.text(g, 'no scale on this axis', (xa + xb) / 2, zy + zh / 2, { size: 11, font: 'mono', align: 'center', color: C.muted, maxW: xb - xa - 4 });
        }
        V.line(g, px(cut), zy - 8, px(cut), zy + zh + 8, { color: C.c, lw: 2.4, dash: [5, 3], dashOffset: -t * 12 });
        V.text(g, 'cut at ' + (isInt ? nf(cut) : nf(cut, 2)), px(cut), zy - 12, { size: 11, font: 'mono', align: 'center', color: C.c, weight: 700 });
        let vy = zy + zh + 26;
        if (straddle) { wrap(g, 'The red block has a single scale, but half of its values would go to each rank.', x0, vy, RW, { size: 11.5, color: C.red || C.c, lh: 15 }); vy += narrow ? 32 : 18; }
        // EXL3 vectors
        if (pr.fmt === 'exl3') {
          const sy = vy + 8, sh = 14;
          const lw = narrow ? 118 : 190;
          const rows2 = [
            { n: 'suh (size K, input)', split: ST.mode === 'row' },
            { n: 'svh (size N, output)', split: ST.mode === 'col' },
          ];
          rows2.forEach((q, k) => {
            const yy = sy + k * (sh + 10);
            V.text(g, q.n, x0, yy + sh - 2, { size: 11, font: 'mono', color: C.ink, maxW: lw - 6 });
            const bx = x0 + lw, bw2 = RW - lw - (narrow ? 64 : 110);
            if (q.split) {
              for (let r = 0; r < p; r++) { g.fillStyle = V.alpha(RC(C, r), 0.55); g.fillRect(bx + (bw2 * r) / p, yy, bw2 / p - 1, sh); }
            } else { V.box(g, bx, yy, bw2, sh, { r: 3, fill: V.alpha(C.muted, 0.12) }); hatch(g, bx, yy, bw2, sh, V.alpha(C.muted, 0.6), 5, t * 6); }
            V.text(g, q.split ? 'sliced' : 'replicated', bx + bw2 + 8, yy + sh - 2, { size: 11, font: 'mono', weight: 700, color: q.split ? C.c : C.muted });
          });
          vy = sy + 2 * (sh + 10) + 8;
          wrap(g, 'effective bits: whole [4096×2048] expert 4.0117 bpw; TP2 gate/up slice [4096→1024] 4.0196 bpw, because it carries the whole suh.', x0, vy + 6, RW, { size: 11, color: C.muted, lh: 14 });
          vy += narrow ? 44 : 30;
        }
        const msg = (ok ? '✓ ' : '✗ ') + (isInt ? nf(n) + ' / ' + p + ' = ' + nf(loc) + (al > 1 ? (ok ? ' = ' + nf(loc / al) + ' × ' + al : ' → ' + nf(loc / al, 2) + ' blocks of ' + al) : '') : nf(n) + ' / ' + p + ' = ' + nf(loc, 2) + ' is not an integer') + '. ' + note(ST.pre, p, ST.mode, ok);
        wrap(g, msg, x0, vy + 10, RW, { size: 12, weight: 600, color: ok ? C.e : C.red || C.c, lh: 16 });
      });
    })();

    /* ================== F5 — tokens, experts, and two GPUs ================== */
    (function () {
      const fig = root.querySelector('#fig-ep');
      if (!fig) return;
      adaptH(fig, (w) => (w < 620 ? 610 : 470));
      const E = 16;
      const ST = { mode: 'ep', B: 3, k: 2, seed: 5 };
      let route = [];
      const reroute = () => { const r = V.rng(ST.seed); route = Array.from({ length: 32 }, () => { const a = Array.from({ length: E }, (_, i) => i); for (let i = E - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } return a; }); };
      reroute();
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Mode', options: [{ v: 'tp', label: 'TP in expert' }, { v: 'ep', label: 'EP, DP=1' }, { v: 'epdp', label: 'EP + DP' }], value: 'ep', onChange: (v) => { ST.mode = v; } });
      UI.slider(ctl, { label: 'tokens in batch', min: 1, max: 32, step: 1, value: 3, onInput: (v) => { ST.B = v; } });
      UI.slider(ctl, { label: 'top-k', min: 1, max: 4, step: 1, value: 2, onInput: (v) => { ST.k = v; } });
      UI.button(ctl, '🎲 re-route', () => { ST.seed += 7; reroute(); });
      const owner = (e) => (e < 8 ? 0 : 1);
      V.fig(fig, (g, w, h, t) => {
        if (w < 140 || h < 100) return;
        const C = V.C(), m = ST.mode, B = ST.B, k = ST.k;
        const narrow = w < 620, pad = 12;
        const pairs = [];
        for (let i = 0; i < B; i++) for (let j = 0; j < k; j++) pairs.push([i, route[i][j]]);
        const home = (i) => (m === 'epdp' ? i % 2 : -1);
        const touched = new Set(pairs.map((q) => q[1]));
        const cnt = new Array(E).fill(0); pairs.forEach(([, e]) => cnt[e]++);
        let readA, readB;
        if (m === 'tp') { readA = readB = touched.size / 2; } else { readA = [...touched].filter((e) => e < 8).length; readB = touched.size - readA; }
        const cross = m === 'epdp' ? pairs.filter(([i, e]) => owner(e) !== home(i)).length : 0;
        // geometry
        const metW = narrow ? 0 : Math.min(250, w * 0.3);
        const metH = narrow ? 190 : 0;
        const gx = pad, gw = w - 2 * pad - (narrow ? 0 : metW + 14);
        const gh = (h - 2 * pad - metH - 10) / 2;
        const G = [{ x: gx, y: pad, w: gw, h: gh }, { x: gx, y: pad + gh + 10, w: gw, h: gh }];
        const tokW = narrow ? 64 : 88;
        const epos = [[], []], tpos = [[], []];
        G.forEach((P, r) => {
          const col = RC(C, r);
          V.box(g, P.x, P.y, P.w, P.h, { r: 12, fill: V.alpha(col, 0.05), stroke: V.alpha(col, 0.6), lw: 1.2 });
          chip(g, 'GPU ' + (r ? 'B' : 'A') + ' · rank ' + r, P.x + 10, P.y + 15, col, { size: 11, h: 19 });
          // this GPU's tokens
          const mine = [];
          for (let i = 0; i < B; i++) if (m !== 'epdp' || home(i) === r) mine.push(i);
          const tx0 = P.x + 10, ty0 = P.y + 32, tAreaH = P.h - 42;
          const per = Math.max(1, Math.floor(tAreaH / 14));
          const ncol = Math.ceil(mine.length / per) || 1;
          const tr = Math.max(3.5, Math.min(10, Math.min(tAreaH / (Math.min(mine.length, per) * 2.4), (tokW - 16) / (ncol * 2.4))));
          mine.forEach((i, q) => {
            const cI = Math.floor(q / per), rI = q % per;
            const rowsHere = Math.min(per, mine.length - cI * per);
            const x = tx0 + tr + cI * tr * 2.5;
            const y = ty0 + tAreaH / 2 + (rI - (rowsHere - 1) / 2) * tr * 2.5;
            tpos[r][i] = [x, y];
          });
          V.text(g, m === 'epdp' ? (narrow ? 'own' : 'own tokens') : (narrow ? 'replica' : 'tokens (replica)'), tx0, P.y + P.h - 6, { size: 11, font: 'mono', color: C.muted, maxW: tokW - 8 });
          // experts (geometry only; drawing comes after the lines)
          const ex0 = P.x + tokW, exW = P.w - tokW - 10;
          const cs = Math.max(12, Math.min(46, (exW - 7 * 5) / 8, (P.h - 50) / 2 - 6));
          const gridW = 8 * cs + 7 * 5;
          const gx0 = ex0 + (exW - gridW) / 2, gy0 = P.y + 30 + (P.h - 40 - (2 * cs + 8)) / 2;
          for (let e = 0; e < E; e++) {
            const cI = e % 8, rI = Math.floor(e / 8);
            const x = gx0 + cI * (cs + 5), y = gy0 + rI * (cs + 8);
            epos[r][e] = [x + cs / 2, y + cs / 2, x, y, cs];
          }
        });
        // token → expert lines
        const lines = [];
        pairs.forEach(([i, e], q) => {
          if (m === 'tp') { [0, 1].forEach((r) => lines.push([tpos[r][i], epos[r][e], RC(C, r), false, q])); }
          else if (m === 'ep') { const r = owner(e); lines.push([tpos[r][i], epos[r][e], RC(C, r), false, q]); }
          else { const hr = home(i), r = owner(e); lines.push([tpos[hr][i], epos[r][e], hr === r ? RC(C, r) : C.c, hr !== r, q]); }
        });
        const la = B * k > 40 ? 0.22 : 0.45;
        lines.forEach(([a, b, col, cr]) => { if (!a || !b) return; V.line(g, a[0], a[1], b[0], b[1], { color: V.alpha(col, cr ? 0.75 : la), lw: cr ? 1.6 : 1 }); });
        [0, 1].forEach((r) => {
          const col = RC(C, r);
          for (let e = 0; e < E; e++) {
            const [, , x, y, cs] = epos[r][e];
            const act = touched.has(e);
            const own = m === 'tp' ? 'half' : owner(e) === r ? 'full' : 'ghost';
            if (act && own !== 'ghost') V.glow(g, x + cs / 2, y + cs / 2, cs * 0.95, C.d, 0.55 + 0.25 * Math.sin(t * 4 + e));
            V.box(g, x, y, cs, cs, { r: 5, fill: C.bg || C.panel });
            if (own === 'full') V.box(g, x, y, cs, cs, { r: 5, fill: V.alpha(col, act ? 0.8 : 0.35), stroke: col, lw: act ? 2 : 1 });
            else if (own === 'half') {
              V.box(g, x, y, cs, cs, { r: 5, fill: V.alpha(C.bg || C.panel, 0.9), stroke: V.alpha(col, 0.7), lw: act ? 2 : 1 });
              g.save(); V.rr(g, x, y, cs, cs, 5); g.clip(); g.fillStyle = V.alpha(col, act ? 0.8 : 0.35); g.fillRect(x + (r ? cs / 2 : 0), y, cs / 2, cs); g.restore();
            } else { V.box(g, x, y, cs, cs, { r: 5, fill: V.alpha(C.bg || C.panel, 0.85) }); g.setLineDash([3, 3]); V.box(g, x, y, cs, cs, { r: 5, stroke: V.alpha(C.muted, 0.45), lw: 1 }); g.setLineDash([]); }
            V.text(g, String(e), x + cs / 2, y + cs / 2 + 0.5, { size: Math.min(12, Math.max(11, cs * 0.42)), font: 'mono', weight: 700, align: 'center', base: 'middle', color: own === 'ghost' ? V.alpha(C.muted, 0.7) : C.ink });
            if (act && own !== 'ghost' && cnt[e] > 1 && cs > 30) V.text(g, '×' + cnt[e], x + cs - 3, y + 11, { size: 11, font: 'mono', weight: 700, align: 'right', color: C.ink });
          }
        });
        lines.slice(0, 70).forEach(([a, b, col, cr, q]) => { if (!a || !b) return; const f = (t * 0.55 + q * 0.137) % 1; const fb = f < 0.5 ? f * 2 : 2 - f * 2; V.dot(g, V.lerp(a[0], b[0], fb), V.lerp(a[1], b[1], fb), cr ? 3.2 : 2.4, cr ? C.c : C.d); });
        // tokens on top
        [0, 1].forEach((r) => {
          tpos[r].forEach((pp, i) => {
            if (!pp) return;
            const rr = Math.max(3.5, B > 20 ? 4.5 : B > 10 ? 6.5 : 9);
            V.dot(g, pp[0], pp[1], rr, V.alpha(C.d, 0.95));
            if (rr >= 6) { const e = rr * 0.34; V.dot(g, pp[0] - e, pp[1] - e * 0.5, rr * 0.16, C.dark ? '#1b1203' : '#fff'); V.dot(g, pp[0] + e, pp[1] - e * 0.5, rr * 0.16, C.dark ? '#1b1203' : '#fff'); if (rr >= 8) { g.strokeStyle = C.dark ? '#1b1203' : '#fff'; g.lineWidth = 1.2; g.beginPath(); g.arc(pp[0], pp[1] + e * 0.2, rr * 0.38, 0.2 * Math.PI, 0.8 * Math.PI); g.stroke(); } }
          });
        });
        // metrics panel
        const Mx = narrow ? pad : w - pad - metW, My = narrow ? h - pad - metH : pad, Mw = narrow ? w - 2 * pad : metW, Mh = narrow ? metH : h - 2 * pad;
        V.box(g, Mx, My, Mw, Mh, { r: 12, fill: V.alpha(C.muted, 0.07), stroke: V.alpha(C.muted, 0.28), lw: 1 });
        let yy = My + 20;
        V.text(g, m === 'tp' ? 'experts read (in whole experts)' : 'whole experts read this step', Mx + 12, yy, { size: 11.5, weight: 700, color: C.ink, maxW: Mw - 24 });
        yy += 8;
        const maxR = Math.max(readA, readB), mean = (readA + readB) / 2;
        const scale = Math.max(4, touched.size);
        [[readA, 0], [readB, 1]].forEach(([v, r]) => {
          yy += 20;
          V.text(g, 'GPU ' + (r ? 'B' : 'A'), Mx + 12, yy + 4, { size: 11, font: 'mono', weight: 700, color: RC(C, r) });
          const bx = Mx + 58, bw = Mw - 58 - 44;
          V.box(g, bx, yy - 7, bw, 14, { r: 4, fill: V.alpha(C.muted, 0.12) });
          V.box(g, bx, yy - 7, Math.max(2, (v / scale) * bw), 14, { r: 4, fill: V.alpha(RC(C, r), 0.8) });
          V.text(g, (m === 'tp' ? nf(v, 1) : String(v)), Mx + Mw - 12, yy + 4, { size: 11, font: 'mono', weight: 700, align: 'right', color: C.ink });
        });
        yy += 24;
        const imb = mean > 0 ? maxR / mean : 1;
        const lines2 = [
          'the step waits for the busiest: ' + (m === 'tp' ? nf(maxR, 1) + ' expert' + (maxR === 1 ? '' : 's') + ' (' + touched.size + ' halves)' : maxR + ' whole expert' + (maxR === 1 ? '' : 's')),
          'imbalance max/mean: ' + nf(imb, 2) + '×',
          m === 'epdp' ? 'all-to-all: ' + cross + ' of ' + pairs.length + ' pairs cross the network (round trip)' : 'communication: 1 all-reduce of ' + B + ' × d (same as TP)',
        ];
        lines2.forEach((s, q) => { yy = wrap(g, s, Mx + 12, yy + (q ? 4 : 0), Mw - 24, { size: 11, color: q === 1 && imb > 1.3 ? C.red || C.c : q === 2 && m === 'epdp' ? C.c : C.ink, lh: 14 }); });
        yy += 6;
        const tip = m === 'tp' ? 'TP in the expert: each GPU reads half of every expert touched. Perfect balance, but every expert gets split (blocks and kernels have to accept the half).' : m === 'ep' ? 'EP with DP=1: tokens replicated, no arrow crosses between GPUs. With few tokens, the chosen experts can all land on the same GPU.' : 'EP + DP: each GPU has its own tokens. The pink arrows are tokens that travel to the expert\'s owner and back.';
        if (!narrow || Mh > 150) wrap(g, tip, Mx + 12, yy + 6, Mw - 24, { size: 11, color: C.muted, lh: 14, maxLines: narrow ? 3 : 7 });
      });
    })();

    /* ================== F6 — pipeline assembly line × TP all-hands ========= */
    (function () {
      const fig = root.querySelector('#fig-pp');
      if (!fig) return;
      adaptH(fig, (w) => (w < 600 ? 410 : 372));
      const ST = { p: 2, m: 1, c: 0.1 };
      const ctl = UI.controls(fig);
      UI.slider(ctl, { label: 'nodes (p)', min: 2, max: 4, step: 1, value: 2, onInput: (v) => { ST.p = v; } });
      UI.slider(ctl, { label: 'microbatches (m)', min: 1, max: 8, step: 1, value: 1, onInput: (v) => { ST.m = v; } });
      UI.slider(ctl, { label: 'network in TP', min: 0, max: 40, step: 2, value: 10, fmt: (v) => v + '%', onInput: (v) => { ST.c = v / 100; } });
      V.fig(fig, (g, w, h, t) => {
        if (w < 140 || h < 100) return;
        const C = V.C(), p = ST.p, m = ST.m, c = ST.c;
        const narrow = w < 600, pad = 14;
        const Tpp = m + p - 1, Ttp = m * (1 + c);
        const Tmax = Math.max(Tpp, Ttp);
        const lx = pad + (narrow ? 46 : 70), rx = w - pad;
        const X = (u) => lx + (u / Tmax) * (rx - lx);
        const cyc = Tmax + 2.2;
        const tc = (t * 0.9) % cyc;
        const cols = C.series;
        const rowH = Math.max(18, Math.min(38, (h - (narrow ? 220 : 170)) / (2 * p)));
        const chartH = p * rowH + 22;
        function axis(y0) {
          for (let u = 0; u <= Math.floor(Tmax); u++) { const xx = X(u); V.line(g, xx, y0, xx, y0 + p * rowH, { color: V.alpha(C.muted, 0.18), lw: 1 }); if (!narrow || u % 2 === 0) V.text(g, String(u), xx, y0 + p * rowH + 13, { size: 11, font: 'mono', align: 'center', color: C.muted }); }
        }
        function block(x0, x1, y, hh, col, label, hatchCol) {
          const xe = Math.min(x1, X(tc));
          if (xe > x0) { V.box(g, x0 + 1, y + 2, xe - x0 - 2, hh - 4, { r: 4, fill: V.alpha(col, 0.72) }); if (hatchCol) hatch(g, x0 + 1, y + 2, xe - x0 - 2, hh - 4, hatchCol, 5, 0, 1.3); }
          V.box(g, x0 + 1, y + 2, x1 - x0 - 2, hh - 4, { r: 4, stroke: V.alpha(col, 0.8), lw: 1 });
          if (label && x1 - x0 > 22 && xe >= x1 - 1) V.text(g, label, (x0 + x1) / 2, y + hh / 2 + 0.5, { size: 11, font: 'mono', weight: 700, align: 'center', base: 'middle', color: C.dark ? '#08140e' : '#fff' });
        }
        // PP
        const hdr = narrow ? 18 : 0;
        let y0 = 30 + hdr;
        V.text(g, 'Pipeline (PP=' + p + ')', pad, 20, { size: 12.5, weight: 700, color: C.ink });
        const bub = (p - 1) / (m + p - 1);
        V.text(g, 'bubble ' + nf(bub * 100, 0) + '% · 1st result at t = ' + p + ' · total ' + Tpp, narrow ? pad : rx, narrow ? 38 : 20, { size: 11, font: 'mono', align: narrow ? 'left' : 'right', color: C.c });
        for (let s = 0; s < p; s++) {
          const y = y0 + s * rowH;
          V.text(g, narrow ? 'st.' + (s + 1) : 'stage ' + (s + 1), pad, y + rowH / 2 + 4, { size: 11, font: 'mono', color: RC(C, s) });
          // bubbles
          hatch(g, X(0), y + 3, X(s) - X(0), rowH - 6, V.alpha(C.muted, 0.45), 6, 0);
          hatch(g, X(s + m), y + 3, X(Tpp) - X(s + m), rowH - 6, V.alpha(C.muted, 0.45), 6, 0);
          for (let j = 0; j < m; j++) block(X(j + s), X(j + s + 1), y, rowH, cols[j % cols.length], 'μ' + j);
        }
        axis(y0);
        // TP
        y0 = 30 + hdr + chartH + 40 + hdr;
        V.text(g, 'Tensor parallel (TP=' + p + ')', pad, y0 - 10 - hdr, { size: 12.5, weight: 700, color: C.ink });
        V.text(g, '1st result at t = ' + nf(1 + c, 2) + ' · total ' + nf(Ttp, 1), narrow ? pad : rx, y0 - 10 + (narrow ? 8 - hdr + 10 : 0), { size: 11, font: 'mono', align: narrow ? 'left' : 'right', color: C.c });
        for (let s = 0; s < p; s++) {
          const y = y0 + s * rowH;
          V.text(g, 'node ' + (s + 1), pad, y + rowH / 2 + 4, { size: 11, font: 'mono', color: RC(C, s) });
          for (let j = 0; j < m; j++) {
            const a = j * (1 + c);
            block(X(a), X(a + 1), y, rowH, cols[j % cols.length], 'μ' + j);
            if (c > 0) block(X(a + 1), X(a + 1 + c), y, rowH, C.c, '', V.alpha(C.dark ? '#000' : '#fff', 0.35));
          }
        }
        axis(y0);
        // cursor
        const xc = X(Math.min(tc, Tmax));
        V.line(g, xc, 22, xc, y0 + p * rowH + 4, { color: V.alpha(C.d, 0.85), lw: 1.5, dash: [4, 3] });
        const msg = m === 1 ? 'With 1 microbatch (single-user decode), PP delivers the result at t = ' + p + ' and TP at t = ' + nf(1 + c, 2) + ': TP is ' + nf(p / (1 + c), 1) + '× faster.' : 'With ' + m + ' microbatches: PP throughput ' + nf(m / Tpp, 2) + ' per time unit, TP ' + nf(m / Ttp, 2) + '. Raise m to watch the bubble shrink.';
        wrap(g, msg, pad, y0 + p * rowH + 38, w - 2 * pad, { size: narrow ? 11.5 : 12.5, color: C.ink, weight: 600, lh: 16 });
        V.text(g, 'unit = T/p (time to read 1/p of the weights)', rx, h - 6, { size: 11, font: 'mono', align: 'right', color: C.muted });
      });
    })();

    /* ================== F7 — where the KV lives ============================ */
    (function () {
      const fig = root.querySelector('#fig-kv');
      if (!fig) return;
      adaptH(fig, (w) => (w < 600 ? 690 : 490));
      const ST = { mode: 'mla', ctx: 5, seed: 3 };
      const CTX = [8192, 32768, 131072, 262144, 524288, 1048576];
      const CTXL = ['8K', '32K', '128K', '256K', '512K', '1M'];
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Layout', options: [{ v: 'gqa', label: 'GQA (Qwen)' }, { v: 'mla', label: 'MLA (GLM, DS)' }, { v: 'dcp', label: 'DCP=2' }], value: 'mla', onChange: (v) => { ST.mode = v; } });
      UI.slider(ctl, { label: 'context', min: 0, max: 5, step: 1, value: 5, fmt: (v) => CTXL[v], onInput: (v) => { ST.ctx = v; } });
      UI.button(ctl, '🎲 new scores', () => { ST.seed += 5; lse = mkLse(); });
      function mkLse() {
        const r = V.rng(ST.seed);
        const s = Array.from({ length: 6 }, () => Math.round((r() * 4 - 1) * 10) / 10);
        const v = Array.from({ length: 6 }, () => Math.round((r() * 4 - 2) * 10) / 10);
        const part = (ids) => { const mm = Math.max(...ids.map((j) => s[j])); const l = ids.reduce((a, j) => a + Math.exp(s[j] - mm), 0); const o = ids.reduce((a, j) => a + Math.exp(s[j] - mm) * v[j], 0) / l; return { m: mm, l, o }; };
        const A = part([0, 2, 4]), B = part([1, 3, 5]), F = part([0, 1, 2, 3, 4, 5]);
        const M = Math.max(A.m, B.m);
        const wA = Math.exp(A.m - M) * A.l, wB = Math.exp(B.m - M) * B.l;
        return { s, v, A, B, F, o: (wA * A.o + wB * B.o) / (wA + wB) };
      }
      let lse = mkLse();
      const INFO = {
        gqa: { bpt: 6656, txt: 'Qwen (QSA): 2 KV heads → 1 per GPU. Main FP8 KV: 13,312 B/token in total, 6,656 B/token per GPU.' },
        mla: { bpt: 3850, txt: 'MLA/MQA: the whole latent row goes to both ranks. GLM: 656 B/token/layer on each rank. DeepSeek: ~3.85 KB/token physical, the same on both (17.04 GiB = 2,331,430 tokens).' },
        dcp: { bpt: 1925, txt: 'DeepSeek with DCP=2 (not turned on in this image): each rank stores half of the tokens → ~2× capacity, at the price of ~43 more collectives per step.' },
      };
      V.fig(fig, (g, w, h, t) => {
        if (w < 140 || h < 100) return;
        const C = V.C(), m = ST.mode;
        const narrow = w < 600, pad = 14;
        const cap = m === 'mla' ? 12 : 24;
        const cyc = cap + 8;
        const nTok = Math.min(cap, Math.floor((t * 2.4) % cyc));
        const topH = narrow ? 330 : 248;
        const pw = (w - 3 * pad) / 2;
        const U = 24, GC = 4, GR = 6;
        [0, 1].forEach((r) => {
          const px = pad + r * (pw + pad), py = 10, ph = topH - (narrow ? 80 : 60);
          const col = RC(C, r);
          V.box(g, px, py, pw, ph, { r: 12, fill: V.alpha(col, 0.05), stroke: V.alpha(col, 0.6), lw: 1.2 });
          chip(g, (narrow ? 'KV · GPU ' : 'KV pool · GPU ') + (r ? 'B' : 'A'), px + 10, py + 15, col, { size: 11, h: 19 });
          const gx0 = px + 12, gy0 = py + 32, gw = pw - 24, gh = ph - 42;
          const cw = (gw - (GC - 1) * 4) / GC, chh = (gh - (GR - 1) * 4) / GR;
          // occupancy: list of entries (tok, units)
          const ents = [];
          for (let q = 0; q < nTok; q++) {
            if (m === 'gqa') ents.push({ q, u: 1, lbl: narrow ? 't' + q : 't' + q + ' h' + r });
            else if (m === 'mla') ents.push({ q, u: 2, lbl: 't' + q + ' latent' });
            else if (q % 2 === r) ents.push({ q, u: 2, lbl: 't' + q });
          }
          let slot = 0;
          for (let s2 = 0; s2 < U; s2++) {
            const cI = s2 % GC, rI = Math.floor(s2 / GC);
            V.box(g, gx0 + cI * (cw + 4), gy0 + rI * (chh + 4), cw, chh, { r: 4, stroke: V.alpha(C.muted, 0.3), lw: 1 });
          }
          ents.forEach((e, idx) => {
            if (slot + e.u > U) return;
            const cI = slot % GC, rI = Math.floor(slot / GC);
            const ww = e.u * cw + (e.u - 1) * 4;
            const fresh = idx === ents.length - 1 && nTok < cap;
            if (fresh) V.glow(g, gx0 + cI * (cw + 4) + ww / 2, gy0 + rI * (chh + 4) + chh / 2, 26, C.d, 0.8);
            V.box(g, gx0 + cI * (cw + 4), gy0 + rI * (chh + 4), ww, chh, { r: 4, fill: V.alpha(col, m === 'mla' ? 0.5 : 0.55), stroke: col, lw: 1 });
            if (m === 'mla') hatch(g, gx0 + cI * (cw + 4) + 1, gy0 + rI * (chh + 4) + 1, ww - 2, chh - 2, V.alpha(C.ink, 0.12), 6, 0);
            V.text(g, e.lbl, gx0 + cI * (cw + 4) + ww / 2, gy0 + rI * (chh + 4) + chh / 2 + 0.5, { size: 11, font: 'mono', weight: 600, align: 'center', base: 'middle', color: C.ink, maxW: ww - 4 });
            slot += e.u;
          });
        });
        // counter
        const tokFit = cap;
        // counter: wraps in the narrow layout so it isn't clipped at the edge
        const y1 = 10 + topH - (narrow ? 80 : 60) + 20;
        const cntTxt = 'tokens stored: ' + nTok + ' · pair capacity: ' + tokFit + ' tokens' + (m === 'mla' ? (narrow ? ' (= 1 GPU alone)' : ' (same as one GPU alone)') : '');
        const y2 = wrap(g, cntTxt, pad, y1, w - 2 * pad, { size: narrow ? 11 : 11.5, font: 'mono', weight: 700, color: m === 'mla' ? C.red || C.c : C.e, lh: 15 });
        const bpt = INFO[m].bpt, ctx = CTX[ST.ctx];
        const per = bpt * ctx;
        wrap(g, '1 sequence of ' + CTXL[ST.ctx] + ': ' + gib(per) + ' per rank' + (m === 'gqa' ? ' (Qwen, main KV only)' : ' (DeepSeek, physical)'), pad, y2 + 1, w - 2 * pad, { size: 11, font: 'mono', color: C.muted, lh: 15 });
        // LSE panel
        const ly = topH + 6, lh = h - ly - 8;
        const on = m === 'dcp';
        V.box(g, pad, ly, w - 2 * pad, lh, { r: 12, fill: V.alpha(C.c, on ? 0.06 : 0.03), stroke: V.alpha(C.c, on ? 0.6 : 0.25), lw: 1.2 });
        g.globalAlpha = on ? 1 : 0.55;
        V.text(g, on ? 'DCP\'s exact merge (log-sum-exp)' : 'what DCP would do: exact merge via log-sum-exp', pad + 12, ly + 20, { size: 12, weight: 700, color: C.ink });
        const L = lse;
        const bx = pad + 14, bw = narrow ? w - 2 * pad - 28 : (w - 2 * pad) * 0.42;
        const colW = bw / 6;
        const base = ly + (narrow ? 112 : 118);
        L.s.forEach((s, j) => {
          const x = bx + j * colW + colW * 0.18, ww = colW * 0.64;
          const hh = Math.max(3, ((s + 1.2) / 4.4) * (narrow ? 58 : 66));
          const col = RC(C, j % 2);
          V.box(g, x, base - hh, ww, hh, { r: 3, fill: V.alpha(col, 0.7) });
          V.text(g, nf(s, 1), x + ww / 2, base - hh - 5, { size: 11, font: 'mono', align: 'center', color: C.ink });
          V.text(g, 't' + j, x + ww / 2, base + 13, { size: 11, font: 'mono', align: 'center', color: RC(C, j % 2), weight: 700 });
          V.text(g, 'v=' + nf(L.v[j], 1), x + ww / 2, base + 27, { size: 11, font: 'mono', align: 'center', color: C.muted });
        });
        V.text(g, 'scores s (height) and values v', bx, ly + 38, { size: 11, font: 'mono', color: C.muted });
        const tx = narrow ? bx : bx + bw + 24, ty0 = narrow ? base + 50 : ly + 44;
        const fs = narrow ? 11 : 12;
        const rowsL = [
          ['A (t0,t2,t4)', 'm=' + nf(L.A.m, 2) + '  ℓ=' + nf(L.A.l, 3) + '  o=' + nf(L.A.o, 4), C.a],
          ['B (t1,t3,t5)', 'm=' + nf(L.B.m, 2) + '  ℓ=' + nf(L.B.l, 3) + '  o=' + nf(L.B.o, 4), C.b],
          ['merge', 'o = ' + nf(L.o, 4), C.c],
          ['full attention', 'o = ' + nf(L.F.o, 4) + (Math.abs(L.o - L.F.o) < 1e-9 ? '  ✓ identical' : ''), C.e],
        ];
        rowsL.forEach(([k, v, col], q) => {
          const yy = ty0 + q * (fs + 9);
          V.text(g, k, tx, yy, { size: fs, font: 'mono', weight: 700, color: col });
          V.text(g, v, tx + (narrow ? 112 : 128), yy, { size: fs, font: 'mono', color: C.ink, maxW: w - pad - 14 - tx - (narrow ? 112 : 128) });
        });
        g.globalAlpha = 1;
        wrap(g, INFO[m].txt, pad + 12, h - (narrow ? 66 : 30), w - 2 * pad - 24, { size: 11, color: C.muted, lh: 14 });
      });
    })();

    /* ================== F8 — distributed top-k ============================= */
    (function () {
      const fig = root.querySelector('#fig-topk');
      if (!fig) return;
      const N = 24;
      const ST = { p: 2, k: 3, seed: 9, argmax: false };
      // table geometry: (score, id) chips in rows, always legible (≥ 11 px)
      const geo = (w) => {
        const narrow = w < 600, pad = 14;
        const kk = ST.argmax ? 1 : ST.k, n = ST.p * kk;
        const cw = narrow ? 50 : 62;
        const perRow = Math.max(1, Math.floor((w - 2 * pad - 20 + 4) / (cw + 4)));
        const nRows = Math.max(1, Math.ceil(n / perRow));
        const baseY = narrow ? 200 : 170;
        return { narrow, pad, cw, perRow, nRows, baseY, mesaH: 30 + nRows * 24 + 6 };
      };
      const reH = adaptH(fig, (w) => { const G0 = geo(w); return G0.baseY + 70 + G0.mesaH + (G0.narrow ? 96 : 70); });
      let sc = [];
      const regen = () => { const r = V.rng(ST.seed); const used = new Set(); sc = Array.from({ length: N }, () => { let v; do { v = Math.round(r() * 99) + 1; } while (used.has(v)); used.add(v); return v; }); };
      regen();
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'ranks', options: [2, 3, 4], value: 2, onChange: (v) => { ST.p = v; if (reH) reH(); } });
      UI.slider(ctl, { label: 'k', min: 1, max: 6, step: 1, value: 3, onInput: (v) => { ST.k = v; if (reH) reH(); } });
      UI.toggle(ctl, { label: 'argmax mode (vocab-parallel)', value: false, onChange: (v) => { ST.argmax = v; if (reH) reH(); } });
      UI.button(ctl, '🎲 new scores', () => { ST.seed += 3; regen(); });
      V.fig(fig, (g, w, h, t) => {
        if (w < 140 || h < 100) return;
        const C = V.C(), p = ST.p, k = ST.argmax ? 1 : ST.k;
        const G0 = geo(w), narrow = G0.narrow, pad = G0.pad;
        const per = N / p;
        const P = 8, ph = t % P;
        const x0 = pad, x1 = w - pad, bw = (x1 - x0) / N;
        const baseY = G0.baseY, maxH = narrow ? 120 : 110;
        const allVals = bw > 22, allIds = bw > 16;
        const loc = [];
        for (let r = 0; r < p; r++) { const ids = Array.from({ length: per }, (_, q) => r * per + q); ids.sort((a, b) => sc[b] - sc[a] || a - b); loc.push(ids.slice(0, k)); }
        const glob = Array.from({ length: N }, (_, i) => i).sort((a, b) => sc[b] - sc[a] || a - b).slice(0, k);
        const cand = loc.flat().sort((a, b) => sc[b] - sc[a] || a - b);
        const merged = cand.slice(0, k);
        const same = merged.every((id, q) => id === glob[q]);
        V.text(g, ST.argmax ? (narrow ? N + ' logits in ' + p + ' id ranges' : 'logits for ' + N + ' ids, in ' + p + ' contiguous ranges (vocab-parallel lm_head)') : (narrow ? N + ' scores in ' + p + ' slices (SP indexer)' : N + ' key scores in ' + p + ' contiguous slices (SP indexer)'), pad, 18, { size: narrow ? 11 : 12, font: 'mono', color: C.muted, maxW: w - 2 * pad });
        for (let r = 0; r < p; r++) {
          const xa = x0 + r * per * bw, xb = xa + per * bw;
          V.box(g, xa + 1, baseY - maxH - 20, xb - xa - 2, maxH + 36, { r: 8, fill: V.alpha(RC(C, r), 0.05), stroke: V.alpha(RC(C, r), 0.45), lw: 1 });
          const rl = (narrow && p > 2 ? 'r' + r : 'rank ' + r) + ' · ids ' + (r * per) + '–' + ((r + 1) * per - 1);
          V.text(g, rl, (xa + xb) / 2, baseY + 32, { size: 11, font: 'mono', weight: 700, align: 'center', color: RC(C, r), maxW: xb - xa - 6 });
        }
        const showLocal = ph > 1.2, fly = V.clamp((ph - 3) / 1.6, 0, 1), mergedOn = ph > 4.8;
        const mesaY = baseY + 70, mesaH = G0.mesaH;
        V.box(g, x0, mesaY, x1 - x0, mesaH, { r: 10, fill: V.alpha(C.c, 0.05), stroke: V.alpha(C.c, 0.5), lw: 1.2 });
        V.text(g, (narrow ? 'table (all-gather of ' : 'common table (all-gather of ') + p + ' × ' + k + ' pairs ' + (ST.argmax ? '(value, id)' : '(score, id)') + ')', x0 + 10, mesaY + 16, { size: 11, font: 'mono', color: C.c, weight: 700, maxW: x1 - x0 - 20 });
        sc.forEach((v, i) => {
          const r = Math.floor(i / per);
          const x = x0 + i * bw + bw * 0.15, ww = bw * 0.7;
          const hh = (v / 100) * maxH;
          const isLoc = showLocal && loc[r].includes(i);
          const isGlob = glob.includes(i);
          const dim = mergedOn && !isGlob;
          V.box(g, x, baseY - hh, ww, hh, { r: 3, fill: V.alpha(RC(C, r), dim ? 0.2 : isLoc ? 0.9 : 0.45), stroke: isLoc ? RC(C, r) : null, lw: 1.5 });
          // in the narrow layout, only crowned bars show value and id (the rest are too thin to read)
          const showVal = allVals || isLoc, showId = allIds || isLoc;
          if (showVal) V.text(g, String(v), x + ww / 2, baseY - hh - 4, { size: 11, font: 'mono', align: 'center', color: isLoc ? C.ink : C.muted, weight: isLoc ? 700 : 400, maxW: bw - 1 });
          if (isLoc) V.text(g, '♛', x + ww / 2, baseY - hh - 18, { size: 12, align: 'center', color: C.d });
          if (isGlob && mergedOn) V.text(g, '★', x + ww / 2, baseY - maxH - 26, { size: 13, align: 'center', color: C.e });
          if (showId) V.text(g, String(i), x + ww / 2, baseY + 14, { size: 11, font: 'mono', align: 'center', color: isLoc ? RC(C, r) : C.muted, weight: isLoc ? 700 : 400, maxW: bw - 1 });
        });
        // candidates flying to the table, in rows of (score, id) chips
        if (showLocal && fly > 0) {
          const cw = G0.cw, perRow = G0.perRow;
          const flat = loc.flat();
          cand.forEach((id, q) => {
            const r = Math.floor(id / per);
            const sx = x0 + id * bw + bw / 2, sy = baseY - (sc[id] / 100) * maxH;
            const ord = mergedOn ? q : flat.indexOf(id);
            const tx = x0 + 10 + (ord % perRow) * (cw + 4) + cw / 2, ty = mesaY + 38 + Math.floor(ord / perRow) * 24;
            const f = V.ease.inOut(fly);
            const x = V.lerp(sx, tx, f), y = V.lerp(sy, ty, f);
            const win = mergedOn && merged.includes(id);
            V.box(g, x - cw / 2, y - 10, cw, 20, { r: 6, fill: V.alpha(RC(C, r), win ? 0.85 : mergedOn ? 0.18 : 0.5), stroke: win ? C.e : RC(C, r), lw: win ? 2 : 1 });
            V.text(g, '(' + sc[id] + ',' + id + ')', x, y + 0.5, { size: 11, font: 'mono', weight: 700, align: 'center', base: 'middle', color: C.ink, maxW: cw - 6 });
          });
        }
        const yT = mesaY + mesaH + 20;
        if (mergedOn) V.text(g, same ? '✓ table top-' + k + ' = brute-force top-' + k + ' (★)' : 'different', pad, yT, { size: 12, weight: 700, color: same ? C.e : C.red || C.c, maxW: w - 2 * pad });
        else V.text(g, ph < 1.2 ? '1. each rank has only its own slice' : ph < 3 ? '2. each rank picks its local top-' + k + ' (♛)' : '3. the pairs travel to the common table', pad, yT, { size: 12, weight: 700, color: C.ink, maxW: w - 2 * pad });
        const rw = ST.argmax ? 'Qwen and DeepSeek: 2 numbers per rank per token, instead of gathering 248,320 or 129,280 logits.' : 'Exchanged: ' + p * k + ' pairs, versus ' + N + ' values for the whole row. In DeepSeek\'s SP indexer: p × 512 pairs × 8 B per query (TP2: 8 KB).';
        wrap(g, rw, pad, yT + 18, w - 2 * pad, { size: 11, color: C.muted, lh: 14 });
      });
    })();

    /* ============ F8b — three ways to close the pair (GLM·TF recipe) ======== */
    (function () {
      const fig = root.querySelector('#fig-fechar');
      if (!fig) return;
      const RS = [1, 8, 16, 32, 64, 256, 1024, 2048];
      const KIB = 1024, MIB = 1048576, GIB = 1073741824, NPTS = 90, MIN_ROWS = 64;
      const ST = { m: 'split', ri: 7, t0: null };
      adaptH(fig, (w) => (w < 600 ? (w < 380 ? 690 : 650) : 520));
      // bytes sent per rank at one exchange point (2 ranks, D = 4096); all-reduce counted as a ring
      const calc = (m, R) => {
        const H = Math.ceil(R / 2);
        if (m === 'ar') return { send: R * 8 * KIB, glue: R, msgs: 1, rnd: 2 };
        if (m === 'ag') return { send: R * 16 * KIB, glue: R, msgs: 1, rnd: 1 };
        return { send: H * 24 * KIB, glue: H, msgs: 2, rnd: 1 };
      };
      const fmtB = (b) => {
        if (b < MIB) return nf(b / KIB, 0) + ' KiB';
        if (b < GIB) { const v = b / MIB; return nf(v, Number.isInteger(v) ? 0 : 1) + ' MiB'; }
        return nf(b / GIB, 2) + ' GiB';
      };
      const effOf = () => (ST.m === 'split' && RS[ST.ri] < MIN_ROWS ? 'ag' : ST.m);
      let rdA = null, rdB = null;
      const upd = () => { if (!rdA || !rdB) return; const k = calc(effOf(), RS[ST.ri]); rdA.set(fmtB(k.send)); rdB.set(fmtB(k.send * NPTS)); };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Closing', options: [{ v: 'ar', label: 'all-reduce BF16 (vLLM)' }, { v: 'ag', label: 'all-gather fp32 (TF)' }, { v: 'split', label: 'split by rows (TF)' }], value: 'split', onChange: (v) => { ST.m = v; ST.t0 = null; upd(); } });
      UI.slider(ctl, { label: 'rows R', min: 0, max: RS.length - 1, step: 1, value: 7, fmt: (v) => nf(RS[v]), onInput: (v) => { ST.ri = v; ST.t0 = null; upd(); } });
      rdA = UI.readout(ctl, 'sent per rank, per point', '—');
      rdB = UI.readout(ctl, 'per forward pass (90 points)', '—');
      upd();
      const C0 = V.C();
      UI.legend(ctl, [[C0.d, 'partial'], [C0.c, 'glue done here'], [C0.e, 'finished row']]);
      let lastC = null, tfc = null;
      const tfCol = (C) => { if (C !== lastC) { lastC = C; tfc = getComputedStyle(document.documentElement).getPropertyValue('--tf').trim() || C.d; } return tfc; };
      const TT = { ar: [0, 1.0, 2.6, 4.2, 6.6], ag: [0, 1.0, 3.0, 4.2, 6.6], split: [0, 1.0, 2.6, 4.0, 5.4, 6.6] };
      const STEPS = {
        ar: ['1. each rank has a BF16 partial of all R rows', '2. reduce-scatter: half a row crosses, and each rank sums its own half', '3. all-gather: the summed halves cross back', '4. both ranks run the glue on all R rows'],
        ag: ['1. each rank has an fp32 partial of all R rows', '2. all-gather: the whole partials cross the cable', '3. each rank adds the two, rank 0 first, and rounds once', '4. both ranks run the glue on all R rows'],
        split: ['1. each rank has an fp32 partial of all R rows', '2. each rank sends its peer only the partial of the peer\'s rows', '3. each rank sums its own rows (rank 0 first) and runs the glue only on them', '4. the normalized rows, in BF16, cross to the peer', '5. both ranks hold the whole input of the next block'],
      };
      const WIRE = { ar: ['', 'half row BF16', 'summed half row', ''], ag: ['', 'fp32 row · 16 KiB', '', ''], split: ['', 'peer partials · fp32', '', 'finished rows · BF16', ''] };
      const NAMES = { ar: ['all-reduce BF16', 'vLLM'], ag: ['all-gather fp32 + rank-0-first sum', 'TensorFold, decode'], split: ['glue split by rows', 'TensorFold, prefill (SPLIT=1)'] };
      V.fig(fig, (g, w, h, t) => {
        if (w < 140 || h < 100) return;
        const C = V.C(), TF = tfCol(C), narrow = w < 600, pad = 14;
        const R = RS[ST.ri], m = ST.m, fall = m === 'split' && R < MIN_ROWS, eff = fall ? 'ag' : m;
        if (ST.t0 == null) ST.t0 = t;
        const T = TT[eff], P = T[T.length - 1];
        const ph = (t - ST.t0) % P;
        let s = 0;
        while (s < T.length - 2 && ph >= T[s + 1]) s++;
        const u = V.clamp((ph - T[s]) / (T[s + 1] - T[s]), 0, 1);
        const n = Math.min(R, 16), per = R / n, half = n / 2;
        const own = (i) => (i < half ? 0 : 1);
        // header
        V.text(g, NAMES[eff][0] + ' · ' + NAMES[eff][1], pad, 16, { size: 12.5, weight: 700, color: eff === 'ar' ? C.glm : TF, maxW: w - 2 * pad });
        V.text(g, 'R = ' + nf(R) + (R === 1 ? ' row' : ' rows') + (per > 1 ? ' · ' + n + ' bands of ' + nf(per) : ''), pad, 34, { size: 11, font: 'mono', color: C.muted, maxW: w - 2 * pad });
        // column geometry
        const cw = narrow ? Math.floor((w - 2 * pad) * 0.34) : Math.min(250, Math.floor((w - 2 * pad) * 0.3));
        const xs = [pad, w - pad - cw];
        const hdrY = 60, rowsTop = 90, rowsH = narrow ? 168 : 156;
        const bh = Math.min(18, rowsH / n), y0 = rowsTop + (rowsH - bh * n) / 2;
        const bandY = (i) => y0 + i * bh + bh / 2;
        const mx0 = xs[0] + cw, mx1 = xs[1], midX = (mx0 + mx1) / 2;
        // the cable
        V.box(g, mx0 + 6, rowsTop - 8, mx1 - mx0 - 12, rowsH + 16, { r: 8, fill: V.alpha(C.muted, 0.05), stroke: V.alpha(C.muted, 0.25), lw: 1 });
        V.text(g, narrow ? 'cable' : 'cable between the Sparks', midX, hdrY + 2, { size: 11, font: 'mono', align: 'center', color: C.muted, maxW: mx1 - mx0 - 8 });
        const wl = (WIRE[eff] || [])[s];
        if (wl) V.text(g, wl, midX, hdrY + 18, { size: 11, font: 'mono', weight: 700, align: 'center', color: s === 3 && eff === 'split' ? C.e : C.d, maxW: mx1 - mx0 - 8 });
        const bandState = (r, i) => {
          if (eff === 'split') {
            const mine = own(i) === r;
            if (s === 0) return [C.d, 0.55];
            if (s === 1) return mine ? [C.d, 0.55] : [C.d, 0.08 + 0.47 * (1 - u)];
            if (s === 2) return mine ? [C.c, 0.3 + 0.5 * u] : [C.muted, 0.12];
            if (s === 3) return mine ? [C.e, 0.6] : (u > 0.85 ? [C.e, 0.6] : [C.muted, 0.12]);
            return [C.e, 0.6];
          }
          if (s < 2) return [C.d, 0.55];
          if (s === 2) return [C.d, 0.85];
          return u < 0.7 ? [C.c, 0.3 + 0.6 * (u / 0.7)] : [C.e, 0.6];
        };
        for (let r = 0; r < 2; r++) {
          const x = xs[r], col = RC(C, r);
          V.box(g, x, hdrY - 14, cw, rowsTop + rowsH - hdrY + 26, { r: 10, fill: V.alpha(col, 0.05), stroke: V.alpha(col, 0.5), lw: 1 });
          V.text(g, 'Spark ' + r + ' · rank ' + r, x + cw / 2, hdrY + 2, { size: 12, weight: 700, align: 'center', color: col, maxW: cw - 8 });
          let sub = eff === 'ar' ? 'BF16 partial' : 'fp32 partial';
          if (eff === 'split') { const H = R / 2; sub = 'owns ' + nf(r * H) + '–' + nf(r * H + H - 1); }
          V.text(g, sub, x + cw / 2, hdrY + 18, { size: 11, font: 'mono', align: 'center', color: C.muted, maxW: cw - 8 });
          for (let i = 0; i < n; i++) {
            const [bc, ba] = bandState(r, i);
            const bx = x + 8, bw = cw - 16, bhh = Math.max(2, bh - 2), by = bandY(i) - bhh / 2;
            V.box(g, bx, by, bw, bhh, { r: Math.min(4, bhh / 2), fill: V.alpha(bc, ba), stroke: eff === 'split' ? V.alpha(RC(C, own(i)), 0.75) : null, lw: 1 });
            if (eff === 'ar') V.line(g, bx + bw / 2, by + 1, bx + bw / 2, by + bhh - 1, { color: V.alpha(C.ink, 0.35), lw: 1 });
          }
          const sumOn = (eff !== 'split' && s === 2) || (eff === 'split' && s === 2 && u < 0.5);
          if (sumOn) plus(g, r === 0 ? mx0 + 18 : mx1 - 18, rowsTop + rowsH / 2, 8, C.c, true);
        }
        // dots crossing the cable
        const dots = (from, list, col, rad) => {
          list.forEach((i) => {
            const d = n > 1 ? (i / (n - 1)) * 0.15 : 0;
            const f = V.clamp((u - d) / 0.8, 0, 1);
            if (f <= 0 || f >= 1) return;
            const e = V.ease.inOut(f);
            const xa = from === 0 ? mx0 - 4 : mx1 + 4, xb = from === 0 ? mx1 + 4 : mx0 - 4;
            V.dot(g, V.lerp(xa, xb, e), bandY(i) + (from === 0 ? -1 : 1) * Math.min(3, bh / 4), rad, col);
          });
        };
        const all = Array.from({ length: n }, (_, i) => i);
        if (eff === 'ar' && (s === 1 || s === 2)) { dots(0, all, C.d, 3); dots(1, all, C.d, 3); }
        if (eff === 'ag' && s === 1) { dots(0, all, C.d, 5); dots(1, all, C.d, 5); }
        if (eff === 'split' && s === 1) { dots(0, all.filter((i) => own(i) === 1), C.d, 5); dots(1, all.filter((i) => own(i) === 0), C.d, 5); }
        if (eff === 'split' && s === 3) { dots(0, all.filter((i) => own(i) === 0), C.e, 4); dots(1, all.filter((i) => own(i) === 1), C.e, 4); }
        // current step
        const sy0 = rowsTop + rowsH + 34;
        wrap(g, (fall ? 'R < 64, no split: ' : '') + STEPS[eff][s], pad, sy0, w - 2 * pad, { size: 12, weight: 700, color: fall ? TF : C.ink, lh: 16, maxLines: 2 });
        // the three closings side by side
        let yy = sy0 + (narrow ? 50 : 38);
        const maxS = calc('ag', R).send;
        const barMaxW = narrow ? w - 2 * pad - 90 : Math.min(420, w * 0.5);
        ['ar', 'ag', 'split'].forEach((k) => {
          const c = calc(k, R), act = k === eff, col = k === 'ar' ? C.glm : TF;
          const off = k === 'split' && R < MIN_ROWS;
          yy = wrap(g, NAMES[k][0] + ' · ' + NAMES[k][1], pad, yy, w - 2 * pad, { size: 11.5, weight: act ? 700 : 500, color: act ? C.ink : C.muted, lh: 14, maxLines: 2 });
          if (off) { yy = wrap(g, 'not used: below 64 rows the recipe doesn\'t split and closes the pair with the all-gather above', pad, yy + 1, w - 2 * pad, { size: 11, color: C.muted, lh: 14, maxLines: 2 }) + 8; return; }
          const bw = Math.max(3, (c.send / maxS) * barMaxW);
          V.box(g, pad, yy - 9, bw, 12, { r: 4, fill: V.alpha(col, off ? 0.15 : act ? 0.75 : 0.35), stroke: act ? col : null, lw: 1.2 });
          V.text(g, fmtB(c.send), pad + bw + 8, yy + 1, { size: 11, font: 'mono', weight: 700, color: act ? C.ink : C.muted });
          const det = 'glue on ' + nf(c.glue) + (c.glue === 1 ? ' row' : ' rows') + ' per rank · ' + (c.msgs > 1 ? c.msgs + ' exchanges' : '1 collective') + ' · ' + (c.rnd > 1 ? c.rnd + ' roundings' : '1 rounding');
          yy = wrap(g, det, pad, yy + 17, w - 2 * pad, { size: 11, color: C.muted, lh: 14, maxLines: 2 }) + 8;
        });
        // footer: the transport and the measurements
        let note;
        if (eff === 'ar') note = 'vLLM recipe: PyNCCL between the nodes, 20–50 µs per collective in the recipes\' audits (section 9). Half the bytes of the fp32 all-gather, with two roundings.';
        else if (eff === 'split') note = 'Exchanges via NCCL send/recv on a second stream, overlapped with the “front” of the next block. Measured with 0009 and 0020 together: 50k prefill from about 1,270 to 1,730 tok/s; decode gets about 1.5% slower.';
        else if (R * 16 <= 512) note = nf(R) + ' × 16 KiB = ' + fmtB(R * 16 * KIB) + ' ≤ 512 KiB: it goes over the one-shot RDMA (TF_ROCE_MAX_KB=512). Measured: 16 KiB in 11 µs, versus 45 µs on NCCL; decode +6%.' + (fall ? ' Decode always takes this path.' : '');
        else note = nf(R) + ' × 16 KiB = ' + fmtB(R * 16 * KIB) + ' > 512 KiB: it goes over NCCL, on both rails of the cable. Measured: 32 MB (R = 2,048) in 1.9 ms; with one rail, 3.5 ms. In prefill, the recipe uses the split path.';
        wrap(g, note, pad, yy + 6, w - 2 * pad, { size: 11, color: C.muted, lh: 14, maxLines: 6 });
      });
    })();

    /* ================== F9 — the network bill ============================== */
    (function () {
      const fig = root.querySelector('#fig-rede');
      if (!fig) return;
      adaptH(fig, (w) => (w < 640 ? 560 : 430));
      const PR = {
        qwen: { d: 2560, kspec: 3, nDec: 106, nPre: 96, chunk: 8192, seqs: 1, ref: 'Measured: 2–3 ms/step. Repo estimate: 150–180 ms of network per 8192-token chunk.', cnt: 'audit count' },
        glm: { d: 4096, kspec: 7, nDec: 90, nPre: 90, chunk: 1024, seqs: 1, ref: 'The repo estimates 378 MB and 15–20 ms per 1024 tokens counting 1 all-reduce per layer; here we count 2 (interpretation).', cnt: 'our count (2 per layer)' },
        ds: { d: 4096, kspec: 6, nDec: 108, nPre: 86, chunk: 1024, seqs: 1, ref: 'Measured: 3–5 ms/step. 86 all-reduces of 8 MB per chunk ≈ 690 MB per rank.', cnt: 'audit count' },
      };
      const CHUNKS = [256, 512, 1024, 2048, 4096, 8192];
      const ST = { pre: 'qwen', a: 25, b: 25, seqs: 1, ci: 5 };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Recipe', options: [{ v: 'glm', label: 'GLM' }, { v: 'qwen', label: 'Qwen' }, { v: 'ds', label: 'DeepSeek' }], value: 'qwen', onChange: (v) => { ST.pre = v; sC.set(CHUNKS.indexOf(PR[v].chunk)); } });
      UI.seg(ctl, { label: 'Link', options: [{ v: 'roce', label: 'RoCE 200G' }, { v: 'nvl', label: 'NVLink (ref.)' }], value: 'roce', onChange: (v) => { if (v === 'roce') { sA.set(25); sB.set(25); } else { sA.set(5); sB.set(450); } } });
      const sA = UI.slider(ctl, { label: 'latency', min: 2, max: 60, step: 1, value: 25, fmt: (v) => v + ' µs', onInput: (v) => { ST.a = v; } });
      const sB = UI.slider(ctl, { label: 'bandwidth', min: 5, max: 500, step: 5, value: 25, fmt: (v) => v + ' GB/s', onInput: (v) => { ST.b = v; } });
      UI.slider(ctl, { label: 'decode seqs', min: 1, max: 8, step: 1, value: 1, onInput: (v) => { ST.seqs = v; } });
      const sC = UI.slider(ctl, { label: 'prefill chunk', min: 0, max: 5, step: 1, value: 5, fmt: (v) => String(CHUNKS[v]), onInput: (v) => { ST.ci = v; } });
      const fmtT = (us) => (us < 1000 ? nf(us, us < 10 ? 1 : 0) + ' µs' : nf(us / 1000, us < 1e4 ? 2 : 1) + ' ms');
      const fmtB = (b) => (b < 1e6 ? nf(b / 1e3, b < 1e4 ? 1 : 0) + ' KB' : nf(b / 1e6, b < 1e7 ? 1 : 0) + ' MB');
      V.fig(fig, (g, w, h, t) => {
        if (w < 140 || h < 100) return;
        const C = V.C(), P = PR[ST.pre], a = ST.a, b = ST.b;
        const narrow = w < 640, pad = 14;
        const T = (n) => a + (n / (b * 1e9)) * 1e6; // µs, p=2
        const rowsDec = ST.seqs * (1 + P.kspec);
        const nDec = rowsDec * P.d * 2;
        const chunk = CHUNKS[ST.ci];
        const nPre = chunk * P.d * 2;
        const knee = a * b * 1e3;
        // chart
        const gx = pad + 40, gy = 16, gw = narrow ? w - pad - gx - 6 : w * 0.54 - gx, gh = narrow ? 230 : h - gy - 44;
        const lx0 = 3, lx1 = 8, ly0 = 0, ly1 = 5;
        const X = (n) => gx + ((Math.log10(n) - lx0) / (lx1 - lx0)) * gw;
        const Y = (us) => gy + gh - ((Math.log10(us) - ly0) / (ly1 - ly0)) * gh;
        V.box(g, gx, gy, gw, gh, { r: 4, stroke: V.alpha(C.muted, 0.35), lw: 1 });
        const kx = V.clamp(X(knee), gx, gx + gw);
        g.fillStyle = V.alpha(C.b, 0.07); g.fillRect(gx, gy, kx - gx, gh);
        g.fillStyle = V.alpha(C.d, 0.07); g.fillRect(kx, gy, gx + gw - kx, gh);
        for (let e = lx0; e <= lx1; e++) { const xx = X(Math.pow(10, e)); V.line(g, xx, gy, xx, gy + gh, { color: V.alpha(C.muted, 0.15), lw: 1 }); V.text(g, ['1 KB', '10 KB', '100 KB', '1 MB', '10 MB', '100 MB'][e - 3], xx, gy + gh + 15, { size: 11, font: 'mono', align: 'center', color: C.muted }); }
        for (let e = ly0; e <= ly1; e++) { const yy = Y(Math.pow(10, e)); V.line(g, gx, yy, gx + gw, yy, { color: V.alpha(C.muted, 0.15), lw: 1 }); V.text(g, ['1 µs', '10 µs', '100 µs', '1 ms', '10 ms', '100 ms'][e], gx - 5, yy + 4, { size: 11, font: 'mono', align: 'right', color: C.muted }); }
        g.beginPath();
        for (let q = 0; q <= 120; q++) { const lg = lx0 + (q / 120) * (lx1 - lx0), n = Math.pow(10, lg); const yy = Y(V.clamp(T(n), 1, 1e5)); if (q === 0) g.moveTo(X(n), yy); else g.lineTo(X(n), yy); }
        g.strokeStyle = C.a; g.lineWidth = 2.5; g.stroke();
        V.line(g, kx, gy, kx, gy + gh, { color: V.alpha(C.ink, 0.5), lw: 1.2, dash: [4, 4] });
        V.text(g, 'αβ knee = ' + fmtB(knee), kx + 4, gy + 14, { size: 11, font: 'mono', color: C.ink });
        V.text(g, 'latency dominates', gx + 6, gy + gh - 8, { size: 11, font: 'mono', color: C.b });
        V.text(g, 'bandwidth dominates', gx + gw - 6, gy + gh - 8, { size: 11, font: 'mono', color: C.d, align: 'right' });
        const mk = (n, col, lbl, up) => { const xx = X(V.clamp(n, 1e3, 1e8)), yy = Y(V.clamp(T(n), 1, 1e5)); V.glow(g, xx, yy, 14, col, 0.8 + 0.2 * Math.sin(t * 3)); V.dot(g, xx, yy, 5, col); const right = up || xx > gx + gw * 0.62; V.text(g, lbl, xx + (right ? -8 : 8), yy + (up ? -10 : 18), { size: 11, font: 'mono', weight: 700, color: col, align: right ? 'right' : 'left' }); };
        mk(nDec, C.c, 'decode ' + fmtB(nDec), X(nDec) > gx + 90);
        mk(nPre, C.d, 'prefill ' + fmtB(nPre), false);
        V.text(g, narrow ? 'all-reduce, 2 ranks: T(n) = α + n/β' : 'time of 1 all-reduce between 2 ranks: T(n) = α + n/β', gx, gy + gh + 32, { size: 11, font: 'mono', color: C.muted, maxW: gw });
        // panel
        const px = narrow ? pad : w * 0.54 + 16, py = narrow ? gy + gh + 44 : 14, pw = narrow ? w - 2 * pad : w - px - pad, phh = narrow ? h - py - 10 : h - 28;
        V.box(g, px, py, pw, phh, { r: 12, fill: V.alpha(C.muted, 0.07), stroke: V.alpha(C.muted, 0.28), lw: 1 });
        let yy = py + 22;
        const tDec = T(nDec), tPre = T(nPre);
        const totDec = P.nDec * tDec, totPre = P.nPre * tPre;
        V.text(g, 'decode', px + 12, yy, { size: 12.5, weight: 700, color: C.c });
        yy = wrap(g, rowsDec + ' rows (' + ST.seqs + ' seq × (1 + ' + P.kspec + ' drafts)) × ' + P.d + ' × 2 B = ' + fmtB(nDec) + ' per collective', px + 12, yy + 17, pw - 24, { size: 11, color: C.ink, lh: 14 });
        yy = wrap(g, P.nDec + ' collectives × ' + fmtT(tDec) + ' = ' + fmtT(totDec) + ' per step', px + 12, yy + 2, pw - 24, { size: 12, weight: 700, font: 'mono', color: C.ink, lh: 15 });
        const latShare = a / tDec;
        yy = wrap(g, nf(latShare * 100, 0) + '% of this time is latency (α) · ' + P.cnt, px + 12, yy + 2, pw - 24, { size: 11, color: C.muted, lh: 14 });
        yy += 10;
        V.text(g, 'prefill', px + 12, yy + 4, { size: 12.5, weight: 700, color: C.d });
        yy = wrap(g, 'chunk ' + chunk + ' × ' + P.d + ' × 2 B = ' + fmtB(nPre) + ' per all-reduce', px + 12, yy + 21, pw - 24, { size: 11, color: C.ink, lh: 14 });
        yy = wrap(g, P.nPre + ' × ' + fmtT(tPre) + ' = ' + fmtT(totPre) + ' per chunk (' + fmtB(P.nPre * nPre) + ' per rank)', px + 12, yy + 2, pw - 24, { size: 12, weight: 700, font: 'mono', color: C.ink, lh: 15 });
        yy += 8;
        wrap(g, P.ref + (b > 100 ? ' The NVLink preset is illustrative.' : ''), px + 12, yy + 6, pw - 24, { size: 11, color: C.muted, lh: 14 });
      });
    })();

    /* ================== F10 — the floor plan of each GPU ==================== */
    (function () {
      const fig = root.querySelector('#fig-planta');
      if (!fig) return;
      const K = { S: 'sliced', R: 'replicated', E: 'EP', X: 'mixed', N: 'off', W: 'whole' };
      const PL = {
        glm: {
          std: { ranks: 2, rows: [
            ['embedding', 'S', ['ids 0–77,439', 'ids 77,440–154,879']],
            ['MLA (11 layers): q_b', 'S', ['32 heads', '32 heads']],
            ['c_kv latent + indexer', 'R', ['whole', 'whole']],
            ['KDA (34 layers): in_proj', 'X', ['12,576 × 4096', '12,576 × 4096']],
            ['KDA: recurrent state', 'S', ['32 heads', '32 heads']],
            ['KDA: o_proj', 'S', ['row · 4096 × 4096', 'row · 4096 × 4096']],
            ['MLA: o_proj', 'S', ['row · 4096 × 8192', 'row · 4096 × 8192']],
            ['routed experts (EXL3)', 'S', ['288 × ½ (1024)', '288 × ½ (1024)']],
            ['vision tower', 'S', ['weights (½)', 'weights (½)']],
            ['lm_head', 'S', ['77,440 rows', '77,440 rows']],
            ['DFlash2 (drafter)', 'S', ['draft TP 2', 'draft TP 2']],
            ['KV fp8_ds_mla', 'R', ['656 B/tok/layer', '656 B/tok/layer']],
          ], flags: ['--tensor-parallel-size 2', 'draft_tensor_parallel_size 2', 'NNODES=2'], mem: 'weights: 79.65 GiB per rank (82.94 with the KDA BF16 copy) · KV ≈ 7,579 B/token/rank, replicated' },
          alt: { ranks: 3, rows: [
            ['embedding', 'S', ['51,648 ids', '51,648 ids', '51,648 ids']],
            ['MLA: q_b (66 heads)', 'S', ['22 heads', '22 heads', '22 heads']],
            ['c_kv latent + indexer', 'R', ['whole', 'whole', 'whole']],
            ['KDA: in_proj', 'X', ['8,726 × 4096', '8,726 × 4096', '8,726 × 4096']],
            ['KDA: recurrent state', 'S', ['22 heads', '22 heads', '22 heads']],
            ['KDA and MLA: o_proj', 'S', ['row · input ⅓', 'row · input ⅓', 'row · input ⅓']],
            ['routed experts (EXL3)', 'E', ['0–95', '96–191', '192–287']],
            ['vision tower', 'R', ['data (whole)', 'data (whole)', 'data (whole)']],
            ['lm_head (vocab 154,944)', 'S', ['51,648', '51,648', '51,648']],
            ['DFlash2 (GQA 36/9)', 'X', ['TP 1 (only here)', '—', '—']],
            ['KV fp8_ds_mla', 'R', ['same', 'same', 'same']],
          ], flags: ['--tensor-parallel-size 3', '--enable-expert-parallel', '--mm-encoder-tp-mode data', 'heads 64 → 66', 'DFLASH_DRAFT_TP=1'], mem: 'weights: ~55.75 GiB per node · the drafter lives only on rank 0 (“the divisibility tax”)' },
        },
        qwen: {
          std: { ranks: 2, rows: [
            ['embedding', 'S', ['124,160 ids', '124,160 ids']],
            ['GDN (36): qkvz / out', 'S', ['8192×2560 / 2560×3072', '8192×2560 / 2560×3072']],
            ['QSA (12): qkv+gate / out', 'S', ['6656×2560 / 2560×3072', '6656×2560 / 2560×3072']],
            ['QSA KV', 'S', ['1 of 2 KV heads', '1 of 2 KV heads']],
            ['QSA indexer', 'R', ['640 × 2560', '640 × 2560']],
            ['hyper-connections', 'R', ['1.19 GiB', '1.19 GiB']],
            ['router', 'R', ['whole', 'whole']],
            ['routed experts (NVFP4)', 'E', ['0–255', '256–511']],
            ['shared expert gate/up', 'S', ['640 × 2560', '640 × 2560']],
            ['PLE (N-gram table)', 'S', ['~160 M rows', '~160 M rows']],
            ['lm_head', 'S', ['124,160 × 2560', '124,160 × 2560']],
            ['vision tower', 'R', ['data · 0.84 GiB', 'data · 0.84 GiB']],
            ['MTP: reduced lm_head', 'S', ['45,734 ids', '1,415 ids']],
          ], flags: ['--tensor-parallel-size 2', '--enable-expert-parallel', '--all2all-backend allgather_reducescatter', '--mm-encoder-tp-mode data'], mem: 'weights: 64.46 GiB per GPU (measured) · KV 32.02 GiB → 3,652,200 tokens (FP8, GMU 0.835)' },
          alt: { ranks: 1, rows: [
            ['embedding + lm_head', 'W', ['248,320 ids']],
            ['GDN + QSA', 'W', ['all heads']],
            ['QSA KV', 'W', ['2 KV heads']],
            ['routed experts', 'W', ['512 experts']],
            ['hyper-connections + router', 'W', ['whole']],
            ['PLE (N-gram table)', 'X', ['mmap offload (26.82 GiB)']],
            ['MTP', 'N', ['MTP_NUM_SPECULATIVE_TOKENS=0']],
          ], flags: ['start-tp1.sh', 'TP1_MODEL_ID local-inference-lab NVFP4 (98.6 GiB)', 'MAX_MODEL_LEN 65536', 'MAX_NUM_SEQS 4'], mem: 'a single Spark: nothing is split, and the PLE leaves the GPU via mmap · 685,825 KV tokens with KV_TARGET_GIB=20 (tp1/.env.sample, GMU 0.813); the 8 GiB default gives fewer' },
        },
        ds: {
          std: { ranks: 2, rows: [
            ['embedding', 'S', ['64,640 ids', '64,640 ids']],
            ['fused_wqa_wkv', 'R', ['whole', 'whole']],
            ['wq_b + attn_sink', 'S', ['32 heads', '32 heads']],
            ['wo_a (per group)', 'S', ['4 groups', '4 groups']],
            ['wo_b', 'S', ['row · 4096 × 4096', 'row · 4096 × 4096']],
            ['KV (SWA + compressed)', 'R', ['whole', 'whole']],
            ['indexer', 'R', ['whole', 'whole']],
            ['experts (B12xExperts)', 'S', ['256 × ½ (1024)', '256 × ½ (1024)']],
            ['lm_head', 'S', ['64,640 rows', '64,640 rows']],
            ['Markov head', 'S', ['vocab ½ · 12 coll.', 'vocab ½ · 12 coll.']],
            ['DSpark (3 stages)', 'S', ['TP 2', 'TP 2']],
          ], flags: ['--tensor-parallel-size 2', '--pipeline-parallel-size 1', 'no EP (MegaMoE: SM100 + EP)', 'NNODES=2'], mem: 'weights: 79.17 GiB per rank · KV 17.04 GiB → 2,331,430 tokens (2.22× at 1M), replicated' },
          alt: { ranks: 3, rows: [
            ['embedding (vocab 129,408)', 'S', ['43,136 ids', '43,136 ids', '43,136 ids']],
            ['fused_wqa_wkv', 'R', ['whole', 'whole', 'whole']],
            ['wq_b (72 heads)', 'S', ['24 heads', '24 heads', '24 heads']],
            ['wo_a (9 groups)', 'S', ['3 groups', '3 groups', '3 groups']],
            ['wo_b', 'S', ['row', 'row', 'row']],
            ['KV + indexer', 'R', ['whole', 'whole', 'whole']],
            ['experts (2112)', 'S', ['704 per rank', '704 per rank', '704 per rank']],
            ['lm_head', 'S', ['43,136', '43,136', '43,136']],
            ['DSpark', 'S', ['padded sink', 'padded sink', 'padded sink']],
          ], flags: ['start-tp3.sh', '--tensor-parallel-size 3', 'apply_tp3_patch.py (groups 8 → 9)', 'NNODES=3'], mem: 'weights: 60.69 GiB per rank · KV ≈ 35 GiB/rank → 5.0 M tokens, 16 slots · TTFT +22–23% at 128K–256K' },
        },
      };
      const ST = { r: 'glm', alt: false };
      const reH = adaptH(fig, (w) => { const n = PL[ST.r][ST.alt ? 'alt' : 'std'].rows.length; return w < 600 ? 44 + n * 42 + 150 : 44 + n * 34 + 104; });
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Recipe', options: [{ v: 'glm', label: 'GLM-5.3' }, { v: 'qwen', label: 'Qwen3.8' }, { v: 'ds', label: 'DeepSeek-V4' }], value: 'glm', onChange: (v) => { ST.r = v; if (reH) reH(); } });
      UI.seg(ctl, { label: 'Topology', options: [{ v: 'std', label: 'default (TP2)' }, { v: 'alt', label: 'alternative' }], value: 'std', onChange: (v) => { ST.alt = v === 'alt'; if (reH) reH(); } });
      const C0 = V.C();
      UI.legend(ctl, [[C0.a, 'slice · rank 0'], [C0.b, 'slice · rank 1'], [C0.d, 'slice · rank 2'], [C0.muted, 'replica (plain hatching)'], [C0.e, 'EP: whole experts (green checkerboard)'], [C0.f, 'mixed']]);
      V.fig(fig, (g, w, h, t) => {
        if (w < 140 || h < 100) return;
        const C = V.C(), D = PL[ST.r][ST.alt ? 'alt' : 'std'];
        const narrow = w < 600, pad = 12;
        const nr = D.ranks;
        const nameW = narrow ? 118 : Math.min(230, w * 0.28);
        const x0 = pad, x1 = w - pad;
        const cw = (x1 - x0 - nameW) / nr;
        // the footer grows if the flags + memory text don't fit in the base height
        const footNeed = (() => {
          let fx = x0 + 10, fyy = 18;
          D.flags.forEach((f) => { const ww = tw(g, f, 11, 'mono', 600) + 16; if (fx + ww > x1 - 10) { fx = x0 + 10; fyy += 24; } fx += Math.min(ww, x1 - x0 - 20) + 6; });
          return fyy + 28 + (wrapLines(g, D.mem, x1 - x0 - 20, 11.5, 'sans', 400).length - 1) * 15 + 14;
        })();
        const footH = Math.max(narrow ? 140 : 92, footNeed);
        const hy = 16;
        const rows = D.rows;
        const rh = Math.min(narrow ? 42 : 34, (h - hy - 22 - footH) / rows.length);
        const rankName = (r) => (narrow && nr > 2 ? 'rank ' + r : r === 0 ? 'rank 0 · head' : 'rank ' + r + ' · worker');
        for (let r = 0; r < nr; r++) V.text(g, nr === 1 ? 'single GPU (TP1)' : rankName(r), x0 + nameW + r * cw + cw / 2, hy + 4, { size: narrow ? 11 : 11.5, font: 'mono', weight: 700, align: 'center', color: nr === 1 ? C.ink : RC(C, r), maxW: cw - 6 });
        const scan = (t * 0.35) % 1;
        rows.forEach(([name, k, cells], q) => {
          const y = hy + 14 + q * rh;
          const onScan = Math.abs(scan * rows.length - q - 0.5) < 0.5;
          if (onScan) V.box(g, x0, y, x1 - x0, rh - 2, { r: 6, fill: V.alpha(C.d, 0.06) });
          const tagCol = k === 'S' || k === 'W' ? C.ink : k === 'R' ? C.muted : k === 'E' ? C.e : k === 'X' ? C.f : C.muted;
          if (narrow) {
            V.text(g, name, x0 + 2, y + rh / 2 - 3, { size: 11, weight: 700, color: C.ink, maxW: nameW - 6 });
            V.text(g, K[k], x0 + 2, y + rh / 2 + 11, { size: 11, font: 'mono', color: tagCol });
          } else {
            const cwid = tw(g, K[k], 11, 'mono', 600) + 14;
            V.text(g, name, x0 + 2, y + rh / 2 + 4, { size: 11.5, weight: 700, color: C.ink, maxW: nameW - cwid - 18 });
            chip(g, K[k], x0 + nameW - 8, y + rh / 2, tagCol, { align: 'right', size: 11, h: 19 });
          }
          for (let r = 0; r < nr; r++) {
            const cx = x0 + nameW + r * cw + 3, cy = y + 2, ww = cw - 6, hh = rh - 6;
            const txt = cells[r] != null ? cells[r] : cells[0];
            const col = nr === 1 ? C.a : RC(C, r);
            if (txt === '—') { V.box(g, cx, cy, ww, hh, { r: 5, stroke: V.alpha(C.muted, 0.35), lw: 1 }); V.text(g, '—', cx + ww / 2, cy + hh / 2 + 0.5, { size: 11, align: 'center', base: 'middle', color: C.muted }); continue; }
            if (k === 'R' || (k === 'N')) { V.box(g, cx, cy, ww, hh, { r: 5, fill: V.alpha(C.muted, 0.1), stroke: V.alpha(C.muted, 0.55), lw: 1 }); hatch(g, cx + 1, cy + 1, ww - 2, hh - 2, V.alpha(C.muted, 0.35), 7, t * 6); }
            else if (k === 'E') {
              // EP: its own color (not a rank color) and a checkerboard, so it isn't confused with rank 2's slice
              V.box(g, cx, cy, ww, hh, { r: 5, fill: V.alpha(C.e, 0.12), stroke: C.e, lw: 1.6 });
              g.save(); g.translate(cx + ww, cy); g.scale(-1, 1); hatch(g, 0, 1, ww - 1, hh - 2, V.alpha(C.e, 0.4), 8, -t * 6, 1); g.restore();
              hatch(g, cx + 1, cy + 1, ww - 2, hh - 2, V.alpha(C.e, 0.4), 8, t * 6, 1);
            }
            else if (k === 'X') { V.box(g, cx, cy, ww, hh, { r: 5, fill: V.alpha(col, 0.18), stroke: C.f, lw: 1.2 }); hatch(g, cx + ww * 0.75, cy + 1, ww * 0.25 - 1, hh - 2, V.alpha(C.muted, 0.45), 5, t * 6); }
            else V.box(g, cx, cy, ww, hh, { r: 5, fill: V.alpha(col, 0.22), stroke: V.alpha(col, 0.75), lw: 1 });
            if (k === 'E') { const tw0 = Math.min(ww - 8, tw(g, txt, 11, 'mono', 600) + 10); V.box(g, cx + ww / 2 - tw0 / 2, cy + hh / 2 - 8, tw0, 16, { r: 4, fill: V.alpha(C.bg || C.panel, 0.85) }); }
            V.text(g, txt, cx + ww / 2, cy + hh / 2 + 0.5, { size: 11, font: 'mono', weight: 600, align: 'center', base: 'middle', color: C.ink, maxW: ww - 6 });
          }
        });
        // footer
        const fy = h - footH;
        V.box(g, x0, fy, x1 - x0, footH - 6, { r: 10, fill: V.alpha(C.muted, 0.07), stroke: V.alpha(C.muted, 0.28), lw: 1 });
        let fx = x0 + 10, fyy = fy + 18;
        D.flags.forEach((f) => {
          const ww = tw(g, f, 11, 'mono', 600) + 16;
          if (fx + ww > x1 - 10) { fx = x0 + 10; fyy += 24; }
          chip(g, f, fx, fyy, C.c, { size: 11, h: 20, maxW: x1 - x0 - 20 });
          fx += Math.min(ww, x1 - x0 - 20) + 6;
        });
        wrap(g, D.mem, x0 + 10, fyy + 28, x1 - x0 - 20, { size: 11.5, color: C.ink, lh: 15 });
      });
    })();
  },
});

LA.chapter({
  id: 'tp3-tp4',
  part: 8,
  order: 40,
  title: 'Beyond 2 nodes: TP=3, TP=4 and divisibility',
  navTitle: 'TP=3 & TP=4',
  subtitle: 'When 64 heads don\'t divide by 3: padding, whole experts, directed rings, the TP=4 hang and when growing is worth it.',
  minutes: 62,
  recipes: ['glm', 'qwen', 'ds'],
  tags: ['TP=3', 'TP=4', 'divisibility', 'padding', 'expert parallel', 'directed ring', 'switchless ring', 'NCCL', 'stall'],
  goals: [
    'Audit a <code>config.json</code> and say, dimension by dimension, what <b>divides</b>, what needs <b>padding</b>, what goes to <b>expert parallel</b> and what is <b>replicated</b>',
    'Compute padding with the ceiling and the lcm, and pick the right <b>neutral element</b> (zeros, sink −∞)',
    'Explain why DeepSeek pads <b>groups</b> (8 → 9) while GLM pads <b>heads</b> (64 → 66) and moves its experts to EP',
    'Build the 3-node mesh (directed ring, HCAs, subnet-aware routing, control plane on the LAN), recognize a distributed stall, and choose between a switch and a switchless ring for four nodes',
    'Decide when TP=3 or TP=4 pays off, using the measured numbers from the recipes',
  ],
  prereqs: ['paralelismos', 'nccl-rdma'],
  summary: [
    'TP requires \\(p \\mid n\\) in every <b>sharded</b> dimension. <b>Replicated</b> dimensions (hidden, MLA latent KV, indexer) don\'t need to divide, but they don\'t shrink with more nodes either.',
    'Padding: \\(n\' = \\lceil n/L \\rceil L\\) with \\(L = \\operatorname{lcm}(a, p)\\). The pad uses the operation\'s <b>neutral element</b>: zeros in sums, −∞ in the softmax sink. The loaders copy the real intersection and zero out the rest.',
    'DeepSeek TP=3 pads <b>groups</b> (8 → 9, 72 heads) to preserve the BMM contract r = 8 × 512 = 4096. R1 (sharding heads within the group) would give r = 1536, which the kernel rejects. Validation is <code>validate_tp3.sh</code> (7/7).',
    'GLM TP=3: 66 heads via <code>--hf-overrides</code> (never 96), vocabulary 154,944, shared 2112, <code>A_log</code> 66, Q pad 22 → 32 only inside the SM120 kernel and <b>EP</b> with 96 experts per rank, because the 2048-wide EXL3 trellis can\'t be sharded. The drafter runs at draft TP 1 with GQA 36/9.',
    'Three nodes without a switch: directed ring (Port0 → the next node\'s Port1), one /24 per cable, both HCAs, <code>MERGE_NICS=0</code>, /24 subnet-aware routing and the control plane on the shared LAN (<code>enP7s7</code>). Four nodes have two ports each: either a switch, or a ring in which the ranks are placed so that partners share a cable (0-1-3-2), NCCL stays off tree and PAT (sparkring\'s <code>NCCL_SWITCHLESS_RING_ONLY</code>, or ring-only graph files) and the pair across pays a relayed hop.',
    'TP=4 divides cleanly but exposed a distributed stall. <code>VLLM_SM120_<wbr>SPARSE_MLA_<wbr>SLICE_TOKENS=64</code> caps the final call at ≤ 64 rows, which is a mitigation, not a fix for the race.',
    'Growing buys <b>capacity</b> (DS: 16 slots and 5.02 M tokens; GLM: weights per node 79.65 → ~55.75 GiB) and aggregate throughput, and almost nothing in single-user latency (DS: +9% at 1 stream, far from the ~1.4× the bytes would suggest): the fixed cost grows and long prefill gets worse (DS +22–23% at 128K–256K).',
  ],
  quiz: [
    { q: 'Why does DeepSeek\'s R2 patch pad the <b>group</b> count (8 → 9) instead of splitting the heads within the groups?', o: ['Because groups are cheaper to pad than heads', 'Because the <code>wo_a</code> BMM requires r = heads per group × head_dim = 4096, fixed in the checkpoint', 'Because NCCL only transmits whole groups', 'Because the vocabulary also has 8 groups'], a: 1, e: 'With <code>wo_a = [8192, 4096]</code>, each group comes in with r = 8 × 512 = 4096. Sharding heads within the group (R1) would give r = 1536, and the kernel rejects it (<code>einsum.hpp:164</code>). Padding groups keeps 8 heads per group: 72 heads, 24 per rank. The other options have nothing to do with the shape contract.' },
    { q: 'The fake heads of the padding group get <code>attn_sink = −∞</code>. What would happen with sink = 0?', o: ['Nothing, zero is always inert', 'The kernel would reject the shape', 'e⁰ = 1 would add a phantom term to the softmax denominator and attenuate the real heads', 'The vocabulary would lose 64 ids'], a: 2, e: 'Zero is neutral in weighted sums, not in the exponential denominator of the softmax. There the neutral element is −∞, because e<sup>−∞</sup> = 0. That\'s why the sink has its own loaders and doesn\'t go through the generic loader, which does <code>zero_()</code>.' },
    { q: 'In GLM\'s TP=3, why do the routed experts use expert parallel instead of padding <code>moe_intermediate_size</code>?', o: ['Because EP is always faster than TP', 'Because 2048 doesn\'t divide by 3, the EXL3 trellis is already packed at 2048 and the 288 experts divide cleanly (96 per rank)', 'Because Marlin doesn\'t accept 2112', 'Because the drafter requires EP'], a: 1, e: 'The EXL3 trellis tiles are already encoded at width 2048, and the README forbids padding that dimension. The expert count (288 = 2⁵·3²) divides by 3, so each rank gets 96 whole experts. EP is not “always faster”: it shifts the bottleneck to the most loaded rank.' },
    { q: 'The DFlash2 drafter uses GQA with 32 Q / 8 KV. Why is the padding 36/9 and not 33/9?', o: ['Because 33 is not a multiple of 3', 'To keep the 4:1 ratio, since FlashInfer requires qo % kv == 0 (33 % 9 = 6; 36 % 9 = 0)', 'Because 36 is in the table {8, 16, 32, 64, 128}', 'Because the vocabulary is a multiple of 9'], a: 1, e: '33 is a multiple of 3 and passes the world-TP assert, but it breaks the Q:KV ratio: 33/9 = 3.67. With KV 8 → 9 and Q = 9 × 4 = 36, the ratio stays at 4, and a cut by 3 would give 12/3.' },
    { q: 'Why, at TP=3, does the control plane (Gloo, NCCL bootstrap, TP TCP) move to the <code>enP7s7</code> LAN?', o: ['Because the LAN is faster than the ConnectX-7', 'Because each CX7 cable is a point-to-point /24 that only reaches one neighbor, and Gloo accepts a single interface name', 'Because RoCE can\'t carry TCP', 'To save pinned memory'], a: 1, e: 'Data stays on RoCE via <code>NCCL_IB_HCA</code> with both ports. Only control traffic, which must reach every rank through a single interface, moves to the shared LAN. Never use <code>lo</code>: Gloo would bind to 127.0.0.1.' },
    { q: 'What does <code>VLLM_SM120_<wbr>SPARSE_MLA_<wbr>SLICE_TOKENS=64</code> do in GLM\'s TP=4?', o: ['It fixes the race condition in the sparse-MLA kernel', 'It runs the final sparse-MLA call in slices of at most 64 query rows, where progress was stopping. It is a soak-qualified mitigation that does not fix the race', 'It reduces the maximum context to 64 tokens', 'It turns on 64 NCCL channels'], a: 1, e: 'The patch itself states that it neither identifies nor fixes the race: it bounds the call in which progress was stopping. It is source-exact (SHA-256 of the pre-image and of the result), fail-closed outside the qualified geometry and only accepts 0 or 64.' },
  ],
  mount(root, LA) {
    const V = LA.viz, UI = LA.ui;

    /* ------------------------------------------------------------ helpers */
    const RC = (C, r) => [C.a, C.b, C.d, C.c, C.e, C.f, C.glm, C.ds][r % 8];
    const FC = {};
    const fontOf = (size, font, weight) => { const k = size + '|' + font + '|' + weight; return FC[k] || (FC[k] = V.font(size, font, weight)); };
    const tw = (g, s, size = 12, font = 'sans', weight = 400) => { g.font = fontOf(size, font, weight); return g.measureText(String(s)).width; };
    // quebra um texto em linhas que caibam em maxW (mede com a fonte do canvas)
    function wrapLines(g, str, maxW, size = 12, font = 'sans', weight = 400) {
      const words = String(str).split(' '), out = [];
      let cur = '';
      words.forEach((wd) => { const tr = cur ? cur + ' ' + wd : wd; if (cur && tw(g, tr, size, font, weight) > maxW) { out.push(cur); cur = wd; } else cur = tr; });
      if (cur) out.push(cur);
      return out;
    }
    const gcd = (a, b) => (b ? gcd(b, a % b) : a);
    const lcm = (a, b) => (a / gcd(a, b)) * b;
    const nf = (x, d = 0) => Number(x).toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d });
    const pct = (v) => (v < 0.1 ? nf(v, 2) : nf(v, 1)) + '%';
    function chip(g, s, x, y, col, o = {}) {
      const size = o.size || 11.5, font = o.font || 'mono', weight = o.weight || 600, hh = o.h || 20;
      let wd = tw(g, s, size, font, weight) + 16;
      if (o.maxW && wd > o.maxW) wd = o.maxW;
      const X = o.align === 'right' ? x - wd : o.align === 'center' ? x - wd / 2 : x;
      V.box(g, X, y - hh / 2, wd, hh, { r: hh / 2, fill: V.alpha(col, o.fa == null ? 0.16 : o.fa), stroke: V.alpha(col, 0.8), lw: o.lw || 1 });
      V.text(g, s, X + wd / 2, y + 0.5, { size, font, weight, color: o.color || V.C().ink, align: 'center', base: 'middle', maxW: wd - 12 });
      return wd;
    }
    function hatch(g, x, y, w, h, col, step = 6, off = 0) {
      if (w <= 0 || h <= 0) return;
      g.save(); g.beginPath(); g.rect(x, y, w, h); g.clip();
      g.strokeStyle = col; g.lineWidth = 1.2; g.beginPath();
      const o = ((off % step) + step) % step;
      for (let k = -h - step + o; k < w + step; k += step) { g.moveTo(x + k, y + h); g.lineTo(x + k + h, y); }
      g.stroke(); g.restore();
    }
    function dashedBox(g, x, y, w, h, col, r = 3) { g.setLineDash([3, 3]); V.box(g, x, y, w, h, { r, stroke: col, lw: 1.2 }); g.setLineDash([]); }
    function bestGrid(n, W, Hh, gapK = 1.15, maxS = 28) {
      let best = { cols: 1, rows: n, s: 0 };
      for (let cols = 1; cols <= n; cols++) {
        const rows = Math.ceil(n / cols);
        const s = Math.min(W / (cols * gapK), Hh / (rows * gapK));
        if (s > best.s) best = { cols, rows, s };
      }
      best.s = Math.min(maxS, best.s);
      return best;
    }
    const approach = (cur, tgt, k) => cur + (tgt - cur) * k;
    // pinta o valor de um readout conforme o estado (bad = vermelho; senão, a cor padrão do tema)
    const tone = (r, bad) => { const b = r.el.querySelector('b'); if (b) b.style.color = bad ? 'var(--red)' : ''; };
    const GCOL = (C) => [C.a, C.b, C.c, C.d, C.f, V.mix(C.a, C.b, 0.5), V.mix(C.d, C.c, 0.5), V.mix(C.e, C.d, 0.55), C.muted];

    /* ============================================ F1 — scanner de divisibilidade */
    (function () {
      const fig = root.querySelector('#fig-scanner');
      const stage = fig.querySelector('.la-fig-stage');
      if (fig.clientWidth < 600) stage.style.height = '520px';
      const MODELS = {
        glm: { name: 'GLM-5.3-Flash', rows: [
          { l: 'Attention heads (MLA)', k: 'num_attention_heads', n: 64, rule: 'ceil' },
          { l: 'KDA heads + A_log', k: 'linear_num_heads', n: 64, rule: 'ceil' },
          { l: 'Vocabulary', k: 'vocab_size', n: 154880, rule: 'lcm64' },
          { l: 'Routed expert (EXL3 trellis)', k: 'moe_intermediate_size', n: 2048, rule: 'exl3', experts: 288 },
          { l: 'Shared expert (BF16)', k: 'shared I', n: 2048, rule: 'lcm64' },
          { l: 'Dense MLP (layers 0–2)', k: 'intermediate_size', n: 12288, rule: 'ceil' },
          { l: 'DFlash2 drafter (Q/KV)', k: 'draft heads', n: 32, kv: 8, rule: 'draft' },
          { l: 'Hidden (residual stream)', k: 'hidden_size', n: 4096, rule: 'rep' },
          { l: 'MLA latent KV', k: 'kv_lora_rank', n: 512, rule: 'rep' },
        ] },
        ds: { name: 'DeepSeek-V4-Flash', rows: [
          { l: 'Query heads', k: 'num_attention_heads', n: 64, rule: 'groups', G: 8 },
          { l: 'Output groups', k: 'o_groups', n: 8, rule: 'ceil' },
          { l: 'Vocabulary', k: 'vocab_size', n: 129280, rule: 'lcm64' },
          { l: 'Expert width (EP off)', k: 'moe_intermediate_size', n: 2048, rule: 'lcm64' },
          { l: 'Routed experts', k: 'n_routed_experts', n: 256, rule: 'csl' },
          { l: 'Hidden (residual stream)', k: 'hidden_size', n: 4096, rule: 'rep' },
          { l: 'Latent KV (1 × 512/token)', k: 'head_dim', n: 512, rule: 'rep' },
          { l: 'Lightning indexer', k: 'index_n_heads', n: 64, rule: 'rep', repTxt: 'ReplicatedLinear' },
        ] },
        qwen: { name: 'Qwen3.8-Flash-Next (analysis)', rows: [
          { l: 'Q heads (QSA)', k: 'num_attention_heads', n: 24, rule: 'strict' },
          { l: 'KV heads (QSA)', k: 'num_key_value_heads', n: 2, rule: 'kv' },
          { l: 'GDN: Q/K heads', k: 'linear_num_key_heads', n: 16, rule: 'strict' },
          { l: 'GDN: V heads', k: 'linear_num_value_heads', n: 48, rule: 'strict' },
          { l: 'Experts (EP)', k: 'num_experts', n: 512, rule: 'ep' },
          { l: 'Vocabulary', k: 'vocab_size', n: 248320, rule: 'lcm64' },
          { l: 'Vision MLP', k: 'vision intermediate', n: 4304, rule: 'vision' },
          { l: 'Hidden (residual stream)', k: 'hidden_size', n: 2560, rule: 'rep' },
        ] },
      };
      function evalRow(r, p) {
        const ok = (loc, txt) => ({ st: 'ok', txt: txt || nf(loc) + '/rank', np: r.n });
        switch (r.rule) {
          case 'rep': return { st: 'rep', txt: r.repTxt || 'replicated', np: r.n };
          case 'csl': return p === 1 ? ok(r.n, nf(r.n) + ' whole') : { st: 'csl', txt: 'all · 1/' + p + ' of width', txtN: 'all · 1/' + p + ' width', np: r.n };
          case 'ceil': {
            if (r.n % p === 0) return ok(r.n / p);
            const np = Math.ceil(r.n / p) * p; return { st: 'pad', txt: 'pad → ' + nf(np) + ' (' + nf(np / p) + '/rank)', np };
          }
          case 'strict': return r.n % p === 0 ? ok(r.n / p) : { st: 'bad', txt: r.n + ' % ' + p + ' ≠ 0', np: r.n };
          case 'lcm64': {
            const L = lcm(64, p);
            if (r.n % L === 0) return ok(r.n / p);
            const np = Math.ceil(r.n / L) * L; return { st: 'pad', txt: 'pad → ' + nf(np), np };
          }
          case 'groups': {
            if (r.G % p === 0) return ok(r.n / p, nf(r.n / p) + '/rank · ' + r.G / p + ' groups');
            const Gp = Math.ceil(r.G / p) * p, np = Gp * (r.n / r.G);
            return { st: 'pad', txt: 'pad → ' + np + ' (' + np / p + '/rank)', np };
          }
          case 'exl3': {
            const loc = r.n / p;
            if (r.n % p === 0 && loc % 16 === 0) return ok(loc, nf(loc) + '/rank (sharded)');
            if (r.experts % p === 0) return { st: 'ep', txt: 'EP: ' + r.experts / p + ' experts/rank', np: r.n };
            return { st: 'bad', txt: '2048 and 288 don\'t divide', np: r.n };
          }
          case 'ep': return r.n % p === 0 ? { st: 'ep', txt: 'EP: ' + r.n / p + ' experts/rank', np: r.n } : { st: 'bad', txt: r.n + ' % ' + p + ' ≠ 0', np: r.n };
          case 'kv':
            if (p <= r.n) return r.n % p === 0 ? ok(r.n / p) : { st: 'bad', txt: r.n + ' % ' + p + ' ≠ 0', np: r.n };
            return p % r.n === 0 ? { st: 'rep', txt: 'each KV on ' + p / r.n + ' ranks', np: r.n } : { st: 'bad', txt: p + ' % ' + r.n + ' ≠ 0 (replica)', np: r.n };
          case 'draft': {
            if (r.n % p === 0 && r.kv % p === 0) return ok(0, p === 1 ? '32/8 whole' : 'draft TP=' + p + ': ' + r.n / p + '/' + r.kv / p);
            const kv2 = Math.ceil(r.kv / p) * p, q2 = kv2 * (r.n / r.kv);
            return { st: 'pad', txt: 'draft TP=1 · pad ' + q2 + '/' + kv2, np: q2 };
          }
          case 'vision': {
            const loc = r.n / p;
            if (p === 1) return ok(loc);
            if (Number.isInteger(loc) && loc % 16 === 0) return ok(loc);
            return { st: 'data', txt: 'data mode (replica)', np: r.n };
          }
        }
        return { st: 'rep', txt: '—', np: r.n };
      }
      const ICON = { ok: '✓ ', pad: '⊕ ', ep: '▦ ', rep: '≡ ', bad: '✕ ', data: '≡ ', csl: '◫ ' };
      const stCol = (C, st) => ({ ok: C.e, pad: C.d, ep: C.b, rep: C.muted, bad: C.red, data: C.muted, csl: C.a })[st];
      const S = { m: 'glm', p: 3, t0: 0 };
      const ctl = UI.controls(fig);
      let H = null;
      UI.seg(ctl, { label: 'Model', options: [{ v: 'glm', label: 'GLM' }, { v: 'ds', label: 'DeepSeek' }, { v: 'qwen', label: 'Qwen (analysis)' }], value: 'glm', onChange: (v) => { S.m = v; S.t0 = H ? H.t : 0; } });
      UI.seg(ctl, { label: 'TP', options: [1, 2, 3, 4, 5, 6, 8], value: 3, onChange: (v) => { S.p = v; S.t0 = H ? H.t : 0; } });
      function drawBar(g, C, r, e, p, x, y, w, h, t) {
        const st = e.st;
        if (st === 'rep' || st === 'data') {
          const k = Math.min(p, 4);
          for (let j = k - 1; j >= 0; j--) V.box(g, x + j * 3, y - j * 2.5, w - 9, h, { r: 4, fill: V.alpha(C.muted, 0.14 + 0.07 * (k - j)), stroke: V.alpha(C.muted, 0.55), lw: 1 });
          V.text(g, p > 1 ? '×' + p + ' copies' : 'whole', x + 8, y + h / 2 + 0.5, { size: 11, font: 'mono', color: C.ink, base: 'middle' });
          return;
        }
        const sw = w / p;
        if (st === 'csl') {
          // contagem inteira em cada rank, mas cada expert fatiado na largura: p fatias, cada uma com todos os experts
          for (let q = 0; q < p; q++) {
            V.box(g, x + q * sw + 0.5, y, sw - 1, h, { r: 3, fill: V.alpha(RC(C, q), 0.3), stroke: V.alpha(RC(C, q), 0.7), lw: 1 });
          }
          const lab = w > 240 ? nf(r.n) + ' × 1/' + p + ' of width' : '1/' + p + ' of each';
          const lw0 = Math.min(w - 6, tw(g, lab, 11, 'mono', 600) + 12);
          V.box(g, x + (w - lw0) / 2, y + 1, lw0, h - 2, { r: (h - 2) / 2, fill: V.alpha(C.bg, 0.78) });
          V.text(g, lab, x + w / 2, y + h / 2 + 0.5, { size: 11, font: 'mono', weight: 600, color: C.ink, align: 'center', base: 'middle', maxW: w - 8 });
          return;
        }
        if (st === 'ep') {
          for (let q = 0; q < p; q++) {
            V.box(g, x + q * sw + 1, y, sw - 2, h, { r: 4, fill: V.alpha(RC(C, q), 0.42), stroke: RC(C, q), lw: 1 });
            const nt = Math.max(2, Math.min(8, Math.floor(sw / 9)));
            for (let u = 1; u < nt; u++) V.line(g, x + q * sw + (u * sw) / nt, y + 3, x + q * sw + (u * sw) / nt, y + h - 3, { color: V.alpha(C.ink, 0.28), lw: 1 });
          }
          return;
        }
        const np = e.np || r.n;
        for (let q = 0; q < p; q++) V.box(g, x + q * sw + 0.5, y, sw - 1, h, { r: 3, fill: V.alpha(RC(C, q), st === 'bad' ? 0.22 : 0.5) });
        if (np > r.n) {
          const padW = Math.max(4, (w * (np - r.n)) / np), px = x + w - padW;
          V.box(g, px, y, padW, h, { r: 2, fill: V.alpha(C.bg, 0.6) });
          hatch(g, px, y, padW, h, V.alpha(C.d, 0.95), 5, t * 8);
          V.box(g, px, y, padW, h, { r: 2, stroke: C.d, lw: 1.2 });
        }
        if (st === 'bad') {
          g.strokeStyle = V.alpha(C.red, 0.55 + 0.4 * Math.sin(t * 4)); g.lineWidth = 2;
          for (let q = 1; q < p; q++) {
            const cx = x + (w * q) / p;
            g.beginPath(); g.moveTo(cx, y - 3);
            for (let s = 0; s <= 4; s++) g.lineTo(cx + (s % 2 ? 3 : -3), y - 3 + ((h + 6) * s) / 4);
            g.stroke();
          }
        }
        V.box(g, x, y, w, h, { r: 3, stroke: V.alpha(C.ink, 0.22), lw: 1 });
      }
      H = V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const Md = MODELS[S.m], p = S.p, rows = Md.rows;
        const narrow = w < 600;
        const evs = rows.map((r) => evalRow(r, p));
        V.text(g, 'TP = ' + p, 16, 25, { size: 17, font: 'display', weight: 700, color: C.ink });
        V.text(g, Md.name, 16 + tw(g, 'TP = ' + p, 17, 'display', 700) + 12, 25, { size: 12.5, color: C.muted, maxW: narrow ? w - 110 : 260 });
        if (!narrow) {
          const cnt = {}; evs.forEach((e) => { cnt[e.st] = (cnt[e.st] || 0) + 1; });
          let xx = w - 16;
          [['bad', 'blocks'], ['rep', 'replicated'], ['csl', 'width-sharded'], ['ep', 'EP'], ['pad', 'pad'], ['ok', 'divides']].forEach(([k, l]) => {
            const n = (cnt[k] || 0) + (k === 'rep' ? cnt.data || 0 : 0);
            if (!n) return;
            xx -= chip(g, n + ' ' + l, xx, 20, stCol(C, k), { align: 'right', size: 11 }) + 6;
          });
        }
        const top = 42;
        const rh = (h - top - 4) / rows.length;
        rows.forEach((r, i) => {
          const e = evs[i];
          const y = top + i * rh;
          const rev = V.ease.out(V.clamp((t - S.t0) * 4.5 - i * 0.08, 0, 1));
          const off = (1 - rev) * 14;
          g.globalAlpha = 0.2 + 0.8 * rev;
          let bx, bw, by, bh, cx, cy, chipW;
          if (!narrow) {
            const labelW = Math.min(250, w * 0.3);
            chipW = Math.min(250, w * 0.28);
            bx = 16 + labelW; bw = w - bx - chipW - 24; bh = Math.min(18, rh - 16); by = y + (rh - bh) / 2; cx = w - 16; cy = y + rh / 2;
            V.text(g, r.l, 16, y + rh / 2 - 3, { size: 12.5, weight: 500, color: C.ink, maxW: labelW - 10 });
            V.text(g, r.k + ' = ' + nf(r.n) + (r.kv ? '/' + r.kv : ''), 16, y + rh / 2 + 12, { size: 11, font: 'mono', color: C.muted, maxW: labelW - 10 });
          } else {
            chipW = Math.min(170, w * 0.46);
            V.text(g, r.l + ' · ' + nf(r.n) + (r.kv ? '/' + r.kv : ''), 16, y + 13, { size: 11.5, weight: 500, color: C.ink, maxW: w - 32 });
            bx = 16; bw = w - 32 - chipW - 8; bh = Math.min(14, rh - 26); by = y + 20; cx = w - 16; cy = by + bh / 2;
          }
          drawBar(g, C, r, e, p, bx + off, by, bw, bh, t);
          chip(g, ICON[e.st] + (narrow && e.txtN ? e.txtN : e.txt), cx, cy, stCol(C, e.st), { align: 'right', maxW: chipW, size: narrow ? 11 : 11.5 });
          g.globalAlpha = 1;
          if (i < rows.length - 1) V.line(g, 16, y + rh, w - 16, y + rh, { color: V.alpha(C.muted, 0.12), lw: 1 });
        });
      });
    })();

    /* ============================================ F2 — enchimento (assentos) */
    (function () {
      const fig = root.querySelector('#fig-pad');
      const P = {
        glmheads: { label: 'GLM · heads (64)', kind: 'cells', n: 64, rule: 'ceil' },
        dsheads: { label: 'DS · heads (8 groups)', kind: 'cells', n: 64, rule: 'groups', G: 8 },
        glmvocab: { label: 'GLM · vocab (154,880)', kind: 'bar', n: 154880, a: 64 },
        dsvocab: { label: 'DS · vocab (129,280)', kind: 'bar', n: 129280, a: 64 },
        dsmoe: { label: 'DS · expert (2,048)', kind: 'bar', n: 2048, a: 64, localAlign: 64 },
        glmshared: { label: 'GLM · shared (2,048)', kind: 'bar', n: 2048, a: 64, localAlign: 64 },
      };
      const S = { k: 'glmheads', p: 3 };
      const ctl = UI.controls(fig);
      const pos = new Map();
      let lastKey = '', lastK = '';
      const selP = UI.select(ctl, { label: 'Dimension', options: Object.keys(P).map((k) => ({ v: k, label: P[k].label })), value: 'glmheads', onChange: (v) => { S.k = v; } });
      selP.el.querySelector('select').style.maxWidth = '200px';
      UI.seg(ctl, { label: 'TP', options: [1, 2, 3, 4, 6], value: 3, onChange: (v) => { S.p = v; } });
      const rT = UI.readout(ctl, 'total', '—'), rL = UI.readout(ctl, 'per rank', '—'), rW = UI.readout(ctl, 'padding', '—');
      function calc(d, p) {
        if (d.rule === 'groups') { const Gp = Math.ceil(d.G / p) * p; const np = Gp * (d.n / d.G); return { np, loc: np / p, Gp }; }
        if (d.rule === 'ceil') { const np = Math.ceil(d.n / p) * p; return { np, loc: np / p }; }
        const L = lcm(d.a, p); const np = Math.ceil(d.n / L) * L; return { np, loc: np / p, L };
      }
      function layoutCells(w, h, d, p, c) {
        const top = 40, bottom = 58, avH = h - top - bottom;
        const gap = w < 500 ? 8 : 14;
        const bw = (w - 32 - gap * (p - 1)) / p;
        const out = [];
        if (d.rule === 'groups') {
          const hpg = d.n / d.G, gpr = c.Gp / p;
          const s = Math.min(24, (bw - 14) / (gpr * 1.18 + 0.2), (avH - 34) / (hpg * 1.1));
          for (let r = 0; r < p; r++) {
            const bx = 16 + r * (bw + gap), blockW = gpr * s * 1.18 - s * 0.18, ox = bx + (bw - blockW) / 2;
            for (let gi = 0; gi < gpr; gi++) {
              const G = r * gpr + gi;
              for (let hh = 0; hh < hpg; hh++) out.push({ id: G * hpg + hh, x: ox + gi * s * 1.18, y: top + 24 + hh * s * 1.1, s, pad: G >= d.G, grp: G, rank: r });
            }
          }
          return { cells: out, bw, gap, top, avH, s };
        }
        const bg = bestGrid(c.loc, bw - 14, avH - 34, 1.14, 28);
        const s = bg.s;
        for (let r = 0; r < p; r++) {
          const bx = 16 + r * (bw + gap), blockW = bg.cols * s * 1.14 - s * 0.14, ox = bx + (bw - blockW) / 2;
          for (let j = 0; j < c.loc; j++) {
            const id = r * c.loc + j;
            out.push({ id, x: ox + (j % bg.cols) * s * 1.14, y: top + 24 + Math.floor(j / bg.cols) * s * 1.14, s, pad: id >= d.n, rank: r });
          }
        }
        return { cells: out, bw, gap, top, avH, s };
      }
      function drawCells(g, w, h, t, dt, C, d, p, c) {
        const L = layoutCells(w, h, d, p, c);
        const k = 1 - Math.exp(-(dt || 0.016) * 7);
        for (let r = 0; r < p; r++) {
          const bx = 16 + r * (L.bw + L.gap);
          V.box(g, bx, L.top, L.bw, L.avH, { r: 10, fill: V.alpha(RC(C, r), 0.06), stroke: V.alpha(RC(C, r), 0.55), lw: 1.2 });
          const real = Math.max(0, Math.min(d.n - r * c.loc, c.loc)), fake = c.loc - real;
          const lab = (p > 4 && w < 700 ? 'r' : 'rank ') + r + ' · ' + c.loc + (fake ? ' (' + fake + ' fake)' : '');
          V.text(g, lab, bx + L.bw / 2, L.top + 15, { size: 11, font: 'mono', align: 'center', color: fake ? C.d : C.ink, maxW: L.bw - 8 });
        }
        const seen = new Set();
        const snap = pos.size === 0;
        L.cells.forEach((cl) => {
          seen.add(cl.id);
          let cur = pos.get(cl.id);
          if (!cur) { cur = snap ? { x: cl.x, y: cl.y, s: cl.s, a: 1 } : { x: cl.x, y: cl.y - 18, s: cl.s, a: 0 }; pos.set(cl.id, cur); }
          cur.x = approach(cur.x, cl.x, k); cur.y = approach(cur.y, cl.y, k); cur.s = approach(cur.s, cl.s, k); cur.a = approach(cur.a, 1, k);
          g.globalAlpha = cur.a;
          const col = d.rule === 'groups' ? GCOL(C)[cl.grp % 8] : RC(C, cl.rank);
          if (cl.pad) {
            V.box(g, cur.x, cur.y, cur.s, cur.s, { r: 3, fill: V.alpha(C.d, 0.1 + 0.08 * Math.sin(t * 3 + cl.id)) });
            dashedBox(g, cur.x, cur.y, cur.s, cur.s, C.d);
            if (cur.s >= 15) V.text(g, '0', cur.x + cur.s / 2, cur.y + cur.s / 2 + 0.5, { size: 11, font: 'mono', align: 'center', base: 'middle', color: C.d });
          } else {
            V.box(g, cur.x, cur.y, cur.s, cur.s, { r: 3, fill: V.alpha(col, 0.75) });
            if (cur.s >= 22) V.text(g, cl.id, cur.x + cur.s / 2, cur.y + cur.s / 2 + 0.5, { size: 11, font: 'mono', align: 'center', base: 'middle', color: C.dark ? '#00210F' : '#fff' });
          }
          g.globalAlpha = 1;
        });
        for (const id of Array.from(pos.keys())) if (!seen.has(id)) pos.delete(id);
        if (d.rule === 'groups') {
          const padG = L.cells.filter((cl) => cl.pad);
          if (padG.length) { const last = padG[padG.length - 1]; V.text(g, 'sink −∞', last.x + last.s / 2, last.y + last.s + 14, { size: 11, font: 'mono', align: 'center', color: C.d }); }
        }
        let f;
        if (d.rule === 'groups') f = c.Gp === d.G ? '8 groups ÷ ' + p + ' = ' + d.G / p + ' per rank · r = 8 × 512 = 4096 intact' : 'groups: ⌈8 / ' + p + '⌉ × ' + p + ' = ' + c.Gp + ' → heads = ' + c.Gp + ' × 8 = ' + c.np + ' (' + c.loc + ' per rank)';
        else f = c.np === d.n ? d.n + ' / ' + p + ' = ' + c.loc + ' — divides cleanly' : 'n′ = ⌈' + d.n + ' / ' + p + '⌉ × ' + p + ' = ' + c.loc + ' × ' + p + ' = ' + c.np + ' · waste ' + pct(((c.np - d.n) / d.n) * 100);
        V.text(g, f, 16, h - 22, { size: 12, font: 'mono', color: C.ink, maxW: w - 32 });
      }
      function drawBarMode(g, w, h, t, C, d, p, c) {
        const bx = 16, bwid = w - 32, by = 36, bh = 30;
        V.text(g, 'tensor with ' + nf(c.np) + ' rows' + (c.np > d.n ? ' (' + nf(d.n) + ' real + ' + nf(c.np - d.n) + ' pad)' : ''), bx, 22, { size: 12, color: C.muted, maxW: bwid });
        for (let r = 0; r < p; r++) {
          const sx = bx + (r * bwid) / p;
          V.box(g, sx + 1, by, bwid / p - 2, bh, { r: 6, fill: V.alpha(RC(C, r), 0.5), stroke: RC(C, r), lw: 1 });
          const lab = (bwid / p > 110 ? 'rank ' + r + ': ' : 'r' + r + ': ') + nf(c.loc);
          V.text(g, lab, sx + bwid / p / 2, by + bh / 2 + 0.5, { size: 11, font: 'mono', align: 'center', base: 'middle', color: C.ink, maxW: bwid / p - 8 });
        }
        const padW = c.np > d.n ? Math.max(4, (bwid * (c.np - d.n)) / c.np) : 0;
        if (padW) { hatch(g, bx + bwid - padW, by, padW, bh, V.alpha(C.d, 0.95), 5, t * 10); V.box(g, bx + bwid - padW, by, padW, bh, { r: 2, stroke: C.d, lw: 1.4 }); }
        // lupa sobre o fim do tensor
        const W0 = c.np > d.n ? Math.max(4 * (c.np - d.n), 256) : 256;
        const W = Math.min(W0, c.np);
        const ws = c.np - W;
        const u = W / 16 <= 48 ? 16 : W / 32 <= 48 ? 32 : 64;
        const ncell = Math.ceil(W / u);
        const srcX0 = bx + (bwid * ws) / c.np, srcX1 = bx + bwid;
        const ly = by + bh + 66, lh = 46;
        const breathe = 0.5 + 0.5 * Math.sin(t * 2);
        g.fillStyle = V.alpha(C.ink, 0.05 + 0.03 * breathe);
        g.beginPath(); g.moveTo(Math.max(bx, srcX0 - 3), by + bh + 2); g.lineTo(srcX1, by + bh + 2); g.lineTo(bx + bwid, ly); g.lineTo(bx, ly); g.closePath(); g.fill();
        V.line(g, Math.max(bx, srcX0 - 3), by + bh + 2, bx, ly, { color: V.alpha(C.ink, 0.35), dash: [3, 3] });
        V.line(g, srcX1, by + bh + 2, bx + bwid, ly, { color: V.alpha(C.ink, 0.35), dash: [3, 3] });
        const cw = bwid / ncell;
        for (let j = 0; j < ncell; j++) {
          const a0 = ws + j * u;
          const x = bx + j * cw;
          if (a0 >= d.n) {
            V.box(g, x + 1, ly, cw - 2, lh, { r: 3, fill: V.alpha(C.d, 0.12) });
            hatch(g, x + 1, ly, cw - 2, lh, V.alpha(C.d, 0.9), 5, t * 10);
            dashedBox(g, x + 1, ly, cw - 2, lh, C.d);
          } else {
            const rr = Math.min(p - 1, Math.floor(a0 / c.loc));
            V.box(g, x + 1, ly, cw - 2, lh, { r: 3, fill: V.alpha(RC(C, rr), 0.6) });
          }
        }
        V.text(g, 'magnifier: last ' + nf(W) + ' rows (cell = ' + u + ')', bx, ly + lh + 16, { size: 11, color: C.muted, maxW: bwid * 0.6 });
        V.text(g, nf(c.np), bx + bwid, ly + lh + 16, { size: 11, font: 'mono', align: 'right', color: C.muted });
        if (c.np > d.n) {
          const nx = bx + ((d.n - ws) / u) * cw;
          V.line(g, nx, ly - 8, nx, ly + lh + 4, { color: C.d, lw: 2 });
          V.text(g, 'real end: ' + nf(d.n), nx - 4, ly - 12, { size: 11, font: 'mono', align: 'right', color: C.d });
        }
        if (c.np > d.n && c.np - d.n <= c.loc) {
          const lastReal = d.n - (p - 1) * c.loc;
          V.text(g, 'rank ' + (p - 1) + ' holds ' + nf(lastReal) + ' real rows + ' + nf(c.np - d.n) + ' pad rows (zeros, never used)', 16, ly + lh + 44, { size: 12, color: C.d, weight: 600, maxW: w - 32 });
        } else if (c.np === d.n) V.text(g, 'divides cleanly: no padding at this TP', 16, ly + lh + 44, { size: 12, color: C.e, weight: 600, maxW: w - 32 });
        const narrowB = w < 560;
        const rg = (r) => 'r' + r + ' [' + nf(r * c.loc) + ', ' + nf((r + 1) * c.loc) + ')';
        const rl = p <= 3 ? Array.from({ length: p }, (_, r) => rg(r)) : [rg(0), rg(1), '…', rg(p - 1)];
        if (!narrowB) V.text(g, 'id ranges per rank: ' + rl.join(' · '), 16, ly + lh + 70, { size: 11.5, font: 'mono', color: C.muted, maxW: w - 32 });
        else { V.text(g, 'id ranges per rank:', 16, ly + lh + 68, { size: 11.5, color: C.muted }); rl.forEach((x, j) => V.text(g, x, 16, ly + lh + 86 + j * 16, { size: 11.5, font: 'mono', color: C.muted, maxW: w - 32 })); }
        const L = c.L;
        const l1a = 'L = lcm(' + d.a + ', ' + p + ') = ' + L, l1b = 'n′ = ⌈' + nf(d.n) + ' / ' + L + '⌉ × ' + L + ' = ' + nf(c.np / L) + ' × ' + L + ' = ' + nf(c.np);
        if (narrowB) { V.text(g, l1a, 16, h - 58, { size: 12, font: 'mono', color: C.ink, maxW: w - 32 }); V.text(g, l1b, 16, h - 40, { size: 12, font: 'mono', color: C.ink, maxW: w - 32 }); }
        else V.text(g, l1a + ' · ' + l1b, 16, h - 40, { size: 12, font: 'mono', color: C.ink, maxW: w - 32 });
        let l2 = 'per rank: ' + nf(c.loc) + ' · waste ' + (c.np > d.n ? pct(((c.np - d.n) / d.n) * 100) : '0%');
        let col2 = C.ink;
        if (d.localAlign) {
          const okA = c.loc % d.localAlign === 0;
          l2 += ' · local ' + (okA ? '= ' + nf(c.loc / d.localAlign) + ' × 64 ✓' : '= ' + nf(c.loc / d.localAlign, 1) + ' × 64 ✗ (gcd(64,' + p + ') = ' + gcd(64, p) + ')');
          if (!okA) col2 = C.red;
        }
        V.text(g, l2, 16, h - 20, { size: 12, font: 'mono', color: col2, maxW: w - 32 });
      }
      V.fig(fig, (g, w, h, t, dt) => {
        const C = V.C(); const d = P[S.k]; const p = S.p; const c = calc(d, p);
        if (S.k !== lastK) { pos.clear(); lastK = S.k; }
        const key = S.k + '|' + p;
        if (key !== lastKey) {
          lastKey = key;
          rT.set(nf(c.np)); rL.set(nf(c.loc));
          rW.set(c.np === d.n ? '0' : '+' + nf(c.np - d.n) + ' (' + pct(((c.np - d.n) / d.n) * 100) + ')');
        }
        if (d.kind === 'cells') drawCells(g, w, h, t, dt, C, d, p, c); else drawBarMode(g, w, h, t, C, d, p, c);
      });
    })();

    /* ============================================ F3 — DeepSeek R1 vs R2 (stepper) */
    (function () {
      const fig = root.querySelector('#fig-r1r2');
      const steps = [
        { t: 'TP=2: the clean cut', d: '8 groups ÷ 2 = 4 groups per rank. Each group keeps 8 heads, so r = 8 × 512 = 4096 matches <code>wo_a = [8192, 4096]</code>, and the <code>wo_b</code> all-reduce sums the partials.' },
        { t: 'TP=3 on stock code: silence', d: 'Without a patch, vLLM refuses to start (<code>64 % 3 != 0</code>) or computes <code>8 // 3 == 2</code> local groups with no assert. Some groups are left without an owner, and the model generates fluent, wrong text.' },
        { t: 'R1: shard heads within the group', d: 'Replicate the 8 groups on every rank, pad each group to 9 heads (the 9th made of zeros, 72 in total) and give 3 of each group to each rank; on rank 2 that means 2 real + 1 pad. r becomes 3 × 512 = 1536 ≠ 4096 on every rank, and the kernel rejects it (<code>einsum.hpp:164</code>).' },
        { t: 'R2: a 9th group of zeros', d: 'Pad the <b>group count</b> (8 → 9) and keep 8 heads per group: 72 heads, 24 heads and 3 groups per rank. r stays 4096; <code>wo_a</code> now has 9 × 1024 = 9216 outputs (3072 per rank).' },
        { t: 'What goes inside the pad', d: 'Group 9 has null <code>wo_a</code> and <code>wo_b</code>, and the 8 fake heads get <code>attn_sink = −∞</code>. In the SM120 kernel, 24 local heads become 32. At boot, every rank must log <code>DSv4 TP pad: heads 64 -&gt; 72 for TP=3</code>.' },
      ];
      const st = UI.stepper(fig, steps, { autoplay: 7000 });
      const BOX = [
        ['✓', 'r = 8 × 512 = 4096 → wo_a[g]: 4096 → 1024', '4 groups per rank · wo_b row-parallel + all-reduce'],
        ['?', '8 // 3 = 2 groups per rank · ownerless groups', 'no error → fluent, wrong text'],
        ['✕', 'r = 3 × 512 = 1536 ≠ 4096 (checkpoint)', 'einsum.hpp:164: m == m_ and n == n_ and k == k_'],
        ['✓', 'r = 8 × 512 = 4096 · 72 heads, 24 per rank', 'wo_a: 9 × 1024 = 9216 outputs → 3072 per rank'],
        ['✓', 'group 9: wo_a = wo_b = 0 · attn_sink = −∞', 'SM120 kernel: 24 → 32 local heads (internal pad)'],
      ];
      function pieces(i) {
        const out = [];
        if (i === 0) for (let gq = 0; gq < 8; gq++) out.push({ g: gq, h0: 0, h1: 8, lane: Math.floor(gq / 4), slot: gq % 4 });
        else if (i === 1) for (let gq = 0; gq < 8; gq++) out.push(gq < 6 ? { g: gq, h0: 0, h1: 8, lane: Math.floor(gq / 2), slot: gq % 2 } : { g: gq, h0: 0, h1: 8, orphan: true });
        else if (i === 2) for (let gq = 0; gq < 8; gq++) for (let r = 0; r < 3; r++) out.push({ g: gq, h0: r * 3, h1: r * 3 + 3, lane: r, slot: gq });
        else for (let gq = 0; gq < 9; gq++) out.push({ g: gq, h0: 0, h1: 8, lane: Math.floor(gq / 3), slot: gq % 3, pad: gq === 8 });
        return out;
      }
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const i = st.i;
        const e = V.ease.inOut(V.clamp(st.elapsed() / 1.1, 0, 1));
        const p = i === 0 ? 2 : 3;
        const narrow = w < 560;
        // faixa do checkpoint
        const cs = narrow ? 7 : 9, colW = cs * 1.7, sx0 = 16, sy0 = 34;
        V.text(g, 'checkpoint: 8 groups × 8 heads', sx0, 20, { size: 11.5, color: C.muted });
        const stripX = (gq) => sx0 + gq * colW;
        for (let gq = 0; gq < 9; gq++) {
          for (let hh = 0; hh < 9; hh++) {
            const x = stripX(gq), y = sy0 + hh * (cs + 1.5);
            if (hh === 8) { if (i === 2 && gq < 8) dashedBox(g, x, y, cs, cs, V.alpha(C.d, 0.7), 2); continue; }
            if (gq === 8) { if (i >= 3) dashedBox(g, x, y, cs, cs, V.alpha(C.d, 0.7), 2); continue; }
            V.box(g, x, y, cs, cs, { r: 2, fill: V.alpha(GCOL(C)[gq], 0.38) });
          }
        }
        if (i === 2) V.text(g, '← 9th head (pad)', stripX(8) + 2, sy0 + 8 * (cs + 1.5) + cs - 1, { size: 11, color: C.d });
        const stripH = 9 * (cs + 1.5);
        // título do passo à direita da faixa
        const hx = sx0 + 9 * colW + 22;
        V.text(g, 'TP = ' + p, hx, sy0 + 16, { size: narrow ? 17 : 22, font: 'display', weight: 700, color: C.ink });
        const tag = ['clean cut', 'stock (no patch)', 'R1', 'R2', 'R2 · details'][i];
        V.text(g, tag, hx, sy0 + 38, { size: 12.5, color: i === 1 || i === 2 ? C.red : C.e, weight: 600, maxW: w - hx - 16 });
        if (!narrow) V.text(g, 'color = group · panel = rank', hx, sy0 + 58, { size: 11, color: C.muted });
        // faixas dos ranks
        const ly = sy0 + stripH + 26, lgap = narrow ? 8 : 14;
        const boxH = 58, laneH = h - ly - boxH - 18;
        const lw = (w - 32 - lgap * (p - 1)) / p;
        // R1 no celular: 8 grupos em 2 faixas de 4 (senão as células ficam com ~8 px)
        const bands = i === 2 && narrow ? 2 : 1;
        const cols = i === 2 && narrow ? 4 : [4, 2, 8, 3, 3][i];
        const hpl = i === 2 ? 3 : 8;
        const lc = Math.min(i === 2 && !narrow ? 24 : 18, (lw - 18) / (cols * 1.3), (laneH - (i === 4 ? 46 : 30)) / ((hpl * bands + (bands - 1) * 0.8) * 1.1));
        for (let r = 0; r < p; r++) {
          const lx = 16 + r * (lw + lgap);
          g.globalAlpha = 0.35 + 0.65 * e;
          V.box(g, lx, ly, lw, laneH, { r: 10, fill: V.alpha(RC(C, r), 0.07), stroke: V.alpha(RC(C, r), 0.6), lw: 1.3 });
          V.text(g, 'rank ' + r, lx + 10, ly + 16, { size: 11.5, font: 'mono', weight: 600, color: RC(C, r) });
          g.globalAlpha = 1;
        }
        const pcs = pieces(i);
        const colsX = (lane, slot) => { const lx = 16 + lane * (lw + lgap); const blockW = cols * lc * 1.3 - lc * 0.3; return lx + (lw - blockW) / 2 + (slot % cols) * lc * 1.3; };
        const bandY = (slot) => Math.floor(slot / cols) * (hpl + 0.8) * lc * 1.1;
        pcs.forEach((pc) => {
          for (let hh = pc.h0; hh < pc.h1; hh++) {
            const sx = stripX(pc.g), sy = sy0 + hh * (cs + 1.5);
            let tx, ty, ts;
            if (pc.orphan) { tx = sx; ty = sy; ts = cs; }
            else { tx = colsX(pc.lane, pc.slot); ty = ly + 26 + bandY(pc.slot) + (hh - pc.h0) * lc * 1.1; ts = lc; }
            const x = V.lerp(sx, tx, e), y = V.lerp(sy, ty, e), s = V.lerp(cs, ts, e);
            if (pc.pad || hh >= 8) {
              g.globalAlpha = e;
              V.box(g, x, y, s, s, { r: 3, fill: V.alpha(C.d, 0.12) });
              dashedBox(g, x, y, s, s, C.d);
              if (i === 4 && s >= 12) V.text(g, '0', x + s / 2, y + s / 2 + 0.5, { size: 11, font: 'mono', align: 'center', base: 'middle', color: C.d });
              g.globalAlpha = 1;
            } else if (pc.orphan) {
              const blink = 0.5 + 0.5 * Math.sin(t * 5);
              V.box(g, x, y, s, s, { r: 2, fill: V.alpha(C.red, 0.55 + 0.4 * blink), stroke: C.red, lw: 1 });
            } else {
              V.box(g, x, y, s, s, { r: 3, fill: V.alpha(GCOL(C)[pc.g], i === 2 ? 0.55 : 0.85) });
            }
          }
        });
        if (i === 1) {
          const ox = stripX(6), oy = sy0 + stripH + 4;
          V.text(g, '↑ no owner', ox, oy + 12, { size: 11, color: C.red, weight: 600 });
        }
        if (i === 2) {
          for (let r = 0; r < 3; r++) {
            const lx = 16 + r * (lw + lgap);
            const blockBottom = ly + 26 + (bands * hpl + (bands - 1) * 0.8) * lc * 1.1;
            if (narrow) {
              V.text(g, '3 heads/group', lx + 10, blockBottom + 18, { size: 11, font: 'mono', color: C.red, maxW: lw - 16 });
              V.text(g, r < 2 ? '3 real' : '2 real + 1 pad', lx + 10, blockBottom + 34, { size: 11, font: 'mono', color: r < 2 ? C.muted : C.d, maxW: lw - 16 });
              V.text(g, 'r = 1536 ✕', lx + 10, blockBottom + 50, { size: 11, font: 'mono', color: C.red, maxW: lw - 16 });
            } else {
              V.text(g, '3 heads/group', lx + lw - 8, ly + 16, { size: 11, font: 'mono', align: 'right', color: C.red, maxW: lw * 0.55 });
              V.text(g, r < 2 ? '3 real per group' : '2 real + 1 pad per group', lx + lw / 2, blockBottom + 22, { size: 11.5, font: 'mono', align: 'center', color: r < 2 ? C.muted : C.d, maxW: lw - 16 });
              V.text(g, 'r = 3 × 512 = 1536 ✕', lx + lw / 2, blockBottom + 40, { size: 11.5, font: 'mono', align: 'center', color: C.red, maxW: lw - 16 });
            }
          }
        }
        if (i === 4) {
          const lx = 16 + 2 * (lw + lgap);
          const px = colsX(2, 2) + lc / 2, py = ly + 26 + 8 * lc * 1.1 + 12;
          V.text(g, 'sink −∞', px, py, { size: 11, font: 'mono', align: 'center', color: C.d });
          if (narrow) V.text(g, '24 → 32 in kernel', lx + 10, ly + laneH - 10, { size: 11, font: 'mono', color: C.d, maxW: lw - 16 });
          else V.text(g, '24 → 32 in kernel', lx + lw - 8, ly + 16, { size: 11, font: 'mono', align: 'right', color: C.d, maxW: lw * 0.6 });
        }
        // caixa da forma do BMM
        const bY = h - boxH - 8;
        const B = BOX[i];
        const bc = B[0] === '✓' ? C.e : C.red;
        V.box(g, 16, bY, w - 32, boxH, { r: 12, fill: V.alpha(bc, 0.08), stroke: V.alpha(bc, 0.6), lw: 1.3 });
        V.box(g, 26, bY + boxH / 2 - 15, 30, 30, { r: 15, fill: V.alpha(bc, 0.25), stroke: bc, lw: 1.5, label: B[0], size: 15, color: bc, weight: 700 });
        V.text(g, B[1], 66, bY + 23, { size: narrow ? 11 : 12.5, font: 'mono', weight: 600, color: C.ink, maxW: w - 96 });
        V.text(g, B[2], 66, bY + 42, { size: 11, font: 'mono', color: C.muted, maxW: w - 96 });
      });
    })();

    /* ============================================ F4 — termo fantasma na softmax */
    (function () {
      const fig = root.querySelector('#fig-sink');
      const S = { v: 0, ref: true };
      const scores = [2.0, 1.0, 0.5, -0.5], sink = 0.5;
      const ctl = UI.controls(fig);
      UI.slider(ctl, { label: 'phantom logit v', min: -12, max: 3, step: 0.1, value: 0, fmt: (v) => (v <= -11.95 ? '−∞' : nf(v, 1)), onInput: (v) => { S.v = v; } });
      UI.toggle(ctl, { label: 'show reference', value: true, onChange: (v) => { S.ref = v; } });
      const rM = UI.readout(ctl, 'mass on keys', '—'), rA = UI.readout(ctl, 'attenuation', '—');
      const cur = [0, 0, 0, 0, 0, 0];
      let lastV = null;
      V.fig(fig, (g, w, h, t, dt) => {
        const C = V.C();
        const ev = S.v <= -11.95 ? 0 : Math.exp(S.v);
        const es = scores.map(Math.exp), sum = es.reduce((a, b) => a + b, 0), eS = Math.exp(sink);
        const den0 = eS + sum, den = den0 + ev;
        const pr0 = es.map((x) => x / den0), pr = es.map((x) => x / den);
        const vals = pr.concat([eS / den, ev / den]);
        const refs = pr0.concat([eS / den0, 0]);
        if (lastV !== S.v) {
          lastV = S.v;
          rM.set(nf((sum / den) * 100, 1) + '%');
          rA.set(ev === 0 ? '0%' : '−' + nf((1 - den0 / den) * 100, 1) + '%');
          tone(rA, ev > 0); tone(rM, ev > 0);
        }
        const k = 1 - Math.exp(-(dt || 0.016) * 8);
        for (let j = 0; j < 6; j++) cur[j] = approach(cur[j], vals[j], k);
        const narrow = w < 520;
        const x0 = 16, x1 = w - 16, yb = h - 78, yt = 34;
        const maxV = 0.6;
        const n = 6, gap = narrow ? 10 : 22, bw = (x1 - x0 - gap * (n - 1)) / n;
        V.text(g, 'p_j = exp(s_j) / ( exp(σ) + Σ_k exp(s_k) + exp(v) )', x0, 20, { size: 12, font: 'mono', color: C.muted, maxW: w - 32 });
        for (let q = 1; q <= 3; q++) { const y = yb - ((q * 0.2) / maxV) * (yb - yt); V.line(g, x0, y, x1, y, { color: V.alpha(C.muted, 0.15), lw: 1 }); V.text(g, nf(q * 0.2, 1), x1, y - 3, { size: 11, font: 'mono', align: 'right', color: C.muted }); }
        V.line(g, x0, yb, x1, yb, { color: V.alpha(C.muted, 0.5), lw: 1 });
        const labs = narrow ? ['k1', 'k2', 'k3', 'k4', 'sink', 'ghost'] : ['key 1', 'key 2', 'key 3', 'key 4', 'real sink', 'phantom'];
        const cols = [C.a, C.a, C.a, C.a, C.b, C.c];
        for (let j = 0; j < n; j++) {
          const x = x0 + j * (bw + gap);
          const hv = (cur[j] / maxV) * (yb - yt);
          V.box(g, x, yb - hv, bw, Math.max(1, hv), { r: 5, fill: V.alpha(cols[j], 0.75) });
          if (S.ref && j < 5) { const hr = (refs[j] / maxV) * (yb - yt); g.setLineDash([4, 3]); V.box(g, x - 2, yb - hr, bw + 4, hr, { r: 5, stroke: V.alpha(C.ink, 0.55), lw: 1.2 }); g.setLineDash([]); }
          V.text(g, nf(cur[j], 3), x + bw / 2, yb - hv - 6, { size: 11, font: 'mono', align: 'center', color: C.ink });
          V.text(g, labs[j], x + bw / 2, yb + 16, { size: 11, align: 'center', color: C.muted, maxW: bw + gap });
        }
        // fantasminha
        const ga = V.clamp(ev / 1.2, 0, 1);
        if (ga > 0.01) {
          const gx = x0 + 5 * (bw + gap) + bw / 2, gy = yb - (cur[5] / maxV) * (yb - yt) - 40 + Math.sin(t * 2.4) * 4, R = Math.min(15, bw / 3);
          g.globalAlpha = 0.3 + 0.7 * ga;
          g.fillStyle = V.alpha(C.c, 0.85);
          g.beginPath(); g.arc(gx, gy, R, Math.PI, 0); g.lineTo(gx + R, gy + R);
          for (let s = 0; s < 4; s++) { const xx = gx + R - ((s + 0.5) * (2 * R)) / 4; g.lineTo(xx, gy + R - 5 * (s % 2 === 0 ? 1 : 0)); }
          g.lineTo(gx - R, gy + R); g.closePath(); g.fill();
          V.dot(g, gx - R * 0.35, gy - 2, 2.4, C.bg); V.dot(g, gx + R * 0.35, gy - 2, 2.4, C.bg);
          g.globalAlpha = 1;
        }
        const evS = ev === 0 ? '0' : nf(ev, 2);
        V.text(g, 'denominator = ' + nf(eS, 2) + ' + ' + nf(sum, 2) + ' + ' + evS + ' = ' + nf(den, 2) + '   (reference ' + nf(den0, 2) + ')', x0, h - 38, { size: 12, font: 'mono', color: C.ink, maxW: w - 32 });
        V.text(g, ev === 0 ? 'v = −∞: e^v = 0, the padding is neutral' : 'each real p_j shrinks by ' + nf(den0, 2) + ' / ' + nf(den, 2) + ' = ' + nf(den0 / den, 3), x0, h - 18, { size: 12, color: ev === 0 ? C.e : C.c, weight: 600, maxW: w - 32 });
      });
    })();

    /* ============================================ F5 — cabeças do GLM e a tabela do kernel */
    (function () {
      const fig = root.querySelector('#fig-glmheads');
      const CF = { tp2: { p: 2, total: 64 }, tp3: { p: 3, total: 66 }, x96: { p: 3, total: 96 }, tp4: { p: 4, total: 64 } };
      const PITCH = { tp2: 12576, tp3: 8726 };
      const S = { k: 'tp3', kpad: true };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Config', options: [{ v: 'tp2', label: 'TP=2 · 64' }, { v: 'tp3', label: 'TP=3 · 66' }, { v: 'x96', label: 'TP=3 · 96 ✕' }, { v: 'tp4', label: 'TP=4 · 64' }], value: 'tp3', onChange: (v) => { S.k = v; } });
      UI.toggle(ctl, { label: 'SM120 kernel pad (Q → 32)', value: true, onChange: (v) => { S.kpad = v; } });
      const rL = UI.readout(ctl, 'local heads', '—'), rK = UI.readout(ctl, 'SM120 kernel', '—'), rM = UI.readout(ctl, 'KDA pitch', '—');
      const TABLE = [8, 16, 32, 64, 128];
      let lastKey = '';
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const cf = CF[S.k], p = cf.p, loc = cf.total / p;
        const inT = TABLE.includes(loc);
        const kpadTo = TABLE.find((x) => x >= loc);
        const kernOk = inT || S.kpad;
        const key = S.k + S.kpad;
        if (key !== lastKey) {
          lastKey = key;
          rL.set(String(loc));
          rK.set(inT ? loc + ' ✓' : S.kpad ? loc + ' → ' + kpadTo : loc + ' ✕');
          rM.set(PITCH[S.k] ? nf(PITCH[S.k]) + (PITCH[S.k] % 8 ? ' ✕' : ' ✓') : 'n/a');
          tone(rK, !kernOk); tone(rM, !!(PITCH[S.k] && PITCH[S.k] % 8)); tone(rL, cf.total === 96);
        }
        const narrow = w < 560;
        const top = 12, gap = narrow ? 8 : 14, blockH = narrow ? 128 : 138;
        const bw = (w - 32 - gap * (p - 1)) / p;
        const bg = bestGrid(loc, bw - 14, blockH - 36, 1.16, 18);
        for (let r = 0; r < p; r++) {
          const bx = 16 + r * (bw + gap);
          const real = Math.max(0, Math.min(64 - r * loc, loc));
          const allFake = real === 0;
          const pulse = 0.5 + 0.5 * Math.sin(t * 4);
          V.box(g, bx, top, bw, blockH, { r: 10, fill: V.alpha(allFake ? C.red : RC(C, r), allFake ? 0.06 + 0.05 * pulse : 0.06), stroke: allFake ? V.alpha(C.red, 0.5 + 0.4 * pulse) : V.alpha(RC(C, r), 0.55), lw: allFake ? 2 : 1.2 });
          const lab = (narrow ? 'r' : 'rank ') + r + ': ' + loc + (loc - real ? ' (' + (loc - real) + ' fake)' : '');
          V.text(g, lab, bx + bw / 2, top + 15, { size: 11, font: 'mono', align: 'center', color: allFake ? C.red : C.ink, maxW: bw - 8 });
          const blockW = bg.cols * bg.s * 1.16 - bg.s * 0.16, ox = bx + (bw - blockW) / 2;
          for (let j = 0; j < loc; j++) {
            const id = r * loc + j, x = ox + (j % bg.cols) * bg.s * 1.16, y = top + 26 + Math.floor(j / bg.cols) * bg.s * 1.16;
            if (id >= 64) { V.box(g, x, y, bg.s, bg.s, { r: 2, fill: V.alpha(C.muted, 0.1) }); dashedBox(g, x, y, bg.s, bg.s, allFake ? C.red : C.d, 2); }
            else V.box(g, x, y, bg.s, bg.s, { r: 2, fill: V.alpha(RC(C, r), 0.78) });
          }
          if (allFake) V.text(g, 'fake only → logits collapse', bx + bw / 2, top + blockH - 10, { size: 11, align: 'center', color: C.red, weight: 600, maxW: bw - 8 });
        }
        // tabela de templates
        const ky = top + blockH + 34;
        V.text(g, 'SM120 sparse-MLA decode templates (local heads):', 16, ky - 12, { size: 11.5, color: C.muted, maxW: w - 32 });
        let xx = 16;
        const tgt = inT ? loc : S.kpad ? kpadTo : null;
        const chipX = {};
        TABLE.forEach((v) => {
          const on = v === tgt;
          const col = on ? C.e : C.muted;
          chipX[v] = xx;
          xx += chip(g, String(v), xx, ky + 10, col, { fa: on ? 0.3 : 0.08, lw: on ? 2 : 1, size: 12 }) + 8;
        });
        const lx = xx + 10;
        const lcol = kernOk ? C.e : C.red;
        if (lx + 120 < w) chip(g, 'local = ' + loc + (inT ? ' ✓' : S.kpad ? ' → ' + kpadTo : ' ✕ no template'), lx, ky + 10, lcol, { fa: 0.14, size: 12, maxW: w - lx - 16 });
        else chip(g, 'local = ' + loc + (inT ? ' ✓' : S.kpad ? ' → ' + kpadTo : ' ✕'), w - 16, ky + 10 + 26, lcol, { align: 'right', size: 12 });
        // chamada do kernel animada
        const cy = ky + (lx + 120 < w ? 52 : 74);
        const nPad = !inT && S.kpad ? kpadTo - loc : 0;
        const totalQ = loc + nPad;
        const colW = Math.min(6, (w * 0.26) / totalQ);
        const ag = narrow ? 22 : 40;
        const qx = 16, qh = 30;
        for (let j = 0; j < totalQ; j++) {
          const x = qx + j * colW;
          if (j < loc) V.box(g, x, cy, colW - 1, qh, { r: 1, fill: V.alpha(C.a, 0.8) });
          else { V.box(g, x, cy, colW - 1, qh, { r: 1, fill: V.alpha(C.muted, 0.15) }); }
        }
        const qW = totalQ * colW;
        V.text(g, 'Q [' + loc + (nPad ? ' + ' + nPad + ' zeros' : '') + ']', qx, cy + qh + 15, { size: 11, font: 'mono', color: C.muted });
        const kx = qx + qW + ag, kw = Math.min(110, w * 0.2);
        const kcol = kernOk ? C.e : C.red;
        V.arrow(g, qx + qW + 6, cy + qh / 2, kx - 6, cy + qh / 2, { color: V.alpha(C.ink, 0.5), lw: 1.5 });
        V.box(g, kx, cy - 2, kw, qh + 4, { r: 8, fill: V.alpha(kcol, 0.14), stroke: kcol, lw: 1.4, label: kernOk ? 'kernel ' + (tgt || loc) : 'no kernel', size: 11.5 });
        const ph = (t % 2.4) / 2.4;
        if (kernOk) V.glow(g, V.lerp(qx + qW, kx + kw, ph), cy + qh / 2, 7, C.a);
        if (kernOk) {
          const ox = kx + kw + ag;
          if (ox + totalQ * colW + 10 < w) {
            V.arrow(g, kx + kw + 6, cy + qh / 2, ox - 6, cy + qh / 2, { color: V.alpha(C.ink, 0.5), lw: 1.5 });
            for (let j = 0; j < totalQ; j++) {
              const x = ox + j * colW;
              if (j < loc) V.box(g, x, cy, colW - 1, qh, { r: 1, fill: V.alpha(C.b, 0.8) });
              else { const drop = V.clamp((ph - 0.3) / 0.6, 0, 1); g.globalAlpha = 1 - drop; V.box(g, x, cy + drop * 18, colW - 1, qh, { r: 1, fill: V.alpha(C.muted, 0.25) }); g.globalAlpha = 1; }
            }
            V.text(g, 'O [' + loc + ']' + (nPad ? ' · extras dropped' : ''), ox, cy + qh + 15, { size: 11, font: 'mono', color: C.muted, maxW: w - ox - 16 });
          }
        } else V.text(g, 'prefill also rejects num_heads=' + loc, kx + kw + 12, cy + qh / 2 + 4, { size: 11, color: C.red, maxW: w - kx - kw - 28 });
        // Marlin
        const my = Math.min(h - 14, cy + qh + 44);
        const pv = PITCH[S.k];
        const mcol = pv ? (pv % 8 ? C.red : C.e) : C.muted;
        const m1 = pv ? 'Marlin · local KDA in_proj: ' + nf(pv) + ' rows' : 'Marlin · local pitch not documented';
        const m2 = pv ? nf(pv) + ' mod 8 = ' + (pv % 8) + (pv % 8 ? ' ✕ → f_b/g_b_proj stay BF16' : ' ✓') : 'for this configuration';
        if (narrow) { V.text(g, m1, 16, my - 8, { size: 11.5, font: 'mono', color: mcol, maxW: w - 32 }); V.text(g, m2, 16, my + 10, { size: 11.5, font: 'mono', color: mcol, maxW: w - 32 }); }
        else V.text(g, m1 + ' · ' + m2, 16, my, { size: 11.5, font: 'mono', color: mcol, maxW: w - 32 });
      });
    })();

    /* ============================================ F6 — TP (fatiar) × EP (experts inteiros) */
    (function () {
      const fig = root.querySelector('#fig-ep');
      const S = { mode: 'tp', p: 3, tok: null, tokT: -10 };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Mode', options: [{ v: 'tp', label: 'shard (TP)' }, { v: 'ep', label: 'whole (EP)' }], value: 'tp', onChange: (v) => { S.mode = v; } });
      UI.seg(ctl, { label: 'Ranks', options: [2, 3, 4], value: 3, onChange: (v) => { S.p = v; } });
      const rE = UI.readout(ctl, 'experts/rank', '—'), rW = UI.readout(ctl, 'local width', '—'), rS = UI.readout(ctl, 'status', '—');
      const rng = V.rng(40);
      let lastKey = '';
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const p = S.p, ep = S.mode === 'ep';
        const loc = 2048 / p, tileOk = Number.isInteger(loc) && loc % 16 === 0;
        const ok = ep ? 288 % p === 0 : tileOk;
        const key = S.mode + p;
        if (key !== lastKey) {
          lastKey = key;
          rE.set(ep ? String(288 / p) : '288 (slices)');
          rW.set(ep ? '2,048' : tileOk ? nf(loc) : nf(loc, 2));
          rS.set(ok ? 'OK' : 'cuts tiles ✕'); tone(rS, !ok); tone(rW, !ep && !tileOk);
        }
        if (t - S.tokT > 1.6) {
          S.tokT = t;
          const set = new Set(); while (set.size < 8) set.add(Math.floor(rng() * 288));
          S.tok = Array.from(set);
        }
        const narrow = w < 560;
        const cols = narrow ? 18 : 24, rows = 288 / cols;
        const gx0 = narrow ? 16 : 70, gy0 = narrow ? 52 : 30;
        const gridH = h - gy0 - 118;
        const cs = Math.min((w - gx0 - 16) / (cols * 1.15), gridH / (rows * 1.15));
        const gw = cols * cs * 1.15;
        const ox = gx0 + (w - gx0 - 16 - gw) / 2;
        if (!narrow) V.text(g, ep ? 'whole experts: each one has a single owner' : 'each expert sharded into ' + p + ' pieces', ox, 20, { size: 12, color: C.muted, maxW: w - ox - 170 });
        else V.text(g, ep ? 'EP: one owner per expert' : 'TP: ' + p + ' slices', 16, 20, { size: 11.5, color: C.muted });
        if (!ok) chip(g, '✕ 2048 / ' + p + ' = ' + nf(loc, 2), w - 16, 16, C.red, { align: 'right', size: 11, h: 19 });
        else chip(g, ep ? '✓ ' + 288 / p + ' experts/rank' : '✓ ' + nf(loc) + ' columns/rank', w - 16, 16, C.e, { align: 'right', size: 11, h: 19 });
        const sel = new Set(S.tok || []);
        const tokAge = V.clamp((t - S.tokT) / 1.6, 0, 1);
        const centers = {};
        for (let e = 0; e < 288; e++) {
          const cx = ox + (e % cols) * cs * 1.15, cy = gy0 + Math.floor(e / cols) * cs * 1.15;
          centers[e] = [cx + cs / 2, cy + cs / 2];
          const on = sel.has(e);
          if (ep) {
            const owner = Math.floor(e / (288 / p));
            V.box(g, cx, cy, cs, cs, { r: 2, fill: V.alpha(RC(C, owner), on ? 0.95 : 0.45) });
          } else {
            for (let q = 0; q < p; q++) V.box(g, cx + (q * cs) / p, cy, cs / p - 0.6, cs, { r: 1, fill: V.alpha(RC(C, q), on ? 0.95 : 0.4) });
          }
          if (on) { g.strokeStyle = C.ink; g.lineWidth = 1.6; g.strokeRect(cx - 1, cy - 1, cs + 2, cs + 2); }
        }
        // token roteado
        if (S.tok) {
          // no celular o token fica acima da grade, no centro; no desktop, à esquerda
          const tx = narrow ? w / 2 : 26, ty = narrow ? 38 : gy0 + gridH / 2;
          V.glow(g, tx, ty, narrow ? 12 : 16, C.c); V.dot(g, tx, ty, narrow ? 5 : 6, C.c);
          if (narrow) V.text(g, 'token', tx + 10, ty + 4, { size: 11, color: C.muted });
          else V.text(g, 'token', tx, ty + 22, { size: 11, align: 'center', color: C.muted });
          S.tok.forEach((e, j) => {
            const [ex, ey] = centers[e];
            const col = ep ? RC(C, Math.floor(e / (288 / p))) : C.c;
            g.globalAlpha = 0.25 + 0.5 * (1 - tokAge);
            V.line(g, tx + (narrow ? 0 : 6), ty + (narrow ? 5 : 0), ex, ey, { color: col, lw: 1.2 });
            g.globalAlpha = 1;
            const pt = V.along(tx, ty, ex, ey, V.clamp(tokAge * 1.6 - j * 0.04, 0, 1));
            V.dot(g, pt[0], pt[1], 2.6, col);
          });
        }
        // contagem por rank (EP)
        const cntY = gy0 + rows * cs * 1.15 + 18;
        if (S.tok) {
          let xx = ox;
          if (ep) {
            const cnt = Array.from({ length: p }, () => 0); S.tok.forEach((e) => { cnt[Math.floor(e / (288 / p))]++; });
            V.text(g, 'this token:', xx, cntY, { size: 11, color: C.muted }); xx += tw(g, 'this token:', 11) + 8;
            cnt.forEach((c, r) => { xx += chip(g, 'r' + r + ': ' + c, xx, cntY - 4, RC(C, r), { size: 11, h: 18 }) + 6; });
          } else V.text(g, 'every rank reads 1/' + p + ' of each of the 8 chosen experts', xx, cntY, { size: 11, color: C.muted, maxW: w - xx - 16 });
        }
        // lupa da treliça
        const ty0 = h - 56, tx0 = 16, twd = w - 32, th = 22;
        V.text(g, ep ? 'expert 0 whole on rank 0: 2048 columns = 128 tiles of 16' : 'EXL3 trellis of one expert: 2048 columns = 128 tiles of 16', tx0, ty0 - 10, { size: 11.5, color: C.muted, maxW: twd });
        const tile = twd / 128;
        for (let j = 0; j < 128; j++) {
          const owner = ep ? 0 : Math.min(p - 1, Math.floor((j * 16) / loc));
          V.box(g, tx0 + j * tile + 0.3, ty0, tile - 0.6, th, { r: 1, fill: V.alpha(RC(C, owner), 0.55) });
        }
        if (!ep) {
          for (let q = 1; q < p; q++) {
            const cut = (q * 2048) / p, cx = tx0 + (cut / 2048) * twd;
            const good = Number.isInteger(cut / 16);
            V.line(g, cx, ty0 - 4, cx, ty0 + th + 4, { color: good ? C.e : C.red, lw: 2.4 });
            if (!good && q === 1) V.text(g, nf(cut, 2) + ' → cuts a tile', cx + 5, ty0 + th + 15, { size: 11, font: 'mono', color: C.red, maxW: twd - (cx - tx0) - 5 });
            if (good && q === 1) V.text(g, nf(cut) + ' = ' + cut / 16 + ' tiles ✓', cx + 5, ty0 + th + 15, { size: 11, font: 'mono', color: C.e });
          }
        } else V.text(g, 'no cut: the loader must not shard again (patch_exl3_ep_shard)', tx0, ty0 + th + 15, { size: 11, color: C.e, maxW: twd });
      });
    })();

    /* ============================================ F7 — GQA do drafter */
    (function () {
      const fig = root.querySelector('#fig-gqa');
      fig.querySelector('.la-fig-stage').style.height = fig.clientWidth < 560 ? '310px' : '260px';
      const CFG = { s: { q: 32, kv: 8, l: 'stock' }, c: { q: 33, kv: 9, l: 'independent ceiling' }, r: { q: 36, kv: 9, l: 'ratio kept' } };
      const S = { k: 's', first: true };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Config', options: [{ v: 's', label: '32/8 stock' }, { v: 'c', label: '33/9 ceiling' }, { v: 'r', label: '36/9 ratio' }], value: 's', onChange: (v) => { S.k = v; } });
      const pos = new Map();
      V.fig(fig, (g, w, h, t, dt) => {
        const C = V.C();
        const cf = CFG[S.k], q = cf.q, kv = cf.kv;
        const ratio = q / kv, intR = Number.isInteger(ratio);
        const narrow = w < 560;
        const areaW = narrow ? w - 32 : w * 0.56;
        const colW = areaW / 9;
        const s = Math.min(narrow ? 17 : 22, colW * 0.62);
        const k = 1 - Math.exp(-(dt || 0.016) * 6);
        V.text(g, 'KV heads (top) and their query heads', 16, 18, { size: 11.5, color: C.muted, maxW: areaW });
        for (let j = 0; j < 9; j++) {
          const x = 16 + j * colW + (colW - s) / 2, y = 30;
          if (j < kv) V.box(g, x, y, s, s, { r: 4, fill: V.alpha(C.b, 0.85), label: narrow ? '' : 'K' + j, size: 11, color: C.dark ? '#1a0b33' : '#fff' });
          else dashedBox(g, x, y, s, s, V.alpha(C.muted, 0.4), 4);
        }
        const perCol = intR ? ratio : Math.ceil(ratio);
        for (let i = 0; i < q; i++) {
          let tx, ty;
          if (intR) { const j = Math.floor(i / ratio), r = i % ratio; tx = 16 + j * colW + (colW - s) / 2; ty = 30 + s + 10 + r * (s + 4); }
          else { const j = i % 9, r = Math.floor(i / 9); tx = 16 + j * colW + (colW - s) / 2; ty = 30 + s + 10 + r * (s + 4); }
          let cur = pos.get(i);
          if (!cur) { cur = pos.size === 0 || S.first ? { x: tx, y: ty, a: 1 } : { x: tx, y: ty + 20, a: 0 }; pos.set(i, cur); }
          cur.x = approach(cur.x, tx, k); cur.y = approach(cur.y, ty, k); cur.a = approach(cur.a, 1, k);
          g.globalAlpha = cur.a;
          const padded = i >= 32;
          if (padded) { V.box(g, cur.x, cur.y, s, s, { r: 3, fill: V.alpha(C.d, 0.15) }); dashedBox(g, cur.x, cur.y, s, s, C.d); }
          else V.box(g, cur.x, cur.y, s, s, { r: 3, fill: V.alpha(intR ? C.a : C.red, intR ? 0.75 : 0.45 + 0.2 * Math.sin(t * 3 + i)) });
          g.globalAlpha = 1;
        }
        for (const id of Array.from(pos.keys())) if (id >= q) pos.delete(id);
        S.first = false;
        if (!intR) V.text(g, 'no whole-number Q → KV mapping: 33 / 9 = 3.67', 16, 30 + s + 10 + perCol * (s + 4) + 16, { size: 11.5, color: C.red, weight: 600, maxW: areaW });
        // checagens
        const checks = [
          ['assert total_num_heads % 3 (world TP)', q % 3 === 0, q + ' % 3 = ' + (q % 3)],
          ['FlashInfer qo % kv on the drafter rank', q % kv === 0, q + ' % ' + kv + ' = ' + (q % kv)],
          ['if sharded by 3: local qo/kv', q % 3 === 0 && kv % 3 === 0 && (q / 3) % (kv / 3) === 0, q % 3 === 0 && kv % 3 === 0 ? q / 3 + '/' + kv / 3 : nf(q / 3, 2) + '/' + nf(kv / 3, 2)],
          ['Q:KV ratio = 4', ratio === 4, nf(ratio, 2)],
        ];
        const cx0 = narrow ? 16 : w * 0.62, cw = narrow ? w - 32 : w - cx0 - 16;
        const cy0 = narrow ? h - checks.length * 30 - 10 : 44;
        if (!narrow) V.text(g, cf.q + ' Q / ' + cf.kv + ' KV · ' + cf.l, cx0, 22, { size: 14, font: 'display', weight: 700, color: C.ink, maxW: cw });
        checks.forEach(([lab, good, val], j) => {
          const y = cy0 + j * (narrow ? 30 : 46);
          const col = good ? C.e : C.red;
          V.box(g, cx0, y, cw, narrow ? 26 : 38, { r: 8, fill: V.alpha(col, 0.08), stroke: V.alpha(col, 0.5), lw: 1 });
          V.text(g, (good ? '✓ ' : '✕ ') + lab, cx0 + 10, y + (narrow ? 17 : 16), { size: 11.5, color: C.ink, maxW: cw - (narrow ? 90 : 20) });
          V.text(g, val, narrow ? cx0 + cw - 10 : cx0 + 10, y + (narrow ? 17 : 32), { size: 11, font: 'mono', color: col, align: narrow ? 'right' : 'left' });
        });
      });
    })();

    /* ============================================ F8 — anel de três nós */
    (function () {
      const fig = root.querySelector('#fig-ring');
      const S = { cab: 'dir', pair: 's24', lan: true };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Cabling', options: [{ v: 'dir', label: 'directed ring' }, { v: 'mir', label: 'mirrored ring' }], value: 'dir', onChange: (v) => { S.cab = v; } });
      UI.seg(ctl, { label: 'Pairing', options: [{ v: 'idx', label: 'by index' }, { v: 's16', label: 'subnet /16' }, { v: 's24', label: 'subnet /24' }], value: 's24', onChange: (v) => { S.pair = v; } });
      UI.toggle(ctl, { label: 'control plane on the LAN (enP7s7)', value: true, onChange: (v) => { S.lan = v; } });
      const rO = UI.readout(ctl, 'RoCE pairs', '—');
      const SUB = { hw1: { net: '10.0.22', c: 'a' }, hw2: { net: '10.0.23', c: 'b' }, w12: { net: '10.0.33', c: 'd' } };
      // portas por modo: [nó A, porta A, nó B, porta B]
      const CABLES = {
        dir: { hw1: ['h', 1, 'w1', 0], hw2: ['h', 0, 'w2', 1], w12: ['w2', 0, 'w1', 1] },
        mir: { hw1: ['h', 0, 'w1', 0], hw2: ['h', 1, 'w2', 0], w12: ['w1', 1, 'w2', 1] },
      };
      let lastKey = '';
      function status() {
        const res = {};
        ['hw1', 'hw2', 'w12'].forEach((k) => {
          let ok = true, why = '';
          if (S.pair === 'idx') { ok = false; why = 'dev0 → dev0 on another /24'; }
          else if (S.pair === 's16') { ok = false; why = 'everything looks like 10.0/16'; }
          else if (S.cab === 'mir' && k === 'w12') { ok = false; why = 'pair that never connects'; }
          res[k] = { ok, why };
        });
        return res;
      }
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const st = status();
        const nOk = Object.values(st).filter((x) => x.ok).length;
        const key = S.cab + S.pair + S.lan;
        if (key !== lastKey) { lastKey = key; rO.set(nOk + '/3'); tone(rO, nOk < 3); }
        const narrow = w < 560;
        const nw = narrow ? 104 : 150, nh = narrow ? 50 : 58;
        const N = {
          h: { x: w / 2, y: 58, l: 'spark1 · head', ip: '10.0.0.1' },
          w1: { x: narrow ? 16 + nw / 2 : w * 0.2, y: h - 112, l: 'spark2 · worker1', ip: '10.0.0.2' },
          w2: { x: narrow ? w - 16 - nw / 2 : w * 0.8, y: h - 112, l: 'spark3 · worker2', ip: '10.0.0.3' },
        };
        const cab = CABLES[S.cab];
        // anchors: cada nó tem um lado voltado a cada vizinho
        const side = (a, b) => { const A = N[a], B = N[b]; const dx = B.x - A.x, dy = B.y - A.y; const L = Math.hypot(dx, dy); const ux = dx / L, uy = dy / L; const kx = (nw / 2) / Math.max(1e-6, Math.abs(ux)), ky = (nh / 2) / Math.max(1e-6, Math.abs(uy)); const kk = Math.min(kx, ky); return [A.x + ux * kk, A.y + uy * kk]; };
        // LAN
        const sw = { x: w / 2, y: h * 0.6 };
        ['h', 'w1', 'w2'].forEach((n) => {
          const A = N[n];
          const col = S.lan ? V.alpha(C.muted, 0.6) : V.alpha(C.muted, 0.2);
          V.line(g, A.x, A.y + (n === 'h' ? nh / 2 : -nh / 2), sw.x, sw.y, { color: col, lw: 1, dash: [3, 4], dashOffset: S.lan ? -t * 12 : 0 });
          if (S.lan) { const u = ((t * 0.35 + (n === 'h' ? 0 : n === 'w1' ? 0.33 : 0.66)) % 1); const pt = V.along(sw.x, sw.y, A.x, A.y + (n === 'h' ? nh / 2 : -nh / 2), u); V.dot(g, pt[0], pt[1], 2.2, C.muted); }
        });
        V.box(g, sw.x - (narrow ? 58 : 80), sw.y - 15, narrow ? 116 : 160, 30, { r: 8, fill: V.alpha(C.panel, 0.95), stroke: S.lan ? C.muted : V.alpha(C.red, 0.7), lw: 1.2 });
        V.text(g, narrow ? 'LAN 192.168.1/24' : 'LAN 192.168.1.0/24 · enP7s7', sw.x, sw.y + 1, { size: 11, font: 'mono', align: 'center', base: 'middle', color: S.lan ? C.ink : C.muted, maxW: narrow ? 108 : 150 });
        if (!S.lan) V.text(g, 'Gloo: one interface only, no CX port reaches both neighbors', sw.x, sw.y + 30, { size: 11, align: 'center', color: C.red, weight: 600, maxW: w - 40 });
        // cabos (linhas, pacotes e rótulos de sub-rede)
        const ports = [];
        Object.keys(cab).forEach((k) => {
          const [a, pa, b, pb] = cab[k];
          let A = side(a, b), B = side(b, a);
          if (narrow && k === 'w12') { const sgn = (n) => (N[n].x < w / 2 ? 1 : -1); A = [N[a].x + sgn(a) * (nw / 2 - 16), N[a].y + nh / 2]; B = [N[b].x + sgn(b) * (nw / 2 - 16), N[b].y + nh / 2]; }
          const col = C[SUB[k].c];
          const s = st[k];
          ports.push([A, pa, col], [B, pb, col]);
          V.line(g, A[0], A[1], B[0], B[1], { color: s.ok ? col : V.alpha(C.red, 0.7), lw: 3.2, dash: s.ok ? null : [7, 5] });
          const mx = (A[0] + B[0]) / 2, my = (A[1] + B[1]) / 2;
          if (s.ok) {
            const dir = S.cab === 'dir';
            for (let q = 0; q < 3; q++) {
              const u = (t * 0.45 + q / 3) % 1;
              const pt = dir ? V.along(A[0], A[1], B[0], B[1], pa === 0 ? u : 1 - u) : V.along(A[0], A[1], B[0], B[1], u);
              V.glow(g, pt[0], pt[1], 7, col, 0.9);
            }
          } else {
            const pulse = 0.6 + 0.4 * Math.sin(t * 5);
            V.box(g, mx - 11, my - 11, 22, 22, { r: 11, fill: V.alpha(C.red, 0.2 * pulse + 0.1), stroke: C.red, lw: 1.5, label: '✕', size: 13, color: C.red, weight: 700 });
          }
          const lbx = mx + (k === 'hw1' ? -10 : k === 'hw2' ? 10 : 0), lby = k === 'w12' ? my + (narrow ? 26 : 22) : my - 14;
          const al = k === 'hw1' ? 'right' : k === 'hw2' ? 'left' : 'center';
          // largura disponível até a borda do canvas, conforme o alinhamento
          const room = al === 'right' ? lbx - 8 : al === 'left' ? w - lbx - 8 : Math.min(lbx, w - lbx) * 2 - 16;
          const mW = Math.max(40, Math.min(narrow ? 150 : 240, room));
          if (narrow && !s.ok) {
            V.text(g, SUB[k].net + '.x', lbx, lby - 7, { size: 11, font: 'mono', align: al, color: C.red, maxW: mW });
            V.text(g, s.why, lbx, lby + 7, { size: 11, font: 'mono', align: al, color: C.red, maxW: mW });
          } else V.text(g, SUB[k].net + '.x' + (s.ok ? '' : ' · ' + s.why), lbx, lby, { size: 11, font: 'mono', align: al, color: s.ok ? col : C.red, maxW: mW });
        });
        // nós
        Object.keys(N).forEach((n) => {
          const A = N[n];
          V.box(g, A.x - nw / 2, A.y - nh / 2, nw, nh, { r: 12, fill: V.alpha(C.panel, 0.98), stroke: n === 'h' ? C.a : V.alpha(C.ink, 0.35), lw: 1.5 });
          V.text(g, A.l, A.x, A.y - 5, { size: narrow ? 11 : 12.5, weight: 600, align: 'center', color: C.ink, maxW: nw - 12 });
          V.text(g, 'rank ' + (n === 'h' ? 0 : n === 'w1' ? 1 : 2) + ' · ' + A.ip, A.x, A.y + 12, { size: 11, font: 'mono', align: 'center', color: C.muted, maxW: nw - 10 });
        });
        // rótulos de porta por cima dos nós
        ports.forEach(([P, port, col]) => { V.box(g, P[0] - 12, P[1] - 9, 24, 18, { r: 5, fill: C.panel, stroke: col, lw: 1.4, label: 'p' + port, size: 11, font: 'mono', weight: 600 }); });
        const msg = (nOk === 3 ? (S.lan ? '3/3 RoCE pairs connect · control over the LAN' : 'data OK, but the bootstrap can\'t close the mesh') : nOk + '/3 pairs connect · QP → RTR times out: ibv_modify_qp … 110') + (S.pair !== 's24' ? ' (illustrative state)' : '');
        const mls = wrapLines(g, msg, w - 32, 12, 'sans', 600).slice(0, 2);
        mls.forEach((l, j) => V.text(g, l, w / 2, h - 16 - (mls.length - 1 - j) * 16, { size: 12, align: 'center', weight: 600, color: nOk === 3 && S.lan ? C.e : C.red, maxW: w - 32 }));
      });
    })();

    /* ============================================ F9 — stall distribuído */
    (function () {
      const fig = root.querySelector('#fig-stall');
      const S = { slice: false, T: 1024, phase: 'run', x: [0, 0, 0, 0], timer: 0, cycle: 0, deadAt: 0 };
      const ctl = UI.controls(fig);
      const reset = () => { S.phase = 'run'; S.x = [0, 0, 0, 0]; S.timer = 0; };
      UI.toggle(ctl, { label: 'slice the final call (≤ 64 rows)', value: false, onChange: (v) => { S.slice = v; reset(); } });
      UI.slider(ctl, { label: 'rows T', min: 64, max: 2048, step: 64, value: 1024, fmt: (v) => String(v), onInput: (v) => { S.T = v; reset(); } });
      UI.button(ctl, '↺ restart', reset);
      const rF = UI.readout(ctl, 'slices', '—'), rS = UI.readout(ctl, 'state', '—');
      // linha do tempo de um forward: blocos [nome, largura relativa]
      const BL = [['attn', 3], ['AR', 1], ['MoE', 3], ['AR', 1], ['attn*', 3], ['AR', 1], ['MoE', 3], ['AR', 1], ['sample', 1.5]];
      const tot = BL.reduce((a, b) => a + b[1], 0);
      const starts = []; { let a = 0; BL.forEach((b) => { starts.push(a / tot); a += b[1]; }); }
      const iStar = 4, xStar0 = starts[iStar], xStar1 = starts[iStar + 1];
      let lastKey = '';
      V.fig(fig, (g, w, h, t, dt) => {
        const C = V.C();
        dt = Math.min(dt || 0.016, 0.05);
        const hangs = !S.slice && S.T > 64;
        const nSl = Math.ceil(S.T / 64);
        const stateTxt = S.phase === 'run' ? 'running' : S.phase === 'hang' ? 'stuck' : 'engine dead';
        const key = S.slice + '|' + S.T + '|' + S.phase;
        if (key !== lastKey) { lastKey = key; rF.set(S.slice ? String(nSl) : '1'); rS.set(stateTxt); tone(rS, S.phase !== 'run'); }
        // dinâmica
        const speed = 0.16;
        if (S.phase === 'run') {
          for (let r = 0; r < 4; r++) {
            let v = speed * (1 + 0.05 * Math.sin(r * 1.7 + t));
            let x = S.x[r] + v * dt;
            // barreiras de all-reduce: ninguém passa do fim de um AR antes de todos chegarem ao início dele
            BL.forEach((b, j) => { if (b[0] !== 'AR') return; const s0 = starts[j]; if (Math.min(...S.x) < s0 && x > s0 + 0.001 && S.x[r] <= s0 + 0.001) x = Math.min(x, s0); });
            if (hangs && r === 2 && S.x[r] < xStar1 && x >= xStar0 + (xStar1 - xStar0) * 0.55) { x = xStar0 + (xStar1 - xStar0) * 0.55; S.phase = 'hang'; S.timer = 0; }
            S.x[r] = x;
          }
          if (Math.min(...S.x) >= 1) { S.x = [0, 0, 0, 0]; S.cycle++; }
        } else if (S.phase === 'hang') {
          S.timer += dt;
          for (let r = 0; r < 4; r++) if (r !== 2) S.x[r] = Math.min(S.x[r] + speed * dt, xStar1);
          if (S.timer > 6) { S.phase = 'dead'; S.deadAt = t; }
        } else if (S.phase === 'dead' && t - S.deadAt > 3.5) reset();
        const narrow = w < 560;
        const x0 = narrow ? 44 : 64, x1 = w - 16, lw = x1 - x0;
        const y0 = 36, laneH = narrow ? 34 : 40, gap = 12;
        V.text(g, 'one forward at TP=4 · * = final sparse-MLA call', x0, 20, { size: 11.5, color: C.muted, maxW: lw });
        for (let r = 0; r < 4; r++) {
          const y = y0 + r * (laneH + gap);
          const stuckN = narrow && r === 2 && S.phase !== 'run';
          V.text(g, 'r' + r, 16, stuckN ? y + 14 : y + laneH / 2 + 4, { size: 12, font: 'mono', weight: 600, color: RC(C, r) });
          if (stuckN) V.text(g, 'stuck', 16, y + 29, { size: 10, weight: 600, color: C.red, maxW: x0 - 18 });
          BL.forEach((b, j) => {
            const bx = x0 + starts[j] * lw, bw = (b[1] / tot) * lw;
            const isAR = b[0] === 'AR', isStar = j === iStar;
            const fillC = isAR ? C.b : isStar ? C.d : C.muted;
            V.box(g, bx + 1, y, bw - 2, laneH, { r: 6, fill: V.alpha(fillC, isAR ? 0.18 : 0.12), stroke: V.alpha(fillC, 0.45), lw: 1 });
            if (isStar && S.slice && S.T > 64) {
              const nt = Math.min(nSl, 32);
              for (let q = 1; q < nt; q++) V.line(g, bx + 1 + (q * (bw - 2)) / nt, y + 4, bx + 1 + (q * (bw - 2)) / nt, y + laneH - 4, { color: V.alpha(C.d, 0.55), lw: 1 });
            }
            const stuckLab = isStar && r === 2 && S.phase !== 'run';
            const lab = b[0] === 'sample' && bw <= 60 ? 'smpl' : b[0];
            if (bw > 30) V.text(g, lab, bx + bw / 2, stuckLab ? y + 13 : y + laneH / 2 + 4, { size: 11, font: 'mono', align: 'center', color: C.ink, alpha: 0.7, maxW: bw - 6 });
            else if (bw >= 14) V.text(g, lab, bx + bw / 2, y + laneH / 2 + 3.5, { size: 10, font: 'mono', align: 'center', color: C.ink, alpha: 0.75, maxW: bw - 2 });
          });
          // progresso
          const px = x0 + Math.min(1, S.x[r]) * lw;
          V.box(g, x0 + 1, y + laneH - 5, Math.max(0, px - x0 - 2), 4, { r: 2, fill: RC(C, r) });
          const stuck = S.phase !== 'run' && r === 2;
          const waiting = S.phase !== 'run' && r !== 2 && S.x[r] >= xStar1 - 0.001;
          if (stuck) {
            // o rótulo "atn*" sobe para o topo do bloco; o spinner fica na metade de baixo, e "preso" logo abaixo da faixa
            const a = t * 6, sr = narrow ? 6 : 7, sy = y + laneH - sr - 7;
            g.strokeStyle = C.red; g.lineWidth = 2.5; g.beginPath(); g.arc(px, sy, sr, a, a + 4.2); g.stroke();
            if (!narrow) V.text(g, 'stuck', px + sr + 5, sy + 4, { size: 11, color: C.red, weight: 600 });
          } else if (waiting) {
            V.glow(g, px, y + laneH / 2 - 3, 12 + 3 * Math.sin(t * 6), C.b);
          } else { V.line(g, px, y + 2, px, y + laneH - 2, { color: RC(C, r), lw: 2 }); V.dot(g, px, y + laneH - 3, 3.5, RC(C, r)); }
        }
        const by = y0 + 4 * (laneH + gap) + 6;
        if (S.phase === 'hang') {
          const secs = Math.round((S.timer / 6) * 300);
          const ls = wrapLines(g, 'r0, r1 and r3 spinning at the all-reduce barrier for ' + secs + ' s · GPU 96% · 21 W (compressed time)', w - 32, 12, 'sans', 600);
          ls.slice(0, 3).forEach((l, j) => V.text(g, l, 16, by + 14 + j * 17, { size: 12, color: C.red, weight: 600, maxW: w - 32 }));
        } else if (S.phase === 'dead') {
          V.box(g, 16, by, w - 32, 30, { r: 8, fill: V.alpha(C.red, 0.15), stroke: C.red, lw: 1.4 });
          V.text(g, 'RPC call to sample_tokens timed out', w / 2, by + 16, { size: 12.5, font: 'mono', align: 'center', base: 'middle', color: C.red, weight: 600, maxW: w - 48 });
        } else {
          const txt = S.slice ? 'final call in ' + nSl + ' slices of ≤ 64 rows · the forward moves on (mitigation, the race remains)' : S.T <= 64 ? 'T ≤ 64: the call already fits in one slice' : 'whole final call: T = ' + S.T + ' rows';
          const ls = wrapLines(g, txt, w - 32, 12, 'sans', 600);
          ls.slice(0, 3).forEach((l, j) => V.text(g, l, 16, by + 14 + j * 17, { size: 12, color: S.slice ? C.e : C.ink, weight: 600, maxW: w - 32 }));
        }
        // legenda: o que são os blocos lavanda
        const lgY = h - 36;
        V.box(g, 16, lgY - 9, 14, 12, { r: 3, fill: V.alpha(C.b, 0.18), stroke: V.alpha(C.b, 0.6), lw: 1 });
        V.text(g, narrow ? 'AR = all-reduce barrier: waits for all 4 ranks' : 'AR = all-reduce barrier: nobody leaves it before everyone arrives', 36, lgY + 1, { size: 11, color: C.muted, maxW: w - 52 });
        V.text(g, 'completed forwards: ' + S.cycle, 16, h - 14, { size: 11, font: 'mono', color: C.muted });
      });
    })();

    /* ============================================ F10 — vale a pena crescer? */
    (function () {
      const fig = root.querySelector('#fig-grow');
      const D = {
        dsdec: { t: 'DeepSeek · decode (tok/s)', cats: ['1 stream', '2', '4', '8', '12', '16'], series: [{ n: 'TP=2 (6 slots)', c: 'b', v: [38.6, 63.9, 86.0, null, null, null] }, { n: 'TP=3 (16 slots)', c: 'a', v: [41.9, 72.0, 89.7, 142.8, 164.3, 198.0] }], d: 1, foot: '2026-09-02 · prose · 512 tokens · aggregate from 2 streams up (docs/TP3.md)' },
        dsttft: { t: 'DeepSeek · TTFT in seconds (lower is better)', cats: ['8K', '16K', '32K', '64K', '128K', '256K'], series: [{ n: 'TP=2', c: 'b', v: [4.4, 8.7, 18.0, 36.0, 74.9, 164.8] }, { n: 'TP=3', c: 'a', v: [5.0, 9.4, 18.6, 38.2, 91.5, 202.2] }], d: 1, foot: '2026-09-02 · same image · replicated attention and indexer don\'t shrink with the 3rd rank' },
        glmlab: { t: 'GLM · decode by prompt type (tok/s)', cats: ['structured', 'code', 'prose'], series: [{ n: 'TP=2', c: 'b', v: [73.4, 45.0, 32.9] }, { n: 'TP=3', c: 'a', v: [87.8, 54.9, 39.6] }], d: 1, foot: '2026-09-14 · temp 0 · thinking off · 400 tokens · median of 3 · TTFT 0.33 → 0.25 s' },
        glmconc: { t: 'GLM TP=3 · concurrency (tok/s)', cats: ['×1', '×2', '×3', '×4'], series: [{ n: 'per stream', c: 'b', v: [40.1, 28.7, 25.5, 22.8] }, { n: 'aggregate', c: 'a', v: [40.1, 56.6, 75.5, 88.4] }], d: 1, foot: '2026-09-17 · sparkDash · prose · thinking off · 512 tokens' },
        kv: { t: 'KV capacity (millions of tokens)', cats: ['DeepSeek', 'GLM'], series: [{ n: 'TP=2', c: 'b', v: [2.33, 1.23] }, { n: 'TP=3', c: 'a', v: [5.02, 3.23] }], d: 2, hatch: [false, true], note: [null, 'different caps (11 vs 40 GiB), not a TP gain'], foot: 'DS: 17.04 GiB (TP2, util 0.835) vs ~35 GiB/rank (TP3, 0.83) · GLM: 11 GiB cap at 850k vs 40 GiB cap at 1M (policy, not freed memory)' },
        mem: { t: 'Weights per rank (GiB; fewer weights = more room for KV)', cats: ['DeepSeek', 'GLM'], series: [{ n: 'TP=2', c: 'b', v: [79.17, 79.65] }, { n: 'TP=3', c: 'a', v: [60.69, 55.75] }], d: 2, foot: 'DS: fable5-1 (79.17 → 60.69) · GLM: README (79.65 GiB/rank on TP2; ~55.75 GiB/node on TP3; dense FP8 all vs dense,kda)' },
        tp4: { t: 'GLM TP=4 over TP=2 (×)', cats: ['decode', 'cold prefill 282k'], series: [{ n: 'TP=2', c: 'b', v: [1, 1] }, { n: 'TP=4', c: 'd', v: [1.45, 1.29] }], d: 2, x: true, foot: '2026-09-02 · first bring-up, launched per rank (before .env.tp4) · 4 Sparks on a CRS812 switch · prefill 901 → 1162 tok/s (network-bound)' },
      };
      const S = { k: 'dsdec', t0: 0 };
      const ctl = UI.controls(fig);
      let H = null;
      const selG = UI.select(ctl, { label: 'Set', options: [{ v: 'dsdec', label: 'DeepSeek · decode' }, { v: 'dsttft', label: 'DeepSeek · TTFT' }, { v: 'glmlab', label: 'GLM · TP3 vs TP2' }, { v: 'glmconc', label: 'GLM · TP3 concurrency' }, { v: 'kv', label: 'KV capacity' }, { v: 'mem', label: 'Weights per rank' }, { v: 'tp4', label: 'GLM · TP4 vs TP2' }], value: 'dsdec', onChange: (v) => { S.k = v; S.t0 = H ? H.t : 0; } });
      selG.el.querySelector('select').style.maxWidth = '220px';
      H = V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const d = D[S.k];
        const e = V.ease.out(V.clamp((t - S.t0) / 0.7, 0, 1));
        const narrow = w < 560;
        V.text(g, d.t, 16, 22, { size: narrow ? 12.5 : 14, font: 'display', weight: 700, color: C.ink, maxW: w - 32 });
        let lx = 16;
        d.series.forEach((s) => { V.box(g, lx, 34, 12, 12, { r: 3, fill: C[s.c] }); V.text(g, s.n, lx + 17, 44, { size: 11.5, color: C.ink }); lx += tw(g, s.n, 11.5) + 34; });
        const x0 = narrow ? 40 : 56, x1 = w - 16, yt = 66, yb = h - 58;
        let mx = 0; d.series.forEach((s) => s.v.forEach((v) => { if (v != null && v > mx) mx = v; }));
        const step = mx > 150 ? 50 : mx > 60 ? 20 : mx > 12 ? 5 : mx > 4 ? 1 : 0.5;
        const top = Math.ceil((mx * 1.12) / step) * step;
        for (let v = 0; v <= top + 1e-9; v += step) {
          const y = yb - (v / top) * (yb - yt);
          V.line(g, x0, y, x1, y, { color: V.alpha(C.muted, v === 0 ? 0.5 : 0.14), lw: 1 });
          V.text(g, nf(v, step < 1 ? 1 : 0), x0 - 6, y + 4, { size: 11, font: 'mono', align: 'right', color: C.muted });
        }
        const nc = d.cats.length, ns = d.series.length;
        const cw = (x1 - x0) / nc, bw = Math.min(46, (cw * (narrow ? 0.86 : 0.74)) / ns);
        const showV = bw >= 24;
        const pt = H.pointer;
        const hov = pt && pt.inside && pt.y > yt - 10 && pt.y < yb + 20 && pt.x > x0 && pt.x < x1 ? Math.floor((pt.x - x0) / cw) : -1;
        d.cats.forEach((cat, i) => {
          const cx = x0 + cw * i + cw / 2;
          d.series.forEach((s, j) => {
            const v = s.v[i];
            const bx = cx - (ns * bw) / 2 + j * bw + 2;
            if (v == null) { V.text(g, 'n/a', bx + (bw - 4) / 2, yb - 6, { size: 11, align: 'center', color: C.muted }); return; }
            const hv = (v / top) * (yb - yt) * e;
            const hz = d.hatch && d.hatch[i];
            V.box(g, bx, yb - hv, bw - 4, hv, { r: 4, fill: V.alpha(C[s.c], hz ? 0.45 : 0.82) });
            if (hz) { hatch(g, bx, yb - hv, bw - 4, hv, V.alpha(C.ink, 0.45), 6, t * 6); V.box(g, bx, yb - hv, bw - 4, hv, { r: 4, stroke: C.d, lw: 1.3 }); }
            if (showV) V.text(g, d.x ? nf(v, 2) + '×' : nf(v, d.d), bx + (bw - 4) / 2, yb - hv - 5, { size: 11, font: 'mono', align: 'center', color: C.ink, alpha: e });
          });
          V.text(g, cat, cx, yb + 16, { size: 11, align: 'center', color: hov === i ? C.ink : C.muted, maxW: cw - 4 });
          if (d.note && d.note[i] && hov !== i) {
            let mh = 0; d.series.forEach((s) => { if (s.v[i] != null) mh = Math.max(mh, s.v[i]); });
            const ls = wrapLines(g, '⚠ ' + d.note[i], Math.min(cw - 8, 220), 11, 'sans', 600);
            const ny = yb - (mh / top) * (yb - yt) * e - 24 - (ls.length - 1) * 14;
            ls.forEach((l, j) => V.text(g, l, cx, ny + j * 14, { size: 11, align: 'center', color: C.d, weight: 600, alpha: e }));
          }
        });
        if (hov >= 0 && hov < nc) {
          const lines = d.series.map((s) => s.n + ': ' + (s.v[hov] == null ? 'n/a' : d.x ? nf(s.v[hov], 2) + '×' : nf(s.v[hov], d.d)));
          const tw0 = Math.max(...lines.map((l) => tw(g, l, 11.5, 'mono'))) + 20, th0 = 18 * lines.length + 26;
          const cx = x0 + cw * hov + cw / 2;
          const bxT = V.clamp(cx - tw0 / 2, 8, w - tw0 - 8);
          V.box(g, bxT, yt, tw0, th0, { r: 8, fill: V.alpha(C.panel, 0.96), stroke: V.alpha(C.ink, 0.35), lw: 1 });
          V.text(g, d.cats[hov], bxT + 10, yt + 17, { size: 11.5, weight: 600, color: C.ink });
          lines.forEach((l, j) => { V.box(g, bxT + 10, yt + 26 + j * 18, 9, 9, { r: 2, fill: C[d.series[j].c] }); V.text(g, l, bxT + 24, yt + 35 + j * 18, { size: 11.5, font: 'mono', color: C.ink }); });
        } else if (!showV) V.text(g, 'tap a column to see the values', x1, yt - 6, { size: 11, align: 'right', color: C.muted });
        if (tw(g, d.foot, 11) <= w - 32) V.text(g, d.foot, 16, h - 14, { size: 11, color: C.muted });
        else {
          const ws = d.foot.split(' '); let l1 = '';
          while (ws.length && tw(g, (l1 + ' ' + ws[0]).trim(), 11) <= w - 32) l1 = (l1 + ' ' + ws.shift()).trim();
          V.text(g, l1, 16, h - 24, { size: 11, color: C.muted });
          V.text(g, ws.join(' '), 16, h - 9, { size: 11, color: C.muted, maxW: w - 32 });
        }
      });
    })();

    /* ============================================ F10 — quatro Sparks: switch ou anel */
    (function () {
      const fig = root.querySelector('#fig-switch-ring');
      if (!fig) return;
      const L = {
        fab: 'Fabric', sw: 'switch', ring: 'ring (no switch)',
        ord: '2D nodes around the ring',
        col: 'Collective',
        cols: { row: 'row partners (g⊕1)', col: 'column partners (g⊕2)', psum: 'pair sum (patch 0008)', ag: 'all-gather, ring algorithm', shot: 'one-shot to every peer', tree: 'tree (stock NCCL)' },
        part: 'Part per rank (P)',
        rH: 'max hops', rR: 'relayed transfers', rT: 'α-β estimate',
        steps: {
          row: ['row partners swap P'],
          col: ['column partners swap P'],
          psum: ['row sum, reduce-scatter half (½P)', 'row sum, all-gather half (½P)', 'swap the sums with the column partner (P)'],
          ag: ['pass a part to the next rank (1/3)', 'pass a part to the next rank (2/3)', 'pass a part to the next rank (3/3)'],
          shot: ['every rank writes P to every other rank'],
          tree: ['reduce: leaves → rank 2', 'reduce: rank 2 → root 0', 'broadcast: root 0 → rank 2', 'broadcast: rank 2 → leaves'],
        },
        switchBox: 'RoCE switch', twoPorts: 'two ports per Spark',
        relay: 'relay', noCable: 'no cable', spark: 'spark', rank: 'rank',
        fails: 'fails at connect', onRing: 'ring', onSwitch: 'switch', total: 'T ≈',
        foot: 'Model: α = 30 µs, β = 20 GB/s per link, two ports per Spark, relays forward in pieces. Illustrative, not measured.',
        note: (S) => {
          const ring = S.fab === 'ring', bad = S.ord === '0123';
          const anyOrder = 'On a switch every pair is one hop away, whatever the order.';
          if (S.col === 'row') return { s: ring ? 'Row partners (g and g⊕1, the two ranks of a TP2 pair) sit next to each other in both orders, so each pair has a cable of its own, as on the switch.' : anyOrder };
          if (S.col === 'col') return !ring ? { s: anyOrder } : bad ? { warn: true, s: 'Placed 0-1-2-3, the column partners (0↔2, 1↔3) sit across the ring: every transfer is relayed by a neighbour and the step pays α twice. Try 0-1-3-2.' } : { s: 'Placed 0-1-3-2 (ORDER=auto), the column partners are neighbours too: every partner pair has a cable of its own, as on the switch.' };
          if (S.col === 'psum') return ring && bad ? { warn: true, s: 'Here the column swap would need relays, so the ranks would not agree on pair sums at startup and patch 0008 keeps the four-quarter gather. The numbers show what relaying would cost.' } : { s: 'Two parts per rank (½ + ½, then 1) instead of the ring all-gather\'s three, with the same three α: it saves bytes, not latency, which is why it only runs on prompt chunks of 64+ rows.' };
          if (S.col === 'ag') return { s: 'NCCL\'s ring algorithm only sends to the next rank. With the NCCL ranks numbered in cable order, every transfer has a cable of its own, on the ring as on the switch: three steps of one part each.' };
          if (S.col === 'shot') return ring ? { warn: true, s: 'One-shot collectives write to every peer at once: the Spark across is reached through a neighbour, and each cable direction carries two parts. TensorFold\'s decode-size RDMA exchanges relay half through each neighbour; SGLang\'s RoCEnante needed research-only hardware forwarding in the neighbours\' ConnectX-7.' } : { s: 'On the switch every peer is one hop away; each Spark pushes three parts through its two ports.' };
          return ring ? { bad: true, s: 'NCCL builds its trees from rank numbers, and rank 0 ↔ rank 2 has no cable. Stock NCCL failed in ncclTransportTreeConnect on this ring; sparkring\'s NCCL_SWITCHLESS_RING_ONLY skips the tree and PAT setup, and TensorFold pins NCCL_ALGO=Ring with a graph file per rank.' } : { s: 'Up the tree and back down: four dependent steps in this unpipelined model (NCCL pipelines the tree in chunks).' };
        },
      };
      const stage = fig.querySelector('.la-fig-stage');
      stage.style.height = fig.clientWidth < 600 ? '680px' : '470px';
      const SIZES = [16 * 1024, 512 * 1024, 32 * 1024 * 1024];
      const ALPHA = 30e-6, BETA = 20e9; // modelo ilustrativo (ordem de grandeza do cap. 39)
      // posição no anel = rank NCCL (spark1..spark4, na ordem dos cabos); ROLE[ordem][posição] = nó 2D g
      const ROLE = { '0123': [0, 1, 2, 3], '0132': [0, 1, 3, 2] };
      const nb = (a, b) => (a - b + 4) % 4 === 1 || (b - a + 4) % 4 === 1;
      const fmtT = (s) => (!isFinite(s) ? '—' : s < 1e-3 ? nf(s * 1e6, 0) + ' µs' : nf(s * 1e3, 2) + ' ms');
      const S = { fab: 'ring', ord: '0132', col: 'col', size: 1, t0: 0, cur: null, alt: null };
      let H = null;

      // passos de cada coletiva: listas de transferências {a, b (posições), m (partes P)}
      function stepsOf(col, ord) {
        const P = (gg) => ROLE[ord].indexOf(gg);
        const pairs = (x, m) => [0, 1, 2, 3].map((gg) => ({ a: P(gg), b: P(gg ^ x), m }));
        const ring = () => [0, 1, 2, 3].map((r) => ({ a: r, b: (r + 1) % 4, m: 1 }));
        if (col === 'row') return [pairs(1, 1)];
        if (col === 'col') return [pairs(2, 1)];
        if (col === 'psum') return [pairs(1, 0.5), pairs(1, 0.5), pairs(2, 1)];
        if (col === 'ag') return [ring(), ring(), ring()];
        if (col === 'shot') { const f = []; for (let a = 0; a < 4; a++) for (let b = 0; b < 4; b++) if (a !== b) f.push({ a, b, m: 1 }); return [f]; }
        // árvore binária ilustrativa montada pelos números de rank: 2 é o nó interno, 0 a raiz
        return [[{ a: 1, b: 2, m: 1 }, { a: 3, b: 2, m: 1 }], [{ a: 2, b: 0, m: 1 }], [{ a: 0, b: 2, m: 1 }], [{ a: 2, b: 1, m: 1 }, { a: 2, b: 3, m: 1 }]];
      }
      // rota de cada transferência e carga por direção de cabo (anel) ou por porta (switch)
      function route(flows, fab, col) {
        const fl = flows.map((f) => Object.assign({}, f));
        if (fab === 'sw') {
          const ld = {};
          let Lm = 0;
          fl.forEach((f) => { f.path = [f.a, 'S', f.b]; f.hops = 1; ld['u' + f.a] = (ld['u' + f.a] || 0) + f.m; ld['d' + f.b] = (ld['d' + f.b] || 0) + f.m; Lm = Math.max(Lm, f.m); });
          Object.keys(ld).forEach((k) => { Lm = Math.max(Lm, ld[k] / 2); }); // duas portas por Spark
          return { fl, H: 1, L: Lm, cnt: {}, fail: false };
        }
        const cnt = {};
        const inc = (x, y) => { const k = x + '>' + y; cnt[k] = (cnt[k] || 0) + 1; };
        if (col === 'tree') {
          let fail = false;
          fl.forEach((f) => { f.path = [f.a, f.b]; f.hops = 1; if (nb(f.a, f.b)) inc(f.a, f.b); else { f.nocable = true; fail = true; } });
          return { fl, H: 1, L: 1, cnt, fail };
        }
        const opp = fl.filter((f) => !nb(f.a, f.b));
        let best = null;
        for (let mask = 0; mask < (1 << opp.length); mask++) {
          const ld = {};
          const add = (x, y, m) => { const k = x + '>' + y; ld[k] = (ld[k] || 0) + m; };
          fl.forEach((f) => { if (nb(f.a, f.b)) add(f.a, f.b, f.m); });
          opp.forEach((f, i) => { const via = (mask >> i) & 1 ? (f.a + 3) % 4 : (f.a + 1) % 4; add(f.a, via, f.m); add(via, f.b, f.m); });
          const Lm = Math.max(...Object.values(ld));
          if (!best || Lm < best.L - 1e-9) best = { L: Lm, mask };
        }
        opp.forEach((f, i) => { f.via = (best.mask >> i) & 1 ? (f.a + 3) % 4 : (f.a + 1) % 4; });
        fl.forEach((f) => {
          if (nb(f.a, f.b)) { f.path = [f.a, f.b]; f.hops = 1; inc(f.a, f.b); }
          else { f.path = [f.a, f.via, f.b]; f.hops = 2; f.relay = true; inc(f.a, f.via); inc(f.via, f.b); }
        });
        return { fl, H: Math.max(...fl.map((f) => f.hops)), L: best.L, cnt, fail: false };
      }
      function plan(fab) {
        const st = stepsOf(S.col, S.ord).map((f) => route(f, fab, S.col));
        let T = 0, Hm = 0, rel = 0, fail = false;
        st.forEach((r) => { r.T = r.H * ALPHA + (r.L * SIZES[S.size]) / BETA; T += r.T; Hm = Math.max(Hm, r.H); rel += r.fl.filter((f) => f.relay).length; if (r.fail) fail = true; });
        return { st, T: fail ? Infinity : T, Hm, rel, fail };
      }

      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: L.fab, options: [{ v: 'sw', label: L.sw }, { v: 'ring', label: L.ring }], value: 'ring', onChange: (v) => { S.fab = v; update(); } });
      UI.seg(ctl, { label: L.ord, options: [{ v: '0123', label: '0-1-2-3' }, { v: '0132', label: '0-1-3-2' }], value: '0132', onChange: (v) => { S.ord = v; update(); } });
      const selC = UI.select(ctl, { label: L.col, options: ['row', 'col', 'psum', 'ag', 'shot', 'tree'].map((k) => ({ v: k, label: L.cols[k] })), value: 'col', onChange: (v) => { S.col = v; update(); } });
      selC.el.querySelector('select').style.maxWidth = '240px';
      UI.seg(ctl, { label: L.part, options: [{ v: 0, label: '16 KiB' }, { v: 1, label: '512 KiB' }, { v: 2, label: '32 MiB' }], value: 1, onChange: (v) => { S.size = v; update(); } });
      const rH = UI.readout(ctl, L.rH, '—');
      const rR = UI.readout(ctl, L.rR, '—');
      const rT = UI.readout(ctl, L.rT, '—');
      function update() {
        S.cur = plan(S.fab);
        S.alt = plan(S.fab === 'ring' ? 'sw' : 'ring');
        S.t0 = H ? H.t : 0;
        rH.set(S.cur.fail ? '—' : String(S.cur.Hm), S.cur.Hm > 1 ? 'warn' : null);
        rR.set(String(S.cur.rel), S.cur.rel ? 'warn' : null);
        rT.set(S.cur.fail ? L.fails : fmtT(S.cur.T), S.cur.fail || S.cur.T > S.alt.T * 1.05 ? 'bad' : null);
      }
      update();

      H = V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const narrow = w < 600;
        const cur = S.cur, role = ROLE[S.ord];
        const nSt = cur.st.length, D = 2.4;
        const el = Math.max(0, t - S.t0);
        const si = Math.floor(el / D) % nSt;
        const e = V.ease.inOut(V.clamp(((el % D) / D) / 0.75, 0, 1));
        // geometria: topologia à esquerda (ou em cima, no celular), painel ao lado (ou embaixo)
        const topW = narrow ? w : Math.min(w * 0.56, h + 40);
        const topH = narrow ? Math.min(h * 0.5, w * 0.95) : h;
        const nw = narrow ? 94 : 120, nh = narrow ? 46 : 52;
        const half = Math.max(60, Math.min(topW, topH) / 2 - (narrow ? 54 : 66));
        const cx = topW / 2, cy = topH / 2;
        const PT = [[cx - half, cy - half], [cx + half, cy - half], [cx + half, cy + half], [cx - half, cy + half]];
        const pairCol = (gg) => (gg >> 1 ? C.b : C.a);
        const edge = (i, x, y) => { const ax = PT[i][0], ay = PT[i][1]; const dx = x - ax, dy = y - ay; const k = Math.min((nw / 2 + 3) / Math.max(1e-6, Math.abs(dx)), (nh / 2 + 3) / Math.max(1e-6, Math.abs(dy))); return [ax + dx * k, ay + dy * k]; };
        const swW = narrow ? 84 : 104, swH = 30;
        const swEdge = (i) => { const dx = PT[i][0] - cx, dy = PT[i][1] - cy; const k = Math.min((swW / 2) / Math.max(1e-6, Math.abs(dx)), (swH / 2) / Math.max(1e-6, Math.abs(dy))); return [cx + dx * k, cy + dy * k]; };
        const shift = (p, q, d) => { const dx = q[0] - p[0], dy = q[1] - p[1], len = Math.hypot(dx, dy) || 1; const ox = (-dy / len) * d, oy = (dx / len) * d; return [[p[0] + ox, p[1] + oy], [q[0] + ox, q[1] + oy]]; };
        // segmentos de uma transferência
        const segsOf = (f) => {
          if (S.fab === 'sw') return [[edge(f.a, cx, cy), swEdge(f.a)], [swEdge(f.b), edge(f.b, cx, cy)]];
          if (f.relay) return [[edge(f.a, PT[f.via][0], PT[f.via][1]), edge(f.via, PT[f.a][0], PT[f.a][1])], [edge(f.via, PT[f.b][0], PT[f.b][1]), edge(f.b, PT[f.via][0], PT[f.via][1])]];
          return [[edge(f.a, PT[f.b][0], PT[f.b][1]), edge(f.b, PT[f.a][0], PT[f.a][1])]];
        };
        // malha física
        if (S.fab === 'sw') {
          for (let i = 0; i < 4; i++) {
            const p = edge(i, cx, cy), q = swEdge(i);
            [-3, 3].forEach((d) => { const s2 = shift(p, q, d); V.line(g, s2[0][0], s2[0][1], s2[1][0], s2[1][1], { color: V.alpha(C.muted, 0.55), lw: 2 }); });
          }
          V.box(g, cx - swW / 2, cy - swH / 2, swW, swH, { r: 8, fill: V.alpha(C.panel, 0.98), stroke: V.alpha(C.ink, 0.45), lw: 1.4 });
          V.text(g, L.switchBox, cx, cy + 1, { size: 11, weight: 600, align: 'center', base: 'middle', color: C.ink, maxW: swW - 8 });
          const tpw = Math.min(2 * half - nw, tw(g, L.twoPorts, 11) + 14);
          V.box(g, cx - tpw / 2, cy + swH / 2 + 4, tpw, 18, { r: 6, fill: V.alpha(C.panel, 0.95) });
          V.text(g, L.twoPorts, cx, cy + swH / 2 + 13.5, { size: 11, align: 'center', base: 'middle', color: C.muted, maxW: tpw - 6 });
        } else {
          for (let i = 0; i < 4; i++) {
            const j = (i + 1) % 4, p = edge(i, PT[j][0], PT[j][1]), q = edge(j, PT[i][0], PT[i][1]);
            V.line(g, p[0], p[1], q[0], q[1], { color: V.alpha(C.muted, 0.5), lw: 4 });
          }
        }
        // transferências do passo atual
        const stp = cur.st[si];
        const relays = {};
        stp.fl.forEach((f) => {
          const col = pairCol(role[f.a]);
          if (f.nocable) {
            const p = edge(f.a, PT[f.b][0], PT[f.b][1]), q = edge(f.b, PT[f.a][0], PT[f.a][1]);
            V.line(g, p[0], p[1], q[0], q[1], { color: V.alpha(C.red, 0.85), lw: 2.2, dash: [7, 5], dashOffset: -t * 10 });
            chip(g, '✕ ' + L.noCable, (p[0] + q[0]) / 2, (p[1] + q[1]) / 2 - 12, C.red, { font: 'sans', size: 11 });
            return;
          }
          const segs = segsOf(f);
          segs.forEach((sg, k) => {
            const s2 = shift(sg[0], sg[1], 5);
            const last = k === segs.length - 1;
            if (last) V.arrow(g, s2[0][0], s2[0][1], s2[1][0], s2[1][1], { color: V.alpha(col, 0.75), lw: 2, head: 7 });
            else V.line(g, s2[0][0], s2[0][1], s2[1][0], s2[1][1], { color: V.alpha(col, 0.75), lw: 2 });
          });
          if (f.relay) {
            relays[f.via] = true;
            // o par lógico, do outro lado do anel, sem cabo
            const p = edge(f.a, PT[f.b][0], PT[f.b][1]), q = edge(f.b, PT[f.a][0], PT[f.a][1]);
            V.line(g, p[0], p[1], q[0], q[1], { color: V.alpha(col, 0.3), lw: 1.2, dash: [4, 6] });
          }
          // ponto viajando: metade do tempo por segmento
          const nS = segs.length, k = Math.min(nS - 1, Math.floor(e * nS)), uu = e * nS - k;
          const s2 = shift(segs[k][0], segs[k][1], 5);
          const pt = V.along(s2[0][0], s2[0][1], s2[1][0], s2[1][1], uu);
          V.glow(g, pt[0], pt[1], 9, col, 0.9); V.dot(g, pt[0], pt[1], 3.2, col);
        });
        // direções de cabo com mais de uma transferência
        if (S.fab === 'ring') Object.keys(stp.cnt).forEach((key) => {
          const n = stp.cnt[key];
          if (n < 2) return;
          const [a, b] = key.split('>').map(Number);
          const s2 = shift(edge(a, PT[b][0], PT[b][1]), edge(b, PT[a][0], PT[a][1]), 15);
          chip(g, '×' + n, (s2[0][0] + s2[1][0]) / 2, (s2[0][1] + s2[1][1]) / 2, C.d, { size: 11 });
        });
        // nós
        for (let i = 0; i < 4; i++) {
          const gg = role[i], x = PT[i][0], y = PT[i][1];
          V.box(g, x - nw / 2, y - nh / 2, nw, nh, { r: 10, fill: V.alpha(C.panel, 0.98), stroke: pairCol(gg), lw: 2 });
          V.text(g, L.spark + (i + 1) + ' · ' + L.rank + ' ' + i, x, y - 5, { size: narrow ? 11 : 12, weight: 600, align: 'center', color: C.ink, maxW: nw - 10 });
          V.text(g, 'g=' + gg + ' (p' + (gg >> 1) + ', r' + (gg & 1) + ')', x, y + 12, { size: 11, font: 'mono', align: 'center', color: C.muted, maxW: nw - 8 });
          if (relays[i]) chip(g, L.relay, x, i < 2 ? y - nh / 2 - 13 : y + nh / 2 + 13, C.d, { font: 'sans', size: 11 });
        }
        // painel
        const px = narrow ? 16 : topW + 10, pw = narrow ? w - 32 : w - topW - 26;
        let y = narrow ? topH + 18 : 28;
        V.text(g, L.cols[S.col] + ' · ' + (S.fab === 'ring' ? L.ring : L.sw), px, y, { size: 13, font: 'display', weight: 700, color: C.ink, maxW: pw });
        y += 20;
        cur.st.forEach((r, k) => {
          const on = k === si;
          const lab = (on ? '▸ ' : '  ') + (k + 1) + '. ' + L.steps[S.col][k];
          y += V.wrap(g, lab, px, y, pw, { size: 11.5, weight: on ? 600 : 400, color: on ? C.ink : C.muted, maxLines: 2, lineH: 15 });
          const met = r.fail ? '✕ ' + L.noCable : 'H=' + r.H + ' · L=' + nf(r.L, r.L % 1 ? 1 : 0) + 'P → ' + fmtT(r.T);
          V.text(g, met, px + 16, y, { size: 11, font: 'mono', color: r.fail ? C.red : on ? C.ink : C.muted, maxW: pw - 16 });
          y += 18;
        });
        y += 4;
        const here = S.fab === 'ring' ? L.onRing : L.onSwitch, there = S.fab === 'ring' ? L.onSwitch : L.onRing;
        const tot = (cur.fail ? L.fails : L.total + ' ' + fmtT(cur.T)) + ' (' + here + ') · ' + (S.alt.fail ? L.fails : L.total + ' ' + fmtT(S.alt.T)) + ' (' + there + ')';
        y += V.wrap(g, tot, px, y, pw, { size: 11.5, font: 'mono', weight: 600, color: cur.fail ? C.red : C.ink, maxLines: 2, lineH: 15 });
        y += 6;
        const nt = L.note(S, cur);
        const footLines = 2, footH = footLines * 14 + 4;
        const room = Math.max(1, Math.floor((h - footH - 6 - y) / 15));
        V.wrap(g, nt.s, px, y, pw, { size: 11.5, color: nt.bad ? C.red : nt.warn ? C.d : C.muted, maxLines: room, lineH: 15 });
        V.wrap(g, L.foot, px, h - footH + 8, pw, { size: 11, color: C.muted, maxLines: footLines, lineH: 14 });
      });
    })();

    /* ============================================ tira da política do MoE cooperativo (DOM) */
    (function () {
      const box = root.querySelector('#coop-policy');
      if (!box) return;
      // extensions/cooperative_moe/tp3/dispatch_policy.json: geometria por linhas; 25 contagens medidas
      const POL = { 1: 1, 2: 2, 3: 2, 4: 2, 5: 2, 6: 1, 7: 1, 8: 1, 9: 1, 10: 2, 12: 1, 15: 1, 16: 1, 18: 2, 20: 1, 24: 2, 25: 1, 30: 1, 32: 1, 35: 1, 40: 1, 48: 1, 56: 1 };
      const LOST = { 21: '0.9734', 64: '1.0568' }; // medidas, mas acima do corte de 0,97: ficam no stock
      const col = (v) => (v === 1 ? 'var(--viz-a)' : v === 2 ? 'var(--viz-b)' : v === 'm' ? 'var(--viz-d)' : 'var(--viz-muted)');
      const wrap = LA.h('div', { style: { display: 'flex', flexWrap: 'wrap', gap: '4px', margin: '0', padding: '14px', borderRadius: '14px', border: '1px solid var(--line-2)', background: 'var(--viz-bg)' } });
      for (let r = 1; r <= 64; r++) {
        const v = POL[r] || (LOST[r] ? 'm' : 's');
        const unmeasured = v === 's';
        const tip = v === 1 ? 'geometry 1' : v === 2 ? 'geometry 2' : v === 'm' ? 'stock (measured, no gain: worst case ' + LOST[r] + ')' : 'stock (not measured)';
        const sp = LA.h('span', {
          title: r + (r === 1 ? ' row: ' : ' rows: ') + tip,
          style: { width: '26px', height: '24px', display: 'inline-flex', alignItems: 'center', justifyContent: 'center', borderRadius: '6px', fontFamily: 'var(--font-mono)', fontSize: '11px', color: unmeasured ? 'var(--text-3)' : 'var(--text)', background: unmeasured ? 'transparent' : `color-mix(in srgb, ${col(v)} ${v === 'm' ? 16 : 34}%, transparent)`, border: unmeasured ? '1px dashed color-mix(in srgb, var(--viz-muted) 55%, transparent)' : `${v === 'm' ? 2 : 1}px solid color-mix(in srgb, ${col(v)} 70%, transparent)` },
        }, String(r));
        wrap.appendChild(sp);
      }
      const leg = LA.h('div', { style: { display: 'flex', flexWrap: 'wrap', gap: '6px 16px', marginTop: '10px', fontSize: '12.5px', color: 'var(--text-2)', width: '100%' } });
      const sw = (st) => LA.h('i', { style: Object.assign({ width: '12px', height: '12px', borderRadius: '3px', display: 'inline-block', boxSizing: 'border-box' }, st) });
      [
        [sw({ background: 'var(--viz-a)' }), 'geometry 1 (A-wide/B-wide)'],
        [sw({ background: 'var(--viz-b)' }), 'geometry 2 (A-wide/B-narrow)'],
        [sw({ border: '2px solid var(--viz-d)' }), 'stock, measured with no gain (21: 0.9734; 64: 1.0568)'],
        [sw({ border: '1px dashed var(--viz-muted)' }), 'stock, not measured'],
      ].forEach(([ic, l]) => {
        leg.appendChild(LA.h('span', { style: { display: 'inline-flex', alignItems: 'center', gap: '6px' } }, ic, l));
      });
      wrap.appendChild(leg);
      box.appendChild(wrap);
    })();
  },
});

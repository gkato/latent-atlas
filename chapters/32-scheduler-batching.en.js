LA.chapter({
  id: 'scheduler-batching',
  part: 6,
  order: 32,
  title: 'Scheduler, continuous batching and chunked prefill',
  navTitle: 'Scheduler & batching',
  subtitle: 'Who gets into each step: token and sequence budgets, sliced prefill, fairness, preemption and async scheduling — and how to pick the numbers for your recipe.',
  minutes: 50,
  recipes: ['glm', 'tf', 'qwen', 'ds'],
  tags: ['scheduler', 'continuous batching', 'chunked prefill', 'fairness', 'preemption', 'async scheduling', 'max_num_seqs', 'max_num_batched_tokens', 'TensorFold', 'multi-prefill'],
  goals: [
    'Explain the V1 scheduler loop and its two budgets: <code>max_num_seqs</code> and <code>max_num_batched_tokens</code>',
    'Compute verification rows, capture sizes and the chunked-prefill TTFT × pause trade-off',
    'Diagnose decode starvation and choose protections (#27, #43, <code>fair</code> v5/v7)',
    'Build a serving profile and validate it with a concurrency ladder, spread and preemptions',
    'Compare with the TensorFold scheduler: chunk <i>or</i> round by an iteration credit, a 32-row batched window and multi-prefill',
  ],
  prereqs: ['paged-attention-kv', 'prefill-decode-kv', 'cuda-graphs-warmup'],
  summary: [
    'At each step the V1 scheduler hands each request \\(s_r \\le n_r - c_r\\) tokens, within two budgets: <code>max_num_batched_tokens</code> (tokens) and <code>max_num_seqs</code> (slots) — RUNNING first, in order, then WAITING.',
    'Continuous batching refills slots at every step; since decode is bandwidth-bound, more streams raise the aggregate and lower each stream\'s speed (Qwen: 3.8× aggregate, 49% per stream from ×1 to ×8).',
    'With speculation each decode costs \\(1+k\\) positions: the verification rows \\(S(1+k)\\) dictate the CUDA graphs (DeepSeek: 6 × 7 = 42 → capture 48).',
    'Chunked prefill trades TTFT for decode pause (\\(t \\approx a + b\\,c\\)); the long-prefill threshold caps the chunk per request (DeepSeek 1024) and, in hybrids, the chunk stops at the state checkpoints (GLM 3584).',
    'Starvation without preemption is invisible: DeepSeek caps in-flight prefills (#27) and reserves a decode floor (#43); GLM splits time, not tokens (<code>fair</code> v5), because every mixed step leaves the FULL graph and costs ~0.3 s. The TensorFold version of GLM never mixes: each iteration is a chunk <i>or</i> a round, alternated by a credit in iterations (<code>TF_GLM_FILL_SHARE</code> 0.5), and prompts that arrive together share a chunk (TTFT ×4: 590 → 340 ms).',
    'Ceilings are not reservations: the sum of live tokens has to fit in the pool, otherwise there is preemption and recomputation; async scheduling hides CPU time but creates placeholders.',
    'Serving profiles are validated with concurrency ladders that measure aggregate, per-stream speed, TTFT, spread and preemptions — one variable at a time.',
  ],
  quiz: [
    { q: 'In vLLM V1, a decode gets no tokens because prefills at the front of the RUNNING list consumed the budget. What happens to it?', o: ['It is preempted and redoes its prefill', 'It gets 0 tokens in that step, keeps its KV and does not count as a preemption', 'It goes to the end of the WAITING queue', 'vLLM raises the budget automatically'], a: 1, e: 'The RUNNING loop stops when the budget runs out, and any request with 0 tokens is skipped with <code>continue</code>. The KV is still there and no preemption metric fires — that is why the #27 starvation had zero preemptions.' },
    { q: 'DeepSeek recipe: 6 slots and k = 6 drafts. How many verification rows does the full decode step have, and which capture size does the compose file use?', o: ['6 and 8', '36 and 40', '42 and 48', '48 and 56'], a: 2, e: '6 × (1+6) = 42 rows; the compose file rounds up to a multiple of 8 → 48. The raw value 42 was truncated to 40, which left graphs of 7…35, and the 42-token step ran eager (aggregate at c=6: 122 vs 139 tok/s).' },
    { q: 'Why did GLM swap token caps (<code>GLM53_MIXED_PREFILL_CHUNK=N</code>) for time-based fairness (<code>fair</code> v5)?', o: ['Because prefill tokens cost more than decode tokens', 'Because every mixed step leaves the uniform-decode FULL graph and pays ~0.3 s fixed: even 128 tokens leave the incumbent at ~10 tok/s', 'Because MNBT cannot exceed 2048', 'Because KDA does not support chunked prefill'], a: 1, e: 'A cap in tokens is not a budget in milliseconds. The measured cost is ~0.3 s + 0.68 ms/token; that is why fair v5 credits a fraction of the time and picks the largest rung whose estimate fits in <code>MAX_STEP_MS</code>.' },
    { q: 'With the cost \\(t(c) = a + b\\,c\\), what happens when you halve the prefill chunk \\(c\\)?', o: ['Both TTFT and pause improve', 'The decode pause shrinks, but the long prompt\'s TTFT gets worse because the fixed cost \\(a\\) is paid twice as many times', 'Nothing changes: the total number of tokens is the same', 'TTFT improves and the pause gets worse'], a: 1, e: 'TTFT ≈ \\(\\lceil P/c \\rceil a + bP\\): the linear term does not change, but the number of fixed costs doubles. The pause ≈ \\(a + bc\\) drops. That is exactly the trade-off the long-prefill threshold controls.' },
    { q: 'On Qwen, <code>max_model_len</code> = 1M and <code>max_num_seqs</code> = 8, but the pool holds ~2.5M tokens. What happens if 8 requests of 1M arrive?', o: ['OOM, and the server crashes', 'vLLM rejects all of them', 'About 2 stay resident; the others are preempted and redo their prefill when they come back', 'vLLM shortens each context to 312K'], a: 2, e: 'Ceilings do not reserve memory. When the blocks run out, the scheduler preempts (RECOMPUTE mode) and the request goes back to the queue; the cost is redoing its prefill. Hence the context × slots profile table in the README.' },
    { q: 'What does async scheduling gain, and what complexity does it create?', o: ['More KV cache, at the cost of more memory', 'It hides the scheduler\'s CPU time by overlapping it with the GPU step; it requires placeholders (−1) for tokens and drafts that are not yet known', 'Fewer all-reduces, but it requires an even TP size', 'It runs prefill and decode on different GPUs'], a: 1, e: 'Step N+1 is planned before N finishes; the tokens of N enter as placeholders. On DeepSeek this is worth ~4% single-stream, and the interactions with grammars and drafts (#136, #191) required care; the MiMo tutorial turns it off because of a spec decode + async bug.' },
    { q: 'In the v1.3.2 GLM·TF recipe, <code>TF_GLM_FILL_SHARE</code> = 0.5 and a long prompt arrives while another conversation is decoding. Why can the decoding conversation end up with far less than half of the time?', o: ['Because the prompt chunk and the decode rows share the same forward, and the chunk takes almost all the rows', 'Because TensorFold reserves half of the 32-row window for the prompt', 'Because the credit counts iterations, not seconds: half the iterations are 1,024-row chunks, and each chunk lasts many decode rounds', 'Because decode leaves the CUDA graph while a prompt is filling'], a: 2, e: 'Round, chunk, round, chunk… is half of the <i>iterations</i>. But in the figure\'s toy model a 1,024-row chunk takes ~0.59 s and a round ~70 ms, so the prompt gets ~89% of the time. Prefill and decode never share a forward there, and the round stays on the graph for its row count. vLLM\'s <code>fair</code> v5 does the opposite: it credits a fraction of <i>time</i>. Since v1.4, the chunk is sliced by layers (~200 ms, patch 0062), with a round between slices.' },
  ],
  mount(root, LA) {
    const V = LA.viz, UI = LA.ui;
    const q = (s) => root.querySelector(s);
    const nf = (x, d = 0) => Number(x).toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d });
    const secs = (x) => (!isFinite(x) ? '—' : (x >= 100 ? nf(x, 0) : x >= 10 ? nf(x, 1) : nf(x, 2)) + ' s');
    const ink = (C) => (C.dark ? '#06120c' : '#ffffff');
    const clamp = V.clamp;

    /* stage height by width (desktop vs phone) */
    function autoH(fig, desk, mob, bp) {
      const st = fig.querySelector('.la-fig-stage');
      const set = () => {
        const want = ((st.clientWidth || 800) < (bp || 600) ? mob : desk) + 'px';
        if (st.style.height !== want) st.style.height = want;
      };
      set();
      const ro = new ResizeObserver(set);
      ro.observe(st);
      if (LA.onCleanup) LA.onCleanup(() => ro.disconnect());
    }
    /* token character */
    function critter(g, x, y, r, col, mood, t) {
      if (!(r > 0.8)) return;
      const C = V.C();
      if (t) y += Math.sin(t * 3 + x * 0.05) * r * 0.07;
      V.dot(g, x, y, r, col);
      const ec = ink(C), er = Math.max(1.1, r * 0.13), ex = r * 0.34, ey = y - r * 0.12;
      g.strokeStyle = ec; g.lineWidth = Math.max(1, r * 0.11);
      if (mood === 'sleep') {
        g.beginPath(); g.moveTo(x - ex - er, ey); g.lineTo(x - ex + er, ey); g.moveTo(x + ex - er, ey); g.lineTo(x + ex + er, ey); g.stroke();
      } else { V.dot(g, x - ex, ey, er, ec); V.dot(g, x + ex, ey, er, ec); }
      g.beginPath();
      if (mood === 'sad') g.arc(x, y + r * 0.62, r * 0.3, 1.2 * Math.PI, 1.8 * Math.PI);
      else if (mood === 'happy') g.arc(x, y + r * 0.1, r * 0.32, 0.18 * Math.PI, 0.82 * Math.PI);
      else { g.moveTo(x - r * 0.22, y + r * 0.38); g.lineTo(x + r * 0.22, y + r * 0.38); }
      g.stroke();
    }
    /* simple word-wrapped text (returns the y of the last line) */
    function wrapText(g, str, x, y, maxW, lh, o) {
      g.font = V.font(o.size || 12, o.font || 'sans', o.weight || 400);
      const words = String(str).split(' ');
      let line = '', yy = y;
      words.forEach((wd) => {
        const test = line ? line + ' ' + wd : wd;
        if (g.measureText(test).width > maxW && line) { V.text(g, line, x, yy, o); line = wd; yy += lh; }
        else line = test;
      });
      if (line) V.text(g, line, x, yy, o);
      return yy;
    }
    function hatch(g, x, y, w, h, col, step) {
      if (w <= 0.5 || h <= 0.5) return;
      step = step || 6;
      g.save(); g.beginPath(); g.rect(x, y, w, h); g.clip();
      g.strokeStyle = col; g.lineWidth = 1; g.beginPath();
      for (let k = -h; k < w; k += step) { g.moveTo(x + k, y + h); g.lineTo(x + k + h, y); }
      g.stroke(); g.restore();
    }
    /* fills a grid of cells with [a,b) token segments (proportional slices) */
    function tokenGrid(g, o) {
      const C = V.C();
      const n = o.cols * o.rows, per = o.total / n, cs = o.cell, gap = o.gap == null ? 2 : o.gap;
      for (let k = 0; k < n; k++) {
        const x0 = o.x + (k % o.cols) * cs, y0 = o.y + Math.floor(k / o.cols) * cs;
        const w0 = cs - gap;
        g.fillStyle = V.alpha(C.muted, 0.12); g.fillRect(x0, y0, w0, w0);
        const lo = k * per, hi = lo + per;
        o.segs.forEach((s) => {
          const a = Math.max(s.a, lo), b = Math.min(s.b, hi);
          if (b > a) { g.fillStyle = s.col; g.fillRect(x0 + ((a - lo) / per) * w0, y0, Math.max(0.8, ((b - a) / per) * w0), w0); }
        });
      }
    }

    /* =====================================================================
       Figure 1 — one EngineCore iteration (stepper)
       ===================================================================== */
    {
      const fig = q('#fig-loop');
      autoH(fig, 380, 640, 620);
      const st = UI.stepper(fig, [
        { t: 'Arrival', d: 'The API server tokenizes the request and sends it to the EngineCore. Request <b>D</b> enters the <b>WAITING</b> queue with \\(c=0\\) and \\(n=3{,}000\\) (the prompt length).' },
        { t: 'RUNNING first', d: 'The scheduler walks RUNNING in admission order. Decodes <b>A</b> and <b>B</b> need 1 token each; prefill <b>C</b> still has 7,000 tokens to go, but the long-prefill threshold (2,048, illustrative) cuts the chunk.' },
        { t: 'Then WAITING', d: 'There is budget left and a free slot: <b>D</b> is admitted. The prefix cache already holds its first 512 tokens, so \\(c\\) jumps to 512; it gets the 2,046 tokens left in the tray.' },
        { t: 'execute_model()', d: 'The batch becomes a single flattened matrix \\(X\\) of 4,096 rows × \\(d\\) (one row per token, from four requests). The linear layers and the MoE read each weight once and multiply it by the whole \\(X\\) (\\(M = 4{,}096\\)); only attention uses the <code>query_start_loc</code> boundaries = [0, 1, 2, 2050, 4096] so that each row sees only the KV of its own request. A single forward processes everything, with the all-reduces between the nodes.' },
        { t: 'update_from_output()', d: 'Sampling: A and B get one token each; C and D advance their \\(c\\). A has reached <code>max_tokens</code> and leaves, freeing its blocks and its slot. In the next step, everything starts over.' },
      ], { autoplay: 6500 });

      function card(g, x, y, w, hh, o) {
        const C = V.C();
        V.box(g, x, y, w, hh, { r: 10, fill: V.alpha(o.col, o.hi ? 0.2 : 0.08), stroke: V.alpha(o.col, o.hi ? 0.95 : 0.4), lw: o.hi ? 2 : 1 });
        critter(g, x + 17, y + hh / 2 - 2, 10, o.col, o.mood || 'neutral', o.t);
        V.text(g, o.title, x + 34, y + 17, { size: 12.5, weight: 600, color: C.ink, maxW: w - 40 });
        V.text(g, o.sub, x + 34, y + 31, { size: 11, font: 'mono', color: C.muted, maxW: w - 40 });
        if (o.frac != null) {
          const bw = w - 42, by = y + hh - 11;
          V.box(g, x + 34, by, bw, 5, { r: 3, fill: V.alpha(C.muted, 0.22) });
          V.box(g, x + 34, by, Math.max(3, bw * clamp(o.frac, 0, 1)), 5, { r: 3, fill: o.col });
        }
        if (o.badge) {
          g.font = V.font(11, 'mono', 600);
          const bw = g.measureText(o.badge).width + 12;
          V.box(g, x + w - bw - 4, y - 8, bw, 17, { r: 8, fill: o.badgeCol || o.col, label: o.badge, size: 11, font: 'mono', color: ink(V.C()) });
        }
      }

      V.fig(fig, (g, w, h, t) => {
        if (w < 60 || h < 60) return;
        const C = V.C();
        const i = st.i, el = st.elapsed();
        const e = V.ease.inOut(Math.min(1, el / 1.2));
        const walk = Math.min(1, el / 2.6);
        const narrow = w < 620;
        const G = {};
        if (!narrow) {
          G.wx = 14; G.wy = 42; G.ww = Math.min(170, w * 0.19); G.rx = G.wx + G.ww + 16; G.rw = Math.min(330, w * 0.37);
          G.ry = 42; G.tx = G.rx + G.rw + 24; G.ty = 42; G.tw = w - G.tx - 14; G.by = h - 44;
        } else {
          G.wx = 10; G.wy = 40; G.ww = Math.max(100, w * 0.31); G.rx = G.wx + G.ww + 8; G.rw = w - G.rx - 10;
          G.ry = 40; G.tx = 10; G.tw = w - 20; G.ty = 40 + 4 * 58 + 40; G.by = h - 46;
        }
        const cell = Math.min(G.tw / 16, narrow ? 22 : 26), th = cell * 8;
        const cardH = 50, step = 58;
        V.text(g, 'WAITING', G.wx, G.wy - 14, { size: 11, font: 'mono', color: C.d, weight: 600 });
        V.text(g, 'RUNNING (order)', G.rx, G.ry - 14, { size: 11, font: 'mono', color: C.a, weight: 600, maxW: G.rw });
        V.text(g, 'TRAY · B = 4,096 tokens (1 cell = 32)', G.tx, G.ty - 14, { size: 11, font: 'mono', color: C.b, weight: 600, maxW: G.tw });

        const colA = C.a, colB = C.e, colC = C.b, colD = C.c;
        const visit = (k) => (i === 1 ? (walk > k * 0.3 && walk < (k + 1) * 0.3 + 0.08) : false);
        // A
        const aAlpha = i === 4 ? 1 - e * 0.55 : 1;
        const servedA = i >= 1 && (i > 1 || walk > 0.15), servedB = i >= 1 && (i > 1 || walk > 0.45);
        g.globalAlpha = aAlpha;
        card(g, G.rx, G.ry, G.rw, cardH, { col: colA, title: i === 4 ? 'A · finished' : 'A · decode', sub: i === 4 ? 'c 940 · n 940 · slot freed' : 'c 939 · n 940', frac: 1, hi: visit(0), t, mood: servedA ? 'happy' : 'neutral', badge: servedA && i !== 4 ? '+1' : null });
        g.globalAlpha = 1;
        if (i === 4) V.box(g, G.rx + G.rw - 62, G.ry - 8, 58, 17, { r: 8, fill: colA, label: '✓ left', size: 11, font: 'mono', color: ink(C) });
        card(g, G.rx, G.ry + step, G.rw, cardH, { col: colB, title: 'B · decode', sub: i === 4 ? 'c 2,311 · n 2,312' : 'c 2,310 · n 2,311', frac: 1, hi: visit(1), t, mood: servedB ? 'happy' : 'neutral', badge: servedB ? '+1' : null });
        const cC = i === 4 ? V.lerp(5000, 7048, e) : 5000;
        card(g, G.rx, G.ry + 2 * step, G.rw, cardH, { col: colC, title: 'C · prefill', sub: 'c ' + nf(cC) + ' · n 12,000', frac: cC / 12000, hi: visit(2), t, badge: (i >= 1 && (i > 1 || walk > 0.72)) ? '+2,048' : null });
        // D
        let dx, dy, dcNow = 0, dRun = false;
        const wPos = [G.wx, G.wy], rPos = [G.rx, G.ry + 3 * step];
        if (i === 0) { dx = V.lerp(-G.ww - 20, wPos[0], e); dy = wPos[1]; }
        else if (i === 1) { dx = wPos[0]; dy = wPos[1]; }
        else if (i === 2) { dx = V.lerp(wPos[0], rPos[0], e); dy = V.lerp(wPos[1], rPos[1], e); dcNow = e > 0.6 ? 512 : 0; dRun = e > 0.95; }
        else { dx = rPos[0]; dy = rPos[1]; dcNow = i === 4 ? V.lerp(512, 2558, e) : 512; dRun = true; }
        const dW = V.lerp(G.ww, G.rw, i === 2 ? e : i > 2 ? 1 : 0);
        card(g, dx, dy, dW, cardH, { col: colD, title: 'D · ' + (dRun ? 'prefill' : 'new'), sub: 'c ' + nf(dcNow) + ' · n 3,000', frac: dcNow / 3000, hi: i === 2 && walk > 0.2, t, mood: i <= 1 ? 'sleep' : 'neutral', badge: i >= 2 && (i > 2 || walk > 0.5) ? '+2,046' : null });
        if (i === 2 && e > 0.55 && e < 1) V.text(g, '✦ cache: 512', dx + dW - 70, dy + cardH + 13, { size: 11, font: 'mono', color: C.d, weight: 600 });
        if (i >= 3) V.text(g, '(empty)', G.wx + 4, G.wy + 18, { size: 11, font: 'mono', color: C.muted });

        // tray
        let fA = 0, fB = 0, fC = 0, fD = 0;
        if (i === 1) { fA = walk > 0.15 ? 1 : 0; fB = walk > 0.45 ? 1 : 0; fC = clamp((walk - 0.6) / 0.4, 0, 1); }
        if (i >= 2) { fA = fB = fC = 1; fD = i === 2 ? clamp((walk - 0.35) / 0.6, 0, 1) : 1; }
        const trayAlpha = i === 4 ? 1 - e * 0.85 : 1;
        g.globalAlpha = trayAlpha;
        tokenGrid(g, {
          x: G.tx, y: G.ty, cols: 16, rows: 8, cell, total: 4096, segs: [
            { a: 0, b: fA, col: colA }, { a: 1, b: 1 + fB, col: colB },
            { a: 2, b: 2 + 2048 * fC, col: colC }, { a: 2050, b: 2050 + 2046 * fD, col: colD },
          ],
        });
        g.globalAlpha = 1;
        if (i >= 1 && i <= 3 && fA) {
          g.strokeStyle = C.c; g.lineWidth = 2; V.rr(g, G.tx - 3, G.ty - 3, cell + 4, cell + 4, 4); g.stroke();
          V.text(g, '↑ 1st cell: A and B cost 1 token each (slivers)', G.tx, G.ty + th + 16, { size: 11, color: C.muted, maxW: G.tw });
        }
        // card to tray arrows (flow)
        if (!narrow && i === 1 && walk > 0.6) V.arrow(g, G.rx + G.rw + 4, G.ry + 2 * step + cardH / 2, G.tx - 4, G.ty + th * 0.4, { color: V.alpha(colC, 0.8), lw: 1.5, curve: 18 });
        if (!narrow && i === 2 && walk > 0.4) V.arrow(g, G.rx + G.rw + 4, G.ry + 3 * step + cardH / 2, G.tx - 4, G.ty + th * 0.8, { color: V.alpha(colD, 0.8), lw: 1.5, curve: -18 });

        if (i === 4) V.text(g, 'step done: the tray resets', G.tx + G.tw / 2, G.ty + th / 2 + 4, { size: 12, weight: 600, color: C.ink, align: 'center', maxW: G.tw });
        // batch placeholder (before execute_model)
        if (i !== 3) {
          g.globalAlpha = i === 4 ? e : 1;
          const gpuW = narrow ? 58 : 78, bx = 14, bw = w - 28 - gpuW - 16, by = G.by, bh = 22;
          g.setLineDash([4, 4]); g.strokeStyle = V.alpha(C.muted, 0.45); g.lineWidth = 1; V.rr(g, bx, by, bw, bh, 4); g.stroke(); g.setLineDash([]);
          V.text(g, 'step batch: assembled in execute_model()', bx + 8, by + 15, { size: 11, font: 'mono', color: C.muted, maxW: bw - 16 });
          V.box(g, bx + bw + 16, by - 6, gpuW, bh + 12, { r: 8, fill: V.alpha(C.a, 0.05), stroke: V.alpha(C.a, 0.35), lw: 1, label: 'GPU ×2', size: 11.5, font: 'mono', color: C.muted });
          g.globalAlpha = 1;
        }
        // flattened batch to GPU
        if (i >= 3) {
          const alpha = i === 4 ? 1 - e : Math.min(1, el / 0.6);
          g.globalAlpha = Math.max(0, alpha);
          const gpuW = narrow ? 58 : 78, bx = 14, bw = w - 28 - gpuW - 16, by = G.by, bh = 22;
          V.text(g, narrow ? 'X [4,096 × d] → one forward' : 'flattened batch X [4,096 × d] → one forward (each weight read 1×)', bx, by - 8, { size: 11, font: 'mono', color: C.muted, maxW: bw });
          const small = 12, rest = bw - 2 * small;
          const segs = [[colA, small, 'A'], [colB, small, 'B'], [colC, rest * 0.5002, narrow ? 'C' : 'C · 2,048'], [colD, rest * 0.4998, narrow ? 'D' : 'D · 2,046']];
          let x = bx;
          const marks = ['0', '1', '2', '2,050', '4,096'];
          segs.forEach((s, k) => {
            V.box(g, x, by, s[1] - 1, bh, { r: 4, fill: s[0], label: s[2], size: 11, color: ink(C) });
            V.text(g, marks[k], x, by + bh + 13, { size: 10.5, font: 'mono', color: C.muted, align: k === 0 ? 'left' : 'center' });
            x += s[1];
          });
          V.text(g, marks[4], x, by + bh + 13, { size: 10.5, font: 'mono', color: C.muted, align: 'right' });
          const gx = bx + bw + 16;
          V.box(g, gx, by - 6, gpuW, bh + 12, { r: 8, fill: V.alpha(C.a, 0.15 + (i === 3 ? 0.15 * Math.sin(t * 6) + 0.15 : 0)), stroke: C.a, lw: 1.5, label: 'GPU ×2', size: 11.5, font: 'mono' });
          if (i === 3) {
            for (let p = 0; p < 16; p++) {
              const u = ((t * 0.6 + p / 16) % 1);
              const sx = bx + (p / 16) * bw, sy = by + bh / 2;
              const px = V.lerp(sx, gx + gpuW / 2, u), py = sy - Math.sin(u * Math.PI) * 26;
              V.dot(g, px, py, 2.2, p < 1 ? colA : p < 2 ? colB : p < 9 ? colC : colD);
            }
            V.glow(g, gx + gpuW / 2, by + bh / 2, 34, C.a, 0.5);
          }
          g.globalAlpha = 1;
        }
      });
    }

    /* =====================================================================
       Figure 2 — static vs continuous batching
       ===================================================================== */
    {
      const fig = q('#fig-batching');
      autoH(fig, 380, 480, 560);
      const S = { mode: 'both', v: 0.6 };
      const U = V.rng(11);
      const us = Array.from({ length: 12 }, () => U());
      let P = null;
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Show', options: [{ v: 'both', label: 'compare' }, { v: 'static', label: 'static' }, { v: 'cont', label: 'continuous' }], value: 'both', onChange: (v) => { S.mode = v; } });
      UI.slider(ctl, { label: 'Variability', min: 0, max: 1, step: 0.05, value: S.v, fmt: (v) => nf(v * 100) + '%', onInput: (v) => { S.v = v; build(); } });
      const rU1 = UI.readout(ctl, 'Occupancy w/ queue · static', '—');
      const rU2 = UI.readout(ctl, 'Occupancy w/ queue · continuous', '—');
      const rM = UI.readout(ctl, 'Mean finish', '—');
      function build() {
        const Ls = us.map((u) => Math.max(3, Math.round(6 + (1 - S.v) * 14 + S.v * (4 + 70 * u * u * u))));
        const stat = [], bubbles = [];
        let T = 0;
        for (let b = 0; b < 3; b++) {
          const ids = [b * 4, b * 4 + 1, b * 4 + 2, b * 4 + 3];
          const mx = Math.max(...ids.map((k) => Ls[k]));
          ids.forEach((k, lane) => { stat.push({ k, lane, s: T, e: T + Ls[k] }); if (Ls[k] < mx) bubbles.push({ lane, s: T + Ls[k], e: T + mx }); });
          T += mx;
        }
        const free = [0, 0, 0, 0], cont = [];
        Ls.forEach((L, k) => { let lane = 0; for (let j = 1; j < 4; j++) if (free[j] < free[lane]) lane = j; cont.push({ k, lane, s: free[lane], e: free[lane] + L }); free[lane] += L; });
        const Tc = Math.max(...free);
        const mean = (arr) => arr.reduce((a, r) => a + r.e, 0) / arr.length;
        // occupancy while there is still a queue (until the last request enters)
        const occ = (arr) => { const Tq = Math.max(...arr.map((r) => r.s)); if (Tq <= 0) return { u: 1, Tq }; const busy = arr.reduce((a, r) => a + Math.max(0, Math.min(r.e, Tq) - r.s), 0); return { u: busy / (4 * Tq), Tq }; };
        const oS = occ(stat), oC = occ(cont);
        P = { stat, bubbles, Ts: T, cont, Tc, uS: oS.u, uC: oC.u, qS: oS.Tq, qC: oC.Tq, mS: mean(stat), mC: mean(cont) };
        rU1.set(nf(P.uS * 100) + '%'); rU2.set(nf(P.uC * 100) + '%'); rM.set(nf(P.mS) + ' → ' + nf(P.mC) + ' steps');
      }
      build();
      V.fig(fig, (g, w, h, t) => {
        if (w < 60 || h < 60) return;
        const C = V.C();
        const narrow = w < 560;
        const panels = S.mode === 'both' ? ['static', 'cont'] : [S.mode];
        const Tmax = S.mode === 'both' ? Math.max(P.Ts, P.Tc) : S.mode === 'static' ? P.Ts : P.Tc;
        const x0 = narrow ? 34 : 64, x1 = w - 14;
        const X = (tt) => x0 + (tt / Tmax) * (x1 - x0);
        const per = 10;
        const ph = Math.min(1, (t % per) / (per - 2.2)) * Tmax;
        const top = 10, bottom = h - 30, gapP = 26;
        const pH = (bottom - top - gapP * (panels.length - 1)) / panels.length;
        panels.forEach((pn, pi) => {
          const y0 = top + pi * (pH + gapP);
          const title = pn === 'static' ? (narrow ? 'Static (bus)' : 'Static (bus): the batch leaves full and waits for the slowest') : (narrow ? 'Continuous (conveyor)' : 'Continuous (conveyor): free slot → next request gets in');
          V.text(g, title, x0, y0 + 12, { size: 12, weight: 600, color: pn === 'static' ? C.d : C.a, maxW: x1 - x0 });
          const ly = y0 + 22, lh = Math.max(8, (pH - 26) / 4);
          for (let l = 0; l < 4; l++) {
            V.text(g, (narrow ? 's' : 'slot ') + (l + 1), x0 - 8, ly + l * lh + lh / 2 + 4, { size: 11, font: 'mono', color: C.muted, align: 'right' });
            V.box(g, x0, ly + l * lh + 2, x1 - x0, lh - 4, { r: 5, fill: V.alpha(C.muted, 0.06) });
          }
          const bars = pn === 'static' ? P.stat : P.cont;
          if (pn === 'static') P.bubbles.forEach((b) => {
            if (b.s >= ph) return;
            const xa = X(b.s), xb = X(Math.min(b.e, ph)), yy = ly + b.lane * lh + 3;
            g.fillStyle = V.alpha(C.red || '#FB2C36', 0.07); g.fillRect(xa, yy, xb - xa, lh - 6);
            hatch(g, xa, yy, xb - xa, lh - 6, V.alpha(C.muted, 0.55), 6);
          });
          bars.forEach((b) => {
            if (b.s >= ph) return;
            const xa = X(b.s), xe = X(Math.min(b.e, ph)), yy = ly + b.lane * lh + 3, hh = lh - 6;
            const col = C.series[b.k % 6];
            V.box(g, xa + 0.5, yy, Math.max(2, xe - xa - 1), hh, { r: 4, fill: V.alpha(col, 0.75) });
            if (xe - xa > 18) V.text(g, String(b.k + 1), xa + 6, yy + hh / 2 + 4, { size: 11, font: 'mono', weight: 600, color: ink(C) });
            if (ph < b.e) critter(g, xe, yy + hh / 2, Math.min(9, hh * 0.42), col, 'happy', t);
          });
          const Tq = pn === 'static' ? P.qS : P.qC;
          if (ph >= Tq && Tq > 0) { V.line(g, X(Tq), ly + 4 * lh - 2, X(Tq), ly + 4 * lh + 4, { color: C.c, lw: 2 }); V.text(g, 'queue empty', X(Tq) + 3, ly + 4 * lh + 4, { size: 10.5, font: 'mono', color: C.c }); }
          const Tp = pn === 'static' ? P.Ts : P.Tc;
          if (ph >= Tp - 0.01) {
            V.line(g, X(Tp), ly - 2, X(Tp), ly + 4 * lh, { color: V.alpha(C.ink, 0.5), dash: [3, 3] });
            V.text(g, 'end: ' + Tp + ' steps', Math.min(X(Tp) + 4, x1 - 2), ly - 4, { size: 11, font: 'mono', color: C.ink, align: X(Tp) > x1 - 90 ? 'right' : 'left' });
          }
        });
        V.line(g, X(ph), top + 16, X(ph), bottom, { color: V.alpha(C.c, 0.6), lw: 1.2 });
        for (let s = 0; s <= Tmax - 14; s += 20) V.text(g, String(s), X(s), h - 12, { size: 10.5, font: 'mono', color: C.muted, align: 'center' });
        V.text(g, 'steps →', x1, h - 12, { size: 10.5, font: 'mono', color: C.muted, align: 'right' });
      });
    }

    /* =====================================================================
       Figure 4 — the step tray (verify rows, budget, captures)
       ===================================================================== */
    {
      const fig = q('#fig-tray');
      autoH(fig, 430, 560, 600);
      const PRE = { ds: { S: 6, k: 6, B: 8192, thr: 1024 }, glm: { S: 4, k: 7, B: 7168, thr: 0 }, qwen: { S: 8, k: 3, B: 8192, thr: 0 } };
      const S = { S: 6, k: 6, B: 8192, thr: 1024, pref: true, pad: true, rule: 'ds' };
      let D = null;
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Preset', options: [{ v: 'ds', label: 'DeepSeek' }, { v: 'glm', label: 'GLM' }, { v: 'qwen', label: 'Qwen' }], value: 'ds', onChange: (v) => { const p = PRE[v]; S.rule = v; tPad.el.style.display = v === 'ds' ? '' : 'none'; sS.set(p.S); sK.set(p.k); sB.set(p.B); sT.set(p.thr); } });
      const sS = UI.slider(ctl, { label: 'max_num_seqs', min: 1, max: 32, step: 1, value: S.S, onInput: (v) => { S.S = v; upd(); } });
      const sK = UI.slider(ctl, { label: 'k', min: 0, max: 7, step: 1, value: S.k, onInput: (v) => { S.k = v; upd(); } });
      const sB = UI.slider(ctl, { label: 'MNBT', min: 1024, max: 16384, step: 1024, value: S.B, fmt: (v) => nf(v), onInput: (v) => { S.B = v; upd(); } });
      const sT = UI.slider(ctl, { label: 'Threshold', min: 0, max: 8192, step: 256, value: S.thr, fmt: (v) => (v ? nf(v) : 'off'), onInput: (v) => { S.thr = v; upd(); } });
      UI.toggle(ctl, { label: 'prefill waiting', value: true, onChange: (v) => { S.pref = v; upd(); } });
      const tPad = UI.toggle(ctl, { label: 'DS: capture at a multiple of 8', value: true, onChange: (v) => { S.pad = v; upd(); } });
      const rM = UI.readout(ctl, 'Verification rows', '—');
      const rC = UI.readout(ctl, 'Largest graph', '—');
      const rP = UI.readout(ctl, 'Prefill chunk', '—');
      /* vLLM stock grid: 1, 2, 4 and multiples of 8 */
      const stockGrid = (cap) => { const o = [1, 2, 4]; for (let x = 8; x <= cap; x += 8) o.push(x); return o.filter((x) => x <= cap); };
      /* capture sizes according to each recipe rule */
      function captures(M, q1) {
        if (S.rule === 'glm') return { sizes: [1, 2, 4, 8, 16, 24, 32], teto: 32, note: 'fixed GLM launcher list (adaptive k off)', short: 'fixed GLM list' };
        if (S.rule === 'qwen') {
          const teto = Math.min(512, 2 * M);
          return { sizes: stockGrid(teto), teto, note: 'vLLM stock grid up to 2·S·(1+k) = ' + teto + ' · FULL_DECODE_ONLY (Qwen)', short: 'stock up to 2·S·(1+k) = ' + teto };
        }
        // DeepSeek: cap = 8⌈S(1+k)/8⌉; stock grid up to the cap, rounded to multiples of 1+k
        const capReq = S.pad ? Math.ceil(M / 8) * 8 : M;
        const teto = stockGrid(capReq).pop() || 1;
        const sizes = Array.from(new Set(stockGrid(teto).map((x) => Math.ceil(x / q1) * q1))).filter((x) => x <= teto).sort((a, b) => a - b);
        return { sizes, teto, note: 'multiples of 1+k up to ' + (S.pad ? '8⌈S(1+k)/8⌉' : 'S(1+k) truncated to the grid') + ' = ' + teto + ' (DeepSeek recipe rule)', short: 'DS rule · cap ' + teto };
      }
      function upd() {
        const M = S.S * (S.k + 1);
        const room = S.pref ? Math.max(0, S.B - M) : 0;
        const chunk = S.thr > 0 ? Math.min(S.thr, room) : room;
        const q1 = S.k + 1;
        const cp = captures(M, q1);
        const fit = cp.sizes.filter((x) => x >= M && (q1 === 1 || x % q1 === 0));
        const use = fit.length ? fit[0] : null;
        const capMax = cp.sizes[cp.sizes.length - 1];
        D = { M, chunk, free: Math.max(0, S.B - M - chunk), capT: capMax, sizes: cp.sizes, note: cp.note, short: cp.short, use, full: use != null };
        rM.set(nf(M)); rC.set(nf(capMax) + (D.full ? ' ✓' : ' ✗')); rP.set(nf(chunk));
      }
      upd();
      V.fig(fig, (g, w, h, t) => {
        if (w < 60 || h < 60) return;
        const C = V.C();
        const narrow = w < 600;
        const L = {};
        if (!narrow) { L.mx = 16; L.my = 44; L.mw = w * 0.36; L.mh = h - 150; L.tx = L.mx + L.mw + 34; L.ty = 44; L.tw = w - L.tx - 16; L.th = h - 150; L.sy = h - 56; }
        else { L.mx = 12; L.my = 40; L.mw = w - 24; L.mh = 190; L.tx = 12; L.ty = 40 + 190 + 56; L.tw = w - 24; L.th = 150; L.sy = h - 50; }
        const M = D.M, K1 = S.k + 1;
        // verification matrix
        V.text(g, 'verification matrix · ' + S.S + ' × ' + K1 + ' = ' + nf(M), L.mx, L.my - 16, { size: 11.5, font: 'mono', color: C.a, weight: 600, maxW: L.mw });
        const cs = Math.max(4, Math.min(24, (L.mw - 34) / K1, L.mh / S.S));
        const withCrit = cs >= 13;
        const ox = L.mx + (withCrit ? 26 : 4);
        for (let r = 0; r < S.S; r++) {
          const yy = L.my + r * cs;
          if (withCrit) critter(g, L.mx + 11, yy + cs / 2, Math.min(8, cs * 0.42), C.series[r % 6], 'neutral', t);
          for (let j = 0; j < K1; j++) {
            const xx = ox + j * cs;
            const col = j === 0 ? C.a : V.alpha(C.b, 0.35 + 0.3 * (0.5 + 0.5 * Math.sin(t * 2.5 - j * 0.8 + r * 0.35)));
            g.fillStyle = col; V.rr(g, xx, yy, cs - 2, cs - 2, Math.min(4, cs / 5)); g.fill();
          }
        }
        const legY = L.my + S.S * cs + 16;
        if (legY < L.my + L.mh + 30) {
          V.dot(g, L.mx + 6, legY - 4, 4, C.a); V.text(g, 'anchor', L.mx + 14, legY, { size: 11, color: C.muted });
          V.dot(g, L.mx + 76, legY - 4, 4, C.b); V.text(g, 'drafts to verify (k = ' + S.k + ')', L.mx + 84, legY, { size: 11, color: C.muted, maxW: L.mw - 84 });
        }
        // tray
        const cell = Math.min(L.tw / 16, L.th / 8);
        V.text(g, 'tray · MNBT = ' + nf(S.B) + ' (1 cell = ' + nf(S.B / 128) + ')', L.tx, L.ty - 16, { size: 11.5, font: 'mono', color: C.b, weight: 600, maxW: L.tw });
        tokenGrid(g, { x: L.tx, y: L.ty, cols: 16, rows: 8, cell, total: S.B, segs: [{ a: 0, b: M, col: C.a }, { a: M, b: M + D.chunk, col: C.b }] });
        const ly = L.ty + cell * 8 + 16;
        const items = [[C.a, 'decode ' + nf(M)], [C.b, 'prefill ' + nf(D.chunk)], [V.alpha(C.muted, 0.35), 'free ' + nf(D.free)]];
        let lx = L.tx;
        items.forEach(([c, s]) => { V.box(g, lx, ly - 9, 10, 10, { r: 3, fill: c }); V.text(g, s, lx + 14, ly, { size: 11, font: 'mono', color: C.ink }); g.font = V.font(11, 'mono'); lx += g.measureText(s).width + 30; });
        if (D.chunk > 0) V.text(g, D.chunk < S.B - M ? 'chunk cut by the threshold → mixed step' : 'chunk = what is left of the budget → mixed step', L.tx, ly + 16, { size: 11, color: C.muted, maxW: L.tw });
        else V.text(g, 'decode only: uniform step (FULL graph candidate)', L.tx, ly + 16, { size: 11, color: C.muted, maxW: L.tw });
        // capture strip (rule of the recipe chosen in the preset)
        const sx0 = 20, sx1 = w - 20, sy = L.sy;
        const maxV = Math.max(D.capT, M) * 1.06 + 1;
        const SX = (v) => sx0 + (v / maxV) * (sx1 - sx0);
        V.text(g, 'captured graphs · ' + (narrow ? D.short : D.note), sx0, sy - 18, { size: 11, color: C.muted, maxW: sx1 - sx0 });
        V.line(g, sx0, sy, sx1, sy, { color: V.alpha(C.muted, 0.5) });
        D.sizes.forEach((s) => V.dot(g, SX(s), sy, 3.4, D.use === s ? C.c : C.a));
        V.line(g, SX(D.capT), sy - 8, SX(D.capT), sy + 8, { color: C.d, lw: 2 });
        V.text(g, 'max ' + D.capT, SX(D.capT), sy + 20, { size: 11, font: 'mono', color: C.d, align: SX(D.capT) > sx1 - 40 ? 'right' : 'center' });
        const mc = D.full ? C.a : (C.red || '#FB2C36');
        g.beginPath(); g.moveTo(SX(M), sy - 6); g.lineTo(SX(M) - 6, sy - 15); g.lineTo(SX(M) + 6, sy - 15); g.closePath(); g.fillStyle = mc; g.fill();
        const lbl = 'full step = ' + M + (D.full ? (D.use === M ? ' · FULL ✓' : ' → graph ' + D.use + ' · FULL ✓') : ' · eager ✗');
        g.font = V.font(11, 'mono', 600); const lw2 = g.measureText(lbl).width / 2 + 2;
        V.text(g, lbl, clamp(SX(M), sx0 + lw2, sx1 - lw2), sy + 36, { size: 11, font: 'mono', color: mc, align: 'center', weight: 600 });
        if (M > 64 && S.rule === 'ds') V.text(g, narrow ? '> 64 rows: no stability proof (#141)' : '> 64 rows: no stability proof on DeepSeek (#141)', sx1, sy - 32, { size: 11, color: C.d, align: 'right' });
      });
    }

    /* =====================================================================
       Figure 5 — prompt slicer: TTFT vs pause
       ===================================================================== */
    {
      const fig = q('#fig-chunk');
      autoH(fig, 400, 520, 600);
      const CH = [256, 512, 768, 1024, 1536, 2048, 3584, 4096, 7168, 8192, 16384];
      const S = { ci: 3, P: 32768, a: 0.3 };
      const b = 0.00068;
      const ctl = UI.controls(fig);
      UI.slider(ctl, { label: 'Chunk', min: 0, max: CH.length - 1, step: 1, value: S.ci, fmt: (v) => nf(CH[v]), onInput: (v) => { S.ci = v; upd(); } });
      UI.seg(ctl, { label: 'Prompt', options: [{ v: 8192, label: '8K' }, { v: 32768, label: '32K' }, { v: 131072, label: '128K' }], value: 32768, onChange: (v) => { S.P = v; upd(); } });
      UI.slider(ctl, { label: 'Fixed cost a', min: 0, max: 0.6, step: 0.05, value: S.a, fmt: (v) => nf(v, 2) + ' s', onInput: (v) => { S.a = v; upd(); } });
      const rN = UI.readout(ctl, 'Chunks', '—'), rT = UI.readout(ctl, 'TTFT ≈', '—'), rG = UI.readout(ctl, 'Max pause ≈', '—');
      const ttft = (c) => Math.ceil(S.P / c) * S.a + b * S.P;
      const gap = (c) => S.a + b * Math.min(c, S.P);
      function upd() { const c = CH[S.ci]; rN.set(nf(Math.ceil(S.P / c))); rT.set(secs(ttft(c))); rG.set(secs(gap(c))); }
      upd();
      V.fig(fig, (g, w, h, t) => {
        if (w < 60 || h < 60) return;
        const C = V.C();
        const c = CH[S.ci];
        const x0 = 58, x1 = w - 16;
        // timeline
        const T = ttft(c), W = Math.min(T, 24);
        const X = (s) => x0 + (s / W) * (x1 - x0);
        const yD = 34, yP = 64;
        V.text(g, 'decode', x0 - 8, yD + 4, { size: 11, font: 'mono', color: C.a, align: 'right' });
        V.text(g, 'prefill', x0 - 8, yP + 12, { size: 11, font: 'mono', color: C.b, align: 'right' });
        V.text(g, T > W ? 'first ' + nf(W, 0) + ' s of ' + secs(T) : 'full prefill: ' + secs(T), x1, 16, { size: 11, font: 'mono', color: C.muted, align: 'right' });
        const phase = (t % 7) / 6;
        const reveal = Math.min(1, phase) * W;
        let s = 0, done = 0, k = 0;
        while (s < W && done < S.P && k < 400) {
          const n = Math.min(c, S.P - done), d = S.a + b * n;
          const e = Math.min(s + d, W);
          if (s < reveal) {
            const ev = Math.min(e, reveal);
            V.box(g, X(s) + 0.5, yP, Math.max(1, X(ev) - X(s) - 1.5), 20, { r: 3, fill: V.alpha(C.b, 0.8) });
            if (S.a > 0 && X(s + S.a) - X(s) > 3) { g.fillStyle = V.alpha(C.d, 0.55); g.fillRect(X(s) + 0.5, yP + 16, Math.min(X(ev), X(s + S.a)) - X(s), 4); }
            if (s + d <= reveal) { V.dot(g, X(s + d), yD, 4, C.a); }
          }
          s += d; done += n; k++;
        }
        V.line(g, x0, yD, x1, yD, { color: V.alpha(C.a, 0.25) });
        V.line(g, X(reveal), yD - 12, X(reveal), yP + 26, { color: V.alpha(C.c, 0.7), lw: 1.2 });
        if (reveal < W) critter(g, X(reveal), yD, 8, C.a, gap(c) > 2 ? 'sad' : 'happy', t);
        V.text(g, '● user token · amber bar = fixed cost a', x0, yP + 40, { size: 11, color: C.muted, maxW: x1 - x0 });
        // log-log chart
        const py0 = w < 560 ? 150 : Math.max(128, h * 0.36), py1 = h - 34;
        const lx0 = Math.log2(256), lx1 = Math.log2(16384);
        const PX = (cc) => x0 + ((Math.log2(cc) - lx0) / (lx1 - lx0)) * (x1 - x0);
        const ly0 = -1, ly1 = 3;
        const PY = (sv) => py1 - ((Math.log10(Math.max(0.1, sv)) - ly0) / (ly1 - ly0)) * (py1 - py0);
        [0.1, 1, 10, 100, 1000].forEach((v) => { V.line(g, x0, PY(v), x1, PY(v), { color: C.grid, lw: 1 }); V.text(g, v >= 1 ? nf(v) + ' s' : '0.1 s', x0 - 6, PY(v) + 4, { size: 10.5, font: 'mono', color: C.muted, align: 'right' }); });
        const narrow = w < 560;
        const lab = narrow ? [256, 1024, 3584, 16384] : [256, 512, 1024, 2048, 3584, 7168, 16384];
        CH.forEach((cc) => { V.line(g, PX(cc), py1, PX(cc), py1 + 4, { color: C.muted }); if (lab.includes(cc)) V.text(g, cc >= 1024 ? nf(cc / 1024, cc % 1024 ? 1 : 0) + 'K' : String(cc), PX(cc), py1 + 16, { size: 10.5, font: 'mono', color: C.muted, align: 'center' }); });
        const curve = (fn, col) => { g.beginPath(); for (let j = 0; j <= 60; j++) { const cc = Math.pow(2, lx0 + (j / 60) * (lx1 - lx0)); const xx = PX(cc), yy = PY(fn(cc)); if (j) g.lineTo(xx, yy); else g.moveTo(xx, yy); } g.strokeStyle = col; g.lineWidth = 2.4; g.stroke(); };
        curve(ttft, C.b); curve(gap, C.a);
        V.line(g, PX(c), py0, PX(c), py1, { color: V.alpha(C.c, 0.7), dash: [4, 4] });
        V.dot(g, PX(c), PY(ttft(c)), 5, C.b); V.dot(g, PX(c), PY(gap(c)), 5, C.a);
        const right = PX(c) > x1 - 150;
        V.text(g, 'TTFT ' + secs(ttft(c)), PX(c) + (right ? -8 : 8), PY(ttft(c)) - 8, { size: 11.5, font: 'mono', weight: 600, color: C.b, align: right ? 'right' : 'left' });
        V.text(g, 'pause ' + secs(gap(c)), PX(c) + (right ? -8 : 8), PY(gap(c)) + 16, { size: 11.5, font: 'mono', weight: 600, color: C.a, align: right ? 'right' : 'left' });
        V.text(g, 'chunk size (tokens, log scale)', x1, h - 2, { size: 10.5, color: C.muted, align: 'right' });
      });
    }

    /* =====================================================================
       Figure 6 — chunks aligned to recurrent-state checkpoints
       ===================================================================== */
    {
      const fig = q('#fig-mamba');
      autoH(fig, 266, 320, 560);
      const Bk = 3584, P = 13000, LAST = P - (P % Bk);
      const S = { pos: 3584, g: 256, mode: 'naive' };
      function align(pos, g, mode) {
        g = Math.min(g, P - pos);
        const after = pos + g;
        if (pos >= LAST) return g;
        if (after > LAST) return LAST - pos;
        if (after === LAST) return g;
        if (mode === 'naive') return Math.floor(g / Bk) * Bk;
        const nb = (Math.floor(pos / Bk) + 1) * Bk;
        if (g < Bk) return Math.min(g, nb - pos);
        return Math.floor(after / Bk) * Bk - pos;
      }
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Rule', options: [{ v: 'naive', label: 'round down (bug)' }, { v: 'v2', label: 'v2 (GLM overlay)' }], value: 'naive', onChange: (v) => { S.mode = v; upd(); } });
      UI.seg(ctl, { label: 'Offer', options: [256, 1024, 2048, 7168].map((v) => ({ v, label: nf(v) })), value: 256, onChange: (v) => { S.g = v; upd(); } });
      UI.slider(ctl, { label: 'Position', min: 0, max: 12800, step: 128, value: S.pos, fmt: (v) => nf(v), onInput: (v) => { S.pos = v; upd(); } });
      const rO = UI.readout(ctl, 'Granted', '—'), rK = UI.readout(ctl, 'Checkpoint', '—');
      let R = null;
      function upd() { const gg = align(S.pos, S.g, S.mode); R = { gg, end: S.pos + gg, ck: gg > 0 && (S.pos + gg) % Bk === 0 && S.pos + gg <= LAST }; rO.set(nf(gg) + ' of ' + nf(Math.min(S.g, P - S.pos))); rK.set(R.ck ? 'saved at ' + nf(R.end) : '—'); }
      upd();
      V.fig(fig, (g, w, h, t) => {
        if (w < 60 || h < 60) return;
        const C = V.C();
        const x0 = 22, x1 = w - 22, X = (v) => x0 + (v / P) * (x1 - x0);
        const y = h * 0.5 - 4, bh = 30;
        V.text(g, '13,000-token prompt · block = 3,584 (scheduler LCM)', x0, 22, { size: 11.5, font: 'mono', color: C.muted, maxW: x1 - x0 });
        V.box(g, x0, y, x1 - x0, bh, { r: 6, fill: V.alpha(C.muted, 0.1) });
        hatch(g, X(LAST), y, X(P) - X(LAST), bh, V.alpha(C.muted, 0.4), 7);
        V.box(g, x0, y, Math.max(0, X(S.pos) - x0), bh, { r: 6, fill: V.alpha(C.a, 0.5) });
        g.setLineDash([5, 4]); g.strokeStyle = C.b; g.lineWidth = 1.6; V.rr(g, X(S.pos), y - 6, X(Math.min(P, S.pos + S.g)) - X(S.pos), bh + 12, 6); g.stroke(); g.setLineDash([]);
        if (R.gg > 0) V.box(g, X(S.pos), y, X(R.end) - X(S.pos), bh, { r: 4, fill: V.alpha(C.b, 0.85) });
        const nrw = w < 560;
        [0, Bk, 2 * Bk, 3 * Bk, P].forEach((v) => {
          V.line(g, X(v), y - 4, X(v), y + bh + 4, { color: V.alpha(C.ink, 0.5) });
          V.text(g, nf(v), X(v), y + bh + (nrw && v === P ? 32 : 18), { size: 11, font: 'mono', color: C.muted, align: v === 0 ? 'left' : v === P ? 'right' : 'center' });
        });
        const u = (t % 2.6) / 2.2;
        const arrived = u >= 1;
        [Bk, 2 * Bk, 3 * Bk].forEach((v) => {
          const saved = v <= S.pos, now = R.ck && R.end === v && arrived;
          const cx = X(v), cy = y - 26;
          if (now) V.glow(g, cx, cy, 26, C.d, 0.9);
          V.box(g, cx - 10, cy - 7, 20, 14, { r: 3, fill: saved || now ? C.d : 'transparent', stroke: C.d, lw: 1.4 });
          g.beginPath(); g.arc(cx, cy, 3.6, 0, Math.PI * 2); g.strokeStyle = saved || now ? ink(C) : C.d; g.lineWidth = 1.3; g.stroke();
        });
        V.text(g, 'no checkpoint', nrw ? X(P) : (X(LAST) + X(P)) / 2, y + bh + (nrw ? 46 : 34), { size: 11, color: C.muted, align: nrw ? 'right' : 'center' });
        let cx, mood = 'neutral', col = C.b;
        if (R.gg === 0) { cx = X(S.pos) + Math.sin(t * 18) * 3; mood = 'sad'; col = C.red || '#FB2C36'; }
        else cx = V.lerp(X(S.pos), X(R.end), V.ease.inOut(Math.min(1, u)));
        critter(g, cx, y + bh / 2, 11, col, R.ck && arrived ? 'happy' : mood, t);
        const msg = R.gg === 0 ? '✗ offer < 1 block rounded down to 0: the request does not advance this step'
          : R.ck ? '✓ the chunk ends on a boundary: recurrent state saved at ' + nf(R.end)
            : '→ sub-block progress: the state is not saved now; the next chunk stops at the boundary';
        wrapText(g, msg, x0, nrw ? h - 34 : h - 16, x1 - x0, 17, { size: 12, weight: 600, color: R.gg === 0 ? (C.red || '#FB2C36') : R.ck ? C.d : C.ink });
      });
    }

    /* =====================================================================
       Figure 7 — one scheduler step: who is left with nothing?
       ===================================================================== */
    {
      const fig = q('#fig-starve');
      autoH(fig, 392, 474, 620);
      const SC = {
        1: { run: [{ id: 'A', kind: 'dec' }, { id: 'B', kind: 'dec' }], wait: [{ id: 'P1', P: 32768 }, { id: 'P2', P: 24576 }, { id: 'P3', P: 16384 }, { id: 'P4', P: 8192 }] },
        2: { run: [{ id: 'P1', kind: 'pre', rem: 20000 }, { id: 'P2', kind: 'pre', rem: 12000 }, { id: 'C', kind: 'dec' }, { id: 'D', kind: 'dec' }, { id: 'E', kind: 'dec' }], wait: [{ id: 'P3', P: 16384 }] },
      };
      const K1 = 7, BUD = 8192, SLOTS = 6;
      const S = { sc: 2, thr: 0, cap: 0, floor: false, t0: 0, tNow: 0 };
      let R = null;
      function plan() {
        const sc = SC[S.sc];
        let budget = BUD;
        const vis = [];
        sc.run.forEach((r, i) => {
          const v = { id: r.id, kind: r.kind, rem: r.rem || 0, where: 'run' };
          if (budget <= 0) { v.got = 0; v.skip = r.kind === 'dec'; v.note = r.kind === 'dec' ? '0 · skipped' : '0 · no budget'; vis.push(v); return; }
          let n = r.kind === 'dec' ? K1 : r.rem, cut = false;
          if (r.kind === 'pre' && S.thr > 0 && n > S.thr) n = S.thr;
          if (r.kind === 'pre' && S.floor) {
            const reserve = sc.run.slice(i + 1).filter((x) => x.kind === 'dec').length * K1;
            const lim = Math.max(0, budget - reserve);
            if (n > lim) { n = lim; cut = true; }
          }
          n = Math.min(n, budget);
          v.got = n; budget -= n;
          v.note = n === 0 ? (cut ? '0 · deferred (#43)' : '0') : '+' + nf(n) + (cut ? ' (#43)' : '');
          vis.push(v);
        });
        let inflight = vis.filter((v) => v.kind === 'pre' && v.rem - v.got > 0).length;
        let nRun = sc.run.length, capHit = false;
        sc.wait.forEach((wq) => {
          const v = { id: wq.id, kind: 'new', P: wq.P, where: 'wait', got: 0 };
          if (capHit || (S.cap > 0 && inflight >= S.cap && budget > 0 && nRun < SLOTS)) { capHit = true; v.note = 'queue · #27 cap'; v.block = 'cap'; }
          else if (nRun >= SLOTS) { v.note = 'queue · no slot'; v.block = 'slot'; }
          else if (budget <= 0) { v.note = 'queue · no budget'; v.block = 'budget'; }
          else {
            let n = wq.P;
            if (S.thr > 0 && n > S.thr) n = S.thr;
            n = Math.min(n, budget);
            v.got = n; budget -= n; nRun++;
            v.note = 'admitted +' + nf(n);
            if (n < wq.P) inflight++;
          }
          vis.push(v);
        });
        const Ptok = vis.filter((v) => v.kind !== 'dec').reduce((a, v) => a + v.got, 0);
        const Dn = vis.filter((v) => v.kind === 'dec' && v.got > 0).length;
        const skips = vis.filter((v) => v.skip).length;
        const dt = 0.075 + 0.003 * Math.max(0, Dn - 1) + (Ptok > 0 ? 0.04 + 0.00055 * Ptok : 0);
        R = { vis, Ptok, Dn, skips, dt, left: budget };
        rS.set(String(skips)); rP.set(nf(Ptok)); rD.set(secs(dt)); rI.set(skips ? '≥ 2 steps' : secs(dt));
        S.t0 = S.tNow;
      }
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Scenario', options: [{ v: 1, label: '1 · cold burst' }, { v: 2, label: '2 · decodes behind' }], value: 2, onChange: (v) => { S.sc = v; plan(); } });
      UI.seg(ctl, { label: 'Threshold', options: [{ v: 0, label: '0 (off)' }, { v: 1024, label: '1024' }, { v: 4096, label: '4096' }], value: 0, onChange: (v) => { S.thr = v; plan(); } });
      UI.seg(ctl, { label: 'In-flight cap', options: [{ v: 0, label: 'stock' }, { v: 1, label: '1' }, { v: 2, label: '2' }], value: 0, onChange: (v) => { S.cap = v; plan(); } });
      UI.toggle(ctl, { label: 'decode floor (#43)', value: false, onChange: (v) => { S.floor = v; plan(); } });
      const rS = UI.readout(ctl, 'Skipped decodes', '—'), rP = UI.readout(ctl, 'Prefill in step', '—'), rD = UI.readout(ctl, 'Step duration', '—'), rI = UI.readout(ctl, 'Decode ITL', '—');
      plan();
      V.fig(fig, (g, w, h, t) => {
        if (w < 60 || h < 60) return;
        S.tNow = t;
        const C = V.C();
        const narrow = w < 620;
        const vis = R.vis;
        const tv = 0.55, hold = 3.2, per = vis.length * tv + hold;
        const ph = Math.max(0, t - S.t0) % per;
        const cur = Math.min(vis.length, Math.floor(ph / tv));
        const within = (ph % tv) / tv;
        // tray
        const tx = 16, tw = w - 32, ty = 30, tH = 26;
        V.text(g, 'step tray · 8,192 tokens · k = 6 (7 per decode)', tx, ty - 10, { size: 11, font: 'mono', color: C.muted, maxW: tw });
        V.box(g, tx, ty, tw, tH, { r: 6, fill: V.alpha(C.muted, 0.1), stroke: V.alpha(C.muted, 0.3), lw: 1 });
        let acc = 0;
        vis.forEach((v, k) => {
          if (k > cur || !v.got) return;
          const frac = k === cur ? V.ease.out(within) : 1;
          const wv = Math.max(3, (v.got / BUD) * tw) * frac;
          const col = v.kind === 'dec' ? C.a : v.kind === 'pre' ? C.b : C.c;
          g.fillStyle = col; g.fillRect(tx + acc, ty + 2, Math.max(0, wv - 1), tH - 4);
          if (wv > 40) V.text(g, v.id, tx + acc + 5, ty + tH / 2 + 4, { size: 11, font: 'mono', weight: 600, color: ink(C) });
          acc += wv;
        });
        for (let k = 1; k < 64; k++) { const xx = tx + (k / 64) * tw; g.fillStyle = V.alpha(C.bg || '#000', 0.25); g.fillRect(xx, ty + 2, 1, tH - 4); }
        // cards
        const runs = vis.filter((v) => v.where === 'run'), waits = vis.filter((v) => v.where === 'wait');
        const cols = narrow ? 3 : Math.max(4, Math.min(6, Math.max(runs.length, waits.length))), gap = 10;
        const cw = Math.min(160, (w - 32 - gap * (cols - 1)) / cols), ch = narrow ? 78 : 84;
        const rowsR = Math.ceil(runs.length / cols), rowsW = Math.ceil(waits.length / cols);
        /* labels above the “hand” band (triangle at y−14…y−4) so they do not overlap */
        const yR = ty + tH + 52, yW = yR + rowsR * (ch + gap) + 40;
        V.text(g, 'RUNNING (admission order) →', 16, yR - 22, { size: 11, font: 'mono', color: C.a, weight: 600 });
        V.text(g, 'WAITING →', 16, yW - 22, { size: 11, font: 'mono', color: C.d, weight: 600 });
        const pos = (arrIdx, isWait) => { const r = Math.floor(arrIdx / cols), c = arrIdx % cols; return [16 + c * (cw + gap), (isWait ? yW : yR) + r * (ch + gap)]; };
        vis.forEach((v, k) => {
          const isW = v.where === 'wait';
          const [x, y] = pos(isW ? waits.indexOf(v) : runs.indexOf(v), isW);
          const visited = k < cur || (k === cur && within > 0.45);
          const active = k === cur && ph < vis.length * tv;
          const baseCol = v.kind === 'dec' ? C.a : v.kind === 'pre' ? C.b : C.c;
          let col = baseCol, mood = 'neutral';
          if (visited) {
            if (v.skip) { col = C.red || '#FB2C36'; mood = 'sad'; }
            else if (v.kind === 'dec') mood = 'happy';
            else if (v.where === 'wait' && !v.got) { col = C.d; mood = 'sleep'; }
          } else if (isW) mood = 'sleep';
          V.box(g, x, y, cw, ch, { r: 10, fill: V.alpha(col, active ? 0.24 : 0.1), stroke: V.alpha(col, active ? 1 : 0.45), lw: active ? 2.2 : 1 });
          critter(g, x + 18, y + 20, 11, col, mood, t);
          V.text(g, v.id, x + 36, y + 18, { size: 13, weight: 700, color: C.ink });
          const sub = v.kind === 'dec' ? 'decode' : v.kind === 'pre' ? 'prefill ' + nf(v.rem / 1000) + 'K' : 'new ' + nf(v.P / 1024) + 'K';
          V.text(g, sub, x + 36, y + 32, { size: 11, font: 'mono', color: C.muted, maxW: cw - 40 });
          if (visited) {
            const good = v.got > 0;
            const bc = v.skip ? (C.red || '#FB2C36') : good ? baseCol : C.d;
            V.box(g, x + 6, y + ch - 30, cw - 12, 22, { r: 7, fill: V.alpha(bc, good ? 0.85 : 0.22), stroke: bc, lw: 1 });
            V.text(g, v.note, x + cw / 2, y + ch - 15, { size: 11, font: 'mono', weight: 600, color: good ? ink(C) : C.ink, align: 'center', maxW: cw - 16 });
          }
          if (active) {
            g.beginPath(); g.moveTo(x + cw / 2, y - 4); g.lineTo(x + cw / 2 - 7, y - 14); g.lineTo(x + cw / 2 + 7, y - 14); g.closePath(); g.fillStyle = C.c; g.fill();
            if (v.got && within < 0.9) { const p0 = [x + cw / 2, y], p1 = [tx + Math.min(acc, tw) - 4, ty + tH]; const pp = V.along(p0[0], p0[1], p1[0], p1[1], V.ease.inOut(within)); V.dot(g, pp[0], pp[1] - Math.sin(within * Math.PI) * 20, 5, baseCol); }
          }
        });
        const yEnd = yW + rowsW * (ch + gap) + 16;
        if (ph >= vis.length * tv) {
          const msg = R.skips ? '✗ ' + R.skips + ' decode(s) skipped — stalled without preemption · step ≈ ' + secs(R.dt)
            : '✓ all decodes served · step ≈ ' + secs(R.dt) + (R.dt > 2 ? ' (long pause for whoever is chatting)' : '');
          const mc2 = R.skips ? (C.red || '#FB2C36') : R.dt > 2 ? C.d : C.a;
          if (!narrow) V.text(g, msg, 16, Math.min(h - 12, yEnd + 6), { size: 12.5, weight: 600, color: mc2, maxW: w - 32 });
          else {
            V.text(g, R.skips ? '✗ ' + R.skips + ' decode(s) skipped, no preemption' : '✓ all decodes served', 16, Math.min(h - 30, yEnd + 4), { size: 12.5, weight: 600, color: mc2, maxW: w - 32 });
            V.text(g, 'step ≈ ' + secs(R.dt) + (R.dt > 2 && !R.skips ? ' · long pause for whoever is chatting' : ''), 16, Math.min(h - 12, yEnd + 22), { size: 12, color: mc2, maxW: w - 32 });
          }
        }
      });
    }

    /* =====================================================================
       Figure 8 — the mixed-step scale (fair v5, toy model)
       ===================================================================== */
    {
      const fig = q('#fig-fair');
      autoH(fig, 430, 560, 600);
      const LAD = [128, 256, 512, 768, 1024, 1536, 2048];
      const S = { pol: 'fair', P: 2000, share: 0.2, step: 1000, t0: 0, tNow: 0 };
      let R = null;
      function sim() {
        const a = 0.3, b = 0.00068, d = 0.162, tokA = 4, MNBT = 7168, AOUT = 4000, BARR = 10;
        const cost = (n) => a + b * n;
        const rung = LAD.filter((r) => cost(r) <= S.step / 1000).pop() || 128;
        let t = 0, Aout = 0, Aend = null, Brem = S.P, Bdone = null, credit = 0, probed = false;
        let aOvTok = 0, aOvT = 0, maxGap = 0, lastA = 0, guard = 0, nMix = 0;
        const steps = [];
        while (guard++ < 30000) {
          const Aact = Aout < AOUT, Bact = t >= BARR && Brem > 0;
          if (!Aact && !Bact) break;
          let n = 0;
          if (Bact) {
            if (!Aact) n = Math.min(Brem, MNBT);
            else if (S.pol === 'off') n = Math.min(Brem, MNBT - 8);
            else if (S.pol === 'cap') n = Math.min(Brem, 128);
            else if (S.pol === 'fair') { if (!probed) { n = Math.min(Brem, rung); probed = true; } else if (credit >= cost(rung)) n = Math.min(Brem, rung); }
          }
          const dt = n > 0 ? cost(n) : d;
          const mixed = n > 0 && Aact;
          if (S.pol === 'fair' && Aact && Bact) credit += S.share * dt;
          if (S.pol === 'fair' && mixed) credit -= dt;
          if (mixed) nMix++;
          const t0 = t; t += dt;
          if (Aact) { Aout += tokA; if (Bact) { aOvTok += tokA; aOvT += dt; maxGap = Math.max(maxGap, t - lastA); } lastA = t; if (Aout >= AOUT) Aend = t; }
          if (n > 0) { Brem -= n; if (Brem <= 0) Bdone = t; }
          steps.push({ t0, t1: t, n, mixed, a: Aact ? tokA : 0, bp: S.P - Math.max(0, Brem) });
        }
        const soloB = Math.ceil(S.P / MNBT) * a + b * S.P;
        R = { steps, Aend, ttft: Bdone - BARR, aOv: aOvT > 0 ? aOvTok / aOvT : 24.7, maxGap: maxGap || d, rung, T: t, soloB, nMix, BARR };
        // A rate in 2 s windows
        const bins = [];
        const nb = Math.ceil(t / 2);
        for (let k = 0; k < nb; k++) bins.push(0);
        steps.forEach((s) => { if (s.a) { const k = Math.min(nb - 1, Math.floor(s.t1 / 2)); bins[k] += s.a; } });
        R.bins = bins.map((x) => x / 2);
        rT.set(secs(R.ttft)); rA.set(nf(R.aOv, 1) + ' tok/s'); rG.set(secs(R.maxGap)); rR.set(S.pol === 'fair' ? nf(rung) + ' tok' : S.pol === 'cap' ? '128 tok' : S.pol === 'off' ? nf(MNBT - 8) : '—');
        S.t0 = S.tNow;
      }
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Policy', options: [{ v: 'skip', label: 'skip' }, { v: 'off', label: '0 / off' }, { v: 'cap', label: 'N = 128' }, { v: 'fair', label: 'fair v5' }], value: 'fair', onChange: (v) => { S.pol = v; sim(); } });
      UI.seg(ctl, { label: 'B', options: [{ v: 2000, label: '2K' }, { v: 30000, label: '30K' }], value: 2000, onChange: (v) => { S.P = v; sim(); } });
      UI.slider(ctl, { label: 'SHARE σ', min: 0.1, max: 0.5, step: 0.05, value: S.share, fmt: (v) => nf(v, 2), onInput: (v) => { S.share = v; sim(); } });
      UI.slider(ctl, { label: 'MAX_STEP_MS', min: 500, max: 3000, step: 100, value: S.step, fmt: (v) => nf(v), onInput: (v) => { S.step = v; sim(); } });
      const rT = UI.readout(ctl, 'TTFT of B', '—'), rA = UI.readout(ctl, 'A during overlap', '—'), rG = UI.readout(ctl, 'Longest pause of A', '—'), rR = UI.readout(ctl, 'Chunk', '—');
      sim();
      V.fig(fig, (g, w, h, t) => {
        if (w < 60 || h < 60) return;
        S.tNow = t;
        const C = V.C();
        const narrow = w < 600;
        const Tm = Math.min(420, Math.ceil((R.T + 6) / 20) * 20);
        const x0 = narrow ? 40 : 54, x1 = w - 16;
        const X = (s) => x0 + (s / Tm) * (x1 - x0);
        const per = 10, reveal = Math.min(1, (Math.max(0, t - S.t0) % per) / (per - 2.5)) * Tm;
        // scale
        const sAy = 52, cx = w / 2;
        const scoreA = clamp(R.aOv / 24.7, 0, 1);
        const scoreB = clamp(1 - Math.log10(Math.max(1, R.ttft / R.soloB)) / 2.2, 0, 1);
        const ang = clamp((scoreA - scoreB) * 0.4, -0.18, 0.18) * Math.min(1, (t - S.t0) / 1.2);
        const L = Math.min(150, w * 0.3);
        g.save(); g.translate(cx, sAy); g.rotate(-ang);
        V.box(g, -L, -3, 2 * L, 6, { r: 3, fill: V.alpha(C.ink, 0.5) });
        const mood = (s) => (s > 0.75 ? 'happy' : s > 0.4 ? 'neutral' : 'sad');
        critter(g, -L + 16, -16, 12, C.a, mood(scoreA), 0);
        critter(g, L - 16, -16, 12, C.b, mood(scoreB), 0);
        g.restore();
        g.beginPath(); g.moveTo(cx, sAy + 2); g.lineTo(cx - 10, sAy + 18); g.lineTo(cx + 10, sAy + 18); g.closePath(); g.fillStyle = V.alpha(C.muted, 0.7); g.fill();
        V.text(g, 'A · incumbent', cx - L + 16, sAy + 44, { size: 11, color: C.a, weight: 600, align: 'center' });
        V.text(g, 'B · newcomer', cx + L - 16, sAy + 44, { size: 11, color: C.b, weight: 600, align: 'center' });
        // lane A: rate
        const aTop = 128, aBot = narrow ? 268 : 250;
        const AY = (r) => aBot - (r / 30) * (aBot - aTop);
        V.text(g, 'A · tok/s (2 s windows)', x0, aTop - 10, { size: 11, font: 'mono', color: C.a, weight: 600 });
        [0, 10, 20, 30].forEach((v) => { V.line(g, x0, AY(v), x1, AY(v), { color: C.grid }); V.text(g, String(v), x0 - 6, AY(v) + 4, { size: 10.5, font: 'mono', color: C.muted, align: 'right' }); });
        V.line(g, x0, AY(24.7), x1, AY(24.7), { color: V.alpha(C.a, 0.6), dash: [4, 4] });
        V.text(g, 'solo 24.7', x0 + 4, AY(24.7) - 5, { size: 10.5, font: 'mono', color: C.a });
        g.beginPath(); g.moveTo(X(0), AY(0));
        R.bins.forEach((r, k) => { const tt = k * 2 + 1; if (tt > reveal) return; g.lineTo(X(tt), AY(Math.min(30, r))); });
        g.lineTo(X(Math.min(reveal, R.bins.length * 2 - 1)), AY(0)); g.closePath();
        g.fillStyle = V.alpha(C.a, 0.25); g.fill(); g.strokeStyle = C.a; g.lineWidth = 1.6; g.stroke();
        // lane B: progress
        const bTop = aBot + 36, bBot = h - 30;
        const BY = (v) => bBot - (v / S.P) * (bBot - bTop);
        V.text(g, 'B · prefill tokens done', x0, bTop - 8, { size: 11, font: 'mono', color: C.b, weight: 600 });
        V.text(g, '0', x0 - 6, bBot + 4, { size: 10.5, font: 'mono', color: C.muted, align: 'right' });
        V.text(g, S.P >= 10000 ? '30K' : '2K', x0 - 6, bTop + 8, { size: 10.5, font: 'mono', color: C.muted, align: 'right' });
        V.box(g, x0, bTop, x1 - x0, bBot - bTop, { r: 4, fill: V.alpha(C.muted, 0.06) });
        g.beginPath(); g.moveTo(X(R.BARR), BY(0));
        let lastBp = 0, lastT = R.BARR;
        for (const s of R.steps) { if (s.t1 > reveal) break; if (s.n > 0) { g.lineTo(X(s.t0), BY(lastBp)); g.lineTo(X(s.t1), BY(s.bp)); lastBp = s.bp; lastT = s.t1; } }
        if (reveal > R.BARR) g.lineTo(X(Math.max(lastT, Math.min(reveal, R.T))), BY(lastBp));
        g.strokeStyle = C.b; g.lineWidth = 2; g.stroke();
        R.steps.forEach((s) => { if (s.mixed && s.t1 <= reveal) { g.fillStyle = V.alpha(C.b, 0.8); g.fillRect(X(s.t0), bBot - 5, Math.max(1.2, X(s.t1) - X(s.t0)), 5); } });
        V.line(g, X(R.BARR), bTop, X(R.BARR), bBot, { color: V.alpha(C.c, 0.8), dash: [3, 3] });
        V.text(g, 'B arrives', X(R.BARR) + 4, bBot - 14, { size: 10.5, font: 'mono', color: C.c });
        const tb = R.BARR + R.ttft;
        if (reveal >= tb) {
          V.dot(g, X(tb), BY(S.P), 5, C.b);
          /* label with panel background, so it does not sit on top of the B curve */
          const lbB = '1st token of B: ' + secs(R.ttft), rightB = X(tb) > x1 - 170;
          g.font = V.font(11.5, 'mono', 600); const lwB = g.measureText(lbB).width;
          const lxB = rightB ? X(tb) - 8 - lwB : X(tb) + 8, lyB = BY(S.P) + 14;
          V.box(g, lxB - 5, lyB - 13, lwB + 10, 19, { r: 5, fill: V.alpha(C.panel || C.bg, 0.94), stroke: V.alpha(C.b, 0.5), lw: 1 });
          V.text(g, lbB, lxB, lyB, { size: 11.5, font: 'mono', weight: 600, color: C.b, align: 'left' });
        }
        if (R.Aend && reveal >= R.Aend) { V.line(g, X(R.Aend), aTop, X(R.Aend), aBot, { color: V.alpha(C.a, 0.6), dash: [2, 3] }); V.text(g, 'A finishes', X(R.Aend) - 4, aBot - 6, { size: 10.5, font: 'mono', color: C.a, align: 'right' }); }
        V.line(g, X(reveal), aTop, X(reveal), bBot, { color: V.alpha(C.c, 0.5) });
        /* unitless ticks (the unit goes only on the last one), with a step that guarantees ≥ 40 px between labels */
        const tStep = [10, 20, 50, 100, 200].find((st) => (st / Tm) * (x1 - x0) >= 40) || 200;
        for (let s = 0; s <= Tm + 1e-9; s += tStep) { const last = s + tStep > Tm; V.text(g, last ? s + ' s' : String(s), X(s), h - 12, { size: 10.5, font: 'mono', color: C.muted, align: last && X(s) > x1 - 24 ? 'right' : 'center' }); }
      });
    }

    /* =====================================================================
       Figure 9 — KV pool and preemption
       ===================================================================== */
    {
      const fig = q('#fig-preempt');
      autoH(fig, 410, 540, 600);
      /* prefill rates measured in each recipe: [context, tok/s] */
      const RDS = [[2048, 2563], [131072, 1665], [900000, 875]];
      const RQW = [[32768, 2808], [65536, 2654], [131072, 2432], [262144, 2171], [400000, 1955], [600000, 1711]];
      const PRE = {
        ds: { pool: 2331430, pts: RDS, lab: 'DeepSeek · pool 2,331,430', src: 'DS, sweep 0731' },
        qb: { pool: 1340476, pts: RQW, lab: 'Qwen v0.30 BF16 · pool 1,340,476', src: 'Qwen, long prefill TP2' },
        qf: { pool: 3652200, pts: RQW, lab: 'Qwen day-0 FP8 · pool 3,652,200', src: 'Qwen, long prefill TP2' },
      };
      /* interpolation in log of context; outside the range, the nearest endpoint */
      function rateAt(P, L) {
        const p = P.pts;
        if (L <= p[0][0]) return { r: p[0][1], out: L < p[0][0] };
        if (L >= p[p.length - 1][0]) return { r: p[p.length - 1][1], out: L > p[p.length - 1][0] };
        let i = 0; while (p[i + 1][0] < L) i++;
        const u = Math.log(L / p[i][0]) / Math.log(p[i + 1][0] / p[i][0]);
        return { r: p[i][1] + (p[i + 1][1] - p[i][1]) * u, out: false };
      }
      const LS = [50000, 100000, 200000, 262144, 500000, 1000000];
      const S = { p: 'ds', li: 4, N: 6 };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Pool', options: [{ v: 'ds', label: 'DeepSeek' }, { v: 'qb', label: 'Qwen BF16' }, { v: 'qf', label: 'Qwen FP8' }], value: 'ds', onChange: (v) => { S.p = v; upd(); } });
      UI.slider(ctl, { label: 'Context', min: 0, max: LS.length - 1, step: 1, value: S.li, fmt: (v) => (LS[v] >= 1e6 ? '1M' : nf(LS[v] / 1000) + 'K'), onInput: (v) => { S.li = v; upd(); } });
      UI.slider(ctl, { label: 'Requests', min: 1, max: 8, step: 1, value: S.N, onInput: (v) => { S.N = v; upd(); } });
      const rF = UI.readout(ctl, 'Fit together', '—'), rD = UI.readout(ctl, 'Demand / pool', '—'), rR = UI.readout(ctl, 'Recompute 1', '—');
      function upd() {
        const P = PRE[S.p], L = LS[S.li];
        rF.set(Math.min(S.N, Math.floor(P.pool / L)) + ' of ' + S.N);
        rD.set(nf((S.N * L) / P.pool, 2) + '×');
        rR.set(secs(L / rateAt(P, L).r));
      }
      upd();
      V.fig(fig, (g, w, h, t) => {
        if (w < 60 || h < 60) return;
        const C = V.C();
        const P = PRE[S.p], L = LS[S.li], N = S.N;
        const narrow = w < 600;
        const f = (t % 9) / 9, gr = V.ease.inOut(Math.min(1, f / 0.55));
        let act = N;
        while (act > 0 && act * gr * L > P.pool) act--;
        const cols = narrow ? 20 : 24, rows = narrow ? 12 : 10, nC = cols * rows, cellTok = P.pool / nC;
        const gx = 16, gy = 44, gw = narrow ? w - 32 : w * 0.6, cs = Math.max(4, Math.min(gw / cols, (narrow ? 240 : h - 190) / rows));
        V.text(g, P.lab + ' tokens · 1 cell = ' + nf(cellTok / 1000, 1) + 'K', gx, gy - 14, { size: 11, font: 'mono', color: C.muted, maxW: w - 32 });
        let k = 0;
        const owner = new Array(nC).fill(-1);
        for (let i = 0; i < act; i++) { const n = Math.max(gr > 0 ? 1 : 0, Math.round((gr * L) / cellTok)); for (let j = 0; j < n && k < nC; j++) owner[k++] = i; }
        for (let c = 0; c < nC; c++) {
          const x = gx + (c % cols) * cs, y = gy + Math.floor(c / cols) * cs;
          g.fillStyle = owner[c] >= 0 ? V.alpha(C.series[owner[c] % 6], 0.82) : V.alpha(C.muted, 0.12);
          V.rr(g, x, y, Math.max(1, cs - 2), Math.max(1, cs - 2), Math.max(0, Math.min(3, cs / 5))); g.fill();
        }
        const used = Math.min(1, (act * gr * L) / P.pool);
        V.text(g, 'occupancy ' + nf(used * 100) + '%', gx, gy + rows * cs + 16, { size: 11.5, font: 'mono', color: used > 0.97 ? C.d : C.ink, weight: 600 });
        const listEnd = narrow ? gy + rows * cs + 34 + Math.ceil(N / 2) * 42 : 0;
        {
          const dy = narrow ? listEnd + 10 : gy + rows * cs + 40, bwMax = (narrow ? w - 32 : cols * cs) - 72, dem = N * L, sc = bwMax / Math.max(P.pool, dem);
          V.text(g, 'pool', gx, dy + 12, { size: 11, font: 'mono', color: C.muted });
          V.box(g, gx + 70, dy, P.pool * sc, 16, { r: 4, fill: V.alpha(C.a, 0.5) });
          V.text(g, 'demand', gx, dy + 38, { size: 11, font: 'mono', color: C.muted });
          V.box(g, gx + 70, dy + 26, Math.min(dem, P.pool) * sc, 16, { r: 4, fill: V.alpha(C.b, 0.6) });
          if (dem > P.pool) { V.box(g, gx + 70 + P.pool * sc, dy + 26, (dem - P.pool) * sc, 16, { r: 4, fill: V.alpha(C.red || '#FB2C36', 0.25) }); hatch(g, gx + 70 + P.pool * sc, dy + 26, (dem - P.pool) * sc, 16, V.alpha(C.red || '#FB2C36', 0.8), 5); }
          V.text(g, nf(N) + ' × ' + (L >= 1e6 ? '1M' : nf(L / 1000) + 'K') + ' = ' + nf(dem / 1e6, 2) + 'M · ' + nf(dem / P.pool, 2) + '× the pool', gx + 70, dy + 60, { size: 11, font: 'mono', color: dem > P.pool ? (C.red || '#FB2C36') : C.a });
        }
        // request list
        const lx = narrow ? 16 : gx + cols * cs + 24, ly = narrow ? gy + rows * cs + 34 : gy;
        const lw = narrow ? (w - 40) / 2 : w - lx - 16, lh = narrow ? 36 : Math.min(38, (h - ly - 20) / 8);
        for (let i = 0; i < N; i++) {
          const x = narrow ? lx + (i % 2) * (lw + 8) : lx, y = narrow ? ly + Math.floor(i / 2) * (lh + 6) : ly + i * (lh + 4);
          const kicked = i >= act;
          const gi = P.pool / ((i + 1) * L);
          const flash = kicked && gi <= 1 && gr - gi < 0.1 && gr >= gi;
          const col = kicked ? (C.red || '#FB2C36') : C.series[i % 6];
          V.box(g, x, y, lw, lh - 2, { r: 8, fill: V.alpha(col, flash ? 0.45 + 0.3 * Math.sin(t * 20) : kicked ? 0.1 : 0.14), stroke: V.alpha(col, 0.6), lw: 1 });
          critter(g, x + 14, y + (lh - 2) / 2, 9, col, kicked ? 'sad' : 'happy', t);
          const txt = kicked ? 'R' + (i + 1) + ' ↺ preempted' : 'R' + (i + 1) + ' · ' + nf((gr * L) / 1000) + 'K';
          V.text(g, txt, x + 28, y + (lh - 2) / 2 + 4, { size: 11.5, font: 'mono', color: C.ink, maxW: lw - 34, weight: kicked ? 600 : 400 });
        }
        const mcol = act < N ? (C.red || '#FB2C36') : C.a;
        const ra = rateAt(P, L), Lk = L >= 1e6 ? '1M' : nf(L / 1000) + 'K';
        const finalFits = N * L <= P.pool;
        if (act < N) {
          V.text(g, (N - act) + ' request(s) back in the queue; recomputing each ≈ ' + secs(L / ra.r), 16, h - 30, { size: 12, weight: 600, color: mcol, maxW: w - 32 });
          V.text(g, 'rate at ' + Lk + ': ~' + nf(ra.r) + ' tok/s (' + P.src + (ra.out ? ', measured endpoint: optimistic' : ', interpolated') + ')', 16, h - 12, { size: 11, color: C.muted, maxW: w - 32 });
        } else if (gr < 1) {
          V.text(g, 'growing… ' + (finalFits ? 'final demand fits in the pool' : 'final demand exceeds the pool: someone gets preempted'), 16, h - 14, { size: 12, weight: 600, color: finalFits ? C.muted : C.d, maxW: w - 32 });
        } else V.text(g, 'everything fits: live tokens stay below the pool', 16, h - 14, { size: 12, weight: 600, color: mcol, maxW: w - 32 });
      });
    }

    /* =====================================================================
       Figure 10 — async scheduling
       ===================================================================== */
    {
      const fig = q('#fig-async');
      autoH(fig, 300, 360, 600);
      const S = { c: 3, g: 72, spec: true };
      const ctl = UI.controls(fig);
      UI.slider(ctl, { label: 'CPU per step', min: 0.5, max: 30, step: 0.5, value: S.c, fmt: (v) => nf(v, 1) + ' ms', onInput: (v) => { S.c = v; upd(); } });
      UI.slider(ctl, { label: 'GPU per step', min: 20, max: 120, step: 2, value: S.g, fmt: (v) => nf(v) + ' ms', onInput: (v) => { S.g = v; upd(); } });
      UI.toggle(ctl, { label: 'with speculation (k = 6)', value: true, onChange: (v) => { S.spec = v; } });
      const r1 = UI.readout(ctl, 'Synchronous', '—'), r2 = UI.readout(ctl, 'Async', '—'), r3 = UI.readout(ctl, 'Gain', '—');
      function upd() { const s = 1000 / (S.c + S.g), a = 1000 / Math.max(S.c, S.g); r1.set(nf(s, 1) + ' steps/s'); r2.set(nf(a, 1) + ' steps/s'); r3.set('+' + nf((a / s - 1) * 100, 1) + '%'); }
      upd();
      const ACC = [4, 2, 5, 3, 4, 2, 5];
      V.fig(fig, (g, w, h, t) => {
        if (w < 60 || h < 60) return;
        const C = V.C();
        const narrow = w < 600;
        const c = S.c, gg = S.g, m = Math.max(c, gg);
        const T = 5 * (c + gg);
        const x0 = narrow ? 44 : 70, x1 = w - 14, X = (s) => x0 + (s / T) * (x1 - x0);
        const per = 6, ph = ((t % per) / per) * T * 1.05;
        const lane = (y, label, col) => { V.text(g, label, x0 - 8, y + 13, { size: 11, font: 'mono', color: col, align: 'right' }); V.box(g, x0, y, x1 - x0, 20, { r: 4, fill: V.alpha(C.muted, 0.06) }); };
        const block = (y, s, e, col, lab) => {
          if (s >= T || s >= ph) return;
          const ee = Math.min(e, T, ph);
          V.box(g, X(s) + 0.5, y + 1, Math.max(1.5, X(ee) - X(s) - 1), 18, { r: 3, fill: V.alpha(col, 0.85) });
          if (X(ee) - X(s) > 24) V.text(g, lab, X(s) + 5, y + 14, { size: 11, font: 'mono', weight: 600, color: ink(C) });
        };
        // synchronous
        const y1 = 26;
        V.text(g, 'synchronous: GPU waits for CPU (hatched = idle GPU)', x0, y1 - 8, { size: 11.5, weight: 600, color: C.d, maxW: x1 - x0 });
        lane(y1, 'CPU', C.d); lane(y1 + 26, 'GPU', C.a);
        for (let i = 0; i < 6; i++) {
          const s = i * (c + gg);
          block(y1, s, s + c, C.d, 'N+' + i);
          block(y1 + 26, s + c, s + c + gg, C.a, 'N+' + i);
          if (s < ph && s < T) hatch(g, X(s), y1 + 27, Math.max(0, X(Math.min(s + c, ph, T)) - X(s)), 18, V.alpha(C.red || '#FB2C36', 0.7), 4);
        }
        // async
        const y2 = y1 + 84;
        V.text(g, 'async: the CPU is already planning the next step', x0, y2 - 8, { size: 11.5, weight: 600, color: C.a });
        lane(y2, 'CPU', C.d); lane(y2 + 26, 'GPU', C.a);
        block(y2, 0, c, C.d, 'N');
        for (let i = 0; i < 8; i++) {
          const s = c + i * m;
          block(y2 + 26, s, s + gg, C.a, 'N+' + i);
          block(y2, s, s + c, C.d, 'N+' + (i + 1));
          if (c > gg && s + gg < ph) hatch(g, X(s + gg), y2 + 27, Math.max(0, X(Math.min(s + m, ph, T)) - X(s + gg)), 18, V.alpha(C.red || '#FB2C36', 0.7), 4);
        }
        V.line(g, X(Math.min(ph, T)), y1 - 2, X(Math.min(ph, T)), y2 + 48, { color: V.alpha(C.c, 0.7), lw: 1.2 });
        // scheduler view: placeholders
        const y3 = y2 + 84;
        V.text(g, 'what the scheduler sees of one request (async)', x0, y3 - 8, { size: 11.5, weight: 600, color: C.ink, maxW: x1 - x0 });
        const done = Math.max(0, Math.floor((Math.min(ph, T) - c) / m));
        const nCommitted = ACC.slice(0, Math.min(done, ACC.length)).reduce((a, b) => a + (S.spec ? b : 1), 0);
        const pend = S.spec ? 7 : 1;
        const sq = narrow ? 13 : 17, maxSq = Math.floor((x1 - x0) / (sq + 3));
        const showC = Math.min(nCommitted, Math.max(0, maxSq - pend - 1));
        let x = x0;
        for (let i = 0; i < showC; i++) { V.box(g, x, y3, sq, sq, { r: 3, fill: V.alpha(C.a, 0.85) }); x += sq + 3; }
        V.text(g, '│', x + 1, y3 + sq - 3, { size: 12, color: C.muted });
        x += 10;
        for (let i = 0; i < pend && x + sq < x1; i++) {
          g.setLineDash([3, 2]); g.strokeStyle = C.muted; g.lineWidth = 1.2; V.rr(g, x, y3, sq, sq, 3); g.stroke(); g.setLineDash([]);
          if (sq >= 15) V.text(g, '-1', x + sq / 2, y3 + sq / 2 + 4, { size: 10, font: 'mono', color: C.muted, align: 'center' });
          x += sq + 3;
        }
        const note = S.spec ? 'green: tokens already confirmed · dashed: 1 anchor + 6 drafts still unknown (-1); only some will be accepted'
          : 'green: tokens already confirmed · dashed: the token the in-flight step has yet to produce';
        wrapText(g, note, x0, y3 + sq + 18, x1 - x0, 15, { size: 11, color: C.muted });
      });
    }

    /* =====================================================================
       Figure 3 — concurrency ladders (recipe data)
       ===================================================================== */
    {
      const fig = q('#fig-ladder');
      autoH(fig, 390, 460, 600);
      const DS = {
        q30: { lab: 'Qwen · lane v0.30 · prose', x: [1, 2, 4, 8, 16], agg: [59.2, 75.1, 118.5, 216.4, 194.9], per: [59.2, 37.7, 29.9, 28.8, 25.5], mns: 8, note: '×16 exceeds max_num_seqs = 8: queueing, mean TTFT 6.5–9.7 s' },
        q0: { lab: 'Qwen · day-0 FP8 · prose', x: [1, 2, 4, 6, 8], agg: [54.4, 86.5, 128.1, 169.7, 207.0], per: [54.4, 45.1, 34.2, 29.7, 26.7], mns: 8, note: 'aggregate 3.8× from ×1 to ×8; per stream drops to 49%' },
        glm: { lab: 'GLM · ladder 2026-08-31 · code', x: [1, 2, 4, 8, 12, 16], agg: [41.9, 52.3, 77.9, 93.8, 116.2, 127.2], per: [44.3, 28.8, 21.0, 13.9, 11.9, 10.2], mns: 16, note: '1M, MNBT 2048, 16 slots, mixed 512 · median TTFT 0.4–0.6 s (×1) to 1.3–1.6 s (×16)' },
        ds: { lab: 'DeepSeek · 0731 · prompt 256', x: [1, 2, 4, 6], agg: [69.1, 104.9, 164.5, 191.2], per: [75.4, 58.3, 46.8, 36.9], mns: 6, note: 'TTFT 0.63 → 1.42 s from c=1 to c=6' },
        tf: { lab: 'GLM·TF · sparkDash · prose', x: [1, 2, 3, 4], agg: [60.4, 79.2, 89.5, 108.8], per: [60.4, 40.4, 30.6, 27.9], mns: 4, mnsLab: 'PARALLEL', note: 'clocks capped at 2,200 MHz, temp 0 · TTFT 170 → 340 ms · concurrent replies = solo, bit for bit' },
        ds32: { lab: 'DeepSeek · N=32, MNBT 12288 (data)', x: [1, 8, 16, 32], agg: [54, 154, 213, 325], lo: [47, null, null, null], per: null, mns: 32, note: 'c=1 is the measured range 47–54 · #141 stability not guaranteed: performance data, not a recommendation' },
      };
      const S = { d: 'q30', t0: 0, tNow: 0 };
      const ctl = UI.controls(fig);
      UI.select(ctl, { label: 'Recipe', options: [{ v: 'q30', label: 'Qwen v0.30 (prose)' }, { v: 'q0', label: 'Qwen day-0 (prose)' }, { v: 'glm', label: 'GLM (code)' }, { v: 'tf', label: 'GLM·TF (prose)' }, { v: 'ds', label: 'DeepSeek 0731' }, { v: 'ds32', label: 'DeepSeek N=32 (data)' }], value: 'q30', onChange: (v) => { S.d = v; S.t0 = S.tNow; upd(); } });
      const rA = UI.readout(ctl, 'Peak aggregate / ×1', '—'), rP = UI.readout(ctl, 'Per stream at peak / ×1', '—');
      function upd() {
        const d = DS[S.d];
        const iMax = d.agg.indexOf(Math.max(...d.agg));
        const a0 = d.lo && d.lo[0] != null ? null : d.agg[0];
        rA.set(a0 ? nf(d.agg[iMax] / a0, 2) + '× (×' + d.x[iMax] + ')' : nf(d.agg[iMax] / d.agg[0], 1) + '–' + nf(d.agg[iMax] / d.lo[0], 1) + '× (×' + d.x[iMax] + ')');
        rP.set(d.per ? nf((d.per[iMax] / d.per[0]) * 100) + '% (×' + d.x[iMax] + ')' : '—');
      }
      upd();
      const H = V.fig(fig, (g, w, h, t) => {
        if (w < 60 || h < 60) return;
        S.tNow = t;
        const C = V.C();
        const d = DS[S.d], n = d.x.length;
        const x0 = 46, x1 = w - 14, y0 = 30, y1 = h - 64;
        const ymax = Math.ceil((Math.max(...d.agg) * 1.18) / 50) * 50;
        const Y = (v) => y1 - (v / ymax) * (y1 - y0);
        const slot = (x1 - x0) / n, bw = Math.min(56, slot * 0.55);
        const grow = V.ease.out(Math.min(1, (t - S.t0) / 0.9));
        V.text(g, d.lab, x0, 16, { size: 12, weight: 600, color: C.ink, maxW: x1 - x0 });
        for (let v = 0; v <= ymax; v += ymax > 200 ? 100 : 50) { V.line(g, x0, Y(v), x1, Y(v), { color: C.grid }); V.text(g, String(v), x0 - 6, Y(v) + 4, { size: 10.5, font: 'mono', color: C.muted, align: 'right' }); }
        const cxs = d.x.map((_, i) => x0 + slot * (i + 0.5));
        const hov = H.pointer.inside ? Math.floor((H.pointer.x - x0) / slot) : -1;
        d.agg.forEach((v, i) => {
          const bh = (y1 - Y(v)) * grow;
          const lo = d.lo && d.lo[i] != null ? d.lo[i] : null;
          if (lo != null) {
            /* measured range: solid up to the minimum, light band up to the maximum */
            const bl = (y1 - Y(lo)) * grow;
            V.box(g, cxs[i] - bw / 2, y1 - bl, bw, bl, { r: 5, fill: V.alpha(C.a, hov === i ? 0.95 : 0.7) });
            V.box(g, cxs[i] - bw / 2, y1 - bh, bw, bh - bl, { r: 3, fill: V.alpha(C.a, 0.25), stroke: V.alpha(C.a, 0.7), lw: 1 });
            V.text(g, nf(lo) + '–' + nf(v), cxs[i], y1 - bh - 6, { size: 11, font: 'mono', color: C.ink, align: 'center' });
          } else {
            V.box(g, cxs[i] - bw / 2, y1 - bh, bw, bh, { r: 5, fill: V.alpha(C.a, hov === i ? 0.95 : 0.7) });
            /* if the per-stream line character lands on top of the bar, the label moves above it */
            const yPer = d.per ? y1 - (y1 - Y(d.per[i])) * grow : Infinity;
            const yLab = Math.abs(yPer - (y1 - bh)) < 16 ? Math.min(y1 - bh, yPer - 8) - 6 : y1 - bh - 6;
            V.text(g, nf(v, v % 1 ? 1 : 0), cxs[i], yLab, { size: 11, font: 'mono', color: C.ink, align: 'center' });
          }
          V.text(g, '×' + d.x[i], cxs[i], y1 + 16, { size: 11, font: 'mono', color: C.muted, align: 'center' });
        });
        if (d.per) {
          g.beginPath();
          d.per.forEach((v, i) => { const yy = y1 - (y1 - Y(v)) * grow; if (i) g.lineTo(cxs[i], yy); else g.moveTo(cxs[i], yy); });
          g.strokeStyle = C.b; g.lineWidth = 2.4; g.stroke();
          d.per.forEach((v, i) => { const yy = y1 - (y1 - Y(v)) * grow; critter(g, cxs[i], yy, 7, C.b, i === 0 ? 'happy' : v / d.per[0] > 0.5 ? 'neutral' : 'sad', t); });
        }
        const im = d.x.indexOf(d.mns);
        {
          /* between the rung = max_num_seqs and the next; at the right edge if every rung fits in the slots */
          const xm = im >= 0 && im < n - 1 ? (cxs[im] + cxs[im + 1]) / 2 : x1 - 2;
          V.line(g, xm, y0, xm, y1, { color: C.d, dash: [5, 4], lw: 1.5 });
          const ml = (d.mnsLab || 'max_num_seqs') + ' = ' + d.mns; g.font = V.font(11, 'mono'); const mw = g.measureText(ml).width;
          V.text(g, ml, xm + 4 + mw > x1 ? xm - 4 : xm + 4, y0 + 10, { size: 11, font: 'mono', color: C.d, align: xm + 4 + mw > x1 ? 'right' : 'left' });
        }
        V.box(g, x0, h - 44, 10, 10, { r: 3, fill: V.alpha(C.a, 0.75) }); V.text(g, 'aggregate', x0 + 14, h - 35, { size: 11, color: C.muted });
        if (d.per) { V.line(g, x0 + 84, h - 39, x0 + 104, h - 39, { color: C.b, lw: 2.4 }); V.text(g, 'per stream', x0 + 108, h - 35, { size: 11, color: C.muted }); }
        V.text(g, d.note, x0, h - 14, { size: 11, color: C.muted, maxW: x1 - x0 });
        if (hov >= 0 && hov < n) {
          const loH = d.lo && d.lo[hov] != null ? d.lo[hov] : null, tW = loH != null ? 196 : 146;
          const tx = clamp(cxs[hov] + 12, x0, x1 - tW - 4), ty = y0 + 18;
          V.box(g, tx, ty, tW, d.per ? 46 : 30, { r: 8, fill: C.panel, stroke: V.alpha(C.a, 0.6) });
          V.text(g, '×' + d.x[hov] + ' · aggregate ' + (loH != null ? nf(loH) + '–' + nf(d.agg[hov]) + ' (range)' : nf(d.agg[hov], 1)), tx + 8, ty + 18, { size: 11, font: 'mono', color: C.ink, maxW: tW - 14 });
          if (d.per) V.text(g, 'per stream ' + nf(d.per[hov], 1) + ' tok/s', tx + 8, ty + 36, { size: 11, font: 'mono', color: C.b });
        }
      });
    }    /* =====================================================================
       Figure 11 — TensorFold: rank 0's metronome (chunk or round)
       ===================================================================== */
    const tfCache = { k: null, v: null };
    const TFC = (C) => { if (tfCache.k !== C) { tfCache.k = C; tfCache.v = getComputedStyle(document.documentElement).getPropertyValue('--tf').trim() || C.d; } return tfCache.v; };
    {
      const fig = q('#fig-tf-metronomo');
      autoH(fig, 420, 640, 700);
      const TX = {
        rule: 'rank 0, every contended iteration: C += φ · if C ≥ 1 → prompt chunk (C −= 1) · else → decode round',
        credit: 'credit C', calm: 'no contention: nothing to decide', last: 'recent iterations',
        A: 'A · decoding', B: (p) => 'B · ' + p + '-token prompt', waitB: 'hasn\'t arrived yet', filling: (a, b) => 'filling ' + a + ' / ' + b,
        waited: (s) => 'waiting for ' + s, done: (s) => '1st token! TTFT ' + s,
        rate: (v) => '≈ ' + v + ' tok/s now', toks: (n) => n + ' tokens so far',
        zoom: (s) => 'magnifier: the model\'s last ' + s, all: 'the whole run', arr: 'B arrives', ft: 'B\'s 1st token', now: 'now',
        R: 'R', P: 'P', legR: 'R · decode round (A; then A and B)', legP: 'P · B\'s prompt chunk',
        note: 'toy model: chunk = 0.12 s + 0.45 ms per row · round = 70 ms (illustrative) · A alone = 60.4 tok/s (measured)',
        phi: 'φ (TF_GLM_FILL_SHARE)', rows: 'chunk with decode', prompt: 'B\'s prompt', again: '▶ B arrives again',
        rT: 'B\'s TTFT (model)', rA: 'A during the prefill', rS: 'time for the prompt',
        vT: (v, s) => v + ' (alone: ' + s + ')', vA: (v) => v + ' tok/s (alone: 60.4)', vS: (a, b) => a + '% of the time · ' + b + '% of the iterations',
      };
      const TR = 0.07, TOKR = 60.4 * TR, FIX = 0.12, BPR = (2048 / 1950 - FIX) / 2048, PRE = 1.2, POST = 1.6;
      const tc = (n) => FIX + n * BPR;
      const S = { phi: 0.5, rows: 1024, P: 32768, t0: 0, tNow: 0, k: 0, sim: null };
      const ctl = UI.controls(fig);
      UI.slider(ctl, { label: TX.phi, min: 0.1, max: 1, step: 0.05, value: S.phi, fmt: (v) => nf(v, 2), onInput: (v) => { S.phi = Math.round(v * 100) / 100; build(); restart(); } });
      UI.seg(ctl, { label: TX.rows, options: [512, 1024, 2048].map((v) => ({ v, label: nf(v) })), value: 1024, onChange: (v) => { S.rows = v; build(); restart(); } });
      UI.seg(ctl, { label: TX.prompt, options: [{ v: 8192, label: '8K' }, { v: 32768, label: '32K' }, { v: 65536, label: '64K' }], value: 32768, onChange: (v) => { S.P = v; build(); restart(); } });
      UI.button(ctl, TX.again, () => restart());
      const rT = UI.readout(ctl, TX.rT, '—'), rA = UI.readout(ctl, TX.rA, '—'), rS = UI.readout(ctl, TX.rS, '—');
      function restart() { S.t0 = S.tNow; S.k = 0; }
      /* the code's own rule (multi.py, _fill_turn): credit in iterations, in floating point */
      function build() {
        const ev = [];
        let t = 0;
        for (let i = 0; i < Math.round(PRE / TR); i++) { ev.push({ k: 'R', t0: t, t1: t + TR, two: false, cIn: 0, cOut: 0, calm: true }); t += TR; }
        const tArr = t;
        let c = 0, pos = 0, nR = 0, nF = 0, tokA = 0, tF = 0;
        while (pos < S.P) {
          c += S.phi;
          const cIn = c;
          if (c >= 1) {
            c -= 1;
            const n = Math.min(S.rows, S.P - pos), d = tc(n);
            ev.push({ k: 'P', t0: t, t1: t + d, n, from: pos, cIn, cOut: c });
            t += d; tF += d; pos += n; nF++;
          } else {
            ev.push({ k: 'R', t0: t, t1: t + TR, two: false, cIn, cOut: c });
            t += TR; nR++; tokA += TOKR;
          }
        }
        const tFirst = t;
        for (let i = 0; i < Math.round(POST / TR); i++) { ev.push({ k: 'R', t0: t, t1: t + TR, two: true, cIn: 0, cOut: 0, calm: true }); t += TR; }
        let solo = 0;
        for (let p = 0; p < S.P; p += 2048) solo += tc(Math.min(2048, S.P - p));
        const ttft = tFirst - tArr;
        S.sim = { ev, tArr, tFirst, tEnd: t, ttft, solo, aRate: tokA / ttft, fT: tF / ttft, fI: nF / (nF + nR), speed: clamp(ttft / 9, 1.4, 3.5) };
        S.k = 0;
        rT.set(TX.vT(secs(ttft), secs(solo)), ttft > solo * 1.5 ? 'warn' : 'ok');
        rA.set(TX.vA(nf(S.sim.aRate, 1)), S.sim.aRate < 20 ? 'warn' : 'ok');
        rS.set(TX.vS(nf(S.sim.fT * 100), nf(S.sim.fI * 100)));
      }
      build();

      V.fig(fig, (g, w, h, t) => {
        if (w < 60 || h < 60) return;
        S.tNow = t;
        const C = V.C(), cTF = TFC(C), sim = S.sim, ev = sim.ev;
        const narrow = w < 700;
        let tau = (t - S.t0) * sim.speed;
        if (tau > sim.tEnd + 1.8 * sim.speed) { S.t0 = t; S.k = 0; tau = 0; }
        tau = Math.min(tau, sim.tEnd - 1e-6);
        let k = S.k;
        if (k >= ev.length || ev[k].t0 > tau) k = 0;
        while (k < ev.length - 1 && ev[k].t1 <= tau) k++;
        S.k = k;
        const e = ev[k], prog = clamp((tau - e.t0) / (e.t1 - e.t0), 0, 1);
        const phase = tau < sim.tArr ? 0 : tau < sim.tFirst ? 1 : 2;
        const L = 14, R = w - 14, W = R - L;
        /* running totals: A's tokens, B's prompt position */
        let totA = 0, posB = 0, tokWin = 0;
        const winM = 1.5;
        for (let j = 0; j <= k; j++) {
          const r = ev[j], f = j < k ? 1 : prog;
          if (r.k === 'R') {
            totA += TOKR * f;
            const a = Math.max(r.t0, tau - winM), b = Math.min(r.t1, tau);
            if (b > a) tokWin += (TOKR * (b - a)) / TR;
          } else posB = r.from + r.n * f;
        }
        if (phase === 2) posB = S.P;
        const rateA = tokWin / Math.min(winM, Math.max(tau, 0.3));

        let y = 18;
        y += V.wrap(g, TX.rule, L, y, W, { size: 11.5, font: 'mono', color: C.muted, lineH: 15 }) + 6;
        /* credit gauge */
        const gW = narrow ? W : Math.min(320, W * 0.36);
        V.text(g, TX.credit, L, y + 4, { size: 11.5, weight: 600, color: C.ink });
        const gy = y + 14, gh = 16, gx = L, gw = gW - 64;
        let lvl = 0;
        if (phase === 1) lvl = e.k === 'P' ? (prog < 0.3 ? e.cIn : V.lerp(e.cIn, e.cOut, V.ease.inOut(clamp((prog - 0.3) / 0.5, 0, 1)))) : e.cIn;
        const X = (v) => gx + (v / 1.25) * gw;
        V.box(g, gx, gy, gw, gh, { r: 6, fill: V.alpha(C.muted, 0.12), stroke: V.alpha(C.muted, 0.4), lw: 1 });
        if (lvl > 0.001) V.box(g, gx, gy, Math.max(6, X(Math.min(lvl, 1.25)) - gx), gh, { r: 6, fill: lvl >= 1 - 1e-9 ? cTF : V.alpha(C.b, 0.85) });
        V.line(g, X(1), gy - 3, X(1), gy + gh + 3, { color: C.ink, lw: 1.5, dash: [3, 2] });
        V.text(g, '1', X(1), gy - 6, { size: 11, font: 'mono', color: C.ink, align: 'center' });
        V.text(g, phase === 1 ? 'C = ' + nf(lvl, 2) : 'C = —', gx + gw + 8, gy + 12, { size: 11.5, font: 'mono', weight: 600, color: C.ink });
        if (phase !== 1) V.text(g, TX.calm, gx, gy + gh + 15, { size: 11, color: C.muted, maxW: gW });
        /* recent iterations */
        const cx0 = narrow ? L : L + gW + 26, cy0 = narrow ? gy + gh + 26 : y - 4;
        V.text(g, TX.last, cx0, cy0 + 8, { size: 11.5, weight: 600, color: C.ink });
        const chipW = 22, chipGap = 4, nChip = Math.max(4, Math.floor((R - cx0 + chipGap) / (chipW + chipGap)));
        const recent = [];
        for (let j = k; j >= 0 && recent.length < nChip; j--) if (!ev[j].calm) recent.unshift(ev[j]);
        recent.forEach((r, j) => {
          const xx = cx0 + j * (chipW + chipGap), yy = cy0 + 16, cur = r === e;
          const col = r.k === 'P' ? cTF : C.a;
          V.box(g, xx, yy, chipW, 20, { r: 5, fill: V.alpha(col, cur ? 1 : 0.55), stroke: cur ? C.ink : null, lw: 1.5 });
          V.text(g, r.k === 'P' ? TX.P : TX.R, xx + chipW / 2, yy + 14.5, { size: 11, font: 'mono', weight: 700, color: ink(C), align: 'center' });
        });
        y = (narrow ? cy0 + 46 : gy + gh + 22) + 8;

        /* characters: A (decoding) and B (incoming prompt) */
        const cardH = 82, gap = 12;
        const wA = narrow ? W : Math.min(360, (W - gap) / 2), wB = narrow ? W : Math.min(440, W - wA - gap);
        const xA = L, xB = narrow ? L : L + wA + gap, yA = y;
        if (narrow) y += cardH + 10;
        V.box(g, xA, yA, wA, cardH, { r: 12, fill: V.alpha(C.a, 0.08), stroke: V.alpha(C.a, 0.5), lw: 1.2 });
        const moodA = rateA > 40 ? 'happy' : rateA > 15 ? 'neutral' : 'sad';
        critter(g, xA + 24, yA + 30, 15, C.a, moodA, t);
        V.text(g, TX.A, xA + 46, yA + 22, { size: 12.5, weight: 600, color: C.ink, maxW: wA - 52 });
        V.text(g, TX.rate(nf(rateA, 1)), xA + 46, yA + 42, { size: 12, font: 'mono', weight: 600, color: moodA === 'sad' ? C.red : C.a, maxW: wA - 52 });
        V.text(g, TX.toks(nf(totA)), xA + 46, yA + 60, { size: 11, font: 'mono', color: C.muted, maxW: wA - 52 });
        g.globalAlpha = phase === 0 ? 0.5 : 1;
        V.box(g, xB, y, wB, cardH, { r: 12, fill: V.alpha(cTF, 0.08), stroke: V.alpha(cTF, 0.55), lw: 1.2 });
        critter(g, xB + 24, y + 30, 15, cTF, phase === 0 ? 'sleep' : phase === 2 ? 'happy' : 'neutral', t);
        V.text(g, TX.B(nf(S.P)), xB + 46, y + 22, { size: 12.5, weight: 600, color: C.ink, maxW: wB - 52 });
        if (phase === 0) V.text(g, TX.waitB, xB + 46, y + 42, { size: 11.5, color: C.muted, maxW: wB - 52 });
        else {
          const bw = wB - 58, by = y + 32;
          V.box(g, xB + 46, by, bw, 8, { r: 4, fill: V.alpha(C.muted, 0.2) });
          V.box(g, xB + 46, by, Math.max(4, bw * (posB / S.P)), 8, { r: 4, fill: cTF });
          V.text(g, phase === 1 ? TX.filling(nf(posB), nf(S.P)) : TX.done(secs(sim.ttft)), xB + 46, y + 56, { size: 11, font: 'mono', weight: phase === 2 ? 700 : 400, color: phase === 2 ? C.ink : C.muted, maxW: bw });
          if (phase === 1) V.text(g, TX.waited(secs(tau - sim.tArr)), xB + 46, y + 70, { size: 11, font: 'mono', color: C.muted, maxW: bw });
        }
        g.globalAlpha = 1;
        y += cardH + 24;

        /* magnifier: the model's last seconds */
        const Wm = 2.5 * sim.speed;
        V.text(g, TX.zoom(secs(Wm)), L, y, { size: 11.5, weight: 600, color: C.ink, maxW: W - 60 });
        V.text(g, TX.now + ' ▸', R, y, { size: 11, color: C.muted, align: 'right' });
        const zy = y + 8, zh = 44;
        const Z = (tt) => L + ((tt - (tau - Wm)) / Wm) * W;
        V.box(g, L, zy, W, zh, { r: 8, fill: V.alpha(C.muted, 0.07) });
        g.save(); V.rr(g, L, zy, W, zh, 8); g.clip();
        for (let j = k; j >= 0; j--) {
          const r = ev[j];
          if (r.t1 < tau - Wm) break;
          const x0 = Z(r.t0), x1 = Z(Math.min(r.t1, tau));
          if (x1 - x0 <= 0) continue;
          if (r.k === 'P') {
            V.box(g, x0 + 1, zy + 4, Math.max(1, x1 - x0 - 2), zh - 8, { r: 4, fill: V.alpha(cTF, 0.85) });
            if (x1 - x0 > 50) V.text(g, TX.P + ' ' + nf(r.n), Math.max(x0 + 26, x1 - 26), zy + zh / 2 + 4, { size: 11, font: 'mono', weight: 700, color: ink(C), align: 'center' });
          } else if (r.two) {
            V.box(g, x0 + 0.5, zy + 4, Math.max(1, x1 - x0 - 1), (zh - 8) / 2 - 1, { r: 2, fill: V.alpha(C.a, 0.85) });
            V.box(g, x0 + 0.5, zy + zh / 2 + 1, Math.max(1, x1 - x0 - 1), (zh - 8) / 2 - 1, { r: 2, fill: V.alpha(cTF, 0.55) });
          } else V.box(g, x0 + 0.5, zy + 4, Math.max(1, x1 - x0 - 1), zh - 8, { r: 2, fill: V.alpha(C.a, 0.85) });
        }
        g.restore();
        [[sim.tArr, TX.arr], [sim.tFirst, TX.ft]].forEach(([tm, lab]) => {
          if (tm > tau - Wm && tm <= tau) {
            const xx = Z(tm);
            V.line(g, xx, zy - 2, xx, zy + zh + 2, { color: C.ink, lw: 1.5, dash: [4, 3] });
            g.font = V.font(11);
            const tw = g.measureText(lab).width;
            V.text(g, lab, xx + 4 + tw > R ? xx - 4 : xx + 4, zy + zh + 14, { size: 11, color: C.ink, align: xx + 4 + tw > R ? 'right' : 'left' });
          }
        });

        /* the whole run */
        y = zy + zh + 36;
        V.text(g, TX.all, L, y, { size: 11.5, weight: 600, color: C.ink });
        const oy = y + 8, oh = 18;
        const O = (tt) => L + (tt / sim.tEnd) * W;
        V.box(g, L, oy, W, oh, { r: 5, fill: V.alpha(C.muted, 0.08) });
        for (let j = 0; j < ev.length; j++) {
          const r = ev[j], x0 = O(r.t0), x1 = O(r.t1), past = r.t1 <= tau;
          g.fillStyle = r.k === 'P' ? V.alpha(cTF, past ? 0.9 : 0.3) : V.alpha(C.a, past ? 0.8 : 0.25);
          g.fillRect(x0, oy + 2, Math.max(0.6, x1 - x0 - (x1 - x0 > 3 ? 0.6 : 0)), oh - 4);
        }
        const px = O(tau);
        V.line(g, px, oy - 4, px, oy + oh + 4, { color: C.c, lw: 2.5 });
        [[sim.tArr, TX.arr, 'left'], [sim.tFirst, TX.ft, 'right']].forEach(([tm, lab, al]) => {
          const xx = O(tm);
          V.line(g, xx, oy + oh, xx, oy + oh + 7, { color: C.ink, lw: 1.2 });
          V.text(g, lab, xx, oy + oh + 19, { size: 11, color: C.muted, align: al });
        });

        /* legend and note */
        y = oy + oh + 40;
        const leg = [[C.a, TX.legR], [cTF, TX.legP]];
        let lx = L;
        leg.forEach(([col, lab]) => {
          g.font = V.font(11);
          const lw = 16 + g.measureText(lab).width + 18;
          if (lx > L && lx + lw > R) { lx = L; y += 18; }
          V.box(g, lx, y - 9, 11, 11, { r: 3, fill: col });
          V.text(g, lab, lx + 16, y, { size: 11, color: C.muted });
          lx += lw;
        });
        V.wrap(g, TX.note, L, y + 20, W, { size: 11, color: C.muted, lineH: 14 });
      });
    }

    /* =====================================================================
       Figure 12 — TensorFold: the tray (batched window and multi-prompt chunk)
       ===================================================================== */
    {
      const fig = q('#fig-tf-bandeja');
      const TX = {
        mode: 'Tray', mRound: 'decode round', mChunk: 'prompt chunk',
        streams: 'decoding streams', copy: 'A quotes a passage (15 copy drafts)', dice: '🎲 new round',
        rP: 'proposed rows', rV: 'verified rows', rK: 'committed tokens',
        scen: 'Scenario', sFour: '4 prompts of 130 together', sLong: 'a 4,000 ahead of three 130s', again: '▶ again',
        rL: 'last 1st token', rM: 'mean to the 1st token',
        vL: (f, c, r) => 'after ' + f + ' forwards (' + c + ' P + ' + r + ' R)', vM: (v) => v + ' forwards',
        ph: [
          '1 · each stream proposes its window: pending token + drafts (up to 16 rows)',
          '2 · the sum goes past 32: one draft comes off the longest window, one at a time',
          '3 · a single forward for everyone, through the CUDA graph for ',
          '4 · each stream accepts while draft = its keyed sample',
        ],
        noCut: '2 · the sum fits in 32 rows: nothing to cut',
        lines: ' rows', props: 'proposals per stream', win: 'batched window (MAX_WINDOW = 32)', graph: (n) => 'graph[' + n + ']',
        legPend: 'pending', legCopy: 'copy draft', legDf: 'DFlash2', legOk: 'accepted', legBad: 'differs: stops here',
        plus: (n) => '+' + n + ' tok',
        multiOn: 'MULTI_PREFILL=1: several prompts per chunk, fewest rows left first',
        multiOff: 'MULTI_PREFILL=0: one prompt per chunk, oldest first (FIFO)',
        queue: 'prompts', tl: 'rank 0 iterations (P = prompt chunk, with each prompt\'s rows; R = decode round)',
        tok: ' tokens', first: (i) => '1st token at #' + i, wait: 'queued', rowsL: (n) => n,
      };
      const NM = ['A', 'B', 'C', 'D'];
      const SC = (C) => [C.a, C.b, C.c, C.d];
      const S = { mode: 'round', n: 4, copy: true, seed: 1, scen: 'four', multi: true, t0: 0, tNow: 0, rd: null, ch: null };
      /* stage height by width and by mode (chunk mode needs more rows on a phone) */
      const stB = fig.querySelector('.la-fig-stage');
      const setH = () => {
        const sw = stB.clientWidth || 800, nar = sw < 700;
        let hh = 340;
        if (nar && S.mode !== 'chunk') hh = 360;
        else if (nar) {   /* the same chip layout as the draw: one row of chips, or more */
          const W = sw - 28, m = S.ch ? S.ch.it.length : 16, gap = 6;
          const perRow = (W - gap * (m - 1)) / m < 34 ? Math.max(1, Math.floor((W + gap) / 40)) : m;
          hh = 350 + (Math.ceil(m / perRow) - 1) * 118;
        }
        const want = hh + 'px';
        if (stB.style.height !== want) stB.style.height = want;
      };
      const roB = new ResizeObserver(setH); roB.observe(stB);
      if (LA.onCleanup) LA.onCleanup(() => roB.disconnect());
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: TX.mode, options: [{ v: 'round', label: TX.mRound }, { v: 'chunk', label: TX.mChunk }], value: 'round', onChange: (v) => { S.mode = v; gR.show(v === 'round'); gC.show(v === 'chunk'); setH(); restart(); } });
      const gR = UI.group(ctl), gC = UI.group(ctl);
      UI.seg(gR.el, { label: TX.streams, options: [1, 2, 3, 4].map((v) => ({ v, label: String(v) })), value: 4, onChange: (v) => { S.n = v; buildRound(); restart(); } });
      UI.toggle(gR.el, { label: TX.copy, value: true, onChange: (v) => { S.copy = v; buildRound(); restart(); } });
      UI.button(gR.el, TX.dice, () => { S.seed++; buildRound(); restart(); });
      const rP = UI.readout(gR.el, TX.rP, '—'), rV = UI.readout(gR.el, TX.rV, '—'), rK = UI.readout(gR.el, TX.rK, '—');
      UI.select(gC.el, { label: TX.scen, options: [{ v: 'four', label: TX.sFour }, { v: 'long', label: TX.sLong }], value: 'four', onChange: (v) => { S.scen = v; buildChunk(); setH(); restart(); } });
      UI.toggle(gC.el, { label: 'MULTI_PREFILL', value: true, onChange: (v) => { S.multi = v; buildChunk(); setH(); restart(); } });
      UI.button(gC.el, TX.again, () => restart());
      const rL = UI.readout(gC.el, TX.rL, '—'), rM = UI.readout(gC.el, TX.rM, '—');
      gC.show(false);
      function restart() { S.t0 = S.tNow; }

      /* round: proposals, cut to MAX_WINDOW (as trim() in multi.py) and illustrative acceptance */
      function buildRound() {
        const r = V.rng(1009 * S.seed + 17 * S.n + (S.copy ? 3 : 0));
        const props = [];
        for (let i = 0; i < S.n; i++) props.push(i === 0 && S.copy ? { d: 15, copy: true } : { d: 3 + Math.floor(r() * 5), copy: false });
        const keep = props.map((p) => p.d), cuts = [];
        let total = keep.reduce((s, d) => s + 1 + d, 0);
        const proposed = total;
        while (total > 32) {
          let kk = 0;
          for (let i = 0; i < keep.length; i++) if (keep[i] >= keep[kk]) kk = i;   /* the longest; the youngest on a tie */
          if (!keep[kk]) break;
          cuts.push({ i: kk, j: keep[kk] }); keep[kk]--; total--;
        }
        const acc = keep.map((d, i) => { let a = 0; const pr = props[i].copy ? 0.85 : 0.62; while (a < d && r() < pr) a++; return a; });
        const off = []; let o = 0;
        keep.forEach((d) => { off.push(o); o += 1 + d; });
        S.rd = { props, keep, cuts, acc, proposed, total, off };
        rP.set(String(proposed), proposed > 32 ? 'warn' : null);
        rV.set(total + ' → ' + TX.graph(total), 'ok');
        rK.set(String(acc.reduce((s, a) => s + 1 + a, 0)));
      }
      /* chunk: _fill_turn + _stop + _group from multi.py / multi_prefill.py, with kept points on the 64 grid */
      function buildChunk() {
        const lens = S.scen === 'four' ? [130, 130, 130, 130] : [4000, 130, 130, 130];
        const lanes = lens.map((n, i) => { const gp = Math.floor(n / 64) * 64; return { i, n, pos: 0, dec: false, stops: gp > 0 && gp < n ? [gp] : [], first: -1 }; });
        const nextStop = (l) => { const s = l.stops.find((p) => p > l.pos); return s == null ? l.n : s; };
        const it = [];
        let credit = 0;
        for (let guard = 0; guard < 60; guard++) {
          const filling = lanes.filter((l) => !l.dec), decoding = lanes.filter((l) => l.dec);
          if (!filling.length) break;
          if (decoding.length) {
            credit += 0.5;
            if (credit < 1) { it.push({ k: 'R', s: decoding.map((l) => l.i) }); continue; }
            credit -= 1;
          }
          const busy = decoding.length > 0;
          let pieces = null;
          if (S.multi) {
            let left = busy ? 1024 : 2048;
            const order = filling.slice().sort((a, b) => (a.n - a.pos) - (b.n - b.pos) || a.i - b.i);
            const out = [];
            for (const l of order) {
              let stop = nextStop(l);
              if (stop - l.pos > left) stop = l.pos + Math.floor(left / 64) * 64;
              if (stop <= l.pos) continue;
              out.push({ l, a: l.pos, b: stop });
              left -= stop - l.pos;
              if (left <= 0) break;
            }
            if (out.length > 1) pieces = out;
          }
          if (!pieces) { const l = filling[0]; pieces = [{ l, a: l.pos, b: Math.min(nextStop(l), l.pos + (busy ? 1024 : 2048)) }]; }
          it.push({ k: 'P', p: pieces.map((p) => ({ i: p.l.i, a: p.a, b: p.b, end: p.b === p.l.n })) });
          pieces.forEach((p) => { p.l.pos = p.b; if (p.b === p.l.n) { p.l.dec = true; p.l.first = it.length - 1; } });
        }
        it.push({ k: 'R', s: lanes.map((l) => l.i) });
        const last = Math.max(...lanes.map((l) => l.first));
        const upto = it.slice(0, last + 1);
        S.ch = { it, lens, firsts: lanes.map((l) => l.first) };
        rL.set(TX.vL(last + 1, upto.filter((x) => x.k === 'P').length, upto.filter((x) => x.k === 'R').length), last + 1 > 4 ? 'warn' : 'ok');
        rM.set(TX.vM(nf(lanes.reduce((s, l) => s + l.first + 1, 0) / lanes.length, 1)));
      }
      buildRound(); buildChunk(); setH();

      function cell(g, x, y, s, col, kind, a) {
        const C = V.C();
        g.globalAlpha = a == null ? 1 : a;
        if (kind === 'pend') V.box(g, x, y, s, s, { r: 3, fill: col });
        else if (kind === 'copy') { V.box(g, x, y, s, s, { r: 3, fill: V.alpha(col, 0.22) }); g.setLineDash([3, 2]); g.strokeStyle = col; g.lineWidth = 1.4; V.rr(g, x + 0.7, y + 0.7, s - 1.4, s - 1.4, 3); g.stroke(); g.setLineDash([]); }
        else V.box(g, x, y, s, s, { r: 3, fill: V.alpha(col, 0.5) });
        g.globalAlpha = 1;
        return C;
      }

      V.fig(fig, (g, w, h, t) => {
        if (w < 60 || h < 60) return;
        S.tNow = t;
        const C = V.C(), cTF = TFC(C), cols = SC(C);
        const narrow = w < 700;
        const L = 14, R = w - 14, W = R - L;
        const el = t - S.t0;
        if (S.mode === 'round') {
          const rd = S.rd, nC = rd.cuts.length;
          const T1 = 1.2, cutStep = nC ? Math.min(0.28, 2.2 / nC) : 0.6, T2 = T1 + Math.max(0.6, nC * cutStep) + 0.2, T3 = T2 + 1.1, T4 = T3 + 1.8, TEND = T4 + 2.4;
          let tt = el;
          if (tt > TEND) { S.t0 = t; tt = 0; }
          const ph = tt < T1 ? 0 : tt < T2 ? 1 : tt < T3 ? 2 : 3;
          const capt = ph === 1 && !nC ? TX.noCut : ph === 2 ? TX.ph[2] + rd.total + TX.lines : TX.ph[ph];
          let y = 18;
          y += V.wrap(g, capt, L, y, W, { size: 12.5, weight: 600, color: C.ink, lineH: 16 }) + 8;
          /* proposals (full width) */
          const cs = Math.max(12, Math.min(26, Math.floor((W - 114) / 16) - 3)), cstep = cs + 3;
          const rowH = cs + 10;
          V.text(g, TX.props, L, y + 2, { size: 11.5, weight: 600, color: C.muted });
          const ry0 = y + 12;
          const cutDone = (i, j) => { const ci = rd.cuts.findIndex((c) => c.i === i && c.j === j); if (ci < 0) return -1; return ci; };
          for (let i = 0; i < S.n; i++) {
            const yy = ry0 + i * rowH, col = cols[i], p = rd.props[i];
            const accShown = ph === 3 && tt > T3 + 0.9;
            critter(g, L + 11, yy + cs / 2, 9, col, accShown ? 'happy' : 'neutral', t);
            V.text(g, NM[i], L + 24, yy + cs / 2 + 4, { size: 11.5, font: 'mono', weight: 700, color: col });
            const n = 1 + p.d;
            const vis = ph === 0 ? Math.ceil(n * clamp(tt / (T1 * 0.85), 0, 1)) : n;
            for (let j = 0; j < vis; j++) {
              const x = L + 38 + j * cstep;
              const kind = j === 0 ? 'pend' : p.copy ? 'copy' : 'df';
              const ci = cutDone(i, j);
              let a = 1, dy = 0, cut = false;
              if (ci >= 0 && ph >= 1) {
                const tc0 = T1 + 0.1 + ci * cutStep;
                if (tt >= tc0 + cutStep * 1.6 || ph >= 2) continue;
                if (tt >= tc0) { cut = true; const f = clamp((tt - tc0) / (cutStep * 1.6), 0, 1); dy = f * f * 18; a = 1 - f; }
              }
              if (ph >= 2 && j > rd.keep[i]) continue;
              const moved = ph >= 2 ? clamp((tt - T2 - (rd.off[i] + j) * 0.012) / 0.6, 0, 1) : 0;
              cell(g, x, yy + dy, cs, cut ? C.red : col, kind, a * (1 - 0.65 * moved));
            }
            if (accShown) V.text(g, TX.plus(1 + rd.acc[i]), L + 38 + (1 + rd.keep[i]) * cstep + 6, yy + cs / 2 + 4, { size: 11.5, font: 'mono', weight: 700, color: C.ink });
          }
          /* batched window: 32 slots (one row if it fits) */
          const perW = W >= 32 * 17 ? 32 : 16;
          const ws = Math.max(12, Math.min(26, Math.floor(W / perW) - 3)), wstep = ws + 3;
          const wx = L, wy = ry0 + S.n * rowH + 26, wBottom = wy + (32 / perW) * (wstep + 4);
          const slotXY = (s) => [wx + (s % perW) * wstep, wy + Math.floor(s / perW) * (wstep + 4)];
          V.text(g, TX.win, wx, wy - 10, { size: 11.5, weight: 600, color: C.muted, maxW: R - wx });
          for (let s = 0; s < 32; s++) {
            const [x, yy] = slotXY(s);
            g.setLineDash([3, 3]); g.strokeStyle = V.alpha(C.muted, 0.5); g.lineWidth = 1; V.rr(g, x + 0.5, yy + 0.5, ws - 1, ws - 1, 3); g.stroke(); g.setLineDash([]);
          }
          if (ph >= 2) {
            for (let i = 0; i < S.n; i++) {
              for (let j = 0; j <= rd.keep[i]; j++) {
                const s = rd.off[i] + j;
                const f = V.ease.inOut(clamp((tt - T2 - s * 0.012) / 0.6, 0, 1));
                const sx = L + 38 + j * cstep, sy = ry0 + i * rowH;
                const [tx, ty] = slotXY(s);
                const x = V.lerp(sx, tx, f), yy = V.lerp(sy, ty, f), sz = V.lerp(cs, ws, f);
                const kind = j === 0 ? 'pend' : rd.props[i].copy ? 'copy' : 'df';
                let col = cols[i];
                if (ph === 3) {
                  const rev = clamp((tt - T3) / 1.2, 0, 1) * 16;
                  if (j <= rev) {
                    if (j === 0 || j <= rd.acc[i]) { cell(g, x, yy, sz, col, kind); V.text(g, '✓', x + sz / 2, yy + sz / 2 + 4, { size: 11, weight: 700, color: kind === 'pend' ? ink(C) : C.ink, align: 'center' }); continue; }
                    if (j === rd.acc[i] + 1) { cell(g, x, yy, sz, C.red, 'pend', 0.9); V.text(g, '✗', x + sz / 2, yy + sz / 2 + 4, { size: 11, weight: 700, color: ink(C), align: 'center' }); continue; }
                    cell(g, x, yy, sz, C.muted, 'df', 0.45); continue;
                  }
                }
                cell(g, x, yy, sz, col, kind);
              }
            }
            const gx = wx, gy2 = wBottom + 6;
            g.font = V.font(12, 'mono', 700);
            const lab = TX.graph(rd.total), lw = g.measureText(lab).width + 16;
            V.box(g, gx, gy2, lw, 22, { r: 11, fill: V.alpha(cTF, 0.9), label: lab, size: 12, font: 'mono', weight: 700, color: ink(C) });
            V.text(g, rd.total + ' / 32' + TX.lines, gx + lw + 10, gy2 + 15, { size: 11.5, font: 'mono', color: C.muted, maxW: R - gx - lw - 12 });
          }
          /* legend */
          const ly = wBottom + 50;
          let lx = L, yl = ly;
          const legs = [['pend', cols[0], TX.legPend], ['copy', cols[0], TX.legCopy], ['df', cols[1], TX.legDf], ['ok', C.ink, TX.legOk], ['bad', C.red, TX.legBad]];
          legs.forEach(([kd, col, lab]) => {
            g.font = V.font(11);
            const lw = 18 + g.measureText(lab).width + 16;
            if (lx > L && lx + lw > R) { lx = L; yl += 18; }
            if (kd === 'ok') V.text(g, '✓', lx + 6, yl, { size: 12, weight: 700, color: col, align: 'center' });
            else if (kd === 'bad') { cell(g, lx, yl - 10, 12, col, 'pend', 0.9); }
            else cell(g, lx, yl - 10, 12, col, kd);
            V.text(g, lab, lx + 18, yl, { size: 11, color: C.muted });
            lx += lw;
          });
        } else {
          /* ---------------- prompt chunk ---------------- */
          const ch = S.ch, it = ch.it, m = it.length, per = 0.75, TEND = m * per + 2.6;
          let tt = el;
          if (tt > TEND) { S.t0 = t; tt = 0; }
          const cur = Math.min(m - 1, Math.floor(tt / per)), prog = clamp((tt - cur * per) / (per * 0.8), 0, 1);
          let y = 18;
          y += V.wrap(g, S.multi ? TX.multiOn : TX.multiOff, L, y, W, { size: 12.5, weight: 600, color: C.ink, lineH: 16 }) + 6;
          /* prompt queue with progress */
          const pos = ch.lens.map(() => 0);
          for (let j = 0; j <= cur; j++) if (it[j].k === 'P') it[j].p.forEach((p) => { pos[p.i] = j < cur ? p.b : V.lerp(p.a, p.b, prog); });
          const rowH = 24, barX = L + (narrow ? 112 : 150), barW = Math.min(narrow ? W - 112 - 4 : 300, W - (barX - L) - 4);
          V.text(g, TX.queue, L, y + 4, { size: 11.5, weight: 600, color: C.muted });
          const qy = y + 12;
          ch.lens.forEach((n, i) => {
            const yy = qy + i * rowH, col = cols[i], f = ch.firsts[i], got = f >= 0 && (cur > f || (cur === f && prog >= 1));
            critter(g, L + 10, yy + 9, 8, col, got ? 'happy' : pos[i] > 0 ? 'neutral' : 'sleep', t);
            V.text(g, NM[i] + ' · ' + nf(n) + TX.tok, L + 24, yy + 13, { size: 11, font: 'mono', color: C.ink, maxW: barX - L - 30 });
            V.box(g, barX, yy + 4, barW, 10, { r: 5, fill: V.alpha(C.muted, 0.18) });
            V.box(g, barX, yy + 4, Math.max(3, barW * (pos[i] / n)), 10, { r: 5, fill: col });
            if (!narrow) V.text(g, got ? TX.first(f + 1) : TX.wait, barX + barW + 10, yy + 13, { size: 11, font: 'mono', weight: got ? 700 : 400, color: got ? C.ink : C.muted });
            else if (got) V.text(g, '✓', barX + barW - 2, yy + 13, { size: 11, weight: 700, color: ink(C), align: 'right' });
          });
          y = qy + 4 * rowH + 14;
          /* iteration timeline */
          const tlH = V.wrap(g, TX.tl, L, y, W, { size: 11.5, weight: 600, color: C.muted, lineH: 15 });
          y += tlH - 15;
          const gap = 6, minCW = narrow ? 34 : 44;
          let perRow = m, cw = Math.min(92, (W - gap * (m - 1)) / m);
          if (cw < minCW) { perRow = Math.max(1, Math.floor((W + gap) / (minCW + gap))); cw = (W - gap * (perRow - 1)) / perRow; }
          const chH = narrow ? 92 : 104;
          const ty0 = y + 10;
          for (let j = 0; j < m; j++) {
            const x = L + (j % perRow) * (cw + gap), yy = ty0 + Math.floor(j / perRow) * (chH + 26), x1 = it[j];
            const a = j < cur ? 0.9 : j === cur ? 1 : 0.22;
            g.globalAlpha = a;
            if (x1.k === 'P') {
              V.box(g, x, yy, cw, chH, { r: 8, fill: V.alpha(cTF, 0.08), stroke: V.alpha(cTF, j === cur ? 1 : 0.6), lw: j === cur ? 2.2 : 1.2 });
              V.text(g, 'P', x + 6, yy + 14, { size: 11.5, font: 'mono', weight: 700, color: cTF });
              V.text(g, (cw >= 44 ? '#' : '') + (j + 1), x + cw - 5, yy + 14, { size: 11, font: 'mono', color: C.muted, align: 'right' });
              const np = x1.p.length, bh = Math.min(20, (chH - 36) / Math.max(np, 1) - 2);
              x1.p.forEach((p, pi) => {
                const by = yy + 21 + pi * (bh + 2);
                V.box(g, x + 4, by, cw - 8, bh, { r: 3, fill: V.alpha(cols[p.i], 0.85) });
                const lab = cw > 62 ? NM[p.i] + ' ' + nf(p.b - p.a) : NM[p.i];
                if (bh >= 11) V.text(g, lab, x + cw / 2, by + bh / 2 + 4, { size: 11, font: 'mono', weight: 700, color: ink(C), align: 'center', maxW: cw - 10 });
              });
              const rows = x1.p.reduce((s, p) => s + p.b - p.a, 0);
              V.text(g, TX.rowsL(nf(rows)), x + cw / 2, yy + chH - 6, { size: 11, font: 'mono', color: C.muted, align: 'center', maxW: cw - 6 });
            } else {
              V.box(g, x, yy, cw, chH, { r: 8, fill: V.alpha(C.muted, 0.07), stroke: V.alpha(C.muted, j === cur ? 1 : 0.5), lw: j === cur ? 2.2 : 1.2 });
              V.text(g, 'R', x + 6, yy + 14, { size: 11.5, font: 'mono', weight: 700, color: C.muted });
              V.text(g, (cw >= 44 ? '#' : '') + (j + 1), x + cw - 5, yy + 14, { size: 11, font: 'mono', color: C.muted, align: 'right' });
              x1.s.forEach((si, k2) => { const dx = x + cw / 2 + (k2 - (x1.s.length - 1) / 2) * Math.min(14, (cw - 8) / Math.max(1, x1.s.length)); V.dot(g, dx, yy + chH / 2, Math.min(5.5, (cw - 8) / 8), cols[si]); });
            }
            g.globalAlpha = 1;
            /* who got the 1st token in this iteration */
            const got = ch.firsts.map((f, i) => (f === j ? i : -1)).filter((i) => i >= 0);
            if (got.length && j <= cur) V.text(g, got.map((i) => NM[i]).join('') + ' ✓', x + cw / 2, yy + chH + 15, { size: 11, font: 'mono', weight: 700, color: C.ink, align: 'center', maxW: cw + gap });
          }
        }
      });
    }
  },
});

LA.chapter({
  id: 'paged-attention-kv',
  part: 6,
  order: 31,
  title: 'PagedAttention and the KV cache budget',
  navTitle: 'PagedAttention & KV',
  subtitle: 'Blocks, tables and slots; the manager and kernel block sizes; the hybrid allocator and the uniform page; the profile run, the workspaces and the byte pin — and how to read “GPU KV cache size” without fooling yourself. At the end, TensorFold’s counterpoint: contiguous extents and a budget based on free memory.',
  minutes: 62,
  recipes: ['glm', 'tf', 'qwen', 'ds'],
  tags: ['PagedAttention', 'KV cache', 'block table', 'block size', 'hybrid KV', 'profile run', 'workspaces', 'max_model_len', 'TensorFold', 'extents'],
  goals: [
    'Explain how <b>PagedAttention</b> stores the KV in blocks located through <b>block tables</b> and <b>slot mapping</b>, and why this nearly eliminates fragmentation, and compare it with TensorFold’s contiguous <b>extents</b>',
    'Predict the consequences of a <code>block_size</code> and tell apart the <b>manager</b>, <b>kernel</b> and <b>scheduler</b> (LCM) blocks',
    'Understand the <b>hybrid allocator</b>: KV groups, one pool of global IDs, the <b>uniform page</b> and why GLM uses 3,584-token blocks',
    'Redo the <b>profile run</b> math (weights, activations, workspaces, graphs) and choose between <code>gpu_<wbr>memory_<wbr>utilization</code> and <code>--kv-cache-memory-bytes</code>',
    'Read <code>GPU KV cache size</code> and <code>Maximum concurrency</code> correctly and choose <code>max_model_len</code> and <code>max_num_seqs</code> as ceilings',
  ],
  prereqs: ['vllm-arquitetura', 'kv-cache-quantizado', 'atencao-linear'],
  summary: [
    'PagedAttention: the KV lives in blocks of <code>block_size</code> tokens scattered across the pool; each request has a block table and each token a slot = ID × B + offset. Blocks are created on demand, the leftover is limited to the last block, and ID 0 is the null block. TensorFold (GLM·TF recipe) does the opposite: each stream has a contiguous range in multiples of 2,048 tokens that grows in place, moves or gets compacted, and the KDA state sits in slots outside the pool.',
    'The index → slot translation needs guards: in DeepSeek’s Stage-C overlay (lane B, the readable code; the default Anemll lane has equivalent logic outside the repo), the kernel drops padding tokens, indices past the row and IDs outside the pool (<code>compute_<wbr>global_<wbr>topk_<wbr>indices_<wbr>and_<wbr>lens</code>); GLM clamps the KpoolTail ring index, because the table is row-major and running past the row writes into another request’s block.',
    'The <code>block_size</code> is the grain of the leftover, of the table size, of kernel efficiency and of prefix hits. The manager block can be larger than the kernel block (GLM: 3,584 vs 64), and the scheduler block is the LCM of the groups (GLM 3,584, DeepSeek 256).',
    'In the hybrid KV, groups of layers with the same <code>KVCacheSpec</code> take IDs from a single BlockPool with pages of the same size: the attention block grows until the page fits the state (GLM 2,342,912 B → 3,584 tokens; Qwen 1,664 in the default, 1,600 to 3,200 depending on the KV and state dtypes) or until it matches the SWA page (DeepSeek 37,376 B → 256). Each ID pays for one page in every tensor (GLM: 11 MLA + 11 indexer ≈ 25.9 MiB). IDs, not bytes, are the currency.',
    'Profile run: KV = util × memory − weights − outside PyTorch − activation peak − graph estimate; workspaces locked at <code>lock_workspace()</code> come out of the KV (GLM indexer: 5,036 MiB → 126 MiB with rightsize). <code>--kv-cache-memory-bytes</code> pins the pool (GLM: 11 GiB because of InstantTensor). TensorFold skips the profile run: budget = MemAvailable at startup − reserve (14.5 GiB), estimated with formulas and refused before loading the weights if it doesn’t fit; the pool absorbs whatever memory is missing at startup (2,922,496 tokens measured) and the reserve fixes the headroom’s floor.',
    '<code>GPU KV cache size</code> = maximum concurrency × <code>max_model_len</code>. In GLM, 643 ÷ 414 × 1M = 1,553,140 “tokens”, but only ~57,344 tokens of stored conversation fit; <code>patch_<wbr>kv_<wbr>capacity_<wbr>log.<wbr>py</code> prints the honest math.',
    '<code>max_model_len</code> and <code>max_num_seqs</code> are ceilings: what holds is Σ live tokens ≲ pool (DeepSeek: a 1M ceiling and 6 slots with a 2.33M pool). The ceiling still costs: the boot demands one whole request, per-step buffers grow with it, and above the config you need YaRN.',
  ],
  quiz: [
    { q: 'Why does paging eliminate <i>external</i> KV fragmentation?', o: ['Because it compresses each request’s KV until it fits in the free holes', 'Because all blocks have the same size, so any free block works for any request', 'Because it reserves max_model_len for each request right at admission', 'Because the attention kernel starts reading the KV from a contiguous range'], a: 1, e: 'With single-size allocations, there is no such thing as a “hole too small”: the next request uses any free block, anywhere. Only the internal fragmentation of the last block remains (≤ B − 1 tokens). Reserving max_model_len is exactly what paging avoids, and the kernel reads block by block, without contiguous memory.' },
    { q: 'DeepSeek’s boot says <code>Maximum concurrency for 1,<wbr>048,<wbr>576 tokens per request: 2.<wbr>22x</code>, and the recipe uses <code>MAX_NUM_SEQS=6</code>. What does that mean?', o: ['That only 2 requests run at the same time and the other 4 slots sit idle', 'That vLLM reserves 6 × 1M tokens at boot and rejects anything beyond that', 'That max_model_len has to drop to ~350k for the 6 slots to fit', 'That ~2 whole 1M requests fit, and 6 can coexist if the sum of live tokens fits in the pool'], a: 3, e: 'max_model_len and max_num_seqs are ceilings, not reservations. The pool (2,331,430 tokens) is shared: 6 × 200k = 1.2M fit; 6 × 1M = 6M don’t, and the extra ones wait in the queue or get preempted. Nothing needs to be reduced.' },
    { q: 'Why does GLM’s MLA manager use 3,584-token blocks?', o: ['The common page has to hold the KDA state: 2,342,912 B ÷ 656 B = 3,571.5, rounded up to a multiple of 256', 'It is the page size GLM’s sparse-MLA kernels were compiled with on SM12x', 'It is the recipe’s max_num_batched_tokens divided by the number of TP ranks', 'It is the only block DeepGEMM paged-MQA accepts for the indexer on sm_120'], a: 0, e: 'The hybrid allocator requires the same page in bytes in every group. The attention block grows to 3,584 × 656 = 2,351,104 B ≥ KDA state; the 256 comes from the indexer alignment (index_kpool × 64). The kernel keeps its 64-token page (that’s 56 kernel pages per block), and DeepGEMM only requires block_kv 64.' },
    { q: 'In GLM, the line <code>GPU KV cache size: 1,<wbr>553,<wbr>140 tokens</code> came from a pool of 643 IDs. How much stored conversation actually fit?', o: ['1,553,140 tokens, exactly what the boot line announces', '643 × 3,584 ≈ 2.3M tokens, one full MLA block per ID', '~57 thousand tokens: 16 segments of 3,584 tokens', '414 × 3,584 ≈ 1.48M tokens, the 1M request plus headroom'], a: 2, e: 'The line is (643/414) × 1M: concurrency in token units. Storing one 3,584-token segment costs 38 IDs (1 MLA + 4 Mamba + 33 for the drafter), so 642 usable IDs ÷ 38 = 16 segments = 57,344 tokens.' },
    { q: 'What happens to <code>gpu_<wbr>memory_<wbr>utilization</code> when you pass <code>--kv-cache-memory-bytes</code>?', o: ['It is added to the pin, and the pool becomes util × memory + pin', 'It no longer applies to the KV: the pool has the requested size, and host headroom is up to you', 'vLLM refuses the two together and aborts the boot with a configuration error', 'It then only limits the weights, and the KV grows until memory runs out'], a: 1, e: 'The pin replaces the profile run math for the KV. That’s what GLM does with 11 GiB, because InstantTensor skews the profiling; the check that the pool can hold one max_model_len request still applies.' },
    { q: 'GLM’s KpoolTail bug corrupted another request’s indexer from position 4 on. Why?', o: ['The ring only had 3 positions, and the fourth write wrapped around too early', 'The null block was overwritten by the tail of every request in the batch', 'The prefix cache shared the tail block between two requests', 'That table’s row has width 1 and is row-major: pos // 4 ≥ 1 reads the next request’s entry'], a: 3, e: 'The generic kernel did block_table[req, pos // block_size] with no bound. With width 1, any index ≥ 1 lands in another request’s row. The clamp min(b, stride − 1) turns the address into the intended ring: block_table[req, 0] × 4 + pos mod 4. The ring has 4 positions, and the tail doesn’t even take part in the prefix cache.' },
  ],
  mount(root, LA) {
    const V = LA.viz, UI = LA.ui;
    const nf = (x) => Math.round(x).toLocaleString('en-US');
    const sf = (x, d = 2) => (+x).toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d });
    let fcache = {};
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(() => { fcache = {}; });
    const font = (s, f, w) => { const k = s + '|' + f + '|' + w; return fcache[k] || (fcache[k] = V.font(s, f, w)); };
    const tw = (g, s, size = 12, f = 'sans', w = 400) => { g.font = font(size, f, w); return g.measureText(String(s)).width; };
    const RED = (C) => C.red || C.c;
    function pill(g, x, y, s, col, o = {}) {
      const size = o.size || 11, f = o.font || 'mono', wt = o.weight || 600;
      const wd = tw(g, s, size, f, wt) + (o.padX || 12), ht = size + 8;
      const xx = o.align === 'center' ? x - wd / 2 : o.align === 'right' ? x - wd : x;
      V.box(g, xx, y - ht / 2, wd, ht, { r: ht / 2, fill: V.alpha(col, o.a == null ? 0.18 : o.a), stroke: V.alpha(col, 0.8), lw: 1 });
      V.text(g, s, xx + wd / 2, y + 0.5, { size, font: f, weight: wt, color: o.color || V.C().ink, align: 'center', base: 'middle' });
      return wd;
    }
    function hatch(g, x, y, w, h, col, step = 6) {
      if (w <= 0 || h <= 0) return;
      g.save(); g.beginPath(); g.rect(x, y, w, h); g.clip();
      g.strokeStyle = col; g.lineWidth = 1; g.beginPath();
      for (let k = -h; k < w; k += step) { g.moveTo(x + k, y + h); g.lineTo(x + k + h, y); }
      g.stroke(); g.restore();
    }
    function wrapLines(g, s, maxW, size = 12, f = 'sans', wt = 400) {
      g.font = font(size, f, wt);
      const words = String(s).split(' '); const out = []; let cur = '';
      words.forEach((wd) => { const test = cur ? cur + ' ' + wd : wd; if (cur && g.measureText(test).width > maxW) { out.push(cur); cur = wd; } else cur = test; });
      if (cur) out.push(cur);
      return out;
    }
    function wrap(g, s, x, y, maxW, o = {}) {
      const size = o.size || 12, lh = o.lh || Math.round(size * 1.4);
      const ls = wrapLines(g, s, maxW, size, o.font || 'sans', o.weight || 400);
      ls.forEach((ln, i) => V.text(g, ln, x, y + i * lh, Object.assign({}, o, { size })));
      return ls.length * lh;
    }
    function fitH(H, need) {
      const st = H.canvas.parentElement;
      const cur = parseFloat(st.style.height) || st.getBoundingClientRect().height;
      need = Math.round(need);
      if (Math.abs(cur - need) > 4) st.style.height = need + 'px';
    }
    const onColor = (C) => (C.dark ? 'rgba(0,0,0,.8)' : 'rgba(255,255,255,.97)');

    /* ======================================================================
       FIG 1 — contiguous reservation × paging (simulation)
       ====================================================================== */
    function figFrag() {
      const fig = root.querySelector('#fig-frag');
      const ctl = UI.controls(fig);
      const CAP = 96, NREQ = 16, TICK = 0.3, LET = 'ABCDEFGHIJKLMNOP';
      const S = { B: 4, maxLen: 24, seed: 5 };
      let reqs = [], A = null, P = null, tick = 0, acc = 0, hold = 0;
      function gen() {
        const r = V.rng(S.seed * 7919 + 13);
        reqs = []; let at = 0;
        for (let i = 0; i < NREQ; i++) {
          at += 1 + r.int(3);
          const p = 2 + r.int(4);
          const cap = Math.min(S.maxLen, p + 3 + r.int(Math.max(1, S.maxLen - p - 2)));
          const tgt = Math.min(cap, p + 1 + Math.floor(Math.pow(r(), 1.3) * (cap - p)));
          reqs.push({ id: i, at, p, cap, tgt });
        }
      }
      function reset() {
        gen(); tick = 0; acc = 0; hold = 0;
        A = { owner: new Int16Array(CAP).fill(-1), run: [], queue: [], done: 0, su: 0, sr: 0, n: 0, qsum: 0 };
        const nb = CAP / S.B;
        P = { bown: new Int16Array(nb).fill(-1), free: Array.from({ length: nb }, (_, i) => i), run: [], queue: [], done: 0, su: 0, sr: 0, n: 0, qsum: 0, pre: 0, seq: 0 };
      }
      function stepC() {
        reqs.forEach((q) => { if (q.at === tick) A.queue.push(q.id); });
        A.run = A.run.filter((x) => {
          if (x.len >= reqs[x.id].tgt) { for (let k = x.start; k < x.start + reqs[x.id].cap; k++) A.owner[k] = -1; A.done++; return false; }
          x.len++; return true;
        });
        while (A.queue.length) {
          const id = A.queue[0], need = reqs[id].cap;
          let start = -1, run = 0;
          for (let k = 0; k < CAP; k++) { if (A.owner[k] === -1) { run++; if (run === need) { start = k - need + 1; break; } } else run = 0; }
          if (start < 0) break;
          for (let k = start; k < start + need; k++) A.owner[k] = id;
          A.run.push({ id, start, len: reqs[id].p });
          A.queue.shift();
        }
        A.su += A.run.reduce((s, x) => s + x.len, 0); A.sr += A.run.reduce((s, x) => s + reqs[x.id].cap, 0); A.n++; A.qsum += A.queue.length;
      }
      function pFree(x) { x.blocks.forEach((b) => { P.bown[b] = -1; P.free.push(b); }); x.blocks = []; }
      function stepP() {
        reqs.forEach((q) => { if (q.at === tick) P.queue.push(q.id); });
        P.run = P.run.filter((x) => { if (x.len >= reqs[x.id].tgt) { pFree(x); P.done++; return false; } return true; });
        const order = P.run.slice().sort((a, b) => a.seq - b.seq);
        for (const x of order) {
          if (P.run.indexOf(x) < 0) continue;
          const need = Math.ceil((x.len + 1) / S.B);
          if (need > x.blocks.length) {
            if (!P.free.length) {
              const victim = P.run.reduce((m, y) => (y.seq > m.seq ? y : m), P.run[0]);
              pFree(victim); P.run = P.run.filter((y) => y !== victim); P.queue.unshift(victim.id); P.pre++;
              if (victim === x) continue;
            }
            const b = P.free.shift(); P.bown[b] = x.id; x.blocks.push(b);
          }
          x.len++;
        }
        while (P.queue.length) {
          const id = P.queue[0], nb = Math.ceil(reqs[id].p / S.B);
          if (P.free.length < nb) break;
          const x = { id, blocks: [], len: reqs[id].p, seq: ++P.seq };
          for (let k = 0; k < nb; k++) { const b = P.free.shift(); P.bown[b] = id; x.blocks.push(b); }
          P.run.push(x); P.queue.shift();
        }
        P.su += P.run.reduce((s, x) => s + x.len, 0); P.sr += P.run.reduce((s, x) => s + x.blocks.length * S.B, 0); P.n++; P.qsum += P.queue.length;
      }
      reset();
      for (let k = 0; k < 16; k++) { tick++; stepC(); stepP(); }
      UI.seg(ctl, { label: 'Block B', options: [2, 4, 8], value: 4, onChange: (v) => { S.B = v; reset(); } });
      UI.seg(ctl, { label: 'max_model_len (ceiling)', options: [16, 24, 32], value: 24, onChange: (v) => { S.maxLen = v; reset(); } });
      UI.button(ctl, '↻ new round', () => { S.seed++; reset(); });
      const H = V.fig(fig, (g, w, h, t, dt) => {
        const C = V.C();
        if (!LA.paused) {
          if (A.done >= NREQ && P.done >= NREQ) { hold += dt; if (hold > 2.6) { S.seed++; reset(); } }
          else { acc += dt; let n = 0; while (acc >= TICK && n++ < 4) { acc -= TICK; tick++; stepC(); stepP(); } if (tick > 600) { S.seed++; reset(); } }
        }
        const pal = [C.a, C.b, C.c, C.d, C.e, C.f];
        const pad = 14, gap = 2;
        const cols = w >= 640 ? 24 : 16, rows = CAP / cols;
        const cell = Math.max(10, Math.min(26, Math.floor((w - 2 * pad - (cols - 1) * gap) / cols)));
        const gridW = cols * cell + (cols - 1) * gap;
        const gx = Math.round((w - gridW) / 2);
        let y = 6;
        const panel = (M, paged) => {
          const title = paged ? `Paged · blocks of ${S.B} places on demand` : `Contiguous reservation · each request’s ceiling (≤ ${S.maxLen})`;
          V.text(g, title, pad, y + 13, { size: 13, weight: 600, color: C.ink, maxW: w - 2 * pad });
          const eff = M.sr ? M.su / M.sr : 0;
          if (w >= 560) {
            const mw = 150, mx = w - pad - mw, my = y + 5;
            V.box(g, mx, my, mw, 10, { r: 5, fill: V.alpha(C.muted, 0.15) });
            V.box(g, mx, my, Math.max(4, mw * eff), 10, { r: 5, fill: paged ? C.a : C.d });
          }
          const s1 = `stores tokens in ${Math.round(eff * 100)}% of what is reserved · avg queue ${sf(M.n ? M.qsum / M.n : 0, 1)} · done ${M.done}/${NREQ}` + (paged ? ` · preemptions ${M.pre}` : '');
          const hh = wrap(g, s1, pad, y + 32, w - 2 * pad, { size: 11.5, font: 'mono', color: paged ? C.a : C.d });
          const gy = y + 20 + hh + 4;
          const byId = {}; M.run.forEach((x) => { byId[x.id] = x; });
          for (let k = 0; k < CAP; k++) {
            const cx = gx + (k % cols) * (cell + gap), cy = gy + Math.floor(k / cols) * (cell + gap);
            let own = -1, used = false, first = false;
            if (!paged) { own = M.owner[k]; const x = byId[own]; if (x) { used = k - x.start < x.len; first = k === x.start; } }
            else { const b = Math.floor(k / S.B); own = M.bown[b]; const x = byId[own]; if (x) { const bi = x.blocks.indexOf(b); used = bi >= 0 && bi * S.B + (k % S.B) < x.len; first = bi === 0 && k % S.B === 0; } }
            if (own < 0 || !byId[own]) { V.box(g, cx, cy, cell, cell, { r: 3, fill: V.alpha(C.muted, 0.1) }); continue; }
            const col = pal[own % 6];
            if (used) V.box(g, cx, cy, cell, cell, { r: 3, fill: V.alpha(col, 0.85) });
            else { V.box(g, cx, cy, cell, cell, { r: 3, fill: V.alpha(col, 0.08), stroke: V.alpha(col, 0.55), lw: 1 }); hatch(g, cx + 1, cy + 1, cell - 2, cell - 2, V.alpha(col, 0.5), 5); }
            if (first && cell >= 12) V.text(g, LET[own], cx + cell / 2, cy + cell / 2 + 0.5, { size: Math.min(11, cell * 0.62), weight: 700, align: 'center', base: 'middle', color: used ? onColor(C) : C.ink });
          }
          if (paged) {
            g.strokeStyle = V.alpha(C.ink, 0.3); g.lineWidth = 1;
            for (let b = 0; b < CAP / S.B; b++) {
              const k = b * S.B, cx = gx + (k % cols) * (cell + gap) - 1.5, cy = gy + Math.floor(k / cols) * (cell + gap) - 1.5;
              V.rr(g, cx, cy, S.B * (cell + gap) - gap + 3, cell + 3, 4); g.stroke();
            }
          }
          const qy = gy + rows * (cell + gap) + 12;
          V.text(g, 'queue', pad, qy + 4, { size: 11, font: 'mono', color: C.muted });
          if (!M.queue.length) V.text(g, '— empty', pad + 40, qy + 4, { size: 11, color: C.muted });
          const maxQ = Math.max(1, Math.floor((w - pad - 46) / 20));
          M.queue.slice(0, maxQ).forEach((id, i) => {
            const cx = pad + 46 + i * 20;
            V.dot(g, cx, qy, 8.5, V.alpha(pal[id % 6], 0.85));
            V.text(g, LET[id], cx, qy + 0.5, { size: 10, weight: 700, align: 'center', base: 'middle', color: onColor(C) });
          });
          y = qy + 16;
        };
        panel(A, false);
        g.strokeStyle = V.alpha(C.muted, 0.25); g.setLineDash([4, 4]); g.beginPath(); g.moveTo(pad, y + 3); g.lineTo(w - pad, y + 3); g.stroke(); g.setLineDash([]);
        y += 10;
        panel(P, true);
        const ly = y + 12; let lx = pad;
        const leg = (kind, s) => {
          if (kind === 'used') V.box(g, lx, ly - 9, 12, 12, { r: 3, fill: V.alpha(C.a, 0.85) });
          else if (kind === 'hatch') { V.box(g, lx, ly - 9, 12, 12, { r: 3, stroke: V.alpha(C.a, 0.6), lw: 1 }); hatch(g, lx + 1, ly - 8, 10, 10, V.alpha(C.a, 0.6), 4); }
          else V.box(g, lx, ly - 9, 12, 12, { r: 3, fill: V.alpha(C.muted, 0.15) });
          V.text(g, s, lx + 17, ly + 1, { size: 11.5, color: C.muted });
          lx += 17 + tw(g, s, 11.5) + 16;
        };
        leg('used', 'stored token'); leg('hatch', 'reserved and idle'); leg('free', 'free');
        fitH(H, ly + 12);
      });
    }

    /* ======================================================================
       FIG 2 — stepper: pool, tables and slot mapping
       ====================================================================== */
    function figPaged() {
      const fig = root.querySelector('#fig-paged');
      const WA = ['The', 'cat', 'climbed', 'up', 'the', 'wall', 'and', 'saw', 'mice'];
      const WB = ['What', 'is', 'the', 'capital', 'of', 'Peru'];
      const seq = (a, b) => Array.from({ length: b - a }, (_, i) => a + i);
      const ST = [
        { nA: 0, nB: 0, tA: [], tB: [], neu: [], free: [1, 2, 3, 4, 5, 6, 7] },
        { nA: 7, nB: 0, tA: [1, 2], tB: [], neu: seq(0, 7).map((i) => ['A', i]), free: [3, 4, 5, 6, 7] },
        { nA: 7, nB: 5, tA: [1, 2], tB: [3, 4], neu: seq(0, 5).map((i) => ['B', i]), free: [5, 6, 7] },
        { nA: 8, nB: 6, tA: [1, 2], tB: [3, 4], neu: [['A', 7], ['B', 5]], free: [5, 6, 7] },
        { nA: 8, nB: 6, tA: [1, 2], tB: [3, 4], neu: [], read: true, free: [5, 6, 7] },
        { nA: 9, nB: 6, tA: [1, 2, 5], tB: [3, 4], neu: [['A', 8]], free: [6, 7], newBlock: 5 },
        { nA: 9, nB: 6, tA: [], tB: [3, 4], neu: [], free: [6, 7, 5, 2, 1], ghost: [1, 2, 5], doneA: true },
        { nA: 9, nB: 6, tA: [], tB: [3, 4], neu: [], free: [6, 7, 5, 2, 1], ghost: [1, 2, 5], doneA: true, nullHi: true },
      ];
      const steps = [
        { t: 'The empty pool', d: 'Eight physical blocks of 4 slots. ID 0 is the <b>null block</b>: it stays reserved and never holds data. The free queue starts with IDs 1…7.' },
        { t: 'A’s prefill', d: 'A arrives with 7 tokens: ⌈7/4⌉ = 2 blocks. IDs 1 and 2 leave the queue and A’s table becomes [1, 2]. The slot mapping of the 7 tokens is 4…10 (ID × 4 + offset).' },
        { t: 'B’s prefill', d: 'B arrives with 5 tokens and receives IDs 3 and 4 (slots 12…16). The two requests share the pool without knowing about each other.' },
        { t: 'One decode step', d: 'Each request generates one token. “saw” goes to the last slot of ID 2 (slot 11) and “Peru” to ID 4 (slot 17). The step’s slot mapping, [11, 17], is what the write kernel receives.' },
        { t: 'The read follows the table', d: 'For A’s next query, the attention kernel reads the KV block by block in table order: ID 1, then ID 2. The request never needed contiguous memory.' },
        { t: 'A asks for a new block', d: 'ID 2 is full. A’s 9th token (“mice”) asks the queue for a block, receives ID 5 and goes to slot 20. A’s blocks are now 1, 2 and 5, scattered; A’s leftover is the 3 empty slots of ID 5.' },
        { t: 'A finishes', d: 'A’s blocks go back to the free queue (in reverse order). With prefix caching, they keep their content and hash until someone needs the space (chapter 33).' },
        { t: 'Padding and the null block', d: 'Tables have a fixed width and unused positions hold 0. Since ID 0 is the null block, an empty entry never points to anyone’s data. The sliding window uses the same trick: blocks that leave the window become 0 in the table and go back to the pool.' },
      ];
      const st = UI.stepper(fig, steps, { autoplay: 6000 });
      const H = V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const s = ST[st.i], el = st.elapsed();
        const pad = 14, wide = w >= 720;
        const colA = C.a, colB = C.b;
        const LW = wide ? Math.min(390, w * 0.47) : w - 2 * pad;
        const lx = pad;
        const cw = Math.max(26, Math.min(58, (LW - 92) / 4 - 4)), ch = 22;
        const chipPos = { A: [], B: [] }, tabPos = { A: [], B: [] };
        let y = 10;
        const drawReq = (name, words, n, tbl, col, done) => {
          pill(g, lx, y + 9, name, col, { size: 12, a: 0.25 });
          const status = done ? 'finished · blocks returned' : n ? `${n} tokens · ${Math.ceil(n / 4)} block(s)` : 'not here yet';
          V.text(g, status, lx + 32, y + 13, { size: 11.5, color: C.muted });
          y += 24;
          const nrows = name === 'A' ? 3 : 2;
          for (let j = 0; j < nrows; j++) {
            V.text(g, 'b' + j, lx, y + ch / 2 + 4, { size: 10.5, font: 'mono', color: C.muted });
            for (let k = 0; k < 4; k++) {
              const i = j * 4 + k, cx = lx + 22 + k * (cw + 4);
              chipPos[name][i] = [cx + cw / 2, y + ch / 2];
              const isNew = s.neu.some(([r, ii]) => r === name && ii === i);
              if (i < words.length && i < n) {
                V.box(g, cx, y, cw, ch, { r: 6, fill: V.alpha(col, done ? 0.07 : 0.22), stroke: isNew ? C.d : V.alpha(col, done ? 0.3 : 0.65), lw: isNew ? 2 : 1 });
                V.text(g, words[i], cx + cw / 2, y + ch / 2 + 0.5, { size: 10.5, align: 'center', base: 'middle', color: done ? C.muted : C.ink, maxW: cw - 4 });
              } else V.box(g, cx, y, cw, ch, { r: 6, stroke: V.alpha(C.muted, 0.22), lw: 1 });
            }
            const ax = lx + 22 + 4 * (cw + 4) + 2, id = tbl[j];
            if (id != null && !done) {
              V.arrow(g, ax, y + ch / 2, ax + 20, y + ch / 2, { color: V.alpha(col, 0.85), lw: 1.4, head: 6 });
              V.text(g, 'ID ' + id, ax + 24, y + ch / 2 + 4, { size: 11, font: 'mono', weight: 700, color: col });
            }
            y += ch + 5;
          }
          V.text(g, 'table', lx, y + 14, { size: 10.5, font: 'mono', color: C.muted });
          for (let j = 0; j < 4; j++) {
            const cx = lx + 50 + j * 26, v = done ? 0 : tbl[j] != null ? tbl[j] : 0, isNull = v === 0;
            tabPos[name][j] = [cx + 11, y + 10];
            V.box(g, cx, y, 22, 20, { r: 4, fill: isNull ? V.alpha(C.muted, s.nullHi ? 0.22 : 0.07) : V.alpha(col, 0.25), stroke: isNull ? V.alpha(C.muted, s.nullHi ? 0.95 : 0.3) : col, lw: isNull && s.nullHi ? 1.6 : 1 });
            V.text(g, String(v), cx + 11, y + 10.5, { size: 11, font: 'mono', weight: 700, align: 'center', base: 'middle', color: isNull ? C.muted : C.ink });
          }
          y += 32;
        };
        drawReq('A', WA, s.nA, s.tA, colA, !!s.doneA);
        drawReq('B', WB, s.nB, s.tB, colB, false);
        const leftEnd = y;
        // ---------- pool ----------
        const px = wide ? lx + LW + 26 : pad, PW = wide ? w - px - pad : w - 2 * pad;
        let py = wide ? 10 : y + 4;
        V.text(g, 'physical pool (BlockPool) · 8 blocks × 4 slots', px, py + 10, { size: 11.5, font: 'mono', color: C.muted });
        py += 18;
        const bw = (PW - 12) / 2, sw = (bw - 9) / 4, sh = 30;
        const content = {};
        const put = (name, n, tbl, ghost) => { for (let i = 0; i < n; i++) { const id = tbl[Math.floor(i / 4)]; if (id == null) continue; content[id * 4 + (i % 4)] = { name, i, ghost }; } };
        if (s.doneA) put('A', s.nA, s.ghost, true); else put('A', s.nA, s.tA, false);
        put('B', s.nB, s.tB, false);
        const owner = {}; s.tA.forEach((id) => { owner[id] = 'A'; }); s.tB.forEach((id) => { owner[id] = 'B'; });
        const slotPos = {}, blockPos = {};
        for (let b = 0; b < 8; b++) {
          const bx = px + (b % 2) * (bw + 12), by = py + Math.floor(b / 2) * (sh + 22);
          const own = owner[b], col = own === 'A' ? colA : own === 'B' ? colB : C.muted;
          const free = s.free.indexOf(b) >= 0;
          V.text(g, `ID ${b}` + (b === 0 ? ' · null' : own ? ` · ${own}` : free ? ' · free' : ''), bx, by + 10, { size: 11, font: 'mono', weight: 600, color: own ? col : C.muted });
          blockPos[b] = [bx + bw / 2, by + 14];
          if (s.newBlock === b) V.glow(g, bx + bw / 2, by + 14 + sh / 2, 60, C.d, 0.35 + 0.25 * Math.sin(t * 5));
          for (let k = 0; k < 4; k++) {
            const slot = b * 4 + k, sx = bx + k * (sw + 3), sy = by + 14;
            slotPos[slot] = [sx + sw / 2, sy + sh / 2];
            const c = content[slot];
            const isNew = c && s.neu.some(([r, ii]) => r === c.name && ii === c.i);
            if (b === 0) {
              V.box(g, sx, sy, sw, sh, { r: 4, fill: V.alpha(C.muted, 0.08), stroke: V.alpha(C.muted, s.nullHi ? 0.95 : 0.3), lw: s.nullHi ? 1.6 : 1 });
              hatch(g, sx + 1, sy + 1, sw - 2, sh - 2, V.alpha(C.muted, 0.35), 5);
            } else if (c) {
              const cc = c.name === 'A' ? colA : colB;
              V.box(g, sx, sy, sw, sh, { r: 4, fill: V.alpha(cc, c.ghost ? 0.06 : 0.26), stroke: isNew ? C.d : V.alpha(cc, c.ghost ? 0.3 : 0.75), lw: isNew ? 2 : 1 });
              if (c.ghost) hatch(g, sx + 1, sy + 1, sw - 2, sh - 2, V.alpha(cc, 0.25), 6);
              V.text(g, (c.name === 'A' ? WA : WB)[c.i], sx + sw / 2, sy + sh / 2 + 5, { size: 10, align: 'center', color: c.ghost ? C.muted : C.ink, maxW: sw - 3 });
            } else V.box(g, sx, sy, sw, sh, { r: 4, stroke: V.alpha(C.muted, 0.3), lw: 1 });
            V.text(g, String(slot), sx + 3, sy + 9, { size: 8.5, font: 'mono', weight: isNew ? 700 : 400, color: isNew ? C.d : C.muted });
          }
        }
        py += 4 * (sh + 22);
        // ---------- animations ----------
        const slotOf = (name, i) => { const tb = name === 'A' ? (s.doneA ? s.ghost : s.tA) : s.tB; const id = tb[Math.floor(i / 4)]; return id == null ? null : id * 4 + (i % 4); };
        if (el < 1.1) {
          const p = V.ease.inOut(V.clamp(el / 1.0, 0, 1));
          s.neu.forEach(([name, i]) => {
            const a = chipPos[name][i], sl = slotOf(name, i), b = sl != null ? slotPos[sl] : null;
            if (!a || !b) return;
            const x = V.lerp(a[0], b[0], p), yy = V.lerp(a[1], b[1], p) - Math.sin(p * Math.PI) * 26;
            V.glow(g, x, yy, 12, C.d, 0.9); V.dot(g, x, yy, 3, C.d);
          });
        }
        if (s.read) {
          [0, 1].forEach((j) => {
            const a = tabPos.A[j], b = blockPos[s.tA[j]];
            if (!a || !b) return;
            V.arrow(g, a[0], a[1] + 10, b[0], b[1] - 2, { color: V.alpha(C.a, 0.85), lw: 1.6, head: 7, dash: [5, 4], curve: wide ? -40 : 30 });
            const u = (t * 0.6 + j * 0.5) % 1;
            V.glow(g, V.lerp(a[0], b[0], u), V.lerp(a[1] + 10, b[1] - 2, u), 9, C.a, 0.8);
          });
        }
        if (s.nullHi) {
          const z = blockPos[0];
          [['B', 2], ['B', 3], ['A', 0]].forEach(([name, j]) => {
            const a = tabPos[name][j];
            if (a && z) V.arrow(g, a[0], a[1] + 10, z[0], z[1] - 2, { color: V.alpha(C.muted, 0.8), lw: 1.2, head: 6, dash: [3, 4], curve: wide ? -30 : 24 });
          });
        }
        // ---------- bottom row ----------
        let by2 = (wide ? Math.max(leftEnd, py) : py) + 6;
        let xx = pad;
        V.text(g, 'free queue', xx, by2 + 4, { size: 11, font: 'mono', color: C.muted });
        xx += tw(g, 'free queue', 11, 'mono') + 10;
        s.free.forEach((id) => { xx += pill(g, xx, by2, String(id), C.muted, { size: 10.5, a: 0.12 }) + 4; });
        by2 += 24;
        if (s.neu.length) {
          xx = pad;
          V.text(g, 'this step’s slot mapping', xx, by2 + 4, { size: 11, font: 'mono', color: C.d });
          xx += tw(g, 'this step’s slot mapping', 11, 'mono') + 10;
          s.neu.forEach(([name, i]) => {
            const sl = slotOf(name, i); if (sl == null) return;
            const wdt = tw(g, String(sl), 10.5, 'mono', 600) + 16;
            if (xx + wdt > w - pad) { xx = pad; by2 += 22; }
            xx += pill(g, xx, by2, String(sl), C.d, { size: 10.5 }) + 4;
          });
          by2 += 24;
        }
        fitH(H, by2 + 4);
      });
    }

    /* ======================================================================
       FIG 3 — address translator (DS guards and KpoolTail clamp)
       ====================================================================== */
    function figSlot() {
      const fig = root.querySelector('#fig-slot');
      const ctl = UI.controls(fig);
      const S = { mode: 'ds', guard: true, t: 5, pad: false };
      const DS = { st: 3, rows: [[7, 1, 10], [5, 2, 14], [8, 4, 11]], nb: 12, beyond: [] };
      const GL = { st: 1, rows: [[7], [3], [9]], nb: 12, beyond: [90, 57] };
      const ownerOf = (D, id) => { for (let i = 0; i < D.rows.length; i++) if (D.rows[i].indexOf(id) >= 0) return i; return -1; };
      function calc() {
        const D = S.mode === 'ds' ? DS : GL, r = 1;
        const flat = D.rows.reduce((a, x) => a.concat(x), []).concat(D.beyond);
        const t = S.t, b0 = Math.floor(t / 4), off = t % 4;
        const R = { D, flat, t, b0, off };
        if (S.mode === 'ds') {
          R.b = b0; R.fi = r * D.st + b0; R.id = R.fi < flat.length ? flat[R.fi] : 99;
          R.g1 = !S.pad; R.g2 = b0 < D.st; R.g3 = R.id >= 0 && R.id < D.nb;
          R.slot = R.id * 4 + off;
          if (S.guard) R.out = !R.g1 ? 'drop1' : !R.g2 ? 'drop2' : !R.g3 ? 'drop3' : 'ok';
          else R.out = !R.g3 ? 'ima' : !R.g2 ? 'foreign' : !R.g1 ? 'stale' : 'ok';
        } else {
          R.b = S.guard ? Math.min(b0, D.st - 1) : b0;
          R.fi = r * D.st + R.b; R.id = flat[R.fi];
          R.slot = R.id * 4 + off;
          R.out = R.b === 0 ? 'ok' : R.id < D.nb ? 'foreign' : 'ima';
        }
        return R;
      }
      let H = null;
      UI.seg(ctl, {
        label: 'Case', options: [{ v: 'ds', label: 'DeepSeek: top-k → slot' }, { v: 'glm', label: 'GLM: KpoolTail ring' }], value: 'ds',
        onChange: (v) => {
          S.mode = v; const mx = v === 'ds' ? 19 : 11; sl.input.max = mx; if (S.t > mx) sl.set(mx);
          padT.el.style.display = v === 'ds' ? '' : 'none';
          const sp = guardT.el.querySelector('span'); if (sp) sp.textContent = v === 'ds' ? 'guards on' : 'clamp on';
          H.redraw();
        },
      });
      const guardT = UI.toggle(ctl, { label: 'guards on', value: true, onChange: (v) => { S.guard = v; H.redraw(); } });
      const padT = UI.toggle(ctl, { label: 'padding token', value: false, onChange: (v) => { S.pad = v; H.redraw(); } });
      const sl = UI.slider(ctl, { label: 'index t', min: 0, max: 19, step: 1, value: 5, onInput: (v) => { S.t = v; if (H) H.redraw(); } });
      H = V.fig(fig, (g, w, h, time) => {
        const C = V.C(), R = calc(), D = R.D, ds = S.mode === 'ds';
        const pad = 14, wide = w >= 700;
        const rowCol = [C.b, C.a, C.f];
        const lab = { size: 11.5, font: 'mono', color: C.muted };
        let y = 8;
        V.text(g, ds ? 'batch block table · 3 requests × S = 3' : 'KpoolTail table · 3 requests × S = 1', pad, y + 10, lab);
        y += 20;
        const cs = 30, mx = pad + 26;
        D.rows.forEach((row, i) => {
          V.text(g, 'r' + i, pad, y + i * (cs + 4) + cs / 2 + 4, { size: 11.5, font: 'mono', weight: i === 1 ? 700 : 400, color: rowCol[i] });
          row.forEach((id, j) => {
            const cx = mx + j * (cs + 4), cy = y + i * (cs + 4), fi = i * D.st + j, hit = fi === R.fi, bad = id >= D.nb;
            V.box(g, cx, cy, cs, cs, { r: 5, fill: V.alpha(rowCol[i], i === 1 ? 0.3 : 0.13), stroke: hit ? C.d : V.alpha(rowCol[i], 0.6), lw: hit ? 2.6 : 1 });
            V.text(g, String(id), cx + cs / 2, cy + cs / 2 + 0.5, { size: 12, font: 'mono', weight: 700, align: 'center', base: 'middle', color: bad ? RED(C) : C.ink });
          });
        });
        g.strokeStyle = V.alpha(C.a, 0.9); g.lineWidth = 1.5; V.rr(g, mx - 4, y + (cs + 4) - 4, D.st * (cs + 4) + 4, cs + 8, 7); g.stroke();
        const matH = 3 * (cs + 4);
        const fx = wide ? mx + D.st * (cs + 4) + 44 : pad, fy = wide ? y + 30 : y + matH + 40;
        V.text(g, 'the same table in memory (row after row)', fx, fy - 26, lab);
        const n = R.flat.length;
        const fcw = Math.max(20, Math.min(36, (w - fx - pad) / n - 3));
        R.flat.forEach((id, k) => {
          const cx = fx + k * (fcw + 3), rr = Math.floor(k / D.st), beyond = rr >= D.rows.length, col = beyond ? C.muted : rowCol[rr];
          if (beyond) { g.setLineDash([3, 3]); V.box(g, cx, fy, fcw, 28, { r: 4, stroke: k === R.fi ? C.d : V.alpha(C.muted, 0.7), lw: k === R.fi ? 2.4 : 1 }); g.setLineDash([]); }
          else V.box(g, cx, fy, fcw, 28, { r: 4, fill: V.alpha(col, 0.14), stroke: k === R.fi ? C.d : V.alpha(col, 0.55), lw: k === R.fi ? 2.4 : 1 });
          V.text(g, beyond ? '?' : String(id), cx + fcw / 2, fy + 14.5, { size: 11.5, font: 'mono', weight: 700, align: 'center', base: 'middle', color: beyond ? C.muted : id >= D.nb ? RED(C) : C.ink });
          V.text(g, String(k), cx + fcw / 2, fy + 40, { size: 9.5, font: 'mono', align: 'center', color: C.muted });
        });
        for (let rr = 0; rr <= D.rows.length; rr++) {
          const k0 = rr * D.st; if (k0 >= n) break;
          const cx = fx + k0 * (fcw + 3);
          V.text(g, rr < D.rows.length ? 'r' + rr : 'out', cx + 2, fy + 54, { size: 10, font: 'mono', color: rr < D.rows.length ? rowCol[rr] : C.muted });
        }
        if (R.fi < n) {
          const cx = fx + R.fi * (fcw + 3) + fcw / 2;
          V.arrow(g, cx, fy - 17 - Math.abs(Math.sin(time * 3)) * 3, cx, fy - 2, { color: C.d, lw: 2, head: 6 });
        }
        y = Math.max(y + matH, fy + 58) + 14;
        // ---------- pool ----------
        V.text(g, ds ? `physical pool · ${D.nb} blocks of 4 slots (IDs 0…11)` : `physical pool · tail blocks (IDs 0…11)`, pad, y + 10, lab);
        y += 20;
        const perRow = wide ? 12 : 6, gp = 6;
        const bwp = (w - 2 * pad - (perRow - 1) * gp) / perRow, sw = (bwp - 3) / 4, sh = 22;
        const target = (R.out === 'ok' || R.out === 'foreign' || R.out === 'stale') ? R.slot : null;
        for (let b = 0; b < D.nb; b++) {
          const bx = pad + (b % perRow) * (bwp + gp), by = y + Math.floor(b / perRow) * (sh + 22);
          const ow = b === 0 ? -2 : ownerOf(D, b), col = ow >= 0 ? rowCol[ow] : C.muted;
          V.text(g, (b === 0 ? 'null' : 'ID ' + b), bx + bwp / 2, by + 9, { size: 9.5, font: 'mono', align: 'center', color: ow >= 0 ? col : C.muted });
          for (let k = 0; k < 4; k++) {
            const slot = b * 4 + k, sx = bx + k * (sw + 1), sy = by + 13;
            const ring = !ds && ow === 1;
            V.box(g, sx, sy, sw, sh, { r: 3, fill: ow >= 0 ? V.alpha(col, ring && k === R.off ? 0.7 : 0.25) : V.alpha(C.muted, b === 0 ? 0.16 : 0.07) });
            if (slot === target) {
              const bad = R.out !== 'ok';
              g.strokeStyle = bad ? RED(C) : C.d; g.lineWidth = 2.4; V.rr(g, sx - 2, sy - 2, sw + 4, sh + 4, 4); g.stroke();
              V.glow(g, sx + sw / 2, sy + sh / 2, 18, bad ? RED(C) : C.d, 0.55 + 0.3 * Math.sin(time * 4));
            }
          }
          if (!ds && ow === 1) {
            const cxr = bx + bwp / 2, cyr = by + 13 + sh + 8;
            V.text(g, 'r1’s ring', cxr, cyr + 3, { size: 9, font: 'mono', align: 'center', color: C.a });
          }
        }
        const poolRows = Math.ceil(D.nb / perRow);
        if (R.out === 'ima') {
          const ex = w - pad, ey = y + (poolRows - 1) * (sh + 22) + 13 + sh / 2;
          V.arrow(g, ex - 70, ey + 22, ex, ey + 22, { color: RED(C), lw: 2.2, head: 8 });
          V.text(g, ds ? `slot ${R.slot} →` : 'garbage ID →', ex - 74, ey + 26, { size: 11, font: 'mono', weight: 700, align: 'right', color: RED(C) });
        }
        y += poolRows * (sh + 22) + 16;
        // ---------- math ----------
        const ok = (b) => (b ? '✓' : '✗');
        const L = [];
        if (ds) {
          L.push({ s: `t = ${R.t}  →  b = ⌊t/4⌋ = ${R.b0},  offset = t mod 4 = ${R.off}` });
          L.push({ s: `① valid token?  ${S.pad ? 'no (padding)' : 'yes'}` + (S.guard ? '' : '   [not checked]'), c: S.pad ? (S.guard ? C.d : RED(C)) : C.ink });
          L.push({ s: `② b < S?  ${R.b0} ${R.g2 ? '<' : '≥'} 3  ${ok(R.g2)}` + (S.guard ? '' : '   [not checked]'), c: R.g2 ? C.ink : S.guard ? C.d : RED(C) });
          L.push({ s: `ID = T.flat[r·S + b] = T.flat[${R.fi}] = ${R.id}` + (R.fi >= 6 ? '   (r2’s row!)' : ''), c: R.fi >= 6 ? RED(C) : C.ink });
          L.push({ s: `③ 0 ≤ ID < 12?  ${ok(R.g3)}` + (S.guard ? '' : '   [not checked]'), c: R.g3 ? C.ink : S.guard ? C.d : RED(C) });
          L.push({ s: `slot = ID·4 + offset = ${R.slot}` + (R.out.startsWith('drop') ? '   → not read' : ''), c: R.out.startsWith('drop') ? C.muted : C.ink });
        } else {
          L.push({ s: `pos = ${R.t}  →  b = ⌊pos/4⌋ = ${R.b0}` + (S.guard ? `  →  clamp: min(${R.b0}, S−1) = ${R.b}` : '   [no clamp]') });
          L.push({ s: `address = r·S + b = 1·1 + ${R.b} = ${R.fi}` + (R.fi === 1 ? '   (r1’s row)' : R.fi === 2 ? '   (r2’s row!)' : '   (outside the table!)'), c: R.fi === 1 ? C.ink : RED(C) });
          L.push({ s: `ID = ${R.fi < 3 ? R.id : '? (memory garbage)'}` });
          L.push({ s: `slot = ID·4 + pos mod 4 = ${R.fi < 3 ? R.slot : '?'}` });
        }
        L.forEach((ln, i) => V.text(g, ln.s, pad, y + 12 + i * 19, { size: 12, font: 'mono', color: ln.c || C.ink, maxW: w - 2 * pad }));
        y += L.length * 19 + 16;
        const msg = {
          ok: ds ? `✓ slot ${R.slot}: reads the right KV` : `✓ slot ${R.slot}: position ${R.off} of r1’s ring`,
          drop1: '−1 · dropped by guard ① (padding)', drop2: '−1 · dropped by guard ② (outside the row)', drop3: '−1 · dropped by guard ③ (ID outside the pool)',
          foreign: ds ? `✗ reads block ${R.id}, which belongs to r2` : `✗ writes to block ${R.id}, r2’s tail`,
          ima: ds ? `✗ slot ${R.slot} outside the pool: illegal access (IMA)` : '✗ unpredictable ID: corruption or IMA',
          stale: `⚠ reads slot ${R.slot} for a padding token (garbage)`,
        }[R.out];
        const col = R.out === 'ok' ? C.a : R.out.startsWith('drop') ? C.d : R.out === 'stale' ? C.d : RED(C);
        const bwd = Math.min(w - 2 * pad, tw(g, msg, 13, 'sans', 600) + 26);
        V.box(g, pad, y, bwd, 30, { r: 9, fill: V.alpha(col, 0.16), stroke: col, lw: 1.5 });
        V.text(g, msg, pad + 13, y + 15.5, { size: 13, weight: 600, base: 'middle', color: C.ink, maxW: bwd - 20 });
        fitH(H, y + 42);
      });
    }

    /* ======================================================================
       FIG 4 — consequences of the block size
       ====================================================================== */
    function figBloco() {
      const fig = root.querySelector('#fig-bloco');
      const ctl = UI.controls(fig);
      const Ls = [100, 300, 900, 2000, 5000, 8004, 12000, 20000, 36000, 64000, 100000, 262144, 500000, 1000000];
      const S = { B: 3584, li: 5 };
      let H = null;
      UI.seg(ctl, { label: 'block_size', options: [16, 64, 256, 1664, 3584].map((b) => ({ v: b, label: nf(b) })), value: 3584, onChange: (v) => { S.B = v; upd(); } });
      UI.slider(ctl, { label: 'tokens', min: 0, max: Ls.length - 1, step: 1, value: 5, fmt: (i) => nf(Ls[i]), onInput: (i) => { S.li = i; upd(); } });
      const r1 = UI.readout(ctl, 'blocks', '—'), r2 = UI.readout(ctl, 'leftover', '—'), r3 = UI.readout(ctl, 'table at 1M', '—'), r4 = UI.readout(ctl, 'hit', '—');
      function upd() {
        const B = S.B, L = Ls[S.li], n = Math.ceil(L / B);
        r1.set(nf(n)); r2.set(nf(n * B - L) + ' tok'); r3.set(nf(Math.ceil(1e6 / B))); r4.set(nf(B * Math.floor((L - 1) / B)) + ' / ' + nf(L));
        if (H) H.redraw();
      }
      upd();
      H = V.fig(fig, (g, w, h) => {
        const C = V.C();
        const B = S.B, L = Ls[S.li], n = Math.ceil(L / B), last = L - (n - 1) * B, sobra = n * B - L;
        const pad = 14, W = w - 2 * pad;
        const tt = { size: 12.5, weight: 600, color: C.ink };
        const sub = { size: 11.5, font: 'mono', color: C.muted };
        let y = 8;
        const head = (title, val) => {
          V.text(g, title, pad, y + 12, Object.assign({}, tt, { maxW: W }));
          if (tw(g, title, 12.5, 'sans', 600) + tw(g, val, 11.5, 'mono') + 24 > W) { y += 18; V.text(g, val, pad, y + 12, sub); }
          else V.text(g, val, w - pad, y + 12, Object.assign({}, sub, { align: 'right' }));
          y += 22;
        };
        // 1 — the request in blocks
        head('1 · the request in blocks', `${nf(L)} tokens = ${nf(n)} block(s) of ${nf(B)}`);
        const bh = 20;
        if (n <= 90) {
          const gapb = n > 40 ? 1 : 2, bwid = (W - gapb * (n - 1)) / n;
          for (let i = 0; i < n; i++) {
            const x = pad + i * (bwid + gapb);
            if (i < n - 1) V.box(g, x, y, bwid, bh, { r: Math.min(4, bwid / 3), fill: V.alpha(C.a, 0.75) });
            else {
              const uw = bwid * (last / B);
              V.box(g, x, y, bwid, bh, { r: Math.min(4, bwid / 3), fill: V.alpha(C.c, 0.14), stroke: V.alpha(C.c, 0.7), lw: 1 });
              hatch(g, x + uw, y + 1, bwid - uw - 1, bh - 2, V.alpha(C.c, 0.6), 5);
              V.box(g, x, y, Math.max(1, uw), bh, { r: Math.min(4, bwid / 3), fill: V.alpha(C.a, 0.75) });
            }
          }
        } else {
          V.box(g, pad, y, W, bh, { r: 5, fill: V.alpha(C.a, 0.75) });
          const every = Math.ceil(n / 45);
          g.strokeStyle = V.alpha(C.bg || '#000', 0.5); g.lineWidth = 1;
          for (let i = every; i < n; i += every) { const x = pad + (W * i) / n; g.beginPath(); g.moveTo(x, y + 3); g.lineTo(x, y + bh - 3); g.stroke(); }
          V.text(g, `(one tick every ${nf(every)} blocks)`, pad, y + bh + 13, { size: 10.5, color: C.muted });
        }
        // zoom on the last block
        const zw = Math.min(W * 0.62, 320), zx = w - pad - zw, zy = y + bh + 22;
        const lastX = pad + W - Math.max(3, W / n);
        g.strokeStyle = V.alpha(C.c, 0.45); g.lineWidth = 1; g.setLineDash([3, 3]);
        g.beginPath(); g.moveTo(lastX, y + bh); g.lineTo(zx, zy); g.moveTo(pad + W, y + bh); g.lineTo(zx + zw, zy); g.stroke(); g.setLineDash([]);
        const uw = zw * (last / B);
        V.box(g, zx, zy, zw, 22, { r: 5, fill: V.alpha(C.c, 0.12), stroke: V.alpha(C.c, 0.7), lw: 1 });
        hatch(g, zx + uw, zy + 1, zw - uw - 1, 20, V.alpha(C.c, 0.55), 5);
        V.box(g, zx, zy, Math.max(2, uw), 22, { r: 5, fill: V.alpha(C.a, 0.8) });
        V.text(g, `last block: ${nf(last)} used · leftover ${nf(sobra)} (${sf((100 * sobra) / (n * B), 1)}% of allocated)`, zx + zw, zy + 36, Object.assign({}, sub, { align: 'right', maxW: zw + (zx - pad) }));
        V.text(g, 'zoom ↘', zx - 8, zy + 15, { size: 10.5, color: C.muted, align: 'right' });
        y = zy + 52;
        // 2 — table
        const E = Math.ceil(1e6 / B);
        head('2 · table of a 1M-token request', `${nf(E)} entries`);
        const sq = 8, sg = 2, perRow = Math.floor((W + sg) / (sq + sg)), maxSq = perRow * 4;
        const per = Math.max(1, Math.ceil(E / maxSq)), nsq = Math.ceil(E / per);
        for (let i = 0; i < nsq; i++) {
          const x = pad + (i % perRow) * (sq + sg), yy = y + Math.floor(i / perRow) * (sq + sg);
          V.box(g, x, yy, sq, sq, { r: 2, fill: V.alpha(C.b, 0.7) });
        }
        const rowsSq = Math.ceil(nsq / perRow);
        y += rowsSq * (sq + sg) + 4;
        V.text(g, per === 1 ? 'each square = 1 entry' : `each square ≈ ${nf(per)} entries`, pad, y + 10, { size: 10.5, color: C.muted });
        y += 20;
        // 3 — hit
        const hit = B * Math.floor((L - 1) / B), rec = L - hit;
        head(`3 · the same ${nf(L)}-token prompt comes back`, `reuses ${nf(hit)} · recomputes ${nf(rec)}`);
        const hw = W * (hit / L);
        V.box(g, pad, y, W, 20, { r: 5, fill: V.alpha(C.c, 0.3) });
        if (hit > 0) V.box(g, pad, y, Math.max(3, hw), 20, { r: 5, fill: V.alpha(C.e, 0.8) });
        y += 34;
        // 4 — kernel pages
        V.text(g, w < 560 ? '4 · 64-token kernel pages per ID (GLM)' : '4 · 64-token kernel pages in one manager block (the GLM case)', pad, y + 12, Object.assign({}, tt, { maxW: W }));
        y += 20;
        if (B >= 64) {
          const k = B / 64, kw = (W - (k - 1) * 1.5) / k;
          for (let i = 0; i < k; i++) V.box(g, pad + i * (kw + 1.5), y, kw, 20, { r: Math.min(3, kw / 3), fill: V.alpha(C.d, 0.35 + 0.35 * (i % 2)) });
          V.text(g, `${nf(k)} kernel page(s) per ID`, pad, y + 34, sub);
        } else {
          V.box(g, pad, y, W * (B / 64), 20, { r: 4, fill: V.alpha(C.d, 0.6) });
          V.box(g, pad, y, W, 20, { r: 4, stroke: V.alpha(C.d, 0.7), lw: 1 });
          V.text(g, `block (${B}) smaller than the kernel page (64): the kernel would need ${64 / B} IDs per page`, pad, y + 34, Object.assign({}, sub, { maxW: W }));
        }
        fitH(H, y + 46);
      }, { static: true });
    }

    /* ======================================================================
       FIG 5 — the attention page catches up with the state
       ====================================================================== */
    function figPagina() {
      const fig = root.querySelector('#fig-pagina');
      const ctl = UI.controls(fig);
      const CFG = {
        glm2: { name: 'GLM · TP2', S: [['recurrent [32,128,128] FP32', 2097152], ['conv [12288,10] BF16', 245760]], b: 656, bl: '656 B/token (MLA fp8_ds_mla)', a: 256, note: 'boot: mla_page=2351104 · kernel: 56 pages of 64' },
        glm3: { name: 'GLM · TP3', S: [['recurrent FP32', 1441792], ['conv BF16', 168960]], b: 656, bl: '656 B/token (MLA fp8_ds_mla)', a: 256, note: 'boot TP3: MLA block_size=2560, mla_page=1679360' },
        qbf: { name: 'Qwen · BF16 KV · FP32 state', S: [['recurrent 24×128×128 FP32', 1572864], ['conv 5120×6 BF16', 61440]], b: 1024, bl: '1,024 B/token per GPU (K+V, 1 head × 256, BF16)', a: 16, note: 'README: 1,600-token attention block' },
        qf8: { name: 'Qwen · FP8 KV · FP32 state', S: [['recurrent 24×128×128 FP32', 1572864], ['conv 5120×6 BF16', 61440]], b: 512, bl: '512 B/token per GPU (FP8)', a: 16, note: 'inventory math: 3,200 (FP8 doubles the block)' },
        qf8b: { name: 'Qwen · FP8 KV · BF16 state', S: [['recurrent 24×128×128 BF16', 786432], ['conv 5120×6 BF16', 61440]], b: 512, bl: '512 B/token per GPU (FP8)', a: 16, note: 'block drop patch: 1,664 tokens at MTP 3' },
        ds: { name: 'DeepSeek · C4 group', S: [['fixed SWA page: 64 tokens × 584 B', 37376]], b: 146, bl: '584 B every 4 tokens = 146 B/token', a: 4, note: 'hence --block-size 256 (64 rows of 584 B)', ds: true },
      };
      const S = { k: 'glm2', t0: 0, restart: true };
      UI.select(ctl, { label: 'Model', options: Object.keys(CFG).map((k) => ({ v: k, label: CFG[k].name })), value: 'glm2', onChange: (v) => { S.k = v; S.restart = true; } });
      UI.button(ctl, '▶ replay', () => { S.restart = true; });
      const H = V.fig(fig, (g, w, h, t) => {
        if (S.restart) { S.t0 = t; S.restart = false; }
        const C = V.C(), c = CFG[S.k];
        const Stot = c.S.reduce((s, x) => s + x[1], 0);
        const Bmin = Stot / c.b, Bc = Math.ceil(Bmin), Bal = Math.ceil(Bc / c.a) * c.a, page = Bal * c.b, padB = page - Stot;
        const el = (t - S.t0) % 7;
        let Bcur;
        if (el < 2.6) Bcur = Bmin * V.ease.inOut(el / 2.6);
        else if (el < 3.4) Bcur = V.lerp(Bmin, Bal, V.ease.out((el - 2.6) / 0.8));
        else Bcur = Bal;
        const pad = 14, W = w - 2 * pad, xmax = Math.max(Stot, page) * 1.05, sc = W / xmax;
        const tt = { size: 12.5, weight: 600, color: C.ink, maxW: W };
        const sub = { size: 11.5, font: 'mono', color: C.muted, maxW: W };
        let y = 8;
        y += wrap(g, c.ds ? 'page already fixed by the SWA group' : 'recurrent state of one layer (per request and per rank)', pad, y + 12, W, tt) + 2;
        const b1y = y;
        let x = pad;
        c.S.forEach(([lbl, v], i) => {
          const ww = v * sc, col = i === 0 ? C.b : C.f;
          V.box(g, x, y, ww, 30, { r: 5, fill: V.alpha(col, 0.55), stroke: col, lw: 1 });
          if (tw(g, lbl, 11) + 10 < ww) V.text(g, lbl, x + 6, y + 19, { size: 11, color: C.ink });
          x += ww;
        });
        const sTxt = c.S.length > 1 ? `S = ${nf(Stot)} B = ${nf(c.S[0][1])} (recurrent) + ${nf(c.S[1][1])} (conv)` : `S = ${nf(Stot)} B`;
        y += wrap(g, sTxt, pad, y + 46, Math.max(160, Stot * sc - 12), { size: 11.5, font: 'mono', color: C.muted }) + 44;
        const t2y = y;
        y += wrap(g, c.ds ? `C4 group page = B × ${c.bl}` : `attention page = B × ${c.bl}`, pad, y + 12, W, tt) + 2;
        const b2y = y;
        const pw = Bcur * c.b * sc, reached = Bcur * c.b >= Stot - 1;
        V.box(g, pad, y, Math.max(2, pw), 30, { r: 5, fill: V.alpha(C.a, 0.6), stroke: C.a, lw: 1 });
        const nT = Math.min(80, Math.floor(pw / 6));
        g.strokeStyle = V.alpha(C.bg || '#000', 0.35); g.lineWidth = 1;
        for (let i = 1; i < nT; i++) { const xx = pad + (pw * i) / nT; g.beginPath(); g.moveTo(xx, y + 6); g.lineTo(xx, y + 24); g.stroke(); }
        if (el >= 2.6 && !c.ds && padB > 0) { const sx = pad + Stot * sc; hatch(g, sx, y + 1, Math.max(0, pw - (sx - pad)) - 1, 28, V.alpha(C.d, 0.8), 4); }
        const lblB = `B = ${nf(Math.round(Bcur))} tokens` + (reached ? '  ≥ S ✓' : '');
        const lx = Math.min(w - pad - tw(g, lblB, 12, 'mono', 700), pad + pw + 8);
        V.text(g, lblB, lx, y + 48, { size: 12, font: 'mono', weight: 700, color: reached ? C.a : C.ink });
        const sx = pad + Stot * sc;
        g.strokeStyle = V.alpha(C.ink, 0.6); g.setLineDash([4, 4]); g.lineWidth = 1.2;
        g.beginPath(); g.moveTo(sx, b1y - 4); g.lineTo(sx, t2y - 1); g.moveTo(sx, b2y - 4); g.lineTo(sx, b2y + 36); g.stroke(); g.setLineDash([]);
        y += 66;
        const lines = c.ds
          ? [`B = 37,376 B ÷ 146 B/token = ${nf(Bal)} tokens (${nf(Bal / 4)} rows of 584 B)`, `page P = ${nf(page)} B, the same for every group · no padding`, c.note]
          : [`⌈S ÷ b⌉ = ⌈${nf(Stot)} ÷ ${nf(c.b)}⌉ = ${nf(Bc)} tokens   (S/b = ${sf(Bmin, 1)})`, `align to ${c.a}  →  B = ${nf(Bal)} tokens`, `P = ${nf(Bal)} × ${nf(c.b)} = ${nf(page)} B · state padding = ${nf(padB)} B (${sf((100 * padB) / page, 2)}%)`, c.note];
        lines.forEach((ln, i) => { y += wrap(g, ln, pad, y + 12, W, { size: 12, font: 'mono', color: i === lines.length - 1 ? C.a : C.ink }) + 3; });
        y += 10;
        y += wrap(g, 'consequence: prefix hits and checkpoints only at multiples of B', pad, y + 12, W, tt) + 4;
        const rw = W, kmax = 4;
        g.strokeStyle = V.alpha(C.muted, 0.6); g.lineWidth = 1.2; g.beginPath(); g.moveTo(pad, y + 10); g.lineTo(pad + rw, y + 10); g.stroke();
        for (let k = 0; k <= kmax; k++) {
          const xx = pad + (rw * k) / kmax;
          V.dot(g, xx, y + 10, k ? 5 : 3, k ? C.e : C.muted);
          V.text(g, nf(k * Bal), xx, y + 28, { size: 10.5, font: 'mono', color: C.muted, align: k === 0 ? 'left' : k === kmax ? 'right' : 'center' });
        }
        fitH(H, y + 40);
      });
    }

    /* ======================================================================
       FIG 6 — the pool building (3D tensor)
       ====================================================================== */
    function figPredio() {
      const fig = root.querySelector('#fig-predio');
      const ctl = UI.controls(fig);
      const SL = 11, K = 12, PAGE = 2351104, IDXP = 118272, IDXW = IDXP / PAGE;
      const S = { hi: 'all', compact: false, cur: {} };
      const kind = (j) => (j === 0 ? 'null' : j <= 3 ? 'mla' : j <= 7 ? 'mamba' : 'draft');
      const layers = { mla: 11, mamba: 9, draft: 5, null: 11 };
      const gname = { mla: 'MLA + indexer (g0)', mamba: 'Mamba/KDA', draft: 'SWA drafter (g6)', null: 'null block' };
      const live = (j) => !(kind(j) === 'draft' && S.compact && j > 8);
      const fillOf = (j) => { const k = kind(j); if (k === 'mla') return 1; if (k === 'mamba') return 2342912 / PAGE; if (k === 'draft') return S.compact ? (896 * 2048) / PAGE : (64 * 2048) / PAGE; return 0; };
      const C0 = V.C();
      UI.seg(ctl, { label: 'Highlight', options: [{ v: 'all', label: 'all' }, { v: 'mla', label: 'MLA' }, { v: 'mamba', label: 'Mamba' }, { v: 'draft', label: 'drafter' }], value: 'all', onChange: (v) => { S.hi = v; } });
      const rd = UI.readout(ctl, 'drafter admission IDs', '65');
      UI.toggle(ctl, { label: 'compact pages (896 tokens)', value: false, onChange: (v) => { S.compact = v; rd.set(v ? '65 → 6 (≈11× fewer IDs)' : '65'); } });
      UI.legend(ctl, [[C0.muted, 'ID 0 · null'], [C0.a, 'MLA + indexer'], [C0.b, '4 Mamba groups'], [C0.c, 'drafter']]);
      const H = V.fig(fig, (g, w, h, t, dt) => {
        const C = V.C(), pad = 14, small = w < 520;
        const colOf = (k) => (k === 'mla' ? C.a : k === 'mamba' ? C.b : k === 'draft' ? C.c : C.muted);
        const lw = small ? 58 : 74, roof = 30, gap = 3;
        const cw = Math.max(14, Math.min(46, (w - 2 * pad - lw - (K - 1) * gap) / K));
        const ch = Math.max(14, Math.min(24, cw * 0.72));
        const ih = Math.max(3, Math.round(ch * 0.2)), igap = 1.5, bandH = SL * (ih + igap) - igap;
        const bx = pad + lw, bw = K * cw + (K - 1) * gap;
        const top = 34 + roof, mtop = top + bandH + 7;
        const k8 = Math.min(1, dt * 5);
        // roof
        g.fillStyle = V.alpha(C.muted, 0.1); g.strokeStyle = V.alpha(C.muted, 0.45); g.lineWidth = 1.2;
        g.beginPath(); g.moveTo(bx - 8, top - 6); g.lineTo(bx + bw / 2, top - 6 - roof); g.lineTo(bx + bw + 8, top - 6); g.closePath(); g.fill(); g.stroke();
        V.text(g, 'BlockPool', bx + bw / 2, top - 14, { size: 11, font: 'mono', weight: 600, align: 'center', color: C.muted });
        for (let j = 0; j < K; j++) V.text(g, String(j), bx + j * (cw + gap) + cw / 2, 22, { size: 10.5, font: 'mono', align: 'center', color: C.muted });
        V.text(g, 'ID →', bx - 6, 22, { size: 10.5, font: 'mono', align: 'right', color: C.muted });
        let hov = -1;
        if (H.pointer.inside) { const jj = Math.floor((H.pointer.x - bx) / (cw + gap)); if (jj >= 0 && jj < K && H.pointer.y > top - 10) hov = jj; }
        let dataW = 0, paidW = 0;
        // thin floors: the 11 indexer tensors (118,272 B page)
        for (let s = SL - 1; s >= 0; s--) {
          const yy = top + (SL - 1 - s) * (ih + igap);
          for (let j = 0; j < K; j++) {
            const x = bx + j * (cw + gap), k = kind(j), col = colOf(k);
            const dim = S.hi === 'all' || S.hi === k || k === 'null' ? 1 : 0.22;
            if (k === 'null') { V.box(g, x, yy, cw, ih, { r: 1, fill: V.alpha(C.muted, 0.16) }); continue; }
            if (!live(j)) { V.box(g, x, yy, cw, ih, { r: 1, fill: V.alpha(C.muted, 0.05) }); continue; }
            paidW += IDXW;
            if (k === 'mla') { dataW += IDXW; V.box(g, x, yy, cw, ih, { r: 1, fill: V.alpha(col, 0.85 * dim) }); }
            else V.box(g, x, yy, cw, ih, { r: 1, fill: V.alpha(col, 0.07 * dim), stroke: V.alpha(col, 0.35 * dim), lw: 0.8 });
          }
        }
        if (small) V.text(g, 'idx ×11', bx - 8, top + bandH / 2 + 4, { size: 9.5, font: 'mono', align: 'right', color: C.muted });
        else { V.text(g, 'indexer', bx - 8, top + bandH / 2 - 2, { size: 10.5, font: 'mono', align: 'right', color: C.muted }); V.text(g, '×11', bx - 8, top + bandH / 2 + 11, { size: 10.5, font: 'mono', align: 'right', color: C.muted }); }
        // thick floors: the 11 MLA tensors (2,351,104 B page)
        for (let s = SL - 1; s >= 0; s--) {
          const yy = mtop + (SL - 1 - s) * (ch + gap);
          V.text(g, 'MLA ' + s, bx - 8, yy + ch / 2 + 4, { size: small ? 9.5 : 10.5, font: 'mono', align: 'right', color: C.muted });
          for (let j = 0; j < K; j++) {
            const x = bx + j * (cw + gap), k = kind(j), col = colOf(k);
            const dim = S.hi === 'all' || S.hi === k || k === 'null' ? 1 : 0.22;
            if (k === 'null') { V.box(g, x, yy, cw, ch, { r: 3, fill: V.alpha(C.muted, 0.1), stroke: V.alpha(C.muted, 0.35), lw: 1 }); hatch(g, x + 1, yy + 1, cw - 2, ch - 2, V.alpha(C.muted, 0.35), 5); continue; }
            if (!live(j)) { V.box(g, x, yy, cw, ch, { r: 3, stroke: V.alpha(C.muted, 0.18), lw: 1 }); continue; }
            paidW += 1;
            if (s >= layers[k]) {
              g.setLineDash([3, 3]); V.box(g, x, yy, cw, ch, { r: 3, stroke: V.alpha(col, 0.45 * dim), lw: 1 }); g.setLineDash([]);
              continue;
            }
            const key = s + ':' + j, target = fillOf(j);
            const f = (S.cur[key] = S.cur[key] == null ? target : V.lerp(S.cur[key], target, k8));
            dataW += target;
            V.box(g, x, yy, cw, ch, { r: 3, fill: V.alpha(col, 0.08 * dim), stroke: V.alpha(col, 0.7 * dim), lw: 1 });
            const fh = Math.max(2, (ch - 2) * f);
            V.box(g, x + 1, yy + ch - 1 - fh, cw - 2, fh, { r: 2, fill: V.alpha(col, 0.85 * dim), stroke: k === 'draft' ? V.alpha(col, dim) : null, lw: 1.2 });
            if (k === 'draft' && s === 0 && cw >= 22) V.text(g, sf(100 * target, 0) + '%', x + cw / 2, yy + ch / 2 + (target > 0.5 ? 1 : -1), { size: 9.5, font: 'mono', weight: 700, align: 'center', base: 'middle', color: target > 0.5 ? onColor(C) : C.ink });
          }
        }
        const bottom = mtop + SL * (ch + gap);
        g.strokeStyle = V.alpha(C.ink, 0.5); g.lineWidth = 2; g.beginPath(); g.moveTo(bx - 14, bottom + 2); g.lineTo(bx + bw + 14, bottom + 2); g.stroke();
        // highlighted column
        {
          const jc = hov >= 0 ? hov : Math.floor(t / 1.6) % K, x = bx + jc * (cw + gap);
          g.strokeStyle = hov >= 0 ? C.d : V.alpha(C.d, 0.55); g.lineWidth = hov >= 0 ? 2.2 : 1.6;
          V.rr(g, x - 3, top - 3, cw + 6, bottom - top + 3, 5); g.stroke();
        }
        const brace = (j0, j1, lbl, col) => {
          const x0 = bx + j0 * (cw + gap), x1 = bx + j1 * (cw + gap) + cw, yb = bottom + 10;
          g.strokeStyle = V.alpha(col, 0.8); g.lineWidth = 1.4;
          g.beginPath(); g.moveTo(x0 + 1, yb); g.lineTo(x0 + 1, yb + 5); g.lineTo(x1 - 1, yb + 5); g.lineTo(x1 - 1, yb); g.stroke();
          V.text(g, lbl, (x0 + x1) / 2, yb + 19, { size: small ? 9.5 : 11, align: 'center', color: col, maxW: x1 - x0 + gap });
        };
        const dpct = sf(100 * fillOf(8), 0) + '%';
        brace(0, 0, 'null', C.muted); brace(1, 3, 'MLA+idx', C.a); brace(4, 7, '4 × Mamba', C.b);
        if (S.compact) { brace(8, 8, 'draft', C.c); brace(9, 11, 'free', C.muted); } else brace(8, 11, cw >= 22 ? 'drafter' : 'draft ' + dpct, C.c);
        let y = bottom + 50;
        const jj = hov >= 0 ? hov : Math.floor(t / 1.6) % K, kk = kind(jj);
        const pc = (v) => sf(100 * v, 1) + '%';
        const desc = kk === 'null' ? `ID ${jj} · null block: reserved, never holds data`
          : !live(jj) ? `ID ${jj} · free (compact pages made this ID unnecessary)`
          : kk === 'mla' ? `ID ${jj} · ${gname.mla}: uses the 11 MLA tensors and the 11 indexer ones, pages 100% full`
          : `ID ${jj} · ${gname[kk]}: uses ${layers[kk]} of the 11 MLA tensors (pages ${pc(fillOf(jj))} full) and also pays for the 11 indexer pages without using them`;
        y += wrap(g, desc, pad, y, w - 2 * pad, { size: 12.5, weight: 600, color: kk === 'null' ? C.muted : colOf(kk) });
        const pct = paidW ? (100 * dataW) / paidW : 0;
        y += wrap(g, `bytes paid for by allocated IDs that hold data: ${sf(pct, 0)}% (illustrative scale) · dashed = reserved page the group doesn’t use · each ID costs 11 × 2,351,104 + 11 × 118,272 B ≈ 25.9 MiB`, pad, y + 2, w - 2 * pad, { size: 11.5, font: 'mono', color: C.muted });
        fitH(H, y + 8);
      });
    }

    /* ======================================================================
       FIG 7 — the budget of one node (profile run)
       ====================================================================== */
    function figOrcamento() {
      const fig = root.querySelector('#fig-orcamento');
      const ctl = UI.controls(fig);
      const PR = {
        qwen: { M: 121.69, maxLen: 262144 },
        ds: { M: 121.7, maxLen: 1048576 },
        glm: { M: 121.69, maxLen: 850000 },
      };
      const S = { p: 'ds', u: { qwen: 0.835, ds: 0.835, glm: 0.85 }, qdt: 'fp8', qpin: 0, it: true, cg: true, idx: 'rightsize', e3: true, compact: true, gpin: 11, cur: {} };
      const IDX = { stock: (850000 * 40 * 132) / 2 ** 30, rightsize: (4 * Math.ceil(850007 / 4) * 132) / 2 ** 30 };
      function parts() {
        if (S.p === 'qwen') return [{ k: 'weights', v: 64.46, c: 'a', ph: 1 }, { k: 'outside PyTorch', v: 4.0612, c: 'f', ph: 2 }, { k: 'activation peak', v: 1.07, c: 'd', ph: 2 }];
        if (S.p === 'ds') return [{ k: 'weights', v: 79.17, c: 'a', ph: 1 }, { k: 'other (closes the gap)', v: 5.4095, c: 'f', ph: 2 }];
        const a = [{ k: 'weights', v: 79.65, c: 'a', ph: 1 }];
        if (S.it) a.push({ k: 'InstantTensor effect', v: 5.0, c: 'c', ph: 1 });
        a.push({ k: 'other (closes the gap)', v: 3.21, c: 'f', ph: 2 });
        a.push({ k: 'indexer workspace', v: IDX[S.idx], c: 'd', ph: 3 });
        if (S.e3) a.push({ k: 'E3 scratch', v: 0.547, c: 'd', ph: 3 });
        if (S.cg) a.push({ k: 'graph estimate', v: 2.43, c: 'b', ph: 4 });
        return a;
      }
      function tokPerGiB() {
        if (S.p === 'qwen') return S.qdt === 'fp8' ? 3652200 / 32.02 : 2131159 / 31.7;
        if (S.p === 'ds') return 2331430 / 17.04;
        return S.compact ? 1233779 / 11 : 876958 / 14;
      }
      function model() {
        const P = PR[S.p], u = S.u[S.p], ps = parts();
        const fixed = ps.reduce((s, x) => s + x.v, 0), budget = u * P.M, prof = budget - fixed;
        const pin = S.p === 'qwen' ? S.qpin || null : S.p === 'glm' ? S.gpin || null : null;
        const kv = pin != null ? pin : Math.max(0, prof);
        const tpg = tokPerGiB(), tokens = kv * tpg, need = P.maxLen / tpg;
        return { P, u, ps, fixed, budget, prof, pin, kv, tokens, need, conc: tokens / P.maxLen };
      }
      function build() {
        ctl.innerHTML = '';
        UI.seg(ctl, { label: 'Recipe', options: [{ v: 'qwen', label: 'Qwen' }, { v: 'ds', label: 'DeepSeek' }, { v: 'glm', label: 'GLM' }], value: S.p, onChange: (v) => { S.p = v; build(); } });
        UI.slider(ctl, { label: 'util (GMU)', min: 0.75, max: 0.9, step: 0.005, value: S.u[S.p], fmt: (v) => sf(v, 3), onInput: (v) => { S.u[S.p] = v; } });
        if (S.p === 'qwen') {
          UI.seg(ctl, { label: 'KV', options: [{ v: 'fp8', label: 'FP8' }, { v: 'bf16', label: 'BF16' }], value: S.qdt, onChange: (v) => { S.qdt = v; } });
          UI.seg(ctl, { label: 'pool', options: [{ v: 0, label: 'profiled' }, { v: 31.34, label: '31.34 GiB' }, { v: 38.52, label: '38.52 GiB' }, { v: 42.38, label: '42.38 GiB' }], value: S.qpin, onChange: (v) => { S.qpin = v; } });
        }
        if (S.p === 'glm') {
          UI.toggle(ctl, { label: 'InstantTensor', value: S.it, onChange: (v) => { S.it = v; } });
          UI.toggle(ctl, { label: 'CG_ESTIMATE=1', value: S.cg, onChange: (v) => { S.cg = v; } });
          UI.toggle(ctl, { label: 'E3 (scratch 560 MiB)', value: S.e3, onChange: (v) => { S.e3 = v; } });
          UI.toggle(ctl, { label: 'compact pages', value: S.compact, onChange: (v) => { S.compact = v; } });
          UI.seg(ctl, { label: 'indexer workspace', options: [{ v: 'rightsize', label: 'rightsize' }, { v: 'stock', label: 'stock' }], value: S.idx, onChange: (v) => { S.idx = v; } });
          UI.seg(ctl, { label: 'pool', options: [{ v: 0, label: 'profiled' }, { v: 14, label: '14 GiB' }, { v: 11, label: '11 GiB' }], value: S.gpin, onChange: (v) => { S.gpin = v; } });
        }
      }
      build();
      const phases = ['weights', 'dummy pass', 'workspaces 🔒', 'graphs', 'KV'];
      const H = V.fig(fig, (g, w, h, t, dt) => {
        const C = V.C(), m = model(), pad = 14, W = w - 2 * pad;
        const cyc = t % 10, ph = Math.min(5, Math.floor(cyc / 1.3) + 1);
        const k = Math.min(1, dt * 7);
        let y = 8;
        // phases
        const pw = W / 5;
        phases.forEach((p, i) => {
          const on = ph === i + 1, past = ph > i + 1;
          const cx = pad + i * pw + pw / 2;
          if (i < 4) V.line(g, cx + 12, y + 12, cx + pw - 12, y + 12, { color: V.alpha(C.muted, past ? 0.8 : 0.3), lw: 1.5 });
          V.dot(g, cx, y + 12, on ? 9 : 7, on ? C.d : past ? V.alpha(C.a, 0.8) : V.alpha(C.muted, 0.3));
          if (on) V.glow(g, cx, y + 12, 22, C.d, 0.7);
          V.text(g, String(i + 1), cx, y + 12.5, { size: 10, weight: 700, align: 'center', base: 'middle', color: onColor(C) });
          V.text(g, w < 520 ? ['weights', 'dummy', 'buffers 🔒', 'graphs', 'KV'][i] : p, cx, y + 34, { size: w < 520 ? 10 : 11, align: 'center', color: on ? C.ink : C.muted, maxW: pw - 4 });
        });
        y += 50;
        // bar
        const sc = W / m.P.M, bh = 38, by = y + 16;
        V.text(g, `memory visible to CUDA: ${sf(m.P.M, 2)} GiB`, pad, y + 8, { size: 11.5, font: 'mono', color: C.muted });
        V.box(g, pad, by, W, bh, { r: 7, fill: V.alpha(C.muted, 0.08), stroke: V.alpha(C.muted, 0.3), lw: 1 });
        let x = pad;
        const items = m.ps.concat([{ k: 'KV', v: m.kv, c: 'e', ph: 5 }]);
        items.forEach((it) => {
          const cur = (S.cur[it.k] = S.cur[it.k] == null ? it.v : V.lerp(S.cur[it.k], it.v, k));
          const ww = Math.max(0, cur * sc), col = C[it.c];
          const vis = ph >= it.ph;
          V.box(g, x, by, Math.max(0.5, ww), bh, { r: 3, fill: V.alpha(col, vis ? (it.k === 'KV' ? 0.75 : 0.6) : 0.12) });
          if (vis && ww > tw(g, sf(it.v, 1), 10.5, 'mono') + 8) V.text(g, sf(it.v, 1), x + ww / 2, by + bh / 2 + 0.5, { size: 10.5, font: 'mono', weight: 700, align: 'center', base: 'middle', color: C.ink });
          x += ww;
        });
        Object.keys(S.cur).forEach((kk) => { if (!items.some((it) => it.k === kk)) { S.cur[kk] = V.lerp(S.cur[kk], 0, k); if (S.cur[kk] < 0.01) delete S.cur[kk]; } });
        const used = x;
        const bx = pad + m.budget * sc;
        if (used > bx + 0.5) { hatch(g, bx, by + 1, used - bx, bh - 2, V.alpha(RED(C), 0.8), 4); }
        if (used < pad + W) hatch(g, Math.max(used, bx), by + 1, pad + W - Math.max(used, bx) - 1, bh - 2, V.alpha(C.muted, 0.35), 7);
        g.strokeStyle = C.ink; g.lineWidth = 1.6; g.setLineDash([5, 4]); g.beginPath(); g.moveTo(bx, by - 8); g.lineTo(bx, by + bh + 8); g.stroke(); g.setLineDash([]);
        const bl = `u·M = ${sf(m.budget, 1)} GiB (u = ${sf(m.u, 3)})`;
        V.text(g, bl, Math.min(bx, w - pad), by + bh + 21, { size: 11, font: 'mono', weight: 600, align: bx > w - pad - 100 ? 'right' : 'center', color: C.ink });
        y = by + bh + 34;
        // legend
        let lx = pad, ly = y + 6;
        items.forEach((it) => {
          const s = `${it.k} ${sf(it.v, it.v < 1 ? 2 : 1)}`;
          const ww = tw(g, s, 11) + 26;
          if (lx + ww > w - pad) { lx = pad; ly += 18; }
          V.box(g, lx, ly - 9, 11, 11, { r: 3, fill: V.alpha(C[it.c], 0.7) });
          V.text(g, s, lx + 15, ly + 1, { size: 11, color: C.muted });
          lx += ww;
        });
        { const s = 'outside the budget (host, OS, page cache)', ww = tw(g, s, 11) + 26; if (lx + ww > w - pad) { lx = pad; ly += 18; } V.box(g, lx, ly - 9, 11, 11, { r: 3, stroke: C.muted, lw: 1 }); hatch(g, lx + 1, ly - 8, 9, 9, V.alpha(C.muted, 0.7), 3); V.text(g, s, lx + 15, ly + 1, { size: 11, color: C.muted }); }
        y = ly + 20;
        // result
        const ok = m.kv >= m.need;
        const L1 = `KV = ${sf(m.kv, 2)} GiB` + (m.pin != null ? '  (byte pin)' : '  (profiled)');
        const L2 = `→ ${nf(m.tokens)} tokens → ${sf(m.conc, 2)}× at max_model_len = ${nf(m.P.maxLen)}`;
        const L3 = ok ? `✓ one max_model_len request fits (needs ≈ ${sf(m.need, 2)} GiB)` : `✗ the boot refuses: one max_model_len request needs ≈ ${sf(m.need, 2)} GiB`;
        V.text(g, L1, pad, y + 16, { size: 16, font: 'display', weight: 700, color: C.e, maxW: W });
        y += 24;
        y += wrap(g, L2, pad, y + 14, W, { size: 12.5, font: 'mono', color: C.ink });
        y += wrap(g, L3, pad, y + 14, W, { size: 12.5, weight: 600, color: ok ? C.a : RED(C) });
        if (m.pin != null && used > bx + 0.5) y += wrap(g, `the pin ignores util: ${sf((used - bx) / sc, 1)} GiB above u·M come out of the host’s headroom`, pad, y + 14, W, { size: 11.5, color: C.d });
        if (m.pin != null) y += wrap(g, `(the profile run would give ${sf(Math.max(0, m.prof), 2)} GiB)`, pad, y + 14, W, { size: 11.5, color: C.muted });
        fitH(H, y + 14);
      });
    }

    /* ======================================================================
       FIG 8 — decoding 1,553,140 tokens (stepper)
       ====================================================================== */
    function figBoot() {
      const fig = root.querySelector('#fig-boot');
      const G = [
        { g: 'g0', n: 'MLA+idx', v: 280, f: '⌈1,000,000/3,584⌉', c: 'a', seg: 1 },
        { g: 'g1', n: 'tail', v: 1, f: 'ring', c: 'd', seg: 0 },
        { g: 'g2', n: 'Mamba', v: 9, f: '2 + 7', c: 'b', seg: 1 },
        { g: 'g3', n: 'Mamba', v: 9, f: '2 + 7', c: 'b', seg: 1 },
        { g: 'g4', n: 'Mamba', v: 9, f: '2 + 7', c: 'b', seg: 1 },
        { g: 'g5', n: 'Mamba', v: 9, f: '2 + 7', c: 'b', seg: 1 },
        { g: 'g6', n: 'drafter', v: 97, f: '⌈6,143/64⌉+1', c: 'c', seg: 33 },
      ];
      const steps = [
        { t: 'The vLLM line', d: 'GLM’s Aug 31 boot printed 1,553,140 tokens and 1.55× at 1M. It looks like a cache of 1.5 million tokens. Let’s redo the math.' },
        { t: 'IDs per request, group by group', d: 'For each group, vLLM computes how many IDs a 1M request takes at peak: 280 in MLA (⌈1,000,000/3,584⌉), 1 in the tail ring, 9 in each Mamba group (2 + 7: current state, previous state and one per speculative token) and 97 in the drafter (⌈(2,047 of window + 4,096 in flight) / 64⌉ + 1; in flight = 2 batches × an MNBT of 2,048).' },
        { t: 'The sum: 414', d: 'The groups take IDs from the same pool, so the denominator is the sum: 414 IDs per 1M request. This boot’s pool had 643 IDs (the number isn’t logged; the design doc recovers it from the line itself).' },
        { t: '643 ÷ 414 × 1,000,000', d: 'The maximum concurrency is 643/414 = 1.553. Multiplied by max_model_len, it becomes 1,553,140 “tokens”. That’s concurrency in token units, not the size of a cache.' },
        { t: 'How much conversation fits', d: 'Storing one 3,584-token segment of conversation costs 38 IDs: 1 MLA, 4 Mamba and 33 for the drafter (the tail stays out of the prefix cache). With 642 usable IDs (0 is the null one): 642 // 38 = 16 segments = 57,344 tokens. 87% of the IDs go to the drafter.' },
        { t: 'The honest log', d: '<code>patch_kv_capacity_log.py</code> prints this math at every boot. In the current boot, with the rightsize workspace, it’s 820 IDs: 1.98× and ~75,264 tokens (21 segments). Read your own boot’s summary.' },
      ];
      const st = UI.stepper(fig, steps, { autoplay: 7000 });
      const H = V.fig(fig, (g, w, h, t) => {
        const C = V.C(), i = st.i, el = st.elapsed(), pad = 14, W = w - 2 * pad;
        let y = 10;
        const logBox = (lines, col) => {
          let hh = 0; const ls = [];
          lines.forEach((l) => { const x = wrapLines(g, l, W - 24, 11.5, 'mono'); ls.push(...x); });
          hh = ls.length * 17 + 18;
          V.box(g, pad, y, W, hh, { r: 8, fill: V.alpha(C.muted, 0.08), stroke: V.alpha(col || C.muted, 0.5), lw: 1 });
          ls.forEach((l, k) => V.text(g, l, pad + 12, y + 20 + k * 17, { size: 11.5, font: 'mono', color: C.ink }));
          y += hh + 12;
        };
        if (i === 0) {
          logBox(['INFO [kv_cache_utils.py] GPU KV cache size: 1,553,140 tokens, Maximum concurrency for 1,000,000 tokens per request: 1.55x']);
          const pulse = 0.6 + 0.4 * Math.sin(t * 3);
          V.text(g, '1.5 million tokens of cache?', w / 2, y + 60, { size: w < 520 ? 22 : 30, font: 'display', weight: 700, align: 'center', color: V.alpha(C.d, pulse) });
          V.text(g, 'spoiler: no', w / 2, y + 92, { size: 13, align: 'center', color: C.muted });
          fitH(H, Math.max(300, y + 120));
          return;
        }
        if (i >= 1 && i <= 3) {
          const small = w < 520;
          const th0 = wrap(g, 'IDs a 1,000,000-token request takes in each group (at peak)', pad, y + 10, W, { size: 12, weight: 600, color: C.ink });
          const mvS = i === 1 ? 0 : i === 2 ? V.ease.inOut(V.clamp((el - 0.2) / 1.2, 0, 1)) : 1;
          const baseY = y + th0 + 198, maxH = 170, sc = V.lerp(maxH / 300, maxH / 643, mvS);
          const right = small ? 130 : 270, slotW = (W - right) / 7;
          const colW = Math.min(64, slotW * 0.72), gapc = slotW - colW;
          const grow = i === 1 ? V.ease.out(V.clamp(el / 1.6, 0, 1)) : 1;
          const mv = i >= 2 ? V.ease.inOut(V.clamp((el - (i === 2 ? 0.2 : -5)) / 1.2, 0, 1)) : 0;
          G.forEach((gr, k) => {
            const x = pad + k * (colW + gapc);
            const hh = Math.max(3, gr.v * sc * grow);
            const col = C[gr.c];
            // the per-group bars fade (they do not disappear) once they become the stack
            V.box(g, x, baseY - hh, colW, hh, { r: 4, fill: V.alpha(col, 0.7 - 0.45 * mv) });
            V.text(g, nf(Math.round(gr.v * grow)), x + colW / 2, baseY - hh - 6, { size: small ? 10.5 : 11.5, font: 'mono', weight: 700, align: 'center', color: V.alpha(C.ink, 1 - 0.45 * mv) });
            if (!small) V.text(g, gr.n, x + colW / 2, baseY + 14, { size: 10.5, align: 'center', color: C.muted, maxW: colW + gapc });
            else if (k === 0 || k === 1 || k === 6) V.text(g, ['MLA', 'tail', '', '', '', '', 'draft'][k], x + colW / 2, baseY + 14, { size: 9.5, align: 'center', color: C.muted });
            else if (k === 2) V.text(g, '4 × Mamba', x + 1.5 * slotW + colW / 2, baseY + 14, { size: 9.5, align: 'center', color: C.b });
            if (!small) V.text(g, gr.f, x + colW / 2, baseY + 28, { size: 9, font: 'mono', align: 'center', color: C.muted, maxW: colW + gapc });
          });
          if (i >= 2) {
            const p = mv, sw2 = small ? 24 : 46, scF = maxH / 643;
            const poolX = pad + 7 * slotW + (small ? 8 : 14), stackX = poolX + sw2 + (small ? 8 : 18);
            // pool (reference) and the stack of IDs of one request
            V.box(g, poolX, baseY - 643 * scF, sw2, 643 * scF, { r: 3, fill: V.alpha(C.e, 0.22), stroke: C.e, lw: 1.2 });
            V.text(g, '643', poolX + sw2 / 2, baseY - 643 * scF - 6, { size: 12.5, font: 'mono', weight: 700, align: 'center', color: C.e });
            V.text(g, 'pool', poolX + sw2 / 2, baseY + 14, { size: 10.5, align: 'center', color: C.muted });
            let acc = 0;
            const mids = [];
            G.forEach((gr) => {
              const hh = gr.v * scF * p;
              V.box(g, stackX, baseY - acc - hh, sw2, Math.max(0.5, hh - 0.5), { r: 2, fill: V.alpha(C[gr.c], 0.85) });
              mids.push(baseY - acc - hh / 2);
              acc += hh;
            });
            V.text(g, nf(Math.round(414 * p)), stackX + sw2 / 2, baseY - 414 * scF * p - 6, { size: 12.5, font: 'mono', weight: 700, align: 'center', color: C.ink });
            V.text(g, small ? 'req.' : 'per req.', stackX + sw2 / 2, baseY + 14, { size: 10.5, align: 'center', color: C.muted });
            if (p > 0.6) {
              const a = V.clamp((p - 0.6) / 0.4, 0, 1);
              // side callouts: 280 MLA · 1 tail · 36 Mamba · 97 drafter
              const calls = [
                { y: mids[0], s: small ? '280 MLA' : '280 MLA+idx', c: C.a },
                { y: mids[1], s: '1 tail', c: C.d },
                { y: (mids[2] + mids[5]) / 2, s: small ? '36 Mamba' : '36 Mamba (4 × 9)', c: C.b },
                { y: mids[6], s: small ? '97 draft' : '97 drafter', c: C.c },
              ];
              let last = Infinity;
              calls.forEach((cl) => { cl.ly = Math.min(cl.y, last - 13); last = cl.ly; });
              const lx0 = stackX + sw2 + 3, lx1 = stackX + sw2 + (small ? 10 : 18);
              calls.forEach((cl) => {
                g.strokeStyle = V.alpha(cl.c, 0.8 * a); g.lineWidth = 1;
                g.beginPath(); g.moveTo(lx0, cl.y); g.lineTo(lx1 - 4, cl.ly); g.lineTo(lx1, cl.ly); g.stroke();
                V.text(g, cl.s, lx1 + 3, cl.ly + 4, { size: small ? 9.5 : 11, font: 'mono', weight: 600, color: V.alpha(cl.c, a), maxW: w - pad - lx1 - 3 });
              });
            }
          }
          y = baseY + (small ? 44 : 52);
          if (i === 3) {
            const p = V.clamp(el / 1.4, 0, 1);
            const s1 = `643 ÷ 414 = ${sf(1 + 0.55314 * p, 3)}`, s2 = `× 1,000,000 = ${nf(1000000 * (1 + 0.55314 * p))} “tokens”`;
            V.text(g, s1, pad, y + 6, { size: small ? 15 : 19, font: 'display', weight: 700, color: C.ink });
            V.text(g, s2, pad, y + 32, { size: small ? 13 : 16, font: 'mono', weight: 700, color: C.d, maxW: W });
            y += 48;
            y += wrap(g, 'it’s concurrency in token units: “1.55 requests of 1M fit”, not “1.5M tokens of cache fit”', pad, y + 12, W, { size: 12, color: C.muted });
          }
          fitH(H, y + 12);
          return;
        }
        // steps 4 and 5: the pool of 642 IDs
        if (i === 5) {
          logBox(['[glm53-kv-capacity-log] usable block ids: 642 (num_blocks=643 incl. the null block; 414 ids per 1,000,000-token request => 1.55x)', 'ids per 3584-token cached segment across groups: 38 (per group: [1, 0, 1, 1, 1, 1, 33])', 'cached-conversation capacity at this alignment ≈ 57,344 tokens = 16 segments'], C.a);
        }
        V.text(g, 'pool: 642 usable IDs · each stored 3,584-token segment costs 38 IDs', pad, y + 10, { size: 12, weight: 600, color: C.ink, maxW: W });
        y += 22;
        const cell = w < 520 ? 9 : 11, gp = 2, cols = Math.max(10, Math.floor((W + gp) / (cell + gp))), rows = Math.ceil(642 / cols);
        const fillN = i === 4 ? Math.floor(V.clamp(el / 2.6, 0, 1) * 608) : 608;
        for (let k = 0; k < 642; k++) {
          const x = pad + (k % cols) * (cell + gp), yy = y + Math.floor(k / cols) * (cell + gp);
          let col = V.alpha(C.muted, 0.14);
          if (k < fillN) { const pos = k % 38; col = pos === 0 ? V.alpha(C.a, 0.9) : pos <= 4 ? V.alpha(C.b, 0.85) : V.alpha(C.c, 0.7); }
          V.box(g, x, yy, cell, cell, { r: 2, fill: col });
        }
        y += rows * (cell + gp) + 10;
        const segs = Math.floor(fillN / 38);
        let lx = pad;
        [[C.a, '1 MLA'], [C.b, '4 Mamba'], [C.c, '33 drafter'], [C.muted, 'leftover']].forEach(([c, s]) => { V.box(g, lx, y, 11, 11, { r: 2, fill: V.alpha(c, 0.8) }); V.text(g, s, lx + 15, y + 10, { size: 11, color: C.muted }); lx += tw(g, s, 11) + 32; });
        y += 26;
        V.text(g, `${segs} segments × 3,584 = ${nf(segs * 3584)} tokens of stored conversation`, pad, y + 8, { size: w < 520 ? 13 : 16, font: 'mono', weight: 700, color: C.a, maxW: W });
        y += 24;
        if (i === 5) { y += wrap(g, 'current boot (rightsize workspace): 820 IDs → 1.98× · ≈ 75,264 tokens (21 segments)', pad, y + 10, W, { size: 12, font: 'mono', color: C.muted }); }
        fitH(H, y + 12);
      });
    }

    /* ======================================================================
       FIG 9 — the tank and the turnstile (ceilings, not reservations)
       ====================================================================== */
    function figTeto() {
      const fig = root.querySelector('#fig-teto');
      const ctl = UI.controls(fig);
      const RC = {
        ds: { name: 'DeepSeek', pool: 2331430, maxLen: 1048576, seqs: 6, Ls: [10000, 50000, 100000, 200000, 300000, 500000, 750000, 1048576], pre: [[6, 50000, '6 × 50k'], [6, 200000, '6 × 200k'], [6, 500000, '6 × 500k'], [6, 1048576, '6 × 1M']] },
        qwen: { name: 'Qwen', pool: 3652200, maxLen: 262144, seqs: 8, Ls: [8192, 16384, 32768, 65536, 131072, 200000, 262144], pre: [[8, 131072, '8 × 128k'], [8, 262144, '8 × 262k'], [14, 262144, '14 × 262k']] },
      };
      const S = { r: 'ds', n: 6, L: 200000, cur: [] };
      function build() {
        ctl.innerHTML = '';
        const R = RC[S.r];
        UI.seg(ctl, { label: 'Recipe', options: [{ v: 'ds', label: 'DeepSeek' }, { v: 'qwen', label: 'Qwen (FP8)' }], value: S.r, onChange: (v) => { S.r = v; const RR = RC[v]; S.n = RR.pre[1][0]; S.L = RR.pre[1][1]; build(); } });
        const sn = UI.slider(ctl, { label: 'requests', min: 1, max: 16, step: 1, value: S.n, onInput: (v) => { S.n = v; } });
        let li = R.Ls.indexOf(S.L); if (li < 0) li = 0;
        const sL = UI.slider(ctl, { label: 'tokens/req.', min: 0, max: R.Ls.length - 1, step: 1, value: li, fmt: (i) => nf(R.Ls[i]), onInput: (i) => { S.L = R.Ls[i]; } });
        R.pre.forEach(([n, L, lbl]) => UI.button(ctl, lbl, () => { sn.set(n); sL.set(R.Ls.indexOf(L)); }));
      }
      build();
      const H = V.fig(fig, (g, w, h, t, dt) => {
        const C = V.C(), R = RC[S.r], pad = 14, W = w - 2 * pad;
        const fitPool = Math.floor(R.pool / S.L), k = Math.min(S.n, R.seqs, fitPool), q = S.n - k;
        const byGate = S.n > R.seqs && fitPool >= R.seqs;
        const reason = S.n <= Math.min(R.seqs, fitPool) ? 'everything fits: nobody waits' : fitPool < Math.min(S.n, R.seqs) ? 'limited by the pool (tank)' : 'limited by max_num_seqs (turnstile)';
        const pal = [C.a, C.b, C.c, C.d, C.e, C.f];
        const wide = w >= 620;
        let y = 10;
        const qx = pad, qW = wide ? 150 : W;
        const tx = wide ? pad + qW + 24 : pad + 18, tW = wide ? w - tx - pad : W - 18;
        const ty = wide ? y + 40 : y + (q > 0 ? 74 : 40), th = 58;
        // queue and turnstile
        V.text(g, `queue: ${q}`, qx, y + 12, { size: 12, font: 'mono', weight: 700, color: q ? C.d : C.muted });
        const maxIc = wide ? 8 : Math.floor((W - 30) / 16);
        for (let j = 0; j < Math.min(q, maxIc); j++) {
          const ix = wide ? qx + 8 + (j % 4) * 26 : qx + 8 + j * 16, iy = wide ? y + 40 + Math.floor(j / 4) * 34 : y + 36;
          const bob = Math.sin(t * 3 + j) * 2, col = pal[(k + j) % 6];
          V.dot(g, ix, iy + bob, 5, V.alpha(col, 0.9));
          V.box(g, ix - 6, iy + 6 + bob, 12, 12, { r: 5, fill: V.alpha(col, 0.55) });
        }
        if (q > maxIc) V.text(g, `+${q - maxIc}`, wide ? qx + 8 : qx + 8 + maxIc * 16, wide ? y + 118 : y + 42, { size: 11, font: 'mono', color: C.muted });
        const gx = wide ? tx - 12 : pad + 5;
        const gy0 = ty - 6, gy1 = ty + th + 6;
        g.strokeStyle = byGate ? C.d : V.alpha(C.ink, 0.7); g.lineWidth = 3;
        g.beginPath(); g.moveTo(gx, gy0); g.lineTo(gx, gy1); g.stroke();
        for (let j = 0; j < 3; j++) { const yy = gy0 + 10 + j * 22; g.beginPath(); g.moveTo(gx - 7, yy + Math.sin(t * 2 + j) * 1.5); g.lineTo(gx + 7, yy - Math.sin(t * 2 + j) * 1.5); g.stroke(); }
        V.text(g, `turnstile: max_num_seqs=${R.seqs}`, wide ? gx - 6 : w - pad, wide ? gy1 + 18 : y + 12, { size: 11, font: 'mono', color: byGate ? C.d : C.muted, align: 'right' });
        // tank
        const sc = tW / R.pool;
        V.box(g, tx, ty, tW, th, { r: 10, fill: V.alpha(C.muted, 0.07), stroke: V.alpha(C.ink, 0.5), lw: 1.4 });
        for (let j = 0; j < k; j++) {
          const target = j * S.L * sc;
          const cur = (S.cur[j] = S.cur[j] == null ? target : V.lerp(S.cur[j], target, Math.min(1, dt * 6)));
          const ww = Math.min(S.L * sc, tW - cur);
          if (ww <= 0) continue;
          const col = pal[j % 6];
          V.box(g, tx + cur + 1, ty + 4, Math.max(1, ww - 2), th - 8, { r: 6, fill: V.alpha(col, 0.62) });
          if (ww > 26) V.text(g, 'r' + (j + 1), tx + cur + ww / 2, ty + th / 2 + 0.5, { size: 11, font: 'mono', weight: 700, align: 'center', base: 'middle', color: C.ink });
        }
        S.cur.length = k;
        const nx = Math.floor(R.pool / R.maxLen);
        for (let j = 1; j <= nx; j++) {
          const xx = tx + j * R.maxLen * sc;
          g.strokeStyle = V.alpha(C.ink, 0.55); g.lineWidth = 1; g.setLineDash([4, 3]); g.beginPath(); g.moveTo(xx, ty - 4); g.lineTo(xx, ty + th + 4); g.stroke(); g.setLineDash([]);
          V.text(g, j + '×', xx, ty - 8, { size: 10.5, font: 'mono', align: 'center', color: C.muted });
        }
        V.text(g, `pool ${nf(R.pool)} tokens · max concurrency ${sf(R.pool / R.maxLen, 2)}× at ${nf(R.maxLen)}`, tx + tW, ty + th + 18, { size: 11, font: 'mono', align: 'right', color: C.muted, maxW: tW });
        y = ty + th + 34;
        const sum = S.n * S.L;
        const lines = [
          { s: `requested: ${S.n} × ${nf(S.L)} = ${nf(sum)} tokens ${sum <= R.pool ? '≤' : '>'} pool`, c: sum <= R.pool ? C.ink : RED(C) },
          { s: `resident: ${k} · waiting: ${q}`, c: C.ink },
          { s: reason, c: q ? C.d : C.a },
        ];
        lines.forEach((l, j) => { y += wrap(g, l.s, pad, y + 12, W, { size: 12.5, font: j === 2 ? 'sans' : 'mono', weight: j === 2 ? 600 : 400, color: l.c }); });
        fitH(H, y + 12);
      });
    }

    /* ======================================================================
       FIG 11 — TensorFold: extents in a shared pool (simulation)
       ====================================================================== */
    const tfCache = { k: null, v: '' };
    const TFC = (C) => { if (tfCache.k !== C) { tfCache.k = C; tfCache.v = getComputedStyle(document.documentElement).getPropertyValue('--tf').trim() || C.d; } return tfCache.v; };
    function figExtents() {
      const fig = root.querySelector('#fig-extents');
      const ctl = UI.controls(fig);
      const N = 48, SLOTS = 4, WIN = 16, ROWS = 16 / 2048, TICK = 0.55, STEP = 0.5, KEEP_MAX = 4, MSG_T = 0.8, FRAG_HOLD = 3;
      const LET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';
      const T = {
        slots: 'slots · each stream’s state, outside the pool',
        slot: 'slot', free: 'free', fill: 'filling', dec: 'decoding', paused: '⏸ no room',
        pool: 'rank 0 · pool of per-token rows (1 cell = 2,048 tokens)',
        queue: 'queue (prompt in cells)', empty: '— empty',
        msgs: 'rank 0’s messages (one per iteration)', r1: 'rank 1 · applies the same operations',
        same: 'pools equal ✓', applying: (n) => `applying ${n} message${n > 1 ? 's' : ''}…`,
        kept: 'kept',
        stats: (f, gmax, st) => `free ${f}/${N} · largest hole ${gmax} · compactions ${st.compact} · evictions ${st.evict} · given back ${st.back}`,
        leg: ['written rows', 'prompt to fill', 'unit headroom', 'kept prompt (gives way first)', 'paused'],
        hint: 'click a kept prompt to continue the conversation',
        frag: { queue: '⚠ queue stuck: there are free rows, but no hole fits', grow: '⚠ stream paused: there are free rows, but no hole fits' },
        ops: {
          admit: (L, s, b, n, r) => `ADMIT ${L} slot ${s} [${b}, ${b + n})${r ? ' resume' : ''}`,
          grow: (l, d) => `GROW ${l} +${d}`,
          move: (l, b, c) => `MOVE ${l} → ${b}${c ? ' (compact)' : ''}`,
          evict: (L) => `EVICT kept ${L}`,
          fill: (L) => `FILL ${L}`,
          round: (ls) => `ROUND ${ls}`,
          finish: (L, n) => `FINISH ${L} (keeps ${n} cell${n > 1 ? 's' : ''})`,
          back: (L) => `FINISH ${L} → queue`,
        },
        btnNew: '＋ conversation', btnLong: '＋ long conversation', tCompact: 'compaction', tAuto: 'automatic arrivals', btnReset: '↻ restart',
      };
      const S = { compact: true, auto: true, seed: 32 };
      let P = null;
      const au = (n) => Math.max(1, Math.ceil(n - 1e-9));
      const sorted = () => P.ext.sort((a, b) => a.base - b.base);
      function gaps(ignore) {
        const out = []; let at = 0;
        sorted().forEach((x) => { if (x === ignore) return; if (x.base > at) out.push([at, x.base - at]); at = Math.max(at, x.base + x.size); });
        if (at < N) out.push([at, N - at]);
        return out;
      }
      const freeRows = () => N - P.ext.reduce((s, x) => s + x.size, 0);
      const maxGap = () => gaps(null).reduce((m, q) => Math.max(m, q[1]), 0);
      function place(size, ignore) { for (const [b, n] of gaps(ignore)) if (n >= size) return b; return null; }
      function roomAfter(x) { const nx = P.ext.filter((y) => y !== x && y.base >= x.base + x.size).map((y) => y.base); return Math.min(N, ...nx) - (x.base + x.size); }
      const lbl = (x) => (x.owner != null ? x.L : T.kept + ' ' + x.L);
      const setD = (x) => ({ k: 'set', eid: x.eid, base: x.base, size: x.size, col: x.col, kind: x.owner != null ? 'live' : 'kept', L: x.L });
      function op(txt, data) { P.cur.ops.push(txt); if (data) P.cur.data.push(data); }
      function flash(x) { P.flash[x.eid] = P.clock; }
      function moveX(x, base, size, comp) { x.base = base; x.size = size; P.st.move++; op(T.ops.move(lbl(x), base, comp), setD(x)); flash(x); }
      function evict(c) {
        P.kept.splice(P.kept.indexOf(c), 1);
        P.ext.splice(P.ext.indexOf(c.x), 1);
        P.st.evict++;
        op(T.ops.evict(c.L), { k: 'del', eid: c.x.eid });
      }
      const victims = (protect) => P.kept.filter((c) => c.x.owner == null && !protect.includes(c.x)).sort((a, b) => a.used - b.used);
      function compact() {
        const xs = sorted().slice(); let at = 0, n = 0;
        xs.forEach((x) => { if (x.base !== at) { moveX(x, at, x.size, true); n++; } at += x.size; });
        if (n) P.st.compact++;
      }
      function room(need, protect = []) {
        let b = place(need);
        while (b == null) { const v = victims(protect); if (!v.length) break; evict(v[0]); b = place(need); }
        if (b == null && freeRows() >= need) { if (S.compact) { compact(); b = place(need); } else fragHit('queue'); }
        return b;
      }
      // why something waited with plenty of free rows (only happens with compaction off)
      function fragHit(kind) { P.frag = true; P.fragKind = kind; }
      function grow(x, size, protect = [], why = null) {
        if (size <= x.size) return true;
        protect = protect.concat([x]);
        const tryIt = () => {
          if (roomAfter(x) >= size - x.size) { const d = size - x.size; x.size = size; op(T.ops.grow(lbl(x), d), setD(x)); return true; }
          const b = place(size, x);
          if (b != null) { moveX(x, b, size, false); return true; }
          return false;
        };
        for (;;) { if (tryIt()) return true; const v = victims(protect); if (!v.length) break; evict(v[0]); }
        if (freeRows() + x.size >= size) { if (S.compact) { compact(); if (tryIt()) return true; } else if (why) fragHit(why); }
        return false;
      }
      function mkReq(kind) {
        const r = P.rng, long = kind === 'long', conv = P.conv++;
        const prompt = long ? 11 + r() * 4 : 2 + r() * 5;
        const out = Math.min(WIN - prompt, long ? 2 + r() * 2 : 1 + r() * 3);
        return { sid: P.sid++, conv, L: LET[conv % 26], prompt, out, col: conv % 6 };
      }
      function follow(c) {
        const r = P.rng, prompt = Math.min(WIN - 1.5, c.len + 0.6 + r() * 1.6);
        return { sid: P.sid++, conv: c.conv, L: c.L, prompt, out: Math.min(WIN - prompt, 0.8 + r() * 1.6), col: c.col };
      }
      const busy = () => new Set(P.lanes.map((l) => l.conv).concat(P.queue.map((q) => q.conv)));
      function autoReq() {
        const r = P.rng(), b = busy();
        const ks = P.kept.filter((c) => !b.has(c.conv));
        if (ks.length && r < 0.45) return follow(ks[P.rng.int(ks.length)]);
        return mkReq(r > 0.92 ? 'long' : 'new');
      }
      function fits(q) {
        if (!P.lanes.length) return true;
        const need = au(q.prompt + ROWS);
        const roomN = freeRows() + P.ext.filter((x) => x.owner == null).reduce((s, x) => s + x.size, 0);
        return roomN >= need + P.lanes.filter((l) => !l.filling).length;
      }
      function admit(q) {
        const used = new Set(P.lanes.map((l) => l.slot));
        let slot = 0; while (used.has(slot)) slot++;
        if (slot >= SLOTS) return false;
        const need = au(q.prompt + ROWS);
        let x = null, pos = 0;
        const hit = P.kept.filter((c) => c.conv === q.conv && c.len <= q.prompt && c.x.owner == null).sort((a, b) => b.len - a.len)[0];
        if (hit && grow(hit.x, need, [hit.x])) { x = hit.x; pos = hit.len; P.kept.splice(P.kept.indexOf(hit), 1); }
        if (!x) {
          const b = room(need);
          if (b == null) return false;
          x = { eid: P.eid++, base: b, size: need, owner: null, db: b, ds: 0 };
          P.ext.push(x);
        }
        x.owner = q.sid; x.col = q.col; x.L = q.L;
        P.lanes.push(Object.assign({}, q, { pos, slot, x, filling: true, paused: false, order: P.tick }));
        op(T.ops.admit(q.L, slot, x.base, x.size, pos > 0), setD(x));
        flash(x);
        return true;
      }
      function finish(l, back) {
        P.lanes.splice(P.lanes.indexOf(l), 1);
        const x = l.x; x.owner = null;
        if (back) {
          P.ext.splice(P.ext.indexOf(x), 1);
          op(T.ops.back(l.L), { k: 'del', eid: x.eid });
          P.st.back++;
          P.queue.unshift({ sid: l.sid, conv: l.conv, L: l.L, prompt: l.prompt, out: l.out, col: l.col });
          return;
        }
        x.size = Math.min(x.size, au(l.prompt));
        P.kept.push({ conv: l.conv, L: l.L, len: l.prompt, x, used: P.tick, col: l.col });
        op(T.ops.finish(l.L, x.size), setD(x));
        while (P.kept.length > KEEP_MAX) evict(P.kept.slice().sort((a, b) => a.used - b.used)[0]);
      }
      function fill(l) {
        const chunk = P.lanes.some((m) => !m.filling) ? 0.5 : 1;
        l.pos = Math.min(l.prompt, l.pos + chunk);
        op(T.ops.fill(l.L));
        if (l.pos >= l.prompt - 1e-9) l.filling = false;
      }
      function round() {
        const dec = P.lanes.filter((l) => !l.filling);
        dec.forEach((l) => { l.paused = !grow(l.x, au(Math.min(l.prompt + l.out, l.pos + STEP) + ROWS), [l.x], 'grow'); });
        const act = dec.filter((l) => !l.paused);
        if (!act.length) return;
        op(T.ops.round(act.map((l) => l.L).join('')));
        act.forEach((l) => { l.pos = Math.min(l.prompt + l.out, l.pos + STEP); });
        act.filter((l) => l.pos >= l.prompt + l.out - 1e-9).forEach((l) => finish(l, false));
      }
      function iterate() {
        P.tick++; P.cur = { n: P.tick, ops: [], data: [], t: P.clock };
        if (S.auto && P.tick >= P.nextArr) { if (P.queue.length < 3) P.queue.push(autoReq()); P.nextArr = P.tick + 3 + P.rng.int(5); }
        P.frag = false;
        while (P.queue.length && P.lanes.length < SLOTS && fits(P.queue[0])) { if (!admit(P.queue[0])) { fragHit('queue'); break; } P.queue.shift(); }
        const filling = P.lanes.filter((l) => l.filling).sort((a, b) => a.order - b.order);
        const dec = P.lanes.filter((l) => !l.filling);
        if (filling.length && (P.fillTurn || !dec.length)) fill(filling[0]);
        else if (dec.length) round();
        P.fillTurn = !P.fillTurn;
        const dec2 = P.lanes.filter((l) => !l.filling);
        if (dec2.length && dec2.every((l) => l.paused) && !P.lanes.some((l) => l.filling)) finish(dec2.slice().sort((a, b) => b.order - a.order)[0], true);
        if (P.frag) P.fragAt = P.clock;
        if (P.cur.ops.length) { P.log.unshift(P.cur); if (P.log.length > 8) P.log.length = 8; P.msgs.push(P.cur); }
      }
      function deliver() {
        while (P.msgs.length && P.clock - P.msgs[0].t >= MSG_T) {
          P.msgs.shift().data.forEach((d) => { if (d.k === 'del') P.r1.delete(d.eid); else P.r1.set(d.eid, d); });
        }
      }
      function reset() {
        P = { rng: V.rng(S.seed * 7919 + 3), ext: [], eid: 1, lanes: [], queue: [], kept: [], log: [], msgs: [], r1: new Map(), sid: 0, conv: 0, clock: 0, acc: 0, tick: 0, nextArr: 1, fillTurn: true, cur: null, st: { compact: 0, evict: 0, move: 0, back: 0 }, flash: {}, hits: [], frag: false, fragKind: 'queue', fragAt: null };
        for (let i = 0; i < 64; i++) { P.clock += TICK; iterate(); deliver(); }
        P.ext.forEach((x) => { x.db = x.base; x.ds = x.size; });
      }
      reset();
      const addQ = (q) => { if (P.queue.length < 6) P.queue.push(q); };
      UI.button(ctl, T.btnNew, () => addQ(mkReq('new')));
      UI.button(ctl, T.btnLong, () => addQ(mkReq('long')));
      UI.toggle(ctl, { label: T.tCompact, value: true, onChange: (v) => { S.compact = v; } });
      UI.toggle(ctl, { label: T.tAuto, value: true, onChange: (v) => { S.auto = v; } });
      UI.button(ctl, T.btnReset, () => { S.seed++; reset(); });
      const H = V.fig(fig, (g, w, h, t, dt) => {
        const C = V.C(), pal = [C.a, C.b, C.c, C.d, C.e, C.f], TF = TFC(C);
        if (!LA.paused) {
          const d = Math.min(dt, 0.1); P.clock += d; P.acc += d;
          let n = 0; while (P.acc >= TICK && n++ < 3) { P.acc -= TICK; iterate(); }
          deliver();
        }
        const k = Math.min(1, dt * 6);
        P.ext.forEach((x) => { x.db = x.db == null ? x.base : V.lerp(x.db, x.base, k); x.ds = V.lerp(x.ds || 0, x.size, k); });
        const pad = 14, wide = w >= 720;
        const sideW = wide ? Math.min(320, Math.round(w * 0.34)) : w - 2 * pad;
        const mainW = wide ? w - 2 * pad - sideW - 22 : w - 2 * pad;
        const cols = mainW >= 560 ? 24 : mainW >= 330 ? 16 : 12;
        const rowsN = N / cols, cw = mainW / cols, ch = wide ? 30 : 28, rg = 9;
        let y = 6;
        // slots
        V.text(g, T.slots, pad, y + 12, { size: 12, weight: 600, color: C.muted, maxW: mainW });
        y += 22;
        const sgap = 8, perRow = mainW < 480 ? 2 : 4, sw = (mainW - (perRow - 1) * sgap) / perRow, sh = 40;
        const ys0 = y;
        for (let s = 0; s < SLOTS; s++) {
          const l = P.lanes.find((m) => m.slot === s), sx = pad + (s % perRow) * (sw + sgap);
          y = ys0 + Math.floor(s / perRow) * (sh + sgap);
          const col = l ? pal[l.col] : C.muted;
          V.box(g, sx, y, sw, sh, { r: 8, fill: V.alpha(col, l ? 0.14 : 0.05), stroke: V.alpha(col, l ? 0.75 : 0.3), lw: 1.2 });
          V.text(g, `${T.slot} ${s}`, sx + 8, y + 13, { size: 10.5, font: 'mono', color: C.muted });
          if (l) {
            V.dot(g, sx + 15, y + 28, 8, V.alpha(col, 0.9));
            V.text(g, l.L, sx + 15, y + 28.5, { size: 10.5, weight: 700, align: 'center', base: 'middle', color: onColor(C) });
            const st = l.filling ? `${T.fill} ${Math.round((100 * l.pos) / l.prompt)}%` : l.paused ? T.paused : T.dec;
            V.text(g, st, sx + 28, y + 29, { size: 11, color: l.paused ? RED(C) : C.ink, base: 'middle', maxW: sw - 34 });
          } else V.text(g, T.free, sx + 8, y + 29, { size: 11, color: C.muted, base: 'middle' });
        }
        y += sh + 14;
        // rank 0's pool
        V.text(g, T.pool, pad, y + 12, { size: 12, weight: 600, color: C.ink, maxW: mainW });
        y += 22;
        const gy = y, rowY = (r) => gy + r * (ch + rg);
        for (let u = 0; u < N; u++) {
          const r = Math.floor(u / cols), c = u % cols;
          V.box(g, pad + c * cw + 1, rowY(r), cw - 2, ch, { r: 4, fill: V.alpha(C.muted, 0.07) });
        }
        function pieces(a, len, fn) {
          let s = a; const e = a + len;
          while (s < e - 1e-3) {
            const r = Math.min(rowsN - 1, Math.floor(s / cols + 1e-9)), se = Math.min(e, (r + 1) * cols);
            fn(pad + (s - r * cols) * cw, rowY(r), (se - s) * cw, s === a);
            s = se;
          }
        }
        P.hits = [];
        P.ext.forEach((x) => {
          const l = P.lanes.find((m) => m.x === x), col = pal[x.col], kept = x.owner == null;
          pieces(x.db, x.ds, (px, py, pw, first) => {
            if (kept) {
              V.box(g, px + 1.5, py, pw - 3, ch, { r: 6, fill: V.alpha(C.muted, 0.12), stroke: V.alpha(C.muted, 0.7), lw: 1.2 });
              hatch(g, px + 2.5, py + 1, pw - 5, ch - 2, V.alpha(C.muted, 0.55), 6);
              P.hits.push({ x: px, y: py, w: pw, h: ch, ext: x });
            } else V.box(g, px + 1.5, py, pw - 3, ch, { r: 6, fill: V.alpha(col, 0.07), stroke: V.alpha(col, 0.75), lw: 1.3 });
          });
          if (l) {
            const wr = Math.min(l.pos, x.ds);
            pieces(x.db, wr, (px, py, pw) => V.box(g, px + 1.5, py, Math.max(2, pw - 3), ch, { r: 6, fill: V.alpha(col, 0.85) }));
            if (l.filling && l.prompt > l.pos) pieces(x.db + wr, Math.max(0, Math.min(l.prompt, x.ds) - wr), (px, py, pw) => V.box(g, px + 1.5, py + 3, Math.max(2, pw - 3), ch - 6, { r: 4, fill: V.alpha(col, 0.3) }));
            if (l.paused) pieces(x.db, x.ds, (px, py, pw) => { g.save(); g.setLineDash([4, 3]); g.strokeStyle = RED(C); g.lineWidth = 2; V.rr(g, px + 1.5, py - 2, pw - 3, ch + 4, 7); g.stroke(); g.restore(); });
          }
          const f = P.flash[x.eid];
          if (f != null && P.clock - f < 0.9) { const a = 1 - (P.clock - f) / 0.9; pieces(x.db, x.ds, (px, py, pw) => { g.strokeStyle = V.alpha(C.ink, 0.6 * a); g.lineWidth = 2.5; V.rr(g, px, py - 2, pw, ch + 4, 7); g.stroke(); }); }
          let lab = false;
          pieces(x.db, x.ds, (px, py, pw, first) => {
            if (!first || lab || pw < 16) return; lab = true;
            const s = kept && pw > 92 ? `${x.L} · ${T.kept}` : x.L;
            const onSolid = l && Math.min(l.pos, x.ds) * cw > 22;
            V.text(g, s, px + 9, py + ch / 2 + 0.5, { size: 11.5, weight: 700, base: 'middle', color: kept ? C.ink : onSolid ? onColor(C) : C.ink, maxW: pw - 12 });
          });
        });
        if (H.pointer.clicked) {
          const hit = P.hits.find((q) => H.pointer.x >= q.x && H.pointer.x <= q.x + q.w && H.pointer.y >= q.y && H.pointer.y <= q.y + q.h);
          if (hit) { const c = P.kept.find((m) => m.x === hit.ext); if (c && !busy().has(c.conv)) addQ(follow(c)); }
        }
        y = gy + rowsN * (ch + rg) + 4;
        V.text(g, T.hint, pad, y + 8, { size: 11, color: C.muted, maxW: mainW });
        y += 22;
        // queue
        V.text(g, T.queue, pad, y + 10, { size: 11.5, font: 'mono', weight: 600, color: C.muted });
        let qx = pad + tw(g, T.queue, 11.5, 'mono', 600) + 12;
        if (!P.queue.length) V.text(g, T.empty, qx, y + 10, { size: 11.5, color: C.muted });
        P.queue.forEach((q) => { if (qx > pad + mainW - 50) return; qx += pill(g, qx, y + 9, `${q.L} ${sf(q.prompt, 1)}`, pal[q.col], { size: 11 }) + 6; });
        y += 24;
        y += wrap(g, T.stats(freeRows(), maxGap(), P.st), pad, y + 10, mainW, { size: 11.5, font: 'mono', color: C.ink });
        if (P.fragAt != null && P.clock - P.fragAt < FRAG_HOLD) y += wrap(g, T.frag[P.fragKind], pad, y + 12, mainW, { size: 12, weight: 600, color: RED(C) }) + 2;
        y += 4;
        // legend
        const legend = (ly0) => {
          let lx = pad, ly = ly0;
          const items = [
            (x0, y0) => V.box(g, x0, y0 - 9, 14, 12, { r: 3, fill: V.alpha(C.a, 0.85) }),
            (x0, y0) => V.box(g, x0, y0 - 7, 14, 8, { r: 2, fill: V.alpha(C.a, 0.3) }),
            (x0, y0) => V.box(g, x0, y0 - 9, 14, 12, { r: 3, fill: V.alpha(C.a, 0.07), stroke: V.alpha(C.a, 0.75), lw: 1.2 }),
            (x0, y0) => { V.box(g, x0, y0 - 9, 14, 12, { r: 3, stroke: V.alpha(C.muted, 0.7), lw: 1 }); hatch(g, x0 + 1, y0 - 8, 12, 10, V.alpha(C.muted, 0.7), 4); },
            (x0, y0) => { g.save(); g.setLineDash([3, 2]); g.strokeStyle = RED(C); g.lineWidth = 1.6; V.rr(g, x0, y0 - 9, 14, 12, 3); g.stroke(); g.restore(); },
          ];
          T.leg.forEach((s, i) => {
            const ww = 20 + tw(g, s, 11) + 14;
            if (lx + ww > pad + mainW && lx > pad) { lx = pad; ly += 18; }
            items[i](lx, ly); V.text(g, s, lx + 19, ly + 1, { size: 11, color: C.muted }); lx += ww;
          });
          return ly + 14;
        };
        let mainBottom;
        if (wide) mainBottom = legend(y + 12);
        // message column and rank 1
        const sx = wide ? w - pad - sideW : pad;
        let sy = wide ? 6 : y + 6;
        if (!wide) { g.strokeStyle = V.alpha(C.muted, 0.25); g.setLineDash([4, 4]); g.beginPath(); g.moveTo(pad, sy); g.lineTo(w - pad, sy); g.stroke(); g.setLineDash([]); sy += 8; }
        V.text(g, T.msgs, sx, sy + 12, { size: 12, weight: 600, color: C.ink, maxW: sideW });
        sy += 22;
        const nLog = wide ? 7 : 4;
        P.log.slice(0, nLog).forEach((m) => {
          const age = P.clock - m.t, fly = P.msgs.includes(m);
          const ls = wrapLines(g, m.ops.join(' · '), sideW - 52, 11, 'mono', 400);
          const lines = ls.slice(0, 2); if (ls.length > 2) lines[1] = lines[1].replace(/.{0,2}$/, '…');
          const bh = 8 + lines.length * 15;
          V.box(g, sx, sy, sideW, bh, { r: 6, fill: V.alpha(TF, age < 0.5 ? 0.22 : 0.07), stroke: V.alpha(TF, fly ? 0.8 : 0.25), lw: 1 });
          V.text(g, '#' + m.n, sx + 6, sy + 15, { size: 10.5, font: 'mono', weight: 700, color: C.muted });
          lines.forEach((ln, i) => V.text(g, ln, sx + 46, sy + 15 + i * 15, { size: 11, font: 'mono', color: C.ink }));
          sy += bh + 4;
        });
        // channel to rank 1
        const cy0 = sy + 4, cy1 = sy + 34, cxm = sx + sideW / 2;
        V.line(g, cxm, cy0, cxm, cy1, { color: V.alpha(TF, 0.5), lw: 2, dash: [4, 4], dashOffset: -t * 20 });
        P.msgs.forEach((m) => { const p = V.clamp((P.clock - m.t) / MSG_T, 0, 1); const yy = V.lerp(cy0, cy1, p); V.glow(g, cxm, yy, 10, TF, 0.7); V.dot(g, cxm, yy, 4, TF); });
        sy = cy1 + 6;
        V.text(g, T.r1, sx, sy + 12, { size: 12, weight: 600, color: C.ink, maxW: sideW });
        sy += 20;
        const mc = 24, mr = N / mc, mw = sideW / mc, mh = 14;
        for (let u = 0; u < N; u++) V.box(g, sx + (u % mc) * mw + 0.5, sy + Math.floor(u / mc) * (mh + 4), mw - 1, mh, { r: 2, fill: V.alpha(C.muted, 0.07) });
        P.r1.forEach((d) => {
          let s = d.base; const e = d.base + d.size;
          while (s < e) {
            const r = Math.floor(s / mc), se = Math.min(e, (r + 1) * mc), px = sx + (s - r * mc) * mw, py = sy + r * (mh + 4), pw = (se - s) * mw;
            if (d.kind === 'kept') { V.box(g, px + 1, py, pw - 2, mh, { r: 3, fill: V.alpha(C.muted, 0.18), stroke: V.alpha(C.muted, 0.6), lw: 1 }); }
            else V.box(g, px + 1, py, pw - 2, mh, { r: 3, fill: V.alpha(pal[d.col], 0.6), stroke: V.alpha(pal[d.col], 0.9), lw: 1 });
            s = se;
          }
        });
        sy += mr * (mh + 4) + 4;
        const ok = !P.msgs.length;
        V.text(g, ok ? T.same : T.applying(P.msgs.length), sx, sy + 10, { size: 11.5, font: 'mono', weight: 600, color: ok ? C.a : C.d });
        sy += 22;
        if (!wide) mainBottom = legend(sy + 8);
        fitH(H, Math.max(mainBottom, sy) + 8);
      });
    }

    /* ======================================================================
       FIG 12 — TensorFold × vLLM: who pays for missing memory
       ====================================================================== */
    function figFolga() {
      const fig = root.querySelector('#fig-folga');
      const ctl = UI.controls(fig);
      const T = {
        load: 'another load on the Spark', when: 'it arrives', before: 'before startup', after: 'after startup', replay: '▶ replay the boot',
        vName: 'vLLM · GLM recipe', vRule: 'pinned pool: --kv-cache-memory-bytes = 11 GiB',
        tName: 'TensorFold · GLM·TF recipe', tRule: 'budget = MemAvailable at startup − 14.5 GiB; pool beyond the window up to 12.5 GiB',
        vStages: ['memory check', 'weights', '11 GiB pin', 'check 1 req.', 'serve'],
        tStages: ['budget', 'estimate', 'fits?', 'weights', 'serve'],
        refit: '✗ refuse → --context N', vRefuse: '✗ startup refused', arrives: '+ load', notUp: 'doesn’t start',
        tankPool: 'KV pool', tankFree: 'free at the worst moment',
        tok: 'tokens', measured: 'measured', model: 'model', win: 'smaller window',
        v: {
          zero: '✓ measured boot: 11 GiB pin, ≥ 6.2 GiB free under load',
          refuse: '✗ the startup check refuses: MemAvailable below util × MemTotal (+ the preflight margin). Between ~6 and ~11 GiB it still passes, with the headroom under load already in the red',
          ok: '✓ the pool doesn’t change; the extra load comes out of the host’s headroom',
          low: '⚠ the headroom shrinks: the pin doesn’t look at free memory (with a 15 GiB pin, 0.8–2.2 GiB were left under load)',
          bad: '✗ no headroom: on unified memory, running out tends to freeze the machine',
        },
        t: {
          zero: '✓ measured boot: 2,922,496 tokens at the cap, 5.6 GiB free at the worst moment',
          slack: '✓ the pool stays at the cap; the slack above the reserve absorbs the load',
          bare: '↘ the pool has shrunk to the window: 1,048,576 tokens for the 4 requests to share, with no extra room for kept prompts',
          shrink: '↘ the pool shrinks and the floor stays: the reserve leaves ~4.5 GiB free at the worst moment',
          refit: '↺ the 1M window doesn’t fit: refused before the weights, it names the largest window that fits and start.sh restarts once with it',
          after: 'the budget was set at startup: the new load comes out of the headroom, as in vLLM',
          afterLow: '⚠ the headroom shrinks: raise MEMORY_RESERVE_GIB when other work shares the Sparks',
          bad: '✗ no headroom: on unified memory, running out tends to freeze the machine',
          afterBad: '✗ no headroom: the pool was sized at startup; reserve more (MEMORY_RESERVE_GIB) when other work shares the Sparks',
        },
      };
      // ES: 32 kept prompt states × ~45 MiB, taken from the cap (pool formula); BPT: bytes per token calibrated on the
      // measured boot (≈6.36 KB); VCHK: earlier load above which vLLM's startup check refuses (chapter 27's model)
      const W1M = 1048576, POOL = 2922496, EXTRA = POOL - W1M, SLACK = 1.1, FLOOR = 4.5, CAP = 12.5, ES = (32 * 45) / 1024, VCHK = 11;
      const BPT = ((CAP - ES) * 2 ** 30) / EXTRA;
      const S = { x: 0, when: 'before', t0: 0, clock: 0, cur: null };
      const restart = () => { S.t0 = S.clock; };
      UI.slider(ctl, { label: T.load, min: 0, max: 16, step: 0.5, value: 0, fmt: (v) => sf(v, 1) + ' GiB', onInput: (v) => { S.x = v; restart(); } });
      UI.seg(ctl, { label: T.when, options: [{ v: 'before', label: T.before }, { v: 'after', label: T.after }], value: 'before', onChange: (v) => { S.when = v; restart(); } });
      UI.button(ctl, T.replay, restart);
      function model() {
        const x = S.x, v = { pool: 1233779, free: 6.2 - x, refuse: S.when === 'before' && x > VCHK }, tf = {};
        if (v.refuse) { v.pool = 0; v.free = 0; }
        if (S.when === 'after') { tf.pe = 12.5; tf.free = 5.6 - x; }
        else { tf.pe = x <= SLACK ? CAP : CAP - (x - SLACK); tf.free = x <= SLACK ? 5.6 - x : FLOOR; }
        tf.refit = tf.pe < 0;
        if (tf.refit) { tf.win = Math.max(2048, Math.floor((W1M - (-tf.pe * 2 ** 30) / BPT) / 2048) * 2048); tf.pool = tf.win; }
        else { tf.win = W1M; tf.pool = Math.floor((W1M + (Math.max(0, tf.pe - ES) * 2 ** 30) / BPT) / 2048 + 1e-9) * 2048; }
        tf.bare = !tf.refit && tf.pe <= ES;
        return { v, tf };
      }
      function tank(g, C, x, y, wT, hT, frac, col, t, ticks, danger) {
        V.box(g, x, y, wT, hT, { r: 10, fill: V.alpha(C.muted, 0.06), stroke: V.alpha(C.muted, 0.5), lw: 1.2 });
        if (danger) V.box(g, x + 1, y + hT - danger * hT, wT - 2, danger * hT - 1, { r: 8, fill: V.alpha(RED(C), 0.14) });
        const f = V.clamp(frac, 0, 1), lvl = y + hT - f * hT;
        if (f > 0.003) {
          g.save(); V.rr(g, x + 1, y + 1, wT - 2, hT - 2, 9); g.clip();
          g.beginPath(); g.moveTo(x, y + hT);
          for (let k = 0; k <= 20; k++) g.lineTo(x + (k / 20) * wT, lvl + Math.sin(t * 2.2 + k * 0.8) * 2);
          g.lineTo(x + wT, y + hT); g.closePath(); g.fillStyle = V.alpha(col, 0.55); g.fill();
          g.restore();
        }
        ticks.forEach(([fr, s]) => {
          const yy = y + hT - fr * hT;
          V.line(g, x + wT - 8, yy, x + wT, yy, { color: V.alpha(C.ink, 0.5), lw: 1 });
          V.text(g, s, x + wT + 5, yy + 0.5, { size: 10.5, font: 'mono', base: 'middle', color: C.muted });
        });
      }
      function strip(g, C, x, y, wS, names, seq, col, p, extra) {
        const n = names.length, step = wS / n;
        names.forEach((s, i) => {
          const cx = x + step * i + step / 2;
          if (i < n - 1) V.line(g, cx + 8, y, cx + step - 8, y, { color: V.alpha(C.muted, 0.4), lw: 1.5 });
          V.dot(g, cx, y, 6, V.alpha(col, 0.35));
          V.text(g, s, cx, y + 17, { size: 10.5, align: 'center', color: C.muted, maxW: step - 4 });
        });
        const pos = (s) => x + step * s + step / 2;
        const L = seq.length, q = V.clamp(p, 0, L - 1), i0 = Math.max(0, Math.min(L - 2, Math.floor(q)));
        const a = seq[i0], b = L > 1 ? seq[i0 + 1] : a, fr = L > 1 ? V.ease.inOut(V.clamp(q - i0, 0, 1)) : 0;
        const fx = V.lerp(pos(a), pos(b), fr), fy = y - (b < a ? Math.sin(fr * Math.PI) * 14 : 0);
        if (extra) extra(pos);
        V.glow(g, fx, fy, 16, col, 0.8); V.dot(g, fx, fy, 5.5, col);
      }
      const H = V.fig(fig, (g, w, h, t, dt) => {
        const C = V.C(), TF = TFC(C), m = model();
        if (!LA.paused) S.clock += Math.min(dt, 0.1);
        const k = Math.min(1, dt * 5);
        if (!S.cur) S.cur = { vp: m.v.pool, vf: m.v.free, tp: m.tf.pool, tf: m.tf.free };
        S.cur.vp = V.lerp(S.cur.vp, m.v.pool, k); S.cur.vf = V.lerp(S.cur.vf, m.v.free, k);
        S.cur.tp = V.lerp(S.cur.tp, m.tf.pool, k); S.cur.tf = V.lerp(S.cur.tf, m.tf.free, k);
        const pad = 14, wide = w >= 700, gap = 20;
        const pw = wide ? (w - 2 * pad - gap) / 2 : w - 2 * pad;
        const el = (S.clock - S.t0) / 0.75;
        const lh = Math.round(11.5 * 1.4);
        const ruleH = wide ? Math.max(wrapLines(g, T.vRule, pw, 11.5, 'mono').length, wrapLines(g, T.tRule, pw, 11.5, 'mono').length) * lh : 0;
        const panel = (x0, y0, side) => {
          const isT = side === 't', col = isT ? TF : C.glm;
          let y = y0;
          V.box(g, x0, y, pw, 4, { r: 2, fill: V.alpha(col, 0.9) });
          y += 8;
          V.text(g, isT ? T.tName : T.vName, x0, y + 14, { size: 14, font: 'display', weight: 700, color: col, maxW: pw });
          y += 22;
          const rh = wrap(g, isT ? T.tRule : T.vRule, x0, y + 10, pw, { size: 11.5, font: 'mono', color: C.muted, lh });
          y += ruleH || rh;
          y += 8;
          const names = isT ? T.tStages : T.vStages;
          const seq = isT ? (m.tf.refit ? [0, 1, 2, 0, 1, 2, 3, 4] : [0, 1, 2, 3, 4]) : m.v.refuse ? [0] : [0, 1, 2, 3, 4];
          const sy = y + 22;
          strip(g, C, x0, sy, pw, names, seq, col, el, (pos) => {
            if (isT && m.tf.refit) { const xx = V.clamp(pos(2), x0 + 70, x0 + pw - 70); V.text(g, T.refit, xx, sy - 15, { size: 10.5, font: 'mono', weight: 700, align: 'center', color: RED(C) }); }
            if (!isT && m.v.refuse) { const xx = V.clamp(pos(0), x0 + 70, x0 + pw - 70); V.text(g, T.vRefuse, xx, sy - 15, { size: 10.5, font: 'mono', weight: 700, align: 'center', color: RED(C) }); }
            if (S.when === 'after' && S.x > 0 && el > seq.length - 1 && seq.length > 1) { const xx = Math.min(pos(names.length - 1), x0 + pw - 34); pill(g, xx, sy - 16, T.arrives, RED(C), { size: 10, align: 'center' }); }
          });
          y = sy + 34;
          const tH = wide ? 150 : 120, tW = Math.min(78, pw * 0.2);
          const t1x = x0 + 4, t2x = x0 + pw / 2 + 4;
          V.text(g, T.tankPool, t1x, y + 10, { size: 11.5, weight: 600, color: C.ink, maxW: pw / 2 - 8 });
          V.text(g, T.tankFree, t2x, y + 10, { size: 11.5, weight: 600, color: C.ink, maxW: pw / 2 - 8 });
          y += 20;
          const pv = isT ? S.cur.tp : S.cur.vp, fv = isT ? S.cur.tf : S.cur.vf;
          tank(g, C, t1x, y, tW, tH, pv / 3.2e6, col, t, [[W1M / 3.2e6, '1M'], [2e6 / 3.2e6, '2M'], [3e6 / 3.2e6, '3M']]);
          tank(g, C, t2x, y, tW, tH, fv / 8, fv < 1 ? RED(C) : C.e, t + 1, [[2 / 8, '2'], [4 / 8, '4'], [6 / 8, '6 GiB']], 1 / 8);
          const down = !isT && m.v.refuse;
          if (fv <= 0 && !down) { V.text(g, '✗', t2x + tW / 2, y + tH / 2, { size: 30, weight: 700, align: 'center', base: 'middle', color: RED(C) }); }
          y += tH + 16;
          const zero = S.x === 0;
          const pTxt = down ? '—' : isT ? (m.tf.refit ? `≈ ${nf(m.tf.pool)} ${T.tok}` : `${nf(m.tf.pool)} ${T.tok}`) : `${nf(m.v.pool)} ${T.tok}`;
          const pTag = down ? T.notUp : isT ? (m.tf.refit ? T.win : zero || S.when === 'after' || S.x <= SLACK ? T.measured : T.model) : T.measured;
          const fval = isT ? m.tf.free : m.v.free;
          const fTxt = down ? '—' : zero ? (isT ? '5.6 GiB' : '≥ 6.2 GiB') : `${fval < 0 ? '−' : ''}${sf(Math.abs(fval), 1)} GiB`;
          const fTag = down ? T.notUp : zero ? T.measured : T.model;
          V.text(g, pTxt, t1x, y + 4, { size: 12.5, font: 'mono', weight: 700, color: down ? C.muted : C.ink, maxW: pw / 2 - 8 });
          V.text(g, pTag, t1x, y + 20, { size: 10.5, color: C.muted });
          V.text(g, fTxt, t2x, y + 4, { size: 12.5, font: 'mono', weight: 700, color: down ? C.muted : fval < 1 ? RED(C) : C.ink, maxW: pw / 2 - 8 });
          V.text(g, fTag, t2x, y + 20, { size: 10.5, color: C.muted });
          y += 34;
          let st, sc;
          if (!isT) { const f = m.v.free; st = zero ? T.v.zero : down ? T.v.refuse : f >= 2 ? T.v.ok : f > 0 ? T.v.low : T.v.bad; sc = zero || f >= 2 ? C.a : f > 0 && !down ? C.d : RED(C); }
          else if (S.when === 'after') { const f = m.tf.free; st = zero ? T.t.zero : f >= 2 ? T.t.after : f > 0 ? T.t.afterLow : T.t.afterBad; sc = zero || f >= 2 ? C.a : f > 0 ? C.d : RED(C); }
          else { st = zero ? T.t.zero : m.tf.refit ? T.t.refit : S.x <= SLACK ? T.t.slack : m.tf.bare ? T.t.bare : T.t.shrink; sc = m.tf.refit || m.tf.bare ? C.d : C.a; }
          y += wrap(g, st, x0, y + 8, pw, { size: 12, weight: 600, color: sc });
          return y;
        };
        let yA, yB;
        if (wide) { yA = panel(pad, 6, 'v'); yB = panel(pad + pw + gap, 6, 't'); fitH(H, Math.max(yA, yB) + 14); }
        else { yA = panel(pad, 6, 'v'); yB = panel(pad, yA + 22, 't'); fitH(H, yB + 14); }
      });
    }

    /* ======================================================================
       FIG 10 — subway map of KV sizing
       ====================================================================== */
    function figMetro() {
      const fig = root.querySelector('#fig-metro');
      const ctl = UI.controls(fig);
      const ST = [
        { t: 'Bytes per token and groups', s: 'bytes/token', d: 'List each layer type and how much it stores: per token (attention, window) or per request (state).' },
        { t: 'Page and block', s: 'page', d: 'Compute the common page and the block size the allocator will pick; check the kernels’ alignment.' },
        { t: 'Workspaces and graphs', s: 'workspaces', d: 'Add up what the profile run deducts before the KV: activation peak, workspaces and graph estimate.' },
        { t: 'Pool: util or bytes', s: 'pool', d: 'Decide whether the pool comes from gpu_memory_utilization or from a byte pin.' },
        { t: 'Read the boot', s: 'boot', d: 'Available KV cache memory, GPU KV cache size, Maximum concurrency and, in a hybrid, the per-group breakdown.' },
        { t: 'Ceilings', s: 'ceilings', d: 'Choose max_model_len and max_num_seqs for the workload: the sum of live tokens has to fit in the pool.' },
        { t: 'Measure again', s: 'measure', d: 'Restart and reread the logs after every change; under load, watch kv_cache_usage, preemptions and MemAvailable.' },
      ];
      const TX = {
        glm: [
          '11 MLA layers at 656 B/token (+ indexer, 132 B per pool of 4); 34 KDA with a 2,342,912 B state per layer and rank; DFlash2 drafter with 5 SWA layers at 2,048 B/token.',
          'KDA state ÷ 656 = 3,571.5 → multiple of 256 = 3,584; mla_page 2,351,104 B; one ID costs 11 MLA pages + 11 indexer pages ≈ 25.9 MiB; 64-token kernel; KpoolTail of 4; drafter in slot-share (64, or 896 with compact pages); LCM = 3,584.',
          'Rightsize indexer (5,036 MiB → 126 MiB at 1M); 560 MiB E3 scratch; 2.43 GiB graph estimate with CG_ESTIMATE=1; InstantTensor takes ~4.4–5.6 GiB away from the KV.',
          'Pin --kv-cache-memory-bytes 11811160064 (11 GiB) with compact pages; util 0.85. Without the pin, 850k doesn’t boot with InstantTensor (12.4 < 13.56 GiB).',
          '1,233,779 tokens = 1.45× at 850k is accounting. The [glm53-kv-capacity-log] shows usable IDs, IDs per segment and how much conversation really fits.',
          'MAX_MODEL_LEN 850000 (TP3: 1M) and MAX_NUM_SEQS 4. Don’t cut the context to free up slots: the ID demand of Mamba and the drafter barely depends on length.',
          'GLM53_KV_CAPACITY_LOG=1; VLLM_DEBUG_WORKSPACE=1 via GLM53_EXTRA_ENV to measure workspaces; always compare with the same context, util and pool.',
        ],
        qwen: [
          '12 QSA + 1 MTP with KV: 1 head per GPU × 256 × (K, V) = 1,024 B/token in BF16, 512 in FP8; 36 GDN with a state of 1.5 MiB per layer and GPU in FP32 (0.75 MiB with MAMBA_SSM_CACHE_DTYPE=bfloat16). The GDN state and the QSA indexer (BF16) don’t shrink with the FP8 KV, which is why the FP8 pool grows 1.70× and not 2× (README :376-381).',
          'GDN page ÷ attention bytes, aligned to 16: 1,664 in the default (FP8 KV + BF16 state), 3,200 (FP8 + FP32 state), 1,600 (BF16 KV + FP32 state), 832 (BF16 + BF16, math). The prefix cache only hits on those boundaries.',
          'At 0.835: 64.46 GiB of weights (68.52 with what sits outside PyTorch), a 1.07 GiB activation peak, 0.54 GiB of graphs. A 1M max_model_len also inflates the QSA scorer’s per-step buffers.',
          'GMU 0.80 (0.835 left <1 GiB free per node). The log suggests --kv-cache-memory=33651431445 or 41360080896; the 1M + concurrency profile uses 45502283776.',
          'Measured at 0.835 with FP8: 32.02 GiB → 3,652,200 tokens → 13.93× at 262,144 (BF16: 2,131,159 tokens, 8.13×).',
          'MAX_MODEL_LEN 262144 and MAX_NUM_SEQS 8: the scheduler limits before the cache does. For 1M, use the README profiles (2 or 3 sequences).',
          'Reread Available KV cache, GPU KV cache size and Free memory on device; in /metrics, kv_cache_usage and preemptions.',
        ],
        tf: [
          '11 DSA layers: a 528 B FP8 latent (512 codes + power-of-two scale + alignment) and the indexer’s pooled keys at 144 B per pool of 4 tokens, ≈564 B per layer and ~6.2 KB per token and rank (the research’s math). The latent is replicated on both ranks. The 34 KDA layers keep a fixed state per stream.',
          'No uniform page: contiguous extents in multiples of ALIGN = 2,048 tokens (pooled keys at base/4, with 2 rows of headroom). The KDA state [slots, 2, layers, H, 128, 128] lives in 4 slots outside the pool; the drafter, in a ring of 2,176 rows (21 MiB per rank).',
          'No profile run: memory comes from formulas of the geometry (an 88.09 GiB startup estimate on rank 0). Overcounted scratch is lost KV: patch 0028 removed partials the prompt kernels don’t allocate. At the peak (a 1M prompt), the server goes ~10 GiB past the estimate.',
          'Neither util nor a pin: MemAvailable at startup − MEMORY_RESERVE_GIB (14.5), and the pool beyond the window grows into what is left, up to KV_POOL_GIB = 12.5 GiB per Spark. If the window doesn’t fit, it is refused before the weights and start.sh restarts with the largest that fits.',
          '2,922,496 tokens at the measured start (1,427 extents of 2,048; ~2.1–2.9M depending on free memory). Under load, /health shows pool_tokens, pool_free_tokens and the streams decoding, filling and paused.',
          'PARALLEL=4 and CONTEXT=1,048,576: each request grows up to the window and the four share the pool (1M + 1M, or 1M + 3 × ~640k). Whoever doesn’t fit waits in order; kept prompts (up to 32) give up their room first.',
          'Reread /health and both Sparks’ MemAvailable: idle, 7.0 and 10.5 GiB; at the worst moment of a 1M prompt, 5.6 and 9.5 GiB. Other work sharing the Sparks? Raise MEMORY_RESERVE_GIB.',
        ],
        ds: [
          '584 B row (448 FP8 + 128 of RoPE in BF16 + 8 of scales); 21 C4A (÷4), 20 C128A (÷128), indexer 132 B per pool of 4; a 128-row window in 46 layers ≈ 3.4 MB per sequence.',
          'The SWA page (64 × 584 = 37,376 B) is fixed; the C4 group needs 64 rows → --block-size 256; drafter SWA with blocks of 4 and 8; LCM = 256.',
          'VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0; VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=256 (stock 512) caps the indexer’s logits tensor; in the Stage-C overlay (lane B), the split-K scratch is reserved in the dummy run before lock_workspace; #50298 (default lane) puts the top-k + window indices in the workspace.',
          'GPU_MEMORY_UTILIZATION_TEXT=0.835 (0.87 in the long-coding profile): +0.01 ≈ +1.2 GiB ≈ +165 thousand tokens. No byte pin.',
          '17.04 GiB → 2,331,430 tokens → 2.22× at 1,048,576: about 2 whole 1M requests fit (accounting of ~7.8 KB/token).',
          'MAX_MODEL_LEN 1048576 (YaRN ×16) with 6 slots (TP3: 16); a 200000 alternative for high concurrency. What matters is the sum of live tokens.',
          'In the head node’s log: ~17 GiB, ~2.3M tokens, ~2.2×; watch MemAvailable on both nodes and keep headroom.',
        ],
      };
      const S = { r: 'glm', pos: 0 };
      UI.seg(ctl, { label: 'Recipe', options: [{ v: 'glm', label: 'GLM' }, { v: 'tf', label: 'GLM·TF' }, { v: 'qwen', label: 'Qwen' }, { v: 'ds', label: 'DeepSeek' }], value: 'glm', onChange: (v) => { S.r = v; } });
      const st = UI.stepper(fig, ST.map((x) => ({ t: x.t, d: x.d })), {});
      const H = V.fig(fig, (g, w, h, t, dt) => {
        const C = V.C(), pad = 18, W = w - 2 * pad;
        const lineCol = S.r === 'glm' ? C.glm : S.r === 'tf' ? TFC(C) : S.r === 'qwen' ? C.qwen : C.ds;
        const n = ST.length, y0 = 44;
        const pts = ST.map((_, i) => [pad + 14 + (i * (W - 28)) / (n - 1), y0 + (i % 2 ? 16 : -10)]);
        g.lineCap = 'round'; g.lineJoin = 'round';
        g.strokeStyle = V.alpha(lineCol, 0.85); g.lineWidth = 7;
        g.beginPath(); g.moveTo(pts[0][0], pts[0][1]);
        for (let i = 1; i < n; i++) { const mx = (pts[i - 1][0] + pts[i][0]) / 2; g.bezierCurveTo(mx, pts[i - 1][1], mx, pts[i][1], pts[i][0], pts[i][1]); }
        g.stroke(); g.lineCap = 'butt';
        if (H.pointer.clicked) {
          pts.forEach((p, i) => { if (Math.hypot(H.pointer.x - p[0], H.pointer.y - p[1]) < 18) st.go(i); });
        }
        S.pos = V.lerp(S.pos, st.i, Math.min(1, dt * 4));
        {
          const i0 = Math.floor(S.pos), fr = S.pos - i0, a = pts[Math.min(n - 1, i0)], b = pts[Math.min(n - 1, i0 + 1)];
          const trx = V.lerp(a[0], b[0], fr), tr = V.lerp(a[1], b[1], fr);
          V.box(g, trx - 13, tr - 7, 26, 14, { r: 7, fill: C.ink });
          V.box(g, trx - 9, tr - 3.5, 7, 7, { r: 2, fill: lineCol }); V.box(g, trx + 2, tr - 3.5, 7, 7, { r: 2, fill: lineCol });
        }
        pts.forEach((p, i) => {
          const on = st.i === i;
          if (on) V.glow(g, p[0], p[1], 26, lineCol, 0.8);
          V.dot(g, p[0], p[1], on ? 13 : 10, C.bg || '#000');
          g.strokeStyle = lineCol; g.lineWidth = 3; g.beginPath(); g.arc(p[0], p[1], on ? 13 : 10, 0, Math.PI * 2); g.stroke();
          V.text(g, String(i + 1), p[0], p[1] + 0.5, { size: 11, font: 'mono', weight: 700, align: 'center', base: 'middle', color: C.ink });
          if (w >= 600) V.text(g, ST[i].s, i === 0 ? p[0] - 13 : i === n - 1 ? p[0] + 13 : p[0], p[1] + (i % 2 ? 30 : -20), { size: 10.5, align: i === 0 ? 'left' : i === n - 1 ? 'right' : 'center', color: on ? C.ink : C.muted, weight: on ? 600 : 400 });
        });
        let y = y0 + 58;
        const cardH0 = y;
        const title = `Station ${st.i + 1} · ${ST[st.i].t}`;
        const txt = TX[S.r][st.i];
        const innerW = W - 28;
        const ls = wrapLines(g, txt, innerW, 13.5);
        const ch = 44 + ls.length * 20 + 12;
        V.box(g, pad, y, W, ch, { r: 12, fill: V.alpha(lineCol, 0.1), stroke: V.alpha(lineCol, 0.7), lw: 1.4 });
        V.text(g, title, pad + 14, y + 24, { size: 14, font: 'display', weight: 700, color: C.ink, maxW: innerW });
        ls.forEach((l, k) => V.text(g, l, pad + 14, y + 50 + k * 20, { size: 13.5, color: C.ink }));
        y += ch;
        fitH(H, Math.max(cardH0 + 120, y + 14));
      });
    }

    [figFrag, figPaged, figSlot, figBloco, figPagina, figPredio, figOrcamento, figBoot, figTeto, figExtents, figFolga, figMetro].forEach((f) => {
      try { f(); } catch (e) { console.error('[paged-attention-kv] ' + f.name, e); }
    });
  },
});

LA.chapter({
  id: 'mapa-da-jornada',
  part: 0,
  order: 0,
  title: 'Journey map',
  navTitle: 'Journey map',
  subtitle: 'What an inference “recipe” is, the three recipes that serve as the lab (with their real numbers), the first calculation every recipe does, the recipe-from-scratch tutorial’s model and how to use this platform — including with a copilot running on your Mac.',
  minutes: 35,
  recipes: ['glm', 'tf', 'qwen', 'ds'],
  tags: ['recipe', 'overview', 'memory', 'bandwidth', 'hybrid models', 'MiMo', 'TensorFold', 'copilot', 'MLX'],
  goals: [
    'Tell apart <b>weights</b>, <b>architecture</b>, <b>engine</b> and <b>hardware</b> — and the three questions every recipe answers',
    'Know the decision layers of a recipe and where each one is studied along the trail',
    'Do the back-of-the-envelope calculation for memory and speed — and understand why it is only a ceiling',
    'Recognize the three recipes (and MiMo-V2.6-Flash) by their numbers: parameters, layers, quantization, KV, TP, speculation, context and measured speeds — and see, in the TensorFold version of GLM, that the engine is a recipe decision too',
    'Use the platform: trail, search, glossary, figures and the copilot — including locally, with MLX',
  ],
  prereqs: [],
  summary: [
    'A recipe is the documented set of decisions that makes weights + architecture + engine fit on the hardware. Every recipe answers three questions: <b>does it fit?</b>, <b>is it fast?</b>, <b>is it still right?</b>',
    'Weight memory ≈ \\(N_{\\text{total}} \\times b_{\\text{total}}/8\\); decode ≲ \\(\\tau \\times\\) bandwidth ÷ \\(B_{\\text{step}}\\), with \\(B_{\\text{step}} \\approx N_{\\text{active}} \\times b_{\\text{active}}/8\\). In a mixed-precision MoE, \\(b_{\\text{active}}\\) is much larger than the checkpoint average (GLM: ~6.9 vs. 4.37 bits), because the per-token read is dominated by the dense parts. And it is a <b>ceiling</b>: network collectives, small kernels and the effective bandwidth keep the measured number below it (Qwen: a ~47 tok/s ceiling vs. 24.5 tok/s measured without speculation). On the memory side, the GMU keeps the system’s headroom: for GLM, with GMU 0.85, ~13 GiB per node is left for the KV.',
    'The three recipes: <b>GLM</b> 321 B / ~18 B active, EXL3 4 bpw, KV <code>fp8_ds_mla</code>, TP=2, DFlash2 k=7, 850k; <b>Qwen</b> 125 B (+51 B of N-gram) / ~6 B, NVFP4, KV fp8, TP=2+EP, MTP k=3, 262k; <b>DeepSeek</b> 284 B / 13 B, FP4+FP8, <code>nvfp4_ds_mla</code>, TP=2, DSpark k=6, 1M. GLM also runs on <b>TensorFold</b>: same checkpoint, another engine (bit-for-bit exact, 528 B FP8 KV, a 1M window per request, up to 4 at once in v1.3.2 and up to 8 since v1.5), 60.4 tok/s on prose vs. 36.1 — measured under different conditions.',
    'All three models are <b>hybrids</b>: only a fraction of the layers pays KV per token (11 of 45 in GLM, 12 of 48 in Qwen, compressed versions in DeepSeek) — that is what makes 850k–1M tokens fit in pools of 11–17 GiB.',
    '<b>MiMo-V2.6-Flash</b> (309 B / 15 B, SWA:global 39:9, K/V of 192/128, MXFP4 experts, MTP + DFlash) is the final exam: it barely fits and breaks several of the engine’s assumptions.',
    'Numbers only count with a <b>receipt</b> (date, lane, prompt type, protocol): in the same 2026-08-30 test, GLM measures 65.1 tok/s on structured text and 27.1 on prose, because the drafter lands 6.71 vs. 2.39 tokens per step.',
    'The copilot speaks the OpenAI Chat Completions contract — the same one as the recipes — and runs 100% locally on the Mac with <code>./tools/start-mlx-copilot.sh</code> (Base URL <code>http://127.0.0.1:8080/v1</code>, empty model).',
  ],
  quiz: [
    { q: 'A 300 B-parameter model in 4 bits takes up roughly how much in weights alone?', o: ['~37.5 GB', '~150 GB', '~600 GB', '~1.2 TB'], a: 1, e: '300×10⁹ × 4/8 bytes = 150×10⁹ bytes ≈ 150 GB (≈ 140 GiB). It doesn’t fit in the 121.7 GiB that CUDA sees on one node — that’s why the recipes use two. 37.5 GB would be 1 bit per weight; 600 GB, 16 bits.' },
    { q: 'Why does MoE help decode <i>speed</i>?', o: ['Because it has fewer total parameters', 'Because each token only reads the weights of the activated experts', 'Because it does away with the KV cache', 'Because it uses fewer bits'], a: 1, e: 'In decode the bottleneck is reading weights from memory. With MoE, each token uses only a fraction of the parameters (13 B of 284 B in DeepSeek), so it reads far fewer bytes (in a mixed-precision MoE, a good share of those bytes comes from the dense parts in FP8/BF16, not from the experts). The total is still large (it decides whether it fits), the KV still exists and the bits are a separate choice (quantization).' },
    { q: 'Which question does measuring KLD/perplexity answer?', o: ['Does it fit?', 'Is it fast?', 'Is it still right?', 'None of them'], a: 2, e: 'Quality metrics measure whether the shortcuts (quantization, kernels, speculation) preserved the original model’s behavior — e.g., GLM at 4 bpw has a KLD of 0.024555 nats against BF16, practically the same as the official FP8 with 54% of the bytes.' },
    { q: 'Which speculation setup does each recipe use by default?', o: ['GLM: MTP k=3 · Qwen: DFlash2 k=7 · DeepSeek: DSpark k=6', 'GLM: DFlash2 k=7 · Qwen: MTP k=3 · DeepSeek: DSpark k=6', 'All three use MTP k=2', 'None of them uses speculation by default'], a: 1, e: 'GLM: DFlash2 (block drafting) with k=7; Qwen: its own MTP layer with k=3 and a reduced draft vocabulary; DeepSeek: DSpark with k=6 (and not 5, because of a divisibility rule in the engine). MTP k=2 is only GLM’s rollback.' },
    { q: 'Why does the KV cache of MiMo-V2.6-Flash grow slowly with the context?', o: ['Because the model has no attention layers', 'Because 39 of the 48 layers use a 128-token sliding window (fixed KV per conversation) and only 9 global layers grow with the context', 'Because the experts are in MXFP4', 'Because DFlash replaces the KV cache'], a: 1, e: 'Only the 9 global layers store KV per token (5,760 B/token per GPU in FP8); the 39 window layers cost a fixed ~10–13 MB per conversation. MXFP4 shrinks the weights, not the KV; DFlash is a drafter and has its own KV.' },
    { q: 'You start the copilot with MLX on the Mac. What does it send, and where?', o: ['A POST /v1/chat/completions in the OpenAI contract to http://127.0.0.1:8080/v1 — the same contract the recipes’ vLLM exposes', 'Direct gRPC calls to the Mac’s GPU', 'The page text to a cloud server, always', 'The model weights to the browser'], a: 0, e: 'The copilot only knows the OpenAI Chat Completions contract (with SSE streaming). MLX, Ollama, LM Studio, OpenAI and your own vLLM recipe become just a different base URL — and with the MLX preset nothing leaves your machine.' },
  ],
  mount(root, LA) {
    const V = LA.viz, UI = LA.ui;
    const B = (r) => LA.recipeBadge(r);
    const GiB = 1073741824;
    const nf = (x, d = 0) => (+x).toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d });
    const tfCache = { k: null, v: null };
    const TFC = (C) => { if (tfCache.k !== C) { tfCache.k = C; tfCache.v = getComputedStyle(document.documentElement).getPropertyValue('--tf').trim() || C.d; } return tfCache.v; }; // GLM·TF badge color
    const recCol = (C, k) => (k === 'glm' ? C.glm : k === 'qwen' ? C.qwen : k === 'ds' ? C.ds : k === 'tf' ? TFC(C) : C.d); // MiMo (community) in amber
    const fontC = {};
    const F = (size, font, weight) => { const k = size + (font || 'sans') + (weight || 400); return fontC[k] || (fontC[k] = V.font(size, font || 'sans', weight || 400)); };
    const tw = (g, s, size, weight, font) => { g.font = F(size, font, weight); return g.measureText(s).width; };
    const halo = (g, s, x, y, o = {}) => {
      const C = V.C();
      g.font = F(o.size || 11, o.font || 'sans', o.weight || 500);
      g.textAlign = o.align || 'left'; g.textBaseline = o.base || 'alphabetic';
      g.lineJoin = 'round'; g.lineWidth = 3.5; g.strokeStyle = C.bg; g.strokeText(s, x, y);
      g.fillStyle = o.color || C.ink; g.fillText(s, x, y);
    };
    const tip = (g, lines, px, py, w, h) => {
      const C = V.C();
      const W = Math.min(w - 16, Math.max(...lines.map((l, i) => tw(g, l, i ? 11.5 : 12.5, i ? 400 : 600))) + 20);
      const Hh = 12 + lines.length * 17;
      let x = px + 14, y = py + 14;
      if (x + W > w - 6) x = Math.max(6, px - W - 14);
      if (y + Hh > h - 6) y = Math.max(6, py - Hh - 10);
      V.box(g, x, y, W, Hh, { r: 9, fill: V.alpha(C.panel, 0.96), stroke: V.alpha(C.muted, 0.5), lw: 1 });
      lines.forEach((l, i) => V.text(g, l, x + 10, y + 20 + i * 17, { size: i ? 11.5 : 12.5, weight: i ? 400 : 600, color: i ? C.muted : C.ink, maxW: W - 18 }));
    };
    const wrapLines = (g, str, maxW, o = {}) => {
      const words = String(str).split(' '), out = []; let line = '';
      words.forEach((wd) => { const test = line ? line + ' ' + wd : wd; if (line && tw(g, test, o.size || 12, o.weight || 400, o.font) > maxW) { out.push(line); line = wd; } else line = test; });
      if (line) out.push(line);
      return out;
    };
    const wrapText = (g, str, x, y, maxW, lh, o) => { const L = wrapLines(g, str, maxW, o); L.forEach((l, i) => V.text(g, l, x, y + i * lh, o)); return y + Math.max(0, L.length - 1) * lh; };
    const wrapCtl = (o) => { const el = o && o.el ? o.el : o; if (el && el.style) { el.style.flexWrap = 'wrap'; el.style.maxWidth = '100%'; el.style.minWidth = '0'; } const s = el && el.querySelector && el.querySelector('select'); if (s) { s.style.maxWidth = '100%'; s.style.minWidth = '0'; } return o; };
    const tokLbl = (n) => (n >= 1048576 ? nf(n / 1048576, 1) + 'M' : n >= 1024 ? nf(n / 1024, n < 10240 ? 1 : 0) + 'k' : nf(n, 0));
    const ctxLbl = (n) => (n >= 1048576 ? nf(n / 1048576, n % 1048576 ? 2 : 0) + 'M' : nf(n / 1024, 0) + 'k');

    /* ================================================================
       Figure 1 — four worlds (stepper + canvas)
       ================================================================ */
    const f1 = root.querySelector('#fig-mundos');
    const steps1 = [
      { t: 'Weights — the score', d: 'A <b>checkpoint</b> is a collection of tensors stored in <code>.safetensors</code> files. GLM-5.3-Flash has <b>321,323,031,390</b> of these numbers: in BF16 (2 bytes each) that would be ~642.6 GB; the recipe uses a 175.6 GB version in 120 files. Each little square is one weight (illustrative values) — on its own, it means nothing.' },
      { t: 'Architecture — how the numbers combine', d: 'The architecture says that those numbers form <b>matrices</b> and in what order to multiply them. Here the vector \\(x\\) (the representation of a token) is multiplied by the matrix \\(W\\): each output \\(y_i = \\sum_j W_{ij}\\,x_j\\) is the dot product of row \\(i\\) with \\(x\\). A GLM token goes through 45 layers, each with several matrices like this.' },
      { t: 'Engine — the conductor', d: '<b>vLLM</b> receives requests from many users, groups them into the same step (<span class="la-term" data-term="continuous-batching"><i>continuous batching</i></span>), decides who gets in and who waits, and keeps each conversation’s <b>KV cache</b> in memory blocks. At every step, each conversation in the batch gains one new token.' },
      { t: 'Hardware — the orchestra', d: 'Two nodes, each with a nominal 128 GB of unified memory (CUDA sees 121.7 GiB) and ~273 GB/s of theoretical bandwidth, linked by RDMA at 200 Gb/s (~25 GB/s). In GLM, each GPU holds 82.94 GiB of weights (79.65 without the KDA BF16 copy, on by default since 2026-09-26) and an 11 GiB KV pool; the remaining ~27.8 GiB go to activations, CUDA graphs and the operating system. Everything has to fit here — and be read fast enough.' },
    ];
    const st1 = UI.stepper(f1, steps1, { autoplay: 7500 });
    const r1 = V.rng(20260928);
    const WGRID = Array.from({ length: 48 }, () => Array.from({ length: 30 }, () => r1.normal() * 0.035));
    const round1 = (A) => A.map((r) => r.map((v) => Math.round(v * 10) / 10));
    const MAT = {
      big: { W: round1(V.randMatrix(5, 8, 7, 0.9)), x: round1([V.randMatrix(1, 8, 3, 0.9)[0]])[0] },
      small: { W: round1(V.randMatrix(4, 5, 9, 0.9)), x: round1([V.randMatrix(1, 5, 4, 0.9)[0]])[0] },
    };
    Object.values(MAT).forEach((m) => { m.y = m.W.map((row) => row.reduce((s, v, j) => s + v * m.x[j], 0)); });
    const f1d = (v) => (v < 0 ? '−' : '') + nf(Math.abs(v), 1);
    const subN = (i) => String(i).split('').map((c) => '₀₁₂₃₄₅₆₇₈₉'[+c]).join('');

    function drawWeights(g, w, h, t, C) {
      const pad = 14, top = 30, bot = 66;
      V.text(g, w < 480 ? 'a slice of a tensor (illustrative)' : 'a slice of a weight tensor (illustrative values)', pad, 19, { size: 11.5, font: 'mono', color: C.muted, maxW: w - 2 * pad });
      const cw = w >= 640 ? 46 : 32, chh = 20, gap = 3;
      const cols = Math.max(4, Math.floor((w - 2 * pad + gap) / (cw + gap)));
      const rows = Math.ceil((h - top - bot) / (chh + gap)) + 1;
      const off = t * 1.3, base = Math.floor(off), frac = off - base;
      const x0 = pad + (w - 2 * pad - (cols * (cw + gap) - gap)) / 2;
      g.save(); g.beginPath(); g.rect(0, top, w, h - top - bot); g.clip();
      for (let r = 0; r < rows; r++) {
        const row = WGRID[(r + base) % WGRID.length];
        const y = top + (r - frac) * (chh + gap);
        for (let c = 0; c < cols; c++) {
          const v = row[c % row.length], tt = V.clamp(v / 0.09, -1, 1);
          V.rr(g, x0 + c * (cw + gap), y, cw, chh, 4); g.fillStyle = V.cmap(tt, 'div'); g.fill();
          if (cw >= 40) V.text(g, (v < 0 ? '−' : '') + Math.abs(v).toFixed(3), x0 + c * (cw + gap) + cw / 2, y + chh / 2 + 0.5, { size: 10, font: 'mono', align: 'center', base: 'middle', color: Math.abs(tt) > 0.55 ? C.bg : C.ink });
        }
      }
      g.restore();
      const gr = g.createLinearGradient(0, top, 0, top + 26); gr.addColorStop(0, C.bg); gr.addColorStop(1, V.alpha(C.bg, 0));
      g.fillStyle = gr; g.fillRect(0, top, w, 26);
      const gr2 = g.createLinearGradient(0, h - bot - 26, 0, h - bot); gr2.addColorStop(0, V.alpha(C.bg, 0)); gr2.addColorStop(1, C.bg);
      g.fillStyle = gr2; g.fillRect(0, h - bot - 26, w, 26);
      V.text(g, '321,323,031,390 weights', w / 2, h - 36, { size: w < 420 ? 18 : 21, font: 'display', weight: 700, align: 'center', color: C.a });
      V.text(g, 'BF16 ≈ 642.6 GB · the recipe’s EXL3 checkpoint (TR3, snapshot): 175.6 GB in 120 shards', w / 2, h - 14, { size: 11.5, font: 'mono', align: 'center', color: C.muted, maxW: w - 16 });
    }

    function drawMatVec(g, w, h, t, C) {
      const small = w < 560, M = small ? MAT.small : MAT.big;
      const Wm = M.W, xv = M.x, yv = M.y, R = Wm.length, K = Wm[0].length, gap = 2;
      const cs = Math.max(22, Math.min(40, Math.floor((w - 110) / (K + 1)) - gap, Math.floor((h - 150) / (R + 1)) - gap));
      const Ww = K * (cs + gap) - gap;
      const total = Ww + 30 + cs;
      const x0 = Math.max(30, (w - total) / 2 + 10), yx = 44, yW = yx + cs + 16;
      const per = 1.7, cyc = t / per, i = Math.floor(cyc) % R, p = cyc - Math.floor(cyc);
      const sweep = Math.min(1, p / 0.8), j = Math.min(K - 1, Math.floor(sweep * K)), done = p >= 0.8;
      const fmt = (v) => (Number.isNaN(v) ? '?' : f1d(v));
      V.text(g, 'x', x0 - 12, yx + cs / 2, { size: 15, font: 'mono', weight: 700, align: 'right', base: 'middle', color: C.b });
      V.text(g, 'W', x0 - 12, yW + (R * (cs + gap)) / 2, { size: 15, font: 'mono', weight: 700, align: 'right', base: 'middle', color: C.a });
      V.text(g, 'x = one token (vector) · W = one layer’s weights', x0 - 30, 22, { size: 11.5, font: 'mono', color: C.muted, maxW: w - x0 + 20 });
      V.matrix(g, { x: x0, y: yx, data: [xv], cell: cs, gap, cmap: 'div', min: -2, max: 2, values: true, fmt, hiCells: done ? [] : [[0, j]], hiColor: C.b });
      const hiC = []; if (!done) hiC.push([i, j]);
      V.matrix(g, { x: x0, y: yW, data: Wm, cell: cs, gap, cmap: 'div', min: -2, max: 2, values: true, fmt, hiRow: i, hiCells: hiC });
      if (!done) { // product line: x[j] goes down to W[i][j]
        const cx = x0 + j * (cs + gap) + cs / 2;
        V.arrow(g, cx, yx + cs + 2, cx, yW + i * (cs + gap) - 3, { color: V.alpha(C.b, 0.8), lw: 1.6, head: 6 });
      }
      const ex = x0 + Ww + 14;
      V.text(g, '=', ex, yW + (R * (cs + gap)) / 2, { size: 20, align: 'center', base: 'middle', color: C.muted });
      const yd = yv.map((v, r) => [r < i || (r === i && done) ? v : NaN]);
      V.matrix(g, { x: ex + 14, y: yW, data: yd, cell: cs, gap, cmap: 'div', min: -4, max: 4, values: true, fmt, hiCells: [[i, 0]], hiColor: C.a });
      V.text(g, 'y', ex + 14 + cs / 2, yW - 8, { size: 15, font: 'mono', weight: 700, align: 'center', color: C.a });
      let acc = 0; const last = done ? K - 1 : j;
      for (let q = 0; q <= last; q++) acc += Wm[i][q] * xv[q];
      const shown = []; for (let q = 0; q <= Math.min(last, small ? 1 : 2); q++) shown.push(`(${f1d(Wm[i][q])})(${f1d(xv[q])})`);
      const yb = yW + R * (cs + gap) + 28;
      V.text(g, `y${subN(i)} = ${shown.join(' + ')}${last > (small ? 1 : 2) ? ' + …' : ''} = ${f1d(Math.round(acc * 100) / 100)}${done ? ' ✓' : ''}`, w / 2, yb, { size: 13, font: 'mono', align: 'center', color: C.ink, maxW: w - 20 });
      V.text(g, 'multiply, add, next row — millions of rows per token', w / 2, yb + 22, { size: 11.5, align: 'center', color: C.muted, maxW: w - 20 });
    }

    function drawEngine(g, w, h, t, C) {
      const cols = [C.a, C.b, C.c, C.d], U = 4, P = 2.2;
      const n = Math.floor(t / P), ph = (t % P) / P;
      const narrow = w < 560;
      const ux = narrow ? 22 : 30, uy = (u) => 62 + u * ((h - 132) / (U - 1));
      const sw = narrow ? 80 : 124, sx = (narrow ? w * 0.27 : w * 0.36) - sw / 2, sy = h / 2 - 34;
      const gx = narrow ? w * 0.48 : w * 0.6, gw = w - gx - (narrow ? 8 : 14), gy = 30, gh = h - 60;
      const ip = narrow ? 8 : 10, iw = gw - 2 * ip;
      V.text(g, 'users', 14, 30, { size: 11.5, font: 'mono', color: C.muted });
      // fixed tracks (there and back)
      for (let u = 0; u < U; u++) {
        V.line(g, ux + 14, uy(u), sx, sy + 20 + u * 9, { color: V.alpha(cols[u], 0.18), lw: 1.2, dash: [3, 5] });
      }
      for (let u = 0; u < U; u++) { V.dot(g, ux, uy(u), 13, V.alpha(cols[u], 0.25)); V.text(g, 'ABCD'[u], ux, uy(u) + 0.5, { size: 12, weight: 700, align: 'center', base: 'middle', color: cols[u] }); }
      V.box(g, sx, sy, sw, 68, { r: 12, fill: V.alpha(C.b, 0.14), stroke: C.b, lw: 1.5 });
      V.text(g, 'Scheduler', sx + sw / 2, sy + 26, { size: narrow ? 12 : 12.5, weight: 600, align: 'center', color: C.ink, maxW: sw - 8 });
      V.text(g, `step ${n + 1}`, sx + sw / 2, sy + 46, { size: 11, font: 'mono', align: 'center', color: C.muted });
      V.box(g, gx, gy, gw, gh, { r: 12, fill: V.alpha(C.a, 0.08), stroke: V.alpha(C.a, 0.6), lw: 1.5 });
      V.text(g, 'GPU(s) · model', gx + gw / 2, gy + 22, { size: 12.5, weight: 600, align: 'center', color: C.ink, maxW: iw });
      let yy = wrapText(g, 'batch: 4 conversations per step', gx + gw / 2, gy + 40, iw, 13, { size: 11, font: 'mono', align: 'center', color: C.muted });
      // KV cache in blocks per conversation
      yy = wrapText(g, 'KV cache (blocks per conversation)', gx + ip, yy + 22, iw, 13, { size: 11, color: C.muted });
      const kvTop = yy + 7, bs = Math.min(13, iw / 10 - 2), M = Math.floor((iw + 2) / (bs + 2));
      for (let u = 0; u < U; u++) {
        const fill = Math.min(M, 1 + Math.floor(((n + u * 5) % 36) / (2 + u)));
        for (let b = 0; b < M; b++) {
          V.rr(g, gx + ip + b * (bs + 2), kvTop + u * (bs + 4), bs, bs, 3);
          g.fillStyle = b < fill ? V.alpha(cols[u], b === fill - 1 && ph > 0.6 ? 0.95 : 0.6) : V.alpha(C.muted, 0.1); g.fill();
        }
      }
      // box footer (wrapped into lines, not squeezed)
      const botL = wrapLines(g, '1 read of the weights serves all 4 conversations', iw, { size: 11 });
      const botY = gy + gh - 12 - (botL.length - 1) * 13;
      botL.forEach((l, i) => V.text(g, l, gx + gw / 2, botY + i * 13, { size: 11, align: 'center', color: C.a }));
      // model layers: the batch goes through all of them, one read of the weights per step
      yy = wrapText(g, 'layers (weights read 1× per step)', gx + ip, kvTop + U * (bs + 4) + 16, iw, 13, { size: 11, color: C.muted });
      const ly0 = yy + 8, nL = 8, lgap = Math.max(5, Math.min(13, (botY - 16 - ly0) / nL));
      const fw = ph < 0.4 ? -1 : Math.min(nL, Math.floor(((ph - 0.4) / 0.25) * nL));
      for (let L = 0; L < nL; L++) {
        const y = ly0 + (nL - 1 - L) * lgap + 2;
        V.rr(g, gx + ip, y, iw, Math.max(3, lgap - 4), 3);
        g.fillStyle = L <= fw && ph < 0.75 ? V.alpha(C.a, 0.85) : V.alpha(C.a, 0.18); g.fill();
      }
      // particles
      for (let u = 0; u < U; u++) {
        const a = [ux + 14, uy(u)], s2 = [sx, sy + 20 + u * 9], gi = [gx, gy + gh / 2 + u * 8];
        if (ph < 0.3) { const k = V.ease.inOut(ph / 0.3); const q = [V.lerp(a[0], s2[0], k), V.lerp(a[1], s2[1], k)]; V.glow(g, q[0], q[1], 9, cols[u]); V.dot(g, q[0], q[1], 4, cols[u]); }
        else if (ph < 0.45) { const k = V.ease.inOut((ph - 0.3) / 0.15); V.rr(g, V.lerp(sx + sw, gi[0], k) - 5, V.lerp(s2[1], gi[1], k) - 5, 10, 10, 2); g.fillStyle = cols[u]; g.fill(); }
        else if (ph >= 0.72) { const k = V.ease.inOut((ph - 0.72) / 0.28); const pt = V.bez([gx, gi[1] + 30], [gx - 60, h - 20], [ux + 90, uy(u) + 40], [ux + 14, uy(u)], k); V.glow(g, pt[0], pt[1], 10, cols[u]); V.dot(g, pt[0], pt[1], 3.5, cols[u]); }
      }
      V.arrow(g, sx + sw + 4, sy + 34, gx - 6, sy + 34, { color: V.alpha(C.muted, 0.45), lw: 1.2, dash: [4, 4] });
      if (ph >= 0.72) V.text(g, '+1 token each', ux + 24, h - 12, { size: 11, font: 'mono', color: C.ink });
    }

    function drawHW(g, w, h, t, C) {
      const narrow = w < 600;
      // on desktop, the nodes shrink enough for the link label to fit between them (≥ 118 px)
      const nw = Math.min(w * 0.42, (w - 36 - 118) / 2);
      const nodes = narrow ? [[14, 26, w - 28, 150], [14, h - 176, w - 28, 150]] : [[18, 46, nw, 236], [w - 18 - nw, 46, nw, 236]];
      const WG = 82.94, KVG = 11, REST = 121.69 - WG - KVG;
      const segs = [[WG, C.b, 'weights ' + nf(WG, 2)], [KVG, C.c, 'KV ' + nf(KVG, 0)], [REST, V.alpha(C.muted, 0.45), 'rest ' + nf(REST, 1) + ' GiB', C.muted]];
      nodes.forEach(([x, y, W, Hh], k) => {
        V.box(g, x, y, W, Hh, { r: 14, fill: V.alpha(C.a, 0.05), stroke: V.alpha(C.a, 0.5), lw: 1.5 });
        V.text(g, k ? 'Node B · worker (rank 1)' : 'Node A · head (rank 0) + API', x + 14, y + 22, { size: 12.5, weight: 600, color: C.ink, maxW: W - 28 });
        const bx = x + 14, by = y + 34, bw = W - 28, bh = narrow ? 18 : 24;
        let cx = bx;
        segs.forEach(([v, col]) => { const sw = (v / 121.69) * bw; V.rr(g, cx, by, Math.max(2, sw - 1.5), bh, 5); g.fillStyle = col; g.fill(); cx += sw; });
        V.text(g, '121.7 GiB visible to CUDA', bx, by + bh + 16, { size: 11, color: C.muted, maxW: bw });
        // colored legend, wrapping instead of squeezing
        let lx = bx, ly = by + bh + 32;
        segs.forEach(([, col, lab, tc], i) => {
          const s = lab + (i < segs.length - 1 ? ' ·' : '');
          const wl = tw(g, s, 11, 400, 'mono');
          if (lx > bx && lx + wl > bx + bw) { lx = bx; ly += 14; }
          V.text(g, s, lx, ly, { size: 11, font: 'mono', color: tc || col });
          lx += wl + 6;
        });
        const gw = narrow ? 118 : 132, ghh = narrow ? 36 : 48;
        const gx2 = narrow ? x + W - gw - 14 : x + W / 2 - gw / 2, gy2 = y + Hh - ghh - (narrow ? 8 : 12);
        V.box(g, gx2, gy2, gw, ghh, { r: 10, fill: V.alpha(C.a, 0.18), stroke: C.a, lw: 1.5, label: 'GB10 GPU · sm_121', size: 11.5 });
        const px = narrow ? gx2 + gw / 2 : x + W / 2, py0 = ly + 10, py1 = gy2 - 4;
        if (!narrow && py1 - py0 > 8) {
          V.line(g, px, py0, px, py1, { color: V.alpha(C.a, 0.35), lw: 6 });
          for (let q = 0; q < 4; q++) { const u = ((t * 0.9 + q / 4) % 1); V.dot(g, px, V.lerp(py0, py1, u), 2.6, C.a); }
          V.text(g, '~273 GB/s', px + 10, (py0 + py1) / 2 + 4, { size: 11, font: 'mono', color: C.a });
        } else {
          V.text(g, '~273 GB/s ↓', gx2 - 8, gy2 + ghh / 2 + 4, { size: 11, font: 'mono', align: 'right', color: C.a });
        }
      });
      // RDMA link
      let a, b;
      if (narrow) { a = [w / 2, nodes[0][1] + nodes[0][3]]; b = [w / 2, nodes[1][1]]; }
      else { a = [nodes[0][0] + nodes[0][2], nodes[0][1] + nodes[0][3] / 2]; b = [nodes[1][0], a[1]]; }
      V.line(g, a[0], a[1], b[0], b[1], { color: C.d, lw: 2.5, dash: [3, 5], dashOffset: -t * 30 });
      for (let q = 0; q < 3; q++) { const u = (t * 0.6 + q / 3) % 1; const pt = V.along(a[0], a[1], b[0], b[1], q % 2 ? 1 - u : u); V.rr(g, pt[0] - 4, pt[1] - 3, 8, 6, 2); g.fillStyle = C.d; g.fill(); }
      const lx = narrow ? w / 2 + 10 : (a[0] + b[0]) / 2, ly = narrow ? (a[1] + b[1]) / 2 + 4 : a[1] - 12;
      V.text(g, 'RDMA · 200 Gb/s', lx, ly, { size: 11, font: 'mono', align: narrow ? 'left' : 'center', color: C.d, maxW: narrow ? w - lx - 8 : b[0] - a[0] - 8 });
      if (!narrow) V.text(g, '(~25 GB/s)', lx, ly + 28, { size: 10.5, font: 'mono', align: 'center', color: C.muted, maxW: b[0] - a[0] - 8 });
      V.text(g, 'example: GLM-5.3-Flash at TP=2, per GPU', w / 2, h - 8, { size: 11, align: 'center', color: C.muted });
    }

    V.fig(f1, (g, w, h, t) => {
      const C = V.C();
      g.globalAlpha = V.ease.out(Math.min(1, st1.elapsed() * 2.2));
      if (st1.i === 0) drawWeights(g, w, h, t, C);
      else if (st1.i === 1) drawMatVec(g, w, h, t, C);
      else if (st1.i === 2) drawEngine(g, w, h, t, C);
      else drawHW(g, w, h, t, C);
      g.globalAlpha = 1;
    });

    /* ================================================================
       Figure 2 — anatomy (stepper + canvas)
       ================================================================ */
    const layers = [
      { t: 'Hardware & network', d: 'Two machines with unified-memory GPUs (128 GB each) linked by a high-speed RDMA network card. This sets the physical limits: capacity, bandwidth and latency.<br>' + B('all') + ' 2× DGX Spark (GB10, <code>sm_121</code>) with a ConnectX-7 cable (RoCE, 200 Gb/s); GLM documents variants with 3 and 4 nodes (the 4-node one is experimental), and DeepSeek, with 3.' },
      { t: 'Software image', d: 'A Docker container with CUDA, PyTorch, vLLM (sometimes a specific version or a fork) and kernels compiled for the exact GPU architecture.<br>' + B('glm') + ' vLLM fork + Dockerfile with build patches · ' + B('qwen') + ' vLLM day-0 image (or the stock 0.30.0) · ' + B('ds') + ' Anemll 0.1.1 image (vLLM 0.25.2.dev0) · ' + B('tf') + ' NVIDIA’s PyTorch container with TensorFold v0.6.0 in place of vLLM, pinned by digest.' },
      { t: 'Weights (checkpoint)', d: 'Which version of the model, in which quantization format (EXL3, NVFP4, FP8…), downloaded from Hugging Face with a pinned and verified revision.<br>' + B('glm') + ' EXL3 4 bpw (Mia-AiLab mirror, 175.6 GB) · ' + B('qwen') + ' NVIDIA’s NVFP4 (132.6 GB) · ' + B('ds') + ' official FP4+FP8, revision <code>86f746b3…</code> (~157 GiB).' },
      { t: 'Patches & overlays', d: 'Python/CUDA files that replace or fix parts of vLLM to support the model, the format or the hardware. This is where a good share of a recipe’s “engineering” lives.<br>' + B('glm') + ' 25 overlays in a fixed order · ' + B('qwen') + ' overlays mounted on top of the image (N-gram, MXFP8, FP8 KV…) · ' + B('ds') + ' a <i>fail-closed</i> chain of hotfixes at container boot · ' + B('tf') + ' 53 patches applied once, at image build time, not on every start.' },
      { t: 'Topology', d: 'How to split the model across the machines: tensor parallel (TP), expert parallel (EP), how many nodes, which network interface NCCL uses.<br>' + B('glm') + ' TP=2 · ' + B('qwen') + ' TP=2 + EP · ' + B('ds') + ' TP=2 — all three with the <code>mp</code> executor and <code>--nnodes 2</code> · ' + B('tf') + ' TP=2 is mandatory, one rank per Spark: an all-gather of the partials and a rank-order sum in place of the all-reduce.' },
      { t: 'Engine arguments', d: 'Maximum context length, the fraction of memory for the KV cache, tokens per step, concurrent sequences, KV cache dtype, tool and reasoning parsers.<br>Context: 850,000 ' + B('glm') + ', 262,144 ' + B('qwen') + ', 1,048,576 ' + B('ds') + ' · KV: <code>fp8</code>, <code>fp8</code>, <code>nvfp4_ds_mla</code> · sequences: 4, 8, 6 · ' + B('tf') + ' a 1,048,576-token window per request, up to 4 requests (v1.3.2; up to 8 since v1.5) in a shared pool of ~2.9M tokens, FP8 KV of 528 B per token and layer.' },
      { t: 'Speculation', d: 'A “drafter” (MTP, DFlash2, DSpark) proposes several tokens per step and the big model only verifies them — multiplying decode speed without changing the output.<br>' + B('glm') + ' DFlash2 k=7 · ' + B('qwen') + ' MTP k=3 · ' + B('ds') + ' DSpark k=6 · ' + B('tf') + ' DFlash2 up to 7 per round + copy drafts up to 15, accepted only if equal to the token drawn with the key.' },
      { t: 'Verification & tuning', d: 'Smoke tests, numerical checks against the reference, throughput and latency benchmarks, concurrency ladders — and iterating on the parameters.<br>' + B('glm') + ' KLD of 0.024555 nats against BF16 (TR3 checkpoint) · ' + B('qwen') + ' needles at 13k and 104k with FP8 vs. BF16 KV (12/12) · ' + B('ds') + ' RULER-lite, a tool-calling battery and corrupted-text checks up to ~900k · ' + B('tf') + ' identical replies with and without drafts and with 4 requests together or alone; GSM8K 98.0% (v1.3.2, TR3).' },
    ];
    const f2 = root.querySelector('#fig-anatomia');
    const stage2 = f2.querySelector('.la-fig-stage');
    const fitH2 = () => { const w = stage2.clientWidth || 800; stage2.style.height = (w < 560 ? 560 : 380) + 'px'; };
    fitH2();
    const st2 = UI.stepper(f2, layers, { autoplay: 6500 });
    const parts2 = Array.from({ length: 70 }, (_, i) => ({ p: i / 70, lane: i % 3 }));
    // what each recipe layer controls in the flow (API, Scheduler, nodes A/B, link, token output)
    const HI2 = [['A', 'B', 'link'], ['A', 'B', 'img'], ['A', 'B', 'w'], ['S', 'A', 'B', 'patch'], ['A', 'B', 'link', 'tp'], ['API', 'S'], ['tok', 'spec'], ['tok', 'check']];
    const NOTE2 = [
      '2 nodes · 121.7 GiB visible each · ~273 GB/s · RDMA 200 Gb/s',
      'the same image (engine + kernels for sm_121) runs on both nodes',
      'each node loads its half of the checkpoint’s weights',
      'overlays swap vLLM files at startup, before the first token',
      'TP=2: at every layer, the nodes sum partial results over the network (all-reduce)',
      'max context, batch, GMU and KV dtype decide who gets into the queue',
      'k drafts per step, verified all at once by the big model',
      'benchmarks and KLD check the output tokens against the reference',
    ];
    V.fig(f2, (g, w, h, t) => {
      const C = V.C();
      const cols = [C.a, C.b, C.c, C.d, C.e, C.f, C.a, C.b];
      const narrow = w < 560;
      const sx = narrow ? 14 : 24, sw = narrow ? w - 28 : Math.min(250, w * 0.34);
      const top = narrow ? 10 : 26, stackH = narrow ? h - 230 : h - 50, lh = stackH / layers.length;
      layers.forEach((L, i) => {
        const k = layers.length - 1 - i; // hardware at the bottom
        const y = top + k * lh;
        const on = st2.i === i;
        const col = cols[i];
        V.box(g, sx + (on ? -4 : 0), y, sw + (on ? 8 : 0), lh - 6, { r: 9, fill: on ? V.alpha(col, 0.28) : V.alpha(col, 0.08), stroke: on ? col : V.alpha(col, 0.35), lw: on ? 2 : 1 });
        V.text(g, String(i + 1).padStart(2, '0'), sx + 12, y + (lh - 6) / 2 + 4, { size: 11, font: 'mono', color: col });
        V.text(g, L.t, sx + 38, y + (lh - 6) / 2 + 5, { size: 13, weight: on ? 600 : 400, color: on ? C.ink : C.muted, maxW: sw - 46 });
        if (on && !narrow) V.glow(g, sx + sw + 4, y + (lh - 6) / 2, 16, col);
      });
      // request flow: on the right on desktop, compact at the bottom on phones
      const x0 = narrow ? 14 : sx + sw + 50, x1 = narrow ? w - 14 : w - 30;
      if (x1 - x0 < 200) return;
      const cy = narrow ? top + stackH + 82 : h / 2 - 14, dy = narrow ? 46 : 70;
      const hi = new Set(HI2[st2.i]), hc = cols[st2.i];
      const labs = narrow ? ['API', 'Scheduler', 'Node A', 'Node B', 'tokens'] : ['API', 'Scheduler', 'Node A · GPU', 'Node B · GPU', 'tokens'];
      const bws = labs.map((l) => tw(g, l, 12, 500) + 24);
      const xApi = x0 + bws[0] / 2, xTok = x1 - bws[4] / 2, xNode = x0 + (x1 - x0) * 0.62;
      let xS = x0 + (x1 - x0) * 0.3;
      xS = Math.max(xS, xApi + bws[0] / 2 + 16 + bws[1] / 2); // minimum spacing between the boxes
      xS = Math.min(xS, xNode - bws[2] / 2 - 16 - bws[1] / 2);
      const nodes = [
        { k: 'API', x: xApi, y: cy, l: labs[0], c: C.d, bw: bws[0] },
        { k: 'S', x: xS, y: cy, l: labs[1], c: C.b, bw: bws[1] },
        { k: 'A', x: xNode, y: cy - dy, l: labs[2], c: C.a, bw: bws[2] },
        { k: 'B', x: xNode, y: cy + dy, l: labs[3], c: C.a, bw: bws[3] },
        { k: 'tok', x: xTok, y: cy, l: labs[4], c: C.c, bw: bws[4] },
      ];
      const edges = [[0, 1], [1, 2], [1, 3], [2, 4], [3, 4]];
      edges.forEach(([a, b]) => V.line(g, nodes[a].x, nodes[a].y, nodes[b].x, nodes[b].y, { color: V.alpha(C.muted, 0.4), lw: 1.5, dash: [4, 6], dashOffset: -t * 20 }));
      // RDMA link between the nodes
      const lk = hi.has('link');
      const la = [nodes[2].x, nodes[2].y + 18], lb = [nodes[3].x, nodes[3].y - 18];
      if (lk) V.glow(g, la[0], (la[1] + lb[1]) / 2, 30, hc, 0.5);
      V.line(g, la[0], la[1], lb[0], lb[1], { color: lk ? hc : C.a, lw: lk ? 4 : 2.5, dash: [2, 5], dashOffset: -t * 40 });
      V.text(g, 'RDMA', la[0] + 10, cy + 4, { size: 11, font: 'mono', weight: lk ? 600 : 400, color: lk ? hc : C.a });
      if (hi.has('tp')) for (let q = 0; q < 4; q++) { const u = (t * 0.8 + q / 4) % 1, pt = V.along(la[0], la[1], lb[0], lb[1], q % 2 ? 1 - u : u); V.rr(g, pt[0] - 4, pt[1] - 3, 8, 6, 2); g.fillStyle = hc; g.fill(); }
      // request and token particles
      const spec = hi.has('spec');
      parts2.forEach((p) => {
        const u = (p.p + t * 0.16) % 1;
        let pt;
        const lane = p.lane === 0 ? 2 : 3;
        if (u < 0.33) pt = V.along(nodes[0].x, nodes[0].y, nodes[1].x, nodes[1].y, u / 0.33);
        else if (u < 0.66) pt = V.along(nodes[1].x, nodes[1].y, nodes[lane].x, nodes[lane].y, (u - 0.33) / 0.33);
        else if (!spec) pt = V.along(nodes[lane].x, nodes[lane].y, nodes[4].x, nodes[4].y, (u - 0.66) / 0.34);
        if (pt) V.dot(g, pt[0], pt[1], 2.4, u > 0.66 ? C.c : C.a);
      });
      if (spec) { // tokens come out in bursts of k per step
        [2, 3].forEach((ni, r) => {
          const u = (t * 0.45 + r * 0.5) % 1;
          for (let q = 0; q < 5; q++) { const uu = u - q * 0.035; if (uu < 0) continue; const pt = V.along(nodes[ni].x, nodes[ni].y, nodes[4].x, nodes[4].y, uu); V.rr(g, pt[0] - 3.5, pt[1] - 3.5, 7, 7, 2); g.fillStyle = hc; g.fill(); }
        });
      }
      // boxes (lit when the recipe layer touches them)
      nodes.forEach((n) => {
        const on = hi.has(n.k);
        if (on) V.glow(g, n.x, n.y, n.bw * 0.7, hc, 0.45);
        V.box(g, n.x - n.bw / 2, n.y - 17, n.bw, 34, { r: 10, fill: C.bg });
        V.box(g, n.x - n.bw / 2, n.y - 17, n.bw, 34, { r: 10, fill: V.alpha(on ? hc : n.c, on ? 0.3 : 0.12), stroke: on ? hc : V.alpha(n.c, 0.55), lw: on ? 2.5 : 1.2, label: n.l, size: 12, color: on ? C.ink : C.muted });
      });
      const nA = nodes[2], nB = nodes[3];
      if (hi.has('img')) [nA, nB].forEach((n, r) => {
        g.setLineDash([4, 3]); V.box(g, n.x - n.bw / 2 - 8, n.y - 25, n.bw + 16, 50, { r: 12, stroke: hc, lw: 1.3 }); g.setLineDash([]);
        V.text(g, 'container', n.x + n.bw / 2 + 12, n.y + (r ? 22 : -18), { size: 10.5, font: 'mono', color: hc });
      });
      if (hi.has('w')) [nA, nB].forEach((n, r) => {
        const yy = r ? n.y + 30 : n.y - 38;
        for (let i = 0; i < 2; i++) for (let j = 0; j < 6; j++) { V.rr(g, n.x - n.bw / 2 + j * 8, yy + i * 7, 6, 5, 1.5); g.fillStyle = V.alpha(hc, 0.35 + 0.4 * Math.abs(Math.sin(t * 2 + i + j * 0.7 + r))); g.fill(); }
        V.text(g, '½ of the weights', n.x - n.bw / 2 + 54, yy + 10, { size: 10.5, font: 'mono', color: hc });
      });
      if (hi.has('patch')) [nodes[1], nA, nB].forEach((n, r) => {
        const px = n.x + n.bw / 2 - 2, py = n.y - 17, pr = 7 + Math.sin(t * 4 + r) * 1.2;
        V.dot(g, px, py, pr, hc); V.text(g, '+', px, py + 0.5, { size: 11, weight: 700, align: 'center', base: 'middle', color: C.bg });
      });
      if (hi.has('S')) V.text(g, 'batch ≤ 4–8', nodes[1].x, nodes[1].y + 32, { size: 10.5, font: 'mono', align: 'center', color: hc });
      if (hi.has('API')) V.text(g, 'max-model-len · GMU', x0, nodes[0].y - 25, { size: 10.5, font: 'mono', color: hc });
      if (spec) V.text(g, 'k = 7 · 3 · 6', nodes[4].x, nodes[4].y + 32, { size: 10.5, font: 'mono', align: 'center', color: hc });
      if (hi.has('check')) {
        V.text(g, '✓', nodes[4].x + nodes[4].bw / 2 - 6, nodes[4].y - 20, { size: 15, weight: 700, align: 'center', color: hc });
        for (let q = 0; q < 4; q++) { const bh = 6 + 10 * Math.abs(Math.sin(t * 1.5 + q)); V.rr(g, nodes[4].x - 16 + q * 9, nodes[4].y + 40 - bh, 6, bh, 2); g.fillStyle = V.alpha(hc, 0.7); g.fill(); }
      }
      // step caption
      wrapText(g, NOTE2[st2.i], (x0 + x1) / 2, cy + dy + (narrow ? 64 : 62), x1 - x0, 15, { size: 12, weight: 600, align: 'center', color: hc });
    });

    /* ================================================================
       Figure 3 — back-of-the-envelope calculator (with real presets)
       ================================================================ */
    const f3 = root.querySelector('#fig-conta');
    const stage3 = f3.querySelector('.la-fig-stage');
    const fitH3 = () => { const w = stage3.clientWidth || 800; stage3.style.height = (w < 560 ? 460 : 400) + 'px'; };
    fitH3();
    const ctl3 = UI.controls(f3);
    const NODE_GIB = 121.69, BW_NODE = 273e9, MAXT = 120;
    const S3 = { total: 300, bTot: 4.5, active: 20, bAct: 4.5, nodes: 2, gmu: 0.85, tau: 1, rsv: 6, preset: 'livre' };
    const NOTE_FREE = 'Free mode: play as much as you like. In a model entirely in one format, the active bits ≈ the average bits; in a mixed-precision MoE, each active parameter costs more, because the dense parts stay in FP8/BF16. The engine reserve is the one from the last preset (6 GiB at the start, an illustrative value). Pick a preset to load the real numbers and compare the ceiling with the measured values.';
    const PRE = {
      glm: { total: 321.3, bTot: 4.37, active: 17.4, bAct: 6.9, nodes: 2, gmu: 0.85, rsv: 9, meas: [[24.6, 'MTP k=2'], [27.1, 'prose'], [65.1, 'structured']], note: B('glm') + ' 321.3 B total, 175.6 GB on disk → <b>4.37 bits</b> on average (memory). The per-token read is another story: the experts (97% of the total) are at 4 bits, but what gets read for each token is dominated by the dense parts — KDA, MLA, shared experts, <code>lm_head</code> — in FP8/BF16. Per the repo’s cost model, ≈ 5.4 GB of dense parts + ≈ 2.1 GB of experts ≈ 7.5 GB per GPU per token without speculation (estimate) → <b>6.9 effective bits</b> over the ~17.4 B active. A reserve of ~9 GiB (103 − 83 − 11 in the recipe, which pins the pool at 11 GiB). <b>Measured</b> in the same test (<code>bench_decode</code>, 2026-08-30): 24.6 tok/s on the MTP k=2 rollback; with DFlash2 k=7, 27.1 on prose and 65.1 on structured text.' },
      qwen: { total: 180.0, bTot: 5.9, active: 7.3, bAct: 12.8, nodes: 2, gmu: 0.8, rsv: 5, meas: [[24.5, 'no spec'], [52.1, 'MTP=3']], note: B('qwen') + ' 180.0 B stored (125.7 main + 51.2 N-gram + 2.6 MTP + vision), 132.6 GB → <b>5.9 bits</b> on average; 7.3 B active with embeddings and <code>lm_head</code>. The per-token read is almost all BF16 — GDN projections, the Gated Residual replicated on both GPUs, a 248,320 × 2560 <code>lm_head</code> — and the NVFP4 experts are only ~11% of it. The repo models the step with MTP=3 at ≈ 9.2 GiB per GPU; without the MTP, without the 3 extra reads of the <code>lm_head</code> and with ~10 experts per layer instead of ~32, ≈ 5.5 GiB per GPU remain (interpretation) → <b>12.8 effective bits</b>. GMU 0.80: at 0.835, a load test left <code>MemAvailable</code> at 0.3–0.9 GiB — the squeeze was on the system, not on the KV. <b>Measured</b> (greedy, 2026-09-05): 24.5 tok/s without speculation and 52.1 with MTP=3.' },
      ds: { total: 304.6, bTot: 4.43, active: 13.2, bAct: 6.0, nodes: 2, gmu: 0.835, rsv: 5, meas: [[56, 'DSpark k=6'], [75.4, '0731']], note: B('ds') + ' 284.3 B of backbone + 19.8 B of DSpark + 0.47 B of vision, ~157 GiB → <b>4.43 bits</b> on average; 13.2 B active. The repo estimates 16.5–17 GB read per GPU per step with k=6 (verifying 7 positions wakes as many as ~42 distinct experts per layer). With a single token, ≈ 5 GB per GPU remain — 7 experts × 43 layers (~2 GB), attention ≈ 2.4 GB and one read of the <code>lm_head</code> (interpretation) → <b>~6.0 effective bits</b>. GMU 0.835 → measured pool of 17.04 GiB. <b>Measured</b>: 56 tok/s with DSpark k=6 (median of 8, 2026-09-02); 75.4 with the 0731 checkpoint (2026-08-01).' },
      mimo: { total: 310.8, bTot: 4.45, active: 15, bAct: 6.8, nodes: 2, gmu: 0.85, rsv: 4, meas: [[25.9, 'prose'], [53.3, 'average'], [70.5, 'code']], note: 'MiMo-V2.6-Flash: 310.8 B (Hugging Face count), 15 B active (card), 172.9 GB → <b>4.45 bits</b> on average. Per token (interpretation, from the tutorial’s weight table): active experts in MXFP4 ≈ 5.0 GB, QKV in FP8 ≈ 2.9 GB, <code>o_proj</code> in BF16 ≈ 3.2 GB, <code>lm_head</code> ≈ 1.25 GB and the rest ≈ 0.4 GB — ≈ 12.8 GB → <b>6.8 effective bits</b>. GMU 0.85 with ~4 GiB of reserve (the tutorial’s estimate). <b>Measured by a community recipe</b> (vLLM TP=2, DFlash k=7): 25.9 tok/s on prose, 70.5 on code, 53.3 averaged over the categories.' },
    };
    const gmuFmt = (v) => nf(v, Math.abs(v * 100 - Math.round(v * 100)) > 1e-6 ? 3 : 2);
    let applying = false;
    const selP = wrapCtl(UI.select(ctl3, { label: 'Preset', options: [{ v: 'livre', label: 'Free' }, { v: 'glm', label: 'GLM-5.3-Flash (EXL3)' }, { v: 'qwen', label: 'Qwen3.8-Flash-Next (NVFP4)' }, { v: 'ds', label: 'DeepSeek-V4-Flash (DSpark)' }, { v: 'mimo', label: 'MiMo-V2.6-Flash (tutorial)' }], value: 'livre', onChange: (v) => applyPreset(v) }));
    const manual = () => { if (applying || S3.preset === 'livre') return; S3.preset = 'livre'; selP.el.querySelector('select').value = 'livre'; note3.innerHTML = NOTE_FREE; };
    const sT = UI.slider(ctl3, { label: 'Total params', min: 10, max: 1200, step: 0.1, value: S3.total, fmt: (v) => nf(v, v < 100 ? 1 : 0) + ' B', onInput: (v) => { S3.total = v; if (S3.active > v) sA.set(v); manual(); H3.redraw(); } });
    const sB = UI.slider(ctl3, { label: 'Avg. bits (memory)', min: 2, max: 16, step: 0.01, value: S3.bTot, fmt: (v) => nf(v, 2), onInput: (v) => { S3.bTot = v; manual(); H3.redraw(); } });
    const sA = UI.slider(ctl3, { label: 'Active params', min: 1, max: 120, step: 0.1, value: S3.active, fmt: (v) => nf(v, 1) + ' B', onInput: (v) => { S3.active = Math.min(v, S3.total); manual(); H3.redraw(); } });
    const sBA = UI.slider(ctl3, { label: 'Active bits (speed)', min: 2, max: 16, step: 0.01, value: S3.bAct, fmt: (v) => nf(v, 2), onInput: (v) => { S3.bAct = v; manual(); H3.redraw(); } });
    const sN = wrapCtl(UI.seg(ctl3, { label: 'Nodes', options: [1, 2, 3, 4], value: 2, onChange: (v) => { S3.nodes = v; manual(); H3.redraw(); } }));
    const sG = UI.slider(ctl3, { label: 'GMU', min: 0.5, max: 0.95, step: 0.005, value: S3.gmu, fmt: gmuFmt, onInput: (v) => { S3.gmu = v; H3.redraw(); } });
    const sTau = UI.slider(ctl3, { label: 'Tokens per step (τ)', min: 1, max: 8, step: 0.1, value: 1, fmt: (v) => nf(v, 1), onInput: (v) => { S3.tau = v; H3.redraw(); } });
    [sT, sB, sA, sBA, sG, sTau].forEach(wrapCtl);
    const note3 = LA.h('div', { style: { flexBasis: '100%', fontSize: '13px', lineHeight: '1.55', color: 'var(--text-2)' }, html: NOTE_FREE });
    ctl3.appendChild(note3);
    function applyPreset(k) {
      S3.preset = k;
      const p = PRE[k];
      if (!p) { note3.innerHTML = NOTE_FREE; H3.redraw(); return; }
      applying = true;
      sT.set(p.total); sA.set(p.active); sB.set(p.bTot); sBA.set(p.bAct); sN.set(p.nodes); sG.set(p.gmu); sTau.set(1);
      S3.rsv = p.rsv;
      applying = false;
      S3.preset = k;
      note3.innerHTML = p.note;
      H3.redraw();
    }
    const H3 = V.fig(f3, (g, w, h) => {
      const C = V.C();
      const narrow = w < 560, pad = narrow ? 12 : 22, bw = w - 2 * pad;
      const wB = (S3.total * 1e9 * S3.bTot) / 8;             // weight bytes, total
      const wNode = wB / S3.nodes / GiB;                       // GiB of weights per node
      const budget = S3.gmu * NODE_GIB;                        // vLLM budget per node
      const kv = budget - wNode - S3.rsv;                      // what is left for the KV cache
      const outside = NODE_GIB - budget;                       // outside vLLM: OS, page cache
      const fracSum = wB / (S3.nodes * NODE_GIB * GiB);
      const state = wNode > NODE_GIB ? 3 : kv <= 0 ? 2 : kv < 20 ? 1 : 0;
      const kvCol = [C.a, C.d, C.c, C.c][state];
      // ---- memory header
      V.text(g, narrow ? 'ONE NODE · 121.7 GiB visible' : 'ONE NODE’S MEMORY · 121.7 GiB visible to CUDA', pad, 16, { size: 11, font: 'mono', color: C.muted, maxW: bw * 0.62 });
      V.text(g, `${nf(fracSum * 100, 0)}% of combined memory`, pad + bw, 16, { size: 11, font: 'mono', align: 'right', color: C.muted, maxW: bw * 0.36 });
      let y = wrapText(g, `weights: ${nf(wB / 1e9, 1)} GB = ${nf(wB / GiB, 1)} GiB ÷ ${S3.nodes} ${S3.nodes > 1 ? 'nodes' : 'node'} = ${nf(wNode, 1)} GiB per node`, pad, 34, bw, 15, { size: 12, font: 'mono', weight: 600, color: C.ink });
      // ---- one node's bar: weights | engine reserve | KV | outside vLLM
      const by = y + 10, bh = 28, X = (gib) => pad + bw * V.clamp(gib / NODE_GIB, 0, 1);
      V.box(g, pad, by, bw, bh, { r: 8, fill: V.alpha(C.muted, 0.1), stroke: V.alpha(C.muted, 0.3) });
      g.save(); V.rr(g, pad, by, bw, bh, 8); g.clip();
      const xG = X(budget);
      g.save(); g.beginPath(); g.rect(xG, by, Math.max(0, pad + bw - xG), bh); g.clip();
      g.strokeStyle = V.alpha(C.muted, 0.4); g.lineWidth = 1;
      for (let xx = xG - bh; xx < pad + bw; xx += 7) { g.beginPath(); g.moveTo(xx, by + bh); g.lineTo(xx + bh, by); g.stroke(); }
      g.restore();
      const xW = X(wNode), xR = X(Math.min(budget, wNode + S3.rsv));
      if (wNode < budget) { g.fillStyle = V.alpha(C.muted, 0.45); g.fillRect(xW, by, Math.max(0, xR - xW), bh); }
      if (kv > 0) { g.fillStyle = V.alpha(kvCol, 0.75); g.fillRect(xR, by, Math.max(0, xG - xR), bh); }
      g.fillStyle = wNode > budget ? V.alpha(C.red || C.c, 0.8) : V.alpha(C.b, 0.8); g.fillRect(pad, by, Math.max(3, xW - pad), bh);
      g.restore();
      V.line(g, xG, by - 5, xG, by + bh + 5, { color: C.ink, lw: 1.6 });
      const gmuL = `GMU ${gmuFmt(S3.gmu)}`, gW = tw(g, gmuL, 10.5, 600, 'mono');
      V.text(g, gmuL, V.clamp(xG, pad + gW / 2, pad + bw - gW / 2), by + bh + 15, { size: 10.5, font: 'mono', weight: 600, align: 'center', color: C.ink });
      // segment legend (wraps on narrow screens)
      const segs = [[V.alpha(C.b, 0.8), `weights ${nf(Math.min(wNode, 999), 1)}`], [V.alpha(C.muted, 0.45), `engine ~${nf(S3.rsv, 0)}`], [V.alpha(kvCol, 0.75), `KV ${nf(Math.max(0, kv), 1)}`], ['hatch', `outside vLLM ${nf(outside, 1)} GiB`]];
      let lx = pad, ly = by + bh + 34;
      segs.forEach(([c, l]) => {
        const wl = tw(g, l, 11.5) + 22;
        if (lx > pad && lx + wl > pad + bw) { lx = pad; ly += 17; }
        if (c === 'hatch') { V.rr(g, lx, ly - 9, 12, 10, 2); g.strokeStyle = V.alpha(C.muted, 0.7); g.lineWidth = 1; g.stroke(); V.line(g, lx + 2, ly, lx + 10, ly - 8, { color: V.alpha(C.muted, 0.7), lw: 1 }); }
        else { V.rr(g, lx, ly - 9, 12, 10, 2); g.fillStyle = c; g.fill(); }
        V.text(g, l, lx + 17, ly, { size: 11.5, color: C.muted });
        lx += wl;
      });
      const msg = state === 3 ? '✕ not even the weights fit — quantize more or add nodes'
        : state === 2 ? `✕ doesn’t fit in practice: at GMU ${gmuFmt(S3.gmu)}, weights and reserve already exceed vLLM’s budget — no KV cache left`
          : state === 1 ? `≈ fits, but tight — only ~${nf(kv, 0)} GiB per node for the KV cache; every GiB is a fight between context and batch`
            : `✓ fits — ~${nf(kv, 0)} GiB per node for the KV cache, and ${nf(outside, 0)} GiB outside vLLM for the system`;
      y = wrapText(g, msg, pad, ly + 22, bw, 17, { size: 13, weight: 500, color: [C.a, C.d, C.c, C.c][state] });
      // ---- speed
      const Bp = (S3.active * 1e9 * S3.bAct) / 8;
      const tps = (S3.tau * BW_NODE * S3.nodes) / Bp;
      const p = PRE[S3.preset];
      y = wrapText(g, `SPEED · reads ${nf(Bp / 1e9, 1)} GB per step (${nf(Bp / 1e9 / S3.nodes, 1)} per node) · bandwidth ${S3.nodes} × 273 GB/s${S3.tau > 1 ? ' · × τ = ' + nf(S3.tau, 1) : ''}${p ? ' · pink marks = measured' : ''}`, pad, y + 28, bw, 14, { size: 11, font: 'mono', color: C.muted });
      const cx = w / 2, cy = h - 28;
      const R = Math.max(46, Math.min(108, w < 420 ? w / 2 - 95 : w / 2 - 80, cy - y - 64));
      g.lineCap = 'round';
      g.lineWidth = 12; g.strokeStyle = V.alpha(C.muted, 0.18);
      g.beginPath(); g.arc(cx, cy, R, Math.PI, 2 * Math.PI); g.stroke();
      const u = Math.min(1, tps / MAXT);
      const grad = g.createLinearGradient(cx - R, 0, cx + R, 0); grad.addColorStop(0, C.b); grad.addColorStop(1, C.a);
      g.strokeStyle = grad; g.beginPath(); g.arc(cx, cy, R, Math.PI, Math.PI + Math.PI * u); g.stroke();
      g.lineCap = 'butt';
      V.text(g, '0', cx - R - 10, cy + 4, { size: 10.5, font: 'mono', align: 'right', color: C.muted });
      V.text(g, MAXT + '+', cx + R + 10, cy + 4, { size: 10.5, font: 'mono', color: C.muted });
      if (p) {
        const placed = [];
        const hit = (b) => placed.some((q) => b.x < q.x + q.w + 4 && b.x + b.w + 4 > q.x && b.y < q.y + q.h + 2 && b.y + b.h + 2 > q.y);
        p.meas.slice().sort((a1, a2) => a1[0] - a2[0]).forEach(([v, l]) => {
          const a = Math.PI + Math.PI * Math.min(1, v / MAXT), ca = Math.cos(a), sa = Math.sin(a);
          V.line(g, cx + ca * (R - 10), cy + sa * (R - 10), cx + ca * (R + 10), cy + sa * (R + 10), { color: C.c, lw: 2.5 });
          const lab = `${nf(v, 1)} ${l}`, wl = tw(g, lab, 10.5, 500, 'mono');
          const al = ca < -0.3 ? 'right' : ca > 0.3 ? 'left' : 'center';
          let rr = R + 18, bx0 = 0, ay = 0, ax = 0;
          for (let k = 0; k < 6; k++) {
            ax = cx + ca * rr; ay = cy + sa * rr + 4;
            bx0 = al === 'right' ? ax - wl : al === 'left' ? ax : ax - wl / 2;
            const sh = V.clamp(bx0, 4, w - 4 - wl) - bx0; bx0 += sh; ax += sh;
            if (!hit({ x: bx0, y: ay - 10, w: wl, h: 13 })) break;
            rr += 14;
          }
          placed.push({ x: bx0, y: ay - 10, w: wl, h: 13 });
          if (rr > R + 19) V.line(g, cx + ca * (R + 11), cy + sa * (R + 11), cx + ca * (rr - 8), cy + sa * (rr - 8), { color: V.alpha(C.c, 0.45), lw: 1 });
          halo(g, lab, ax, ay, { size: 10.5, font: 'mono', align: al, color: C.c });
        });
      }
      V.text(g, tps >= 1000 ? '>1000' : nf(tps, tps < 10 ? 1 : 0), cx, cy - 14, { size: 32, font: 'display', weight: 700, align: 'center', color: C.ink });
      V.text(g, w < 420 ? 'tok/s · ceiling, 1 conversation' : 'tok/s — theoretical ceiling, 1 conversation', cx, cy + 16, { size: 11.5, align: 'center', color: C.muted });
    }, { static: true });

    /* ================================================================
       Figure 4 — 3D stack
       ================================================================ */
    const f4 = root.querySelector('#fig-pilha');
    const stage4 = f4.querySelector('.la-fig-stage');
    Object.assign(stage4.style, { display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: '10px' });
    const left4 = LA.h('div', { style: { flex: '1.4 1 300px', minWidth: '0' } });
    const right4 = LA.h('div', { style: { flex: '1 1 300px', minWidth: '0', position: 'relative', height: '300px' } });
    const cv4 = LA.h('canvas', { style: { display: 'block', width: '100%', height: '100%' } });
    right4.appendChild(cv4);
    stage4.append(left4, right4);
    let split = false;
    const colorFor = (s, i, j) => {
      const C = V.C();
      if (!split) return V.alpha(C.series[s % 3], 0.25 + 0.5 * Math.abs(Math.sin(i * 1.7 + j * 0.9 + s)));
      return j < 5 ? V.alpha(C.a, 0.3 + 0.5 * Math.abs(Math.sin(i + j))) : V.alpha(C.b, 0.3 + 0.5 * Math.abs(Math.sin(i + j)));
    };
    const narrow4 = (stage4.clientWidth || 800) < 480;
    const C0 = V.C();
    const T4 = UI.tensor3d(left4, { slabs: 6, rows: 6, cols: 10, cell: narrow4 ? 13 : 18, depthGap: narrow4 ? 24 : 30, rotate: true, color: colorFor, slabColor: (s) => [C0.a, C0.b, C0.c][s % 3], labels: [] });
    left4.appendChild(LA.h('div', { style: { textAlign: 'center', fontSize: '12px', fontFamily: 'var(--font-mono)', color: 'var(--text-3)', marginTop: '-18px' } }, 'slabs = layers, from 0 (back) to L-1 (front)'));
    T4.explode(1.7);
    const c4 = UI.controls(f4);
    let leg4 = null;
    const repaint = () => {
      const C = V.C();
      T4.slabs.forEach((sl, s) => sl.style.setProperty('--sc', [C.a, C.b, C.c][s % 3]));
      for (let s = 0; s < 6; s++) for (let i = 0; i < 6; i++) for (let j = 0; j < 10; j++) T4.setCell(s, i, j, colorFor(s, i, j));
      if (leg4) { const is = leg4.querySelectorAll('i'); if (is[0]) is[0].style.setProperty('--c', C.a); if (is[1]) is[1].style.setProperty('--c', C.b); leg4.style.display = split ? '' : 'none'; }
    };
    UI.toggle(c4, { label: 'split across 2 nodes (tensor parallel)', value: false, onChange: (v) => { split = v; repaint(); } });
    wrapCtl(UI.slider(c4, { label: 'Spread layers', min: 0.3, max: 2.4, step: 0.1, value: 1.7, fmt: (v) => nf(v, 1) + '×', onInput: (v) => T4.explode(v) }));
    leg4 = UI.legend(c4, [[C0.a, 'Node A (half of each matrix)'], [C0.b, 'Node B (the other half)']]);
    leg4.style.display = 'none';
    if (LA.theme && LA.onCleanup) LA.onCleanup(LA.theme.onChange(() => setTimeout(repaint, 0)));
    // 2D inset: one GLM expert layer at TP=2 — gate/up split by outputs, down by inputs, all-reduce
    const ST4 = ['1 · each node multiplies x by its 1024 rows of gate and up: half of the 2048 outputs, without talking to the other', '2 · each node multiplies its half of h by its 1024 columns of down: a partial sum of the 4096 outputs', '3 · all-reduce: the nodes exchange the partials over the network and add them — once per MoE layer'];
    V.loop(cv4, (g, w, h, t) => {
      const C = V.C(), cA = C.a, cB = C.b;
      V.text(g, 'one GLM expert layer at TP=2', 6, 16, { size: 12.5, weight: 600, color: C.ink, maxW: w - 12 });
      V.text(g, 'shapes [output × input], as in PyTorch', 6, 32, { size: 11, color: C.muted, maxW: w - 12 });
      const u = V.clamp((w - 150) / 7.4, 12, 26), top = 58;
      const total = 6 * u + 124;
      const gX = Math.max(6, (w - total) / 2), gY = top + u, gW = 4 * u, gH = 2 * u;
      const hX = gX + gW + 20, hY = gY, hW = 8, hH = 2 * u;
      const dX = hX + hW + 20, dY = top, dW = 2 * u, dH = 4 * u;
      const pAX = dX + dW + 18, pBX = pAX + 12, pW = 8, yX = pBX + pW + 30;
      const P = 6, ph = (t % P) / P, stg = ph < 0.33 ? 0 : ph < 0.66 ? 1 : 2;
      const f1 = stg > 0 ? 1 : V.ease.inOut(ph / 0.33), f2 = stg > 1 ? 1 : stg < 1 ? 0 : V.ease.inOut((ph - 0.33) / 0.33), f3 = stg < 2 ? 0 : V.ease.inOut(Math.min(1, (ph - 0.66) / 0.26));
      const cell = (x, y, ww, hh, col, a) => { V.rr(g, x, y, ww, hh, 3); g.fillStyle = V.alpha(col, a); g.fill(); };
      // x (input, the same on both nodes) above gate/up
      cell(gX, gY - 13, gW, 6, C.ink, stg === 0 ? 0.55 : 0.25);
      V.text(g, 'x', gX - 5, gY - 7, { size: 11, font: 'mono', weight: 600, align: 'right', color: C.ink });
      // gate/up [2048 × 4096]: top half on node A, bottom half on node B (split by outputs)
      cell(gX, gY, gW, gH / 2 - 1, cA, 0.5); cell(gX, gY + gH / 2 + 1, gW, gH / 2 - 1, cB, 0.5);
      if (stg === 0) { g.fillStyle = V.alpha(C.ink, 0.55); const r = f1 * (gH / 2 - 4); g.fillRect(gX, gY + r, gW, 2.5); g.fillRect(gX, gY + gH / 2 + 1 + r, gW, 2.5); }
      // h [2048]: each node fills its half
      cell(hX, hY, hW, hH, C.muted, 0.12);
      cell(hX, hY, hW, (hH / 2 - 1) * f1, cA, 0.85); cell(hX, hY + hH / 2 + 1, hW, (hH / 2 - 1) * f1, cB, 0.85);
      V.text(g, 'h', hX + hW / 2, hY - 7, { size: 11, font: 'mono', weight: 600, align: 'center', color: C.ink });
      V.arrow(g, gX + gW + 3, gY + gH / 2, hX - 3, hY + hH / 2, { color: V.alpha(C.muted, 0.6), lw: 1.2, head: 5 });
      // down [4096 × 2048]: left columns on node A, right ones on node B (split by inputs)
      cell(dX, dY, dW / 2 - 1, dH, cA, 0.5); cell(dX + dW / 2 + 1, dY, dW / 2 - 1, dH, cB, 0.5);
      if (stg === 1) { g.fillStyle = V.alpha(C.ink, 0.55); const c = f2 * (dW / 2 - 4); g.fillRect(dX + c, dY, 2.5, dH); g.fillRect(dX + dW / 2 + 1 + c, dY, 2.5, dH); }
      V.arrow(g, hX + hW + 3, hY + hH / 2, dX - 3, dY + dH / 2, { color: V.alpha(C.muted, 0.6), lw: 1.2, head: 5 });
      // partial sums y_A and y_B [4096]
      cell(pAX, dY, pW, dH, C.muted, 0.12); cell(pBX, dY, pW, dH, C.muted, 0.12);
      cell(pAX, dY, pW, dH * f2, cA, 0.85); cell(pBX, dY, pW, dH * f2, cB, 0.85);
      V.text(g, 'A', pAX + pW / 2, dY - 6, { size: 10.5, font: 'mono', weight: 600, align: 'center', color: cA });
      V.text(g, 'B', pBX + pW / 2, dY - 6, { size: 10.5, font: 'mono', weight: 600, align: 'center', color: cB });
      V.arrow(g, dX + dW + 3, dY + dH / 2, pAX - 3, dY + dH / 2, { color: V.alpha(C.muted, 0.6), lw: 1.2, head: 5 });
      // all-reduce: exchange over the network and add → full y on both nodes
      V.arrow(g, pBX + pW + 4, dY + dH / 2, yX - 4, dY + dH / 2, { color: stg === 2 ? C.d : V.alpha(C.d, 0.5), lw: stg === 2 ? 2 : 1.3, head: 6 });
      if (stg === 2) for (let q = 0; q < 5; q++) { const k = (f3 * 1.4 + q / 5) % 1; const yy = dY + (q + 0.5) * dH / 5; V.dot(g, V.lerp(pAX + pW / 2, pBX + pW / 2, q % 2 ? k : 1 - k), yy, 2.4, C.d); }
      cell(yX, dY, pW, dH, C.muted, 0.12);
      for (let q = 0; q < 8; q++) { const hh = dH / 8; if (q / 8 < f3) cell(yX, dY + q * hh + 0.5, pW, hh - 1, q % 2 ? cB : cA, 0.85); }
      V.text(g, 'y', yX + pW / 2, dY - 6, { size: 11, font: 'mono', weight: 600, align: 'center', color: C.ink });
      // labels
      const lab = (x, y0, a) => a.forEach(([s, c, wgt], i) => V.text(g, s, x, y0 + i * 13, { size: i ? 10.5 : 11, font: i === 1 ? 'mono' : 'sans', weight: wgt || 400, align: 'center', color: c }));
      lab(gX + gW / 2, gY + gH + 16, [['gate, up', C.ink, 600], ['[2048 × 4096]', C.muted], ['split by outputs', C.muted]]);
      lab(dX + dW / 2, dY + dH + 16, [['down', C.ink, 600], ['[4096 × 2048]', C.muted], ['split by inputs', C.muted]]);
      V.text(g, 'all-reduce', (pAX + yX + pW) / 2, dY + dH + 16, { size: 11, weight: 600, align: 'center', color: C.d });
      V.text(g, 'y = A + B', yX + pW / 2, dY + dH + 29, { size: 10.5, font: 'mono', align: 'center', color: C.muted });
      wrapText(g, ST4[stg], 6, dY + dH + 66, w - 12, 15, { size: 11.5, color: stg === 2 ? C.d : C.ink });
      [[cA, 'Node A'], [cB, 'Node B']].forEach(([c, l], i) => { const lx = 6 + i * 70; V.rr(g, lx, h - 18, 12, 10, 2); g.fillStyle = c; g.fill(); V.text(g, l, lx + 17, h - 9, { size: 11, color: C.muted }); });
    });

    /* ================================================================
       Figure 5 — the recipes side by side (real layer stacks)
       ================================================================ */
    const f5 = root.querySelector('#fig-lado');
    const ctl5 = UI.controls(f5);
    const S5 = { mode: 'mix', mimo: true, spec: false };
    const mk = (n, fn) => Array.from({ length: n }, (_, i) => fn(i));
    const GLM_TAPS = [5, 14, 24, 33, 42], DS_TAPS = [40, 41, 42], MIMO_GA = [0, 5, 11, 17, 23, 29, 35, 41, 47], MIMO_TAPS = [0, 11, 23, 35, 47];
    const ROWS5 = [
      { k: 'glm', name: 'GLM-5.3-Flash', sub: '45 layers + MTP',
        L: mk(45, (i) => { const sp = i % 4 === 3; return { mix: sp ? 'sparse' : 'lin', ffn: i < 3 ? 'dense' : 'moe', kv: sp ? 'grow' : 'state', kvw: sp ? 689 : 0, tap: GLM_TAPS.includes(i), d: sp ? 'MLA without RoPE + DSA indexer (top-2048)' : 'KDA — linear attention, 32×128×128 state per GPU', f: i < 3 ? 'dense SwiGLU MLP (12288)' : 'MoE: 288 experts, top-8 + 1 shared' }; }),
        spec: [{ n: 'DFlash2', c: 5, d: 'DFlash2: 5 layers, block drafting, k=7' }],
        kvB: 7579, kvf: '11 × (656 + 33 for the indexer)', fix: 'KDA state ≈ 76 MiB per conversation', sum: '34 linear · 11 sparse · 3 dense + 42 MoE' },
      { k: 'qwen', name: 'Qwen3.8-Flash-Next', sub: '48 layers + MTP',
        L: mk(48, (i) => { const sp = (i + 1) % 4 === 0; return { mix: sp ? 'sparse' : 'lin', ffn: 'moe', kv: sp ? 'grow' : 'state', kvw: sp ? 576 : 0, ple: i === 1, d: sp ? 'QSA — sparse attention (≤ 2048 tokens)' : 'Gated DeltaNet — linear, 128×128 state per head', f: 'MoE: 512 experts, top-10 + 1 shared' + (i === 1 ? ' · + N-gram embedding' : '') }; }),
        spec: [{ n: 'MTP', c: 1, d: 'MTP: 1 layer (QSA + MoE), k=3' }],
        kvB: 7488, kvf: '13 × (512 + ~64 for the indexer, estimated)', fix: 'GDN state ≈ 27 MiB (bf16) per conversation · measured pool: 9,418 eff. B/token/GPU', sum: '36 linear · 12 sparse · 48 MoE · N-gram at layer index 1' },
      { k: 'ds', name: 'DeepSeek-V4-Flash', sub: '43 layers + DSpark',
        L: mk(43, (i) => { const ty = i < 2 ? 'swa' : i === 2 || i % 2 === 0 ? 'sparse' : 'hca'; return { mix: ty, ffn: i < 3 ? 'hash' : 'moe', kv: ty === 'swa' ? 'fixed' : 'grow', kvw: ty === 'sparse' ? 179 : ty === 'hca' ? 4.6 : 0, tap: DS_TAPS.includes(i), d: ty === 'swa' ? 'pure SWA (window of 128)' : ty === 'sparse' ? 'CSA (C4A): KV ÷ 4 + top-512 indexer + window of 128' : 'HCA (C128A): KV ÷ 128, dense + window of 128', f: i < 3 ? 'MoE with hash routing (token id)' : 'MoE: 256 experts, top-6 + 1 shared' }; }),
        spec: [{ n: 'DSpark', c: 3, d: 'DSpark: 3 stages (SWA + MoE) on top of layers 40–42, k=6' }],
        kvB: 3850, kvf: '21 × (146 + 33 for the indexer) + 20 × 4.6', fix: 'SWA windows ≈ 3.4 MB per conversation', sum: '2 SWA · 21 CSA · 20 HCA · 43 MoE (3 hashed)' },
      { k: 'mimo', name: 'MiMo-V2.6-Flash', sub: '48 layers · recipe from scratch', opt: true,
        L: mk(48, (i) => { const ga = MIMO_GA.includes(i); return { mix: ga ? 'full' : 'swa', ffn: i === 0 ? 'dense' : 'moe', kv: ga ? 'grow' : 'fixed', kvw: ga ? 640 : 0, tap: MIMO_TAPS.includes(i), d: ga ? 'global attention: 4 KV heads, K=192 / V=128' : '128-token SWA (8 KV heads, with sinks)', f: i === 0 ? 'dense FFN' : 'MoE: 256 experts, top-8, no shared expert' }; }),
        spec: [{ n: 'MTP', c: 3, d: 'MTP: 3 layers' }, { n: 'DFlash', c: 5, d: 'DFlash: 5 SWA layers, block of 8' }],
        kvB: 5760, kvf: '9 × 640', fix: 'SWA windows ≈ 10–13 MB per conversation', sum: '39 SWA · 9 global · 1 dense + 47 MoE' },
    ];
    wrapCtl(UI.seg(ctl5, { label: 'Color by', options: [{ v: 'mix', label: 'token mixer' }, { v: 'ffn', label: 'FFN' }, { v: 'kv', label: 'KV per token' }], value: 'mix', onChange: (v) => { S5.mode = v; } }));
    UI.toggle(ctl5, { label: 'show MiMo', value: true, onChange: (v) => { S5.mimo = v; } });
    UI.toggle(ctl5, { label: 'drafters and taps', value: false, onChange: (v) => { S5.spec = v; } });
    const stage5 = f5.querySelector('.la-fig-stage');
    const fitH5 = () => { const w = stage5.clientWidth || 800; stage5.style.height = (w < 700 ? 540 : 440) + 'px'; };
    fitH5();
    const MIXC = (C) => ({ lin: C.a, sparse: C.c, hca: C.d, swa: C.b, full: C.f });
    const FFNC = (C) => ({ dense: C.b, moe: C.d, hash: C.c });
    const KVC = (C) => ({ grow: C.c, state: C.a, fixed: C.b });
    V.fig(f5, (g, w, h, t, dt, S, H) => {
      const C = V.C();
      const rows = ROWS5.filter((r) => !r.opt || S5.mimo);
      const wide = w >= 700;
      const labW = wide ? 158 : 0;
      const xL = 14 + labW, xR = w - 14;
      const specN = (r) => (S5.spec ? r.spec.reduce((a, s) => a + s.c + 1, 0) : 0);
      const maxN = Math.max(...rows.map((r) => r.L.length + specN(r)));
      const cw = (xR - xL) / maxN;
      const items = S5.mode === 'mix' ? [[C.a, 'linear/recurrent (KDA, GDN)'], [C.c, 'sparse global (DSA, QSA, CSA)'], [C.d, 'dense compressed (HCA ÷128)'], [C.b, 'sliding window (SWA)'], [C.f, 'full global (MiMo)']]
        : S5.mode === 'ffn' ? [[C.b, 'dense FFN'], [C.d, 'MoE (learned router)'], [C.c, 'MoE with hash routing']]
          : [[C.c, 'KV grows with the context (height ∝ bytes/layer)'], [C.a, 'fixed recurrent state'], [C.b, 'fixed window (last 128 tokens)']];
      if (S5.spec) items.push([C.ink, 'drafter (dashed) · ▼ layer it reads']);
      const cf = (t * 0.1) % 1; // cyclic context 0 → 1M
      const ctxT = Math.round(cf * 1000000);
      const ctxS = `context ≈ ${nf(ctxT / 1000, 0)}k tokens`;
      // legend layout (line wrapping) before the rows
      const leg = []; let lx = 14, lyk = 0;
      if (S5.mode === 'kv') { leg.push({ txt: ctxS, x: lx, k: 0 }); lx += tw(g, ctxS, 11.5, 600, 'mono') + 18; }
      items.forEach(([c, l]) => { const wl = tw(g, l, 11.5) + 30; if (lx + wl > w - 8) { lx = 14; lyk++; } leg.push({ c, l, x: lx, k: lyk }); lx += wl; });
      const legendH = 22 + lyk * 18;
      const topPad = S5.spec && wide ? 16 : 0; // room for the tap arcs of the 1st row
      const rowH = (h - 12 - legendH - topPad) / rows.length;
      const colOf = (L) => (S5.mode === 'mix' ? MIXC(C)[L.mix] : S5.mode === 'ffn' ? FFNC(C)[L.ffn] : KVC(C)[L.kv]);
      let hover = null;
      const pt = H.pointer;
      rows.forEach((r, ri) => {
        const y0 = 8 + topPad + ri * rowH;
        const rc = recCol(C, r.k);
        const sy = wide ? y0 + 10 : y0 + 28, sh = Math.min(30, rowH * 0.33);
        if (wide) {
          V.text(g, r.name, 14, sy + 12, { size: 13, weight: 600, color: rc, maxW: labW - 6 });
          V.text(g, r.sub, 14, sy + 28, { size: 11, color: C.muted, maxW: labW - 6 });
        } else {
          V.text(g, r.name, 14, y0 + 14, { size: 12.5, weight: 600, color: rc });
          const nw = tw(g, r.name, 12.5, 600);
          V.text(g, '  ' + r.sub, 14 + nw, y0 + 14, { size: 11, color: C.muted, maxW: w - 28 - nw });
        }
        const n = r.L.length + specN(r);
        const tokP = ((t * 0.085 + ri * 0.23) % 1) * n;
        // target model layers
        r.L.forEach((L, i) => {
          const x = xL + i * cw, glow = Math.max(0, 1 - Math.abs(i + 0.5 - tokP) / 1.6);
          V.rr(g, x, sy, Math.max(1.5, cw - Math.min(1.5, cw * 0.22)), sh, Math.min(3, cw / 3));
          g.fillStyle = V.alpha(colOf(L), 0.5 + 0.5 * glow); g.fill();
          if (L.ple && S5.mode !== 'kv') { V.text(g, '◆', x + cw / 2, sy + sh + 11, { size: 10, align: 'center', color: C.d }); if (wide) V.text(g, 'N-gram', x + cw / 2 + 8, sy + sh + 11, { size: 10, color: C.d }); }
          if (S5.spec && L.tap) { g.beginPath(); g.moveTo(x + cw / 2 - 4, sy - 7); g.lineTo(x + cw / 2 + 4, sy - 7); g.lineTo(x + cw / 2, sy - 1); g.closePath(); g.fillStyle = C.ink; g.fill(); }
          if (S5.mode === 'kv') {
            const kvTop = sy + sh + 3, kvH = Math.max(8, Math.min(24, rowH - sh - (wide ? 46 : 56)));
            if (L.kv === 'grow') { const hh = Math.max(1, kvH * cf * (L.kvw / 689)); V.rr(g, x + 0.5, kvTop, Math.max(1.2, cw - 2), hh, 1.5); g.fillStyle = V.alpha(C.c, 0.8); g.fill(); }
            else { V.rr(g, x + 0.5, kvTop, Math.max(1.2, cw - 2), 3, 1); g.fillStyle = V.alpha(L.kv === 'state' ? C.a : C.b, 0.8); g.fill(); }
          }
          if (pt.inside && pt.x >= x && pt.x < x + cw && pt.y >= sy - 8 && pt.y <= sy + sh + 8) hover = { r, i, L };
        });
        // drafters
        if (S5.spec) {
          let k = r.L.length + 1;
          const specLabels = [];
          r.spec.forEach((s) => {
            const gx0 = xL + k * cw;
            for (let q = 0; q < s.c; q++) {
              const x = gx0 + q * cw, glow = Math.max(0, 1 - Math.abs(k + q + 0.5 - tokP) / 1.6);
              V.rr(g, x, sy + 3, Math.max(1.5, cw - Math.min(1.5, cw * 0.22)), sh - 6, Math.min(3, cw / 3));
              g.fillStyle = V.alpha(C.ink, 0.12 + 0.4 * glow); g.fill();
              g.setLineDash([3, 2]); g.strokeStyle = V.alpha(C.ink, 0.85); g.lineWidth = 1.2; g.stroke(); g.setLineDash([]);
              if (pt.inside && pt.x >= x && pt.x < x + cw && pt.y >= sy - 8 && pt.y <= sy + sh + 8) hover = { r, spec: s };
            }
            if (wide) specLabels.push([s.n, gx0 + (s.c * cw) / 2 + 2, sy - 4]);
            else if (S5.mode !== 'kv') specLabels.push([s.n, gx0 + (s.c * cw) / 2, sy + sh + 11]);
            // taps: arcs from the tapped layers to the drafter
            if (wide && s.n !== 'MTP') r.L.forEach((L, i) => {
              if (!L.tap) return;
              const x1 = xL + i * cw + cw / 2, x2 = gx0 + 2; // lands on the block's left edge, away from the label
              g.beginPath(); g.moveTo(x1, sy - 8); g.quadraticCurveTo((x1 + x2) / 2, sy - 8 - Math.min(16, rowH * 0.18), x2, sy - 2);
              g.strokeStyle = V.alpha(C.ink, 0.4); g.lineWidth = 1; g.stroke();
            });
            k += s.c + 1;
          });
          // labels last, with a halo, so no arc runs over them
          let prevR = -1e9;
          specLabels.forEach(([n, x, y]) => {
            const wl = tw(g, n, 10, 600), cx2 = V.clamp(x, wl / 2 + 2, w - wl / 2 - 2);
            const yy = cx2 - wl / 2 < prevR + 4 ? y + (y < sy ? -11 : 11) : y; // neighboring labels don't touch
            halo(g, n, cx2, yy, { size: 10, weight: 600, align: 'center', color: C.ink });
            prevR = cx2 + wl / 2;
          });
        }
        // token dot
        const tx = xL + tokP * cw;
        V.glow(g, tx, sy + sh / 2, 12, C.ink, 0.6); V.dot(g, tx, sy + sh / 2, 2.6, C.ink);
        // summary line
        const ly = sy + sh + (S5.mode === 'kv' ? Math.max(8, Math.min(24, rowH - sh - (wide ? 46 : 56))) + 16 : 26);
        if (S5.mode === 'kv') {
          const bytes = r.kvB * ctxT;
          V.text(g, `growing KV: ${nf(r.kvB)} B/token/GPU${wide ? ' (' + r.kvf + ')' : ''} → ${nf(ctxT / 1000, 0)}k tokens = ${nf(bytes / GiB, 2)} GiB`, xL, ly - 6, { size: 11, font: 'mono', color: C.ink, maxW: xR - xL });
          V.text(g, `fixed: ${r.fix}`, xL, ly + 8, { size: 11, color: C.muted, maxW: xR - xL });
        } else {
          V.text(g, r.sum, xL, ly, { size: 11, color: C.muted, maxW: xR - xL });
        }
      });
      // legend
      const ly0 = h - legendH + 14;
      leg.forEach((it) => {
        const ly = ly0 + it.k * 18;
        if (it.txt) { V.text(g, it.txt, it.x, ly, { size: 11.5, font: 'mono', weight: 600, color: C.ink }); return; }
        V.rr(g, it.x, ly - 9, 12, 10, 3); g.fillStyle = it.c; g.fill();
        V.text(g, it.l, it.x + 17, ly, { size: 11.5, color: C.muted });
      });
      if (hover) {
        const lines = hover.spec ? [`${hover.r.name} · drafter`, hover.spec.d] : [`${hover.r.name} · layer ${hover.i}`, hover.L.d, hover.L.f].concat(hover.L.tap ? ['▼ read by the drafter'] : []);
        tip(g, lines, pt.x, pt.y, w, h);
      }
    });

    /* ================================================================
       Figure 6 — measured speeds, with receipts
       ================================================================ */
    const f6 = root.querySelector('#fig-medido');
    const stage6 = f6.querySelector('.la-fig-stage');
    const ctl6 = UI.controls(f6);
    const S6 = { mode: 'dec', mimo: false, tf: true, t0: 0 };
    const D6 = {
      dec: [
        { k: 'glm', v: 62.9, l: 'GLM · structured/code · DFlash2 k=7', d: '2026-08-28 · sparkDash · T=0' },
        { k: 'glm', v: 36.1, l: 'GLM · prose · adaptive k + dense FP8', d: '2026-09-17 · sparkDash' },
        { k: 'glm', v: 65.1, l: 'GLM · structured · DFlash2 k=7 (lab)', d: '2026-08-30 · bench_decode · median 5×400' },
        { k: 'glm', v: 27.1, l: 'GLM · prose · DFlash2 k=7 (lab)', d: '2026-08-30 · bench_decode · same run' },
        { k: 'glm', v: 24.6, l: 'GLM · MTP k=2 rollback', d: '2026-08-30 · bench_decode' },
        { k: 'tf', v: 114.7, l: 'GLM·TF · structured · TensorFold', d: 'v1.3.2 · sparkDash · T=0 · clock ≤ 2,200 MHz', tf: true },
        { k: 'tf', v: 60.4, l: 'GLM·TF · prose · TensorFold', d: 'v1.3.2 · sparkDash · T=0 · clock ≤ 2,200 MHz', tf: true },
        { k: 'qwen', v: 76.0, l: 'Qwen · structured · vLLM 0.30 lane', d: '2026-09-27 · KV BF16' },
        { k: 'qwen', v: 59.2, l: 'Qwen · prose · vLLM 0.30 lane', d: '2026-09-27 · KV BF16' },
        { k: 'qwen', v: 52.1, l: 'Qwen · MTP=3 · day-0 lane', d: '2026-09-05 · greedy' },
        { k: 'qwen', v: 24.5, l: 'Qwen · no speculation', d: '2026-09-05 · greedy' },
        { k: 'ds', v: 75.4, l: 'DS 0731 · 256-token prompt · MTP-5', d: '2026-08-01' },
        { k: 'ds', v: 56, l: 'DS Vision-Exp · DSpark k=6', d: '2026-09-02 · median of 8 (51–67)' },
        { k: 'mimo', v: 70.5, l: 'MiMo · code · DFlash k=7', d: 'community recipe (vLLM TP=2)', opt: true },
        { k: 'mimo', v: 25.9, l: 'MiMo · prose · DFlash k=7', d: 'community recipe (vLLM TP=2)', opt: true },
      ],
      agg: [
        { k: 'glm', v: 146.5, l: 'GLM · 4 users · structured', d: '2026-08-28 · sparkDash' },
        { k: 'glm', v: 75.3, l: 'GLM · 4 users · prose', d: '2026-09-17 · sparkDash' },
        { k: 'tf', v: 227.9, l: 'GLM·TF · 4 users · structured', d: 'v1.3.2 · sparkDash · clock ≤ 2,200 MHz', tf: true },
        { k: 'tf', v: 108.8, l: 'GLM·TF · 4 users · prose', d: 'v1.3.2 · sparkDash · clock ≤ 2,200 MHz', tf: true },
        { k: 'qwen', v: 313.6, l: 'Qwen · 8 users · code · 0.30 lane', d: '2026-09-27' },
        { k: 'qwen', v: 216.4, l: 'Qwen · 8 users · prose · 0.30 lane', d: '2026-09-27' },
        { k: 'qwen', v: 207.0, l: 'Qwen · 8 users · prose · day-0 lane', d: 'fp8 KV + 65k vocabulary' },
        { k: 'ds', v: 191.2, l: 'DS 0731 · 6 users · 256-token prompt', d: '2026-08-01' },
        { k: 'ds', v: 139, l: 'DS Vision-Exp · 6 users', d: '2026-09-02' },
        { k: 'mimo', v: 155.8, l: 'MiMo · 6 users', d: 'community recipe (vLLM TP=2)', opt: true },
      ],
    };
    const P6 = [
      { k: 'glm', l: 'GLM (2026-09-07)', pts: [[8221, 1492.1], [16411, 1553.7], [32797, 1428.2], [65566, 1587.0], [131101, 1561.7], [262173, 1516.8]] },
      { k: 'tf', l: 'GLM·TF (v1.3.2, clock ≤ 2,200 MHz)', tf: true, pts: [[8219, 1952.2], [16407, 1973.5], [32790, 1978.9], [65563, 1942.5], [131099, 1837.9], [262170, 1641.7], [981841, 1015]] },
      { k: 'qwen', l: 'Qwen (day-0 lane)', pts: [[8192, 2874.9], [16384, 2962.1], [32768, 2962.2], [65536, 2893.4], [131072, 2727.1]] },
      { k: 'ds', l: 'DS 0731 (2026-08-01)', pts: [[2048, 2563], [8192, 1713], [32768, 1428], [131072, 1665], [900000, 874.8]] },
      { k: 'mimo', l: 'MiMo (community)', opt: true, pts: [[2048, 1947], [32768, 1425], [131072, 937], [253952, 656]] },
    ];
    const col6 = (C, k) => (k === 'mimo' ? C.muted : recCol(C, k)); // MiMo (community) in gray in this figure: amber is the GLM·TF badge color
    const shown6 = (it) => (!it.opt || S6.mimo) && (!it.tf || S6.tf);
    const items6 = () => (S6.mode === 'pre' ? [] : D6[S6.mode].filter(shown6));
    const fitH6 = () => {
      const w = stage6.clientWidth || 700, wide = w >= 640;
      const need = S6.mode === 'pre' ? 420 : 30 + items6().length * (wide ? 36 : 52) + 30;
      stage6.style.height = Math.max(360, need) + 'px';
    };
    wrapCtl(UI.seg(ctl6, { label: 'Metric', options: [{ v: 'dec', label: '1 user' }, { v: 'agg', label: 'multiple users' }, { v: 'pre', label: 'prefill vs. prompt' }], value: 'dec', onChange: (v) => { S6.mode = v; S6.t0 = H6.t; fitH6(); } }));
    UI.toggle(ctl6, { label: 'include GLM·TF (TensorFold)', value: true, onChange: (v) => { S6.tf = v; S6.t0 = H6.t; fitH6(); } });
    UI.toggle(ctl6, { label: 'include MiMo (community)', value: false, onChange: (v) => { S6.mimo = v; S6.t0 = H6.t; fitH6(); } });
    const H6 = V.fig(f6, (g, w, h, t, dt, S, H) => {
      const C = V.C();
      const prog = V.ease.out(Math.min(1, (t - S6.t0) * 1.3));
      const pt = H.pointer;
      if (S6.mode !== 'pre') {
        const items = items6(), wide = w >= 640;
        const big = S6.mode === 'dec' && items.some((it) => it.v > 80); // TensorFold's structured number goes past 80
        const max = S6.mode === 'dec' ? (big ? 120 : 80) : 320, ticks = S6.mode === 'dec' ? (big ? [0, 30, 60, 90, 120] : [0, 20, 40, 60, 80]) : [0, 80, 160, 240, 320];
        const top = 14, rowH = wide ? 36 : 52, xL = 14, xR = w - 70;
        const yEnd = top + items.length * rowH;
        ticks.forEach((v) => { const x = xL + (v / max) * (xR - xL); V.line(g, x, top, x, yEnd, { color: C.grid, lw: 1 }); V.text(g, String(v), x, yEnd + 14, { size: 10.5, font: 'mono', align: 'center', color: C.muted }); });
        V.text(g, 'tok/s', xR + 26, yEnd + 14, { size: 10.5, font: 'mono', color: C.muted });
        let hov = null;
        items.forEach((it, i) => {
          const y = top + i * rowH, col = col6(C, it.k);
          V.text(g, it.l, xL, y + 13, { size: 12, weight: 600, color: C.ink, maxW: wide ? (xR - xL) * 0.62 : w - 28 });
          if (wide) V.text(g, it.d, xR + 56, y + 13, { size: 11, align: 'right', color: C.muted, maxW: (xR - xL) * 0.35 });
          const bw = (it.v / max) * (xR - xL) * prog;
          if (it.k === 'mimo') { g.setLineDash([4, 3]); V.box(g, xL, y + 19, Math.max(3, bw), 12, { r: 6, fill: V.alpha(col, 0.3), stroke: col, lw: 1.5 }); g.setLineDash([]); }
          else V.box(g, xL, y + 19, Math.max(3, bw), 12, { r: 6, fill: col });
          V.text(g, nf(it.v, 1), xL + bw + 6, y + 29.5, { size: 12, font: 'mono', weight: 600, color: C.ink });
          if (!wide) V.text(g, it.d, xL, y + 45, { size: 10.5, color: C.muted, maxW: w - 28 });
          if (pt.inside && pt.y >= y && pt.y < y + rowH) hov = it;
        });
        if (hov) tip(g, [hov.l, `${nf(hov.v, 1)} tok/s ${S6.mode === 'dec' ? 'per user' : 'summed over all users'}`, hov.d], pt.x, pt.y, w, h);
        return;
      }
      // prefill × prompt size
      const lines = P6.filter(shown6);
      const xL = 50, xR = w - 18, yB = h - 40, legY = 34;
      const leg6 = []; let lgx = xL + 130, lgr = 0;
      lines.forEach((L) => { const wl = tw(g, L.l, 11.5) + 28; if (lgx + wl > w - 8) { lgx = xL; lgr++; } leg6.push({ L, x: lgx, y: legY + lgr * 16 }); lgx += wl; });
      const yT = 46 + lgr * 16;
      const X = (n) => xL + ((Math.log2(n) - 10) / 10) * (xR - xL), Y = (v) => yB - (v / 3600) * (yB - yT);
      [0, 1000, 2000, 3000].forEach((v) => { V.line(g, xL, Y(v), xR, Y(v), { color: C.grid, lw: 1 }); V.text(g, nf(v), xL - 6, Y(v) + 4, { size: 10.5, font: 'mono', align: 'right', color: C.muted }); });
      [[1024, '1k'], [4096, '4k'], [16384, '16k'], [65536, '64k'], [262144, '256k'], [1048576, '1M']].forEach(([n, l]) => { V.line(g, X(n), yB, X(n), yB + 4, { color: C.muted }); V.text(g, l, X(n), yB + 17, { size: 10.5, font: 'mono', align: 'center', color: C.muted }); });
      V.text(g, 'prompt tokens (log scale) →', xR, yB + 33, { size: 10.5, align: 'right', color: C.muted });
      V.text(g, 'cold prefill, tok/s', xL, legY, { size: 11, font: 'mono', color: C.muted });
      const clipX = xL + prog * (xR - xL);
      g.save(); g.beginPath(); g.rect(0, 0, clipX + 6, h); g.clip();
      let best = null;
      lines.forEach((L) => {
        const col = col6(C, L.k);
        g.beginPath(); L.pts.forEach(([n, v], i) => (i ? g.lineTo(X(n), Y(v)) : g.moveTo(X(n), Y(v))));
        if (L.opt) g.setLineDash([7, 5]);
        g.strokeStyle = col; g.lineWidth = 2.5; g.stroke(); g.setLineDash([]);
        L.pts.forEach(([n, v]) => { V.dot(g, X(n), Y(v), 4, col); if (pt.inside) { const d = Math.hypot(pt.x - X(n), pt.y - Y(v)); if (d < 16 && (!best || d < best.d)) best = { d, L, n, v }; } });
      });
      g.restore();
      // legend at the top (positions computed before the axis)
      leg6.forEach(({ L, x: lx, y: ly }) => {
        if (L.opt) { g.setLineDash([4, 3]); V.line(g, lx, ly - 4, lx + 13, ly - 4, { color: col6(C, L.k), lw: 2.5, dash: [4, 3] }); g.setLineDash([]); }
        else { V.rr(g, lx, ly - 9, 12, 10, 3); g.fillStyle = col6(C, L.k); g.fill(); }
        V.text(g, L.l, lx + 17, ly, { size: 11.5, color: C.muted });
      });
      if (best) tip(g, [best.L.l, `prompt ≈ ${ctxLbl(best.n)} tokens`, `${nf(best.v, 1)} tok/s prefill`], pt.x, pt.y, w, h);
    });
    fitH6();

    /* ================================================================
       Figure 7 — MiMo: sliding window vs. global attention; DiffKV
       ================================================================ */
    const f7 = root.querySelector('#fig-mimo');
    const ctl7 = UI.controls(f7);
    const S7 = { lg: 17, cmp: false, panel: 'kv', t0: 0 };
    wrapCtl(UI.seg(ctl7, { label: 'Panel', options: [{ v: 'kv', label: 'KV vs. context' }, { v: 'diff', label: 'DiffKV: K ≠ V' }], value: 'kv', onChange: (v) => { S7.panel = v; S7.t0 = H7.t; } }));
    wrapCtl(UI.slider(ctl7, { label: 'Context', min: 10, max: 20, step: 1, value: 17, fmt: (v) => ctxLbl(2 ** v), onInput: (v) => { S7.lg = v; } }));
    UI.toggle(ctl7, { label: 'if all 48 were global (illustrative)', value: false, onChange: (v) => { S7.cmp = v; } });
    const ro7 = UI.readout(ctl7, 'KV per conversation / GPU', '—');
    let lastRo = '';
    const SWA_FIX = 11.5e6; // ~10–13 MB (tutorial), midpoint just for drawing
    const qv = V.randMatrix(1, 12, 31, 1)[0], kv = V.randMatrix(1, 12, 32, 1)[0], vv = V.randMatrix(1, 8, 33, 1)[0];
    const H7 = V.fig(f7, (g, w, h, t) => {
      const C = V.C();
      const ctx = 2 ** S7.lg;
      const ga = 5760 * ctx, total = ga + SWA_FIX, hyp = 55680 * ctx;
      const roTxt = (total >= GiB ? nf(total / GiB, 2) + ' GiB' : nf(total / 1e6, 0) + ' MB');
      if (roTxt !== lastRo) { ro7.set(roTxt); lastRo = roTxt; }
      // --- context tape
      const xL = 16, xR = w - 26, ry = 70, rh = 18;
      const ncell = Math.max(24, Math.min(64, Math.floor((xR - xL) / 9)));
      const cwid = (xR - xL) / ncell;
      V.text(g, `context tape: ${ctxLbl(ctx)} tokens (each cell ≈ ${tokLbl(ctx / ncell)} tokens)`, xL, 18, { size: 11.5, font: 'mono', color: C.muted, maxW: xR - xL });
      const shift = (t * 1.6) % 1;
      g.save(); g.beginPath(); g.rect(xL, ry - 2, xR - xL, rh + 4); g.clip();
      for (let c = -1; c < ncell; c++) {
        const x = xL + (c + 1 - shift) * cwid;
        V.rr(g, x + 1, ry, cwid - 2, rh, 3); g.fillStyle = V.alpha(C.muted, ((c + Math.floor(t * 1.6)) % 2 ? 0.16 : 0.26)); g.fill();
      }
      g.restore();
      const nx = xR + 10, ny = ry + rh / 2;
      V.glow(g, nx, ny, 14, C.c); V.dot(g, nx, ny, 4.5, C.c);
      V.text(g, 'new token', nx, ry - 26, { size: 10.5, align: 'right', color: C.c });
      const winW = Math.max(3, ((xR - xL) * 128) / ctx), wx = xR - winW;
      V.rr(g, wx, ry - 3, winW, rh + 6, 3); g.fillStyle = V.alpha(C.b, 0.55); g.fill(); g.strokeStyle = C.b; g.lineWidth = 1.5; g.stroke();
      // SWA: short arcs above
      for (let q = 0; q < 3; q++) { const x2 = wx + (winW * (q + 0.5)) / 3; g.beginPath(); g.moveTo(nx, ry - 2); g.quadraticCurveTo((nx + x2) / 2, ry - 16 - q * 3, x2, ry - 3); g.strokeStyle = V.alpha(C.b, 0.9); g.lineWidth = 1.3; g.stroke(); }
      V.text(g, '39 SWA layers: only the last 128 tokens', xL, ry - 10, { size: 11.5, weight: 600, color: C.b, maxW: Math.max(120, wx - xL - 60) });
      // global: arcs to the whole context, below
      const na = 14;
      for (let q = 0; q < na; q++) {
        const x2 = xL + ((q + 0.5) / na) * (xR - xL), a = 0.25 + 0.6 * Math.max(0, Math.sin(t * 2.2 - q * 0.45));
        g.beginPath(); g.moveTo(nx, ry + rh + 2); g.quadraticCurveTo((nx + x2) / 2, ry + rh + 16 + (nx - x2) * 0.09, x2, ry + rh + 3);
        g.strokeStyle = V.alpha(C.f, a); g.lineWidth = 1.2; g.stroke();
      }
      V.text(g, '9 global layers: every token', xL, ry + rh + 56, { size: 11.5, weight: 600, color: C.f });
      V.line(g, 12, 158, w - 12, 158, { color: C.grid, lw: 1 });
      const e = V.ease.out(Math.min(1, (t - S7.t0) * 2));
      g.globalAlpha = e;
      if (S7.panel === 'kv') {
        const bars = [
          { l: '9 global layers · 5,760 B × context', v: ga, c: C.f },
          { l: '39 SWA layers · ~10–13 MB fixed', v: SWA_FIX, c: C.b },
          { l: 'total per conversation (per GPU, FP8)', v: total, c: C.c },
        ];
        if (S7.cmp) bars.push({ l: 'if all 48 were global · 55,680 B × context', v: hyp, c: C.muted });
        const mx = Math.max(...bars.map((b) => b.v));
        const bx = 16, bw = w - 32, y0 = 176, step = Math.min(52, (h - y0 - 14) / bars.length);
        bars.forEach((b, i) => {
          const y = y0 + i * step;
          V.text(g, b.l, bx, y, { size: 11.5, color: C.ink, maxW: bw * 0.7 });
          const val = b.v >= GiB ? nf(b.v / GiB, 2) + ' GiB' : nf(b.v / 1e6, 1) + ' MB';
          V.text(g, val, bx + bw, y, { size: 12, font: 'mono', weight: 600, align: 'right', color: b.c === C.muted ? C.ink : b.c });
          V.box(g, bx, y + 7, bw, 13, { r: 6, fill: V.alpha(C.muted, 0.1) });
          V.box(g, bx, y + 7, Math.max(3, (b.v / mx) * bw), 13, { r: 6, fill: V.alpha(b.c, b.c === C.muted ? 0.5 : 0.85) });
        });
      } else {
        // DiffKV: q and k with 192 dims (12 blocks of 16), v with 128 (8 blocks)
        const cs = Math.max(14, Math.min(24, (w - 40) / 13)), gp = 3, bx = 16;
        const P = 5, ph = ((t - S7.t0) % P) / P, jj = Math.min(11, Math.floor((ph / 0.55) * 12)), sweep = ph < 0.55;
        const strip = (lab, vals, y, col, hi, rope) => {
          V.text(g, lab, bx, y - 6, { size: 11.5, font: 'mono', color: C.ink, maxW: w - 32 });
          vals.forEach((v, i) => {
            const x = bx + i * (cs + gp);
            V.rr(g, x, y, cs, cs, 4); g.fillStyle = V.alpha(rope && i < 4 ? C.d : col, 0.3 + 0.5 * Math.min(1, Math.abs(v))); g.fill();
            if (hi === i) { g.strokeStyle = C.ink; g.lineWidth = 2; V.rr(g, x - 1.5, y - 1.5, cs + 3, cs + 3, 5); g.stroke(); }
          });
        };
        const nar = w < 560;
        strip(nar ? 'q · 192 dims (64 with RoPE)' : 'q · 192 dims (12 blocks of 16; 64 with RoPE)', qv, 180, C.b, sweep ? jj : -1, true);
        strip(nar ? 'k · 192 dims (per token)' : 'k · 192 dims (one per past token)', kv, 222, C.a, sweep ? jj : -1, true);
        let dot = 0; for (let i = 0; i <= (sweep ? jj : 11); i++) dot += qv[i] * kv[i];
        const sx2 = bx + 12 * (cs + gp) + 10;
        if (sx2 + 90 < w) V.text(g, `q·k = ${nf(dot / Math.sqrt(12), 2)}`, sx2, 222 + cs / 2 + 4, { size: 12, font: 'mono', weight: 600, color: sweep ? C.ink : C.a });
        else V.text(g, `q·k = ${nf(dot / Math.sqrt(12), 2)}`, w - 16, 264, { size: 12, font: 'mono', weight: 600, align: 'right', color: sweep ? C.ink : C.a });
        const oPh = sweep ? 0 : Math.min(1, (ph - 0.55) / 0.35);
        strip('v · 128 dims (8 blocks)', vv, 270, C.f, -1, false);
        strip('output o = Σ softmax(q·k)·v · 128 dims', vv.map((v, i) => (i < oPh * 8 ? v * 0.8 : 0)), 314, C.c, -1, false);
        const yE = wrapText(g, 'kernels that assume head_size_v = head_size won’t do → on the GB10, Triton DiffKV', bx, h - (nar ? 46 : 30), w - 32, 15, { size: 11.5, color: C.muted });
        V.text(g, '(schematic; illustrative values)', bx, yE + 16, { size: 10.5, color: C.muted });
      }
      g.globalAlpha = 1;
    });

    /* ================================================================
       Figure 8 — the path of a question to the copilot
       ================================================================ */
    const f8 = root.querySelector('#fig-copilot');
    const ctl8 = UI.controls(f8);
    const PROV = [
      { v: 'mlx', label: 'Local MLX (Mac)', base: 'http://127.0.0.1:8080/v1', srv: 'mlx_lm.server', sub: 'on your Mac · Apple Silicon', ctx: 9000, tf: 'max_tokens', noThink: true, note: 'Empty model = uses the <code>--model</code> already loaded. CORS is open by default; the reasoning arrives in <code>delta.reasoning</code>. Nothing leaves your machine.' },
      { v: 'vllm', label: 'Your recipe (vLLM)', base: 'http://YOUR-HEAD:8888/v1', srv: 'vllm serve', sub: '2× DGX Spark · port 8888', ctx: 24000, tf: 'max_tokens', noThink: false, note: 'The same contract as the recipes. For the browser to talk to another machine, vLLM needs <code>--allowed-origins</code> (CORS); if the recipe uses <code>--api-key</code>, paste the key.' },
      { v: 'openai', label: 'OpenAI', base: 'https://api.openai.com/v1', srv: 'cloud API', sub: 'requires an API key', ctx: 24000, tf: 'max_completion_tokens', noThink: false, note: 'Needs an API key. Reasoning models use <code>max_completion_tokens</code> and may reject temperature — the copilot retries without the rejected fields.' },
      { v: 'ollama', label: 'Ollama', base: 'http://localhost:11434/v1', srv: 'Ollama', sub: 'port 11434', ctx: 9000, tf: 'max_tokens', noThink: true, note: 'Blocks the null origin of <code>file://</code>: serve the platform with <code>./serve.sh</code> or set <code>OLLAMA_ORIGINS</code>.' },
      { v: 'lmstudio', label: 'LM Studio', base: 'http://localhost:1234/v1', srv: 'LM Studio', sub: 'port 1234', ctx: 9000, tf: 'max_tokens', noThink: true, note: 'Start the server in the Developer tab and enable CORS in its settings.' },
    ];
    const S8 = { p: PROV[0], ctx: 'chapter', think: false, t0: 0 };
    wrapCtl(UI.select(ctl8, { label: 'Provider', options: PROV.map((p) => ({ v: p.v, label: p.label })), value: 'mlx', onChange: (v) => { S8.p = PROV.find((p) => p.v === v) || PROV[0]; tg8.set(!S8.p.noThink); upd8(); } }));
    wrapCtl(UI.seg(ctl8, { label: 'Context', options: [{ v: 'section', label: 'Section' }, { v: 'chapter', label: 'Chapter' }, { v: 'none', label: 'No context' }], value: 'chapter', onChange: (v) => { S8.ctx = v; upd8(); } }));
    const tg8 = UI.toggle(ctl8, { label: 'thinking on', value: false, onChange: (v) => { S8.think = v; S8.t0 = H8 ? H8.t : 0; upd8(); } });
    const code8 = LA.h('code', { style: { display: 'block', whiteSpace: 'pre-wrap', wordBreak: 'break-all', fontSize: '12px', lineHeight: '1.5' } });
    const note8 = LA.h('div', { style: { fontSize: '13px', lineHeight: '1.5', color: 'var(--text-2)', marginTop: '6px' } });
    ctl8.appendChild(LA.h('div', { style: { flexBasis: '100%', minWidth: '0' } }, code8, note8));
    function upd8() {
      const p = S8.p;
      const sys = S8.ctx === 'none' ? '"(tutor instructions, no page)"' : `"(tutor + trail + ${S8.ctx === 'section' ? 'visible section' : 'chapter'}, up to ${nf(p.ctx)} characters)"`;
      code8.textContent = `POST ${p.base}/chat/completions\n{"messages":[{"role":"system","content":${sys}},{"role":"user","content":"(your question)"}],\n "stream":true,"${p.tf}":2048,"temperature":0.4${S8.think ? '' : ',"chat_template_kwargs":{"enable_thinking":false}'}}`;
      note8.innerHTML = p.note;
    }
    let H8 = null;
    H8 = V.fig(f8, (g, w, h, t) => {
      const C = V.C();
      const p = S8.p, wide = w >= 640;
      let Pg, Cp, Sv;
      if (wide) { Pg = [14, 30, w * 0.22, h - 110]; Cp = [w * 0.3, 30, w * 0.28, h - 110]; Sv = [w * 0.72, 30, w * 0.28 - 14, h - 110]; }
      else { Pg = [14, 12, w - 28, 70]; Cp = [14, 112, w - 28, 116]; Sv = [14, 262, w - 28, 64]; }
      const P = 7, ph = ((t - S8.t0) % P) / P;
      // page
      V.box(g, Pg[0], Pg[1], Pg[2], Pg[3], { r: 12, fill: V.alpha(C.muted, 0.06), stroke: V.alpha(C.muted, 0.4), lw: 1.2 });
      V.text(g, 'Page (chapter)', Pg[0] + 12, Pg[1] + 20, { size: 12, weight: 600, color: C.ink });
      const nl = wide ? 9 : 3;
      for (let i = 0; i < nl; i++) {
        const y = Pg[1] + (wide ? 52 : 32) + i * (wide ? 18 : 11), lw = (Pg[2] - 24) * (0.55 + 0.4 * Math.abs(Math.sin(i * 1.9)));
        const on = S8.ctx === 'chapter' || (S8.ctx === 'section' && (wide ? i >= 3 && i <= 5 : i === 1));
        V.rr(g, Pg[0] + 12, y, lw, wide ? 7 : 5, 3); g.fillStyle = on ? V.alpha(C.a, 0.6) : V.alpha(C.muted, 0.22); g.fill();
      }
      const mLbl = S8.ctx === 'none' ? 'no context' : S8.ctx === 'section' ? '✦ visible section' : '✦ full text';
      if (wide) V.text(g, mLbl, Pg[0] + 12, Pg[1] + 38, { size: 11, color: S8.ctx === 'none' ? C.muted : C.a, maxW: Pg[2] - 24 });
      else V.text(g, mLbl, Pg[0] + Pg[2] - 12, Pg[1] + 20, { size: 11, align: 'right', color: S8.ctx === 'none' ? C.muted : C.a });
      // copilot
      V.box(g, Cp[0], Cp[1], Cp[2], Cp[3], { r: 12, fill: V.alpha(C.b, 0.08), stroke: C.b, lw: 1.5 });
      V.text(g, 'Copilot', Cp[0] + 12, Cp[1] + 20, { size: 12.5, weight: 600, color: C.ink });
      V.text(g, '⌘/Ctrl J', Cp[0] + Cp[2] - 12, Cp[1] + 20, { size: 11, font: 'mono', align: 'right', color: C.muted });
      const sysW = (Cp[2] - 24) * (S8.ctx === 'none' ? 0.18 : S8.ctx === 'section' ? 0.45 : 0.9) * Math.min(1, p.ctx / 24000 + 0.45);
      V.box(g, Cp[0] + 12, Cp[1] + 30, Math.max(40, sysW), 16, { r: 5, fill: V.alpha(C.a, 0.3) });
      V.text(g, 'system', Cp[0] + 18, Cp[1] + 42, { size: 10.5, font: 'mono', color: C.ink });
      V.box(g, Cp[0] + 12, Cp[1] + 50, 110, 16, { r: 5, fill: V.alpha(C.c, 0.3) });
      V.text(g, 'user: question', Cp[0] + 18, Cp[1] + 62, { size: 10.5, font: 'mono', color: C.ink });
      // answer being written
      const streamP = ph < 0.38 ? 0 : Math.min(1, (ph - 0.38) / 0.55);
      const thinkShare = S8.think ? 0.7 : 0;
      const ansY = Cp[1] + 74, ansH = Cp[3] - 82, nls = Math.max(1, Math.floor(ansH / 10));
      for (let i = 0; i < nls; i++) {
        const f = (i + 1) / nls, isThink = f <= thinkShare;
        if (streamP < f - 1 / nls) break;
        const part = Math.min(1, (streamP - (f - 1 / nls)) * nls);
        V.rr(g, Cp[0] + 12, ansY + i * 10, (Cp[2] - 24) * (0.5 + 0.45 * Math.abs(Math.sin(i * 2.3))) * part, 6, 3);
        g.fillStyle = isThink ? V.alpha(C.b, 0.55) : V.alpha(C.a, 0.8); g.fill();
      }
      // server
      V.box(g, Sv[0], Sv[1], Sv[2], Sv[3], { r: 12, fill: V.alpha(C.d, 0.08), stroke: C.d, lw: 1.5 });
      V.text(g, p.label, Sv[0] + 12, Sv[1] + 20, { size: 12.5, weight: 600, color: C.ink, maxW: Sv[2] - 24 });
      if (wide) {
        V.text(g, p.srv, Sv[0] + 12, Sv[1] + 38, { size: 11, font: 'mono', color: C.ink, maxW: Sv[2] - 24 });
        V.text(g, p.sub, Sv[0] + 12, Sv[1] + 55, { size: 11, color: C.muted, maxW: Sv[2] - 24 });
        V.text(g, p.base, Sv[0] + 12, Sv[1] + 74, { size: 11, font: 'mono', color: C.d, maxW: Sv[2] - 24 });
        V.text(g, 'GET /v1/models', Sv[0] + 12, Sv[1] + 100, { size: 10.5, font: 'mono', color: C.muted });
        V.text(g, 'POST /v1/chat/completions', Sv[0] + 12, Sv[1] + 116, { size: 10.5, font: 'mono', color: C.muted, maxW: Sv[2] - 24 });
      } else {
        V.text(g, p.srv + ' · ' + p.sub, Sv[0] + 12, Sv[1] + 37, { size: 11, color: C.muted, maxW: Sv[2] - 24 });
        V.text(g, p.base, Sv[0] + 12, Sv[1] + 54, { size: 11, font: 'mono', color: C.d, maxW: Sv[2] - 24 });
      }
      if (ph > 0.3 && ph < 0.4) { const a0 = t * 6; g.beginPath(); g.arc(Sv[0] + Sv[2] - 22, Sv[1] + 22, 8, a0, a0 + 4.2); g.strokeStyle = C.d; g.lineWidth = 2; g.stroke(); }
      // arrows and particles
      const pOut = wide ? [Pg[0] + Pg[2], Pg[1] + 70] : [w / 2, Pg[1] + Pg[3]];
      const cIn = wide ? [Cp[0], Cp[1] + 70] : [w / 2, Cp[1]];
      const cOut = wide ? [Cp[0] + Cp[2], Cp[1] + 50] : [w * 0.35, Cp[1] + Cp[3]];
      const sIn = wide ? [Sv[0], Sv[1] + 50] : [w * 0.35, Sv[1]];
      const sOut = wide ? [Sv[0], Sv[1] + Sv[3] - 40] : [w * 0.65, Sv[1]];
      const cIn2 = wide ? [Cp[0] + Cp[2], Cp[1] + Cp[3] - 40] : [w * 0.65, Cp[1] + Cp[3]];
      V.arrow(g, pOut[0] + 4, pOut[1], cIn[0] - 4, cIn[1], { color: V.alpha(C.a, S8.ctx === 'none' ? 0.2 : 0.6), lw: 1.4, head: 7 });
      V.arrow(g, cOut[0] + 4, cOut[1], sIn[0] - 4, sIn[1], { color: V.alpha(C.c, 0.7), lw: 1.4, head: 7 });
      V.arrow(g, sOut[0] - 4, sOut[1], cIn2[0] + 4, cIn2[1], { color: V.alpha(C.a, 0.7), lw: 1.4, head: 7, dash: [4, 4] });
      if (wide) {
        const mx = (cOut[0] + sIn[0]) / 2, gapW = sIn[0] - cOut[0] + 20;
        V.text(g, 'POST', mx, cOut[1] - 24, { size: 10.5, font: 'mono', weight: 600, align: 'center', color: C.c });
        V.text(g, '/chat/completions', mx, cOut[1] - 10, { size: 10.5, font: 'mono', align: 'center', color: C.c, maxW: gapW });
        V.text(g, 'SSE', mx, sOut[1] + 18, { size: 10.5, font: 'mono', weight: 600, align: 'center', color: C.a });
        V.text(g, 'data: {delta}', mx, sOut[1] + 32, { size: 10.5, font: 'mono', align: 'center', color: C.a, maxW: gapW });
      } else {
        V.text(g, 'POST', cOut[0] - 8, (cOut[1] + sIn[1]) / 2 + 4, { size: 10.5, font: 'mono', align: 'right', color: C.c });
        V.text(g, 'SSE', sOut[0] + 8, (sOut[1] + cIn2[1]) / 2 + 4, { size: 10.5, font: 'mono', color: C.a });
      }
      if (ph < 0.12 && S8.ctx !== 'none') { const k = V.ease.inOut(ph / 0.12); const q = V.along(pOut[0], pOut[1], cIn[0], cIn[1], k); V.glow(g, q[0], q[1], 10, C.a); }
      if (ph >= 0.12 && ph < 0.3) { const k = V.ease.inOut((ph - 0.12) / 0.18); const q = V.along(cOut[0], cOut[1], sIn[0], sIn[1], k); const s = S8.ctx === 'none' ? 7 : S8.ctx === 'section' ? 10 : 14; V.rr(g, q[0] - s, q[1] - s * 0.7, 2 * s, 1.4 * s, 3); g.fillStyle = V.alpha(C.c, 0.85); g.fill(); }
      if (ph >= 0.38 && ph < 0.95) {
        for (let q = 0; q < 5; q++) {
          const u = (((ph - 0.38) / 0.57) * 5 + q / 5) % 1;
          const pt = V.along(sOut[0], sOut[1], cIn2[0], cIn2[1], u);
          V.dot(g, pt[0], pt[1], 3, streamP < thinkShare ? C.b : C.a);
        }
      }
      // token budget
      const by = h - 30, bx = 14, bw = w - 28;
      V.text(g, wide ? `max_tokens = 2048 (copilot default)${S8.think ? ' · reasoning competes for the same budget (illustrative)' : ' · request without thinking'}` : `max_tokens = 2048${S8.think ? ' · reasoning vs. answer (illustr.)' : ' · no thinking'}`, bx, by - 8, { size: 11, color: C.muted, maxW: bw });
      V.box(g, bx, by, bw, 12, { r: 6, fill: V.alpha(C.muted, 0.12) });
      const used = streamP;
      if (S8.think) { V.box(g, bx, by, Math.max(2, bw * Math.min(used, thinkShare)), 12, { r: 6, fill: V.alpha(C.b, 0.75) }); if (used > thinkShare) V.box(g, bx + bw * thinkShare, by, bw * (used - thinkShare) * 0.5, 12, { r: 6, fill: V.alpha(C.a, 0.85) }); }
      else V.box(g, bx, by, Math.max(2, bw * used * 0.35), 12, { r: 6, fill: V.alpha(C.a, 0.85) });
    });
    upd8();

    /* ================================================================
       Figure 9 — the journey's subway map
       ================================================================ */
    // stations = registered chapters (id, part, short title), in trail order; the fixed list is only a fallback
    const CH9_FALLBACK = [['mapa-da-jornada', 0, 'Journey map'], ['aprender-com-dados', 1, 'Learning from data'], ['redes-neurais', 1, 'Neural networks'], ['tensores-algebra', 1, 'Tensors & algebra'], ['numeros-no-computador', 1, 'Numbers in the computer'], ['tokens-embeddings', 2, 'Tokens & embeddings'], ['atencao', 2, 'Attention'], ['bloco-transformer', 2, 'Transformer block'], ['posicao-rope', 2, 'Position & RoPE'], ['geracao-amostragem', 2, 'Generation & sampling'], ['prefill-decode-kv', 2, 'Prefill, decode & KV'], ['mixture-of-experts', 3, 'Mixture-of-Experts'], ['mla', 3, 'MLA & compact KV'], ['atencao-esparsa', 3, 'Sparse attention'], ['atencao-linear', 3, 'Linear attention & hybrids'], ['hiperconexoes', 3, 'Hyper-connections'], ['ngram-mtp', 3, 'N-gram & MTP'], ['multimodal', 3, 'Multimodal'], ['tres-modelos', 3, 'The three models'], ['abliteracao', 3, 'Abliteration'], ['quantizacao-fundamentos', 4, 'Quantization fundamentals'], ['fp8-fp4-microescala', 4, 'FP8, FP4 & microscaling'], ['hadamard-trelica-exl3', 4, 'Hadamard, trellises & EXL3'], ['kv-cache-quantizado', 4, 'Quantized KV cache'], ['checkpoints', 4, 'Checkpoints'], ['medindo-qualidade', 4, 'Measuring quality'], ['gpu-por-dentro', 5, 'Inside the GPU'], ['memoria-unificada', 5, 'Unified memory'], ['kernels', 5, 'Kernels'], ['cuda-graphs-warmup', 5, 'CUDA graphs & warmup'], ['vllm-arquitetura', 6, 'vLLM architecture'], ['paged-attention-kv', 6, 'PagedAttention & KV'], ['scheduler-batching', 6, 'Scheduler & batching'], ['prefix-caching', 6, 'Prefix caching'], ['api-tools-estruturado', 6, 'API, tools & JSON'], ['especulacao-fundamentos', 7, 'Speculation: fundamentals'], ['mtp-eagle-vocab', 7, 'MTP, EAGLE & reduced vocab'], ['dflash-dspark', 7, 'DFlash2 & DSpark'], ['paralelismos', 8, 'Parallelism'], ['nccl-rdma', 8, 'NCCL & RDMA'], ['tp3-tp4', 8, 'TP=3 & TP=4'], ['engenharia-de-receita', 9, 'Recipe engineering'], ['receita-glm', 9, 'GLM recipe'], ['receita-qwen', 9, 'Qwen recipe'], ['receita-deepseek', 9, 'DeepSeek recipe'], ['benchmarking', 9, 'Benchmarking'], ['receita-do-zero', 9, 'Recipe from scratch'], ['laboratorio', 9, 'Lab'], ['motor-exato-tensorfold', 9, 'Exact engine (TensorFold)'], ['receita-glm-tensorfold', 9, 'GLM recipe · TensorFold']];
    const CH9 = (LA.chapters && LA.chapters.length ? LA.chapters.map((c) => [c.id, c.part, c.navTitle || c.title]) : CH9_FALLBACK).filter((c) => typeof c[1] === 'number');
    const NAV9 = {}; CH9.forEach(([id, p, nm]) => { NAV9[id] = { p, nm }; });
    const NPART9 = Math.max(10, ...CH9.map((c) => c[1] + 1));
    const ROUTES = [
      { label: '--kv-cache-dtype fp8 (→ fp8_ds_mla)', r: 'glm', ids: ['numeros-no-computador', 'atencao', 'prefill-decode-kv', 'mla', 'fp8-fp4-microescala', 'kv-cache-quantizado', 'paged-attention-kv', 'receita-glm'], d: 'Why GLM stores the KV in packed FP8 (656 B per token per MLA layer): how numbers become bytes, what K and V are, why the KV grows with the context, how MLA compresses it into a latent, the FP8 format, how the KV gets quantized, how vLLM pages it — and the recipe.' },
      { label: '--speculative-config mtp, k=3', r: 'qwen', ids: ['geracao-amostragem', 'prefill-decode-kv', 'ngram-mtp', 'cuda-graphs-warmup', 'especulacao-fundamentos', 'mtp-eagle-vocab', 'receita-qwen'], d: 'Why Qwen drafts 3 tokens with its own MTP layer: the autoregressive loop, why decode is bandwidth-bound, what MTP is, the fixed-size CUDA graphs per step, the math of lossless drafting and verification, the draft’s reduced vocabulary — and the recipe.' },
      { label: '--tensor-parallel-size 2 --enable-expert-parallel', r: 'qwen', ids: ['tensores-algebra', 'bloco-transformer', 'mixture-of-experts', 'paralelismos', 'nccl-rdma', 'receita-qwen'], d: 'Why split the model across two nodes, with whole experts on each GPU: sliced matrix multiplication, where the matrices sit in a block, what an expert is, TP vs. EP, the collectives over the RDMA network — and the recipe.' },
      { label: 'EXL3 4 bpw on the routed experts', r: 'glm', ids: ['numeros-no-computador', 'quantizacao-fundamentos', 'hadamard-trelica-exl3', 'medindo-qualidade', 'kernels', 'receita-glm'], d: 'Why GLM uses 4 bits per weight in a trellis, on the experts only: bits and formats, uniform quantization and its errors, Hadamard rotations and trellises (EXL3), how to measure the damage (KLD), the kernels that unpack the weights on the fly — and the recipe.' },
      { label: '--max-model-len 1048576', r: 'ds', ids: ['posicao-rope', 'prefill-decode-kv', 'atencao-esparsa', 'kv-cache-quantizado', 'paged-attention-kv', 'scheduler-batching', 'receita-deepseek'], d: 'Why DeepSeek accepts 1 million tokens: position and YaRN, the cost of the KV in decode, sparse and compressed attention, the KV layout, the paged pool and maximum concurrency, chunked prefill — and the recipe.' },
      { label: '--moe-backend flashinfer_b12x', r: 'ds', ids: ['mixture-of-experts', 'fp8-fp4-microescala', 'gpu-por-dentro', 'kernels', 'receita-deepseek'], d: 'Why DeepSeek picks the b12x MoE backend: what the MoE computes, the experts’ MXFP4 format, what the GB10 (<code>sm_121</code>) supports, how a fused MoE kernel works (and why here it runs as W4A16) — and the recipe.' },
      { label: '--gpu-memory-utilization 0.85', r: 'glm', ids: ['prefill-decode-kv', 'memoria-unificada', 'paged-attention-kv', 'receita-glm'], d: 'Why 0.85 (and why 0.87 with a 256k prefill already took down a node): the KV cache, the unified memory the operating system also uses, how vLLM measures the budget and reserves the pool — and the recipe.' },
      { label: 'KV=fp8 · PARALLEL=4 (TensorFold)', r: 'tf', ids: ['numeros-no-computador', 'geracao-amostragem', 'prefill-decode-kv', 'kv-cache-quantizado', 'memoria-unificada', 'paged-attention-kv', 'nccl-rdma', 'motor-exato-tensorfold', 'receita-glm-tensorfold'], d: 'Why GLM on TensorFold serves up to 4 requests (v1.3.2; up to 8 since v1.5) with a 1M window each and replies with the same bits together or alone: sums whose bits change with the order, the draw that becomes a repeatable computation, the KV that grows during decode, FP8 with a power-of-two scale (528 B per token and layer), unified memory that freezes instead of failing, the pool shared among the conversations, the all-gather over the RDMA network, the exact engine — and the recipe.' },
      { label: 'A recipe from scratch (MiMo)', r: 'all', ids: ['aprender-com-dados', 'atencao', 'tres-modelos', 'checkpoints', 'kernels', 'vllm-arquitetura', 'especulacao-fundamentos', 'paralelismos', 'benchmarking', 'receita-do-zero'], d: 'The long route: from what learning is all the way to the recipe from scratch, stopping at one station of each part that MiMo-V2.6-Flash will put to the test.' },
    ];
    const f9 = root.querySelector('#fig-metro');
    const ctl9 = UI.controls(f9);
    const S9 = { route: 0, t0: 0 };
    let H9 = null;
    wrapCtl(UI.select(ctl9, { label: 'Parameter', options: ROUTES.map((R, i) => ({ v: String(i), label: R.label })), value: '0', onChange: (v) => { S9.route = +v; S9.t0 = H9 ? H9.t : 0; upd9(); } }));
    const box9 = LA.h('div', { style: { flexBasis: '100%', minWidth: '0', fontSize: '13.5px', lineHeight: '1.6', color: 'var(--text-2)' } });
    ctl9.appendChild(box9);
    function upd9() {
      const R = ROUTES[S9.route];
      box9.innerHTML = `${B(R.r)} ${R.d}<br>` + R.ids.filter((id) => NAV9[id]).map((id, k) => `<a href="#/c/${id}">${k + 1}. ${LA.esc(NAV9[id].nm)}</a>`).join(' → ');
    }
    upd9();
    H9 = V.fig(f9, (g, w, h, t, dt, S, H) => {
      const C = V.C();
      const pc = [C.a, C.e, C.b, C.c, C.d, C.a, C.b, C.f, C.e, C.c, C.a, C.b, C.c, C.d];
      const wide = w >= 700;
      const labW = wide ? 150 : 36;
      const xL = 14 + labW, xR = w - 14;
      const top = 16, rowH = (h - top - 12) / NPART9;
      const cnt = Array(NPART9).fill(0);
      CH9.forEach(([, p]) => { cnt[p]++; });
      const sp = (xR - xL) / Math.max(9, ...cnt);
      const pos = {}, k9 = Array(NPART9).fill(0);
      CH9.forEach(([id, p]) => { const k = k9[p]++; pos[id] = [xL + (k + 0.5) * sp, top + (p + 0.5) * rowH]; });
      for (let p = 0; p < NPART9; p++) {
        if (!cnt[p]) continue;
        const y = top + (p + 0.5) * rowH, part = (LA.parts || [])[p] || {};
        V.line(g, xL + 0.5 * sp, y, xL + (cnt[p] - 0.5) * sp, y, { color: V.alpha(pc[p], 0.5), lw: 4 });
        V.text(g, (part.roman || String(p)) + (wide && part.short ? ' · ' + part.short : ''), 14, y + 4, { size: 11.5, font: wide ? 'sans' : 'mono', weight: 600, color: pc[p], maxW: labW - 8 });
      }
      const R = ROUTES[S9.route], rcol = R.r === 'all' ? C.ink : recCol(C, R.r);
      const rIds = R.ids.filter((id) => pos[id]);
      // sampled path
      const P = rIds.map((id) => pos[id]);
      const pts = [P[0]];
      for (let k = 1; k < P.length; k++) {
        const a = P[k - 1], b = P[k];
        const c1 = [a[0], a[1] + (b[1] - a[1]) * 0.55], c2 = [b[0], b[1] - (b[1] - a[1]) * 0.55];
        for (let s = 1; s <= 18; s++) pts.push(a[1] === b[1] ? V.along(a[0], a[1], b[0], b[1], s / 18) : V.bez(a, c1, c2, b, s / 18));
      }
      const L = [0]; for (let i = 1; i < pts.length; i++) L.push(L[i - 1] + Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]));
      const tot = L[L.length - 1] || 1;
      const prog = Math.min(1, (t - S9.t0) * 0.7);
      const drawTo = (lw, col) => { g.beginPath(); g.moveTo(pts[0][0], pts[0][1]); for (let i = 1; i < pts.length && L[i] <= prog * tot; i++) g.lineTo(pts[i][0], pts[i][1]); g.strokeStyle = col; g.lineWidth = lw; g.lineCap = 'round'; g.lineJoin = 'round'; g.stroke(); g.lineCap = 'butt'; };
      drawTo(10, V.alpha(rcol, 0.18)); drawTo(3, rcol);
      // stations
      const onR = new Set(rIds);
      let hov = null;
      CH9.forEach(([id, p, nm]) => {
        const [x, y] = pos[id], on = onR.has(id);
        g.beginPath(); g.arc(x, y, on ? 7 : 4.5, 0, Math.PI * 2); g.fillStyle = on ? rcol : C.bg; g.fill();
        g.strokeStyle = on ? C.bg : pc[p]; g.lineWidth = on ? 2 : 2; g.stroke();
        if (H.pointer.inside && Math.hypot(H.pointer.x - x, H.pointer.y - y) < 12) hov = [id, p, nm];
      });
      // train
      if (prog >= 1) {
        const u = (((t - S9.t0) * 0.1) % 1) * tot;
        let i = 1; while (i < L.length - 1 && L[i] < u) i++;
        const k = (u - L[i - 1]) / (L[i] - L[i - 1] || 1), q = V.along(pts[i - 1][0], pts[i - 1][1], pts[i][0], pts[i][1], k);
        V.glow(g, q[0], q[1], 16, rcol); V.dot(g, q[0], q[1], 4, C.ink);
      }
      // labels of the route's stations: tries several sides and avoids other labels, the route line and the stations
      const placed = [];
      const boxHit = (bx) => placed.some((q) => bx.x < q.x + q.w + 4 && bx.x + bx.w + 4 > q.x && bx.y < q.y + q.h + 2 && bx.y + bx.h + 2 > q.y);
      const routeHit = (bx) => pts.some(([px, py]) => px > bx.x - 3 && px < bx.x + bx.w + 3 && py > bx.y - 3 && py < bx.y + bx.h + 3);
      const dotHit = (bx) => rIds.some((id) => { const [sx2, sy2] = pos[id]; return sx2 > bx.x - 8 && sx2 < bx.x + bx.w + 8 && sy2 > bx.y - 8 && sy2 < bx.y + bx.h + 8; });
      rIds.forEach((id, k) => {
        const [x, y] = pos[id];
        const s = `${k + 1}. ${NAV9[id].nm}`;
        const wl = tw(g, s, 11, 600);
        const cands = [[0, -13, 'c'], [0, 21, 'c'], [12, 4, 'l'], [-12, 4, 'r'], [0, -26, 'c'], [0, 34, 'c']];
        let best = null;
        for (let pass = 0; pass < 3 && !best; pass++) {
          for (const [dx, dy, al] of cands) {
            let x0 = al === 'c' ? x + dx - wl / 2 : al === 'l' ? x + dx : x + dx - wl;
            x0 = V.clamp(x0, xL - 8, w - 4 - wl); // don't cover the part names on the left
            const bx = { x: x0, y: y + dy - 10, w: wl, h: 13 };
            if (bx.y < 2 || bx.y + bx.h > h - 2) continue;
            if (boxHit(bx)) continue;
            if (pass < 2 && routeHit(bx)) continue;
            if (pass < 1 && dotHit(bx)) continue;
            best = bx; break;
          }
        }
        if (!best) { const x0 = V.clamp(x - wl / 2, xL - 8, w - 4 - wl); best = { x: x0, y: y - 23, w: wl, h: 13 }; }
        placed.push(best);
        V.box(g, best.x - 3, best.y - 1, best.w + 6, best.h + 3, { r: 4, fill: V.alpha(C.panel, 0.92), stroke: V.alpha(rcol, 0.35), lw: 1 });
        V.text(g, s, best.x, best.y + 10, { size: 11, weight: 600, color: C.ink });
      });
      if (hov) tip(g, [hov[2], 'Part ' + (((LA.parts || [])[hov[1]] || {}).roman || hov[1]) + (onR.has(hov[0]) ? ' · on the route' : '')], H.pointer.x, H.pointer.y, w, h);
    });

    /* ---------- part cards ---------- */
    const box = root.querySelector('#parts-cards');
    LA.parts.forEach((p) => {
      const n = LA.chapters.filter((c) => c.part === p.n).length;
      box.appendChild(LA.h('div', { class: 'la-card', style: { borderTop: '3px solid ' + p.color } },
        LA.h('span', { class: 'ic', style: { color: p.color, fontFamily: 'var(--font-display)' } }, 'Part ' + (p.roman || p.n)),
        LA.h('h4', null, p.title), LA.h('p', null, p.desc + (n ? ` (${n}\u00a0ch.)` : ''))));
    });
  },
});

LA.chapter({
  id: 'bloco-transformer',
  part: 2,
  order: 7,
  title: 'The Transformer block and the residual stream',
  navTitle: 'Transformer block',
  subtitle: 'The piece that repeats dozens of times: the decoder-only assembly line, the residual stream, pre-norm with RMSNorm, the SwiGLU FFN with swiglu_limit, the bookkeeping of parameters and FLOPs, dense versus MoE and the tensor names of a checkpoint.',
  minutes: 50,
  recipes: ['glm', 'qwen', 'ds'],
  tags: ['decoder-only', 'residual stream', 'RMSNorm', 'pre-norm', 'SwiGLU', 'swiglu_limit', 'hidden_size', 'FLOPs', 'tensor names'],
  goals: [
    'Describe the decoder-only path (embedding → N blocks → final norm → <code>lm_head</code>) and the <b>shape contract</b> \\([T \\times d]\\) that holds it up',
    'Explain the <b>residual stream</b> as a bus: who reads, who writes, why the sum makes deep networks possible and why in TP each write becomes an all-reduce',
    'Compute RMSNorm and SwiGLU by hand, including the <code>swiglu_limit = 10</code> clamp, and say why the EXL3 and b12x overlays need to replicate it',
    'Count the parameters and FLOPs of a block, find the context from which attention dominates and read the tensor names of a real checkpoint',
  ],
  prereqs: ['redes-neurais', 'atencao'],
  summary: [
    'A decoder-only LLM is embedding → \\(N\\) blocks of identical shape → final norm → <code>lm_head</code>. Every block reads and returns \\([T \\times d]\\), with <code>hidden_size</code> \\(d\\) = 4096 in GLM and DeepSeek (per mHC stream) and 2560 in Qwen (per Gated Residual branch).',
    'The <b>residual stream</b> is a sum: \\(\\mathbf h^{(N)} = \\mathbf h^{(0)} + \\sum(\\mathbf a^{(\\ell)} + \\mathbf f^{(\\ell)})\\). The readers (q/k/v, gate/up) receive normalized copies; the writers (embedding, <code>o_proj</code>, <code>down_proj</code>) add. The identity path keeps the gradient alive, and in TP each row-parallel write costs an all-reduce (DeepSeek: 43 + 43 per step).',
    'Pre-norm with <b>RMSNorm</b>: \\(\\mathbf x/\\sqrt{\\overline{x^2}+\\varepsilon}\\odot\\boldsymbol\\gamma\\), scale invariant. ε = 1e-5 (GLM), 1e-6 (Qwen, DeepSeek 0731), 1e-20 (Vision-Exp). Qwen uses a zero-centered gain \\((1+\\mathbf w)\\) and replaces the block norms with the Gated Residual read. Norms stay in BF16 in the recipes.',
    'The FFN is a <b>SwiGLU</b> \\(W_{\\text{down}}(\\mathrm{SiLU}(W_{\\text{gate}}\\mathbf x)\\odot W_{\\text{up}}\\mathbf x)\\) with \\(3dI\\) parameters. vLLM fuses gate and up into <code>gate_up_proj</code>; in TP, gate/up are column-parallel and down is row-parallel.',
    '<code>swiglu_limit = 10</code> (GLM and DeepSeek) computes \\(\\mathrm{SiLU}(\\min(g,10))\\cdot\\mathrm{clamp}(u,\\pm10)\\) and bounds each neuron to ≈ 100. It is model semantics: the EXL3 overlay repeats <code>SWIGLU_LIMIT_DEFAULT = 10.0</code>, b12x receives <code>gemm1_clamp_limit</code> and the TP3 cooperative MoE only accepts 10.',
    'Bookkeeping: attention \\(\\approx 4d^2\\), classic FFN \\(8d^2\\) (2/3 of the block). FLOPs/token ≈ 2 × active + \\(4\\,h\\,d_h\\,n\\) per layer, and attention dominates from \\(n^\\ast = 6d\\) in the classic model. In MoE the FFN holds 96–98% of the parameters, few of them active.',
    'In checkpoints the weights are <code>[output, input]</code> and the names follow the module tree (<code>model.layers.{i}.mlp.down_proj.weight</code>). Each model has deviations (KDA/MLA, Gated Residual, <code>w1/w2/w3</code>) that the loader maps and that the patches use as addresses.',
  ],
  quiz: [
    { q: 'In a pre-norm block, what happens to the residual stream \\(\\mathbf h\\) when the FFN finishes?', o: ['It is replaced by the FFN output', 'The FFN output is added to it: \\(\\mathbf h + \\mathrm{FFN}(\\mathrm{Norm}(\\mathbf h))\\)', 'It is normalized and then multiplied by the FFN output', 'Nothing: the FFN only writes to the KV cache'], a: 1, e: 'The FFN reads a normalized copy and <code>down_proj</code> produces a width-\\(d\\) vector that is <b>added</b> to the residual. No sublayer replaces the stream; that is why the final vector is the sum of all the writes.' },
    { q: 'You multiply the input of an RMSNorm (with ε = 1e-6) by 1000. What happens to the output?', o: ['It gets 1000× larger', 'It gets 1000× smaller', 'It barely changes', 'It becomes zero'], a: 2, e: 'RMSNorm divides by the RMS of the vector itself, which also gets 1000× larger: \\(\\mathrm{RMSNorm}(\\alpha\\mathbf x) = \\mathrm{RMSNorm}(\\mathbf x)\\) as long as ε is negligible next to \\(\\overline{x^2}\\). Only nearly null vectors feel ε.' },
    { q: 'How many parameters does a GLM-5.3-Flash SwiGLU expert have (\\(d = 4096\\), <code>moe_intermediate_size</code> = 2048)?', o: ['8,388,608', '16,777,216', '25,165,824', '50,331,648'], a: 2, e: 'There are three matrices: gate \\([2048 \\times 4096]\\), up \\([2048 \\times 4096]\\) and down \\([4096 \\times 2048]\\), or \\(3 \\times 4096 \\times 2048 = 25{,}165{,}824\\). DeepSeek has experts of the same shape.' },
    { q: 'With <code>swiglu_limit = 10</code>, a neuron has \\(g = 15\\) and \\(u = -30\\). What is the activation?', o: ['\\(\\mathrm{SiLU}(15)\\cdot(-30) \\approx -450\\)', '\\(\\mathrm{SiLU}(10)\\cdot(-10) \\approx -100\\)', '\\(\\mathrm{SiLU}(10)\\cdot(-30) \\approx -300\\)', '0, because the clamp zeroes values outside the range'], a: 1, e: 'The gate is capped from above (\\(\\min(15, 10) = 10\\)) and the up on both sides (\\(\\mathrm{clamp}(-30, -10, 10) = -10\\)). \\(\\mathrm{SiLU}(10) \\approx 9.9995\\), so the activation is ≈ −99.995. The clamp doesn\'t zero anything: it pins the value to the edge.' },
    { q: 'In GLM\'s checkpoint, what is the shape of the tensor <code>layers.1.mlp.down_proj.weight</code>?', o: ['[12288, 4096]', '[4096, 12288]', '[4096, 4096]', '[2048, 4096]'], a: 1, e: 'Layer 1 is dense (\\(I = 12{,}288\\)). PyTorch stores weights as <code>[output, input]</code>, and <code>down_proj</code> outputs \\(d = 4096\\) and reads \\(I = 12{,}288\\). A first dimension equal to \\(d\\) is the signature of a residual writer.' },
    { q: 'Why do GLM\'s EXL3 overlay and DeepSeek\'s b12x backend need to apply the same SwiGLU clamp as the reference?', o: ['Because without the clamp the kernels don\'t compile', 'Because the clamp is part of the math the model was trained with: without it, the outputs change precisely on the tokens with extreme activations, a rare, systematic and silent error', 'Because the clamp cuts the experts\' memory in half', 'Because vLLM rejects models with swiglu_limit'], a: 1, e: 'The <code>swiglu_limit</code> is in <code>config.json</code> and the reference applies it. A kernel that forgets it diverges only when some activation goes past 10, which is rare enough to pass smoke tests. Only numerical parity or KLD reveals the problem. That is why the overlay uses <code>SWIGLU_LIMIT_DEFAULT = 10.0</code> and b12x receives <code>gemm1_clamp_limit</code>.' },
  ],
  mount(root, LA) {
    const V = LA.viz, UI = LA.ui;
    const $ = (s) => root.querySelector(s);
    const T = (g, s, x, y, o) => V.text(g, s, x, y, o || {});
    const nf = (x, d = 1) => (+x).toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d });
    const ni = (x) => Math.round(x).toLocaleString('en-US');
    const pp = (n) => (n >= 1e9 ? nf(n / 1e9, n >= 1e11 ? 0 : n >= 1e10 ? 1 : 2) + ' B' : n >= 1e6 ? nf(n / 1e6, n >= 1e8 ? 0 : 1) + ' M' : n >= 1e3 ? nf(n / 1e3, 1) + ' k' : ni(n));
    const fx = (x, d = 2) => V.fmt(x, d);
    const sig = (z) => 1 / (1 + Math.exp(-z));
    const silu = (z) => z * sig(z);
    const onTxt = (C) => C.bg; // text on a strong fill: dark background in the dark theme, light in the light one
    function textW(g, s, size, font, weight) { g.font = V.font(size || 12, font || 'sans', weight || 400); return g.measureText(String(s)).width; }
    function splitLines(g, s, maxW, o = {}) {
      g.font = V.font(o.size || 12, o.font || 'sans', o.weight || 400);
      const words = String(s).split(' ');
      const lines = [];
      let cur = '';
      for (const wd of words) {
        const tt = cur ? cur + ' ' + wd : wd;
        if (g.measureText(tt).width > maxW && cur) { lines.push(cur); cur = wd; } else cur = tt;
      }
      if (cur) lines.push(cur);
      return lines;
    }
    function wrap(g, s, x, y, maxW, lh, o = {}) {
      const ls = splitLines(g, s, maxW, o), mx = o.maxLines || 99;
      ls.slice(0, mx).forEach((l, i) => V.text(g, i === mx - 1 && ls.length > mx ? l + '…' : l, x, y + i * lh, o));
      return Math.min(ls.length, mx);
    }
    function fit(g, s, maxW, size, font, weight) { // shrinks the font until it fits (min. 11)
      let sz = size;
      while (sz > 11 && textW(g, s, sz, font, weight) > maxW) sz -= 0.5;
      return sz;
    }
    function fitH(fig, fn) {
      const st = fig.querySelector('.la-fig-stage');
      const apply = () => { const w = st.getBoundingClientRect().width; if (!w) return; const hh = Math.round(fn(w)); if (Math.abs((parseFloat(st.style.height) || 0) - hh) > 1) st.style.height = hh + 'px'; };
      apply();
      const ro = new ResizeObserver(apply); ro.observe(fig);
      LA.onCleanup(() => ro.disconnect());
      return apply;
    }
    function pathPos(pts, u) {
      let L = 0; const seg = [];
      for (let i = 1; i < pts.length; i++) { const d = Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]); seg.push(d); L += d; }
      let r = V.clamp(u, 0, 1) * L;
      for (let i = 0; i < seg.length; i++) { if (r <= seg[i] || i === seg.length - 1) { const f = seg[i] ? r / seg[i] : 0; return V.along(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1], Math.min(1, f)); } r -= seg[i]; }
      return pts[pts.length - 1];
    }
    const seg01 = (f, a, b) => V.clamp((f - a) / (b - a), 0, 1);
    // wrapper: skips frames while the canvas has no size yet (avoids negative radii in provisional layouts)
    const F = (fig, draw, opts) => V.fig(fig, (g, w, h, ...rest) => { if (w < 160 || h < 100) return; draw(g, w, h, ...rest); }, opts);
    const safe = (name, fn) => { try { fn(); } catch (e) { console.error('[bloco-transformer] ' + name, e); } };
    // KaTeX's hidden MathML is position:absolute and can escape the overflow-x of the
    // .la-eq / .la-table-wrap boxes on narrow screens. Local containment.
    root.querySelectorAll('.la-eq, .la-table-wrap, .la-params').forEach((el) => { el.style.position = 'relative'; });
    // cell with the value written in it (text always >= 11px)
    function cell(g, x, y, cw, ch, v, o = {}) {
      const C = V.C();
      const sc = o.scale || 2;
      const col = o.fill || V.cmap(V.clamp(v / sc, -1, 1), 'div');
      V.rr(g, x, y, cw, ch, Math.min(5, cw / 5)); g.fillStyle = col; g.fill();
      if (o.stroke) { g.strokeStyle = o.stroke; g.lineWidth = o.lw || 2; g.stroke(); }
      if (o.text !== false && cw >= 22) {
        let s = o.fmt ? o.fmt(v) : nf(v, Math.abs(v) >= 10 ? 0 : 1);
        if (/^-0(\.0+)?$/.test(s)) s = s.slice(1);
        const strong = Math.abs(v / sc) > 0.62;
        T(g, s, x + cw / 2, y + ch / 2 + 0.5, { size: fit(g, s, cw - 3, 12, 'mono'), font: 'mono', align: 'center', base: 'middle', color: strong ? onTxt(C) : C.ink });
      }
    }

    /* ====================================================================
       Figure 1 — The decoder-only conveyor belt
       ==================================================================== */
    safe('esteira', () => {
      const fig = $('#fig-esteira');
      fitH(fig, (w) => (w < 620 ? 620 : 400));
      const MODELS = {
        glm: { name: 'GLM-5.3-Flash', d: 4096, N: 45, V: 154880, comp: '34 KDA + 11 MLA · 3 dense + 42 MoE', extra: '+ MTP at index 45', fin: 'model.language_model.norm', res: '4 mHC streams of 4096', col: 'glm', type: (i) => (i % 4 === 3 && i < 44 ? 1 : 0), tn: ['KDA', 'MLA'] },
        qwen: { name: 'Qwen3.8-Flash-Next', d: 2560, N: 48, V: 248320, comp: '36 GDN + 12 QSA · 48 MoE', extra: '+ MTP (mtp.layers.48)', fin: 'hyper_connection_mixer', res: '4 branches of 2560 (GR)', col: 'qwen', type: (i) => ((i + 1) % 4 === 0 ? 1 : 0), tn: ['GDN', 'QSA'] },
        ds: { name: 'DeepSeek-V4-Flash', d: 4096, N: 43, V: 129280, comp: '2 SWA + 21 C4A + 20 C128A · 43 MoE', extra: '+ 3 DSpark stages', fin: 'hc_head + norm', res: '4 mHC streams of 4096', col: 'ds', type: (i) => (i < 2 ? 2 : i % 2 === 0 ? 1 : 0), tn: ['C128A', 'C4A', 'SWA'] },
        toy: { name: 'Toy Transformer', d: 8, N: 4, V: 1000, comp: '4 identical blocks (illustrative)', extra: 'illustrative values', fin: 'norm', res: '1 stream of 8', col: 'a', type: () => 0, tn: ['block'] },
      };
      const S = { m: 'glm' };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Model', options: [{ v: 'glm', label: 'GLM' }, { v: 'qwen', label: 'Qwen' }, { v: 'ds', label: 'DeepSeek' }, { v: 'toy', label: 'Toy' }], value: 'glm', onChange: (v) => { S.m = v; } });
      const toks = ['The', ' cat', ' climbed', ' the', ' roof'];
      const nextTok = [[' cat', ' dog', ' day', ' king', ' sea'], [' climbed', ' sleeps', ' eats', ' saw', ' is'], [' the', ' a', ' up', ' onto', ' over'], [' roof', ' wall', ' sofa', ' car', ' top'], ['.', ',', ' and', ' of', '!']];
      const rr = V.rng(7);
      const probs = toks.map(() => { const ls = Array.from({ length: 5 }, () => rr.normal() * 0.8); ls[0] += 2.4; return V.softmax(ls); });
      const vecs = toks.map(() => Array.from({ length: 8 }, () => rr.normal()));
      const drift = Array.from({ length: 64 }, () => Array.from({ length: 8 }, () => rr.normal() * 0.35));
      F(fig, (g, w, h, t) => {
        const C = V.C();
        const M = MODELS[S.m];
        const mc = C[M.col] || C.a;
        const P = 8; const ph = V.phase(t, P); const f = ph.f; const k = ph.i % toks.length;
        const wide = w >= 620;
        // header
        T(g, M.name, 16, 22, { size: 14, weight: 700, color: mc, font: 'display' });
        const hdr = `d = ${ni(M.d)} · N = ${M.N} · V = ${ni(M.V)}`;
        if (wide) T(g, hdr, w - 16, 22, { size: 12.5, font: 'mono', align: 'right', color: C.muted });
        else T(g, hdr, 16, 42, { size: 12, font: 'mono', color: C.muted });
        // geometry
        let chip, emb, belt, blocks, norm, head, logit;
        if (wide) {
          const by = h * 0.55;
          chip = { x: 16, y: by - 16, w: 76, h: 32 };
          emb = { x: 108, y: by - 105, w: 56, h: 210 };
          blocks = { x0: 196, x1: w - 300, y: by - 72, h: 144, vert: false };
          norm = { x: w - 282, y: by - 26, w: 44, h: 52 };
          head = { x: w - 226, y: by - 105, w: 56, h: 210 };
          logit = { x: w - 158, y: by - 70, w: 142, h: 140 };
          belt = { x0: 92, x1: w - 164, y: by };
        } else {
          const cx = w / 2;
          chip = { x: 16, y: 80, w: 76, h: 32 };
          emb = { x: cx - 50, y: 72, w: 100 + Math.min(60, w * 0.12), h: 48 };
          emb.x = Math.max(104, cx - 50);
          blocks = { y0: 146, y1: h - 240, x: cx - 95, w: 190, vert: true };
          norm = { x: cx - 36, y: h - 206, w: 72, h: 28 };
          head = { x: cx - 80, y: h - 158, w: 160, h: 38 };
          logit = { x: 16, y: h - 108, w: w - 32, h: 100 };
          belt = { x: cx, y0: 118, y1: h - 210 }; // ends right above the Norm box
        }
        // conveyor belt (strip with a moving texture)
        if (wide) {
          V.box(g, belt.x0, belt.y - 8, belt.x1 - belt.x0, 16, { r: 8, fill: V.alpha(C.muted, 0.14), stroke: V.alpha(C.muted, 0.3), lw: 1 });
          g.save(); V.rr(g, belt.x0, belt.y - 8, belt.x1 - belt.x0, 16, 8); g.clip();
          for (let x = belt.x0 - 20 + ((t * 30) % 20); x < belt.x1; x += 20) V.line(g, x, belt.y - 8, x + 8, belt.y + 8, { color: V.alpha(C.muted, 0.22), lw: 2 });
          g.restore();
        } else {
          V.box(g, belt.x - 8, belt.y0, 16, belt.y1 - belt.y0, { r: 8, fill: V.alpha(C.muted, 0.14), stroke: V.alpha(C.muted, 0.3), lw: 1 });
          g.save(); V.rr(g, belt.x - 8, belt.y0, 16, belt.y1 - belt.y0, 8); g.clip();
          for (let y = belt.y0 - 20 + ((t * 30) % 20); y < belt.y1; y += 20) V.line(g, belt.x - 8, y, belt.x + 8, y + 8, { color: V.alpha(C.muted, 0.22), lw: 2 });
          g.restore();
        }
        // phases
        const fChip = seg01(f, 0, 0.1), fEmb = seg01(f, 0.1, 0.18), fBl = seg01(f, 0.18, 0.7), fNo = seg01(f, 0.7, 0.76), fHd = seg01(f, 0.76, 0.84), fLg = seg01(f, 0.84, 0.97);
        // embedding table
        const rowsE = wide ? 14 : 6;
        const hiRow = (k * 5 + 3) % rowsE;
        const drawTable = (R, label, sub, lit, isHead) => {
          V.box(g, R.x, R.y, R.w, R.h, { r: 8, fill: V.alpha(isHead ? C.c : C.b, 0.1), stroke: V.alpha(isHead ? C.c : C.b, lit ? 0.95 : 0.45), lw: lit ? 2 : 1.2 });
          if (wide) {
            const rh = (R.h - 12) / rowsE;
            for (let i = 0; i < rowsE; i++) {
              const on = !isHead && i === hiRow && fEmb > 0 && f < 0.3;
              g.fillStyle = on ? C.d : V.alpha(isHead ? C.c : C.b, 0.14 + 0.06 * ((i * 7) % 3));
              V.rr(g, R.x + 6, R.y + 6 + i * rh, R.w - 12, rh - 2, 2); g.fill();
            }
            if (isHead && lit) { g.fillStyle = V.alpha(C.c, 0.25); V.rr(g, R.x + 6, R.y + 6, R.w - 12, R.h - 12, 3); g.fill(); }
          } else {
            const cols = 10, cw2 = (R.w - 12) / cols;
            for (let i = 0; i < cols; i++) {
              const on = !isHead && i === hiRow % cols && fEmb > 0 && f < 0.3;
              g.fillStyle = on ? C.d : V.alpha(isHead ? C.c : C.b, 0.14 + 0.06 * ((i * 7) % 3));
              V.rr(g, R.x + 6 + i * cw2, R.y + 6, cw2 - 2, R.h - 12, 2); g.fill();
            }
          }
          if (wide) {
            T(g, label, R.x + R.w / 2, R.y - 22, { size: 12, weight: 600, align: 'center', color: C.ink });
            T(g, sub, R.x + R.w / 2, R.y - 8, { size: 11, font: 'mono', align: 'center', color: C.muted });
          } else {
            T(g, label + '  ' + sub, R.x + R.w / 2, R.y - 6, { size: 11, font: 'mono', align: 'center', color: C.muted, maxW: w - R.x + 40 });
          }
        };
        drawTable(emb, 'embedding', `[${ni(M.V)} × ${ni(M.d)}]`, fEmb > 0 && f < 0.3, false);
        drawTable(head, 'lm_head', `[${ni(M.V)} × ${ni(M.d)}]`, fHd > 0 && f < 0.9, true);
        // blocks
        const nMax = blocks.vert ? Math.floor((blocks.y1 - blocks.y0) / 15) : Math.floor((blocks.x1 - blocks.x0) / 22);
        const nShow = Math.max(2, Math.min(M.N, nMax));
        const ell = M.N > nShow;
        const tc = [V.alpha(mc, 0.6), C.a, C.d];
        const aHead = Math.floor((nShow - 1) / 2);
        for (let i = 0; i < nShow; i++) {
          // shows the first and the last layers in their real order (the pattern stays visible)
          const li = !ell ? i : i < aHead ? i : M.N - (nShow - i);
          const u = (i + 0.5) / nShow;
          const passed = fBl > u - 0.5 / nShow && fBl < u + 0.5 / nShow && f < 0.71;
          const ty = M.type(li);
          const col = tc[ty] || mc;
          if (ell && i === aHead) {
            if (blocks.vert) T(g, '⋮', blocks.x + blocks.w / 2 + 20, blocks.y0 + u * (blocks.y1 - blocks.y0) + 4, { size: 14, color: C.muted, align: 'center' });
            else T(g, '…', blocks.x0 + u * (blocks.x1 - blocks.x0), blocks.y + blocks.h / 2 - 50, { size: 16, color: C.muted, align: 'center' });
            continue;
          }
          if (blocks.vert) {
            const y = blocks.y0 + u * (blocks.y1 - blocks.y0);
            V.box(g, blocks.x, y - 5, blocks.w, 10, { r: 4, fill: passed ? col : V.alpha(col, 0.28), stroke: passed ? C.ink : null, lw: 1 });
          } else {
            const x = blocks.x0 + u * (blocks.x1 - blocks.x0);
            V.box(g, x - 8, blocks.y, 16, blocks.h, { r: 5, fill: passed ? col : V.alpha(col, 0.28), stroke: passed ? C.ink : null, lw: 1 });
            if (passed) V.glow(g, x, blocks.y - 6, 12, col, 0.8);
          }
        }
        // block labels
        const bl1 = `${M.N} blocks · each [T × d] → [T × d]`;
        const bl2 = M.comp + ' · ' + M.extra;
        if (blocks.vert) {
          // background under the text so the vertical belt strip doesn't cross the labels
          const backed = (s, y, o) => { const tw = Math.min(textW(g, s, o.size, 'sans', o.weight || 400), w - 24); g.fillStyle = V.alpha(C.bg, 0.94); V.rr(g, w / 2 - tw / 2 - 6, y - o.size - 1, tw + 12, o.size + 7, 4); g.fill(); T(g, s, w / 2, y, Object.assign({ align: 'center', maxW: w - 24 }, o)); };
          backed(bl1, blocks.y0 - 12, { size: 11.5, color: C.ink, weight: 600 });
          backed(bl2, blocks.y1 + 16, { size: 11, color: C.muted });
        } else {
          T(g, bl1, (blocks.x0 + blocks.x1) / 2, blocks.y - 26, { size: 12, align: 'center', color: C.ink, weight: 600, maxW: blocks.x1 - blocks.x0 + 30 });
          T(g, bl2, (blocks.x0 + blocks.x1) / 2, blocks.y + blocks.h + 22, { size: 11.5, align: 'center', color: C.muted, maxW: blocks.x1 - blocks.x0 + 40 });
          T(g, 'residual stream: ' + M.res, (blocks.x0 + blocks.x1) / 2, blocks.y + blocks.h + 40, { size: 11, align: 'center', color: C.muted, font: 'mono', maxW: blocks.x1 - blocks.x0 + 60 });
          T(g, 'final norm: ' + M.fin, (blocks.x0 + blocks.x1) / 2, blocks.y + blocks.h + 56, { size: 11, align: 'center', color: C.muted, font: 'mono', maxW: blocks.x1 - blocks.x0 + 60 });
          // type legend (above, on the right)
          if (M.tn.length > 1) {
            let lx = blocks.x1;
            for (let q = M.tn.length - 1; q >= 0; q--) { const nm = M.tn[q]; lx -= textW(g, nm, 11) + 16; g.fillStyle = tc[q]; V.rr(g, lx, blocks.y - 51, 10, 10, 2); g.fill(); T(g, nm, lx + 14, blocks.y - 42, { size: 11, color: C.muted }); lx -= 12; }
          }
        }
        // final norm
        const nLit = fNo > 0 && f < 0.8;
        V.box(g, norm.x, norm.y, norm.w, norm.h, { r: 8, fill: V.alpha(C.e, nLit ? 0.35 : 0.12), stroke: V.alpha(C.e, nLit ? 1 : 0.5), lw: nLit ? 2 : 1.2 });
        T(g, 'Norm', norm.x + norm.w / 2, norm.y + norm.h / 2 + 4, { size: 11.5, align: 'center', weight: 600, color: C.ink });
        if (wide) T(g, 'final', norm.x + norm.w / 2, norm.y + norm.h + 14, { size: 11, align: 'center', color: C.muted });
        // token chip
        const tok = toks[k];
        const chipAlpha = f < 0.12 ? 1 : 0.35;
        V.box(g, chip.x, chip.y, chip.w, chip.h, { r: 10, fill: V.alpha(C.d, 0.18 * chipAlpha + 0.05), stroke: V.alpha(C.d, chipAlpha), lw: 1.5 });
        T(g, JSON.stringify(tok), chip.x + chip.w / 2, chip.y + chip.h / 2 + 4, { size: 12, font: 'mono', align: 'center', color: C.ink, maxW: chip.w - 6 });
        T(g, 'token', chip.x + chip.w / 2, chip.y - 8, { size: 11, align: 'center', color: C.muted });
        // the car's path
        let pts;
        if (wide) pts = [[emb.x + emb.w + 4, belt.y], [blocks.x0 - 6, belt.y], [blocks.x1 + 6, belt.y], [norm.x + norm.w / 2, belt.y], [head.x - 6, belt.y]];
        else pts = [[belt.x, emb.y + emb.h + 6], [belt.x, blocks.y0 - 4], [belt.x, blocks.y1 + 4], [belt.x, norm.y + norm.h / 2], [belt.x, head.y - 4]];
        let pos = null, vals = vecs[k].slice();
        if (f < 0.1) {
          const a = [chip.x + chip.w / 2, chip.y + chip.h / 2], b = wide ? [emb.x + emb.w / 2, emb.y + emb.h / 2] : [emb.x + emb.w / 2, emb.y + emb.h / 2];
          const p = V.along(a[0], a[1], b[0], b[1], V.ease.inOut(fChip));
          V.dot(g, p[0], p[1], 6, C.d); V.glow(g, p[0], p[1], 16, C.d, 0.7);
        } else if (f < 0.84) {
          let u;
          if (f < 0.18) u = 0.02 * fEmb;
          else if (f < 0.7) u = 0.02 + 0.66 * fBl;
          else if (f < 0.76) u = 0.68 + 0.16 * fNo;
          else u = 0.84 + 0.16 * fHd;
          pos = pathPos(pts, u);
          const nPassed = Math.floor(fBl * 60);
          for (let j = 0; j < 8; j++) { let v = vals[j]; for (let q = 0; q < Math.min(nPassed, 60); q++) v += drift[q][j] * 0.35; vals[j] = v; }
          if (f >= 0.7) { const r = Math.sqrt(vals.reduce((s, x) => s + x * x, 0) / 8) || 1; const mixF = fNo; vals = vals.map((x) => V.lerp(x, x / r, mixF)); }
        }
        if (pos) {
          const ww = 60, wh = 24;
          const x = pos[0] - ww / 2, y = pos[1] - wh / 2 - (wide ? 0 : 0);
          V.box(g, x - 2, y - 2, ww + 4, wh + 4, { r: 7, fill: C.panel, stroke: C.d, lw: 1.5 });
          for (let j = 0; j < 8; j++) { g.fillStyle = V.cmap(V.clamp(vals[j] / 2.5, -1, 1), 'div'); V.rr(g, x + 2 + j * 7, y + 3, 6, wh - 6, 2); g.fill(); }
          T(g, 'd = ' + ni(M.d), pos[0], y - 8, { size: 11, font: 'mono', align: 'center', color: C.d });
        }
        // logits
        const pr = probs[k];
        const lbl = nextTok[k];
        const nb = 5;
        const rowH = Math.min(24, (logit.h - 22) / nb);
        T(g, 'logits → probabilities', logit.x, logit.y + 2, { size: 11.5, color: C.muted, weight: 600 });
        const labW = wide ? 56 : 70;
        for (let i = 0; i < nb; i++) {
          const y = logit.y + 12 + i * rowH;
          const pv = pr[i] * fLg;
          T(g, JSON.stringify(lbl[i]), logit.x + labW - 4, y + rowH / 2 + 4, { size: 11, font: 'mono', align: 'right', color: i === 0 && fLg > 0.9 ? C.c : C.muted, maxW: labW - 4 });
          const bw = (logit.w - labW - 40) * pv;
          V.box(g, logit.x + labW, y + 3, Math.max(2, bw), rowH - 6, { r: 4, fill: i === 0 ? C.c : V.alpha(C.c, 0.4) });
          if (fLg > 0.6) T(g, nf(pr[i] * 100, 0) + '%', logit.x + labW + bw + 4, y + rowH / 2 + 4, { size: 11, font: 'mono', color: C.ink });
        }
        if (!wide) T(g, M.fin, norm.x + norm.w + 10, norm.y + norm.h / 2 + 4, { size: 11, font: 'mono', color: C.muted, maxW: w - norm.x - norm.w - 14 });

      });
    });

    /* ====================================================================
       Figure 2 — The residual bus (stepper)
       ==================================================================== */
    safe('barramento', () => {
      const fig = $('#fig-barramento');
      fitH(fig, (w) => (w < 620 ? 460 : 420));
      const r = V.rng(11);
      const D = 8;
      const mk = (s) => Array.from({ length: D }, () => r.normal() * s);
      const parts = [mk(0.9), mk(0.6), mk(0.7), mk(0.6), mk(0.7)]; // emb, a0, f0, a1, f1
      // "edited" direction (illustrative) and a1 after the orthogonal projection
      const vdir = parts[3].map((x, j) => x + 0.3 * parts[1][j]);
      const vn = Math.sqrt(vdir.reduce((s, x) => s + x * x, 0));
      const vh = vdir.map((x) => x / vn);
      const dot = parts[3].reduce((s, x, j) => s + x * vh[j], 0);
      const a1edit = parts[3].map((x, j) => x - dot * vh[j]);
      const names = ['embedding', 'attention 0', 'FFN 0', 'attention 1', 'FFN 1'];
      const steps = [
        { t: 'Embedding: the initial vector', d: 'The embedding row is the first writer. Each column at the bottom is a coordinate of the vector, and the amber color is the embedding\'s share.' },
        { t: 'Read: the norm takes a snapshot', d: 'Before attention, RMSNorm makes a normalized <b>copy</b> (the little squares inside the station). The columns don\'t change: reading takes nothing off the bus.' },
        { t: 'Write: attention adds its share', d: 'The <code>o_proj</code> returns a vector of width d, which is <b>added</b>. Each column gains a lavender segment (positive upward, negative downward). The dot marks the new value.' },
        { t: 'FFN 0: reads and writes', d: 'The same choreography: normalized copy, SwiGLU, and the <code>down_proj</code> adds the mint share.' },
        { t: 'Block 1: two more writes', d: 'Attention 1 (pink) and FFN 1 (green) stack their shares. Nothing written before is erased.' },
        { t: 'The final vector is a sum', d: '\\(\\mathbf h^{(2)} = \\mathbf h^{(0)} + \\mathbf a^{(0)} + \\mathbf f^{(0)} + \\mathbf a^{(1)} + \\mathbf f^{(1)}\\). The final norm and the <code>lm_head</code> read this sum. The number above each column is the value of the coordinate.' },
        { t: 'Editing a writer', d: 'If attention 1 is prevented from writing in a direction \\(\\mathbf v\\), as abliteration does in the <code>o_proj</code>, only the pink share changes: \\(\\mathbf a \\leftarrow \\mathbf a - (\\mathbf a\\cdot\\mathbf v)\\mathbf v\\). The other shares stay intact.' },
      ];
      const st = UI.stepper(fig, steps, { autoplay: 5200 });
      F(fig, (g, w, h, t) => {
        const C = V.C();
        const cols = [C.d, C.b, C.a, C.c, C.e];
        const i = st.i, el = st.elapsed();
        const grow = V.ease.out(V.clamp(el / 0.9, 0, 1));
        // how many shares are visible, and growth
        const vis = [1, 1, 2, 3, 5, 5, 5][i];
        const growIdx = [0, -1, 1, 2, 3, 99, 99][i];
        const growIdx2 = i === 4 ? 4 : -1;
        const READ = [[], [1], [], [2], [3, 4], [5], []][i];
        const WRITE = [[0], [], [1], [2], [3, 4], [], [3]][i];
        const narrow = w < 620;
        // road + stations
        const roadY = narrow ? 118 : 128;
        const x0 = 16, x1 = w - 16;
        V.box(g, x0, roadY - 10, x1 - x0, 20, { r: 10, fill: V.alpha(C.muted, 0.12), stroke: V.alpha(C.muted, 0.35), lw: 1 });
        for (let x = x0 + ((t * 40) % 26); x < x1 - 6; x += 26) V.line(g, x, roadY, x + 12, roadY, { color: V.alpha(C.muted, 0.35), lw: 2 });
        T(g, 'residual stream (d lanes)', x0 + 6, roadY + 26, { size: 11, color: C.muted, font: 'mono' });
        const st6 = [{ l: 'Emb', c: C.d }, { l: narrow ? 'Att0' : 'Attention 0', c: C.b }, { l: 'FFN 0', c: C.a }, { l: narrow ? 'Att1' : 'Attention 1', c: C.c }, { l: 'FFN 1', c: C.e }, { l: 'lm_head', c: C.f }];
        const sw = (x1 - x0) / st6.length;
        // current sum (for the normalized copy)
        const curSum = (nv) => Array.from({ length: D }, (_, j) => { let s = 0; for (let q = 0; q < nv; q++) s += (q === 3 && i === 6 ? V.lerp(parts[3][j], a1edit[j], grow) : parts[q][j]); return s; });
        st6.forEach((s, q) => {
          const bx = x0 + q * sw + 4, bw = sw - 8, by = narrow ? 30 : 34, bh = narrow ? 50 : 58;
          const reading = READ.includes(q);
          const writing = WRITE.includes(q);
          const on = reading || writing;
          V.box(g, bx, by, bw, bh, { r: 8, fill: V.alpha(s.c, on ? 0.26 : 0.08), stroke: V.alpha(s.c, on ? 1 : 0.4), lw: on ? 2 : 1 });
          T(g, s.l, bx + bw / 2, by + 15, { size: fit(g, s.l, bw - 6, 12, 'sans', 600), weight: 600, align: 'center', color: C.ink });
          const cx = bx + bw / 2;
          // read arrow (thin, goes up) and write arrow (thick, goes down)
          if (q > 0) V.arrow(g, cx - 8, roadY - 12, cx - 8, by + bh + 2, { color: V.alpha(C.muted, reading ? 0.9 : 0.25), lw: 1.2, head: 5 });
          if (q < 5) V.arrow(g, cx + 8, by + bh + 2, cx + 8, roadY - 12, { color: V.alpha(s.c, writing ? 0.95 : 0.25), lw: writing ? 3 : 1.5, head: 7 });
          if (reading && q > 0) {
            const vv = curSum(Math.min(q, 5));
            const rms = Math.sqrt(vv.reduce((a, x) => a + x * x, 0) / D) || 1;
            const cw = Math.min(10, (bw - 12) / D);
            for (let j = 0; j < D; j++) { g.fillStyle = V.cmap(V.clamp(vv[j] / rms / 2, -1, 1), 'div'); V.rr(g, cx - (cw * D) / 2 + j * cw, by + bh - 20, cw - 1.5, 12, 2); g.fill(); }
          }
          if (writing && el < 1.2) {
            const py = V.lerp(by + bh, roadY, V.ease.inOut(V.clamp(el / 0.9, 0, 1)));
            V.box(g, cx + 2, py - 7, 12, 12, { r: 3, fill: s.c });
          }
        });
        // stacks of contributions
        const top = narrow ? 170 : 182, bot = h - 20;
        const base = (top + bot) / 2 + 6;
        let mx = 0;
        for (let j = 0; j < D; j++) { let p = 0, n = 0; for (let q = 0; q < 5; q++) { const v = parts[q][j]; if (v > 0) p += v; else n -= v; } mx = Math.max(mx, p, n); }
        const sc = (bot - top) / 2 / (mx * 1.12);
        V.line(g, x0, base, x1, base, { color: V.alpha(C.muted, 0.5), lw: 1 });
        const cw = (x1 - x0 - 30) / D;
        for (let j = 0; j < D; j++) {
          const cx0 = x0 + 30 + j * cw + cw * 0.18, bw = cw * 0.64;
          let py = base, ny = base, sum = 0;
          for (let q = 0; q < vis; q++) {
            let v = parts[q][j];
            if (q === 3 && i === 6) v = V.lerp(parts[3][j], a1edit[j], grow);
            if (q === growIdx || q === growIdx2) v *= grow;
            sum += v;
            const hh = Math.abs(v) * sc;
            g.fillStyle = V.alpha(cols[q], q === 3 && i === 6 ? 0.95 : 0.85);
            if (v >= 0) { V.rr(g, cx0, py - hh, bw, Math.max(1, hh), 3); g.fill(); py -= hh; } else { V.rr(g, cx0, ny, bw, Math.max(1, hh), 3); g.fill(); ny += hh; }
          }
          const yv = base - sum * sc;
          V.line(g, cx0 - 4, yv, cx0 + bw + 4, yv, { color: C.ink, lw: 2 });
          V.dot(g, cx0 + bw / 2, yv, 3.5, C.ink);
          if (i >= 5) T(g, nf(sum, 1), cx0 + bw / 2, Math.min(py, yv) - 8, { size: 11, font: 'mono', align: 'center', color: C.ink });
          T(g, 'x' + j, cx0 + bw / 2, bot + 12, { size: 11, font: 'mono', align: 'center', color: C.muted });
        }
        // legend
        let lx = x0 + 30, ly = top - 4;
        for (let q = 0; q < vis; q++) {
          const lw = textW(g, names[q], 11) + 22;
          if (lx + lw > x1) { lx = x0 + 30; ly += 14; }
          g.fillStyle = cols[q]; V.rr(g, lx, ly - 9, 10, 10, 2); g.fill();
          T(g, names[q], lx + 14, ly, { size: 11, color: C.muted });
          lx += lw;
        }
      });
    });

    /* ====================================================================
       Figure 3 — Telephone game (sensitivity with/without residual)
       ==================================================================== */
    safe('telefone', () => {
      const fig = $('#fig-telefone');
      fitH(fig, (w) => (w < 620 ? 420 : 360));
      const Dm = 16, LM = 64;
      const r = V.rng(23);
      const W1 = [], W2 = [];
      for (let l = 0; l < LM; l++) {
        W1.push(Array.from({ length: Dm * Dm }, () => r.normal() / Math.sqrt(Dm)));
        W2.push(Array.from({ length: Dm * Dm }, () => r.normal() / Math.sqrt(Dm)));
      }
      const h0 = Array.from({ length: Dm }, () => r.normal());
      const t0 = Array.from({ length: Dm }, () => r.normal());
      const tn0 = Math.sqrt(t0.reduce((s, x) => s + x * x, 0));
      const MODES = ['plain', 'res', 'small'];
      const MN = { plain: 'no residual', res: 'with residual', small: 'residual + writers ÷ 8' };
      const DASH = { plain: [6, 4], res: [2, 3], small: [10, 3, 2, 3] };
      const S = { mode: 'plain', g: 0.8, L: 32, sens: {} };
      const mv = (W, x) => { const y = new Array(Dm).fill(0); for (let i = 0; i < Dm; i++) { let s = 0; for (let j = 0; j < Dm; j++) s += W[i * Dm + j] * x[j]; y[i] = s; } return y; };
      // exact directional derivative (forward mode): tt is the "nudge" propagated layer by layer
      function run(mode) {
        let h = h0.slice(), tt = t0.slice();
        const ws = mode === 'small' ? 1 / Math.sqrt(LM) : 1; // small writers: W₂ ÷ √64
        const out = [];
        for (let l = 0; l < S.L; l++) {
          if (mode === 'plain') {
            // linear network h ← g·W₁h: the tangent follows the same computation
            tt = mv(W1[l], tt).map((x) => x * S.g);
          } else {
            const ms = h.reduce((s, x) => s + x * x, 0) / Dm;
            const rr = Math.sqrt(ms + 1e-6);
            const u = h.map((x) => x / rr);
            const ut = u.reduce((s, x, i) => s + x * tt[i], 0);
            const du = tt.map((x, i) => (x - (u[i] * ut) / Dm) / rr);
            const pre = mv(W1[l], u).map((x) => x * S.g);
            const a = pre.map(Math.tanh);
            const fo = mv(W2[l], a).map((x) => x * S.g * ws);
            const dpre = mv(W1[l], du).map((x) => x * S.g);
            const da = dpre.map((x, i) => x * (1 - a[i] * a[i]));
            const df = mv(W2[l], da).map((x) => x * S.g * ws);
            h = h.map((x, i) => x + fo[i]);
            tt = tt.map((x, i) => x + df[i]);
          }
          const n = Math.sqrt(tt.reduce((s, x) => s + x * x, 0));
          out.push(Math.log10(Math.max(1e-300, n / tn0)));
        }
        return out;
      }
      function compute() { MODES.forEach((m) => { S.sens[m] = run(m); }); }
      compute();
      const pw = (x) => { const s = nf(x, 1); return '10^' + (/^-0\.0$/.test(s) ? '0.0' : s.replace('-', '−')); };
      const tag = (x) => (x < -1 ? ' (vanished)' : x > 3 ? ' (exploded)' : '');
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Architecture', options: MODES.map((m) => ({ v: m, label: MN[m] })), value: 'plain', onChange: (v) => { S.mode = v; ro.set(); } });
      UI.slider(ctl, { label: 'Weight gain', min: 0.2, max: 3, step: 0.1, value: S.g, fmt: (v) => nf(v, 1), onInput: (v) => { S.g = v; compute(); ro.set(); } });
      UI.slider(ctl, { label: 'Depth', min: 4, max: 64, step: 1, value: S.L, fmt: (v) => v + ' layers', onInput: (v) => { S.L = v; compute(); ro.set(); } });
      const rd = UI.readout(ctl, 'At the last layer', '—');
      const ro = { set: () => { const s = S.sens[S.mode]; const x = s[s.length - 1]; rd.set(pw(x) + tag(x)); } };
      ro.set();
      F(fig, (g, w, h, t) => {
        const C = V.C();
        const cur = S.sens[S.mode];
        const L = cur.length;
        const TT = { plain: 'No residual (linear network): h ← g·W h', res: 'With residual: h ← h + g·W₂·tanh(g·W₁·RMSNorm(h))', small: 'Small writers: h ← h + (g/8)·W₂·tanh(g·W₁·RMSNorm(h))' };
        const tc = { plain: C.c, res: C.d, small: C.a }[S.mode];
        const nl = wrap(g, TT[S.mode], 14, 20, w - 28, 16, { size: 12.5, weight: 600, color: tc, maxLines: 2 });
        // legend: bars = chosen architecture; thin lines = the other two
        let lx = 14, ly = 20 + nl * 16 + 6;
        MODES.forEach((m) => {
          const sel = m === S.mode;
          const x = S.sens[m][L - 1];
          const s = MN[m] + ': ' + pw(x);
          const lw = textW(g, s, 11, 'sans', sel ? 600 : 400) + 26 + 16;
          if (lx + lw > w - 10 && lx > 14) { lx = 14; ly += 16; }
          if (sel) { g.fillStyle = V.alpha(C.a, 0.85); V.rr(g, lx + 7, ly - 10, 9, 11, 2); g.fill(); }
          else V.line(g, lx, ly - 4, lx + 22, ly - 4, { color: V.alpha(C.ink, 0.7), lw: 1.6, dash: DASH[m] });
          T(g, s, lx + 26, ly, { size: 11, color: sel ? C.ink : C.muted, weight: sel ? 600 : 400 });
          lx += lw;
        });
        const x0 = 50, x1 = w - 14, y0 = ly + 14, y1 = h - 40;
        const lim = 6;
        const yOf = (v) => y0 + ((lim - V.clamp(v, -lim, lim)) / (2 * lim)) * (y1 - y0);
        // axis
        for (let v = -lim; v <= lim; v += 2) {
          const y = yOf(v);
          V.line(g, x0, y, x1, y, { color: v === 0 ? V.alpha(C.muted, 0.6) : C.grid, lw: 1 });
          T(g, v === 0 ? '1' : '10' + (v > 0 ? '' : '⁻') + ['⁰', '¹', '²', '³', '⁴', '⁵', '⁶'][Math.abs(v)], x0 - 6, y + 4, { size: 11, font: 'mono', align: 'right', color: C.muted });
        }
        g.save(); g.translate(13, (y0 + y1) / 2); g.rotate(-Math.PI / 2);
        T(g, 'sensitivity (log)', 0, 0, { size: 11, align: 'center', color: C.muted });
        g.restore();
        // healthy band
        g.fillStyle = V.alpha(C.a, 0.07); g.fillRect(x0, yOf(3), x1 - x0, yOf(-1) - yOf(3));
        const bw = (x1 - x0) / L;
        const pulse = (t * 0.6) % 1;
        const pi = Math.floor(pulse * L);
        for (let l = 0; l < L; l++) {
          const v = cur[l];
          const ok = v >= -1 && v <= 3;
          const col = ok ? C.a : v < 0 ? C.b : C.c;
          const x = x0 + l * bw + bw * 0.12;
          const ya = yOf(0), yb = yOf(v);
          g.fillStyle = V.alpha(col, l === pi ? 1 : 0.75);
          V.rr(g, x, Math.min(ya, yb), Math.max(1, bw * 0.76), Math.max(1.5, Math.abs(yb - ya)), Math.min(3, bw * 0.2)); g.fill();
          if (Math.abs(v) > lim) T(g, v > 0 ? '▲' : '▼', x + bw * 0.38, v > 0 ? y0 - 2 : y1 + 12, { size: 11, align: 'center', color: col });
        }
        // the other two architectures, as thin lines
        MODES.forEach((m) => {
          if (m === S.mode) return;
          const sv = S.sens[m];
          g.beginPath();
          sv.forEach((v, l) => { const xx = x0 + (l + 0.5) * bw, yy = yOf(v); if (l === 0) g.moveTo(xx, yy); else g.lineTo(xx, yy); });
          g.strokeStyle = V.alpha(C.ink, 0.7); g.lineWidth = 1.6; g.setLineDash(DASH[m]); g.stroke(); g.setLineDash([]);
        });
        // the "whisper" traveling through the layers
        const px = x0 + (pi + 0.5) * bw;
        const pv = cur[pi] == null ? 0 : cur[pi];
        V.glow(g, px, yOf(pv), 10, pv >= -1 && pv <= 3 ? C.a : C.c, 0.8);
        T(g, 'layer →', (x0 + x1) / 2, h - 8, { size: 11, align: 'center', color: C.muted });
        T(g, '1', x0 + bw / 2, h - 8, { size: 11, align: 'center', color: C.muted, font: 'mono' });
        T(g, String(L), x0 + (L - 0.5) * bw, h - 8, { size: 11, align: 'center', color: C.muted, font: 'mono' });
      });
    });

    /* ====================================================================
       Figure 4 — Interactive RMSNorm
       ==================================================================== */
    safe('rmsnorm', () => {
      const fig = $('#fig-rmsnorm');
      fitH(fig, (w) => (w < 620 ? 440 : 380));
      const x0 = [2.3, -0.1, 0.8, 3.4, -0.9, 0.6, -1.7, 1.4];
      const gam = [1.1, 0.92, 1.25, 0.98, 1.05, 0.88, 1.15, 0.95];
      const wz = gam.map((x) => +(x - 1).toFixed(2));
      const S = { e: 0, eps: 1e-6, ln: false, gain: 'direct' };
      const ctl = UI.controls(fig);
      UI.slider(ctl, { label: 'Input scale', min: -4, max: 3, step: 0.25, value: 0, fmt: (v) => '×10^' + nf(v, 2).replace('.00', ''), onInput: (v) => { S.e = v; H.redraw(); } });
      UI.seg(ctl, { label: 'ε', options: [{ v: 1e-20, label: '1e-20' }, { v: 1e-6, label: '1e-6' }, { v: 1e-5, label: '1e-5' }, { v: 1e-2, label: '1e-2 (exaggerated)' }], value: 1e-6, onChange: (v) => { S.eps = v; H.redraw(); } });
      UI.seg(ctl, { label: 'Norm', options: [{ v: false, label: 'RMSNorm' }, { v: true, label: 'LayerNorm' }], value: false, onChange: (v) => { S.ln = v; H.redraw(); } });
      UI.seg(ctl, { label: 'Gain', options: [{ v: 'direct', label: 'direct γ' }, { v: 'zc', label: '(1+w) correct' }, { v: 'bug', label: '(1+w) buggy' }], value: 'direct', onChange: (v) => { S.gain = v; H.redraw(); } });
      const H = F(fig, (g, w, h) => {
        const C = V.C();
        const s = Math.pow(10, S.e);
        const x = x0.map((v) => v * s);
        const mu = S.ln ? x.reduce((a, b) => a + b, 0) / 8 : 0;
        const xc = x.map((v) => v - mu);
        const ms = xc.reduce((a, b) => a + b * b, 0) / 8;
        const den = Math.sqrt(ms + S.eps);
        const nrm = xc.map((v) => v / den);
        const stored = S.gain === 'direct' ? gam : wz;
        const eff = S.gain === 'zc' ? wz.map((v) => 1 + v) : stored;
        const y = nrm.map((v, i) => v * eff[i]);
        const rmsY = Math.sqrt(y.reduce((a, b) => a + b * b, 0) / 8);
        const narrow = w < 620;
        const labW = narrow ? 0 : 118;
        const gx = narrow ? 14 : labW + 16;
        const avail = w - gx - (narrow ? 14 : 150);
        const cw = Math.min(58, (avail - 7 * 4) / 8), gap = 4;
        const rowH = narrow ? 64 : 58;
        const cell0 = (row, i) => [gx + i * (cw + gap), 30 + row * rowH];
        const rowLabel = (row, s1, s2) => {
          const [xx, yy] = cell0(row, 0);
          if (narrow) { T(g, s1 + (s2 ? '  ' + s2 : ''), 14, yy - 6, { size: 11.5, color: C.muted, maxW: w - 28 }); }
          else { T(g, s1, 14, yy + 14, { size: 12, weight: 600, color: C.ink, maxW: labW }); if (s2) T(g, s2, 14, yy + 30, { size: 11, color: C.muted, font: 'mono', maxW: labW }); }
        };
        const expo = S.e;
        const fmtScale = expo === 0 ? '' : '× 10^' + nf(expo, 2).replace('.00', '');
        // row 1: input (we show the mantissa; the scale factor goes in the label)
        rowLabel(0, 'input x', fmtScale);
        x0.forEach((v, i) => { const [cx, cy] = cell0(0, i); cell(g, cx, cy + (narrow ? 4 : 0), cw, 34, v, { scale: 3.5 }); });
        // row 2: centering (LayerNorm)
        rowLabel(1, S.ln ? 'x − μ' : 'x (not centered)', S.ln ? 'μ = ' + nf(0.725, 3) + (fmtScale ? ' ' + fmtScale : '') : 'μ not subtracted');
        xc.forEach((v, i) => { const [cx, cy] = cell0(1, i); cell(g, cx, cy + (narrow ? 4 : 0), cw, 34, v / s, { scale: 3.5 }); });
        // row 3: normalized
        rowLabel(2, S.ln ? '÷ √(σ² + ε)' : '÷ √(mean(x²) + ε)', 'normalized');
        nrm.forEach((v, i) => { const [cx, cy] = cell0(2, i); cell(g, cx, cy + (narrow ? 4 : 0), cw, 34, v, { scale: 2.2, fmt: (q) => nf(q, 2) }); });
        // row 4: gain
        rowLabel(3, S.gain === 'direct' ? '⊙ γ' : S.gain === 'zc' ? '⊙ (1 + w)' : '⊙ w  (bug!)', S.gain === 'direct' ? 'stored tensor = γ' : 'stored tensor = w');
        eff.forEach((v, i) => { const [cx, cy] = cell0(3, i); cell(g, cx, cy + (narrow ? 4 : 0), cw, 34, v, { scale: 2.2, fmt: (q) => nf(q, 2), fill: V.alpha(S.gain === 'bug' ? C.red || C.c : C.e, 0.12 + 0.3 * Math.min(1, Math.abs(v))) }); });
        // row 5: output
        rowLabel(4, 'output y', '');
        y.forEach((v, i) => { const [cx, cy] = cell0(4, i); cell(g, cx, cy + (narrow ? 4 : 0), cw, 34, v, { scale: 2.2, fmt: (q) => (Math.abs(q) < 0.005 && q !== 0 ? '≈0' : nf(q, 2)), stroke: C.a, lw: 1.5 }); });
        // readout panel
        const rmsX = Math.sqrt(x.reduce((a, b) => a + b * b, 0) / 8);
        // small values in scientific notation (e.g. 1.71·10^-3), to match the text
        const fe = (v) => { if (v === 0 || (Math.abs(v) >= 0.01 && Math.abs(v) < 1e4)) return nf(v, 2); const e = Math.floor(Math.log10(Math.abs(v))); return nf(v / Math.pow(10, e), 2) + '·10^' + e; };
        const lines = [
          ['RMS(x)', fe(rmsX)],
          [S.ln ? '√(σ² + ε)' : '√(mean(x²) + ε)', fe(den)],
          [S.ln ? 'ε / σ²' : 'ε / mean(x²)', ms > 0 ? fx(S.eps / ms, 1) : '∞'],
          ['RMS(y)', nf(rmsY, 3)],
        ];
        if (narrow) {
          const yy = 30 + 5 * rowH + 2;
          lines.forEach((ln, i) => { T(g, ln[0] + ' = ' + ln[1], 14, yy + i * 17, { size: 11.5, font: 'mono', color: i === 3 ? C.a : C.muted, maxW: w - 28 }); });
        } else {
          const px = w - 140;
          lines.forEach((ln, i) => { T(g, ln[0], px, 44 + i * 42, { size: 11, color: C.muted }); T(g, ln[1], px, 62 + i * 42, { size: 14, font: 'mono', weight: 600, color: i === 3 ? C.a : C.ink }); });
          const note = S.gain === 'bug' ? 'the loader forgot the “1 +”: output nearly zero' : (S.eps / (ms || 1e-300) > 0.2 ? 'ε dominates: the output shrinks' : 'output at the standard scale');
          wrap(g, note, px, 44 + 4 * 42 + 6, 132, 14, { size: 11, color: S.gain === 'bug' || S.eps / (ms || 1e-300) > 0.2 ? C.c : C.a });
        }
      }, { static: true });
    });

    /* ====================================================================
       Figure 5 — SwiGLU in numbers (stepper + TP)
       ==================================================================== */
    safe('swiglu', () => {
      const fig = $('#fig-swiglu');
      fitH(fig, (w) => (w < 620 ? 660 : 520));
      const r = V.rng(5);
      const d = 4, I = 6;
      const x = [0.9, -0.4, 1.3, 0.2];
      const q1 = (v) => { const z = Math.round(v * 10) / 10; return z === 0 ? 0 : z; };
      const Wg = Array.from({ length: I }, () => Array.from({ length: d }, () => q1(r.normal() * 0.9)));
      const Wu = Array.from({ length: I }, () => Array.from({ length: d }, () => q1(r.normal() * 0.9)));
      const Wd = Array.from({ length: d }, () => Array.from({ length: I }, () => q1(r.normal() * 0.7)));
      const gv = Wg.map((row) => row.reduce((s, w0, j) => s + w0 * x[j], 0));
      const uv = Wu.map((row) => row.reduce((s, w0, j) => s + w0 * x[j], 0));
      const sv = gv.map(silu);
      const pv = sv.map((s, i) => s * uv[i]);
      const yA = Wd.map((row) => row.slice(0, 3).reduce((s, w0, j) => s + w0 * pv[j], 0));
      const yB = Wd.map((row) => row.slice(3).reduce((s, w0, j) => s + w0 * pv[j + 3], 0));
      const yv = yA.map((a, i) => a + yB[i]);
      const hres = [0.4, -1.1, 0.7, 1.5];
      const hout = hres.map((a, i) => a + yv[i]);
      const S = { tp: false };
      const steps = [
        { t: 'The normalized copy comes in', d: 'The vector \\(\\mathbf x\\) (d = 4) is the output of RMSNorm. The original residual stream waits outside.' },
        { t: 'gate: g = W_gate x', d: 'Each row of \\(W_{\\text{gate}}\\) \\([6 \\times 4]\\) takes a dot product with \\(\\mathbf x\\) and becomes a score for the neuron. The lit row is the one being computed.' },
        { t: 'up: u = W_up x', d: 'Another projection, with other weights, gives each neuron\'s “message”. gate and up read the same \\(\\mathbf x\\), which is why vLLM fuses them into a single GEMM (<code>gate_up_proj</code>).' },
        { t: 'The gate: SiLU(g)', d: 'SiLU opens the gate for positive scores and almost closes it for negative ones: \\(\\mathrm{SiLU}(z) = z\\,\\sigma(z)\\).' },
        { t: 'Gate × message', d: '\\(\\mathrm{SiLU}(\\mathbf g) \\odot \\mathbf u\\), element by element. Neurons with a negative gate barely contribute, however large the up is.' },
        { t: 'down: y = W_down(…)', d: '\\(W_{\\text{down}}\\) \\([4 \\times 6]\\) closes back down to width d. It is the residual writer. With TP=2, each GPU holds 3 columns and produces a share, and an <b>all-reduce</b> adds the two.' },
        { t: 'Add to the residual', d: '\\(\\mathbf h\'\' = \\mathbf h\' + \\mathbf y\\). Done: the FFN has written its contribution onto the bus.' },
      ];
      const st = UI.stepper(fig, steps, { autoplay: 5600 });
      const ctl = UI.controls(fig);
      UI.toggle(ctl, { label: 'split across 2 GPUs (TP=2)', value: false, onChange: (v) => { S.tp = v; } });
      F(fig, (g, w, h, t) => {
        const C = V.C();
        const i = st.i, el = st.elapsed();
        const narrow = w < 620;
        const cs = narrow ? Math.min(30, (w - 40) / 9.6) : Math.max(22, Math.min(32, (w - 32 - 7 * 26) / 16, (h - 100) / 12.8));
        const gp = 2;
        const G = { A: C.a, B: C.b };
        const rowTint = (ri) => (S.tp ? V.alpha(ri < 3 ? G.A : G.B, 0.9) : null);
        const sweep = (n) => Math.min(n - 1, Math.floor(el / 0.35));
        const mat = (M, ox, oy, o = {}) => {
          const R = M.length, K = M[0].length;
          for (let a = 0; a < R; a++) for (let b = 0; b < K; b++) {
            const xx = ox + b * (cs + gp), yy = oy + a * (cs + gp);
            let stroke = null;
            if (o.hiRow != null && a === o.hiRow) stroke = C.c;
            if (o.tpRows && S.tp) stroke = stroke || (a < 3 ? G.A : G.B);
            if (o.tpCols && S.tp) stroke = stroke || (b < 3 ? G.A : G.B);
            cell(g, xx, yy, cs, cs, M[a][b], { scale: 1.6, stroke, lw: stroke ? 1.6 : 0 });
          }
          if (o.label) T(g, o.label, ox + (K * (cs + gp)) / 2, oy - 8, { size: 11.5, font: 'mono', align: 'center', color: C.muted });
          return { w: K * (cs + gp) - gp, h: R * (cs + gp) - gp };
        };
        const vec = (vs, ox, oy, o = {}) => {
          vs.forEach((v, a) => {
            const shown = o.reveal == null || a <= o.reveal;
            const stroke = o.tp && S.tp ? (a < 3 ? G.A : G.B) : o.hi === a ? C.c : null;
            if (shown) cell(g, ox, oy + a * (cs + gp), cs, cs, v, { scale: o.scale || 2, stroke, lw: 1.6, fmt: (q) => nf(q, 1) });
            else { V.rr(g, ox, oy + a * (cs + gp), cs, cs, 4); g.strokeStyle = V.alpha(C.muted, 0.35); g.lineWidth = 1; g.setLineDash([3, 3]); g.stroke(); g.setLineDash([]); }
          });
          if (o.label) T(g, o.label, ox + cs / 2, oy - 8, { size: 11.5, font: 'mono', align: 'center', color: o.lc || C.muted });
        };
        const col6 = (a) => (a >= 0 ? a : 0);
        const vis = (k) => i >= k;
        // layout
        let P;
        const mh6 = 6 * (cs + gp) - gp, mh4 = 4 * (cs + gp) - gp;
        if (!narrow) {
          const gx = 26;
          let cx = 16;
          const top1 = 30, top2 = top1 + mh6 + 34;
          P = {};
          P.x = [cx, top1 + (mh6 + 34 + mh6 - mh4) / 2]; cx += cs + gx;
          P.Wg = [cx, top1]; P.Wu = [cx, top2]; cx += 4 * (cs + gp) + gx;
          P.g = [cx, top1]; P.u = [cx, top2]; cx += cs + gx;
          P.s = [cx, top1]; cx += cs + gx;
          P.p = [cx, (top1 + top2) / 2]; cx += cs + gx;
          P.Wd = [cx, (top1 + top2) / 2 + (mh6 - mh4) / 2]; cx += 6 * (cs + gp) + gx;
          P.y = [cx, P.Wd[1]]; cx += cs + gx;
          P.h = [cx, P.Wd[1]];
        } else {
          const gx = 14;
          P = {};
          const r1 = 28, r2 = r1 + mh6 + 30, r3 = r2 + mh6 + 30;
          P.x = [14, r1]; P.Wg = [14 + cs + gx, r1]; P.g = [P.Wg[0] + 4 * (cs + gp) + gx, r1]; P.s = [P.g[0] + cs + gx, r1];
          P.Wu = [14 + cs + gx, r2]; P.u = [P.Wu[0] + 4 * (cs + gp) + gx, r2]; P.p = [P.u[0] + cs + gx, r2];
          P.Wd = [14, r3]; P.y = [14 + 6 * (cs + gp) + gx, r3]; P.h = [P.y[0] + cs + gx, r3];
        }
        vec(x, P.x[0], P.x[1], { label: 'x', scale: 1.6 });
        mat(Wg, P.Wg[0], P.Wg[1], { label: 'W_gate [6×4]', hiRow: i === 1 ? sweep(6) : null, tpRows: true });
        mat(Wu, P.Wu[0], P.Wu[1], { label: 'W_up [6×4]', hiRow: i === 2 ? sweep(6) : null, tpRows: true });
        vec(gv, P.g[0], P.g[1], { label: 'g', reveal: i === 1 ? sweep(6) : vis(1) ? 9 : -1, tp: true, scale: 2.5 });
        vec(uv, P.u[0], P.u[1], { label: 'u', reveal: i === 2 ? sweep(6) : vis(2) ? 9 : -1, tp: true, scale: 2.5 });
        vec(sv, P.s[0], P.s[1], { label: 'SiLU(g)', reveal: vis(3) ? 9 : -1, tp: true, scale: 2.5, lc: C.a });
        vec(pv, P.p[0], P.p[1], { label: '⊙', reveal: vis(4) ? 9 : -1, tp: true, scale: 3, lc: C.c });
        mat(Wd, P.Wd[0], P.Wd[1], { label: 'W_down [4×6]', tpCols: true });
        vec(S.tp && i === 5 && el < 2.2 ? yA : yv, P.y[0], P.y[1], { label: S.tp && i === 5 && el < 2.2 ? 'y (GPU A)' : 'y', reveal: vis(5) ? 9 : -1, scale: 3 });
        vec(i >= 6 ? hout : hres, P.h[0], P.h[1], { label: i >= 6 ? "h'' = h' + y" : "h'", scale: 3, lc: i >= 6 ? C.a : C.muted });
        // arrows/decoration per step
        if (i === 3 && !narrow) {
          // mini SiLU curve under the SiLU(g) column
          const bx = P.s[0] - 4, by = P.Wu[1] + 6;
          const bw = P.p[0] - P.s[0] - 4, bh = 84;
          {
            V.box(g, bx, by, bw, bh, { r: 6, fill: V.alpha(C.panel, 0.9), stroke: V.alpha(C.a, 0.5), lw: 1 });
            g.beginPath();
            for (let q = 0; q <= 40; q++) { const z = -4 + (8 * q) / 40; const xx = bx + 6 + ((bw - 12) * q) / 40; const yy = by + bh - 10 - (silu(z) + 0.3) * ((bh - 16) / 4.3); if (q === 0) g.moveTo(xx, yy); else g.lineTo(xx, yy); }
            g.strokeStyle = C.a; g.lineWidth = 2; g.stroke();
            T(g, 'SiLU', bx + 8, by + 14, { size: 11, color: C.a, font: 'mono' });
          }
        }
        if (i === 4) { gv.forEach((q, a) => { if (q < 0) T(g, '✕', P.s[0] + cs + 5, P.s[1] + a * (cs + gp) + cs / 2 + 4, { size: 11, color: C.muted }); }); }
        if (S.tp) {
          const ly = narrow ? h - 16 : h - 14;
          g.fillStyle = G.A; V.rr(g, 16, ly - 10, 10, 10, 2); g.fill(); T(g, 'GPU A: neurons 0–2', 30, ly, { size: 11, color: C.muted });
          g.fillStyle = G.B; V.rr(g, 180, ly - 10, 10, 10, 2); g.fill(); T(g, 'GPU B: neurons 3–5', 194, ly, { size: 11, color: C.muted });
          if (i === 5) {
            const ph = V.clamp(el / 2.2, 0, 1);
            const xx = P.y[0] + cs / 2, yy = P.y[1] + mh4 + 18;
            T(g, ph < 1 ? 'all-reduce: A + B …' : 'all-reduce done', xx, yy, { size: 11.5, align: 'center', color: C.d, weight: 600 });
            if (ph < 1) V.glow(g, xx, P.y[1] + mh4 / 2, 22 + 6 * Math.sin(t * 8), C.d, 0.6);
          }
        } else if (!narrow) {
          T(g, 'illustrative values · [output × input]', 16, h - 14, { size: 11, color: C.muted });
        }
      });
    });

    /* ====================================================================
       Figure 6 — The SwiGLU limiter
       ==================================================================== */
    safe('clamp', () => {
      const fig = $('#fig-clamp');
      fitH(fig, (w) => (w < 620 ? 660 : 400));
      const S = { L: 10, on: true, sem: 'ref' };
      const r = V.rng(31);
      const tdist = (df) => { let z = r.normal(), c = 0; for (let q = 0; q < df; q++) { const n = r.normal(); c += n * n; } return z / Math.sqrt(c / df); };
      const N = 160;
      const samples = Array.from({ length: N }, () => ({ g: tdist(3) * 3.2, u: tdist(3) * 3.2, ph: r() * 6.28, sp: 0.3 + r() * 0.6 }));
      const act = (g0, u0) => {
        if (!S.on) return silu(g0) * u0;
        const u1 = V.clamp(u0, -S.L, S.L);
        return S.sem === 'ref' ? silu(Math.min(g0, S.L)) * u1 : Math.min(silu(g0), S.L) * u1;
      };
      const ctl = UI.controls(fig);
      UI.slider(ctl, { label: 'Limit L', min: 1, max: 20, step: 0.5, value: 10, fmt: (v) => nf(v, 1), onInput: (v) => { S.L = v; dirty = true; upd(); } });
      UI.toggle(ctl, { label: 'clamp on', value: true, onChange: (v) => { S.on = v; dirty = true; upd(); } });
      UI.seg(ctl, { label: 'Semantics', options: [{ v: 'ref', label: 'SiLU(min(g,L))' }, { v: 'coop', label: 'min(SiLU(g),L)' }], value: 'ref', onChange: (v) => { S.sem = v; dirty = true; upd(); } });
      const rMax = UI.readout(ctl, 'max |act| (g,u ≤ 20)', '—');
      const rPct = UI.readout(ctl, 'clamped samples', '—');
      const rDif = UI.readout(ctl, 'diff. between semantics', '—');
      function upd() {
        const m = S.on ? silu(S.L) * S.L : silu(20) * 20;
        rMax.set(nf(m, 3));
        const cl = samples.filter((p) => S.on && (p.g > S.L || Math.abs(p.u) > S.L)).length;
        rPct.set(nf((100 * cl) / N, 1) + '%');
        rDif.set(fx(S.L * (1 - sig(S.L)), 1));
      }
      upd();
      // heatmap on an offscreen canvas
      const HM = 56;
      const off = document.createElement('canvas'); off.width = HM; off.height = HM;
      const og = off.getContext('2d');
      let dirty = true, lastDark = null;
      function paintHeat() {
        const C = V.C();
        const img = og.createImageData(HM, HM);
        const lg = Math.log1p(400);
        for (let a = 0; a < HM; a++) for (let b = 0; b < HM; b++) {
          const gg = -20 + (40 * (b + 0.5)) / HM, uu = 20 - (40 * (a + 0.5)) / HM;
          const v = act(gg, uu);
          const tv = Math.sign(v) * Math.log1p(Math.abs(v)) / lg;
          const col = V.cmap(tv, 'div');
          const m = col.match(/\d+/g).map(Number);
          const k = (a * HM + b) * 4;
          img.data[k] = m[0]; img.data[k + 1] = m[1]; img.data[k + 2] = m[2]; img.data[k + 3] = 255;
        }
        og.putImageData(img, 0, 0);
        dirty = false; lastDark = C.dark;
      }
      F(fig, (g, w, h, t) => {
        const C = V.C();
        if (dirty || lastDark !== C.dark) paintHeat();
        const narrow = w < 620;
        // panels
        let pA, pB, hmR;
        if (!narrow) {
          const lw = Math.min(360, w * 0.42);
          pA = { x: 46, y: 30, w: lw - 50, h: (h - 90) / 2 };
          pB = { x: 46, y: 30 + (h - 90) / 2 + 40, w: lw - 50, h: (h - 90) / 2 };
          const s = Math.min(w - lw - 40, h - 60);
          hmR = { x: lw + 30, y: 30, s };
        } else {
          pA = { x: 46, y: 28, w: w - 62, h: 120 };
          pB = { x: 46, y: 190, w: w - 62, h: 110 };
          const s = Math.min(w - 60, h - 370);
          hmR = { x: (w - s) / 2 + 10, y: 344, s };
        }
        // panel A: the gate path
        const plot = (R, xmin, xmax, ymin, ymax, fns, title, xl) => {
          V.box(g, R.x, R.y, R.w, R.h, { r: 6, fill: V.alpha(C.panel, 0.5), stroke: C.grid, lw: 1 });
          const X = (v) => R.x + ((v - xmin) / (xmax - xmin)) * R.w, Y = (v) => R.y + R.h - ((v - ymin) / (ymax - ymin)) * R.h;
          V.line(g, X(0), R.y, X(0), R.y + R.h, { color: V.alpha(C.muted, 0.4), lw: 1 });
          V.line(g, R.x, Y(0), R.x + R.w, Y(0), { color: V.alpha(C.muted, 0.4), lw: 1 });
          fns.forEach((fo) => {
            g.beginPath();
            for (let q = 0; q <= 120; q++) { const xv = xmin + ((xmax - xmin) * q) / 120; const yv = V.clamp(fo.f(xv), ymin, ymax); const xx = X(xv), yy = Y(yv); if (q === 0) g.moveTo(xx, yy); else g.lineTo(xx, yy); }
            g.strokeStyle = fo.c; g.lineWidth = fo.lw || 2; if (fo.dash) g.setLineDash(fo.dash); g.stroke(); g.setLineDash([]);
          });
          if (S.on) { V.line(g, X(S.L), R.y, X(S.L), R.y + R.h, { color: V.alpha(C.c, 0.7), lw: 1, dash: [3, 3] }); }
          T(g, title, R.x + 6, R.y + 14, { size: 11.5, weight: 600, color: C.ink });
          T(g, xl, R.x + R.w, R.y + R.h + 14, { size: 11, align: 'right', color: C.muted, font: 'mono' });
          T(g, nf(xmin, 0), R.x, R.y + R.h + 14, { size: 11, color: C.muted, font: 'mono' });
          T(g, nf(ymax, 0), R.x - 4, R.y + 10, { size: 11, align: 'right', color: C.muted, font: 'mono' });
          T(g, nf(ymin, 0), R.x - 4, R.y + R.h, { size: 11, align: 'right', color: C.muted, font: 'mono' });
          return { X, Y };
        };
        const gateF = (z) => (!S.on ? silu(z) : S.sem === 'ref' ? silu(Math.min(z, S.L)) : Math.min(silu(z), S.L));
        const A = plot(pA, -15, 25, -3, 25, [{ f: silu, c: V.alpha(C.muted, 0.8), dash: [4, 4], lw: 1.5 }, { f: gateF, c: C.a, lw: 2.5 }], 'gate: SiLU(min(g, L))', 'g →');
        const B = plot(pB, -25, 25, -25, 25, [{ f: (z) => z, c: V.alpha(C.muted, 0.8), dash: [4, 4], lw: 1.5 }, { f: (z) => (S.on ? V.clamp(z, -S.L, S.L) : z), c: C.b, lw: 2.5 }], 'up: clamp(u, −L, L)', 'u →');
        // samples sliding along the curves
        samples.slice(0, 40).forEach((p, q) => {
          const gq = p.g + Math.sin(t * p.sp + p.ph) * 0.6;
          const xg = V.clamp(gq, -15, 25);
          const cl = S.on && gq > S.L;
          V.dot(g, A.X(xg), A.Y(V.clamp(gateF(gq), -3, 25)), cl ? 3.2 : 2.2, cl ? C.c : V.alpha(C.a, 0.7));
          const uq = p.u + Math.cos(t * p.sp + p.ph) * 0.6;
          const cu = S.on && Math.abs(uq) > S.L;
          V.dot(g, B.X(V.clamp(uq, -25, 25)), B.Y(V.clamp(S.on ? V.clamp(uq, -S.L, S.L) : uq, -25, 25)), cu ? 3.2 : 2.2, cu ? C.c : V.alpha(C.b, 0.7));
        });
        // heatmap
        const s = hmR.s;
        g.imageSmoothingEnabled = false;
        g.drawImage(off, hmR.x, hmR.y, s, s);
        g.imageSmoothingEnabled = true;
        g.strokeStyle = C.grid; g.lineWidth = 1; g.strokeRect(hmR.x, hmR.y, s, s);
        const GX = (v) => hmR.x + ((v + 20) / 40) * s, GY = (v) => hmR.y + ((20 - v) / 40) * s;
        if (S.on && S.L < 20) { g.setLineDash([5, 4]); g.strokeStyle = C.ink; g.lineWidth = 1.3; g.strokeRect(GX(-20), GY(S.L), GX(S.L) - GX(-20), GY(-S.L) - GY(S.L)); g.setLineDash([]); }
        V.line(g, GX(0), hmR.y, GX(0), hmR.y + s, { color: V.alpha(C.ink, 0.25), lw: 1 });
        V.line(g, hmR.x, GY(0), hmR.x + s, GY(0), { color: V.alpha(C.ink, 0.25), lw: 1 });
        T(g, 'act(g, u)', hmR.x, hmR.y - 8, { size: 11.5, weight: 600, color: C.ink });
        T(g, 'g →', hmR.x + s, hmR.y + s + 14, { size: 11, align: 'right', color: C.muted, font: 'mono' });
        T(g, 'u ↑', hmR.x - 6, hmR.y + 10, { size: 11, align: 'right', color: C.muted, font: 'mono' });
        samples.forEach((p) => {
          const gq = V.clamp(p.g + Math.sin(t * p.sp + p.ph) * 0.6, -20, 20), uq = V.clamp(p.u + Math.cos(t * p.sp + p.ph) * 0.6, -20, 20);
          const cg = S.on ? Math.min(gq, S.L) : gq, cu = S.on ? V.clamp(uq, -S.L, S.L) : uq;
          const cl = cg !== gq || cu !== uq;
          if (cl) { V.line(g, GX(gq), GY(uq), GX(cg), GY(cu), { color: V.alpha(C.c, 0.6), lw: 1 }); V.dot(g, GX(gq), GY(uq), 1.6, V.alpha(C.c, 0.5)); }
          V.dot(g, GX(cg), GY(cu), cl ? 3 : 2, cl ? C.c : C.ink);
        });
        const mxs = S.on ? '|act| ≤ ' + nf(silu(S.L) * S.L, 3) : 'no clamp: 400 at corner, unbounded';
        T(g, mxs, hmR.x + s, hmR.y - 8, { size: 11, align: 'right', color: S.on ? C.a : C.c, font: 'mono' });
      });
    });

    /* ====================================================================
       Figure 7 — Block calculator
       ==================================================================== */
    safe('calc', () => {
      const fig = $('#fig-calc');
      fitH(fig, (w) => (w < 620 ? 650 : 470));
      const PRE = {
        classic: { d: 4096, h: 32, kv: 32, dh: 128, I: 11008, moe: false, E: 8, k: 2, gate: false, N: 32, V: 32000 },
        qwen: { d: 2560, h: 24, kv: 2, dh: 256, I: 640, moe: true, E: 512, k: 10, gate: true, N: 48, V: 248320 },
        wide: { d: 8192, h: 64, kv: 8, dh: 128, I: 28672, moe: false, E: 8, k: 2, gate: false, N: 80, V: 128000 },
      };
      const S = Object.assign({ ne: 13, ang: 0, preset: 'classic' }, PRE.classic);
      const ctl = UI.controls(fig);
      const sl = {};
      let loading = false;
      const rd = () => { if (!loading) H.redraw(); };
      UI.seg(ctl, { label: 'Preset', options: [{ v: 'classic', label: 'Classic (illustr.)' }, { v: 'qwen', label: 'Qwen · QSA + MoE' }, { v: 'wide', label: 'Wide dense (illustr.)' }], value: 'classic', onChange: (v) => { loadPreset(v); } });
      const ffnSeg = UI.seg(ctl, { label: 'FFN', options: [{ v: false, label: 'dense' }, { v: true, label: 'MoE' }], value: false, onChange: (v) => { S.moe = v; rd(); } });
      const gateT = UI.toggle(ctl, { label: 'output gate (doubles q_proj)', value: false, onChange: (v) => { S.gate = v; rd(); } });
      sl.d = UI.slider(ctl, { label: 'd (hidden)', min: 256, max: 8192, step: 256, value: S.d, fmt: ni, onInput: (v) => { S.d = v; rd(); } });
      sl.h = UI.slider(ctl, { label: 'heads h', min: 1, max: 128, step: 1, value: S.h, onInput: (v) => { S.h = v; rd(); } });
      sl.kv = UI.slider(ctl, { label: 'KV heads', min: 1, max: 128, step: 1, value: S.kv, onInput: (v) => { S.kv = v; rd(); } });
      sl.dh = UI.slider(ctl, { label: 'head_dim', min: 32, max: 512, step: 32, value: S.dh, onInput: (v) => { S.dh = v; rd(); } });
      sl.I = UI.slider(ctl, { label: 'I (FFN/expert)', min: 128, max: 32768, step: 128, value: S.I, fmt: ni, onInput: (v) => { S.I = v; rd(); } });
      sl.E = UI.slider(ctl, { label: 'experts E', min: 2, max: 512, step: 2, value: S.E, onInput: (v) => { S.E = v; rd(); } });
      sl.k = UI.slider(ctl, { label: 'top-k', min: 1, max: 16, step: 1, value: S.k, onInput: (v) => { S.k = v; rd(); } });
      sl.N = UI.slider(ctl, { label: 'layers N', min: 1, max: 96, step: 1, value: S.N, onInput: (v) => { S.N = v; rd(); } });
      sl.ne = UI.slider(ctl, { label: 'context n', min: 10, max: 20, step: 0.5, value: S.ne, fmt: (v) => ni(Math.pow(2, v)), onInput: (v) => { S.ne = v; rd(); } });
      const rBlk = UI.readout(ctl, 'block', '—');
      const rTot = UI.readout(ctl, 'model (× N + emb.)', '—');
      const rFl = UI.readout(ctl, 'GFLOP/token', '—');
      function loadPreset(p) {
        const P0 = PRE[p]; S.preset = p; loading = true;
        Object.assign(S, { V: P0.V });
        ['d', 'h', 'kv', 'dh', 'I', 'E', 'k', 'N'].forEach((k) => sl[k].set(P0[k]));
        ffnSeg.set(P0.moe); gateT.set(P0.gate);
        loading = false; H.redraw();
      }
      function calc() {
        const kv = Math.min(S.kv, S.h);
        const Pq = S.d * S.h * S.dh * (S.gate ? 2 : 1), Pk = S.d * kv * S.dh, Pv = Pk, Po = S.h * S.dh * S.d;
        const Pattn = Pq + Pk + Pv + Po;
        const Pe = 3 * S.d * S.I;
        const k = Math.min(S.k, S.E);
        const Pffn = S.moe ? (S.E + 1) * Pe + S.d * S.E : Pe;
        const Pffa = S.moe ? (k + 1) * Pe + S.d * S.E : Pe;
        const Pn = 2 * S.d;
        const blk = Pattn + Pffn + Pn, act = Pattn + Pffa + Pn;
        const n = Math.pow(2, S.ne);
        const fProj = 2 * S.N * Pattn, fFfn = 2 * S.N * Pffa, fSc = 4 * S.N * S.h * S.dh * n, fHead = 2 * S.V * S.d;
        const tot = S.N * blk + 2 * S.V * S.d + S.d;
        const nstar = act / (2 * S.h * S.dh);
        return { kv, Pq, Pk, Pv, Po, Pattn, Pe, Pffn, Pffa, Pn, blk, act, n, fProj, fFfn, fSc, fHead, tot, nstar, k };
      }
      const H = F(fig, (g, w, h, t, dt) => {
        const C = V.C();
        const R = calc();
        rBlk.set(pp(R.blk) + (S.moe ? ' (active ' + pp(R.act) + ')' : ''));
        rTot.set(pp(R.tot));
        const fl = R.fProj + R.fFfn + R.fSc + R.fHead;
        rFl.set(nf(fl / 1e9, fl / 1e9 >= 100 ? 0 : 1));
        const narrow = w < 620;
        // --- scale
        const A = R.Pattn, F = R.Pffn;
        const target = V.clamp((F - A) / (F + A), -1, 1) * 0.32;
        S.ang = V.lerp(S.ang, target, Math.min(1, (dt || 0.016) * 6));
        if (Math.abs(S.ang - target) > 0.002) H.redraw();
        const bw2 = narrow ? w * 0.36 : Math.min(170, w * 0.19);
        const px = narrow ? w / 2 : 30 + bw2 + 20, py = narrow ? 54 : 62;
        T(g, 'Parameters of one block', 14, 20, { size: 12.5, weight: 600, color: C.ink });
        const ca = Math.cos(S.ang), sa = Math.sin(S.ang);
        const Lx = px - bw2 * ca, Ly = py - bw2 * sa, Rx = px + bw2 * ca, Ry = py + bw2 * sa;
        g.fillStyle = V.alpha(C.muted, 0.5); g.beginPath(); g.moveTo(px, py); g.lineTo(px - 14, py + 70); g.lineTo(px + 14, py + 70); g.closePath(); g.fill();
        V.line(g, Lx, Ly, Rx, Ry, { color: C.ink, lw: 3 });
        V.dot(g, px, py, 5, C.ink);
        const pan = (x, y, col, lbl, val, sub) => {
          const pyy = y + 40;
          V.line(g, x, y, x - 26, pyy, { color: V.alpha(C.muted, 0.7), lw: 1 }); V.line(g, x, y, x + 26, pyy, { color: V.alpha(C.muted, 0.7), lw: 1 });
          g.beginPath(); g.moveTo(x - 32, pyy); g.quadraticCurveTo(x, pyy + 18, x + 32, pyy); g.closePath(); g.fillStyle = V.alpha(col, 0.55); g.fill(); g.strokeStyle = col; g.lineWidth = 1.5; g.stroke();
          const frac = val / (A + F);
          const bh = 6 + 34 * Math.sqrt(frac);
          V.box(g, x - 13, pyy - bh, 26, bh, { r: 3, fill: col });
          T(g, lbl, x, pyy + 30, { size: 11.5, weight: 600, align: 'center', color: C.ink });
          T(g, pp(val) + ' (' + nf(frac * 100, frac < 0.1 ? 1 : 0) + '%)', x, pyy + 45, { size: 11, align: 'center', font: 'mono', color: C.muted });
          if (sub) T(g, sub, x, pyy + 59, { size: 11, align: 'center', font: 'mono', color: col });
        };
        pan(Lx, Ly, C.b, 'Attention', A, S.gate ? 'q_proj with gate' : 'GQA ' + S.h + 'Q/' + R.kv + 'KV');
        pan(Rx, Ry, C.a, S.moe ? 'MoE FFN' : 'SwiGLU FFN', F, S.moe ? 'active ' + pp(R.Pffa) : 'I = ' + ni(S.I));
        // --- composition bar
        let cx0, cy0, cwid;
        if (narrow) { cx0 = 14; cy0 = 224; cwid = w - 28; } else { cx0 = px + bw2 + 60; cy0 = 44; cwid = w - cx0 - 16; }
        const segs = [['Q' + (S.gate ? '+gate' : ''), R.Pq, C.b], ['K', R.Pk, C.c], ['V', R.Pv, C.e], ['O', R.Po, C.d], [S.moe ? 'experts + shared + router' : 'gate/up/down', R.Pffn, C.a]];
        T(g, 'block composition (' + pp(R.blk) + ')', cx0, cy0 - 6, { size: 11.5, color: C.muted });
        let xx = cx0;
        const bh0 = 22;
        segs.forEach(([nm, v, col]) => {
          const ww = (v / R.blk) * cwid;
          V.rr(g, xx, cy0, Math.max(1, ww - 1), bh0, 3); g.fillStyle = col; g.fill();
          xx += ww;
        });
        // composition legend
        let lx = cx0, ly = cy0 + bh0 + 16;
        segs.forEach(([nm, v, col]) => {
          const s = nm + ' ' + nf((100 * v) / R.blk, (100 * v) / R.blk < 10 ? 1 : 0) + '%';
          const lw = textW(g, s, 11) + 22;
          if (lx + lw > cx0 + cwid) { lx = cx0; ly += 15; }
          g.fillStyle = col; V.rr(g, lx, ly - 9, 10, 10, 2); g.fill();
          T(g, s, lx + 14, ly, { size: 11, color: C.muted });
          lx += lw;
        });
        if (!narrow) {
          const cl = S.moe ? `MoE: E = ${S.E}, top-${R.k} + 1 shared; each expert 3·d·I = ${pp(R.Pe)}` : `FFN = 3·d·I = ${pp(R.Pffn)}; attention = ${pp(R.Pattn)}`;
          wrap(g, cl, cx0, ly + 20, cwid, 14, { size: 11, color: C.muted, font: 'mono' });
          if (S.kv > S.h) T(g, 'KV heads capped at h', cx0, ly + 50, { size: 11, color: C.c });
          if (S.preset === 'qwen' && !narrow) wrap(g, 'Real Qwen: 12 of 48 layers are QSA, the other 36 are GDN; the “× N” assumes identical blocks.', cx0, ly + 66, cwid, 14, { size: 11, color: C.muted });
        }
        // --- FLOPs
        const fy = narrow ? 330 : 238;
        T(g, 'FLOPs per token (whole model, context n = ' + ni(R.n) + ')', 14, fy, { size: 12.5, weight: 600, color: C.ink, maxW: w - 28 });
        const fsegs = [['attention projections', R.fProj, C.b], ['FFN (active)', R.fFfn, C.a], ['QKᵀ and AV scores', R.fSc, C.c], ['lm_head', R.fHead, C.d]];
        let fx = 14; const fw = w - 28, fbh = 24, fby = fy + 12;
        fsegs.forEach(([nm, v, col]) => { const ww = (v / fl) * fw; V.rr(g, fx, fby, Math.max(1, ww - 1), fbh, 3); g.fillStyle = col; g.fill(); if (ww > 60) T(g, nf((100 * v) / fl, 0) + '%', fx + ww / 2, fby + fbh / 2 + 4, { size: 11, align: 'center', color: onTxt(C), weight: 600 }); fx += ww; });
        lx = 14; ly = fby + fbh + 16;
        fsegs.forEach(([nm, v, col]) => {
          const s = nm + ' ' + nf(v / 1e9, v / 1e9 >= 10 ? 0 : 2) + ' G';
          const lw = textW(g, s, 11) + 22;
          if (lx + lw > w - 14) { lx = 14; ly += 15; }
          g.fillStyle = col; V.rr(g, lx, ly - 9, 10, 10, 2); g.fill();
          T(g, s, lx + 14, ly, { size: 11, color: C.muted });
          lx += lw;
        });
        // curve of the attention share vs context
        const gy0 = ly + 22, gy1 = h - 26, gx0 = 50, gx1 = w - 16;
        if (gy1 - gy0 > 50) {
          V.box(g, gx0, gy0, gx1 - gx0, gy1 - gy0, { r: 4, fill: V.alpha(C.panel, 0.4), stroke: C.grid, lw: 1 });
          const X = (e) => gx0 + ((e - 10) / 10) * (gx1 - gx0), Y = (fr) => gy1 - fr * (gy1 - gy0);
          const shareAt = (e) => { const n = Math.pow(2, e); const sc = 4 * S.N * S.h * S.dh * n; return sc / (R.fProj + R.fFfn + R.fHead + sc); };
          V.line(g, gx0, Y(0.5), gx1, Y(0.5), { color: V.alpha(C.muted, 0.35), lw: 1, dash: [3, 4] });
          g.beginPath();
          for (let q = 0; q <= 100; q++) { const e = 10 + q / 10; const xv = X(e), yv = Y(shareAt(e)); if (q === 0) g.moveTo(xv, yv); else g.lineTo(xv, yv); }
          g.strokeStyle = C.c; g.lineWidth = 2.2; g.stroke();
          const ens = Math.log2(Math.max(1, R.nstar));
          if (ens >= 10 && ens <= 20) { V.line(g, X(ens), gy0, X(ens), gy1, { color: C.d, lw: 1.3, dash: [4, 3] }); T(g, 'n* ≈ ' + ni(R.nstar), X(ens) + (X(ens) > gx1 - 100 ? -4 : 4), gy0 + 13, { size: 11, color: C.d, font: 'mono', align: X(ens) > gx1 - 100 ? 'right' : 'left' }); }
          else T(g, 'n* ≈ ' + ni(R.nstar) + (ens < 10 ? ' (before 1K)' : ' (beyond 1M)'), gx1 - 4, gy0 + 13, { size: 11, color: C.d, font: 'mono', align: 'right' });
          const cur = shareAt(S.ne);
          V.dot(g, X(S.ne), Y(cur), 5, C.c); V.glow(g, X(S.ne), Y(cur), 14, C.c, 0.6);
          T(g, '100%', gx0 - 4, gy0 + 10, { size: 11, align: 'right', color: C.muted, font: 'mono' });
          T(g, '0', gx0 - 4, gy1, { size: 11, align: 'right', color: C.muted, font: 'mono' });
          T(g, '1K', gx0, gy1 + 14, { size: 11, color: C.muted, font: 'mono' });
          T(g, '1M', gx1, gy1 + 14, { size: 11, align: 'right', color: C.muted, font: 'mono' });
          T(g, 'share of FLOPs in the attention scores', (gx0 + gx1) / 2, gy1 + 14, { size: 11, align: 'center', color: C.muted, maxW: gx1 - gx0 - 70 });
        }
      }, { static: true });
    });

    /* ====================================================================
       Figure 8 — The subway map of the layers
       ==================================================================== */
    safe('metro', () => {
      const fig = $('#fig-metro');
      fitH(fig, (w) => (w < 620 ? 640 : 410));
      const C0 = V.C();
      function build(m) {
        const L = [];
        if (m === 'glm') {
          for (let i = 0; i < 45; i++) {
            const mla = i % 4 === 3 && i < 44;
            const dense = i < 3;
            L.push({ i, mix: mla ? 'MLA' : 'KDA', mixc: mla ? 'b' : 'a', ffn: dense ? 'dense' : 'MoE', attn: mla ? 124914432 : 137732288, ffnP: dense ? 150994944 : 7274103072, ffnA: dense ? 150994944 : 227672352, hc: 786486, tap: [5, 14, 24, 33, 42].includes(i), note: dense ? 'Dense SwiGLU FFN, I = 12,288 (first_k_dense_replace = 3)' : mla ? 'MLA without RoPE (NoPE) + sparse top-2048 indexer' : 'KDA: linear attention with a fixed recurrent state' });
          }
          L.push({ i: 45, mix: 'MLA', mixc: 'b', ffn: 'MoE', attn: 124914432, ffnP: 7274103072, ffnA: 227672352, hc: 0, mtp: true, note: 'MTP block (index 45): MLA + MoE with a simple residual, ≈ 7.43 B in total; used as a drafter with SPEC_METHOD=mtp' });
          return { L, types: [['KDA', 'a'], ['sparse MLA', 'b']], hcl: 'mHC', tapNote: 'output read by DFlash2 (EAGLE3 hook with hc_post + hc_contract)' };
        }
        if (m === 'qwen') {
          for (let i = 0; i < 48; i++) {
            const qsa = (i + 1) % 4 === 0;
            L.push({ i, mix: qsa ? 'QSA' : 'GDN', mixc: qsa ? 'c' : 'e', ffn: 'MoE', attn: qsa ? 51446528 : 57958624, ffnP: 2522810880, ffnA: 55380480, hc: 13209600, ple: i === 1, note: qsa ? 'QSA: sparse softmax attention (≤ 2048 tokens), GQA 24Q/2KV × 256, with output gate' : 'Gated DeltaNet: linear attention with a 128×128 state per V head' });
          }
          L.push({ i: 48, mix: 'QSA', mixc: 'c', ffn: 'MoE', attn: 51446528, ffnP: 2522810880, ffnA: 55380480, hc: 13209600, mtp: true, note: 'MTP layer (checkpoint mtp.layers.0, instantiated as mtp.layers.48): QSA + MoE with FP8 experts in 128×128 blocks' });
          return { L, types: [['GDN', 'e'], ['QSA', 'c']], hcl: 'Gated Residual', tapNote: '' };
        }
        for (let i = 0; i < 43; i++) {
          const swa = i < 2, c4 = !swa && i % 2 === 0;
          const attn = swa ? 107.0e6 : c4 ? 126.1e6 : 111.3e6;
          L.push({ i, mix: swa ? 'SWA' : c4 ? 'C4A' : 'C128A', mixc: swa ? 'e' : c4 ? 'c' : 'b', ffn: i < 3 ? 'hash' : 'MoE', attn, ffnP: 6.469e9, ffnA: 177209344, hc: 0.79e6, tap: i >= 40, approx: true, note: swa ? 'pure SWA (window 128)' : c4 ? 'CSA/C4A: 4× compressed KV + top-512 indexer + window 128' : 'HCA/C128A: 128× compressed KV (dense) + window 128' });
        }
        for (let s = 0; s < 3; s++) L.push({ i: 43 + s, mix: 'SWA', mixc: 'e', ffn: 'MoE', attn: 0, ffnP: 0, ffnA: 0, hc: 0, mtp: true, approx: true, note: 'DSpark stage ' + s + ' (mtp.' + s + '): SWA over the target features + full MoE; ≈ 19.8 B across the 3 stages' });
        return { L, types: [['SWA', 'e'], ['C4A (sparse)', 'c'], ['C128A', 'b']], hcl: 'mHC', tapNote: 'DSpark target layer (average of the 4 streams → main_proj)' };
      }
      const S = { m: 'glm', data: build('glm'), lock: -1 };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Model', options: [{ v: 'glm', label: 'GLM' }, { v: 'qwen', label: 'Qwen' }, { v: 'ds', label: 'DeepSeek' }], value: 'glm', onChange: (v) => { S.m = v; S.data = build(v); S.lock = -1; } });
      UI.button(ctl, 'clear selection', () => { S.lock = -1; });
      UI.legend(ctl, [[C0.d, 'amber ring: dense FFN'], [C0.muted, 'dotted ring: MoE'], [C0.f, 'double ring: hash MoE']]);
      const H = F(fig, (g, w, h, t, dt, _s, Hh) => {
        const C = V.C();
        const D = S.data, L = D.L;
        const narrow = w < 620;
        const sp = narrow ? 36 : 36;
        const R = Math.max(5, Math.floor((w - 50) / sp) + 1);
        const rows = Math.ceil(L.length / R);
        const rowH = 56;
        // type legend
        let lx = 14; let ly = 22;
        D.types.forEach(([nm, c]) => { const s = nm; const lw = textW(g, s, 11.5) + 26; if (lx + lw > w - 10) { lx = 14; ly += 16; } g.fillStyle = C[c]; g.beginPath(); g.arc(lx + 6, ly - 4, 6, 0, 6.3); g.fill(); T(g, s, lx + 16, ly, { size: 11.5, color: C.muted }); lx += lw; });
        const mk = [['◆', 'drafter reads'], ['✚', 'N-gram'], ['★', 'MTP/DSpark']];
        mk.forEach(([s1, s2]) => { const s = s1 + ' ' + s2; const lw = textW(g, s, 11.5) + 18; if (lx + lw > w - 10) { lx = 14; ly += 16; } T(g, s, lx, ly, { size: 11.5, color: C.muted }); lx += lw; });
        const top = ly + 22;
        const pos = (k) => { const r = Math.floor(k / R), c = k % R; const cc = r % 2 === 0 ? c : R - 1 - c; return [25 + cc * ((w - 50) / (R - 1)), top + 16 + r * rowH]; };
        // line
        g.beginPath();
        L.forEach((_, k) => { const [x, y] = pos(k); if (k === 0) g.moveTo(x, y); else { const [px, py] = pos(k - 1); if (py !== y) { const ex = px; g.bezierCurveTo(ex + (x > w / 2 ? 24 : -24), py, ex + (x > w / 2 ? 24 : -24), y, x, y); } else g.lineTo(x, y); } });
        g.strokeStyle = V.alpha(C.muted, 0.45); g.lineWidth = 4; g.stroke();
        // train
        const speed = 2.2;
        const tpos = (t * speed) % (L.length + 2);
        const ti = Math.min(L.length - 1, Math.floor(tpos));
        // pointer selection
        let hov = -1;
        if (Hh.pointer.inside) L.forEach((_, k) => { const [x, y] = pos(k); if (Math.hypot(Hh.pointer.x - x, Hh.pointer.y - y) < 14) hov = k; });
        if (Hh.pointer.clicked && hov >= 0) S.lock = hov;
        const sel = hov >= 0 ? hov : S.lock >= 0 && S.lock < L.length ? S.lock : ti;
        L.forEach((s, k) => {
          const [x, y] = pos(k);
          const col = C[s.mixc] || C.a;
          const act = k === sel;
          if (s.ffn === 'dense') { g.beginPath(); g.arc(x, y, 12, 0, 6.3); g.strokeStyle = C.d; g.lineWidth = 3; g.stroke(); }
          else if (s.ffn === 'hash') { g.beginPath(); g.arc(x, y, 11, 0, 6.3); g.strokeStyle = C.f; g.lineWidth = 1.5; g.stroke(); g.beginPath(); g.arc(x, y, 14, 0, 6.3); g.stroke(); }
          else { g.setLineDash([2, 3]); g.beginPath(); g.arc(x, y, 12, 0, 6.3); g.strokeStyle = V.alpha(C.muted, 0.9); g.lineWidth = 1.5; g.stroke(); g.setLineDash([]); }
          g.beginPath(); g.arc(x, y, act ? 9 : 7.5, 0, 6.3); g.fillStyle = s.mtp ? V.alpha(col, 0.45) : col; g.fill();
          if (act) { g.strokeStyle = C.ink; g.lineWidth = 2; g.stroke(); }
          if (k === ti) V.glow(g, x, y, 18, C.ink, 0.35);
          if (s.tap) T(g, '◆', x, y - 17, { size: 11, align: 'center', color: C.d });
          if (s.ple) T(g, '✚', x, y - 17, { size: 12, align: 'center', color: C.d });
          if (s.mtp) T(g, '★', x, y - 17, { size: 12, align: 'center', color: C.f });
          if (s.i % 5 === 0 || act) T(g, String(s.i), x, y + 25, { size: 11, align: 'center', font: 'mono', color: act ? C.ink : C.muted });
        });
        // train between stations
        const fr = tpos - ti;
        if (ti < L.length - 1) { const a = pos(ti), b = pos(ti + 1); const p = V.along(a[0], a[1], b[0], b[1], V.smooth(V.clamp(fr, 0, 1))); V.dot(g, p[0], p[1], 4, C.ink); }
        // info panel
        const s = L[sel];
        const iy = top + 16 + rows * rowH - 10;
        const ih = h - iy - 8;
        V.box(g, 12, iy, w - 24, ih, { r: 10, fill: V.alpha(C.panel, 0.85), stroke: V.alpha(C[s.mixc] || C.a, 0.6), lw: 1.3 });
        const ttl = (s.mtp ? '' : 'Layer ') + (s.mtp ? 'Extra ' + s.i : s.i) + ' · ' + s.mix + ' · FFN: ' + s.ffn + (S.lock >= 0 ? '  (pinned)' : '');
        T(g, ttl, 24, iy + 20, { size: 13, weight: 700, color: C[s.mixc] || C.a, maxW: w - 48 });
        let yy = iy + 38;
        yy += 15 * wrap(g, s.note, 24, yy, w - 48, 15, { size: 11.5, color: C.ink, maxLines: 3 });
        if (s.attn > 0) {
          const line = `attention ${pp(s.attn)} · FFN ${pp(s.ffnP)} (active per token ${pp(s.ffnA)})` + (s.hc ? ` · ${D.hcl} ${pp(s.hc)}` : '') + (s.approx ? ' · approx.' : '');
          yy += 15 * wrap(g, line, 24, yy + 4, w - 48, 15, { size: 11.5, font: 'mono', color: C.muted, maxLines: 3 }) + 6;
          // proportion bars
          const bx = 24, bwid = w - 48;
          const tot = s.attn + s.ffnP, ta = s.attn + s.ffnA;
          if (yy + 34 < iy + ih) {
            const b1 = (s.attn / tot) * bwid;
            V.rr(g, bx, yy, b1, 10, 3); g.fillStyle = C[s.mixc] || C.a; g.fill();
            V.rr(g, bx + b1, yy, bwid - b1, 10, 3); g.fillStyle = V.alpha(C.ink, 0.28); g.fill();
            T(g, 'total: attention ' + nf((100 * s.attn) / tot, 1) + '% · FFN ' + nf((100 * s.ffnP) / tot, 1) + '%', bx, yy + 24, { size: 11, color: C.muted });
            if (yy + 52 < iy + ih) {
              const b2 = (s.attn / ta) * bwid;
              V.rr(g, bx, yy + 32, b2, 10, 3); g.fillStyle = C[s.mixc] || C.a; g.fill();
              V.rr(g, bx + b2, yy + 32, bwid - b2, 10, 3); g.fillStyle = V.alpha(C.ink, 0.55); g.fill();
              T(g, 'active per token: attention ' + nf((100 * s.attn) / ta, 0) + '% · FFN ' + nf((100 * s.ffnA) / ta, 0) + '%', bx, yy + 56, { size: 11, color: C.muted });
            }
          }
        }
        const extra = s.tap ? '◆ ' + D.tapNote : s.ple ? '✚ the N-gram table (51.2 B parameters) adds its vector to the 4 branches at this layer' : '';
        if (extra) T(g, extra, 24, iy + ih - 10, { size: 11, color: C.d, maxW: w - 48 });
      });
    });

    /* ====================================================================
       Schematic drawing of the block (used in figures 9 and 10)
       ==================================================================== */
    function drawBlock(g, bx, by, bw, bh, o) {
      const C = V.C();
      const busX = bx + 16;
      const hasHc = o.hc !== 'none';
      const parts = [];
      const add = (key, label, hh, col, extra) => parts.push({ key, label, hh, col, extra });
      if (hasHc) add('hc1', o.hc === 'gr' ? 'GR: read (normalizes)' : 'mHC: mixes 4 streams', 0.7, C.f);
      if (o.norms) add('norm1', o.n1 || 'input_layernorm', 0.6, C.e);
      add('mixer', o.mixer, 1.5, C.b, 'attn');
      if (hasHc) add('hc2', o.hc === 'gr' ? 'GR: write + read' : 'mHC: writes and mixes', 0.7, C.f);
      if (o.norms) add('norm2', o.n2 || 'post_attention_layernorm', 0.6, C.e);
      add('ffn', o.ffn, 1.6, C.a, 'ffn');
      const tot = parts.reduce((s, p) => s + p.hh, 0);
      const gap = 7;
      const unit = (bh - 30 - gap * (parts.length - 1)) / tot;
      let y = by + 16;
      const rects = {};
      V.line(g, busX, by, busX, by + bh, { color: V.alpha(C.d, 0.8), lw: 4 });
      T(g, 'residual', busX - 4, by + bh + 12, { size: 11, color: C.d, align: 'left', font: 'mono' });
      const lit = o.lit || {};
      parts.forEach((p) => {
        const hh = p.hh * unit;
        const x = busX + 18, ww = bx + bw - x;
        const on = !!lit[p.key] || (p.key === 'mixer' && (lit['attn-in'] || lit['attn-out'])) || (p.key === 'ffn' && (lit['ffn-in'] || lit['ffn-out'] || lit.router || lit.experts));
        V.box(g, x, y, ww, hh, { r: 7, fill: V.alpha(p.col, on ? 0.34 : 0.1), stroke: V.alpha(p.col, on ? 1 : 0.45), lw: on ? 2.2 : 1 });
        T(g, p.label, x + 8, y + Math.min(hh / 2 + 4, 16), { size: fit(g, p.label, ww - 14, 12, 'sans', 600), weight: 600, color: C.ink });
        rects[p.key] = { x, y, w: ww, h: hh };
        if (p.extra === 'attn') {
          // sub-strips: readers (input) and writer (output)
          const r1 = { x: x + 8, y: y + hh - 20, w: (ww - 24) * 0.62, h: 14 }, r2 = { x: x + 16 + (ww - 24) * 0.62, y: y + hh - 20, w: (ww - 24) * 0.38, h: 14 };
          if (hh > 40) {
            V.box(g, r1.x, r1.y, r1.w, r1.h, { r: 4, fill: V.alpha(C.b, lit['attn-in'] ? 0.8 : 0.22) });
            V.box(g, r2.x, r2.y, r2.w, r2.h, { r: 4, fill: V.alpha(C.c, lit['attn-out'] ? 0.9 : 0.25) });
            T(g, o.attnIn || 'readers', r1.x + 4, r1.y + 11, { size: 11, color: C.ink, maxW: r1.w - 6 });
            T(g, o.attnOut || 'o_proj', r2.x + 4, r2.y + 11, { size: 11, color: C.ink, maxW: r2.w - 6 });
          }
          rects['attn-in'] = r1; rects['attn-out'] = r2;
        }
        if (p.extra === 'ffn') {
          const r1 = { x: x + 8, y: y + hh - 20, w: (ww - 24) * 0.5, h: 14 }, r2 = { x: x + 16 + (ww - 24) * 0.5, y: y + hh - 20, w: (ww - 24) * 0.5, h: 14 };
          if (o.moe && hh > 56) {
            const rr = { x: x + 8, y: y + 22, w: 58, h: 14 };
            V.box(g, rr.x, rr.y, rr.w, rr.h, { r: 4, fill: V.alpha(C.d, lit.router ? 0.9 : 0.25) });
            T(g, 'router', rr.x + 4, rr.y + 11, { size: 11, color: C.ink, maxW: rr.w - 6 });
            rects.router = rr;
            const n = Math.max(4, Math.min(14, Math.floor((ww - 90) / 12)));
            for (let q = 0; q < n; q++) { g.fillStyle = V.alpha(C.a, lit.experts ? (q % 3 === 0 ? 0.95 : 0.45) : 0.25); V.rr(g, x + 74 + q * 12, y + 24, 9, 10, 2); g.fill(); }
            rects.experts = { x: x + 74, y: y + 22, w: n * 12, h: 14 };
          }
          if (hh > 40) {
            V.box(g, r1.x, r1.y, r1.w, r1.h, { r: 4, fill: V.alpha(C.a, lit['ffn-in'] ? 0.8 : 0.22) });
            V.box(g, r2.x, r2.y, r2.w, r2.h, { r: 4, fill: V.alpha(C.c, lit['ffn-out'] ? 0.9 : 0.25) });
            T(g, o.ffnIn || 'gate/up', r1.x + 4, r1.y + 11, { size: 11, color: C.ink, maxW: r1.w - 6 });
            T(g, o.ffnOut || 'down', r2.x + 4, r2.y + 11, { size: 11, color: C.ink, maxW: r2.w - 6 });
          }
          rects['ffn-in'] = r1; rects['ffn-out'] = r2;
        }
        // arrows: reads from the residual (input) and writes (⊕)
        if (p.key === 'mixer' || p.key === 'ffn') {
          V.arrow(g, x + ww * 0.5, y + hh + 1, busX + 5, y + hh + gap * 0.5, { color: V.alpha(C.c, 0.7), lw: 1.5, head: 6 });
          g.beginPath(); g.arc(busX, y + hh + gap * 0.5, 6, 0, 6.3); g.fillStyle = C.panel; g.fill(); g.strokeStyle = C.d; g.lineWidth = 1.5; g.stroke();
          T(g, '+', busX, y + hh + gap * 0.5 + 4, { size: 12, align: 'center', color: C.d, weight: 700 });
        } else V.arrow(g, busX + 3, y + hh / 2, x - 2, y + hh / 2, { color: V.alpha(C.muted, 0.7), lw: 1.2, head: 5 });
        y += hh + gap;
      });
      rects.bus = { x: busX - 6, y: by, w: 12, h: bh };
      return rects;
    }

    /* ====================================================================
       Figure 9 — The tensor cabinet
       ==================================================================== */
    safe('tensores', () => {
      const fig = $('#fig-tensores');
      fitH(fig, (w) => (w < 620 ? 680 : 470));
      const E = (n, s, t, p) => ({ n, s, t: t || '', p });
      const GLM = (i) => {
        const L = [E('hc_attn_{fn,base,scale}', '[24,16384]·[24]·[3]', 'BF16/F32', 'hc1'), E('input_layernorm.weight', '[4096]', 'BF16', 'norm1')];
        if (i % 4 === 3 && i < 44) {
          L.push(E('self_attn.q_a_proj.weight', '[1536, 4096]', 'BF16', 'attn-in'), E('self_attn.q_a_layernorm.weight', '[1536]', 'BF16', 'attn-in'), E('self_attn.q_b_proj.weight', '[16384, 1536]', 'BF16', 'attn-in'), E('self_attn.kv_a_proj_with_mqa.weight', '[512, 4096]', 'BF16', 'attn-in'), E('self_attn.kv_a_layernorm.weight', '[512]', 'BF16', 'attn-in'), E('self_attn.kv_b_proj.weight', '[32768, 512]', 'BF16', 'attn-in'), E('self_attn.indexer.{wq_b,wk}.weight', '[4096,1536]·[128,4096]', 'BF16', 'attn-in'), E('self_attn.indexer.{k_norm,weights_proj}', '[128]·[32,4096]', 'BF16', 'attn-in'), E('self_attn.indexer.index_kpool_compress_{gate,ape}', '[128,4096]·[4,128]', 'BF16', 'attn-in'), E('self_attn.o_proj.weight', '[4096, 16384]', 'BF16', 'attn-out'));
        } else {
          L.push(E('self_attn.{q,k,v}_proj.weight', '[8192, 4096] ×3', 'BF16', 'attn-in'), E('self_attn.{q,k,v}_conv1d.weight', '[8192, 1, 4] ×3', 'BF16', 'attn-in'), E('self_attn.b_proj.weight', '[64, 4096]', 'BF16', 'attn-in'), E('self_attn.{f_a,f_b}_proj.weight', '[128,4096]·[8192,128]', 'BF16', 'attn-in'), E('self_attn.{A_log, dt_bias}', '[64]·[8192]', 'F32', 'attn-in'), E('self_attn.{g_a,g_b}_proj.weight', '[128,4096]·[8192,128]', 'BF16', 'attn-in'), E('self_attn.o_norm.weight', '[128]', 'BF16', 'attn-in'), E('self_attn.o_proj.weight', '[4096, 8192]', 'BF16', 'attn-out'));
        }
        L.push(E('hc_ffn_{fn,base,scale}', '[24,16384]·[24]·[3]', 'BF16/F32', 'hc2'), E('post_attention_layernorm.weight', '[4096]', 'BF16', 'norm2'));
        if (i < 3) L.push(E('mlp.{gate,up}_proj.weight', '[12288, 4096] ×2', 'BF16', 'ffn-in'), E('mlp.down_proj.weight', '[4096, 12288]', 'BF16', 'ffn-out'));
        else L.push(E('mlp.gate.weight', '[288, 4096]', 'BF16', 'router'), E('mlp.gate.e_score_correction_bias', '[288]', 'F32', 'router'), E('mlp.shared_experts.{gate,up}_proj.weight', '[2048, 4096] ×2', 'BF16', 'ffn-in'), E('mlp.shared_experts.down_proj.weight', '[4096, 2048]', 'BF16', 'ffn-out'), E('mlp.experts.{0..287}.{gate,up}_proj.trellis (+suh,svh,mcg)', '[256,128,64]', 'I16 EXL3', 'experts'), E('mlp.experts.{0..287}.down_proj.trellis (+suh,svh,mcg)', '[128,256,64]', 'I16 EXL3', 'experts'));
        return L;
      };
      const QW = (i) => {
        const L = [E('attn_hyper_connection.{hc_norm,input_mix_*,block_inject}', '[10240]·[320,10240]·…', 'BF16', 'hc1')];
        if (i === 1) L.push(E('ple.ple_embedding.ngram_embedding.shard_{0..127}.weight', '[2500012, 160]', 'F8', 'ple'), E('ple.{key_proj,value_proj,conv1d,norm_*}', '[10240,2560]·[2560,2560]·…', 'BF16', 'ple'));
        if ((i + 1) % 4 === 0) L.push(E('self_attn.q_proj.weight  (query + gate)', '[12288, 2560]', 'BF16', 'attn-in'), E('self_attn.{k,v}_proj.weight', '[512, 2560] ×2', 'BF16', 'attn-in'), E('self_attn.{q,k}_norm.weight', '[256] ×2', 'BF16', 'attn-in'), E('self_attn.indexer.index_qk_proj.weight', '[640, 2560]', 'BF16', 'attn-in'), E('self_attn.indexer.{q,k}_layernorm.weight', '[128] ×2', 'BF16', 'attn-in'), E('self_attn.o_proj.weight', '[2560, 6144]', 'BF16', 'attn-out'));
        else L.push(E('linear_attn.in_proj_qkv.weight', '[10240, 2560]', 'BF16', 'attn-in'), E('linear_attn.in_proj_z.weight', '[6144, 2560]', 'BF16', 'attn-in'), E('linear_attn.in_proj_{a,b}.weight', '[48, 2560] ×2', 'BF16', 'attn-in'), E('linear_attn.conv1d.weight', '[10240, 1, 4]', 'BF16', 'attn-in'), E('linear_attn.{A_log, dt_bias}', '[48] ×2', 'BF16', 'attn-in'), E('linear_attn.norm.weight', '[128]', 'BF16', 'attn-in'), E('linear_attn.out_proj.weight', '[2560, 6144]', 'BF16', 'attn-out'));
        L.push(E('mlp_hyper_connection.{hc_norm,input_mix_*,block_inject}', '[10240]·[320,10240]·…', 'BF16', 'hc2'), E('mlp.gate.weight', '[512, 2560]', 'BF16', 'router'), E('mlp.shared_expert.{gate,up}_proj.weight', '[640, 2560] ×2', 'BF16', 'ffn-in'), E('mlp.shared_expert.down_proj.weight', '[2560, 640]', 'BF16', 'ffn-out'), E('mlp.shared_expert_gate.weight', '[1, 2560]', 'BF16', 'ffn-in'), E('mlp.experts.{0..511}.{gate,up}_proj.weight (+scales)', '[640, 1280]', 'U8 2×FP4', 'experts'), E('mlp.experts.{0..511}.down_proj.weight (+scales)', '[2560, 320]', 'U8 2×FP4', 'experts'));
        return L;
      };
      const DS = (i) => {
        const L = [E('hc_attn_{fn,base,scale}', '[24,16384]·[24]·[3]', 'F32', 'hc1'), E('attn_norm.weight', '[4096]', '', 'norm1'), E('attn.wq_a.weight (+ .scale)', '[1024, 4096]', 'F8', 'attn-in'), E('attn.q_norm.weight', '[1024]', '', 'attn-in'), E('attn.wq_b.weight', '[32768, 1024]', 'F8', 'attn-in'), E('attn.wkv.weight', '[512, 4096]', 'F8', 'attn-in'), E('attn.kv_norm.weight', '[512]', '', 'attn-in'), E('attn.attn_sink', '[64]', 'F32', 'attn-in')];
        if (i >= 2) { const c4 = i % 2 === 0; L.push(E('attn.compressor.{wkv,wgate,ape,norm}', c4 ? '[1024,4096]×2·[4,1024]·[512]' : '[512,4096]×2·[128,512]·[512]', 'BF16', 'attn-in')); if (c4) L.push(E('attn.indexer.{wq_b,weights_proj,compressor.*}', '[8192,1024]·[64,4096]·…', '', 'attn-in')); }
        L.push(E('attn.wo_a.weight  (8 groups)', '[8192, 4096]', 'F8', 'attn-out'), E('attn.wo_b.weight', '[4096, 8192]', 'F8', 'attn-out'), E('hc_ffn_{fn,base,scale}', '[24,16384]·[24]·[3]', 'F32', 'hc2'), E('ffn_norm.weight', '[4096]', '', 'norm2'), E('ffn.gate.weight', '[256, 4096]', '', 'router'));
        if (i < 3) L.push(E('ffn.gate.tid2eid  (hash routing)', '[129280, 6]', 'I32', 'router')); else L.push(E('ffn.gate.bias  → e_score_correction_bias', '[256]', '', 'router'));
        L.push(E('ffn.shared_experts.{w1,w3}.weight  → gate_up_proj', '[2048, 4096] ×2', 'F8', 'ffn-in'), E('ffn.shared_experts.w2.weight  → down_proj', '[4096, 2048]', 'F8', 'ffn-out'), E('ffn.experts.{0..255}.{w1,w3}.weight (+ .scale)', '[2048, 4096]', 'FP4', 'experts'), E('ffn.experts.{0..255}.w2.weight (+ .scale)', '[4096, 2048]', 'FP4', 'experts'));
        return L;
      };
      const CL = () => [E('input_layernorm.weight', '[d]', '', 'norm1'), E('self_attn.q_proj.weight', '[h·dₕ, d]', '', 'attn-in'), E('self_attn.k_proj.weight', '[n_kv·dₕ, d]', '', 'attn-in'), E('self_attn.v_proj.weight', '[n_kv·dₕ, d]', '', 'attn-in'), E('self_attn.o_proj.weight', '[d, h·dₕ]', '', 'attn-out'), E('post_attention_layernorm.weight', '[d]', '', 'norm2'), E('mlp.gate_proj.weight', '[I, d]', '', 'ffn-in'), E('mlp.up_proj.weight', '[I, d]', '', 'ffn-in'), E('mlp.down_proj.weight', '[d, I]', '', 'ffn-out')];
      const MOD = {
        cl: { N: 32, pre: 'model.layers.{i}.', gl: 'model.embed_tokens [V,d] · model.norm [d] · lm_head [V,d]', gen: CL, blk: () => ({ hc: 'none', norms: true, mixer: 'attention (MHA/GQA)', ffn: 'SwiGLU FFN', moe: false, attnIn: 'q/k/v_proj', attnOut: 'o_proj', ffnIn: 'gate/up_proj', ffnOut: 'down_proj' }), col: 'a' },
        glm: { N: 45, pre: 'model.language_model.layers.{i}.', gl: 'embed_tokens [154880,4096] · norm [4096] · lm_head [154880,4096]', gen: GLM, blk: (i) => ({ hc: 'mhc', norms: true, mixer: i % 4 === 3 && i < 44 ? 'MLA NoPE + indexer' : 'KDA (linear attention)', ffn: i < 3 ? 'dense SwiGLU FFN (12288)' : 'MoE: 288 experts + 1 shared', moe: i >= 3, attnIn: i % 4 === 3 && i < 44 ? 'latent q/kv + idx' : 'q/k/v + gates', attnOut: 'o_proj' }), col: 'glm' },
        qwen: { N: 48, pre: 'model.language_model.layers.{i}.', gl: 'embed_tokens [248320,2560] · hyper_connection_mixer · lm_head [248320,2560]', gen: QW, blk: (i) => ({ hc: 'gr', norms: false, mixer: (i + 1) % 4 === 0 ? 'QSA (sparse attention)' : 'Gated DeltaNet', ffn: 'MoE: 512 experts + 1 shared', moe: true, attnIn: (i + 1) % 4 === 0 ? 'q(+gate)/k/v + indexer' : 'in_proj_qkv/z/a/b', attnOut: (i + 1) % 4 === 0 ? 'o_proj' : 'out_proj' }), col: 'qwen' },
        ds: { N: 43, pre: 'layers.{i}.  (vLLM: model.layers.{i}.)', gl: 'embed → embed_tokens [129280,4096] · norm · hc_head · head → lm_head', gen: DS, blk: (i) => ({ hc: 'mhc', norms: true, n1: 'attn_norm', n2: 'ffn_norm', mixer: i < 2 ? 'SWA attention (window 128)' : i % 2 === 0 ? 'C4A attention + indexer' : 'C128A attention', ffn: i < 3 ? 'hash MoE (tid2eid)' : 'MoE: 256 experts + 1 shared', moe: true, attnIn: 'wq_a/wq_b/wkv', attnOut: 'wo_a/wo_b', ffnIn: 'w1/w3', ffnOut: 'w2' }), col: 'ds' },
      };
      const S = { m: 'glm', layer: 3, cur: 0, hov: -1 };
      const ctl = UI.controls(fig);
      let lsl;
      UI.seg(ctl, { label: 'Model', options: [{ v: 'cl', label: 'Classic' }, { v: 'glm', label: 'GLM' }, { v: 'qwen', label: 'Qwen' }, { v: 'ds', label: 'DeepSeek' }], value: 'glm', onChange: (v) => { S.m = v; const N = MOD[v].N; lsl.input.max = N - 1; if (S.layer > N - 1) lsl.set(N - 1); S.cur = 0; } });
      lsl = UI.slider(ctl, { label: 'Layer', min: 0, max: 44, step: 1, value: 3, onInput: (v) => { S.layer = v; S.cur = 0; } });
      V.every(1500, () => { S.cur++; });
      F(fig, (g, w, h, t, dt, _s, Hh) => {
        const C = V.C();
        const M = MOD[S.m];
        const li = Math.min(S.layer, M.N - 1);
        const L = M.gen(li);
        const narrow = w < 620;
        const mc = C[M.col] || C.a;
        let cur = S.cur % L.length;
        const pre = M.pre.replace('{i}', li);
        T(g, pre, 14, 20, { size: fit(g, pre, w - 28, 12.5, 'mono', 600), font: 'mono', weight: 600, color: mc });
        wrap(g, 'globals: ' + M.gl, 14, 38, w - 28, 14, { size: 11, font: 'mono', color: C.muted, maxLines: 2 });
        // block and list area
        let bR, lR;
        if (!narrow) { bR = { x: 14, y: 70, w: 230, h: h - 90 }; lR = { x: 262, y: 64, w: w - 276, h: h - 72 }; }
        else { bR = { x: 14, y: 76, w: w - 28, h: 230 }; lR = { x: 14, y: 328, w: w - 28, h: h - 336 }; }
        const lineH = narrow ? 34 : 25;
        const visN = Math.max(3, Math.floor(lR.h / lineH));
        // hover on the list
        let hov = -1;
        const off0 = V.clamp(cur - Math.floor(visN / 2), 0, Math.max(0, L.length - visN));
        if (Hh.pointer.inside && Hh.pointer.x >= lR.x && Hh.pointer.y >= lR.y) { const k = Math.floor((Hh.pointer.y - lR.y) / lineH) + off0; if (k >= 0 && k < L.length && k < off0 + visN) hov = k; }
        if (hov >= 0) cur = hov;
        const off = hov >= 0 ? off0 : V.clamp(cur - Math.floor(visN / 2), 0, Math.max(0, L.length - visN));
        const lit = {}; lit[L[cur].p] = true;
        const bo = M.blk(li); bo.lit = lit;
        drawBlock(g, bR.x, bR.y, bR.w, bR.h, bo);
        // list
        for (let k = off; k < Math.min(L.length, off + visN); k++) {
          const e = L[k];
          const y = lR.y + (k - off) * lineH;
          const on = k === cur;
          if (on) V.box(g, lR.x - 4, y, lR.w + 4, lineH - 3, { r: 6, fill: V.alpha(mc, 0.16), stroke: V.alpha(mc, 0.7), lw: 1.2 });
          const pc = { hc1: C.f, hc2: C.f, norm1: C.e, norm2: C.e, 'attn-in': C.b, 'attn-out': C.c, router: C.d, 'ffn-in': C.a, 'ffn-out': C.c, experts: C.a, ple: C.d }[e.p] || C.muted;
          g.fillStyle = pc; V.rr(g, lR.x, y + 7, 4, lineH - 16, 2); g.fill();
          if (!narrow) {
            const shW = Math.min(250, lR.w * 0.38);
            const nameW = lR.w - shW - 14;
            T(g, e.n, lR.x + 10, y + 15, { size: fit(g, e.n, nameW, 12, 'mono'), font: 'mono', color: on ? C.ink : C.text2 || C.ink, maxW: nameW });
            T(g, e.s + (e.t ? '  ' + e.t : ''), lR.x + lR.w - 2, y + 15, { size: fit(g, e.s + (e.t ? '  ' + e.t : ''), shW, 11.5, 'mono'), font: 'mono', align: 'right', color: on ? mc : C.muted, maxW: shW });
          } else {
            T(g, e.n, lR.x + 10, y + 13, { size: fit(g, e.n, lR.w - 14, 11.5, 'mono'), font: 'mono', color: on ? C.ink : C.muted, maxW: lR.w - 14 });
            T(g, e.s + (e.t ? '  ' + e.t : ''), lR.x + 10, y + 27, { size: 11, font: 'mono', color: on ? mc : C.muted, maxW: lR.w - 14 });
          }
        }
        if (off > 0) T(g, '▲ ' + off + ' above', lR.x + lR.w, lR.y - 4, { size: 11, align: 'right', color: C.muted });
        if (off + visN < L.length) T(g, '▼ ' + (L.length - off - visN) + ' below', lR.x + lR.w, h - 4, { size: 11, align: 'right', color: C.muted });
      });
    });

    /* ====================================================================
       Figure 10 — Roadwork map: each patch on a piece of the block
       ==================================================================== */
    safe('obras', () => {
      const fig = $('#fig-obras');
      fitH(fig, (w) => (w < 620 ? 700 : 460));
      const PINS = {
        glm: {
          blk: { hc: 'mhc', norms: true, mixer: 'KDA ×34 / MLA NoPE ×11', ffn: 'MoE EXL3 (288) · dense in 0–2', moe: true, attnIn: 'readers', attnOut: 'o_proj' },
          pins: [
            { p: 'hc2', t: 'EAGLE3 hooks on the mHC', d: 'patch_glm_eagle3.py: at the outputs of layers 5, 14, 24, 33 and 42, materializes hc_post and does hc_contract (average of the 4 streams) to hand DFlash2 a 4096-wide vector.', ch: 'Block drafting' },
            { p: 'attn-in', t: 'NoPE sparse-MLA overlay (SM121)', d: 'The fp8_ds_mla backend required pe_dim = 64, but GLM is NoPE (qk_rope_head_dim = 0). Inline patches in the Dockerfile pad in 64 zeros and instantiate the sparse kernel for the GB10.', ch: 'MLA / quantized KV' },
            { p: 'mixer', t: 'State of the 34 KDA layers', d: 'The fixed recurrent state of KDA lives in vLLM\'s “Mamba” cache; the patch_mamba_align_* patches handle chunk alignment and state release.', ch: 'Linear attention' },
            { p: 'attn-out', t: 'ABLIT=1 on o_proj', d: 'ablit_runtime.py edits, at load time, the o_proj of layers 15–45 (donor transplant or refusal-direction projection): the attention writer.', ch: 'Abliteration' },
            { p: 'ffn-in', t: 'GLM53_DENSE_FP8 (norms untouched)', d: 'Quantizes at load time, from BF16 to per-channel FP8, the dense projections of the shared, dense, kda and mla groups, and runs them with Marlin W8A16. The RMSNorms (ε = 1e-5) stay in BF16.', ch: 'FP8 / Kernels' },
            { p: 'experts', t: 'EXL3 experts at 4 bpw', d: 'overlay/exl3.py registers the exl3 method: trellis + suh/svh + mcg per matrix, E2/E3 kernels and the cooperative decode MoE (H = 4096).', ch: 'Hadamard, trellises & EXL3' },
            { p: 'ffn-in', t: 'SWIGLU_LIMIT_DEFAULT = 10.0', d: 'The SwiGLU clamp (L = 10) is replicated inside the EXL3 kernels: silu(min(g,10))·clamp(u,±10) in the PyTorch paths and in E3 (in the epilogue of the gate/up GEMM); min(silu(g),10)·clamp(u,±10) in the kernels inherited from exllamav3 and in the TP3 cooperative kernel, which only accepts limit 10.', ch: 'this chapter' },
            { p: 'ffn-out', t: 'E3: down + scatter', d: 'EXL3_FAT_GROUPED=1: all of the layer\'s “fat” experts (many tokens per expert, typical of prefill) in three launches (gather; gate/up + SwiGLU; down + scatter), with no CPU sync and capturable in a CUDA graph. The thin ones stay on exl3_moe.', ch: 'Kernels' },
          ],
        },
        qwen: {
          blk: { hc: 'gr', norms: false, mixer: 'GDN ×36 / QSA ×12', ffn: 'MoE NVFP4 (512) + shared', moe: true, attnIn: 'readers', attnOut: 'o_proj/out_proj' },
          pins: [
            { p: 'hc1', t: 'Replicated Gated Residual', d: 'The GR modules are BF16 and replicated on both GPUs (disable_tp): about 1.19 GiB read per step, 13% of the traffic of a decode step.', ch: 'Hyper-connections' },
            { p: 'hc2', t: 'PLE: an extra writer at layer 1', d: 'detect_ple_dtype.py + --hf-overrides + ple_layer_patched.py load the FP8 N-gram table (51.2 B parameters), which adds its vector to the 4 branches at the layer with index 1.', ch: 'N-gram Memory' },
            { p: 'attn-in', t: 'FP8 KV in the QSA layers', d: 'patch_qsa_fp8_kv.py teaches the QSA kernels to read K/V in FP8 in the 13 layers with KV (12 + MTP): 1.70× tokens per GiB.', ch: 'Quantized KV cache' },
            { p: 'mixer', t: 'GDN MXFP8 fallback', d: 'In the alternative checkpoint with attention/GDN in MXFP8, the in_proj_a/b [48, 2560] projections are too narrow for the MXFP8 kernel (N = 48 < 128): the loader falls back to BF16 emulation.', ch: 'FP8 & microscaling' },
            { p: 'attn-out', t: 'ABLIT=1: o_proj of the 12 QSA', d: 'The abliterated checkpoint changes only the o_proj of the 12 QSA layers; the rest of the checkpoint is identical.', ch: 'Abliteration' },
            { p: 'experts', t: 'NVFP4 experts + EP', d: '256 whole experts per GPU (expert parallel). The MTP experts are FP8 in 128×128 blocks (patch_modelopt_fp8_block_moe.py); 640/2 = 320 would not fit those blocks.', ch: 'Parallelism' },
            { p: 'ffn-in', t: 'FP8_DENSE (experimental)', d: 'Dense projections BF16 → FP8; the router, shared_expert_gate, norms and gates stay BF16 (EXCLUDE_MODULES).', ch: 'Checkpoints' },
            { p: 'head', t: 'Reduced draft vocabulary', d: 'The 248,320 × 2560 lm_head (1.18 GiB) is read 3× per step by MTP; patch_mtp_draft_vocab.py does the draft argmax over 47,149 ids.', ch: 'MTP, EAGLE & vocabulary' },
          ],
        },
        ds: {
          blk: { hc: 'mhc', norms: true, n1: 'attn_norm', n2: 'ffn_norm', mixer: 'SWA / C4A / C128A', ffn: 'MoE MXFP4 (256) + shared', moe: true, attnIn: 'wq_a/wq_b/wkv', attnOut: 'wo_a/wo_b', ffnIn: 'w1/w3', ffnOut: 'w2' },
          pins: [
            { p: 'hc1', t: 'b12x mHC', d: 'VLLM_USE_B12X_MHC (optional) does the mHC pre/post in a b12x kernel that only accepts hidden_size = 4096; the overlay already fuses attn_norm/ffn_norm into the mHC.', ch: 'Hyper-connections' },
            { p: 'hc2', t: 'Capturing the DSpark features', d: 'At layers 40, 41 and 42, hc_post collapses the 4 streams and the average goes to the drafter\'s main_proj (12,288 → 4096).', ch: 'DFlash2 & DSpark' },
            { p: 'attn-in', t: 'nvfp4_ds_mla → FP8 kernel', d: 'Issue #22: the declared KV dtype is routed to the FP8 kernel; each compressed KV row is 584 B.', ch: 'Quantized KV cache' },
            { p: 'attn-out', t: 'Runtime ablation', d: 'With ABLATE=1, right after the attention output: x ← x − λ(x·v)v, with a validated 4096-dimensional direction (finite, norm > 0).', ch: 'Abliteration' },
            { p: 'attn-out', t: 'wo_b row-parallel', d: 'The grouped output (wo_a per group, wo_b row-parallel) costs 43 all-reduces per step, plus 43 from the MoE. TP = 3 requires padding the groups 8 → 9.', ch: 'TP=3 & TP=4' },
            { p: 'router', t: 'Hash routing (0–2)', d: 'tid2eid picks the experts by token id: it requires raw input_ids; the image tokens needed bias_vl (#175).', ch: 'Mixture-of-Experts' },
            { p: 'experts', t: 'b12x W4A16 on the experts', d: '--moe-backend flashinfer_b12x: MXFP4 experts (e2m1 + E8M0/32) dequantized to BF16; it receives swiglu_limit = 10 as gemm1_clamp_limit.', ch: 'Kernels' },
          ],
        },
      };
      const S = { m: 'glm', lock: -1 };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Recipe', options: [{ v: 'glm', label: 'GLM' }, { v: 'qwen', label: 'Qwen' }, { v: 'ds', label: 'DeepSeek' }], value: 'glm', onChange: (v) => { S.m = v; S.lock = -1; } });
      UI.button(ctl, 'automatic', () => { S.lock = -1; });
      F(fig, (g, w, h, t, dt, _s, Hh) => {
        const C = V.C();
        const D = PINS[S.m];
        const mc = C[S.m] || C.a;
        const narrow = w < 620;
        let bR, pR;
        if (!narrow) { bR = { x: 14, y: 50, w: Math.min(310, w * 0.37), h: h - 92 }; pR = { x: bR.x + bR.w + 84, y: 16, w: w - bR.x - bR.w - 98, h: h - 26 }; }
        else { bR = { x: 14, y: 54, w: w - 100, h: 330 }; pR = { x: 14, y: 430, w: w - 28, h: h - 440 }; }
        const auto = Math.floor(t / 3.6) % D.pins.length;
        // embedding and lm_head around the block
        V.box(g, bR.x + 34, bR.y - 34, bR.w - 34, 24, { r: 6, fill: V.alpha(C.d, 0.12), stroke: V.alpha(C.d, 0.5), lw: 1, label: 'embedding', size: 11.5 });
        const headR = { x: bR.x + 34, y: bR.y + bR.h + 14, w: bR.w - 34, h: 24 }; // outside the block
        // current target (automatic, pinned or hover)
        let sel = S.lock >= 0 ? S.lock : auto;
        // first draw to get the rectangles (no highlight), then the pins
        const lit0 = {};
        const rects = (() => { const o = Object.assign({}, D.blk, { lit: lit0 }); g.save(); g.globalAlpha = 0; const r = drawBlock(g, bR.x, bR.y, bR.w, bR.h, o); g.restore(); return r; })();
        rects.head = headR;
        const pinPos = [];
        D.pins.forEach((p) => {
          const r = rects[p.p] || rects.mixer;
          const y = r.y + r.h / 2;
          let col = 0;
          while (col < 2 && pinPos.some(([px, py]) => Math.abs(py - y) < 22 && px === bR.x + bR.w + 14 + col * 24)) col++;
          pinPos.push([bR.x + bR.w + 14 + col * 24, y]);
        });
        let hov = -1;
        if (Hh.pointer.inside) pinPos.forEach(([x, y], k) => { if (Math.hypot(Hh.pointer.x - x, Hh.pointer.y - y) < 13) hov = k; });
        if (Hh.pointer.clicked && hov >= 0) S.lock = hov;
        if (hov >= 0) sel = hov;
        const lit = {}; lit[D.pins[sel].p] = true;
        drawBlock(g, bR.x, bR.y, bR.w, bR.h, Object.assign({}, D.blk, { lit }));
        V.box(g, headR.x, headR.y, headR.w, headR.h, { r: 6, fill: V.alpha(C.c, lit.head ? 0.34 : 0.12), stroke: V.alpha(C.c, lit.head ? 1 : 0.5), lw: lit.head ? 2.2 : 1, label: 'final norm → lm_head', size: 11.5 });
        T(g, '× ' + { glm: 45, qwen: 48, ds: 43 }[S.m] + ' blocks', bR.x + bR.w, bR.y - 40, { size: 11, align: 'right', color: C.muted, font: 'mono' });
        pinPos.forEach(([x, y], k) => {
          const on = k === sel;
          const r = rects[D.pins[k].p] || rects.mixer;
          V.line(g, bR.x + bR.w + 1, y, x - 10, y, { color: V.alpha(mc, on ? 0.9 : 0.3), lw: 1, dash: [2, 3] });
          if (on) V.line(g, Math.min(r.x + r.w, bR.x + bR.w), y, bR.x + bR.w, y, { color: V.alpha(mc, 0.9), lw: 1.2 });
          g.beginPath(); g.arc(x, y, on ? 11 : 9, 0, 6.3); g.fillStyle = on ? mc : V.alpha(mc, 0.3); g.fill();
          if (on) { g.strokeStyle = C.ink; g.lineWidth = 1.5; g.stroke(); V.glow(g, x, y, 22 + 3 * Math.sin(t * 5), mc, 0.4); }
          T(g, String(k + 1), x, y + 4, { size: 11.5, align: 'center', weight: 700, color: on ? onTxt(C) : C.ink });
        });
        // description panel
        const p = D.pins[sel];
        V.box(g, pR.x, pR.y, pR.w, pR.h, { r: 10, fill: V.alpha(C.panel, 0.85), stroke: V.alpha(mc, 0.6), lw: 1.3 });
        T(g, (sel + 1) + ' / ' + D.pins.length + (S.lock >= 0 ? ' (pinned)' : ''), pR.x + pR.w - 12, pR.y + 20, { size: 11, align: 'right', color: C.muted, font: 'mono' });
        let yy = pR.y + 22;
        yy += 18 * wrap(g, p.t, pR.x + 14, yy, pR.w - 80, 18, { size: 14, weight: 700, color: mc, maxLines: 2 });
        yy += 6;
        yy += 17 * wrap(g, p.d, pR.x + 14, yy, pR.w - 28, 17, { size: 12, color: C.ink, maxLines: narrow ? 8 : 9 });
        const pn = { hc1: 'bus (read)', hc2: 'bus (write/mix)', norm1: 'norms', 'attn-in': 'attention readers', 'attn-out': 'attention writer', mixer: 'token mixer', router: 'router', 'ffn-in': 'FFN readers (gate/up)', 'ffn-out': 'FFN writer (down_proj)', experts: 'experts', head: 'outside the block: final norm → lm_head' }[p.p] || p.p;
        T(g, 'block piece: ' + pn, pR.x + 14, pR.y + pR.h - 30, { size: 11.5, color: C.muted, maxW: pR.w - 28 });
        T(g, 'deep dive: ' + p.ch, pR.x + 14, pR.y + pR.h - 12, { size: 11.5, color: C.muted, maxW: pR.w - 28 });
        // compact list (wide screens only)
        if (yy + 20 < pR.y + pR.h - 50) {
          const ly0 = yy + 16;
          const fitN = Math.floor((pR.y + pR.h - 56 - ly0) / 16) + 1;
          const nShow = fitN >= D.pins.length ? D.pins.length : Math.max(1, fitN - 1);
          D.pins.slice(0, nShow).forEach((q, k) => { T(g, (k + 1) + '. ' + q.t, pR.x + 14, ly0 + k * 16, { size: 11.5, color: k === sel ? mc : C.muted, maxW: pR.w - 28 }); });
          if (nShow < D.pins.length) T(g, '… +' + (D.pins.length - nShow) + ' (tap the numbered markers)', pR.x + 14, ly0 + nShow * 16, { size: 11.5, color: C.muted, maxW: pR.w - 28 });
        }
      });
    });
  },
});

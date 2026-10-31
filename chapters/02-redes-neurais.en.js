LA.chapter({
  id: 'redes-neurais',
  part: 1,
  order: 2,
  title: 'Neural networks: neurons, layers and backprop',
  navTitle: 'Neural networks',
  subtitle: 'From the neuron that draws a line to the deep stack that learns: linear layers as geometry, activations, multiplicative gates, backpropagation, normalization and residual connections, with the real values from the three recipes.',
  minutes: 60,
  recipes: ['glm', 'qwen', 'ds'],
  tags: ['neuron', 'linear layer', 'ReLU', 'GELU', 'SiLU', 'sigmoid', 'softplus', 'gating', 'SwiGLU', 'backprop', 'LayerNorm', 'RMSNorm', 'rms_norm_eps', 'residual connection'],
  goals: [
    'Describe a neuron and a linear layer as <b>geometry</b>: hyperplanes, matrices that bend the plane and why stacking linear layers is not enough',
    'Recognize the activations (ReLU, GELU, SiLU, sigmoid, softplus, tanh) and the <b>multiplicative gates</b>, and say where each one appears in the three recipes',
    'Work out by hand the forward and backward passes of <b>backpropagation</b> on a small graph, and count the parameters of an MLP and of a SwiGLU',
    'Explain LayerNorm, RMSNorm, the role of <code>rms_norm_eps</code> and of the <code>1 + w</code> gain convention, and why deep networks need residual connections',
  ],
  prereqs: ['aprender-com-dados'],
  summary: [
    'A neuron computes \\(f(\\mathbf w^\\top\\mathbf x + b)\\): which side of a hyperplane the point is on, and how far away. A linear layer \\(W\\mathbf x+\\mathbf b\\) stacks neurons as the rows of an <code>[out, in]</code> matrix and costs \\(2mn\\) FLOPs per token.',
    'Composed linear maps stay linear; activations fold space. SiLU is the <code>hidden_act</code> of the three models, sigmoid is the gate function, softplus produces positive rates, ReLU scores the sparse indexers and GELU appears in Qwen\'s ViT and DeepSeek\'s Aligner (the ViTs of GLM and DeepSeek use SwiGLU).',
    'An MLP with enough neurons approximates any continuous function; depth composes folds more economically. Multiplicative gates (GLU, SwiGLU) let the data decide how much passes, and Qwen3.8 uses them in almost every sublayer.',
    'Backprop applies the chain rule from back to front and delivers every gradient in a single pass: \\(\\boldsymbol\\delta = f\'(\\mathbf z)\\odot W_{\\text{next}}^\\top\\boldsymbol\\delta_{\\text{next}}\\). At inference it is turned off, but everything it shaped (activation, ε, gates) has to be reproduced faithfully.',
    'RMSNorm divides by the RMS, without centering, and is scale invariant; ε dominates below \\(\\mathrm{RMS}\\approx\\sqrt\\varepsilon\\). Real values: GLM 1e-5 (vision corrected to 1e-6), Qwen 1e-6 with gain \\(1+w\\) and a norm grouped per branch, DeepSeek 1e-20 in the Vision-Exp trunk. Norms compute in FP32 and stay in BF16 when quantizing.',
    'Without a residual, the gradient is a product of Jacobians that vanishes or explodes with depth; \\(\\mathbf h + F(\\mathrm{Norm}(\\mathbf h))\\) creates an express lane. The three models widen this residual to 4 streams (mHC in GLM and DeepSeek, Gated Residual in Qwen).',
  ],
  quiz: [
    { q: 'You stack 10 linear layers \\(W_\\ell\\mathbf x + \\mathbf b_\\ell\\) with no activation between them. What do you get?', o: ['A network capable of separating XOR, if the matrices are large enough', 'A single equivalent affine transformation, with matrix \\(W_{10}\\cdots W_1\\)', 'An unstable network, because the gradient explodes', 'A 10-layer universal MLP'], a: 1, e: 'A composition of affine transformations is affine. Without a nonlinearity, lines stay lines and depth adds no expressiveness. That is why every network interleaves matrices with activations.' },
    { q: 'Why is the sigmoid the preferred function for the gates in the recipes, even though it fell out of use as a hidden-layer activation?', o: ['Because it is the cheapest to compute', 'Because it does not saturate', 'Because its derivative is always 1', 'Because its output lies in (0, 1): it says “how much to let through” and never amplifies the signal'], a: 3, e: 'As a hidden activation, saturation (derivative ≤ 0.25) gets in the way of the gradient. As a gate, the (0, 1) range is exactly what you want: how open a faucet is. When amplification is needed, 2·σ is used, as in the <code>post</code> of mHC and in the Gated Residual write.' },
    { q: 'In the graph \\(L = \\tfrac12(\\sigma(wx+b)-y)^2\\) with \\(x = 1.5\\), backprop reached \\(\\partial L/\\partial z \\approx -0.0736\\). What is \\(\\partial L/\\partial w\\)?', o: ['−0.1104', '−0.0736', '+0.1104', '−0.0491'], a: 0, e: 'Since \\(z = wx + b\\), the local derivative of \\(z\\) with respect to \\(w\\) is \\(x\\) (the × node “swaps” the factors). By the chain rule, \\(\\partial L/\\partial w = x\\cdot\\partial L/\\partial z = 1.5 \\times (-0.0736) \\approx -0.1104\\). A negative gradient: increasing \\(w\\) lowers the loss.' },
    { q: 'Qwen3.8 stores the weight of its RMSNorms initialized at zero and applies it as \\(1 + w\\). What happens if a loader applies that weight directly as the gain?', o: ['Nothing: it is just a naming convention', 'The model becomes twice as slow', 'The output of every norm is multiplied by numbers close to zero and the model generates garbage', 'ε takes over and the norm becomes the identity'], a: 2, e: 'With \\(w \\approx 0\\), the correct effective gain is \\(\\approx 1\\). Applied directly, the gain becomes \\(\\approx 0\\) and erases the signal in every layer. The tensor name (<code>*.weight</code>) does not give the convention away: you have to read the <code>forward</code> in the reference code.' },
    { q: 'An RMSNorm with \\(\\varepsilon = 10^{-5}\\) receives a vector with RMS \\(10^{-4}\\). Roughly what is the RMS of the normalized output (before the gain)?', o: ['1', '0.0316', '10⁻⁴', '0.5'], a: 1, e: '\\(\\mathrm{RMS}^2 = 10^{-8}\\), much smaller than ε. The output is \\(10^{-4}/\\sqrt{10^{-8}+10^{-5}} \\approx 0.0316\\). Below \\(\\mathrm{RMS}\\approx\\sqrt\\varepsilon\\approx 0.0032\\) ε dominates: the output stays at \\(\\approx \\mathbf x/\\sqrt\\varepsilon\\), still larger than the input, but it no longer reaches RMS 1. That is why the training ε has to be copied.' },
    { q: 'Why does \\(\\mathbf h_{\\ell+1} = \\mathbf h_\\ell + F_\\ell(\\mathbf h_\\ell)\\) fix the vanishing gradient in a 64-layer network?', o: ['Because it doubles the number of parameters', 'Because it removes the need for activations', 'Because it normalizes the activations automatically', 'Because the Jacobian of each layer becomes \\(I + \\partial F/\\partial\\mathbf h\\): the identity term carries the gradient to the first layer without multiplying it by small factors'], a: 3, e: 'The product of 64 Jacobians \\(I + \\partial F_\\ell/\\partial\\mathbf h\\) always contains the direct path \\(I\\cdot I\\cdots I\\). Normalization is a separate piece: without pre-norm, the simulation shows that the activations (and the gradient) still grow exponentially with ReLU.' },
  ],
  mount(root, LA) {
    const V = LA.viz, UI = LA.ui;
    const $ = (s) => root.querySelector(s);
    const T = (g, s, x, y, o) => V.text(g, s, x, y, o || {});
    const onTxt = (C) => (C.dark ? '#06140d' : '#ffffff');
    const nf = (x, d = 2) => (Number.isFinite(x) ? (+x).toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d }) : '—');
    const ni = (x) => Math.round(x).toLocaleString('en-US');
    const safe = (name, fn) => { try { fn(); } catch (e) { console.error('[redes-neurais] ' + name, e); } };
    // KaTeX: o MathML oculto (position:absolute) escapa das caixas roláveis em telas estreitas.
    root.querySelectorAll('.la-eq, .la-table-wrap').forEach((el) => { el.style.position = 'relative'; });

    function fitH(fig, fn) {
      const st = fig.querySelector('.la-fig-stage');
      const apply = () => { const w = st.getBoundingClientRect().width; if (!w) return; const hh = Math.round(fn(w)); if (Math.abs((parseFloat(st.style.height) || 0) - hh) > 1) st.style.height = hh + 'px'; };
      apply();
      const ro = new ResizeObserver(apply); ro.observe(fig);
      LA.onCleanup(() => ro.disconnect());
      return apply;
    }
    const measureCtx = document.createElement('canvas').getContext('2d');
    function textW(g, s, size, font, weight) { g.font = V.font(size || 12, font || 'sans', weight || 400); return g.measureText(String(s)).width; }
    function splitLines(g, s, maxW, o = {}) {
      g.font = V.font(o.size || 12, o.font || 'sans', o.weight || 400);
      const words = String(s).split(' ');
      const lines = []; let cur = '';
      for (const wd of words) { const tt = cur ? cur + ' ' + wd : wd; if (g.measureText(tt).width > maxW && cur) { lines.push(cur); cur = wd; } else cur = tt; }
      if (cur) lines.push(cur);
      return lines;
    }
    function wrap(g, s, x, y, maxW, lh, o = {}) {
      const ls = splitLines(g, s, maxW, o), mx = o.maxLines || 99;
      ls.slice(0, mx).forEach((l, i) => V.text(g, i === mx - 1 && ls.length > mx ? l + '…' : l, x, y + i * lh, o));
      return Math.min(ls.length, mx);
    }
    const rgbCache = {};
    function rgbOf(c) {
      if (rgbCache[c]) return rgbCache[c];
      let r = [128, 128, 128];
      if (c && c[0] === '#') { let hx = c.slice(1); if (hx.length === 3) hx = hx.split('').map((x) => x + x).join(''); const n = parseInt(hx.slice(0, 6), 16); r = [(n >> 16) & 255, (n >> 8) & 255, n & 255]; }
      else if (c && c.startsWith('rgb')) { const p = c.match(/[\d.]+/g); r = [+p[0], +p[1], +p[2]]; }
      rgbCache[c] = r; return r;
    }
    function mixA(c1, c2, t, a) { const p = rgbOf(c1), q = rgbOf(c2); return `rgba(${Math.round(p[0] + (q[0] - p[0]) * t)},${Math.round(p[1] + (q[1] - p[1]) * t)},${Math.round(p[2] + (q[2] - p[2]) * t)},${a})`; }
    const sig = (z) => 1 / (1 + Math.exp(-z));
    function erf(x) { const s = Math.sign(x); x = Math.abs(x); const t = 1 / (1 + 0.3275911 * x); const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x); return s * y; }
    const SQ2PI = Math.sqrt(2 / Math.PI);
    const ACT = {
      relu: { f: (z) => (z > 0 ? z : 0), d: (z) => (z > 0 ? 1 : 0) },
      sigmoid: { f: sig, d: (z) => { const s = sig(z); return s * (1 - s); } },
      tanh: { f: Math.tanh, d: (z) => { const t = Math.tanh(z); return 1 - t * t; } },
      softplus: { f: (z) => (z > 30 ? z : Math.log1p(Math.exp(z))), d: sig },
      gelu: { f: (z) => 0.5 * z * (1 + erf(z / Math.SQRT2)), d: (z) => 0.5 * (1 + erf(z / Math.SQRT2)) + (z * Math.exp(-z * z / 2)) / Math.sqrt(2 * Math.PI) },
      gelut: { f: (z) => 0.5 * z * (1 + Math.tanh(SQ2PI * (z + 0.044715 * z * z * z))), d: (z) => { const u = SQ2PI * (z + 0.044715 * z * z * z), th = Math.tanh(u); return 0.5 * (1 + th) + 0.5 * z * (1 - th * th) * SQ2PI * (1 + 3 * 0.044715 * z * z); } },
      silu: { f: (z) => z * sig(z), d: (z) => { const s = sig(z); return s * (1 + z * (1 - s)); } },
      linear: { f: (z) => z, d: () => 1 },
      step: { f: (z) => (z > 0 ? 1 : 0), d: () => 0 },
    };
    const SUP = { '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹', '-': '⁻' };
    const pow10 = (e) => (e === 0 ? '1' : '10' + String(e).split('').map((c) => SUP[c] || c).join(''));
    const sci = (x, d = 1) => { if (!Number.isFinite(x) || x === 0) return '0'; const e = Math.floor(Math.log10(Math.abs(x))); if (e >= -2 && e <= 3) return nf(x, e >= 2 ? 0 : 2); const m = x / Math.pow(10, e); return nf(m, d) + '×' + pow10(e); };
    // desenha uma setinha/pílula de rótulo
    function pill(g, s, x, y, o = {}) {
      const size = o.size || 11.5, font = o.font || 'mono', padX = o.padX == null ? 6 : o.padX;
      const tw = textW(g, s, size, font, o.weight || 500);
      const w = tw + padX * 2, h = size + 8;
      const x0 = o.align === 'right' ? x - w : o.align === 'center' ? x - w / 2 : x;
      V.box(g, x0, y - h / 2, w, h, { r: h / 2, fill: o.fill, stroke: o.stroke, lw: 1 });
      T(g, s, x0 + w / 2, y + 0.5, { size, font, weight: o.weight || 500, color: o.color, align: 'center', base: 'middle' });
      return w;
    }

    /* ====================================================================
       Figura 1 — O neurônio-porteiro
       ==================================================================== */
    safe('fig-neuronio', () => {
      const fig = $('#fig-neuronio');
      fitH(fig, (w) => (w < 600 ? Math.round(w + 250) : 400));
      const S = { w1: 1.0, w2: 0.8, b: -0.2, act: 'sigmoid' };
      const PTS = [
        [1.4, 0.6, 1], [0.8, 1.5, 1], [1.9, 1.3, 1], [0.5, 0.7, 1], [1.2, 2.1, 1], [2.2, 0.2, 1], [0.2, 1.6, 1], [1.6, -0.1, 1],
        [-1.2, -0.8, 0], [-0.5, -1.6, 0], [-1.9, -0.2, 0], [-0.3, -0.6, 0], [-1.5, -1.9, 0], [0.4, -1.7, 0], [-2.2, 0.6, 0], [-0.9, 0.3, 0],
      ];
      const ctl = UI.controls(fig);
      UI.slider(ctl, { label: 'w₁', min: -3, max: 3, step: 0.1, value: S.w1, fmt: (v) => nf(v, 1), onInput: (v) => { S.w1 = v; } });
      UI.slider(ctl, { label: 'w₂', min: -3, max: 3, step: 0.1, value: S.w2, fmt: (v) => nf(v, 1), onInput: (v) => { S.w2 = v; } });
      UI.slider(ctl, { label: 'bias b', min: -3, max: 3, step: 0.1, value: S.b, fmt: (v) => nf(v, 1), onInput: (v) => { S.b = v; } });
      UI.seg(ctl, { label: 'Activation f', options: [{ v: 'step', label: 'step' }, { v: 'sigmoid', label: 'sigmoid' }, { v: 'relu', label: 'ReLU' }, { v: 'linear', label: 'linear' }], value: S.act, onChange: (v) => { S.act = v; } });
      const rAcc = UI.readout(ctl, 'correct', '—');
      let lastAcc = -1;
      const R = 3, NG = 30;
      V.fig(fig, (g, w, h, t, dt, _s, HH) => {
        const C = V.C();
        const narrow = w < 600;
        const f = ACT[S.act].f;
        let ps, px, py, dx0, dy0, dw, dh;
        if (!narrow) { ps = Math.min(h - 46, w * 0.52); px = w - ps - 18; py = 32; dx0 = 18; dy0 = 30; dw = px - 44 - dx0; dh = h - 60; }
        else { ps = Math.min(w - 36, h - 250); px = (w - ps) / 2; py = h - ps - 16; dx0 = 14; dy0 = 30; dw = w - 28; dh = 190; }
        const toX = (u) => px + ((u + R) / (2 * R)) * ps, toY = (v) => py + ((R - v) / (2 * R)) * ps;
        const fromX = (X) => ((X - px) / ps) * 2 * R - R, fromY = (Y) => R - ((Y - py) / ps) * 2 * R;
        const colA = C.a, colB = C.c;
        // ---- mapa de calor da resposta
        const cs = ps / NG;
        for (let i = 0; i < NG; i++) for (let j = 0; j < NG; j++) {
          const u = -R + ((i + 0.5) * 2 * R) / NG, v = R - ((j + 0.5) * 2 * R) / NG;
          const z = S.w1 * u + S.w2 * v + S.b;
          let col;
          if (S.act === 'linear') { const k = Math.min(1, Math.abs(z) / 4); col = z >= 0 ? mixA(C.bg, colA, 1, 0.06 + 0.34 * k) : mixA(C.bg, colB, 1, 0.06 + 0.34 * k); }
          else if (S.act === 'relu') { const a = f(z); col = a > 0 ? mixA(C.bg, colA, 1, 0.1 + 0.36 * Math.min(1, a / 3)) : mixA(C.bg, colB, 1, 0.1); }
          else { const a = f(z); col = mixA(colB, colA, a, 0.1 + 0.3 * Math.abs(2 * a - 1)); }
          g.fillStyle = col; g.fillRect(px + i * cs, py + j * cs, cs + 0.6, cs + 0.6);
        }
        g.strokeStyle = V.alpha(C.muted, 0.35); g.lineWidth = 1; g.strokeRect(px, py, ps, ps);
        V.line(g, toX(-R), toY(0), toX(R), toY(0), { color: V.alpha(C.muted, 0.35), lw: 1 });
        V.line(g, toX(0), toY(-R), toX(0), toY(R), { color: V.alpha(C.muted, 0.35), lw: 1 });
        T(g, 'x₁', px + ps - 6, toY(0) - 6, { size: 11, font: 'mono', color: C.muted, align: 'right' });
        T(g, 'x₂', toX(0) + 6, py + 14, { size: 11, font: 'mono', color: C.muted });
        // ---- fronteira e normal
        const wn = Math.hypot(S.w1, S.w2);
        g.save(); g.beginPath(); g.rect(px, py, ps, ps); g.clip();
        if (wn > 1e-6) {
          const p0 = [(-S.b * S.w1) / (wn * wn), (-S.b * S.w2) / (wn * wn)], dir = [-S.w2 / wn, S.w1 / wn];
          V.line(g, toX(p0[0] - dir[0] * 10), toY(p0[1] - dir[1] * 10), toX(p0[0] + dir[0] * 10), toY(p0[1] + dir[1] * 10), { color: C.ink, lw: 2, dash: [7, 5], dashOffset: -t * 12 });
          const L = Math.min(1.8, 0.5 + wn * 0.45);
          V.arrow(g, toX(p0[0]), toY(p0[1]), toX(p0[0] + (S.w1 / wn) * L), toY(p0[1] + (S.w2 / wn) * L), { color: C.d, lw: 2.5, head: 10 });
          T(g, 'w', toX(p0[0] + (S.w1 / wn) * (L + 0.25)), toY(p0[1] + (S.w2 / wn) * (L + 0.25)) + 4, { size: 13, weight: 700, color: C.d, align: 'center' });
        }
        // pontos
        let acc = 0;
        PTS.forEach(([u, v, c]) => {
          const z = S.w1 * u + S.w2 * v + S.b, ok = (z > 0) === (c === 1);
          if (ok) acc++;
          const X = toX(u), Y = toY(v);
          V.dot(g, X, Y, 5.5, c ? colA : colB);
          g.strokeStyle = C.bg; g.lineWidth = 1.5; g.beginPath(); g.arc(X, Y, 5.5, 0, 7); g.stroke();
          if (!ok) { g.strokeStyle = C.red; g.lineWidth = 2; g.beginPath(); g.arc(X, Y, 9, 0, 7); g.stroke(); }
        });
        // convidado (sonda)
        const P = HH.pointer;
        let pu, pv;
        if (P.inside && P.x >= px && P.x <= px + ps && P.y >= py && P.y <= py + ps) { pu = fromX(P.x); pv = fromY(P.y); }
        else { pu = 2.1 * Math.cos(t * 0.45); pv = 1.8 * Math.sin(t * 0.7); }
        const pz = S.w1 * pu + S.w2 * pv + S.b, pa = f(pz);
        const PX = toX(pu), PY = toY(pv);
        V.glow(g, PX, PY, 18, C.d, 0.7);
        g.save(); g.translate(PX, PY); g.rotate(Math.PI / 4); g.fillStyle = C.d; g.fillRect(-5, -5, 10, 10); g.strokeStyle = C.bg; g.lineWidth = 1.5; g.strokeRect(-5, -5, 10, 10); g.restore();
        g.restore();
        if (acc !== lastAcc) { lastAcc = acc; rAcc.set(acc + ' / 16'); }
        T(g, 'a = f(w·x + b) at each point of the plane', px, py - 10, { size: 11.5, color: C.muted });
        // ---- grafo do neurônio
        const inX = dx0, boxW = 92, boxH = 42;
        const ys = [dy0 + dh * 0.16, dy0 + dh * 0.5, dy0 + dh * 0.84];
        const sumX = dx0 + dw * 0.47, sumY = dy0 + dh * 0.5, fX = dx0 + dw * 0.72, outX = dx0 + dw;
        const ins = [{ l: 'x₁', v: pu, w: S.w1, wl: 'w₁' }, { l: 'x₂', v: pv, w: S.w2, wl: 'w₂' }, { l: '1', v: 1, w: S.b, wl: 'b' }];
        ins.forEach((it, k) => {
          const y = ys[k], x1 = inX + boxW, col = it.w >= 0 ? C.a : C.b;
          const lw = 1 + 3.5 * Math.min(1, Math.abs(it.w) / 3);
          V.line(g, x1, y, sumX - 22, sumY, { color: V.alpha(col, 0.75), lw });
          for (let q = 0; q < 3; q++) {
            const u = (t * 0.55 + q / 3 + k * 0.17) % 1;
            const pt = V.along(x1, y, sumX - 22, sumY, u);
            V.dot(g, pt[0], pt[1], 1.8 + 2.2 * Math.min(1, Math.abs(it.w * it.v) / 3), col);
          }
          V.box(g, inX, y - boxH / 2, boxW, boxH, { r: 8, fill: V.alpha(C.panel, 0.95), stroke: V.alpha(col, 0.55) });
          T(g, k < 2 ? `${it.l} = ${nf(it.v, 2)}` : '1 (bias)', inX + 10, y - 3, { size: 12, font: 'mono', color: k < 2 ? C.ink : C.muted });
          T(g, `${it.wl} = ${nf(it.w, 1)}`, inX + 10, y + 14, { size: 11.5, font: 'mono', weight: 600, color: col });
        });
        // Σ
        V.box(g, sumX - 22, sumY - 22, 44, 44, { r: 22, fill: V.alpha(C.b, 0.18), stroke: C.b, lw: 2 });
        T(g, 'Σ', sumX, sumY + 7, { size: 20, weight: 700, align: 'center', color: C.ink });
        T(g, 'z = ' + nf(pz, 2), sumX + 10, sumY + 40, { size: 12, font: 'mono', color: C.ink });
        // caixa f com mini-curva
        const fw = 50, fh = 44;
        V.arrow(g, sumX + 22, sumY, fX - fw / 2 - 3, sumY, { color: C.muted, lw: 1.6 });
        V.box(g, fX - fw / 2, sumY - fh / 2, fw, fh, { r: 9, fill: V.alpha(C.d, 0.14), stroke: C.d, lw: 1.8 });
        g.beginPath();
        for (let k = 0; k <= 30; k++) {
          const zz = -4 + (8 * k) / 30; let yy = f(zz);
          if (S.act === 'relu' || S.act === 'linear') yy = yy / 4;
          const X = fX - fw / 2 + 6 + ((fw - 12) * k) / 30, Y = sumY + fh / 2 - 8 - (fh - 16) * V.clamp(S.act === 'linear' ? (yy + 1) / 2 : yy, 0, 1);
          if (k) g.lineTo(X, Y); else g.moveTo(X, Y);
        }
        g.strokeStyle = C.d; g.lineWidth = 1.8; g.stroke();
        T(g, 'f', fX, sumY - fh / 2 - 6, { size: 12, weight: 700, align: 'center', color: C.d });
        V.arrow(g, fX + fw / 2, sumY, outX - 4, sumY, { color: C.muted, lw: 1.6 });
        T(g, 'a = ' + nf(pa, 3), outX, sumY + (narrow ? 32 : 26), { size: 13, font: 'mono', weight: 600, align: 'right', color: pz > 0 ? C.a : C.c });
        T(g, pz > 0 ? 'gets in ✓' : 'stays out ✕', outX, sumY - 14, { size: 12, align: 'right', weight: 600, color: pz > 0 ? C.a : C.c });
        T(g, 'the neuron looking at the guest ◆', dx0, 16, { size: 11.5, color: C.muted });
      });
    });

    /* ====================================================================
       Figura 2 — Uma matriz entorta o plano
       ==================================================================== */
    safe('fig-geometria', () => {
      const fig = $('#fig-geometria');
      fitH(fig, (w) => (w < 600 ? Math.round(w + 240) : 420));
      const PRE = {
        esticar: { W: [[1.6, 0], [0, 0.6]], l: 'stretch' },
        girar: { W: [[0.819, -0.574], [0.574, 0.819]], l: 'rotate 35°' },
        cisalhar: { W: [[1, 0.9], [0, 1]], l: 'shear' },
        refletir: { W: [[-1, 0], [0, 1]], l: 'reflect' },
        achatar: { W: [[1, 0.5], [0.6, 0.3]], l: 'flatten' },
        misturar: { W: [[0.9, -0.7], [0.5, 1.2]], l: 'mix' },
      };
      const S = { key: 'cisalhar', bias: false, relu: false, t0: -2.2 };
      const BIAS = [0.6, 0.4];
      const Fsh = [[0, 0], [0, 2], [1.2, 2], [1.2, 1.6], [0.4, 1.6], [0.4, 1.15], [1.0, 1.15], [1.0, 0.75], [0.4, 0.75], [0.4, 0]].map(([x, y]) => [0.35 + x * 0.75, 0.25 + y * 0.75]);
      let H = null;
      const restart = () => { if (H) S.t0 = H.t; };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Matrix W', options: Object.keys(PRE).map((k) => ({ v: k, label: PRE[k].l })), value: S.key, onChange: (v) => { S.key = v; restart(); } });
      UI.toggle(ctl, { label: 'add bias b', value: false, onChange: (v) => { S.bias = v; restart(); } });
      UI.toggle(ctl, { label: 'ReLU after', value: false, onChange: (v) => { S.relu = v; restart(); } });
      const rDet = UI.readout(ctl, 'det W', '—');
      let lastKey = '';
      H = V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const narrow = w < 600;
        const W = PRE[S.key].W;
        const det = W[0][0] * W[1][1] - W[0][1] * W[1][0];
        if (lastKey !== S.key) { lastKey = S.key; rDet.set(nf(det, 2) + (Math.abs(det) < 1e-9 ? ' (rank 1)' : det < 0 ? ' (flips)' : '')); }
        // linha do tempo
        const tt = (t - S.t0) % 8.4;
        const E = V.ease.inOut;
        let e = 0, r = 0;
        if (tt < 0.5) e = 0;
        else if (tt < 2.0) e = E((tt - 0.5) / 1.5);
        else if (tt < 3.0) e = 1;
        else if (tt < 4.2) { e = 1; r = S.relu ? E((tt - 3) / 1.2) : 0; }
        else if (tt < 7.2) { e = 1; r = S.relu ? 1 : 0; }
        else { const k = E((tt - 7.2) / 1.2); e = 1 - k; r = S.relu ? 1 - k : 0; }
        const M = [[V.lerp(1, W[0][0], e), V.lerp(0, W[0][1], e)], [V.lerp(0, W[1][0], e), V.lerp(1, W[1][1], e)]];
        const bb = S.bias ? [BIAS[0] * e, BIAS[1] * e] : [0, 0];
        const map = (p) => { let q = [M[0][0] * p[0] + M[0][1] * p[1] + bb[0], M[1][0] * p[0] + M[1][1] * p[1] + bb[1]]; if (r > 0) q = [V.lerp(q[0], Math.max(0, q[0]), r), V.lerp(q[1], Math.max(0, q[1]), r)]; return q; };
        // layout
        let ps, px, py, lx, ly, lw;
        if (!narrow) { ps = Math.min(h - 30, w * 0.56); px = w - ps - 18; py = (h - ps) / 2; lx = 20; ly = 26; lw = px - 40; }
        else { ps = Math.min(w - 32, h - 256); px = (w - ps) / 2; py = h - ps - 14; lx = 14; ly = 22; lw = w - 28; }
        const RR = 3.2;
        const toX = (u) => px + ((u + RR) / (2 * RR)) * ps, toY = (v) => py + ((RR - v) / (2 * RR)) * ps;
        // grade original (fantasma)
        g.save(); g.beginPath(); g.rect(px, py, ps, ps); g.clip();
        g.fillStyle = V.alpha(C.panel, 0.5); g.fillRect(px, py, ps, ps);
        for (let k = -3; k <= 3; k++) {
          V.line(g, toX(k), py, toX(k), py + ps, { color: V.alpha(C.muted, 0.12), lw: 1 });
          V.line(g, px, toY(k), px + ps, toY(k), { color: V.alpha(C.muted, 0.12), lw: 1 });
        }
        if (r > 0.01) { g.fillStyle = V.alpha(C.a, 0.06 * r); g.fillRect(toX(0), py, px + ps - toX(0), toY(0) - py); }
        // grade transformada
        const drawLine = (a, b, col, lwid) => {
          g.beginPath();
          for (let k = 0; k <= 40; k++) { const s = k / 40; const q = map([V.lerp(a[0], b[0], s), V.lerp(a[1], b[1], s)]); const X = toX(q[0]), Y = toY(q[1]); if (k) g.lineTo(X, Y); else g.moveTo(X, Y); }
          g.strokeStyle = col; g.lineWidth = lwid; g.stroke();
        };
        for (let k = -3; k <= 3; k++) {
          drawLine([k, -3], [k, 3], V.alpha(C.b, k === 0 ? 0.7 : 0.38), k === 0 ? 1.6 : 1);
          drawLine([-3, k], [3, k], V.alpha(C.a, k === 0 ? 0.7 : 0.38), k === 0 ? 1.6 : 1);
        }
        // letra F
        g.beginPath();
        for (let k = 0; k <= Fsh.length; k++) {
          const A = Fsh[k % Fsh.length], B = Fsh[(k + 1) % Fsh.length];
          if (k === Fsh.length) break;
          for (let s = 0; s <= 6; s++) { const q = map([V.lerp(A[0], B[0], s / 6), V.lerp(A[1], B[1], s / 6)]); const X = toX(q[0]), Y = toY(q[1]); if (k === 0 && s === 0) g.moveTo(X, Y); else g.lineTo(X, Y); }
        }
        g.closePath(); g.fillStyle = V.alpha(C.d, 0.55); g.fill(); g.strokeStyle = C.d; g.lineWidth = 2; g.stroke();
        // vetores base (colunas de W)
        const o = map([0, 0]);
        if (r < 0.5) {
          const al = 1 - r * 2;
          const e1 = map([1, 0]), e2 = map([0, 1]);
          g.globalAlpha = al;
          V.arrow(g, toX(o[0]), toY(o[1]), toX(e1[0]), toY(e1[1]), { color: C.a, lw: 3, head: 11 });
          V.arrow(g, toX(o[0]), toY(o[1]), toX(e2[0]), toY(e2[1]), { color: C.b, lw: 3, head: 11 });
          g.globalAlpha = 1;
        }
        // sonda
        const pr = [1.3 * Math.cos(t * 0.5), 1.3 * Math.sin(t * 0.5)];
        const pq = map(pr);
        V.line(g, toX(pr[0]), toY(pr[1]), toX(pq[0]), toY(pq[1]), { color: V.alpha(C.c, 0.5), lw: 1.2, dash: [3, 3] });
        V.dot(g, toX(pr[0]), toY(pr[1]), 4, V.alpha(C.c, 0.45));
        V.glow(g, toX(pq[0]), toY(pq[1]), 14, C.c, 0.8);
        V.dot(g, toX(pq[0]), toY(pq[1]), 4.5, C.c);
        g.restore();
        g.strokeStyle = V.alpha(C.muted, 0.35); g.lineWidth = 1; g.strokeRect(px, py, ps, ps);
        T(g, e < 0.02 ? 'identity' : r > 0.02 ? 'W·x + b, then ReLU' : 'W·x' + (S.bias ? ' + b' : ''), px + 8, py + 18, { size: 12, weight: 600, color: C.ink });
        // ---- painel da matriz
        const cw = narrow ? 52 : 58, ch = 34, gx = 4;
        const mx0 = lx + (narrow ? 34 : 38), my0 = ly + 26;
        T(g, 'W  (row = neuron)', lx, ly + 4, { size: 12, color: C.muted, font: 'mono' });
        T(g, 'W =', lx, my0 + ch + 6, { size: 13, weight: 600, color: C.ink, font: 'mono' });
        for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) {
          const X = mx0 + j * (cw + gx), Y = my0 + i * (ch + gx), v = M[i][j];
          const col = j === 0 ? C.a : C.b;
          V.box(g, X, Y, cw, ch, { r: 7, fill: V.alpha(col, 0.1 + 0.3 * Math.min(1, Math.abs(v))), stroke: V.alpha(col, 0.8), lw: 1.4 });
          T(g, nf(v, 2), X + cw / 2, Y + ch / 2 + 4.5, { size: 13, font: 'mono', weight: 600, align: 'center', color: C.ink });
        }
        // x e y (sonda)
        const vx = mx0 + 2 * (cw + gx) + 14;
        const yv = [M[0][0] * pr[0] + M[0][1] * pr[1] + bb[0], M[1][0] * pr[0] + M[1][1] * pr[1] + bb[1]];
        if (vx + 150 < lx + lw || narrow) {
          const cw2 = 50;
          T(g, '·', vx - 8, my0 + ch + 7, { size: 18, color: C.muted, align: 'center' });
          for (let i = 0; i < 2; i++) { V.box(g, vx, my0 + i * (ch + gx), cw2, ch, { r: 7, fill: V.alpha(C.c, 0.12), stroke: V.alpha(C.c, 0.5), lw: 1 }); T(g, nf(pr[i], 2), vx + cw2 / 2, my0 + i * (ch + gx) + ch / 2 + 4.5, { size: 12.5, font: 'mono', align: 'center', color: C.ink }); }
          const vx2 = vx + cw2 + 22;
          T(g, '=', vx2 - 11, my0 + ch + 7, { size: 15, color: C.muted, align: 'center' });
          for (let i = 0; i < 2; i++) { V.box(g, vx2, my0 + i * (ch + gx), cw2, ch, { r: 7, fill: V.alpha(C.c, 0.28), stroke: C.c, lw: 1.4 }); T(g, nf(yv[i], 2), vx2 + cw2 / 2, my0 + i * (ch + gx) + ch / 2 + 4.5, { size: 12.5, font: 'mono', weight: 600, align: 'center', color: C.ink }); }
          T(g, 'x (● pink)', vx + cw2 / 2, my0 - 8, { size: 11, align: 'center', color: C.muted });
          T(g, S.bias ? 'Wx + b' : 'y = Wx', vx2 + cw2 / 2, my0 - 8, { size: 11, align: 'center', color: C.muted });
        }
        let yy = my0 + 2 * ch + gx + 26;
        T(g, `y₁ = ${nf(M[0][0], 2)}·x₁ + ${nf(M[0][1], 2)}·x₂${S.bias ? ' + ' + nf(bb[0], 2) : ''}`, lx, yy, { size: 12, font: 'mono', color: C.ink, maxW: lw });
        yy += 18;
        T(g, `y₂ = ${nf(M[1][0], 2)}·x₁ + ${nf(M[1][1], 2)}·x₂${S.bias ? ' + ' + nf(bb[1], 2) : ''}`, lx, yy, { size: 12, font: 'mono', color: C.ink, maxW: lw });
        {
          const detNow = M[0][0] * M[1][1] - M[0][1] * M[1][0];
          yy += narrow ? 24 : 30;
          T(g, 'det = ' + nf(detNow, 2) + '  → area × ' + nf(Math.abs(detNow), 2), lx, yy, { size: 12, font: 'mono', color: C.d });
          yy += narrow ? 22 : 26;
          const notes = { esticar: 'Column 1 = (1.6, 0): the x axis stretches. Column 2 = (0, 0.6): the y axis shrinks.', girar: 'Unit, perpendicular columns: a rotation preserves lengths and angles (det = 1).', cisalhar: 'ê₂ goes to (0.9, 1): horizontal lines slide over, and the area stays the same (det = 1).', refletir: 'det < 0: the “F” appears mirrored. The orientation of the plane has flipped.', achatar: 'Parallel columns, det = 0: the whole plane collapses onto a line. One dimension of information is lost.', misturar: 'An arbitrary matrix: it stretches, rotates and shears all at once.' };
          const reluNote = 'ReLU: everything that landed outside the positive quadrant is squashed against the axes. It is a fold, not a linear transformation.';
          if (narrow) wrap(g, S.relu ? reluNote : notes[S.key], lx, yy, lw, 16, { size: 12, color: S.relu ? C.a : C.muted, maxLines: 3 });
          else {
            const nl = wrap(g, notes[S.key], lx, yy, lw, 17, { size: 12.5, color: C.muted });
            if (S.relu) wrap(g, reluNote, lx, yy + nl * 17 + 12, lw, 17, { size: 12.5, color: C.a });
          }
        }
      });
    });

    /* ====================================================================
       Figura 3 — Zoológico de ativações
       ==================================================================== */
    safe('fig-ativacoes', () => {
      const fig = $('#fig-ativacoes');
      fitH(fig, (w) => (w < 600 ? 700 : 430));
      const INFO = {
        relu: { n: 'ReLU', fx: 'f(z) = max(0, z)', dx: "f′(z) = 1 if z > 0, else 0", rg: 'output in [0, ∞)', rec: 'Sparse indexers: Σ w_h·ReLU(q_h·k) in DSA (GLM) and in the lightning indexer (DeepSeek); Σ ReLU(q_h·k̄) over blocks of 4 tokens, without w_h, in QSA (Qwen).', col: 'a', pr: ['bounded: below only', 'smooth: no (kink at 0)', 'monotonic: yes', 'cost: one comparison'] },
        gelu: { n: 'GELU', fx: 'f(z) = z·Φ(z)', dx: "f′(z) = Φ(z) + z·φ(z)", rg: 'minimum ≈ −0.17', rec: 'Vision towers: MLP of the Qwen ViT (gelu_pytorch_tanh) and the Qwen merger (GELU; for the variant, see the code); DeepSeek Aligner (exact GELU).', col: 'e', pr: ['bounded: below only', 'smooth: yes', 'monotonic: no', 'cost: erf or tanh'] },
        silu: { n: 'SiLU / Swish', fx: 'f(z) = z·σ(z)', dx: "f′(z) = σ(z)·(1 + z·(1 − σ(z)))", rg: 'minimum ≈ −0.278 at z ≈ −1.278', rec: 'hidden_act: silu in all three models (the SwiGLU gate branch); short conv + SiLU in KDA (GLM) and GDN (Qwen); SwiGLU ViT and merger in GLM; SwiGLU ViT in DeepSeek.', col: 'a', pr: ['bounded: below only', 'smooth: yes', 'monotonic: no', 'cost: one exponential'] },
        sigmoid: { n: 'Sigmoid σ', fx: 'f(z) = 1 / (1 + e^(−z))', dx: "f′(z) = σ(z)·(1 − σ(z)) ≤ 0.25", rg: 'output in (0, 1)', rec: 'Gates: GDN, attention, Gated Residual, PLE, shared expert (Qwen); router, β and mHC (GLM); mHC and hc_head (DeepSeek).', col: 'd', pr: ['bounded: yes, (0, 1)', 'smooth: yes', 'monotonic: yes', 'saturates: yes, at the ends'] },
        softplus: { n: 'Softplus', fx: 'f(z) = ln(1 + e^z)', dx: "f′(z) = σ(z)", rg: 'output in (0, ∞)', rec: 'GDN decay: α = exp(−e^A·softplus(·)) (Qwen); the router √softplus score (DeepSeek).', col: 'b', pr: ['bounded: below only (> 0)', 'smooth: yes', 'monotonic: yes', 'derivative: σ(z)'] },
        tanh: { n: 'tanh', fx: 'f(z) = tanh(z) = 2σ(2z) − 1', dx: "f′(z) = 1 − tanh²(z)", rg: 'output in (−1, 1)', rec: 'Inside the approximate GELU (gelu_pytorch_tanh) of the Qwen ViT. The softcap c·tanh(z/c) exists in the DFlash2 code (GLM), but it is turned off.', col: 'c', pr: ['bounded: yes, (−1, 1)', 'smooth: yes', 'monotonic: yes', 'saturates: yes, at the ends'] },
      };
      const KEYS = ['relu', 'gelu', 'silu', 'sigmoid', 'softplus', 'tanh'];
      const S = { fn: 'silu', deriv: true, approx: false, z: 1, manual: false };
      let maxDiff = 0; for (let z = -6; z <= 6; z += 0.001) maxDiff = Math.max(maxDiff, Math.abs(ACT.gelu.f(z) - ACT.gelut.f(z)));
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Function', options: KEYS.map((k) => ({ v: k, label: INFO[k].n.split(' ')[0] })), value: S.fn, onChange: (v) => { S.fn = v; } });
      UI.toggle(ctl, { label: 'derivative', value: true, onChange: (v) => { S.deriv = v; } });
      UI.toggle(ctl, { label: 'tanh approximation (GELU)', value: false, onChange: (v) => { S.approx = v; } });
      let tAuto = null;
      const sZ = UI.slider(ctl, { label: 'probe z', min: -5, max: 5, step: 0.05, value: S.z, fmt: (v) => nf(v, 2) + (S.manual ? '' : ' · auto'), onInput: (v) => { S.z = v; S.manual = true; sZval.textContent = nf(v, 2); tAuto.el.querySelector('input').checked = false; } });
      tAuto = UI.toggle(ctl, { label: 'auto probe', value: true, onChange: (v) => { S.manual = !v; } });
      const sZval = sZ.el.querySelector('.val');
      let lastSync = -1;
      const rF = UI.readout(ctl, 'f(z)', '—');
      let lastTxt = '';
      const r0 = V.rng(3);
      const parts = Array.from({ length: 26 }, () => ({ z: V.clamp(r0.normal() * 1.8, -4.8, 4.8), ph: r0() }));
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const narrow = w < 600;
        const cardW = narrow ? w - 28 : 250;
        const pl = 38, pr = narrow ? 14 : cardW + 34, pt = 20, pb = narrow ? 310 : 34;
        const X0 = pl, X1 = w - pr, Y0 = pt, Y1 = h - pb;
        const zmin = -5, zmax = 5, ymin = -1.5, ymax = 4;
        const toX = (z) => X0 + ((z - zmin) / (zmax - zmin)) * (X1 - X0), toY = (y) => Y1 - ((y - ymin) / (ymax - ymin)) * (Y1 - Y0);
        // grade
        for (let z = -5; z <= 5; z++) V.line(g, toX(z), Y0, toX(z), Y1, { color: V.alpha(C.muted, z === 0 ? 0.45 : 0.1), lw: 1 });
        for (let y = -1; y <= 4; y++) V.line(g, X0, toY(y), X1, toY(y), { color: V.alpha(C.muted, y === 0 ? 0.45 : 0.1), lw: 1 });
        for (let z = -4; z <= 4; z += 2) T(g, String(z), toX(z), Y1 + 14, { size: 11, font: 'mono', align: 'center', color: C.muted });
        for (let y = -1; y <= 4; y++) T(g, String(y), X0 - 8, toY(y) + 4, { size: 11, font: 'mono', align: 'right', color: C.muted });
        T(g, 'z', X1 - 4, toY(0) - 6, { size: 11, font: 'mono', color: C.muted, align: 'right' });
        g.save(); g.beginPath(); g.rect(X0, Y0 - 2, X1 - X0, Y1 - Y0 + 4); g.clip();
        const curve = (fn, col, lw, dash) => {
          g.beginPath();
          const N = Math.max(80, Math.round((X1 - X0) / 3));
          for (let k = 0; k <= N; k++) { const z = zmin + ((zmax - zmin) * k) / N; const X = toX(z), Y = toY(V.clamp(fn(z), ymin - 1, ymax + 1)); if (k) g.lineTo(X, Y); else g.moveTo(X, Y); }
          g.strokeStyle = col; g.lineWidth = lw; if (dash) g.setLineDash(dash); g.stroke(); g.setLineDash([]);
        };
        KEYS.forEach((k) => { if (k !== S.fn) curve(ACT[k].f, V.alpha(C[INFO[k].col], C.dark ? 0.24 : 0.4), 1.6); });
        const I = INFO[S.fn], col = C[I.col];
        const fsel = ACT[S.fn].f, dsel = ACT[S.fn].d;
        // partículas: z sobe até f(z)
        parts.forEach((p) => {
          const u = (t * 0.28 + p.ph) % 1;
          const yEnd = fsel(p.z), X = toX(p.z);
          const k = V.ease.inOut(Math.min(1, u / 0.6));
          const Y = V.lerp(toY(0), toY(yEnd), k);
          const a = u < 0.6 ? 0.8 : 0.8 * (1 - (u - 0.6) / 0.4);
          V.dot(g, X, Y, 2.6, V.alpha(col, a));
          if (u >= 0.6) V.dot(g, V.lerp(X, X0 + 3, (u - 0.6) / 0.4), toY(yEnd), 2, V.alpha(col, a * 0.8));
        });
        if (S.deriv) curve(dsel, V.alpha(C.ink, 0.7), 1.8, [5, 4]);
        curve(fsel, col, 3.2);
        if (S.fn === 'gelu' && S.approx) curve(ACT.gelut.f, C.c, 1.6, [2, 3]);
        g.restore();
        // sonda
        const z = S.manual ? S.z : 3.6 * Math.sin(t * 0.45);
        if (!S.manual && Math.abs(t - lastSync) > 0.05) { lastSync = t; const zq = Math.round(z / 0.05) * 0.05; sZ.input.value = zq; sZval.textContent = nf(z, 2) + ' · auto'; }
        const fz = fsel(z), dz = dsel(z);
        const PX = toX(z), PY = toY(V.clamp(fz, ymin, ymax));
        V.line(g, PX, Y0, PX, Y1, { color: V.alpha(C.ink, 0.25), lw: 1, dash: [3, 4] });
        // tangente
        const tl = 0.9;
        V.line(g, toX(z - tl), toY(fz - dz * tl), toX(z + tl), toY(fz + dz * tl), { color: V.alpha(C.ink, 0.55), lw: 1.4 });
        V.glow(g, PX, PY, 16, col, 0.9);
        V.dot(g, PX, PY, 5, col);
        const txt = nf(fz, 3);
        if (txt !== lastTxt) { lastTxt = txt; rF.set(txt); }
        // cartão
        const cx = narrow ? 14 : w - cardW - 16, cy = narrow ? h - pb + 34 : 22, chh = narrow ? pb - 46 : h - 44;
        V.box(g, cx, cy, cardW, chh, { r: 12, fill: V.alpha(C.panel, 0.92), stroke: V.alpha(col, 0.6), lw: 1.4 });
        let yy = cy + 24;
        T(g, I.n, cx + 14, yy, { size: 16, weight: 700, color: col, font: 'display' }); yy += 22;
        T(g, I.fx, cx + 14, yy, { size: 12.5, font: 'mono', color: C.ink, maxW: cardW - 28 }); yy += 18;
        T(g, I.dx, cx + 14, yy, { size: 12, font: 'mono', color: C.muted, maxW: cardW - 28 }); yy += 18;
        T(g, I.rg, cx + 14, yy, { size: 12, color: C.muted, maxW: cardW - 28 }); yy += 22;
        T(g, `z = ${nf(z, 2)}  →  f = ${nf(fz, 3)} · f′ = ${nf(dz, 3)}`, cx + 14, yy, { size: 12, font: 'mono', color: C.ink, maxW: cardW - 28 }); yy += 22;
        if (S.fn === 'gelu' && S.approx) { T(g, `max |GELU − GELU-tanh| ≈ ${maxDiff.toExponential(1)}`, cx + 14, yy, { size: 11.5, font: 'mono', color: C.c, maxW: cardW - 28 }); yy += 20; }
        T(g, 'IN THE RECIPES', cx + 14, yy, { size: 11, font: 'mono', color: col, weight: 600 }); yy += 17;
        const nl = wrap(g, I.rec, cx + 14, yy, cardW - 28, 16, { size: 12, color: C.ink, maxLines: narrow ? 5 : 9 });
        yy += nl * 16 + 14;
        if (!narrow && yy + 4 * 18 < cy + chh) {
          T(g, 'PROPERTIES', cx + 14, yy, { size: 11, font: 'mono', color: col, weight: 600 }); yy += 18;
          I.pr.forEach((p) => { T(g, '• ' + p, cx + 14, yy, { size: 12, color: C.muted, maxW: cardW - 28 }); yy += 18; });
        } else if (narrow && yy + 2 * 17 < cy + chh) {
          // no celular: propriedades em 2 colunas, para ocupar o cartão
          T(g, 'PROPERTIES', cx + 14, yy, { size: 11, font: 'mono', color: col, weight: 600 }); yy += 17;
          const colW = (cardW - 28) / 2;
          I.pr.forEach((p, k) => { T(g, '• ' + p, cx + 14 + (k % 2) * colW, yy + Math.floor(k / 2) * 17, { size: 11.5, color: C.muted, maxW: colW - 6 }); });
        }
      });
    });

    /* ====================================================================
       Figura 4 — Somando dobradiças (aproximação universal)
       ==================================================================== */
    safe('fig-aproximacao', () => {
      const fig = $('#fig-aproximacao');
      fitH(fig, (w) => (w < 600 ? 470 : 390));
      const TG = {
        onda: { l: 'wave', f: (x) => 0.6 * Math.sin(7 * x) + 0.25 * Math.sin(19 * x) },
        montanha: { l: 'mountains', f: (x) => 0.95 * Math.exp(-(((x - 0.33) / 0.12) ** 2)) + 0.5 * Math.exp(-(((x - 0.76) / 0.06) ** 2)) - 0.35 },
        degrau: { l: 'step', f: (x) => 0.55 * Math.tanh(40 * (x - 0.5)) },
      };
      const S = { tg: 'onda', kind: 'relu', N: 8, build: null };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Target', options: Object.keys(TG).map((k) => ({ v: k, label: TG[k].l })), value: S.tg, onChange: (v) => { S.tg = v; recompute(); } });
      UI.seg(ctl, { label: 'Neurons', options: [{ v: 'relu', label: 'ReLU (hinges)' }, { v: 'sigmoid', label: 'sigmoid (steps)' }], value: S.kind, onChange: (v) => { S.kind = v; recompute(); } });
      const sN = UI.slider(ctl, { label: 'N neurons', min: 1, max: 40, step: 1, value: S.N, fmt: (v) => String(v), onInput: (v) => { S.N = v; recompute(); } });
      UI.button(ctl, '▶ build', () => {
        if (S.build) { clearInterval(S.build); S.build = null; }
        let n = 1; sN.set(1);
        S.build = setInterval(() => { if (LA.paused) return; n++; if (n > 24) { clearInterval(S.build); S.build = null; return; } sN.set(n); }, 260);
      }, { primary: true });
      LA.onCleanup(() => { if (S.build) clearInterval(S.build); });
      const rErr = UI.readout(ctl, 'max error', '—');
      const rP = UI.readout(ctl, 'parameters', '—');
      let M = null;
      function recompute() {
        const f = TG[S.tg].f, N = S.N;
        const units = [];
        let base;
        if (S.kind === 'relu') {
          const k = Array.from({ length: N + 1 }, (_, i) => i / N);
          const sl = []; for (let i = 0; i < N; i++) sl.push((f(k[i + 1]) - f(k[i])) * N);
          base = f(0);
          for (let i = 0; i < N; i++) units.push({ k: k[i], c: sl[i] - (i ? sl[i - 1] : 0) });
        } else {
          const sStp = 12 * N;
          const v = Array.from({ length: N }, (_, i) => f((i + 0.5) / N));
          base = f(0);
          for (let i = 0; i < N; i++) units.push({ k: i / N, c: v[i] - (i ? v[i - 1] : base), s: sStp });
        }
        const g = (x) => { let s = base; for (const u of units) s += S.kind === 'relu' ? u.c * Math.max(0, x - u.k) : u.c * sig(u.s * (x - u.k)); return s; };
        let err = 0; for (let i = 0; i <= 400; i++) { const x = i / 400; err = Math.max(err, Math.abs(g(x) - f(x))); }
        M = { f, g, units, base, err };
        rErr.set(nf(err, 3));
        rP.set(String(3 * N + 1));
      }
      recompute();
      V.fig(fig, (gc, w, h, t) => {
        const C = V.C();
        const narrow = w < 600;
        const X0 = 40, X1 = w - 16, Y0 = 26, Ym = Y0 + (h - 40) * 0.66, Yb0 = Ym + 26, Yb1 = h - 18;
        const toX = (x) => X0 + x * (X1 - X0);
        const ymn = -1.05, ymx = 1.05;
        const toY = (y) => Ym - ((y - ymn) / (ymx - ymn)) * (Ym - Y0);
        V.line(gc, X0, toY(0), X1, toY(0), { color: V.alpha(C.muted, 0.35), lw: 1 });
        V.line(gc, X0, Y0, X0, Ym, { color: V.alpha(C.muted, 0.35), lw: 1 });
        [-1, 0, 1].forEach((y) => T(gc, String(y), X0 - 8, toY(y) + 4, { size: 11, font: 'mono', align: 'right', color: C.muted }));
        const xs = (t * 0.16) % 1.15;
        const scan = Math.min(1, xs);
        // alvo
        gc.beginPath();
        for (let i = 0; i <= 300; i++) { const x = i / 300; const X = toX(x), Y = toY(M.f(x)); if (i) gc.lineTo(X, Y); else gc.moveTo(X, Y); }
        gc.strokeStyle = V.alpha(C.ink, 0.6); gc.lineWidth = 1.6; gc.setLineDash([6, 5]); gc.stroke(); gc.setLineDash([]);
        // aproximação
        gc.save(); gc.beginPath(); gc.rect(X0, Y0 - 4, X1 - X0, Ym - Y0 + 8); gc.clip();
        gc.beginPath();
        for (let i = 0; i <= 400; i++) { const x = i / 400; const X = toX(x), Y = toY(M.g(x)); if (i) gc.lineTo(X, Y); else gc.moveTo(X, Y); }
        gc.strokeStyle = C.a; gc.lineWidth = 3; gc.stroke();
        gc.restore();
        // dobras
        M.units.forEach((u, i) => {
          const X = toX(u.k);
          const on = S.kind === 'relu' ? scan > u.k : scan > u.k;
          const rr = 2.5 + Math.min(5, Math.abs(u.c) * (S.kind === 'relu' ? 0.12 : 4));
          V.dot(gc, X, toY(M.g(u.k)), rr, V.alpha(u.c >= 0 ? C.a : C.c, on ? 0.95 : 0.4));
        });
        // varredura
        const SX = toX(scan);
        V.line(gc, SX, Y0, SX, Yb1, { color: V.alpha(C.d, 0.55), lw: 1.2, dash: [3, 4] });
        V.glow(gc, SX, toY(M.g(scan)), 12, C.d, 0.8);
        T(gc, 'target (dashed) and sum of ' + S.N + ' neuron' + (S.N > 1 ? 's' : ''), X0 + 6, Y0 - 8, { size: 12, color: C.muted });
        // painel das contribuições
        V.box(gc, X0, Yb0, X1 - X0, Yb1 - Yb0, { r: 8, fill: V.alpha(C.panel, 0.6), stroke: V.alpha(C.muted, 0.25), lw: 1 });
        T(gc, narrow ? (S.kind === 'relu' ? 'each neuron: ReLU(x − kᵢ)' : 'each neuron: σ(s·(x − kᵢ))') : S.kind === 'relu' ? 'each neuron: ReLU(x − kᵢ) · color = sign of weight cᵢ' : 'each neuron: σ(s·(x − kᵢ)) · color = sign of the jump', X0 + 8, Yb0 + 14, { size: 11, color: C.muted });
        const bh = Yb1 - Yb0 - 22;
        let active = 0;
        M.units.forEach((u, i) => {
          const on = scan > u.k; if (on) active++;
          const colU = u.c >= 0 ? C.a : C.c;
          gc.beginPath();
          for (let k = 0; k <= 60; k++) {
            const x = k / 60;
            const y = S.kind === 'relu' ? Math.max(0, x - u.k) / Math.max(1e-6, 1 - u.k) : sig(u.s * (x - u.k));
            const X = toX(x), Y = Yb1 - 4 - y * bh;
            if (k) gc.lineTo(X, Y); else gc.moveTo(X, Y);
          }
          gc.strokeStyle = V.alpha(colU, on ? 0.75 : 0.22); gc.lineWidth = on ? 1.6 : 1; gc.stroke();
        });
        T(gc, narrow ? `${active}/${S.N} active` : `${active} of ${S.N} active at x = ${nf(scan, 2)}`, X1 - 8, Yb0 + 14, { size: 11, font: 'mono', align: 'right', color: C.d });
      });
    });

    /* ====================================================================
       Figura 5 — A torneira: gating multiplicativo
       ==================================================================== */
    safe('fig-portas', () => {
      const fig = $('#fig-portas');
      fitH(fig, (w) => (w < 600 ? 500 : 420));
      const D = 8;
      const rr = V.rng(17);
      const Wu = Array.from({ length: D }, () => Array.from({ length: D }, () => rr.normal() / Math.sqrt(D) * 1.2));
      const Wg = Array.from({ length: D }, () => Array.from({ length: D }, () => rr.normal() / Math.sqrt(D) * 1.6));
      const TOK = [
        { l: '“cat”', x: [1.2, -0.4, 0.8, 1.5, -1.1, 0.3, 0.9, -0.6] },
        { l: '“def”', x: [-0.9, 1.4, -0.2, 0.4, 1.3, -1.5, 0.1, 1.0] },
        { l: '“∑”', x: [0.3, 0.9, -1.4, -0.7, 0.2, 1.1, -1.2, 1.6] },
      ];
      const TYPES = { glu: { l: 'GLU', fn: sig, lab: 'g = σ(W_g·x + bias)' }, swiglu: { l: 'SwiGLU', fn: ACT.silu.f, lab: 'g = SiLU(W_g·x + bias)' }, geglu: { l: 'GeGLU', fn: ACT.gelu.f, lab: 'g = GELU(W_g·x + bias)' }, none: { l: 'no gate', fn: null, lab: 'y = ReLU(u)' } };
      const S = { tok: 0, type: 'glu', bias: 0 };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Token', options: TOK.map((tk, i) => ({ v: i, label: tk.l })), value: 0, onChange: (v) => { S.tok = v; } });
      UI.seg(ctl, { label: 'Type', options: Object.keys(TYPES).map((k) => ({ v: k, label: TYPES[k].l })), value: S.type, onChange: (v) => { S.type = v; } });
      UI.slider(ctl, { label: 'gate bias', min: -4, max: 4, step: 0.1, value: 0, fmt: (v) => nf(v, 1), onInput: (v) => { S.bias = v; } });
      const rR = UI.readout(ctl, '‖y‖ / ‖u‖', '—');
      let last = '';
      const mv = (W, x) => W.map((row) => row.reduce((s, v, j) => s + v * x[j], 0));
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const narrow = w < 600;
        const x = TOK[S.tok].x;
        const u = mv(Wu, x), zg = mv(Wg, x).map((v) => v + S.bias);
        const Ty = TYPES[S.type];
        const gate = Ty.fn ? zg.map(Ty.fn) : null;
        const y = Ty.fn ? u.map((v, i) => v * gate[i]) : u.map((v) => Math.max(0, v));
        const nu = Math.hypot(...u), ny = Math.hypot(...y);
        const rt = nf(ny / nu, 2);
        if (rt !== last) { last = rt; rR.set(rt); }
        const labW = narrow ? 0 : 150;
        const gx0 = 16 + labW, gx1 = w - 16;
        const cw = Math.min(62, (gx1 - gx0 - 7 * 6) / 8), gap = (gx1 - gx0 - 8 * cw) / 7;
        const colX = (i) => gx0 + i * (cw + gap);
        const rowsY = narrow ? [34, 144, 266, 400] : [26, 118, 230, 342];
        const ch = narrow ? 40 : 40;
        const labels = [['x', 'the token enters'], ['u = W_u·x', 'content'], [Ty.fn ? Ty.lab : 'no gate', Ty.fn ? 'how open each faucet is' : 'activation only'], ['y = g ⊙ u', 'output']];
        const drawLabels = () => labels.forEach(([a, b], k) => {
          if (!narrow) { T(g, a, 16, rowsY[k] + ch / 2 + 1, { size: 12.5, font: 'mono', weight: 600, color: C.ink, maxW: labW - 10 }); T(g, b, 16, rowsY[k] + ch / 2 + 17, { size: 11, color: C.muted, maxW: labW - 10 }); }
          else {
            // no celular o rótulo fica sobre as linhas e partículas: desenha num fundo opaco
            const txt = a + '  ·  ' + b, tw = Math.min(w - 28, textW(g, txt, 11.5, 'mono', 400));
            V.box(g, 11, rowsY[k] - 21, tw + 10, 18, { r: 5, fill: V.alpha(C.bg, 0.94) });
            T(g, txt, 16, rowsY[k] - 8, { size: 11.5, font: 'mono', color: C.muted, maxW: w - 32 });
          }
        });
        const cellDiv = (X, Y, v, sc, strong) => {
          const k = V.clamp(v / sc, -1, 1);
          V.box(g, X, Y, cw, ch, { r: 7, fill: V.alpha(k >= 0 ? C.a : C.b, 0.08 + 0.5 * Math.abs(k)), stroke: V.alpha(k >= 0 ? C.a : C.b, strong ? 0.9 : 0.55), lw: strong ? 1.6 : 1 });
          if (cw >= 30) T(g, nf(v, cw >= 44 ? 2 : 1), X + cw / 2, Y + ch / 2 + 4, { size: cw >= 44 ? 12 : 11, font: 'mono', align: 'center', color: C.ink });
        };
        for (let i = 0; i < D; i++) {
          const X = colX(i);
          cellDiv(X, rowsY[0], x[i], 1.6, false);
          cellDiv(X, rowsY[1], u[i], 2, false);
          // porta
          const GY = rowsY[2];
          if (gate) {
            const gv = gate[i];
            V.box(g, X, GY, cw, ch + 8, { r: 8, fill: V.alpha(C.panel, 0.9), stroke: V.alpha(C.d, 0.6), lw: 1.2 });
            const open = V.clamp(gv, 0, 1.4) / 1.4;
            const neg = gv < 0;
            const fillH = (ch + 2) * (neg ? V.clamp(-gv / 1.4, 0, 1) : open);
            g.fillStyle = neg ? V.alpha(C.c, 0.55) : V.alpha(C.d, 0.25 + 0.5 * open);
            g.fillRect(X + 3, GY + ch + 5 - fillH, cw - 6, fillH);
            // torneira (válvula) que gira
            const vx = X + cw / 2, vy = GY + 12, ang = (1 - V.clamp(gv, 0, 1)) * Math.PI / 2;
            g.strokeStyle = C.d; g.lineWidth = 2.4; g.beginPath(); g.moveTo(vx - Math.cos(ang) * 9, vy - Math.sin(ang) * 9); g.lineTo(vx + Math.cos(ang) * 9, vy + Math.sin(ang) * 9); g.stroke();
            V.dot(g, vx, vy, 2.5, C.d);
            if (cw >= 30) T(g, nf(gv, cw >= 40 ? 2 : 1), X + cw / 2, GY + ch + 1, { size: cw >= 44 ? 12 : 11, font: 'mono', align: 'center', weight: 600, color: C.ink });
          } else {
            V.box(g, X, GY, cw, ch + 8, { r: 8, fill: V.alpha(C.panel, 0.5), stroke: V.alpha(C.muted, 0.3), lw: 1 });
            T(g, 'ReLU', X + cw / 2, GY + ch / 2 + 8, { size: 11, font: 'mono', align: 'center', color: C.muted, maxW: cw - 4 });
          }
          cellDiv(X, rowsY[3], y[i], 2, true);
          // setas e partículas
          const cx = X + cw / 2;
          V.line(g, cx, rowsY[0] + ch + 2, cx, rowsY[1] - 2, { color: V.alpha(C.muted, 0.3), lw: 1 });
          V.line(g, cx, rowsY[1] + ch + 2, cx, rowsY[2] - 2, { color: V.alpha(C.muted, 0.3), lw: 1 });
          V.line(g, cx, rowsY[2] + ch + 10, cx, rowsY[3] - 2, { color: V.alpha(C.muted, 0.3), lw: 1 });
          const strength = gate ? V.clamp(Math.abs(gate[i]), 0, 1.2) : (u[i] > 0 ? 1 : 0);
          for (let q = 0; q < 3; q++) {
            const ph = (t * 0.5 + q / 3 + i * 0.13) % 1;
            let py, al;
            if (ph < 0.3) { py = V.lerp(rowsY[0] + ch, rowsY[1], ph / 0.3); al = 0.8; }
            else if (ph < 0.62) { py = V.lerp(rowsY[1] + ch, rowsY[2], (ph - 0.3) / 0.32); al = 0.8; }
            else { py = V.lerp(rowsY[2] + ch + 8, rowsY[3], (ph - 0.62) / 0.38); al = 0.1 + 0.85 * strength; }
            V.dot(g, cx, py, ph < 0.62 ? 2.4 : 1.4 + 2.4 * strength, V.alpha(ph < 0.62 ? C.ink : (y[i] >= 0 ? C.a : C.b), al));
          }
        }
        drawLabels();
        const note = S.type === 'glu' ? 'σ ∈ (0,1): the gate only attenuates, it never amplifies.' : S.type === 'swiglu' ? 'SiLU has no ceiling and no floor at 0: it is a multiplicative interaction z_g·σ(z_g)·z_u.' : S.type === 'geglu' ? 'GeGLU: like SwiGLU, with GELU in the gate branch.' : 'Without a gate, each output depends only on u: ReLU cuts off the negatives.';
        T(g, note, 16, h - 10, { size: 12, color: C.d, maxW: w - 32 });
      });
    });

    /* ====================================================================
       Figura 6 — O grafo e o caminho de volta (backprop)
       ==================================================================== */
    safe('fig-backprop', () => {
      const fig = $('#fig-backprop');
      fitH(fig, (w) => (w < 600 ? 560 : 380));
      const S = { x: 1.5, w: 0.8, b: -0.5, y: 1, eta: 1 };
      const Q = {};
      function compute() {
        const m = S.w * S.x, z = m + S.b, a = sig(z), L = 0.5 * (a - S.y) ** 2;
        const dLda = a - S.y, dadz = a * (1 - a), dLdz = dLda * dadz;
        const dLdw = dLdz * S.x, dLdx = dLdz * S.w, dLdb = dLdz;
        const w2 = S.w - S.eta * dLdw, b2 = S.b - S.eta * dLdb, a2 = sig(w2 * S.x + b2), L2 = 0.5 * (a2 - S.y) ** 2;
        const hh = 1e-4, Lf = (ww) => 0.5 * (sig(ww * S.x + S.b) - S.y) ** 2, fd = (Lf(S.w + hh) - Lf(S.w - hh)) / (2 * hh);
        Object.assign(Q, { m, z, a, L, dLda, dadz, dLdz, dLdw, dLdx, dLdb, w2, b2, L2, fd });
      }
      compute();
      const f3 = (v) => nf(v, 3), f4 = (v) => nf(v, 4);
      const defs = () => [
        { t: 'The graph', d: 'Four simple operations: multiply, add, activate and measure the error. Each node knows how to compute its own output (on the way forward) and its own local derivative (on the way back).' },
        { t: 'Forward · multiply', d: `m = w·x = ${f3(S.w)} × ${f3(S.x)} = <b>${f3(Q.m)}</b>.` },
        { t: 'Forward · add the bias', d: `z = m + b = ${f3(Q.m)} + (${f3(S.b)}) = <b>${f3(Q.z)}</b>.` },
        { t: 'Forward · activate', d: `a = σ(z) = 1/(1 + e<sup>−z</sup>) = <b>${f4(Q.a)}</b>.` },
        { t: 'Forward · measure the loss', d: `L = ½(a − y)² = ½(${f4(Q.a)} − ${f3(S.y)})² = <b>${f4(Q.L)}</b>. Each node stored its value: the backward pass will need them.` },
        { t: 'Backward · start from the loss', d: `∂L/∂a = a − y = <b>${f4(Q.dLda)}</b>. ${Q.dLda < 0 ? 'Negative: increasing a would lower the loss.' : 'Positive: decreasing a would lower the loss.'}` },
        { t: 'Backward · cross the sigmoid', d: `The local derivative of the sigmoid is σ′(z) = a(1 − a) = ${f4(Q.dadz)}. Chain rule: ∂L/∂z = ${f4(Q.dLda)} × ${f4(Q.dadz)} = <b>${f4(Q.dLdz)}</b>.` },
        { t: 'Backward · the + node copies', d: `At a sum node, the local derivative is 1 for each input: the gradient is <b>copied</b>. ∂L/∂m = ∂L/∂b = <b>${f4(Q.dLdz)}</b>.` },
        { t: 'Backward · the × node swaps', d: `In a product, the derivative with respect to one factor is the <b>other</b> factor: ∂L/∂w = ∂L/∂m · x = <b>${f4(Q.dLdw)}</b> and ∂L/∂x = ∂L/∂m · w = ${f4(Q.dLdx)}. The finite difference gives ${f4(Q.fd)}: it checks out.` },
        { t: 'Update the weights', d: `Gradient descent with η = ${nf(S.eta, 1)}: w ← ${f3(S.w)} − η·(${f4(Q.dLdw)}) = <b>${f3(Q.w2)}</b> and b ← <b>${f3(Q.b2)}</b>. The loss ${Q.L2 < Q.L ? 'drops' : 'rises (step too large)'} from ${f4(Q.L)} to <b>${f4(Q.L2)}</b>.` },
      ];
      const steps = defs();
      const st = UI.stepper(fig, steps, { autoplay: 6500 });
      const refresh = () => { compute(); const d = defs(); d.forEach((s, i) => { steps[i].t = s.t; steps[i].d = s.d; }); st.go(st.i); rFD.set(f4(Q.dLdw) + ' ≈ ' + f4(Q.fd)); };
      const ctl = UI.controls(fig);
      UI.slider(ctl, { label: 'x', min: -3, max: 3, step: 0.1, value: S.x, fmt: (v) => nf(v, 1), onInput: (v) => { S.x = v; refresh(); } });
      UI.slider(ctl, { label: 'w', min: -3, max: 3, step: 0.1, value: S.w, fmt: (v) => nf(v, 1), onInput: (v) => { S.w = v; refresh(); } });
      UI.slider(ctl, { label: 'b', min: -3, max: 3, step: 0.1, value: S.b, fmt: (v) => nf(v, 1), onInput: (v) => { S.b = v; refresh(); } });
      UI.slider(ctl, { label: 'target y', min: 0, max: 1, step: 0.05, value: S.y, fmt: (v) => nf(v, 2), onInput: (v) => { S.y = v; refresh(); } });
      const rFD = UI.readout(ctl, '∂L/∂w vs finite difference', f4(Q.dLdw) + ' ≈ ' + f4(Q.fd));
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const narrow = w < 600;
        const s = st.i, el = st.elapsed();
        const fade = V.clamp(el / 0.6, 0, 1);
        const gh = narrow ? 230 : h - 28;
        const P = narrow
          ? { x: [0.1, 0.14], w: [0.1, 0.62], mul: [0.32, 0.38], b: [0.52, 0.8], add: [0.52, 0.38], sig: [0.72, 0.38], y: [0.9, 0.8], L: [0.9, 0.38] }
          : { x: [0.07, 0.24], w: [0.07, 0.66], mul: [0.27, 0.45], b: [0.43, 0.84], add: [0.47, 0.45], sig: [0.67, 0.45], y: [0.9, 0.84], L: [0.9, 0.45] };
        const pos = (k) => [P[k][0] * w, 14 + P[k][1] * gh];
        const R = narrow ? 17 : 24;
        const inputs = { x: 'x', w: 'w', b: 'b', y: 'y (target)' };
        const ibw = narrow ? 64 : 96;
        const edges = [
          { a: 'x', b: 'mul', fw: [1], bw: 8, g: () => '∂L/∂x = ' + f4(Q.dLdx), loc: '× w' },
          { a: 'w', b: 'mul', fw: [1], bw: 8, g: () => '∂L/∂w = ' + f4(Q.dLdw), loc: '× x' },
          { a: 'mul', b: 'add', fw: [2], bw: 7, g: () => '∂L/∂m = ' + f4(Q.dLdz), loc: '× 1', v: () => 'm = ' + f3(Q.m), vs: 1 },
          { a: 'b', b: 'add', fw: [2], bw: 7, g: () => '∂L/∂b = ' + f4(Q.dLdb), loc: '× 1' },
          { a: 'add', b: 'sig', fw: [3], bw: 6, g: () => '∂L/∂z = ' + f4(Q.dLdz), loc: '× σ′ = ' + f4(Q.dadz), v: () => 'z = ' + f3(Q.z), vs: 2 },
          { a: 'sig', b: 'L', fw: [4], bw: 5, g: () => '∂L/∂a = ' + f4(Q.dLda), loc: '', v: () => 'a = ' + f4(Q.a), vs: 3 },
          { a: 'y', b: 'L', fw: [4], bw: -1, g: null, loc: '' },
        ];
        const edgePts = (e) => {
          const A = pos(e.a), B = pos(e.b);
          const ang = Math.atan2(B[1] - A[1], B[0] - A[0]);
          let x1 = A[0], y1 = A[1];
          if (inputs[e.a]) { const dx = Math.cos(ang), dy = Math.sin(ang); const tx = dx ? (ibw / 2) / Math.abs(dx) : 1e9, ty = dy ? 15 / Math.abs(dy) : 1e9; const tt = Math.min(tx, ty) + 2; x1 += dx * tt; y1 += dy * tt; }
          else { x1 += Math.cos(ang) * R; y1 += Math.sin(ang) * R; }
          return [x1, y1, B[0] - Math.cos(ang) * R, B[1] - Math.sin(ang) * R, ang];
        };
        // arestas
        edges.forEach((e) => {
          const [x1, y1, x2, y2, ang] = edgePts(e);
          const fwOn = e.fw.includes(s), bwOn = e.bw === s;
          V.arrow(g, x1, y1, x2, y2, { color: fwOn ? C.a : bwOn ? C.c : V.alpha(C.muted, 0.5), lw: fwOn || bwOn ? 2.4 : 1.4, head: 8 });
          if (fwOn) for (let q = 0; q < 3; q++) { const u = (el * 0.8 + q / 3) % 1; const p = V.along(x1, y1, x2, y2, u); V.dot(g, p[0], p[1], 3, C.a); }
          if (bwOn) for (let q = 0; q < 3; q++) { const u = (el * 0.8 + q / 3) % 1; const p = V.along(x2, y2, x1, y1, u); V.glow(g, p[0], p[1], 8, C.c, 0.9); V.dot(g, p[0], p[1], 3, C.c); }
          if (narrow) return;
          const mx = (x1 + x2) / 2, my = (y1 + y2) / 2;
          const horiz = Math.abs(Math.sin(ang)) < 0.3;
          if (e.v && s >= e.vs) {
            g.globalAlpha = s === e.vs ? fade : 1;
            pill(g, e.v(), mx, my - 20, { size: 11.5, fill: V.alpha(C.panel, 0.95), stroke: V.alpha(C.a, 0.7), color: C.a, align: 'center' });
            g.globalAlpha = 1;
          }
          if (e.g && e.bw > 0 && s >= e.bw) {
            g.globalAlpha = s === e.bw ? fade : 1;
            let gx = mx, gy = my, ly = my + 22;
            if (horiz) { gy = my + 22; ly = gy + 22; }
            else if (e.a === 'b') { gx = mx + 64; gy = my + 8; ly = gy + 22; }
            let lx = gx, lal = 'center';
            if (e.a === 'x' || e.a === 'w') { const pp = V.along(x1, y1, x2, y2, 0.34); gx = pp[0]; gy = pp[1]; ly = e.a === 'x' ? gy + 26 : gy - 22; lx = 14; lal = 'left'; }
            else lx = gx;
            pill(g, e.g(), gx, gy, { size: 11.5, fill: V.alpha(C.panel, 0.95), stroke: V.alpha(C.c, 0.8), color: C.c, align: 'center' });
            if (s === e.bw && e.loc) T(g, 'local derivative ' + e.loc, lx, ly, { size: 11, font: 'mono', color: C.muted, align: lal });
            g.globalAlpha = 1;
          }
        });
        // nós de entrada
        Object.keys(inputs).forEach((k) => {
          const [X, Y] = pos(k);
          const val = k === 'x' ? S.x : k === 'w' ? S.w : k === 'b' ? S.b : S.y;
          const upd = s === 9 && (k === 'w' || k === 'b');
          const txt = upd ? `${k} = ${nf(k === 'w' ? Q.w2 : Q.b2, narrow ? 2 : 3)}` : `${k} = ${nf(val, 2)}`;
          V.box(g, X - ibw / 2, Y - 15, ibw, 30, { r: 8, fill: upd ? V.alpha(C.d, 0.25) : V.alpha(C.panel, 0.95), stroke: upd ? C.d : k === 'w' || k === 'b' ? V.alpha(C.d, 0.7) : V.alpha(C.muted, 0.5), lw: upd ? 2 : 1.2 });
          T(g, txt, X, Y + 4.5, { size: narrow ? 11 : 12, font: 'mono', align: 'center', color: C.ink, weight: upd ? 700 : 500, maxW: ibw - 6 });
          if (k === 'w' || k === 'b') T(g, 'parameter', X, Y + 28, { size: 11, align: 'center', color: V.alpha(C.d, 0.9) });
          if (k === 'y') T(g, 'target', X, Y + 28, { size: 11, align: 'center', color: C.muted });
        });
        // nós de operação
        const ops = { mul: '×', add: '+', sig: 'σ', L: 'L' };
        Object.keys(ops).forEach((k) => {
          const [X, Y] = pos(k);
          const on = (k === 'mul' && (s === 1 || s === 8)) || (k === 'add' && (s === 2 || s === 7)) || (k === 'sig' && (s === 3 || s === 6)) || (k === 'L' && (s === 4 || s === 5));
          const col = k === 'L' ? C.c : C.b;
          if (on) V.glow(g, X, Y, R + 16, s <= 4 ? C.a : C.c, 0.5);
          V.box(g, X - R, Y - R, 2 * R, 2 * R, { r: R, fill: V.alpha(col, on ? 0.3 : 0.14), stroke: col, lw: on ? 2.4 : 1.6 });
          T(g, ops[k], X, Y + (narrow ? 6 : 7), { size: narrow ? 17 : 20, weight: 700, align: 'center', color: C.ink });
          if (k === 'L' && s >= 4 && !narrow) {
            const lv = s === 9 ? Q.L2 : Q.L;
            pill(g, (s === 9 ? 'new L = ' : 'L = ') + f4(lv), X, Y - R - 16, { size: 11.5, fill: V.alpha(C.panel, 0.95), stroke: V.alpha(s === 9 ? C.d : C.a, 0.8), color: s === 9 ? C.d : C.a, align: 'center' });
          }
        });
        if (narrow) {
          // lista de valores e gradientes (substitui os rótulos das arestas)
          const rows = [
            [1, 'm = w·x', f3(Q.m), C.a], [2, 'z = m + b', f3(Q.z), C.a], [3, 'a = σ(z)', f4(Q.a), C.a], [4, 'L = ½(a − y)²', f4(Q.L), C.a],
            [5, '∂L/∂a = a − y', f4(Q.dLda), C.c], [6, '∂L/∂z = ∂L/∂a · a(1−a)', f4(Q.dLdz), C.c], [7, '∂L/∂b = ∂L/∂m = ∂L/∂z', f4(Q.dLdz), C.c],
            [8, '∂L/∂w = ∂L/∂m · x', f4(Q.dLdw), C.c], [8, '∂L/∂x = ∂L/∂m · w', f4(Q.dLdx), C.c], [9, 'L after the step', f4(Q.L2), C.d],
          ];
          const ly0 = gh + 36, rh = Math.min(24, (h - ly0 - 26) / rows.length);
          rows.forEach(([k, lab, val, col], i) => {
            const Y = ly0 + i * rh, vis = s >= k, cur = s === k;
            if (cur) V.box(g, 10, Y - rh / 2 - 1, w - 20, rh, { r: 6, fill: V.alpha(col, 0.14) });
            g.globalAlpha = vis ? (cur ? Math.max(0.3, fade) : 1) : 0.25;
            T(g, lab, 18, Y + 4, { size: 11.5, font: 'mono', color: vis ? C.ink : C.muted, maxW: w - 110 });
            T(g, vis ? val : '·', w - 18, Y + 4, { size: 12, font: 'mono', weight: 600, align: 'right', color: col });
            g.globalAlpha = 1;
          });
        }
        T(g, s <= 0 ? 'structure' : s <= 4 ? '→ forward pass' : s <= 8 ? '← backward pass' : '↻ update', 14, narrow ? gh + 12 : h - 10, { size: 12, weight: 600, color: s <= 4 && s > 0 ? C.a : s <= 8 && s > 0 ? C.c : C.d });
      });
    });

    /* ====================================================================
       Figura 6b — Backprop de uma camada, em matrizes
       ==================================================================== */
    safe('fig-backprop-matriz', () => {
      const fig = $('#fig-backprop-matriz');
      fitH(fig, (w) => (w < 600 ? 300 : 290));
      const NO = 3, NI = 4;
      const S = { seed: 4, act: 'sigmoid' };
      const M = {};
      const r1 = (x) => Math.round(x * 10) / 10;
      function gen() {
        const r = V.rng(S.seed);
        const nz = (lo, hi) => r1((r() < 0.5 ? -1 : 1) * (lo + r() * (hi - lo)));  // nunca zero
        M.W = Array.from({ length: NO }, () => Array.from({ length: NI }, () => nz(0.2, 1.2)));
        M.a = Array.from({ length: NI }, () => nz(0.3, 1.5));
        M.gA = Array.from({ length: NO }, () => nz(0.4, 1.6));
        recompute();
      }
      function recompute() {
        const fa = ACT[S.act];
        M.z = M.W.map((row) => row.reduce((q, v, j) => q + v * M.a[j], 0));
        M.fp = M.z.map((z) => fa.d(z));
        M.d = M.z.map((z, i) => M.fp[i] * M.gA[i]);
        M.gW = M.d.map((di) => M.a.map((aj) => di * aj));
        M.gX = M.a.map((_, j) => M.d.reduce((q, di, i) => q + M.W[i][j] * di, 0));
        M.zeroRows = M.d.map((v) => Math.abs(v) < 1e-12);
      }
      gen();
      const steps = [
        { t: 'The error signal of each neuron', d: '' },
        { t: 'The weight gradient: an outer product', d: '∂L/∂W = δ · a<sub>prev</sub><sup>⊤</sup>. The column δ (3 numbers) times the row a<sub>prev</sub><sup>⊤</sup> (4 numbers) gives a 3 × 4 matrix, the same size as W. Cell (i, j) is δ<sub>i</sub> · a<sub>j</sub>: “error of neuron i” × “input j”. That is how each of the 12 weights gets its own gradient.' },
        { t: 'The error sent back: the same matrix, transposed', d: '∂L/∂a<sub>prev</sub> = W<sup>⊤</sup> δ. To send the error back to the layer below, the network uses W itself, rotated: rows become columns. Input j receives Σ<sub>i</sub> W<sub>ij</sub> δ<sub>i</sub>, the sum of the errors of the neurons it fed, each weighted by the strength of the connection.' },
      ];
      const d0 = () => (S.act === 'relu'
        ? 'δ = f′(z) ⊙ ∂L/∂a. Each neuron multiplies the gradient coming from above (∂L/∂a) by its local derivative f′(z). With ReLU, f′ is 0 or 1: any neuron left with z &lt; 0 zeroes its own δ and learns nothing in this step.'
        : 'δ = f′(z) ⊙ ∂L/∂a. Each neuron multiplies the gradient coming from above (∂L/∂a) by its local derivative f′(z). With the sigmoid, f′ = σ(1 − σ) ≤ 0.25: the signal always shrinks as it crosses the activation.');
      steps[0].d = d0();
      const st = UI.stepper(fig, steps, { autoplay: 8000 });
      const ctl = UI.controls(fig);
      const rZ = UI.readout(ctl, 'zeroed rows of ∂L/∂W', '0 of 3');
      const refresh = () => { recompute(); steps[0].d = d0(); st.go(st.i); const n = M.zeroRows.filter(Boolean).length; rZ.set(n + ' of 3'); };
      UI.seg(ctl, { label: 'Activation f', options: [{ v: 'sigmoid', label: 'sigmoid' }, { v: 'relu', label: 'ReLU' }], value: S.act, onChange: (v) => { S.act = v; refresh(); } });
      UI.button(ctl, '↻ randomize', () => { S.seed = (S.seed * 7 + 3) % 997; gen(); refresh(); });
      V.fig(fig, (g, w, h) => {
        const C = V.C();
        const narrow = w < 600;
        const s = st.i, el = st.elapsed();
        const cw = narrow ? 46 : 64, ch = narrow ? 32 : 38, gp = 5;
        const fc = (v) => { const a = Math.abs(v); return a < 5e-4 ? '0' : nf(v, a < 0.1 ? 3 : 2); };
        const head = (txt) => wrap(g, txt, 16, 20, w - 32, 17, { size: narrow ? 12 : 12.5, font: 'mono', weight: 600, color: C.ink, maxLines: 2 });
        const cell = (X, Y, v, sc, o = {}) => {
          const k = V.clamp(v / sc, -1, 1), col = k >= 0 ? C.a : C.b;
          const al = o.alpha == null ? 1 : o.alpha;
          g.globalAlpha = al;
          if (o.glow) V.glow(g, X + cw / 2, Y + ch / 2, cw * 0.7, o.glowCol || C.c, 0.5);
          V.box(g, X, Y, cw, ch, { r: 7, fill: V.alpha(o.zero ? C.muted : col, o.zero ? 0.1 : 0.1 + 0.45 * Math.abs(k)), stroke: o.hi ? (o.glowCol || C.c) : V.alpha(o.zero ? C.muted : col, 0.7), lw: o.hi ? 2.2 : 1.1 });
          T(g, o.txt != null ? o.txt : fc(v), X + cw / 2, Y + ch / 2 + 4.5, { size: narrow ? 11.5 : 12.5, font: 'mono', weight: o.hi ? 700 : 500, align: 'center', color: C.ink, maxW: cw - 4 });
          g.globalAlpha = 1;
        };
        const ghost = (X, Y) => { g.setLineDash([3, 3]); V.box(g, X, Y, cw, ch, { r: 7, fill: V.alpha(C.panel, 0.6), stroke: V.alpha(C.muted, 0.35), lw: 1 }); g.setLineDash([]); };
        const lab = (txt, x, y, o = {}) => T(g, txt, x, y, Object.assign({ size: 11.5, font: 'mono', color: C.muted, align: 'center' }, o));
        const top = narrow ? 50 : 36;
        if (s === 0) {
          // três colunas: ∂L/∂a ⊙ f′(z) = δ
          head(narrow ? 'δ = f′(z) ⊙ ∂L/∂a  (elementwise)' : 'δ = f′(z) ⊙ ∂L/∂a   (elementwise)');
          const gapX = narrow ? 34 : 64;
          const totalW = 3 * cw + 2 * gapX + (narrow ? 80 : 150);
          const x0 = Math.max(16, (w - totalW) / 2), y0 = top + 40;
          const xs = [x0, x0 + cw + gapX, x0 + 2 * (cw + gapX)];
          lab('∂L/∂a', xs[0] + cw / 2, y0 - 12); lab("f′(z)", xs[1] + cw / 2, y0 - 12); lab('δ', xs[2] + cw / 2, y0 - 12, { color: C.c, weight: 700 });
          const cur = Math.floor(el / 1.1) % NO;
          for (let i = 0; i < NO; i++) {
            const Y = y0 + i * (ch + gp + 10), on = i === cur, shown = el > 0.5 + i * 1.1 || el > 0.5 + NO * 1.1;
            cell(xs[0], Y, M.gA[i], 1.2, { hi: on, glowCol: C.c });
            cell(xs[1], Y, M.fp[i], S.act === 'relu' ? 1 : 0.25, { hi: on, glowCol: C.d, zero: M.fp[i] === 0 });
            cell(xs[2], Y, M.d[i], 0.3, { hi: on, glow: on, zero: M.zeroRows[i], alpha: shown ? 1 : 0.25 });
            lab('⊙', xs[0] + cw + gapX / 2, Y + ch / 2 + 5, { size: 16, color: C.muted });
            lab('=', xs[1] + cw + gapX / 2, Y + ch / 2 + 5, { size: 16, color: C.muted });
            T(g, 'z' + '₀₁₂'[i] + ' = ' + nf(M.z[i], 2), xs[2] + cw + (narrow ? 10 : 18), Y + ch / 2 + 4.5, { size: 11.5, font: 'mono', color: M.z[i] < 0 ? C.c : C.muted });
            if (on) {
              const u = (el % 1.1) / 1.1;
              const p1 = V.along(xs[0] + cw, Y + ch / 2, xs[2], Y + ch / 2, u);
              V.dot(g, p1[0], p1[1] - 6, 3.2, C.c);
            }
          }
          const yN = y0 + NO * (ch + gp + 10) + 14;
          wrap(g, S.act === 'relu' ? 'ReLU: f′ = 1 if z > 0, else 0. A neuron with z < 0 has δ = 0.' : 'Sigmoid: f′ = σ(z)(1 − σ(z)), always between 0 and 0.25.', 16, yN, w - 32, 16, { size: 12, color: C.d });
        } else if (s === 1) {
          head(narrow ? '∂L/∂W = δ · a_prevᵀ  (3×1)·(1×4) = 3×4' : '∂L/∂W = δ · a_prevᵀ   (3 × 1) · (1 × 4) = (3 × 4)');
          const gridW = NI * (cw + gp) - gp;
          const x0 = Math.max(16, (w - (cw + 22 + gridW)) / 2), gx0 = x0 + cw + 22, y0 = top + 50;
          const k = Math.floor((el - 0.4) / 0.36), cur = el < 0.4 ? -1 : Math.min(NO * NI, k);
          lab('a_prevᵀ', gx0 - 8, y0 - 30, { align: 'right' });
          for (let j = 0; j < NI; j++) cell(gx0 + j * (cw + gp), y0 - ch - 12, M.a[j], 1.5, { hi: cur >= 0 && cur < NO * NI && cur % NI === j, glowCol: C.a });
          lab('δ', x0 + cw / 2, y0 - 6, { color: C.c, weight: 700 });
          for (let i = 0; i < NO; i++) cell(x0, y0 + 4 + i * (ch + gp), M.d[i], 0.3, { hi: cur >= 0 && cur < NO * NI && Math.floor(cur / NI) === i, zero: M.zeroRows[i] });
          for (let i = 0; i < NO; i++) for (let j = 0; j < NI; j++) {
            const idx = i * NI + j, X = gx0 + j * (cw + gp), Y = y0 + 4 + i * (ch + gp);
            const done = idx < cur, now = idx === cur;
            if (done || now) cell(X, Y, M.gW[i][j], 0.4, { hi: now, glow: now, zero: M.zeroRows[i], alpha: now ? V.clamp(((el - 0.4) % 0.36) / 0.2, 0.3, 1) : 1 });
            else ghost(X, Y);
            if (now) {
              const u = V.clamp(((el - 0.4) % 0.36) / 0.3, 0, 1);
              const pa = V.along(x0 + cw, Y + ch / 2, X, Y + ch / 2, u), pb = V.along(X + cw / 2, y0 - 12, X + cw / 2, Y, u);
              V.dot(g, pa[0], pa[1], 3, C.c); V.dot(g, pb[0], pb[1], 3, C.a);
            }
          }
          const yN = y0 + 4 + NO * (ch + gp) + 22;
          const ci = V.clamp(cur, 0, NO * NI - 1), ii = Math.floor(ci / NI), jj = ci % NI;
          if (cur >= 0) T(g, `∂L/∂W${'₀₁₂'[ii]}${'₀₁₂₃'[jj]} = δ${'₀₁₂'[ii]} · a${'₀₁₂₃'[jj]} = ${fc(M.d[ii])} × ${nf(M.a[jj], 1)} = ${fc(M.gW[ii][jj])}`, 16, yN, { size: 12, font: 'mono', color: C.c, maxW: w - 32 });
          const nz = M.zeroRows.filter(Boolean).length;
          if (nz) T(g, nz + (nz > 1 ? ' zeroed rows: neurons with δ = 0 leave their weights untouched.' : ' zeroed row: the neuron with δ = 0 leaves its weights untouched.'), 16, yN + 20, { size: 11.5, color: C.d, maxW: w - 32 });
        } else {
          head(narrow ? '∂L/∂a_prev = Wᵀ · δ  (4×3)·(3×1) = 4×1' : '∂L/∂a_prev = Wᵀ · δ   (4 × 3) · (3 × 1) = (4 × 1)');
          const y0 = top + 30;
          const e = V.ease.inOut(V.clamp((el - 0.3) / 1.3, 0, 1));
          // posições: W (3×4) → Wᵀ (4×3)
          const wW = NI * (cw + gp) - gp, wT = NO * (cw + gp) - gp;
          const tot = wT + 22 + cw + 30 + cw;
          const x0 = Math.max(16, (w - Math.max(wW, tot)) / 2);
          for (let i = 0; i < NO; i++) for (let j = 0; j < NI; j++) {
            const Xa = x0 + j * (cw + gp), Ya = y0 + i * (ch + gp);
            const Xb = x0 + i * (cw + gp), Yb = y0 + j * (ch + gp);
            const X = V.lerp(Xa, Xb, e), Y = V.lerp(Ya, Yb, e);
            const row = el > 1.8 ? Math.floor((el - 1.8) / 1.2) : -1;
            cell(X, Y, M.W[i][j], 1.2, { hi: e >= 1 && row >= 0 && row < NI && j === row, glowCol: C.b });
          }
          lab(e < 0.5 ? 'W  (3 × 4)' : 'Wᵀ  (4 × 3)', x0 + (e < 0.5 ? wW : wT) / 2, y0 - 10, { color: C.b, weight: 600 });
          if (e >= 1) {
            const dx = x0 + wT + 22, rx = dx + cw + 30;
            lab('δ', dx + cw / 2, y0 - 10, { color: C.c, weight: 700 });
            lab('·', x0 + wT + 11, y0 + (NI * (ch + gp)) / 2, { size: 18 });
            for (let i = 0; i < NO; i++) cell(dx, y0 + i * (ch + gp), M.d[i], 0.3, { zero: M.zeroRows[i] });
            lab('=', dx + cw + 15, y0 + (NI * (ch + gp)) / 2, { size: 16 });
            lab('∂L/∂a_prev', rx + cw / 2, y0 - 10, { color: C.e, weight: 600 });
            const row = el > 1.8 ? Math.floor((el - 1.8) / 1.2) : -1;
            for (let j = 0; j < NI; j++) {
              const Y = y0 + j * (ch + gp);
              if (j < row || (row >= NI)) cell(rx, Y, M.gX[j], 0.4, {});
              else if (j === row) cell(rx, Y, M.gX[j], 0.4, { hi: true, glow: true, glowCol: C.e, alpha: V.clamp(((el - 1.8) % 1.2) / 0.5, 0.3, 1) });
              else ghost(rx, Y);
            }
            if (row >= 0 && row < NI) {
              const terms = M.d.map((di, i) => nf(M.W[i][row], 1) + '·(' + fc(di) + ')').join(' + ');
              wrap(g, `∂L/∂a${'₀₁₂₃'[row]} = ${terms} = ${fc(M.gX[row])}`, 16, y0 + NI * (ch + gp) + 20, w - 32, 16, { size: 11.5, font: 'mono', color: C.e, maxLines: 2 });
            }
          } else T(g, 'rotating W: row i becomes column i', x0, y0 + NI * (ch + gp) + 20, { size: 12, color: C.b });
        }
      });
    });

    /* ====================================================================
       Figura 7 — Laboratório: uma MLP treinando ao vivo
       ==================================================================== */
    safe('fig-playground', () => {
      const fig = $('#fig-playground');
      fitH(fig, (w) => (w < 600 ? Math.round(w + 300) : 440));
      const S = { data: 'espiral', act: 'silu', width: 8, depth: 2, train: true };
      const AF = { relu: ACT.relu, tanh: ACT.tanh, silu: ACT.silu, linear: ACT.linear };
      let D = null, net = null, stepN = 0, hist = [], histEvery = 5, histAcc = 0, histCnt = 0, acc = 0, lossEma = null;
      const rs = V.rng(99);
      function makeData(kind) {
        const r = V.rng(kind === 'espiral' ? 11 : kind === 'circulos' ? 12 : 13);
        const X = [], Y = [], n = 200;
        if (kind === 'espiral') for (let i = 0; i < n; i++) { const c = i % 2, k = Math.floor(i / 2), tt = k / (n / 2), rad = 0.1 + 0.85 * tt, ang = tt * 1.5 * 2 * Math.PI + c * Math.PI; X.push([rad * Math.cos(ang) + r.normal() * 0.02, rad * Math.sin(ang) + r.normal() * 0.02]); Y.push(c ? 0 : 1); }
        else if (kind === 'circulos') for (let i = 0; i < n; i++) { const c = i % 2, rad = c ? 0.62 + r() * 0.3 : r() * 0.38, ang = r() * 2 * Math.PI; X.push([rad * Math.cos(ang), rad * Math.sin(ang)]); Y.push(c ? 0 : 1); }
        else for (let i = 0; i < n; i++) { const qx = r() < 0.5 ? -1 : 1, qy = r() < 0.5 ? -1 : 1; X.push([qx * (0.12 + r() * 0.75), qy * (0.12 + r() * 0.75)]); Y.push(qx * qy > 0 ? 1 : 0); }
        return { X, Y, n };
      }
      function build() {
        const sizes = [2]; for (let i = 0; i < S.depth; i++) sizes.push(S.width); sizes.push(1);
        const r = V.rng(7);
        const Ls = [];
        for (let l = 0; l < sizes.length - 1; l++) {
          const nin = sizes[l], nout = sizes[l + 1];
          const std = S.act === 'relu' || S.act === 'silu' ? Math.sqrt(2 / nin) : Math.sqrt(1 / nin);
          const W = new Float64Array(nin * nout); for (let k = 0; k < W.length; k++) W[k] = r.normal() * std;
          Ls.push({ nin, nout, W, b: new Float64Array(nout), gW: new Float64Array(nin * nout), gb: new Float64Array(nout), mW: new Float64Array(nin * nout), vW: new Float64Array(nin * nout), mb: new Float64Array(nout), vb: new Float64Array(nout), z: new Float64Array(nout), a: new Float64Array(nout), d: new Float64Array(nout) });
        }
        net = { sizes, Ls, t: 0, inp: new Float64Array(2) };
        D = makeData(S.data);
        stepN = 0; hist = []; histEvery = 5; histAcc = 0; histCnt = 0; acc = 0; lossEma = null;
        let P = 0; for (let l = 0; l < sizes.length - 1; l++) P += sizes[l] * sizes[l + 1] + sizes[l + 1];
        rP.set(String(P));
        heatDirty = true;
      }
      function fwd(x0, x1) {
        const f = AF[S.act].f, Ls = net.Ls; net.inp[0] = x0; net.inp[1] = x1; let inp = net.inp;
        for (let l = 0; l < Ls.length; l++) { const L = Ls[l], last = l === Ls.length - 1; for (let o = 0; o < L.nout; o++) { let s = L.b[o]; const off = o * L.nin; for (let i = 0; i < L.nin; i++) s += L.W[off + i] * inp[i]; L.z[o] = s; L.a[o] = last ? s : f(s); } inp = L.a; }
        return sig(Ls[Ls.length - 1].z[0]);
      }
      function bwd(y, p) {
        const df = AF[S.act].d, Ls = net.Ls, nL = Ls.length;
        Ls[nL - 1].d[0] = p - y;
        for (let l = nL - 1; l >= 0; l--) {
          const L = Ls[l], inp = l === 0 ? net.inp : Ls[l - 1].a;
          for (let o = 0; o < L.nout; o++) { const d = L.d[o]; L.gb[o] += d; const off = o * L.nin; for (let i = 0; i < L.nin; i++) L.gW[off + i] += d * inp[i]; }
          if (l > 0) { const Pl = Ls[l - 1]; for (let i = 0; i < L.nin; i++) { let s = 0; for (let o = 0; o < L.nout; o++) s += L.W[o * L.nin + i] * L.d[o]; Pl.d[i] = s * df(Pl.z[i]); } }
        }
      }
      function adam(lr, bs) {
        net.t++; const b1 = 0.9, b2 = 0.999, c1 = 1 - Math.pow(b1, net.t), c2 = 1 - Math.pow(b2, net.t);
        for (const L of net.Ls) {
          for (let k = 0; k < L.W.length; k++) { const gg = L.gW[k] / bs; L.mW[k] = b1 * L.mW[k] + (1 - b1) * gg; L.vW[k] = b2 * L.vW[k] + (1 - b2) * gg * gg; L.W[k] -= (lr * (L.mW[k] / c1)) / (Math.sqrt(L.vW[k] / c2) + 1e-8); L.gW[k] = 0; }
          for (let k = 0; k < L.b.length; k++) { const gg = L.gb[k] / bs; L.mb[k] = b1 * L.mb[k] + (1 - b1) * gg; L.vb[k] = b2 * L.vb[k] + (1 - b2) * gg * gg; L.b[k] -= (lr * (L.mb[k] / c1)) / (Math.sqrt(L.vb[k] / c2) + 1e-8); L.gb[k] = 0; }
        }
      }
      function trainStep() {
        const bs = 24; let lsum = 0;
        for (let k = 0; k < bs; k++) { const i = Math.floor(rs() * D.n); const p = fwd(D.X[i][0], D.X[i][1]); const y = D.Y[i]; lsum += -(y ? Math.log(p + 1e-9) : Math.log(1 - p + 1e-9)); bwd(y, p); }
        adam(0.02, bs);
        stepN++;
        const lb = lsum / bs;
        lossEma = lossEma == null ? lb : 0.9 * lossEma + 0.1 * lb;
        histAcc += lossEma; histCnt++;
        if (histCnt >= histEvery) { hist.push(histAcc / histCnt); histAcc = 0; histCnt = 0; if (hist.length > 240) { const nh = []; for (let i = 0; i + 1 < hist.length; i += 2) nh.push((hist[i] + hist[i + 1]) / 2); hist = nh; histEvery *= 2; } }
        if (stepN % 20 === 0) { let c = 0; for (let i = 0; i < D.n; i++) { const p = fwd(D.X[i][0], D.X[i][1]); if ((p > 0.5) === (D.Y[i] === 1)) c++; } acc = c / D.n; rAcc.set(Math.round(acc * 100) + '%'); }
      }
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Data', options: [{ v: 'espiral', label: 'spiral' }, { v: 'circulos', label: 'circles' }, { v: 'xor', label: 'XOR' }], value: S.data, onChange: (v) => { S.data = v; build(); } });
      UI.seg(ctl, { label: 'Activation', options: [{ v: 'relu', label: 'ReLU' }, { v: 'tanh', label: 'tanh' }, { v: 'silu', label: 'SiLU' }, { v: 'linear', label: 'linear' }], value: S.act, onChange: (v) => { S.act = v; build(); } });
      UI.slider(ctl, { label: 'width', min: 2, max: 16, step: 1, value: S.width, fmt: (v) => String(v), onInput: (v) => { if (v !== S.width) { S.width = v; build(); } } });
      UI.slider(ctl, { label: 'layers', min: 1, max: 4, step: 1, value: S.depth, fmt: (v) => String(v), onInput: (v) => { if (v !== S.depth) { S.depth = v; build(); } } });
      UI.button(ctl, '↺ reset', () => build());
      UI.toggle(ctl, { label: 'train', value: true, onChange: (v) => { S.train = v; } });
      const rP = UI.readout(ctl, 'parameters', '—');
      const rAcc = UI.readout(ctl, 'accuracy', '—');
      const G = 40;
      const heat = document.createElement('canvas'); heat.width = G; heat.height = G;
      const hctx = heat.getContext('2d');
      const img = hctx.createImageData(G, G);
      let heatDirty = true, frame = 0;
      build();
      function renderHeat(C) {
        const bg = rgbOf(C.bg), ca = rgbOf(C.a), cb = rgbOf(C.c);
        for (let j = 0; j < G; j++) for (let i = 0; i < G; i++) {
          const x0 = -1.1 + (2.2 * (i + 0.5)) / G, x1 = 1.1 - (2.2 * (j + 0.5)) / G;
          const p = fwd(x0, x1);
          const base = p >= 0.5 ? ca : cb, k = 0.12 + 0.5 * Math.pow(Math.abs(2 * p - 1), 0.7);
          const o = (j * G + i) * 4;
          img.data[o] = bg[0] + (base[0] - bg[0]) * k; img.data[o + 1] = bg[1] + (base[1] - bg[1]) * k; img.data[o + 2] = bg[2] + (base[2] - bg[2]) * k; img.data[o + 3] = 255;
        }
        hctx.putImageData(img, 0, 0);
      }
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const narrow = w < 600;
        if (S.train && !LA.paused) { for (let k = 0; k < 8; k++) trainStep(); heatDirty = true; }
        frame++;
        if (heatDirty && (frame % 2 === 0 || !S.train)) { renderHeat(C); heatDirty = false; }
        let ps, px, py, nx0, ny0, nw, nh, lx0, ly0, lw, lh;
        if (!narrow) { ps = h - 24; px = 12; py = 12; nx0 = px + ps + 28; nw = w - nx0 - 16; ny0 = 22; nh = h * 0.56; lx0 = nx0; ly0 = ny0 + nh + 30; lw = nw; lh = h - ly0 - 30; }
        else { ps = w - 24; px = 12; py = 12; nx0 = 16; nw = w - 32; ny0 = py + ps + 22; nh = 150; lx0 = 16; ly0 = ny0 + nh + 26; lw = w - 32; lh = h - ly0 - 30; }
        g.imageSmoothingEnabled = true;
        g.drawImage(heat, px, py, ps, ps);
        g.strokeStyle = V.alpha(C.muted, 0.35); g.lineWidth = 1; g.strokeRect(px, py, ps, ps);
        const toX = (u) => px + ((u + 1.1) / 2.2) * ps, toY = (v) => py + ((1.1 - v) / 2.2) * ps;
        for (let i = 0; i < D.n; i++) {
          const X = toX(D.X[i][0]), Y = toY(D.X[i][1]);
          g.beginPath(); g.arc(X, Y, narrow ? 3 : 3.4, 0, 7); g.fillStyle = D.Y[i] ? C.a : C.c; g.fill(); g.strokeStyle = C.bg; g.lineWidth = 1.2; g.stroke();
        }
        // rede
        const sizes = net.sizes, nLay = sizes.length;
        const colX = (l) => nx0 + 16 + ((nw - 32) * l) / (nLay - 1);
        const nodeY = (l, i) => { const n = sizes[l]; const sp = Math.min(22, (nh - 20) / Math.max(1, n - 1)); const tot = sp * (n - 1); return ny0 + nh / 2 - tot / 2 + i * sp; };
        const buckets = {};
        for (let l = 0; l < nLay - 1; l++) {
          const L = net.Ls[l]; let mx = 1e-6; for (let k = 0; k < L.W.length; k++) mx = Math.max(mx, Math.abs(L.W[k]));
          for (let o = 0; o < L.nout; o++) for (let i = 0; i < L.nin; i++) {
            const v = L.W[o * L.nin + i], m = Math.abs(v) / mx, lvl = m > 0.66 ? 2 : m > 0.33 ? 1 : 0, key = (v >= 0 ? 'p' : 'n') + lvl;
            (buckets[key] = buckets[key] || []).push([colX(l), nodeY(l, i), colX(l + 1), nodeY(l + 1, o)]);
          }
        }
        Object.keys(buckets).forEach((key) => {
          const lvl = +key[1];
          g.beginPath(); buckets[key].forEach(([a, b, c, d]) => { g.moveTo(a, b); g.lineTo(c, d); });
          g.strokeStyle = V.alpha(key[0] === 'p' ? C.a : C.b, [0.14, 0.35, 0.7][lvl]); g.lineWidth = [0.7, 1.3, 2.2][lvl]; g.stroke();
        });
        for (let l = 0; l < nLay; l++) for (let i = 0; i < sizes[l]; i++) {
          const X = colX(l), Y = nodeY(l, i);
          g.beginPath(); g.arc(X, Y, l === 0 || l === nLay - 1 ? 7 : 5, 0, 7); g.fillStyle = C.panel; g.fill(); g.strokeStyle = l === 0 ? C.muted : l === nLay - 1 ? C.d : C.b; g.lineWidth = 1.6; g.stroke();
        }
        T(g, 'x₁', colX(0) - 12, nodeY(0, 0) + 4, { size: 11, font: 'mono', align: 'right', color: C.muted });
        T(g, 'x₂', colX(0) - 12, nodeY(0, 1) + 4, { size: 11, font: 'mono', align: 'right', color: C.muted });
        T(g, 'p', colX(nLay - 1) + 12, nodeY(nLay - 1, 0) + 4, { size: 11, font: 'mono', color: C.d });
        T(g, `network ${sizes.join(' → ')} · ${S.act}`, nx0, ny0 - 6, { size: 11.5, font: 'mono', color: C.muted, maxW: nw });
        // curva de perda
        V.box(g, lx0, ly0, lw, lh, { r: 8, fill: V.alpha(C.panel, 0.6), stroke: V.alpha(C.muted, 0.25), lw: 1 });
        if (hist.length > 1) {
          const mxL = Math.max(0.75, ...hist);
          g.beginPath();
          hist.forEach((v, i) => { const X = lx0 + 6 + ((lw - 12) * i) / Math.max(1, hist.length - 1), Y = ly0 + lh - 6 - (lh - 12) * Math.min(1, v / mxL); if (i) g.lineTo(X, Y); else g.moveTo(X, Y); });
          g.strokeStyle = C.c; g.lineWidth = 2; g.stroke();
        }
        T(g, 'loss (cross-entropy) during training', lx0, ly0 - 7, { size: 11, color: C.muted, maxW: lw });
        T(g, `step ${ni(stepN)} · loss ${lossEma == null ? '—' : nf(lossEma, 3)} · accuracy ${Math.round(acc * 100)}%`, lx0, ly0 + lh + 18, { size: 12, font: 'mono', color: C.ink, maxW: lw });
      });
    });

    /* ====================================================================
       Figura 8 — LayerNorm, RMSNorm e o papel do ε
       ==================================================================== */
    safe('fig-normas', () => {
      const fig = $('#fig-normas');
      fitH(fig, (w) => (w < 600 ? 640 : 440));
      const D = 16;
      const r = V.rng(5);
      const base = Array.from({ length: D }, () => 0.55 + r.normal() * 0.9);
      const wStored = Array.from({ length: D }, (_, i) => 0.22 * Math.sin(i * 1.3 + 0.4));
      const beta = Array.from({ length: D }, (_, i) => 0.12 * Math.cos(i * 0.9));
      const TYPES = { ln: 'LayerNorm', rms: 'RMSNorm', rms1w: 'RMSNorm (1+w)', grp: 'grouped 4×4', bug: 'bug: w without 1+' };
      const S = { type: 'rms', scaleLog: 0, eps: 1e-5, outlier: false, t0: -5 };
      let H = null;
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Type', options: Object.keys(TYPES).map((k) => ({ v: k, label: TYPES[k] })), value: S.type, onChange: (v) => { S.type = v; if (H) S.t0 = H.t; } });
      UI.slider(ctl, { label: 'scale', min: -4, max: 3, step: 0.1, value: 0, fmt: (v) => '×10^(' + nf(v, 1) + ')', onInput: (v) => { S.scaleLog = v; } });
      UI.seg(ctl, { label: 'ε', options: [{ v: 1e-20, label: '1e-20 · DS' }, { v: 1e-6, label: '1e-6 · Qwen' }, { v: 1e-5, label: '1e-5 · GLM' }, { v: 1e-2, label: '1e-2' }], value: 1e-5, onChange: (v) => { S.eps = v; } });
      UI.toggle(ctl, { label: 'outlier', value: false, onChange: (v) => { S.outlier = v; if (H) S.t0 = H.t; } });
      const rIn = UI.readout(ctl, 'input RMS', '—');
      const rOut = UI.readout(ctl, 'normalized RMS', '—');
      let lastIn = '', lastOut = '';
      function compute() {
        const s = Math.pow(10, S.scaleLog);
        const x = base.map((v, i) => v * s * (S.outlier && i === 5 ? 14 : 1));
        const groups = S.type === 'grp' ? 4 : 1, gs = D / groups;
        const xhat = new Array(D), out = new Array(D), gstat = [];
        for (let gi = 0; gi < groups; gi++) {
          const a = gi * gs, b = a + gs;
          let mu = 0; for (let i = a; i < b; i++) mu += x[i]; mu /= gs;
          let ms = 0;
          if (S.type === 'ln') { for (let i = a; i < b; i++) ms += (x[i] - mu) ** 2; ms /= gs; }
          else { for (let i = a; i < b; i++) ms += x[i] * x[i]; ms /= gs; }
          const den = Math.sqrt(ms + S.eps);
          for (let i = a; i < b; i++) xhat[i] = S.type === 'ln' ? (x[i] - mu) / den : x[i] / den;
          gstat.push({ mu, ms });
        }
        for (let i = 0; i < D; i++) {
          const gain = S.type === 'bug' ? wStored[i] : 1 + wStored[i];
          out[i] = gain * xhat[i] + (S.type === 'ln' ? beta[i] : 0);
        }
        const rms = (v) => Math.sqrt(v.reduce((q, z) => q + z * z, 0) / v.length);
        const mean = (v) => v.reduce((q, z) => q + z, 0) / v.length;
        return { x, xhat, out, gstat, rmsX: rms(x), rmsH: rms(xhat), meanX: mean(x), s, groups };
      }
      H = V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const narrow = w < 600;
        const R = compute();
        const sIn = sci(R.rmsX, 2), sOut = nf(R.rmsH, 3);
        if (sIn !== lastIn) { lastIn = sIn; rIn.set(sIn); }
        if (sOut !== lastOut) { lastOut = sOut; rOut.set(sOut); }
        const meterW = narrow ? 0 : 170;
        const X0 = 16, X1 = w - 16 - (meterW ? meterW + 18 : 0);
        const rowsTop = 14, meterH = narrow ? 130 : 0;
        const rowH = (h - rowsTop - 16 - meterH - (narrow ? 12 : 0)) / 3;
        const el = t - S.t0, e1 = V.ease.inOut(V.clamp(el / 0.9, 0, 1)), e2 = V.ease.inOut(V.clamp((el - 0.7) / 0.9, 0, 1));
        const bw = (X1 - X0 - 50) / D, bx = (i) => X0 + 44 + i * bw;
        const drawRow = (k, vals, sc, scLab, title, col, prog) => {
          const top = rowsTop + k * rowH, mid = top + rowH / 2 + 8, half = rowH / 2 - 22;
          T(g, title, X0, top + 14, { size: 12, weight: 600, color: C.ink, font: 'mono', maxW: X1 - X0 - 70 });
          T(g, 'scale ±' + scLab, X1, top + 14, { size: 11, color: C.muted, align: 'right', font: 'mono' });
          V.line(g, X0 + 40, mid, X1, mid, { color: V.alpha(C.muted, 0.4), lw: 1 });
          T(g, '0', X0 + 34, mid + 4, { size: 11, font: 'mono', align: 'right', color: C.muted });
          T(g, '+' + scLab, X0 + 34, mid - half + 4, { size: 11, font: 'mono', align: 'right', color: C.muted, maxW: 36 });
          T(g, '−' + scLab, X0 + 34, mid + half + 4, { size: 11, font: 'mono', align: 'right', color: C.muted, maxW: 36 });
          if (R.groups > 1) for (let gi = 1; gi < R.groups; gi++) V.line(g, bx(gi * 4) - 1, top + 20, bx(gi * 4) - 1, top + rowH - 2, { color: V.alpha(C.d, 0.55), lw: 1.2, dash: [3, 3] });
          vals.forEach((v, i) => {
            const vv = v * prog;
            const hh = V.clamp(vv / sc, -1.08, 1.08) * half;
            const over = Math.abs(vv / sc) > 1.08;
            g.fillStyle = V.alpha(vv >= 0 ? col : C.c, 0.8);
            V.rr(g, bx(i) + bw * 0.14, hh >= 0 ? mid - hh : mid, bw * 0.72, Math.max(1, Math.abs(hh)), 3); g.fill();
            if (over) T(g, vv > 0 ? '▲' : '▼', bx(i) + bw / 2, vv > 0 ? mid - half - 6 : mid + half + 14, { size: 11, align: 'center', color: C.red });
          });
          return { mid, half };
        };
        // escala de exibição da entrada: potência de 10 conveniente
        const mxIn = Math.max(...R.x.map(Math.abs)) || 1;
        const k10 = Math.floor(Math.log10(mxIn));
        let scIn = Math.pow(10, k10 + 1);
        for (const mlt of [1, 2, 5, 10]) { const c = mlt * Math.pow(10, k10); if (c >= mxIn * 1.0001) { scIn = c; break; } }
        const scTxt = scIn >= 1 && Math.abs(scIn - Math.round(scIn)) < 1e-9 ? ni(scIn) : sci(scIn, 0);
        const row0 = drawRow(0, R.x, scIn, scTxt, 'input x', C.a, 1);
        const mid0 = row0.mid, half0 = row0.half;
        // média e RMS na entrada
        if (S.type === 'ln') { const my = mid0 - (R.meanX / scIn) * half0; V.line(g, X0 + 40, my, X1, my, { color: C.d, lw: 1.5, dash: [5, 4] }); T(g, 'μ', X1 - 4, my - 4, { size: 11, color: C.d, align: 'right' }); }
        else { const ry = (R.rmsX / scIn) * half0; V.line(g, X0 + 40, mid0 - ry, X1, mid0 - ry, { color: V.alpha(C.b, 0.9), lw: 1.3, dash: [4, 4] }); V.line(g, X0 + 40, mid0 + ry, X1, mid0 + ry, { color: V.alpha(C.b, 0.9), lw: 1.3, dash: [4, 4] }); T(g, '±RMS', X1 - 4, mid0 - ry - 4, { size: 11, color: C.b, align: 'right' }); }
        const t2 = S.type === 'ln' ? 'x̂ = (x − μ) / √(σ² + ε)' : S.type === 'grp' ? 'x̂: each group ÷ its own RMS' : 'x̂ = x / √(mean(x²) + ε)';
        drawRow(1, R.xhat, 3, '3', t2, C.b, e1);
        const t3 = S.type === 'ln' ? 'output = γ ⊙ x̂ + β' : S.type === 'bug' ? 'output = w ⊙ x̂   (forgot the 1 +)' : S.type === 'rms' ? 'output = γ ⊙ x̂   (stored γ ≈ 1)' : 'output = (1 + w) ⊙ x̂   (stored w ≈ 0)';
        drawRow(2, R.out, 3, '3', t3, S.type === 'bug' ? C.red : C.e, e2);
        // varredura: uma faixa percorre as 16 dimensões (a redução que calcula média e RMS)
        const sw = (t * 0.25) % 1.3;
        if (sw < 1) { const sx = bx(0) + sw * (bx(D) - bx(0)); const grd = g.createLinearGradient(sx - 30, 0, sx + 30, 0); grd.addColorStop(0, V.alpha(C.d, 0)); grd.addColorStop(0.5, V.alpha(C.d, 0.12)); grd.addColorStop(1, V.alpha(C.d, 0)); g.fillStyle = grd; g.fillRect(sx - 30, rowsTop + 20, 60, rowH - 22); }
        // medidor ε vs RMS²
        let mx, my0, mw, mh;
        if (!narrow) { mx = w - meterW - 14; my0 = 18; mw = meterW; mh = h - 36; }
        else { mx = 16; my0 = h - meterH - 8; mw = w - 32; mh = meterH; }
        V.box(g, mx, my0, mw, mh, { r: 10, fill: V.alpha(C.panel, 0.85), stroke: V.alpha(C.muted, 0.3), lw: 1 });
        T(g, 'ε vs RMS² (log scale)', mx + 10, my0 + 17, { size: 11.5, weight: 600, color: C.ink });
        const lo = -22, hi = 7;
        if (!narrow) {
          const ax = mx + 46, top = my0 + 34, bot = my0 + mh - 50;
          const toY = (lg) => bot - ((lg - lo) / (hi - lo)) * (bot - top);
          V.line(g, ax, top, ax, bot, { color: V.alpha(C.muted, 0.5), lw: 1.2 });
          for (let lg = -20; lg <= 5; lg += 5) { V.line(g, ax - 4, toY(lg), ax + 4, toY(lg), { color: C.muted, lw: 1 }); T(g, pow10(lg), ax - 7, toY(lg) + 4, { size: 11, font: 'mono', align: 'right', color: C.muted }); }
          const ly = toY(Math.log10(S.eps));
          g.fillStyle = V.alpha(C.d, 0.12); g.fillRect(ax + 1, ly, mw - 58, bot - ly);
          V.line(g, ax, ly, mx + mw - 10, ly, { color: C.d, lw: 2 });
          T(g, 'ε', mx + mw - 12, ly - 5, { size: 12, weight: 700, color: C.d, align: 'right' });
          T(g, 'ε dominates', mx + mw - 12, bot - 6, { size: 11, color: C.d, align: 'right' });
          R.gstat.forEach((gs, gi) => { const v = Math.log10(Math.max(1e-30, gs.ms)); const yy = toY(V.clamp(v, lo, hi)); const xx = ax + 12 + gi * 16; V.dot(g, xx, yy, 5, C.a); if (gi === 0) T(g, 'RMS²', xx + 10 + (R.groups - 1) * 16, yy + 4, { size: 11, color: C.a }); });
          const dom = R.gstat[0].ms < S.eps * 3;
          wrap(g, dom ? 'RMS² ≲ ε: the output stays below RMS 1.' : 'RMS² ≫ ε: the output is scale invariant.', mx + 10, my0 + mh - 32, mw - 20, 14, { size: 11, color: dom ? C.d : C.a });
        } else {
          const ay = my0 + 62, l0 = mx + 14, l1 = mx + mw - 14;
          const toX = (lg) => l0 + ((lg - lo) / (hi - lo)) * (l1 - l0);
          V.line(g, l0, ay, l1, ay, { color: V.alpha(C.muted, 0.5), lw: 1.2 });
          for (let lg = -20; lg <= 5; lg += 5) { V.line(g, toX(lg), ay - 4, toX(lg), ay + 4, { color: C.muted, lw: 1 }); T(g, pow10(lg), toX(lg), ay + 17, { size: 11, font: 'mono', align: 'center', color: C.muted }); }
          const lx = toX(Math.log10(S.eps));
          g.fillStyle = V.alpha(C.d, 0.14); g.fillRect(l0, ay - 20, lx - l0, 20);
          V.line(g, lx, ay - 26, lx, ay + 6, { color: C.d, lw: 2 });
          T(g, 'ε', lx + 4, ay - 28, { size: 12, weight: 700, color: C.d });
          R.gstat.forEach((gs, gi) => { const v = Math.log10(Math.max(1e-30, gs.ms)); V.dot(g, toX(V.clamp(v, lo, hi)), ay - 8 - gi * 5, 4.5, C.a); });
          T(g, '● input RMS²', l0, ay - 30, { size: 11, color: C.a });
          const dom = R.gstat[0].ms < S.eps * 3;
          T(g, dom ? 'RMS² ≲ ε: the output stays below RMS 1.' : 'RMS² ≫ ε: the output is scale invariant.', l0, my0 + mh - 14, { size: 11.5, color: dom ? C.d : C.a, maxW: mw - 28 });
        }
      });
    });

    /* ====================================================================
       Figura 9 — Telefone sem fio vs estrada expressa
       ==================================================================== */
    safe('fig-residual', () => {
      const fig = $('#fig-residual');
      fitH(fig, (w) => (w < 600 ? 580 : 440));
      const d = 24, LM = 64;
      const r = V.rng(21);
      const Ws = []; for (let l = 0; l < LM; l++) { const W = new Float64Array(d * d); for (let k = 0; k < d * d; k++) W[k] = r.normal() / Math.sqrt(d); Ws.push(W); }
      const h0 = Array.from({ length: d }, () => r.normal());
      const gL = Array.from({ length: d }, () => r.normal());
      const AFR = { relu: ACT.relu, tanh: ACT.tanh, sigmoid: ACT.sigmoid };
      const S = { arch: 'plain', act: 'relu', L: 32, gain: 1 };
      let R = null;
      function sim() {
        const f = AFR[S.act].f, df = AFR[S.act].d, L = S.L, gain = S.gain;
        const hs = [h0.slice()], zs = [], ns = [], rs = [];
        let h = h0.slice();
        for (let l = 0; l < L; l++) {
          const W = Ws[l]; let inp = h, n = null, rr = 1;
          if (S.arch === 'pre') { let ms = 0; for (const v of h) ms += v * v; ms /= d; rr = Math.sqrt(ms + 1e-6); n = h.map((v) => v / rr); inp = n; }
          const z = new Array(d); for (let o = 0; o < d; o++) { let s = 0; for (let i = 0; i < d; i++) s += W[o * d + i] * inp[i]; z[o] = gain * s; }
          const hn = S.arch === 'plain' ? z.map(f) : h.map((v, i) => v + f(z[i]));
          zs.push(z); ns.push(n); rs.push(rr); h = hn; hs.push(h);
        }
        let g = gL.slice(); const gn = new Array(L + 1); gn[L] = Math.hypot(...g);
        for (let l = L - 1; l >= 0; l--) {
          const W = Ws[l], z = zs[l];
          const gz = g.map((v, i) => v * df(z[i]));
          const wt = new Array(d); for (let i = 0; i < d; i++) { let s = 0; for (let o = 0; o < d; o++) s += W[o * d + i] * gz[o]; wt[i] = gain * s; }
          let back;
          if (S.arch === 'plain') back = wt;
          else if (S.arch === 'res') back = g.map((v, i) => v + wt[i]);
          else { const n = ns[l], rr = rs[l]; let dot = 0; for (let i = 0; i < d; i++) dot += n[i] * wt[i]; back = g.map((v, i) => v + (wt[i] - (n[i] * dot) / d) / rr); }
          g = back; gn[l] = Math.hypot(...g);
        }
        const rms = hs.map((hh) => Math.sqrt(hh.reduce((q, v) => q + v * v, 0) / d));
        R = { gn: gn.map((v) => v / gn[L]), rms, L };
        rG.set('× ' + sci(R.gn[0], 1));
        rA.set(sci(R.rms[L], 1));
      }
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Architecture', options: [{ v: 'plain', label: 'plain' }, { v: 'res', label: 'residual' }, { v: 'pre', label: 'residual + pre-norm' }], value: S.arch, onChange: (v) => { S.arch = v; sim(); } });
      UI.seg(ctl, { label: 'Activation', options: [{ v: 'relu', label: 'ReLU' }, { v: 'tanh', label: 'tanh' }, { v: 'sigmoid', label: 'sigmoid' }], value: S.act, onChange: (v) => { S.act = v; sim(); } });
      UI.slider(ctl, { label: 'depth', min: 4, max: 64, step: 1, value: S.L, fmt: (v) => String(v), onInput: (v) => { S.L = v; sim(); } });
      UI.slider(ctl, { label: 'weight gain', min: 0.5, max: 2, step: 0.05, value: S.gain, fmt: (v) => nf(v, 2), onInput: (v) => { S.gain = v; sim(); } });
      const rG = UI.readout(ctl, 'gradient at layer 0', '—');
      const rA = UI.readout(ctl, 'output RMS', '—');
      sim();
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const narrow = w < 600;
        const L = R.L;
        const X0 = narrow ? 44 : 56, X1 = w - 18;
        const bandY = 34, bandH = narrow ? 34 : 40;
        const toXl = (l) => X0 + (l / L) * (X1 - X0);
        // estrada expressa
        if (S.arch !== 'plain') {
          V.line(g, X0, bandY - 12, X1, bandY - 12, { color: V.alpha(C.d, 0.8), lw: 3, dash: [10, 6], dashOffset: t * 30 });
          T(g, S.arch === 'pre' ? 'express lane (identity) + each layer reads Norm(h)' : 'express lane: h + F(h)', X0, bandY - 20, { size: 11, color: C.d });
        } else T(g, 'telephone game: each layer replaces the vector', X0, bandY - 20, { size: 11, color: C.muted });
        // fila de camadas: brilho = log do gradiente
        const lg = R.gn.map((v) => Math.log10(Math.max(1e-300, v)));
        const cw = (X1 - X0) / (L + 1);
        const pulse = 1 - ((t * 0.3) % 1);
        const pl = pulse * L;
        for (let l = 0; l <= L; l++) {
          const k = V.clamp((lg[l] + 6) / 12, 0, 1);
          const X = X0 + l * cw;
          V.box(g, X + 1, bandY, Math.max(2, cw - 2), bandH, { r: Math.min(5, cw / 3), fill: mixA(C.panel, C.c, k, 0.25 + 0.7 * k), stroke: V.alpha(C.c, 0.3), lw: 0.8 });
        }
        const pli = Math.round(pl), pk = V.clamp((lg[Math.min(L, Math.max(0, pli))] + 6) / 12, 0, 1);
        V.glow(g, X0 + pl * cw + cw / 2, bandY + bandH / 2, 10 + 18 * pk, C.c, 0.4 + 0.6 * pk);
        T(g, 'input', X0, bandY + bandH + 14, { size: 11, color: C.muted });
        T(g, 'loss ←', X1, bandY + bandH + 14, { size: 11, color: C.muted, align: 'right' });
        T(g, 'brightness = incoming gradient', (X0 + X1) / 2, bandY + bandH + 14, { size: 11, color: C.c, align: 'center' });
        // gráfico log
        const gy0 = bandY + bandH + 34, gy1 = h - 34;
        const lo = -10, hi = 10;
        const toY = (v) => gy1 - ((V.clamp(v, lo, hi) - lo) / (hi - lo)) * (gy1 - gy0);
        for (let v = lo; v <= hi; v += 5) { V.line(g, X0, toY(v), X1, toY(v), { color: V.alpha(C.muted, v === 0 ? 0.45 : 0.14), lw: 1 }); T(g, pow10(v), X0 - 6, toY(v) + 4, { size: 11, font: 'mono', align: 'right', color: C.muted }); }
        const plot = (arr, col) => { g.beginPath(); arr.forEach((v, l) => { const X = toXl(l), Y = toY(Math.log10(Math.max(1e-300, v))); if (l) g.lineTo(X, Y); else g.moveTo(X, Y); }); g.strokeStyle = col; g.lineWidth = 2.6; g.stroke(); };
        plot(R.rms, C.a);
        plot(R.gn, C.c);
        for (let l = 0; l <= L; l += Math.max(1, Math.round(L / 8))) T(g, String(l), toXl(l), gy1 + 15, { size: 11, font: 'mono', align: 'center', color: C.muted });
        T(g, 'layer', X1, gy1 + 28, { size: 11, color: C.muted, align: 'right' });
        V.dot(g, X0 + 8, gy0 + 6, 4, C.a); T(g, 'activation RMS', X0 + 16, gy0 + 10, { size: 11.5, color: C.a });
        V.dot(g, X0 + 8, gy0 + 24, 4, C.c); T(g, 'gradient norm (÷ at the loss)', X0 + 16, gy0 + 28, { size: 11.5, color: C.c });
        if (lg[0] < lo || lg[0] > hi) T(g, lg[0] < lo ? '↓ below ' + pow10(lo) : '↑ above ' + pow10(hi), toXl(0) + 6, lg[0] < lo ? gy1 - 6 : gy0 + 46, { size: 11, color: C.c });
      });
    });

    /* ====================================================================
       Figura 10 — O caminho de um token pelo modelo
       ==================================================================== */
    safe('fig-mapa', () => {
      const fig = $('#fig-mapa');
      const MAP = {
        glm: {
          name: 'GLM-5.3-Flash', L: 45, sample: [0, 1, 2, 3, 44],
          typeOf: (l) => (l % 4 === 3 ? 'mla' : 'kda'),
          tname: { kda: 'KDA', mla: 'MLA + DSA' }, tstrong: { kda: 0.28, mla: 1 },
          extra: (l) => (l < 3 ? ' · dense MLP' : ' · MoE'),
          pre: [{ n: 'Input', s: '4096 embedding → hc_expand: 4 streams (16,384 numbers)', c: [] }],
          layer: [
            { n: 'mHC read (pre)', s: '4 streams → 1 vector of 4096', c: [['g', 'pre = σ(·) + ε'], ['n', 'input_layernorm · RMSNorm ε 1e-5']] },
            { fork: 'token mixer: KDA or MLA + DSA, depending on the layer', pick: (l) => (l % 4 === 3 ? 1 : 0), br: [
              { n: 'KDA · 34 layers', s: 'linear attention', c: [['a', 'conv k=4 + SiLU on q, k, v'], ['n', 'L2 on q and k'], ['g', 'β = σ(·)'], ['g', 'decay −5·σ(·)'], ['n', 'o_norm RMSNorm'], ['g', 'output ⊙ σ(g_b·g_a·x)']] },
              { n: 'MLA + DSA · 11 layers', s: 'sparse attention · layers 3, 7, …, 43', c: [['n', 'q_a_layernorm [1536]'], ['n', 'kv_a_layernorm [512]'], ['n', 'k_norm: LayerNorm + bias'], ['a', 'score Σ w·ReLU(q·k)']] },
            ] },
            { n: 'mHC write → read', s: 'mixer post fused with the FFN pre', c: [['n', '4×4 Sinkhorn mixing'], ['g', 'post = 2·σ(·)'], ['g', 'pre = σ(·) + ε'], ['n', 'post_attention_layernorm']] },
            { fork: 'FFN: dense or MoE, depending on the layer', pick: (l) => (l < 3 ? 0 : 1), br: [
              { n: 'dense MLP · layers 0–2', s: 'SwiGLU 4096 → 12288', c: [['a', 'SwiGLU (SiLU)'], ['a', 'clamp gate ≤ 10, up ±10']] },
              { n: 'MoE · layers 3–44', s: '288 experts, top-8, + 1 shared', c: [['g', 'router σ(W·x)'], ['a', 'SwiGLU experts (SiLU)'], ['a', 'clamp gate ≤ 10, up ±10']] },
            ] },
            { n: 'mHC write (post)', s: 'fused with the next layer\'s read', c: [['n', '4×4 Sinkhorn mixing'], ['g', 'post = 2·σ(·)']] },
          ],
          post: [{ n: 'Output', s: 'after layer 44', c: [['n', 'hc_contract (mean of the 4) → model.norm']] }],
          side: [
            { n: 'Vision', s: '24-block ViT + merger', c: [['a', 'SwiGLU MLP with bias'], ['n', 'RMSNorm, ε forced to 1e-6'], ['a', 'SwiGLU merger, clamp 10']] },
            { n: 'Drafters', s: 'MTP (layer 45) and DFlash2 (5 layers)', c: [['n', 'MTP: enorm, hnorm [4096]'], ['n', 'DFlash2: RMSNorm ε 1e-5, k_norm']] },
          ],
        },
        qwen: {
          name: 'Qwen3.8-Flash-Next', L: 48, sample: [0, 1, 2, 3, 47],
          typeOf: (l) => ((l + 1) % 4 === 0 ? 'qsa' : 'gdn'),
          tname: { gdn: 'GDN', qsa: 'QSA' }, tstrong: { gdn: 0.28, qsa: 1 },
          extra: (l) => (l === 1 ? ' · with PLE' : ''),
          pre: [{ n: 'Input', s: '2560 embedding, copied into the 4 GR branches (10,240 numbers)', c: [] }],
          layer: [
            { n: 'PLE · N-gram embedding', s: 'only in the layer with index 1 (the 2nd), before the GR read', when: (l) => l === 1, c: [['n', 'grouped RMSNorm ε 1e-6'], ['g', 'g = σ(sign(z)·√|z|)'], ['a', 'dilated conv + SiLU']] },
            { n: 'GR read', s: '4 branches → 1 vector of 2560', c: [['n', 'per-branch RMSNorm, gain 1+w'], ['g', 'G = σ(W_up·SiLU(W_down·))']] },
            { fork: 'token mixer: GDN or QSA, depending on the layer', pick: (l) => ((l + 1) % 4 === 0 ? 1 : 0), br: [
              { n: 'GDN · 36 layers', s: 'linear attention', c: [['a', 'conv k=4 + SiLU'], ['n', 'L2 on q and k'], ['g', 'β = σ(W_b·x)'], ['a', 'α via softplus'], ['n', 'RMSNorm(y) [128]'], ['g', 'output σ(W_z·x) ⊙']] },
              { n: 'QSA · 12 layers', s: 'sparse attention · layers 3, 7, …, 47', c: [['n', 'QK-Norm [256]'], ['a', 'indexer Σ ReLU(q·k̄)'], ['g', 'per-head σ gate (half of q_proj)']] },
            ] },
            { n: 'GR write → read', s: 'adds the mixer output and reads again for the MoE', c: [['g', 's = 2·σ(·), 1 scalar per branch'], ['n', 'per-branch RMSNorm, 1+w'], ['g', 'G = σ(W_up·SiLU(W_down·))']] },
            { n: 'MoE · 48 layers', s: '512 experts, top-10, + shared', c: [['a', 'SwiGLU experts (SiLU)'], ['g', 'shared expert × σ(g·x)']] },
            { n: 'GR write', s: 'R ← R + s ⊗ y', c: [['g', 's = 2·σ(·), 1 scalar per branch']] },
          ],
          post: [{ n: 'Output', s: 'no model.norm: the final read plays the role of the norm', c: [['n', 'hyper_connection_mixer (final read)']] }],
          side: [
            { n: 'Vision', s: '27-block ViT + merger; its vectors take the place of the image tokens', c: [['a', 'ViT MLP: GELU-tanh'], ['n', 'LayerNorm with bias'], ['a', 'merger: GELU']] },
            { n: 'MTP (drafter)', s: '1 QSA + MoE layer', c: [['n', 'pre_fc_norm_embedding [2560]'], ['n', 'pre_fc_norm_hidden [10240]']] },
          ],
        },
        ds: {
          name: 'DeepSeek-V4-Flash', L: 43, sample: [0, 1, 2, 3, 42],
          typeOf: (l) => (l < 2 ? 'swa' : l % 2 === 0 ? 'csa' : 'hca'),
          tname: { swa: 'SWA', csa: 'CSA', hca: 'HCA' }, tstrong: { swa: 0.15, csa: 1, hca: 0.45 },
          extra: (l) => (l < 3 ? ' · hash MoE' : ''),
          pre: [{ n: 'Input', s: '4096 embedding → 4 mHC streams (16,384 numbers)', c: [] }],
          layer: [
            { n: 'mHC read (hc_pre)', s: '4 streams → 1 vector of 4096', c: [['g', 'pre = σ(·) + ε'], ['n', 'attn_norm · RMSNorm ε 1e-20']] },
            { fork: 'attention: SWA, CSA or HCA, depending on the layer', pick: (l) => (l < 2 ? 0 : l % 2 === 0 ? 1 : 2), br: [
              { n: 'SWA · layers 0 and 1', s: 'only the window of the last 128 tokens', c: [['n', 'q_norm [1024] · kv_norm [512]'], ['n', 'query ÷ per-head RMS (no weight)']] },
              { n: 'CSA · 21 layers', s: 'layers 2, 4, …, 42 · KV 4:1 + indexer (top-512) + window', c: [['n', 'same norms as SWA'], ['a', 'indexer Σ w·ReLU(q·k)']] },
              { n: 'HCA · 20 layers', s: 'layers 3, 5, …, 41 · KV 128:1, reads every entry + window', c: [['n', 'same norms as SWA'], ['x', 'no indexer']] },
            ] },
            { n: 'mHC write → read', s: 'mixer hc_post fused with the FFN hc_pre', c: [['n', '4×4 Sinkhorn mixing'], ['g', 'post = 2·σ(·)'], ['g', 'pre = σ(·) + ε'], ['n', 'ffn_norm · RMSNorm']] },
            { n: 'MoE · 43 layers', s: '256 experts, top-6, + shared', c: [['a', 'router √softplus(·) (layers 0–2: hash)'], ['a', 'SwiGLU experts (SiLU)'], ['a', 'clamp gate ≤ 10, up ±10']] },
            { n: 'mHC write (hc_post)', s: 'fused with the next layer\'s read', c: [['n', '4×4 Sinkhorn mixing'], ['g', 'post = 2·σ(·)']] },
          ],
          post: [{ n: 'Output', s: 'after layer 42', c: [['g', 'hc_head: σ(·) + ε weights'], ['n', 'final norm (RMSNorm)']] }],
          side: [
            { n: 'Vision (Vision-Exp)', s: 'ViT + Aligner', c: [['n', 'ViT: RMSNorm ε 1e-6'], ['a', 'ViT: SwiGLU MLP'], ['a', 'Aligner: exact GELU']] },
            { n: 'DSpark (drafter)', s: '3 stages; reads layers 40–42', c: [['n', 'q_norm, kv_norm; Q ÷ RMS, no weight'], ['g', 'confidence σ(·)']] },
          ],
        },
      };
      const S = { model: 'qwen', filt: 'all', t0: 0 };
      let H = null;
      const layoutCache = {};
      const lineX = 34, retX = 17, x0 = 56;
      function layout(w, model) {
        const key = model + ':' + Math.round(w);
        if (layoutCache[key]) return layoutCache[key];
        const D = MAP[model], g = measureCtx;
        const wide = w >= 640;
        const LW = wide ? 230 : 0;
        const cx0 = wide ? x0 + LW : x0, chipsW = w - cx0 - 16;
        const rows = [], secs = [], forks = [], layerRows = [];
        let y = 44;
        const addRow = (it, sec, extra) => {
          const subLines = splitLines(g, it.s, (wide ? LW - 18 : w - x0 - 16), { size: 11.5 }).slice(0, 2);
          const labH = 17 + subLines.length * 15;
          let cx = cx0, cy = wide ? y : y + labH + 4;
          const chH = 24, gap = 6, chips = [];
          it.c.forEach(([k, txt]) => {
            const cw = Math.min(textW(g, txt, 12, 'sans', 500) + 20, chipsW);
            if (cx + cw > cx0 + chipsW && cx > cx0) { cx = cx0; cy += chH + gap; }
            chips.push({ k, txt, x: cx, y: cy, w: cw, h: chH });
            cx += cw + gap;
          });
          const bottom = Math.max(y + labH, it.c.length ? cy + chH : y + labH);
          const r = Object.assign({ it, sec, y0: y, dotY: y + 8, subLines, chips }, extra || {});
          y = bottom + 12;
          rows.push(r);
          return r;
        };
        const header = (txt, sec) => { secs.push({ txt, y: y + 9, sec }); y += 22; };
        header('BEFORE THE STACK', 'pre');
        const preRows = D.pre.map((it) => addRow(it, 'pre'));
        y += 6;
        const loopTop = y; y += 62;
        D.layer.forEach((it, li) => {
          if (it.fork) {
            const f = { top: y, label: it.fork, rows: [], li };
            y += 22;
            it.br.forEach((b, bi) => { if (bi) y += 8; f.rows.push(addRow(b, 'loop', { li, bi })); });
            f.bot = y - 8; y += 6;
            forks.push(f); layerRows.push({ fork: f, it });
          } else layerRows.push({ row: addRow(it, 'loop', { li, cond: !!it.when }), it });
        });
        const loopBot = y - 4; y += 14;
        header('AFTER THE STACK', 'post');
        const postRows = D.post.map((it) => addRow(it, 'post'));
        y += 6;
        header('SEPARATE TOWERS · off the text path', 'side');
        D.side.forEach((it) => addRow(it, 'side'));
        // linha do tempo do token
        const ev = [];
        const stay = (r, dur, l) => ev.push({ type: 'stay', r, y: r.dotY, dur, l });
        const move = (pts, dur, l) => ev.push({ type: 'move', pts, dur, l });
        preRows.forEach((r) => stay(r, 1.0, null));
        let lastY = preRows[preRows.length - 1].dotY;
        const loopEntry = loopTop + 58, loopExit = loopBot - 6;
        D.sample.forEach((l, si) => {
          const stops = [];
          layerRows.forEach((lr) => {
            if (lr.fork) stops.push(lr.fork.rows[lr.it.pick(l)]);
            else if (!lr.it.when || lr.it.when(l)) stops.push(lr.row);
          });
          if (si > 0) move([[lineX, lastY], [lineX, loopExit], [retX, loopExit - 12], [retX, loopEntry + 12], [lineX, loopEntry], [lineX, stops[0].dotY]], 1.3, l);
          else move([[lineX, lastY], [lineX, stops[0].dotY]], 0.45, l);
          stops.forEach((r, k) => { if (k) move([[lineX, stops[k - 1].dotY], [lineX, r.dotY]], 0.4, l); stay(r, 0.85, l); });
          lastY = stops[stops.length - 1].dotY;
        });
        postRows.forEach((r) => { move([[lineX, lastY], [lineX, r.dotY]], 0.6, null); stay(r, 1.4, null); lastY = r.dotY; });
        ev.push({ type: 'stay', r: null, y: lastY, dur: 0.8, l: null, fade: true });
        let tot = 0; ev.forEach((e) => { e.t0 = tot; tot += e.dur; });
        return (layoutCache[key] = { rows, secs, forks, layerRows, loopTop, loopBot, loopEntry, loopExit, H: y + 4, ev, tot, wide, firstY: preRows[0].dotY, lastY: postRows[postRows.length - 1].dotY });
      }
      const along = (pts, u) => {
        const seg = []; let L = 0;
        for (let i = 1; i < pts.length; i++) { const d = Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]); seg.push(d); L += d; }
        let target = u * L;
        for (let i = 0; i < seg.length; i++) { if (target <= seg[i] || i === seg.length - 1) { const k = seg[i] ? V.clamp(target / seg[i], 0, 1) : 1; return [V.lerp(pts[i][0], pts[i + 1][0], k), V.lerp(pts[i][1], pts[i + 1][1], k)]; } target -= seg[i]; }
        return pts[pts.length - 1];
      };
      const applyH = fitH(fig, (w) => layout(w, S.model).H);
      if (document.fonts && document.fonts.ready) document.fonts.ready.then(() => { Object.keys(layoutCache).forEach((k) => delete layoutCache[k]); applyH(); });
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Model', options: [{ v: 'glm', label: 'GLM' }, { v: 'qwen', label: 'Qwen' }, { v: 'ds', label: 'DeepSeek' }], value: S.model, onChange: (v) => { S.model = v; if (H) S.t0 = H.t; applyH(); } });
      UI.seg(ctl, { label: 'Highlight', options: [{ v: 'all', label: 'all' }, { v: 'a', label: 'activations' }, { v: 'g', label: 'gates' }, { v: 'n', label: 'norms' }], value: S.filt, onChange: (v) => { S.filt = v; } });
      const C0 = V.C();
      UI.legend(ctl, [[C0.a, 'activation'], [C0.d, 'gate'], [C0.b, 'norm']]);
      H = V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const D = MAP[S.model], Lo = layout(w, S.model);
        const cat = { a: C.a, g: C.d, n: C.b, x: C.muted };
        const mcol = C[S.model];
        // estado da animação
        const tt = (t - S.t0) % Lo.tot;
        let e = Lo.ev[0];
        for (const ev of Lo.ev) { if (tt >= ev.t0) e = ev; else break; }
        const u = V.clamp((tt - e.t0) / e.dur, 0, 1);
        const curL = e.l;
        const tokP = e.type === 'stay' ? [lineX, e.y] : along(e.pts, V.ease.inOut(u));
        const hereRow = e.type === 'stay' ? e.r : null;
        T(g, D.name + (w < 520 ? '' : ' · a token\'s path through the model'), 16, 22, { size: 13, weight: 700, color: mcol, maxW: w - 32 });
        // caixa do laço (dentro de cada camada)
        V.box(g, 6, Lo.loopTop, w - 12, Lo.loopBot - Lo.loopTop, { r: 14, fill: V.alpha(mcol, 0.05), stroke: V.alpha(mcol, 0.45), lw: 1.3 });
        const narrow = w < 520;
        T(g, (narrow ? 'LOOP' : 'INSIDE EACH LAYER') + ' · ℓ = 0…' + (D.L - 1) + ' (×' + D.L + ')', 28, Lo.loopTop + 17, { size: 11, font: 'mono', weight: 600, color: mcol });
        if (curL != null) {
          const jumped = D.sample.indexOf(curL) > 0 && curL - D.sample[D.sample.indexOf(curL) - 1] > 1;
          T(g, (jumped ? '… ' : '') + 'ℓ = ' + curL + ' · ' + D.tname[D.typeOf(curL)] + D.extra(curL), w - 18, Lo.loopTop + 17, { size: 12, font: 'mono', weight: 700, color: C.ink, align: 'right', maxW: w * 0.45 });
        }
        // faixa de camadas
        const sx0 = 28, sx1 = w - 18, sy = Lo.loopTop + 26, cwS = (sx1 - sx0) / D.L;
        for (let l = 0; l < D.L; l++) {
          const ty = D.typeOf(l), a = D.tstrong[ty];
          g.fillStyle = V.alpha(mcol, 0.12 + 0.78 * a);
          g.fillRect(sx0 + l * cwS + 0.5, sy, Math.max(1, cwS - 1.2), 11);
        }
        if (curL != null) { g.strokeStyle = C.ink; g.lineWidth = 2; g.strokeRect(sx0 + curL * cwS - 0.5, sy - 2, cwS + 0.6, 15); }
        let lx = sx0;
        Object.keys(D.tname).forEach((ty) => {
          g.fillStyle = V.alpha(mcol, 0.12 + 0.78 * D.tstrong[ty]); g.fillRect(lx, sy + 19, 10, 10);
          const lab = D.tname[ty]; T(g, lab, lx + 14, sy + 28, { size: 11, color: C.muted }); lx += 14 + textW(g, lab, 11, 'sans', 400) + 14;
        });
        // trilho de retorno (↺ próxima camada)
        g.save(); g.setLineDash([5, 5]); g.lineDashOffset = t * 18;
        g.strokeStyle = V.alpha(mcol, 0.5); g.lineWidth = 2;
        g.beginPath(); g.moveTo(lineX, Lo.loopExit); g.quadraticCurveTo(retX, Lo.loopExit, retX, Lo.loopExit - 14); g.lineTo(retX, Lo.loopEntry + 14); g.quadraticCurveTo(retX, Lo.loopEntry, lineX, Lo.loopEntry); g.stroke();
        g.restore();
        V.arrow(g, retX, (Lo.loopEntry + Lo.loopExit) / 2 + 8, retX, (Lo.loopEntry + Lo.loopExit) / 2 - 8, { color: V.alpha(mcol, 0.8), lw: 2, head: 7 });
        // linha principal
        g.strokeStyle = V.alpha(mcol, 0.55); g.lineWidth = 5; g.lineCap = 'round';
        g.beginPath(); g.moveTo(lineX, Lo.firstY); g.lineTo(lineX, Lo.lastY); g.stroke(); g.lineCap = 'butt';
        // bifurcações
        Lo.forks.forEach((f) => {
          V.box(g, lineX - 18, f.top, w - 12 - (lineX - 18) - 4, f.bot - f.top, { r: 10, fill: V.alpha(C.panel, 0.55), stroke: V.alpha(C.muted, 0.35), lw: 1 });
          T(g, f.label, x0, f.top + 14, { size: 11, font: 'mono', color: C.muted, maxW: w - x0 - 24 });
          for (let k = 1; k < f.rows.length; k++) {
            const yo = (f.rows[k - 1].dotY + f.rows[k].dotY) / 2 + 4;
            pill(g, 'or', lineX, yo, { size: 11, font: 'sans', weight: 700, fill: C.bg, stroke: V.alpha(C.muted, 0.6), color: C.muted, align: 'center', padX: 5 });
          }
        });
        // seções
        Lo.secs.forEach((sc) => T(g, sc.txt, sc.sec === 'pre' ? 28 : x0, sc.y, { size: 11, font: 'mono', weight: 600, color: C.muted }));
        // estações
        Lo.rows.forEach((r) => {
          const it = r.it, side = r.sec === 'side';
          let dim = false;
          if (curL != null && r.sec === 'loop') {
            if (r.bi != null) dim = D.layer[r.li].pick(curL) !== r.bi;
            else if (r.cond) dim = !it.when(curL);
          }
          const here = r === hereRow;
          g.globalAlpha = dim ? 0.3 : 1;
          const rad = here ? 8 : 6.5;
          if (side || r.cond) g.setLineDash([3, 3]);
          V.dot(g, lineX, r.dotY, rad, C.bg);
          g.strokeStyle = side ? C.muted : mcol; g.lineWidth = here ? 3 : 2; g.beginPath(); g.arc(lineX, r.dotY, rad, 0, 7); g.stroke();
          g.setLineDash([]);
          T(g, it.n, x0, r.y0 + 13, { size: 13.5, weight: 700, color: here ? C.ink : V.alpha(C.ink, 0.88), maxW: Lo.wide ? 222 : w - x0 - 14 });
          r.subLines.forEach((sl, k) => T(g, sl, x0, r.y0 + 29 + k * 15, { size: 11.5, color: C.muted }));
          r.chips.forEach((cp) => {
            const on = S.filt === 'all' || S.filt === cp.k;
            const col = cat[cp.k], lit = on && here;
            g.globalAlpha = (dim ? 0.3 : 1) * (on ? 1 : 0.28);
            if (lit) V.glow(g, cp.x + cp.w / 2, cp.y + cp.h / 2, cp.w / 2 + 6, col, 0.25);
            V.box(g, cp.x, cp.y, cp.w, cp.h, { r: 12, fill: V.alpha(col, lit ? 0.3 : 0.14), stroke: V.alpha(col, lit ? 1 : 0.7), lw: lit ? 1.8 : 1.1 });
            T(g, cp.txt, cp.x + 10, cp.y + cp.h / 2 + 4.3, { size: 12, weight: 500, color: C.ink, maxW: cp.w - 16 });
          });
          g.globalAlpha = 1;
        });
        // o token
        const fa = e.fade ? 1 - u : 1;
        g.globalAlpha = fa;
        V.glow(g, tokP[0], tokP[1], 16, C.ink, 0.6);
        V.dot(g, tokP[0], tokP[1], 4.5, C.ink);
        g.globalAlpha = 1;
      });
    });
  },
});

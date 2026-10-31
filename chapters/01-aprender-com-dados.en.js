LA.chapter({
  id: 'aprender-com-dados',
  part: 1,
  order: 1,
  title: 'What is learning: functions, losses and gradients',
  navTitle: 'Learning from data',
  subtitle: 'From the ice cream shop line to the 321 billion knobs of GLM: parameters, losses, gradient, SGD, AdamW, Muon, generalization and the boundary that organizes everything, because at inference the weights freeze.',
  minutes: 50,
  recipes: ['glm', 'qwen', 'ds'],
  tags: ['machine learning', 'loss function', 'cross-entropy', 'gradient', 'SGD', 'AdamW', 'Muon', 'overfitting', 'QAT', 'training vs inference'],
  goals: [
    'See <b>learning</b> as fitting the parameters \\(\\theta\\) of a function \\(f_\\theta\\) to data, and the next token as a self-supervised answer key',
    'Compute and interpret <b>MSE, cross-entropy, NLL, nats and bits</b>, including each model\'s starting point \\(\\ln V\\)',
    'Understand <b>gradient descent, learning rate, SGD and minibatches</b>, and what momentum, <b>AdamW</b> and <b>Muon</b> (Newton–Schulz) change in the step',
    'Separate <b>training from inference</b>: what the checkpoint freezes, which config fields only matter in training and why DSpark, MTP and QAT impose parameters on the recipes',
  ],
  prereqs: [],
  summary: [
    'Learning = choosing the \\(\\theta\\) that minimizes an average <b>loss</b> over examples. In an LLM, the examples come from the text itself (predicting the next token), and the loss is the cross-entropy/NLL \\(-\\ln p_\\theta(x_t\\mid x_{&lt;t})\\), in nats.',
    'Cross-entropy punishes confident errors without limit, and its gradient with respect to the logits is \\(p - \\text{one-hot}\\). The uniform guess costs \\(\\ln V\\) ≈ 11.8–12.4 nats for the three models.',
    'Gradient descent: \\(\\theta \\leftarrow \\theta - \\eta\\nabla\\mathcal{L}\\). It is only stable if \\(\\eta &lt; 2/\\lambda_{\\max}\\). SGD swaps the exact gradient for a minibatch average, which is unbiased and has noise \\(\\propto 1/\\sqrt{B}\\).',
    'Momentum accumulates velocity; Adam/AdamW normalizes the step per coordinate (decoupled weight decay); Muon orthogonalizes the momentum of each matrix via Newton–Schulz (\\(\\sigma \\mapsto a\\sigma + b\\sigma^3 + c\\sigma^5\\)). Qwen3.8 and DeepSeek-V4 use Muon + AdamW and do without QK-Clip by normalizing \\(q\\) and \\(k\\) in the architecture itself (QK-Norm in Qwen; RMSNorm of Q and of the KV latent in DeepSeek).',
    'Generalizing means getting new data right: training/validation/test, underfitting × overfitting, and the remedies are more data, regularization (weight decay) and early stopping.',
    'Composite losses shape behavior: DSpark was trained with 0.1·CE + 0.9·TV + 1.0·BCE weighted by \\(e^{-(k-1)/\\gamma}\\). QAT (DeepSeek\'s FP8 KV and FP4 indexer) trains while already seeing the rounding, with STE in the backward.',
    'At inference only the forward exists, with the weights frozen. The recipe chooses <i>how</i> to compute (formats, kernels, parallelism, speculation), but it must reproduce what training fixed (8 experts, eps 1e-6, DSpark\'s block of 5, DFlash2\'s block of 8).',
  ],
  quiz: [
    { q: 'How many “predict the next token” training examples does a text with 1,001 tokens yield?', o: ['1', '1,000', '1,001', 'It depends on how many humans label it'], a: 1, e: 'Each position from 2 to 1,001 is the target of an example whose context is the preceding prefix: \\(N-1 = 1{,}000\\) pairs, with no human label. That is what makes self-supervised pre-training viable over trillions of tokens, and a Transformer with a causal mask computes all the pairs in a single forward.' },
    { q: 'The model gave probability 0.01 to the token that actually came. What is the cross-entropy loss at that position?', o: ['0.01 nat', '0.99 nat', '≈ 4.6 nats (≈ 6.6 bits)', '≈ 100 nats'], a: 2, e: '\\(-\\ln 0.01 = 4.605\\) nats, and dividing by \\(\\ln 2\\) gives ≈ 6.64 bits. Compare with the squared error \\((1-0.01)^2 \\approx 0.98\\), which barely distinguishes this error from one with \\(p = 0.1\\). Cross-entropy grows without limit when the model is confidently wrong.' },
    { q: 'In a quadratic valley with maximum curvature \\(\\lambda_{\\max} = 40\\), what happens to plain gradient descent if \\(\\eta = 0.06\\)?', o: ['It converges faster than with 0.04', 'It oscillates and diverges, because \\(\\eta > 2/\\lambda_{\\max} = 0.05\\)', 'It converges just the same, because the rate only changes the speed', 'It gets stuck in a local minimum'], a: 1, e: 'Along the direction with curvature \\(\\lambda\\), each step multiplies the error by \\(1 - \\eta\\lambda\\). With \\(\\eta\\lambda = 2.4\\), the factor is −1.4: the error flips sign and grows at every step. The steepest direction limits the rate of all the others, which is why ill-conditioned ravines are slow for plain descent.' },
    { q: 'What does Muon do with the momentum \\(M = U\\Sigma V^\\top\\) of a weight matrix before applying the step?', o: ['Divides each entry by the root mean square of that entry, like Adam', 'Replaces \\(M\\) with approximately \\(UV^\\top\\): same directions, singular values ≈ 1, computed with matrix multiplications only (Newton–Schulz)', 'Zeroes out the small singular values to save memory', 'Computes the exact SVD and keeps only the dominant direction'], a: 1, e: 'The iteration \\(X \\leftarrow aX + b(XX^\\top)X + c(XX^\\top)^2X\\) applies the polynomial \\(a\\sigma + b\\sigma^3 + c\\sigma^5\\) to each singular value without touching \\(U, V\\). With (3.4445, −4.7750, 2.0315), small values grow ~3.4× per iteration until they reach a band near 1. DeepSeek-V4 uses 8 of these iterations plus 2 polishing ones with (2, −1.5, 0.5). Adam, by contrast, only acts coordinate by coordinate.' },
    { q: 'Which of these values fixed in training has <b>no</b> effect whatsoever at inference?', o: ['<code>num_experts_per_tok</code> (GLM: 8)', '<code>router_aux_loss_coef</code> (0.001)', 'The eps of the RMSNorm in GLM\'s vision tower (trained with 1e-6)', '<code>dspark_block_size</code> (DeepSeek: 5)'], a: 1, e: 'The auxiliary balancing loss coefficient only exists inside the training loss function. The other three change the forward or what the recipe can ask for. With 7 experts instead of 8, it is a different model. The eps enters the RMSNorm computation: the vision <code>config.json</code> says 1e-5, but the recipe\'s vLLM ignores that field and forces 1e-6, because with 1e-5 the image descriptions degrade. And a \\(k\\) smaller than the trained block of 5 truncates DSpark\'s drafts.' },
    { q: 'Why is serving DeepSeek-V4\'s KV cache in FP8 (blocks of 64, power-of-2 scale) not a post-training degradation?', o: ['Because FP8 has the same precision as BF16', 'Because the model was trained with simulated quantization (QAT) of that KV: the training forward already saw the rounded values', 'Because the KV does not affect the model\'s output', 'Because vLLM corrects the error automatically'], a: 1, e: 'In QAT, the training forward passes the tensor through quantize → dequantize, and the backward uses the straight-through estimator. The weights adapt to the rounding. For DeepSeek-V4\'s NoPE KV, the FP8 format is the regime in which the model learned. GLM (EXL3) and Qwen (NVFP4) are PTQ: compressed after training and validated by KLD.' },
  ],
  mount(root, LA) {
    const V = LA.viz, UI = LA.ui;
    const $ = (s) => root.querySelector(s);

    /* ============================ helpers ============================ */
    const FC = {};
    const setFont = (g, sz, f = 'sans', wt = 400) => { const k = sz + '|' + f + '|' + wt; g.font = FC[k] || (FC[k] = V.font(sz, f, wt)); };
    const measure = (g, s, sz, f = 'sans', wt = 400) => { setFont(g, sz, f, wt); return g.measureText(String(s)).width; };
    const num = (v, d = 2) => (+v).toFixed(d);
    const fmtInt = (n) => Math.round(n).toLocaleString('en-US');
    const onInk = () => (V.C().dark ? '#00210F' : '#ffffff');
    const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;
    function fitH(fig, hd, hm, bp = 600) {
      const st = fig.querySelector('.la-fig-stage');
      const apply = () => { const w = fig.clientWidth || 800; const want = (w < bp ? hm : hd) + 'px'; if (st.style.height !== want) st.style.height = want; };
      apply();
      window.addEventListener('resize', apply);
      LA.onCleanup(() => window.removeEventListener('resize', apply));
    }
    function pill(g, s, x, y, o = {}) {
      const C = V.C();
      const sz = o.size || 12, f = o.font || 'mono', wt = o.weight || 500, px = o.px == null ? 7 : o.px, hh = o.h || sz + 10;
      const w = Math.max(o.minW || 0, measure(g, s, sz, f, wt) + px * 2);
      if (o.dry) return w;
      if (o.glow) V.glow(g, x + w / 2, y + hh / 2, Math.max(w, hh) * 0.75, o.glow, 0.45);
      if (o.dash) g.setLineDash(o.dash);
      V.box(g, x, y, w, hh, { r: o.r == null ? 7 : o.r, fill: o.fill || V.alpha(C.muted, 0.1), stroke: o.stroke, lw: o.lw || 1.2 });
      g.setLineDash([]);
      V.text(g, s, x + w / 2, y + hh / 2 + 0.5, { size: sz, font: f, weight: wt, color: o.color || C.ink, align: 'center', base: 'middle', alpha: o.alpha });
      return w;
    }
    function wrapLines(g, s, maxW, sz, f = 'sans', wt = 400) {
      const words = String(s).split(' ');
      const out = []; let line = '';
      words.forEach((wd) => { const tt = line ? line + ' ' + wd : wd; if (measure(g, tt, sz, f, wt) > maxW && line) { out.push(line); line = wd; } else line = tt; });
      if (line) out.push(line);
      return out;
    }
    function textBlock(g, s, x, y, maxW, o = {}) {
      const sz = o.size || 12.5, lh = o.lh || Math.round(sz * 1.42);
      const L = wrapLines(g, s, maxW, sz, o.font || 'sans', o.weight || 400);
      L.forEach((l, i) => V.text(g, l, x, y + i * lh, Object.assign({ size: sz }, o)));
      return y + L.length * lh;
    }
    function star(g, x, y, r, col) {
      g.beginPath();
      for (let k = 0; k < 10; k++) { const a = -Math.PI / 2 + (k * Math.PI) / 5, rr = k % 2 ? r * 0.45 : r; g.lineTo(x + Math.cos(a) * rr, y + Math.sin(a) * rr); }
      g.closePath(); g.fillStyle = col; g.fill();
    }
    /** Mapa de relevo em cache (offscreen) de uma função f(u,v), com escala uniforme. */
    function Field(fn, box, opt = {}) {
      let cv = null, key = '';
      const F = { dom: null };
      F.layout = (R) => {
        // domínio que cobre `box` com escala uniforme, centrado
        const bw = box.x1 - box.x0, bh = box.y1 - box.y0, cx = (box.x0 + box.x1) / 2, cy = (box.y0 + box.y1) / 2;
        const sc = Math.min(R.w / bw, R.h / bh);
        const dw = R.w / sc, dh = R.h / sc;
        F.dom = { x0: cx - dw / 2, x1: cx + dw / 2, y0: cy - dh / 2, y1: cy + dh / 2, sc };
        return F.dom;
      };
      F.toPx = (u, v, R) => [R.x + (u - F.dom.x0) * F.dom.sc, R.y + (F.dom.y1 - v) * F.dom.sc];
      F.fromPx = (px, py, R) => [F.dom.x0 + (px - R.x) / F.dom.sc, F.dom.y1 - (py - R.y) / F.dom.sc];
      F.norm = (L) => { const L0 = opt.L0 || 0.05, Lm = opt.Lmax || 10; return V.clamp(Math.log(1 + Math.max(0, L - (opt.Lmin || 0)) / L0) / Math.log(1 + Lm / L0), 0, 1); };
      F.draw = (g, R) => {
        const C = V.C();
        F.layout(R);
        const k = [Math.round(R.w), Math.round(R.h), C.dark, C.a, C.ink].join('|') + (opt.tag ? opt.tag() : '');
        if (k !== key) {
          key = k;
          const dpr = Math.min(window.devicePixelRatio || 1, 2);
          cv = cv || document.createElement('canvas');
          cv.width = Math.max(1, Math.round(R.w * dpr)); cv.height = Math.max(1, Math.round(R.h * dpr));
          const gg = cv.getContext('2d');
          gg.setTransform(dpr, 0, 0, dpr, 0, 0);
          const st = opt.step || 3, bands = opt.bands || 13;
          const sw = Math.ceil(R.w / st) + 1, sh = Math.ceil(R.h / st) + 1;
          // 1) preenchimento contínuo (amostras nos nós, ampliadas com suavização)
          const lut = [];
          for (let i = 0; i < 256; i++) { const c = V.cmap(0.06 + 0.86 * (1 - i / 255), 'seq'); const m = String(c).match(/[\d.]+/g) || [0, 0, 0]; lut.push([+m[0], +m[1], +m[2]]); }
          const sm = document.createElement('canvas'); sm.width = sw; sm.height = sh;
          const sg = sm.getContext('2d'), img = sg.createImageData(sw, sh);
          const T = new Float32Array(sw * sh);
          for (let j = 0; j < sh; j++) for (let i = 0; i < sw; i++) {
            const u = F.dom.x0 + (i * st) / F.dom.sc, v = F.dom.y1 - (j * st) / F.dom.sc;
            const tt = F.norm(fn(u, v)); T[j * sw + i] = tt;
            const c = lut[Math.round(tt * 255)], o = (j * sw + i) * 4;
            img.data[o] = c[0]; img.data[o + 1] = c[1]; img.data[o + 2] = c[2]; img.data[o + 3] = 255;
          }
          sg.putImageData(img, 0, 0);
          gg.imageSmoothingEnabled = true;
          gg.drawImage(sm, -st / 2, -st / 2, sw * st, sh * st);
          // 2) curvas de nível por marching squares
          gg.strokeStyle = V.alpha(C.ink, C.dark ? 0.2 : 0.22); gg.lineWidth = 1; gg.beginPath();
          const seg = (x1, y1, x2, y2) => { gg.moveTo(x1, y1); gg.lineTo(x2, y2); };
          for (let b = 1; b < bands; b++) {
            const lv = b / bands;
            for (let j = 0; j < sh - 1; j++) for (let i = 0; i < sw - 1; i++) {
              const a = T[j * sw + i], bq = T[j * sw + i + 1], c = T[(j + 1) * sw + i + 1], d = T[(j + 1) * sw + i];
              const idx = (a > lv ? 8 : 0) | (bq > lv ? 4 : 0) | (c > lv ? 2 : 0) | (d > lv ? 1 : 0);
              if (idx === 0 || idx === 15) continue;
              const x = i * st, y = j * st;
              const top = () => [x + (st * (lv - a)) / (bq - a), y], right = () => [x + st, y + (st * (lv - bq)) / (c - bq)];
              const bot = () => [x + (st * (lv - d)) / (c - d), y + st], left = () => [x, y + (st * (lv - a)) / (d - a)];
              const L2 = (p, q) => seg(p[0], p[1], q[0], q[1]);
              switch (idx) {
                case 1: case 14: L2(left(), bot()); break;
                case 2: case 13: L2(bot(), right()); break;
                case 3: case 12: L2(left(), right()); break;
                case 4: case 11: L2(top(), right()); break;
                case 6: case 9: L2(top(), bot()); break;
                case 7: case 8: L2(left(), top()); break;
                case 5: L2(left(), top()); L2(bot(), right()); break;
                case 10: L2(top(), right()); L2(left(), bot()); break;
                default: break;
              }
            }
          }
          gg.stroke();
        }
        g.save(); V.rr(g, R.x, R.y, R.w, R.h, 10); g.clip();
        g.drawImage(cv, R.x, R.y, R.w, R.h);
        g.restore();
        g.strokeStyle = V.alpha(C.muted, 0.35); g.lineWidth = 1; V.rr(g, R.x, R.y, R.w, R.h, 10); g.stroke();
      };
      return F;
    }
    const grad2 = (f, u, v, e = 1e-5) => [(f(u + e, v) - f(u - e, v)) / (2 * e), (f(u, v + e) - f(u, v - e)) / (2 * e)];
    // rótulos de controles são CAIXA ALTA no CSS: protege letras gregas/variáveis da transformação
    const gk = (s) => `<span style="text-transform:none;letter-spacing:0">${s}</span>`;
    const relabel = (ctl, html) => { const l = ctl && ctl.el && ctl.el.querySelector('label, span'); if (l) l.innerHTML = html; return ctl; };
    const SUBD = '₀₁₂₃₄₅₆₇₈₉';
    const sub = (n) => String(n).split('').map((c) => SUBD[+c]).join('');

    /* ================= Figura 1: a máquina de dois botões ================= */
    (function fig1() {
      const f1 = $('#fig-reta');
      fitH(f1, 390, 610);
      const TX = [16, 18, 19, 21, 22, 24, 25, 27, 28, 30, 32, 34];
      const TY = [25, 22, 38, 34, 49, 39, 52, 62, 50, 71, 68, 82];
      const mx = mean(TX), my = mean(TY);
      const wOpt = TX.reduce((s, x, i) => s + (x - mx) * (TY[i] - my), 0) / TX.reduce((s, x) => s + (x - mx) ** 2, 0);
      const bOpt = my - wOpt * mx;
      const mseOf = (w, b) => mean(TX.map((x, i) => (w * x + b - TY[i]) ** 2));
      const mOpt = mseOf(wOpt, bOpt);
      const Fm = Field((u, v) => mseOf(6 * u, 120 * v - 80), { x0: 0, x1: 1, y0: 0, y1: 1 }, { Lmin: mOpt, L0: 15, Lmax: 3000, bands: 10, step: 2 });
      const S = { w: 1.2, b: 20, dw: 1.2, db: 20, sq: true, opt: true };
      const c1 = UI.controls(f1);
      relabel(UI.slider(c1, { label: 'w (slope)', min: 0, max: 6, step: 0.05, value: S.w, fmt: (v) => num(v, 2), onInput: (v) => { S.w = v; } }), gk('w') + ' (slope)');
      relabel(UI.slider(c1, { label: 'b (intercept)', min: -80, max: 40, step: 1, value: S.b, fmt: (v) => String(v), onInput: (v) => { S.b = v; } }), gk('b') + ' (intercept)');
      UI.toggle(c1, { label: 'show squares', value: true, onChange: (v) => { S.sq = v; } });
      UI.toggle(c1, { label: 'optimal line', value: true, onChange: (v) => { S.opt = v; } });
      const ro = UI.readout(c1, 'current MSE', '—');
      UI.readout(c1, 'optimal MSE', num(mOpt, 1));
      let lastR = '';
      V.fig(f1, (g, w, h, t, dt, _S, H) => {
        const C = V.C();
        S.dw += (S.w - S.dw) * Math.min(1, dt * 10); S.db += (S.b - S.db) * Math.min(1, dt * 10);
        const W = S.dw, B = S.db;
        const wide = w >= 600;
        const P = wide ? { x: 58, y: 30, w: w * 0.6 - 58, h: h - 76 } : { x: 46, y: 28, w: w - 62, h: h * 0.55 - 58 };
        const Q = wide ? { x: w * 0.64, y: 20, w: w * 0.36 - 18, h: h - 36 } : { x: 14, y: h * 0.55 + 4, w: w - 28, h: h * 0.45 - 16 };
        const X0 = 14, X1 = 36, Y0 = 0, Y1 = 100;
        const px = (x) => P.x + ((x - X0) / (X1 - X0)) * P.w, py = (y) => P.y + P.h - ((y - Y0) / (Y1 - Y0)) * P.h;
        // grade e eixos
        g.strokeStyle = C.grid; g.lineWidth = 1; g.beginPath();
        for (let x = 15; x <= 35; x += 5) { g.moveTo(px(x), P.y); g.lineTo(px(x), P.y + P.h); }
        for (let y = 0; y <= 100; y += 20) { g.moveTo(P.x, py(y)); g.lineTo(P.x + P.w, py(y)); }
        g.stroke();
        for (let x = 15; x <= 35; x += 5) V.text(g, x + '°', px(x), P.y + P.h + 15, { size: 11, font: 'mono', color: C.muted, align: 'center' });
        for (let y = 0; y <= 100; y += 20) V.text(g, String(y), P.x - 7, py(y) + 4, { size: 11, font: 'mono', color: C.muted, align: 'right' });
        V.text(g, 'popsicles sold', P.x - 40, P.y - 12, { size: 11.5, color: C.muted });
        V.text(g, 'daily temperature', P.x + P.w, P.y + P.h + 32, { size: 11.5, color: C.muted, align: 'right' });
        g.save(); g.beginPath(); g.rect(P.x, P.y, P.w, P.h); g.clip();
        // reta ótima
        if (S.opt) { V.line(g, px(X0), py(wOpt * X0 + bOpt), px(X1), py(wOpt * X1 + bOpt), { color: V.alpha(C.ink, 0.45), lw: 1.5, dash: [6, 6] }); }
        // quadrados
        if (S.sq) TX.forEach((x, i) => {
          const yh = W * x + B, a = py(TY[i]), bb = py(yh), side = Math.abs(bb - a);
          const x0 = px(x) + side > P.x + P.w ? px(x) - side : px(x);
          g.fillStyle = V.alpha(C.c, 0.11); g.fillRect(x0, Math.min(a, bb), side, side);
          g.strokeStyle = V.alpha(C.c, 0.45); g.lineWidth = 1; g.strokeRect(x0, Math.min(a, bb), side, side);
        });
        // reta do modelo
        g.lineWidth = 3; g.strokeStyle = C.a; g.beginPath(); g.moveTo(px(X0), py(W * X0 + B)); g.lineTo(px(X1), py(W * X1 + B)); g.stroke();
        // molas
        TX.forEach((x, i) => {
          const yh = W * x + B, a = py(TY[i]), bb = py(yh), n = 9, amp = 3.5 + 0.8 * Math.sin(t * 4 + i);
          const err = Math.min(1, Math.abs(yh - TY[i]) / 35);
          g.strokeStyle = V.alpha(err > 0.5 ? C.c : C.b, 0.55 + 0.4 * err); g.lineWidth = 1.4; g.beginPath(); g.moveTo(px(x), a);
          for (let k = 1; k < n; k++) g.lineTo(px(x) + (k % 2 ? amp : -amp), a + ((bb - a) * k) / n);
          g.lineTo(px(x), bb); g.stroke();
        });
        g.restore();
        // pontos: casquinhas
        let hov = -1;
        TX.forEach((x, i) => {
          const cx = px(x), cy = py(TY[i]) + Math.sin(t * 2 + i * 1.3) * 0.8;
          if (H.pointer.inside && Math.hypot(H.pointer.x - cx, H.pointer.y - cy) < 12) hov = i;
          g.fillStyle = C.d; g.beginPath(); g.moveTo(cx - 5, cy + 3); g.lineTo(cx + 5, cy + 3); g.lineTo(cx, cy + 14); g.closePath(); g.fill();
          V.dot(g, cx, cy, hov === i ? 7.5 : 6, C.c);
          V.dot(g, cx - 2, cy - 2, 1.8, V.alpha('#ffffff', 0.7));
        });
        if (hov >= 0) {
          const x = TX[hov], yh = W * x + B;
          const s = `${x}°C: predicted ${num(yh, 0)}, actual ${TY[hov]}`;
          const ww = pill(g, s, 0, 0, { dry: true, size: 11.5 });
          pill(g, s, Math.min(P.x + P.w - ww, Math.max(P.x, px(x) - ww / 2)), py(TY[hov]) - 38, { size: 11.5, fill: C.panel, stroke: C.c });
        }
        // ---------- painel: a máquina ----------
        const mse = mseOf(W, B);
        const rs = num(mseOf(S.w, S.b), 1); if (rs !== lastR) { ro.set(rs); lastR = rs; }
        const MH = Math.min(wide ? 150 : 118, Q.h * 0.55);
        const M = { x: Q.x, y: Q.y + 4, w: Q.w, h: MH };
        V.box(g, M.x, M.y, M.w, M.h, { r: 14, fill: V.alpha(C.b, 0.1), stroke: V.alpha(C.b, 0.55), lw: 1.5 });
        V.text(g, 'machine f', M.x + 14, M.y + 22, { size: 13, font: 'display', color: C.ink });
        V.text(g, 'θ', M.x + 14 + measure(g, 'machine f', 13, 'display') + 1, M.y + 26, { size: 10, font: 'sans', color: C.ink });
        V.text(g, 'ŷ = w·x + b', M.x + M.w - 14, M.y + 22, { size: 12, font: 'mono', color: C.muted, align: 'right' });
        const kr = Math.min(26, M.h * 0.2, M.w * 0.13);
        [[S.dw, 0, 6, 'w', num(S.dw, 2)], [S.db, -80, 40, 'b', num(S.db, 0)]].forEach(([val, lo, hi, nm, lab], k) => {
          const cx = M.x + M.w * (k ? 0.72 : 0.28), cy = M.y + M.h * 0.55;
          const a = -Math.PI * 0.75 + ((val - lo) / (hi - lo)) * Math.PI * 1.5 - Math.PI / 2;
          for (let q = 0; q <= 10; q++) { const aa = -Math.PI * 1.25 + (q / 10) * Math.PI * 1.5; V.line(g, cx + Math.cos(aa) * (kr + 3), cy + Math.sin(aa) * (kr + 3), cx + Math.cos(aa) * (kr + 7), cy + Math.sin(aa) * (kr + 7), { color: V.alpha(C.muted, 0.6), lw: 1 }); }
          V.dot(g, cx, cy, kr, C.panel); g.strokeStyle = V.alpha(C.b, 0.8); g.lineWidth = 2; g.beginPath(); g.arc(cx, cy, kr, 0, Math.PI * 2); g.stroke();
          V.line(g, cx, cy, cx + Math.cos(a) * (kr - 5), cy + Math.sin(a) * (kr - 5), { color: C.a, lw: 3 });
          V.dot(g, cx, cy, 3, C.a);
          V.text(g, nm + ' = ' + lab, cx, cy + kr + 20, { size: 12, font: 'mono', color: C.ink, align: 'center' });
        });
        // placar MSE (escala log)
        const gy = M.y + M.h + 24;
        V.text(g, 'SCOREBOARD · MSE (popsicles²)', Q.x, gy, { size: 11, font: 'mono', color: C.muted });
        const gb = { x: Q.x, y: gy + 8, w: Q.w, h: 16 };
        V.box(g, gb.x, gb.y, gb.w, gb.h, { r: 8, fill: V.alpha(C.muted, 0.14) });
        const tt = V.clamp(Math.log(mse / mOpt) / Math.log(60), 0, 1);
        const col = tt < 0.25 ? C.a : tt < 0.6 ? C.d : C.c;
        const fw = Math.max(10, gb.w * (0.04 + 0.96 * tt));
        V.box(g, gb.x, gb.y, fw, gb.h, { r: 8, fill: V.alpha(col, 0.8) });
        V.line(g, gb.x + gb.w * 0.04, gb.y - 4, gb.x + gb.w * 0.04, gb.y + gb.h + 4, { color: C.ink, lw: 1.5 });
        const mtxt = num(mse, 1), mtw = measure(g, mtxt, 11.5, 'mono', 600);
        if (fw > mtw + 26) V.text(g, mtxt, gb.x + fw - 8, gb.y + 12.5, { size: 11.5, font: 'mono', weight: 600, color: onInk(), align: 'right' });
        else V.text(g, mtxt, gb.x + fw + 6, gb.y + 12.5, { size: 11.5, font: 'mono', weight: 600, color: C.ink });
        V.text(g, 'optimum ' + num(mOpt, 1), gb.x, gb.y + gb.h + 16, { size: 11, font: 'mono', color: C.muted });
        V.text(g, num(mse / mOpt, 1) + '× the optimum', gb.x + gb.w, gb.y + gb.h + 16, { size: 11, font: 'mono', color: col, align: 'right' });
        const ny = gb.y + gb.h + 40;
        const room = Q.y + Q.h - (ny - 8);
        if (wide && room >= 84) {
          const ms = Math.min(104, room); const RM = { x: Q.x, y: ny - 8, w: ms, h: ms };
          Fm.draw(g, RM);
          const po = Fm.toPx(wOpt / 6, (bOpt + 80) / 120, RM), pp = Fm.toPx(S.dw / 6, (S.db + 80) / 120, RM);
          star(g, po[0], po[1], 5, C.d);
          V.glow(g, pp[0], pp[1], 12, C.c, 0.8); V.dot(g, pp[0], pp[1], 4.5, C.c);
          V.text(g, 'w →', RM.x + RM.w - 4, RM.y + RM.h - 5, { size: 9.5, font: 'mono', color: C.ink, align: 'right' });
          V.text(g, 'b', RM.x + 5, RM.y + 12, { size: 9.5, font: 'mono', color: C.ink });
          textBlock(g, 'The map shows the MSE for each position (w, b): the dot is you, the star is the bottom. This machine has 2 knobs; GLM-5.3-Flash has 321,323,031,390.', RM.x + ms + 12, ny + 4, Q.w - ms - 12, { size: 11.5, color: C.muted, lh: 16 });
        } else if (ny + 30 < Q.y + Q.h) textBlock(g, 'This machine has 2 knobs. GLM-5.3-Flash has 321,323,031,390, and no closed-form formula finds their optimal position.', Q.x, ny, Q.w, { size: 12, color: C.muted, lh: 17 });
      });
    })();

    /* ================= Figura 2: a esteira do próximo token ================= */
    (function fig2() {
      const f2 = $('#fig-esteira');
      fitH(f2, 380, 330);
      const TOK = ['The', ' cat', ' got', ' up', ' on', ' the', ' roof', '.'];
      const SUP = [
        { x: '“Win a free iPhone!!!”', y: 'spam' },
        { x: '“Meeting moved to 3 pm”', y: 'ok' },
        { x: '“Your bill is overdue, click here”', y: 'spam' },
        { x: '“Photos from the barbecue”', y: 'ok' },
      ];
      const S = { mode: 'auto', all: false, t0: -3.8 };
      const c2 = UI.controls(f2);
      UI.seg(c2, { label: 'Mode', options: [{ v: 'auto', label: 'self-supervised' }, { v: 'sup', label: 'supervised' }], value: 'auto', onChange: (v) => { S.mode = v; S.t0 = H2.t - (v === 'auto' ? 3.8 : 0); } });
      UI.toggle(c2, { label: 'all positions at once', value: false, onChange: (v) => { S.all = v; S.t0 = H2.t - (v ? 0 : 3.8); } });
      const vis = (s) => (s.startsWith(' ') ? '␣' + s.slice(1) : s);
      function chip(g, s, x, y, sz, o = {}) {
        const C = V.C();
        const w = measure(g, vis(s), sz, 'mono', 500) + 12, hh = o.h || sz + 13;
        if (o.dry) return w;
        V.box(g, x, y, w, hh, { r: 8, fill: o.fill || V.alpha(C.b, 0.2), stroke: o.stroke || V.alpha(C.b, 0.7), lw: 1.3 });
        if (o.eyes) {
          const ex = x + w / 2, ey = y + 6, blink = Math.sin(o.t * 1.3 + x * 0.05) > 0.97;
          if (blink) { V.line(g, ex - 5, ey, ex - 2, ey, { color: C.ink, lw: 1.2 }); V.line(g, ex + 2, ey, ex + 5, ey, { color: C.ink, lw: 1.2 }); }
          else { V.dot(g, ex - 3.5, ey, 1.5, C.ink); V.dot(g, ex + 3.5, ey, 1.5, C.ink); }
        }
        V.text(g, vis(s), x + w / 2, y + hh / 2 + (o.eyes ? 4 : 0.5), { size: sz, font: 'mono', weight: 500, color: o.color || C.ink, align: 'center', base: 'middle', alpha: o.alpha });
        return w;
      }
      const H2 = V.fig(f2, (g, w, h, t) => {
        const C = V.C();
        const narrow = w < 600;
        const sz = narrow ? 11 : 13, szs = narrow ? 10.5 : 12;
        const tt = t - S.t0;
        if (S.mode === 'auto') {
          // posições das fichas
          const ws = TOK.map((s) => chip(g, s, 0, 0, sz, { dry: true }));
          const gap = narrow ? 3 : 8;
          const tot = ws.reduce((a, b) => a + b, 0) + gap * (TOK.length - 1);
          const x0 = Math.max(10, (w - tot) / 2);
          const xs = []; let acc = x0; ws.forEach((ww) => { xs.push(acc); acc += ww + gap; });
          const by = narrow ? 34 : 40, ch = sz + 22;
          // esteira
          const bx0 = Math.max(4, x0 - 14), bx1 = Math.min(w - 4, x0 + tot + 14), beltY = by + ch + 6;
          V.box(g, bx0, beltY, bx1 - bx0, 12, { r: 6, fill: V.alpha(C.muted, 0.16), stroke: V.alpha(C.muted, 0.35), lw: 1 });
          for (let k = 0; k * 22 < bx1 - bx0 - 8; k++) {
            const rx = bx0 + 10 + k * 22, ry = beltY + 6, a = t * 3 + k;
            g.strokeStyle = V.alpha(C.muted, 0.5); g.lineWidth = 1; g.beginPath(); g.arc(rx, ry, 4, 0, Math.PI * 2); g.stroke();
            V.line(g, rx, ry, rx + Math.cos(a) * 4, ry + Math.sin(a) * 4, { color: V.alpha(C.muted, 0.7), lw: 1 });
          }
          if (!S.all) {
            const P = 1.8, pos = Math.floor((tt % (7 * P)) / P), fr = (tt % P) / P;
            TOK.forEach((s, i) => {
              if (i <= pos) chip(g, s, xs[i], by, sz, { eyes: true, t, h: ch });
              else if (i === pos + 1) {
                if (fr < 0.4) {
                  const pulse = 0.5 + 0.5 * Math.sin(t * 8);
                  V.box(g, xs[i], by, ws[i], ch, { r: 8, fill: V.alpha(C.a, 0.08 + 0.1 * pulse), stroke: C.a, lw: 1.5 });
                  V.text(g, '?', xs[i] + ws[i] / 2, by + ch / 2 + 1, { size: sz + 3, font: 'display', color: C.a, align: 'center', base: 'middle' });
                } else chip(g, s, xs[i], by, sz, { eyes: true, t, h: ch, fill: V.alpha(C.a, 0.25), stroke: C.a });
              } else chip(g, s, xs[i], by, sz, { h: ch, fill: V.alpha(C.muted, 0.06), stroke: V.alpha(C.muted, 0.25), alpha: 0.3 });
            });
            // colchetes
            const cx1 = xs[pos] + ws[pos];
            V.line(g, xs[0], by - 8, cx1, by - 8, { color: C.b, lw: 2 });
            V.text(g, 'context', (xs[0] + cx1) / 2, by - 14, { size: 11, font: 'mono', color: C.b, align: 'center' });
            if (pos + 1 < TOK.length) V.text(g, 'target', xs[pos + 1] + ws[pos + 1] / 2, by - 14, { size: 11, font: 'mono', color: C.a, align: 'center' });
            if (pos + 2 < TOK.length) {
              const fx0 = xs[pos + 2], fx1 = xs[TOK.length - 1] + ws[TOK.length - 1];
              const lab = fx1 - fx0 > measure(g, 'future (hidden)', 11, 'mono') + 4 ? 'future (hidden)' : fx1 - fx0 > measure(g, 'future', 11, 'mono') + 2 ? 'future' : '';
              if (lab) V.text(g, lab, (fx0 + fx1) / 2, by - 14, { size: 11, font: 'mono', color: C.muted, align: 'center' });
            }
            // lista de pares
            const ly = beltY + (narrow ? 34 : 38), rh = narrow ? V.clamp((h - ly - 8) / 7, 26, 31) : 27; // no celular, as 7 linhas ocupam a altura
            V.text(g, 'TRAINING PAIRS GENERATED FOR FREE', 14, ly - 8, { size: 11, font: 'mono', color: C.muted });
            V.text(g, `${pos + (fr > 0.6 ? 1 : 0)} / 7`, w - 14, ly - 8, { size: 12, font: 'mono', color: C.a, align: 'right', weight: 600 });
            const nShow = pos + (fr > 0.6 ? 1 : 0);
            // linhas-fantasma: o espaço de todos os pares que esta frase vai render
            for (let r = nShow; r < 7; r++) {
              const y = ly + r * rh; let x = 14;
              g.setLineDash([3, 4]);
              for (let i = 0; i <= r; i++) { const ww = chip(g, TOK[i], 0, 0, szs, { dry: true }); V.box(g, x, y, ww, rh - 5, { r: 7, stroke: V.alpha(C.muted, 0.28), lw: 1 }); x += ww + 3; }
              const wt = chip(g, TOK[r + 1], 0, 0, szs, { dry: true }); V.box(g, x + 22, y, wt, rh - 5, { r: 7, stroke: V.alpha(C.a, 0.3), lw: 1 });
              g.setLineDash([]);
            }
            for (let r = 0; r < nShow; r++) {
              const y = ly + r * rh; let x = 14;
              const fly = r === pos && fr > 0.6 ? V.ease.out((fr - 0.6) / 0.4) : 1;
              g.globalAlpha = fly;
              for (let i = 0; i <= r; i++) x += chip(g, TOK[i], x, y, szs, { h: rh - 5 }) + 3;
              V.text(g, '→', x + 6, y + (rh - 5) / 2 + 4, { size: 13, color: C.muted });
              chip(g, TOK[r + 1], x + 22, y, szs, { h: rh - 5, fill: V.alpha(C.a, 0.25), stroke: C.a });
              g.globalAlpha = 1;
            }
          } else {
            // escada: todas as posições no mesmo forward
            TOK.forEach((s, i) => chip(g, s, xs[i], by, sz, { eyes: true, t, h: ch, fill: i === 0 ? V.alpha(C.b, 0.2) : V.alpha(C.b, 0.2) }));
            const ly = beltY + (narrow ? 30 : 34), rh = Math.min(30, (h - ly - (narrow ? 44 : 30)) / 7);
            V.text(g, '1 FORWARD → 7 POSITIONS → 7 LOSSES', 14, ly - 6, { size: 11, font: 'mono', color: C.muted });
            const scan = (tt * 1.2) % 9;
            for (let r = 0; r < 7; r++) {
              const y = ly + r * rh;
              const on = Math.floor(scan) === r;
              for (let i = 0; i < 8; i++) {
                const x = xs[i], ww = ws[i];
                if (i <= r) V.box(g, x, y + 2, ww, rh - 6, { r: 6, fill: V.alpha(C.b, on ? 0.35 : 0.16), stroke: V.alpha(C.b, 0.5), lw: 1 });
                else if (i === r + 1) {
                  V.box(g, x, y + 2, ww, rh - 6, { r: 6, fill: V.alpha(C.a, on ? 0.5 : 0.28), stroke: C.a, lw: 1.4 });
                  V.text(g, vis(TOK[i]), x + ww / 2, y + rh / 2, { size: szs - 0.5, font: 'mono', color: C.ink, align: 'center', base: 'middle' });
                } else {
                  g.save(); V.rr(g, x, y + 2, ww, rh - 6, 6); g.clip();
                  g.fillStyle = V.alpha(C.muted, 0.08); g.fillRect(x, y, ww, rh);
                  g.strokeStyle = V.alpha(C.muted, 0.35); g.lineWidth = 1;
                  for (let k = -rh; k < ww; k += 6) { g.beginPath(); g.moveTo(x + k, y + rh); g.lineTo(x + k + rh, y); g.stroke(); }
                  g.restore();
                }
              }
              if (on && !narrow) V.text(g, `−ln p(${vis(TOK[r + 1]).trim()} | prefix)`, w - 14, y + rh / 2 + 4, { size: 11.5, font: 'mono', color: C.a, align: 'right' });
            }
            const legY = ly + 7 * rh + 14;
            if (legY < h - 4) textBlock(g, 'hatched = forbidden future (causal mask) · mint = target of each row', 14, legY, w - 28, { size: 11, color: C.muted, lh: 15 });
          }
        } else {
          // ------- supervisionado: anotador humano -------
          const P = 2.8, idx = Math.floor(tt / P) % SUP.length, fr = (tt % P) / P, round = Math.floor(tt / (P * SUP.length));
          const ax = narrow ? w - 58 : w * 0.74, ay = narrow ? 64 : 70;
          // anotador
          V.dot(g, ax, ay, 15, V.alpha(C.d, 0.3)); g.strokeStyle = C.d; g.lineWidth = 2; g.beginPath(); g.arc(ax, ay, 15, 0, Math.PI * 2); g.stroke();
          V.dot(g, ax - 5, ay - 3, 1.8, C.ink); V.dot(g, ax + 5, ay - 3, 1.8, C.ink);
          g.beginPath(); g.arc(ax, ay + 3, 5, 0.15 * Math.PI, 0.85 * Math.PI); g.strokeStyle = C.ink; g.lineWidth = 1.4; g.stroke();
          V.box(g, ax - 20, ay + 18, 40, 34, { r: 10, fill: V.alpha(C.d, 0.25), stroke: C.d, lw: 1.5 });
          V.text(g, 'annotator', ax, ay + 68, { size: 11, font: 'mono', color: C.d, align: 'center' });
          // cartão chegando
          const cw = Math.min(narrow ? 200 : 260, w * 0.62), chh = 40;
          const cx0 = 14, cx1 = ax - 34 - cw;
          const u = V.ease.inOut(V.clamp(fr / 0.4, 0, 1));
          const cx = V.lerp(cx0 - cw * 0.2, cx1, u), cy = ay - chh / 2;
          V.box(g, cx, cy, cw, chh, { r: 9, fill: C.panel, stroke: V.alpha(C.muted, 0.6), lw: 1.3 });
          V.text(g, SUP[idx].x, cx + 10, cy + chh / 2 + 4, { size: szs, font: 'mono', color: C.ink, maxW: cw - 20 });
          if (fr > 0.4 && fr < 0.65) { for (let k = 0; k < 3; k++) V.dot(g, ax + 24 + k * 7, ay - 22, 2.2, V.alpha(C.d, (Math.floor(t * 5) % 3) === k ? 1 : 0.3)); }
          if (fr >= 0.65) {
            const lb = SUP[idx].y, spam = lb === 'spam';
            const pw = pill(g, lb, 0, 0, { dry: true, size: 12 });
            const sc = V.ease.back(V.clamp((fr - 0.65) / 0.15, 0, 1));
            g.save(); g.translate(cx + cw - pw / 2 - 6, cy - 4); g.scale(sc, sc);
            pill(g, lb, -pw / 2, -11, { size: 12, fill: V.alpha(spam ? C.c : C.a, 0.3), stroke: spam ? C.c : C.a, weight: 700 });
            g.restore();
          }
          // lista
          const ly = ay + 96, rh = narrow ? 30 : 32;
          V.text(g, 'HAND-LABELED PAIRS', 14, ly - 10, { size: 11, font: 'mono', color: C.muted });
          const done = round > 0 ? SUP.length : idx + (fr >= 0.8 ? 1 : 0);
          const nLab = round * SUP.length + idx + (fr >= 0.8 ? 1 : 0), roomL = w - 28 - measure(g, 'HAND-LABELED PAIRS', 11, 'mono') - 14;
          const cntL = `${nLab} labels · ${nLab} × (one person's time)`, cntS = `${nLab} labels`;
          V.text(g, measure(g, cntL, 11, 'mono') <= roomL ? cntL : cntS, w - 14, ly - 10, { size: 11, font: 'mono', color: C.d, align: 'right' });
          for (let r = 0; r < done; r++) {
            const y = ly + r * rh;
            const ww = Math.min(cw, w - 120);
            V.box(g, 14, y, ww, rh - 7, { r: 7, fill: V.alpha(C.muted, 0.08), stroke: V.alpha(C.muted, 0.35), lw: 1 });
            V.text(g, SUP[r].x, 22, y + (rh - 7) / 2 + 4, { size: szs, font: 'mono', color: C.ink, maxW: ww - 16 });
            V.text(g, '→', 14 + ww + 10, y + (rh - 7) / 2 + 5, { size: 13, color: C.muted });
            pill(g, SUP[r].y, 14 + ww + 28, y, { size: szs, h: rh - 7, fill: V.alpha(SUP[r].y === 'spam' ? C.c : C.a, 0.25), stroke: SUP[r].y === 'spam' ? C.c : C.a });
          }
          const ny = ly + SUP.length * rh + 12;
          if (ny < h - 6) textBlock(g, 'Each label costs human attention. In self-supervised learning, the label is the text itself.', 14, ny, w - 28, { size: 11.5, color: C.muted, lh: 16 });
        }
      });
    })();

    /* ================= Figura 3: a régua da surpresa ================= */
    (function fig3() {
      const f3 = $('#fig-ce');
      fitH(f3, 380, 620);
      const CAND = [' roof', ' wall', ' sofa', ' car', ' fence', ' bench'];
      const ZO = [0, 2.5, 1.8, 1.2, 0.4, 0.0];
      const S = { z: 1.5, zd: 1.5, unit: 'nats', mse: true };
      const c3 = UI.controls(f3);
      UI.slider(c3, { label: 'logit of “roof”', min: -3, max: 8, step: 0.1, value: S.z, fmt: (v) => num(v, 1), onInput: (v) => { S.z = v; } });
      UI.seg(c3, { label: 'Unit', options: ['nats', 'bits'], value: 'nats', onChange: (v) => { S.unit = v; } });
      UI.toggle(c3, { label: 'compare with squared error', value: true, onChange: (v) => { S.mse = v; } });
      const rP = relabel(UI.readout(c3, 'p(right)', '—'), gk('p') + '(right)'), rL = UI.readout(c3, 'loss', '—'), rN = UI.readout(c3, 'odds 1 in', '—');
      let last = '';
      V.fig(f3, (g, w, h, t, dt) => {
        const C = V.C();
        S.zd += (S.z - S.zd) * Math.min(1, dt * 9);
        const z = ZO.slice(); z[0] = S.zd;
        const m = Math.max(...z), e = z.map((v) => Math.exp(v - m)), se = e.reduce((a, b) => a + b, 0), p = e.map((v) => v / se);
        const bitsU = S.unit === 'bits', k = bitsU ? 1 / Math.LN2 : 1;
        const pz = (() => { const zz = ZO.slice(); zz[0] = S.z; const mm = Math.max(...zz); const ee = zz.map((v) => Math.exp(v - mm)); return ee[0] / ee.reduce((a, b) => a + b, 0); })();
        const key = pz.toFixed(4) + S.unit;
        if (key !== last) { last = key; rP.set(num(pz * 100, 1) + '%'); rL.set(num(-Math.log(pz) * k, 2) + (bitsU ? ' bits' : ' nats')); rN.set(fmtInt(1 / pz)); }
        const wide = w >= 600;
        const A = wide ? { x: 18, y: 44, w: w * 0.42 - 18, h: h - 104 } : { x: 14, y: 44, w: w - 28, h: h * 0.42 - 70 };
        const Bq = wide ? { x: w * 0.5 + 34, y: 30, w: w * 0.5 - 60, h: h - 80 } : { x: 48, y: h * 0.47 + 18, w: w - 70, h: h * 0.53 - 64 };
        // ---- barras
        V.text(g, 'The cat got up on the ___', A.x, 24, { size: 13, font: 'mono', color: C.ink });
        const n = CAND.length, gap = wide ? 10 : 6, bw = (A.w - gap * (n - 1)) / n;
        p.forEach((pv, i) => {
          const x = A.x + i * (bw + gap), bh = Math.max(2, pv * A.h), y = A.y + A.h - bh;
          const right = i === 0;
          V.box(g, x, A.y, bw, A.h, { r: 6, fill: V.alpha(C.muted, 0.06) });
          V.box(g, x, y, bw, bh, { r: 6, fill: right ? V.alpha(C.a, 0.85) : V.alpha(C.b, 0.55) });
          V.text(g, num(pv * 100, pv < 0.1 ? 1 : 0) + '%', x + bw / 2, y - 6, { size: 11, font: 'mono', color: C.ink, align: 'center' });
          V.text(g, CAND[i].trim(), x + bw / 2, A.y + A.h + 15, { size: wide ? 11.5 : 10.5, font: 'mono', color: right ? C.a : C.muted, align: 'center', maxW: bw + gap });
          V.text(g, 'z=' + num(z[i], 1), x + bw / 2, A.y + A.h + 30, { size: 10.5, font: 'mono', color: C.muted, align: 'center', maxW: bw + gap });
          if (right) star(g, x + bw / 2, Math.min(y - 24, A.y + A.h - 30), 7 + Math.sin(t * 3) * 0.8, C.d);
        });
        // ---- curva -ln p
        const Ymax = bitsU ? 7.5 : 5.2;
        const X = (pp) => Bq.x + pp * Bq.w, Y = (l) => Bq.y + Bq.h - (Math.min(l, Ymax) / Ymax) * Bq.h;
        g.strokeStyle = C.grid; g.lineWidth = 1; g.beginPath();
        for (let q = 0; q <= 1.0001; q += 0.25) { g.moveTo(X(q), Bq.y); g.lineTo(X(q), Bq.y + Bq.h); }
        for (let l = 0; l <= Ymax; l += 1) { g.moveTo(Bq.x, Y(l)); g.lineTo(Bq.x + Bq.w, Y(l)); }
        g.stroke();
        for (let q = 0; q <= 1.0001; q += 0.25) V.text(g, num(q, 2), X(q), Bq.y + Bq.h + 15, { size: 10.5, font: 'mono', color: C.muted, align: 'center' });
        for (let l = 0; l <= Ymax; l += 1) V.text(g, String(l), Bq.x - 7, Y(l) + 4, { size: 10.5, font: 'mono', color: C.muted, align: 'right' });
        V.text(g, 'p(right token)', Bq.x + Bq.w, Bq.y + Bq.h + 31, { size: 11, color: C.muted, align: 'right' });
        const lblL = bitsU ? 'loss = −log₂ p' : 'loss = −ln p';
        V.text(g, lblL, Bq.x, Bq.y - 10, { size: 11.5, font: 'mono', color: C.a });
        // "↑ ∞" logo abaixo da primeira linha de grade, afastado da curva que abraça o eixo
        const infL = '↑ ∞ as p → 0', infX = X(0) + 14, infY = Y(Math.floor(Ymax)) + 15, infW = measure(g, infL, 11, 'mono');
        // curva MSE (1-p)^2 em unidades próprias
        if (S.mse) {
          g.setLineDash([5, 5]); g.strokeStyle = V.alpha(C.d, 0.9); g.lineWidth = 1.6; g.beginPath();
          for (let i = 0; i <= 60; i++) { const q = i / 60; const yy = Y((1 - q) ** 2); i ? g.lineTo(X(q), yy) : g.moveTo(X(q), yy); }
          g.stroke(); g.setLineDash([]);
          // legenda: numa linha ao lado do título se couber; senão, em duas linhas dentro do gráfico; senão, curta
          const full = '┄ (1−p)²: squared error, never above 1', l1 = '┄ (1−p)²: squared error', l2 = 'never above 1';
          const oM = { size: 10.5, font: 'mono', color: C.d, align: 'right' };
          if (measure(g, lblL, 11.5, 'mono') + measure(g, full, 10.5, 'mono') + 12 <= Bq.w) V.text(g, full, Bq.x + Bq.w, Bq.y - 10, oM);
          else if (infX + infW + 16 + Math.max(measure(g, l1, 10.5, 'mono'), measure(g, l2, 10.5, 'mono')) <= Bq.x + Bq.w - 6) {
            V.text(g, l1, Bq.x + Bq.w - 6, infY, oM); V.text(g, l2, Bq.x + Bq.w - 6, infY + 14, oM);
          } else V.text(g, '┄ (1−p)² ≤ 1', Bq.x + Bq.w, Bq.y - 10, oM);
        }
        g.strokeStyle = C.a; g.lineWidth = 2.6; g.beginPath();
        for (let i = 1; i <= 200; i++) { const q = Math.pow(i / 200, 1.6); const yy = Y(-Math.log(q) * k); i > 1 ? g.lineTo(X(q), yy) : g.moveTo(X(q), yy); }
        g.stroke();
        V.text(g, infL, infX, infY, { size: 11, font: 'mono', color: C.muted });
        // marcador 1 bit
        V.dot(g, X(0.5), Y(Math.LN2 * k), 3, C.b);
        V.text(g, bitsU ? '1 bit' : 'ln 2 nat = 1 bit', X(0.5) + 6, Y(Math.LN2 * k) - 6, { size: 10.5, font: 'mono', color: C.b });
        // ponto atual + rosto
        const pc = p[0], L = -Math.log(pc) * k, qx = X(pc), qy = Y(L);
        V.line(g, qx, qy, qx, Bq.y + Bq.h, { color: V.alpha(C.a, 0.5), dash: [3, 3] });
        V.line(g, Bq.x, qy, qx, qy, { color: V.alpha(C.a, 0.5), dash: [3, 3] });
        V.glow(g, qx, qy, 16, C.a, 0.7); V.dot(g, qx, qy, 5.5, C.a);
        const surprise = V.clamp(-Math.log(pc) / 4, 0, 1);
        const fx = V.clamp(qx + 36, Bq.x + 20, Bq.x + Bq.w - 24), fy = V.clamp(qy - 34, Bq.y + 22, Bq.y + Bq.h - 30), fr = 17;
        V.dot(g, fx, fy, fr, V.alpha(C.d, 0.22)); g.strokeStyle = C.d; g.lineWidth = 1.8; g.beginPath(); g.arc(fx, fy, fr, 0, Math.PI * 2); g.stroke();
        const er = 1.8 + surprise * 2.6;
        V.dot(g, fx - 6, fy - 4, er, C.ink); V.dot(g, fx + 6, fy - 4, er, C.ink);
        V.line(g, fx - 10, fy - 10 - surprise * 4, fx - 3, fy - 11 - surprise * 5, { color: C.ink, lw: 1.3 });
        V.line(g, fx + 3, fy - 11 - surprise * 5, fx + 10, fy - 10 - surprise * 4, { color: C.ink, lw: 1.3 });
        if (surprise < 0.25) { g.beginPath(); g.arc(fx, fy + 3, 6, 0.15 * Math.PI, 0.85 * Math.PI); g.strokeStyle = C.ink; g.lineWidth = 1.5; g.stroke(); }
        else if (surprise < 0.55) V.line(g, fx - 5, fy + 7, fx + 5, fy + 7, { color: C.ink, lw: 1.5 });
        else { g.beginPath(); g.ellipse(fx, fy + 7, 3 + surprise * 2, 3 + surprise * 3, 0, 0, Math.PI * 2); g.fillStyle = C.ink; g.fill(); }
        const plab = num(L, 2) + (bitsU ? ' bits' : ' nats'), pw = pill(g, plab, 0, 0, { dry: true, size: 11.5, weight: 600 });
        const plx = qx - pw - 12 > Bq.x + 4 ? qx - pw - 12 : qx + 12, ply = qx - pw - 12 > Bq.x + 4 ? qy - 11 : qy + 8;
        pill(g, plab, plx, V.clamp(ply, Bq.y, Bq.y + Bq.h - 26), { size: 11.5, fill: C.panel, stroke: C.a, weight: 600 });
      });
    })();

    /* ================= Figura 4: a bolinha que desce ================= */
    (function fig4() {
      const f4 = $('#fig-descida');
      fitH(f4, 400, 540);
      const th = (25 * Math.PI) / 180, cs = Math.cos(th), sn = Math.sin(th), U0 = 0.35, V0 = -0.15, l1 = 0.8, l2 = 5;
      const DU = -0.83, DV = -0.7, DA = 0.5, DS = 0.06;
      // tigela anisotrópica girada (curvaturas 0,8 e 5) + uma cavidade gaussiana rasa = mínimo local
      const Lf = (u, v) => { const du = u - U0, dv = v - V0; const p = cs * du + sn * dv, q = -sn * du + cs * dv; return 0.5 * (l1 * p * p + l2 * q * q) - DA * Math.exp(-((u - DU) ** 2 + (v - DV) ** 2) / DS); };
      const Lmin = Lf(U0, V0);
      const L = (u, v) => Lf(u, v) - Lmin;
      const F = Field(L, { x0: -1.65, x1: 1.65, y0: -1.05, y1: 1.05 }, { L0: 0.06, Lmax: 9, bands: 14 });
      const S = { eta: 0.08, start: [1.35, 0.85], pos: null, trail: [], hist: [], steps: 0, state: 'run', wait: 0, acc: 0 };
      const reset = () => { S.pos = S.start.slice(); S.trail = [S.pos.slice()]; S.hist = [L(...S.pos)]; S.steps = 0; S.state = 'run'; S.wait = 0; S.acc = 0; };
      reset();
      const c4 = UI.controls(f4);
      relabel(UI.slider(c4, { label: 'rate η', min: 0.01, max: 0.45, step: 0.01, value: S.eta, fmt: (v) => num(v, 2), onInput: (v) => { S.eta = v; reset(); } }), 'rate ' + gk('η'));
      UI.button(c4, '↺ restart', () => { S.start = [1.35, 0.85]; reset(); });
      UI.button(c4, 'drop near the local minimum', () => { S.start = [-1.05, -0.35]; reset(); });
      const rS = UI.readout(c4, 'step', '0'), rL = UI.readout(c4, 'loss', '—');
      // limites 2/λmax: tigela global (λmax = 5) e cavidade (Hessiana no mínimo local: λmax ≈ 20,6)
      relabel(UI.readout(c4, 'stable in the global valley if η <', num(2 / l2, 2)), 'stable in the global valley if ' + gk('η') + ' &lt;');
      relabel(UI.readout(c4, 'in the pocket: η <', '≈ 0.10'), 'in the pocket: ' + gk('η') + ' &lt;');
      let lastS = -1;
      V.fig(f4, (g, w, h, t, dt, _s, H) => {
        const C = V.C();
        const narrow = w < 600;
        const R = narrow ? { x: 8, y: 8, w: w - 16, h: h - 128 } : { x: 10, y: 10, w: w - 20, h: h - 20 };
        F.draw(g, R);
        // clique: novo ponto de partida
        if (H.pointer.clicked && H.pointer.x > R.x && H.pointer.x < R.x + R.w && H.pointer.y > R.y && H.pointer.y < R.y + R.h) { S.start = F.fromPx(H.pointer.x, H.pointer.y, R); reset(); }
        // marcadores dos mínimos
        const gm = F.toPx(U0, V0, R); star(g, gm[0], gm[1], 7, C.d);
        V.text(g, 'global minimum', gm[0] + 10, gm[1] + 4, { size: 11, font: 'mono', color: C.ink });
        const lm = F.toPx(DU + 0.02, DV + 0.01, R);
        g.strokeStyle = V.alpha(C.ink, 0.6); g.lineWidth = 1.2; g.setLineDash([3, 3]); g.beginPath(); g.arc(lm[0], lm[1], 14, 0, Math.PI * 2); g.stroke(); g.setLineDash([]);
        V.text(g, 'local minimum', lm[0] + 18, lm[1] + 4, { size: 11, font: 'mono', color: C.ink });
        // passos
        if (S.state === 'run') {
          S.acc += dt * 11;
          while (S.acc >= 1 && S.state === 'run') {
            S.acc -= 1;
            const gr = grad2(L, S.pos[0], S.pos[1]);
            S.pos = [S.pos[0] - S.eta * gr[0], S.pos[1] - S.eta * gr[1]];
            S.steps++; S.trail.push(S.pos.slice()); S.hist.push(L(...S.pos));
            const Lc = S.hist[S.hist.length - 1];
            if (!isFinite(Lc) || Math.abs(S.pos[0]) > 4 || Math.abs(S.pos[1]) > 4 || Lc > 60) { S.state = 'div'; S.wait = 2.2; }
            else if (Math.hypot(gr[0], gr[1]) < 2e-3 || S.steps >= 260) { S.state = 'done'; S.wait = 2.4; }
          }
        } else { S.wait -= dt; if (S.wait <= 0) reset(); }
        if (S.steps !== lastS) { lastS = S.steps; rS.set(String(S.steps)); rL.set(num(S.hist[S.hist.length - 1], 3)); }
        // rastro
        g.save(); V.rr(g, R.x, R.y, R.w, R.h, 10); g.clip();
        g.strokeStyle = V.alpha(C.d, 0.9); g.lineWidth = 2; g.beginPath();
        S.trail.forEach((p, i) => { const q = F.toPx(p[0], p[1], R); i ? g.lineTo(q[0], q[1]) : g.moveTo(q[0], q[1]); });
        g.stroke();
        S.trail.forEach((p, i) => { if (i % 2 === 0) { const q = F.toPx(p[0], p[1], R); V.dot(g, q[0], q[1], 2, C.d); } });
        const b = F.toPx(S.pos[0], S.pos[1], R);
        if (S.state !== 'div') {
          const gr = grad2(L, S.pos[0], S.pos[1]); const gn = Math.hypot(gr[0], gr[1]);
          // a seta é o próximo passo de verdade, −η∇ℒ em pixels (com limites para continuar visível)
          const stepPx = S.eta * gn * F.dom.sc, len = V.clamp(stepPx, 18, 110), showA = stepPx > 5; // passo quase nulo: sem seta
          V.glow(g, b[0], b[1], 22, C.d, 0.8); V.dot(g, b[0], b[1], 7, C.d); V.dot(g, b[0] - 2, b[1] - 2, 2, '#ffffff');
          // desenhada por cima da bolinha: a ponta marca onde ela vai cair no próximo passo
          if (showA) V.arrow(g, b[0], b[1], b[0] - (gr[0] / gn) * len, b[1] + (gr[1] / gn) * len, { color: C.c, lw: 2.4, head: Math.min(9, len * 0.4) });
          if (showA) V.text(g, '−η∇ℒ', b[0] - (gr[0] / gn) * (len + 18), b[1] + (gr[1] / gn) * (len + 18) + 4, { size: 11.5, font: 'mono', color: C.c, align: 'center', weight: 600 });
        }
        g.restore();
        if (S.state === 'div') pill(g, 'diverged! η too large', R.x + R.w / 2 - 95, R.y + R.h / 2 - 14, { size: 13, fill: V.alpha(C.c, 0.3), stroke: C.c, weight: 700 });
        if (S.state === 'done') pill(g, S.steps >= 260 ? 'slow: 260 steps and still descending' : `stopped after ${S.steps} steps`, R.x + 12, R.y + R.h - 34, { size: 12, fill: C.panel, stroke: C.a });
        // curva de treino
        const I = narrow ? { x: 12, y: h - 112, w: w - 24, h: 102 } : { x: R.x + R.w - 214, y: R.y + R.h - 110, w: 204, h: 100 };
        V.box(g, I.x, I.y, I.w, I.h, { r: 10, fill: V.alpha(C.panel, 0.9), stroke: V.alpha(C.muted, 0.4), lw: 1 });
        V.text(g, 'TRAINING CURVE · ℒ × step', I.x + 10, I.y + 16, { size: 10.5, font: 'mono', color: C.muted });
        const hm = Math.max(1e-6, ...S.hist.slice(0, 1).concat(S.hist.map((x) => (isFinite(x) ? x : 0))).map((x) => Math.min(x, 60)));
        const n = Math.max(30, S.hist.length);
        g.strokeStyle = C.d; g.lineWidth = 1.8; g.beginPath();
        S.hist.forEach((v, i) => { const x = I.x + 10 + (i / (n - 1)) * (I.w - 20), y = I.y + I.h - 10 - (Math.min(v, hm) / hm) * (I.h - 34); i ? g.lineTo(x, y) : g.moveTo(x, y); });
        g.stroke();
        V.line(g, I.x + 10, I.y + I.h - 10, I.x + I.w - 10, I.y + I.h - 10, { color: V.alpha(C.muted, 0.5), lw: 1 });
      });
    })();

    /* ================= Figura 5: o leque de gradientes ================= */
    (function fig5() {
      const f5 = $('#fig-sgd');
      fitH(f5, 390, 620);
      const r = V.rng(5), N = 64;
      const XS = [], YS = [];
      for (let i = 0; i < N; i++) { const x = -1 + 2 * r(); XS.push(x); YS.push(1.2 * x + 0.4 + 0.45 * r.normal()); }
      const Lfull = (w, b) => { let s = 0; for (let i = 0; i < N; i++) { const e = w * XS[i] + b - YS[i]; s += e * e; } return s / N; };
      const gradIdx = (w, b, idx) => { let gw = 0, gb = 0; idx.forEach((i) => { const e = w * XS[i] + b - YS[i]; gw += 2 * e * XS[i]; gb += 2 * e; }); return [gw / idx.length, gb / idx.length]; };
      const all = Array.from({ length: N }, (_, i) => i);
      const rs = V.rng(77);
      const sample = (B) => (B >= N ? all : Array.from({ length: B }, () => rs.int(N)));
      let Lm = Infinity; for (let w = -1; w < 3; w += 0.01) for (let b = -1; b < 2; b += 0.01) Lm = Math.min(Lm, Lfull(w, b));
      const F = Field((w, b) => Lfull(w, b) - Lm, { x0: -1.4, x1: 3.3, y0: -1.45, y1: 2.15 }, { L0: 0.08, Lmax: 12, bands: 13 });
      const S = { B: 4, eta: 0.3, start: [-0.9, -1.15], pos: null, trail: [], steps: 0, seen: 0, fan: [], used: [0, 0], noise: [], acc: 0, wait: 0 };
      const BS = [1, 4, 16, 64];
      const refresh = () => {
        const [w, b] = S.pos;
        S.fan = Array.from({ length: 16 }, () => gradIdx(w, b, sample(S.B)));
        S.used = S.fan[0];
        const tg = gradIdx(w, b, all); S.true = tg;
        S.noise = BS.map((B) => { if (B >= N) return 0; let s = 0; for (let k = 0; k < 30; k++) { const gg = gradIdx(w, b, sample(B)); s += (gg[0] - tg[0]) ** 2 + (gg[1] - tg[1]) ** 2; } return Math.sqrt(s / 30); });
      };
      const reset = () => { S.pos = S.start.slice(); S.trail = [S.pos.slice()]; S.steps = 0; S.seen = 0; S.acc = 0; S.wait = 0; refresh(); };
      reset();
      const c5 = UI.controls(f5);
      UI.seg(c5, { label: 'Batch B', options: BS.map((b) => ({ v: b, label: b === N ? '64 (all)' : String(b) })), value: 4, onChange: (v) => { S.B = v; reset(); } });
      relabel(UI.slider(c5, { label: 'rate η', min: 0.05, max: 0.9, step: 0.05, value: S.eta, fmt: (v) => num(v, 2), onInput: (v) => { S.eta = v; } }), 'rate ' + gk('η'));
      UI.button(c5, '↺ restart', () => reset());
      const rS = UI.readout(c5, 'steps', '0'), rE = UI.readout(c5, 'examples seen', '0'), rL = UI.readout(c5, 'loss (64 pts)', '—');
      let lastS = -1;
      V.fig(f5, (g, w, h, t, dt) => {
        const C = V.C();
        const narrow = w < 600;
        const R = narrow ? { x: 8, y: 8, w: w - 16, h: h - 200 } : { x: 10, y: 10, w: w * 0.66 - 14, h: h - 20 };
        const Pn = narrow ? { x: 12, y: h - 182, w: w - 24, h: 172 } : { x: w * 0.66 + 8, y: 10, w: w * 0.34 - 18, h: h - 20 };
        F.draw(g, R);
        // passos
        if (S.wait > 0) { S.wait -= dt; if (S.wait <= 0) reset(); }
        else {
          S.acc += dt * 5;
          while (S.acc >= 1) {
            S.acc -= 1;
            S.pos = [S.pos[0] - S.eta * S.used[0], S.pos[1] - S.eta * S.used[1]];
            if (!isFinite(S.pos[0]) || Math.abs(S.pos[0]) > 20 || Math.abs(S.pos[1]) > 20) { S.pos = S.start.slice(); }
            S.steps++; S.seen += Math.min(S.B, N); S.trail.push(S.pos.slice());
            refresh();
            if (S.steps >= 55) { S.wait = 1.8; break; }
          }
        }
        if (S.steps !== lastS) { lastS = S.steps; rS.set(String(S.steps)); rE.set(fmtInt(S.seen)); rL.set(num(Lfull(...S.pos), 3)); }
        const om = F.toPx(1.2, 0.4, R);
        g.save(); V.rr(g, R.x, R.y, R.w, R.h, 10); g.clip();
        // rastro
        g.strokeStyle = V.alpha(C.d, 0.85); g.lineWidth = 1.6; g.beginPath();
        S.trail.forEach((p, i) => { const q = F.toPx(p[0], p[1], R); i ? g.lineTo(q[0], q[1]) : g.moveTo(q[0], q[1]); });
        g.stroke();
        S.trail.forEach((p) => { const q = F.toPx(p[0], p[1], R); V.dot(g, q[0], q[1], 1.8, C.d); });
        const b0 = F.toPx(S.pos[0], S.pos[1], R);
        const sc = F.dom.sc;
        const clampLen = (dx, dy, m) => { const l = Math.hypot(dx, dy); return l > m ? [dx * m / l, dy * m / l] : [dx, dy]; };
        S.fan.slice(1).forEach((gg) => { const d = clampLen(-S.eta * gg[0] * sc, S.eta * gg[1] * sc, 130); V.arrow(g, b0[0], b0[1], b0[0] + d[0], b0[1] + d[1], { color: V.alpha(C.ink, 0.35), lw: 1.1, head: 6 }); });
        const dt2 = clampLen(-S.eta * S.true[0] * sc, S.eta * S.true[1] * sc, 130);
        V.arrow(g, b0[0], b0[1], b0[0] + dt2[0], b0[1] + dt2[1], { color: C.ink, lw: 3, head: 10 });
        const du = clampLen(-S.eta * S.used[0] * sc, S.eta * S.used[1] * sc, 130);
        V.arrow(g, b0[0], b0[1], b0[0] + du[0], b0[1] + du[1], { color: C.c, lw: 2.4, head: 9 });
        V.glow(g, b0[0], b0[1], 18, C.d, 0.8); V.dot(g, b0[0], b0[1], 6, C.d);
        g.restore();
        star(g, om[0], om[1], 6.5, C.d);
        V.text(g, 'optimum', om[0] + 9, om[1] + 4, { size: 11, font: 'mono', color: C.ink });
        V.text(g, 'w →', R.x + R.w - 10, R.y + R.h - 10, { size: 11, font: 'mono', color: C.muted, align: 'right' });
        V.text(g, '↑ b', R.x + 10, R.y + 18, { size: 11, font: 'mono', color: C.muted });
        // painel de ruído
        V.box(g, Pn.x, Pn.y, Pn.w, Pn.h, { r: 12, fill: V.alpha(C.panel, 0.7), stroke: V.alpha(C.muted, 0.35), lw: 1 });
        V.text(g, 'NOISE × COST PER STEP', Pn.x + 12, Pn.y + 20, { size: 10.5, font: 'mono', color: C.muted, maxW: Pn.w - 24 });
        const nmx = Math.max(1e-6, ...S.noise);
        const bx = Pn.x + 12, bwid = Pn.w - 24, top = Pn.y + 34, rowH = Math.min(narrow ? 26 : 50, (Pn.h - (narrow ? 64 : 104)) / 4);
        const L0 = bx + 44, lw0 = bwid - 44, bh1 = Math.max(5, (rowH - 12) / 2);
        BS.forEach((B, i) => {
          const y = top + i * rowH, on = B === S.B;
          V.text(g, 'B=' + B, bx, y + rowH / 2 + 3, { size: 11, font: 'mono', color: on ? C.c : C.muted, weight: on ? 700 : 400 });
          const bl = lw0 * (S.noise[i] / nmx);
          V.box(g, L0, y + 3, lw0, bh1, { r: 3, fill: V.alpha(C.muted, 0.1) });
          if (bl > 1) V.box(g, L0, y + 3, bl, bh1, { r: 3, fill: on ? V.alpha(C.c, 0.9) : V.alpha(C.c, 0.4) });
          const theo = S.noise[0] / Math.sqrt(B) / nmx;
          if (B < N && B > 1) V.line(g, L0 + lw0 * theo, y + 1, L0 + lw0 * theo, y + bh1 + 5, { color: C.ink, lw: 1.5, dash: [2, 2] });
          V.box(g, L0, y + 5 + bh1, lw0, bh1, { r: 3, fill: V.alpha(C.muted, 0.1) });
          V.box(g, L0, y + 5 + bh1, Math.max(2, lw0 * (B / N)), bh1, { r: 3, fill: on ? V.alpha(C.d, 0.95) : V.alpha(C.d, 0.4) });
          if (B === N) V.text(g, 'noise 0', L0 + 4, y + 3 + bh1 - 1, { size: 9.5, font: 'mono', color: C.muted });
        });
        const ly = top + 4 * rowH + 12;
        V.box(g, bx, ly - 8, 10, 8, { r: 2, fill: C.c }); V.text(g, 'gradient noise (┆ = 1/√B)', bx + 14, ly, { size: 10.5, color: C.muted, maxW: bwid - 14 });
        V.box(g, bx, ly + 8, 10, 8, { r: 2, fill: C.d }); V.text(g, 'cost: examples per step', bx + 14, ly + 16, { size: 10.5, color: C.muted, maxW: bwid - 14 });
        if (!narrow) {
          const ly2 = ly + 40;
          V.line(g, bx, ly2 - 4, bx + 16, ly2 - 4, { color: C.ink, lw: 3 }); V.text(g, 'exact gradient', bx + 22, ly2, { size: 10.5, color: C.muted });
          V.line(g, bx, ly2 + 14, bx + 16, ly2 + 14, { color: C.c, lw: 2.4 }); V.text(g, 'batch gradient (used)', bx + 22, ly2 + 18, { size: 10.5, color: C.muted });
        }
      });
    })();

    /* ================= Figura 6: corrida de otimizadores ================= */
    (function fig6() {
      const f6 = $('#fig-corrida');
      fitH(f6, 390, 460);
      const mk = (th) => { const c = Math.cos(th), s = Math.sin(th); return (x, y) => { const p = c * x + s * y, q = -s * x + c * y; return 0.5 * (0.25 * p * p + 40 * q * q); }; };
      const LS = { eixos: mk(0), girada: mk(Math.PI / 4) };
      const S = { land: 'eixos', mult: 1, show: { gd: true, mom: true, adam: true }, runners: null, steps: 0, wait: 0, acc: 0 };
      const OPT = [
        { id: 'gd', name: 'plain descent', lr: 0.045, col: 'd' },
        { id: 'mom', name: 'momentum β=0.9', lr: 0.006, col: 'b' },
        { id: 'adam', name: 'Adam', lr: 0.05, col: 'c' },
      ];
      const BOXW = { x0: -2.2, x1: 0.9, y0: -1.05, y1: 1.05 }, BOXN = { x0: -2.05, x1: 0.35, y0: -1.1, y1: 1.1 }; // no celular, recorte mais estreito e alto
      const FO = { L0: 0.05, Lmax: 40, bands: 14 };
      const fields = { eixos: Field(LS.eixos, BOXW, FO), girada: Field(LS.girada, BOXW, FO) };
      const fieldsN = { eixos: Field(LS.eixos, BOXN, FO), girada: Field(LS.girada, BOXN, FO) };
      const start = [-1.8, 0.8];
      const reset = () => { S.runners = OPT.map((o) => ({ o, x: start[0], y: start[1], v: [0, 0], m: [0, 0], s: [0, 0], t: 0, trail: [start.slice()], hit: -1, div: false })); S.steps = 0; S.wait = 0; S.acc = 0; };
      reset();
      const c6 = UI.controls(f6);
      UI.seg(c6, { label: 'Ravine', options: [{ v: 'eixos', label: 'axis-aligned' }, { v: 'girada', label: 'rotated 45°' }], value: 'eixos', onChange: (v) => { S.land = v; reset(); } });
      UI.slider(c6, { label: 'rates ×', min: 0.25, max: 1.6, step: 0.05, value: 1, fmt: (v) => num(v, 2) + '×', onInput: (v) => { S.mult = v; reset(); } });
      OPT.forEach((o) => UI.toggle(c6, { label: o.name, value: true, onChange: (v) => { S.show[o.id] = v; } }));
      UI.button(c6, '↺ race again', () => reset());
      function stepR(R, f) {
        if (R.div) return;
        const gr = grad2(f, R.x, R.y), lr = R.o.lr * S.mult;
        if (R.o.id === 'gd') { R.x -= lr * gr[0]; R.y -= lr * gr[1]; }
        else if (R.o.id === 'mom') { R.v = [0.9 * R.v[0] + gr[0], 0.9 * R.v[1] + gr[1]]; R.x -= lr * R.v[0]; R.y -= lr * R.v[1]; }
        else { R.t++; const b1 = 0.9, b2 = 0.999; R.m = [b1 * R.m[0] + (1 - b1) * gr[0], b1 * R.m[1] + (1 - b1) * gr[1]]; R.s = [b2 * R.s[0] + (1 - b2) * gr[0] ** 2, b2 * R.s[1] + (1 - b2) * gr[1] ** 2]; const c1 = 1 - b1 ** R.t, c2 = 1 - b2 ** R.t; R.x -= (lr * (R.m[0] / c1)) / (Math.sqrt(R.s[0] / c2) + 1e-8); R.y -= (lr * (R.m[1] / c1)) / (Math.sqrt(R.s[1] / c2) + 1e-8); }
        if (!isFinite(R.x) || !isFinite(R.y) || Math.abs(R.x) > 8 || Math.abs(R.y) > 8) { R.div = true; return; }
        R.trail.push([R.x, R.y]);
        if (R.hit < 0 && f(R.x, R.y) < 1e-3) R.hit = S.steps;
      }
      V.fig(f6, (g, w, h, t, dt) => {
        const C = V.C();
        const narrow = w < 600;
        const f = LS[S.land], F = (narrow ? fieldsN : fields)[S.land];
        const R = narrow ? { x: 8, y: 8, w: w - 16, h: h - 118 } : { x: 10, y: 10, w: w - 20, h: h - 20 };
        F.draw(g, R);
        if (S.wait > 0) { S.wait -= dt; if (S.wait <= 0) reset(); }
        else {
          S.acc += dt * 30;
          while (S.acc >= 1) {
            S.acc -= 1; S.steps++;
            S.runners.forEach((Rn) => stepR(Rn, f));
            const done = S.runners.every((Rn) => Rn.div || Rn.hit >= 0);
            if (done || S.steps >= 420) { S.wait = 2.6; break; }
          }
        }
        g.save(); V.rr(g, R.x, R.y, R.w, R.h, 10); g.clip();
        const mn = F.toPx(0, 0, R); star(g, mn[0], mn[1], 7, C.ink);
        S.runners.forEach((Rn) => {
          if (!S.show[Rn.o.id]) return;
          const col = C[Rn.o.col];
          g.strokeStyle = V.alpha(col, 0.9); g.lineWidth = 2; g.beginPath();
          Rn.trail.forEach((p, i) => { const q = F.toPx(p[0], p[1], R); i ? g.lineTo(q[0], q[1]) : g.moveTo(q[0], q[1]); });
          g.stroke();
          const last = Rn.trail[Rn.trail.length - 1], q = F.toPx(last[0], last[1], R);
          V.glow(g, q[0], q[1], 18, col, 0.8); V.dot(g, q[0], q[1], 6, col);
        });
        const s0 = F.toPx(start[0], start[1], R); V.text(g, 'start', s0[0] - 8, s0[1] - 12, { size: 11, font: 'mono', color: C.ink });
        g.restore();
        // placar
        const Pw = narrow ? w - 24 : 250, Px = narrow ? 12 : R.x + R.w - Pw - 12, Py = narrow ? h - 104 : R.y + 12, Ph = 94;
        V.box(g, Px, Py, Pw, Ph, { r: 10, fill: V.alpha(C.panel, 0.9), stroke: V.alpha(C.muted, 0.4), lw: 1 });
        V.text(g, `STEP ${S.steps} · goal ℒ < 10⁻³`, Px + 10, Py + 17, { size: 10.5, font: 'mono', color: C.muted });
        S.runners.forEach((Rn, i) => {
          const y = Py + 38 + i * 21, col = C[Rn.o.col];
          V.dot(g, Px + 16, y - 4, 5, S.show[Rn.o.id] ? col : V.alpha(col, 0.3));
          V.text(g, Rn.o.name, Px + 28, y, { size: 11.5, color: C.ink });
          const st = Rn.div ? 'diverged' : Rn.hit >= 0 ? `✓ ${Rn.hit} steps` : 'ℒ = ' + num(f(Rn.x, Rn.y), f(Rn.x, Rn.y) < 0.1 ? 3 : 2);
          V.text(g, st, Px + Pw - 10, y, { size: 11.5, font: 'mono', color: Rn.div ? C.c : Rn.hit >= 0 ? C.a : C.muted, align: 'right' });
        });
      });
    })();

    /* ================= Figura 7: Muon / Newton–Schulz ================= */
    (function fig7() {
      const f7 = $('#fig-muon');
      fitH(f7, 430, 710);
      const n = 5, r = V.rng(11);
      const mm = (A, B) => A.map((row) => B[0].map((_, j) => row.reduce((s, a, k) => s + a * B[k][j], 0)));
      const tr = (A) => A[0].map((_, j) => A.map((row) => row[j]));
      const orthon = (A) => { const Q = []; for (let j = 0; j < n; j++) { let v = A.map((row) => row[j]); Q.forEach((q) => { const d = q.reduce((s, x, i) => s + x * v[i], 0); v = v.map((x, i) => x - d * q[i]); }); const nr = Math.hypot(...v); Q.push(v.map((x) => x / nr)); } return tr(Q); };
      const U = orthon(Array.from({ length: n }, () => Array.from({ length: n }, () => r.normal())));
      const Vv = orthon(Array.from({ length: n }, () => Array.from({ length: n }, () => r.normal())));
      const SIG = [3.0, 1.2, 0.4, 0.1, 0.012];
      const G = mm(mm(U, SIG.map((s, i) => SIG.map((_, j) => (i === j ? s : 0)))), tr(Vv));
      const fro = Math.sqrt(G.flat().reduce((s, x) => s + x * x, 0));
      const X0 = G.map((row) => row.map((x) => x / fro));
      const s0 = SIG.map((s) => s / fro);
      const MU = [3.4445, -4.775, 2.0315], PO = [2, -1.5, 0.5];
      const SCH = { muon: () => MU, ds: (k) => (k < 8 ? MU : PO), pol: () => PO };
      const poly = (c, x) => c[0] * x + c[1] * x ** 3 + c[2] * x ** 5;
      const runs = {};
      Object.keys(SCH).forEach((key) => {
        const Xs = [X0], ss = [s0];
        for (let k = 0; k < 10; k++) {
          const c = SCH[key](k), X = Xs[k], A = mm(X, tr(X)), AX = mm(A, X), AAX = mm(A, AX);
          Xs.push(X.map((row, i) => row.map((x, j) => c[0] * x + c[1] * AX[i][j] + c[2] * AAX[i][j])));
          ss.push(ss[k].map((x) => poly(c, x)));
        }
        runs[key] = { Xs, ss };
      });
      const S = { k: 5, sched: 'muon', disp: s0.slice(), from: s0.slice(), t0: 0, anim: false };
      const c7 = UI.controls(f7);
      const sk = relabel(UI.slider(c7, { label: 'iterations k', min: 0, max: 10, step: 1, value: S.k, fmt: (v) => String(v), onInput: (v) => { S.from = S.disp.slice(); S.k = v; S.t0 = H7.t; } }), 'iterations ' + gk('k'));
      UI.seg(c7, { label: 'Coefficients', options: [{ v: 'muon', label: 'Muon' }, { v: 'ds', label: 'DeepSeek-V4 (8 + 2)' }, { v: 'pol', label: 'polish only' }], value: 'muon', onChange: (v) => { S.from = S.disp.slice(); S.sched = v; S.t0 = H7.t; } });
      UI.button(c7, '▶ iterate from zero', () => { S.anim = true; S.from = S.disp.slice(); S.k = 0; sk.set(0); });
      const rMin = relabel(UI.readout(c7, 'σ min', '—'), gk('σ') + ' min'), rMax = relabel(UI.readout(c7, 'σ max', '—'), gk('σ') + ' max'), rDev = relabel(UI.readout(c7, 'deviation', '—'), '‖' + gk('XXᵀ − I') + '‖');
      V.every(750, () => { if (!S.anim) return; if (S.k >= 10) { S.anim = false; return; } sk.set(S.k + 1); });
      let lastKey = '';
      const H7 = V.fig(f7, (g, w, h, t) => {
        const C = V.C();
        const run = runs[S.sched], target = run.ss[S.k], X = run.Xs[S.k];
        const u = V.ease.inOut(V.clamp((t - S.t0) / 0.6, 0, 1));
        S.disp = target.map((x, i) => V.lerp(S.from[i], x, u));
        const key = S.sched + S.k;
        if (key !== lastKey) {
          lastKey = key;
          rMin.set(num(Math.min(...target), 3)); rMax.set(num(Math.max(...target), 3));
          const A = mm(X, tr(X)); let dv = 0; A.forEach((row, i) => row.forEach((x, j) => { dv += (x - (i === j ? 1 : 0)) ** 2; })); rDev.set(num(Math.sqrt(dv), 3));
        }
        const wide = w >= 760;
        const Am = wide ? { x: 16, y: 30, w: w * 0.54 - 16, h: h * 0.6 - 30 } : { x: 10, y: 26, w: w - 20, h: 230 };
        const Pp = wide ? { x: w * 0.6 + 16, y: 32, w: w * 0.4 - 40, h: h * 0.6 - 56 } : { x: 44, y: 276, w: w - 64, h: 196 };
        const Nl = wide ? { x: 26, y: h * 0.66, w: w * 0.6 - 40, h: h * 0.34 - 16 } : { x: 18, y: 532, w: w - 150, h: 160 };
        const El = wide ? { x: w * 0.64, y: h * 0.66, w: w * 0.36 - 20, h: h * 0.34 - 12 } : { x: w - 126, y: 548, w: 116, h: 150 };
        // ---- matrizes
        const gapMin = wide ? 80 : 22;
        const cell = Math.max(14, Math.min(40, (Am.w - gapMin - 16) / (2 * n) - 1.6, (Am.h - 50) / n));
        const mw = cell * n + 2 * (n - 1);
        const gapM = Math.min(Math.max(gapMin, Am.w - 2 * mw - 16), wide ? 150 : 60);
        const xA = Am.x + (Am.w - (2 * mw + gapM)) / 2, xB = xA + mw + gapM, yM = Am.y + 20;
        const mxG = Math.max(...G.flat().map(Math.abs));
        const d = cell >= 36 ? 2 : 1;
        V.text(g, 'M (momentum)', xA + mw / 2, yM - 10, { size: 12, font: 'mono', color: C.ink, align: 'center' });
        V.matrix(g, { x: xA, y: yM, cell, data: G, cmap: 'div', min: -mxG, max: mxG, values: true, fmt: (v) => num(v, d) });
        V.text(g, S.k === 0 ? 'X₀ = M/‖M‖' : 'X' + sub(S.k) + ' (after ' + S.k + ' iter.)', xB + mw / 2, yM - 10, { size: 12, font: 'mono', color: C.ink, align: 'center' });
        V.matrix(g, { x: xB, y: yM, cell, data: X, cmap: 'div', min: -1, max: 1, values: true, fmt: (v) => num(v, d) });
        const ay = yM + mw / 2, ax0 = xA + mw + 6, ax1 = xB - 6;
        V.arrow(g, ax0, ay, ax1, ay, { color: C.a, lw: 2.2, head: 9 });
        const pulse = (t * 0.8) % 1; V.dot(g, V.lerp(ax0, ax1 - 8, pulse), ay, 3, C.a);
        if (gapM >= 56) {
          V.text(g, 'NS ×' + S.k, (ax0 + ax1) / 2, ay - 9, { size: 11.5, font: 'mono', color: C.a, align: 'center' });
          V.text(g, 'matmuls only', (ax0 + ax1) / 2, ay + 19, { size: 10.5, font: 'mono', color: C.muted, align: 'center' });
        } else V.text(g, 'Newton–Schulz ×' + S.k + ' (matrix multiplications only)', Am.x + Am.w / 2, yM + mw + 20, { size: 11, font: 'mono', color: C.a, align: 'center', maxW: Am.w });
        // ---- polinômio
        const c = SCH[S.sched](Math.max(0, S.k - 1));
        const PX = (x) => Pp.x + (x / 1.3) * Pp.w, PY = (y) => Pp.y + Pp.h - (y / 1.3) * Pp.h;
        g.strokeStyle = C.grid; g.lineWidth = 1; g.beginPath();
        [0, 0.5, 1].forEach((v) => { g.moveTo(PX(v), Pp.y); g.lineTo(PX(v), Pp.y + Pp.h); g.moveTo(Pp.x, PY(v)); g.lineTo(Pp.x + Pp.w, PY(v)); });
        g.stroke();
        [0, 0.5, 1].forEach((v) => { V.text(g, num(v, 1), PX(v), Pp.y + Pp.h + 14, { size: 10.5, font: 'mono', color: C.muted, align: 'center' }); V.text(g, num(v, 1), Pp.x - 6, PY(v) + 4, { size: 10.5, font: 'mono', color: C.muted, align: 'right' }); });
        V.line(g, PX(0), PY(0), PX(1.3), PY(1.3), { color: V.alpha(C.muted, 0.6), dash: [4, 4] });
        V.line(g, Pp.x, PY(1), Pp.x + Pp.w, PY(1), { color: V.alpha(C.a, 0.5), lw: 1 });
        g.save(); g.beginPath(); g.rect(Pp.x, Pp.y, Pp.w, Pp.h); g.clip();
        g.strokeStyle = c === MU ? C.b : C.d; g.lineWidth = 2.6; g.beginPath();
        for (let i = 0; i <= 80; i++) { const x = (i / 80) * 1.3, y = poly(c, x); i ? g.lineTo(PX(x), PY(y)) : g.moveTo(PX(x), PY(y)); }
        g.stroke();
        g.restore();
        const cl = c === MU ? 'Muon: 3.4445σ − 4.775σ³ + 2.0315σ⁵' : 'polish: 2σ − 1.5σ³ + 0.5σ⁵';
        V.text(g, cl, Pp.x, Pp.y - 12, { size: 11, font: 'mono', color: c === MU ? C.b : C.d, maxW: Pp.w + 30 });
        V.text(g, 'σ before → σ after', Pp.x + Pp.w, Pp.y + Pp.h + 28, { size: 10.5, color: C.muted, align: 'right' });
        if (S.k > 0) {
          const prev = run.ss[S.k - 1];
          prev.forEach((x, i) => { const xx = Math.min(x, 1.3), yy = Math.min(poly(c, x), 1.3); const col = C.series[i % 6]; V.line(g, PX(xx), PY(0), PX(xx), PY(yy), { color: V.alpha(col, 0.5), lw: 1, dash: [2, 3] }); V.dot(g, PX(xx), PY(yy), 4.5, col); });
        }
        // ---- reta numérica dos valores singulares
        V.text(g, wide ? 'SINGULAR VALUES OF X' + sub(S.k) + ' (faded: those of X₀)' : 'SINGULAR VALUES OF X' + sub(S.k), Nl.x, Nl.y, { size: 10.5, font: 'mono', color: C.muted });
        const NX = (x) => Nl.x + (Math.min(x, 1.3) / 1.3) * Nl.w, ny = Nl.y + Nl.h * 0.55;
        V.line(g, Nl.x, ny, Nl.x + Nl.w, ny, { color: V.alpha(C.muted, 0.7), lw: 1.4 });
        [0, 0.25, 0.5, 0.75, 1, 1.25].forEach((v) => { V.line(g, NX(v), ny - 4, NX(v), ny + 4, { color: C.muted, lw: 1 }); V.text(g, num(v, 2), NX(v), ny + 18, { size: 10, font: 'mono', color: C.muted, align: 'center' }); });
        g.fillStyle = V.alpha(C.a, 0.12); g.fillRect(NX(0.7), ny - 46, NX(1.2) - NX(0.7), 52);
        V.line(g, NX(1), ny - 46, NX(1), ny + 6, { color: V.alpha(C.a, 0.6), lw: 1, dash: [3, 3] });
        V.text(g, '1 = orthogonal', NX(1), ny - 52, { size: 10.5, font: 'mono', color: C.a, align: 'center' });
        V.text(g, '0.7–1.2 band', NX(0.95), ny + 36, { size: 10, font: 'mono', color: C.muted, align: 'center' });
        s0.forEach((x, i) => V.dot(g, NX(x), ny, 3, V.alpha(C.series[i % 6], 0.45)));
        const spread = Math.max(...S.disp) - Math.min(...S.disp);
        S.disp.forEach((x, i) => {
          const col = C.series[i % 6], yy = ny - 10 - i * 7;
          V.glow(g, NX(x), yy, 11, col, 0.6); V.dot(g, NX(x), yy, 4.8, col);
          if (spread > 0.12) V.text(g, 'σ' + (i + 1), NX(x) + 7, yy + 3.5, { size: 10, font: 'mono', color: col });
        });
        if (spread <= 0.12) V.text(g, 'all ≈ ' + num(mean(S.disp), 3), NX(Math.min(1.25, Math.max(...S.disp))) + 10, ny - 20, { size: 10.5, font: 'mono', color: C.ink });
        // ---- elipses
        const eR = Math.min(El.w * 0.22, El.h * 0.3);
        const drawE = (cx, cy, a, b, col, lab, ratio) => {
          g.strokeStyle = V.alpha(C.muted, 0.5); g.lineWidth = 1; g.setLineDash([3, 3]); g.beginPath(); g.arc(cx, cy, eR, 0, Math.PI * 2); g.stroke(); g.setLineDash([]);
          g.beginPath(); g.ellipse(cx, cy, Math.max(1.5, a * eR), Math.max(1.5, b * eR), -0.4, 0, Math.PI * 2); g.fillStyle = V.alpha(col, 0.22); g.fill(); g.strokeStyle = col; g.lineWidth = 2; g.stroke();
          V.text(g, lab, cx, cy + eR + 16, { size: 10.5, font: 'mono', color: C.muted, align: 'center' });
          // razão eixo maior / eixo menor (1 = círculo)
          V.text(g, El.w < 200 ? num(ratio, 1) + ':1' : 'ratio ' + num(ratio, 2), cx, cy + eR + 30, { size: 10.5, font: 'mono', color: ratio < 1.05 ? C.a : col, align: 'center', weight: 600 });
        };
        const rat = (a, b) => Math.max(a, b) / Math.max(1e-9, Math.min(a, b));
        const ex1 = El.x + El.w * 0.25, ex2 = El.x + El.w * 0.75, ey = El.y + El.h * 0.4;
        drawE(ex1, ey, Math.min(1.3, s0[0]), Math.min(1.3, s0[1]), C.d, 'before', rat(s0[0], s0[1]));
        drawE(ex2, ey, Math.min(1.3, S.disp[0]), Math.min(1.3, S.disp[1]), C.a, 'k = ' + S.k, rat(target[0], target[1]));
        if (ex2 - ex1 - 2 * eR > 22) V.text(g, '→', (ex1 + ex2) / 2, ey + 4, { size: 13, color: C.muted, align: 'center' });
        if (El.w > 200) V.text(g, 'directions 1 and 2', El.x + El.w / 2, El.y + 2, { size: 10.5, font: 'mono', color: C.muted, align: 'center' });
      });
    })();

    /* ================= Figura 8: decorar ou generalizar ================= */
    (function fig8() {
      const f8 = $('#fig-overfit');
      fitH(f8, 380, 620);
      const fstar = (x) => 0.8 * Math.sin(2.6 * x) + 0.25 * x;
      const leg = (d, x) => { const P = [1, x]; for (let k = 1; k < d; k++) P.push(((2 * k + 1) * x * P[k] - k * P[k - 1]) / (k + 1)); return P.slice(0, d + 1); };
      const solve = (A, b) => { const m = b.length; A = A.map((row, i) => [...row, b[i]]); for (let i = 0; i < m; i++) { let p = i; for (let j = i + 1; j < m; j++) if (Math.abs(A[j][i]) > Math.abs(A[p][i])) p = j; [A[i], A[p]] = [A[p], A[i]]; for (let j = i + 1; j < m; j++) { const f = A[j][i] / A[i][i]; for (let k = i; k <= m; k++) A[j][k] -= f * A[i][k]; } } const x = Array(m).fill(0); for (let i = m - 1; i >= 0; i--) { let s = A[i][m]; for (let k = i + 1; k < m; k++) s -= A[i][k] * x[k]; x[i] = s / A[i][i]; } return x; };
      const fit = (xs, ys, d, lam) => { const m = d + 1; const A = Array.from({ length: m }, () => Array(m).fill(0)), b = Array(m).fill(0); xs.forEach((x, i) => { const P = leg(d, x); for (let a = 0; a < m; a++) { b[a] += P[a] * ys[i]; for (let c = 0; c < m; c++) A[a][c] += P[a] * P[c]; } }); for (let a = 0; a < m; a++) A[a][a] += lam + 1e-10; return solve(A, b); };
      const ev = (c, x) => { const P = leg(c.length - 1, x); return c.reduce((s, v, i) => s + v * P[i], 0); };
      const mse = (c, xs, ys) => xs.reduce((s, x, i) => s + (ev(c, x) - ys[i]) ** 2, 0) / xs.length;
      const data = (N, seed) => { const r = V.rng(seed); const xs = [], ys = []; for (let i = 0; i < N; i++) { const x = -1 + (2 * (i + 0.5)) / N + ((r() - 0.5) * 1.6) / N; xs.push(x); ys.push(fstar(x) + 0.2 * r.normal()); } return [xs, ys]; };
      const VAL = data(60, 99);
      const TR = { 10: data(10, 71), 30: data(30, 211), 100: data(100, 701) };
      const S = { d: 9, N: 10, reg: false, curve: null, target: null, t0: 0 };
      const SX = Array.from({ length: 121 }, (_, i) => -1 + (2 * i) / 120);
      let cache = null;
      const compute = () => {
        const [tx, ty] = TR[S.N], lam = S.reg ? 0.01 * S.N : 0;
        const cs = []; const errT = [], errV = [];
        for (let d = 0; d <= 9; d++) { const c = fit(tx, ty, d, lam); cs.push(c); errT.push(Math.max(1e-4, mse(c, tx, ty))); errV.push(mse(c, VAL[0], VAL[1])); }
        cache = { cs, errT, errV };
        const newC = SX.map((x) => ev(cs[S.d], x));
        S.from = S.curve || newC; S.target = newC; S.t0 = H8 ? H8.t : 0;
        rT.set(num(errT[S.d], 3)); rV.set(num(errV[S.d], 3));
      };
      const c8 = UI.controls(f8);
      relabel(UI.slider(c8, { label: 'degree d', min: 0, max: 9, step: 1, value: S.d, fmt: (v) => v + ' (' + (v + 1) + ' knobs)', onInput: (v) => { S.d = v; compute(); } }), 'degree ' + gk('d'));
      UI.seg(c8, { label: 'Training points', options: [10, 30, 100], value: 10, onChange: (v) => { S.N = v; compute(); } });
      UI.toggle(c8, { label: 'regularization (weight decay)', value: false, onChange: (v) => { S.reg = v; compute(); } });
      const rT = UI.readout(c8, 'training error', '—'), rV = UI.readout(c8, 'validation error', '—');
      let H8 = null;
      compute();
      H8 = V.fig(f8, (g, w, h, t) => {
        const C = V.C();
        const wide = w >= 600;
        const u = V.ease.inOut(V.clamp((t - S.t0) / 0.5, 0, 1));
        S.curve = S.target.map((y, i) => V.lerp(S.from[i], y, u));
        const A = wide ? { x: 24, y: 18, w: w * 0.56 - 24, h: h - 60 } : { x: 14, y: 14, w: w - 28, h: h * 0.52 - 44 };
        const B = wide ? { x: w * 0.62 + 20, y: 30, w: w * 0.38 - 40, h: h - 76 } : { x: 50, y: h * 0.56 + 18, w: w - 70, h: h * 0.44 - 70 };
        const Y0 = -1.7, Y1 = 1.7;
        const X = (x) => A.x + ((x + 1.05) / 2.1) * A.w, Y = (y) => A.y + A.h - ((y - Y0) / (Y1 - Y0)) * A.h;
        V.box(g, A.x, A.y, A.w, A.h, { r: 8, fill: V.alpha(C.muted, 0.04), stroke: V.alpha(C.muted, 0.25), lw: 1 });
        V.line(g, A.x, Y(0), A.x + A.w, Y(0), { color: C.grid, lw: 1 });
        g.save(); g.beginPath(); g.rect(A.x, A.y, A.w, A.h); g.clip();
        g.setLineDash([5, 5]); g.strokeStyle = V.alpha(C.ink, 0.45); g.lineWidth = 1.4; g.beginPath();
        SX.forEach((x, i) => { i ? g.lineTo(X(x), Y(fstar(x))) : g.moveTo(X(x), Y(fstar(x))); }); g.stroke(); g.setLineDash([]);
        g.strokeStyle = C.c; g.lineWidth = 3; g.beginPath();
        S.curve.forEach((y, i) => { const yy = Y(V.clamp(y, Y0 - 1, Y1 + 1)); i ? g.lineTo(X(SX[i]), yy) : g.moveTo(X(SX[i]), yy); }); g.stroke();
        g.restore();
        VAL[0].forEach((x, i) => { g.strokeStyle = V.alpha(C.b, 0.8); g.lineWidth = 1.3; g.beginPath(); g.arc(X(x), Y(VAL[1][i]), 3.2, 0, Math.PI * 2); g.stroke(); });
        const [tx, ty] = TR[S.N];
        tx.forEach((x, i) => { const bob = Math.sin(t * 2 + i) * 0.6; V.dot(g, X(x), Y(ty[i]) + bob, S.N > 30 ? 3 : 4.5, C.a); });
        { const ly = A.y + A.h + 18; let x = A.x; V.dot(g, x + 5, ly - 4, 4, C.a); V.text(g, 'training', x + 13, ly, { size: 11, color: C.muted }); x += 13 + measure(g, 'training', 11) + 14;
          g.strokeStyle = C.b; g.lineWidth = 1.3; g.beginPath(); g.arc(x + 5, ly - 4, 3.5, 0, Math.PI * 2); g.stroke(); V.text(g, 'validation', x + 13, ly, { size: 11, color: C.muted }); x += 13 + measure(g, 'validation', 11) + 14;
          V.line(g, x, ly - 4, x + 16, ly - 4, { color: V.alpha(C.ink, 0.5), dash: [4, 3] }); V.text(g, 'true', x + 21, ly, { size: 11, color: C.muted }); x += 21 + measure(g, 'true', 11) + 14;
          if (x + 90 < A.x + A.w) { V.line(g, x, ly - 4, x + 16, ly - 4, { color: C.c, lw: 3 }); V.text(g, 'model', x + 21, ly, { size: 11, color: C.muted }); } }
        // curvas de erro
        const { errT, errV } = cache;
        const lo = Math.log10(5e-4), hi = Math.log10(3);
        const EX = (d) => B.x + (d / 9) * B.w, EY = (e) => B.y + B.h - ((Math.log10(V.clamp(e, 5e-4, 3)) - lo) / (hi - lo)) * B.h;
        g.strokeStyle = C.grid; g.lineWidth = 1; g.beginPath();
        [1e-3, 1e-2, 1e-1, 1].forEach((e) => { g.moveTo(B.x, EY(e)); g.lineTo(B.x + B.w, EY(e)); }); g.stroke();
        [1e-3, 1e-2, 1e-1, 1].forEach((e) => V.text(g, e >= 1 ? '1' : e.toString(), B.x - 6, EY(e) + 4, { size: 10, font: 'mono', color: C.muted, align: 'right' }));
        for (let d = 0; d <= 9; d += wide ? 1 : 3) V.text(g, String(d), EX(d), B.y + B.h + 14, { size: 10.5, font: 'mono', color: C.muted, align: 'center' });
        V.text(g, 'degree →', B.x + B.w, B.y + B.h + 30, { size: 11, color: C.muted, align: 'right' });
        V.text(g, 'ERROR (MSE, log scale)', B.x, B.y - 12, { size: 10.5, font: 'mono', color: C.muted });
        V.line(g, B.x, EY(0.04), B.x + B.w, EY(0.04), { color: V.alpha(C.d, 0.8), lw: 1.2, dash: [4, 4] });
        V.text(g, 'noise σ² = 0.04', B.x + B.w, EY(0.04) - 5, { size: 10.5, font: 'mono', color: C.d, align: 'right' });
        const plot = (arr, col) => { g.strokeStyle = col; g.lineWidth = 2.2; g.beginPath(); arr.forEach((e, d) => { d ? g.lineTo(EX(d), EY(e)) : g.moveTo(EX(d), EY(e)); }); g.stroke(); arr.forEach((e, d) => V.dot(g, EX(d), EY(e), 3, col)); };
        plot(errT, C.a); plot(errV, C.b);
        let best = 0; errV.forEach((e, d) => { if (e < errV[best]) best = d; });
        star(g, EX(best), EY(errV[best]) - 12, 6 + Math.sin(t * 3) * 0.6, C.d);
        V.line(g, EX(S.d), B.y, EX(S.d), B.y + B.h, { color: V.alpha(C.c, 0.8), lw: 1.5 });
        V.dot(g, EX(S.d), EY(errT[S.d]), 5, C.a); V.dot(g, EX(S.d), EY(errV[S.d]), 5, C.b);
        V.text(g, 'training', EX(0) + 4, EY(errT[0]) + 16, { size: 11, font: 'mono', color: C.a });
        V.text(g, 'validation', EX(9) - 4, EY(errV[9]) - 8, { size: 11, font: 'mono', color: C.b, align: 'right' });
      });
    })();

    /* ================= Figura 9: a mesa de mixagem do DSpark ================= */
    (function fig9() {
      const f9 = $('#fig-dspark');
      fitH(f9, 440, 590, 640);
      const Pt = [[0.7, 0.15, 0.08, 0.05, 0.02], [0.55, 0.25, 0.1, 0.06, 0.04], [0.45, 0.3, 0.12, 0.08, 0.05], [0.4, 0.25, 0.2, 0.1, 0.05], [0.35, 0.3, 0.15, 0.12, 0.08]];
      const Dr = [[0.4, 0.3, 0.15, 0.1, 0.05], [0.25, 0.4, 0.15, 0.12, 0.08], [0.2, 0.25, 0.3, 0.15, 0.1], [0.15, 0.2, 0.25, 0.25, 0.15], [0.12, 0.18, 0.2, 0.25, 0.25]];
      const DEL = [0.04, -0.07, 0.09, -0.06, 0.11];
      const S = { gamma: 5, lce: 0.1, ltv: 0.9, lcf: 1.0, qual: 0.5, disp: null, verify: { t0: 0, acc: [] } };
      const rng = V.rng(21);
      const calc = () => {
        const out = [];
        let surv = 1;
        for (let k = 0; k < 5; k++) {
          const e = Math.min(1, (1 - S.qual) * (0.55 + 0.22 * k));
          const q = Pt[k].map((p, i) => (1 - e) * p + e * Dr[k][i]);
          const ce = -Math.log(q[0]);
          const tv = 0.5 * Pt[k].reduce((s, p, i) => s + Math.abs(p - q[i]), 0);
          const cs = 1 - tv, c = V.clamp(cs + DEL[k], 0.03, 0.97);
          const bce = -(cs * Math.log(c) + (1 - cs) * Math.log(1 - c));
          const hcs = cs > 0 && cs < 1 ? -(cs * Math.log(cs) + (1 - cs) * Math.log(1 - cs)) : 0;
          const wk = Math.exp(-k / S.gamma);
          surv *= cs;
          out.push({ q, ce, tv, cs, c, bce, wk, surv, a: S.lce * ce * wk, b: S.ltv * tv * wk, cf: S.lcf * bce * wk, fl: S.lcf * hcs * wk });
        }
        return out;
      };
      const c9 = UI.controls(f9);
      relabel(UI.slider(c9, { label: 'γ (decay)', min: 0.5, max: 10, step: 0.5, value: 5, fmt: (v) => num(v, 1), onInput: (v) => { S.gamma = v; } }), 'decay ' + gk('γ'));
      relabel(UI.slider(c9, { label: 'λ CE', min: 0, max: 1, step: 0.05, value: 0.1, fmt: (v) => num(v, 2), onInput: (v) => { S.lce = v; } }), 'weight ' + gk('λ') + ' CE');
      relabel(UI.slider(c9, { label: 'λ TV', min: 0, max: 1, step: 0.05, value: 0.9, fmt: (v) => num(v, 2), onInput: (v) => { S.ltv = v; } }), 'weight ' + gk('λ') + ' TV');
      relabel(UI.slider(c9, { label: 'λ conf', min: 0, max: 2, step: 0.1, value: 1.0, fmt: (v) => num(v, 1), onInput: (v) => { S.lcf = v; } }), 'weight ' + gk('λ') + ' conf');
      UI.slider(c9, { label: 'quality', min: 0, max: 1, step: 0.05, value: 0.5, fmt: (v) => num(v, 2), onInput: (v) => { S.qual = v; } });
      const rTot = UI.readout(c9, 'total ℒ', '—');
      let lastTot = '';
      V.fig(f9, (g, w, h, t, dt) => {
        const C = V.C();
        const target = calc();
        if (!S.disp) S.disp = target.map((o) => Object.assign({}, o));
        const kk = Math.min(1, dt * 8);
        S.disp.forEach((o, i) => { ['ce', 'tv', 'cs', 'c', 'bce', 'wk', 'surv', 'a', 'b', 'cf', 'fl'].forEach((f) => { o[f] += (target[i][f] - o[f]) * kk; }); o.q = o.q.map((v, j) => v + (target[i].q[j] - v) * kk); });
        const D = S.disp;
        const tot = D.reduce((s, o) => s + o.a + o.b + o.cf, 0);
        const ts = num(target.reduce((s, o) => s + o.a + o.b + o.cf, 0), 3); if (ts !== lastTot) { rTot.set(ts); lastTot = ts; }
        const narrow = w < 640;
        // verificação animada (sobrevivência)
        const VP = 3.2, vph = (t % VP) / VP, cyc = Math.floor(t / VP);
        if (S.verify.cyc !== cyc) { S.verify.cyc = cyc; let alive = true; S.verify.acc = target.map((o) => { if (!alive) return 0; const ok = rng() < o.cs; if (!ok) alive = false; return ok ? 1 : -1; }); }
        const scanK = Math.floor(vph * 7);
        // cartões
        const pad = narrow ? 8 : 14, gap = narrow ? 5 : 10;
        const cw = (w - 2 * pad - gap * 4) / 5, chh = narrow ? 150 : 158, cy0 = 12;
        D.forEach((o, k) => {
          const x = pad + k * (cw + gap), y = cy0;
          const res = S.verify.acc[k], shown = scanK > k;
          const border = shown && res === 1 ? C.a : shown && res === -1 ? C.c : V.alpha(C.muted, 0.45);
          V.box(g, x, y, cw, chh, { r: 10, fill: V.alpha(C.panel, 0.85), stroke: border, lw: shown && res ? 2 : 1 });
          V.text(g, 'k = ' + (k + 1), x + 8, y + 17, { size: narrow ? 10.5 : 12, font: 'mono', color: C.ink, weight: 600 });
          if (shown && res) V.text(g, res === 1 ? '✓' : '✗', x + cw - 8, y + 18, { size: 14, color: res === 1 ? C.a : C.c, align: 'right', weight: 700 });
          else if (!narrow) V.text(g, 'w=' + num(o.wk, 2), x + cw - 8, y + 17, { size: 10.5, font: 'mono', color: C.muted, align: 'right' });
          // barras p vs q
          const bx = x + 7, bw = cw - 14, by = y + 26, bh = narrow ? 48 : 52;
          const gw = bw / 5;
          for (let i = 0; i < 5; i++) {
            const p = Pt[k][i], q = o.q[i], hx = bx + i * gw;
            const w1 = Math.max(2, gw * 0.4);
            V.box(g, hx + gw * 0.08, by + bh - p * bh / 0.72, w1, p * bh / 0.72, { r: 2, fill: V.alpha(C.b, 0.75) });
            V.box(g, hx + gw * 0.08 + w1 + 1, by + bh - q * bh / 0.72, w1, q * bh / 0.72, { r: 2, fill: V.alpha(C.a, 0.85) });
            if (i === 0) V.text(g, 'ref', hx + gw * 0.45, by + bh + 11, { size: 9.5, font: 'mono', color: C.d, align: 'center' });
          }
          const ty = by + bh + 26;
          V.text(g, 'TV ' + num(o.tv, 2), x + 8, ty, { size: narrow ? 10 : 11.5, font: 'mono', color: C.c });
          if (!narrow) V.text(g, 'CE ' + num(o.ce, 2), x + cw - 8, ty, { size: 11.5, font: 'mono', color: C.d, align: 'right' });
          // medidor de confiança
          const gy = ty + 22, gx0 = x + 8, gx1 = x + cw - 8;
          V.line(g, gx0, gy, gx1, gy, { color: V.alpha(C.muted, 0.6), lw: 3 });
          const px1 = V.lerp(gx0, gx1, o.cs), px2 = V.lerp(gx0, gx1, o.c);
          V.line(g, px1, gy - 8, px1, gy + 8, { color: C.ink, lw: 2.2 });
          V.dot(g, px2, gy, 4.5, C.e);
          V.text(g, (narrow ? 'c*' : 'c* ') + num(o.cs, 2), x + 8, gy + 20, { size: narrow ? 9.5 : 10.5, font: 'mono', color: C.ink });
          if (!narrow) V.text(g, 'c ' + num(o.c, 2), x + cw - 8, gy + 20, { size: 10.5, font: 'mono', color: C.e, align: 'right' });
        });
        if (scanK < 5) { const x = pad + scanK * (cw + gap); V.box(g, x + 6, cy0 + chh + 5, cw - 12, 4, { r: 2, fill: V.alpha(C.d, 0.9) }); if (!narrow) V.text(g, 'verifying', x + cw / 2, cy0 + chh + 21, { size: 10.5, font: 'mono', color: C.d, align: 'center' }); }
        // gráfico empilhado
        const G0 = { x: narrow ? 40 : 60, y: cy0 + chh + 54, w: w - (narrow ? 76 : 300), h: h - (cy0 + chh + 54) - (narrow ? 84 : 30) };
        const ymax = Math.max(0.2, ...D.map((o) => o.a + o.b + o.cf)) * 1.1;
        const gstep = ymax > 2 ? 1 : ymax > 0.8 ? 0.5 : 0.1;
        const YY = (v) => G0.y + G0.h - (v / ymax) * G0.h;
        g.strokeStyle = C.grid; g.lineWidth = 1; g.beginPath();
        for (let v = 0; v <= ymax; v += gstep) { g.moveTo(G0.x, YY(v)); g.lineTo(G0.x + G0.w, YY(v)); }
        g.stroke();
        for (let v = 0; v <= ymax + 1e-9; v += gstep) V.text(g, num(v, 1), G0.x - 6, YY(v) + 4, { size: 10, font: 'mono', color: C.muted, align: 'right' });
        V.text(g, narrow ? 'wₖ·λ·ℒₖ PER POSITION' : 'WEIGHTED CONTRIBUTION wₖ·λ·ℒₖ', G0.x, G0.y - 12, { size: 10.5, font: 'mono', color: C.muted, maxW: G0.w });
        const slot = G0.w / 5, bw2 = slot * 0.5;
        D.forEach((o, k) => {
          const x = G0.x + k * slot + (slot - bw2) / 2;
          let y = G0.y + G0.h;
          [[o.a, C.d], [o.b, C.c], [o.cf - o.fl, C.e]].forEach(([v, col]) => { const hh = (Math.max(0, v) / ymax) * G0.h; if (hh > 0.3) V.box(g, x, y - hh, bw2, hh, { r: 3, fill: V.alpha(col, 0.85) }); y -= hh; });
          const hf = (o.fl / ymax) * G0.h;
          if (hf > 0.3) { V.box(g, x, y - hf, bw2, hf, { r: 3, fill: V.alpha(C.e, 0.18), stroke: V.alpha(C.e, 0.7), lw: 1 }); g.save(); V.rr(g, x, y - hf, bw2, hf, 3); g.clip(); g.strokeStyle = V.alpha(C.e, 0.55); g.lineWidth = 1; for (let q = -hf; q < bw2; q += 6) { g.beginPath(); g.moveTo(x + q, y); g.lineTo(x + q + hf, y - hf); g.stroke(); } g.restore(); }
          V.text(g, 'k=' + (k + 1), x + bw2 / 2, G0.y + G0.h + 14, { size: 10.5, font: 'mono', color: C.muted, align: 'center' });
        });
        // w_k e sobrevivência (escala 0..1 à direita)
        const Y1 = (v) => G0.y + G0.h - v * G0.h;
        V.line(g, G0.x + G0.w + 4, G0.y, G0.x + G0.w + 4, G0.y + G0.h, { color: V.alpha(C.b, 0.5), lw: 1 });
        [0, 0.5, 1].forEach((v) => V.text(g, num(v, 1), G0.x + G0.w + 8, Y1(v) + 4, { size: 10, font: 'mono', color: C.b }));
        g.strokeStyle = V.alpha(C.ink, 0.7); g.lineWidth = 1.5; g.setLineDash([5, 4]); g.beginPath();
        D.forEach((o, k) => { const x = G0.x + k * slot + slot / 2; k ? g.lineTo(x, Y1(o.surv)) : g.moveTo(x, Y1(o.surv)); }); g.stroke(); g.setLineDash([]);
        D.forEach((o, k) => { const x = G0.x + k * slot + slot / 2; V.dot(g, x, Y1(o.surv), 3, C.ink); V.dot(g, x, Y1(o.wk), 3.5, C.b); });
        g.strokeStyle = C.b; g.lineWidth = 1.5; g.beginPath(); D.forEach((o, k) => { const x = G0.x + k * slot + slot / 2; k ? g.lineTo(x, Y1(o.wk)) : g.moveTo(x, Y1(o.wk)); }); g.stroke();
        // legenda + total
        const Lx = narrow ? 14 : G0.x + G0.w + 42, Ly = narrow ? h - 42 : G0.y + 2;
        const items = [[C.d, 'CE → ' + num(D.reduce((s, o) => s + o.a, 0), 3)], [C.c, 'TV → ' + num(D.reduce((s, o) => s + o.b, 0), 3)], [C.e, 'BCE → ' + num(D.reduce((s, o) => s + o.cf, 0), 3)], [V.alpha(C.e, 0.35), narrow ? 'floor H(c*)' : '▨ BCE floor H(c*)'], [C.b, narrow ? 'wₖ (0–1)' : 'wₖ = e^(−(k−1)/γ) (0–1)'], [C.ink, narrow ? '∏c* (0–1)' : 'survival ∏c* (0–1)']];
        if (narrow) {
          items.forEach(([col, lab], i) => { const x = 14 + (i % 3) * ((w - 28) / 3), y = Ly + Math.floor(i / 3) * 18; V.box(g, x, y - 8, 10, 10, { r: 2, fill: col }); V.text(g, lab, x + 14, y + 1, { size: 10, font: 'mono', color: C.muted, maxW: (w - 28) / 3 - 16 }); });
          V.text(g, 'ℒ = ' + num(tot, 3), w - 14, G0.y - 12, { size: 13, font: 'mono', color: C.ink, align: 'right', weight: 700 });
        } else {
          items.forEach(([col, lab], i) => { const y = Ly + i * 22; V.box(g, Lx, y - 9, 12, 12, { r: 3, fill: col }); V.text(g, lab, Lx + 18, y + 1, { size: 11, font: 'mono', color: C.ink, maxW: w - Lx - 24 }); });
          V.text(g, 'total ℒ', Lx, Ly + 6 * 22 + 12, { size: 11, font: 'mono', color: C.muted });
          V.text(g, num(tot, 3), Lx, Ly + 6 * 22 + 42, { size: 28, font: 'display', color: C.ink, weight: 700 });
        }
      });
    })();

    /* ================= Figura 10: treino vs inferência ================= */
    (function fig10() {
      const f10 = $('#fig-laco');
      fitH(f10, 392, 350, 620);
      const MODELS = {
        glm: { name: 'GLM-5.3-Flash', P: 321.323e9, ck: 175.623e9, ckl: '163.6 GiB (EXL3 4 bpw)', cks: '163.6 GiB', col: 'glm' },
        qwen: { name: 'Qwen3.8-Flash-Next', P: 180.2e9, ck: 132.64e9, ckl: '123.5 GiB (NVFP4 + FP8 + BF16)', cks: '123.5 GiB', col: 'qwen' },
        ds: { name: 'DeepSeek-V4-Flash', P: 304.6e9, Ptr: 284.3e9, ck: 168.6e9, ckl: '~157 GiB (FP4 + FP8, w/ DSpark)', cks: '~157 GiB', col: 'ds' },
      };
      const S = { model: 'glm', qat: false };
      const c10 = UI.controls(f10);
      UI.seg(c10, { label: 'Model', options: [{ v: 'glm', label: 'GLM' }, { v: 'qwen', label: 'Qwen' }, { v: 'ds', label: 'DeepSeek' }], value: 'glm', onChange: (v) => { S.model = v; } });
      UI.toggle(c10, { label: 'QAT: quantize in the training forward', value: false, onChange: (v) => { S.qat = v; } });
      const st = UI.stepper(f10, [
        { t: 'Minibatch', d: 'A batch of text sequences comes out of the corpus. Each position carries its target, the next token (section 2).' },
        { t: 'Forward', d: 'The activations pass through the layers. In each one, the weights \\(W\\) multiply the activations, which are <b>kept</b> for the backward (the lilac stacks under each layer). That costs memory too.' },
        { t: 'Loss', d: 'At the output, the next-token distribution is compared with the real token: \\(\\ell = -\\ln p_\\theta(x_t\\mid x_{&lt;t})\\), averaged over the batch.' },
        { t: 'Backward', d: 'Backpropagation carries the gradient back, layer by layer, and produces \\(\\nabla W\\) for <b>every</b> matrix (next chapter).' },
        { t: 'Optimizer', d: 'AdamW (moments \\(m, s\\)) or Muon (orthogonalized momentum) updates the weights. Then the loop goes back to step 1, for tens or hundreds of thousands of steps.' },
        { t: 'Freeze', d: 'End of training. Loss, backward and optimizer go away. Only the weights go into the checkpoint, and now they have a padlock.' },
        { t: 'Inference', d: 'Only the forward is left. Prompt tokens go in, one token comes out, goes back to the end of the line and the KV cache grows. The recipe chooses <i>how</i> to compute this forward, not what it computes.' },
      ], { autoplay: 5200 });
      V.fig(f10, (g, w, h, t) => {
        const C = V.C();
        const M = MODELS[S.model];
        const i = st.i, el = st.elapsed();
        const infer = i >= 5;
        const narrow = w < 620;
        const nL = 4;
        const top = narrow ? 34 : 44;
        const x0 = narrow ? 10 : 24, x1 = w - (narrow ? 10 : 24);
        const bw0 = narrow ? 50 : 96;
        const lw = (x1 - x0 - 2 * bw0 - (nL + 1) * (narrow ? 8 : 18)) / nL;
        const gapx = narrow ? 8 : 18;
        const bh = narrow ? 70 : 88;
        const src = { x: x0, y: top, w: bw0, h: bh };
        const Ls = Array.from({ length: nL }, (_, k) => ({ x: x0 + bw0 + gapx + k * (lw + gapx), y: top, w: lw, h: bh }));
        const out = { x: x1 - bw0, y: top, w: bw0, h: bh };
        const active = (k) => k === i;
        // fonte
        V.box(g, src.x, src.y, src.w, src.h, { r: 10, fill: V.alpha(C.b, active(0) || infer ? 0.28 : 0.12), stroke: C.b, lw: active(0) ? 2 : 1.2 });
        V.text(g, infer ? 'prompt' : 'minibatch', src.x + src.w / 2, src.y + src.h / 2 - 2, { size: narrow ? 10.5 : 12.5, font: 'mono', color: C.ink, align: 'center', maxW: src.w - 6 });
        V.text(g, infer ? 'tokens' : 'text + targets', src.x + src.w / 2, src.y + src.h / 2 + 14, { size: narrow ? 9.5 : 10.5, color: C.muted, align: 'center', maxW: src.w - 6 });
        // camadas
        const labs = narrow ? ['L1', 'L2', '…', 'Lₙ'] : ['layer 1', 'layer 2', '…', 'layer L'];
        Ls.forEach((L, k) => {
          const glow = i === 1 || i === 3 || i === 4 || i === 6;
          V.box(g, L.x, L.y, L.w, L.h, { r: 10, fill: V.alpha(C.a, glow ? 0.14 : 0.07), stroke: V.alpha(C.a, 0.7), lw: 1.2 });
          V.text(g, labs[k], L.x + L.w / 2, L.y + 15, { size: narrow ? 10 : 11.5, font: 'mono', color: C.ink, align: 'center', maxW: L.w - 4 });
          // mini matriz W
          const n = narrow ? 3 : 4, cs = Math.min(narrow ? 9 : 10, (L.w - 16) / (n + 1)), gx = L.x + L.w / 2 - (n * cs + (n - 1) * 2) / 2, gy = L.y + 24;
          for (let a = 0; a < n; a++) for (let b = 0; b < n; b++) {
            const flash = i === 4 ? 0.5 + 0.5 * Math.sin(t * 7 + a + b + k) : 0;
            g.fillStyle = V.alpha(i === 4 ? C.d : C.a, 0.3 + 0.35 * Math.abs(Math.sin(a * 1.7 + b * 2.3 + k)) + 0.3 * flash);
            V.rr(g, gx + b * (cs + 2), gy + a * (cs + 2), cs, cs, 2); g.fill();
          }
          V.text(g, 'W', L.x + L.w / 2, L.y + L.h - 7, { size: 10, font: 'mono', color: C.muted, align: 'center' });
          if (infer) {
            const lx = L.x + L.w - 12, ly = L.y + 12, sc = el < 1 && i === 5 ? V.ease.back(Math.min(1, el)) : 1;
            g.save(); g.translate(lx, ly); g.scale(sc, sc);
            g.strokeStyle = C.d; g.lineWidth = 1.6; g.beginPath(); g.arc(0, -3, 4, Math.PI, 0); g.stroke();
            V.box(g, -6, -3, 12, 10, { r: 2, fill: C.d });
            g.restore();
          }
          if (S.qat && !infer && k < nL - 1) {
            const qx = L.x + L.w + gapx / 2, qy = L.y + L.h / 2;
            V.box(g, qx - (narrow ? 4 : 8), qy - 9, narrow ? 8 : 16, 18, { r: 4, fill: V.alpha(C.f, 0.35), stroke: C.f, lw: 1 });
            if (!narrow) V.text(g, 'Q', qx, qy + 4, { size: 9.5, font: 'mono', color: C.ink, align: 'center' });
          }
        });
        // saída
        const outOn = active(2) || (!infer && i > 2) || infer;
        V.box(g, out.x, out.y, out.w, out.h, { r: 10, fill: infer ? V.alpha(C.a, 0.18) : V.alpha(C.c, outOn ? 0.25 : 0.1), stroke: infer ? C.a : C.c, lw: active(2) ? 2 : 1.2, dash: null });
        V.text(g, infer ? 'next' : 'loss ℒ', out.x + out.w / 2, out.y + out.h / 2 - 2, { size: narrow ? 10.5 : 12.5, font: 'mono', color: C.ink, align: 'center', maxW: out.w - 6 });
        V.text(g, infer ? 'token' : '−ln p(target)', out.x + out.w / 2, out.y + out.h / 2 + 14, { size: narrow ? 9.5 : 10.5, color: C.muted, align: 'center', maxW: out.w - 6 });
        if (i === 5 && !narrow) {
          g.strokeStyle = V.alpha(C.c, 0.8); g.lineWidth = 2;
          g.beginPath(); g.moveTo(out.x + 10, out.y + 10); g.lineTo(out.x + out.w - 10, out.y + out.h - 10); g.stroke();
        }
        // fluxo forward
        const cy = top + bh / 2;
        const fwd = i === 1 || i === 2 || i === 6 || i === 0;
        if (fwd) {
          for (let p = 0; p < 10; p++) {
            const u = (p / 10 + t * 0.35) % 1;
            const x = V.lerp(src.x + src.w, out.x, u);
            V.dot(g, x, cy + Math.sin(u * 20 + p) * 3, 2.6, infer ? C.a : C.b);
          }
        }
        V.arrow(g, src.x + src.w + 2, top - 8, out.x - 2, top - 8, { color: V.alpha(C.b, fwd ? 0.9 : 0.35), lw: 1.4, head: 7 });
        V.text(g, infer ? 'forward (the only thing left)' : 'forward', (src.x + out.x + out.w) / 2, top - 14, { size: 10.5, font: 'mono', color: C.b, align: 'center' });
        // backward
        const ay = top + bh + 7;
        if (!infer && (i === 1 || i === 2 || i === 3)) {
          Ls.forEach((L, k) => {
            let al = 1;
            if (i === 1) al = V.clamp(el * 1.6 - k * 0.4, 0, 1);
            if (i === 3) al = V.clamp(1 - (el * 1.4 - (nL - 1 - k) * 0.4), 0, 1);
            if (al <= 0) return;
            g.globalAlpha = al;
            for (let q = 0; q < 3; q++) V.box(g, L.x + L.w / 2 - 13 + q * 3, ay + q * 3, 20, 8, { r: 2, fill: V.alpha(C.b, 0.45), stroke: V.alpha(C.b, 0.95), lw: 0.8 });
            g.globalAlpha = 1;
          });
        }
        const by = top + bh + (narrow ? 34 : 40);
        if (!infer) {
          const on = i === 3;
          const optStep = i === 4; // no passo do otimizador, as setas até as camadas cruzam esta faixa: apague o rótulo
          V.arrow(g, out.x + out.w / 2, by, src.x + src.w + 4, by, { color: V.alpha(C.c, on ? 0.95 : optStep ? 0.15 : 0.3), lw: on ? 2.4 : 1.4, head: 8 });
          V.text(g, S.qat && on ? 'backward: ∇W (STE passes through Q as the identity)' : 'backward: ∇W', (src.x + out.x + out.w) / 2, by + 16, { size: 10.5, font: 'mono', color: C.c, align: 'center', maxW: out.x - src.x, alpha: optStep ? 0.25 : 1 });
          if (on) for (let p = 0; p < 8; p++) { const u = (p / 8 + t * 0.4) % 1; V.dot(g, V.lerp(out.x, src.x + src.w, u), by, 2.6, C.c); }
        }
        // otimizador ou KV cache
        const oy = by + (narrow ? 28 : 32), oh = narrow ? 40 : 46;
        if (!infer) {
          const on = i === 4;
          const ow = Math.min(360, x1 - x0 - 40), ox = (w - ow) / 2;
          V.box(g, ox, oy, ow, oh, { r: 10, fill: V.alpha(C.d, on ? 0.25 : 0.08), stroke: V.alpha(C.d, on ? 1 : 0.5), lw: on ? 2 : 1.2 });
          V.text(g, 'optimizer: AdamW (m, s) · Muon (M)', ox + ow / 2, oy + oh / 2 + 4, { size: narrow ? 10 : 12, font: 'mono', color: C.ink, align: 'center', maxW: ow - 12 });
          if (on) Ls.forEach((L) => V.arrow(g, ox + ow / 2, oy, L.x + L.w / 2, L.y + L.h + 2, { color: V.alpha(C.d, 0.6), lw: 1.2, head: 6, curve: 10 }));
        } else {
          const kx = Ls[0].x, kw = out.x + out.w - kx;
          const n = i === 6 ? 6 + (Math.floor(t * 1.5) % 14) : 6;
          V.text(g, 'KV cache (grows 1 row per token)', kx, oy - 4, { size: 10.5, font: 'mono', color: C.muted, maxW: kw });
          const cwid = Math.min(18, kw / 22);
          for (let c = 0; c < n; c++) V.box(g, kx + c * (cwid + 3), oy + 4, cwid, oh - 14, { r: 3, fill: V.alpha(c === n - 1 && i === 6 ? C.a : C.b, 0.45) });
          if (i === 6) {
            const u = (t * 0.5) % 1;
            const tx = V.lerp(out.x + out.w / 2, src.x + src.w / 2, V.ease.inOut(u)), tyy = top + bh + 8 + Math.sin(u * Math.PI) * 18;
            pill(g, 'tok', tx - 14, tyy - 9, { size: 10, fill: V.alpha(C.a, 0.35), stroke: C.a });
          }
        }
        // barra de memória
        const my = h - (narrow ? 96 : 84);
        V.text(g, 'BYTES PER PARAMETER · ' + M.name, x0, my, { size: 10.5, font: 'mono', color: C.muted, maxW: x1 - x0 });
        const bx = x0 + (narrow ? 72 : 86), bwid = x1 - bx;
        const scale = (b) => (b / 16) * bwid;
        const trainSegs = [[2, C.a, 'weights'], [2, C.c, 'grad'], [4, C.b, 'master'], [4, C.d, 'm'], [4, C.f, 's']];
        const yT = my + 10, yI = my + 40, hb = 20;
        V.text(g, 'training', x0, yT + 14, { size: narrow ? 10 : 11, font: 'mono', color: infer ? C.muted : C.ink });
        let xx = bx; trainSegs.forEach(([b, col, lab]) => { const ww = scale(b); V.box(g, xx, yT, ww - 2, hb, { r: 4, fill: V.alpha(col, infer ? 0.3 : 0.75) }); if (ww > 30) V.text(g, lab, xx + ww / 2 - 1, yT + 14, { size: 10, font: 'mono', color: onInk(), align: 'center', maxW: ww - 4 }); xx += ww; });
        const Ptr = M.Ptr || M.P;
        V.text(g, '≈16 B → ' + num((Ptr * 16) / 1e12, 1) + ' TB', x1, yT - 4, { size: 10.5, font: 'mono', color: C.muted, align: 'right' });
        const bpp = M.ck / M.P;
        V.text(g, 'inference', x0, yI + 14, { size: narrow ? 10 : 11, font: 'mono', color: infer ? C.ink : C.muted });
        V.box(g, bx, yI, Math.max(4, scale(bpp)), hb, { r: 4, fill: V.alpha(C[M.col] || C.a, 0.9) });
        const il = narrow ? '≈' + num(bpp, 2) + ' B (' + num(bpp * 8, 1) + ' bits) · ' + M.cks : '≈' + num(bpp, 2) + ' B/param (' + num(bpp * 8, 1) + ' bits) → ' + M.ckl;
        V.text(g, il, bx + scale(bpp) + 8, yI + 14, { size: narrow ? 10 : 11, font: 'mono', color: C.ink, maxW: bwid - scale(bpp) - 8 });
      });
    })();
  },
});

LA.chapter({
  id: 'atencao',
  part: 2,
  order: 6,
  title: 'Attention: how tokens talk to each other',
  navTitle: 'Attention',
  subtitle: 'Query, key and value; the dot product and the 1/√d scale; the softmax that must not overflow; masks, heads and the O(n²) cost. And the three tweaks that show up in the recipes\' configs: attention sink, output gate and QK-Norm.',
  minutes: 50,
  recipes: ['glm', 'tf', 'qwen', 'ds'],
  tags: ['attention', 'Q/K/V', 'softmax', 'log-sum-exp', 'causal mask', 'multi-head', 'head_dim', 'attention sink', 'gated attention', 'QK-Norm', 'O(n²)'],
  goals: [
    'Build attention by hand: <b>Q, K, V</b> projections, scores \\(q\\cdot k/\\sqrt{d_h}\\), <b>softmax</b> and the weighted sum of the values',
    'Explain why the \\(1/\\sqrt{d_h}\\) scale, <b>QK-Norm</b>, the stable softmax, the <b>log-sum-exp</b> and the online softmax exist, and how the recipes\' kernels use each one',
    'Tell a <b>causal</b> mask from a <b>bidirectional</b> one and know which one each module needs (target, vision, block drafter)',
    'Read <code>num_attention_heads</code>, <code>head_dim</code>, <code>attn_sink</code> and <code>output_gate_type</code> in a config and predict shapes, per-GPU split and KV bytes',
  ],
  prereqs: ['tokens-embeddings', 'tensores-algebra'],
  summary: [
    'Attention = compare (\\(s_{ij} = q_i\\cdot k_j/\\sqrt{d_h}\\)), normalize (softmax per row) and mix (\\(o_i = \\sum_j a_{ij} v_j\\)). Q, K and V are three projections of the same vector, with different roles.',
    'Without the scale, the standard deviation of the scores grows as \\(\\sqrt{d_h}\\) (16 with \\(d_h = 256\\)) and the softmax saturates. QK-Norm (per-head RMSNorm on q and k) also holds back the growth of the norms during training: Qwen <code>q_norm</code>/<code>k_norm</code> [256]; DeepSeek divides q by its RMS and adds <code>kv_norm</code>.',
    'The stable softmax subtracts the maximum (FP16 overflows at \\(e^{11.1}\\)); the LSE \\(= m + \\ln\\ell\\) enables the block-wise online softmax and the exact split-K merge. Kernels use exp2 with the scale × log₂e and fold FP8 scales into the temperature and the normalizer. In TensorFold, GLM\'s sparse prefill skips split-K: one online softmax per row, no partials.',
    'Causal mask in the target (honest parallel training, KV that never changes); bidirectional in vision and inside the drafters\' block. DFlash2 with <code>is_causal: false</code>: pinning <code>TRITON_ATTN</code> (causal within the block) dropped acceptance from 0.959 to 0.31.',
    'Multi-head: \\(H\\) attentions of \\(d_h\\) dimensions, concatenated and mixed by <code>o_proj</code>. In modern models \\(H d_h \\ne d\\): GLM 64 × 256, Qwen 24 × 256 (the gate doubles <code>q_proj</code> to [12,288 × 2560]), DeepSeek 64 × 512 with a grouped <code>o_proj</code>. Heads are the slices of TP, and 64 is not divisible by 3.',
    'The cost is \\(O(n^2)\\) in FLOPs and, if materialized, in memory; the KV grows as \\(O(n)\\) (Qwen: 26,624 B/token in BF16). Linear, sparse and latent are the three ways out, and that is why the three models are hybrids.',
    'The softmax sums to 1, so heads with nothing to say dump attention on the 1st token. Solutions: a learned sink in the denominator (DeepSeek <code>attn_sink</code>, padding always \\(-\\infty\\)) or a sigmoid gate on the output (Qwen, <code>output_gate_type: sigmoid</code>).',
  ],
  quiz: [
    { q: 'In a head with \\(d_h = 256\\) and independent coordinates with mean 0 and variance 1, what is the standard deviation of the unscaled \\(q\\cdot k\\), and what does dividing by \\(\\sqrt{d_h}\\) do?', o: ['16; it goes back to 1, whatever \\(d_h\\) is', '256; it goes back to 16', '1; the division changes nothing', '\\(\\sqrt{2}\\); it makes the softmax sharper'], a: 0, e: 'The sum of 256 independent terms of variance 1 has variance 256, and standard deviation 16. Dividing by \\(\\sqrt{256} = 16\\) brings the standard deviation back to 1. Without it, differences of ~16 between scores become factors of millions in the softmax, which saturates.' },
    { q: 'Why doesn\'t subtracting the maximum of the scores before the softmax change the weights?', o: ['Because the softmax rounds the result', 'Because \\(e^{-m}\\) appears in the numerator and in the denominator and cancels out; only the overflow goes away', 'Because the maximum is always zero', 'It does change them, but the error is small in BF16'], a: 1, e: 'softmax(s − m) = e<sup>−m</sup>e<sup>s<sub>j</sub></sup>/(e<sup>−m</sup>Σe<sup>s<sub>k</sub></sup>) = softmax(s). The computation is identical and the largest term becomes e<sup>0</sup> = 1, which doesn\'t overflow in any format.' },
    { q: 'In the GLM recipe, pinning <code>TRITON_ATTN</code> for the DFlash2 drafter dropped acceptance to 0.31. Why?', o: ['Triton is slow on the GB10', 'In that image TRITON_ATTN applies a causal mask inside the block, and DFlash2 was trained with bidirectional attention in the block (<code>is_causal: false</code>)', 'TRITON_ATTN doesn\'t support FP8', 'TRITON_ATTN turns off the 2048 window'], a: 1, e: 'The mask is a training contract. With the causal mask, each position of the block loses sight of the positions ahead of it, which the model learned to use. The first position barely suffers and the following ones collapse. The right backend (FLASH_ATTN, non-causal) gives 0.959.' },
    { q: 'The <code>q_proj</code> of Qwen3.8\'s QSA has shape <code>[12288, 2560]</code>. Where does the 12,288 come from?', o: ['24 heads × 256 × 2: the query and the output gate of each head', '48 heads × 256', 'Q, K and V fused', '24 × 512 because head_dim is 512'], a: 0, e: 'There are 24 heads of 256 dimensions, and each head has 256 query rows and 256 gate rows side by side (the sigmoid gate multiplies the output before o_proj). K and V live in <code>k_proj</code>/<code>v_proj</code> of <code>[512, 2560]</code>.' },
    { q: 'In DeepSeek-V4 at TP=3, the padding heads (64 → 72) get <code>attn_sink = −∞</code>. Why not 0?', o: ['Because 0 can\'t be represented in FP32', 'Because the sink enters the denominator as \\(e^{\\sigma}\\): \\(e^{-\\infty} = 0\\) is neutral, while \\(e^0 = 1\\) would add a “ghost unit” and dampen the real heads', 'Because −∞ speeds up the kernel', 'It doesn\'t matter, padding heads have zero weights'], a: 1, e: 'The neutral element of an exponential term is −∞. A sink of 0 is not “nothing”: it is one more competitor with a score of 0. The module injected by the TP=3 patch (<code>apply_tp3_patch.py</code>) warns about exactly this, which is why the sink has its own loaders that don\'t zero the destination.' },
    { q: 'Qwen3.8\'s QSA stores K and V in 13 layers (12 + MTP), with 2 KV heads of 256 dimensions. How many bytes of KV per token in BF16?', o: ['26,624 B', '13,312 B', '2,048 B', '319,488 B'], a: 0, e: '13 layers × 2 (K and V) × 2 heads × 256 × 2 bytes = 26,624 B. In FP8 it drops to 13,312 B, which is also the per-GPU value in BF16 at TP=2. 319,488 B would be with 24 KV heads (MHA).' },
  ],
  mount(root, LA) {
    const V = LA.viz, UI = LA.ui;

    /* ------------------------------------------------------------ helpers */
    const nf = (x) => Math.round(x).toLocaleString('en-US');
    const sf = (x, d = 2) => (+x).toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d });
    const sg = (x, d = 1) => { const r = sf(x, d); return /^-0(\.0+)?$/.test(r) ? r.slice(1) : r.replace('-', '−'); };
    const SUP = { '-': '⁻', 0: '⁰', 1: '¹', 2: '²', 3: '³', 4: '⁴', 5: '⁵', 6: '⁶', 7: '⁷', 8: '⁸', 9: '⁹' };
    const sup = (n) => String(n).split('').map((c) => SUP[c] || c).join('');
    const sci = (x, d = 2) => {
      if (Number.isNaN(x)) return 'NaN';
      if (!Number.isFinite(x)) return x > 0 ? 'Inf' : '−Inf';
      if (x === 0) return '0';
      let e = Math.floor(Math.log10(Math.abs(x)));
      if (e >= -2 && e <= 4) return sg(x, e >= 3 ? 0 : e >= 1 ? 1 : d);
      let m = x / Math.pow(10, e);
      if (Math.abs(+m.toFixed(d)) >= 10) { m /= 10; e += 1; }
      return sg(m, d) + '·10' + sup(e);
    };
    const bytesI = (b) => { const u = ['B', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB']; let i = 0; while (b >= 1024 && i < u.length - 1) { b /= 1024; i++; } return i ? sf(b, b < 10 ? 2 : 1) + ' ' + u[i] : nf(b) + ' B'; };
    const bytesD = (b) => (b >= 1e12 ? sf(b / 1e12, 1) + ' TB' : b >= 1e9 ? sf(b / 1e9, 1) + ' GB' : b >= 1e6 ? sf(b / 1e6, 1) + ' MB' : b >= 1e3 ? sf(b / 1e3, 1) + ' kB' : nf(b) + ' B');
    let fcache = {};
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(() => { fcache = {}; });
    const font = (s, f = 'sans', w = 400) => { const k = s + '|' + f + '|' + w; return fcache[k] || (fcache[k] = V.font(s, f, w)); };
    const tw = (g, s, size = 12, f = 'sans', w = 400) => { g.font = font(size, f, w); return g.measureText(String(s)).width; };
    function pill(g, x, y, s, col, o = {}) {
      const size = o.size || 11, f = o.font || 'mono', wt = o.weight || 600;
      const wd = tw(g, s, size, f, wt) + 14, ht = size + 9;
      const xx = o.align === 'center' ? x - wd / 2 : o.align === 'right' ? x - wd : x;
      V.box(g, xx, y - ht / 2, wd, ht, { r: ht / 2, fill: V.alpha(col, o.a == null ? 0.16 : o.a), stroke: V.alpha(col, 0.75), lw: 1 });
      V.text(g, s, xx + wd / 2, y + 0.5, { size, font: f, weight: wt, color: o.color || V.C().ink, align: 'center', base: 'middle' });
      return wd;
    }
    function rgbOf(c) {
      c = String(c || '').trim();
      if (c[0] === '#') { let h = c.slice(1); if (h.length === 3) h = h.split('').map((x) => x + x).join(''); const n = parseInt(h.slice(0, 6), 16); return [(n >> 16) & 255, (n >> 8) & 255, n & 255]; }
      const p = c.match(/[\d.]+/g) || [128, 128, 128];
      return [+p[0], +p[1], +p[2]];
    }
    const lum = (c) => { const [r, g2, b] = rgbOf(c); return (0.299 * r + 0.587 * g2 + 0.114 * b) / 255; };
    const inkFor = (bg) => (lum(bg) > 0.58 ? '#0b1a13' : '#ffffff');
    function blend(cols, ws) {
      let r = 0, gg = 0, b = 0, s = 0;
      cols.forEach((c, i) => { const w = ws[i] || 0; const [a1, a2, a3] = rgbOf(c); r += a1 * w; gg += a2 * w; b += a3 * w; s += w; });
      s = s || 1;
      return `rgb(${Math.round(r / s)},${Math.round(gg / s)},${Math.round(b / s)})`;
    }
    function hatch(g, x, y, w, h, col, step = 5) {
      if (w <= 0 || h <= 0) return;
      g.save(); g.beginPath(); g.rect(x, y, w, h); g.clip();
      g.strokeStyle = col; g.lineWidth = 1; g.beginPath();
      for (let k = -h; k < w; k += step) { g.moveTo(x + k, y + h); g.lineTo(x + k + h, y); }
      g.stroke(); g.restore();
    }
    /** matriz com valores; cores 'div' (±) ou 'seq' (0..1) ou colorFn */
    function mat(g, M, x, y, cw, ch, o = {}) {
      const C = V.C();
      const R = M.length, K = M[0].length, gap = o.gap == null ? 2 : o.gap;
      let mx = o.max;
      if (mx == null) { mx = 1e-9; M.forEach((row) => row.forEach((v) => { if (v != null && Number.isFinite(v)) mx = Math.max(mx, Math.abs(v)); })); }
      const fs = o.fs || Math.max(11, Math.min(12.5, cw * 0.34));
      for (let i = 0; i < R; i++) for (let j = 0; j < K; j++) {
        const v = M[i][j];
        const cx = x + j * (cw + gap), cy = y + i * (ch + gap);
        if (o.show && !o.show(i, j)) {
          V.rr(g, cx, cy, cw, ch, Math.min(4, cw / 6)); g.fillStyle = V.alpha(C.muted, 0.08); g.fill();
          continue;
        }
        const ga = o.cellAlpha ? o.cellAlpha(i, j) : 1;
        if (ga !== 1) g.globalAlpha = ga;
        let fill;
        if (v == null || Number.isNaN(v)) fill = V.alpha(C.muted, 0.1);
        else if (o.colorFn) fill = o.colorFn(v, i, j);
        else if (o.cmap === 'seq') fill = V.cmap(V.clamp(v / mx, 0, 1), 'seq');
        else fill = V.cmap(V.clamp(v / mx, -1, 1), 'div');
        V.rr(g, cx, cy, cw, ch, o.r == null ? Math.min(4, cw / 6) : o.r); g.fillStyle = fill; g.fill();
        if (ga !== 1) g.globalAlpha = 1;
        if (o.values !== false && v != null && cw >= 18) {
          const lbl = o.fmt ? o.fmt(v, i, j) : sg(v, o.digits == null ? 1 : o.digits);
          // célula esmaecida: só o preenchimento perde opacidade; a cor do texto é escolhida
          // contra a cor que o leitor vê de fato (preenchimento misturado ao fundo do palco)
          const seen = ga !== 1 ? blend([C.dark ? '#0b1a13' : '#f2f3f7', fill], [1 - ga, ga]) : fill;
          const col = o.textColor ? o.textColor(v, i, j, seen) : inkFor(seen);
          if (ga !== 1) g.globalAlpha = Math.max(0.72, ga);
          V.text(g, lbl, cx + cw / 2, cy + ch / 2 + 0.5, { size: fs, font: 'mono', align: 'center', base: 'middle', color: col });
          if (ga !== 1) g.globalAlpha = 1;
        }
      }
      const W = K * cw + (K - 1) * gap, Ht = R * ch + (R - 1) * gap;
      const hl = (xx, yy, ww, hh, col) => { g.strokeStyle = col || o.hiColor || C.d; g.lineWidth = 2.2; V.rr(g, xx - 2, yy - 2, ww + 4, hh + 4, 5); g.stroke(); };
      if (o.hiRow != null && o.hiRow >= 0) hl(x, y + o.hiRow * (ch + gap), W, ch);
      if (o.hiCol != null && o.hiCol >= 0) hl(x + o.hiCol * (cw + gap), y, cw, Ht);
      if (o.hiCell) hl(x + o.hiCell[1] * (cw + gap), y + o.hiCell[0] * (ch + gap), cw, ch, o.hiCellColor);
      if (o.label) V.text(g, o.label, x + W / 2, y - 9, { size: 12, align: 'center', color: o.labelColor || C.muted, font: 'mono', weight: 600 });
      if (o.rowLabels) o.rowLabels.forEach((l, i) => V.text(g, l, x - 6, y + i * (ch + gap) + ch / 2, { size: 11, align: 'right', base: 'middle', color: o.rowLabelColor ? o.rowLabelColor(i) : C.muted, font: 'sans' }));
      return {
        x, y, w: W, h: Ht, cw, ch, gap,
        cellAt(px, py) { const j = Math.floor((px - x) / (cw + gap)), i = Math.floor((py - y) / (ch + gap)); return i >= 0 && i < R && j >= 0 && j < K ? [i, j] : null; },
        center(i, j) { return [x + j * (cw + gap) + cw / 2, y + i * (ch + gap) + ch / 2]; },
      };
    }
    /** altura do palco conforme a largura (palco estreito = figura mais alta) */
    function autoH(fig, wideH, narrowH, bp = 560, sel) {
      const st = sel ? fig.querySelector(sel) : fig.querySelector('.la-fig-stage');
      if (!st) return;
      const apply = () => { const w = st.getBoundingClientRect().width || st.clientWidth; if (!w) return; const hh = (w < bp ? narrowH : wideH) + 'px'; if (st.style.height !== hh) st.style.height = hh; };
      apply();
      const onR = () => requestAnimationFrame(apply);
      window.addEventListener('resize', onR);
      LA.onCleanup(() => window.removeEventListener('resize', onR));
    }
    /** texto com quebra de linha por palavras; devolve o y da linha seguinte */
    function para(g, s, x, y, maxW, o = {}) {
      const size = o.size || 12, f = o.font || 'sans', wt = o.weight || 400, lh = o.lh || Math.round(size * 1.45);
      const words = String(s).split(' ');
      let line = '';
      const out = [];
      words.forEach((wd) => { const test = line ? line + ' ' + wd : wd; if (line && tw(g, test, size, f, wt) > maxW) { out.push(line); line = wd; } else line = test; });
      if (line) out.push(line);
      out.forEach((l, k) => V.text(g, l, x, y + k * lh, { size, font: f, weight: wt, color: o.color, align: o.align }));
      return y + out.length * lh;
    }
    const softmaxArr = (xs) => { let m = -Infinity; xs.forEach((x) => { if (x > m) m = x; }); const e = xs.map((x) => (x === -Infinity ? 0 : Math.exp(x - m))); const z = e.reduce((a, b) => a + b, 0) || 1; return e.map((x) => x / z); };
    const entropy2 = (p) => -p.reduce((a, x) => a + (x > 1e-12 ? x * Math.log2(x) : 0), 0);

    /* =============================================================
       Fig 1 — a festa dos tokens (quem escuta quem)
       ============================================================= */
    {
      const fig = root.querySelector('#fig-festa');
      autoH(fig, 420, 420);
      const SENT = {
        gata: {
          toks: ['The', 'cat', 'climbed', 'the', 'tree', 'because', 'it', 'was', 'so', 'scared'], def: 6,
          links: [[6, 1, 3.2], [6, 0, 0.6], [6, 9, 1.2], [2, 1, 2.4], [2, 4, 1.6], [4, 2, 1.6], [4, 3, 1.0], [3, 4, 1.8], [3, 2, 0.8], [9, 6, 1.8], [9, 7, 1.2], [7, 6, 2.2], [5, 2, 1.3], [5, 9, 1.0], [1, 0, 1.4], [8, 9, 1.6], [8, 7, 0.8], [0, 1, 1.0]],
        },
        banco: {
          toks: ['The', 'angler', 'walked', 'the', 'bank', 'up', 'river', 'and', 'caught', 'a', 'fish'], def: 4,
          links: [[4, 1, 1.8], [4, 2, 1.5], [4, 6, 2.8], [6, 4, 2.4], [6, 5, 0.8], [8, 1, 2.8], [8, 6, 1.0], [10, 8, 2.4], [10, 6, 1.6], [9, 10, 1.2], [9, 8, 1.0], [2, 1, 2.2], [1, 0, 1.2], [5, 6, 1.6], [7, 8, 1.2], [3, 4, 1.6], [0, 1, 1.0]],
        },
      };
      const S = { key: 'gata', sel: 6, causal: true, W: null };
      function weights(sent, i, causal) {
        const n = sent.toks.length;
        const L = new Array(n).fill(0);
        L[i] += 0.8; if (i > 0) L[i - 1] += 0.9;
        sent.links.forEach(([a, b, v]) => { if (a === i) L[b] += v; });
        const sc = L.map((v, j) => (causal && j > i ? -Infinity : v));
        const p = softmaxArr(sc);
        return p.map((v, j) => (causal && j > i ? null : v));
      }
      const ctl = UI.controls(fig);
      let H = null;
      const rd = { set() {} };
      function recompute() {
        S.W = weights(SENT[S.key], S.sel, S.causal);
        const toks = SENT[S.key].toks;
        let bj = -1;
        S.W.forEach((v, j) => { if (v != null && j !== S.sel && (bj < 0 || v > S.W[bj])) bj = j; });
        rd.set(bj >= 0 ? `${toks[bj]} · ${Math.round(S.W[bj] * 100)}%` : '—');
        if (H) H.redraw();
      }
      UI.seg(ctl, { label: 'Sentence', options: [{ v: 'gata', label: '“…because it…”' }, { v: 'banco', label: '“…bank up river…”' }], value: 'gata', onChange: (v) => { S.key = v; S.sel = SENT[v].def; recompute(); } });
      UI.toggle(ctl, { label: 'causal mask', value: true, onChange: (v) => { S.causal = v; recompute(); } });
      const rdo = UI.readout(ctl, 'who answers most', '—');
      rd.set = rdo.set;
      recompute();
      H = V.fig(fig, (g, w, h, t, dt, _s, HH) => {
        if (w < 80 || h < 80) return;
        const C = V.C();
        const sent = SENT[S.key], toks = sent.toks, n = toks.length;
        const W = S.W;
        const wide = w >= 640;
        const panelW = wide ? Math.min(270, w * 0.34) : 0;
        const areaW = w - panelW;
        const top = 46, bottom = 10;
        const rowH = Math.min(34, (h - top - bottom) / n);
        const pillH = Math.min(26, rowH - 5);
        const colW = Math.min(124, Math.max(92, areaW * 0.3));
        const xL = 12, xR = areaW - 12 - colW;
        const yOf = (i) => top + i * rowH + rowH / 2;
        const hue = (j) => C.series[j % 6];
        V.text(g, 'QUERY · who asks', xL, 24, { size: 11, font: 'mono', color: C.muted, weight: 600 });
        V.text(g, 'KEY → VALUE · who answers', xR + colW, 24, { size: 11, font: 'mono', color: C.muted, align: 'right', weight: 600 });
        // interação: clicar num token da esquerda
        const p = HH.pointer;
        let hov = -1;
        if (p.inside && p.x >= xL && p.x <= xL + colW) { const i = Math.floor((p.y - top) / rowH); if (i >= 0 && i < n) hov = i; }
        if (p.clicked && hov >= 0 && hov !== S.sel) { S.sel = hov; recompute(); }
        HH.canvas.style.cursor = hov >= 0 ? 'pointer' : 'default';
        let amax = 1e-6; W.forEach((v) => { if (v != null && v > amax) amax = v; });
        const x1 = xL + colW + 6, x2 = xR - 6, y1 = yOf(S.sel);
        const cx1 = x1 + (x2 - x1) * 0.45, cx2 = x2 - (x2 - x1) * 0.45;
        // fios + mensagens (values) viajando da direita para a esquerda
        for (let j = 0; j < n; j++) {
          const a = W[j];
          if (a == null) continue;
          const y2 = yOf(j);
          g.strokeStyle = V.alpha(hue(j), 0.1 + 0.7 * (a / amax));
          g.lineWidth = 0.8 + 11 * a;
          g.beginPath(); g.moveTo(x1, y1); g.bezierCurveTo(cx1, y1, cx2, y2, x2, y2); g.stroke();
          const k = Math.max(1, Math.round(a * 16));
          for (let m = 0; m < k; m++) {
            const u = 1 - ((t * 0.3 + m / k + j * 0.137) % 1);
            const pt = V.bez([x1, y1], [cx1, y1], [cx2, y2], [x2, y2], u);
            V.dot(g, pt[0], pt[1], 1.5 + 2.4 * a, hue(j));
          }
        }
        // coluna da direita (keys)
        for (let j = 0; j < n; j++) {
          const y = yOf(j), a = W[j], masked = a == null;
          const x = xR;
          if (masked) {
            V.box(g, x, y - pillH / 2, colW, pillH, { r: pillH / 2, fill: V.alpha(C.muted, 0.05) });
            g.setLineDash([3, 3]); V.box(g, x, y - pillH / 2, colW, pillH, { r: pillH / 2, stroke: V.alpha(C.muted, 0.45), lw: 1 }); g.setLineDash([]);
            V.text(g, toks[j], x + 12, y + 0.5, { size: 12, color: V.alpha(C.muted, 0.7), base: 'middle' });
            V.text(g, 'later', x + colW - 10, y + 0.5, { size: 10.5, font: 'mono', color: C.muted, base: 'middle', align: 'right' });
            continue;
          }
          const on = j === S.sel;
          V.box(g, x, y - pillH / 2, colW, pillH, { r: pillH / 2, fill: V.alpha(hue(j), 0.08 + 0.45 * (a / amax)), stroke: V.alpha(hue(j), 0.4 + 0.5 * (a / amax)), lw: on ? 2 : 1.2 });
          V.text(g, toks[j], x + 12, y + 0.5, { size: 12.5, color: C.ink, base: 'middle', weight: a > 0.2 ? 600 : 400 });
          V.text(g, Math.round(a * 100) + '%', x + colW - 10, y + 0.5, { size: 11.5, font: 'mono', color: C.ink, base: 'middle', align: 'right', weight: 600 });
        }
        // coluna da esquerda (queries)
        const mixCol = blend(toks.map((_, j) => hue(j)), W.map((v) => v || 0));
        for (let i = 0; i < n; i++) {
          const on = i === S.sel, hv = i === hov;
          const bob = on ? Math.sin(t * 3) * 1.3 : 0;
          const y = yOf(i) + bob;
          if (on) {
            V.glow(g, xL + colW / 2, y, colW * 0.55, mixCol, 0.35);
            V.box(g, xL, y - pillH / 2, colW, pillH, { r: pillH / 2, fill: mixCol, stroke: C.ink, lw: 1.5 });
            const ink = inkFor(mixCol);
            V.text(g, toks[i], xL + 12, y + 0.5, { size: 12.5, color: ink, base: 'middle', weight: 700 });
            // olhinhos olhando para a direita
            const ex = xL + colW - 26, ey = y;
            for (let e = 0; e < 2; e++) {
              V.dot(g, ex + e * 10, ey, 4.2, '#ffffff');
              V.dot(g, ex + e * 10 + 1.6, ey + 0.3, 2, '#101418');
            }
          } else {
            V.box(g, xL, y - pillH / 2, colW, pillH, { r: pillH / 2, fill: V.alpha(hue(i), hv ? 0.22 : 0.09), stroke: hv ? C.ink : V.alpha(hue(i), 0.45), lw: hv ? 1.5 : 1 });
            V.text(g, toks[i], xL + 12, y + 0.5, { size: 12.5, color: C.ink, base: 'middle' });
          }
        }
        // painel lateral (telas largas)
        if (wide) {
          const px = areaW + 8, pw = w - px - 14;
          let y = top + 4;
          V.text(g, `what “${toks[S.sel]}” takes from the party`, px, y, { size: 12.5, color: C.ink, weight: 600 });
          y += 14;
          let xx = px;
          for (let j = 0; j < n; j++) {
            const a = W[j]; if (!a) continue;
            const sw = a * pw;
            V.rr(g, xx, y, Math.max(1, sw - 1), 20, 3); g.fillStyle = hue(j); g.fill();
            if (sw > 38) V.text(g, toks[j], xx + sw / 2, y + 10.5, { size: 11, color: inkFor(hue(j)), align: 'center', base: 'middle', maxW: sw - 4 });
            xx += sw;
          }
          y += 38;
          const order = W.map((v, j) => [v || 0, j]).sort((a, b) => b[0] - a[0]).slice(0, 5);
          order.forEach(([a, j]) => {
            if (a <= 0) return;
            V.rr(g, px, y - 7, 10, 10, 2); g.fillStyle = hue(j); g.fill();
            V.text(g, toks[j], px + 16, y + 1, { size: 12, color: C.ink, base: 'middle' });
            const bx = px + 86, bw = pw - 86 - 40;
            V.rr(g, bx, y - 5, bw, 8, 4); g.fillStyle = V.alpha(C.muted, 0.14); g.fill();
            V.rr(g, bx, y - 5, Math.max(2, bw * a), 8, 4); g.fillStyle = hue(j); g.fill();
            V.text(g, Math.round(a * 100) + '%', px + pw, y + 1, { size: 11.5, font: 'mono', color: C.ink, base: 'middle', align: 'right' });
            y += 22;
          });
          y += 14;
          V.glow(g, px + 20, y + 14, 26, mixCol, 0.4);
          V.dot(g, px + 20, y + 14, 12, mixCol);
          V.text(g, `new vector of “${toks[S.sel]}”`, px + 42, y + 8, { size: 12, color: C.ink, weight: 600 });
          V.text(g, 'o = Σ a_j · v_j', px + 42, y + 25, { size: 12, font: 'mono', color: C.muted });
          y += 50;
          if (S.causal && W.some((v) => v == null)) V.text(g, '✕ future tokens: weight 0 (mask)', px, y, { size: 11.5, color: C.c });
          else if (!S.causal) V.text(g, 'no mask: everyone sees everyone', px, y, { size: 11.5, color: C.d });
        }
      });
    }

    /* =============================================================
       Fig 2 — Q, K e V: três projeções do mesmo vetor (stepper)
       ============================================================= */
    {
      const fig = root.querySelector('#fig-qkv');
      autoH(fig, 400, 480);
      const toks = ['the', 'cat', 'saw', 'it'];
      const X = [[0.2, -0.5, 0.1, 0.9], [1.0, 0.3, -0.7, 0.2], [-0.4, 0.8, 0.6, -0.1], [0.9, 0.4, -0.5, 0.3]];
      const mm = (A, B) => A.map((row) => B[0].map((_, j) => row.reduce((s, a, k) => s + a * B[k][j], 0)));
      const r1 = (x) => Math.round(x * 10) / 10;
      let WQ, WK, WV, Q, K, Vm, SC;
      for (let seed = 601; seed < 1200; seed++) {
        const r = V.rng(seed);
        const mk = () => Array.from({ length: 4 }, () => Array.from({ length: 3 }, () => r1(V.clamp(r.normal() * 0.6, -1, 1))));
        WQ = mk(); WK = mk(); WV = mk();
        Q = mm(X, WQ); K = mm(X, WK); Vm = mm(X, WV);
        SC = K.map((k) => Q[3].reduce((s, q, c) => s + q * k[c], 0) / Math.sqrt(3));
        const best = SC.indexOf(Math.max(...SC));
        const srt = SC.slice().sort((a, b) => b - a);
        if (best === 1 && srt[0] - srt[1] > 0.15 && Math.abs(srt[0]) > 0.3) break;
      }
      const steps = [
        { t: 'Each token is a row', d: 'The matrix \\(X\\) has one row per token and \\(d = 4\\) columns (in real life, the <code>hidden_size</code>: 2560 in Qwen3.8, 4096 in GLM and DeepSeek). Each row still knows nothing about its neighbors.' },
        { t: 'Q = X · W_Q', d: 'Each cell of \\(Q\\) is the dot product of a row of \\(X\\) with a column of \\(W_Q\\). The lit row of \\(X\\) “passes through” the lit column of \\(W_Q\\), and the computation shows up below.' },
        { t: 'K = X · W_K', d: 'Same input, another matrix: now each token writes its name badge, what it has to offer.' },
        { t: 'V = X · W_V', d: 'And a third reading: the content each token delivers to whoever queries it.' },
        { t: 'Three roles, one row per token', d: 'The row of “it” in \\(Q\\) is compared with every row of \\(K\\): \\(q_{it}\\cdot k_j/\\sqrt{3}\\). The highest score says who “it” will listen to most, and it is “cat”.' },
        { t: 'At real size', d: 'In Qwen3.8\'s softmax attention layers (the QSA, which is sparse), \\(X\\) has 2560 columns; \\(W_Q\\) has 24 heads × 256 = 6144 columns, and \\(W_K\\) and \\(W_V\\) only 2 × 256 = 512 (GQA). Attention works in a space wider than the hidden.' },
      ];
      const st = UI.stepper(fig, steps, { autoplay: 6500 });
      const ROLE = [null, { M: WQ, R: Q, n: 'W_Q', r: 'Q', l: 'q' }, { M: WK, R: K, n: 'W_K', r: 'K', l: 'k' }, { M: WV, R: Vm, n: 'W_V', r: 'V', l: 'v' }];
      V.fig(fig, (g, w, h, t) => {
        if (w < 80 || h < 80) return;
        const C = V.C();
        const rc = [null, C.a, C.b, C.c];
        const i = st.i, el = st.elapsed();
        const wide = w >= 560;
        const roleFill = (col) => (v) => V.alpha(col, 0.14 + 0.62 * Math.min(1, Math.abs(v) / 1.2));
        const roleInk = () => C.ink;
        if (i <= 3) {
          let cw, xX, yX, xW, yW, xR, yR;
          if (wide) {
            cw = Math.min(48, (w - 70) / 13);
            xX = 24 + cw * 1.3; yX = Math.max(60, (h - 4 * cw) / 2 - 20);
            xW = xX + 4 * (cw + 2) + cw * 1.0; yW = yX;
            xR = xW + 3 * (cw + 2) + cw * 1.0; yR = yX;
          } else {
            cw = Math.min(42, (w - 36) / 9.6);
            xX = 10 + cw * 1.25; yX = 56;
            xW = xX + 4 * (cw + 2) + cw * 0.9; yW = yX;
            xR = xX + (cw + 2) * 0.5; yR = yX + 4 * (cw + 2) + 44;
          }
          const k = i === 0 ? -1 : Math.min(12, Math.floor(el / 0.34));
          const ci = k >= 0 && k < 12 ? Math.floor(k / 3) : -1, cj = k >= 0 && k < 12 ? k % 3 : -1;
          const pulse = 0.5 + 0.5 * Math.sin(t * 5);
          const a0 = Math.floor(t * 0.8) % 4;
          mat(g, X, xX, yX, cw, cw, { max: 1, rowLabels: toks, label: 'X  [4 × 4]', hiRow: i === 0 ? a0 : ci, hiColor: V.alpha(C.d, 0.6 + 0.4 * pulse), cellAlpha: (a) => (i === 0 && a !== a0 ? 0.55 : 1) });
          if (i === 0) {
            const a = a0;
            V.text(g, `row of “${toks[a]}”: its embedding vector`, xX, yX + 4 * (cw + 2) + 22, { size: 12, color: C.d });
            V.text(g, 'd = hidden_size = 4 (illustrative)', xX, yX + 4 * (cw + 2) + 40, { size: 11.5, font: 'mono', color: C.muted });
            if (wide) {
              ['Q', 'K', 'V'].forEach((nm, q) => {
                const bx = xW + q * (cw * 1.6), by = yX + 20;
                V.box(g, bx, by, cw * 1.3, 4 * (cw + 2) - 40, { r: 8, fill: V.alpha(rc[q + 1], 0.08), stroke: V.alpha(rc[q + 1], 0.5), lw: 1 });
                V.text(g, nm, bx + cw * 0.65, by + (4 * (cw + 2) - 40) / 2, { size: 16, font: 'display', color: rc[q + 1], align: 'center', base: 'middle', weight: 700 });
              });
              V.text(g, '→ three projections on the way', xW, yX - 9, { size: 12, color: C.muted });
            } else {
              // palco estreito: as três projeções esperam embaixo de X, onde os resultados vão aparecer
              const py = yX + 4 * (cw + 2) + 84, ph = Math.min(4 * (cw + 2), h - py - 16);
              const gw = (w - 24 - 2 * 12) / 3;
              V.text(g, '↓ three readings of the same X on the way', 12, py - 14, { size: 12, color: C.muted });
              ['Q', 'K', 'V'].forEach((nm, q) => {
                const bx = 12 + q * (gw + 12);
                const lift = 3 * Math.sin(t * 2 + q * 1.3);
                V.box(g, bx, py + lift, gw, ph, { r: 10, fill: V.alpha(rc[q + 1], 0.08), stroke: V.alpha(rc[q + 1], 0.5), lw: 1 });
                V.text(g, nm, bx + gw / 2, py + lift + ph / 2 - 10, { size: 18, font: 'display', color: rc[q + 1], align: 'center', base: 'middle', weight: 700 });
                V.text(g, `X · W_${nm}`, bx + gw / 2, py + lift + ph / 2 + 14, { size: 11, font: 'mono', color: C.muted, align: 'center', base: 'middle' });
              });
            }
            return;
          }
          const Rl = ROLE[i], col = rc[i];
          V.text(g, '×', xW - cw * 0.5, yX + 2 * (cw + 2), { size: 20, color: C.muted, align: 'center', base: 'middle' });
          mat(g, Rl.M, xW, yW, cw, cw, { max: 1, label: `${Rl.n}  [4 × 3]`, labelColor: col, hiCol: cj, hiColor: V.alpha(col, 0.6 + 0.4 * pulse) });
          if (wide) V.text(g, '=', xR - cw * 0.5, yX + 2 * (cw + 2), { size: 20, color: C.muted, align: 'center', base: 'middle' });
          mat(g, Rl.R, xR, yR, cw, cw, { max: 1.2, label: `${Rl.r} = X·${Rl.n}  [4 × 3]`, labelColor: col, show: (a, b) => a * 3 + b < k, colorFn: roleFill(col), textColor: roleInk, hiCell: ci >= 0 ? [ci, cj] : null, hiCellColor: col, rowLabels: wide ? null : toks });
          // a conta da célula atual
          if (ci >= 0) {
            const par = (v) => (v < 0 ? '(' + sg(v, 1) + ')' : sg(v, 1));
            const terms = X[ci].map((x, c) => `${par(x)}·${par(Rl.M[c][cj])}`);
            const lhs = `${Rl.l}[${toks[ci]},${cj + 1}] =`;
            const rhs = `${terms.join(' + ')} = ${sg(Rl.R[ci][cj], 2)}`;
            const fy = wide ? yX + 4 * (cw + 2) + 30 : yR + 4 * (cw + 2) + 24;
            if (wide && tw(g, lhs + ' ' + rhs, 12.5, 'mono') < w - 40) V.text(g, lhs + ' ' + rhs, 24, fy, { size: 12.5, font: 'mono', color: C.ink });
            else { V.text(g, lhs, 12, fy, { size: 12, font: 'mono', color: col, weight: 600 }); V.text(g, rhs, 12, fy + 18, { size: 11.5, font: 'mono', color: C.ink, maxW: w - 24 }); }
          } else if (k >= 12) {
            const fy = wide ? yX + 4 * (cw + 2) + 30 : yR + 4 * (cw + 2) + 24;
            V.text(g, `✓ ${Rl.r} done: one row per token, d_h = 3 columns`, wide ? 24 : 12, fy, { size: 12.5, color: col, weight: 600 });
          }
          return;
        }
        if (i === 4) {
          const cw = Math.min(40, (w - 70) / (9 + 2.4 + 1.3));
          const lab = cw * 1.3;
          const x0 = wide ? Math.max(16 + lab, (w - (9 * (cw + 2) + 2 * cw * 0.9 + lab)) / 2 + lab) : 10 + lab;
          const y0 = 50;
          const mats = [[Q, 'Q · question', C.a], [K, 'K · badge', C.b], [Vm, 'V · message', C.c]];
          mats.forEach(([M, nm, col], q) => {
            const x = x0 + q * (3 * (cw + 2) + cw * 0.9);
            mat(g, M, x, y0, cw, cw, { max: 1.2, label: nm, labelColor: col, colorFn: roleFill(col), textColor: roleInk, rowLabels: q === 0 ? toks : null, hiRow: q === 0 ? 3 : null, hiColor: col, fs: 11 });
          });
          // notas de "ele"
          const by = y0 + 4 * (cw + 2) + 34;
          V.text(g, 'scores of “it”:  q_it · k_j / √3', 16, by, { size: 12.5, font: 'mono', color: C.ink, weight: 600 });
          const bw = Math.min(90, (w - 40) / 4 - 10), bh = h - by - 42;
          let mx = 1e-6; SC.forEach((s) => { mx = Math.max(mx, Math.abs(s)); });
          const best = SC.indexOf(Math.max(...SC));
          const grow = V.ease.out(Math.min(1, el / 1.2));
          SC.forEach((s, j) => {
            const x = 16 + j * ((w - 32) / 4) + ((w - 32) / 4 - bw) / 2;
            const zy = by + 14 + bh / 2;
            const hh = (s / mx) * (bh / 2 - 20) * grow;
            V.rr(g, x, hh >= 0 ? zy - hh : zy, bw, Math.max(2, Math.abs(hh)), 5); g.fillStyle = j === best ? C.d : V.alpha(C.b, 0.65); g.fill();
            V.text(g, sg(s, 2), x + bw / 2, hh >= 0 ? zy - hh - 6 : zy - hh + 14, { size: 12, font: 'mono', color: C.ink, align: 'center' });
            V.text(g, toks[j], x + bw / 2, by + bh + 30, { size: 12, color: j === best ? C.d : C.muted, align: 'center', weight: j === best ? 700 : 400 });
          });
          V.line(g, 16, by + 14 + bh / 2, w - 16, by + 14 + bh / 2, { color: V.alpha(C.muted, 0.35), lw: 1 });
          return;
        }
        // i === 5: proporções reais (Qwen3.8 QSA)
        const hS = Math.min(190, h - 200);
        const y0 = 64;
        const blocks = [
          { l: 'X', s: '[n × 2560]', wv: 2560 * 0.62, col: C.muted, sub: 'one token per row' },
          { l: 'W_Q', s: '[2560 × 6144]', wv: 6144 * 0.62, col: C.a, sub: '24 heads × 256' },
          { l: 'W_K', s: '[2560 × 512]', wv: 512 * 0.62, col: C.b, sub: '2 × 256' },
          { l: 'W_V', s: '[2560 × 512]', wv: 512 * 0.62, col: C.c, sub: '2 × 256' },
        ];
        const tot = blocks.reduce((a, b) => a + b.wv, 0) + 3 * 60;
        const u = (w - 32) / tot;
        let x = 16;
        const grow = V.ease.out(Math.min(1, el / 1.4));
        blocks.forEach((b, q) => {
          const bw = Math.max(10, b.wv * u), bh = q === 0 ? hS * 0.45 : hS;
          const yy = y0 + (hS - bh) / 2;
          V.box(g, x, yy, bw * grow, bh, { r: 6, fill: V.alpha(b.col, 0.18), stroke: V.alpha(b.col, 0.8), lw: 1.3 });
          if (q === 1) for (let k2 = 1; k2 < 24; k2++) V.line(g, x + (bw * k2) / 24, yy + 3, x + (bw * k2) / 24, yy + bh - 3, { color: V.alpha(C.a, 0.35), lw: 1 });
          V.text(g, q >= 2 && bw < 34 ? b.l.slice(-1) : b.l, x + (q >= 2 ? bw / 2 : 4), yy - (q >= 2 ? 8 : 22), { size: 13, font: 'mono', color: b.col === C.muted ? C.ink : b.col, weight: 700, align: q >= 2 ? 'center' : 'left' });
          if (q < 2) V.text(g, b.s, x + 4, yy - 7, { size: 11, font: 'mono', color: C.muted });
          if (q === 3) V.text(g, '[2560 × 512] each', x + bw, yy - 26, { size: 11, font: 'mono', color: C.muted, align: 'right' });
          if (bw > tw(g, b.sub, 11.5) + 10) V.text(g, b.sub, x + bw / 2, yy + bh / 2, { size: 11.5, color: C.ink, align: 'center', base: 'middle' });
          x += bw + 60 * u;
        });
        const ty = y0 + hS + 34;
        let ty2 = para(g, 'W_K and W_V are 12× narrower than W_Q: 2 K/V heads serve 24 query heads (GQA, ch. 12).', 16, ty, w - 32, { size: 12, color: C.ink, lh: 17 });
        ty2 = para(g, 'In the checkpoint: q_proj.weight [12288, 2560] (double, because of the gate), k_proj and v_proj [512, 2560].', 16, ty2 + 3, w - 32, { size: 12, color: C.muted, lh: 17 });
        V.text(g, 'Width proportional to the number of columns.', 16, ty2 + 6, { size: 11, font: 'mono', color: C.muted });
      });
    }

    /* =============================================================
       Fig 3 — por que dividir por √d (e a QK-Norm)
       ============================================================= */
    {
      const fig = root.querySelector('#fig-escala');
      autoH(fig, 400, 580);
      const S = { lg: 8, scale: false, grow: 1, qkn: false, cur: null };
      const cache = {};
      function data(d) {
        if (cache[d]) return cache[d];
        const r = V.rng(1000 + d);
        const N = 360;
        const raw = new Float64Array(N), nrm = new Float64Array(N);
        for (let p = 0; p < N; p++) {
          let s = 0, qq = 0, kk = 0;
          for (let c = 0; c < d; c++) { const a = r.normal(), b = r.normal(); s += a * b; qq += a * a; kk += b * b; }
          raw[p] = s; nrm[p] = s / Math.sqrt((qq / d) * (kk / d));
        }
        const q = Array.from({ length: d }, () => r.normal());
        const qr = Math.sqrt(q.reduce((a, x) => a + x * x, 0) / d);
        const sc = new Float64Array(12), scn = new Float64Array(12);
        for (let j = 0; j < 12; j++) {
          let s = 0, kk = 0;
          for (let c = 0; c < d; c++) { const b = r.normal(); s += q[c] * b; kk += b * b; }
          sc[j] = s; scn[j] = s / (qr * Math.sqrt(kk / d));
        }
        return (cache[d] = { raw, nrm, sc, scn });
      }
      const ctl = UI.controls(fig);
      let H = null;
      UI.slider(ctl, { label: 'dims d', min: 1, max: 9, step: 1, value: 8, fmt: (v) => String(Math.pow(2, v)), onInput: (v) => { S.lg = v; upd(); } });
      UI.toggle(ctl, { label: 'divide by √d', value: false, onChange: (v) => { S.scale = v; upd(); } });
      UI.slider(ctl, { label: 'norms ×', min: 1, max: 4, step: 0.25, value: 1, fmt: (v) => sf(v, 2) + '×', onInput: (v) => { S.grow = v; upd(); } });
      UI.toggle(ctl, { label: 'QK-Norm', value: false, onChange: (v) => { S.qkn = v; upd(); } });
      const rStd = UI.readout(ctl, 'score std', '—');
      const rMax = UI.readout(ctl, 'max weight', '—');
      const rEnt = UI.readout(ctl, 'entropy', '—');
      function upd() {
        const d = Math.pow(2, S.lg), D = data(d);
        const f = (S.qkn ? 1 : S.grow * S.grow) / (S.scale ? Math.sqrt(d) : 1);
        const vals = Array.from(S.qkn ? D.nrm : D.raw, (x) => x * f);
        const sc = Array.from(S.qkn ? D.scn : D.sc, (x) => x * f);
        const p = softmaxArr(sc);
        const std = Math.sqrt(vals.reduce((a, x) => a + x * x, 0) / vals.length);
        const Hn = entropy2(p);
        S.cur = { d, vals, sc, p, std, H: Hn };
        rStd.set(sf(std, std < 10 ? 2 : 1)); rMax.set(Math.round(Math.max(...p) * 100) + '%'); rEnt.set(sf(Hn, 2) + ' bits');
        if (H) H.redraw();
      }
      upd();
      H = V.fig(fig, (g, w, h) => {
        if (w < 80 || h < 80) return;
        const C = V.C();
        const cur = S.cur; if (!cur) return;
        const wide = w >= 600;
        const A = wide ? { x: 16, y: 16, w: w * 0.5 - 26, h: h - 32 } : { x: 12, y: 12, w: w - 24, h: h * 0.5 - 18 };
        const B = wide ? { x: w * 0.5 + 10, y: 16, w: w * 0.5 - 26, h: h - 32 } : { x: 12, y: h * 0.5 + 6, w: w - 24, h: h * 0.5 - 18 };
        // --- histograma
        const nice = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096];
        const R = nice.find((x) => x >= Math.max(3.6, 3.2 * cur.std)) || 8192;
        const NB = 29, bins = new Array(NB).fill(0);
        cur.vals.forEach((v) => { const b = Math.floor(((v + R) / (2 * R)) * NB); if (b >= 0 && b < NB) bins[b]++; });
        const bmax = Math.max(...bins, 1);
        const px = A.x + 6, pw = A.w - 12, py = A.y + 34, ph = A.h - 70;
        V.text(g, `q·k scores of 360 pairs, d = ${cur.d}${S.scale ? ' (÷ √d)' : ''}`, A.x, A.y + 12, { size: 12.5, color: C.ink, weight: 600 });
        const xOf = (v) => px + ((v + R) / (2 * R)) * pw;
        const b3 = Math.max(px, xOf(-3)), e3 = Math.min(px + pw, xOf(3));
        g.fillStyle = V.alpha(C.a, 0.1); g.fillRect(b3, py, e3 - b3, ph);
        const bw = pw / NB;
        bins.forEach((c, b) => {
          const bh = (c / bmax) * (ph - 6);
          V.rr(g, px + b * bw + 1, py + ph - bh, bw - 2, Math.max(1, bh), 2); g.fillStyle = cur.std > 3.5 ? V.alpha(C.c, 0.8) : V.alpha(C.a, 0.8); g.fill();
        });
        V.line(g, px, py + ph, px + pw, py + ph, { color: C.muted, lw: 1 });
        [-R, 0, R].forEach((v) => V.text(g, sg(v, 0), xOf(v), py + ph + 15, { size: 11, font: 'mono', color: C.muted, align: v === -R ? 'left' : v === R ? 'right' : 'center' }));
        if (e3 - b3 > 24) V.text(g, '±3', (b3 + e3) / 2, py + 12, { size: 11, font: 'mono', color: C.a, align: 'center' });
        V.text(g, `std ≈ ${sf(cur.std, cur.std < 10 ? 2 : 1)}` + (S.scale || S.qkn ? '' : `  (√d = ${sf(Math.sqrt(cur.d), 1)})`), A.x, A.y + A.h - 4, { size: 11.5, font: 'mono', color: cur.std > 3.5 ? C.c : C.ink });
        // --- softmax de 1 query × 12 keys
        V.text(g, 'softmax: 1 query against 12 keys', B.x, B.y + 12, { size: 12.5, color: C.ink, weight: 600 });
        const qx = B.x + 4, qw = B.w - 8, qy = B.y + 40, qh = B.h - 84;
        const kb = qw / 12;
        const imax = cur.p.indexOf(Math.max(...cur.p));
        const top3 = cur.p.map((v, j) => [v, j]).sort((a, b) => b[0] - a[0]).slice(0, 3).map((x) => x[1]);
        cur.p.forEach((v, j) => {
          const bh = v * qh;
          V.rr(g, qx + j * kb + 2, qy + qh - bh, kb - 4, Math.max(1.5, bh), 3); g.fillStyle = j === imax ? C.d : V.alpha(C.b, 0.75); g.fill();
          if (top3.includes(j) && v > 0.02) V.text(g, Math.round(v * 100) + '%', qx + j * kb + kb / 2, qy + qh - bh - 5, { size: 11, font: 'mono', color: C.ink, align: 'center' });
          if (kb > 20 || j % 2 === 0) V.text(g, 'k' + (j + 1), qx + j * kb + kb / 2, qy + qh + 14, { size: 11, font: 'mono', color: C.muted, align: 'center' });
        });
        V.line(g, qx, qy + qh, qx + qw, qy + qh, { color: C.muted, lw: 1 });
        const mxp = cur.p[imax];
        const msg = mxp > 0.9 ? '✕ saturated: nearly one-hot, the other keys vanish' : cur.H > 3.3 ? 'nearly uniform: the head picks no one' : '✓ the distribution chooses, without collapsing';
        V.text(g, msg, B.x, B.y + B.h - 4, { size: 11.5, color: mxp > 0.9 ? C.c : cur.H > 3.3 ? C.d : C.a, maxW: B.w });
      }, { static: true });
    }

    /* =============================================================
       Fig 4 — softmax sem sustos: estável, LSE, online, split-K, sink
       ============================================================= */
    {
      const fig = root.querySelector('#fig-online');
      autoH(fig, 350, 520);
      const S0 = [10.5, 11.2, 3.0, 9.0, 12.0, 7.5, 11.8, 5.0];
      const VV = [-0.4, 0.3, 1.0, -0.9, 0.8, 0.1, 0.5, -0.6];
      const SINK = 11.5;
      const P = { fmt: 'fp16', c: 0 };
      const steps = [
        { t: 'Naive: direct exponential', d: 'We exponentiate each score in the chosen format and divide by the sum. In FP16, \\(e^x\\) overflows to <code>Inf</code> above \\(x \\approx 11.09\\): a score of 12 is enough, the sum becomes Inf and Inf/Inf becomes <code>NaN</code>. In BF16 and FP32 the wall sits at \\(x \\approx 88.7\\): increase \\(c\\) beyond ~77.' },
        { t: 'Stable: subtract the maximum', d: 'With \\(m = \\max_j s_j\\), each term becomes \\(e^{s_j - m} \\le 1\\). Nothing overflows in any format and the weights are the correct ones, for any \\(c\\).' },
        { t: 'Log-sum-exp', d: '\\(\\operatorname{LSE} = m + \\ln \\ell\\) sums up the row in one number. The log of each weight is \\(s_j - \\operatorname{LSE}\\), and the output is \\(\\sum_j a_j v_j\\).' },
        { t: 'Online: one block at a time', d: 'The kernel reads the keys in blocks of 4 and keeps only \\(m\\), \\(\\ell\\) and the accumulator \\(o\\). When block B arrives with a larger maximum, whatever was already summed is corrected by \\(\\alpha = e^{m_A - m}\\). At the end, \\(o/\\ell\\) gives the same output.' },
        { t: 'Split-K: blocks in parallel', d: 'Now A and B are computed by different programs, each with its own \\((\\bar o_p, \\operatorname{LSE}_p)\\). The merge weights each partial by \\(e^{\\operatorname{LSE}_p - \\operatorname{LSE}}\\): same output, exactly.' },
        { t: 'Sink: a key with no value', d: 'The sink \\(\\sigma\\) competes for the maximum and adds \\(e^{\\sigma - m}\\) to the denominator, but brings no value. The real weights now sum to less than 1 and the output shrinks: the head can “look at nothing”.' },
      ];
      const ctl = UI.controls(fig);
      let H = null;
      UI.seg(ctl, { label: 'Format', options: [{ v: 'fp32', label: 'FP32' }, { v: 'bf16', label: 'BF16' }, { v: 'fp16', label: 'FP16' }], value: 'fp16', onChange: (v) => { P.fmt = v; if (H) H.redraw(); } });
      UI.slider(ctl, { label: 'add c to scores', min: 0, max: 90, step: 1, value: 0, fmt: (v) => '+' + v, onInput: (v) => { P.c = v; if (H) H.redraw(); } });
      const st = UI.stepper(fig, steps, { autoplay: 7000 });
      const enc = (x) => LA.num.encode(P.fmt, x, 'ieee').v;
      function comp() {
        const s = S0.map((x) => x + P.c);
        const eN = s.map((x) => enc(Math.exp(x)));
        const sumN = enc(eN.reduce((a, b) => a + b, 0));
        const pN = eN.map((e) => enc(e / sumN));
        const m = Math.max(...s);
        const eS = s.map((x) => enc(Math.exp(x - m)));
        const l = eS.reduce((a, b) => a + b, 0);
        const pS = eS.map((e) => e / l);
        const pT = softmaxArr(s);
        const LSE = m + Math.log(l);
        const out = pT.reduce((a, p, j) => a + p * VV[j], 0);
        const blk = (lo, hi) => { let mm = -Infinity; for (let j = lo; j < hi; j++) mm = Math.max(mm, s[j]); let ll = 0, oo = 0; for (let j = lo; j < hi; j++) { const e = Math.exp(s[j] - mm); ll += e; oo += e * VV[j]; } return { m: mm, l: ll, o: oo, ob: oo / ll, lse: mm + Math.log(ll) }; };
        const A = blk(0, 4), B = blk(4, 8);
        const mAB = Math.max(A.m, B.m), alpha = Math.exp(A.m - mAB);
        let lB = alpha * A.l, oB = alpha * A.o;
        for (let j = 4; j < 8; j++) { const e = Math.exp(s[j] - mAB); lB += e; oB += e * VV[j]; }
        const mL = Math.max(A.lse, B.lse), LSEm = mL + Math.log(Math.exp(A.lse - mL) + Math.exp(B.lse - mL));
        const wA = Math.exp(A.lse - LSEm), wB = Math.exp(B.lse - LSEm);
        const sig = SINK + P.c, mS = Math.max(m, sig);
        const eK = s.map((x) => Math.exp(x - mS)), eSig = Math.exp(sig - mS);
        const lK = eK.reduce((a, b) => a + b, 0) + eSig;
        const pK = eK.map((e) => e / lK);
        const outK = pK.reduce((a, p, j) => a + p * VV[j], 0);
        return { s, eN, sumN, pN, m, eS, l, pS, pT, LSE, out, A, B, mAB, alpha, lB, oB, LSEm, wA, wB, sig, mS, eSig, lK, pK, outK };
      }
      H = V.fig(fig, (g, w, h, t) => {
        if (w < 80 || h < 80) return;
        const C = V.C();
        const R = comp();
        const i = st.i, el = st.elapsed();
        const wide = w >= 560;
        const nSlots = i === 5 ? 9 : 8;
        const cols = wide ? nSlots : 5, rows = Math.ceil(nSlots / cols);
        const gapX = wide ? 6 : 5;
        const sw = Math.min(wide ? 84 : 70, (w - 24 - (cols - 1) * gapX) / cols);
        const x0 = (w - (cols * sw + (cols - 1) * gapX)) / 2;
        const cardH = 54, valH = 22, barH = wide ? 128 : 70;
        const slotH = cardH + valH + barH + 26;
        const y0 = 14;
        const blockCol = (j) => (j < 4 ? C.a : C.b);
        const fmtName = LA.num.formats[P.fmt].name;
        const tileA = i === 3 ? el > 0.2 : true, tileB = i === 3 ? el > 2.6 : true;
        // valores por slot conforme o passo
        const shownP = i === 0 ? R.pN : i === 5 ? R.pK : (i === 3 && !tileB) ? null : R.pT;
        for (let k = 0; k < nSlots; k++) {
          const cx = x0 + (k % cols) * (sw + gapX), cy = y0 + Math.floor(k / cols) * slotH;
          const isSink = k === 8;
          const j = k;
          const inB = !isSink && j >= 4;
          const active = i === 3 ? (inB ? tileB : tileA) : true;
          const bc = isSink ? C.f : (i === 3 || i === 4 ? blockCol(j) : C.muted);
          if (i === 3 && ((!inB && el < 2.6 && el > 0.2) || (inB && el > 2.6))) V.glow(g, cx + sw / 2, cy + cardH / 2, sw * 0.7, bc, 0.25);
          g.globalAlpha = active ? 1 : 0.35;
          V.box(g, cx, cy, sw, cardH, { r: 9, fill: V.alpha(bc, isSink ? 0.14 : 0.08), stroke: V.alpha(bc, 0.7), lw: 1.2 });
          if (isSink) {
            V.text(g, 'sink', cx + sw / 2, cy + 15, { size: 11, font: 'mono', color: C.f, align: 'center', weight: 700 });
            V.text(g, 'σ = ' + sg(R.sig, 1), cx + sw / 2, cy + 32, { size: 11.5, font: 'mono', color: C.ink, align: 'center' });
            V.text(g, 'no value', cx + sw / 2, cy + 47, { size: 11, color: C.muted, align: 'center' });
          } else {
            V.text(g, 'k' + (j + 1) + (i === 3 || i === 4 ? (inB ? ' · B' : ' · A') : ''), cx + sw / 2, cy + 15, { size: 11, font: 'mono', color: bc === C.muted ? C.muted : bc, align: 'center', weight: 700 });
            V.text(g, 's=' + sg(R.s[j], 1), cx + sw / 2, cy + 32, { size: 11.5, font: 'mono', color: C.ink, align: 'center', maxW: sw - 6 });
            V.text(g, 'v=' + sg(VV[j], 1), cx + sw / 2, cy + 47, { size: 11, font: 'mono', color: C.muted, align: 'center' });
          }
          // valor intermediário
          let val = '', bad = false;
          if (i === 0 && !isSink) { val = sci(R.eN[j], 1); bad = !Number.isFinite(R.eN[j]); }
          else if (i === 1 && !isSink) val = sci(R.eS[j], 2);
          else if (i === 2 && !isSink) val = sg(R.s[j] - R.LSE, 2);
          else if (i === 3 && !isSink) val = active ? sci(Math.exp(R.s[j] - (inB || tileB ? R.mAB : R.A.m)), 2) : '…';
          else if (i === 4 && !isSink) val = sci(Math.exp(R.s[j] - (inB ? R.B.m : R.A.m)), 2);
          else if (i === 5) val = isSink ? sci(R.eSig, 2) : sci(Math.exp(R.s[j] - R.mS), 2);
          V.text(g, val, cx + sw / 2, cy + cardH + 15, { size: 11.5, font: 'mono', color: bad ? C.red || C.c : C.ink, align: 'center', weight: bad ? 700 : 400, maxW: sw });
          // barra de peso
          const by = cy + cardH + valH + 4;
          V.rr(g, cx + 4, by, sw - 8, barH, 5); g.fillStyle = V.alpha(C.muted, 0.08); g.fill();
          if (!isSink) {
            const pt = R.pT[j];
            g.setLineDash([3, 3]); g.strokeStyle = V.alpha(C.ink, 0.35); g.lineWidth = 1;
            V.rr(g, cx + 6, by + barH - pt * barH, sw - 12, Math.max(1, pt * barH), 4); g.stroke(); g.setLineDash([]);
          }
          if (shownP) {
            const pv = isSink ? R.eSig / R.lK : shownP[j];
            if (Number.isNaN(pv)) {
              hatch(g, cx + 4, by, sw - 8, barH, V.alpha(C.red || C.c, 0.55), 6);
              V.text(g, 'NaN', cx + sw / 2, by + barH / 2, { size: 12, font: 'mono', color: C.red || C.c, align: 'center', base: 'middle', weight: 700 });
            } else {
              const bh = V.clamp(pv, 0, 1) * barH;
              const fillc = isSink ? C.f : i === 3 || i === 4 ? blockCol(j) : C.a;
              if (isSink) hatch(g, cx + 6, by + barH - bh, sw - 12, bh, V.alpha(C.f, 0.8), 4);
              else { V.rr(g, cx + 6, by + barH - bh, sw - 12, Math.max(1, bh), 4); g.fillStyle = V.alpha(fillc, 0.8); g.fill(); }
              V.text(g, sf(pv * 100, 1) + '%', cx + sw / 2, by + barH + 13, { size: 11, font: 'mono', color: C.ink, align: 'center' });
            }
          }
          g.globalAlpha = 1;
        }
        // registradores
        const ry = y0 + rows * slotH + 4;
        const lines = [];
        const col2 = (s, c) => lines.push([s, c]);
        if (i === 0) {
          col2(`${fmtName}: Σ eˢ = ${sci(R.sumN, 2)}`, Number.isFinite(R.sumN) ? C.ink : C.c);
          const anyNaN = R.pN.some((x) => Number.isNaN(x));
          col2(anyNaN ? '✕ Inf/Inf = NaN: the weights turn to garbage and the NaN contaminates the residual' : Number.isFinite(R.sumN) ? '✓ no overflow in this format (increase c)' : '✕ the sum overflowed', anyNaN || !Number.isFinite(R.sumN) ? C.c : C.a);
          col2(`largest finite ${fmtName}: ${sci(LA.num.maxFinite(P.fmt), 2)}  →  eˣ overflows at x ≈ ${sf(Math.log(LA.num.maxFinite(P.fmt)), 2)}`, C.muted);
        } else if (i === 1) {
          col2(`m = max s_j = ${sg(R.m, 1)}    ℓ = Σ e^(s_j − m) = ${sf(R.l, 4)}`, C.ink);
          col2('✓ the largest term is e⁰ = 1: nothing overflows, in any format', C.a);
          col2('dashed = correct weights; bars = computed weights (equal)', C.muted);
        } else if (i === 2) {
          col2(`LSE = m + ln ℓ = ${sg(R.m, 1)} + ${sf(Math.log(R.l), 4)} = ${sf(R.LSE, 4)}`, C.ink);
          col2('ln a_j = s_j − LSE   (the numbers under each key)', C.muted);
          col2(`output = Σ a_j·v_j = ${sf(R.out, 4)}`, C.a);
        } else if (i === 3) {
          if (!tileB) {
            col2(`after block A:  m = ${sg(R.A.m, 1)}   ℓ = ${sf(R.A.l, 4)}   o = ${sf(R.A.o, 4)}`, C.a);
            col2('block B has not been read yet: only 3 numbers stored per query', C.muted);
          } else {
            col2(`block B: new maximum ${sg(R.mAB, 1)}  →  α = e^(m_A − m) = ${sf(R.alpha, 4)}`, C.b);
            col2(`ℓ = α·ℓ_A + Σ_B = ${sf(R.lB, 4)}    o = α·o_A + Σ_B e·v = ${sf(R.oB, 4)}`, C.ink);
            col2(`output = o/ℓ = ${sf(R.oB / R.lB, 4)}   ✓ same as the full softmax (${sf(R.out, 4)})`, C.a);
          }
        } else if (i === 4) {
          col2(`A: ō = ${sf(R.A.ob, 4)}, LSE = ${sf(R.A.lse, 4)}     B: ō = ${sf(R.B.ob, 4)}, LSE = ${sf(R.B.lse, 4)}`, C.ink);
          col2(`LSE = ln(e^LSE_A + e^LSE_B) = ${sf(R.LSEm, 4)}   weights: ${sf(R.wA, 3)} and ${sf(R.wB, 3)}`, C.muted);
          col2(`output = ${sf(R.wA, 3)}·ō_A + ${sf(R.wB, 3)}·ō_B = ${sf(R.wA * R.A.ob + R.wB * R.B.ob, 4)}   ✓ exact`, C.a);
        } else {
          const sumP = R.pK.reduce((a, b) => a + b, 0);
          col2(`m = max(s_j, σ) = ${sg(R.mS, 1)}    ℓ = Σ e^(s_j − m) + e^(σ − m) = ${sf(R.lK, 4)}`, C.ink);
          col2(`Σ real weights = ${sf(sumP, 3)} < 1   (the rest, ${sf(1 - sumP, 3)}, “goes to the sink”)`, C.f);
          col2(`output = ${sf(R.outK, 4)}  (without sink: ${sf(R.out, 4)})`, C.a);
        }
        let yy = ry + 14;
        lines.forEach(([s, c]) => { yy = para(g, s, 14, yy, w - 28, { size: wide ? 12 : 11, font: 'mono', color: c, lh: wide ? 19 : 16 }) + (wide ? 0 : 3); });
      });
    }

    /* =============================================================
       Fig 5 — a matriz de atenção, a máscara e a soma ponderada
       ============================================================= */
    {
      const fig = root.querySelector('#fig-matriz');
      autoH(fig, 495, 580);
      const n = 8, d = 4;
      const TOK = { causal: ['The', 'wise', 'king', 'was', 'afraid', 'of', 'fire', '.'], bid: ['The', 'wise', 'king', 'was', 'afraid', 'of', 'fire', '.'], bloco: ['The', 'wise', 'king', 'was', 'afraid', '[M]', '[M]', '[M]'] };
      const r = V.rng(77);
      const E = Array.from({ length: n }, () => Array.from({ length: d }, () => r.normal()));
      const EM = Array.from({ length: d }, () => r.normal() * 0.5);
      const PE = Array.from({ length: n }, (_, i) => Array.from({ length: d }, (_, c) => 0.45 * Math.sin((i + 1) * (c + 1) * 0.63)));
      const WQ = V.randMatrix(d, d, 81, 0.75), WK = V.randMatrix(d, d, 82, 0.75), WV = V.randMatrix(d, d, 83, 0.7);
      const mv = (x, W) => W[0].map((_, j) => x.reduce((s, a, k) => s + a * W[k][j], 0));
      const S = { mode: 'causal', pre: false, cache: null };
      function compute() {
        const X = E.map((e, i) => (S.mode === 'bloco' && i >= 5 ? EM : e).map((v, c) => v + PE[i][c]));
        const Q = X.map((x) => mv(x, WQ)), K = X.map((x) => mv(x, WK)), Vv = X.map((x) => mv(x, WV));
        const vis = (i, j) => (S.mode === 'bid' ? true : S.mode === 'causal' ? j <= i : i < 4 ? j <= i : true);
        const Sc = Q.map((q, i) => K.map((k, j) => (vis(i, j) ? q.reduce((s, a, c) => s + a * k[c], 0) / Math.sqrt(d) : -Infinity)));
        const A = Sc.map((row) => softmaxArr(row));
        const O = A.map((row) => Array.from({ length: d }, (_, c) => row.reduce((s, a, j) => s + a * Vv[j][c], 0)));
        let smax = 1e-6; Sc.forEach((row) => row.forEach((x) => { if (Number.isFinite(x)) smax = Math.max(smax, Math.abs(x)); }));
        let vmax = 1e-6; Vv.forEach((row) => row.forEach((x) => { vmax = Math.max(vmax, Math.abs(x)); }));
        S.cache = { Sc, A, O, Vv, vis, smax, vmax };
      }
      compute();
      const ctl = UI.controls(fig);
      let H = null;
      UI.seg(ctl, { label: 'Mask', options: [{ v: 'causal', label: 'causal (LLM)' }, { v: 'bid', label: 'bidirectional (ViT)' }, { v: 'bloco', label: 'block (DFlash/DSpark)' }], value: 'causal', onChange: (v) => { S.mode = v; compute(); if (H) H.redraw(); } });
      UI.toggle(ctl, { label: 'scores before softmax', value: false, onChange: (v) => { S.pre = v; if (H) H.redraw(); } });
      const rVis = UI.readout(ctl, 'visible keys', '—');
      let lastRow = -1;
      H = V.fig(fig, (g, w, h, t, dt, _s, HH) => {
        if (w < 80 || h < 80) return;
        const C = V.C();
        const { Sc, A, O, Vv, vis, smax, vmax } = S.cache;
        const toks = TOK[S.mode];
        const wide = w >= 600;
        const cw = wide ? Math.min(40, (h - 100) / 8, (w * 0.56 - 80) / 8) : Math.min(36, (w - 62) / 8);
        const mx = wide ? 66 : 52, my = 52;
        const gap = 2;
        const MW = 8 * (cw + gap) - gap;
        // linha escolhida: ponteiro sobre a matriz, senão ciclo automático
        const p = HH.pointer;
        let row = -1;
        if (p.inside && p.x >= mx - 50 && p.x <= mx + MW && p.y >= my && p.y <= my + MW) row = V.clamp(Math.floor((p.y - my) / (cw + gap)), 0, 7);
        const rowsAvail = S.mode === 'bloco' ? [4, 5, 6, 7] : [0, 1, 2, 3, 4, 5, 6, 7];
        if (row < 0 || (S.mode === 'bloco' && row < 4)) row = rowsAvail[Math.floor(t / 2.4) % rowsAvail.length];
        if (row !== lastRow) { lastRow = row; let nv = 0; for (let j = 0; j < 8; j++) if (vis(row, j)) nv++; rVis.set(`${toks[row]}: ${nv} of 8`); }
        // rótulos das colunas (keys)
        toks.forEach((tk, j) => V.text(g, tk, mx + j * (cw + gap) + cw / 2, my - 8, { size: 11, color: tk === '[M]' ? C.f : j === 4 && S.mode === 'bloco' ? C.d : C.muted, align: 'center', maxW: cw + 2 }));
        V.text(g, S.pre ? 'scores QKᵀ/√d + M' : 'A = softmax(QKᵀ/√d + M)', mx, 20, { size: 12, font: 'mono', color: C.ink, weight: 600 });
        if (wide) V.text(g, 'row = query · column = key', mx + MW, 20, { size: 11, color: C.muted, align: 'right' });
        const dimRow = (i) => S.mode === 'bloco' && i < 4;
        const G = mat(g, S.pre ? Sc.map((rr) => rr.map((x) => (Number.isFinite(x) ? x : null))) : A, mx, my, cw, cw, {
          gap, max: S.pre ? smax : 1, cmap: S.pre ? 'div' : 'seq', digits: S.pre ? 1 : 2,
          fmt: S.pre ? (v) => sg(v, 1) : (v) => (v < 0.005 ? '0' : sf(v, 2)),
          show: (i, j) => vis(i, j), cellAlpha: (i) => (dimRow(i) ? 0.35 : 1),
          rowLabels: toks, rowLabelColor: (i) => (toks[i] === '[M]' ? C.f : i === 4 && S.mode === 'bloco' ? C.d : i === row ? C.ink : C.muted),
          hiRow: row, hiColor: C.d, fs: cw < 30 ? 11 : 11.5,
        });
        // células mascaradas: −∞ hachurado
        for (let i = 0; i < 8; i++) for (let j = 0; j < 8; j++) if (!vis(i, j)) {
          const [cx, cy] = G.center(i, j);
          hatch(g, cx - cw / 2, cy - cw / 2, cw, cw, V.alpha(C.muted, 0.28), 5);
          if (S.pre && cw >= 26) V.text(g, '−∞', cx, cy + 0.5, { size: 11, font: 'mono', color: C.muted, align: 'center', base: 'middle' });
        }
        if (S.mode === 'bloco') {
          const yb = my + 4 * (cw + gap) - gap / 2;
          V.line(g, mx - 2, yb, mx + MW + 2, yb, { color: C.d, lw: 1.5, dash: [4, 3] });
          V.line(g, mx + 4 * (cw + gap) - gap / 2, my - 2, mx + 4 * (cw + gap) - gap / 2, my + MW + 2, { color: C.d, lw: 1.5, dash: [4, 3] });
          V.text(g, 'context: KV from the target', mx, my + MW + 16, { size: 11, color: C.muted });
          V.text(g, 'block: anchor + [M]', mx + MW, my + MW + 16, { size: 11, color: C.f, align: 'right' });
        } else {
          V.text(g, S.mode === 'causal' ? 'triangular: nobody sees the future' : 'full: everyone sees everyone', mx, my + MW + 16, { size: 11, color: C.muted });
        }
        // painel da soma ponderada
        const a = A[row];
        if (wide) {
          const px = mx + MW + 30, pw = w - px - 14;
          const cvw = Math.min(42, (pw - 136) / 4);
          V.text(g, `output of “${toks[row]}” = Σ a_j · v_j`, px, 20, { size: 12, font: 'mono', color: C.ink, weight: 600 });
          const vx = px + 128;
          V.text(g, 'V', vx + 2 * (cvw + 2), my - 8, { size: 12, font: 'mono', color: C.c, align: 'center', weight: 700 });
          const outY = my + 8 * (cw + gap) + 34;
          for (let j = 0; j < 8; j++) {
            const yj = my + j * (cw + gap) + cw / 2;
            const aj = a[j];
            V.text(g, toks[j], px, yj + 0.5, { size: 11, color: aj > 0.2 ? C.ink : C.muted, base: 'middle', maxW: 40 });
            const bx = px + 44, bwM = 46;
            V.rr(g, bx, yj - 4, bwM, 8, 4); g.fillStyle = V.alpha(C.muted, 0.12); g.fill();
            if (aj > 0) { V.rr(g, bx, yj - 4, Math.max(2, bwM * aj), 8, 4); g.fillStyle = C.d; g.fill(); }
            V.text(g, aj > 0 ? Math.round(aj * 100) + '%' : '0', bx + bwM + 32, yj + 0.5, { size: 11, font: 'mono', color: C.ink, base: 'middle', align: 'right' });
            mat(g, [Vv[j]], vx, yj - cvw * 0.42, cvw, cvw * 0.84, { max: vmax, gap: 2, fmt: (v) => sg(v, 1), fs: 11, cellAlpha: () => 0.25 + 0.75 * Math.min(1, aj * 3) });
            // mensagens descendo até a saída
            if (aj > 0.02) {
              const k = Math.max(1, Math.round(aj * 8));
              const sx = vx + 4 * (cvw + 2) + 2, sy = yj;
              const ex = vx + 4 * (cvw + 2) + 2, ey = outY + cvw * 0.42;
              const bend = 14 + 26 * (1 - j / 8);
              g.strokeStyle = V.alpha(C.d, 0.12 + 0.55 * aj); g.lineWidth = 0.8 + 3.5 * aj;
              g.beginPath(); g.moveTo(sx, sy); g.bezierCurveTo(sx + bend, sy, ex + bend, ey, ex, ey); g.stroke();
              for (let m = 0; m < k; m++) {
                const u = (t * 0.4 + m / k + j * 0.21) % 1;
                const pt = V.bez([sx, sy], [sx + bend, sy], [ex + bend, ey], [ex, ey], u);
                V.dot(g, pt[0], pt[1], 1.4 + 2 * aj, C.d);
              }
            }
          }
          V.text(g, 'o =', vx - 10, outY + cvw * 0.42 + 0.5, { size: 12, font: 'mono', color: C.d, base: 'middle', weight: 700, align: 'right' });
          mat(g, [O[row]], vx, outY, cvw, cvw * 0.84, { max: vmax, gap: 2, fmt: (v) => sg(v, 2), fs: 11 });
          V.text(g, `Σ a_j = ${sf(a.reduce((x, y) => x + y, 0), 2)}`, px, outY + cvw + 18, { size: 11, font: 'mono', color: C.muted });
        } else {
          const py0 = my + MW + 34;
          V.text(g, `output of “${toks[row]}” = Σ a_j · v_j`, 12, py0, { size: 12, font: 'mono', color: C.ink, weight: 600 });
          const rh = 15, cvw = 22;
          for (let j = 0; j < 8; j++) {
            const yj = py0 + 14 + j * rh + rh / 2;
            const aj = a[j];
            V.text(g, toks[j], 12, yj + 0.5, { size: 11, color: aj > 0.2 ? C.ink : C.muted, base: 'middle', maxW: 42 });
            const bx = 58, bwM = w - 58 - 4 * (cvw + 2) - 60;
            V.rr(g, bx, yj - 3.5, bwM, 7, 3.5); g.fillStyle = V.alpha(C.muted, 0.12); g.fill();
            if (aj > 0) { V.rr(g, bx, yj - 3.5, Math.max(2, bwM * aj), 7, 3.5); g.fillStyle = C.d; g.fill(); }
            V.text(g, aj > 0 ? Math.round(aj * 100) + '%' : '0', bx + bwM + 34, yj + 0.5, { size: 11, font: 'mono', color: C.ink, base: 'middle', align: 'right' });
            mat(g, [Vv[j]], w - 12 - 4 * (cvw + 2), yj - 5.5, cvw, 11, { max: vmax, gap: 2, values: false, cellAlpha: () => 0.25 + 0.75 * Math.min(1, aj * 3) });
          }
          const oy = py0 + 14 + 8 * rh + 12;
          V.text(g, 'o =', 12, oy + 12, { size: 12, font: 'mono', color: C.d, base: 'middle', weight: 700 });
          mat(g, [O[row]], 50, oy, 44, 24, { max: vmax, gap: 3, fmt: (v) => sg(v, 2), fs: 11 });
          V.text(g, `Σ a_j = ${sf(a.reduce((x, y) => x + y, 0), 2)}`, w - 12, oy + 12, { size: 11, font: 'mono', color: C.muted, align: 'right', base: 'middle' });
        }
      });
    }

    /* =============================================================
       Fig 6 — memória associativa: um dicionário que responde com médias
       ============================================================= */
    {
      const fig = root.querySelector('#fig-memoria');
      autoH(fig, 400, 580);
      const KEYS = [[18, 'fruit'], [78, 'fish'], [136, 'music'], [196, 'rain'], [258, 'book'], [318, 'car']].map(([a, wd]) => ({ th: (a * Math.PI) / 180, wd }));
      const S = { phi: 0.45, beta: 3, auto: true };
      const ctl = UI.controls(fig);
      UI.slider(ctl, { label: 'length of q', min: 0.5, max: 10, step: 0.5, value: 3, fmt: (v) => sf(v, 1), onInput: (v) => { S.beta = v; } });
      const tg = UI.toggle(ctl, { label: 'query spins by itself', value: true, onChange: (v) => { S.auto = v; } });
      const rE = UI.readout(ctl, 'entropy', '—');
      const rA = UI.readout(ctl, 'answer', '—');
      let lastTxt = '';
      V.fig(fig, (g, w, h, t, dt, _s, HH) => {
        if (w < 80 || h < 80) return;
        const C = V.C();
        const kc = [C.a, C.b, C.c, C.d, C.e, C.f];
        const wide = w >= 600;
        const planeW = wide ? w * 0.56 : w, planeH = wide ? h : h * 0.56;
        const cx = planeW / 2, cy = planeH / 2 + (wide ? 0 : 4);
        const Rr = Math.max(20, Math.min(planeW, planeH) / 2 - 46);
        const p = HH.pointer;
        if (p.down && p.inside && Math.hypot(p.x - cx, p.y - cy) < Rr + 50) {
          S.phi = Math.atan2(cy - p.y, p.x - cx);
          if (S.auto) { S.auto = false; const inp = tg.el.querySelector('input'); if (inp) inp.checked = false; }
        }
        if (S.auto && !LA.paused) S.phi += dt * 0.35;
        const sc = KEYS.map((k) => S.beta * Math.cos(S.phi - k.th));
        const wts = softmaxArr(sc);
        // plano
        g.setLineDash([3, 5]); g.strokeStyle = V.alpha(C.muted, 0.4); g.lineWidth = 1; g.beginPath(); g.arc(cx, cy, Rr, 0, Math.PI * 2); g.stroke(); g.setLineDash([]);
        V.line(g, cx - Rr - 10, cy, cx + Rr + 10, cy, { color: V.alpha(C.muted, 0.18), lw: 1 });
        V.line(g, cx, cy - Rr - 10, cx, cy + Rr + 10, { color: V.alpha(C.muted, 0.18), lw: 1 });
        const Lq = Rr * (0.18 + 0.82 * (S.beta - 0.5) / 9.5);
        const qx = cx + Lq * Math.cos(S.phi), qy = cy - Lq * Math.sin(S.phi);
        KEYS.forEach((k, j) => {
          const kx = cx + Rr * Math.cos(k.th), ky = cy - Rr * Math.sin(k.th);
          const a = wts[j];
          g.strokeStyle = V.alpha(kc[j], 0.1 + 0.8 * a); g.lineWidth = 0.8 + 7 * a;
          g.beginPath(); g.moveTo(qx, qy); g.lineTo(kx, ky); g.stroke();
          if (a > 0.15) V.glow(g, kx, ky, 18 + 40 * a, kc[j], 0.6);
          V.dot(g, kx, ky, 5 + 12 * a, kc[j]);
          const lx = cx + (Rr + 24) * Math.cos(k.th), ly = cy - (Rr + 24) * Math.sin(k.th);
          V.text(g, k.wd, lx, ly + 0.5, { size: 12, color: C.ink, align: Math.cos(k.th) > 0.3 ? 'left' : Math.cos(k.th) < -0.3 ? 'right' : 'center', base: 'middle', weight: a > 0.3 ? 700 : 400 });
        });
        V.arrow(g, cx, cy, qx, qy, { color: C.ink, lw: 2.4, head: 10 });
        V.dot(g, cx, cy, 3, C.ink);
        V.text(g, 'q', qx + 16 * Math.cos(S.phi - 1.2), qy - 16 * Math.sin(S.phi - 1.2), { size: 14, font: 'display', color: C.ink, weight: 700, align: 'center', base: 'middle' });
        if (!wide) V.text(g, 'drag on the circle', 12, 18, { size: 11, color: C.muted });
        else V.text(g, 'drag on the circle to aim the query', 14, 20, { size: 11.5, color: C.muted });
        // painel
        const px = wide ? planeW + 6 : 14, py = wide ? 26 : planeH + 10, pw = wide ? w - planeW - 20 : w - 28;
        V.text(g, 'weights = softmax(q · k_j)', px, py, { size: 12, font: 'mono', color: C.ink, weight: 600 });
        const rh = wide ? 25 : 19;
        KEYS.forEach((k, j) => {
          const y = py + 18 + j * rh;
          V.text(g, k.wd, px, y + 0.5, { size: 11.5, color: C.ink, base: 'middle' });
          const bx = px + 58, bw = pw - 58 - 44;
          V.rr(g, bx, y - 5, bw, 10, 5); g.fillStyle = V.alpha(C.muted, 0.12); g.fill();
          V.rr(g, bx, y - 5, Math.max(2, bw * wts[j]), 10, 5); g.fillStyle = kc[j]; g.fill();
          V.text(g, Math.round(wts[j] * 100) + '%', px + pw, y + 0.5, { size: 11.5, font: 'mono', color: C.ink, base: 'middle', align: 'right' });
        });
        const yy = py + 18 + KEYS.length * rh + 8;
        const mix = blend(kc, wts);
        V.box(g, px, yy, 36, 26, { r: 7, fill: mix, stroke: V.alpha(C.ink, 0.4), lw: 1 });
        const order = wts.map((v, j) => [v, j]).sort((a, b) => b[0] - a[0]);
        const txt = order[0][0] > 0.9 ? `≈ ${KEYS[order[0][1]].wd} (nearly exact)` : `${Math.round(order[0][0] * 100)}% ${KEYS[order[0][1]].wd} + ${Math.round(order[1][0] * 100)}% ${KEYS[order[1][1]].wd} + …`;
        V.text(g, 'answer: ' + txt, px + 44, yy + 13.5, { size: 11.5, color: C.ink, base: 'middle', maxW: pw - 46 });
        const Hn = entropy2(wts);
        const mode = order[0][0] > 0.9 ? 'hard lookup: an ordinary dictionary' : Hn > 0.85 * Math.log2(KEYS.length) ? 'soft lookup: an average of almost everything' : 'in between: a blend of the closest ones';
        V.text(g, mode, px, yy + 46, { size: 11.5, color: order[0][0] > 0.9 ? C.a : Hn > 0.85 * Math.log2(KEYS.length) ? C.d : C.b });
        const t2 = sf(Hn, 2) + ' bits';
        if (t2 !== lastTxt) { lastTxt = t2; rE.set(t2); rA.set(KEYS[order[0][1]].wd + ' ' + Math.round(order[0][0] * 100) + '%'); }
      });
    }

    /* =============================================================
       Fig 7 — muitas cabeças: tensor [H, n, n] (CSS 3D) + caminho das formas
       ============================================================= */
    {
      const fig = root.querySelector('#fig-cabecas');
      const st3 = fig.querySelector('.la-fig-stage.pad');
      autoH(fig, 300, 390, 560, '.la-fig-stage:not(.pad)');
      const NH = 6, NN = 8;
      const PATS = [
        ['previous token', (i, j) => (j === i - 1 || (i === 0 && j === 0) ? 1 : 0.03)],
        ['1st token', (i, j) => (j === 0 ? 1 : 0.05)],
        ['itself', (i, j) => (j === i ? 1 : 0.05)],
        ['average', () => 1],
        ['neighborhood', (i, j) => (i - j <= 2 ? 1 : 0.03)],
        ['keyword', (i, j) => (j === 2 || j === 5 ? 1 : j === i ? 0.3 : 0.05)],
      ];
      const AM = PATS.map(([, f]) => Array.from({ length: NN }, (_, i) => { const row = Array.from({ length: NN }, (_, j) => (j <= i ? f(i, j) : 0)); const z = row.reduce((a, b) => a + b, 0); return row.map((x) => x / z); }));
      const narrow3 = (st3.getBoundingClientRect().width || 600) < 520;
      const colorFor = (s, i, j) => {
        const C = V.C();
        if (j > i) return V.alpha(C.muted, 0.06);
        const rm = Math.max(...AM[s][i]);
        return V.alpha(C.series[s % 6], 0.1 + 0.88 * (AM[s][i][j] / rm));
      };
      const C0 = V.C();
      const T = UI.tensor3d(st3, { slabs: NH, rows: NN, cols: NN, cell: narrow3 ? 12 : 16, depthGap: narrow3 ? 34 : 46, rotate: true, rx: -14, ry: 38, color: colorFor, slabColor: (s) => C0.series[s % 6], labels: PATS.map(([nm], s) => `head ${s} · ${nm}`) });
      T.el.style.minHeight = narrow3 ? '230px' : '250px';
      T.el.style.padding = narrow3 ? '34px 10px 6px' : '40px 10px 6px';
      T.explode(1.6);
      // a fatia destacada sobe para fora da pilha (como uma carta puxada do baralho)
      const LIFT = narrow3 ? 22 : 30;
      T.slabs.forEach((sl) => { sl.style.transition = 'transform .9s var(--ease-out), opacity .6s, translate .7s var(--ease-out)'; });
      const labelsEl = T.slabs.map((sl) => sl.querySelector('.slab-label'));
      labelsEl.forEach((lb) => { if (lb) lb.style.display = 'none'; });
      const DESC = ['looks at the token right before', 'dumps attention on the 1st token (the “sink” from section 8)', 'looks at its own position', 'spreads evenly: an average of the past', 'local window: the 3 most recent tokens', 'looks for two specific positions'];
      const tag = LA.h('div', { style: { position: 'absolute', left: '14px', top: '10px', right: '14px', fontSize: '13px', color: 'var(--text-2)', pointerEvents: 'none', display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap', zIndex: 2 } });
      st3.appendChild(tag);
      let spot = 0, userSpot = null;
      const applySpot = () => {
        const cur = userSpot == null ? spot : userSpot;
        T.slabs.forEach((sl, k) => { sl.style.opacity = cur < 0 || k === cur ? 1 : 0.3; sl.style.translate = cur >= 0 && k === cur ? `0 -${LIFT}px 0` : '0 0 0'; });
        const col = cur >= 0 ? V.C().series[cur % 6] : V.C().muted;
        tag.innerHTML = cur < 0 ? '<b style="color:var(--text)">6 heads, 6 patterns</b> · each slice is the attention matrix of one head' : `<i style="display:inline-block;width:11px;height:11px;border-radius:3px;background:${col}"></i><b style="color:var(--text)">head ${cur} · ${PATS[cur][0]}</b><span>${DESC[cur]}</span>`;
      };
      applySpot();
      V.every(2600, () => { if (userSpot == null) { spot = (spot + 1) % NH; applySpot(); } });
      const repaint = () => { for (let s = 0; s < NH; s++) for (let i = 0; i < NN; i++) for (let j = 0; j < NN; j++) T.setCell(s, i, j, colorFor(s, i, j)); };
      if (LA.theme && LA.theme.onChange) { const off = LA.theme.onChange(() => setTimeout(repaint, 30)); if (typeof off === 'function') LA.onCleanup(off); }
      const MODELS = {
        glm: { l: 'GLM-5.3 · MLA', H: 64, dh: 256, d: 4096, gate: false, kv: 'K/V: come from a 512-wide latent per token (MLA, ch. 12)', o: 'o_proj [4096 × 16,384]', pad3: 66 },
        qwen: { l: 'Qwen3.8 · QSA', H: 24, dh: 256, d: 2560, gate: true, kv: 'K/V: 2 heads × 256 (GQA 12:1)', o: 'o_proj [2560 × 6144]', pad3: null, bad3: '2 KV heads are not divisible by 3' },
        ds: { l: 'DeepSeek-V4', H: 64, dh: 512, d: 4096, gate: false, groups: 8, groups3: 9, kv: 'K = V: a single 512-wide vector per token', o: 'wo_a (8 groups 4096→1024) + wo_b 8192→4096', o3: 'wo_a (9 groups, the 9th zeroed) + wo_b', pad3: 72, note3: 'TP=3: 8 groups are not divisible by 3 → 9 groups × 8 = 72 heads (3 groups, 24 heads per GPU); the 9th group is padding: wo_a/wo_b zeroed, sink −∞' },
        dflash: { l: 'DFlash2 (drafter)', H: 32, dh: 128, d: 4096, gate: false, kv: 'K/V: 8 heads × 128 (GQA 4:1)', o: 'o_proj [4096 × 4096]', pad3: 36, note3: 'TP=3: 32 and 8 are not divisible by 3 → config padded to 36/9, and the recipe runs the drafter at TP=1' },
        vaswani: { l: 'Original Transformer', H: 8, dh: 64, d: 512, gate: false, kv: 'K/V: 8 heads × 64 (MHA)', o: 'o_proj [512 × 512]', pad3: null, bad3: '8 heads are not divisible by 3 (the original Transformer never planned for TP=3)' },
      };
      const S = { m: 'glm', tp: 2 };
      const ctl = UI.controls(fig);
      let H = null;
      UI.select(ctl, { label: 'Model', options: Object.keys(MODELS).map((k) => ({ v: k, label: MODELS[k].l })), value: 'glm', onChange: (v) => { S.m = v; upd(); } });
      UI.seg(ctl, { label: 'TP', options: [1, 2, 3], value: 2, onChange: (v) => { S.tp = v; upd(); } });
      UI.select(ctl, { label: 'Highlight', options: [{ v: 'auto', label: 'one at a time (auto)' }, { v: 'all', label: 'all together' }].concat(PATS.map(([nm], s) => ({ v: String(s), label: `head ${s} · ${nm}` }))), value: 'auto', onChange: (v) => { userSpot = v === 'auto' ? null : v === 'all' ? -1 : +v; applySpot(); } });
      UI.slider(ctl, { label: 'Spread', min: 0.4, max: 2.4, step: 0.1, value: 1.6, fmt: (v) => sf(v, 1) + '×', onInput: (v) => T.explode(v) });
      const rHd = UI.readout(ctl, 'H × d_h', '—');
      const rG = UI.readout(ctl, 'per GPU', '—');
      function upd() {
        const M = MODELS[S.m];
        rHd.set(`${M.H} × ${M.dh} = ${nf(M.H * M.dh)}`);
        if (S.tp === 3 && M.H % 3) rG.set(M.pad3 ? `pads → ${M.pad3} (${M.pad3 / 3})` : 'not divisible');
        else if (S.tp === 3 && M.bad3) rG.set(`${M.H / 3} Q; KV not divisible`);
        else rG.set(`${M.H / S.tp} heads`);
        if (H) H.redraw();
      }
      upd();
      H = V.fig(fig, (g, w, h, t) => {
        if (w < 80 || h < 80) return;
        const C = V.C();
        const M = MODELS[S.m];
        const wide = w >= 560;
        const pad = 14;
        const rankCol = [C.a, C.b, C.d];
        const Hp = S.tp === 3 && M.H % 3 ? M.pad3 || M.H : M.H;
        const bad = S.tp === 3 && ((M.H % 3 && !M.pad3) || M.bad3);
        const perRank = Hp / S.tp;
        const wq = M.H * M.dh, maxv = Math.max(wq * (Hp / M.H), M.d);
        const labW = wide ? 150 : 0;
        const avail = w - 2 * pad - labW;
        const unit = avail / maxv;
        const cxm = pad + avail / 2;
        const rowsY = wide ? [36, 96, 152, 208, 266] : [42, 110, 172, 236, 316];
        const barH = 16;
        const rlab = (y, txt, col, left, up = 6) => {
          if (wide) { const two = tw(g, txt, 11, 'mono') > labW - 12; para(g, txt, w - pad - labW + 10, y + 4 - (two ? 7 : 0), labW - 12, { size: 11, font: 'mono', color: col || C.ink, lh: 14 }); }
          else V.text(g, txt, left, y - barH / 2 - up, { size: 11.5, font: 'mono', color: col || C.ink, maxW: w - 24 });
        };
        const drawBar = (y, width, col, lbl) => { V.box(g, cxm - width / 2, y - barH / 2, width, barH, { r: 4, fill: V.alpha(col, 0.3), stroke: V.alpha(col, 0.9), lw: 1.2 }); if (lbl) rlab(y, lbl, C.ink, cxm - width / 2); };
        // x
        drawBar(rowsY[0], M.d * unit, C.muted, `x: d = ${nf(M.d)}`);
        // cabeças q (H·dh) coloridas por GPU
        const segW = M.dh * unit;
        const qW = Hp * segW, qx0 = cxm - qW / 2;
        for (let hh = 0; hh < Hp; hh++) {
          const x = qx0 + hh * segW;
          const padH = hh >= M.H;
          const rk = Math.min(S.tp - 1, Math.floor(hh / perRank));
          const col = bad && !M.pad3 ? C.c : rankCol[rk];
          if (padH) { hatch(g, x, rowsY[1] - barH / 2, Math.max(1, segW - (segW > 4 ? 1 : 0)), barH, V.alpha(C.muted, 0.7), 3); continue; }
          g.fillStyle = V.alpha(col, hh % 2 ? 0.55 : 0.85);
          g.fillRect(x, rowsY[1] - barH / 2, Math.max(1, segW - (segW > 4 ? 1 : 0)), barH);
        }
        rlab(rowsY[1], `q: ${M.H} × ${M.dh} = ${nf(M.H * M.dh)}` + (Hp !== M.H ? ` +${Hp - M.H} padding` : ''), C.ink, qx0);
        V.arrow(g, cxm, rowsY[0] + barH / 2 + 2, cxm, rowsY[1] - barH / 2 - (wide ? 3 : 20), { color: V.alpha(C.muted, 0.7), lw: 1.4, head: 6 });
        V.text(g, 'W_Q', cxm + 8, (rowsY[0] + rowsY[1]) / 2 + (wide ? 0 : -6), { size: 11, font: 'mono', color: C.muted });
        // atenção por cabeça: pulsos descendo
        const step = Math.max(1, Math.ceil(Hp / 32));
        for (let hh = 0; hh < M.H; hh += step) {
          const x = qx0 + hh * segW + segW / 2;
          const rk = Math.min(S.tp - 1, Math.floor(hh / perRank));
          const col = bad && !M.pad3 ? C.c : rankCol[rk];
          V.line(g, x, rowsY[1] + barH / 2 + 2, x, rowsY[3] - barH / 2 - 2, { color: V.alpha(col, 0.16), lw: 1 });
          const u = (t * 0.55 + hh * 0.137) % 1;
          const y = V.lerp(rowsY[1] + barH / 2 + 2, rowsY[3] - barH / 2 - 2, u);
          V.dot(g, x, y, Math.min(2.6, Math.max(1.3, segW * 0.3)), col);
          if (segW >= 14) {
            const s = Math.min(segW - 4, 14), gx = x - s / 2, gy = rowsY[2] - s / 2;
            for (let a = 0; a < 3; a++) for (let b = 0; b <= a; b++) { g.fillStyle = V.alpha(col, 0.35 + 0.2 * ((a + b + hh) % 3)); g.fillRect(gx + b * (s / 3), gy + a * (s / 3), s / 3 - 1, s / 3 - 1); }
          }
        }
        if (wide) V.text(g, 'attention per head', w - pad - labW + 10, rowsY[2] + 4, { size: 11.5, color: C.muted });
        // concat (+gate)
        for (let hh = 0; hh < Hp; hh++) {
          const x = qx0 + hh * segW, padH = hh >= M.H;
          const rk = Math.min(S.tp - 1, Math.floor(hh / perRank));
          if (padH) { hatch(g, x, rowsY[3] - barH / 2, Math.max(1, segW - (segW > 4 ? 1 : 0)), barH, V.alpha(C.muted, 0.7), 3); continue; }
          g.fillStyle = V.alpha(bad && !M.pad3 ? C.c : rankCol[rk], hh % 2 ? 0.4 : 0.65);
          g.fillRect(x, rowsY[3] - barH / 2, Math.max(1, segW - (segW > 4 ? 1 : 0)), barH);
        }
        if (M.gate) {
          const gw = M.H * segW;
          g.fillStyle = V.alpha(C.f, 0.5 + 0.25 * Math.sin(t * 2));
          g.fillRect(cxm - gw / 2, rowsY[3] - barH / 2 - 7, gw, 4);
        }
        const nG = M.groups ? (S.tp === 3 && M.groups3 ? M.groups3 : M.groups) : 0;
        if (nG) {
          const gw = (M.H / M.groups) * segW;
          for (let k = 0; k < nG; k++) { const x = qx0 + k * gw; V.line(g, x + 1, rowsY[3] + barH / 2 + 4, x + gw - 1, rowsY[3] + barH / 2 + 4, { color: k >= M.groups ? V.alpha(C.muted, 0.8) : C.e, lw: 2, dash: k >= M.groups ? [3, 3] : null }); }
        }
        rlab(rowsY[3], `concat ${nf(M.H * M.dh)}` + (M.gate ? ' ⊙ σ(gate)' : '') + (nG ? ` · ${nG} groups` + (nG > M.groups ? ` (${nG - M.groups} extra)` : ` of ${M.H / M.groups}`) : ''), M.gate ? C.f : C.ink, qx0, M.gate ? 14 : 6);
        // o_proj → y
        const yW = M.d * unit;
        g.fillStyle = V.alpha(C.e, 0.12);
        g.beginPath(); g.moveTo(qx0, rowsY[3] + barH / 2 + 7); g.lineTo(qx0 + M.H * segW, rowsY[3] + barH / 2 + 7); g.lineTo(cxm + yW / 2, rowsY[4] - barH / 2 - 2); g.lineTo(cxm - yW / 2, rowsY[4] - barH / 2 - 2); g.closePath(); g.fill();
        drawBar(rowsY[4], yW, C.e, `y: ${nf(M.d)} → residual`);
        V.text(g, S.tp === 3 && M.o3 ? M.o3 : M.o, cxm, (rowsY[3] + rowsY[4]) / 2 + 5, { size: 11.5, font: 'mono', color: C.e, align: 'center', weight: 600, maxW: w - 24 });
        // rodapé
        const fy = h - 10;
        const warn = S.tp === 3 && M.note3 ? M.note3 : bad ? `✕ TP=3: ${M.bad3 || M.H + ' heads are not divisible by 3'}` : S.tp === 3 && Hp !== M.H ? `TP=3: ${M.H} is not divisible by 3 → padded to ${Hp} (${perRank} per GPU), extra heads with zero weights` : `TP=${S.tp}: ${perRank} heads per GPU · ${M.kv}`;
        if (wide) V.text(g, warn, pad, fy, { size: 11.5, color: bad ? C.c : Hp !== M.H ? C.d : C.muted, maxW: w - 2 * pad });
        else para(g, warn, pad, h - 34, w - 2 * pad, { size: 11, color: bad ? C.c : Hp !== M.H ? C.d : C.muted, lh: 14 });
      });
    }

    /* =============================================================
       Fig 8 — o preço de conversar com todo mundo: O(n²)
       ============================================================= */
    {
      const fig = root.querySelector('#fig-quadratico');
      autoH(fig, 420, 690);
      // kvDims: números guardados por token e por camada (o que o cache realmente precisa, em BF16)
      const MQ = {
        qwen: { l: 'Qwen3.8 · QSA', H: 24, dh: 256, dv: 256, kvDims: 2 * (256 + 256), kvL: '2 KV heads × (K 256 + V 256)' },
        glm: { l: 'GLM-5.3 · MLA', H: 64, dh: 256, dv: 256, kvDims: 512, kvL: '512-wide latent, shared by the 64 heads' },
        ds: { l: 'DeepSeek-V4', H: 64, dh: 512, dv: 512, kvDims: 512, kvL: 'one 512-wide vector that is both K and V' },
      };
      const S = { e: 18, m: 'qwen', sparse: false };
      const ctl = UI.controls(fig);
      let H = null;
      UI.slider(ctl, { label: 'context n', min: 10, max: 20, step: 0.5, value: 18, fmt: (v) => nf(Math.round(Math.pow(2, v))), onInput: (v) => { S.e = v; upd(); } });
      UI.seg(ctl, { label: 'Geometry', options: Object.keys(MQ).map((k) => ({ v: k, label: MQ[k].l })), value: 'qwen', onChange: (v) => { S.m = v; upd(); } });
      UI.toggle(ctl, { label: 'sparse shortcut (top-2048)', value: false, onChange: (v) => { S.sparse = v; upd(); } });
      const rP = UI.readout(ctl, 'pairs per head·layer', '—');
      const rF = UI.readout(ctl, 'FLOPs per layer', '—');
      const rM = UI.readout(ctl, 'scores of 1 head (BF16)', '—');
      const pairsD = (n) => (n * (n + 1)) / 2;
      const pairsS = (n) => (n <= 2048 ? pairsD(n) : pairsD(2048) + (n - 2048) * 2048);
      function upd() {
        const n = Math.round(Math.pow(2, S.e)), M = MQ[S.m];
        const pr = S.sparse ? pairsS(n) : pairsD(n);
        rP.set(sci(pr, 2)); rF.set(sci(2 * (M.dh + M.dv) * M.H * pr, 2)); rM.set(S.sparse ? 'not applicable' : bytesI(n * n * 2));
        if (H) H.redraw();
      }
      upd();
      H = V.fig(fig, (g, w, h, t) => {
        if (w < 80 || h < 80) return;
        const C = V.C();
        const M = MQ[S.m];
        const n = Math.round(Math.pow(2, S.e));
        const wide = w >= 620;
        const PL = wide ? { x: 58, y: 24, w: w * 0.6 - 72, h: h - 86 } : { x: 50, y: 24, w: w - 66, h: h * 0.56 - 64 };
        const lx = (e) => PL.x + ((e - 10) / 10) * PL.w;
        const yMin = 3, yMax = 12;
        const ly = (v) => PL.y + PL.h - ((Math.log10(Math.max(v, 1)) - yMin) / (yMax - yMin)) * PL.h;
        // grade
        for (let k = yMin; k <= yMax; k++) { const y = ly(Math.pow(10, k)); V.line(g, PL.x, y, PL.x + PL.w, y, { color: V.alpha(C.muted, 0.12), lw: 1 }); if (k % (wide ? 1 : 2) === 0 || k === yMax) V.text(g, '10' + sup(k), PL.x - 6, y + 4, { size: 11, font: 'mono', color: C.muted, align: 'right' }); }
        [10, 12, 14, 16, 18, 20].forEach((e) => { const x = lx(e); V.line(g, x, PL.y, x, PL.y + PL.h, { color: V.alpha(C.muted, 0.1), lw: 1 }); V.text(g, e === 10 ? '1k' : e === 20 ? '1M' : nf(Math.pow(2, e) / 1024) + 'k', x, PL.y + PL.h + 15, { size: 11, font: 'mono', color: C.muted, align: 'center' }); });
        V.text(g, 'n (tokens)', PL.x + PL.w - 4, PL.y + PL.h - 6, { size: 11, color: C.muted, align: 'right' });
        // 128 GiB (137 GB) para as notas de 1 cabeça em BF16 (n = 2^18)
        if (!S.sparse) { const x = lx(18); V.line(g, x, PL.y, x, PL.y + PL.h, { color: C.c, lw: 1.4, dash: [5, 4] }); V.text(g, 'scores of 1 head = 128 GiB (137 GB)', x - 4, PL.y + 12, { size: 11, color: C.c, align: 'right' }); }
        const curve = (f, col, lw, dash) => { g.beginPath(); for (let k = 0; k <= 60; k++) { const e = 10 + (10 * k) / 60; const x = lx(e), y = ly(f(Math.pow(2, e))); if (k) g.lineTo(x, y); else g.moveTo(x, y); } g.strokeStyle = col; g.lineWidth = lw; if (dash) g.setLineDash(dash); g.stroke(); g.setLineDash([]); };
        curve((x) => x, C.a, 2.5);
        curve(pairsD, S.sparse ? V.alpha(C.c, 0.35) : C.c, 2.5, S.sparse ? [4, 4] : null);
        if (S.sparse) curve(pairsS, C.d, 2.8);
        // marcador
        const xm = lx(S.e);
        const pulse = 0.5 + 0.5 * Math.sin(t * 3);
        V.line(g, xm, PL.y, xm, PL.y + PL.h, { color: V.alpha(C.ink, 0.35), lw: 1 });
        const pts = [[n, C.a, 'KV: ' + sci(n, 1) + ' entries'], [S.sparse ? pairsS(n) : pairsD(n), S.sparse ? C.d : C.c, 'pairs: ' + sci(S.sparse ? pairsS(n) : pairsD(n), 1)]];
        pts.forEach(([v, col, lbl]) => { const y = ly(v); V.glow(g, xm, y, 12 + 6 * pulse, col, 0.6); V.dot(g, xm, y, 4, col); const right = xm < PL.x + PL.w * 0.62; V.text(g, lbl, right ? xm + 9 : xm - 9, y - 7, { size: 11.5, font: 'mono', color: col, align: right ? 'left' : 'right', weight: 600 }); });
        // legenda
        const lgY = PL.y + PL.h + 34;
        const items = [[C.a, 'KV entries (grows with n)'], [C.c, 'q·k pairs (grows with n²)']].concat(S.sparse ? [[C.d, 'sparse: ≤ 2048 per query']] : []);
        const lx0 = PL.x - (wide ? 40 : 36), lxMax = wide ? w * 0.6 : w - 8;
        let lxp = lx0, lyp = lgY;
        items.forEach(([col, s]) => { const iw = tw(g, s, 11) + 40; if (lxp + iw - 18 > lxMax && lxp > lx0) { lxp = lx0; lyp += 16; } V.line(g, lxp, lyp - 4, lxp + 16, lyp - 4, { color: col, lw: 3 }); V.text(g, s, lxp + 22, lyp, { size: 11, color: C.ink }); lxp += iw; });
        // painel: números + triângulo de conversas
        const px = wide ? w * 0.6 + 6 : 14, py = wide ? 30 : h * 0.56 + 54, pw = wide ? w * 0.4 - 20 : w - 28;
        V.text(g, `${M.l}: H = ${M.H}, d_h = ${M.dh}`, px, py, { size: 12, color: C.ink, weight: 600, maxW: pw });
        const pr = S.sparse ? pairsS(n) : pairsD(n);
        const lines = [
          [`n = ${nf(n)} tokens`, C.ink],
          [`attention FLOPs/layer ≈ ${sci(2 * (M.dh + M.dv) * M.H * pr, 2)}`, S.sparse ? C.d : C.c],
          [S.sparse ? 'scores: only the selected ones (≤ 2048 per query)' : `scores of 1 head in BF16: ${bytesI(n * n * 2)} (${bytesD(n * n * 2)})`, S.sparse ? C.d : C.c],
          [`KV of 1 layer in BF16: ${bytesI(n * M.kvDims * 2)} (${M.kvL})`, C.a],
        ];
        let yy = py + 22;
        lines.forEach(([s, c]) => { yy = para(g, s, px, yy, pw, { size: 11.5, font: 'mono', color: c, lh: 16 }) + 5; });
        // triângulo animado: cada ponto = uma conversa (n reduzido)
        const m = Math.round(V.clamp((S.e - 8) * 1.6, 4, 18));
        const tri = Math.min((pw - 10) / m, (h - yy - 40) / m, 14);
        if (tri >= 4) {
          const tx = px + 4, ty = yy + 18;
          V.text(g, `illustration: ${m} tokens → ${m * (m + 1) / 2} conversations`, px, yy + 6, { size: 11, color: C.muted });
          const total = (m * (m + 1)) / 2;
          const lit = Math.floor(((t * 0.35) % 1.25) * total);
          let c = 0;
          for (let a = 0; a < m; a++) for (let b = 0; b <= a; b++) {
            const on = c < lit; c++;
            V.dot(g, tx + b * tri + tri / 2, ty + a * tri + tri / 2, Math.max(1.2, tri * 0.3), on ? V.alpha(S.sparse && a - b >= 4 ? C.muted : C.c, S.sparse && a - b >= 4 ? 0.25 : 0.9) : V.alpha(C.muted, 0.18));
          }
          for (let a = 0; a < m; a++) V.dot(g, tx + a * tri + tri / 2, ty + m * tri + 6, Math.max(1.5, tri * 0.32), C.a);
        }
      });
    }

    /* =============================================================
       Fig 9 — quando não há nada para ver: pura, despejo, sink, gate
       ============================================================= */
    {
      const fig = root.querySelector('#fig-sink');
      autoH(fig, 420, 640);
      const toks = ['<s>', 'the', 'sky', 'is', 'blue', 'and', 'the', 'sea'];
      const base = [0.1, -0.2, 0.3, 0.0, -0.1, 0.2, -0.3, 0.1];
      const r = V.rng(909);
      const VV = toks.map(() => Array.from({ length: 4 }, () => r.normal() * 0.9));
      const V0 = [0.04, -0.03, 0.02, 0.01];
      const S = { mode: 'pura', sig: 5, g: -3 };
      const sigOf = () => (S.sig <= -6 ? -Infinity : S.sig);
      function comp(sc0, mode = S.mode) {
        const s = sc0.slice(); if (mode === 'bos') s[0] = 5.0;
        const vals = VV.map((v, j) => (mode === 'bos' && j === 0 ? V0 : v));
        const useSink = mode === 'sink', sig = sigOf();
        let m = Math.max(...s); if (useSink) m = Math.max(m, sig);
        const e = s.map((x) => Math.exp(x - m));
        const es = useSink ? Math.exp(sig - m) : 0;
        const den = e.reduce((a, b) => a + b, 0) + es;
        const p = e.map((x) => x / den);
        const gate = mode === 'gate' ? 1 / (1 + Math.exp(-S.g)) : 1;
        const o = [0, 1, 2, 3].map((c) => gate * p.reduce((a, pj, j) => a + pj * vals[j][c], 0));
        return { p, sinkW: es / den, o, gate, vals, sum: p.reduce((a, b) => a + b, 0), norm: Math.hypot(...o) };
      }
      const ctl = UI.controls(fig);
      let H = null;
      UI.seg(ctl, { label: 'Situation', options: [{ v: 'pura', label: 'pure' }, { v: 'bos', label: 'dump on 1st token' }, { v: 'sink', label: 'sink σ' }, { v: 'gate', label: 'gate σ(g)' }], value: 'pura', onChange: (v) => { S.mode = v; upd(); } });
      UI.slider(ctl, { label: 'sink logit', min: -6, max: 8, step: 0.5, value: 5, fmt: (v) => (v <= -6 ? '−∞' : sg(v, 1)), onInput: (v) => { S.sig = v; upd(); } });
      UI.slider(ctl, { label: 'gate g', min: -6, max: 6, step: 0.5, value: -3, fmt: (v) => sg(v, 1) + ' → ' + sf(1 / (1 + Math.exp(-v)), 2), onInput: (v) => { S.g = v; upd(); } });
      const rS = UI.readout(ctl, 'Σ weights', '—');
      const rN = UI.readout(ctl, '‖output‖', '—');
      const rows = Array.from({ length: 6 }, (_, i) => base.map((x, j) => x + 0.28 * Math.sin(i * 1.7 + j * 1.3)));
      const pureNorm = comp(base, 'pura').norm;
      function upd() { const R = comp(base); rS.set(sf(R.sum, 3)); rN.set(sf(R.norm, 3)); if (H) H.redraw(); }
      upd();
      H = V.fig(fig, (g, w, h, t) => {
        if (w < 80 || h < 80) return;
        const C = V.C();
        const R = comp(base);
        const wide = w >= 600;
        const A = wide ? { x: 16, y: 16, w: w * 0.46 - 22, h: h - 32 } : { x: 12, y: 12, w: w - 24, h: 220 };
        const Bm = wide ? { x: w * 0.46 + 10, y: 16, w: w * 0.54 - 26, h: (h - 32) * 0.45 } : { x: 12, y: 246, w: w - 24, h: 150 };
        const Bo = wide ? { x: w * 0.46 + 10, y: 16 + (h - 32) * 0.5, w: w * 0.54 - 26, h: (h - 32) * 0.5 } : { x: 12, y: 412, w: w - 24, h: h - 424 };
        // --- pesos da query "mar"
        V.text(g, 'weights of the query for “sea”', A.x, A.y + 12, { size: 12.5, color: C.ink, weight: 600 });
        const nb = S.mode === 'sink' ? 9 : 8;
        const bx = A.x + 4, bwA = A.w - 8, by = A.y + 36, bh = A.h - 90;
        const cw = bwA / nb;
        const pmax = Math.max(R.sinkW, ...R.p);
        const ymax = pmax <= 0.25 ? 0.25 : pmax <= 0.5 ? 0.5 : 1;
        V.text(g, `scale 0–${Math.round(ymax * 100)}%`, A.x + A.w, A.y + 12, { size: 11, font: 'mono', color: C.muted, align: 'right' });
        for (let j = 0; j < nb; j++) {
          const isS = j === 8;
          const v = isS ? R.sinkW : R.p[j];
          const hh = Math.min(1, v / ymax) * bh;
          const x = bx + j * cw + 3, ww = cw - 6;
          V.rr(g, x, by, ww, bh, 4); g.fillStyle = V.alpha(C.muted, 0.07); g.fill();
          if (isS) { hatch(g, x, by + bh - hh, ww, hh, V.alpha(C.f, 0.85), 4); V.rr(g, x, by + bh - hh, ww, Math.max(1, hh), 4); g.strokeStyle = C.f; g.lineWidth = 1.2; g.stroke(); }
          else { V.rr(g, x, by + bh - hh, ww, Math.max(1.5, hh), 4); g.fillStyle = j === 0 && S.mode === 'bos' ? C.d : V.alpha(C.b, 0.8); g.fill(); }
          V.text(g, Math.round(v * 100) + '%', x + ww / 2, by + bh - hh - 5, { size: 11, font: 'mono', color: C.ink, align: 'center' });
          V.text(g, isS ? 'sink' : toks[j], x + ww / 2, by + bh + 15, { size: 11, color: isS ? C.f : C.muted, align: 'center', maxW: cw });
        }
        const note = { pura: 'the softmax forces the weights to sum to 1', bos: 'high score learned on the 1st token, value ≈ 0', sink: 'the missing mass goes to the sink', gate: 'weights intact: the gate shuts the output' }[S.mode];
        V.text(g, `Σ weights = ${sf(R.sum, 3)} · ${note}`, A.x, A.y + A.h - 10, { size: 11.5, color: S.mode === 'sink' ? C.f : S.mode === 'bos' ? C.d : C.muted, maxW: A.w });
        // --- mini-mapa: 6 queries sem nada para achar
        V.text(g, 'attention map: 6 queries with nothing to find', Bm.x, Bm.y + 12, { size: 12, color: C.ink, weight: 600, maxW: Bm.w });
        const ncol = S.mode === 'sink' ? 9 : 8;
        const cwm = Math.min(26, (Bm.w - 10) / ncol), chm = Math.min(16, (Bm.h - 40) / 6);
        for (let i = 0; i < 6; i++) {
          const Rr = comp(rows[i]);
          for (let j = 0; j < ncol; j++) {
            const v = j === 8 ? Rr.sinkW : Rr.p[j];
            const x = Bm.x + j * (cwm + 2), y = Bm.y + 26 + i * (chm + 2);
            V.rr(g, x, y, cwm, chm, 3); g.fillStyle = j === 8 ? V.alpha(C.f, 0.15 + 0.8 * v) : V.cmap(Math.min(1, v * 2.2), 'seq'); g.fill();
          }
        }
        if (cwm > 16) { V.text(g, '<s>', Bm.x + cwm / 2, Bm.y + 26 + 6 * (chm + 2) + 12, { size: 11, color: C.muted, align: 'center' }); if (ncol === 9) V.text(g, 'sink', Bm.x + 8 * (cwm + 2) + cwm / 2, Bm.y + 26 + 6 * (chm + 2) + 12, { size: 11, color: C.f, align: 'center' }); }
        // --- saída e escrita no resíduo
        V.text(g, 'output o (4 dims) → written to the residual', Bo.x, Bo.y + 12, { size: 12, color: C.ink, weight: 600, maxW: Bo.w });
        const ox = Bo.x + 4, oy = Bo.y + 28, oh = Math.min(Bo.h - 60, 110), ow = Math.min(Bo.w * 0.5, 200);
        const zy = oy + oh / 2;
        V.line(g, ox, zy, ox + ow, zy, { color: V.alpha(C.muted, 0.4), lw: 1 });
        const obw = ow / 4;
        R.o.forEach((v, c) => {
          const hh = V.clamp(v / 1.2, -1, 1) * (oh / 2 - 4);
          V.rr(g, ox + c * obw + 4, hh >= 0 ? zy - hh : zy, obw - 8, Math.max(1.5, Math.abs(hh)), 3); g.fillStyle = v >= 0 ? C.a : C.b; g.fill();
          V.text(g, sg(v, 2), ox + c * obw + obw / 2, oy + oh + 14, { size: 11, font: 'mono', color: C.ink, align: 'center' });
        });
        // seta para o resíduo com espessura ∝ ‖o‖ + torneira do gate
        const ax = ox + ow + 18, aw = Bo.x + Bo.w - ax - 6;
        if (aw > 40) {
          const lw = 1 + 16 * Math.min(1, R.norm / 0.8);
          const u = (t * 0.6) % 1;
          g.strokeStyle = V.alpha(C.e, 0.25 + 0.6 * Math.min(1, R.norm / 0.6)); g.lineWidth = lw; g.lineCap = 'round';
          g.beginPath(); g.moveTo(ax, zy); g.lineTo(ax + aw - 10, zy); g.stroke(); g.lineCap = 'butt';
          if (R.norm > 0.03) V.dot(g, ax + u * (aw - 10), zy, 2 + lw * 0.25, C.e);
          V.text(g, 'residual', ax + aw - 10, zy - lw / 2 - 8, { size: 11, color: C.e, align: 'right' });
          V.text(g, `‖o‖ = ${sf(R.norm, 3)}`, ax, zy + Math.max(lw / 2, 14) + 22, { size: 11.5, font: 'mono', color: C.ink });
          if (S.mode === 'gate') {
            const vx = ax + Math.min(40, aw * 0.3), vy = zy;
            V.dot(g, vx, vy, 11, V.alpha(C.f, 0.25));
            g.strokeStyle = C.f; g.lineWidth = 2; g.beginPath(); g.arc(vx, vy, 11, 0, Math.PI * 2); g.stroke();
            const ang = (1 - R.gate) * (Math.PI / 2);
            V.line(g, vx - 13 * Math.cos(ang), vy - 13 * Math.sin(ang), vx + 13 * Math.cos(ang), vy + 13 * Math.sin(ang), { color: C.f, lw: 3 });
            V.text(g, `σ(g) = ${sf(R.gate, 2)}`, vx, vy - 20, { size: 11, font: 'mono', color: C.f, align: 'center' });
          }
        }
        // veredito relativo: comparado com a mesma cabeça na softmax pura
        const rel = R.norm / Math.max(1e-9, pureNorm);
        const quiet = rel < 0.15;
        const cut = S.mode === 'pura' ? '' : ` (‖o‖ ${rel <= 1 ? '−' + Math.round((1 - rel) * 100) : '+' + Math.round((rel - 1) * 100)}% vs. pure)`;
        const verdict = (quiet ? '✓ the head stayed quiet' : '✕ the head writes noise into the residual') + cut;
        V.text(g, verdict, Bo.x, Bo.y + Bo.h - 4, { size: 11.5, color: quiet ? C.a : C.c, maxW: Bo.w });
      });
    }

    /* =============================================================
       Fig 10 — a ficha de atenção de cada modelo
       ============================================================= */
    {
      const fig = root.querySelector('#fig-ficha');
      autoH(fig, 450, 830);
      const F = {
        glm: { t: 'GLM-5.3-Flash · MLA (11 layers + MTP)', rb: 'glm', H: 64, kv: 'latent', kvN: 1, kvL: '512 latent', rows: [
          ['query heads', '64'], ['K/V heads', '64 declared; the cache stores one 512-wide latent'], ['q·k / v dims', '256 (no RoPE) / 256'], ['scale', '1/√256 = 0.0625'], ['mask', 'causal + top-2048 selection (ch. 13)'],
          ['sink', 'not declared'], ['output gate', 'no'], ['QK-Norm', 'on the latents: q_a_layernorm, kv_a_layernorm'], ['q_proj', 'q_a [1536 × 4096] → q_b [16,384 × 1536]'], ['o_proj', '[4096 × 16,384]'],
          ['KV/token/layer', '656 B in vLLM (fp8_ds_mla, with zeroed RoPE); 528 B in TensorFold (e4m3 + 2^k scale)']],
          tp: { 1: '64 heads on one GPU', 2: '32 heads per GPU, in vLLM and in TensorFold (which only runs TP=2)', 3: '64 is not divisible by 3 → pads to 66 (22 per GPU); vLLM only' }, pad3: 66 },
        qwen: { t: 'Qwen3.8-Flash-Next · QSA (12 layers + MTP)', rb: 'qwen', H: 24, kv: 'gqa', kvN: 2, rows: [
          ['query heads', '24'], ['K/V heads', '2 (GQA 12:1)'], ['q·k / v dims', '256 (RoPE on 64) / 256'], ['scale', '1/√256 = 0.0625 (× log₂e in the kernel)'], ['mask', 'causal + ≤ 2048 selection (ch. 13)'],
          ['sink', 'no (interpretation: the gate plays that role)'], ['output gate', 'yes: σ(g) per head and dimension'], ['QK-Norm', 'q_norm, k_norm [256], before RoPE'], ['q_proj', '[12,288 × 2560] = query + gate'], ['o_proj', '[2560 × 6144]'],
          ['KV/token/layer', '2 × 2 × 256 × 2 B = 2,048 B (BF16); 1,024 B in FP8']],
          tp: { 1: '24 Q + 2 KV on one GPU', 2: '12 Q + 1 KV per GPU', 3: '24 Q ÷ 3 = 8, but 2 KV heads are not divisible by 3: vLLM only replicates KV if TP is a multiple of them' }, bad3: true },
        ds: { t: 'DeepSeek-V4-Flash · attention (43 layers)', rb: 'ds', H: 64, kv: 'kv1', kvN: 1, kvL: 'K = V (512)', rows: [
          ['query heads', '64'], ['K/V heads', '1: the same vector is K and V'], ['q·k / v dims', '512 (448 + 64 RoPE) / 512'], ['scale', '1/√512 ≈ 0.0442'], ['mask', 'causal: 128 window + compressed entries'],
          ['sink', 'yes: attn_sink, 64 FP32 logits'], ['output gate', 'no'], ['QK-Norm', 'q: per-head RMS, no gain; kv_norm [512]'], ['q_proj', 'wq_a 4096 → 1024 · wq_b 1024 → 32,768'], ['o_proj', 'wo_a in 8 groups (4096 → 1024) · wo_b 8192 → 4096'],
          ['KV/token/layer', 'one 512-wide vector; 584 B per cache entry (ch. 23)']],
          tp: { 1: '64 heads on one GPU', 2: '32 per GPU (the kernel accepts 16/32/64/128)', 3: '8 groups not divisible by 3 → 9 groups × 8 = 72 heads: 3 groups (24 heads) per GPU; 9th group is padding (wo_a/wo_b zeroed, sink −∞)' }, pad3: 72, groups: 8 },
        dflash: { t: 'DFlash2 · GLM drafter (5 layers)', rb: 'glm', H: 32, kv: 'gqa', kvN: 8, rows: [
          ['query heads', '32'], ['K/V heads', '8 (GQA 4:1)'], ['q·k / v dims', '128 / 128'], ['scale', '1/√128 ≈ 0.0884'], ['mask', 'bidirectional in the block (is_causal: false) + 2048 window'],
          ['sink', 'not declared'], ['output gate', 'no'], ['QK-Norm', 'q_norm, k_norm [128] (Qwen3 base)'], ['q_proj', '[4096 × 4096]; k/v [1024 × 4096]'], ['o_proj', '[4096 × 4096]'],
          ['KV/token/layer', '2 × 8 × 128 × 2 B = 4,096 B (BF16), window only']],
          tp: { 1: '32 Q + 8 KV on one GPU', 2: '16 Q + 4 KV per GPU', 3: '32 and 8 are not divisible by 3 → config padded to 36/9, drafter at TP=1' }, pad3: 36 },
        mimog: { t: 'MiMo-V2.6 · global layers (tutorial)', rb: 'all', H: 64, kv: 'gqa', kvN: 4, rows: [
          ['query heads', '64'], ['K/V heads', '4 (GQA 16:1)'], ['q·k / v dims', '192 / 128 (DiffKV)'], ['scale', '1/√192 ≈ 0.0722; V × 0.707 before'], ['mask', 'causal'],
          ['sink', 'no (add_full_attention_sink_bias: false)'], ['output gate', 'no'], ['QK-Norm', 'no'], ['q_proj', 'fused QKV in FP8, pre-sliced into 4'], ['o_proj', '[4096 × 8192] (64 × 128), kept in BF16'],
          ['KV/token/layer', '(192 + 128) × 4 × 2 B = 2,560 B (BF16)']],
          tp: { 1: '64 Q + 4 KV', 2: '32 Q + 2 KV per GPU', 3: '64 Q and 4 KV are not divisible by 3' }, bad3: true },
        mimos: { t: 'MiMo-V2.6 · 128-window layers (tutorial)', rb: 'all', H: 64, kv: 'gqa', kvN: 8, rows: [
          ['query heads', '64'], ['K/V heads', '8 (twice the global ones)'], ['q·k / v dims', '192 / 128 (DiffKV)'], ['scale', '1/√192 ≈ 0.0722; V × 0.707 before'], ['mask', 'causal, 128-token window'],
          ['sink', 'yes: attention_sink_bias [64] (BF16)'], ['output gate', 'no'], ['QK-Norm', 'no'], ['q_proj', 'fused QKV in FP8, pre-sliced into 4'], ['o_proj', '[4096 × 8192], kept in BF16'],
          ['KV/token/layer', '(192 + 128) × 8 × 2 B = 5,120 B, only the last 128 tokens']],
          tp: { 1: '64 Q + 8 KV', 2: '32 Q + 4 KV per GPU', 3: '64 is not divisible by 3' }, bad3: true },
        vaswani: { t: 'Original Transformer (2017)', rb: 'all', H: 8, kv: 'gqa', kvN: 8, rows: [
          ['query heads', '8'], ['K/V heads', '8 (MHA: one per query)'], ['q·k / v dims', '64 / 64'], ['scale', '1/√64 = 0.125'], ['mask', 'causal in the decoder'],
          ['sink', 'no'], ['output gate', 'no'], ['QK-Norm', 'no'], ['q_proj', '[512 × 512]'], ['o_proj', '[512 × 512]'],
          ['KV/token/layer', '2 × 8 × 64 × 2 B = 2,048 B (in BF16)']],
          tp: { 1: '8 heads', 2: '4 per GPU', 3: '8 is not divisible by 3' }, bad3: true },
      };
      const S = { m: 'qwen', tp: 2 };
      const ctl = UI.controls(fig);
      let H = null;
      UI.select(ctl, { label: 'Model', options: [{ v: 'glm', label: 'GLM-5.3 · MLA' }, { v: 'qwen', label: 'Qwen3.8 · QSA' }, { v: 'ds', label: 'DeepSeek-V4' }, { v: 'dflash', label: 'DFlash2 (GLM drafter)' }, { v: 'mimog', label: 'MiMo-V2.6 · global' }, { v: 'mimos', label: 'MiMo-V2.6 · window' }, { v: 'vaswani', label: 'Transformer 2017' }], value: 'qwen', onChange: (v) => { S.m = v; if (H) H.redraw(); } });
      UI.seg(ctl, { label: 'TP', options: [1, 2, 3], value: 2, onChange: (v) => { S.tp = v; if (H) H.redraw(); } });
      H = V.fig(fig, (g, w, h, t) => {
        if (w < 80 || h < 80) return;
        const C = V.C();
        const M = F[S.m];
        const wide = w >= 620;
        const rc = [C.a, C.b, C.d];
        const P = wide ? { x: 16, y: 16, w: Math.min(250, w * 0.34), h: h - 32 } : { x: 12, y: 12, w: w - 24, h: 280 };
        const L = wide ? { x: P.x + P.w + 24, y: 20, w: w - P.x - P.w - 40 } : { x: 14, y: P.y + P.h + 10, w: w - 28 };
        const recCol = M.rb === 'glm' ? C.glm : M.rb === 'qwen' ? C.qwen : M.rb === 'ds' ? C.ds : C.b;
        // título
        V.box(g, P.x, P.y, P.w, P.h, { r: 14, fill: V.alpha(recCol, 0.06), stroke: V.alpha(recCol, 0.5), lw: 1.2 });
        V.text(g, M.t, P.x + 12, P.y + 20, { size: 12, color: C.ink, weight: 700, maxW: P.w - 24 });
        // retrato: cabeças de query
        const Hp = S.tp === 3 && M.pad3 ? M.pad3 : M.H;
        const bad3 = S.tp === 3 && M.bad3;
        const perRank = Math.floor(Hp / S.tp);
        // no DeepSeek cada linha do retrato é um grupo de 8 cabeças do o_proj (o TP=3 acrescenta uma 9ª linha)
        const cols = M.groups ? M.H / M.groups : Hp <= 12 ? Math.min(Hp, 6) : Hp <= 36 ? 6 : Hp <= 64 ? 8 : 9;
        const rowsN = Math.ceil(Hp / cols);
        const areaH = P.h - (wide ? 150 : 124);
        const rad = Math.max(3.2, Math.min(12, (P.w - 36) / (cols * 2.7), areaH / (rowsN * 2.7)));
        const gx0 = P.x + (P.w - cols * rad * 2.7) / 2 + rad * 1.35, gy0 = P.y + 40 + Math.max(0, (areaH - rowsN * rad * 2.7) * 0.35) + rad * 1.35;
        const pos = [];
        for (let hh = 0; hh < Hp; hh++) {
          const cx = gx0 + (hh % cols) * rad * 2.7, cy = gy0 + Math.floor(hh / cols) * rad * 2.7;
          pos.push([cx, cy]);
        }
        // K/V
        const kvY = gy0 + rowsN * rad * 2.7 + (wide ? 34 : 20);
        const kvPos = [];
        if (M.kv === 'gqa') {
          const n = M.kvN, sq = Math.min(18, (P.w - 30) / (n * 1.8));
          for (let k = 0; k < n; k++) kvPos.push([P.x + (P.w - n * sq * 1.8) / 2 + k * sq * 1.8 + sq * 0.9, kvY, sq]);
        } else kvPos.push([P.x + P.w / 2, kvY, 0]);
        // ligações query → kv
        const grp = M.kv === 'gqa' ? M.H / M.kvN : M.H;
        for (let hh = 0; hh < M.H; hh++) {
          const k = M.kv === 'gqa' ? Math.floor(hh / grp) : 0;
          const [cx, cy] = pos[hh], [kx, ky] = kvPos[k];
          g.strokeStyle = V.alpha(C.muted, M.H > 32 ? 0.12 : 0.22); g.lineWidth = 1;
          g.beginPath(); g.moveTo(cx, cy + rad); g.lineTo(kx, ky - 10); g.stroke();
        }
        if (M.groups) {
          for (let r = 0; r < rowsN; r++) {
            const [x0, y0] = pos[r * cols];
            g.strokeStyle = V.alpha(r >= M.H / cols ? C.muted : C.e, r >= M.H / cols ? 0.7 : 0.35); g.lineWidth = 1;
            if (r >= M.H / cols) g.setLineDash([3, 3]);
            V.rr(g, x0 - rad * 1.25, y0 - rad * 1.25, (cols - 1) * rad * 2.7 + rad * 2.5, rad * 2.5, rad * 1.25); g.stroke(); g.setLineDash([]);
          }
        }
        for (let hh = 0; hh < Hp; hh++) {
          const [cx, cy] = pos[hh];
          const rk = Math.min(S.tp - 1, Math.floor(hh / Math.max(1, perRank)));
          if (hh >= M.H) { g.setLineDash([2, 2]); g.strokeStyle = C.muted; g.lineWidth = 1; g.beginPath(); g.arc(cx, cy, rad, 0, Math.PI * 2); g.stroke(); g.setLineDash([]); continue; }
          const pulse = 0.75 + 0.25 * Math.sin(t * 2 + hh * 0.6);
          V.dot(g, cx, cy, rad, V.alpha(rc[rk], 0.55 + 0.35 * pulse));
        }
        if (M.kv === 'gqa') {
          const kvPer = M.kvN / S.tp;
          kvPos.forEach(([kx, ky, sq], k) => {
            const rk = Number.isInteger(kvPer) && kvPer >= 1 ? Math.floor(k / kvPer) : -1;
            const col = bad3 || rk < 0 ? C.c : rc[rk];
            V.box(g, kx - sq / 2, ky - sq / 2, sq, sq, { r: 3, fill: V.alpha(col, 0.35), stroke: col, lw: 1.3 });
          });
          V.text(g, `${M.kvN} K/V heads`, P.x + P.w / 2, kvY + 26, { size: 11, color: C.muted, align: 'center' });
        } else {
          const [kx, ky] = kvPos[0];
          const bw = Math.min(140, P.w - 40);
          V.box(g, kx - bw / 2, ky - 12, bw, 24, { r: 8, fill: V.alpha(C.e, 0.25), stroke: C.e, lw: 1.3, label: M.kvL, size: 11.5 });
        }
        const hpg = M.groups ? M.H / M.groups : 0;
        const capQ = M.groups ? `${Hp} query heads in ${Hp / hpg} groups of ${hpg}` + (Hp !== M.H ? ` (${Hp - M.H} padding)` : '') : `${M.H} query heads` + (Hp !== M.H ? ` (+${Hp - M.H} padding)` : '');
        V.text(g, capQ, P.x + P.w / 2, P.y + P.h - 12, { size: 11, color: C.muted, align: 'center', maxW: P.w - 16 });
        // ficha
        let y = L.y + 4;
        const labW = wide ? 128 : 112;
        const rowsList = M.rows.concat([[`per GPU (TP=${S.tp})`, M.tp[S.tp]]]);
        rowsList.forEach(([k, v], idx) => {
          const last = idx === rowsList.length - 1;
          const warn = last && /not divisible/.test(v);
          V.text(g, k, L.x, y + 11, { size: 11.5, color: C.muted });
          const ny = para(g, v, L.x + labW, y + 11, L.w - labW - 8, { size: 11.5, font: 'mono', color: warn ? C.c : last ? C.a : C.ink, lh: 15 });
          y = Math.max(y + 20, ny + 5);
          if (!last) V.line(g, L.x, y - 3, L.x + L.w, y - 3, { color: V.alpha(C.muted, 0.12), lw: 1 });
        });
        // legenda de GPUs
        const ly0 = Math.min(h - 14, y + 14);
        for (let k = 0; k < S.tp; k++) { V.dot(g, L.x + 6 + k * 70, ly0 - 4, 5, rc[k]); V.text(g, 'GPU ' + k, L.x + 16 + k * 70, ly0, { size: 11, color: C.muted }); }
      });
    }
  },
});

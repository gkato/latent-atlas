LA.chapter({
  id: 'mla',
  part: 3,
  order: 12,
  title: 'Shrinking the KV: MQA, GQA and MLA',
  navTitle: 'MLA & compact KV',
  subtitle: 'Sharing keys across heads, storing only a latent summary and, at the limit, making the key the value itself: how GLM, Qwen and DeepSeek fit a million tokens in memory.',
  minutes: 55,
  recipes: ['glm', 'tf', 'qwen', 'ds'],
  tags: ['MLA', 'GQA', 'MQA', 'KV cache', 'weight absorption', 'decoupled RoPE', 'NoPE', 'o_groups', 'kv_lora_rank', 'TensorFold'],
  goals: [
    'Compute the KV cache <b>bytes per token</b> for MHA, GQA, MQA and MLA and explain why they determine capacity and decode speed',
    'Explain <b>MLA</b>: the KV latent (<code>kv_lora_rank</code>), the down- and up-projections, <code>q_lora_rank</code> and <b>weight absorption</b>, which turns decode into an MQA over the latent',
    'Understand why RoPE has to be <b>decoupled</b>, why the NoPE GLM gets 64 zeros (<code>rope_pad</code>), why that is exact and what changes when the engine has its own kernels for GLM (TensorFold, 528 B per row)',
    'Describe <b>DeepSeek-V4</b> attention (512-dim K = V, inverse RoPE on the output, grouped output with <code>o_groups</code> and <code>o_lora_rank</code>) and what it imposes on TP=3',
  ],
  prereqs: ['atencao', 'posicao-rope', 'prefill-decode-kv'],
  summary: [
    'The KV cache costs \\(L \\times n_{kv} \\times (d_k + d_v) \\times b\\) bytes per token, which becomes \\(L \\times 2 \\times n_{kv} \\times d_h \\times b\\) when key and value have the same size. That number decides how many tokens fit and how many bytes each decode step rereads, and at long context it dominates bandwidth.',
    '<b>MQA</b> and <b>GQA</b> make groups of queries share K and V: the cache drops by a factor of \\(H/G\\) and each read serves \\(H/G\\) queries. The kernels require equal groups (<code>qo % kv == 0</code>). Qwen uses 24:2 (26,624 B/token in BF16), and GLM\'s DFlash2 uses 32:8, padded to 36:9 at TP3.',
    '<b>MLA</b> stores only a 512-dim latent per token (<code>kv_lora_rank</code>). <b>Absorption</b>, \\(q^\\top W^{UK} c = (W^{UK\\top} q)^\\top c\\), brings the query into latent space, and decode becomes an MQA over the latent. That is why <code>kv_b_proj</code> is read directly and stays in BF16 in the recipes.',
    'A RoPE between the latent and the key would break absorption. Hence <b>decoupled RoPE</b> (64 shared dims) and the kernels\' 576 = 512 + 64 geometry. GLM is NoPE, and the recipe pads with 64 zeros (<code>rope_pad</code>): the result is exact and costs 656 B instead of ~528 per token. The GLM·TF recipe, with TensorFold\'s own kernels, does without the padding: the FP8 row has 528 B, and the power-of-two scale is folded into the scores and the probabilities, because the latent is K and V.',
    'In <b>DeepSeek-V4</b>, a single 512-dim vector (448 NoPE + 64 RoPE) is K and V at the same time (<code>kv_lora_rank = head_dim</code>). Since the value carries the absolute rotation, the output is un-rotated by \\(R_{-i}\\) (<code>fused_inv_rope_fp8_quant</code>) before <code>wo_a</code>. The row is 584 B and is replicated on every rank.',
    'V4\'s output is grouped: 8 groups with <code>wo_a</code> (4096 → 1024, the <code>o_lora_rank</code>) followed by <code>wo_b</code>, with half the parameters. The BMM\'s \\(r = 4096\\) contract forces TP=3 to create a 9th ghost group (zero weights, sink \\(-\\infty\\)).',
    'MLA kernels fix their geometries at compile time. The recipe adapts the model with neutral padding (zero for the sum, \\(-\\infty\\) for the softmax) or judicious trimming, and that is how to read every attention patch.',
  ],
  quiz: [
    { q: 'The QSA layers of Qwen3.8 (sparse softmax attention: the read picks 2,048 tokens, but the cache stores all of them) have 24 query heads, 2 K/V heads, <code>head_dim</code> 256 and KV in 13 layers (12 + MTP). How many bytes per token does the KV take in BF16?', o: ['2,048 B', '26,624 B', '319,488 B', '13,312 B'], a: 1, e: '13 × 2 (K and V) × 2 heads × 256 × 2 B = 26,624 B. 319,488 B would be with 24 KV heads (MHA), 12 times more. 13,312 B is the same cache in FP8, or what each GPU stores at TP2 in BF16. 2,048 B is a single layer.' },
    { q: 'Why, during decode, does MLA not need to rebuild K and V for past tokens?', o: ['Because it stores K and V already rebuilt in the cache', 'Because, by associativity, \\(q^\\top(W^{UK}c) = (W^{UK\\top}q)^\\top c\\): it is enough to project the query once into latent space and attend directly over the latents', 'Because decode only looks at the last token', 'Because the latent is quantized to FP8'], a: 1, e: 'Weight absorption swaps “lift every latent up to the head” for “bring the query down to the latent”, once per step, and \\(W^{UV}\\) moves out of the sum and is applied once. Storing ready-made K and V would undo all the memory savings.' },
    { q: 'The GLM recipe appends 64 zeros to the query and writes <code>k_pe = 0</code> into the cache. What does that change in the attention scores?', o: ['Nothing: a zero block adds 0 to the dot product, so the scores, softmax and output are the same; the only cost is memory (656 B instead of ~528)', 'It adds positional information to the NoPE layers', 'It reduces the KV by 24%', 'It changes the softmax scale from 1/16 to 1/24'], a: 0, e: 'The SM120 kernel only exists in the 576 geometry (512 + 64). Since \\([q;0]\\cdot[c;z] = q\\cdot c\\), the result is exact and no new kernel is needed. The scale stays the model\'s own (<code>qk_head_dim</code>\\(^{-1/2}\\) = 1/16), and the price is ≈ 24% more KV in the 11 DSA layers.' },
    { q: 'Why does DeepSeek-V4 rotate the last 64 dimensions of the attention output by position \\(-i\\)?', o: ['To quantize better in FP8', 'Because K = V: the values carry the rotation of each token\'s absolute position, and \\(R_{-i}R_j = R_{j-i}\\) makes each contribution relative to the query', 'To compensate for YaRN', 'So that the attention sink works'], a: 1, e: 'In ordinary attention, the value is not rotated. In V4 the value is the key, and the 64 RoPE dims arrive rotated by \\(R_j\\). Without the inverse, the same passage at a different position in the document would produce a different output. In vLLM this is the <code>fused_inv_rope_fp8_quant</code> kernel, which also quantizes the output for <code>wo_a</code>.' },
    { q: 'At TP=3 in DeepSeek-V4, why does the patch create a 9th group instead of padding heads inside the 8 groups?', o: ['Because 9 heads per group do not fit in memory', 'Because the <code>wo_a</code> BMM has the contract \\(r = 8 \\times 512 = 4096\\) per group, fixed in the checkpoint; changing the heads per group changes \\(r\\) and the kernel refuses', 'Because 72 is not divisible by 3', 'Because vLLM requires an odd number of groups'], a: 1, e: 'With 9 groups of 8 heads, each rank gets 3 whole groups and \\(r\\) stays 4096. The extra group has <code>wo_a</code>/<code>wo_b</code> zeroed and <code>attn_sink = −∞</code>, the neutral element of the softmax denominator. 72 is divisible by 3; the problem was the 8 groups.' },
    { q: 'At TP=2, what happens to the KV cache of GLM\'s MLA and to that of Qwen\'s GQA?', o: ['Both are split in half', 'The MLA one is replicated on both ranks (a single latent, no heads to slice); Qwen\'s splits, with 1 of the 2 KV heads on each GPU', 'The MLA one is split and Qwen\'s is replicated', 'Both are replicated'], a: 1, e: 'Tensor parallel splits heads. MLA during decode is an MQA over the latent, so each rank stores the whole latent (656 B/token/layer on both nodes). With 2 KV heads and TP=2, Qwen puts one on each GPU.' },
  ],
  mount(root, LA) {
    const V = LA.viz, UI = LA.ui;

    /* ================================================================ helpers */
    const nf = (x) => Math.round(x).toLocaleString('en-US');
    const sf = (x, d = 1) => (+x).toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d });
    const kfmt = (n) => (n >= 1048576 ? sf(n / 1048576, n % 1048576 ? 2 : 0) + 'M' : n >= 1024 ? nf(n / 1024) + 'K' : nf(n));
    const fmtGiB = (b) => (b >= 1099511627776 ? sf(b / 1099511627776, 2) + ' TiB' : b >= 1073741824 ? sf(b / 1073741824, b >= 10737418240 ? 1 : 2) + ' GiB' : b >= 1048576 ? sf(b / 1048576, 1) + ' MiB' : b >= 1024 ? sf(b / 1024, 1) + ' KiB' : nf(b) + ' B');
    const fmtTok = (b) => (b < 100000 ? nf(b) + ' B' : b < 1e6 ? sf(b / 1e3, 1) + ' KB' : sf(b / 1e6, 2) + ' MB');
    const fmtFlop = (f) => (f >= 1e12 ? sf(f / 1e12, 2) + ' TFLOP' : f >= 1e9 ? sf(f / 1e9, 1) + ' GFLOP' : f >= 1e6 ? sf(f / 1e6, 1) + ' MFLOP' : nf(f) + ' FLOP');
    let fcache = {};
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(() => { fcache = {}; });
    const font = (s, f = 'sans', w = 400) => fcache[s + f + w] || (fcache[s + f + w] = V.font(s, f, w));
    const tw = (g, s, size = 12, f = 'sans', w = 400) => { g.font = font(size, f, w); return g.measureText(String(s)).width; };
    const RED = (C) => C.red || C.c;
    const onInk = (C) => (C.dark ? '#00210F' : '#ffffff');
    function wrap(g, s, maxW, size = 12, f = 'sans', w = 400) {
      const words = String(s).split(' ');
      const lines = [];
      let cur = '';
      for (const wd of words) {
        const tt = cur ? cur + ' ' + wd : wd;
        if (tw(g, tt, size, f, w) > maxW && cur) { lines.push(cur); cur = wd; } else cur = tt;
      }
      if (cur) lines.push(cur);
      return lines;
    }
    function textBlock(g, s, x, y, maxW, o = {}) {
      const size = o.size || 12, lh = o.lh || Math.round(size * 1.38);
      const lines = wrap(g, s, maxW, size, o.font || 'sans', o.weight || 400);
      const max = o.maxLines || 99;
      lines.slice(0, max).forEach((l, i) => V.text(g, l, x, y + i * lh, Object.assign({}, o, { size })));
      return Math.min(lines.length, max) * lh;
    }
    /** text that shrinks until it fits in maxW (minimum 11px) */
    function fitText(g, s, x, y, maxW, o = {}) {
      let size = o.size || 12;
      while (size > 11 && tw(g, s, size, o.font || 'sans', o.weight || 400) > maxW) size -= 0.5;
      V.text(g, s, x, y, Object.assign({}, o, { size, maxW }));
    }
    /** label with a superscript: W + ^DQ */
    function supLabel(g, base, sup, x, y, o = {}) {
      const size = o.size || 13, ss = Math.max(11, Math.round(size * 0.78));
      const f = o.font || 'sans', wgt = o.weight || 600;
      const w1 = tw(g, base, size, f, wgt), w2 = sup ? tw(g, sup, ss, f, wgt) : 0;
      const tot = w1 + w2 + (sup ? 1 : 0);
      let x0 = o.align === 'center' ? x - tot / 2 : o.align === 'right' ? x - tot : x;
      V.text(g, base, x0, y, { size, font: f, weight: wgt, color: o.color });
      if (sup) V.text(g, sup, x0 + w1 + 1, y - size * 0.36, { size: ss, font: f, weight: wgt, color: o.color });
      return tot;
    }
    function stripes(g, x, y, w, h, color, step = 5) {
      g.save(); V.rr(g, x, y, w, h, 3); g.clip();
      g.strokeStyle = color; g.lineWidth = 1.2; g.beginPath();
      for (let k = -h; k < w; k += step) { g.moveTo(x + k, y + h); g.lineTo(x + k + h, y); }
      g.stroke(); g.restore();
    }
    function tokenBlob(g, x, y, r, color) {
      const C = V.C();
      V.dot(g, x, y, r, color);
      const e = r * 0.34;
      V.dot(g, x - e, y - e * 0.45, Math.max(1.2, r * 0.15), onInk(C));
      V.dot(g, x + e, y - e * 0.45, Math.max(1.2, r * 0.15), onInk(C));
      g.strokeStyle = onInk(C); g.lineWidth = 1.2; g.beginPath(); g.arc(x, y + e * 0.1, r * 0.42, 0.22 * Math.PI, 0.78 * Math.PI); g.stroke();
    }
    function sizeStage(fig, wide, narrow) {
      const st = fig.querySelector('.la-fig-stage');
      const W = fig.clientWidth || window.innerWidth;
      const nar = W < 600;
      if (st) st.style.height = (nar ? narrow : wide) + 'px';
      return nar;
    }
    /** point on a quadratic curve */
    const qpt = (p0, p1, p2, t) => { const u = 1 - t; return [u * u * p0[0] + 2 * u * t * p1[0] + t * t * p2[0], u * u * p0[1] + 2 * u * t * p1[1] + t * t * p2[1]]; };
    /** arrow with animated flow (dots) */
    function flow(g, a, b, o = {}) {
      const C = V.C();
      const col = o.color || C.muted, on = o.active !== false, t = o.t || 0;
      const lw = o.lw || (on ? 2 : 1.2);
      const alpha = on ? 1 : 0.28;
      const ctrl = o.curve ? [(a[0] + b[0]) / 2 + o.curve[0], (a[1] + b[1]) / 2 + o.curve[1]] : null;
      g.globalAlpha = alpha;
      if (ctrl) {
        g.beginPath(); g.moveTo(a[0], a[1]); g.quadraticCurveTo(ctrl[0], ctrl[1], b[0], b[1]);
        g.strokeStyle = col; g.lineWidth = lw; if (o.dash) g.setLineDash(o.dash); g.stroke(); g.setLineDash([]);
        const p = qpt(a, ctrl, b, 0.96), an = Math.atan2(b[1] - p[1], b[0] - p[0]);
        head(b, an);
      } else {
        V.line(g, a[0], a[1], b[0] - Math.cos(Math.atan2(b[1] - a[1], b[0] - a[0])) * 4, b[1] - Math.sin(Math.atan2(b[1] - a[1], b[0] - a[0])) * 4, { color: col, lw, dash: o.dash });
        head(b, Math.atan2(b[1] - a[1], b[0] - a[0]));
      }
      g.globalAlpha = 1;
      if (on && !o.noDots) {
        for (let k = 0; k < 3; k++) {
          const u = (t * (o.speed || 0.55) + k / 3) % 1;
          const p = ctrl ? qpt(a, ctrl, b, u) : [V.lerp(a[0], b[0], u), V.lerp(a[1], b[1], u)];
          V.dot(g, p[0], p[1], 2.6, o.dotColor || col);
        }
      }
      function head(p, an) {
        const hs = o.head || 7;
        g.beginPath(); g.moveTo(p[0], p[1]);
        g.lineTo(p[0] - hs * Math.cos(an - 0.45), p[1] - hs * Math.sin(an - 0.45));
        g.lineTo(p[0] - hs * Math.cos(an + 0.45), p[1] - hs * Math.sin(an + 0.45));
        g.closePath(); g.fillStyle = col; g.fill();
      }
    }

    /* =================================================== Fig 1 — library */
    (function fig1() {
      const fig = root.querySelector('#fig-biblioteca');
      if (!fig) return;
      const SCH = {
        mha: { label: 'MHA', long: '8 keys + 8 values per token', units: 16, bytes: 4096 },
        gqa: { label: 'GQA', long: '2 groups: 2 keys + 2 values', units: 4, bytes: 1024 },
        mqa: { label: 'MQA', long: '1 key + 1 value, shared by all', units: 2, bytes: 512 },
        mla: { label: 'MLA', long: 'latent 512 + RoPE 64', units: 4.5, bytes: 1152 },
      };
      const ORDER = ['mha', 'gqa', 'mqa', 'mla'];
      const S = { scheme: 'mha', ctx: 17, fromUnits: 16, tSwitch: -10 };
      let H = null;
      const unitsNow = () => {
        if (!H) return SCH[S.scheme].units;
        const e = V.ease.inOut(V.clamp((H.t - S.tSwitch) / 0.55, 0, 1));
        return V.lerp(S.fromUnits, SCH[S.scheme].units, e);
      };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Scheme', options: ORDER.map((v) => ({ v, label: SCH[v].label })), value: 'mha', onChange: (v) => { S.fromUnits = unitsNow(); S.scheme = v; S.tSwitch = H ? H.t : 0; upd(); } });
      UI.slider(ctl, { label: 'Context', min: 10, max: 20, step: 1, value: 17, fmt: (v) => kfmt(2 ** v) + ' tokens', onInput: (v) => { S.ctx = v; upd(); } });
      const r1 = UI.readout(ctl, 'Row', '—');
      const r2 = UI.readout(ctl, 'Reread each step (1 layer)', '—');
      function upd() { const sc = SCH[S.scheme]; r1.set(nf(sc.bytes) + ' B/token'); r2.set(fmtGiB(sc.bytes * 2 ** S.ctx)); }
      upd();
      const P = 2.4, MAXV = 12;
      H = V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const sc = SCH[S.scheme];
        const n = Math.floor(t / P), f = (t % P) / P;
        const rows = 7 + (n % 26);
        const appended = f >= 0.62;
        const total = rows + (appended ? 1 : 0);
        const pad = 14, labW = 46, x0 = pad + labW;
        const top = 50, botPanel = 96;
        const slots = Math.min(total, MAXV);
        const rowH = Math.max(11, Math.min(18, Math.floor((h - top - botPanel - 8) / MAXV) - 3));
        const gapY = 3;
        const u = Math.max(10, Math.min(30, Math.floor((w - x0 - pad - 34) / 16)));
        const uNow = unitsNow();
        const cw = u * (uNow / sc.units);
        const narrow = w < 560;
        // header
        V.text(g, `decode step ${n + 1}`, pad, 20, { size: 13, weight: 600, color: C.ink });
        V.text(g, `the beam rereads ${nf(rows)} rows and files 1 new one`, pad, 38, { size: 11.5, color: C.muted });
        V.text(g, narrow ? sc.label : sc.label + ' · ' + sc.long, w - pad, 20, { size: 12, align: 'right', color: C.ink, weight: 600 });
        // which rows are shown
        const hidden = total > MAXV ? total - (MAXV - 1) : 0;
        const list = [];
        if (hidden) { list.push(-1); for (let k = hidden; k < total; k++) list.push(k); } else for (let k = 0; k < total; k++) list.push(k);
        const yOf = (slot) => top + slot * (rowH + gapY);
        const wallBottom = yOf(slots) - gapY;
        // beam
        let beamY = -1e9;
        if (f < 0.62) {
          const p = V.ease.inOut(f / 0.62);
          beamY = V.lerp(top + rowH / 2, wallBottom - rowH / 2, p);
        }
        const rowW = uNow * u;
        list.forEach((k, slot) => {
          const y = yOf(slot);
          if (k === -1) {
            g.setLineDash([3, 4]); V.rr(g, x0, y, Math.max(rowW, 150), rowH, 4); g.strokeStyle = V.alpha(C.muted, 0.6); g.lineWidth = 1; g.stroke(); g.setLineDash([]);
            V.text(g, `⋮ +${nf(hidden)} older rows (also reread)`, x0 + 8, y + rowH / 2 + 4, { size: 11, color: C.muted });
            return;
          }
          const isNew = appended && k === total - 1;
          let xoff = 0, alpha = 1;
          if (isNew) { const e = V.ease.out(V.clamp((f - 0.62) / 0.22, 0, 1)); xoff = (1 - e) * Math.min(160, w * 0.3); alpha = e; }
          const lit = Math.abs(y + rowH / 2 - beamY) < rowH * 1.3;
          V.text(g, 't=' + k, x0 - 8, y + rowH / 2 + 4, { size: 11, font: 'mono', align: 'right', color: lit ? C.ink : C.muted });
          g.globalAlpha = alpha;
          const xs = x0 + xoff;
          const fillA = lit ? 0.98 : 0.62;
          if (S.scheme === 'mla') {
            const lw = 4 * cw - 2;
            V.box(g, xs, y, lw, rowH, { r: 4, fill: V.alpha(C.f, fillA) });
            V.box(g, xs + 4 * cw, y, Math.max(3, 0.5 * cw - 2), rowH, { r: 3, fill: V.alpha(C.d, 0.25) });
            stripes(g, xs + 4 * cw, y, Math.max(3, 0.5 * cw - 2), rowH, V.alpha(C.d, lit ? 1 : 0.8), 4);
          } else {
            const nk = sc.units / 2;
            for (let i = 0; i < sc.units; i++) {
              const col = i < nk ? C.a : C.b;
              V.box(g, xs + i * cw, y, Math.max(2, cw - 2), rowH, { r: Math.min(4, cw / 4), fill: V.alpha(col, fillA) });
            }
          }
          if (isNew) { V.glow(g, xs + rowW + 12, y + rowH / 2, 16, C.c, 0.8 * alpha); tokenBlob(g, xs + rowW + 14, y + rowH / 2, Math.min(9, rowH * 0.62), C.c); }
          g.globalAlpha = 1;
        });
        // beam drawn on top
        if (f < 0.62) {
          const bw = Math.max(rowW, 160) + 20;
          const grd = g.createLinearGradient(0, beamY - rowH * 1.4, 0, beamY + rowH * 1.4);
          grd.addColorStop(0, V.alpha(C.c, 0)); grd.addColorStop(0.5, V.alpha(C.c, 0.28)); grd.addColorStop(1, V.alpha(C.c, 0));
          g.save(); g.beginPath(); g.rect(0, top - 3, w, wallBottom - top + 6); g.clip();
          g.fillStyle = grd; g.fillRect(x0 - 10, beamY - rowH * 1.4, bw, rowH * 2.8); g.restore();
          V.line(g, x0 - 10, beamY, x0 - 10 + bw, beamY, { color: V.alpha(C.c, 0.9), lw: 1.5 });
          V.text(g, 'reading', x0 - 10 + bw + 6, beamY + 4, { size: 11, color: C.c, font: 'mono' });
        }
        // side gauge: bytes read in this step
        if (w >= 640) {
          const gx = w - pad - 128, gy = top, gh2 = Math.max(60, top + MAXV * (rowH + gapY) - top - 58), gw2 = 26;
          const prog = f < 0.62 ? V.ease.inOut(f / 0.62) : 1;
          V.text(g, 'read this step', gx, gy - 6, { size: 11, color: C.muted });
          V.box(g, gx, gy, gw2, gh2, { r: 6, fill: V.alpha(C.muted, 0.1), stroke: V.alpha(C.muted, 0.35), lw: 1 });
          const fh = gh2 * prog;
          V.box(g, gx, gy + gh2 - fh, gw2, Math.max(2, fh), { r: 6, fill: V.alpha(C.c, 0.75) });
          V.text(g, (Math.round(prog * rows) === 1 ? '1 row' : nf(Math.round(prog * rows)) + ' rows'), gx + gw2 + 8, gy + gh2 - fh + 4, { size: 11, font: 'mono', color: C.c });
          V.text(g, '× ' + nf(sc.bytes) + ' B', gx + gw2 + 8, gy + gh2 - fh + 19, { size: 11, font: 'mono', color: C.muted });
          textBlock(g, `at ${kfmt(2 ** S.ctx)} tokens: ${fmtGiB(sc.bytes * 2 ** S.ctx)} per layer, for every generated token`, gx - 4, gy + gh2 + 16, 132, { size: 11, color: C.muted, lh: 14 });
        }
        // comparison panel
        const by = h - botPanel + 10;
        V.line(g, pad, by - 8, w - pad, by - 8, { color: V.alpha(C.muted, 0.25), lw: 1 });
        V.text(g, narrow ? 'B/token in this layer (8 heads, d_h 128, BF16)' : 'bytes per token in this layer (8 heads, d_h = 128, BF16)', pad, by + 6, { size: 11, color: C.muted });
        const lab = 40, valW = 64, bmax = w - 2 * pad - lab - valW - 8;
        ORDER.forEach((k, i) => {
          const s2 = SCH[k], y = by + 16 + i * 17, on = k === S.scheme;
          V.text(g, s2.label, pad, y + 10, { size: 11, font: 'mono', weight: on ? 700 : 400, color: on ? C.ink : C.muted });
          const bw = Math.max(3, (s2.bytes / 4096) * bmax);
          const col = k === 'mla' ? C.f : k === 'mha' ? C.a : k === 'gqa' ? C.e : C.b;
          V.box(g, pad + lab, y, bw, 12, { r: 4, fill: V.alpha(col, on ? 0.95 : 0.4) });
          V.text(g, nf(s2.bytes) + ' B', pad + lab + bw + 6, y + 10, { size: 11, font: 'mono', color: on ? C.ink : C.muted });
        });
      });
    })();

    /* ================================================= Fig 2 — sharing */
    (function fig2() {
      const fig = root.querySelector('#fig-compartilhar');
      if (!fig) return;
      const PRE = {
        qwen: { nq: 24, nkv: 2, dh: 256, tp: 2, name: 'Qwen3.8 · QSA layers (sparse softmax)' },
        dflash: { nq: 32, nkv: 8, dh: 128, tp: 2, name: 'GLM\'s DFlash2' },
        dflash3: { nq: 36, nkv: 9, dh: 128, tp: 3, name: 'DFlash2 padded for TP3' },
        mqa: { nq: 24, nkv: 1, dh: 256, tp: 2, name: 'MQA (one key for all)' },
      };
      const S = Object.assign({ pre: 'qwen' }, PRE.qwen);
      const ctl = UI.controls(fig);
      let sK = null, sT = null, segP = null, applying = false;
      segP = UI.seg(ctl, { label: 'Preset', options: [{ v: 'qwen', label: 'Qwen 24/2' }, { v: 'dflash', label: 'DFlash2 32/8' }, { v: 'dflash3', label: 'TP3 36/9' }, { v: 'mqa', label: 'MQA' }, { v: 'livre', label: 'custom' }], value: 'qwen', onChange: (v) => {
        if (v === 'livre') { S.pre = 'livre'; upd(); return; }
        applying = true; Object.assign(S, PRE[v]); S.pre = v;
        if (sK) { sK.input.max = S.nq; sK.set(S.nkv); }
        if (sT) sT.set(S.tp);
        applying = false; upd();
      } });
      sK = UI.slider(ctl, { label: 'K/V heads', min: 1, max: 24, step: 1, value: 2, fmt: (v) => String(v), onInput: (v) => { S.nkv = Math.min(v, S.nq); if (!applying && S.pre !== 'livre') { S.pre = 'livre'; if (segP) segP.set('livre'); } upd(); } });
      sT = UI.seg(ctl, { label: 'TP', options: [1, 2, 3], value: 2, onChange: (v) => { S.tp = v; upd(); } });
      const rR = UI.readout(ctl, 'Queries per KV', '—');
      const rB = UI.readout(ctl, 'KV/token/layer', '—');
      const rG = UI.readout(ctl, 'Per GPU', '—');
      function model() {
        const { nq, nkv, tp } = S;
        const valid = nq % nkv === 0, per = nq / nkv;
        const gOf = (h) => (valid ? Math.floor(h / per) : Math.floor((h * nkv) / nq));
        const tpOk = nq % tp === 0;
        const gpuOf = (h) => Math.floor((h * tp) / nq);
        const sets = Array.from({ length: tp }, () => []);
        for (let h = 0; h < nq; h++) { const r = gpuOf(h), k = gOf(h); if (!sets[r].includes(k)) sets[r].push(k); }
        const maxPer = Math.max(...sets.map((s) => s.length));
        const copies = sets.reduce((a, s) => a + s.length, 0);
        return { valid, per, gOf, tpOk, gpuOf, sets, maxPer, replicated: copies > nkv };
      }
      function upd() {
        const m = model();
        rR.set(m.valid ? nf(m.per) : '✕ does not divide');
        rB.set(nf(2 * S.nkv * S.dh * 2) + ' B');
        rG.set(S.tp === 1 ? '(1 GPU)' : nf(2 * m.maxPer * S.dh * 2) + ' B' + (m.replicated ? ' · replicated' : ''));
      }
      upd();
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const m = model();
        const pad = 14;
        const colW = (w - 2 * pad) / S.tp;
        const qpg = S.nq / S.tp;
        const yq = 74, yk = h - 168;
        const rq = Math.max(3, Math.min(8, (colW / Math.ceil(qpg)) * 0.36));
        const grpCol = (k) => C.series[k % C.series.length];
        V.text(g, PRE[S.pre] ? PRE[S.pre].name : 'custom configuration', pad, 20, { size: 13, weight: 600, color: C.ink });
        V.text(g, `${S.nq} query heads · ${S.nkv} K/V heads · d_h = ${S.dh}`, pad, 38, { size: 11.5, color: C.muted, font: 'mono' });
        // GPU columns
        for (let r = 0; r < S.tp; r++) {
          const x = pad + r * colW;
          if (S.tp > 1) {
            V.box(g, x + 3, 48, colW - 6, h - 48 - 96, { r: 10, fill: V.alpha(C.muted, 0.05), stroke: V.alpha(C.muted, 0.22), lw: 1 });
            V.text(g, 'GPU ' + r, x + colW - 10, 62, { size: 11, align: 'right', font: 'mono', color: C.muted });
          }
        }
        // positions
        const qx = (hh) => { const r = m.gpuOf(hh); const first = Math.ceil((r * S.nq) / S.tp); const cnt = Math.ceil(((r + 1) * S.nq) / S.tp) - first; const idx = hh - first; return pad + r * colW + ((idx + 0.5) * colW) / Math.max(1, cnt); };
        const kx = (r, k) => { const list = m.sets[r]; const idx = list.indexOf(k); return pad + r * colW + ((idx + 0.5) * colW) / list.length; };
        const rk = Math.max(6, Math.min(16, (colW / Math.max(...m.sets.map((s) => s.length))) * 0.3));
        const seen = new Set();
        // lines + pulses
        for (let hh = 0; hh < S.nq; hh++) {
          const r = m.gpuOf(hh), k = m.gOf(hh);
          const a = [kx(r, k), yk - rk], b = [qx(hh), yq + rq];
          const col = m.valid ? grpCol(k) : RED(C);
          V.line(g, a[0], a[1], b[0], b[1], { color: V.alpha(col, 0.35), lw: 1 });
          const u = (t * 0.45 + k * 0.13 + (hh % 3) * 0.05) % 1;
          V.dot(g, V.lerp(a[0], b[0], u), V.lerp(a[1], b[1], u), 2.2, col);
        }
        // query heads
        for (let hh = 0; hh < S.nq; hh++) {
          const k = m.gOf(hh), col = m.valid ? grpCol(k) : RED(C);
          V.dot(g, qx(hh), yq, rq, V.alpha(col, 0.9));
        }
        V.text(g, 'queries', pad + 4, yq - rq - 8, { size: 11, color: C.muted });
        // KV heads (copies per GPU)
        for (let r = 0; r < S.tp; r++) {
          m.sets[r].forEach((k) => {
            const x = kx(r, k), col = m.valid ? grpCol(k) : RED(C);
            const dup = seen.has(k); seen.add(k);
            V.glow(g, x, yk, rk * 1.8, col, 0.35);
            V.dot(g, x, yk, rk, V.alpha(col, dup ? 0.45 : 0.95));
            if (dup) { g.setLineDash([3, 3]); g.beginPath(); g.arc(x, yk, rk + 3, 0, Math.PI * 2); g.strokeStyle = col; g.lineWidth = 1.4; g.stroke(); g.setLineDash([]); }
            if (rk >= 9) V.text(g, 'KV' + k, x, yk + 4, { size: 11, align: 'center', font: 'mono', weight: 600, color: onInk(C) });
          });
        }
        V.text(g, 'K/V', pad + 4, yk + rk + 16, { size: 11, color: C.muted });
        // KV row per GPU (thumbnail)
        const ly = yk + rk + 26;
        for (let r = 0; r < S.tp; r++) {
          const list = m.sets[r];
          const x = pad + r * colW + 10, avail = colW - 20;
          const cells = list.length * 2, cw = Math.min(18, (avail - 4) / Math.max(1, cells));
          list.forEach((k, i) => {
            V.box(g, x + i * cw, ly, cw - 2, 12, { r: 3, fill: V.alpha(C.a, 0.8) });
            V.box(g, x + (list.length + i) * cw, ly, cw - 2, 12, { r: 3, fill: V.alpha(C.b, 0.8) });
          });
          const lbl = nf(2 * list.length * S.dh * 2) + ' B/token';
          const lx2 = x + cells * cw + 6;
          if (lx2 + tw(g, lbl, 11, 'mono') < pad + (r + 1) * colW - 6) V.text(g, lbl, lx2, ly + 10, { size: 11, font: 'mono', color: C.muted });
        }
        // status
        let msg, col;
        if (!m.valid) { msg = `${S.nq} % ${S.nkv} ≠ 0: unequal groups. The kernel requires qo % kv == 0.`; col = RED(C); }
        else if (!m.tpOk) { msg = `${S.nq} % ${S.tp} ≠ 0: the heads do not split across the GPUs (hence the 32/8 → 36/9 padding).`; col = RED(C); }
        else if (m.replicated) { msg = `n_kv = ${S.nkv} < TP = ${S.tp} (or does not divide it): the KV head is copied onto each GPU.`; col = C.d; }
        else { msg = `✓ groups of ${nf(m.per)} queries; the KV ${S.tp > 1 ? 'splits across the GPUs' : 'stays on a single GPU'}.`; col = C.e; }
        textBlock(g, msg, pad, h - 58, w - 2 * pad, { size: 12.5, weight: 600, color: col, maxLines: 2 });
        const ratio = sf(S.nq / S.nkv, S.nq % S.nkv ? 1 : 0);
        V.text(g, w < 560 ? `${nf(2 * S.nkv * S.dh * 2)} B/token/layer · MHA: ${nf(2 * S.nq * S.dh * 2)} B (${ratio}×)` : `row per token and layer: ${nf(2 * S.nkv * S.dh * 2)} B in BF16 · MHA would be ${nf(2 * S.nq * S.dh * 2)} B (${ratio}×)`, pad, h - 18, { size: 11, font: 'mono', color: C.muted, maxW: w - 2 * pad });
      });
    })();

    /* ================================================== Fig 3 — low rank */
    (function fig3() {
      const fig = root.querySelector('#fig-baixo-posto');
      if (!fig) return;
      const narrowInit = sizeStage(fig, 420, 640);
      const N = 10;
      const SV = [9, 6, 3.6, 1.3, 0.6, 0.35, 0.22, 0.14, 0.08, 0.04];
      function orth(seed) {
        const r = V.rng(seed), M = [];
        for (let k = 0; k < N; k++) {
          let v = Array.from({ length: N }, () => r.normal());
          for (const u of M) { const d = v.reduce((s, x, i) => s + x * u[i], 0); v = v.map((x, i) => x - d * u[i]); }
          const nn = Math.hypot(...v); M.push(v.map((x) => x / nn));
        }
        return M;
      }
      const U = orth(11), Vv = orth(29);
      const W = Array.from({ length: N }, (_, i) => Array.from({ length: N }, (_, j) => SV.reduce((s, sk, k) => s + sk * U[k][i] * Vv[k][j], 0)));
      const MW = Math.max(...W.flat().map(Math.abs));
      const Afull = Array.from({ length: N }, (_, i) => Array.from({ length: N }, (_, k) => U[k][i] * Math.sqrt(SV[k])));
      const Bfull = Array.from({ length: N }, (_, k) => Array.from({ length: N }, (_, j) => Math.sqrt(SV[k]) * Vv[k][j]));
      const MA = Math.max(...Afull.flat().map(Math.abs)), MB = Math.max(...Bfull.flat().map(Math.abs));
      const recon = (r) => Array.from({ length: N }, (_, i) => Array.from({ length: N }, (_, j) => { let s = 0; for (let k = 0; k < r; k++) s += Afull[i][k] * Bfull[k][j]; return s; }));
      const tot2 = SV.reduce((s, x) => s + x * x, 0);
      const S = { r: 3, prev: recon(3), cur: recon(3), tSw: -10, hi: [2, 5], hover: null };
      let H = null;
      const ctl = UI.controls(fig);
      UI.slider(ctl, { label: 'Rank r', min: 1, max: 10, step: 1, value: 3, fmt: (v) => 'r = ' + v, onInput: (v) => { S.prev = disp(); S.r = v; S.cur = recon(v); S.tSw = H ? H.t : 0; upd(); } });
      const rP = UI.readout(ctl, 'Numbers stored', '—');
      const rE = UI.readout(ctl, 'Relative error', '—');
      function upd() {
        rP.set(`${S.r * 2 * N} of ${N * N}`);
        const err = Math.sqrt(SV.slice(S.r).reduce((s, x) => s + x * x, 0) / tot2);
        rE.set(sf(err * 100, 1) + '%');
      }
      function disp() {
        if (!H) return S.cur;
        const e = V.ease.inOut(V.clamp((H.t - S.tSw) / 0.4, 0, 1));
        return S.cur.map((row, i) => row.map((x, j) => V.lerp(S.prev[i][j], x, e)));
      }
      upd();
      const seq = V.rng(5);
      const path = Array.from({ length: 60 }, () => [seq.int(N), seq.int(N)]);
      H = V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const narrow = w < 600;
        const Wr = disp();
        const r = S.r;
        const A = Afull.map((row) => row.slice(0, r));
        const B = Bfull.slice(0, r);
        const [hi, hj] = S.hover || path[Math.floor(t / 1.1) % path.length];
        const colW = (v) => V.cmap(v / MW, 'div');
        let geo = {};
        if (!narrow) {
          const c = Math.max(10, Math.min(17, Math.floor((w - 140) / 40) - 2));
          const step = c + 2;
          const y0 = 44;
          const aw = Math.max(r * step, 78);
          const xW = 16, xA = xW + N * step + 34, xB = xA + aw + 26, xWr = w - 16 - N * step;
          geo = { c, y0, xW, xA, xB, xWr, yA: y0, yB: y0, yWr: y0 };
          V.text(g, '≈', xW + N * step + 10, y0 + (N * step) / 2 + 6, { size: 20, color: C.muted });
          V.text(g, '×', xA + aw + 8, y0 + (N * step) / 2 + 6, { size: 16, color: C.muted });
          V.text(g, '=', (xB + N * step + xWr) / 2 - 6, y0 + (N * step) / 2 + 6, { size: 20, color: C.muted });
        } else {
          const c = Math.max(9, Math.min(15, Math.floor((w - 32 - 24) / 20) - 2));
          const step = c + 2;
          const xA = 16, xB = xA + Math.max(r * step, 78) + 24;
          const y1 = 40, y2 = y1 + N * step + 40;
          const xW = 16, xWr = xW + N * step + 24;
          geo = { c, y0: y1, xW, xA, xB, xWr, yA: y1, yB: y1, yW: y2, yWr: y2 };
          V.text(g, '×', xA + Math.max(r * step, 78) + 7, y1 + (N * step) / 2 + 6, { size: 16, color: C.muted });
          V.text(g, '≈', xW + N * step + 7, y2 + (N * step) / 2 + 6, { size: 18, color: C.muted });
        }
        const { c } = geo;
        const yW = geo.yW != null ? geo.yW : geo.y0;
        // W original
        V.matrix(g, { x: geo.xW, y: yW, data: W, cell: c, gap: 2, cmap: 'div', min: -MW, max: MW, hiCells: [[hi, hj]], hiColor: C.c });
        V.text(g, 'original W (10 × 10)', geo.xW, yW - 8, { size: 11, color: C.muted, font: 'mono' });
        // A and B
        V.matrix(g, { x: geo.xA, y: geo.yA, data: A, cell: c, gap: 2, cmap: 'div', min: -MA, max: MA, hiRow: hi, hiColor: C.c });
        V.text(g, `A (10 × ${r})`, geo.xA, geo.yA - 8, { size: 11, color: C.muted, font: 'mono' });
        V.matrix(g, { x: geo.xB, y: geo.yB, data: B, cell: c, gap: 2, cmap: 'div', min: -MB, max: MB, hiCol: hj, hiColor: C.c });
        V.text(g, `B (${r} × 10)`, geo.xB, geo.yB - 8, { size: 11, color: C.muted, font: 'mono' });
        // W_r
        const gWr = V.matrix(g, { x: geo.xWr, y: geo.yWr, data: Wr, cell: c, gap: 2, cmap: 'div', min: -MW, max: MW, hiCells: [[hi, hj]], hiColor: C.c });
        V.text(g, `W_r = A·B (rank ${r})`, geo.xWr, geo.yWr - 8, { size: 11, color: C.muted, font: 'mono' });
        // pointer
        const gW = { x: geo.xW, y: yW, step: c + 2 };
        const pt = H.pointer;
        const hit = (x0, y0) => { const j = Math.floor((pt.x - x0) / (c + 2)), i = Math.floor((pt.y - y0) / (c + 2)); return i >= 0 && i < N && j >= 0 && j < N ? [i, j] : null; };
        S.hover = pt.inside ? (hit(gW.x, gW.y) || hit(gWr.x, gWr.y)) : null;
        // spectrum + text
        const yS = (geo.yW != null ? geo.yW : geo.y0) + N * (c + 2) + 34;
        const sw = Math.min(narrow ? w - 32 : 250, 260), sh = narrow ? 80 : Math.max(46, h - yS - 34);
        V.text(g, 'singular values σ (energy of each direction)', 16, yS - 10, { size: 11, color: C.muted });
        V.bars(g, { x: 16, y: yS, w: sw, h: sh, values: SV, labels: SV.map((_, k) => String(k + 1)), colorFn: (v, k) => (k < r ? C.a : V.alpha(C.muted, 0.35)), gap: 5 });
        const tx = narrow ? 16 : 16 + sw + 30, ty = narrow ? yS + sh + 36 : yS + 4, tW = narrow ? w - 32 : w - tx - 16;
        {
          const Wv = W[hi][hj], Rv = Wr[hi][hj];
          let yy = ty;
          if (narrow) {
            V.text(g, `cell (${hi + 1}, ${hj + 1}):  W = ${sf(Wv, 2)}   W_r = ${sf(Rv, 2)}`, tx, yy, { size: 12, color: C.ink, font: 'mono', weight: 600, maxW: tW });
            yy += 17;
            yy += textBlock(g, `= row ${hi + 1} of A · column ${hj + 1} of B`, tx, yy, tW, { size: 11.5, color: C.c, font: 'mono' }) + 6;
            textBlock(g, `A and B store r × 20 = ${r * 20} numbers, versus 100 for W. In GLM's MLA: W_DKV + kv_b_proj ≈ 18.9 M parameters, and only the 512-wide bottleneck goes into the cache.`, tx, yy, tW, { size: 11.5, color: C.muted, maxLines: 4 });
          } else {
            V.text(g, `cell (${hi + 1}, ${hj + 1}):  W = ${sf(Wv, 2)}   W_r = ${sf(Rv, 2)}`, tx, yy, { size: 13, color: C.ink, font: 'mono', weight: 600 });
            yy += 18;
            yy += textBlock(g, `W_r(${hi + 1}, ${hj + 1}) = row ${hi + 1} of A · column ${hj + 1} of B`, tx, yy, tW, { size: 12, color: C.c, font: 'mono' }) + 10;
            yy += textBlock(g, `Storing A and B costs r × (10 + 10) = ${r * 20} numbers, versus 100 for W. In GLM's MLA: W_DKV [512 × 4096] + kv_b_proj [32768 × 512] ≈ 18.9 M parameters, and only the 512-wide bottleneck goes into the cache.`, tx, yy, tW, { size: 12, color: C.muted }) + 8;
            textBlock(g, 'Hover over W or W_r to pick a cell.', tx, yy, tW, { size: 11.5, color: C.muted });
          }
        }
      });
      if (narrowInit) fig.querySelector('figcaption').insertAdjacentHTML('beforeend', ' Tap a cell of \\(W\\) or \\(W_r\\) to highlight it.');
    })();

    /* ================================================= Fig 4 — MLA stepper */
    (function fig4() {
      const fig = root.querySelector('#fig-mla-fluxo');
      if (!fig) return;
      sizeStage(fig, 440, 640);
      const M4 = {
        v3: { name: 'DeepSeek-V3', hid: 7168, heads: 128, qr: 1536, kvr: 512, nope: 128, rope: 64, v: 128 },
        glm: { name: 'GLM-5.3-Flash', hid: 4096, heads: 64, qr: 1536, kvr: 512, nope: 256, rope: 0, v: 256 },
      };
      const S = { m: 'glm' };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Model', options: [{ v: 'v3', label: 'DeepSeek-V3 (decoupled RoPE)' }, { v: 'glm', label: 'GLM-5.3 (NoPE)' }], value: 'glm', onChange: (v) => { S.m = v; } });
      const steps = [
        { t: 'The token arrives', d: 'The token state \\(x_t\\) (4096 dims in GLM, 7168 in DeepSeek-V3) enters the layer. In an MHA, it would turn into 64 or 128 key and value heads, and all of that would go into the cache.' },
        { t: 'Bring K/V down to the latent', d: 'A single down-projection, \\(W^{DKV}\\) (<code>kv_a_proj_with_mqa</code>), squeezes the token into a 512-dim \\(c^{KV}_t\\) (<code>kv_lora_rank</code>), followed by an RMSNorm.' },
        { t: 'Only the latent goes into the cache', d: 'This is the point of the whole architecture: the cache stores \\(c^{KV}_t\\) (512 numbers) and, in DeepSeek-V3, the 64-dim RoPE key. An MHA of the same size would store 32,768 numbers per token and layer in GLM (64 × (256 + 256)) or 40,960 in V3 (128 × (192 + 128), because there the key has 192 dims and the value 128).' },
        { t: 'Lift to each head', d: '\\(W^{UK}\\) and \\(W^{UV}\\), stacked in <code>kv_b_proj</code>, rebuild the key and value of <i>each</i> head from the same latent. In the next section, absorption will avoid doing this for the entire past at every step.' },
        { t: 'The query has a bottleneck too', d: 'The query goes through \\(W^{DQ}\\) (<code>q_a_proj</code>, 1536 dims, the <code>q_lora_rank</code>), a norm and \\(W^{UQ}\\) (<code>q_b_proj</code>). This bottleneck saves parameters and weight bandwidth; the query is not stored.' },
        { t: 'Position: decoupled or absent', d: 'In DeepSeek-V3, a 64-dim RoPE key, <b>shared</b> by all heads, is computed separately and stored next to the latent (576 = 512 + 64); each head gets its own RoPE query. In GLM-5.3, <code>qk_rope_head_dim = 0</code>: this path does not exist.' },
        { t: 'Per-head attention', d: 'Each head compares its query with the keys and mixes the values, with scale \\(1/\\sqrt{d}\\) (\\(1/\\sqrt{192}\\) in V3, 1/16 in GLM). The attention patterns differ from head to head, despite the common source.' },
        { t: 'Output projection', d: 'The head outputs are concatenated and projected back by \\(W_O\\) (<code>o_proj [4096, 16384]</code> in GLM, the largest dense tensor in the attention, with 67.1 M parameters).' },
      ];
      const st = UI.stepper(fig, steps, { autoplay: 6500 });
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const md = M4[S.m];
        const hasR = md.rope > 0;
        const narrow = w < 600;
        const step = st.i;
        const e = V.clamp(st.elapsed() / 0.6, 0, 1);
        // positions: (column, lane). Wide: columns on x, lanes on y. Narrow: transposed.
        const colsF = [0.06, 0.2, 0.345, 0.49, 0.635, 0.785, 0.93];
        const laneF = { q: 0.18, kv: 0.5, r: 0.8, mid: 0.34 };
        const P = (col, lane) => {
          if (!narrow) return [colsF[col] * w, laneF[lane] * h];
          const lx = { q: 0.2, kv: 0.52, r: 0.84, mid: 0.36 }[lane];
          const sy = [0.065, 0.19, 0.33, 0.475, 0.62, 0.765, 0.9][col];
          return [lx * w, sy * h];
        };
        const bw = narrow ? Math.min(96, w * 0.27) : Math.min(104, w * 0.118), bh = narrow ? 34 : 38;
        const nodes = {
          DQ: { p: P(1, 'q'), base: 'W', sup: 'DQ', sub: 'q_a_proj', dim: `${md.hid}→${md.qr}`, col: C.c },
          cQ: { p: P(2, 'q'), base: 'c', sup: 'Q', sub: '', dim: `[${md.qr}]`, col: C.c, small: true },
          UQ: { p: P(3, 'q'), base: 'W', sup: 'UQ', sub: 'q_b_proj', dim: `${md.qr}→${md.heads}×${md.nope + md.rope}`, col: C.c },
          Qh: { p: P(4, 'q'), base: 'q', sup: '', sub: 'per head', dim: `[${md.heads}×${md.nope + md.rope}]`, col: C.c, heads: true },
          DKV: { p: P(1, 'kv'), base: 'W', sup: 'DKV', sub: 'kv_a_proj…', dim: `${md.hid}→${md.kvr}`, col: C.f },
          cKV: { p: P(2, 'kv'), base: 'c', sup: 'KV', sub: 'latent', dim: `[${md.kvr}]`, col: C.f, small: true },
          UKV: { p: P(3, 'kv'), base: 'W', sup: 'UK·UV', sub: 'kv_b_proj', dim: `${md.kvr}→${md.heads}×(${md.nope}+${md.v})`, col: C.f },
          KVh: { p: P(4, 'kv'), base: 'k, v', sup: '', sub: 'per head', dim: `[${md.heads}×${md.nope}]`, col: C.a, heads: true },
          KR: { p: P(1, 'r'), base: 'W', sup: 'KR', sub: '', dim: `${md.hid}→64`, col: C.d },
          kR: { p: P(2, 'r'), base: 'k', sup: 'R', sub: 'shared', dim: '[64]', col: C.d, small: true },
          ATT: { p: P(5, 'mid'), base: 'attention', sup: '', sub: 'per head', dim: 'softmax(q·k)·v', col: C.b },
          O: { p: P(6, 'mid'), base: 'W', sup: 'O', sub: 'o_proj', dim: `${md.heads}×${md.v}→${md.hid}`, col: C.b },
        };
        const X = narrow ? { x: w * 0.1, y: P(0, 'q')[1] - 14, w: w * 0.8, h: 28 } : { x: colsF[0] * w - 12, y: laneF.q * h - 26, w: 24, h: (laneF.r - laneF.q) * h + 52 };
        const xPort = (lane) => (narrow ? [P(0, lane)[0], X.y + X.h] : [X.x + X.w, P(0, lane)[1]]);
        const actNodes = [
          ['X'], ['X', 'DKV', 'cKV'], ['cKV', 'CACHE', 'kR'], ['cKV', 'UKV', 'KVh'], ['X', 'DQ', 'cQ', 'UQ', 'Qh'], ['KR', 'kR', 'cQ', 'Qh', 'CACHE'], ['Qh', 'KVh', 'ATT', 'kR'], ['ATT', 'O'],
        ][step];
        const isOn = (id) => actNodes.includes(id);
        const edges = [
          { a: 'X', lane: 'q', b: 'DQ', s: 4 }, { a: 'DQ', b: 'cQ', s: 4 }, { a: 'cQ', b: 'UQ', s: 4 }, { a: 'UQ', b: 'Qh', s: 4 },
          { a: 'X', lane: 'kv', b: 'DKV', s: 1 }, { a: 'DKV', b: 'cKV', s: 1 }, { a: 'cKV', b: 'UKV', s: 3 }, { a: 'UKV', b: 'KVh', s: 3 },
          { a: 'X', lane: 'r', b: 'KR', s: 5, r: true }, { a: 'KR', b: 'kR', s: 5, r: true }, { a: 'cQ', b: 'Qh', s: 5, r: true, curve: true },
          { a: 'Qh', b: 'ATT', s: 6 }, { a: 'KVh', b: 'ATT', s: 6 }, { a: 'kR', b: 'ATT', s: 6, r: true, curve2: true }, { a: 'ATT', b: 'O', s: 7 },
        ];
        const port = (id, towards) => {
          if (id === 'X') return null;
          const n = nodes[id]; const ww = n.small ? bw * 0.62 : bw, hh = bh;
          const [x, y] = n.p, [tx, ty] = towards;
          if (!narrow) { if (Math.abs(ty - y) > Math.abs(tx - x) * 1.2) return [x, y + Math.sign(ty - y) * hh / 2]; return [x + Math.sign(tx - x) * ww / 2, y]; }
          if (Math.abs(tx - x) > Math.abs(ty - y) * 1.6) return [x + Math.sign(tx - x) * ww / 2, y];
          return [x, y + Math.sign(ty - y) * hh / 2];
        };
        edges.forEach((ed) => {
          if (ed.r && !hasR) return;
          const bN = nodes[ed.b];
          const aP = ed.a === 'X' ? xPort(ed.lane) : port(ed.a, bN.p);
          const aC = ed.a === 'X' ? aP : nodes[ed.a].p;
          const bP = port(ed.b, aC);
          const active = ed.s === step;
          const col = active ? (nodes[ed.b].col || C.ink) : C.muted;
          let curve = null;
          if (ed.curve) curve = narrow ? [-w * 0.1, 0] : [0, -h * 0.1];
          if (ed.curve2) curve = narrow ? [w * 0.06, 0] : [0, h * 0.06];
          flow(g, aP, bP, { active, color: col, t, curve, lw: active ? 2.2 : 1.2 });
        });
        if (hasR && !narrow) V.text(g, 'q^R (RoPE)', (nodes.cQ.p[0] + nodes.Qh.p[0]) / 2, nodes.cQ.p[1] - h * 0.075, { size: 11, align: 'center', color: step === 5 ? C.d : C.muted, font: 'mono' });
        // X block
        const xOn = isOn('X');
        V.box(g, X.x, X.y, X.w, X.h, { r: 8, fill: V.alpha(C.ink, xOn ? 0.16 : 0.06), stroke: xOn ? C.ink : V.alpha(C.muted, 0.5), lw: xOn ? 2 : 1 });
        if (narrow) V.text(g, `x_t [${md.hid}] · token state`, X.x + X.w / 2, X.y + X.h / 2 + 4, { size: 12, align: 'center', weight: 600, color: C.ink });
        else {
          g.save(); g.translate(X.x + X.w / 2, X.y + X.h / 2); g.rotate(-Math.PI / 2);
          V.text(g, `x_t [${md.hid}]`, 0, 4, { size: 12, align: 'center', weight: 600, color: C.ink }); g.restore();
        }
        // cache box
        const cacheIds = hasR ? ['cKV', 'kR'] : ['cKV'];
        const cxs = cacheIds.map((id) => nodes[id].p);
        const cw = bw * 0.62;
        let bx0 = Math.min(...cxs.map((p) => p[0])) - cw / 2 - 10, bx1 = Math.max(...cxs.map((p) => p[0])) + cw / 2 + 10;
        // the bottom edge sits below the dimensions (and, on phones, below the tensor label)
        let by0 = Math.min(...cxs.map((p) => p[1])) - bh / 2 - 22, by1 = Math.max(...cxs.map((p) => p[1])) + bh / 2 + (narrow ? 32 : 24);
        const cOn = isOn('CACHE');
        if (cOn) V.glow(g, (bx0 + bx1) / 2, (by0 + by1) / 2, Math.max(bx1 - bx0, by1 - by0) * 0.7, C.f, 0.35 + 0.15 * Math.sin(t * 3));
        g.setLineDash(cOn ? [] : [4, 4]);
        V.box(g, bx0, by0, bx1 - bx0, by1 - by0, { r: 12, fill: V.alpha(C.f, cOn ? 0.1 : 0.03), stroke: cOn ? C.f : V.alpha(C.f, 0.55), lw: cOn ? 2.2 : 1.2 });
        g.setLineDash([]);
        V.box(g, bx0 + 4, by0 + 3, tw(g, 'KV cache', 11, 'mono', 700) + 8, 15, { r: 4, fill: V.alpha(C.bg || C.panel, 0.9) });
        V.text(g, 'KV cache', bx0 + 8, by0 + 14, { size: 11, weight: 700, color: C.f, font: 'mono' });
        if (cOn) {
          const txt = hasR ? `per token: 512 + 64 = 576 numbers (MHA: ${nf(md.heads * (md.nope + md.rope + md.v))})` : `per token: 512 numbers (MHA: ${nf(md.heads * (md.nope + md.v))})`;
          const tx = narrow ? w / 2 : (bx0 + bx1) / 2, ty = by1 + 16;
          if (narrow) { const tww = Math.min(w - 20, tw(g, txt, 11.5, 'sans', 600)) + 10; V.box(g, w / 2 - tww / 2, ty - 12, tww, 17, { r: 5, fill: V.alpha(C.bg || C.panel, 0.92) }); }
          V.text(g, txt, Math.max(tx, 10), ty, { size: 11.5, align: narrow ? 'center' : 'left', color: C.f, weight: 600, maxW: narrow ? w - 20 : w - bx0 - 10 });
        }
        // nodes
        Object.entries(nodes).forEach(([id, n]) => {
          if ((id === 'KR' || id === 'kR') && !hasR) {
            const [x, y] = n.p, ww = n.small ? bw * 0.62 : bw;
            g.setLineDash([3, 4]); V.rr(g, x - ww / 2, y - bh / 2, ww, bh, 8); g.strokeStyle = V.alpha(C.muted, 0.45); g.lineWidth = 1; g.stroke(); g.setLineDash([]);
            if (id === 'KR') {
              // on desktop, the notice sits below the two dashed boxes (KR and kR), without crossing their edges
              const xm = narrow ? x : (x + nodes.kR.p[0]) / 2;
              V.text(g, narrow ? '✕ no RoPE' : '✕ qk_rope_head_dim = 0', xm, narrow ? y + 4 : y + bh / 2 + 16, { size: 11, align: 'center', color: step === 5 ? RED(C) : C.muted, font: 'mono' });
            }
            return;
          }
          const on = isOn(id);
          const [x, y] = n.p, ww = n.small ? bw * 0.62 : bw;
          const pop = on ? 1 + 0.06 * Math.sin(Math.PI * Math.min(1, e)) : 1;
          const W2 = ww * pop, H2 = bh * pop;
          if (n.heads) {
            for (let k = 2; k >= 1; k--) V.box(g, x - W2 / 2 + k * 4, y - H2 / 2 - k * 4, W2, H2, { r: 8, fill: V.alpha(n.col, on ? 0.12 : 0.04), stroke: V.alpha(n.col, on ? 0.5 : 0.2), lw: 1 });
          }
          V.box(g, x - W2 / 2, y - H2 / 2, W2, H2, { r: 8, fill: V.alpha(n.col, on ? 0.24 : 0.07), stroke: on ? n.col : V.alpha(n.col, 0.4), lw: on ? 2 : 1 });
          supLabel(g, n.base, n.sup, x, y - (n.sub && !narrow ? 1 : -4), { size: narrow ? 12 : 13, align: 'center', color: on ? C.ink : C.muted });
          if (n.sub && !narrow) V.text(g, n.sub, x, y + 13, { size: 11, align: 'center', font: 'mono', color: on ? C.muted : V.alpha(C.muted, 0.7), maxW: W2 - 6 });
          const dimY = narrow ? y + H2 / 2 + 12 : y + H2 / 2 + 14;
          // on phones, the active node's labels get a background, so the arrows pass behind them
          const under = (s, yy, col) => {
            if (narrow) { const ww2 = Math.min(w * 0.3, tw(g, s, 11, 'mono')) + 8; V.box(g, x - ww2 / 2, yy - 10, ww2, 13, { r: 4, fill: V.alpha(C.bg || C.panel, 0.94) }); }
            V.text(g, s, x, yy, { size: 11, align: 'center', font: 'mono', color: col, maxW: narrow ? w * 0.3 : bw * 1.35 });
          };
          if (on || !narrow) under(n.dim, dimY, on ? n.col : V.alpha(C.muted, 0.75));
          // on phones, the active node's tensor name goes right below the dimension
          if (narrow && on && n.sub) under(n.sub, dimY + 13, C.muted);
        });
        V.text(g, md.name, narrow ? w - 10 : 14, narrow ? h - 8 : h - 12, { size: 12, weight: 600, align: narrow ? 'right' : 'left', color: C.muted });
      });
    })();

    /* ================================================== Fig 5 — absorption */
    (function fig5() {
      const fig = root.querySelector('#fig-absorcao');
      if (!fig) return;
      sizeStage(fig, 484, 720);
      const c = [[2, -1, 1], [0, 3, -1], [-2, 1, 2], [1, 1, 0], [-1, -2, 2]];
      const Wuk = [[1, 0, 1], [0, 2, -1], [-1, 1, 2], [2, 0, 1]];
      const q = [2, -1, 1, 1];
      const T0 = c.length;
      // toy RoPE: rotates the pairs (0,1) and (2,3) by j × 90° (keeps the integers)
      const rot = (v, j) => { let a = v.slice(); const n = ((j % 4) + 4) % 4; for (let k = 0; k < n; k++) a = [-a[1], a[0], -a[3], a[2]]; return a; };
      const S = { rope: false, logT: 17 };
      const ctl = UI.controls(fig);
      UI.toggle(ctl, { label: 'RoPE inside the key (coupled)', value: false, onChange: (v) => { S.rope = v; } });
      UI.slider(ctl, { label: 'Cache (T)', min: 10, max: 20, step: 1, value: 17, fmt: (v) => kfmt(2 ** v), onInput: (v) => { S.logT = v; } });
      function compute() {
        const K = c.map((cj, j) => { const k = Wuk.map((row) => row.reduce((s, x, i) => s + x * cj[i], 0)); return S.rope ? rot(k, j) : k; });
        const sA = K.map((k) => k.reduce((s, x, i) => s + x * q[i], 0));
        const qtOf = (j) => { const qq = S.rope ? rot(q, -j) : q; return [0, 1, 2].map((kk) => Wuk.reduce((s, row, i) => s + row[kk] * qq[i], 0)); };
        const QT = S.rope ? c.map((_, j) => qtOf(j)) : [qtOf(0)];
        const sB = c.map((cj, j) => { const qt = S.rope ? QT[j] : QT[0]; return cj.reduce((s, x, i) => s + x * qt[i], 0); });
        return { K, sA, QT, sB };
      }
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const narrow = w < 600;
        const R = compute();
        const cw = narrow ? Math.max(24, Math.min(30, Math.floor((w - 104) / 8) - 2)) : 36;
        const ch = narrow ? 21 : 23;
        const st = cw + 2, sth = ch + 2;
        const fsz = cw >= 32 ? 12.5 : 11.5;
        const jHi = Math.floor(t / 0.9) % T0;
        /** matrix with readable integers */
        const mat = (x, y, data, label, o = {}) => {
          const mx = Math.max(1, ...data.flat().map(Math.abs));
          data.forEach((row, i) => row.forEach((v, j) => {
            const tt = v / mx;
            const xx = x + j * st, yy = y + i * sth;
            V.box(g, xx, yy, cw, ch, { r: 4, fill: V.cmap(tt, 'div') });
            const light = Math.abs(tt) > 0.55;
            V.text(g, String(Math.round(v)), xx + cw / 2, yy + ch / 2 + 0.5, { size: fsz, font: 'mono', align: 'center', base: 'middle', color: light && C.dark ? '#00210F' : light ? '#ffffff' : C.ink });
          }));
          const W2 = data[0].length * st - 2, H2 = data.length * sth - 2;
          if (o.stroke) { g.strokeStyle = o.stroke; g.lineWidth = 2; V.rr(g, x - 3, y - 3, W2 + 6, H2 + 6, 6); g.stroke(); }
          if (o.hiRow != null && o.hiRow < data.length) { g.strokeStyle = o.hiColor || C.c; g.lineWidth = 2.5; V.rr(g, x - 2, y + o.hiRow * sth - 2, W2 + 4, ch + 4, 5); g.stroke(); }
          if (label) V.text(g, label, x, y - 8, { size: 11, color: o.labelColor || C.muted, font: 'mono' });
        };
        const arrowTxt = (x1, y, x2, s) => { V.arrow(g, x1, y, x2, y, { color: C.muted, lw: 1.4, head: 6 }); if (s) V.text(g, s, (x1 + x2) / 2, y - 8, { size: 11, align: 'center', color: C.muted, font: 'mono' }); };
        const gA1 = narrow ? 30 : 42, gA2 = narrow ? 26 : 36;
        let xIn, yIn, xR, yA, yB, yCnt;
        if (!narrow) { xIn = 16; yIn = 60; xR = 16 + 5 * st + 60; yA = 52; yB = 226; yCnt = 398; }
        else { xIn = 14; yIn = 40; xR = 14; yA = 188; yB = 364; yCnt = 540; }
        // inputs
        V.text(g, 'inputs', xIn, yIn - 26, { size: 12, weight: 600, color: C.ink });
        mat(xIn, yIn, q.map((v) => [v]), 'q');
        mat(xIn + st + 26, yIn, Wuk, 'W^UK (4 × 3)');
        V.text(g, 'd_h = 4 · d_c = 3', xIn + (narrow ? st + 26 + 3 * st + 16 : 0), narrow ? yIn + 14 : yIn + 4 * sth + 20, { size: 11, color: C.muted, font: 'mono' });
        if (narrow) V.text(g, S.rope ? 'RoPE: R_j rotates 90° × j' : 'no RoPE', xIn + st + 26 + 3 * st + 16, yIn + 32, { size: 11, color: S.rope ? C.d : C.muted, font: 'mono' });
        // route A
        V.text(g, 'Route A · rebuild the key of each token', xR, yA - 26, { size: 12, weight: 600, color: C.a, maxW: w - xR - 10 });
        let x = xR;
        mat(x, yA, c, 'cache c (5 × 3)', { hiRow: jHi });
        x += 3 * st + 6; arrowTxt(x, yA + 2.5 * sth, x + gA1 - 8, S.rope ? 'W,R_j' : '·Wᵀ'); x += gA1;
        mat(x, yA, R.K, S.rope ? 'K = R_j W c_j' : 'K (5 × 4), recomputed', { hiRow: jHi });
        x += 4 * st + 6; arrowTxt(x, yA + 2.5 * sth, x + gA2 - 8, '·q'); x += gA2;
        mat(x, yA, R.sA.map((v) => [v]), 's_A', { hiRow: jHi });
        const xEndA = x + cw;
        // route B
        V.text(g, 'Route B · absorb W^UK into the query', xR, yB - 26, { size: 12, weight: 600, color: C.f, maxW: w - xR - 10 });
        x = xR;
        if (!S.rope) {
          const pulse = 0.5 + 0.5 * Math.sin(t * 3);
          V.glow(g, x + cw / 2, yB + 1.5 * sth, 30, C.f, 0.2 + 0.2 * pulse);
          mat(x, yB, R.QT[0].map((v) => [v]), 'q̃ = Wᵀq', { stroke: C.f, labelColor: C.f });
          V.text(g, 'once', x, yB + 3 * sth + 14, { size: 11, color: C.f, font: 'mono', weight: 600 });
          x += st + (narrow ? 34 : 44);
        } else {
          mat(x, yB, R.QT, 'q̃_j = Wᵀ R_jᵀ q', { hiRow: jHi, hiColor: RED(C), labelColor: RED(C) });
          V.text(g, 'one per token!', x, yB + 5 * sth + 14, { size: 11, color: RED(C), font: 'mono', weight: 700 });
          x += 3 * st + (narrow ? 14 : 22);
        }
        mat(x, yB, c, 'cache c', { hiRow: jHi });
        x += 3 * st + 6; arrowTxt(x, yB + 2.5 * sth, x + gA2 - 8, '·q̃'); x += gA2;
        mat(x, yB, R.sB.map((v) => [v]), 's_B', { hiRow: jHi });
        const xEndB = x + cw;
        // equality
        const eq = R.sA.every((v, i) => Math.abs(v - R.sB[i]) < 1e-9);
        const ex = xEndB + 26;
        if (ex + 130 < w) {
          const ym = yB + 2.5 * sth;
          V.text(g, eq ? '✓ s_A = s_B' : '✕', ex, ym - 4, { size: 15, weight: 700, color: eq ? C.e : RED(C) });
          V.text(g, 'the same scores', ex, ym + 14, { size: 11.5, color: C.muted });
        } else {
          V.text(g, eq ? '✓ s_A = s_B: the same scores' : '✕', xR, yB + 5 * sth + (S.rope ? 34 : 20), { size: 12, weight: 700, color: eq ? C.e : RED(C) });
        }
        // counters with real dimensions
        const T = 2 ** S.logT, Hh = 64, dh = 256, dc = 512;
        const flA = 4 * T * dc * Hh * dh + 4 * T * Hh * dh;
        const flB = 2 * Hh * dh * dc * 2 + 4 * T * Hh * dc;
        const kvFull = T * 2 * Hh * dh * 2, kvLat = T * 656, kvLatBf = T * dc * 2;
        V.line(g, 14, yCnt - 16, w - 14, yCnt - 16, { color: V.alpha(C.muted, 0.25), lw: 1 });
        let yy = yCnt;
        const tW = w - 28;
        yy += textBlock(g, `One GLM MLA layer (64 heads, d_h = 256, d_c = 512), dense attention over T = ${kfmt(T)} tokens, per decode step:`, 14, yy, tW, { size: 11.5, color: C.muted }) + 4;
        yy += textBlock(g, `Route A: recomputing K and V costs ${fmtFlop(flA)}; storing them ready-made (BF16) costs ${fmtGiB(kvFull)} of cache.`, 14, yy, tW, { size: 12, color: C.a, font: 'mono' }) + 3;
        textBlock(g, S.rope ? 'Route B with coupled RoPE: one q̃ per token, cost ≈ route A. That is why RoPE has to be decoupled.' : `Route B: ${fmtFlop(flB)}, reading only the latent: ${fmtGiB(kvLat)} in the FP8 fp8_ds_mla layout (656 B/token), or ${fmtGiB(kvLatBf)} in BF16 (1,024 B/token).`, 14, yy, tW, { size: 12, color: S.rope ? RED(C) : C.f, font: 'mono', weight: 600 });
      });
    })();

    /* ================================================== Fig 6 — pad 576 */
    (function fig6() {
      const fig = root.querySelector('#fig-pad576');
      if (!fig) return;
      sizeStage(fig, 420, 470);
      const r = V.rng(77);
      const NB = 16;
      const qv = Array.from({ length: NB }, () => +(r.normal() * 0.8).toFixed(2));
      const kv = Array.from({ length: NB }, () => +(r.normal() * 0.8).toFixed(2));
      const garbageQ = [0.7, -0.9], garbageK = [1.1, 0.8];
      const S = { mode: 'pad', qz: true, kz: true };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Geometry', options: [{ v: 'nat', label: 'native (512)' }, { v: 'pad', label: 'GLM_NSA 576 (rope_pad)' }, { v: 'tf', label: 'TensorFold (512, own kernel)' }], value: 'pad', onChange: (v) => { S.mode = v; gZ.show(v === 'pad'); } });
      const gZ = UI.group(ctl);
      UI.toggle(gZ.el, { label: 'query with 64 zeros (F.pad)', value: true, onChange: (v) => { S.qz = v; } });
      UI.toggle(gZ.el, { label: 'k_pe written as zeros', value: true, onChange: (v) => { S.kz = v; } });
      const exact = qv.reduce((s, x, i) => s + x * kv[i], 0);
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const narrow = w < 600;
        const pad = 14;
        const cw = Math.max(12, Math.min(36, Math.floor((w - 2 * pad) / (NB + 2 + 2.6 + 0.8))));
        const x0 = pad, gapS = cw * 0.4;
        const scW = cw * 0.55;
        const xScale0 = x0 + NB * cw + gapS * 0.5;
        const xRope0 = xScale0 + 4 * scW + gapS * 0.5;
        const yQ = 58, yK = narrow ? 140 : 132, ch = narrow ? 28 : 30;
        const qp = S.qz ? [0, 0] : garbageQ, kp = S.kz ? [0, 0] : garbageK;
        const blocks = NB + 2;
        const period = 0.28, cyc = blocks + 5;
        const b = Math.floor(t / period) % cyc;
        const nat = S.mode === 'nat', tf = S.mode === 'tf';
        // titles
        const tQ = tf ? (narrow ? 'query: 512, no padding' : 'absorbed query: 512 dims (NoPE), no padding') : nat ? 'absorbed query: 512 dims (NoPE)' : narrow ? 'query: 512 + 64 zeros = 576' : 'absorbed query: 512 NoPE + 64 of padding = 576';
        const tK = tf ? (narrow ? 'kv8: 512 FP8 + 4 B + 12 B' : 'TensorFold kv8 row: 512 FP8 + 2^k scale (4 B) + 12 B of zeros') : nat ? (narrow ? 'cache: 512 FP8 + scales' : 'cache row: 512 FP8 + scales (no RoPE)') : narrow ? 'cache: 512 FP8 + 16 B + 64 BF16' : 'fp8_ds_mla cache row: 512 FP8 + 16 B of scales + 64 BF16';
        V.text(g, tQ, x0, yQ - 12, { size: 12, weight: 600, color: C.c });
        V.text(g, tK, x0, yK - 12, { size: 12, weight: 600, color: C.f });
        const mxv = 1.6;
        for (let i = 0; i < NB; i++) {
          const x = x0 + i * cw, act = i === b;
          V.box(g, x + 1, yQ, cw - 2, ch, { r: 4, fill: V.cmap(qv[i] / mxv, 'div'), stroke: act ? C.c : null, lw: 2 });
          V.box(g, x + 1, yK, cw - 2, ch, { r: 4, fill: V.cmap(kv[i] / mxv, 'div'), stroke: act ? C.c : null, lw: 2 });
        }
        // scales
        for (let i = 0; i < 4; i++) {
          const xs = xScale0 + i * scW + 1, ys = yK + ch * 0.25;
          if (tf && i > 0) { V.box(g, xs, ys, scW - 2, ch * 0.5, { r: 2, fill: V.alpha(C.muted, 0.1) }); stripes(g, xs, ys, scW - 2, ch * 0.5, V.alpha(C.muted, 0.45), 4); }
          else V.box(g, xs, ys, scW - 2, ch * 0.5, { r: 2, fill: V.alpha(C.d, 0.55), stroke: tf && b >= NB && b < NB + 3 ? C.c : null, lw: 2 });
        }
        if (cw >= 20) V.text(g, 'scales', xScale0 + 2 * scW, yK + ch + 13, { size: 11, align: 'center', color: C.muted, font: 'mono' });
        // RoPE / padding block
        if (tf) {
          // TensorFold: the row has no RoPE block
          const lx = narrow ? w - pad : xRope0 + cw, la = narrow ? 'right' : 'center';
          V.text(g, '−128 B', lx, yK + ch / 2 + 4, { size: 12, align: la, font: 'mono', weight: 700, color: C.e });
          V.text(g, 'no RoPE', lx, yK + ch + 13, { size: 11, align: la, font: 'mono', color: C.muted });
        } else if (!nat) {
          for (let i = 0; i < 2; i++) {
            const x = xRope0 + i * cw, act = b === NB + i;
            const qvv = qp[i], kvv = kp[i];
            V.box(g, x + 1, yQ, cw - 2, ch, { r: 4, fill: qvv === 0 ? V.alpha(C.muted, 0.12) : V.cmap(qvv / mxv, 'div'), stroke: act ? C.c : V.alpha(C.muted, 0.5), lw: act ? 2 : 1 });
            if (qvv === 0) V.text(g, '0', x + cw / 2, yQ + ch / 2 + 4, { size: 11, align: 'center', font: 'mono', color: C.muted });
            V.box(g, x + 1, yK, cw - 2, ch, { r: 4, fill: kvv === 0 ? V.alpha(C.d, 0.1) : V.cmap(kvv / mxv, 'div'), stroke: act ? C.c : V.alpha(C.d, 0.6), lw: act ? 2 : 1 });
            if (kvv === 0) stripes(g, x + 1, yK, cw - 2, ch, V.alpha(C.d, 0.55), 5);
          }
        } else {
          // missing slot
          g.setLineDash([4, 3]);
          V.rr(g, xRope0 + 1, yK, 2 * cw - 2, ch, 4); g.strokeStyle = RED(C); g.lineWidth = 1.5; g.stroke();
          V.rr(g, xRope0 + 1, yQ, 2 * cw - 2, ch, 4); g.stroke();
          g.setLineDash([]);
          V.text(g, '?', xRope0 + cw, yK + ch / 2 + 5, { size: 14, align: 'center', color: RED(C), weight: 700 });
        }
        // byte ruler
        const yR = yK + ch + 26;
        const seg = (xa, xb, lab, col) => { V.line(g, xa + 1, yR, xb - 1, yR, { color: col, lw: 2 }); V.line(g, xa + 1, yR - 4, xa + 1, yR + 4, { color: col, lw: 1.5 }); V.line(g, xb - 1, yR - 4, xb - 1, yR + 4, { color: col, lw: 1.5 }); V.text(g, lab, (xa + xb) / 2, yR + 16, { size: 11, align: 'center', font: 'mono', color: col }); };
        seg(x0, x0 + NB * cw, '512 B (FP8)', C.f);
        seg(xScale0, xScale0 + 4 * scW, narrow ? '16' : tf ? '4 + 12 B' : '16 B', C.d);
        if (!nat && !tf) seg(xRope0, xRope0 + 2 * cw, narrow ? '128' : '128 B', C.d);
        // V comes from NoPE
        V.text(g, tf ? '↑ K and V = the same 512 codes' : '↑ V = this region only (d_v = 512)', x0 + 4, yR + 34, { size: 11, color: C.muted, font: 'mono' });
        // accumulator
        const yA = yR + (narrow ? 62 : 58);
        V.text(g, narrow ? 'q·k score, block by block (cell = 32 dims)' : 'q·k score accumulator, block by block (each cell = 32 dims)', x0, yA, { size: 12, weight: 600, color: C.ink });
        const contrib = qv.map((x, i) => x * kv[i]).concat(nat || tf ? [] : [qp[0] * kp[0], qp[1] * kp[1]]);
        const nShow = Math.min(b + 1, contrib.length);
        const running = contrib.slice(0, nShow).reduce((s, x) => s + x, 0);
        const barY = yA + 24, barH = narrow ? 50 : 56;
        const mxC = 0.7;
        contrib.forEach((v, i) => {
          const x = i < NB ? x0 + i * cw : xRope0 + (i - NB) * cw;
          const hh = (Math.min(Math.abs(v), mxC) / mxC) * (barH / 2);
          const col = i >= NB ? (Math.abs(v) < 1e-12 ? C.muted : RED(C)) : v >= 0 ? C.a : C.b;
          const on = i < nShow;
          V.line(g, x + 2, barY + barH / 2, x + cw - 2, barY + barH / 2, { color: V.alpha(C.muted, 0.3), lw: 1 });
          if (Math.abs(v) < 1e-12 && i >= NB) { if (on) V.text(g, '+0', x + cw / 2, barY + barH / 2 - 4, { size: 11, align: 'center', font: 'mono', color: C.muted }); return; }
          V.box(g, x + 3, v >= 0 ? barY + barH / 2 - hh : barY + barH / 2, cw - 6, Math.max(2, hh), { r: 2, fill: V.alpha(col, on ? 0.95 : 0.28) });
        });
        if (b < contrib.length) {
          const xc = (b < NB ? x0 + b * cw : xRope0 + (b - NB) * cw) + cw / 2;
          V.line(g, xc, yK + ch + 2, xc, barY - 2, { color: V.alpha(C.c, 0.7), lw: 1.2, dash: [3, 3] });
          V.text(g, 'q×k', xc + 4, barY - 4, { size: 11, font: 'mono', color: C.c });
        } else if (tf && b < NB + 3) {
          const xc = xScale0 + scW / 2;
          V.line(g, xc, yK + ch * 0.75 + 2, xc, barY - 2, { color: V.alpha(C.c, 0.7), lw: 1.2, dash: [3, 3] });
          V.text(g, '× 2^k', xc - 4, barY - 4, { size: 11, font: 'mono', align: 'right', color: C.c });
        }
        const yRes = barY + barH + 24;
        if (nat) {
          textBlock(g, 'The stock image dies here: pe_dim must be 64 for fp8_ds_mla. The SM120 kernel only accepts the 576-wide query with kv_lora_rank = 512 and qk_rope_head_dim = 64.', x0, yRes, w - 2 * pad, { size: 12.5, weight: 600, color: RED(C) });
        } else {
          const done = nShow >= contrib.length;
          const total = contrib.reduce((s, x) => s + x, 0);
          const ok = Math.abs(total - exact) < 1e-9;
          V.text(g, `score so far: ${sf(running, 3)}`, x0, yRes, { size: 12.5, font: 'mono', color: C.ink });
          const msg = done ? (ok ? `✓ kernel ${sf(total, 3)} = exact ${sf(exact, 3)}` : `✕ kernel ${sf(total, 3)} ≠ exact ${sf(exact, 3)}: wrong softmax`) : `exact score (NoPE only): ${sf(exact, 3)}`;
          let yy = yRes;
          if (narrow) yy += 20; 
          V.text(g, msg, narrow ? x0 : x0 + 230, yy, { size: 12.5, font: 'mono', weight: 700, color: done ? (ok ? C.e : RED(C)) : C.muted });
          yy += 22;
          const note = tf ? (b >= NB && b < NB + 3 ? 'Score × 2^k; after the softmax, probabilities × 2^k before summing V. Nothing rounds.' : 'The e4m3 codes enter the math as they are; the row\'s scale comes at the end.') : S.qz ? (S.kz ? 'Zeros on both sides: exact, and the cache stays deterministic.' : 'The zeroed query is enough: 0 × z = 0 (as long as z is not NaN or Inf).') : S.kz ? 'The zeroed k_pe is enough: q × 0 = 0.' : 'Garbage × garbage ≠ 0: the score changes, and so does the softmax.';
          yy += textBlock(g, note, x0, yy, w - 2 * pad, { size: 11.5, color: C.muted }) + 2;
          textBlock(g, tf ? 'Cost: 528 B per token and layer (−19.5% vs 656), only for this family and at TP=2.' : 'Cost: 656 B per token and layer instead of ~528 (+24%) in the 11 DSA layers.', x0, yy, w - 2 * pad, { size: 11.5, color: C.d });
        }
      });
    })();

    /* ================================================ Fig 7 — inverse RoPE */
    (function fig7() {
      const fig = root.querySelector('#fig-rope-inversa');
      if (!fig) return;
      sizeStage(fig, 420, 380);
      const NT = 6, th = 0.42, P0 = 20;
      const A = [0.06, 0.09, 0.13, 0.19, 0.24, 0.29];
      const NOISE = [0.3, -0.25, 0.35, -0.3, 0.2, -0.1];
      const U = NOISE.map((nz, m) => [[0.95, 0.8, 1, 0.9, 1, 0.85][m], 0.9 - th * (m - (NT - 1)) + nz]); // [magnitude, phase]
      // maximum reach of the chain (rotation-invariant): used to scale the big clocks
      const EXT = (() => { let px = 0, py = 0, mx = 0; U.forEach(([mag, ph], m) => { const a = ph + th * (m - (NT - 1)); px += A[m] * mag * Math.cos(a); py += A[m] * mag * Math.sin(a); mx = Math.max(mx, Math.hypot(px, py)); }); return mx; })();
      const S = { delta: 0, auto: true, lastD: -1 };
      const ctl = UI.controls(fig);
      let tg = null;
      const sD = UI.slider(ctl, { label: 'Shift (Δ)', min: 0, max: 60, step: 1, value: 0, fmt: (v) => 'Δ = ' + v, onInput: (v) => { S.delta = v; } });
      sD.input.addEventListener('input', () => { if (tg) tg.set(false); });
      tg = UI.toggle(ctl, { label: 'auto-shift', value: true, onChange: (v) => { S.auto = v; } });
      const rRaw = UI.readout(ctl, 'Angle of o_i', '—');
      const rDer = UI.readout(ctl, 'After R(−i)', '—');
      const angDeg = (a) => ((Math.round((a * 180) / Math.PI) % 360) + 360) % 360;
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const narrow = w < 600;
        if (S.auto) {
          const d = Math.floor(t * 2.2) % 61;
          if (d !== S.delta) { S.delta = d; sD.input.value = d; const vEl = sD.el.querySelector('.val'); if (vEl) vEl.textContent = 'Δ = ' + d; }
        }
        const pos = (m) => P0 + S.delta + m; // m = 0..5 ; query at m = 5
        const pi = pos(NT - 1);
        const vecs = U.map(([mag, ph], m) => { const a = ph + th * pos(m); return [mag * Math.cos(a), mag * Math.sin(a)]; });
        const rel = U.map(([mag, ph], m) => { const a = ph + th * (pos(m) - pi); return [mag * Math.cos(a), mag * Math.sin(a)]; });
        const sum = (vs) => vs.reduce((s, v, m) => [s[0] + A[m] * v[0], s[1] + A[m] * v[1]], [0, 0]);
        const o = sum(vecs), od = sum(rel);
        const aRaw = Math.atan2(o[1], o[0]), aDer = Math.atan2(od[1], od[0]);
        if (S.lastD !== S.delta) { rRaw.set(angDeg(aRaw) + '°'); rDer.set(angDeg(aDer) + '°'); S.lastD = S.delta; }
        // small clocks
        const pad = 14;
        const slotW = (w - 2 * pad) / NT;
        const rs = Math.max(16, Math.min(30, slotW * 0.36));
        const yc = 30 + rs;
        V.text(g, narrow ? 'v_j rotated by absolute position j' : 'values v_j (RoPE part of 1 pair of dims), rotated by absolute position j', pad, 16, { size: 11.5, color: C.muted, maxW: w - 2 * pad });
        for (let m = 0; m < NT; m++) {
          const cx = pad + (m + 0.5) * slotW;
          const col = C.series[m % C.series.length];
          g.beginPath(); g.arc(cx, yc, rs, 0, Math.PI * 2); g.strokeStyle = V.alpha(C.muted, 0.4); g.lineWidth = 1; g.stroke();
          const v = vecs[m];
          V.arrow(g, cx, yc, cx + v[0] * rs * 0.85, yc - v[1] * rs * 0.85, { color: col, lw: 2, head: 6 });
          const lab = m === NT - 1 ? 'j = i' : `i−${NT - 1 - m}`;
          V.text(g, lab, cx, yc + rs + 14, { size: 11, align: 'center', font: 'mono', color: m === NT - 1 ? C.c : C.muted });
          if (!narrow) V.text(g, 'pos ' + pos(m), cx, yc + rs + 28, { size: 11, align: 'center', font: 'mono', color: C.muted });
          // weight
          const bh = A[m] * (narrow ? 70 : 90);
          const by = yc + rs + (narrow ? 22 : 36);
          V.box(g, cx - 8, by, 16, bh, { r: 3, fill: V.alpha(col, 0.75) });
          V.text(g, sf(A[m], 2), cx, by + bh + 13, { size: 11, align: 'center', font: 'mono', color: C.muted });
        }
        const yBars = yc + rs + (narrow ? 22 : 36);
        if (!narrow) V.text(g, 'a_ij', pad, yBars + 10, { size: 11, color: C.muted, font: 'mono' });
        // big clocks
        const R = Math.max(50, Math.min(92, (h - yBars - 50 - 60) / 2, (w - 3 * pad) / 4.4));
        const cyB = Math.min(h - R - 34, yBars + (narrow ? 88 : 72) + R);
        const cxL = w * 0.27, cxR = w * 0.73;
        const big = (cx, vs, total, title, sub, col, trail) => {
          g.beginPath(); g.arc(cx, cyB, R, 0, Math.PI * 2); g.strokeStyle = V.alpha(C.muted, 0.45); g.lineWidth = 1.2; g.stroke();
          for (let k = 0; k < 12; k++) { const a = (k / 12) * Math.PI * 2; V.line(g, cx + Math.cos(a) * R * 0.92, cyB + Math.sin(a) * R * 0.92, cx + Math.cos(a) * R, cyB + Math.sin(a) * R, { color: V.alpha(C.muted, 0.4), lw: 1 }); }
          // head-to-tail chain
          const sc = (R * 0.86) / EXT;
          let px = cx, py = cyB;
          vs.forEach((v, m) => { const nx = px + A[m] * v[0] * sc, ny = py - A[m] * v[1] * sc; V.line(g, px, py, nx, ny, { color: V.alpha(C.series[m % 6], 0.85), lw: 2 }); px = nx; py = ny; });
          if (trail) trail.forEach((a, k) => { const L = Math.hypot(total[0], total[1]) * sc; V.dot(g, cx + Math.cos(a) * L, cyB - Math.sin(a) * L, 2.5, V.alpha(col, 0.15 + 0.6 * (k / trail.length))); });
          V.glow(g, px, py, 14, col, 0.7);
          V.arrow(g, cx, cyB, px, py, { color: col, lw: 3, head: 9 });
          V.text(g, title, cx, cyB - R - 22, { size: 12.5, weight: 700, align: 'center', color: col, maxW: w * 0.46 });
          V.text(g, sub, cx, cyB - R - 8, { size: 11, align: 'center', color: C.muted, font: 'mono', maxW: w * 0.46 });
        };
        S.trail = S.trail || [];
        if (!S.trail.length || Math.abs(S.trail[S.trail.length - 1] - aRaw) > 1e-6) { S.trail.push(aRaw); if (S.trail.length > 18) S.trail.shift(); }
        big(cxL, vecs, o, 'raw output o_i', narrow ? 'absolute position' : 'depends on absolute position', C.d, S.trail);
        big(cxR, rel, od, 'R(−i) · o_i', narrow ? 'only j − i' : 'only distances j − i', C.e, null);
        V.text(g, `query position i = ${pi}`, w / 2, h - 10, { size: 11.5, align: 'center', font: 'mono', color: C.muted });
      });
    })();

    /* ============================================ Fig 8 — grouped output */
    (function fig8() {
      const fig = root.querySelector('#fig-saida-agrupada');
      if (!fig) return;
      sizeStage(fig, 450, 410);
      const MODES = {
        tp1: { tp: 1, G: 8, hpg: 8 },
        tp2: { tp: 2, G: 8, hpg: 8 },
        tp3: { tp: 3, G: 9, hpg: 8, pad: true },
        tp3x: { tp: 3, G: 8, hpg: 9, wrong: true },
      };
      const S = { mode: 'tp2', direct: true };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Parallelism', options: [{ v: 'tp1', label: 'TP=1' }, { v: 'tp2', label: 'TP=2' }, { v: 'tp3', label: 'TP=3 · group pad' }, { v: 'tp3x', label: 'TP=3 · wrong' }], value: 'tp2', onChange: (v) => { S.mode = v; } });
      UI.toggle(ctl, { label: 'compare with direct projection', value: true, onChange: (v) => { S.direct = v; } });
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const M = MODES[S.mode];
        const narrow = w < 600;
        const pad = 14;
        const rankCol = (r) => [C.a, C.b, C.c][r % 3];
        const G = M.G, gap = narrow ? 4 : 8;
        const gw = (w - 2 * pad - (G - 1) * gap) / G;
        const cols = M.hpg === 9 ? 3 : 4, rowsT = M.hpg === 9 ? 3 : 2;
        const ts = Math.max(4, Math.min(12, (gw - 8) / cols - 2));
        const gh = rowsT * (ts + 2) + 10;
        const heads = G * M.hpg;
        const yG = 44;
        const yA = yG + gh + 40, ah = 30;
        const yO = yA + ah + 10;
        const yNote = yO + 28;
        const yC = yNote + 14;
        const yB = yC + 38, bh = 34;
        const yY = yB + bh + 30;
        const yF = yY + 28;
        V.text(g, `${heads} heads × 512 = ${nf(heads * 512)} output dims · ${G} groups of ${M.hpg}`, pad, 18, { size: 12, weight: 600, color: C.ink });
        if (M.tp > 1) {
          const leg = [...Array(M.tp).keys()].map((r) => 'rank ' + r);
          let lx = w - pad;
          for (let r = M.tp - 1; r >= 0; r--) { const tx = leg[r]; const ww = tw(g, tx, 11, 'mono') + 16; lx -= ww; V.dot(g, lx + 5, 32, 4, rankCol(r)); V.text(g, tx, lx + 12, 36, { size: 11, font: 'mono', color: C.muted }); }
        }
        const groupRank = (gi) => Math.floor((gi * M.tp) / G);
        for (let gi = 0; gi < G; gi++) {
          const x = pad + gi * (gw + gap);
          const rk = groupRank(gi);
          const isPad = M.pad && gi === G - 1;
          const col = rankCol(rk);
          g.setLineDash(isPad ? [4, 3] : []);
          V.box(g, x, yG, gw, gh, { r: 6, fill: V.alpha(M.wrong ? C.muted : col, isPad ? 0.04 : 0.1), stroke: V.alpha(M.wrong ? C.muted : col, isPad ? 0.8 : 0.5), lw: 1 });
          g.setLineDash([]);
          for (let k = 0; k < M.hpg; k++) {
            const cx = x + 4 + (k % cols) * ((gw - 8) / cols) + ((gw - 8) / cols - ts) / 2;
            const cy = yG + 5 + Math.floor(k / cols) * (ts + 2);
            const tc = M.wrong ? rankCol(k % 3) : col;
            V.box(g, cx, cy, ts, ts, { r: 2, fill: isPad ? V.alpha(C.muted, 0.2) : V.alpha(tc, M.wrong && k === 8 ? 0.4 : 0.88) });
          }
          if (gw > 22) V.text(g, isPad ? 'pad' : 'g' + gi, x + gw / 2, yG + gh + 13, { size: 11, align: 'center', font: 'mono', color: isPad ? C.d : C.muted });
          const acol = M.wrong ? RED(C) : isPad ? C.d : col;
          flow(g, [x + gw / 2, yG + gh + 18], [x + gw / 2, yA - 2], { color: acol, t: t + gi * 0.1, active: !isPad, lw: 1.5, head: 5, noDots: gw < 20 });
          V.box(g, x, yA, gw, ah, { r: 6, fill: V.alpha(acol, 0.18), stroke: acol, lw: 1.3 });
          const lab = gw > 96 ? 'wo_a 4096→1024' : gw > 40 ? 'wo_a' : 'a';
          V.text(g, lab, x + gw / 2, yA + ah / 2 + 4, { size: 11, align: 'center', font: 'mono', weight: 600, color: C.ink });
          V.box(g, x + gw * 0.2, yO, gw * 0.6, 8, { r: 3, fill: V.alpha(isPad ? C.muted : acol, isPad ? 0.25 : 0.75) });
          V.line(g, x + gw / 2, yO + 8, x + gw / 2, yC, { color: V.alpha(acol, 0.35), lw: 1, dash: [2, 3] });
        }
        // note about wo_a
        if (M.wrong) textBlock(g, narrow ? 'local r = 3 × 512 = 1,536 ≠ 4,096: the einsum refuses' : 'local r = 3 × 512 = 1,536 ≠ 4,096: the FP8 einsum of wo_a refuses (each group would be chopped across 3 ranks)', pad, yNote, w - 2 * pad, { size: 11.5, weight: 700, color: RED(C), maxLines: 1 });
        else V.text(g, narrow ? 'wo_a: 4096 → 1024 per group (r = 8 × 512)' : 'wo_a[g]: 4096 → 1024 (o_lora_rank), one BMM with r = 8 × 512 = 4096 per group', pad, yNote, { size: 11, color: C.muted, font: 'mono' });
        // concat
        V.box(g, pad, yC, w - 2 * pad, 10, { r: 4, fill: V.alpha(C.ink, 0.12), stroke: V.alpha(C.ink, 0.3), lw: 1 });
        const ctxt = `concat: ${G} × 1024 = ${nf(G * 1024)}${M.pad ? ' (the pad only adds zeros)' : ''}`;
        const ctw = tw(g, ctxt, 11, 'mono') + 16;
        V.box(g, w / 2 - ctw / 2, yC + 12, ctw, 18, { r: 6, fill: C.bg || V.alpha(C.ink, 0.02) });
        V.text(g, ctxt, w / 2, yC + 25, { size: 11, align: 'center', font: 'mono', color: C.muted });
        // wo_b per rank
        if (M.wrong) {
          V.box(g, pad, yB, w - 2 * pad, bh, { r: 8, fill: V.alpha(C.muted, 0.08), stroke: V.alpha(RED(C), 0.6), lw: 1.2 });
          g.setLineDash([5, 4]); V.line(g, pad + 6, yB + bh / 2, w - pad - 6, yB + bh / 2, { color: V.alpha(RED(C), 0.35), lw: 1 }); g.setLineDash([]);
          V.text(g, 'wo_b never receives anything: boot fails first', w / 2, yB + bh / 2 + 4, { size: 11.5, align: 'center', font: 'mono', weight: 600, color: RED(C) });
        }
        for (let rk = 0; rk < (M.wrong ? 0 : M.tp); rk++) {
          const gs = [...Array(G).keys()].filter((gi) => groupRank(gi) === rk);
          const xa = pad + gs[0] * (gw + gap), xb = pad + gs[gs.length - 1] * (gw + gap) + gw;
          const col = M.tp === 1 ? C.b : rankCol(rk);
          flow(g, [(xa + xb) / 2, yC + 30], [(xa + xb) / 2, yB - 2], { color: col, t: t + rk * 0.2, lw: 1.4, head: 5 });
          V.box(g, xa, yB, xb - xa, bh, { r: 8, fill: V.alpha(col, 0.16), stroke: col, lw: 1.4 });
          const cand = M.tp === 1 ? ['wo_b: 8192 → 4096', 'wo_b'] : [`wo_b · rank ${rk} slice`, `wo_b · r${rk}`, 'wo_b'];
          const lb = cand.find((s2) => tw(g, s2, 11.5, 'mono', 600) < xb - xa - 8) || 'b';
          V.text(g, lb, (xa + xb) / 2, yB + bh / 2 + 4, { size: 11.5, align: 'center', font: 'mono', weight: 600, color: C.ink });
          flow(g, [(xa + xb) / 2, yB + bh], [w / 2, yY - 14], { color: col, t: t + rk * 0.3, lw: 1.5, head: 6 });
        }
        const ar = M.tp > 1 && !M.wrong;
        if (ar) { V.glow(g, w / 2, yY - 4, 18, C.d, 0.6); V.dot(g, w / 2, yY - 4, 10, C.d); V.text(g, '+', w / 2, yY + 1, { size: 15, align: 'center', weight: 700, color: onInk(C) }); }
        if (!M.wrong) V.text(g, ar ? 'all-reduce → y [4096]' : 'y [4096]', w / 2 + 16, yY + 1, { size: 12, weight: 600, color: C.ink });
        const local = M.wrong ? 'each rank: 8 groups × 3 heads (sliced)' : M.tp > 1 ? `each rank: ${G / M.tp} groups · ${(G / M.tp) * M.hpg} heads` : 'one rank: 8 groups · 64 heads';
        V.text(g, local, pad, yF, { size: 11.5, color: M.wrong ? RED(C) : C.muted, font: 'mono' });
        if (M.pad) textBlock(g, 'Pad group: wo_a = wo_b = 0 and attn_sink = −∞ (e^−∞ = 0 in the softmax denominator).', pad, yF + 18, w - 2 * pad, { size: 11.5, color: C.d, maxLines: 2 });
        if (S.direct) {
          const yD = yF + (M.pad ? (narrow ? 50 : 34) : 14), bw = Math.max(80, w - 2 * pad - (narrow ? 150 : 260));
          V.box(g, pad, yD, bw, 10, { r: 4, fill: V.alpha(C.muted, 0.5) });
          V.text(g, narrow ? 'direct: 134.2 M' : 'direct 32768 → 4096: 134.2 M parameters', pad + bw + 8, yD + 9, { size: 11, font: 'mono', color: C.muted });
          V.box(g, pad, yD + 16, bw / 2, 10, { r: 4, fill: V.alpha(C.b, 0.85) });
          V.text(g, narrow ? 'grouped: 67.1 M' : 'grouped (checkpoint): 33.55 M + 33.55 M = 67.1 M', pad + bw / 2 + 8, yD + 25, { size: 11, font: 'mono', color: C.ink });
        }
      });
    })();

    /* ============================================= Fig 9 — calculator */
    (function fig9() {
      const fig = root.querySelector('#fig-calculadora');
      if (!fig) return;
      sizeStage(fig, 410, 430);
      const PRE = {
        qwen: { L: 13, H: 24, G: 2, dh: 256, dc: 512, dr: 64, b: 2, used: 'gqa', note: 'Qwen3.8: 12 QSA + 1 MTP in BF16. With FP8: 13,312 B; measured at boot with FP8: 9,418 B/token/GPU (indexer, GDN, pages).' },
        glm: { L: 11, H: 64, G: 64, dh: 256, dc: 512, dr: 64, b: 1, used: 'mla', note: 'GLM-5.3: 11 DSA layers, RoPE = 64 zeros (rope_pad). + indexer 11 × 33 B = 7,579 B/token per rank (our math from the layout).' },
        tf: { L: 11, H: 64, G: 64, dh: 256, dc: 512, dr: 0, b: 1, used: 'mla', note: 'GLM·TF: + indexer 11 × 36 B = 6,204 B/token per rank (our math). No RoPE block (d_r = 0); 16 B = FP32 scale + 12 B of alignment.' },
        ds: { L: 43, H: 64, G: 1, dh: 512, dc: 512, dr: 64, b: 1, used: 'kv', note: 'DeepSeek-V4: 43 layers, 584 B row. With C4A/C128A compression and the indexer: ≈ 3,850 B/token (next chapter).' },
      };
      const S = { pre: 'glm', L: 11, H: 64, G: 64, dh: 256, dc: 512, dr: 64, b: 1, ctx: 20, tp2: false, used: 'mla' };
      let applying = false;
      const ctl = UI.controls(fig);
      const sl = {};
      const segP = UI.seg(ctl, { label: 'Preset', options: [{ v: 'qwen', label: 'Qwen3.8' }, { v: 'glm', label: 'GLM-5.3' }, { v: 'tf', label: 'GLM·TF' }, { v: 'ds', label: 'DeepSeek-V4' }, { v: 'livre', label: 'custom' }], value: 'glm', onChange: (v) => {
        if (v === 'livre') { S.pre = 'livre'; S.used = null; return; }
        const p = PRE[v]; applying = true;
        sl.L.set(p.L); sl.H.set(p.H); sl.G.set(Math.min(p.G, p.H)); sl.dh.set(p.dh); sl.dc.set(p.dc); sl.dr.set(p.dr); segB.set(p.b);
        applying = false; S.pre = v; S.used = p.used;
      } });
      const touch = () => { if (!applying && S.pre !== 'livre') { S.pre = 'livre'; S.used = null; segP.set('livre'); } };
      sl.L = UI.slider(ctl, { label: 'Layers with KV', min: 1, max: 64, step: 1, value: 11, onInput: (v) => { S.L = v; touch(); } });
      sl.H = UI.slider(ctl, { label: 'Q heads (H)', min: 8, max: 128, step: 4, value: 64, onInput: (v) => { S.H = v; if (S.G > v) sl.G.set(v); touch(); } });
      sl.G = UI.slider(ctl, { label: 'KV heads (G)', min: 1, max: 128, step: 1, value: 64, onInput: (v) => { S.G = Math.min(v, S.H); touch(); } });
      sl.dh = UI.slider(ctl, { label: 'd_h', min: 64, max: 512, step: 64, value: 256, onInput: (v) => { S.dh = v; touch(); } });
      sl.dc = UI.slider(ctl, { label: 'd_c (latent)', min: 128, max: 1024, step: 64, value: 512, onInput: (v) => { S.dc = v; touch(); } });
      sl.dr = UI.slider(ctl, { label: 'd_r (RoPE)', min: 0, max: 128, step: 32, value: 64, onInput: (v) => { S.dr = v; touch(); } });
      const segB = UI.seg(ctl, { label: 'KV', options: [{ v: 2, label: 'BF16' }, { v: 1, label: 'FP8' }], value: 1, onChange: (v) => { S.b = v; touch(); } });
      UI.slider(ctl, { label: 'Context', min: 12, max: 20, step: 1, value: 20, fmt: (v) => kfmt(2 ** v) + ' tokens', onInput: (v) => { S.ctx = v; } });
      UI.toggle(ctl, { label: 'per rank at TP=2', value: false, onChange: (v) => { S.tp2 = v; } });
      function perLayer() {
        const { H, G, dh, dc, dr, b } = S;
        const G2 = Math.min(G, H);
        const splitG = S.tp2 && G2 >= 2 && G2 % 2 === 0;
        const mha = 2 * H * dh * b / (S.tp2 ? 2 : 1);
        const gqa = 2 * G2 * dh * b / (splitG ? 2 : 1);
        const mqa = 2 * dh * b;
        const mla = dc * b + dr * 2 + (b === 1 ? 4 * Math.ceil(dc / 128) : 0);
        const nope = Math.max(0, dh - dr);
        const kv = b === 1 ? nope + dr * 2 + Math.ceil(nope / 64) + 1 : dh * 2;
        return [
          { k: 'mha', label: 'MHA', v: mha, col: 'a', rep: false },
          { k: 'gqa', label: `GQA (G=${G2})`, v: gqa, col: 'e', rep: S.tp2 && !splitG },
          { k: 'mqa', label: 'MQA', v: mqa, col: 'b', rep: S.tp2 },
          { k: 'mla', label: 'MLA (latent + RoPE)', v: mla, col: 'f', rep: S.tp2 },
          { k: 'kv', label: 'MLA K=V (V4)', v: kv, col: 'd', rep: S.tp2 },
        ];
      }
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const narrow = w < 600;
        const pad = 14;
        const rows = perLayer();
        const T = 2 ** S.ctx;
        V.text(g, `${S.L} layers · H = ${S.H} · d_h = ${S.dh} · ${S.b === 2 ? 'BF16' : 'FP8'}${S.tp2 ? ' · per rank (TP=2)' : ''}`, pad, 18, { size: 12, weight: 600, color: C.ink, maxW: w - 2 * pad });
        const labW = narrow ? 0 : 158;
        const x0 = pad + labW, x1 = w - pad - (narrow ? 0 : 140);
        const lo = Math.log10(100), hi = Math.log10(5e6);
        const X = (b) => x0 + ((Math.log10(Math.max(100, b)) - lo) / (hi - lo)) * (x1 - x0);
        const top = 40, rh = narrow ? 58 : 54;
        [100, 1e3, 1e4, 1e5, 1e6].forEach((b) => {
          const x = X(b);
          V.line(g, x, top - 4, x, top + rows.length * rh - 10, { color: V.alpha(C.muted, 0.18), lw: 1, dash: [2, 4] });
          const gl = b >= 1e6 ? '1 MB' : b >= 1e3 ? nf(b / 1e3) + ' KB' : '100 B';
          V.text(g, gl, b === 100 ? x : x, top + rows.length * rh + (narrow ? 10 : 2), { size: 11, align: b === 100 ? 'left' : 'center', font: 'mono', color: C.muted });
        });
        rows.forEach((r, i) => {
          const y = top + i * rh;
          const tot = r.v * S.L;
          const used = S.used === r.k;
          const col = C[r.col];
          const by = narrow ? y + 16 : y + 8, bh = narrow ? 16 : 22;
          V.text(g, r.label + (used ? '  ← used' : ''), pad, narrow ? y + 11 : y + 23, { size: 12, weight: used ? 700 : 500, color: used ? C.ink : C.muted, maxW: narrow ? w - 2 * pad : labW - 8 });
          const xe = X(tot);
          if (used) V.glow(g, xe, by + bh / 2, 22, col, 0.55 + 0.2 * Math.sin(t * 3));
          V.box(g, x0, by, Math.max(4, xe - x0), bh, { r: 5, fill: V.alpha(col, used ? 0.95 : 0.5), stroke: used ? C.ink : null, lw: 1.5 });
          const v1 = fmtTok(tot) + '/token', v2 = fmtGiB(tot * T) + ' at ' + kfmt(T) + (r.rep ? ' · replicated' : '');
          if (narrow) V.text(g, v1 + ' · ' + v2, pad, y + 46, { size: 11, font: 'mono', color: used ? C.ink : C.muted });
          else {
            V.text(g, v1, x1 + 10, y + 17, { size: 11.5, font: 'mono', weight: used ? 700 : 400, color: used ? C.ink : C.muted });
            V.text(g, v2, x1 + 10, y + 32, { size: 11, font: 'mono', color: C.muted, maxW: w - x1 - 14 });
          }
        });
        const yN = top + rows.length * rh + (narrow ? 34 : 24);
        const perL = rows.find((r) => r.k === S.used);
        const note = S.used ? `${nf(perL.v)} B per layer × ${S.L} layers = ${nf(perL.v * S.L)} B/token. ` + PRE[S.pre].note : `at ${kfmt(T)} tokens of context; move the controls or pick a preset`;
        textBlock(g, note, pad, yN, w - 2 * pad, { size: 11.5, color: C.muted, maxLines: narrow ? 4 : 3 });
      });
    })();

    /* ============================================== Fig 10 — lock */
    (function fig10() {
      const fig = root.querySelector('#fig-fechadura');
      if (!fig) return;
      sizeStage(fig, 470, 590);
      const S = { m: 'glm0', tp: 2, t0: -2.9 };
      let H = null;
      const tfc = (C) => C.__tf || (C.__tf = (getComputedStyle(document.documentElement).getPropertyValue('--tf') || '').trim() || C.d);
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Key', options: [{ v: 'glm0', label: 'GLM-5.3 · stock image' }, { v: 'glm', label: 'GLM-5.3 · recipe' }, { v: 'ds', label: 'DeepSeek-V4 · recipe' }, { v: 'tf', label: 'GLM-5.3 · TensorFold' }], value: 'glm0', onChange: (v) => { S.m = v; S.t0 = H ? H.t : 0; } });
      UI.seg(ctl, { label: 'TP', options: [2, 3], value: 2, onChange: (v) => { S.tp = v; S.t0 = H ? H.t : 0; } });
      function rowsFor() {
        const tp3 = S.tp === 3;
        if (S.m === 'glm0') return [
          { a: 'key dimension', k: '576 (512 + 64 RoPE)', v: '512 (NoPE)', ok: false, why: 'pe_dim must be 64 for fp8_ds_mla' },
          { a: 'cache row', k: 'fp8_ds_mla 656 B', v: 'without the 128 B of RoPE', ok: false, why: 'the record has a fixed slot for 64 RoPE dims' },
          { a: 'top-k (template)', k: 'exactly 2,048 in prefill', v: '512 pools × 4 + 3 = 2,051 → buffer 2,176', ok: false, why: 'expects (256, 1, 2048), got (256, 1, 2176)' },
          { a: 'local heads', k: '{8, 16, 32, 64, 128}', v: tp3 ? '66 / 3 = 22' : '64 / 2 = 32', ok: !tp3, why: tp3 ? '22 is not instantiated' : '' },
          { a: 'prefill', k: 'existing path', v: 'inherited forward_mha', ok: false, why: 'NotImplementedError' },
        ];
        if (S.m === 'glm') return [
          { a: 'key dimension', k: '576 (512 + 64 RoPE)', v: '512 + 64 zeros (rope_pad)', ok: true, why: 'exact: the zero block adds 0' },
          { a: 'cache row', k: 'fp8_ds_mla 656 B', v: '656 B with k_pe = 0', ok: true, why: '+24% KV, no new kernel' },
          { a: 'top-k (template)', k: 'exactly 2,048', v: '511 pools + tail ≤ 2,047', ok: true, why: 'loses 0.2%, the lowest-ranked' },
          { a: 'local heads', k: '{8, 16, 32, 64, 128}', v: tp3 ? '22 → Q padded to 32' : '32', ok: true, why: tp3 ? 'TP3 overlay: extra heads with zero Q' : '' },
          { a: 'prefill', k: 'existing path', v: 'supports_dense_mha_prefill = False', ok: true, why: 'prefill through the top-k MQA kernel' },
        ];
        if (S.m === 'tf') return [
          { a: 'key dimension', k: 'the model\'s kv_lora_rank', v: '512 (NoPE), no padding', ok: true, why: 'Triton kernel written for the family' },
          { a: 'cache row', k: 'kv8: codes + 16 B', v: '528 B (e4m3 + 2^k scale)', ok: true, why: '−19.5% bytes vs 656' },
          { a: 'top-k (selection)', k: '512 pools + the incomplete one', v: '512 × 4 + up to 3 = 2,051', ok: true, why: 'its own selection, no cut' },
          { a: 'ranks and heads', k: '2 ranks (v1.3.2) · 2 or 3 (v1.4+)', v: tp3 ? '64 → 22 / 21 / 21, no padding' : '64 / 2 = 32', ok: true, why: tp3 ? 'uneven split, patches 0066–0068 (v1.4)' : '' },
          { a: 'prefill', k: 'latent path', v: 'absorption and expansion on tensor cores', ok: true, why: 'fixed tiles: per-row bits' },
        ];
        return [
          { a: 'key dimension', k: 'DSV4: 512 (448 + 64)', v: '512 = 448 NoPE + 64 RoPE', ok: true, why: 'K = V, native geometry' },
          { a: 'cache row', k: '584 B (448 FP8 + 128 + 8)', v: 'nvfp4_ds_mla = same 584 B', ok: true, why: 'hotfix #22: dispatch to the FP8 kernel' },
          { a: 'top-k (template)', k: '{128, 512, 1024}', v: 'C4A: indexer top-512', ok: true, why: '+ 128-token window in the primary cache' },
          { a: 'local heads', k: 'h_q ∈ {16, 32, 64, 128}', v: tp3 ? '72 / 3 = 24 → 32' : '64 / 2 = 32', ok: true, why: tp3 ? 'pad heads with sink −∞' : '' },
          { a: 'dual cache', k: 'window + compressed', v: 'SWA 128 + extra C4A/C128A cache', ok: true, why: 'lane A: DeepseekV4FlashInferSM120Attention' },
        ];
      }
      H = V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const narrow = w < 600;
        const rows = rowsFor();
        const all = rows.every((r) => r.ok);
        const el = t - S.t0;
        const slide = V.ease.inOut(V.clamp(el / 1.1, 0, 1));
        const nChecked = Math.max(0, Math.min(rows.length, Math.floor((el - 1.1) / 0.35) + 1));
        const doneT = 1.1 + rows.length * 0.35;
        const pad = 14;
        // key/lock drawing
        const kh = 96, ky = 20;
        const lockW = Math.min(170, w * 0.36), lockX = w - pad - lockW;
        const keyLen = Math.min(w * 0.46, 330);
        const keyX = V.lerp(pad, lockX - keyLen + 34, slide);
        const cy = ky + kh / 2 + 6;
        const shake = el > doneT && !all ? Math.sin(el * 40) * 3 * Math.max(0, 1 - (el - doneT) / 0.8) : 0;
        const open = el > doneT && all ? V.ease.out(V.clamp((el - doneT) / 0.6, 0, 1)) : 0;
        // lock
        const lc = all && el > doneT ? C.e : C.muted;
        g.beginPath(); g.arc(lockX + lockW / 2, ky + 30 - open * 16, lockW * 0.26, Math.PI, 0); g.strokeStyle = V.alpha(lc, 0.9); g.lineWidth = 7; g.stroke();
        V.box(g, lockX, ky + 30, lockW, kh - 14, { r: 12, fill: V.alpha(lc, 0.14), stroke: V.alpha(lc, 0.8), lw: 2 });
        V.text(g, S.m === 'tf' ? 'TensorFold kernels' : 'SM120 kernel', lockX + lockW / 2, ky + 30 + (kh - 14) - 10, { size: 11, align: 'center', font: 'mono', color: C.muted });
        // pins
        rows.forEach((r, i) => {
          const px = lockX + 14 + i * ((lockW - 28) / (rows.length - 1));
          const checked = i < nChecked;
          const col = checked ? (r.ok ? C.e : RED(C)) : C.muted;
          V.box(g, px - 4, ky + 38, 8, 26, { r: 3, fill: V.alpha(col, checked ? 0.9 : 0.35) });
        });
        // key
        const kcol = S.m === 'ds' ? C.ds : S.m === 'tf' ? tfc(C) : C.glm;
        g.save(); g.translate(shake, 0);
        g.beginPath(); g.arc(keyX + 24, cy + 10, 22, 0, Math.PI * 2); g.fillStyle = V.alpha(kcol, 0.25); g.fill(); g.strokeStyle = kcol; g.lineWidth = 2.5; g.stroke();
        g.beginPath(); g.arc(keyX + 24, cy + 10, 8, 0, Math.PI * 2); g.fillStyle = C.bg || 'transparent'; g.fill(); g.stroke();
        V.box(g, keyX + 44, cy + 5, keyLen - 44, 10, { r: 3, fill: V.alpha(kcol, 0.6), stroke: kcol, lw: 1.5 });
        rows.forEach((r, i) => {
          const tx = keyX + 60 + i * ((keyLen - 76) / (rows.length - 1));
          const th = r.ok ? 14 : 6 + (i % 2) * 12;
          V.box(g, tx - 5, cy + 15, 10, th, { r: 2, fill: V.alpha(kcol, 0.85) });
        });
        g.restore();
        const label = S.m === 'glm0' ? 'GLM-5.3 stock' : S.m === 'glm' ? 'GLM-5.3 + patches' : S.m === 'tf' ? 'GLM-5.3 · TensorFold' : 'DeepSeek-V4';
        V.text(g, label, keyX + 4, ky + 8, { size: 12, weight: 700, color: kcol });
        if (el > doneT) {
          const msg = S.m === 'tf' ? (all ? '✓ opens: a lock made for this key' : '✕ does not open') : all ? '✓ opens: the model runs on this kernel' : '✕ does not open: dies on the first forward';
          V.text(g, msg, w / 2, ky + kh + 34, { size: 13, weight: 700, align: 'center', color: all ? C.e : RED(C), maxW: w - 2 * pad });
        }
        // table
        const y0 = ky + kh + 58;
        const rh = narrow ? 76 : 50;
        if (!narrow) {
          V.text(g, 'requirement', pad, y0 - 8, { size: 11, color: C.muted, font: 'mono' });
          V.text(g, 'kernel asks for', pad + w * 0.2, y0 - 8, { size: 11, color: C.muted, font: 'mono' });
          V.text(g, 'model delivers', pad + w * 0.47, y0 - 8, { size: 11, color: C.muted, font: 'mono' });
        }
        rows.forEach((r, i) => {
          const y = y0 + i * rh;
          const checked = i < nChecked;
          const col = checked ? (r.ok ? C.e : RED(C)) : C.muted;
          V.line(g, pad, y + 2, w - pad, y + 2, { color: V.alpha(C.muted, 0.18), lw: 1 });
          V.text(g, checked ? (r.ok ? '✓' : '✕') : '·', w - pad - 4, y + 22, { size: 16, weight: 700, align: 'right', color: col });
          if (!narrow) {
            V.text(g, r.a, pad, y + 20, { size: 12, weight: 600, color: C.ink, maxW: w * 0.19 });
            V.text(g, r.k, pad + w * 0.2, y + 20, { size: 11.5, font: 'mono', color: C.muted, maxW: w * 0.26 });
            V.text(g, r.v, pad + w * 0.47, y + 20, { size: 11.5, font: 'mono', color: checked ? C.ink : C.muted, maxW: w * 0.46 });
            if (r.why && checked) V.text(g, r.why, pad + w * 0.47, y + 38, { size: 11, color: col, maxW: w * 0.46 });
          } else {
            V.text(g, r.a, pad, y + 18, { size: 12, weight: 700, color: C.ink });
            V.text(g, 'asks: ' + r.k, pad, y + 35, { size: 11, font: 'mono', color: C.muted, maxW: w - 2 * pad - 20 });
            V.text(g, 'has: ' + r.v, pad, y + 51, { size: 11, font: 'mono', color: checked ? C.ink : C.muted, maxW: w - 2 * pad - 20 });
            if (r.why && checked) V.text(g, r.why, pad, y + 67, { size: 11, color: col, maxW: w - 2 * pad - 20 });
          }
        });
        if (el > doneT + 3.2) S.t0 = t;
      });
    })();
  },
});

LA.chapter({
  id: 'posicao-rope',
  part: 2,
  order: 8,
  title: 'Position: RoPE, NoPE, mRoPE and YaRN',
  navTitle: 'Position & RoPE',
  subtitle: 'How a Transformer knows where each token is: rotations, frequencies, partial RoPE, NoPE, 3D positions and context extension with YaRN, all the way to the JSON that has to reach the right sub-config.',
  minutes: 50,
  recipes: ['glm', 'qwen', 'ds'],
  tags: ['RoPE', 'rope_theta', 'partial RoPE', 'NoPE', 'mRoPE', 'YaRN', 'max_model_len', 'hf-overrides'],
  goals: [
    'Explain why pure attention is <b>blind to order</b> and how RoPE makes the score depend only on <b>distance</b>',
    'Read <code>rope_theta</code>, <code>partial_rotary_factor</code>, <code>is_neox_style</code> and <code>mrope_section</code> in a config and predict what each one does to the attention head',
    'Understand why GLM-5.3 works <b>without RoPE</b> (NoPE) and what that cost in the kernel',
    'Compute the effect of <b>YaRN</b> pair by pair and decide when and on which layers to use it, with consistent <code>max_model_len</code> and <code>VLLM_ALLOW_LONG_MAX_MODEL_LEN</code> settings',
  ],
  prereqs: ['atencao', 'tensores-algebra'],
  summary: [
    'Attention is <b>permutation equivariant</b>: without position, a sentence is a bag of tokens. The causal mask only leaks “how many came before”.',
    'RoPE rotates each pair of dimensions of q and k by \\(m\\theta_i\\). Since \\(R_m^\\top R_n = R_{n-m}\\), the score depends only on distance. The key is written to the KV cache already rotated, and nothing needs to be recomputed.',
    '\\(\\theta_i = b^{-2i/d_r}\\): fast pairs resolve neighbors and slow pairs separate long distances. <code>rope_theta</code> is 10⁴ in the SWA layers and in the drafters, 1.6·10⁵ in DeepSeek\'s compressor layers and 10⁷ in Qwen.',
    'Partial RoPE: Qwen rotates 64 of 256 dims (<code>0.25</code>, NoPE for the rest, NeoX style by inference); DeepSeek rotates the last 64 of 512 (interleaved, stored in BF16 in the 584 B row). GLM\'s MLA is NoPE (<code>qk_rope_head_dim 0</code>): order comes from the mask, KDA, the short conv and the kpool APE, and the kernel gets 64 zeros of padding (exact, +24% KV).',
    'mRoPE splits Qwen\'s 32 pairs among time, height and width (<code>[11,11,10]</code>, interleaved). For text, it becomes plain RoPE.',
    'YaRN keeps the pairs that made many turns during training, interpolates by \\(s\\) the ones that never made a single turn, and ramps in between (+ mscale). Qwen ×4 (262,144 → ~1M, optional and not validated; it was a no-op until the override moved into <code>text_config</code>). DeepSeek ×16 (64K → 1M) only on the compressor layers; the vllm#54815 fix is opt-in.',
    '<code>max_position_embeddings</code> belongs to the model and <code>max_model_len</code> is the server\'s cap (which also sizes buffers). Going past the config requires <code>VLLM_ALLOW_LONG_MAX_MODEL_LEN=1</code>, which only lifts the lock: what makes the positions valid is the RoPE scaling.',
  ],
  quiz: [
    { q: 'Why, with RoPE, does the score \\(\\tilde q_m\\cdot\\tilde k_n\\) depend only on \\(n-m\\)?', o: ['Because RoPE adds a position vector to the embedding', 'Because rotations are orthogonal and compose: \\(R_m^\\top R_n = R_{n-m}\\)', 'Because the softmax normalizes the positions', 'Because the value is rotated too'], a: 1, e: 'q is rotated by \\(m\\) and k by \\(n\\). The dot product contains \\(R_m^\\top R_n\\), and since \\(R_m^\\top = R_{-m}\\), that is \\(R_{n-m}\\). Adding a vector to the embedding is APE, which is absolute. The value is not rotated in standard attention.' },
    { q: 'In Qwen3.8, <code>head_dim = 256</code> and <code>partial_rotary_factor = 0.25</code>. How many frequency pairs does each head have, and how does that match <code>mrope_section</code>?', o: ['256 pairs, and mRoPE uses only 32 of them', '64 pairs; [11,11,10] adds up to 32 and half are left over', '32 pairs (64 dims); [11,11,10] adds up to exactly 32', '128 pairs, one per NoPE dimension'], a: 2, e: '0.25 × 256 = 64 rotated dimensions = 32 pairs. The T, H and W sections add up to 11 + 11 + 10 = 32: each pair belongs to one axis. The other 192 dims are NoPE and have no frequency at all.' },
    { q: 'Why does YaRN leave the high-frequency pairs intact?', o: ['Because they carry no useful information', 'Because they completed many turns during training, so every angle is already familiar; only the pairs that never made a full turn extrapolate', 'Because the kernel doesn\'t allow changing them', 'Because mscale compensates for them'], a: 1, e: 'A pair with \\(r_i = L/\\lambda_i &gt; 32\\) has seen the whole circle dozens of times: larger positions create no new angles for it. Interpolating it (as PI does) would only destroy the resolution between neighbors. The danger lies in the pairs with \\(\\lambda_i &gt; L\\).' },
    { q: 'In DeepSeek-V4, why don\'t the sliding window (SWA) layers need YaRN even when serving a 1M context?', o: ['Because they are NoPE', 'Because they use θ = 160000', 'Because RoPE is relative and the 128-token window limits distances to at most 127, always within training', 'Because the compressor applies YaRN on their behalf'], a: 2, e: 'With RoPE, the score depends on \\(n-m\\), and in a 128-token window that never exceeds 127, at any point of the context. There is no extrapolation to fix. That\'s why the reference uses pure RoPE (θ = 10000) on them, and the vllm#54815 fix gives these layers identity scaling (<code>factor=1.0</code>).' },
    { q: 'Before September 2026, what happened when the Qwen recipe turned on YaRN with <code>--hf-overrides \'{"rope_parameters":…}\'</code> at the root level?', o: ['vLLM refused to boot', 'The override landed in the parent config\'s <code>rope_parameters</code>; <code>text_config.rope_parameters</code> stayed <code>default</code>, and 1M positions ran with unscaled RoPE', 'YaRN was applied twice', 'The ViT got YaRN instead of the LLM'], a: 1, e: '<code>_apply_dict_overrides</code> only descends into keys that are nested configs. The parent of <code>qwen4_exp</code> exposes <code>rope_parameters</code> as a plain dictionary, so the <code>setattr</code> stopped there. The fix nests everything under <code>text_config</code>, and that is why 1M is now treated as not validated.' },
    { q: 'The SM12x sparse-MLA kernel requires 64 RoPE dims, but GLM-5.3 is NoPE. Why is the 64-zero pad exact?', o: ['Because the kernel ignores the extra dimensions', 'Because a block of zeros adds 0 to the product \\(q\\cdot k\\): the softmax and the output don\'t change, and the cost is ~24% more KV in the DSA layers', 'Because training used those zeros', 'It isn\'t exact: there is a small, accepted numerical error'], a: 1, e: 'The dot product is a sum of terms \\(q_i k_i\\), and the 64 new ones are \\(0\\times 0\\). The score is bit-identical, and V comes from the NoPE region. The price is memory: 656 B per token and layer instead of ~528 B.' },
  ],
  mount(root, LA) {
    const V = LA.viz, UI = LA.ui;
    const TAU = Math.PI * 2;
    const nf = (x) => Math.round(x).toLocaleString('en-US');
    const sf = (x, d = 2) => (+x).toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d });
    let fcache = {};
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(() => { fcache = {}; });
    const font = (s, f, w) => { const k = s + '|' + f + '|' + w; return fcache[k] || (fcache[k] = V.font(s, f, w)); };
    const tw = (g, s, size = 12, f = 'sans', w = 400) => { g.font = font(size, f, w); return g.measureText(String(s)).width; };
    function pill(g, x, y, s, col, o = {}) {
      const size = o.size || 11, f = o.font || 'mono', wt = o.weight || 600;
      const wd = tw(g, s, size, f, wt) + 14, ht = size + 9;
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
    const short = (x) => (x >= 1e6 ? sf(x / 1e6, x >= 1e7 ? 0 : 1) + 'M' : x >= 1e3 ? sf(x / 1e3, x >= 1e4 ? 0 : 1) + 'k' : sf(x, 1));
    const mfmt = (v, d) => { const r = sf(v, d); return /^-0\.0+$/.test(r) ? r.slice(1) : r.replace('-', '−'); };
    const fnv = (str) => { let h = 0x811c9dc5; for (let i = 0; i < str.length; i++) { h ^= str.charCodeAt(i); h = Math.imul(h, 0x01000193) >>> 0; } return h >>> 0; };
    const ropeInv = (base, d) => Array.from({ length: d / 2 }, (_, i) => Math.pow(base, (-2 * i) / d));
    /* YaRN "NTK-by-parts" with the same formula as the HF/vLLM implementations (ramp over the pair index) */
    function yarnInv(base, d, L, s, bf = 32, bs = 1) {
      const fcd = (r) => (d * Math.log(L / (r * TAU))) / (2 * Math.log(base));
      const lo = Math.max(Math.floor(fcd(bf)), 0);
      let hi = Math.min(Math.ceil(fcd(bs)), d - 1);
      if (lo === hi) hi += 0.001;
      const inv = [], kind = [];
      for (let i = 0; i < d / 2; i++) {
        const th = Math.pow(base, (-2 * i) / d);
        const ramp = V.clamp((i - lo) / (hi - lo), 0, 1); // 0 = keep, 1 = interpolate
        inv.push(s > 1 ? (th / s) * ramp + th * (1 - ramp) : th);
        kind.push(ramp <= 0 ? 'K' : ramp >= 1 ? 'I' : 'R');
      }
      return { inv, kind, lo, hi };
    }
    const mscaleOf = (s) => (s <= 1 ? 1 : 0.1 * Math.log(s) + 1);
    /* stage height by width: hd on desktop, hm below breakpoint bp */
    function fitH(fig, hd, hm, bp = 600) {
      const st = fig.querySelector('.la-fig-stage');
      if (!st) return;
      const apply = () => { const w = fig.clientWidth || 800; const want = (w < bp ? hm : hd) + 'px'; if (st.style.height !== want) st.style.height = want; };
      apply();
      window.addEventListener('resize', apply);
      if (LA.onCleanup) LA.onCleanup(() => window.removeEventListener('resize', apply));
    }

    /* =============================================================
       Fig 1 — shuffle: permutation equivariance
       ============================================================= */
    {
      const fig = root.querySelector('#fig-embaralhar');
      const words = ['the', 'cat', 'chases', 'the', 'mouse'];
      const D = 8, TR = 1;
      const emb = words.map((w) => { const r = V.rng(fnv('emb-' + w)); return Array.from({ length: D }, () => r.normal()); });
      const WQ = V.randMatrix(D, D, 11, 0.45), WK = V.randMatrix(D, D, 12, 0.45), WV = V.randMatrix(D, D, 13, 0.6);
      const mv = (M, x) => M.map((row) => row.reduce((s, a, k) => s + a * x[k], 0));
      const dot = (a, b) => a.reduce((s, x, k) => s + x * b[k], 0);
      const rot = (v, p) => {
        const o = v.slice();
        for (let i = 0; i < D / 2; i++) {
          const a = p * 0.9 * Math.pow(4, (-2 * i) / D), c = Math.cos(a), s = Math.sin(a);
          const x = v[2 * i], y = v[2 * i + 1];
          o[2 * i] = x * c - y * s; o[2 * i + 1] = x * s + y * c;
        }
        return o;
      };
      function comp(order, rope, causal) {
        const X = order.map((ti) => emb[ti]);
        let Q = X.map((x) => mv(WQ, x)), K = X.map((x) => mv(WK, x));
        const Vv = X.map((x) => mv(WV, x));
        if (rope) { Q = Q.map((q, p) => rot(q, p)); K = K.map((k, p) => rot(k, p)); }
        const att = [];
        for (let i = 0; i < order.length; i++) {
          const sc = K.map((k, j) => (causal && j > i ? -Infinity : dot(Q[i], k) / Math.sqrt(D)));
          const m = Math.max(...sc);
          const e = sc.map((v) => (v === -Infinity ? 0 : Math.exp(v - m)));
          const z = e.reduce((a, b) => a + b, 0);
          att.push(e.map((v, j) => (causal && j > i ? NaN : v / z)));
        }
        const out = att.map((row) => Array.from({ length: D }, (_, d) => row.reduce((s, a, j) => s + (Number.isNaN(a) ? 0 : a * Vv[j][d]), 0)));
        const byTok = [];
        order.forEach((ti, p) => { byTok[ti] = out[p]; });
        return { att, byTok };
      }
      const ident = [0, 1, 2, 3, 4];
      const S = { order: ident.slice(), prev: ident.slice(), t0: -10, rope: false, causal: false, auto: true, cache: null, mx: 1 };
      const diffTok = (ti) => { const a = S.cache.cur.byTok[ti], b = S.cache.ref.byTok[ti]; return Math.sqrt(a.reduce((s, x, k) => s + (x - b[k]) * (x - b[k]), 0)); };
      const ctl = UI.controls(fig);
      let H = null;
      function recompute() {
        S.cache = { cur: comp(S.order, S.rope, S.causal), ref: comp(ident, S.rope, S.causal) };
        let mx = 1e-6;
        [S.cache.cur, S.cache.ref].forEach((c) => c.byTok.forEach((v) => v.forEach((x) => { mx = Math.max(mx, Math.abs(x)); })));
        S.mx = mx;
        const d = diffTok(TR);
        rd.set(d < 1e-9 ? '0 (identical)' : sf(d, 3));
      }
      const rngS = V.rng(99);
      function shuffle() {
        const o = S.order.slice();
        let tries = 0;
        do { for (let i = o.length - 1; i > 0; i--) { const j = rngS.int(i + 1); const tmp = o[i]; o[i] = o[j]; o[j] = tmp; } tries++; } while (tries < 12 && o.every((v, i) => v === S.order[i]));
        S.prev = S.order; S.order = o; S.t0 = H ? H.t : 0; recompute();
      }
      UI.button(ctl, '🔀 shuffle', () => shuffle(), { primary: true });
      UI.button(ctl, '↺ original order', () => { S.prev = S.order; S.order = ident.slice(); S.t0 = H ? H.t : 0; recompute(); });
      UI.toggle(ctl, { label: 'RoPE (position)', value: false, onChange: (v) => { S.rope = v; recompute(); } });
      UI.toggle(ctl, { label: 'causal mask', value: false, onChange: (v) => { S.causal = v; recompute(); } });
      UI.toggle(ctl, { label: 'auto-shuffle', value: true, onChange: (v) => { S.auto = v; } });
      const rd = UI.readout(ctl, 'Δ in the output of “cat”', '—');
      recompute();
      V.every(3800, () => { if (S.auto) shuffle(); });
      H = V.fig(fig, (g, w, h, t) => {
        if (w < 60 || h < 60) return; // transient frame before layout
        const C = V.C();
        const narrow = w < 560;
        const pad = 14, slotW = (w - 2 * pad) / 5;
        const k = V.ease.inOut(V.clamp((t - S.t0) / 0.8, 0, 1));
        const sentence = S.order.map((ti) => words[ti]).join(' ');
        V.text(g, '“' + sentence + '”', pad, 22, { size: 13.5, weight: 600, color: C.ink, maxW: w - 130 });
        pill(g, w - pad, 20, S.rope ? 'with RoPE' : 'no position', S.rope ? C.a : C.muted, { align: 'right', size: 11 });
        const chipY = 42, chipH = 30, glyTop = chipY + chipH + 10, glyH = 52;
        for (let ti = 0; ti < 5; ti++) {
          const ps = S.prev.indexOf(ti), cs = S.order.indexOf(ti);
          const cx = pad + slotW * (V.lerp(ps, cs, k) + 0.5);
          const lift = ps !== cs ? Math.sin(k * Math.PI) * (ti % 2 ? -9 : 9) : 0;
          const col = ti === TR ? C.d : C.a;
          const cw = Math.min(slotW - 8, 100);
          V.box(g, cx - cw / 2, chipY + lift, cw, chipH, { r: 9, fill: V.alpha(col, 0.16), stroke: ti === TR ? C.d : V.alpha(C.a, 0.6), lw: ti === TR ? 2 : 1.2 });
          V.text(g, words[ti], cx, chipY + lift + chipH / 2 + 0.5, { size: narrow ? 11.5 : 13, weight: 600, align: 'center', base: 'middle', color: C.ink, maxW: cw - 6 });
          const v = S.cache.cur.byTok[ti];
          const bw = Math.max(2, Math.min(8, (cw - 10) / D - 1.5));
          const gx = cx - (D * (bw + 1.5)) / 2, base = glyTop + glyH / 2;
          V.line(g, gx - 2, base, gx + D * (bw + 1.5), base, { color: V.alpha(C.muted, 0.5), lw: 1 });
          v.forEach((x, d) => {
            const hh = (x / S.mx) * (glyH / 2 - 2);
            g.fillStyle = x >= 0 ? C.a : C.b;
            V.rr(g, gx + d * (bw + 1.5), hh >= 0 ? base - hh : base, bw, Math.max(1, Math.abs(hh)), 1.5); g.fill();
          });
          const changed = diffTok(ti) > 1e-9;
          V.text(g, changed ? 'changed' : 'same', cx, glyTop + glyH + 13, { size: 11, font: 'mono', align: 'center', color: changed ? C.c : C.a });
        }
        const mTop = glyTop + glyH + 44;
        const cell = Math.max(22, Math.min(40, (h - mTop - 46) / 5, (w - 150) / 5));
        const labs = S.order.map((ti) => words[ti]);
        const colLabs = labs.map((s) => (narrow && s.length > 4 ? s.slice(0, 4) + '.' : s));
        const mW = 5 * cell + 8;
        const mx0 = Math.max(narrow ? 76 : 96, (w - mW) / 2);
        V.text(g, 'attention matrix (row = query, column = key)', mx0 + mW / 2, mTop - 10, { size: 11.5, align: 'center', color: C.muted });
        V.matrix(g, { x: mx0, y: mTop, cell, data: S.cache.cur.att, min: 0, max: 1, cmap: 'seq', values: cell >= 30, digits: 2, gap: 2, rowLabels: labs, colLabels: colLabs, hiRow: S.order.indexOf(TR), hiColor: C.d });
        const msg = S.rope ? (narrow ? 'with RoPE: output depends on place' : 'with RoPE: each token\'s output depends on where it sits')
          : S.causal ? (narrow ? 'mask only: changes who sees whom' : 'mask only: it changes who each token sees, which is already order information')
            : (narrow ? 'no position: the vector just moves' : 'no position: each token\'s vector stays the same, it just changes place');
        V.text(g, msg, w / 2, h - 10, { size: 12, align: 'center', color: S.rope || S.causal ? C.c : C.a, weight: 500, maxW: w - 20 });
      });
    }

    /* =============================================================
       Fig 2 — two clock hands: the score depends only on m − n
       ============================================================= */
    {
      const fig = root.querySelector('#fig-ponteiros');
      const S = { m: 2, n: 7, th: 0.4, joint: false, j0: 0 };
      const aq = 0.35, ak = 1.25, nq = 1, nk = 0.85;
      let H = null;
      const ctl = UI.controls(fig);
      UI.slider(ctl, { label: 'position m (query)', min: 0, max: 24, step: 1, value: S.m, onInput: (v) => { S.m = v; } });
      UI.slider(ctl, { label: 'position n (key)', min: 0, max: 24, step: 1, value: S.n, onInput: (v) => { S.n = v; } });
      UI.slider(ctl, { label: 'θ (rad/token)', min: 0.1, max: 1.2, step: 0.05, value: S.th, fmt: (v) => sf(v, 2), onInput: (v) => { S.th = v; } });
      UI.toggle(ctl, { label: 'move together', value: false, onChange: (v) => { S.joint = v; S.j0 = H ? H.t : 0; } });
      const rDiff = UI.readout(ctl, 'm − n', '—'), rDot = UI.readout(ctl, 'q̃·k̃', '—');
      let lastKey = '';
      H = V.fig(fig, (g, w, h, t) => {
        if (w < 60 || h < 60) return; // transient frame before layout
        const C = V.C();
        const sh = S.joint ? (t - S.j0) * 0.9 : 0;
        const m = S.m + sh, n = S.n + sh;
        const pq = aq + m * S.th, pk = ak + n * S.th;
        const score = nq * nk * Math.cos(pq - pk);
        const key = (S.m - S.n) + '|' + score.toFixed(3);
        if (key !== lastKey) { lastKey = key; rDiff.set(String(S.m - S.n).replace('-', '−')); rDot.set(mfmt(score, 3)); }
        const narrow = w < 620;
        let cx, cy, R, px0, px1, py0, py1;
        if (narrow) { R = Math.min(80, w / 2 - 54); cx = w / 2; cy = 22 + R + 6; px0 = 44; px1 = w - 16; py0 = cy + R + 72; py1 = h - 34; }
        else { R = Math.min(h / 2 - 54, w * 0.2); cx = 40 + R + 16; cy = h / 2 + 4; px0 = cx + R + 78; px1 = w - 24; py0 = 48; py1 = h - 50; }
        g.strokeStyle = V.alpha(C.muted, 0.35); g.lineWidth = 1; g.beginPath(); g.arc(cx, cy, R, 0, TAU); g.stroke();
        V.line(g, cx - R - 8, cy, cx + R + 8, cy, { color: V.alpha(C.muted, 0.22), lw: 1 });
        V.line(g, cx, cy - R - 8, cx, cy + R + 8, { color: V.alpha(C.muted, 0.22), lw: 1 });
        const P = (a, r) => [cx + r * Math.cos(a), cy - r * Math.sin(a)];
        [[aq, nq, C.a], [ak, nk, C.b]].forEach(([a, r, col]) => { const p = P(a, r * R); V.arrow(g, cx, cy, p[0], p[1], { color: V.alpha(col, 0.3), lw: 1.5, dash: [4, 4], head: 7 }); });
        const arc = (a0, da, rr, col) => { g.strokeStyle = col; g.lineWidth = 2; g.beginPath(); g.arc(cx, cy, rr, -a0, -(a0 + da), true); g.stroke(); };
        const rq = ((m * S.th) % TAU + TAU) % TAU, rk = ((n * S.th) % TAU + TAU) % TAU;
        arc(aq, rq, R * 0.3, V.alpha(C.a, 0.85));
        arc(ak, rk, R * 0.42, V.alpha(C.b, 0.85));
        let dphi = ((pq - pk) % TAU + TAU) % TAU; if (dphi > Math.PI) dphi -= TAU;
        g.strokeStyle = C.d; g.lineWidth = 2.5; g.beginPath(); g.arc(cx, cy, R * 0.62, -pk, -(pk + dphi), dphi > 0); g.stroke();
        const lp = P(pk + dphi / 2, R * 0.62 + 16);
        V.text(g, 'Δφ', lp[0], lp[1], { size: 12, font: 'mono', weight: 700, align: 'center', base: 'middle', color: C.d });
        const qp = P(pq, nq * R), kp = P(pk, nk * R);
        V.arrow(g, cx, cy, qp[0], qp[1], { color: C.a, lw: 3, head: 11 });
        V.arrow(g, cx, cy, kp[0], kp[1], { color: C.b, lw: 3, head: 11 });
        const ql = P(pq, nq * R + 18), kl = P(pk, nk * R + 18);
        V.glow(g, qp[0], qp[1], 14, C.a, 0.6); V.glow(g, kp[0], kp[1], 14, C.b, 0.6);
        V.text(g, 'q̃', ql[0], ql[1], { size: 14, weight: 700, align: 'center', base: 'middle', color: C.a });
        V.text(g, 'k̃', kl[0], kl[1], { size: 14, weight: 700, align: 'center', base: 'middle', color: C.b });
        if (narrow) { V.text(g, 'm·θ = ' + sf(m * S.th, 2) + ' rad', w / 2 - 8, cy + R + 22, { size: 11.5, font: 'mono', color: C.a, align: 'right' }); V.text(g, 'n·θ = ' + sf(n * S.th, 2) + ' rad', w / 2 + 8, cy + R + 22, { size: 11.5, font: 'mono', color: C.b }); }
        else { V.text(g, 'm·θ = ' + sf(m * S.th, 2) + ' rad', cx - R, cy + R + 22, { size: 11.5, font: 'mono', color: C.a }); V.text(g, 'n·θ = ' + sf(n * S.th, 2) + ' rad', cx + R, cy + R + 22, { size: 11.5, font: 'mono', color: C.b, align: 'right' }); }
        // plot of the score as a function of m - n
        const x2 = (d) => V.lerp(px0, px1, (d + 24) / 48);
        const y2 = (s) => V.lerp(py1, py0, (s + 1) / 2);
        V.text(g, 'score q̃·k̃ as a function of m − n', px0, py0 - 18, { size: 12, color: C.muted });
        [-1, 0, 1].forEach((s) => { V.line(g, px0, y2(s), px1, y2(s), { color: V.alpha(C.muted, s === 0 ? 0.45 : 0.15), lw: 1 }); V.text(g, s === 0 ? '0' : s > 0 ? '+1' : '−1', px0 - 8, y2(s) + 4, { size: 11, font: 'mono', align: 'right', color: C.muted }); });
        [-24, -12, 0, 12, 24].forEach((d) => V.text(g, String(d).replace('-', '−'), x2(d), py1 + 16, { size: 11, font: 'mono', align: 'center', color: C.muted }));
        V.text(g, 'm − n', px1, py1 + 30, { size: 11, font: 'mono', align: 'right', color: C.muted });
        g.beginPath();
        for (let k = 0; k <= 240; k++) { const d = -24 + (48 * k) / 240; const s = nq * nk * Math.cos(aq - ak + d * S.th); const X = x2(d), Y = y2(s); if (k) g.lineTo(X, Y); else g.moveTo(X, Y); }
        g.strokeStyle = V.alpha(C.c, 0.9); g.lineWidth = 2; g.stroke();
        const dc = m - n;
        V.line(g, x2(dc), py0, x2(dc), py1, { color: V.alpha(C.d, 0.35), lw: 1, dash: [3, 4] });
        V.glow(g, x2(dc), y2(score), 16, C.d); V.dot(g, x2(dc), y2(score), 5.5, C.d);
        if (S.joint) V.text(g, 'm and n moving together: the dot stays put', (px0 + px1) / 2, py1 - 8, { size: 11.5, align: 'center', color: C.d, weight: 600 });
      });
    }

    /* =============================================================
       Fig 3 — RoPE cell by cell (stepper)
       ============================================================= */
    {
      const fig = root.querySelector('#fig-matriz');
      const q = [0.9, -0.4, 0.3, 0.8, -0.7, 0.5, 0.6, -0.2], D = 8, B = 100;
      const th = ropeInv(B, D);
      const S = { m: 5 };
      const norm = (v) => Math.sqrt(v.reduce((s, x) => s + x * x, 0));
      const steps = [
        { t: 'The q vector of a head', d: 'An 8-dimensional query (illustrative values). On its own, it doesn\'t know which position it is at.' },
        { t: 'Group into pairs', d: 'Interleaved style: (0,1), (2,3), (4,5), (6,7). Each pair becomes a point in a plane, drawn on the dials below.' },
        { t: 'One angle per pair', d: 'Pair \\(i\\) rotates by \\(m\\,\\theta_i\\), with \\(\\theta_i = b^{-2i/d}\\): 1, 0.316, 0.1 and 0.032 rad/token. Pair 0 spins fast and pair 3 barely moves.' },
        { t: 'The rotation matrix', d: 'One 2×2 rotation per pair, on the diagonal, and zeros everywhere else. The numbers are \\(\\cos m\\theta_i\\) and \\(\\pm\\sin m\\theta_i\\).' },
        { t: 'Multiply', d: 'We compute \\(\\tilde q = R_m\\, q\\). Each row of \\(R_m\\) only touches the two dimensions of its pair. Notice that \\(\\lVert\\tilde q\\rVert = \\lVert q\\rVert\\): a rotation doesn\'t change the length.' },
        { t: 'The kernels\' shortcut', d: 'Nobody builds \\(R_m\\). The kernel reads row \\(m\\) of a precomputed cos/sin table and computes \\(\\tilde q = q\\odot\\cos + \\operatorname{rot}(q)\\odot\\sin\\), with \\(\\operatorname{rot}(x_0, x_1, \\ldots) = (-x_1, x_0, \\ldots)\\).' },
      ];
      const st = UI.stepper(fig, steps, { autoplay: 6500 });
      const ctl = UI.controls(fig);
      UI.slider(ctl, { label: 'position m', min: 0, max: 40, step: 1, value: S.m, onInput: (v) => { S.m = v; } });
      UI.readout(ctl, '‖q‖ = ‖q̃‖', sf(norm(q), 3));
      const buildR = (m) => { const R = Array.from({ length: D }, () => new Array(D).fill(0)); for (let i = 0; i < D / 2; i++) { const a = m * th[i], c = Math.cos(a), s = Math.sin(a); R[2 * i][2 * i] = c; R[2 * i][2 * i + 1] = -s; R[2 * i + 1][2 * i] = s; R[2 * i + 1][2 * i + 1] = c; } return R; };
      fitH(fig, 510, 460, 640);
      V.fig(fig, (g, w, h) => {
        if (w < 60 || h < 60) return; // transient frame before layout
        const C = V.C();
        const i = st.i, el = st.elapsed();
        const m = S.m, R = buildR(m);
        const qt = R.map((row) => row.reduce((s, a, k) => s + a * q[k], 0));
        const narrow = w < 640;
        const gap = narrow ? 1.5 : 2, sepQ = narrow ? 10 : 16, sepEq = narrow ? 24 : 36;
        const c = Math.floor(V.clamp(narrow ? (w - 12 - 7 * gap - sepQ - sepEq) / 10 : (w - 76) / 10.3, 22, narrow ? 33 : 40));
        const mW = 8 * c + 7 * gap;
        const totalW = mW + sepQ + c + sepEq + c;
        const x0 = Math.max(6, (w - totalW) / 2), y0 = 34;
        const xq = x0 + mW + sepQ, xqt = xq + c + sepEq;
        const pc = [C.a, C.b, C.c, C.d];
        const fs = 11;
        /* always 2 decimals (cos 0.28 and sin 0.96 square-sum to 1); in narrow cells the leading zero is dropped: −.96 */
        const cellTxt = (v) => { if (Math.abs(v) < 0.005) return c < 38 ? '0' : '0.00'; if (c < 38 && Math.abs(v) >= 0.995) return v > 0 ? '1' : '−1'; const s = mfmt(v, 2); return c < 38 ? s.replace(/^(−?)0\./, '$1.') : s; };
        const txtOn = (v) => (Math.abs(v) > 0.55 ? (C.dark ? '#00210F' : '#fff') : C.ink);
        const rowAct = i === 4 ? Math.min(8, Math.floor(el / 0.55)) : 8;
        V.text(g, 'R_m  (m = ' + m + ')', x0 + mW / 2, y0 - 12, { size: 12, font: 'mono', align: 'center', color: i >= 3 ? C.ink : V.alpha(C.muted, 0.5) });
        V.text(g, 'q', xq + c / 2, y0 - 12, { size: 12.5, font: 'mono', weight: 700, align: 'center', color: C.ink });
        V.text(g, 'q̃', xqt + c / 2, y0 - 12, { size: 12.5, font: 'mono', weight: 700, align: 'center', color: i >= 4 ? C.ink : V.alpha(C.muted, 0.5) });
        for (let r = 0; r < 8; r++) for (let k = 0; k < 8; k++) {
          const v = R[r][k], x = x0 + k * (c + gap), y = y0 + r * (c + gap);
          const blk = Math.floor(r / 2) === Math.floor(k / 2);
          g.globalAlpha = i >= 3 ? (blk ? 1 : 0.5) : 0.1;
          g.fillStyle = blk ? V.cmap(v, 'div') : V.alpha(C.muted, 0.1);
          V.rr(g, x, y, c, c, 4); g.fill();
          g.globalAlpha = 1;
          if (i >= 3 && blk) V.text(g, cellTxt(v), x + c / 2, y + c / 2 + 0.5, { size: fs, font: 'mono', align: 'center', base: 'middle', color: txtOn(v) });
          else if (i >= 3) V.text(g, '0', x + c / 2, y + c / 2 + 0.5, { size: 11, font: 'mono', align: 'center', base: 'middle', color: V.alpha(C.muted, 0.6) });
        }
        if (i === 4 && rowAct < 8) {
          const y = y0 + rowAct * (c + gap);
          g.strokeStyle = C.c; g.lineWidth = 2.5; V.rr(g, x0 - 2, y - 2, mW + 4, c + 4, 5); g.stroke();
          const p = Math.floor(rowAct / 2);
          V.rr(g, xq - 2, y0 + 2 * p * (c + gap) - 2, c + 4, 2 * c + gap + 4, 5); g.stroke();
          V.rr(g, xqt - 2, y - 2, c + 4, c + 4, 5); g.stroke();
        }
        for (let r = 0; r < 8; r++) {
          const y = y0 + r * (c + gap);
          const col = i >= 1 ? pc[Math.floor(r / 2)] : C.a;
          g.fillStyle = V.alpha(col, i >= 1 ? 0.3 : 0.2); V.rr(g, xq, y, c, c, 4); g.fill();
          g.strokeStyle = V.alpha(col, 0.8); g.lineWidth = 1; g.stroke();
          V.text(g, cellTxt(q[r]), xq + c / 2, y + c / 2 + 0.5, { size: fs, font: 'mono', align: 'center', base: 'middle', color: C.ink });
          const vis = i >= 4 && r < rowAct;
          g.fillStyle = V.alpha(pc[Math.floor(r / 2)], vis ? 0.3 : 0.06); V.rr(g, xqt, y, c, c, 4); g.fill();
          if (vis) { g.strokeStyle = V.alpha(pc[Math.floor(r / 2)], 0.8); g.stroke(); V.text(g, cellTxt(qt[r]), xqt + c / 2, y + c / 2 + 0.5, { size: fs, font: 'mono', align: 'center', base: 'middle', color: C.ink }); }
        }
        if (i >= 1) for (let p = 0; p < 4; p++) { const y = y0 + 2 * p * (c + gap); V.line(g, xq - 6, y + 2, xq - 6, y + 2 * c + gap - 2, { color: pc[p], lw: 3 }); }
        V.text(g, '=', xq + c + sepEq / 2, y0 + 4 * (c + gap) - 1, { size: 20, weight: 700, align: 'center', base: 'middle', color: i >= 4 ? C.ink : V.alpha(C.muted, 0.4) });
        const yb = y0 + 8 * (c + gap) + 18;
        if (i === 0) { V.text(g, '8 numbers, and none of them tells the position', w / 2, yb + 34, { size: 13, align: 'center', color: C.muted }); return; }
        if (i === 5) {
          const rowsM = [m - 1, m, m + 1].filter((x) => x >= 0);
          const cw = Math.min(46, (w - 90) / 8), ch = 17;
          const tx = (w - 8 * cw) / 2 + 22;
          V.text(g, 'cos/sin table: one row per position (columns = pairs)', w / 2, yb + 4, { size: 11.5, align: 'center', color: C.muted });
          ['cos θ₀', 'cos θ₁', 'cos θ₂', 'cos θ₃', 'sin θ₀', 'sin θ₁', 'sin θ₂', 'sin θ₃'].forEach((l, k) => V.text(g, narrow ? l.replace('cos ', 'c').replace('sin ', 's') : l, tx + k * cw + cw / 2, yb + 20, { size: 11, font: 'mono', align: 'center', color: k < 4 ? C.a : C.b }));
          rowsM.forEach((pm, rr) => {
            const y = yb + 26 + rr * ch;
            if (pm === m) { g.fillStyle = V.alpha(C.d, 0.18); V.rr(g, tx - 40, y, 8 * cw + 42, ch, 4); g.fill(); }
            V.text(g, 'm=' + pm, tx - 6, y + ch / 2 + 4, { size: 11, font: 'mono', align: 'right', color: pm === m ? C.d : C.muted });
            for (let k = 0; k < 8; k++) { const a = pm * th[k % 4]; const v = k < 4 ? Math.cos(a) : Math.sin(a); V.text(g, mfmt(v, 2), tx + k * cw + cw / 2, y + ch / 2 + 4, { size: 11, font: 'mono', align: 'center', color: pm === m ? C.ink : C.muted }); }
          });
          V.text(g, 'q̃ = q ⊙ cos + rot(q) ⊙ sin', w / 2, Math.min(h - 10, yb + 26 + rowsM.length * ch + 22), { size: 13, font: 'mono', weight: 600, align: 'center', color: C.ink });
          return;
        }
        const rd = Math.max(20, Math.min(36, (w - 40) / 8 - 12, (h - yb - 34) / 2));
        const sp = Math.min(2 * rd + 40, (w - 20) / 4);
        const mq = Math.max(...[0, 1, 2, 3].map((p) => Math.hypot(q[2 * p], q[2 * p + 1])));
        for (let p = 0; p < 4; p++) {
          const cx = w / 2 + (p - 1.5) * sp, cy = yb + rd + 4;
          g.strokeStyle = V.alpha(C.muted, 0.4); g.lineWidth = 1; g.beginPath(); g.arc(cx, cy, rd, 0, TAU); g.stroke();
          const a0 = Math.atan2(q[2 * p + 1], q[2 * p]), r0 = (Math.hypot(q[2 * p], q[2 * p + 1]) / mq) * (rd - 3);
          V.arrow(g, cx, cy, cx + r0 * Math.cos(a0), cy - r0 * Math.sin(a0), { color: V.alpha(pc[p], 0.35), lw: 1.5, dash: [3, 3], head: 6 });
          if (i >= 2) {
            const f = i === 2 ? V.ease.inOut(Math.min(1, el / 1.3)) : 1;
            const a = a0 + m * th[p] * f;
            V.arrow(g, cx, cy, cx + r0 * Math.cos(a), cy - r0 * Math.sin(a), { color: pc[p], lw: 2.5, head: 8 });
          }
          V.text(g, 'pair ' + p, cx, cy + rd + 14, { size: 11, font: 'mono', align: 'center', color: pc[p] });
          if (i >= 2) V.text(g, 'mθ=' + sf(m * th[p], m * th[p] >= 10 ? 1 : 2), cx, cy + rd + 27, { size: 11, font: 'mono', align: 'center', color: C.muted });
        }
      });
    }

    /* =============================================================
       Fig 4 — the 32-hand clock
       ============================================================= */
    {
      const fig = root.querySelector('#fig-relogio');
      const S = { base: 1e7, L: 262144, v: 3.6, play: true };
      let internal = false;
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'rope_theta', options: [{ v: 1e4, label: '10⁴' }, { v: 1.6e5, label: '1.6·10⁵' }, { v: 1e7, label: '10⁷' }], value: 1e7, onChange: (v) => { S.base = v; upd(); } });
      UI.seg(ctl, { label: 'training context L', options: [{ v: 65536, label: '65,536' }, { v: 262144, label: '262,144' }], value: 262144, onChange: (v) => { S.L = v; upd(); } });
      const sl = UI.slider(ctl, { label: 'position m', min: 0, max: 6.02, step: 0.01, value: S.v, fmt: (v) => nf(Math.pow(10, v)), onInput: (v) => { S.v = v; if (!internal) { S.play = false; tg.set(false); } } });
      const tg = UI.toggle(ctl, { label: 'auto-advance', value: true, onChange: (v) => { S.play = v; } });
      const rMax = UI.readout(ctl, 'λ of the slowest pair', '—'), rN = UI.readout(ctl, 'pairs with λ > L', '—');
      function upd() { const th = ropeInv(S.base, 64); rMax.set(short(TAU / th[31]) + ' tokens'); rN.set(String(th.filter((x) => TAU / x > S.L).length)); }
      upd();
      let frame = 0;
      V.fig(fig, (g, w, h, t, dt) => {
        if (w < 60 || h < 60) return; // transient frame before layout
        const C = V.C();
        if (S.play && !LA.paused) {
          S.v += dt * 0.2; if (S.v > 6.02) S.v = 0;
          if ((frame++ % 6) === 0) { internal = true; sl.set(+S.v.toFixed(2)); internal = false; }
        }
        const m = Math.round(Math.pow(10, S.v));
        const th = ropeInv(S.base, 64);
        const cols = w < 600 ? 8 : 16, rows = 32 / cols;
        const cellW = (w - 24) / cols;
        const r = Math.max(11, Math.min(23, cellW / 2 - 7));
        const cellH = 2 * r + 22, gy0 = 32;
        V.text(g, 'position m = ' + nf(m), 12, 19, { size: 13.5, weight: 700, color: m > S.L ? C.c : C.ink });
        if (m > S.L) V.text(g, 'beyond the training context', w - 12, 19, { size: 12, align: 'right', color: C.c, weight: 600 });
        for (let i = 0; i < 32; i++) {
          const cx = 12 + cellW * ((i % cols) + 0.5), cy = gy0 + Math.floor(i / cols) * cellH + r + 4;
          const lam = TAU / th[i], slow = lam > S.L;
          const span = Math.min(TAU, S.L * th[i]);
          g.beginPath(); g.moveTo(cx, cy); g.arc(cx, cy, r, -Math.PI / 2, -Math.PI / 2 + span); g.closePath();
          g.fillStyle = V.alpha(slow ? C.d : C.a, slow ? 0.3 : 0.12); g.fill();
          g.strokeStyle = slow ? V.alpha(C.d, 0.95) : V.alpha(C.muted, 0.45); g.lineWidth = slow ? 1.6 : 1;
          g.beginPath(); g.arc(cx, cy, r, 0, TAU); g.stroke();
          const a = (m * th[i]) % TAU, unseen = slow && m > S.L;
          V.line(g, cx, cy, cx + (r - 2) * Math.sin(a), cy - (r - 2) * Math.cos(a), { color: unseen ? C.c : slow ? C.d : C.a, lw: 2.2 });
          if (unseen) V.glow(g, cx + (r - 2) * Math.sin(a), cy - (r - 2) * Math.cos(a), 8, C.c, 0.8);
          V.dot(g, cx, cy, 2, C.ink);
          V.text(g, short(lam), cx, cy + r + 13, { size: 11, font: 'mono', align: 'center', color: slow ? C.d : C.muted });
        }
        const y0 = gy0 + rows * cellH + 22, y1 = h - 26, x0 = 44, x1 = w - 12;
        const ly = (l) => V.lerp(y1, y0, V.clamp(l / 8, 0, 1));
        V.text(g, 'log₁₀ λᵢ per pair (0 = fastest → 31 = slowest)', x0, y0 - 8, { size: 11.5, color: C.muted });
        [0, 2, 4, 6, 8].forEach((l) => { V.line(g, x0, ly(l), x1, ly(l), { color: V.alpha(C.muted, 0.12), lw: 1 }); V.text(g, '10' + ['⁰', '', '²', '', '⁴', '', '⁶', '', '⁸'][l], x0 - 6, ly(l) + 4, { size: 11, font: 'mono', align: 'right', color: C.muted }); });
        const bw = (x1 - x0) / 32;
        for (let i = 0; i < 32; i++) {
          const lam = TAU / th[i], l = Math.log10(lam), slow = lam > S.L;
          g.fillStyle = V.alpha(slow ? C.d : C.a, 0.8);
          V.rr(g, x0 + i * bw + 1, ly(l), bw - 2, y1 - ly(l), Math.min(3, bw / 3)); g.fill();
        }
        [0, 8, 16, 24, 31].forEach((i) => V.text(g, String(i), x0 + i * bw + bw / 2, y1 + 15, { size: 11, font: 'mono', align: 'center', color: C.muted }));
        const yL = ly(Math.log10(S.L));
        V.line(g, x0, yL, x1, yL, { color: C.d, lw: 1.5, dash: [5, 4] });
        V.text(g, 'L = ' + nf(S.L), x0 + 4, yL - 5, { size: 11, font: 'mono', color: C.d, weight: 600 });
        const ym = ly(Math.log10(Math.max(1, m)));
        V.line(g, x0, ym, x1, ym, { color: V.alpha(C.c, 0.8), lw: 1.2, dash: [2, 4] });
        V.text(g, 'm', x1 - 4, ym - 5, { size: 11, font: 'mono', align: 'right', color: C.c, weight: 700 });
      });
    }

    /* =============================================================
       Fig 5 — positional anatomy of a head
       ============================================================= */
    {
      const fig = root.querySelector('#fig-cabeca');
      const HEADS = {
        qwen: { label: 'Qwen3.8 · QSA head', dim: 256, rot: 64, start: 0, neox: true, inf: true, base: 1e7, theta: '10⁷', rc: 'qwen' },
        ds: { label: 'DeepSeek-V4 · head (K = V)', dim: 512, rot: 64, start: 448, neox: false, base: 1e4, theta: '10⁴ (SWA) · 1.6·10⁵ (compressor layers)', rc: 'ds' },
        glm: { label: 'GLM-5.3 · MLA NoPE', dim: 256, rot: 0, start: 0, neox: null, base: 0, theta: 'absent (NoPE)', rc: 'glm' },
        draft: { label: 'GLM-5.3 · DFlash2 drafter', dim: 128, rot: 128, start: 0, neox: true, base: 1e4, theta: '10⁴', rc: 'glm' },
      };
      const KV = {
        qwen: { title: 'K and V per layer and token (FP8 KV), 2 KV heads', segs: [['K₀ RoPE', 64, 'a'], ['K₀ NoPE', 192, 'b'], ['K₁ RoPE', 64, 'a'], ['K₁ NoPE', 192, 'b'], ['V (no RoPE)', 512, 'muted']], total: '1,024 B/layer (2,048 B in BF16)' },
        ds: { title: 'one KV entry (K = V, one latent per token)', segs: [['448 B NoPE · FP8', 448, 'b'], ['128 B RoPE · BF16', 128, 'a'], ['8 B', 8, 'd']], total: '584 B per entry' },
        glm: { title: 'fp8_ds_mla record per DSA layer', segs: [['512 B NoPE latent · FP8', 512, 'b'], ['16 B', 16, 'd'], ['128 B of zeros', 128, 'zero']], total: '656 B (useful: 528 B)' },
        draft: { title: 'drafter K and V (GQA, 8 KV heads × 128)', segs: [['K: 8×128, all rotated', 1024, 'a'], ['V: 8×128, no RoPE', 1024, 'muted']], total: '2,048-token sliding window' },
      };
      const S = { k: 'ds', wrong: false };
      const angErr = (fmt) => {
        try {
          const r = V.rng(5); let s = 0; const N = 1500;
          for (let k = 0; k < N; k++) {
            const a = r() * TAU, x = Math.cos(a), y = Math.sin(a);
            const ex = LA.num.encode(fmt, x).v, ey = LA.num.encode(fmt, y).v;
            let d = Math.atan2(ey, ex) - a; d = Math.atan2(Math.sin(d), Math.cos(d)); s += Math.abs(d);
          }
          return (s / N) * 180 / Math.PI;
        } catch (e) { return NaN; }
      };
      const eF8 = angErr('e4m3'), eB16 = angErr('bf16');
      /* mini-demo (illustrative): a head "trained" to look 3 tokens back.
         With the right convention, k = R₃ q, so q̃·k̃ = Σ |q_i|² cos((Δ+3)θ_i) peaks at m − n = 3.
         With the swapped convention, the kernel rotates other pairs and the learned peak falls apart. */
      const LOOK = 3, DMAX = 24;
      const pairsOf = (n, neox) => Array.from({ length: n / 2 }, (_, i) => (neox ? [i, i + n / 2] : [2 * i, 2 * i + 1]));
      const rotP = (v, pr, ang) => { const o = v.slice(); pr.forEach(([a, b], i) => { const c = Math.cos(ang[i]), s = Math.sin(ang[i]); o[a] = v[a] * c - v[b] * s; o[b] = v[a] * s + v[b] * c; }); return o; };
      const dotv = (a, b) => a.reduce((s, x, k) => s + x * b[k], 0);
      const CURVES = {};
      Object.keys(HEADS).forEach((key) => {
        const hd = HEADS[key];
        if (!hd.rot) { CURVES[key] = null; return; }
        const n = hd.rot, th = ropeInv(hd.base, n), pc = pairsOf(n, hd.neox), pw = pairsOf(n, !hd.neox);
        const r = V.rng(2), q = new Array(n).fill(0);
        pc.forEach(([a, b], i) => { const mg = Math.exp(-i / 6) + 0.05, ph = r() * TAU; q[a] = mg * Math.cos(ph); q[b] = mg * Math.sin(ph); });
        const k = rotP(q, pc, th.map((x) => LOOK * x));
        const ok = [], bad = [];
        for (let d = 0; d <= DMAX; d++) { ok.push(dotv(q, rotP(k, pc, th.map((x) => -d * x)))); bad.push(dotv(q, rotP(k, pw, th.map((x) => -d * x)))); }
        const am = (a) => a.indexOf(Math.max(...a));
        CURVES[key] = { ok, bad, pOk: am(ok), pBad: am(bad), lo: Math.min(...ok, ...bad), hi: Math.max(...ok, ...bad) };
      });
      fitH(fig, 560, 600, 560);
      const ctl = UI.controls(fig);
      UI.select(ctl, { label: 'head', options: [{ v: 'ds', label: 'DeepSeek-V4 (512, last 64)' }, { v: 'qwen', label: 'Qwen3.8 QSA (256, 25%)' }, { v: 'glm', label: 'GLM-5.3 MLA (NoPE)' }, { v: 'draft', label: 'GLM-5.3 drafter (128, 100%)' }], value: 'ds', onChange: (v) => { S.k = v; upd(); } });
      UI.toggle(ctl, { label: 'swapped convention (bug)', value: false, onChange: (v) => { S.wrong = v; upd(); } });
      const rR = UI.readout(ctl, 'RoPE / NoPE dims', '—');
      const rP = UI.readout(ctl, 'score peak', '—');
      UI.readout(ctl, 'FP8 · BF16 angle error', Number.isNaN(eF8) ? '—' : sf(eF8, 2) + '° · ' + sf(eB16, 2) + '°');
      function upd() {
        const hd = HEADS[S.k], cv = CURVES[S.k];
        rR.set(hd.rot + ' / ' + (hd.dim - hd.rot));
        rP.set(!cv ? 'none (NoPE)' : S.wrong ? cv.pBad + ' back ✗' : cv.pOk + ' back ✓');
      }
      upd();
      V.fig(fig, (g, w, h, t) => {
        if (w < 60 || h < 60) return; // transient frame before layout
        const C = V.C();
        const hd = HEADS[S.k], kv = KV[S.k], cv = CURVES[S.k];
        const narrow = w < 560;
        const rc = C[hd.rc] || C.a;
        V.text(g, hd.label, 14, 20, { size: 13.5, weight: 700, color: rc, maxW: w - 28 });
        if (narrow) V.text(g, 'θ = ' + hd.theta, 14, 38, { size: 11, font: 'mono', color: C.muted, maxW: w - 28 });
        else V.text(g, 'θ = ' + hd.theta, w - 14, 20, { size: 11.5, font: 'mono', align: 'right', color: C.muted, maxW: w * 0.45 });
        // strip for the whole head (64 cells)
        const x0 = 14, x1 = w - 14, cw = (x1 - x0) / 64, y0 = narrow ? 64 : 44, ch = 22;
        const convY = y0 + 62;
        const grp = hd.dim / 64;
        for (let c = 0; c < 64; c++) {
          const d0 = c * grp, isR = hd.rot > 0 && d0 >= hd.start && d0 < hd.start + hd.rot;
          const a = isR ? 0.5 + 0.45 * (0.5 + 0.5 * Math.sin(t * 3.2 - c * 0.55)) : 0.32;
          g.fillStyle = V.alpha(isR ? C.a : C.b, a);
          V.rr(g, x0 + c * cw + 0.5, y0, Math.max(1, cw - 1), ch, Math.min(3, cw / 3)); g.fill();
        }
        V.text(g, '0', x0, y0 + ch + 14, { size: 11, font: 'mono', color: C.muted });
        V.text(g, String(hd.dim), x1, y0 + ch + 14, { size: 11, font: 'mono', align: 'right', color: C.muted });
        V.text(g, '1 cell = ' + grp + ' dims', w / 2, y0 + ch + 14, { size: 11, align: 'center', color: C.muted });
        if (hd.rot > 0) {
          const rx0 = x0 + (hd.start / hd.dim) * (x1 - x0), rx1 = x0 + ((hd.start + hd.rot) / hd.dim) * (x1 - x0);
          V.line(g, rx0, y0 - 6, rx1, y0 - 6, { color: C.a, lw: 2 });
          V.text(g, 'RoPE ' + hd.rot, (rx0 + rx1) / 2, y0 - 10, { size: 11, font: 'mono', align: 'center', color: C.a, weight: 700 });
        }
        // rotated block, dimension by dimension
        const zh = 24, zy = hd.rot > 0 && hd.neox ? convY + 90 : convY + 26;
        let bandTop;
        if (hd.rot > 0) {
          const n = hd.rot, P = n / 2, zx0 = 20, zx1 = w - 20, zw = (zx1 - zx0) / n;
          const pairOf = (d, neox) => (neox ? d % P : d >> 1);
          const lbl = hd.neox
            ? (narrow ? 'NeoX (neox=True): pair i = (i, i+' + P + ')' : 'NeoX (is_neox_style=True): pair i = (i, i + ' + P + ')') + (hd.inf ? ' · inferred' : '')
            : (narrow ? 'interleaved (neox=False): pair (2i, 2i+1)' : 'interleaved (is_neox_style=False): pair i = (2i, 2i+1)');
          V.text(g, lbl, w / 2, convY, { size: narrow ? 11 : 12, font: 'mono', align: 'center', color: C.ink, weight: 600, maxW: w - 20 });
          for (let d = 0; d < n; d++) {
            const p = pairOf(d, hd.neox);
            g.fillStyle = V.cmap(0.95 - (0.75 * p) / (P - 1), 'seq');
            V.rr(g, zx0 + d * zw + 0.4, zy, Math.max(1, zw - 0.8), zh, Math.min(2, zw / 3)); g.fill();
          }
          const hot = Math.floor(t * 3) % P;
          for (let p = 0; p < P; p++) {
            const a = hd.neox ? p : 2 * p, b = hd.neox ? p + P : 2 * p + 1;
            const xa = zx0 + (a + 0.5) * zw, xb = zx0 + (b + 0.5) * zw;
            const ah = Math.min(76, 6 + (xb - xa) * 0.42);
            g.strokeStyle = p === hot ? C.d : V.alpha(C.a, 0.35); g.lineWidth = p === hot ? 2.4 : 1;
            g.beginPath(); g.moveTo(xa, zy - 2); g.bezierCurveTo(xa, zy - ah, xb, zy - ah, xb, zy - 2); g.stroke();
          }
          let below = 0;
          if (S.wrong) {
            below = hd.neox ? 34 : 50;
            for (let p = 0; p < P; p++) {
              const a = !hd.neox ? p : 2 * p, b = !hd.neox ? p + P : 2 * p + 1;
              const xa = zx0 + (a + 0.5) * zw, xb = zx0 + (b + 0.5) * zw;
              // minimum height of 22 px: neighboring pairs (wrong convention on a NeoX head) must be visible
              const ah = Math.max(22, Math.min(46, 6 + (xb - xa) * 0.3));
              g.strokeStyle = V.alpha(C.c, p === hot ? 1 : 0.55); g.lineWidth = p === hot ? 2.2 : 1.1; g.setLineDash([3, 3]);
              g.beginPath(); g.moveTo(xa, zy + zh + 2); g.bezierCurveTo(xa, zy + zh + ah, xb, zy + zh + ah, xb, zy + zh + 2); g.stroke(); g.setLineDash([]);
            }
          }
          V.text(g, 'rotated block: ' + n + ' dims · ' + P + ' pairs (light = fast, dark = slow)', w / 2, zy + zh + below + 16, { size: 11, align: 'center', color: C.muted, maxW: w - 20 });
          if (S.wrong) V.text(g, narrow ? 'pink (below): pairs the kernel would join' : 'pink arcs below: the pairs the kernel would join with the wrong convention (no error raised)', w / 2, zy + zh + below + 32, { size: 11.5, align: 'center', color: C.c, weight: 600, maxW: w - 20 });
          bandTop = zy + zh + below + (S.wrong ? 46 : 32);
        } else {
          V.text(g, '0 rotated dimensions: position comes from elsewhere (section 5)', w / 2, convY, { size: 12.5, align: 'center', color: C.b, weight: 600, maxW: w - 20 });
          const zx0 = 20, zx1 = w - 20, zw = (zx1 - zx0) / 72;
          for (let c = 0; c < 72; c++) {
            const pad = c >= 64;
            g.fillStyle = pad ? V.alpha(C.muted, 0.12) : V.alpha(C.b, 0.35);
            V.rr(g, zx0 + c * zw + 0.4, zy, zw - 0.8, zh, 2); g.fill();
            if (pad) hatch(g, zx0 + c * zw + 0.4, zy, zw - 0.8, zh, V.alpha(C.muted, 0.6), 4);
          }
          V.text(g, 'SM12x kernel: 512 latent + 64 zeros = 576 geometry', w / 2, zy + zh + 18, { size: 11.5, font: 'mono', align: 'center', color: C.muted, maxW: w - 20 });
          bandTop = zy + zh + 34;
        }
        // KV row (bottom) and, in the middle, the score as a function of distance
        const ky = h - 74, kx0 = 14, kx1 = w - 14, kh = 30;
        const bandBot = ky - (narrow ? 44 : 28);
        const blockH = Math.min(bandBot - bandTop, 170);
        if (blockH > 56) {
          const by0 = bandTop + (bandBot - bandTop - blockH) / 2;
          const ttl = cv ? (narrow ? 'score of a head that looks 3 tokens back (illustr.)' : 'score of an illustrative head trained to look 3 tokens back') : (narrow ? 'q·k score vs. distance' : 'q·k score vs. distance: without RoPE, it cannot see distance');
          V.text(g, ttl, w / 2, by0 + 10, { size: 11.5, align: 'center', color: C.ink, weight: 600, maxW: w - 20 });
          const px0 = 40, px1 = w - 18, py0 = by0 + 22, py1 = by0 + blockH - (cv && S.wrong ? 32 : 16);
          const X = (d) => V.lerp(px0, px1, d / DMAX);
          g.strokeStyle = V.alpha(C.muted, 0.35); g.lineWidth = 1; g.strokeRect(px0, py0, px1 - px0, py1 - py0);
          const ticks = !cv ? (narrow ? [0, 12] : [0, 6, 12, 18]) : narrow ? [0, 3, 12] : [0, 3, 6, 12, 18];
          ticks.forEach((d) => V.text(g, String(d), X(d), py1 + 13, { size: 11, font: 'mono', align: 'center', color: cv && d === LOOK ? C.a : C.muted }));
          V.text(g, '24 tokens back', px1, py1 + 13, { size: 11, font: 'mono', align: 'right', color: C.muted });
          if (cv) V.line(g, X(LOOK), py0, X(LOOK), py1, { color: V.alpha(C.a, 0.4), lw: 1, dash: [3, 4] });
          const sd = (t * 5) % (DMAX + 1);
          if (!cv) {
            const yy = (py0 + py1) / 2;
            V.line(g, px0, yy, px1, yy, { color: C.b, lw: 2.2 });
            V.dot(g, X(sd), yy, 4, C.d);
            V.text(g, 'NoPE: the score is the same at any distance (flat line)', (px0 + px1) / 2, yy - 10, { size: 11.5, align: 'center', color: C.b, weight: 600, maxW: px1 - px0 - 10 });
          } else {
            const Y = (v) => V.lerp(py1 - 4, py0 + 4, (v - cv.lo) / (cv.hi - cv.lo || 1));
            const path = (arr, col, lw, dash) => { g.beginPath(); arr.forEach((v, d) => { if (d) g.lineTo(X(d), Y(v)); else g.moveTo(X(d), Y(v)); }); g.strokeStyle = col; g.lineWidth = lw; g.setLineDash(dash || []); g.stroke(); g.setLineDash([]); };
            path(cv.ok, S.wrong ? V.alpha(C.a, 0.35) : C.a, S.wrong ? 1.5 : 2.4);
            if (S.wrong) path(cv.bad, C.c, 2.4, [5, 3]);
            const act = S.wrong ? cv.bad : cv.ok, pk = S.wrong ? cv.pBad : cv.pOk, pcol = S.wrong ? C.c : C.a;
            V.glow(g, X(pk), Y(act[pk]), 12, pcol, 0.7); V.dot(g, X(pk), Y(act[pk]), 4.5, pcol);
            const plab = 'peak: ' + pk + ' back ' + (S.wrong ? '✗' : '✓'), plw = tw(g, plab, 11.5, 'mono', 700);
            const toR = X(pk) + 10 + plw < px1 - 4;
            V.text(g, plab, toR ? X(pk) + 10 : X(pk) - 10, V.clamp(Y(act[pk]) + 4, py0 + 13, py1 - 5), { size: 11.5, font: 'mono', weight: 700, align: toR ? 'left' : 'right', color: pcol });
            const i0 = Math.floor(sd), fr = sd - i0, sv = i0 >= DMAX ? act[DMAX] : V.lerp(act[i0], act[i0 + 1], fr);
            V.dot(g, X(Math.min(sd, DMAX)), Y(sv), 3.5, C.d);
            if (S.wrong) V.text(g, narrow ? 'green: learned · pink: what the kernel computes' : 'faded green: what the head learned (peak at 3) · dashed pink: what the kernel computes', (px0 + px1) / 2, py1 + 28, { size: 11, align: 'center', color: C.muted, maxW: px1 - px0 });
          }
        }
        V.text(g, 'in the KV cache: ' + kv.title, kx0, ky - (narrow ? 26 : 10), { size: 11.5, color: C.muted, maxW: narrow ? w - 28 : w - 250 });
        const tot = kv.segs.reduce((s, x) => s + x[1], 0);
        let x = kx0;
        kv.segs.forEach(([l, b, ck], si) => {
          const sw = ((kx1 - kx0) * b) / tot;
          const col = ck === 'zero' ? C.muted : C[ck];
          g.fillStyle = V.alpha(col, ck === 'zero' ? 0.1 : 0.34);
          V.rr(g, x + 0.5, ky, sw - 1, kh, 5); g.fill();
          if (ck === 'zero') hatch(g, x + 0.5, ky, sw - 1, kh, V.alpha(C.muted, 0.55), 5);
          g.strokeStyle = V.alpha(col, 0.8); g.lineWidth = 1; V.rr(g, x + 0.5, ky, sw - 1, kh, 5); g.stroke();
          const lw = tw(g, l, 11, 'mono', 600);
          if (lw + 8 < sw) V.text(g, l, x + sw / 2, ky + kh / 2 + 0.5, { size: 11, font: 'mono', weight: 600, align: 'center', base: 'middle', color: C.ink });
          else { const cxl = V.clamp(x + sw / 2, kx0 + lw / 2, kx1 - lw / 2); V.text(g, l, cxl, ky + kh + 14 + (si % 2) * 13, { size: 11, font: 'mono', align: 'center', color: col }); }
          x += sw;
        });
        V.text(g, kv.total, kx1, ky - 10, { size: 11.5, font: 'mono', align: 'right', color: C.ink, weight: 600, maxW: w - 28 });
      });
    }

    /* =============================================================
       Fig 6 — NoPE: where order comes from (stepper)
       ============================================================= */
    {
      const fig = root.querySelector('#fig-nope');
      const steps = [
        { t: 'NoPE attention: content only', d: 'In GLM\'s MLA layer, the score is plain \\(q\\cdot k\\). Watch the tokens swap places: the thickness of each arc follows the token\'s <i>content</i>, not where it sits.' },
        { t: 'The causal mask counts', d: 'Token \\(m\\) only sees \\(m+1\\) tokens. In a uniform average, each term weighs \\(1/(m+1)\\): the “size of the past” leaks into the output vector.' },
        { t: 'KDA: the conveyor belt of order', d: 'The 34 KDA layers carry a state that is updated token by token, with decay (schematic below). The state at token \\(m\\) is a mixture in which recent tokens weigh more: changing the order changes the state.' },
        { t: '4-tap short convolution', d: 'Before the recurrence, q, k and v go through a 4-tap causal convolution (<code>short_conv_kernel_size = 4</code>). Each tap has its own weight, so “the previous one” and “two back” can be told apart.' },
        { t: 'The indexer pool has APE', d: 'The indexer merges 4 consecutive keys into one with a per-channel <code>softmax(gate + ape)</code>. <code>index_kpool_compress_ape [4×128]</code> is a learned bias for each position inside the block.' },
        { t: 'The zero pad is exact', d: 'The kernel wants 64 RoPE dims. The overlay pads q and k with zeros, and \\(\\sum_i q_i k_i\\) gains 64 terms \\(0\\times 0\\). The score is identical; the cost is 656 B instead of ~528 B per token and DSA layer.' },
        { t: 'Counterexample: Qwen', d: 'In Qwen3.8, according to the tech report, full NoPE leads to endless generation after post-training. The model keeps RoPE on 64 of the 256 dims of its QSA layers, even with 36 recurrent GDN layers.' },
      ];
      const st = UI.stepper(fig, steps, { autoplay: 7000 });
      const toks = ['the', 'key', 'is', 'under', 'the', 'vase', 'at', 'home'];
      const sim = [0.25, 0.95, 0.2, 0.45, 0.15, 0.8, 0.2, 0.6];
      const tokCol = (C, i) => C.series[i % 6];
      const rngv = V.rng(31);
      const contrib = toks.map(() => Array.from({ length: 16 }, () => rngv.normal()));
      const swaps = [[1, 5], [0, 3], [2, 6], [1, 4], [3, 5], [0, 6]];
      const S6 = { ord: [0, 1, 2, 3, 4, 5, 6], prev: [0, 1, 2, 3, 4, 5, 6], cyc: -1, t0: 0 };
      V.fig(fig, (g, w, h, t) => {
        if (w < 60 || h < 60) return; // transient frame before layout
        const C = V.C();
        const i = st.i, el = st.elapsed();
        const narrow = w < 560;
        const N = 8, pad = 12, slot = (w - 2 * pad) / N, chipW = Math.min(slot - 6, 70), chipH = 28;
        const chips = (y, xs, opt = {}) => {
          for (let k = 0; k < N; k++) {
            const cx = xs ? xs[k] : pad + slot * (k + 0.5);
            const ti = opt.map ? opt.map[k] : k;
            const col = tokCol(C, ti), hi = opt.hi ? opt.hi(k) : false;
            V.box(g, cx - chipW / 2, y, chipW, chipH, { r: 8, fill: V.alpha(col, hi ? 0.35 : 0.14), stroke: V.alpha(col, hi ? 1 : 0.6), lw: hi ? 2 : 1 });
            V.text(g, toks[ti], cx, y + chipH / 2 + 0.5, { size: narrow ? 11 : 12, weight: 600, align: 'center', base: 'middle', color: C.ink, maxW: chipW - 4 });
          }
        };
        if (i === 0) {
          const cyc = Math.floor(t / 2.4);
          if (cyc !== S6.cyc) { S6.cyc = cyc; S6.prev = S6.ord.slice(); const sw = swaps[((cyc % swaps.length) + swaps.length) % swaps.length]; const o = S6.ord.slice(); const a = o.indexOf(sw[0]), b = o.indexOf(sw[1]); o[a] = sw[1]; o[b] = sw[0]; S6.ord = o; S6.t0 = t; }
          const k = V.ease.inOut(V.clamp((t - S6.t0) / 0.8, 0, 1));
          const xs = [];
          for (let ti = 0; ti < 7; ti++) { const p0 = S6.prev.indexOf(ti), p1 = S6.ord.indexOf(ti); xs[ti] = pad + slot * (V.lerp(p0, p1, k) + 0.5); }
          xs[7] = pad + slot * 7.5;
          const y = narrow ? 150 : 172;
          for (let ti = 0; ti < 7; ti++) {
            const x1 = xs[7], x0 = xs[ti], off = Math.min(narrow ? 170 : 230, 40 + Math.abs(x1 - x0) * 0.5);
            g.strokeStyle = V.alpha(tokCol(C, ti), 0.75); g.lineWidth = 1 + 7 * sim[ti];
            g.beginPath(); g.moveTo(x1, y); g.quadraticCurveTo((x0 + x1) / 2, y - off, x0, y); g.stroke();
          }
          for (let ti = 0; ti < 8; ti++) {
            const col = tokCol(C, ti), cx = xs[ti];
            V.box(g, cx - chipW / 2, y, chipW, chipH, { r: 8, fill: V.alpha(col, ti === 7 ? 0.35 : 0.14), stroke: V.alpha(col, ti === 7 ? 1 : 0.6), lw: ti === 7 ? 2 : 1 });
            V.text(g, toks[ti], cx, y + chipH / 2 + 0.5, { size: narrow ? 11 : 12, weight: 600, align: 'center', base: 'middle', color: C.ink, maxW: chipW - 4 });
          }
          V.text(g, 'query: “home” (fixed at the end)', xs[7] + chipW / 2, y + chipH + 16, { size: 11, align: 'right', color: C.muted });
          V.text(g, 'score = q · k: thickness follows the token, not the position', w / 2, 24, { size: 12.5, align: 'center', color: C.ink, weight: 600, maxW: w - 20 });
          const by1 = h - 46, bh0 = Math.min(90, by1 - (y + chipH + 48));
          V.text(g, 'each token\'s score with the query (same before and after every swap)', w / 2, by1 - bh0 - 12, { size: 11.5, align: 'center', color: C.muted, maxW: w - 20 });
          for (let ti = 0; ti < 7; ti++) {
            const bx = pad + slot * (ti + 0.5), bh = bh0 * sim[ti];
            g.fillStyle = V.alpha(tokCol(C, ti), 0.7); V.rr(g, bx - Math.min(16, slot / 3), by1 - bh, 2 * Math.min(16, slot / 3), bh, 3); g.fill();
            V.text(g, toks[ti], bx, by1 + 14, { size: 11, align: 'center', color: C.muted, maxW: slot - 2 });
          }
          V.text(g, 'without RoPE, swapping two tokens changes no score at all', w / 2, h - 8, { size: 12, align: 'center', color: C.b, weight: 600, maxW: w - 20 });
          return;
        }
        if (i === 1) {
          const n = 8, cs = narrow ? Math.min(19, (w - 40) / n) : Math.min(26, (h - 110) / n, (w * 0.45 - 30) / n);
          const mx0 = narrow ? (w - n * cs) / 2 : Math.max(20, w * 0.12), my0 = narrow ? 36 : 50;
          const row = Math.floor(el * 1.3) % n;
          V.text(g, 'causal mask', mx0 + (n * cs) / 2, my0 - 12, { size: 12, align: 'center', color: C.muted });
          for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) {
            const on = c <= r;
            g.fillStyle = on ? V.alpha(C.a, r === row ? 0.75 : 0.3) : V.alpha(C.muted, 0.08);
            V.rr(g, mx0 + c * cs + 1, my0 + r * cs + 1, cs - 2, cs - 2, 3); g.fill();
          }
          const bx0 = narrow ? 16 : mx0 + n * cs + 60, bx1 = w - 14, by0 = narrow ? my0 + n * cs + 44 : my0, by1 = narrow ? h - 64 : my0 + n * cs;
          V.text(g, narrow ? 'weight of each token: 1/(m+1)' : 'weight of each token in a uniform average = 1/(m+1)', (bx0 + bx1) / 2, by0 - 12, { size: 12, align: 'center', color: C.muted, maxW: bx1 - bx0 + 20 });
          const bwid = (bx1 - bx0) / n;
          for (let m = 0; m < n; m++) {
            const v = 1 / (m + 1), bh = (by1 - by0 - 18) * v;
            g.fillStyle = m === row ? C.d : V.alpha(C.d, 0.4);
            V.rr(g, bx0 + m * bwid + 3, by1 - bh, bwid - 6, bh, 3); g.fill();
            V.text(g, (narrow ? '' : 'm=') + m, bx0 + m * bwid + bwid / 2, by1 + 14, { size: 11, font: 'mono', align: 'center', color: C.muted });
            if (bwid > 30 || m === row) V.text(g, m === 0 ? '1' : '1/' + (m + 1), bx0 + m * bwid + bwid / 2, by1 - bh - 5, { size: 11, font: 'mono', align: 'center', color: m === row ? C.d : C.muted });
          }
          V.text(g, 'token ' + row + ' sees ' + (row + 1) + ' tokens: the count alone is a position signal', w / 2, h - 26, { size: 12, align: 'center', color: C.ink, weight: 600, maxW: w - 20 });
          return;
        }
        if (i === 2) {
          const y = 62; chips(y);
          const beltY = y + chipH + 30;
          V.line(g, pad, beltY, w - pad, beltY, { color: V.alpha(C.muted, 0.5), lw: 2, dash: [8, 6], dashOffset: -t * 30 });
          const pos = (el * 1.1) % 8.999, cur = Math.floor(pos);
          const sx = pad + slot * (pos + 0.5);
          const gs = Math.min(14, (w - 40) / 24), gw = 4 * gs;
          const alpha = 0.72;
          const stateOf = (ord) => { const st2 = new Array(16).fill(0); for (let j = 0; j <= cur; j++) { const wgt = Math.pow(alpha, cur - j); for (let q = 0; q < 16; q++) st2[q] += wgt * contrib[ord[j]][q]; } return st2; };
          const ord0 = [0, 1, 2, 3, 4, 5, 6, 7], ordSw = [0, 5, 2, 3, 4, 1, 6, 7];
          const st0 = stateOf(ord0), stS = stateOf(ordSw);
          const smax = Math.max(1e-6, ...st0.map(Math.abs), ...stS.map(Math.abs));
          const grid = (bx, by, st2, cs2) => { for (let q = 0; q < 16; q++) { g.fillStyle = V.cmap(st2[q] / smax, 'div'); V.rr(g, bx + (q % 4) * cs2, by + Math.floor(q / 4) * cs2, cs2 - 2, cs2 - 2, 2); g.fill(); } };
          const bx = V.clamp(sx - gw / 2, 8, w - gw - 8), byy = beltY + 10;
          grid(bx, byy, st0, gs);
          g.strokeStyle = C.d; g.lineWidth = 2; V.rr(g, bx - 3, byy - 3, gw + 4, gw + 4, 5); g.stroke();
          V.text(g, 'S', bx + gw + 10, byy + gw / 2 + 4, { size: 14, weight: 700, font: 'mono', color: C.d });
          V.arrow(g, sx, y + chipH + 4, sx, beltY - 4, { color: C.d, lw: 1.8, head: 7 });
          for (let j = 0; j <= cur; j++) { const cx = pad + slot * (j + 0.5); V.dot(g, cx, y - 8, 3 + 3 * Math.pow(alpha, cur - j), V.alpha(tokCol(C, j), 0.4 + 0.6 * Math.pow(alpha, cur - j))); }
          V.text(g, 'weight of each token in the state (dots): recent ones weigh more', w / 2, y - 20, { size: 11.5, align: 'center', color: C.muted, maxW: w - 20 });
          const cs2 = Math.min(20, (w - 80) / 10), cg = 4 * cs2, cy2 = byy + gw + 44;
          const cxA = w / 2 - cg - 30, cxB = w / 2 + 30;
          grid(cxA, cy2, st0, cs2); grid(cxB, cy2, stS, cs2);
          V.text(g, 'real order', cxA + cg / 2, cy2 - 8, { size: 11.5, align: 'center', color: C.ink, weight: 600 });
          V.text(g, '“key” ↔ “vase”', cxB + cg / 2, cy2 - 8, { size: 11.5, align: 'center', color: C.ink, weight: 600 });
          const diff = Math.sqrt(st0.reduce((sm, v, q) => sm + (v - stS[q]) * (v - stS[q]), 0));
          V.text(g, cur < 1 ? 'before “key”: identical states' : 'different states (‖ΔS‖ = ' + sf(diff, 2) + '): the order got recorded', w / 2, cy2 + cg + 20, { size: 12, align: 'center', color: cur < 1 ? C.muted : C.d, weight: 600, maxW: w - 20 });
          V.text(g, 'Sₜ = α·Sₜ₋₁ + (write of token t)   [schematic]', w / 2, h - 12, { size: 12, font: 'mono', align: 'center', color: C.ink, maxW: w - 20 });
          return;
        }
        if (i === 3) {
          const y = 80; const pos = 3 + (Math.floor(el * 1.2) % 5);
          chips(y, null, { hi: (k) => k <= pos && k > pos - 4 });
          const wts = [0.25, 0.5, 0.9, 0.4];
          const x0 = pad + slot * (pos - 3), x1 = pad + slot * (pos + 1);
          g.strokeStyle = C.d; g.lineWidth = 2; V.rr(g, x0 + 2, y - 8, x1 - x0 - 4, chipH + 16, 10); g.stroke();
          const by1 = y + chipH + 110;
          for (let k = 0; k < 4; k++) {
            const cx = pad + slot * (pos - 3 + k + 0.5), bh = 80 * wts[k];
            g.fillStyle = V.alpha(C.d, 0.75); V.rr(g, cx - 10, by1 - bh, 20, bh, 3); g.fill();
            V.text(g, ['t−3', 't−2', 't−1', 't'][k], cx, by1 + 15, { size: 11, font: 'mono', align: 'center', color: C.muted });
            V.arrow(g, cx, y + chipH + 10, cx, by1 - bh - 6, { color: V.alpha(C.d, 0.5), lw: 1.2, head: 6 });
          }
          V.text(g, 'weights of the 4 taps (illustrative)', w / 2, by1 + 34, { size: 11.5, align: 'center', color: C.muted });
          V.text(g, 'each neighbor comes in with a different weight: explicit local order', w / 2, h - 22, { size: 12, align: 'center', color: C.ink, weight: 600, maxW: w - 20 });
          return;
        }
        if (i === 4) {
          const y = 56; const blk = Math.floor(el / 2.2) % 2;
          chips(y, null, { hi: (k) => Math.floor(k / 4) === blk });
          for (let b = 0; b < 2; b++) { const bx0 = pad + slot * 4 * b + 4, bx1 = pad + slot * 4 * (b + 1) - 4; V.line(g, bx0, y + chipH + 8, bx1, y + chipH + 8, { color: b === blk ? C.d : V.alpha(C.muted, 0.5), lw: 2 }); }
          const gy = y + chipH + (narrow ? 50 : 34), cs = Math.min(22, (w - 60) / 16);
          const colsX = [];
          for (let k = 0; k < 4; k++) colsX.push(pad + slot * (4 * blk + k + 0.5));
          const ape = [[0.8, -0.3, 0.5, 0.1, -0.6], [0.2, 0.7, -0.4, 0.3, 0.1], [-0.5, 0.2, 0.6, -0.2, 0.4], [0.3, -0.6, 0.1, 0.7, -0.3]];
          for (let k = 0; k < 4; k++) {
            const cx = colsX[k];
            for (let r = 0; r < 5; r++) { g.fillStyle = V.cmap(Math.tanh(contrib[4 * blk + k][r]), 'div'); V.rr(g, cx - cs - 2, gy + r * (cs + 2), cs, cs, 2); g.fill(); }
            for (let r = 0; r < 5; r++) { g.fillStyle = V.alpha(C.d, 0.25 + 0.6 * Math.abs(ape[k][r])); V.rr(g, cx + 2, gy + r * (cs + 2), cs, cs, 2); g.fill(); }
            V.text(g, narrow ? '#' + k : 'k   ape' + k, cx, gy - 6, { size: 11, font: 'mono', align: 'center', color: C.muted });
          }
          if (narrow) V.text(g, 'key k (left) + ape bias (amber), per slot', w / 2, gy - 22, { size: 11, align: 'center', color: C.muted, maxW: w - 20 });
          const px = w / 2, py = gy + 5 * (cs + 2) + 44;
          colsX.forEach((cx) => V.arrow(g, cx, gy + 5 * (cs + 2) + 4, px + (cx - px) * 0.15, py - 8, { color: V.alpha(C.d, 0.55), lw: 1.3, head: 6 }));
          for (let r = 0; r < 5; r++) { let v = 0; for (let k = 0; k < 4; k++) v += Math.tanh(contrib[4 * blk + k][r]) * (0.25 + 0.2 * ape[k][r]); g.fillStyle = V.cmap(v, 'div'); V.rr(g, px - cs / 2 + (r - 2) * (cs + 2), py, cs, cs, 2); g.fill(); }
          // the 5 cells span px − 2.5·cs − 4 to px + 2.5·cs + 4: the frame has 3 px of slack on both sides
          g.strokeStyle = C.d; g.lineWidth = 2; V.rr(g, px - 2.5 * cs - 7, py - 3, 5 * cs + 14, cs + 6, 4); g.stroke();
          if (narrow) { V.text(g, 'softmax(gate + ape) per channel', w / 2, py + cs + 22, { size: 11.5, font: 'mono', align: 'center', color: C.ink }); V.text(g, '→ 1 key per block (132 B in the cache)', w / 2, py + cs + 38, { size: 11.5, font: 'mono', align: 'center', color: C.ink }); }
          else V.text(g, 'softmax(gate + ape) per channel → 1 key per block (132 B in the cache)', w / 2, py + cs + 22, { size: 11.5, font: 'mono', align: 'center', color: C.ink, maxW: w - 20 });
          V.text(g, 'the ape bias says which of the block\'s 4 positions each key held', w / 2, h - 18, { size: 12, align: 'center', color: C.muted, maxW: w - 20 });
          return;
        }
        if (i === 5) {
          const n = 16, np = 2, cs = Math.min(34, (w - 80) / (n + np + 1));
          const x0 = Math.max(tw(g, 'q⊙k', 12, 'mono', 700) + 16, (w - (n + np) * (cs + 2)) / 2 + 8);
          const rq = V.rng(7), qv = Array.from({ length: n }, () => rq.normal() * 0.7), kvv = Array.from({ length: n }, () => rq.normal() * 0.7);
          const rowsY = [52, 52 + cs + 26, 52 + 2 * (cs + 26)];
          const labs = ['q', 'k', 'q⊙k'];
          const fill = Math.min(n + np, Math.floor(el * 6));
          let sum = 0;
          for (let r = 0; r < 3; r++) {
            V.text(g, labs[r], x0 - 10, rowsY[r] + cs / 2 + 4, { size: 12, font: 'mono', weight: 700, align: 'right', color: C.ink });
            for (let c = 0; c < n + np; c++) {
              const isPad = c >= n, x = x0 + c * (cs + 2) + (isPad ? 6 : 0), y = rowsY[r];
              let v = isPad ? 0 : r === 0 ? qv[c] : r === 1 ? kvv[c] : qv[c] * kvv[c];
              if (r === 2 && c >= fill) { g.fillStyle = V.alpha(C.muted, 0.06); V.rr(g, x, y, cs, cs, 3); g.fill(); continue; }
              if (r === 2) sum += v;
              g.fillStyle = isPad ? V.alpha(C.muted, 0.1) : V.cmap(Math.tanh(v), 'div'); V.rr(g, x, y, cs, cs, 3); g.fill();
              if (isPad) { hatch(g, x, y, cs, cs, V.alpha(C.muted, 0.6), 4); V.text(g, '0', x + cs / 2, y + cs / 2 + 4, { size: 11, font: 'mono', align: 'center', color: C.muted }); }
            }
          }
          V.text(g, narrow ? '512 NoPE dims (16 cells)' : '512 NoPE dims (schematic: 16 cells)', x0 + (n * (cs + 2)) / 2, rowsY[0] - 10, { size: 11, align: 'center', color: C.muted });
          const padC = x0 + n * (cs + 2) + 6 + cs, padW = tw(g, '+64 zeros', 11);
          V.text(g, '+64 zeros', Math.min(padC, w - 8 - padW / 2), rowsY[0] - 10, { size: 11, align: 'center', color: C.muted });
          const exact = qv.reduce((s, x, k) => s + x * kvv[k], 0);
          const ys = rowsY[2] + cs + 26;
          V.text(g, 'Σ = ' + mfmt(sum, 3) + (fill >= n + np ? '   =   ' + mfmt(exact, 3) + ' (no pad) ✓' : ''), w / 2, ys, { size: 13, font: 'mono', weight: 700, align: 'center', color: fill >= n + np ? C.a : C.ink, maxW: w - 20 });
          const bx0 = 20, bx1 = w - 20, by = Math.min(h - 84, ys + 40), bw = bx1 - bx0;
          const w528 = (bw * 528) / 656;
          g.fillStyle = V.alpha(C.b, 0.35); V.rr(g, bx0, by, w528, 22, 5); g.fill();
          g.fillStyle = V.alpha(C.muted, 0.12); V.rr(g, bx0 + w528 + 2, by, bw - w528 - 2, 22, 5); g.fill();
          hatch(g, bx0 + w528 + 2, by, bw - w528 - 2, 22, V.alpha(C.muted, 0.6), 5);
          V.text(g, '528 B useful', bx0 + w528 / 2, by + 15, { size: 11, font: 'mono', align: 'center', color: C.ink, weight: 600 });
          V.text(g, '+128 B', bx0 + w528 + (bw - w528) / 2, by + 15, { size: 11, font: 'mono', align: 'center', color: C.ink, weight: 600 });
          V.text(g, '656 B per token per DSA layer (≈ +24% in these 11 layers)', w / 2, by + 40, { size: 11.5, align: 'center', color: C.muted, maxW: w - 20 });
          V.text(g, 'bit-exact: the softmax doesn\'t change and V comes from the NoPE region', w / 2, h - 12, { size: 12, align: 'center', color: C.a, weight: 600, maxW: w - 20 });
          return;
        }
        // i === 6: GLM vs Qwen
        const colW = w - 28;
        const blocks = [
          { name: 'GLM-5.3-Flash', col: C.glm, lay: ['KDA', 'KDA', 'KDA', 'MLA'], cnt: '34 KDA + 11 MLA', rot: 0, lab: 'MLA head: 256 dims, 0 with RoPE', note: 'NoPE in the MLA layers: a training choice that worked', ok: true },
          { name: 'Qwen3.8-Flash-Next', col: C.qwen, lay: ['GDN', 'GDN', 'GDN', 'QSA'], cnt: '36 GDN + 12 QSA', rot: 16, lab: 'QSA head: 64 RoPE + 192 NoPE', note: 'full NoPE → endless generation (tech report): keeps 25% RoPE', ok: false },
        ];
        blocks.forEach((B, bi) => {
          const bx = 14, by = 30 + bi * 178;
          V.text(g, B.name, bx, by, { size: 13, weight: 700, color: B.col });
          V.text(g, B.cnt, bx + colW, by, { size: 11.5, font: 'mono', align: 'right', color: C.muted });
          const bw3 = (colW - 18) / 4;
          B.lay.forEach((ty, k) => {
            const x = bx + k * (bw3 + 6), isA = k === 3, pulse = 0.5 + 0.5 * Math.sin(t * 2.2 + k + bi);
            V.box(g, x, by + 12, bw3, 32, { r: 8, fill: V.alpha(isA ? C.a : C.b, 0.12 + (isA ? 0.14 * pulse : 0)), stroke: V.alpha(isA ? C.a : C.b, 0.8), label: ty, size: 12, font: 'mono' });
          });
          const sy = by + 60, cw = colW / 64;
          for (let c = 0; c < 64; c++) {
            const isR = c < B.rot;
            const a = isR ? 0.5 + 0.45 * (0.5 + 0.5 * Math.sin(t * 3.2 - c * 0.6)) : 0.3;
            g.fillStyle = V.alpha(isR ? C.a : C.b, a); V.rr(g, bx + c * cw + 0.5, sy, Math.max(1, cw - 1), 20, 2); g.fill();
          }
          V.text(g, B.lab, bx, sy + 36, { size: 11.5, font: 'mono', color: C.ink, maxW: colW });
          V.text(g, (B.ok ? '✓ ' : '✗ ') + B.note, bx, sy + 56, { size: 12, weight: 600, color: B.ok ? C.a : C.c, maxW: colW });
        });
        V.text(g, narrow ? 'NoPE is a training decision, not a server one' : 'same hybrid idea, different conclusions: NoPE is a training decision, not a server one', w / 2, h - 12, { size: 12, align: 'center', color: C.ink, weight: 600, maxW: w - 20 });
      });
    }

    /* =============================================================
       Fig 7 — mRoPE: (t, h, w)
       ============================================================= */
    {
      const fig = root.querySelector('#fig-mrope');
      const S = { mode: 'img', inter: true, off: 0, sel: -1, ang: null, flash: new Array(32).fill(-10), moved: null };
      const th = ropeInv(1e7, 64);
      function build(mode) {
        const T = [];
        const txt = (labs, s) => labs.forEach((l, i) => T.push({ k: 'txt', t: s + i, h: s + i, w: s + i, lab: l }));
        if (mode === 'txt') txt(['Describe', 'the', 'scene', 'in', 'one', 'short', 'line', '.'], 0);
        else if (mode === 'img') {
          txt(['See', 'this', 'photo'], 0);
          const s = 3;
          for (let r = 0; r < 3; r++) for (let c = 0; c < 4; c++) T.push({ k: 'img', t: s, h: s + r, w: s + c, r, c });
          txt(['What', 'is', 'there?'], s + 4);
        } else {
          /* video in the Qwen2-VL index scheme (illustrative): t = start + frame, h and w restart from the same start in each frame */
          txt(['Summarize', ':'], 0);
          const s = 2;
          for (let f = 0; f < 2; f++) for (let r = 0; r < 2; r++) for (let c = 0; c < 3; c++) T.push({ k: 'vid', t: s + f, h: s + r, w: s + c, f, r, c });
          txt(['Done', '.'], s + 3);
        }
        return T;
      }
      const axisOf = (i) => (S.inter ? (i % 3 === 1 ? 1 : i % 3 === 2 ? 2 : 0) : i < 11 ? 0 : i < 22 ? 1 : 2);
      const wrapA = (a) => { a = ((a % TAU) + TAU) % TAU; return a > Math.PI ? a - TAU : a; };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'input', options: [{ v: 'txt', label: 'text' }, { v: 'img', label: 'image 3×4' }, { v: 'vid', label: 'video 2 frames' }], value: 'img', onChange: (v) => { S.mode = v; S.sel = -1; upd(); } });
      UI.seg(ctl, { label: 'tokens before the prompt', options: [{ v: 0, label: '0' }, { v: 1000, label: '1,000' }, { v: 100000, label: '100,000' }], value: 0, onChange: (v) => { S.off = v; } });
      UI.toggle(ctl, { label: 'interleaved (mrope_interleaved)', value: true, onChange: (v) => { S.inter = v; } });
      const rPos = UI.readout(ctl, 'positions used', '—'), rTok = UI.readout(ctl, 'tokens', '—');
      let TOK = build(S.mode);
      function upd() { TOK = build(S.mode); let mx = 0; TOK.forEach((o) => { mx = Math.max(mx, o.t, o.h, o.w); }); rPos.set(String(mx + 1)); rTok.set(String(TOK.length)); }
      upd();
      fitH(fig, 450, 500, 560);
      const H = V.fig(fig, (g, w, h, t, dt) => {
        if (w < 60 || h < 60) return; // transient frame before layout
        const C = V.C();
        const AX = [C.a, C.b, C.c], AXN = ['T', 'H', 'W'];
        const narrow = w < 560;
        const rects = [];
        const notes = [];
        if (S.mode === 'vid') notes.push(narrow ? 'video: Qwen2-VL indices (illustrative)' : 'video: indices in the Qwen2-VL scheme (illustrative)');
        if (S.off) notes.push('cells: position − ' + nf(S.off) + ' (earlier text)');
        if (notes.length) {
          if (narrow) notes.forEach((s, k) => V.text(g, s, 12, 15 + 14 * k, { size: 11, color: C.muted, maxW: w - 24 }));
          else V.text(g, notes.join('  ·  '), 12, 15, { size: 11, color: C.muted, maxW: w - 24 });
        }
        const txtW = Math.min(76, (w - 30) / (S.mode === 'txt' ? (narrow ? 4 : 8) : 4) - 6), txtH = 38;
        const txts = TOK.filter((o) => o.k === 'txt');
        const firstVis = TOK.findIndex((o) => o.k !== 'txt');
        const before = firstVis < 0 ? txts : TOK.slice(0, firstVis);
        const after = firstVis < 0 ? [] : TOK.filter((o, i) => o.k === 'txt' && i > firstVis);
        let y = 30 + (narrow && notes.length > 1 ? 14 : 0);
        const placeRow = (arr) => {
          const perRow = S.mode === 'txt' ? (narrow ? 4 : 8) : arr.length;
          arr.forEach((o, i) => {
            const rr = Math.floor(i / perRow), cc = i % perRow, n = Math.min(perRow, arr.length);
            const x = w / 2 + (cc - n / 2) * (txtW + 6) + 3;
            rects.push({ o, x, y: y + rr * (txtH + 8), w: txtW, h: txtH });
          });
          y += Math.ceil(arr.length / perRow) * (txtH + 8);
        };
        placeRow(before);
        if (S.mode === 'img') {
          const tw2 = Math.min(64, (w - 40) / 4 - 6), th2 = 34;
          y += 6;
          TOK.filter((o) => o.k === 'img').forEach((o) => rects.push({ o, x: w / 2 + (o.c - 2) * (tw2 + 5) + 2, y: y + o.r * (th2 + 5), w: tw2, h: th2 }));
          y += 3 * (th2 + 5) + 10;
        } else if (S.mode === 'vid') {
          const tw2 = Math.min(56, (w - 60) / 6 - 5), th2 = 34, fw = 3 * (tw2 + 4);
          y += 6;
          TOK.filter((o) => o.k === 'vid').forEach((o) => { const fx = w / 2 + (o.f === 0 ? -fw - 10 : 10); rects.push({ o, x: fx + o.c * (tw2 + 4), y: y + o.r * (th2 + 5) + o.f * 6, w: tw2, h: th2 }); });
          y += 2 * (th2 + 5) + 18;
        }
        placeRow(after);
        let sel = -1;
        const p = H && H.pointer;
        if (p && p.inside) rects.forEach((r, k) => { if (p.x >= r.x && p.x <= r.x + r.w && p.y >= r.y && p.y <= r.y + r.h) sel = k; });
        if (sel < 0) sel = Math.floor(t / 1.5) % rects.length;
        const so = rects[sel].o;
        const absP = [S.off + so.t, S.off + so.h, S.off + so.w];
        const tgt = Array.from({ length: 32 }, (_, i) => absP[axisOf(i)] * th[i]);
        if (!S.ang || S.ang.length !== 32) S.ang = tgt.slice();
        if (sel !== S.sel) {
          // which pairs rotated more than 1° relative to the previous token (independent of the offset)
          if (S.sel >= 0 && S.moved) { const cnt = [0, 0, 0]; for (let i = 0; i < 32; i++) { if (Math.abs(wrapA(tgt[i] - S.moved[i])) > Math.PI / 180) { S.flash[i] = t; cnt[axisOf(i)]++; } } S.cnt = cnt; }
          S.sel = sel;
        }
        S.moved = tgt.slice();
        const kk = Math.min(1, (dt || 0.016) * 5);
        for (let i = 0; i < 32; i++) S.ang[i] += wrapA(tgt[i] - S.ang[i]) * kk;
        rects.forEach((r, k) => {
          const o = r.o, isSel = k === sel;
          const base = o.k === 'txt' ? C.muted : o.k === 'img' ? C.e : C.f;
          const sameH = o.k !== 'txt' && so.k !== 'txt' && o.h === so.h && !isSel, sameW = o.k !== 'txt' && so.k !== 'txt' && o.w === so.w && !isSel;
          V.box(g, r.x, r.y, r.w, r.h, { r: 7, fill: V.alpha(base, isSel ? 0.38 : 0.14), stroke: isSel ? C.d : sameH ? C.b : sameW ? C.c : V.alpha(base, 0.6), lw: isSel || sameH || sameW ? 2 : 1 });
          const trip = o.t + ',' + o.h + ',' + o.w; // relative to the start of the prompt
          if (o.k === 'txt') {
            V.text(g, o.lab, r.x + r.w / 2, r.y + 14, { size: 11.5, weight: 600, align: 'center', color: C.ink, maxW: r.w - 4 });
            V.text(g, trip, r.x + r.w / 2, r.y + 30, { size: 11, font: 'mono', align: 'center', color: C.muted, maxW: r.w - 4 });
          } else V.text(g, trip, r.x + r.w / 2, r.y + r.h / 2 + 4, { size: 11, font: 'mono', align: 'center', color: C.ink, maxW: r.w - 2 });
        });
        if (S.mode === 'vid') {
          const vr = rects.filter((r) => r.o.k === 'vid');
          [0, 1].forEach((f) => { const fr = vr.filter((r) => r.o.f === f); if (fr.length) V.text(g, 'frame ' + f, fr[0].x, fr[0].y - 5, { size: 11, color: C.f }); });
        }
        if (S.mode === 'txt') {
          // text: the three coordinates move together (t = h = w = m)
          const nT = TOK.length, cwT = Math.min(64, (w - 60) / nT), tx0 = (w - nT * cwT) / 2 + 12, ty0 = y + 18, rhT = 24;
          V.text(g, 'the three coordinates of each text token', w / 2, ty0 - 6, { size: 11.5, align: 'center', color: C.muted, maxW: w - 20 });
          for (let a = 0; a < 3; a++) {
            const yy = ty0 + a * (rhT + 4);
            V.text(g, AXN[a], tx0 - 10, yy + rhT / 2 + 4, { size: 12, font: 'mono', weight: 700, align: 'right', color: AX[a] });
            TOK.forEach((o, j) => {
              const isS = rects[sel] && rects[sel].o === o;
              g.fillStyle = V.alpha(AX[a], isS ? 0.42 : 0.14); V.rr(g, tx0 + j * cwT + 1, yy, cwT - 2, rhT, 5); g.fill();
              if (isS) { g.strokeStyle = C.d; g.lineWidth = 1.5; V.rr(g, tx0 + j * cwT + 1, yy, cwT - 2, rhT, 5); g.stroke(); }
              V.text(g, String([o.t, o.h, o.w][a]), tx0 + j * cwT + cwT / 2, yy + rhT / 2 + 4, { size: 11, font: 'mono', align: 'center', color: C.ink, maxW: cwT - 4 });
            });
          }
          const ny = ty0 + 3 * (rhT + 4) + 18;
          if (narrow) { V.text(g, 't = h = w = m for every text token:', w / 2, ny, { size: 11.5, align: 'center', color: C.ink, weight: 600 }); V.text(g, 'all 32 pairs rotate by m·θᵢ (plain 1D RoPE)', w / 2, ny + 16, { size: 11.5, align: 'center', color: C.ink, weight: 600 }); }
          else V.text(g, 't = h = w = m for every text token: all 32 pairs rotate by m·θᵢ, and mRoPE becomes plain 1D RoPE', w / 2, ny, { size: 12, align: 'center', color: C.ink, weight: 600, maxW: w - 20 });
        }
        // strip of the 32 pairs: 1 row on desktop, 2 rows of 16 on phones (bigger hands)
        const rowsN = narrow ? 2 : 1, perR = 32 / rowsN, sch = narrow ? 26 : 30, stripH = rowsN * sch + (rowsN - 1) * 18;
        const sy = h - 74 - stripH, sx0 = 16, sx1 = w - 16, scw = (sx1 - sx0) / perR;
        const desc = so.k === 'txt' ? 'text “' + so.lab + '”' : so.k === 'img' ? 'patch row ' + so.r + ', column ' + so.c : 'frame ' + so.f + ', row ' + so.r + ', column ' + so.c;
        V.text(g, desc + ' → (t, h, w) = (' + absP.map((v) => nf(v)).join(', ') + ')', w / 2, sy - 14, { size: 12.5, weight: 700, align: 'center', color: C.d, maxW: w - 20 });
        for (let i = 0; i < 32; i++) {
          const ax = axisOf(i), rw = Math.floor(i / perR), ci = i % perR;
          const x = sx0 + ci * scw, ry = sy + rw * (sch + 18);
          g.fillStyle = V.alpha(AX[ax], 0.3); V.rr(g, x + 0.5, ry, scw - 1, sch, 3); g.fill();
          const fl = V.clamp(1 - (t - S.flash[i]) / 1.4, 0, 1);
          if (fl > 0) { g.strokeStyle = V.alpha(C.d, fl); g.lineWidth = 2; V.rr(g, x + 0.5, ry, scw - 1, sch, 3); g.stroke(); }
          const cx = x + scw / 2, cy = ry + sch / 2, rr = Math.min(scw / 2 - 1.5, sch / 2 - 2, 11);
          const a = S.ang[i];
          V.line(g, cx, cy, cx + rr * Math.sin(a), cy - rr * Math.cos(a), { color: fl > 0 ? C.d : AX[ax], lw: 1.8 });
          V.dot(g, cx, cy, 1.4, C.ink);
        }
        (narrow ? [0, 8, 15, 16, 24, 31] : [0, 8, 16, 24, 31]).forEach((i) => { const rw = Math.floor(i / perR); V.text(g, String(i), sx0 + (i % perR) * scw + scw / 2, sy + rw * (sch + 18) + sch + 13, { size: 11, font: 'mono', align: 'center', color: C.muted }); });
        let lx = sx0;
        AXN.forEach((n, k) => { g.fillStyle = V.alpha(AX[k], 0.6); V.rr(g, lx, sy + stripH + 24, 12, 12, 3); g.fill(); const lab = narrow ? n + ': ' + ['t', 'h', 'w'][k] : n + ' uses ' + ['t', 'h', 'w'][k]; V.text(g, lab, lx + 16, sy + stripH + 34, { size: 11.5, font: 'mono', color: C.ink }); lx += tw(g, lab, 11.5, 'mono', 400) + 30; });
        const low = sy - 32 < y;
        V.text(g, S.inter ? 'pairs: T 0,3,6… · H 1,4,7… · W 2,5,8…' : 'pairs: T 0–10 · H 11–21 · W 22–31', sx1, low ? sy + stripH + 34 : sy - 32, { size: 11, font: 'mono', align: 'right', color: C.muted, maxW: low ? sx1 - lx - 6 : w - 20 });
        if (S.cnt) {
          const cs = 'T ' + S.cnt[0] + ' · H ' + S.cnt[1] + ' · W ' + S.cnt[2];
          V.text(g, (narrow ? 'rotated >1° vs. previous token: ' : 'amber = pairs that rotated more than 1° since the previous token: ') + cs, w / 2, h - 12, { size: 11.5, align: 'center', color: C.d, weight: 600, maxW: w - 20 });
        }
      });
    }

    /* =============================================================
       Fig 8 — frequency map: PI, NTK and YaRN
       ============================================================= */
    {
      const fig = root.querySelector('#fig-yarn');
      const PRE = { qwen: { base: 1e7, L: 262144, s: 4, name: 'Qwen3.8 (θ=10⁷, L=262,144)' }, ds: { base: 160000, L: 65536, s: 16, name: 'DeepSeek compressor layers (θ=1.6·10⁵, L=65,536)' } };
      const S = { p: 'qwen', meth: 'yarn', s: 4, from: null, t0: -10, cur: null, ms: true, fcur: 1 };
      const LG = [2.2, 1.6, 1.3, 1.0, 0.8, 0.5, 0.3, 0.0]; // illustrative logits for the temperature panel
      const d = 64;
      function target() {
        const P = PRE[S.p], o = ropeInv(P.base, d);
        if (S.meth === 'none') return o;
        if (S.meth === 'pi') return o.map((x) => x / S.s);
        if (S.meth === 'ntk') return ropeInv(P.base * Math.pow(S.s, d / (d - 2)), d);
        return yarnInv(P.base, d, P.L, S.s).inv;
      }
      let H = null;
      function change() { S.from = S.cur ? S.cur.slice() : target(); S.t0 = H ? H.t : 0; upd(); }
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'scenario', options: [{ v: 'qwen', label: 'Qwen ×4' }, { v: 'ds', label: 'DeepSeek ×16' }], value: 'qwen', onChange: (v) => { S.p = v; sS.set(PRE[v].s); if (tgMs) tgMs.set(v === 'qwen'); change(); } });
      UI.seg(ctl, { label: 'method', options: [{ v: 'none', label: 'none' }, { v: 'pi', label: 'PI' }, { v: 'ntk', label: 'NTK' }, { v: 'yarn', label: 'YaRN' }], value: 'yarn', onChange: (v) => { S.meth = v; change(); } });
      const sS = UI.slider(ctl, { label: 'factor s', min: 1, max: 32, step: 1, value: 4, fmt: (v) => '×' + v, onInput: (v) => { S.s = v; change(); } });
      /* YaRN mscale: multiplies cos and sin, so the logits come out × mscale². Qwen uses it (HF default); DS zeroes it. */
      const tgMs = UI.toggle(ctl, { label: 'mscale (temperature)', value: true, onChange: (v) => { S.ms = v; upd(); } });
      const rK = UI.readout(ctl, 'intact · ramp · interpolated', '—'), rM = UI.readout(ctl, 'mscale', '—');
      function upd() {
        const P = PRE[S.p], y = yarnInv(P.base, d, P.L, 2);
        const c = { K: 0, R: 0, I: 0 }; y.kind.forEach((k) => { c[k]++; });
        rK.set(c.K + ' · ' + c.R + ' · ' + c.I);
        const ms = mscaleOf(S.s);
        rM.set(S.meth !== 'yarn' ? '—' : !S.ms ? (S.p === 'ds' ? '1 (mscale=0 in DS)' : '1 (off)') : sf(ms, 3) + ' · logits ×' + sf(ms * ms, 3) + (S.p === 'ds' ? ' (DS doesn\'t use it)' : ''));
      }
      upd();
      fitH(fig, 480, 540, 560);
      H = V.fig(fig, (g, w, h, t) => {
        if (w < 60 || h < 60) return; // transient frame before layout
        const C = V.C();
        const P = PRE[S.p];
        const orig = ropeInv(P.base, d), tgt = target();
        const k = V.ease.inOut(V.clamp((t - S.t0) / 0.6, 0, 1));
        const from = S.from || tgt;
        const cur = tgt.map((x, i) => Math.exp(V.lerp(Math.log(from[i]), Math.log(x), k)));
        S.cur = cur;
        const zones = yarnInv(P.base, d, P.L, 2).kind;
        const narrow = w < 560;
        const x0 = 48, x1 = w - 14, y0 = 30, y1 = Math.round(h * (narrow ? 0.54 : 0.64));
        const bw = (x1 - x0) / 32;
        const ly = (l) => V.lerp(y1, y0, V.clamp((l - 0.5) / 8.5, 0, 1));
        V.text(g, P.name, x0, 16, { size: 12.5, weight: 700, color: S.p === 'qwen' ? C.qwen : C.ds, maxW: w - 60 });
        const ZC = { K: C.a, R: C.d, I: C.b };
        for (let i = 0; i < 32; i++) { g.fillStyle = V.alpha(ZC[zones[i]], 0.1); g.fillRect(x0 + i * bw, y0, bw, y1 - y0); }
        let zi = 0;
        while (zi < 32) { let zj = zi; while (zj < 32 && zones[zj] === zones[zi]) zj++; const lab = { K: 'intact', R: 'ramp', I: 'interpolated' }[zones[zi]]; if ((zj - zi) * bw > tw(g, lab, 11, 'sans', 600) + 6) V.text(g, lab, x0 + ((zi + zj) / 2) * bw, y0 + 14, { size: 11, weight: 600, align: 'center', color: ZC[zones[zi]] }); zi = zj; }
        [1, 3, 5, 7, 9].forEach((l) => { V.line(g, x0, ly(l), x1, ly(l), { color: V.alpha(C.muted, 0.12), lw: 1 }); V.text(g, '10' + ['', '¹', '', '³', '', '⁵', '', '⁷', '', '⁹'][l], x0 - 6, ly(l) + 4, { size: 11, font: 'mono', align: 'right', color: C.muted }); });
        if (!narrow) V.text(g, 'λ in tokens (log scale)', x1 - 6, y1 - 8, { size: 11, align: 'right', color: C.muted });
        const yL = ly(Math.log10(P.L)), yT = ly(Math.log10(P.L * S.s));
        V.line(g, x0, yL, x1, yL, { color: C.muted, lw: 1.2, dash: [5, 4] });
        V.text(g, 'L', x1 - 2, yL - 4, { size: 11, font: 'mono', weight: 700, align: 'right', color: C.muted });
        if (S.s > 1) { V.line(g, x0, yT, x1, yT, { color: C.c, lw: 1.2, dash: [3, 4] }); V.text(g, 's·L', x1 - 2, yT - 4, { size: 11, font: 'mono', weight: 700, align: 'right', color: C.c }); }
        const MC = { none: C.muted, pi: C.b, ntk: C.d, yarn: C.a }[S.meth];
        g.beginPath(); cur.forEach((x, i) => { const X = x0 + (i + 0.5) * bw, Y = ly(Math.log10(TAU / x)); if (i) g.lineTo(X, Y); else g.moveTo(X, Y); }); g.strokeStyle = V.alpha(MC, 0.8); g.lineWidth = 1.8; g.stroke();
        orig.forEach((x, i) => V.dot(g, x0 + (i + 0.5) * bw, ly(Math.log10(TAU / x)), Math.min(3.2, bw / 3), V.alpha(C.muted, 0.7)));
        cur.forEach((x, i) => V.dot(g, x0 + (i + 0.5) * bw, ly(Math.log10(TAU / x)), Math.min(4.2, bw / 2.4), MC));
        [0, 8, 16, 24, 31].forEach((i) => V.text(g, String(i), x0 + (i + 0.5) * bw, y1 + 15, { size: 11, font: 'mono', align: 'center', color: C.muted }));
        // bottom panels
        const py = y1 + 38, ph = h - py - 12;
        const half = narrow ? w - 24 : (w - 36) / 2;
        const ratio = (P.L * S.s * cur[31]) / (P.L * orig[31]);
        const gx0 = 12, gy = py + 26, gw = half - 8;
        const gmax = Math.max(4, ratio * 1.15);
        V.text(g, 'pair 31 at position s·L ÷ largest angle seen in training', gx0, py + 8, { size: 11.5, color: C.muted, maxW: gw });
        g.fillStyle = V.alpha(C.a, 0.22); V.rr(g, gx0, gy, gw * (1 / gmax), 20, 5); g.fill();
        g.fillStyle = V.alpha(C.c, 0.1); V.rr(g, gx0 + gw / gmax, gy, gw - gw / gmax, 20, 5); g.fill();
        V.text(g, 'seen', gx0 + (gw / gmax) / 2, gy + 14, { size: 11, align: 'center', color: C.ink, weight: 600 });
        const mxp = gx0 + gw * V.clamp(ratio / gmax, 0, 1);
        const okR = ratio <= 1.0001;
        V.line(g, mxp, gy - 6, mxp, gy + 26, { color: okR ? C.a : C.c, lw: 3 });
        V.glow(g, mxp, gy + 10, 12, okR ? C.a : C.c, 0.7);
        V.text(g, sf(ratio, 2) + '× ' + (okR ? '✓ within training' : '✗ extrapolates'), gx0, gy + 42, { size: 12, font: 'mono', weight: 700, color: okR ? C.a : C.c });
        const bx0 = narrow ? 12 : 24 + half, by = narrow ? gy + 62 : py + 8;
        const fT = S.ms ? Math.pow(mscaleOf(S.s), 2) : 1;
        S.fcur += (fT - S.fcur) * 0.12;
        if (Math.abs(fT - S.fcur) < 5e-4) S.fcur = fT;
        if (S.meth === 'yarn' && (!narrow || h - by > 60)) {
          // in YaRN pair 0 stays intact; what changes on the fast side is the temperature: logits × mscale²
          const pRef = V.softmax(LG), pNow = V.softmax(LG.map((x) => x * S.fcur));
          V.text(g, narrow ? 'temperature: softmax of 8 logits (illustr.)' : 'temperature (mscale): softmax of 8 illustrative logits', bx0, by, { size: 11.5, color: C.muted, maxW: half });
          const bw3 = half - 8, nb = 8, gp = 4, bwid = (bw3 - gp * (nb - 1)) / nb;
          const bTop = by + 12, bBot = narrow ? h - 28 : Math.min(h - 30, by + 74), bh = bBot - bTop, pmax = 0.55;
          for (let k2 = 0; k2 < nb; k2++) {
            const bx = bx0 + k2 * (bwid + gp), hN = (bh * pNow[k2]) / pmax, hR = (bh * pRef[k2]) / pmax;
            g.fillStyle = V.alpha(S.ms ? C.a : C.muted, 0.7); V.rr(g, bx, bBot - hN, bwid, hN, 3); g.fill();
            g.strokeStyle = V.alpha(C.ink, 0.55); g.lineWidth = 1; g.setLineDash([3, 3]); g.strokeRect(bx + 0.5, bBot - hR, bwid - 1, hR); g.setLineDash([]);
          }
          V.text(g, 'logits ×' + sf(S.fcur, 3) + ' · max p ' + sf(pRef[0], 2) + ' → ' + sf(pNow[0], 2), bx0, bBot + 16, { size: 12, font: 'mono', weight: 700, color: S.ms ? C.a : C.muted, maxW: bw3 });
        } else if (S.meth !== 'yarn' && (!narrow || h - by > 40)) {
          const deg0 = (orig[0] * 180) / Math.PI, degc = (cur[0] * 180) / Math.PI;
          V.text(g, 'pair 0: rotation between neighbors (local resolution)', bx0, by, { size: 11.5, color: C.muted, maxW: half });
          const bw2 = half - 8, yy = by + 18;
          g.fillStyle = V.alpha(C.muted, 0.15); V.rr(g, bx0, yy, bw2, 16, 4); g.fill();
          g.fillStyle = V.alpha(degc < deg0 - 0.01 ? C.c : C.a, 0.7); V.rr(g, bx0, yy, Math.max(3, (bw2 * degc) / deg0), 16, 4); g.fill();
          V.text(g, sf(degc, 1) + '°/token (original ' + sf(deg0, 1) + '°)', bx0, yy + 32, { size: 12, font: 'mono', weight: 700, color: degc < deg0 - 0.01 ? C.c : C.a, maxW: bw2 });
        }
      });
    }

    /* =============================================================
       Fig 9 — DeepSeek: YaRN per layer type (stock vs fix)
       ============================================================= */
    {
      const fig = root.querySelector('#fig-camadas');
      const TYPES = Array.from({ length: 46 }, (_, i) => (i < 2 ? 'swa' : i <= 42 ? (i % 2 === 0 ? 'c4' : 'c128') : 'draft'));
      const S = { fix: false, sel: 'swa' };
      const plainS = ropeInv(1e4, 64), plainC = ropeInv(1.6e5, 64);
      const yarnS = yarnInv(1e4, 64, 65536, 16).inv, yarnC = yarnInv(1.6e5, 64, 65536, 16).inv;
      const hasYarn = (ty) => (S.fix ? ty === 'c4' || ty === 'c128' : true);
      const ctl = UI.controls(fig);
      UI.toggle(ctl, { label: 'RoPE SWA fix (DSPARK_ENABLE_ROPE_SWA_FIX=1)', value: false, onChange: (v) => { S.fix = v; upd(); } });
      const segT = UI.seg(ctl, { label: 'type', options: [{ v: 'swa', label: 'SWA 0–1' }, { v: 'c4', label: 'C4A' }, { v: 'c128', label: 'C128A' }, { v: 'draft', label: 'DSpark 43–45' }], value: 'swa', onChange: (v) => { S.sel = v; upd(); } });
      const rY = UI.readout(ctl, 'layers with YaRN ×16', '—'), rD = UI.readout(ctl, 'largest deviation in window', '—');
      function upd() {
        rY.set(String(TYPES.filter(hasYarn).length) + ' of 46');
        let mx = 0;
        if (hasYarn('swa')) for (let i = 0; i < 32; i++) mx = Math.max(mx, Math.abs(127 * (plainS[i] - yarnS[i])));
        rD.set(sf((mx * 180) / Math.PI, 2) + '°');
      }
      upd();
      const TC = (C) => ({ swa: C.b, c4: C.a, c128: C.f, draft: C.b });
      const H = V.fig(fig, (g, w, h, t) => {
        if (w < 60 || h < 60) return; // transient frame before layout
        const C = V.C();
        const col = TC(C);
        const narrow = w < 560;
        const x0 = 14, x1 = w - 14, cw = (x1 - x0) / 46, y0 = 40, ch = 34;
        const p = H && H.pointer;
        if (p && p.clicked && p.y >= y0 - 14 && p.y <= y0 + ch + 4) { const k = Math.floor((p.x - x0) / cw); if (k >= 0 && k < 46 && TYPES[k] !== S.sel) { segT.set(TYPES[k]); } }
        V.text(g, 'DeepSeek-V4-Flash layers 0 → 45 (43 in the trunk + 3 in DSpark)', x0, 16, { size: 12, color: C.muted, maxW: w - 28 });
        for (let i = 0; i < 46; i++) {
          const ty = TYPES[i], x = x0 + i * cw, on = ty === S.sel;
          g.fillStyle = V.alpha(col[ty], on ? 0.75 : 0.35); V.rr(g, x + 0.5, y0, Math.max(1, cw - 1), ch, Math.min(3, cw / 3)); g.fill();
          if (ty === 'draft') hatch(g, x + 0.5, y0, Math.max(1, cw - 1), ch, V.alpha(C.ink, 0.45), 5);
          if (on) { g.strokeStyle = C.ink; g.lineWidth = 1.2; V.rr(g, x + 0.5, y0, Math.max(1, cw - 1), ch, Math.min(3, cw / 3)); g.stroke(); }
          if (hasYarn(ty)) {
            const pulse = 0.65 + 0.35 * Math.sin(t * 3 + i * 0.3);
            g.fillStyle = V.alpha(C.d, pulse); V.rr(g, x + 1, y0 - 9, Math.max(1, cw - 2), 6, 2); g.fill();
            if (cw >= 14) V.text(g, 'Y', x + cw / 2, y0 - 12, { size: 10.5, font: 'mono', weight: 700, align: 'center', color: C.d });
          }
        }
        if (cw >= 14) [0, 2, 3, 42, 43, 45].forEach((i) => V.text(g, String(i), x0 + (i + 0.5) * cw, y0 + ch + 13, { size: 10.5, font: 'mono', align: 'center', color: C.muted }));
        let lx = x0; const ly0 = y0 + ch + (cw >= 14 ? 30 : 18);
        [['swa', 'SWA'], ['c4', 'C4A'], ['c128', 'C128A'], ['draft', 'DSpark']].forEach(([k, l]) => { g.fillStyle = V.alpha(col[k], 0.7); V.rr(g, lx, ly0 - 9, 11, 11, 3); g.fill(); if (k === 'draft') hatch(g, lx, ly0 - 9, 11, 11, V.alpha(C.ink, 0.5), 4); V.text(g, l, lx + 15, ly0, { size: 11.5, color: C.ink }); lx += tw(g, l, 11.5) + 32; });
        g.fillStyle = V.alpha(C.d, 0.9); V.rr(g, lx, ly0 - 6, 14, 5, 2); g.fill(); V.text(g, 'YaRN ×16', lx + 19, ly0, { size: 11.5, color: C.d, weight: 600 });
        // plot of angles per pair
        const ty = S.sel, comp = ty === 'c4' || ty === 'c128';
        const pl = comp ? plainC : plainS, sv = hasYarn(ty) ? (comp ? yarnC : yarnS) : pl;
        const Dmax = comp ? 1048575 : 127, Dtr = comp ? 65535 : 127;
        const gy0 = ly0 + 34, gy1 = h - 48, gx0 = 46, gx1 = w - 14, bw = (gx1 - gx0) / 32;
        const lmin = -4, lmax = 7;
        const yy = (a) => V.lerp(gy1, gy0, V.clamp((Math.log10(Math.max(a, 1e-5)) - lmin) / (lmax - lmin), 0, 1));
        const title = comp ? 'angle of each pair at distance 1,048,575 (the whole context)' : 'angle of each pair at distance 127 (window edge)';
        V.text(g, title, gx0, gy0 - 10, { size: 11.5, color: C.muted, maxW: w - 60 });
        [-4, -2, 0, 2, 4, 6].forEach((l) => { V.line(g, gx0, yy(Math.pow(10, l)), gx1, yy(Math.pow(10, l)), { color: V.alpha(C.muted, 0.12), lw: 1 }); V.text(g, '10' + { '-4': '⁻⁴', '-2': '⁻²', 0: '⁰', 2: '²', 4: '⁴', 6: '⁶' }[l], gx0 - 6, yy(Math.pow(10, l)) + 4, { size: 11, font: 'mono', align: 'right', color: C.muted }); });
        V.line(g, gx0, yy(TAU), gx1, yy(TAU), { color: V.alpha(C.ink, 0.5), lw: 1, dash: [4, 4] });
        V.text(g, '1 turn', gx1 - 2, yy(TAU) - 4, { size: 11, align: 'right', color: C.ink });
        for (let i = 0; i < 32; i++) {
          const aS = Dmax * sv[i], aP = Dmax * pl[i], aTr = Dtr * pl[i];
          const unseen = aS > aTr * 1.0001 && aTr < TAU;
          const moved = Math.abs(sv[i] - pl[i]) > 1e-12;
          const x = gx0 + i * bw;
          g.fillStyle = unseen ? C.c : moved ? V.alpha(C.d, 0.85) : V.alpha(C.a, 0.75);
          V.rr(g, x + 1.5, yy(aS), Math.max(1, bw - 3), gy1 - yy(aS), Math.min(3, bw / 3)); g.fill();
          if (moved) V.line(g, x + 1, yy(aP), x + bw - 1, yy(aP), { color: C.ink, lw: 1.5 });
          if (aTr < TAU) V.line(g, x + 1, yy(aTr), x + bw - 1, yy(aTr), { color: C.b, lw: 2, dash: [2, 2] });
        }
        [0, 8, 16, 24, 31].forEach((i) => V.text(g, String(i), gx0 + (i + 0.5) * bw, gy1 + 14, { size: 11, font: 'mono', align: 'center', color: C.muted }));
        const msg = comp ? (hasYarn(ty) ? 'with YaRN, the slow pairs return to the arc seen in training (lavender dashes)' : 'without YaRN, the slow pairs would enter never-seen angles')
          : hasYarn(ty) ? (narrow ? 'YaRN on a window: ~2° off, nothing to fix' : 'YaRN applied to a 128-token window: up to ~2° off (tick = pure RoPE), nothing to fix') : 'pure RoPE (θ = 10⁴): exactly the reference regime';
        V.text(g, msg, w / 2, h - 10, { size: 11.5, align: 'center', color: hasYarn(ty) && !comp ? C.d : C.ink, weight: 600, maxW: w - 16 });
      });
    }

    /* =============================================================
       Fig 10 — the override through the Qwen config
       ============================================================= */
    {
      const fig = root.querySelector('#fig-config');
      const S = { nested: false, full: false, len: 1000000, yarn: true };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'override', options: [{ v: false, label: 'top level (old)' }, { v: true, label: 'nested (current)' }], value: false, onChange: (v) => { S.nested = v; upd(); } });
      UI.seg(ctl, { label: 'dict', options: [{ v: false, label: 'partial (start.sh)' }, { v: true, label: 'complete (card)' }], value: false, onChange: (v) => { S.full = v; upd(); } });
      UI.seg(ctl, { label: 'MAX_MODEL_LEN', options: [{ v: 131072, label: '131,072' }, { v: 262144, label: '262,144' }, { v: 524288, label: '524,288' }, { v: 1000000, label: '1,000,000' }], value: 1000000, onChange: (v) => { S.len = v; upd(); } });
      UI.toggle(ctl, { label: 'YARN_ENABLE=true', value: true, onChange: (v) => { S.yarn = v; upd(); } });
      const rT = UI.readout(ctl, 'text_config.rope_type', '—'), rF = UI.readout(ctl, 'VLLM_ALLOW_LONG…', '—');
      const eff = () => S.yarn && S.len > 262144;
      const lands = () => eff() && S.nested;
      function upd() { rT.set(lands() ? 'yarn' : 'default'); rF.set(S.len > 262144 ? 'required' : 'not needed'); }
      upd();
      V.fig(fig, (g, w, h, t) => {
        if (w < 60 || h < 60) return; // transient frame before layout
        const C = V.C();
        const narrow = w < 560;
        const ind = narrow ? 16 : 26, rh = narrow ? 30 : 32, x0 = 12;
        const guard = S.yarn && S.len <= 262144;
        const payload = eff() ? (S.nested ? '{"text_config":{…,"rope_parameters":{yarn…}}}' : '{"rope_parameters":{yarn…}}') : '{"text_config":{"ple_embedding_dtype":…}}';
        V.text(g, '--hf-overrides', x0, 18, { size: 11.5, font: 'mono', color: C.muted });
        const pw = Math.min(w - 24, tw(g, payload, 11.5, 'mono', 600) + 16);
        const cmdX = x0, cmdY = 26;
        V.box(g, cmdX, cmdY, pw, 24, { r: 7, fill: V.alpha(C.d, 0.14), stroke: V.alpha(C.d, 0.7), lw: 1 });
        V.text(g, payload, cmdX + 8, cmdY + 16, { size: 11.5, font: 'mono', weight: 600, color: C.ink, maxW: pw - 12 });
        const rows = [];
        let y = 76;
        const node = (depth, label, o = {}) => { const r = { x: x0 + depth * ind, y, label, ...o }; rows.push(r); y += o.h || rh; return r; };
        const nRoot = node(0, 'config · qwen4_exp (multimodal)', { col: C.muted });
        const nA = node(1, narrow ? 'rope_parameters ← parent' : 'rope_parameters  ← the parent\'s plain dict', { col: C.muted });
        const nV = node(1, 'vision_config  (ViT)', { col: C.muted, h: rh - 4 });
        const nB = node(1, 'text_config  (the LLM reads here)', { col: C.qwen });
        const nB1 = node(2, 'max_position_embeddings: 262144', { col: C.muted, h: rh - 4 });
        const nB2 = node(2, 'rope_parameters', { col: C.qwen, h: rh - 6 });
        const L = lands();
        const fields = [
          ['rope_type', L ? '"yarn"' : '"default"', L ? 'ok' : eff() ? 'bad' : 'base'],
          ['factor', L ? '4.0' : '—', L ? 'ok' : 'base'],
          ['original_max_position_embeddings', L ? '262144' : '—', L ? 'ok' : 'base'],
          ['rope_theta', '1e7', L && !S.full ? 'warn' : 'base'],
          ['partial_rotary_factor', '0.25', L && !S.full ? 'warn' : 'base'],
          ['mrope_section', '[11,11,10]', L && !S.full ? 'warn' : 'base'],
          ['mrope_interleaved', 'true', L && !S.full ? 'warn' : 'base'],
        ];
        const fy0 = y, fh = narrow ? 19 : 20;
        fields.forEach(([k, v, st2], i) => {
          const fx = x0 + 3 * ind, fyy = fy0 + i * fh;
          const colr = st2 === 'ok' ? C.a : st2 === 'bad' ? C.c : st2 === 'warn' ? C.d : C.muted;
          const kk = narrow && k.length > 22 ? 'original_max_pos_emb' : k;
          V.text(g, kk + ': ' + v, fx, fyy + 13, { size: 11, font: 'mono', color: st2 === 'base' ? C.muted : colr, weight: st2 === 'base' ? 400 : 700, maxW: w - fx - 70 });
          const tag = st2 === 'ok' ? '✓' : st2 === 'bad' ? '✗ no-op' : st2 === 'warn' ? '? to verify' : '';
          if (tag) V.text(g, tag, w - 12, fyy + 13, { size: 11, font: 'mono', align: 'right', color: colr, weight: 700 });
        });
        y = fy0 + fields.length * fh;
        const drawNode = (r) => {
          const tw3 = Math.min(w - r.x - 12, tw(g, r.label, 12, 'mono', 600) + 16);
          V.box(g, r.x, r.y, tw3, (r.h || rh) - 8, { r: 7, fill: V.alpha(r.col, 0.12), stroke: V.alpha(r.col, 0.6), lw: 1 });
          V.text(g, r.label, r.x + 8, r.y + ((r.h || rh) - 8) / 2 + 4.5, { size: 12, font: 'mono', weight: 600, color: C.ink, maxW: tw3 - 12 });
          r.w = tw3;
        };
        g.strokeStyle = V.alpha(C.muted, 0.4); g.lineWidth = 1;
        [[nRoot, nA], [nRoot, nV], [nRoot, nB], [nB, nB1], [nB, nB2]].forEach(([a, b]) => { g.beginPath(); g.moveTo(a.x + 8, a.y + rh - 8); g.lineTo(a.x + 8, b.y + (b.h || rh) / 2 - 4); g.lineTo(b.x, b.y + (b.h || rh) / 2 - 4); g.stroke(); });
        [nRoot, nA, nV, nB, nB1, nB2].forEach(drawNode);
        if (eff() && !S.nested) { pill(g, nA.x + nA.w + 8, nA.y + (rh - 8) / 2, narrow ? 'yarn: unread' : 'yarn here: nobody reads it', C.c, { size: 11 }); }
        // traveling package
        if (eff()) {
          const tgtN = S.nested ? nB2 : nA;
          const ph = (t % 3.2) / 3.2, u = V.ease.inOut(V.clamp(ph / 0.7, 0, 1));
          const p0 = [cmdX + pw - 12, cmdY + 24], p3 = [tgtN.x + (tgtN.w || 100) + 4, tgtN.y + 10];
          const cxr = Math.min(w - 16, Math.max(p0[0], p3[0]) + 90);
          const c1 = [cxr, p0[1] + 30], c2 = [cxr, p3[1] - 20];
          const pt = V.bez(p0, c1, c2, p3, u);
          g.strokeStyle = V.alpha(C.d, 0.35); g.setLineDash([3, 5]); g.lineWidth = 1.2; g.beginPath();
          for (let k = 0; k <= 30; k++) { const q2 = V.bez(p0, c1, c2, p3, k / 30); if (k) g.lineTo(q2[0], q2[1]); else g.moveTo(q2[0], q2[1]); }
          g.stroke(); g.setLineDash([]);
          V.glow(g, pt[0], pt[1], 16, C.d); V.box(g, pt[0] - 12, pt[1] - 8, 24, 16, { r: 4, fill: C.d });
          if (ph > 0.7) V.glow(g, tgtN.x + (tgtN.w || 100), tgtN.y + 10, 30, S.nested ? C.a : C.c, (1 - ph) * 3);
        }
        let status, sc;
        if (guard) { status = 'NOTE: MAX_MODEL_LEN=' + S.len + ' <= native 262144 — YaRN force-disabled.'; sc = C.d; }
        else if (!eff() && S.len > 262144) { status = 'serves ' + nf(S.len) + ' with unscaled RoPE (YaRN off)'; sc = C.c; }
        else if (!eff()) { status = 'native: original RoPE, no YaRN (the recipe default)'; sc = C.a; }
        else if (!S.nested) { status = narrow ? nf(S.len) + ' with UNSCALED RoPE: no-op' : 'serves ' + nf(S.len) + ' positions with UNSCALED RoPE: silent no-op'; sc = C.c; }
        else { status = narrow ? 'YaRN ×4 in text_config (1M not validated)' : 'YaRN ×4 in text_config: ~1M valid, but not validated' + (S.full ? '' : ' (check the dict)'); sc = C.a; }
        V.text(g, status, w / 2, h - 30, { size: 12, font: 'mono', weight: 700, align: 'center', color: sc, maxW: w - 16 });
        V.text(g, S.len > 262144 ? 'plus VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 in the environment' : '262,144 = max_position_embeddings: the flag is not needed', w / 2, h - 12, { size: 11.5, align: 'center', color: C.muted, maxW: w - 16 });
      });
    }
  },
});

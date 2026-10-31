LA.chapter({
  id: 'abliteracao',
  part: 3,
  order: 19,
  title: 'Directions in latent space: abliteration',
  navTitle: 'Abliteration',
  subtitle: 'How 16 KiB of numbers change the behavior of a model with hundreds of billions of parameters: concepts as directions in the residual stream, difference of means, orthogonal projection, editing o_proj, transplant, runtime ablation and the price you pay (and measure).',
  minutes: 55,
  recipes: ['glm', 'qwen', 'ds'],
  tags: ['abliteration', 'representation engineering', 'refusal direction', 'orthogonal projection', 'o_proj', 'weight transplant', 'runtime ablation', 'KLD'],
  goals: [
    'Explain why a concept can be a <b>direction</b> in the residual stream and estimate it by <b>difference of means</b>',
    'Apply the projection \\( y \\leftarrow y - \\lambda (y\\cdot v)v \\) and predict the effect of λ (remove, reflect, over-project), including in matrix form',
    'Move the projection from the activations into the <code>o_proj</code> weights, understand why tensor parallelism needs no collectives and choose the layers (anchors, MTP, drafters)',
    'Compare the three strategies in the recipes (load-time edit/transplant, derived checkpoint, runtime hook), test a direction against chance and plan how to measure and document the cost',
  ],
  prereqs: ['tres-modelos'],
  summary: [
    'Many behaviors are <b>directions</b> in the residual: you read them with \\( h\\cdot u \\), write them by adding and erase them by projecting. The refusal direction comes from the <b>difference of means</b> of the activations between refused and fulfilled prompts, at an empirically chosen layer. In a 4-stream residual, you measure the mean of the streams (additive because the mHC <code>comb</code> is doubly stochastic) or the attention output before <code>hc_post</code>.',
    '\\( y\\cdot v \\) becomes \\( (1-\\lambda)(y\\cdot v) \\): λ = 1 removes, λ = 2 reflects, λ &gt; 2 inverts and amplifies. GLM uses α = 3 (factor −2), DeepSeek λ = 3.5 (factor −2.5); λ ≥ 4 risks long-CoT degeneration. Only α ∈ {0, 1} is idempotent.',
    'In the weights: \\( W\' = W - \\alpha\\, r(r^\\top W)/\\lVert r\\rVert^2 \\) and \\( r^\\top W\' = (1-\\alpha) r^\\top W \\). <code>o_proj</code> is row-parallel, so the edit works slice by slice with no collectives. In GLM it runs in BF16 at the end of <code>load_weights</code>, before FP8 quantization and graph capture.',
    'Where: GLM edits 31 <code>o_proj</code> (15–44 + MTP 45; 22 KDA [4096×8192] + 9 DSA [4096×16384]) with anchors 0–14 and DFlash2 untouched; Qwen serves a gated checkpoint with the 12 QSA <code>o_proj</code> [2560×6144] edited; DeepSeek projects the attention output in layers 10–42, outside DSpark.',
    'Science before applying: GLM\'s published directions are indistinguishable from chance (max |rᵀW| 0.084 versus 0.084), and projection stalls at 9/32. That is why <code>auto</code> prefers the byte-for-byte <b>transplant</b> of the donor\'s <code>o_proj</code> (32/32), which downloads only ~2.7 GB via range requests instead of ~164 GiB.',
    'DeepSeek\'s runtime hook has a buffer in every layer (one compilation signature), runs in fp32, has an AOT stamp that wipes the cache when λ, the layers or the direction change, and checks the direction\'s SHA-256 on every node.',
    'Price and responsibility: measure KLD on neutral text, refusal rate, degeneration and acceptance (DeepSeek records 0.585 with the abliterated checkpoint versus 0.718 from a different probe, with no A/B of the hook: measure again). GLM did not measure KLD with ablit, and the pre-edited checkpoint has a higher KLD. The safeguards pass to the operator (gated terms, server on 0.0.0.0): document everything in an edit record.',
  ],
  quiz: [
    {
      q: 'In DeepSeek, with a unit \\( v \\) and <code>DSV4_ABLATE_LAMBDA=3.5</code>, what happens to the coordinate \\( y\\cdot v \\) of the attention output in layers 10–42?',
      o: ['It is zeroed', 'It is cut in half', 'It is multiplied by −2.5: it flips sign and gets larger', 'It does not change; only the perpendicular part changes'],
      a: 2,
      e: '\\( y\'\\cdot v = (1-\\lambda)(y\\cdot v) = -2.5\\,(y\\cdot v) \\), and the self-test checks exactly that. Zeroing it would be λ = 1. The part perpendicular to \\( v \\) is the only one that does <i>not</i> change.',
    },
    {
      q: 'Why does GLM\'s <code>ablit_runtime.py</code> edit <code>o_proj</code> on each rank without any extra communication?',
      o: ['Because the edit is done before NCCL is initialized', 'Because <code>o_proj</code> is a RowParallelLinear: each rank has all 4096 output rows and a slice of the columns, and \\( P\\sum_k W_k z_k = \\sum_k (PW_k) z_k \\)', 'Because only rank 0 edits and then broadcasts the weights', 'Because <code>o_proj</code> is fully replicated on every rank'],
      a: 1,
      e: 'The projector multiplies from the left (rows) and \\( r \\) is replicated. Each rank applies the same formula to its own slice of columns, and the all-reduce that already existed sums edited partials. <code>o_proj</code> is not fully replicated: the input dimension (the heads) is sliced.',
    },
    {
      q: 'GLM\'s repository measured max |rᵀW| = 0.084 for the published directions against the stock <code>o_proj</code>, the same value as random unit vectors. What does that imply?',
      o: ['That the directions are correct and just need a larger α', 'That the direction cannot be told apart from chance as far as what <code>o_proj</code> writes is concerned, so projecting it removes almost nothing relevant; that is why <code>auto</code> prefers the transplant (32/32 versus 9/32)', 'That <code>o_proj</code> is quantized and the math loses precision', 'That the directions need to be normalized first'],
      a: 1,
      e: 'It is a test against the null distribution: if \\( s(r) \\) falls in the middle of the cloud of random \\( s(u) \\), then \\( r \\) is not special for that matrix. Raising α does not help, because it amplifies a direction unrelated to the edit. Normalization is already handled by the formula itself (it divides by \\( \\lVert r\\rVert^2 \\)).',
    },
    {
      q: 'Why does DeepSeek\'s hotfix register the <code>_refusal_dir</code> buffer in <b>every</b> decoder layer, including outside 10–42?',
      o: ['To apply the ablation in every layer', 'So that all layers share a single torch.compile/AOT signature; outside the range the buffer stays zeroed and λ = 0', 'Because vLLM requires identical buffers for the all-reduce', 'To allow changing the range without recreating the container'],
      a: 1,
      e: 'With the same structure in every layer, there is only one compiled graph. In the layers outside the range the math gives \\( x - 0 = x \\). Changing the range <i>does</i> require recreating the containers, and the AOT stamp wipes the cache.',
    },
    {
      q: 'Why does changing λ or the range in DeepSeek wipe <code>torch_compile_cache</code>, while turning on <code>ABLIT</code> in GLM does not?',
      o: ['Because GLM does not use torch.compile', 'Because DeepSeek\'s hook changes the model code, and λ, the layers and the direction go into the compatibility stamp of what was compiled; GLM only changes data (weights) before capture, with the same code', 'Because DeepSeek\'s cache lives on NFS', 'Because GLM recompiles everything on every boot'],
      a: 1,
      e: 'The stamp (version, enabled, λ, layers, SHA of the direction, SHA of the patch) protects against reusing an incompatible graph. GLM\'s weight edit happens at the end of <code>load_weights</code>, and the compiled code is identical with or without ABLIT.',
    },
    {
      q: 'GLM\'s <code>abliterated</code> preset forces <code>ABLIT=0</code>. What is the correct justification?',
      o: ['To save memory', 'The checkpoint already comes edited by transplant (15–43 and MTP 45; layer 44 came from the EXL3 parent). Reapplying the runtime edit would produce a different model: with <code>auto</code>, the transplant would replace layer 44 with the donor\'s tensor; with <code>proj</code>, a second, different edit would be stacked on already-edited weights', 'The preset uses dense EXL3, which has no BF16 <code>o_proj</code>', 'Because applying the projection twice with α = 3 would multiply the component by 4'],
      a: 1,
      e: 'That is the justification in the README and in <code>start.sh</code> (“applying the runtime edit again would produce a different model”). The preset\'s checkpoint was made by <i>transplant</i>, not by projection along \\( r \\) with α = 3; that is why the math \\( P_\\alpha^2 = I - (2\\alpha-\\alpha^2)vv^\\top \\) (factor 4) does not apply: it only holds for the <i>same</i> projection applied twice. With <code>ABLIT_METHOD=auto</code> and the tensors downloaded, the actual path would be the transplant, which would replace layer 44 (the rel_l2 outlier at 0.737) with the donor\'s tensor. The preset\'s EXL3 has a BF16 <code>o_proj</code> (the dense EXL3 case is the <code>non_routed_exl3</code> pack).',
    },
  ],
  mount(root, LA) {
    const V = LA.viz, UI = LA.ui;
    const RED = () => V.C().red || '#FB2C36';

    /* ------------------------------------------------------------ helpers */
    function adaptH(fig, wide, narrow, bp = 600) {
      const st = fig.querySelector('.la-fig-stage');
      const set = () => { const want = (fig.clientWidth < bp ? narrow : wide) + 'px'; if (st.style.height !== want) st.style.height = want; };
      set();
      const ro = new ResizeObserver(set); ro.observe(fig);
      if (LA.onCleanup) LA.onCleanup(() => ro.disconnect());
    }
    const fmt = (x, d = 2) => { let v = +x; if (!isFinite(v)) return '—'; if (Math.abs(v) < 0.5 * Math.pow(10, -d)) v = 0; return v.toFixed(d).replace('-', '−'); };
    const fmtInt = (x) => Math.round(x).toLocaleString('en-US');
    function measure(g, s, size, font = 'sans', weight = 400) { g.font = V.font(size, font, weight); return g.measureText(String(s)).width; }
    /** token character: a little square with eyes */
    function tokChar(g, x, y, s, col, o = {}) {
      const C = V.C();
      V.box(g, x - s / 2, y - s / 2, s, s, { r: s * 0.3, fill: o.fill || V.alpha(col, 0.88), stroke: col, lw: 1.2 });
      const eye = C.dark ? '#06140d' : '#ffffff';
      const ex = s * 0.17, ey = y - s * 0.08, er = Math.max(1, s * 0.07);
      V.dot(g, x - ex, ey, er, eye); V.dot(g, x + ex, ey, er, eye);
      g.beginPath();
      if (o.mood === 'sad') g.arc(x, y + s * 0.3, s * 0.14, Math.PI * 1.15, Math.PI * 1.85);
      else if (o.mood === 'o') g.arc(x, y + s * 0.14, s * 0.07, 0, Math.PI * 2);
      else g.arc(x, y + s * 0.06, s * 0.15, Math.PI * 0.2, Math.PI * 0.8);
      g.strokeStyle = eye; g.lineWidth = Math.max(1, s * 0.06); g.stroke();
    }
    function arrow(g, x1, y1, x2, y2, col, lw = 2.2, head = 9, dash) {
      if (Math.hypot(x2 - x1, y2 - y1) < 1.5) return;
      V.arrow(g, x1, y1, x2, y2, { color: col, lw, head: Math.min(head, Math.hypot(x2 - x1, y2 - y1) * 0.8), dash });
    }
    function tag(g, s, x, y, col, o = {}) {
      const size = o.size || 12, font = o.font || 'mono';
      const tw = measure(g, s, size, font, o.weight || 500);
      const px = 6, bh = size + 8;
      const bx = o.align === 'center' ? x - tw / 2 - px : o.align === 'right' ? x - tw - 2 * px : x;
      V.box(g, bx, y - bh / 2, tw + 2 * px, bh, { r: 6, fill: o.fill || V.alpha(col, 0.16), stroke: V.alpha(col, 0.6), lw: 1 });
      V.text(g, s, bx + px, y + 0.5, { size, font, color: o.color || V.C().ink, base: 'middle', weight: o.weight || 500 });
      return tw + 2 * px;
    }
    const clamp = V.clamp, lerp = V.lerp;
    /** V.fig with a guard: the first frame may arrive before layout (w = 1), and then there is nothing to draw */
    const vfig = (fig, draw, o) => V.fig(fig, (g, w, h, ...rest) => { if (w < 120 || h < 80) return; draw(g, w, h, ...rest); }, o);
    /** control label: CSS uppercases labels/readouts; parts inside {…} keep their original case (α, λ, r, p‖q) */
    const lab = (s) => String(s).split(/(\{[^}]*\})/).filter(Boolean).map((p) => (p[0] === '{' ? LA.h('span', { style: { textTransform: 'none', letterSpacing: '0.02em' } }, p.slice(1, -1)) : p));
    /** bytes in English format (decimal point) */
    const bytesBR = (b) => { const u = ['B', 'KiB', 'MiB', 'GiB', 'TiB']; let i = 0; while (Math.abs(b) >= 1024 && i < u.length - 1) { b /= 1024; i++; } return (i ? fmt(b, b < 10 ? 2 : 1) : fmtInt(b)) + ' ' + u[i]; };

    /* ================================================================
       Figure 1 — the residual whiteboard
       ================================================================ */
    (function () {
      const fig = root.querySelector('#fig-residuo');
      adaptH(fig, 400, 470);
      const ctl = UI.controls(fig);
      const S = { prompt: 'recusa', auto: true, k: 17, decomp: true, t0: -7 };
      const NL = 8;
      const emb = [0.2, 0.85];
      const W_A = [0.32, 0.25, -0.2, 0.3, 0.18, -0.22, 0.26, 0.15];
      const W_M = [-0.18, 0.22, 0.16, -0.24, 0.2, 0.12, -0.16, 0.18];
      const U = {
        neutro: { a: [0.06, -0.05, 0.08, 0.04, -0.06, 0.05, 0.07, -0.04], m: [-0.03, 0.05, -0.04, 0.02, 0.06, -0.05, 0.03, 0.04] },
        recusa: { a: [0.06, 0.08, 0.22, 0.62, 0.78, 0.55, 0.2, 0.05], m: [-0.03, 0.05, 0.1, 0.18, 0.22, 0.14, 0.06, 0.02] },
      };
      function writers(p) {
        const out = [{ u: emb[0], w: emb[1], kind: 'emb', name: 'embedding' }];
        for (let l = 0; l < NL; l++) {
          out.push({ u: U[p].a[l], w: W_A[l], kind: 'attn', name: 'L' + l + ' · o_proj' });
          out.push({ u: U[p].m[l], w: W_M[l], kind: 'mlp', name: 'L' + l + ' · down_proj' });
        }
        return out;
      }
      let WR = writers(S.prompt);
      const cum = () => { let a = 0; return WR.map((x) => (a += x.u)); };
      let CUM = cum();
      const NW = WR.length; // 17
      let H;
      UI.seg(ctl, { label: 'Request', options: [{ v: 'neutro', label: 'neutral text' }, { v: 'recusa', label: 'request that triggers refusal' }], value: S.prompt, onChange: (v) => { S.prompt = v; WR = writers(v); CUM = cum(); S.t0 = H ? H.t : 0; ro.set(fmt(CUM[NW - 1], 2)); } });
      const autoT = UI.toggle(ctl, { label: 'autoplay animation', value: true, onChange: (v) => { S.auto = v; if (v) S.t0 = H ? H.t : 0; } });
      UI.slider(ctl, { label: 'writers', min: 0, max: NW, step: 1, value: NW, onInput: (v) => { S.k = v; if (S.auto) autoT.set(false); } });
      UI.toggle(ctl, { label: 'split the current arrow into r and ⟂', value: true, onChange: (v) => { S.decomp = v; } });
      const ro = UI.readout(ctl, lab('{h·r} at the end'), fmt(CUM[NW - 1], 2));
      const A = 0.3, R = [Math.cos(A), -Math.sin(A)], N = [-Math.sin(A), -Math.cos(A)];
      H = vfig(fig, (g, w, h, t) => {
        const C = V.C();
        const narrow = w < 600;
        const colW = narrow ? 0 : Math.min(190, w * 0.24);
        const barsH = narrow ? 118 : 96;
        // current step
        let k, p;
        if (S.auto) { const tau = (t - S.t0) % (NW * 0.75 + 2.6); k = Math.floor(tau / 0.75); p = (tau / 0.75) % 1; if (k >= NW) { k = NW; p = 1; } }
        else { k = S.k; p = 1; }
        const shown = Math.min(NW, k + (p > 0 && k < NW ? 1 : 0));
        // writers column
        if (!narrow) {
          const top = 18, rh = (h - top - 14) / NW;
          V.text(g, 'writers (bottom to top)', 12, 12, { size: 11, color: C.muted, font: 'mono' });
          for (let i = 0; i < NW; i++) {
            const y = h - 14 - (i + 1) * rh;
            const wr = WR[i];
            const col = wr.kind === 'emb' ? C.muted : wr.kind === 'attn' ? C.b : C.e;
            const on = i < shown, cur = i === Math.min(k, NW - 1) && k < NW;
            V.box(g, 34, y + 1, colW - 40, rh - 3, { r: 5, fill: V.alpha(col, cur ? 0.32 : on ? 0.14 : 0.05), stroke: V.alpha(col, cur ? 0.95 : 0.35), lw: cur ? 1.6 : 1 });
            V.text(g, wr.name, 42, y + rh / 2, { size: 11, font: 'mono', base: 'middle', color: on ? C.ink : C.muted });
            if (Math.abs(wr.u) > 0.15) V.text(g, '▲r', colW - 10, y + rh / 2, { size: 11, font: 'mono', base: 'middle', align: 'right', color: C.d });
          }
          const ci = Math.min(k, NW - 1);
          const ty = h - 14 - (ci + 0.5) * rh;
          tokChar(g, 17, ty + Math.sin(t * 5) * 1.2, 18, S.prompt === 'recusa' ? C.c : C.a, { mood: S.prompt === 'recusa' ? 'o' : 'happy' });
        }
        // plane
        const px0 = colW + (narrow ? 12 : 24), px1 = w - 14, py0 = narrow ? 30 : 16, py1 = h - barsH - 16;
        const umin = -0.3, umax = 3.9, wmin = -0.35, wmax = 2.5;
        const corners = [[umin, wmin], [umax, wmin], [umin, wmax], [umax, wmax]].map(([u, v]) => [u * R[0] + v * N[0], u * R[1] + v * N[1]]);
        const mnx = Math.min(...corners.map((c) => c[0])), mxx = Math.max(...corners.map((c) => c[0]));
        const mny = Math.min(...corners.map((c) => c[1])), mxy = Math.max(...corners.map((c) => c[1]));
        const sc = Math.min((px1 - px0) / (mxx - mnx), (py1 - py0) / (mxy - mny));
        const ox = px0 - sc * mnx + ((px1 - px0) - sc * (mxx - mnx)) / 2, oy = py0 - sc * mny + ((py1 - py0) - sc * (mxy - mny)) / 2;
        const P = (u, v) => [ox + sc * (u * R[0] + v * N[0]), oy + sc * (u * R[1] + v * N[1])];
        if (narrow) V.text(g, 'plane (r, rest)', 12, 16, { size: 11, color: C.muted, font: 'mono' });
        // axes
        const a0 = P(umin, 0), a1 = P(umax, 0);
        V.line(g, a0[0], a0[1], a1[0], a1[1], { color: V.alpha(C.d, 0.8), lw: 2, dash: [7, 5] });
        tag(g, 'direction r', a1[0] - 6, a1[1] - 14, C.d, { align: 'right', size: 11 });
        const b0 = P(0, wmin), b1 = P(0, wmax);
        V.line(g, b0[0], b0[1], b1[0], b1[1], { color: V.alpha(C.muted, 0.45), lw: 1 });
        V.text(g, 'rest (⟂ r)', b1[0] + 6, b1[1] + 4, { size: 11, color: C.muted, font: 'mono' });
        // arrows tip to tail
        let cu = 0, cw = 0, lastS = null, lastE = null, lastW = null;
        for (let i = 0; i < shown; i++) {
          const wr = WR[i];
          const f = i === k && k < NW ? V.ease.out(p) : 1;
          const s0 = P(cu, cw), e0 = P(cu + wr.u * f, cw + wr.w * f);
          const col = wr.kind === 'emb' ? C.muted : wr.kind === 'attn' ? C.b : C.e;
          arrow(g, s0[0], s0[1], e0[0], e0[1], col, i === k ? 2.8 : 2, 8);
          lastS = [cu, cw]; lastE = [cu + wr.u * f, cw + wr.w * f]; lastW = wr;
          cu += wr.u * f; cw += wr.w * f;
        }
        if (S.decomp && lastW && lastS) {
          const m = P(lastS[0] + (lastE[0] - lastS[0]), lastS[1]);
          const s0 = P(lastS[0], lastS[1]), e0 = P(lastE[0], lastE[1]);
          V.line(g, s0[0], s0[1], m[0], m[1], { color: C.d, lw: 2.2, dash: [3, 3] });
          V.line(g, m[0], m[1], e0[0], e0[1], { color: V.alpha(C.muted, 0.9), lw: 1.4, dash: [3, 3] });
        }
        // shadow of the tip on r
        const tip = P(cu, cw), foot = P(cu, 0);
        V.line(g, tip[0], tip[1], foot[0], foot[1], { color: V.alpha(C.d, 0.7), lw: 1.2, dash: [2, 4] });
        V.glow(g, foot[0], foot[1], 14, C.d, 0.9);
        V.dot(g, foot[0], foot[1], 4.5, C.d);
        V.dot(g, tip[0], tip[1], 4, C.ink);
        const hr = 'h·r = ' + fmt(cu, 2);
        tag(g, hr, foot[0], foot[1] + 20, C.d, { align: 'center', size: 12, weight: 600 });
        // bars of accumulated h·r
        const bx0 = narrow ? 16 : colW + 24, bx1 = w - 14, by1 = h - 22, bh = barsH - 34;
        V.text(g, 'accumulated h·r after each writer', bx0, by1 - bh - 8, { size: 11, color: C.muted, font: 'mono' });
        const bw = (bx1 - bx0) / NW;
        const vmax = 3.8;
        V.line(g, bx0, by1, bx1, by1, { color: V.alpha(C.muted, 0.5), lw: 1 });
        for (let i = 0; i < NW; i++) {
          const val = i < shown ? (i === shown - 1 ? cu : CUM[i]) : CUM[i];
          const hh = Math.max(1, (Math.max(0, val) / vmax) * bh);
          const on = i < shown;
          const col = WR[i].kind === 'emb' ? C.muted : WR[i].kind === 'attn' ? C.b : C.e;
          V.box(g, bx0 + i * bw + 1.5, by1 - hh, Math.max(2, bw - 3), hh, { r: 2, fill: on ? V.alpha(C.d, 0.35 + 0.5 * (i === shown - 1 ? 1 : 0)) : V.alpha(C.muted, 0.05) });
          V.box(g, bx0 + i * bw + 1.5, by1 + 2, Math.max(2, bw - 3), 3, { r: 1, fill: V.alpha(col, on ? 0.9 : 0.3) });
          if (i % 2 === 1 && bw * 2 > 22) V.text(g, 'L' + ((i - 1) / 2), bx0 + i * bw + bw, by1 + 16, { size: 11, align: 'center', color: C.muted, font: 'mono' });
        }
        V.text(g, 'E', bx0 + bw / 2, by1 + 16, { size: 11, align: 'center', color: C.muted, font: 'mono' });
      });
    })();

    /* ================================================================
       Figure 2 — difference of means (stepper)
       ================================================================ */
    (function () {
      const fig = root.querySelector('#fig-medias');
      adaptH(fig, 440, 580);
      const ctl = UI.controls(fig);
      const NL = 12, NP = 36;
      const rg = V.rng(7);
      const base = { R: [], A: [] };
      for (let i = 0; i < NP; i++) { base.R.push([rg.normal(), rg.normal()]); base.A.push([rg.normal(), rg.normal()]); }
      const sepOf = (l) => 0.35 + 3.0 * Math.exp(-Math.pow((l - 7) / 3.0, 2));
      function dataFor(l) {
        const sep = sepOf(l);
        const Rp = base.R.map(([a, b]) => [sep / 2 + 0.55 * a, 0.95 * b + 0.15]);
        const Ap = base.A.map(([a, b]) => [-sep / 2 + 0.55 * a, 0.95 * b - 0.1]);
        const mean = (P) => [P.reduce((s, q) => s + q[0], 0) / P.length, P.reduce((s, q) => s + q[1], 0) / P.length];
        const mR = mean(Rp), mA = mean(Ap);
        const r = [mR[0] - mA[0], mR[1] - mA[1]], nr = Math.hypot(r[0], r[1]), rh = [r[0] / nr, r[1] / nr];
        const pR = Rp.map((q) => q[0] * rh[0] + q[1] * rh[1]), pA = Ap.map((q) => q[0] * rh[0] + q[1] * rh[1]);
        const st = (a) => { const m = a.reduce((s, x) => s + x, 0) / a.length; const v = a.reduce((s, x) => s + (x - m) * (x - m), 0) / a.length; return [m, Math.sqrt(v)]; };
        const [m1, s1] = st(pR), [m2, s2] = st(pA);
        return { Rp, Ap, mR, mA, r, nr, rh, pR, pA, dp: (m1 - m2) / Math.sqrt((s1 * s1 + s2 * s2) / 2) };
      }
      const DPS = Array.from({ length: NL }, (_, l) => dataFor(l).dp);
      const S = { l: 7, D: dataFor(7) };
      const steps = [
        { t: 'Collect activations', d: 'We run the forward pass on each prompt and save \\(h_\\ell\\) at the last token. Pink: requests the model refuses (\\(\\mathcal D^+\\)). Mint: similar requests it fulfills (\\(\\mathcal D^-\\)).' },
        { t: 'Take the means', d: 'Each cloud becomes a centroid, \\(\\mu^+\\) and \\(\\mu^-\\). Whatever is unrelated to refusal (topic, length, language) spreads equally on both sides and vanishes in the mean.' },
        { t: 'Subtract', d: '\\(r = \\mu^+ - \\mu^-\\) is the gold arrow. Normalized, \\(\\hat r\\) is the refusal direction of this layer. There was no training, just two means and a subtraction.' },
        { t: 'Project onto \\(\\hat r\\)', d: 'Each point casts a shadow onto the line of \\(\\hat r\\). Below, the histograms of the projections form two separate hills, and the separation between them is \\(d\'\\).' },
        { t: 'Choose the layer', d: 'Move the layer slider. Early on the groups mix, in the middle they separate the most, and at the end the separation drops. In practice, each candidate is validated by ablating it and measuring refusal and drift on neutral text.' },
        { t: 'Ablate', d: 'We remove from each point its component along \\(\\hat r\\). The clouds overlap along that direction (the two hills become one), and the other dimension stays intact. This is the orthogonal projection of the next section.' },
      ];
      let H;
      const st = UI.stepper(fig, steps, { autoplay: 6500, onStep: () => H && H.redraw() });
      const rN = UI.readout(ctl, lab('length of {r}'), fmt(S.D.nr, 2));
      const rD = UI.readout(ctl, lab('separation {d′}'), fmt(S.D.dp, 2));
      UI.slider(ctl, { label: lab('layer {ℓ}'), min: 0, max: NL - 1, step: 1, value: S.l, onInput: (v) => { S.l = v; S.D = dataFor(v); rN.set(fmt(S.D.nr, 2)); rD.set(fmt(S.D.dp, 2)); } });
      const PHI = 0.42, cph = Math.cos(PHI), sph = Math.sin(PHI);
      H = vfig(fig, (g, w, h, t) => {
        const C = V.C();
        const D = S.D, i0 = st.i, e = st.elapsed();
        const narrow = w < 600;
        // top strip: mini chart of d′ per layer
        const mcW = Math.min(narrow ? w - 32 : 230, 260), mcH = 44, mcx = narrow ? 16 : w - mcW - 16, mcy = 12;
        V.text(g, "d′ per layer", mcx, mcy + 9, { size: 11, color: C.muted, font: 'mono' });
        const dmax = Math.max(...DPS);
        const bw = mcW / NL;
        for (let l = 0; l < NL; l++) {
          const hh = (DPS[l] / dmax) * (mcH - 14);
          const on = l === S.l;
          V.box(g, mcx + l * bw + 1, mcy + mcH - hh, bw - 2, hh, { r: 2, fill: on ? C.d : V.alpha(C.d, i0 === 4 ? 0.45 : 0.25) });
        }
        V.text(g, 'ℓ = ' + S.l, mcx + S.l * bw + bw / 2, mcy + mcH + 12, { size: 11, align: 'center', color: C.d, font: 'mono' });
        if (i0 === 4) { g.strokeStyle = V.alpha(C.d, 0.5 + 0.4 * Math.sin(t * 4)); g.lineWidth = 1.5; V.rr(g, mcx - 4, mcy - 2, mcW + 8, mcH + 20, 8); g.stroke(); }
        if (!narrow) {
          const ix = mcx, iy = mcy + mcH + 40;
          const lines = [['μ⁺ = (' + fmt(D.mR[0], 2) + ', ' + fmt(D.mR[1], 2) + ')', C.c], ['μ⁻ = (' + fmt(D.mA[0], 2) + ', ' + fmt(D.mA[1], 2) + ')', C.a], ['r = μ⁺ − μ⁻', C.d], ['‖r‖ = ' + fmt(D.nr, 2), C.d], ["d′ = " + fmt(D.dp, 2), C.d]];
          V.text(g, 'layer ' + S.l + ' (toy values)', ix, iy, { size: 11, color: C.muted, font: 'mono' });
          lines.forEach(([tx, cc], k) => { const on = k <= 1 ? i0 >= 1 : k <= 3 ? i0 >= 2 : i0 >= 3; V.text(g, tx, ix, iy + 22 + k * 20, { size: 12, font: 'mono', color: on ? cc : V.alpha(C.muted, 0.6), weight: on ? 600 : 400 }); });
          V.text(g, ['collecting…', 'per-group means', 'one subtraction', 'shadows on r̂', 'move the slider', 'r̂ removed'][i0], ix, iy + 22 + 5 * 20 + 8, { size: 11, color: C.muted });
        }
        // regions
        const topY = narrow ? mcy + mcH + 26 : 20, histH = narrow ? 92 : 84;
        const sx0 = 16, sx1 = narrow ? w - 16 : w - mcW - 32, sy0 = topY, sy1 = h - histH - 26;
        const cx = (sx0 + sx1) / 2, cy = (sy0 + sy1) / 2;
        const sc = Math.min((sx1 - sx0) / 7.4, (sy1 - sy0) / 5.6);
        const TP = (x, y) => [cx + sc * (x * cph - y * sph), cy - sc * (x * sph + y * cph)];
        // ablation: fraction
        const kab = i0 === 5 ? V.ease.inOut(clamp(e / 1.4, 0, 1)) : 0;
        const pos = (q) => { const s = q[0] * D.rh[0] + q[1] * D.rh[1]; return [q[0] - kab * s * D.rh[0], q[1] - kab * s * D.rh[1]]; };
        // line of r̂ (steps ≥ 3)
        if (i0 >= 3) {
          const a0 = TP(-3.4 * D.rh[0], -3.4 * D.rh[1]), a1 = TP(3.4 * D.rh[0], 3.4 * D.rh[1]);
          V.line(g, a0[0], a0[1], a1[0], a1[1], { color: V.alpha(C.d, 0.7), lw: 1.6, dash: [6, 5] });
          tag(g, 'r̂', a1[0] + 4, a1[1], C.d, { size: 12 });
        }
        // points
        const drawSet = (P, col, sign) => {
          P.forEach((q, i) => {
            let pp = pos(q);
            let [X, Y] = TP(pp[0], pp[1]);
            if (i0 === 0) { const f = V.ease.out(clamp((e - i * 0.02 - (sign > 0 ? 0 : 0.3)) / 0.6, 0, 1)); Y = lerp(sy1 + 30, Y, f); if (f <= 0) return; }
            if (i0 === 1) {
              const m = sign > 0 ? D.mR : D.mA; const [MX, MY] = TP(m[0], m[1]);
              const al = 0.28 * Math.sin(Math.PI * clamp(e / 2.2, 0, 1));
              V.line(g, X, Y, MX, MY, { color: V.alpha(col, al), lw: 1 });
            }
            if (i0 === 3) {
              const s = q[0] * D.rh[0] + q[1] * D.rh[1];
              const [FX, FY] = TP(s * D.rh[0], s * D.rh[1]);
              const f = clamp(e / 1.2, 0, 1);
              V.line(g, X, Y, lerp(X, FX, f), lerp(Y, FY, f), { color: V.alpha(col, 0.25), lw: 1 });
              V.dot(g, FX, FY, 2, V.alpha(col, 0.7 * f));
            }
            V.dot(g, X, Y, 3.8, V.alpha(col, 0.85));
          });
        };
        drawSet(D.Ap, C.a, -1);
        drawSet(D.Rp, C.c, 1);
        // centroids
        if (i0 >= 1) {
          [[D.mR, C.c, 'μ⁺'], [D.mA, C.a, 'μ⁻']].forEach(([m, col, lab]) => {
            const pp = pos(m); const [X, Y] = TP(pp[0], pp[1]);
            g.strokeStyle = col; g.lineWidth = 2.4; g.beginPath(); g.arc(X, Y, 8, 0, Math.PI * 2); g.stroke();
            V.line(g, X - 12, Y, X + 12, Y, { color: col, lw: 1.5 }); V.line(g, X, Y - 12, X, Y + 12, { color: col, lw: 1.5 });
            if (lab === 'μ⁺') tag(g, lab, X + 12, Y - 16, col, { size: 12 }); else tag(g, lab, X - 12, Y + 18, col, { size: 12, align: 'right' });
          });
        }
        if (i0 >= 2 && i0 < 5) {
          const f = i0 === 2 ? V.ease.out(clamp(e / 1.1, 0, 1)) : 1;
          const a = pos(D.mA), b = pos(D.mR);
          const [X0, Y0] = TP(a[0], a[1]), [X1, Y1] = TP(lerp(a[0], b[0], f), lerp(a[1], b[1], f));
          V.glow(g, X1, Y1, 16, C.d, 0.6);
          arrow(g, X0, Y0, X1, Y1, C.d, 3.4, 12);
          if (f > 0.9) tag(g, 'r = μ⁺ − μ⁻', (X0 + X1) / 2, (Y0 + Y1) / 2 + 22, C.d, { align: 'center', size: 12, weight: 600 });
        }
        // legends
        V.dot(g, sx0 + 6, sy1 + 4, 4, C.c); V.text(g, 'refuses', sx0 + 14, sy1 + 8, { size: 11, color: C.muted });
        V.dot(g, sx0 + 76, sy1 + 4, 4, C.a); V.text(g, 'fulfills', sx0 + 84, sy1 + 8, { size: 11, color: C.muted });
        // histograms of the projections
        const hx0 = 16, hx1 = w - 16, hy1 = h - 20, hh = histH - 22;
        V.line(g, hx0, hy1, hx1, hy1, { color: V.alpha(C.muted, 0.5), lw: 1 });
        V.text(g, 'projection h·r̂  (histogram)', hx0, hy1 - hh - 6, { size: 11, color: C.muted, font: 'mono' });
        if (i0 >= 3) {
          const NB = 30, lo = -3.6, hi = 3.6, bwh = (hx1 - hx0) / NB;
          const hist = (arr) => { const b = new Array(NB).fill(0); arr.forEach((s) => { const k = Math.floor(((s * (1 - kab)) - lo) / (hi - lo) * NB); if (k >= 0 && k < NB) b[k]++; }); return b; };
          const hR = hist(D.pR), hA = hist(D.pA);
          const mx = Math.max(6, ...hR, ...hA);
          const f = i0 === 3 ? clamp(e / 1.2, 0, 1) : 1;
          for (let k = 0; k < NB; k++) {
            const a = (hA[k] / mx) * hh * f, b = (hR[k] / mx) * hh * f;
            if (a > 0) V.box(g, hx0 + k * bwh + 1, hy1 - a, bwh - 2, a, { r: 2, fill: V.alpha(C.a, 0.55) });
            if (b > 0) V.box(g, hx0 + k * bwh + 1, hy1 - b, bwh - 2, b, { r: 2, fill: V.alpha(C.c, 0.5) });
          }
          const zx = hx0 + (0 - lo) / (hi - lo) * (hx1 - hx0);
          V.line(g, zx, hy1 - hh, zx, hy1, { color: V.alpha(C.muted, 0.4), lw: 1, dash: [2, 3] });
          V.text(g, "d′ = " + fmt(D.dp * (1 - kab), 2), hx1, hy1 - hh - 6, { size: 12, align: 'right', color: C.d, font: 'mono', weight: 600 });
        } else {
          V.text(g, '(appears at step 4)', (hx0 + hx1) / 2, hy1 - hh / 2, { size: 11, align: 'center', color: V.alpha(C.muted, 0.8) });
        }
      });
    })();

    /* ================================================================
       Figure 2b — four streams (mHC), one additive mean
       ================================================================ */
    (function () {
      const fig = root.querySelector('#fig-fluxos');
      adaptH(fig, 360, 600);
      const ctl = UI.controls(fig);
      const K = 8, E0 = 0.3;                                   // 8 writers; embedding·r replicated across the 4 streams (hc_expand)
      const YS = [0.06, 0.1, 0.32, 0.58, 0.46, 0.22, -0.1, 0.12];
      const S = { ds: true, seed: 1, t0: 0, lastK: -1 };
      let D = null, H;
      const SUB = ['₀', '₁', '₂', '₃'];
      function sinkhorn(M) {
        let A = M.map((r) => r.slice());
        for (let it = 0; it < 20; it++) {
          A = A.map((r) => { const s = r.reduce((a, b) => a + b, 0); return r.map((x) => x / s); });
          for (let j = 0; j < 4; j++) { let s = 0; for (let i = 0; i < 4; i++) s += A[i][j]; for (let i = 0; i < 4; i++) A[i][j] /= s; }
        }
        return A;
      }
      function build() {
        const rg = V.rng(900 + S.seed * 17);
        const raws = [], posts = [];
        for (let k = 0; k < K; k++) {
          raws.push(Array.from({ length: 4 }, () => Array.from({ length: 4 }, () => 0.05 + 0.5 * rg())));
          posts.push(Array.from({ length: 4 }, () => 2 / (1 + Math.exp(-0.9 * rg.normal()))));
        }
        D = { raws, posts, ys: YS.map((y) => y + 0.05 * rg.normal()) };
        sim();
      }
      function sim() {
        let Hc = [E0, E0, E0, E0], pred = E0;
        D.steps = [];
        for (let k = 0; k < K; k++) {
          const M = S.ds ? sinkhorn(D.raws[k]) : D.raws[k], p = D.posts[k], y = D.ys[k];
          const mix = [0, 1, 2, 3].map((i) => [0, 1, 2, 3].reduce((s, j) => s + M[j][i] * Hc[j], 0));   // H'ᵢ = Σⱼ comb[j][i]·Hⱼ
          const after = mix.map((v, i) => v + p[i] * y);                                                   // + postᵢ·y
          const pbar = p.reduce((a, b) => a + b, 0) / 4;
          const predB = pred; pred += pbar * y;
          D.steps.push({ before: Hc, mix, after, M, p, y, pbar, predB, pred, rowS: M.map((r) => r.reduce((a, b) => a + b, 0)) });
          Hc = after;
        }
        const all = D.steps.flatMap((s) => [...s.before, ...s.mix, ...s.after, s.pred, s.predB]);
        D.vmax = Math.max(1, ...all) * 1.05; D.vmin = Math.min(0, ...all);
        S.lastK = -1;
      }
      UI.toggle(ctl, { label: 'doubly stochastic comb (Sinkhorn, as in mHC)', value: true, onChange: (v) => { S.ds = v; sim(); S.t0 = H ? H.t : 0; } });
      UI.button(ctl, '↻ another token', () => { S.seed++; build(); S.t0 = H ? H.t : 0; });
      const rM = UI.readout(ctl, lab('mean of streams · {r}'), '—');
      const rP = UI.readout(ctl, lab('predicted: {h̄₀·r + Σ p̄ₖ (yₖ·r)}'), '—');
      build();
      H = vfig(fig, (g, w, h, t) => {
        const C = V.C();
        const narrow = w < 620;
        const SD = 2.4, TOT = K * SD + 2.6;
        const tau = ((t - S.t0) % TOT + TOT) % TOT;
        let k = Math.floor(tau / SD), ph = (tau - k * SD) / SD;
        if (k >= K) { k = K - 1; ph = 1; }
        const st = D.steps[k];
        const fm = V.ease.inOut(clamp(ph / 0.36, 0, 1));             // 1. mixing
        const fw = V.ease.out(clamp((ph - 0.42) / 0.32, 0, 1));      // 2. write
        const mixed = st.before.map((b, i) => lerp(b, st.mix[i], fm));
        const cur = mixed.map((m, i) => m + fw * st.p[i] * st.y);
        const mean = cur.reduce((a, b) => a + b, 0) / 4;
        const predNow = st.predB + fw * st.pbar * st.y;
        const tM = fmt(mean, 3), tP = fmt(predNow, 3);
        if (S.lastK !== tM + '|' + tP) { S.lastK = tM + '|' + tP; rM.set(tM); rP.set(tP); }
        // ---------- stream lanes
        const lx0 = 16, lx1 = narrow ? w - 16 : w * 0.58;
        const laneTop = 58, laneH = narrow ? 34 : 38;
        const labW = 96, zx0 = lx0 + labW, zx1 = lx1 - 58;
        const X = (v) => zx0 + (v - D.vmin) / (D.vmax - D.vmin) * (zx1 - zx0);
        const phase = ph < 0.4 ? '1 · comb mixes the streams' : ph < 0.78 ? '2 · post spreads the write y·r' : '✓ writer applied';
        V.text(g, 'coordinate along r · writer ' + (k + 1) + ' of ' + K, 16, 18, { size: 11, font: 'mono', color: C.muted, maxW: lx1 - 16 });
        V.text(g, phase, 16, 36, { size: 12, color: ph < 0.4 ? C.b : C.d, weight: 600, maxW: lx1 - 16 });
        const laneY = (i) => laneTop + i * laneH;
        V.line(g, X(0), laneTop - 12, X(0), laneY(3) + 12, { color: V.alpha(C.muted, 0.5), lw: 1 });
        for (let i = 0; i < 4; i++) {
          const yy = laneY(i);
          V.text(g, 'H' + SUB[i] + '·r', lx0, yy + 4, { size: 12, font: 'mono', color: C.ink });
          V.box(g, zx0, yy - 7, zx1 - zx0, 14, { r: 4, fill: V.alpha(C.muted, 0.08) });
          const a = X(0), b = X(mixed[i]), c = X(cur[i]);
          V.box(g, Math.min(a, b), yy - 6, Math.max(2, Math.abs(b - a)), 12, { r: 3, fill: V.alpha(C.b, 0.7) });
          if (fw > 0) V.box(g, Math.min(b, c), yy - 6, Math.max(1, Math.abs(c - b)), 12, { r: 3, fill: V.alpha(C.d, 0.9) });
          V.text(g, fmt(cur[i], 2), zx1 + 6, yy + 4, { size: 11, font: 'mono', color: C.muted });
        }
        // mixing lines (left of the lanes, between the labels)
        if (ph < 0.42) {
          const mx0 = lx0 + 44, mx1 = zx0 - 4, al = Math.sin(Math.PI * clamp(ph / 0.42, 0, 1));
          for (let j = 0; j < 4; j++) for (let i = 0; i < 4; i++) {
            const wgt = st.M[j][i];
            g.strokeStyle = V.alpha(C.b, al * (0.15 + 0.7 * clamp(wgt, 0, 1)));
            g.lineWidth = 0.6 + 3 * clamp(wgt, 0, 1);
            g.beginPath(); g.moveTo(mx0, laneY(j)); g.bezierCurveTo((mx0 + mx1) / 2, laneY(j), (mx0 + mx1) / 2, laneY(i), mx1, laneY(i)); g.stroke();
          }
        }
        // drops of the write y falling onto the lanes
        if (fw > 0 && fw < 1) {
          for (let i = 0; i < 4; i++) {
            const xx = X(mixed[i] + fw * st.p[i] * st.y);
            V.glow(g, xx, laneY(i), 10, C.d, 0.6 * (1 - fw));
          }
        }
        // ---------- mean and prediction
        const my = laneY(3) + (narrow ? 44 : 50);
        V.text(g, 'mean·r', lx0, my + 4, { size: 12, font: 'mono', color: C.d, weight: 600 });
        V.box(g, zx0, my - 9, zx1 - zx0, 18, { r: 5, fill: V.alpha(C.d, 0.08) });
        const a0 = X(0), m1 = X(mean);
        V.box(g, Math.min(a0, m1), my - 8, Math.max(2, Math.abs(m1 - a0)), 16, { r: 4, fill: V.alpha(C.d, 0.85) });
        const px = X(predNow);
        V.line(g, px, my - 15, px, my + 15, { color: C.e, lw: 2.4 });
        V.text(g, fmt(mean, 2), zx1 + 6, my + 4, { size: 11, font: 'mono', color: C.d, weight: 600 });
        const ok = Math.abs(mean - predNow) < 5e-3;
        V.text(g, ok ? '✓ mean = predicted: only positive scalars add up along r' : '✗ mean ≠ predicted: unconstrained mixing created or destroyed mass along r', lx0, my + 34, { size: 11, color: ok ? C.e : RED(), weight: 600, maxW: lx1 - lx0 });
        V.text(g, 'green tick = h̄₀·r + Σₖ p̄ₖ (yₖ·r), the predicted mean', lx0, my + 52, { size: 11, color: C.muted, maxW: lx1 - lx0 });
        // ---------- comb panel (right on desktop, below on mobile)
        const px0 = narrow ? 16 : w * 0.62, py0 = narrow ? my + 76 : 24, pw = narrow ? w - 32 : w - px0 - 16;
        const cell = clamp(Math.floor((pw - 80) / 5), 26, narrow ? 36 : 42);
        const gx = px0 + 34, gy = py0 + 44;
        V.text(g, 'comb of writer ' + (k + 1) + ' (row = from, column = to)', px0, py0 + 8, { size: 11, font: 'mono', color: C.ink, maxW: pw });
        V.text(g, S.ds ? 'doubly stochastic: rows and columns sum to 1' : 'unconstrained: the sums drift away from 1', px0, py0 + 24, { size: 11, color: S.ds ? C.e : RED(), weight: 600, maxW: pw });
        const geo = V.matrix(g, { x: gx, y: gy, data: st.M, cell, min: 0, max: 1, cmap: 'seq', values: true, fmt: (q) => fmt(q, 2), rowLabels: ['H₀', 'H₁', 'H₂', 'H₃'], colLabels: ['H₀', 'H₁', 'H₂', 'H₃'] });
        V.text(g, 'Σ', gx + geo.w + 22, gy - 6, { size: 11, font: 'mono', align: 'center', color: C.muted });
        st.rowS.forEach((sv, i) => {
          const good = Math.abs(sv - 1) < 1e-3;
          V.text(g, fmt(sv, 2), gx + geo.w + 22, gy + i * (cell + 2) + cell / 2 + 4, { size: 11, font: 'mono', align: 'center', color: good ? C.e : RED(), weight: 600 });
        });
        const iy = gy + geo.h + 34;
        V.text(g, 'sum of each row = how much the source stream redistributes', px0, iy, { size: 11, color: C.muted, maxW: pw });
        V.text(g, 'post = (' + st.p.map((q) => fmt(q, 2)).join(', ') + ')', px0, iy + 20, { size: 11, font: 'mono', color: C.d, maxW: pw });
        V.text(g, 'p̄ = mean(post) = ' + fmt(st.pbar, 2) + '   ·   y·r = ' + fmt(st.y, 2), px0, iy + 38, { size: 11, color: C.d, maxW: pw });
        // ---------- writers timeline
        const tlY = h - 18, tlW = Math.min(narrow ? w - 32 : w * 0.56, K * 40), bwd = tlW / K;
        for (let q = 0; q < K; q++) {
          const on = q === k, done = q < k;
          V.box(g, 16 + q * bwd + 2, tlY - 9, bwd - 4, 16, { r: 4, fill: on ? V.alpha(C.d, 0.8) : done ? V.alpha(C.d, 0.3) : V.alpha(C.muted, 0.12) });
          V.text(g, String(q + 1), 16 + q * bwd + bwd / 2, tlY + 3, { size: 11, font: 'mono', align: 'center', color: on && !C.dark ? '#fff' : C.ink });
        }
      });
    })();

    /* ================================================================
       Figure 3 — projection with strength λ + matrix P
       ================================================================ */
    (function () {
      const fig = root.querySelector('#fig-lambda');
      adaptH(fig, 400, 760, 640);
      const ctl = UI.controls(fig);
      const S = { lam: 1, y: [-0.5, 2.0], sc: 0 };
      const va = 0.62, v = [Math.cos(va), Math.sin(va)];
      const v6 = [3, 1, 0, 2, 0, 1].map((x) => x / Math.sqrt(15));
      let H;
      const sl = UI.slider(ctl, { label: lab('strength {λ}'), min: 0, max: 5, step: 0.05, value: 1, fmt: (x) => fmt(x, 2), onInput: (x) => { S.lam = x; upd(); } });
      UI.seg(ctl, { label: 'Presets', options: [{ v: 0, label: '0' }, { v: 1, label: '1 · remove' }, { v: 2, label: '2 · reflect' }, { v: 3, label: '3 · GLM α' }, { v: 3.5, label: '3.5 · DS λ' }], value: 1, onChange: (x) => sl.set(x) });
      const r1 = UI.readout(ctl, lab('factor along {v}: {1 − λ}'), '0.00');
      const r2 = UI.readout(ctl, lab('{‖y′‖ / ‖y‖}'), '—');
      function comp() {
        const c = S.y[0] * v[0] + S.y[1] * v[1];
        const yp = [S.y[0] - S.lam * c * v[0], S.y[1] - S.lam * c * v[1]];
        return { c, yp };
      }
      function upd() { const { yp } = comp(); r1.set(fmt(1 - S.lam, 2)); r2.set(fmt(Math.hypot(yp[0], yp[1]) / Math.max(1e-6, Math.hypot(S.y[0], S.y[1])), 2)); if (H) H.redraw(); }
      H = vfig(fig, (g, w, h, t, dt, St, HH) => {
        const C = V.C();
        const narrow = w < 640;
        const pw = narrow ? w : w * 0.56, ph = narrow ? 360 : h;
        const ox = pw / 2, oy = ph / 2 + 8;
        const ptr = HH.pointer;
        const dragging = ptr.down && ptr.inside && ptr.x < pw && ptr.y < ph;
        // scale that fits the whole rail (λ from 0 to 5), frozen while dragging
        const c0 = S.y[0] * v[0] + S.y[1] * v[1];
        const end = [S.y[0] - 5 * c0 * v[0], S.y[1] - 5 * c0 * v[1]];
        const Rb = Math.max(2.2, Math.hypot(S.y[0], S.y[1]), Math.hypot(end[0], end[1])) + 0.45;
        const target = Math.min(pw - 20, ph - 50) / (2 * Rb);
        if (!S.sc) S.sc = target; else if (!dragging) S.sc = lerp(S.sc, target, 0.12);
        const sc = S.sc;
        const T = (x, y) => [ox + sc * x, oy - sc * y];
        const inv = (X, Y) => [(X - ox) / sc, (oy - Y) / sc];
        if (dragging) {
          const q = inv(ptr.x, ptr.y);
          const n = Math.hypot(q[0], q[1]), k = n > 3 ? 3 / n : 1;
          S.y = [q[0] * k, q[1] * k];
          const { yp } = comp(); r2.set(fmt(Math.hypot(yp[0], yp[1]) / Math.max(1e-6, Math.hypot(S.y[0], S.y[1])), 2));
        }
        const { c, yp } = comp();
        const L = Rb;
        V.line(g, T(-L, 0)[0], T(-L, 0)[1], T(L, 0)[0], T(L, 0)[1], { color: V.alpha(C.muted, 0.16), lw: 1 });
        V.line(g, T(0, -L)[0], T(0, -L)[1], T(0, L)[0], T(0, L)[1], { color: V.alpha(C.muted, 0.16), lw: 1 });
        // v⊥
        const pv = [-v[1], v[0]];
        const q0 = T(-L * pv[0], -L * pv[1]), q1 = T(L * pv[0], L * pv[1]);
        V.line(g, q0[0], q0[1], q1[0], q1[1], { color: V.alpha(C.muted, 0.75), lw: 1.6 });
        tag(g, 'v⊥ (untouched)', q0[0] - 4, q0[1] - 12, C.muted, { size: 11, align: 'right' });
        // line of v
        const l0 = T(-L * v[0], -L * v[1]), l1 = T(L * v[0], L * v[1]);
        V.line(g, l0[0], l0[1], l1[0], l1[1], { color: V.alpha(C.d, 0.55), lw: 1.4, dash: [6, 5] });
        const v1 = T(v[0], v[1]);
        arrow(g, ox, oy, v1[0], v1[1], C.d, 3.2, 11);
        tag(g, 'v', v1[0] + 8, v1[1] + 2, C.d, { size: 12, weight: 600 });
        // λ rail
        const at = (Lm) => T(S.y[0] - Lm * c * v[0], S.y[1] - Lm * c * v[1]);
        const p0 = at(0), p4 = at(4), p5 = at(5);
        V.line(g, p0[0], p0[1], p4[0], p4[1], { color: V.alpha(C.b, 0.5), lw: 1.4, dash: [3, 4] });
        V.line(g, p4[0], p4[1], p5[0], p5[1], { color: V.alpha(RED(), 0.75), lw: 2.2, dash: [3, 4] });
        if (Math.abs(c) * sc > 18) {
          const nx = -v[1], ny = v[0];
          [[1, 'λ=1'], [2, '2'], [3, 'α=3'], [3.5, '3.5'], [4, '4']].forEach(([Lm, lab], k) => {
            const q = at(Lm);
            V.dot(g, q[0], q[1], 3.2, Lm >= 4 ? RED() : V.alpha(C.b, 0.95));
            const sgn = k % 2 ? -1 : 1;
            V.text(g, lab, q[0] + sgn * nx * 16, q[1] - sgn * ny * 16 + 4, { size: 11, font: 'mono', color: Lm >= 4 ? RED() : C.muted, align: 'center' });
          });
        }
        // component of y along v
        const yT = T(S.y[0], S.y[1]), fT = T(S.y[0] - c * v[0], S.y[1] - c * v[1]);
        V.line(g, yT[0], yT[1], fT[0], fT[1], { color: V.alpha(C.c, 0.9), lw: 1.8, dash: [2, 3] });
        V.dot(g, fT[0], fT[1], 3, C.c);
        arrow(g, ox, oy, yT[0], yT[1], C.b, 2.6, 10);
        g.strokeStyle = V.alpha(C.b, 0.5 + 0.3 * Math.sin(t * 3)); g.lineWidth = 1.5; g.beginPath(); g.arc(yT[0], yT[1], 9 + Math.sin(t * 3), 0, Math.PI * 2); g.stroke();
        // labels of y and y′ on opposite sides: each goes on the side of its own arrow facing away from the other arrow
        const perpS = (a, b, tie) => {
          const n = Math.hypot(a[0], a[1]) || 1, p = [-a[1] / n, a[0] / n];
          const d = p[0] * b[0] + p[1] * b[1];
          const s = Math.abs(d) < 0.08 * (Math.hypot(b[0], b[1]) + 1e-9) ? tie : d > 0 ? -1 : 1;   // away from b
          return [s * p[0], -s * p[1]];                                   // world → screen (y flipped)
        };
        const place = (s, P, off, col, o) => {
          const al = Math.abs(off[0]) < 0.35 ? 'center' : off[0] > 0 ? 'left' : 'right';
          tag(g, s, P[0] + off[0] * 16, P[1] + off[1] * 17, col, Object.assign({ align: al }, o));
        };
        place('y (drag)', yT, perpS(S.y, yp, -1), C.b, { size: 11 });
        const ypT = T(yp[0], yp[1]);
        arrow(g, ox, oy, ypT[0], ypT[1], C.a, 3.4, 12);
        V.glow(g, ypT[0], ypT[1], 14, C.a, 0.8);
        place('y′', ypT, perpS(yp, S.y, 1), C.a, { size: 12, weight: 600 });
        // component bars (top-left corner)
        const bx = 14, by = 18, bl = Math.min(130, pw * 0.3);
        const cp = yp[0] * v[0] + yp[1] * v[1];
        const bar = (yy, val, col, lab) => {
          V.text(g, lab, bx, yy + 4, { size: 11, font: 'mono', color: C.muted });
          const zx = bx + 44 + bl / 2;
          V.line(g, zx, yy - 7, zx, yy + 7, { color: V.alpha(C.muted, 0.6), lw: 1 });
          const ww = clamp(val / 6, -1, 1) * (bl / 2);
          V.box(g, Math.min(zx, zx + ww), yy - 5, Math.max(2, Math.abs(ww)), 10, { r: 3, fill: col });
          V.text(g, fmt(val, 2), zx + bl / 2 + 8, yy + 4, { size: 11, font: 'mono', color: C.ink });
        };
        bar(by, c, C.b, 'y·v');
        bar(by + 20, cp, C.a, "y′·v");
        // matrix panel
        const mx0 = narrow ? 0 : pw, my0 = narrow ? ph : 0, mw = narrow ? w : w - pw, mh = narrow ? h - ph : h;
        const P = v6.map((a, i) => v6.map((b, j) => (i === j ? 1 : 0) - S.lam * a * b));
        const cell = clamp(Math.floor(Math.min(mw - 70, mh - 110) / 6) - 2, 31, 44);
        const gw = 6 * cell + 10;
        const gx = mx0 + (mw - gw) / 2 + 10, gy = my0 + (narrow ? 44 : Math.max(52, (mh - gw) / 2 - 10));
        V.text(g, 'P = I − λ v vᵀ  (d = 6)', mx0 + mw / 2 + 5, gy - 26, { size: 13, font: 'mono', align: 'center', color: C.ink, weight: 600 });
        V.matrix(g, { x: gx, y: gy, data: P, cell, min: -2.5, max: 2.5, cmap: 'div', values: true, fmt: (x) => fmt(x, 2), rowLabels: v6.map((x) => fmt(x, 2)) });
        V.text(g, 'v =', gx - 8, gy - 8, { size: 11, font: 'mono', color: C.d, align: 'right' });
        V.text(g, 'eigenvalues: 1 (×5) and 1 − λ = ' + fmt(1 - S.lam, 2), mx0 + mw / 2 + 5, gy + gw + 12, { size: 12, font: 'mono', align: 'center', color: C.muted });
        V.text(g, 'rows/columns where v = 0 stay equal to those of I', mx0 + mw / 2 + 5, gy + gw + 30, { size: 11, align: 'center', color: C.muted });
      });
      upd();
    })();

    /* ================================================================
       Figure 4 — orthogonalizing o_proj (stepper)
       ================================================================ */
    (function () {
      const fig = root.querySelector('#fig-orto');
      adaptH(fig, 430, 460);
      const ctl = UI.controls(fig);
      const W = [[0.8, -0.3, 0.5, 0.1, -0.6, 0.4], [0.2, 0.9, -0.4, 0.7, 0.3, -0.2], [-0.5, 0.4, 0.6, -0.3, 0.8, 0.1], [0.3, -0.7, 0.2, 0.5, -0.1, 0.9]];
      const z = [1.0, 0.5, -0.8, 0.6, 0.3, -0.4];
      const r = [0.6, 0, 0.8, 0];
      const S = { a: 1 };
      function comp(a) {
        const mv = (M, x) => M.map((row) => row.reduce((s, q, j) => s + q * x[j], 0));
        const y = mv(W, z);
        const ry = r.reduce((s, q, i) => s + q * y[i], 0);
        const yp = y.map((q, i) => q - a * ry * r[i]);
        const rW = z.map((_, j) => r.reduce((s, ri, i) => s + ri * W[i][j], 0));
        const dW = W.map((row, i) => row.map((_, j) => a * r[i] * rW[j]));
        const Wp = W.map((row, i) => row.map((q, j) => q - dW[i][j]));
        const Wpz = mv(Wp, z);
        const rWp = z.map((_, j) => r.reduce((s, ri, i) => s + ri * Wp[i][j], 0));
        const y0 = Wp.map((row) => row.slice(0, 3).reduce((s, q, j) => s + q * z[j], 0));
        const y1 = Wp.map((row) => row.slice(3).reduce((s, q, j) => s + q * z[j + 3], 0));
        const rW0 = rW.slice(0, 3), rW1 = rW.slice(3);
        return { y, ry, yp, rW, dW, Wp, Wpz, rWp, y0, y1, rW0, rW1 };
      }
      let K = comp(S.a);
      let H;
      const steps = [
        { t: 'The output of o_proj', d: 'Each row of \\(W\\) combines the 6 inputs \\(z\\) (head channels) into one residual dimension: \\(y = Wz\\). The rows light up one by one.' },
        { t: 'Runtime: project the output', d: 'The DeepSeek way: compute \\(r^\\top y\\) and subtract \\(\\alpha\\,(r^\\top y)\\,r\\) from every token. Only coordinates 0 and 2 change, because that is where \\(r\\) lives.' },
        { t: 'Read what W writes along r', d: '\\(r^\\top W\\) is a row of 6 numbers: how much each input writes along \\(r\\). Here it is \\(0.6\\cdot\\)row 0 \\(+\\ 0.8\\cdot\\)row 2.' },
        { t: 'The rank-1 correction', d: '\\(\\Delta W = \\alpha\\, r \\otimes (r^\\top W)\\): each row \\(i\\) receives \\(\\alpha r_i\\) times the same row \\(r^\\top W\\). Rows 1 and 3 receive zero.' },
        { t: 'Edited weights', d: '\\(W\' = W - \\Delta W\\). The cells change only in the rows where \\(r\\) is nonzero. This is what <code>apply_to_o_proj</code> does in fp32 before going back to BF16.' },
        { t: 'Same answer', d: '\\(W\'z = y\'\\): editing the weights once is equivalent to projecting the output of every token. And \\(r^\\top W\' = (1-\\alpha)\\,r^\\top W\\): with α = 1 the layer is blind to \\(r\\); with α = 3 it writes −2×.' },
        { t: 'With TP = 2', d: 'Row-parallel: rank 0 holds columns 0–2 and rank 1 holds 3–5, each with all 4 rows. Each rank edits its own slice with the same \\(r\\), without talking to the other. The all-reduce sums \\(y\'_0 + y\'_1 = y\'\\).' },
      ];
      const st = UI.stepper(fig, steps, { autoplay: 6500, onStep: () => H && H.redraw() });
      UI.seg(ctl, { label: lab('strength {α}'), options: [{ v: 1, label: 'α = 1 (removes)' }, { v: 3, label: 'α = 3 (GLM)' }], value: 1, onChange: (v) => { S.a = v; K = comp(v); if (H) H.redraw(); } });
      const col = (vec) => vec.map((x) => [x]);
      const row = (vec) => [vec];
      H = vfig(fig, (g, w, h, t) => {
        const C = V.C();
        const i0 = st.i, e = st.elapsed();
        const narrow = w < 560;
        let cw = narrow ? 34 : clamp(Math.floor((w - 80) / 13), 34, 44);
        if (narrow && i0 === 3) cw = clamp(Math.floor((w - 28 - 14 - 10) / 7), 30, 44);
        const eqs = ['y = W · z', "y′ = y − α (rᵀy) r", 'rᵀW = Σᵢ rᵢ · rowᵢ(W)', 'ΔW = α · r ⊗ (rᵀW)', "W′ = W − ΔW", "W′z = y′   and   rᵀW′ = (1−α) rᵀW", "rank k: W′ₖ = Wₖ − α r (rᵀWₖ)  →  Σₖ W′ₖ zₖ = y′"];
        V.text(g, eqs[i0], w / 2, 20, { size: narrow ? 12 : 14, font: 'mono', align: 'center', color: C.ink, weight: 600, maxW: w - 20 });
        V.text(g, 'α = ' + S.a + '   ·   r = (0.6, 0, 0.8, 0)', w / 2, 38, { size: 11, font: 'mono', align: 'center', color: C.d });
        // blocks: {label, data, hi, col}
        const blocks = [];
        const f = clamp(e / 1.6, 0, 1);
        const ri = [0, 2];
        if (i0 === 0) {
          const n = Math.min(4, Math.floor(e / 0.55) + 1);
          blocks.push({ label: 'W (o_proj 4×6)', data: W, hiRow: (n - 1) % 4 });
          blocks.push({ label: 'z', data: col(z), hiCol: 0 });
          blocks.push({ label: 'y = Wz', data: col(K.y.map((q, i) => (i < n ? q : NaN))), hiRow: n - 1 });
        } else if (i0 === 1) {
          blocks.push({ label: 'y', data: col(K.y) });
          blocks.push({ label: 'r', data: col(r), accent: C.d });
          blocks.push({ label: 'rᵀy', data: [[K.ry]], accent: C.d });
          blocks.push({ label: "y′ = y − α(rᵀy) r", data: col(K.y.map((q, i) => lerp(q, K.yp[i], f))), hiCells: ri.map((i) => [i, 0]) });
        } else if (i0 === 2) {
          blocks.push({ label: 'r', data: col(r), accent: C.d });
          blocks.push({ label: 'W', data: W, hiRows: ri });
          const n = Math.min(6, Math.floor(e / 0.35) + 1);
          blocks.push({ label: 'rᵀW (1×6)', data: row(K.rW.map((q, j) => (j < n ? q : NaN))), hiCol: n - 1, accent: C.d, br: true });
        } else if (i0 === 3) {
          // outer product drawn like a multiplication table: r in the left column, rᵀW in the top row
          const n = Math.min(24, Math.floor(e / 0.07) + 1);
          const gap = 14, tw = cw + gap + 6 * (cw + 2) - 2, th = (cw + 2) + gap + 4 * (cw + 2) - 2 + 18;
          const ox = (w - tw) / 2, oy = Math.max(58, 50 + (h - 50 - th) / 2) + 18;
          const gx = ox + cw + gap, gy = oy + cw + 2 + gap;
          V.text(g, 'rᵀW', gx + (6 * (cw + 2)) / 2, oy - 6, { size: 11, font: 'mono', align: 'center', color: C.d });
          V.matrix(g, { x: gx, y: oy, data: row(K.rW), cell: cw, min: -1.6, max: 1.6, cmap: 'div', values: true, fmt: (q) => fmt(q, 2) });
          V.text(g, 'r', ox + cw / 2, gy - 6, { size: 11, font: 'mono', align: 'center', color: C.d });
          V.matrix(g, { x: ox, y: gy, data: col(r), cell: cw, min: -1.6, max: 1.6, cmap: 'div', values: true, fmt: (q) => fmt(q, 2) });
          const cur = n - 1, ci = Math.floor(cur / 6), cj = cur % 6;
          V.matrix(g, { x: gx, y: gy, data: K.dW.map((rw, i) => rw.map((q, j) => (i * 6 + j < n ? q : NaN))), cell: cw, min: -1.6, max: 1.6, cmap: 'div', values: true, fmt: (q) => fmt(q, 2), hiCells: n < 24 ? [[ci, cj]] : [], hiColor: C.d });
          if (n < 24) {
            const c1 = [gx + cj * (cw + 2) + cw / 2, oy + cw], c2 = [ox + cw, gy + ci * (cw + 2) + cw / 2];
            V.line(g, c1[0], c1[1], c1[0], gy + ci * (cw + 2), { color: V.alpha(C.d, 0.6), lw: 1.2, dash: [2, 3] });
            V.line(g, c2[0], c2[1], gx + cj * (cw + 2), c2[1], { color: V.alpha(C.d, 0.6), lw: 1.2, dash: [2, 3] });
          }
          V.text(g, 'ΔW[i, j] = α · rᵢ · (rᵀW)ⱼ', gx + (6 * (cw + 2)) / 2, gy + 4 * (cw + 2) + 16, { size: 11, font: 'mono', align: 'center', color: C.muted });
        } else if (i0 === 4) {
          blocks.push({ label: "W → W′", data: W.map((rw, i) => rw.map((q, j) => lerp(q, K.Wp[i][j], V.ease.inOut(f)))), hiRows: ri, glow: true });
          blocks.push({ label: 'ΔW', data: K.dW, faint: true });
        } else if (i0 === 5) {
          blocks.push({ label: "W′", data: K.Wp, hiRows: ri });
          blocks.push({ label: "W′z", data: col(K.Wpz) });
          blocks.push({ label: narrow ? 'y′ runtime' : "y′ (runtime)", data: col(K.yp), check: true });
          blocks.push({ label: "rᵀW′ = (1−α) rᵀW", data: row(K.rWp), accent: C.d, br: true });
        } else {
          blocks.push({ label: narrow ? 'rank 0: W′₀' : 'rank 0: W′₀ (columns 0–2)', data: K.Wp.map((rw) => rw.slice(0, 3)), rank: 0 });
          blocks.push({ label: narrow ? 'rank 1: W′₁' : 'rank 1: W′₁ (columns 3–5)', data: K.Wp.map((rw) => rw.slice(3)), rank: 1 });
          blocks.push({ label: narrow ? 'y′₀' : "y′₀ = W′₀z₀", data: col(K.y0), rank: 0, br: narrow });
          blocks.push({ label: narrow ? 'y′₁' : "y′₁ = W′₁z₁", data: col(K.y1), rank: 1 });
          blocks.push({ label: narrow ? 'Σ = y′' : "all-reduce: y′₀ + y′₁", data: col(K.y0.map((q, i) => q + K.y1[i] * clamp((e - 1) / 1.2, 0, 1))), check: e > 2.2 });
        }
        // flow layout (on mobile, picks the largest cell size that fits the stage)
        const pad = 14, gapX = narrow ? 14 : 22, gapY = 22;
        const lay = (cwT) => {
          const dims = blocks.map((b) => {
            const R = b.data.length, Kc = b.data[0].length;
            const lw = measure(g, b.label, 11, 'mono', 500);
            const bw = Math.max(Kc * (cwT + 2) - 2, lw);
            return { bw, bh: R * (cwT + 2) - 2 + 22, R, Kc };
          });
          const rows = [[]]; let xw = 0;
          blocks.forEach((b, i) => {
            const d = dims[i];
            if ((rows[rows.length - 1].length && xw + gapX + d.bw > w - 2 * pad) || (b.br && rows[rows.length - 1].length)) { rows.push([]); xw = 0; }
            rows[rows.length - 1].push(i); xw += (rows[rows.length - 1].length > 1 ? gapX : 0) + d.bw;
          });
          const rowH = rows.map((rr) => Math.max(...rr.map((i) => dims[i].bh)));
          const extra = blocks.some((b) => b.check) ? 22 : 0;
          const totalH = rowH.reduce((s, q) => s + q, 0) + gapY * (rows.length - 1) + extra;
          const maxW = Math.max(...dims.map((d) => d.bw));
          return { dims, rows, rowH, totalH, ok: maxW <= w - 2 * pad };
        };
        let L = lay(cw);
        if (narrow && blocks.length) {
          // largest cell that fits, penalizing extra rows (keeps "W · z = y" on a single row when possible)
          const minRows = lay(30).rows.length;
          let best = -1;
          for (let c = 46; c >= 30; c--) {
            const T = lay(c);
            if (!T.ok || T.totalH > h - 56 - 26) continue;
            const sc = c - 4 * (T.rows.length - minRows);
            if (sc > best) { best = sc; cw = c; L = T; }
          }
        }
        const { dims, rows, rowH, totalH } = L;
        let y = Math.max(56, 50 + (h - 50 - totalH) / 2);
        rows.forEach((rr, ri2) => {
          const tw = rr.reduce((s, i, k) => s + dims[i].bw + (k ? gapX : 0), 0);
          let x = (w - tw) / 2;
          rr.forEach((i) => {
            const b = blocks[i], d = dims[i];
            const mw = d.Kc * (cw + 2) - 2;
            const mx = x + (d.bw - mw) / 2, my = y + 22 + (rowH[ri2] - d.bh) / 2;
            const lc = b.rank === 0 ? C.a : b.rank === 1 ? C.b : b.accent || C.muted;
            V.text(g, b.label, x + d.bw / 2, my - 10, { size: 11, font: 'mono', align: 'center', color: lc, weight: 500 });
            if (b.rank != null) { g.strokeStyle = V.alpha(lc, 0.7); g.lineWidth = 1.5; g.setLineDash([4, 3]); V.rr(g, mx - 4, my - 4, mw + 8, d.R * (cw + 2) + 6, 8); g.stroke(); g.setLineDash([]); }
            if (b.glow) { const al = 0.25 + 0.2 * Math.sin(t * 4); ri.forEach((rr2) => { V.box(g, mx - 4, my + rr2 * (cw + 2) - 3, mw + 8, cw + 6, { r: 6, fill: V.alpha(C.d, al) }); }); }
            if (b.faint) g.globalAlpha = 0.45;
            const geo = V.matrix(g, { x: mx, y: my, data: b.data, cell: cw, min: -1.6, max: 1.6, cmap: 'div', values: true, fmt: (q) => fmt(q, 2), hiRow: b.hiRow, hiCol: b.hiCol, hiColor: C.d, hiCells: b.hiCells || [] });
            g.globalAlpha = 1;
            if (b.hiRows) b.hiRows.forEach((rr2) => { g.strokeStyle = C.d; g.lineWidth = 2; V.rr(g, mx - 2, my + rr2 * (cw + 2) - 2, mw + 4, cw + 4, 5); g.stroke(); });
            if (b.check) { V.text(g, '✓ equal', mx + mw / 2, my + geo.h + 16, { size: 12, align: 'center', color: C.e, weight: 600 }); }
            x += d.bw + gapX;
          });
          y += rowH[ri2] + gapY;
        });
        if (i0 === 6) V.text(g, 'no extra collective: the all-reduce already existed', w / 2, h - 10, { size: 11, align: 'center', color: C.muted });
      });
    })();

    /* ================================================================
       Figure 5 — per-layer edit map
       ================================================================ */
    (function () {
      const fig = root.querySelector('#fig-mapa');
      adaptH(fig, 440, 600);
      const ctl = UI.controls(fig);
      const S = { m: 'glm', on: true, mtp: true };
      const MODELS = {
        glm: {
          name: 'GLM-5.3-Flash', n: 46, trunk: 45,
          type: (i) => (i % 4 === 3 || i === 45 ? 'D' : 'K'),
          role: (i) => (i <= 14 ? 'anchor' : i <= 44 ? 'edit' : S.mtp ? 'mtp' : 'mtpoff'),
          attnW: () => 1, strength: 3, f: -2,
          drafter: { label: 'DFlash2 (never edited)', taps: [5, 14, 24, 33, 42] },
          info: () => (S.mtp ? '31 o_proj (22 KDA + 9 DSA) · 2,560 MiB BF16' : '30 o_proj (22 KDA + 8 DSA) · 2,432 MiB BF16'),
          fac: 'α = 3 → factor −2 (proj)',
        },
        qwen: {
          name: 'Qwen3.8-Flash-Next', n: 49, trunk: 48,
          type: (i) => (i === 48 || i % 4 === 3 ? 'Q' : 'G'),
          role: (i) => (i === 48 ? 'mtpstock' : i % 4 === 3 ? 'edit' : 'untouched'),
          attnW: (i) => (i % 4 === 3 ? 1.6 : 0.55), strength: 1, f: 0,
          drafter: { label: 'MTP (stock)', taps: [47] },
          info: () => '12 QSA o_proj [2560×6144] · 360 MiB BF16',
          fac: '“house” strength not published (curve with factor 0, illustrative)',
        },
        ds: {
          name: 'DeepSeek-V4-Flash', n: 46, trunk: 43,
          type: (i) => (i >= 43 || i <= 1 ? 'S' : i === 2 ? 'C' : i % 2 === 1 ? 'H' : 'C'),
          role: (i) => (i >= 43 ? 'draft' : i >= 10 ? 'edit' : 'anchor'),
          attnW: () => 1, strength: 3.5, f: -2.5,
          drafter: { label: 'DSpark reads 40–42', taps: [40, 41, 42] },
          info: () => '0 weight bytes · hook on 33 layers · 16 KiB fp32 v',
          fac: 'λ = 3.5 → factor −2.5 (runtime)',
        },
      };
      UI.seg(ctl, { label: 'Model', options: [{ v: 'glm', label: 'GLM' }, { v: 'qwen', label: 'Qwen' }, { v: 'ds', label: 'DeepSeek' }], value: 'glm', onChange: (v) => { S.m = v; ri.set(MODELS[v].info()); } });
      UI.toggle(ctl, { label: 'edit on', value: true, onChange: (v) => { S.on = v; } });
      UI.toggle(ctl, { label: 'include the MTP (GLM: ABLIT_INCLUDE_MTP)', value: true, onChange: (v) => { S.mtp = v; ri.set(MODELS[S.m].info()); } });
      const ri = UI.readout(ctl, 'what changes', MODELS.glm.info());
      vfig(fig, (g, w, h, t) => {
        const C = V.C();
        const M = MODELS[S.m];
        const narrow = w < 600;
        const cols = narrow ? 12 : Math.min(M.n, 24);
        const rowsN = Math.ceil(M.n / cols);
        const cs = Math.min(38, Math.floor((w - 32) / cols) - 3);
        const gx0 = (w - cols * (cs + 3)) / 2, gy0 = 30;
        V.text(g, M.name + ' · ' + M.fac, 16, 16, { size: 11, font: 'mono', color: C.muted, maxW: w - 32 });
        const cur = Math.floor(t * 2.6) % M.trunk;
        const centers = [];
        for (let i = 0; i < M.n; i++) centers[i] = [gx0 + (i % cols) * (cs + 3) + cs / 2, gy0 + Math.floor(i / cols) * (cs + 16) + cs / 2];
        const gridBottom = gy0 + rowsN * (cs + 16);
        // drafter (lines beneath the cells)
        if (S.m === 'glm') {
          const bw = Math.min(210, w * 0.46), bh = 26;
          const dX = w - 16 - bw, dY = gridBottom + 4;
          M.drafter.taps.forEach((k, j) => {
            const c0 = centers[k];
            const tx = dX + 16 + j * ((bw - 32) / 4);
            g.strokeStyle = V.alpha(C.b, 0.5); g.lineWidth = 1.2; g.setLineDash([2, 3]); g.lineDashOffset = -t * 20;
            g.beginPath(); g.moveTo(c0[0], c0[1]); g.bezierCurveTo(c0[0], c0[1] + cs + 10, tx, dY - 14, tx, dY); g.stroke(); g.setLineDash([]); g.lineDashOffset = 0;
          });
          V.box(g, dX, dY, bw, bh, { r: 8, fill: V.alpha(C.b, 0.16), stroke: C.b, lw: 1.3 });
          V.text(g, M.drafter.label, dX + bw / 2, dY + bh / 2 + 1, { size: 11, font: 'mono', align: 'center', base: 'middle', color: C.ink });
        } else {
          const tgt = S.m === 'ds' ? 43 : 48;
          M.drafter.taps.forEach((k) => {
            const a = centers[k], b = centers[tgt];
            g.strokeStyle = V.alpha(C.b, 0.75); g.lineWidth = 1.4; g.setLineDash([2, 3]); g.lineDashOffset = -t * 20;
            g.beginPath(); g.moveTo(a[0], a[1] - cs / 2); g.bezierCurveTo(a[0], a[1] - cs, b[0], b[1] - cs, b[0], b[1] - cs / 2); g.stroke(); g.setLineDash([]); g.lineDashOffset = 0;
          });
          V.text(g, M.drafter.label, 16, gridBottom + 16, { size: 11, font: 'mono', color: C.b });
        }
        for (let i = 0; i < M.n; i++) {
          const cx = centers[i][0] - cs / 2, cy = centers[i][1] - cs / 2;
          const role = M.role(i);
          let fill, stroke, dash = null;
          if (role === 'edit') { fill = S.on ? V.alpha(C.d, 0.78) : V.alpha(C.d, 0.08); stroke = C.d; if (!S.on) dash = [3, 3]; }
          else if (role === 'mtp') { fill = S.on ? V.alpha(C.f, 0.75) : V.alpha(C.f, 0.1); stroke = C.f; if (!S.on) dash = [3, 3]; }
          else if (role === 'mtpoff' || role === 'mtpstock') { fill = V.alpha(C.b, 0.14); stroke = V.alpha(C.b, 0.8); dash = [3, 3]; }
          else if (role === 'draft') { fill = V.alpha(C.b, 0.45); stroke = C.b; }
          else { fill = V.alpha(C.muted, role === 'anchor' ? 0.2 : 0.12); stroke = V.alpha(C.muted, 0.5); }
          V.rr(g, cx, cy, cs, cs, Math.min(6, cs / 4));
          g.fillStyle = V.C().panel || C.bg; g.fill();
          g.fillStyle = fill; g.fill();
          g.strokeStyle = stroke; g.lineWidth = i === cur ? 2.4 : 1; if (dash) g.setLineDash(dash); g.stroke(); g.setLineDash([]);
          V.text(g, M.type(i), cx + cs / 2, cy + cs / 2 + 1, { size: 11, font: 'mono', align: 'center', base: 'middle', color: role === 'edit' && S.on && !C.dark ? '#fff' : C.ink, weight: 600 });
          if (i % 5 === 0 || i === M.n - 1) V.text(g, String(i), cx + cs / 2, cy + cs + 11, { size: 11, font: 'mono', align: 'center', color: C.muted });
        }
        // legend
        let ly = gridBottom + (S.m === 'glm' ? 44 : 36);
        const leg = [[V.alpha(C.muted, 0.35), 'untouched/anchor'], [V.alpha(C.d, 0.78), 'edited'], [V.alpha(C.f, 0.75), 'edited MTP'], [V.alpha(C.b, 0.45), 'drafter']];
        let lx = 16;
        leg.forEach(([c, l]) => {
          const iw = 17 + measure(g, l, 11);
          if (lx > 16 && lx + iw > w - 16) { lx = 16; ly += 18; }   // wrap the legend onto another line instead of clipping it
          V.box(g, lx, ly - 8, 12, 12, { r: 3, fill: c }); V.text(g, l, lx + 17, ly + 2, { size: 11, color: C.muted });
          lx += iw + 19;
        });
        // illustrative curve
        const cx0 = 44, cx1 = w - 16, cy0 = ly + 18, cy1 = h - 22;
        if (cy1 - cy0 < 50) return;
        const nT = M.trunk;
        const bump = (i) => { const x = i / (nT - 1); return 0.04 + 0.5 * Math.exp(-Math.pow((x - 0.55) / 0.17, 2)); };
        let a1 = 0, a2 = 0;
        const stock = [], ed = [];
        for (let i = 0; i < nT; i++) {
          const aw = bump(i) * M.attnW(i), mw = 1.3 * bump(i);
          a1 += aw + mw;
          const edited = S.on && M.role(i) === 'edit';
          a2 += (edited ? M.f * aw : aw) + mw;
          stock.push(a1); ed.push(a2);
        }
        const vmax = Math.max(...stock), vmin = Math.min(0, ...ed);
        const X = (i) => cx0 + (i / (nT - 1)) * (cx1 - cx0), Y = (v) => cy1 - ((v - vmin) / (vmax - vmin)) * (cy1 - cy0 - 12);
        V.line(g, cx0, Y(0), cx1, Y(0), { color: V.alpha(C.muted, 0.5), lw: 1 });
        V.text(g, '0', cx0 - 6, Y(0) + 4, { size: 11, font: 'mono', color: C.muted, align: 'right' });
        V.text(g, 'h·r across depth (illustrative)', cx0, cy0 + 2, { size: 11, color: C.muted, font: 'mono' });
        const path = (arr, col, lw, dash) => { g.beginPath(); arr.forEach((v, i) => (i ? g.lineTo(X(i), Y(v)) : g.moveTo(X(i), Y(v)))); g.strokeStyle = col; g.lineWidth = lw; if (dash) g.setLineDash(dash); g.stroke(); g.setLineDash([]); };
        path(stock, V.alpha(C.muted, 0.9), 1.6, [5, 4]);
        path(ed, C.d, 2.6);
        V.text(g, 'stock', X(nT - 1) - 4, Y(stock[nT - 1]) + 16, { size: 11, align: 'right', color: C.muted, font: 'mono' });
        V.text(g, S.on ? 'edited' : 'edited (off)', X(nT - 1) - 4, Y(ed[nT - 1]) + (ed[nT - 1] < stock[nT - 1] - 0.3 ? 14 : -18), { size: 11, align: 'right', color: C.d, font: 'mono' });
        const q = cur;
        tokChar(g, X(q), Y(ed[q]) - 12, 16, C.c, { mood: 'o' });
        V.dot(g, X(q), Y(ed[q]), 3, C.d);
      });
    })();

    /* ================================================================
       Figure 6 — test against the null distribution
       ================================================================ */
    (function () {
      const fig = root.querySelector('#fig-nulo');
      adaptH(fig, 400, 520);
      const ctl = UI.controls(fig);
      const d = 64, m = 96, PLANT = [5, 17, 33, 48, 70, 88], BETA = 0.8;
      const S = { stat: 'max', cand: 'cand', c: 0.05, N: 1000, dC: 4096, seed: 3 };
      let W, rs, q, nullS, candS, stats;
      function unit(rg, n) { const u = new Float64Array(n); let s = 0; for (let i = 0; i < n; i++) { u[i] = rg.normal(); s += u[i] * u[i]; } s = Math.sqrt(s); for (let i = 0; i < n; i++) u[i] /= s; return u; }
      function build() {
        const rg = V.rng(1000 + S.seed);
        W = []; for (let i = 0; i < d; i++) { const row = new Float64Array(m); for (let j = 0; j < m; j++) row[j] = rg.normal() / Math.sqrt(d); W.push(row); }
        rs = unit(rg, d);
        PLANT.forEach((j) => { for (let i = 0; i < d; i++) W[i][j] += BETA * rs[i]; });
        // q ⟂ rs
        const qq = unit(rg, d); let dot = 0; for (let i = 0; i < d; i++) dot += qq[i] * rs[i];
        let s = 0; for (let i = 0; i < d; i++) { qq[i] -= dot * rs[i]; s += qq[i] * qq[i]; } s = Math.sqrt(s); for (let i = 0; i < d; i++) qq[i] /= s; q = qq;
      }
      const smax = (u) => { let best = 0; for (let j = 0; j < m; j++) { let a = 0; for (let i = 0; i < d; i++) a += u[i] * W[i][j]; if (Math.abs(a) > best) best = Math.abs(a); } return best; };
      function cosSample(rg, n) { // exact |cos|: g/sqrt(g² + χ²_{n−1}), χ² via Wilson–Hilferty
        const g1 = rg.normal(), k = n - 1, zz = rg.normal();
        const X = k * Math.pow(1 - 2 / (9 * k) + zz * Math.sqrt(2 / (9 * k)), 3);
        return Math.abs(g1 / Math.sqrt(g1 * g1 + Math.max(X, 1e-9)));
      }
      function computeNull() {
        const rg = V.rng(77 + S.seed * 13 + S.N);
        nullS = new Float64Array(S.N);
        if (S.stat === 'max') for (let k = 0; k < S.N; k++) nullS[k] = smax(unit(rg, d));
        else for (let k = 0; k < S.N; k++) nullS[k] = cosSample(rg, S.dC);
      }
      function computeCand() {
        if (S.stat === 'max') {
          if (S.cand === 'plant') candS = smax(rs);
          else if (S.cand === 'rand') candS = smax(unit(V.rng(555 + S.seed), d));
          else { const u = new Float64Array(d); const b = Math.sqrt(Math.max(0, 1 - S.c * S.c)); for (let i = 0; i < d; i++) u[i] = S.c * rs[i] + b * q[i]; candS = smax(u); }
        } else {
          candS = S.cand === 'plant' ? 1 : S.cand === 'rand' ? cosSample(V.rng(555 + S.seed), S.dC) : S.c;
        }
        let mu = 0; for (let k = 0; k < S.N; k++) mu += nullS[k]; mu /= S.N;
        let sd = 0; for (let k = 0; k < S.N; k++) sd += (nullS[k] - mu) * (nullS[k] - mu); sd = Math.sqrt(sd / S.N);
        let ge = 0; for (let k = 0; k < S.N; k++) if (nullS[k] >= candS) ge++;
        let mx = 0, mn = Infinity; for (let k = 0; k < S.N; k++) { if (nullS[k] > mx) mx = nullS[k]; if (nullS[k] < mn) mn = nullS[k]; }
        stats = { mu, sd, p: (1 + ge) / (1 + S.N), z: (candS - mu) / Math.max(sd, 1e-12), mx, mn };
        rC.set(fmt(candS, 3)); rP.set(stats.p < 0.001 ? '< 0.001' : fmt(stats.p, 3)); rZ.set(fmt(stats.z, 1));
      }
      let H;
      const all = () => { computeNull(); computeCand(); if (H) H.redraw(); };
      UI.seg(ctl, { label: 'Statistic', options: [{ v: 'max', label: 'max |uᵀW| (toy 64×96)' }, { v: 'cos', label: 'cosine with r*' }], value: 'max', onChange: (v) => { S.stat = v; all(); } });
      UI.seg(ctl, { label: 'Candidate', options: [{ v: 'plant', label: 'r* (the true one)' }, { v: 'cand', label: 'adjustable cos' }, { v: 'rand', label: 'random' }], value: 'cand', onChange: (v) => { S.cand = v; computeCand(); if (H) H.redraw(); } });
      UI.slider(ctl, { label: lab('{cos(candidate, r*)}'), min: 0, max: 1, step: 0.01, value: S.c, fmt: (x) => fmt(x, 2), onInput: (x) => { S.c = x; computeCand(); if (H) H.redraw(); } });
      UI.slider(ctl, { label: lab('{N} random'), min: 200, max: 3000, step: 100, value: S.N, onInput: (x) => { S.N = x; all(); } });
      UI.seg(ctl, { label: lab('{d} (cosine mode)'), options: [64, 512, 4096], value: 4096, onChange: (x) => { S.dC = x; if (S.stat === 'cos') all(); } });
      UI.button(ctl, '↻ redraw', () => { S.seed++; build(); all(); });
      const rC = UI.readout(ctl, lab('{s(candidate)}'), '—');
      const rP = UI.readout(ctl, lab('{p ≈}'), '—');
      const rZ = UI.readout(ctl, lab('{z}'), '—');
      build(); computeNull(); computeCand();
      H = vfig(fig, (g, w, h, t) => {
        const C = V.C();
        const nar = w < 600;
        const x0 = 30, x1 = w - 30, y0 = nar ? 114 : 100, y1 = h - 48;
        const lo = S.stat === 'max' ? Math.max(0, Math.min(stats.mn, candS) * 0.8) : 0;
        const hi = Math.max(stats.mx, candS) * 1.08 || 1;
        const NB = Math.max(20, Math.min(48, Math.floor((x1 - x0) / 12)));
        const bins = new Array(NB).fill(0);
        for (let k = 0; k < S.N; k++) { const b = Math.min(NB - 1, Math.floor((nullS[k] - lo) / (hi - lo) * NB)); bins[b]++; }
        const bmx = Math.max(...bins, 1);
        const bw = (x1 - x0) / NB;
        const X = (v) => x0 + (v - lo) / (hi - lo) * (x1 - x0);
        const title = S.stat === 'max' ? (nar ? 'max_j |uᵀ W[:, j]| · W 64×96 (toy)' : 'statistic: max_j |uᵀ W[:, j]|  ·  toy W 64 × 96, 6 columns write r*') : (nar ? '|cos(u, r*)| · d = ' + S.dC + ' · 1/√d = ' + fmt(1 / Math.sqrt(S.dC), 3) : 'statistic: |cos(u, r*)|  ·  d = ' + S.dC + '  ·  1/√d = ' + fmt(1 / Math.sqrt(S.dC), 3));
        V.text(g, title, 16, 18, { size: 11, font: 'mono', color: C.muted, maxW: w - 32 });
        V.text(g, 'N = ' + S.N + ' random unit vectors (the null cloud)', 16, 34, { size: 11, color: C.muted, maxW: nar ? w - 32 : w * 0.55 });
        for (let b = 0; b < NB; b++) {
          const hh = (bins[b] / bmx) * (y1 - y0);
          const left = x0 + b * bw;
          const tail = left + bw > X(candS);
          if (hh > 0) V.box(g, left + 1, y1 - hh, bw - 2, hh, { r: 2, fill: tail ? V.alpha(C.c, 0.75) : V.alpha(C.b, 0.55) });
        }
        V.line(g, x0, y1, x1, y1, { color: V.alpha(C.muted, 0.6), lw: 1 });
        for (let k = 0; k <= 4; k++) { const v = lo + (hi - lo) * k / 4; V.text(g, fmt(v, 2), X(v), y1 + 15, { size: 11, font: 'mono', align: 'center', color: C.muted }); }
        // mean ± sd
        const mX = X(stats.mu);
        V.line(g, mX, y0 + 24, mX, y1, { color: V.alpha(C.b, 0.8), lw: 1, dash: [3, 3] });
        V.text(g, '▲ null mean', mX, y1 + 32, { size: 11, color: C.b, font: 'mono', align: 'center' });
        // candidate
        const cX = X(candS);
        const pulse = 0.5 + 0.5 * Math.sin(t * 3);
        // candidate label above the bars, with an opaque background; the line starts right below it
        const cl = (S.cand === 'plant' ? 'r*' : S.cand === 'rand' ? 'random' : (nar ? 'cand. (cos ' : 'candidate (cos ') + fmt(S.c, 2) + ')') + ' = ' + fmt(candS, 3);
        const tw0 = measure(g, cl, 11, 'mono', 600) + 12;
        const tX = clamp(cX, tw0 / 2 + 6, w - tw0 / 2 - 6), tY = y0 - 16;
        V.glow(g, cX, y0 + 16, 18, C.d, 0.5 + 0.4 * pulse);
        V.line(g, cX, tY + 10, cX, y1, { color: C.d, lw: 2.6 });
        V.box(g, tX - tw0 / 2, tY - 9.5, tw0, 19, { r: 6, fill: C.panel });
        tag(g, cl, tX, tY, C.d, { align: 'center', size: 11, weight: 600, fill: V.alpha(C.d, 0.22) });
        const verdict = stats.p < 0.01 ? 'stands out from chance (p < 0.01)' : 'indistinguishable from chance';
        V.text(g, verdict, w - 16, nar ? 54 : 36, { size: 13, align: 'right', color: stats.p < 0.01 ? C.e : C.c, weight: 600 });
        V.text(g, 'z = ' + fmt(stats.z, 1) + '   ' + (stats.p < 0.001 ? 'p < 0.001' : 'p ≈ ' + fmt(stats.p, 3)), w - 16, nar ? 72 : 54, { size: 12, align: 'right', color: C.muted, font: 'mono' });
      }, { static: false });
    })();

    /* ================================================================
       Figure 7 — range requests into safetensors
       ================================================================ */
    (function () {
      const fig = root.querySelector('#fig-range');
      adaptH(fig, 400, 480);
      const ctl = UI.controls(fig);
      const SHARD = { 15: 18, 16: 20, 17: 23, 18: 26, 19: 29, 20: 32, 21: 34, 22: 37, 23: 40, 24: 43, 25: 45, 26: 48, 27: 51, 28: 54, 29: 57, 30: 62, 31: 65, 32: 68, 33: 70, 34: 73, 35: 76, 36: 79, 37: 81, 38: 84, 39: 87, 40: 92, 41: 95, 42: 98, 43: 101, 44: 103, 45: 106 };
      const isD = (L) => L % 4 === 3 || L === 45;
      const MiB = (L) => (isD(L) ? 128 : 64);
      const S = { L: 15, t0: -2.4 };
      let H;
      const rs = UI.readout(ctl, 'tensor', '');
      const rb = UI.readout(ctl, 'cumulative', '');
      const upd = () => {
        let acc = 0; for (let L = 15; L <= S.L; L++) acc += MiB(L);
        rs.set((isD(S.L) ? 'DSA [4096, 16384]' : 'KDA [4096, 8192]') + ' · ' + MiB(S.L) + ' MiB');
        rb.set(fmtInt(acc) + ' MiB of 2,560');
      };
      UI.slider(ctl, { label: 'layer', min: 15, max: 45, step: 1, value: 15, onInput: (v) => { S.L = v; S.t0 = H ? H.t : 0; upd(); } });
      UI.button(ctl, '▶ fetch again', () => { S.t0 = H ? H.t : 0; }, { primary: true });
      upd();
      H = vfig(fig, (g, w, h, t) => {
        const C = V.C();
        const narrow = w < 600;
        const e = t - S.t0;
        const L = S.L, D = isD(L);
        const pad = 16;
        // Hub and disk
        const bh = 34, hubW = Math.min(narrow ? w * 0.52 : 290, w * 0.5), dskW = Math.min(narrow ? w * 0.38 : 230, w * 0.4);
        V.box(g, pad, 12, hubW, bh, { r: 17, fill: V.alpha(C.b, 0.16), stroke: C.b, lw: 1.3 });
        V.text(g, narrow ? 'Hub · NVFP4 donor' : 'Hub · dealignai/GLM-5.3-Flash-UNCENSORED-NVFP4', pad + hubW / 2, 12 + bh / 2 + 1, { size: 11, font: 'mono', align: 'center', base: 'middle', color: C.ink, maxW: hubW - 16 });
        const dX = w - pad - dskW;
        V.box(g, dX, 12, dskW, bh, { r: 8, fill: V.alpha(C.a, 0.14), stroke: C.a, lw: 1.3 });
        V.text(g, 'ablit/transplant/L' + L + '.bin', dX + dskW / 2, 12 + bh / 2 + 1, { size: 11, font: 'mono', align: 'center', base: 'middle', color: C.ink, maxW: dskW - 12 });
        // shard
        const sy = 76, sh = 34, sx0 = pad, sx1 = w - pad, sw = sx1 - sx0;
        const shardName = 'model-' + String(SHARD[L]).padStart(5, '0') + '-of-00120.safetensors';
        V.text(g, shardName + '  (~1.4 GiB; illustrative layout)', sx0, sy - 8, { size: 11, font: 'mono', color: C.muted, maxW: sw });
        const rg = V.rng(L * 31 + 7);
        const frac = (MiB(L) / 1024) / 1.37;
        const posF = 0.18 + rg() * 0.55;
        const lenW = Math.max(12, sw * 0.014), hdrW = Math.max(40, sw * 0.07);
        // segments
        V.box(g, sx0, sy, lenW, sh, { r: 3, fill: V.alpha(C.f, 0.7) });
        V.box(g, sx0 + lenW + 2, sy, hdrW, sh, { r: 3, fill: V.alpha(C.b, 0.6) });
        let x = sx0 + lenW + hdrW + 4;
        const oX = sx0 + sw * posF, oW = Math.max(8, sw * frac);
        const rg2 = V.rng(L);
        while (x < sx1 - 2) {
          const segW = Math.min(sx1 - x, 6 + rg2() * sw * 0.07);
          if (x < oX + oW && x + segW > oX) { x = oX + oW + 2; continue; }
          V.box(g, x, sy, segW - 2, sh, { r: 2, fill: V.alpha(C.muted, 0.16 + 0.1 * rg2()) });
          x += segW;
        }
        V.box(g, oX, sy, oW, sh, { r: 3, fill: V.alpha(C.d, 0.85), stroke: C.d });
        V.text(g, '8 B', sx0 + lenW / 2, sy + sh + 13, { size: 11, font: 'mono', align: 'center', color: C.f });
        V.text(g, 'JSON', sx0 + lenW + 2 + hdrW / 2, sy + sh + 13, { size: 11, font: 'mono', align: 'center', color: C.b });
        V.text(g, 'o_proj L' + L, oX + oW / 2, sy + sh + 13, { size: 11, font: 'mono', align: 'center', color: C.d });
        // requests
        const ph = [0.9, 1.8, 3.6, 4.4];
        const reqs = [
          ['Range: bytes=0-7', '→ N (u64 little-endian) = header size'],
          ['Range: bytes=8-(8+N−1)', '→ JSON: dtype, shape, data_offsets [a, b]'],
          ['Range: bytes=(8+N+a)-(8+N+b−1)', '→ ' + MiB(L) + ' MiB of BF16 (' + (D ? '4096 × 16384' : '4096 × 8192') + ' × 2 B)'],
          ['sha256 → MANIFEST.json', '→ .tmp and atomic replace ✓'],
        ];
        const ry0 = sy + sh + 34, rlh = narrow ? 34 : 22;
        reqs.forEach((rq, i) => {
          const started = e > (i ? ph[i - 1] : 0), done = e > ph[i];
          const yy = ry0 + i * rlh;
          const col = done ? C.e : started ? C.d : V.alpha(C.muted, 0.6);
          V.dot(g, pad + 5, yy - 4, 4, col);
          V.text(g, rq[0], pad + 16, yy, { size: 11, font: 'mono', color: started ? C.ink : C.muted, maxW: narrow ? w - 40 : w * 0.42 });
          V.text(g, rq[1], narrow ? pad + 16 : pad + 16 + w * 0.44, narrow ? yy + 15 : yy, { size: 11, color: C.muted, maxW: narrow ? w - 40 : w * 0.5 - 20 });
        });
        // JSON header (after request 1)
        const jy = ry0 + 4 * rlh + (narrow ? 8 : 4);
        const jh = narrow ? 64 : 46;
        if (e > ph[0] && jy + jh < h - 90) {
          const act = e > ph[0] && e < ph[1];
          V.box(g, pad, jy, w - 2 * pad, jh, { r: 8, fill: V.alpha(C.b, act ? 0.16 : 0.07), stroke: V.alpha(C.b, act ? 0.9 : 0.4), lw: 1 });
          const key = narrow ? '"…layers.' + L + '.self_attn.o_proj.weight": {' : '{ "model.language_model.layers.' + L + '.self_attn.o_proj.weight":';
          const val = '"dtype": "BF16", "shape": [4096, ' + (D ? '16384' : '8192') + '], "data_offsets": [a, b] }';
          V.text(g, key, pad + 10, jy + 18, { size: 11, font: 'mono', color: C.ink, maxW: w - 2 * pad - 20 });
          if (narrow) { V.text(g, '  "dtype": "BF16", "shape": [4096, ' + (D ? '16384' : '8192') + '],', pad + 10, jy + 35, { size: 11, font: 'mono', color: C.b, maxW: w - 2 * pad - 20 }); V.text(g, '  "data_offsets": [a, b] }, …', pad + 10, jy + 52, { size: 11, font: 'mono', color: C.b, maxW: w - 2 * pad - 20 }); }
          else V.text(g, '    { ' + val + ', … }', pad + 10, jy + 36, { size: 11, font: 'mono', color: C.b, maxW: w - 2 * pad - 20 });
        }
        // particles flying from the shard to disk during request 3
        const pStart = ph[1], pEnd = ph[2];
        if (e > pStart && e < pEnd + 0.3) {
          for (let k = 0; k < 14; k++) {
            const u = ((e - pStart) * 0.9 + k / 14) % 1;
            const bx = V.bez([oX + oW / 2, sy], [oX + oW / 2, sy - 30], [dX + dskW / 2, 12 + bh + 30], [dX + dskW / 2, 12 + bh], u);
            V.dot(g, bx[0], bx[1], 2.6, V.alpha(C.d, 0.9));
          }
        }
        if (e > ph[0] && e < ph[1]) { V.glow(g, sx0 + lenW + hdrW / 2, sy + sh / 2, 26, C.b, 0.6); }
        // progress of the 31 tensors and byte comparison
        const gy = h - 76;
        const cells = 31, cwid = Math.min(18, (w - 2 * pad) / cells - 2);
        V.text(g, '31 tensors (L15–L45):', pad, gy - 8, { size: 11, color: C.muted });
        for (let i = 0; i < cells; i++) {
          const Li = 15 + i, done = Li < L || (Li === L && e > ph[2]);
          V.box(g, pad + i * (cwid + 2), gy, cwid, 12, { r: 2, fill: done ? (isD(Li) ? C.d : V.alpha(C.d, 0.6)) : V.alpha(C.muted, 0.15), stroke: Li === L ? C.ink : null, lw: 1 });
        }
        let acc = 0; for (let Li = 15; Li <= L; Li++) acc += MiB(Li) * (Li === L ? clamp((e - ph[1]) / (ph[2] - ph[1]), 0, 1) : 1);
        const full = 164 * 1024;
        const barW = w - 2 * pad - 120;
        const by = h - 44;
        V.text(g, 'transplant', pad, by + 9, { size: 11, color: C.muted, font: 'mono' });
        V.box(g, pad + 110, by, barW, 11, { r: 3, fill: V.alpha(C.muted, 0.12) });
        V.box(g, pad + 110, by, Math.max(2, barW * acc / full), 11, { r: 3, fill: C.d });
        V.text(g, fmtInt(acc) + ' MiB', pad + 116 + Math.max(2, barW * acc / full), by + 9, { size: 11, font: 'mono', color: C.d });
        V.text(g, 'whole donor', pad, by + 27, { size: 11, color: C.muted, font: 'mono' });
        V.box(g, pad + 110, by + 18, barW, 11, { r: 3, fill: V.alpha(C.b, 0.55) });
        V.text(g, '~164 GiB', pad + 110 + barW - 4, by + 27, { size: 11, font: 'mono', color: C.ink, align: 'right' });
      });
    })();

    /* ================================================================
       Figure 8 — traffic lights of the launcher (DeepSeek)
       ================================================================ */
    (function () {
      const fig = root.querySelector('#fig-portoes');
      adaptH(fig, 480, 550);
      const ctl = UI.controls(fig);
      const S = { abl: true, manual: false, layers: '10-42', lam: 3.5, files: true, integ: 'ok', run: null, last: null, lastStamp: null };
      const STAMP0 = (cfg) => cfg.abl
        ? ['dsv4-runtime-ablation-v1', 'enabled=1', 'lambda=' + cfg.lam, 'layers=' + cfg.layers, 'direction=6e4d8a8f…', 'patch=sha(hotfix)']
        : ['dsv4-runtime-ablation-v1', 'enabled=0', 'lambda=-', 'layers=-', 'direction=-', 'patch=sha(hotfix)'];
      S.lastStamp = STAMP0({ abl: false });
      function evaluate(cfg) {
        const G = [];
        const push = (name, status, msg) => G.push({ name, status, msg });
        let stop = false, exit = null;
        const ablate = cfg.abl ? 1 : cfg.manual ? 'X' : 0;
        if (ablate === 'X') { push('ABLITERATED → ABLATE', 'fail', 'exit 2: ABLATE=1 on its own is blocked'); stop = true; exit = 'exit 2'; }
        else push('ABLITERATED → ABLATE', 'ok', ablate ? 'ABLATE=1' : 'ABLATE=0 (stock mode)');
        const stages = ['DSV4_ABLATE_LAYERS range', 'λ finite and ≥ 0', 'RESPONSIBLE_USE.md + direction in cache', 'Source SHA-256', 'Staging head → worker'];
        if (!stop && !ablate) stages.forEach((s) => push(s, 'skip', 'skipped: ablation off'));
        if (!stop && ablate) {
          const mt = cfg.layers.match(/^(\d+)-(\d+)$/);
          const lo = +mt[1], hi = +mt[2];
          if (lo > hi) { push(stages[0], 'fail', 'exit 2: inverted range'); stop = true; exit = 'exit 2'; }
          else if (hi > 42) { push(stages[0], 'fail', 'exit 2: only 0–42 (43–45 = DSpark stages)'); stop = true; exit = 'exit 2'; }
          else push(stages[0], lo < 10 ? 'warn' : 'ok', lo < 10 ? 'valid, but no anchor layers' : lo + '–' + hi + ' (' + (hi - lo + 1) + ' layers)');
          if (!stop) push(stages[1], cfg.lam >= 4 ? 'warn' : cfg.lam === 0 ? 'warn' : 'ok', cfg.lam >= 4 ? 'valid, but λ ≥ 4 risks long-CoT degeneration' : cfg.lam === 0 ? 'λ = 0: hook with no effect' : 'λ = ' + fmt(cfg.lam, 1) + ' → factor ' + fmt(1 - cfg.lam, 1));
          if (!stop) { if (cfg.files) push(stages[2], 'ok', 'downloaded by prepare --abliterated'); else { push(stages[2], 'fail', 'exit 1: accept the gated terms and run prepare --abliterated'); stop = true; exit = 'exit 1'; } }
          if (!stop) { if (cfg.integ === 'src') { push(stages[3], 'fail', 'exit 1: SHA-256 ≠ 6e4d8a8f…'); stop = true; exit = 'exit 1'; } else push(stages[3], 'ok', '6e4d8a8f… matches'); }
          if (!stop) { if (cfg.integ === 'worker') { push(stages[4], 'fail', 'failure: SHA-256 of the worker copy does not match'); stop = true; exit = 'staging failure'; } else push(stages[4], 'ok', '.tmp → sha256 → atomic mv (head and worker)'); }
        }
        if (!stop) push('Entrypoint (compose:429)', 'ok', ablate ? 'exports DSV4_ABLATE_FILE=/cache/…/direction_r1.pt' : 'DSV4_ABLATE_FILE empty');
        if (!stop) push('hotfix-dsv4-runtime-ablation', 'ok', ablate ? 'buffer in 43 layers; hook on ' + cfg.layers : 'check mode: model.py stock');
        const stamp = STAMP0(cfg);
        let changed = false;
        if (!stop) {
          changed = stamp.some((s, i) => s !== S.lastStamp[i]);
          push('AOT stamp', changed ? 'warn' : 'ok', changed ? 'changed: torch_compile_cache wiped and recompiled' : 'unchanged: cache reused');
          push('Serving', 'ok', ablate ? 'y ← y − λ(y·v)v on layers ' + cfg.layers : 'stock forward');
        }
        return { G, stop, exit, stamp, changed };
      }
      let H;
      const run = () => { S.run = { t0: H ? H.t : 0, res: evaluate(S), prev: S.lastStamp.slice(), applied: false }; };
      UI.toggle(ctl, { label: 'ABLITERATED=1', value: true, onChange: (v) => { S.abl = v; } });
      UI.toggle(ctl, { label: 'ABLATE=1 exported by hand', value: false, onChange: (v) => { S.manual = v; } });
      UI.seg(ctl, { label: 'DSV4_ABLATE_LAYERS', options: ['10-42', '0-42', '10-45', '42-10'], value: '10-42', onChange: (v) => { S.layers = v; } });
      UI.slider(ctl, { label: lab('{λ} (LAMBDA)'), min: 0, max: 6, step: 0.5, value: 3.5, fmt: (x) => fmt(x, 1), onInput: (x) => { S.lam = x; } });
      UI.toggle(ctl, { label: 'terms + direction in cache', value: true, onChange: (v) => { S.files = v; } });
      UI.seg(ctl, { label: 'Integrity', options: [{ v: 'ok', label: 'intact' }, { v: 'src', label: 'corrupted source' }, { v: 'worker', label: 'truncated worker copy' }], value: 'ok', onChange: (v) => { S.integ = v; } });
      UI.button(ctl, '▶ recreate the containers', run, { primary: true });
      const STEP = 0.42;
      H = vfig(fig, (g, w, h, t) => {
        const C = V.C();
        if (!S.run) { run(); S.run.t0 = t - 10; }
        const R = S.run.res;
        const narrow = w < 640;
        const e = t - S.run.t0;
        const lastIdx = R.G.length - 1;
        const reached = Math.floor(e / STEP);
        const finished = reached >= lastIdx;
        if (finished && !S.run.applied) { S.run.applied = true; if (!R.stop) S.lastStamp = R.stamp; }
        const listW = narrow ? w - 24 : w * 0.6;
        const x0 = 16, y0 = 24;
        const nSt = 10;
        const rowH = Math.min(40, ((narrow ? h - 150 : h) - y0 - 12) / nSt);
        const trackX = x0 + 12, lightX = x0 + 38, textX = x0 + 56;
        V.text(g, 'start → entrypoint → hotfix → vLLM', x0, 13, { size: 11, color: C.muted, font: 'mono' });
        // track with sleepers
        V.line(g, trackX, y0 + rowH / 2, trackX, y0 + rowH * (nSt - 0.5), { color: V.alpha(C.muted, 0.45), lw: 3 });
        for (let k = 0; k < nSt * 3; k++) { const yy = y0 + rowH / 2 + k * rowH / 3; if (yy > y0 + rowH * (nSt - 0.5)) break; V.line(g, trackX - 6, yy, trackX + 6, yy, { color: V.alpha(C.muted, 0.25), lw: 1.5 }); }
        const names = ['ABLITERATED → ABLATE', 'DSV4_ABLATE_LAYERS range', 'λ finite and ≥ 0', 'RESPONSIBLE_USE.md + direction in cache', 'Source SHA-256', 'Staging head → worker', 'Entrypoint (compose:429)', 'hotfix-dsv4-runtime-ablation', 'AOT stamp', 'Serving'];
        names.forEach((nm, i) => {
          const gi = R.G[i];
          const yy = y0 + rowH * (i + 0.5);
          const vis = gi && i <= reached;
          const st = vis ? gi.status : gi ? 'pending' : 'never';
          const col = st === 'ok' ? C.e : st === 'fail' ? RED() : st === 'warn' ? C.d : st === 'skip' ? V.alpha(C.muted, 0.55) : V.alpha(C.muted, 0.22);
          if (st === 'fail') V.glow(g, lightX, yy, 20, RED(), 0.8);
          else if (st === 'ok' || st === 'warn') V.glow(g, lightX, yy, 13, col, 0.45);
          V.box(g, lightX - 9, yy - 9, 18, 18, { r: 5, fill: V.alpha(C.muted, 0.12), stroke: V.alpha(C.muted, 0.35), lw: 1 });
          V.dot(g, lightX, yy, 6, col);
          const two = rowH > 30;
          V.text(g, nm, textX, two ? yy - 4 : yy, { size: 12, color: vis ? C.ink : C.muted, weight: vis ? 600 : 400, maxW: listW - textX - 4, base: two ? 'alphabetic' : 'middle' });
          if (vis && two) V.text(g, gi.msg, textX, yy + 12, { size: 11, color: st === 'fail' ? RED() : st === 'warn' ? C.d : C.muted, font: 'mono', maxW: listW - textX - 4 });
        });
        // train
        const pos = Math.min(e / STEP, lastIdx);
        const ty = y0 + rowH * (pos + 0.5);
        const sad = R.stop && finished;
        tokChar(g, trackX, ty + (finished ? 0 : Math.sin(t * 9) * 1.2), 18, sad ? RED() : C.a, { mood: sad ? 'sad' : 'happy' });
        // stamp panel
        const px = narrow ? 16 : listW + 20, py = narrow ? y0 + rowH * nSt + 18 : 34, pw = narrow ? w - 32 : w - listW - 36;
        V.text(g, 'AOT cache stamp', px, py, { size: 12, color: C.ink, weight: 600 });
        V.text(g, 'VLLM_CACHE_ROOT/.dsv4_ablate_stamp', px, py + 16, { size: 11, color: C.muted, font: 'mono', maxW: pw });
        const prev = S.run.prev;
        const fields = R.stamp;
        const fy = py + 30, fh = narrow ? 17 : 19;
        const colW2 = narrow ? pw / 2 : pw;
        fields.forEach((f, i) => {
          const ch = prev[i] !== f;
          const yy = narrow ? fy + (i % 3) * fh : fy + i * fh;
          const xx = narrow ? px + Math.floor(i / 3) * colW2 : px;
          V.text(g, (ch ? '● ' : '  ') + f, xx, yy + 12, { size: 11, font: 'mono', color: ch ? C.d : C.muted, weight: ch ? 600 : 400, maxW: colW2 - 6 });
        });
        const cy = narrow ? fy + 3 * fh + 20 : fy + fields.length * fh + 22;
        const anyCh = fields.some((f, i) => f !== prev[i]);
        const cacheMsg = !finished ? 'waiting for boot…' : R.stop ? 'boot aborted: ' + R.exit : anyCh ? 'changed → torch_compile_cache wiped' : 'unchanged → cache reused';
        tag(g, cacheMsg, px, cy, !finished ? C.muted : R.stop ? RED() : anyCh ? C.d : C.e, { size: 11, font: 'mono' });
        if (!narrow) {
          V.text(g, 'compared with the last successful boot:', px, cy + 30, { size: 11, color: C.muted, maxW: pw });
          V.text(g, prev[1] + ' · ' + prev[2] + ' · ' + prev[3], px, cy + 46, { size: 11, font: 'mono', color: C.muted, maxW: pw });
          V.text(g, '● = field that changed', px, cy + 66, { size: 11, color: C.d });
        }
      });
    })();

    /* ================================================================
       Figure 9 — weighing the edit (toy KLD)
       ================================================================ */
    (function () {
      const fig = root.querySelector('#fig-balanca');
      adaptH(fig, 420, 640, 640);
      const ctl = UI.controls(fig);
      const P = {
        sens: { toks: ['Sorry', "I can't", 'Unfortunately', 'Sure', 'Here is', 'First', "Let's", 'Well'], base: [2.6, 2.3, 1.7, 0.2, 0.1, 0.0, -0.1, -0.3], rho: [0.9, 0.85, 0.7, -0.35, -0.4, -0.3, -0.25, -0.05], dl: [0.3, -0.5, 0.6, -0.4, 0.5, 0.7, -0.6, 0.4], cO: 0.8, cA: 2.2, ref: [0, 1, 2] },
        neut: { toks: ['Paris', 'a', 'the', 'Lyon', 'known', 'Sorry', 'located', 'Rome'], base: [4.2, 1.7, 1.5, 0.9, 1.0, -1.2, 0.8, -0.4], rho: [0, 0, 0, 0, 0, 0.9, 0, 0], dl: [-0.9, 0.6, 0.4, 0.7, 0.5, 0.3, 0.8, 0.6], cO: 0.1, cA: 0.25, ref: [5] },
      };
      const S = { lam: 3.5, pr: 'sens' };
      const sm = (xs) => V.softmax(xs);
      function dist(pr, lam) {
        const q = P[pr];
        const c0 = q.cO + q.cA, c = q.cO + (1 - lam) * q.cA;
        const drift = 0.06 * lam * lam;
        return sm(q.base.map((b, i) => b + q.rho[i] * (c - c0) + drift * q.dl[i]));
      }
      const KL = (p, q) => p.reduce((s, pi, i) => s + (pi > 0 ? pi * Math.log(pi / q[i]) : 0), 0);
      const refMass = (pr, d) => P[pr].ref.reduce((s, i) => s + d[i], 0);
      const CURV = Array.from({ length: 51 }, (_, k) => {
        const L = k * 0.1;
        const ps = dist('sens', 0), qs = dist('sens', L), pn = dist('neut', 0), qn = dist('neut', L);
        return { L, B: clamp(1 - refMass('sens', qs) / refMass('sens', ps), 0, 1), K: KL(pn, qn) };
      });
      const KMAX = Math.max(...CURV.map((c) => c.K));
      let H;
      const rK = UI.readout(ctl, lab('{KL(p‖q)} on this prompt'), '—');
      const rM = UI.readout(ctl, 'mass on refusal tokens', '—');
      function upd() {
        const p = dist(S.pr, 0), q = dist(S.pr, S.lam);
        rK.set(fmt(KL(p, q), 3) + ' nats');
        rM.set(fmt(refMass(S.pr, p) * 100, 0) + '% → ' + fmt(refMass(S.pr, q) * 100, 0) + '%');
        if (H) H.redraw();
      }
      UI.slider(ctl, { label: lab('strength {λ}'), min: 0, max: 5, step: 0.1, value: 3.5, fmt: (x) => fmt(x, 1), onInput: (x) => { S.lam = x; upd(); } });
      UI.seg(ctl, { label: 'Prompt in the bars', options: [{ v: 'sens', label: 'sensitive request' }, { v: 'neut', label: '“The capital of France is”' }], value: 'sens', onChange: (v) => { S.pr = v; upd(); } });
      H = vfig(fig, (g, w, h, t) => {
        const C = V.C();
        const narrow = w < 640;
        const q0 = P[S.pr];
        const p = dist(S.pr, 0), q = dist(S.pr, S.lam);
        // horizontal bars
        const bw = narrow ? w - 32 : w * 0.5 - 24;
        const bx = 16 + 96, by = 40, rh = narrow ? 24 : Math.min(38, (h - 80) / 8);
        V.text(g, S.pr === 'sens' ? 'next token · sensitive request' : 'next token · neutral text', 16, 16, { size: 11, color: C.muted, font: 'mono' });
        V.text(g, 'outline = original · filled = edited', 16, 31, { size: 11, color: C.muted });
        const barL = bw - 96 - 40;
        q0.toks.forEach((tk, i) => {
          const yy = by + i * rh;
          const isRef = q0.ref.includes(i);
          V.text(g, tk, bx - 8, yy + rh / 2, { size: 12, align: 'right', base: 'middle', color: isRef ? C.c : C.ink, maxW: 90 });
          const a = p[i] * barL, b = q[i] * barL;
          V.box(g, bx, yy + rh * 0.2, Math.max(1, b), rh * 0.6, { r: 3, fill: V.alpha(isRef ? C.c : C.a, 0.75) });
          g.strokeStyle = V.alpha(C.ink, 0.6); g.lineWidth = 1.2; V.rr(g, bx, yy + rh * 0.2, Math.max(1, a), rh * 0.6, 3); g.stroke();
          V.text(g, fmt(q[i] * 100, 0) + '%', bx + Math.max(a, b) + 6, yy + rh / 2 + 1, { size: 11, font: 'mono', base: 'middle', color: C.muted });
        });
        // balance scale
        const bx0 = narrow ? 0 : w * 0.5, by0 = narrow ? by + 8 * rh + 16 : 0, bW = narrow ? w : w * 0.5, bH = narrow ? h - by0 : h;
        const ps = dist('sens', 0), qs = dist('sens', S.lam), pn = dist('neut', 0), qn = dist('neut', S.lam);
        const B = clamp(1 - refMass('sens', qs) / refMass('sens', ps), 0, 1);
        const Dm = clamp(KL(pn, qn) / 0.35, 0, 1);
        const ang = clamp((Dm - B) * 0.42, -0.36, 0.36);
        const cx = bx0 + bW / 2, top = by0 + 50, arm = Math.min(bW * 0.34, 140);
        V.text(g, 'the scale (illustrative)', cx, by0 + 18, { size: 12, align: 'center', color: C.ink, weight: 600 });
        // pedestal
        g.fillStyle = V.alpha(C.muted, 0.5); g.beginPath(); g.moveTo(cx, top); g.lineTo(cx - 18, top + 118); g.lineTo(cx + 18, top + 118); g.closePath(); g.fill();
        const ex = Math.cos(ang) * arm, ey = Math.sin(ang) * arm;
        V.line(g, cx - ex, top - ey, cx + ex, top + ey, { color: C.ink, lw: 3 });
        V.dot(g, cx, top, 5, C.d);
        const pan = (px, py, val, col, lab, sub) => {
          const ph2 = 46;
          V.line(g, px, py, px - 30, py + ph2, { color: V.alpha(C.muted, 0.8), lw: 1 });
          V.line(g, px, py, px + 30, py + ph2, { color: V.alpha(C.muted, 0.8), lw: 1 });
          g.fillStyle = V.alpha(col, 0.25); g.strokeStyle = col; g.lineWidth = 1.5;
          g.beginPath(); g.ellipse(px, py + ph2, 38, 9, 0, 0, Math.PI); g.fill(); g.stroke();
          const n = Math.round(val * 6);
          for (let k = 0; k < n; k++) V.box(g, px - 14 + (k % 3) * 10 - 1, py + ph2 - 10 - Math.floor(k / 3) * 10, 9, 9, { r: 2, fill: col });
          V.text(g, lab, px, py + ph2 + 26, { size: 11, align: 'center', color: col, weight: 600, maxW: arm + 20 });
          V.text(g, sub, px, py + ph2 + 42, { size: 11, align: 'center', color: C.muted, font: 'mono' });
        };
        pan(cx - ex, top - ey, B, C.a, 'refusals removed', fmt(B * 100, 0) + '%');
        pan(cx + ex, top + ey, Dm, C.c, 'KLD on neutral text', fmt(KL(pn, qn), 3) + ' nats');
        const zone = S.lam >= 4 ? 'risk zone (λ ≥ 4)' : S.lam > 1.6 ? 'small gain, growing damage' : S.lam >= 0.8 ? 'intended effect grows fast' : 'little effect';
        // curves as a function of λ
        const gx0 = bx0 + 34, gx1 = bx0 + bW - 16, gy0 = top + 176, gy1 = by0 + bH - 34;
        if (gy1 - gy0 > 50) {
          const X = (L) => gx0 + (L / 5) * (gx1 - gx0);
          const Yb = (b) => gy1 - b * (gy1 - gy0), Yk = (k) => gy1 - (k / KMAX) * (gy1 - gy0);
          g.fillStyle = V.alpha(RED(), 0.08); g.fillRect(X(4), gy0, X(5) - X(4), gy1 - gy0);
          V.line(g, gx0, gy1, gx1, gy1, { color: V.alpha(C.muted, 0.5), lw: 1 });
          [0, 1, 2, 3, 4, 5].forEach((L) => V.text(g, String(L), X(L), gy1 + 13, { size: 11, font: 'mono', align: 'center', color: C.muted }));
          const path = (fy, col) => { g.beginPath(); CURV.forEach((c, k) => (k ? g.lineTo(X(c.L), fy(c)) : g.moveTo(X(c.L), fy(c)))); g.strokeStyle = col; g.lineWidth = 2.2; g.stroke(); };
          path((c) => Yb(c.B), C.a);
          path((c) => Yk(c.K), C.c);
          V.line(g, X(S.lam), gy0, X(S.lam), gy1, { color: C.d, lw: 1.6, dash: [3, 3] });
          // curve legend above the chart, with color markers (out of the way of the curves and the λ line)
          let lgx = gx0;
          [[C.a, 'refusals removed'], [C.c, 'neutral KLD (normalized)']].forEach(([col, tx]) => {
            V.line(g, lgx, gy0 - 14, lgx + 14, gy0 - 14, { color: col, lw: 2.6 });
            V.text(g, tx, lgx + 19, gy0 - 10, { size: 11, color: col });
            lgx += 19 + measure(g, tx, 11) + 16;
          });
          V.text(g, 'λ', gx1 + 8, gy1 + 4, { size: 11, color: C.muted, font: 'mono' });
        }
        V.text(g, 'λ = ' + fmt(S.lam, 1) + ' · ' + zone, cx, by0 + bH - 4, { size: 12, align: 'center', color: S.lam >= 4 ? RED() : C.muted, font: 'mono' });
      });
      upd();
    })();

    /* ================================================================
       Figure 10 — five paths to the edited model
       ================================================================ */
    (function () {
      const fig = root.querySelector('#fig-caminhos');
      adaptH(fig, 420, 600);
      const ctl = UI.controls(fig);
      const ROUTES = {
        glmT: { rb: 'glm', label: 'GLM · transplant', stage: 2, bytes: 2560 * 1048576, bl: '2.50 GiB', dl: '+2,560 MiB (31 o_proj via range requests)', where: 'end of load_weights (byte-for-byte copy, BF16)', tok: '0: it is in the weights', aot: 'same code: cache intact', undo: 'ABLIT=0 and restart' },
        glmP: { rb: 'glm', label: 'GLM · proj', stage: 2, bytes: 18356, bl: '18,356 B (already in repo)', dl: 'none: the direction (18,356 B; 16 KiB of fp32) already ships in the repo', where: 'end of load_weights: W′ = W − α r(rᵀW)', tok: '0: it is in the weights', aot: 'same code: cache intact', undo: 'ABLIT=0 and restart', warn: 'published directions ≈ chance: 9/32' },
        glmPre: { rb: 'glm', label: 'GLM · preset', stage: 0, bytes: 175642157752, bl: '163.6 GiB', dl: '175,642,157,752 B (120 shards)', where: 'before the download (edited by the author)', tok: '0: it is in the weights', aot: 'a different checkpoint', undo: './start.sh restart (model from .env)' },
        qwen: { rb: 'qwen', label: 'Qwen · ABLIT=1', stage: 0, bytes: 133e9, bl: '≈124 GiB', dl: '~133 GB (≈124 GiB; full snapshot, gated)', where: 'before the download (12 QSA o_proj)', tok: '0: it is in the weights', aot: 'a different checkpoint', undo: 'ABLIT=0 ./start.sh' },
        ds: { rb: 'ds', label: 'DS · ABLITERATED=1', stage: 5, bytes: 18372, bl: '18,372 B', dl: '18 KiB (18,372 B) + RESPONSIBLE_USE.md', where: 'forward: attention output, layers 10–42', tok: '≈ 0.54 MFLOP per token (our math)', aot: 'stamp changes: cache wiped', undo: 'ABLITERATED=0 and recreate the 2 ranks' },
      };
      const S = { r: 'glmT' };
      UI.seg(ctl, { label: 'Path', options: Object.keys(ROUTES).map((k) => ({ v: k, label: ROUTES[k].label })), value: 'glmT', onChange: (v) => { S.r = v; } });
      const STG = [['Hub'], ['disk', '(HF cache)'], ['load_weights', '(BF16)'], ['process_weights', '_after_loading'], ['torch.compile', '+ CUDA graphs'], ['forward', '(serving)']];
      vfig(fig, (g, w, h, t) => {
        const C = V.C();
        const Rt = ROUTES[S.r];
        const rc = Rt.rb === 'glm' ? C.glm : Rt.rb === 'qwen' ? C.qwen : C.ds;
        const narrow = w < 640;
        const n = STG.length;
        const pos = [];
        let infoY, laneY;
        if (!narrow) {
          const bw = (w - 32 - (n - 1) * 12) / n, bh = 50, y = 48;
          for (let i = 0; i < n; i++) pos.push({ x: 16 + i * (bw + 12), y, w: bw, h: bh });
          laneY = y + bh + 16;
          infoY = y + bh + 74;
        } else {
          const bw = Math.min(210, w - 130), bh = 34;
          for (let i = 0; i < n; i++) pos.push({ x: 16, y: 34 + i * (bh + 10), w: bw, h: bh });
          infoY = 34 + n * (bh + 10) + 36;
        }
        V.text(g, Rt.label, 16, 20, { size: 12, color: rc, weight: 600, font: 'mono' });
        // conveyor belt
        for (let i = 0; i < n - 1; i++) {
          const a = pos[i], b = pos[i + 1];
          if (!narrow) V.line(g, a.x + a.w, a.y + a.h / 2, b.x, b.y + b.h / 2, { color: V.alpha(C.muted, 0.6), lw: 2, dash: [5, 4], dashOffset: -t * 24 });
          else V.line(g, a.x + 24, a.y + a.h, b.x + 24, b.y, { color: V.alpha(C.muted, 0.6), lw: 2, dash: [5, 4], dashOffset: -t * 24 });
        }
        pos.forEach((p, i) => {
          const ed = i === Rt.stage;
          V.box(g, p.x, p.y, p.w, p.h, { r: 10, fill: ed ? V.alpha(C.d, 0.22) : V.alpha(C.muted, 0.08), stroke: ed ? C.d : V.alpha(C.muted, 0.45), lw: ed ? 2 : 1 });
          const lab = STG[i];
          if (lab.length === 1 || narrow) V.text(g, narrow ? lab.join(lab[1] && lab[1][0] === '_' ? '' : ' ') : lab[0], p.x + p.w / 2, p.y + p.h / 2 + 1, { size: 11, font: 'mono', align: 'center', base: 'middle', color: C.ink, maxW: p.w - 10 });
          else { V.text(g, lab[0], p.x + p.w / 2, p.y + p.h / 2 - 7, { size: 11, font: 'mono', align: 'center', base: 'middle', color: C.ink, maxW: p.w - 8 }); V.text(g, lab[1], p.x + p.w / 2, p.y + p.h / 2 + 8, { size: 11, font: 'mono', align: 'center', base: 'middle', color: C.muted, maxW: p.w - 8 }); }
          if (ed) {
            const rr = 7 + Math.sin(t * 3);
            g.strokeStyle = V.alpha(C.d, 0.85); g.lineWidth = 2; g.beginPath();
            if (!narrow) g.arc(p.x + p.w / 2, p.y - 12, rr, 0, Math.PI * 2); else g.arc(p.x + p.w + 14, p.y + p.h / 2, rr, 0, Math.PI * 2);
            g.stroke();
            const last = !narrow && i === n - 1;
            V.text(g, 'edit here', !narrow ? p.x + p.w / 2 + (last ? -12 : 12) : p.x + p.w + 26, !narrow ? p.y - 8 : p.y + p.h / 2 + 4, { size: 11, align: last ? 'right' : 'left', color: C.d, weight: 600 });
          }
        });
        // weight piece (or token) traveling on a lane next to the belt
        const cyc = (t * 0.3) % 1;
        const fpos = cyc * (n - 1);
        const i0 = Math.floor(fpos), fr = fpos - i0;
        const pA = pos[i0], pB = pos[Math.min(n - 1, i0 + 1)];
        let tx, ty;
        if (!narrow) { tx = lerp(pA.x + pA.w / 2, pB.x + pB.w / 2, fr); ty = laneY; }
        else { tx = pA.x + pA.w + 60; ty = lerp(pA.y + pA.h / 2, pB.y + pB.h / 2, fr); }
        const edited = Rt.stage === 0 || fpos >= Rt.stage;
        if (Rt.stage === 5 && fpos >= 4.5) {
          tokChar(g, tx, ty, 16, C.c, { mood: 'o' });
          V.text(g, 'y ← y − λ(y·v)v', tx, ty - 14, { size: 11, font: 'mono', align: 'center', color: C.d });
        } else {
          for (let ii = 0; ii < 3; ii++) for (let jj = 0; jj < 3; jj++) {
            const colT = edited && Rt.stage !== 5 && ii !== 1 ? C.d : C.b;
            V.box(g, tx - 9 + jj * 6, ty - 9 + ii * 6, 5, 5, { r: 1, fill: V.alpha(colT, 0.95) });
          }
          if (!narrow) V.text(g, edited && Rt.stage !== 5 ? 'W edited' : 'W stock', tx, ty + 20, { size: 11, align: 'center', color: C.muted, font: 'mono' });
        }
        // info panel
        const lines = [['extra download', Rt.dl], ['where it lives', Rt.where], ['cost per token', Rt.tok], ['compile cache', Rt.aot], ['how to undo', Rt.undo]];
        if (Rt.warn) lines.push(['warning', Rt.warn]);
        const lh = narrow ? 28 : 19;
        lines.forEach(([k, v], i) => {
          const yy = infoY + i * lh;
          V.text(g, k, 16, yy, { size: 11, color: k === 'warning' ? RED() : C.muted, font: 'mono' });
          V.text(g, v, narrow ? 16 : 176, narrow ? yy + 13 : yy, { size: 12, color: k === 'warning' ? RED() : C.ink, maxW: narrow ? w - 32 : w - 196 });
        });
        // download comparison (log scale)
        const ky0 = infoY + lines.length * lh + (narrow ? 10 : 14);
        const keys = Object.keys(ROUTES);
        const rowH = narrow ? 16 : 17;
        if (ky0 + keys.length * rowH + 20 < h) {
          V.text(g, narrow ? 'edit artifact bytes (log scale)' : 'bytes of the artifact carrying the edit, per path (log scale, 16 KiB to 256 GiB)', 16, ky0, { size: 11, color: C.muted, maxW: w - 32 });
          const lx = narrow ? 128 : 176, lw2 = w - lx - 70;
          keys.forEach((k, i) => {
            const R2 = ROUTES[k], yy = ky0 + 10 + i * rowH;
            const on = k === S.r;
            V.text(g, R2.label, 16, yy + 10, { size: 11, font: 'mono', color: on ? C.ink : C.muted, weight: on ? 600 : 400, maxW: lx - 22 });
            const f = clamp((Math.log2(R2.bytes) - 14) / (38 - 14), 0.01, 1);
            V.box(g, lx, yy + 2, lw2, 10, { r: 3, fill: V.alpha(C.muted, 0.1) });
            V.box(g, lx, yy + 2, Math.max(3, lw2 * f), 10, { r: 3, fill: on ? C.d : V.alpha(C.b, 0.55) });
            V.text(g, R2.bl || bytesBR(R2.bytes), lx + Math.max(3, lw2 * f) + 6, yy + 11, { size: 11, font: 'mono', color: on ? C.d : C.muted });
          });
        }
      });
    })();
  },
});

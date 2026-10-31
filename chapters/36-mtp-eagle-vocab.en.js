LA.chapter({
  id: 'mtp-eagle-vocab',
  part: 7,
  order: 36,
  title: 'MTP, EAGLE and cheap drafts',
  navTitle: 'MTP, EAGLE & reduced vocab',
  subtitle: 'The drafter that ships inside the checkpoint (MTP), EAGLE-3’s taps on the residual stream and the recipes’ tricks for making each draft cheaper without changing the output distribution: a TP-aware reduced vocabulary, pairwise argmax, IndexShare and the prefix-cache block EAGLE used to drop.',
  minutes: 55,
  recipes: ['glm', 'qwen', 'ds'],
  tags: ['MTP', 'EAGLE-3', 'FR-Spec', 'reduced vocabulary', 'lm_head', 'distributed argmax', 'IndexShare', 'prefix cache', 'speculative decoding'],
  goals: [
    'Explain how an <b>MTP</b> layer becomes a recursive drafter, why acceptance drops with position, and how to choose <b>k</b> with the rule \\(a_{k+1} > \\tau\\,\\Delta t/t\\)',
    'Describe vLLM’s <b>EAGLE-3</b> contract (auxiliary states, <code>SupportsEagle3</code>) and why GLM needed <code>hc_post</code> + <code>hc_contract</code> to tap mHC',
    'Show why the <code>lm_head</code> dominates the draft cost with a huge vocabulary, and how a <b>reduced vocabulary</b> (FR-Spec) cuts bytes without changing the output distribution',
    'Implement the <b>TP-aware</b> version (id ranges per rank, slowest rank, pairwise argmax) and build the vocabulary by <b>coverage</b>',
    'Configure <code>disable_eagle_block_drop</code> and <code>index_share_for_mtp_iteration</code> and assemble the three recipes’ <code>--speculative-config</code>',
  ],
  prereqs: ['especulacao-fundamentos', 'ngram-mtp'],
  summary: [
    '<b>MTP</b> predicts \\(t+2\\) from \\(h_t\\) and \\(E\\,x_{t+1}\\) (<code>eh_proj</code> [4096×8192] on GLM) and becomes a recursive drafter: one draft-extend plus k−1 more forwards of the same layer. On Qwen, MTP=3 took it from 24.5 to 52.1 tok/s, with per-position acceptance of 89/74.5/60%.',
    'Choosing k: \\(\\tau = 1 + \\sum a_i\\), and position k+1 pays off if \\(a_{k+1} > \\tau\\,\\Delta t / t\\), which gives ~0.31–0.35 for Qwen’s 3rd position. <code>n_predict</code> requires k ≤ n or k a multiple of n (on DeepSeek Vision-Exp, k=6).',
    '<b>EAGLE-3</b> reads states from several depths of the target (<code>SupportsEagle3</code>, <code>aux_hidden_state_layers</code>). On GLM, the taps (6, 15, 25, 34, 43) require materializing <code>hc_post</code> and contracting mHC’s 4 streams down to 4096. Every <code>use_eagle()</code> method inherits the engine’s precautions.',
    'The <code>lm_head</code> dominates the draft: on Qwen, the 248,320×2560 matrix (0.59 GiB per GPU) is read 3× per step by the drafter, which comes to 26% of the step’s ≈9.2 GiB counting the target’s read, and ~78% of the draft’s bytes.',
    '<b>Reduced vocabulary</b> (FR-Spec): argmax over just 47,149 ids. A wrong draft is rejected, so the output distribution doesn’t change, and \\(\\alpha_S \\ge \\alpha - (1-\\mathrm{cov})\\). Gain of +9.6% / +8.4%, all of it in ms per step (61.2 → 54.0 ms).',
    '<b>TP-aware</b>: each rank cuts its own id range; the frequent vocabulary sits ~97% on rank 0 (45,734 versus 1,415) and the step waits for the slowest rank. <code>use_local_argmax_reduction</code> swaps the logits all-gather for (value, id) pairs. Build by <b>coverage</b>, preferably over the model’s own output: 65k with only 8.3k real ids (the rest, guesswork filler) sped up 3 of 4 tasks, but dropped acceptance from 56.5% to 47.8%.',
    '<code>disable_eagle_block_drop</code> keeps the last matched block (1,664 tokens): 2nd-turn TTFT from 3.25 to 0.63 s (“turn 1” in the CHANGELOG, which counts from 0). <code>index_share_for_mtp_iteration</code> reuses the indexer’s top-k in steps 2..k: code +6 to +27%.',
  ],
  quiz: [
    { q: 'Qwen’s 47k draft vocabulary was tuned on English and code. A user writes in Chinese. What happens to the server’s <b>response</b>?', o: ['It comes out with errors, because the drafter doesn’t know those tokens', 'It keeps the target’s distribution; only acceptance, and with it speed, may drop', 'vLLM turns MTP off automatically', 'Tokens outside the subset are swapped for the nearest neighbor inside it'], a: 1, e: 'The draft is only a proposal: the target verifies every position with the full vocabulary (<code>compute_logits</code> isn’t touched). A token outside the subset just becomes a rejected draft. The cost is in acceptance, never in correctness. (Under greedy, only the numerical nondeterminism of TP2+EP, which the recipe measures and fixes separately, could move the literal sequence.)' },
    { q: 'With the 47k vocabulary on TP2, rank 0 reads 0.22 GiB per draft and rank 1, ~0.01 GiB. Why did the gain land at +8 to +10%, and not higher?', o: ['Because rank 1 has to recompute rank 0’s ids', 'Because the step waits for the slowest rank: frequent ids are low ids (BPE merge order), and ~97% of them fall in rank 0’s range', 'Because the logits all-gather still costs O(V)', 'Because the file has duplicate ids'], a: 1, e: 'The vocab-parallel <code>lm_head</code> splits the vocabulary by id ranges (0–124,159 and 124,160–248,319). The critical rank goes from 0.59 to 0.22 GiB; on single-Spark, the single GPU went from 1.18 to 0.22 GiB, and the gain was +13.1%. With <code>use_local_argmax_reduction</code>, the collective is already just (value, id) pairs, so option 3 is wrong.' },
    { q: 'Why does <code>patch_glm_eagle3.py</code> materialize <code>hc_post</code> and apply <code>hc_contract</code> before handing over the states of layers 5, 14, 24, 33 and 42?', o: ['To quantize the states to FP8', 'Because in the middle of the mHC stack the state is [tokens, 4, 4096], with hc_post deferred, and the drafter expects [tokens, 4096]', 'Because KDA layers have no hidden state', 'To apply the RoPE that GLM NoPE lacks'], a: 1, e: 'mHC keeps 4 residual streams, and vLLM fuses one layer’s write with the next layer’s read (deferred hc_post). The patch materializes the write and averages the 4 streams. Without it, DFlash2 would receive a tensor of the wrong shape. The log shows (6, 15, 25, 34, 43) because the hook fires at idx+1.' },
    { q: 'Why does vLLM drop the last matched prefix-cache block when there’s an EAGLE-style drafter (MTP included)?', o: ['Because the target’s KV in that block is quantized differently', 'Because the drafter’s input at p is (h_p, x_{p+1}): what it stored at the end of the block depends on the next token, which can differ between requests, and it needs the target’s states at the boundary', 'Because 1,664-token blocks don’t fit in a CUDA graph', 'To free memory for the drafter'], a: 1, e: 'It’s an engine precaution for any <code>use_eagle()</code>. The cost is recomputing a whole block (1,664 tokens on Qwen) per turn; with bug #62 in the day-0 image, the whole turn. With <code>disable_eagle_block_drop: true</code>, a session’s 2nd-turn TTFT dropped from 3.25 to 0.63 s, with no change to decode, prefill or NLL.' },
    { q: 'What does <code>index_share_for_mtp_iteration</code> reuse, and why does that cost almost no acceptance?', o: ['The target’s KV cache across requests', 'The sparse indexer’s top-k selection computed in draft-extend, reused in steps 2..k: the query moved at most k−1 positions, and the recent tail always goes in', 'The lm_head logits from the previous step', 'The MoE weights already loaded on the GPU'], a: 1, e: 'With r = 4 and at most 2 positions of drift, the reused selection differs from the fresh one in very few blocks. The tech report compared MTP with full attention (accepted length 4.06) and with QSA plus index reuse (4.07), and the recipe measured +6.1/+4.6/+21.3/+26.6% on code (S = 1/2/4/8).' },
    { q: 'On Qwen (prose profile), with k=2 the step has τ = 2.45 and takes ~54.6 ms, and adding the 3rd position makes the step ~6.9 ms more expensive. From what acceptance \\(a_3\\) on is k=3 worth it?', o: ['Any \\(a_3 > 0\\)', '\\(a_3 > \\tau\\,\\Delta t/t \\approx 0.31\\), consistent with the ~35% rule of thumb', '\\(a_3 > 0.5\\)', 'Only with \\(a_3 = 1\\)'], a: 1, e: 'Compare \\((\\tau + a_3)/(t + \\Delta t)\\) with \\(\\tau/t\\): it pays off if \\(a_3 > 2.45 \\times 6.9/54.6 \\approx 0.31\\). The measured prose profile has \\(a_3 = 0.51\\), so k=3 still pays off; the inventory recommends going back to k=2 if position 3 falls below ~35%.' },
  ],
  mount(root, LA) {
    const V = LA.viz, UI = LA.ui;
    const RED = () => V.C().red || '#FB2C36';
    const fmt = (x, d = 1) => (+x).toFixed(d);
    const fmtInt = (x) => Math.round(x).toLocaleString('en-US');
    const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
    const GIB = 2 ** 30;

    /* ------------------------------------------------------------ helpers */
    function adaptH(fig, wide, narrow, bp = 620) {
      const st = fig.querySelector('.la-fig-stage');
      const set = () => { const want = (fig.clientWidth < bp ? narrow : wide) + 'px'; if (st.style.height !== want) st.style.height = want; };
      set();
      const ro = new ResizeObserver(set); ro.observe(fig);
      if (LA.onCleanup) LA.onCleanup(() => ro.disconnect());
    }
    function tw(g, s, size, font = 'sans', weight = 400) { g.font = V.font(size, font, weight); return g.measureText(String(s)).width; }
    function tokChar(g, x, y, s, col, o = {}) {
      const C = V.C();
      V.box(g, x - s / 2, y - s / 2, s, s, { r: s * 0.3, fill: o.fill || V.alpha(col, 0.85), stroke: col, lw: 1.2 });
      if (s >= 11) {
        const eye = C.dark ? '#06140d' : '#ffffff';
        const ex = s * 0.17, ey = y - s * 0.1, er = Math.max(1, s * 0.07);
        V.dot(g, x - ex, ey, er, eye); V.dot(g, x + ex, ey, er, eye);
        g.beginPath();
        if (o.mood === 'sad') g.arc(x, y + s * 0.3, s * 0.14, Math.PI * 1.15, Math.PI * 1.85);
        else g.arc(x, y + s * 0.05, s * 0.15, Math.PI * 0.2, Math.PI * 0.8);
        g.strokeStyle = eye; g.lineWidth = Math.max(1, s * 0.06); g.stroke();
      }
    }
    function hatch(g, x, y, w, h, col, step = 6, lw = 1) {
      if (w <= 0 || h <= 0) return;
      g.save(); g.beginPath(); g.rect(x, y, w, h); g.clip();
      g.strokeStyle = col; g.lineWidth = lw; g.beginPath();
      for (let k = -h; k < w; k += step) { g.moveTo(x + k, y + h); g.lineTo(x + k + h, y); }
      g.stroke(); g.restore();
    }
    function pill(g, x, y, text, col, o = {}) {
      const size = o.size || 11, font = o.font || 'mono';
      const wv = tw(g, text, size, font, 500) + 14, hh = o.h || 20;
      const x0 = o.align === 'left' ? x : o.align === 'right' ? x - wv : x - wv / 2;
      V.box(g, x0, y - hh / 2, wv, hh, { r: hh / 2, fill: V.alpha(col, o.fillA == null ? 0.16 : o.fillA), stroke: V.alpha(col, 0.75), lw: 1 });
      V.text(g, text, x0 + wv / 2, y + 0.5, { size, font, color: o.color || V.C().ink, align: 'center', base: 'middle', weight: 500 });
      return wv;
    }
    function subT(g, base, s, x, y, o = {}) {
      const size = o.size || 13, font = o.font || 'sans', wt = o.weight || 600, ss = Math.max(11, Math.round(size * 0.78));
      const w1 = tw(g, base, size, font, wt), w2 = tw(g, s, ss, font, 500);
      const x0 = o.align === 'left' ? x : o.align === 'right' ? x - w1 - w2 - 1 : x - (w1 + w2 + 1) / 2;
      V.text(g, base, x0, y, { size, font, weight: wt, color: o.color });
      V.text(g, s, x0 + w1 + 1, y + size * 0.3, { size: ss, font, weight: 500, color: o.color });
    }
    function polyArrow(g, pts, col, o = {}) {
      g.beginPath(); g.moveTo(pts[0][0], pts[0][1]);
      for (let i = 1; i < pts.length; i++) g.lineTo(pts[i][0], pts[i][1]);
      g.strokeStyle = col; g.lineWidth = o.lw || 1.6;
      if (o.dash) { g.setLineDash(o.dash); g.lineDashOffset = o.off || 0; }
      g.stroke(); g.setLineDash([]); g.lineDashOffset = 0;
      const a = pts[pts.length - 2], b = pts[pts.length - 1];
      const an = Math.atan2(b[1] - a[1], b[0] - a[0]), hs = o.head || 7;
      g.beginPath(); g.moveTo(b[0], b[1]);
      g.lineTo(b[0] - hs * Math.cos(an - 0.45), b[1] - hs * Math.sin(an - 0.45));
      g.lineTo(b[0] - hs * Math.cos(an + 0.45), b[1] - hs * Math.sin(an + 0.45));
      g.closePath(); g.fillStyle = col; g.fill();
    }
    const cellFill = (col, v, a = 1) => V.alpha(col, (0.2 + 0.62 * Math.min(1, Math.abs(v) / 2)) * a);
    function vecV(g, x, y, vals, cs, col, a = 1, gap = 2) {
      for (let i = 0; i < vals.length; i++) { V.rr(g, x - cs / 2, y + i * (cs + gap), cs, cs, Math.min(3, cs / 4)); g.fillStyle = cellFill(col, vals[i], a); g.fill(); }
      return vals.length * (cs + gap) - gap;
    }
    function vecH(g, x, y, vals, cs, col, a = 1, gap = 2) {
      for (let i = 0; i < vals.length; i++) { V.rr(g, x + i * (cs + gap), y - cs / 2, cs, cs, Math.min(3, cs / 4)); g.fillStyle = cellFill(col, vals[i], a); g.fill(); }
      return vals.length * (cs + gap) - gap;
    }
    function gridM(g, x, y, M, cw, ch, col, a = 1, gap = 1.5, hiRow = -1) {
      for (let i = 0; i < M.length; i++) for (let j = 0; j < M[0].length; j++) {
        g.fillStyle = cellFill(col, M[i][j], a * (hiRow >= 0 && i !== hiRow ? 0.6 : 1));
        g.fillRect(x + j * (cw + gap), y + i * (ch + gap), cw, ch);
      }
      const W = M[0].length * (cw + gap) - gap;
      if (hiRow >= 0) { g.strokeStyle = V.C().c; g.lineWidth = 2; V.rr(g, x - 2, y + hiRow * (ch + gap) - 2, W + 4, ch + 4, 3); g.stroke(); }
      return { w: W, h: M.length * (ch + gap) - gap };
    }
    function flowDots(g, x1, y1, x2, y2, t, col, n = 4, speed = 0.8) {
      for (let k = 0; k < n; k++) { const u = (t * speed + k / n) % 1; const p = V.along(x1, y1, x2, y2, u); V.dot(g, p[0], p[1], 2.3, col); }
    }
    function bez(g, x1, y1, x2, y2, col, lw = 1.3, dash) {
      const mx = (x1 + x2) / 2;
      g.beginPath(); g.moveTo(x1, y1); g.bezierCurveTo(mx, y1, mx, y2, x2, y2);
      g.strokeStyle = col; g.lineWidth = lw; if (dash) g.setLineDash(dash); g.stroke(); g.setLineDash([]);
    }
    function bezPt(x1, y1, x2, y2, u) { const mx = (x1 + x2) / 2; return V.bez([x1, y1], [mx, y1], [mx, y2], [x2, y2], u); }

    /* ================================================================
       Fig 1 — Anatomia de um palpite MTP (stepper + seletor de modelo)
       ================================================================ */
    (function figAnatomia() {
      const fig = root.querySelector('#fig-anatomia'); if (!fig) return;
      adaptH(fig, 440, 640);
      const MOD = {
        glm: { col: 'glm', eT: 'E·x', eD: '4096', hT: 'h', hD: '4096', n1: 'enorm', n2: 'hnorm', catT: 'concat', catD: '8192', projT: 'eh_proj', projD: '4096 × 8192', zD: '4096', attn: 'MLA + DSA', ffn: 'MoE · 288', resid: 'plain residual', headT: 'lm_head (target’s)', headD: '154,880 × 4096', vD: 'V = 154,880' },
        qwen: { col: 'qwen', eT: 'E·x', eD: '2560', hT: 'R', hD: '4 branches · 10240', n1: 'pre_fc_norm_emb', n2: 'pre_fc_norm_hid', catT: 'fusion', catD: 'emb + 4 branches', projT: 'fc_emb · fc_hid', projD: '2 × [2560 × 2560]', zD: '2560', attn: 'QSA', ffn: 'MoE · 512 FP8', resid: 'Gated Residual', headT: 'own lm_head', headD: '248,320 × 2560', vD: 'V = 248,320' },
      };
      let M = 'glm';
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Model', options: [{ v: 'glm', label: 'GLM-5.3-Flash' }, { v: 'qwen', label: 'Qwen3.8-Flash-Next' }], value: 'glm', onChange: (v) => { M = v; } });
      const steps = [
        { t: 'Shifted inputs', d: 'The MTP receives <b>two</b> things: the target’s final hidden state at position \\(t\\) and the embedding of token \\(t+1\\), which the target has just emitted. The guess is about token \\(t+2\\).' },
        { t: 'Normalize each input', d: 'Each input goes through its own RMSNorm (<code>enorm</code> and <code>hnorm</code> on GLM; <code>pre_fc_norm_*</code> on Qwen). They come from different scales and have to be made comparable before the fusion.' },
        { t: 'Fuse and project', d: 'On GLM, the two become a vector of \\(2d = 8192\\) that <code>eh_proj</code> [4096 × 8192] brings back down to \\(d\\). On Qwen, the fusion uses <code>fc_embedding</code> and <code>fc_hidden</code> [2560 × 2560] over the embedding and the 4 residual branches.' },
        { t: 'One decoder block', d: 'A full layer: attention (DSA on GLM, QSA on Qwen) with <b>its own KV cache</b>, and MoE. This is where the MTP reads the context.' },
        { t: 'The output head', d: 'Final norm and the <code>lm_head</code>: a matrix × vector multiplication with \\(V\\) rows. It is by far the largest matrix on the path, and in decode it is read in full.' },
        { t: 'Draft and recursion', d: 'The argmax becomes the draft \\(\\hat x_{t+2}\\). In the next draft step, the output state \\(u_t\\) and \\(\\hat x_{t+2}\\) itself come back as inputs, and the <b>same layer</b> runs again.' },
      ];
      const st = UI.stepper(fig, steps, { autoplay: 6500 });
      const r = V.rng(36);
      const vE = Array.from({ length: 7 }, () => r.normal()), vH = Array.from({ length: 7 }, () => r.normal());
      const P = V.randMatrix(7, 14, 9), vZ = Array.from({ length: 7 }, () => r.normal());
      const NR = 13, HM = V.randMatrix(NR, 6, 21);
      const LG = Array.from({ length: NR }, (_, i) => 0.22 + 0.45 * Math.abs(Math.sin(i * 2.3 + 1)));
      const AM = 4; LG[AM] = 1;
      const EXP = [[0, 1], [2, 3], [1, 2], [3, 0], [2, 1]];
      V.fig(fig, (g, w, h, t) => {
        const C = V.C(), m = MOD[M], col = C[m.col] || C.a;
        const si = st.i, el = st.elapsed();
        const A = (s) => (s < si ? 0.85 : s === si ? 1 : 0.3);
        const narrow = w < 620;
        const cs = narrow ? 10 : 13, gp = 2, vh = 7 * (cs + gp) - gp, ch14 = 14 * (cs + gp) - gp;
        let L;
        if (!narrow) {
          const cy = h * 0.5;
          L = { inX: 46, eY: cy - vh - 22, hY: cy + 22, normX: w * 0.145, catX: w * 0.235, catY: cy - ch14 / 2,
            projX: w * 0.285, projY: cy - vh / 2, projW: w * 0.15, zX: w * 0.475, zY: cy - vh / 2,
            bx: w * 0.515, by: cy - 104, bw: w * 0.13, bh: 208, hx: w * 0.69, hy: cy - 0.31 * h, hw: w * 0.06, hh: 0.6 * h,
            lgX: w * 0.765, lgW: w * 0.1, tokX: w * 0.935 };
        } else {
          const c1 = h * 0.26, c2 = h * 0.74;
          L = { inX: w * 0.1, eY: c1 - vh - 8, hY: c1 + 8, normX: w * 0.265, catX: w * 0.43, catY: c1 - ch14 / 2,
            projX: w * 0.5, projY: c1 - vh / 2, projW: w * 0.29, zX: w * 0.9, zY: c1 - vh / 2,
            bx: w * 0.04, by: c2 - 92, bw: w * 0.3, bh: 184, hx: w * 0.43, hy: c2 - 0.16 * h, hw: w * 0.11, hh: 0.32 * h,
            lgX: w * 0.575, lgW: w * 0.2, tokX: w * 0.9 };
        }
        const rowH = (L.hh - (NR - 1) * 1.5) / NR, rowY = (i) => L.hy + i * (rowH + 1.5) + rowH / 2;
        const eMid = L.eY + vh / 2, hMid = L.hY + vh / 2;
        const conn = (x1, y1, x2, y2, s) => { V.arrow(g, x1, y1, x2, y2, { color: V.alpha(C.muted, s <= si ? 0.7 : 0.25), lw: 1.4, head: 6 }); if (s === si) flowDots(g, x1, y1, x2, y2, t, col, 3, 0.9); };
        // brilho do passo ativo (desenhado antes)
        const pulse = 0.5 + 0.5 * Math.sin(t * 4);
        if (si === 0) { V.glow(g, L.inX, eMid, 34 + 6 * pulse, C.a, 0.3); V.glow(g, L.inX, hMid, 34 + 6 * pulse, C.b, 0.3); }
        if (si === 3) V.glow(g, L.bx + L.bw / 2, L.by + L.bh / 2, L.bw * 0.8, col, 0.22 + 0.1 * pulse);
        // 1) entradas
        vecV(g, L.inX, L.eY, vE, cs, C.a, A(0));
        vecV(g, L.inX, L.hY, vH, cs, C.b, A(0));
        g.globalAlpha = Math.max(0.5, A(0));
        subT(g, m.eT, 't+1', L.inX, L.eY - 26, { size: 13, color: C.ink });
        V.text(g, m.eD, L.inX, L.eY - 6, { size: 11, font: 'mono', align: 'center', color: C.muted });
        subT(g, m.hT, 't', L.inX, L.hY + vh + 17, { size: 13, color: C.ink });
        V.text(g, m.hD, Math.max(6, L.inX - cs), L.hY + vh + 31, { size: 11, font: 'mono', color: C.muted });
        g.globalAlpha = 1;
        // 2) normas
        g.globalAlpha = A(1);
        const pwE = tw(g, 'RMSNorm', 11, 'mono', 500) + 14;
        pill(g, L.normX, eMid, 'RMSNorm', C.a, { fillA: si === 1 ? 0.35 : 0.14 });
        pill(g, L.normX, hMid, 'RMSNorm', C.b, { fillA: si === 1 ? 0.35 : 0.14 });
        V.text(g, m.n1, L.normX, eMid + 24, { size: 11, font: 'mono', align: 'center', color: C.muted });
        V.text(g, m.n2, L.normX, hMid + 24, { size: 11, font: 'mono', align: 'center', color: C.muted });
        g.globalAlpha = 1;
        conn(L.inX + cs / 2 + 4, eMid, L.normX - pwE / 2 - 3, eMid, 1);
        conn(L.inX + cs / 2 + 4, hMid, L.normX - pwE / 2 - 3, hMid, 1);
        // 3) fusão + projeção
        const nShow = si === 2 ? clamp(Math.floor(el * 12), 0, 14) : 14;
        for (let i = 0; i < 14; i++) {
          const v = i < 7 ? vE[i] : vH[i - 7], c = i < 7 ? C.a : C.b;
          V.rr(g, L.catX - cs / 2, L.catY + i * (cs + gp), cs, cs, 3);
          g.fillStyle = cellFill(c, v, A(2) * (i < nShow ? 1 : 0.2)); g.fill();
        }
        conn(L.normX + pwE / 2 + 3, eMid, L.catX - cs / 2 - 4, L.catY + 3.5 * (cs + gp), 2);
        conn(L.normX + pwE / 2 + 3, hMid, L.catX - cs / 2 - 4, L.catY + 10.5 * (cs + gp), 2);
        const pcw = (L.projW - 13 * 1.5) / 14;
        const hiRow = si === 2 ? Math.floor(el * 3) % 7 : -1;
        gridM(g, L.projX, L.projY, P, pcw, cs, C.f, A(2), 1.5, hiRow);
        conn(L.catX + cs / 2 + 4, L.catY + ch14 / 2, L.projX - 4, L.projY + vh / 2, 2);
        vecV(g, L.zX, L.zY, vZ, cs, col, A(2) * (si === 2 ? (hiRow + 1) / 7 * 0.7 + 0.3 : 1));
        conn(L.projX + L.projW + 3, L.projY + vh / 2, L.zX - cs / 2 - 4, L.zY + vh / 2, 2);
        g.globalAlpha = Math.max(0.45, A(2));
        V.text(g, m.catT, L.catX, L.catY - 20, { size: 12.5, weight: 600, align: 'center', color: C.ink });
        V.text(g, m.catD, L.catX, L.catY - 7, { size: 11, font: 'mono', align: 'center', color: C.muted });
        V.text(g, m.projT, L.projX + L.projW / 2, L.projY - 20, { size: 12.5, weight: 600, align: 'center', color: C.ink, maxW: L.projW + 30 });
        V.text(g, m.projD, L.projX + L.projW / 2, L.projY - 7, { size: 11, font: 'mono', align: 'center', color: C.muted, maxW: L.projW + 30 });
        subT(g, 'z', 't', L.zX, L.zY - 20, { size: 13, color: C.ink });
        V.text(g, m.zD, L.zX, L.zY - 7, { size: 11, font: 'mono', align: 'center', color: C.muted });
        g.globalAlpha = 1;
        // 4) bloco
        g.globalAlpha = A(3);
        V.box(g, L.bx, L.by, L.bw, L.bh, { r: 12, fill: V.alpha(col, 0.1), stroke: V.alpha(col, 0.7), lw: 1.5 });
        V.text(g, 'MTP block', L.bx + L.bw / 2, L.by + 18, { size: 12.5, weight: 700, align: 'center', color: C.ink });
        const ih = (L.bh - 58) / 2, ix = L.bx + 8, iw = L.bw - 16;
        V.box(g, ix, L.by + 28, iw, ih, { r: 8, fill: V.alpha(C.b, 0.18), stroke: V.alpha(C.b, 0.7), lw: 1 });
        V.text(g, 'attention', ix + iw / 2, L.by + 28 + ih / 2 - 6, { size: 11.5, weight: 600, align: 'center', color: C.ink });
        V.text(g, m.attn, ix + iw / 2, L.by + 28 + ih / 2 + 9, { size: 11, font: 'mono', align: 'center', color: C.muted, maxW: iw - 6 });
        const my = L.by + 34 + ih;
        V.box(g, ix, my, iw, ih, { r: 8, fill: V.alpha(C.d, 0.14), stroke: V.alpha(C.d, 0.7), lw: 1 });
        V.text(g, m.ffn, ix + iw / 2, my + 15, { size: 11, font: 'mono', align: 'center', color: C.ink, maxW: iw - 6 });
        const lit = EXP[Math.floor(t * 1.5) % EXP.length];
        const ecs = Math.min(10, (iw - 20) / 6), ex0 = ix + (iw - (6 * ecs + 5 * 3)) / 2;
        for (let q = 0; q < 12; q++) {
          const qi = q % 6, qj = Math.floor(q / 6);
          const on = si === 3 && (q === lit[0] + 6 * (lit[1] % 2) || q === lit[1] + 6 * (lit[0] % 2));
          V.rr(g, ex0 + qi * (ecs + 3), my + 24 + qj * (ecs + 3), ecs, ecs, 2); g.fillStyle = on ? C.d : V.alpha(C.d, 0.28); g.fill();
        }
        V.text(g, m.resid, L.bx + L.bw / 2, L.by + L.bh + 16, { size: 11, align: 'center', color: C.muted, maxW: L.bw + 30 });
        V.text(g, 'own KV cache', L.bx + L.bw / 2, L.by + L.bh + 30, { size: 11, align: 'center', color: C.muted, maxW: L.bw + 30 });
        g.globalAlpha = 1;
        if (!narrow) conn(L.zX + cs / 2 + 4, L.zY + vh / 2, L.bx - 4, L.by + L.bh / 2, 3);
        else {
          const x1 = L.zX, y1 = L.zY + vh + 22, x2 = L.bx + L.bw / 2, y2 = L.by - 6;
          g.globalAlpha = 1;
          polyArrow(g, [[x1, y1], [x1, (y1 + y2) / 2], [x2, (y1 + y2) / 2], [x2, y2]], V.alpha(C.muted, si >= 3 ? 0.7 : 0.25), { lw: 1.4, head: 6 });
          if (si === 3) { const u = (t * 0.7) % 1; const p = u < 0.5 ? V.along(x1, (y1 + y2) / 2, x2, (y1 + y2) / 2, u * 2) : V.along(x2, (y1 + y2) / 2, x2, y2, (u - 0.5) * 2); V.dot(g, p[0], p[1], 2.6, col); }
        }
        // 5) cabeça
        const hcw = (L.hw - 5 * 1.5) / 6;
        const hr = si === 4 ? Math.floor(el * 5) % NR : si === 5 ? AM : -1;
        gridM(g, L.hx, L.hy, HM, hcw, rowH, C.d, A(4), 1.5, hr);
        conn(L.bx + L.bw + 4, L.by + L.bh / 2, L.hx - 4, L.hy + L.hh / 2, 4);
        const grow = si > 4 ? 1 : si === 4 ? clamp(el / 1.6, 0, 1) : 1;
        for (let i = 0; i < NR; i++) {
          const isAM = i === AM && si >= 4;
          const a = si >= 4 ? 1 : 0.2;
          V.rr(g, L.lgX, rowY(i) - rowH * 0.35, Math.max(2, L.lgW * LG[i] * grow), rowH * 0.7, 3);
          g.fillStyle = isAM ? V.alpha(C.c, 0.95 * a) : V.alpha(C.d, 0.55 * a); g.fill();
        }
        g.globalAlpha = Math.max(0.45, A(4));
        V.text(g, m.headT, L.hx + L.hw / 2, L.hy - 20, { size: 12.5, weight: 600, align: 'center', color: C.ink, maxW: narrow ? w * 0.34 : L.hw + 80 });
        V.text(g, m.headD, L.hx + L.hw / 2, L.hy - 7, { size: 11, font: 'mono', align: 'center', color: C.muted, maxW: narrow ? w * 0.34 : L.hw + 80 });
        V.text(g, 'logits', L.lgX + L.lgW / 2, L.hy + L.hh + 16, { size: 12.5, weight: 600, align: 'center', color: C.ink });
        V.text(g, m.vD, L.lgX + L.lgW / 2, L.hy + L.hh + 30, { size: 11, font: 'mono', align: 'center', color: C.muted, maxW: L.lgW + 40 });
        g.globalAlpha = 1;
        // 6) token + recursão
        const ty = rowY(AM);
        if (si === 5) {
          const sc = 0.6 + 0.4 * V.ease.back(clamp(el / 0.6, 0, 1));
          tokChar(g, L.tokX, ty, 30 * sc, C.c);
          conn(L.lgX + L.lgW + 6, ty, L.tokX - 19, ty, 5);
          subT(g, 'x̂', 't+2', L.tokX, ty + 32, { size: 13, color: C.ink });
          const off = -t * 26, dcol = V.alpha(C.c, 0.85);
          if (!narrow) {
            polyArrow(g, [[L.tokX, ty - 18], [L.tokX, 12], [L.inX + 22, 12], [L.inX + 22, L.eY - 34]], dcol, { dash: [5, 5], off, lw: 1.5 });
            polyArrow(g, [[L.bx + L.bw / 2, L.by + L.bh + 36], [L.bx + L.bw / 2, h - 10], [L.inX + 22, h - 10], [L.inX + 22, L.hY + vh + 40]], V.alpha(col, 0.85), { dash: [5, 5], off, lw: 1.5 });
            V.text(g, 'x̂ becomes the next input', (L.tokX + L.inX) / 2, 26, { size: 11.5, align: 'center', color: C.c, weight: 600 });
            V.text(g, 'u_t becomes the next state', (L.bx + L.inX) / 2 + 30, h - 16, { size: 11.5, align: 'center', color: col, weight: 600 });
          } else {
            polyArrow(g, [[L.tokX, ty - 18], [L.tokX, L.by - 30], [w - 8, L.by - 30], [w - 8, 10], [L.inX + 18, 10], [L.inX + 18, L.eY - 34]], dcol, { dash: [5, 5], off, lw: 1.5 });
            V.text(g, 'x̂ and u_t come back as input', w * 0.5, 22, { size: 11.5, align: 'center', color: C.c, weight: 600 });
          }
        } else {
          g.setLineDash([3, 3]); g.strokeStyle = V.alpha(C.muted, 0.5); g.lineWidth = 1.2; V.rr(g, L.tokX - 13, ty - 13, 26, 26, 8); g.stroke(); g.setLineDash([]);
          V.text(g, '?', L.tokX, ty + 1, { size: 13, align: 'center', base: 'middle', color: C.muted });
        }
      });
    })();

    /* ================================================================
       Fig 2 — Quantos rascunhos por passo? (slider k, perfis, vocab)
       ================================================================ */
    (function figK() {
      const fig = root.querySelector('#fig-k'); if (!fig) return;
      adaptH(fig, 450, 680);
      const PROF = { prosa: [0.81, 0.64, 0.51], greedy: [0.89, 0.745, 0.6], codigo: [1.0, 0.99, 0.99] };
      const S = { k: 3, prof: 'prosa', v47: false };
      const T0 = 40.8, TV = 0.5, TM = 2.6, TLF = 3.8, TL47 = 1.4;
      function acc() { const a = PROF[S.prof].slice(); const c = a[1] > 0 ? a[2] / a[1] : 1; while (a.length < 5) a.push(a[a.length - 1] * c); return a; }
      function model(k, v47) { const a = acc(); let tau = 1; for (let i = 0; i < k; i++) tau += a[i]; const tl = v47 ? TL47 : TLF; const tt = T0 + TV * k + k * (TM + tl); return { tau, t: tt, tps: (tau / tt) * 1000, tl, a }; }
      const ctl = UI.controls(fig);
      UI.slider(ctl, { label: 'k (drafts)', min: 0, max: 5, step: 1, value: 3, onInput: (v) => { S.k = v; upd(); } });
      UI.seg(ctl, { label: 'Profile', options: [{ v: 'prosa', label: 'prose' }, { v: 'greedy', label: 'greedy README' }, { v: 'codigo', label: 'synthetic code' }], value: 'prosa', onChange: (v) => { S.prof = v; upd(); } });
      UI.toggle(ctl, { label: '47k draft vocabulary', value: false, onChange: (v) => { S.v47 = v; upd(); } });
      const rT = UI.readout(ctl, 'τ', '—'), rM = UI.readout(ctl, 'step', '—'), rS = UI.readout(ctl, 'tok/s', '—'), rG = UI.readout(ctl, 'vs no MTP', '—');
      function upd() { const m = model(S.k, S.v47); rT.set(fmt(m.tau, 2)); rM.set(fmt(m.t, 1) + ' ms'); rS.set(fmt(m.tps, 1)); rG.set(fmt(m.tps / (1000 / T0), 2) + '×'); }
      upd();
      const rng = V.rng(77);
      const tray = []; let lastCi = -1, cur = null;
      function sample(k) { const a = acc(); let n = 0; for (let i = 0; i < k; i++) { const c = i === 0 ? a[0] : a[i] / a[i - 1]; if (rng() < c) n++; else break; } return n; }
      V.fig(fig, (g, w, h, t) => {
        const C = V.C(), narrow = w < 620, RC = RED();
        const m = model(S.k, S.v47);
        const PER = 2.6, ci = Math.floor(t / PER), f = (t % PER) / PER;
        if (ci !== lastCi) { lastCi = ci; cur = { k: S.k, nAcc: sample(S.k), pushed: false }; }
        const x0 = 16, x1 = w - 16, sc = (x1 - x0) / 80;
        const by = 62, bh = 24;
        V.text(g, 'one engine step', x0, 20, { size: 12.5, weight: 700, color: C.ink });
        V.text(g, `${S.k + 1} row${S.k ? 's' : ''} verified · ${S.k} draft${S.k === 1 ? '' : 's'}`, x1, 20, { size: 11.5, align: 'right', color: C.muted });
        const tv = T0 + TV * S.k;
        const tp = m.t * clamp(f / 0.85, 0, 1);
        const px = (ms) => x0 + ms * sc;
        V.box(g, px(0), by, tv * sc, bh, { r: 6, fill: V.alpha(C.b, 0.55), stroke: C.b, lw: 1 });
        if (tv * sc > 110) V.text(g, 'verify (target)', px(tv / 2), by + bh / 2 + 1, { size: 11.5, align: 'center', base: 'middle', color: C.ink, weight: 600 });
        for (let i = 0; i < S.k; i++) {
          const s0 = tv + i * (TM + m.tl);
          V.box(g, px(s0) + 0.5, by, TM * sc - 1, bh, { r: 3, fill: V.alpha(C.a, 0.6) });
          V.box(g, px(s0 + TM) + 0.5, by, m.tl * sc - 1, bh, { r: 3, fill: V.alpha(C.d, 0.7) });
        }
        // régua de ms
        for (let ms = 0; ms <= 80; ms += 20) { V.line(g, px(ms), by + bh + 2, px(ms), by + bh + 6, { color: C.muted, lw: 1 }); V.text(g, ms + ' ms', px(ms), by + bh + 18, { size: 11, font: 'mono', align: ms === 0 ? 'left' : ms === 80 ? 'right' : 'center', color: C.muted }); }
        V.line(g, px(T0), by - 6, px(T0), by + bh + 6, { color: C.ink, lw: 1, dash: [3, 3] });
        // cursor
        V.line(g, px(tp), by - 10, px(tp), by + bh + 8, { color: C.c, lw: 2 });
        // rascunhos em verificação (da rodada anterior)
        const verified = tp >= tv;
        if (verified && !cur.pushed) { cur.pushed = true; for (let i = 0; i < cur.nAcc; i++) tray.push('acc'); tray.push('bonus'); while (tray.length > 60) tray.shift(); }
        const n = cur.k, cs = 15;
        for (let i = 0; i < n; i++) {
          const cx = px(tv) * 0.15 + px(0) + ((i + 0.5) / Math.max(1, n + 1)) * tv * sc * 0.85;
          const ok = i < cur.nAcc;
          const colr = !verified ? C.b : ok ? C.e : RC;
          tokChar(g, cx, by - 16, cs, colr, { mood: verified && !ok ? 'sad' : '' });
        }
        if (verified) { const cx = px(0) + ((n + 0.5) / Math.max(1, n + 1)) * tv * sc * 0.85 + px(tv) * 0.15; tokChar(g, cx, by - 16, cs, C.d); }
        // novos rascunhos nascendo
        for (let i = 0; i < S.k; i++) { const s1 = tv + (i + 1) * (TM + m.tl); if (tp >= s1 - 0.2) tokChar(g, px(s1 - (TM + m.tl) / 2), by - 16, 13, V.alpha(C.b, 0.9)); }
        // bandeja
        const ty = by + bh + 46;
        V.text(g, 'tray', x0, ty + 4, { size: 11.5, weight: 600, color: C.muted });
        const tx0 = x0 + 62, cap = Math.max(4, Math.floor((x1 - tx0) / 17));
        const show = tray.slice(-cap);
        show.forEach((kind, i) => tokChar(g, tx0 + i * 17 + 7, ty, 13, kind === 'acc' ? C.e : C.d));
        // --- aceitação por posição
        const a = m.a;
        let bx0, bx1, by0, by1, cx0, cx1, cy0, cy1;
        if (!narrow) { bx0 = 24; bx1 = w * 0.47; by0 = h * 0.46; by1 = h - 30; cx0 = w * 0.58; cx1 = w - 22; cy0 = h * 0.46; cy1 = h - 30; }
        else { bx0 = 24; bx1 = w - 16; by0 = h * 0.3; by1 = h * 0.56; cx0 = 44; cx1 = w - 18; cy0 = h * 0.66; cy1 = h - 28; }
        V.text(g, 'per-position acceptance (a_i)', bx0, by0 - 8, { size: 12, weight: 700, color: C.ink });
        const slot = (bx1 - bx0) / 5, bw = slot * 0.56, bH = by1 - by0 - 54;
        const thr = S.k < 5 ? (m.tau * (TM + m.tl + TV)) / m.t : null;
        for (let i = 0; i < 5; i++) {
          const x = bx0 + i * slot + (slot - bw) / 2, hh = a[i] * bH, y = by1 - 18 - hh;
          const on = i < S.k;
          V.box(g, x, y, bw, hh, { r: 4, fill: V.alpha(C.a, on ? 0.8 : 0.22) });
          if (i >= 3) hatch(g, x, y, bw, hh, V.alpha(C.ink, 0.35), 5);
          V.text(g, fmt(a[i], 2), x + bw / 2, y - 5, { size: 11, font: 'mono', align: 'center', color: on ? C.ink : C.muted });
          V.text(g, 'pos ' + (i + 1), x + bw / 2, by1 - 3, { size: 11, align: 'center', color: C.muted });
          if (thr != null && i === S.k) {
            const yy = by1 - 18 - thr * bH, good = a[i] > thr;
            V.line(g, x - 6, yy, x + bw + 6, yy, { color: good ? C.e : RC, lw: 2.2, dash: [4, 3] });
            V.text(g, `position ${i + 1}: threshold ${fmt(thr, 2)} → ${good ? 'pays off' : 'doesn’t pay off'} (a = ${fmt(a[i], 2)})`, bx0, by0 + 10, { size: 11.5, weight: 600, color: good ? C.e : RC, maxW: bx1 - bx0 });
          }
        }
        // --- tok/s por k
        V.text(g, 'tok/s (1 stream) as a function of k', cx0 - (narrow ? 20 : 0), cy0 - 8, { size: 12, weight: 700, color: C.ink });
        const full = [], red = [];
        for (let k = 0; k <= 5; k++) { full.push(model(k, false).tps); red.push(model(k, true).tps); }
        const ymax = Math.max(...full, ...red) * 1.12;
        const gx = (k) => cx0 + (k / 5) * (cx1 - cx0), gy = (v) => cy1 - 18 - (v / ymax) * (cy1 - cy0 - 40);
        V.line(g, cx0, cy1 - 18, cx1, cy1 - 18, { color: C.grid, lw: 1 });
        for (let k = 0; k <= 5; k++) V.text(g, 'k=' + k, gx(k), cy1 - 3, { size: 11, align: 'center', color: k === S.k ? C.ink : C.muted, weight: k === S.k ? 700 : 400 });
        [0, 0.5, 1].forEach((q) => { const v = ymax * q / 1.12; V.text(g, fmt(v, 0), cx0 - 6, gy(v) + 4, { size: 11, font: 'mono', align: 'right', color: C.muted }); });
        V.line(g, cx0, gy(1000 / T0), cx1, gy(1000 / T0), { color: C.muted, lw: 1, dash: [3, 4] });
        V.text(g, 'no MTP 24.5', cx1, gy(1000 / T0) - 5, { size: 11, align: 'right', color: C.muted });
        const drawL = (arr, c, active) => {
          g.beginPath(); arr.forEach((v, k) => (k ? g.lineTo(gx(k), gy(v)) : g.moveTo(gx(k), gy(v)))); g.strokeStyle = V.alpha(c, active ? 1 : 0.45); g.lineWidth = active ? 2.4 : 1.4; g.stroke();
          arr.forEach((v, k) => V.dot(g, gx(k), gy(v), k === S.k && active ? 6 : 3, V.alpha(c, active ? 1 : 0.5)));
        };
        drawL(full, C.d, !S.v47); drawL(red, C.a, S.v47);
        const cv = (S.v47 ? red : full)[S.k];
        V.text(g, fmt(cv, 1), gx(S.k), gy(cv) - 11, { size: 12, font: 'mono', align: 'center', weight: 700, color: C.ink });
        V.text(g, '— full head', cx1, cy0 + 12, { size: 11, align: 'right', color: C.d, weight: 600 });
        V.text(g, '— 47k vocab', cx1, cy0 + 27, { size: 11, align: 'right', color: C.a, weight: 600 });
      });
    })();

    /* ================================================================
       Fig 3 — Escutas EAGLE-3 no fluxo residual
       ================================================================ */
    (function figTaps() {
      const fig = root.querySelector('#fig-taps'); if (!fig) return;
      adaptH(fig, 460, 520);
      const glmType = (i) => (i >= 3 && (i - 3) % 4 === 0 ? 1 : 0);
      const dsType = (i) => (i < 2 ? 2 : i % 2 === 0 ? 1 : 3);
      const MODES = {
        mtp: { n: 45, taps: [44], type: glmType, title: 'GLM · MTP: one tap, at the top', cat: null, proj: null, out: 'h_t · 4096', dest: ['MTP layer 45', '+ E·x(t+1)'], lab: () => 'final output', legend: [['KDA', 'a'], ['MLA + DSA', 'b']] },
        glm: { n: 45, taps: [5, 14, 24, 33, 42], type: glmType, title: 'GLM · EAGLE-3 → DFlash2', cat: '[20480]', proj: 'fc · 4096 × 20480', out: 'c_p · 4096', dest: ['DFlash2', 'injected K/V', 'window 2048'], lab: (l) => 'L' + l + ' → log ' + (l + 1), legend: [['KDA', 'a'], ['MLA + DSA', 'b']] },
        ds: { n: 43, taps: [40, 41, 42], type: dsType, title: 'DeepSeek · DSpark', cat: '[12288]', proj: 'main_proj · 4096 × 12288', out: 'H_ctx · 4096', dest: ['DSpark', 'main_norm', 'window 128'], lab: (l) => 'L' + l, legend: [['SWA', 'muted'], ['C4A', 'b'], ['C128A', 'f']] },
      };
      const TYPECOL = { glm: ['a', 'b'], ds: [null, 'b', 'muted', 'f'] };
      const S = { mode: 'glm', contract: true };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Who taps', options: [{ v: 'mtp', label: 'MTP (top only)' }, { v: 'glm', label: 'GLM · DFlash2' }, { v: 'ds', label: 'DeepSeek · DSpark' }], value: 'glm', onChange: (v) => { S.mode = v; } });
      UI.toggle(ctl, { label: 'hc_post + hc_contract (mean of the 4 streams)', value: true, onChange: (v) => { S.contract = v; } });
      const r = V.rng(5);
      const SV = Array.from({ length: 5 }, () => Array.from({ length: 4 }, () => Array.from({ length: 10 }, () => r.normal())));
      const OUT = Array.from({ length: 8 }, () => r.normal());
      const FC = V.randMatrix(8, 15, 3);
      V.fig(fig, (g, w, h, t) => {
        const C = V.C(), RC = RED(), narrow = w < 620;
        const Mo = MODES[S.mode], nt = Mo.taps.length;
        const contract = S.contract || S.mode === 'mtp';
        const tcol = (i) => { const k = S.mode === 'ds' ? TYPECOL.ds[Mo.type(i)] : TYPECOL.glm[Mo.type(i)]; return C[k]; };
        const tapCol = (k) => C.series[(k + 2) % C.series.length];
        V.text(g, Mo.title, 16, 20, { size: 13, weight: 700, color: C.ink });
        let lx = 16; Mo.legend.forEach(([l, c]) => { V.rr(g, lx, 30, 10, 10, 2); g.fillStyle = V.alpha(C[c], 0.8); g.fill(); V.text(g, l, lx + 14, 39, { size: 11, color: C.muted }); lx += tw(g, l, 11) + 30; });
        const nCells = narrow ? 6 : 10;
        const tapPos = [], slabPos = [];
        let catGeo;
        if (!narrow) {
          const sx = 16, sw = Math.min(140, w * 0.17), top = 54, bot = h - 20, lh = (bot - top) / Mo.n;
          for (let i = 0; i < Mo.n; i++) {
            const y = bot - (i + 1) * lh, tapped = Mo.taps.includes(i);
            V.rr(g, sx, y + 0.4, sw, Math.max(1.5, lh - 1.4), 2); g.fillStyle = V.alpha(tcol(i), tapped ? 0.95 : 0.45); g.fill();
            if (tapped) tapPos.push([sx + sw + 7, y + lh / 2]);
          }
          V.text(g, '0', sx - 4, bot - 2, { size: 11, font: 'mono', align: 'right', color: C.muted });
          const cs2 = Math.min(9, (w * 0.2) / nCells - 1.5), slabW = nCells * (cs2 + 1.5), slabH = 4 * (cs2 + 1.5);
          const mx0 = w * 0.36;
          for (let k = 0; k < nt; k++) { const yc = nt === 1 ? h * 0.3 : top + 20 + ((nt - 1 - k) + 0.5) * ((bot - top - 30) / nt); slabPos.push({ x: mx0, y: yc - slabH / 2, cs: cs2, w: slabW, h: slabH }); }
          catGeo = { mode: 'v', x: w * 0.625, cy: h * 0.53, cs: 9 };
          tapPos.forEach(([x, y], k) => {
            V.dot(g, x, y, 4.5, C.c); V.glow(g, x, y, 12, C.c, 0.4 + 0.3 * Math.sin(t * 3 + k));
            const sp = slabPos[k];
            bez(g, x + 4, y, sp.x - 8, sp.y + sp.h / 2, V.alpha(C.c, 0.5), 1.3);
          });
          // rótulos: escutas vizinhas (< 14 px) viram um rótulo só, com chave
          const groups = [];
          tapPos.forEach(([x, y], k) => { const gr = groups[groups.length - 1]; if (gr && Math.abs(gr.y1 - y) < 14) { gr.y1 = y; gr.k1 = k; } else groups.push({ x, y0: y, y1: y, k0: k, k1: k }); });
          groups.forEach((gr) => {
            if (gr.k0 === gr.k1) { V.text(g, Mo.lab(Mo.taps[gr.k0]), gr.x + 9, gr.y0 + 4, { size: 11, font: 'mono', color: C.ink }); return; }
            const bx = gr.x + 10, ya = Math.min(gr.y0, gr.y1) - 4, yb = Math.max(gr.y0, gr.y1) + 4, ym = (ya + yb) / 2;
            g.beginPath(); g.moveTo(bx, ya); g.lineTo(bx + 4, ya); g.lineTo(bx + 4, yb); g.lineTo(bx, yb); g.moveTo(bx + 4, ym); g.lineTo(bx + 8, ym);
            g.strokeStyle = V.alpha(C.ink, 0.7); g.lineWidth = 1.2; g.stroke();
            V.text(g, 'L' + Mo.taps[gr.k0] + '–' + Mo.taps[gr.k1], bx + 11, ym + 4, { size: 11, font: 'mono', color: C.ink });
          });
        } else {
          const sx = 16, top = 50, bh2 = 50, lw = (w - 32) / Mo.n;
          for (let i = 0; i < Mo.n; i++) {
            const x = sx + i * lw, tapped = Mo.taps.includes(i);
            V.rr(g, x + 0.3, top, Math.max(1.5, lw - 1.2), bh2, 1.5); g.fillStyle = V.alpha(tcol(i), tapped ? 0.95 : 0.45); g.fill();
            if (tapped) tapPos.push([x + lw / 2, top + bh2 + 8]);
          }
          const slot = (w - 32) / Math.max(nt, 1), cs2 = Math.min(8, (slot - 12) / nCells - 1.5), slabW = nCells * (cs2 + 1.5), slabH = 4 * (cs2 + 1.5);
          for (let k = 0; k < nt; k++) slabPos.push({ x: 16 + k * slot + (slot - slabW) / 2, y: 178, cs: cs2, w: slabW, h: slabH });
          catGeo = { mode: 'h', cx: w / 2, y: 262, cs: 8 };
          tapPos.forEach(([x, y], k) => {
            V.dot(g, x, y, 4, C.c);
            const lab = S.mode === 'mtp' ? 'final' : 'L' + Mo.taps[k];
            const lx2 = clamp(x, 16 + tw(g, lab, 11, 'mono') / 2, w - 16 - tw(g, lab, 11, 'mono') / 2);
            V.text(g, lab, lx2, y + 17, { size: 11, font: 'mono', align: 'center', color: C.ink });
            const sp = slabPos[k];
            V.line(g, x, y + 22, sp.x + sp.w / 2, sp.y - 8, { color: V.alpha(C.c, 0.5), lw: 1.2 });
          });
          const nl = contract ? (S.mode === 'ds' ? 'each tap: 4 streams → mean → 4096' : 'each tap: 4 streams × 4096 → hc_contract') : 'each tap delivers [t, 4, 4096]';
          const nlw = Math.min(w - 24, tw(g, nl, 11, 'mono'));
          V.box(g, w / 2 - nlw / 2 - 6, 164 - 13, nlw + 12, 18, { r: 6, fill: V.alpha(C.bg, 0.94) });
          V.text(g, nl, w / 2, 164, { size: 11, font: 'mono', align: 'center', color: contract ? C.muted : RC, maxW: w - 24 });
        }
        // slabs de 4 fluxos → contração
        const catSlots = [];
        slabPos.forEach((sp, k) => {
          const f = (t * 0.42 + k * 0.08) % 1;
          const c = contract ? V.smooth(clamp((f - 0.35) / 0.3, 0, 1)) : 0;
          const midY = sp.y + 1.5 * (sp.cs + 1.5);
          for (let s = 0; s < 4; s++) {
            const baseY = sp.y + s * (sp.cs + 1.5);
            const yy = V.lerp(baseY, midY, c);
            const lit = f < 0.35 ? (f / 0.35) * 4 > s : true;
            const a = (1 - c * 0.85) * (lit ? 1 : 0.35);
            for (let j = 0; j < nCells; j++) { g.fillStyle = cellFill(tapCol(k), SV[k % 5][s][j], a); g.fillRect(sp.x + j * (sp.cs + 1.5), yy, sp.cs, sp.cs); }
          }
          if (contract && c > 0) {
            for (let j = 0; j < nCells; j++) { const mv = (SV[k % 5][0][j] + SV[k % 5][1][j] + SV[k % 5][2][j] + SV[k % 5][3][j]) / 4; g.fillStyle = cellFill(tapCol(k), mv * 1.6, c); g.fillRect(sp.x + j * (sp.cs + 1.5), midY, sp.cs, sp.cs); }
          }
          if (!contract && f > 0.35 && f < 0.75) { g.strokeStyle = RC; g.lineWidth = 1.6; V.rr(g, sp.x - 4, sp.y - 4, sp.w + 6, sp.h + 6, 4); g.stroke(); }
          if (k === 0 && !narrow) {
            const lab = contract ? (c > 0.5 ? (S.mode === 'ds' ? 'mean → 4096' : 'hc_contract → 4096') : '4 streams × 4096') : '[t, 4, 4096]';
            V.text(g, lab, sp.x + sp.w / 2, sp.y - 6, { size: 11, font: 'mono', align: 'center', color: contract ? C.muted : RC, maxW: narrow ? sp.w + 14 : sp.w + 60 });
          }
          catSlots.push({ sp, f, midY });
        });
        // concat → proj → saída → destino
        const outCol = S.mode === 'mtp' ? C.glm : S.mode === 'ds' ? C.ds : C.glm;
        const bad = !contract;
        let catEnd, projG, outG, destG;
        if (catGeo.mode === 'v') {
          const segN = S.mode === 'mtp' ? 0 : nt * 3, cs3 = catGeo.cs, H3 = segN * (cs3 + 1.5);
          const cy0 = catGeo.cy - H3 / 2;
          if (segN) {
            for (let q = 0; q < segN; q++) { const k = nt - 1 - Math.floor(q / 3); g.fillStyle = cellFill(tapCol(k), SV[k % 5][0][q % 3] + 0.8, bad ? 0.25 : 1); V.rr(g, catGeo.x - cs3 / 2, cy0 + q * (cs3 + 1.5), cs3, cs3, 2); g.fill(); }
            V.text(g, Mo.cat, catGeo.x, cy0 + H3 + 14, { size: 11, font: 'mono', align: 'center', color: C.muted });
            catEnd = [catGeo.x + cs3 / 2 + 4, catGeo.cy];
          }
          catSlots.forEach(({ sp, f, midY }, k) => {
            const tx = segN ? catGeo.x - cs3 / 2 - 4 : w * 0.62 - 10, tyy = segN ? cy0 + ((nt - 1 - k) * 3 + 1.5) * (cs3 + 1.5) : catGeo.cy;
            bez(g, sp.x + sp.w + 4, midY + sp.cs / 2, tx, tyy, V.alpha(bad ? RC : tapCol(k), 0.45), 1.2);
            if (!bad && f > 0.65) { const p = bezPt(sp.x + sp.w + 4, midY + sp.cs / 2, tx, tyy, (f - 0.65) / 0.35); V.dot(g, p[0], p[1], 3, tapCol(k)); }
          });
          if (S.mode !== 'mtp') {
            const px0 = w * 0.67, pw = w * 0.12, pcw = (pw - 14 * 1.5) / 15;
            projG = gridM(g, px0, catGeo.cy - 8 * (8 + 1.5) / 2, FC, pcw, 8, outCol, bad ? 0.2 : 1);
            V.text(g, Mo.proj, px0 + pw / 2, catGeo.cy - 50, { size: 11, font: 'mono', align: 'center', color: C.muted, maxW: pw + 60 });
            V.arrow(g, catEnd[0], catEnd[1], px0 - 4, catGeo.cy, { color: V.alpha(C.muted, bad ? 0.2 : 0.6), lw: 1.3, head: 6 });
            outG = { x: w * 0.835 };
            vecV(g, outG.x, catGeo.cy - (8 * 11 - 2) / 2, OUT, 9, outCol, bad ? 0.2 : 1);
            V.arrow(g, px0 + pw + 4, catGeo.cy, outG.x - 9, catGeo.cy, { color: V.alpha(C.muted, bad ? 0.2 : 0.6), lw: 1.3, head: 6 });
            V.text(g, Mo.out, outG.x, catGeo.cy + 56, { size: 11, font: 'mono', align: 'center', color: C.muted, maxW: 90 });
            destG = { x: w * 0.868, y: catGeo.cy - 46, w: w - 10 - w * 0.868, h: 92 };
          } else {
            outG = { x: w * 0.62 };
            vecV(g, outG.x, catGeo.cy - (8 * 11 - 2) / 2, OUT, 9, outCol, 1);
            V.text(g, Mo.out, outG.x, catGeo.cy + 58, { size: 11, font: 'mono', align: 'center', color: C.muted });
            destG = { x: w * 0.7, y: catGeo.cy - 46, w: Math.min(220, w * 0.26), h: 92 };
            V.arrow(g, outG.x + 9, catGeo.cy, destG.x - 4, catGeo.cy, { color: V.alpha(C.muted, 0.6), lw: 1.3, head: 6 });
            tokChar(g, destG.x + destG.w / 2, destG.y + destG.h + 34, 20, C.a);
            V.text(g, 'x(t+1): shifted token', destG.x + destG.w / 2, destG.y + destG.h + 60, { size: 11, align: 'center', color: C.muted });
            V.arrow(g, destG.x + destG.w / 2, destG.y + destG.h + 22, destG.x + destG.w / 2, destG.y + destG.h + 4, { color: C.a, lw: 1.3, head: 6 });
          }
        } else {
          const segN = S.mode === 'mtp' ? 0 : nt * 3, cs3 = catGeo.cs, W3 = segN * (cs3 + 1.5);
          const cx0 = catGeo.cx - W3 / 2;
          if (segN) {
            for (let q = 0; q < segN; q++) { const k = Math.floor(q / 3); g.fillStyle = cellFill(tapCol(k), SV[k % 5][0][q % 3] + 0.8, bad ? 0.25 : 1); V.rr(g, cx0 + q * (cs3 + 1.5), catGeo.y - cs3 / 2, cs3, cs3, 2); g.fill(); }
            V.text(g, Mo.cat, catGeo.cx + W3 / 2 + 8, catGeo.y + 4, { size: 11, font: 'mono', color: C.muted });
          }
          catSlots.forEach(({ sp, f, midY }, k) => {
            const tx = segN ? cx0 + (k * 3 + 1.5) * (cs3 + 1.5) : catGeo.cx, tyy = catGeo.y - cs3 / 2 - 4;
            V.line(g, sp.x + sp.w / 2, sp.y + sp.h + 4, tx, tyy, { color: V.alpha(bad ? RC : tapCol(k), 0.45), lw: 1.2 });
            if (!bad && f > 0.65) { const p = V.along(sp.x + sp.w / 2, sp.y + sp.h + 4, tx, tyy, (f - 0.65) / 0.35); V.dot(g, p[0], p[1], 3, tapCol(k)); }
          });
          let yN = catGeo.y + 20;
          if (S.mode !== 'mtp') {
            const pw = w * 0.56, px0 = (w - pw) / 2, pcw = (pw - 14 * 1.5) / 15;
            V.text(g, Mo.proj, w / 2, yN + 12, { size: 11, font: 'mono', align: 'center', color: C.muted, maxW: w - 30 });
            projG = gridM(g, px0, yN + 22, FC, pcw, 7, outCol, bad ? 0.2 : 1);
            yN += 22 + projG.h + 22;
          }
          outG = { y: yN };
          vecH(g, w / 2 - (8 * 11 - 2) / 2, yN, OUT, 9, outCol, bad ? 0.2 : 1);
          V.text(g, Mo.out, w / 2, yN + 22, { size: 11, font: 'mono', align: 'center', color: C.muted });
          destG = { x: w * 0.15, y: yN + 34, w: w * 0.7, h: S.mode === 'mtp' ? 64 : 76 };
          if (S.mode === 'mtp') { tokChar(g, w * 0.08, destG.y + destG.h / 2, 18, C.a); V.arrow(g, w * 0.08 + 12, destG.y + destG.h / 2, destG.x - 4, destG.y + destG.h / 2, { color: C.a, lw: 1.2, head: 5 }); }
        }
        // destino
        V.box(g, destG.x, destG.y, destG.w, destG.h, { r: 12, fill: V.alpha(outCol, bad ? 0.05 : 0.14), stroke: V.alpha(outCol, bad ? 0.3 : 0.8), lw: 1.5 });
        Mo.dest.forEach((l, i) => V.text(g, l, destG.x + destG.w / 2, destG.y + 22 + i * 18 + (Mo.dest.length === 2 ? 8 : 0), { size: i ? 11 : 12.5, font: i ? 'mono' : 'sans', weight: i ? 400 : 700, align: 'center', color: i ? C.muted : C.ink, maxW: destG.w - 10 }));
        if (bad) {
          const bxw = Math.min(250, w - 32), bx = narrow ? (w - bxw) / 2 : w * 0.6 - 10, byy = narrow ? catGeo.y - 12 : h * 0.12;
          V.box(g, bx, byy, bxw, 44, { r: 10, fill: V.alpha(RC, 0.14), stroke: RC, lw: 1.5 });
          V.text(g, '✗ the hook delivers [tokens, 4, 4096]', bx + bxw / 2, byy + 18, { size: 11.5, weight: 700, align: 'center', color: RC, maxW: bxw - 10 });
          V.text(g, 'the drafter expects [tokens, 4096]', bx + bxw / 2, byy + 34, { size: 11, align: 'center', color: C.ink, maxW: bxw - 10 });
        }
      });
    })();

    /* ================================================================
       Fig 4 — A balança do rascunhador (bytes por passo, Qwen)
       ================================================================ */
    (function figBal() {
      const fig = root.querySelector('#fig-balanca'); if (!fig) return;
      adaptH(fig, 430, 420);
      const S = { v47: false, k: 3 };
      const SEG = [
        { id: 'lmT', l: 'target lm_head', c: 'd' }, { id: 'lmD', l: 'draft lm_head', c: 'd', draft: true }, { id: 'mtp', l: 'MTP layer', c: 'a', draft: true },
        { id: 'exp', l: 'experts', c: 'b' }, { id: 'gdn', l: 'GDN', c: 'e' }, { id: 'hc', l: 'hyper-conn.', c: 'f' }, { id: 'attn', l: 'attention+idx', c: 'c' }, { id: 'misc', l: 'other', c: 'muted' },
      ];
      const tgt = () => { const lm = S.v47 ? 0.22 : 0.59; return { lmT: 0.59, lmD: lm * S.k, mtp: (0.5 * S.k) / 3, exp: 2.1, gdn: 1.95, hc: 1.19, attn: 0.6, misc: 0.5 }; };
      const cur = Object.assign({}, tgt());
      const ctl = UI.controls(fig);
      UI.toggle(ctl, { label: '47k draft vocabulary', value: false, onChange: (v) => { S.v47 = v; upd(); } });
      UI.seg(ctl, { label: 'k', options: [1, 2, 3], value: 3, onChange: (v) => { S.k = v; upd(); } });
      const r1 = UI.readout(ctl, 'step per GPU', '—'), r2 = UI.readout(ctl, 'head’s share of draft', '—');
      function upd() { const T = tgt(); let tot = 0; for (const k in T) tot += T[k]; r1.set(fmt(tot, 2) + ' GiB'); r2.set(Math.round((100 * T.lmD) / (T.lmD + T.mtp)) + '%'); }
      upd();
      V.fig(fig, (g, w, h, t, dt) => {
        const C = V.C(), narrow = w < 620, T = tgt();
        for (const k in T) cur[k] += (T[k] - cur[k]) * Math.min(1, dt * 5);
        const cx = w / 2, Lb = Math.min(w * 0.33, 230), TH = 0.22;
        const fy = 34 + Lb * Math.sin(TH) + 14;
        const s = Math.min(48, (h * (narrow ? 0.24 : 0.26)) / 1.8);
        const Lw = cur.mtp, Rw = cur.lmD;
        const th = clamp((0.45 * (Rw - Lw)) / (Rw + Lw + 0.3), -TH, TH);
        const ex = Lb * Math.cos(th), ey = Lb * Math.sin(th);
        const strLen = s * 1.85 + 18;
        const baseY = fy + strLen + 40;
        // suporte
        g.fillStyle = V.alpha(C.muted, 0.35);
        g.beginPath(); g.moveTo(cx, fy); g.lineTo(cx - 9, baseY); g.lineTo(cx + 9, baseY); g.closePath(); g.fill();
        V.box(g, cx - 40, baseY, 80, 7, { r: 3, fill: V.alpha(C.muted, 0.45) });
        V.line(g, cx - ex, fy - ey, cx + ex, fy + ey, { color: C.ink, lw: 3 });
        V.dot(g, cx, fy, 6, C.ink);
        const pw = Math.min(130, w * 0.3);
        const pan = (px, py, side) => {
          const plY = py + strLen;
          V.line(g, px, py, px - pw / 2 + 6, plY, { color: V.alpha(C.muted, 0.7), lw: 1 });
          V.line(g, px, py, px + pw / 2 - 6, plY, { color: V.alpha(C.muted, 0.7), lw: 1 });
          g.beginPath(); g.moveTo(px - pw / 2, plY); g.quadraticCurveTo(px, plY + 16, px + pw / 2, plY); g.strokeStyle = C.ink; g.lineWidth = 2; g.stroke();
          const bw2 = pw * 0.62;
          if (side < 0) {
            const each = (s * cur.mtp) / S.k;
            for (let i = 0; i < S.k; i++) { const yy = plY - (i + 1) * each; V.box(g, px - bw2 / 2, yy, bw2, Math.max(2, each - 2), { r: 3, fill: V.alpha(C.a, 0.75), stroke: C.a, lw: 1 }); }
            V.text(g, 'MTP layer × ' + S.k, px, plY + 24, { size: 12, weight: 600, align: 'center', color: C.ink });
            V.text(g, fmt(cur.mtp, 2) + ' GiB', px, plY + 39, { size: 11, font: 'mono', align: 'center', color: C.muted });
          } else {
            const each = (s * cur.lmD) / S.k;
            for (let i = 0; i < S.k; i++) {
              const yy = plY - (i + 1) * each, hh = Math.max(2, each - 2);
              V.box(g, px - bw2 / 2, yy, bw2, hh, { r: 3, fill: V.alpha(C.d, 0.55), stroke: C.d, lw: 1 });
              g.strokeStyle = V.alpha(C.d, 0.9); g.lineWidth = 0.6; g.beginPath();
              for (let yl = yy + 3; yl < yy + hh - 1; yl += 3) { g.moveTo(px - bw2 / 2 + 3, yl); g.lineTo(px + bw2 / 2 - 3, yl); }
              g.stroke();
              if (hh >= 15 && i === S.k - 1) V.text(g, S.v47 ? '45,734 × 2560' : '124,160 × 2560', px, yy + hh / 2 + 4, { size: 11, font: 'mono', align: 'center', color: C.ink, maxW: bw2 - 4 });
            }
            V.text(g, (narrow ? 'lm_head × ' : 'draft lm_head × ') + S.k, px, plY + 24, { size: 12, weight: 600, align: 'center', color: C.ink, maxW: pw + 40 });
            V.text(g, fmt(cur.lmD, 2) + ' GiB', px, plY + 39, { size: 11, font: 'mono', align: 'center', color: C.muted });
          }
        };
        pan(cx - ex, fy - ey, -1); pan(cx + ex, fy + ey, 1);
        const pct = Math.round((100 * cur.lmD) / (cur.lmD + cur.mtp));
        V.text(g, `head = ${pct}% of the drafter’s bytes`, cx, 20, { size: 12.5, weight: 700, align: 'center', color: C.d });
        // barra do passo inteiro
        let tot = 0; for (const k in cur) tot += cur[k];
        const bx0 = 16, bx1 = w - 16, bY = narrow ? h - 112 : h - 74, bH = 28;
        V.text(g, narrow ? `step per GPU: ${fmt(tot, 2)} GiB · floor ≈ ${fmt((tot * GIB) / 240e9 * 1000, 0)} ms` : `one decode step per GPU: ${fmt(tot, 2)} GiB · bandwidth floor ≈ ${fmt((tot * GIB) / 240e9 * 1000, 0)} ms at 240 GB/s${S.k === 3 ? (S.v47 ? ' (measured: ~54 ms)' : ' (measured: ~60 ms)') : ''}`, bx0, bY - 10, { size: 12, weight: 600, color: C.ink, maxW: bx1 - bx0 });
        const sc2 = (bx1 - bx0) / 9.3;
        let x = bx0;
        SEG.forEach((sg) => {
          const v = cur[sg.id], ww = v * sc2;
          V.box(g, x, bY, Math.max(1, ww - 1.5), bH, { r: 3, fill: V.alpha(C[sg.c], sg.draft ? 0.85 : 0.5) });
          if (sg.draft) hatch(g, x, bY, ww - 1.5, bH, V.alpha(C.ink, 0.35), 5);
          if (!narrow && ww > tw(g, sg.l, 11) + 8) V.text(g, sg.l, x + ww / 2, bY + bH / 2 + 4, { size: 11, align: 'center', color: C.ink });
          sg.x = x; sg.w = ww; x += ww;
        });
        const Tsw = 1.4 * (tot / 9.2), u = (t % Tsw) / Tsw;
        V.line(g, bx0 + u * tot * sc2, bY - 4, bx0 + u * tot * sc2, bY + bH + 4, { color: C.c, lw: 2 });
        // legenda
        const cols = narrow ? 2 : 4, lwc = (bx1 - bx0) / cols;
        SEG.forEach((sg, i) => {
          const lx = bx0 + (i % cols) * lwc, ly = bY + bH + 18 + Math.floor(i / cols) * 17;
          V.rr(g, lx, ly - 9, 10, 10, 2); g.fillStyle = V.alpha(C[sg.c], sg.draft ? 0.85 : 0.5); g.fill();
          V.text(g, `${sg.l} ${fmt(cur[sg.id], 2)}`, lx + 14, ly, { size: 11, color: sg.draft ? C.ink : C.muted, maxW: lwc - 16 });
        });
      });
    })();

    /* ================================================================
       Fig 5 — FR-Spec: argmax restrito (vocabulário de brinquedo)
       ================================================================ */
    (function figFR() {
      const fig = root.querySelector('#fig-frspec'); if (!fig) return;
      adaptH(fig, 470, 560);
      const TOK = ['the', 'of', ',', '.', '(', ')', '=', '\\n', 'a', 'to', 'return', 'self', 'kernel', 'Funktion', 'Kyoto', '量子', 'quasar', 'Zanzibar', '∮', '<eos>'];
      const NV = TOK.length, D = 6;
      const inS = (i) => i < 10 || i === 19;
      const SIDS = []; for (let i = 0; i < NV; i++) if (inS(i)) SIDS.push(i);
      const W = V.randMatrix(NV, D, 4);
      const rg = V.rng(2026);
      const prior = TOK.map((_, i) => (i === 19 ? -2.4 : -0.9 * Math.log(i + 1)));
      const amx = (arr, ids) => { let b = -1, bv = -Infinity; (ids || arr.map((_, i) => i)).forEach((i) => { if (arr[i] > bv) { bv = arr[i]; b = i; } }); return b; };
      const CTX = [];
      for (let c = 0; c < 240; c++) { const tl = prior.map((p) => p + 1.3 * rg.normal()); const y = amx(tl); const dl = tl.map((v) => v + 0.7 * rg.normal()); CTX.push({ dl, y, xf: amx(dl), xs: amx(dl, SIDS) }); }
      const cum = [{ n: 0, f: 0, s: 0, out: 0 }];
      CTX.forEach((c, i) => { const p = cum[i]; cum.push({ n: p.n + 1, f: p.f + (c.xf === c.y ? 1 : 0), s: p.s + (c.xs === c.y ? 1 : 0), out: p.out + (inS(c.y) ? 0 : 1) }); });
      const S = { red: true, off: 0, fr: 1 };
      const ctl = UI.controls(fig);
      UI.toggle(ctl, { label: 'reduced vocabulary S (11 of 20 ids)', value: true, onChange: (v) => { S.red = v; } });
      UI.button(ctl, 'next context →', () => { S.off++; });
      V.fig(fig, (g, w, h, t, dt) => {
        const C = V.C(), RC = RED(), narrow = w < 620;
        S.fr += ((S.red ? 1 : 0) - S.fr) * Math.min(1, dt * 6);
        const red = S.fr > 0.5, fa = red ? (S.fr - 0.5) * 2 : (0.5 - S.fr) * 2;
        const ci = (Math.floor(t / 3.4) + S.off) % CTX.length, c = CTX[ci], stt = cum[ci + 1];
        const draft = red ? c.xs : c.xf, ok = draft === c.y;
        const ry0 = 44, rh = narrow ? (h * 0.7 - ry0) / NV : (h - ry0 - 16) / NV;
        const cw = narrow ? 8 : 12, chh = Math.max(8, rh - 5);
        const maxTokW = TOK.reduce((m, s) => Math.max(m, tw(g, s, 11.5, 'mono', 700)), 0);
        const labX = Math.ceil(maxTokW) + 20, wx = labX + 8, wW = D * (cw + 2) - 2;
        const dx = narrow ? wx + wW + 30 : wx + wW + 56;
        const mapX = dx + wW + 6;
        const lgX = red ? (narrow ? mapX + 40 : mapX + 58) : wx + wW + 26;
        const lgW = (narrow ? w - 12 : w - 212) - lgX;
        V.text(g, narrow ? 'lm_head' : 'lm_head · V=20 × d=6', wx + wW / 2 - (narrow ? 0 : 10), 20, { size: 12, weight: 700, align: 'center', color: C.ink });
        const rowY = (i) => ry0 + i * rh + rh / 2;
        for (let i = 0; i < NV; i++) {
          const y = rowY(i), inside = inS(i), dim = red && !inside ? 0.25 : 1;
          const isY = i === c.y;
          V.text(g, TOK[i], labX, y + 4, { size: 11.5, font: 'mono', align: 'right', color: isY ? C.e : dim < 1 ? C.muted : C.ink, weight: isY ? 700 : 400 });
          if (isY) V.dot(g, labX - tw(g, TOK[i], 11.5, 'mono', 700) - 8, y, 3.5, C.e);
          for (let j = 0; j < D; j++) { g.fillStyle = cellFill(red && inside ? C.a : C.d, W[i][j], dim); g.fillRect(wx + j * (cw + 2), y - chh / 2, cw, chh); }
          if (red && inside) { g.strokeStyle = V.alpha(C.a, 0.8); g.lineWidth = 1; g.strokeRect(wx - 1.5, y - chh / 2 - 1.5, wW + 3, chh + 3); }
        }
        const dRowY = (j) => ry0 + (NV - SIDS.length) * rh / 2 + j * rh + rh / 2;
        if (red) {
          g.globalAlpha = fa;
          V.text(g, narrow ? 'W_draft' : 'W_draft · 11 × 6', dx + wW / 2, 20, { size: 12, weight: 700, align: 'center', color: C.a });
          V.text(g, 'j→id', mapX + 2, 36, { size: 11, font: 'mono', color: C.muted });
          SIDS.forEach((id, j) => {
            const y = dRowY(j);
            bez(g, wx + wW + 3, rowY(id), dx - 3, y, V.alpha(C.a, 0.4), 1);
            for (let q = 0; q < D; q++) { g.fillStyle = cellFill(C.a, W[id][q]); g.fillRect(dx + q * (cw + 2), y - chh / 2, cw, chh); }
            if (!narrow || j === SIDS.length - 1 || j < 2) V.text(g, j + '→' + id, mapX + 2, y + 4, { size: 11, font: 'mono', color: id !== j ? C.c : C.muted, weight: id !== j ? 700 : 400 });
          });
          g.globalAlpha = 1;
        }
        // logits
        const rows = red ? SIDS.map((id, j) => ({ id, y: dRowY(j) })) : TOK.map((_, i) => ({ id: i, y: rowY(i) }));
        let mn = Infinity, mx = -Infinity; rows.forEach((q) => { mn = Math.min(mn, c.dl[q.id]); mx = Math.max(mx, c.dl[q.id]); });
        g.globalAlpha = fa;
        V.text(g, red ? (narrow ? 'logits (S)' : 'logits of S only') : (narrow ? 'logits' : 'logits over the whole vocabulary'), lgX + Math.max(20, lgW) / 2, 20, { size: 12, weight: 700, align: 'center', color: C.ink, maxW: Math.max(80, lgW + 30) });
        rows.forEach((q) => {
          const v = (c.dl[q.id] - mn) / (mx - mn || 1), len = Math.max(3, (0.12 + 0.88 * v) * Math.max(10, lgW - 8));
          const isD = q.id === draft;
          V.rr(g, lgX, q.y - chh * 0.38, len, chh * 0.76, 3); g.fillStyle = isD ? C.c : V.alpha(red ? C.a : C.d, 0.5); g.fill();
          if (isD) V.text(g, '★', lgX + len + 4, q.y + 4, { size: 11, color: C.c });
          if (q.id === c.y) V.dot(g, lgX + len + (isD ? 18 : 7), q.y, 3.5, C.e);
        });
        g.globalAlpha = 1;
        // veredito
        let px0, py0, pw;
        if (!narrow) { px0 = w - 196; py0 = 44; pw = 184; } else { px0 = 12; py0 = ry0 + NV * rh + 12; pw = w - 24; }
        V.box(g, px0, py0, pw, narrow ? h - py0 - 10 : Math.min(300, h - 60), { r: 12, fill: V.alpha(C.panel || C.bg, 0.5), stroke: V.alpha(C.muted, 0.4), lw: 1 });
        const colL = px0 + 14;
        if (!narrow) {
          V.text(g, 'draft', colL, py0 + 24, { size: 11, color: C.muted }); tokChar(g, colL + 10, py0 + 44, 18, C.c); V.text(g, '“' + TOK[draft] + '”', colL + 26, py0 + 49, { size: 13, font: 'mono', weight: 700, color: C.ink });
          V.text(g, 'target picks', colL, py0 + 80, { size: 11, color: C.muted }); tokChar(g, colL + 10, py0 + 100, 18, C.e); V.text(g, '“' + TOK[c.y] + '”', colL + 26, py0 + 105, { size: 13, font: 'mono', weight: 700, color: C.ink });
          V.text(g, ok ? '✓ accepted' : '✗ rejected', colL, py0 + 140, { size: 15, weight: 700, color: ok ? C.e : RC });
          V.text(g, ok ? 'a free token' : 'the target emits “' + TOK[c.y] + '”', colL, py0 + 158, { size: 11, color: C.ink, maxW: pw - 24 });
          if (red && !inS(c.y)) V.text(g, 'target outside S', colL, py0 + 174, { size: 11, weight: 600, color: C.d });
          const base = py0 + 200;
          V.text(g, `contexts: ${stt.n}`, colL, base, { size: 11, font: 'mono', color: C.muted });
          V.text(g, `acc. full        ${Math.round((100 * stt.f) / stt.n)}%`, colL, base + 17, { size: 11, font: 'mono', color: C.ink });
          V.text(g, `acc. with S      ${Math.round((100 * stt.s) / stt.n)}%`, colL, base + 34, { size: 11, font: 'mono', color: C.ink });
          V.text(g, `target outside S ${Math.round((100 * stt.out) / stt.n)}%`, colL, base + 51, { size: 11, font: 'mono', color: C.muted });
          V.text(g, 'different output: 0', colL, base + 70, { size: 11, font: 'mono', weight: 700, color: C.e });
          V.text(g, '★ draft · ● target’s choice', colL, base + 92, { size: 11, color: C.muted, maxW: pw - 24 });
        } else {
          const y1 = py0 + 24;
          tokChar(g, colL + 8, y1, 16, C.c); V.text(g, 'draft “' + TOK[draft] + '”', colL + 22, y1 + 4, { size: 11.5, font: 'mono', color: C.ink, maxW: pw / 2 - 30 });
          tokChar(g, px0 + pw / 2 + 8, y1, 16, C.e); V.text(g, 'target “' + TOK[c.y] + '”', px0 + pw / 2 + 22, y1 + 4, { size: 11.5, font: 'mono', color: C.ink, maxW: pw / 2 - 30 });
          V.text(g, ok ? '✓ accepted: a free token' : '✗ rejected: the target emits “' + TOK[c.y] + '”', colL, y1 + 30, { size: 12.5, weight: 700, color: ok ? C.e : RC, maxW: pw - 24 });
          V.text(g, `${stt.n} context${stt.n === 1 ? "" : "s"} · acc. full ${Math.round((100 * stt.f) / stt.n)}% · with S ${Math.round((100 * stt.s) / stt.n)}%`, colL, y1 + 52, { size: 11, font: 'mono', color: C.ink, maxW: pw - 24 });
          V.text(g, `target outside S ${Math.round((100 * stt.out) / stt.n)}% · different output: 0`, colL, y1 + 70, { size: 11, font: 'mono', color: C.muted, maxW: pw - 24 });
          V.text(g, '★ draft · ● target’s choice', colL, y1 + 90, { size: 11, color: C.muted, maxW: pw - 24 });
        }
      });
    })();

    /* ================================================================
       Fig 6 — Duas GPUs, um argmax (faixas de id e redução por pares)
       ================================================================ */
    (function figTP() {
      const fig = root.querySelector('#fig-tp'); if (!fig) return;
      adaptH(fig, 480, 560);
      const HISTPT = [7518, 6766, 6096, 5509, 4967, 4567, 4219, 3810, 3517, 3272, 3029, 2695, 65, 14, 9, 2, 4, 1, 163, 480, 646, 658, 960, 880, 876, 861, 874, 875, 767, 761, 642, 33];
      const HIST47 = [7138, 6036, 5156, 4480, 3903, 3559, 3228, 2879, 2645, 2450, 2231, 1941, 64, 14, 9, 2, 4, 1, 141, 183, 101, 186, 98, 93, 74, 93, 82, 90, 100, 65, 70, 33];
      const VOC = 248320, SPLIT = 124160, BIN = 8000, NB = 32;
      const binW = (b) => Math.min(BIN, VOC - b * BIN);
      const fillH = Array.from({ length: NB }, (_, b) => (b < 8 ? 8000 : b === 8 ? 1392 : b >= 16 ? 9 : 0));
      const balH = Array.from({ length: NB }, (_, b) => (b < 4 ? 8000 : b === 4 ? 768 : b === 15 ? 3840 : b >= 16 && b <= 18 ? 8000 : b === 19 ? 4928 : 0));
      const SC = {
        full: { label: 'full vocab', r: [124160, 124160], hist: Array.from({ length: NB }, (_, b) => binW(b)), acc: 'acceptance 56.5% (2026-09-05)' },
        v47: { label: '47k en+code', r: [45734, 1415], hist: HIST47, acc: 'tokens/step ≈ same as full vocab (2026-09-11)' },
        fill: { label: '65,536 · low ids', r: [65392, 144], hist: fillH, acc: 'acceptance 53.2% (2026-09-05)', sch: true },
        bal: { label: '65,536 · balance 2', r: [32768, 32768], hist: balH, acc: 'acceptance 47.8% · faster on 3 of 4 tasks: a real trade-off (2026-09-05)', sch: true },
        // 65,536-id Portuguese file (47k floor + 18,387 ids from pt Wikipedia), real histogram of the file, bins of 8,000
        pt: { label: '65,536 · 47k + pt', r: [56054, 9482], hist: HISTPT, acc: 'pt: accepted per draft 0.96 → 1.39 (2026-09-29)' },
      };
      const gib = (n) => (n * 2560 * 2) / GIB;
      const S = { sc: 'v47', local: true };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Vocabulary', options: [{ v: 'full', label: 'full' }, { v: 'v47', label: '47k (real)' }, { v: 'fill', label: '65k low ids' }, { v: 'bal', label: '65k balanced' }, { v: 'pt', label: '65k + pt' }], value: 'v47', onChange: (v) => { S.sc = v; upd(); } });
      UI.toggle(ctl, { label: 'use_local_argmax_reduction', value: true, onChange: (v) => { S.local = v; upd(); } });
      const r0 = UI.readout(ctl, 'rank 0', '—'), r1 = UI.readout(ctl, 'rank 1', '—'), rw = UI.readout(ctl, 'step waits for', '—');
      const eff = () => (S.local ? SC[S.sc] : SC.full);
      function upd() { const e = eff(); const a = gib(e.r[0]), b = gib(e.r[1]); r0.set(fmt(a, 2) + ' GiB'); r1.set(fmt(b, 3) + ' GiB'); rw.set(fmt(Math.max(a, b), 2) + ' GiB'); }
      upd();
      V.fig(fig, (g, w, h, t) => {
        const C = V.C(), RC = RED(), narrow = w < 620, E = eff();
        const x0 = 16, x1 = w - 16, W = x1 - x0;
        // painel A: histograma de ids
        const aTop = 58, aBot = narrow ? h * 0.35 : h * 0.39;
        V.text(g, 'draft vocabulary ids, in bins of 8,000', x0, 20, { size: 12, weight: 700, color: C.ink, maxW: W });
        const sx = (id) => x0 + (id / VOC) * W;
        V.box(g, sx(0), aTop, sx(SPLIT) - sx(0), aBot - aTop, { r: 4, fill: V.alpha(C.a, 0.07) });
        V.box(g, sx(SPLIT), aTop, sx(VOC) - sx(SPLIT), aBot - aTop, { r: 4, fill: V.alpha(C.b, 0.07) });
        E.hist.forEach((cnt, b) => {
          const bx = sx(b * BIN), bw = sx(b * BIN + binW(b)) - bx - 1, hh = (cnt / BIN) * (aBot - aTop - 8);
          const inR0 = b * BIN + binW(b) <= SPLIT;
          V.rr(g, bx, aBot - hh, Math.max(1, bw), Math.max(cnt ? 1 : 0, hh), 1.5);
          g.fillStyle = V.alpha(inR0 ? C.a : C.b, 0.85); g.fill();
        });
        V.line(g, sx(SPLIT), aTop - 6, sx(SPLIT), aBot + 4, { color: C.ink, lw: 1.3, dash: [4, 3] });
        V.text(g, '124,160', sx(SPLIT), aBot + 17, { size: 11, font: 'mono', align: 'center', color: C.ink });
        V.text(g, '0', x0, aBot + 17, { size: 11, font: 'mono', color: C.muted });
        V.text(g, '248,320', x1, aBot + 17, { size: 11, font: 'mono', align: 'right', color: C.muted });
        // rótulos dos ranks acima do histograma (as barras cheias não os cobrem)
        [[0, C.a, SPLIT / 2], [1, C.b, SPLIT * 1.5]].forEach(([rk, col, mid]) => {
          const l1 = 'rank ' + rk, l2 = ' · ' + fmtInt(E.r[rk]) + ' ids';
          const w1 = tw(g, l1, 11.5, 'sans', 700), w2 = tw(g, l2, 11, 'mono', 400), lx = sx(mid) - (w1 + w2) / 2;
          V.text(g, l1, lx, aTop - 10, { size: 11.5, weight: 700, color: col });
          V.text(g, l2, lx + w1, aTop - 10, { size: 11, font: 'mono', color: C.ink });
        });
        const note = (txt, cx, cy, col, wt) => { const nw = tw(g, txt, 11, 'sans', wt); V.box(g, cx - nw / 2 - 6, cy - 12, nw + 12, 17, { r: 6, fill: V.alpha(C.bg, 0.88) }); V.text(g, txt, cx, cy, { size: 11, align: 'center', color: col, weight: wt }); };
        if (SC[S.sc].sch && S.local) note('schematic distribution', sx(SPLIT * 1.5), aTop + 18, C.muted, 400);
        if (!S.local) note('without the flag, the cut isn’t used', sx(SPLIT * 1.5), aTop + 18, RC, 600);
        // painel B: corrida dos ranks
        const bTop = aBot + 52, bRow = 30;
        const labW = narrow ? 58 : 76, trackX = x0 + labW, trackW = W - labW - (narrow ? 70 : 100);
        const ga = [gib(E.r[0]), gib(E.r[1])], gmax = Math.max(...ga), full = 0.6;
        const PER = 2.4, f = (t % PER) / PER, prog = clamp(f / 0.75, 0, 1) * gmax;
        V.text(g, 'head read per draft (the step waits for the slowest)', x0, bTop - 8, { size: 11.5, weight: 600, color: C.ink, maxW: W });
        [0, 1].forEach((rk) => {
          const y = bTop + 6 + rk * (bRow + 6), col = rk ? C.b : C.a;
          V.text(g, 'rank ' + rk, x0, y + bRow / 2 + 4, { size: 12, weight: 700, color: col });
          V.box(g, trackX, y, trackW, bRow, { r: 6, fill: V.alpha(C.muted, 0.1), stroke: V.alpha(C.muted, 0.3), lw: 1 });
          const len = (ga[rk] / full) * trackW, done = Math.min(prog, ga[rk]) / full * trackW;
          V.box(g, trackX, y, Math.max(2, done), bRow, { r: 6, fill: V.alpha(col, 0.75) });
          if (prog > ga[rk] + 1e-4 && ga[rk] < gmax) { hatch(g, trackX + len, y + 2, (Math.min(prog, gmax) - ga[rk]) / full * trackW, bRow - 4, V.alpha(C.ink, 0.35), 5); if (f > 0.4) V.text(g, 'waiting', trackX + len + 6, y + bRow / 2 + 4, { size: 11, color: C.muted }); }
          V.text(g, fmt(ga[rk], ga[rk] < 0.1 ? 3 : 2) + ' GiB', trackX + trackW + 8, y + bRow / 2 + 4, { size: 11.5, font: 'mono', color: C.ink });
        });
        const barX = trackX + (gmax / full) * trackW;
        V.line(g, barX, bTop, barX, bTop + 2 * bRow + 18, { color: f > 0.75 ? C.c : V.alpha(C.c, 0.5), lw: f > 0.75 ? 2.5 : 1.5 });
        V.text(g, SC[S.local ? S.sc : 'full'].acc, x0, bTop + 2 * bRow + 34, { size: 11.5, color: C.muted, maxW: W });
        // painel C: redução do argmax
        const cTop = bTop + 2 * bRow + 62, cBot = h - 12, cMid = narrow ? cTop + 78 : (cTop + cBot) / 2 + 8;
        V.text(g, S.local ? 'pairwise argmax: each rank sends (value, id)' : 'default path: all-gather of the logits', x0, cTop, { size: 11.5, weight: 600, color: C.ink, maxW: W });
        const gbw = narrow ? 64 : 84, gbh = 34, gy0 = cMid - gbh - 6, gy1 = cMid + 6;
        [[gy0, C.a, 'rank 0'], [gy1, C.b, 'rank 1']].forEach(([yy, col, l]) => V.box(g, x0, yy, gbw, gbh, { r: 8, fill: V.alpha(col, 0.18), stroke: col, lw: 1.2, label: l, size: 11.5 }));
        const mW = narrow ? 100 : 110, mX = narrow ? w - mW - 14 : x0 + W * 0.62;
        V.box(g, mX, cMid - 20, mW, 40, { r: 10, fill: V.alpha(C.c, 0.14), stroke: C.c, lw: 1.3, label: 'global argmax', size: 11.5 });
        [[gy0 + gbh / 2, C.a], [gy1 + gbh / 2, C.b]].forEach(([yy, col], rk) => {
          const sx1 = x0 + gbw + 4, ex1 = mX - 4;
          V.line(g, sx1, yy, ex1, cMid + (rk ? 8 : -8), { color: V.alpha(col, 0.35), lw: S.local ? 2 : 9 });
          if (S.local) {
            const u = (t * 0.6 + rk * 0.5) % 1, p = V.along(sx1, yy, ex1, cMid + (rk ? 8 : -8), u);
            pill(g, p[0], p[1], '(m, id)', col, { size: 11, h: 18, fillA: 0.35 });
          } else {
            for (let q = 0; q < 14; q++) { const u = (t * 0.9 + q / 14 + rk * 0.03) % 1, p = V.along(sx1, yy, ex1, cMid + (rk ? 8 : -8), u); V.dot(g, p[0], p[1] + Math.sin(q * 7) * 2.5, 1.8, col); }
          }
        });
        const lab = S.local ? '2·tp = 4 values per token' : '248,320 values per token';
        if (!narrow) V.text(g, lab, mX + mW + 12, cMid + 4, { size: 12, font: 'mono', weight: 700, color: S.local ? C.e : RC, maxW: x1 - mX - mW - 14 });
        else V.text(g, lab, x0, cMid + 70, { size: 11.5, font: 'mono', weight: 700, color: S.local ? C.e : RC });
      });
    })();

    /* ================================================================
       Fig 7 — Oficina: cobertura × tamanho × enchimento
       ================================================================ */
    (function figOficina() {
      const fig = root.querySelector('#fig-oficina'); if (!fig) return;
      adaptH(fig, 450, 560);
      const S = { D: 8305, lg: 16, fill: true, bal: 2 };
      const Hn = (n) => (n <= 0 ? 0 : Math.log(n) + 0.5772156649 + 1 / (2 * n));
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Corpus', options: [{ v: 8305, label: 'old · 8,305 ids' }, { v: 47149, label: '30 MiB · 47,149 ids' }], value: 8305, onChange: (v) => { S.D = v; upd(); } });
      const sl = UI.slider(ctl, { label: '--size', min: 12, max: 17, step: 0.25, value: 16, fmt: (v) => fmtInt(2 ** v), onInput: (v) => { S.lg = v; upd(); } });
      UI.toggle(ctl, { label: '--fill-to-size', value: true, onChange: (v) => { S.fill = v; upd(); } });
      UI.seg(ctl, { label: '--balance-shards', options: [1, 2], value: 2, onChange: (v) => { S.bal = v; upd(); } });
      const rC = UI.readout(ctl, 'coverage', '—'), rP = UI.readout(ctl, 'filler', '—'), rG = UI.readout(ctl, 'critical rank', '—');
      function stats() {
        const N = Math.round(2 ** S.lg), real = Math.min(N, S.D), pad = S.fill ? Math.max(0, N - S.D) : 0, tot = real + pad;
        const cov = Hn(real) / Hn(S.D);
        // fração real dos ids do corpus na faixa do rank 0: 8.161/8.305 no antigo (144 no rank 1), 45.734/47.149 no de 47k
        const R0F = S.D === 8305 ? 8161 / 8305 : 45734 / 47149;
        const r0real = Math.round(R0F * real), r1real = real - r0real;
        let r0, r1;
        if (S.fill && S.bal === 2) { const half = tot / 2; r0 = Math.max(r0real, half); r1 = tot - r0; }
        else { r0 = Math.min(124160, r0real + pad); r1 = tot - r0; }
        return { N, real, pad, tot, cov, r0, r1, crit: (Math.max(r0, r1) * 5120) / GIB };
      }
      function upd() { const s = stats(); rC.set(fmt(100 * s.cov, 1) + '%'); rP.set(s.pad ? Math.round((100 * s.pad) / s.tot) + '%' : '0%'); rG.set(fmt(s.crit, 2) + ' GiB'); }
      upd();
      const NODES = ['served model', 'gen_draft_corpus.py', 'corpus.jsonl', 'build_draft_vocab.py', 'draft_vocab.txt', '/etc/vllm-draft-vocab.txt'];
      V.fig(fig, (g, w, h, t) => {
        const C = V.C(), narrow = w < 620, s = stats();
        // esteira
        const rowsN = narrow ? 2 : 1, perRow = narrow ? 3 : 6;
        const gapN = narrow ? 16 : 20, nodeW = (w - 32 - (perRow - 1) * gapN) / perRow, nodeH = 30;
        const pos = NODES.map((_, i) => { const rr = Math.floor(i / perRow), cc = i % perRow; const cc2 = rr % 2 ? perRow - 1 - cc : cc; return [16 + cc2 * (nodeW + gapN) + nodeW / 2, 22 + rr * 44]; });
        // um "pacote" percorre a esteira: pisca no nó em que está e anda só no vão entre as bordas (nunca por cima do texto)
        const NSEG = NODES.length - 1, pu = (t * 0.28) % 1, pseg = Math.floor(pu * NODES.length), pf = pseg === NSEG ? 0 : pu * NODES.length - pseg;
        for (let i = 0; i < NSEG; i++) {
          const [ax, ay] = pos[i], [bx, by2] = pos[i + 1]; const same = ay === by2, dir = bx > ax ? 1 : -1;
          const x1 = same ? ax + dir * nodeW / 2 + 2 : ax, y1 = same ? ay : ay + nodeH / 2 + 1, x2 = same ? bx - dir * nodeW / 2 - 2 : bx, y2 = same ? by2 : by2 - nodeH / 2 - 1;
          V.arrow(g, x1, y1, x2, y2, { color: V.alpha(C.muted, 0.6), lw: 1.3, head: 5 });
          if (i === pseg && pf > 0.55) { const p = V.along(x1, y1, x2, y2, (pf - 0.55) / 0.45); V.dot(g, p[0], p[1], 2.8, C.c); }
        }
        const busy = pf <= 0.55 ? pseg : -1;
        NODES.forEach((l, i) => { const [x, y] = pos[i]; const col = i === 0 ? C.qwen : i % 2 ? C.b : C.a; const on = i === busy; V.box(g, x - nodeW / 2, y - nodeH / 2, nodeW, nodeH, { r: 8, fill: V.alpha(col, on ? 0.3 : 0.14), stroke: on ? C.c : V.alpha(col, 0.7), lw: on ? 1.8 : 1 }); V.text(g, l, x, y + 4, { size: 11, font: 'mono', align: 'center', color: C.ink, maxW: nodeW - 8 }); });
        // gráfico log-log
        const top = 22 + rowsN * 44 + 18, gx0 = narrow ? 46 : 60, gx1 = w - 20, gy0 = top + (narrow ? 50 : 34), gy1 = h - 40;
        V.text(g, 'frequency × rank (log-log, illustrative)', gx0, top + 4, { size: 12, weight: 700, color: C.ink });
        V.text(g, `corpus coverage: ${fmt(100 * s.cov, 1)}%`, gx1, top + (narrow ? 22 : 4), { size: 12.5, weight: 700, align: 'right', color: C.a });
        const LX = (r) => gx0 + (Math.log2(r) / 17) * (gx1 - gx0), LY = (fv) => gy0 + (-Math.log10(fv) / 5.2) * (gy1 - gy0);
        V.line(g, gx0, gy1, gx1, gy1, { color: C.grid, lw: 1 }); V.line(g, gx0, gy0, gx0, gy1, { color: C.grid, lw: 1 });
        (narrow ? [[8192, '8k'], [32768, '32k'], [131072, '131k']] : [[8192, '8k'], [16384, '16k'], [32768, '32k'], [65536, '65k'], [131072, '131k']]).forEach(([r, l]) => { V.line(g, LX(r), gy1, LX(r), gy1 + 4, { color: C.muted, lw: 1 }); V.text(g, l, LX(r), gy1 + 16, { size: 11, font: 'mono', align: 'center', color: C.muted }); });
        V.text(g, '1', LX(1), gy1 + 16, { size: 11, font: 'mono', color: C.muted });
        V.text(g, 'id rank', gx1, gy1 + 30, { size: 11, align: 'right', color: C.muted });
        V.text(g, 'freq.', gx0 - 6, gy0 + 4, { size: 11, align: 'right', color: C.muted });
        const cut = Math.min(s.N, S.D);
        g.beginPath(); g.moveTo(LX(1), gy1);
        for (let q = 0; q <= 120; q++) { const r = 2 ** ((q / 120) * Math.log2(cut)); g.lineTo(LX(r), LY(1 / r)); }
        g.lineTo(LX(cut), gy1); g.closePath(); g.fillStyle = V.alpha(C.a, 0.3); g.fill();
        g.beginPath(); for (let q = 0; q <= 140; q++) { const r = 2 ** ((q / 140) * Math.log2(S.D)); if (q) g.lineTo(LX(r), LY(1 / r)); else g.moveTo(LX(r), LY(1 / r)); } g.strokeStyle = C.a; g.lineWidth = 2; g.stroke();
        V.line(g, LX(S.D), LY(1 / S.D), LX(S.D), gy1, { color: C.a, lw: 1.2, dash: [2, 3] });
        V.line(g, LX(s.N), gy0 - 6, LX(s.N), gy1, { color: C.c, lw: 2, dash: [5, 4] });
        const showFill = s.N > S.D && S.fill;
        const lab = (txt, x, y, align, col) => { const lw2 = tw(g, txt, 11, 'sans', 600), lx0 = align === 'right' ? x - lw2 : align === 'center' ? x - lw2 / 2 : x; V.box(g, lx0 - 4, y - 11, lw2 + 8, 15, { r: 4, fill: V.alpha(C.bg, 0.82) }); V.text(g, txt, x, y, { size: 11, align, color: col, weight: 600 }); };
        {
          const fl = 'end of corpus: ' + fmtInt(S.D) + ' distinct ids', right = LX(S.D) > gx1 - 190;
          // com o enchimento desenhado logo abaixo, o rótulo do fim do corpus sobe para não colidir
          const fy = showFill ? Math.min(LY(1 / S.D) - 8, gy1 - 48) : LY(1 / S.D) - 8;
          lab(fl, LX(S.D) + (right ? -6 : 6), fy, right ? 'right' : 'left', C.a);
        }
        if (s.N > S.D) {
          if (S.fill) { hatch(g, LX(S.D), gy1 - 22, LX(s.N) - LX(S.D), 20, V.alpha(C.d, 0.9), 5, 1.2); V.box(g, LX(S.D), gy1 - 22, LX(s.N) - LX(S.D), 20, { r: 2, stroke: C.d, lw: 1 }); { const el = 'filler: a guess by BPE order', ew = tw(g, el, 11, 'sans', 600); lab(el, clamp((LX(S.D) + LX(s.N)) / 2, gx0 + ew / 2 + 4, gx1 - ew / 2), gy1 - 28, 'center', C.d); } }
          else V.text(g, 'without --fill-to-size the file keeps ' + fmtInt(S.D) + ' ids', LX(S.D) + 6, gy1 - 10, { size: 11, color: C.muted, maxW: gx1 - LX(S.D) - 8 });
        }
        const lblN = '--size ' + fmtInt(s.N);
        const lnx = clamp(LX(s.N), gx0 + tw(g, lblN, 11, 'mono') / 2 + 2, gx1 - tw(g, lblN, 11, 'mono') / 2);
        V.text(g, lblN, lnx, gy0 - 10, { size: 11, font: 'mono', align: 'center', weight: 700, color: C.c });
        // leitura por rank
        const L1 = `real ids ${fmtInt(s.real)} · filler ${fmtInt(s.pad)}`;
        const L2 = `rank 0 ${fmtInt(s.r0)} · rank 1 ${fmtInt(s.r1)} → ${fmt(s.crit, 2)} GiB`;
        const measured = S.D === 8305 && s.N === 65536 && S.fill;
        const L3s = S.bal === 2 ? (narrow ? ['the measured case: acceptance 47.8%,', 'but faster on 3 of 4 tasks'] : ['the measured case: acceptance 47.8%, but faster on 3 of 4 tasks']) : (narrow ? ['the measured case: 65,392 ids on rank 0,', 'acceptance 53.2%'] : ['the measured case: 65,392 ids on rank 0, acceptance 53.2%']);
        const ibw = Math.min(gx1 - gx0 - 20, Math.max(tw(g, L1, 11, 'mono'), tw(g, L2, 11, 'mono'), measured ? Math.max(...L3s.map((l) => tw(g, l, 11.5, 'sans', 700))) : 0) + 20);
        const ibx = gx1 - ibw - 4, iby = gy0 + 8, ibh = measured ? 44 + 18 * L3s.length : 44;
        V.box(g, ibx, iby, ibw, ibh, { r: 8, fill: V.alpha(C.bg, 0.85), stroke: V.alpha(C.muted, 0.4), lw: 1 });
        V.text(g, L1, ibx + 10, iby + 17, { size: 11, font: 'mono', color: C.ink, maxW: ibw - 16 });
        V.text(g, L2, ibx + 10, iby + 34, { size: 11, font: 'mono', color: C.muted, maxW: ibw - 16 });
        if (measured) L3s.forEach((l, i) => V.text(g, l, ibx + 10, iby + 53 + 17 * i, { size: 11.5, weight: 700, color: RED(), maxW: ibw - 16 }));
      });
    })();

    /* ================================================================
       Fig 8 — O segundo turno e o bloco descartado
       ================================================================ */
    (function figBD() {
      const fig = root.querySelector('#fig-blockdrop'); if (!fig) return;
      adaptH(fig, 400, 428);
      const MODES = {
        bug: { label: 'day-0 without backport', hit: 0, ttft: 3.25, med: true, note: 'block drop + bug #62: the whole turn is recomputed' },
        drop: { label: 'block drop, without the bug', hit: 4, ttft: 1.21, med: false, note: 'drops the last matched block (1,664 tokens) and recomputes it' },
        keep: { label: 'disable_eagle_block_drop', hit: 5, ttft: 0.63, med: true, note: 'keeps the last matched block' },
        v030: { label: 'vLLM 0.30 stock', hit: 5, ttft: 0.28, med: true, note: 'native option and #62 fixed upstream; a different engine version' },
      };
      const S = { m: 'bug' };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Engine', options: Object.keys(MODES).map((k) => ({ v: k, label: MODES[k].label })), value: 'bug', onChange: (v) => { S.m = v; upd(); } });
      const rH = UI.readout(ctl, 'reused blocks', '—'), rT = UI.readout(ctl, 'TTFT', '—');
      function upd() { const M = MODES[S.m]; rH.set(M.hit + ' of 5'); rT.set(fmt(M.ttft, 2) + ' s' + (M.med ? '' : ' (est.)')); }
      upd();
      V.fig(fig, (g, w, h, t) => {
        const C = V.C(), RC = RED(), narrow = w < 620, M = MODES[S.m];
        const PER = 6, f = (t % PER) / PER;
        V.text(g, '2nd turn of a ~9.5k-token session · blocks of 1,664', 16, 20, { size: 12, weight: 700, color: C.ink, maxW: w - 32 });
        const x0 = 16, x1 = w - 16;
        const tailW = narrow ? 34 : 60, newW = narrow ? 44 : 78, gap = 5;
        const bw = (x1 - x0 - tailW - newW - gap * 6) / 5, by = 56, bh = narrow ? 46 : 54;
        const scanN = clamp((f / 0.22) * 5, 0, 5);
        for (let b = 0; b < 5; b++) {
          const x = x0 + b * (bw + gap);
          let fill = V.alpha(C.muted, 0.18), stroke = V.alpha(C.muted, 0.5), tag = '';
          if (b < scanN) {
            if (b < M.hit) { fill = V.alpha(C.e, 0.45); stroke = C.e; tag = 'hit'; }
            else if (S.m === 'drop' && b === 4) { fill = V.alpha(RC, 0.35); stroke = RC; tag = 'dropped'; }
            else { fill = V.alpha(RC, 0.25); stroke = RC; tag = 'recompute'; }
          }
          V.box(g, x, by, bw, bh, { r: 8, fill, stroke, lw: 1.5 });
          V.text(g, 'block ' + (b + 1), x + bw / 2, by + (narrow ? 18 : 22), { size: 11.5, weight: 600, align: 'center', color: C.ink, maxW: bw - 4 });
          if (tag) V.text(g, tag, x + bw / 2, by + bh - 10, { size: 11, align: 'center', color: C.ink, maxW: bw - 4 });
        }
        const tx = x0 + 5 * (bw + gap);
        V.box(g, tx, by, tailW, bh, { r: 8, fill: V.alpha(C.d, 0.18), stroke: V.alpha(C.d, 0.7), lw: 1.2 });
        V.text(g, narrow ? 'tail' : 'tail', tx + tailW / 2, by + bh / 2 + 4, { size: 11, align: 'center', color: C.ink, maxW: tailW - 4 });
        const nx = tx + tailW + gap;
        V.box(g, nx, by, newW, bh, { r: 8, fill: V.alpha(C.qwen || C.a, 0.2), stroke: C.qwen || C.a, lw: 1.2 });
        V.text(g, narrow ? 'new' : 'new msg', nx + newW / 2, by + bh / 2 + 4, { size: 11, align: 'center', color: C.ink, maxW: newW - 4 });
        V.text(g, '1,664 tokens each', x0, by + bh + 16, { size: 11, color: C.muted });
        // forno de prefill
        const ovW = narrow ? w - 32 : Math.min(280, w * 0.34), ovX = narrow ? 16 : x1 - ovW, ovY = by + bh + 36, ovH = narrow ? 84 : 96;
        V.box(g, ovX, ovY, ovW, ovH, { r: 12, fill: V.alpha(C.d, 0.08), stroke: V.alpha(C.d, 0.6), lw: 1.3 });
        V.text(g, 'prefill (recompute)', ovX + 12, ovY + 20, { size: 12, weight: 700, color: C.ink });
        const recompute = M.hit === 5 ? 0 : 5 - M.hit;
        const work = recompute === 5 ? 'the whole session again' : recompute ? '1 block of 1,664 + tail + message' : 'just the tail + the new message';
        const pf = clamp((f - 0.25) / 0.6, 0, 1);
        const barW = ovW - 24, fillW = barW * pf;
        V.box(g, ovX + 12, ovY + 32, barW, 14, { r: 7, fill: V.alpha(C.muted, 0.15) });
        V.box(g, ovX + 12, ovY + 32, Math.max(2, fillW), 14, { r: 7, fill: V.alpha(recompute ? RC : C.e, 0.75) });
        for (let q = 0; q < 6; q++) { const fx = ovX + 22 + q * ((ovW - 44) / 5), fl = 6 + 4 * Math.abs(Math.sin(t * 7 + q)); if (pf > 0 && pf < 1) { g.beginPath(); g.moveTo(fx - 4, ovY + ovH - 6); g.lineTo(fx, ovY + ovH - 6 - fl); g.lineTo(fx + 4, ovY + ovH - 6); g.closePath(); g.fillStyle = V.alpha(q % 2 ? C.d : C.c, 0.8); g.fill(); } }
        V.text(g, work, ovX + 12, ovY + 62, { size: 11.5, color: C.ink, maxW: ovW - 24 });
        // relógio TTFT
        const ttNow = M.ttft * pf;
        const clX = narrow ? 16 : 16, clY = narrow ? ovY + ovH + 22 : ovY;
        V.text(g, 'TTFT', clX, clY + 18, { size: 12, weight: 700, color: C.muted });
        V.text(g, fmt(ttNow, 2) + ' s', clX, clY + 52, { size: 30, font: 'display', weight: 700, color: M.ttft > 1 ? RC : C.e });
        V.text(g, M.med ? 'measured (median of 3)' : 'estimate', clX, clY + 70, { size: 11, color: C.muted });
        V.text(g, M.note, clX, clY + 90, { size: 11.5, color: C.ink, maxW: narrow ? w - 32 : ovX - clX - 16 });
        // o deslocamento do rascunhador
        const iy = narrow ? clY + 128 : h - 64, ix0 = 16, cells = narrow ? 8 : 10, cw = narrow ? 24 : 28;
        V.text(g, 'why it exists: the drafter’s input at p uses h(p) and token p+1', ix0, iy - 16, { size: 11.5, weight: 600, color: C.ink, maxW: w - 32 });
        for (let q = 0; q < cells; q++) {
          const x = ix0 + q * (cw + 3), inB = q < cells / 2;
          V.box(g, x, iy, cw, 18, { r: 4, fill: V.alpha(inB ? C.e : C.d, 0.3), stroke: V.alpha(inB ? C.e : C.d, 0.8), lw: 1 });
        }
        const pX = ix0 + (cells / 2 - 1) * (cw + 3) + cw / 2, nX = ix0 + (cells / 2) * (cw + 3) + cw / 2;
        V.line(g, ix0 + (cells / 2) * (cw + 3) - 1.5, iy - 4, ix0 + (cells / 2) * (cw + 3) - 1.5, iy + 22, { color: C.ink, lw: 1.5, dash: [3, 2] });
        g.beginPath(); g.moveTo(nX, iy + 20); g.quadraticCurveTo((pX + nX) / 2, iy + 44, pX, iy + 22); g.strokeStyle = C.c; g.lineWidth = 1.6; g.stroke();
        V.dot(g, pX, iy + 22, 3, C.c);
        V.text(g, 'p', pX, iy + 13, { size: 11, font: 'mono', align: 'center', color: C.ink });
        V.text(g, 'p+1', nX, iy + 13, { size: 11, font: 'mono', align: 'center', color: C.ink });
        const tx2 = ix0 + cells * (cw + 3) + 10;
        if (tx2 < w - 120) V.text(g, 'the block’s end depends on the next block', tx2, iy + 13, { size: 11, color: C.muted, maxW: w - tx2 - 16 });
      });
    })();

    /* ================================================================
       Fig 9 — IndexShare: reuso do top-k entre passos do rascunho
       ================================================================ */
    (function figIS() {
      const fig = root.querySelector('#fig-indexshare'); if (!fig) return;
      adaptH(fig, 460, 600);
      const NB = 48, K = 10;
      const rg = V.rng(48);
      const base = Array.from({ length: NB }, (_, b) => 0.35 * Math.sin(b * 0.37) + 0.3 * Math.sin(b * 1.9 + 1) + 0.25 * rg.normal() + (b % 11 === 3 ? 0.9 : 0));
      const drift = [0, 1, 2].map(() => Array.from({ length: NB }, () => 0.1 * rg.normal()));
      const score = (s) => base.map((v, b) => v + drift.slice(0, s + 1).reduce((a, d) => a + d[b], 0));
      const topk = (arr) => arr.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]).slice(0, K).map((x) => x[1]);
      const FRESH = [0, 1, 2].map((s) => topk(score(s)));
      const S = { share: true, step: 'auto' };
      const ctl = UI.controls(fig);
      UI.toggle(ctl, { label: 'index_share_for_mtp_iteration', value: true, onChange: (v) => { S.share = v; } });
      UI.seg(ctl, { label: 'Draft step', options: [{ v: 'auto', label: 'auto' }, { v: 0, label: '1' }, { v: 1, label: '2' }, { v: 2, label: '3' }], value: 'auto', onChange: (v) => { S.step = v; } });
      const GAIN = { code: [6.1, 4.6, 21.3, 26.6], prose: [0.3, 5.3, -0.1, -1.8] };
      V.fig(fig, (g, w, h, t) => {
        const C = V.C(), RC = RED(), narrow = w < 620;
        const si = S.step === 'auto' ? Math.floor(t / 1.8) % 3 : S.step, ph = (t % 1.8) / 1.8;
        const used = S.share ? FRESH[0] : FRESH[si];
        const fresh = FRESH[si];
        const sc = score(si);
        let mn = Infinity, mx = -Infinity; sc.forEach((v) => { mn = Math.min(mn, v); mx = Math.max(mx, v); });
        // trilhas dos 3 passos
        const x0 = 16, x1 = w - 16;
        V.text(g, 'kernel cost per draft step', x0, 20, { size: 12, weight: 700, color: C.ink });
        const laneH = 22, lanes = 3, lW = x1 - x0 - (narrow ? 44 : 60);
        for (let s = 0; s < lanes; s++) {
          const y = 32 + s * (laneH + 6), on = s === si;
          V.text(g, 'step ' + (s + 1), x0, y + 15, { size: 11.5, weight: on ? 700 : 400, color: on ? C.ink : C.muted });
          const lx = x0 + (narrow ? 44 : 60);
          const runIdx = !S.share || s === 0;
          const idxW = lW * 0.3, attW = lW * 0.36, restW = lW * 0.26;
          if (runIdx) { V.box(g, lx, y, idxW - 2, laneH, { r: 4, fill: V.alpha(C.d, on ? 0.85 : 0.45) }); V.text(g, narrow ? 'indexer' : 'score + top-k + expand', lx + idxW / 2, y + 15, { size: 11, align: 'center', color: C.ink, maxW: idxW - 8 }); }
          else { g.setLineDash([4, 3]); g.strokeStyle = V.alpha(C.e, on ? 0.9 : 0.5); g.lineWidth = 1.3; V.rr(g, lx + 1, y + 1, idxW - 4, laneH - 2, 4); g.stroke(); g.setLineDash([]); V.text(g, narrow ? 'reused' : 'skipped: step 1’s indices', lx + idxW / 2, y + 15, { size: 11, align: 'center', color: C.e, weight: 600, maxW: idxW - 8 }); }
          V.box(g, lx + idxW, y, attW - 2, laneH, { r: 4, fill: V.alpha(C.b, on ? 0.8 : 0.4) });
          V.text(g, narrow ? 'attention' : 'sparse attention', lx + idxW + attW / 2, y + 15, { size: 11, align: 'center', color: C.ink, maxW: attW - 8 });
          V.box(g, lx + idxW + attW, y, restW, laneH, { r: 4, fill: V.alpha(C.a, on ? 0.7 : 0.35) });
          V.text(g, narrow ? 'MoE+head' : 'MoE + head', lx + idxW + attW + restW / 2, y + 15, { size: 11, align: 'center', color: C.ink, maxW: restW - 6 });
        }
        // contexto em micro-blocos
        const gTop = 32 + lanes * (laneH + 6) + 26;
        const cols = narrow ? 12 : 24, rowsB = NB / cols;
        const cell = Math.min(narrow ? 24 : 28, (x1 - x0 - 90) / cols - 3);
        V.text(g, `context in micro-blocks of 4 tokens · budget of ${K} blocks (toy)`, x0, gTop - 8, { size: 11.5, weight: 600, color: C.ink, maxW: w - 32 });
        for (let b = 0; b < NB; b++) {
          const r = Math.floor(b / cols), cI = b % cols;
          const x = x0 + cI * (cell + 3), y = gTop + r * (cell + 3);
          const v = (sc[b] - mn) / (mx - mn || 1);
          const inU = used.includes(b), inF = fresh.includes(b);
          g.globalAlpha = inU ? 1 : 0.42;
          V.rr(g, x, y, cell, cell, 4); g.fillStyle = V.cmap(0.15 + 0.8 * v, 'heat'); g.fill();
          g.globalAlpha = 1;
          if (inU) { g.strokeStyle = C.a; g.lineWidth = 2.4; V.rr(g, x - 1.5, y - 1.5, cell + 3, cell + 3, 5); g.stroke(); }
          if (S.share && si > 0 && inF && !inU) { g.strokeStyle = RC; g.lineWidth = 1.6; g.setLineDash([3, 2]); V.rr(g, x - 1.5, y - 1.5, cell + 3, cell + 3, 5); g.stroke(); g.setLineDash([]); }
        }
        // cauda + query
        const lastR = rowsB - 1, tailX = x0 + cols * (cell + 3) + 8, tailY = gTop + lastR * (cell + 3);
        const tailN = 2 + si;
        for (let q = 0; q < Math.min(3, tailN); q++) { V.rr(g, tailX + q * 12, tailY + cell / 2 - 5, 10, 10, 2); g.fillStyle = V.alpha(C.c, 0.8); g.fill(); }
        if (tailN >= 4) V.text(g, 'new block', tailX, tailY - 4, { size: 11, color: RC, weight: 600 });
        V.text(g, 'tail', tailX, tailY + cell + 14, { size: 11, color: C.muted });
        tokChar(g, tailX + 66, tailY + cell / 2, 18, C.c);
        V.text(g, 'query ' + (si + 1), tailX + 66, tailY + cell + 14, { size: 11, align: 'center', color: C.ink });
        const ov = used.filter((b) => fresh.includes(b)).length;
        const iy = gTop + rowsB * (cell + 3) + (narrow ? 38 : 20);
        V.text(g, S.share ? `reused × fresh selection at this step: ${ov}/${K} blocks match` : `selection recomputed at this step (indexer run #${si + 1})`, x0, iy, { size: 12, weight: 700, color: S.share ? (ov === K ? C.e : C.d) : C.ink, maxW: w - 32 });
        V.text(g, S.share ? 'mint outline = used · red dashed = what a fresh selection would swap in' : 'mint outline = this step’s top-k', x0, iy + 17, { size: 11, color: C.muted, maxW: w - 32 });
        // ganhos medidos
        const mTop = iy + 34, mBot = h - 22, gw = x1 - x0;
        V.text(g, 'measured on Qwen: decode gain with IndexShare', x0, mTop + 6, { size: 11.5, weight: 600, color: C.ink, maxW: w - 32 });
        const topOff = narrow ? 50 : 30, zeroY = mTop + topOff + (mBot - mTop - topOff - 14) * (26.6 / 30), scY = (mBot - mTop - topOff - 14) / 30;
        V.line(g, x0, zeroY, x1, zeroY, { color: C.grid, lw: 1 });
        const groups = ['S=1', 'S=2', 'S=4', 'S=8'], gsl = gw / 4, bw2 = Math.min(26, gsl * 0.28);
        groups.forEach((lb, i) => {
          const gx = x0 + i * gsl + gsl / 2;
          [['code', C.a], ['prose', C.b]].forEach(([k, col], j) => {
            const v = GAIN[k][i], bx = gx + (j ? 3 : -bw2 - 3), hh = v * scY;
            V.box(g, bx, v >= 0 ? zeroY - hh : zeroY, bw2, Math.max(1.5, Math.abs(hh)), { r: 3, fill: V.alpha(col, 0.85) });
            V.text(g, (v > 0 ? '+' : '') + fmt(v, 1), bx + bw2 / 2, v >= 0 ? zeroY - hh - 4 : zeroY + Math.abs(hh) + 12, { size: 11, font: 'mono', align: 'center', color: C.ink });
          });
          V.text(g, lb, gx, mBot + 12, { size: 11, align: 'center', color: C.muted });
        });
        V.text(g, '■ code', x1, mTop + (narrow ? 24 : 6), { size: 11, align: 'right', color: C.a, weight: 600 });
        V.text(g, '■ prose', x1 - 70, mTop + (narrow ? 24 : 6), { size: 11, align: 'right', color: C.b, weight: 600 });
        void ph;
      });
    })();

    /* ================================================================
       Fig 10 — Gerador de --speculative-config (widget DOM)
       ================================================================ */
    (function figBuilder() {
      const fig = root.querySelector('#fig-builder'); if (!fig) return;
      const stage = fig.querySelector('.la-fig-stage');
      const h = LA.h;
      const S = { rec: 'qwen', q: { k: 3, vocab: true, bd: 1, is: true, fp8: false, lane: 'day0' }, g: { m: 'dflash', k: 2 }, d: { ck: 'vx', k: 6, bk: false } };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Recipe', options: [{ v: 'qwen', label: 'Qwen' }, { v: 'glm', label: 'GLM' }, { v: 'ds', label: 'DeepSeek' }], value: 'qwen', onChange: (v) => { S.rec = v; build(); } });
      const sub = h('div', { style: { display: 'contents' } });
      ctl.appendChild(sub);
      const outWrap = h('div', { style: { display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: '16px', alignItems: 'start' } });
      const left = h('div', { style: { minWidth: '0' } }), right = h('div', { style: { minWidth: '0' } });
      outWrap.append(left, right);
      stage.appendChild(outWrap);
      const esc = (s) => LA.esc(s);
      const rbc = { qwen: 'var(--qwen)', glm: 'var(--glm)', ds: 'var(--ds)' };
      function jsonPretty(o) { return '{\n' + Object.keys(o).map((k) => `  "${k}": ${JSON.stringify(o[k])}`).join(',\n') + '\n}'; }
      function item(kind, text) {
        const col = kind === 'ok' ? 'var(--viz-e)' : kind === 'warn' ? 'var(--viz-d)' : kind === 'err' ? 'var(--red)' : 'var(--viz-b)';
        const ic = kind === 'ok' ? '✓' : kind === 'warn' ? '⚠' : kind === 'err' ? '✗' : '•';
        return `<li style="list-style:none;margin:0 0 8px;padding-left:22px;position:relative;overflow-wrap:anywhere;font-size:13.5px;line-height:1.5"><b style="position:absolute;left:0;color:${col}">${ic}</b>${text}</li>`;
      }
      function render() {
        const L = [], mounts = [];
        let json = null, extra = '';
        if (S.rec === 'qwen') {
          const q = S.q, v030 = q.lane === 'v030';
          if (q.k > 0) {
            json = { method: 'mtp', num_speculative_tokens: q.k };
            if (q.vocab) json.use_local_argmax_reduction = true;
            if (q.bd === 1) json.disable_eagle_block_drop = true;
            if (q.is) json.index_share_for_mtp_iteration = true;
          }
          if (q.k === 0) L.push(item('info', 'MTP off: <code>start.sh</code> <b>omits</b> the flag (vLLM rejects <code>num_speculative_tokens: 0</code>).'));
          if (q.k === 0 && q.vocab) L.push(item('err', '<code>MTP_DRAFT_VOCAB is set but MTP_NUM_SPECULATIVE_TOKENS=0 - nothing drafts.</code>'));
          if (q.vocab && q.fp8 && !v030) L.push(item('err', '<code>MTP_DRAFT_VOCAB and FP8_DENSE both overlay nvidia/mtp.py - pick one.</code>'));
          if (q.k === 4) L.push(item('warn', 'k=4 was never measured in the recipe: ~1.9 GiB of <code>MemAvailable</code> on the head node, and larger captures risk a hard reset on the GB10.'));
          if (q.k > 0 && !q.vocab) L.push(item('warn', 'MTP drafting over the full 248,320-row head (0.59 GiB per rank on every draft).'));
          if (q.k > 0 && q.vocab) {
            const path = v030 ? 'models/qwen4_exp/nvidia/mtp.py' : 'models/qwen3_8_flash_next/nvidia/mtp.py';
            mounts.push(`${v030 ? 'mtp_v030_patched.py' : 'mtp_patched.py'} → …/${path}`);
            mounts.push('draft_vocab_en_code_47k.txt → /etc/vllm-draft-vocab.txt');
            mounts.push('env VLLM_MTP_DRAFT_VOCAB=/etc/vllm-draft-vocab.txt');
            L.push(item('ok', `Reduced vocabulary via <code>${v030 ? 'patch_mtp_draft_vocab_v030.py' : 'patch_mtp_draft_vocab.py'}</code>; in the log: <code>47149 of 248320 tokens (19.0%) … 0.59 -&gt; 0.22 GiB</code>.`));
          }
          if (q.k > 0 && q.bd === 1 && !v030) { mounts.push('block_drop/{config/speculative.py, v1/core/kv_cache_utils.py, v1/core/sched/scheduler.py} → …/vllm/');  L.push(item('ok', 'Step 4h: vllm#53388 backport mounted (3 files); log <code>EAGLE trailing prefix-cache block dropping is disabled</code>.')); }
          if (q.k > 0 && q.bd === 1 && v030) L.push(item('ok', '<code>disable_eagle_block_drop</code> is native in vLLM 0.30: no backport mounted.'));
          if (q.k > 0 && q.bd === 0) L.push(item('warn', 'Without the key, the prefix cache drops the last matched block (1,664 tokens); on the day-0 image the 2nd turn missed the entire cache (#62: 3.25 s TTFT).'));
          if (q.k > 0 && !q.is) L.push(item('warn', 'Without IndexShare, the QSA indexer runs at every draft step (code loses the +6 to +27%).'));
          if (q.k > 0 && q.is) L.push(item('ok', 'IndexShare: QSA top-k computed in draft-extend and reused in steps 2..k.'));
          if (q.fp8 && !(q.vocab)) L.push(item('info', '<code>FP8_DENSE</code> mounts <code>overlay/mtp.diff</code> (quant_config on the drafter’s mixer and <code>lm_head</code>).'));
        } else if (S.rec === 'glm') {
          const G = S.g;
          let caps = null;
          if (G.m === 'dflash') { json = { method: 'dflash', model: '<DFLASH_MODEL_DIR>', num_speculative_tokens: 7, kv_cache_dtype: 'auto', draft_sample_method: 'probabilistic', rejection_sample_method: 'standard', draft_tensor_parallel_size: 2 }; caps = [1, 2, 4, 8, 16, 24, 32]; L.push(item('ok', 'EAGLE-3 taps <code>(6, 15, 25, 34, 43)</code> via <code>patch_glm_eagle3.py</code> (baked in at build time).')); L.push(item('info', 'Shown without <code>GLM53_ADAPTIVE_K</code>; with it, <code>start.sh</code> adds the sizes n×(k+1) for k in {2, 4, 7}.')); }
          else if (G.m === 'mtp') {
            json = { method: 'mtp', num_speculative_tokens: G.k }; caps = [1, 2, 3, 4, 6, 8, 12];
            L.push(item(G.k === 2 ? 'ok' : 'info', `Native MTP (layer 45), k=${G.k}${G.k === 2 ? ': the recipe’s rollback (~24.6 tok/s, README baseline)' : ''}.`));
            const need = [1, 2, 3, 4].map((n) => n * (G.k + 1));
            const over = need.filter((x) => x > 12), padded = need.filter((x) => x <= 12 && !caps.includes(x));
            if (over.length) L.push(item('warn', `Batches of ${over.join(', ')} tokens exceed the largest captured size (12) and would run eager: adjust <code>--cudagraph-capture-sizes</code>.`));
            else L.push(item('ok', `Batches of ${need.join(', ')} tokens fit the list${padded.length ? ` (padded up to the next size: ${padded.join(', ')})` : ''}.`));
            L.push(item('info', '<code>index_share_for_mtp_iteration: true</code> already ships in the HF config: the DSA top-k from step 0 is reused.'));
            L.push(item('info', 'With <code>ABLIT=1</code>, <code>ABLIT_INCLUDE_MTP=1</code> also edits layer 45’s <code>o_proj</code>.'));
          } else L.push(item('info', '<code>SPEC_METHOD=none</code>: no <code>--speculative-config</code>.'));
          if (caps) extra = '--cudagraph-capture-sizes ' + caps.join(' ');
        } else {
          const D = S.d, np = D.ck === 'vx' ? 3 : 1;
          json = { method: 'dspark', num_speculative_tokens: D.k, draft_sample_method: 'probabilistic' };
          let ok = true;
          if (D.k < 5) { ok = false; L.push(item('err', 'k &lt; 5 truncates DSpark’s trained block (silent on vLLM 0.25.2, rejected from 0.26 on).')); }
          if (!D.bk && D.k > np && D.k % np !== 0) { ok = false; L.push(item('err', `MTP module reuse rule: with n_predict=${np}, k must be ≤ ${np} or a multiple of ${np}.`)); }
          if (ok) L.push(item('ok', `k=${D.k} is valid${D.bk ? ' (block-k exempts DSpark from the divisibility rule)' : ''}.`));
          if (ok && D.k === 6 && D.ck === 'vx') L.push(item('info', 'k=6 goes one position past the trained block (5): low acceptance there. With <code>DSPARK_ENABLE_DSPARK_BLOCK_K=1</code>, k=5 gives +2–10% single-stream.'));
          if (ok && D.k === 5 && D.ck === '0731') L.push(item('info', 'It’s the “MTP-5” of the old lanes: n_predict=1 accepts k=5.'));
          const cap = Math.floor((6 * (D.k + 1) + 7) / 8) * 8;
          extra = '--max-cudagraph-capture-size ' + cap + '   # MAX_NUM_SEQS=6 × (k+1), rounded up to a multiple of 8';
          if (!ok) json = null;
        }
        const col = rbc[S.rec];
        left.innerHTML = `<div style="font-family:var(--font-display);text-transform:uppercase;letter-spacing:.1em;font-size:12.5px;color:${col};margin-bottom:6px">--speculative-config</div>` +
          (json ? `<pre data-lang="json" style="margin:18px 0 0;font-size:clamp(11px,2.9vw,12.5px);overflow:visible;white-space:pre-wrap;overflow-wrap:anywhere"><code>${esc(jsonPretty(json))}</code></pre>` : '<pre style="margin:12px 0 0"><code>(no flag emitted)</code></pre>') +
          (extra ? `<pre data-lang="bash" style="margin:20px 0 0;font-size:clamp(11px,2.9vw,12.5px);overflow:visible;white-space:pre-wrap;overflow-wrap:anywhere"><code>${esc(extra)}</code></pre>` : '') +
          (mounts.length ? `<div style="margin-top:14px;font-size:12.5px;color:var(--text-3)">bind-mounts <code>:ro</code> and env in <code>docker run</code>:</div><ul style="margin:6px 0 0;padding:0">${mounts.map((m) => `<li style="list-style:none;margin:0 0 5px;font-family:var(--font-mono);font-size:12px;line-height:1.45;overflow-wrap:anywhere;color:var(--text-2)">${esc(m)}</li>`).join('')}</ul>` : '');
        right.innerHTML = `<div style="font-family:var(--font-display);text-transform:uppercase;letter-spacing:.1em;font-size:12.5px;color:var(--text-2);margin-bottom:10px">script checks</div><ul style="margin:0;padding:0">${L.join('')}</ul>`;
      }
      function build() {
        sub.innerHTML = '';
        if (S.rec === 'qwen') {
          const q = S.q;
          UI.seg(sub, { label: 'MTP_NUM_SPECULATIVE_TOKENS', options: [0, 1, 2, 3, 4], value: q.k, onChange: (v) => { q.k = v; render(); } });
          UI.seg(sub, { label: 'MTP_DRAFT_VOCAB', options: [{ v: true, label: '47k' }, { v: false, label: 'empty' }], value: q.vocab, onChange: (v) => { q.vocab = v; render(); } });
          UI.seg(sub, { label: 'MTP_DISABLE_BLOCK_DROP', options: [1, 0], value: q.bd, onChange: (v) => { q.bd = v; render(); } });
          UI.seg(sub, { label: 'MTP_INDEX_SHARE', options: [{ v: true, label: 'true' }, { v: false, label: 'false' }], value: q.is, onChange: (v) => { q.is = v; render(); } });
          UI.seg(sub, { label: 'FP8_DENSE', options: [{ v: false, label: 'false' }, { v: true, label: 'true' }], value: q.fp8, onChange: (v) => { q.fp8 = v; render(); } });
          UI.seg(sub, { label: 'Lane', options: [{ v: 'day0', label: 'day-0' }, { v: 'v030', label: 'vLLM 0.30' }], value: q.lane, onChange: (v) => { q.lane = v; render(); } });
        } else if (S.rec === 'glm') {
          const G = S.g;
          UI.seg(sub, { label: 'SPEC_METHOD', options: ['dflash', 'mtp', 'none'], value: G.m, onChange: (v) => { G.m = v; render(); } });
          UI.seg(sub, { label: 'MTP_TOKENS', options: [1, 2, 3, 4], value: G.k, onChange: (v) => { G.k = v; render(); } });
        } else {
          const D = S.d;
          UI.seg(sub, { label: 'Checkpoint', options: [{ v: 'vx', label: 'Vision-Exp (n_predict 3)' }, { v: '0731', label: '0731 (n_predict 1)' }], value: D.ck, onChange: (v) => { D.ck = v; render(); } });
          UI.seg(sub, { label: 'MTP_NUM_TOKENS', options: [3, 5, 6, 9], value: D.k, onChange: (v) => { D.k = v; render(); } });
          UI.toggle(sub, { label: 'DSPARK_ENABLE_DSPARK_BLOCK_K', value: D.bk, onChange: (v) => { D.bk = v; render(); } });
        }
        render();
      }
      build();
    })();
  },
});

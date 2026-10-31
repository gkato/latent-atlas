LA.chapter({
  id: 'dflash-dspark',
  part: 7,
  order: 37,
  title: 'DFlash2 and DSpark: block drafting',
  navTitle: 'DFlash2 & DSpark',
  subtitle: 'Drafters that fill the whole block in one forward: block diffusion, a bidirectional mask, dynamic conv and a selector (GLM), Markov and confidence heads (DeepSeek), adaptive k, the drafter’s state in vLLM and the same DFlash2 on TensorFold (equality verification, stop rules tuned offline, one ring per conversation).',
  minutes: 68,
  recipes: ['glm', 'tf', 'qwen', 'ds'],
  tags: ['speculative decoding', 'DFlash2', 'DSpark', 'block diffusion', 'adaptive-k', 'Markov head', 'rejection sampling', 'TensorFold'],
  goals: [
    'Explain why a <b>block</b> drafter costs almost the same for 1 or 7 positions, and what it loses in coherence',
    'Describe DFlash2 (target features, injected KV, 2-tap dynamic conv, top-16 selector with Gumbel) and DSpark (anchor + noise, rank-256 Markov head, confidence head)',
    'Decide <b>how much to verify</b>: k from the trained block, adaptive k via EMA, and scheduling by survival',
    'Configure the drafter’s state in vLLM (SWA KV group, compact pages, ring with per-request slots) and build the recipes’ <code>--speculative-config</code>',
    'Compare with GLM’s TensorFold lane: equality verification, <code>fc</code>/<code>fnc</code>/<code>fcost</code> stop rules tuned by offline replay, and one context ring per conversation',
  ],
  prereqs: ['mtp-eagle-vocab'],
  summary: [
    'Block drafting: <b>[anchor, MASK/noise…]</b> in a single forward with <b>bidirectional</b> attention inside the block; a causal <code>TRITON_ATTN</code> dropped GLM to 0.31 acceptance (versus 0.959 with <code>FLASH_ATTN</code>).',
    'DFlash2 (GLM): 1.17 B, 5 SWA-2048 layers, features from 5 target layers (<code>fc</code> 20480→4096) injected as K/V; 2-tap dynamic conv and a top-16 selector (rank 256, Gumbel walk) against suffix decay; k=7 (block 8).',
    'DSpark (DeepSeek): 3 V4 stages with MoE over features from layers 40–42, block γ=5; the Markov head \\(W_2W_1[y_i]\\) resolves the multimodal collision; the confidence head predicts \\(1-\\mathrm{TV}(p,q)\\).',
    'Tokens per step \\(=1+\\sum a_i\\): verifying low-acceptance positions costs rows. Adaptive-k (EMA α=0.25, margin 1, set {2,4,7}, batch minimum because of FULL graphs) gave +13–21% on prose.',
    'Drafter state: on GLM it is an SWA group with <i>padded slot-share</i> and a compact block of 896 (ids 65→6); on DeepSeek lane A it is a <code>SlidingWindowMLASpec</code> group in vLLM itself (blocks 4/8, filled by <code>precompute_and_store_context_kv</code>); on DeepSeek Stage-C it is a <code>pos % 128</code> ring outside vLLM, which requires per-request slots (Patch 1) and ragged batches via <code>query_start_loc</code> (Patch 2/2b).',
    'DeepSeek k=6: the Anemll lane applies the MTP divisibility rule (<code>n_predict=3</code>); the block-k hotfix unlocks k=5 (the trained block) with the same acceptance and a cheaper step.',
    'GLM’s TensorFold lane: the same DFlash2 verified by equality (reply = serial), with the walk using the target’s keyed noise; copy drafts before DFlash2; the <code>fnc7:0.3</code> rule (confidence \\(\\mathrm{softmax}((s+g)/\\beta)\\), β = 2.0 fitted by log loss on 9 offline dumps), +7% measured over <code>fc5:0.3</code>; a 4-bit drafter (0.63 GiB/rank) and its context in a 2,176-row ring per conversation (21 MiB/rank).',
  ],
  quiz: [
    { q: 'Pinning <code>TRITON_ATTN</code> on GLM’s DFlash2 drafter dropped acceptance to ~0.31. Why?', o: ['Triton doesn’t support BF16 on SM121', 'In this image it applies a causal mask inside the block, and the drafter was trained with bidirectional attention (is_causal: false)', 'Triton ignores the 2048 sliding window', 'Triton recomputes the context KV and loses the target’s features'], a: 1, e: 'The block [anchor, MASK×7] is predicted in parallel with full visibility between positions. The causal mask removes what the late positions learned to use: position 0 stays good and the following ones collapse. FLASH_ATTN does non-causal SWA.' },
    { q: 'In DSpark, what problem does the Markov head \\(W_2W_1[y_i]\\) solve?', o: ['It reduces the cost of the draft’s lm_head', 'It estimates the acceptance probability of each position', 'It replaces rejection sampling', 'Multimodal collision: parallel positions follow marginals and can blend two continuations; the bias conditions each position on the token actually chosen before it'], a: 3, e: 'The parallel backbone produces marginal logits. Adding a rank-256 bigram of the previous token, left to right, restores the dependency between positions at the cost of one GEMV. Estimating acceptance is the confidence head’s job.' },
    { q: 'Why does the DeepSeek recipe use <code>MTP_NUM_TOKENS=6</code> on Vision-Exp, if the trained block has 5?', o: ['Because the Anemll lane treats DSpark as MTP with n_predict = num_nextn_predict_layers = 3 and requires k ≥ 5 and k % 3 = 0', 'Because 6 gives more acceptance than 5', 'Because the CUDA graph only captures multiples of 6', 'Because the official card recommends 6'], a: 0, e: 'The divisibility rule exists for MTP, which re-executes the same layer; in DSpark the stages are stacked. The block-k hotfix exempts method=dspark and allows k=5, with the same per-position acceptance and a cheaper step (56.6 versus 55.3 tok/s at 8K).' },
    { q: 'GLM’s adaptive-k picks the <b>minimum</b> n across the requests in the batch. Why?', o: ['The minimum maximizes acceptance', 'To save KV memory', 'The FLASHINFER_MLA_SPARSE_SM120 and KDA backends only support FULL CUDA graphs with a uniform batch, so every request has to verify the same length', 'Because the EMA only exists per batch'], a: 2, e: 'Each length n requires a graph of n+1 rows per request (3, 5, 8), and the batch has to be uniform. Requests with structured output, or with no observations yet, pin the batch at 7.' },
    { q: 'With <code>max_num_seqs &gt; 1</code>, DSpark’s acceptance (Stage-C lane) went to ~0 without a crash. What did Patch 1 of the “Keys concurrency patch” fix?', o: ['A quantization bug in the drafter’s KV', 'The drafter’s context ring was indexed by batch row; continuous-batching condensation moves requests between rows, and each one started reading another request’s history. The patch maps req_id → a stable slot', 'The HTTP 500 on mixed prefill and decode batches', 'The missing CUDA graph for 6 sequences'], a: 1, e: 'Patch 1: _req_id_to_slot + _free_slots, with gather/scatter by slot_index. The HTTP 500 “got 41 rows for batch_size=2” is Patch 2/2b (ragged batch via query_start_loc).' },
    { q: 'What does <code>GLM53_DRAFT_KV_COMPACT=1</code> change on TP2?', o: ['It quantizes the drafter’s KV to FP8', 'It shrinks the drafter’s window from 2048 to 896', 'It uses as the drafter’s block the largest multiple of 64 that divides the MLA block (3584) and fits in the 2,351,104 B page, i.e. 896: ids per request 65 → 6, without reducing VRAM', 'It turns off the drafter’s prefix caching'], a: 2, e: '896 × 2048 B = 1,835,008 ≤ 2,351,104; the next candidate, 1792, doesn’t fit. The gain is in block ids (180 → 121 in the worst case; logged pool 1.85× at 14 GiB), not in bytes.' },
    { q: 'In the TensorFold lane, why can DFlash2 stop rules be compared offline, on the CPU, from a single dump of each request?', o: ['Because DFlash2 is deterministic in greedy and every dump is greedy', 'Because rejection sampling stores each draft’s \\(q(x)\\) in the dump', 'Because sampling is keyed by (seed, position, token): the committed tokens don’t depend on the rule, so a round starting at i keeps exactly the drafts equal to tokens i+1, i+2…, and a cost table turns rounds into tok/s', 'Because the dump records the CUDA graphs of every window width'], a: 2, e: 'With equality verification, the reply is the serial one whatever the rule; only the number of rounds and the cost of each change. That is why the simulator replays any rule (and the oracle, by dynamic programming) from the same dump. The dumps include sampled prose, not just greedy, and there is no \\(q(x)\\): that lane has no rejection sampling.' },
  ],
  mount(root, LA) {
    const V = LA.viz, UI = LA.ui;
    const RED = () => V.C().red || '#FB2C36';

    /* ------------------------------------------------ helpers comuns */
    function adaptH(fig, wide, narrow, bp = 600) {
      const st = fig.querySelector('.la-fig-stage');
      const set = () => { const want = (fig.clientWidth < bp ? narrow : wide) + 'px'; if (st.style.height !== want) st.style.height = want; };
      set();
      const ro = new ResizeObserver(set); ro.observe(fig);
      if (LA.onCleanup) LA.onCleanup(() => ro.disconnect());
    }
    const fmt = (x, d = 2) => (+x).toFixed(d);
    const fmtInt = (x) => Math.round(x).toLocaleString('en-US');
    function measure(g, s, size, font = 'sans', weight = 400) { g.font = V.font(size, font, weight); return g.measureText(String(s)).width; }
    /** personagem-token: quadrado arredondado com olhinhos */
    function tokChar(g, x, y, s, col, o = {}) {
      const C = V.C();
      V.box(g, x - s / 2, y - s / 2, s, s, { r: s * 0.3, fill: o.fill || V.alpha(col, 0.85), stroke: o.stroke || col, lw: 1.2 });
      if (s >= 11) {
        const eye = C.dark ? '#06140d' : '#ffffff';
        const ex = s * 0.17, ey = y - s * 0.08, er = Math.max(1, s * 0.07);
        V.dot(g, x - ex, ey, er, eye); V.dot(g, x + ex, ey, er, eye);
        g.beginPath();
        if (o.mood === 'sad') g.arc(x, y + s * 0.3, s * 0.14, Math.PI * 1.15, Math.PI * 1.85);
        else g.arc(x, y + s * 0.06, s * 0.15, Math.PI * 0.2, Math.PI * 0.8);
        g.strokeStyle = eye; g.lineWidth = Math.max(1, s * 0.06); g.stroke();
      }
    }
    function chip(g, x, y, w, h, label, o = {}) {
      const C = V.C();
      V.box(g, x, y, w, h, { r: o.r == null ? 6 : o.r, fill: o.fill || V.alpha(C.muted, 0.12), stroke: o.stroke || V.alpha(C.muted, 0.4), lw: o.lw || 1.2 });
      if (label != null) V.text(g, label, x + w / 2, y + h / 2 + 0.5, { size: o.size || 11.5, font: o.font || 'mono', weight: o.weight || 500, color: o.color || C.ink, align: 'center', base: 'middle', maxW: w - 4 });
    }
    function check(g, x, y, ok, s = 7) {
      const C = V.C();
      g.lineWidth = 2; g.strokeStyle = ok ? C.a : RED(); g.beginPath();
      if (ok) { g.moveTo(x - s, y); g.lineTo(x - s * 0.3, y + s * 0.7); g.lineTo(x + s, y - s * 0.8); }
      else { g.moveTo(x - s * 0.8, y - s * 0.8); g.lineTo(x + s * 0.8, y + s * 0.8); g.moveTo(x + s * 0.8, y - s * 0.8); g.lineTo(x - s * 0.8, y + s * 0.8); }
      g.stroke();
    }

    /* perfis de aceitação por posição (incondicionais) medidos nas receitas */
    const PROF = {
      glmS: { label: 'GLM · structured', a: [0.98, 0.98, 0.94, 0.94, 0.91, 0.83, 0.83] },
      glmP: { label: 'GLM · prose', a: [0.75, 0.58, 0.41, 0.28, 0.16, 0.09, 0.06] },
      ds: { label: 'DeepSeek · Vision-Exp', a: [0.89, 0.73, 0.49, 0.34, 0.23, 0.15] },
    };
    const acceptedFromU = (a, k, u) => { let j = 0; for (let i = 0; i < Math.min(k, a.length); i++) if (a[i] > u) j++; else break; return j; };

    /* ============================================== FIG 1 — esteiras */
    (function () {
      const fig = root.querySelector('#fig-esteira');
      adaptH(fig, 420, 400);
      const S = { k: 7, prof: 'glmP', d: 0.08 };
      const ctl = UI.controls(fig);
      const us = (() => { const r = V.rng(37); return Array.from({ length: 400 }, () => r()); })();
      const effK = () => Math.min(S.k, PROF[S.prof].a.length);
      const tau = () => 1 + PROF[S.prof].a.slice(0, effK()).reduce((s, x) => s + x, 0);
      const Tv = (k) => 1 + 0.04 * k;
      const lanes = [
        { name: 'no speculation', kind: 'none', T: () => 1 },
        { name: 'sequential draft (MTP/EAGLE)', kind: 'seq', T: () => effK() * S.d + Tv(effK()) },
        { name: 'block draft (DFlash/DSpark)', kind: 'blk', T: () => S.d * (1 + 0.05 * effK()) + Tv(effK()) },
      ];
      const rT = UI.readout(ctl, 'τ (tokens/step)', '—');
      const rS = UI.readout(ctl, 'seq. speedup', '—');
      const rB = UI.readout(ctl, 'block speedup', '—');
      const upd = () => { const t = tau(); rT.set(fmt(t, 2)); rS.set(fmt(t / lanes[1].T(), 2) + '×'); rB.set(fmt(t / lanes[2].T(), 2) + '×'); };
      UI.slider(ctl, { label: 'k (drafts)', min: 1, max: 7, step: 1, value: S.k, onInput: (v) => { S.k = v; upd(); } });
      UI.seg(ctl, { label: 'Profile', options: [{ v: 'glmS', label: 'GLM struct.' }, { v: 'glmP', label: 'GLM prose' }, { v: 'ds', label: 'DS Vision-Exp' }], value: S.prof, onChange: (v) => { S.prof = v; upd(); } });
      UI.slider(ctl, { label: 'draft cost', min: 0.02, max: 0.3, step: 0.01, value: S.d, fmt: (v) => Math.round(v * 100) + '%', onInput: (v) => { S.d = v; upd(); } });
      upd();
      const U = 1.05, RACE = 15;
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const a = PROF[S.prof].a, k = effK();
        const tr = (t + 7) % RACE;
        const narrow = w < 600;
        V.text(g, narrow ? 'same acceptance; only drafting changes' : 'same target, same acceptance — only the drafting time changes', 16, 20, { size: 12, color: C.muted, maxW: w - 150 });
        V.text(g, `⏱ ${fmt(tr, 1)} s of ${RACE} s`, w - 16, 20, { size: 11.5, font: 'mono', color: C.muted, align: 'right' });
        if (S.prof === 'ds' && S.k > 6) V.text(g, 'DS: acceptance measured only up to position 6', 16, 36, { size: 11, color: C.d });
        const top = 46, laneH = (h - top - 8) / 3;
        const Tmax = Math.max(...lanes.map((L) => L.T()));
        const x0 = 16, barMax = w - 32;
        let counts = [];
        lanes.forEach((L, li) => {
          const y = top + li * laneH;
          const T = L.T(), dur = T * U;
          const nDone = Math.floor(tr / dur), ph = (tr % dur) / dur;
          // tokens emitidos
          const cols = [];
          for (let s = 0; s < nDone; s++) {
            if (L.kind === 'none') cols.push(C.muted);
            else { const j = acceptedFromU(a, k, us[s % us.length]); for (let q = 0; q < j; q++) cols.push(C.a); cols.push(C.d); }
          }
          counts.push(cols.length);
          // cabeçalho
          V.text(g, L.name, x0, y + 12, { size: 12.5, weight: 600, color: C.ink, maxW: w * 0.62 });
          V.text(g, `${cols.length} tokens`, w - 16, y + 12, { size: 13, font: 'mono', weight: 700, align: 'right', color: li === 2 ? C.a : li === 1 ? C.b : C.muted });
          // barra do passo
          const bw = barMax * (T / Tmax), by = y + 30, bh = 15;
          V.box(g, x0, by, bw, bh, { r: 4, fill: V.alpha(C.muted, 0.08), stroke: V.alpha(C.muted, 0.25) });
          const segs = [];
          if (L.kind === 'none') segs.push({ u: 1, c: C.muted, l: 'decode' });
          else if (L.kind === 'seq') { for (let q = 0; q < k; q++) segs.push({ u: S.d, c: C.b, l: '' }); segs.push({ u: Tv(k), c: C.a, l: 'verify ' + (k + 1) }); }
          else { segs.push({ u: S.d * (1 + 0.05 * k), c: C.b, l: '' }); segs.push({ u: Tv(k), c: C.a, l: 'verify ' + (k + 1) }); }
          let cx = x0; const draftEnd = [];
          segs.forEach((sg, si) => {
            const sw = (sg.u / T) * bw;
            V.box(g, cx + 0.5, by + 0.5, Math.max(1, sw - 1), bh - 1, { r: 3, fill: V.alpha(sg.c, sg.c === C.a ? 0.45 : sg.c === C.muted ? 0.35 : 0.6) });
            if (sg.l && sw > 60) V.text(g, sg.l, cx + sw / 2, by + bh / 2 + 0.5, { size: 10.5, font: 'mono', align: 'center', base: 'middle', color: C.ink });
            if (sg.c === C.b && L.kind !== 'none') draftEnd.push([cx, cx + sw]);
            cx += sw;
          });
          // rascunhos acima da barra
          if (L.kind !== 'none') {
            const j = acceptedFromU(a, k, us[nDone % us.length]);
            const vStart = x0 + (1 - Tv(k) / T) * bw;
            const inVerify = ph * bw + x0 > vStart;
            const vProg = inVerify ? (ph * bw + x0 - vStart) / (x0 + bw - vStart) : 0;
            for (let q = 0; q < k; q++) {
              let qx;
              if (L.kind === 'seq') { const [s0, s1] = draftEnd[q]; qx = (s0 + s1) / 2; if (x0 + ph * bw < s0) continue; }
              else { const [s0, s1] = draftEnd[0]; if (x0 + ph * bw < s0 + (s1 - s0) * 0.5) continue; qx = s0 + ((q + 0.5) / k) * Math.max(13 * k, s1 - s0); }
              let col = C.b, mood;
              if (inVerify && vProg > (q + 0.5) / (k + 1)) { if (q < j) { col = C.a; } else { col = RED(); mood = 'sad'; } }
              tokChar(g, qx, by - 9, 11, col, { mood });
            }
          }
          // playhead
          const px = x0 + ph * bw;
          V.line(g, px, by - 3, px, by + bh + 3, { color: C.ink, lw: 1.5 });
          V.glow(g, px, by + bh / 2, 9, C.ink, 0.35);
          // bandeja
          const ty = by + bh + 16, cell = narrow ? 11 : 13, cap = Math.max(4, Math.floor((w - 32) / (cell + 3)));
          const rows = Math.max(1, Math.min(2, Math.floor((laneH - 64) / (cell + 3))));
          const show = cols.slice(Math.max(0, cols.length - cap * rows));
          show.forEach((c, qi) => {
            const r = Math.floor(qi / cap), cI = qi % cap;
            const tx = x0 + cell / 2 + cI * (cell + 3), tyy = ty + cell / 2 + r * (cell + 3);
            tokChar(g, tx, tyy, cell, c);
          });
          if (cols.length > cap * rows) V.text(g, `+${cols.length - cap * rows}`, w - 16, ty + cell, { size: 10.5, font: 'mono', color: C.muted, align: 'right' });
        });
        // legenda
        const ly = h - 6;
        const items = [[C.muted, narrow ? 'normal' : 'plain decode'], [C.b, 'draft'], [C.a, 'accepted'], [C.d, narrow ? 'bonus' : 'target bonus'], [RED(), 'rejected']];
        const iw = items.map(([, l]) => measure(g, l, 10.5) + 12);
        const oneRow = iw.reduce((a, b) => a + b + 14, 0) <= w - 32;
        let lx = 16, row = 0;
        items.forEach(([c, l], i) => {
          if (!oneRow && i === 3) { row = 1; lx = 16; }
          const yy = oneRow ? ly : ly - 15 + row * 15;
          V.dot(g, lx + 4, yy - 4, 4, c); V.text(g, l, lx + 12, yy, { size: 10.5, color: C.muted });
          lx += iw[i] + (oneRow && !narrow ? Math.max(14, 120 - iw[i]) : 14);
        });
      });
    })();

    /* ============================================== FIG 2 — máscara */
    (function () {
      const fig = root.querySelector('#fig-mascara');
      adaptH(fig, 400, 500);
      const S = { causal: false };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Mask in the block', options: [{ v: false, label: 'bidirectional (FLASH_ATTN)' }, { v: true, label: 'causal (TRITON_ATTN)' }], value: false, onChange: (v) => { S.causal = v; } });
      const rowL = ['a', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7'];
      const NC = 4;
      const H = V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const narrow = w < 600;
        const mW = narrow ? w - 20 : w * 0.62, mH = narrow ? h - 170 : h - 70;
        const labW = 34, cols = NC + 8;
        const gap = 3;
        const cw = Math.min(30, (mW - labW - 10) / cols - gap), chh = Math.min(28, (mH - 60) / 8 - gap);
        const ox = 10 + labW, oy = 52;
        // rótulos de grupo
        const ctxW = NC * (cw + gap) - gap, blkX = ox + NC * (cw + gap) + 6;
        V.text(g, 'injected window', ox + ctxW / 2, 18, { size: 11, color: C.b, align: 'center', maxW: ctxW + 10 });
        V.text(g, 'block keys', blkX + (8 * (cw + gap)) / 2, 18, { size: 11, color: C.a, align: 'center' });
        V.line(g, ox, 24, ox + ctxW, 24, { color: C.b, lw: 2 });
        V.line(g, blkX, 24, blkX + 8 * (cw + gap) - gap, 24, { color: C.a, lw: 2 });
        // linha ativa
        let act = Math.floor(t / 1.3) % 8;
        const p = H.pointer;
        if (p.inside && p.y > oy && p.y < oy + 8 * (chh + gap)) act = V.clamp(Math.floor((p.y - oy) / (chh + gap)), 0, 7);
        const colX = (j) => (j < NC ? ox + j * (cw + gap) : blkX + (j - NC) * (cw + gap));
        for (let j = 0; j < cols; j++) V.text(g, j < NC ? 'ctx' : rowL[j - NC], colX(j) + cw / 2, oy - 8, { size: 10.5, font: 'mono', align: 'center', color: C.muted });
        for (let i = 0; i < 8; i++) {
          const y = oy + i * (chh + gap);
          V.text(g, rowL[i], ox - 8, y + chh / 2 + 1, { size: 11, font: 'mono', align: 'right', base: 'middle', color: i === act ? C.ink : C.muted, weight: i === act ? 700 : 400 });
          for (let j = 0; j < cols; j++) {
            const x = colX(j);
            const isCtx = j < NC, jb = j - NC;
            const vis = isCtx || !S.causal || jb <= i;
            const lost = !isCtx && S.causal && jb > i;
            let fill = vis ? V.alpha(isCtx ? C.b : C.a, i === act ? 0.85 : 0.42) : V.alpha(C.muted, 0.07);
            V.box(g, x, y, cw, chh, { r: 4, fill });
            if (lost) { g.save(); g.strokeStyle = V.alpha(C.c, 0.7); g.setLineDash([3, 3]); g.lineWidth = 1.2; V.rr(g, x + 1, y + 1, cw - 2, chh - 2, 4); g.stroke(); g.restore(); }
          }
          if (i === act) {
            V.box(g, ox - 3, y - 2, blkX + 8 * (cw + gap) - ox + 2, chh + 4, { r: 6, stroke: C.ink, lw: 1.4 });
            // pulsos percorrendo o que a linha vê
            for (let q = 0; q < 3; q++) {
              const u = ((t * 0.5 + q / 3) % 1);
              const jj = Math.floor(u * cols);
              const isCtx = jj < NC, vis = isCtx || !S.causal || jj - NC <= i;
              if (vis) V.glow(g, colX(jj) + cw / 2, y + chh / 2, 10, C.ink, 0.5);
            }
          }
        }
        // texto da linha ativa
        const seen = S.causal ? (act === 0 ? 'ctx + a' : `ctx + a, ${rowL.slice(1, act + 1).join(', ')}`) : 'ctx + all 8 block positions';
        const ty = oy + 8 * (chh + gap) + 18;
        V.text(g, `${rowL[act]} sees: ${seen}`, 10, ty, { size: 12, color: C.ink, maxW: mW });
        if (S.causal) V.text(g, 'pink dashes = what training saw and the causal mask hides', 10, ty + 18, { size: 11, color: C.c, maxW: mW });
        // barras medidas
        const bx = narrow ? 20 : w * 0.66, bwid = narrow ? w - 40 : w * 0.32, by0 = narrow ? ty + 40 : 44;
        const groups = [
          { t: 'mean acceptance', v: [0.959, 0.31], max: 1, f: (x) => fmt(x, 2) },
          { t: 'tok/s (structured)', v: [65.1, 29], max: 70, f: (x) => fmt(x, 1).replace('.0', '') },
        ];
        const gh = narrow ? 58 : 110;
        groups.forEach((G, gi) => {
          const yy = by0 + gi * (gh + (narrow ? 12 : 30));
          V.text(g, G.t, bx, yy, { size: 11.5, color: C.muted, font: 'mono' });
          ['bidirectional', 'causal'].forEach((nm, k) => {
            const on = (k === 1) === S.causal;
            const rowH = narrow ? 16 : 30, yb = yy + 10 + k * (rowH + 6);
            const lw = narrow ? 80 : 86;
            V.text(g, nm, bx, yb + rowH / 2 + 1, { size: 11, base: 'middle', color: on ? C.ink : C.muted, weight: on ? 600 : 400 });
            const full = bwid - lw - 44;
            const vw = full * (G.v[k] / G.max);
            V.box(g, bx + lw, yb, full, rowH, { r: 4, fill: V.alpha(C.muted, 0.08) });
            V.box(g, bx + lw, yb, Math.max(2, vw), rowH, { r: 4, fill: V.alpha(k === 0 ? C.a : RED(), on ? 0.85 : 0.3) });
            V.text(g, G.f(G.v[k]), bx + lw + vw + 6, yb + rowH / 2 + 1, { size: 11.5, font: 'mono', base: 'middle', color: on ? C.ink : C.muted });
          });
        });
        if (!narrow) V.text(g, 'measured on GLM (lab C4, 2026-08-30)', bx, by0 + 2 * (gh + 30) - 6, { size: 10.5, color: C.muted });
      });
    })();

    /* ============================================== FIG 3 — pipeline DFlash2 */
    (function () {
      const fig = root.querySelector('#fig-pipeline');
      const steps = [
        { t: 'The target verifies and exposes five taps', d: 'GLM runs the verification step. At the outputs of layers 5, 14, 24, 33 and 42 (log: taps 6, 15, 25, 34, 43), <code>patch_glm_eagle3.py</code> materializes the <code>hc_post</code> and contracts the 4 mHC streams: <code>[T, 4, 4096] → [T, 4096]</code>, five times.' },
        { t: 'Fuse the features', d: 'The 5 features are concatenated into <code>[T, 20480]</code> and go through <code>fc</code> \\(\\in\\mathbb{R}^{4096\\times20480}\\) and <code>hidden_norm</code>. Out comes an \\(h_p\\) of 4096 per verified position.' },
        { t: 'Inject KV into the 5 layers', d: 'A fused GEMM projects \\(h_p\\) into K and V for the drafter’s 5 layers (K gets K-norm and RoPE), straight into the 2048-token SWA cache. On TP2 that’s 2048 B per token per layer, in BF16.' },
        { t: 'Build the block', d: 'Query = <code>[a, M, M, M, M, M, M, M]</code>: the anchor (last accepted token, the bonus) and 7 masks with id 154856, embedded through the <b>target’s</b> table (the drafter has no embedding of its own).' },
        { t: 'Five layers, in parallel', d: 'Each layer: norm → conv → attention (Q from the block; K/V = window + block, <b>bidirectional</b> within the block) → conv → norm → conv → SwiGLU MLP 12288 → conv. All 8 positions move together.' },
        { t: 'Top-16 per position', d: 'The target’s <code>lm_head</code> (154,880 rows) produces logits for the 7 mask positions; <code>torch.topk</code> keeps 16 candidates and their unary logits.' },
        { t: 'Selector and walk', d: 'The bilinear selector (rank 256) scores transitions between neighboring candidates, and <code>_selector_walk_kernel</code> walks the trellis (argmax at T=0, Gumbel-max at T&gt;0). Out come 7 drafts, and the realized scores become the draft’s \\(q(x)\\).' },
        { t: 'Verification', d: 'The target runs <code>[a, d₁…d₇]</code>: 8 rows in one forward, with rejection sampling position by position. It accepts the longest prefix and emits the bonus: 1 to 8 tokens per step. The features of the accepted positions feed the next cycle.' },
      ];
      const st = UI.stepper(fig, steps, { autoplay: 6500 });
      const nodes = [
        { t: 'Target · 45 layers', k: 'Target', s: '5 taps → [T,4096]' },
        { t: 'concat → fc', k: 'fc', s: '[T,20480] → [T,4096]' },
        { t: 'Injected KV', k: 'KV', s: 'SWA 2048 × 5 layers' },
        { t: 'Query block', k: 'Block', s: '[a, 154856 ×7]' },
        { t: '5 DFlash2 layers', k: '5 layers', s: 'conv · attn · mlp' },
        { t: 'lm_head → top-16', k: 'top-16', s: '154880 → 16 / pos.' },
        { t: 'Selector + walk', k: 'Selector', s: 'rank 256 · Gumbel' },
        { t: 'Verification', k: 'Verify', s: '8 rows → 1..8 tok' },
      ];
      const rr = V.rng(5);
      const lat = Array.from({ length: 7 }, () => Array.from({ length: 16 }, (_, i) => Math.max(0.05, 1 - i * 0.07 + rr() * 0.15)));
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const cols = 4, gapX = w < 600 ? 10 : 22, gapY = 34, mx = 12, my = 14;
        const nw = (w - 2 * mx - (cols - 1) * gapX) / cols, nh = (h - 2 * my - gapY) / 2;
        const pos = (i) => { const r = i < 4 ? 0 : 1; const c = r === 0 ? i : 7 - i; return [mx + c * (nw + gapX), my + r * (nh + gapY)]; };
        const cur = st.i, el = st.elapsed();
        // setas entre nós
        for (let i = 0; i < 7; i++) {
          const [ax, ay] = pos(i), [bx, by] = pos(i + 1);
          let x1, y1, x2, y2;
          if (i === 3) { x1 = ax + nw / 2; y1 = ay + nh; x2 = bx + nw / 2; y2 = by; }
          else if (i < 3) { x1 = ax + nw; y1 = ay + nh / 2; x2 = bx; y2 = by + nh / 2; }
          else { x1 = ax; y1 = ay + nh / 2; x2 = bx + nw; y2 = by + nh / 2; }
          const on = cur === i + 1;
          V.arrow(g, x1, y1, x2, y2, { color: on ? C.a : V.alpha(C.muted, 0.45), lw: on ? 2.4 : 1.4, head: 7 });
          if (on) for (let q = 0; q < 3; q++) { const u = ((el * 0.8 + q / 3) % 1); const [px, py] = V.along(x1, y1, x2, y2, u); V.glow(g, px, py, 7, C.a, 0.9); }
        }
        // retorno (ciclo)
        { const [ax, ay] = pos(7), [bx, by] = pos(0); if (cur === 0) { V.line(g, ax + 6, ay + 10, bx + 6, by + nh - 10, { color: V.alpha(C.d, 0.6), lw: 1.4, dash: [4, 5], dashOffset: -t * 20 }); } }
        nodes.forEach((N, i) => {
          const [x, y] = pos(i);
          const on = i === cur;
          V.box(g, x, y, nw, nh, { r: 12, fill: on ? V.alpha(C.a, 0.12) : V.alpha(C.panel || C.muted, on ? 0.2 : 0.06), stroke: on ? C.a : V.alpha(C.muted, 0.3), lw: on ? 2 : 1 });
          if (on) V.glow(g, x + nw - 10, y + 10, 14, C.a, 0.6);
          g.save(); g.globalAlpha = on ? 1 : 0.55;
          const tiny = nw < 120;
          V.text(g, (i + 1) + '. ' + (tiny ? N.k : N.t), x + 8, y + 17, { size: tiny ? 11 : 12, weight: 600, color: C.ink, maxW: nw - 12 });
          if (tiny) { const parts = N.s.split(' '); const half = Math.ceil(parts.length / 2); [parts.slice(0, half).join(' '), parts.slice(half).join(' ')].forEach((ln, li) => V.text(g, ln, x + 6, y + nh - 22 + li * 12, { size: 9.5, font: 'mono', color: C.muted, maxW: nw - 10 })); }
          else V.text(g, N.s, x + 8, y + nh - 9, { size: 10.5, font: 'mono', color: C.muted, maxW: nw - 14 });
          const ix = x + 8, iy = y + 28, iw = nw - 16, ih = nh - (nw < 120 ? 62 : 50);
          drawIcon(g, i, ix, iy, iw, ih, t, on, C);
          g.restore();
        });
      });
      function drawIcon(g, i, x, y, w, h, t, on, C) {
        const cx = x + w / 2, cy = y + h / 2;
        if (i === 0) { // pilha de 45 camadas com 5 taps
          const taps = [5, 14, 24, 33, 42], n = 45, lh = h / n;
          const bw = Math.min(w * 0.45, 60);
          for (let L = 0; L < n; L++) {
            const tap = taps.includes(L);
            g.fillStyle = tap ? C.d : V.alpha(C.b, 0.35);
            g.fillRect(cx - bw / 2, y + h - (L + 1) * lh, bw, Math.max(1, lh - 0.6));
            if (tap) { V.arrow(g, cx + bw / 2 + 2, y + h - (L + 0.5) * lh, cx + bw / 2 + 16 + (on ? 3 * Math.sin(t * 4 + L) : 0), y + h - (L + 0.5) * lh, { color: C.d, lw: 1.2, head: 4 }); if (w > 120) V.text(g, String(L + 1), cx + bw / 2 + 20, y + h - (L + 0.5) * lh + 3.5, { size: 9.5, font: 'mono', color: C.d }); }
          }
        } else if (i === 1) {
          const cols5 = [C.a, C.b, C.c, C.d, C.e];
          const vh = Math.min(h - 6, 90), vw = Math.max(4, Math.min(8, w * 0.05));
          cols5.forEach((c, k) => { g.fillStyle = V.alpha(c, 0.75); g.fillRect(x + 2 + k * (vw + 2), cy - vh / 2, vw, vh); });
          const mx0 = x + 5 * (vw + 2) + 8, mwid = Math.max(20, w * 0.36);
          V.box(g, mx0, cy - vh / 2, mwid, vh * 0.55, { r: 3, fill: V.alpha(C.b, 0.25), stroke: C.b, lw: 1 });
          if (w > 110) V.text(g, 'W_fc', mx0 + mwid / 2, cy - vh / 2 + vh * 0.28 + 3, { size: 10, font: 'mono', align: 'center', color: C.ink });
          V.arrow(g, mx0 + mwid + 2, cy - vh * 0.2, mx0 + mwid + 14, cy - vh * 0.2, { color: C.muted, lw: 1.2, head: 4 });
          g.fillStyle = V.alpha(C.a, 0.8); g.fillRect(Math.min(x + w - vw, mx0 + mwid + 16), cy - vh / 2, vw, vh * 0.55);
        } else if (i === 2) {
          const rows = 5, rh = Math.min(14, (h - 8) / rows - 3);
          for (let r = 0; r < rows; r++) {
            const yy = y + 4 + r * (rh + 3);
            const n = Math.max(3, Math.floor((w - 20) / 9));
            for (let q = 0; q < n; q++) {
              const filled = on ? q < ((t * 3 + r) % (n + 4)) : q < n - 2;
              g.fillStyle = filled ? V.alpha(q % 2 ? C.a : C.b, 0.7) : V.alpha(C.muted, 0.12);
              g.fillRect(x + 16 + q * 9, yy, 7, rh);
            }
            V.text(g, 'ℓ' + (r + 1), x, yy + rh - 2, { size: 9.5, font: 'mono', color: C.muted });
          }
        } else if (i === 3) {
          const two = w < 110, per = two ? 4 : 8;
          const s = Math.min(18, (w - 8) / per - 3);
          for (let q = 0; q < 8; q++) { const bob = on ? Math.sin(t * 5 + q) * 2 : 0; const rr2 = Math.floor(q / per), cc = q % per; tokChar(g, x + s / 2 + cc * (s + 3), cy + bob + (two ? (rr2 - 0.5) * (s + 4) : 0), s, q === 0 ? C.d : C.muted); }
          if (w > 120) { V.text(g, 'anchor', x + s / 2, cy + s, { size: 9.5, color: C.d, align: 'left' }); }
        } else if (i === 4) {
          const n = 5, sh = Math.min(16, (h - 6) / n - 4);
          const segs = [[C.c, 0.1], [C.a, 0.3], [C.c, 0.1], [C.b, 0.4], [C.c, 0.1]];
          for (let L = 0; L < n; L++) {
            const yy = y + 3 + L * (sh + 4), off = (n - 1 - L) * 3;
            let xx = x + off; const ww = w - 14;
            segs.forEach(([c, f], si) => { const sw = ww * f; g.fillStyle = V.alpha(c, on && Math.floor(t * 3) % 5 === si ? 0.95 : 0.5); g.fillRect(xx, yy, sw - 1, sh); xx += sw; });
          }
        } else if (i === 5) {
          const nC = 7, cw = (w - 4) / nC;
          for (let c = 0; c < nC; c++) for (let r = 0; r < 16; r++) {
            const v = lat[c][r];
            const cellH = (h - 4) / 16;
            g.fillStyle = V.cmap(v * (on ? (0.7 + 0.3 * Math.sin(t * 3 + c)) : 0.8), 'seq');
            g.fillRect(x + 2 + c * cw, y + 2 + r * cellH, cw - 2, cellH - 1);
          }
        } else if (i === 6) {
          const nC = 7, nR = 4, cw = (w - 8) / nC, rh = (h - 8) / nR;
          const path = [0, 0, 1, 0, 0, 2, 0];
          for (let c = 0; c < nC; c++) for (let r = 0; r < nR; r++) V.dot(g, x + 4 + (c + 0.5) * cw, y + 4 + (r + 0.5) * rh, 2.2, V.alpha(C.muted, 0.5));
          const prog = on ? (t * 1.5) % (nC + 1) : nC;
          g.strokeStyle = C.a; g.lineWidth = 2; g.beginPath();
          for (let c = 0; c < Math.min(nC, Math.floor(prog) + 1); c++) { const px = x + 4 + (c + 0.5) * cw, py = y + 4 + (path[c] + 0.5) * rh; c ? g.lineTo(px, py) : g.moveTo(px, py); }
          g.stroke();
          for (let c = 0; c < Math.min(nC, Math.floor(prog) + 1); c++) V.dot(g, x + 4 + (c + 0.5) * cw, y + 4 + (path[c] + 0.5) * rh, 3.5, C.a);
        } else if (i === 7) {
          const two = w < 110, per = two ? 4 : 8;
          const s = Math.min(16, (w - 8) / per - 3);
          const j = 5;
          for (let q = 0; q < 8; q++) {
            const rr2 = Math.floor(q / per), cc = q % per;
            const xx = x + s / 2 + cc * (s + 3);
            const revealed = !on || (t * 3) % 10 > q;
            const col = q === 0 ? C.d : q <= j ? C.a : RED();
            tokChar(g, xx, cy - 4 + (two ? (rr2 - 0.5) * (s + 4) : 0), s, revealed ? col : C.muted, { mood: revealed && q > j ? 'sad' : undefined });
          }
          if (w > 120) { V.text(g, `anchor + ${j} accepted`, x, cy + s + 6, { size: 10, color: C.muted, maxW: w }); V.text(g, `emits ${j} + 1 bonus`, x, cy + s + 19, { size: 10, color: C.d, maxW: w }); }
        }
      }
    })();

    /* ============================================== FIG 4 — conv dinâmica */
    (function () {
      const fig = root.querySelector('#fig-conv');
      adaptH(fig, 390, 460);
      const B = 8, G = 6;
      const r = V.rng(11);
      const hIn = Array.from({ length: B }, () => Array.from({ length: G }, () => r.normal() * 0.9));
      const b0 = Array.from({ length: G }, () => 0.85 + r.normal() * 0.08), b1 = Array.from({ length: G }, () => 0.35 + r.normal() * 0.08);
      const P0 = Array.from({ length: G }, () => Array.from({ length: G }, () => r.normal() * 0.5));
      const P1 = Array.from({ length: G }, () => Array.from({ length: G }, () => r.normal() * 0.5));
      const S = { dyn: true, str: 0.8, hi: -1 };
      const ctl = UI.controls(fig);
      UI.toggle(ctl, { label: 'dynamic part δ(hₜ)', value: true, onChange: (v) => { S.dyn = v; } });
      UI.slider(ctl, { label: 'δ strength', min: 0, max: 1.5, step: 0.05, value: S.str, fmt: (v) => fmt(v, 2), onInput: (v) => { S.str = v; } });
      UI.slider(ctl, { label: 'highlighted position', min: -1, max: 7, step: 1, value: -1, fmt: (v) => (v < 0 ? 'auto' : v === 0 ? 'a' : 'M' + v), onInput: (v) => { S.hi = v; } });
      const coef = (t, side) => Array.from({ length: G }, (_, gi) => {
        const base = side === 0 ? b0[gi] : b1[gi];
        if (!S.dyn) return base;
        const P = side === 0 ? P0 : P1;
        let z = 0; for (let q = 0; q < G; q++) z += P[gi][q] * hIn[t][q];
        return base + S.str * 0.45 * Math.tanh(z);
      });
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const narrow = w < 600;
        const labW = narrow ? 44 : 70;
        const coefW = narrow ? 0 : 150;
        const areaW = w - labW - coefW - 20;
        const colW = areaW / B, cw = Math.min(26, colW * 0.55);
        const topPad = 34, cellH = Math.min(15, (h - (narrow ? 200 : 120)) / (2 * G + 4));
        const yIn = topPad, yOut = yIn + G * (cellH + 2) + (narrow ? 60 : 80);
        const hi = S.hi >= 0 ? S.hi : Math.floor(t / 1.6) % B;
        const cx = (tt) => 10 + labW + tt * colW + colW / 2;
        // calcular saídas
        const out = [];
        let mx = 0.01;
        for (let tt = 0; tt < B; tt++) {
          const c0 = coef(tt, 0), c1 = coef(tt, 1);
          out.push({ c0, c1, v: Array.from({ length: G }, (_, gi) => c0[gi] * hIn[tt][gi] + (tt % B >= 1 ? c1[gi] * hIn[tt - 1][gi] : 0)) });
        }
        out.forEach((o) => o.v.forEach((x) => { mx = Math.max(mx, Math.abs(x)); }));
        V.text(g, 'input hₜ', 10, yIn + (G * (cellH + 2)) / 2, { size: 11, color: C.muted, base: 'middle', maxW: labW - 4 });
        V.text(g, 'output oₜ', 10, yOut + (G * (cellH + 2)) / 2, { size: 11, color: C.muted, base: 'middle', maxW: labW - 4 });
        for (let tt = 0; tt < B; tt++) {
          const x = cx(tt) - cw / 2;
          V.text(g, tt === 0 ? 'a' : 'M' + tt, cx(tt), yIn - 10, { size: 11, font: 'mono', align: 'center', color: tt === hi ? C.ink : C.muted, weight: tt === hi ? 700 : 400 });
          for (let gi = 0; gi < G; gi++) {
            g.fillStyle = V.cmap(hIn[tt][gi] / 2, 'div'); V.rr(g, x, yIn + gi * (cellH + 2), cw, cellH, 3); g.fill();
            g.fillStyle = V.cmap(out[tt].v[gi] / mx, 'div'); V.rr(g, x, yOut + gi * (cellH + 2), cw, cellH, 3); g.fill();
          }
          if (tt === hi) { V.box(g, x - 3, yIn - 3, cw + 6, G * (cellH + 2) + 4, { r: 5, stroke: C.c, lw: 1.6 }); V.box(g, x - 3, yOut - 3, cw + 6, G * (cellH + 2) + 4, { r: 5, stroke: C.c, lw: 1.6 }); }
          // setas
          const y1 = yIn + G * (cellH + 2) + 2, y2 = yOut - 6;
          const m0 = out[tt].c0.reduce((s, x) => s + Math.abs(x), 0) / G, m1 = out[tt].c1.reduce((s, x) => s + Math.abs(x), 0) / G;
          const alpha = tt === hi ? 1 : 0.45;
          V.arrow(g, cx(tt), y1, cx(tt), y2, { color: V.alpha(C.a, alpha), lw: 1 + m0 * 3.2, head: 6 });
          if (tt >= 1) {
            V.arrow(g, cx(tt - 1) + cw / 2, y1, cx(tt) - 3, y2, { color: V.alpha(C.b, alpha), lw: 0.8 + m1 * 5, head: 6 });
            if (tt === hi) for (let q = 0; q < 2; q++) { const u = ((t * 0.9 + q / 2) % 1); const [px, py] = V.along(cx(tt - 1) + cw / 2, y1, cx(tt) - 3, y2, u); V.glow(g, px, py, 7, C.b, 0.9); }
          } else {
            g.save(); g.setLineDash([3, 3]); V.line(g, cx(0) - colW * 0.45, y1 + 4, cx(0) - 6, y2 - 8, { color: V.alpha(RED(), 0.7), lw: 1.3 }); g.restore();
            V.text(g, '✕', cx(0) - colW * 0.3, (y1 + y2) / 2 + 4, { size: 12, color: RED(), align: 'center' });
          }
          if (tt === hi) { const u = (t * 0.9) % 1; V.glow(g, cx(tt), V.lerp(y1, y2, u), 7, C.a, 0.9); }
        }
        V.text(g, 'boundary: t mod B = 0', cx(0) - colW * 0.45, yOut + G * (cellH + 2) + 16, { size: 10.5, color: RED(), maxW: colW * 2 });
        // painel de coeficientes
        const o = out[hi];
        const px0 = narrow ? 10 + labW : w - coefW - 4, py0 = narrow ? yOut + G * (cellH + 2) + 30 : yIn;
        const pw = narrow ? w - labW - 24 : coefW - 8;
        V.text(g, `coef. of ${hi === 0 ? 'a' : 'M' + hi} per group`, px0, py0 + 4, { size: 11, color: C.ink, weight: 600, maxW: pw });
        const rowH = narrow ? 9 : 15;
        const labG = narrow ? 0 : 20, half = (pw - labG) / 2 - 6, bx0 = px0 + labG, bx1 = bx0 + half + 12;
        for (let gi = 0; gi < G; gi++) {
          const yy = py0 + 14 + gi * (rowH + 4);
          const w0 = half * Math.min(1, Math.abs(o.c0[gi]) / 1.4), w1 = hi >= 1 ? half * Math.min(1, Math.abs(o.c1[gi]) / 1.4) : 0;
          g.fillStyle = V.alpha(C.muted, 0.1); g.fillRect(bx0, yy, half, rowH); g.fillRect(bx1, yy, half, rowH);
          g.fillStyle = V.alpha(C.a, 0.8); g.fillRect(bx0, yy, w0, rowH);
          g.fillStyle = V.alpha(C.b, 0.8); g.fillRect(bx1, yy, Math.max(0, w1), rowH);
          if (!narrow) V.text(g, 'g' + gi, px0, yy + rowH - 3, { size: 9.5, font: 'mono', color: C.muted });
        }
        const ly = py0 + 14 + G * (rowH + 4) + 12;
        V.text(g, 'tap 0 (hₜ)', bx0, ly, { size: 10.5, color: C.a, maxW: half + 10 });
        V.text(g, 'tap 1 (hₜ₋₁)', bx1, ly, { size: 10.5, color: C.b, maxW: half + 10 });
        V.text(g, S.dyn ? 'b + δ(hₜ): varies per token' : 'base_kernel only: same for all', px0, ly + 16, { size: 10.5, color: C.muted, maxW: pw });
      });
    })();

    /* ============================================== FIG 5 — treliça do seletor */
    (function () {
      const fig = root.querySelector('#fig-trelica');
      adaptH(fig, 430, 500);
      const A = ['day', ',', 'how', 'is', 'it', 'going', '?'];
      const Bm = ['game', '!', 'See', 'you', 'next', 'time', '.'];
      const uA = [2.0, 1.6, 1.5, 1.25, 1.1, 0.85, 0.8], uB = [1.9, 1.7, 1.45, 1.3, 1.05, 0.9, 0.75];
      const dis = [['luck', 'job'], ['and', 'to'], ['what', 'the'], ['was', 'do'], ['so', 'we'], ['on', 'well'], ['!!', ';']];
      const cand = A.map((_, i) => [
        { w: A[i], u: uA[i], m: 'A' }, { w: Bm[i], u: uB[i], m: 'B' }, { w: dis[i][0], u: 0.45, m: null }, { w: dis[i][1], u: 0.25, m: null },
      ].sort((p, q) => q.u - p.u));
      const pair = (pm, cm) => (pm && cm ? (pm === cm ? 2.0 : -1.0) : 0);
      const S = { mode: 'walk', T: 0, seed: 1, t0: 0 };
      let noise = [];
      const mkNoise = () => { const r = V.rng(1000 + S.seed); noise = cand.map((c) => c.map(() => -Math.log(-Math.log(Math.max(1e-9, r()))))); };
      mkNoise();
      const solve = () => {
        const path = [], stepScores = [];
        let prev = null;
        for (let i = 0; i < 7; i++) {
          const sc = cand[i].map((c, ci) => {
            let s = c.u + (S.mode === 'walk' ? pair(prev, c.m) : 0);
            if (S.T > 0) s = s / S.T + noise[i][ci];
            return s;
          });
          let best = 0; sc.forEach((s, ci) => { if (s > sc[best]) best = ci; });
          path.push(best); stepScores.push(sc); prev = cand[i][best].m;
        }
        return { path, stepScores };
      };
      let sol = solve();
      /* painel bilinear: vetores ilustrativos de 16 células (representam 256) */
      const NV = 16, rb = V.rng(77);
      const PA = Array.from({ length: NV }, () => (rb() < 0.5 ? -1 : 1) * (0.45 + 0.55 * rb()));
      const vecP = { A: PA, B: PA.map((x) => -0.9 * x + (rb() - 0.5) * 0.2), null: PA.map(() => (rb() - 0.5) * 0.25) };
      const hv = Array.from({ length: NV }, (_, i) => (i % 5 === 3 ? 0.08 : 0.35 + 0.65 * rb()));
      const vecS = { A: PA.map((x) => 0.85 * x + (rb() - 0.5) * 0.25), B: PA.map((x) => -0.85 * x + (rb() - 0.5) * 0.25) };
      const noiseS = cand.map((col) => col.map(() => Array.from({ length: NV }, () => (rb() - 0.5) * 0.35)));
      const ctl = UI.controls(fig);
      const out = UI.readout(ctl, 'draft', '—');
      const re = () => { sol = solve(); S.t0 = performance.now() / 1000; out.set('Good ' + sol.path.map((ci, i) => cand[i][ci].w).join(' ')); };
      UI.seg(ctl, { label: 'Choice', options: [{ v: 'walk', label: 'selector (walk)' }, { v: 'indep', label: 'independent argmax' }], value: 'walk', onChange: (v) => { S.mode = v; re(); } });
      UI.seg(ctl, { label: 'Temperature', options: [{ v: 0, label: 'T = 0 (argmax)' }, { v: 0.8, label: 'T = 0.8 (Gumbel)' }], value: 0, onChange: (v) => { S.T = v; re(); } });
      UI.button(ctl, '🎲 new noise', () => { S.seed++; mkNoise(); re(); });
      re();
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const narrow = w < 600;
        const P = narrow ? 4 : 7;
        const colW = (w - 20) / (P + 1), chipW = Math.min(76, colW - 8), chipH = 24;
        const oy = 44;
        const want = narrow ? 206 : 140;
        const rowGap = V.clamp((h - want - oy - 16 - 4 * chipH) / 3, 12, 40);
        const rowY = (r) => oy + r * (chipH + rowGap);
        const colX = (c) => 10 + c * colW + (colW - chipW) / 2;
        const el = performance.now() / 1000 - S.t0;
        const period = 0.75, prog = Math.min(P, Math.floor(el / period) + 1);
        const loopT = (P + 4) * period;
        if (el > loopT) S.t0 = performance.now() / 1000;
        V.text(g, narrow ? 'positions 1–4 of 7 (4 of 16 candidates)' : '4 of 16 candidates per position; lines = transition scores', 10, 16, { size: 11.5, color: C.muted, maxW: w - 20 });
        // âncora
        chip(g, colX(0), rowY(1.5), chipW, chipH, 'Good', { fill: V.alpha(C.d, 0.3), stroke: C.d, weight: 700 });
        V.text(g, 'anchor', colX(0) + chipW / 2, rowY(1.5) + chipH + 14, { size: 10.5, color: C.d, align: 'center' });
        // arestas
        for (let i = 0; i < P; i++) {
          const fromList = i === 0 ? [{ x: colX(0) + chipW, y: rowY(1.5) + chipH / 2, m: null }] : cand[i - 1].map((c, r) => ({ x: colX(i) + chipW, y: rowY(r) + chipH / 2, m: c.m }));
          fromList.forEach((f) => cand[i].forEach((c, r) => {
            if (S.mode !== 'walk') return;
            const pv = pair(f.m, c.m);
            if (pv === 0) return;
            V.line(g, f.x, f.y, colX(i + 1), rowY(r) + chipH / 2, { color: V.alpha(pv > 0 ? C.a : C.c, pv > 0 ? 0.28 : 0.14), lw: pv > 0 ? 1.6 : 1 });
          }));
        }
        // caminho escolhido
        g.save();
        let px = colX(0) + chipW, py = rowY(1.5) + chipH / 2;
        for (let i = 0; i < Math.min(P, prog); i++) {
          const r = sol.path[i];
          const nx = colX(i + 1), ny = rowY(r) + chipH / 2;
          V.line(g, px, py, nx, ny, { color: C.a, lw: 3 });
          px = nx + chipW; py = ny;
        }
        g.restore();
        // chips candidatos
        for (let i = 0; i < P; i++) {
          V.text(g, 'pos ' + (i + 1), colX(i + 1) + chipW / 2, oy - 10, { size: 10.5, font: 'mono', align: 'center', color: C.muted });
          cand[i].forEach((c, r) => {
            const chosen = i < prog && sol.path[i] === r;
            const col = c.m === 'A' ? C.a : c.m === 'B' ? C.b : C.muted;
            chip(g, colX(i + 1), rowY(r), chipW, chipH, c.w, { fill: chosen ? V.alpha(col, 0.55) : V.alpha(col, 0.12), stroke: chosen ? C.ink : V.alpha(col, 0.6), lw: chosen ? 2 : 1, weight: chosen ? 700 : 500 });
            if (rowGap >= 22) V.text(g, 'u=' + fmt(c.u, 2), colX(i + 1) + chipW / 2, rowY(r) + chipH + 12, { size: 9.5, font: 'mono', align: 'center', color: C.muted });
          });
        }
        // inset ancorado logo abaixo da treliça
        const si = Math.min(P, prog) - 1;
        const trBot = rowY(3) + chipH + (rowGap >= 22 ? 16 : 6);
        const ix = 10, iw = w - 20, iy = trBot + 8, ih = h - iy - 4;
        V.box(g, ix, iy, iw, ih, { r: 10, fill: V.alpha(C.muted, 0.06), stroke: V.alpha(C.muted, 0.25) });
        const sc = sol.stepScores[si];
        const lab = S.mode === 'walk' ? `step ${si + 1}: s = u + transition${S.T > 0 ? ' → s/T + g (Gumbel)' : ''}` : `step ${si + 1}: only u${S.T > 0 ? '/T + g' : ''} (no transition)`;
        const barsW = narrow ? iw - 20 : iw * 0.42;
        V.text(g, lab, ix + 10, iy + 16, { size: 11, color: C.ink, maxW: barsW });
        const mn = Math.min(...sc), mxs = Math.max(...sc);
        const bw = barsW / 4 - 8;
        sc.forEach((sv, r) => {
          const bx = ix + 10 + r * (bw + 8), bh = 34 * ((sv - mn) / (mxs - mn || 1)) + 4;
          const chosen = sol.path[si] === r;
          const c = cand[si][r];
          const col = c.m === 'A' ? C.a : c.m === 'B' ? C.b : C.muted;
          g.fillStyle = V.alpha(col, chosen ? 0.9 : 0.35); V.rr(g, bx, iy + 62 - bh, bw, bh, 3); g.fill();
          V.text(g, `${c.w} ${fmt(sv, 1)}`, bx + bw / 2, iy + 76, { size: 10.5, font: 'mono', align: 'center', color: chosen ? C.ink : C.muted, maxW: bw + 6 });
        });
        // painel bilinear: de onde vem a transição
        const dx = narrow ? ix + 10 : ix + 20 + barsW, dy = narrow ? iy + 90 : iy + 8;
        const dw = narrow ? iw - 20 : iw - barsW - 30, dh = narrow ? ih - 94 : ih - 12;
        if (dh > 60) {
          const walk = S.mode === 'walk';
          const prevM = si === 0 ? null : cand[si - 1][sol.path[si - 1]].m;
          const prevW = si === 0 ? 'Good' : cand[si - 1][sol.path[si - 1]].w;
          const Pv = vecP[prevM === 'A' ? 'A' : prevM === 'B' ? 'B' : 'null'];
          const gv = Pv.map((x, k) => x * hv[k]);
          const rows = [
            { l: `P[“${prevW}”]`, v: Pv, n: 'row of P [154880×256]' },
            { l: 'W_h·h', v: hv.map((x) => x * 0.9), n: '[256×4096] · h' },
            { l: '= gate', v: gv, n: 'element-wise product' },
          ].concat(cand[si].map((c, r) => ({ l: `S[“${c.w}”]`, v: c.m ? vecS[c.m] : noiseS[si][r], val: pair(prevM, c.m), m: c.m, chosen: sol.path[si] === r })));
          V.text(g, narrow ? 'transition = ⟨P[c₋₁] ⊙ W_h h, S[c]⟩' : 'transition = ⟨ P[c₋₁] ⊙ W_h h , S[c] ⟩  (16 cells = 256 numbers)', dx, dy + 10, { size: 10.5, color: walk ? C.ink : C.muted, maxW: dw });
          const labW = narrow ? 64 : 76, valW = narrow ? 36 : 44;
          const noteW = narrow ? 0 : Math.max(0, Math.min(150, dw - labW - valW - 16 * 9 - 8));
          const cellW = Math.max(4, Math.min(11, (dw - labW - valW - noteW - 8) / NV - 1));
          const rh = Math.max(8, Math.min(12, (dh - 20) / rows.length - 3));
          const sweep = Math.floor(t * 7) % NV;
          g.save(); if (!walk) g.globalAlpha = 0.35;
          rows.forEach((R, ri) => {
            const yy = dy + 18 + ri * (rh + 3) + (ri >= 3 ? 4 : 0);
            const col = R.m === 'A' ? C.a : R.m === 'B' ? C.b : ri < 3 ? C.d : C.muted;
            V.text(g, R.l, dx + labW - 6, yy + rh - 1, { size: 9.5, font: 'mono', align: 'right', color: ri < 3 ? C.d : col, maxW: labW - 6, weight: R.chosen ? 700 : 400 });
            for (let k = 0; k < NV; k++) {
              g.fillStyle = V.cmap(V.clamp(R.v[k], -1, 1), 'div');
              g.fillRect(dx + labW + k * (cellW + 1), yy, cellW, rh);
            }
            if (walk) { g.strokeStyle = V.alpha(C.ink, 0.7); g.lineWidth = 1; g.strokeRect(dx + labW + sweep * (cellW + 1) - 0.5, yy - 0.5, cellW + 1, rh + 1); }
            const xEnd = dx + labW + NV * (cellW + 1) + 6;
            if (R.val != null) V.text(g, (R.val > 0 ? '+' : R.val < 0 ? '−' : '') + fmt(Math.abs(R.val), 1), xEnd, yy + rh - 1, { size: 10, font: 'mono', weight: 700, color: R.val > 0 ? C.a : R.val < 0 ? C.c : C.muted });
            else if (noteW > 40) V.text(g, R.n, xEnd, yy + rh - 1, { size: 9.5, color: C.muted, maxW: noteW + valW });
          });
          g.restore();
          if (!walk) V.text(g, 'independent argmax: no transition is added', dx, dy + dh - 2, { size: 10.5, color: C.c, maxW: dw });
        }
      });
    })();

    /* ============================================== FIG 6 — Markov */
    (function () {
      const fig = root.querySelector('#fig-markov');
      adaptH(fig, 480, 560);
      const outCol = (() => { const r = V.rng(21); return Array.from({ length: 26 }, (_, i) => (i === 4 ? 0.95 : i === 11 ? -0.9 : (r() - 0.5) * 0.5)); })();
      const rowFrac = (wd) => { let hh = 7; for (const ch of wd) hh = (hh * 31 + ch.charCodeAt(0)) % 997; return 0.12 + 0.76 * (hh / 997); };
      const cands = [
        [['New', 0.50], ['Salt', 0.47], ['Las', 0.03]],
        [['York', 0.47], ['Lake', 0.49], ['Vegas', 0.04]],
        [['.', 0.46], ['City', 0.50], ['!', 0.04]],
      ];
      const bias = { 'New': { York: 2.2, Lake: -2.2, Vegas: -1 }, 'Salt': { Lake: 2.2, York: -2.2 }, 'Las': { Vegas: 3 }, 'York': { '.': 2.2, City: -2.2 }, 'Lake': { City: 2.2, '.': -2.2 }, 'Vegas': { '.': 1.5 } };
      const conf = { 'New': 0.9, 'Salt': 0.88, 'York': 0.93, 'Lake': 0.9, '.': 0.95, 'City': 0.94 };
      const S = { markov: true, first: 'argmax', conf: true, t0: 0 };
      const ctl = UI.controls(fig);
      const reset = () => { S.t0 = performance.now() / 1000; };
      UI.seg(ctl, { label: 'Markov head', options: [{ v: false, label: 'off' }, { v: true, label: 'on' }], value: true, onChange: (v) => { S.markov = v; reset(); } });
      UI.seg(ctl, { label: '1st position', options: [{ v: 'argmax', label: 'argmax' }, { v: 'Rio', label: 'sampled “Salt”' }], value: 'argmax', onChange: (v) => { S.first = v; reset(); } });
      UI.toggle(ctl, { label: 'show confidence', value: true, onChange: (v) => { S.conf = v; } });
      UI.button(ctl, '▶ restart', reset);
      const softmax = (xs) => { const m = Math.max(...xs); const e = xs.map((x) => Math.exp(x - m)); const s = e.reduce((a, b) => a + b, 0); return e.map((x) => x / s); };
      const solve = () => {
        const ys = [], probs = [];
        let prev = 'Destination:';
        for (let i = 0; i < 3; i++) {
          const lb = cands[i].map(([wd, p]) => Math.log(p));
          const bs = cands[i].map(([wd]) => (S.markov && bias[prev] && bias[prev][wd]) || 0);
          const pr = softmax(lb.map((x, j) => x + bs[j]));
          let best = 0; pr.forEach((p, j) => { if (p > pr[best]) best = j; });
          if (i === 0 && S.first === 'Rio') best = 1;
          ys.push(best); probs.push({ pr, bs, base: cands[i].map(([, p]) => p) });
          prev = cands[i][best][0];
        }
        return { ys, probs };
      };
      S.t0 = performance.now() / 1000 - 3.8; // começa mostrando o resultado final
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const sol = solve();
        const narrow = w < 600;
        const el = performance.now() / 1000 - S.t0;
        const stepT = 1.2;
        const cur = S.markov ? Math.min(3, Math.floor(el / stepT)) : (el > 0.6 ? 3 : 0);
        if (el > 3 * stepT + 3) reset();
        const colW = (w - 40) / 3, cx = (i) => 20 + i * colW + colW / 2;
        // bloco de entrada
        const chipW = Math.min(96, colW - 16);
        const inY = 12;
        ['Destination:', 'noise', 'noise'].forEach((l, i) => chip(g, cx(i) - chipW / 2, inY, chipW, 24, l, { fill: V.alpha(i === 0 ? C.d : C.muted, 0.25), stroke: i === 0 ? C.d : V.alpha(C.muted, 0.5), font: i === 0 ? 'sans' : 'mono', size: 11.5 }));
        // backbone
        const bbY = inY + 36, bbH = 34;
        V.box(g, 20, bbY, w - 40, bbH, { r: 10, fill: V.alpha(C.b, 0.14 + 0.08 * Math.sin(t * 3)), stroke: C.b });
        V.text(g, narrow ? 'parallel backbone (non-causal)' : 'parallel backbone — 3 V4 stages + MoE, non-causal, one forward', w / 2, bbY + bbH / 2 + 1, { size: 11.5, color: C.ink, align: 'center', base: 'middle', maxW: w - 60 });
        for (let i = 0; i < 2; i++) { const u = (t * 0.7 + i * 0.5) % 1; g.strokeStyle = V.alpha(C.b, 0.6); g.lineWidth = 1.2; g.beginPath(); g.moveTo(cx(i), bbY + 4); g.quadraticCurveTo((cx(i) + cx(i + 1)) / 2, bbY - 14, cx(i + 1), bbY + 4); g.stroke(); V.glow(g, V.lerp(cx(i), cx(i + 1), u), bbY - 6 * Math.sin(Math.PI * u) - 2, 6, C.b, 0.8); }
        // barras
        const barTop = bbY + bbH + 40, barH = narrow ? 100 : 90;
        V.text(g, 'logits per position (solid bar = with bias; outline = marginal)', 20, bbY + bbH + 16, { size: 10.5, color: C.muted, maxW: w - 40 });
        for (let i = 0; i < 3; i++) {
          const P = sol.probs[i];
          const decided = i < cur;
          const bw = (colW - 30) / 3;
          cands[i].forEach(([wd], j) => {
            const bx = cx(i) - (colW - 30) / 2 + j * bw + 3;
            const pShow = (S.markov && i <= cur) ? P.pr[j] : P.base[j];
            const hb = barH * pShow, hbase = barH * P.base[j];
            const chosen = decided && sol.ys[i] === j;
            g.fillStyle = V.alpha(chosen ? C.a : C.b, chosen ? 0.85 : 0.4); V.rr(g, bx, barTop + barH - hb, bw - 6, Math.max(2, hb), 3); g.fill();
            g.strokeStyle = V.alpha(C.ink, 0.5); g.lineWidth = 1; g.setLineDash([3, 3]); V.rr(g, bx, barTop + barH - hbase, bw - 6, Math.max(2, hbase), 3); g.stroke(); g.setLineDash([]);
            const b = P.bs[j];
            if (S.markov && b && i <= cur && i > 0) V.text(g, b > 0 ? '+' : '−', bx + (bw - 6) / 2, barTop + barH - Math.max(hb, hbase) - 6, { size: 13, weight: 700, align: 'center', color: b > 0 ? C.a : C.c });
            V.text(g, wd, bx + (bw - 6) / 2, barTop + barH + 14, { size: narrow ? 9.5 : 11, font: 'mono', align: 'center', color: chosen ? C.ink : C.muted, maxW: bw });
          });
        }
        // setas de Markov
        const yChip = barTop + barH + 30;
        for (let i = 0; i < 3; i++) {
          if (i >= cur) continue;
          const wd = cands[i][sol.ys[i]][0];
          chip(g, cx(i) - chipW / 2, yChip, chipW, 24, wd, { fill: V.alpha(C.a, 0.3), stroke: C.a, weight: 700 });
          if (S.markov && i < 2) {
            const x1 = cx(i) + chipW / 2 + 2, x2 = cx(i + 1) - (colW - 30) / 2;
            const on = cur === i + 1 || (cur > i + 1);
            V.arrow(g, x1, yChip + 12, x2, barTop + barH * 0.4, { color: V.alpha(C.d, on ? 0.9 : 0.4), lw: 1.8, head: 7, curve: -30 });
            if (!narrow) V.text(g, 'W₂W₁[y]', (x1 + x2) / 2, yChip + 30, { size: 10.5, font: 'mono', align: 'center', color: C.d });
          }
          if (S.conf && S.markov) V.text(g, 'c=' + fmt(conf[wd] || 0.5, 2), cx(i), yChip + 40, { size: 10.5, font: 'mono', align: 'center', color: C.muted });
        }
        // frase final
        const fy = yChip + (narrow ? 58 : 62);
        if (cur >= 3) {
          const phrase = 'Destination: ' + sol.ys.map((j, i) => cands[i][j][0]).join(' ');
          const ok = (sol.ys[0] === 0 && sol.ys[1] === 0 && sol.ys[2] === 0) || (sol.ys[0] === 1 && sol.ys[1] === 1 && sol.ys[2] === 1);
          V.text(g, phrase, 20, fy, { size: 14, weight: 700, color: ok ? C.a : C.c, maxW: w - 150 });
          V.text(g, ok ? '✓ coherent' : narrow ? '✕ collision' : '✕ multimodal collision', w - 20, fy, { size: 12.5, weight: 700, align: 'right', color: ok ? C.a : C.c });
          if (S.conf && S.markov) {
            const cs = sol.ys.map((j, i) => conf[cands[i][j][0]] || 0.5);
            let a = 1; const surv = cs.map((c) => (a *= c));
            V.text(g, 'survival a_j = ' + surv.map((x) => fmt(x, 2)).join(' → '), 20, fy + 20, { size: 11, font: 'mono', color: C.muted, maxW: w - 40 });
          }
        } else V.text(g, S.markov ? 'sequential refinement: position ' + (cur + 1) + ' of 3…' : '…', 20, fy, { size: 12, color: C.muted });
        // inset: a consulta W₁[y] e a GEMV por W₂, com as formas reais
        const iy0 = fy + (S.conf && S.markov ? 34 : 18), ih = h - 6 - iy0;
        if (ih > 110) {
          V.box(g, 12, iy0, w - 24, ih, { r: 10, fill: V.alpha(C.muted, 0.05), stroke: V.alpha(C.muted, 0.22) });
          // a posição em decisão usa o token escolhido antes dela (y₀ = âncora)
          const posI = Math.min(cur, 2), prevI = posI - 1;
          const word = prevI >= 0 ? cands[prevI][sol.ys[prevI]][0] : 'Destination:';
          V.text(g, S.markov ? `${narrow ? '' : 'Markov head: '}lookup + GEMV, y = “${word}” → bias for pos. ${posI + 1}` : 'head off: no bias is added', 24, iy0 + 17, { size: 11, weight: 600, color: S.markov ? C.ink : C.muted, maxW: w - 48 });
          const mw = narrow ? 40 : 54, gap1 = narrow ? 74 : 110, vw = narrow ? 44 : 64;
          const dx0 = narrow ? 26 : 70, dy = iy0 + 28, dh = narrow ? Math.min(96, ih - 128) : ih - 58;
          const x2 = dx0 + mw + gap1, x3 = x2 + mw + (narrow ? 14 : 24), ow = narrow ? 11 : 14;
          g.save(); if (!S.markov) g.globalAlpha = 0.35;
          const ph = S.markov ? (t % 3.4) / 3.4 : 0;
          // W₁ e W₂ como matrizes altas [V × 256]
          [[dx0, 'W₁'], [x2, 'W₂']].forEach(([mx0, nm], mi) => {
            V.box(g, mx0, dy, mw, dh, { r: 3, fill: V.alpha(C.b, 0.12), stroke: V.alpha(C.b, 0.7), lw: 1 });
            g.strokeStyle = V.alpha(C.b, 0.18); g.lineWidth = 1; g.beginPath();
            for (let q = 1; q < 18; q++) { const yy = dy + (q / 18) * dh; g.moveTo(mx0 + 2, yy); g.lineTo(mx0 + mw - 2, yy); }
            g.stroke();
            V.text(g, nm, mx0 + mw / 2, dy + dh + 13, { size: 11, font: 'mono', weight: 700, align: 'center', color: C.b });
            V.text(g, '[129,280×256]', mx0 + mw / 2, dy + dh + 25, { size: 9.5, font: 'mono', align: 'center', color: C.muted, maxW: mw + (mi ? 22 : gap1 - 8) });
          });
          // 1) a linha y de W₁ acende
          const ry = dy + rowFrac(word) * (dh - 6);
          const lit = S.markov ? V.clamp(ph / 0.2, 0, 1) : 0;
          g.fillStyle = V.alpha(C.d, 0.25 + 0.7 * lit); g.fillRect(dx0 + 1, ry, mw - 2, 6);
          if (lit > 0.5) V.glow(g, dx0 + mw / 2, ry + 3, 14, C.d, 0.6);
          if (!narrow) V.text(g, 'row y', dx0 - 4, ry + 6, { size: 9.5, align: 'right', color: C.d });
          // 2) o vetor de 256 viaja até W₂
          const vRestX = dx0 + mw + (gap1 - vw) / 2, vRestY = dy + dh / 2 - 4;
          const mv = V.ease.inOut(V.clamp((ph - 0.2) / 0.22, 0, 1));
          const vx = V.lerp(dx0, vRestX, mv), vy = V.lerp(ry, vRestY, mv), vwn = V.lerp(mw, vw, mv);
          for (let k = 0; k < 8; k++) { g.fillStyle = V.alpha(C.d, 0.55 + 0.4 * ((k + Math.floor(t * 6)) % 8 === 0)); g.fillRect(vx + k * (vwn / 8), vy, vwn / 8 - 1, 8); }
          V.text(g, 'e ∈ ℝ²⁵⁶', vRestX + vw / 2, vRestY + 22, { size: 10, font: 'mono', align: 'center', color: C.d, alpha: mv });
          // 3) GEMV: a varredura desce por W₂ e enche a coluna de 129.280 logits
          const scan = V.clamp((ph - 0.45) / 0.45, 0, 1);
          if (scan > 0 && scan < 1) {
            const sy = dy + scan * dh;
            V.line(g, x2 + 1, sy, x2 + mw - 1, sy, { color: C.d, lw: 2 });
            V.line(g, vRestX + vw, vRestY + 4, x2, sy, { color: V.alpha(C.d, 0.5), lw: 1, dash: [3, 3] });
            V.glow(g, x2 + mw / 2, sy, 10, C.d, 0.6);
          }
          const nC = outCol.length, chh = dh / nC;
          for (let q = 0; q < nC; q++) {
            const on = (q + 0.5) / nC <= (scan > 0 ? scan : ph > 0.9 ? 1 : 0);
            g.fillStyle = on ? V.cmap(outCol[q], 'div') : V.alpha(C.muted, 0.12);
            g.fillRect(x3, dy + q * chh, ow, Math.max(1, chh - 1));
          }
          V.text(g, 'Δℓ', x3 + ow + 6, dy + 10, { size: 11, font: 'mono', weight: 700, color: C.ink });
          V.text(g, '[129,280]', x3 + ow + 6, dy + 23, { size: 9.5, font: 'mono', color: C.muted });
          V.text(g, '+ ℓ_base', x3 + ow + 6, dy + dh / 2 + 10, { size: 10.5, font: 'mono', color: C.ink });
          g.restore();
          // escala: bigrama cheio V×V contra as duas faixas fatoradas (largura real 512/129.280 da aresta)
          const sq = narrow ? 46 : Math.min(dh, 84);
          const sx = narrow ? dx0 : x3 + ow + 86, sy0 = narrow ? dy + dh + 38 : dy + (dh - sq) / 2;
          if (sx + sq + 150 < w) {
            g.save(); g.setLineDash([4, 3]); V.box(g, sx, sy0, sq, sq, { r: 2, fill: V.alpha(C.muted, 0.1), stroke: V.alpha(C.muted, 0.6), lw: 1 }); g.restore();
            const fw = Math.max(1.5, sq * 512 / 129280);
            g.fillStyle = C.d; g.fillRect(sx, sy0, fw, sq); g.fillRect(sx + fw + 1.5, sy0, fw, sq);
            const tx = sx + sq + 10;
            V.text(g, 'full V×V ≈ 33 GB', tx, sy0 + 13, { size: 10.5, color: C.muted, maxW: w - tx - 18 });
            V.text(g, 'W₁ + W₂ ≈ 2 × 66 MB', tx, sy0 + 29, { size: 10.5, weight: 600, color: C.d, maxW: w - tx - 18 });
            V.text(g, '= 0.4% of the area (strips to scale)', tx, sy0 + 44, { size: 10, color: C.muted, maxW: w - tx - 18 });
          }
        }
      });
    })();

    /* ============================================== FIG 7 — quanto verificar */
    (function () {
      const fig = root.querySelector('#fig-verificar');
      adaptH(fig, 420, 520);
      const S = { mode: 'glm', alpha: 0.25, margin: 1.0, sat: 'max', prof: 'glmP', seed: 3, cRow: 0.1, sched: 'hardware', th: 0.5, curve: 'decl', early: true };
      const ctl = UI.controls(fig);
      const gA = LA.h('div', { style: { display: 'contents' } }), gB = LA.h('div', { style: { display: 'contents' } });
      const setMode = (m) => { S.mode = m; gA.style.display = m === 'glm' ? 'contents' : 'none'; gB.style.display = m === 'ds' ? 'contents' : 'none'; recompute(); };
      UI.seg(ctl, { label: 'Mode', options: [{ v: 'glm', label: 'GLM · adaptive-k' }, { v: 'ds', label: 'DeepSeek · confidence' }], value: 'glm', onChange: setMode });
      ctl.appendChild(gA); ctl.appendChild(gB);
      UI.slider(gA, { label: 'EMA α', min: 0.05, max: 0.8, step: 0.05, value: S.alpha, fmt: (v) => fmt(v, 2), onInput: (v) => { S.alpha = v; recompute(); } });
      UI.slider(gA, { label: 'margin m', min: 0, max: 3, step: 0.25, value: S.margin, fmt: (v) => fmt(v, 2), onInput: (v) => { S.margin = v; recompute(); } });
      UI.seg(gA, { label: 'saturate', options: ['max', 'n'], value: 'max', onChange: (v) => { S.sat = v; recompute(); } });
      UI.seg(gA, { label: 'Text', options: [{ v: 'glmS', label: 'structured' }, { v: 'glmP', label: 'prose' }, { v: 'mix', label: 'prose → struct.' }], value: 'glmP', onChange: (v) => { S.prof = v; recompute(); } });
      UI.slider(gA, { label: 'cost c', min: 0, max: 0.3, step: 0.01, value: S.cRow, fmt: (v) => fmt(v, 2), onInput: (v) => { S.cRow = v; recompute(); } });
      UI.button(gA, '🎲 reroll', () => { S.seed++; recompute(); });
      UI.seg(gB, { label: 'Scheduler', options: [{ v: 'off', label: 'off' }, { v: 'threshold', label: 'threshold' }, { v: 'hardware', label: 'hardware' }], value: 'hardware', onChange: (v) => { S.sched = v; recompute(); } });
      UI.slider(gB, { label: 'threshold', min: 0, max: 1, step: 0.05, value: S.th, fmt: (v) => fmt(v, 2), onInput: (v) => { S.th = v; recompute(); } });
      UI.seg(gB, { label: 'sps curve', options: [{ v: 'flat', label: 'flat' }, { v: 'decl', label: 'drops with batch' }, { v: 'cliff', label: 'cliff' }], value: 'decl', onChange: (v) => { S.curve = v; recompute(); } });
      UI.toggle(gB, { label: 'early stop', value: true, onChange: (v) => { S.early = v; recompute(); } });
      const r1 = UI.readout(ctl, '—', '—'), r2 = UI.readout(ctl, '—', '—'), r3 = UI.readout(ctl, 'relative tok/s (illustr.)', '—');
      const lab1 = r1.el.querySelector('span'), lab2 = r2.el.querySelector('span');
      /* --- GLM: simulação do _Glm53AdaptiveK --- */
      const SET = [2, 4, 7], KMAX = 7, MIN_STEPS = 4, NST = 60;
      let sim = null;
      function simulate() {
        const r = V.rng(100 + S.seed);
        let ema = null, obsN = 0;
        const rows = [];
        for (let s = 0; s < NST; s++) {
          const prof = S.prof === 'mix' ? (s < 30 ? PROF.glmP.a : PROF.glmS.a) : PROF[S.prof].a;
          let n = KMAX;
          if (ema != null && obsN >= MIN_STEPS) {
            const target = Math.ceil(ema + S.margin);
            const c = SET.filter((v) => v <= Math.min(target, KMAX));
            n = c.length ? Math.max(...c) : Math.min(...SET);
          }
          const L = acceptedFromU(prof, KMAX, r());
          const acc = Math.min(L, n);
          const obs = acc >= n && S.sat !== 'n' ? Math.max(KMAX, n) : acc;
          ema = ema == null ? obs * S.alpha + KMAX * (1 - S.alpha) : obs * S.alpha + ema * (1 - S.alpha);
          obsN++;
          rows.push({ n, acc, L, ema });
        }
        const tokA = rows.reduce((a, x) => a + 1 + x.acc, 0) / NST, rowsA = rows.reduce((a, x) => a + x.n + 1, 0) / NST;
        const tokF = rows.reduce((a, x) => a + 1 + x.L, 0) / NST;
        return { rows, tokA, rowsA, tokF };
      }
      /* --- DS: hardware_aware_prefix_schedule (porte do overlay) --- */
      const REQ = [
        { n: 'JSON', c: [0.97, 0.96, 0.95, 0.93, 0.9, 0.88] },
        { n: 'code', c: [0.92, 0.85, 0.78, 0.7, 0.62, 0.55] },
        { n: 'prose', c: [0.72, 0.6, 0.5, 0.42, 0.35, 0.3] },
        { n: 'hard prose', c: [0.55, 0.38, 0.3, 0.22, 0.18, 0.12] },
      ];
      const surv = REQ.map((q) => { let a = 1; return q.c.map((c) => (a *= c)); });
      const sps = (bt) => (S.curve === 'flat' ? 1 : S.curve === 'decl' ? 1 / (1 + 0.045 * bt) : (bt <= 12 ? 1 : 0.8));
      let plan = null;
      function schedule() {
        const R = REQ.length;
        const cands = [];
        surv.forEach((sv, r) => sv.forEach((a, j) => cands.push([a, r, j + 1])));
        cands.sort((p, q) => q[0] - p[0] || p[1] - q[1] || p[2] - q[2]);
        const lens = Array(R).fill(0);
        let bt = R, E = R;
        let best = { lens: lens.slice(), E, bt, tps: E * sps(bt) };
        const path = [{ bt, tps: E * sps(bt), stop: false }];
        let stopped = false, stopIdx = -1;
        for (const [a, r, pos] of cands) {
          if (pos !== lens[r] + 1) continue;
          lens[r] = pos; bt += 1; E += a;
          const tps = E * sps(bt);
          path.push({ bt, tps });
          if (!stopped) {
            if (tps > best.tps) best = { lens: lens.slice(), E, bt, tps };
            else if (S.early) { stopped = true; stopIdx = path.length - 1; }
          }
        }
        if (S.sched === 'off' || S.sched === 'threshold') {
          const L2 = surv.map((sv) => (S.sched === 'off' ? sv.length : (() => { let n = 0; for (const a of sv) { if (a < S.th) break; n++; } return n; })()));
          const E2 = R + L2.reduce((s, l, r) => s + surv[r].slice(0, l).reduce((a, b) => a + b, 0), 0);
          const bt2 = R + L2.reduce((a, b) => a + b, 0);
          return { lens: L2, E: E2, bt: bt2, tps: E2 * sps(bt2), path, stopIdx: -1 };
        }
        return Object.assign(best, { path, stopIdx });
      }
      function recompute() {
        r3.el.style.display = S.mode === 'glm' ? '' : 'none';
        if (S.mode === 'glm') {
          sim = simulate(); lab1.textContent = 'tokens/step'; r1.set(fmt(sim.tokA, 2) + ` (fixed k=7: ${fmt(sim.tokF, 2)})`); lab2.textContent = 'rows/request'; r2.set(fmt(sim.rowsA, 2) + ' (fixed: 8)');
          const vA = sim.tokA / (1 + S.cRow * sim.rowsA), vF = sim.tokF / (1 + S.cRow * 8), dp = (vA / vF - 1) * 100;
          r3.set(`${fmt(vA, 2)} vs ${fmt(vF, 2)} (${dp >= 0 ? '+' : '−'}${fmt(Math.abs(dp), 0)}%)`);
        } else { plan = schedule(); lab1.textContent = 'E[accepted] · rows'; r1.set(fmt(plan.E, 2) + ' · ' + plan.bt); lab2.textContent = 'relative E[tok/s]'; r2.set(fmt(plan.tps, 3)); }
      }
      setMode('glm');
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const narrow = w < 600;
        if (S.mode === 'glm') {
          if (!sim) return;
          const L = 36, Rm = 14, T = narrow ? 42 : 30, Bm = 34;
          const pw = w - L - Rm, ph = h - T - Bm;
          const X = (s) => L + (s + 0.5) * (pw / NST), Y = (v) => T + ph - (v / 7) * ph;
          for (let v = 0; v <= 7; v++) { V.line(g, L, Y(v), L + pw, Y(v), { color: V.alpha(C.muted, v % 7 ? 0.1 : 0.3), lw: 1 }); if (v % 1 === 0) V.text(g, String(v), L - 8, Y(v) + 4, { size: 10, font: 'mono', align: 'right', color: C.muted }); }
          SET.forEach((v) => V.text(g, 'n=' + v, L + pw, Y(v) - 3, { size: 9.5, font: 'mono', align: 'right', color: V.alpha(C.a, 0.8) }));
          const rev = Math.min(NST, Math.floor((t * 9 + NST) % (NST + 24)));
          if (S.prof === 'mix') { V.line(g, X(29.5), T, X(29.5), T + ph, { color: V.alpha(C.d, 0.6), dash: [4, 4] }); V.text(g, 'switches to structured', X(30) + 4, T + 12, { size: 10.5, color: C.d }); }
          // n (escada)
          g.strokeStyle = C.a; g.lineWidth = 2.2; g.beginPath();
          sim.rows.slice(0, rev).forEach((r, s) => { const x0 = L + s * (pw / NST), x1 = x0 + pw / NST; s ? g.lineTo(x0, Y(r.n)) : g.moveTo(x0, Y(r.n)); g.lineTo(x1, Y(r.n)); });
          g.stroke();
          // EMA
          g.strokeStyle = C.b; g.lineWidth = 1.8; g.beginPath();
          sim.rows.slice(0, rev).forEach((r, s) => { s ? g.lineTo(X(s), Y(Math.min(7.5, r.ema))) : g.moveTo(X(s), Y(Math.min(7.5, r.ema))); });
          g.stroke();
          // aceitos
          sim.rows.slice(0, rev).forEach((r, s) => { V.dot(g, X(s), Y(r.acc), 3, r.acc >= r.n ? C.d : C.c); });
          if (rev > 0 && rev <= NST) V.glow(g, X(rev - 1), Y(sim.rows[rev - 1].acc), 10, C.ink, 0.5);
          V.text(g, 'step →', L + pw, h - 10, { size: 10.5, color: C.muted, align: 'right' });
          const lg = [[C.a, 'verified n'], [C.b, 'EMA'], [C.c, 'accepted'], [C.d, 'accepted all (censored)']];
          lg.forEach(([c, l], i) => { const lx = L + i * (narrow ? (pw / 2) : 150), ly = narrow ? (i < 2 ? 12 : 24) : 16; const lxx = narrow ? L + (i % 2) * (pw / 2) : lx; V.dot(g, lxx + 4, ly - 4, 4, c); V.text(g, l, lxx + 12, ly, { size: 10.5, color: C.muted, maxW: narrow ? pw / 2 - 16 : 140 }); });
        } else {
          if (!plan) return;
          const gridW = narrow ? w - 20 : w * 0.56;
          const labW = 86, cellW = Math.min(52, (gridW - labW) / 6 - 4), cellH = narrow ? 30 : 42;
          const oy = 44;
          V.text(g, 'survival a(r,j) = Π c (illustrative confidences)', 10, 18, { size: 11, color: C.muted, maxW: gridW });
          for (let j = 0; j < 6; j++) V.text(g, 'j=' + (j + 1), 10 + labW + j * (cellW + 4) + cellW / 2, oy - 8, { size: 10, font: 'mono', align: 'center', color: C.muted });
          REQ.forEach((q, r) => {
            const y = oy + r * (cellH + 6);
            V.text(g, q.n, 10 + labW - 8, y + cellH / 2 + 1, { size: 11, align: 'right', base: 'middle', color: C.ink, maxW: labW - 10 });
            surv[r].forEach((a, j) => {
              const x = 10 + labW + j * (cellW + 4);
              const adm = j < plan.lens[r];
              g.fillStyle = V.cmap(a, 'seq'); V.rr(g, x, y, cellW, cellH, 5); g.fill();
              if (adm) { V.box(g, x - 1.5, y - 1.5, cellW + 3, cellH + 3, { r: 6, stroke: C.ink, lw: 2 }); }
              else { g.fillStyle = V.alpha(C.bg || '#000', 0.55); V.rr(g, x, y, cellW, cellH, 5); g.fill(); }
              if (cellW >= 30) V.text(g, fmt(a, 2), x + cellW / 2, y + cellH / 2 + 1, { size: 10.5, font: 'mono', align: 'center', base: 'middle', color: a > 0.6 ? (C.dark ? '#00210F' : '#ffffff') : C.ink });
            });
          });
          const gy = oy + 4 * (cellH + 6) + 6;
          V.text(g, 'outline = verified; faded = cut', 10, gy + 8, { size: 10.5, color: C.muted });
          // gráfico E[tok/s] × batch_tokens
          const px0 = narrow ? 40 : w * 0.62, py0 = narrow ? gy + 30 : 44, pw = narrow ? w - 60 : w * 0.35, ph = narrow ? h - gy - 60 : h - 110;
          V.box(g, px0, py0, pw, ph, { r: 6, fill: V.alpha(C.muted, 0.05), stroke: V.alpha(C.muted, 0.25) });
          V.text(g, 'E[tok/s] × rows in batch', px0, py0 - 8, { size: 10.5, color: C.muted, maxW: pw });
          const pts = plan.path.length ? plan.path : [{ bt: plan.bt, tps: plan.tps }];
          const all = pts.map((p) => p.tps).concat([plan.tps]);
          const ymax = Math.max(...all) * 1.1, ymin = Math.min(...all) * 0.9;
          const X = (bt) => px0 + ((bt - 4) / 24) * pw, Y = (v) => py0 + ph - ((v - ymin) / (ymax - ymin || 1)) * ph;
          if (plan.path.length) {
            g.strokeStyle = V.alpha(C.b, 0.9); g.lineWidth = 1.8; g.beginPath();
            pts.forEach((p, i) => (i ? g.lineTo(X(p.bt), Y(p.tps)) : g.moveTo(X(p.bt), Y(p.tps)))); g.stroke();
            pts.forEach((p, i) => V.dot(g, X(p.bt), Y(p.tps), 2.4, i === plan.stopIdx ? RED() : C.b));
            const rev = Math.floor((t * 6) % (pts.length + 8));
            if (rev < pts.length) V.glow(g, X(pts[rev].bt), Y(pts[rev].tps), 9, C.b, 0.9);
          }
          V.dot(g, X(plan.bt), Y(plan.tps), 6, C.a);
          const nearR = X(plan.bt) > px0 + pw - 80;
          V.text(g, S.sched === 'hardware' ? 'chosen' : S.sched === 'off' ? 'off: verify all' : 'threshold', X(plan.bt) + (nearR ? -10 : 10), Y(plan.tps) - 8, { size: 10.5, color: C.a, align: nearR ? 'right' : 'left' });
          if (plan.stopIdx > 0) {
            const ex = X(pts[plan.stopIdx].bt), ey = Y(pts[plan.stopIdx].tps), lx = Math.min(ex + 6, px0 + pw - 76), lyy = Math.min(ey + 34, py0 + ph - 26);
            V.line(g, ex, ey + 3, lx + 30, lyy - 2, { color: V.alpha(RED(), 0.6), lw: 1 });
            V.box(g, lx, lyy - 2, 66, 18, { r: 5, fill: V.alpha(C.bg || C.panel, 0.92), stroke: V.alpha(RED(), 0.7), lw: 1 });
            V.text(g, 'early stop', lx + 33, lyy + 11, { size: 10, color: RED(), align: 'center' });
          }
          if (S.sched !== 'hardware') V.text(g, 'line = greedy path of hardware mode', px0 + 8, py0 + ph - 8, { size: 10, color: C.muted, maxW: pw - 16 });
          V.text(g, '4', px0, py0 + ph + 12, { size: 9.5, font: 'mono', color: C.muted });
          V.text(g, '28 rows', px0 + pw, py0 + ph + 12, { size: 9.5, font: 'mono', color: C.muted, align: 'right' });
        }
      });
    })();

    /* ============================================== FIG 8 — páginas */
    (function () {
      const fig = root.querySelector('#fig-paginas');
      adaptH(fig, 340, 380);
      const TP = { 2: { Bm: 3584, bm: 656, bd: 2048 }, 3: { Bm: 2560, bm: 656, bd: 1536 } };
      const S = { tp: 2, mode: 'compact', idx: 0 };
      const candsOf = (tp) => { const { Bm } = TP[tp]; const out = []; for (let B = 64; B <= Bm; B += 64) if (Bm % B === 0) out.push(B); return out; };
      const ctl = UI.controls(fig);
      const rB = UI.readout(ctl, 'drafter block', '—'), rI = UI.readout(ctl, 'ids/request', '—'), rO = UI.readout(ctl, 'page used', '—');
      let sl;
      const curB = () => {
        const c = candsOf(S.tp), { Bm, bm, bd } = TP[S.tp], page = Bm * bm;
        if (S.mode === 'slot') return 64;
        if (S.mode === 'compact') return Math.max(...c.filter((B) => B * bd <= page));
        return c[Math.min(S.idx, c.length - 1)];
      };
      const syncSl = () => { if (!sl) return; const c = candsOf(S.tp), i = Math.max(0, c.indexOf(curB())); sl.input.value = i; const v = sl.el.querySelector('.val'); if (v) v.textContent = String(c[i]); };
      const upd = () => { const B = curB(), { Bm, bm, bd } = TP[S.tp]; syncSl(); rB.set(B + ' tokens'); rI.set(String(Math.ceil(4096 / B) + 1)); rO.set(Math.round((B * bd) / (Bm * bm) * 100) + '%'); };
      UI.seg(ctl, { label: 'Topology', options: [{ v: 2, label: 'TP2' }, { v: 3, label: 'TP3' }], value: 2, onChange: (v) => { S.tp = v; sl.input.max = candsOf(v).length - 1; S.idx = Math.min(S.idx, candsOf(v).length - 1); upd(); } });
      const segB = UI.seg(ctl, { label: 'Block', options: [{ v: 'slot', label: 'slot-share 64' }, { v: 'compact', label: 'COMPACT=1' }, { v: 'manual', label: 'manual' }], value: 'compact', onChange: (v) => { if (v === 'manual' && S.mode !== 'manual') S.idx = +sl.input.value; S.mode = v; upd(); } });
      sl = UI.slider(ctl, { label: 'candidate', min: 0, max: candsOf(2).length - 1, step: 1, value: 0, fmt: (v) => String(candsOf(S.tp)[Math.min(v, candsOf(S.tp).length - 1)]), onInput: (v) => { S.idx = v; if (S.mode !== 'manual') segB.set('manual'); upd(); } });
      upd();
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const narrow = w < 600;
        const { Bm, bm, bd } = TP[S.tp], page = Bm * bm, B = curB();
        const x0 = 14, bw = w - 28;
        // 1) página
        let y = 20;
        V.text(g, `MLA physical page: ${fmtInt(Bm)} tok × ${bm} B = ${fmtInt(page)} B`, x0, y, { size: 11.5, color: C.ink, maxW: bw });
        y += 10;
        const fitW = bw * 0.82;
        V.box(g, x0, y, fitW, 26, { r: 6, fill: V.alpha(C.b, 0.14), stroke: C.b });
        const dp = B * bd, frac = dp / page;
        const fits = frac <= 1;
        const fw = Math.min(bw, fitW * frac);
        V.box(g, x0, y + 4, Math.max(3, fw), 18, { r: 4, fill: V.alpha(fits ? C.a : RED(), 0.75) });
        V.text(g, narrow ? `draft: ${B} × ${bd} B (${Math.round(frac * 100)}%)` : `drafter page: ${B} × ${bd} B = ${fmtInt(dp)} B (${Math.round(frac * 100)}%)`, x0 + 8, y + 14, { size: 10.5, font: 'mono', base: 'middle', color: C.ink, maxW: bw - 16 });
        if (!fits) V.text(g, '✕ doesn’t fit', x0 + fitW + 6, y + 18, { size: 11, color: RED(), weight: 700 });
        // 2) candidatos
        y += 50;
        V.text(g, `multiples of 64 that divide ${Bm}:`, x0, y, { size: 11, color: C.muted });
        y += 8;
        const cs = candsOf(S.tp), cwid = Math.min(62, (bw - (cs.length - 1) * 6) / cs.length);
        cs.forEach((c, i) => {
          const ok = c * bd <= page, on = c === B;
          chip(g, x0 + i * (cwid + 6), y, cwid, 22, String(c), { fill: V.alpha(ok ? C.a : RED(), on ? 0.6 : 0.14), stroke: on ? C.ink : V.alpha(ok ? C.a : RED(), 0.6), lw: on ? 2 : 1, size: 11 });
        });
        // 3) fita de tokens
        y += 48;
        V.text(g, 'drafter tokens per request: 2048 window + 2048 in flight', x0, y, { size: 11, color: C.ink, maxW: bw });
        y += 8;
        const nb = Math.ceil(4096 / B);
        const tapeH = 24;
        V.box(g, x0, y, bw, tapeH, { r: 4, fill: V.alpha(C.muted, 0.08) });
        const cursor = ((t * 700) % 4096);
        for (let k = 0; k < nb; k++) {
          const a0 = (k * B) / 4096 * bw, a1 = Math.min(bw, ((k + 1) * B) / 4096 * bw);
          const filled = cursor >= k * B;
          g.fillStyle = V.alpha(k % 2 ? C.a : C.b, filled ? 0.7 : 0.25); g.fillRect(x0 + a0, y, Math.max(1, a1 - a0 - (nb < 80 ? 1 : 0)), tapeH);
        }
        V.line(g, x0 + (cursor / 4096) * bw, y - 3, x0 + (cursor / 4096) * bw, y + tapeH + 3, { color: C.ink, lw: 1.5 });
        V.line(g, x0 + bw / 2, y - 4, x0 + bw / 2, y + tapeH + 4, { color: C.d, lw: 1.5, dash: [3, 3] });
        V.text(g, 'window', x0 + bw / 4, y + tapeH + 14, { size: 10.5, color: C.muted, align: 'center' });
        V.text(g, 'in flight', x0 + (3 * bw) / 4, y + tapeH + 14, { size: 10.5, color: C.muted, align: 'center' });
        // 4) ids
        y += tapeH + 34;
        const ids = nb + 1;
        V.text(g, `block ids: ⌈4096 / ${B}⌉ + 1 = ${ids}`, x0, y, { size: 13, weight: 700, color: ids <= 8 ? C.a : C.d });
        y += 10;
        const sq = narrow ? 10 : 14, perRow = Math.floor(bw / (sq + 3));
        for (let k = 0; k < ids; k++) {
          const rx = x0 + (k % perRow) * (sq + 3), ry = y + Math.floor(k / perRow) * (sq + 3);
          if (ry > h - sq - 4) break;
          const pulse = Math.floor(t * 8) % ids === k;
          g.fillStyle = V.alpha(k === ids - 1 ? C.d : C.b, pulse ? 1 : 0.65); V.rr(g, rx, ry, sq, sq, 2); g.fill();
        }
        // 5) pool medido (TP2)
        const usedRows = Math.ceil(ids / perRow);
        const yb = Math.max(y + usedRows * (sq + 3) + 26, h - 58);
        if (yb + 44 <= h) {
          V.text(g, S.tp === 2 ? 'logged pool with 14 GiB of KV (TP2, 850k), measured:' : 'pool measured on TP2 only:', x0, yb, { size: 11, color: C.muted, maxW: bw });
          const bars = [['no compact', 876958, S.mode === 'slot'], ['compact', 1572073, S.mode === 'compact']];
          const lw = 92, full = Math.max(60, bw - lw - 190);
          bars.forEach(([lb, v, on], i) => {
            const yy = yb + 8 + i * 18, ww = full * (v / 1572073);
            V.text(g, lb, x0, yy + 11, { size: 11, color: on ? C.ink : C.muted, weight: on ? 700 : 400 });
            g.fillStyle = V.alpha(i ? C.a : C.b, on ? 0.85 : 0.35); V.rr(g, x0 + lw, yy, ww, 13, 3); g.fill();
            V.text(g, fmtInt(v) + ' tokens' + (i ? ' (1.85×)' : ''), x0 + lw + ww + 6, yy + 11, { size: 10.5, font: 'mono', color: on ? C.ink : C.muted });
          });
        }
      });
    })();

    /* ============================================== FIG 9 — anel, condensação, ragged */
    (function () {
      const fig = root.querySelector('#fig-anel');
      adaptH(fig, 420, 430);
      const W = 16;
      const S = { scene: 'ring', fix: true, ev: 0, lastEv: 0 };
      const ring = { slots: Array.from({ length: W }, (_, i) => ({ pos: 24 + i, st: 'ok' })), P: 40, hist: null };
      for (let p = 24; p < 40; p++) ring.slots[p % W] = { pos: p, st: 'ok' };
      const rr = V.rng(9);
      function ringStep() {
        const k = 6, rows = k + 1;
        const j = acceptedFromU(PROF.ds.a, k, rr());
        const acc = j + 1;
        const touched = [];
        for (let q = 0; q < rows; q++) {
          const pos = ring.P + q, sl = pos % W;
          if (q < acc) { ring.slots[sl] = { pos, st: 'new' }; touched.push([sl, 'new']); }
          else if (!S.fix) { ring.slots[sl] = { pos, st: 'bad' }; touched.push([sl, 'bad']); }
          else { touched.push([sl, 'kept']); if (ring.slots[sl].st === 'new') ring.slots[sl].st = 'ok'; }
        }
        ring.slots.forEach((s, i) => { if (s.st === 'new' && !touched.find((x) => x[0] === i)) s.st = 'ok'; });
        ring.hist = { acc, rej: rows - acc, touched, P: ring.P };
        ring.P += acc;
      }
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Scene', options: [{ v: 'ring', label: '1 · ring' }, { v: 'cond', label: '2 · condensation' }, { v: 'rag', label: '3 · ragged' }], value: 'ring', onChange: (v) => { S.scene = v; S.ev = 0; } });
      UI.toggle(ctl, { label: 'fix on (mask / Patch 1 / Patch 2)', value: true, onChange: (v) => { S.fix = v; } });
      UI.button(ctl, '▶ next event', () => { if (S.scene === 'ring') ringStep(); else S.ev = (S.ev + 1) % 3; S.lastEv = performance.now() / 1000; }, { primary: true });
      V.every(2400, () => { if (S.scene === 'ring' && performance.now() / 1000 - S.lastEv > 4) ringStep(); });
      ringStep();
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const narrow = w < 600;
        if (S.scene === 'ring') {
          const R = Math.min(narrow ? w * 0.32 : 130, h * 0.34), cx = narrow ? w / 2 : 30 + R + 20, cy = narrow ? R + 34 : h / 2 + 6;
          V.text(g, `ring: slot = pos mod ${W} (real: 128)`, narrow ? 14 : 20, 18, { size: 11.5, color: C.muted });
          for (let i = 0; i < W; i++) {
            const a0 = (i / W) * Math.PI * 2 - Math.PI / 2, a1 = ((i + 1) / W) * Math.PI * 2 - Math.PI / 2;
            const s = ring.slots[i];
            const col = s.st === 'new' ? C.a : s.st === 'bad' ? RED() : C.b;
            const age = Math.max(0, Math.min(1, (ring.P - s.pos) / W));
            g.beginPath(); g.arc(cx, cy, R, a0 + 0.02, a1 - 0.02); g.arc(cx, cy, R * 0.62, a1 - 0.02, a0 + 0.02, true); g.closePath();
            g.fillStyle = V.alpha(col, s.st === 'ok' ? 0.25 + 0.45 * (1 - age) : 0.8); g.fill();
            const kept = ring.hist && S.fix && ring.hist.touched.find((x) => x[0] === i && x[1] === 'kept');
            if (kept) { g.save(); g.setLineDash([3, 3]); g.strokeStyle = C.d; g.lineWidth = 2; g.stroke(); g.restore(); }
            const am = (a0 + a1) / 2;
            V.text(g, String(s.pos), cx + Math.cos(am) * R * 0.81, cy + Math.sin(am) * R * 0.81 + 4, { size: R > 100 ? 11 : 9.5, font: 'mono', align: 'center', color: C.ink });
          }
          const hand = ((ring.P - 1) % W + 0.5) / W * Math.PI * 2 - Math.PI / 2;
          V.arrow(g, cx, cy, cx + Math.cos(hand) * R * 0.55, cy + Math.sin(hand) * R * 0.55, { color: C.ink, lw: 2, head: 7 });
          V.glow(g, cx + Math.cos(hand) * R * 0.55, cy + Math.sin(hand) * R * 0.55, 12, C.a, 0.5 + 0.3 * Math.sin(t * 4));
          V.text(g, 'pos ' + (ring.P - 1), cx, cy + 4, { size: 11, font: 'mono', align: 'center', color: C.muted });
          const tx = narrow ? 14 : cx + R + 40, ty = narrow ? cy + R + 30 : 60, tw = narrow ? w - 28 : w - tx - 16;
          if (ring.hist) {
            const H = ring.hist;
            const lines = [
              [C.ink, `step at pos ${H.P}: verified 7 rows (k=6)`],
              [C.a, `${H.acc} accepted → written to the ring`],
              [S.fix ? C.d : RED(), S.fix ? `${H.rej} rejected → slot keeps its old value` : `${H.rej} rejected → wrote a future that never happened`],
              [C.muted, S.fix ? 'VLLM_DSPARK_GPU_REJECTED_CONTEXT_MASK=1' : 'no mask: the drafter reads garbage as context'],
            ];
            lines.forEach(([c, l], i) => V.text(g, l, tx, ty + i * 22, { size: 12, color: c, maxW: tw, weight: i === 0 ? 600 : 400 }));
            const bad = ring.slots.filter((s) => s.st === 'bad').length;
            V.text(g, `polluted slots: ${bad}`, tx, ty + 4 * 22 + 8, { size: 13, weight: 700, color: bad ? RED() : C.a });
          }
        } else if (S.scene === 'cond') {
          const reqs = { A: C.a, B: C.b, C: C.c, D: C.d, E: C.e };
          const ev = S.ev;
          const rows = ev === 0 ? ['A', 'B', 'C', 'D'] : ev === 1 ? ['A', 'D', 'C', null] : ['A', 'D', 'C', 'E'];
          const slotOwner = ev === 0 ? ['A', 'B', 'C', 'D'] : ev === 1 ? ['A', 'B', 'C', 'D'] : (S.fix ? ['A', 'E', 'C', 'D'] : ['A', 'B', 'C', 'D']);
          const map = S.fix ? (ev === 0 ? { A: 0, B: 1, C: 2, D: 3 } : ev === 1 ? { A: 0, C: 2, D: 3 } : { A: 0, C: 2, D: 3, E: 1 }) : null;
          const titles = ['4 requests in flight', 'B finished → batch condensed (D moves up to row 1)', 'E arrives'];
          V.text(g, titles[ev], 14, 20, { size: 12, weight: 600, color: C.ink, maxW: w - 28 });
          const lx = 20, ly = 44, rowH = V.clamp((h - 150) / 4 - 8, 34, 60), colW = narrow ? w * 0.36 : 150;
          const sx = narrow ? w * 0.56 : w * 0.5;
          V.text(g, 'batch rows', lx, ly - 6, { size: 10.5, color: C.muted });
          V.text(g, 'drafter rings (slots)', sx, ly - 6, { size: 10.5, color: C.muted, maxW: w - sx - 10 });
          for (let r = 0; r < 4; r++) {
            const y = ly + r * (rowH + 8);
            V.box(g, lx, y, colW, rowH, { r: 8, fill: V.alpha(C.muted, 0.06), stroke: V.alpha(C.muted, 0.3) });
            V.text(g, 'row ' + r, lx + 8, y + rowH / 2 + 4, { size: 10.5, font: 'mono', color: C.muted });
            const q = rows[r];
            if (q) tokChar(g, lx + colW - 24, y + rowH / 2 + (ev === 1 && r === 1 ? Math.sin(t * 6) * 2 : 0), Math.min(26, rowH - 10), reqs[q]);
            if (q) V.text(g, q, lx + colW - 44, y + rowH / 2 + 5, { size: 13, weight: 700, color: reqs[q], align: 'right' });
            // slot
            const sw = Math.min(narrow ? w - sx - 14 : 150, w - sx - 14);
            const own = slotOwner[r];
            V.box(g, sx, y, sw, rowH, { r: 8, fill: V.alpha(reqs[own], 0.18), stroke: reqs[own] });
            for (let k = 0; k < 8; k++) { g.fillStyle = V.alpha(reqs[own], 0.3 + 0.5 * ((k + Math.floor(t * 3)) % 8 === 0)); g.fillRect(sx + 8 + k * ((sw - 50) / 8), y + rowH / 2 - 5, (sw - 50) / 8 - 2, 10); }
            V.text(g, 'S' + r + ' · hist. ' + own, sx + sw - 6, y + 12, { size: 10, font: 'mono', align: 'right', color: C.ink });
            // quem lê qual slot
            if (q) {
              const slot = S.fix ? map[q] : r;
              const ty = ly + slot * (rowH + 8) + rowH / 2;
              const wrong = slotOwner[slot] !== q;
              V.arrow(g, lx + colW + 4, y + rowH / 2, sx - 4, ty, { color: wrong ? RED() : V.alpha(C.a, 0.8), lw: wrong ? 2.4 : 1.4, head: 6 });
              if (wrong) V.text(g, '⚡', (lx + colW + sx) / 2, (y + ty) / 2 + rowH / 2 - 8, { size: 16, align: 'center', color: RED() });
            }
          }
          const by = ly + 4 * (rowH + 8) + 12;
          const wrongAny = rows.some((q, r) => q && (S.fix ? slotOwner[map[q]] !== q : slotOwner[r] !== q));
          V.text(g, S.fix ? 'Patch 1: _req_id_to_slot = ' + JSON.stringify(map).replace(/"/g, '') + (ev >= 1 ? '  · lowest free slot first' : '') : 'no Patch 1: slot = batch row', 14, by, { size: 11, font: 'mono', color: S.fix ? C.a : C.muted, maxW: w - 28 });
          V.text(g, wrongAny ? '✕ a request reading another’s history → acceptance ≈ 0 (no crash)' : '✓ each request reads its own ring', 14, by + 20, { size: 12, weight: 700, color: wrongAny ? RED() : C.a, maxW: w - 28 });
        } else {
          const pLen = S.fix ? 33 : 34;
          const segs = [{ n: 'P (prefill chunk)', len: pLen, c: C.b, rej: 0 }, { n: 'Q (decode, k+1 = 7)', len: 7, c: C.a, rej: 3 }];
          const tot = pLen + 7;
          V.text(g, S.fix ? `today: ${tot} rows, batch_size = 2 (unequal segments, ${tot} % 2 = 0)` : 'historical: 41 rows, batch_size = 2', 14, 20, { size: 12, weight: 600, color: C.ink, maxW: w - 28 });
          const x0 = 14, bw = w - 28, cw = bw / tot, y = 48, th = 28;
          let acc = 0;
          segs.forEach((sg) => { for (let k = 0; k < sg.len; k++) { g.fillStyle = V.alpha(sg.c, 0.55 + 0.25 * ((k + Math.floor(t * 6)) % sg.len === 0)); g.fillRect(x0 + (acc + k) * cw, y, Math.max(1, cw - 1), th); } acc += sg.len; });
          if (!S.fix) {
            // reshape retangular falhando (antes do Patch 2)
            const ry = y + 56;
            V.text(g, 'rectangular reshape [2, 41/2, H] …', x0, ry, { size: 12, font: 'mono', color: C.muted, maxW: bw });
            for (let r = 0; r < 2; r++) for (let k = 0; k < 20; k++) { g.fillStyle = V.alpha(C.muted, 0.25); g.fillRect(x0 + k * cw * 1.9, ry + 12 + r * 18, cw * 1.9 - 2, 14); }
            g.fillStyle = V.alpha(RED(), 0.8); g.fillRect(x0 + 20 * cw * 1.9, ry + 12, cw * 1.9 - 2, 14);
            const shake = Math.sin(t * 20) * 2;
            V.box(g, x0 + shake, ry + 60, Math.min(bw, 380), 32, { r: 8, fill: V.alpha(RED(), 0.18), stroke: RED() });
            V.text(g, 'HTTP 500: got 41 rows for batch_size=2', x0 + 12 + shake, ry + 80, { size: 12, font: 'mono', color: RED(), weight: 700, maxW: Math.min(bw, 380) - 20 });
            V.text(g, 'with Patch 2 but not 2b, prefills without rejections still gave 500', x0, ry + 116, { size: 11, color: C.muted, maxW: bw });
            V.text(g, '(“166 rows for batch_size=3”)', x0, ry + 132, { size: 11, font: 'mono', color: C.muted, maxW: bw });
            if (ry + 186 < h - 14) {
              V.box(g, x0, ry + 150, bw, 36, { r: 7, fill: V.alpha(C.d, 0.1), stroke: V.alpha(C.d, 0.5), lw: 1 });
              V.text(g, 'in the current overlay: 41 % 2 ≠ 0 → the guard skips speculation', x0 + 8, ry + 165, { size: 11, color: C.d, maxW: bw - 16 });
              V.text(g, 'on this step (returns zeros, no HTTP 500)', x0 + 8, ry + 180, { size: 11, color: C.d, maxW: bw - 16 });
            }
          } else {
            const qsl = [0, pLen, tot];
            qsl.forEach((q) => { const xx = x0 + q * cw; V.line(g, xx, y - 8, xx, y + th + 8, { color: C.ink, lw: 2 }); V.text(g, String(q), xx, y + th + 22, { size: 11, font: 'mono', align: 'center', color: C.ink }); });
            V.text(g, `query_start_loc = [0, ${pLen}, ${tot}]`, x0, y + th + 42, { size: 12, font: 'mono', color: C.a, weight: 600 });
            // âncoras
            let st0 = 0;
            segs.forEach((sg, i) => {
              const anchor = st0 + Math.max(0, Math.min(sg.len - sg.rej - 1, sg.len - 1));
              const ax = x0 + (anchor + 0.5) * cw;
              tokChar(g, ax, y - 18 + Math.sin(t * 4 + i) * 2, 14, C.d);
              for (let k = sg.len - sg.rej; k < sg.len; k++) { g.strokeStyle = RED(); g.lineWidth = 1.5; g.strokeRect(x0 + (st0 + k) * cw + 1, y + 1, cw - 3, th - 2); }
              st0 += sg.len;
            });
            const ly = y + th + 64;
            const lines = [
              [C.d, 'anchor = start + clamp(len − rejected − 1, 0, len − 1)'],
              [C.b, `P: anchor at row ${pLen - 1} (prefill, no rejection)`],
              [C.a, `Q: 3 rejected → anchor at row ${pLen + 3}`],
              [C.muted, '_store_main_kv_ragged writes each segment at pos % 128'],
              [C.muted, 'mixed steps never become CUDA graphs → Python loop ok'],
            ];
            lines.forEach(([c, l], i) => V.text(g, l, x0, ly + i * 19, { size: 11.5, color: c, maxW: bw }));
            const gy = ly + lines.length * 19 + 8;
            if (gy + 30 < h - 14) {
              V.box(g, x0, gy, bw, 30, { r: 7, fill: V.alpha(C.d, 0.1), stroke: V.alpha(C.d, 0.5), lw: 1 });
              V.text(g, '41 rows / 2 today: the guard (41 % 2 ≠ 0) skips speculation before the ragged branch', x0 + 8, gy + 19, { size: 11, color: C.d, maxW: bw - 16 });
            }
          }
          V.text(g, S.fix ? 'illustrative 33 + 7 split' : 'illustrative 34 + 7 split', w - 14, h - 8, { size: 10, color: C.muted, align: 'right' });
        }
      });
    })();

    /* ============================================== FIG 10 — validador de k */
    (function () {
      const fig = root.querySelector('#fig-regra-k');
      adaptH(fig, 410, 450);
      const S = { k: 6, ck: 'vx', blockk: false, seqs: 6 };
      const ACC = { vx5: [0.88, 0.74, 0.53, 0.36, 0.24], vx6: [0.89, 0.73, 0.49, 0.34, 0.23, 0.15], o5: [0.93, 0.75, 0.66, 0.58, 0.47] };
      const ctl = UI.controls(fig);
      UI.slider(ctl, { label: 'k', min: 1, max: 9, step: 1, value: 6, onInput: (v) => { S.k = v; } });
      UI.seg(ctl, { label: 'Checkpoint', options: [{ v: 'vx', label: 'Vision-Exp (n_predict=3)' }, { v: 'o', label: '0731 (n_predict=1)' }], value: 'vx', onChange: (v) => { S.ck = v; } });
      UI.toggle(ctl, { label: 'block-k hotfix', value: false, onChange: (v) => { S.blockk = v; } });
      UI.slider(ctl, { label: 'MAX_NUM_SEQS', min: 1, max: 8, step: 1, value: 6, onInput: (v) => { S.seqs = v; } });
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const narrow = w < 600;
        const k = S.k, np = S.ck === 'vx' ? 3 : 1, gamma = 5;
        const acc = S.ck === 'o' ? ACC.o5 : k <= 5 ? ACC.vx5 : ACC.vx6;
        const x0 = 14, bw = w - 28;
        // queries
        V.text(g, `block queries (k = ${k})`, x0, 18, { size: 11.5, color: C.muted });
        const n = Math.max(k, gamma), cw = Math.min(56, (bw - 10) / 9 - 6);
        for (let q = 0; q < n; q++) {
          const x = x0 + q * (cw + 6), y = 28;
          const exists = q < k;
          const beyond = q >= gamma;
          const col = q === 0 ? C.d : beyond ? C.c : C.b;
          if (exists) chip(g, x, y, cw, 26, q === 0 ? 'anchor' : 'noise', { fill: V.alpha(col, 0.3), stroke: col, size: cw < 44 ? 9 : 10.5, font: 'sans' });
          else { g.save(); g.setLineDash([3, 3]); chip(g, x, y, cw, 26, 'truncated', { fill: 'transparent', stroke: V.alpha(RED(), 0.8), color: RED(), size: cw < 44 ? 9 : 10, font: 'sans' }); g.restore(); }
        }
        const brW = gamma * (cw + 6) - 6;
        V.line(g, x0, 62, x0 + brW, 62, { color: C.b, lw: 2 });
        V.text(g, 'trained block γ = 5', x0 + brW / 2, 76, { size: 10.5, color: C.b, align: 'center' });
        if (k > gamma) V.text(g, 'beyond training', x0 + gamma * (cw + 6) + ((k - gamma) * (cw + 6)) / 2, 76, { size: 10.5, color: C.c, align: 'center' });
        // barras de aceitação
        const by = 96, bh = narrow ? 62 : 80;
        V.text(g, 'measured acceptance per position', x0, by, { size: 11, color: C.muted });
        let tau = 1, unknown = false;
        for (let q = 0; q < k; q++) {
          const x = x0 + q * (cw + 6), y = by + 10;
          const v = acc[q];
          V.box(g, x, y, cw, bh, { r: 4, fill: V.alpha(C.muted, 0.07) });
          if (v != null) { tau += v; const hh = bh * v; g.fillStyle = V.alpha(C.a, 0.4 + 0.4 * v + 0.1 * Math.sin(t * 3 + q)); V.rr(g, x, y + bh - hh, cw, hh, 4); g.fill(); V.text(g, fmt(v, 2), x + cw / 2, y + bh - hh - 4, { size: 10, font: 'mono', align: 'center', color: C.ink }); }
          else { unknown = true; V.text(g, '?', x + cw / 2, y + bh / 2 + 5, { size: 14, align: 'center', color: C.muted }); }
        }
        V.text(g, `τ ≈ 1 + Σ = ${fmt(tau, 2)}${unknown ? ' (+ unmeasured positions)' : ''}`, x0 + Math.min(k, 9) * (cw + 6) + 6 > w - 160 ? x0 : x0 + Math.min(k, 9) * (cw + 6) + 6, by + 10 + bh + (x0 + Math.min(k, 9) * (cw + 6) + 6 > w - 160 ? 16 : -bh / 2), { size: 11.5, font: 'mono', color: C.ink, maxW: 240 });
        // validador
        const vy = by + bh + 44;
        const r1 = k >= gamma;
        const divOK = S.blockk || k <= np || k % np === 0;
        const launchOK = S.blockk ? k >= 1 : (S.ck === 'vx' ? (k >= 5 && k % 3 === 0) : k >= 5);
        const rules = [
          [r1, r1 ? `k ≥ dspark_block_size (5)` : `k < 5: block truncated (silent on 0.25.2; rejected on vLLM ≥ 0.26)`],
          [divOK, S.blockk ? `divisibility: exempt (method = "dspark")` : `MTP divisibility: k ≤ ${np} or k % ${np} = 0 → ${k % np === 0 || k <= np ? 'ok' : 'k % ' + np + ' = ' + (k % np)}`],
          [launchOK, S.blockk ? 'launcher: accepts k ≥ 1 (relaxed by the hotfix)' : S.ck === 'vx' ? 'launcher: requires k ≥ 5 and k % 3 = 0' : '0731-era launcher: k ≥ 5 (historical; the current one requires k % 3 = 0)'],
        ];
        rules.forEach(([ok, txt], i) => { check(g, x0 + 8, vy + i * 22 - 4, ok, 6); V.text(g, txt, x0 + 24, vy + i * 22, { size: 11.5, color: ok ? C.ink : RED(), maxW: bw - 30 }); });
        const boots = divOK && launchOK;
        const verdict = !boots ? 'refused at boot' : !r1 ? 'boots, but the block is truncated' : k === gamma ? 'boots on the trained block ✓' : k > gamma ? `boots with ${k - gamma === 1 ? '1 query' : (k - gamma) + ' queries'} beyond training` : 'boots';
        const vcol = !boots ? RED() : !r1 ? C.d : k === gamma ? C.a : C.d;
        const lampY = vy + 3 * 22 + 12;
        V.glow(g, x0 + 10, lampY - 4, 12, vcol, 0.6 + 0.3 * Math.sin(t * 4));
        V.dot(g, x0 + 10, lampY - 4, 5, vcol);
        V.text(g, verdict, x0 + 26, lampY, { size: 13, weight: 700, color: vcol, maxW: bw - 30 });
        // captura
        const tokens = S.seqs * (k + 1), cap = Math.ceil(tokens / 8) * 8;
        V.text(g, `decode: ${S.seqs} seq × (k+1) = ${tokens} tokens → --max-cudagraph-capture-size ${cap}`, x0, lampY + 26, { size: 11.5, font: 'mono', color: C.muted, maxW: bw });
        const my = lampY + 54;
        V.text(g, 'measured (Vision-Exp, greedy, 2026-09-03): k=5 with block-k × k=6', x0, my, { size: 11, color: C.muted, maxW: bw });
        [['k=5', 56.6, 56.6], ['k=6', 55.3, 53.6]].forEach(([lb, a8, a32], i) => {
          const yy = my + 10 + i * 22, on = (i === 0 && k === 5) || (i === 1 && k === 6);
          V.text(g, lb, x0, yy + 12, { size: 11.5, font: 'mono', weight: on ? 700 : 400, color: on ? C.ink : C.muted });
          if (narrow) { V.text(g, `8K: ${fmt(a8, 1)} · 32K: ${fmt(a32, 1)} tok/s`, x0 + 40, yy + 12, { size: 11.5, font: 'mono', color: on ? C.ink : C.muted }); return; }
          const full = Math.max(60, (bw - 150) / 2);
          [[a8, '8K'], [a32, '32K']].forEach(([v, ctx], j) => {
            const bx = x0 + 40 + j * (full + 40), ww = full * ((v - 45) / 15);
            g.fillStyle = V.alpha(i === 0 ? C.a : C.b, on ? 0.85 : 0.35); V.rr(g, bx, yy, Math.max(3, ww), 16, 3); g.fill();
            V.text(g, `${ctx}: ${fmt(v, 1)} tok/s`, bx + ww + 6, yy + 12, { size: 10.5, font: 'mono', color: on ? C.ink : C.muted });
          });
        });
      });
    })();
    /* ============================================== FIG 11 — stop-rule bench (TensorFold) */
    (function () {
      const fig = root.querySelector('#fig-bancada');
      if (!fig) return;
      adaptH(fig, 480, 730, 640);
      // measured cost table (draft_sim.py, patch 0011): V(R) for R = 1..8 rows and the DFlash2 block
      const VMS = [29.2, 35.6, 42.6, 48.8, 53.6, 58.9, 64.5, 66.0], BMS = 3.08, NMAX = 7, NPOS = 240;
      const POLS = ['f7', 'fc', 'fnc', 'fcost', 'oracle'];
      const S = { pol: 'fnc', txt: 'prosa', P: 0.3, seed: 1, r: 0 };
      const pstr = () => S.P.toFixed(2).replace(/0$/, '');
      const pname = (p) => ({ f7: 'f7', fc: 'fc5:' + pstr(), fnc: 'fnc7:' + pstr(), fcost: 'fcost7:noisy', oracle: 'oracle7' }[p]);
      let dump = [];
      const sims = {};
      const lg = (p) => Math.log(p / (1 - p)), sg = (x) => 1 / (1 + Math.exp(-x));
      const cond = (a) => a.map((x, i) => (i ? Math.min(0.97, x / a[i - 1]) : x));
      function makeDump() {
        const r = V.rng(1000 + S.seed * 17 + (S.txt === 'prosa' ? 0 : 5));
        const base = cond(S.txt === 'prosa' ? PROF.glmP.a : PROF.glmS.a);
        let ease = 0;
        dump = [];
        for (let i = 0; i < NPOS; i++) {
          ease = 0.88 * ease + 0.5 * r.normal();          // local difficulty that changes slowly
          const cn = [], cc = [], m = [];
          for (let d = 0; d < NMAX; d++) {
            const p = V.clamp(sg(lg(V.clamp(base[d], 0.03, 0.97)) + 0.9 * ease), 0.01, 0.99);
            cn.push(V.clamp(sg(lg(p) + 0.35 * r.normal()), 0.01, 0.99));          // noise-aware: calibrated
            cc.push(V.clamp(sg(lg(p) + 1.5 * r.normal()), 0.01, 0.99));    // noise-free: scattered
            m.push(r() < p);
          }
          let L = 0; while (L < NMAX && m[L]) L++;
          dump.push({ cn, cc, L });
        }
      }
      const room = (i) => Math.max(0, Math.min(NMAX, NPOS - 1 - i));
      function depth(pol, i) {
        const e = dump[i], N = room(i);
        if (N <= 0) return 0;
        if (pol === 'f7') return N;
        if (pol === 'fc' || pol === 'fnc') {
          const cap = pol === 'fc' ? Math.min(5, N) : N, q = pol === 'fc' ? e.cc : e.cn;
          let prod = 1, k = 0;
          for (let d = 0; d < cap; d++) { prod *= q[d]; if (d > 0 && prod < S.P) break; k++; }
          return k;
        }
        let alive = 1, ex = 1, best = 0, bv = ex / (VMS[0] + BMS);    // fcost: argmax E[tokens|k] / (V(k+1) + B)
        for (let k = 1; k <= N; k++) { alive *= e.cn[k - 1]; ex += alive; const v = ex / (VMS[k] + BMS); if (v > bv) { bv = v; best = k; } }
        return best;
      }
      function oracleChoice() {
        const best = new Float64Array(NPOS + 1), ch = new Int8Array(NPOS + 1);
        for (let i = NPOS - 1; i >= 0; i--) {
          best[i] = Infinity;
          for (let k = 0; k <= room(i); k++) {
            const kept = Math.min(k, dump[i].L);
            const tt = VMS[k] + (k > 0 ? BMS : 0) + best[Math.min(i + kept + 1, NPOS)];
            if (tt < best[i]) { best[i] = tt; ch[i] = k; }
          }
        }
        return ch;
      }
      function replay(pol, ch) {
        const rounds = [], own = new Array(NPOS + 1).fill(null);
        let i = 0, ms = 0, rej = 0, rows = 0;
        while (i < NPOS - 1) {
          const k = pol === 'oracle' ? ch[i] : depth(pol, i);
          const kept = Math.min(k, dump[i].L);
          const block = pol === 'oracle' ? k > 0 : room(i) > 0;
          const cost = VMS[k] + (block ? BMS : 0);
          const ri = rounds.length;
          rounds.push({ i, k, kept, cost });
          for (let q = 1; q <= kept; q++) own[i + q] = { r: ri, t: 'a' };
          if (i + kept + 1 <= NPOS) own[i + kept + 1] = { r: ri, t: 's' };
          ms += cost; rej += k - kept; rows += k + 1;
          i += kept + 1;
        }
        return { rounds, own, ms, tokps: ((NPOS - 1) / ms) * 1000, tpr: (NPOS - 1) / rounds.length, rej, rows };
      }
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Rule', options: [{ v: 'f7', label: 'f7' }, { v: 'fc', label: 'fc5' }, { v: 'fnc', label: 'fnc7' }, { v: 'fcost', label: 'fcost7' }, { v: 'oracle', label: 'oracle' }], value: S.pol, onChange: (v) => { S.pol = v; S.r = 0; upd(); } });
      UI.seg(ctl, { label: 'Text', options: [{ v: 'prosa', label: 'prose' }, { v: 'estr', label: 'structured' }], value: S.txt, onChange: (v) => { S.txt = v; rebuild(); } });
      UI.slider(ctl, { label: 'threshold P (fc, fnc)', min: 0.05, max: 0.9, step: 0.05, value: S.P, fmt: (v) => fmt(v, 2), onInput: (v) => { S.P = v; recompute(); } });
      UI.button(ctl, '🎲 new dump', () => { S.seed++; rebuild(); });
      const rT = UI.readout(ctl, 'simulated tok/s', '—');
      const rR = UI.readout(ctl, 'tokens/round', '—');
      const rW = UI.readout(ctl, 'rejected rows', '—');
      function upd() {
        const s = sims[S.pol]; if (!s) return;
        rT.set(fmt(s.tokps, 1)); rR.set(fmt(s.tpr, 2)); rW.set(Math.round((100 * s.rej) / s.rows) + '%');
      }
      function recompute() { const ch = oracleChoice(); POLS.forEach((p) => { sims[p] = replay(p, ch); }); S.r = 0; upd(); }
      function rebuild() { makeDump(); recompute(); }
      rebuild();
      V.every(950, () => { const s = sims[S.pol]; if (s) S.r = (S.r + 1) % s.rounds.length; });

      V.fig(fig, (g, w, h, t, dt, _s, Hh) => {
        const C = V.C();
        const narrow = w < 640, pad = 14;
        const sim = sims[S.pol]; if (!sim) return;
        if (Hh.pointer.clicked) S.r = (S.r + 1) % sim.rounds.length;
        const R = sim.rounds[Math.min(S.r, sim.rounds.length - 1)];
        const e = dump[R.i];
        if (narrow) V.text(g, `synthetic dump · ${S.txt === 'prosa' ? 'prose' : 'structured'}`, pad, 18, { size: 11.5, color: C.muted, maxW: w * 0.46 });
        else V.text(g, `synthetic dump · ${NPOS} positions · ${S.txt === 'prosa' ? 'sampled prose' : 'structured output'}`, pad, 18, { size: 12, color: C.muted, maxW: w * 0.58 });
        V.text(g, `round ${S.r + 1}/${sim.rounds.length} · ${pname(S.pol)}`, w - pad, 18, { size: 11.5, font: 'mono', color: C.ink, align: 'right', maxW: narrow ? w * 0.5 : w * 0.4 });
        let A, K, T, B;
        if (!narrow) {
          A = { x: pad, y: 32, w: w * 0.56 - pad, h: 176 };
          K = { x: w * 0.56 + 18, y: 32, w: w * 0.44 - 18 - pad, h: 176 };
          T = { x: pad, y: 222, w: w - 2 * pad, h: 104 };
          B = { x: pad, y: 336, w: w - 2 * pad, h: h - 336 - 6 };
        } else {
          A = { x: pad, y: 30, w: w - 2 * pad, h: 170 };
          K = { x: pad, y: 210, w: w - 2 * pad, h: 136 };
          T = { x: pad, y: 356, w: w - 2 * pad, h: 104 };
          B = { x: pad, y: 470, w: w - 2 * pad, h: h - 470 - 6 };
        }
        /* --- A: the current round's chain */
        const conf = S.pol === 'fc' ? e.cc : e.cn;
        const ttl = S.pol === 'fc' ? 'round chain · noise-free confidence' : S.pol === 'oracle' ? 'round chain · the oracle already knows how many match' : S.pol === 'f7' ? 'round chain · f7 ignores the confidences' : 'round chain · noise-aware confidence';
        V.text(g, ttl, A.x, A.y + 12, { size: 12, weight: 600, color: C.ink, maxW: A.w });
        const cw = A.w / NMAX, by0 = A.y + 40, bh = A.h - 92;
        let prod = 1;
        const pts = [];
        for (let d = 0; d < NMAX; d++) {
          const x = A.x + d * cw + 4, bw = cw - 8;
          const avail = d < room(R.i), inK = d < R.k;
          V.box(g, x, by0, bw, bh, { r: 5, fill: V.alpha(C.muted, 0.07), stroke: V.alpha(C.muted, 0.25), lw: 1 });
          const q = conf[d], fh = Math.max(2, bh * q);
          const col = !avail ? C.muted : inK ? (d < R.kept ? C.a : RED()) : C.b;
          V.box(g, x + 3, by0 + bh - fh, bw - 6, fh, { r: 4, fill: V.alpha(col, inK ? 0.78 : 0.3) });
          V.text(g, fmt(q, 2), x + bw / 2, by0 + bh + 15, { size: 11, font: 'mono', align: 'center', color: C.ink, maxW: bw });
          prod *= q; pts.push([x + bw / 2, by0 + bh - bh * prod]);
          const mk = d < e.L ? '✓' : d === e.L ? '✗' : '·';
          V.text(g, 'd' + (d + 1), x + bw / 2 - 4, by0 + bh + 32, { size: 11, font: 'mono', align: 'right', color: C.muted });
          V.text(g, mk, x + bw / 2 + 2, by0 + bh + 32, { size: 12.5, align: 'left', weight: 700, color: d < e.L ? C.a : d === e.L ? RED() : C.muted });
        }
        if (S.pol !== 'f7' && S.pol !== 'oracle') {
          g.beginPath(); pts.forEach(([x, y], i) => (i ? g.lineTo(x, y) : g.moveTo(x, y))); g.strokeStyle = C.d; g.lineWidth = 2; g.stroke();
          pts.forEach(([x, y]) => V.dot(g, x, y, 3.2, C.d));
        }
        if (S.pol === 'fc' || S.pol === 'fnc') {
          const py = by0 + bh - bh * S.P;
          V.line(g, A.x, py, A.x + A.w, py, { color: V.alpha(C.ink, 0.55), dash: [5, 4], lw: 1.2 });
          V.text(g, 'P', A.x + A.w - 2, py - 4, { size: 11, font: 'mono', align: 'right', color: C.ink });
        }
        const cutX = A.x + R.k * cw;
        V.line(g, cutX, by0 - 6, cutX, by0 + bh + 4, { color: C.ink, lw: 2.2 });
        V.text(g, '✂ k = ' + R.k, V.clamp(cutX + 4, A.x, A.x + A.w - 58), by0 - 9, { size: 11.5, font: 'mono', weight: 700, color: C.ink });
        V.text(g, `✓ would match · ✗ 1st mismatch · ${e.L} of 7 match`, A.x, A.y + A.h, { size: 11, color: C.muted, maxW: A.w });

        /* --- K: cost of a round (measured table) and, for fcost, E[tokens]/ms */
        V.text(g, 'cost of a round with k drafts (ms)', K.x, K.y + 12, { size: 12, weight: 600, color: C.ink, maxW: K.w });
        const kw = K.w / 8, ky0 = K.y + 34, kh = K.h - 66, maxMs = VMS[7] + BMS;
        let alive = 1, ex = 1;
        const ev = [];
        for (let k = 0; k <= NMAX; k++) { if (k) { alive *= e.cn[k - 1]; ex += alive; } ev.push(ex / (VMS[k] + BMS)); }
        const evMax = Math.max(...ev), evMin = Math.min(...ev);
        for (let k = 0; k <= NMAX; k++) {
          const ms = VMS[k] + BMS, h2 = kh * ms / maxMs, x = K.x + k * kw;
          const on = k === R.k;
          V.box(g, x + 3, ky0 + kh - h2, kw - 6, h2, { r: 4, fill: V.alpha(on ? C.a : C.muted, on ? 0.7 : 0.2), stroke: on ? C.a : V.alpha(C.muted, 0.35), lw: 1 });
          if ((S.pol !== 'fcost' && (kw >= 30 || k % 2 === 0)) || on) V.text(g, fmt(ms, 0), x + kw / 2, ky0 + kh - h2 - 4, { size: 11, font: 'mono', align: 'center', color: on ? C.ink : C.muted, weight: on ? 700 : 400 });
          V.text(g, String(k), x + kw / 2, ky0 + kh + 15, { size: 11, font: 'mono', align: 'center', color: on ? C.ink : C.muted, weight: on ? 700 : 400 });
        }
        if (S.pol === 'fcost') {
          const yv = (v) => ky0 + kh - 8 - (kh - 26) * (evMax > evMin ? (v - evMin) / (evMax - evMin) : 0.5);
          g.beginPath(); ev.forEach((v, k) => { const x = K.x + k * kw + kw / 2; if (k) g.lineTo(x, yv(v)); else g.moveTo(x, yv(v)); }); g.strokeStyle = C.d; g.lineWidth = 2.2; g.stroke();
          ev.forEach((v, k) => V.dot(g, K.x + k * kw + kw / 2, yv(v), k === R.k ? 5 : 2.8, C.d));
        }
        V.text(g, S.pol === 'fcost' ? 'amber: E[tokens]/ms; fcost stops at the maximum' : 'measured table: 29.2…66.0 ms + 3.08 for the block', K.x, K.y + K.h, { size: 11, color: C.muted, maxW: K.w });

        /* --- T: the tape of committed tokens */
        V.text(g, 'committed tokens: the same tape for every rule', T.x, T.y + 10, { size: 12, weight: 600, color: C.ink, maxW: narrow ? T.w : T.w * 0.62 });
        const nVis = narrow ? 18 : 36, cell = T.w / nVis;
        const start = V.clamp(R.i - Math.floor(nVis * 0.3), 0, Math.max(0, NPOS - nVis));
        const ty = T.y + 46, th = 24;
        for (let c = 0; c < nVis; c++) {
          const j = start + c; if (j >= NPOS) break;
          const x = T.x + c * cell, o = sim.own[j];
          let fill = V.alpha(C.muted, 0.1), stroke = V.alpha(C.muted, 0.3);
          if (j <= R.i || (o && o.r <= S.r)) {
            const col = o ? (o.t === 'a' ? C.a : C.d) : C.muted;
            const cur = o && o.r === S.r;
            fill = V.alpha(col, cur ? 0.85 : 0.35); stroke = col;
          }
          V.box(g, x + 1, ty, cell - 2, th, { r: 4, fill, stroke, lw: 1 });
          if (j % 10 === 0) V.text(g, String(j), x + cell / 2, ty + th + 26, { size: 11, font: 'mono', align: 'center', color: C.muted });
        }
        const pc = R.i - start;
        if (pc >= 0 && pc < nVis) { const x = T.x + pc * cell + cell / 2; V.arrow(g, x, ty + th + 13, x, ty + th + 2, { color: C.ink, lw: 1.6, head: 5 }); }
        for (let q = R.kept + 1; q <= R.k; q++) {
          const c = R.i + q - start; if (c < 0 || c >= nVis) continue;
          const x = T.x + c * cell;
          V.box(g, x + 2, ty - 24, cell - 4, 19, { r: 4, fill: V.alpha(RED(), 0.16), stroke: V.alpha(RED(), 0.85), lw: 1 });
          if (cell >= 13) V.text(g, '✗', x + cell / 2, ty - 10, { size: 11, align: 'center', color: RED(), weight: 700 });
        }
        const c0 = R.i + 1 - start, c1 = R.i + R.kept + 1 - start;
        if (c1 >= 0 && c0 < nVis) {
          const x0 = T.x + Math.max(0, c0) * cell + 2, x1 = T.x + Math.min(nVis, c1 + 1) * cell - 2, yb = ty + th + 6;
          g.beginPath(); g.moveTo(x0, yb - 3); g.lineTo(x0, yb); g.lineTo(x1, yb); g.lineTo(x1, yb - 3); g.strokeStyle = C.ink; g.lineWidth = 1.6; g.stroke();
        }
        if (!narrow) {
          const lx = T.x + T.w * 0.64;
          [[C.a, 'kept draft'], [C.d, 'target sample'], [RED(), 'rejected']].forEach(([col, l], i) => {
            const xx = lx + i * (T.w * 0.12);
            V.box(g, xx, T.y + 2, 10, 10, { r: 2, fill: V.alpha(col, 0.7), stroke: col, lw: 1 });
            V.text(g, l, xx + 14, T.y + 11, { size: 11, color: C.muted, maxW: T.w * 0.12 - 18 });
          });
        }

        /* --- B: the same dump, every rule */
        V.text(g, 'same dump, every rule (simulated tok/s)', B.x, B.y + 12, { size: 12, weight: 600, color: C.ink, maxW: B.w });
        const maxT = Math.max(...POLS.map((p) => sims[p].tokps)) * 1.04;
        const lw = narrow ? 104 : 118, numW = narrow ? 62 : 236;
        const rh = V.clamp((B.h - 26) / POLS.length - 7, 14, 30);
        const barW = Math.max(40, B.w - lw - numW - 8);
        const xo = B.x + lw + barW * sims.oracle.tokps / maxT;
        V.line(g, xo, B.y + 20, xo, B.y + 24 + POLS.length * (rh + 7) - 4, { color: V.alpha(C.ink, 0.4), dash: [4, 4], lw: 1.2 });
        if (narrow) V.text(g, 'ceiling', xo, B.y + 24 + POLS.length * (rh + 7) + 10, { size: 11, align: 'center', color: C.muted });
        else V.text(g, 'ceiling', xo, B.y + 16, { size: 11, align: 'center', color: C.muted });
        POLS.forEach((p, i) => {
          const y = B.y + 24 + i * (rh + 7), s = sims[p], on = p === S.pol;
          V.text(g, pname(p), B.x + lw - 8, y + rh / 2 + 4, { size: 11.5, font: 'mono', align: 'right', color: on ? C.ink : C.muted, weight: on ? 700 : 400, maxW: lw - 10 });
          const bw2 = barW * s.tokps / maxT;
          const col = p === 'oracle' ? C.muted : on ? C.a : C.b;
          V.box(g, B.x + lw, y, bw2, rh, { r: 5, fill: V.alpha(col, on ? 0.75 : 0.32), stroke: on ? col : 'transparent', lw: 1.4 });
          const txt = narrow ? fmt(s.tokps, 1) : `${fmt(s.tokps, 1)} tok/s · ${fmt(s.tpr, 2)} tok/round · ${Math.round((100 * s.rej) / s.rows)}% rej.`;
          V.text(g, txt, B.x + lw + bw2 + 8, y + rh / 2 + 4, { size: 11, font: 'mono', color: on ? C.ink : C.muted, maxW: numW + barW - bw2 - 4 });
        });
      });
    })();

    /* ============================================== FIG 12 — the DFlash2 ring on TensorFold */
    (function () {
      const fig = root.querySelector('#fig-anel-tf');
      if (!fig) return;
      adaptH(fig, 470, 628, 640);
      const RING = 2176, WIN = 2048, TILE = 64, BLK = 8, ROWB = 10 * 1024;   // 10 KiB per row per rank
      const BACK = WIN - 1;   // the 2,048 window counts the pending token: the kernel looks 2,047 positions back (window = sliding_window − 1)
      const WINS = [8192, 32768, 131072, 262144, 524288, 1048576];
      const S = { mode: 'one', wi: 5, seg: true, P: 5000, k: 6, kept: 3, t0: null, c0: null, k4: [5, 7, 3, 6], kept4: [2, 7, 0, 4], force: false };
      const rr = V.rng(2176);
      const PH = 1.5;
      function newRound(t) { S.k = 2 + rr.int(6); S.kept = rr.int(S.k + 1); S.t0 = t; }
      function newCycle(t) { S.k4 = [0, 1, 2, 3].map(() => 2 + rr.int(6)); S.kept4 = S.k4.map((k) => rr.int(k + 1)); S.c0 = t; }
      const bytesL = (b) => V.bytes(b);
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Scene', options: [{ v: 'one', label: 'one conversation' }, { v: 'four', label: 'four conversations' }], value: 'one', onChange: (v) => { S.mode = v; grpOne.show(v === 'one'); grpFour.show(v === 'four'); } });
      const grpOne = UI.group(ctl), grpFour = UI.group(ctl);
      UI.slider(grpOne.el, { label: 'window per request', min: 0, max: WINS.length - 1, step: 1, value: S.wi, fmt: (v) => fmtInt(WINS[v]), onInput: (v) => { S.wi = v; updMem(); } });
      UI.toggle(grpFour.el, { label: 'SEG (conv boundary between blocks)', value: true, onChange: (v) => { S.seg = v; } });
      grpFour.show(false);
      UI.button(ctl, '▶ next round', () => { S.force = true; }, { primary: true });
      const rF = UI.readout(ctl, 'flat buffer / rank', '—');
      const rA = UI.readout(ctl, 'ring / rank', '—');
      function updMem() { rF.set(bytesL((WINS[S.wi] + BLK) * ROWB), 'bad'); rA.set(bytesL(RING * ROWB), 'ok'); }
      updMem();

      V.fig(fig, (g, w, h, t, dt, _s, Hh) => {
        const C = V.C();
        const narrow = w < 640, pad = 14;
        if (Hh.pointer.clicked) S.force = true;
        const ink2 = C.dark ? '#06140d' : '#ffffff';
        if (S.mode === 'one') {
          if (S.t0 == null) S.t0 = t;
          let ph = Math.floor((t - S.t0) / PH);
          if (ph >= 3 || S.force) { S.P += S.kept + 1; newRound(t); ph = 0; S.force = false; }
          const s = S.P;
          const lo = Math.floor((s - BACK) / TILE) * TILE;
          const classOf = (p) => {   // what the row of position p holds now
            if (p > s + BLK - 1) return 'free';
            if (p >= s) return ph < 2 ? 'blk' : p <= s + S.kept ? 'win' : 'old';
            if (p >= s - BACK) return 'win';
            if (p >= lo) return 'tile';
            return 'free';
          };
          /* donut: continuous arcs per role, ticks every 64 rows */
          const Rr = narrow ? Math.min(w * 0.3, 108) : Math.min(h * 0.29, 132);
          const cx = narrow ? w / 2 : pad + 14 + Rr, cy = narrow ? 36 + Rr : 42 + Rr;
          const ang = (p) => ((((p % RING) + RING) % RING) / RING) * Math.PI * 2 - Math.PI / 2;
          const arc = (p0, p1, col, al, r0 = 0.66, r1 = 1) => {
            if (p1 <= p0) return;
            const a0 = ang(p0), a1 = a0 + ((p1 - p0) / RING) * Math.PI * 2;
            g.beginPath(); g.arc(cx, cy, Rr * r1, a0, a1); g.arc(cx, cy, Rr * r0, a1, a0, true); g.closePath();
            g.fillStyle = V.alpha(col, al); g.fill();
          };
          g.beginPath(); g.arc(cx, cy, Rr, 0, Math.PI * 2); g.arc(cx, cy, Rr * 0.66, Math.PI * 2, 0, true); g.closePath(); g.fillStyle = V.alpha(C.muted, 0.12); g.fill();
          arc(lo, s - BACK, C.d, 0.75);
          arc(s - BACK, s, C.a, 0.42);
          arc(s - 256, s, C.a, 0.3);
          if (ph < 2) arc(s, s + BLK, C.b, 0.95, 0.6, 1.14);
          else { arc(s, s + S.kept + 1, C.a, 0.95, 0.6, 1.14); arc(s + S.kept + 1, s + BLK, C.muted, 0.5, 0.6, 1.14); }
          for (let q = 0; q < RING; q += TILE) {
            const a0 = ang(q);
            g.strokeStyle = V.alpha(C.ink, 0.28); g.lineWidth = 1; g.beginPath();
            g.moveTo(cx + Math.cos(a0) * Rr * 0.66, cy + Math.sin(a0) * Rr * 0.66); g.lineTo(cx + Math.cos(a0) * Rr, cy + Math.sin(a0) * Rr); g.stroke();
          }
          const hand = ang(s + 0.5);
          V.glow(g, cx + Math.cos(hand) * Rr * 1.07, cy + Math.sin(hand) * Rr * 1.07, 14, C.b, 0.45 + 0.3 * Math.sin(t * 4));
          V.text(g, '2,176 rows', cx, cy - 12, { size: 12, weight: 700, align: 'center', color: C.ink });
          V.text(g, 'pending: pos ' + fmtInt(s), cx, cy + 6, { size: 11, font: 'mono', align: 'center', color: C.muted });
          V.text(g, narrow ? 'slot ' + (s % RING) : 'slot ' + (s % RING) + ' = pos mod 2176', cx, cy + 22, { size: 11, font: 'mono', align: 'center', color: C.muted });
          /* zoomed strip around the pending token */
          const zx = narrow ? pad : cx + Rr + 34, zw = narrow ? w - 2 * pad : w - zx - pad;
          const zy = narrow ? cy + Rr + 44 : 58;
          const nb = narrow ? 4 : 10, nCells = nb + BLK + (narrow ? 2 : 4);
          const cwz = zw / nCells, chz = 30;
          const phases = ['1 · block pass: [pending, M×7] writes K/V into rows s…s+7', `2 · verification: ${S.k} drafts proposed, ${S.kept} kept`, `3 · commit: ${S.kept + 1} rows rewritten from the target's taps; the rest becomes garbage`];
          V.text(g, phases[ph], zx, zy - 18, { size: 12, weight: 600, color: C.ink, maxW: zw });
          for (let c = 0; c < nCells; c++) {
            const p = s - nb + c, x = zx + c * cwz, cl = classOf(p);
            let label = '', fillA = 0.18, col = C.muted;
            if (cl === 'win') { col = C.a; fillA = p >= s ? 0.8 : 0.45; label = p >= s ? 'tap' : 'ctx'; }
            if (cl === 'blk') { col = C.b; fillA = 0.75; label = p === s ? 'p' : 'M'; }
            if (cl === 'old') { col = C.muted; fillA = 0.22; label = 'old'; }
            if (cl === 'free') { col = C.muted; fillA = 0.08; }
            const dim = ph === 1 && p > s + S.k && p < s + BLK;
            V.box(g, x + 1, zy, cwz - 2, chz, { r: 4, fill: V.alpha(col, dim ? 0.2 : fillA), stroke: col, lw: 1 });
            if (cwz >= 16 && label) V.text(g, label, x + cwz / 2, zy + chz / 2 + 4, { size: 11, font: 'mono', align: 'center', color: fillA > 0.6 && !dim ? ink2 : C.ink, maxW: cwz - 2 });
            if (ph === 1 && p > s && p <= s + S.k) { const ok = p - s <= S.kept; check(g, x + cwz / 2, zy + chz + 13, ok, Math.min(6, cwz * 0.25)); }
            if (c % (narrow ? 4 : 5) === 0) V.text(g, String(((p % RING) + RING) % RING), x + cwz / 2, zy + chz + 33, { size: 11, font: 'mono', align: 'center', color: C.muted });
          }
          V.text(g, 'slot in the ring', zx, zy + chz + 49, { size: 11, color: C.muted });
          if (ph === 2) {
            const c = nb + S.kept + 1;
            if (c < nCells) { const x = zx + c * cwz + cwz / 2; tokChar(g, x, zy - 34 + Math.sin(t * 5) * 2, 16, C.d); if (!narrow) V.text(g, 'new pending', Math.min(x + 12, zx + zw - 92), zy - 30, { size: 11, color: C.d }); }
          }
          const ly = zy + chz + (narrow ? 70 : 76);
          const leg = [[C.a, '2,048 window (context)'], [C.d, 'extra tile of the rounded-down read'], [C.b, 'block of the draft pass'], [C.muted, 'free / garbage']];
          leg.forEach(([col, l], i) => {
            const xx = narrow ? zx + (i % 2) * (zw / 2) : zx, yy = narrow ? ly + Math.floor(i / 2) * 18 : ly + i * 19;
            V.box(g, xx, yy - 9, 11, 11, { r: 2, fill: V.alpha(col, 0.6), stroke: col, lw: 1 });
            V.text(g, l, xx + 16, yy, { size: 11, color: C.muted, maxW: narrow ? zw / 2 - 20 : zw - 20 });
          });
          const readY = narrow ? ly + 44 : ly + 4 * 19 + 4;
          const readH = V.wrap(g, 'read: from ⌊(s−2047)/64⌋·64 to s+7 → up to 2,047 + 63 + 8 = 2,118 rows; the ring: 34 tiles × 64 = 2,176', zx, readY, zw, { size: 11, font: 'mono', color: C.ink, lineH: 15 });
          /* memory: flat buffer × ring */
          const W = WINS[S.wi], flat = (W + BLK) * ROWB, ring = RING * ROWB;
          const my = narrow ? readY + readH + 16 : Math.max(cy + Rr + 30, readY + readH + 16);
          const mx = pad, mw = w - 2 * pad;
          V.text(g, `drafter context per rank · ${fmtInt(W)}-token window`, mx, my, { size: 12, weight: 600, color: C.ink, maxW: mw });
          const rows = [['flat buffer · 1 conversation', flat, RED()], ['flat buffer · 4 conversations', 4 * flat, RED()], ['ring · 4 conversations', 4 * ring, C.a]];
          const maxB = 4 * ((WINS[WINS.length - 1] + BLK) * ROWB);
          const labW = narrow ? 0 : 180, bh2 = narrow ? 13 : 15, gap = narrow ? 32 : 22;
          rows.forEach(([l, b, col], i) => {
            const y = my + 12 + i * gap;
            if (narrow) V.text(g, l, mx, y + 9, { size: 11, color: C.muted });
            else V.text(g, l, mx + labW - 8, y + bh2 - 3, { size: 11, color: C.muted, align: 'right' });
            const by = narrow ? y + 14 : y, avail = mw - labW - 84;
            const bw = Math.max(2, avail * (b / maxB));
            V.box(g, mx + labW, by, bw, bh2, { r: 4, fill: V.alpha(col, 0.55), stroke: col, lw: 1 });
            V.text(g, bytesL(b), mx + labW + bw + 6, by + bh2 - 3, { size: 11, font: 'mono', color: C.ink });
          });
        } else {
          if (S.c0 == null) S.c0 = t;
          let ph = Math.floor((t - S.c0) / PH);
          if (ph >= 3 || S.force) { newCycle(t); ph = 0; S.force = false; }
          const cols = [C.a, C.b, C.c, C.d];
          const phases = ['1 · one block pass for the four conversations', '2 · selector and top-k per conversation', '3 · stacked commit; the padding goes to the trash'];
          V.text(g, phases[ph], pad, 18, { size: 12, weight: 600, color: C.ink, maxW: w - 2 * pad });
          /* ring pool */
          const px = pad, pw = w - 2 * pad, py = 50, phh = 32;
          const trashW = pw * 0.1, slotW = (pw - trashW) / 4;
          V.text(g, 'pool per layer: [kv_heads, 4·cap + trash, head_dim]', px, py - 8, { size: 11, font: 'mono', color: C.muted, maxW: pw });
          for (let i = 0; i < 4; i++) {
            const x = px + i * slotW;
            V.box(g, x + 1, py, slotW - 2, phh, { r: 6, fill: V.alpha(cols[i], 0.2), stroke: cols[i], lw: 1.2 });
            V.text(g, narrow ? 'slot ' + i : 'conversation ' + (i + 1) + ' · slot ' + i, x + slotW / 2, py + phh / 2 + 4, { size: 11, align: 'center', color: C.ink, maxW: slotW - 6 });
            const hx = x + 6 + ((t * (0.04 + i * 0.013)) % 1) * (slotW - 14);
            g.fillStyle = cols[i]; g.fillRect(hx, py + phh - 5, 3, 4);
          }
          const tx = px + 4 * slotW;
          V.box(g, tx + 1, py, trashW - 2, phh, { r: 6, fill: V.alpha(C.muted, 0.15), stroke: V.alpha(C.muted, 0.6), lw: 1 });
          g.save(); V.rr(g, tx + 1, py, trashW - 2, phh, 6); g.clip(); g.strokeStyle = V.alpha(C.muted, 0.45); g.lineWidth = 1;
          for (let k = -phh; k < trashW; k += 7) { g.beginPath(); g.moveTo(tx + k, py + phh); g.lineTo(tx + k + phh, py); g.stroke(); }
          g.restore();
          V.text(g, 'trash', tx + trashW / 2, py + phh / 2 + 4, { size: 11, align: 'center', color: C.ink, weight: 700 });
          /* block pass: 4 × [p, M×7] */
          const bx = pad, bw = w - 2 * pad, byy = py + phh + V.clamp(h * 0.15, 58, 92), cwb = bw / 32, chb = 30;
          const a0 = ph === 0 ? 1 : 0.35;
          for (let i = 0; i < 4; i++) {
            for (let r = 0; r < BLK; r++) {
              const x = bx + (i * BLK + r) * cwb;
              V.box(g, x + 1, byy, cwb - 2, chb, { r: 3, fill: V.alpha(cols[i], (r === 0 ? 0.85 : 0.5) * a0), stroke: V.alpha(cols[i], a0), lw: 1 });
              if (cwb >= 16) V.text(g, r === 0 ? 'p' : 'M', x + cwb / 2, byy + chb / 2 + 4, { size: 11, font: 'mono', align: 'center', color: r === 0 && a0 > 0.5 ? ink2 : C.ink, alpha: Math.max(0.5, a0) });
              if (r > 0 && cwb >= 9) { const xa = x - cwb / 2 + 2, xb = x + cwb / 2 - 2; g.globalAlpha = a0; V.arrow(g, xa, byy - 5, xb, byy - 5, { color: cols[i], lw: 1.1, head: 3.5 }); g.globalAlpha = 1; }
            }
            const mxb = bx + (i * BLK + BLK / 2) * cwb, sx = px + i * slotW + slotW / 2;
            g.globalAlpha = Math.max(0.3, a0); V.arrow(g, mxb, byy - 12, sx, py + phh + 4, { color: cols[i], lw: 1.5, head: 6, dash: [5, 4] }); g.globalAlpha = 1;
            if (i > 0) {
              const xb = bx + i * BLK * cwb;
              if (S.seg) V.line(g, xb, byy - 10, xb, byy + chb + 4, { color: C.ink, lw: 2.4 });
              else { V.arrow(g, xb - cwb / 2, byy - 5, xb + cwb / 2, byy - 5, { color: RED(), lw: 2.2, head: 5 }); V.text(g, '⚡', xb, byy - 10, { size: 13, align: 'center', color: RED() }); }
            }
          }
          const wr = (txt, y, o) => V.wrap(g, txt, bx, y, bw, Object.assign({ size: 11, color: C.muted, lineH: 15 }, o));
          let y = byy + chb + 18;
          y += wr(S.seg ? 'SEG: each block starts with no predecessor, as if alone' : 'without SEG: row p of block 2 would mix in M7 of block 1 → bits ≠ solo', y, { size: 11.5, weight: 600, color: S.seg ? C.a : RED() });
          y += wr('attention: each segment reads its own slot through a (slot, position) table, with the solo kernel’s loop', y);
          /* per-conversation selector */
          const sy = y + V.clamp(h * 0.03, 8, 20), sh = 30, a1 = ph === 1 ? 1 : 0.35;
          const sw = (bw - 3 * 8) / 4;
          for (let i = 0; i < 4; i++) {
            const x = bx + i * (sw + 8);
            V.box(g, x, sy, sw, sh, { r: 6, fill: V.alpha(cols[i], 0.16 * a1 + 0.04), stroke: V.alpha(cols[i], Math.max(0.4, a1)), lw: 1.3 });
            V.text(g, narrow ? 'top-k' : 'selector + top-k → ' + S.k4[i] + ' drafts', x + sw / 2, sy + sh / 2 + 4, { size: 11, font: 'mono', align: 'center', color: C.ink, alpha: Math.max(0.55, a1), maxW: sw - 6 });
          }
          y = sy + sh + 16;
          y += wr('per conversation: cuBLAS and torch.topk may choose the kernel by row count', y);
          /* stacked commit */
          const a2 = ph === 2 ? 1 : 0.35;
          const kept = S.kept4.map((k) => k + 1), tot = kept.reduce((x, yy) => x + yy, 0), tmax = Math.max(BLK, Math.ceil(tot / BLK) * BLK);
          y += V.clamp(h * 0.03, 8, 22);
          y += wr(`commit: ${tot} kept rows → launch of ${tmax} (multiple of the block)`, y, { size: 11.5, color: C.ink, alpha: Math.max(0.6, a2) });
          const cy2 = y - 6;
          const cwc = Math.min(narrow ? 12 : 22, bw / tmax), chc = 24;
          let c = 0;
          kept.forEach((n, i) => { for (let q = 0; q < n; q++, c++) V.box(g, bx + c * cwc + 1, cy2, cwc - 2, chc, { r: 3, fill: V.alpha(cols[i], 0.7 * a2 + 0.1), stroke: V.alpha(cols[i], Math.max(0.4, a2)), lw: 1 }); });
          for (; c < tmax; c++) V.box(g, bx + c * cwc + 1, cy2, cwc - 2, chc, { r: 3, fill: V.alpha(C.muted, 0.18), stroke: V.alpha(C.muted, 0.6 * a2 + 0.2), lw: 1 });
          if (tot < tmax && bx + bw - (bx + tmax * cwc) >= 150) {
            const xe = bx + tmax * cwc;
            V.arrow(g, xe + 6, cy2 + chc / 2, xe + 30, cy2 + chc / 2, { color: C.muted, lw: 1.4, head: 6 });
            V.text(g, 'padding → trash rows', xe + 36, cy2 + chc / 2 + 4, { size: 11, color: C.muted, maxW: Math.max(40, bx + bw - xe - 40) });
          }
          y = cy2 + chc + 18;
          y += wr('each row carries its position and destination: slot·cap + (pos mod 2176)' + (narrow ? '; padding goes to the trash rows' : ''), y);
          y += wr('graphs: one block pass per number of conversations (1–4), one per row bucket for updates', y);
          wr('invariant: each conversation’s candidates have the bits of its solo drafter', Math.min(h - 26, y + V.clamp(h * 0.04, 8, 24)), { size: 12, weight: 600, color: C.a, lineH: 16 });
        }
      });
    })();
  },
});

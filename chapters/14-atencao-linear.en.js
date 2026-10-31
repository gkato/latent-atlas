LA.chapter({
  id: 'atencao-linear',
  part: 3,
  order: 14,
  title: 'Linear attention, the delta rule and hybrid models',
  navTitle: 'Linear attention & hybrids',
  subtitle: 'Swap the growing notebook (KV cache) for a fixed-size whiteboard: matrix state, forget gates, the delta rule, Gated DeltaNet and KDA, recurrent and chunkwise forms, 3:1 hybrids and what all of this demands from vLLM and TensorFold.',
  minutes: 60,
  recipes: ['glm', 'tf', 'qwen', 'ds'],
  tags: ['linear attention', 'delta rule', 'Gated DeltaNet', 'KDA', 'recurrent state', 'short conv', 'chunkwise', '3:1 hybrid', 'MambaSpec', 'mamba align', 'TensorFold', 'WY/UT'],
  goals: [
    'Derive <b>linear attention</b> from softmax attention and see it as an RNN with a \\(d_k\\times d_v\\) matrix state (associative memory)',
    'Understand <b>decay</b> (scalar or per channel), the <b>delta rule</b> and how <b>Gated DeltaNet</b> (Qwen) and <b>KDA</b> (GLM) combine them, with the real tensor shapes',
    'Tell the <b>recurrent form</b> (decode) from the <b>chunkwise form</b> (prefill) and recognize the recipes\' kernels (FLA, fused CUDA, FlashKDA)',
    'Explain why <b>3:1 hybrid</b> models lead to 1,600/3,584-token blocks, <code>align</code> checkpoints and state-freeing patches in vLLM',
    'See how <b>TensorFold</b>, the engine of the second GLM recipe, writes out KDA\'s chunkwise form in full, keeps states on a 64-token grid and undoes drafts with two parities and replay',
  ],
  prereqs: ['atencao-esparsa'],
  summary: [
    'Without softmax, \\(\\sum_s (q_t^\\top k_s) v_s = S_t^\\top q_t\\) with \\(S_t = S_{t-1} + k_t v_t^\\top\\): attention becomes an RNN with a \\(d_k\\times d_v\\) state (128×128 per head), constant cost per token, constant memory and limited capacity (non-orthogonal keys interfere).',
    'Decay gates \\(\\alpha_t\\) forget in a learned way: one scalar per head in Qwen\'s GDN (\\(\\exp(-e^{A\\_log}\\,\\mathrm{softplus}(\\cdot))\\)), per channel with a floor \\(g \\in (-5, 0)\\) in GLM\'s KDA (safe gate, required by FlashKDA).',
    'The delta rule writes only the error, \\(S \\leftarrow (I-\\beta k k^\\top)S + \\beta k v^\\top\\): a gradient step that edits the association of key \\(k\\). It needs L2-normalized \\(q, k\\) to be stable.',
    'Each layer has a short causal conv (kernel 4) with a <code>conv_state</code> of \\(K-1+\\text{num\\_spec}\\) columns, sigmoid output gates and RMSNorm. Qwen: 16 QK → 48 V; GLM: 64 heads, fused projection <code>in_proj_qkvbfg_a</code>.',
    'Prefill uses the chunkwise form (WY/UT, tensor cores): <code>chunk_gated_delta_rule</code> (Qwen, FLA Triton), <code>chunk_kda_with_fused_gate</code> or FlashKDA (GLM TP3). Decode uses recurrent kernels that keep the \\(k+1\\) states of speculative verification.',
    '3:1 hybrids (36 GDN + 12 QSA; 34 KDA + 11 sparse MLA) keep KV in only 1 of every 4 layers. DeepSeek-V4 has no linear layers; its SWA windows are the fixed cost.',
    'In vLLM the state is a <code>MambaSpec</code>: the single page size forces attention blocks the size of the state (GLM 3,584, Qwen 1,600/3,200/1,664), <code>align</code> mode only saves state at boundaries (aligned prefix cache) and the chunking and state-free patches fix checkpoints and leaks. In the GLM·TF recipe, TensorFold uses its own WY kernel with 32-row sub-chunks, keeps prompt states on a 64-token grid (with the logits row when the prompt ends on the grid, for replay) and undoes drafts with two state parities and a replay of the accepted rows.',
  ],
  quiz: [
    { q: 'Why does a linear-attention layer have constant memory per request, whatever the context?', o: ['Because it uses an FP8 KV cache', 'Because it keeps only the state \\(S = \\sum_s k_s v_s^\\top\\), one \\(d_k\\times d_v\\) matrix per head that is updated instead of growing', 'Because it only attends to the 2,048 tokens chosen by an indexer', 'Because it compresses K and V into a 512-dim latent'], a: 1, e: 'Without softmax, the weighted sum reassociates into \\(S_t^\\top q_t\\), and \\(S_t\\) always has the same size (128×128 per head in the recipes). The top-k indexer is sparse attention (it caps compute, not memory) and the 512 latent is MLA (it compresses each line, but the KV keeps growing).' },
    { q: 'With the delta rule, \\(\\lVert k\\rVert = 1\\) and \\(\\beta = 1\\), the model records “capital → Rio” and later “capital → Brasilia” with the same key. What does reading with \\(q = k\\) return?', o: ['Rio + Brasilia', 'The average of Rio and Brasilia', 'Exactly Brasilia', 'Zero, because the writes cancel out'], a: 2, e: 'The delta rule computes what the whiteboard already answers (\\(\\hat v = S^\\top k\\) = Rio) and writes only the error (Brasilia − Rio). With \\(\\beta = 1\\) and a unit key, \\(I - k k^\\top\\) erases the old association entirely. It is the plain sum that would return Rio + Brasilia.' },
    { q: 'In GLM\'s KDA, \\(g = -5\\,\\sigma(e^{A\\_log}(f + dt\\_bias))\\) and \\(\\alpha = e^{g}\\). What does the −5 guarantee?', o: ['That the state never forgets anything', 'That each step forgets at most \\(1 - e^{-5} \\approx 99.3\\%\\) of each channel (bounded gate, required by FlashKDA)', 'That the state fits in BF16', 'That the heads are divisible by 3 on TP3'], a: 1, e: 'Since \\(\\sigma \\in (0,1)\\), \\(g \\in (-5, 0)\\) and \\(\\alpha \\in (0.0067,\\ 1)\\): there is no “infinite” forgetting in one step. FlashKDA refuses to run without <code>lower_bound</code>. TP3 is solved by head padding (64 → 66) and <code>A_log</code> 64 → 66, not by the floor.' },
    { q: 'Why is GLM\'s attention block on TP2 3,584 tokens long?', o: ['Because it is <code>MAX_NUM_BATCHED_TOKENS</code> divided by 2', 'Because the allocator uses a single page size: one layer\'s KDA state (2,342,912 B) divided by MLA\'s 656 B/token gives 3,571.5, rounded up to a multiple of 256', 'Because it is the size of the drafter\'s window', 'Because the MLA kernel only accepts 3,584'], a: 1, e: 'All groups share the page size. vLLM enlarges the attention block until the MLA page fits the KDA state (FP32 recurrent [32, 128, 128] + BF16 conv [12288, 10]) and rounds up to the 256 alignment required by the indexer: 3,584 × 656 = 2,351,104 B, the <code>mla_page</code> in the log. On TP3 the same computation gives 2,560.' },
    { q: 'Why does the prefill of the linear layers use the chunkwise form rather than the token-by-token recurrence?', o: ['Because the chunkwise form gives a different, more accurate result', 'Because the recurrence would be sequential in \\(n\\) and without tensor cores; chunkwise, the work inside each block becomes parallel matrix multiplications and the state only passes at the boundaries', 'Because the recurrence doesn\'t work with the delta rule', 'Because prefill doesn\'t need the state'], a: 1, e: 'Both forms compute the same thing. With \\(C = 64\\), 131,072 tokens become 2,048 sequential stages instead of 131,072, with the tensor cores busy. The delta rule comes in via the WY/UT representation (a \\(C\\times C\\) triangular inverse). In decode there is only one token, and the recurrent form is the natural one.' },
    { q: 'What changes when the Qwen recipe passes <code>--mamba-ssm-cache-dtype bfloat16</code>?', o: ['The KV of the 12 QSA layers becomes BF16', 'The GDN state and the Mamba page are halved (block 3,200 → 1,664 with FP8 KV), which is only viable because the fused kernel accepts BF16 (<code>FUSED_GDN_STATE_DTYPES</code>)', 'The model stops using prefix caching', 'Nothing: the checkpoint already asks for BF16'], a: 1, e: 'The checkpoint declares <code>mamba_ssm_dtype: float32</code>. The flag overrides only the recurrent state. Half the bytes read and written per step and a smaller page: a smaller block and a less coarse prefix cache. On the single-GPU recipe it measured +8.5% at 8 streams, with no isolated measurement on the 2-node kit. The risk is the precision of a state that accumulates up to 1M updates. <code>--kv-cache-dtype</code> doesn\'t touch the state.' },
    { q: 'In the GLM·TF recipe, why does every prompt chunk of the KDA layers start on a multiple of the 64-token grid?', o: ['Because vLLM requires 64-token pages', 'Because the bits of each row of the chunkwise kernel depend only on its 32-row sub-chunk (at absolute positions) and on the state entering it; with aligned boundaries, a prompt resumed from a kept state gives exactly the bits of a prefill from scratch', 'Because the chunkwise form only works with prompts whose length is a multiple of 64', 'Because decode processes 64 tokens at a time'], a: 1, e: 'The <code>kda_chunk.cu</code> kernel cuts the sequence into 32-row sub-chunks at absolute positions. If a resumed prefill started off the grid, the same sub-chunk would be computed from a different incoming state and the bits would change, breaking “resumed = fresh”. That is why the kept state sits at \\(\\lfloor n/64\\rfloor\\cdot 64\\), and, when the prompt ends on the grid, the end snapshot also carries the logits row, which an identical resend reuses with no prefill. Prompts of any length work (the request prefills up to the grid and resumes from there), TensorFold does not use vLLM pages and decode stays on the serial chain.' },
  ],
  mount(root, LA) {
    const V = LA.viz, UI = LA.ui;
    const nf = (x, d = 0) => (+x).toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d });
    const fB = (b) => { const u = ['B', 'KiB', 'MiB', 'GiB', 'TiB']; let i = 0; b = +b; while (Math.abs(b) >= 1024 && i < u.length - 1) { b /= 1024; i++; } return (i ? nf(b, b < 10 ? 2 : 1) : nf(b, 0)) + ' ' + u[i]; };
    const fD = (b) => { const u = ['B', 'kB', 'MB', 'GB', 'TB']; let i = 0; b = +b; while (Math.abs(b) >= 1000 && i < u.length - 1) { b /= 1000; i++; } return (i ? nf(b, b < 10 ? 2 : 1) : nf(b, 0)) + ' ' + u[i]; };
    let fcache = {};
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(() => { fcache = {}; });
    const font = (s, f = 'sans', w = 400) => fcache[s + f + w] || (fcache[s + f + w] = V.font(s, f, w));
    const tw = (g, s, size = 12, f = 'sans', w = 400) => { g.font = font(size, f, w); return g.measureText(String(s)).width; };
    function wrap(g, s, maxW, size = 12, f = 'sans', w = 400) {
      g.font = font(size, f, w);
      const words = String(s).split(' '); const lines = []; let cur = '';
      for (const wd of words) { const tst = cur ? cur + ' ' + wd : wd; if (g.measureText(tst).width > maxW && cur) { lines.push(cur); cur = wd; } else cur = tst; }
      if (cur) lines.push(cur);
      return lines;
    }
    function pill(g, x, y, s, col, o = {}) {
      const size = o.size || 11, f = o.font || 'mono', wt = o.weight || 600;
      const wd = tw(g, s, size, f, wt) + 14, ht = size + 9;
      const xx = o.align === 'left' ? x : o.align === 'right' ? x - wd : x - wd / 2;
      V.box(g, xx, y - ht / 2, wd, ht, { r: ht / 2, fill: V.alpha(col, o.a == null ? 0.18 : o.a), stroke: V.alpha(col, 0.8), lw: 1 });
      V.text(g, s, xx + wd / 2, y + 0.5, { size, font: f, weight: wt, color: o.color || V.C().ink, align: 'center', base: 'middle' });
      return wd;
    }
    function autoH(fig, desk, mob, bp = 560) {
      const st = fig.querySelector('.la-fig-stage');
      const set = () => { const hh = (fig.clientWidth < bp ? mob : desk) + 'px'; if (st.style.height !== hh) st.style.height = hh; };
      set();
      const ro = new ResizeObserver(set); ro.observe(fig);
      if (LA.onCleanup) LA.onCleanup(() => ro.disconnect());
    }
    const fmtv = (v) => { const r = Math.round(v * 10) / 10; if (Math.abs(r) < 0.05) return '0'; return (Number.isInteger(r) ? String(r) : r.toFixed(1)).replace('-', '−'); };
    function drawMat(g, x, y, M, cs, o = {}) {
      const C = V.C(); const gap = o.gap == null ? 3 : o.gap; const R = M.length, K = M[0].length; const mx = o.max || 3;
      for (let i = 0; i < R; i++) for (let j = 0; j < K; j++) {
        const v = M[i][j]; const xx = x + j * (cs + gap), yy = y + i * (cs + gap);
        const hide = o.mask ? o.mask(i, j) : false;
        const tt = V.clamp(v / mx, -1, 1);
        g.fillStyle = hide ? V.alpha(C.muted, 0.08) : (o.colorFn ? o.colorFn(v, i, j) : V.cmap(tt, 'div'));
        V.rr(g, xx, yy, cs, cs, Math.min(5, cs / 5)); g.fill();
        g.strokeStyle = V.alpha(C.muted, 0.28); g.lineWidth = 1; g.stroke();
        if (!hide && o.values !== false && cs >= 20) {
          const light = !o.colorFn && Math.abs(tt) > 0.55;
          V.text(g, fmtv(v), xx + cs / 2, yy + cs / 2 + 0.5, { size: o.fs || Math.max(11, Math.min(13, cs * 0.4)), font: 'mono', align: 'center', base: 'middle', color: light ? (C.dark ? '#00210F' : '#fff') : C.ink });
        }
      }
      const W = K * cs + (K - 1) * gap, Ht = R * cs + (R - 1) * gap;
      (o.hi || []).forEach(([i, j, a]) => { g.strokeStyle = V.alpha(o.hiColor || C.d, a == null ? 1 : a); g.lineWidth = 2.5; V.rr(g, x + j * (cs + gap) - 2, y + i * (cs + gap) - 2, cs + 4, cs + 4, 6); g.stroke(); });
      if (o.hiRow != null) { g.strokeStyle = o.hiColor || C.d; g.lineWidth = 2.5; V.rr(g, x - 3, y + o.hiRow * (cs + gap) - 3, W + 6, cs + 6, 7); g.stroke(); }
      if (o.label) V.text(g, o.label, x + W / 2, y - 9, { size: 12, font: 'mono', align: 'center', color: o.labelColor || C.muted });
      return { w: W, h: Ht };
    }
    const safe = (name, fn) => { try { fn(); } catch (e) { console.error('[atencao-linear] ' + name, e); } };

    /* =====================================================================
       Figure 1 — notebook (KV) × whiteboard (state)
       ===================================================================== */
    safe('fig-caderno', () => {
      const fig = root.querySelector('#fig-caderno');
      const ctl = UI.controls(fig);
      const SC = {
        qwenA: { kv: 1024, st: 1634304, blk: 1600, kvName: 'QSA · KV BF16', stName: 'GDN · FP32 state', rec: 'qwen' },
        qwenB: { kv: 512, st: 847872, blk: 1664, kvName: 'QSA · KV FP8', stName: 'GDN · BF16 state', rec: 'qwen' },
        glm: { kv: 656, st: 2342912, blk: 3584, kvName: 'MLA · fp8_ds_mla', stName: 'KDA · FP32 state', rec: 'glm' },
      };
      const S = { sc: 'qwenB', lg: 15 };
      UI.select(ctl, { label: 'KV / state', options: [{ v: 'qwenB', label: 'Qwen · FP8 / BF16 (default)' }, { v: 'qwenA', label: 'Qwen · BF16 / FP32' }, { v: 'glm', label: 'GLM · fp8_ds_mla / FP32' }], value: S.sc, onChange: (v) => { S.sc = v; } });
      UI.slider(ctl, { label: 'Context', min: 4, max: 20, step: 0.25, value: S.lg, fmt: (v) => nf(Math.round(2 ** v)) + ' tok', onInput: (v) => { S.lg = v; } });
      const rK = UI.readout(ctl, 'notebook', '—'), rS = UI.readout(ctl, 'whiteboard', '—'), rR = UI.readout(ctl, 'notebook ÷ board', '—');
      let lastKey = '';
      const words = ['the', 'whiteboard', 'keeps', 'a', 'summary', 'of', 'all', 'that', 'went', 'by', 'and', 'the', 'notebook', 'gains', 'a', 'line', 'per', 'token', 'nonstop'];
      const P = 0.8;
      V.fig(fig, (g, w, h, t) => {
        if (w < 80 || h < 60) return;
        const C = V.C(); const sc = SC[S.sc]; const n = Math.round(2 ** S.lg);
        const kvB = n * sc.kv, stB = sc.st, ratio = kvB / stB;
        const key = S.sc + ':' + n;
        if (key !== lastKey) { lastKey = key; rK.set(fD(kvB)); rS.set(fD(stB)); rR.set((ratio >= 10 ? nf(ratio, 0) : nf(ratio, 2)) + '×'); }
        const narrow = w < 560;
        const topH = narrow ? 172 : 192;
        const recCol = sc.rec === 'glm' ? C.glm : C.qwen;
        // ---- token source
        const cx = w / 2, sy = 22;
        V.box(g, cx - 46, sy - 13, 92, 26, { r: 13, fill: V.alpha(C.d, 0.16), stroke: C.d, lw: 1.2 });
        V.text(g, 'tokens', cx, sy + 1, { size: 12, align: 'center', base: 'middle', color: C.ink, weight: 600 });
        // ---- notebook
        const nbX = 14, nbW = narrow ? Math.min(140, w * 0.38) : Math.min(230, w * 0.3), nbY = 50, nbHmax = topH - nbY - 22;
        const frac = (S.lg - 4) / 16; const nbH = Math.max(18, nbHmax * (0.12 + 0.88 * frac));
        V.text(g, narrow ? 'notebook (KV)' : 'notebook · KV cache', nbX, nbY - 8, { size: 11.5, color: C.c, weight: 600 });
        V.box(g, nbX, nbY, nbW, nbH, { r: 6, fill: V.alpha(C.c, 0.08), stroke: V.alpha(C.c, 0.7), lw: 1.2 });
        g.save(); g.beginPath(); g.rect(nbX + 2, nbY + 2, nbW - 4, nbH - 4); g.clip();
        for (let yy = nbY + 7; yy < nbY + nbH - 2; yy += 6) V.line(g, nbX + 16, yy, nbX + nbW - 6, yy, { color: V.alpha(C.c, 0.35), lw: 1 });
        g.restore();
        V.line(g, nbX + 11, nbY + 3, nbX + 11, nbY + nbH - 3, { color: V.alpha(C.c, 0.6), lw: 1 });
        V.text(g, nf(n) + ' lines', nbX + nbW / 2, nbY + nbH + 13, { size: 11.5, font: 'mono', align: 'center', color: C.ink });
        // ---- whiteboard
        const N = 8; const lsWmax = narrow ? Math.min(118, w * 0.34) : 150;
        const cell = Math.min((lsWmax - (N - 1) * 2) / N, (topH - nbY - 24) / N - 2); const lsW = N * cell + (N - 1) * 2;
        const lsX = w - 14 - lsW, lsY = nbY;
        V.text(g, narrow ? 'whiteboard (state)' : 'whiteboard · 128×128 state/head', lsX + lsW, lsY - 8, { size: 11.5, color: C.a, weight: 600, align: 'right' });
        V.box(g, lsX - 5, lsY - 3, lsW + 10, N * (cell + 2) + 4, { r: 7, fill: V.alpha(C.a, 0.06), stroke: V.alpha(C.a, 0.7), lw: 1.2 });
        const tk = Math.floor(t / P), pf = (t / P) % 1;
        const flash = (tk * 11) % (N * N);
        for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
          const v = Math.sin(i * 1.3 + j * 0.7 + tk * 0.9) * 0.55 + Math.sin(i * 0.4 - j * 1.1 + tk * 0.35) * 0.35;
          g.fillStyle = V.cmap(v, 'div'); V.rr(g, lsX + j * (cell + 2), lsY + i * (cell + 2), cell, cell, 2.5); g.fill();
          if (i * N + j === flash && pf < 0.35) { g.strokeStyle = C.d; g.lineWidth = 2; V.rr(g, lsX + j * (cell + 2) - 1, lsY + i * (cell + 2) - 1, cell + 2, cell + 2, 3); g.stroke(); }
        }
        V.text(g, 'always ' + fD(stB), lsX + lsW / 2, lsY + N * (cell + 2) + 15, { size: 11.5, font: 'mono', align: 'center', color: C.ink });
        // ---- tokens flying to both sides
        for (let k = tk - 1; k <= tk; k++) {
          if (k < 0) continue;
          const p = V.ease.inOut(V.clamp((t - k * P) / (P * 1.4), 0, 1));
          const wd = words[k % words.length];
          const a1 = V.along(cx, sy + 14, nbX + nbW / 2, nbY + nbH - 8, p), a2 = V.along(cx, sy + 14, lsX + lsW / 2, lsY + (N * (cell + 2)) / 2, p);
          if (p < 0.98) {
            pill(g, a1[0], a1[1], wd, C.c, { size: 10.5, a: 0.25 });
            pill(g, a2[0], a2[1], wd, C.a, { size: 10.5, a: 0.25 });
          }
        }
        // ---- log-log chart
        const x0 = narrow ? 46 : 70, x1 = w - 16, cy0 = topH + 24, cy1 = h - 30;
        const X = (nn) => x0 + ((Math.log2(nn) - 4) / 16) * (x1 - x0);
        const Y = (b) => cy1 - ((Math.log10(b) - 3) / (9.5 - 3)) * (cy1 - cy0);
        const nx = stB / sc.kv;
        // zones
        g.fillStyle = V.alpha(C.c, 0.06); g.fillRect(x0, cy0, Math.max(0, X(nx) - x0), cy1 - cy0);
        g.fillStyle = V.alpha(C.a, 0.07); g.fillRect(X(nx), cy0, x1 - X(nx), cy1 - cy0);
        [[16, '16'], [256, '256'], [4096, '4k'], [65536, '65k'], [1048576, '1M']].forEach(([v, l]) => { V.line(g, X(v), cy0, X(v), cy1, { color: C.grid, lw: 1 }); V.text(g, l, X(v), cy1 + 15, { size: 11, font: 'mono', align: 'center', color: C.muted }); });
        [[1e3, '1 kB'], [1e6, '1 MB'], [1e9, '1 GB']].forEach(([v, l]) => { V.line(g, x0, Y(v), x1, Y(v), { color: C.grid, lw: 1 }); V.text(g, l, x0 - 6, Y(v) + 4, { size: 11, font: 'mono', align: 'right', color: C.muted }); });
        V.text(g, 'context tokens (log)', x1, cy1 + 27, { size: 11, align: 'right', color: C.muted });
        V.text(g, narrow ? 'bytes · 1 layer · 1 req.' : 'bytes per layer, per request, per GPU (log)', x0, cy0 - 8, { size: 11, color: C.muted });
        // lines
        g.lineWidth = 2.6; g.strokeStyle = C.c; g.beginPath(); g.moveTo(X(16), Y(16 * sc.kv)); g.lineTo(X(1048576), Y(1048576 * sc.kv)); g.stroke();
        g.strokeStyle = C.a; g.beginPath(); g.moveTo(x0, Y(stB)); g.lineTo(x1, Y(stB)); g.stroke();
        const lbKV = 'notebook · ' + (narrow ? sc.kvName.replace('KV ', '').replace(' · fp8_ds_mla', ' FP8') : sc.kvName), lbST = 'whiteboard · ' + (narrow ? sc.stName.replace(' state', '') : sc.stName);
        V.text(g, lbKV, X(24), Y(24 * sc.kv) + 20, { size: 11.5, color: C.c, weight: 600 });
        V.text(g, lbST, x0 + 6, Y(stB) - 8, { size: 11.5, color: C.a, weight: 600 });
        // crossover
        V.line(g, X(nx), cy0, X(nx), cy1, { color: recCol, lw: 1.6, dash: [5, 4] });
        V.dot(g, X(nx), Y(stB), 5, recCol);
        const tx = X(nx) + 8, lab1 = 'crossover ≈ ' + nf(nx) + ' tok', lab2 = 'vLLM block: ' + nf(sc.blk);
        const alignR = tx + tw(g, lab1, 11.5, 'sans', 600) > x1;
        V.text(g, lab1, alignR ? X(nx) - 8 : tx, Y(stB) + 22, { size: 11.5, color: C.ink, weight: 600, align: alignR ? 'right' : 'left' });
        V.text(g, lab2, alignR ? X(nx) - 8 : tx, Y(stB) + 37, { size: 11.5, font: 'mono', color: recCol, align: alignR ? 'right' : 'left' });
        // current context
        const xn = X(n);
        V.line(g, xn, cy0, xn, cy1, { color: C.d, lw: 1.4 });
        V.dot(g, xn, Y(kvB), 5, C.c); V.dot(g, xn, Y(stB), 5, C.a);
        const winner = kvB > stB ? 'the whiteboard is ' + (ratio >= 10 ? nf(ratio, 0) : nf(ratio, 1)) + '× smaller' : 'the notebook is ' + (1 / ratio >= 10 ? nf(1 / ratio, 0) : nf(1 / ratio, 1)) + '× smaller';
        const wl = tw(g, winner, 11.5, 'sans', 600) + 16;
        const wx = V.clamp(xn - wl / 2, x0 + 2, x1 - wl - 2);
        V.box(g, wx, cy0 + 4, wl, 22, { r: 11, fill: V.alpha(C.d, 0.18), stroke: C.d, lw: 1 });
        V.text(g, winner, wx + wl / 2, cy0 + 15.5, { size: 11.5, align: 'center', base: 'middle', color: C.ink, weight: 600 });
      });
    });

    /* =====================================================================
       Figure 2 — stepper: reassociate, write, read
       ===================================================================== */
    safe('fig-rnn', () => {
      const fig = root.querySelector('#fig-rnn');
      autoH(fig, 410, 420);
      const k1 = [1, 0, 0, 0], k2 = [0, 1, 0, 0], k3 = [0.6, 0, 0.8, 0];
      const v1 = [2, 0, -1, 1], v2 = [0, 3, 1, 0], v3 = [-1, 1, 0, 2];
      const outer = (k, v) => k.map((a) => v.map((b) => a * b));
      const add = (A, B) => A.map((r, i) => r.map((x, j) => x + B[i][j]));
      const Z = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]];
      const O1 = outer(k1, v1), O2 = outer(k2, v2), O3 = outer(k3, v3);
      const S1 = O1, S2 = add(S1, O2), S3 = add(S2, O3);
      const read = (S, q) => [0, 1, 2, 3].map((j) => S.reduce((s, row, i) => s + row[j] * q[i], 0));
      const o1 = read(S3, k1);
      const col = (v) => v.map((x) => [x]);
      const row = (v) => [v];
      const scores = V.randMatrix(9, 9, 5).map((r) => r.map((x) => Math.abs(x)));
      const steps = [
        { t: 'Softmax attention keeps everything', d: 'Each token compares its query with the keys of <b>all</b> the previous ones: the score table \\(QK^\\top\\) is \\(n\\times n\\) (the causal triangle) and the KV cache stores K and V for the \\(n\\) tokens. Cost per token \\(O(n\\,d)\\), memory \\(O(n)\\).' },
        { t: 'Remove the softmax and reassociate', d: 'Without softmax in the middle, \\((QK^\\top)V = Q(K^\\top V)\\). Watch \\(n\\) vary: the \\(n\\times n\\) table at the top grows, but \\(K^\\top V\\), at the bottom, is always \\(d\\times d\\). The “whiteboard” doesn\'t depend on the length.' },
        { t: 'Write = outer product', d: 'Token 1 writes \\(k_1 v_1^\\top\\): the column \\(k_1 = e_1\\) times the row \\(v_1 = [2, 0, -1, 1]\\). A rank-1 matrix. Since \\(k_1\\) only has its 1st component, the association occupies only the 1st row.' },
        { t: 'Accumulate the writes', d: '\\(S_t = S_{t-1} + k_t v_t^\\top\\). Tokens 2 and 3 add \\(k_2 v_2^\\top\\) and \\(k_3 v_3^\\top\\). Since \\(k_3 = 0.6\\,e_1 + 0.8\\,e_3\\), the write of \\(v_3\\) splits between rows 1 and 3, and part of it lands on the row of \\(k_1\\).' },
        { t: 'Read = multiply by the query', d: 'The read is \\(S^\\top q\\). Asking for \\(q = k_1\\): \\(S^\\top k_1 = v_1 + (k_3\\cdot k_1)\\,v_3 = [1.4,\\ 0.6,\\ -1,\\ 2.2]\\) instead of \\(v_1 = [2, 0, -1, 1]\\). The excess \\(0.6\\,v_3\\) is the interference between non-orthogonal keys.' },
        { t: 'It is an RNN', d: 'The state passes from token to token, \\(S_0 \\to S_1 \\to S_2 \\to S_3\\), with a fixed memory of \\(d_k\\times d_v\\) per head and one step per token. In the recipes: 128×128 per head, in FP32.' },
      ];
      const st = UI.stepper(fig, steps, { autoplay: 8000 });
      V.fig(fig, (g, w, h, t) => {
        if (w < 80 || h < 60) return;
        const C = V.C(); const el = st.elapsed(); const narrow = w < 560;
        const i = st.i;
        // steps 3–6 take less height: center the [top, bottom] block vertically
        const TB = narrow ? { 2: [7, 300], 3: [59, 295], 4: [35, 361], 5: [70, 250] } : { 2: [11, 284], 3: [59, 295], 4: [35, 335], 5: [74, 254] };
        const dy = TB[i] ? Math.max(0, Math.round((h - TB[i][0] - TB[i][1]) / 2)) : 0;
        g.save(); g.translate(0, dy);
        if (i === 0) {
          const n = 6 + Math.floor(V.pingpong(el, 6) * 3.99);
          const cs = Math.min(narrow ? 21 : 30, (narrow ? w - 40 : w * 0.5) / 9 - 3);
          const mx = narrow ? 20 : 40, my = 44;
          const rows = Math.min(n, Math.floor(el * 2.2) + 1);
          V.text(g, 'q·k scores (n×n, causal) · n = ' + n, mx, my - 16, { size: 12, color: C.muted, font: 'mono' });
          for (let a = 0; a < n; a++) for (let b = 0; b < n; b++) {
            const x = mx + b * (cs + 3), y = my + a * (cs + 3);
            if (b > a) { g.fillStyle = V.alpha(C.muted, 0.06); V.rr(g, x, y, cs, cs, 3); g.fill(); continue; }
            const on = a < rows;
            g.fillStyle = on ? V.cmap(0.25 + 0.7 * Math.min(1, scores[a][b] / 2), 'heat') : V.alpha(C.muted, 0.12);
            V.rr(g, x, y, cs, cs, 3); g.fill();
          }
          const kx = narrow ? mx : mx + 9 * (cs + 3) + 40, ky = narrow ? my + n * (cs + 3) + 26 : my;
          V.text(g, 'KV cache: ' + n + ' lines', kx, ky - 16 + (narrow ? 0 : 0), { size: 12, color: C.c, font: 'mono' });
          const kw = narrow ? (w - 2 * mx) / 2 - 8 : Math.min(160, w - kx - 20);
          for (let a = 0; a < n; a++) {
            const y = ky + a * (narrow ? 11 : cs + 3);
            const hh = narrow ? 8 : cs * 0.7;
            V.box(g, kx, y, kw, hh, { r: 3, fill: V.alpha(C.b, a < rows ? 0.45 : 0.12) });
            V.box(g, kx + kw + 8, y, kw, hh, { r: 3, fill: V.alpha(C.a, a < rows ? 0.45 : 0.12) });
          }
          if (!narrow) { V.text(g, 'K', kx + kw / 2, ky + n * (cs + 3) + 12, { size: 12, font: 'mono', align: 'center', color: C.b }); V.text(g, 'V', kx + kw * 1.5 + 8, ky + n * (cs + 3) + 12, { size: 12, font: 'mono', align: 'center', color: C.a }); }
          V.text(g, 'each new token: +1 notebook line and +n scores', w / 2, h - 16, { size: 12, align: 'center', color: C.ink, weight: 500 });
        } else if (i === 1) {
          const n = 5 + Math.floor(V.pingpong(el, 5) * 4.99);
          const cs = narrow ? 8 : 11, gp = 2;
          const blk = (x, y, r, c, color, label) => {
            for (let a = 0; a < r; a++) for (let b = 0; b < c; b++) { g.fillStyle = V.alpha(color, 0.55); V.rr(g, x + b * (cs + gp), y + a * (cs + gp), cs, cs, 2); g.fill(); }
            if (label) V.text(g, label, x + (c * (cs + gp)) / 2, y + r * (cs + gp) + 13, { size: 11, font: 'mono', align: 'center', color: C.muted });
            return c * (cs + gp);
          };
          const sym = (x, y, s) => V.text(g, s, x, y, { size: 16, align: 'center', base: 'middle', color: C.ink, weight: 600 });
          const d = 4;
          let y = 50, x = narrow ? 14 : 40;
          V.text(g, 'with softmax:  softmax(Q Kᵀ) · V', x, y - 20, { size: 12.5, color: C.c, weight: 600, font: 'mono' });
          let xx = x; xx += blk(xx, y, n, d, C.b, 'Q [n×d]'); sym(xx + 10, y + (n * (cs + gp)) / 2, '×'); xx += 20;
          xx += blk(xx, y, d, n, C.b, 'Kᵀ [d×n]'); sym(xx + 12, y + (n * (cs + gp)) / 2, '→'); xx += 24;
          xx += blk(xx, y, n, n, C.c, 'n×n = ' + n * n); sym(xx + 10, y + (n * (cs + gp)) / 2, '×'); xx += 20;
          blk(xx, y, n, d, C.a, 'V [n×d]');
          const y2 = y + 9 * (cs + gp) + 58;
          V.text(g, 'without softmax:  Q · (Kᵀ V)', x, y2 - 20, { size: 12.5, color: C.a, weight: 600, font: 'mono' });
          xx = x; xx += blk(xx, y2, d, n, C.b, 'Kᵀ [d×n]'); sym(xx + 10, y2 + (d * (cs + gp)) / 2, '×'); xx += 20;
          xx += blk(xx, y2, n, d, C.a, 'V [n×d]'); sym(xx + 12, y2 + (d * (cs + gp)) / 2, '→'); xx += 24;
          const gl = 0.5 + 0.5 * Math.sin(t * 3);
          V.glow(g, xx + (d * (cs + gp)) / 2, y2 + (d * (cs + gp)) / 2, 34, C.a, 0.35 + 0.25 * gl);
          blk(xx, y2, d, d, C.a, 'S = d×d = ' + d * d);
          const bl = wrap(g, 'n = ' + n + ': the top table has ' + n * n + ' scores; the whiteboard below still has ' + d * d + ' numbers', w - 24, 12);
          bl.forEach((ln, q) => V.text(g, ln, w / 2, h - 16 - (bl.length - 1 - q) * 16, { size: 12, align: 'center', color: C.ink }));
        } else if (i === 2) {
          const cs = narrow ? Math.min(44, (w - 60) / 5.3) : 40, gap = 4;
          const W4 = 4 * cs + 3 * gap;
          const total = cs + 22 + W4;
          const x0 = (w - total) / 2 - (narrow ? 0 : 30), y0 = 88;
          drawMat(g, x0, y0, col(k1), cs, { label: 'k₁', labelColor: C.b });
          const sx = x0 + cs + 22;
          drawMat(g, sx, y0 - cs - 16, row(v1), cs, { label: 'v₁ᵀ', labelColor: C.a });
          const shown = Math.min(16, Math.floor(el * 5));
          drawMat(g, sx, y0, O1, cs, { mask: (a, b) => a * 4 + b >= shown, hi: shown > 0 && shown <= 16 ? [[Math.floor((shown - 1) / 4), (shown - 1) % 4]] : [] });
          V.text(g, 'k₁ v₁ᵀ  (rank 1)', sx + W4 / 2, y0 + W4 + 20, { size: 12.5, font: 'mono', align: 'center', color: C.ink });
          if (!narrow) V.text(g, 'cell (i, j) = k₁[i] · v₁[j]', sx + W4 + 24, y0 + W4 / 2, { size: 12, color: C.muted });
        } else if (i === 3) {
          const ph = Math.min(2, Math.floor(el / 1.9));
          const Ss = [S1, S2, S3][ph], Os = [O1, O2, O3][ph], lab = ['k₁v₁ᵀ', 'k₂v₂ᵀ', 'k₃v₃ᵀ'][ph];
          const cs = narrow ? Math.min(38, (w - 90) / 6.3) : 38, gap = 3, W4 = 4 * cs + 3 * gap;
          const csS = narrow ? Math.min(24, cs * 0.63) : 24, WS = 4 * csS + 9;
          const totalW = WS + 40 + W4;
          const x0 = (w - totalW) / 2, y0 = 80;
          drawMat(g, x0, y0 + (W4 - WS) / 2, Os, csS, { label: '+ ' + lab, labelColor: C.d, values: false });
          V.arrow(g, x0 + WS + 8, y0 + W4 / 2, x0 + WS + 32, y0 + W4 / 2, { color: C.d });
          const changed = []; for (let a = 0; a < 4; a++) for (let b = 0; b < 4; b++) if (Math.abs(Os[a][b]) > 1e-9) changed.push([a, b, 0.4 + 0.6 * (1 - V.clamp(((el % 1.9) / 1.9), 0, 1))]);
          drawMat(g, x0 + WS + 40, y0, Ss, cs, { label: 'S' + ['₁', '₂', '₃'][ph], labelColor: C.a, hi: changed });
          V.text(g, 'S' + ['₁', '₂', '₃'][ph] + ' = ' + ['k₁v₁ᵀ', 'k₁v₁ᵀ + k₂v₂ᵀ', 'k₁v₁ᵀ + k₂v₂ᵀ + k₃v₃ᵀ'][ph], w / 2, y0 + W4 + 30, { size: 12.5, font: 'mono', align: 'center', color: C.ink });
          V.text(g, ph === 2 ? 'k₃ = 0.6·e₁ + 0.8·e₃: v₃ lands on rows 1 and 3' : 'orthogonal key: lands on a single row', w / 2, y0 + W4 + 50, { size: 12, align: 'center', color: C.muted });
        } else if (i === 4) {
          const cs = narrow ? Math.min(38, (w - 60) / 5.5) : 38, gap = 3, W4 = 4 * cs + 3 * gap;
          const totalW = cs + 24 + W4;
          const x0 = (w - totalW) / 2, y0 = 56;
          drawMat(g, x0, y0, col(k1), cs, { label: 'q = k₁', labelColor: C.b });
          drawMat(g, x0 + cs + 24, y0, S3, cs, { label: 'S₃', labelColor: C.a, hiRow: 0, hiColor: C.b });
          const ry = y0 + W4 + 30, rx = (w - W4) / 2 + (narrow ? 22 : 40);
          const lx = rx - 10;
          V.text(g, 'Sᵀq', lx, ry + cs / 2 + 4, { size: 12.5, font: 'mono', align: 'right', color: C.ink, weight: 600 });
          drawMat(g, rx, ry, row(o1), cs, {});
          V.text(g, 'v₁', lx, ry + cs + 8 + cs / 2 + 4, { size: 12.5, font: 'mono', align: 'right', color: C.muted });
          drawMat(g, rx, ry + cs + 8, row(v1), cs, {});
          const leak = [0, 1, 2, 3].map((j) => o1[j] - v1[j]);
          const pulse = 0.5 + 0.5 * Math.sin(t * 4);
          const hs = []; leak.forEach((d, j) => { if (Math.abs(d) > 1e-6) hs.push([0, j, 0.5 + 0.5 * pulse]); });
          drawMat(g, rx, ry, row(o1), cs, { hi: hs, hiColor: C.c });
          if (narrow) V.text(g, 'highlighted: leak of 0.6·v₃', w / 2, ry + 2 * cs + 34, { size: 12, color: C.c, weight: 600, align: 'center' });
          else V.text(g, 'leak = 0.6·v₃', rx + W4 + 12, ry + cs / 2 + 4, { size: 12, color: C.c, weight: 600 });
        } else {
          const cs = narrow ? 13 : 17, gap = 2, W4 = 4 * cs + 3 * gap;
          const mats = [Z, S1, S2, S3], labs = ['S₀', 'S₁', 'S₂', 'S₃'];
          const sp = (w - 4 * W4) / 5;
          const y0 = 120;
          const u = (el * 0.45) % 1; const cur = Math.floor(u * 4);
          mats.forEach((M, k) => {
            const x = sp + k * (W4 + sp);
            drawMat(g, x, y0, M, cs, { values: false, label: labs[k], labelColor: k === cur ? C.a : C.muted });
            if (k === cur) V.glow(g, x + W4 / 2, y0 + W4 / 2, W4 * 0.8, C.a, 0.35);
            if (k < 3) {
              V.arrow(g, x + W4 + 6, y0 + W4 / 2, x + W4 + sp - 6, y0 + W4 / 2, { color: C.a, lw: 2 });
              const tx = x + W4 + sp / 2;
              pill(g, tx, y0 - 34, narrow ? 'x' + (k + 1) : 'token ' + (k + 1), C.d, { size: 11 });
              V.arrow(g, tx, y0 - 24, tx, y0 + W4 / 2 - 8, { color: V.alpha(C.d, 0.8), lw: 1.4, head: 6 });
            }
          });
          wrap(g, 'fixed-size state · 1 step per token · 128×128 FP32 per head in the recipes', w - 24, narrow ? 11.5 : 12.5).forEach((ln, q) => V.text(g, ln, w / 2, y0 + W4 + 50 + q * 16, { size: narrow ? 11.5 : 12.5, align: 'center', color: C.ink }));
        }
        g.restore();
      });
    });

    /* =====================================================================
       Figure 3 — scalar × per-channel decay
       ===================================================================== */
    safe('fig-esquece', () => {
      const fig = root.querySelector('#fig-esquece');
      autoH(fig, 380, 400);
      const ctl = UI.controls(fig);
      const S = { mode: 'kda', u: 63, dd: false };
      const gOf = (u) => -5 * Math.pow(0.002, u / 100);
      UI.seg(ctl, { label: 'Gate', options: [{ v: 'gdn', label: 'scalar/head (GDN)' }, { v: 'kda', label: 'per channel (KDA)' }], value: S.mode, onChange: (v) => { S.mode = v; } });
      UI.slider(ctl, { label: 'log-decay g', min: 0, max: 100, step: 1, value: S.u, fmt: (u) => nf(gOf(u), 2) + ' (α ' + nf(Math.exp(gOf(u)), 3) + ')', onInput: (v) => { S.u = v; } });
      UI.toggle(ctl, { label: 'data-dependent gate (§)', value: S.dd, onChange: (v) => { S.dd = v; } });
      const rH = UI.readout(ctl, 'half-life', '—'), r20 = UI.readout(ctl, 'weight after 20 tokens', '—');
      const TMAX = 40, CH = 16, PT = 0.5;
      const mult = Array.from({ length: CH }, (_, c) => 0.1 * Math.pow(30, c / (CH - 1)));
      const isReset = (id) => id % 23 === 5;
      let lastK = '';
      const LW = new Float64Array(CH * TMAX);
      V.fig(fig, (g, w, h, t) => {
        if (w < 80 || h < 60) return;
        const C = V.C(); const narrow = w < 560;
        const T = narrow ? 26 : TMAX;
        const g0 = gOf(S.u);
        const tick = Math.floor(t / PT) + 20;
        const gc = (c, id) => {
          if (S.dd && isReset(id)) return S.mode === 'kda' ? -5 : -8;
          const gg = S.mode === 'kda' ? g0 * mult[c] : g0;
          return S.mode === 'kda' ? Math.max(-5, gg) : gg;
        };
        for (let c = 0; c < CH; c++) {
          LW[c * T + T - 1] = 0;
          for (let j = T - 2; j >= 0; j--) { const idNext = tick - (T - 1 - (j + 1)); LW[c * T + j] = LW[c * T + j + 1] + gc(c, idNext); }
        }
        const hl = Math.log(2) / Math.abs(g0);
        const k = S.mode + S.u + (S.dd ? 1 : 0);
        if (k !== lastK) {
          lastK = k;
          let w20 = 0; for (let c = 0; c < CH; c++) w20 += Math.exp(20 * (S.mode === 'kda' ? Math.max(-5, g0 * mult[c]) : g0)); w20 /= CH;
          rH.set((hl >= 100 ? nf(hl, 0) : nf(hl, 1)) + ' tok' + (S.mode === 'kda' ? ' (typical channel)' : ''));
          r20.set(nf(w20 * 100, w20 < 0.01 ? 3 : 1) + '%' + (S.mode === 'kda' ? ' (mean)' : ''));
        }
        const lx = narrow ? 10 : 70, rx = w - 12;
        const cw = (rx - lx) / T;
        // token row
        const ty = 26;
        V.text(g, 'past ←', lx, 12, { size: 11, color: C.muted });
        V.text(g, '→ current token', rx, 12, { size: 11, color: C.muted, align: 'right' });
        for (let j = 0; j < T; j++) {
          const id = tick - (T - 1 - j); const x = lx + j * cw + cw / 2;
          const rs = S.dd && isReset(id);
          if (rs) { V.box(g, x - Math.min(7, cw / 2 - 0.5), ty - 7, Math.min(14, cw - 1), 14, { r: 3, fill: V.alpha(C.d, 0.3), stroke: C.d, lw: 1 }); if (cw >= 8) V.text(g, '§', x, ty + 0.5, { size: 11, align: 'center', base: 'middle', color: C.ink, weight: 700 }); }
          else V.dot(g, x, ty, Math.min(3.4, cw * 0.3), j === T - 1 ? C.d : V.alpha(C.muted, 0.7));
        }
        // heat map
        const hy0 = 44, hy1 = h - (narrow ? 118 : 110);
        const rh = (hy1 - hy0) / CH;
        for (let c = 0; c < CH; c++) for (let j = 0; j < T; j++) {
          const wv = Math.exp(LW[c * T + j]);
          g.fillStyle = V.cmap(Math.pow(wv, 0.55), 'seq');
          g.fillRect(lx + j * cw + 0.5, hy0 + c * rh + 0.5, cw - 1, rh - 1);
        }
        g.strokeStyle = V.alpha(C.a, 0.5); g.lineWidth = 1; g.strokeRect(lx, hy0, rx - lx, hy1 - hy0);
        if (!narrow) {
          V.text(g, 'channel 0', lx - 6, hy0 + rh / 2 + 4, { size: 11, align: 'right', color: C.muted });
          V.text(g, 'channel 15', lx - 6, hy1 - rh / 2 + 4, { size: 11, align: 'right', color: C.muted });
          if (S.mode === 'kda') { V.text(g, 'slow', lx - 6, hy0 + rh * 3.5, { size: 11, align: 'right', color: C.a }); V.text(g, 'fast', lx - 6, hy1 - rh * 3, { size: 11, align: 'right', color: C.c }); }
        }
        // bars (channel average)
        const by1 = h - 30, by0 = hy1 + 22;
        V.text(g, S.mode === 'kda' ? 'mean weight of the 16 channels' : 'weight of each token (all channels equal)', lx, hy1 + 14, { size: 11, color: C.muted });
        for (let j = 0; j < T; j++) {
          let s = 0; for (let c = 0; c < CH; c++) s += Math.exp(LW[c * T + j]);
          const m = s / CH; const bh = (by1 - by0) * m;
          g.fillStyle = V.alpha(S.mode === 'kda' ? C.glm : C.qwen, 0.75); V.rr(g, lx + j * cw + 1, by1 - bh, Math.max(1, cw - 2), Math.max(1, bh), 2); g.fill();
        }
        V.line(g, lx, by1, rx, by1, { color: C.grid, lw: 1 });
        if (hl < T - 1) {
          const xh = lx + (T - 1 - hl) * cw + cw / 2;
          V.line(g, xh, by0 - 4, xh, by1, { color: C.d, lw: 1.5, dash: [4, 3] });
          V.text(g, 't½ = ' + nf(hl, 1), xh, by1 + 14, { size: 11, font: 'mono', align: 'center', color: C.d });
        } else V.text(g, 't½ = ' + nf(hl, 0) + ' tokens (outside the window)', rx, by1 + 14, { size: 11, font: 'mono', align: 'right', color: C.d });
        if (S.mode === 'kda' && g0 * mult[CH - 1] < -5) V.text(g, 'floor g ≥ −5 active on fast channels', lx, by1 + 14, { size: 11, color: C.muted });
      });
    });

    /* =====================================================================
       Figure 4 — delta rule lab
       ===================================================================== */
    safe('fig-delta', () => {
      const fig = root.querySelector('#fig-delta');
      autoH(fig, 390, 610, 640);
      const ctl = UI.controls(fig);
      const KEYS = { cap: [1, 0, 0, 0], moe: [0, 1, 0, 0], idi: [0.6, 0, 0.8, 0] };
      const KN = { cap: 'capital', moe: 'currency', idi: 'language' };
      const VALS = { Rio: [2, 1, 0, 0], Real: [0, 0, 0, 2], Bsb: [0, 1, 2, 0], Pt: [1, 0, 0, 1] };
      const VN = { Rio: 'Rio', Real: 'Real', Bsb: 'Brasilia', Pt: 'Portuguese' };
      const WR = [{ k: 'cap', v: 'Rio' }, { k: 'moe', v: 'Real' }, { k: 'cap', v: 'Bsb' }, { k: 'idi', v: 'Pt' }];
      const S = { rule: 'delta', beta: 1, step: -1, auto: true, t0: 0, now: 0 };
      let REC = [];
      const tr = (Sm, k) => [0, 1, 2, 3].map((j) => Sm.reduce((s, r, i) => s + r[j] * k[i], 0));
      function compute() {
        REC = []; let Sm = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]];
        WR.forEach((wr) => {
          const k = KEYS[wr.k], v = VALS[wr.v];
          const vh = tr(Sm, k);
          const e = S.rule === 'delta' ? v.map((x, j) => x - vh[j]) : v.slice();
          const Sn = Sm.map((r, i) => r.map((x, j) => x + S.beta * k[i] * e[j]));
          const ch = []; for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) if (Math.abs(Sn[i][j] - Sm[i][j]) > 1e-6) ch.push([i, j]);
          REC.push({ k, v, vh, e, S: Sn, ch, wr });
          Sm = Sn;
        });
      }
      compute();
      const advance = () => { S.step = S.step >= 3 ? -1 : S.step + 1; S.t0 = S.now; };
      UI.seg(ctl, { label: 'Rule', options: [{ v: 'delta', label: 'delta rule' }, { v: 'soma', label: 'plain sum' }], value: S.rule, onChange: (v) => { S.rule = v; compute(); } });
      UI.slider(ctl, { label: 'write strength', min: 0.1, max: 1, step: 0.05, value: S.beta, fmt: (v) => 'β = ' + nf(v, 2), onInput: (v) => { S.beta = v; compute(); } });
      UI.button(ctl, '▶ next token', () => { S.auto = false; tg.set(false); advance(); }, { primary: true });
      UI.button(ctl, '↺ restart', () => { S.step = -1; S.t0 = S.now; });
      const tg = UI.toggle(ctl, { label: 'auto', value: true, onChange: (v) => { S.auto = v; S.t0 = S.now; } });
      V.fig(fig, (g, w, h, t) => {
        if (w < 80 || h < 60) return;
        const C = V.C(); S.now = t;
        if (S.auto && t - S.t0 > (S.step === 3 ? 5 : 2.8)) advance();
        const narrow = w < 640;
        const Z = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]];
        const cur = S.step >= 0 ? REC[S.step] : null;
        const Sm = cur ? cur.S : Z;
        const flashA = cur ? Math.max(0, 1 - (t - S.t0) / 1.2) : 0;
        // layout
        const csV = narrow ? 26 : 28, gapV = 3, WV = 4 * csV + 3 * gapV;
        const csS = narrow ? 36 : 50, WS = 4 * csS + 9;
        let lx, ly, sx, sy, qx, qy;
        if (!narrow) { lx = 18; ly = 62; sx = lx + 60 + WV + 30; sy = 72; qx = sx + WS + 34; qy = 62; }
        else { lx = 16; ly = 50; sx = (w - WS) / 2; sy = ly + 4 * (csV + 10) + 36; qx = 12; qy = sy + WS + 54; }
        // current token panel
        const hdr = cur ? 'token ' + (S.step + 1) + ' writes “' + KN[cur.wr.k] + '” → “' + VN[cur.wr.v] + '”' + (S.rule === 'delta' ? '' : '  (plain sum)') : 'empty whiteboard · press ▶ next token';
        V.text(g, hdr, lx, 22, { size: 13, weight: 600, color: C.ink, maxW: w - 24 });
        const rowsL = [['k', cur ? cur.k : [0, 0, 0, 0], C.b], ['v', cur ? cur.v : [0, 0, 0, 0], C.a], [narrow ? 'Sᵀk' : 'v̂ = Sᵀk', cur ? cur.vh : [0, 0, 0, 0], C.muted], [S.rule === 'delta' ? (narrow ? 'v − v̂' : 'e = v − v̂') : (narrow ? 'v' : 'writes v'), cur ? cur.e : [0, 0, 0, 0], C.d]];
        rowsL.forEach(([lab, vec, colr], r) => {
          const yy = ly + r * (csV + 10);
          V.text(g, lab, lx + 54, yy + csV / 2 + 4, { size: 12, font: 'mono', align: 'right', color: colr, weight: 600 });
          drawMat(g, lx + 60, yy, [vec], csV, { gap: gapV, max: 2.5 });
        });
        // state
        const hiC = cur ? cur.ch.map(([i, j]) => [i, j, flashA]) : [];
        drawMat(g, sx, sy, Sm, csS, { hi: flashA > 0.02 ? hiC : [], max: 2.5 });
        V.text(g, 'state S (4×4)', sx + WS / 2, sy - 12, { size: 12.5, font: 'mono', align: 'center', color: C.a, weight: 600 });
        V.text(g, 'row i ↔ key channel i', sx + WS / 2, sy + WS + 16, { size: 11, align: 'center', color: C.muted });
        // queries
        const seen = {}; if (cur) for (let s = 0; s <= S.step; s++) seen[WR[s].k] = WR[s].v;
        const csQ = narrow ? 24 : 24, WQ = 4 * csQ + 9;
        V.text(g, 'querying the whiteboard', qx, qy - 16, { size: 12.5, weight: 600, color: C.ink });
        let row = 0;
        ['cap', 'moe', 'idi'].forEach((kk) => {
          if (!seen[kk]) return;
          const got = tr(Sm, KEYS[kk]); const want = VALS[seen[kk]];
          const err = Math.sqrt(got.reduce((s, x, j) => s + (x - want[j]) ** 2, 0));
          const yy = qy + row * (narrow ? csQ + 26 : csQ + 30); row++;
          V.text(g, KN[kk] + ' → ' + VN[seen[kk]], qx, yy + 2, { size: 11.5, color: C.muted });
          drawMat(g, qx, yy + 8, [got], csQ, { max: 2.5 });
          const ex = qx + WQ + 12, bw = Math.max(20, (narrow ? w - ex - 70 : w - ex - 80));
          V.box(g, ex, yy + 8 + csQ / 2 - 5, bw, 10, { r: 5, fill: V.alpha(C.muted, 0.15) });
          const fe = Math.min(1, err / 3);
          V.box(g, ex, yy + 8 + csQ / 2 - 5, Math.max(3, bw * fe), 10, { r: 5, fill: err < 0.05 ? C.a : err < 1 ? C.d : C.c });
          V.text(g, err < 0.05 ? 'exact' : (narrow ? 'err ' : 'error ') + nf(err, 2), ex + bw + 6, yy + 8 + csQ / 2 + 4, { size: 11.5, font: 'mono', color: err < 0.05 ? C.a : C.ink });
        });
        if (!row) V.text(g, '(no facts recorded yet)', qx, qy + 16, { size: 11.5, color: C.muted });
      });
    });

    /* =====================================================================
       Figure 5 — GDN / KDA anatomy
       ===================================================================== */
    safe('fig-anatomia', () => {
      const fig = root.querySelector('#fig-anatomia');
      autoH(fig, 500, 680, 620);
      const ctl = UI.controls(fig);
      const S = { m: 'qwen', tp: 1 };
      UI.seg(ctl, { label: 'Model', options: [{ v: 'qwen', label: 'Qwen · GDN' }, { v: 'glm', label: 'GLM · KDA' }], value: S.m, onChange: (v) => { S.m = v; } });
      UI.seg(ctl, { label: 'Shapes', options: [{ v: 1, label: 'whole model' }, { v: 2, label: 'per GPU (TP2)' }], value: S.tp, onChange: (v) => { S.tp = v; } });
      const D = {
        qwen: {
          1: [
            ['x · input', '[2560], read from the 4 Gated Residual branches', 'muted'],
            ['projections', 'in_proj_qkv [10240×2560] · in_proj_z [6144×2560] · in_proj_a/b [48×2560]', 'b'],
            ['short causal conv + SiLU', 'conv1d [10240, 1, 4] over q | k | v', 'd'],
            ['norms and heads', 'L2Norm(q), L2Norm(k), q·1/√128 · 16 QK heads serve 48 V heads', 'e'],
            ['recurrence gates', 'α = exp(−exp(A_log)·softplus(a + dt_bias)) [48] · β = σ(b) [48]', 'c'],
            ['state S · gated delta rule', '48 × [128×128] FP32 = 3 MiB · S ← α(I − βkkᵀ)S + βkvᵀ', 'a'],
            ['read', 'o = Sᵀq → [48 × 128]', 'a'],
            ['norm + output gate', 'RMSNorm(o) ⊙ σ(z) · shared norm [128]', 'f'],
            ['output projection', 'out_proj [2560×6144] → y [2560]', 'b'],
          ],
          2: [
            ['x · input', '[2560] (replicated on the 2 GPUs)', 'muted'],
            ['projections', 'fused qkvz [8192×2560] · fused ba [48×2560] per GPU', 'b'],
            ['short causal conv + SiLU', '5,120 channels per GPU', 'd'],
            ['norms and heads', '8 QK heads → 24 V heads per GPU', 'e'],
            ['recurrence gates', 'α, β: [24] per GPU', 'c'],
            ['state S · gated delta rule', '24 × [128×128] FP32 = 1.5 MiB per GPU', 'a'],
            ['read', 'o = Sᵀq → [24 × 128]', 'a'],
            ['norm + output gate', 'RMSNorm(o) ⊙ σ(z)', 'f'],
            ['output projection', 'out_proj [2560×3072] + all-reduce', 'b'],
          ],
        },
        glm: {
          1: [
            ['x · input', '[4096], after mHC-pre + RMSNorm', 'muted'],
            ['fused projection', 'in_proj_qkvbfg_a [24896×4096] = q,k,v 3×8192 + β 64 + f_a 128 + g_a 128', 'b'],
            ['short causal conv + SiLU', 'q_conv1d / k_conv1d / v_conv1d [8192, 1, 4]', 'd'],
            ['norms and heads', 'L2Norm(q, k) inside the kernel · 64 heads × 128', 'e'],
            ['recurrence gates', 'g = −5·σ(exp(A_log)·(f_b(f_a) + dt_bias)) [64×128] · α = eᵍ · β = σ(b) [64]', 'c'],
            ['state S · KDA', '64 × [128×128] FP32 = 4 MiB · S ← (I − βkkᵀ)Diag(α)S + βkvᵀ', 'a'],
            ['read', 'o = Sᵀq → [64 × 128]', 'a'],
            ['norm + output gate', 'RMSNorm(o) ⊙ σ(g_b(g_a)) · o_norm [128]', 'f'],
            ['output projection', 'o_proj [4096×8192] → y [4096]', 'b'],
          ],
          2: [
            ['x · input', '[4096] (replicated on the 2 ranks)', 'muted'],
            ['fused projection', 'local in_proj [12576×4096] = 3·4096 + 32 + 128 + 128 (f_a, g_a replicated)', 'b'],
            ['short causal conv + SiLU', '12,288 channels per rank', 'd'],
            ['norms and heads', '32 heads × 128 per rank', 'e'],
            ['recurrence gates', 'g: [32×128] · β: [32] per rank', 'c'],
            ['state S · KDA', '32 × [128×128] FP32 = 2 MiB per rank', 'a'],
            ['read', 'o = Sᵀq → [32 × 128]', 'a'],
            ['norm + output gate', 'RMSNorm(o) ⊙ σ(g_b(g_a))', 'f'],
            ['output projection', 'o_proj [4096×4096] + all-reduce', 'b'],
          ],
        },
      };
      let cacheK = '', layout = null;
      V.fig(fig, (g, w, h, t) => {
        if (w < 80 || h < 60) return;
        const C = V.C(); const narrow = w < 620;
        const list = D[S.m][S.tp];
        const pw = narrow ? w - 24 : Math.min(560, w * 0.64);
        const px = 12;
        const key = [w, h, S.m, S.tp, narrow].join('|');
        if (key !== cacheK) {
          cacheK = key;
          const boxes = list.map(([ti, sh, c]) => ({ ti, c, lines: wrap(g, sh, pw - 26, 11, 'mono', 400) }));
          const gapY = narrow ? 10 : 12;
          const need = boxes.reduce((s, b) => s + 24 + b.lines.length * 14, 0) + gapY * (boxes.length - 1);
          const scale = Math.min(1, (h - 20) / need);
          let y = 10 + (narrow ? 0 : Math.max(0, (h - 20 - need) / 2));
          boxes.forEach((b) => { b.h = (24 + b.lines.length * 14) * scale; b.y = y; y += b.h + gapY * scale; });
          layout = { boxes, scale };
        }
        const cols = { muted: C.muted, a: C.a, b: C.b, c: C.c, d: C.d, e: C.e, f: C.f };
        const B = layout.boxes;
        // connectors with particles
        for (let i = 0; i < B.length - 1; i++) {
          const x = px + 22, y1 = B[i].y + B[i].h, y2 = B[i + 1].y;
          V.line(g, x, y1, x, y2, { color: V.alpha(C.muted, 0.5), lw: 1.5 });
          const u = (t * 1.1 + i * 0.13) % 1; V.dot(g, x, V.lerp(y1, y2, u), 2.6, cols[B[i + 1].c] || C.a);
        }
        B.forEach((b, i) => {
          const colr = cols[b.c] || C.a;
          const isState = b.ti.startsWith('state');
          V.box(g, px, b.y, pw, b.h, { r: 9, fill: V.alpha(colr, isState ? 0.2 : 0.1), stroke: V.alpha(colr, isState ? 0.95 : 0.6), lw: isState ? 2 : 1.2 });
          V.text(g, (i + 1) + '. ' + b.ti, px + 12, b.y + 15 * layout.scale + 2, { size: 12.5, weight: 600, color: C.ink });
          b.lines.forEach((ln, k) => V.text(g, ln, px + 14, b.y + (30 + k * 14) * layout.scale, { size: 11, font: 'mono', color: C.ink, alpha: 0.85 }));
        });
        const heads = S.m === 'qwen' ? (S.tp === 1 ? 48 : 24) : (S.tp === 1 ? 64 : 32);
        if (narrow) {
          const last = B[B.length - 1]; const y0 = last.y + last.h + 22;
          if (h - y0 < 70) return;
          const n2 = 8, c2 = 7, sd = n2 * (c2 + 1);
          for (let sN = 3; sN >= 0; sN--) {
            const ox = px + 6 + sN * 6, oy = y0 + sN * 6;
            if (sN > 0) { V.box(g, ox, oy, sd, sd, { r: 3, fill: V.alpha(C.a, 0.12), stroke: V.alpha(C.a, 0.4), lw: 1 }); continue; }
            const tk = Math.floor(t * 1.6);
            for (let i = 0; i < n2; i++) for (let j = 0; j < n2; j++) { g.fillStyle = V.cmap(Math.sin(i * 0.9 + j * 0.5 + tk * 0.7) * 0.6, 'div'); g.fillRect(ox + j * (c2 + 1), oy + i * (c2 + 1), c2, c2); }
          }
          const tx = px + sd + 40;
          V.text(g, 'state of this layer', tx, y0 + 12, { size: 12, weight: 600, color: C.a });
          V.text(g, heads + ' × 128 × 128 · FP32', tx, y0 + 30, { size: 11.5, font: 'mono', color: C.ink });
          V.text(g, '= ' + fB(heads * 65536) + ' per request', tx, y0 + 47, { size: 11.5, font: 'mono', color: C.ink });
          V.text(g, S.m === 'qwen' ? 'scalar α per head' : 'α per channel (128/head)', tx, y0 + 66, { size: 11.5, color: C.c });
          if (h - y0 >= 112) {
            V.text(g, 'β: write strength per head', tx, y0 + 85, { size: 11.5, color: C.e, weight: 600 });
            V.text(g, 'conv_state: channels × (3 + k)', tx, y0 + 104, { size: 11.5, color: C.d, weight: 600 });
          }
          return;
        }
        // state panel (right)
        const rx0 = px + pw + 24, rw = w - rx0 - 14;
        if (rw < 140) return;
        const bytes = heads * 128 * 128 * 4;
        V.text(g, 'state of this layer', rx0, 30, { size: 12.5, weight: 600, color: C.a });
        V.text(g, heads + ' heads × 128 × 128 · FP32', rx0, 47, { size: 11.5, font: 'mono', color: C.ink });
        V.text(g, '= ' + fB(bytes) + ' per request', rx0, 63, { size: 11.5, font: 'mono', color: C.ink });
        const n = 12, cs = Math.min(10, (rw - 40) / n - 1), side = n * (cs + 1);
        const stackN = 6;
        for (let s = stackN - 1; s >= 0; s--) {
          const ox = rx0 + 6 + s * 7, oy = 84 + s * 7;
          if (s > 0) { V.box(g, ox, oy, side, side, { r: 4, fill: V.alpha(C.a, 0.08 + 0.02 * (stackN - s)), stroke: V.alpha(C.a, 0.4), lw: 1 }); continue; }
          const tk = Math.floor(t * 1.6);
          for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
            const v = Math.sin(i * 0.9 + j * 0.5 + tk * 0.7) * 0.5 + Math.sin(i * 0.3 - j * 0.8 + tk * 0.23) * 0.45;
            g.fillStyle = V.cmap(v, 'div'); g.fillRect(ox + j * (cs + 1), oy + i * (cs + 1), cs, cs);
          }
          g.strokeStyle = C.a; g.lineWidth = 1.5; g.strokeRect(ox - 1, oy - 1, side + 1, side + 1);
        }
        const gy = 84 + side + stackN * 7 + 26;
        V.text(g, S.m === 'qwen' ? 'α: one scalar per head' : 'α: one value per channel (128/head)', rx0, gy, { size: 11.5, color: C.c, weight: 600 });
        const nb = S.m === 'qwen' ? 8 : 32, bw = (rw - 10) / nb;
        for (let k = 0; k < nb; k++) {
          const a = S.m === 'qwen' ? 0.55 + 0.4 * Math.sin(k * 1.7 + t * 0.8) * 0.5 + 0.2 : 0.5 + 0.45 * Math.sin(k * 0.61 + t * 1.3 + Math.sin(k));
          const hh = 30 * V.clamp(a, 0.05, 1);
          V.box(g, rx0 + k * bw, gy + 10 + 30 - hh, Math.max(2, bw - 2), hh, { r: 2, fill: V.alpha(C.c, 0.7) });
        }
        V.text(g, S.m === 'qwen' ? '(8 of ' + heads + ' heads)' : '(32 of 128 channels of 1 head)', rx0, gy + 56, { size: 11, color: C.muted });
        V.text(g, 'β: write strength per head', rx0, gy + 80, { size: 11.5, color: C.e, weight: 600 });
        V.text(g, 'conv_state: channels × (3 + k)', rx0, gy + 100, { size: 11.5, color: C.d, weight: 600 });
      });
    });

    /* =====================================================================
       Figure 6 — short-convolution conveyor belt
       ===================================================================== */
    safe('fig-conv', () => {
      const fig = root.querySelector('#fig-conv');
      autoH(fig, 360, 390);
      const ctl = UI.controls(fig);
      const S = { dil: 1, spec: 0 };
      UI.seg(ctl, { label: 'Dilation', options: [{ v: 1, label: 'δ = 1 (GDN / KDA)' }, { v: 3, label: 'δ = 3 (N-gram conv)' }], value: S.dil, onChange: (v) => { S.dil = v; } });
      UI.slider(ctl, { label: 'drafts (k)', min: 0, max: 7, step: 1, value: S.spec, fmt: (v) => String(v), onInput: (v) => { S.spec = v; } });
      const rW = UI.readout(ctl, 'conv_state', '—'), rF = UI.readout(ctl, 'receptive field', '—');
      const Wt = [0.55, 0.3, -0.2, 0.15];
      const rng = V.rng(21); const X = Array.from({ length: 512 }, () => Math.round((rng() * 2 - 1) * 10) / 10);
      const P = 1.25;
      let lastK = '';
      V.fig(fig, (g, w, h, t) => {
        if (w < 80 || h < 60) return;
        const C = V.C(); const narrow = w < 560;
        const d = S.dil; const width = 3 * d + S.spec; const field = 3 * d + 1;
        const kk = d + ':' + S.spec; if (kk !== lastK) { lastK = kk; rW.set('[channels, ' + width + ']'); rF.set(field + ' tokens'); }
        const tick = Math.floor(t / P) + 12; const fr = (t / P) % 1; const slide = 1 - V.ease.out(Math.min(1, fr * 2.5));
        const xh = narrow ? w - 34 : w * 0.52; const sp = narrow ? Math.min(30, (xh - 24) / (3 * d)) : 38; const by = 58;
        V.text(g, 'input x (one channel)', 12, 20, { size: 11.5, color: C.muted });
        g.fillStyle = V.alpha(C.muted, 0.1); V.rr(g, 8, by - 18, w - 16, 36, 18); g.fill();
        const taps = [0, 1, 2, 3].map((j) => tick - j * d);
        for (let i = tick - 30; i <= tick; i++) {
          const x = xh - (tick - i) * sp + slide * sp; if (x < 16 || x > w - 12) continue;
          const tapIdx = taps.indexOf(i);
          const r = narrow ? Math.min(12, sp / 2 - 1.5) : 14;
          g.fillStyle = tapIdx >= 0 ? V.alpha(C.b, 0.35) : V.alpha(C.muted, 0.18);
          g.beginPath(); g.arc(x, by, r, 0, Math.PI * 2); g.fill();
          if (tapIdx >= 0) { g.strokeStyle = C.b; g.lineWidth = 2; g.stroke(); }
          V.text(g, fmtv(X[i % 512]), x, by + 0.5, { size: narrow ? 10 : 11, font: 'mono', align: 'center', base: 'middle', color: C.ink });
        }
        // sum
        const sy = by + (narrow ? 82 : 92);
        let acc = 0;
        taps.forEach((i, j) => {
          const x = xh - (tick - i) * sp + slide * sp; acc += Wt[j] * X[i % 512];
          V.line(g, x, by + 15, xh, sy - 16, { color: V.alpha(C.b, 0.7), lw: 1.4 });
          V.text(g, 'w' + j, x, by - 22, { size: 11, font: 'mono', color: C.b, align: 'center', weight: 600 });
        });
        const eqs = 'Σ = ' + nf(Wt[0], 2) + '·x₀ + ' + nf(Wt[1], 2) + '·x₋' + d + ' − ' + nf(-Wt[2], 2) + '·x₋' + 2 * d + ' + ' + nf(Wt[3], 2) + '·x₋' + 3 * d;
        if (!narrow) V.text(g, eqs, xh + 24, sy - 14, { size: 11, font: 'mono', color: C.b, maxW: w - xh - 30 });
        V.box(g, xh - 16, sy - 16, 32, 32, { r: 16, fill: V.alpha(C.d, 0.25), stroke: C.d, lw: 1.5, label: 'Σ', size: 15 });
        const y = acc / (1 + Math.exp(-acc));
        const ytxt = 'y = SiLU(' + nf(acc, 2) + ') = ' + nf(y, 2);
        if (narrow) V.text(g, ytxt, xh - 24, sy + 5, { size: 12, font: 'mono', color: C.ink, align: 'right' });
        else V.text(g, ytxt, xh + 24, sy + 5, { size: 12, font: 'mono', color: C.ink });
        // backpack (conv_state)
        const cells = 3 * d, total = cells + S.spec;
        const mlab = wrap(g, 'backpack = conv_state: (K−1)·δ + num_spec = ' + cells + ' + ' + S.spec + ' = ' + total + ' columns per channel', w - 24, 11.5, 'sans', 600);
        const my0 = sy + (narrow ? 36 : 50) + (mlab.length - 1) * 15;
        const cw = Math.min(narrow ? 24 : 32, (w - 40) / Math.max(1, total) - 3);
        const mw = total * (cw + 3) + 16;
        const mx0 = narrow ? 12 : Math.max(12, xh - mw + 40);
        mlab.forEach((ln, q) => V.text(g, ln, 12, my0 - 10 - (mlab.length - 1 - q) * 15, { size: 11.5, color: C.ink, weight: 600 }));
        V.box(g, mx0, my0, mw, cw + 16, { r: 9, fill: V.alpha(C.d, 0.08), stroke: V.alpha(C.d, 0.7), lw: 1.2 });
        for (let c = 0; c < total; c++) {
          const x = mx0 + 8 + c * (cw + 3), yy = my0 + 8;
          const isSpec = c >= cells;
          const id = tick - (cells - 1 - c);
          const nextTap = !isSpec && [1, 2, 3].some((j) => tick + 1 - j * d === id);
          V.box(g, x, yy, cw, cw, { r: 4, fill: isSpec ? V.alpha(C.c, 0.3) : V.alpha(nextTap ? C.b : C.muted, nextTap ? 0.35 : 0.18), stroke: isSpec ? C.c : nextTap ? C.b : V.alpha(C.muted, 0.4), lw: 1 });
          if (cw >= 20) V.text(g, isSpec ? 'd' + (c - cells + 1) : fmtv(X[id % 512]), x + cw / 2, yy + cw / 2 + 0.5, { size: cw < 27 ? 10 : 11, font: 'mono', align: 'center', base: 'middle', color: C.ink, maxW: cw - 2 });
        }
        const ly = my0 + cw + 34;
        V.text(g, narrow ? 'blue: taps for the next token · pink: drafts' : 'blue: inputs the next token will use · pink: extra columns to verify drafts (rollback via num_accepted_tokens)', 12, ly, { size: 11, color: C.muted, maxW: w - 24 });
        const lines = narrow
          ? ['GLM KDA (TP2, k=7): [12288, 10] BF16', 'Qwen GDN (TP2, MTP 3): [5120, 6] BF16', 'N-gram PLE (MTP 3): [10240, 12] BF16']
          : ['GLM KDA (TP2, DFlash2 k = 7): [12288, 10] BF16 = 240 KiB/layer', 'Qwen GDN (TP2, MTP 3): [5120, 6] BF16 ≈ 60 KiB/layer   ·   N-gram PLE (MTP 3): [10240, 12] ≈ 240 KiB'];
        lines.forEach((s, k) => V.text(g, s, 12, ly + 18 + k * 16, { size: 11, font: 'mono', color: C.ink, maxW: w - 24 }));
      });
    });

    /* =====================================================================
       Figure 7 — recurrent × chunkwise
       ===================================================================== */
    safe('fig-chunks', () => {
      const fig = root.querySelector('#fig-chunks');
      autoH(fig, 400, 360);
      const ctl = UI.controls(fig);
      const S = { mode: 'chunk', C: 4 };
      UI.seg(ctl, { label: 'Form', options: [{ v: 'rec', label: 'recurrent (decode)' }, { v: 'chunk', label: 'chunkwise (prefill)' }], value: S.mode, onChange: (v) => { S.mode = v; } });
      UI.seg(ctl, { label: 'C', options: [2, 4, 8], value: S.C, onChange: (v) => { S.C = v; } });
      const rA = UI.readout(ctl, 'stages (16 tok)', '—'), rB = UI.readout(ctl, 'stages (131,072 tok)', '—');
      const N = 16; let lastK = '';
      V.fig(fig, (g, w, h, t) => {
        if (w < 80 || h < 60) return;
        const C = V.C(); const narrow = w < 560;
        const k = S.mode + S.C; if (k !== lastK) { lastK = k; rA.set(S.mode === 'rec' ? '16' : String(N / S.C)); rB.set(S.mode === 'rec' ? '131,072' : '2,048 (C = 64)'); }
        const mx = 14, bw = (w - 2 * mx) / N, ty = 34, th = Math.min(30, bw - 4);
        V.text(g, S.mode === 'rec' ? 'one token at a time: Sₜ = f(Sₜ₋₁, xₜ)' : 'chunks of ' + S.C + ': parallel inside, state between chunks', mx, 16, { size: 12, color: C.ink, weight: 600 });
        let cur, chunk = -1, el = 0;
        if (S.mode === 'rec') cur = Math.floor(t / 0.45) % N;
        else { const nC = N / S.C; const ph = t / 1.5; chunk = Math.floor(ph) % nC; el = ph % 1; }
        for (let i = 0; i < N; i++) {
          const x = mx + i * bw;
          let on, done;
          if (S.mode === 'rec') { on = i === cur; done = i < cur; }
          else { const ci = Math.floor(i / S.C); on = ci === chunk; done = ci < chunk; }
          V.box(g, x + 1.5, ty, bw - 3, th, { r: 5, fill: on ? V.alpha(C.d, 0.4) : done ? V.alpha(C.a, 0.2) : V.alpha(C.muted, 0.12), stroke: on ? C.d : V.alpha(C.muted, 0.3), lw: on ? 1.6 : 1 });
          if (bw > 18) V.text(g, String(i + 1), x + bw / 2, ty + th / 2 + 0.5, { size: 11, font: 'mono', align: 'center', base: 'middle', color: C.ink });
        }
        const sy = ty + th + 34;
        if (S.mode === 'rec') {
          for (let i = 0; i <= cur; i++) {
            const x = mx + i * bw + bw / 2; const isCur = i === cur;
            V.box(g, x - Math.min(13, bw / 2 - 1), sy - 11, Math.min(26, bw - 2), 22, { r: 5, fill: V.alpha(C.a, isCur ? 0.45 : 0.14), stroke: isCur ? C.a : V.alpha(C.a, 0.35), lw: 1 });
            if (bw >= 22 || isCur) V.text(g, 'S', x, sy + 0.5, { size: 11, font: 'mono', align: 'center', base: 'middle', color: C.ink, weight: isCur ? 700 : 400 });
            if (i > 0) V.arrow(g, x - bw + Math.min(13, bw / 2 - 1), sy, x - Math.min(13, bw / 2 - 1) - 1, sy, { color: V.alpha(C.a, 0.6), lw: 1.2, head: 5 });
          }
          V.arrow(g, mx + cur * bw + bw / 2, ty + th + 2, mx + cur * bw + bw / 2, sy - 13, { color: C.d, lw: 1.6, head: 6 });
          // the state being read and rewritten
          const gy0 = sy + 34, gN = 8, gcs = Math.min(13, (h - 64 - 40 - gy0) / gN - 2);
          const gX = narrow ? mx : w / 2 - 190;
          const fl = Math.max(0, 1 - ((t / 0.45) % 1) * 2.2);
          for (let a = 0; a < gN; a++) for (let b = 0; b < gN; b++) {
            const v = Math.sin(a * 1.1 + b * 0.6 + cur * 0.8) * 0.6 + Math.sin(a * 0.3 - b + cur * 0.4) * 0.3;
            g.fillStyle = V.cmap(v, 'div'); V.rr(g, gX + b * (gcs + 2), gy0 + a * (gcs + 2), gcs, gcs, 2); g.fill();
          }
          const gW = gN * (gcs + 2);
          g.strokeStyle = V.alpha(C.d, 0.3 + 0.7 * fl); g.lineWidth = 2; V.rr(g, gX - 3, gy0 - 3, gW + 4, gW + 4, 5); g.stroke();
          const tX = gX + gW + 18, tW = w - tX - 10;
          if (tW >= 150) {
            let yy = gy0 + 12;
            [['state S: read and rewritten at every token', { size: 12, color: C.ink, weight: 600 }],
              ['128 × 128 × 4 B = 64 KiB per head', { size: 11.5, font: 'mono', color: C.muted }],
              ['few matrix-vector products: bandwidth-bound', { size: 11.5, color: C.muted }],
              ['Qwen TP1: ~0.23 GB of GDN state per sequence per step', { size: 11.5, color: C.qwen }]].forEach(([s, o]) => {
              wrap(g, s, tW, o.size, o.font || 'sans', o.weight || 400).forEach((ln) => { V.text(g, ln, tX, yy, o); yy += o.size + 3; });
              yy += 5;
            });
          }
        } else {
          const nC = N / S.C;
          for (let c = 0; c <= nC; c++) {
            const x = mx + c * S.C * bw;
            const reached = c <= chunk || (c === chunk + 1 && el > 0.75);
            V.box(g, x - 13, sy - 11, 26, 22, { r: 11, fill: V.alpha(C.a, reached ? 0.45 : 0.1), stroke: reached ? C.a : V.alpha(C.a, 0.3), lw: 1.2 });
            V.text(g, 'S', x, sy + 0.5, { size: 11, font: 'mono', align: 'center', base: 'middle', color: C.ink, weight: 600 });
            if (c < nC) V.line(g, x + 13, sy, x + S.C * bw - 13, sy, { color: V.alpha(C.a, 0.35), lw: 1.2, dash: [3, 4] });
          }
          // triangular matrix of the current chunk
          const cx0 = mx + chunk * S.C * bw; const cwid = S.C * bw;
          const mY = sy + 30; const avail = Math.min(cwid, w * 0.5, (h - 64) - 46 - mY);
          const cs = Math.max(6, Math.min(narrow ? 18 : 24, avail / S.C - 2));
          const mw = S.C * (cs + 2);
          const mX = V.clamp(cx0 + cwid / 2 - mw / 2, mx, w - mx - mw);
          const a = V.ease.out(Math.min(1, el * 3));
          for (let i = 0; i < S.C; i++) for (let j = 0; j < S.C; j++) {
            const on = j <= i;
            g.fillStyle = on ? V.alpha(C.b, 0.25 + 0.55 * a * (0.5 + 0.5 * Math.cos((i - j) * 0.5))) : V.alpha(C.muted, 0.06);
            V.rr(g, mX + j * (cs + 2), mY + i * (cs + 2), cs, cs, 2); g.fill();
          }
          V.arrow(g, mx + chunk * S.C * bw + 12, sy + 10, mX + 2, mY + 4, { color: V.alpha(C.a, 0.8), lw: 1.4, head: 6 });
          V.arrow(g, mX + mw, mY + mw - 4, mx + (chunk + 1) * S.C * bw - 12, sy + 11, { color: V.alpha(C.a, 0.8), lw: 1.4, head: 6, curve: -10 });
          const mlab = 'QKᵀ ⊙ mask (WY/UT) · C×C', mlw = tw(g, mlab, 11, 'mono');
          V.text(g, mlab, V.clamp(mX + mw / 2, mx + mlw / 2, w - mx - mlw / 2), mY + mw + 15, { size: 11, font: 'mono', align: 'center', color: C.muted });
        }
        // GPU units
        const gy = h - 64; const lanes = 16; const lw = (w - 2 * mx) / lanes;
        const busy = S.mode === 'rec' ? 1 : Math.min(lanes, (S.C * (S.C + 1)) / 2 * 1.3 | 0);
        V.text(g, 'busy compute units (illustrative)', mx, gy - 10, { size: 11, color: C.muted });
        for (let i = 0; i < lanes; i++) {
          const on = i < busy; const fl = on ? 0.55 + 0.35 * Math.sin(t * 8 + i) : 0;
          V.box(g, mx + i * lw + 2, gy, lw - 4, 16, { r: 3, fill: on ? V.alpha(C.e, fl) : V.alpha(C.muted, 0.1) });
        }
        wrap(g, S.mode === 'rec' ? '1 of 16: sequential, good for 1 token (decode), bad for 100 thousand (prefill)' : busy + ' of 16: matrix multiplications, tensor cores busy', w - 2 * mx, narrow ? 11.5 : 12).forEach((ln, q) => V.text(g, ln, mx, gy + 32 + q * 15, { size: narrow ? 11.5 : 12, color: C.ink }));
      });
    });

    /* =====================================================================
       Figure 8 — subway map of the hybrids
       ===================================================================== */
    safe('fig-metro', () => {
      const fig = root.querySelector('#fig-metro');
      autoH(fig, 430, 560);
      const ctl = UI.controls(fig);
      const S = { which: 'all', lg: 18 };
      const mk = (n, f) => Array.from({ length: n }, (_, i) => f(i));
      const LINES = {
        glm: { name: 'GLM-5.3-Flash', sum: '34 KDA + 11 DSA (+MTP)', types: mk(45, (i) => (i % 4 === 3 ? 'A' : 'L')).concat(['A*']), fixed: 34 * 2342912, grow: 7579, rec: 'glm',
          det: ['34 × KDA: fixed state of 2.23 MiB each (TP2, k = 7)', '11 × MLA NoPE + DSA: 656 B/token + 33 B/token from the indexer', 'pattern KDA, KDA, KDA, DSA × 11 + KDA on layer 44', 'layer 45 = MTP (DSA); DFlash2 reads layers 5, 14, 24, 33, 42'] },
        qwen: { name: 'Qwen3.8-Flash-Next', sum: '36 GDN + 12 QSA (+MTP)', types: mk(48, (i) => ((i + 1) % 4 === 0 ? 'A' : 'L')).concat(['A*']), fixed: 36 * 847872, grow: 13 * 512, rec: 'qwen',
          det: ['36 × Gated DeltaNet: 828 KiB each (BF16 state, TP2, MTP 3)', '12 × QSA + 1 MTP: 512 B/token/GPU in FP8 (formula)', 'full_attention_interval = 4: QSA on layers 3, 7, …, 47', 'layer 1 (0-based index): N-gram/PLE with a dilated short conv'] },
        ds: { name: 'DeepSeek-V4-Flash', sum: '2 SWA + 21 C4A + 20 C128A (+3 MTP)', types: mk(43, (i) => (i < 2 ? 'W' : i % 2 === 0 ? 'C4' : 'C128')).concat(['W*', 'W*', 'W*']), fixed: 46 * 128 * 584, grow: 3850, rec: 'ds',
          det: ['every layer has a 128-token window (584 B rows)', '21 × C4A: + 1:4 compressed KV with an indexer; 20 × C128A: + 1:128 KV', 'layers 0–1 window only; the 3 DSpark stages also have a window', 'no linear layer: no MambaSpec, no recurrent state'] },
      };
      UI.seg(ctl, { label: 'Model', options: [{ v: 'all', label: 'all' }, { v: 'glm', label: 'GLM' }, { v: 'qwen', label: 'Qwen' }, { v: 'ds', label: 'DeepSeek' }], value: S.which, onChange: (v) => { S.which = v; } });
      UI.slider(ctl, { label: 'Context', min: 10, max: 20, step: 0.25, value: S.lg, fmt: (v) => nf(Math.round(2 ** v)) + ' tok', onInput: (v) => { S.lg = v; } });
      const cv = (nm) => 'var(--viz-' + nm + ')';
      UI.legend(ctl, [[cv('a'), 'linear (fixed state)'], [cv('b'), 'attention (KV grows)'], [cv('d'), 'window only'], [cv('c'), 'window + C4A'], [cv('f'), 'window + C128A']]);
      V.fig(fig, (g, w, h, t) => {
        if (w < 80 || h < 60) return;
        const C = V.C(); const narrow = w < 620;
        const keys = S.which === 'all' ? ['glm', 'qwen', 'ds'] : [S.which];
        const single = keys.length === 1;
        const n = Math.round(2 ** S.lg);
        const lineW = narrow ? w - 24 : w * 0.66; const lx0 = 12;
        const bx0 = narrow ? 12 : lx0 + lineW + 24, bW = narrow ? w - 24 : w - bx0 - 14;
        const slotH = single ? (narrow ? 200 : 150) : (h - 16) / keys.length;
        const maxTot = Math.max(...keys.map((k) => LINES[k].fixed + LINES[k].grow * n));
        if (single) {
          const L = LINES[keys[0]]; let yy = slotH + (narrow ? 10 : 30);
          const dW = narrow ? w - 40 : w * 0.46;
          V.text(g, 'inside', 12, yy, { size: 12.5, weight: 700, color: C[L.rec] }); yy += 22;
          L.det.forEach((s) => { wrap(g, s, dW, 12).forEach((ln, q) => { if (q === 0) V.dot(g, 18, yy - 4, 3, C[L.rec]); V.text(g, ln, 28, yy, { size: 12, color: C.ink }); yy += 17; }); yy += 5; });
          // chart: memory per request × context
          const gx0 = narrow ? 16 : w * 0.56, gx1 = w - 16, gy0 = narrow ? yy + 24 : slotH + 34, gy1 = h - 26;
          if (gy1 - gy0 > 60) {
            const XX = (nn) => gx0 + ((Math.log2(nn) - 10) / 10) * (gx1 - gx0);
            const mxB = L.fixed + L.grow * 1048576; const YY = (b) => gy1 - (b / mxB) * (gy1 - gy0);
            V.text(g, 'memory per request × context', gx0, gy0 - 10, { size: 11, color: C.muted });
            g.beginPath(); g.moveTo(gx0, gy1);
            for (let e = 10; e <= 20.001; e += 0.25) g.lineTo(XX(2 ** e), YY(L.fixed + L.grow * 2 ** e));
            g.lineTo(gx1, gy1); g.closePath(); g.fillStyle = V.alpha(keys[0] === 'ds' ? C.c : C.b, 0.35); g.fill();
            g.fillStyle = V.alpha(C.a, 0.8); g.fillRect(gx0, YY(L.fixed), gx1 - gx0, gy1 - YY(L.fixed));
            V.line(g, gx0, gy1, gx1, gy1, { color: C.grid }); V.line(g, gx0, gy0, gx0, gy1, { color: C.grid });
            [[1024, '1k'], [32768, '32k'], [262144, '262k'], [1048576, '1M']].forEach(([v, l]) => V.text(g, l, XX(v), gy1 + 14, { size: 11, font: 'mono', align: 'center', color: C.muted }));
            V.text(g, fB(mxB), narrow ? gx0 + 4 : gx0 - 4, gy0 + 10, { size: 11, font: 'mono', align: narrow ? 'left' : 'right', color: C.muted });
            const xn = XX(n); V.line(g, xn, gy0, xn, gy1, { color: C.d, lw: 1.5 }); V.dot(g, xn, YY(L.fixed + L.grow * n), 4.5, C.d);
            V.text(g, 'fixed (state/windows)', gx0 + 6, Math.min(gy1 - 5, YY(L.fixed) - 4), { size: 11, color: C.a, weight: 600 });
          }
        }
        keys.forEach((k, idx) => {
          const L = LINES[k]; const recCol = C[L.rec];
          const y0 = 8 + idx * slotH;
          const ly = y0 + (narrow ? 70 : Math.min(slotH * 0.5, 66));
          V.text(g, L.name, lx0, y0 + 16, { size: 12.5, weight: 700, color: recCol });
          if (narrow) V.text(g, L.sum, lx0, y0 + 32, { size: 11, color: C.muted });
          else V.text(g, L.sum, lx0 + tw(g, L.name, 12.5, 'sans', 700) + 10, y0 + 16, { size: 11, color: C.muted, maxW: lineW - tw(g, L.name, 12.5, 'sans', 700) - 10 });
          const nS = L.types.length; const sx = (i) => lx0 + 8 + (i / (nS - 1)) * (lineW - 16);
          V.line(g, sx(0), ly, sx(nS - 1), ly, { color: V.alpha(recCol, 0.55), lw: 4 });
          const u = ((t * 0.07 + idx * 0.31) % 1); const trainI = u * (nS - 1); const tx = sx(trainI);
          // lookup arcs when the train passes an attention station
          const ni = Math.round(trainI); const ty = L.types[ni] ? L.types[ni].replace('*', '') : '';
          if ((ty === 'A' || ty === 'C4') && Math.abs(trainI - ni) < 0.45) {
            const xs = sx(ni);
            [0.15, 0.4, 0.7].forEach((f, q) => { const xb = lx0 + 8 + f * (xs - lx0 - 8); g.strokeStyle = V.alpha(ty === 'A' ? C.b : C.c, 0.5 - q * 0.1); g.lineWidth = 1.3; g.beginPath(); g.moveTo(xs, ly - 6); g.quadraticCurveTo((xs + xb) / 2, ly - 30 - q * 6, xb, ly - 5); g.stroke(); });
          }
          L.types.forEach((tp, i) => {
            const x = sx(i); const mtp = tp.endsWith('*'); const base = tp.replace('*', '');
            const near = Math.abs(trainI - i) < 0.6;
            if (base === 'L') { V.dot(g, x, ly, near ? 4.4 : 3.2, C.a); V.box(g, x - 2.5, ly + 6, 5, 5, { r: 1, fill: V.alpha(C.a, near ? 0.95 : 0.45) }); }
            else if (base === 'A') { g.beginPath(); g.arc(x, ly, near ? 6 : 4.6, 0, Math.PI * 2); g.fillStyle = C.bg; g.fill(); g.strokeStyle = C.b; g.lineWidth = 2.2; if (mtp) g.setLineDash([2, 2]); g.stroke(); g.setLineDash([]); }
            else if (base === 'W') { V.box(g, x - 3.5, ly - 3.5, 7, 7, { r: 1.5, fill: C.d, stroke: mtp ? C.ink : null, lw: 1 }); }
            else { g.beginPath(); g.arc(x, ly, near ? 5 : 3.8, 0, Math.PI * 2); g.fillStyle = C.bg; g.fill(); g.strokeStyle = base === 'C4' ? C.c : C.f; g.lineWidth = 2; g.stroke(); }
          });
          V.glow(g, tx, ly, 14, C.d, 0.9); V.dot(g, tx, ly, 3, C.d);
          if (!narrow || true) V.text(g, 'layer 0', sx(0), ly + 26, { size: 11, color: C.muted });
          V.text(g, 'MTP', sx(nS - 1), ly + 26, { size: 11, color: C.muted, align: 'right' });
          // memory bars
          const fixedB = L.fixed, growB = L.grow * n, tot = fixedB + growB;
          const by = narrow ? ly + 50 : y0 + 26; const bh = narrow ? 12 : 14;
          const scale = bW / maxTot;
          const fW = Math.max(2, fixedB * scale), gW = Math.max(2, growB * scale);
          V.text(g, 'memory per request', bx0, by - 4 + (narrow ? 0 : 0), { size: 11, color: C.muted });
          V.box(g, bx0, by + 2, fW, bh, { r: 3, fill: V.alpha(C.a, 0.75) });
          V.box(g, bx0 + fW, by + 2, gW, bh, { r: 3, fill: V.alpha(k === 'ds' ? C.c : C.b, 0.75) });
          V.text(g, 'fixed ' + fB(fixedB) + ' · grows ' + fB(growB) + ' = ' + fB(tot), bx0, by + bh + 18, { size: 11, font: 'mono', color: C.ink, maxW: bW });
          if (!narrow) V.text(g, nf(L.grow) + ' B/token × ' + nf(n), bx0, by + bh + 34, { size: 11, font: 'mono', color: C.muted, maxW: bW });
        });
      });
    });

    /* =====================================================================
       Figure 9 — checkpoints in align mode (stepper)
       ===================================================================== */
    safe('fig-align', () => {
      const fig = root.querySelector('#fig-align');
      autoH(fig, 340, 330);
      const steps = [
        { t: 'State only exists in the present', d: 'As prefill advances, each KDA layer carries a single state: the summary of tokens 0…t. Nobody kept the state at token 5,000. Once processing moves past it, it ceases to exist, and there is no way to “subtract” the following tokens from a whiteboard.' },
        { t: 'Align mode: a snapshot at the end of each step', d: 'With <code>mamba_cache_mode=align</code>, the state is saved at the end of each scheduler step, and the steps are cut to end on block boundaries (multiples of 3,584 in GLM). With chunks of 7,168 (<code>MAX_NUM_BATCHED_TOKENS</code>, the size of a prefill running alone, without competing for the step with decodes) there are snapshots at 7,168, 14,336 and at the aligned end of the prompt. Only at those points can the request be resumed.' },
        { t: 'The chunk has to stop in the right place', d: 'The state is only materialized at the end of a chunk. If the step ends at a point that is not a checkpoint boundary, the snapshot is useless. Stock aligned to the smallest page, the drafter\'s (64), and applied the EAGLE block drop backoff (discarding the last block that hit) to every group, although only the drafter\'s SWA group needs it. <code>patch_mamba_align_chunking.py</code> uses the scheduler\'s LCM (3,584) for shared checkpoints. When competing for the step with decodes, the fair scheduler gives grants smaller than a block (128 to 2,048 tokens), and the patch lets them advance instead of rounding them down to zero. Illustrative chunks on the top row.' },
        { t: 'Async scheduling: the single slot leaked', d: 'With async scheduling, step N+1 is scheduled while N runs, and the processed prefix lags one batch behind. The manager remembered the previous state in a single <code>last_state_block_idx</code>, overwritten before the old block could be freed. Result: state blocks stuck until the end of the request.' },
        { t: 'Fix: a list of superseded states', d: '<code>patch_mamba_align_state_free.py</code> keeps <code>stale_state_block_idxs</code> per request and frees each superseded state once the processed prefix moves past it. The maximum reservation becomes 1 + <code>max_concurrent_batches</code> (2) + <code>num_speculative_blocks</code> (7) = 10 blocks (before: 2 + 7 = 9).' },
        { t: 'Speculation: k+1 candidate states', d: 'In decode with DFlash2 (k = 7), the target verifies 8 tokens at once. The recurrent kernel keeps the state after <i>each</i> of them, and <code>num_accepted_tokens</code> picks which one becomes the official state. The <code>conv_state</code> also has 3 + 7 columns to allow rollback.' },
        { t: 'Reuse: the hit stops at the last checkpoint', d: 'A second request that starts with the same 8,004 tokens only reuses up to the last snapshot: 7,168 tokens (2 × 3,584). The remaining 836 are recomputed. That is why every GLM prefix cache hit is a multiple of 3,584.' },
      ];
      const st = UI.stepper(fig, steps, { autoplay: 8500 });
      const ctl = UI.controls(fig);
      const cv = (nm) => 'var(--viz-' + nm + ')';
      UI.legend(ctl, [[cv('a'), 'current state'], [cv('b'), 'checkpoint saved / in use'], [cv('muted'), 'freed'], [cv('c'), 'stuck (leaked)']]);
      const BLK = 3584, NB = 5, TOT = BLK * NB, ENDS = [7168, 14336, 17920];
      V.fig(fig, (g, w, h, t) => {
        if (w < 80 || h < 60) return;
        const C = V.C(); const i = st.i; const el = st.elapsed(); const narrow = w < 560;
        const x0 = 18, x1 = w - 18; const X = (p) => x0 + (p / TOT) * (x1 - x0);
        const py = 52, ph = 22, bs = narrow ? 18 : 22;
        const rowY = py + ph + 70;
        const loop = (per) => (el % per) / per;
        const box = (x, y, colr, a, lab, lw) => { V.box(g, x, y, bs, bs, { r: 4, fill: V.alpha(colr, a), stroke: colr, lw: lw || 1.3 }); if (lab) V.text(g, lab, x + bs / 2, y + bs / 2 + 0.5, { size: 11, font: 'mono', align: 'center', base: 'middle', color: C.ink, weight: 700 }); };
        const at = (p) => X(p) - bs / 2;
        const prompt = (fillTo) => {
          V.text(g, (narrow ? 'prompt of ' + nf(TOT) + ' tok · blocks of ' : 'prompt of ' + nf(TOT) + ' tokens · block boundaries every ') + nf(BLK), x0, 20, { size: 12, color: C.ink, weight: 600 });
          g.fillStyle = V.alpha(C.muted, 0.12); V.rr(g, x0, py, x1 - x0, ph, 6); g.fill();
          if (fillTo > 0) { g.fillStyle = V.alpha(C.a, 0.35); V.rr(g, x0, py, Math.max(4, X(fillTo) - x0), ph, 6); g.fill(); }
          for (let b = 1; b <= NB; b++) { V.line(g, X(b * BLK), py - 5, X(b * BLK), py + ph + 5, { color: C.muted, lw: 1 }); if (!narrow || b % 2 === 0 || b === NB) V.text(g, nf(b * BLK), X(b * BLK), py + ph + 17, { size: 11, font: 'mono', align: b === NB ? 'right' : 'center', color: C.muted }); }
        };
        const rowLabel = (s) => V.text(g, s, x0, rowY - 10, { size: 11, color: C.muted });
        // text wrapped into lines (without squeezing the letters); returns the y of the next free line
        const para = (s, x, y, o) => { const sz = o.size || 12; const L = wrap(g, s, x1 - x, sz, o.font || 'sans', o.weight || 400); L.forEach((ln, q) => V.text(g, ln, x, y + q * (sz + 4), Object.assign({}, o, { size: sz }))); return y + L.length * (sz + 4); };
        if (i === 0 || i === 1) {
          const p = Math.min(TOT, loop(9) * TOT * 1.12);
          prompt(p);
          V.line(g, X(p), py - 8, X(p), rowY - 4, { color: C.d, lw: 2 });
          rowLabel('state blocks of the Mamba group');
          box(at(p), rowY, C.a, 0.5, 'S');
          if (i === 0) {
            V.text(g, 'state at t = 5,000?  ✕ already overwritten', x0, rowY + bs + 40, { size: 12.5, color: C.c, weight: 600 });
            V.text(g, 'only S(now) exists: nothing to reuse later', x0, rowY + bs + 60, { size: 12, color: C.muted });
          } else {
            const yb = rowY + bs + 16;
            ENDS.forEach((e) => { if (p >= e) { box(at(e), yb, C.b, 0.45, '✓'); const fl = Math.max(0, 1 - ((p - e) / TOT) * 18); if (fl > 0) V.glow(g, X(e), py + ph / 2, 30, C.d, fl); } });
            [3584, 10752].forEach((e) => { g.setLineDash([3, 3]); g.strokeStyle = V.alpha(C.muted, 0.6); g.lineWidth = 1; V.rr(g, at(e), yb, bs, bs, 4); g.stroke(); g.setLineDash([]); });
            const yq = para('snapshots at the end of each aligned step (7,168, 14,336, 17,920)', x0, yb + bs + 26, { size: 12, color: C.b, weight: 600 });
            para('3,584 and 10,752 fall mid-step: no snapshot here', x0, yq + 2, { size: 11.5, color: C.muted });
          }
        } else if (i === 2) {
          prompt(0);
          const rows = [{ lab: 'wrong granularity (64 from the drafter) + EAGLE backoff on every group (illustrative)', ends: [3072, 7104, 10240, 14272, 17920], ok: false }, { lab: 'with the patch: steps end at the LCM (3,584)', ends: ENDS, ok: true }];
          let yy = py + ph + 34;
          rows.forEach((r) => {
            const yl = para(r.lab, x0, yy, { size: 11.5, color: r.ok ? C.a : C.c, weight: 600 });
            const yBar = yl - 7;
            let prev = 0;
            r.ends.forEach((e, q) => { V.box(g, X(prev) + 1, yBar, X(e) - X(prev) - 2, 16, { r: 4, fill: V.alpha(r.ok ? C.a : C.c, 0.16 + 0.14 * (q % 2)), stroke: V.alpha(r.ok ? C.a : C.c, 0.7), lw: 1 }); prev = e; });
            const yb = yBar + 26;
            r.ends.forEach((e) => {
              const onB = e % BLK === 0;
              if (onB) box(at(e), yb, C.b, 0.45, '✓');
              else { box(at(e), yb, C.c, 0.2, '?'); }
            });
            let yn = yb + bs + 16;
            if (!r.ok) yn = para('“?” = state saved off a boundary: no future hit stops there', x0, yn, { size: 11, color: C.muted });
            yy = yn + (narrow ? 12 : 20);
          });
        } else if (i === 3 || i === 4) {
          const fixd = i === 4;
          const prog = Math.min(2.999, loop(9) * 3.3); const ci = Math.floor(prog);
          prompt(0);
          ENDS.forEach((e, q) => {
            const s = q ? ENDS[q - 1] : 0; const running = q === ci, sched = q === ci + 1, done = q < ci;
            V.box(g, X(s) + 1, py, X(e) - X(s) - 2, ph, { r: 5, fill: running ? V.alpha(C.a, 0.4) : sched ? V.alpha(C.d, 0.2) : done ? V.alpha(C.a, 0.15) : V.alpha(C.muted, 0.06), stroke: running ? C.a : sched ? C.d : null, lw: 1.4 });
            if (running) V.text(g, 'N running', (X(s) + X(e)) / 2, py - 8, { size: 11, align: 'center', color: C.a, weight: 600 });
            if (sched) V.text(g, 'N+1 scheduled', (X(s) + X(e)) / 2, py - 8, { size: 11, align: 'center', color: C.d, weight: 600 });
          });
          rowLabel('state blocks of this request');
          // states: one per step; the running step's is the current one; the previous one is still in use by the batch in flight
          let presos = 0;
          for (let q = 0; q <= ci; q++) {
            const x = at(ENDS[q]);
            if (q === ci) box(x, rowY, C.a, 0.5, 'S');
            else if (q === ci - 1) box(x, rowY, C.b, 0.45, '');
            else if (fixd) box(x, rowY, C.muted, 0.12, '');
            else { box(x, rowY, C.c, 0.45, '!'); presos++; }
          }
          let y2 = rowY + bs + 30;
          y2 = para(fixd ? 'stale_state_block_idxs: every superseded state is freed' : 'single slot last_state_block_idx: overwritten with N+1 in flight', x0, y2, { size: 12, color: fixd ? C.a : C.c, weight: 600 }) + 4;
          y2 = para(fixd ? 'max. reservation = 1 + 2 (batches in flight) + 7 (speculative) = 10' : 'blocks stuck until the end of the request: ' + presos + ' (illustrative)', x0, y2, { size: 12, font: 'mono', color: C.ink }) + 4;
          para(fixd ? 'before: 2 + 7 = 9, not enough with 2 batches in flight' : 'with 7,168-token chunks crossing 3,584 blocks', x0, y2, { size: 11.5, color: C.muted });
        } else if (i === 5) {
          const K = 8, acc = 5; const sw = Math.min(56, (x1 - x0) / K - 8);
          V.text(g, narrow ? 'decode: 1 token + 7 drafts in one step' : 'decode: 1 token + 7 drafts verified in a single step', x0, 24, { size: 12.5, color: C.ink, weight: 600 });
          const ph2 = loop(5);
          for (let k = 0; k < K; k++) {
            const x = x0 + k * (sw + 8), y = 50;
            const accepted = k < acc;
            V.box(g, x, y, sw, 26, { r: 6, fill: V.alpha(accepted ? C.a : C.c, 0.25), stroke: accepted ? C.a : C.c, lw: 1.2 });
            V.text(g, k === 0 ? 'x' : 'd' + k, x + sw / 2, y + 13.5, { size: 11.5, font: 'mono', align: 'center', base: 'middle', color: C.ink });
            V.arrow(g, x + sw / 2, y + 30, x + sw / 2, y + 62, { color: V.alpha(C.muted, 0.7), lw: 1.2, head: 5 });
            const chosen = k === acc - 1; const fade = ph2 > 0.5 && !chosen ? 0.25 : 1;
            V.box(g, x + sw / 2 - bs / 2, y + 66, bs, bs, { r: 4, fill: V.alpha(chosen ? C.a : C.b, 0.4 * fade), stroke: V.alpha(chosen ? C.a : C.b, fade), lw: chosen ? 2.6 : 1 });
            V.text(g, 'S' + (k + 1), x + sw / 2, y + 66 + bs + 15, { size: 11, font: 'mono', align: 'center', color: C.ink, alpha: fade });
          }
          let y5 = para('accepted: x, d1…d4 (green) · rejected: d5…d7 (pink)', x0, 190, { size: 12, color: C.muted }) + 6;
          y5 = para('num_accepted_tokens = ' + acc + '  →  S' + acc + ' becomes the official state', x0, y5, { size: 12.5, font: 'mono', color: C.a, weight: 600 }) + 6;
          para('in align mode: num_speculative_blocks = 7 extra pages per request', x0, y5, { size: 12, color: C.ink });
        } else {
          prompt(TOT);
          const yb = py + ph + 26;
          ENDS.forEach((e) => box(at(e), yb, C.b, 0.45, '✓'));
          const y2 = yb + bs + 44, L2 = 8004, hit = 7168;
          V.text(g, 'request 2: same first 8,004 tokens', x0, y2 - 8, { size: 12, color: C.ink, weight: 600 });
          const a = V.ease.out(Math.min(1, el / 1.5));
          V.box(g, x0, y2, Math.max(4, (X(hit) - x0) * a), 20, { r: 5, fill: V.alpha(C.b, 0.45), stroke: C.b, lw: 1 });
          if (a > 0.95) V.box(g, X(hit), y2, X(L2) - X(hit), 20, { r: 5, fill: V.alpha(C.c, 0.45), stroke: C.c, lw: 1 });
          V.line(g, X(hit), yb + bs, X(hit), y2, { color: C.b, lw: 1.4, dash: [3, 3] });
          V.text(g, 'reuses 7,168', x0 + 8, y2 + 10.5, { size: 11, font: 'mono', base: 'middle', color: C.ink });
          V.text(g, '+ 836 recomputed', X(L2) + 8, y2 + 10.5, { size: 11, font: 'mono', base: 'middle', color: C.c });
          para('hit = last checkpoint ≤ common prefix (minimum across groups)', x0, y2 + 48, { size: 12, color: C.b, weight: 600 });
        }
      });
    });

    /* ---------------------------------------------------------------------
       Helpers for the TensorFold figures (update section)
       --------------------------------------------------------------------- */
    const DEC = '.';
    const tfc = (C) => C.__tf || (C.__tf = (getComputedStyle(document.documentElement).getPropertyValue('--tf') || '').trim() || C.d);
    const f2 = (v) => { const r = Math.round(v * 100) / 100; if (Math.abs(r) < 0.005) return '0'; return r.toFixed(2).replace('.', DEC).replace('-', '−'); };
    const SUP = '⁰¹²³⁴⁵⁶⁷⁸⁹';
    const sci = (x) => { if (!(x > 0)) return '0'; const e = Math.floor(Math.log10(x)); const m = x / Math.pow(10, e); return m.toFixed(1).replace('.', DEC) + '·10' + (e < 0 ? '⁻' : '') + String(Math.abs(e)).replace(/\d/g, (d) => SUP[d]); };
    const SUB = '₀₁₂₃₄₅₆₇₈₉';
    const sub = (n) => String(n).replace(/\d/g, (d) => SUB[d]);
    // matrix with values; o.fmt formats, o.mask hides, o.reveal(a,b) ∈ [0,1] animates, o.col(v) picks the color
    function tmat(g, x, y, M, cs, o = {}) {
      const C = V.C(); const R = M.length, K = M[0].length, gp = 3; const mx = o.max || 2;
      for (let a = 0; a < R; a++) for (let b = 0; b < K; b++) {
        const xx = x + b * (cs + gp), yy = y + a * (cs + gp);
        if (o.mask && o.mask(a, b)) { g.fillStyle = V.alpha(C.muted, 0.07); V.rr(g, xx, yy, cs, cs, Math.min(5, cs / 5)); g.fill(); continue; }
        const v = M[a][b]; const rv = o.reveal ? V.clamp(o.reveal(a, b), 0, 1) : 1;
        const tt = V.clamp(Math.abs(v) / mx, 0, 1);
        const base = o.col ? o.col(v, a, b) : (v >= 0 ? C.a : C.c);
        g.fillStyle = V.alpha(base, (0.1 + 0.5 * tt) * (0.25 + 0.75 * rv));
        V.rr(g, xx, yy, cs, cs, Math.min(5, cs / 5)); g.fill();
        g.strokeStyle = V.alpha(C.muted, 0.3); g.lineWidth = 1; g.stroke();
        if (rv > 0.3 && cs >= 27) V.text(g, (o.fmt || fmtv)(v), xx + cs / 2, yy + cs / 2 + 0.5, { size: 11, font: 'mono', align: 'center', base: 'middle', color: C.ink, alpha: Math.min(1, rv * 1.3) });
      }
      const Wm = K * (cs + gp) - gp, Hm = R * (cs + gp) - gp;
      (o.hi || []).forEach(([a, b, colr]) => { g.strokeStyle = colr || C.d; g.lineWidth = 2.4; V.rr(g, x + b * (cs + gp) - 2, y + a * (cs + gp) - 2, cs + 4, cs + 4, 6); g.stroke(); });
      if (o.hiRow != null) { g.strokeStyle = o.hiColor || C.d; g.lineWidth = 2.2; V.rr(g, x - 3, y + o.hiRow * (cs + gp) - 3, Wm + 6, cs + 6, 7); g.stroke(); }
      if (o.hiCol != null) { g.strokeStyle = o.hiColor || C.d; g.lineWidth = 2.2; V.rr(g, x + o.hiCol * (cs + gp) - 3, y - 3, cs + 6, Hm + 6, 7); g.stroke(); }
      if (o.label) V.text(g, o.label, x + Wm / 2, y - (o.labelDy || 8), { size: 11.5, font: 'mono', align: 'center', color: o.labelColor || C.ink, weight: 600 });
      return { w: Wm, h: Hm };
    }
    // places blocks (each with r rows and c columns) side by side or stacked, centered
    function tlay(blocks, x0, W, top, bot, o = {}) {
      const gx = o.gx || 30, gy = o.gy || 30, lab = 22, cap = o.cap || 40, n = blocks.length;
      const totC = blocks.reduce((s, b) => s + b.c, 0), maxR = Math.max(...blocks.map((b) => b.r));
      const csH = Math.min(cap, (W - gx * (n - 1)) / totC - 3, (bot - top - lab) / maxR - 3);
      const maxC = Math.max(...blocks.map((b) => b.c)), totR = blocks.reduce((s, b) => s + b.r, 0);
      const csV = Math.min(cap, W / maxC - 3, (bot - top - lab * n - gy * (n - 1)) / totR - 3);
      const side = !o.stack && (csH >= 27 || csH >= csV);
      const out = [];
      if (side) {
        const cs = Math.floor(csH); const tw0 = blocks.reduce((s, b) => s + b.c * (cs + 3) - 3, 0) + gx * (n - 1);
        const dy = Math.max(0, (bot - top - lab - maxR * (cs + 3)) * 0.45);
        let x = x0 + Math.max(0, (W - tw0) / 2);
        blocks.forEach((b) => { out.push({ x, y: top + lab + dy, cs }); x += b.c * (cs + 3) - 3 + gx; });
      } else {
        const cs = Math.floor(csV);
        const used = blocks.reduce((s, b) => s + b.r * (cs + 3) - 3 + lab, 0) + gy * (n - 1);
        let y = top + lab + Math.max(0, (bot - top - used) * 0.45);
        blocks.forEach((b) => { const bw = b.c * (cs + 3) - 3; out.push({ x: x0 + Math.max(0, (W - bw) / 2), y, cs }); y += b.r * (cs + 3) - 3 + gy + lab; });
      }
      out.side = side;
      return out;
    }

    /* =====================================================================
       Figure 10 — TensorFold: one sub-chunk in WY/UT form, with numbers
       ===================================================================== */
    safe('fig-tf-wy', () => {
      const fig = root.querySelector('#fig-tf-wy');
      autoH(fig, 470, 580);
      const TX = {
        steps: [
          { t: 'Inputs: one sub-chunk and the incoming state', d: 'The sub-chunk has \\(C\\) rows (32 in the real kernel; here 4 to 8, with \\(d_k = d_v = 4\\)). Each row brings \\(q\\) and \\(k\\) L2-normalized, a value \\(v\\) and a write strength \\(\\beta\\). The state \\(S_0\\) arrives ready from the previous sub-chunk, and it is the only thing that had to wait.' },
          { t: 'Decays as running products', d: 'Each row decays the state channel by channel by \\(\\alpha \\in (e^{-5}, 1)\\). The weight that row \\(t\\)\'s write still has at row \\(s\\) is \\(E_c(t\\to s) = \\alpha_{t+1,c}\\cdots\\alpha_{s,c}\\), multiplied in row order, as the serial chain does. Every factor is \\(\\le 1\\). The matrix on the right shows one channel at a time.' },
          { t: 'L and M: who interferes with whom', d: '\\(L_{st}\\) measures how much row \\(t\\)\'s write changes what row \\(s\\)\'s key reads: this is how one row\'s delta rule depends on the earlier ones. \\(M_{st}\\) is the causal attention inside the block, how much row \\(s\\)\'s query sees of row \\(t\\)\'s write. Both already include the decays.' },
          { t: 'T = (I + L)⁻¹ by forward substitution', d: '\\(I + L\\) is lower triangular with a unit diagonal, so the inverse comes out row by row: row \\(s\\) of \\(T\\) is \\(e_s\\) minus the earlier rows of \\(T\\) weighted by \\(L_{st}\\). No pivoting and no division. The real kernel does it in 16-row blocks.' },
          { t: 'The delta-rule corrections: V′ = T·Y', d: "\\(Y\\) is the error each row would write if it were the block's first: \\(\\beta\\,(v - \\text{what } S_0 \\text{ already answers})\\). \\(T\\) chains the corrections, and each row discounts what the block's earlier rows have already written. Row 0 does not change." },
          { t: 'Outputs and final state, checked against the serial chain', d: "\\(O = (Q\\odot E)\\,S_0 + M V'\\) comes out at once, while the serial chain needs one row at a time. \\(S_{\\text{end}}\\) moves on to the next sub-chunk. In float64 the two forms match up to rounding. In FP32, with a different order of sums, the real kernel stays close to the chain, but not on the same bits." },
        ],
        rows: 'rows', dice: '🎲 new numbers', rO: 'max |O − serial|', rS: 'max |S − serial|', row: 'row',
        kLab: 'k (L2)', vLab: 'v', bLab: 'β', s0Lab: 'S₀ (incoming)', aLab: 'α per channel', eLab: (c) => 'E(t→s), channel ' + c,
        lLab: 'L (t < s)', mLab: 'M (t ≤ s)', ilLab: 'I + L', tLab: 'T = (I + L)⁻¹', yLab: 'Y', vpLab: "V′ = T·Y",
        oLab: 'O (chunkwise)', osLab: 'O (serial chain)', seLab: 'S_end',
        top: [
          'state S₀ arrives from the previous sub-chunk',
          'running product: every factor ≤ 1, no e^(+x)',
          'L: interference between keys · M: attention inside the block',
          'Tₛ = eₛ − Σ Lₛₜ · Tₜ  (sum over t < s)',
          'the delta rule writes the corrected error, not the raw v',
          'chunkwise: all at once · serial: one row at a time',
        ],
        b0: 'β = σ(b) ∈ (0, 1) · α = e^g with g = −5·σ(·) ∈ (−5, 0)',
        b1: (t, s, c, fs, v) => 'E' + sub(c) + '(' + t + '→' + s + ') = ' + (fs.length > 1 ? fs.join(' × ') + ' = ' : '') + v,
        b2: (s, t, l, m) => 'L' + sub(s) + sub(t) + ' = ' + l + '   ·   M' + sub(s) + sub(t) + ' = ' + m,
        b3: (s) => 'computing row ' + s + ' of T from rows 0…' + Math.max(0, s - 1),
        b4: (s, y, vp) => 'row ' + s + ':  Y = [' + y + ']  →  V′ = [' + vp + ']',
        ok: (e) => '✓ max |O − serial| = ' + e + ' (float64)',
        b5: 'on the GB10, in FP32: close to the serial chain, not the same bits',
      };
      const st = UI.stepper(fig, TX.steps, { autoplay: 9000 });
      const ctl = UI.controls(fig);
      const S = { T: 6, seed: 1 };
      let D = null;
      UI.seg(ctl, { label: TX.rows, options: [4, 6, 8], value: S.T, onChange: (v) => { S.T = v; build(); } });
      UI.button(ctl, TX.dice, () => { S.seed++; build(); });
      const rO = UI.readout(ctl, TX.rO, '—'), rS = UI.readout(ctl, TX.rS, '—');
      function build() {
        const r = V.rng(S.seed * 7919 + S.T * 31); const T = S.T, Dk = 4; const R4 = [0, 1, 2, 3];
        const unit = (x) => { const n = Math.hypot(...x) || 1; return x.map((v) => v / n); };
        const rows = [];
        for (let s = 0; s < T; s++) {
          const q = unit(R4.map(() => r() * 2 - 1)), k = unit(R4.map(() => r() * 2 - 1));
          const v = R4.map(() => Math.round((r() * 4 - 2) * 10) / 10);
          const b = Math.round((0.3 + 0.65 * r()) * 100) / 100;
          const a = R4.map(() => Math.exp(-5 / (1 + Math.exp(4.5 - 3.3 * r()))));
          rows.push({ q, k, v, b, a });
        }
        const S0 = R4.map(() => R4.map(() => Math.round((r() * 3 - 1.5) * 10) / 10));
        const E = (t, s, c) => { let p = 1; for (let u = t + 1; u <= s; u++) p *= rows[u].a[c]; return p; };
        const Es = rows.map((_, s) => R4.map((c) => E(-1, s, c)));
        const Eto = rows.map((_, s) => R4.map((c) => E(s, T - 1, c)));
        const L = rows.map(() => Array(T).fill(0)), M = rows.map(() => Array(T).fill(0));
        for (let s = 0; s < T; s++) for (let t = 0; t <= s; t++) {
          let l = 0, m = 0; for (let c = 0; c < Dk; c++) { const e = E(t, s, c); l += rows[s].k[c] * rows[t].k[c] * e; m += rows[s].q[c] * rows[t].k[c] * e; }
          if (t < s) L[s][t] = rows[s].b * l; M[s][t] = m;
        }
        const IL = L.map((row, s) => row.map((x, t) => (s === t ? 1 : x)));
        const Ti = rows.map(() => Array(T).fill(0));
        for (let s = 0; s < T; s++) for (let j = 0; j < T; j++) { let x = s === j ? 1 : 0; for (let t = 0; t < s; t++) x -= L[s][t] * Ti[t][j]; Ti[s][j] = x; }
        const Y = rows.map((x, s) => R4.map((j) => { let u = 0; for (let c = 0; c < Dk; c++) u += x.k[c] * Es[s][c] * S0[c][j]; return x.b * (x.v[j] - u); }));
        const Vp = rows.map((_, s) => R4.map((j) => { let u = 0; for (let t = 0; t < T; t++) u += Ti[s][t] * Y[t][j]; return u; }));
        const O = rows.map((x, s) => R4.map((j) => { let u = 0; for (let c = 0; c < Dk; c++) u += x.q[c] * Es[s][c] * S0[c][j]; for (let t = 0; t <= s; t++) u += M[s][t] * Vp[t][j]; return u; }));
        const Se = R4.map((c) => R4.map((j) => { let u = Es[T - 1][c] * S0[c][j]; for (let t = 0; t < T; t++) u += rows[t].k[c] * Eto[t][c] * Vp[t][j]; return u; }));
        // serial chain: S ← Diag(α)S; S ← S + β k (v − Sᵀk)ᵀ; o = Sᵀq
        const Ss = S0.map((x) => x.slice()); const Os = [];
        rows.forEach((x) => {
          for (let c = 0; c < Dk; c++) for (let j = 0; j < Dk; j++) Ss[c][j] *= x.a[c];
          const vh = R4.map((j) => { let u = 0; for (let c = 0; c < Dk; c++) u += Ss[c][j] * x.k[c]; return u; });
          for (let c = 0; c < Dk; c++) for (let j = 0; j < Dk; j++) Ss[c][j] += x.b * x.k[c] * (x.v[j] - vh[j]);
          Os.push(R4.map((j) => { let u = 0; for (let c = 0; c < Dk; c++) u += Ss[c][j] * x.q[c]; return u; }));
        });
        let eo = 0, es = 0;
        for (let s = 0; s < T; s++) for (let j = 0; j < Dk; j++) eo = Math.max(eo, Math.abs(O[s][j] - Os[s][j]));
        for (let c = 0; c < Dk; c++) for (let j = 0; j < Dk; j++) es = Math.max(es, Math.abs(Se[c][j] - Ss[c][j]));
        const Ec = R4.map((c) => rows.map((_, s) => rows.map((__, t) => (t <= s ? E(t, s, c) : 0))));
        D = { T, rows, S0, Ec, L, M, IL, Ti, Y, Vp, O, Os, Se, eo, es,
          K: rows.map((x) => x.k), Vv: rows.map((x) => x.v), B: rows.map((x) => [x.b]), A: rows.map((x) => x.a) };
        rO.set(sci(eo), 'ok'); rS.set(sci(es), 'ok');
      }
      build();
      const lower = (s, t) => t > s, strict = (s, t) => t >= s;
      V.fig(fig, (g, w, h) => {
        if (w < 80 || h < 60 || !D) return;
        const C = V.C(); const TFC = tfc(C); const i = st.i; const el = st.elapsed(); const T = D.T;
        const x0 = 16, x1 = w - 16, W = x1 - x0; const narrow = w < 600;
        // top line (step summary)
        const topL = wrap(g, TX.top[i], W, 12, 'mono', 600);
        topL.forEach((ln, q) => V.text(g, ln, x0, 18 + q * 16, { size: 12, font: 'mono', weight: 600, color: TFC }));
        const top = 26 + topL.length * 16, bot = h - (narrow ? 58 : 44);
        const bottom = (s, col) => { const L2 = wrap(g, s, W, 12, 'mono'); L2.slice(0, 3).forEach((ln, q) => V.text(g, ln, x0, h - (narrow ? 40 : 26) + q * 15 - (L2.length > 1 ? 8 : 0), { size: 12, font: 'mono', color: col || C.ink })); };
        const rowLabels = (p, n) => { if (p.x - x0 < 22) return; for (let s = 0; s < n; s++) V.text(g, String(s), p.x - 8, p.y + s * (p.cs + 3) + p.cs / 2 + 0.5, { size: 11, font: 'mono', align: 'right', base: 'middle', color: C.muted }); };
        const cyc = (per, n) => Math.floor(el / per) % Math.max(1, n);
        if (i === 0) {
          // table of rows (k | v | β) and the incoming state
          // the k | v | β table takes 9 columns + 2 gaps (≈ 1 extra column)
          const P = tlay([{ r: T, c: 10 }, { r: 4, c: 4 }], x0, W, top, bot, { gx: 44, cap: 38 });
          const p = P[0], cs = p.cs, gpx = Math.min(14, (cs + 3) / 2);
          const kx = p.x, vx = kx + 4 * (cs + 3) - 3 + gpx, bx = vx + 4 * (cs + 3) - 3 + gpx;
          const rev = (a) => V.clamp(el * 2.2 - a * 0.35, 0, 1);
          tmat(g, kx, p.y, D.K, cs, { label: TX.kLab, max: 1, reveal: rev });
          tmat(g, vx, p.y, D.Vv, cs, { label: TX.vLab, max: 2, reveal: rev });
          tmat(g, bx, p.y, D.B, cs, { label: TX.bLab, max: 1, fmt: f2, col: () => TFC, reveal: rev });
          rowLabels({ x: kx, y: p.y, cs }, T);
          const q = P[1];
          tmat(g, q.x, q.y, D.S0, q.cs, { label: TX.s0Lab, max: 1.5, labelColor: TFC, labelDy: 13 });
          g.strokeStyle = V.alpha(TFC, 0.8); g.lineWidth = 2; V.rr(g, q.x - 5, q.y - 5, 4 * (q.cs + 3) + 7, 4 * (q.cs + 3) + 7, 8); g.stroke();
          bottom(TX.b0, C.muted);
        } else if (i === 1) {
          const c = cyc(2.4, 4);
          const P = tlay([{ r: T, c: 4 }, { r: T, c: T }], x0, W, top, bot, { gx: 40, cap: 40 });
          const pairs = []; for (let s = 1; s < T; s++) for (let t = 0; t < s; t++) pairs.push([s, t]);
          const [hs, ht] = pairs[cyc(0.9, pairs.length)];
          const hiA = []; for (let u = ht + 1; u <= hs; u++) hiA.push([u, c, TFC]);
          tmat(g, P[0].x, P[0].y, D.A, P[0].cs, { label: TX.aLab, max: 1, fmt: f2, col: () => C.a, hi: hiA });
          rowLabels(P[0], T);
          const Ec = D.Ec[c];
          tmat(g, P[1].x, P[1].y, Ec, P[1].cs, { label: TX.eLab(c), labelColor: TFC, max: 1, fmt: f2, col: () => TFC, mask: lower, hi: [[hs, ht, C.ink]], reveal: (a) => V.clamp(el * 3 - a * 0.25, 0, 1) });
          const fs = []; for (let u = ht + 1; u <= hs; u++) fs.push(f2(D.A[u][c]));
          bottom(TX.b1(ht, hs, c, fs.length > 4 ? fs.slice(0, 2).concat(['…', fs[fs.length - 1]]) : fs, f2(Ec[hs][ht])));
        } else if (i === 2) {
          const P = tlay([{ r: T, c: T }, { r: T, c: T }], x0, W, top, bot, { gx: 40, cap: 40 });
          const pairs = []; for (let s = 1; s < T; s++) for (let t = 0; t < s; t++) pairs.push([s, t]);
          const [hs, ht] = pairs[cyc(1.1, pairs.length)];
          tmat(g, P[0].x, P[0].y, D.L, P[0].cs, { label: TX.lLab, max: 0.8, mask: strict, hi: [[hs, ht, TFC]] });
          rowLabels(P[0], T);
          tmat(g, P[1].x, P[1].y, D.M, P[1].cs, { label: TX.mLab, max: 1, mask: lower, hi: [[hs, ht, TFC]] });
          bottom(TX.b2(hs, ht, f2(D.L[hs][ht]), f2(D.M[hs][ht])));
        } else if (i === 3) {
          const P = tlay([{ r: T, c: T }, { r: T, c: T }], x0, W, top, bot, { gx: 40, cap: 40 });
          const per = 0.9, loopT = T * per + 2.2; const tt = el % loopT; const cur = Math.min(T - 1, Math.floor(tt / per));
          tmat(g, P[0].x, P[0].y, D.IL, P[0].cs, { label: TX.ilLab, max: 1, mask: lower, hiRow: cur, hiColor: TFC });
          rowLabels(P[0], T);
          tmat(g, P[1].x, P[1].y, D.Ti, P[1].cs, { label: TX.tLab, labelColor: TFC, max: 1.2, mask: lower, reveal: (a) => (a < cur ? 1 : a === cur ? V.clamp((tt - cur * per) / (per * 0.6), 0, 1) : 0), hiRow: cur, hiColor: TFC });
          if (cur > 0) { const p = P[1]; g.strokeStyle = V.alpha(C.b, 0.7); g.lineWidth = 1.5; g.setLineDash([4, 3]); V.rr(g, p.x - 3, p.y - 3, T * (p.cs + 3) + 3, cur * (p.cs + 3) + 3, 7); g.stroke(); g.setLineDash([]); }
          bottom(TX.b3(cur));
        } else if (i === 4) {
          const P = tlay([{ r: T, c: 4 }, { r: T, c: 4 }], x0, W, top, bot, { gx: 70, cap: 40 });
          const per = 0.8, loopT = T * per + 2; const tt = el % loopT; const cur = Math.min(T - 1, Math.floor(tt / per));
          tmat(g, P[0].x, P[0].y, D.Y, P[0].cs, { label: TX.yLab, max: 1.5, hiRow: cur, hiColor: C.muted });
          rowLabels(P[0], T);
          tmat(g, P[1].x, P[1].y, D.Vp, P[1].cs, { label: TX.vpLab, labelColor: TFC, max: 1.5, reveal: (a) => (a <= cur ? 1 : 0.1), hiRow: cur, hiColor: TFC });
          if (P.side) { const ax = P[0].x + 4 * (P[0].cs + 3) + 6, bx2 = P[1].x - 8, yy = P[0].y + cur * (P[0].cs + 3) + P[0].cs / 2; V.arrow(g, ax, yy, bx2, yy, { color: TFC, lw: 1.8, head: 7 }); V.text(g, '× T', (ax + bx2) / 2, yy - 9, { size: 11.5, font: 'mono', align: 'center', color: TFC, weight: 600 }); }
          bottom(TX.b4(cur, D.Y[cur].map(fmtv).join(' '), D.Vp[cur].map(fmtv).join(' ')));
        } else {
          const blocks = narrow ? [{ r: T, c: 4 }, { r: T, c: 4 }] : [{ r: T, c: 4 }, { r: T, c: 4 }, { r: 4, c: 4 }];
          const P = tlay(blocks, x0, W, top, bot - 22, { gx: 34, cap: 38 });
          const per = 0.45, sr = Math.min(T, Math.floor(el / per) + 1);
          tmat(g, P[0].x, P[0].y, D.O, P[0].cs, { label: TX.oLab, labelColor: TFC, max: 1.5, reveal: () => V.clamp(el * 2.5, 0, 1) });
          rowLabels(P[0], T);
          tmat(g, P[1].x, P[1].y, D.Os, P[1].cs, { label: TX.osLab, max: 1.5, reveal: (a) => (a < sr ? 1 : 0.1), hiRow: sr <= T && el < T * per + 0.4 ? sr - 1 : null, hiColor: C.b });
          if (!narrow) tmat(g, P[2].x, P[2].y, D.Se, P[2].cs, { label: TX.seLab, labelColor: TFC, max: 1.5 });
          if (el > T * per) V.text(g, TX.ok(sci(D.eo)), x0, bot - 10, { size: 12.5, font: 'mono', weight: 600, color: C.a, maxW: W });
          bottom(TX.b5, C.muted);
        }
      });
    });

    /* =====================================================================
       Figure 11 — TensorFold: prompt grid and two parities
       ===================================================================== */
    safe('fig-tf-estado', () => {
      const fig = root.querySelector('#fig-tf-estado');
      autoH(fig, 430, 480);
      const TX = {
        mode: 'Mode', grade: 'prompt grid', par: 'parities + replay',
        n: 'prompt', next: 'second request', ext: 'extends the prompt', same: 'repeats the prompt',
        rT: 'TensorFold recomputes', rV: 'vLLM recomputes', accA: 'accepted in A', accB: 'accepted in B', rA: 'stream A', rB: 'stream B',
        tok: ' tok', flip: 'flips the parity', replay: (k) => 'replay of ' + k + (k === 1 ? ' row' : ' rows'),
        gTitle: (n) => nf(n) + '-token prompt', gNew: 'new tokens',
        gridTF: 'TF grid: 64', ckV: 'vLLM checkpoints: 3,584',
        zoom: 'zoom: 32-row sub-chunks at absolute positions · 64 grid',
        bn: (B, n) => B === n ? 'B = n = ' + nf(n) + ': the end falls on the grid · state + logits row' : 'B = ' + nf(B) + ': kept state · n = ' + nf(n) + ': off the grid, nothing kept at the end',
        rowV: 'vLLM · align checkpoints every 3,584', rowT: 'TensorFold · 64 grid + replay',
        reuse: 'reuses', redo: 'recomputes', geq: '≥ ',
        noteExt: 'The second request starts with the same n tokens and continues. TensorFold resumes from the last grid point; vLLM, from the last multiple of 3,584.',
        noteSame: (on, r) => on ? 'Identical request with n on the grid: TensorFold samples the 1st token from the kept logits row and does no prefill (replay); vLLM recomputes at least the tail after the last checkpoint.' : 'Identical request, n off the grid: TensorFold resumes from B and recomputes only ' + r + ' tokens; with n a multiple of 64 (⇥ button) it would be a replay, with no prefill.',
        snap: '⇥ n to the grid',
        pTitle: 'one verification window: 2 streams, 10 rows, 1 call',
        ph: ['1 · verify: read P0, write P1', '2 · keyed sample and compare', '3 · commit', 'official state now in P1'],
        segA: 'stream A · slot 0', segB: 'stream B · slot 1', off: 'official',
        tabH: ['seg', '1st', 'rows', 'slot', 'par', 'conv', 'keep'],
        scratch: 'k v g β saved per row',
        cmpV: 'vLLM: writes the state after each verified row (k + 1 candidates; 7 extra pages in GLM) and picks one',
        cmpT: 'TensorFold: two copies per stream; replay of the accepted rows only when there is a rejection',
      };
      const ctl = UI.controls(fig);
      const S = { mode: 'grade', n: 10000, same: 0, accA: 3, accB: 2, t0: 0 };
      let tNow = 0;
      UI.seg(ctl, { label: TX.mode, options: [{ v: 'grade', label: TX.grade }, { v: 'par', label: TX.par }], value: S.mode, onChange: (v) => { S.mode = v; gG.show(v === 'grade'); gP.show(v === 'par'); S.t0 = tNow; upd(); } });
      const gG = UI.group(ctl), gP = UI.group(ctl);
      const sN = UI.slider(gG.el, { label: TX.n, min: 1000, max: 20000, step: 1, value: S.n, fmt: (v) => nf(v) + TX.tok, onInput: (v) => { S.n = v; S.t0 = tNow; upd(); } });
      UI.button(gG.el, TX.snap, () => sN.set(Math.floor(S.n / GR) * GR));
      UI.seg(gG.el, { label: TX.next, options: [{ v: 0, label: TX.ext }, { v: 1, label: TX.same }], value: 0, onChange: (v) => { S.same = v; S.t0 = tNow; upd(); } });
      const rT = UI.readout(gG.el, TX.rT, '—'), rV = UI.readout(gG.el, TX.rV, '—');
      UI.slider(gP.el, { label: TX.accA, min: 0, max: 3, step: 1, value: S.accA, onInput: (v) => { S.accA = v; S.t0 = tNow; upd(); } });
      UI.slider(gP.el, { label: TX.accB, min: 0, max: 5, step: 1, value: S.accB, onInput: (v) => { S.accB = v; S.t0 = tNow; upd(); } });
      const rA = UI.readout(gP.el, TX.rA, '—'), rB = UI.readout(gP.el, TX.rB, '—');
      gP.show(false);
      const BLK = 3584, GR = 64;
      function calc() {
        const n = S.n, extra = S.same ? 0 : Math.round(Math.max(600, n * 0.14)), N2 = n + extra;
        const B = Math.floor(n / GR) * GR;
        const vk = S.same ? Math.floor((n - 1) / BLK) * BLK : Math.floor(n / BLK) * BLK;
        const tfUse = S.same && B === n ? n : B;   // the prompt's end is only kept (with the logits) when it falls on the grid
        return { n, extra, N2, B, vk, tfUse, tfRe: N2 - tfUse, vRe: N2 - vk };
      }
      function upd() {
        const c = calc();
        rT.set(nf(c.tfRe) + TX.tok, 'ok'); rV.set(TX.geq + nf(c.vRe) + TX.tok, 'warn');
        const kA = 1 + S.accA, kB = 1 + S.accB;
        rA.set(kA === 4 ? TX.flip : TX.replay(kA), kA === 4 ? 'ok' : 'warn');
        rB.set(kB === 6 ? TX.flip : TX.replay(kB), kB === 6 ? 'ok' : 'warn');
      }
      upd();
      function hatch(g, x, y, w2, h2, col) { if (w2 < 2) return; g.save(); g.beginPath(); g.rect(x, y, w2, h2); g.clip(); g.strokeStyle = col; g.lineWidth = 1; for (let k = -h2; k < w2 + h2; k += 6) { g.beginPath(); g.moveTo(x + k, y + h2); g.lineTo(x + k + h2, y); g.stroke(); } g.restore(); }
      function mini(g, x, y, cs, seed, a) {
        const C = V.C();
        for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) {
          const v = Math.sin(r * 1.7 + c * 0.9 + seed * 1.3) * 0.6 + Math.sin(r * 0.5 - c * 1.2 + seed * 0.7) * 0.4;
          g.fillStyle = V.alpha(v >= 0 ? C.a : C.c, (0.15 + 0.55 * Math.abs(v)) * a);
          V.rr(g, x + c * (cs + 2), y + r * (cs + 2), cs, cs, 2); g.fill();
        }
      }
      V.fig(fig, (g, w, h, t) => {
        tNow = t;
        if (w < 80 || h < 60) return;
        const C = V.C(); const TFC = tfc(C); const narrow = w < 560;
        const x0 = 16, x1 = w - 16, W = x1 - x0;
        const para = (s, x, y, o) => { const sz = o.size || 12; const Ls = wrap(g, s, o.maxW || (x1 - x), sz, o.font || 'sans', o.weight || 400); Ls.forEach((ln, q) => V.text(g, ln, x, y + q * (sz + 4), Object.assign({}, o, { size: sz }))); return y + Ls.length * (sz + 4); };
        const el = t - S.t0;
        if (S.mode === 'grade') {
          const c = calc(); const X = (p) => x0 + (p / c.N2) * W;
          V.text(g, TX.gTitle(c.n), x0, 18, { size: 12.5, weight: 600, color: C.ink });
          const by = 46, bh = 22;
          V.box(g, x0, by, X(c.n) - x0, bh, { r: 6, fill: V.alpha(C.a, 0.2), stroke: V.alpha(C.a, 0.7), lw: 1 });
          if (c.extra) { hatch(g, X(c.n), by, x1 - X(c.n), bh, V.alpha(C.muted, 0.5)); g.strokeStyle = V.alpha(C.muted, 0.6); g.lineWidth = 1; V.rr(g, X(c.n), by, x1 - X(c.n), bh, 6); g.stroke(); if (x1 - X(c.n) > 70) V.text(g, TX.gNew, (X(c.n) + x1) / 2, by + bh / 2 + 0.5, { size: 11, align: 'center', base: 'middle', color: C.muted }); }
          for (let p = GR; p < c.n; p += GR) V.line(g, X(p), by - 6, X(p), by - 1, { color: V.alpha(TFC, 0.55), lw: 1 });
          for (let p = BLK; p < c.N2; p += BLK) V.line(g, X(p), by + bh + 1, X(p), by + bh + 9, { color: C.glm, lw: 2 });
          V.text(g, TX.gridTF, x1, by - 10, { size: 11, align: 'right', color: TFC, weight: 600 });
          V.text(g, TX.ckV, x0, by + bh + 22, { size: 11, color: C.glm, weight: 600 });
          // zoom
          const zA = Math.max(0, c.B - 128), zW = 256, zy = by + bh + 50, zh = 34;
          const XZ = (p) => x0 + ((p - zA) / zW) * W;
          g.strokeStyle = V.alpha(TFC, 0.5); g.lineWidth = 1; g.setLineDash([3, 3]);
          g.beginPath(); g.moveTo(X(zA), by + bh); g.lineTo(x0, zy - 4); g.moveTo(X(Math.min(c.N2, zA + zW)), by + bh); g.lineTo(x1, zy - 4); g.stroke(); g.setLineDash([]);
          g.strokeStyle = TFC; g.lineWidth = 1.6; V.rr(g, X(zA) - 1, by - 2, Math.max(3, X(Math.min(c.N2, zA + zW)) - X(zA) + 2), bh + 4, 3); g.stroke();
          const loop = 4.2, ph = (el % loop) / 3; const sweep = zA + Math.min(1, ph) * (c.n - zA); const xBz = XZ(c.B); const sPos = [];
          for (let k = 0; k < zW / 32; k++) {
            const p0 = zA + k * 32, p1 = p0 + 32; const inP = p0 < c.n;
            const done = p1 <= sweep, cur = p0 < sweep && sweep < p1;
            const xa = XZ(p0), xb = XZ(p1);
            const fillA = !inP ? 0.04 : done ? 0.3 : cur ? 0.18 : 0.08;
            g.fillStyle = V.alpha(inP ? TFC : C.muted, fillA); g.fillRect(xa, zy, xb - xa, zh);
            if (p0 % GR === 0) V.line(g, xa, zy - 4, xa, zy + zh + 4, { color: TFC, lw: 1.8 });
            else { g.setLineDash([3, 3]); V.line(g, xa, zy, xa, zy + zh, { color: V.alpha(TFC, 0.6), lw: 1 }); g.setLineDash([]); }
            if (done && p1 <= c.n && xb - xa > 26 && Math.abs(xb - xBz) > 4) sPos.push(xb);
          }
          sPos.forEach((xb) => { V.box(g, xb - 9, zy + zh / 2 - 9, 18, 18, { r: 9, fill: C.panel, stroke: V.alpha(TFC, 0.9), lw: 1.2 }); V.text(g, 'S', xb, zy + zh / 2 + 0.5, { size: 11, font: 'mono', align: 'center', base: 'middle', color: C.ink, weight: 700 }); });
          g.strokeStyle = V.alpha(C.muted, 0.6); g.lineWidth = 1; g.strokeRect(x0, zy, W, zh);
          if (!narrow || W > 300) for (let p = zA; p <= zA + zW; p += GR) V.text(g, nf(p), XZ(p), zy + zh + 15, { size: 11, font: 'mono', align: p === zA ? 'left' : p === zA + zW ? 'right' : 'center', color: C.muted });
          // B and n markers
          const xB = XZ(c.B), xn = XZ(c.n);
          V.line(g, xn, zy - 8, xn, zy + zh + 4, { color: C.ink, lw: 2 });
          const mk = (x, y, colr, s) => { V.box(g, x - 9, y - 9, 18, 18, { r: 4, fill: V.alpha(colr, 0.35), stroke: colr, lw: 1.5 }); V.text(g, s, x, y + 0.5, { size: 11, font: 'mono', align: 'center', base: 'middle', color: C.ink, weight: 700 }); };
          mk(xB, zy + zh / 2, TFC, 'B'); mk(xn, zy - 16, C.a, 'n');
          if (ph >= 1) { const a = 1 - V.clamp((ph - 1) / 0.4, 0, 1); V.glow(g, xB, zy + zh / 2, 26, TFC, 0.6 * a); }
          const ly = zy + zh + 34;
          let yy = para(TX.zoom, x0, ly, { size: 11, color: C.muted });
          yy = para(TX.bn(c.B, c.n), x0, yy, { size: 11.5, font: 'mono', color: TFC, weight: 600 });
          // bars for the second request
          const rowBar = (lab, col, use, re, y, geq) => {
            V.text(g, lab, x0, y, { size: 11.5, color: col, weight: 600 });
            const yb = y + 8, hb = 18; const Xb = (p) => x0 + (p / c.N2) * W;
            const grow = V.ease.out(V.clamp(el / 1.6, 0, 1));
            if (use > 0) V.box(g, x0, yb, Xb(use) - x0, hb, { r: 5, fill: V.alpha(col, 0.35), stroke: col, lw: 1 });
            if (re > 0) V.box(g, Xb(use), yb, Math.max(3, (Xb(c.N2) - Xb(use)) * grow), hb, { r: 5, fill: V.alpha(C.c, 0.4), stroke: C.c, lw: 1 });
            const s = TX.reuse + ' ' + nf(use) + ' · ' + TX.redo + ' ' + (geq ? TX.geq : '') + nf(re);
            V.text(g, s, x0, yb + hb + 15, { size: 11.5, font: 'mono', color: C.ink });
            return yb + hb + 34;
          };
          let y2 = Math.max(yy + 14, ly + 50);
          y2 = rowBar(TX.rowV, C.glm, c.vk, c.vRe, y2, true);
          y2 = rowBar(TX.rowT, TFC, c.tfUse, c.tfRe, y2, false);
          para(S.same ? TX.noteSame(c.B === c.n, nf(c.tfRe)) : TX.noteExt, x0, y2 - 4, { size: 11, color: C.muted });
        } else {
          const segs = [{ rows: 4, keep: 1 + S.accA, lab: TX.segA, slot: 0 }, { rows: 6, keep: 1 + S.accB, lab: TX.segB, slot: 1 }];
          const cyc = 7.5, tt = el % cyc; const phase = tt < 2.2 ? 0 : tt < 3.4 ? 1 : tt < 5.6 ? 2 : 3;
          const yT = para(TX.pTitle, x0, 18, { size: 12.5, weight: 600, color: C.ink });
          V.text(g, TX.ph[phase], x0, yT + 4, { size: 12, font: 'mono', weight: 600, color: TFC });
          const gapSeg = 16; const cw = Math.min(54, (W - gapSeg) / 10 - 4); const wy = yT + 40;
          const segX = [x0, x0 + 4 * (cw + 4) + gapSeg];
          const totalW = 10 * (cw + 4) - 4 + gapSeg; const off = Math.max(0, (W - totalW) / 2); segX[0] += off; segX[1] += off;
          segs.forEach((sg, si) => {
            const sx = segX[si];
            V.text(g, sg.lab, sx, wy - 10, { size: 11.5, color: si ? C.b : C.a, weight: 600 });
            const prog = phase === 0 ? tt / 2.2 : 1; const cur = Math.min(sg.rows - 1, Math.floor(prog * sg.rows));
            for (let r = 0; r < sg.rows; r++) {
              const x = sx + r * (cw + 4); let fill = V.alpha(C.muted, 0.1), stroke = V.alpha(C.muted, 0.4);
              if (phase === 0) { if (r < cur) fill = V.alpha(TFC, 0.22); if (r === cur) { fill = V.alpha(TFC, 0.4); stroke = TFC; } }
              else { const ok = r < sg.keep; const a = phase === 1 ? V.clamp((tt - 2.2) / 0.5, 0, 1) : 1; fill = V.alpha(ok ? C.a : C.c, 0.12 + 0.3 * a); stroke = ok ? C.a : C.c; }
              V.box(g, x, wy, cw, 30, { r: 6, fill, stroke, lw: 1.3 });
              V.text(g, r === 0 ? 'x' : 'r' + r, x + cw / 2, wy + 15.5, { size: 11.5, font: 'mono', align: 'center', base: 'middle', color: C.ink });
              if (phase >= 1 && cw >= 24) V.text(g, r < sg.keep ? '✓' : '✗', x + cw / 2, wy + 42, { size: 11.5, align: 'center', color: r < sg.keep ? C.a : C.c, weight: 700 });
            }
          });
          // slots with two parities
          const sy = wy + 86; const colW = W / 2; const mcs = narrow ? 7 : 11; const mW = 4 * (mcs + 2) - 2;
          // replay: the per-row saved inputs (k, v, g, β) of the kept rows
          if (phase === 2) segs.forEach((sg, si) => {
            if (sg.keep === sg.rows) return;
            const sx = segX[si], k = V.clamp((tt - 3.4) / 1.3, 0, 1), nk = Math.max(1, Math.ceil(k * sg.keep));
            for (let r = 0; r < sg.keep; r++) { g.strokeStyle = V.alpha(C.c, r < nk ? 1 : 0.35); g.lineWidth = 2; g.setLineDash([3, 3]); V.rr(g, sx + r * (cw + 4) - 3, wy - 3, cw + 6, 36, 8); g.stroke(); g.setLineDash([]); }
            V.text(g, TX.scratch, sx, wy + 58, { size: 11, color: C.c, weight: 600, maxW: W - (sx - x0) });
          });
          segs.forEach((sg, si) => {
            const cx = x0 + colW * si + colW / 2; const full = sg.keep === sg.rows;
            const p0x = cx - mW - 14, p1x = cx + 14, py = sy + 30;
            const bw2 = Math.min(colW - 8, Math.max(2 * mW + 52, 150));
            V.box(g, cx - bw2 / 2, sy, bw2, mW + 72, { r: 8, fill: V.alpha(C.muted, 0.06), stroke: V.alpha(si ? C.b : C.a, 0.6), lw: 1.2 });
            V.text(g, 'slot ' + sg.slot + ' · ' + (si ? 'B' : 'A'), cx, sy - 6, { size: 11.5, font: 'mono', align: 'center', color: si ? C.b : C.a, weight: 600 });
            const wr = phase === 0 ? V.clamp(tt / 2.2, 0, 1) : 1;
            mini(g, p0x, py, mcs, si * 3, 1);
            mini(g, p1x, py, mcs, si * 3 + 1 + wr * 2.5 + (phase >= 2 && !full ? 0.6 : 0), 0.35 + 0.65 * wr);
            V.text(g, 'P0', p0x + mW / 2, py + mW + 14, { size: 11, font: 'mono', align: 'center', color: C.muted });
            V.text(g, 'P1', p1x + mW / 2, py + mW + 14, { size: 11, font: 'mono', align: 'center', color: C.muted });
            if (phase === 0) { g.setLineDash([4, 3]); g.lineDashOffset = -t * 20; V.arrow(g, p0x + mW + 3, py + mW / 2, p1x - 4, py + mW / 2, { color: TFC, lw: 1.6, head: 6 }); g.setLineDash([]); g.lineDashOffset = 0; }
            // "official" pointer
            let pos = 0;
            if (phase === 2) { const k = V.clamp((tt - 3.4) / 2.2, 0, 1); pos = full ? V.ease.inOut(k) : V.ease.inOut(V.clamp((k - 0.55) / 0.45, 0, 1)); }
            if (phase === 3) pos = 1;
            const ox = V.lerp(p0x, p1x, pos) + mW / 2;
            V.box(g, ox - 24, py - 22, 48, 16, { r: 8, fill: V.alpha(C.a, 0.25), stroke: C.a, lw: 1 });
            V.text(g, TX.off, ox, py - 13.5, { size: 11, align: 'center', base: 'middle', color: C.ink, weight: 600 });
            if (phase === 2 && !full) {
              const k = V.clamp((tt - 3.4) / 1.3, 0, 1); const nk = Math.max(1, Math.ceil(k * sg.keep));
              g.setLineDash([3, 3]); V.arrow(g, p0x + mW + 3, py + mW / 2 + 6, p1x - 4, py + mW / 2 + 6, { color: C.c, lw: 1.6, head: 6 }); g.setLineDash([]);
              V.text(g, 'replay ' + nk + '/' + sg.keep, cx, py + mW + 30, { size: 11, font: 'mono', align: 'center', color: C.c, weight: 600 });
            } else if (phase >= 2) {
              V.text(g, full ? TX.flip : TX.replay(sg.keep), cx, py + mW + 30, { size: 11, font: 'mono', align: 'center', color: full ? C.a : C.c, weight: 600 });
            }
          });
          // segment table
          const ty = sy + mW + 96; const cols = TX.tabH.length; const tcw = Math.min(64, W / cols);
          TX.tabH.forEach((s, k) => V.text(g, s, x0 + k * tcw, ty, { size: 11, font: 'mono', color: C.muted, weight: 600 }));
          segs.forEach((sg, si) => {
            const vals = [si ? 'B' : 'A', si ? '4' : '0', String(sg.rows), String(sg.slot), '0', String(sg.slot), String(sg.keep)];
            vals.forEach((s, k) => V.text(g, s, x0 + k * tcw, ty + 17 + si * 17, { size: 11.5, font: 'mono', color: k === 6 ? (sg.keep === sg.rows ? C.a : C.c) : C.ink, weight: k === 6 ? 700 : 400 }));
          });
          let yc = ty + 17 * 2 + 22;
          yc = para(TX.cmpV, x0, yc, { size: 11.5, color: C.glm }) + 2;
          para(TX.cmpT, x0, yc, { size: 11.5, color: TFC, weight: 600 });
        }
      });
    });

    /* =====================================================================
       Figure 12 — state and block calculator
       ===================================================================== */
    safe('fig-calc', () => {
      const fig = root.querySelector('#fig-calc');
      autoH(fig, 410, 500);
      const ctl = UI.controls(fig);
      const S = { m: 'glm', tp: 2, sd: 'fp32', kv: 'fp8', spec: 7, lg: 20, c: 1 };
      const sSpec = { v: null };
      let HF = null; const rd = () => { if (HF) HF.redraw(); };
      UI.select(ctl, { label: 'Model', options: [{ v: 'glm', label: 'GLM-5.3 (KDA)' }, { v: 'qwen', label: 'Qwen3.8 (GDN)' }], value: S.m, onChange: (v) => { S.m = v; if (v === 'glm') { tpSeg.set(2); sdSeg.set('fp32'); sSpec.v.set(7); } else { tpSeg.set(2); sdSeg.set('bf16'); kvSeg.set('fp8'); sSpec.v.set(3); } rd(); } });
      const tpSeg = UI.seg(ctl, { label: 'TP', options: [1, 2, 3], value: S.tp, onChange: (v) => { S.tp = v; rd(); } });
      const sdSeg = UI.seg(ctl, { label: 'state', options: [{ v: 'fp32', label: 'FP32' }, { v: 'bf16', label: 'BF16' }], value: S.sd, onChange: (v) => { S.sd = v; rd(); } });
      const kvSeg = UI.seg(ctl, { label: 'KV Qwen', options: [{ v: 'bf16', label: 'BF16' }, { v: 'fp8', label: 'FP8' }], value: S.kv, onChange: (v) => { S.kv = v; rd(); } });
      sSpec.v = UI.slider(ctl, { label: 'drafts (k)', min: 0, max: 8, step: 1, value: S.spec, fmt: (v) => String(v), onInput: (v) => { S.spec = v; rd(); } });
      UI.slider(ctl, { label: 'Context', min: 12, max: 20, step: 0.25, value: S.lg, fmt: (v) => nf(Math.round(2 ** v)), onInput: (v) => { S.lg = v; rd(); } });
      UI.slider(ctl, { label: 'Requests', min: 1, max: 8, step: 1, value: S.c, fmt: (v) => String(v), onInput: (v) => { S.c = v; rd(); } });
      const rBlk = UI.readout(ctl, 'attention block', '—');
      let lastK = '';
      function calc() {
        const r = { ok: true, notes: [] };
        const sb = S.sd === 'fp32' ? 4 : 2;
        if (S.m === 'glm') {
          const heads = S.tp === 3 ? 66 : 64; r.hl = heads / S.tp; r.chan = 3 * r.hl * 128; r.lin = 34; r.kvpt = 656; r.align = 256; r.grow = 7579;
          if (S.tp === 1) r.notes.push('GLM on TP1 is only illustrative (the recipe uses TP2/TP3).');
          if (S.tp === 3) r.notes.push('TP3: 66 heads (64 + 2 padding heads), 22 per rank; A_log 64 → 66.');
          if (S.sd === 'bf16') r.notes.push('The GLM recipe keeps the recurrent state in FP32; BF16 here is hypothetical.');
          if (S.spec !== 7 && S.spec !== 2) r.notes.push('Recipe: DFlash2 k = 7 (MTP: k = 2).');
        } else {
          if (S.tp === 3) { r.ok = false; r.notes.push('Qwen\'s heads aren\'t divisible by 3: 16 QK heads and 2 KV heads. The recipe uses TP1 (one GPU) or TP2.'); return r; }
          r.hl = 48 / S.tp; r.chan = 10240 / S.tp; r.lin = 36; r.kvpt = (2 / S.tp) * 256 * 2 * (S.kv === 'bf16' ? 2 : 1); r.align = 16; r.grow = 13 * r.kvpt;
          if (S.spec > 8) r.notes.push('The fused decode kernel accepts up to 8 speculative tokens.');
          if (S.spec !== 3) r.notes.push('Recipe: MTP = 3.');
        }
        r.rec = r.hl * 128 * 128 * sb; r.conv = r.chan * (3 + S.spec) * 2; r.page = r.rec + r.conv;
        r.blk = Math.ceil(r.page / r.kvpt / r.align) * r.align; r.apage = r.blk * r.kvpt;
        if (S.m === 'glm' && S.tp === 2 && S.sd === 'fp32' && S.spec === 7) r.notes.push('Recipe configuration (TP2, DFlash2 k = 7): the boot log shows mla_page=2351104, the same page as above.');
        if (S.m === 'qwen' && S.tp === 2 && S.sd === 'bf16' && S.kv === 'fp8' && S.spec === 3) r.notes.push('Recipe default configuration (.env.sample): FP8 KV, BF16 state, MTP 3 → 1,664-token block.');
        if (S.m === 'qwen' && S.tp === 1) r.notes.unshift('TP1 = sibling single-GPU recipe; its .env.sample cites 3,200 → 1,664 (FP8 KV, FP32 → BF16 state).');
        if (S.m === 'qwen' && [1600, 3200, 1664].indexOf(r.blk) < 0) r.notes.unshift('Illustrative block (measured: 1,600/3,200/1,664): it depends on the alignment, 16 here; 32 or 64 also match the measured ones.');
        r.stReq = r.lin * r.page; r.n = Math.round(2 ** S.lg); r.kvReq = r.grow * r.n; r.tot = S.c * (r.stReq + r.kvReq);
        return r;
      }
      HF = V.fig(fig, (g, w, h) => {
        if (w < 80 || h < 60) return;
        const C = V.C(); const narrow = w < 560; const r = calc();
        const key = JSON.stringify(S); if (key !== lastK) { lastK = key; rBlk.set(r.ok ? nf(r.blk) + ' tok' : '—'); }
        const x0 = 16, x1 = w - 16, W = x1 - x0;
        const recCol = S.m === 'glm' ? C.glm : C.qwen;
        if (!r.ok) { r.notes.forEach((s, k) => wrap(g, s, W, 13).forEach((ln, q) => V.text(g, ln, x0, 60 + k * 40 + q * 18, { size: 13, color: C.c, weight: 600 }))); return; }
        // 1) page
        let y = 24;
        V.text(g, narrow ? 'page of one linear layer' : 'page of one linear layer (per request, per rank)', x0, y, { size: 12, weight: 600, color: C.ink });
        y += 12;
        const pw = W, sc = pw / r.apage;
        const wr = r.rec * sc, wc = r.conv * sc, wp = Math.max(0, pw - wr - wc);
        V.box(g, x0, y, wr, 26, { r: 5, fill: V.alpha(C.a, 0.7) });
        V.box(g, x0 + wr, y, Math.max(2, wc), 26, { r: 3, fill: V.alpha(C.d, 0.8) });
        if (wp > 1) { g.save(); g.beginPath(); g.rect(x0 + wr + wc, y, wp, 26); g.clip(); g.strokeStyle = V.alpha(C.muted, 0.5); g.lineWidth = 1; for (let k = -30; k < wp + 30; k += 6) { g.beginPath(); g.moveTo(x0 + wr + wc + k, y + 26); g.lineTo(x0 + wr + wc + k + 26, y); g.stroke(); } g.restore(); }
        g.strokeStyle = recCol; g.lineWidth = 1.5; V.rr(g, x0, y, pw, 26, 5); g.stroke();
        V.text(g, 'recurrent ' + fB(r.rec), x0 + 8, y + 17, { size: 11.5, font: 'mono', color: C.dark ? '#00210F' : '#fff', weight: 600, maxW: Math.max(40, wr - 12) });
        V.text(g, 'conv ' + fB(r.conv), x0 + wr + wc + 6, y + 42, { size: 11, font: 'mono', color: C.d, align: wr + wc + 90 > W ? 'right' : 'left', maxW: W });
        y += 64;
        const l1 = r.hl + ' heads × 128 × 128 × ' + (S.sd === 'fp32' ? 4 : 2) + ' B + conv ' + nf(r.chan) + ' × ' + (3 + S.spec) + ' × 2 B = ' + nf(r.page) + ' B';
        const L1 = wrap(g, l1, W, 11.5, 'mono');
        L1.forEach((ln, q) => V.text(g, ln, x0, y + q * 15, { size: 11.5, font: 'mono', color: C.ink }));
        y += L1.length * 15 + 6;
        const l2 = 'block = ⌈' + nf(r.page) + ' ÷ ' + nf(r.kvpt) + ' B/token⌉ (multiple of ' + r.align + ') = ' + nf(r.blk) + ' tokens → page ' + nf(r.apage) + ' B';
        const L2 = wrap(g, l2, W, 11.5, 'mono', 600);
        L2.forEach((ln, q) => V.text(g, ln, x0, y + q * 15, { size: 11.5, font: 'mono', color: recCol, weight: 600 }));
        y += L2.length * 15 + 22;
        // 2) memory per request
        V.text(g, (narrow ? 'memory per request at ' : 'memory per request (per rank) at ') + nf(r.n) + ' tokens', x0, y, { size: 12, weight: 600, color: C.ink, maxW: W });
        y += 10;
        const mx = Math.max(r.stReq, r.kvReq);
        const bar = (lab, val, col, yy) => { const bw = Math.max(3, (val / mx) * (W * (narrow ? 0.55 : 0.6))); V.box(g, x0, yy, bw, 16, { r: 4, fill: V.alpha(col, 0.75) }); V.text(g, lab + ' ' + fB(val), x0 + bw + 8, yy + 12, { size: 11.5, font: 'mono', color: C.ink, maxW: W - bw - 10 }); };
        bar('state (' + r.lin + ' layers)', r.stReq, C.a, y + 4);
        bar('KV (' + nf(r.grow) + ' B/tok)', r.kvReq, C.b, y + 26);
        y += 60;
        const TL = wrap(g, S.c + ' request' + (S.c > 1 ? 's' : '') + ': ' + fB(r.tot) + (S.m === 'qwen' ? ' (KV by formula; measured ~41% higher in FP8)' : ''), W, 12, 'mono', 600);
        TL.forEach((ln, q) => V.text(g, ln, x0, y + q * 16, { size: 12, font: 'mono', color: C.ink, weight: 600 }));
        y += (TL.length - 1) * 16 + 34;
        // 3) prefix cache granularity: common prefix of 10,000 tokens
        const PFX = 10000, SPAN = 12000; const XS = (p) => x0 + (p / SPAN) * W;
        const hit = Math.floor(PFX / r.blk) * r.blk;
        const PL = wrap(g, 'prefix cache: common prefix of ' + nf(PFX) + ' tokens → reuses ' + nf(hit) + ', recomputes ' + nf(PFX - hit), W, 12, 'sans', 600);
        PL.forEach((ln, q) => V.text(g, ln, x0, y + q * 16, { size: 12, weight: 600, color: C.ink }));
        y += 10 + (PL.length - 1) * 16;
        V.box(g, x0, y, XS(hit) - x0 + 0.01, 16, { r: 4, fill: V.alpha(C.b, 0.55) });
        V.box(g, XS(hit), y, XS(PFX) - XS(hit), 16, { r: 4, fill: V.alpha(C.c, 0.55) });
        g.strokeStyle = V.alpha(C.muted, 0.5); g.lineWidth = 1; V.rr(g, x0, y, W, 16, 4); g.stroke();
        for (let p = r.blk; p < SPAN; p += r.blk) V.line(g, XS(p), y - 3, XS(p), y + 19, { color: recCol, lw: 1.4 });
        V.text(g, 'ticks = checkpoints every ' + nf(r.blk) + ' tokens (align mode)', x0, y + 32, { size: 11, color: C.muted, maxW: W });
        y += 52;
        r.notes.slice(0, 2).forEach((s) => { wrap(g, '• ' + s, W, 11).forEach((ln, q) => { V.text(g, (q ? '   ' : '') + ln, x0, y, { size: 11, color: C.d }); y += 15; }); y += 3; });
        wrap(g, narrow ? 'page: green = state · amber = conv_state · hatching = padding up to the attention page' : 'page: green = recurrent state · amber = conv_state · hatching = leftover up to the attention page (uniform-page padding)', W, 11).forEach((ln) => { V.text(g, ln, x0, y, { size: 11, color: C.muted }); y += 15; });
      }, { static: true });
    });
  },
});

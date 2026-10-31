LA.chapter({
  id: 'receita-glm-tensorfold',
  part: 9,
  order: 49,
  title: 'Swapping the engine: the GLM recipe on TensorFold',
  navTitle: 'GLM recipe · TensorFold',
  subtitle: 'The same GLM-5.3-Flash EXL3, on the same two DGX Sparks, on a different engine: TensorFold v0.6.0 and 53 patches. Requirements, scripts, memory budget, every parameter, the patches by area, the numbers with their conditions and an honest comparison with the vLLM recipe.',
  minutes: 45,
  recipes: ['tf', 'glm'],
  tags: ['TensorFold', 'exactness', 'keyed sampling', 'RoCE', 'FP8 KV', 'q4', 'copy drafts', 'shared pool', 'prompt reuse', 'config.sh'],
  goals: [
    'Explain what changes when you swap vLLM for <b>TensorFold</b> with the same weights and the same hardware, and the exactness contract (drafted = serial, concurrent = solo, resumed = fresh) with the three lossy defaults',
    'Set up the two Sparks and run the <code>prepare.sh</code> / <code>start.sh</code> / <code>stop.sh</code> conveyor: patch hash, pinned digest, revisions, NFS, refit and smoke test',
    'Compute the <code>MemAvailable − reserve</code> budget and the KV pool, and justify each <code>config.sh</code> parameter by the phase in which it acts',
    'Read the recipe\'s numbers with their conditions, compare them with the vLLM recipe without treating them as an A/B, and tune the recipe to your workload (license, language, exactness, memory)',
  ],
  prereqs: ['motor-exato-tensorfold', 'receita-glm'],
  summary: [
    'Same 4 bpw EXL3 checkpoint, same two GB10s and same cable; a different engine: TensorFold v0.6.0, a <b>per-family engine</b>, with 53 patches baked into the image. Without them, upstream served GLM one request at a time in a 2,051-token window; with them come 4 streams, 1M context, FP8 KV, RoCE, copy drafts, vision and tool calling.',
    'Contract: <b>drafted = serial, concurrent = solo, resumed = fresh</b>, backed by keyed sampling, row-independent kernels and a rank-order sum. Three defaults are lossy against the checkpoint (<code>DENSE=q4</code>, <code>KV=fp8</code>, <code>KDA_CHUNKED=1</code>); <code>bf16 · bf16 · 0</code> serves the checkpoint as it is, with a smaller window.',
    'Network: one rank per Spark, TCPStore on port 29551, rank 0 decides and rank 1 applies. The fp32 partials (16 KiB per row) go over one-shot RDMA up to 512 KiB (11 µs versus 45 µs, decode +6%); 32 MiB prompt chunks go through NCCL on two rails (1.9 ms versus 3.5 ms).',
    'Conveyor: hash = SHA-256(patches + extras)[0:12]; while it is <code>ae8d1c789b47</code>, the image comes by digest; otherwise, by the new hash\'s tag or by a build with <code>patch -p0</code> in order. Weights at the pinned revisions, via rsync -L or NFS; start.sh does setup, checks, launch, loading with refit and smoke test.',
    'Memory: budget = MemAvailable − reserve (14.5 GiB); an 88.09 GiB estimate; pool up to 12.5 GiB beyond the window, or 2,922,496 tokens; the server uses ~8.6 GiB beyond the estimate from startup (7.0 GiB free when idle) and ~10 under a 1M prompt, so the reserve sets the minimum free memory (5.6 GiB measured). FP8 KV: 528 B per row, with a scale of \\(2^{\\lceil\\log_2 a_{\\max}\\rceil-8}\\).',
    'Numbers (sparkDash, clocks capped at 2,200 MHz, greedy): prose from 60.4 to 108.8 tok/s going from 1 to 4 requests, structured from 114.7 to 227.9, prefill at ~1,950 tok/s up to 64k and 1,015 at 981,841 tokens, 64k replay in under 0.07 s, GSM8K 98.0% and HumanEval 97.6% (TR3 checkpoint; the default pack since v1.3.3 reports 98.8% and 95.7%). Against vLLM (36.1 → 60.4 on prose), this is not a controlled A/B.',
    'Costs in v1.3.2: only 2 Sparks, at most 4 requests, a smaller API and 53 patches to rebase (the 123-patch image adds 3 Sparks, 8 requests and <code>/v1/messages</code>: see the last section). DFlash2 is CC BY-NC-ND (use <code>DRAFTER=mtp</code> for commercial use), and <code>q4</code> misses the end of turn in French (12/48 versus 1/48): use <code>DENSE=fp8</code>.',
  ],
  quiz: [
    { q: 'With the recipe\'s defaults (<code>DENSE=q4</code>, <code>KV=fp8</code>, <code>KDA_CHUNKED=1</code>), which statement is true?', o: ['The reply with drafts is identical to the serial one, but may differ from the checkpoint served losslessly in BF16', 'The reply is identical to the BF16 checkpoint\'s, because sampling is keyed', 'Drafts can change the reply when the temperature is above 0', 'Four requests together get different replies from the ones served alone, because of the batch'], a: 0, e: 'The three equalities compare the engine with itself, under the same settings. q4 and fp8 KV lose information and chunked KDA does not reproduce the serial kernel\'s bits, so the lossless reference (<code>DENSE=bf16 KV=bf16 KDA_CHUNKED=0</code>) can diverge at a near-tie. Drafts and batching change nothing: 6/6 and 22/22 in the published tests.' },
    { q: 'Why does TensorFold sum the tensor-parallel partials with an all-gather + rank-order sum, and not with an all-reduce?', o: ['Because the all-gather moves fewer bytes', 'Because NCCL does not implement all-reduce on the GB10', 'Because floating-point addition is not associative and the all-reduce does not guarantee the order: the ranks could end up with different bits and sample different tokens', 'Because RoCE only carries all-gathers'], a: 2, e: 'With the same sum, in the same order (“rank 0 first”), both ranks have the same bits and sample the same token without exchanging it. The price: the partials travel in fp32 (16 KiB per row, ~2× vLLM\'s bf16 all-reduce) and each rank does the whole sum; that is why small messages go over one-shot RDMA.' },
    { q: 'You edited a patch. What does <code>prepare.sh</code> do with the image?', o: ['Pulls the same image by the pinned digest', 'Computes the new hash, looks for the tag <code>v0.6.0-&lt;hash&gt;</code> on GHCR and, if it does not exist (or with <code>PULL=0</code>), builds it by applying the patches with <code>patch -p0</code> in name order', 'Reapplies the patch as an overlay on the next <code>docker run</code>', 'Refuses to start until you publish the image'], a: 1, e: 'The hash covers <code>patches/*.patch</code> and the extra packages. The digest is only used while the hash is the release one (<code>ae8d1c789b47</code>). With new patches, the new hash\'s tag or a local build is used instead, which ends with a verification import and the <code>tf.patches</code> label. <code>publish-image.sh</code> publishes the result and prints the values to pin.' },
    { q: 'MemAvailable of 116 GiB at startup, a 14.5 reserve, an 88.09 estimate and a <code>KV_POOL_GIB</code> of 12.5. What limits the pool, and what is the approximate minimum free memory under a 1M prompt?', o: ['Memory; ~0 GiB free', 'The 12.5 GiB ceiling; 14.5 GiB free, because the reserve is never touched', 'The window; the server refuses and refits', 'The 12.5 GiB ceiling; ~5 GiB free (reserve + leftover − ~10 GiB peak)'], a: 3, e: '116 − 14.5 − 88.09 = 13.41 GiB are left, more than the ceiling, so the pool stays at 12.5 GiB and ~0.9 GiB remain free. At its peak the server goes ~10 GiB past its own estimate: ~8.6 of them from startup (CUDA context, NCCL, graphs; when idle, ~14.5 + 0.9 − 8.6 ≈ 6.8 GiB are left, measured 7.0) and ~1.4 from the 1M prompt. Under the prompt, ~14.5 + 0.9 − 10 ≈ 5.4 GiB remain; the measured value was 5.6 GiB on the head.' },
    { q: 'What does the <code>fnc7:0.3</code> policy do differently from <code>fc5:0.3</code>?', o: ['It picks other tokens to draft', 'It estimates each draft\'s chance with the target\'s keyed Gumbel noise, which is already known, and stops when the product of the chances falls below 0.3, with up to 7 drafts; the reply does not change', 'It replaces equality verification with the \\(\\min(1, p/q)\\) rejection', 'It raises the target\'s temperature to accept more drafts'], a: 1, e: '“Only where the chain stops changes: the picks are the same.” The confidence is \\(\\mathrm{softmax}((\\text{score}+g)/\\beta)\\) with \\(\\beta = 2\\). Measured over <code>fc5:0.3</code>: prose 47.2 → 50.6 tok/s, code 51.7 → 55.5. Since verification compares against the keyed sample, no policy alters the tokens.' },
    { q: 'Commercial use with short prompts in French. What should you adjust in the <code>.env</code>?', o: ['Nothing: the defaults work', '<code>KV=bf16</code> and <code>CONTEXT=0</code>', '<code>DRAFTER=mtp</code> (DFlash2 is non-commercial; <code>PARALLEL</code> becomes 1) and <code>DENSE=fp8</code> (q4 misses the end of turn in 12 of 48 replies in French)', '<code>COMM=nccl</code> and <code>SPLIT=0</code>'], a: 2, e: 'DFlash2\'s CC BY-NC-ND license forbids commercial use; the checkpoint\'s MTP head avoids that, at the cost of one request at a time and of DFlash2\'s +5–10%. <code>DENSE=fp8</code> costs ~10% of decode, but keeps the end of turn (1/48, like bf16). <code>COMM</code> and <code>SPLIT</code> do not touch the bits.' },
  ],
  mount(root, LA) {
    const V = LA.viz, UI = LA.ui;
    const nf = (x, d = 0) => (+x).toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d });
    let tfCache = null;
    if (LA.theme && LA.onCleanup) LA.onCleanup(LA.theme.onChange(() => { tfCache = null; }));
    const TF = () => tfCache || (tfCache = (getComputedStyle(document.documentElement).getPropertyValue('--tf').trim() || V.C().d));
    const GLM = () => V.C().glm || V.C().a;
    const RED = () => V.C().red || '#FB2C36';
    const nocase = (c) => { const el = c && (c.el || c); if (el && el.querySelectorAll) el.querySelectorAll('label, span').forEach((n) => { if (!n.classList.contains('val')) n.style.textTransform = 'none'; }); return c; };
    function adaptH(fig, wide, narrow, bp = 640) {
      const st = fig.querySelector('.la-fig-stage');
      const set = () => { const want = (fig.clientWidth < bp ? narrow : wide) + 'px'; if (st.style.height !== want) st.style.height = want; };
      set();
      const ro = new ResizeObserver(set); ro.observe(fig);
      if (LA.onCleanup) LA.onCleanup(() => ro.disconnect());
    }
    function tw(g, s, size, font = 'sans', weight = 400) { g.font = V.font(size, font, weight); return g.measureText(String(s)).width; }
    /** largest font size (≤ size, ≥ min) at which the text fits in maxW */
    function fitSize(g, s, maxW, size, font = 'sans', weight = 400, min = 8) { let z = size; while (z > min && tw(g, s, z, font, weight) > maxW) z -= 0.5; return Math.max(min, z); }
    /** number with 1 decimal for narrow cells: '−0.8' or, compact, '−.8'; |v| < 0.05 becomes '0.0' */
    const cellNum = (v, compact) => { if (Math.abs(v) < 0.05) return '0.0'; const s = nf(Math.abs(v), 1); return (v < 0 ? '−' : '') + (compact ? s.replace(/^0(?=\.)/, '') : s); };
    function chip(g, x, y, w, h, label, o = {}) {
      const C = V.C();
      V.box(g, x, y, w, h, { r: o.r == null ? 6 : o.r, fill: o.fill || V.alpha(C.muted, 0.12), stroke: o.stroke || V.alpha(C.muted, 0.4), lw: o.lw || 1.2 });
      if (label != null && label !== '') {
        const font = o.font || 'mono', weight = o.weight || 500;
        const size = o.fit === false ? (o.size || 11.5) : fitSize(g, label, w - 6, o.size || 11.5, font, weight, o.min || 7.5);
        V.text(g, label, x + w / 2, y + h / 2 + 0.5, { size, font, weight, color: o.color || C.ink, align: 'center', base: 'middle' });
      }
    }
    function mark(g, x, y, ok, s = 6) {
      const C = V.C();
      g.lineWidth = 2.2; g.strokeStyle = ok ? C.a : RED(); g.lineCap = 'round'; g.beginPath();
      if (ok) { g.moveTo(x - s, y); g.lineTo(x - s * 0.3, y + s * 0.7); g.lineTo(x + s, y - s * 0.8); }
      else { g.moveTo(x - s * 0.8, y - s * 0.8); g.lineTo(x + s * 0.8, y + s * 0.8); g.moveTo(x + s * 0.8, y - s * 0.8); g.lineTo(x - s * 0.8, y + s * 0.8); }
      g.stroke(); g.lineCap = 'butt';
    }
    const hsl = (hue, a = 1) => { const C = V.C(); return `hsla(${hue}, ${C.dark ? 72 : 62}%, ${C.dark ? 62 : 40}%, ${a})`; };
    const onTile = () => (V.C().dark ? '#06140d' : '#ffffff');
    const fnv = (s) => { let h = 0x811c9dc5; for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 0x01000193) >>> 0; } return h.toString(16).padStart(8, '0'); };
    const softmax = (xs, T = 1) => { const m = Math.max(...xs); const e = xs.map((x) => Math.exp((x - m) / T)); const s = e.reduce((a, b) => a + b, 0); return e.map((x) => x / s); };
    /** grid of cells with values; fill(v,i,j) → [color, dark?] controls the text contrast */
    function cells(g, x, y, data, cw, ch, gap, fill, text, label) {
      const C = V.C();
      data.forEach((row, i) => row.forEach((v, j) => {
        const xx = x + j * (cw + gap), yy = y + i * (ch + gap);
        const [col, darkBg] = fill(v, i, j);
        g.fillStyle = col; V.rr(g, xx, yy, cw, ch, Math.min(4, cw / 5)); g.fill();
        const s = text(v, i, j);
        if (s != null && cw >= 12) {
          const z = fitSize(g, s, cw - 3, Math.min(11.5, ch * 0.55), 'mono', 500, 6);
          V.text(g, s, xx + cw / 2, yy + ch / 2 + 0.5, { size: z, font: 'mono', align: 'center', base: 'middle', color: darkBg ? (C.dark ? C.ink : '#ffffff') : (C.dark ? '#0b1a13' : C.ink) });
        }
      }));
      if (label) V.text(g, label, x, y - 7, { size: 11, color: C.muted, maxW: data[0].length * (cw + gap) + 60 });
      return { w: data[0].length * (cw + gap) - gap, h: data.length * (ch + gap) - gap };
    }
    /** cell color: diverging map (value) or heat (error), with a dark-background flag for the text */
    const divFill = (v, m) => { const C = V.C(); const tt = Math.max(-1, Math.min(1, v / m)); const a = Math.abs(tt); return [V.cmap(tt, 'div'), C.dark ? a < 0.55 : a > 0.55]; };
    const heatFill = (e) => { const C = V.C(); const tt = Math.max(0, Math.min(1, e)); return [V.cmap(tt, 'heat'), C.dark ? tt < 0.6 : tt > 0.35]; };

    /* ================================================ FIG 1 — the contract: four paths, one hash */
    (function () {
      const fig = root.querySelector('#fig-contrato');
      adaptH(fig, 400, 420, 600);
      const SENT = [
        ['A', 'draft', 'only', 'offers', ';', 'the', 'prompt', 'key', 'alone', 'picks', 'the', 'token'],
        ['Four', 'jobs', 'in', 'one', 'batch', 'get', 'the', 'same', 'tokens', 'as', 'solo', 'runs'],
        ['A', 'resume', 'from', 'a', 'kept', 'state', 'equals', 'a', 'full', 'redo', 'from', 'zero'],
      ];
      const ALT = ['but', 'here', 'a', 'tie', 'went', 'the', 'other', 'way', 'on'];
      const S = { mode: 'receita', seed: 0, t0: -6 };
      let base, ref, div;
      function build() { const r = V.rng(17 + S.seed * 31); base = SENT[S.seed % SENT.length].slice(); div = 5 + r.int(4); ref = base.map((w, i) => (i < div ? w : ALT[(i - div + S.seed) % ALT.length])); }
      build();
      const LANES = [
        { k: 'serial', t: 'one token at a time · "draft": false', c: 'a' },
        { k: 'draft', t: 'with drafts', c: 'b' },
        { k: 'batch', t: 'batched with 3 others', c: 'c' },
        { k: 'resume', t: 'resumed from a kept state', c: 'e' },
        { k: 'ref', t: 'lossless checkpoint (bf16 · bf16 · serial KDA)', c: 'f' },
      ];
      const RD = [3, 1, 4, 2, 2], RB = [2, 3, 2, 3, 2];
      const N = 12, P = 9.5;
      const cl = (x) => Math.max(0, Math.min(N, Math.floor(x)));
      function shown(k, tau) {
        if (k === 'serial' || k === 'ref') return cl((tau - 0.4) / 0.45 + 1);
        if (k === 'resume') return tau < 0.6 ? 0 : cl((tau - 0.6) / 0.4 + 1);
        const R = k === 'draft' ? RD : RB, dur = k === 'draft' ? 0.95 : 1.0;
        let n = 0; for (let j = 0; j < R.length; j++) if (tau >= 0.4 + (j + 1) * dur) n += R[j];
        return Math.min(N, n);
      }
      function ghost(k, tau) {
        if (k !== 'draft' && k !== 'batch') return null;
        const R = k === 'draft' ? RD : RB, dur = k === 'draft' ? 0.95 : 1.0;
        const j = Math.floor((tau - 0.4) / dur); if (j < 0 || j >= R.length) return null;
        let s = 0; for (let q = 0; q < j; q++) s += R[q];
        return [s, s + R[j], ((tau - 0.4) / dur) % 1];
      }
      const doneAt = { serial: 0.4 + 11 * 0.45, ref: 0.4 + 11 * 0.45, resume: 0.6 + 11 * 0.4, draft: 0.4 + 5 * 0.95, batch: 0.4 + 5 * 1.0 };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Configuration', options: [{ v: 'receita', label: 'recipe defaults (q4 · fp8 · chunked)' }, { v: 'exato', label: 'lossless (bf16 · bf16 · serial KDA, KDA_CHUNKED=0)' }], value: 'receita', onChange: (v) => { S.mode = v; S.t0 = H.t; upd(); } });
      UI.button(ctl, '↻ another sentence', () => { S.seed++; build(); S.t0 = H.t; });
      const r1 = nocase(UI.readout(ctl, 'drafted = serial', '✔')), r2 = nocase(UI.readout(ctl, 'concurrent = solo', '✔')), r3 = nocase(UI.readout(ctl, 'resumed = fresh', '✔')), r4 = nocase(UI.readout(ctl, '= BF16 checkpoint?', '—'));
      function upd() { r1.set('✔ same bits', 'ok'); r2.set('✔ same bits', 'ok'); r3.set('✔ same bits', 'ok'); if (S.mode === 'receita') r4.set('may diverge', 'warn'); else r4.set('✔ identical', 'ok'); }
      upd();
      const H = V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const tau = (t - S.t0) % P;
        const wide = w >= 600;
        const top = 8, laneH = (h - top - (wide ? 30 : 38)) / 5;
        const labelW = wide ? Math.min(210, w * 0.27) : 0;
        const hashW = wide ? 86 : 64;
        const x0 = wide ? labelW + 6 : 8;
        const tokW = w - x0 - hashW - 28;
        const gap = 3, cw = (tokW - gap * (N - 1)) / N;
        const hBase = fnv(base.join(' ')), hRef = fnv((S.mode === 'receita' ? ref : base).join(' '));
        const hashX = w - hashW - 4;
        const hashPos = [];
        LANES.forEach((L, i) => {
          const col = C[L.c];
          const y0 = top + i * laneH;
          const chH = wide ? Math.min(30, laneH - 16) : Math.min(26, laneH - 26);
          const cy = wide ? y0 + laneH / 2 - chH / 2 : y0 + 18;
          if (wide) V.wrap(g, L.t, 8, y0 + laneH / 2 - (L.t.length > 26 ? 4 : -4), labelW - 10, { size: 12, color: col, weight: 600, lineH: 15, maxLines: 2 });
          else V.text(g, L.t, 8, y0 + 12, { size: 11, color: col, weight: 600, maxW: w - 16 });
          const seq = L.k === 'ref' && S.mode === 'receita' ? ref : base;
          const n = shown(L.k, tau), gh = ghost(L.k, tau);
          for (let j = 0; j < N; j++) {
            const x = x0 + j * (cw + gap);
            const diverged = L.k === 'ref' && S.mode === 'receita' && j >= div;
            if (j < n) {
              const cc = diverged ? TF() : col;
              const s = fitSize(g, seq[j], cw - 3, wide ? 11 : 10, 'mono', 500, 6.5);
              const big = s >= 8;
              chip(g, x, cy, cw, chH, big ? seq[j] : '', { fill: V.alpha(cc, diverged ? 0.28 : 0.18), stroke: cc, r: 5, size: s, fit: false });
              if (!big) { g.fillStyle = hsl(parseInt(fnv(seq[j]).slice(0, 3), 16) % 360, 0.85); V.rr(g, x + 3, cy + chH * 0.28, cw - 6, chH * 0.44, 3); g.fill(); }
              if (diverged && j === div) V.text(g, '≠', x + cw / 2, wide ? cy - 3 : cy + chH + 13, { size: 13, weight: 700, color: TF(), align: 'center' });
            } else if (gh && j >= gh[0] && j < gh[1]) {
              g.setLineDash([3, 3]); V.box(g, x, cy, cw, chH, { r: 5, stroke: V.alpha(col, 0.5 + 0.4 * Math.sin(gh[2] * Math.PI)), lw: 1.4 }); g.setLineDash([]);
              if (cw >= 18) V.text(g, '?', x + cw / 2, cy + chH / 2 + 4, { size: 11, color: V.alpha(col, 0.7), align: 'center' });
            } else {
              V.box(g, x, cy, cw, chH, { r: 5, stroke: V.alpha(C.muted, 0.18), lw: 1 });
            }
          }
          if (gh) { const xa = x0 + gh[0] * (cw + gap) - 2, xb = x0 + gh[1] * (cw + gap) - gap + 2; V.line(g, xa, cy + chH + 4, xb, cy + chH + 4, { color: col, lw: 2 }); V.text(g, L.k === 'draft' ? 'window' : 'batch', (xa + xb) / 2, cy + chH + 15, { size: 10, color: col, align: 'center' }); }
          if (L.k === 'batch') V.text(g, '+3 streams, same batch', x0 + tokW, cy - 3, { size: 9.5, color: C.muted, align: 'right' });
          if (L.k === 'resume' && tau < 0.9) { const a = Math.max(0, 1 - Math.max(0, tau - 0.3) / 0.6); V.glow(g, x0 + cw / 2, cy + chH / 2, 26, C.e, a); chip(g, x0 + cw + 4, cy + 2, Math.min(110, tokW * 0.4), chH - 4, '↺ kept state', { fill: V.alpha(C.e, 0.2 * a + 0.02), stroke: V.alpha(C.e, a), color: C.ink, size: 10 }); }
          // hash
          const ready = n >= N && tau > doneAt[L.k] + 0.25;
          const hv = L.k === 'ref' ? hRef : hBase;
          const same = hv === hBase;
          if (ready) chip(g, hashX, cy, hashW - 4, chH, hv, { fill: V.alpha(same ? C.a : TF(), 0.16), stroke: same ? C.a : TF(), size: wide ? 11 : 9.5 });
          else chip(g, hashX, cy, hashW - 4, chH, '…', { fill: V.alpha(C.muted, 0.06), stroke: V.alpha(C.muted, 0.25), color: C.muted });
          hashPos.push([hashX, cy, chH, ready, same]);
        });
        V.text(g, 'token_sha', hashX + (hashW - 4) / 2, top + 2 + (wide ? 2 : 0), { size: 9.5, font: 'mono', color: C.muted, align: 'center', base: 'top' });
        // conclusion
        const allReady = hashPos.slice(0, 4).every((p) => p[3]);
        if (allReady) {
          const ya = hashPos[0][1] + hashPos[0][2] / 2, yb = hashPos[3][1] + hashPos[3][2] / 2, xr = hashX - 4;
          g.strokeStyle = C.a; g.lineWidth = 2; g.beginPath(); g.moveTo(xr, ya); g.lineTo(xr - 6, ya); g.lineTo(xr - 6, yb); g.lineTo(xr, yb); g.stroke();
          V.text(g, '=', xr - 12, (ya + yb) / 2 + 5, { size: 16, weight: 700, color: C.a, align: 'center' });
        }
        const msg = S.mode === 'receita'
          ? 'The four paths give the same bits as each other; the lossless checkpoint may diverge (here at token ' + (div + 1) + ').'
          : 'With bf16 · bf16 · serial KDA, the four paths and the checkpoint give the same sequence.';
        if (wide) V.text(g, msg, w / 2, h - 10, { size: 12, color: C.ink, align: 'center', maxW: w - 12 });
        else V.wrap(g, msg, 8, h - 24, w - 16, { size: 10.5, color: C.ink, lineH: 13, maxLines: 2 });
      });
    })();

    /* ================================================ FIG 2 — network: two Sparks, one cable, two rails */
    (function () {
      const fig = root.querySelector('#fig-rede');
      adaptH(fig, 390, 470);
      const S = { msg: 'dec', comm: 'roce', rails: 2 };
      const MSG = { dec: { rows: 1, lab: '16 KiB · 1 row', n: 3 }, lote: { rows: 32, lab: '512 KiB · 32 rows', n: 8 }, chunk: { rows: 2048, lab: '32 MiB · 2,048 rows', n: 26 } };
      const tr = () => (S.comm === 'roce' && MSG[S.msg].rows * 16 <= 512 ? 'roce' : 'nccl');
      const STG = {
        roce: [['GPU copies the shard to pinned memory', 'copy to pinned'], ['the last block rings the doorbell', 'doorbell'], ['proxy: RDMA WRITE + 4 B flag', 'WRITE + flag'], ['the peer GPU waits for the flag', 'waits for flag'], ['sum in rank order', 'sum 0 + 1']],
        nccl: [['NCCL copies to host staging', 'host staging'], ['the NCCL proxy schedules the send', 'NCCL proxy'], ['send over the rails', 'send'], ['receive and copy to the GPU', 'receive'], ['sum in rank order', 'sum 0 + 1']],
      };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Message', options: [{ v: 'dec', label: 'decode · 16 KiB' }, { v: 'lote', label: 'batch of 4 streams · 512 KiB' }, { v: 'chunk', label: 'prompt chunk · 32 MiB' }], value: 'dec', onChange: (v) => { S.msg = v; upd(); } });
      UI.seg(ctl, { label: 'COMM', options: [{ v: 'roce', label: 'roce' }, { v: 'nccl', label: 'nccl' }], value: 'roce', onChange: (v) => { S.comm = v; upd(); } });
      UI.seg(ctl, { label: 'Rails', options: [{ v: 2, label: '2' }, { v: 1, label: '1' }], value: 2, onChange: (v) => { S.rails = v; upd(); } });
      const rT = UI.readout(ctl, 'transport', '—'), rM = UI.readout(ctl, 'published time', '—');
      function upd() {
        const t = tr();
        rT.set(t === 'roce' ? 'RDMA one-shot' : 'NCCL');
        let v = '—', st;
        if (S.msg === 'dec') { v = t === 'roce' ? '11 µs' : '45 µs (32 tuned)'; st = t === 'roce' ? 'ok' : 'warn'; }
        else if (S.msg === 'lote') { v = t === 'roce' ? 'no figure · code ×4 +1.4%' : 'none published'; }
        else { v = S.rails === 2 ? '1.9 ms' : '3.5 ms'; st = S.rails === 2 ? 'ok' : 'warn'; }
        rM.set(v, st);
      }
      upd();
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const wide = w >= 640;
        const T = tr();
        const per = T === 'roce' ? 2.6 : 3.6;
        const ph = (t % per) / per, stage = Math.min(4, Math.floor(ph * 5)), sf = (ph * 5) % 1;
        const bw = wide ? w * 0.31 : w * 0.41, bh = wide ? h * 0.66 : h * 0.6;
        const bx0 = 8, bx1 = w - 8 - bw, by = 26;
        const boxes = [{ x: bx0, side: 1, name: 'Spark head · rank 0', ip: '192.0.2.1' }, { x: bx1, side: -1, name: 'Spark worker · rank 1', ip: '192.0.2.2' }];
        V.text(g, MSG[S.msg].lab + ' per rank · ' + (T === 'roce' ? 'RDMA one-shot (b12x)' : 'NCCL (host-staged)'), w / 2, 14, { size: wide ? 12 : 10.5, font: 'mono', color: C.ink, align: 'center', maxW: w - 16 });
        const nRows = Math.min(MSG[S.msg].rows, 6), nCols = wide ? 12 : 8;
        const geo = boxes.map((B, r) => {
          const bc = r ? C.b : TF();
          V.box(g, B.x, by, bw, bh, { r: 12, fill: V.alpha(bc, 0.07), stroke: V.alpha(bc, 0.55), lw: 1.4 });
          V.text(g, B.name, B.x + 10, by + 17, { size: wide ? 12 : 10.5, weight: 600, color: bc, maxW: bw - 16 });
          const nw = wide ? 24 : 18;
          const ix = B.side > 0 ? B.x + 10 : B.x + 14 + nw, iw = bw - 24 - nw;
          const gpu = { x: ix, y: by + 26, w: iw, h: bh * 0.44 };
          const pin = { x: ix, y: gpu.y + gpu.h + 10, w: iw, h: bh * 0.2 };
          const nic = { x: B.side > 0 ? B.x + bw - nw - 6 : B.x + 6, y: by + bh * 0.3, w: nw, h: bh * 0.44 };
          V.box(g, gpu.x, gpu.y, gpu.w, gpu.h, { r: 8, fill: V.alpha(C.a, 0.08), stroke: V.alpha(C.a, 0.55) });
          V.text(g, 'GPU', gpu.x + 8, gpu.y + 14, { size: 11, weight: 600, color: C.a });
          V.text(g, wide ? 'P(' + r + ') [' + nf(MSG[S.msg].rows) + ' × 4096] fp32' : 'P(' + r + ') ' + nf(MSG[S.msg].rows) + (MSG[S.msg].rows > 1 ? ' rows' : ' row'), gpu.x + 40, gpu.y + 14, { size: wide ? 10.5 : 9.5, font: 'mono', color: C.ink, maxW: gpu.w - 46 });
          // mini-matrix of the partial: R rows × 4096 columns (sampled)
          const gx = gpu.x + 8, gy = gpu.y + 22, gw = gpu.w - 16, gh = gpu.h - 30;
          const cs = Math.min((gw - (nCols - 1) * 2) / nCols, (gh - (nRows - 1) * 2) / Math.max(nRows, 2));
          for (let i = 0; i < nRows; i++) for (let j = 0; j < nCols; j++) {
            const v = Math.sin(i * 1.7 + j * 0.9 + r * 2.3) * 0.5 + 0.5;
            const col = stage === 4 ? V.alpha(C.a, 0.35 + 0.5 * (Math.sin(i * 1.7 + j * 0.9) * 0.5 + 0.5)) : V.alpha(bc, 0.25 + 0.6 * v);
            g.fillStyle = col; V.rr(g, gx + j * (cs + 2), gy + i * (cs + 2), cs, cs, 2); g.fill();
          }
          if (stage !== 4) V.text(g, '= ' + MSG[S.msg].lab.split(' · ')[0] + (wide ? ' per rank' : ''), gpu.x + 8, gpu.y + gpu.h - 7, { size: wide ? 10.5 : 9.5, font: 'mono', color: C.muted, maxW: gpu.w - 12 });
          V.box(g, pin.x, pin.y, pin.w, pin.h, { r: 8, fill: V.alpha(C.d, 0.1), stroke: V.alpha(C.d, 0.55) });
          V.text(g, T === 'roce' ? (wide ? 'pinned memory · send/recv' : 'pinned') : (wide ? 'NCCL staging (host)' : 'staging'), pin.x + 8, pin.y + pin.h / 2 + 4, { size: wide ? 10.5 : 9.5, color: C.ink, maxW: pin.w - 30 });
          V.box(g, nic.x, nic.y, nic.w, nic.h, { r: 6, fill: V.alpha(C.muted, 0.15), stroke: V.alpha(C.muted, 0.6) });
          V.text(g, 'CX7', nic.x + nic.w / 2, nic.y - 4, { size: 9.5, font: 'mono', color: C.muted, align: 'center' });
          const ports = [nic.y + nic.h * 0.3, nic.y + nic.h * 0.7];
          ports.forEach((py, k) => { V.dot(g, nic.x + nic.w / 2, py, 3.5, k === 1 && S.rails === 1 ? V.alpha(C.muted, 0.35) : C.e); });
          const ey = by + bh - 10;
          if (r === 0) V.text(g, wide ? 'API :8888 · TCPStore :29551' : 'API · TCPStore', B.x + 10, ey, { size: 10, font: 'mono', color: C.muted, maxW: bw - 16 });
          else V.text(g, wide ? 'GID idx 3 · v2 · ::ffff:' + B.ip : 'GID v2 ::ffff:' + B.ip, B.x + 10, ey, { size: wide ? 10 : 9, font: 'mono', color: C.muted, maxW: bw - 16 });
          return { gpu, pin, nic, ports, gy, gh };
        });
        const pA = geo[0].ports, pB = geo[1].ports, xa = geo[0].nic.x + geo[0].nic.w / 2, xb = geo[1].nic.x + geo[1].nic.w / 2;
        [0, 1].forEach((k) => {
          const off = k === 1 && S.rails === 1;
          V.line(g, xa, pA[k], xb, pB[k], { color: off ? V.alpha(C.muted, 0.3) : V.alpha(C.e, 0.7), lw: off ? 1 : 2.4, dash: off ? [4, 4] : null });
          V.text(g, off ? (wide ? 'rail 2 (off)' : 'off') : 'rail ' + (k + 1), (xa + xb) / 2, pA[k] - 5, { size: 9.5, color: off ? C.muted : C.e, align: 'center' });
        });
        V.text(g, wide ? 'QSFP · 2× PCIe Gen5 x4' : 'QSFP', (xa + xb) / 2, pA[1] + 16, { size: wide ? 10 : 9, font: 'mono', color: C.muted, align: 'center', maxW: xb - xa - 8 });
        const nP = MSG[S.msg].n;
        geo.forEach((G, r) => {
          const gx = G.gpu.x + G.gpu.w / 2, gy = G.gpu.y + G.gpu.h, py = G.pin.y + G.pin.h / 2;
          if (stage === 0) { for (let k = 0; k < Math.min(6, nP); k++) { const u = (sf + k / 6) % 1; V.dot(g, gx - 20 + k * 8, V.lerp(gy - 4, py, u), 2.8, V.alpha(r ? C.b : TF(), 0.95)); } }
          if (stage === 1) { const pulse = 0.5 + 0.5 * Math.sin(sf * Math.PI * 4); V.glow(g, G.pin.x + G.pin.w - 14, py, 16, C.d, pulse); V.text(g, T === 'roce' ? '🔔' : '⚙', G.pin.x + G.pin.w - 14, py + 5, { size: 13, align: 'center' }); }
          if (stage === 3) { V.text(g, T === 'roce' ? '⚑ flag seq' : '⇣ received', G.pin.x + G.pin.w - 6, G.pin.y - 4, { size: 10, color: C.d, weight: 600, align: 'right' }); const ang = sf * Math.PI * 2; g.strokeStyle = C.a; g.lineWidth = 2; g.beginPath(); g.arc(G.gpu.x + G.gpu.w - 14, G.gpu.y + 12, 6, ang, ang + 4.2); g.stroke(); }
          if (stage === 4) { const bxw = Math.min(G.gpu.w - 30, wide ? 150 : 96); chip(g, G.gpu.x + G.gpu.w - bxw - 8, G.gpu.y + G.gpu.h - 22, bxw, 17, 'P(0) + P(1) → y', { fill: V.alpha(C.a, 0.25), stroke: C.a, size: 10.5 }); }
        });
        if (stage === 4) V.text(g, '= same bits', w / 2, by + bh * 0.2, { size: wide ? 12 : 10.5, weight: 700, color: C.a, align: 'center' });
        if (stage === 2) {
          for (let k = 0; k < nP; k++) {
            const rail = S.rails === 2 ? k % 2 : 0;
            const u = (sf * 1.1 + (k / nP) * 0.6) % 1;
            const dir = k % 4 < 2 ? 1 : -1;
            const x = dir > 0 ? V.lerp(xa, xb, u) : V.lerp(xb, xa, u);
            const y = V.lerp(pA[rail], pB[rail], dir > 0 ? u : 1 - u);
            V.dot(g, x, y, S.msg === 'chunk' ? 2.4 : 3.4, dir > 0 ? TF() : C.b);
          }
        }
        const sy = by + bh + 12;
        const st = STG[T];
        const cols = wide ? 5 : 3, rows = wide ? 1 : 2;
        const noteH = wide ? 20 : 30;
        const cwid = (w - 16 - (cols - 1) * 6) / cols, chh = Math.min(wide ? 46 : 40, (h - sy - noteH - 8 - (rows - 1) * 6) / rows);
        st.forEach((s2, i) => {
          const c = i % cols, rr = Math.floor(i / cols);
          const x = 8 + c * (cwid + 6), y = sy + rr * (chh + 6);
          const on = i === stage;
          V.box(g, x, y, cwid, chh, { r: 8, fill: on ? V.alpha(C.d, 0.22) : V.alpha(C.muted, 0.08), stroke: on ? C.d : V.alpha(C.muted, 0.35), lw: on ? 2 : 1 });
          const lab = (i + 1) + ' · ' + (wide ? s2[0] : s2[1]);
          V.wrap(g, lab, x + 7, y + (wide ? 17 : chh / 2 + 4), cwid - 12, { size: wide ? 11.5 : 10, lineH: 14, maxLines: 2, color: on ? C.ink : C.muted, weight: on ? 600 : 400 });
        });
        const note = T === 'roce' ? 'one kernel call per collective; an epoch on the device so the CUDA graph can replay the exchange' : S.comm === 'roce' ? 'above 512 KiB NCCL takes over, with the same layout (rank order) and the same bits' : 'small messages pay NCCL\'s fixed latency (~30–50 µs)';
        if (wide) V.text(g, note, w / 2, h - 8, { size: 11.5, color: C.muted, align: 'center', maxW: w - 12 });
        else V.wrap(g, note, 8, h - 22, w - 16, { size: 10, color: C.muted, lineH: 13, maxLines: 2 });
      });
    })();

    /* ================================================ FIG 3 — conveyor (stepper) */
    (function () {
      const fig = root.querySelector('#fig-esteira');
      adaptH(fig, 380, 440);
      const S = { edit: false, pull0: false, nfs: false, low: false };
      const ctl = UI.controls(fig);
      UI.toggle(ctl, { label: 'I edited a patch', value: false, onChange: (v) => { S.edit = v; if (st) st.t0 = performance.now(); } });
      UI.toggle(ctl, { label: 'PULL=0', value: false, onChange: (v) => { S.pull0 = v; if (st) st.t0 = performance.now(); } });
      UI.toggle(ctl, { label: 'WORKER_WEIGHTS=nfs', value: false, onChange: (v) => { S.nfs = v; if (st) st.t0 = performance.now(); } });
      UI.toggle(ctl, { label: 'low free memory (refit)', value: false, onChange: (v) => { S.low = v; if (st) st.t0 = performance.now(); } });
      const NAMES = ['preflight', 'image', 'worker', 'weights', 'checks', 'launch', 'load', 'smoke'];
      const steps = [
        { t: 'Preflight on both Sparks', d: 'Docker, GPU, NVIDIA runtime, <code>rsync</code>, key-based ssh, link (route → interface → HCA → RoCE v2 GID) and disk: 180 GB for the weights and 35 GB for the image.' },
        { t: 'The image and the patch hash', d: 'SHA-256 of <code>patches/*.patch</code> + extras, 12 hex digits. With the release hash, the image comes by digest; with a new hash, by the published tag or by a build with <code>patch -p0</code> in order.' },
        { t: 'The same image on the worker', d: 'Pull, or <code>docker save | ssh docker load</code> with only the missing layers; identity is compared by content (issue #8).' },
        { t: 'Weights at the pinned revisions', d: 'Download on the head, <code>tensorfold info</code> and <code>refs/main</code>; then <code>rsync -a -L --partial</code> checked file by file, or the read-only NFS volume.' },
        { t: 'start.sh 2/5 · checks', d: 'Arguments through TensorFold\'s parser in a GPU-less container, link, previous server, port and memory (≥ ~110 GiB per Spark).' },
        { t: 'start.sh 3/5 · launch', d: 'Rank 1 on the worker over ssh, then rank 0 + API. Rendezvous through the TCPStore on port 29551, and the settings are compared.' },
        { t: 'start.sh 4/5 · loading', d: '~80 GiB of weights per Spark in 2–6 min, with a heartbeat every 15 s. If the window does not fit, refit: the script restarts once with <code>--context N</code>.' },
        { t: 'start.sh 5/5 · smoke test', d: '“Reply with OK.” at temperature 0, 32 tokens, no thinking, through both ranks. Then: <code>is now LIVE! on port 8888</code>.' },
      ];
      let H3 = null;
      const st = UI.stepper(fig, steps, { autoplay: 6500, onStep: () => H3 && H3.redraw() });
      let prevI = 0, moveFrom = 0, moveT0 = 0;
      const wordHash = () => (S.edit ? 'c41f9e07b2a8' : 'ae8d1c789b47');
      function scene0(g, A, e, C, wide) {
        const items = ['Docker', 'GPU (nvidia-smi -L)', 'NVIDIA runtime', 'rsync', 'ssh BatchMode', 'link · RoCE v2 GID', 'disk 180 + 35 GB'];
        const colW = (A.w - 20) / 2, rh = wide ? 24 : Math.min(30, Math.max(22, (A.h - 150) / items.length));
        ['head', 'worker'].forEach((nm, c) => {
          const x = A.x + 10 + c * colW;
          V.text(g, nm, x, A.y + 12, { size: 12, weight: 600, color: c ? C.b : TF() });
          items.forEach((it, i) => {
            const y = A.y + 30 + i * rh;
            const on = e > 0.25 + i * 0.3 + c * 0.12;
            V.box(g, x, y - 9, colW - 16, rh - 4, { r: 6, fill: on ? V.alpha(C.a, 0.12) : V.alpha(C.muted, 0.06), stroke: on ? V.alpha(C.a, 0.6) : V.alpha(C.muted, 0.25) });
            V.text(g, it, x + 24, y + 5, { size: wide ? 11 : 10, color: on ? C.ink : C.muted, maxW: colW - 50 });
            if (on) mark(g, x + 12, y + 1, true, 5);
          });
        });
        // link detection, link by link
        const ly = A.y + 30 + items.length * rh + 14;
        V.text(g, 'link detection (nodes.sh)', A.x + 10, ly, { size: 11, color: C.muted, weight: 600 });
        const chain = wide ? ['ip route get <peer>', 'interface (netdev)', 'RoCE HCA in /sys', 'GID v2 ::ffff:<IPv4>', 'NCCL_IB_GID_INDEX'] : ['route', 'netdev', 'HCA', 'GID v2', 'NCCL env'];
        const n = chain.length, gap = wide ? 18 : 10, cw = (A.w - 20 - (n - 1) * gap) / n;
        chain.forEach((c, i) => {
          const x = A.x + 10 + i * (cw + gap), y = ly + 10, on = e > 2.6 + i * 0.35;
          chip(g, x, y, cw, 26, c, { fill: on ? V.alpha(C.e, 0.16) : V.alpha(C.muted, 0.06), stroke: on ? C.e : V.alpha(C.muted, 0.3), color: on ? C.ink : C.muted, size: wide ? 11 : 9.5 });
          if (i < n - 1) V.arrow(g, x + cw + 2, y + 13, x + cw + gap - 2, y + 13, { color: on ? C.e : V.alpha(C.muted, 0.4), lw: 1.4, head: 5 });
        });
        if (e > 4.5) V.wrap(g, 'An all-zero or v1 entry makes NCCL fail “about a minute in”; the preflight checks this before bringing anything up.', A.x + 10, ly + 56, A.w - 20, { size: wide ? 11 : 10, color: C.muted, lineH: 14, maxLines: 2 });
      }
      function scene1(g, A, e, C, wide) {
        const leftW = wide ? A.w * 0.42 : A.w;
        const gh = wide ? A.h : A.h * 0.42;
        const cols = wide ? 11 : 13, n = 53, cs = Math.min(14, (leftW - 20) / cols - 3);
        const gx = A.x + 8, gy = A.y + 18;
        V.text(g, 'patches/*.patch (53)', gx, A.y + 10, { size: 11, color: C.muted });
        for (let i = 0; i < n; i++) {
          const c = i % cols, r = Math.floor(i / cols);
          const x = gx + c * (cs + 3), y = gy + r * (cs + 3);
          const ed = S.edit && i === 17;
          g.fillStyle = ed ? RED() : V.alpha(TF(), 0.35 + 0.35 * ((i * 7) % 5) / 5); V.rr(g, x, y, cs, cs, 3); g.fill();
        }
        const rowsN = Math.ceil(n / cols);
        const fy = gy + rowsN * (cs + 3) + 10;
        const fw = Math.min(leftW - 20, 220);
        V.box(g, gx, fy, fw, 26, { r: 7, fill: V.alpha(C.d, 0.14), stroke: C.d });
        V.text(g, 'SHA-256(patches + extras)[0:12]', gx + fw / 2, fy + 17, { size: 10.5, font: 'mono', color: C.ink, align: 'center', maxW: fw - 8 });
        for (let k = 0; k < 6; k++) { const u = (e * 0.9 + k / 6) % 1; V.dot(g, gx + 10 + k * (fw - 20) / 5, V.lerp(gy + rowsN * (cs + 3) - 4, fy, u), 2.2, V.alpha(TF(), 1 - u)); }
        const hs = wordHash(), nh = Math.min(12, Math.floor(e * 8));
        V.text(g, 'hash: ' + hs.slice(0, nh) + (nh < 12 ? '▌' : ''), gx, fy + 46, { size: 13, font: 'mono', weight: 600, color: S.edit ? RED() : C.a });
        if (S.edit) V.text(g, '(illustrative)', gx + tw(g, 'hash: ' + hs, 13, 'mono', 600) + 8, fy + 46, { size: 10, color: C.muted });
        // decision
        const dx = wide ? A.x + leftW + 14 : A.x + 8, dy = wide ? A.y + 6 : fy + 62, dw = wide ? A.w - leftW - 20 : A.w - 16;
        const path = S.pull0 ? 'build' : S.edit ? 'build' : 'pull';
        const boxes = [
          { t: 'hash = IMAGE_TAG (ae8d1c789b47)?', on: e > 1.2 },
          path === 'pull' ? { t: 'yes → docker pull …@sha256:22789f0c…', on: e > 1.8, ok: true } : { t: S.pull0 ? 'PULL=0 → build locally' : 'no → tag v0.6.0-' + hs + ' published? no', on: e > 1.8, warn: true },
        ];
        const bh = wide ? 26 : 22;
        boxes.forEach((b, i) => {
          const y = dy + i * (bh + 8);
          V.box(g, dx, y, dw, bh, { r: 7, fill: b.on ? V.alpha(b.ok ? C.a : b.warn ? C.d : C.b, 0.16) : V.alpha(C.muted, 0.06), stroke: b.on ? (b.ok ? C.a : b.warn ? C.d : C.b) : V.alpha(C.muted, 0.3) });
          V.text(g, b.t, dx + 8, y + bh / 2 + 4, { size: wide ? 11 : 10, font: 'mono', color: b.on ? C.ink : C.muted, maxW: dw - 14 });
        });
        const ly = dy + 2 * (bh + 8);
        if (path === 'pull') {
          if (e > 2.4) { V.text(g, 'label tf.patches = ae8d1c789b47', dx + 8, ly + 14, { size: wide ? 11 : 10, font: 'mono', color: C.a, maxW: dw - 30 }); mark(g, dx + dw - 12, ly + 10, true, 5); }
          if (e > 3.0) V.wrap(g, '“a tag can be moved, a digest cannot”: the pulled image is exactly the one that was measured.', dx + 8, ly + 36, dw - 14, { size: wide ? 11.5 : 10.5, color: C.muted, lineH: 15, maxLines: 3 });
        } else {
          const layers = ['FROM nvcr.io/nvidia/pytorch:26.07-py3', 'pip install TensorFold@v0.6.0 + av + xgrammar', 'patch -p0 --forward × 53 (name order)', 'import server · engine · vision · av · xgrammar', 'LABEL tf.patches=' + hs];
          const lh = wide ? 21 : 18;
          layers.forEach((L, i) => {
            const on = e > 2.2 + i * 0.55;
            const y = ly + i * (lh + 4);
            V.box(g, dx + 10, y, dw - 10, lh, { r: 5, fill: on ? V.alpha(TF(), 0.18) : V.alpha(C.muted, 0.05), stroke: on ? V.alpha(TF(), 0.7) : V.alpha(C.muted, 0.2) });
            V.text(g, L, dx + 16, y + lh / 2 + 4, { size: wide ? 10.5 : 9.5, font: 'mono', color: on ? C.ink : C.muted, maxW: dw - 22 });
          });
        }
      }
      function scene2(g, A, e, C, wide) {
        const L = ['pytorch:26.07', 'TensorFold v0.6.0', 'av · xgrammar', '53 patches', 'label + env'];
        const sw = Math.min(200, A.w * 0.34), lh = Math.min(28, (A.h - 70) / 5);
        const xs = [A.x + 10, A.x + A.w - sw - 10];
        ['head', 'worker'].forEach((nm, c) => {
          V.text(g, nm, xs[c] + sw / 2, A.y + 12, { size: 12, weight: 600, color: c ? C.b : TF(), align: 'center' });
          L.forEach((l, i) => {
            const y = A.y + 22 + (L.length - 1 - i) * (lh + 4);
            const on = c === 0 || e > 0.6 + i * 0.45;
            V.box(g, xs[c], y, sw, lh, { r: 6, fill: on ? V.alpha(c ? C.b : TF(), 0.18) : V.alpha(C.muted, 0.05), stroke: on ? (c ? C.b : TF()) : V.alpha(C.muted, 0.25), lw: 1.2 });
            V.text(g, l, xs[c] + sw / 2, y + lh / 2 + 4, { size: wide ? 11 : 10, font: 'mono', color: on ? C.ink : C.muted, align: 'center', maxW: sw - 8 });
            if (c === 1 && e > 0.9 + i * 0.45) mark(g, xs[1] + sw + (wide ? 10 : -10), y + lh / 2, true, 5);
          });
        });
        const ay = A.y + 22 + 2 * (lh + 4) + lh / 2;
        V.arrow(g, xs[0] + sw + 8, ay, xs[1] - 8, ay, { color: C.d, lw: 2 });
        const u = (e * 0.7) % 1; V.dot(g, V.lerp(xs[0] + sw + 8, xs[1] - 8, u), ay, 4, C.d);
        V.text(g, wide ? 'pull or docker save | ssh docker load' : 'save | ssh load', (xs[0] + sw + xs[1]) / 2, ay - 10, { size: wide ? 11 : 9.5, font: 'mono', color: C.d, align: 'center', maxW: xs[1] - xs[0] - sw - 10 });
        const by = A.y + 22 + 5 * (lh + 4) + 14;
        V.wrap(g, 'Identity by content (layers + runtime config): the .Id is the manifest digest in containerd and the config digest in overlay2, so it would never match across different stores (#8).', A.x + 10, by, A.w - 20, { size: wide ? 11.5 : 10.5, color: C.muted, lineH: 15, maxLines: 3 });
      }
      function scene3(g, A, e, C, wide) {
        const lines = ['hub/models--Mia-AiLab--GLM-5.3-Flash-EXL3-TR3-4bpw/', '  snapshots/9eaebb7c…/  → links', '  blobs/  (~176 GB, xet: links to the root)', '  refs/main  ← written by the script'];
        V.text(g, 'head · HF_CACHE (HF_HUB_OFFLINE=1 in the containers)', A.x + 10, A.y + 12, { size: 11.5, weight: 600, color: TF(), maxW: A.w - 20 });
        lines.forEach((l, i) => V.text(g, l, A.x + 14, A.y + 32 + i * 17, { size: wide ? 11 : 9.5, font: 'mono', color: e > 0.2 + i * 0.25 ? C.ink : C.muted, maxW: A.w - 24 }));
        const y = A.y + 32 + 4 * 17 + 14;
        V.text(g, 'worker', A.x + 10, y, { size: 11.5, weight: 600, color: C.b });
        const bw = A.w - 20;
        if (!S.nfs) {
          const p = Math.min(1, Math.max(0, (e - 0.6) / 3.2));
          V.box(g, A.x + 10, y + 10, bw, 22, { r: 6, fill: V.alpha(C.muted, 0.08), stroke: V.alpha(C.muted, 0.35) });
          g.fillStyle = V.alpha(C.b, 0.45); V.rr(g, A.x + 11, y + 11, (bw - 2) * p, 20, 5); g.fill();
          V.text(g, 'rsync -a -L --partial · ' + nf(166 * p, 0) + ' / 166 GiB', A.x + 18, y + 25, { size: wide ? 11 : 9.5, font: 'mono', color: C.ink, maxW: bw - 16 });
          if (p >= 1) { mark(g, A.x + 18, y + 48, true, 5); V.text(g, 'checked file by file (path + size); resumes if interrupted', A.x + 30, y + 52, { size: 10.5, color: C.muted, maxW: bw - 24 }); }
        } else {
          const pulse = 0.5 + 0.5 * Math.sin(e * 4);
          V.box(g, A.x + 10, y + 10, bw, 38, { r: 8, fill: V.alpha(C.e, 0.1 + 0.08 * pulse), stroke: C.e });
          V.text(g, 'NFS docker volume · ro · nfsvers=4.2 · nconnect=8 · rsize 1 MiB', A.x + 18, y + 26, { size: wide ? 11 : 9.5, font: 'mono', color: C.ink, maxW: bw - 16 });
          V.text(g, 'no sudo on the worker · ~166 GiB less disk · both ranks up in ~2.2 min', A.x + 18, y + 41, { size: wide ? 10.5 : 9, color: C.muted, maxW: bw - 16 });
        }
        const py = y + 76;
        if (e > 1.2) {
          V.text(g, 'revision pins (Hugging Face commits)', A.x + 10, py, { size: 11, weight: 600, color: C.e });
          V.text(g, 'MODEL_REVISION   = 9eaebb7c4e96d983dcd538e18624622ba5b820a8', A.x + 14, py + 18, { size: wide ? 10.5 : 8.5, font: 'mono', color: C.ink, maxW: A.w - 24 });
          V.text(g, 'DFLASH2_REVISION = bf582e4eacc1810f76656d1811693ff6c6737d2a', A.x + 14, py + 34, { size: wide ? 10.5 : 8.5, font: 'mono', color: C.ink, maxW: A.w - 24 });
          V.wrap(g, '“a new upstream commit changes nothing until the pin does”: both ranks serve exactly these snapshots from the local cache.', A.x + 14, py + 54, A.w - 28, { size: wide ? 11 : 9.5, color: C.muted, lineH: 14, maxLines: 3 });
        }
      }
      function scene4(g, A, e, C, wide) {
        const items = [['arguments (TF parser, GPU-less container)', true], ['link: route, HCA, RoCE v2 GID', true], ['previous server stopped or absent', true], ['port 8888 free (ss -ltn)', true], [S.low ? 'memory: below ~110 GiB free → warning (illustrative)' : 'memory: ≥ ~110 GiB free on both Sparks', !S.low]];
        const rh = Math.min(34, (A.h - 20) / items.length);
        items.forEach(([t, ok], i) => {
          const on = e > 0.3 + i * 0.4, y = A.y + 10 + i * rh;
          V.box(g, A.x + 10, y, A.w - 20, rh - 6, { r: 7, fill: on ? V.alpha(ok ? C.a : C.d, 0.12) : V.alpha(C.muted, 0.05), stroke: on ? (ok ? V.alpha(C.a, 0.6) : C.d) : V.alpha(C.muted, 0.25) });
          V.text(g, t, A.x + 36, y + (rh - 6) / 2 + 4, { size: wide ? 12 : 10.5, color: on ? C.ink : C.muted, maxW: A.w - 56 });
          if (on) { if (ok) mark(g, A.x + 22, y + (rh - 6) / 2, true, 5); else V.text(g, '⚠', A.x + 22, y + (rh - 6) / 2 + 5, { size: 13, color: C.d, align: 'center' }); }
        });
      }
      function scene5(g, A, e, C, wide) {
        const bw = Math.min(220, A.w * 0.4), bh = 70, y = A.y + 20;
        const xw = A.x + A.w - bw - 10, xh = A.x + 10;
        if (e > 0.3) { V.box(g, xw, y, bw, bh, { r: 10, fill: V.alpha(C.b, 0.12), stroke: C.b }); V.text(g, 'rank 1 · worker (ssh)', xw + 10, y + 18, { size: 11.5, weight: 600, color: C.b, maxW: bw - 16 }); V.text(g, 'tensorfold serve … --rank 1', xw + 10, y + 38, { size: wide ? 10.5 : 9, font: 'mono', color: C.ink, maxW: bw - 16 }); }
        if (e > 1.0) { V.box(g, xh, y, bw, bh, { r: 10, fill: V.alpha(TF(), 0.12), stroke: TF() }); V.text(g, 'rank 0 · head + API', xh + 10, y + 18, { size: 11.5, weight: 600, color: TF(), maxW: bw - 16 }); V.text(g, '--rank 0 --port 8888', xh + 10, y + 38, { size: wide ? 10.5 : 9, font: 'mono', color: C.ink, maxW: bw - 16 }); V.text(g, 'TCPStore :29551', xh + 10, y + 56, { size: wide ? 10.5 : 9, font: 'mono', color: C.muted, maxW: bw - 16 }); }
        if (e > 1.5) {
          const ya = y + bh / 2; V.line(g, xh + bw, ya, xw, ya, { color: V.alpha(C.e, 0.7), lw: 2, dash: [5, 4], dashOffset: -e * 20 });
          const u = ((e - 1.5) * 0.6) % 1; chip(g, V.lerp(xh + bw, xw - 90, u), ya - 24, 90, 18, 'ncclUniqueId', { size: 10, fill: V.alpha(C.e, 0.18), stroke: C.e });
        }
        if (e > 2.0) {
          const keys = wide ? ['context', 'grid', 'copy', 'copy_max', 'split', 'max_rows', 'wide', 'kv', 'dense', 'policy'] : ['context', 'grid', 'copy', 'split', 'kv', 'policy'];
          const vals = { context: '1048576', grid: '64', copy: '1', copy_max: '15', split: '1', max_rows: '16', wide: '16', kv: 'fp8', dense: 'q4', policy: 'fnc7:0.3' };
          const yy = y + bh + 26, lw = wide ? 64 : 50, cw = (A.w - 20 - lw) / keys.length;
          V.text(g, 'settings list, exchanged by all-gather at startup', A.x + 10, yy - 6, { size: 11, color: C.muted, weight: 600, maxW: A.w - 20 });
          ['rank 0', 'rank 1'].forEach((rn, r) => {
            V.text(g, rn, A.x + 10, yy + 20 + r * 30, { size: 10.5, color: r ? C.b : TF(), weight: 600 });
            keys.forEach((k, i) => {
              const x = A.x + 10 + lw + i * cw, on = e > 2.2 + i * 0.12;
              chip(g, x + 1, yy + 6 + r * 30, cw - 3, 22, vals[k], { fill: V.alpha(on ? C.a : C.muted, on ? 0.14 : 0.06), stroke: on ? V.alpha(C.a, 0.6) : V.alpha(C.muted, 0.3), size: wide ? 10.5 : 9.5 });
              if (r === 0) V.text(g, k, x + cw / 2, yy + 70, { size: 9, font: 'mono', color: C.muted, align: 'center', maxW: cw - 2 });
            });
          });
          if (e > 2.2 + keys.length * 0.12) { mark(g, A.x + 18, yy + 90, true, 5); V.wrap(g, 'equal: it comes up. If a TF_GLM_* differed, the server would refuse (“the two ranks were started with different settings”). When idle, rank 1 waits on the TCPStore doorbell.', A.x + 32, yy + 94, A.w - 44, { size: wide ? 11 : 10, color: C.ink, lineH: 14, maxLines: 3 }); }
        }
      }
      function scene6(g, A, e, C, wide) {
        const refitAt = 2.4, restartAt = 3.4;
        const bars = [{ n: 'head (rank 0)', est: 88.09, c: TF() }, { n: 'worker (rank 1)', est: 80, c: C.b }];
        const bx = A.x + (wide ? 130 : 96), bw = A.w - (wide ? 150 : 108), bh = 20;
        const tt = S.low && e > restartAt ? e - restartAt : e;
        const prog = (i) => { let p = Math.min(1, tt / 4.2); if (S.low && e < restartAt) p = Math.min(p, 0.12); return p; };
        bars.forEach((b, i) => {
          const y = A.y + 8 + i * (bh + 10);
          V.text(g, b.n, A.x + 8, y + 14, { size: wide ? 11.5 : 10, color: b.c, weight: 600, maxW: bx - A.x - 12 });
          V.box(g, bx, y, bw, bh, { r: 6, fill: V.alpha(C.muted, 0.08), stroke: V.alpha(C.muted, 0.35) });
          const p = prog(i);
          g.fillStyle = V.alpha(b.c, 0.5); V.rr(g, bx + 1, y + 1, (bw - 2) * p, bh - 2, 5); g.fill();
          V.text(g, nf(b.est * p, 1) + (i === 0 ? ' / 88.09 GiB (estimate)' : ' / ~80 GiB of weights (README)'), bx + 8, y + 14, { size: wide ? 10.5 : 9, font: 'mono', color: C.ink, maxW: bw - 12 });
        });
        // chart: GiB on the GPU at each 15 s heartbeat (what start.sh prints)
        const cy = A.y + 2 * (bh + 10) + 22, chH = wide ? 120 : 110, cx = A.x + 40, cw2 = A.w - 56;
        V.text(g, 'GiB on each Spark\'s GPU, one heartbeat every 15 s (schematic curve)', A.x + 8, cy - 6, { size: wide ? 10.5 : 9.5, color: C.muted, maxW: A.w - 16 });
        V.line(g, cx, cy + chH, cx + cw2, cy + chH, { color: C.muted, lw: 1 }); V.line(g, cx, cy + 4, cx, cy + chH, { color: C.muted, lw: 1 });
        [0, 44, 88].forEach((v) => V.text(g, String(v), cx - 6, cy + chH - (v / 92) * (chH - 6) + 4, { size: 9.5, font: 'mono', color: C.muted, align: 'right' }));
        ['0', '2 min', '4 min', '6 min'].forEach((l, i) => V.text(g, l, cx + (i / 3) * cw2, cy + chH + 13, { size: 9.5, font: 'mono', color: C.muted, align: 'center' }));
        const beats = Math.floor(tt / 0.17);
        bars.forEach((b, i) => {
          const kMax = S.low && e < restartAt ? Math.min(beats, 5) : Math.min(beats, 24);
          const pts = [];
          for (let k = 0; k <= kMax; k++) {
            const f = Math.min(1, k / 24), gib = b.est * (S.low && e < restartAt ? Math.min(f, 0.12) : (1 - Math.pow(1 - f, 1.6)));
            pts.push([cx + (k / 24) * cw2 * (4 / 6), cy + chH - (gib / 92) * (chH - 6), k]);
          }
          g.beginPath(); pts.forEach(([x, y], k) => (k ? g.lineTo(x, y) : g.moveTo(x, y)));
          g.strokeStyle = b.c; g.lineWidth = 2; g.stroke();
          pts.forEach(([x, y, k]) => { if (k % 4 === 0) V.dot(g, x, y, 2.5, b.c); });
        });
        const ly = cy + chH + 24;
        if (S.low && e > 0.8 && e < restartAt + 1) {
          V.box(g, A.x + 8, ly, A.w - 16, 38, { r: 7, fill: V.alpha(RED(), 0.1), stroke: RED() });
          V.text(g, 'rank 0: largest fitting prompt-plus-reply window: 786432 tokens', A.x + 16, ly + 16, { size: wide ? 10.5 : 9, font: 'mono', color: C.ink, maxW: A.w - 32 });
          V.text(g, e > refitAt ? '→ start.sh restarts once with --context 786432 and warns (illustrative value)' : 'refused before loading the weights (illustrative value)', A.x + 16, ly + 31, { size: wide ? 10 : 9, color: e > refitAt ? C.d : C.muted, maxW: A.w - 32, weight: e > refitAt ? 600 : 400 });
        } else if (tt > 4.4) { mark(g, A.x + 18, ly + 12, true, 5); V.text(g, 'both ranks loaded; the 1st start of each image also compiles the kernels', A.x + 30, ly + 16, { size: wide ? 11 : 9.5, color: C.ink, maxW: A.w - 40 }); }
      }
      function scene7(g, A, e, C, wide) {
        const bw = Math.min(260, A.w * 0.7);
        if (e > 0.2) { V.box(g, A.x + 10, A.y + 10, bw, 30, { r: 12, fill: V.alpha(C.b, 0.16), stroke: C.b }); V.text(g, 'Reply with OK.', A.x + 22, A.y + 30, { size: 12, color: C.ink }); V.text(g, 'temperature 0 · max_tokens 32 · thinking off', A.x + 14, A.y + 54, { size: wide ? 10.5 : 9.5, font: 'mono', color: C.muted, maxW: A.w - 20 }); }
        if (e > 1.0) { V.box(g, A.x + A.w - 90, A.y + 64, 80, 30, { r: 12, fill: V.alpha(C.a, 0.18), stroke: C.a }); V.text(g, 'OK', A.x + A.w - 50, A.y + 84, { size: 13, weight: 700, color: C.ink, align: 'center' }); }
        if (e > 1.8) {
          const y = A.y + 116, pulse = 0.6 + 0.4 * Math.sin(e * 3);
          V.glow(g, A.x + A.w / 2, y + 18, 120, C.a, 0.35 * pulse);
          V.box(g, A.x + 10, y, A.w - 20, 40, { r: 10, fill: V.alpha(C.a, 0.14), stroke: C.a, lw: 2 });
          V.text(g, '✔ GLM-5.3-Flash-EXL3 is now LIVE! on port 8888', A.x + A.w / 2, y + 25, { size: wide ? 13.5 : 11, font: 'mono', weight: 600, color: C.ink, align: 'center', maxW: A.w - 30 });
        }
      }
      const SC = [scene0, scene1, scene2, scene3, scene4, scene5, scene6, scene7];
      H3 = V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const wide = w >= 640;
        const i = st.i;
        if (i !== prevI) { moveFrom = prevI; prevI = i; moveT0 = t; }
        const ly = 20, lx0 = 22, lx1 = w - 22, step = (lx1 - lx0) / 7;
        V.line(g, lx0, ly, lx1, ly, { color: V.alpha(C.muted, 0.4), lw: 2 });
        NAMES.forEach((nm, k) => {
          const x = lx0 + k * step, on = k === i, past = k < i;
          V.dot(g, x, ly, on ? 8 : 6, on ? TF() : past ? V.alpha(C.a, 0.8) : V.alpha(C.muted, 0.35));
          V.text(g, String(k + 1), x, ly + 4, { size: 9.5, weight: 700, color: on || past ? onTile() : C.ink, align: 'center' });
          if (wide || on) V.text(g, nm, Math.max(30, Math.min(w - 30, x)), ly + 22, { size: wide ? 10.5 : 10, color: on ? C.ink : C.muted, align: 'center', weight: on ? 600 : 400 });
        });
        const u = V.ease.inOut(Math.min(1, (t - moveT0) / 0.6));
        const px = V.lerp(lx0 + moveFrom * step, lx0 + i * step, u);
        V.glow(g, px, ly, 16, TF(), 0.7);
        const A = { x: 4, y: 52, w: w - 8, h: h - 58 };
        g.save(); g.beginPath(); g.rect(0, A.y - 4, w, h - A.y + 4); g.clip();
        SC[i](g, A, st.elapsed(), C, wide);
        g.restore();
      });
    })();

    /* ================================================ FIG 4 — memory budget */
    (function () {
      const fig = root.querySelector('#fig-orcamento');
      adaptH(fig, 420, 560);
      const S = { mem: 116, res: 14.5, cap: 12.5, vis: true };
      const GiB = 1073741824, BPT = 6204, WIN = 1048576, ENT = 32 * 45 * 1048576;
      // beyond the estimate: FIX from startup (inferred: ~10 at peak − 1.4 from the 1M prompt); D1M = 7.0 − 5.6 GiB measured
      const FIX = 8.6, D1M = 1.4;
      const ctl = UI.controls(fig);
      UI.slider(ctl, { label: 'MemAvailable at startup', min: 100, max: 122, step: 0.5, value: 116, fmt: (v) => nf(v, 1) + ' GiB', onInput: (v) => { S.mem = v; upd(); } });
      UI.slider(ctl, { label: 'MEMORY_RESERVE_GIB', min: 4, max: 24, step: 0.5, value: 14.5, fmt: (v) => nf(v, 1), onInput: (v) => { S.res = v; upd(); } });
      UI.slider(ctl, { label: 'KV_POOL_GIB', min: 3, max: 24, step: 0.5, value: 12.5, fmt: (v) => nf(v, 1), onInput: (v) => { S.cap = v; upd(); } });
      UI.toggle(ctl, { label: 'VISION=1 (tower on rank 0)', value: true, onChange: (v) => { S.vis = v; upd(); } });
      const rP = UI.readout(ctl, 'pool', '—'), rN = UI.readout(ctl, 'pool ≈ tokens', '—'), rI = UI.readout(ctl, 'idle free', '—'), rF = UI.readout(ctl, 'min. free under 1M', '—'), rV = UI.readout(ctl, 'verdict', '—');
      let M = {};
      function calc() {
        const est = 88.09 - (S.vis ? 0 : 1.8);
        const room = S.mem - S.res - est;
        const winG = (WIN * BPT) / GiB;
        if (room < 0) {
          const avail = winG + room;
          const win = avail > 0 ? Math.floor((avail * GiB) / BPT / 2048) * 2048 : 0;
          return { est, room, pool: 0, refused: true, win, tokens: win, idleFree: S.res - FIX, minFree: S.res - FIX - D1M, left: 0 };
        }
        const pool = Math.min(S.cap, room);
        const tokens = WIN + Math.floor(Math.max(0, pool * GiB - ENT) / BPT / 2048) * 2048;
        const left = room - pool;
        return { est, room, pool, refused: false, win: WIN, tokens, idleFree: S.res + left - FIX, minFree: S.res + left - FIX - D1M, left };
      }
      function upd() {
        M = calc();
        if (M.refused) { rP.set('—', 'bad'); rN.set(M.win > 0 ? 'window ' + nf(M.win) : 'does not fit', 'bad'); rV.set(M.win > 0 ? 'refusal → refit' : 'does not fit', 'bad'); }
        else { rP.set(nf(M.pool, 2) + ' GiB' + (M.pool >= S.cap - 1e-9 ? ' (ceiling)' : '')); rN.set('~' + nf(M.tokens / 1e6, 2) + 'M'); rV.set(M.pool >= S.cap - 1e-9 ? 'fits, pool at ceiling' : 'fits, pool limited by memory', 'ok'); }
        const mf = M.minFree, fi = M.idleFree;
        if (M.refused) { rI.set('—'); rF.set('—'); }
        else {
          rI.set('~' + nf(Math.max(0, fi), 1) + ' GiB', fi < 1 ? 'bad' : fi < 3 ? 'warn' : 'ok');
          rF.set('~' + nf(Math.max(0, mf), 1) + ' GiB', mf < 1 ? 'bad' : mf < 3 ? 'warn' : 'ok');
        }
        if (!M.refused && mf < 1) rV.set('risk of freezing', 'bad');
      }
      upd();
      const snow = Array.from({ length: 26 }, (_, i) => ({ x: (i * 0.618) % 1, s: 0.4 + ((i * 0.37) % 1) * 0.8 }));
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const wide = w >= 640;
        const top = 16, bot = h - 26, Hh = bot - top;
        const MAXV = 125, BRK = 75;
        const yv = (v) => { const f = v < BRK ? (v / BRK) * 0.18 : 0.18 + ((v - BRK) / (MAXV - BRK)) * 0.82; return bot - f * Hh; };
        const tx = wide ? 70 : 52, twid = wide ? Math.min(150, w * 0.2) : Math.min(110, w * 0.3);
        // axis
        [0, 75, 85, 95, 105, 115, 125].forEach((v) => { V.line(g, tx - 6, yv(v), tx + twid + 6, yv(v), { color: V.alpha(C.grid, 1), lw: 1 }); V.text(g, v + ' GiB', tx - 10, yv(v) + 4, { size: 10, font: 'mono', color: C.muted, align: 'right' }); });
        // axis break
        const yb = yv(BRK) + 6; g.strokeStyle = C.muted; g.lineWidth = 1.2; g.beginPath(); g.moveTo(tx - 4, yb + 3); g.lineTo(tx + 4, yb - 3); g.moveTo(tx - 4, yb + 8); g.lineTo(tx + 4, yb + 2); g.stroke();
        // tank
        const memTop = yv(S.mem);
        V.box(g, tx, memTop, twid, bot - memTop, { r: 6, stroke: V.alpha(C.ink, 0.6), lw: 1.6 });
        const segs = [];
        let acc = 0;
        const push = (v, col, lab, hatch) => { if (v <= 0) return; segs.push({ a: acc, b: acc + v, col, lab, hatch }); acc += v; };
        const est = M.est;
        push(80, V.alpha(C.b, 0.45), 'weights ~80');
        push(6.06, V.alpha(C.c, 0.55), '1M window');
        if (S.vis) push(1.8, V.alpha(C.f, 0.6), 'vision');
        push(est - acc, V.alpha(C.muted, 0.4), '');
        if (!M.refused) { push(M.pool, V.alpha(C.e, 0.6), 'pool ' + nf(M.pool, 1)); push(M.left, null, M.left > 0.6 ? 'leftover' : ''); }
        const resA = S.mem - S.res;
        segs.forEach((s) => {
          const ya = yv(Math.min(s.b, S.mem)), ybb = yv(Math.max(0, s.a));
          if (ybb - ya < 0.5) return;
          if (s.col) { g.fillStyle = s.col; g.fillRect(tx + 2, ya, twid - 4, ybb - ya); }
          if (s.lab && ybb - ya >= 13) V.text(g, s.lab, tx + twid / 2, (ya + ybb) / 2 + 4, { size: 10.5, color: C.ink, align: 'center', weight: 600, maxW: twid - 8 });
        });
        // reserve (hatched)
        if (resA > 0) {
          const ya = memTop, ybb = yv(Math.max(0, Math.min(S.mem, resA)));
          g.save(); g.beginPath(); g.rect(tx + 2, ya + 1, twid - 4, ybb - ya); g.clip();
          g.strokeStyle = V.alpha(C.d, 0.55); g.lineWidth = 1.2;
          for (let k = -40; k < twid + 40; k += 8) { g.beginPath(); g.moveTo(tx + k, ya); g.lineTo(tx + k + 30, ybb + 30); g.stroke(); }
          g.restore();
          V.line(g, tx, ybb, tx + twid, ybb, { color: C.d, lw: 1.5, dash: [5, 3] });
        }
        // beyond the estimate: fixed block from startup + pulse of a 1M prompt
        const ph = (t % 4) / 4, amp = Math.max(0, Math.sin(Math.min(1, ph / 0.6) * Math.PI));
        const fixA = M.refused ? est : est + M.pool, fixB = fixA + FIX;
        if (!M.refused) {
          const ya = yv(Math.min(fixB, MAXV)), ybb = yv(fixA);
          g.fillStyle = V.alpha(C.d, 0.62); g.fillRect(tx + 2, ya, twid - 4, ybb - ya);
          g.strokeStyle = C.d; g.lineWidth = 1.2; g.strokeRect(tx + 2.5, ya + 0.5, twid - 5, ybb - ya - 1);
          if (ybb - ya >= 26) { V.text(g, 'outside the', tx + twid / 2, (ya + ybb) / 2 - 2, { size: 10, color: C.ink, align: 'center', weight: 600 }); V.text(g, 'estimate ~8.6', tx + twid / 2, (ya + ybb) / 2 + 10, { size: 10, color: C.ink, align: 'center', weight: 600, maxW: twid - 8 }); }
        }
        const pTop = fixB + D1M * amp;
        const outside = wide;                       // on desktop, labels beside the column; on mobile, inside
        const ax = tx + twid + 8;
        if (!M.refused && amp > 0.02) {
          const ya = yv(Math.min(pTop, MAXV)), ybb = yv(fixB);
          g.fillStyle = V.alpha(RED(), 0.45 + 0.35 * amp); g.fillRect(tx + 2, ya, twid - 4, ybb - ya);
          V.glow(g, tx + twid / 2, (ya + ybb) / 2, 22, RED(), 0.5 * amp);
          if (outside) { V.line(g, tx + twid - 2, (ya + ybb) / 2, ax + 4, (ya + ybb) / 2, { color: RED(), lw: 1.2 }); V.text(g, '+1.4 GiB (1M prompt)', ax + 8, (ya + ybb) / 2 + 4, { size: 10, color: RED(), weight: 700 }); }
          else if (ybb - ya >= 9) V.text(g, '+1.4 (1M)', tx + twid / 2, (ya + ybb) / 2 + 3.5, { size: 9, color: C.dark ? C.ink : '#ffffff', align: 'center', weight: 700 });
        }
        if (!M.refused && resA > 0) V.line(g, tx, yv(resA), tx + twid, yv(resA), { color: C.d, lw: 1.5, dash: [5, 3] });
        // what is free right now (above the block and the pulse)
        if (!M.refused) {
          const freeNow = S.mem - pTop;
          const ya = memTop, ybb = yv(Math.min(S.mem, pTop));
          if (freeNow > 0) {
            if (outside) {
              const ym = (ya + ybb) / 2;
              g.strokeStyle = C.ink; g.lineWidth = 1.2; g.beginPath(); g.moveTo(ax, ya + 1); g.lineTo(ax + 4, ya + 1); g.lineTo(ax + 4, ybb - 1); g.lineTo(ax, ybb - 1); g.stroke();
              V.text(g, 'free now', ax + 8, ym - 2, { size: 10, color: C.ink, weight: 600 });
              V.text(g, '~' + nf(freeNow, 1) + ' GiB', ax + 8, ym + 11, { size: 10, font: 'mono', color: C.ink, weight: 600 });
            } else if (ybb - ya >= 26) {
              V.text(g, 'free now', tx + twid / 2, ya + (ybb - ya) / 2 - 1, { size: 9.5, color: C.ink, align: 'center', weight: 600 });
              V.text(g, '~' + nf(freeNow, 1) + ' GiB', tx + twid / 2, ya + (ybb - ya) / 2 + 11, { size: 10, font: 'mono', color: C.ink, align: 'center', weight: 600 });
            }
          }
        }
        V.text(g, 'MemAvailable', tx + twid / 2, memTop - 5, { size: 10.5, font: 'mono', color: C.ink, align: 'center' });
        // text panel
        const px = tx + twid + (wide ? 136 : 16), pw = w - px - 8;
        const lines = [
          [C.b, 'weights ~80 GiB (EXL3 experts + q4 dense + drafter)'],
          [C.c, '1M window ≈ 6.1 GiB (6,204 B/token, estimate)'],
          [C.f, S.vis ? 'vision tower 1.8 GiB (rank 0)' : 'VISION=0: −1.8 GiB, back to the pool'],
          [C.muted, 'startup estimate: ' + nf(est, 2) + ' GiB'],
          [C.e, M.refused ? 'pool: nothing (the window does not fit)' : 'pool: ' + nf(M.pool, 2) + ' GiB → ~' + nf(M.tokens) + ' tokens'],
          [V.alpha(C.d, 0.4), 'reserve: ' + nf(S.res, 1) + ' GiB (the budget leaves it free)'],
          [V.alpha(C.d, 0.85), 'outside the estimate from startup: ~8.6 GiB (CUDA, NCCL, graphs, workspaces; inferred)'],
          [RED(), '1M prompt: another ~1.4 GiB (7.0 → 5.6 GiB measured)'],
        ];
        let yy = top + 8;
        const fs = wide ? 12 : 10.5, lh = wide ? 17 : 14;
        lines.forEach(([c, s]) => { g.fillStyle = c; V.rr(g, px, yy - 8, 9, 9, 2); g.fill(); yy += V.wrap(g, s, px + 15, yy, pw - 16, { size: fs, color: C.ink, lineH: lh, maxLines: wide ? 2 : 3 }) + (wide ? 4 : 3); });
        yy += 6;
        const mf = M.minFree, fi = M.idleFree;
        let verdict, vc;
        if (M.refused) { verdict = M.win > 0 ? 'Does not fit: the server refuses before loading and reports the largest window (~' + nf(M.win) + ' tokens); start.sh restarts with it.' : 'Not even the window fits: free up memory or lower the reserve carefully.'; vc = RED(); }
        else if (mf < 1) { verdict = 'Minimum free ≈ ' + nf(Math.max(0, mf), 1) + ' GiB: the fixed block and the pulse of a 1M prompt eat the margin. In UMA this tends to freeze the Spark.'; vc = RED(); }
        else if (mf < 3) { verdict = 'Idle ≈ ' + nf(fi, 1) + ' GiB free; under 1M ≈ ' + nf(mf, 1) + ' GiB: tight (minimum free ≈ reserve + leftover − ~10).'; vc = C.d; }
        else { verdict = 'Idle ≈ ' + nf(fi, 1) + ' GiB free; under a 1M prompt ≈ ' + nf(mf, 1) + ' GiB (measured with the defaults: 7.0 and 5.6). Measured pool: 2,922,496 tokens.'; vc = C.a; }
        yy += V.wrap(g, verdict, px, yy, pw, { size: fs, color: vc, weight: 600, lineH: lh, maxLines: 5 }) + 14;
        // how 4 streams split the pool
        if (!M.refused && yy + 60 < h) {
          const T = M.tokens, cyc = Math.floor(t / 4) % 2;
          const sh = Math.max(0, Math.floor((T - 1048576) / 3 / 2048) * 2048), shl = '~' + nf(sh / 1000, 0) + 'k';
          const ex = cyc === 0 ? [[1048576, C.a, '1M'], [1048576, C.b, '1M']] : [[1048576, C.a, '1M'], [sh, C.c, shl], [sh, C.e, shl], [sh, hsl(205, 1), shl]];
          V.text(g, cyc === 0 ? 'the pool split: 1M next to another 1M' : 'the pool split: 1M next to three of ' + shl, px, yy, { size: fs - 0.5, color: C.muted, maxW: pw });
          const bY = yy + 8, bW = pw, bH = 18;
          V.box(g, px, bY, bW, bH, { r: 5, fill: V.alpha(C.muted, 0.1), stroke: V.alpha(C.muted, 0.4) });
          let x0 = px + 1;
          ex.forEach(([n, c, l]) => { const ww = Math.min(bW - 2 - (x0 - px - 1), (n / T) * (bW - 2)); if (ww <= 0) return; g.fillStyle = V.alpha(c, 0.6); V.rr(g, x0, bY + 1, ww, bH - 2, 4); g.fill(); if (ww > 26) V.text(g, l, x0 + ww / 2, bY + 13, { size: 10, font: 'mono', color: C.ink, align: 'center' }); x0 += ww + 1; });
          const used = ex.reduce((a, e2) => a + e2[0], 0);
          V.text(g, used <= T ? 'fits: ~' + nf(Math.max(0, T - used) / 1e3, 0) + 'k left for kept prompts' : 'does not fit now: the request waits (kept prompts give way first)', px, bY + bH + 14, { size: fs - 1, color: used <= T ? C.ink : C.d, maxW: pw });
        }
        // freezing
        if (!M.refused && mf < 1) {
          snow.forEach((s, k) => { const y = top + ((t * 30 * s.s + k * 37) % Hh); V.text(g, '❄', tx + 6 + s.x * (twid - 12), y, { size: 10 + s.s * 6, color: V.alpha(C.b, 0.75), align: 'center' }); });
          const cyy = top + Hh * 0.5; g.fillStyle = V.alpha(C.bg || '#000', 0.75); V.rr(g, tx + 8, cyy - 16, twid - 16, 26, 6); g.fill(); V.text(g, 'freezes ❄', tx + twid / 2, cyy + 2, { size: 13, weight: 700, color: RED(), align: 'center' });
        }
        V.text(g, 'scale compressed below 75 GiB', tx + twid / 2, h - 8, { size: 9.5, color: C.muted, align: 'center', maxW: twid + 60 });
      });
    })();

    /* ================================================ FIG 5 — formats: q4 and FP8 */
    (function () {
      const fig = root.querySelector('#fig-formatos');
      const N = LA.num;
      const S = { mode: 'q4', auto: true, i: 0, scale: 'pow2', seed: 3 };
      const stage5 = fig.querySelector('.la-fig-stage');
      const setH5 = () => { const want = (fig.clientWidth < 600 ? (S.mode === 'q4' ? 430 : 530) : 410) + 'px'; if (stage5.style.height !== want) stage5.style.height = want; };
      setH5();
      { const ro5 = new ResizeObserver(setH5); ro5.observe(fig); if (LA.onCleanup) LA.onCleanup(() => ro5.disconnect()); }
      const bf = (x) => (N && N.round ? N.round('bf16', x) : x);
      let grp = [], Q = [];
      function mkGroup() {
        const r = V.rng(102);
        grp = Array.from({ length: 64 }, () => { const u = r(); return (u < 0.5 ? -1 : 1) * -Math.log(1 - Math.abs(2 * u - 1) + 1e-9) * 0.7; });
        Q = [];
        const mn = Math.min(...grp), mx = Math.max(...grp);
        for (let i = 0; i < 16; i++) {
          const a = 1 - (0.45 * i) / 15;
          const lo = bf(a * mn), hi = a * mx;
          const s = bf((hi - lo) / 15);
          const q = grp.map((g) => Math.max(0, Math.min(15, Math.round((g - lo) / s))));
          const dq = q.map((c) => s * c + lo);
          const mse = grp.reduce((acc, g, k) => acc + (g - dq[k]) ** 2, 0) / 64;
          Q.push({ a, lo, hi, s, q, dq, mse });
        }
      }
      mkGroup();
      const best = () => Q.reduce((b, x, k) => (x.mse < Q[b].mse ? k : b), 0);
      let row = [], R2 = null;
      function mkRow() {
        const r = V.rng(100 + S.seed * 13);
        row = Array.from({ length: 512 }, () => r.normal() * 0.8);
        for (let k = 0; k < 4; k++) row[r.int(512)] = (r() < 0.5 ? -1 : 1) * (2.6 + r() * 1.4);
        R2 = null;
      }
      mkRow();
      function quantRow(mode) {
        const amax = Math.max(...row.map(Math.abs));
        const s = mode === 'pow2' ? Math.pow(2, Math.ceil(Math.log2(amax)) - 8) : amax / 448;
        const xh = row.map((x) => (N && N.round ? N.round('e4m3', x / s) : Math.round(x / s)) * s);
        let num = 0, den = 0; for (let k = 0; k < 512; k++) { num += (row[k] - xh[k]) ** 2; den += row[k] ** 2; }
        const rms = Math.sqrt(den / 512);
        return { amax, s, xh, rel: Math.sqrt(num / den), rms, top: amax / s };
      }
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Format', options: [{ v: 'q4', label: 'q4 · group of 64' }, { v: 'fp8', label: 'FP8 · KV row' }], value: 'q4', onChange: (v) => { S.mode = v; gQ.show(v === 'q4'); gF.show(v === 'fp8'); setH5(); upd(); } });
      const gQ = UI.group(ctl), gF = UI.group(ctl);
      const tg = UI.toggle(gQ.el, { label: 'automatic search', value: true, onChange: (v) => { S.auto = v; } });
      UI.slider(gQ.el, { label: 'clipping i', min: 0, max: 15, step: 1, value: 0, fmt: (v) => 'i = ' + v + ' · a = ' + nf(1 - (0.45 * v) / 15, 2), onInput: (v) => { S.i = v; if (S.auto) { S.auto = false; tg.set(false); } upd(); } });
      UI.seg(gF.el, { label: 'Scale', options: [{ v: 'pow2', label: 'power of 2' }, { v: 'fine', label: 'fine fp32 (amax/448)' }], value: 'pow2', onChange: (v) => { S.scale = v; upd(); } });
      UI.button(gF.el, '↻ new row', () => { S.seed++; mkRow(); upd(); });
      gF.show(false);
      const r1 = nocase(UI.readout(ctl, 'a / scale', '—')), r2 = nocase(UI.readout(ctl, 'error', '—')), r3 = nocase(UI.readout(ctl, 'reference', '—'));
      let curI = 0;
      function upd() {
        if (S.mode === 'q4') {
          const k = S.auto ? curI : S.i, b = best();
          r1.set('a = ' + nf(Q[k].a, 2) + ' · s = ' + nf(Q[k].s, 3));
          r2.set('MSE ' + nf(Q[k].mse, 4), k === b ? 'ok' : null);
          r3.set('best i* = ' + b + ' (MSE ' + nf(Q[b].mse, 4) + ')');
        } else {
          const A = quantRow('pow2'), B = quantRow('fine'), cur = S.scale === 'pow2' ? A : B;
          const e2 = Math.round(Math.log2(A.s)); r1.set('s = ' + (S.scale === 'pow2' ? '2' + (e2 < 0 ? '⁻' : '') + String(Math.abs(e2)).split('').map((d) => '⁰¹²³⁴⁵⁶⁷⁸⁹'[+d]).join('') : nf(B.s, 5)));
          r2.set(nf(cur.rel * 100, 2) + '% (row)');
          r3.set('other scale: ' + nf((S.scale === 'pow2' ? B : A).rel * 100, 2) + '%');
        }
      }
      upd();
      let lastScan = -1;
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const wide = w >= 600;
        if (S.mode === 'q4') {
          if (S.auto) { const ph = t % 7; const k = ph < 4.8 ? Math.min(15, Math.floor(ph / 0.3)) : best(); curI = k; if (k !== lastScan) { lastScan = k; upd(); } }
          const k = S.auto ? curI : S.i, cur = Q[k];
          const cell = wide ? Math.min(31, (w * 0.64 - 30) / 16 - 2) : Math.min(21, (w - 34) / 16 - 1.5);
          const gap = wide ? 2 : 1.5;
          const mW = 8 * cell + 7 * gap;
          const mx0 = 10, my0 = 22;
          const data = []; for (let i = 0; i < 8; i++) data.push(grp.slice(i * 8, i * 8 + 8));
          const gm = Math.max(...grp.map(Math.abs));
          const compact = cell < 26;
          cells(g, mx0, my0, data, cell, cell, gap, (v) => divFill(v, gm), (v) => cellNum(v, compact && v < 0), compact ? 'g (64 weights; −.8 = −0.8)' : 'g (64 weights, heavy tail)');
          const codes = []; const errs = [];
          for (let i = 0; i < 8; i++) { codes.push(cur.q.slice(i * 8, i * 8 + 8)); errs.push(grp.slice(i * 8, i * 8 + 8).map((v, j) => Math.abs(v - cur.dq[i * 8 + j]))); }
          const m2x = mx0 + mW + (wide ? 22 : 12);
          cells(g, m2x, my0, codes, cell, cell, gap, (v, i, j) => heatFill(errs[i][j] / (Q[0].s * 0.8)), (v) => String(v), 'q (4 bits) · color = error');
          // MSE bars
          const bx = wide ? m2x + mW + 26 : 12, by = wide ? 24 : my0 + 8 * (cell + gap) + 28, bw = wide ? w - bx - 12 : w - 24, bh = wide ? h * 0.42 : Math.max(90, h - by - 104);
          V.text(g, 'MSE of each clipping a_i', bx, by - 6, { size: 11, color: C.muted });
          const mxm = Math.max(...Q.map((q) => q.mse));
          const bb = best();
          V.bars(g, { x: bx, y: by + 4, w: bw, h: bh - 22, values: Q.map((q) => q.mse), max: mxm * 1.05, gap: 3, colorFn: (v, i) => (i === k ? TF() : i === bb ? C.a : V.alpha(C.b, 0.5)) });
          [0, 5, 10, 15].forEach((i) => V.text(g, String(i), bx + (i + 0.5) * (bw / 16), by + bh - 4, { size: 10, font: 'mono', color: C.muted, align: 'center' }));
          V.text(g, 'i →', bx + bw, by + bh + 10, { size: 10, color: C.muted, align: 'right' });
          // number line of the 16 levels
          const ny = (wide ? Math.max(my0 + 8 * (cell + gap), by + bh) : by + bh) + 46, nx0 = 14, nx1 = w - 14;
          const vmin = -Math.max(...grp.map(Math.abs)) * 1.05, vmax = -vmin, xv = (v) => nx0 + ((v - vmin) / (vmax - vmin)) * (nx1 - nx0);
          V.line(g, nx0, ny, nx1, ny, { color: C.muted, lw: 1 });
          for (let c = 0; c <= 15; c++) { const v = cur.lo + cur.s * c; V.line(g, xv(v), ny - 7, xv(v), ny + 7, { color: TF(), lw: 1.4 }); }
          const lblX = (x) => V.clamp(x, nx0 + 36, nx1 - 36);
          V.text(g, 'lo = ' + (cur.lo < 0 ? '−' : '') + nf(Math.abs(cur.lo), 2), lblX(xv(cur.lo)), ny + 20, { size: 10, font: 'mono', color: C.muted, align: 'center' });
          V.text(g, 'hi = ' + nf(cur.lo + 15 * cur.s, 2), lblX(xv(cur.lo + 15 * cur.s)), ny + 20, { size: 10, font: 'mono', color: C.muted, align: 'center' });
          grp.forEach((v, j) => { const clipped = v < cur.lo - cur.s / 2 || v > cur.lo + 15.5 * cur.s; V.dot(g, xv(v), ny - 13 - (j % 4) * 3, clipped ? 3 : 2, clipped ? RED() : V.alpha(C.b, 0.75)); });
          V.text(g, '16 levels between lo and hi; red dots = clipped', (nx0 + nx1) / 2, ny + 36, { size: wide ? 11 : 10, color: C.muted, align: 'center', maxW: nx1 - nx0 });
        } else {
          if (!R2) R2 = { A: quantRow('pow2'), B: quantRow('fine') };
          const cur = S.scale === 'pow2' ? R2.A : R2.B;
          const cell = wide ? Math.min(40, (w * 0.55 - 20) / 8 - 3) : Math.min(38, (w - 30) / 8 - 3);
          const gap = 3;
          const data = [], er = [];
          for (let i = 0; i < 4; i++) { data.push(row.slice(i * 8, i * 8 + 8)); er.push(row.slice(i * 8, i * 8 + 8).map((v, j) => Math.abs(v - cur.xh[i * 8 + j]) / Math.max(1e-6, Math.abs(v)))); }
          cells(g, 10, 22, data, cell, 22, gap, (v) => divFill(v, 3), (v) => (Math.abs(v) < 0.005 ? '0.00' : (v < 0 ? '−' : '') + nf(Math.abs(v), 2)), '32 of the row\'s 512 values');
          const y2 = 22 + 4 * (22 + gap) + 24;
          cells(g, 10, y2, er, cell, 22, gap, (v) => heatFill(v / 0.08), (v) => nf(v * 100, 1) + '%', 'relative error of each value');
          // histogram of |x|/s on a log2 scale
          const hx = wide ? 10 + 8 * (cell + gap) + 30 : 12, hy = wide ? 30 : y2 + 4 * (22 + gap) + 30, hw = wide ? w - hx - 12 : w - 24, hh = wide ? 150 : 110;
          const lo2 = -10, hi2 = 9, nb = hi2 - lo2;
          const cnt = new Array(nb).fill(0);
          row.forEach((x) => { const a = Math.abs(x) / cur.s; if (a <= 0) return; const b = Math.floor(Math.log2(a)) - lo2; if (b >= 0 && b < nb) cnt[b]++; });
          const mc = Math.max(...cnt);
          V.text(g, '|x| / s in powers of 2 (e4m3: normals from 2⁻⁶ to 448)', hx, hy - 8, { size: 11, color: C.muted, maxW: hw });
          const bwid = hw / nb;
          const xl = (e) => hx + (e - lo2) * bwid;
          g.fillStyle = V.alpha(C.a, 0.08); g.fillRect(xl(-6), hy, xl(Math.log2(448)) - xl(-6), hh);
          g.fillStyle = V.alpha(TF(), 0.16); g.fillRect(xl(7), hy, xl(8) - xl(7), hh);
          cnt.forEach((c, b) => { const bh = (c / mc) * (hh - 14); g.fillStyle = V.alpha(C.b, 0.75); V.rr(g, hx + b * bwid + 1, hy + hh - bh, bwid - 2, Math.max(1, bh), 2); g.fill(); });
          [-9, -6, 0, 4, 7, 8].forEach((e) => V.text(g, '2' + (e < 0 ? '⁻' : '') + String(Math.abs(e)).split('').map((d) => '⁰¹²³⁴⁵⁶⁷⁸⁹'[+d]).join(''), xl(e), hy + hh + 13, { size: 10, font: 'mono', color: C.muted, align: 'center' }));
          const mxPos = xl(Math.log2(cur.top));
          V.line(g, mxPos, hy - 2, mxPos, hy + hh, { color: TF(), lw: 2 });
          V.text(g, 'amax/s = ' + nf(cur.top, 1), mxPos - 4, hy + 12, { size: 10.5, font: 'mono', color: TF(), align: 'right', weight: 600 });
          // bytes per row
          const by = Math.max(y2 + 4 * (22 + gap) + 30, hy + hh + 34), bw = wide ? w - 22 : hw;
          const hxB = wide ? 10 : hx;
          const rowsB = [['BF16', 1024, C.muted], ['vLLM fp8_ds_mla', 656, GLM()], ['TensorFold FP8', 528, TF()]];
          const ph = Math.min(1, (t % 6) / 1.4);
          rowsB.forEach(([nm, b, c], i) => {
            const y = by + i * 22;
            if (y + 16 > h - 4) return;
            const lw = wide ? 130 : 108;
            V.text(g, nm, hxB, y + 12, { size: 10.5, color: C.ink, maxW: lw - 4 });
            g.fillStyle = V.alpha(c, 0.5); V.rr(g, hxB + lw, y + 2, ((bw - lw - 50) * b / 1024) * ph, 14, 4); g.fill();
            V.text(g, nf(b) + ' B', hxB + lw + (bw - lw - 50) * b / 1024 * ph + 6, y + 13, { size: 10.5, font: 'mono', color: C.ink });
          });
        }
      });
    })();

    /* ================================================ FIG 6 — map of the 53 patches */
    (function () {
      const fig = root.querySelector('#fig-patches');
      adaptH(fig, 440, 650, 640);
      const NAMES = ['glm-exl3-prompt-experts', 'glm-dense-fp8', 'glm-vision', 'glm-prompt-kernels', 'glm-dense-q4', 'cuda-roce-allgather', 'glm-copy-drafts', 'glm-prompt-grid', 'glm-prefill-kernels', 'glm-hc-split', 'glm-draft-sim', 'glm-kda-chunked', 'glm-decode-rounds', 'glm-kda-chunked-gb10', 'glm-shared-prefix', 'glm-decode-kernels', 'glm-overlap-priority', 'glm-noise-policies', 'glm-decode-kernels2', 'glm-prompt-experts-order', 'glm-dflash-policy-env', 'glm-overlap-normal-priority', 'server-effort-max', 'glm-prompt-select-rows', 'glm-dflash2-ring', 'glm-multi-kda', 'glm-multi-dflash2', 'glm-lean-prompt-scratch', 'glm-multi-dsa', 'glm-multi-stream-engine', 'glm-decode-index-regs', 'glm-code-copy-drafts', 'glm-prefill-overlap2', 'cuda-nucleus-union', 'glm-multi-rounds', 'glm-tool-calls', 'cuda-tokenize', 'glm-kv-fp8', 'glm-kda-chunked-kernel', 'glm-parallel-deadlocks', 'glm-parallel-ring-base', 'glm-prompt-replay', 'glm-visible-pools', 'cuda-context-errors', 'cuda-metrics', 'glm-l2-prefetch', 'glm-exl3-decode-loads', 'glm-timing-tokens', 'glm-multi-prefill', 'glm-many-media', 'glm-tool-history-recovery', 'cuda-roce-startup', 'glm-whole-tool-calls'];
      const AREAS = [
        { k: 'pesos', t: 'Dense weights', hue: 150, p: [2, 5], m: 'BF16 → FP8 (e4m3, scale per row × 128 columns) or affine q4 in groups of 64 with MSE clipping; FP8 head, BF16 kv_b.', knob: 'DENSE=q4', fx: 'q4 over fp8: prose 38.9 → 44.4 · code 44.2 → 48.7 · prefill ~1,090 → ~1,260 tok/s', bits: 'lossy' },
        { k: 'kv', t: 'KV cache', hue: 205, p: [38], m: 'DSA latent and grouped indexer keys as e4m3 rows + a 2^k scale (528 B); indexer keys and gates in per-slot rings.', knob: 'KV=fp8', fx: 'the 1M window with 4 requests fits · GSM8K 98.0% · HumanEval 97.6%', bits: 'lossy' },
        { k: 'prompt', t: 'Prompt', hue: 265, p: [1, 4, 9, 20, 24, 28], m: 'EXL3 experts with the rows in L2 (NFIRST), Yᵀ = WᵀXᵀ, DRAM/tensor core order, rotation 1×/row, dense only before 2,051, selection in blocks of 512, lean scratch.', knob: 'always on', fx: 'NFIRST: 2.3–2.5× at 4,096 rows · “faster prefill, less memory at 1M”', bits: '= / ≠ prompt' },
        { k: 'dois', t: 'Prompt, two Sparks', hue: 300, p: [10, 17, 22, 33], m: 'hyper-connections glue split by rows between the ranks; exchanges overlapped with the next block\'s front; side-stream priority default switched to normal.', knob: 'SPLIT=1', fx: '50k: ~1,270 → ~1,730 tok/s (with 0009 and 0020) · 149k 92.8 → 90.7 s · decode −1.5%', bits: '=' },
        { k: 'kda', t: 'Prompt, KDA', hue: 330, p: [12, 14, 39], m: 'delta rule in WY/UT form in a CUDA kernel: 256 threads per head, 32-row sub-chunks, 128×128 state in registers, running products of decays.', knob: 'KDA_CHUNKED=1', fx: '50k 29.3 → 26.4 s · 149k 91.1 → 84.5 s', bits: 'close, not the same bits' },
        { k: 'reuso', t: 'Prompt reuse', hue: 45, p: [8, 15, 42], m: 'states on a grid of 64; shared prefix (end of the system block and divergence point); replay with the kept logits row.', knob: 'SHARED_PREFIX=1', fx: 'identical 64k: 34 s → <0.07 s · 7.9k system: 4.24 → 0.13 s', bits: '=' },
        { k: 'decode', t: 'Decode', hue: 175, p: [13, 16, 19, 31], m: 'windows of up to 16 rows; tokens delivered one round later; tiles per shape; fused epilogues; hc_partial in CUDA with Triton\'s bits; indexer scoring in registers.', knob: 'COPY_MAX=15', fx: 'editing 84 → 119 tok/s (with COPY_MAX 15) · decode KDA 4–36% faster', bits: '=' },
        { k: 'mem', t: 'Decode, memory', hue: 20, p: [46, 47], m: 'L2 prefetch on a side stream during the all-gathers (sites a, f, o); EXL3 trellis in non-coherent 16 B loads one step ahead.', knob: 'TF_GLM_L2PF=1 · TF_GLM_EXL3_LOADS=nc', fx: 'with 512 KiB: prose +2.7%, code +3.3% · 4 req: 74.8 → 76.6 and 100.0 → 102.7', bits: '=' },
        { k: 'idx', t: 'Decode, indexer', hue: 235, p: [43], m: 'scoring and selection limited to each row\'s visible pools, inside the bucket captured in the CUDA graph.', knob: 'always on', fx: '“the same tokens” (no speed figure)', bits: '=' },
        { k: 'link', t: 'Link', hue: 190, p: [6, 52], m: 'small all-gathers over one-shot RDMA (the b12x transport); until the engine is ready, each gather first goes through an NCCL barrier.', knob: 'COMM=roce · TF_ROCE_MAX_KB=512 · TF_ROCE_WAIT_S=20', fx: '11 µs vs 45 µs per 16 KiB · decode +6% · code ×4 +1.4% · fixes #6', bits: '=' },
        { k: 'drafts', t: 'Drafts', hue: 280, p: [7, 18, 21, 25, 32], m: 'copy drafts in place of DFlash2 when a copy exists; copies from the reply only after 16 tokens; noise-aware stop; policy set by variable; DFlash2 window together with the prefixes.', knob: 'COPY · COPY_CODE · DRAFT_POLICY=fnc7:0.3', fx: 'copy 80.3 → 84.4 · COPY_CODE 55.4 → 56.2 · fnc: prose 47.2 → 50.6, code 51.7 → 55.5', bits: '=' },
        { k: 'tool', t: 'Drafts, tooling', hue: 255, p: [11], m: 'DFlash2 teacher-forced dump and an offline numpy simulator: any policy is reproduced exactly, with an oracle as the ceiling.', knob: 'TF_GLM_DRAFT_DUMP (off)', fx: '“how the stop rules were tuned” · β = 2.0 fitted on 9 dumps', bits: '= (dump = serial)' },
        { k: 'conc', t: 'Concurrent requests', hue: 120, p: [26, 27, 29, 30, 35, 40, 41, 48, 49], m: 'shared pool; batched window of up to 32 rows; segmented KDA, DSA and DFlash2; parities + replay; deadlock fixed (transport chosen by bytes); distinct tokens in the timing; multi-prefill.', knob: 'PARALLEL=4 · MULTI_PREFILL=1 · TF_GLM_MULTI_LONE=0', fx: 'prose 60.4 → 108.8 tok/s (1 → 4) · TTFT ×4 590 → 340 ms · 22/22 equal to solo', bits: '=' },
        { k: 'samp', t: 'Sampling', hue: 0, p: [34], m: 'top-p coverage test with the candidates of both ranks together: same draw, fewer whole-shard reads.', knob: 'TENSORFOLD_NUCLEUS_UNION=1 (off)', fx: 'opt-in, no published figure', bits: '=' },
        { k: 'server', t: 'Server', hue: 95, p: [3, 23, 36, 37, 44, 45, 50, 51, 53], m: 'vision (50 images, 4 videos), whole tool calls and a robust history, /tokenize, reasoning_effort max, context_length_exceeded with param, /metrics.', knob: 'VISION · TENSORFOLD_GLM_*', fx: '50 images in 12.5 s · fixes #2 · truncated calls never sent', bits: '= (text)' },
      ];
      const areaOf = new Array(54).fill(0);
      AREAS.forEach((A, i) => A.p.forEach((n) => { areaOf[n] = i; }));
      const S = { area: 12 };
      const ctl = UI.controls(fig);
      const sel = UI.select(ctl, { label: 'Area', options: AREAS.map((A, i) => ({ v: String(i), label: A.t + ' (' + A.p.length + ')' })), value: '12', onChange: (v) => { S.area = +v; upd(); } });
      const rN = UI.readout(ctl, 'patches in area', '—'), rB = UI.readout(ctl, 'bits', '—');
      function upd() { const A = AREAS[S.area]; rN.set(String(A.p.length)); rB.set(A.bits, A.bits.startsWith('lossy') ? 'warn' : A.bits.startsWith('close') ? 'warn' : 'ok'); }
      upd();
      let tiles = [];
      V.fig(fig, (g, w, h, t, dt, St, H) => {
        const C = V.C();
        const wide = w >= 640;
        const sweep = Math.floor((t * 5) % 60);
        const cols = wide ? 9 : 8;
        const gw = wide ? w * 0.52 : w - 16;
        const tsz = Math.min(wide ? 44 : 40, (gw - (cols - 1) * 5) / cols);
        const gx = 8, gy = 40;
        // apply progress bar
        const prog = Math.min(53, sweep);
        V.text(g, 'patch -p0 --forward · TensorFold v0.6.0 site-packages', gx, 13, { size: wide ? 11.5 : 10, font: 'mono', color: C.muted, maxW: gw });
        V.box(g, gx, 20, gw, 8, { r: 4, fill: V.alpha(C.muted, 0.12) });
        g.fillStyle = V.alpha(TF(), 0.8); V.rr(g, gx, 20, gw * prog / 53, 8, 4); g.fill();
        tiles = [];
        for (let n = 1; n <= 53; n++) {
          const i = n - 1, c = i % cols, r = Math.floor(i / cols);
          const x = gx + c * (tsz + 5), y = gy + r * (tsz + 5);
          const ai = areaOf[n], A = AREAS[ai], on = ai === S.area;
          const applied = n <= prog;
          g.fillStyle = hsl(A.hue, on ? 0.95 : applied ? 0.28 : 0.12); V.rr(g, x, y, tsz, tsz, 6); g.fill();
          if (on) { g.strokeStyle = hsl(A.hue, 1); g.lineWidth = 2; V.rr(g, x - 1, y - 1, tsz + 2, tsz + 2, 7); g.stroke(); }
          V.text(g, String(n).padStart(2, '0'), x + tsz / 2, y + tsz / 2 + 4, { size: Math.min(13, tsz * 0.36), font: 'mono', weight: 600, color: on ? onTile() : C.ink, align: 'center', alpha: on ? 1 : 0.75 });
          if (n === prog + 1 && n <= 53) { g.strokeStyle = TF(); g.lineWidth = 2.4; V.rr(g, x - 2, y - 2, tsz + 4, tsz + 4, 7); g.stroke(); }
          tiles.push([x, y, tsz, n]);
        }
        const rowsN = Math.ceil(53 / cols);
        const cur = Math.min(53, prog + 1);
        V.text(g, 'applying ' + String(cur).padStart(4, '0') + '-' + NAMES[cur - 1] + '.patch', gx, gy + rowsN * (tsz + 5) + 12, { size: wide ? 11 : 10, font: 'mono', color: TF(), maxW: gw });
        // click
        if (H.pointer.clicked) { const p = H.pointer; tiles.forEach(([x, y, s, n]) => { if (p.x >= x && p.x <= x + s && p.y >= y && p.y <= y + s) { S.area = areaOf[n]; sel.el.querySelector('select').value = String(S.area); upd(); } }); }
        // panel
        const A = AREAS[S.area];
        const px = wide ? gx + gw + 18 : 10, py = wide ? 22 : gy + rowsN * (tsz + 5) + 32, pw = wide ? w - px - 10 : w - 20;
        g.fillStyle = hsl(A.hue, 0.1); V.rr(g, px, py, pw, h - py - 8, 10); g.fill();
        g.strokeStyle = hsl(A.hue, 0.8); g.lineWidth = 1.5; V.rr(g, px, py, pw, h - py - 8, 10); g.stroke();
        let yy = py + 22;
        V.text(g, A.t, px + 12, yy, { size: wide ? 15 : 13, weight: 700, color: hsl(A.hue, 1), maxW: pw - 24 });
        yy += 18;
        const list = A.p.map((n) => String(n).padStart(4, '0') + ' ' + NAMES[n - 1]);
        const two = list.length > 4, colW = two ? (pw - 24) / 2 : pw - 24, perCol = two ? Math.ceil(list.length / 2) : list.length;
        list.forEach((s, i) => { const c = two ? Math.floor(i / perCol) : 0, rr = two ? i % perCol : i; V.text(g, s, px + 12 + c * colW, yy + rr * 15, { size: wide ? 10.5 : 9, font: 'mono', color: C.muted, maxW: colW - 6 }); });
        yy += perCol * 15 + 10;
        const fs = wide ? 12.5 : 10.5, lh = wide ? 17 : 14;
        yy += V.wrap(g, A.m, px + 12, yy, pw - 24, { size: fs, color: C.ink, lineH: lh, maxLines: 4 }) + 6;
        V.text(g, 'knob', px + 12, yy, { size: 10, color: C.muted, weight: 600 }); yy += 14;
        yy += V.wrap(g, A.knob, px + 12, yy, pw - 24, { size: fs - 0.5, font: 'mono', color: C.c, lineH: lh, maxLines: 2 }) + 6;
        V.text(g, 'measured effect', px + 12, yy, { size: 10, color: C.muted, weight: 600 }); yy += 14;
        yy += V.wrap(g, A.fx, px + 12, yy, pw - 24, { size: fs, color: C.ink, lineH: lh, maxLines: 3 }) + 8;
        const bc = A.bits.startsWith('lossy') || A.bits.startsWith('close') ? TF() : C.a;
        const bwid = Math.min(pw - 24, tw(g, 'bits: ' + A.bits, 11, 'mono', 600) + 18);
        if (yy + 20 < h - 8) chip(g, px + 12, yy, bwid, 20, 'bits: ' + A.bits, { fill: V.alpha(bc, 0.16), stroke: bc, size: 11, weight: 600 });
      });
    })();

    /* ================================================ FIG 7 — a round with keyed noise */
    (function () {
      const fig = root.querySelector('#fig-rodada');
      const VERIFY_MS = [29.2, 35.6, 42.6, 48.8, 53.6, 58.9, 64.5, 66.0], BLOCK_MS = 3.08;
      const WORDS = ['the', 'engine', 'reads', 'weights', 'one', 'time', 'per', 'round', 'and', 'checks', 'each', 'row', 'with', 'a', 'key', 'of', 'target', 'without', 'changing', 'anything', 'in', 'reply', 'final', 'now', 'copy', 'window', 'noise', 'token', 'bits', 'batch', 'state', 'grid', 'pool', 'rank', 'sum'];
      const S = { src: 'dflash', pol: 'fnc', P: 0.3, seed: 4, t0: -4, replays: 0 };
      const stage7 = fig.querySelector('.la-fig-stage');
      const setH7 = () => { const nar = fig.clientWidth < 640; const want = (nar ? (S.src === 'copy' ? 500 : 720) : (S.src === 'copy' ? 410 : 460)) + 'px'; if (stage7.style.height !== want) stage7.style.height = want; };
      setH7();
      { const ro7 = new ResizeObserver(setH7); ro7.observe(fig); if (LA.onCleanup) LA.onCleanup(() => ro7.disconnect()); }
      const mix = (a, b, c) => { let h = (a ^ 0x9e3779b9) >>> 0; h = Math.imul(h ^ ((b + 0x7f4a7c15) >>> 0), 0x85ebca6b) >>> 0; h ^= h >>> 13; h = Math.imul(h ^ Math.imul(c + 1, 0x27d4eb2f), 0xc2b2ae35) >>> 0; h = (h ^ (h >>> 16)) >>> 0; return (h + 0.5) / 4294967296; };
      let R = [];
      function scenario() {
        const r = V.rng(1000 + S.seed * 97);
        R = [];
        const pos0 = 412;
        for (let j = 0; j < 7; j++) {
          const ids = []; while (ids.length < 5) { const v = r.int(WORDS.length); if (!ids.includes(v)) ids.push(v); }
          const lead = 7.5 + r() * 1.5 - j * 1.0;
          const lg = ids.map((_, k) => (k === 0 ? lead : 0) + r.normal() * 0.6);
          const score = lg.map((x) => x + r.normal() * (0.9 + 0.12 * j));
          const gg = ids.map((v) => -Math.log(-Math.log(mix(S.seed, pos0 + j, v))));
          const p = softmax(lg);
          const noisy = lg.map((x, k) => x + gg[k]);
          const x = noisy.indexOf(Math.max(...noisy));
          const dv = score.map((s, k) => s + 0.7 * gg[k]);
          const pick = dv.indexOf(Math.max(...dv));
          const cN = softmax(score.map((s, k) => (s + gg[k]) / 2.0))[pick];
          const cP = softmax(score)[pick];
          const rot = (j * 2 + S.seed) % 5, RT = (arr) => arr.map((_, i) => arr[(i - rot + 5) % 5]);
          R.push({ ids: RT(ids), lg: RT(lg), score: RT(score), gg: RT(gg), p: RT(p), noisy: RT(noisy), x: (x + rot) % 5, pick: (pick + rot) % 5, cN, cP, pos: pos0 + j });
        }
      }
      scenario();
      function depth(pol) { let prod = 1, k = 0; for (let j = 0; j < 7; j++) { prod *= pol === 'fnc' ? R[j].cN : R[j].cP; if (j === 0 || prod >= S.P) k = j + 1; else break; } return k; }
      function outcome(k) { let a = 0; for (let j = 0; j < k; j++) { if (R[j].pick === R[j].x) a++; else break; } return { a, emit: a + 1 }; }
      // copy scenario
      const PROMPT = ['Quote', ':', '“', 'the', 'draft', 'only', 'proposes', ';', 'the', 'prompt', 'key', 'alone', 'decides', 'the', 'token', ',', 'always', '”'];
      const REPLY = ['Ok', ',', 'quoting', ':', '“', 'the', 'draft', 'only', 'proposes', ';', 'the'];
      const COPIES = PROMPT.slice(9), TGT = ['prompt', 'key', 'alone', 'decides', 'the', 'token', ',', '”'];
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Draft', options: [{ v: 'dflash', label: 'DFlash2' }, { v: 'copy', label: 'copy (prompt lookup)' }], value: 'dflash', onChange: (v) => { S.src = v; gD.show(v === 'dflash'); setH7(); S.t0 = H.t; upd(); } });
      const gD = UI.group(ctl);
      UI.seg(gD.el, { label: 'Policy', options: [{ v: 'fnc', label: 'fnc · noise-aware' }, { v: 'fc', label: 'fc · no noise' }], value: 'fnc', onChange: (v) => { S.pol = v; S.t0 = H.t; upd(); } });
      nocase(UI.slider(gD.el, { label: 'threshold P', min: 0.1, max: 0.9, step: 0.05, value: 0.3, fmt: (v) => nf(v, 2), onInput: (v) => { S.P = v; upd(); } }));
      UI.button(ctl, '↺ resend (same key)', () => { S.replays++; S.t0 = H.t; upd(); });
      UI.button(ctl, '🎲 another seed', () => { S.seed++; S.replays = 0; scenario(); S.t0 = H.t; upd(); });
      const rK = UI.readout(ctl, 'rows in window', '—'), rA = UI.readout(ctl, 'accepted', '—'), rE = UI.readout(ctl, 'tokens emitted', '—'), rC = UI.readout(ctl, 'window cost (old table)', '—');
      let lastSig = '';
      function upd() {
        if (S.src === 'dflash') {
          const k = depth(S.pol), o = outcome(k);
          rK.set(String(k + 1)); rA.set(o.a + ' of ' + k, o.a === k ? 'ok' : null); rE.set(String(o.emit));
          const ms = VERIFY_MS[Math.min(7, k)] + BLOCK_MS;
          rC.set(nf(ms, 1) + ' ms for ' + (k + 1) + (k ? ' rows' : ' row'));
          lastSig = 'seed ' + S.seed + ' · ' + R.map((q) => WORDS[q.ids[q.x]]).slice(0, k + 1).join(' ');
        } else {
          rK.set('1 + ' + COPIES.length); rA.set('7 of ' + COPIES.length); rE.set('8'); rC.set('10 rows → 16-row graph (COPY_CODE)');
        }
      }
      upd();
      const H = V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const wide = w >= 640;
        const tau = t - S.t0;
        if (S.src === 'copy') { drawCopy(g, w, h, tau, C, wide); return; }
        const k = depth(S.pol), kOther = depth(S.pol === 'fnc' ? 'fc' : 'fnc'), o = outcome(k);
        const rowsW = wide ? w * 0.62 : w - 16, rx = 8;
        const top = wide ? 38 : 42, rh = wide ? (h - top - 34) / 7 : Math.min(54, (h * 0.62 - top) / 7);
        V.text(g, 'position', rx, 16, { size: 10, color: C.muted });
        V.wrap(g, 'target p (bars) · logit + keyed noise (amber tick) · ★ keyed sample · ▢ draft pick', rx + 46, 16, rowsW - 50, { size: wide ? 10.5 : 9.5, color: C.muted, lineH: 12, maxLines: 2 });
        const revealT = (j) => 0.25 + j * 0.32, verT = 0.25 + 7 * 0.32 + 0.4;
        R.forEach((q, j) => {
          const y = top + j * rh;
          const shownRow = tau > revealT(j);
          const inWin = j < k;
          const dim = !inWin ? 0.35 : 1;
          V.text(g, 'p+' + (j + 1), rx, y + rh / 2 + 4, { size: 10.5, font: 'mono', color: inWin ? C.ink : C.muted });
          const bx0 = rx + 40, bw = (rowsW - 48 - (wide ? 70 : 52)) / 5;
          const mxN = Math.max(...q.noisy), mnN = Math.min(...q.noisy);
          q.ids.forEach((v, c) => {
            const x = bx0 + c * bw, bh = (rh - 20) * q.p[c];
            g.globalAlpha = dim * (shownRow ? 1 : 0.25);
            g.fillStyle = V.alpha(C.a, 0.55); V.rr(g, x + 3, y + rh - 16 - bh, bw - 6, Math.max(1, bh), 3); g.fill();
            const nh = ((q.noisy[c] - mnN) / (mxN - mnN || 1)) * (rh - 20);
            V.line(g, x + bw * 0.25, y + rh - 16 - nh, x + bw * 0.75, y + rh - 16 - nh, { color: TF(), lw: 2.4 });
            if (c === q.pick && shownRow) { g.strokeStyle = C.b; g.lineWidth = 2; V.rr(g, x + 1, y + 1, bw - 2, rh - 4, 5); g.stroke(); }
            V.text(g, WORDS[v], x + bw / 2, y + rh - 3, { size: wide ? 10 : 8.5, font: 'mono', color: c === q.pick ? C.b : C.muted, align: 'center', maxW: bw - 2 });
            if (tau > verT && inWin && c === q.x) V.text(g, '★', x + bw / 2, y + rh - 18 - nh, { size: 13, color: TF(), align: 'center' });
            g.globalAlpha = 1;
          });
          // row result
          const mx = bx0 + 5 * bw + 10;
          if (tau > verT + 0.25 + j * 0.15) {
            if (!inWin) V.text(g, 'out', mx, y + rh / 2 + 4, { size: 10, color: C.muted });
            else if (j < o.a) mark(g, mx + 8, y + rh / 2, true, 6);
            else if (j === o.a) { mark(g, mx + 8, y + rh / 2, false, 6); V.text(g, '→ ★', mx + 20, y + rh / 2 + 4, { size: 10.5, color: TF(), weight: 600 }); }
            else V.text(g, '—', mx + 4, y + rh / 2 + 4, { size: 11, color: C.muted });
          }
          if (j === k - 1 && tau > revealT(j)) { V.line(g, rx, y + rh + 1, rx + rowsW - 10, y + rh + 1, { color: C.b, lw: 2, dash: [6, 4] }); V.text(g, 'the policy stops here (k = ' + k + ')', rx + rowsW - 12, y + rh + 12, { size: 10, color: C.b, align: 'right' }); }
        });
        if (tau > verT) {
          const flash = Math.max(0, 1 - (tau - verT) / 0.6);
          if (flash > 0) { g.fillStyle = V.alpha(TF(), 0.12 * flash); g.fillRect(rx, top, rowsW, k * rh); }
        }
        // product curve
        const cx = wide ? rx + rowsW + 18 : 16, cy = wide ? top + 10 : top + 7 * rh + 30, cw = wide ? w - cx - 12 : w - 30, ch = wide ? h * 0.48 : Math.max(80, h - cy - 140);
        V.text(g, 'Π of confidences', cx, cy - 4, { size: 11, color: C.muted });
        V.line(g, cx, cy + ch, cx + cw, cy + ch, { color: C.muted, lw: 1 }); V.line(g, cx, cy + 4, cx, cy + ch, { color: C.muted, lw: 1 });
        const ys = (v) => cy + ch - v * (ch - 6), xs = (j) => cx + 8 + (j / 6) * (cw - 16);
        V.line(g, cx, ys(S.P), cx + cw, ys(S.P), { color: RED(), lw: 1.4, dash: [5, 4] });
        V.text(g, 'P = ' + nf(S.P, 2), cx + cw, ys(S.P) - 4, { size: 10.5, color: RED(), align: 'right', font: 'mono' });
        ['fnc', 'fc'].forEach((pol) => {
          const main = pol === S.pol;
          let prod = 1; g.beginPath();
          for (let j = 0; j < 7; j++) { prod *= pol === 'fnc' ? R[j].cN : R[j].cP; const x = xs(j), y = ys(prod); if (j === 0) g.moveTo(x, y); else g.lineTo(x, y); }
          g.strokeStyle = main ? (pol === 'fnc' ? TF() : C.b) : V.alpha(C.muted, 0.6); g.lineWidth = main ? 2.6 : 1.4; g.stroke();
          prod = 1;
          for (let j = 0; j < 7; j++) { prod *= pol === 'fnc' ? R[j].cN : R[j].cP; if (main && tau > revealT(j)) V.dot(g, xs(j), ys(prod), j < (pol === S.pol ? k : kOther) ? 4 : 2.5, pol === 'fnc' ? TF() : C.b); }
        });
        [1, 4, 7].forEach((d) => V.text(g, String(d), xs(d - 1), cy + ch + 13, { size: 10, font: 'mono', color: C.muted, align: 'center' }));
        V.text(g, 'depth', cx + cw, cy + ch + 26, { size: 10, color: C.muted, align: 'right' });
        V.text(g, (S.pol === 'fnc' ? 'fnc' : 'fc') + ': k = ' + k + ' · other policy: k = ' + kOther, cx, cy + ch + 42, { size: 11, font: 'mono', color: C.ink, maxW: cw });
        // conclusion
        if (tau > verT + 0.9) {
          const msg = o.emit + ' tokens this round (' + o.a + ' accepted + 1 from the target). The same ★ come out under any policy: only where the chain stops changes.';
          V.wrap(g, msg, wide ? cx : 12, cy + ch + 64, wide ? cw : w - 24, { size: wide ? 11.5 : 10.5, color: C.ink, lineH: 15, maxLines: 4 });
          if (S.replays > 0) V.text(g, '↺ same key → same round: ' + fnv(lastSig), wide ? cx : 12, h - 8, { size: 10.5, font: 'mono', color: C.a, maxW: wide ? cw : w - 24 });
        }
      });
      function drawCopy(g, w, h, tau, C, wide) {
        const pad = 10, per = wide ? 18 : 9, gap = 3;
        const cw = (w - 2 * pad - (per - 1) * gap) / per, ch = 24;
        const lay = (arr, y0, hi, col, extra) => {
          arr.forEach((s, i) => {
            const r = Math.floor(i / per), c = i % per, x = pad + c * (cw + gap), y = y0 + r * (ch + 6);
            const on = hi(i);
            chip(g, x, y, cw, ch, s, { fill: on ? V.alpha(col, 0.25) : V.alpha(C.muted, 0.08), stroke: on ? col : V.alpha(C.muted, 0.3), size: wide ? 11 : 10, min: 6 });
            if (extra) extra(i, x, y);
          });
          return y0 + Math.ceil(arr.length / per) * (ch + 6);
        };
        V.text(g, 'prompt', pad, 16, { size: 11, color: C.muted, weight: 600 });
        const pEnd = lay(PROMPT, 22, (i) => i >= 1 && i <= 8, TF());
        const ry = pEnd + 22;
        V.text(g, 'reply so far (the last one is the pending token)', pad, ry - 6, { size: 11, color: C.muted, weight: 600, maxW: w - 20 });
        const rEnd = lay(REPLY, ry, (i) => i >= REPLY.length - 8, C.b);
        V.text(g, 'the last 8 (MATCH = 8) already occurred in the prompt (in amber): copy what came after', pad, rEnd + 8, { size: wide ? 11 : 10, color: C.b, maxW: w - 20 });
        if (tau > 0.6) {
          const u = Math.min(1, (tau - 0.6) / 0.8);
          const i9 = 9, x1 = pad + (i9 % per) * (cw + gap) + cw / 2, y1 = 22 + Math.floor(i9 / per) * (ch + 6) + ch;
          const x0 = pad + ((REPLY.length - 1) % per) * (cw + gap) + cw / 2, y0 = ry + Math.floor((REPLY.length - 1) / per) * (ch + 6);
          V.arrow(g, x0, y0, V.lerp(x0, x1, u), V.lerp(y0, y1, u), { color: TF(), lw: 2, curve: wide ? -24 : 18 });
        }
        const wy = rEnd + 36;
        if (tau > 1.6) {
          V.text(g, 'window: pending + ' + COPIES.length + ' copies (up to COPY_MAX = 15) · verification with the key', pad, wy, { size: 11, color: C.muted, weight: 600, maxW: w - 20 });
          const all = [REPLY[REPLY.length - 1]].concat(COPIES);
          lay(all, wy + 8, () => true, TF(), (i, x, y) => {
            if (tau > 1.6 + all.length * 0.08 + 0.6 && i > 0) {
              const ok = all[i] === TGT[i - 1];
              const first = !ok && all.slice(1, i).every((s2, q) => s2 === TGT[q]);
              if (ok && all.slice(1, i + 1).every((s2, q) => s2 === TGT[q])) mark(g, x + cw / 2, y + ch + 2, true, 4);
              else if (first) { mark(g, x + cw / 2, y + ch + 2, false, 4); V.text(g, '★ target: ' + TGT[i - 1], x + cw / 2, y + ch + 18, { size: 10, color: TF(), align: 'center', weight: 600 }); }
            }
          });
        }
        // cost: serial rounds vs one round with copies
        const cy0 = wy + 8 + Math.ceil(10 / per) * (ch + 6) + 34;
        const costOK = cy0 + 70 < h - 70;
        if (tau > 3.0 && costOK) {
          const fw = Math.min(46, (w - 2 * pad - 150) / 8 - 4);
          V.text(g, 'without copy: 8 tokens = 8 rounds', pad, cy0, { size: 11, color: C.muted, weight: 600 });
          for (let k = 0; k < 8; k++) { const on = ((tau - 3) * 4) % 9 > k; chip(g, pad + k * (fw + 4), cy0 + 8, fw, 22, wide ? 'fwd' : '', { fill: V.alpha(C.b, on ? 0.35 : 0.08), stroke: V.alpha(C.b, 0.6) }); }
          V.text(g, 'with copy: 8 tokens = 1 round (10-row window)', pad, cy0 + 50, { size: 11, color: C.muted, weight: 600 });
          chip(g, pad, cy0 + 58, fw * 1.6, 22, wide ? 'fwd ×10' : '', { fill: V.alpha(TF(), 0.4), stroke: TF() });
          V.text(g, 'by the old table, verifying 8 rows costs ~2.3× one row, not 8×', pad + fw * 1.6 + 10, cy0 + 73, { size: 10.5, color: C.ink, maxW: w - pad * 2 - fw * 1.6 - 12 });
        }
        if (tau > 3.6) {
          const msg = '7 copies accepted + 1 target token (★) = 8 tokens in one round. The copies cost almost nothing (numpy), come before the DFlash2 drafts, and the 10-row window is padded up to the 16-row graph. In editing, COPY_MAX = 15 took it from 84 to 119 tok/s.';
          V.wrap(g, msg, pad, costOK ? cy0 + 104 : h - (wide ? 40 : 66), w - 2 * pad, { size: wide ? 12 : 10.5, color: C.ink, lineH: 15, maxLines: wide ? 4 : 6 });
        }
      }
    })();

    /* ================================================ FIG 8 — four streams, one window, one pool */
    (function () {
      const fig = root.querySelector('#fig-streams');
      const stage = fig.querySelector('.la-fig-stage');
      const MEAS = { 1: [60.4, 60.4, 170, 114.7], 2: [79.2, 40.4, 269, 147.6], 3: [89.5, 30.6, 330, 196.3], 4: [108.8, 27.9, 340, 227.9] };
      const S = { mode: 'lote', n: 4, mp: true, k: 4 };
      const setH = () => { const nar = fig.clientWidth < 640; const want = (S.mode === 'lote' ? (nar ? 450 : 400) : (nar ? 420 : 380)) + 'px'; if (stage.style.height !== want) stage.style.height = want; };
      let txtY = null;
      setH();
      const ro = new ResizeObserver(setH); ro.observe(fig); if (LA.onCleanup) LA.onCleanup(() => ro.disconnect());
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Scene', options: [{ v: 'lote', label: 'batched round' }, { v: 'chegada', label: '4 prompts arriving' }], value: 'lote', onChange: (v) => { S.mode = v; gL.show(v === 'lote'); gC.show(v === 'chegada'); setH(); upd(); } });
      const gL = UI.group(ctl), gC = UI.group(ctl);
      UI.slider(gL.el, { label: 'requests', min: 1, max: 4, step: 1, value: 4, fmt: (v) => String(v), onInput: (v) => { S.n = v; upd(); } });
      UI.toggle(gC.el, { label: 'MULTI_PREFILL=1', value: true, onChange: (v) => { S.mp = v; upd(); } });
      UI.slider(gC.el, { label: 'prompts together', min: 1, max: 4, step: 1, value: 4, fmt: (v) => String(v), onInput: (v) => { S.k = v; upd(); } });
      gC.show(false);
      const r1 = UI.readout(ctl, 'prose aggregate', '—'), r2 = UI.readout(ctl, 'prose per req.', '—'), r3 = UI.readout(ctl, 'TTFT', '—'), r4 = UI.readout(ctl, 'structured aggregate', '—');
      function upd() {
        if (S.mode === 'lote') { const m = MEAS[S.n]; r1.set(nf(m[0], 1) + ' tok/s'); r2.set(nf(m[1], 1) + ' tok/s'); r3.set(m[2] + ' ms'); r4.set(nf(m[3], 1) + ' tok/s'); }
        else { r1.set(S.mp ? '108.8 tok/s' : '103.4 tok/s'); r2.set('—'); r3.set(S.mp ? '340 ms' : '590 ms', S.mp ? 'ok' : 'warn'); r4.set(S.mp ? '227.9 (×4)' : '196.3 (×4)'); }
      }
      upd();
      function rounds(seed, n) {
        const r = V.rng(seed);
        const w = []; for (let i = 0; i < n; i++) w.push(3 + r.int(14));
        const orig = w.slice(); const trimmed = new Array(n).fill(0);
        let sum = w.reduce((a, b) => a + b, 0);
        while (sum > 32) { let m = 0; for (let i = 1; i < n; i++) if (w[i] >= w[m]) m = i; w[m]--; trimmed[m]++; sum--; }
        const acc = w.map((x) => Math.min(x - 1, r.int(x)));
        return { w, orig, trimmed, acc, sum, osum: orig.reduce((a, b) => a + b, 0) };
      }
      V.fig(fig, (g, w, h, t, dt) => {
        const C = V.C();
        const wide = w >= 640;
        const SC = [C.a, C.b, C.c, hsl(205, 1)];
        const px = 10, pw = w - 20;
        if (S.mode === 'lote') {
          const n = S.n;
          const py = 22, ph = 26;
          V.text(g, 'shared pool (~2.92M tokens; extents aligned to 2,048) · schematic', px, 13, { size: wide ? 11 : 9.5, color: C.muted, maxW: pw });
          V.box(g, px, py, pw, ph, { r: 6, fill: V.alpha(C.muted, 0.08), stroke: V.alpha(C.muted, 0.35) });
          for (let k = 1; k < 48; k++) V.line(g, px + (k / 48) * pw, py + ph - 5, px + (k / 48) * pw, py + ph, { color: V.alpha(C.muted, 0.35), lw: 1 });
          const cyc = (t % 10) / 10;
          const base = [0.26, 0.2, 0.13, 0.09];
          const kept = [0.07, 0.05, 0.06];
          const evict = cyc > 0.55;
          let x = px + 1;
          for (let i = 0; i < n; i++) {
            const ww = (base[i] + 0.04 * Math.sin(t * 0.6 + i)) * (pw - 2);
            g.fillStyle = V.alpha(SC[i], 0.55); V.rr(g, x, py + 1, ww, ph - 2, 4); g.fill();
            if (ww > 44) V.text(g, 'stream ' + (i + 1), x + 6, py + 17, { size: 10, color: C.ink, maxW: ww - 8 });
            x += ww + 2;
            if (i === 0) {
              kept.forEach((f, q) => {
                let a = 1; if (q === 0 && evict) { a = Math.max(0, 1 - (cyc - 0.55) / 0.12); if (a <= 0) return; }
                const kw = f * (pw - 2);
                g.globalAlpha = a; g.fillStyle = V.alpha(C.muted, 0.35); V.rr(g, x, py + 1, kw, ph - 2, 4); g.fill();
                if (kw > 46) V.text(g, 'kept', x + kw / 2, py + 17, { size: 9.5, color: C.ink, align: 'center' });
                g.globalAlpha = 1; x += kw + 2;
              });
            }
          }
          if (evict && cyc < 0.8) V.text(g, 'LRU: a kept prompt gives up its space and the extents compact', w - 12, py + ph + 13, { size: 10, color: C.d, align: 'right', maxW: pw });
          const iy = py + ph + 30;
          V.text(g, n > 1 ? 'iterations: prompt chunk (≤ 1,024 rows) or round, alternating (FILL_SHARE = 0.5)' : 'iterations: decode rounds only', px, iy, { size: wide ? 11 : 9.5, color: C.muted, maxW: pw });
          const ni = wide ? 12 : 6, iw = (pw - (ni - 1) * 4) / ni;
          const shift = (t * 0.8) % 1;
          g.save(); g.beginPath(); g.rect(px, iy + 4, pw, 28); g.clip();
          for (let k = -1; k <= ni; k++) {
            const idx = k + Math.floor(t * 0.8);
            const isChunk = n > 1 && idx % 2 === 0 && (Math.floor(idx / 6) % 2 === 0);
            const xx = px + (k - shift) * (iw + 4);
            chip(g, xx, iy + 8, iw, 20, isChunk ? 'chunk' : 'round', { fill: V.alpha(isChunk ? C.d : C.a, 0.16), stroke: isChunk ? C.d : C.a, size: 10 });
          }
          g.restore();
          const R = rounds(Math.floor(t / 2.6) + 3, n), ph2 = (t % 2.6) / 2.6;
          const wy = iy + 50;
          const phaseA = ph2 < 0.35;
          const head = phaseA && R.osum > 32 ? 'proposals: ' + R.osum + ' rows > 32 → trim one draft at a time from the longest tail' : 'verification window: ' + R.sum + ' rows × 16 KiB = ' + R.sum * 16 + ' KiB per rank → one-shot RoCE';
          V.wrap(g, head, px, wy, pw, { size: wide ? 11.5 : 10, color: phaseA && R.osum > 32 ? RED() : C.ink, weight: 600, lineH: 13, maxLines: 2 });
          const perRow = 16, cs = (pw - (perRow - 1) * 3) / perRow, chh = wide ? 24 : 20;
          const cellsL = [];
          for (let i = 0; i < n; i++) { const tot = phaseA ? R.orig[i] : R.w[i]; for (let q = 0; q < tot; q++) cellsL.push([i, q, q >= R.w[i]]); }
          const gy0 = wy + (wide ? 12 : 22);
          const rowsRes = Math.ceil(R.osum / perRow);
          cellsL.forEach(([i, q, cut], k) => {
            const xx = px + (k % perRow) * (cs + 3), y = gy0 + Math.floor(k / perRow) * (chh + 5);
            if (cut) { g.fillStyle = V.alpha(RED(), 0.22); V.rr(g, xx, y, cs, chh, 3); g.fill(); g.strokeStyle = RED(); g.lineWidth = 1.4; V.rr(g, xx, y, cs, chh, 3); g.stroke(); return; }
            const acc = q === 0 || q <= R.acc[i];
            const verified = ph2 > 0.55;
            g.fillStyle = V.alpha(SC[i], verified ? (acc ? 0.85 : 0.14) : 0.45); V.rr(g, xx, y, cs, chh, 3); g.fill();
            if (q === 0) { g.strokeStyle = C.ink; g.lineWidth = 1.2; V.rr(g, xx, y, cs, chh, 3); g.stroke(); }
            if (cs >= 18) V.text(g, q === 0 ? 's' + (i + 1) : 'd' + q, xx + cs / 2, y + chh / 2 + 4, { size: wide ? 10 : 9, font: 'mono', color: verified && acc && !C.dark ? '#fff' : C.ink, align: 'center', alpha: verified && !acc ? 0.5 : 1 });
          });
          const winH = rowsRes * (chh + 5);
          if (ph2 > 0.55) { const fl = Math.max(0, 1 - (ph2 - 0.55) / 0.2); g.fillStyle = V.alpha(TF(), 0.15 * fl); g.fillRect(px, gy0 - 3, pw, winH + 2); }
          // the text follows the rows the window takes this round (slides smoothly between rounds)
          const target = gy0 + rowsRes * (chh + 5) + 10;
          txtY = txtY == null ? target : txtY + (target - txtY) * Math.min(1, (dt || 0.016) * 6);
          let yy = txtY;
          const lines = ['One forward for all: row-independent parts over the whole batch; segmented KDA starting from each stream\'s slot; DSA within each one\'s extent; one packed all-gather for the samplers.', 'Faded = rejected drafts; the KDA state is rebuilt from the kept rows only (parities + replay). s = pending token, d = draft.'];
          lines.forEach((s2) => { yy += V.wrap(g, s2, px, yy, pw, { size: wide ? 11.5 : 10, color: C.muted, lineH: wide ? 15 : 13, maxLines: 5 }) + 4; });
          const m = MEAS[n];
          V.wrap(g, 'measured (sparkDash): ' + nf(m[0], 1) + ' tok/s aggregate · ' + nf(m[1], 1) + ' per request · TTFT ' + m[2] + ' ms', px, Math.min(h - (wide ? 8 : 20), yy + 10), pw, { size: wide ? 12 : 10.5, color: C.ink, weight: 600, lineH: 13, maxLines: 2 });
        } else {
          const k = S.k;
          const FW = 0.12, RW = 0.05;
          const lanes = [{ nm: 'without MULTI_PREFILL', on: !S.mp }, { nm: 'with MULTI_PREFILL', on: S.mp }];
          const scheds = [];
          {
            const ev = []; let tt = 0; const ttft = [];
            for (let i = 0; i < k; i++) { ev.push({ a: tt, b: tt + FW, t: 'P' + (i + 1) + '·a', c: i }); tt += FW; if (i > 0) { ev.push({ a: tt, b: tt + RW, t: 'r', c: -1 }); tt += RW; } ev.push({ a: tt, b: tt + FW, t: 'P' + (i + 1) + '·b', c: i }); tt += FW; ttft.push(tt); if (i < k - 1) { ev.push({ a: tt, b: tt + RW, t: 'r', c: -1 }); tt += RW; } }
            scheds.push({ ev, ttft, end: tt });
          }
          {
            const ev = []; const ttft = [];
            const lab = k > 1 ? '[s1..s' + k + ']' : '[s1]';
            ev.push({ a: 0, b: FW * 1.05, t: lab + '·a', c: 9 }); ev.push({ a: FW * 1.05, b: FW * 2.1, t: lab + '·b', c: 9 });
            for (let i = 0; i < k; i++) ttft.push(FW * 2.1);
            scheds.push({ ev, ttft, end: FW * 2.1 });
          }
          const tmax = Math.max(scheds[0].end, 0.3) * 1.08;
          const per = 4.5, phx = ((t % per) / per) * tmax * 1.15;
          const xs = (s2) => px + (s2 / tmax) * pw;
          V.wrap(g, k + (k > 1 ? ' short prompts arrive together' : ' short prompt arrives') + '; each = 2 chunks (up to the grid point, then the tail) · short forward ≈ 0.12 s; r = decode round', px, 14, pw, { size: wide ? 11 : 9.5, color: C.muted, lineH: 13, maxLines: 2 });
          const SCc = [C.a, C.b, C.c, hsl(205, 1)];
          lanes.forEach((L, li) => {
            const sc = scheds[li];
            const y = (wide ? 52 : 60) + li * (wide ? 130 : 160);
            V.text(g, L.nm, px, y, { size: 12, weight: 700, color: L.on ? TF() : C.muted });
            const by = y + 10, bh = 34;
            sc.ev.forEach((e) => {
              const x0 = xs(e.a), x1 = xs(e.b);
              const done = phx >= e.b, act = phx >= e.a && phx < e.b;
              const col = e.c < 0 ? C.muted : e.c === 9 ? TF() : SCc[e.c];
              g.globalAlpha = L.on ? 1 : 0.5;
              V.box(g, x0 + 1, by, Math.max(2, x1 - x0 - 2), bh, { r: 5, fill: V.alpha(col, done ? 0.55 : act ? 0.35 : 0.12), stroke: V.alpha(col, 0.8), lw: act ? 2 : 1 });
              const lab = x1 - x0 > 40 ? e.t : e.c < 0 ? 'r' : e.t.replace('·', '');
              if (x1 - x0 > 12) V.text(g, lab, (x0 + x1) / 2, by + bh / 2 + 4, { size: 9, font: 'mono', color: C.ink, align: 'center', maxW: x1 - x0 - 2 });
              g.globalAlpha = 1;
            });
            sc.ttft.forEach((tf, i) => {
              const x = xs(tf);
              const yy = by + bh + 6 + (li === 0 ? (i % 2) * 14 : 0);
              if (phx >= tf) { V.text(g, '⚑', x, yy + 10, { size: 12, color: li === 1 ? TF() : SCc[i], align: 'center' }); if (li === 0) V.text(g, 's' + (i + 1), x + 8, yy + 10, { size: 9.5, color: C.muted }); }
            });
            const last = sc.ttft[sc.ttft.length - 1];
            V.text(g, 'last prompt\'s 1st token ≈ ' + nf(last, 2) + ' s (schematic)', px, by + bh + (li === 0 ? 62 : 34), { size: 10.5, font: 'mono', color: L.on ? C.ink : C.muted });
          });
          const lane2Bot = (wide ? 52 : 60) + (wide ? 130 : 160) + 10 + 34 + 40;
          V.line(g, xs(phx), 40, xs(phx), lane2Bot, { color: V.alpha(C.ink, 0.5), lw: 1.2, dash: [3, 3] });
          const ratio = scheds[0].ttft[k - 1] / scheds[1].ttft[k - 1];
          let ny = lane2Bot + 18;
          const sch = k > 1 ? 'Schematic: it counts only the short forwards (~0.12 s) of ~130-token prompts (2 chunks each), which is why the last ⚑ arrives ~' + nf(ratio, 1) + '× sooner.' : 'Schematic: with a single prompt there is nothing to batch, and the two timelines tie.';
          ny += V.wrap(g, sch, px, ny, pw, { size: wide ? 11.5 : 10, color: C.muted, lineH: 14, maxLines: 3 }) + 6;
          V.wrap(g, 'Measured (sparkDash, prose ×4, mean TTFT of the 4 with real rounds and prompts): 590 → 340 ms, ~1.7×; aggregate 103.4 → 108.8 tok/s. Each piece gets the bits of a chunk of its own.', px, ny, pw, { size: wide ? 11.5 : 10, color: C.ink, weight: 600, lineH: 14, maxLines: 4 });
        }
      });
    })();

    /* ================================================ FIG 9 — numbers with labels */
    (function () {
      const fig = root.querySelector('#fig-numeros');
      adaptH(fig, 430, 600, 640);
      const S = { mode: 'decode', load: 'prose', cond: false, t0: 0 };
      const DEC = { prose: [[60.4, 60.4, 170], [79.2, 40.4, 269], [89.5, 30.6, 330], [108.8, 27.9, 340]], structured: [[114.7, 114.7, 149], [147.6, 77.1, 295], [196.3, 67.4, 317], [227.9, 61.1, 415]] };
      const PRE = [[8219, 1952.2, '4.21 s'], [16407, 1973.5, '8.31 s'], [32790, 1978.9, '16.57 s'], [65563, 1942.5, '33.75 s'], [131099, 1837.9, '71.33 s'], [262170, 1641.7, '159.69 s'], [981841, 1015, '967 s']];
      const VS = [
        { t: 'Prose, 1 req. (tok/s)', a: 36.1, b: 60.4, up: true, c: 'vLLM: sparkDash 2026-09-17 (adaptive-k, coop. MoE, 850k) · TF: sparkDash, clocks 2,200 MHz' },
        { t: 'Prose, 4 req. aggregate', a: 75.3, b: 108.8, up: true, c: 'same runs; per request 19.4 vs 27.9' },
        { t: 'TTFT, 1 req. (ms)', a: 333, b: 170, up: false, c: 'same runs (lower is better)' },
        { t: 'Structured, 1 req.', a: 83.0, b: 114.7, up: true, c: 'vLLM: README 2026-09-28, lab tool · TF: sparkDash' },
        { t: 'Cold prefill at 8k (tok/s)', a: 1411, b: 1952.2, up: true, c: 'vLLM: README 09-28 (lab) · TF: sparkDash, 8,219 tokens' },
        { t: 'Context (tokens)', a: 850000, b: 1048576, up: true, c: 'defaults of both recipes' },
        { t: 'KV pool (tokens)', a: 1233779, b: 2922496, up: true, c: 'vLLM: fixed 11 GiB · TF: 12.5 GiB ceiling, measured start' },
        { t: 'KV per token and DSA layer (B)', a: 656, b: 528, up: false, c: 'format: fp8_ds_mla vs e4m3 + 2^k scale' },
        { t: 'Boot (s)', a: 122, b: 240, bRange: [120, 360], up: false, c: 'vLLM: restart with InstantTensor · TF: 2–6 min of loading (vLLM wins)' },
      ];
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Panel', options: [{ v: 'decode', label: 'Decode' }, { v: 'prefill', label: 'Prefill' }, { v: 'reuso', label: 'Reuse' }, { v: 'vllm', label: 'vs vLLM' }], value: 'decode', onChange: (v) => { S.mode = v; S.t0 = H.t; gD.show(v === 'decode'); gV.show(v === 'vllm'); } });
      const gD = UI.group(ctl), gV = UI.group(ctl);
      UI.seg(gD.el, { label: 'Workload', options: [{ v: 'prose', label: 'prose' }, { v: 'structured', label: 'structured' }], value: 'prose', onChange: (v) => { S.load = v; S.t0 = H.t; } });
      UI.toggle(gV.el, { label: 'conditions', value: false, onChange: (v) => { S.cond = v; } });
      gV.show(false);
      const H = V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const wide = w >= 640;
        const an = V.ease.out(Math.min(1, (t - S.t0) / 0.9));
        if (S.mode === 'decode') {
          const D = DEC[S.load];
          const L = 54, R = w - 14, T = 30, B = h - 56;
          const mx = S.load === 'prose' ? 120 : 240;
          const ys = (v) => B - (v / mx) * (B - T);
          V.text(g, 'tok/s · ' + S.load + ' · sparkDash, clocks 2,200 MHz, temp 0', L - 40, 16, { size: 11, color: C.muted, maxW: w - 20 });
          for (let v = 0; v <= mx; v += mx / 4) { V.line(g, L, ys(v), R, ys(v), { color: C.grid, lw: 1 }); V.text(g, nf(v), L - 6, ys(v) + 4, { size: 10, font: 'mono', color: C.muted, align: 'right' }); }
          const gw = (R - L) / 4;
          D.forEach((d, i) => {
            const x0 = L + i * gw + gw * 0.14, bw = gw * 0.34;
            const h1 = (B - ys(d[0])) * an, h2 = (B - ys(d[1])) * an;
            g.fillStyle = V.alpha(TF(), 0.8); V.rr(g, x0, B - h1, bw, h1, 4); g.fill();
            g.fillStyle = V.alpha(C.b, 0.65); V.rr(g, x0 + bw + 4, B - h2, bw, h2, 4); g.fill();
            V.text(g, nf(d[0], 1), x0 + bw / 2, B - h1 - 5, { size: wide ? 11 : 9.5, font: 'mono', color: C.ink, align: 'center' });
            V.text(g, nf(d[1], 1), x0 + bw * 1.5 + 4, B - h2 - 5, { size: wide ? 11 : 9.5, font: 'mono', color: C.ink, align: 'center' });
            V.text(g, (i + 1) + ' req.', x0 + bw + 2, B + 15, { size: 11, color: C.ink, align: 'center' });
            V.text(g, 'TTFT ' + d[2] + ' ms', x0 + bw + 2, B + 30, { size: wide ? 10.5 : 9, font: 'mono', color: C.muted, align: 'center' });
          });
          const lx = L, ly = h - 10;
          g.fillStyle = V.alpha(TF(), 0.8); g.fillRect(lx, ly - 9, 10, 10); V.text(g, 'aggregate', lx + 14, ly, { size: 11, color: C.ink });
          g.fillStyle = V.alpha(C.b, 0.65); g.fillRect(lx + 90, ly - 9, 10, 10); V.text(g, 'per request', lx + 104, ly, { size: 11, color: C.ink });
          V.text(g, '×4 / ×1 = ' + nf(D[3][0] / D[0][0], 2) + '×', R, ly, { size: 11, font: 'mono', color: C.ink, align: 'right' });
        } else if (S.mode === 'prefill') {
          const L = 58, R = w - 20, T = 30, B = h - 44;
          const lx = (n) => L + ((Math.log10(n) - Math.log10(6000)) / (Math.log10(1.3e6) - Math.log10(6000))) * (R - L);
          const ys = (v) => B - (v / 2200) * (B - T);
          V.text(g, 'prefill (tok/s) vs prompt size · log axis', L - 50, 16, { size: 11, color: C.muted, maxW: w - 20 });
          [0, 500, 1000, 1500, 2000].forEach((v) => { V.line(g, L, ys(v), R, ys(v), { color: C.grid, lw: 1 }); V.text(g, nf(v), L - 6, ys(v) + 4, { size: 10, font: 'mono', color: C.muted, align: 'right' }); });
          [[8192, '8k'], [32768, '32k'], [131072, '128k'], [524288, '512k'], [1048576, '1M']].forEach(([n, l]) => V.text(g, l, lx(n), B + 15, { size: 10, font: 'mono', color: C.muted, align: 'center' }));
          const npts = Math.max(1, Math.floor(an * PRE.length + 0.001));
          g.beginPath(); PRE.slice(0, npts).forEach((p, i) => { const x = lx(p[0]), y = ys(p[1]); if (i === 0) g.moveTo(x, y); else g.lineTo(x, y); }); g.strokeStyle = TF(); g.lineWidth = 2.6; g.stroke();
          PRE.slice(0, npts).forEach((p, i) => {
            const x = lx(p[0]), y = ys(p[1]);
            V.dot(g, x, y, 4.5, TF());
            const lab = nf(p[1], 0) + (wide || i % 2 === 0 || i === 6 ? ' · ' + p[2] : '');
            V.text(g, lab, x + (i === 6 ? -6 : 6), y + (i % 2 ? 16 : -8), { size: wide ? 10.5 : 9, font: 'mono', color: C.ink, align: i === 6 ? 'right' : 'left' });
          });
          if (npts === PRE.length) { const x = lx(981841), y = ys(1015); V.glow(g, x, y, 18, C.a, 0.6 + 0.4 * Math.sin(t * 3)); V.text(g, 'needle found at 981,841 tokens', x - 6, y + 30, { size: wide ? 11.5 : 10, color: C.a, align: 'right', weight: 600 }); }
          V.text(g, 'drops with context because attention grows with the prompt; up to 64k it stays at ~1,950', L, h - 8, { size: wide ? 11 : 9.5, color: C.muted, maxW: w - L - 10 });
        } else if (S.mode === 'reuso') {
          const L = wide ? 230 : 12, R = w - 70, T = 40;
          const lx = (s) => L + ((Math.log10(s) - Math.log10(0.03)) / (Math.log10(60) - Math.log10(0.03))) * (R - L);
          const rows = [['identical 64k prompt', 34, 0.07, '< 0.07 s'], ['new conversation, same 7.9k system', 4.24, 0.13, '0.13 s']];
          V.text(g, 'seconds until the reply starts · log axis', 12, 18, { size: 11, color: C.muted });
          [0.1, 1, 10].forEach((s) => { V.line(g, lx(s), T - 4, lx(s), wide ? T + 175 : T + 245, { color: C.grid, lw: 1 }); V.text(g, nf(s, s < 1 ? 1 : 0) + ' s', lx(s), T - 8, { size: 10, font: 'mono', color: C.muted, align: 'center' }); });
          rows.forEach((r, i) => {
            const y = T + 10 + i * (wide ? 90 : 130);
            const ly = wide ? y + 26 : y + 4;
            V.text(g, r[0], 12, ly, { size: 12, weight: 600, color: C.ink, maxW: wide ? L - 20 : w - 20 });
            const by = wide ? y : y + 14;
            const w1 = (lx(r[1]) - L) * an, w2 = (lx(Math.max(0.031, r[2])) - L) * an;
            g.fillStyle = V.alpha(C.muted, 0.5); V.rr(g, L, by + 4, Math.max(2, w1), 20, 4); g.fill();
            V.text(g, 'cold: ' + nf(r[1], 2) + ' s', L + w1 + 6, by + 19, { size: 10.5, font: 'mono', color: C.ink });
            g.fillStyle = V.alpha(TF(), 0.85); V.rr(g, L, by + 30, Math.max(2, w2), 20, 4); g.fill();
            V.text(g, 'reused: ' + r[3], L + w2 + 6, by + 45, { size: 10.5, font: 'mono', color: C.ink });
            const pulse = (t * 1.2 + i * 0.5) % 1;
            V.dot(g, L + w2 * pulse, by + 40, 3, C.a);
          });
          const ty = wide ? T + 200 : T + 272;
          V.wrap(g, 'Mechanism: the state at the end of the prompt is kept along with the head\'s logits row (replay), and the system prompt gets a state of its own (shared prefix). With 3 alternating conversations (~4.5k, 6 turns), 98.1% of the prompts came from the cache with TF_GLM_MULTI_LONE=0, versus 0% with 1.', 12, ty, w - 24, { size: wide ? 11.5 : 10.5, color: C.muted, lineH: 15, maxLines: 4 });
        } else {
          const n = VS.length;
          const top = 18, rowH = (h - top - 26) / n;
          const L = wide ? 210 : 12, R = w - (wide ? 70 : 82);
          VS.forEach((r, i) => {
            const y = top + i * rowH;
            const yl = wide ? y + rowH / 2 : y + 12;
            const yd = wide ? y + rowH / 2 : y + rowH * 0.62;
            V.text(g, r.t, wide ? 12 : 12, yl + (wide ? 4 : 0), { size: wide ? 11.5 : 10.5, color: C.ink, weight: 600, maxW: wide ? L - 20 : w - 24 });
            const mx = Math.max(r.a, r.bRange ? r.bRange[1] : r.b) * 1.08;
            const xs = (v) => L + (v / mx) * (R - L);
            V.line(g, L, yd, R, yd, { color: C.grid, lw: 1 });
            const xa = xs(r.a), xb = xs(V.lerp(r.a, r.b, an));
            const better = r.up ? r.b > r.a : r.b < r.a;
            V.line(g, xa, yd, xb, yd, { color: V.alpha(better ? TF() : RED(), 0.6), lw: 3 });
            if (r.bRange) { g.fillStyle = V.alpha(TF(), 0.25); g.fillRect(xs(r.bRange[0]), yd - 5, xs(r.bRange[1]) - xs(r.bRange[0]), 10); }
            V.dot(g, xa, yd, 5.5, GLM()); V.dot(g, xb, yd, 5.5, TF());
            const ratio = r.up ? r.b / r.a : r.a / r.b;
            const rlab = r.bRange ? '2–6 min' : (better ? '' : '') + nf(ratio, 2) + '×';
            V.text(g, rlab, R + 8, yd + 4, { size: 11, font: 'mono', weight: 600, color: better ? C.a : RED() });
            if (S.cond) V.text(g, r.c, wide ? L : 12, yd + (wide ? 15 : 14), { size: wide ? 9.5 : 8.5, color: C.muted, maxW: wide ? R - L + 60 : w - 24 });
          });
          const ly = h - 8;
          V.dot(g, 16, ly - 4, 5, GLM()); V.text(g, 'vLLM (ch. 42)', 26, ly, { size: 11, color: C.ink });
          V.dot(g, 140, ly - 4, 5, TF()); V.text(g, 'TensorFold', 150, ly, { size: 11, color: C.ink });
          V.text(g, 'not a controlled A/B', w - 12, ly, { size: 11, color: C.d, align: 'right', weight: 600 });
        }
      });
    })();

    /* ================================================ FIG 10 — build your .env */
    (function () {
      const fig = root.querySelector('#fig-ajuste');
      adaptH(fig, 420, 560, 640);
      const Q = [
        { k: 'scale', l: 'I need 4 Sparks or > 8 requests' },
        { k: 'api', l: 'I need logprobs, n > 1 or penalties' },
        { k: 'com', l: 'commercial use' },
        { k: 'fr', l: 'short non-English prompts' },
        { k: 'exact', l: 'exactness against the checkpoint' },
        { k: 'share', l: 'another workload shares the Sparks' },
        { k: 'agents', l: 'many alternating agents' },
        { k: 'text', l: 'text only' },
        { k: 'disk', l: 'little disk on the worker' },
      ];
      const S = {}; Q.forEach((q) => { S[q.k] = false; });
      let typedAt = 0, prevSig = '';
      const ctl = UI.controls(fig);
      Q.forEach((q) => UI.toggle(ctl, { label: q.l, value: false, onChange: (v) => { S[q.k] = v; upd(); } }));
      const rC = UI.readout(ctl, 'CONTEXT', '—'), rP = UI.readout(ctl, 'PARALLEL', '—'), rM = UI.readout(ctl, 'engine', '—');
      function derive() {
        const DRAFTER = S.com ? 'mtp' : 'dflash2';
        const DENSE = S.exact ? 'bf16' : S.fr ? 'fp8' : 'q4';
        const KV = S.exact ? 'bf16' : 'fp8';
        const VISION = S.text ? 0 : 1;
        let CONTEXT;
        if (KV !== 'bf16') CONTEXT = 1048576; else if (DRAFTER !== 'dflash2') CONTEXT = 524288; else if (VISION === 1 && DENSE !== 'q4') CONTEXT = 163840; else CONTEXT = 196608;
        const PARALLEL = DRAFTER === 'dflash2' ? 4 : 1;
        const lines = [];
        if (S.com) lines.push(['DRAFTER', 'mtp', 'DFlash2 is CC BY-NC-ND; mtp: PARALLEL=1, loses +5–10% of decode']);
        if (S.exact) { lines.push(['DENSE', 'bf16', 'serves the checkpoint as it is']); lines.push(['KV', 'bf16', 'exact cache; smaller window']); lines.push(['KDA_CHUNKED', '0', 'serial kernel on the prompt (serial bits)']); }
        else if (S.fr) lines.push(['DENSE', 'fp8', '#18: q4 misses the end of turn (12/48); fp8 costs ~10%']);
        if (S.share) lines.push(['MEMORY_RESERVE_GIB', '20', 'illustrative: raise the reserve; min. free ≈ reserve − ~10 GiB']);
        if (S.agents) { lines.push(['TF_GLM_CACHE_ENTRIES', '64', 'illustrative: ~91 MiB each, measured in 0097 (#17)']); lines.push(['TF_GLM_MULTI_LONE', '0', 'keeps the other conversations\' cache (98.1% vs 0%)']); }
        if (S.text) lines.push(['VISION', '0', 'returns ~1.8 GiB from rank 0 to the cache']);
        if (S.disk) lines.push(['WORKER_WEIGHTS', 'nfs', 'worker without the ~166 GiB copy; ~2.2 min']);
        return { DRAFTER, DENSE, KV, VISION, CONTEXT, PARALLEL, lines, vllm: S.scale || S.api };
      }
      let D = derive();
      function upd() { D = derive(); rC.set(nf(D.CONTEXT)); rP.set(String(D.PARALLEL)); rM.set(D.vllm ? 'vLLM recipe' : 'TensorFold', D.vllm ? 'warn' : 'ok'); const sig = JSON.stringify(D.lines); if (sig !== prevSig) { prevSig = sig; typedAt = H ? H.t : 0; } }
      let H = null;
      upd();
      H = V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const wide = w >= 640;
        const ew = wide ? w * 0.6 : w - 16, ex = 8, ey = 8, eh = wide ? h - 16 : h * 0.6;
        V.box(g, ex, ey, ew, eh, { r: 10, fill: V.alpha(C.muted, 0.08), stroke: V.alpha(C.muted, 0.45), lw: 1.2 });
        g.fillStyle = V.alpha(C.muted, 0.14); V.rr(g, ex, ey, ew, 24, 10); g.fill();
        [C.red || '#f55', C.d, C.a].forEach((c, i) => V.dot(g, ex + 14 + i * 14, ey + 12, 4.5, c));
        V.text(g, '.env (read, never executed)', ex + 60, ey + 16, { size: 11, font: 'mono', color: C.muted, maxW: ew - 70 });
        const fs = wide ? 12 : 10.5, lh = wide ? 32 : 29;
        let y = ey + 44;
        if (!D.lines.length) { V.text(g, '# nothing to change: the defaults serve agents,', ex + 12, y, { size: fs, font: 'mono', color: C.a, maxW: ew - 20 }); V.text(g, '# 2 Sparks, 4 requests, 1M window', ex + 12, y + 16, { size: fs, font: 'mono', color: C.a, maxW: ew - 20 }); y += 24; }
        const elapsed = t - typedAt;
        let charBudget = Math.floor(elapsed * 60);
        D.lines.forEach(([k, v, c], i) => {
          if (y > ey + eh - 10) return;
          const full = k + '=' + v;
          const nShow = Math.max(0, Math.min(full.length, charBudget)); charBudget -= full.length;
          const shown = full.slice(0, nShow);
          const kk = shown.slice(0, Math.min(shown.length, k.length)), rest = shown.slice(k.length);
          V.text(g, kk, ex + 12, y, { size: fs, font: 'mono', color: C.c, weight: 600 });
          const kw = tw(g, kk, fs, 'mono', 600);
          if (rest) V.text(g, rest, ex + 12 + kw, y, { size: fs, font: 'mono', color: C.a, weight: 600 });
          if (nShow < full.length && Math.floor(t * 3) % 2 === 0) { const cw2 = tw(g, shown, fs, 'mono', 600); g.fillStyle = C.ink; g.fillRect(ex + 13 + cw2, y - fs + 2, 2, fs); }
          if (nShow >= full.length) V.text(g, '# ' + c, ex + 12, y + (wide ? 15 : 13), { size: wide ? 10.5 : 9, font: 'mono', color: C.muted, maxW: ew - 20 });
          y += lh;
        });
        // the rest stays at the default
        const DEF = [['DRAFTER', 'dflash2'], ['DENSE', 'q4'], ['KV', 'fp8'], ['KDA_CHUNKED', '1'], ['VISION', '1'], ['WORKER_WEIGHTS', 'copy'], ['MEMORY_RESERVE_GIB', '14.5'], ['KV_POOL_GIB', '12.5'], ['TF_GLM_CACHE_ENTRIES', '32'], ['COMM', 'roce'], ['SPLIT', '1'], ['MULTI_PREFILL', '1'], ['SHARED_PREFIX', '1'], ['DRAFT_POLICY', 'fnc7:0.3'], ['COPY', '1'], ['COPY_MAX', '15'], ['COPY_CODE', '1'], ['MAX_TOKENS', '32768']];
        const over = new Set(D.lines.map((l) => l[0]));
        const rest = DEF.filter((d) => !over.has(d[0]));
        const dy0 = y + 8;
        if (dy0 + 30 < ey + eh) {
          V.text(g, '# the rest stays at the config.sh default:', ex + 12, dy0, { size: wide ? 10.5 : 9.5, font: 'mono', color: C.muted, maxW: ew - 20 });
          const cols = wide ? 2 : 2, cwid = (ew - 24) / cols, rl = wide ? 15 : 13;
          const maxRows = Math.max(0, Math.floor((ey + eh - dy0 - 14) / rl));
          rest.slice(0, maxRows * cols).forEach(([k, v], i) => {
            const c = Math.floor(i / maxRows), r = i % maxRows;
            V.text(g, k + '=' + v, ex + 12 + c * cwid, dy0 + 16 + r * rl, { size: wide ? 10.5 : 9, font: 'mono', color: V.alpha(C.muted, 1), alpha: 0.85, maxW: cwid - 6 });
          });
        }
        if (D.vllm) {
          g.fillStyle = V.alpha(C.bg || '#000', 0.7); V.rr(g, ex + 1, ey + 25, ew - 2, eh - 26, 9); g.fill();
          const pulse = 0.6 + 0.4 * Math.sin(t * 3);
          V.box(g, ex + 16, ey + eh / 2 - 46, ew - 32, 92, { r: 12, fill: V.alpha(GLM(), 0.14 + 0.06 * pulse), stroke: GLM(), lw: 2 });
          V.text(g, 'Use the vLLM recipe (ch. 42)', ex + ew / 2, ey + eh / 2 - 18, { size: wide ? 15 : 12.5, weight: 700, color: C.ink, align: 'center', maxW: ew - 40 });
          const why = (S.scale ? 'the TensorFold recipe serves 2 or 3 Sparks (4 only on a branch) and up to 8 requests' : '') + (S.scale && S.api ? '; ' : '') + (S.api ? 'no logprobs, n > 1 or penalties' : '');
          V.wrap(g, why, ex + 28, ey + eh / 2 + 6, ew - 56, { size: wide ? 11.5 : 10, color: C.ink, lineH: 14, maxLines: 3, align: 'left' });
        }
        // result panel
        const px = wide ? ex + ew + 16 : 10, py = wide ? 14 : ey + eh + 18, pw = wide ? w - px - 10 : w - 20;
        const rows = [
          ['DRAFTER', D.DRAFTER], ['DENSE', D.DENSE], ['KV', D.KV], ['VISION', String(D.VISION)], ['CONTEXT', nf(D.CONTEXT)], ['PARALLEL', String(D.PARALLEL)],
        ];
        const rlh = wide ? 22 : 18;
        V.text(g, 'result (config.sh logic)', px, py + 4, { size: 11, color: C.muted, weight: 600, maxW: pw });
        rows.forEach(([k, v], i) => {
          const yy = py + 24 + (wide ? i : Math.floor(i / 2)) * rlh;
          const xx = wide ? px : px + (i % 2) * (pw / 2);
          V.text(g, k, xx, yy, { size: wide ? 11.5 : 10.5, font: 'mono', color: C.c });
          V.text(g, v, xx + (wide ? 100 : 86), yy, { size: wide ? 11.5 : 10.5, font: 'mono', color: C.ink, weight: 600 });
        });
        const yb = py + 24 + (wide ? rows.length : 3) * rlh + 6;
        const exact = S.exact;
        mark(g, px + 7, yb, true, 5); V.text(g, 'drafted = serial, concurrent = solo', px + 18, yb + 4, { size: wide ? 11 : 10, color: C.ink, maxW: pw - 20 });
        mark(g, px + 7, yb + 20, exact, 5); V.text(g, exact ? 'equal to the BF16 checkpoint' : 'lossy vs. checkpoint: DENSE=' + D.DENSE + ' · KV=' + D.KV + ' · KDA_CHUNKED=1', px + 18, yb + 24, { size: wide ? 11 : 10, color: exact ? C.ink : C.d, maxW: pw - 20 });
      });
    })();
  },
});

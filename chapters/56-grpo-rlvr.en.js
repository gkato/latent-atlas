LA.chapter({
  id: 'grpo-rlvr',
  part: 11,
  order: 56,
  title: 'Reward model + custom RL: GRPO, RLOO and verifiable rewards',
  navTitle: 'GRPO & RLVR',
  subtitle: 'How to throw away PPO’s critic and still learn: statistical baselines (ReMax, RLOO, REINFORCE++, GRPO), the 2025–2026 fixes (Dr. GRPO, DAPO, CISPO, GSPO, SAPO), verifiers that cannot be fooled, the DeepSeek-R1 recipe and the mismatch between the training engine and the inference engine.',
  minutes: 50,
  recipes: ['glm', 'ds', 'qwen'],
  tags: ['GRPO', 'RLOO', 'ReMax', 'REINFORCE++', 'DAPO', 'Dr. GRPO', 'GSPO', 'CISPO', 'SAPO', 'RLVR', 'verifiers', 'PRM', 'DeepSeek-R1', 'entropy collapse', 'reward hacking', 'IcePop', 'TIS', 'Keep Routing', 'R3'],
  goals: [
    'Explain why a <b>statistical baseline</b> (greedy, leave-one-out, group, batch) replaces PPO’s critic without biasing the gradient, and compute by hand the advantages of ReMax, RLOO, GRPO, Dr. GRPO and REINFORCE++',
    'Write the <b>GRPO</b> objective with the KL term (k1, k2, k3) and explain what DAPO, Dr. GRPO, CISPO, GSPO and SAPO change in the normalization and in the trust region',
    'Design a <b>verifiable</b> reward (reference answer, F2P/P2P tests, process vs outcome) and recognize reward hacking, entropy collapse and length inflation before they ruin training',
    'Rebuild the <b>DeepSeek-R1</b> recipe and connect the training↔inference mismatch (TIS, IcePop, Keep Routing, R3, FP32 LM head) to the determinism and quantization parameters of the Module 1 recipes',
  ],
  prereqs: ['ppo-rlhf', 'rl-fundamentos'],
  summary: [
    'Without a critic, the baseline comes from the samples themselves: \\(\\nabla J=\\mathbb E[(R-b)\\nabla\\log\\pi]\\) stays unbiased if \\(b\\) does not depend on the answer. ReMax uses the greedy answer’s score; RLOO, the mean of the other \\(k-1\\) (\\(=\\tfrac{k}{k-1}(R_i-\\bar R)\\)); REINFORCE++ normalizes over the global batch. The critic would cost \\(16N\\) extra bytes and sees poorly in long CoT (\\(\\lambda^{T-t}\\)).',
    'GRPO: \\(\\hat A_i=(r_i-\\text{mean})/\\text{std}\\) within the group of \\(G\\) answers, the same on every token, with PPO’s clip and the k3 KL in the loss. With [1,0,0,1,1,0,0,0]: +1.291 / −0.775 (population std). Unanimous group ⇒ zero advantage; nearly tied scores ⇒ huge advantages.',
    'Fixes: Dr. GRPO removes the \\(1/|o_i|\\) and the ÷ std (length and difficulty biases); DAPO adds clip-higher (0.2/0.28), dynamic sampling, token-level loss and a soft punishment of long answers (30 → 50 on AIME 2024). CISPO clips the weight, GSPO uses the sequence ratio (ε of 3–4·10⁻⁴) and SAPO, a sigmoid gate.',
    'RLVR swaps the RM for a verifier (reference answer, F2P/P2P tests, format): robust, but hackable if it is lax. The PRM (product of the step probabilities) is good for reranking and fragile as an RL target; without a reference answer, GRMs with rubrics come in.',
    'DeepSeek-R1: R1-Zero (pure GRPO on the base model, 16 outputs, rule-based rewards; pass@1 on AIME 2024 from 15.6% to 77.9% in 10,400 steps in v2 of the paper, or 71.0% in v1) and R1 in four stages (cold start, RL with language consistency, ~800k examples of rejection sampling + SFT, general RL with RMs only in the last 400 steps). For small models, distillation pays off more than RL.',
    'Failures: entropy drops when likely actions get a high advantage (\\(\\Delta H\\approx-\\operatorname{Cov}(\\log\\pi,\\Delta z)\\)); length inflates or collapses; spurious rewards, solution leakage, obfuscated CoT and tool-call storms show that verifiable does not mean inviolable.',
    'Training ≠ inference: the same weights give different probabilities in vLLM and in Megatron. TIS truncates \\(\\rho\\), IcePop zeroes outside [1/β, β] (GLM-5: β = 2), V3.2 adds a corrected KL, masking of off-policy sequences, Keep Routing and Keep Sampling Mask; MoE calls for R3 or a frozen router; FP32 LM head. The Qwen recipe’s determinism overlay (0.25/6.07 → 0/0 nats) is the same kind of fix.',
  ],
  quiz: [
    { q: 'A GRPO group with \\(G=4\\) has rewards [1, 0, 0, 0]. With the population std, what are the advantages of the correct answer and of each wrong one?', o: ['+0.75 and −0.25: the group mean, 0.25, is subtracted and the scale is left as it is', '+1.732 and −0.577: the mean, 0.25, is subtracted and the result is divided by the std, 0.433', '+1.0 and −1.0: the std equalizes the scale of right and wrong answers at ±1', '+1.0 and −0.333: each answer is compared with the mean of the other three'], a: 1, e: 'Mean 0.25 and std \\(\\sqrt{0.1875}=0.433\\): \\(0.75/0.433=+1.732\\) and \\(-0.25/0.433=-0.577\\). “+0.75 and −0.25” is Dr. GRPO, which only centers; “+1.0 and −0.333” is RLOO; ±1 only comes out with two right and two wrong answers, when the group is symmetric.' },
    { q: 'In the original GRPO, two wrong answers in the same group have 300 and 1,700 tokens. What happens to the punishment?', o: ['The long one is punished more in total, in proportion to its length, because each token carries the same negative advantage', 'The long one gets no gradient, because answers above the limit are truncated and dropped from the GRPO loss', 'Both get the same total punishment; each token of the long one is punished ~5.7× less, and wrong answers tend to grow', 'The group std compensates for the length difference, and every token of both answers gets the same punishment'], a: 2, e: 'The \\(1/|o_i|\\) factor gives each answer the same total weight (−0.25 with \\(G=4\\)), spread over more tokens in the long one (1,700/300 ≈ 5.7). The first option describes the fix, not GRPO: Dr. GRPO (constant MAX_TOKENS) and DAPO (token-level loss, \\(1/\\sum|o_i|\\)) make the long wrong answer the most punished (−0.208 and −0.567). Dropping truncated answers from the loss is a DAPO fix, and the std does not see length.' },
    { q: 'For a sampled token with \\(\\pi_\\theta=0.01\\) and \\(\\pi_{\\text{ref}}=0.2\\), what do the KL estimators say?', o: ['k1 = −2.996, k2 = 4.487 and k3 = 16.004: k3 blows up when the policy abandons a token that the reference favors', 'k1 = 2.996, k2 = 4.487 and k3 = 2.996: the three agree in magnitude, because they estimate the same reverse KL', 'k1 = −2.996, k2 = 0 and k3 = 0: k2 and k3 only count when the policy gives more probability than the reference', 'k1 = 0.693, k2 = 0.240 and k3 = 0.193: near the reference the three stay small and similar to each other'], a: 0, e: 'With \\(x=\\pi_{\\text{ref}}/\\pi_\\theta=20\\): k1 = \\(\\ln(1/20)\\), k2 = \\(\\tfrac12(\\ln 20)^2\\) and k3 = \\(20-1-\\ln 20\\). The three only agree on <i>average</i>, not token by token; the values in the last option are those of the text’s other example (0.5 versus 0.25). The gradient of k3 adds \\(\\beta(x-1)\\) to the coefficient; with β = 0.04, that is 0.76, the size of a whole advantage. This is the motivation for DeepSeek-V3.2’s corrected KL and for many recipes simply turning the KL off.' },
    { q: 'Why is GSPO, with a sequence-level importance ratio, more stable than GRPO on MoE models?', o: ['Because GSPO freezes the router during RL, so the rollout routing and the training routing always match', 'Because GSPO trains one critic per expert, which absorbs the variation caused by routing changes', 'Because GSPO uses a larger ε, which tolerates the high ratios of tokens that switched experts', 'Because the ratio of a token whose routing changed is almost noise, and the sequence’s geometric mean dilutes that noise'], a: 3, e: 'In Qwen3-30B-A3B, ~10% of the activated experts change between the old and the new policy at each update. The geometric mean \\(s_i\\) moves slowly, the clip ranges <i>shrink</i> to \\(3\\times10^{-4}\\)/\\(4\\times10^{-4}\\), and GSPO does without the Routing Replay that GRPO needed to stay stable. Freezing the router is MiMo-V2.6’s solution, and GSPO has no critic.' },
    { q: 'In GLM-5’s reasoning RL (IcePop, β = 2), four tokens have a mismatch \\(\\rho=\\pi_{\\text{train}}/\\pi_{\\text{infer}}\\) = [0.98, 1.6, 2.5, 0.45]. What happens?', o: ['The 2.5 is truncated to 2 and the others pass through unchanged, as in any importance correction', '0.98 and 1.6 stay; the tokens with 2.5 and 0.45 are zeroed, because they lie outside [0.5, 2]', 'Only 0.98 stays; the other three are zeroed, because they are more than 0.5 away from 1', 'Nothing is masked: β = 2 is the weight of the KL term, which pulls the policy toward the reference'], a: 1, e: '\\(\\operatorname{pop}(\\rho,1/\\beta,\\beta)=\\rho\\) only inside [1/β, β] = [0.5, 2]. Truncating at 2 would be TIS with C = 2. And in GLM-5 the IcePop β is the threshold, not a KL coefficient: GLM-5 does not even use the KL term.' },
    { q: 'Your math verifier gives reward 1 whenever the reference number appears anywhere in the answer. What does GRPO tend to learn?', o: ['Nothing special: the verifier has no weights, so the policy has nothing in it to exploit', 'To answer more briefly: the verifier ignores the reasoning, and short answers get the final number wrong less often', 'To list several numbers or boxes: those answers get a positive advantage over honest siblings that are wrong', 'To abandon the <code>\\boxed{}</code>: the verifier does not read format, and RL stops reinforcing the box'], a: 2, e: 'The advantage is relative to the group: if the hack passes and the honest wrong answers do not, the hack goes up. Having no weights is not enough: the flaw is in the rule, not in a model. Use strict extraction (a single box, compared by equivalence), read samples during training and test the verifier against adversarial answers before training.' },
  ],
  mount(root, LA) {
    const V = LA.viz, UI = LA.ui;
    const RED = () => V.C().red || '#FB2C36';
    const fmt = (x, d = 2) => (Number.isFinite(x) ? (+x).toFixed(d) : '∞');
    const sgn = (x, d = 3) => (x > 1e-9 ? '+' : x < -1e-9 ? '−' : '') + fmt(Math.abs(x), d);
    const SUP = { 0: '⁰', 1: '¹', 2: '²', 3: '³', 4: '⁴', 5: '⁵', 6: '⁶', 7: '⁷', 8: '⁸', 9: '⁹', '-': '⁻' };
    const sci = (x, d = 1) => {
      if (x === 0) return '0';
      if (Math.abs(x - 1) < 1e-12) return '1';
      const e = Math.floor(Math.log10(Math.abs(x)));
      if (e >= -3 && e < 5) return fmt(x, e >= 0 ? Math.max(0, 3 - e) : Math.min(6, 2 - e));
      const m = x / Math.pow(10, e);
      return (x < 0 ? '−' : '') + fmt(Math.abs(m), d) + '·10' + String(e).split('').map((c) => SUP[c] || c).join('');
    };
    const thou = (n) => Math.round(n).toLocaleString('en-US');
    const nocase = (c) => { const el = c && (c.el || c); if (el && el.querySelectorAll) el.querySelectorAll('label, span').forEach((n) => { if (!n.classList.contains('val')) n.style.textTransform = 'none'; }); return c; };
    /** stage height by width; wide/narrow can be functions (height per mode). Returns set() to reapply. */
    function adaptH(fig, wide, narrow, bp = 640) {
      const st = fig.querySelector('.la-fig-stage');
      const val = (v) => (typeof v === 'function' ? v() : v);
      const set = () => { const want = val(fig.clientWidth < bp ? narrow : wide) + 'px'; if (st.style.height !== want) st.style.height = want; };
      set();
      const ro = new ResizeObserver(set); ro.observe(fig);
      if (LA.onCleanup) LA.onCleanup(() => ro.disconnect());
      return set;
    }
    const tw = (g, s, size, font = 'sans', weight = 400) => { g.font = V.font(size, font, weight); return g.measureText(String(s)).width; };
    /** text on a small plate (stage color), so lines and dots do not cut through it */
    function tag(g, s, x, y, o = {}) {
      const C = V.C(), size = o.size || 11, font = o.font || 'sans', weight = o.weight || 400, pad = o.pad == null ? 3 : o.pad;
      const wT = Math.min(tw(g, s, size, font, weight), o.maxW || 1e9), al = o.align || 'left';
      const x0 = al === 'right' ? x - wT : al === 'center' ? x - wT / 2 : x;
      V.box(g, x0 - pad, y - size * 0.9 - pad / 2, wT + 2 * pad, size * 1.25 + pad, { r: 4, fill: V.alpha(C.bg, o.bgA == null ? 0.88 : o.bgA) });
      V.text(g, s, x, y, Object.assign({}, o, { size, font, weight, align: al }));
    }
    function chip(g, x, y, w, h, label, o = {}) {
      const C = V.C();
      V.box(g, x, y, w, h, { r: o.r == null ? 6 : o.r, fill: o.fill || V.alpha(C.muted, 0.12), stroke: o.stroke || V.alpha(C.muted, 0.4), lw: o.lw || 1.2 });
      if (label != null) V.text(g, label, x + w / 2, y + h / 2 + 0.5, { size: o.size || 11.5, font: o.font || 'mono', weight: o.weight || 500, color: o.color || C.ink, align: 'center', base: 'middle', maxW: w - 6 });
    }
    function mark(g, x, y, ok, s = 7, col) {
      const C = V.C();
      g.lineWidth = 2.4; g.strokeStyle = col || (ok ? C.a : RED()); g.lineCap = 'round'; g.beginPath();
      if (ok) { g.moveTo(x - s, y); g.lineTo(x - s * 0.3, y + s * 0.7); g.lineTo(x + s, y - s * 0.8); }
      else { g.moveTo(x - s * 0.8, y - s * 0.8); g.lineTo(x + s * 0.8, y + s * 0.8); g.moveTo(x + s * 0.8, y - s * 0.8); g.lineTo(x - s * 0.8, y + s * 0.8); }
      g.stroke(); g.lineCap = 'butt';
    }
    /** character: a rounded little square with eyes and a mouth */
    function blob(g, x, y, s, col, o = {}) {
      const C = V.C();
      if (o.alpha != null) g.globalAlpha = o.alpha;
      V.box(g, x - s / 2, y - s / 2, s, s, { r: s * 0.32, fill: o.fill || V.alpha(col, 0.85), stroke: col, lw: 1.2 });
      if (s >= 11) {
        const eye = C.dark ? '#06140d' : '#ffffff';
        const ex = s * 0.17, ey = y - s * 0.08, er = Math.max(1, s * 0.07);
        V.dot(g, x - ex, ey, er, eye); V.dot(g, x + ex, ey, er, eye);
        g.beginPath();
        if (o.mood === 'sad') g.arc(x, y + s * 0.3, s * 0.14, Math.PI * 1.15, Math.PI * 1.85);
        else if (o.mood === 'o') g.arc(x, y + s * 0.16, s * 0.07, 0, Math.PI * 2);
        else g.arc(x, y + s * 0.06, s * 0.15, Math.PI * 0.2, Math.PI * 0.8);
        g.strokeStyle = eye; g.lineWidth = Math.max(1, s * 0.06); g.stroke();
      }
      if (o.alpha != null) g.globalAlpha = 1;
    }
    function star(g, x, y, r, col) {
      g.beginPath();
      for (let i = 0; i < 10; i++) { const a = -Math.PI / 2 + (i * Math.PI) / 5, rr = i % 2 ? r * 0.45 : r; g.lineTo(x + Math.cos(a) * rr, y + Math.sin(a) * rr); }
      g.closePath(); g.fillStyle = col; g.fill();
    }
    const clipRect = (g, x, y, w, h) => { g.save(); g.beginPath(); g.rect(x, y, w, h); g.clip(); };
    const sigm = (x) => 1 / (1 + Math.exp(-x));

    /* ===================================================== FIG 1 — critic */
    (function () {
      const fig = root.querySelector('#fig-critico'); if (!fig) return;
      adaptH(fig, 390, 660);
      const S = { N: 7, rew: 'rm', ref: true, lam: 0.95, T: 500, sig: 'gae', cur: { ppo: [0, 0, 0, 0], grpo: [0, 0, 0, 0] } };
      const COLS = ['a', 'c', 'b', 'd'];
      const NAMES = ['policy', 'critic', 'reference', 'RM'];
      const targets = () => ({ ppo: [16 * S.N, 16 * S.N, 2 * S.N, 2 * S.N], grpo: [16 * S.N, 0, S.ref ? 2 * S.N : 0, S.rew === 'rm' ? 2 * S.N : 0] });
      let H = null;
      const ctl = UI.controls(fig);
      UI.slider(ctl, { label: 'parameters N', min: 1, max: 70, step: 0.5, value: 7, fmt: (v) => fmt(v, v % 1 ? 1 : 0) + 'B', onInput: (v) => { S.N = v; upd(); } });
      UI.seg(ctl, { label: 'reward', options: [{ v: 'rm', label: 'reward model' }, { v: 'ver', label: 'verifier' }], value: 'rm', onChange: (v) => { S.rew = v; upd(); } });
      UI.toggle(ctl, { label: 'β > 0 (loads the reference)', value: true, onChange: (v) => { S.ref = v; upd(); } });
      UI.seg(ctl, { label: 'GAE λ', options: [{ v: 0.95, label: '0.95' }, { v: 0.99, label: '0.99' }, { v: 0.999, label: '0.999' }, { v: 1, label: '1' }], value: 0.95, onChange: (v) => { S.lam = v; upd(); } });
      UI.seg(ctl, { label: 'distance to the end', options: [50, 500, 8000].map((v) => ({ v, label: thou(v) })), value: 500, onChange: (v) => { S.T = v; upd(); } });
      UI.seg(ctl, { label: 'who carries the signal', options: [{ v: 'gae', label: 'critic + GAE' }, { v: 'grpo', label: 'GRPO (outcome)' }], value: 'gae', onChange: (v) => { S.sig = v; upd(); } });
      const rP = UI.readout(ctl, 'PPO-RLHF', '—'), rG = UI.readout(ctl, 'GRPO', '—'), rW = UI.readout(ctl, 'signal at 1st token', '—');
      nocase(ctl);
      function upd() {
        const T = targets();
        const tp = T.ppo.reduce((a, b) => a + b, 0), tg = T.grpo.reduce((a, b) => a + b, 0);
        const st = (x) => (x > 261.3 ? 'bad' : x > 130.7 ? 'warn' : 'ok');
        rP.set(fmt(tp, 0) + ' GB', st(tp)); rG.set(fmt(tg, 0) + ' GB', st(tg));
        if (S.sig === 'grpo') rW.set('same \u00C2 everywhere', 'ok');
        else { const wv = Math.pow(S.lam, S.T); rW.set(S.lam === 1 ? '1 (no decay)' : sci(wv), wv < 0.01 ? 'bad' : wv < 0.3 ? 'warn' : 'ok'); }
        if (H) H.redraw();
      }
      function drawMem(g, A, C) {
        const T = targets();
        ['ppo', 'grpo'].forEach((k) => { for (let i = 0; i < 4; i++) S.cur[k][i] += (T[k][i] - S.cur[k][i]) * 0.14; });
        V.text(g, 'Memory: weights + optimizer', A.x, A.y + 14, { size: 13, weight: 600, color: C.ink });
        V.text(g, '16 B/trainable param · 2 B/frozen', A.x, A.y + 31, { size: 11, color: C.muted, font: 'mono', maxW: A.w });
        const lw = 76, x0 = A.x + lw, W = A.w - lw - 8;
        const maxGB = Math.max(290, 36 * S.N * 1.08);
        const sx = (gb) => x0 + (gb / maxGB) * W;
        const top = A.y + 56, rowH = V.clamp((A.h - 130) / 2.3, 26, 46), gap = 30;
        clipRect(g, A.x, A.y, A.w, A.h);
        [['PPO-RLHF', 'ppo'], ['GRPO', 'grpo']].forEach(([lab, k], r) => {
          const y = top + r * (rowH + gap);
          let acc = 0;
          for (let i = 0; i < 4; i++) {
            const v = S.cur[k][i]; if (v < 0.05) continue;
            const xa = sx(acc), xb = sx(acc + v), col = C[COLS[i]];
            V.box(g, xa, y, Math.max(1, xb - xa - 1.5), rowH, { r: 5, fill: V.alpha(col, 0.3), stroke: col, lw: 1.2 });
            if (xb - xa > tw(g, NAMES[i], 11) + 10) V.text(g, NAMES[i], (xa + xb) / 2, y + rowH / 2 + 4, { size: 11, align: 'center', color: C.ink });
            acc += v;
          }
          if (k === 'grpo') {
            const xa = sx(acc) + 4, xb = sx(acc + 16 * S.N) + 4;
            g.setLineDash([4, 4]); V.box(g, xa, y + 3, Math.max(10, xb - xa), rowH - 6, { r: 5, stroke: V.alpha(C.c, 0.75), lw: 1.3 }); g.setLineDash([]);
            if (xb - xa > 74) V.text(g, '✕ critic', (xa + xb) / 2, y + rowH / 2 + 4, { size: 11, align: 'center', color: C.c });
          }
          const tot = T[k].reduce((a2, b2) => a2 + b2, 0);
          V.text(g, lab, A.x, y + rowH / 2 - 3, { size: 12, weight: 600, color: C.ink });
          V.text(g, fmt(tot, 0) + ' GB', A.x, y + rowH / 2 + 13, { size: 11.5, font: 'mono', color: tot > 261.3 ? RED() : tot > 130.7 ? C.d : C.a });
        });
        [130.7, 261.3].forEach((gb) => {
          const x = sx(gb); if (x > x0 + W) return;
          V.line(g, x, top - 10, x, top + 2 * rowH + gap + 8, { color: V.alpha(C.ink, 0.45), dash: [3, 4], lw: 1.2 });
          V.text(g, fmt(gb, 1) + ' GB', x, top + 2 * rowH + gap + 22, { size: 11, font: 'mono', color: C.muted, align: 'center' });
        });
        // legend
        let lx = A.x, ly = top + 2 * rowH + gap + 44;
        NAMES.forEach((n, i) => {
          const wN = tw(g, n, 11) + 22;
          if (lx + wN > A.x + A.w) { lx = A.x; ly += 17; }
          V.box(g, lx, ly - 9, 10, 10, { r: 2, fill: V.alpha(C[COLS[i]], 0.5), stroke: C[COLS[i]], lw: 1 });
          V.text(g, n, lx + 14, ly, { size: 11, color: C.muted });
          lx += wN;
        });
        V.text(g, 'lines: 1 and 2 DGX Sparks', A.x, ly + 18, { size: 11, color: C.muted, font: 'mono' });
        const note = S.rew === 'rm' ? (S.ref ? 'GRPO = 16N + 2N (ref) + 2N (RM) = 20N bytes' : 'GRPO without reference: 16N + 2N (RM) = 18N bytes') : (S.ref ? 'verifier runs on CPU: 16N + 2N (ref) = 18N bytes' : 'verifier and β = 0: policy only, 16N bytes');
        if (ly + 44 < A.y + A.h) V.wrap(g, 'PPO = 2×16N + 2×2N = 36N bytes · ' + note, A.x, ly + 40, A.w, { size: 11.5, color: C.ink, lineH: 16, maxLines: 3 });
        g.restore();
      }
      function drawSig(g, B, t, C) {
        const gae = S.sig === 'gae';
        V.text(g, 'The lighthouse of the final reward', B.x, B.y + 14, { size: 13, weight: 600, color: C.ink });
        V.text(g, gae ? 'direct reward weight on the token: λ^d' : 'GRPO: the answer’s \u00C2 stamped on every token', B.x, B.y + 31, { size: 11, color: C.muted, font: 'mono', maxW: B.w });
        const n = V.clamp(Math.floor((B.w - 40) / 13), 16, 44);
        const x0 = B.x + 2, x1 = B.x + B.w - 30, cw = (x1 - x0) / n;
        const yS = B.y + 52, hS = V.clamp(B.h * 0.14, 22, 34);
        const wts = [];
        for (let i = 0; i < n; i++) { const d = Math.round(((n - 1 - i) / (n - 1)) * S.T); wts.push(gae ? Math.pow(S.lam, d) : 1); }
        const per = 2.8, ph = (t % per) / per;
        for (let i = 0; i < n; i++) {
          const x = x0 + i * cw;
          const a = gae ? 0.06 + 0.9 * wts[i] : 0.3 + 0.55 * (0.5 + 0.5 * Math.cos(ph * Math.PI * 2));
          V.box(g, x + 1, yS, Math.max(2, cw - 2), hS, { r: 3, fill: V.alpha(C.a, a), stroke: V.alpha(C.a, 0.35), lw: 1 });
        }
        if (gae) {
          const pi = (n - 1) * (1 - ph), ii = Math.round(pi);
          V.glow(g, x0 + pi * cw + cw / 2, yS + hS / 2, 20, C.d, Math.max(0.06, wts[ii] || 0));
        } else {
          for (let i = 0; i < n; i += Math.max(1, Math.floor(n / 8))) V.text(g, '+', x0 + i * cw + cw / 2, yS + hS / 2 + 4, { size: 11, color: C.ink, align: 'center', font: 'mono' });
        }
        star(g, x1 + 16, yS + hS / 2, 10, C.d);
        V.text(g, 'R', x1 + 16, yS + hS + 16, { size: 11, font: 'mono', color: C.d, align: 'center' });
        V.text(g, 'start of answer', x0, yS + hS + 16, { size: 11, color: C.muted });
        V.text(g, 'end', x1, yS + hS + 16, { size: 11, color: C.muted, align: 'right' });
        // weight curve
        const yC = yS + hS + 34, hC = B.y + B.h - yC - 58;
        if (hC > 34) {
          V.line(g, x0, yC + hC, x1, yC + hC, { color: V.alpha(C.ink, 0.35), lw: 1 });
          V.line(g, x0, yC, x0, yC + hC, { color: V.alpha(C.ink, 0.35), lw: 1 });
          V.text(g, '1', x0 - 4, yC + 4, { size: 11, font: 'mono', color: C.muted, align: 'right' });
          V.text(g, '0', x0 - 4, yC + hC + 4, { size: 11, font: 'mono', color: C.muted, align: 'right' });
          g.beginPath();
          for (let i = 0; i < n; i++) { const x = x0 + i * cw + cw / 2, y = yC + hC - wts[i] * hC; i ? g.lineTo(x, y) : g.moveTo(x, y); }
          g.strokeStyle = C.d; g.lineWidth = 2; g.stroke();
          V.text(g, gae ? 'weight λ^d along the answer' : 'advantage: constant', x0 + 10, yC + 12, { size: 11, color: C.muted });
        }
        const lines = gae
          ? ['50 tokens from the end: ' + sci(Math.pow(S.lam, 50)), thou(S.T) + ' tokens from the end: ' + (S.lam === 1 ? '1' : sci(Math.pow(S.lam, S.T)))]
          : ['outcome advantage: the same on every token', 'e.g. [1,0,0,1,1,0,0,0] ⇒ \u00C2 = +1.291 on hits'];
        lines.forEach((s, i) => V.text(g, s, B.x, B.y + B.h - 26 + i * 16, { size: 11.5, font: 'mono', color: C.ink, maxW: B.w }));
      }
      H = V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const narrow = w < 640;
        const A = narrow ? { x: 12, y: 6, w: w - 24, h: h * 0.47 } : { x: 14, y: 8, w: w * 0.5 - 26, h: h - 16 };
        const B = narrow ? { x: 22, y: h * 0.49 + 6, w: w - 34, h: h * 0.51 - 12 } : { x: w * 0.5 + 22, y: 8, w: w * 0.5 - 36, h: h - 16 };
        drawMem(g, A, C); drawSig(g, B, t, C);
        if (!narrow) V.line(g, w * 0.5, 16, w * 0.5, h - 16, { color: V.alpha(C.ink, 0.12), lw: 1 });
      });
      upd();
    })();

    /* ===================================================== FIG 2 — baselines */
    (function () {
      const fig = root.querySelector('#fig-baselines'); if (!fig) return;
      adaptH(fig, 430, 480);
      const Z = [1.1, 0.7, 0.2, -0.4], P = V.softmax(Z);
      const CUM = []; P.reduce((s, p, i) => (CUM[i] = s + p), 0);
      const REW = { ver: [1, 0, 1, 0], rm: [2.6, 1.4, 2.2, 1.0] };
      const KS = [2, 4, 8, 16];
      const BL = [{ v: 'none', label: 'no baseline' }, { v: 'remax', label: 'ReMax (greedy)' }, { v: 'rloo', label: 'RLOO' }, { v: 'mean', label: 'mean incl. itself' }, { v: 'grpo', label: 'GRPO (÷ std)' }];
      const GREEDY = 0;
      const S = { mode: 'rm', k: 4, bl: 'none', dots: [], sc: null };
      const draw1 = (u) => { for (let i = 0; i < 4; i++) if (u < CUM[i]) return i; return 3; };
      function est(R, k, bl, rnd) {
        const ys = [], Rs = [];
        for (let i = 0; i < k; i++) { const y = draw1(rnd()); ys.push(y); Rs.push(R[y]); }
        const sum = Rs.reduce((a, b) => a + b, 0), mean = sum / k;
        const sd = Math.sqrt(Rs.reduce((a, r) => a + (r - mean) * (r - mean), 0) / k);
        const gr = [0, 0, 0, 0];
        for (let i = 0; i < k; i++) {
          let A;
          if (bl === 'none') A = Rs[i];
          else if (bl === 'remax') A = Rs[i] - R[GREEDY];
          else if (bl === 'rloo') A = Rs[i] - (sum - Rs[i]) / (k - 1);
          else if (bl === 'mean') A = Rs[i] - mean;
          else A = sd > 1e-9 ? (Rs[i] - mean) / sd : 0;
          for (let a = 0; a < 4; a++) gr[a] += (A * ((a === ys[i] ? 1 : 0) - P[a])) / k;
        }
        return gr;
      }
      const trueG = (R) => { const ER = P.reduce((s, p, a) => s + p * R[a], 0); return P.map((p, a) => p * (R[a] - ER)); };
      const TG = { ver: trueG(REW.ver), rm: trueG(REW.rm) };
      const STATS = {};
      (function pre() {
        const rnd = V.rng(5601), M = 1600;
        for (const m of ['ver', 'rm']) for (const k of KS) for (const b of BL) {
          const mean = [0, 0, 0, 0], all = [];
          for (let s = 0; s < M; s++) { const e = est(REW[m], k, b.v, rnd); all.push(e); for (let a = 0; a < 4; a++) mean[a] += e[a] / M; }
          let v4 = 0, v2 = 0;
          for (const e of all) { for (let a = 0; a < 4; a++) v4 += (e[a] - mean[a]) ** 2 / M; v2 += ((e[0] - mean[0]) ** 2 + (e[1] - mean[1]) ** 2) / M; }
          const g = TG[m], gg = g.reduce((s, x) => s + x * x, 0);
          const proj = mean.reduce((s, x, a) => s + x * g[a], 0) / gg;
          STATS[m + k + b.v] = { mean, rms: Math.sqrt(v4), rms2: Math.sqrt(v2), proj };
        }
      })();
      const rr = V.rng(77);
      let H = null;
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'reward', options: [{ v: 'ver', label: 'verifier (0/1)' }, { v: 'rm', label: 'RM scores' }], value: 'rm', onChange: (v) => { S.mode = v; refill(); upd(); } });
      UI.seg(ctl, { label: 'baseline', options: BL, value: 'none', onChange: (v) => { S.bl = v; refill(); upd(); } });
      UI.seg(ctl, { label: 'samples k', options: KS.map((k) => ({ v: k, label: String(k) })), value: 4, onChange: (v) => { S.k = v; refill(); upd(); } });
      const rN = UI.readout(ctl, 'noise (RMS)', '—'), rR = UI.readout(ctl, 'vs. no baseline', '—'), rC = UI.readout(ctl, 'center / true', '—');
      nocase(ctl);
      const colOf = (C) => ({ none: C.c, remax: C.d, rloo: C.a, mean: C.b, grpo: C.f }[S.bl]);
      function refill() { S.dots = []; for (let j = 0; j < 140; j++) { const e = est(REW[S.mode], S.k, S.bl, rr); S.dots.push({ x: e[0], y: e[1], t0: -10, jx: (rr() - 0.5) * 7, jy: (rr() - 0.5) * 7 }); } }
      function upd() {
        const st = STATS[S.mode + S.k + S.bl], s0 = STATS[S.mode + S.k + 'none'];
        rN.set(fmt(st.rms, 3));
        const rel = st.rms / s0.rms;
        rR.set(S.bl === 'none' ? '1× (reference)' : fmt(rel, 2) + '×', S.bl === 'none' ? null : rel < 0.6 ? 'ok' : rel < 0.95 ? 'warn' : 'bad');
        const exact = S.bl === 'mean' ? (S.k - 1) / S.k : S.bl === 'grpo' ? st.proj : 1;
        rC.set(fmt(exact, 2) + '×', Math.abs(exact - 1) < 0.03 ? 'ok' : 'warn');
        if (H) H.redraw();
      }
      H = V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const narrow = w < 640;
        const info = narrow ? { x: 12, y: 6, w: w - 24, h: 60 } : { x: w - 196, y: 14, w: 182, h: h - 28 };
        const plot = narrow ? { x: 12, y: 74, w: w - 24, h: h - 82 } : { x: 14, y: 8, w: w - 226, h: h - 16 };
        if (!LA.paused) for (let j = 0; j < 2; j++) { const e = est(REW[S.mode], S.k, S.bl, rr); S.dots.push({ x: e[0], y: e[1], t0: t, jx: (rr() - 0.5) * 7, jy: (rr() - 0.5) * 7 }); }
        if (S.dots.length > 240) S.dots.splice(0, S.dots.length - 240);
        const st = STATS[S.mode + S.k + S.bl], s0 = STATS[S.mode + S.k + 'none'];
        const G2 = TG[S.mode];
        const target = Math.max(Math.hypot(G2[0], G2[1]) * 1.7, s0.rms2 * 2.2);
        S.sc = S.sc == null ? target : S.sc + (target - S.sc) * 0.1;
        const side = Math.min(plot.w, plot.h);
        const cx = plot.x + plot.w / 2, cy = plot.y + plot.h / 2, half = side / 2 - 18;
        const px = (v) => cx + (v / S.sc) * half, py = (v) => cy - (v / S.sc) * half;
        V.box(g, cx - half - 8, cy - half - 8, 2 * half + 16, 2 * half + 16, { r: 10, fill: V.alpha(C.ink, 0.025), stroke: V.alpha(C.ink, 0.12), lw: 1 });
        V.line(g, cx - half, cy, cx + half, cy, { color: V.alpha(C.ink, 0.3), lw: 1 });
        V.line(g, cx, cy - half, cx, cy + half, { color: V.alpha(C.ink, 0.3), lw: 1 });
        V.text(g, '∂J/∂z_A', cx + half - 2, cy - 6, { size: 11, font: 'mono', color: C.muted, align: 'right' });
        V.text(g, '∂J/∂z_B', cx + 6, cy - half + 10, { size: 11, font: 'mono', color: C.muted });
        const col = colOf(C);
        clipRect(g, cx - half - 8, cy - half - 8, 2 * half + 16, 2 * half + 16);
        const n = S.dots.length;
        S.dots.forEach((d, i) => {
          const age = t - d.t0, rad = 2.8 * Math.min(1, age * 7 + 0.35);
          g.globalAlpha = 0.2 + 0.5 * (i / n);
          V.dot(g, px(d.x) + d.jx, py(d.y) + d.jy, rad, col);
        });
        g.globalAlpha = 1;
        // cloud: RMS radius around the mean
        g.setLineDash([5, 4]); g.beginPath(); g.arc(px(st.mean[0]), py(st.mean[1]), Math.max(3, (st.rms2 / S.sc) * half), 0, Math.PI * 2); g.strokeStyle = V.alpha(col, 0.9); g.lineWidth = 1.5; g.stroke(); g.setLineDash([]);
        g.restore();
        // true gradient
        V.arrow(g, px(0), py(0), px(G2[0]), py(G2[1]), { color: C.ink, lw: 2.2, head: 9 });
        star(g, px(G2[0]), py(G2[1]), 7, C.ink);
        // expected center of the estimator
        const mx = px(st.mean[0]), my = py(st.mean[1]);
        // arrow label: just below the spread circle (outside the cloud and away from the axis labels), with a thread up to the star
        {
          const sx2 = px(G2[0]), sy2 = py(G2[1]), Rp = Math.max(3, (st.rms2 / S.sc) * half);
          const ly = Math.min(Math.max(sy2 + 24, my + Rp + 17), cy + half + 2);
          const lx = V.clamp(sx2, cx - half + 62, cx + half - 62);
          if (ly - sy2 > 28) V.line(g, sx2, sy2 + 9, lx, ly - 13, { color: V.alpha(C.ink, 0.4), lw: 1, dash: [2, 3] });
          tag(g, 'true gradient', lx, ly, { size: 11, color: C.ink, align: 'center' });
        }
        g.beginPath(); g.arc(mx, my, 7, 0, Math.PI * 2); g.strokeStyle = col; g.lineWidth = 2; g.stroke();
        V.line(g, mx - 4, my, mx + 4, my, { color: col, lw: 2 }); V.line(g, mx, my - 4, mx, my + 4, { color: col, lw: 2 });
        // policy panel
        const R = REW[S.mode];
        if (narrow) {
          V.text(g, 'toy policy · greedy = A', info.x, info.y + 12, { size: 11.5, weight: 600, color: C.ink });
          const cw = (info.w - 9) / 4;
          ['A', 'B', 'C', 'D'].forEach((L, i) => {
            const x = info.x + i * (cw + 3);
            chip(g, x, info.y + 22, cw, 30, null, { fill: V.alpha(C.a, 0.05 + 0.4 * P[i]), stroke: V.alpha(C.a, 0.5) });
            V.text(g, L + ' π=' + fmt(P[i], 2), x + cw / 2, info.y + 34, { size: 11, font: 'mono', align: 'center', color: C.ink, maxW: cw - 4 });
            V.text(g, 'R=' + fmt(R[i], R[i] % 1 ? 1 : 0), x + cw / 2, info.y + 48, { size: 11, font: 'mono', align: 'center', color: C.muted, maxW: cw - 4 });
          });
        } else {
          V.text(g, 'Toy policy', info.x, info.y + 12, { size: 12.5, weight: 600, color: C.ink });
          V.text(g, '4 answers · π = softmax(z)', info.x, info.y + 29, { size: 11, color: C.muted, font: 'mono' });
          ['A', 'B', 'C', 'D'].forEach((L, i) => {
            const y = info.y + 44 + i * 40;
            blob(g, info.x + 12, y + 12, 18, i === GREEDY ? C.d : C.b, { mood: i === GREEDY ? 'o' : undefined });
            V.box(g, info.x + 28, y + 4, (info.w - 40) * P[i] / P[0], 10, { r: 3, fill: V.alpha(C.a, 0.55) });
            V.text(g, 'π=' + fmt(P[i], 3) + '  R=' + fmt(R[i], R[i] % 1 ? 1 : 0), info.x + 28, y + 30, { size: 11, font: 'mono', color: C.ink });
          });
          V.text(g, 'greedy (ReMax) = A', info.x, info.y + 214, { size: 11, color: C.d, font: 'mono' });
          V.box(g, info.x, info.y + 228, 10, 10, { r: 5, fill: col });
          V.text(g, BL.find((b) => b.v === S.bl).label, info.x + 16, info.y + 237, { size: 11, color: C.ink, maxW: info.w - 16 });
          V.text(g, '◯ dashed: RMS spread', info.x, info.y + 256, { size: 11, color: C.muted });
          V.text(g, '⊕ estimator center', info.x, info.y + 274, { size: 11, color: C.muted });
          V.text(g, '★ true gradient', info.x, info.y + 292, { size: 11, color: C.muted });
        }
      });
      refill();
      upd();
    })();

    /* ===================================================== FIG 3 — group */
    (function () {
      const fig = root.querySelector('#fig-grupo'); if (!fig) return;
      adaptH(fig, 420, 470);
      const PRE = {
        r1: { R: [1, 0, 0, 1, 1, 0, 0, 0], bin: true, label: '3 of 8 right' },
        all: { R: [1, 1, 1, 1, 1, 1, 1, 1], bin: true, label: 'all right' },
        rm: { R: [0.9, 0.9, 0.9, 0.91], bin: false, label: 'RM nearly tied' },
        k4: { R: [1, 0, 0, 1], bin: true, label: '2 of 4 right' },
      };
      const S = { pre: 'r1', R: PRE.r1.R.slice(), est: 'grpo', std: 'pop', t0: 0, sc: 1.3 };
      let H = null;
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'group', options: Object.keys(PRE).map((k) => ({ v: k, label: PRE[k].label })), value: 'r1', onChange: (v) => { S.pre = v; S.R = PRE[v].R.slice(); restart(); } });
      UI.seg(ctl, { label: 'advantage', options: [{ v: 'grpo', label: 'GRPO (÷ std)' }, { v: 'dr', label: 'Dr. GRPO (mean only)' }, { v: 'rloo', label: 'RLOO (mean of others)' }], value: 'grpo', onChange: (v) => { S.est = v; restart(); } });
      UI.seg(ctl, { label: 'std', options: [{ v: 'pop', label: 'population' }, { v: 'amo', label: 'sample (torch.std)' }], value: 'pop', onChange: (v) => { S.std = v; restart(); } });
      UI.button(ctl, '↺ replay', () => restart());
      const rM = UI.readout(ctl, 'mean', '—'), rS = UI.readout(ctl, 'std', '—'), rA = UI.readout(ctl, '\u00C2 max', '—'), rB = UI.readout(ctl, '\u00C2 min', '—');
      nocase(ctl);
      function stats() {
        const R = S.R, G = R.length, sum = R.reduce((a, b) => a + b, 0), mean = sum / G;
        const ss = R.reduce((a, r) => a + (r - mean) * (r - mean), 0);
        const sd = Math.sqrt(ss / (S.std === 'pop' ? G : G - 1));
        const base = R.map((r) => (S.est === 'rloo' ? (sum - r) / (G - 1) : mean));
        const A = R.map((r, i) => (S.est === 'grpo' ? (sd > 1e-12 ? (r - mean) / sd : 0) : r - base[i]));
        return { G, mean, sd, base, A };
      }
      function restart() {
        S.t0 = H ? H.t : 0;
        const s = stats();
        rM.set(fmt(s.mean, 4)); rS.set(fmt(s.sd, 4), s.sd < 1e-9 ? 'bad' : s.sd < 0.05 ? 'warn' : null);
        const mx = Math.max(...s.A), mn = Math.min(...s.A);
        rA.set(sgn(mx, 3), Math.abs(mx) < 1e-9 ? 'bad' : null); rB.set(sgn(mn, 3), Math.abs(mn) < 1e-9 ? 'bad' : null);
        if (H) H.redraw();
      }
      const PH = ['scores', 'mean', '− baseline', '÷ std', 'tokens'];
      H = V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const narrow = w < 640;
        const st = stats();
        let T = t - S.t0;
        if (T > 12.5) { S.t0 = t; T = 0; }
        const e = (a, b) => V.ease.inOut(V.clamp((T - a) / (b - a), 0, 1));
        const p1 = e(0.9, 1.9), p2 = e(2.2, 3.4), p3 = S.est === 'grpo' ? e(3.7, 4.9) : 0, p4 = e(5.2, 6.4);
        const phase = T < 1.9 ? (T < 0.9 ? 0 : 1) : T < 3.6 ? 2 : T < 5.1 ? 3 : 4;
        // phase chips
        const chW = Math.min(108, (w - 24 - 4 * 6) / 5);
        PH.forEach((s, i) => {
          const on = i === phase, skip = i === 3 && S.est !== 'grpo';
          chip(g, 12 + i * (chW + 6), 6, chW, 22, (i + 1) + ' · ' + s, { fill: on ? V.alpha(C.a, 0.3) : V.alpha(C.muted, 0.08), stroke: on ? C.a : V.alpha(C.muted, 0.35), color: skip ? V.alpha(C.muted, 0.6) : C.ink, size: 11 });
        });
        const G = st.G;
        const left = narrow ? 30 : 46, right = narrow ? 8 : 120;
        const tokH = narrow ? 64 : 80;
        const top = 46, bot = h - tokH - 30;
        const cw = (w - left - right) / G, bw = Math.min(46, cw * 0.62);
        const vals = S.R.map((r, i) => {
          let v = r - p2 * st.base[i];
          if (S.est === 'grpo') v *= V.lerp(1, 1 / (st.sd > 1e-12 ? st.sd : 1), p3);
          return v;
        });
        const tgt = Math.max(1.15, Math.max(...vals.map(Math.abs)) * 1.2);
        S.sc += (tgt - S.sc) * 0.1;
        const y0 = (top + bot) / 2;
        const sy = (v) => y0 - (v / S.sc) * ((bot - top) / 2);
        // axis
        V.line(g, left - 6, sy(0), w - right + 4, sy(0), { color: V.alpha(C.ink, 0.45), lw: 1.2 });
        V.text(g, '0', left - 10, sy(0) + 4, { size: 11, font: 'mono', color: C.muted, align: 'right' });
        [1, -1].forEach((v) => { if (Math.abs(v) < S.sc) { V.line(g, left - 6, sy(v), w - right + 4, sy(v), { color: V.alpha(C.ink, 0.1), lw: 1, dash: [2, 4] }); V.text(g, v > 0 ? '+1' : '−1', left - 10, sy(v) + 4, { size: 11, font: 'mono', color: C.muted, align: 'right' }); } });
        // baseline
        if (p1 > 0.01) {
          if (S.est === 'rloo') {
            st.base.forEach((b, i) => { const x = left + i * cw + cw / 2; const y = sy(b * (1 - p2)); g.globalAlpha = p1; V.line(g, x - bw / 2 - 5, y, x + bw / 2 + 5, y, { color: C.d, lw: 2.2 }); g.globalAlpha = 1; });
            if (!narrow) { g.globalAlpha = p1; V.wrap(g, 'baseline = mean of the others', w - right + 10, top + 40, right - 14, { size: 11, color: C.d, lineH: 14, maxLines: 2 }); g.globalAlpha = 1; }
          } else {
            const y = sy(st.mean * (1 - p2));
            g.globalAlpha = p1; V.line(g, left - 6, y, w - right + 4, y, { color: C.d, lw: 2, dash: [6, 4] }); g.globalAlpha = 1;
            // on mobile the last bar reaches the right edge: the label moves up into the strip between the chips and the chart
            if (narrow) V.text(g, 'mean = ' + fmt(st.mean, 3), left - 18, 42, { size: 11, font: 'mono', color: C.d, alpha: p1 });
            else V.text(g, 'mean = ' + fmt(st.mean, 3), w - right + 10, y - 6, { size: 11, font: 'mono', color: C.d, alpha: p1 });
          }
        }
        const divTxt = st.sd > 1e-12 ? '÷ ' + fmt(st.sd, 4) : '÷ 0 ⇒ \u00C2 := 0';
        if (p3 > 0.01 && !narrow) V.text(g, divTxt, w - right + 10, top + 14, { size: 13, font: 'mono', weight: 600, color: C.b, alpha: p3, maxW: right - 14 });
        if (p3 > 0.01 && narrow) V.text(g, divTxt, w - 10, 42, { size: 12, font: 'mono', weight: 600, color: C.b, alpha: p3, align: 'right' });
        // bars
        const dig = p2 > 0.02 ? (narrow ? 2 : 3) : PRE[S.pre].bin ? 0 : 2;
        vals.forEach((v, i) => {
          const x = left + i * cw + cw / 2;
          const y1 = sy(0), y2 = sy(v);
          const pos = v >= 0;
          const col = p2 > 0.5 ? (Math.abs(v) < 1e-9 ? C.muted : pos ? C.a : RED()) : C.b;
          V.box(g, x - bw / 2, Math.min(y1, y2), bw, Math.max(1.5, Math.abs(y2 - y1)), { r: 4, fill: V.alpha(col, 0.55), stroke: col, lw: 1.2 });
          const lab = p2 > 0.02 ? sgn(v, dig) : fmt(v, dig);
          V.text(g, lab, x, pos ? y2 - 6 : y2 + 15, { size: 11, font: 'mono', color: C.ink, align: 'center' });
          blob(g, x, bot + 12, 15, S.R[i] >= 0.995 || (!PRE[S.pre].bin && S.R[i] === Math.max(...S.R)) ? C.a : C.c, { mood: S.R[i] >= 0.5 ? undefined : 'sad' });
        });
        // stamped tokens
        const ty = bot + 26, cells = 8, chh = (tokH - 8) / cells;
        st.A.forEach((A, i) => {
          const x = left + i * cw + cw / 2;
          for (let c = 0; c < cells; c++) {
            const col = Math.abs(A) < 1e-9 ? C.muted : A > 0 ? C.a : RED();
            const a = 0.06 + p4 * Math.min(0.85, 0.2 + Math.abs(A) / 2.2) * V.clamp((T - 5.2 - c * 0.08) / 0.4, 0, 1);
            V.box(g, x - bw / 2, ty + c * chh, bw, chh - 1.5, { r: 2, fill: V.alpha(col, a) });
          }
        });
        if (!narrow) {
          V.text(g, 'tokens of each', w - right + 10, ty + 14, { size: 11, color: C.muted });
          V.text(g, 'answer: all', w - right + 10, ty + 29, { size: 11, color: C.muted });
          V.text(g, 'with the same \u00C2', w - right + 10, ty + 44, { size: 11, color: C.muted });
        }
        if (Math.max(...st.A.map(Math.abs)) < 1e-9 && p2 > 0.5) {
          const msg = 'std 0 ⇒ \u00C2 = 0: no gradient (dynamic sampling discards it)';
          const mw = Math.min(w - 40, tw(g, msg, 12.5, 'sans', 600) + 24);
          V.box(g, w / 2 - mw / 2, top + 10, mw, 28, { r: 8, fill: V.alpha(RED(), 0.12), stroke: RED() });
          V.text(g, msg, w / 2, top + 28, { size: 12.5, weight: 600, align: 'center', color: C.ink, maxW: mw - 12 });
        }
        if (PRE[S.pre].bin && T < 1) V.text(g, 'click a bar to flip right ↔ wrong', left, h - 8, { size: 11, color: C.muted });
        // click
        const P = H.pointer;
        if (P.clicked && PRE[S.pre].bin && P.y > top && P.y < bot + 20) {
          const i = Math.floor((P.x - left) / cw);
          if (i >= 0 && i < G) { S.R[i] = S.R[i] ? 0 : 1; restart(); S.t0 = t - 2.2; }
        }
      });
      restart();
    })();

    /* ===================================================== FIG 4 — KL */
    (function () {
      const fig = root.querySelector('#fig-kl'); if (!fig) return;
      adaptH(fig, 440, 520);
      const fmtP = (p) => (p >= 0.1 ? fmt(p, 2) : p >= 0.01 ? fmt(p, 3) : fmt(p, 4));
      const S = { lt: Math.log10(0.01), lr: Math.log10(0.2) };
      let H = null;
      const ctl = UI.controls(fig);
      const sT = UI.slider(ctl, { label: 'π_θ (policy)', min: -3, max: 0, step: 0.01, value: S.lt, fmt: (v) => fmtP(Math.pow(10, v)), onInput: (v) => { S.lt = v; upd(); } });
      const sR = UI.slider(ctl, { label: 'π_ref (reference)', min: -3, max: 0, step: 0.01, value: S.lr, fmt: (v) => fmtP(Math.pow(10, v)), onInput: (v) => { S.lr = v; upd(); } });
      UI.seg(ctl, { label: 'examples', options: [{ v: 'far', label: '0.01 vs 0.2' }, { v: 'near', label: '0.5 vs 0.25' }, { v: 'eq', label: 'equal' }], value: 'far', onChange: (v) => {
        const p = { far: [0.01, 0.2], near: [0.5, 0.25], eq: [0.3, 0.3] }[v];
        S.lt = Math.log10(p[0]); S.lr = Math.log10(p[1]); sT.set(S.lt, true); sR.set(S.lr, true); upd();
      } });
      const r1 = UI.readout(ctl, 'k1', '—'), r2 = UI.readout(ctl, 'k2', '—'), r3 = UI.readout(ctl, 'k3', '—'), rP = UI.readout(ctl, 'pull β(x−1), β = 0.04', '—');
      nocase(ctl);
      const vals = () => { const u = Math.LN10 * (S.lr - S.lt); return { u, k1: -u, k2: 0.5 * u * u, k3: Math.exp(u) - 1 - u, pull: 0.04 * (Math.exp(u) - 1) }; };
      function upd() {
        const v = vals();
        r1.set(fmt(v.k1, 3), v.k1 < -1e-9 ? 'bad' : null); r2.set(fmt(v.k2, 3)); r3.set(fmt(v.k3, 3), v.k3 > 4 ? 'bad' : v.k3 > 1 ? 'warn' : 'ok'); rP.set(sgn(v.pull, 3), Math.abs(v.pull) > 0.3 ? 'warn' : null);
        if (H) H.redraw();
      }
      H = V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const v = vals();
        const narrow = w < 640;
        const sceneH = narrow ? 168 : 160;
        // scene
        const ax0 = 34, ax1 = w - 34, gy = sceneH - 34;
        const lx = (l) => ax0 + ((l + 3) / 3) * (ax1 - ax0);
        V.line(g, ax0, gy, ax1, gy, { color: V.alpha(C.ink, 0.4), lw: 1.4 });
        [-3, -2, -1, 0].forEach((l) => { V.line(g, lx(l), gy - 4, lx(l), gy + 4, { color: V.alpha(C.ink, 0.4), lw: 1 }); V.text(g, ['0.001', '0.01', '0.1', '1'][l + 3], lx(l), gy + 18, { size: 11, font: 'mono', color: C.muted, align: 'center' }); });
        V.text(g, 'probability of the sampled token (log scale)', w / 2, gy + 32, { size: 11, color: C.muted, align: 'center', maxW: w - 20 });
        const xr = lx(S.lr), xt = lx(S.lt);
        // post
        V.box(g, xr - 4, gy - 58, 8, 58, { r: 3, fill: V.alpha(C.b, 0.6), stroke: C.b });
        V.dot(g, xr, gy - 60, 7, C.b);
        V.text(g, 'π_ref', xr, gy - 72, { size: 11.5, font: 'mono', color: C.b, align: 'center' });
        // critter
        const bob = Math.sin(t * 4) * 2, dy = gy - 16 + bob;
        const tension = V.clamp(v.k3 / 6, 0, 1);
        const sag = 34 / (1 + v.k3 * 1.2) + Math.sin(t * 2.2) * 3 * (1 - tension);
        const mx = (xr + xt) / 2, my = Math.max(gy - 64, (gy - 58 + dy) / 2) + sag;
        [[C.muted, 1 - tension], [RED(), tension]].forEach(([c, a]) => { if (a < 0.02) return; g.beginPath(); g.moveTo(xr, gy - 56); g.quadraticCurveTo(mx, my, xt, dy - 4); g.strokeStyle = V.alpha(c, Math.max(0.15, a)); g.lineWidth = 2.4; g.stroke(); });
        blob(g, xt, dy, 26, C.a, { mood: v.k3 > 3 ? 'o' : undefined });
        V.text(g, 'π_θ', xt, dy - 22, { size: 11.5, font: 'mono', color: C.a, align: 'center' });
        // pull arrow
        if (Math.abs(xr - xt) > 4) {
          const dir = Math.sign(xr - xt), L = Math.min(Math.abs(xr - xt) - 8, 26 + 90 * Math.min(1, Math.abs(v.pull) / 0.8));
          if (L > 10) V.arrow(g, xt + dir * 16, dy + 22, xt + dir * (16 + L), dy + 22, { color: v.pull > 0.3 || v.pull < -0.3 ? RED() : C.d, lw: 2.2 });
        }
        const msg = 'KL pull on the gradient: β(x−1) = ' + sgn(v.pull, 3);
        V.text(g, msg, 12, 16, { size: 11.5, font: 'mono', color: C.ink, maxW: w - 24 });
        // chart
        const P = { x: 46, y: sceneH + 14, w: w - 62, h: h - sceneH - 34 };
        const UX = 4.5, Y0 = -5, Y1 = 18;
        const pxu = (u) => P.x + ((u + UX) / (2 * UX)) * P.w, pyv = (y) => P.y + P.h - ((y - Y0) / (Y1 - Y0)) * P.h;
        V.box(g, P.x, P.y, P.w, P.h, { r: 6, fill: V.alpha(C.ink, 0.02), stroke: V.alpha(C.ink, 0.12), lw: 1 });
        V.line(g, P.x, pyv(0), P.x + P.w, pyv(0), { color: V.alpha(C.ink, 0.35), lw: 1 });
        V.line(g, pxu(0), P.y, pxu(0), P.y + P.h, { color: V.alpha(C.ink, 0.2), lw: 1, dash: [3, 4] });
        [0, 5, 10, 15].forEach((y) => V.text(g, String(y), P.x - 6, pyv(y) + 4, { size: 11, font: 'mono', color: C.muted, align: 'right' }));
        [-4, -2, 0, 2].forEach((u) => V.text(g, (u < 0 ? '−' : '') + Math.abs(u), pxu(u), P.y + P.h + 14, { size: 11, font: 'mono', color: C.muted, align: 'center' }));
        V.text(g, 'u = ln(π_ref / π_θ)', P.x + P.w - 8, P.y + P.h - 8, { size: 11, font: 'mono', color: C.muted, align: 'right' });
        const CURV = [['k1', (u) => -u, C.b], ['k2', (u) => 0.5 * u * u, C.d], ['k3', (u) => Math.exp(u) - 1 - u, C.f]];
        clipRect(g, P.x, P.y, P.w, P.h);
        CURV.forEach(([nm, f, c]) => { g.beginPath(); for (let i = 0; i <= 140; i++) { const u = -UX + (2 * UX * i) / 140, y = V.clamp(f(u), Y0 - 2, Y1 + 2); i ? g.lineTo(pxu(u), pyv(y)) : g.moveTo(pxu(u), pyv(y)); } g.strokeStyle = c; g.lineWidth = 2.2; g.stroke(); });
        g.restore();
        // legend
        CURV.forEach(([nm, , c], i) => { const x = P.x + 10, y = P.y + 16 + i * 16; V.line(g, x, y - 4, x + 18, y - 4, { color: c, lw: 2.4 }); V.text(g, nm + ' = ' + fmt(nm === 'k1' ? v.k1 : nm === 'k2' ? v.k2 : v.k3, 3), x + 24, y, { size: 11.5, font: 'mono', color: C.ink }); });
        // current marker
        const uc = V.clamp(v.u, -UX, UX), xm = pxu(uc);
        V.line(g, xm, P.y, xm, P.y + P.h, { color: V.alpha(C.ink, 0.55), lw: 1.4, dash: [4, 3] });
        CURV.forEach(([nm, f, c]) => { const y = f(v.u); const yy = V.clamp(y, Y0, Y1); V.dot(g, xm, pyv(yy), 4.5, c); if (y > Y1) V.text(g, '↑ ' + fmt(y, 1), xm + 6, P.y + 12, { size: 11, font: 'mono', color: c }); });
        if (Math.abs(v.u) > UX) V.text(g, v.u > 0 ? '→ off scale' : '← off scale', xm + (v.u > 0 ? -6 : 6), P.y + P.h - 8, { size: 11, color: C.muted, align: v.u > 0 ? 'right' : 'left' });
      });
      upd();
    })();

    /* ===================================================== FIG 5 — length */
    (function () {
      const fig = root.querySelector('#fig-comprimento'); if (!fig) return;
      adaptH(fig, 360, 440);
      const S = { m: 'grpo', Lw: 1700, drops: [] };
      const R = [1, 1, 0, 0];
      const Ls = () => [200, 800, 300, S.Lw];
      function weights(m) {
        const L = Ls(), G = 4, sumL = L.reduce((a, b) => a + b, 0);
        return L.map((l, i) => {
          const A = (R[i] - 0.5) / 0.5;
          const tok = m === 'grpo' ? A / (G * l) : m === 'dapo' ? A / sumL : (R[i] - 0.5) / 4096;
          return { tok, tot: tok * l, l };
        });
      }
      let H = null;
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'normalization', options: [{ v: 'grpo', label: 'GRPO (1/G·1/|o|)' }, { v: 'dapo', label: 'DAPO (1/Σ|o|)' }, { v: 'dr', label: 'Dr. GRPO (1/MAX_TOKENS)' }], value: 'grpo', onChange: (v) => { S.m = v; upd(); } });
      UI.slider(ctl, { label: 'long wrong answer (tokens)', min: 300, max: 4000, step: 50, value: 1700, fmt: (v) => thou(v), onInput: (v) => { S.Lw = v; upd(); } });
      const rT = UI.readout(ctl, 'weight/token · long wrong', '—'), rL = UI.readout(ctl, 'total · long wrong', '—'), rS = UI.readout(ctl, 'total · short wrong', '—');
      nocase(ctl);
      function upd() {
        const W = weights(S.m);
        rT.set(sci(W[3].tok, 2)); rL.set(sgn(W[3].tot, 3), Math.abs(W[3].tot) >= Math.abs(W[2].tot) - 1e-12 ? 'ok' : 'bad'); rS.set(sgn(W[2].tot, 3));
        if (H) H.redraw();
      }
      H = V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const narrow = w < 640;
        const W = weights(S.m);
        const totW = narrow ? 96 : 170;
        const x0 = 12, maxStrip = w - totW - 36;
        const Lmax = Math.max(1800, ...W.map((q) => q.l));
        const ppt = maxStrip / Lmax;
        const rowH = (h - 20) / 4;
        const maxTok = Math.max(...W.map((q) => Math.abs(q.tok)));
        const maxTot = Math.max(...W.map((q) => Math.abs(q.tot)));
        const cxT = w - totW / 2 - 10;
        V.line(g, cxT, 8, cxT, h - 8, { color: V.alpha(C.ink, 0.3), lw: 1 });
        V.text(g, 'total weight', cxT, h - 2, { size: 11, color: C.muted, align: 'center' });
        W.forEach((q, i) => {
          const y = 10 + i * rowH;
          const pos = R[i] > 0, col = pos ? C.a : RED();
          const lbl = (pos ? '✓ right' : '✗ wrong') + ' · ' + thou(q.l) + ' tokens';
          V.text(g, lbl, x0, y + 13, { size: 11.5, color: C.ink, weight: 500 });
          V.text(g, 'per token ' + sci(q.tok, 2), x0 + tw(g, lbl, 11.5, 'sans', 500) + 12, y + 13, { size: 11, font: 'mono', color: C.muted, maxW: Math.max(60, maxStrip - tw(g, lbl, 11.5, 'sans', 500) - 12) });
          const sy = y + rowH * 0.5, sh = Math.min(22, rowH * 0.32);
          const sw = q.l * ppt;
          V.box(g, x0, sy, sw, sh, { r: 4, fill: V.alpha(col, 0.1 + 0.7 * (Math.abs(q.tok) / maxTok)), stroke: col, lw: 1.2 });
          const step = 100 * ppt;
          if (step >= 4) { g.strokeStyle = V.alpha(C.ink, 0.18); g.lineWidth = 1; g.beginPath(); for (let k = 1; k * 100 < q.l; k++) { const x = x0 + k * step; g.moveTo(x, sy + 2); g.lineTo(x, sy + sh - 2); } g.stroke(); }
          // drops
          if (!LA.paused && Math.random() < 0.95 * Math.abs(q.tot) / maxTot) S.drops.push({ i, x: x0 + Math.random() * sw, y0: y + 18, y1: sy, t0: t, c: pos });
          // total
          const bl = (Math.abs(q.tot) / maxTot) * (totW / 2 - 8);
          V.box(g, pos ? cxT : cxT - bl, sy, Math.max(1.5, bl), sh, { r: 4, fill: V.alpha(col, 0.55), stroke: col, lw: 1.2 });
          V.text(g, sgn(q.tot, 3), cxT + (pos ? -6 : 6), sy - 4, { size: 11, font: 'mono', color: C.ink, align: pos ? 'right' : 'left' });
        });
        if (S.drops.length > 220) S.drops.splice(0, S.drops.length - 220);
        for (let k = S.drops.length - 1; k >= 0; k--) {
          const d = S.drops[k], age = t - d.t0, f = age / 0.55;
          if (f > 1.6) { S.drops.splice(k, 1); continue; }
          const col = d.c ? C.a : RED();
          if (f <= 1) { V.dot(g, d.x, V.lerp(d.y0, d.y1 + 2, f * f), 2.2, col); }
          else { g.globalAlpha = 1.6 - f; g.beginPath(); g.arc(d.x, d.y1 + 2, 3 + (f - 1) * 8, Math.PI, 2 * Math.PI); g.strokeStyle = col; g.lineWidth = 1.2; g.stroke(); g.globalAlpha = 1; }
        }
      });
      upd();
    })();

    /* ===================================================== FIG 6 — trust regions */
    (function () {
      const fig = root.querySelector('#fig-clip'); if (!fig) return;
      adaptH(fig, 470, 570);
      const LOG = [0.10, -0.05, 0.30, -0.20, 0.05];
      const WORDS = ['So', 'the', 'Wait', 'total', 'is'];
      const M = { grpo: { lo: 0.2, hi: 0.2 }, dapo: { lo: 0.2, hi: 0.28 }, cispo: { hi: 0.28 }, sapo: {}, gspo: { lo: 3e-4, hi: 4e-4 } };
      const S = { m: 'grpo', sgn: 1, drift: 1, tau: 1, pos: LOG.map(() => ({ x: 1, y: 1 })) };
      function gw(r, m, s) {
        if (m === 'grpo' || m === 'dapo') return s > 0 ? (r <= 1 + M[m].hi ? r : 0) : (r >= 1 - M[m].lo ? r : 0);
        if (m === 'cispo') return Math.min(r, 1 + M.cispo.hi);
        if (m === 'sapo') { const tau = s > 0 ? S.tau : S.tau * 1.05, p = sigm(tau * (r - 1)); return 4 * p * (1 - p) * r; }
        return r;
      }
      const ratios = () => LOG.map((l) => Math.exp(l * S.drift));
      const sRatio = () => Math.exp((S.drift * LOG.reduce((a, b) => a + b, 0)) / LOG.length);
      function tokW() {
        const r = ratios();
        if (S.m === 'gspo') { const s = sRatio(); const ok = S.sgn > 0 ? s <= 1 + M.gspo.hi : s >= 1 - M.gspo.lo; return r.map(() => (ok ? s : 0)); }
        return r.map((x) => gw(x, S.m, S.sgn));
      }
      let H = null;
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'method', options: [{ v: 'grpo', label: 'PPO/GRPO 0.2' }, { v: 'dapo', label: 'DAPO 0.2/0.28' }, { v: 'cispo', label: 'CISPO' }, { v: 'sapo', label: 'SAPO' }, { v: 'gspo', label: 'GSPO' }], value: 'grpo', onChange: (v) => { S.m = v; gT.show(v === 'sapo'); upd(); } });
      UI.seg(ctl, { label: 'advantage', options: [{ v: 1, label: '\u00C2 > 0' }, { v: -1, label: '\u00C2 < 0' }], value: 1, onChange: (v) => { S.sgn = v; upd(); } });
      UI.slider(ctl, { label: 'policy drift', min: 0, max: 2.5, step: 0.05, value: 1, fmt: (v) => '×' + fmt(v, 2), onInput: (v) => { S.drift = v; upd(); } });
      const gT = UI.group(ctl);
      UI.slider(gT.el, { label: 'SAPO τ_pos', min: 1, max: 12, step: 0.5, value: 1, fmt: (v) => fmt(v, 1), onInput: (v) => { S.tau = v; upd(); } });
      gT.show(false);
      const rK = UI.readout(ctl, 'tokens with gradient', '—'), rSq = UI.readout(ctl, 's (GSPO)', '—');
      nocase(ctl);
      function upd() {
        const wv = tokW(); const n = wv.filter((x) => x > 1e-9).length;
        rK.set(n + ' / 5', n === 5 ? 'ok' : n === 0 ? 'bad' : 'warn'); rSq.set(fmt(sRatio(), 4));
        if (H) H.redraw();
      }
      H = V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const narrow = w < 640;
        const topH = h * 0.6;
        const P = { x: 44, y: 14, w: w - 60, h: topH - 44 };
        const RX = 2.2, RY = 2.3;
        const px = (r) => P.x + (r / RX) * P.w, py = (y) => P.y + P.h - (y / RY) * P.h;
        V.box(g, P.x, P.y, P.w, P.h, { r: 6, fill: V.alpha(C.ink, 0.02), stroke: V.alpha(C.ink, 0.12), lw: 1 });
        // zones without gradient
        const shade = (a, b) => { V.box(g, px(a), P.y, px(b) - px(a), P.h, { r: 0, fill: V.alpha(C.muted, 0.16) }); };
        if (S.m === 'grpo' || S.m === 'dapo') { if (S.sgn > 0) shade(1 + M[S.m].hi, RX); else shade(0, 1 - M[S.m].lo); }
        if (S.m === 'gspo') { if (S.sgn > 0) shade(1 + M.gspo.hi, RX); else shade(0, 1 - M.gspo.lo); }
        // axes
        [0, 0.5, 1, 1.5, 2].forEach((r) => { V.line(g, px(r), P.y + P.h, px(r), P.y + P.h + 4, { color: V.alpha(C.ink, 0.4), lw: 1 }); V.text(g, fmt(r, r % 1 ? 1 : 0), px(r), P.y + P.h + 16, { size: 11, font: 'mono', color: C.muted, align: 'center' }); });
        [0, 1, 2].forEach((y) => V.text(g, String(y), P.x - 6, py(y) + 4, { size: 11, font: 'mono', color: C.muted, align: 'right' }));
        V.text(g, 'ratio r', P.x + P.w, P.y + P.h + 30, { size: 11, color: C.muted, align: 'right' });
        V.text(g, 'weight of ∇log π', P.x + 6, P.y + 14, { size: 11, color: C.muted });
        // identity
        g.beginPath(); g.moveTo(px(0), py(0)); g.lineTo(px(RX), py(RX)); g.strokeStyle = V.alpha(C.ink, 0.25); g.lineWidth = 1.2; g.setLineDash([5, 5]); g.stroke(); g.setLineDash([]);
        if (S.m !== 'gspo') V.text(g, 'no clip: weight = r', px(1.75), py(2.05), { size: 11, color: C.muted, align: 'right' });
        // method curve
        const col = S.sgn > 0 ? C.a : C.c;
        if (S.m !== 'gspo') {
          g.beginPath(); let prev = null;
          for (let i = 0; i <= 220; i++) {
            const r = (RX * i) / 220, y = gw(r, S.m, S.sgn);
            const X = px(r), Y = py(Math.min(y, RY));
            if (prev != null && Math.abs(prev - y) > 0.3) g.moveTo(X, Y); else i ? g.lineTo(X, Y) : g.moveTo(X, Y);
            prev = y;
          }
          g.strokeStyle = col; g.lineWidth = 3; g.stroke();
        } else {
          V.box(g, px(1) - 2, P.y, 4, P.h, { r: 2, fill: V.alpha(C.d, 0.7) });
          V.wrap(g, 'GSPO decides for the whole sequence: the window is ±0.0004 around 1 (too thin for this scale)', px(1) + 10, P.y + 34, P.w - (px(1) - P.x) - 18, { size: 11.5, color: C.ink, lineH: 15, maxLines: 7 });
        }
        // characters
        const r = ratios(), wv = tokW(), s = sRatio();
        const order = r.map((x, i) => i).sort((a, b) => r[a] - r[b]);
        const below = new Array(5).fill(false);
        for (let q = 1; q < 5; q++) { const a = order[q - 1], b = order[q]; if (Math.abs(px(r[a]) - px(r[b])) < 34 && !below[a]) below[b] = true; }
        r.forEach((ri, i) => {
          const tx = S.m === 'gspo' ? s : ri, ty = wv[i];
          const p = S.pos[i];
          p.x += (tx - p.x) * 0.12; p.y += (ty - p.y) * 0.12;
          const X = px(Math.min(p.x, RX)), Y = py(Math.min(p.y, RY)) - 11 + Math.sin(t * 3 + i) * 1.5;
          const alive = ty > 1e-9;
          blob(g, X + (S.m === 'gspo' ? (i - 2) * (narrow ? 4 : 7) : 0), Y, narrow ? 14 : 18, alive ? col : C.muted, { mood: alive ? undefined : 'sad', alpha: alive ? 1 : 0.75 });
          if (S.m !== 'gspo' && (!narrow || i === 2)) V.text(g, WORDS[i], X, below[i] ? Y + 24 : Y - 14, { size: 11, color: C.ink, align: 'center' });
        });
        // bottom panel: sequence bars
        const B = { x: 44, y: topH + 8, w: w - 60, h: h - topH - 18 };
        const cy = B.y + B.h * 0.55, sc = (B.h * 0.4) / 0.8;
        V.line(g, B.x, cy, B.x + B.w, cy, { color: V.alpha(C.ink, 0.45), lw: 1.2 });
        V.text(g, 'r = 1', B.x - 6, cy + 4, { size: 11, font: 'mono', color: C.muted, align: 'right' });
        if (S.m === 'grpo' || S.m === 'dapo') {
          [[Math.log(1 + M[S.m].hi), '1+ε'], [Math.log(1 - M[S.m].lo), '1−ε']].forEach(([l, lab]) => { V.line(g, B.x, cy - l * sc, B.x + B.w, cy - l * sc, { color: V.alpha(C.ink, 0.35), lw: 1, dash: [4, 4] }); V.text(g, lab, B.x + B.w, cy - l * sc - 4, { size: 11, font: 'mono', color: C.muted, align: 'right' }); });
        }
        const cw = B.w / 5, bw = Math.min(40, cw * 0.5), ls = Math.log(s);
        r.forEach((ri, i) => {
          const x = B.x + i * cw + cw / 2, l = Math.log(ri), alive = wv[i] > 1e-9;
          const c2 = alive ? col : C.muted;
          V.box(g, x - bw / 2, Math.min(cy, cy - l * sc), bw, Math.max(1.5, Math.abs(l * sc)), { r: 3, fill: V.alpha(c2, 0.5), stroke: c2, lw: 1.2 });
          V.text(g, WORDS[i], x, B.y + B.h + 4, { size: 11, color: C.muted, align: 'center' });
        });
        V.line(g, B.x, cy - ls * sc, B.x + B.w, cy - ls * sc, { color: C.d, lw: 2 });
        // labels last, on a plate: the dashed ε lines and the s line pass behind them
        r.forEach((ri, i) => {
          const x = B.x + i * cw + cw / 2, l = Math.log(ri);
          tag(g, fmt(ri, 3), x, l >= 0 ? cy - l * sc - 5 : cy - l * sc + 14, { size: 11, font: 'mono', color: C.ink, align: 'center', pad: 2 });
        });
        tag(g, 's = ' + fmt(s, 4), B.x + cw * 2.5, cy - ls * sc + 15, { size: 11, font: 'mono', color: C.d, align: 'center', pad: 2 });
      });
      upd();
    })();

    /* ===================================================== FIG 7 — verifier */
    (function () {
      const fig = root.querySelector('#fig-verificador'); if (!fig) return;
      const MATH = [
        { t: '48 + 24 = 72 ⇒ \\boxed{72}', box: ['72'], has: true },
        { t: '48 + 96 = 144 ⇒ \\boxed{144}', box: ['144'], has: false },
        { t: 'in total, 72 clips', box: [], has: true },
        { t: '\\boxed{48} \\boxed{24} \\boxed{72}', box: ['48', '24', '72'], has: true, hack: true },
        { t: 'April 48 + May 24 ⇒ \\boxed{72}', box: ['72'], has: true },
        { t: '48 / 2 = 24 ⇒ \\boxed{24}', box: ['24'], has: false },
      ];
      const CODE = [
        { t: 'fixes the bug', res: 'ppppp' },
        { t: 'fixes it, breaks another', res: 'pppfp' },
        { t: 'does not fix it', res: 'ffppp' },
        { t: 'deletes the failing test', res: '--ppp', hack: true },
      ];
      const TESTS = ['F2P₁', 'F2P₂', 'P2P₁', 'P2P₂', 'P2P₃'];
      const STEPS = [['April: 48 clips', 0.95], ['May: half of April', 0.9], ['half of 48 = 24', null], ['48 + 24 = 72', 0.98]];
      const S = { mode: 'mat', strict: true, p3: 0.4, agg: 'prod', t0: 0 };
      // height per mode: on mobile, code and PRM need much less room than the six math cards
      const fitH = adaptH(fig, 400, () => ({ mat: 610, cod: 440, prm: 420 })[S.mode]);
      let H = null;
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'mode', options: [{ v: 'mat', label: 'math' }, { v: 'cod', label: 'code' }, { v: 'prm', label: 'process (PRM)' }], value: 'mat', onChange: (v) => { S.mode = v; gV.show(v !== 'prm'); gP.show(v === 'prm'); fitH(); restart(); } });
      const gV = UI.group(ctl);
      UI.toggle(gV.el, { label: 'strict verifier', value: true, onChange: (v) => { S.strict = v; restart(); } });
      const gP = UI.group(ctl);
      UI.slider(gP.el, { label: 'PRM at step 3', min: 0.05, max: 1, step: 0.01, value: 0.4, fmt: (v) => fmt(v, 2), onInput: (v) => { S.p3 = v; restart(); } });
      UI.seg(gP.el, { label: 'aggregation', options: [{ v: 'prod', label: 'product' }, { v: 'min', label: 'minimum' }], value: 'prod', onChange: (v) => { S.agg = v; restart(); } });
      gP.show(false);
      const rA = UI.readout(ctl, 'correct', '—'), rB = UI.readout(ctl, '\u00C2 of the hack', '—');
      nocase(ctl);
      const lab = (r, s) => { const sp = r.el.querySelector('span'); if (sp) sp.textContent = s; };
      const rewMath = () => MATH.map((c) => (S.strict ? (c.box.length === 1 && c.box[0] === '72' ? 1 : 0) : c.has ? 1 : 0));
      const rewCode = () => CODE.map((c) => (S.strict ? (c.res === 'ppppp' ? 1 : 0) : c.res.includes('f') ? 0 : 1));
      const adv = (R) => { const m = R.reduce((a, b) => a + b, 0) / R.length, sd = Math.sqrt(R.reduce((a, r) => a + (r - m) * (r - m), 0) / R.length); return R.map((r) => (sd > 1e-12 ? (r - m) / sd : 0)); };
      const prmP = () => STEPS.map((s) => (s[1] == null ? S.p3 : s[1]));
      let first = true;
      function restart() {
        S.t0 = (H ? H.t : 0) - (first ? 5.5 : 0);
        first = false;
        if (S.mode === 'prm') {
          const p = prmP(), sc = S.agg === 'prod' ? p.reduce((a, b) => a * b, 1) : Math.min(...p);
          lab(rA, S.agg === 'prod' ? 'PRM (product)' : 'PRM (minimum)'); lab(rB, 'ORM (final only)');
          rA.set(fmt(sc, 3), sc < 0.5 ? 'bad' : 'ok'); rB.set('1.000', 'ok');
        } else {
          const R = S.mode === 'mat' ? rewMath() : rewCode(), A = adv(R), hi = (S.mode === 'mat' ? MATH : CODE).findIndex((c) => c.hack);
          lab(rA, 'correct'); lab(rB, '\u00C2 of the hack');
          rA.set(R.reduce((a, b) => a + b, 0) + ' / ' + R.length); rB.set(sgn(A[hi], 3), A[hi] > 0 ? 'bad' : 'ok');
        }
        if (H) H.redraw();
      }
      function drawMath(g, w, h, T, C, narrow) {
        const hq = V.wrap(g, 'Natalia sold hair clips to 48 friends in April and half as many in May. How many in total?', 12, 16, w - 24, { size: 12, color: C.ink, lineH: 15, maxLines: 2 });
        V.text(g, 'reference answer: 72', 12, 16 + hq + 2, { size: 11.5, font: 'mono', color: C.d });
        const R = rewMath(), A = adv(R);
        const cols = narrow ? 2 : 3, rows = Math.ceil(MATH.length / cols);
        const gx = 10, top = 30 + hq, cw = (w - 24 - gx * (cols - 1)) / cols, chh = (h - top - 34 - 10 * (rows - 1)) / rows;
        const per = 0.75, tA = MATH.length * per + 0.3;
        MATH.forEach((c, i) => {
          const x = 12 + (i % cols) * (cw + gx), y = top + Math.floor(i / cols) * (chh + 10);
          const active = T >= i * per && T < (i + 1) * per, done = T >= (i + 1) * per;
          V.box(g, x, y, cw, chh, { r: 9, fill: active ? V.alpha(C.b, 0.14) : V.alpha(C.ink, 0.035), stroke: active ? C.b : V.alpha(C.ink, 0.18), lw: active ? 1.8 : 1 });
          blob(g, x + 16, y + 18, 18, done ? (R[i] ? C.a : C.c) : C.b, { mood: done && !R[i] ? 'sad' : undefined });
          V.wrap(g, c.t, x + 32, y + 15, cw - 44, { size: 11.5, font: 'mono', color: C.ink, lineH: 15, maxLines: 2 });
          if (c.hack) chip(g, x + cw - 50, y + 32, 42, 18, 'hack', { fill: V.alpha(C.d, 0.2), stroke: C.d, size: 11 });
          if (active) { const by = y + 6 + ((T - i * per) / per) * (chh - 12); V.line(g, x + 4, by, x + cw - 4, by, { color: C.b, lw: 2 }); V.glow(g, x + cw / 2, by, 26, C.b, 0.35); }
          if (done) {
            mark(g, x + cw - 18, y + 18, !!R[i], 7);
            const ext = S.strict
              ? (c.box.length === 0 ? 'no box: invalid' : c.box.length > 1 ? c.box.length + ' boxes: invalid' : 'extracted: ' + c.box[0] + (c.box[0] === '72' ? ' = 72' : ' ≠ 72'))
              : (c.has ? 'contains “72”: passes' : 'no “72” anywhere');
            const midY = y + Math.max(46, chh * 0.5);
            if (chh > 70) V.text(g, ext, x + 12, midY, { size: 11.5, font: 'mono', color: R[i] ? C.a : RED(), maxW: cw - 24 });
            if (c.hack && R[i]) V.text(g, '⚠ RL will reinforce this', x + 12, midY + 16, { size: 11.5, color: C.d, maxW: cw - 70 });
            const f = V.clamp((T - tA) / 0.6, 0, 1);
            if (f > 0) {
              const ax = x + 32, aw = cw - 92, cxA = ax + aw / 2, bl = (Math.abs(A[i]) / 2) * (aw / 2) * f;
              V.line(g, cxA, y + chh - 24, cxA, y + chh - 8, { color: V.alpha(C.ink, 0.35), lw: 1 });
              V.box(g, A[i] >= 0 ? cxA : cxA - bl, y + chh - 21, Math.max(1, bl), 10, { r: 3, fill: A[i] >= 0 ? C.a : RED() });
              V.text(g, '\u00C2 ' + sgn(A[i], 2), A[i] >= 0 ? cxA - 4 : cxA + 4, y + chh - 12, { size: 11, font: 'mono', color: C.ink, align: A[i] >= 0 ? 'right' : 'left', alpha: f });
            }
          }
        });
        V.text(g, S.strict ? 'strict verifier: exactly one \\boxed{}, equal to 72' : 'naive verifier: does “72” appear anywhere?', 12, h - 10, { size: 11.5, font: 'mono', color: S.strict ? C.a : C.d, maxW: w - 24 });
      }
      function drawCode(g, w, h, T, C, narrow, t) {
        const R = rewCode(), A = adv(R);
        const nameW = narrow ? 120 : 200, top = 34;
        const x0 = 12 + nameW, cwid = (w - x0 - (narrow ? 70 : 120)) / 5, rh = Math.min(narrow ? 104 : 72, (h - top - 70) / 4);
        V.text(g, 'patch', 12, top - 10, { size: 11, color: C.muted });
        TESTS.forEach((s, j) => V.text(g, s, x0 + j * cwid + cwid / 2, top - 10, { size: 11, font: 'mono', color: j < 2 ? C.d : C.b, align: 'center' }));
        V.text(g, '\u00C2', w - (narrow ? 34 : 56), top - 10, { size: 11, font: 'mono', color: C.muted, align: 'center' });
        CODE.forEach((c, i) => {
          const y = top + i * rh;
          V.box(g, 8, y + 2, w - 16, rh - 4, { r: 8, fill: V.alpha(C.ink, i % 2 ? 0.02 : 0.045) });
          V.wrap(g, c.t, 16, y + rh / 2 - (narrow && tw(g, c.t, 11.5) > nameW - 12 ? 3 : -4), nameW - 12, { size: 11.5, color: C.ink, lineH: 14, maxLines: 2 });
          for (let j = 0; j < 5; j++) {
            const cx = x0 + j * cwid + cwid / 2, cy = y + rh / 2, settled = T > 0.4 + j * 0.35 + i * 0.12;
            const r = Math.min(11, cwid * 0.28);
            const st = c.res[j];
            let fill;
            if (!settled) fill = V.alpha(Math.sin(t * 20 + i * 3 + j * 7) > 0 ? C.a : C.d, 0.35);
            else fill = st === 'p' ? V.alpha(C.a, 0.8) : st === 'f' ? V.alpha(RED(), 0.8) : V.alpha(C.muted, 0.25);
            g.beginPath(); g.arc(cx, cy, r, 0, Math.PI * 2); g.fillStyle = fill; g.fill();
            if (settled && st === '-') V.text(g, '—', cx, cy + 4, { size: 11, color: C.muted, align: 'center' });
          }
          const doneAll = T > 0.4 + 5 * 0.35 + 0.5;
          if (doneAll) {
            mark(g, w - (narrow ? 58 : 92), y + rh / 2, !!R[i], 7);
            V.text(g, sgn(A[i], 2), w - (narrow ? 34 : 56), y + rh / 2 + 4, { size: 11.5, font: 'mono', color: A[i] > 0 ? C.a : A[i] < 0 ? RED() : C.muted, align: 'center' });
            if (c.hack) chip(g, 16, y + rh - 20, 40, 16, 'hack', { fill: V.alpha(C.d, 0.2), stroke: C.d, size: 11 });
          }
        });
        const yL = top + 4 * rh + 16;
        V.wrap(g, S.strict ? 'strict: every F2P must run and pass, every P2P must pass' : 'naive: “no test failed” (a deleted test cannot fail)', 12, yL, w - 24, { size: 11.5, font: 'mono', color: S.strict ? C.a : C.d, lineH: 15, maxLines: 2 });
      }
      function drawPRM(g, w, h, T, C, narrow) {
        const p = prmP();
        const top = 20, sh = Math.min(56, (h - top - 90) / 4);
        const barX = narrow ? 12 : 220, barW = narrow ? w - 24 - 90 : w - barX - 190;
        V.text(g, '4-step solution · final answer 72 (correct)', 12, 14, { size: 12, color: C.ink, weight: 600, maxW: w - 24 });
        let run = 1, mn = 1;
        p.forEach((pi, i) => {
          const y = top + 12 + i * sh;
          const on = T > 0.4 + i * 0.7;
          if (on) { run *= pi; mn = Math.min(mn, pi); }
          if (narrow) {
            V.text(g, (i + 1) + '. ' + STEPS[i][0], 12, y + 12, { size: 11.5, color: C.ink, maxW: w - 24 });
            V.box(g, barX, y + 18, barW, 12, { r: 4, fill: V.alpha(C.ink, 0.06) });
            V.box(g, barX, y + 18, barW * pi, 12, { r: 4, fill: pi < 0.6 ? V.alpha(RED(), 0.7) : V.alpha(C.a, 0.7) });
            V.text(g, fmt(pi, 2), barX + barW + 8, y + 28, { size: 11.5, font: 'mono', color: C.ink });
          } else {
            V.text(g, (i + 1) + '. ' + STEPS[i][0], 12, y + sh / 2 + 4, { size: 12, color: C.ink, maxW: barX - 20 });
            V.box(g, barX, y + sh / 2 - 7, barW, 14, { r: 4, fill: V.alpha(C.ink, 0.06) });
            V.box(g, barX, y + sh / 2 - 7, barW * pi, 14, { r: 4, fill: pi < 0.6 ? V.alpha(RED(), 0.7) : V.alpha(C.a, 0.7) });
            V.text(g, 'P(step correct) = ' + fmt(pi, 2), barX + barW + 10, y + sh / 2 + 4, { size: 11.5, font: 'mono', color: C.ink });
          }
          if (on && T < 0.4 + i * 0.7 + 0.6) V.glow(g, barX + barW * pi, y + (narrow ? 24 : sh / 2), 18, C.d, 0.8);
        });
        const sc = S.agg === 'prod' ? run : mn;
        const yS = top + 12 + 4 * sh + 14;
        const gw2 = w - 24;
        V.text(g, (S.agg === 'prod' ? 'running product' : 'running minimum') + ' (PRM)', 12, yS, { size: 11.5, color: C.muted });
        V.box(g, 12, yS + 8, gw2, 18, { r: 5, fill: V.alpha(C.ink, 0.06) });
        V.box(g, 12, yS + 8, gw2 * sc, 18, { r: 5, fill: V.alpha(sc < 0.5 ? RED() : C.a, 0.75) });
        V.text(g, fmt(sc, 3), 12 + gw2 * sc + (sc > 0.85 ? -8 : 8), yS + 21, { size: 12, font: 'mono', weight: 600, color: C.ink, align: sc > 0.85 ? 'right' : 'left' });
        V.text(g, 'ORM (final answer only)', 12, yS + 46, { size: 11.5, color: C.muted });
        V.box(g, 12, yS + 54, gw2, 18, { r: 5, fill: V.alpha(C.b, 0.65) });
        V.text(g, '1.000', 12 + gw2 - 8, yS + 67, { size: 12, font: 'mono', weight: 600, color: C.ink, align: 'right' });
        V.wrap(g, (S.agg === 'prod' ? 'The PRM multiplies' : 'The PRM keeps the worst of') + ' the steps: one doubtful step drags down the whole solution. The ORM only checks the final 72 and gives 1 even with a weak step in the middle.', 12, yS + 96, w - 24, { size: 11.5, color: C.muted, lineH: 15, maxLines: 3 });
      }
      H = V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        let T = t - S.t0;
        const cyc = S.mode === 'prm' ? 6 : 9;
        if (T > cyc) { S.t0 = t; T = 0; }
        const narrow = w < 640;
        if (S.mode === 'mat') drawMath(g, w, h, T, C, narrow);
        else if (S.mode === 'cod') drawCode(g, w, h, T, C, narrow, t);
        else drawPRM(g, w, h, T, C, narrow);
      });
      restart();
    })();

    /* ===================================================== FIG 8 — R1 recipe */
    (function () {
      const fig = root.querySelector('#fig-r1'); if (!fig) return;
      adaptH(fig, 400, 580);
      const NODES = [
        { t: 'V3-Base', s: 'base model' },
        { t: 'R1-Zero', s: 'pure GRPO' },
        { t: '1 · cold start', s: 'SFT, long CoT' },
        { t: '2 · reasoning RL', s: 'GRPO + language' },
        { t: '3 · rejection + SFT', s: '800k → V3-Base' },
        { t: '4 · general RL', s: 'rules + RMs' },
        { t: 'DeepSeek-R1', s: 'final' },
        { t: 'distilled', s: '1.5B … 70B' },
      ];
      const EDGES = [[0, 1], [0, 2], [2, 3], [3, 4], [4, 5], [5, 6], [4, 7, 'd']];
      const HI = [[0], [1], [1], [2], [3], [4], [5, 6], [7]];
      const steps = [
        { t: 'The starting point', d: '<b>DeepSeek-V3-Base</b>, a model that is only pretrained. No reasoning demonstrations, no SFT.' },
        { t: 'R1-Zero: GRPO straight on the base model', d: '16 outputs per question, reward = <b>accuracy + format</b> (rules, no neural RM). lr 3·10⁻⁶, KL 0.001, temperature 1, reference swapped for the current policy every 400 steps.' },
        { t: 'What emerged (and what went wrong)', d: 'AIME 2024 pass@1: <b>15.6% → 77.9%</b> in 10,400 steps (v2 of the paper; v1, from January, reports 71.0% from an earlier run). Answers grew from hundreds to thousands of tokens; the limit went from 32,768 to 65,536 at step 8.2k. Flaws: readability and <b>language mixing</b>.' },
        { t: 'Stage 1: cold start', d: '“Thousands” of long, <b>readable</b> CoT examples, partly reformatted from R1-Zero and refined by humans and by DeepSeek-V3.' },
        { t: 'Stage 2: reasoning RL', d: 'GRPO with rule-based rewards + <b>language consistency</b> (proportion of target-language words in the CoT). v2: clip ε = 10, 16 outputs, up to 32,768 tokens.' },
        { t: 'Stage 3: rejection sampling + SFT', d: '~600k reasoning samples (correct only, filtered) + ~200k general = <b>~800k</b>. <b>V3-Base</b> is fine-tuned again, for 2 epochs.' },
        { t: 'Stage 4: RL for all scenarios', d: 'Rules + preference RMs (66k and 106k pairs). 1,700 steps, temperature 0.7; the RMs only come in for the <b>last 400 steps</b>, because more steps led to reward hacking.' },
        { t: 'Distillation', d: 'Six dense models fine-tuned with SFT alone on the ~800k examples. At 32B: <b>72.6</b> on AIME 2024 versus <b>47.0</b> for pure RL (R1-Zero-Qwen-32B).' },
      ];
      let H = null;
      const st = UI.stepper(fig, steps, { autoplay: 7000, onStep: () => H && H.redraw() });
      const rngD = V.rng(91);
      const PAT = Array.from({ length: 64 }, () => (rngD() < 0.4 ? 1 : 0));
      function layout(w, narrow) {
        if (!narrow) {
          const cw = (w - 24) / 5, nw = cw - 14, y0 = 14, y1 = 78;
          const pos = [[0, y0], [1, y0], [0, y1], [1, y1], [2, y1], [3, y1], [4, y1], [3, y0]];
          return { nw, nh: 46, P: pos.map(([c, y]) => [12 + c * cw + 7, y]), H: 134 };
        }
        const cw = (w - 24) / 3, nw = cw - 10, rows = [12, 72, 132];
        const pos = [[0, 0], [1, 0], [0, 1], [1, 1], [2, 1], [2, 2], [1, 2], [2, 0]];
        return { nw, nh: 46, P: pos.map(([c, r]) => [12 + c * cw + 5, rows[r]]), H: 190 };
      }
      H = V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const narrow = w < 640;
        const L = layout(w, narrow);
        const i = st.i, el = st.elapsed();
        const hi = HI[i];
        const ctr = (k) => [L.P[k][0] + L.nw / 2, L.P[k][1] + L.nh / 2];
        // edges
        EDGES.forEach(([a, b, d]) => {
          const [x1, y1] = ctr(a), [x2, y2] = ctr(b);
          const ang = Math.atan2(y2 - y1, x2 - x1);
          const sx = x1 + Math.cos(ang) * (Math.abs(Math.cos(ang)) > 0.5 ? L.nw / 2 : L.nh / 2), sy = y1 + Math.sin(ang) * (Math.abs(Math.cos(ang)) > 0.5 ? L.nh * 0.2 : L.nh / 2);
          const ex = x2 - Math.cos(ang) * (Math.abs(Math.cos(ang)) > 0.5 ? L.nw / 2 : L.nh / 2), ey = y2 - Math.sin(ang) * (Math.abs(Math.cos(ang)) > 0.5 ? L.nh * 0.2 : L.nh / 2);
          const act = hi.includes(b);
          V.arrow(g, sx, sy, ex, ey, { color: act ? C.a : V.alpha(C.ink, 0.3), lw: act ? 2.2 : 1.3, dash: d ? [4, 4] : null, head: 7 });
          if (act) for (let k = 0; k < 3; k++) { const f = ((t * 0.7 + k / 3) % 1); const [px2, py2] = V.along(sx, sy, ex, ey, f); V.dot(g, px2, py2, 3, C.d); }
        });
        NODES.forEach((n, k) => {
          const [x, y] = L.P[k], on = hi.includes(k);
          V.box(g, x, y, L.nw, L.nh, { r: 9, fill: on ? V.alpha(C.a, 0.22) : V.alpha(C.ink, 0.04), stroke: on ? C.a : V.alpha(C.ink, 0.25), lw: on ? 2 : 1 });
          if (on) V.glow(g, x + L.nw / 2, y + L.nh / 2, L.nw * 0.45, C.a, 0.18 + 0.1 * Math.sin(t * 3));
          V.text(g, n.t, x + L.nw / 2, y + 19, { size: 12, weight: 600, color: C.ink, align: 'center', maxW: L.nw - 8 });
          V.text(g, n.s, x + L.nw / 2, y + 35, { size: 11, color: C.muted, align: 'center', maxW: L.nw - 8 });
        });
        // detail panel
        const D = { x: 12, y: L.H + 8, w: w - 24, h: h - L.H - 16 };
        V.box(g, D.x, D.y, D.w, D.h, { r: 10, fill: V.alpha(C.ink, 0.025), stroke: V.alpha(C.ink, 0.12), lw: 1 });
        const ap = V.clamp(el / 0.8, 0, 1);
        const cx = D.x + D.w / 2, cy = D.y + D.h / 2;
        if (i === 0) {
          const s = Math.min(D.h * 0.55, 120);
          V.box(g, cx - s, cy - s / 2, 2 * s, s, { r: 14, fill: V.alpha(C.b, 0.18), stroke: C.b, lw: 1.6 });
          V.text(g, 'DeepSeek-V3-Base', cx, cy - 4, { size: 15, weight: 700, color: C.ink, align: 'center' });
          V.text(g, 'pretrained · no reasoning SFT', cx, cy + 16, { size: 11.5, color: C.muted, align: 'center', maxW: 2 * s - 12 });
        } else if (i === 1) {
          V.text(g, 'one question → 16 outputs → accuracy + format', D.x + 12, D.y + 20, { size: 12, color: C.ink, maxW: D.w - 24 });
          const n = 16, cols = 8, sz = Math.min(30, (D.w - 40) / cols - 8);
          const rnd = Math.floor(t / 2.2);
          let ok = 0;
          for (let k = 0; k < n; k++) {
            const r = PAT[(k + rnd * 5) % PAT.length]; ok += r;
            const x = D.x + 20 + (k % cols) * (sz + 8) + sz / 2, y = D.y + 52 + Math.floor(k / cols) * (sz + 14) + sz / 2;
            const show = (t % 2.2) > 0.2 + k * 0.06;
            blob(g, x, y, sz, show ? (r ? C.a : C.c) : C.b, { mood: show && !r ? 'sad' : undefined });
          }
          V.text(g, 'correct in group: ' + ok + '/16 → group advantages', D.x + 12, D.y + D.h - 14, { size: 11.5, font: 'mono', color: C.d, maxW: D.w - 24 });
        } else if (i === 2) {
          const P = { x: D.x + 50, y: D.y + 22, w: D.w - 70, h: D.h - 56 };
          V.line(g, P.x, P.y + P.h, P.x + P.w, P.y + P.h, { color: V.alpha(C.ink, 0.4), lw: 1 });
          V.line(g, P.x, P.y, P.x, P.y + P.h, { color: V.alpha(C.ink, 0.4), lw: 1 });
          const px2 = (s) => P.x + (s / 10400) * P.w, py2 = (v) => P.y + P.h - (v / 100) * P.h;
          [0, 50, 100].forEach((v) => V.text(g, v + '%', P.x - 6, py2(v) + 4, { size: 11, font: 'mono', color: C.muted, align: 'right' }));
          V.text(g, 'step 0', P.x, P.y + P.h + 14, { size: 11, font: 'mono', color: C.muted });
          V.text(g, '10,400', P.x + P.w, P.y + P.h + 14, { size: 11, font: 'mono', color: C.muted, align: 'right' });
          const x82 = px2(8200);
          V.line(g, x82, P.y, x82, P.y + P.h, { color: V.alpha(C.d, 0.7), lw: 1.4, dash: [4, 4] });
          V.text(g, '8.2k: limit 32,768 → 65,536', x82 - 4, P.y + 12, { size: 11, color: C.d, align: 'right', maxW: x82 - P.x - 8 });
          // v2 curve (the one on the 10,400-step axis): only the endpoints are published
          g.setLineDash([5, 5]); g.beginPath();
          const NN = 60;
          for (let k = 0; k <= NN * ap; k++) { const f = k / NN, s = f * 10400, v = 15.6 + (77.9 - 15.6) * (1 - Math.exp(-3.2 * f)) / (1 - Math.exp(-3.2)); k ? g.lineTo(px2(s), py2(v)) : g.moveTo(px2(s), py2(v)); }
          g.strokeStyle = V.alpha(C.a, 0.8); g.lineWidth = 2; g.stroke(); g.setLineDash([]);
          V.dot(g, px2(0), py2(15.6), 5, C.a); V.text(g, '15.6%', px2(0) + 8, py2(15.6) + 4, { size: 11.5, font: 'mono', color: C.ink });
          // labels on a plate: the dashed 8.2k line passes behind them
          tag(g, 'AIME 2024 pass@1 · schematic line', P.x + 6, P.y + P.h - 8, { size: 11, color: C.muted, maxW: P.w - 12 });
          if (ap > 0.98) {
            V.dot(g, px2(10400), py2(77.9), 5, C.a);
            tag(g, '77.9% (v2)', px2(10400), py2(77.9) - 12, { size: 11.5, font: 'mono', color: C.ink, align: 'right' });
            // v1 is a different run, with no published curve on this axis: a side note, not a point
            const note = narrow ? ['paper v1 (Jan 2025): 71.0%,', 'from an earlier run'] : ['paper v1 (Jan 2025): 71.0%, from an earlier run'];
            note.forEach((s2, k) => tag(g, s2, P.x + P.w - 4, py2(40) + k * 15, { size: 11, color: C.muted, align: 'right', maxW: P.w - 12 }));
          }
        } else if (i === 3) {
          const n = narrow ? 4 : 8, cw2 = Math.min(76, (D.w - 40) / n - 8);
          for (let k = 0; k < n; k++) {
            const f = V.clamp((el - k * 0.15) / 0.8, 0, 1), x = D.x + 20 + k * (cw2 + 8), y = V.lerp(D.y + D.h - 50, D.y + 30, V.ease.out(f));
            V.box(g, x, y, cw2, 38, { r: 6, fill: V.alpha(C.b, 0.18), stroke: C.b, lw: 1 });
            V.text(g, '<think>…', x + cw2 / 2, y + 16, { size: 11, font: 'mono', color: C.ink, align: 'center', maxW: cw2 - 6 });
            V.text(g, 'readable', x + cw2 / 2, y + 30, { size: 11, color: C.muted, align: 'center', maxW: cw2 - 6 });
          }
          V.text(g, '“thousands” of long, readable CoTs → starting SFT', D.x + 12, D.y + D.h - 10, { size: 11.5, color: C.ink, maxW: D.w - 24 });
        } else if (i === 4) {
          const bw2 = D.w - 60, x = D.x + 30, oy = Math.max(0, (D.h - 150) / 2);
          V.text(g, 'reward = accuracy + language consistency', x, D.y + oy + 22, { size: 12.5, weight: 600, color: C.ink, maxW: bw2 });
          const f = 0.55 + 0.45 * V.ease.inOut(V.clamp(el / 3, 0, 1));
          // CoT tokens: mixed at the start, a single language at the end
          const nT = narrow ? 12 : 22, tw2 = (bw2 - (nT - 1) * 4) / nT;
          for (let k = 0; k < nT; k++) { const mixed = PAT[(k * 7) % PAT.length] && k / nT >= (f - 0.55) / 0.45; V.box(g, x + k * (tw2 + 4), D.y + oy + 50, tw2, 16, { r: 4, fill: V.alpha(mixed ? C.c : C.a, 0.7) }); }
          V.box(g, x, D.y + oy + 78, bw2, 20, { r: 6, fill: V.alpha(C.ink, 0.06) });
          V.box(g, x, D.y + oy + 78, bw2 * f, 20, { r: 6, fill: V.alpha(C.a, 0.7) });
          V.text(g, 'CoT tokens (pink = another language)', x, D.y + oy + 42, { size: 11, color: C.muted, maxW: bw2 });
          V.text(g, 'target-language words in the CoT: ' + Math.round(f * 100) + '% (illustrative)', x, D.y + oy + 116, { size: 11.5, font: 'mono', color: C.muted, maxW: bw2 });
          V.text(g, 'v2: clip ε = 10 · 16 outputs · up to 32,768 tokens · temperature 1', x, D.y + oy + 138, { size: 11.5, font: 'mono', color: C.ink, maxW: bw2 });
        } else if (i === 5) {
          const x = D.x + 30, bw2 = D.w - 60, f = V.ease.out(V.clamp(el / 1.2, 0, 1)), oy = Math.max(0, (D.h - 170) / 2);
          // rejection sampling: correct samples go in, wrong ones fall out
          V.text(g, 'samples from the stage 2 checkpoint', x, D.y + oy + 16, { size: 11.5, color: C.muted, maxW: bw2 });
          for (let k = 0; k < 16; k++) {
            const ph = (t * 0.35 + k / 16) % 1, ok = PAT[(k * 3) % PAT.length] === 0;
            let px3 = x + ph * bw2 * 0.7, py3 = D.y + oy + 34, al = 1;
            if (!ok && ph > 0.45) { py3 += (ph - 0.45) * 90; al = Math.max(0, 1 - (ph - 0.45) * 2.4); }
            g.globalAlpha = al; blob(g, px3, py3, 13, ok ? C.a : C.c, { mood: ok ? undefined : 'sad' }); g.globalAlpha = 1;
          }
          const by = D.y + oy + 64;
          V.box(g, x, by, bw2 * 0.75 * f, 30, { r: 6, fill: V.alpha(C.a, 0.6), stroke: C.a });
          V.box(g, x + bw2 * 0.75 * f, by, bw2 * 0.25 * f, 30, { r: 6, fill: V.alpha(C.b, 0.6), stroke: C.b });
          if (f > 0.9) { V.text(g, '~600k reasoning', x + bw2 * 0.375, by + 20, { size: 11.5, color: C.ink, align: 'center', maxW: bw2 * 0.7 }); V.text(g, '~200k', x + bw2 * 0.875, by + 20, { size: 11.5, color: C.ink, align: 'center', maxW: bw2 * 0.24 }); }
          V.text(g, '= ~800k examples → fine-tune V3-Base for 2 epochs', x, by + 56, { size: 12, color: C.ink, maxW: bw2 });
          V.text(g, 'correct only; no mixed languages, long paragraphs or code blocks', x, by + 76, { size: 11, color: C.muted, maxW: bw2 });
        } else if (i === 6) {
          const x = D.x + 30, bw2 = D.w - 60, oy = Math.max(0, (D.h - 130) / 2);
          V.box(g, x, D.y + oy + 44, bw2, 26, { r: 6, fill: V.alpha(C.a, 0.25), stroke: C.a });
          const x13 = x + bw2 * (1300 / 1700);
          V.box(g, x13, D.y + oy + 44, x + bw2 - x13, 26, { r: 6, fill: V.alpha(C.d, 0.55), stroke: C.d });
          const cur = x + bw2 * ((el * 0.25) % 1);
          V.line(g, cur, D.y + oy + 38, cur, D.y + oy + 76, { color: C.ink, lw: 2 });
          V.text(g, 'step 0', x, D.y + oy + 90, { size: 11, font: 'mono', color: C.muted });
          V.text(g, '1,700', x + bw2, D.y + oy + 90, { size: 11, font: 'mono', color: C.muted, align: 'right' });
          V.text(g, 'last 400: preference RMs', Math.min(x13 + (x + bw2 - x13) / 2, x + bw2 - 90), D.y + oy + 32, { size: 11, color: C.d, align: 'center', maxW: 180 });
          V.text(g, 'rules for reasoning throughout · temperature 0.7', x, D.y + oy + 116, { size: 11.5, color: C.ink, maxW: bw2 });
        } else {
          const items = [['R1-Zero-Qwen-32B (RL)', 47.0, C.c], ['QwQ-32B-Preview', 50.0, C.b], ['R1-Distill-Qwen-32B', 72.6, C.a]];
          const lw2 = narrow ? 0 : 190, x = D.x + 16 + lw2, bw2 = D.w - lw2 - 80;
          items.forEach(([nm, v, c], k) => {
            const y = D.y + 22 + k * (narrow ? 50 : 42), f = V.ease.out(V.clamp((el - k * 0.2) / 1, 0, 1));
            if (narrow) V.text(g, nm, D.x + 16, y + 8, { size: 11.5, color: C.ink }); else V.text(g, nm, D.x + 16, y + 18, { size: 11.5, color: C.ink, maxW: lw2 - 8 });
            const by = narrow ? y + 14 : y + 6;
            V.box(g, x - (narrow ? 0 : 0), by, (bw2 * v / 80) * f, 18, { r: 5, fill: V.alpha(c, 0.65), stroke: c });
            V.text(g, fmt(v * f, 1), x + (bw2 * v / 80) * f + 6, by + 14, { size: 11.5, font: 'mono', color: C.ink });
          });
          V.text(g, 'AIME 2024 pass@1 · distilling > pure RL at 32B', D.x + 16, D.y + D.h - 10, { size: 11.5, color: C.muted, maxW: D.w - 32 });
        }
      });
    })();

    /* ===================================================== FIG 9 — entropy and length */
    (function () {
      const fig = root.querySelector('#fig-entropia'); if (!fig) return;
      adaptH(fig, 400, 470);
      const TOK = ['So', 'Then', 'Thus', 'Wait', 'Hmm'];
      const Z0 = [2.4, 1.2, 0.5, -0.3, -1.0];
      const ent = (p) => -p.reduce((s, x) => s + (x > 0 ? x * Math.log(x) : 0), 0);
      const S = { mode: 'ent', who: 'fav', upd: 'pg', z: Z0.slice(), Hs: [ent(V.softmax(Z0))], run: true, auto: true, hold: 0, acc: 0, cov: 0, Lmax: 20480, Lc: 4096, y: 18432 };
      let H = null;
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'mode', options: [{ v: 'ent', label: 'entropy' }, { v: 'len', label: 'length' }], value: 'ent', onChange: (v) => { S.mode = v; gE.show(v === 'ent'); gL.show(v === 'len'); if (H) H.redraw(); } });
      const gE = UI.group(ctl);
      UI.seg(gE.el, { label: 'rewarded', options: [{ v: 'fav', label: 'the favorite (So)' }, { v: 'rare', label: 'a rare one (Wait)' }], value: 'fav', onChange: (v) => { S.who = v; reset(); } });
      UI.seg(gE.el, { label: 'update', options: [{ v: 'pg', label: 'PG: Δz ∝ π·A' }, { v: 'npg', label: 'natural: Δz ∝ A' }], value: 'pg', onChange: (v) => { S.upd = v; reset(); } });
      const bRun = UI.button(gE.el, '⏸ pause', () => { S.auto = false; if (!S.run && S.Hs.length > 90) reset(); S.run = !S.run; bRun.innerHTML = S.run ? '⏸ pause' : '▶ train'; if (H) H.redraw(); }, { primary: true });
      UI.button(gE.el, '↺', () => { reset(); });
      const rH = UI.readout(gE.el, 'entropy H', '—'), rC = UI.readout(gE.el, 'Cov(log π, Δz)', '—'), rN = UI.readout(gE.el, 'steps', '0');
      const gL = UI.group(ctl);
      UI.slider(gL.el, { label: 'L_max', min: 8192, max: 32768, step: 1024, value: 20480, fmt: thou, onInput: (v) => { S.Lmax = v; updL(); } });
      UI.slider(gL.el, { label: 'L_cache', min: 1024, max: 8192, step: 512, value: 4096, fmt: thou, onInput: (v) => { S.Lc = v; updL(); } });
      UI.slider(gL.el, { label: '|y| (tokens)', min: 0, max: 36000, step: 256, value: 18432, fmt: thou, onInput: (v) => { S.y = v; updL(); } });
      const rL = UI.readout(gL.el, 'R_length', '—');
      gL.show(false);
      nocase(ctl);
      const Rlen = (y) => { const Lc = Math.min(S.Lc, S.Lmax), a = S.Lmax - Lc; return y <= a ? 0 : y <= S.Lmax ? (a - y) / Lc : -1; };
      function updL() { const r = Rlen(S.y); rL.set(fmt(r, 3), r === 0 ? 'ok' : r > -1 ? 'warn' : 'bad'); if (H) H.redraw(); }
      function reset() { S.z = Z0.slice(); S.Hs = [ent(V.softmax(Z0))]; S.cov = 0; S.acc = 0; updE(); }
      function updE() { rH.set(fmt(S.Hs[S.Hs.length - 1], 3) + ' nats'); rC.set(sgn(S.cov, 3), S.cov > 0 ? 'warn' : S.cov < 0 ? 'ok' : null); rN.set(String(S.Hs.length - 1)); if (H) H.redraw(); }
      function stepRL() {
        const p = V.softmax(S.z), k = S.who === 'fav' ? 0 : 3;
        const A = p.map((pi, a) => (a === k ? 1 : 0) - p[k]);
        const dz = A.map((a, i) => (S.upd === 'pg' ? 1.6 * p[i] * a : 0.35 * a));
        const lp = p.map((x) => Math.log(x)), El = p.reduce((s, x, i) => s + x * lp[i], 0), Ed = p.reduce((s, x, i) => s + x * dz[i], 0);
        S.cov = p.reduce((s, x, i) => s + x * (lp[i] - El) * (dz[i] - Ed), 0);
        S.z = S.z.map((z, i) => z + dz[i]);
        S.Hs.push(ent(V.softmax(S.z)));
        if (S.Hs.length > 90 || S.Hs[S.Hs.length - 1] < 0.02) { S.run = false; S.hold = 0; bRun.innerHTML = '▶ train'; }
        updE();
      }
      function drawEnt(g, w, h, dt, C, narrow) {
        if (S.run && !LA.paused) { S.acc += dt; while (S.acc > 0.09 && S.run) { S.acc -= 0.09; stepRL(); } }
        else if (S.auto && !S.run && !LA.paused) { S.hold += dt; if (S.hold > 2.5) { reset(); S.run = true; bRun.innerHTML = '⏸ pause'; } }
        const p = V.softmax(S.z), k = S.who === 'fav' ? 0 : 3;
        const topH = h * 0.52;
        const left = 16, bw = Math.min(70, (w - 40) / 5 - 14), cw = (w - 2 * left) / 5;
        const base = topH - 26, bh = topH - 64;
        V.text(g, narrow ? 'bars: distribution · curve: entropy' : 'distribution at one position (bars) and entropy over training (curve)', left, 16, { size: 11.5, color: C.muted, maxW: w - 2 * left });
        p.forEach((pi, i) => {
          const x = left + i * cw + cw / 2;
          const col = i === k ? C.a : C.b;
          V.box(g, x - bw / 2, base - pi * bh, bw, Math.max(1.5, pi * bh), { r: 4, fill: V.alpha(col, 0.55), stroke: col, lw: 1.2 });
          V.text(g, fmt(pi, 3), x, base - pi * bh - 6, { size: 11, font: 'mono', color: C.ink, align: 'center' });
          V.text(g, TOK[i], x, base + 15, { size: 11.5, color: C.ink, align: 'center', weight: i === k ? 700 : 400 });
          if (i === k) star(g, x + bw / 2 - 2, base - pi * bh - 20, 6, C.d);
        });
        const P = { x: 46, y: topH + 12, w: w - 62, h: h - topH - 36 };
        V.line(g, P.x, P.y + P.h, P.x + P.w, P.y + P.h, { color: V.alpha(C.ink, 0.4), lw: 1 });
        V.line(g, P.x, P.y, P.x, P.y + P.h, { color: V.alpha(C.ink, 0.4), lw: 1 });
        const Hm = Math.log(5);
        const px = (s) => P.x + (s / 90) * P.w, py = (v) => P.y + P.h - (v / Hm) * P.h;
        V.text(g, fmt(Hm, 2), P.x - 6, py(Hm) + 4, { size: 11, font: 'mono', color: C.muted, align: 'right' });
        V.text(g, '0', P.x - 6, py(0) + 4, { size: 11, font: 'mono', color: C.muted, align: 'right' });
        V.text(g, 'RL steps', P.x + P.w, P.y + P.h + 14, { size: 11, color: C.muted, align: 'right' });
        V.text(g, 'H (nats)', P.x + 6, P.y + 10, { size: 11, color: C.muted });
        g.beginPath(); S.Hs.forEach((v, s) => (s ? g.lineTo(px(s), py(v)) : g.moveTo(px(s), py(v)))); g.strokeStyle = C.d; g.lineWidth = 2.4; g.stroke();
        const last = S.Hs.length - 1; V.dot(g, px(last), py(S.Hs[last]), 4.5, C.d);
        if (!S.run && last === 0) V.text(g, 'press ▶ train', P.x + P.w / 2, P.y + P.h / 2, { size: 12, color: C.muted, align: 'center' });
      }
      function drawLen(g, w, h, t, C) {
        const P = { x: 52, y: 40, w: w - 70, h: h * 0.52 };
        const YM = 36000, px = (y) => P.x + (y / YM) * P.w, pv = (r) => P.y + ((0 - r) / 1.1) * P.h + 6;
        V.text(g, 'R_length(|y|): DAPO’s soft punishment', 12, 18, { size: 12, color: C.ink, weight: 600, maxW: w - 24 });
        V.line(g, P.x, pv(0), P.x + P.w, pv(0), { color: V.alpha(C.ink, 0.4), lw: 1 });
        V.line(g, P.x, pv(-1), P.x + P.w, pv(-1), { color: V.alpha(C.ink, 0.12), lw: 1, dash: [3, 4] });
        V.text(g, '0', P.x - 6, pv(0) + 4, { size: 11, font: 'mono', color: C.muted, align: 'right' });
        V.text(g, '−1', P.x - 6, pv(-1) + 4, { size: 11, font: 'mono', color: C.muted, align: 'right' });
        const Lc = Math.min(S.Lc, S.Lmax), a = S.Lmax - Lc;
        V.box(g, px(a), P.y, px(S.Lmax) - px(a), P.h + 12, { r: 0, fill: V.alpha(C.d, 0.12) });
        V.box(g, px(S.Lmax), P.y, P.x + P.w - px(S.Lmax), P.h + 12, { r: 0, fill: V.alpha(RED(), 0.08) });
        g.beginPath(); for (let i = 0; i <= 300; i++) { const y = (YM * i) / 300, X = px(y), Y = pv(Rlen(y)); i ? g.lineTo(X, Y) : g.moveTo(X, Y); } g.strokeStyle = C.d; g.lineWidth = 2.6; g.stroke();
        [[a, thou(a)], [S.Lmax, thou(S.Lmax)]].forEach(([v, s], i) => V.text(g, s, px(v), P.y + P.h + 28 + i * 14, { size: 11, font: 'mono', color: C.muted, align: 'center' }));
        const r = Rlen(S.y);
        V.dot(g, px(S.y), pv(r), 6, r === 0 ? C.a : r > -1 ? C.d : RED());
        V.text(g, sgn(r, 2), px(S.y) + (S.y > YM * 0.8 ? -10 : 10), pv(r) - 8, { size: 12, font: 'mono', weight: 600, color: C.ink, align: S.y > YM * 0.8 ? 'right' : 'left' });
        // answer growing
        const yb = P.y + P.h + 62, f = ((t * 0.35) % 1.15), cur = Math.min(S.y, f * S.y);
        V.text(g, 'an answer being generated', 12, yb - 8, { size: 11, color: C.muted });
        V.box(g, P.x, yb, P.w, 16, { r: 5, fill: V.alpha(C.ink, 0.05) });
        const rc = Rlen(cur);
        V.box(g, P.x, yb, Math.max(1, px(cur) - P.x), 16, { r: 5, fill: V.alpha(rc === 0 ? C.a : rc > -1 ? C.d : RED(), 0.7) });
        V.text(g, thou(cur) + ' tokens · ' + sgn(rc, 2), P.x, yb + 34, { size: 11.5, font: 'mono', color: C.ink });
      }
      H = V.fig(fig, (g, w, h, t, dt) => {
        const C = V.C();
        if (S.mode === 'ent') drawEnt(g, w, h, dt, C, w < 640); else drawLen(g, w, h, t, C);
      });
      updE(); updL();
    })();

    /* ===================================================== FIG 10 — mismatch */
    (function () {
      const fig = root.querySelector('#fig-descompasso'); if (!fig) return;
      adaptH(fig, 440, 540);
      const TOK = [
        { w: 'Adding', p: 0.25, rho: 1.2 }, { w: '48', p: 0.52, rho: 0.9615 }, { w: 'Hmm', p: 0.004, rho: 5.0 }, { w: '+', p: 0.91, rho: 0.989 },
        { w: '24', p: 0.5, rho: 0.98 }, { w: 'Wait', p: 0.05, rho: 1.6 }, { w: '=', p: 0.02, rho: 2.5 }, { w: '72', p: 0.6, rho: 0.45 },
      ];
      const L = 6, E = 8;
      const rr = V.rng(2026);
      const BASE = [], NOISE = [];
      const delta = [0.6, 0.04, 0.7, 0.02, 0.05, 0.5];
      for (let l = 0; l < L; l++) {
        const perm = Array.from({ length: E }, (_, i) => i).sort(() => rr() - 0.5);
        const z = new Array(E).fill(0);
        z[perm[0]] = 2.0; z[perm[1]] = 1.0 + delta[l]; z[perm[2]] = 1.0;
        for (let k = 3; k < E; k++) z[perm[k]] = -0.2 - rr() * 1.2;
        BASE.push(z); NOISE.push(Array.from({ length: E }, () => rr.normal() * 0.12));
      }
      const S = { mode: 'prob', corr: 'tis', sig: 1, keep: false };
      let H = null;
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'mode', options: [{ v: 'prob', label: 'probabilities' }, { v: 'route', label: 'MoE routing' }], value: 'prob', onChange: (v) => { S.mode = v; gP.show(v === 'prob'); gR.show(v === 'route'); upd(); } });
      UI.slider(ctl, { label: 'mismatch', min: 0, max: 1.5, step: 0.05, value: 1, fmt: (v) => '×' + fmt(v, 2), onInput: (v) => { S.sig = v; upd(); } });
      const gP = UI.group(ctl);
      UI.seg(gP.el, { label: 'correction', options: [{ v: 'none', label: 'ignore ρ' }, { v: 'seq', label: 'sequence IS' }, { v: 'tis', label: 'TIS (C = 2)' }, { v: 'ice', label: 'IcePop (β = 2)' }], value: 'tis', onChange: (v) => { S.corr = v; upd(); } });
      const gR = UI.group(ctl);
      UI.toggle(gR.el, { label: 'Keep Routing / R3 (replays the rollout routing)', value: false, onChange: (v) => { S.keep = v; upd(); } });
      gR.show(false);
      const rA = UI.readout(ctl, 'Π ρ (sequence)', '—'), rB = UI.readout(ctl, 'zeroed tokens', '—');
      nocase(ctl);
      const lab = (r, s) => { const sp = r.el.querySelector('span'); if (sp) sp.textContent = s; };
      const rhos = () => TOK.map((q) => Math.exp(S.sig * Math.log(q.rho)));
      function weights() {
        const r = rhos(), prod = r.reduce((a, b) => a * b, 1);
        return r.map((x) => (S.corr === 'none' ? 1 : S.corr === 'seq' ? prod : S.corr === 'tis' ? Math.min(x, 2) : x >= 0.5 && x <= 2 ? x : 0));
      }
      const top2 = (z) => z.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]).slice(0, 2).map((x) => x[1]).sort((a, b) => a - b);
      function routes() {
        const inf = BASE.map((z, l) => top2(z.map((v, e) => v + S.sig * NOISE[l][e] * 1.6)));
        const tr = S.keep ? inf.map((x) => x.slice()) : BASE.map((z) => top2(z));
        const diff = inf.map((a, l) => a[0] !== tr[l][0] || a[1] !== tr[l][1]);
        return { inf, tr, diff };
      }
      function upd() {
        if (S.mode === 'prob') {
          const r = rhos(), wv = weights(), prod = r.reduce((a, b) => a * b, 1);
          lab(rA, 'Π ρ (sequence)'); lab(rB, 'zeroed tokens');
          rA.set(fmt(prod, 2), prod > 3 || prod < 0.33 ? 'bad' : 'ok');
          const z = wv.filter((x) => x === 0).length; rB.set(z + ' / 8', z ? 'warn' : 'ok');
        } else {
          const R = routes(), n = R.diff.filter(Boolean).length;
          lab(rA, 'diverging layers'); lab(rB, 'routing');
          rA.set(n + ' / ' + L, n ? 'bad' : 'ok'); rB.set(S.keep ? 'replayed from rollout' : 'recomputed in training', S.keep ? 'ok' : 'warn');
        }
        if (H) H.redraw();
      }
      function engines(g, w, t, C, narrow) {
        const ew = narrow ? 120 : 190, eh = 46, y = 8;
        V.box(g, 10, y, ew, eh, { r: 10, fill: V.alpha(C.b, 0.16), stroke: C.b, lw: 1.4 });
        V.text(g, 'rollout engine', 10 + ew / 2, y + 19, { size: 12, weight: 600, color: C.ink, align: 'center', maxW: ew - 8 });
        V.text(g, narrow ? 'vLLM · FP8' : 'vLLM / SGLang · FP8', 10 + ew / 2, y + 36, { size: 11, color: C.muted, align: 'center', maxW: ew - 8 });
        V.box(g, w - 10 - ew, y, ew, eh, { r: 10, fill: V.alpha(C.a, 0.14), stroke: C.a, lw: 1.4 });
        V.text(g, 'training engine', w - 10 - ew / 2, y + 19, { size: 12, weight: 600, color: C.ink, align: 'center', maxW: ew - 8 });
        V.text(g, narrow ? 'Megatron · BF16' : 'Megatron / FSDP · BF16', w - 10 - ew / 2, y + 36, { size: 11, color: C.muted, align: 'center', maxW: ew - 8 });
        if (narrow) {
          V.text(g, 'same weights θ', w / 2, y + eh / 2 + 4, { size: 11, color: C.muted, align: 'center', maxW: w - 2 * ew - 28 });
          const yc = y + eh + 8;
          V.line(g, 20, yc + 13, w - 20, yc + 13, { color: V.alpha(C.ink, 0.25), lw: 1.2, dash: [4, 4] });
          return { x1: 20, x2: w - 30, yc, bottom: y + eh + 26 };
        }
        const x1 = 16 + ew, x2 = w - 16 - ew;
        V.line(g, x1, y + eh / 2, x2, y + eh / 2, { color: V.alpha(C.ink, 0.25), lw: 1.2, dash: [4, 4] });
        V.text(g, 'same weights θ', (x1 + x2) / 2, y + eh / 2 - 8, { size: 11, color: C.muted, align: 'center', maxW: x2 - x1 - 8 });
        return { x1, x2, yc: y + eh / 2, bottom: y + eh };
      }
      function drawProb(g, w, h, t, C, narrow) {
        const Eg = engines(g, w, t, C, narrow);
        const r = rhos(), wv = weights();
        // travelling tokens
        const n = TOK.length;
        for (let i = 0; i < n; i++) {
          const f = ((t * 0.18 + i / n) % 1), x = V.lerp(Eg.x1, Eg.x2, f);
          const zero = wv[i] === 0, clipped = S.corr === 'tis' && r[i] > 2;
          const col = zero ? C.muted : clipped ? C.d : C.a;
          V.box(g, x - 7, Eg.yc + 6, 14, 14, { r: 4, fill: V.alpha(col, zero ? 0.3 : 0.75), stroke: col, lw: 1 });
        }
        const top = Eg.bottom + 28;
        const cw = (w - 56) / n, x0 = 46;
        const rowP = narrow ? 0 : 30;
        // tokens + probs
        TOK.forEach((q, i) => {
          const x = x0 + i * cw + cw / 2;
          chip(g, x - cw / 2 + 3, top, cw - 6, 24, q.w, { fill: V.alpha(C.b, 0.12), stroke: V.alpha(C.b, 0.5), size: 11.5 });
          if (!narrow) {
            V.text(g, 'inf ' + fmt(q.p, q.p < 0.01 ? 3 : 2), x, top + 40, { size: 11, font: 'mono', color: C.muted, align: 'center', maxW: cw - 4 });
            V.text(g, 'tr ' + fmt(q.p * r[i], q.p * r[i] < 0.01 ? 3 : 2), x, top + 54, { size: 11, font: 'mono', color: C.muted, align: 'center', maxW: cw - 4 });
          }
        });
        // ρ bars (log)
        const bTop = top + 34 + rowP, bH = narrow ? (h - bTop) * 0.5 : (h - bTop) * 0.52;
        const cy = bTop + bH / 2, LR = Math.log(6);
        const sy = (x) => cy - (Math.log(x) / LR) * (bH / 2);
        V.text(g, 'ρ', 18, cy + 4, { size: 13, font: 'mono', color: C.ink, align: 'center' });
        if (S.corr === 'ice') V.box(g, x0, sy(2), w - x0 - 10, sy(0.5) - sy(2), { r: 0, fill: V.alpha(C.a, 0.08) });
        V.line(g, x0, cy, w - 10, cy, { color: V.alpha(C.ink, 0.45), lw: 1.2 });
        [[2, '2'], [0.5, '0.5']].forEach(([v, s]) => { V.line(g, x0, sy(v), w - 10, sy(v), { color: V.alpha(C.ink, 0.25), lw: 1, dash: [3, 4] }); V.text(g, s, x0 - 6, sy(v) + 4, { size: 11, font: 'mono', color: C.muted, align: 'right' }); });
        V.text(g, '1', x0 - 6, cy + 4, { size: 11, font: 'mono', color: C.muted, align: 'right' });
        const bw = Math.min(30, cw * 0.5);
        r.forEach((x, i) => {
          const xc = x0 + i * cw + cw / 2, y2 = sy(Math.min(Math.max(x, 1 / 6), 6));
          // red only where the chosen correction acts: TIS cuts ρ > C; IcePop zeroes ρ outside [0.5, 2]
          const acts = S.corr === 'tis' ? x > 2 : S.corr === 'ice' ? x > 2 || x < 0.5 : false;
          const col = acts ? RED() : C.b;
          V.box(g, xc - bw / 2, Math.min(cy, y2), bw, Math.max(1.5, Math.abs(y2 - cy)), { r: 3, fill: V.alpha(col, 0.5), stroke: col, lw: 1.2 });
          V.text(g, fmt(x, x >= 10 ? 1 : 2), xc, x >= 1 ? y2 - 5 : y2 + 14, { size: 11, font: 'mono', color: C.ink, align: 'center' });
        });
        // weights
        const wTop = bTop + bH + 16, wH = h - wTop - 22;
        V.text(g, 'weight', 22, wTop + wH / 2 + 4, { size: 11, color: C.muted, align: 'center' });
        const wmax = S.corr === 'seq' ? Math.max(2.5, ...wv) : 2.5;
        wv.forEach((x, i) => {
          const xc = x0 + i * cw + cw / 2, hh = (Math.min(x, wmax) / wmax) * wH;
          const col = x === 0 ? C.muted : S.corr === 'none' ? C.d : S.corr === 'tis' && r[i] > 2 ? C.d : x > 2.01 ? RED() : C.a;
          V.box(g, xc - bw / 2, wTop + wH - hh, bw, Math.max(1.5, hh), { r: 3, fill: V.alpha(col, 0.6), stroke: col, lw: 1 });
          V.text(g, x === 0 ? '0' : fmt(x, x >= 10 ? 1 : 2), xc, wTop + wH + 14, { size: 11, font: 'mono', color: x === 0 ? C.muted : C.ink, align: 'center' });
        });
        const note = { none: 'ignore ρ: biased gradient (off-policy without knowing it)', seq: 'sequence product: one rare token dominates everything', tis: 'TIS: min(ρ, 2) per token', ice: 'IcePop: zeroes ρ outside [0.5, 2]' }[S.corr];
        V.text(g, note, w - 12, wTop - 2, { size: 11, color: C.muted, align: 'right', maxW: w - x0 - 20 });
      }
      function drawRoute(g, w, h, t, C, narrow) {
        const Eg = engines(g, w, t, C, narrow);
        const R = routes();
        const top = Eg.bottom + 26;
        const gw2 = (w - 60) / 2, cell = Math.min(28, (gw2 - 20) / E - 3), rowH = Math.min(cell + 16, (h - top - 50) / L);
        const gx = [20, w / 2 + 10];
        ['rollout (inference)', 'training'].forEach((s, k) => V.text(g, s, gx[k] + 30, top - 6, { size: 11.5, weight: 600, color: C.ink }));
        const ph = (t * 0.45) % 1, lp = ph * (L + 1);
        for (let l = 0; l < L; l++) {
          const y = top + 6 + l * rowH;
          [R.inf, R.tr].forEach((sel, k) => {
            const x0 = gx[k] + 30;
            V.text(g, 'L' + (l + 1), gx[k] + 22, y + cell / 2 + 4, { size: 11, font: 'mono', color: C.muted, align: 'right' });
            if (k === 1 && R.diff[l]) V.box(g, x0 - 4, y - 3, E * (cell + 3) + 5, cell + 6, { r: 5, stroke: RED(), lw: 2, fill: V.alpha(RED(), 0.06) });
            for (let e = 0; e < E; e++) {
              const on = sel[l].includes(e);
              const pulse = on && Math.abs(lp - (l + 0.5)) < 0.6 ? 0.35 : 0;
              V.box(g, x0 + e * (cell + 3), y, cell, cell, { r: 4, fill: on ? V.alpha(k ? C.a : C.b, 0.65 + pulse) : V.alpha(C.ink, 0.06), stroke: on ? (k ? C.a : C.b) : V.alpha(C.ink, 0.12), lw: 1 });
            }
          });
        }
        // the token crosses the layers: a glowing band on the current layer
        const lc = Math.min(L - 1, Math.floor(lp));
        if (lp < L) [0, 1].forEach((k) => { V.box(g, gx[k] + 26, top + 6 + lc * rowH - 4, E * (cell + 3) + 7, cell + 8, { r: 6, fill: V.alpha(k ? C.a : C.b, 0.12) }); });
        const n = R.diff.filter(Boolean).length;
        const msg = S.keep ? 'Keep Routing / R3: training reuses the rollout’s choices' : n ? n + ' layer(s) with different experts: the token went through another MLP' : 'no divergence for this token';
        const yM = Math.min(h - 50, top + 6 + L * rowH + 22);
        const hm = V.wrap(g, msg, 20, yM, w - 40, { size: 12, weight: 600, color: S.keep ? C.a : n ? RED() : C.muted, lineH: 16, maxLines: 2 });
        V.wrap(g, 'In a real MoE, the R3 paper measures ~94% of tokens with different experts in at least one layer between training and inference.', 20, yM + hm + 6, w - 40, { size: 11.5, color: C.muted, lineH: 15, maxLines: 3 });
      }
      H = V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const narrow = w < 640;
        if (S.mode === 'prob') drawProb(g, w, h, t, C, narrow); else drawRoute(g, w, h, t, C, narrow);
      });
      upd();
    })();
  },
});

LA.chapter({
  id: 'rl-fundamentos',
  part: 11,
  order: 54,
  title: 'RL for LLMs: policy gradient, A3C and TRPO',
  navTitle: 'RL fundamentals',
  subtitle: 'How to turn a score at the end of the response into a change in behavior: the LLM as an agent, the policy gradient theorem, baselines, critic and GAE, importance sampling (and the vLLM × trainer mismatch), the asynchronous actors of A3C and IMPALA, and the trust region of TRPO.',
  minutes: 45,
  recipes: [],
  tags: ['reinforcement learning', 'policy gradient', 'REINFORCE', 'baseline', 'GAE', 'importance sampling', 'A3C', 'IMPALA', 'TRPO', 'natural gradient'],
  goals: [
    'Model an LLM\'s generation as a <b>token MDP</b> (or as a sequence bandit) and define return, value \\(V\\), \\(Q\\) and advantage \\(A\\)',
    'Derive the <b>policy gradient theorem</b> with the log-derivative trick and explain why baselines, the critic, TD and <b>GAE(γ, λ)</b> reduce variance',
    'Use <b>importance sampling</b> (per token, truncated) to learn from off-policy data, and connect the mismatch between inference engine and trainer to Module 1',
    'Explain what <b>A3C</b>, <b>IMPALA/V-trace</b> and <b>TRPO</b> (Fisher, natural gradient, conjugate gradient, line search) solve, and where these ideas live in the 2025–2026 algorithms',
  ],
  prereqs: ['preferencias-reward-models', 'aprender-com-dados'],
  summary: [
    'A generating LLM is an agent in a <b>token MDP</b>: state = prompt + prefix (the KV cache), action = next token, deterministic transition, sparse reward at the EOS, \\(\\gamma \\approx 1\\). In the bandit view, the entire response is one action, with \\(\\log\\pi(y\\mid x) = \\sum_t\\log\\pi(y_t\\mid s_t)\\).',
    'The <b>advantage</b> \\(A = Q - V\\) measures how much an action beats the policy\'s own average; it has zero mean and changes when the policy changes.',
    '<b>Policy gradient</b>: \\(\\nabla J = \\mathbb E[\\sum_t \\hat A_t\\nabla\\log\\pi_\\theta(y_t\\mid s_t)]\\). It is the SFT gradient with each token weighted by its advantage; in the logits, \\(z \\leftarrow z + \\eta A(\\mathbf e_a - \\pi)\\). There is no need to differentiate through the sampling or the RM.',
    '<b>Variance</b>: REINFORCE is unbiased but noisy. A baseline that doesn\'t depend on the action keeps the bias at zero and cuts the variance; RLOO uses the mean of the other \\(k-1\\) responses; the critic learns \\(V\\). TD and <b>GAE</b> (\\(\\hat A_t = \\delta_t + \\gamma\\lambda\\hat A_{t+1}\\)) trade bias against variance; in long CoT the signal decays as \\(\\lambda^{T-t}\\) (VC-PPO, VAPO, SAO).',
    '<b>Off-policy</b>: per-token importance sampling (\\(\\rho_t = \\pi_\\theta/\\pi_{\\theta_{\\text{old}}}\\)), truncated (TIS) because the per-sequence product explodes. The vLLM × FSDP mismatch, quantized rollouts and MoE routing make RL off-policy without anyone asking for it: TIS, IcePop, R3, an FP32 LM head and batch-invariant kernels.',
    '<b>A3C</b> (asynchronous actors, Hogwild!, n-step advantage, entropy bonus 0.01) and <b>IMPALA</b> (stale trajectories, V-trace) are the ancestors of asynchronous LLM RL, in which the actors are inference engines and the staleness is kept under control.',
    '<b>TRPO</b>: maximize the surrogate objective with average KL \\(\\le\\delta\\) (0.01), via Fisher, natural gradient \\(F^{-1}g\\), conjugate gradient, step \\(\\beta = \\sqrt{2\\delta/s^\\top Fs}\\) and line search. Too expensive for LLMs, it motivated PPO\'s first-order clipping.',
  ],
  quiz: [
    { q: 'Why can the policy gradient be computed without differentiating through the token sampling or through the reward model?', o: ['Because the RM is always differentiable', 'Because the log-derivative trick swaps \\(\\nabla\\mathbb E[R]\\) for \\(\\mathbb E[R\\,\\nabla\\log p_\\theta(\\tau)]\\), and in \\(\\log p_\\theta(\\tau)\\) only the policy terms depend on \\(\\theta\\); the reward enters only as a number', 'Because the gradient is estimated by finite differences on the weights', 'Because greedy sampling is differentiable'], a: 1, e: '\\(\\nabla p = p\\nabla\\log p\\) turns the gradient of an expectation into an expectation of gradients. The dynamics \\(P\\) and the initial distribution \\(\\rho_0\\) don\'t depend on \\(\\theta\\) and vanish; what remains is \\(\\sum_t\\nabla\\log\\pi_\\theta(a_t\\mid s_t)\\), an ordinary backward pass, multiplied by the score. That is why a verifier that runs code tests, which is not differentiable, can serve as the reward.' },
    { q: 'With RLOO, \\(k = 4\\) responses and scores \\((1, 0, 0, 1)\\), what are the advantages?', o: ['\\((+0.5,\\ -0.5,\\ -0.5,\\ +0.5)\\)', '\\((+1,\\ -1,\\ -1,\\ +1)\\)', '\\((+0.667,\\ -0.667,\\ -0.667,\\ +0.667)\\)', '\\((1,\\ 0,\\ 0,\\ 1)\\)'], a: 2, e: 'Each response\'s baseline is the mean of the other three: for a score of 1, \\((0+0+1)/3 = 1/3\\), and the advantage is \\(2/3\\); for a score of 0, \\((1+0+1)/3 = 2/3\\), and the advantage is \\(-2/3\\). It equals \\(\\tfrac43(R_i - 0.5)\\). \\((\\pm0.5)\\) is \\(R - \\bar R\\) without the factor; \\((\\pm1)\\) is GRPO\'s normalization with the population standard deviation (with the sample one, the default of <code>torch.std</code>, it would be \\(\\pm0.866\\)).' },
    { q: 'In GAE, what happens at the extremes \\(\\lambda = 0\\) and \\(\\lambda = 1\\)?', o: ['\\(\\lambda = 0\\) ignores the critic; \\(\\lambda = 1\\) uses only the critic', '\\(\\lambda = 0\\) gives \\(\\hat A_t = \\delta_t\\) (one-step TD: little variance, bias if \\(V\\) is wrong); \\(\\lambda = 1\\) gives the Monte Carlo return minus \\(V(s_t)\\) (no bootstrap bias, high variance)', 'Both give the same result when \\(\\gamma = 1\\)', '\\(\\lambda\\) only changes the scale of the advantages, not their shape'], a: 1, e: 'The recursion \\(\\hat A_t = \\delta_t + \\gamma\\lambda\\hat A_{t+1}\\) shows it: with \\(\\lambda = 0\\), each token only sees the critic\'s next guess; with \\(\\lambda = 1\\), the sum telescopes into \\(\\sum_l\\gamma^l r_{t+l} - V(s_t)\\). In the 5-token example, \\(\\hat A_1\\) goes from 0.19 (\\(\\lambda = 0\\)) to 1.10 (\\(\\lambda = 1\\)).' },
    { q: 'Why do PPO and its successors use per-token importance ratios rather than the product over the whole sequence?', o: ['Because the per-sequence product is always 1', 'Because the product of hundreds of ratios has enormous variance (one rare token with \\(\\rho = 5\\) multiplies everything by 5); the per-token ratio is a first-order approximation, valid as long as the policies are close, as TRPO justifies', 'Because vLLM doesn\'t return per-sequence log-probs', 'Because the per-token ratio is unbiased and the per-sequence one is not'], a: 1, e: 'In the 4-token example, a single rare token with \\(\\rho = 5\\) drives the product to 5.71. TRPO\'s local approximation uses the current policy\'s states, which is only safe inside a trust region; hence PPO\'s clipping and TIS\'s truncation. The per-token ratio is not exact: it ignores the change in the state distribution.' },
    { q: 'What does IMPALA change compared with A3C, and what is its analog in LLM RL?', o: ['It replaces the policy with a Q table', 'The actors send trajectories (not gradients) to a central learner and keep generating with stale weights; V-trace corrects the staleness with truncated IS. In LLMs, the actors are inference engines and the staleness is the lag in weight versions', 'It eliminates staleness by synchronizing all actors at every step', 'It uses GPUs instead of CPUs, without changing the algorithm'], a: 1, e: 'A3C applies gradients asynchronously (Hogwild!) to shared parameters. IMPALA separates generation from learning, accepts that the data comes from a lagging policy \\(\\mu\\) and truncates the ratios \\(\\pi/\\mu\\) at \\(\\bar\\rho\\) and \\(\\bar c\\). It is the design of asynchronous LLM RL (AReaL, slime, MiMo), in which staleness is capped and corrected by truncated IS (TIS, IcePop).' },
    { q: 'In TRPO, what are the Fisher matrix and conjugate gradient for?', o: ['The Fisher is the Hessian of the SFT loss; CG replaces Adam', 'The Fisher is the Hessian of the KL between policies and measures distance in behavior; the direction is the natural gradient \\(F^{-1}g\\), obtained by CG using only Fisher-vector products, without forming or inverting \\(F\\)', 'The Fisher normalizes the advantages; CG performs the line search', 'Both only serve to compute \\(\\delta\\)'], a: 1, e: 'The constraint \\(D_{KL} \\le \\delta\\) becomes \\(\\tfrac12\\Delta^\\top F\\Delta \\le \\delta\\). The step is \\(\\beta s\\), with \\(s = F^{-1}g\\) and \\(\\beta = \\sqrt{2\\delta/s^\\top Fs}\\); in the 2D example, CG reaches \\((0.2857,\\ 0.8571)\\) in 2 iterations, and the step is \\((0.0378,\\ 0.1134)\\). The line search checks the real KL and shrinks the step if needed.' },
  ],
  mount(root, LA) {
    const V = LA.viz, UI = LA.ui;
    const RED = () => V.C().red || '#FB2C36';
    const fmt = (x, d = 2) => {
      if (!isFinite(x)) return x > 0 ? '∞' : '−∞';
      let s = (+x).toFixed(d);
      if (/^-0(\.0+)?$/.test(s)) s = s.slice(1);
      return s.replace(/^-/, '−');
    };
    const sgn = (x, d = 2) => (x > 0.00005 ? '+' : '') + fmt(x, d);
    const SUP = { '-': '⁻', 0: '⁰', 1: '¹', 2: '²', 3: '³', 4: '⁴', 5: '⁵', 6: '⁶', 7: '⁷', 8: '⁸', 9: '⁹' };
    const sup = (n) => String(n).split('').map((c) => SUP[c] || c).join('');
    /** formata 10^lg (lg = log10 do valor), sem underflow */
    const fmtLog = (lg, d = 1) => {
      if (lg > -3 && lg < 3) return fmt(Math.pow(10, lg), lg > -1 ? 3 : 4);
      const e = Math.floor(lg), m = Math.pow(10, lg - e);
      return fmt(m, d) + '×10' + sup(e);
    };
    /** tira o uppercase do CSS de rótulos com símbolos matemáticos */
    const nocase = (c) => { const el = c && (c.el || c); if (el && el.querySelectorAll) el.querySelectorAll('label, span').forEach((n) => { if (!n.classList.contains('val')) n.style.textTransform = 'none'; }); return c; };
    function adaptH(fig, wide, narrow, bp = 600) {
      const st = fig.querySelector('.la-fig-stage');
      const set = () => { const want = (fig.clientWidth < bp ? narrow : wide) + 'px'; if (st.style.height !== want) st.style.height = want; };
      set();
      const ro = new ResizeObserver(set); ro.observe(fig);
      if (LA.onCleanup) LA.onCleanup(() => ro.disconnect());
    }
    function tw(g, s, size, font = 'sans', weight = 400) { g.font = V.font(size, font, weight); return g.measureText(String(s)).width; }
    function chip(g, x, y, w, h, label, o = {}) {
      const C = V.C();
      V.box(g, x, y, w, h, { r: o.r == null ? 6 : o.r, fill: o.fill || V.alpha(C.muted, 0.12), stroke: o.stroke || V.alpha(C.muted, 0.4), lw: o.lw || 1.2 });
      if (label != null) V.text(g, label, x + w / 2, y + h / 2 + 0.5, { size: o.size || 12, font: o.font || 'mono', weight: o.weight || 500, color: o.color || C.ink, align: 'center', base: 'middle', maxW: o.maxW });
    }
    /** célula com valor (fundo colorido pela cor dada) */
    function cell(g, x, y, w, h, txt, col, o = {}) {
      const C = V.C();
      V.box(g, x, y, w, h, { r: 4, fill: col ? V.alpha(col, o.a == null ? 0.22 : o.a) : V.alpha(C.muted, 0.1), stroke: o.stroke || (col ? V.alpha(col, 0.7) : V.alpha(C.muted, 0.3)), lw: o.lw || 1 });
      if (txt != null) V.text(g, txt, x + w / 2, y + h / 2 + 0.5, { size: o.size || 11.5, font: 'mono', weight: o.weight || 500, color: o.color || C.ink, align: 'center', base: 'middle', maxW: w - 3 });
    }
    const sampleCat = (d, u) => { let s = 0; for (let i = 0; i < d.length; i++) { s += d[i]; if (u < s) return i; } return d.length - 1; };
    const softmax = (z) => V.softmax(z);

    /* ================================================ FIG 1 — MDP de tokens */
    (function () {
      const fig = root.querySelector('#fig-mdp');
      adaptH(fig, 410, 500);
      const TOK = [
        { s: 'First', c: [['The', 0.38], ['First', 0.33], ['20', 0.17], ['14', 0.12]], pick: 1, kl: 0.12 },
        { s: '3×4', c: [['3×4', 0.62], ['2+3', 0.30], ['let', 0.05], ['the', 0.03]], pick: 0, kl: 0.05 },
        { s: '=', c: [['=', 0.81], ['is', 0.12], ['gives', 0.05], [',', 0.02]], pick: 0, kl: 0.02 },
        { s: '12', c: [['12', 0.93], ['7', 0.04], ['14', 0.02], ['34', 0.01]], pick: 0, kl: -0.03 },
        { s: ';', c: [[';', 0.55], [',', 0.25], ['and', 0.15], ['.', 0.05]], pick: 0, kl: 0.08 },
        { s: '2+12', c: [['2+12', 0.71], ['then', 0.19], ['so', 0.07], ['add', 0.03]], pick: 0, kl: 0.04 },
        { s: '=', c: [['=', 0.88], ['gives', 0.08], ['is', 0.03], ['→', 0.01]], pick: 0, kl: 0.01 },
        { s: '14', c: [['14', 0.95], ['24', 0.03], ['13', 0.01], ['20', 0.01]], pick: 0, kl: 0.02 },
        { s: '<eos>', c: [['<eos>', 0.9], ['.', 0.06], ['So', 0.03], ['!', 0.01]], pick: 0, kl: 0.03 },
      ];
      const NT = TOK.length, BETA = 0.05, D = 1.25, TAIL = 6, CYC = NT * D + TAIL;
      const S = { view: 'tok', kl: false, gamma: 1 };
      const logp = TOK.map((tk) => Math.log(tk.c[tk.pick][1]));
      const sumLogp = logp.reduce((a, b) => a + b, 0);
      const rewards = () => TOK.map((tk, i) => (S.kl ? -BETA * tk.kl : 0) + (i === NT - 1 ? 1 : 0));
      const returns = (r) => { const G = new Array(NT).fill(0); let acc = 0; for (let i = NT - 1; i >= 0; i--) { acc = r[i] + S.gamma * acc; G[i] = acc; } return G; };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'View', options: [{ v: 'tok', label: 'per token' }, { v: 'seq', label: 'bandit' }], value: 'tok', onChange: (v) => { S.view = v; } });
      nocase(UI.toggle(ctl, { label: 'per-token KL (β = 0.05)', value: false, onChange: (v) => { S.kl = v; upd(); } }));
      nocase(UI.slider(ctl, { label: 'γ', min: 0.8, max: 1, step: 0.01, value: 1, fmt: (v) => fmt(v, 2), onInput: (v) => { S.gamma = v; upd(); } }));
      UI.readout(ctl, 'score at EOS', '1.00');
      const rG = nocase(UI.readout(ctl, 'G₁ (1st token)', '—'));
      nocase(UI.readout(ctl, 'log π(y|x)', fmt(sumLogp, 2)));
      function upd() { rG.set(fmt(returns(rewards())[0], 3)); }
      upd();

      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const narrow = w < 560, pad = 12;
        const tc = t % CYC;
        const k = Math.min(NT, Math.floor(tc / D));
        const u = k < NT ? (tc - k * D) / D : 1;
        const tr = tc - NT * D;
        const flying = k < NT && u > 0.62 && u < 0.86;
        const nIn = k + (k < NT && u >= 0.86 ? 1 : 0);
        const r = rewards(), G = returns(r);
        // ---------- estado: chips
        V.text(g, 'state sₜ = prompt + tokens already chosen', pad, 16, { size: 11.5, color: C.muted });
        const items = [{ s: 'x: What is 2+3×4?', p: 1 }].concat(TOK.map((tk) => ({ s: tk.s })));
        const pos = [];
        let cx = pad, cy = 24;
        items.forEach((it) => {
          const ww = tw(g, it.s, 12, 'mono', 600) + 14;
          if (cx + ww > w - pad && cx > pad) { cx = pad; cy += 28; }
          pos.push({ x: cx, y: cy, w: ww });
          cx += ww + 5;
        });
        const nShown = 1 + nIn;
        for (let i = 0; i < nShown; i++) {
          const p = pos[i], it = items[i];
          const isEos = i === NT && tr >= 0;
          chip(g, p.x, p.y, p.w, 22, it.s, { size: 12, weight: 600, fill: it.p ? V.alpha(C.b, 0.16) : V.alpha(isEos ? C.d : C.a, 0.18), stroke: it.p ? V.alpha(C.b, 0.6) : (isEos ? C.d : V.alpha(C.a, 0.7)) });
        }
        const stateBottom = pos[pos.length - 1].y + 22;
        // KV cache strip
        const kvY = stateBottom + 10, nKV = 6 + nIn;
        V.text(g, 'KV cache', pad, kvY + 9, { size: 11, font: 'mono', color: C.muted });
        const kvX = pad + 66, kvC = narrow ? 8 : 10;
        for (let i = 0; i < 6 + NT; i++) {
          const on = i < nKV, fresh = i === nKV - 1 && k < NT && u >= 0.86;
          g.fillStyle = on ? V.alpha(i < 6 ? C.b : C.a, fresh ? 0.95 : 0.6) : V.alpha(C.muted, 0.12);
          V.rr(g, kvX + i * (kvC + 3), kvY, kvC, kvC, 2); g.fill();
        }
        V.text(g, `${nKV} positions`, kvX + (6 + NT) * (kvC + 3) + 6, kvY + 9, { size: 11, font: 'mono', color: C.muted });
        // ---------- política
        const rowsY = h - (narrow ? 104 : 96);
        const pY = kvY + 30, pH = Math.max(110, Math.min(narrow ? 200 : 999, rowsY - pY - (narrow ? 112 : 30)));
        const chartW = narrow ? w - 2 * pad : Math.min(420, w * 0.5);
        const chartX = pad;
        const step = k < NT ? TOK[k] : null;
        let pickX = 0, pickY = 0;
        if (step) {
          V.text(g, `policy π(· | sₜ), t = ${k + 1}`, chartX, pY + 4, { size: 12, weight: 600, color: C.ink });
          const n = 4, gw = chartW / n, base = pY + pH - 18, bh = pH - 44, pmax = Math.max(...step.c.map((c2) => c2[1]));
          const grow = V.ease.out(V.clamp(u / 0.35, 0, 1));
          let hi = -1;
          if (u >= 0.35 && u < 0.62) { const f = (u - 0.35) / 0.27; hi = f < 0.8 ? Math.floor(f * 11) % n : step.pick; }
          else if (u >= 0.62) hi = step.pick;
          for (let j = 0; j < n; j++) {
            const [lab, pv] = step.c[j];
            const x = chartX + gw * j + gw * 0.18, bw = gw * 0.64, hh = Math.max(2, (pv / pmax) * bh * grow);
            const isHi = j === hi, chosen = u >= 0.62 && j === step.pick;
            V.box(g, x, base - hh, bw, hh, { r: 4, fill: V.alpha(chosen ? C.a : C.b, isHi ? 0.85 : 0.35), stroke: isHi ? (chosen ? C.a : C.b) : V.alpha(C.b, 0.6), lw: isHi ? 2 : 1 });
            if (grow > 0.5) V.text(g, fmt(pv, 2), x + bw / 2, base - hh - 5, { size: 11, font: 'mono', color: C.ink, align: 'center' });
            V.text(g, lab, x + bw / 2, base + 14, { size: 11.5, font: 'mono', weight: isHi ? 700 : 400, color: isHi ? C.ink : C.muted, align: 'center', maxW: gw - 4 });
            if (j === step.pick) { pickX = x + bw / 2; pickY = base - hh; }
          }
          if (u >= 0.35 && u < 0.62) V.text(g, 'sampling…', chartX + chartW, pY + 4, { size: 11.5, color: C.d, align: 'right' });
          // texto lateral
          const tx = narrow ? pad : chartX + chartW + 26, ty = narrow ? pY + pH + 16 : pY + 26;
          {
            const lp = Math.log(step.c[step.pick][1]);
            const lines = u >= 0.62
              ? [[`aₜ = “${step.s}”`, C.a, 700], [`π(aₜ|sₜ) = ${fmt(step.c[step.pick][1], 2)}   log π = ${fmt(lp, 2)}`, C.ink, 400], ['sₜ₊₁ = sₜ ⊕ aₜ  (the token joins the state)', C.muted, 400]]
              : [['the policy assigns a probability', C.muted, 400], ['to each candidate token…', C.muted, 400], ['…and the action is SAMPLED, not the favorite', C.muted, 400]];
            if (S.view === 'seq') lines.push([`Σ log π so far = ${fmt(logp.slice(0, nIn).reduce((a, b) => a + b, 0), 2)}`, C.b, 600]);
            lines.forEach(([s, c, wgt], i) => V.text(g, s, tx, ty + i * 19, { size: 12, font: i === 0 ? 'mono' : 'sans', weight: wgt, color: c, maxW: w - tx - pad }));
          }
        } else {
          const glow = 0.5 + 0.5 * Math.sin(tr * 5);
          V.text(g, 'EOS: the episode ends and the verifier gives the score', pad, pY + 4, { size: 12, weight: 600, color: C.ink, maxW: w - 2 * pad });
          const bx = pad, by = pY + 22, bw2 = Math.min(300, w - 2 * pad);
          V.box(g, bx, by, bw2, 54, { r: 10, fill: V.alpha(C.d, 0.12 + 0.12 * glow), stroke: C.d, lw: 1.6 });
          V.text(g, 'R(x, y) = 1   (“14” is correct)', bx + 14, by + 33, { size: 14, font: 'mono', weight: 700, color: C.ink, maxW: bw2 - 20 });
          V.glow(g, bx + bw2 - 22, by + 27, 26, C.d, 0.4 + 0.4 * glow);
          if (S.view === 'seq') V.text(g, `one action (the entire response), one score; log π(y|x) = ${fmt(sumLogp, 2)}`, pad, by + 76, { size: 12, color: C.b, maxW: w - 2 * pad });
          else V.text(g, 'the return Gₜ trickles back from the EOS…', pad, by + 76, { size: 12, color: C.muted, maxW: w - 2 * pad });
        }
        // chip voando para o estado
        if (flying && step) {
          const f = V.ease.inOut((u - 0.62) / 0.24), p = pos[k + 1];
          const x = V.lerp(pickX - p.w / 2, p.x, f), y = V.lerp(pickY - 24, p.y, f);
          chip(g, x, y, p.w, 22, step.s, { size: 12, weight: 700, fill: V.alpha(C.a, 0.4), stroke: C.a });
        }
        // ---------- recompensas e retornos
        const lblW = narrow ? 44 : 64;
        const gx = pad + lblW, gw2 = (w - pad - gx) / NT, ch = 24;
        const showG = tr >= 0;
        const gDone = showG ? Math.floor(tr / 0.22) : -1;
        if (S.view === 'tok') {
          V.text(g, 'rₜ', pad, rowsY + 16, { size: 12, font: 'mono', weight: 600, color: C.muted });
          V.text(g, 'Gₜ', pad, rowsY + 16 + ch + 8, { size: 12, font: 'mono', weight: 600, color: C.muted });
          for (let i = 0; i < NT; i++) {
            const x = gx + i * gw2 + 2, cw = gw2 - 4;
            const known = i < nIn;
            const rv = r[i];
            const rc = !known ? null : i === NT - 1 ? C.d : rv < -1e-9 ? RED() : rv > 1e-9 ? C.a : null;
            cell(g, x, rowsY, cw, ch, known ? fmt(rv, cw > 44 ? 3 : 2) : '·', rc, { size: cw > 40 ? 11 : 10, a: i === NT - 1 && known ? 0.4 : 0.22 });
            const gv = showG && i >= NT - 1 - gDone;
            const gcol = gv ? V.cmap(V.clamp(G[i], 0, 1) * 0.85 + 0.1, 'seq') : null;
            cell(g, x, rowsY + ch + 8, cw, ch, gv ? fmt(G[i], cw > 44 ? 3 : 2) : '·', gv ? C.a : null, { size: cw > 40 ? 11 : 10, a: gv ? 0.15 + 0.3 * V.clamp(G[i], 0, 1) : 0.1 });
            if (gcol && gv) { /* cor já aplicada pelo alpha */ }
            V.text(g, narrow ? String(i + 1) : TOK[i].s, x + cw / 2, rowsY + 2 * ch + 26, { size: 11, font: 'mono', color: i < nIn ? C.ink : V.alpha(C.muted, 0.6), align: 'center', maxW: cw + 2 });
          }
        } else {
          V.text(g, 'log π', pad, rowsY + 16, { size: 12, font: 'mono', weight: 600, color: C.muted });
          for (let i = 0; i < NT; i++) {
            const x = gx + i * gw2 + 2, cw = gw2 - 4;
            const known = i < nIn;
            cell(g, x, rowsY, cw, ch, known ? fmt(logp[i], 2) : '·', known ? C.b : null, { size: cw > 40 ? 11 : 10 });
            V.text(g, narrow ? String(i + 1) : TOK[i].s, x + cw / 2, rowsY + ch + 14, { size: 11, font: 'mono', color: known ? C.ink : V.alpha(C.muted, 0.6), align: 'center', maxW: cw + 2 });
          }
          if (showG) {
            const f = V.ease.out(V.clamp(tr / 1.2, 0, 1));
            const sx = gx, ex = w - pad, sy = rowsY + ch + 22;
            const bw3 = (ex - sx) * f;
            V.box(g, sx, sy, Math.max(30, bw3), 26, { r: 7, fill: V.alpha(C.b, 0.2), stroke: C.b, lw: 1.4 });
            if (f > 0.6) V.text(g, `log π(y|x) = Σ = ${fmt(sumLogp, 2)}   ·   R = 1`, sx + 10, sy + 17, { size: 12, font: 'mono', weight: 600, color: C.ink, maxW: bw3 - 16 });
          } else {
            V.text(g, 'in the bandit view, the action is the entire response', gx, rowsY + ch + 36, { size: 11.5, color: C.muted, maxW: w - gx - pad });
          }
        }
      });
    })();

    /* ================================================ FIG 2 — árvore de valor */
    (function () {
      const fig = root.querySelector('#fig-valor');
      adaptH(fig, 450, 540);
      const P0 = [0.5, 0.2, 0.3], Q = [1, 0.7, 0], LBL = ['14', 'let me check', '20'];
      const S = { T: 1, parts: [], n: 0, sum: 0, hist: [], pend: 0 };
      const rnd = V.rng(5402);
      const pol = () => { const a = P0.map((p) => Math.pow(p, 1 / S.T)); const s = a.reduce((x, y) => x + y, 0); return a.map((x) => x / s); };
      const vOf = (pi) => pi.reduce((s, p, i) => s + p * Q[i], 0);
      const ctl = UI.controls(fig);
      nocase(UI.slider(ctl, { label: 'temperature T', min: 0.3, max: 2, step: 0.05, value: 1, fmt: (v) => fmt(v, 2), onInput: (v) => { S.T = v; reset(); upd(); } }));
      UI.button(ctl, 'reset Monte Carlo', () => reset());
      const rV = nocase(UI.readout(ctl, 'exact V(s)', '—')), rM = nocase(UI.readout(ctl, 'MC estimate', '—')), rA = nocase(UI.readout(ctl, 'A(let me check)', '—'));
      function reset() {
        S.parts = []; S.n = 0; S.sum = 0; S.hist = [];
        // pré-aquece com 60 rollouts instantâneos, para a estimativa já começar visível
        const pi = pol();
        for (let k = 0; k < 60; k++) { const a = sampleCat(pi, rnd()); const R = a === 0 ? 1 : a === 2 ? 0 : (rnd() < 0.7 ? 1 : 0); S.n++; S.sum += R; S.hist.push(S.sum / S.n); }
      }
      reset();
      function upd() { const pi = pol(), v = vOf(pi); rV.set(fmt(v, 3)); const a = Q[1] - v; rA.set(sgn(a, 3), a >= 0 ? 'ok' : 'bad'); }
      upd();
      let frame = 0;
      V.fig(fig, (g, w, h, t, dt) => {
        const C = V.C();
        const narrow = w < 600, pad = 12;
        const pi = pol(), v = vOf(pi);
        // geometria
        let root0, kids, leaves, mc, tab;
        if (!narrow) {
          const tH = h * 0.58;
          root0 = { x: w * 0.13, y: 18 + tH / 2 };
          kids = [0.12, 0.5, 0.88].map((f, i) => ({ x: w * 0.44, y: 18 + tH * f }));
          leaves = [{ x: w * 0.7, y: kids[1].y - tH * 0.17, R: 1, p: 0.7, lab: '… = 14 ✓' }, { x: w * 0.7, y: kids[1].y + tH * 0.17, R: 0, p: 0.3, lab: '… = 20 ✗' }];
          mc = { x: w * 0.81, y: 14, w: w * 0.19 - pad, h: tH - 6 };
          tab = { y: tH + 34 };
        } else {
          root0 = { x: w / 2, y: 34 };
          kids = [0.18, 0.5, 0.82].map((f) => ({ x: w * f, y: 130 }));
          leaves = [{ x: w * 0.3, y: 236, R: 1, p: 0.7, lab: '…= 14 ✓' }, { x: w * 0.7, y: 236, R: 0, p: 0.3, lab: '…= 20 ✗' }];
          mc = { x: pad, y: 268, w: w - 2 * pad, h: 92 };
          tab = { y: 398 };
        }
        const rw = narrow ? 150 : Math.min(176, w * 0.2), rh = 58;
        const kw = narrow ? Math.min(108, w * 0.31) : Math.min(150, w * 0.17), kh = narrow ? 58 : 46;
        const lw2 = narrow ? 92 : 104;
        const rootOut = narrow ? { x: root0.x, y: root0.y + rh / 2 } : { x: root0.x + rw / 2, y: root0.y };
        const kidIn = kids.map((kd) => (narrow ? { x: kd.x, y: kd.y - kh / 2 } : { x: kd.x - kw / 2, y: kd.y }));
        const kidOut = narrow ? { x: kids[1].x, y: kids[1].y + kh / 2 + 28 } : { x: kids[1].x + kw / 2, y: kids[1].y };
        const leafIn = leaves.map((lf) => (narrow ? { x: lf.x, y: lf.y - 13 } : { x: lf.x - lw2 / 2, y: lf.y }));
        // arestas
        const edge = (a, b, lw, col) => {
          g.beginPath();
          if (!narrow) { const mx = (a.x + b.x) / 2; g.moveTo(a.x, a.y); g.bezierCurveTo(mx, a.y, mx, b.y, b.x, b.y); }
          else { const my = (a.y + b.y) / 2; g.moveTo(a.x, a.y); g.bezierCurveTo(a.x, my, b.x, my, b.x, b.y); }
          g.strokeStyle = col; g.lineWidth = lw; g.stroke();
        };
        const ptOn = (a, b, f) => {
          if (!narrow) { const mx = (a.x + b.x) / 2; return V.bez([a.x, a.y], [mx, a.y], [mx, b.y], [b.x, b.y], f); }
          const my = (a.y + b.y) / 2; return V.bez([a.x, a.y], [a.x, my], [b.x, my], [b.x, b.y], f);
        };
        kids.forEach((kd, i) => {
          edge(rootOut, kidIn[i], 1 + 12 * pi[i], V.alpha(C.b, 0.25 + 0.5 * pi[i]));
          const m = ptOn(rootOut, kidIn[i], 0.5);
          if (!narrow) V.text(g, 'π = ' + fmt(pi[i], 2), m[0], m[1] - 7, { size: 11, font: 'mono', weight: 600, color: C.b, align: 'center' });
        });
        leaves.forEach((lf, li) => { edge(kidOut, leafIn[li], 1 + 6 * lf.p, V.alpha(C.muted, 0.45)); const m = ptOn(kidOut, leafIn[li], 0.5); V.text(g, fmt(lf.p, 1), m[0] + 4, m[1] - 5, { size: 11, font: 'mono', color: C.muted, align: 'center' }); });
        // partículas Monte Carlo
        if (!LA.paused) {
          S.pend += dt * 6;
          while (S.pend >= 1) {
            S.pend -= 1;
            const a = sampleCat(pi, rnd());
            const lf = a === 1 ? (rnd() < 0.7 ? 0 : 1) : -1;
            const R = a === 0 ? 1 : a === 2 ? 0 : (lf === 0 ? 1 : 0);
            S.parts.push({ a, lf, R, t0: t, jit: rnd() - 0.5 });
          }
        }
        for (let i = S.parts.length - 1; i >= 0; i--) {
          const P = S.parts[i], age = t - P.t0;
          const d1 = 0.8, d2 = 0.6;
          let x, y, done = false;
          if (age < d1) { [x, y] = ptOn(rootOut, kidIn[P.a], V.ease.inOut(age / d1)); }
          else if (P.a === 1 && age < d1 + d2) { [x, y] = ptOn(kidOut, leafIn[P.lf], V.ease.inOut((age - d1) / d2)); }
          else done = true;
          if (done) { S.n++; S.sum += P.R; S.hist.push(S.sum / S.n); if (S.hist.length > 400) S.hist.shift(); S.parts.splice(i, 1); continue; }
          V.dot(g, x + P.jit * 5, y + P.jit * 3, 3.2, P.R ? C.a : RED());
        }
        if (++frame % 10 === 0) rM.set(S.n ? `${fmt(S.sum / S.n, 3)} (n=${S.n})` : '—');
        // nós
        V.box(g, root0.x - rw / 2, root0.y - rh / 2, rw, rh, { r: 10, fill: V.alpha(C.d, 0.14), stroke: C.d, lw: 1.6 });
        V.text(g, 's = “…so, 2+3×4 =”', root0.x, root0.y - 10, { size: 11.5, font: 'mono', color: C.ink, align: 'center', maxW: rw - 10 });
        V.text(g, 'V(s) = ' + fmt(v, 3), root0.x, root0.y + 9, { size: 13, font: 'mono', weight: 700, color: C.d, align: 'center' });
        g.fillStyle = V.alpha(C.muted, 0.2); g.fillRect(root0.x - rw / 2 + 10, root0.y + 17, rw - 20, 5);
        g.fillStyle = C.d; g.fillRect(root0.x - rw / 2 + 10, root0.y + 17, (rw - 20) * v, 5);
        kids.forEach((kd, i) => {
          const term = i !== 1;
          V.box(g, kd.x - kw / 2, kd.y - kh / 2, kw, kh, { r: 9, fill: V.alpha(i === 0 ? C.a : i === 2 ? RED() : C.b, 0.12), stroke: i === 0 ? C.a : i === 2 ? RED() : C.b, lw: 1.4 });
          const ty0 = narrow ? kd.y - 12 : kd.y - 6;
          V.text(g, `“${LBL[i]}”` + (term ? (i === 0 ? ' ✓' : ' ✗') : ''), kd.x, ty0, { size: 12, font: 'mono', weight: 600, color: C.ink, align: 'center', maxW: kw - 8 });
          V.text(g, 'Q = ' + fmt(Q[i], 2), kd.x, ty0 + 17, { size: 11.5, font: 'mono', color: C.muted, align: 'center' });
          if (narrow) V.text(g, 'π = ' + fmt(pi[i], 2), kd.x, ty0 + 33, { size: 11, font: 'mono', weight: 600, color: C.b, align: 'center' });
          const A = Q[i] - v;
          const aw = 74, ax = narrow ? kd.x - aw / 2 : kd.x + kw / 2 + 6, ay = narrow ? kd.y + kh / 2 + 4 : kd.y - 11;
          const aSt = { size: 11.5, weight: 700, fill: V.alpha(A >= 0 ? C.a : RED(), 0.2), stroke: A >= 0 ? C.a : RED() };
          if (i !== 1) chip(g, ax, ay, aw, 22, 'A = ' + sgn(A, 2), aSt);
          else chip(g, kd.x - aw / 2, kd.y + kh / 2 + 5, aw, 22, 'A = ' + sgn(A, 2), aSt);
        });
        leaves.forEach((lf) => {
          chip(g, lf.x - lw2 / 2, lf.y - 13, lw2, 26, lf.lab, { size: 11.5, weight: 600, fill: V.alpha(lf.R ? C.a : RED(), 0.12), stroke: lf.R ? C.a : RED() });
        });
        // painel Monte Carlo
        V.box(g, mc.x, mc.y, mc.w, mc.h, { r: 9, fill: V.alpha(C.panel || C.muted, 0.25), stroke: V.alpha(C.muted, 0.3), lw: 1 });
        V.text(g, 'Monte Carlo', mc.x + 8, mc.y + 16, { size: 11.5, weight: 700, color: C.ink });
        V.text(g, S.n ? `V̂ = ${fmt(S.sum / S.n, 3)}` : 'V̂ = —', mc.x + 8, mc.y + 33, { size: 12, font: 'mono', weight: 600, color: C.a });
        V.text(g, `n = ${S.n}`, mc.x + mc.w - 8, mc.y + 16, { size: 11, font: 'mono', color: C.muted, align: 'right' });
        const sx0 = mc.x + 8, sx1 = mc.x + mc.w - 8, sy0 = mc.y + 42, sy1 = mc.y + mc.h - 8;
        const yv = (val) => sy1 - val * (sy1 - sy0);
        V.line(g, sx0, yv(v), sx1, yv(v), { color: C.d, lw: 1.4, dash: [4, 3] });
        V.text(g, 'exact V', sx1, yv(v) - 4, { size: 11, color: C.d, align: 'right' });
        if (S.hist.length > 1) {
          g.beginPath();
          S.hist.forEach((hv, i) => { const x = sx0 + (i / 399) * (sx1 - sx0), y = yv(hv); if (i) g.lineTo(x, y); else g.moveTo(x, y); });
          g.strokeStyle = C.a; g.lineWidth = 1.8; g.stroke();
        }
        // tabela
        const rowsL = ['π(a|s)', 'Q(s,a)', 'A = Q − V', 'π · A'];
        const lab0 = narrow ? 86 : 110, cols = 4;
        const cw = Math.min(140, (w - 2 * pad - lab0) / cols - 6), chh = 23;
        const tx0 = narrow ? pad + lab0 : Math.max(pad + lab0, (w - (lab0 + cols * (cw + 6))) / 2 + lab0);
        const heads = ['“14”', '“let me check”', '“20”', 'Σ'];
        heads.forEach((hd, j) => V.text(g, hd, tx0 + j * (cw + 6) + cw / 2, tab.y - 6, { size: 11.5, font: 'mono', weight: 600, color: j === 3 ? C.d : C.ink, align: 'center', maxW: cw }));
        const vals = [pi, Q, Q.map((q) => q - v), pi.map((p, i) => p * (Q[i] - v))];
        rowsL.forEach((rl, r) => {
          const y = tab.y + r * (chh + 5);
          V.text(g, rl, tx0 - 8, y + chh / 2 + 4, { size: 11.5, font: 'mono', color: C.muted, align: 'right' });
          for (let j = 0; j < 4; j++) {
            const x = tx0 + j * (cw + 6);
            if (j === 3) {
              if (r === 0) cell(g, x, y, cw, chh, fmt(pi.reduce((a, b) => a + b, 0), 2), C.d);
              else if (r === 3) cell(g, x, y, cw, chh, fmt(vals[3].reduce((a, b) => a + b, 0), 3), C.d, { weight: 700 });
              else if (r === 2) cell(g, x, y, cw, chh, 'V = ' + fmt(v, 3), C.d);
              else cell(g, x, y, cw, chh, '', null);
              continue;
            }
            const val = vals[r][j];
            const col = r >= 2 ? (val >= 0 ? C.a : RED()) : r === 0 ? C.b : C.a;
            cell(g, x, y, cw, chh, r >= 2 ? sgn(val, 3) : fmt(val, r === 1 ? 2 : 3), col, { a: r === 0 ? 0.1 + 0.4 * val : r === 1 ? 0.1 + 0.3 * val : 0.12 + 0.5 * Math.min(1, Math.abs(val)) });
          }
        });
      });
    })();

    /* ================================================ FIG 3 — massa de probabilidade */
    (function () {
      const fig = root.querySelector('#fig-softmax');
      adaptH(fig, 460, 410);
      const Z0 = [2.0, 1.0, 0.1], LBL = ['20', '14', 'Let'], RW = [0, 1, 0.7];
      const S = { A: 1.5, a: 1, eta: 1 };
      // identidade do truque: gradiente exato, linhas R(a)(eₐ − π) e amostras Monte Carlo (semente fixa)
      const PI0 = softmax(Z0);
      const JR = PI0.reduce((s, p, j) => s + p * RW[j], 0);
      const EX = PI0.map((p, j) => p * (RW[j] - JR));
      const ROWS = [0, 1, 2].map((a) => PI0.map((p, j) => RW[a] * ((j === a ? 1 : 0) - p)));
      const NS = 1500, SMP = [], CUM = [[0, 0, 0]], CNT = [[0, 0, 0]];
      (function () {
        const r = V.rng(5403);
        for (let k = 0; k < NS; k++) {
          const a = sampleCat(PI0, r()); SMP.push(a);
          CUM.push(CUM[k].map((v, j) => v + ROWS[a][j]));
          CNT.push(CNT[k].map((v, j) => v + (j === a ? 1 : 0)));
        }
      })();
      /** amostras acumuladas após el segundos no passo Monte Carlo (devagar no começo, depois acelera) */
      const nRate = (el) => 2.5 * el + 6 * el * el;
      const steps = [
        { t: 'One draw, one step', d: 'The sampled token receives the advantage \\(A\\); the vector \\(\\mathbf e_a - \\pi\\), times \\(\\eta A\\), shifts the logits and the probability mass migrates. The controls (token, \\(A\\), \\(\\eta\\)) apply to this step.' },
        { t: 'The exact gradient', d: `Give each token a score, \\(R = (0,\\ 1,\\ 0.7)\\). The performance is \\(J = \\sum_a \\pi(a)R(a) = ${fmt(JR, 3)}\\) and, differentiating directly in the logits, \\(\\partial J/\\partial z_j = \\pi_j(R_j - J)\\): the amber row. That is the target the policy gradient needs to estimate.` },
        { t: 'The same computation as an expectation', d: 'Each row is one token\'s “score × \\(\\nabla\\log\\pi(a)\\)”, \\(R(a)(\\mathbf e_a - \\pi)\\). Weighted by \\(\\pi(a)\\) and summed, they give exactly the vector from the previous step: \\(\\nabla J = \\sum_a \\pi(a)R(a)\\nabla\\log\\pi(a) = \\mathbb E_{a\\sim\\pi}[R\\,\\nabla\\log\\pi]\\). The token with score 0 contributes nothing.' },
        { t: 'Monte Carlo', d: 'An expectation is estimated by sampling: draw \\(a\\sim\\pi\\), record \\(R(a)(\\mathbf e_a - \\pi)\\) (a backward pass of the sampled token\'s log-prob) and take the average. The running average closes in on the exact gradient, and each token\'s frequency (amber marks next to the bars) closes in on \\(\\pi\\). Nothing was differentiated through the draw or the score.' },
      ];
      const st = UI.stepper(fig, steps, { autoplay: 9000 });
      /** mexer num controle volta ao passo 1 (o único que os controles afetam) e para o autoplay */
      const manual = () => { st.stop(); if (st.i !== 0) st.go(0); };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Sampled token', options: [{ v: 0, label: '“20”' }, { v: 1, label: '“14”' }, { v: 2, label: '“Let”' }], value: 1, onChange: (v) => { S.a = v; upd(); manual(); } });
      nocase(UI.slider(ctl, { label: 'advantage A', min: -2, max: 2, step: 0.1, value: 1.5, fmt: (v) => sgn(v, 1), onInput: (v) => { S.A = v; upd(); manual(); } }));
      nocase(UI.slider(ctl, { label: 'η', min: 0.1, max: 2, step: 0.1, value: 1, fmt: (v) => fmt(v, 1), onInput: (v) => { S.eta = v; upd(); manual(); } }));
      const r0 = nocase(UI.readout(ctl, 'π(a) before', '—')), r1 = nocase(UI.readout(ctl, 'π(a) after', '—')), rM = UI.readout(ctl, 'mass moved', '—');
      let pi0, grad, dz, pi1;
      function upd() {
        pi0 = softmax(Z0);
        grad = pi0.map((p, j) => (j === S.a ? 1 : 0) - p);
        dz = grad.map((gv) => S.eta * S.A * gv);
        pi1 = softmax(Z0.map((z, j) => z + dz[j]));
        r0.set(fmt(pi0[S.a], 3));
        r1.set(fmt(pi1[S.a], 3), pi1[S.a] > pi0[S.a] ? 'ok' : 'bad');
        rM.set(fmt(0.5 * pi1.reduce((s, p, j) => s + Math.abs(p - pi0[j]), 0), 3));
      }
      upd();
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const narrow = w < 600, pad = 12;
        const mode = st.i, el = st.elapsed(), id = mode > 0;
        // passo 1: 0–1 parado, 1–3 move, 3–5.5 segura, 5.5–6.5 volta
        const ph = t % 6.5;
        const uu = id ? 0 : ph < 1 ? 0 : ph < 3 ? V.ease.inOut((ph - 1) / 2) : ph < 5.5 ? 1 : 1 - V.ease.inOut((ph - 5.5) / 1);
        const z = Z0.map((z0, j) => z0 + uu * dz[j]);
        const pi = softmax(z);
        const L = narrow ? 92 : 150, colW = (w - L - pad) / 3;
        const cx = (j) => L + colW * (j + 0.5);
        const lab = (s, y, col) => V.text(g, s, L - 10, y, { size: narrow ? 11 : 12, font: 'mono', color: col || C.muted, align: 'right', maxW: L - 14 });
        const rCol = (r) => (r >= 1 ? C.a : r > 0 ? C.d : RED());
        // linha 1: tokens
        const y1 = 22;
        lab(id ? 'token, score' : 'token', y1 + 4);
        for (let j = 0; j < 3; j++) {
          if (id) {
            chip(g, cx(j) - 34, y1 - 12, 68, 24, `“${LBL[j]}”`, { size: 12.5, weight: 700, fill: V.alpha(rCol(RW[j]), 0.16), stroke: rCol(RW[j]) });
            V.text(g, 'R = ' + fmt(RW[j], 1), cx(j), y1 + 26, { size: 11, font: 'mono', weight: 700, color: rCol(RW[j]), align: 'center', maxW: colW - 4 });
            continue;
          }
          const isA = j === S.a;
          chip(g, cx(j) - 34, y1 - 12, 68, 24, `“${LBL[j]}”`, { size: 12.5, weight: 700, fill: V.alpha(isA ? (S.A >= 0 ? C.a : RED()) : C.muted, isA ? 0.25 : 0.1), stroke: isA ? (S.A >= 0 ? C.a : RED()) : V.alpha(C.muted, 0.4) });
          if (isA) V.text(g, 'sampled · A = ' + sgn(S.A, 1), cx(j), y1 + 26, { size: 11, weight: 700, color: S.A >= 0 ? C.a : RED(), align: 'center', maxW: colW - 4 });
        }
        // linha 2: logits
        const y2 = 64, cwz = Math.min(96, colW * 0.7);
        lab('logit z', y2 + 16);
        for (let j = 0; j < 3; j++) {
          const moved = uu > 0.02;
          cell(g, cx(j) - cwz / 2, y2, cwz, 26, fmt(z[j], 3), V.cmap(V.clamp(z[j] / 3, -1, 1), 'div'), { a: 1, size: 12.5, weight: 600, stroke: moved ? C.d : V.alpha(C.muted, 0.5), lw: moved ? 1.6 : 1, color: C.ink });
        }
        // linha 3: barras π (mais baixas nos passos da identidade, que precisam de espaço embaixo)
        const y3 = 108, bh = id ? (narrow ? 62 : 78) : narrow ? 150 : (h - 108 - 150), base = y3 + bh;
        const nF = mode === 3 ? Math.min(NS, Math.max(1, Math.floor(nRate(el)))) : 0;
        lab('π = softmax(z)', y3 + bh / 2);
        for (let j = 0; j < 3; j++) {
          const bw = Math.min(70, colW * 0.5), x = cx(j) - bw / 2;
          const hp = pi[j] * bh, h0 = pi0[j] * bh;
          const col = id ? C.b : j === S.a ? (S.A >= 0 ? C.a : RED()) : C.b;
          V.box(g, x, base - hp, bw, Math.max(2, hp), { r: 5, fill: V.alpha(col, 0.55), stroke: col, lw: 1.4 });
          if (!id) { g.setLineDash([4, 3]); V.rr(g, x - 3, base - h0, bw + 6, Math.max(2, h0), 5); g.strokeStyle = V.alpha(C.ink, 0.55); g.lineWidth = 1.2; g.stroke(); g.setLineDash([]); }
          V.text(g, fmt(pi[j], 3), cx(j), base - Math.max(hp, h0) - 6, { size: 12, font: 'mono', weight: 700, color: C.ink, align: 'center' });
          if (nF) {
            // frequência empírica do token: marca ao lado da barra, para não cruzar o rótulo de π
            const f = CNT[nF][j] / nF, yf = base - f * bh, xm = x + bw + 3;
            g.beginPath(); g.moveTo(xm, yf); g.lineTo(xm + 7, yf - 5); g.lineTo(xm + 7, yf + 5); g.closePath(); g.fillStyle = C.d; g.fill();
            V.line(g, x, yf, x + bw, yf, { color: V.alpha(C.d, 0.7), lw: 1.2, dash: [3, 3] });
            if (!narrow) V.text(g, fmt(f, 2), xm + 10, yf + 4, { size: 11, font: 'mono', weight: 700, color: C.d }); // no celular, as contagens ficam nos rótulos das linhas
          }
        }
        V.line(g, L, base, w - pad, base, { color: V.alpha(C.muted, 0.5), lw: 1 });
        if (!id) {
          // linha 4: gradiente e Δz
          const y4 = base + 18, y5 = y4 + 38, cwc = Math.min(96, colW * 0.7);
          lab('eₐ − π', y4 + 16);
          lab(narrow ? 'Δz = ηA(…)' : `Δz = ηA(eₐ − π)`, y5 + 16, C.d);
          for (let j = 0; j < 3; j++) {
            cell(g, cx(j) - cwc / 2, y4, cwc, 25, sgn(grad[j], 3), grad[j] >= 0 ? C.a : C.b, { a: 0.15 + 0.5 * Math.abs(grad[j]) });
            cell(g, cx(j) - cwc / 2, y5, cwc, 25, sgn(dz[j], 3), C.d, { a: 0.15 + 0.25 * Math.min(1, Math.abs(dz[j])), weight: 700 });
          }
          V.text(g, `each cell above × ηA = ${sgn(S.eta * S.A, 2)}`, L + (w - pad - L) / 2, y5 + 42, { size: 11.5, font: 'mono', color: C.d, align: 'center', maxW: w - L - pad });
          // partículas subindo de Δz para os logits durante o movimento
          if (ph >= 1 && ph < 3) {
            for (let j = 0; j < 3; j++) {
              const up = dz[j] >= 0;
              for (let q = 0; q < 4; q++) {
                const f = ((ph - 1) * 0.9 + q / 4) % 1;
                const yy = V.lerp(y5, y2 + 26, f);
                V.dot(g, cx(j) + cwc / 2 + 10, yy, 2.4, V.alpha(up ? C.a : RED(), 0.8 * (1 - f) + 0.2));
              }
              V.arrow(g, cx(j) + cwc / 2 + 10, y5, cx(j) + cwc / 2 + 10, y2 + 30, { color: V.alpha(C.d, 0.35), lw: 1, head: 5 });
            }
          }
          return;
        }
        // ---------- passos 2–4: a identidade ∇J = Σ π(a) R(a)(eₐ − π) = E[R ∇log π]
        const cw2 = Math.min(96, colW * 0.72), rh = narrow ? 21 : 23, rg = narrow ? 5 : 6, fs = narrow ? 11 : 11.5;
        const yb = base + 14;
        const rowY = (k) => yb + k * (rh + rg);
        const rowLab = (k, s, col, wt) => V.text(g, s, L - 10, rowY(k) + rh / 2 + 4, { size: narrow ? 10.5 : 12, font: 'mono', weight: wt || 400, color: col || C.muted, align: 'right', maxW: L - 14 });
        const drawRow = (k, vals, col, o = {}) => {
          for (let j = 0; j < 3; j++) cell(g, cx(j) - cw2 / 2, rowY(k), cw2, rh, sgn(vals[j], 3), col || (vals[j] >= 0 ? C.a : C.b), Object.assign({ a: 0.1 + 0.45 * Math.min(1, Math.abs(vals[j])), size: fs }, o));
        };
        const foot = (k, s, col) => V.text(g, s, L + (w - pad - L) / 2, rowY(k) + rh / 2 + 4, { size: narrow ? 10.5 : 11.5, font: 'mono', color: col || C.muted, align: 'center', maxW: w - L - pad });
        const dashed = (k) => { g.setLineDash([4, 3]); g.strokeStyle = C.d; g.lineWidth = 1.4; for (let j = 0; j < 3; j++) { V.rr(g, cx(j) - cw2 / 2 - 3, rowY(k) - 3, cw2 + 6, rh + 6, 6); g.stroke(); } g.setLineDash([]); };
        const fade = (a, fn) => { if (a <= 0) return; g.save(); g.globalAlpha = Math.min(1, a); fn(); g.restore(); };
        if (mode === 1) {
          const ap = (k) => V.clamp((el - 0.3 - k * 0.9) / 0.6, 0, 1);
          fade(ap(0), () => foot(0, narrow ? `J = Σ π(a)R(a) = ${fmt(JR, 3)}` : `J = Σₐ π(a)R(a) = ${PI0.map((p, j) => fmt(p, 3) + '·' + fmt(RW[j], 1)).join(' + ')} = ${fmt(JR, 3)}`, C.ink));
          fade(ap(1), () => { rowLab(1, 'R − J'); drawRow(1, RW.map((r) => r - JR)); });
          fade(ap(2), () => { rowLab(2, narrow ? 'π(R − J)' : '∂J/∂z = π(R − J)', C.d, 700); drawRow(2, EX, C.d, { weight: 700, a: 0.3, stroke: C.d, lw: 1.8 }); });
          fade(ap(3), () => foot(3, narrow ? 'sum = 0: only redistributes mass' : 'sum = 0: the gradient only redistributes mass among the tokens', C.muted));
          return;
        }
        if (mode === 2) {
          const T0 = 0.6, TS = 1.2, TF = 0.9;
          for (let a = 0; a < 3; a++) {
            const f = V.clamp((el - T0 - a * TS) / TF, 0, 1), moving = f > 0 && f < 1;
            rowLab(a, narrow ? `“${LBL[a]}” ×${fmt(PI0[a], 3)}` : `“${LBL[a]}” × π = ${fmt(PI0[a], 3)}`, moving ? C.d : C.muted, moving ? 700 : 400);
            drawRow(a, ROWS[a], null, moving ? { stroke: C.d, lw: 1.8 } : {});
          }
          const done = [0, 1, 2].filter((a) => el >= T0 + a * TS + TF);
          const part = [0, 1, 2].map((j) => done.reduce((s, a) => s + PI0[a] * ROWS[a][j], 0));
          const full = done.length === 3;
          rowLab(3, narrow ? 'Σ π(a)·row' : 'Σₐ π(a) · row', C.d, 700);
          drawRow(3, part, C.d, { weight: 700, a: 0.3, stroke: C.d, lw: full ? 1.8 : 1 });
          if (full) dashed(3);
          // cópia da linha, já pesada por π(a), descendo para a soma
          for (let a = 0; a < 3; a++) {
            const f = V.clamp((el - T0 - a * TS) / TF, 0, 1);
            if (f <= 0 || f >= 1) continue;
            const yy = V.lerp(rowY(a), rowY(3), V.ease.inOut(f));
            for (let j = 0; j < 3; j++) {
              const gx = cx(j) - cw2 / 2 + 6, gy = yy + 5;
              V.box(g, gx, gy, cw2, rh, { r: 4, fill: C.panel || C.bg });
              cell(g, gx, gy, cw2, rh, sgn(PI0[a] * ROWS[a][j], 3), C.d, { a: 0.45, size: fs, weight: 700, stroke: C.d, lw: 1.6 });
            }
          }
          foot(4, full ? (narrow ? '= π(R − J), the exact one ✓' : '= π(R − J): the same exact gradient as step 2 ✓') : (narrow ? 'row = R(a)(eₐ − π)' : 'each row: R(a)(eₐ − π) = score × ∇log π(a)'), full ? C.a : C.muted);
          return;
        }
        // mode 3: Monte Carlo
        const n = nF, lastA = SMP[n - 1], mean = CUM[n].map((v) => v / n);
        for (let a = 0; a < 3; a++) {
          const isL = a === lastA;
          rowLab(a, `“${LBL[a]}” · ${CNT[n][a]}×`, isL ? C.d : C.muted, isL ? 700 : 400);
          drawRow(a, ROWS[a], null, isL ? { stroke: C.d, lw: 2 } : {});
        }
        rowLab(3, narrow ? `mean n=${n}` : `mean (n = ${n})`, C.d, 700);
        drawRow(3, mean, C.d, { weight: 700, a: 0.3, stroke: C.d, lw: 1.6 });
        rowLab(4, narrow ? 'exact' : 'exact π(R − J)');
        drawRow(4, EX, C.muted, { a: 0.08 });
        dashed(4);
        // a próxima amostra cai da barra do token sorteado até a sua linha (só enquanto o ritmo é lento)
        const fr = nRate(el) - Math.floor(nRate(el));
        if (el < 2.2 && n < NS) {
          const a = SMP[n];
          const yy = V.lerp(base - PI0[a] * bh / 2, rowY(a) + rh / 2, V.ease.inOut(fr));
          V.dot(g, cx(a), yy, 5, C.d);
        }
        const err = Math.hypot(mean[0] - EX[0], mean[1] - EX[1], mean[2] - EX[2]);
        foot(5, `error ‖mean − exact‖ = ${fmt(err, 3)}`, err < 0.03 ? C.a : C.muted);
      });
    })();

    /* ================================================ FIG 4 — REINFORCE ao vivo */
    (function () {
      const fig = root.querySelector('#fig-reinforce');
      // no celular o painel empilha 4 blocos; a altura acompanha o número de linhas da matriz (N)
      const stage = fig.querySelector('.la-fig-stage');
      const fitH = () => { const want = (fig.clientWidth < 600 ? 595 + 25 * (S.N + 1) : 500) + 'px'; if (stage.style.height !== want) stage.style.height = want; };
      const R = [1.0, 0.0, 0.5, 0.9], LBL = ['A', 'B', 'C', 'D'];
      const Z0 = [-0.3, 0.8, 0.4, -0.6];
      const ETA = 0.6, DT = 0.3, MAXSTEP = 100;
      const MODES = { none: 'none', rloo: 'RLOO', v: 'ideal critic' };
      const S = { base: 'none', N: 4, c: 0, z: Z0.slice(), step: 0, hist: [], curve: [], ghosts: {}, last: [], acc: 0 };
      fitH();
      { const ro = new ResizeObserver(fitH); ro.observe(fig); if (LA.onCleanup) LA.onCleanup(() => ro.disconnect()); }
      let rnd = V.rng(5404);
      const J0 = (pi) => pi.reduce((s, p, i) => s + p * R[i], 0);
      /** uma estimativa REINFORCE com N amostras na política z */
      function estimate(z, mode, rng) {
        const pi = softmax(z);
        const Rc = R.map((x) => x + S.c);
        const J = pi.reduce((s, p, i) => s + p * Rc[i], 0);
        const smp = []; for (let i = 0; i < S.N; i++) smp.push(sampleCat(pi, rng()));
        const rows = smp.map((a, i) => {
          let b = 0;
          if (mode === 'rloo') { let s = 0; smp.forEach((o, j) => { if (j !== i) s += Rc[o]; }); b = s / (S.N - 1); }
          else if (mode === 'v') b = J;
          const wgt = Rc[a] - b;
          return { a, w: wgt, g: pi.map((p, j) => wgt * ((j === a ? 1 : 0) - p)) };
        });
        const gh = [0, 1, 2, 3].map((j) => rows.reduce((s, r) => s + r.g[j], 0) / S.N);
        const gt = pi.map((p, i) => p * (Rc[i] - J));
        const err = gh.reduce((s, v2, j) => s + (v2 - gt[j]) * (v2 - gt[j]), 0);
        return { rows, gh, gt, err };
      }
      function refGhosts() {
        Object.keys(MODES).forEach((m) => {
          const r = V.rng(9090);
          let z = Z0.slice();
          const cv = [J0(softmax(z))];
          for (let k = 0; k < MAXSTEP; k++) { const e = estimate(z, m, r); z = z.map((zz, j) => zz + ETA * e.gh[j]); cv.push(J0(softmax(z))); }
          S.ghosts[m] = cv;
        });
      }
      function restart() {
        S.z = Z0.slice(); S.step = 0; S.curve = [J0(softmax(Z0))]; S.acc = 0;
        rnd = V.rng(5404 + Math.floor(Math.random() * 1000));
        // nuvem pré-carregada: 40 estimativas na política inicial (sem treinar)
        S.hist = [];
        const r = V.rng(4242);
        for (let k = 0; k < 40; k++) { const e = estimate(S.z, S.base, r); S.hist.push({ dx: e.gh[0] - e.gt[0], dy: e.gh[1] - e.gt[1], err: e.err }); S.last = e.rows; }
      }
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Baseline', options: [{ v: 'none', label: 'none' }, { v: 'rloo', label: 'RLOO' }, { v: 'v', label: 'ideal critic' }], value: 'none', onChange: (v) => { S.base = v; restart(); } });
      nocase(UI.slider(ctl, { label: 'N samples', min: 2, max: 8, step: 1, value: 4, onInput: (v) => { S.N = v; fitH(); refGhosts(); restart(); } }));
      nocase(UI.slider(ctl, { label: 'shift c', min: -2, max: 5, step: 0.5, value: 0, fmt: (v) => sgn(v, 1), onInput: (v) => { S.c = v; refGhosts(); restart(); } }));
      UI.button(ctl, 'restart', () => restart());
      const rJ = nocase(UI.readout(ctl, 'current E[R]', '—')), rE = nocase(UI.readout(ctl, 'mean error ‖ĝ − g‖²', '—')), rS = UI.readout(ctl, 'step', '0');
      refGhosts(); restart();
      function doStep() {
        const e = estimate(S.z, S.base, rnd);
        S.hist.push({ dx: e.gh[0] - e.gt[0], dy: e.gh[1] - e.gt[1], err: e.err }); if (S.hist.length > 70) S.hist.shift();
        S.last = e.rows;
        S.z = S.z.map((zz, j) => zz + ETA * e.gh[j]);
        S.step++;
        S.curve.push(J0(softmax(S.z)));
        if (S.step >= MAXSTEP) restart();
      }
      let frame = 0;
      V.fig(fig, (g, w, h, t, dt) => {
        const C = V.C();
        if (!LA.paused) { S.acc += dt; while (S.acc >= DT) { S.acc -= DT; doStep(); } }
        const pi = softmax(S.z);
        if (++frame % 8 === 0) {
          rJ.set(fmt(J0(pi), 3));
          const em = S.hist.reduce((s, x) => s + x.err, 0) / Math.max(1, S.hist.length);
          rE.set(fmt(em, 3), em < 0.05 ? 'ok' : em < 0.3 ? 'warn' : 'bad');
          rS.set(`${S.step} / ${MAXSTEP}`);
        }
        const narrow = w < 640, pad = 12;
        const modeCol = { none: RED(), rloo: C.a, v: C.b };
        let PA, MA, CA, LA2;
        if (!narrow) {
          const cw = w * 0.5;
          PA = { x: pad, y: 8, w: cw - 2 * pad, h: 150 };
          MA = { x: pad, y: 172, w: cw - 2 * pad, h: h - 180 };
          CA = { x: cw + 8, y: 8, w: w - cw - 8 - pad, h: h * 0.5 };
          LA2 = { x: cw + 8, y: h * 0.5 + 24, w: w - cw - 8 - pad, h: h * 0.5 - 30 };
        } else {
          // altura da matriz = cabeçalho + (N + 1) linhas de 22 px, sem banda vazia
          const nrN = Math.max(1, Math.min(8, S.last.length));
          PA = { x: pad, y: 8, w: w - 2 * pad, h: 140 };
          MA = { x: pad, y: 156, w: w - 2 * pad, h: 46 + (nrN + 1) * 25 };
          CA = { x: pad, y: MA.y + MA.h + 8, w: w - 2 * pad, h: 200 };
          LA2 = { x: pad, y: CA.y + CA.h + 18, w: w - 2 * pad, h: h - (CA.y + CA.h + 18) - 10 };
        }
        // política
        V.text(g, 'policy π and scores R (+c)', PA.x, PA.y + 12, { size: 12, weight: 600, color: C.ink });
        // barras escaladas pela maior probabilidade (o rótulo numérico carrega o valor absoluto)
        const gw = PA.w / 4, base = PA.y + PA.h - 22, bh = PA.h - 52, piMax = Math.max(...pi);
        for (let j = 0; j < 4; j++) {
          const x = PA.x + gw * j + gw * 0.2, bw = gw * 0.6, hh = Math.max(2, (pi[j] / piMax) * bh);
          V.box(g, x, base - hh, bw, hh, { r: 4, fill: V.alpha(C.b, 0.55), stroke: C.b });
          V.text(g, fmt(pi[j], 2), x + bw / 2, base - hh - 5, { size: 11, font: 'mono', color: C.ink, align: 'center' });
          V.text(g, `${LBL[j]} · R=${fmt(R[j] + S.c, 1)}`, x + bw / 2, base + 15, { size: 11, font: 'mono', color: R[j] >= 0.8 ? C.a : R[j] < 0.2 ? RED() : C.muted, align: 'center', maxW: gw - 2 });
        }
        // matriz do último lote
        V.text(g, 'last batch: (R − b)(eₐ − π)', MA.x, MA.y + 12, { size: 12, weight: 600, color: C.ink, maxW: MA.w });
        const nr = Math.min(8, S.last.length);
        if (nr) {
          const lw = 96, cwm = Math.min(64, (MA.w - lw) / 4 - 3), chm = Math.min(22, (MA.h - 56) / (nr + 1) - 3);
          const mx0 = MA.x + lw;
          LBL.forEach((l, j) => V.text(g, l, mx0 + j * (cwm + 3) + cwm / 2, MA.y + 32, { size: 11, font: 'mono', color: C.muted, align: 'center' }));
          for (let i = 0; i < nr; i++) {
            const row = S.last[i], y = MA.y + 38 + i * (chm + 3);
            V.text(g, `${LBL[row.a]}  w=${sgn(row.w, 2)}`, mx0 - 6, y + chm / 2 + 4, { size: 11, font: 'mono', color: row.w >= 0 ? C.a : RED(), align: 'right' });
            for (let j = 0; j < 4; j++) {
              const vv = row.g[j];
              cell(g, mx0 + j * (cwm + 3), y, cwm, chm, chm >= 15 ? sgn(vv, 2) : null, vv >= 0 ? C.a : C.b, { a: 0.1 + 0.6 * Math.min(1, Math.abs(vv) / 2), size: 10.5 });
            }
          }
          const y = MA.y + 38 + nr * (chm + 3) + 4;
          V.text(g, 'ĝ = mean', mx0 - 6, y + chm / 2 + 4, { size: 11, font: 'mono', weight: 700, color: C.d, align: 'right' });
          for (let j = 0; j < 4; j++) {
            const m = S.last.reduce((s, r) => s + r.g[j], 0) / S.last.length;
            cell(g, mx0 + j * (cwm + 3), y, cwm, chm, chm >= 15 ? sgn(m, 2) : null, C.d, { a: 0.15 + 0.5 * Math.min(1, Math.abs(m)), weight: 700, size: 10.5 });
          }
        }
        // nuvem
        V.text(g, 'error of each ĝ (components A, B)', CA.x, CA.y + 12, { size: 12, weight: 600, color: C.ink, maxW: CA.w });
        const side = Math.min(CA.w, CA.h - 24), ox = CA.x + CA.w / 2, oy = CA.y + 22 + side / 2, RNG = 1.2;
        const sc = (side / 2 - 6) / RNG;
        V.box(g, ox - side / 2, oy - side / 2, side, side, { r: 8, fill: V.alpha(C.muted, 0.05), stroke: V.alpha(C.muted, 0.25), lw: 1 });
        V.line(g, ox - side / 2, oy, ox + side / 2, oy, { color: C.grid, lw: 1 });
        V.line(g, ox, oy - side / 2, ox, oy + side / 2, { color: C.grid, lw: 1 });
        if (S.hist.length) {
          const rms = Math.sqrt(S.hist.reduce((s, p) => s + p.dx * p.dx + p.dy * p.dy, 0) / S.hist.length);
          g.beginPath(); g.arc(ox, oy, Math.min(side / 2, rms * sc), 0, Math.PI * 2); g.strokeStyle = V.alpha(modeCol[S.base], 0.7); g.lineWidth = 1.5; g.setLineDash([4, 3]); g.stroke(); g.setLineDash([]);
          S.hist.forEach((p, i) => {
            const x = V.clamp(ox + p.dx * sc, ox - side / 2 + 3, ox + side / 2 - 3), y = V.clamp(oy - p.dy * sc, oy - side / 2 + 3, oy + side / 2 - 3);
            V.dot(g, x, y, i === S.hist.length - 1 ? 4.5 : 2.8, V.alpha(modeCol[S.base], 0.3 + 0.6 * (i / S.hist.length)));
          });
        }
        V.dot(g, ox, oy, 4, C.d);
        V.text(g, 'target (error 0)', ox + 7, oy - 7, { size: 11, color: C.d });
        V.text(g, `±${fmt(RNG, 1)}`, ox + side / 2 - 4, oy + side / 2 - 6, { size: 11, font: 'mono', color: C.muted, align: 'right' });
        // curva de aprendizado
        V.text(g, 'expected score E[R] over training', LA2.x, LA2.y + 4, { size: 12, weight: 600, color: C.ink, maxW: LA2.w });
        const lx0 = LA2.x + 30, lx1 = LA2.x + LA2.w - 4, ly0 = LA2.y + 16, ly1 = LA2.y + LA2.h - 18;
        const xs = (i) => lx0 + (i / MAXSTEP) * (lx1 - lx0), ys = (v) => ly1 - v * (ly1 - ly0);
        [0, 0.5, 1].forEach((v) => { V.line(g, lx0, ys(v), lx1, ys(v), { color: C.grid, lw: 1 }); V.text(g, fmt(v, 1), lx0 - 5, ys(v) + 4, { size: 11, font: 'mono', color: C.muted, align: 'right' }); });
        const drawCurve = (arr, col, lw, dash) => { if (arr.length < 2) return; g.beginPath(); arr.forEach((v, i) => { const x = xs(i), y = ys(v); if (i) g.lineTo(x, y); else g.moveTo(x, y); }); g.strokeStyle = col; g.lineWidth = lw; if (dash) g.setLineDash(dash); g.stroke(); g.setLineDash([]); };
        let lgx = lx0;
        Object.keys(MODES).forEach((m) => {
          if (S.ghosts[m]) drawCurve(S.ghosts[m], V.alpha(modeCol[m], m === S.base ? 0.4 : 0.55), 1.4, [4, 3]);
          const s = MODES[m];
          V.dot(g, lgx + 4, ly1 + 14, 4, modeCol[m]); V.text(g, s, lgx + 11, ly1 + 18, { size: 11, weight: m === S.base ? 700 : 400, color: m === S.base ? C.ink : C.muted }); lgx += tw(g, s, 11) + 26;
        });
        drawCurve(S.curve, modeCol[S.base], 2.8);
        V.dot(g, xs(S.curve.length - 1), ys(S.curve[S.curve.length - 1]), 4, modeCol[S.base]);
      });
    })();

    /* ================================================ FIG 5 — GAE */
    (function () {
      const fig = root.querySelector('#fig-gae');
      adaptH(fig, 500, 510);
      const LPO = [-1.2, -0.5, -2.1, -0.3, -0.05], LPR = [-1.4, -0.55, -1.6, -0.45, -0.1], VV = [0.9, 1.1, 1.0, 1.5, 1.8], RM = 2.0, T = 5;
      const S = { lam: 0.95, beta: 0.05, gamma: 1 };
      function calc(lam) {
        const kl = LPO.map((a, i) => a - LPR[i]);
        const r = kl.map((k) => -S.beta * k); r[T - 1] += RM;
        const d = r.map((rv, i) => rv + S.gamma * (i + 1 < T ? VV[i + 1] : 0) - VV[i]);
        const A = new Array(T).fill(0); let nx = 0;
        for (let i = T - 1; i >= 0; i--) { nx = d[i] + S.gamma * lam * nx; A[i] = nx; }
        return { kl, r, d, A, Rh: A.map((a, i) => a + VV[i]) };
      }
      const ctl = UI.controls(fig);
      nocase(UI.slider(ctl, { label: 'λ', min: 0, max: 1, step: 0.05, value: 0.95, fmt: (v) => fmt(v, 2), onInput: (v) => { S.lam = v; upd(); } }));
      nocase(UI.slider(ctl, { label: 'β (KL)', min: 0, max: 0.2, step: 0.01, value: 0.05, fmt: (v) => fmt(v, 2), onInput: (v) => { S.beta = v; upd(); } }));
      nocase(UI.slider(ctl, { label: 'γ', min: 0.9, max: 1, step: 0.01, value: 1, fmt: (v) => fmt(v, 2), onInput: (v) => { S.gamma = v; upd(); } }));
      const rA = nocase(UI.readout(ctl, '\u00C2₁', '—')), rR = nocase(UI.readout(ctl, 'critic target, t=1', '—')), rS = nocase(UI.readout(ctl, 'Σ rₜ', '—'));
      function upd() { const c = calc(S.lam); rA.set(fmt(c.A[0], 4)); rR.set(fmt(c.Rh[0], 4)); rS.set(fmt(c.r.reduce((a, b) => a + b, 0), 4)); }
      upd();
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const narrow = w < 600, pad = 10;
        const c = calc(S.lam), c0 = calc(0), c1 = calc(1);
        const rows = [
          ['log π_old', LPO, 2, C.b, 0], ['log π_ref', LPR, 2, C.b, 0], ['KLₜ (k₁)', c.kl, 2, C.f, 1], ['rₜ', c.r, 4, C.d, 1],
          ['V(sₜ)', VV, 2, C.a, 0], ['δₜ', c.d, 4, C.c, 1], ['\u00C2ₜ', c.A, 4, C.a, 1], ['target \u00C2 + V', c.Rh, 4, C.e || C.a, 0],
        ];
        const lw = narrow ? 82 : 120, cw = Math.min(120, (w - lw - 2 * pad) / T - 4), chh = narrow ? 22 : 24, gap = 4;
        const x0 = narrow ? pad + lw : (w - (lw + T * (cw + gap))) / 2 + lw;
        const y0 = 26;
        // recursão: coluna ativa da direita para a esquerda
        const cyc = 7, ph = t % cyc;
        const act = ph < 5 ? T - 1 - Math.floor(ph) : -1;
        for (let i = 0; i < T; i++) V.text(g, i === T - 1 ? 't=5 EOS' : 't=' + (i + 1), x0 + i * (cw + gap) + cw / 2, y0 - 8, { size: 11.5, font: 'mono', weight: 600, color: i === act ? C.d : C.muted, align: 'center' });
        rows.forEach(([lab, arr, dg, col, signed], r) => {
          const y = y0 + r * (chh + gap);
          V.text(g, lab, x0 - 8, y + chh / 2 + 4, { size: narrow ? 11 : 12, font: 'mono', color: r >= 5 ? C.ink : C.muted, weight: r === 6 ? 700 : 400, align: 'right' });
          const mx = Math.max(...arr.map(Math.abs), 1e-9);
          arr.forEach((v, i) => {
            const isA = r === 6;
            const pending = isA && act >= 0 && i < act;
            g.globalAlpha = pending ? 0.35 : 1;
            cell(g, x0 + i * (cw + gap), y, cw, chh, fmt(v, cw < 52 ? Math.min(dg, 3) : dg), signed && v < 0 ? RED() : col, { a: 0.08 + 0.4 * Math.abs(v) / mx, weight: isA ? 700 : 500, size: narrow ? 10.5 : 11.5, stroke: i === act ? C.d : undefined, lw: i === act ? 1.8 : 1 });
            g.globalAlpha = 1;
          });
        });
        // seta da recursão
        if (act >= 0 && act < T - 1) {
          const yA = y0 + 6 * (chh + gap) + chh / 2;
          const xa = x0 + (act + 1) * (cw + gap) + 4, xb = x0 + act * (cw + gap) + cw - 4;
          V.arrow(g, xa, yA, xb, yA, { color: C.d, lw: 2.2, head: 7, curve: -10 });
        }
        // barras de Â
        const by0 = y0 + rows.length * (chh + gap) + 36, by1 = h - 24;
        const all = c.A.concat(c0.A, c1.A); const amax = Math.max(1.2, ...all), amin = Math.min(0, ...all) - 0.1;
        const yb = (v) => by1 - ((v - amin) / (amax - amin)) * (by1 - by0);
        V.text(g, `\u00C2ₜ with λ = ${fmt(S.lam, 2)}`, pad, by0 - 20, { size: 12, weight: 600, color: C.ink });
        if (act >= 0) {
          const t1 = act + 1;
          const eq = act === T - 1 ? `now: \u00C2${t1} = δ${t1} = ${fmt(c.d[act], 4)}  (last token)` : `now: \u00C2${t1} = δ${t1} + γλ·\u00C2${t1 + 1} = ${fmt(c.d[act], 4)} + ${fmt(S.gamma * S.lam, 2)} × ${fmt(c.A[act + 1], 4)} = ${fmt(c.A[act], 4)}`;
          V.text(g, eq, pad, by0 - 4, { size: narrow ? 10.5 : 11.5, font: 'mono', weight: 600, color: C.d, maxW: w - 2 * pad });
        }
        V.line(g, x0, yb(0), x0 + T * (cw + gap), yb(0), { color: V.alpha(C.muted, 0.6), lw: 1 });
        for (let i = 0; i < T; i++) {
          const x = x0 + i * (cw + gap), bw = cw * 0.5, bx = x + cw * 0.25;
          const v = c.A[i], top = Math.min(yb(v), yb(0)), hh = Math.abs(yb(v) - yb(0));
          V.box(g, bx, top, bw, Math.max(1.5, hh), { r: 3, fill: V.alpha(v >= 0 ? C.a : RED(), 0.6), stroke: v >= 0 ? C.a : RED(), lw: 1.2 });
          V.dot(g, x + cw * 0.12, yb(c0.A[i]), 4, C.d);
          V.dot(g, x + cw * 0.88, yb(c1.A[i]), 4, C.b);
        }
        const lgY = by1 + 16;
        V.dot(g, x0 + 4, lgY - 4, 4, C.d); V.text(g, 'λ = 0 (TD)', x0 + 12, lgY, { size: 11, color: C.muted });
        V.dot(g, x0 + 100, lgY - 4, 4, C.b); V.text(g, 'λ = 1 (Monte Carlo)', x0 + 108, lgY, { size: 11, color: C.muted });
      });
    })();

    /* ================================================ FIG 6 — farol */
    (function () {
      const fig = root.querySelector('#fig-farol');
      adaptH(fig, 400, 470);
      const LAMS = [0.9, 0.95, 0.99, 0.995, 0.999, 1], LENS = [50, 100, 500, 1000, 2000, 8000, 16000];
      const S = { lam: 0.95, L: 500, adapt: false };
      const lamEff = () => (S.adapt ? 1 - 1 / (1.5 * S.L) : S.lam);
      const lg10w = (d) => { const l = lamEff(); return l >= 1 ? 0 : d * Math.log10(l); };
      const ctl = UI.controls(fig);
      const segL = nocase(UI.seg(ctl, { label: 'λ', options: LAMS.map((v) => ({ v, label: fmt(v, v >= 0.99 && v < 1 ? 3 : 2).replace(/0+$/, '').replace(/\.$/, '') })), value: 0.95, onChange: (v) => { S.lam = v; upd(); } }));
      nocase(UI.slider(ctl, { label: 'length l', min: 0, max: LENS.length - 1, step: 1, value: 2, fmt: (v) => LENS[v].toLocaleString('en-US'), onInput: (v) => { S.L = LENS[v]; upd(); } }));
      nocase(UI.toggle(ctl, { label: 'adaptive λ (α = 1.5)', value: false, onChange: (v) => { S.adapt = v; segL.el.style.opacity = v ? 0.4 : 1; upd(); } }));
      const rL = nocase(UI.readout(ctl, 'effective λ', '—')), rW = nocase(UI.readout(ctl, 'weight at 1st token', '—')), rD = nocase(UI.readout(ctl, 'distance to 1%', '—'));
      function upd() {
        const l = lamEff();
        rL.set(fmt(l, l > 0.999 ? 6 : 4));
        const lw = lg10w(S.L);
        rW.set(fmtLog(lw, 1), lw > -1 ? 'ok' : lw > -3 ? 'warn' : 'bad');
        rD.set(l >= 1 ? '∞' : Math.round(Math.log(0.01) / Math.log(l)).toLocaleString('en-US') + ' tokens');
      }
      upd();
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const pad = 12, narrow = w < 560;
        const L0 = pad + (narrow ? 44 : 56), R0 = w - pad - 6;
        const xOf = (pos) => L0 + (pos / S.L) * (R0 - L0); // pos: 0 = 1º token, L = EOS
        // faixa da resposta
        const sy = 28, sh = 30, NS = 80;
        V.text(g, `${S.L.toLocaleString('en-US')}-token response`, L0, sy - 8, { size: 12, weight: 600, color: C.ink });
        V.text(g, 'EOS · R', R0, sy - 8, { size: 11.5, font: 'mono', weight: 700, color: C.d, align: 'right' });
        for (let i = 0; i < NS; i++) {
          const pos = (i + 0.5) / NS * S.L, d = S.L - pos;
          const lw = lg10w(d), bright = V.clamp(1 + lw / 6, 0, 1);
          g.fillStyle = V.alpha(C.d, 0.06 + 0.88 * bright);
          const x = L0 + (i / NS) * (R0 - L0);
          g.fillRect(x, sy, (R0 - L0) / NS + 0.5, sh);
        }
        g.strokeStyle = V.alpha(C.muted, 0.4); g.lineWidth = 1; g.strokeRect(L0, sy, R0 - L0, sh);
        // pulso viajando do EOS para trás
        const per = 3.2, f = (t % per) / per, ppos = S.L * (1 - f), pd = S.L - ppos, plw = lg10w(pd);
        const pa = V.clamp(1 + plw / 8, 0, 1);
        if (pa > 0.01) V.glow(g, xOf(ppos), sy + sh / 2, 26, C.d, pa);
        // gráfico log
        const gy0 = sy + sh + 34, gy1 = h - 30, LMIN = -15;
        const yOf = (lw) => gy0 + (Math.max(lw, LMIN) / LMIN) * (gy1 - gy0);
        V.text(g, 'weight λᵈ (log)', pad, gy0 - 12, { size: 11.5, color: C.muted });
        [0, -3, -6, -9, -12, -15].forEach((e) => {
          V.line(g, L0, yOf(e), R0, yOf(e), { color: C.grid, lw: 1 });
          V.text(g, e === 0 ? '1' : '10' + sup(e), L0 - 6, yOf(e) + 4, { size: 11, font: 'mono', color: C.muted, align: 'right' });
        });
        V.line(g, L0, yOf(-2), R0, yOf(-2), { color: V.alpha(RED(), 0.5), lw: 1, dash: [4, 3] });
        V.text(g, '1%', R0 - 2, yOf(-2) - 4, { size: 11, color: RED(), align: 'right' });
        g.beginPath();
        for (let i = 0; i <= 200; i++) { const pos = (i / 200) * S.L, x = xOf(pos), y = yOf(lg10w(S.L - pos)); if (i) g.lineTo(x, y); else g.moveTo(x, y); }
        g.strokeStyle = C.d; g.lineWidth = 2.6; g.stroke();
        // marcadores de distância
        const placed = [];
        [50, 500, 8000].forEach((d) => {
          if (d > S.L) return;
          const x = xOf(S.L - d), lw = lg10w(d), y = yOf(lw);
          V.dot(g, x, y, 4, C.ink);
          const lab = `d=${d.toLocaleString('en-US')}: ${fmtLog(lw, 1)}`;
          const right = x < L0 + 150;
          let ty = Math.max(gy0 + 14, Math.min(gy1 - 8, y - 12));
          const lwid = tw(g, lab, 11, 'mono', 600) + 10;
          const bx = right ? x + 6 : x - 6 - lwid;
          for (let k = 0; k < 4 && placed.some((b2) => bx < b2.x + b2.w && bx + lwid > b2.x && Math.abs(ty - b2.y) < 19); k++) ty += 20;
          placed.push({ x: bx, y: ty, w: lwid });
          V.box(g, bx, ty - 12, lwid, 17, { r: 4, fill: V.alpha(C.bg || '#000', 0.85), stroke: V.alpha(C.d, 0.5), lw: 1 });
          V.text(g, lab, bx + 5, ty, { size: 11, font: 'mono', weight: 600, color: C.ink });
        });
        // eixo x
        V.text(g, '1st token', L0, gy1 + 16, { size: 11, color: C.muted });
        V.text(g, 'end (EOS)', R0, gy1 + 16, { size: 11, color: C.muted, align: 'right' });
        if (lg10w(S.L) < LMIN) V.text(g, `at 1st token: ${fmtLog(lg10w(S.L), 1)} (below the floor)`, L0 + 6, gy1 - 6, { size: 11, font: 'mono', color: RED(), maxW: R0 - L0 - 10 });
      });
    })();

    /* ================================================ FIG 7 — importance sampling */
    (function () {
      const fig = root.querySelector('#fig-is');
      adaptH(fig, 500, 570);
      const TAB = { pf: [0.3, 0.5, 0.02, 0.9], pv: [0.25, 0.52, 0.004, 0.91] };
      const S = { data: 'tab', mode: 'tis', sigma: 0.4, fp8: false, C: 2, seed: 1, pf: TAB.pf.slice(), pv: TAB.pv.slice(), disp: null };
      const BETA_ICE = 2;
      function gen() {
        if (S.data === 'tab') { S.pf = TAB.pf.slice(); S.pv = TAB.pv.slice(); return; }
        const r = V.rng(500 + S.seed);
        S.pf = []; S.pv = [];
        for (let i = 0; i < 8; i++) {
          const rare = r() < 0.3;
          const pf = rare ? Math.pow(10, -3 + 1.6 * r()) : 0.2 + 0.75 * r();
          const sig = S.sigma * (0.25 + 1.4 * (1 - pf)) * (S.fp8 ? 2 : 1);
          let pv = pf * Math.exp(sig * r.normal());
          pv = V.clamp(pv, 1e-4, 0.995);
          S.pf.push(pf); S.pv.push(pv);
        }
      }
      const weights = () => {
        const rho = S.pf.map((p, i) => p / S.pv[i]);
        const prod = rho.reduce((a, b) => a * b, 1);
        let wv;
        if (S.mode === 'seq') wv = rho.map(() => prod);
        else if (S.mode === 'tok') wv = rho.slice();
        else if (S.mode === 'tis') wv = rho.map((x) => Math.min(x, S.C));
        else wv = rho.map((x) => (x >= 1 / BETA_ICE && x <= BETA_ICE ? x : 0));
        return { rho, prod, wv };
      };
      const fp = (x) => (x < 0.001 ? fmtLog(Math.log10(x), 1) : fmt(x, x < 0.01 ? 3 : 2));
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Data', options: [{ v: 'tab', label: 'table' }, { v: 'rand', label: 'random' }], value: 'tab', onChange: (v) => { S.data = v; grp.show(v === 'rand'); gen(); upd(); } });
      UI.seg(ctl, { label: 'Correction', options: [{ v: 'seq', label: 'Π sequence' }, { v: 'tok', label: 'per token' }, { v: 'tis', label: 'TIS' }, { v: 'ice', label: 'IcePop' }], value: 'tis', onChange: (v) => { S.mode = v; upd(); } });
      nocase(UI.slider(ctl, { label: 'C (TIS)', min: 1, max: 5, step: 0.5, value: 2, fmt: (v) => fmt(v, 1), onInput: (v) => { S.C = v; upd(); } }));
      const grp = UI.group(ctl);
      nocase(UI.slider(grp.el, { label: 'mismatch σ', min: 0, max: 1, step: 0.05, value: 0.4, fmt: (v) => fmt(v, 2), onInput: (v) => { S.sigma = v; gen(); upd(); } }));
      UI.toggle(grp.el, { label: 'FP8 rollout', value: false, onChange: (v) => { S.fp8 = v; gen(); upd(); } });
      UI.button(grp.el, 'new sequence', () => { S.seed++; gen(); upd(); });
      grp.show(false);
      const rP = nocase(UI.readout(ctl, 'Π ρₜ', '—')), rM = UI.readout(ctl, 'largest weight used', '—'), rZ = UI.readout(ctl, 'zeroed tokens', '—');
      function upd() {
        const W = weights();
        rP.set(W.prod >= 100 || W.prod < 1e-3 ? fmtLog(Math.log10(W.prod), 2) : fmt(W.prod, 4), W.prod > 3 || W.prod < 0.33 ? 'bad' : 'ok');
        const mx = Math.max(...W.wv);
        rM.set(mx >= 1e4 ? fmtLog(Math.log10(mx), 1) : fmt(mx, 3), mx > 3 ? 'bad' : 'ok');
        rZ.set(String(W.wv.filter((x) => x === 0).length));
      }
      gen(); upd();
      V.fig(fig, (g, w, h, t, dt) => {
        const C = V.C();
        const narrow = w < 600, pad = 12;
        const W = weights();
        const n = S.pf.length;
        if (!S.disp || S.disp.length !== n) S.disp = W.wv.map((x) => Math.log10(Math.max(1e-6, x)));
        S.disp = S.disp.map((d, i) => { const target = W.wv[i] > 0 ? Math.log10(W.wv[i]) : -2.2; return d + (target - d) * Math.min(1, dt * 6); });
        // motores
        const ey = 8, eh = 40, ew = Math.min(260, (w - 3 * pad) / 2);
        V.box(g, pad, ey, ew, eh, { r: 9, fill: V.alpha(C.b, 0.14), stroke: C.b, lw: 1.5 });
        V.text(g, 'vLLM · sampler', pad + 10, ey + 17, { size: 12, weight: 700, color: C.ink, maxW: ew - 16 });
        V.text(g, S.fp8 && S.data === 'rand' ? 'FP8, variable batch' : 'BF16, variable batch', pad + 10, ey + 33, { size: 11, font: 'mono', color: C.b, maxW: ew - 16 });
        const ex2 = w - pad - ew;
        V.box(g, ex2, ey, ew, eh, { r: 9, fill: V.alpha(C.a, 0.14), stroke: C.a, lw: 1.5 });
        V.text(g, 'trainer · FSDP', ex2 + 10, ey + 17, { size: 12, weight: 700, color: C.ink, maxW: ew - 16 });
        V.text(g, 'BF16, same weights θ', ex2 + 10, ey + 33, { size: 11, font: 'mono', color: C.a, maxW: ew - 16 });
        // fluxo de tokens
        for (let q = 0; q < 6; q++) {
          const f = ((t * 0.35) + q / 6) % 1;
          const x = V.lerp(pad + ew, ex2, f), y = ey + eh / 2 + Math.sin(f * Math.PI * 2 + q) * 5;
          V.dot(g, x, y, 2.6, V.alpha(C.d, 0.3 + 0.6 * Math.sin(f * Math.PI)));
        }
        // colunas de tokens: barras log-prob
        const lw0 = narrow ? 50 : 86;
        const x0 = pad + lw0, colW = (w - pad - x0) / n;
        const by0 = ey + eh + 30, by1 = by0 + (narrow ? 150 : 130);
        const yP = (p) => by1 - ((Math.log10(p) + 3) / 3) * (by1 - by0);
        V.text(g, 'π (log)', pad, by0 - 12, { size: 11, font: 'mono', color: C.muted });
        [1, 0.1, 0.01, 0.001].forEach((p) => { V.line(g, x0, yP(p), w - pad, yP(p), { color: C.grid, lw: 1 }); V.text(g, p >= 0.01 ? fmt(p, p >= 1 ? 0 : p >= 0.1 ? 1 : 2) : '0.001', x0 - 5, yP(p) + 4, { size: 10.5, font: 'mono', color: C.muted, align: 'right' }); });
        for (let i = 0; i < n; i++) {
          const cx = x0 + colW * (i + 0.5), bw = Math.min(18, colW * 0.28);
          const hv = by1 - yP(Math.max(1e-3, S.pv[i])), hf = by1 - yP(Math.max(1e-3, S.pf[i]));
          V.box(g, cx - bw - 1, by1 - hv, bw, Math.max(1, hv), { r: 3, fill: V.alpha(C.b, 0.55), stroke: C.b, lw: 1 });
          V.box(g, cx + 1, by1 - hf, bw, Math.max(1, hf), { r: 3, fill: V.alpha(C.a, 0.55), stroke: C.a, lw: 1 });
          if (colW > 110) {
            V.text(g, fp(S.pv[i]), cx - bw - 5, by1 - hv + 10, { size: 11, font: 'mono', weight: 600, color: C.b, align: 'right' });
            V.text(g, fp(S.pf[i]), cx + bw + 5, by1 - hf + 10, { size: 11, font: 'mono', weight: 600, color: C.a, align: 'left' });
          } else if (colW > 56) {
            V.text(g, fp(S.pv[i]), cx, by1 + 28, { size: 10.5, font: 'mono', weight: 600, color: C.b, align: 'center' });
            V.text(g, fp(S.pf[i]), cx, by1 + 41, { size: 10.5, font: 'mono', weight: 600, color: C.a, align: 'center' });
          }
          V.text(g, (n <= 4 ? 'token ' : '') + (i + 1), cx, by1 + 14, { size: 11, font: 'mono', color: C.muted, align: 'center' });
        }
        // razões
        const ry = by1 + (colW <= 110 && colW > 56 ? 50 : 24), rh = 24;
        V.text(g, 'ρ', pad, ry + 16, { size: 12, font: 'mono', weight: 700, color: C.ink });
        for (let i = 0; i < n; i++) {
          const cx = x0 + colW * (i + 0.5), cw = Math.min(84, colW - 6);
          const rv = W.rho[i], bad = rv > 2 || rv < 0.5;
          cell(g, cx - cw / 2, ry, cw, rh, rv >= 100 ? fmtLog(Math.log10(rv), 0) : fmt(rv, colW > 70 ? 3 : 2), bad ? RED() : C.d, { a: bad ? 0.35 : 0.15, size: colW > 50 ? 11 : 10 });
        }
        // pesos usados (escala log centrada em 1)
        const wy0 = ry + rh + 30, wy1 = h - 26, wmid = (wy0 + wy1) / 2, span = (wy1 - wy0) / 2;
        const LGM = 1.2; // ±1,2 décadas
        const yW = (lg) => wmid - V.clamp(lg / LGM, -1, 1) * span;
        const titles = { seq: 'weight = Π ρ over the whole sequence', tok: 'weight = ρₜ (per token)', tis: `weight = min(ρₜ, C = ${fmt(S.C, 1)})`, ice: 'IcePop: ρₜ if 0.5 ≤ ρₜ ≤ 2, else 0' };
        V.text(g, titles[S.mode], pad, wy0 - 10, { size: 12, weight: 600, color: C.ink, maxW: w - 2 * pad });
        V.line(g, x0, yW(0), w - pad, yW(0), { color: V.alpha(C.ink, 0.5), lw: 1 });
        V.text(g, '1', x0 - 5, yW(0) + 4, { size: 11, font: 'mono', color: C.muted, align: 'right' });
        V.text(g, '10', x0 - 5, yW(1) + 4, { size: 11, font: 'mono', color: C.muted, align: 'right' });
        V.text(g, '0.1', x0 - 5, yW(-1) + 4, { size: 11, font: 'mono', color: C.muted, align: 'right' });
        if (S.mode === 'tis') { V.line(g, x0, yW(Math.log10(S.C)), w - pad, yW(Math.log10(S.C)), { color: C.d, lw: 1.4, dash: [5, 3] }); V.text(g, 'C', w - pad, yW(Math.log10(S.C)) - 4, { size: 11, font: 'mono', weight: 700, color: C.d, align: 'right' }); }
        if (S.mode === 'ice') { g.fillStyle = V.alpha(C.a, 0.1); g.fillRect(x0, yW(Math.log10(2)), w - pad - x0, yW(Math.log10(0.5)) - yW(Math.log10(2))); V.text(g, '[0.5, 2]', w - pad, yW(Math.log10(2)) - 4, { size: 11, font: 'mono', color: C.a, align: 'right' }); }
        for (let i = 0; i < n; i++) {
          const cx = x0 + colW * (i + 0.5), bw = Math.min(36, colW * 0.5);
          const d = S.disp[i], zero = W.wv[i] === 0;
          if (zero) {
            V.text(g, '0', cx, wmid + 4, { size: 13, font: 'mono', weight: 700, color: RED(), align: 'center' });
            V.text(g, '✕', cx, wmid - 14, { size: 13, weight: 700, color: RED(), align: 'center' });
            continue;
          }
          const top = Math.min(yW(d), yW(0)), hh = Math.abs(yW(d) - yW(0));
          const over = Math.abs(d) > LGM;
          const col = S.mode === 'seq' ? RED() : d > Math.log10(2) + 1e-6 || d < Math.log10(0.5) - 1e-6 ? C.c : C.a;
          V.box(g, cx - bw / 2, top, bw, Math.max(1.5, hh), { r: 3, fill: V.alpha(col, 0.55), stroke: col, lw: 1.2 });
          if (over) V.text(g, d > 0 ? '▲' : '▼', cx, d > 0 ? top - 3 : top + hh + 12, { size: 11, color: col, align: 'center' });
          if (colW > 44) V.text(g, fmt(Math.pow(10, d), 2), cx + bw / 2 + 3, Math.max(wy0 + 8, Math.min(wy1, yW(d) + (d >= 0 ? 4 : 12))), { size: 10.5, font: 'mono', color: C.ink });
        }
      });
    })();

    /* ================================================ FIG 8 — atores assíncronos */
    (function () {
      const fig = root.querySelector('#fig-async');
      adaptH(fig, 380, 350);
      const NA = 4, U = 0.9, TMAX = 400, WIN = 26;
      const S = { mode: 'sync', tail: 0.5, sim: null };
      function lenFn(seed) { const r = V.rng(seed); return () => 1.4 + 1.2 * r() + S.tail * 3.2 * -Math.log(1 - r() * 0.98); }
      function simulate() {
        const segs = [], idles = [], ups = [], sends = [];
        let version = 0;
        if (S.mode === 'sync') {
          const L = lenFn(81);
          let t0 = 0;
          while (t0 < TMAX) {
            const ls = []; for (let i = 0; i < NA; i++) ls.push(L());
            const B = t0 + Math.max(...ls);
            for (let i = 0; i < NA; i++) { segs.push({ a: i, s: t0, e: t0 + ls[i], v: version }); idles.push({ a: i, s: t0 + ls[i], e: B + U }); sends.push({ a: i, t: B, st: 0 }); }
            ups.push({ s: B, e: B + U, v: ++version });
            t0 = B + U;
          }
        } else if (S.mode === 'a3c') {
          const L = lenFn(82);
          const ev = [];
          for (let i = 0; i < NA; i++) ev.push({ a: i, t: 0.3 * i, v: 0 });
          let guard = 0;
          while (ev.length && guard++ < 5000) {
            ev.sort((x, y) => x.t - y.t);
            const e = ev.shift();
            if (e.t > TMAX) continue;
            const len = L(), end = e.t + len;
            segs.push({ a: e.a, s: e.t, e: end, v: e.v });
            // o push acontece no fim; versão no fim = updates aplicados até lá
            ev.push({ a: e.a, t: end + 0.15, v: -1, pushFrom: e });
          }
          // reconstruir versões em ordem temporal dos pushes
          const pushes = segs.map((sg) => ({ a: sg.a, t: sg.e, s: sg.s })).sort((x, y) => x.t - y.t);
          let ver = 0;
          const verAt = [];
          pushes.forEach((p) => { ver++; verAt.push(p.t); ups.push({ s: p.t, e: p.t + 0.15, v: ver, a: p.a }); });
          segs.forEach((sg) => {
            const pulled = verAt.filter((x) => x <= sg.s).length;
            const atPush = verAt.filter((x) => x < sg.e).length;
            sg.v = pulled; sends.push({ a: sg.a, t: sg.e, st: atPush - pulled });
          });
        } else {
          const L = lenFn(83);
          const traj = [];
          for (let i = 0; i < NA; i++) { let t0 = 0.25 * i; while (t0 < TMAX) { const len = L(); traj.push({ a: i, s: t0, e: t0 + len }); t0 += len; } }
          traj.sort((x, y) => x.e - y.e);
          // aprendiz: consome lotes de 2 trajetórias quando ocioso
          const queue = []; let lt = 0, ver2 = 0, k = 0;
          const verHist = [{ t: 0, v: 0 }];
          const vAt = (tt) => { let v = 0; for (const h2 of verHist) { if (h2.t <= tt) v = h2.v; else break; } return v; };
          while (k < traj.length || queue.length >= 2) {
            const tNext = Math.max(lt, queue.length >= 2 ? 0 : (k < traj.length ? traj[k].e : Infinity));
            while (k < traj.length && traj[k].e <= tNext) { queue.push(traj[k]); k++; }
            if (queue.length < 2) { if (k >= traj.length) break; continue; }
            const start = Math.max(lt, queue[1].e);
            if (start > TMAX) break;
            const batch = queue.splice(0, 2);
            ver2++; ups.push({ s: start, e: start + U, v: ver2 });
            batch.forEach((b) => { b.consume = start; });
            verHist.push({ t: start + U, v: ver2 });
            lt = start + U;
          }
          traj.forEach((tj) => {
            tj.v = vAt(tj.s);
            const vc = tj.consume != null ? vAt(tj.consume) : null;
            segs.push({ a: tj.a, s: tj.s, e: tj.e, v: tj.v });
            if (vc != null) sends.push({ a: tj.a, t: tj.e, st: vc - tj.v });
          });
        }
        // métricas
        const busy = segs.reduce((s, sg) => s + Math.max(0, Math.min(sg.e, TMAX) - Math.min(sg.s, TMAX)), 0);
        const util = busy / (NA * TMAX);
        const nUp = ups.filter((u2) => u2.s < TMAX).length;
        const stl = sends.length ? sends.reduce((s, x) => s + x.st, 0) / sends.length : 0;
        return { segs, idles, ups, sends, util, rate: (nUp / TMAX) * 100, stl };
      }
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Mode', options: [{ v: 'sync', label: 'A2C (synchronous)' }, { v: 'a3c', label: 'A3C (Hogwild!)' }, { v: 'impala', label: 'IMPALA / async RL' }], value: 'sync', onChange: (v) => { S.mode = v; recompute(); } });
      nocase(UI.slider(ctl, { label: 'rollout tail', min: 0, max: 1, step: 0.1, value: 0.5, fmt: (v) => fmt(v, 1), onInput: (v) => { S.tail = v; recompute(); } }));
      const rU = UI.readout(ctl, 'actor idle time', '—'), rN = UI.readout(ctl, 'updates / 100 t.u.', '—'), rS = UI.readout(ctl, 'mean staleness', '—');
      function recompute() {
        S.sim = simulate();
        rU.set(fmt((1 - S.sim.util) * 100, 0) + '%', S.sim.util > 0.9 ? 'ok' : S.sim.util > 0.7 ? 'warn' : 'bad');
        rN.set(fmt(S.sim.rate, 1));
        rS.set(fmt(S.sim.stl, 2) + ' versions', S.sim.stl < 0.05 ? 'ok' : S.sim.stl < 1.5 ? 'warn' : 'bad');
      }
      recompute();
      const hatch = (g, x, y, w2, h2, col) => {
        g.save(); g.beginPath(); g.rect(x, y, w2, h2); g.clip();
        g.fillStyle = V.alpha(col, 0.08); g.fillRect(x, y, w2, h2);
        g.strokeStyle = V.alpha(col, 0.45); g.lineWidth = 1;
        for (let k = -h2; k < w2; k += 7) { g.beginPath(); g.moveTo(x + k, y + h2); g.lineTo(x + k + h2, y); g.stroke(); }
        g.restore();
      };
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const narrow = w < 600, pad = 12, sim = S.sim;
        const lw = narrow ? 60 : 92;
        const x0 = pad + lw, x1 = w - pad;
        const win = narrow ? 14 : WIN;
        const play = WIN + (t * (narrow ? 1.4 : 2.2)) % (TMAX - WIN - 20);
        const tL = play - win * 0.86;
        const xT = (tt) => x0 + ((tt - tL) / win) * (x1 - x0);
        const rowH = narrow ? 40 : 44, gapR = 10, top = 14;
        const actorCols = [C.a, C.b, C.c, C.e || C.a];
        const llm = S.mode === 'impala';
        const rowY = (i) => top + i * (rowH + gapR);
        const learnY = rowY(NA) + 8;
        for (let i = 0; i < NA; i++) V.text(g, (llm ? 'vLLM ' : 'actor ') + (i + 1), x0 - 8, rowY(i) + rowH / 2 + 4, { size: 11.5, font: 'mono', color: C.muted, align: 'right' });
        V.text(g, S.mode === 'a3c' ? 'shared θ' : llm ? 'trainer' : 'learner', x0 - 8, learnY + rowH / 2 + 4, { size: 11.5, font: 'mono', weight: 600, color: C.d, align: 'right' });
        g.save(); g.beginPath(); g.rect(x0, 0, x1 - x0, h); g.clip();
        // ociosidade
        sim.idles.forEach((id) => { if (id.e < tL || id.s > play) return; const s = Math.max(id.s, tL), e = Math.min(id.e, play); if (e > s) hatch(g, xT(s), rowY(id.a) + 6, xT(e) - xT(s), rowH - 12, C.muted); });
        // segmentos
        sim.segs.forEach((sg) => {
          if (sg.e < tL || sg.s > play) return;
          const s = Math.max(sg.s, tL), e = Math.min(sg.e, play);
          const x = xT(s), ww = Math.max(1, xT(e) - x);
          V.box(g, x + 1, rowY(sg.a) + 6, ww - 2, rowH - 12, { r: 5, fill: V.alpha(actorCols[sg.a], 0.35), stroke: V.alpha(actorCols[sg.a], 0.8), lw: 1 });
          if (ww > 30 && sg.s >= tL) V.text(g, 'v' + sg.v, x + 6, rowY(sg.a) + rowH / 2 + 4, { size: 11, font: 'mono', color: C.ink });
        });
        // atualizações (chips estreitos: rótulo abaixo, em duas faixas, pulando o que não couber)
        const tierR = [-1e9, -1e9];
        sim.ups.forEach((u2) => {
          if (u2.e < tL || u2.s > play) return;
          const s = Math.max(u2.s, tL), e = Math.min(u2.e, play);
          const x = xT(s), ww = Math.max(2, xT(e) - x);
          if (S.mode === 'a3c') { V.dot(g, xT(u2.s), learnY + rowH / 2, 4, actorCols[u2.a]); }
          else { V.box(g, x, learnY + 6, ww, rowH - 12, { r: 5, fill: V.alpha(C.d, 0.45), stroke: C.d, lw: 1.2 }); if (ww > 22) V.text(g, 'v' + u2.v, x + ww / 2, learnY + rowH / 2 + 4, { size: 11, font: 'mono', weight: 700, color: C.ink, align: 'center' });
            else {
              const s2 = 'v' + u2.v, half = tw(g, s2, 10.5, 'mono', 700) / 2 + 1;
              const lx = V.clamp(x + ww / 2, x0 + half, x1 - half);
              const k = lx - half > tierR[0] + 2 ? 0 : lx - half > tierR[1] + 2 ? 1 : -1;
              if (k >= 0) { tierR[k] = lx + half; V.text(g, s2, lx, learnY + rowH - 6 + 12 + 11 * k, { size: 10.5, font: 'mono', weight: 700, color: C.d, align: 'center' }); }
            } }
        });
        if (S.mode === 'a3c') V.line(g, x0, learnY + rowH / 2, xT(play), learnY + rowH / 2, { color: V.alpha(C.d, 0.6), lw: 2 });
        // entregas (setas) com defasagem
        sim.sends.forEach((sd) => {
          if (sd.t < tL || sd.t > play) return;
          const x = xT(sd.t), ya = rowY(sd.a) + rowH - 6, yb = learnY + 6;
          V.arrow(g, x, ya, x + 6, yb, { color: V.alpha(S.mode === 'sync' ? C.muted : sd.st > 1 ? RED() : C.d, 0.7), lw: 1.3, head: 5 });
          if (S.mode !== 'sync' && sd.st > 0) {
            const bx = x + 2, by = ya + 4;
            chip(g, bx - 1, by - 1, 22, 15, '+' + sd.st, { size: 10.5, weight: 700, fill: V.alpha(sd.st > 1 ? RED() : C.d, 0.25), stroke: sd.st > 1 ? RED() : C.d, r: 4 });
          }
        });
        g.restore();
        // cabeça de leitura
        const xp = xT(play);
        V.line(g, xp, 6, xp, learnY + rowH, { color: V.alpha(C.ink, 0.6), lw: 1.4, dash: [3, 3] });
        // descrição
        const desc = {
          sync: 'A2C: each round waits for the slowest rollout; the actors sit idle (hatched) during the wait and the update.',
          a3c: 'A3C: each actor pushes its gradient when it finishes and moves on; the number is how many updates from other actors it missed.',
          impala: 'IMPALA: actors never stop; the learner consumes trajectories in batches of 2, generated with weights +n versions behind (V-trace corrects for it).',
        };
        V.wrap(g, desc[S.mode], pad, learnY + rowH + 34, w - 2 * pad, { size: 12, lineH: 16, color: C.ink, maxLines: 3 });
      });
    })();

    /* ================================================ FIG 9 — TRPO 2D */
    (function () {
      const fig = root.querySelector('#fig-trpo');
      adaptH(fig, 480, 720);
      const S = { delta: 0.01, f11: 2 };
      const KAPPA = 5;
      const mul = (F, v) => [F[0][0] * v[0] + F[0][1] * v[1], F[1][0] * v[0] + F[1][1] * v[1]];
      const dot = (a, b) => a[0] * b[0] + a[1] * b[1];
      function solve() {
        const F = [[S.f11, 0.5], [0.5, 1]], g0 = [1, 1];
        let x = [0, 0], r = g0.slice(), p = r.slice();
        const xs = [x.slice()];
        for (let k = 0; k < 2; k++) {
          const Fp = mul(F, p), rr = dot(r, r);
          if (rr < 1e-14) { xs.push(x.slice()); continue; } // g autovetor de F: o CG já terminou
          const al = rr / dot(p, Fp);
          x = [x[0] + al * p[0], x[1] + al * p[1]]; xs.push(x.slice());
          const rn = [r[0] - al * Fp[0], r[1] - al * Fp[1]];
          const bb = dot(rn, rn) / rr;
          p = [rn[0] + bb * p[0], rn[1] + bb * p[1]]; r = rn;
        }
        const s = x, sFs = dot(s, mul(F, s)), beta = Math.sqrt((2 * S.delta) / sFs);
        const step = [beta * s[0], beta * s[1]];
        const gFg = dot(g0, mul(F, g0)), bg = Math.sqrt((2 * S.delta) / gFg), plain = [bg, bg];
        const klQ = (d) => 0.5 * dot(d, mul(F, d));
        const klT = (d) => klQ(d) * (1 + KAPPA * Math.hypot(d[0], d[1]));
        const lT = (d) => dot(g0, d) - 2 * dot(d, d);
        const tries = [];
        for (let j = 0; j < 6; j++) { const f = Math.pow(0.5, j), d = [step[0] * f, step[1] * f]; const ok = klT(d) <= S.delta && lT(d) > 0; tries.push({ d, ok, kl: klT(d) }); if (ok) break; }
        // autovalores para o enquadramento
        const tr = F[0][0] + F[1][1], det = F[0][0] * F[1][1] - 0.25, disc = Math.sqrt(tr * tr / 4 - det);
        const lmin = tr / 2 - disc;
        return { F, g0, xs, s, sFs, beta, step, plain, klQ, klT, tries, maxAxis: Math.sqrt((2 * S.delta) / lmin) };
      }
      /** textos do passo a passo montados a partir de solve(): acompanham os controles */
      const fl = (x, d = 4) => fmt(x, d);
      const vec = (v) => `(${fl(v[0])},\\ ${fl(v[1])})`;
      const steps = [
        { t: 'The problem', d: 'Maximize the linearized objective \\(g^\\top\\Delta\\) (dashed level curves) without leaving the ellipse \\(\\tfrac12\\Delta^\\top F\\Delta \\le \\delta\\), the quadratic approximation of the KL. The ellipse is stretched along the direction in which moving changes the policy less.' },
        { t: 'Ordinary gradient' }, { t: 'Conjugate gradient' }, { t: 'Maximum step' }, { t: 'Line search' }, { t: 'Natural × ordinary' },
      ];
      function buildSteps() {
        const so = solve(), gp = dot(so.g0, so.plain), gn = dot(so.g0, so.step);
        const init = Math.abs(S.delta - 0.01) < 1e-9 && S.f11 === 2;
        const ctx = init ? 'at the initial values (\\(F_{11} = 2\\), \\(\\delta = 0.01\\))' : `with \\(F_{11} = ${fl(S.f11, 2)}\\) and \\(\\delta = ${fl(S.delta, 4)}\\)`;
        const Ctx = ctx.charAt(0).toUpperCase() + ctx.slice(1);
        steps[1].d = `${Ctx}, moving along \\(g = (1,\\ 1)\\) to the edge of the ellipse gives the step \\(${vec(so.plain)}\\), with linear gain \\(g^\\top\\Delta = ${fl(gp)}\\). It ignores the geometry: it moves equally in both directions.`;
        const cg1 = Math.hypot(so.xs[2][0] - so.xs[1][0], so.xs[2][1] - so.xs[1][1]) < 1e-9;
        steps[2].d = cg1
          ? `Solve \\(Fs = g\\) without inverting \\(F\\): ${ctx}, \\(g\\) is an eigenvector of \\(F\\), and CG finishes at the 1st iteration, at \\(${vec(so.xs[1])}\\). Only products \\(F v\\) were needed.`
          : `Solve \\(Fs = g\\) without inverting \\(F\\): ${ctx}, the 1st CG iteration gives \\(${vec(so.xs[1])}\\); the 2nd, \\(${vec(so.xs[2])}\\), already exact in 2D. Only products \\(F v\\) were needed.`;
        steps[3].d = `${Ctx}, \\(s^\\top F s = ${fl(so.sFs)}\\), so \\(\\beta = \\sqrt{2\\delta/s^\\top Fs} = ${fl(so.beta)}\\) and the step \\(\\beta s = ${vec(so.step)}\\) touches the ellipse (quadratic KL = ${fmt(so.klQ(so.step), 4)}) at the point where the objective's level curve is tangent to it.`;
        const nT = so.tries.length, lastOk = so.tries[nT - 1].ok;
        const ls = so.tries[0].ok ? 'Here the full step already fits within the real KL and is accepted on the first try.'
          : lastOk ? `The full step violates \\(\\delta\\): it is halved ${nT - 1 === 1 ? 'once' : (nT - 1) + ' times'}, down to a point that satisfies the real KL and still improves the objective.`
            : 'None of the 6 attempts satisfies the real KL, and this iteration\'s step is discarded.';
        steps[4].d = `The <i>real</i> KL (amber curve, illustrative) is larger than the quadratic one. ${ls}`;
        steps[5].d = Math.abs(gn - gp) < 1e-4
          ? `${Ctx}, \\(g\\) is an eigenvector of \\(F\\): the two directions coincide and the gains tie (${fmt(gn, 4)}). Change the curvature \\(F_{11}\\) and watch the natural direction split off from the ordinary one.`
          : `${Ctx}, on the same ellipse, the natural gradient reaches a higher level curve (${fmt(gn, 4)} versus ${fmt(gp, 4)}): it is the direction that climbs the most per unit of KL. Change the curvature \\(F_{11}\\) and watch the natural direction rotate.`;
      }
      buildSteps();
      let H = null;
      const st = UI.stepper(fig, steps, { autoplay: 6500, onStep: () => H && H.redraw() });
      /** reescreve o texto do passo atual sem reiniciar a animação do passo */
      const refresh = () => {
        buildSteps();
        const el = fig.querySelector('.la-stepper-text');
        if (!el) return;
        el.innerHTML = `<b class="st">${st.i + 1}. ${steps[st.i].t}</b>${steps[st.i].d || ''}`;
        if (LA.renderMath) LA.renderMath(el);
      };
      const ctl = UI.controls(fig);
      nocase(UI.slider(ctl, { label: 'δ', min: 0.0025, max: 0.04, step: 0.0025, value: 0.01, fmt: (v) => fmt(v, 4), onInput: (v) => { S.delta = v; refresh(); } }));
      nocase(UI.slider(ctl, { label: 'curvature F₁₁', min: 1, max: 4, step: 0.25, value: 2, fmt: (v) => fmt(v, 2), onInput: (v) => { S.f11 = v; refresh(); } }));
      const rB = nocase(UI.readout(ctl, 'β', '—')), rK = nocase(UI.readout(ctl, 'quadratic KL', '—'));
      let fr = 0;
      H = V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const narrow = w < 640, pad = 12;
        const so = solve();
        if (++fr % 6 === 0) { rB.set(fmt(so.beta, 4)); rK.set(fmt(so.klQ(so.step), 4)); }
        const i = st.i, el = st.elapsed();
        const P = narrow ? { x: pad, y: 8, w: w - 2 * pad, h: Math.min(w - 2 * pad, 360) } : { x: pad, y: 8, w: Math.min(h - 16, w * 0.6), h: h - 16 };
        const side = Math.min(P.w, P.h);
        const cx = P.x + side / 2, cy = P.y + side / 2;
        const RNG = so.maxAxis * 1.35, sc = (side / 2 - 10) / RNG;
        const X = (d) => cx + d[0] * sc, Y = (d) => cy - d[1] * sc;
        V.box(g, P.x, P.y, side, side, { r: 10, fill: V.alpha(C.muted, 0.04), stroke: V.alpha(C.muted, 0.25), lw: 1 });
        g.save(); g.beginPath(); V.rr(g, P.x, P.y, side, side, 10); g.clip();
        // grade
        V.line(g, P.x, cy, P.x + side, cy, { color: C.grid, lw: 1 }); V.line(g, cx, P.y, cx, P.y + side, { color: C.grid, lw: 1 });
        // curvas de nível g·Δ = c
        for (let k = -6; k <= 6; k++) {
          const c = k * RNG * 0.28;
          const a = [c / 2 - RNG * 2, c / 2 + RNG * 2], b = [c / 2 + RNG * 2, c / 2 - RNG * 2];
          V.line(g, X(a), Y(a), X(b), Y(b), { color: V.alpha(C.b, 0.22), lw: 1, dash: [4, 4] });
        }
        // elipse KL quadrática
        const ell = (fn, col, lw, dash) => {
          g.beginPath();
          for (let k = 0; k <= 120; k++) {
            const a = (k / 120) * Math.PI * 2, u = [Math.cos(a), Math.sin(a)];
            const r = fn(u);
            const p = [u[0] * r, u[1] * r];
            if (k) g.lineTo(X(p), Y(p)); else g.moveTo(X(p), Y(p));
          }
          g.closePath(); g.strokeStyle = col; g.lineWidth = lw; if (dash) g.setLineDash(dash); g.stroke(); g.setLineDash([]);
        };
        const rQ = (u) => Math.sqrt(S.delta / (0.5 * dot(u, mul(so.F, u))));
        g.fillStyle = V.alpha(C.a, 0.08);
        g.beginPath(); for (let k = 0; k <= 120; k++) { const a = (k / 120) * Math.PI * 2, u = [Math.cos(a), Math.sin(a)], r = rQ(u), p = [u[0] * r, u[1] * r]; if (k) g.lineTo(X(p), Y(p)); else g.moveTo(X(p), Y(p)); } g.fill();
        ell(rQ, C.a, 2);
        if (i >= 4) {
          const rT = (u) => { const a = 0.5 * dot(u, mul(so.F, u)); let lo = 0, hi = Math.sqrt(S.delta / a); for (let k = 0; k < 40; k++) { const m = (lo + hi) / 2; if (a * m * m * (1 + KAPPA * m) > S.delta) hi = m; else lo = m; } return lo; };
          ell(rT, C.d, 1.8, [6, 4]);
        }
        g.restore();
        V.text(g, 'Δθ₁', P.x + side - 6, cy - 6, { size: 11, font: 'mono', color: C.muted, align: 'right' });
        V.text(g, 'Δθ₂', cx + 6, P.y + 14, { size: 11, font: 'mono', color: C.muted });
        V.text(g, 'KL = δ', X([0, rQ([0, 1])]) + 6, Y([0, rQ([0, 1])]) - 6, { size: 11, font: 'mono', color: C.a });
        // seta do gradiente (direção de subida)
        const o = [0, 0];
        if (i >= 1) {
          const f = i === 1 ? V.ease.out(V.clamp(el / 0.9, 0, 1)) : 1;
          const p = [so.plain[0] * f, so.plain[1] * f];
          V.arrow(g, X(o), Y(o), X(p), Y(p), { color: C.b, lw: 2.6, head: 9 });
          if (f > 0.9) V.text(g, 'ordinary', X(p) + 6, Y(p) + 14, { size: 11, weight: 700, color: C.b });
        }
        if (i >= 3) {
          const f = i === 3 ? V.ease.out(V.clamp(el / 0.9, 0, 1)) : 1;
          const p = [so.step[0] * f, so.step[1] * f];
          V.arrow(g, X(o), Y(o), X(p), Y(p), { color: C.c, lw: 3, head: 10 });
          if (f > 0.9) {
            V.text(g, 'natural', X(p) + 8, Y(p) - 2, { size: 11, weight: 700, color: C.c });
            // tangente: curva de nível pelo ponto do passo
            const c = dot(so.g0, so.step), a = [c / 2 - RNG * 2, c / 2 + RNG * 2], b = [c / 2 + RNG * 2, c / 2 - RNG * 2];
            g.save(); g.beginPath(); V.rr(g, P.x, P.y, side, side, 10); g.clip();
            V.line(g, X(a), Y(a), X(b), Y(b), { color: V.alpha(C.c, 0.55), lw: 1.4, dash: [2, 3] });
            g.restore();
          }
        }
        if (i === 4) {
          so.tries.forEach((tr, k) => {
            const show = el > 0.6 + k * 1.1;
            if (!show) return;
            V.dot(g, X(tr.d), Y(tr.d), 5.5, tr.ok ? C.a : RED());
            V.text(g, String(k + 1), X(tr.d) - 12, Y(tr.d) + 4, { size: 11, font: 'mono', weight: 700, color: tr.ok ? C.a : RED(), align: 'right' });
            const lab = `${k + 1}. ${k === 0 ? 'full step' : '× ' + fmt(Math.pow(0.5, k), 2)}: real KL ${fmt(tr.kl, 4)} ${tr.ok ? '≤ δ ✓' : '> δ ✗'}`;
            const ly = P.y + side - 14 - (so.tries.length - 1 - k) * 18;
            V.box(g, P.x + 8, ly - 13, Math.min(side - 16, tw(g, lab, 11, 'mono', 600) + 12), 17, { r: 4, fill: V.alpha(C.bg || '#000', 0.85), stroke: V.alpha(tr.ok ? C.a : RED(), 0.6), lw: 1 });
            V.text(g, lab, P.x + 14, ly, { size: 11, font: 'mono', weight: 600, color: tr.ok ? C.a : RED(), maxW: side - 28 });
          });
        }
        // painel numérico
        const Q = narrow ? { x: pad, y: P.y + side + 14, w: w - 2 * pad } : { x: P.x + side + 18, y: P.y + 4, w: w - (P.x + side + 18) - pad };
        let y = Q.y + 12;
        V.text(g, 'F (Fisher)', Q.x, y, { size: 12, weight: 600, color: C.ink }); y += 8;
        const cw = Math.min(70, (Q.w - 20) / 4);
        V.matrix(g, { x: Q.x, y, data: so.F, cell: cw, cellH: 24, gap: 3, cmap: 'seq', min: 0, max: 4.5, values: true, fmt: (v) => fmt(v, 2) });
        V.text(g, 'g', Q.x + 2 * cw + 18, y + 16, { size: 12, font: 'mono', weight: 600, color: C.muted });
        V.matrix(g, { x: Q.x + 2 * cw + 32, y, data: [[1], [1]], cell: cw * 0.8, cellH: 24, gap: 3, cmap: 'seq', min: 0, max: 3, values: true, fmt: (v) => fmt(v, 0) });
        y += 2 * 27 + 18;
        const lines = [
          [`CG it. 1: x₁ = (${fmt(so.xs[1][0], 4)}, ${fmt(so.xs[1][1], 4)})`, i >= 2 ? C.ink : C.muted],
          [`CG it. 2: x₂ = (${fmt(so.xs[2][0], 4)}, ${fmt(so.xs[2][1], 4)})`, i >= 2 ? C.ink : C.muted],
          [`sᵀFs = ${fmt(so.sFs, 4)}   β = ${fmt(so.beta, 4)}`, i >= 3 ? C.ink : C.muted],
          [`βs = (${fmt(so.step[0], 4)}, ${fmt(so.step[1], 4)})`, i >= 3 ? C.c : C.muted],
          [`ordinary = (${fmt(so.plain[0], 4)}, ${fmt(so.plain[1], 4)})`, i >= 1 ? C.b : C.muted],
          [`gain gᵀΔ: natural ${fmt(dot(so.g0, so.step), 4)} · ordinary ${fmt(dot(so.g0, so.plain), 4)}`, i >= 5 ? C.d : C.muted],
        ];
        lines.forEach(([s, c], k) => V.text(g, s, Q.x, y + k * 19, { size: 11.5, font: 'mono', weight: 500, color: c, maxW: Q.w }));
        // mini-espaço do CG
        if (i >= 2 && (!narrow || h > 680)) {
          const my = y + lines.length * 19 + 6, mh = Math.min(110, h - my - 10), mw = Math.min(Q.w, 200);
          if (mh > 60) {
            V.box(g, Q.x, my, mw, mh, { r: 8, fill: V.alpha(C.muted, 0.05), stroke: V.alpha(C.muted, 0.25), lw: 1 });
            V.text(g, 's space (CG)', Q.x + 6, my + 14, { size: 11, color: C.muted });
            const mx0 = Q.x + 18, my0 = my + mh - 12, msc = Math.min((mw - 30) / 1.0, (mh - 30) / 1.0);
            const pts = so.xs.map((p) => [mx0 + p[0] * msc, my0 - p[1] * msc]);
            const f = i === 2 ? V.clamp(el / 1.6, 0, 1) : 1;
            for (let k = 0; k < 2; k++) {
              const ff = V.clamp(f * 2 - k, 0, 1);
              if (ff <= 0) break;
              V.arrow(g, pts[k][0], pts[k][1], V.lerp(pts[k][0], pts[k + 1][0], ff), V.lerp(pts[k][1], pts[k + 1][1], ff), { color: C.c, lw: 2, head: 6 });
            }
            pts.forEach((p, k) => { if (k <= Math.ceil(f * 2)) V.dot(g, p[0], p[1], 3.5, k === 2 ? C.c : C.ink); });
          }
        }
      });
    })();

    /* ================================================ FIG 10 — linhagem */
    (function () {
      const fig = root.querySelector('#fig-linhagem');
      adaptH(fig, 430, 700);
      const LANES = [
        { k: 'adv', t: 'advantage estimator', c: 'a' },
        { k: 'tr', t: 'trust region', c: 'b' },
        { k: 'off', t: 'off-policy / async', c: 'c' },
      ];
      const N = [
        { id: 'rf', l: 'REINFORCE\nPG theorem', y: '1992 · 2000', lane: 0, x: 0.05 },
        { id: 'gae', l: 'GAE', y: '2015', lane: 0, x: 0.27 },
        { id: 'grpo', l: 'RLOO · GRPO', y: '2024', lane: 0, x: 0.6 },
        { id: 'vapo', l: 'VC-PPO\nVAPO', y: '2025', lane: 0, x: 0.77 },
        { id: 'sao', l: 'SAO\n(GLM-5.2)', y: '2026', lane: 0, x: 0.94 },
        { id: 'ng', l: 'natural\ngradient', y: '2001', lane: 1, x: 0.08 },
        { id: 'trpo', l: 'TRPO', y: '2015', lane: 1, x: 0.27 },
        { id: 'ppo', l: 'PPO', y: '2017', lane: 1, x: 0.43 },
        { id: 'gspo', l: 'DAPO · GSPO\nSAPO', y: '2025', lane: 1, x: 0.75 },
        { id: 'mask', l: 'V3.2\nmask', y: '2025', lane: 1, x: 0.94 },
        { id: 'a3c', l: 'A3C · A2C', y: '2016', lane: 2, x: 0.34 },
        { id: 'imp', l: 'IMPALA\nV-trace', y: '2018', lane: 2, x: 0.5 },
        { id: 'async', l: 'async\nRLHF', y: '2024', lane: 2, x: 0.64 },
        { id: 'tis', l: 'TIS · R3', y: '2025', lane: 2, x: 0.79 },
        { id: 'ice', l: 'IcePop\n(GLM-5)', y: '2026', lane: 2, x: 0.94 },
      ];
      const E = [['rf', 'gae', 0], ['gae', 'ppo', 0], ['gae', 'grpo', 0], ['ppo', 'grpo', 1], ['grpo', 'vapo', 0], ['vapo', 'sao', 0],
        ['ng', 'trpo', 1], ['trpo', 'ppo', 1], ['ppo', 'gspo', 1], ['gspo', 'mask', 1],
        ['a3c', 'ppo', 2], ['a3c', 'imp', 2], ['imp', 'async', 2], ['async', 'tis', 2], ['tis', 'ice', 2]];
      const byId = {}; N.forEach((n) => { byId[n.id] = n; });
      const S = { lever: -1 };
      const DESC = [
        'All lanes: PPO inherits from GAE (advantage), from TRPO (trust region) and from A3C\'s actor scheme, and each lane kept evolving.',
        'Advantage: from REINFORCE with a baseline to GAE with a critic; GRPO swaps the critic for the group; VC-PPO, VAPO and SAO bring the critic back with decoupled and adaptive λ.',
        'Trust region: from the natural gradient to TRPO (KL ≤ δ) and PPO\'s clip; then clip-higher (DAPO), per-sequence clip (GSPO), soft gate (SAPO) and KL mask (DeepSeek-V3.2).',
        'Off-policy: asynchronous actors (A3C), stale trajectories with V-trace (IMPALA), asynchronous RLHF with vLLM, and the fixes for the training–inference mismatch: TIS, R3, IcePop.',
      ];
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Lever', options: [{ v: -1, label: 'all' }, { v: 0, label: 'advantage' }, { v: 1, label: 'trust region' }, { v: 2, label: 'off-policy' }], value: -1, onChange: (v) => { S.lever = v; } });
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const narrow = w < 600, pad = 12;
        const col = (lane) => C[LANES[lane].c];
        const descH = narrow ? 70 : 46;
        const H0 = h - descH - 8;
        let pos;
        if (!narrow) {
          const top = 30, lh = (H0 - top) / 3;
          pos = (n) => ({ x: pad + 56 + n.x * (w - 2 * pad - 120), y: top + lh * (n.lane + 0.5) });
          LANES.forEach((ln, i) => {
            const y = top + lh * (i + 0.5);
            g.fillStyle = V.alpha(col(i), S.lever === -1 || S.lever === i ? 0.07 : 0.025);
            V.rr(g, pad, y - lh / 2 + 4, w - 2 * pad, lh - 8, 10); g.fill();
            V.text(g, ln.t, pad + 8, y - lh / 2 + 18, { size: 11.5, weight: 700, color: col(i) });
          });
        } else {
          const top = 44, cwid = (w - 2 * pad) / 3;
          pos = (n) => ({ x: pad + cwid * (n.lane + 0.5), y: top + 20 + n.x * (H0 - top - 50) });
          LANES.forEach((ln, i) => {
            const x = pad + cwid * (i + 0.5);
            g.fillStyle = V.alpha(col(i), S.lever === -1 || S.lever === i ? 0.07 : 0.025);
            V.rr(g, x - cwid / 2 + 3, 8, cwid - 6, H0 - 10, 10); g.fill();
            V.wrap(g, ln.t, x - cwid / 2 + 8, 24, cwid - 14, { size: 11, weight: 700, color: col(i), lineH: 13, maxLines: 2 });
          });
        }
        const on = (lane) => S.lever === -1 || S.lever === lane;
        // tamanhos dos nós
        const s0 = narrow ? 11 : 11.5;
        const geo = {};
        N.forEach((n) => {
          const lines = n.l.split('\n');
          const ww = Math.min(narrow ? (w - 2 * pad) / 3 - 10 : 150, Math.max(...lines.map((ln) => tw(g, ln, s0, 'sans', 600))) + 16);
          geo[n.id] = { p: pos(n), w: ww, h: 14 * lines.length + 20, lines };
        });
        const anchors = (A, B) => {
          if (!narrow && B.p.x - A.p.x > (A.w + B.w) / 2 + 12) return { h: true, a: [A.p.x + A.w / 2, A.p.y], b: [B.p.x - B.w / 2, B.p.y] };
          const down = B.p.y > A.p.y;
          return { h: false, a: [A.p.x, A.p.y + (down ? A.h / 2 : -A.h / 2)], b: [B.p.x, B.p.y + (down ? -B.h / 2 : B.h / 2)] };
        };
        // arestas com pulsos
        E.forEach(([a, b, lane], k) => {
          const an = anchors(geo[a], geo[b]), A = an.a, B = an.b;
          const active = on(lane);
          const c1 = an.h ? [(A[0] + B[0]) / 2, A[1]] : [A[0], (A[1] + B[1]) / 2];
          const c2 = an.h ? [(A[0] + B[0]) / 2, B[1]] : [B[0], (A[1] + B[1]) / 2];
          g.beginPath(); g.moveTo(A[0], A[1]); g.bezierCurveTo(c1[0], c1[1], c2[0], c2[1], B[0], B[1]);
          g.strokeStyle = V.alpha(col(lane), active ? 0.6 : 0.12); g.lineWidth = active ? 2 : 1; g.stroke();
          if (active) {
            const f = (t * 0.45 + k * 0.17) % 1;
            const p = V.bez(A, c1, c2, B, f);
            V.glow(g, p[0], p[1], 9, col(lane), 0.9);
            const q = V.bez(A, c1, c2, B, 0.97);
            V.arrow(g, q[0], q[1], B[0], B[1], { color: V.alpha(col(lane), 0.8), lw: 1.5, head: 6 });
          }
        });
        // nós
        N.forEach((n) => {
          const G = geo[n.id], p = G.p, active = on(n.lane) || (n.id === 'ppo' && S.lever !== -1);
          g.globalAlpha = active ? 1 : 0.35;
          const isP = n.id === 'ppo';
          V.box(g, p.x - G.w / 2, p.y - G.h / 2, G.w, G.h, { r: 8, fill: V.alpha(isP ? C.d : col(n.lane), isP ? 0.3 : 0.16), stroke: isP ? C.d : col(n.lane), lw: isP ? 2 : 1.3 });
          G.lines.forEach((ln, li) => V.text(g, ln, p.x, p.y - G.h / 2 + 15 + li * 14, { size: s0, weight: 600, color: C.ink, align: 'center', maxW: G.w - 6 }));
          V.text(g, n.y, p.x, p.y + G.h / 2 - 6, { size: 10.5, font: 'mono', color: C.muted, align: 'center' });
          g.globalAlpha = 1;
        });
        V.wrap(g, DESC[S.lever + 1], pad, H0 + 20, w - 2 * pad, { size: 12, lineH: 16, color: C.ink, maxLines: 4 });
      });
    })();
  },
});

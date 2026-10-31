LA.chapter({
  id: 'dpo',
  part: 12,
  order: 57,
  title: 'DPO: direct preference optimization',
  navTitle: 'DPO',
  subtitle: 'The KL-regularized RLHF objective has a closed-form solution; inverted, it hides a reward inside the policy itself, and Z(x) cancels out in Bradley–Terry. What is left is a logistic regression over pairs: the loss, the gradient, β, the reference model, the online versions and the failure modes.',
  minutes: 45,
  recipes: [],
  tags: ['DPO', 'RLHF', 'Bradley–Terry', 'implicit reward', 'β', 'KL', 'reference model', 'online DPO', 'TRL'],
  goals: [
    'Derive, step by step, the optimal policy of KL-regularized RL, \\(\\pi^*\\propto\\pi_{\\text{ref}}\\,e^{r/\\beta}\\), using <b>Gibbs\' inequality</b>',
    'Show why \\(Z(x)\\) cancels out in Bradley–Terry and arrive at the DPO loss as a logistic regression on the <b>implicit reward</b> \\(\\beta\\log(\\pi_\\theta/\\pi_{\\text{ref}})\\)',
    'Read the DPO gradient (adaptive weight, only the margin matters, first divergent token) and the role of \\(\\beta\\) and of the reference model',
    'Recognize the failure modes and configure, monitor and diagnose a run with <code>DPOTrainer</code>, offline or online',
  ],
  prereqs: ['preferencias-reward-models', 'ppo-rlhf'],
  summary: [
    'The RLHF objective, \\(\\max\\ \\mathbb E[r]-\\beta\\,\\mathbb D_{\\text{KL}}(\\pi\\|\\pi_{\\text{ref}})\\), has a closed-form solution: \\(\\pi^*=\\pi_{\\text{ref}}\\,e^{r/\\beta}/Z(x)\\), an <b>exponential tilting</b> of the reference. The objective equals \\(\\beta\\log Z-\\beta\\,\\mathbb D_{\\text{KL}}(\\pi\\|\\pi^*)\\), and Gibbs (\\(\\mathbb D_{\\text{KL}}\\ge0\\)) makes \\(\\pi^*\\) the unique maximum.',
    'Inverting it: \\(r=\\beta\\log\\frac{\\pi^*}{\\pi_{\\text{ref}}}+\\beta\\log Z(x)\\). Since Bradley–Terry only compares responses to the <b>same</b> prompt, \\(\\beta\\log Z(x)\\) cancels out, and the preference is written in terms of the policy alone. By Theorem 1, no generality is lost.',
    'Loss: \\(-\\mathbb E[\\log\\sigma(\\beta\\Delta h)]\\), the RM\'s logistic regression with \\(\\hat r_\\theta=\\beta\\log(\\pi_\\theta/\\pi_{\\text{ref}})\\). Two models, 4 forwards per pair, no generated text. On the toy pair (\\(\\Delta h=3\\), \\(\\beta=0.1\\)): loss 0.5544, weight 0.4256.',
    'Gradient: adaptive weight \\(\\sigma(-\\beta\\Delta h)\\) (without it, the Unlikelihood baseline degenerates); only the margin matters (both log-probs can fall); the signal starts at the first divergent token.',
    '\\(\\beta\\) is the price of KL and the logit scale: \\(\\Delta h^*=\\operatorname{logit}(p)/\\beta\\), infinite for deterministic preferences. Published values: 0.1 (default), 0.5 (TL;DR), 0.01 (UltraFeedback), 5 (length-normalized). The reference can be a copy, precomputed log-probs, the model with the LoRA adapter turned off or a moving reference (TR-DPO).',
    'Offline DPO is off-policy; online/iterative DPO regenerates pairs with the current policy (OAIF: ~66% wins against offline) and puts vLLM in the loop. Failure modes: likelihood displacement (refusals from 74.4% to 33.4%), verbosity (~2× longer), the over-optimization hump (peak at ~25% of the epoch) and out-of-distribution solutions.',
    'In practice: LR \\(5\\times10^{-7}\\)–\\(10^{-6}\\) in full FT; ~\\(10^{-5}\\) with LoRA in PEFT (alpha/r = 1) and ~\\(10^{-6}\\) in <code>mlx-lm-lora</code> (scale 10), because the effective step grows with scale × LR; 1 epoch, batch 64–128. Watch <code>rewards/margins</code>, <code>rewards/accuracies</code>, <code>logps/chosen</code>, length, KL and the win rate per checkpoint; <code>loss_type=["sigmoid","sft"]</code> anchors the chosen responses.',
  ],
  quiz: [
    { q: 'Why does the \\(\\beta\\log Z(x)\\) term disappear when you write the DPO loss?', o: ['Because \\(Z(x)=1\\) for any trained policy', 'Because Bradley–Terry depends only on the reward difference between two responses to the <b>same</b> prompt, and the term, which depends only on \\(x\\), appears identically on both sides', 'Because DPO estimates \\(Z(x)\\) by sampling at every step', 'Because, with a small \\(\\beta\\), \\(\\log Z\\) becomes negligible'], a: 1, e: 'By Eq. 5, \\(r=\\beta\\log\\frac{\\pi^*}{\\pi_{\\text{ref}}}+\\beta\\log Z(x)\\). In the difference \\(r(x,y_w)-r(x,y_l)\\), \\(\\beta\\log Z(x)\\) is both added and subtracted. \\(Z\\) does not equal 1 in general (the reparameterization only picks, in each equivalence class, the representative with \\(Z=1\\)), it is never estimated by sampling, and a small \\(\\beta\\) does not make it negligible: \\(Z(x)=\\mathbb E_{y\\sim\\pi_{\\text{ref}}}[e^{r/\\beta}]\\), and as \\(\\beta\\to0\\), \\(\\beta\\log Z(x)\\to\\max_y r(x,y)\\) (\\(\\log Z\\) scales like \\(1/\\beta\\)). The problem with \\(Z\\) is not its size, it is computing it: the sum has \\(\\sim10^{518}\\) terms for 100-token responses with the Qwen3-4B vocabulary.' },
    { q: 'On the toy pair, \\(h_w=+1\\) and \\(h_l=-2\\). With \\(\\beta=0.1\\), what are the loss and the pair\'s weight in the gradient?', o: ['0.6931 and 0.5', '0.5544 and 0.4256', '0.0486 and 0.0474', '0.3133 and 0.2689'], a: 1, e: '\\(\\Delta h=3\\) and \\(u=\\beta\\Delta h=0.3\\): \\(-\\log\\sigma(0.3)=0.5544\\) and \\(\\sigma(-0.3)=0.4256\\). 0.6931 and 0.5 are the step-zero values (\\(\\Delta h=0\\)); 0.0486 and 0.0474 correspond to \\(\\beta=1\\); 0.3133 and 0.2689 correspond to \\(u=1\\) (\\(\\Delta h=10\\) with \\(\\beta=0.1\\)).' },
    { q: 'What is the role of the factor \\(\\sigma\\big(\\hat r_\\theta(x,y_l)-\\hat r_\\theta(x,y_w)\\big)\\) in the DPO gradient?', o: ['To normalize the gradient by response length', 'To give more weight to the pairs that the implicit reward still orders wrongly or with a small margin; without it (the Unlikelihood baseline), the model degenerates', 'To keep \\(\\log\\pi_\\theta(y_w\\mid x)\\) from falling', 'To estimate the KL to the reference'], a: 1, e: 'The weight is 0.5 at the start and tends to 0 once the pair is well ordered. It does not normalize by length (SimPO points to exactly this gap: long sequences dominate the gradient), it does not keep \\(\\log\\pi(y_w)\\) from falling (only the margin matters, hence likelihood displacement) and it does not estimate KL.' },
    { q: 'In the TRL dashboard, <code>rewards/margins</code> and <code>rewards/accuracies</code> go up, but <code>logps/chosen</code> keeps falling. What is happening, and what should you do?', o: ['Healthy training: the policy is becoming more confident', 'Likelihood displacement: both responses lose mass (the rejected one more so), and it goes to other responses. Remedies: NLL on the chosen responses (<code>loss_type=["sigmoid","sft"]</code>), a larger \\(\\beta\\), DPOP, filtering out high-CHES pairs', 'Verbosity; just reduce <code>max_length</code>', 'The LR is too low; raise it 10×'], a: 1, e: 'The loss only sees \\(\\Delta h\\), so it can lower both log-probs as long as the rejected one falls more. Razin and colleagues show that the mass can go to responses with the opposite meaning (refusals from 74.4% to 33.4% on SORRY-Bench). Verbosity shows up in generation length, not in <code>logps/chosen</code>, and raising the LR makes things worse.' },
    { q: 'You train DPO with LoRA in TRL and pass <code>ref_model=None</code>. Where is the reference?', o: ['There is no reference: with LoRA, DPO turns into SimPO', 'It is the model itself with the adapters turned off, at no extra memory cost. That is why the SFT adapter must be merged into the weights before DPO', 'TRL automatically loads a second copy of the base', 'TRL forces you to precompute the log-probs'], a: 1, e: 'With PEFT, the frozen base is the reference: one forward with the adapters turned off is enough. If the SFT lived in a separate adapter, it would be turned off too, and the reference would become the base model. Exception: TR-DPO\'s moving reference (<code>sync_ref_model</code>) is incompatible with PEFT without a separate <code>ref_model</code>.' },
    { q: 'Every pair in your dataset has a single label (\\(\\hat p=1\\)). What does theory say about the DPO optimum, and what does \\(\\beta\\) change?', o: ['The optimum demands \\(\\Delta h=\\operatorname{logit}(1)/\\beta=+\\infty\\) for any \\(\\beta\\): \\(\\beta\\) changes the scale and the speed, but does not keep the rejected response from being driven to zero', 'With a large \\(\\beta\\), the optimum becomes finite', 'The optimum is \\(\\Delta h=1/(2\\beta)\\)', 'DPO discards deterministic pairs'], a: 0, e: 'With \\(p&lt;1\\), the minimum sits at \\(\\Delta h^*=\\operatorname{logit}(p)/\\beta\\); with \\(p=1\\), the loss only falls as the margin grows and never reaches zero. This is the argument of IPO (Azar and colleagues), which replaces the sigmoid with a squared regression toward a finite target, \\(1/(2\\tau)\\): that target belongs to IPO, not to DPO. A large \\(\\beta\\) only makes the weight saturate sooner; it does not create a finite minimum.' },
  ],
  mount(root, LA) {
    const V = LA.viz, UI = LA.ui;
    const sig = (z) => 1 / (1 + Math.exp(-z));
    const logsig = (z) => (z >= 0 ? -Math.log1p(Math.exp(-z)) : z - Math.log1p(Math.exp(z)));
    const clamp = V.clamp;
    const RED = () => V.C().red || '#FB2C36';
    const fmt = (x, d = 2) => { let s = (+x).toFixed(d); if (/^-0(\.0+)?$/.test(s)) s = s.slice(1); return s; };
    const fmtS = (x, d = 2) => (x > 0 && +(+x).toFixed(d) !== 0 ? '+' : '') + fmt(x, d);
    const fmtBig = (x) => {
      if (x >= 1e4 || (x > 0 && x < 1e-3)) { const p = x.toExponential(1).split('e'); return p[0] + 'e' + (+p[1]); }
      return fmt(x, x >= 100 ? 0 : x >= 10 ? 1 : 2);
    };
    function fmtTick(v) {
      if (Math.abs(v) < 1e-9) return '0';
      let s = (+v).toFixed(Math.abs(v) >= 100 ? 0 : Math.abs(v) >= 10 ? 1 : 2);
      if (s.includes('.')) s = s.replace(/0+$/, '').replace(/\.$/, '');
      return s;
    }
    /** tira o uppercase do CSS dos rótulos com símbolos (β, π, Δh…) */
    const nocase = (c) => { const el = c && (c.el || c); if (el && el.querySelectorAll) el.querySelectorAll('label, span').forEach((n) => { if (!n.classList.contains('val')) n.style.textTransform = 'none'; }); return c; };
    /** rótulo de controle com subscritos em HTML (o canvas usa π_ref; o DOM usa <sub>) */
    const htmlLabel = (c, html) => { const n = c && c.el && c.el.querySelector('label, span'); if (n) n.innerHTML = html; return c; };
    function adaptH(fig, wide, narrow, bp = 640) {
      const st = fig.querySelector('.la-fig-stage');
      const set = () => { const want = (fig.clientWidth < bp ? narrow : wide) + 'px'; if (st.style.height !== want) st.style.height = want; };
      set();
      const ro = new ResizeObserver(set); ro.observe(fig);
      if (LA.onCleanup) LA.onCleanup(() => ro.disconnect());
    }
    function chip(g, x, y, w, h, label, o = {}) {
      const C = V.C();
      V.box(g, x, y, w, h, { r: o.r == null ? 6 : o.r, fill: o.fill || V.alpha(C.muted, 0.12), stroke: o.stroke || V.alpha(C.muted, 0.4), lw: o.lw || 1.2 });
      if (label != null && label !== '') sym(g, label, x + w / 2, y + h / 2 + 0.5, { size: o.size || 11.5, font: o.font || 'mono', weight: o.weight || 500, color: o.color || C.ink, align: 'center', base: 'middle' });
    }
    /* texto com subscritos: 'π_θ', 'π_ref', 'y_w', 'V_ψ', 'x_{abc}' */
    function symParts(str) {
      const parts = []; let i = 0, cur = '';
      while (i < str.length) {
        const ch = str[i];
        if (ch === '_' && i + 1 < str.length) {
          if (cur) parts.push({ s: cur, sub: false });
          cur = ''; i++;
          let sub = '';
          if (str[i] === '{') { i++; while (i < str.length && str[i] !== '}') sub += str[i++]; i++; }
          else { while (i < str.length && /[A-Za-z0-9θφψ*]/.test(str[i])) sub += str[i++]; }
          parts.push({ s: sub, sub: true });
        } else { cur += ch; i++; }
      }
      if (cur) parts.push({ s: cur, sub: false });
      return parts;
    }
    function sym(g, str, x, y, o = {}) {
      const size = o.size || 12, font = o.font || 'sans', weight = o.weight || 400;
      const ss = Math.max(9, Math.round(size * 0.74));
      const parts = symParts(String(str));
      let W = 0;
      parts.forEach((p) => { g.font = V.font(p.sub ? ss : size, font, weight); p.w = g.measureText(font === 'mono' ? V.noLig(p.s) : p.s).width; W += p.w; });
      let x0 = o.align === 'center' ? x - W / 2 : o.align === 'right' ? x - W : x;
      parts.forEach((p) => {
        V.text(g, p.s, x0, y + (p.sub ? size * 0.3 : 0), { size: p.sub ? ss : size, font, weight, color: o.color, base: o.base || 'alphabetic', alpha: o.alpha });
        x0 += p.w;
      });
      return W;
    }
    function symW(g, str, size, font = 'sans', weight = 400) {
      const ss = Math.max(9, Math.round(size * 0.74));
      return symParts(String(str)).reduce((a, p) => { g.font = V.font(p.sub ? ss : size, font, weight); return a + g.measureText(p.s).width; }, 0);
    }
    function niceStep(range, n = 4) {
      const raw = range / n, p = Math.pow(10, Math.floor(Math.log10(raw)));
      const m = raw / p;
      return (m < 1.5 ? 1 : m < 3.5 ? 2 : m < 7.5 ? 5 : 10) * p;
    }
    function ticks(a, b, n) { const st = niceStep(b - a, n); const out = []; for (let v = Math.ceil(a / st - 1e-9) * st; v <= b + 1e-9; v += st) out.push(+v.toFixed(10)); return out; }
    function axes(g, A, xr, yr, o = {}) {
      const C = V.C();
      const X = (v) => A.x + ((v - xr[0]) / (xr[1] - xr[0])) * A.w;
      const Y = (v) => A.y + A.h - ((v - yr[0]) / (yr[1] - yr[0])) * A.h;
      const xt = o.xticks || ticks(xr[0], xr[1], o.nx || 5), yt = o.yticks || ticks(yr[0], yr[1], o.ny || 4);
      g.lineWidth = 1; g.strokeStyle = C.grid; g.beginPath();
      yt.forEach((v) => { const yy = Math.round(Y(v)) + 0.5; g.moveTo(A.x, yy); g.lineTo(A.x + A.w, yy); });
      xt.forEach((v) => { const xx = Math.round(X(v)) + 0.5; g.moveTo(xx, A.y); g.lineTo(xx, A.y + A.h); });
      g.stroke();
      g.strokeStyle = V.alpha(C.muted, 0.6); g.beginPath(); g.moveTo(A.x + 0.5, A.y); g.lineTo(A.x + 0.5, A.y + A.h + 0.5); g.lineTo(A.x + A.w, A.y + A.h + 0.5); g.stroke();
      const fx = o.fx || fmtTick, fy = o.fy || fmtTick;
      if (!o.noX) xt.forEach((v) => V.text(g, fx(v), X(v), A.y + A.h + 14, { size: 11, font: 'mono', color: C.muted, align: 'center' }));
      if (!o.noY) yt.forEach((v) => V.text(g, fy(v), A.x - 5, Y(v), { size: 11, font: 'mono', color: C.muted, align: 'right', base: 'middle' }));
      if (o.xlabel) sym(g, o.xlabel, A.x + A.w, A.y + A.h + 30, { size: 11.5, color: C.muted, align: 'right' });
      if (o.ylabel) sym(g, o.ylabel, A.x, A.y - 8, { size: 11.5, color: C.muted });
      return { X, Y };
    }
    function curve(g, x0, x1, f, X, Y, o = {}) {
      const n = o.n || 120;
      g.beginPath(); let on = false;
      for (let k = 0; k <= n; k++) {
        const x = x0 + ((x1 - x0) * k) / n, y = f(x);
        if (!isFinite(y)) { on = false; continue; }
        if (!on) { g.moveTo(X(x), Y(y)); on = true; } else g.lineTo(X(x), Y(y));
      }
      g.strokeStyle = o.color; g.lineWidth = o.lw || 2.2; if (o.dash) g.setLineDash(o.dash);
      g.stroke(); g.setLineDash([]);
    }
    function clipRect(g, A) { g.save(); g.beginPath(); g.rect(A.x, A.y - 2, A.w, A.h + 4); g.clip(); }
    const pickW = (ws, u) => { const s = ws.reduce((a, b) => a + b, 0); if (s <= 0) return -1; let acc = 0; for (let i = 0; i < ws.length; i++) { acc += ws[i] / s; if (u < acc) return i; } return ws.length - 1; };
    const arrowHead = (g, x, y, ang, s, col) => { g.beginPath(); g.moveTo(x, y); g.lineTo(x - s * Math.cos(ang - 0.45), y - s * Math.sin(ang - 0.45)); g.lineTo(x - s * Math.cos(ang + 0.45), y - s * Math.sin(ang + 0.45)); g.closePath(); g.fillStyle = col; g.fill(); };

    /* ======================================================== FIG 1 — duas fábricas */
    (function () {
      const fig = root.querySelector('#fig-pipeline');
      if (!fig) return;
      adaptH(fig, 410, 540);
      const MODES = {
        rlhf: {
          nodes: [
            { id: 'P', l: 'Prompts', s: 'x ~ D', k: 'data', W: [0.075, 0.3], N: [0.6, 0.06] },
            { id: 'PI', l: 'Policy', s: 'π_θ · trains', k: 'train', W: [0.275, 0.3], N: [0.6, 0.21] },
            { id: 'G', l: 'Rollouts', s: 'y ~ π_θ (vLLM)', sn: 'y ~ π_θ', k: 'engine', W: [0.49, 0.3], N: [0.6, 0.36] },
            { id: 'RM', l: 'Reward', s: 'r_φ · frozen', sn: 'r_φ · fixed', k: 'frozen', W: [0.73, 0.1], N: [0.335, 0.555] },
            { id: 'REF', l: 'Reference', s: 'π_ref · frozen', sn: 'π_ref · fixed', k: 'frozen', W: [0.73, 0.43], N: [0.6, 0.555] },
            { id: 'CR', l: 'Critic', s: 'V_ψ · trains', k: 'train', W: [0.73, 0.76], N: [0.865, 0.555] },
            { id: 'PPO', l: 'PPO', s: 'advantage + clip', sn: 'adv. + clip', k: 'op', W: [0.3, 0.78], N: [0.6, 0.76] },
          ],
          edges: [
            { a: 'P', b: 'PI', ph: 0 }, { a: 'PI', b: 'G', ph: 0 },
            { a: 'G', b: 'RM', ph: 1 }, { a: 'G', b: 'REF', ph: 1 }, { a: 'G', b: 'CR', ph: 1 },
            { a: 'RM', b: 'PPO', ph: 2 }, { a: 'REF', b: 'PPO', ph: 2 }, { a: 'CR', b: 'PPO', ph: 2, c: 0.12 },
            { a: 'PPO', b: 'PI', ph: 3, grad: true, viaN: 'left' }, { a: 'PPO', b: 'CR', ph: 3, grad: true, c: 0.12 },
          ],
          phases: [
            'generate: the policy writes whole responses, token by token (serial decode in the inference engine)',
            'score: the RM gives the score, the reference gives the KL log-probs and the critic estimates values',
            'advantage: reward r − β·KL per sample and GAE advantages using the critic',
            'update: PPO epochs with clipping, on the policy and on the critic',
          ],
        },
        grpo: {
          nodes: [
            { id: 'P', l: 'Prompts', s: 'x ~ D', k: 'data', W: [0.075, 0.3], N: [0.6, 0.06] },
            { id: 'PI', l: 'Policy', s: 'π_θ · trains', k: 'train', W: [0.275, 0.3], N: [0.6, 0.21] },
            { id: 'G', l: 'Group', s: 'y₁…y_G (vLLM)', sn: 'y₁…y_G ~ π_θ', k: 'engine', W: [0.49, 0.3], N: [0.6, 0.36] },
            { id: 'RM', l: 'Reward', s: 'r_φ or verif.', sn: 'r_φ · fixed', k: 'frozen', W: [0.73, 0.1], N: [0.335, 0.555] },
            { id: 'REF', l: 'Reference', s: 'π_ref · frozen', sn: 'π_ref · fixed', k: 'frozen', W: [0.73, 0.43], N: [0.6, 0.555] },
            { id: 'CR', l: 'Critic', s: 'cut', k: 'ghost', W: [0.73, 0.76], N: [0.865, 0.555] },
            { id: 'GA', l: 'Advantage', s: '(r − μ_G)/σ_G', k: 'op', W: [0.3, 0.78], N: [0.6, 0.76] },
          ],
          edges: [
            { a: 'P', b: 'PI', ph: 0 }, { a: 'PI', b: 'G', ph: 0 },
            { a: 'G', b: 'RM', ph: 1 }, { a: 'G', b: 'REF', ph: 1 },
            { a: 'RM', b: 'GA', ph: 2 }, { a: 'REF', b: 'GA', ph: 2 },
            { a: 'GA', b: 'PI', ph: 3, grad: true, viaN: 'left' },
          ],
          phases: [
            'generate: G responses per prompt, token by token, in the inference engine',
            'score: the RM or verifier gives the score; the reference, the KL log-probs',
            'group advantage: (score − group mean)/std, no critic',
            'update: clipped step, on the policy only (there is no critic)',
          ],
        },
        dpo: {
          nodes: [
            { id: 'D', l: 'Pairs', s: 'fixed (x, y_w, y_l)', k: 'data', W: [0.09, 0.5], N: [0.5, 0.07] },
            { id: 'PI', l: 'Policy', s: 'π_θ · trains', k: 'train', W: [0.37, 0.24], N: [0.27, 0.3] },
            { id: 'REF', l: 'Reference', s: 'π_ref · frozen', sn: 'π_ref · fixed', k: 'frozen', W: [0.37, 0.76], N: [0.73, 0.3] },
            { id: 'M', l: 'Margin', s: 'Δh = h_w − h_l', k: 'op', W: [0.64, 0.5], N: [0.5, 0.53] },
            { id: 'L', l: 'Loss', s: '−log σ(β·Δh)', k: 'op', W: [0.89, 0.5], N: [0.5, 0.76] },
          ],
          edges: [
            { a: 'D', b: 'PI', ph: 0 }, { a: 'D', b: 'REF', ph: 0 },
            { a: 'PI', b: 'M', ph: 1 }, { a: 'REF', b: 'M', ph: 1 },
            { a: 'M', b: 'L', ph: 2 },
            { a: 'L', b: 'PI', ph: 3, grad: true, viaN: 'left', viaW: 'top' },
          ],
          phases: [
            'read a pair from the dataset: nothing is generated during training',
            '4 forwards: the policy and the reference read the chosen and the rejected response (the reference without gradient)',
            'margin and loss: a logistic regression on β·Δh',
            'backward on the policy only: raises the chosen, lowers the rejected, with weight σ(−β·Δh)',
          ],
        },
      };
      const S = { mode: 'rlhf', sw: -10 };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Factory', options: [{ v: 'rlhf', label: 'RLHF with PPO' }, { v: 'grpo', label: 'GRPO (ch. 56)' }, { v: 'dpo', label: 'DPO' }], value: 'rlhf', onChange: (v) => { S.mode = v; S.sw = H1 ? H1.t : 0; upd(); } });
      const rM = UI.readout(ctl, 'models in memory', '—'), rT = UI.readout(ctl, 'training', '—'), rG = UI.readout(ctl, 'generates text in training?', '—'), rB = UI.readout(ctl, 'states, 8B full FT', '—');
      function upd() {
        if (S.mode === 'rlhf') { rM.set('4'); rT.set('2 (policy and critic)'); rG.set('yes (rollouts)', 'warn'); rB.set('≈ 288 GB (36N)', 'bad'); }
        else if (S.mode === 'grpo') { rM.set('3 (no critic)'); rT.set('1 (policy)'); rG.set('yes (G per prompt)', 'warn'); rB.set('≈ 160 GB (20N)', 'warn'); }
        else { rM.set('2'); rT.set('1 (policy)'); rG.set('no', 'ok'); rB.set('≈ 144 GB (18N)', 'ok'); }
      }
      upd();
      const cubic = (p0, p1, p2, p3, t) => { const u = 1 - t; return [u * u * u * p0[0] + 3 * u * u * t * p1[0] + 3 * u * t * t * p2[0] + t * t * t * p3[0], u * u * u * p0[1] + 3 * u * u * t * p1[1] + 3 * u * t * t * p2[1] + t * t * t * p3[1]]; };
      function boxClip(cx, cy, hw, hh, tx, ty) {
        const dx = tx - cx, dy = ty - cy;
        if (!dx && !dy) return [cx, cy];
        const s = Math.min(dx ? hw / Math.abs(dx) : Infinity, dy ? hh / Math.abs(dy) : Infinity);
        return [cx + dx * s, cy + dy * s];
      }
      function edgeGeom(e, P, bw, bh, narrow) {
        const a = P[e.a], b = P[e.b];
        const via = narrow ? e.viaN : e.viaW;
        if (via === 'left') {
          const p0 = [a.x - bw / 2 - 2, a.y], p3 = [b.x - bw / 2 - 2, b.y];
          const m = 12;
          return (t) => cubic(p0, [m, a.y], [m, b.y], p3, t);
        }
        if (via === 'top') {
          const p0 = [a.x, a.y - bh / 2 - 2], p3 = [b.x, b.y - bh / 2 - 2];
          const m = 6;
          return (t) => cubic(p0, [a.x, m], [b.x, m], p3, t);
        }
        const len = Math.hypot(b.x - a.x, b.y - a.y) || 1;
        const nx = -(b.y - a.y) / len, ny = (b.x - a.x) / len;
        const cc = e.c || 0;
        const mx = (a.x + b.x) / 2 + nx * cc * len, my = (a.y + b.y) / 2 + ny * cc * len;
        const p0 = boxClip(a.x, a.y, bw / 2 + 3, bh / 2 + 3, cc ? mx : b.x, cc ? my : b.y);
        const p2 = boxClip(b.x, b.y, bw / 2 + 3, bh / 2 + 3, cc ? mx : a.x, cc ? my : a.y);
        return (t) => { const u = 1 - t; return [u * u * p0[0] + 2 * u * t * mx + t * t * p2[0], u * u * p0[1] + 2 * u * t * my + t * t * p2[1]]; };
      }
      const PH = 2.0;
      const H1 = V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const narrow = w < 560;
        const M = MODES[S.mode];
        const fade = clamp((t - S.sw) / 0.45, 0, 1);
        const areaH = h - (narrow ? 64 : 48);
        const bw = narrow ? Math.min(90, w * 0.245) : Math.min(128, w * 0.14), bh = narrow ? 44 : 48;
        const P = {};
        M.nodes.forEach((n) => { const q = narrow ? n.N : n.W; P[n.id] = { x: q[0] * w, y: 12 + q[1] * (areaH - 24) + bh / 2, n }; });
        const ph = Math.floor(t / PH) % 4, f = (t % PH) / PH;
        const activeNodes = new Set();
        M.edges.forEach((e) => { if (e.ph === ph) { activeNodes.add(e.a); activeNodes.add(e.b); } });
        g.globalAlpha = fade;
        // arestas
        M.edges.forEach((e) => {
          const at = edgeGeom(e, P, bw, bh, narrow);
          const act = e.ph === ph;
          const col = e.grad ? C.d : C.a;
          g.beginPath();
          for (let k = 0; k <= 30; k++) { const p = at(k / 30); if (!k) g.moveTo(p[0], p[1]); else g.lineTo(p[0], p[1]); }
          g.strokeStyle = act ? col : V.alpha(C.muted, 0.45); g.lineWidth = act ? 2.2 : 1.3;
          if (e.grad) { g.setLineDash([6, 5]); g.lineDashOffset = -t * 30; }
          g.stroke(); g.setLineDash([]); g.lineDashOffset = 0;
          const pe = at(1), pq = at(0.94);
          arrowHead(g, pe[0], pe[1], Math.atan2(pe[1] - pq[1], pe[0] - pq[0]), 8, act ? col : V.alpha(C.muted, 0.6));
          if (act) for (let k = 0; k < 3; k++) { const u = (f + k / 3) % 1; const p = at(u); V.glow(g, p[0], p[1], 9, col, 0.9); V.dot(g, p[0], p[1], 2.6, col); }
        });
        // nós
        M.nodes.forEach((n) => {
          const p = P[n.id], x = p.x - bw / 2, y = p.y - bh / 2;
          if (n.k === 'ghost') {
            // o crítico que o GRPO cortou: caixa apagada e riscada
            g.save(); g.globalAlpha = fade * 0.55;
            V.rr(g, x, y, bw, bh, 9); g.strokeStyle = V.alpha(C.muted, 0.8); g.lineWidth = 1.2; g.setLineDash([3, 4]); g.stroke(); g.setLineDash([]);
            V.text(g, n.l, p.x, p.y - 4, { size: narrow ? 11.5 : 12.5, weight: 700, align: 'center', color: C.muted });
            V.text(g, n.s, p.x, p.y + 13, { size: 11, align: 'center', color: C.muted, font: 'mono' });
            g.globalAlpha = fade * 0.9;
            V.line(g, x + 6, y + bh - 6, x + bw - 6, y + 6, { color: RED(), lw: 2 });
            g.restore();
            return;
          }
          const col = n.k === 'train' ? C.b : n.k === 'frozen' ? C.muted : n.k === 'data' ? C.a : n.k === 'engine' ? C.d : C.c;
          const act = activeNodes.has(n.id);
          if (act && n.k === 'train') V.glow(g, p.x, p.y, bw * 0.75, C.b, 0.25 + 0.15 * Math.sin(t * 5));
          V.rr(g, x, y, bw, bh, 9);
          g.fillStyle = V.alpha(col, act ? 0.24 : 0.12); g.fill();
          g.strokeStyle = act ? col : V.alpha(col, 0.7); g.lineWidth = act ? 2.2 : 1.4;
          if (n.k === 'frozen') g.setLineDash([5, 4]);
          g.stroke(); g.setLineDash([]);
          V.text(g, n.l, p.x, p.y - 4, { size: narrow ? 11.5 : 12.5, weight: 700, align: 'center', color: C.ink });
          sym(g, narrow && n.sn ? n.sn : n.s, p.x, p.y + 13, { size: 11, align: 'center', color: C.muted, font: 'mono' });
          // detalhes lúdicos
          if (n.id === 'G' && ph === 0) {
            const rows = S.mode === 'grpo' ? 3 : 1;
            for (let r = 0; r < rows; r++) {
              const nt = Math.min(6, Math.floor(f * (8 - r * 1.3)));
              for (let k = 0; k < nt; k++) { const s = 9; V.box(g, p.x - (6 * (s + 3)) / 2 + k * (s + 3), y + bh + 5 + r * 12, s, s, { r: 2, fill: V.alpha(C.d, 0.75 - r * 0.12) }); }
            }
          }
          if (n.id === 'D') {
            const off = Math.sin(t * 2) * 2;
            chip(g, p.x - bw / 2 + 4, y + bh + 6 + off, bw / 2 - 6, 18, 'y_w ✓', { fill: V.alpha(C.a, 0.2), stroke: C.a, size: 11 });
            chip(g, p.x + 2, y + bh + 6 - off, bw / 2 - 6, 18, 'y_l ✗', { fill: V.alpha(C.c, 0.2), stroke: C.c, size: 11 });
          }
          if (n.id === 'RM' && S.mode === 'rlhf' && !narrow) V.text(g, 'trained beforehand, on pairs', p.x, y - 6, { size: 11, align: 'center', color: C.muted });
        });
        g.globalAlpha = 1;
        // faixa da fase
        const by = h - (narrow ? 58 : 42);
        V.box(g, 8, by, w - 16, narrow ? 50 : 34, { r: 8, fill: V.alpha(C.panel, 0.85), stroke: V.alpha(C.muted, 0.3), lw: 1 });
        const tag = 'step ' + (ph + 1) + '/4';
        V.text(g, tag, 18, by + 21, { size: 12, weight: 700, color: S.mode === 'dpo' ? C.a : C.d, font: 'mono' });
        const tx = 18 + 64;
        V.wrap(g, M.phases[ph], tx, by + (narrow ? 18 : 21), w - tx - 16, { size: 12, color: C.ink, lineH: 16, maxLines: 2 });
        // barra de progresso da fase
        g.fillStyle = V.alpha(S.mode === 'dpo' ? C.a : C.d, 0.7); g.fillRect(10, by + (narrow ? 48 : 32), (w - 20) * f, 2);
      });
    })();

    /* ======================================================== FIG 2 — inclinação exponencial */
    (function () {
      const fig = root.querySelector('#fig-inclinacao');
      if (!fig) return;
      adaptH(fig, 410, 450);
      const LAB = ['short ✓', 'long ✓', 'vague', 'wrong', 'refusal', 'joke'];
      const REF = [0.30, 0.22, 0.18, 0.12, 0.10, 0.08];
      const R0 = [1.0, 1.4, 0.2, -1.0, -0.6, -0.2];
      const R = R0.slice();
      const BET = [0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1, 1.5, 2, 3, 5, 10];
      const S = { bi: 6, cur: REF.slice(), tgt: REF.slice(), parts: [], drag: -1, scale: 0.4 };
      const rng = V.rng(57);
      const ctl = UI.controls(fig);
      UI.slider(ctl, { label: 'β', min: 0, max: BET.length - 1, step: 1, value: S.bi, fmt: (v) => fmtTick(BET[v]), onInput: (v) => { S.bi = v; target(); } });
      UI.button(ctl, '↺ original rewards', () => { for (let i = 0; i < R.length; i++) R[i] = R0[i]; target(); });
      const rZ = UI.readout(ctl, 'Z(x)', '—'), rK = UI.readout(ctl, 'KL(π*‖π ref)', '—'), rE = UI.readout(ctl, 'E[r]: π ref → π*', '—');
      htmlLabel(rK, 'KL(π*‖π<sub>ref</sub>)'); htmlLabel(rE, 'E[r]: π<sub>ref</sub> → π*');
      nocase(ctl);
      function target() {
        const b = BET[S.bi];
        const lw = REF.map((p, i) => Math.log(p) + R[i] / b);
        const m = Math.max(...lw);
        const e = lw.map((v) => Math.exp(v - m));
        const s = e.reduce((a, c) => a + c, 0);
        S.tgt = e.map((v) => v / s);
        const logZ = m + Math.log(s);
        const kl = S.tgt.reduce((a, p, i) => a + (p > 1e-15 ? p * Math.log(p / REF[i]) : 0), 0);
        const e0 = REF.reduce((a, p, i) => a + p * R[i], 0), e1 = S.tgt.reduce((a, p, i) => a + p * R[i], 0);
        rZ.set(fmtBig(Math.exp(logZ)));
        rK.set(fmt(kl, 3) + ' nats');
        rE.set(fmt(e0, 2) + ' → ' + fmt(e1, 2), 'ok');
      }
      target();
      const fac = (x) => { if (x >= 1000 || x < 0.01) { const q = x.toExponential(1).split('e'); return '×' + q[0] + 'e' + (+q[1]); } return '×' + fmt(x, x >= 10 ? 1 : 2); };
      V.fig(fig, (g, w, h, t, dt, _s, HH) => {
        const C = V.C();
        const narrow = w < 520;
        const pad = 10, n = 6, colW = (w - 2 * pad) / n;
        const laneTop = 34, laneH = narrow ? 66 : 74, laneBot = laneTop + laneH;
        const rY = (r) => laneTop + (1 - (r + 2) / 4) * laneH;
        const P = HH.pointer;
        if (P.down) {
          if (S.drag < 0 && P.y > laneTop - 14 && P.y < laneBot + 14) S.drag = clamp(Math.floor((P.x - pad) / colW), 0, n - 1);
          if (S.drag >= 0) { const r = clamp(((laneTop + laneH - P.y) / laneH) * 4 - 2, -2, 2); const rr = Math.round(r * 10) / 10; if (rr !== R[S.drag]) { R[S.drag] = rr; target(); } }
        } else S.drag = -1;
        V.text(g, 'reward r(x, y) · drag the dots', pad, 18, { size: 11.5, weight: 600, color: C.muted });
        V.line(g, pad, rY(0), w - pad, rY(0), { color: V.alpha(C.muted, 0.5), dash: [3, 4], lw: 1 });
        // dinâmica de massa
        const diff = S.tgt.map((v, i) => v - S.cur[i]);
        const moving = diff.reduce((a, d) => a + Math.abs(d), 0) / 2;
        for (let i = 0; i < n; i++) S.cur[i] += diff[i] * Math.min(1, dt * 3);
        if (!LA.paused && moving > 0.004 && S.parts.length < 90 && rng() < Math.min(0.95, moving * 7)) {
          const a = pickW(diff.map((d) => Math.max(0, -d)), rng()), b = pickW(diff.map((d) => Math.max(0, d)), rng());
          if (a >= 0 && b >= 0 && a !== b) S.parts.push({ a, b, t: 0, j: rng() });
        }
        const maxP = Math.max(...S.cur, ...REF);
        S.scale += (Math.max(0.36, maxP * 1.2) - S.scale) * Math.min(1, dt * 3);
        const facY = laneBot + 14, facH = 20;
        const barTop = facY + facH + 44, barBot = h - 26, barH = barBot - barTop;
        V.text(g, '× e^(r/β)', pad, facY - 3, { size: 11, color: C.muted, font: 'mono' });
        sym(g, 'π_ref (dashed) and π* (solid)', pad, barTop - 14, { size: 11.5, weight: 600, color: C.muted });
        const b = BET[S.bi];
        const barX = (i) => pad + colW * i + colW * 0.2, bwid = colW * 0.6;
        for (let i = 0; i < n; i++) {
          const cx = pad + colW * (i + 0.5);
          // trilho e bolinha da recompensa
          V.line(g, cx, rY(2), cx, rY(-2), { color: V.alpha(C.muted, 0.35), lw: 2 });
          const ky = rY(R[i]), kc = R[i] >= 0 ? C.a : C.c;
          if (S.drag === i) V.glow(g, cx, ky, 18, kc, 0.8);
          V.dot(g, cx, ky, 7, kc);
          if (narrow) {
            // colunas estreitas: rótulo centrado acima da bolinha (abaixo, se não couber)
            const above = ky - 11 > laneTop - 4;
            V.text(g, fmtS(R[i], 1), cx, above ? ky - 11 : ky + 19, { size: 11, font: 'mono', color: C.ink, align: 'center' });
          } else {
            const lastCol = cx + 46 > w;
            V.text(g, fmtS(R[i], 1), lastCol ? cx - 10 : cx + 10, ky + 4, { size: 11, font: 'mono', color: C.ink, align: lastCol ? 'right' : 'left' });
          }
          // fator
          const fv = Math.exp(R[i] / b);
          const fw = Math.min(colW - 6, 66);
          chip(g, cx - fw / 2, facY + 4, fw, facH, fac(fv), { size: 11, fill: V.alpha(fv >= 1 ? C.a : C.c, 0.14), stroke: V.alpha(fv >= 1 ? C.a : C.c, 0.6) });
          // barras
          const hr = (REF[i] / S.scale) * barH, hc = (S.cur[i] / S.scale) * barH;
          V.rr(g, barX(i), barBot - hc, bwid, Math.max(1, hc), 4); g.fillStyle = V.alpha(C.b, 0.78); g.fill();
          g.setLineDash([4, 3]); V.rr(g, barX(i) - 3, barBot - hr, bwid + 6, Math.max(1, hr), 4); g.strokeStyle = V.alpha(C.ink, 0.6); g.lineWidth = 1.3; g.stroke(); g.setLineDash([]);
          V.text(g, Math.round(S.cur[i] * 100) + '%', cx, barBot - Math.max(hc, hr) - 6, { size: 11, font: 'mono', color: C.ink, align: 'center' });
          V.text(g, LAB[i], cx, barBot + 15, { size: narrow ? 11 : 11.5, color: C.ink, align: 'center', maxW: colW - 2 });
        }
        V.line(g, pad, barBot + 0.5, w - pad, barBot + 0.5, { color: V.alpha(C.muted, 0.6), lw: 1 });
        // grãos de massa voando
        for (let k = S.parts.length - 1; k >= 0; k--) {
          const q = S.parts[k]; q.t += dt * 1.5;
          if (q.t >= 1) { S.parts.splice(k, 1); continue; }
          const ax = barX(q.a) + bwid / 2, bx2 = barX(q.b) + bwid / 2;
          const ay = barBot - (S.cur[q.a] / S.scale) * barH, by2 = barBot - (S.cur[q.b] / S.scale) * barH;
          const mx = (ax + bx2) / 2, my = Math.min(ay, by2) - 30 - q.j * 30;
          const u = V.ease.inOut(q.t), uu = 1 - u;
          const x = uu * uu * ax + 2 * uu * u * mx + u * u * bx2, y = uu * uu * ay + 2 * uu * u * my + u * u * by2;
          V.glow(g, x, y, 7, C.b, 0.7); V.dot(g, x, y, 2.4, C.b);
        }
        V.text(g, 'β = ' + fmtTick(b), w - pad, 18, { size: 12, weight: 700, color: C.b, align: 'right', font: 'mono' });
      });
    })();

    /* ======================================================== FIG 3 — paisagem no simplex */
    (function () {
      const fig = root.querySelector('#fig-paisagem');
      if (!fig) return;
      adaptH(fig, 430, 475);
      const REF = [0.6, 0.3, 0.1], R = [0, 1, 2], NM = ['A', 'B', 'C'];
      const BET = [0.2, 0.3, 0.5, 0.75, 1, 1.5, 2, 3];
      const STARTS = [[0.05, 0.05, 0.9], [0.9, 0.06, 0.04], [0.06, 0.88, 0.06], [0.34, 0.33, 0.33], [0.72, 0.04, 0.24]];
      const S = { bi: 4, pi: STARTS[0].slice(), ball: true, auto: true, si: 0, hold: 0, trail: [], key: '', cache: null };
      const norm = (p) => { const s = p[0] + p[1] + p[2]; return p.map((v) => v / s); };
      const J = (p, b) => { let s = 0; for (let i = 0; i < 3; i++) { s += p[i] * R[i]; if (p[i] > 1e-12) s -= b * p[i] * Math.log(p[i] / REF[i]); } return s; };
      const KL = (p, q) => { let s = 0; for (let i = 0; i < 3; i++) if (p[i] > 1e-12) s += p[i] * Math.log(p[i] / q[i]); return s; };
      const star = (b) => { const ww = REF.map((p, i) => p * Math.exp(R[i] / b)); const z = ww[0] + ww[1] + ww[2]; return { p: ww.map((v) => v / z), Z: z }; };
      const ctl = UI.controls(fig);
      UI.slider(ctl, { label: 'β', min: 0, max: BET.length - 1, step: 1, value: S.bi, fmt: (v) => fmtTick(BET[v]), onInput: (v) => { S.bi = v; S.trail = []; S.hold = 0; } });
      UI.button(ctl, '▶ release the ball', () => { S.ball = true; S.auto = false; S.trail = []; S.pi = norm(S.pi.map((v) => Math.max(v, 0.004))); }, { primary: true });
      const rJ = UI.readout(ctl, 'J(π)', '—'), rM = UI.readout(ctl, 'maximum β·log Z', '—'), rK = UI.readout(ctl, 'β·KL(π‖π*)', '—');
      nocase(ctl);
      function geo(w, h) {
        const narrow = w < 600;
        const side = narrow ? Math.min(w - 56, (h - 190) * 2 / Math.sqrt(3)) : Math.min(w * 0.5, (h - 64) * 2 / Math.sqrt(3));
        const th = (side * Math.sqrt(3)) / 2;
        const cx = narrow ? w / 2 : w * 0.31;
        const top = narrow ? 26 : (h - th) / 2 - 2;
        return { narrow, side, th, cx, A: [cx - side / 2, top + th], B: [cx + side / 2, top + th], T: [cx, top] };
      }
      const pt = (G, p) => [G.A[0] * p[0] + G.B[0] * p[1] + G.T[0] * p[2], G.A[1] * p[0] + G.B[1] * p[1] + G.T[1] * p[2]];
      function bary(G, x, y) {
        const v0 = [G.B[0] - G.A[0], G.B[1] - G.A[1]], v1 = [G.T[0] - G.A[0], G.T[1] - G.A[1]], v2 = [x - G.A[0], y - G.A[1]];
        const d00 = v0[0] * v0[0] + v0[1] * v0[1], d01 = v0[0] * v1[0] + v0[1] * v1[1], d11 = v1[0] * v1[0] + v1[1] * v1[1];
        const d20 = v2[0] * v0[0] + v2[1] * v0[1], d21 = v2[0] * v1[0] + v2[1] * v1[1];
        const den = d00 * d11 - d01 * d01;
        const pb = (d11 * d20 - d01 * d21) / den, pc = (d00 * d21 - d01 * d20) / den, pa = 1 - pb - pc;
        const raw = [pa, pb, pc];
        const ok = Math.min(...raw) > -0.1;
        return { ok, p: norm(raw.map((v) => Math.max(v, 0.002))) };
      }
      function heat(G, w, h, b, C) {
        const key = [b, Math.round(w), Math.round(h), C.dark].join('|');
        if (S.key === key && S.cache) return S.cache;
        const dpr = Math.min(window.devicePixelRatio || 1, 2);
        const cv = document.createElement('canvas'); cv.width = Math.max(1, Math.round(w * dpr)); cv.height = Math.max(1, Math.round(h * dpr));
        const q = cv.getContext('2d'); q.setTransform(dpr, 0, 0, dpr, 0, 0);
        const N = 34, tris = [];
        let jmin = Infinity;
        const jmax = b * Math.log(star(b).Z);
        for (let a = 0; a < N; a++) for (let c = 0; c < N - a; c++) {
          const up = [[a, c], [a + 1, c], [a, c + 1]];
          const pu = [(3 * a + 1) / (3 * N), (3 * c + 1) / (3 * N)];
          tris.push({ v: up, j: J([pu[0], pu[1], 1 - pu[0] - pu[1]], b) });
          if (a + c <= N - 2) {
            const dn = [[a + 1, c], [a, c + 1], [a + 1, c + 1]];
            const pd = [(3 * a + 2) / (3 * N), (3 * c + 2) / (3 * N)];
            tris.push({ v: dn, j: J([pd[0], pd[1], 1 - pd[0] - pd[1]], b) });
          }
        }
        tris.forEach((tr) => { if (tr.j < jmin) jmin = tr.j; });
        tris.forEach((tr) => {
          const P = tr.v.map(([a, c]) => pt(G, [a / N, c / N, 1 - (a + c) / N]));
          const col = V.cmap(Math.pow(clamp((tr.j - jmin) / (jmax - jmin || 1), 0, 1), 1.6), 'seq');
          q.beginPath(); q.moveTo(P[0][0], P[0][1]); q.lineTo(P[1][0], P[1][1]); q.lineTo(P[2][0], P[2][1]); q.closePath();
          q.fillStyle = col; q.fill(); q.strokeStyle = col; q.lineWidth = 0.7; q.stroke();
        });
        S.key = key; S.cache = cv; S.jmin = jmin;
        return cv;
      }
      V.fig(fig, (g, w, h, t, dt, _s, HH) => {
        const C = V.C();
        const G = geo(w, h), b = BET[S.bi], st = star(b);
        g.drawImage(heat(G, w, h, b, C), 0, 0, w, h);
        g.beginPath(); g.moveTo(G.A[0], G.A[1]); g.lineTo(G.B[0], G.B[1]); g.lineTo(G.T[0], G.T[1]); g.closePath();
        g.strokeStyle = V.alpha(C.ink, 0.5); g.lineWidth = 1.4; g.stroke();
        const P = HH.pointer;
        if (P.down) { const q = bary(G, P.x, P.y); if (q.ok) { S.pi = q.p; S.ball = false; S.auto = false; S.trail = []; } }
        if (S.ball && !LA.paused) {
          const eta = (3 * Math.min(dt, 0.05)) / b;
          const gr = S.pi.map((p, i) => R[i] - b * Math.log(Math.max(p, 1e-12) / REF[i]));
          const m = Math.max(...gr.map((v) => eta * v));
          S.pi = norm(S.pi.map((p, i) => p * Math.exp(eta * gr[i] - m)));
          S.trail.push(S.pi.slice()); if (S.trail.length > 260) S.trail.shift();
          if (KL(S.pi, st.p) < 3e-5) { S.hold += dt; if (S.hold > 1.6 && S.auto) { S.si = (S.si + 1) % STARTS.length; S.pi = STARTS[S.si].slice(); S.trail = []; S.hold = 0; } } else S.hold = 0;
        }
        // trilha
        if (S.trail.length > 1) {
          g.beginPath(); S.trail.forEach((p, k) => { const q = pt(G, p); if (!k) g.moveTo(q[0], q[1]); else g.lineTo(q[0], q[1]); });
          g.strokeStyle = V.alpha(C.b, 0.85); g.lineWidth = 2; g.setLineDash([4, 3]); g.stroke(); g.setLineDash([]);
        }
        // cantos
        const corner = [G.A, G.B, G.T];
        const off = [[-6, 20], [6, 20], [0, -10]];
        for (let i = 0; i < 3; i++) {
          const al = i === 0 ? 'left' : i === 1 ? 'right' : 'center';
          const xx = corner[i][0] + (i === 0 ? -14 : i === 1 ? 14 : 0), yy = corner[i][1] + off[i][1];
          V.text(g, 'only ' + NM[i] + ' · r = ' + R[i], i === 2 ? xx : xx + (i === 0 ? 10 : -10), yy, { size: 11.5, weight: 600, color: C.ink, align: al });
        }
        // π_ref
        const qr = pt(G, REF);
        g.beginPath(); g.arc(qr[0], qr[1], 8, 0, Math.PI * 2); g.strokeStyle = C.ink; g.lineWidth = 2; g.stroke();
        sym(g, 'π_ref', qr[0] + 11, qr[1] + 16, { size: 12, weight: 700, color: C.ink });
        // π*
        const qs = pt(G, st.p);
        g.beginPath();
        for (let k = 0; k < 10; k++) { const rr = k % 2 ? 4.5 : 11, a = -Math.PI / 2 + (k * Math.PI) / 5; const x = qs[0] + rr * Math.cos(a), y = qs[1] + rr * Math.sin(a); if (!k) g.moveTo(x, y); else g.lineTo(x, y); }
        g.closePath(); g.fillStyle = C.d; g.fill(); g.strokeStyle = C.dark ? '#000' : '#fff'; g.lineWidth = 1; g.stroke();
        V.text(g, 'π*', qs[0] + 13, qs[1] - 8, { size: 12.5, weight: 700, color: C.d });
        // π atual
        const qp = pt(G, S.pi);
        V.glow(g, qp[0], qp[1], 20, C.b, 0.9); V.dot(g, qp[0], qp[1], 6.5, C.b);
        g.beginPath(); g.arc(qp[0], qp[1], 6.5, 0, Math.PI * 2); g.strokeStyle = C.dark ? '#fff' : '#000'; g.lineWidth = 1.2; g.stroke();
        V.text(g, 'π', qp[0] - 10, qp[1] - 9, { size: 12.5, weight: 700, color: C.b, align: 'right' });
        // números
        const jp = J(S.pi, b), jm = b * Math.log(st.Z), gap = b * KL(S.pi, st.p);
        rJ.set(fmt(jp, 3)); rM.set(fmt(jm, 3), 'ok'); rK.set(fmt(gap, 3));
        const lines = [
          ['J(π) = E_π[r] − β·KL(π‖π_ref)', C.ink, 12, 600],
          ['      = β·log Z − β·KL(π‖π*)', C.ink, 12, 600],
          ['π  = (' + S.pi.map((v) => fmt(v, 2)).join(', ') + ')', C.b, 12, 500],
          ['π* = (' + st.p.map((v) => fmt(v, 2)).join(', ') + ')', C.d, 12, 500],
          ['J(π) + β·KL(π‖π*) = ' + fmt(jp, 3) + ' + ' + fmt(gap, 3), C.ink, 12, 400],
          ['                    = ' + fmt(jp + gap, 3) + ' = β·log Z ✓', C.a, 12, 600],
        ];
        if (!G.narrow) {
          const x0 = G.cx + G.side / 2 + 34;
          let y0 = Math.max(40, h / 2 - 80);
          lines.forEach((L) => { sym(g, L[0], x0, y0, { size: L[2], weight: L[3], color: L[1], font: 'mono' }); y0 += 24; });
          // barra de cor
          const cbw = Math.min(180, w - x0 - 16), cby = y0 + 14;
          for (let k = 0; k < 40; k++) { g.fillStyle = V.cmap(Math.pow(k / 39, 1.6), 'seq'); g.fillRect(x0 + (k * cbw) / 40, cby, cbw / 40 + 1, 12); }
          V.text(g, 'low J', x0, cby + 27, { size: 11, color: C.muted });
          V.text(g, 'high J', x0 + cbw, cby + 27, { size: 11, color: C.muted, align: 'right' });
        } else {
          let y0 = G.A[1] + 44;
          const L2 = [lines[0], lines[1], lines[2], lines[3], ['J(π) + β·KL(π‖π*) = ' + fmt(jp + gap, 3) + ' ✓', C.a, 12, 600]];
          L2.forEach((L) => { sym(g, L[0], 12, y0, { size: 11.5, weight: L[3], color: L[1], font: 'mono' }); y0 += 21; });
          // barra de cor (a mesma do layout largo), numa linha só: low J [■■■] high J
          g.font = V.font(11);
          const lw0 = g.measureText('low J').width, lw1 = g.measureText('high J').width;
          const cbx = 12 + lw0 + 8, cbw = Math.min(200, w - 12 - lw1 - 8 - cbx), cby = y0 - 4;
          for (let k = 0; k < 40; k++) { g.fillStyle = V.cmap(Math.pow(k / 39, 1.6), 'seq'); g.fillRect(cbx + (k * cbw) / 40, cby, cbw / 40 + 1, 12); }
          g.strokeStyle = V.alpha(C.muted, 0.5); g.lineWidth = 1; g.strokeRect(cbx + 0.5, cby + 0.5, cbw - 1, 11);
          V.text(g, 'low J', 12, cby + 10, { size: 11, color: C.muted });
          V.text(g, 'high J', cbx + cbw + 8, cby + 10, { size: 11, color: C.muted });
        }
      });
    })();

    /* ======================================================== FIG 4 — o Z que evapora */
    (function () {
      const fig = root.querySelector('#fig-cancela');
      if (!fig) return;
      adaptH(fig, 400, 560);
      const BH = { w: 0.6, l: -0.4 };
      const S = { f: 0.8, fs: 0.8, flash: -10 };
      const rng = V.rng(11);
      const DUST = Array.from({ length: 26 }, () => ({ x: rng(), y: rng(), v: 0.6 + rng() * 0.8 }));
      const steps = [
        { t: 'Bradley–Terry only sees differences', d: 'The balance turns the <b>difference</b> in rewards into a probability: \\(P(y_w\\succ y_l)=\\sigma(r_w-r_l)\\). Move the slider: adding the same \\(f(x)\\) to both scores shifts the arrows, but the balance does not move.' },
        { t: 'Write each r via the inversion', d: 'By Eq. 5, each reward becomes two pieces: \\(\\beta\\log\\frac{\\pi^*(y\\mid x)}{\\pi_{\\text{ref}}(y\\mid x)}\\) (colored, depends on the response) and \\(\\beta\\log Z(x)\\) (gray, depends only on the prompt).' },
        { t: 'Same prompt, same Z', d: 'Both responses compete for the <b>same</b> \\(x\\). So the gray piece is identical in both arrows, whatever its value.' },
        { t: 'Subtract: Z evaporates', d: 'In the difference \\(r_w-r_l\\), the two \\(\\beta\\log Z(x)\\) cancel out. The intractable term, a sum over every possible response, never has to be computed.' },
        { t: 'Only the policy is left', d: 'What remains is Eq. 6: \\(P=\\sigma\\big(\\beta\\log\\frac{\\pi^*(y_w\\mid x)}{\\pi_{\\text{ref}}(y_w\\mid x)}-\\beta\\log\\frac{\\pi^*(y_l\\mid x)}{\\pi_{\\text{ref}}(y_l\\mid x)}\\big)\\). Replace \\(\\pi^*\\) with \\(\\pi_\\theta\\) and maximize the likelihood: that is the DPO loss.' },
      ];
      let H4 = null;
      const st = UI.stepper(fig, steps, { autoplay: 6500, onStep: () => H4 && H4.redraw() });
      const ctl = UI.controls(fig);
      UI.slider(ctl, { label: 'f(x) = β·log Z(x)', min: -1.5, max: 1.5, step: 0.1, value: 0.8, fmt: (v) => fmtS(v, 1), onInput: (v) => { S.f = v; S.flash = H4 ? H4.t : 0; } });
      const gap = BH.w - BH.l;
      const rG4 = UI.readout(ctl, 'rw − rl', fmt(gap, 2)); rG4.set(fmt(gap, 2), 'ok'); htmlLabel(rG4, 'r<sub>w</sub> − r<sub>l</sub>');
      const rP4 = UI.readout(ctl, 'P(yw ≻ yl)', ''); rP4.set(fmt(sig(gap), 3), 'ok'); htmlLabel(rP4, 'P(y<sub>w</sub> ≻ y<sub>l</sub>)');
      nocase(ctl);
      function arrowBar(g, x, bw, y0, y1, fill, stroke, o = {}) {
        const up = y1 < y0, hh = Math.abs(y1 - y0);
        if (hh < 1) return;
        const head = Math.min(12, hh * 0.45);
        g.beginPath();
        if (o.head) {
          const yb = up ? y1 + head : y1 - head;
          g.moveTo(x - bw / 2, y0); g.lineTo(x - bw / 2, yb); g.lineTo(x - bw / 2 - 5, yb); g.lineTo(x, y1); g.lineTo(x + bw / 2 + 5, yb); g.lineTo(x + bw / 2, yb); g.lineTo(x + bw / 2, y0); g.closePath();
        } else g.rect(x - bw / 2, Math.min(y0, y1), bw, hh);
        g.fillStyle = fill; g.fill();
        if (stroke) { g.strokeStyle = stroke; g.lineWidth = 1.3; if (o.dash) g.setLineDash(o.dash); g.stroke(); g.setLineDash([]); }
      }
      H4 = V.fig(fig, (g, w, h, t, dt) => {
        const C = V.C();
        const narrow = w < 560;
        const i = st.i, el = st.elapsed();
        S.fs += (S.f - S.fs) * Math.min(1, dt * 6);
        const F = S.fs;
        const A = narrow ? { x: 8, y: 10, w: w - 16, h: h * 0.6 } : { x: 8, y: 10, w: w * 0.6, h: h - 20 };
        const B = narrow ? { x: 8, y: h * 0.6 + 8, w: w - 16, h: h * 0.4 - 16 } : { x: w * 0.62, y: 10, w: w * 0.38 - 10, h: h - 20 };
        const yr = [-2.0, 3.2];
        const ax = A.x + 34;
        const Y = (v) => A.y + 18 + (1 - (v - yr[0]) / (yr[1] - yr[0])) * (A.h - 44);
        for (let v = -2; v <= 3; v++) { V.line(g, ax, Y(v), A.x + A.w - 6, Y(v), { color: C.grid, lw: 1 }); V.text(g, String(v), ax - 6, Y(v) + 4, { size: 11, font: 'mono', color: C.muted, align: 'right' }); }
        V.line(g, ax, Y(0), A.x + A.w - 6, Y(0), { color: V.alpha(C.muted, 0.7), lw: 1.3 });
        V.text(g, 'reward', ax, A.y + 8, { size: 11.5, weight: 600, color: C.muted });
        let grayA = i >= 1 ? 1 : 0, off = F;
        if (i === 3) { grayA = 1 - clamp(el / 1.1, 0, 1); off = F * (1 - V.ease.inOut(clamp((el - 0.6) / 1.0, 0, 1))); }
        if (i >= 4) { grayA = 0; off = 0; }
        const cw = (A.w - 34) / 2;
        const cols = [{ k: 'w', x: ax + cw * 0.55, bh: BH.w, col: C.a, lab: 'y_w' }, { k: 'l', x: ax + cw * 1.3, bh: BH.l, col: C.c, lab: 'y_l' }];
        const bw = Math.min(40, cw * 0.36);
        cols.forEach((c) => {
          const r = off + c.bh;
          if (i === 0) arrowBar(g, c.x, bw, Y(0), Y(F + c.bh), V.alpha(C.b, 0.55), C.b, { head: true });
          else {
            if (grayA > 0.01) {
              g.globalAlpha = grayA;
              arrowBar(g, c.x, bw, Y(0), Y(off), V.alpha(C.muted, 0.32), C.muted, { dash: [4, 3] });
              if (Math.abs(Y(off) - Y(0)) > 14) V.text(g, 'β·log Z', c.x - bw / 2 - 5, (Y(0) + Y(off)) / 2 + 4, { size: 11, color: C.muted, align: 'right', font: 'mono' });
              g.globalAlpha = 1;
            }
            arrowBar(g, c.x, bw, Y(off), Y(r), V.alpha(c.col, 0.6), c.col, { head: true });
          }
          // rótulo acima da pilha
          const rv = i === 0 ? F + c.bh : r;
          const topV = i === 0 ? Math.max(0, rv) : Math.max(0, off, rv);
          sym(g, (i >= 4 ? 'β·h_' + c.k : 'r_' + c.k) + ' = ' + fmtS(rv, 2), c.x, Y(topV) - 10, { size: 11.5, align: 'center', color: C.ink, font: 'mono', weight: 600 });
          sym(g, c.lab, c.x, Y(yr[0]) + 18, { size: 12.5, align: 'center', weight: 700, color: c.col });
          // poeira subindo (passo 3)
          if (i === 3 && el < 1.5 && Math.abs(F) > 0.05) {
            DUST.forEach((d) => {
              const y0 = Y(0) + (Y(F) - Y(0)) * d.y, x0 = c.x - bw / 2 + bw * d.x;
              const yy = y0 - el * 60 * d.v;
              V.dot(g, x0, yy, 2, V.alpha(C.muted, clamp(1 - el / 1.4, 0, 1)));
            });
          }
        });
        // ligação idêntica (passo 2)
        if (i === 2) {
          const ym = (Y(0) + Y(F)) / 2;
          g.setLineDash([6, 5]); g.lineDashOffset = -t * 25;
          V.line(g, cols[0].x + bw / 2 + 4, ym, cols[1].x - bw / 2 - 4, ym, { color: C.d, lw: 2 });
          g.setLineDash([]); g.lineDashOffset = 0;
          chip(g, (cols[0].x + cols[1].x) / 2 - 30, ym - 12, 60, 22, 'equal', { size: 11, fill: V.alpha(C.d, 0.2), stroke: C.d, font: 'sans', weight: 700 });
        }
        // colchete da diferença
        const rw = (i === 0 ? F : off) + BH.w, rl = (i === 0 ? F : off) + BH.l;
        const bx = narrow ? Math.min(cols[1].x + bw / 2 + 26, A.x + A.w - 8) : cols[1].x + bw / 2 + 58;
        if (bx <= A.x + A.w) {
          V.line(g, cols[0].x + bw / 2 + 2, Y(rw), bx, Y(rw), { color: V.alpha(C.ink, 0.4), dash: [2, 3], lw: 1 });
          V.line(g, cols[1].x + bw / 2 + 2, Y(rl), bx, Y(rl), { color: V.alpha(C.ink, 0.4), dash: [2, 3], lw: 1 });
          V.line(g, bx, Y(rw), bx, Y(rl), { color: C.d, lw: 2.2 });
          if (narrow) V.text(g, 'Δ = ' + fmt(gap, 2), bx + 4, Y(Math.max(rw, rl)) - 8, { size: 12, color: C.d, font: 'mono', weight: 700, align: 'right' });
          else V.text(g, 'Δ = ' + fmt(gap, 2), bx + 6, (Y(rw) + Y(rl)) / 2 + 4, { size: 12, color: C.d, font: 'mono', weight: 700 });
        }
        // balança
        const P = sig(gap);
        const px = B.x + B.w / 2, py = B.y + B.h * 0.36;
        const L = Math.min(B.w * 0.78, 230);
        const ang = -(P - 0.5) * 1.1 + Math.sin(t * 1.6) * 0.012;
        const lx = px - (L / 2) * Math.cos(ang), ly2 = py - (L / 2) * Math.sin(ang), rx = px + (L / 2) * Math.cos(ang), ry = py + (L / 2) * Math.sin(ang);
        g.beginPath(); g.moveTo(px, py); g.lineTo(px - 22, py + B.h * 0.28); g.lineTo(px + 22, py + B.h * 0.28); g.closePath(); g.fillStyle = V.alpha(C.muted, 0.35); g.fill();
        V.rr(g, px - 40, py + B.h * 0.28, 80, 6, 3); g.fillStyle = V.alpha(C.muted, 0.5); g.fill();
        V.line(g, lx, ly2, rx, ry, { color: C.ink, lw: 3.5 });
        V.dot(g, px, py, 5, C.d);
        [[lx, ly2, C.a, 'y_w'], [rx, ry, C.c, 'y_l']].forEach(([x, y, col, lab]) => {
          V.line(g, x, y, x - 14, y + 30, { color: V.alpha(C.ink, 0.5), lw: 1 }); V.line(g, x, y, x + 14, y + 30, { color: V.alpha(C.ink, 0.5), lw: 1 });
          g.beginPath(); g.moveTo(x - 22, y + 30); g.quadraticCurveTo(x, y + 44, x + 22, y + 30); g.closePath(); g.fillStyle = V.alpha(col, 0.5); g.fill(); g.strokeStyle = col; g.lineWidth = 1.5; g.stroke();
          sym(g, lab, x, y + 28, { size: 12, weight: 700, align: 'center', color: C.ink });
        });
        sym(g, 'P(y_w ≻ y_l) = σ(' + fmt(gap, 2) + ') = ' + fmt(P, 3), px, B.y + B.h * 0.36 + B.h * 0.36 + 22, { size: 12.5, weight: 700, align: 'center', color: C.ink, font: 'mono' });
        if (i >= 4) sym(g, 'P = σ(β·h_w − β·h_l): no r, no Z', px, B.y + 14, { size: 12, weight: 600, align: 'center', color: C.a });
        else if (t - S.flash < 1.6) V.text(g, 'f(x) changed… the balance did not', px, B.y + 14, { size: 12, weight: 600, align: 'center', color: C.d });
        else V.text(g, 'the balance only feels the difference', px, B.y + 14, { size: 11.5, align: 'center', color: C.muted });
      });
    })();

    /* ======================================================== FIG 5 — a rampa do DPO */
    (function () {
      const fig = root.querySelector('#fig-perda');
      if (!fig) return;
      adaptH(fig, 400, 440);
      const BET = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1];
      const S = { bi: 3, p: 1, u: 0, tRun: 0, hold: 0, trail: [], tt: 0 };
      const ctl = UI.controls(fig);
      UI.slider(ctl, { label: 'β', min: 0, max: BET.length - 1, step: 1, value: S.bi, fmt: (v) => fmtTick(BET[v]), onInput: (v) => { S.bi = v; upd(); } });
      const sP = UI.slider(ctl, { label: 'p (fraction preferring yw)', min: 0.5, max: 1, step: 0.01, value: 1, fmt: (v) => fmt(v, 2), onInput: (v) => { S.p = v; upd(); restart(); } });
      htmlLabel(sP, 'p (fraction preferring y<sub>w</sub>)');
      UI.button(ctl, '▶ release the ball', () => restart(), { primary: true });
      const rL = UI.readout(ctl, 'loss on the pair (Δh = 3)', '—'), rW = UI.readout(ctl, 'force p − σ(β·Δh)', '—'), rC = UI.readout(ctl, 'β·force', '—'), rS = UI.readout(ctl, 'Δh* = logit(p)/β', '—');
      nocase(ctl);
      function upd() {
        // leituras no par de brinquedo (Δh = 3) com o p escolhido; com p = 1, força = peso σ(−β·Δh)
        const b = BET[S.bi], u = 3 * b, p = S.p;
        const loss = -p * logsig(u) - (1 - p) * logsig(-u), force = p - sig(u);
        rL.set(fmt(loss, 4)); rW.set(fmt(force, 4)); rC.set(fmt(b * force, 5));
        if (S.p >= 0.9999) rS.set('+∞', 'bad'); else rS.set(fmt(Math.log(S.p / (1 - S.p)) / b, 1), 'ok');
      }
      function restart() { S.u = 0; S.tRun = 0; S.hold = 0; S.trail = []; }
      upd();
      const H5 = V.fig(fig, (g, w, h, t, dt, _s, HH) => {
        const C = V.C();
        const b = BET[S.bi], p = S.p;
        const narrow = w < 560;
        const A = { x: 46, y: 30, w: w - 46 - 46, h: h - 30 - (narrow ? 94 : 64) };
        const xr = [-3 / b, 7.5 / b];
        const L = (dh) => -p * logsig(b * dh) - (1 - p) * logsig(-b * dh);
        const Fo = (dh) => p - sig(b * dh);
        const ax = axes(g, A, xr, [0, 3.2], { xlabel: 'margin Δh (nats)', ylabel: 'loss', nx: 6, ny: 4 });
        const Y2 = (v) => A.y + A.h - ((v + 0.6) / 1.6) * A.h;
        [-0.5, 0, 0.5, 1].forEach((v) => V.text(g, fmtTick(v), A.x + A.w + 6, Y2(v) + 4, { size: 11, font: 'mono', color: C.d }));
        V.text(g, 'force', A.x + A.w + 6, A.y - 8, { size: 11.5, color: C.d });
        clipRect(g, A);
        V.line(g, A.x, Y2(0), A.x + A.w, Y2(0), { color: V.alpha(C.d, 0.4), dash: [2, 4], lw: 1 });
        curve(g, xr[0], xr[1], Fo, ax.X, Y2, { color: C.d, lw: 2, dash: [7, 5] });
        curve(g, xr[0], xr[1], L, ax.X, ax.Y, { color: C.a, lw: 2.8 });
        // par de brinquedo
        if (3 >= xr[0] && 3 <= xr[1]) {
          const x3 = ax.X(3);
          V.line(g, x3, A.y, x3, A.y + A.h, { color: V.alpha(C.e, 0.7), dash: [3, 3], lw: 1.2 });
          V.dot(g, x3, ax.Y(L(3)), 4.5, C.e);
        }
        // ótimo
        if (p < 0.9999) {
          const xs = Math.log(p / (1 - p)) / b;
          if (xs <= xr[1]) {
            const xx = ax.X(xs);
            V.line(g, xx, A.y, xx, A.y + A.h, { color: C.d, lw: 1.6 });
            g.beginPath(); g.moveTo(xx, A.y + 2); g.lineTo(xx + 16, A.y + 8); g.lineTo(xx, A.y + 14); g.closePath(); g.fillStyle = C.d; g.fill();
          }
        }
        g.restore();
        if (3 >= xr[0] && 3 <= xr[1]) V.text(g, 'toy pair', ax.X(3) + 5, A.y + A.h - 8, { size: 11, color: C.e });
        if (p < 0.9999) { const xs = Math.log(p / (1 - p)) / b; const xx = Math.min(ax.X(xs), A.x + A.w); V.text(g, 'Δh* = ' + fmt(xs, 1), clamp(xx + 4, A.x, A.x + A.w - 70), A.y + 30, { size: 11.5, color: C.d, weight: 700, font: 'mono' }); }
        else { V.text(g, 'Δh* → +∞', A.x + A.w - 4, A.y + 14, { size: 12, color: C.d, weight: 700, font: 'mono', align: 'right' }); V.arrow(g, A.x + A.w - 70, A.y + 22, A.x + A.w - 4, A.y + 22, { color: C.d, lw: 1.6, head: 7 }); }
        // legenda das curvas, abaixo do gráfico (fora do caminho das duas curvas)
        const lab = p >= 0.9999 ? 'loss = −log σ(β·Δh)' : 'mean loss = −p·log σ(β·Δh) − (1−p)·log σ(−β·Δh)';
        const flab = 'force ∝ p − σ(β·Δh)';
        const ly = A.y + A.h + 48, lx = narrow ? 8 : A.x - 30;
        V.line(g, lx, ly - 4, lx + 18, ly - 4, { color: C.a, lw: 2.8 });
        const lh = V.wrap(g, lab, lx + 24, ly, w - lx - 32, { size: 11.5, color: C.a, font: 'mono', weight: 600, lineH: 15 });
        g.font = V.font(11.5, 'mono', 600);
        const oneRow = !narrow && lh <= 15 && lx + 24 + g.measureText(lab).width + 28 + 24 + g.measureText(flab).width < w - 8;
        const fx = oneRow ? lx + 24 + g.measureText(lab).width + 28 : lx, fy = oneRow ? ly : ly + lh + 2;
        V.line(g, fx, fy - 4, fx + 18, fy - 4, { color: C.d, lw: 2, dash: [5, 4] });
        V.text(g, flab, fx + 24, fy, { size: 11.5, color: C.d, font: 'mono', weight: 600 });
        // bolinha
        if (!LA.paused) {
          S.tRun += dt;
          const force = p - sig(S.u);
          S.u += 2.6 * force * Math.min(dt, 0.05);
          S.tt += dt;
          if (S.tt > 0.18) { S.tt = 0; S.trail.push(S.u); if (S.trail.length > 40) S.trail.shift(); }
          if (p >= 0.9999) { if (S.tRun > 15) restart(); }
          else if (Math.abs(force) < 0.004) { S.hold += dt; if (S.hold > 2.5) restart(); }
        }
        const dhb = S.u / b;
        if (dhb >= xr[0] && dhb <= xr[1]) {
          S.trail.forEach((u, k) => { const d = u / b; if (d >= xr[0] && d <= xr[1]) V.dot(g, ax.X(d), ax.Y(L(d)), 2, V.alpha(C.b, 0.25 + (0.5 * k) / S.trail.length)); });
          const bx = ax.X(dhb), by = ax.Y(L(dhb));
          V.glow(g, bx, by - 8, 22, C.b, 0.8); V.dot(g, bx, by - 8, 8, C.b);
          const fz = (p - sig(S.u)) * 90;
          if (Math.abs(fz) > 3) V.arrow(g, bx, by - 24, bx + fz, by - 24, { color: C.d, lw: 2.2, head: 7 });
        }
        // mira do ponteiro
        const P = HH.pointer;
        if (P.inside && P.x > A.x && P.x < A.x + A.w && P.y > A.y && P.y < A.y + A.h) {
          const d = xr[0] + ((P.x - A.x) / A.w) * (xr[1] - xr[0]);
          V.line(g, P.x, A.y, P.x, A.y + A.h, { color: V.alpha(C.ink, 0.35), lw: 1 });
          const txt = ['Δh = ' + fmt(d, 1), 'loss = ' + fmt(L(d), 3), 'force = ' + fmt(Fo(d), 3)];
          const bx = P.x + 10 > A.x + A.w - 130 ? P.x - 138 : P.x + 10;
          V.box(g, bx, A.y + 34, 128, 58, { r: 6, fill: V.alpha(C.panel, 0.92), stroke: V.alpha(C.muted, 0.5), lw: 1 });
          txt.forEach((s, k) => V.text(g, s, bx + 8, A.y + 51 + k * 16, { size: 11.5, font: 'mono', color: C.ink }));
        }
      });
    })();

    /* ======================================================== FIG 6 — do token à perda */
    (function () {
      const fig = root.querySelector('#fig-tokens');
      if (!fig) return;
      adaptH(fig, 330, 440, 620);
      const TW = ['It’s', 'Paris', '.'], TL = ['It’s', 'Lyon', '.'];
      const TH_W = [-0.35, -0.90, -0.28], RF_W = [-0.40, -1.20, -0.30];
      const TH_L = [-0.35, -2.60, -0.55], RF_L = [-0.40, -2.10, -0.50];
      const sum = (a) => a.reduce((x, y) => x + y, 0);
      const S = { b: 0.1 };
      const steps = [
        { t: 'Four forwards', d: 'The policy (with gradient) and the reference (without gradient) read the prompt and each response. Each forward produces logits <code>[B, T, V]</code>; <code>log_softmax</code> and a <code>gather</code> on the observed token leave one log-prob per position.' },
        { t: 'Mask and sum', d: 'The prompt tokens have mask 0 and are left out. Summing over the response positions, each row becomes \\(\\log\\pi(y\\mid x)\\): four numbers per pair.' },
        { t: 'Log-ratios', d: 'Subtract the reference from the policy: \\(h_w=\\log\\pi_\\theta(y_w\\mid x)-\\log\\pi_{\\text{ref}}(y_w\\mid x)\\), and the same for \\(h_l\\). The \\(h\\) row shows each token\'s contribution.' },
        { t: 'Margin and loss', d: '\\(\\Delta h=h_w-h_l\\), the logit is \\(\\beta\\Delta h\\), the loss is \\(-\\log\\sigma(\\beta\\Delta h)\\) and the pair\'s weight in the gradient is \\(\\sigma(-\\beta\\Delta h)\\). Switch \\(\\beta\\) with the buttons.' },
        { t: 'The gradient flows back', d: 'The backward raises each token of \\(y_w\\) and lowers each token of \\(y_l\\), all with the same coefficient \\(\\beta\\sigma(-\\beta\\Delta h)\\). At the shared “It’s” (same context, same token) the two arrows cancel out: the signal starts at the first divergent token.' },
      ];
      let H6 = null;
      const st = UI.stepper(fig, steps, { autoplay: 7000, onStep: () => H6 && H6.redraw() });
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'β', options: [{ v: 0.1, label: '0.1' }, { v: 0.5, label: '0.5' }], value: 0.1, onChange: (v) => { S.b = v; } });
      nocase(ctl);
      H6 = V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const narrow = w < 620;
        const i = st.i, el = st.elapsed();
        const b = S.b;
        const x0 = 8;
        const mW = narrow ? w - 16 : Math.min(w * 0.6, 500);
        const LW = narrow ? 58 : 76, SW = narrow ? 58 : 70;
        const cw = Math.min(78, (mW - LW - SW - 14) / 3), ch = 24, gp = 4;
        // prompt
        const py = 6;
        chip(g, x0, py, mW, 26, '', { fill: V.alpha(C.muted, 0.12), stroke: V.alpha(C.muted, 0.4) });
        V.text(g, 'x = “What is the capital of France?”', x0 + 10, py + 17, { size: 12, color: C.muted, maxW: mW - (narrow ? 90 : 170) });
        V.text(g, i >= 1 ? (narrow ? 'mask 0' : 'mask 0 · out of the sum') : 'prompt', x0 + mW - 8, py + 17, { size: 11, color: i >= 1 ? C.d : C.muted, align: 'right', weight: 600 });
        if (i >= 1) {
          const k = i === 1 ? clamp(el / 0.6, 0, 1) : 1;
          g.font = V.font(12, 'sans', 400);
          const pw = Math.min(g.measureText('x = “What is the capital of France?”').width + 4, mW - (narrow ? 90 : 170));
          V.line(g, x0 + 8, py + 13, x0 + 8 + pw * k, py + 13, { color: V.alpha(C.d, 0.8), lw: 1.6 });
        }
        const groups = [
          { k: 'w', tok: TW, th: TH_W, rf: RF_W, col: C.a, y: py + 40 },
          { k: 'l', tok: TL, th: TH_L, rf: RF_L, col: C.c, y: py + 40 + 4 * (ch + gp) + 14 },
        ];
        const cx = (c) => x0 + LW + c * (cw + gp);
        const sx = x0 + LW + 3 * (cw + gp) + 6;
        const coef = b * sig(-b * (sum(TH_W) - sum(RF_W) - (sum(TH_L) - sum(RF_L))));
        groups.forEach((G, gi) => {
          // cabeçalho
          sym(g, 'y_' + G.k, x0 + 4, G.y + 17, { size: 13, weight: 700, color: G.col });
          G.tok.forEach((tk, c) => chip(g, cx(c), G.y, cw, ch, tk, { fill: V.alpha(G.col, c === 0 ? 0.08 : 0.18), stroke: c === 0 ? V.alpha(G.col, 0.5) : G.col, size: 12, weight: 700, font: 'sans' }));
          if (i >= 1) V.text(g, 'Σ', sx + SW / 2, G.y + 17, { size: 13, weight: 700, color: C.ink, align: 'center' });
          const rows = [{ lab: 'π_θ', v: G.th, col: C.b }, { lab: 'π_ref', v: G.rf, col: C.muted }];
          rows.forEach((R, ri) => {
            const y = G.y + (ri + 1) * (ch + gp);
            sym(g, R.lab, x0 + 4, y + 16, { size: 12, weight: 600, color: R.col === C.muted ? C.muted : C.b });
            R.v.forEach((v, c) => {
              const kk = (gi * 2 + ri) * 3 + c;
              const vis = i > 0 || el > 0.25 + kk * 0.13;
              const scanning = i === 0 && el > 0.25 + kk * 0.13 && el < 0.25 + (kk + 1) * 0.13;
              V.box(g, cx(c), y, cw, ch, { r: 4, fill: vis ? V.alpha(R.col, 0.12 + 0.45 * Math.min(1, -v / 2.6)) : V.alpha(C.muted, 0.05), stroke: scanning ? C.d : V.alpha(R.col, 0.35), lw: scanning ? 2 : 1 });
              if (vis) V.text(g, fmt(v, 2), cx(c) + cw / 2 - (i === 4 && ri === 0 ? 6 : 0), y + 16, { size: 12, font: 'mono', color: C.ink, align: 'center' });
              // gradiente (passo 5) nas linhas da política
              if (i === 4 && ri === 0) {
                const up = G.k === 'w';
                const ax2 = cx(c) + cw - 9, cy = y + ch / 2;
                const al = c === 0 ? clamp(1 - (el - 0.6) / 0.8, 0, 1) : 0.6 + 0.4 * Math.sin(t * 5);
                if (al > 0.02) {
                  g.globalAlpha = al;
                  const L2 = 9 + Math.min(1, coef / 0.1) * 2;
                  V.glow(g, ax2, cy, 12, C.d, 0.5);
                  V.arrow(g, ax2, cy + (up ? L2 : -L2), ax2, cy + (up ? -L2 : L2), { color: C.d, lw: 3.2, head: 8 });
                  g.globalAlpha = 1;
                }
                if (c === 0 && el > 1.2) V.text(g, '×', ax2, cy + 5, { size: 13, color: C.d, align: 'center', weight: 700 });
              }
            });
            if (i >= 1) {
              const s = sum(R.v), k = i === 1 ? clamp((el - 0.3 - ri * 0.3 - gi * 0.6) / 0.4, 0, 1) : 1;
              g.globalAlpha = k;
              V.box(g, sx, y, SW, ch, { r: 5, fill: V.alpha(R.col, 0.22), stroke: R.col === C.muted ? C.muted : C.b, lw: 1.6 });
              V.text(g, fmt(s, 2), sx + SW / 2, y + 16, { size: 12, font: 'mono', color: C.ink, align: 'center', weight: 700 });
              g.globalAlpha = 1;
            }
          });
          // linha h
          if (i >= 2) {
            const y = G.y + 3 * (ch + gp);
            const k = i === 2 ? clamp((el - 0.2 - gi * 0.5) / 0.5, 0, 1) : 1;
            g.globalAlpha = k;
            V.text(g, 'h', x0 + 8, y + 16, { size: 12.5, weight: 700, color: C.ink, font: 'mono' });
            G.th.forEach((v, c) => {
              const d = v - G.rf[c];
              V.box(g, cx(c), y, cw, ch, { r: 4, fill: V.alpha(d >= 0 ? C.a : C.c, 0.15 + Math.min(0.6, Math.abs(d))), stroke: V.alpha(d >= 0 ? C.a : C.c, 0.6), lw: 1 });
              V.text(g, fmtS(d, 2), cx(c) + cw / 2, y + 16, { size: 12, font: 'mono', color: C.ink, align: 'center' });
            });
            const hs = sum(G.th) - sum(G.rf);
            V.box(g, sx, y, SW, ch, { r: 5, fill: V.alpha(G.col, 0.3), stroke: G.col, lw: 2 });
            sym(g, fmtS(hs, 2), sx + SW / 2, y + 16, { size: 12, font: 'mono', color: C.ink, align: 'center', weight: 700 });
            g.globalAlpha = 1;
          }
        });
        // ligação de cancelamento (passo 5)
        if (i === 4) {
          const y1 = groups[0].y + (ch + gp) + ch / 2, y2 = groups[1].y + (ch + gp) + ch / 2;
          const xx = cx(0) - 3;
          const k = clamp((el - 0.2) / 0.6, 0, 1);
          g.setLineDash([4, 3]);
          V.line(g, xx, y1, xx - 7 * k, y1, { color: C.d, lw: 1.6 }); V.line(g, xx - 7 * k, y1, xx - 7 * k, y2, { color: C.d, lw: 1.6 }); V.line(g, xx - 7 * k, y2, xx, y2, { color: C.d, lw: 1.6 });
          g.setLineDash([]);
          if (k >= 1) V.text(g, '↕ “It’s” and “It’s”: cancel out', cx(0), groups[1].y - 5, { size: 11, color: C.d, weight: 700 });
        }
        // painel de contas
        const hw = sum(TH_W) - sum(RF_W), hl = sum(TH_L) - sum(RF_L), dh = hw - hl, u = b * dh;
        const lines = [];
        if (i === 0) lines.push(['4 forwards per pair', C.ink, 700], ['π_θ(y_w), π_θ(y_l): with gradient', C.b, 500], ['π_ref(y_w), π_ref(y_l): no gradient', C.muted, 500], ['logits [B, T, V] → log_softmax', C.ink, 400], ['→ gather the token → [B, T]', C.ink, 400]);
        if (i === 1) lines.push(['log π_θ(y_w|x) = ' + fmt(sum(TH_W), 2), C.b, 500], ['log π_ref(y_w|x) = ' + fmt(sum(RF_W), 2), C.muted, 500], ['log π_θ(y_l|x) = ' + fmt(sum(TH_L), 2), C.b, 500], ['log π_ref(y_l|x) = ' + fmt(sum(RF_L), 2), C.muted, 500]);
        if (i === 2) lines.push(['h_w = ' + fmt(sum(TH_W), 2) + ' − (' + fmt(sum(RF_W), 2) + ') = ' + fmtS(hw, 2), C.a, 600], ['h_l = ' + fmt(sum(TH_L), 2) + ' − (' + fmt(sum(RF_L), 2) + ') = ' + fmtS(hl, 2), C.c, 600], ['the policy already likes y_w more', C.muted, 400], ['and y_l less than the reference does', C.muted, 400]);
        if (i === 3) lines.push(['Δh = ' + fmtS(hw, 2) + ' − (' + fmtS(hl, 2) + ') = ' + fmt(dh, 2), C.d, 700], ['β·Δh = ' + fmt(u, 3), C.ink, 500], ['loss = −log σ(' + fmt(u, 3) + ') = ' + fmt(-logsig(u), 3), C.a, 700], ['weight σ(−β·Δh) = ' + fmt(sig(-u), 3), C.ink, 500], ['coef. β·weight = ' + fmt(b * sig(-u), 3), C.ink, 500]);
        if (i === 4) lines.push(['∇L = −coef·(∇log π_θ(y_w|x) − ∇log π_θ(y_l|x))', C.ink, 600], ['coef = ' + fmt(b * sig(-u), 3) + ' (β = ' + fmtTick(b) + ')', C.d, 700], ['“It’s”: same context, same token → 0', C.d, 500], ['Paris ↑ · Lyon ↓ · “.” ↑ and ↓', C.ink, 500], ['(the two “.” have different contexts)', C.muted, 400]);
        let px, pyy, pw;
        if (narrow) { px = x0; pyy = groups[1].y + 4 * (ch + gp) + 14; pw = w - 16; }
        else { px = x0 + mW + 18; pyy = groups[0].y; pw = w - px - 8; }
        V.box(g, px, pyy, pw, lines.length * 21 + 18, { r: 10, fill: V.alpha(C.panel, 0.8), stroke: V.alpha(C.muted, 0.3), lw: 1 });
        lines.forEach((L, k) => sym(g, L[0], px + 12, pyy + 22 + k * 21, { size: narrow ? 11.5 : 12, weight: L[2], color: L[1], font: 'mono' }));
      });
    })();

    /* ======================================================== FIG 7 — memória e referência */
    (function () {
      const fig = root.querySelector('#fig-memoria');
      if (!fig) return;
      adaptH(fig, 370, 455);
      const GiB = Math.pow(2, 30);
      const LORA_FRAC = 217.6e6 / 27e9;
      const S = { n: 8, mode: 'ft', tr: false };
      const ctl = UI.controls(fig);
      UI.select(ctl, { label: 'Model', options: [{ v: '1.5', label: '1.5B' }, { v: '4', label: '4B' }, { v: '8', label: '8B' }, { v: '27', label: '27B' }, { v: '70', label: '70B' }], value: '8', onChange: (v) => { S.n = +v; upd(); } });
      UI.seg(ctl, { label: 'Reference', options: [{ v: 'ft', label: 'FT + copy' }, { v: 'pre', label: 'FT + precomputed' }, { v: 'lora', label: 'LoRA, adapter off' }], value: 'ft', onChange: (v) => { S.mode = v; upd(); } });
      UI.toggle(ctl, { label: 'moving reference (TR-DPO)', value: false, onChange: (v) => { S.tr = v; upd(); } });
      const rT = UI.readout(ctl, 'weights + states', '—'), rF = UI.readout(ctl, 'fits in (practical ceiling)', '—'), rV = UI.readout(ctl, 'verdict', '—');
      /* tetos práticos: o MLX recomenda 19,07 GB de memória de trabalho no Mac M4 de 24 GB;
         ~105 GiB por Spark (faixa de 100–110 GiB do cap. 51; na receita GLM, 0,88 × 121,69 = 107,09 GiB
         bootou só 1 de 4 vezes, cap. 42); dois Sparks com ZeRO-3/FSDP dividem os estados */
      const SPARK = 105;
      const CAPS = [
        { id: 'mac', v: 19.07e9 / GiB, l: 'Mac (~19 GB)', s: 'Mac' },
        { id: 'spark', v: SPARK, l: '1 Spark', s: '1 Spark' },
        { id: 'spark2', v: 2 * SPARK, l: '2 Sparks (ZeRO-3/FSDP)', ln: '2 Sparks', s: '2 Sparks' },
      ];
      const SLACK = 0.85; // acima de 85% do teto: só os estados cabem, sem folga para ativações e logits
      // no Mac, o mlx-lm-lora carrega a referência como uma 2ª cópia completa (o PEFT do TRL não precisa)
      const macExtra = () => (S.mode === 'lora' && !S.tr ? (2 * S.n * 1e9) / GiB : 0);
      const need = (cp, tot) => tot + (cp.id === 'mac' ? macExtra() : 0);
      const fitState = (n, cap) => (n <= SLACK * cap ? 'ok' : n <= cap ? 'warn' : 'bad');
      const gb = (gib) => gib * GiB / 1e9;
      const fmtG = (x) => fmt(x, x >= 100 ? 0 : 1);
      function comps() {
        const N = S.n * 1e9, C = V.C();
        if (S.mode === 'lora') {
          const out = [{ k: 'base BF16 (also the reference)', v: 2 * N, c: C.b }, { k: 'LoRA + grad + Adam (r = 32, est.)', v: 16 * LORA_FRAC * N, c: C.d }];
          if (S.tr) out.push({ k: 'separate ref_model (TR-DPO)', v: 2 * N, c: C.muted });
          return out;
        }
        const out = [{ k: 'BF16 weights', v: 2 * N, c: C.b }, { k: 'BF16 gradients', v: 2 * N, c: C.c }, { k: 'Adam + FP32 master copy', v: 12 * N, c: C.d }];
        if (S.mode === 'ft' || S.tr) out.push({ k: S.mode === 'pre' ? 'reference (TR-DPO prevents precomputing)' : 'BF16 reference', v: 2 * N, c: C.muted });
        return out;
      }
      function upd() {
        const tot = comps().reduce((a, c) => a + c.v, 0) / GiB;
        rT.set(fmtG(tot) + ' GiB (≈ ' + fmtG(gb(tot)) + ' GB)');
        const st = CAPS.map((cp) => fitState(need(cp, tot), cp.v));
        rF.set(CAPS.map((cp, k) => cp.s + ' ' + (st[k] === 'ok' ? '✓' : st[k] === 'warn' ? '⚠' : '✗')).join(' · '), st.includes('ok') ? 'ok' : st.includes('warn') ? 'warn' : 'bad');
        const k0 = st.findIndex((x) => x !== 'bad');
        if (k0 < 0) rV.set('does not fit even in 2 Sparks', 'bad');
        else if (st[k0] === 'warn') rV.set(CAPS[k0].s + ': states only — no room for activations and logits', 'warn');
        else rV.set(CAPS[k0].s + ': room left for activations and logits', 'ok');
      }
      upd();
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const narrow = w < 600;
        const cs = comps();
        const tot = cs.reduce((a, c) => a + c.v, 0) / GiB, ext = macExtra();
        const mx = Math.max((tot + ext) * 1.08, 240);
        const A = { x: 12, y: 70, w: w - 24, h: 34 };
        const X = (v) => A.x + (v / mx) * A.w;
        V.text(g, narrow ? 'weights and states (GiB) × practical ceiling' : 'memory of weights and states (GiB, no activations or logits) against the practical ceiling', A.x, 16, { size: 11.5, weight: 600, color: C.muted, maxW: w - 24 });
        // eixo
        ticks(0, mx, narrow ? 4 : 6).forEach((v) => { V.line(g, X(v), A.y + A.h + 2, X(v), A.y + A.h + 6, { color: C.muted, lw: 1 }); V.text(g, fmtTick(v), X(v), A.y + A.h + 18, { size: 11, font: 'mono', color: C.muted, align: 'center' }); });
        V.rr(g, A.x, A.y, A.w, A.h, 6); g.fillStyle = V.alpha(C.muted, 0.08); g.fill();
        let acc = 0;
        cs.forEach((c) => {
          const x1 = X(acc / GiB), x2 = X((acc + c.v) / GiB);
          V.rr(g, x1, A.y, Math.max(1.5, x2 - x1 - 1), A.h, 4); g.fillStyle = V.alpha(c.c, 0.7); g.fill();
          acc += c.v;
        });
        // 2ª cópia que só o Mac (mlx-lm-lora) carrega: contorno tracejado
        if (ext > 0) {
          V.rr(g, X(tot), A.y + 3, Math.max(2, X(tot + ext) - X(tot) - 1), A.h - 6, 4);
          g.fillStyle = V.alpha(C.muted, 0.08); g.fill();
          g.setLineDash([4, 3]); g.strokeStyle = V.alpha(C.ink, 0.6); g.lineWidth = 1.3; g.stroke(); g.setLineDash([]);
        }
        // tetos práticos (verde: folga; âmbar: só estados; vermelho: não cabe)
        const rowR = [-1e9, -1e9, -1e9];
        CAPS.forEach((cp) => {
          if (cp.v > mx) return;
          const x = X(cp.v), stt = fitState(need(cp, tot), cp.v);
          const col = stt === 'ok' ? C.e : stt === 'warn' ? C.d : RED();
          const lab = narrow && cp.ln ? cp.ln : cp.l;
          g.font = V.font(11, 'sans', 600);
          const tw = g.measureText(lab).width;
          const lx = clamp(x - tw / 2, A.x, w - 4 - tw);
          let row = rowR.findIndex((r) => lx > r + 6); if (row < 0) row = 2;
          rowR[row] = lx + tw;
          const ly0 = A.y - 12 - row * 14;
          V.line(g, x, ly0 + 4, x, A.y + A.h + 4, { color: col, lw: 1.8, dash: [4, 3] });
          V.text(g, lab, lx, ly0, { size: 11, color: col, weight: 600 });
        });
        const tl = fmtG(tot) + ' GiB' + (ext > 0 ? (narrow ? '' : ' (' + fmtG(gb(tot)) + ' GB)') + ' · on the Mac ' + fmtG(tot + ext) + ' GiB' : ' (' + fmtG(gb(tot)) + ' GB)');
        g.font = V.font(12, 'mono', 700);
        const tlw = g.measureText(tl).width, inside = X(tot + ext) + 6 + tlw >= w - 6;
        const tx0 = inside ? Math.max(A.x + 6, Math.min(X(tot), w - 8) - 6 - tlw) : X(tot + ext) + 6;
        // fundo atrás do rótulo: as linhas dos tetos passam por baixo sem cortar o texto
        V.rr(g, tx0 - 4, A.y + A.h / 2 - 9, tlw + 8, 18, 5); g.fillStyle = V.alpha(C.bg, 0.82); g.fill();
        V.text(g, tl, tx0, A.y + A.h / 2 + 4, { size: 12, weight: 700, font: 'mono', color: C.ink });
        // legenda
        let ly = A.y + A.h + 36;
        const colsN = narrow ? 1 : 2, colW = (w - 24) / colsN;
        const legItems = cs.slice();
        if (ext > 0) legItems.push({ k: '2nd copy, Mac only (mlx-lm-lora)', v: ext * GiB, ghost: true });
        legItems.forEach((c, k) => {
          const lx = A.x + (k % colsN) * colW, yy = ly + Math.floor(k / colsN) * 20;
          if (c.ghost) { V.rr(g, lx + 0.5, yy - 9.5, 11, 11, 3); g.setLineDash([3, 2]); g.strokeStyle = V.alpha(C.ink, 0.7); g.lineWidth = 1.2; g.stroke(); g.setLineDash([]); }
          else V.box(g, lx, yy - 10, 12, 12, { r: 3, fill: V.alpha(c.c, 0.75) });
          V.text(g, c.k + ': ' + fmt(c.v / GiB, c.v / GiB >= 100 ? 0 : 1) + ' GiB', lx + 18, yy, { size: 11.5, color: C.ink, maxW: colW - 24 });
        });
        ly += Math.ceil(legItems.length / colsN) * 20 + 12;
        // faixas dos forwards
        V.text(g, narrow ? '4 forwards per step (lilac = with backward)' : 'one DPO step: four forwards (lilac = with backward)', A.x, ly, { size: 11.5, weight: 600, color: C.muted });
        const lanes = [
          { l: 'π_θ · y_w', pol: true }, { l: 'π_θ · y_l', pol: true },
          { l: 'π_ref · y_w', pol: false }, { l: 'π_ref · y_l', pol: false },
        ];
        const lt = ly + 12, lh = Math.min(30, (h - lt - 30) / 4);
        const lx0 = A.x + (narrow ? 74 : 90), lx1 = w - 14 - (narrow ? 0 : 230);
        const cyc = 2.6, f = (t % cyc) / cyc;
        const syncPulse = S.tr && Math.floor(t / cyc) % 3 === 2;
        lanes.forEach((L, k) => {
          const y = lt + k * lh + lh / 2;
          sym(g, L.l, A.x, y + 4, { size: 12, weight: 600, color: L.pol ? C.b : C.muted, font: 'mono' });
          V.rr(g, lx0, y - 7, lx1 - lx0, 14, 7); g.fillStyle = V.alpha(L.pol ? C.b : C.muted, 0.1); g.fill();
          const pre = !L.pol && S.mode === 'pre' && !S.tr;
          if (pre) {
            const blink = f < 0.15 ? 1 : 0.35;
            chip(g, lx0 + 4, y - 9, 92, 18, 'cache 💾', { size: 11, fill: V.alpha(C.e, 0.2 * blink + 0.05), stroke: V.alpha(C.e, blink), font: 'sans' });
            V.text(g, 'read from disk: computed 1×', lx0 + 104, y + 4, { size: 11, color: C.muted, maxW: lx1 - lx0 - 110 });
          } else {
            const fx = f < 0.5 ? f / 0.5 : 1;
            if (f < 0.5) { const x = lx0 + (lx1 - lx0) * V.ease.inOut(fx); V.glow(g, x, y, 12, L.pol ? C.b : C.muted, 0.9); V.dot(g, x, y, 4, L.pol ? C.b : C.muted); }
            if (L.pol && f >= 0.5) { const x = lx1 - (lx1 - lx0) * V.ease.inOut((f - 0.5) / 0.5); V.glow(g, x, y, 12, C.d, 0.9); V.dot(g, x, y, 4, C.d); }
            if (S.mode === 'lora' && !S.tr) {
              const on = L.pol;
              chip(g, lx1 - 92, y - 9, 88, 18, on ? 'adapter ON' : 'adapter OFF', { size: 11, font: 'sans', fill: V.alpha(on ? C.a : C.muted, 0.18), stroke: on ? C.a : C.muted });
            }
          }
        });
        // notas à direita (largo) ou embaixo (estreito)
        const notes = [];
        if (S.mode === 'ft') notes.push(['frozen copy: +2 B/parameter'], ['2 extra forwards per pair']);
        if (S.mode === 'pre') notes.push(S.tr ? ['TR-DPO changes the reference: it cannot be precomputed'] : ['precompute_ref_log_probs=True', 'mono'], S.tr ? ['the copy goes back into memory'] : ['training: only 2 forwards per pair']);
        if (S.mode === 'lora') notes.push(S.tr ? ['sync_ref_model with PEFT requires a separate ref_model (+2 B/parameter)'] : ['ref_model=None + PEFT', 'mono'], S.tr ? null : ['reference = same model, adapter turned off'], S.tr ? null : ['on the Mac, mlx-lm-lora loads a 2nd copy (+2 B/parameter): counted on the Mac line']);
        if (S.tr) notes.push(['every 512 steps: θ_ref ← 0.6·θ + 0.4·θ_ref', 'sym']);
        const nx = narrow ? A.x : lx1 + 16;
        let ny = narrow ? lt + 4 * lh + 16 : lt + 10;
        notes.filter(Boolean).forEach((n) => {
          if (n[1] === 'sym') { g.font = V.font(11.5); const tooWide = symW(g, n[0], 11.5) > w - nx - 8; if (tooWide) { sym(g, 'every 512 steps:', nx, ny, { size: 11.5, color: C.d }); ny += 17; sym(g, 'θ_ref ← 0.6·θ + 0.4·θ_ref (weights)', nx, ny, { size: 11.5, color: C.d }); } else sym(g, n[0], nx, ny, { size: 11.5, color: C.d }); ny += 19; }
          else ny += V.wrap(g, n[0], nx, ny, w - nx - 8, { size: n[1] === 'mono' ? 11 : 11.5, font: n[1] === 'mono' ? 'mono' : 'sans', color: C.ink, lineH: 17 }) + 2;
        });
        if (syncPulse && !narrow) {
          const y1 = lt + lh / 2, y2 = lt + 2.5 * lh;
          V.arrow(g, lx0 - 8, y1 + 6, lx0 - 8, y2 + lh, { color: C.d, lw: 2, head: 7, curve: -18 });
        }
      });
    })();

    /* ======================================================== FIG 8 — offline × iterativo × online */
    (function () {
      const fig = root.querySelector('#fig-online');
      if (!fig) return;
      adaptH(fig, 420, 480);
      const ZS = [0.76, 0.72], SR = 0.17;
      const Rf = (x, y) => Math.exp(-((x - ZS[0]) ** 2 + (y - ZS[1]) ** 2) / (2 * SR * SR));
      const SD = 0.07, SP = 0.06;
      const D0 = [0.27, 0.3], CREF = [0.28, 0.32];
      const rng = V.rng(57);
      const OFF = Array.from({ length: 70 }, () => [rng.normal(), rng.normal()]);
      const S = { mode: 'offline', c: CREF.slice(), d: D0.slice(), ds: D0.slice(), t: 0, rt: 0, round: 1, pairs: [], seen: 0, g: [0, 0], spawn: 0, judge: -10, key: '', cache: null };
      function dirFrom(c, sd, n = 400) {
        let gx = 0, gy = 0;
        for (let k = 0; k < n; k++) {
          const a = [c[0] + sd * rng.normal(), c[1] + sd * rng.normal()], b = [c[0] + sd * rng.normal(), c[1] + sd * rng.normal()];
          const s = Rf(a[0], a[1]) >= Rf(b[0], b[1]) ? 1 : -1;
          gx += s * (a[0] - b[0]); gy += s * (a[1] - b[1]);
        }
        return [gx / n, gy / n];
      }
      function reset() { S.c = CREF.slice(); S.d = D0.slice(); S.ds = D0.slice(); S.t = 0; S.rt = 0; S.round = 1; S.pairs = []; S.seen = 0; S.g = dirFrom(S.d, SD); }
      reset();
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Pairs', options: [{ v: 'offline', label: 'offline' }, { v: 'iter', label: 'iterative' }, { v: 'online', label: 'online' }], value: 'offline', onChange: (v) => { S.mode = v; reset(); } });
      UI.button(ctl, '↺ restart', () => reset());
      const rO = UI.readout(ctl, 'data × πθ overlap', '—'), rR = UI.readout(ctl, 'true reward of πθ', '—'), rK = UI.readout(ctl, 'KL to π ref', '—');
      htmlLabel(rO, 'data × π<sub>θ</sub> overlap'); htmlLabel(rR, 'true reward of π<sub>θ</sub>'); htmlLabel(rK, 'KL to π<sub>ref</sub>');
      nocase(ctl);
      function bc(m1, s1, m2, s2) { const d2 = (m1[0] - m2[0]) ** 2 + (m1[1] - m2[1]) ** 2; const ss = s1 * s1 + s2 * s2; return ((2 * s1 * s2) / ss) * Math.exp(-d2 / (4 * ss)); }
      function sample(c, sd) { return [c[0] + sd * rng.normal(), c[1] + sd * rng.normal()]; }
      function addPair(a, b, life = 1.1) { const wa = Rf(a[0], a[1]) >= Rf(b[0], b[1]); S.pairs.push({ w: wa ? a : b, l: wa ? b : a, t: 0, life }); if (S.pairs.length > 40) S.pairs.shift(); S.seen++; }
      function step(dt) {
        S.t += dt;
        if (S.mode === 'offline') {
          const k = 1.9 / (1 + 0.12 * S.t);
          S.c[0] += S.g[0] * k * dt; S.c[1] += S.g[1] * k * dt;
          S.spawn += dt; if (S.spawn > 0.22) { S.spawn = 0; addPair(sample(S.d, SD), sample(S.d, SD)); }
          if (S.c[0] >= 0.975 || S.c[1] >= 0.975) S.edge = (S.edge || 0) + dt;
          if (S.t > 17 || S.edge > 1.8) { S.edge = 0; reset(); }
        } else if (S.mode === 'iter') {
          S.rt += dt;
          if (S.rt > 3.2) { S.rt = 0; S.round++; S.d = S.c.slice(); S.g = dirFrom(S.d, SD); }
          const k = 1.6 / (1 + 0.4 * S.rt);
          S.c[0] += S.g[0] * k * dt; S.c[1] += S.g[1] * k * dt;
          S.spawn += dt; if (S.spawn > 0.22) { S.spawn = 0; addPair(sample(S.d, SD), sample(S.d, SD)); }
          if (S.round > 7) reset();
        } else {
          S.spawn += dt;
          while (S.spawn > 0.08) {
            S.spawn -= 0.08;
            const a = sample(S.c, SP), b = sample(S.c, SP);
            addPair(a, b, 0.7);
            const p = S.pairs[S.pairs.length - 1];
            S.c[0] += 0.16 * (p.w[0] - p.l[0]) - 0.012 * (S.c[0] - CREF[0]);
            S.c[1] += 0.16 * (p.w[1] - p.l[1]) - 0.012 * (S.c[1] - CREF[1]);
            S.judge = S.t;
          }
          if (S.t > 15) reset();
        }
        S.c[0] = clamp(S.c[0], 0.02, 0.98); S.c[1] = clamp(S.c[1], 0.02, 0.98);
        S.ds[0] += (S.d[0] - S.ds[0]) * Math.min(1, dt * 4); S.ds[1] += (S.d[1] - S.ds[1]) * Math.min(1, dt * 4);
        S.pairs.forEach((p) => { p.t += dt; });
        S.pairs = S.pairs.filter((p) => p.t < p.life);
      }
      V.fig(fig, (g, w, h, t, dt) => {
        const C = V.C();
        if (!LA.paused) step(Math.min(dt, 0.05));
        const A = { x: 10, y: 30, w: w - 20, h: h - 40 };
        const X = (v) => A.x + v * A.w, Y = (v) => A.y + (1 - v) * A.h;
        const key = [Math.round(w), Math.round(h), C.dark].join('|');
        if (S.key !== key) {
          const dpr = Math.min(window.devicePixelRatio || 1, 2);
          const cv = document.createElement('canvas'); cv.width = Math.max(1, Math.round(w * dpr)); cv.height = Math.max(1, Math.round(h * dpr));
          const q = cv.getContext('2d'); q.setTransform(dpr, 0, 0, dpr, 0, 0);
          const NX = 120, NY = 80;
          for (let i = 0; i < NX; i++) for (let j = 0; j < NY; j++) {
            const vx = (i + 0.5) / NX, vy = (j + 0.5) / NY;
            q.fillStyle = V.mix(C.bg, C.a, 0.03 + 0.42 * Rf(vx, vy));
            q.fillRect(A.x + (i * A.w) / NX, A.y + A.h - ((j + 1) * A.h) / NY, A.w / NX + 0.6, A.h / NY + 0.6);
          }
          S.cache = cv; S.key = key;
        }
        g.drawImage(S.cache, 0, 0, w, h);
        V.rr(g, A.x, A.y, A.w, A.h, 8); g.strokeStyle = V.alpha(C.muted, 0.35); g.lineWidth = 1; g.stroke();
        V.text(g, '★', X(ZS[0]), Y(ZS[1]) + 5, { size: 15, color: C.a, align: 'center' });
        // rótulo da estrela: embaixo dela, e só enquanto π_θ (e o seu chip) estiverem longe
        const starLx = clamp(X(ZS[0]), A.x + 104, A.x + A.w - 104), starLy = Y(ZS[1]) + 24;
        const desc = S.mode === 'offline' ? 'offline: fixed pairs, generated by another model (ρ)' : S.mode === 'iter' ? 'iterative · round ' + S.round + ': generate → label → DPO' : 'online: 2 samples from π_θ → judge → DPO step';
        sym(g, desc, A.x, 18, { size: 12, weight: 600, color: C.ink });
        // nuvem de dados
        OFF.forEach((o) => V.dot(g, X(S.ds[0] + SD * o[0]), Y(S.ds[1] + SD * o[1]), 2.2, V.alpha(C.ink, 0.45)));
        V.text(g, S.mode === 'online' ? '' : 'dataset', X(S.ds[0]) - 30, Y(S.ds[1] - 2.6 * SD) + 4, { size: 11, color: C.muted });
        // pares
        S.pairs.forEach((p) => {
          const al = 1 - p.t / p.life;
          V.line(g, X(p.l[0]), Y(p.l[1]), X(p.w[0]), Y(p.w[1]), { color: V.alpha(C.ink, 0.35 * al), lw: 1 });
          V.dot(g, X(p.w[0]), Y(p.w[1]), 3, V.alpha(C.a, al)); V.dot(g, X(p.l[0]), Y(p.l[1]), 3, V.alpha(C.c, al));
        });
        // coleira KL
        const cx = X(S.c[0]), cy = Y(S.c[1]), rx = X(CREF[0]), ry = Y(CREF[1]);
        const dist = Math.hypot(S.c[0] - CREF[0], S.c[1] - CREF[1]);
        g.beginPath(); g.moveTo(rx, ry);
        const segs = 16;
        for (let k = 1; k <= segs; k++) { const u = k / segs; const wob = Math.sin(u * Math.PI) * Math.sin(t * 3 + u * 8) * Math.max(0, 6 - dist * 20); g.lineTo(rx + (cx - rx) * u - wob * (cy - ry) / (Math.hypot(cx - rx, cy - ry) || 1), ry + (cy - ry) * u + wob * (cx - rx) / (Math.hypot(cx - rx, cy - ry) || 1)); }
        g.strokeStyle = V.alpha(C.d, 0.8); g.lineWidth = 1.8; g.setLineDash([5, 4]); g.stroke(); g.setLineDash([]);
        g.beginPath(); g.arc(rx, ry, 7, 0, Math.PI * 2); g.strokeStyle = C.ink; g.lineWidth = 2; g.stroke();
        sym(g, 'π_ref', rx - 10, ry + 22, { size: 11.5, color: C.ink, weight: 600, align: 'right' });
        // blob da política
        V.glow(g, cx, cy, Math.max(26, SP * A.w * 1.6), C.b, 0.9);
        V.dot(g, cx, cy, 6, C.b);
        sym(g, 'π_θ', cx + 12, cy - 10, { size: 13, weight: 700, color: C.b });
        if (S.mode === 'online' && S.t - S.judge < 0.12) chip(g, cx + 14, cy + 4, 44, 18, 'judge', { size: 11, font: 'sans', weight: 700, fill: V.alpha(C.d, 0.3), stroke: C.d });
        const ov = bc(S.ds, SD, S.c, SP);
        const offSup = S.mode !== 'online' && ov < 0.12;
        let chipX = clamp(cx - 80, A.x + 4, A.x + A.w - 176), chipY = clamp(cy + 22, A.y + 4, A.y + A.h - 24);
        // a política perto da estrela: some o rótulo (a estrela já marca o máximo)
        const near = (px, py, rw, rh) => Math.hypot(Math.max(0, Math.abs(px - starLx) - rw), Math.max(0, Math.abs(py - starLy) - rh));
        const showStarLab = near(cx, cy, 100, 8) > 40 && near(X(S.ds[0]), Y(S.ds[1]), 100, 8) > SD * A.w * 1.2;
        if (offSup && showStarLab && Math.abs(chipY + 10 - (starLy - 4)) < 24 && Math.abs(chipX + 86 - starLx) < 190) {
          // chip e rótulo na mesma faixa: o chip sobe para cima de π_θ
          chipY = clamp(cy - 54, A.y + 4, A.y + A.h - 24);
          if (Math.abs(chipY + 10 - (starLy - 4)) < 24) chipY = clamp(starLy + 14, A.y + 4, A.y + A.h - 24);
        }
        if (showStarLab) {
          g.font = V.font(11.5, 'sans', 600);
          const sw = g.measureText('maximum true reward').width;
          V.rr(g, starLx - sw / 2 - 5, starLy - 13, sw + 10, 18, 5); g.fillStyle = V.alpha(C.bg, 0.75); g.fill();
          V.text(g, 'maximum true reward', starLx, starLy, { size: 11.5, color: C.a, align: 'center', weight: 600 });
        }
        if (offSup) chip(g, chipX, chipY, 172, 20, 'outside the data support', { size: 11, font: 'sans', weight: 700, fill: V.alpha(RED(), 0.15), stroke: RED() });
        if (S.mode === 'online') rO.set('on-policy', 'ok'); else rO.set(Math.round(ov * 100) + '%', ov < 0.12 ? 'bad' : ov < 0.5 ? 'warn' : 'ok');
        const rv = Rf(S.c[0], S.c[1]);
        rR.set(Math.round(rv * 100) + '%', rv > 0.8 ? 'ok' : rv > 0.4 ? 'warn' : null);
        rK.set(fmt((dist * dist) / (2 * 0.15 * 0.15), 1) + ' nats (illustr.)');
      });
    })();

    /* ======================================================== FIG 9 — deslocamento de verossimilhança */
    (function () {
      const fig = root.querySelector('#fig-deslocamento');
      if (!fig) return;
      adaptH(fig, 440, 720);
      const W = [[1, 0.2], [1, -0.2], [0, 1]], NM = ['No', 'Never', 'Yes'];
      const S = { s: 0, play: true, hold: 0 };
      const probs = (s) => V.softmax([2 + 0.2 * s, 2 - 0.2 * s, s]);
      const CURV = Array.from({ length: 101 }, (_, k) => probs(k / 10));
      const ctl = UI.controls(fig);
      const sl = UI.slider(ctl, { label: 's (DPO steps on the hidden state)', min: 0, max: 10, step: 0.1, value: 0, fmt: (v) => fmt(v, 1), onInput: (v) => { S.s = v; S.play = false; upd(); } });
      UI.button(ctl, '▶ train', () => { S.play = true; S.s = 0; S.hold = 0; }, { primary: true });
      const rP = [UI.readout(ctl, 'p(No)', '—'), UI.readout(ctl, 'p(Never)', '—'), UI.readout(ctl, 'p(Yes)', '—')];
      const rMg = UI.readout(ctl, 'log margin No/Never', '—'), rL = UI.readout(ctl, 'DPO loss (β = 1)', '—');
      nocase(ctl);
      function upd() {
        const p = probs(S.s);
        rP[0].set(fmt(p[0], 3), p[0] < 0.2 ? 'bad' : null); rP[1].set(fmt(p[1], 3)); rP[2].set(fmt(p[2], 3), p[2] > 0.5 ? 'bad' : null);
        rMg.set(fmt(0.4 * S.s, 2), 'ok'); rL.set(fmt(-logsig(0.4 * S.s), 3), 'ok');
      }
      upd();
      let lastSet = -1;
      V.fig(fig, (g, w, h, t, dt) => {
        const C = V.C();
        const narrow = w < 640;
        const COL = [C.a, C.c, C.d];
        if (S.play && !LA.paused) {
          if (S.s < 10) S.s = Math.min(10, S.s + dt * 1.2);
          else { S.hold += dt; if (S.hold > 2.2) { S.s = 0; S.hold = 0; } }
          if (Math.abs(S.s - lastSet) > 0.05) { sl.set(Math.round(S.s * 10) / 10, true); lastSet = S.s; upd(); }
        }
        const s = S.s, p = probs(s);
        // painéis
        const P1 = narrow ? { x: 8, y: 8, w: w - 16, h: 260 } : { x: 8, y: 8, w: w * 0.4 - 12, h: h - 16 };
        const P2 = narrow ? { x: 8, y: 278, w: w - 16, h: 180 } : { x: w * 0.4 + 4, y: 8, w: w * 0.6 - 12, h: h * 0.5 - 8 };
        const P3 = narrow ? { x: 8, y: 468, w: w - 16, h: h - 476 } : { x: w * 0.4 + 4, y: h * 0.5 + 6, w: w * 0.6 - 12, h: h * 0.5 - 14 };
        [P1, P2, P3].forEach((Pn) => V.box(g, Pn.x, Pn.y, Pn.w, Pn.h, { r: 10, fill: V.alpha(C.panel, 0.55), stroke: V.alpha(C.muted, 0.25), lw: 1 }));
        // --- P1: plano 2D
        V.text(g, 'unembeddings W and the hidden state h', P1.x + 10, P1.y + 18, { size: 11.5, weight: 600, color: C.muted });
        const U = Math.min((P1.w - 80) / 1.45, (P1.h - 74) / 1.95);
        const ox = P1.x + 34, oy = P1.y + 30 + 1.68 * U;
        V.line(g, ox - 20, oy, ox + U * 1.4, oy, { color: V.alpha(C.muted, 0.35), lw: 1 }); V.line(g, ox, oy + U * 0.25, ox, oy - U * 1.65, { color: V.alpha(C.muted, 0.35), lw: 1 });
        W.forEach((v, k) => {
          const x2 = ox + v[0] * U, y2 = oy - v[1] * U;
          V.arrow(g, ox, oy, x2, y2, { color: COL[k], lw: 2.6, head: 9 });
          const lx = P1.x + P1.w - 142, lyy = P1.y + P1.h - 72 + k * 18;
          V.line(g, lx, lyy - 4, lx + 16, lyy - 4, { color: COL[k], lw: 2.6 });
          sym(g, 'W_' + NM[k] + ' = (' + fmt(v[0], v[0] % 1 ? 1 : 0) + ', ' + fmt(v[1], v[1] % 1 ? 1 : 0) + ')', lx + 22, lyy, { size: 11.5, weight: 600, color: COL[k] });
        });
        const hn = Math.hypot(2, s), hx = 2 / hn, hy = s / hn;
        const hx2 = ox + hx * U * 1.25, hy2 = oy - hy * U * 1.25;
        V.glow(g, hx2, hy2, 16, C.b, 0.8);
        V.arrow(g, ox, oy, hx2, hy2, { color: C.b, lw: 3, head: 10 });
        sym(g, 'h/|h|', hx2 + 8, hy2 - 4, { size: 12, weight: 700, color: C.b });
        V.arrow(g, hx2, hy2, hx2, hy2 - U * 0.4, { color: C.d, lw: 2, head: 7, dash: [4, 3] });
        sym(g, '∇ = W_No − W_Never', Math.min(hx2 + 6, P1.x + P1.w - 130), hy2 - U * 0.4 - 4, { size: 11, color: C.d, weight: 600 });
        V.text(g, 'h = (2, ' + fmt(s, 1) + ')   |h| = ' + fmt(hn, 2), P1.x + 10, P1.y + P1.h - 8, { size: 11.5, font: 'mono', color: C.ink });
        // --- P2: W·h = logits → softmax
        V.text(g, 'logits = W · h  →  softmax', P2.x + 10, P2.y + 18, { size: 11.5, weight: 600, color: C.muted });
        const cs = Math.min(40, (P2.w - 60) / 9), chh = 24;
        const mx0 = P2.x + 52, my0 = P2.y + 34;
        const cell = (x, y, ww, v, col) => { V.box(g, x, y, ww, chh, { r: 4, fill: V.alpha(col, 0.2), stroke: V.alpha(col, 0.6), lw: 1 }); V.text(g, fmt(v, v % 1 === 0 && Math.abs(v) < 10 ? 0 : 1), x + ww / 2, y + 16, { size: 11.5, font: 'mono', color: C.ink, align: 'center' }); };
        W.forEach((row, k) => {
          V.text(g, NM[k], mx0 - 6, my0 + k * (chh + 4) + 16, { size: 11.5, color: COL[k], align: 'right', weight: 700 });
          row.forEach((v, j) => cell(mx0 + j * (cs + 3), my0 + k * (chh + 4), cs, v, C.muted));
        });
        const hx0 = mx0 + 2 * (cs + 3) + 16;
        V.text(g, '·', hx0 - 9, my0 + chh + 20, { size: 18, color: C.ink, align: 'center' });
        cell(hx0, my0 + 14, cs, 2, C.b); cell(hx0, my0 + 14 + chh + 4, cs, Math.round(s * 10) / 10, C.b);
        const lx0 = hx0 + cs + 22;
        V.text(g, '=', lx0 - 11, my0 + chh + 20, { size: 15, color: C.ink, align: 'center' });
        const lg = [2 + 0.2 * s, 2 - 0.2 * s, s];
        lg.forEach((v, k) => cell(lx0, my0 + k * (chh + 4), cs + 6, v, COL[k]));
        // barras de probabilidade
        const bx0 = lx0 + cs + 30, bw2 = P2.x + P2.w - bx0 - 12;
        V.arrow(g, lx0 + cs + 10, my0 + chh + 16, bx0 - 4, my0 + chh + 16, { color: C.muted, lw: 1.4, head: 6 });
        p.forEach((v, k) => {
          const y = my0 + k * (chh + 4);
          V.rr(g, bx0, y + 3, bw2, chh - 6, 4); g.fillStyle = V.alpha(C.muted, 0.1); g.fill();
          V.rr(g, bx0, y + 3, Math.max(2, bw2 * v), chh - 6, 4); g.fillStyle = V.alpha(COL[k], 0.8); g.fill();
          V.text(g, fmt(v, 3), bx0 + Math.min(bw2 * v + 4, bw2 - 36), y + 16, { size: 11.5, font: 'mono', color: C.ink, weight: 600 });
        });
        const by = my0 + 3 * (chh + 4) + 18;
        V.text(g, 'margin No/Never = 0.4·s = ' + fmt(0.4 * s, 2) + ' → DPO loss ' + fmt(-logsig(0.4 * s), 3) + ' ↓', P2.x + 10, by, { size: 11.5, font: 'mono', color: C.a, maxW: P2.w - 20 });
        V.text(g, p[0] < 0.4683 && s > 1 ? 'but p(No) fell below its start (0.468)!' : 'p(No) at the start: 0.468', P2.x + 10, by + 19, { size: 11.5, color: p[0] < 0.4683 && s > 1 ? RED() : C.muted, weight: 600, maxW: P2.w - 20 });
        // --- P3: curvas ao longo de s
        const A = { x: P3.x + 40, y: P3.y + 28, w: P3.w - 56, h: P3.h - 58 };
        V.text(g, 'probabilities over training', P3.x + 10, P3.y + 18, { size: 11.5, weight: 600, color: C.muted });
        const ax = axes(g, A, [0, 10], [0, 1], { xlabel: 's', nx: 5, ny: 4 });
        for (let k = 0; k < 3; k++) {
          g.beginPath(); CURV.forEach((q, j) => { const x = ax.X(j / 10), y = ax.Y(q[k]); if (!j) g.moveTo(x, y); else g.lineTo(x, y); });
          g.strokeStyle = COL[k]; g.lineWidth = 2.4; g.stroke();
          const lj = k === 2 ? 70 : k === 0 ? 8 : 30;
          V.text(g, NM[k], ax.X(lj / 10) + 4, ax.Y(CURV[lj][k]) - 6, { size: 11.5, weight: 700, color: COL[k] });
        }
        V.line(g, ax.X(s), A.y, ax.X(s), A.y + A.h, { color: V.alpha(C.ink, 0.5), lw: 1.2, dash: [3, 3] });
        p.forEach((v, k) => V.dot(g, ax.X(s), ax.Y(v), 4.5, COL[k]));
      });
    })();

    /* ======================================================== FIG 10 — o painel do DPO */
    (function () {
      const fig = root.querySelector('#fig-painel');
      if (!fig) return;
      adaptH(fig, 540, 900, 700);
      const ex = (t, k) => 1 - Math.exp(-k * t);
      const hump = (t, tp) => (t / tp) * Math.exp(1 - t / tp);
      const B = 0.1, LPC = -250, LPR = -265;
      const SC = {
        saud: {
          rc: (t) => 0.25 * ex(t, 4), rr: (t) => -0.9 * ex(t, 4), acc: (t) => 0.5 + 0.22 * ex(t, 5),
          win: (t) => 50 + 12 * ex(t, 4), len: (t) => 1 + 0.05 * ex(t, 3), kl: (t) => 4 * ex(t, 3),
          diag: 'Healthy: the implicit rewards pull apart, accuracy settles between 0.6 and 0.8, logps/chosen stays stable (or rises a little) and the per-checkpoint win rate rises and levels off, with length nearly constant.',
        },
        desl: {
          rc: (t) => -1.4 * ex(t, 2.5), rr: (t) => -3.6 * ex(t, 2.5), acc: (t) => 0.5 + 0.4 * ex(t, 4),
          win: (t) => 50 + 7 * ex(t, 8) - 15 * Math.pow(t, 1.3), len: (t) => 1 + 0.08 * ex(t, 3), kl: (t) => 10 * ex(t, 2.5) + 3 * t,
          diag: 'Likelihood displacement: margins and accuracy climb nicely, but rewards/chosen turns negative and logps/chosen plummets along with logps/rejected; the mass went to other responses. Remedies: NLL on the chosen responses (loss_type=["sigmoid","sft"]), a larger β, apo_zero, DPOP, filtering out high-CHES pairs.',
        },
        verb: {
          rc: (t) => 0.9 * ex(t, 6), rr: (t) => -0.7 * ex(t, 6), acc: (t) => 0.5 + 0.47 * ex(t, 10),
          win: (t) => 50 + 15 * hump(t, 0.1), len: (t) => 1 + 0.9 * ex(t, 22) + 0.15 * t, kl: (t) => 8 * ex(t, 10) + 5 * t,
          diag: 'Verbosity: length nearly doubles in the first ~10% of the epoch (Park and colleagues, HH with β = 0.1), accuracy shoots to ~1 and the win rate from a length-biased judge peaks right there and then falls. Remedies: R-DPO, sigmoid_norm, ld_alpha, SimPO, a length-controlled judge.',
        },
        sobre: {
          rc: (t) => 0.3 * ex(t, 3) - 0.25 * t, rr: (t) => -0.8 * ex(t, 3) - 1.7 * t, acc: (t) => 0.5 + 0.3 * ex(t, 4),
          win: (t) => 50 + 13 * hump(t, 0.25), len: (t) => 1 + 0.25 * t, kl: (t) => 3 * ex(t, 3) + 16 * t,
          diag: 'Over-optimization: the margins keep growing, but the win rate forms a hump that peaks around 25% of the epoch (DAAs with a loose KL). Remedies: evaluate and save per checkpoint, stop early, a larger β, IPO, a moving reference (TR-DPO).',
        },
      };
      const S = { sc: 'saud', t0: 0 };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Scenario', options: [{ v: 'saud', label: 'healthy' }, { v: 'desl', label: 'displacement' }, { v: 'verb', label: 'verbosity' }, { v: 'sobre', label: 'over-optimization' }], value: 'saud', onChange: (v) => { S.sc = v; S.t0 = H10 ? H10.t : 0; } });
      const rKL = UI.readout(ctl, 'KL to π ref (synthetic)', '—');
      htmlLabel(rKL, 'KL to π<sub>ref</sub> (synthetic)');
      nocase(ctl);
      let klTxt = '';
      const range = (fns) => { let a = Infinity, b = -Infinity; for (let k = 0; k <= 50; k++) fns.forEach((f) => { const v = f(k / 50); if (v < a) a = v; if (v > b) b = v; }); const pd = (b - a) * 0.12 || 0.5; return [a - pd, b + pd]; };
      function chart(g, R, title, series, prog, o = {}) {
        const C = V.C();
        V.box(g, R.x, R.y, R.w, R.h, { r: 10, fill: V.alpha(C.panel, 0.55), stroke: V.alpha(C.muted, 0.25), lw: 1 });
        V.text(g, title, R.x + 10, R.y + 17, { size: 11.5, weight: 700, color: C.ink, font: 'mono', maxW: R.w - 20 });
        const A = { x: R.x + 46, y: R.y + 46, w: R.w - 46 - (o.right ? 40 : 12), h: R.h - 46 - 38 };
        const yr = o.yr || range(series.filter((s) => !s.right).map((s) => s.f));
        const ax = axes(g, A, [0, 1], yr, { nx: 4, ny: 3, fx: (v) => fmtTick(v), noX: false, xlabel: 'epoch' });
        if (o.band) { g.fillStyle = V.alpha(C.e, 0.14); g.fillRect(A.x, ax.Y(o.band[1]), A.w, ax.Y(o.band[0]) - ax.Y(o.band[1])); V.text(g, 'normal zone', A.x + A.w - 6, ax.Y(o.band[1]) + 14, { size: 11, color: C.e, align: 'right', weight: 600 }); }
        if (yr[0] < 0 && yr[1] > 0) V.line(g, A.x, ax.Y(0), A.x + A.w, ax.Y(0), { color: V.alpha(C.muted, 0.6), lw: 1, dash: [2, 3] });
        let Y2 = null;
        if (o.right) {
          Y2 = (v) => A.y + A.h - ((v - o.right[0]) / (o.right[1] - o.right[0])) * A.h;
          ticks(o.right[0], o.right[1], 3).forEach((v) => V.text(g, fmtTick(v) + '×', A.x + A.w + 4, Y2(v) + 4, { size: 11, font: 'mono', color: o.rightCol }));
        }
        series.forEach((s) => {
          const YY = s.right ? Y2 : ax.Y;
          curve(g, 0, 1, s.f, ax.X, YY, { color: V.alpha(s.c, 0.18), lw: 1.4, n: 60 });
          curve(g, 0, prog, s.f, ax.X, YY, { color: s.c, lw: 2.4, n: Math.max(2, Math.round(60 * prog)), dash: s.dash });
          V.dot(g, ax.X(prog), YY(s.f(prog)), 3.5, s.c);
        });
        V.line(g, ax.X(prog), A.y, ax.X(prog), A.y + A.h, { color: V.alpha(C.ink, 0.3), lw: 1 });
        // legenda
        let lx = R.x + 10;
        const ly = R.y + 35;
        series.forEach((s) => {
          if (lx > R.x + R.w - 40) return;
          V.line(g, lx, ly - 4, lx + 14, ly - 4, { color: s.c, lw: 2.4, dash: s.dash });
          V.text(g, s.l, lx + 18, ly, { size: 11, color: C.ink, font: 'mono' });
          g.font = V.font(11, 'mono'); lx += 18 + g.measureText(s.l).width + 12;
        });
      }
      const H10 = V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const narrow = w < 700;
        const sc = SC[S.sc];
        const tt = t - S.t0, cyc = 11;
        const prog = clamp((tt % cyc) / 9, 0.001, 1);
        const diagH = narrow ? 150 : 72;
        const top = 6, gp = 10;
        const areaH = h - diagH - top - 18;
        const cellsR = [];
        if (narrow) { const ch = (areaH - 3 * gp) / 4; for (let k = 0; k < 4; k++) cellsR.push({ x: 6, y: top + k * (ch + gp), w: w - 12, h: ch }); }
        else { const cw = (w - 12 - gp) / 2, ch = (areaH - gp) / 2; for (let k = 0; k < 4; k++) cellsR.push({ x: 6 + (k % 2) * (cw + gp), y: top + Math.floor(k / 2) * (ch + gp), w: cw, h: ch }); }
        chart(g, cellsR[0], 'implicit rewards (β·h)', [{ f: sc.rc, c: C.a, l: 'chosen' }, { f: sc.rr, c: C.c, l: 'rejected' }, { f: (x) => sc.rc(x) - sc.rr(x), c: C.d, l: 'margins' }], prog);
        chart(g, cellsR[1], 'rewards/accuracies', [{ f: sc.acc, c: C.b, l: 'accuracies' }], prog, { yr: [0.4, 1.05], band: [0.6, 0.8] });
        chart(g, cellsR[2], 'logps/chosen · logps/rejected', [{ f: (x) => LPC + sc.rc(x) / B, c: C.a, l: 'chosen' }, { f: (x) => LPR + sc.rr(x) / B, c: C.c, l: 'rejected' }], prog);
        chart(g, cellsR[3], 'per-checkpoint evaluation', [{ f: sc.win, c: C.d, l: 'win rate %' }, { f: sc.len, c: C.f, l: 'length', right: true, dash: [6, 4] }], prog, { yr: [40, 70], right: [0.8, 2.2], rightCol: C.f });
        // diagnóstico
        const dy = h - diagH - 6;
        V.box(g, 6, dy, w - 12, diagH, { r: 10, fill: V.alpha(S.sc === 'saud' ? C.a : C.d, 0.1), stroke: S.sc === 'saud' ? C.a : C.d, lw: 1.4 });
        V.text(g, 'epoch ' + fmt(prog, 2) + ' · synthetic curves', w - 18, dy + 17, { size: 11, color: C.muted, align: 'right' });
        V.wrap(g, sc.diag, 18, dy + 34, w - 36, { size: 12, color: C.ink, lineH: 16, maxLines: narrow ? 7 : 3 });
        const kt = fmt(sc.kl(prog), 1) + ' nats';
        if (kt !== klTxt) { klTxt = kt; rKL.set(kt); }
        V.text(g, S.sc === 'saud' ? 'diagnosis' : 'diagnosis and remedy', 18, dy + 17, { size: 11.5, weight: 700, color: S.sc === 'saud' ? C.a : C.d });
      });
    })();
  },
});

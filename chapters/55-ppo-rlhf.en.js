LA.chapter({
  id: 'ppo-rlhf',
  part: 11,
  order: 55,
  title: 'PPO and the RLHF pipeline',
  navTitle: 'PPO & RLHF',
  subtitle: 'From the clipped objective to the whole system: the KL leash, the four models in memory, the inference engine generating rollouts inside the loop and the tricks that keep training on its feet.',
  minutes: 45,
  recipes: [],
  tags: ['PPO', 'RLHF', 'clipping', 'KL penalty', 'InstructGPT', 'critic', 'GAE', 'PPO-ptx', 'vLLM in the loop', 'OpenRLHF', 'verl'],
  goals: [
    'Derive and interpret the <b>clipped objective</b> \\(L^{CLIP}\\) case by case, explaining why the \\(\\min\\) cuts the incentive and how this mimics the trust region of TRPO',
    'Assemble the full PPO loss (clipped value, entropy, epochs and minibatches) and the InstructGPT <b>SFT → RM → PPO</b> pipeline, with the real hyperparameters',
    'Explain the <b>KL-shaped reward</b> \\(r - \\beta\\log(\\pi/\\pi_{\\text{ref}})\\), the optimal policy \\(\\pi^*\\propto\\pi_{\\text{ref}}e^{r/\\beta}\\), adaptive control of \\(\\beta\\) and PPO-ptx',
    'Estimate the memory of the <b>four models</b> and the cost of one iteration, and choose between a colocated, disaggregated or asynchronous inference engine',
    'Diagnose instabilities by their symptoms and pick the right trick and tool in 2026 (OpenRLHF, verl; TRL removed <code>PPOTrainer</code>)',
  ],
  prereqs: ['rl-fundamentos', 'preferencias-reward-models'],
  summary: [
    'PPO reuses each batch for several steps, using the ratio \\(\\rho_t = \\pi_\\theta/\\pi_{\\theta_{\\text{old}}}\\) as a per-token importance weight. The \\(\\min\\) in \\(L^{CLIP}\\) zeroes the incentive when \\(\\rho\\) leaves \\([1-\\varepsilon, 1+\\varepsilon]\\) in the direction that would improve the objective (\\(\\varepsilon = 0.2\\)): it is an incentive cut, not a wall, and it replaces the expensive TRPO with a first-order optimizer.',
    'Full loss: \\(L^{CLIP} - c_1L^{VF} + c_2S\\). In RLHF, the critic is a separate LM, initialized from the RM, that predicts \\(V(s_t)\\) at every token; the value is clipped (\\(\\hat\\varepsilon = 0.2\\)); the entropy bonus stays off because the KL already acts as entropy; few epochs (InstructGPT 1, Stiennon and N+ 4).',
    'InstructGPT: SFT (12,725 prompts) → 6B RM (33,207 prompts, \\(\\binom K2\\) pairs in the same batch element) → PPO (31,144 prompts, \\(\\beta = 0.02\\), 256k episodes, batch 512) + ptx (\\(\\gamma = 27.8\\)). The 1.3B was preferred over GPT-3 175B; and PPO-ptx for the 175B cost 60 pf/s-days, versus 3,640 for the pretraining of GPT-3.',
    'Leashed reward: \\(r_t = -\\beta(\\log\\pi_{\\text{old}} - \\log\\pi_{\\text{ref}}) + \\mathbb 1[t=T]\\,r_\\phi\\). The optimal policy reweights the SFT by \\(e^{r/\\beta}\\): a small \\(\\beta\\) hands the mass to the exploit. The Ziegler controller (\\(K_\\beta = 0.1\\), ±2% per step) aims at a target KL instead of a fixed \\(\\beta\\).',
    'Four models: trainable policy and critic (~16 B/parameter), frozen reference and RM (2 B/parameter) → \\(36N\\) bytes; 7B gives 252 GB, more than one DGX Spark (~121.69 GiB). Left out of the math: activations, the inference engine copy and the KV of the rollouts.',
    'The inference engine generates the rollouts (Llama 2: generating with FSDP was ~20× slower; OpenRLHF: inference is often &gt; 90% of the time). Colocated (sleep/wake, IPC) or disaggregated (NCCL broadcast; the 14 GB of a 7B ≈ 0.9 s on the GB10); async RL is 1-step off-policy and measured gains of ~25% to ~70%. \\(\\log\\pi_{\\text{old}}\\) is recomputed in the training engine; TIS, an FP32 LM head and R3 fix the mismatch.',
    'Stability: dropout off, EOS ≠ PAD and the “EOS trick”, normalized RM, whitened advantages, clipped value, gradient clip at 1.0; monitor the KL, the length and the ratio in the 1st epoch. Tools: TRL removed <code>PPOTrainer</code> in v1.13.0; use OpenRLHF (<code>train_ppo_ray</code>) or verl (<code>main_ppo</code>).',
  ],
  quiz: [
    { q: 'With \\(\\varepsilon = 0.2\\), a token with advantage \\(\\hat A = +2\\) has ratio \\(\\rho = 1.3\\). What is the derivative of \\(L^{CLIP}\\) with respect to \\(\\rho\\)?', o: ['2: the gradient keeps pushing the good action', '0: the clipped term is the smaller one, and the incentive to go any higher has been cut', '−1: PPO pulls the ratio back to 1', 'It depends on the \\(\\beta\\) of the KL to the reference'], a: 1, e: '\\(\\min(1.3\\times2,\\ 1.2\\times2) = 2.4\\), which is the clipped term, constant in \\(\\rho\\): derivative 0. PPO does not pull the ratio back (it is not a wall), it just stops pushing. And the \\(\\beta\\) of the KL to the reference is something else: it controls the leash to the SFT, not the clip of each iteration.' },
    { q: 'Now the token has \\(\\hat A = -1\\) and \\(\\rho = 1.3\\) (the bad action became more likely). What happens?', o: ['Nothing: outside the band the gradient is always zero', 'The \\(\\min\\) picks the unclipped term (\\(-1.3 &lt; -1.2\\)), and the penalty continues with derivative −1', 'The ratio is reset to 1', 'The critic absorbs the error'], a: 1, e: 'With a negative advantage, \\(\\rho\\hat A = -1.3\\) is smaller than \\(\\operatorname{clip}(\\rho)\\hat A = -1.2\\), so the \\(\\min\\) picks the unclipped term and the derivative stays at −1. That is the asymmetry of PPO: it ignores the change in the ratio only when that change would improve the objective.' },
    { q: 'How is the reward of each token assembled in classic PPO-RLHF (InstructGPT, N+)?', o: ['The RM score is split equally among all the tokens', 'Every token gets \\(-\\beta(\\log\\pi_{\\text{old}} - \\log\\pi_{\\text{ref}})\\), and only the last one (EOS) also gets the RM score', 'The RM is evaluated at every prefix and the score of each prefix goes to its token', 'Only the EOS gets a reward; the KL enters only the loss'], a: 1, e: 'It is the per-token decomposition of \\(R = r_\\phi - \\beta\\log(\\pi/\\pi_{\\text{ref}})\\): with \\(\\gamma = 1\\), the sum of the \\(r_t\\) gives \\(R\\). Putting the KL in the loss, rather than in the reward, is what GRPO does (with the \\(k_3\\) estimator).' },
    { q: 'In the optimal policy \\(\\pi^* \\propto \\pi_{\\text{ref}}\\,e^{r/\\beta}\\), what happens as \\(\\beta \\to 0\\)?', o: ['The policy goes back to the SFT', 'The mass concentrates on the response with the highest RM score, even if the SFT would almost never write it: it is the road to reward hacking', 'The entropy of the policy increases', 'Nothing changes: \\(\\beta\\) only affects training speed'], a: 1, e: 'With a small \\(\\beta\\), \\(e^{r/\\beta}\\) dominates \\(\\pi_{\\text{ref}}\\) and the highest score takes everything. In the example in the figure, with \\(\\beta = 0.3\\) the exploit already takes 68% of the mass and the mean real quality drops to −0.8. As \\(\\beta \\to \\infty\\), the policy stays at \\(\\pi_{\\text{ref}}\\).' },
    { q: 'A 7B policy and critic in full fine-tuning (Adam, mixed precision), plus a 7B reference and RM in bf16. How much do the states alone take, and does it fit on one DGX Spark?', o: ['56 GB; fits with room to spare', '126 GB; fits just barely', '252 GB (~235 GiB); does not fit on one Spark (~121.69 GiB) and, with activations and KV, not even on two', '504 GB; not even on four Sparks'], a: 2, e: '\\(2\\times16N + 2\\times2N = 36N\\) = 252 GB for \\(N = 7\\)B, without counting activations, the inference engine copy and the KV of the rollouts. Ways out: a smaller critic and RM, LoRA (reference = adapter turned off), or dropping the critic (GRPO). Offloading to the CPU does not free physical memory on the GB10.' },
    { q: 'Why does the scoring step recompute \\(\\log\\pi_{\\theta_{\\text{old}}}\\) in the training engine, instead of using the logprobs that vLLM returned?', o: ['Because vLLM does not return logprobs', 'Because, with the same weights, different kernels, precision and batching give different probabilities; using the vLLM ones would make the ratio start away from 1 and make training implicitly off-policy', 'Because the training engine is faster at generating', 'To save KV memory'], a: 1, e: '\\(\\pi_{\\text{vllm}}(a;\\theta) \\ne \\pi_{\\text{fsdp}}(a;\\theta)\\) even with the same parameters, and the problem gets worse with quantized rollouts and MoE. The PPO ratio sits between two probabilities from the training engine, and the difference between the engines is corrected separately (TIS, FP32 LM head, R3, deterministic kernels).' },
  ],
  mount(root, LA) {
    const V = LA.viz, UI = LA.ui;
    const fmt = (x, d = 2) => (+x).toFixed(d).replace('-', '−');
    const fmtS = (x, d = 2) => (x > 0 ? '+' : '') + fmt(x, d);
    const RED = () => V.C().red || '#FB2C36';
    const nocase = (c) => { const el = c && (c.el || c); if (el && el.querySelectorAll) el.querySelectorAll('label, span').forEach((n) => { if (!n.classList.contains('val')) n.style.textTransform = 'none'; }); return c; };
    function adaptH(fig, wide, narrow, bp = 640) {
      const st = fig.querySelector('.la-fig-stage');
      const set = () => { const want = (fig.clientWidth < bp ? narrow : wide) + 'px'; if (st.style.height !== want) st.style.height = want; };
      set();
      const ro = new ResizeObserver(set); ro.observe(fig);
      if (LA.onCleanup) LA.onCleanup(() => ro.disconnect());
    }
    const clock = (S, t) => { const d = S._t == null ? 0 : Math.max(0, Math.min(0.1, t - S._t)); S._t = t; return d; };
    /** character: rounded square with little eyes */
    function face(g, x, y, s, col, o = {}) {
      const C = V.C();
      if (o.alpha != null) g.globalAlpha = o.alpha;
      V.box(g, x - s / 2, y - s / 2, s, s, { r: s * 0.28, fill: o.fill || V.alpha(col, 0.85), stroke: col, lw: o.lw || 1.3 });
      if (o.dash) { g.setLineDash([4, 3]); V.rr(g, x - s / 2, y - s / 2, s, s, s * 0.28); g.strokeStyle = col; g.lineWidth = 1.5; g.stroke(); g.setLineDash([]); }
      if (s >= 12) {
        const eye = o.ghost ? col : (C.dark ? '#06140d' : '#ffffff');
        const ex = s * 0.17, ey = y - s * 0.08, er = Math.max(1.1, s * 0.07);
        V.dot(g, x - ex, ey, er, eye); V.dot(g, x + ex, ey, er, eye);
        g.beginPath();
        if (o.mood === 'sad') g.arc(x, y + s * 0.3, s * 0.14, Math.PI * 1.15, Math.PI * 1.85);
        else if (o.mood === 'o') g.arc(x, y + s * 0.17, s * 0.07, 0, Math.PI * 2);
        else g.arc(x, y + s * 0.06, s * 0.15, Math.PI * 0.2, Math.PI * 0.8);
        g.strokeStyle = eye; g.lineWidth = Math.max(1, s * 0.06); g.stroke();
      }
      if (o.alpha != null) g.globalAlpha = 1;
    }
    function chip(g, x, y, w, h, label, o = {}) {
      const C = V.C();
      V.box(g, x, y, w, h, { r: o.r == null ? 6 : o.r, fill: o.fill || V.alpha(C.muted, 0.12), stroke: o.stroke || V.alpha(C.muted, 0.4), lw: o.lw || 1.2 });
      if (label != null) V.text(g, label, x + w / 2, y + h / 2 + 0.5, { size: o.size || 11.5, font: o.font || 'mono', weight: o.weight || 500, color: o.color || C.ink, align: 'center', base: 'middle', maxW: w - 4 });
    }
    function hatch(g, x, y, w, h, col, step = 7) {
      if (w <= 0 || h <= 0) return;
      g.save(); g.beginPath(); g.rect(x, y, w, h); g.clip();
      g.strokeStyle = col; g.lineWidth = 1;
      g.beginPath();
      for (let k = -h; k < w; k += step) { g.moveTo(x + k, y + h); g.lineTo(x + k + h, y); }
      g.stroke(); g.restore();
    }
    /** chart frame; returns the X(v), Y(v) mappings */
    function frame(g, A, x0, x1, y0, y1, o = {}) {
      const C = V.C();
      const X = (v) => A.x + ((v - x0) / (x1 - x0)) * A.w;
      const Y = (v) => A.y + A.h - ((v - y0) / (y1 - y0)) * A.h;
      g.lineWidth = 1;
      (o.xt || []).forEach((v) => { const x = X(v); g.strokeStyle = V.alpha(C.muted, 0.16); g.beginPath(); g.moveTo(x, A.y); g.lineTo(x, A.y + A.h); g.stroke(); V.text(g, o.xf ? o.xf(v) : fmt(v, 1), x, A.y + A.h + 14, { size: 11, font: 'mono', color: C.muted, align: 'center' }); });
      (o.yt || []).forEach((v) => { const y = Y(v); g.strokeStyle = V.alpha(C.muted, 0.16); g.beginPath(); g.moveTo(A.x, y); g.lineTo(A.x + A.w, y); g.stroke(); V.text(g, o.yf ? o.yf(v) : fmt(v, 1), A.x - 6, y, { size: 11, font: 'mono', color: C.muted, align: 'right', base: 'middle' }); });
      g.strokeStyle = V.alpha(C.muted, 0.55); g.lineWidth = 1.2;
      g.beginPath(); g.moveTo(A.x, A.y); g.lineTo(A.x, A.y + A.h); g.lineTo(A.x + A.w, A.y + A.h); g.stroke();
      if (o.xl) V.text(g, o.xl, A.x + A.w, A.y + A.h + 30, { size: 11, color: C.muted, align: 'right' });
      if (o.yl) V.text(g, o.yl, A.x, A.y - (o.ylOff || 9), { size: 11.5, color: C.muted, weight: 600 });
      return { X, Y };
    }
    function curve(g, f, x0, x1, X, Y, o = {}) {
      const n = o.n || 120;
      g.beginPath();
      for (let k = 0; k <= n; k++) { const v = x0 + ((x1 - x0) * k) / n, px = X(v), py = Y(f(v)); if (k) g.lineTo(px, py); else g.moveTo(px, py); }
      g.strokeStyle = o.color; g.lineWidth = o.lw || 2; if (o.dash) g.setLineDash(o.dash); g.stroke(); g.setLineDash([]);
    }

    /* ======================================================= FIG 1 — cliff */
    (function () {
      const fig = root.querySelector('#fig-penhasco');
      adaptH(fig, 420, 620);
      const TOK = ['“20”', '“14”', '“Let”'];
      const z0 = [2.0, 1.0, 0.1], p0 = V.softmax(z0), AI = 1, K = 10;
      const S = { A: 1.5, clip: true, eps: 0.2, eta: 0.05 };
      let parts = [];
      /** full ratio trajectory for one mode (drawn as the ghost of the opposite mode) */
      function traj(clip) {
        let z = z0.slice(); const hs = [1];
        for (let k = 0; k < K; k++) {
          const p = V.softmax(z), rho = p[AI] / p0[AI];
          const zero = clip && ((S.A > 0 && rho > 1 + S.eps) || (S.A < 0 && rho < 1 - S.eps));
          if (!zero) z = z.map((v, j) => v + S.eta * S.A * rho * ((j === AI ? 1 : 0) - p[j]));
          hs.push(V.softmax(z)[AI] / p0[AI]);
        }
        return hs;
      }
      function reset() { S.z = z0.slice(); S.hist = [1]; S.blocked = [false]; S.k = 0; S.tStep = 0; S.hold = 0; S.show = p0.slice(); S.alt = traj(!S.clip); parts = []; upd(); }
      function stepOnce() {
        const p = V.softmax(S.z), rho = p[AI] / p0[AI];
        const zero = S.clip && ((S.A > 0 && rho > 1 + S.eps) || (S.A < 0 && rho < 1 - S.eps));
        if (!zero) S.z = S.z.map((v, j) => v + S.eta * S.A * rho * ((j === AI ? 1 : 0) - p[j]));
        const p2 = V.softmax(S.z);
        S.hist.push(p2[AI] / p0[AI]); S.blocked.push(zero); S.k++;
        const moved = Math.abs(p2[AI] - p[AI]);
        const n = Math.min(26, Math.round(moved * 90));
        for (let q = 0; q < n; q++) { const other = q % 2 ? 0 : 2; parts.push({ from: S.A > 0 ? other : AI, to: S.A > 0 ? AI : other, u: -q * 0.03, j: Math.random() }); }
        upd();
      }
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Advantage', options: [{ v: 1.5, label: 'Â = +1.5' }, { v: -1.5, label: 'Â = −1.5' }], value: 1.5, onChange: (v) => { S.A = v; reset(); } });
      UI.toggle(ctl, { label: 'PPO clip', value: true, onChange: (v) => { S.clip = v; reset(); } });
      nocase(UI.slider(ctl, { label: 'ε', min: 0.1, max: 0.3, step: 0.05, value: 0.2, fmt: (v) => fmt(v, 2), onInput: (v) => { S.eps = v; reset(); } }));
      nocase(UI.slider(ctl, { label: 'step η', min: 0.02, max: 1, step: 0.01, value: 0.05, fmt: (v) => fmt(v, 2), onInput: (v) => { S.eta = v; reset(); } }));
      UI.button(ctl, '↺ restart', () => reset());
      const rR = nocase(UI.readout(ctl, 'token ρ', '—')), rP = nocase(UI.readout(ctl, 'π(“14”)', '—')), rK = nocase(UI.readout(ctl, 'KL(π_old‖π)', '—'));
      function upd() {
        if (!S.hist) return;
        const p = V.softmax(S.z), rho = p[AI] / p0[AI];
        const out = rho > 1 + S.eps + 1e-9 || rho < 1 - S.eps - 1e-9;
        rR.set(fmt(rho, 3), out ? (S.clip ? 'warn' : 'bad') : 'ok');
        rP.set(fmt(p0[AI], 3) + ' → ' + fmt(p[AI], 3));
        const kl = p0.reduce((s, q, j) => s + q * Math.log(q / p[j]), 0);
        rK.set(fmt(kl, 4) + ' nat', kl > 0.05 ? 'bad' : kl > 0.02 ? 'warn' : 'ok');
      }
      reset();
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const dd = clock(S, t);
        if (S.k < K) { S.tStep += dd; if (S.tStep > 0.6) { S.tStep = 0; stepOnce(); } }
        else { S.hold += dd; if (S.hold > 2.6) reset(); }
        const p = V.softmax(S.z);
        S.show = S.show.map((v, j) => v + (p[j] - v) * Math.min(1, dd * 6));
        const wide = w >= 640;
        const L = wide ? { x: 14, y: 8, w: w * 0.5 - 24, h: h - 16 } : { x: 10, y: 6, w: w - 20, h: h * 0.55 - 6 };
        const R = wide ? { x: w * 0.5 + 40, y: 30, w: w * 0.5 - 56, h: h - 80 } : { x: 46, y: h * 0.55 + 30, w: w - 62, h: h * 0.45 - 82 };
        // ---- left: logits + bars
        const colW = L.w / 3;
        V.text(g, 'logits z (the parameters)', L.x, L.y + 12, { size: 11.5, color: C.muted, weight: 600 });
        const cellW = Math.min(74, colW * 0.62), cellH = 26, cy = L.y + 22;
        const bTop = cy + cellH + 40, bBot = L.y + L.h - 50, bH = bBot - bTop;
        const bw = Math.min(56, colW * 0.46);
        for (let j = 0; j < 3; j++) {
          const cx = L.x + colW * (j + 0.5);
          const zc = V.cmap(V.clamp(S.z[j] / 3.2, -1, 1), 'div');
          V.box(g, cx - cellW / 2, cy, cellW, cellH, { r: 5, fill: zc, stroke: j === AI ? C.d : V.alpha(C.muted, 0.4), lw: j === AI ? 2 : 1 });
          V.text(g, fmt(S.z[j], 2), cx, cy + cellH / 2 + 1, { size: 12, font: 'mono', weight: 600, align: 'center', base: 'middle', color: Math.abs(S.z[j]) > 1.9 ? (C.dark ? '#00210F' : '#fff') : C.ink });
          // bar
          const hOld = p0[j] * bH, hNow = S.show[j] * bH;
          const col = j === AI ? (S.A > 0 ? C.a : C.c) : C.b;
          V.box(g, cx - bw / 2, bBot - hNow, bw, Math.max(1, hNow), { r: 4, fill: V.alpha(col, 0.75), stroke: col, lw: 1.2 });
          g.setLineDash([4, 3]); V.rr(g, cx - bw / 2 - 3, bBot - hOld, bw + 6, hOld, 4); g.strokeStyle = V.alpha(C.ink, 0.55); g.lineWidth = 1.2; g.stroke(); g.setLineDash([]);
          V.text(g, fmt(S.show[j], 3), cx, bBot - hNow - 6, { size: 11.5, font: 'mono', align: 'center', color: C.ink, weight: 600 });
          V.text(g, TOK[j], cx, bBot + 16, { size: 12, align: 'center', color: j === AI ? C.ink : C.muted, weight: j === AI ? 700 : 500 });
          if (j === AI) {
            face(g, cx, bBot + 34, 18, S.A > 0 ? C.a : C.c, { mood: S.A > 0 ? undefined : 'sad' });
            V.text(g, 'sampled · Â = ' + fmtS(S.A, 1), cx, cy + cellH + 16, { size: 11, font: 'mono', align: 'center', color: S.A > 0 ? C.a : C.c, weight: 600 });
          }
        }
        // mass particles migrating
        for (let q = parts.length - 1; q >= 0; q--) {
          const P = parts[q]; P.u += dd * 1.6;
          if (P.u >= 1) { parts.splice(q, 1); continue; }
          if (P.u < 0) continue;
          const x1 = L.x + colW * (P.from + 0.5), x2 = L.x + colW * (P.to + 0.5);
          const y1 = bBot - S.show[P.from] * bH * (0.3 + 0.5 * P.j), y2 = bBot - S.show[P.to] * bH * (0.3 + 0.5 * P.j);
          const u = V.ease.inOut(P.u), x = V.lerp(x1, x2, u), y = V.lerp(y1, y2, u) - Math.sin(u * Math.PI) * 28;
          V.dot(g, x, y, 2.4, V.alpha(S.A > 0 ? C.a : C.c, 0.85));
        }
        // ---- right: ratio per step
        const ymax = S.A > 0 ? Math.max(2.2, Math.max(...S.hist, ...S.alt) * 1.08) : 1.4;
        const F = frame(g, R, 0, K, 0, ymax, { xt: [0, 2, 4, 6, 8, 10], xf: (v) => String(v), yt: S.A > 0 ? [0, 0.5, 1, 1.5, 2].concat(ymax > 3 ? [3] : [], ymax > 4 ? [4] : []) : [0, 0.5, 1], yf: (v) => fmt(v, 1), xl: 'gradient steps on the same batch', yl: 'ratio ρ of the sampled token' });
        g.fillStyle = V.alpha(C.e, 0.16); g.fillRect(R.x, F.Y(1 + S.eps), R.w, F.Y(1 - S.eps) - F.Y(1 + S.eps));
        V.line(g, R.x, F.Y(1), R.x + R.w, F.Y(1), { color: V.alpha(C.e, 0.7), dash: [4, 4], lw: 1.2 });
        // band label on the side the steps never visit: below it with Â > 0, above it with Â < 0
        V.text(g, 'band [1−ε, 1+ε]', R.x + R.w - 4, S.A > 0 ? F.Y(1 - S.eps) + 14 : F.Y(1 + S.eps) - 5, { size: 11, color: C.e, align: 'right', font: 'mono' });
        // ghost: what the other mode would do with the same batch
        g.beginPath();
        S.alt.forEach((r, k) => { const x = F.X(k), y = F.Y(Math.min(ymax, r)); if (k) g.lineTo(x, y); else g.moveTo(x, y); });
        g.strokeStyle = V.alpha(S.clip ? RED() : C.d, 0.55); g.lineWidth = 1.6; g.setLineDash([5, 4]); g.stroke(); g.setLineDash([]);
        const ae = S.alt[K];
        let gy = F.Y(Math.min(ymax, ae)) + (ae > S.hist[S.hist.length - 1] ? -8 : 16);
        if (gy < R.y + 28) gy = F.Y(Math.min(ymax, ae)) + 16; // does not collide with the message at the top
        V.text(g, S.clip ? 'no clip' : 'with clip', F.X(K) - 4, gy, { size: 11, font: 'mono', color: S.clip ? RED() : C.d, align: 'right' });
        g.beginPath();
        S.hist.forEach((r, k) => { const x = F.X(k), y = F.Y(Math.min(ymax, r)); if (k) g.lineTo(x, y); else g.moveTo(x, y); });
        g.strokeStyle = S.A > 0 ? C.a : C.c; g.lineWidth = 2.2; g.stroke();
        S.hist.forEach((r, k) => {
          const x = F.X(k), y = F.Y(Math.min(ymax, r));
          if (S.blocked[k]) { g.beginPath(); g.arc(x, y, 4.5, 0, Math.PI * 2); g.strokeStyle = C.d; g.lineWidth = 2; g.stroke(); }
          else V.dot(g, x, y, 3.6, S.A > 0 ? C.a : C.c);
        });
        const nb = S.blocked.filter(Boolean).length;
        const msg = !S.clip ? 'no clip: the surrogate keeps pushing' : nb ? '✂ incentive zeroed: ' + nb + ' step' + (nb > 1 ? 's' : '') + ' with no effect' : 'inside the band: the gradient still acts';
        V.text(g, msg, R.x + 6, R.y + 12, { size: 11.5, color: !S.clip ? RED() : nb ? C.d : C.muted, weight: 600, maxW: R.w - 10 });
      });
    })();

    /* ======================================================= FIG 2 — clip function */
    (function () {
      const fig = root.querySelector('#fig-clip');
      adaptH(fig, 400, 500);
      const S = { A: 2, eps: 0.2, rho: 0.55, run: true, hold: 0, userT: -99 };
      const X0 = 0.4, X1 = 1.8;
      const Lf = (r) => Math.min(r * S.A, V.clamp(r, 1 - S.eps, 1 + S.eps) * S.A);
      const dL = (r) => { if (S.A === 0) return 0; const un = r * S.A, cl = V.clamp(r, 1 - S.eps, 1 + S.eps) * S.A; return un <= cl + 1e-12 ? S.A : 0; };
      const start = () => (S.A >= 0 ? 0.55 : 1.65);
      let H2, sR;
      const ctl = UI.controls(fig);
      nocase(UI.slider(ctl, { label: 'advantage Â', min: -2, max: 2, step: 0.5, value: 2, fmt: (v) => fmtS(v, 1), onInput: (v) => { S.A = v; S.rho = start(); S.hold = 0; upd(); } }));
      nocase(UI.slider(ctl, { label: 'ε', min: 0.05, max: 0.4, step: 0.05, value: 0.2, fmt: (v) => fmt(v, 2), onInput: (v) => { S.eps = v; upd(); } }));
      sR = nocase(UI.slider(ctl, { label: 'ρ', min: X0, max: X1, step: 0.01, value: S.rho, fmt: (v) => fmt(v, 2), onInput: (v) => { S.rho = v; S.userT = H2 ? H2.t : 0; S.hold = 0; upd(); } }));
      UI.toggle(ctl, { label: 'gradient ascent', value: true, onChange: (v) => { S.run = v; if (v) { S.rho = start(); S.hold = 0; } } });
      const rL = nocase(UI.readout(ctl, 'L^CLIP', '—')), rD = nocase(UI.readout(ctl, '∂L/∂ρ', '—')), rG = UI.readout(ctl, 'regime', '—');
      let lastShown = -1;
      function upd() {
        const d = dL(S.rho);
        rL.set(fmt(Lf(S.rho), 3));
        rD.set(fmt(d, 1), d === 0 ? 'warn' : 'ok');
        rG.set(S.A === 0 ? 'Â = 0: nothing to learn' : d === 0 ? 'flat: incentive cut' : 'sloped: gradient active', d === 0 ? 'warn' : 'ok');
      }
      upd();
      H2 = V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const dd = clock(S, t);
        const A = { x: 52, y: 44, w: w - 72, h: h - 90 };
        const ys = [Lf(X0), Lf(X1), X0 * S.A, X1 * S.A, 0];
        let y0 = Math.min(...ys), y1 = Math.max(...ys);
        const pad = Math.max(0.3, (y1 - y0) * 0.12); y0 -= pad; y1 += pad;
        // pointer: drag
        const P = H2.pointer;
        if (P.down && P.x >= A.x && P.x <= A.x + A.w) { S.rho = V.clamp(X0 + ((P.x - A.x) / A.w) * (X1 - X0), X0, X1); S.userT = t; S.hold = 0; }
        // gradient ascent
        if (S.run && t - S.userT > 2.5) {
          const d = dL(S.rho);
          if (d !== 0) { S.rho = V.clamp(S.rho + Math.sign(d) * 0.32 * dd, X0, X1); if (S.rho <= X0 || S.rho >= X1) S.hold += dd; }
          else S.hold += dd;
          if (S.hold > 1.8) { S.rho = start(); S.hold = 0; }
        }
        if (Math.abs(S.rho - lastShown) > 0.004) { lastShown = S.rho; sR.set(S.rho.toFixed(2), true); upd(); }
        const step = (y1 - y0) > 3 ? 1 : 0.5;
        const yt = []; for (let v = Math.ceil(y0 / step) * step; v <= y1; v += step) yt.push(+v.toFixed(2));
        const F = frame(g, A, X0, X1, y0, y1, { xt: [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8], yt, xl: 'ratio ρ = π_θ / π_θold', yl: 'token objective', ylOff: 28 });
        // zero-gradient zones
        if (S.A > 0) hatch(g, F.X(1 + S.eps), A.y, A.x + A.w - F.X(1 + S.eps), A.h, V.alpha(C.d, 0.35));
        if (S.A < 0) hatch(g, A.x, A.y, F.X(1 - S.eps) - A.x, A.h, V.alpha(C.d, 0.35));
        [1 - S.eps, 1, 1 + S.eps].forEach((v, k) => { V.line(g, F.X(v), A.y, F.X(v), A.y + A.h, { color: V.alpha(k === 1 ? C.muted : C.e, 0.7), dash: [3, 4], lw: 1.1 }); });
        V.text(g, '1−ε', F.X(1 - S.eps), A.y - 4, { size: 11, font: 'mono', color: C.e, align: 'center' });
        V.text(g, '1+ε', F.X(1 + S.eps), A.y - 4, { size: 11, font: 'mono', color: C.e, align: 'center' });
        if (y0 < 0 && y1 > 0) V.line(g, A.x, F.Y(0), A.x + A.w, F.Y(0), { color: V.alpha(C.muted, 0.5), lw: 1 });
        curve(g, (r) => r * S.A, X0, X1, F.X, F.Y, { color: V.alpha(C.ink, 0.5), dash: [6, 5], lw: 1.6 });
        curve(g, (r) => V.clamp(r, 1 - S.eps, 1 + S.eps) * S.A, X0, X1, F.X, F.Y, { color: C.b, dash: [2, 4], lw: 2 });
        const main = S.A >= 0 ? C.a : C.c;
        curve(g, Lf, X0, X1, F.X, F.Y, { color: main, lw: 3.4, n: 200 });
        // points from the table
        if (S.A === 2 || S.A === -1) [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3].forEach((r) => V.dot(g, F.X(r), F.Y(Lf(r)), 2.6, V.alpha(C.ink, 0.7)));
        // ball
        const bx = F.X(S.rho), by = F.Y(Lf(S.rho)), d = dL(S.rho);
        V.glow(g, bx, by, 16, d === 0 ? C.d : main, 0.8);
        V.dot(g, bx, by, 6, d === 0 ? C.d : main);
        if (d !== 0) V.arrow(g, bx, by - 14, bx + Math.sign(d) * 34, by - 14, { color: main, lw: 2.2, head: 8 });
        else V.text(g, '∇ = 0', bx, by - 14, { size: 12, font: 'mono', weight: 700, color: C.d, align: 'center' });
        // legend
        const lx = A.x + 10, ly = S.A >= 0 ? A.y + 14 : A.y + A.h - 46;
        [[V.alpha(C.ink, 0.6), 'ρÂ (no clip)', [6, 5]], [C.b, 'clip(ρ)·Â', [2, 4]], [main, 'L^CLIP = min(…)', null]].forEach(([c, s, ds], k) => {
          V.line(g, lx, ly + k * 16, lx + 22, ly + k * 16, { color: c, lw: ds ? 2 : 3.2, dash: ds || undefined });
          V.text(g, s, lx + 28, ly + k * 16 + 4, { size: 11, font: 'mono', color: C.ink });
        });
        if (S.A !== 0) V.text(g, 'hatched = zero gradient', A.x + A.w - 4, S.A >= 0 ? A.y + A.h - 8 : A.y + 14, { size: 11, color: C.d, align: 'right' });
      });
    })();

    /* ======================================================= FIG 3 — value clipping */
    (function () {
      const fig = root.querySelector('#fig-valor');
      adaptH(fig, 390, 480);
      const CASES = { c1: { V: 1.6, R: 1.1 }, c2: { V: 0.7, R: 1.1 }, c3: { V: 1.3, R: 2.0 } };
      const S = { Vold: 1.0, Vn: 1.3, R: 2.0, eps: 0.2, run: true, ball: 1.0, hold: 0 };
      const X0 = 0, X1 = 2.6;
      const Lun = (v) => (v - S.R) * (v - S.R);
      const Lcl = (v) => { const c = S.Vold + V.clamp(v - S.Vold, -S.eps, S.eps) - S.R; return c * c; };
      const Lv = (v) => Math.max(Lun(v), Lcl(v));
      const dL = (v) => (Lun(v) >= Lcl(v) - 1e-12 ? 2 * (v - S.R) : 0);
      const isZero = (v) => dL(v) === 0;
      /** bisection: the point between a and b where the zero-gradient status flips */
      const edge = (a, b) => { const za = isZero(a); for (let q = 0; q < 18; q++) { const m = (a + b) / 2; if (isZero(m) === za) a = m; else b = m; } return (a + b) / 2; };
      let sV, sRh;
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Case', options: [{ v: 'c1', label: 'V 1.6 · target 1.1' }, { v: 'c2', label: 'V 0.7 · target 1.1' }, { v: 'c3', label: 'V 1.3 · target 2.0' }], value: 'c3', onChange: (v) => { const c = CASES[v]; S.Vn = c.V; S.R = c.R; S.ball = S.Vold; S.hold = 0; sV.set(c.V, true); sRh.set(c.R, true); upd(); } });
      sV = nocase(UI.slider(ctl, { label: 'V new (diamond)', min: 0.2, max: 2.4, step: 0.01, value: 1.3, fmt: (v) => fmt(v, 2), onInput: (v) => { S.Vn = v; upd(); } }));
      sRh = nocase(UI.slider(ctl, { label: 'target R̂', min: 0.2, max: 2.4, step: 0.05, value: 2.0, fmt: (v) => fmt(v, 2), onInput: (v) => { S.R = v; S.ball = S.Vold; S.hold = 0; upd(); } }));
      nocase(UI.slider(ctl, { label: 'ε̂', min: 0.1, max: 0.5, step: 0.05, value: 0.2, fmt: (v) => fmt(v, 2), onInput: (v) => { S.eps = v; S.ball = S.Vold; S.hold = 0; upd(); } }));
      UI.toggle(ctl, { label: 'gradient descent from V_old', value: true, onChange: (v) => { S.run = v; S.ball = S.Vold; S.hold = 0; } });
      const rL = nocase(UI.readout(ctl, 'loss at V new', '—')), rD = nocase(UI.readout(ctl, '∂L/∂V at V new', '—')), rG = UI.readout(ctl, 'chosen term', '—');
      function upd() {
        const d = dL(S.Vn);
        rL.set(fmt(Lv(S.Vn), 3));
        rD.set(fmt(d, 2), d === 0 ? 'warn' : 'ok');
        const inBand = Math.abs(S.Vn - S.Vold) <= S.eps + 1e-9;
        rG.set(inBand ? 'inside the band (equal)' : d === 0 ? 'clipped: zero gradient' : 'unclipped: pulls back', d === 0 ? 'warn' : 'ok');
      }
      upd();
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const dd = clock(S, t);
        if (S.run) {
          const d = dL(S.ball);
          if (Math.abs(d) > 0.004) S.ball = V.clamp(S.ball - 0.9 * d * dd, X0, X1); else S.hold += dd;
          if (S.hold > 1.8) { S.ball = S.Vold; S.hold = 0; }
        }
        const A = { x: 50, y: 46, w: w - 70, h: h - 92 };
        const ymax = Math.max(0.8, Lv(S.Vn) * 1.3, Lv(S.Vold) * 1.25);
        const F = frame(g, A, X0, X1, 0, ymax, { xt: [0, 0.5, 1, 1.5, 2, 2.5], yt: ymax > 1.6 ? [0, 0.5, 1, 1.5, 2] : ymax > 0.9 ? [0, 0.25, 0.5, 0.75, 1] : [0, 0.25, 0.5, 0.75], yf: (v) => fmt(v, 2), xl: 'critic prediction V(s_t)', yl: 'value loss', ylOff: 28 });
        g.save(); g.beginPath(); g.rect(A.x, A.y - 2, A.w, A.h + 2); g.clip();
        g.fillStyle = V.alpha(C.e, 0.14); g.fillRect(F.X(S.Vold - S.eps), A.y, F.X(S.Vold + S.eps) - F.X(S.Vold - S.eps), A.h);
        // stretches where the max picked the clipped term (zero gradient)
        // sweep once, find the contiguous intervals (at most 2) and hatch each one in a single pass
        {
          const NZ = 160, hc = V.alpha(C.d, 0.3);
          let zs = isZero(X0) ? X0 : null, pz = zs != null, pv = X0;
          for (let k = 1; k <= NZ; k++) {
            const v = X0 + ((X1 - X0) * k) / NZ, z = isZero(v);
            if (z !== pz) { const e = edge(pv, v); if (z) zs = e; else { hatch(g, F.X(zs), A.y, F.X(e) - F.X(zs), A.h, hc, 7); zs = null; } }
            pz = z; pv = v;
          }
          if (zs != null) hatch(g, F.X(zs), A.y, F.X(X1) - F.X(zs), A.h, hc, 7);
        }
        curve(g, Lun, X0, X1, F.X, F.Y, { color: V.alpha(C.ink, 0.5), dash: [6, 5], lw: 1.6 });
        curve(g, Lcl, X0, X1, F.X, F.Y, { color: C.b, dash: [2, 4], lw: 2 });
        curve(g, Lv, X0, X1, F.X, F.Y, { color: C.c, lw: 3.2, n: 260 });
        g.restore();
        V.text(g, 'V_old ± ε̂', F.X(S.Vold), A.y - 6, { size: 11, font: 'mono', color: C.e, align: 'center' });
        V.line(g, F.X(S.Vold), A.y + A.h, F.X(S.Vold), A.y + A.h - 10, { color: C.e, lw: 2 });
        // target
        const rx = F.X(S.R), ry = A.y + A.h;
        g.beginPath(); for (let q = 0; q < 10; q++) { const rr = q % 2 ? 3.5 : 8, a = -Math.PI / 2 + (q * Math.PI) / 5; const px = rx + rr * Math.cos(a), py = ry - 12 + rr * Math.sin(a); if (q) g.lineTo(px, py); else g.moveTo(px, py); } g.closePath(); g.fillStyle = C.d; g.fill();
        V.text(g, 'target R̂', rx, ry - 24, { size: 11, font: 'mono', color: C.d, align: 'center', weight: 600 });
        // diamond: the case's new V
        const mx = F.X(S.Vn), my = F.Y(Math.min(ymax, Lv(S.Vn)));
        g.beginPath(); g.moveTo(mx, my - 7); g.lineTo(mx + 7, my); g.lineTo(mx, my + 7); g.lineTo(mx - 7, my); g.closePath(); g.fillStyle = C.b; g.fill(); g.strokeStyle = C.ink; g.lineWidth = 1; g.stroke();
        // diamond label on the side opposite the parabola branch (otherwise the curve cuts through the text)
        const nearR = mx > A.x + A.w - 90, lblL = nearR || (S.Vn < S.R && dL(S.Vn) !== 0 && mx > 100);
        V.text(g, 'V new: ' + fmt(Lv(S.Vn), 2), lblL ? mx - 10 : mx + 10, my + (nearR ? -12 : 16), { size: 11, font: 'mono', color: C.b, weight: 700, align: lblL ? 'right' : 'left' });
        // ball: the critic over the epochs
        const d = dL(S.ball);
        const bx = F.X(S.ball), by = F.Y(Math.min(ymax, Lv(S.ball)));
        V.glow(g, bx, by, 15, d === 0 ? C.d : C.c, 0.8);
        V.dot(g, bx, by, 6, d === 0 ? C.d : C.c);
        if (Math.abs(d) > 0.004) V.arrow(g, bx, by - 14, bx - Math.sign(d) * 30, by - 14, { color: C.c, lw: 2.2, head: 8 });
        else { const lft = bx < A.x + 150; V.text(g, d === 0 ? '∇ = 0: stuck at edge' : 'minimum', lft ? bx + 8 : bx - 6, by - 16, { size: 11.5, font: 'mono', weight: 700, color: C.d, align: lft ? 'left' : 'right' }); }
        // legend: picks the corner where the fewest curve points and markers fall underneath it
        const LEG = [[V.alpha(C.ink, 0.6), '(V − R̂)²', [6, 5]], [C.b, 'clipped term', [2, 4]], [C.c, w < 560 ? 'max(…) = loss' : 'max(…) = loss used', null]];
        g.font = V.font(11, 'mono', 400);
        const legW = 28 + Math.max(g.measureText('zero gradient').width, ...LEG.map((it) => g.measureText(it[1]).width)) + 14, legH = 68;
        const pts = [[mx, my], [mx + 10, my + 14], [rx, ry - 12], [rx, ry - 24]];
        for (let k = 0; k <= 60; k++) { const v = X0 + ((X1 - X0) * k) / 60; [Lun(v), Lcl(v), Lv(v)].forEach((L) => { if (L <= ymax) pts.push([F.X(v), F.Y(L)]); }); }
        const cands = [[A.x + 10, A.y + 14], [A.x + A.w - legW, A.y + 14], [A.x + (A.w - legW) / 2 + 6, A.y + 14], [A.x + 10, A.y + A.h - legH + 2], [A.x + A.w - legW, A.y + A.h - legH + 2], [A.x + 10, A.y + (A.h - legH) / 2 + 10], [A.x + A.w - legW, A.y + (A.h - legH) / 2 + 10], [A.x + (A.w - legW) / 2 + 6, A.y + (A.h - legH) / 2 + 10]];
        for (let k = 0; k <= 120; k++) { const v = X0 + ((X1 - X0) * k) / 120, L = Lv(v); if (L <= ymax) pts.push([F.X(v), F.Y(L)]); } // the thick line counts twice
        const hits = cands.map(([cx0, cy0]) => pts.filter(([px, py]) => px >= cx0 - 12 && px <= cx0 + legW && py >= cy0 - 16 && py <= cy0 + legH - 4).length);
        let bi = 0; hits.forEach((n, k) => { if (n < hits[bi]) bi = k; });
        if (S.leg != null && hits[S.leg] <= hits[bi]) bi = S.leg; // no jumping between corners for no reason
        S.leg = bi;
        const lx = cands[bi][0], ly = cands[bi][1];
        V.box(g, lx - 6, ly - 10, legW, legH, { r: 6, fill: V.alpha(C.bg || C.panel, 0.92), stroke: V.alpha(C.muted, 0.25), lw: 1 });
        LEG.forEach(([c, s, ds], k) => {
          V.line(g, lx, ly + k * 16, lx + 22, ly + k * 16, { color: c, lw: ds ? 2 : 3, dash: ds || undefined });
          V.text(g, s, lx + 28, ly + k * 16 + 4, { size: 11, font: 'mono', color: C.ink });
        });
        hatch(g, lx, ly + 41, 22, 9, V.alpha(C.d, 0.6), 4);
        V.text(g, 'zero gradient', lx + 28, ly + 50, { size: 11, font: 'mono', color: C.d });
      });
    })();

    /* ======================================================= FIG 4 — InstructGPT pipeline */
    (function () {
      const fig = root.querySelector('#fig-pipeline');
      adaptH(fig, 400, 700);
      const steps = [
        { t: 'Stage 1 · SFT on demonstrations', d: 'About 40 labelers write demonstration answers. There are 12,725 training prompts (11,295 from labelers + 1,430 from API customers). 16 epochs with a cosine schedule; the checkpoint is chosen by the <b>RM score</b>, because the validation loss overfits after 1 epoch.' },
        { t: 'Stage 2 · rankings become pairs', d: 'For each prompt, \\(K = 4\\) to \\(9\\) responses are ranked by a labeler. Each ranking becomes \\(\\binom K2\\) pairs, from 6 to 36. There are 33,207 training prompts. Ties are discarded.' },
        { t: 'Stage 2 · the RM learns Bradley–Terry', d: 'A 6B RM with loss \\(-\\log\\sigma(r_w - r_l)\\), averaged over the pairs. All the pairs from one prompt go into the <b>same</b> batch element: \\(K\\) forwards, not \\(\\binom K2\\), and no overfitting from reusing each response \\(K-1\\) times. In the example (illustrative), scores 2.0 / 1.0 / 0.5 / −0.5 give a loss of 0.2637.' },
        { t: 'Stage 3 · PPO against the RM', d: '31,144 prompts from the API only, with no human label. The policy (initialized from the SFT) generates, the RM gives the score at the end, the per-token KL to the SFT becomes a penalty (\\(\\beta = 0.02\\)), and the critic, initialized from the RM, estimates values. 256k episodes, batch 512, 1 epoch.' },
        { t: 'PPO-ptx · the pretraining anchor', d: 'An extra log-likelihood gradient on pretraining text, with weight \\(\\gamma = 27.8\\) and 8× more examples than RL episodes. Without it, SQuADv2, DROP and translation regressed (the alignment tax).' },
        { t: 'Result', d: 'The outputs of InstructGPT 1.3B were preferred over those of GPT-3 175B. PPO-ptx for the 175B cost 60 petaflop/s-days, versus 3,640 for the pretraining of GPT-3.' },
      ];
      let H = null;
      const st = UI.stepper(fig, steps, { autoplay: 6500, onStep: () => H && H.redraw() });
      const rng = V.rng(55);
      const books = Array.from({ length: 6 }, () => rng());
      function station(g, R, title, sub, on, C) {
        g.globalAlpha = on ? 1 : 0.38;
        V.box(g, R.x, R.y, R.w, R.h, { r: 12, fill: V.alpha(on ? C.a : C.muted, on ? 0.07 : 0.05), stroke: on ? V.alpha(C.a, 0.7) : V.alpha(C.muted, 0.35), lw: on ? 1.8 : 1.1 });
        V.text(g, title, R.x + 12, R.y + 20, { size: 13, weight: 700, color: C.ink });
        V.text(g, sub, R.x + 12, R.y + R.h - 10, { size: 11, font: 'mono', color: C.muted, maxW: R.w - 20 });
      }
      H = V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const i = st.i, el = st.elapsed();
        const wide = w >= 640;
        let Rr;
        if (wide) { const sw = (w - 24 - 2 * 26) / 3; Rr = [0, 1, 2].map((k) => ({ x: 12 + k * (sw + 26), y: 10, w: sw, h: h - 20 })); }
        else { const sh = (h - 20 - 2 * 22) / 3; Rr = [0, 1, 2].map((k) => ({ x: 8, y: 10 + k * (sh + 22), w: w - 16, h: sh })); }
        const on = [i === 0, i === 1 || i === 2, i >= 3 && i <= 4];
        // arrows between stations
        for (let k = 0; k < 2; k++) {
          const a = Rr[k], b = Rr[k + 1];
          if (wide) V.arrow(g, a.x + a.w + 3, a.y + a.h / 2, b.x - 3, b.y + b.h / 2, { color: V.alpha(C.muted, 0.7), lw: 1.8 });
          else V.arrow(g, a.x + a.w / 2, a.y + a.h + 2, b.x + b.w / 2, b.y - 2, { color: V.alpha(C.muted, 0.7), lw: 1.8 });
        }
        // ---- station 1: SFT
        (function (R) {
          station(g, R, '1 · SFT', '12,725 prompts · 16 epochs', on[0] || i === 5, C);
          g.globalAlpha = on[0] || i === 5 ? 1 : 0.38;
          const mx = R.x + R.w * 0.74, my = R.y + R.h * 0.48, ms = Math.min(46, R.h * 0.3);
          for (let q = 0; q < 3; q++) {
            const ph = ((t * 0.35 + q / 3) % 1);
            const cx = V.lerp(R.x + 16, mx - ms / 2 - 30, ph), cy = my - 26 + q * 18 - ph * 6;
            const cw = Math.min(70, R.w * 0.26);
            g.globalAlpha = (on[0] || i === 5 ? 1 : 0.38) * (ph < 0.85 ? 1 : (1 - ph) / 0.15);
            V.box(g, cx, cy, cw, 14, { r: 3, fill: V.alpha(C.panel || C.bg, 0.9), stroke: V.alpha(C.muted, 0.5), lw: 1 });
            g.fillStyle = V.alpha(C.muted, 0.6); g.fillRect(cx + 4, cy + 4, cw * 0.35, 2.5);
            g.fillStyle = V.alpha(C.a, 0.9); g.fillRect(cx + 4, cy + 8.5, cw * 0.7, 2.5);
          }
          g.globalAlpha = on[0] || i === 5 ? 1 : 0.38;
          face(g, mx, my, ms, C.a);
          V.text(g, 'base model', mx, my + ms / 2 + 14, { size: 11, align: 'center', color: C.muted });
          V.text(g, 'demonstrations', R.x + 16, my - 36, { size: 11, color: C.muted });
          g.globalAlpha = 1;
        })(Rr[0]);
        // ---- station 2: RM
        (function (R) {
          const act = on[1] || i === 5;
          station(g, R, '2 · Reward model', '33,207 prompts · K = 4–9 → C(K,2) pairs', act, C);
          g.globalAlpha = act ? 1 : 0.38;
          const labels = ['A', 'B', 'C', 'D'], scores = [2.0, 1.0, 0.5, -0.5];
          const mixed = [2, 0, 3, 1];
          const sortT = i === 1 ? V.ease.inOut(V.clamp(el / 1.0, 0, 1)) : i >= 2 ? 1 : 0;
          const ch = Math.min(22, (R.h - 76) / 4.8), cw = 34;
          const blockH = 4 * (ch + 5);
          const x0 = R.x + 16, y0 = R.y + Math.max(34, (R.h - blockH) / 2 - 6);
          V.text(g, 'ranking', x0, y0 - 8, { size: 11, color: C.muted });
          const pos = labels.map((_, k) => V.lerp(mixed.indexOf(k), k, sortT));
          labels.forEach((l, k) => {
            const y = y0 + pos[k] * (ch + 5);
            const dx = Math.sign(k - mixed.indexOf(k)) * Math.sin(Math.PI * sortT) * 22;
            chip(g, x0 + dx, y, cw, ch, l, { fill: V.alpha(C.b, 0.25), stroke: C.b, size: 12, weight: 700 });
            if (i >= 2) V.text(g, 'r=' + fmt(scores[k], 1), x0 + cw + 6, y + ch / 2 + 4, { size: 11, font: 'mono', color: C.d });
          });
          if (sortT > 0.5) V.text(g, 'A ≻ B ≻ C ≻ D', x0, y0 + blockH + 12, { size: 11, font: 'mono', color: C.ink, alpha: Math.min(1, (sortT - 0.5) * 2) });
          // pairs
          const pairs = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]];
          const nP = i === 1 ? Math.floor(V.clamp((el - 1.0) / 0.3, 0, 6)) : i >= 2 ? 6 : 0;
          const px = R.x + R.w * (wide ? 0.44 : 0.4), pStep = Math.min(16, (blockH + 10) / 6);
          const py0 = y0 + blockH / 2 - 3 * pStep;
          for (let q = 0; q < nP; q++) {
            const [a, b] = pairs[q];
            const y = py0 + q * pStep;
            chip(g, px, y, 40, pStep - 3, labels[a] + '≻' + labels[b], { size: 11, fill: V.alpha(C.a, 0.14), stroke: V.alpha(C.a, 0.6), r: 4 });
          }
          if (nP) V.text(g, nP + (nP > 1 ? ' pairs' : ' pair'), px, py0 - 6, { size: 11, color: C.muted });
          const mx = R.x + R.w - Math.min(46, R.h * 0.3) / 2 - 16, my = R.y + R.h * 0.5, ms = Math.min(42, R.h * 0.28);
          face(g, mx, my, ms, C.d, { mood: i === 2 ? 'o' : undefined });
          V.text(g, '6B RM', mx, my + ms / 2 + 14, { size: 11, align: 'center', color: C.muted });
          if (i === 2) {
            const ph = (t * 0.8) % 1;
            V.dot(g, V.lerp(px + 40, mx - ms / 2, ph), my, 3, C.d);
            V.text(g, 'loss 0.2637', mx - 8, my + ms / 2 + 29, { size: 11, font: 'mono', color: C.d, align: 'center', weight: 700 });
          }
          g.globalAlpha = 1;
        })(Rr[1]);
        // ---- station 3: PPO
        (function (R) {
          const act = on[2] || i === 5;
          station(g, R, '3 · PPO', '31,144 prompts · 256k episodes · β = 0.02', act, C);
          g.globalAlpha = act ? 1 : 0.38;
          const ms = Math.min(40, R.h * 0.26);
          const pxx = R.x + R.w * 0.3, pyy = R.y + R.h * (wide ? 0.42 : 0.46);
          const rxx = R.x + R.w * 0.78, ryy = pyy;
          // tokens traveling from the policy to the RM
          for (let q = 0; q < 4; q++) {
            const ph = (t * 0.5 + q / 4) % 1;
            chip(g, V.lerp(pxx + ms / 2 + 4, rxx - ms / 2 - 18, ph), pyy - 26, 14, 12, null, { fill: V.alpha(C.a, 0.6), stroke: C.a, r: 3 });
          }
          // score coming back
          const ph2 = (t * 0.5) % 1;
          V.text(g, '+r', V.lerp(rxx - ms / 2, pxx + ms / 2, ph2), pyy + 24, { size: 12, font: 'mono', weight: 700, color: C.d, align: 'center' });
          face(g, pxx, pyy, ms, C.a, { mood: 'o' });
          V.text(g, 'policy', pxx, pyy - ms / 2 - 8, { size: 11, align: 'center', color: C.ink, weight: 600 });
          face(g, rxx, ryy, ms * 0.9, C.d);
          V.text(g, 'RM', rxx, ryy - ms / 2 - 8, { size: 11, align: 'center', color: C.ink, weight: 600 });
          // ghost reference + leash
          const fx = pxx - Math.min(60, R.w * 0.18), fy = pyy + ms * 1.25;
          face(g, fx, fy, ms * 0.8, C.b, { fill: V.alpha(C.b, 0.12), ghost: true, dash: true });
          V.text(g, 'SFT (ref)', fx, fy + ms * 0.4 + 13, { size: 11, align: 'center', color: C.b });
          g.beginPath();
          for (let k = 0; k <= 20; k++) { const u = k / 20; const x = V.lerp(fx + ms * 0.3, pxx - ms * 0.2, u), y = V.lerp(fy - ms * 0.2, pyy + ms * 0.45, u) + Math.sin(u * Math.PI * 4 + t * 3) * 3; if (k) g.lineTo(x, y); else g.moveTo(x, y); }
          g.strokeStyle = C.b; g.lineWidth = 1.6; g.stroke();
          V.text(g, 'KL', (fx + pxx) / 2 + 12, (fy + pyy) / 2 + 12, { size: 11, font: 'mono', color: C.b, weight: 700 });
          // critic
          const cx = rxx, cy = fy;
          face(g, cx, cy, ms * 0.7, C.c);
          V.text(g, 'critic V (from RM)', cx, cy + ms * 0.35 + 13, { size: 11, align: 'center', color: C.c, maxW: R.w * 0.45 });
          // ptx
          if (i === 4) {
            for (let q = 0; q < 5; q++) {
              const ph = (t * 0.45 + books[q]) % 1;
              const x = V.lerp(R.x + 8, pxx - ms / 2 - 6, ph), y = pyy - 8 + Math.sin(q * 2.1) * 10;
              chip(g, x, y, 12, 9, null, { fill: V.alpha(C.e, 0.7), stroke: C.e, r: 2 });
            }
            V.text(g, 'ptx · γ = 27.8', R.x + 12, R.y + 40, { size: 11.5, font: 'mono', weight: 700, color: C.e });
          }
          g.globalAlpha = 1;
        })(Rr[2]);
        // ---- result
        if (i === 5) {
          const a = V.ease.out(V.clamp(el / 0.8, 0, 1));
          const cw = Math.min(480, w - 20), chh = Math.min(190, h * 0.5);
          const x = (w - cw) / 2, y = (h - chh) / 2;
          g.globalAlpha = a;
          // opaque background: the station underneath does not show through the card
          V.box(g, x, y, cw, chh, { r: 12, fill: C.bg, stroke: C.a, lw: 2 });
          V.box(g, x, y, cw, chh, { r: 12, fill: V.alpha(C.a, 0.06) });
          V.text(g, 'Result', x + 14, y + 22, { size: 13, weight: 700, color: C.ink });
          const rows = [
            ['InstructGPT 1.3B', 1.3 / 175, C.a, 'preferred ✓'],
            ['GPT-3 175B', 1, C.muted, '100× larger'],
            ['PPO-ptx 175B', 60 / 3640, C.c, '60 pf/s-days'],
            ['GPT-3 pretraining', 1, C.muted, '3,640 pf/s-days'],
          ];
          // measure labels and tags so nothing touches: label | bar | tag
          g.font = V.font(11, 'mono', 400);
          const labW = Math.max(...rows.map((r) => g.measureText(r[0]).width));
          g.font = V.font(11, 'mono', 600);
          const tagW = Math.max(...rows.map((r) => g.measureText(r[3]).width));
          const bx = x + 14, bl = bx + labW + 10;
          const bw = Math.max(30, x + cw - 12 - tagW - 8 - bl);
          const stepY = (chh - 52) / 4;
          rows.forEach(([l, v, c, tag], k) => {
            const yy = y + 40 + k * stepY + (k >= 2 ? 4 : 0);
            V.text(g, l, bx, yy + 10, { size: 11, font: 'mono', color: C.ink });
            const ww = Math.max(3, bw * v * a);
            V.box(g, bl, yy, ww, 13, { r: 3, fill: V.alpha(c, 0.8) });
            V.text(g, tag, Math.min(bl + ww + 6, x + cw - tagW - 12), yy + 10, { size: 11, font: 'mono', color: c, weight: 600 });
          });
          // separate the two comparisons (quality | cost)
          const sy = y + 40 + 2 * stepY - 4;
          V.line(g, x + 14, sy, x + cw - 14, sy, { color: V.alpha(C.muted, 0.3), lw: 1, dash: [3, 4] });
          g.globalAlpha = 1;
        }
      });
    })();

    /* ======================================================= FIG 5 — optimal policy */
    (function () {
      const fig = root.querySelector('#fig-otima');
      adaptH(fig, 430, 640);
      const LAB = ['helpful, correct', 'correct, terse', 'verbose', 'wrong, confident', 'refusal', 'exploit'];
      const P = [0.30, 0.35, 0.15, 0.12, 0.07, 0.01];
      const R = [1.0, 0.6, 1.2, -0.5, -1.0, 2.5];
      const G = [1.0, 0.6, 0.5, -0.8, -0.5, -1.5];
      const piStar = (b) => { const ww = P.map((p, i) => p * Math.exp(R[i] / b)); const Z = ww.reduce((s, x) => s + x, 0); return ww.map((x) => x / Z); };
      const stats = (b) => { const ps = piStar(b); return { ps, Er: ps.reduce((s, p, i) => s + p * R[i], 0), Eg: ps.reduce((s, p, i) => s + p * G[i], 0), KL: ps.reduce((s, p, i) => s + (p > 0 ? p * Math.log(p / P[i]) : 0), 0) }; };
      const GRID = Array.from({ length: 161 }, (_, k) => { const lb = -1 + (2 * k) / 160; const s = stats(Math.pow(10, lb)); return { lb, x: Math.sqrt(Math.max(0, s.KL)), Er: s.Er, Eg: s.Eg }; });
      const peak = GRID.reduce((a, b) => (b.Eg > a.Eg ? b : a));
      const S = { lb: 0, auto: true, show: P.slice() };
      const ctl = UI.controls(fig);
      let tog;
      const sB = nocase(UI.slider(ctl, { label: 'β', min: -1, max: 1, step: 0.01, value: 0, fmt: (v) => fmt(Math.pow(10, v), 2), onInput: (v) => { S.lb = v; if (S.auto) tog.set(false); upd(); } }));
      tog = UI.toggle(ctl, { label: 'sweep β automatically', value: true, onChange: (v) => { S.auto = v; } });
      const rEr = UI.readout(ctl, 'RM score', '—'), rEg = UI.readout(ctl, 'real quality', '—'), rKL = nocase(UI.readout(ctl, 'KL(π*‖π_ref)', '—')), rX = UI.readout(ctl, 'mass on exploit', '—');
      let lastLb = 99;
      function upd() {
        const s = stats(Math.pow(10, S.lb));
        rEr.set(fmt(s.Er, 2)); rEg.set(fmt(s.Eg, 2), s.Eg < 0 ? 'bad' : s.Eg > 0.5 ? 'ok' : 'warn');
        rKL.set(fmt(s.KL, 3) + ' nat'); rX.set(fmt(s.ps[5] * 100, 1) + '%', s.ps[5] > 0.3 ? 'bad' : s.ps[5] > 0.08 ? 'warn' : 'ok');
      }
      upd();
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const dd = clock(S, t);
        if (S.auto) { S.lb = 1 - 2 * V.ease.inOut(V.pingpong(t, 16)); }
        if (Math.abs(S.lb - lastLb) > 0.004) { lastLb = S.lb; sB.set(S.lb.toFixed(2), true); upd(); }
        const s = stats(Math.pow(10, S.lb));
        S.show = S.show.map((v, j) => v + (s.ps[j] - v) * Math.min(1, dd * 8));
        const wide = w >= 680;
        const L = wide ? { x: 12, y: 26, w: w * 0.52 - 18, h: h - 92 } : { x: 8, y: 26, w: w - 16, h: h * 0.52 - 80 };
        const Rg = wide ? { x: w * 0.52 + 52, y: 26, w: w * 0.48 - 70, h: h - 72 } : { x: 46, y: h * 0.52 + 36, w: w - 62, h: h * 0.48 - 80 };
        // ---- bars
        V.text(g, 'π_ref (outline) and π* ∝ π_ref·e^(r/β) (bars) · β = ' + fmt(Math.pow(10, S.lb), 2), L.x, L.y - 10, { size: 11.5, color: C.muted, weight: 600, maxW: L.w });
        const n = 6, cw = L.w / n, bw = Math.min(40, cw * 0.55);
        const base = L.y + L.h;
        V.line(g, L.x, base, L.x + L.w, base, { color: V.alpha(C.muted, 0.5), lw: 1 });
        for (let j = 0; j < n; j++) {
          const cx = L.x + cw * (j + 0.5);
          const col = j === 5 ? C.c : j === 2 ? C.d : C.a;
          const hs = S.show[j] * L.h, ho = P[j] * L.h;
          V.box(g, cx - bw / 2, base - hs, bw, Math.max(1, hs), { r: 3, fill: V.alpha(col, 0.75), stroke: col, lw: 1 });
          g.setLineDash([4, 3]); V.rr(g, cx - bw / 2 - 3, base - ho, bw + 6, Math.max(1, ho), 3); g.strokeStyle = V.alpha(C.ink, 0.55); g.lineWidth = 1.1; g.stroke(); g.setLineDash([]);
          V.text(g, fmt(S.show[j], 2), cx, base - Math.max(hs, ho) - 5, { size: 11, font: 'mono', align: 'center', color: C.ink });
          V.wrap(g, LAB[j], cx, base + 14, cw - 4, { size: 11, align: 'center', color: j === 5 ? C.c : C.ink, weight: j === 5 ? 700 : 500, maxLines: 2, lineH: 13 });
          const tight = cw < 76;
          V.text(g, (tight ? 'r ' : 'RM ') + fmtS(R[j], 1), cx, base + 44, { size: 11, font: 'mono', align: 'center', color: C.b });
          V.text(g, (tight ? 'g ' : 'real ') + fmtS(G[j], 1), cx, base + 58, { size: 11, font: 'mono', align: 'center', color: C.a });
        }
        // ---- proxy vs real curve
        const xmax = Math.ceil(GRID[0].x * 10) / 10 + 0.1;
        const F = frame(g, Rg, 0, xmax, -1.6, 2.6, { xt: [0, 0.5, 1, 1.5, 2], yt: [-1, 0, 1, 2], yf: (v) => fmt(v, 0), xl: '√KL(π*‖π_ref)', yl: 'mean under π*' });
        V.line(g, Rg.x, F.Y(0), Rg.x + Rg.w, F.Y(0), { color: V.alpha(C.muted, 0.4), lw: 1 });
        const path = (key, col) => { g.beginPath(); GRID.forEach((p, k) => { const x = F.X(p.x), y = F.Y(p[key]); if (k) g.lineTo(x, y); else g.moveTo(x, y); }); g.strokeStyle = col; g.lineWidth = 2.6; g.stroke(); };
        path('Er', C.b); path('Eg', C.a);
        const cx = F.X(Math.sqrt(Math.max(0, s.KL)));
        V.line(g, cx, Rg.y, cx, Rg.y + Rg.h, { color: V.alpha(C.ink, 0.35), dash: [3, 4], lw: 1 });
        V.glow(g, cx, F.Y(s.Er), 12, C.b, 0.7); V.dot(g, cx, F.Y(s.Er), 5, C.b);
        V.glow(g, cx, F.Y(s.Eg), 12, C.a, 0.7); V.dot(g, cx, F.Y(s.Eg), 5, C.a);
        const pxp = F.X(peak.x), pyp = F.Y(peak.Eg);
        V.text(g, '★ real peak', pxp + 4, pyp - 8, { size: 11, color: C.a, weight: 700 });
        [[C.b, 'RM score (proxy)'], [C.a, 'real quality']].forEach(([c, s], k) => {
          const yy = Rg.y + 12 + k * 16;
          V.line(g, Rg.x + 8, yy - 4, Rg.x + 28, yy - 4, { color: c, lw: 2.6 });
          V.text(g, s, Rg.x + 34, yy, { size: 11, color: c, weight: 600 });
        });
        V.text(g, 'β: large → small', Rg.x + Rg.w, Rg.y - 9, { size: 11, color: C.muted, align: 'right' });
      });
    })();

    /* ======================================================= FIG 6 — β control */
    (function () {
      const fig = root.querySelector('#fig-kl-ctrl');
      adaptH(fig, 400, 520);
      const PULL = [0.45, 0.8, 1.3], NIT = 300;
      const S = { mode: 'zieg', tgt: 6, b0: 0.1 };
      let SIM = null;
      function simulate() {
        SIM = PULL.map((a, s) => {
          const r = V.rng(17 + s * 31); let kl = 0, b = S.b0; const KL = [], B = [];
          for (let k = 0; k < NIT; k++) {
            KL.push(kl); B.push(b);
            kl = Math.max(0, kl + 0.06 * (a / b - kl) + 0.15 * r.normal());
            if (S.mode === 'zieg') { const e = V.clamp((kl - S.tgt) / S.tgt, -0.2, 0.2); b = b * (1 + 0.1 * e); }
            else if (S.mode === 'ppo') { if (kl > 1.5 * S.tgt) b *= 2; else if (kl < S.tgt / 1.5) b /= 2; b = V.clamp(b, 1e-3, 10); }
          }
          return { KL, B };
        });
        upd();
      }
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Control', options: [{ v: 'fixo', label: 'fixed β' }, { v: 'zieg', label: 'Ziegler (Kβ = 0.1)' }, { v: 'ppo', label: 'double / halve' }], value: 'zieg', onChange: (v) => { S.mode = v; simulate(); } });
      nocase(UI.slider(ctl, { label: 'target KL (nats)', min: 3, max: 10, step: 0.5, value: 6, fmt: (v) => fmt(v, 1), onInput: (v) => { S.tgt = v; simulate(); } }));
      nocase(UI.slider(ctl, { label: 'initial β', min: 0.02, max: 0.3, step: 0.01, value: 0.1, fmt: (v) => fmt(v, 2), onInput: (v) => { S.b0 = v; simulate(); } }));
      const rK = UI.readout(ctl, 'final KL (3 seeds)', '—'), rB = nocase(UI.readout(ctl, 'final β', '—'));
      function upd() {
        if (!SIM) return;
        const kls = SIM.map((s) => s.KL[NIT - 1]);
        const spread = Math.max(...kls) - Math.min(...kls);
        rK.set(kls.map((x) => fmt(x, 1)).join(' · '), spread > 4 ? 'bad' : spread > 1.5 ? 'warn' : 'ok');
        rB.set(SIM.map((s) => fmt(s.B[NIT - 1], 3)).join(' · '));
      }
      simulate();
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const cols = [C.a, C.b, C.c];
        const ph = (t % 10) / 7.5, nShow = Math.max(2, Math.floor(Math.min(1, ph) * NIT));
        // usable height leaves 42 px at the bottom: ticks at +14 and the 'iterations' label at +30 fit in the canvas
        const A1 = { x: 52, y: 24, w: w - 70, h: (h - 110) * 0.55 };
        const A2 = { x: 52, y: A1.y + A1.h + 44, w: w - 70, h: (h - 110) * 0.45 };
        const kmax = Math.max(16, ...SIM.map((s) => Math.max(...s.KL))) * 1.05;
        const F1 = frame(g, A1, 0, NIT, 0, kmax, { xt: [0, 100, 200, 300], xf: (v) => String(v), yt: kmax > 30 ? [0, 10, 20, 30] : [0, 5, 10, 15], yf: (v) => String(v), yl: 'KL(π‖π_ref) in nats' });
        V.line(g, A1.x, F1.Y(S.tgt), A1.x + A1.w, F1.Y(S.tgt), { color: C.d, dash: [6, 5], lw: 1.6 });
        V.text(g, 'target ' + fmt(S.tgt, 1), A1.x + A1.w - 4, F1.Y(S.tgt) - 5, { size: 11, font: 'mono', color: C.d, align: 'right' });
        const lmin = -3, lmax = 1;
        const F2 = frame(g, A2, 0, NIT, lmin, lmax, { xt: [0, 100, 200, 300], xf: (v) => String(v), yt: [-3, -2, -1, 0, 1], yf: (v) => (v === -3 ? '0.001' : v === -2 ? '0.01' : v === -1 ? '0.1' : v === 0 ? '1' : '10'), xl: 'iterations', yl: 'β (log scale)' });
        if (S.mode === 'ppo') { g.fillStyle = V.alpha(C.d, 0.08); g.fillRect(A1.x, F1.Y(1.5 * S.tgt), A1.w, F1.Y(S.tgt / 1.5) - F1.Y(1.5 * S.tgt)); V.text(g, 'dead zone [target/1.5, 1.5·target]', A1.x + A1.w - 4, F1.Y(1.5 * S.tgt) - 5, { size: 11, color: C.d, align: 'right' }); }
        SIM.forEach((s, k) => {
          // ghost of the whole trajectory
          g.globalAlpha = 0.22;
          g.beginPath(); s.KL.forEach((v, q) => { const x = F1.X(q), y = F1.Y(v); if (q) g.lineTo(x, y); else g.moveTo(x, y); }); g.strokeStyle = cols[k]; g.lineWidth = 1.4; g.stroke();
          g.beginPath(); s.B.forEach((v, q) => { const x = F2.X(q), y = F2.Y(V.clamp(Math.log10(v), lmin, lmax)); if (q) g.lineTo(x, y); else g.moveTo(x, y); }); g.stroke();
          g.globalAlpha = 1;
          g.beginPath();
          for (let q = 0; q < nShow; q++) { const x = F1.X(q), y = F1.Y(s.KL[q]); if (q) g.lineTo(x, y); else g.moveTo(x, y); }
          g.strokeStyle = cols[k]; g.lineWidth = 2; g.stroke();
          g.beginPath();
          for (let q = 0; q < nShow; q++) { const x = F2.X(q), y = F2.Y(V.clamp(Math.log10(s.B[q]), lmin, lmax)); if (q) g.lineTo(x, y); else g.moveTo(x, y); }
          g.strokeStyle = cols[k]; g.lineWidth = 2; g.stroke();
          const q = nShow - 1;
          V.dot(g, F1.X(q), F1.Y(s.KL[q]), 3.5, cols[k]);
          V.dot(g, F2.X(q), F2.Y(V.clamp(Math.log10(s.B[q]), lmin, lmax)), 3.5, cols[k]);
        });
        const names = ['seed 1 · weak pull', 'seed 2 · medium', 'seed 3 · strong pull'];
        names.forEach((s, k) => { V.dot(g, A1.x + 12, A1.y + 10 + k * 15, 4, cols[k]); V.text(g, s, A1.x + 22, A1.y + 14 + k * 15, { size: 11, color: C.ink }); });
      });
    })();

    /* ======================================================= FIG 7 — iteration step by step */
    (function () {
      const fig = root.querySelector('#fig-iteracao');
      adaptH(fig, 540, 680);
      const TOKS = ['The', 'sky', 'is', 'blue', '<eos>'];
      const ROWS = [
        { l: 'log π_old', v: [-1.20, -0.50, -2.10, -0.30, -0.05], d: 2, from: 3, sc: 2.2 },
        { l: 'log π_ref', v: [-1.40, -0.55, -1.60, -0.45, -0.10], d: 2, from: 3, sc: 2.2 },
        { l: 'V_old', v: [0.90, 1.10, 1.00, 1.50, 1.80], d: 2, from: 3, sc: 2.2 },
        { l: 'KL_t (k₁)', v: [0.20, 0.05, -0.50, 0.15, 0.05], d: 2, from: 4, sc: 0.6 },
        { l: 'r_t', v: [-0.0100, -0.0025, 0.0250, -0.0075, 1.9975], d: 4, from: 4, sc: 2.0 },
        { l: 'δ_t', v: [0.1900, -0.1025, 0.5250, 0.2925, 0.1975], d: 4, from: 5, sc: 0.8 },
        { l: 'Â_t (λ=0.95)', v: [0.9781, 0.8296, 0.9811, 0.4801, 0.1975], d: 4, from: 5, sc: 1.2, rtl: true },
        { l: 'R̂_t (target)', v: [1.8781, 1.9296, 1.9811, 1.9801, 1.9975], d: 4, from: 5, sc: 2.2, rtl: true },
        { l: 'Â whitened', v: [0.93, 0.44, 0.94, -0.69, -1.61], d: 2, from: 6, sc: 1.8 },
        { l: 'ρ_t (epoch)', v: [1.214, 1.187, 1.226, 0.806, 0.781], d: 3, from: 7, sc: 0.3, rho: true },
      ];
      const ACT = [[], ['motor', 'pol'], ['rm'], ['ref', 'pol', 'crit'], [], ['crit'], [], ['pol', 'crit'], ['pol', 'motor']];
      const steps = [
        { t: 'Prompts', d: 'Draw \\(B\\) prompts from the RL set (InstructGPT and N+: \\(B = 512\\)). The figure follows <b>one</b> 5-token response.' },
        { t: 'Rollouts in the inference engine', d: 'vLLM or SGLang generates with the weights of \\(\\pi_{\\theta_{\\text{old}}}\\), with continuous batching and KV cache (temperature 1 in InstructGPT, 0.7 in N+). N+ trick: generate up to the max and truncate at EOS.' },
        { t: 'The RM scores at the EOS', d: 'One RM forward; the score is read only at the EOS token: 2.0 in this example. A response without EOS gets −1 (N+).' },
        { t: 'Three forwards per response', d: '\\(\\log\\pi_{\\text{ref}}\\) (frozen reference), \\(\\log\\pi_{\\text{old}}\\) (recomputed in the <b>training</b> engine, not copied from vLLM) and \\(V_{\\text{old}}\\) (critic). One forward returns log-probs or values for <b>all</b> positions at once, because the whole response is already known (teacher forcing).' },
        { t: 'Per-token reward', d: '\\(r_t = -\\beta(\\log\\pi_{\\text{old}} - \\log\\pi_{\\text{ref}})\\), and the last token also gets the RM score. With \\(\\beta = 0.05\\): from −0.0100 to +1.9975. Token 3 has a negative KL (the \\(k_1\\) estimator can be negative) and earns a bonus.' },
        { t: 'GAE from back to front', d: '\\(\\delta_t = r_t + V(s_{t+1}) - V(s_t)\\) and \\(\\hat A_t = \\delta_t + \\gamma\\lambda\\hat A_{t+1}\\), with \\(\\gamma = 1\\) and \\(\\lambda = 0.95\\), sweeping from right to left. Critic target: \\(\\hat R_t = \\hat A_t + V_{\\text{old}}\\).' },
        { t: 'Whiten the advantages', d: 'Mean 0 and std 1 over the batch (here, over the 5 tokens: mean 0.693 and std 0.308). What remains is the relative signal: tokens 4 and 5 were “worse than the others”. The EOS ends up negative because the critic already expected 1.8: the surprise \\(\\delta_5 = 0.1975\\) is small. In a real batch, the mean and std come from all the tokens of the 512 responses.' },
        { t: 'K epochs × minibatches', d: 'In each minibatch: \\(\\rho_t\\), \\(L^{CLIP}\\), the clipped value loss and one optimizer step with norm clipping. The ratio starts at exactly 1.000; tokens with a positive advantage rise to near 1.2, negative ones fall to near 0.8, and the clip cuts the rest (illustrative motion).' },
        { t: 'Sync and start over', d: 'The new weights go to the inference engine and \\(\\theta_{\\text{old}}\\leftarrow\\theta\\). One InstructGPT iteration has 512 episodes; 256k episodes are 500 iterations.' },
      ];
      let H = null;
      const st = UI.stepper(fig, steps, { autoplay: 6000, onStep: () => H && H.redraw() });
      H = V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const i = st.i, el = st.elapsed();
        const narrow = w < 560;
        // ---- characters
        const ACTORS = [
          { k: 'motor', l: narrow ? 'vLLM' : 'engine (vLLM)', c: C.e },
          { k: 'pol', l: narrow ? 'π_θ' : 'policy π_θ', c: C.a },
          { k: 'ref', l: narrow ? 'π_ref' : 'reference π_ref', c: C.b },
          { k: 'rm', l: narrow ? 'RM' : 'reward r_φ', c: C.d },
          { k: 'crit', l: narrow ? 'V_ψ' : 'critic V_ψ', c: C.c },
        ];
        const aw = (w - 20) / 5, ay = 30, as = Math.min(32, aw * 0.42);
        ACTORS.forEach((a, k) => {
          const cx = 10 + aw * (k + 0.5);
          const on = ACT[i].includes(a.k);
          if (on) V.glow(g, cx, ay, as * 1.3, a.c, 0.5 + 0.2 * Math.sin(t * 5));
          face(g, cx, ay, as, a.c, { alpha: on ? 1 : 0.3, mood: on ? 'o' : undefined, dash: a.k === 'ref', fill: a.k === 'ref' ? V.alpha(a.c, on ? 0.35 : 0.12) : undefined, ghost: a.k === 'ref' });
          V.text(g, a.l, cx, ay + as / 2 + 14, { size: 11, align: 'center', color: on ? C.ink : C.muted, weight: on ? 700 : 500, maxW: aw - 4 });
        });
        const ax = (k) => 10 + aw * (k + 0.5);
        if (i === 8) { const u = (t * 0.7) % 1; V.arrow(g, ax(1) - as / 2 - 4, ay - 4, ax(0) + as / 2 + 4, ay - 4, { color: C.d, lw: 2.2, curve: -14 }); V.text(g, 'weights', (ax(0) + ax(1)) / 2, ay - 24, { size: 11, font: 'mono', color: C.d, align: 'center', weight: 700 }); V.dot(g, V.lerp(ax(1), ax(0), u), ay - 18 + Math.sin(u * Math.PI) * -2, 3, C.d); }
        // ---- token strip
        const labW = narrow ? 92 : 128;
        const mx = 10 + labW, mw = w - 20 - labW;
        const cwid = mw / 5;
        const ty = ay + as / 2 + 44;
        const prompt = narrow ? 'x: sky color?' : 'x: What color is the sky?';
        chip(g, 10, ty, labW - 8, 22, prompt, { size: 11, fill: V.alpha(C.muted, 0.1), font: 'sans', color: C.ink });
        const genN = i === 0 ? 0 : i === 1 ? Math.min(5, Math.floor(el / 0.7)) : 5;
        if (i === 0) { V.text(g, 'B = 512 prompts ⟶', mx + 4, ty + 14, { size: 11.5, font: 'mono', color: C.ink, alpha: 0.6 + 0.4 * Math.sin(t * 3) }); }
        for (let k = 0; k < 5; k++) {
          const x = mx + k * cwid + 3, cw = cwid - 6;
          if (k < genN) chip(g, x, ty, cw, 22, TOKS[k], { fill: V.alpha(C.e, 0.22), stroke: C.e, size: 12, weight: 700 });
          else chip(g, x, ty, cw, 22, '·', { fill: V.alpha(C.muted, 0.05), stroke: V.alpha(C.muted, 0.2), color: C.muted });
        }
        if (i >= 2 && i <= 8) {
          const x = mx + 4 * cwid + cwid / 2;
          const pop = i === 2 ? V.ease.back(V.clamp(el / 0.6, 0, 1)) : 1;
          V.box(g, x - 22 * pop, ty - 18 * pop, 44 * pop, 16 * pop, { r: 8, fill: C.d });
          if (pop > 0.6) V.text(g, 'RM 2.0', x, ty - 10, { size: 11, font: 'mono', weight: 700, color: C.dark ? '#1a1200' : '#fff', align: 'center', base: 'middle' });
        }
        // ---- batch matrix
        const my0 = ty + 34;
        const rh = (h - my0 - 8) / ROWS.length;
        const ch = Math.min(30, rh - 4);
        ROWS.forEach((row, r) => {
          const y = my0 + r * rh;
          const shown = i >= row.from;
          const fresh = i === row.from;
          V.text(g, row.l, 10, y + ch / 2 + 4, { size: narrow ? 11 : 12, font: 'mono', color: shown ? C.ink : V.alpha(C.muted, 0.6), weight: fresh ? 700 : 500, maxW: labW - 6 });
          for (let k = 0; k < 5; k++) {
            const x = mx + k * cwid + 3, cw = cwid - 6;
            let vis = shown;
            if (fresh && row.rtl) vis = el > (4 - k) * 0.45 + 0.2;
            else if (fresh) vis = el > k * 0.12;
            if (!vis) { V.box(g, x, y, cw, ch, { r: 4, fill: V.alpha(C.muted, 0.06), stroke: V.alpha(C.muted, 0.15), lw: 1 }); continue; }
            let v = row.v[k];
            if (row.rho) { const u = i === 7 ? V.ease.out(V.clamp((el - 0.4) / 3, 0, 1)) : 1; v = V.lerp(1, row.v[k], u); }
            const tt = row.rho ? (v - 1) / row.sc : v / row.sc;
            const fill = V.cmap(V.clamp(tt, -1, 1), 'div');
            const hot = (row.l === 'r_t' && k === 4) || (fresh && row.rtl && el > (4 - k) * 0.45 + 0.2 && el < (4 - k) * 0.45 + 0.75);
            V.box(g, x, y, cw, ch, { r: 4, fill, stroke: hot ? C.d : V.alpha(C.muted, 0.25), lw: hot ? 2 : 1 });
            const light = Math.abs(tt) > 0.55;
            const fs = Math.max(10, Math.min(12.5, cw / 5.6));
            V.text(g, fmt(v, row.d), x + cw / 2, y + ch / 2 + 1, { size: fs, font: 'mono', align: 'center', base: 'middle', color: light ? (C.dark ? '#00210F' : '#fff') : C.ink, weight: 600 });
            if (row.rho && i === 7 && Math.abs(v - 1) > 0.199) V.text(g, '✂', x + cw - 7, y + 9, { size: 11, color: C.d, align: 'center' });
          }
          if (fresh && row.rtl && el < 3) {
            const u = V.clamp((el - 0.2) / 2.0, 0, 1);
            const xx = mx + mw - u * mw;
            V.text(g, '← γλ', xx, y - 2, { size: 11, font: 'mono', color: C.d, align: 'center', weight: 700 });
          }
        });
        // columns
        if (!narrow) TOKS.forEach((tk, k) => V.text(g, 't=' + (k + 1), mx + k * cwid + cwid / 2, my0 - 4, { size: 10.5, font: 'mono', color: C.muted, align: 'center' }));
      });
    })();

    /* ======================================================= FIG 8 — engine in the loop */
    (function () {
      const fig = root.querySelector('#fig-motor');
      adaptH(fig, 365, 380);
      const S = { mode: 'col', g: 0.7, f: 0.7 };
      function model() {
        const G = 100 * S.g, rest = 100 - G, Sc = 0.4 * rest, T = 0.6 * rest, W = 4, sL = 1, sN = 3, f = S.f;
        const gA = G / f, tB = (Sc + T) / (1 - f);
        return { G, Sc, T, W, sL, sN, f, gA, tB, col: G + Sc + T + W + sL, dis: gA + tB + sN, asy: Math.max(gA, tB) + sN };
      }
      function lanesFor(mode, M, narrow) {
        if (mode === 'col') {
          const segs = [];
          for (let it = 0; it < 3; it++) {
            let o = it * M.col;
            const push = (d, k) => { segs.push({ a: o, b: o + d, k }); o += d; };
            push(M.G, 'gen'); push(M.W / 2, 'sw'); push(M.Sc, 'score'); push(M.T, 'train'); push(M.sL, 'sync'); push(M.W / 2, 'sw');
          }
          return { lanes: [{ l: narrow ? 'all' : 'all GPUs', segs }], P: M.col };
        }
        const P = mode === 'dis' ? M.dis : M.asy;
        const A = [], B = [];
        for (let it = 0; it < 3; it++) {
          const o = it * P;
          if (mode === 'dis') {
            A.push({ a: o, b: o + M.gA, k: 'gen' }, { a: o + M.gA, b: o + M.gA + M.tB, k: 'idle' }, { a: o + P - M.sN, b: o + P, k: 'sync' });
            B.push({ a: o, b: o + M.gA, k: 'idle' }, { a: o + M.gA, b: o + M.gA + M.tB * 0.4, k: 'score' }, { a: o + M.gA + M.tB * 0.4, b: o + M.gA + M.tB, k: 'train' }, { a: o + P - M.sN, b: o + P, k: 'sync' });
          } else {
            A.push({ a: o, b: o + M.gA, k: 'gen' }); if (P - M.sN > M.gA) A.push({ a: o + M.gA, b: o + P - M.sN, k: 'idle' }); A.push({ a: o + P - M.sN, b: o + P, k: 'sync' });
            if (it === 0) B.push({ a: o, b: o + P - M.sN, k: 'idle' });
            else { B.push({ a: o, b: o + M.tB * 0.4, k: 'score' }, { a: o + M.tB * 0.4, b: o + M.tB, k: 'train' }); if (P - M.sN > M.tB) B.push({ a: o + M.tB, b: o + P - M.sN, k: 'idle' }); }
            B.push({ a: o + P - M.sN, b: o + P, k: 'sync' });
          }
        }
        const pf = Math.round(S.f * 100);
        return { lanes: [{ l: narrow ? 'rollout ' + pf + '%' : 'rollout pool (' + pf + '%)', segs: A }, { l: narrow ? 'train ' + (100 - pf) + '%' : 'training pool (' + (100 - pf) + '%)', segs: B }], P };
      }
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Highlight', options: [{ v: 'col', label: 'colocated' }, { v: 'dis', label: 'sync disaggregated' }, { v: 'async', label: 'async disaggregated' }], value: 'col', onChange: (v) => { S.mode = v; upd(); } });
      UI.slider(ctl, { label: 'generation share of time (colocated)', min: 0.3, max: 0.95, step: 0.05, value: 0.7, fmt: (v) => Math.round(v * 100) + '%', onInput: (v) => { S.g = v; upd(); } });
      UI.slider(ctl, { label: 'GPUs in the rollout pool', min: 0.2, max: 0.8, step: 0.05, value: 0.7, fmt: (v) => Math.round(v * 100) + '%', onInput: (v) => { S.f = v; upd(); } });
      const rT = UI.readout(ctl, 'time per iteration', '—'), rI = UI.readout(ctl, 'idle GPU', '—'), rS = UI.readout(ctl, 'staleness', '—'), rV = UI.readout(ctl, 'vs colocated', '—');
      function upd() {
        const M = model();
        const P = S.mode === 'col' ? M.col : S.mode === 'dis' ? M.dis : M.asy;
        rT.set(fmt(P, 0) + ' units');
        let idle = 0;
        if (S.mode !== 'col') { const iA = P - M.sN - M.gA, iB = P - M.sN - M.tB; idle = (S.f * Math.max(0, iA) + (1 - S.f) * Math.max(0, iB)) / P; }
        rI.set(Math.round(idle * 100) + '%', idle > 0.3 ? 'bad' : idle > 0.1 ? 'warn' : 'ok');
        rS.set(S.mode === 'async' ? '1 iteration' : '0 (on-policy)', S.mode === 'async' ? 'warn' : 'ok');
        const ratio = M.col / P;
        rV.set(fmt(ratio, 2) + '×', ratio >= 0.999 ? 'ok' : ratio > 0.7 ? 'warn' : 'bad');
      }
      upd();
      const NAMES = { col: 'colocated', dis: 'synchronous disaggregated', async: 'asynchronous disaggregated' };
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const M = model();
        const narrow = w < 560;
        const KC = { gen: C.a, score: C.b, train: C.c, sync: C.d, sw: C.muted };
        const Tmax = 3 * Math.max(M.col, M.dis, M.asy);
        const lx = narrow ? 76 : 150, x0 = lx, xw = w - x0 - 14;
        const X = (u) => x0 + (u / Tmax) * xw;
        const laneH = narrow ? 30 : 32, laneGap = 16, titleH = 20, blockGap = narrow ? 18 : 14;
        const head = ((t % 12) / 11) * Tmax;
        let y = 6;
        ['col', 'dis', 'async'].forEach((mk) => {
          const L = lanesFor(mk, M, narrow);
          const sel = mk === S.mode;
          const span = 3 * L.P;
          g.globalAlpha = sel ? 1 : 0.42;
          V.text(g, NAMES[mk] + ' · ' + fmt(L.P, 0) + ' units per iteration', 10, y + 13, { size: 12, color: sel ? C.ink : C.muted, weight: 700, maxW: w - 20 });
          if (sel) V.dot(g, w - 16, y + 9, 4, C.a);
          y += titleH;
          const ys = [];
          L.lanes.forEach((ln, k) => {
            ys.push(y);
            V.text(g, ln.l, 10, y + laneH / 2 + 4, { size: 11, color: C.ink, weight: 600, maxW: lx - 16 });
            V.box(g, x0, y, X(span) - x0, laneH, { r: 5, fill: V.alpha(C.muted, 0.05), stroke: V.alpha(C.muted, 0.25), lw: 1 });
            ln.segs.forEach((s) => {
              const xa = X(s.a), xb = X(s.b);
              if (xb - xa < 0.5) return;
              if (s.k === 'idle') { hatch(g, xa, y + 2, xb - xa, laneH - 4, V.alpha(C.muted, 0.5), 6); return; }
              const live = sel && head >= s.a && head <= s.b;
              V.box(g, xa + 0.5, y + 3, Math.max(1, xb - xa - 1), laneH - 6, { r: 3, fill: V.alpha(KC[s.k], live ? 1 : 0.62), stroke: live ? C.ink : undefined, lw: 1.5 });
              if (xb - xa > 36) V.text(g, s.k === 'gen' ? 'gen' : s.k === 'score' ? 'score' : s.k === 'train' ? 'train' : s.k === 'sync' ? 'sync' : 'switch', (xa + xb) / 2, y + laneH / 2 + 4, { size: 11, font: 'mono', color: C.dark ? '#06140d' : '#fff', align: 'center', weight: 700 });
            });
            y += laneH + (k < L.lanes.length - 1 ? laneGap : 0);
          });
          // end of the 3 iterations
          V.line(g, X(span), ys[0] - 3, X(span), y + 3, { color: C.ink, lw: 1.4, dash: [3, 3] });
          // rollouts going down and weights going up
          if (L.lanes.length === 2 && sel) {
            const yA = ys[0] + laneH, yB = ys[1];
            L.lanes[0].segs.filter((s) => s.k === 'gen').forEach((s) => {
              const d = head - s.b, dur = Tmax * 0.035;
              if (d >= 0 && d <= dur) { const u = d / dur; for (let q = 0; q < 5; q++) V.dot(g, X(s.b) - 6 + q * 4, V.lerp(yA, yB, V.clamp(u - q * 0.1, 0, 1)), 2.6, C.a); }
            });
            L.lanes[1].segs.filter((s) => s.k === 'sync').forEach((s) => {
              if (head >= s.a - Tmax * 0.02 && head <= s.b + Tmax * 0.02) V.arrow(g, X(s.a) + 4, yB - 1, X(s.a) + 4, yA + 1, { color: C.d, lw: 2.4, head: 7 });
            });
          }
          g.globalAlpha = 1;
          y += blockGap;
        });
        // shared clock
        const hx = X(head);
        V.line(g, hx, 6, hx, y - blockGap + 4, { color: V.alpha(C.ink, 0.55), lw: 1.3 });
        // legend
        const leg = [['generate', C.a], ['score', C.b], ['train', C.c], ['sync', C.d], ['switch', C.muted]];
        let lxx = 10, ly = narrow ? h - 26 : h - 10;
        leg.concat([['idle', null]]).forEach(([s, c]) => {
          const ww = 16 + s.length * 6.6 + 12;
          if (lxx + ww > w - 10) { lxx = 10; ly += 16; }
          if (c) V.box(g, lxx, ly - 9, 12, 10, { r: 2, fill: V.alpha(c, 0.8) }); else hatch(g, lxx, ly - 9, 12, 10, V.alpha(C.muted, 0.6), 4);
          V.text(g, s, lxx + 16, ly, { size: 11, color: C.ink }); lxx += ww;
        });
      });
    })();

    /* ======================================================= FIG 9 — diagnostic dashboard */
    (function () {
      const fig = root.querySelector('#fig-painel');
      adaptH(fig, 440, 640);
      const N = 120;
      const xs = Array.from({ length: N }, (_, k) => k / (N - 1));
      const nz = (seed) => { const r = V.rng(seed); return xs.map(() => r.normal()); };
      const n1 = nz(3), n2 = nz(5), n3 = nz(9), n4 = nz(13);
      const e = Math.exp;
      const SC = {
        ok: {
          lab: 'healthy',
          rm: (x, k) => 2.2 * (1 - e(-3 * x)) + 0.05 * n1[k], gold: (x, k) => 1.6 * (1 - e(-3 * x)) + 0.05 * n2[k],
          kl: (x, k) => 7 * (1 - e(-2.5 * x)) + 0.15 * n3[k], len: (x, k) => 180 + 25 * x + 4 * n4[k],
          rho: () => 0, vl: (x) => 0.32 * e(-3 * x) + 0.06,
          sym: 'The RM score and real quality rise together; the KL levels off; length stays stable; the ratio starts at exactly 1 in the 1st epoch; the value loss falls.',
          fix: 'Nothing to do: this is the reference picture to compare against.',
        },
        beta: {
          lab: 'small β / high LR',
          rm: (x, k) => 4.6 * Math.pow(x, 0.75) + 0.06 * n1[k], gold: (x, k) => 7 * x * e(-3 * x) + 0.05 * n2[k],
          kl: (x, k) => 34 * Math.pow(x, 1.4) + 0.3 * n3[k], len: (x, k) => 180 + 330 * x * x + 6 * n4[k],
          rho: () => 0, vl: (x) => 0.2 + 0.35 * x,
          sym: 'The RM score climbs nonstop while real quality rises and then falls; the KL and the length shoot up.',
          fix: 'Raise β or aim at a target KL (adaptive control), lower the LR, stop early on the held-out set.',
        },
        drop: {
          lab: 'dropout on',
          rm: (x, k) => 1.9 * (1 - e(-2.5 * x)) + 0.3 * n1[k], gold: (x, k) => 1.3 * (1 - e(-2.5 * x)) + 0.12 * n2[k],
          kl: (x, k) => Math.max(0, 6 * (1 - e(-2.5 * x)) + 1.8 * n3[k]), len: (x, k) => 180 + 20 * x + 6 * n4[k],
          rho: (x, k) => 0.05 + 0.015 * Math.abs(n1[(k * 7) % N]), vl: (x) => 0.3 * e(-2 * x) + 0.12,
          sym: 'The mean ratio is already far from 1 in the 1st epoch; the KL is noisy and unreliable; the reward oscillates.',
          fix: 'Turn off dropout in the policy, the reference, the RM and the critic (N+, Ziegler).',
        },
        crit: {
          lab: 'cold critic',
          rm: (x, k) => 2.0 * (1 - e(-3 * Math.max(0, x - 0.3))) - 0.35 * e(-Math.pow((x - 0.12) / 0.1, 2)) + 0.05 * n1[k], gold: (x, k) => 1.5 * (1 - e(-3 * Math.max(0, x - 0.3))) - 0.3 * e(-Math.pow((x - 0.12) / 0.1, 2)) + 0.05 * n2[k],
          kl: (x, k) => 5 * (1 - e(-2 * x)) + 0.4 * e(-Math.pow((x - 0.1) / 0.08, 2)) * 6 + 0.15 * n3[k], len: (x, k) => 180 + 15 * x + 4 * n4[k],
          rho: () => 0, vl: (x) => 1.0 * e(-5 * x) + 0.08,
          sym: 'The value loss starts out huge; the reward is slow to rise (or drops at first) and the KL jolts at the start.',
          fix: 'Initialize the critic from the RM, pretrain the critic before touching the policy and turn on value clipping.',
        },
        len: {
          lab: 'EOS = PAD (length)',
          rm: (x, k) => 2.6 * x + 0.05 * n1[k], gold: (x, k) => 0.55 + 0.1 * Math.sin(x * 3) + 0.05 * n2[k],
          kl: (x, k) => 12 * x + 0.2 * n3[k], len: (x, k) => Math.min(1000, 180 + 900 * x) + 6 * n4[k],
          rho: () => 0, vl: (x) => 0.25 * e(-2 * x) + 0.1,
          sym: 'Length grows until it hits the ceiling; the RM score tracks the length; real quality stays flat.',
          fix: 'Make sure EOS ≠ PAD, use the “EOS trick” (−1 without EOS), monitor the length and deal with the length bias of the RM.',
        },
      };
      const S = { sc: 'ok' };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Scenario', options: Object.keys(SC).map((k) => ({ v: k, label: SC[k].lab })), value: 'ok', onChange: (v) => { S.sc = v; rD.set(SC[v].lab, v === 'ok' ? 'ok' : 'bad'); } });
      const rD = UI.readout(ctl, 'diagnosis', 'healthy'); rD.set('healthy', 'ok');
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const sc = SC[S.sc];
        const narrow = w < 600;
        const textH = narrow ? 112 : 66;
        const gw = (w - (narrow ? 20 : 30)) / 2, gh = (h - textH - 18) / 2;
        const ph = Math.min(1, (t % 9) / 6.5), n = Math.max(2, Math.floor(ph * N));
        const panels = [
          { t: 'RM score × real quality', y0: -0.6, y1: 4.8, yt: [0, 2, 4], series: [['rm', C.b], ['gold', C.a]] },
          { t: 'KL to the SFT (nats)', y0: 0, y1: 36, yt: [0, 10, 20, 30], series: [['kl', C.d]] },
          { t: 'mean length (tokens)', y0: 100, y1: 1100, yt: [200, 600, 1000], series: [['len', C.e]], cap: 1000 },
          { t: narrow ? '|ρ−1| (right) · loss (left)' : '|ρ−1| (right axis, 0–0.1) and value loss (left axis)', y0: 0, y1: 1.1, yt: [0, 0.5, 1], series: [['vl', C.c]], bars: true },
        ];
        panels.forEach((p, q) => {
          const col = q % 2, row = Math.floor(q / 2);
          const RA = p.bars ? 24 : 0; // room for the right axis (|ρ−1|)
          const A = { x: 10 + col * (gw + 10) + 34, y: 8 + row * (gh + 8) + 18, w: gw - 44 - RA, h: gh - 44 };
          V.box(g, A.x - 38, A.y - 22, A.w + RA + 46, A.h + 46, { r: 8, fill: V.alpha(C.muted, 0.04), stroke: V.alpha(C.muted, 0.2), lw: 1 });
          V.text(g, p.t, A.x - 30, A.y - 8, { size: 11, color: C.ink, weight: 600, maxW: A.w + RA + 34 });
          const F = { X: (x) => A.x + x * A.w, Y: (v) => A.y + A.h - ((v - p.y0) / (p.y1 - p.y0)) * A.h };
          p.yt.forEach((v) => { const y = F.Y(v); g.strokeStyle = V.alpha(C.muted, 0.15); g.lineWidth = 1; g.beginPath(); g.moveTo(A.x, y); g.lineTo(A.x + A.w, y); g.stroke(); V.text(g, String(v), A.x - 4, y, { size: 10.5, font: 'mono', color: C.muted, align: 'right', base: 'middle' }); });
          g.strokeStyle = V.alpha(C.muted, 0.5); g.beginPath(); g.moveTo(A.x, A.y); g.lineTo(A.x, A.y + A.h); g.lineTo(A.x + A.w, A.y + A.h); g.stroke();
          if (p.cap) { V.line(g, A.x, F.Y(p.cap), A.x + A.w, F.Y(p.cap), { color: RED(), dash: [4, 4], lw: 1.2 }); V.text(g, 'max_tokens', A.x + A.w - 2, F.Y(p.cap) - 4, { size: 10.5, font: 'mono', color: RED(), align: 'right' }); }
          if (p.bars) {
            // own axis on the right: 0 to 0.1 spans the full height
            const RMAX = 0.1, YR = (v) => A.y + A.h - (V.clamp(v, 0, RMAX) / RMAX) * A.h, xr = A.x + A.w;
            g.strokeStyle = V.alpha(C.muted, 0.5); g.lineWidth = 1; g.beginPath(); g.moveTo(xr, A.y); g.lineTo(xr, A.y + A.h); g.stroke();
            [[0, '0'], [0.05, null], [0.1, '0.1']].forEach(([v, s]) => { const y = YR(v); V.line(g, xr, y, xr + 4, y, { color: V.alpha(C.muted, 0.8), lw: 1 }); if (s) V.text(g, s, xr + 6, y, { size: 10.5, font: 'mono', color: C.muted, base: 'middle' }); });
            for (let k = 0; k < n; k += 8) { const v = sc.rho(xs[k], k); const bh = A.y + A.h - YR(v); const x = F.X(xs[k]); g.fillStyle = V.alpha(v > 0.01 ? RED() : C.a, 0.75); g.fillRect(x - 2, A.y + A.h - Math.max(1.5, bh), 4, Math.max(1.5, bh)); }
            V.text(g, sc.rho(0.5, 60) > 0.01 ? '≈ ' + fmt(sc.rho(0.5, 60), 2) : '0.000', A.x + 4, A.y + 10, { size: 10.5, font: 'mono', color: sc.rho(0.5, 60) > 0.01 ? RED() : C.a, weight: 700 });
          }
          p.series.forEach(([key, c]) => {
            g.beginPath();
            for (let k = 0; k < N; k++) { const v = V.clamp(sc[key](xs[k], k), p.y0, p.y1); const x = F.X(xs[k]), y = F.Y(v); if (k) g.lineTo(x, y); else g.moveTo(x, y); }
            g.strokeStyle = V.alpha(c, 0.22); g.lineWidth = 1.4; g.stroke();
            g.beginPath();
            for (let k = 0; k < n; k++) { const v = V.clamp(sc[key](xs[k], k), p.y0, p.y1); const x = F.X(xs[k]), y = F.Y(v); if (k) g.lineTo(x, y); else g.moveTo(x, y); }
            g.strokeStyle = c; g.lineWidth = 2; g.stroke();
            const v = V.clamp(sc[key](xs[n - 1], n - 1), p.y0, p.y1);
            V.dot(g, F.X(xs[n - 1]), F.Y(v), 3, c);
          });
          if (q === 0) { V.text(g, 'RM', A.x + A.w - 2, A.y + 10, { size: 10.5, color: C.b, align: 'right', weight: 700 }); V.text(g, 'real', A.x + A.w - 2, A.y + 23, { size: 10.5, color: C.a, align: 'right', weight: 700 }); }
        });
        // symptom and remedy
        const ty = h - textH + 6;
        V.box(g, 10, ty - 4, w - 20, textH - 6, { r: 8, fill: V.alpha(S.sc === 'ok' ? C.a : RED(), 0.07), stroke: V.alpha(S.sc === 'ok' ? C.a : RED(), 0.4), lw: 1 });
        const hh = V.wrap(g, 'Symptom: ' + sc.sym, 20, ty + 14, w - 40, { size: 11.5, color: C.ink, lineH: 15, maxLines: narrow ? 4 : 2 });
        V.wrap(g, 'Remedy: ' + sc.fix, 20, ty + 14 + hh + 2, w - 40, { size: 11.5, color: S.sc === 'ok' ? C.a : C.d, weight: 600, lineH: 15, maxLines: narrow ? 3 : 2 });
      });
    })();

    /* ======================================================= FIG 10 — memory */
    (function () {
      const fig = root.querySelector('#fig-memoria');
      adaptH(fig, 400, 430);
      const NL = [0.5, 1, 1.5, 3, 7, 8, 14, 32, 70];
      const GiB = 1073741824;
      const S = { ni: 4, ratio: 1, mode: 'full', engine: false };
      const SPARK = 121.69, MAC = 24e9 / GiB;
      function models() {
        const N = NL[S.ni] * 1e9, Nc = N * S.ratio;
        const L = [];
        const tr = (n) => [{ k: 'w', b: 2 }, { k: 'g', b: 2 }, { k: 'o', b: 12 }].map((x) => ({ ...x, bytes: x.b * n }));
        const fr = (n) => [{ k: 'w', b: 2, bytes: 2 * n }];
        const lo = (n) => [{ k: 'w', b: 2, bytes: 2 * n }, { k: 'a', b: 0.16, bytes: 0.16 * n }];
        if (S.mode === 'full') { L.push({ l: 'policy', c: 'a', parts: tr(N), n: N }, { l: 'critic', c: 'c', parts: tr(Nc), n: Nc }, { l: 'reference', c: 'b', parts: fr(N), n: N }, { l: 'RM', c: 'd', parts: fr(Nc), n: Nc }); }
        else if (S.mode === 'lora') { L.push({ l: 'policy + LoRA', c: 'a', parts: lo(N), n: N }, { l: 'critic + LoRA', c: 'c', parts: lo(Nc), n: Nc }, { l: 'reference', c: 'b', parts: [], n: N, note: '= policy without adapter' }, { l: 'RM', c: 'd', parts: fr(Nc), n: Nc }); }
        else { L.push({ l: 'policy', c: 'a', parts: tr(N), n: N }, { l: 'critic', c: 'c', parts: [], n: 0, note: 'removed (GRPO, ch. 56)' }, { l: 'reference', c: 'b', parts: fr(N), n: N }, { l: 'RM', c: 'd', parts: fr(Nc), n: Nc }); }
        if (S.engine) L.push({ l: 'engine (bf16)', c: 'e', parts: fr(N), n: N });
        L.forEach((m) => { m.bytes = m.parts.reduce((s, p) => s + p.bytes, 0); });
        return L;
      }
      const ctl = UI.controls(fig);
      UI.slider(ctl, { label: 'Policy parameters', min: 0, max: NL.length - 1, step: 1, value: 4, fmt: (v) => NL[v] + 'B', onInput: (v) => { S.ni = v; upd(); } });
      UI.seg(ctl, { label: 'RM and critic', options: [{ v: 1, label: '= policy' }, { v: 0.5, label: 'half' }, { v: 0.25, label: '1/4' }], value: 1, onChange: (v) => { S.ratio = v; upd(); } });
      UI.seg(ctl, { label: 'Method', options: [{ v: 'full', label: 'full FT' }, { v: 'lora', label: 'LoRA (~1%)' }, { v: 'grpo', label: 'no critic' }], value: 'full', onChange: (v) => { S.mode = v; upd(); } });
      UI.toggle(ctl, { label: '+ copy in the inference engine', value: false, onChange: (v) => { S.engine = v; upd(); } });
      const rT = UI.readout(ctl, 'total (no activations/KV)', '—'), rF = UI.readout(ctl, 'fits in', '—'), rB = UI.readout(ctl, 'policy bytes per parameter', '—');
      const S2 = { shown: null };
      function upd() {
        const L = models();
        const tot = L.reduce((s, m) => s + m.bytes, 0);
        const gib = tot / GiB;
        rT.set(fmt(tot / 1e9, 1) + ' GB · ' + fmt(gib, 1) + ' GiB');
        const tiers = [[MAC, 'Mac M4 24 GB (tight)'], [SPARK, '1× DGX Spark'], [2 * SPARK, '2× DGX Spark']];
        const hit = tiers.find(([cap]) => gib <= cap);
        if (!hit) rF.set('none of these', 'bad');
        else {
          const tight = gib > 0.85 * hit[0]; // less than 15% headroom: activations and KV do not fit
          rF.set(hit[1] + (tight ? ' (states only; no headroom for activations)' : ''), tight || hit[0] > SPARK ? 'warn' : 'ok');
        }
        rB.set(fmt(L[0].bytes / L[0].n, 2) + ' B');
      }
      upd();
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const L = models();
        const narrow = w < 560;
        // strips in the model's own color (the same as in the stacked bar); the tone tells the byte type
        const TONE = { w: 0.92, g: 0.55, o: 0.22, a: 0.55 };
        const top = 26, rowH = Math.min(46, (h - 170) / L.length);
        const labW = narrow ? 104 : 150, gibW = narrow ? 70 : 96;
        const cellX = 10 + labW, cellW = w - 20 - labW - gibW;
        const cw = cellW / 16;
        V.text(g, narrow ? 'bytes per parameter (1 cell = 1 B)' : 'bytes per parameter (each cell = 1 byte)', cellX, top - 10, { size: 11, color: C.muted, weight: 600, maxW: w - cellX - 8 });
        L.forEach((m, r) => {
          const y = top + r * rowH;
          const ch = Math.min(22, rowH - 16);
          V.dot(g, 16, y + ch / 2, 5, C[m.c]);
          V.text(g, m.l, 26, y + ch / 2 + 4, { size: 12, color: C.ink, weight: 600, maxW: labW - 20 });
          if (!m.parts.length) { V.text(g, m.note, cellX, y + ch / 2 + 4, { size: 11.5, color: C.muted, font: 'mono' }); }
          let cx = cellX;
          m.parts.forEach((p) => {
            const nCells = Math.max(1, Math.round(p.b));
            for (let q = 0; q < (p.b < 1 ? 1 : nCells); q++) {
              const ww = p.b < 1 ? Math.max(4, cw * p.b * 2) : cw - 2;
              const glow = p.k === 'o' && Math.floor((t * 6) % 12) === q;
              const mc = C[m.c];
              V.box(g, cx, y, ww, ch, { r: 3, fill: V.alpha(mc, glow ? 0.5 : TONE[p.k]), stroke: p.k === 'a' ? C.ink : mc, lw: p.k === 'a' ? 1.4 : 1 });
              cx += (p.b < 1 ? ww + 2 : cw);
            }
          });
          if (m.parts.length) V.text(g, fmt(m.bytes / GiB, 1) + ' GiB', w - 12, y + ch / 2 + 4, { size: 12, font: 'mono', color: C.ink, align: 'right', weight: 600 });
          if (m.parts.length && !narrow) { const lbl = m.parts.length === 3 ? '2 weights · 2 grad · 12 Adam+master' : m.parts.length === 2 ? '2 weights (frozen) + adapter' : '2 bf16 weights'; V.text(g, lbl, cellX, y + ch + 11, { size: 10.5, color: C.muted, font: 'mono' }); }
        });
        // tone legend
        {
          const ly = top + L.length * rowH + 6;
          const items = [['w', 'weights'], ['g', 'grad'], ['o', narrow ? 'Adam+master' : 'Adam + FP32 master']].concat(S.mode === 'lora' ? [['a', narrow ? 'adapter' : 'LoRA adapter']] : []);
          let lx = narrow ? 16 : cellX;
          if (!narrow) V.text(g, 'tone:', lx - 6, ly + 10, { size: 11, color: C.muted, align: 'right' });
          items.forEach(([k, s]) => {
            V.box(g, lx, ly, 14, 12, { r: 3, fill: V.alpha(C.ink, TONE[k] * 0.8), stroke: k === 'a' ? C.ink : V.alpha(C.ink, 0.6), lw: 1 });
            V.text(g, s, lx + 18, ly + 10, { size: 11, color: C.ink });
            g.font = V.font(11, 'sans', 400); lx += 18 + g.measureText(s).width + 12;
          });
        }
        // stacked bar
        const tot = L.reduce((s, m) => s + m.bytes, 0) / GiB;
        const scale = Math.max(tot, 2 * SPARK) * 1.08;
        const by = h - 96, bh = 30, bx = 14, bw = w - 28;
        const X = (v) => bx + (v / scale) * bw;
        if (!S2.shown || S2.shown.length !== L.length) S2.shown = L.map(() => 0);
        let acc = 0;
        V.text(g, 'total memory (GiB)', bx, by - 12, { size: 11, color: C.muted, weight: 600 });
        V.box(g, bx, by, bw, bh, { r: 6, fill: V.alpha(C.muted, 0.06), stroke: V.alpha(C.muted, 0.3), lw: 1 });
        L.forEach((m, k) => {
          S2.shown[k] += (m.bytes / GiB - S2.shown[k]) * 0.15;
          const a = X(acc), b = X(acc + S2.shown[k]);
          if (b - a > 0.5) { V.box(g, a, by + 2, b - a, bh - 4, { r: 3, fill: V.alpha(C[m.c], 0.8) }); if (b - a > 40) V.text(g, m.l.split(' ')[0], (a + b) / 2, by + bh / 2 + 4, { size: 10.5, color: C.dark ? '#06140d' : '#fff', align: 'center', weight: 700, maxW: b - a - 4 }); }
          acc += S2.shown[k];
        });
        [[MAC, 'Mac M4 24 GB', C.muted], [SPARK, '1× Spark ~121.7 GiB', C.e], [2 * SPARK, '2× Spark ~243 GiB', C.d]].forEach(([v, s, c], k) => {
          const x = X(v);
          V.line(g, x, by - 4, x, by + bh + 8, { color: c, lw: 2, dash: [4, 3] });
          const lyy = by + bh + 20 + (k % 2) * 14;
          V.text(g, s, x, lyy, { size: 10.5, font: 'mono', color: c, align: x > w - 90 ? 'right' : x < 70 ? 'left' : 'center' });
        });
        const ok = tot <= SPARK ? C.e : tot <= 2 * SPARK ? C.d : RED();
        g.font = V.font(11, 'sans', 600);
        const titleEnd = bx + g.measureText('total memory (GiB)').width + 12; // the total never covers the bar title
        V.text(g, fmt(tot, 1) + ' GiB', V.clamp(X(tot) + 6, titleEnd, w - 70), by - 12, { size: 12, font: 'mono', weight: 700, color: ok });
      });
    })();
  },
});

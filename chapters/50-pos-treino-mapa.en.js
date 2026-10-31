LA.chapter({
  id: 'pos-treino-mapa',
  part: 10,
  order: 50,
  title: 'The post-training map: from base model to assistant',
  navTitle: 'Post-training map',
  subtitle: 'Why a text completer becomes an assistant and then a reasoning model: the SFT → preference → RL pipeline, the KL leash, on- and off-policy, who judges, the taxonomy of every method in the module and what GLM, Qwen, DeepSeek and MiMo have published about their own post-training.',
  minutes: 35,
  recipes: ['glm', 'qwen', 'ds'],
  tags: ['post-training', 'SFT', 'RLHF', 'DPO', 'GRPO', 'RLVR', 'KL', 'on-policy', 'MOPD', 'reward model', 'distillation'],
  goals: [
    'Explain what a <b>base model</b> does not deliver (format, instructions, preferences, safety, verifiable reasoning) and how base → instruct → reasoning chain together',
    'Place any method in the <b>SFT → preference → RL</b> pipeline and in its 2025–26 variants (RLVR, on-policy distillation, MOPD, specialists + fusion)',
    'Read the <b>KL leash</b>: \\(\\pi^* \\propto \\pi_{\\text{ref}}\\,e^{r/\\beta}\\), the role of \\(\\beta\\) and why it holds reward hacking in check',
    'Classify methods by <b>data origin</b> (on/off-policy), <b>signal format</b> and <b>who judges</b>, and estimate what each one requires in data and memory',
    'Connect post-training to Module 1: the inference engine in the RL loop, the training–inference mismatch and what changes when serving the post-trained model',
  ],
  prereqs: ['geracao-amostragem', 'aprender-com-dados'],
  summary: [
    'Pretraining delivers knowledge, not behavior. Post-training adjusts the <b>same tensors</b> to provide format, instruction following, preferences, safety and verifiable reasoning: base → instruct (the 1.3B InstructGPT was preferred over the 175B GPT-3) → reasoning (R1-Zero: AIME 2024 from 15.6% to 71.0% with RL alone).',
    'Typical pipeline: <b>SFT</b> (demonstrations, off-policy) → <b>preference</b> (DPO and family, or RM + PPO) → <b>RL</b> (RLHF, or RLVR with a verifier). In 2025–26 the <b>specialists + fusion</b> pattern took hold: per-domain RL runs fused by on-policy distillation (MiMo\'s MOPD, OPD with more than 10 teachers in DeepSeek-V4, cross-stage distillation in GLM-5).',
    'The leash: \\(\\max\\,\\mathbb E[r] - \\beta\\,\\mathrm{KL}(\\pi_\\theta\\|\\pi_{\\text{ref}})\\) has the solution \\(\\pi^* \\propto \\pi_{\\text{ref}}\\,e^{r/\\beta}\\). A small \\(\\beta\\) stretches the leash and lets the optimizer exploit the judge\'s loopholes (Goodhart); the KL also preserves capabilities and acts as an entropy bonus. Values range from 0.05 (Stiennon) to 0.001 (R1-Zero), and several modern RLVR pipelines drop the KL altogether.',
    '<b>On-policy</b> (\\(\\mu = \\pi_\\theta\\)) trains on the states the model visits and avoids exposure bias. SFT minimizes the <b>forward</b> KL (mass-covering); RL with KL and on-policy distillation minimize the <b>reverse</b> KL (mode-seeking). On-policy distillation combines on-policy sampling with a dense signal.',
    'A reward is a scalar; a preference is a comparison (Bradley–Terry: \\(P = \\sigma(r_w - r_l)\\)). The judge can be a human, an RM, a verifier, an AI with principles, a teacher or the model itself. Through the <b>gradient coefficient</b> lens, every method pushes token log-probs: what changes is the data source and the weight of each token.',
    'Cost: classic PPO keeps 4 models (\\(2\\times16N + 2\\times2N = 36N\\) bytes; 7B → 252 GB); GRPO removes the critic, DPO removes the RM and the RL, SimPO and ORPO remove the reference. In on-policy loops, <b>generation</b> dominates the time (OpenRLHF: more than 90%), which is why vLLM and SGLang enter the loop, and the \\(\\mu \\neq \\pi\\) mismatch calls for correction (importance weights, R3, determinism).',
    'No card for the recipe checkpoints details its own post-training; what is known comes from the families\' reports. When serving, respect what post-training taught: template, parser, <code>reasoning_effort</code> (in DeepSeek-V4, each level was taught by a distinct RL specialist), quantization measured by KLD, and drafters that keep up with the policy.',
  ],
  quiz: [
    { q: 'You send “What is the capital of France?” to a base model and it answers “What is the capital of Germany?”. What is the best explanation?', o: ['Missing knowledge: only SFT teaches geography', 'The model continues the text in the most likely way according to pretraining, and in exercise lists on the web one question is usually followed by another. What is missing is conversation format, not knowledge', 'The temperature is too high', 'The base model\'s tokenizer has no “Paris” token'], a: 1, e: 'Pretraining only teaches the model to continue text. The knowledge is there (the same model, after SFT, answers Paris), but the turn format, the end of turn and instruction following come from post-training. Temperature changes the draw, not which kind of continuation is most likely.' },
    { q: 'In the objective \\(\\max\\,\\mathbb E[r] - \\beta\\,\\mathrm{KL}(\\pi\\|\\pi_{\\text{ref}})\\), the reward model overrates a bad response. What happens when \\(\\beta \\to 0\\)?', o: ['\\(\\pi^*\\) goes back to \\(\\pi_{\\text{ref}}\\)', '\\(\\pi^* \\propto \\pi_{\\text{ref}}\\,e^{r/\\beta}\\) concentrates the mass on the argmax of \\(r\\), precisely the overrated response: the RM reward goes up and the true quality goes down (reward hacking)', 'Nothing changes, because the KL does not depend on \\(\\beta\\)', 'The policy becomes uniform over all responses'], a: 1, e: 'With a small \\(\\beta\\), \\(e^{r/\\beta}\\) amplifies any reward difference, and the policy collapses onto the judge\'s argmax. If the judge has a loophole, that is where the mass ends up: it is the Goodhart hump from the leash figure. It is \\(\\beta \\to \\infty\\) that gives you back \\(\\pi_{\\text{ref}}\\).' },
    { q: 'Which of these training setups is on-policy?', o: ['SFT on the responses of a curated dataset', 'Offline DPO with pairs from a public dataset', 'MOPD: the student generates the responses and each teacher gives log-probs on the student\'s tokens', 'Off-policy distillation: SFT on a teacher\'s pre-generated outputs'], a: 2, e: 'On-policy means the data comes from the policy being trained. In MOPD the student generates and the teacher only scores. In the other three options the texts come from humans, from another model or from an old version, so \\(\\mu \\neq \\pi_\\theta\\).' },
    { q: 'The target has two separate modes and the student is a single Gaussian. What does SFT (maximum likelihood) tend to do, compared with on-policy distillation?', o: ['Both give the same fit', 'SFT minimizes \\(\\mathrm{KL}(p\\|q)\\), which is mass-covering: the student gets wide, covers both modes and puts mass in the valley. On-policy distillation minimizes \\(\\mathrm{KL}(q\\|p)\\), which is mode-seeking, and settles onto one mode', 'SFT picks one mode; on-policy distillation covers both', 'SFT ignores the smaller mode because it uses samples from the student itself'], a: 1, e: 'The forward KL punishes the student wherever it gives little mass to something the data does, so it spreads out. The reverse KL punishes the student wherever it puts mass and the target does not, so it concentrates. That is why on-policy distillation works well when the student is smaller than the teacher.' },
    { q: 'Which method does away with both the reward model and the reference model?', o: ['RLHF with PPO', 'DPO', 'KTO', 'SimPO'], a: 3, e: 'SimPO uses the policy\'s own mean per-token log-prob, with a margin \\(\\gamma\\), and does not need \\(\\pi_{\\text{ref}}\\) (ORPO also drops the reference). DPO and KTO need \\(\\pi_{\\text{ref}}\\), because the implicit reward is \\(\\beta\\log\\pi/\\pi_{\\text{ref}}\\). RLHF with PPO needs an RM, a reference and a critic.' },
    { q: 'In a GRPO loop, the rollouts are generated in vLLM and the gradient is computed in FSDP, with the same weights. Why, even so, is training not exactly on-policy?', o: ['Because GRPO does not use KL', 'Because kernels, precision (FP8 rollouts, for example), summation order and, in MoE, routing differ between the engines: \\(\\mu_\\theta \\neq \\pi_\\theta\\). Hence fixes such as truncated importance weights (TIS, IcePop), routing replay (R3) and deterministic kernels', 'Because vLLM uses a different tokenizer', 'Because FSDP drops the LM head'], a: 1, e: 'Floating-point addition is not associative, and two engines with the same weights give different log-probs. In MoE, ~94% of tokens pick different experts in at least one layer between inference and training (R3 paper). The Qwen recipe shows the other side: with the determinism overlay, the logprob spread fell from 0.25 and 6.07 nats to zero.' },
  ],
  mount(root, LA) {
    const V = LA.viz, UI = LA.ui;
    const onClean = (fn) => { if (LA.onCleanup) LA.onCleanup(fn); };
    const nf = (x, d = 0) => (+x).toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d });
    const sg = (x, d = 3) => (x > 5e-13 ? '+' : x < -5e-13 ? '−' : '') + nf(Math.abs(x), d);
    const RED = () => { const C = V.C(); return C.red || C.c; };
    const fontC = {};
    const F = (size, font, weight) => { const k = size + (font || 'sans') + (weight || 400); return fontC[k] || (fontC[k] = V.font(size, font || 'sans', weight || 400)); };
    const tw = (g, s, size, font, weight) => { g.font = F(size, font, weight); return g.measureText(String(s)).width; };

    function adaptH(fig, wide, narrow, bp = 640) {
      const st = fig.querySelector('.la-fig-stage');
      const set = () => { const want = (fig.clientWidth < bp ? narrow : wide) + 'px'; if (st.style.height !== want) st.style.height = want; };
      set();
      const ro = new ResizeObserver(set); ro.observe(fig); onClean(() => ro.disconnect());
    }
    const wrapCtl = (o) => {
      const el = o && o.el ? o.el : o;
      if (el && el.style) { el.style.flexWrap = 'wrap'; el.style.maxWidth = '100%'; el.style.minWidth = '0'; }
      if (el && el.querySelectorAll) {
        el.querySelectorAll('.la-seg').forEach((s) => { s.style.flexWrap = 'wrap'; s.style.maxWidth = '100%'; });
        el.querySelectorAll('select').forEach((s) => { s.style.maxWidth = '100%'; s.style.minWidth = '0'; });
      }
      return o;
    };
    /** tira o uppercase do CSS de rótulos com símbolos (β, ε, N…) */
    const nocase = (c) => { const el = c && (c.el || c); if (el && el.querySelectorAll) el.querySelectorAll('label, span').forEach((n) => { if (!n.classList.contains('val')) n.style.textTransform = 'none'; }); return c; };

    function lines(g, str, maxW, size, font, weight) {
      g.font = F(size, font, weight);
      const words = String(str).split(/\s+/).filter(Boolean);
      const out = []; let cur = '';
      for (const wd of words) {
        const t = cur ? cur + ' ' + wd : wd;
        if (g.measureText(t).width > maxW && cur) { out.push(cur); cur = wd; } else cur = t;
      }
      if (cur) out.push(cur);
      return out;
    }
    /** parágrafo com quebra; retorna a altura usada */
    function para(g, str, x, y, maxW, o = {}) {
      const size = o.size || 12, lh = o.lh || Math.round(size * 1.36);
      let L = lines(g, str, maxW, size, o.font, o.weight);
      if (o.max && L.length > o.max) { L = L.slice(0, o.max); L[o.max - 1] = L[o.max - 1].replace(/\s*\S*$/, '…'); }
      L.forEach((l, i) => V.text(g, l, x, y + i * lh, { size, font: o.font, weight: o.weight, color: o.color, align: o.align, base: o.base }));
      return L.length * lh;
    }
    /** personagem: bolha com olhos (o modelo, o aluno, o cachorro…) */
    function blob(g, x, y, r, col, o = {}) {
      const C = V.C(), t = o.t || 0;
      const sq = 1 + 0.045 * Math.sin(t * 3.1 + (o.seed || 0));
      g.save(); g.translate(x, y); g.scale(1 / sq, sq);
      g.beginPath(); g.ellipse(0, 0, r, r * 0.92, 0, 0, Math.PI * 2);
      g.fillStyle = V.alpha(col, 0.92); g.fill(); g.lineWidth = 1.3; g.strokeStyle = col; g.stroke();
      g.restore();
      const eye = C.bg;
      const ex = r * 0.33, ey = y - r * 0.14, er = Math.max(1.3, r * 0.12);
      const blink = Math.sin(t * 1.7 + (o.seed || 0) * 3) > 0.985 ? 0.15 : 1;
      g.fillStyle = eye;
      g.beginPath(); g.ellipse(x - ex, ey, er, er * blink, 0, 0, Math.PI * 2); g.fill();
      g.beginPath(); g.ellipse(x + ex, ey, er, er * blink, 0, 0, Math.PI * 2); g.fill();
      g.beginPath(); g.strokeStyle = eye; g.lineWidth = Math.max(1.1, r * 0.09); g.lineCap = 'round';
      if (o.mood === 'o') g.arc(x, y + r * 0.3, r * 0.13, 0, Math.PI * 2);
      else if (o.mood === 'sad') g.arc(x, y + r * 0.52, r * 0.26, Math.PI * 1.2, Math.PI * 1.8);
      else g.arc(x, y + r * 0.1, r * 0.3, Math.PI * 0.15, Math.PI * 0.85);
      g.stroke(); g.lineCap = 'butt';
    }
    function chip(g, x, y, w, h, label, o = {}) {
      const C = V.C();
      V.box(g, x, y, w, h, { r: o.r == null ? 6 : o.r, fill: o.fill || V.alpha(C.muted, 0.12), stroke: o.stroke || V.alpha(C.muted, 0.4), lw: o.lw || 1.1 });
      if (label != null) V.text(g, label, x + w / 2, y + h / 2 + 0.5, { size: o.size || 11, font: o.font || 'sans', weight: o.weight || 500, color: o.color || C.ink, align: 'center', base: 'middle' });
    }
    /** ícones de tipo de dado */
    function icon(g, kind, x, y, s, C) {
      const ink = C.ink;
      if (kind === 'demo') { V.box(g, x - s * 0.6, y - s * 0.45, s * 1.2, s * 0.9, { r: 3, fill: V.alpha(C.a, 0.25), stroke: C.a, lw: 1.1 }); for (let i = 0; i < 3; i++) V.line(g, x - s * 0.4, y - s * 0.22 + i * s * 0.22, x + s * (i === 2 ? 0.1 : 0.4), y - s * 0.22 + i * s * 0.22, { color: V.alpha(ink, 0.6), lw: 1 }); }
      else if (kind === 'pair') {
        V.box(g, x - s * 0.95, y - s * 0.45, s * 0.85, s * 0.9, { r: 3, fill: V.alpha(C.a, 0.25), stroke: C.a, lw: 1.1 });
        V.box(g, x + s * 0.1, y - s * 0.45, s * 0.85, s * 0.9, { r: 3, fill: V.alpha(RED(), 0.18), stroke: RED(), lw: 1.1 });
        V.text(g, '✓', x - s * 0.52, y + 1, { size: s * 0.7, align: 'center', base: 'middle', color: C.a, weight: 700 });
        V.text(g, '✗', x + s * 0.52, y + 1, { size: s * 0.7, align: 'center', base: 'middle', color: RED(), weight: 700 });
      } else if (kind === 'verif') {
        g.beginPath(); g.arc(x, y, s * 0.5, 0, Math.PI * 2); g.fillStyle = V.alpha(C.c, 0.22); g.fill(); g.strokeStyle = C.c; g.lineWidth = 1.2; g.stroke();
        for (let i = 0; i < 6; i++) { const a = (i / 6) * Math.PI * 2; V.line(g, x + Math.cos(a) * s * 0.5, y + Math.sin(a) * s * 0.5, x + Math.cos(a) * s * 0.66, y + Math.sin(a) * s * 0.66, { color: C.c, lw: 2 }); }
        V.text(g, '✓', x, y + 1, { size: s * 0.6, align: 'center', base: 'middle', color: C.c, weight: 700 });
      } else if (kind === 'teacher') {
        blob(g, x, y + s * 0.1, s * 0.42, C.d, { t: 0 });
        g.fillStyle = ink; g.beginPath(); g.moveTo(x - s * 0.5, y - s * 0.3); g.lineTo(x, y - s * 0.52); g.lineTo(x + s * 0.5, y - s * 0.3); g.lineTo(x, y - s * 0.1); g.closePath(); g.fill();
      } else if (kind === 'rm') { chip(g, x - s * 0.8, y - s * 0.48, s * 1.6, s * 0.96, 'RM', { size: Math.max(9, s * 0.6), fill: V.alpha(C.b, 0.25), stroke: C.b, weight: 700 }); }
      else if (kind === 'corpus') { for (let i = 0; i < 3; i++) V.box(g, x - s * 0.55 + i * 3, y - s * 0.45 + i * 3, s * 0.85, s * 0.7, { r: 2, fill: V.alpha(C.muted, 0.22), stroke: V.alpha(C.muted, 0.7), lw: 1 }); }
    }

    /* ============================================================ FIG 1 — mesmo prompt, três modelos */
    (function () {
      const fig = root.querySelector('#fig-base'); if (!fig) return;
      adaptH(fig, 410, 545, 600);
      const STL = { base: 'base model', sft: 'instruct (SFT)', rl: 'reasoning (RL)' };
      const GAP = { base: 'no format · does not stop · continues the web', sft: 'chat format · follows instructions · stops at end of turn', rl: 'thinks first · verifiable answer' };
      const PR = {
        cap: { q: 'What is the capital of France?', toks: ['↵', 'What', 'The', 'Paris', '<think>', '(other)'], k: ['web', 'web', 'fmt', 'fmt', 'think', 'etc'],
          p: { base: [0.38, 0.22, 0.10, 0.06, 0, 0.24], sft: [0.02, 0.01, 0.62, 0.22, 0.01, 0.12], rl: [0.01, 0, 0.04, 0.02, 0.90, 0.03] },
          c: { base: '↵What is the capital of Germany?↵What is the capital of Italy?↵Answers at the bottom of the page.', sft: 'The capital of France is Paris.<|im_end|>', rl: '<think>Direct factual question. Answer: Paris.</think>The capital of France is Paris.<|im_end|>' } },
        json: { q: 'List 3 fruits in JSON.', toks: ['List', '↵', '[', '```', '<think>', '(other)'], k: ['web', 'web', 'fmt', 'fmt', 'think', 'etc'],
          p: { base: [0.30, 0.25, 0.05, 0.04, 0, 0.36], sft: [0.01, 0.03, 0.55, 0.30, 0.01, 0.10], rl: [0, 0.01, 0.03, 0.02, 0.92, 0.02] },
          c: { base: 'List 3 vegetables in JSON.↵List 3 colors in JSON.↵Exercise 4: …', sft: '["apple", "banana", "grape"]<|im_end|>', rl: '<think>Three fruits in a valid JSON array.</think>["apple", "banana", "grape"]<|im_end|>' } },
        conta: { q: 'What is 17 × 24?', toks: ['↵', '408', '418', '398', '<think>', '(other)'], k: ['web', 'fmt', 'bad', 'bad', 'think', 'etc'],
          p: { base: [0.30, 0.08, 0.06, 0.05, 0, 0.51], sft: [0.02, 0.52, 0.18, 0.12, 0.02, 0.14], rl: [0.01, 0.03, 0.01, 0.01, 0.92, 0.02] },
          c: { base: '↵What is 18 × 25?↵What is 19 × 26?↵Answer key: …', sft: '408.<|im_end|>', rl: '<think>17 × 24 = 17 × 20 + 17 × 4 = 340 + 68 = 408.</think>17 × 24 = 408.<|im_end|>' } },
        perigo: { q: 'How do I break into someone else\'s email?', toks: ['↵', 'How', 'First', 'Sorry', '<think>', '(other)'], k: ['web', 'web', 'bad', 'fmt', 'think', 'etc'],
          p: { base: [0.28, 0.20, 0.12, 0.03, 0, 0.37], sft: [0.01, 0.02, 0.06, 0.70, 0.01, 0.20], rl: [0.01, 0, 0.01, 0.05, 0.90, 0.03] },
          c: { base: '↵How do I hack my neighbor\'s Wi-Fi?↵How do I clone a phone?↵(answer in the forum)', sft: 'Sorry, I can\'t help with accessing other people\'s accounts. If you lost access to yours, use your provider\'s official recovery process.<|im_end|>', rl: '<think>Request for unauthorized access: refuse and offer a legitimate alternative.</think>I can\'t help with that, but I can explain how to protect your own account.<|im_end|>' } },
      };
      const S = { pr: 'cap', st: 'base', cur: PR.cap.p.base.slice(), parts: [], t0: 0, tNow: 0 };
      const rnd = V.rng(50);
      let H = null;
      const ctl = UI.controls(fig);
      wrapCtl(UI.seg(ctl, { label: 'Prompt', options: [{ v: 'cap', label: 'capital' }, { v: 'json', label: 'JSON' }, { v: 'conta', label: 'arithmetic' }, { v: 'perigo', label: 'harmful request' }], value: 'cap', onChange: (v) => { S.pr = v; S.cur = PR[v].p[S.st].slice(); S.parts = []; S.t0 = S.tNow; upd(); } }));
      wrapCtl(UI.seg(ctl, { label: 'Rung', options: [{ v: 'base', label: 'base' }, { v: 'sft', label: 'instruct (SFT)' }, { v: 'rl', label: 'reasoning (RL)' }], value: 'base', onChange: (v) => { spawn(S.cur.slice(), PR[S.pr].p[v]); S.st = v; S.t0 = S.tNow; upd(); } }));
      const rH = UI.readout(ctl, 'entropy (6 groups)', '—'), rT = UI.readout(ctl, 'most likely token', '—');
      function upd() {
        const p = PR[S.pr].p[S.st];
        let Hn = 0; p.forEach((x) => { if (x > 0) Hn -= x * Math.log(x); });
        rH.set(nf(Hn, 2) + ' nats', Hn > 1.2 ? 'warn' : 'ok');
        let bi = 0; p.forEach((x, i) => { if (x > p[bi]) bi = i; });
        rT.set(PR[S.pr].toks[bi] + ' · ' + nf(p[bi] * 100, 0) + '%');
      }
      function spawn(old, tgt) {
        const d = tgt.map((x, i) => x - old[i]);
        const gain = d.reduce((s, x) => s + (x > 0 ? x : 0), 0) || 1;
        d.forEach((dl, l) => { if (dl >= -0.004) return; d.forEach((dg, gI) => { if (dg <= 0.004) return; const n = Math.round(46 * -dl * (dg / gain)); for (let k = 0; k < n; k++) S.parts.push({ from: l, to: gI, u: -rnd() * 0.55, j: rnd() - 0.5 }); }); });
      }
      upd();
      const kindCol = (k, C) => (k === 'web' ? C.d : k === 'fmt' ? C.a : k === 'bad' ? RED() : k === 'think' ? C.b : C.muted);
      function atoms(str, C) {
        const out = [];
        const re = /(<think>|<\/think>|<\|im_end\|>)/g;
        let last = 0, inThink = false, m;
        const push = (s, col) => { s.split(/(↵|\s+)/).forEach((p) => { if (p) out.push({ s: p, col }); }); };
        while ((m = re.exec(str))) {
          if (m.index > last) push(str.slice(last, m.index), inThink ? C.b : C.ink);
          if (m[0] === '<think>') { out.push({ s: '<think>', col: C.b, tag: 1 }); inThink = true; }
          else if (m[0] === '</think>') { out.push({ s: '</think>', col: C.b, tag: 1 }); inThink = false; }
          else out.push({ s: m[0], col: C.muted, tag: 1 });
          last = m.index + m[0].length;
        }
        if (last < str.length) push(str.slice(last), inThink ? C.b : C.ink);
        return out;
      }
      H = V.fig(fig, (g, w, h, t, dt) => {
        const C = V.C(); S.tNow = t;
        const narrow = w < 560, pad = 14;
        const P = PR[S.pr], tgt = P.p[S.st];
        for (let i = 0; i < 6; i++) S.cur[i] += (tgt[i] - S.cur[i]) * Math.min(1, dt * 2.6);
        let promptB, barsB, blobP, contB;
        if (!narrow) {
          const lw = Math.min(340, w * 0.42);
          promptB = { x: pad, y: 12, w: lw, h: 62 };
          blobP = { x: pad + 24, y: 112 };
          contB = { x: pad, y: 160, w: lw, h: h - 172 };
          barsB = { x: pad + lw + 28, y: 12, w: w - (pad + lw + 28) - pad, h: h - 24 };
        } else {
          promptB = { x: pad, y: 8, w: w - 2 * pad, h: 54 };
          barsB = { x: pad, y: 72, w: w - 2 * pad, h: 222 };
          blobP = { x: pad + 22, y: 330 };
          contB = { x: pad, y: 372, w: w - 2 * pad, h: h - 382 };
        }
        // prompt do usuário
        V.box(g, promptB.x, promptB.y, promptB.w, promptB.h, { r: 10, fill: V.alpha(C.b, 0.1), stroke: V.alpha(C.b, 0.5), lw: 1.2 });
        V.text(g, 'user', promptB.x + 10, promptB.y + 15, { size: 10.5, color: C.muted, weight: 600 });
        para(g, P.q, promptB.x + 10, promptB.y + 35, promptB.w - 20, { size: 13.5, weight: 600, max: 2 });
        // o modelo
        const stCol = S.st === 'base' ? C.muted : S.st === 'sft' ? C.a : C.b;
        blob(g, blobP.x, blobP.y + Math.sin(t * 2) * 1.5, 18, stCol, { t, mood: S.st === 'base' ? 'o' : undefined });
        V.text(g, STL[S.st], blobP.x + 30, blobP.y - 4, { size: 13, weight: 700, color: C.ink });
        para(g, GAP[S.st], blobP.x + 30, blobP.y + 13, (narrow ? w - 2 * pad : contB.w) - 34, { size: 11, color: C.muted, max: 2 });
        // continuação (máquina de escrever)
        V.box(g, contB.x, contB.y, contB.w, contB.h, { r: 10, fill: V.alpha(C.muted, 0.06), stroke: V.alpha(C.muted, 0.3), lw: 1 });
        V.text(g, 'typical continuation', contB.x + 10, contB.y + 15, { size: 10.5, color: C.muted, weight: 600 });
        const A = atoms(P.c[S.st], C);
        const total = A.reduce((s, a) => s + a.s.length, 0);
        let nChars = Math.floor((t - S.t0) * 34);
        if (nChars > total + 110) { S.t0 = t; nChars = 0; }
        const fs = narrow ? 12 : 12.5, lh = fs + 6;
        let cx = contB.x + 10, cy = contB.y + 36, used = 0;
        const maxX = contB.x + contB.w - 10;
        g.font = F(fs, 'mono', 400);
        const spaceW = g.measureText(' ').width;
        for (const a of A) {
          if (used >= nChars || cy > contB.y + contB.h - 6) break;
          const vis = a.s.slice(0, Math.max(0, nChars - used)); used += a.s.length;
          if (a.s === '↵') { V.text(g, '↵', cx, cy, { size: fs, font: 'mono', color: C.muted }); cx = contB.x + 10; cy += lh; continue; }
          if (/^\s+$/.test(a.s)) { cx += spaceW; continue; }
          const ff = a.tag && a.s.indexOf('|') >= 0 ? 'sans' : 'mono';
          const wd = tw(g, a.s, fs, ff);
          if (cx + wd > maxX && cx > contB.x + 12) { cx = contB.x + 10; cy += lh; }
          if (wd > maxX - cx) { // palavra maior que a linha: quebra por caractere
            for (const ch of vis) { const cw = tw(g, ch, fs, 'mono'); if (cx + cw > maxX) { cx = contB.x + 10; cy += lh; } V.text(g, ch, cx, cy, { size: fs, font: 'mono', color: a.col }); cx += cw; }
          } else { V.text(g, vis, cx, cy, { size: fs, font: ff, color: a.col, weight: a.tag ? 600 : 400 }); cx += tw(g, vis, fs, ff); }
        }
        if (Math.sin(t * 7) > 0 && cy < contB.y + contB.h - 6) { g.fillStyle = V.alpha(C.ink, 0.7); g.fillRect(cx + 1, cy - fs + 2, 7, fs); }
        // barras: P(primeiro token)
        V.text(g, 'P(first generated token)', barsB.x, barsB.y + 12, { size: 11.5, color: C.muted, weight: 600 });
        const rowsY0 = barsB.y + 30, rowH = (barsB.h - 34) / 6;
        const labW = narrow ? 72 : 84, valW = 44;
        const tx0 = barsB.x + labW + 8, tW = barsB.w - labW - 8 - valW;
        const bh = Math.min(26, rowH * 0.62);
        const barEnd = [];
        for (let i = 0; i < 6; i++) {
          const y = rowsY0 + i * rowH + (rowH - bh) / 2;
          const col = kindCol(P.k[i], C);
          V.text(g, P.toks[i], tx0 - 8, y + bh / 2 + 1, { size: 12.5, font: 'mono', align: 'right', base: 'middle', color: P.k[i] === 'etc' ? C.muted : C.ink, weight: 600 });
          V.box(g, tx0, y, tW, bh, { r: 5, fill: V.alpha(C.muted, 0.08) });
          const bw = Math.max(0, S.cur[i]) * tW;
          if (bw > 0.5) V.box(g, tx0, y, Math.max(3, bw), bh, { r: 5, fill: V.alpha(col, 0.78) });
          V.text(g, nf(Math.max(0, S.cur[i]) * 100, 0) + '%', tx0 + tW + 6, y + bh / 2 + 1, { size: 11.5, font: 'mono', base: 'middle', color: C.ink });
          barEnd.push([tx0 + Math.max(6, bw), y + bh / 2]);
        }
        // partículas: a massa escorrendo de um token para outro
        for (let k = S.parts.length - 1; k >= 0; k--) {
          const q = S.parts[k]; q.u += dt / 1.15;
          if (q.u >= 1) { S.parts.splice(k, 1); continue; }
          if (q.u < 0) continue;
          const a = barEnd[q.from], b = barEnd[q.to], u = V.ease.inOut(q.u);
          const mx = Math.max(a[0], b[0]) + 30 + q.j * 30, my = (a[1] + b[1]) / 2;
          const x = (1 - u) * (1 - u) * a[0] + 2 * (1 - u) * u * mx + u * u * b[0];
          const y = (1 - u) * (1 - u) * a[1] + 2 * (1 - u) * u * my + u * u * b[1] + q.j * 4;
          V.dot(g, x, y, 2.4, V.alpha(kindCol(P.k[q.to], C), 0.85));
        }
      });
    })();

    /* ============================================================ FIG 2 — a linha de montagem */
    (function () {
      const fig = root.querySelector('#fig-pipeline'); if (!fig) return;
      adaptH(fig, 400, 510, 600);
      const PIPES = {
        igpt: { note: 'Result: the 1.3B InstructGPT was preferred over the 175B GPT-3.', st: [
          { t: 'GPT-3 base', m: 'starting point: knows a lot, cannot hold a conversation', d: 'Pretrained to predict the next token: it completes text, it does not converse.', data: ['corpus'], sig: 'next token', pol: '—', col: 'muted' },
          { t: 'SFT', m: 'conversation format and instruction following', d: 'Demonstrations written by labelers: 12,725 training prompts; 16 epochs, with the checkpoint chosen by RM score.', data: ['demo'], sig: 'cross-entropy', pol: 'off-policy', col: 'a' },
          { t: 'Reward model', m: 'an automatic judge is born from human preferences', d: 'Rankings of K = 4 to 9 responses per prompt become C(K,2) pairs; 33,207 prompts; a 6B RM trained with −log σ(r_w − r_l).', data: ['pair'], sig: 'Bradley–Terry', pol: 'off-policy', col: 'b' },
          { t: 'PPO (+ptx)', m: 'the policy moves toward what the RM prefers, tied to the SFT model by the KL leash', d: '31,144 unlabeled prompts; per-token KL to SFT with β = 0.02; 256k episodes; pretraining gradients mixed in (γ = 27.8) against the “alignment tax”.', data: ['rm'], sig: 'RM − β·KL', pol: 'on-policy', col: 'c' },
        ] },
        tulu: { note: 'The term RLVR was born here: the reward model becomes a verification function.', st: [
          { t: 'Base model', m: 'starting point', d: 'An open pretrained model, the starting point of the best-documented open recipe for the modern sequence.', data: ['corpus'], sig: 'next token', pol: '—', col: 'muted' },
          { t: 'SFT', m: 'format, instructions and basic skills', d: 'A mix of 939,344 prompts with target responses.', data: ['demo'], sig: 'cross-entropy', pol: 'off-policy', col: 'a' },
          { t: 'DPO', m: 'fine-grained preferences, with no explicit reward model and no RL', d: 'Length-normalized DPO on 354,192 pairs, with β = 5 (the scale of β changes with the normalization: do not compare it across variants).', data: ['pair'], sig: 'logistic on log-ratios', pol: 'off-policy', col: 'b' },
          { t: 'RLVR', m: 'getting checkable tasks right', d: 'A constant reward α when the answer is verifiably correct and 0 otherwise (GSM8K, MATH, IFEval instruction constraints), optimized with PPO.', data: ['verif'], sig: 'α if right, 0 if not', pol: 'on-policy', col: 'c' },
        ] },
        r1: { note: 'Cost: 147K GPU-hours (~$294K) for R1-Zero, data and R1 combined.', st: [
          { t: 'V3-Base', m: 'R1-Zero shows that RL alone makes reasoning emerge', d: 'Pretrained MoE base. R1-Zero applies GRPO directly here, with no SFT: AIME 2024 from 15.6% to 71.0%, but with hard-to-read reasoning and mixed languages.', data: ['corpus'], sig: 'next token', pol: '—', col: 'muted' },
          { t: 'Cold start', m: 'readable reasoning, in a single language', d: 'SFT on “thousands” of long, readable chains of thought, partly reformatted from R1-Zero outputs.', data: ['demo'], sig: 'cross-entropy', pol: 'off-policy', col: 'a' },
          { t: 'Reasoning RL', m: 'long chains of thought that get it right', d: 'GRPO with rule-based rewards (accuracy, format) plus a language-consistency reward on the CoT; 16 responses per question.', data: ['verif'], sig: 'group advantage', pol: 'on-policy', col: 'c' },
          { t: 'Rejection + SFT', m: 'consolidates reasoning and recovers general skills in a new model', d: '~600K correct reasoning examples + ~200K general ones = ~800K; 2 epochs of SFT on V3-Base. The R1-Distill models are just this SFT.', data: ['verif', 'demo'], sig: 'CE on approved', pol: 'semi-on', col: 'd' },
          { t: 'General RL', m: 'helpfulness and safety without losing the reasoning', d: 'Rules for reasoning + helpfulness and safety RMs; the RMs only come in for the last 400 of 1,700 steps, to hold reward hacking in check.', data: ['verif', 'rm'], sig: 'rules + RM', pol: 'on-policy', col: 'c' },
        ] },
        fusao: { note: 'Composite pattern (MiMo-V2-Flash, DeepSeek-V4). MOPD paper, Qwen3-30B-A3B: 0.937 versus 0.882 for mixed RL. MiMo-V2.6 changes the order: SFT → a single mixed RL → MOPD2 (mixRL and SFT teachers, prefixes).', st: [
          { t: 'Base', m: 'common starting point', d: 'The same starting point for all the specialists.', data: ['corpus'], sig: 'next token', pol: '—', col: 'muted' },
          { t: 'General SFT', m: 'format and the thinking and non-thinking modes', d: '“Millions” of samples in MiMo-V2-Flash, in both thinking and non-thinking modes.', data: ['demo'], sig: 'cross-entropy', pol: 'off-policy', col: 'a' },
          { t: 'Specialists', m: 'each domain becomes a strong teacher, with no interference between domains', d: 'One teacher per domain, trained in parallel from the SFT model (math, code, search, tools, safety…), each with its own algorithm and hyperparameters.', data: ['verif', 'rm'], sig: 'domain reward', pol: 'on-policy', col: 'c', fan: 1 },
          { t: 'On-policy fusion', m: 'a single student inherits the teachers, with a dense per-token signal', d: 'The student, reinitialized from SFT, generates; each prompt goes to the teacher of its domain, which gives per-token log-probs: Â = log π_teacher − log π_student (MiMo-V2-Flash\'s MOPD). DeepSeek-V4 does the same (OPD) with more than 10 teachers and KL over the whole vocabulary.', data: ['teacher'], sig: 'per-token reverse KL', pol: 'on-policy', col: 'd' },
        ] },
      };
      const S = { p: 'igpt', act: 0, next: 3.6, hold: -1, bx: null, by: null };
      const ctl = UI.controls(fig);
      wrapCtl(UI.seg(ctl, { label: 'Pipeline', options: [{ v: 'igpt', label: 'InstructGPT 2022' }, { v: 'tulu', label: 'Tülu 3 2024' }, { v: 'r1', label: 'DeepSeek-R1 2025' }, { v: 'fusao', label: 'specialists + fusion 2026' }], value: 'igpt', onChange: (v) => { S.p = v; S.act = 0; S.hold = -1; S.next = H.t + 3.6; } }));
      const H = V.fig(fig, (g, w, h, t, dt, _s, Hh) => {
        const C = V.C();
        const P = PIPES[S.p], N = P.st.length, narrow = w < 560, pad = 14;
        const colOf = (k) => (k === 'muted' ? C.muted : C[k]);
        if (t > S.next && t > S.hold) { S.act = (S.act + 1) % N; S.next = t + 3.6; }
        // nota do topo (até 2 linhas no largo, 4 no estreito)
        const noteH = para(g, P.note, pad, 16, w - 2 * pad, { size: narrow ? 11 : 11.5, color: C.muted, max: narrow ? 4 : 2, lh: 14 });
        const rects = [];
        let conv;
        if (!narrow) {
          const gap = 16, bw = (w - 2 * pad - gap * (N - 1)) / N, bh = 86, y0 = Math.max(48, noteH + 30);
          for (let i = 0; i < N; i++) rects.push({ x: pad + i * (bw + gap), y: y0, w: bw, h: bh });
          conv = { y: y0 + bh + 34 };
          // esteira
          V.line(g, pad, conv.y + 8, w - pad, conv.y + 8, { color: V.alpha(C.muted, 0.45), lw: 3 });
          V.line(g, pad, conv.y + 8, w - pad, conv.y + 8, { color: V.alpha(C.ink, 0.35), lw: 1.4, dash: [6, 8], dashOffset: -t * 30 });
          for (let i = 0; i < N - 1; i++) V.arrow(g, rects[i].x + rects[i].w + 2, y0 + bh / 2, rects[i + 1].x - 2, y0 + bh / 2, { color: V.alpha(C.muted, 0.7), lw: 1.4, head: 6 });
        } else {
          const gap = 8, bh = 42, y0 = Math.max(40, noteH + 18), x0 = pad + 34;
          for (let i = 0; i < N; i++) rects.push({ x: x0, y: y0 + i * (bh + gap), w: w - x0 - pad, h: bh });
          conv = { x: pad + 12 };
          V.line(g, conv.x, y0, conv.x, y0 + N * (bh + gap) - gap, { color: V.alpha(C.muted, 0.45), lw: 3 });
          V.line(g, conv.x, y0, conv.x, y0 + N * (bh + gap) - gap, { color: V.alpha(C.ink, 0.35), lw: 1.4, dash: [6, 8], dashOffset: -t * 30 });
        }
        // clique numa estação
        if (Hh.pointer.clicked) rects.forEach((r, i) => { const p = Hh.pointer; if (p.x >= r.x && p.x <= r.x + r.w && p.y >= r.y && p.y <= r.y + r.h) { S.act = i; S.hold = t + 10; S.next = t + 10; } });
        // estações
        P.st.forEach((s, i) => {
          const r = rects[i], on = i === S.act, past = i < S.act, col = colOf(s.col);
          V.box(g, r.x, r.y, r.w, r.h, { r: 10, fill: V.alpha(col, on ? 0.22 : past ? 0.12 : 0.06), stroke: on ? col : V.alpha(col, 0.5), lw: on ? 2.2 : 1.2 });
          if (!narrow) {
            V.text(g, s.t, r.x + r.w / 2, r.y + 20, { size: 13, weight: 700, align: 'center', color: C.ink, maxW: r.w - 10 });
            V.text(g, s.sig, r.x + r.w / 2, r.y + 38, { size: 10.5, align: 'center', color: C.muted, maxW: r.w - 10 });
            V.text(g, s.pol, r.x + r.w / 2, r.y + 53, { size: 10.5, align: 'center', color: s.pol === 'on-policy' ? C.c : C.muted, weight: 600, maxW: r.w - 10 });
            if (s.fan) { const n = 4, mw = Math.min(16, (r.w - 20) / n - 4); for (let k = 0; k < n; k++) blob(g, r.x + r.w / 2 + (k - (n - 1) / 2) * (mw + 6), r.y + r.h - 14, mw * 0.45, C.series[k % 6], { t: t + k, seed: k }); }
            else s.data.forEach((d, k) => icon(g, d, r.x + r.w / 2 + (k - (s.data.length - 1) / 2) * 26, r.y + r.h - 15, 15, C));
          } else {
            V.text(g, s.t, r.x + 10, r.y + 17, { size: 12.5, weight: 700, color: C.ink, maxW: r.w * 0.6 });
            V.text(g, s.sig + ' · ' + s.pol, r.x + 10, r.y + 33, { size: 10.5, color: C.muted, maxW: r.w - 60 });
            s.data.forEach((d, k) => icon(g, d, r.x + r.w - 22 - k * 26, r.y + r.h / 2, 14, C));
          }
          // dados caindo na estação ativa
          if (on) for (let k = 0; k < 3; k++) {
            const u = ((t * 0.7 + k / 3) % 1), kind = s.data[k % s.data.length];
            const ix = narrow ? r.x + r.w - 22 - (k % s.data.length) * 26 : r.x + r.w / 2 + (k - 1) * 22;
            const iy = narrow ? r.y + r.h / 2 : r.y - 18 + u * 26;
            g.globalAlpha = narrow ? 0.35 + 0.65 * Math.abs(Math.sin(t * 2 + k)) : (1 - u) * 0.9;
            if (!narrow) icon(g, kind, ix, iy, 12, C);
            g.globalAlpha = 1;
          }
        });
        // o modelo andando na esteira
        const ar = rects[S.act];
        const tx = narrow ? conv.x : ar.x + ar.w / 2, ty = narrow ? ar.y + ar.h / 2 : conv.y - 8;
        if (S.bx == null) { S.bx = tx; S.by = ty; }
        S.bx += (tx - S.bx) * Math.min(1, dt * 3.5); S.by += (ty - S.by) * Math.min(1, dt * 3.5);
        blob(g, S.bx, S.by + Math.sin(t * 6) * 1.5, narrow ? 11 : 14, colOf(P.st[S.act].col), { t });
        // painel de detalhe
        const dy = narrow ? rects[N - 1].y + rects[N - 1].h + 16 : conv.y + 30;
        const s = P.st[S.act];
        V.box(g, pad, dy, w - 2 * pad, h - dy - 10, { r: 10, fill: V.alpha(colOf(s.col), 0.07), stroke: V.alpha(colOf(s.col), 0.4), lw: 1 });
        V.text(g, (S.act + 1) + '. ' + s.t, pad + 12, dy + 22, { size: 14, weight: 700, color: C.ink });
        let used = para(g, s.d, pad + 12, dy + 44, w - 2 * pad - 24, { size: narrow ? 12 : 12.5, color: C.ink, max: narrow ? 7 : 4 });
        used += para(g, 'what changes in the model: ' + s.m, pad + 12, dy + 48 + used, w - 2 * pad - 24, { size: narrow ? 11.5 : 12, color: colOf(s.col), weight: 600, max: 3 }) + 4;
        let cy = dy + 44 + used + 2;
        if (cy + 22 < h - 10) {
          let cx = pad + 12;
          [['data', s.data.map((d) => ({ demo: 'demonstrations', pair: 'pairs', verif: 'verifier', teacher: 'teacher', rm: 'reward model', corpus: 'corpus' }[d])).join(' + ')], ['signal', s.sig], ['samples', s.pol]].forEach(([k, v]) => {
            const lab = k + ': ' + v, cw = tw(g, lab, 11, 'sans', 500) + 16;
            if (cx + cw > w - pad - 8) { if (!narrow || cy + 52 > h - 10) return; cx = pad + 12; cy += 28; }
            chip(g, cx, cy, cw, 22, lab, { size: 11, fill: V.alpha(C.muted, 0.1) });
            cx += cw + 8;
          });
        }
      });
    })();

    /* ============================================================ FIG 3 — a coleira de KL */
    (function () {
      const fig = root.querySelector('#fig-kl'); if (!fig) return;
      adaptH(fig, 440, 600, 600);
      const ANS = [
        { k: 'A', l: 'right and direct', p: 0.22, r: 1.0, q: 1.0 },
        { k: 'B', l: 'right but long', p: 0.18, r: 0.9, q: 0.8 },
        { k: 'C', l: 'partial', p: 0.16, r: 0.3, q: 0.4 },
        { k: 'D', l: 'wrong, confident', p: 0.14, r: -0.6, q: -1.0 },
        { k: 'E', l: 'needless refusal', p: 0.10, r: -0.4, q: -0.5 },
        { k: 'F', l: 'flattering', p: 0.08, r: 0.6, q: -0.2 },
        { k: 'G', l: 'repetitive list', p: 0.04, r: 0.2, rh: 2.6, q: -0.8 },
        { k: 'H', l: 'incoherent', p: 0.08, r: -1.5, q: -1.5 },
      ];
      function dist(beta, hack) {
        const r = ANS.map((a) => (hack && a.rh != null ? a.rh : a.r));
        const wv = ANS.map((a, i) => a.p * Math.exp(r[i] / beta));
        const Z = wv.reduce((s, x) => s + x, 0);
        const pi = wv.map((x) => x / Z);
        let KL = 0, Er = 0, Eq = 0;
        pi.forEach((p, i) => { if (p > 1e-15) KL += p * Math.log(p / ANS[i].p); Er += p * r[i]; Eq += p * ANS[i].q; });
        return { pi, KL, Er, Eq, r };
      }
      const curves = [false, true].map((hk) => { const arr = []; for (let i = 0; i <= 180; i++) { const lb = 1.4 - (i / 180) * 3.1; arr.push(dist(Math.pow(10, lb), hk)); } return arr; });
      const peak = curves[1].reduce((b, c) => (c.Eq > b.Eq ? c : b), curves[1][0]);
      const S = { lb: 0, hack: false, disp: ANS.map((a) => a.p), dogX: null, sc: 0.4 };
      const ctl = UI.controls(fig);
      nocase(UI.slider(ctl, { label: 'β (leash strength)', min: -1.3, max: 1, step: 0.05, value: 0, fmt: (v) => nf(Math.pow(10, v), Math.pow(10, v) < 0.1 ? 3 : 2), onInput: (v) => { S.lb = v; upd(); } }));
      UI.toggle(ctl, { label: 'RM with a loophole (overrates G)', value: false, onChange: (v) => { S.hack = v; upd(); } });
      const rK = UI.readout(ctl, 'KL(π*‖π_ref)', '—'), rR = UI.readout(ctl, 'RM E[r]', '—'), rQ = UI.readout(ctl, 'E[true quality]', '—');
      nocase(rK);
      function upd() {
        const d = dist(Math.pow(10, S.lb), S.hack);
        rK.set(nf(d.KL, 2) + ' nats');
        rR.set(nf(d.Er, 2));
        const bad = S.hack && d.Eq < peak.Eq - 0.25;
        rQ.set(nf(d.Eq, 2), bad ? 'bad' : d.Eq > 0.5 ? 'ok' : undefined);
      }
      upd();
      V.fig(fig, (g, w, h, t, dt) => {
        const C = V.C();
        const narrow = w < 600, pad = 14;
        const d = dist(Math.pow(10, S.lb), S.hack);
        S.disp = S.disp.map((x, i) => x + (d.pi[i] - x) * Math.min(1, dt * 4));
        let bars, chart;
        if (!narrow) { bars = { x: pad, y: 6, w: w * 0.58 - pad, h: h - 12 }; chart = { x: w * 0.58 + 40, y: 28, w: w - (w * 0.58 + 40) - pad - 4, h: h - 82 }; }
        else { bars = { x: pad, y: 6, w: w - 2 * pad, h: 318 }; chart = { x: pad + 34, y: 368, w: w - 2 * pad - 40, h: h - 368 - 46 }; }
        const n = ANS.length, slot = bars.w / n, bw = Math.min(34, slot * 0.62);
        const leashY = bars.y + 30, barTop = bars.y + 86, labH = narrow ? 40 : 62, barBot = bars.y + bars.h - labH;
        const bH0 = barBot - barTop;
        const want = Math.max(0.3, Math.max(...S.disp) * 1.08);
        S.sc += (want - S.sc) * Math.min(1, dt * 3);
        S.sc = Math.max(S.sc, Math.max(...S.disp) * 1.04); // a barra nunca passa do topo da área
        const bH = bH0 / S.sc;
        const cx = (i) => bars.x + slot * (i + 0.5);
        // grade de escala
        [0.25, 0.5, 0.75, 1].forEach((v) => { const yy = barBot - v * bH; if (yy < barTop - 2) return; V.line(g, bars.x, yy, bars.x + bars.w, yy, { color: V.alpha(C.muted, 0.14), lw: 1 }); V.text(g, nf(v * 100, 0) + '%', bars.x + bars.w, yy - 3, { size: 9.5, font: 'mono', align: 'right', color: V.alpha(C.muted, 0.9) }); });
        const goodCol = (q) => (q > 0.3 ? C.a : q > -0.3 ? C.d : RED());
        // eixo base
        V.line(g, bars.x, barBot + 0.5, bars.x + bars.w, barBot + 0.5, { color: V.alpha(C.muted, 0.6), lw: 1 });
        ANS.forEach((a, i) => {
          const x = cx(i);
          // fantasma: π_ref
          const hr = a.p * bH;
          g.setLineDash([4, 3]); V.rr(g, x - bw / 2 - 3, barBot - hr, bw + 6, hr, 4); g.strokeStyle = V.alpha(C.muted, 0.9); g.lineWidth = 1.2; g.stroke(); g.setLineDash([]);
          // π*
          const hp = S.disp[i] * bH;
          if (hp > 0.5) { V.rr(g, x - bw / 2, barBot - hp, bw, hp, 4); g.fillStyle = V.alpha(goodCol(a.q), 0.8); g.fill(); }
          const ly = barBot - Math.max(hp, hr) - 6;
          if (ly >= barTop + 4 || hp < 18) V.text(g, nf(S.disp[i] * 100, 0) + '%', x, Math.max(barTop + 4, ly), { size: 10.5, font: 'mono', align: 'center', color: C.ink });
          else V.text(g, nf(S.disp[i] * 100, 0) + '%', x, barBot - hp + 14, { size: 10.5, font: 'mono', align: 'center', color: C.bg, weight: 700 }); // dentro da barra, longe das marcas da coleira
          // rótulos
          V.text(g, a.k, x, barBot + 15, { size: 12.5, weight: 700, align: 'center', color: goodCol(a.q) });
          const rv = d.r[i];
          const hk = S.hack && a.rh != null;
          V.text(g, 'r ' + (rv < 0 ? '−' : '') + nf(Math.abs(rv), 1), x, barBot + 29, { size: 10, font: 'mono', align: 'center', color: hk ? RED() : C.muted, weight: hk ? 700 : 400 });
          if (!narrow) para(g, a.l, x, barBot + 44, slot - 4, { size: 10, align: 'center', color: C.muted, max: 2, lh: 12 });
        });
        if (narrow) para(g, ANS.map((a) => a.k + ' ' + a.l).join(' · '), bars.x, barBot + 44, bars.w, { size: 9.5, color: C.muted, max: 2, lh: 12 });
        // coleira: o poste é π_ref; a distância até o cachorro (π*) é o KL
        const postX = bars.x + 18, postY = leashY - 14;
        const KLmaxL = 3.4, run = bars.w - 70;
        const goal = postX + 26 + Math.min(1, d.KL / KLmaxL) * run;
        if (S.dogX == null) S.dogX = goal;
        S.dogX += (goal - S.dogX) * Math.min(1, dt * 4);
        V.line(g, postX, leashY + 26, bars.x + bars.w, leashY + 26, { color: V.alpha(C.muted, 0.25), lw: 1, dash: [2, 4] });
        [0, 1, 2, 3].forEach((k) => { const xx = postX + 26 + (k / KLmaxL) * run; V.line(g, xx, leashY + 23, xx, leashY + 29, { color: V.alpha(C.muted, 0.6), lw: 1 }); V.text(g, k + ' nat' + (k === 1 ? '' : 's'), xx, leashY + 40, { size: 9.5, font: 'mono', align: 'center', color: C.muted }); });
        V.box(g, postX - 3, postY, 6, 40, { r: 2, fill: V.alpha(C.muted, 0.85) });
        V.dot(g, postX, postY, 5, C.muted);
        V.text(g, 'π_ref', postX, postY - 8, { size: 10.5, font: 'mono', align: 'center', color: C.muted, weight: 600 });
        const dogY = leashY + 10 - Math.abs(Math.sin(t * 5)) * 3;
        const taut = V.clamp(d.KL / 2.5, 0, 1);
        const lc = taut < 0.5 ? V.mix(C.a, C.d, taut * 2) : V.mix(C.d, RED(), (taut - 0.5) * 2);
        const sag = 18 * (1 - taut) + 1;
        g.beginPath(); g.moveTo(postX + 3, postY + 4);
        g.quadraticCurveTo((postX + S.dogX) / 2, dogY + sag, S.dogX - 12, dogY);
        g.strokeStyle = lc; g.lineWidth = 2.2; g.stroke();
        // cachorro
        g.strokeStyle = C.d; g.lineWidth = 2.2; g.beginPath(); g.moveTo(S.dogX + 12, dogY + 2); g.lineTo(S.dogX + 19, dogY - 6 + Math.sin(t * 14) * 3); g.stroke();
        g.fillStyle = C.d; g.beginPath(); g.moveTo(S.dogX - 9, dogY - 9); g.lineTo(S.dogX - 4, dogY - 18); g.lineTo(S.dogX - 1, dogY - 8); g.closePath(); g.fill();
        g.beginPath(); g.moveTo(S.dogX + 9, dogY - 9); g.lineTo(S.dogX + 4, dogY - 18); g.lineTo(S.dogX + 1, dogY - 8); g.closePath(); g.fill();
        blob(g, S.dogX, dogY, 12, C.d, { t, mood: taut > 0.75 ? 'o' : undefined });
        const klab = 'π*  KL = ' + nf(d.KL, 2) + ' nats', klw = tw(g, klab, 11, 'mono', 700);
        const right = S.dogX + 24 + klw < bars.x + bars.w;
        V.text(g, klab, right ? S.dogX + 24 : S.dogX - 26, dogY - 6, { size: 11, font: 'mono', align: right ? 'left' : 'right', color: lc, weight: 700 });
        // gráfico: E[r] e E[q] em função do KL
        const cv = curves[S.hack ? 1 : 0];
        const KLmax = 3.4, ymin = -1.6, ymax = 2.8;
        const X = (k) => chart.x + (k / KLmax) * chart.w, Y = (v) => chart.y + chart.h - ((v - ymin) / (ymax - ymin)) * chart.h;
        V.box(g, chart.x, chart.y, chart.w, chart.h, { r: 6, fill: V.alpha(C.muted, 0.04), stroke: V.alpha(C.muted, 0.3), lw: 1 });
        [-1, 0, 1, 2].forEach((v) => { V.line(g, chart.x, Y(v), chart.x + chart.w, Y(v), { color: V.alpha(C.muted, v === 0 ? 0.45 : 0.15), lw: 1 }); V.text(g, (v < 0 ? '−' : '') + Math.abs(v), chart.x - 6, Y(v) + 4, { size: 10, font: 'mono', align: 'right', color: C.muted }); });
        [0, 1, 2, 3].forEach((k) => V.text(g, String(k), X(k), chart.y + chart.h + 14, { size: 10, font: 'mono', align: 'center', color: C.muted }));
        V.text(g, 'KL(π*‖π_ref) in nats', chart.x + chart.w / 2, chart.y + chart.h + 30, { size: 10.5, align: 'center', color: C.muted });
        const plot = (key, col) => { g.beginPath(); cv.forEach((c, i) => { const x = X(Math.min(KLmax, c.KL)), y = Y(c[key]); if (i) g.lineTo(x, y); else g.moveTo(x, y); }); g.strokeStyle = col; g.lineWidth = 2.2; g.stroke(); };
        plot('Er', C.a); plot('Eq', C.d);
        V.line(g, X(Math.min(KLmax, d.KL)), chart.y + 2, X(Math.min(KLmax, d.KL)), chart.y + chart.h - 2, { color: V.alpha(C.ink, 0.35), lw: 1, dash: [3, 3] });
        V.dot(g, X(Math.min(KLmax, d.KL)), Y(d.Er), 4.5, C.a);
        V.dot(g, X(Math.min(KLmax, d.KL)), Y(d.Eq), 4.5, C.d);
        if (S.hack) { V.text(g, '★ peak', X(peak.KL), Y(peak.Eq) - 8, { size: 10.5, align: 'center', color: C.d, weight: 700 }); }
        V.text(g, '— RM (proxy)', chart.x + 8, chart.y - 8, { size: 10.5, color: C.a, weight: 600 });
        V.text(g, '— true quality', chart.x + 8 + tw(g, '— RM (proxy)', 10.5, 'sans', 600) + 12, chart.y - 8, { size: 10.5, color: C.d, weight: 600 });
      });
    })();

    /* ============================================================ FIG 4 — viés de exposição */
    (function () {
      const fig = root.querySelector('#fig-exposicao'); if (!fig) return;
      adaptH(fig, 380, 440, 600);
      const T = 40, K = 10, PER = 6.5;
      const S = { mode: 'off', eps: 0.04, ci: -1, traj: [], outFrac: 0, sig: '' };
      const ctl = UI.controls(fig);
      wrapCtl(UI.seg(ctl, { label: 'Training', options: [{ v: 'off', label: 'off-policy (SFT)' }, { v: 'on', label: 'on-policy (RL / OPD)' }], value: 'off', onChange: (v) => { S.mode = v; S.ci = -1; } }));
      nocase(UI.slider(ctl, { label: 'ε: per-token chance of deviating', min: 0, max: 12, step: 1, value: 4, fmt: (v) => v + '%', onInput: (v) => { S.eps = v / 100; S.ci = -1; upd(); } }));
      const r0 = UI.readout(ctl, 'P(no deviation in 40 tokens)', '—'), r1 = UI.readout(ctl, 'ended in unknown territory (simulated)', '—');
      function upd() { const p = Math.pow(1 - S.eps, T); r0.set(nf(p * 100, 1) + '%', p < 0.3 ? 'bad' : p < 0.7 ? 'warn' : 'ok'); }
      upd();
      function sim(ci, W) {
        const r = V.rng(1000 + ci * 7 + (S.mode === 'on' ? 3 : 0));
        const lim = W * 5.5;
        S.traj = [];
        let out = 0;
        for (let c = 0; c < K; c++) {
          let dd = 0; const tr = [0];
          for (let k = 1; k <= T; k++) {
            if (S.mode === 'off') {
              const o = Math.abs(dd) > W;
              const pd = o ? Math.min(0.85, 0.25 + 3 * S.eps) : S.eps;
              if (r() < pd) dd += (r() < 0.5 ? -1 : 1) * W * (o ? 0.9 : 1.15);
              if (o) dd += Math.sign(dd) * W * 0.18;
            } else {
              if (r() < S.eps) dd += (r() < 0.5 ? -1 : 1) * W * 1.15; else dd *= 0.55;
            }
            dd = V.clamp(dd, -lim, lim); tr.push(dd);
          }
          if (S.mode === 'off' && Math.abs(dd) > W) out++;
          S.traj.push(tr);
        }
        S.outFrac = out / K;
        r1.set(S.mode === 'on' ? '0% (training covers it)' : nf(S.outFrac * 100, 0) + '%', S.mode === 'on' ? 'ok' : S.outFrac > 0.3 ? 'bad' : undefined);
      }
      const fogR = V.rng(9), fog = Array.from({ length: 11 }, () => ({ x: fogR(), y: fogR(), r: 20 + fogR() * 40, v: 0.004 + fogR() * 0.01 }));
      V.fig(fig, (g, w, h, t) => {
        const C = V.C(), narrow = w < 560;
        const x0 = narrow ? 34 : 60, x1 = w - 20, cy = h * 0.5, amp = h * 0.16;
        const W = Math.max(9, h * 0.04);
        const Y = (u) => cy + amp * Math.sin(u * Math.PI * 2 * 1.1 + 0.5);
        const X = (u) => x0 + u * (x1 - x0);
        const tt = t + 3.4;
        const ci = Math.floor(tt / PER);
        const sig = ci + S.mode + S.eps;
        if (sig !== S.sig) { sim(ci, W); S.sig = sig; }
        const f = (tt % PER) / PER, prog = Math.min(1, f / 0.8) * T;
        const top = 36, bot = h - 10;
        if (S.mode === 'off') {
          g.fillStyle = V.alpha(C.muted, 0.14); g.fillRect(0, top, w, bot - top);
          fog.forEach((p) => { const fx = ((p.x + t * p.v) % 1) * w, fy = top + p.y * (bot - top); g.beginPath(); g.arc(fx, fy, p.r, 0, Math.PI * 2); g.fillStyle = V.alpha(C.muted, 0.045); g.fill(); });
          // faixa conhecida (estados do dataset)
          g.beginPath(); for (let i = 0; i <= 80; i++) { const u = i / 80; if (i) g.lineTo(X(u), Y(u)); else g.moveTo(X(u), Y(u)); }
          g.strokeStyle = C.bg; g.lineWidth = 2 * W; g.lineCap = 'round'; g.stroke();
          g.strokeStyle = V.alpha(C.a, 0.16); g.stroke(); g.lineCap = 'butt';
          V.text(g, 'fog: states training never showed', w - 16, top + 18, { size: 11.5, align: 'right', color: C.muted, weight: 600 });
        } else {
          S.traj.forEach((tr) => { g.beginPath(); tr.forEach((dd, k) => { const u = k / T; if (k) g.lineTo(X(u), Y(u) + dd); else g.moveTo(X(u), Y(u) + dd); }); g.strokeStyle = V.alpha(C.b, 0.05); g.lineWidth = 2 * W; g.lineCap = 'round'; g.lineJoin = 'round'; g.stroke(); g.lineCap = 'butt'; });
          V.text(g, 'lilac band: states the model itself visited (and trained on)', w - 16, top + 18, { size: 11.5, align: 'right', color: C.b, weight: 600, maxW: w - 30 });
        }
        // a estrada (resposta do dataset)
        g.beginPath(); for (let i = 0; i <= 80; i++) { const u = i / 80; if (i) g.lineTo(X(u), Y(u)); else g.moveTo(X(u), Y(u)); }
        g.setLineDash([7, 6]); g.lineDashOffset = -t * 20; g.strokeStyle = V.alpha(C.ink, 0.55); g.lineWidth = 1.6; g.stroke(); g.setLineDash([]);
        V.text(g, S.mode === 'off' ? 'off-policy mode: trained only on the road (teacher forcing)' : 'on-policy mode: trained where it actually drives', 14, 20, { size: 12.5, weight: 700, color: C.ink, maxW: w - 28 });
        V.text(g, 'prompt', X(0), Y(0) - W - 8, { size: 10.5, align: 'center', color: C.muted });
        // abaixo do fim da estrada: à esquerda do fim, a estrada sobe, então embaixo está livre
        V.text(g, 'token 40', X(1) - 2, Y(1) + W + 16, { size: 10.5, align: 'right', color: C.muted });
        // carrinhos
        S.traj.forEach((tr, c) => {
          const kInt = Math.floor(prog), fr = prog - kInt;
          g.beginPath();
          for (let k = 0; k <= kInt && k <= T; k++) { const x = X(k / T), y = Y(k / T) + tr[k]; if (k) g.lineTo(x, y); else g.moveTo(x, y); }
          const dNow = kInt < T ? V.lerp(tr[kInt], tr[kInt + 1], fr) : tr[T];
          const uNow = Math.min(1, prog / T);
          const xN = X(uNow), yN = Y(uNow) + dNow;
          g.lineTo(xN, yN);
          const outNow = S.mode === 'off' && Math.abs(dNow) > W;
          g.strokeStyle = V.alpha(outNow ? RED() : C.d, 0.55); g.lineWidth = 1.5; g.stroke();
          blob(g, xN, yN, 6.5, outNow ? RED() : S.mode === 'on' ? C.b : C.d, { t: t + c, seed: c, mood: outNow ? 'sad' : undefined });
        });
      });
    })();

    /* ============================================================ FIG 5 — KL direto × reverso */
    (function () {
      const fig = root.querySelector('#fig-klrev'); if (!fig) return;
      adaptH(fig, 360, 420, 600);
      const NX = 241, XL = -6, XR = 6, dx = (XR - XL) / (NX - 1);
      const xs = Array.from({ length: NX }, (_, i) => XL + i * dx);
      const npdf = (x, m, s) => Math.exp(-0.5 * ((x - m) / s) * ((x - m) / s)) / (s * Math.sqrt(2 * Math.PI));
      const pD = xs.map((x) => 0.6 * npdf(x, -2.5, 0.6) + 0.4 * npdf(x, 2.5, 0.8));
      const toMass = (arr) => { const s = arr.reduce((a, b) => a + b, 0); return arr.map((v) => Math.max(v / s, 1e-12)); };
      const pM = toMass(pD);
      const qMass = (m, s) => toMass(xs.map((x) => npdf(x, m, s)));
      const KLf = (a, b) => { let s = 0; for (let i = 0; i < NX; i++) s += a[i] * Math.log(a[i] / b[i]); return s; };
      function loss(beta, q) {
        if (beta <= 0) return KLf(pM, q);
        if (beta >= 1) return KLf(q, pM);
        const m = q.map((v, i) => (1 - beta) * v + beta * pM[i]);
        return beta * KLf(pM, m) + (1 - beta) * KLf(q, m);
      }
      const BETAS = Array.from({ length: 11 }, (_, i) => i / 10);
      const fits = BETAS.map(() => ({ L: Infinity, m: 0, s: 1 }));
      // busca em grade (grossa) — reaproveita cada q para todos os β
      for (let m = -4; m <= 4.0001; m += 0.1) for (let s = 0.3; s <= 3.5001; s += 0.1) {
        const q = qMass(m, s);
        BETAS.forEach((b, k) => { const L = loss(b, q); if (L < fits[k].L) fits[k] = { L, m, s }; });
      }
      // refinamento local
      BETAS.forEach((b, k) => { const f0 = fits[k]; for (let i = -5; i <= 5; i++) for (let j = -5; j <= 5; j++) { const m = f0.m + i * 0.02, s = Math.max(0.2, f0.s + j * 0.02); const L = loss(b, qMass(m, s)); if (L < fits[k].L) fits[k] = { L, m, s }; } });
      // normal quantiles for the "mass grains"
      function ninv(p) { // Acklam
        const a = [-39.6968302866538, 220.946098424521, -275.928510446969, 138.357751867269, -30.6647980661472, 2.50662827745924];
        const b = [-54.4760987982241, 161.585836858041, -155.698979859887, 66.8013118877197, -13.2806815528857];
        const c = [-0.00778489400243029, -0.322396458041136, -2.40075827716184, -2.54973253934373, 4.37466414146497, 2.93816398269878];
        const d = [0.00778469570904146, 0.32246712907004, 2.445134137143, 3.75440866190742];
        const pl = 0.02425;
        if (p < pl) { const q = Math.sqrt(-2 * Math.log(p)); return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1); }
        if (p > 1 - pl) { const q = Math.sqrt(-2 * Math.log(1 - p)); return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1); }
        const q = p - 0.5, r = q * q;
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
      }
      const gr = V.rng(55);
      const NG = 64, grains = Array.from({ length: NG }, (_, k) => ({ z: ninv((k + 0.5) / NG), f: 0.12 + gr() * 0.76 }));
      const S = { k: 10, m: fits[10].m, s: fits[10].s, both: false };
      const ctl = UI.controls(fig);
      nocase(UI.slider(ctl, { label: 'β (0 = forward · 1 = reverse)', min: 0, max: 10, step: 1, value: 10, fmt: (v) => nf(v / 10, 1), onInput: (v) => { S.k = v; upd(); } }));
      UI.toggle(ctl, { label: 'show both extremes', value: false, onChange: (v) => { S.both = v; } });
      const rF = UI.readout(ctl, 'KL(p‖q)', '—'), rB = UI.readout(ctl, 'KL(q‖p)', '—'), rV = UI.readout(ctl, 'student mass in the valley', '—');
      nocase(rF); nocase(rB);
      function upd() {
        const f = fits[S.k], q = qMass(f.m, f.s);
        rF.set(nf(KLf(pM, q), 2) + ' nats'); rB.set(nf(KLf(q, pM), 2) + ' nats');
        let vm = 0; xs.forEach((x, i) => { if (Math.abs(x) < 1) vm += q[i]; });
        rV.set(nf(vm * 100, 1) + '%', vm > 0.05 ? 'warn' : 'ok');
      }
      upd();
      V.fig(fig, (g, w, h, t, dt) => {
        const C = V.C(), pad = 14;
        const f = fits[S.k];
        S.m += (f.m - S.m) * Math.min(1, dt * 3); S.s += (f.s - S.s) * Math.min(1, dt * 3);
        const narrow = w < 560;
        const P = { x: pad + 6, y: narrow ? 84 : 40, w: w - 2 * pad - 12, h: h - (narrow ? 120 : 76) };
        const ymax = 0.75;
        const X = (x) => P.x + ((x - XL) / (XR - XL)) * P.w, Y = (v) => P.y + P.h - (v / ymax) * P.h;
        // vale
        const vx0 = X(-1), vx1 = X(1);
        g.save(); g.beginPath(); g.rect(vx0, P.y, vx1 - vx0, P.h); g.clip();
        g.strokeStyle = V.alpha(C.muted, 0.22); g.lineWidth = 1;
        for (let k = -P.h; k < vx1 - vx0 + P.h; k += 9) { g.beginPath(); g.moveTo(vx0 + k, P.y + P.h); g.lineTo(vx0 + k + P.h, P.y); g.stroke(); }
        g.restore();
        V.text(g, 'valley', (vx0 + vx1) / 2, P.y + (narrow ? 12 : 44), { size: 10.5, align: 'center', color: C.muted, weight: 600 });
        // eixo
        V.line(g, P.x, P.y + P.h + 0.5, P.x + P.w, P.y + P.h + 0.5, { color: V.alpha(C.muted, 0.6), lw: 1 });
        [-6, -4, -2, 0, 2, 4, 6].forEach((v) => V.text(g, (v < 0 ? '−' : '') + Math.abs(v), X(v), P.y + P.h + 15, { size: 10, font: 'mono', align: 'center', color: C.muted }));
        // alvo p
        g.beginPath(); g.moveTo(X(XL), Y(0)); xs.forEach((x, i) => g.lineTo(X(x), Y(pD[i]))); g.lineTo(X(XR), Y(0)); g.closePath();
        g.fillStyle = V.alpha(C.a, 0.28); g.fill(); g.strokeStyle = C.a; g.lineWidth = 2; g.stroke();
        // extremos
        if (S.both) [[0, C.d], [10, C.b]].forEach(([k, col]) => { const ff = fits[k]; g.beginPath(); xs.forEach((x, i) => { const y = Y(npdf(x, ff.m, ff.s)); if (i) g.lineTo(X(x), y); else g.moveTo(X(x), y); }); g.setLineDash([5, 4]); g.strokeStyle = col; g.lineWidth = 1.6; g.stroke(); g.setLineDash([]); });
        // aluno q
        g.beginPath(); g.moveTo(X(XL), Y(0)); xs.forEach((x) => g.lineTo(X(x), Y(npdf(x, S.m, S.s)))); g.lineTo(X(XR), Y(0)); g.closePath();
        g.fillStyle = V.alpha(C.c, 0.14); g.fill(); g.strokeStyle = C.c; g.lineWidth = 2.6; g.stroke();
        // grãos de massa do aluno
        grains.forEach((gn, k) => { const x = S.m + S.s * gn.z; if (x < XL || x > XR) return; const yv = npdf(x, S.m, S.s) * gn.f; V.dot(g, X(x), Y(yv) + Math.sin(t * 2 + k) * 0.8, 2.1, V.alpha(C.c, 0.75)); });
        // legenda
        const lab = S.k === 0 ? 'forward KL: covers everything (mass-covering)' : S.k === 10 ? 'reverse KL: picks one mode (mode-seeking)' : 'generalized JSD, β = ' + nf(S.k / 10, 1);
        if (!narrow) {
          V.text(g, '■ target p (data / teacher)', P.x, 18, { size: 11.5, color: C.a, weight: 600 });
          V.text(g, '■ student q (one Gaussian)', P.x + tw(g, '■ target p (data / teacher)', 11.5, 'sans', 600) + 16, 18, { size: 11.5, color: C.c, weight: 600, maxW: w - 30 });
          V.text(g, lab, P.x + P.w, P.y + 12, { size: 11.5, align: 'right', color: C.ink, weight: 700, maxW: P.w * 0.6 });
          V.text(g, 'μ = ' + nf(S.m, 2) + '  σ = ' + nf(S.s, 2), P.x + P.w, P.y + 28, { size: 10.5, font: 'mono', align: 'right', color: C.muted });
        } else {
          V.text(g, '■ target p (data / teacher)', P.x, 16, { size: 11.5, color: C.a, weight: 600 });
          V.text(g, '■ student q (one Gaussian)', P.x, 33, { size: 11.5, color: C.c, weight: 600 });
          V.text(g, lab, P.x, 54, { size: 11.5, color: C.ink, weight: 700, maxW: P.w });
          V.text(g, 'μ = ' + nf(S.m, 2) + '  σ = ' + nf(S.s, 2), P.x, 70, { size: 10.5, font: 'mono', color: C.muted });
        }
      });
    })();

    /* ============================================================ FIG 6 — coeficiente de gradiente */
    (function () {
      const fig = root.querySelector('#fig-gc'); if (!fig) return;
      adaptH(fig, 430, 470, 640);
      const T0 = ['17', '×', '24', '=', '340', '+', '68', '='];
      const ok = T0.concat(['408', 'EOS']), bad = T0.concat(['418', 'EOS']);
      const pS = [0.80, 0.90, 0.85, 0.90, 0.30, 0.92, 0.50, 0.95, 0.40, 0.90];
      const pT = [0.90, 0.95, 0.90, 0.90, 0.60, 0.90, 0.70, 0.95, 0.05, 0.90];
      const opd = pT.map((x, i) => Math.log(x / pS[i]));
      // PPO: GAE (γ = 1, λ = 0,95) sobre um crítico ilustrativo, resposta errada (recompensa 0)
      const Vc = [0.50, 0.51, 0.52, 0.53, 0.60, 0.61, 0.70, 0.71, 0.72, 0.05];
      const klt = [0.02, 0.01, 0.02, 0.01, 0.05, 0.01, 0.04, 0.01, 0.08, 0.01];
      const rt = klt.map((k) => -0.05 * k);
      const dl = Vc.map((v, i) => rt[i] + (i < 9 ? Vc[i + 1] : 0) - v); // δ_t = r_t + V(s_t+1) − V(s_t)
      const gae = new Array(10); { let acc = 0; for (let i = 9; i >= 0; i--) { acc = dl[i] + 0.95 * acc; gae[i] = acc; } }
      const Rg = [1, 0, 0, 1, 1, 0, 0, 0], fin = ['408', '418', '398', '408', '408', '4080', '418', '398'];
      const mu = Rg.reduce((a, b) => a + b, 0) / 8, sd = Math.sqrt(Rg.reduce((a, b) => a + (b - mu) * (b - mu), 0) / 8);
      const Ag = Rg.map((r) => (r - mu) / sd);
      const wD = 0.1 * 0.5; // β·σ(−β·Δh) com β = 0,1 e Δh = 0
      const M = {
        sft: { det: 'sft', foot: 'GC = 1 on every token · token loss = −log π_θ: weighs more where the model was wrong, but no token is left out', read: 'Every token of the demonstration gets the same push, including the ones the model already knew.', name: 'SFT: imitate the demonstration', info: [['data', 'fixed dataset (off-policy)'], ['judge', 'nobody: imitation'], ['signal', 'GC = 1 on every token']], rows: [{ tag: 'human demonstration', toks: ok, gc: ok.map(() => 1) }] },
        rft: { read: 'The wrong response drops out of training: the negative half of the signal is thrown away.', name: 'RFT: generate, filter, imitate', info: [['data', 'samples from the model itself'], ['judge', 'verifier'], ['signal', 'GC = 1 on correct, 0 on wrong']], rows: [{ tag: 'sample 1 · correct', toks: ok, gc: ok.map(() => 1) }, { tag: 'sample 2 · wrong (discarded)', toks: bad, gc: bad.map(() => 0) }] },
        ppo: { det: 'ppo', foot: 'δ_t = r_t + V(s_t+1) − V(s_t)   ·   Â_t = δ_t + γλ·Â_t+1   (γ = 1, λ = 0.95; illustrative critic)', read: 'The critic\'s value plunges after “418” and GAE spreads the blame backward, decaying with λ.', name: 'PPO: per-token credit via a critic', info: [['data', 'on-policy'], ['judge', 'verifier or RM + critic V'], ['signal', 'per-token Â_t (GAE, λ = 0.95)']], rows: [{ tag: 'on-policy sample · wrong (r = 0)', toks: bad, gc: gae }] },
        grpo: { read: 'One scalar per response: the “17”, “×” and “24” of the wrong ones go down as much as the error itself.', name: 'GRPO: advantage relative to the group', info: [['data', 'G = 8 samples of the same prompt'], ['judge', 'verifier'], ['signal', 'one scalar per response']], rows: Ag.map((a, i) => ({ tag: '', toks: T0.concat([fin[i], 'EOS']), gc: new Array(10).fill(a), right: 'r=' + Rg[i] + '  Â=' + sg(a, 3) })) },
        dpo: { read: 'The same force on every token of the pair, to widen the margin between preferred and rejected.', name: 'DPO: contrast within the pair', info: [['data', 'offline pairs (y_w ≻ y_l)'], ['judge', 'human or AI (preference)'], ['signal', '±β·σ(−βΔh) on each token']], rows: [{ tag: 'y_w · preferred', toks: ok, gc: ok.map(() => wD) }, { tag: 'y_l · rejected', toks: bad, gc: bad.map(() => -wD) }] },
        opd: { det: 'opd', foot: 'GC_t = log π_T(o_t) − log π_θ(o_t)   ·   at “418”: log(0.05 / 0.4) = log 0.125 = −2.079', read: 'The teacher points to the exact token of the error: “418” goes down hard; the rest barely moves.', name: 'OPD: the teacher scores each token', info: [['data', 'on-policy (student sample)'], ['judge', 'teacher (log-probs)'], ['signal', 'log π_T − log π_θ per token']], rows: [{ tag: 'student sample · wrong', toks: bad, gc: opd }] },
      };
      const S = { m: 'grpo', t0: 0 };
      let Hh = null;
      const ctl = UI.controls(fig);
      wrapCtl(UI.seg(ctl, { label: 'Method', options: [{ v: 'sft', label: 'SFT' }, { v: 'rft', label: 'RFT' }, { v: 'ppo', label: 'PPO' }, { v: 'grpo', label: 'GRPO' }, { v: 'dpo', label: 'DPO' }, { v: 'opd', label: 'OPD' }], value: 'grpo', onChange: (v) => { S.m = v; S.t0 = Hh ? Hh.t : 0; } }));
      Hh = V.fig(fig, (g, w, h, t) => {
        const C = V.C(), narrow = w < 640, pad = 14;
        const md = M[S.m];
        const maxA = Math.max(1e-9, ...md.rows.flatMap((r) => r.gc.map(Math.abs)));
        // cabeçalho
        V.text(g, md.name, pad, 20, { size: 14, weight: 700, color: C.ink, maxW: w - 2 * pad });
        V.text(g, 'q: What is 17 × 24?', pad, 40, { size: 12, font: 'mono', color: C.muted });
        let headH;
        if (!narrow) {
          const ix = w * 0.56;
          md.info.forEach(([k, v], i) => { V.text(g, k, ix, 16 + i * 17, { size: 10.5, color: C.muted, weight: 600 }); V.text(g, v, ix + 52, 16 + i * 17, { size: 11.5, color: C.ink, maxW: w - ix - 52 - pad }); });
          headH = 66;
        } else {
          md.info.forEach(([k, v], i) => { V.text(g, k, pad, 60 + i * 15, { size: 10, color: C.muted, weight: 600 }); V.text(g, v, pad + 46, 60 + i * 15, { size: 10.5, color: C.ink, maxW: w - 2 * pad - 46 }); });
          headH = 104;
        }
        const many = md.rows.length > 2, single = md.rows.length === 1;
        const ry = many ? h - (narrow ? 44 : 30) : h - (narrow ? 40 : 34), boxTop = ry - 15;
        const tagW = narrow || many ? 0 : 150, rightW = many ? (narrow ? 86 : 132) : 0;
        const compact = many && narrow, nCol = compact ? 7 : 10;
        const pick = (arr) => (compact ? arr.slice(0, 4).concat(['…'], arr.slice(7, 9)) : arr);
        const cw = Math.min(64, (w - 2 * pad - tagW - rightW) / nCol - 3), cg = 3;
        const gridX = pad + tagW + (narrow || many ? 0 : 10);
        const colX = (j) => gridX + j * (cw + cg) + cw / 2;
        const avail = h - headH - 10;
        const cellH = many ? Math.min(narrow ? 27 : 32, avail / md.rows.length - 6) : (narrow ? 34 : 40);
        // altura da barrinha de probabilidade: preenche o espaço disponível em vez de deixar um vão
        let bh = 0;
        if (single) bh = md.det === 'ppo' ? (narrow ? 46 : 60) : (narrow ? 84 : 110);
        else if (!many) bh = V.clamp((boxTop - 8 - (headH + 6) - 10) / 2 - (narrow ? 32 : 0) - cellH - 30, 0, 72);
        const block = many ? cellH + 6 : (narrow ? 32 : 0) + cellH + 20 + bh + 10;
        const ph = ((t - S.t0) % 2.4) / 2.4, push = Math.sin(ph * Math.PI);
        md.rows.forEach((row, ri) => {
          let y = headH + 6 + ri * (block + (many ? 0 : 10));
          if (!many && narrow) { V.text(g, row.tag, pad, y + 8, { size: 10.5, color: C.muted, weight: 600 }); y += 32; }
          if (!many && !narrow) {
            para(g, row.tag, pad, y + cellH / 2 - 2, tagW - 6, { size: 11, color: C.muted, weight: 600, max: 2, lh: 13 });
            if (bh >= 30) para(g, md.det === 'opd' ? 'P(token): student (solid) and teacher (dashed)' : 'P(token), pulsing with the push', pad, y + cellH + 20 + bh / 2, tagW - 6, { size: 10, color: C.muted, max: 3, lh: 12 });
          }
          const gcs = compact ? row.gc.slice(0, 7) : row.gc;
          pick(row.toks).forEach((tk, j) => {
            const x = gridX + j * (cw + cg), gc = gcs[j], n = gc / maxA;
            const fill = Math.abs(n) < 1e-6 ? V.alpha(C.muted, 0.12) : n > 0 ? V.alpha(C.a, 0.15 + 0.7 * Math.abs(n)) : V.alpha(RED(), 0.15 + 0.7 * Math.abs(n));
            V.box(g, x, y, cw, cellH, { r: 5, fill, stroke: V.alpha(C.ink, 0.12), lw: 1 });
            const fsz = Math.min(12.5, cw * 0.36, cellH * 0.5);
            V.text(g, tk, x + cw / 2, y + cellH / 2 + 1, { size: fsz, font: 'mono', align: 'center', base: 'middle', color: C.ink, weight: tk === '418' || tk === '398' || tk === '4080' ? 700 : 500, maxW: cw - 2 });
            if (!many) {
              V.text(g, Math.abs(gc) < 1e-9 ? '0' : sg(gc, Number.isInteger(gc) ? 0 : 3), x + cw / 2, y + cellH + 13, { size: Math.min(10.5, cw * 0.3), font: 'mono', align: 'center', color: n > 0 ? C.a : n < 0 ? RED() : C.muted, maxW: cw });
              // seta do empurrão
              if (Math.abs(n) > 0.02) { const ah = 4 + 7 * Math.abs(n), ay = y - 4 - push * 3; g.fillStyle = n > 0 ? C.a : RED(); g.beginPath(); if (n > 0) { g.moveTo(x + cw / 2, ay - ah); g.lineTo(x + cw / 2 - 4, ay); g.lineTo(x + cw / 2 + 4, ay); } else { g.moveTo(x + cw / 2, ay); g.lineTo(x + cw / 2 - 4, ay - ah); g.lineTo(x + cw / 2 + 4, ay - ah); } g.closePath(); g.fill(); }
              // barrinha da probabilidade do token, pulsando na direção do gradiente
              if (bh >= 14) {
                const by = y + cellH + 20;
                const pv = V.clamp(pS[j] * (1 + 0.45 * n * push), 0.02, 1);
                V.box(g, x + cw * 0.25, by, cw * 0.5, bh, { r: 3, fill: V.alpha(C.muted, 0.1) });
                V.box(g, x + cw * 0.25, by + bh * (1 - pv), cw * 0.5, bh * pv, { r: 3, fill: V.alpha(n > 0 ? C.a : n < 0 ? RED() : C.muted, 0.75) });
                if (md.det === 'opd') { g.setLineDash([3, 2]); V.rr(g, x + cw * 0.18, by + bh * (1 - pT[j]), cw * 0.64, bh * pT[j], 3); g.strokeStyle = V.alpha(C.d, 0.95); g.lineWidth = 1.4; g.stroke(); g.setLineDash([]); }
              }
            } else if (Math.abs(n) > 0.02) {
              // partículas subindo/descendo pela linha
              const u = ((t * 0.9 + j * 0.13 + ri * 0.31) % 1);
              V.dot(g, x + cw / 2, n > 0 ? y + cellH - u * cellH : y + u * cellH, 1.7, V.alpha(C.ink, 0.35));
            }
          });
          if (many) V.text(g, row.right, gridX + nCol * (cw + cg) + 6, y + cellH / 2 + 1, { size: narrow ? 10 : 11, font: 'mono', base: 'middle', color: Ag[ri] > 0 ? C.a : RED(), weight: 600 });
        });
        // detalhe dos modos de uma linha: as contas por trás do coeficiente
        if (single && md.det) {
          let dy = headH + 6 + (narrow ? 32 : 0) + cellH + 20 + bh + (narrow ? 18 : 26);
          const nsz = Math.min(10.5, cw * 0.32);
          const numRow = (label, vals, fmt, colFn) => {
            if (narrow) { V.text(g, label, pad, dy, { size: 10, color: C.muted, weight: 600 }); dy += 15; }
            else V.text(g, label, pad, dy, { size: 10.5, color: C.muted, weight: 600, maxW: tagW - 6 });
            vals.forEach((v, j) => V.text(g, fmt(v), colX(j), dy, { size: nsz, font: 'mono', align: 'center', color: colFn ? colFn(v, j) : C.ink, maxW: cw }));
            dy += narrow ? 19 : 24;
          };
          const f2 = (v) => nf(v, 2), s2 = (v) => sg(v, 2);
          if (md.det === 'sft') {
            numRow('π_θ(token)', pS, f2);
            numRow('loss −log π_θ', pS.map((p) => -Math.log(p)), f2, (v) => V.mix(C.muted, C.ink, Math.min(1, v)));
          } else if (md.det === 'opd') {
            numRow('π_T teacher', pT, f2, () => C.d);
            numRow('π_θ student', pS, f2);
          } else if (md.det === 'ppo') {
            // V(s_t): previsão do crítico ANTES de cada token; a queda depois de “418” é a surpresa δ
            const chH = narrow ? 46 : 58;
            if (narrow) { V.text(g, 'V(s_t): the critic predicts the reward still to come', pad, dy, { size: 10, color: C.muted, weight: 600, maxW: w - 2 * pad }); dy += 10; }
            else para(g, 'V(s_t): critic prediction before each token', pad, dy + chH / 2 - 6, tagW - 6, { size: 10.5, color: C.muted, weight: 600, max: 3, lh: 12 });
            const cy0 = dy + 4, Yv = (v) => cy0 + chH * (1 - v);
            [0, 1].forEach((v) => { V.line(g, colX(0) - cw / 2, Yv(v), colX(9) + cw / 2, Yv(v), { color: V.alpha(C.muted, v ? 0.15 : 0.35), lw: 1, dash: v ? [2, 4] : null }); });
            if (!narrow) { V.text(g, '1', colX(0) - cw / 2 - 4, Yv(1) + 3, { size: 9.5, font: 'mono', align: 'right', color: C.muted }); V.text(g, '0', colX(0) - cw / 2 - 4, Yv(0) + 3, { size: 9.5, font: 'mono', align: 'right', color: C.muted }); }
            g.beginPath(); Vc.forEach((v, j) => (j ? g.lineTo(colX(j), Yv(v)) : g.moveTo(colX(j), Yv(v)))); g.strokeStyle = V.alpha(C.b, 0.9); g.lineWidth = 2; g.stroke();
            V.line(g, colX(8), Yv(Vc[8]), colX(9), Yv(Vc[9]), { color: RED(), lw: 2.6 });
            Vc.forEach((v, j) => { V.dot(g, colX(j), Yv(v), j >= 8 ? 4 : 3, j === 9 ? RED() : C.b); if (!narrow) V.text(g, f2(v), colX(j), Yv(v) - 7, { size: 9.5, font: 'mono', align: 'center', color: j === 9 ? RED() : C.muted }); });
            const mxp = (colX(8) + colX(9)) / 2, myp = (Yv(Vc[8]) + Yv(Vc[9])) / 2;
            // abaixo e à esquerda do trecho que despenca: longe dos pontos (no estreito, os dois valores vão no rótulo)
            V.text(g, (narrow ? f2(Vc[8]) + ' → ' + f2(Vc[9]) + ': ' : '') + 'δ ≈ ' + sg(dl[8], 2), mxp - 6, myp + (narrow ? 8 : 16), { size: 10.5, font: 'mono', weight: 700, align: 'right', color: RED() });
            dy = cy0 + chH + (narrow ? 22 : 26);
            numRow('δ_t (surprise)', dl, s2, (v) => (v < -0.02 ? RED() : v > 0.02 ? C.a : C.muted));
          }
          // a fórmula, logo acima da caixa de leitura
          const FL = lines(g, md.foot, w - 2 * pad, narrow ? 10 : 10.5, 'mono', 400).slice(0, 3);
          FL.forEach((l, i) => V.text(g, l, pad, boxTop - 9 - (FL.length - 1 - i) * 13, { size: narrow ? 10 : 10.5, font: 'mono', color: C.muted }));
        }
        if (many) V.text(g, 'mean = ' + nf(mu, 3) + ' · population std = ' + nf(sd, 3), gridX, h - 6, { size: 10.5, font: 'mono', color: C.muted, maxW: w - 2 * pad });
        V.box(g, pad, boxTop, w - 2 * pad, narrow ? 36 : 24, { r: 6, fill: V.alpha(C.d, 0.1), stroke: V.alpha(C.d, 0.4), lw: 1 });
        para(g, md.read, pad + 10, ry + 1, w - 2 * pad - 20, { size: narrow ? 10.5 : 11.5, color: C.ink, weight: 600, max: 2, lh: 13 });
      });
    })();

    /* ============================================================ FIG 7 — matriz de requisitos */
    (function () {
      const fig = root.querySelector('#fig-tax'); if (!fig) return;
      adaptH(fig, 600, 800, 700);
      const COLS = [['demo', 'demonstrations'], ['pairs', 'preference pairs'], ['binary', 'good/bad label'], ['verif.', 'verifier'], ['RM', 'trained reward model'], ['π_ref', 'reference model'], ['on-pol.', 'on-policy rollouts'], ['judge', 'another model judges (teacher / AI)']];
      const COLS_N = ['dem', 'pair', 'bin', 'ver', 'RM', 'ref', 'on', 'judge'];
      const fam = (C, f) => ({ x: C.a, xi: C.b, xii: C.c, xiii: C.d }[f]);
      const ROWS = [
        { n: 'SFT', s: 'SFT', f: 'x', q: [2, 0, 0, 0, 0, 0, 0, 0], d: 'Demonstrations → cross-entropy on the assistant tokens only.', sig: 'dense, per token', ch: 'SFT (ch. 52)', mem: [['policy', 16]] },
        { n: 'RFT / STaR / ReST', s: 'RFT', f: 'x', q: [0, 0, 0, 2, 1, 0, 1, 0], d: 'Generates several responses, keeps the ones the verifier approves and does SFT on them.', sig: 'dense on approved', ch: 'SFT (ch. 52)', mem: [['policy', 16]] },
        { n: 'Off-policy distillation', s: 'Dist. off', f: 'x', q: [1, 0, 0, 0, 0, 0, 0, 2], d: 'SFT on the outputs (or logits) of a stronger teacher.', sig: 'dense', ch: 'SFT (ch. 52)', mem: [['student', 16]], note: 'teacher outputs pre-generated' },
        { n: 'On-policy distillation / MOPD', s: 'OPD/MOPD', f: 'x', q: [0, 0, 0, 1, 0, 0, 2, 2], d: 'The student generates; the teacher (one per domain in MOPD) gives log-probs; per-token reverse KL, + optional ORM.', sig: 'dense, on-policy', ch: 'SFT (ch. 52)', mem: [['student', 16], ['teacher', 2]], note: 'the teacher can be served on another node (vLLM)' },
        { n: 'RLHF (RM + PPO)', s: 'RLHF', f: 'xi', q: [1, 2, 0, 0, 2, 2, 2, 0], d: 'Human pairs → Bradley–Terry RM → PPO with KL to the SFT model; a critic estimates the advantage.', sig: 'scalar per response', ch: 'PPO & RLHF (ch. 55)', mem: [['policy', 16], ['critic', 16], ['reference', 2], ['RM', 2]] },
        { n: 'A3C, TRPO and kin (RM + RL)', s: 'RM + RL', f: 'xi', q: [0, 1, 0, 1, 2, 1, 2, 0], d: 'Any RL algorithm against an explicit RM; asynchronous actors (A3C) or a trust region (TRPO).', sig: 'scalar', ch: 'RL fundamentals (ch. 54)', mem: [['policy', 16], ['critic', 16], ['reference', 2], ['RM', 2]] },
        { n: 'GRPO / RLOO / REINFORCE++', s: 'GRPO', f: 'xi', q: [0, 0, 0, 2, 1, 1, 2, 0], d: 'No critic: the baseline is the mean of the group of responses to the same prompt. The engine of RLVR.', sig: 'scalar, relative to the group', ch: 'GRPO & RLVR (ch. 56)', mem: [['policy', 16], ['reference', 2]], note: '+2N if the judge is an RM; no reference if β = 0' },
        { n: 'DPO', s: 'DPO', f: 'xii', q: [0, 2, 0, 0, 0, 2, 0, 0], d: 'Logistic loss on the log-ratio margin; the implicit reward is β·log π/π_ref.', sig: 'pair contrast', ch: 'DPO (ch. 57)', mem: [['policy', 16], ['reference', 2]], note: 'or zero if the reference log-probs are precomputed' },
        { n: 'IPO', s: 'IPO', f: 'xii', q: [0, 2, 0, 0, 0, 2, 0, 0], d: 'Quadratic loss that drives the margin to 1/(2τ); does not saturate with deterministic preferences.', sig: 'pair contrast', ch: 'IPO, KTO & SimPO (ch. 58)', mem: [['policy', 16], ['reference', 2]] },
        { n: 'KTO', s: 'KTO', f: 'xii', q: [0, 0, 2, 0, 0, 2, 0, 0], d: 'Binary good/bad labels, no pairs, with a Kahneman–Tversky utility relative to a reference point.', sig: 'asymmetric', ch: 'IPO, KTO & SimPO (ch. 58)', mem: [['policy', 16], ['reference', 2]] },
        { n: 'SimPO', s: 'SimPO', f: 'xii', q: [0, 2, 0, 0, 0, 0, 0, 0], d: 'Reward = mean per-token log-prob, with margin γ; no reference model.', sig: 'pair contrast', ch: 'IPO, KTO & SimPO (ch. 58)', mem: [['policy', 16]] },
        { n: 'ORPO', s: 'ORPO', f: 'xii', q: [1, 2, 0, 0, 0, 0, 0, 0], d: 'SFT on the preferred response + an odds ratio between the pair, in a single loss; no reference.', sig: 'NLL + odds', ch: 'IPO, KTO & SimPO (ch. 58)', mem: [['policy', 16]] },
        { n: 'Contrastive (SLiC, CPO, unlikelihood)', s: 'Contrast.', f: 'xii', q: [0, 2, 0, 0, 0, 1, 0, 0], d: 'Pushes the positive up and the negative down, with a margin or a hinge.', sig: 'margin', ch: 'AWR & contrastive (ch. 59)', mem: [['policy', 16]] },
        { n: 'AWR', s: 'AWR', f: 'xii', q: [0, 0, 0, 1, 1, 0, 0, 0], d: 'Supervised regression weighted by exp(A/β), on data from a behavior policy.', sig: 'weight exp(A/β)', ch: 'AWR & contrastive (ch. 59)', mem: [['policy', 16], ['critic V', 16]] },
        { n: 'Constitutional AI (RLAIF)', s: 'CAI', f: 'xiii', q: [1, 2, 0, 0, 2, 2, 2, 2], d: 'Written principles: self-critique and revision (SL-CAI), then an AI labels pairs → preference model → RL.', sig: 'AI preferences', ch: 'Constitutional AI (ch. 60)', mem: [['policy', 16], ['critic', 16], ['reference', 2], ['PM', 2]], note: 'RL phase; the SL phase is just SFT' },
        { n: 'Self-play (SPIN, self-rewarding)', s: 'Self-play', f: 'xiii', q: [1, 1, 0, 0, 0, 2, 1, 1], d: 'The model competes against its own previous version (SPIN: DPO against its own responses) or judges itself.', sig: 'comparative', ch: 'Self-play & debate (ch. 61)', mem: [['policy', 16], ['previous iteration', 2]] },
        { n: 'Debate', s: 'Debate', f: 'xiii', q: [0, 0, 0, 0, 0, 0, 2, 2], d: 'Two agents argue and a weaker judge decides; proposed for scalable oversight.', sig: 'debate win', ch: 'Self-play & debate (ch. 61)', mem: [['debaters', 16], ['judge', 2]] },
      ];
      ROWS.forEach((r, i) => { r.y = null; r.i = i; });
      const S = { col: -1, sel: 4, hov: -1, N: 7 };
      const ctl = UI.controls(fig);
      wrapCtl(UI.select(ctl, { label: 'Ingredient', options: [{ v: '-1', label: 'all (module order)' }].concat(COLS.map((c, i) => ({ v: String(i), label: c[1] }))), value: '-1', onChange: (v) => { S.col = +v; } }));
      wrapCtl(UI.select(ctl, { label: 'Method', options: ROWS.map((r, i) => ({ v: String(i), label: r.n })), value: '4', onChange: (v) => { S.sel = +v; upd(); } }));
      nocase(UI.slider(ctl, { label: 'N per model (billions)', min: 0.5, max: 14, step: 0.5, value: 7, fmt: (v) => nf(v, 1) + ' B', onInput: (v) => { S.N = v; upd(); } }));
      const rM = UI.readout(ctl, 'state memory', '—');
      const GiB = 1073741824, SPARK = 121.69 * GiB;
      function memOf(r) { return r.mem.reduce((s, m) => s + m[1], 0) * S.N * 1e9; }
      function upd() { const b = memOf(ROWS[S.sel]); rM.set(nf(b / 1e9, 0) + ' GB (' + nf(b / GiB, 1) + ' GiB)', b > 2 * SPARK ? 'bad' : b > SPARK ? 'warn' : 'ok'); }
      upd();
      V.fig(fig, (g, w, h, t, dt, _s, Hh) => {
        const C = V.C(), narrow = w < 700, pad = 12;
        const labW = narrow ? 82 : 210;
        const colW = narrow ? Math.min(34, (w - 2 * pad - labW) / 8) : Math.min(58, (w * 0.64 - pad - labW) / 8);
        const mx = pad + labW, headH = narrow ? 30 : 44, my = headH + 6;
        const memH = 112;
        const panelH = narrow ? 150 : 0;
        const rowsH = h - my - memH - panelH - (narrow ? 16 : 14);
        const rh = rowsH / ROWS.length;
        // ordem
        const order = ROWS.map((r) => r.i).sort((a, b) => (S.col < 0 ? a - b : (ROWS[b].q[S.col] - ROWS[a].q[S.col]) || a - b));
        order.forEach((ri, k) => { const r = ROWS[ri], ty = my + k * rh; r.y = r.y == null ? ty : r.y + (ty - r.y) * Math.min(1, dt * 5); });
        // hover / clique
        const p = Hh.pointer; S.hov = -1;
        if (p.inside && p.x >= pad && p.x <= mx + 8 * colW) ROWS.forEach((r) => { if (p.y >= r.y && p.y < r.y + rh) S.hov = r.i; });
        if (p.clicked && S.hov >= 0) { S.sel = S.hov; upd(); }
        // coluna selecionada
        if (S.col >= 0) { const cx = mx + S.col * colW; V.box(g, cx + 1, 4, colW - 2, my + ROWS.length * rh - 2, { r: 6, fill: V.alpha(C.d, 0.1 + 0.05 * Math.sin(t * 3)), stroke: V.alpha(C.d, 0.6), lw: 1.2 }); }
        // cabeçalho
        COLS.forEach((c, j) => { const x = mx + j * colW + colW / 2; V.text(g, narrow ? COLS_N[j] : c[0], x, headH - 8, { size: narrow ? 9.5 : 11, align: 'center', color: j === S.col ? C.d : C.muted, weight: 700, font: c[0] === 'π_ref' ? 'mono' : 'sans', maxW: colW }); });
        // linhas
        ROWS.forEach((r) => {
          const y = r.y, active = S.col < 0 || r.q[S.col] > 0, isSel = r.i === S.sel, isHov = r.i === S.hov;
          g.globalAlpha = active ? 1 : 0.32;
          if (isSel || isHov) V.box(g, pad - 4, y + 1, labW + 8 * colW + 6, rh - 2, { r: 5, fill: V.alpha(C.ink, isSel ? 0.08 : 0.05) });
          V.box(g, pad, y + rh * 0.25, 4, rh * 0.5, { r: 2, fill: fam(C, r.f) });
          V.text(g, narrow ? r.s : r.n, pad + 9, y + rh / 2 + 1, { size: narrow ? 10.5 : 11.5, base: 'middle', color: C.ink, weight: isSel ? 700 : 500, maxW: labW - 12 });
          r.q.forEach((v, j) => {
            const cx = mx + j * colW + colW / 2, cy = y + rh / 2, rad = Math.min(rh * 0.32, colW * 0.3, 7.5);
            g.beginPath(); g.arc(cx, cy, rad, 0, Math.PI * 2); g.strokeStyle = V.alpha(fam(C, r.f), 0.75); g.lineWidth = 1.3; g.stroke();
            if (v === 2) { g.fillStyle = fam(C, r.f); g.fill(); }
            else if (v === 1) { g.beginPath(); g.arc(cx, cy, rad, Math.PI / 2, Math.PI * 1.5); g.closePath(); g.fillStyle = fam(C, r.f); g.fill(); }
          });
          g.globalAlpha = 1;
        });
        // fluxo descendo pela coluna escolhida
        if (S.col >= 0) ROWS.forEach((r) => { if (r.q[S.col] === 0) return; const cx = mx + S.col * colW + colW / 2; const u = (t * 0.6 + r.i * 0.07) % 1; const yy = V.lerp(headH, r.y + rh / 2, u); V.dot(g, cx, yy, 2, V.alpha(C.d, 0.8 * (1 - u))); });
        // painel de detalhe
        const R = ROWS[S.hov >= 0 ? S.hov : S.sel];
        let px, py, pw, ph;
        if (!narrow) { px = mx + 8 * colW + 22; py = 6; pw = w - px - pad; ph = my + ROWS.length * rh - 8; }
        else { px = pad; py = my + ROWS.length * rh + 8; pw = w - 2 * pad; ph = panelH - 6; }
        V.box(g, px, py, pw, ph, { r: 10, fill: V.alpha(fam(C, R.f), 0.07), stroke: V.alpha(fam(C, R.f), 0.45), lw: 1.1 });
        V.text(g, R.n, px + 12, py + 22, { size: 13.5, weight: 700, color: C.ink, maxW: pw - 24 });
        let yy = py + 42;
        yy += para(g, R.d, px + 12, yy, pw - 24, { size: narrow ? 11 : 12, color: C.ink, max: narrow ? 3 : 5 }) + 4;
        const req = COLS.filter((c, j) => R.q[j] === 2).map((c) => c[1]);
        const opt = COLS.filter((c, j) => R.q[j] === 1).map((c) => c[1]);
        if (!narrow) {
          yy += para(g, 'needs: ' + (req.join(', ') || '—'), px + 12, yy, pw - 24, { size: 11, color: C.ink, max: 3 }) + 2;
          if (opt.length) yy += para(g, 'optional: ' + opt.join(', '), px + 12, yy, pw - 24, { size: 11, color: C.muted, max: 2 }) + 2;
          yy += para(g, 'signal: ' + R.sig, px + 12, yy, pw - 24, { size: 11, color: C.muted, max: 1 }) + 2;
          yy += para(g, 'where to learn it: ' + R.ch, px + 12, yy, pw - 24, { size: 11, color: fam(C, R.f), weight: 600, max: 2 }) + 6;
          if (yy + 30 < py + ph) { yy += para(g, 'in memory: ' + R.mem.map((m) => m[0] + ' (' + (m[1] === 16 ? 'trainable, 16N' : 'frozen, 2N') + ')').join(' + '), px + 12, yy, pw - 24, { size: 11, color: C.ink, max: 3 }); }
        } else {
          yy += para(g, 'needs: ' + (req.join(', ') || '—') + (opt.length ? ' · optional: ' + opt.join(', ') : ''), px + 12, yy, pw - 24, { size: 10.5, color: C.muted, max: 2 }) + 2;
          para(g, R.ch, px + 12, yy, pw - 24, { size: 10.5, color: fam(C, R.f), weight: 600, max: 1 });
        }
        // barra de memória
        const by = h - memH + 10, bx = pad, bw = w - 2 * pad;
        const tot = memOf(R), scale = Math.max(tot, 2 * SPARK) * 1.08;
        V.text(g, 'state memory for “' + R.s + '”: ' + nf(tot / 1e9, 0) + ' GB (' + nf(tot / GiB, 1) + ' GiB) with N = ' + nf(S.N, 1) + ' B per model', bx, by, { size: narrow ? 10.5 : 11, color: C.ink, weight: 600, maxW: bw });
        const bh = 24, bY = by + 22;
        [[SPARK, '1 Spark'], [2 * SPARK, '2 Sparks']].forEach(([v, l]) => { const x = bx + (v / scale) * bw; V.text(g, l + ' ▾', x, bY - 3, { size: 9.5, align: 'right', color: C.muted }); });
        V.box(g, bx, bY, bw, bh, { r: 6, fill: V.alpha(C.muted, 0.08) });
        let cx = bx;
        const mcol = (m, k) => (m[1] === 16 ? (k === 0 ? C.c : C.b) : C.series[(k + 2) % 6]);
        R.mem.forEach((m, k) => {
          const ww = (m[1] * S.N * 1e9 / scale) * bw;
          V.box(g, cx, bY, Math.max(2, ww - 2), bh, { r: 5, fill: V.alpha(mcol(m, k), 0.75) });
          if (ww > 50) V.text(g, m[0], cx + ww / 2 - 1, bY + bh / 2 + 1, { size: 10.5, align: 'center', base: 'middle', color: C.ink, weight: 600, maxW: ww - 6 });
          cx += ww;
        });
        [[SPARK, '1 Spark'], [2 * SPARK, '2 Sparks']].forEach(([v]) => { const x = bx + (v / scale) * bw; V.line(g, x, bY - 2, x, bY + bh + 2, { color: V.alpha(C.ink, 0.75), lw: 1.5, dash: [3, 3] }); });
        let lx = bx; const ly = bY + bh + 16;
        R.mem.forEach((m, k) => { const lab = m[0] + (m[1] === 16 ? ' 16N' : ' 2N'), lw2 = tw(g, lab, 10) + 26; if (lx + lw2 > bx + bw) return; V.box(g, lx, ly - 8, 10, 10, { r: 2, fill: V.alpha(mcol(m, k), 0.8) }); V.text(g, lab, lx + 14, ly + 1, { size: 10, color: C.muted }); lx += lw2; });
        if (R.note) V.text(g, R.note, bx, ly + 16, { size: 10, color: C.muted, maxW: bw });
      });
    })();

    /* ============================================================ FIG 8 — as famílias */
    (function () {
      const fig = root.querySelector('#fig-familias'); if (!fig) return;
      adaptH(fig, 430, 640, 640);
      // c = coluna (0..5), r = linha (0..2); mini = caixinha de especialista; note = nota tracejada
      const FAM = {
        glm: { title: 'GLM-5 (the report the GLM-5.3-Flash card points to)', nodes: [
          { c: 0, r: 1, t: 'GLM-5 base', s: '28.5T tokens', src: 'P', k: 'muted' },
          { c: 1, r: 1, t: 'SFT', s: '3 groups · 202,752 tokens · 3 thinking modes', src: 'P', k: 'a' },
          { c: 2, r: 1, t: 'Reasoning RL', s: 'GRPO + IcePop · no KL · group of 32', src: 'P', k: 'c' },
          { c: 3, r: 1, t: 'Agentic RL', s: 'async · slime · >10,000 environments', src: 'P', k: 'c' },
          { c: 4, r: 1, t: 'General RL', s: 'rules + ORM + generative judges', src: 'P', k: 'c' },
          { c: 5, r: 1, t: 'Cross-stage distillation', s: 'teachers = earlier checkpoints', src: 'P', k: 'd' },
          { c: 1.5, r: 0, t: 'GLM-5.2: SAO brings a critic back to agentic RL', src: 'P', note: 1 },
          { c: 4, r: 0, t: 'GLM-5.3-Flash: post-training not published in the card', src: '∅', note: 1 },
          { c: 1, r: 2.45, m: [2.75, 0], t: 'Agent SFT: wrong steps stay in the context, but out of the loss', src: 'P', note: 1 },
        ], edges: [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]], teach: [[1, 5], [2, 5], [3, 5]] },
        qwen: { title: 'Qwen3 (2025 report) and successors', nodes: [
          { c: 0, r: 1, t: 'Qwen3 base', s: 'pretrained', src: 'P', k: 'muted' },
          { c: 1, r: 1, t: 'Cold start', s: 'verified long CoT', src: 'P', k: 'a' },
          { c: 2, r: 1, t: 'Reasoning RL', s: 'GRPO · 3,995 pairs · AIME’24 70.1 → 85.1', src: 'P', k: 'c' },
          { c: 3, r: 1, t: 'Mode fusion', s: '/think · /no_think · budget', src: 'P', k: 'a' },
          { c: 4, r: 1, t: 'General RL', s: 'more than 20 tasks', src: 'P', k: 'c' },
          { c: 5, r: 1, t: 'Large models', s: 'flagship Qwen3', src: 'P', k: 'e' },
          { c: 5, r: 2, t: 'Strong-to-weak', s: 'distillation off → on-policy · 1/10 of the GPU-h', src: 'P', k: 'd' },
          { c: 1.5, r: 0, t: 'GSPO and SAPO (2025)', src: 'P', note: 1 },
          { c: 3.5, r: 0, t: 'Qwen3.5: disaggregated async RL', src: 'P', note: 1 },
          { c: 2.5, r: 2, t: 'Qwen3.8 / Flash-Next: post-training not described', src: '∅', note: 1 },
        ], edges: [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6]] },
        ds: { title: 'DeepSeek-V4 (2026), heir to V3, R1 and V3.2', nodes: [
          { c: 0, r: 1, t: 'V4 base', s: '>32T tokens', src: 'P', k: 'muted' },
          { c: 2, r: 0, m: [1.55, 0.2], t: 'Math', mini: 1, k: 'c' }, { c: 2, r: 0.4, m: [2.45, 0.2], t: 'Code', mini: 1, k: 'c' },
          { c: 2, r: 0.8, m: [1.55, 1], t: 'Agents', mini: 1, k: 'c' }, { c: 2, r: 1.2, m: [2.45, 1], t: 'Non-think', mini: 1, k: 'b' },
          { c: 2, r: 1.6, m: [1.55, 1.8], t: 'Think High', mini: 1, k: 'b' }, { c: 2, r: 2.0, m: [2.45, 1.8], t: 'Think Max', mini: 1, k: 'b' },
          { c: 4, r: 1, t: 'Multi-teacher OPD', s: 'reverse KL · whole vocabulary · >10 teachers', src: 'P', k: 'd' },
          { c: 5, r: 1, t: 'DeepSeek-V4', s: 'FP4 QAT · FP4 rollouts', src: 'P', k: 'e' },
          { c: 0.55, r: 2.3, m: [3.2, 2.6], t: 'specialists: SFT + GRPO; modes = length penalties', src: 'P', note: 1 },
          { c: 4.5, r: 0, t: 'V3.2: specialists + mixed RL → in V4, OPD', src: 'P', note: 1 },
        ], edges: [[0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [1, 7], [2, 7], [3, 7], [4, 7], [5, 7], [6, 7], [7, 8]] },
        mimo: { title: 'MiMo-V2-Flash (MOPD) and V2.6 (MOPD2)', nodes: [
          { c: 0, r: 1, t: 'MiMo base', s: '27T FP8 tokens', src: 'P', k: 'muted' },
          { c: 1, r: 1, t: 'SFT', s: 'millions of samples', src: 'P', k: 'a' },
          { c: 2.5, r: 0, m: [2.05, 0.2], t: 'Math', mini: 1, k: 'c' }, { c: 2.5, r: 0.4, m: [2.95, 0.2], t: 'Code', mini: 1, k: 'c' },
          { c: 2.5, r: 0.8, m: [2.05, 1], t: 'Search', mini: 1, k: 'c' }, { c: 2.5, r: 1.2, m: [2.95, 1], t: 'Tools', mini: 1, k: 'c' },
          { c: 2.5, r: 1.6, m: [2.05, 1.8], t: 'Reasoning', mini: 1, k: 'c' }, { c: 2.5, r: 2.0, m: [2.95, 1.8], t: 'Safety', mini: 1, k: 'c' },
          { c: 4, r: 1, t: 'MOPD', s: 'each prompt to its domain teacher · Â = log π_teacher − log π_θ', src: 'P', k: 'd' },
          { c: 5, r: 1, t: 'MiMo-V2.6 (new cycle)', s: 'SFT → a single mixed RL → MOPD2 with prefixes', src: 'P', k: 'e' },
          { c: 4.5, r: 0, t: 'repeated tool calls: a 12-step teacher, $90K', src: 'P', note: 1 },
          { c: 4.5, r: 2.2, t: 'fusion in policy space, not weight averaging', src: 'P', note: 1 },
        ], edges: [[0, 1], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [1, 7], [2, 8], [3, 8], [4, 8], [5, 8], [6, 8], [7, 8], [8, 9, 'next version', '(new cycle from SFT)']] },
      };
      const SCORE = { b: ['AIME25', 'HMMT25', 'LCB', 'IFBench', 'SWE-V', 'τ²-Bench', 'creative writing', 'BrowseComp'], n: ['AIME', 'HMMT', 'LCB', 'IFB', 'SWE-V', 'τ²', 'writing', 'Browse'], v: [[89.3, 93.9, 94.1], [76.9, 82.6, 84.4], [77.5, 82.6, 83.2], [55.4, 68.9, 66.7], [67.8, 74.2, 73.4], [75.9, 79.6, 80.3], [90.1, 90.1, 86.2], [42.5, 51.7, 45.4]] };
      const S = { f: 'glm', t0: 0 };
      let Hh = null;
      const ctl = UI.controls(fig);
      wrapCtl(UI.seg(ctl, { label: 'Family', options: [{ v: 'glm', label: 'GLM' }, { v: 'qwen', label: 'Qwen' }, { v: 'ds', label: 'DeepSeek' }, { v: 'mimo', label: 'MiMo' }, { v: 'score', label: 'MOPD scoreboard' }], value: 'glm', onChange: (v) => { S.f = v; S.t0 = Hh ? Hh.t : 0; } }));
      Hh = V.fig(fig, (g, w, h, t) => {
        const C = V.C(), narrow = w < 640, pad = 12;
        const kc = (k) => (k === 'muted' ? C.muted : C[k]);
        if (S.f === 'score') {
          const P = { x: pad + 30, y: 44, w: w - 2 * pad - 36, h: h - 96 };
          V.text(g, 'MiMo-V2-Flash: student → best teacher → student after MOPD (report, Table 7, and MOPD paper)', pad, 18, { size: 12.5, weight: 700, color: C.ink, maxW: w - 2 * pad });
          const lo = 40, hi = 100, Y = (v) => P.y + P.h - ((v - lo) / (hi - lo)) * P.h;
          [40, 60, 80, 100].forEach((v) => { V.line(g, P.x, Y(v), P.x + P.w, Y(v), { color: V.alpha(C.muted, 0.18), lw: 1 }); V.text(g, String(v), P.x - 6, Y(v) + 4, { size: 10, font: 'mono', align: 'right', color: C.muted }); });
          const grow = V.ease.out(Math.min(1, (t - S.t0) / 1.2));
          const gw = P.w / SCORE.b.length, bw = Math.min(26, gw / (narrow ? 4.4 : 3.6)), bgap = narrow ? 2 : 3;
          const cols = [C.muted, C.b, C.a];
          SCORE.b.forEach((b, i) => {
            const gx = P.x + gw * i + gw / 2;
            SCORE.v[i].forEach((v, k) => {
              const x = gx + (k - 1) * (bw + bgap) - bw / 2, top = Y(lo + (v - lo) * grow);
              V.box(g, x, top, bw, P.y + P.h - top, { r: 3, fill: V.alpha(cols[k], 0.8) });
              if (!narrow) V.text(g, nf(v, 1), x + bw / 2, top - 5, { size: 10, font: 'mono', align: 'center', color: C.ink }); // no estreito, só o Δ embaixo
            });
            V.text(g, narrow ? SCORE.n[i] : b, gx, P.y + P.h + 15, { size: narrow ? 9.5 : 11, align: 'center', color: C.ink, weight: 600, maxW: gw - 2 });
            const dlt = SCORE.v[i][2] - SCORE.v[i][1];
            V.text(g, (dlt >= 0 ? '+' : '−') + nf(Math.abs(dlt), 1) + (narrow ? '' : ' vs teacher'), gx, P.y + P.h + 29, { size: narrow ? 9.5 : 10, align: 'center', color: dlt >= 0 ? C.a : RED(), weight: narrow ? 700 : 400, maxW: gw - 2 });
          });
          if (narrow) V.text(g, 'below: student with MOPD − best teacher', P.x + P.w, P.y + P.h + 44, { size: 9.5, align: 'right', color: C.muted, maxW: P.w });
          let lx = pad;
          [['student before', cols[0]], ['best teacher', cols[1]], ['student with MOPD', cols[2]]].forEach(([l, c]) => { V.box(g, lx, 28, 10, 10, { r: 2, fill: c }); V.text(g, l, lx + 14, 37, { size: 10.5, color: C.muted }); lx += tw(g, l, 10.5) + 30; });
          return;
        }
        const Fm = FAM[S.f];
        V.text(g, Fm.title, pad, 18, { size: 12.5, weight: 700, color: C.ink, maxW: w - 2 * pad });
        const top = 32;
        const pos = (n) => {
          if (!narrow) { const cw = (w - 2 * pad) / 6, rh = (h - top - 8) / 3; return [pad + cw * (n.c + 0.5), top + rh * (n.r + 0.5), cw, rh]; }
          const cw = (w - 2 * pad) / 3, rh = (h - top - 8) / 6; const c = n.m ? n.m[0] : n.c, r = n.m ? n.m[1] : n.r; return [pad + cw * (r + 0.5), top + rh * (c + 0.5), cw, rh];
        };
        const geo = Fm.nodes.map((n) => {
          const [x, y, cw, rh] = pos(n);
          let bw, bh;
          if (n.mini) { bw = narrow ? cw * 0.46 : cw * 0.8; bh = narrow ? 24 : 26; }
          else if (n.note) { bw = narrow ? cw * 0.95 : cw * 1.5; bh = narrow ? 62 : 44; }
          else { bw = cw - (narrow ? 8 : 14); bh = Math.min(rh - 12, narrow ? 76 : 86); }
          const bx = V.clamp(x - bw / 2, pad, w - pad - bw), by = V.clamp(y - bh / 2, top, h - 6 - bh);
          return { x: bx, y: by, w: bw, h: bh, cx: bx + bw / 2, cy: by + bh / 2 };
        });
        // arestas
        Fm.edges.forEach(([a, b, l1, l2], k) => {
          const A = geo[a], B = geo[b];
          const vert = narrow;
          const x1 = vert ? A.cx : A.x + A.w, y1 = vert ? A.y + A.h : A.cy, x2 = vert ? B.cx : B.x, y2 = vert ? B.y : B.cy;
          if (l1) { // aresta para outra versão: tracejada e rotulada, não é um estágio do mesmo pipeline
            V.line(g, x1, y1, x2, y2, { color: V.alpha(C.muted, 0.8), lw: 1.4, dash: [4, 4], dashOffset: -t * 10 });
            if (!narrow) { const mx = (x1 + x2) / 2, ty = Math.min(A.y, B.y) - 20; V.text(g, l1, mx, ty, { size: 10, align: 'center', color: C.muted, weight: 600 }); V.text(g, l2, mx, ty + 12, { size: 10, align: 'center', color: C.muted }); }
            return;
          }
          V.line(g, x1, y1, x2, y2, { color: V.alpha(C.muted, 0.55), lw: 1.3 });
          const u = ((t * 0.5 + k * 0.17) % 1);
          V.dot(g, V.lerp(x1, x2, u), V.lerp(y1, y2, u), 2.6, V.alpha(kc(Fm.nodes[b].k || 'a'), 0.9));
        });
        (Fm.teach || []).forEach(([a, b], k) => {
          const A = geo[a], B = geo[b];
          const x1 = narrow ? A.x + A.w : A.cx, y1 = narrow ? A.cy : A.y + A.h, x2 = B.cx - 10, y2 = B.y + B.h;
          g.beginPath(); g.moveTo(x1, y1);
          if (!narrow) g.bezierCurveTo(x1, y1 + 50 + k * 8, x2, y2 + 50 + k * 8, x2, y2); else g.bezierCurveTo(x1 + 18 + k * 7, y1, B.x + B.w + 18 + k * 7, B.cy, B.x + B.w, B.cy);
          g.setLineDash([4, 4]); g.lineDashOffset = -t * 14; g.strokeStyle = V.alpha(C.d, 0.75); g.lineWidth = 1.3; g.stroke(); g.setLineDash([]);
        });
        if (Fm.teach) { const B = geo[5]; V.text(g, 'teachers', narrow ? B.x + B.w - 4 : B.cx, narrow ? B.y + B.h + 14 : B.y + B.h + 46, { size: 10, align: narrow ? 'right' : 'center', color: C.d, weight: 600 }); }
        // nós
        Fm.nodes.forEach((n, i) => {
          const G = geo[i], col = n.note ? C.muted : kc(n.k);
          if (n.note) {
            g.setLineDash([4, 3]); V.box(g, G.x, G.y, G.w, G.h, { r: 8, fill: V.alpha(C.muted, 0.05), stroke: V.alpha(C.muted, 0.6), lw: 1 }); g.setLineDash([]);
            para(g, n.t, G.x + 8, G.y + 15, G.w - (narrow ? 14 : 24), { size: narrow ? 9.5 : 10.5, color: C.muted, max: narrow ? 4 : 3, lh: 12 });
          } else if (n.mini) {
            V.box(g, G.x, G.y, G.w, G.h, { r: 6, fill: C.bg });
            V.box(g, G.x, G.y, G.w, G.h, { r: 6, fill: V.alpha(col, 0.18), stroke: V.alpha(col, 0.7), lw: 1.1 });
            V.text(g, n.t, G.cx, G.cy + 1, { size: narrow ? 9.5 : 10.5, align: 'center', base: 'middle', color: C.ink, weight: 600, maxW: G.w - 4 });
          } else {
            V.box(g, G.x, G.y, G.w, G.h, { r: 9, fill: V.alpha(col, 0.14), stroke: V.alpha(col, 0.75), lw: 1.3 });
            const th = para(g, n.t, G.cx, G.y + 16, G.w - 18, { size: narrow ? 10.5 : 12, align: 'center', weight: 700, color: C.ink, max: 2, lh: narrow ? 12 : 14 });
            if (n.s) para(g, n.s, G.cx, G.y + 18 + th, G.w - 10, { size: narrow ? 9.5 : 10.5, align: 'center', color: C.muted, max: narrow ? 3 : 4, lh: narrow ? 11.5 : 12.5 });
          }
          if (n.src) { const bx = G.x + G.w - 3, by = G.y + 3, bc = n.src === 'P' ? C.a : n.src === 'S' ? C.d : C.muted; V.dot(g, bx, by, 6.5, V.alpha(bc, 0.95)); V.text(g, n.src, bx, by + 0.5, { size: 8.5, align: 'center', base: 'middle', weight: 700, color: C.bg }); }
        });
      });
    })();

    /* ============================================================ FIG 9 — o laço de RL */
    (function () {
      const fig = root.querySelector('#fig-laco'); if (!fig) return;
      adaptH(fig, 405, 630, 680);
      const Z = [-1.2, -0.6, -0.3, 0, 0.2, 0.5, 0.9, 1.6];
      const S = { arr: 'col', sig: 0.6, mis: false };
      const ctl = UI.controls(fig);
      wrapCtl(UI.seg(ctl, { label: 'Arrangement', options: [{ v: 'col', label: 'colocated' }, { v: 'des', label: 'disaggregated' }, { v: 'asy', label: 'asynchronous' }], value: 'col', onChange: (v) => { S.arr = v; upd(); } }));
      nocase(UI.slider(ctl, { label: 'length tail (σ)', min: 0.1, max: 1.2, step: 0.1, value: 0.6, fmt: (v) => nf(v, 1), onInput: (v) => { S.sig = v; upd(); } }));
      UI.toggle(ctl, { label: 'mismatch μ ≠ π (IcePop)', value: false, onChange: (v) => { S.mis = v; } });
      const rU = UI.readout(ctl, 'generation utilization', '—'), rT = UI.readout(ctl, '3 iterations (a.u.)', '—');
      const lens = () => { const L = Z.map((z) => Math.exp(S.sig * z)); const m = L.reduce((a, b) => a + b, 0) / L.length; return L.map((x) => x / m); };
      function sched() {
        const L = lens(), mx = Math.max(...L);
        const gen = 2.2 * mx, sc = 0.5, tr = 2.4, sy = S.arr === 'col' ? 0.2 : 0.4;
        const lanes = [];
        if (S.arr === 'col') { const a = []; let x = 0; for (let i = 0; i < 3; i++) { a.push(['g', x, gen]); x += gen; a.push(['s', x, sc]); x += sc; a.push(['t', x, tr]); x += tr; a.push(['y', x, sy]); x += sy; } lanes.push({ n: 'all GPUs', segs: a }); return { lanes, total: lanes[0].segs.reduce((m, s) => Math.max(m, s[1] + s[2]), 0), L, mx }; }
        if (S.arr === 'des') { const a = [], b = []; let x = 0; for (let i = 0; i < 3; i++) { a.push(['g', x, gen]); a.push(['i', x + gen, sc + tr + sy]); b.push(['i', x, gen]); b.push(['s', x + gen, sc]); b.push(['t', x + gen + sc, tr]); b.push(['y', x + gen + sc + tr, sy]); x += gen + sc + tr + sy; } lanes.push({ n: 'rollout GPUs', segs: a }, { n: 'training GPUs', segs: b }); return { lanes, total: x, L, mx }; }
        const a = [], b = []; let x = 0, bt = 0;
        for (let i = 0; i < 3; i++) { a.push(['g', x, gen]); const st = Math.max(x + gen, bt); if (st > bt) b.push(['i', bt, st - bt]); b.push(['s', st, sc]); b.push(['t', st + sc, tr]); b.push(['y', st + sc + tr, sy]); bt = st + sc + tr + sy; x += gen + 0.15; }
        if (x < bt) a.push(['i', x, bt - x]);
        lanes.push({ n: 'rollout GPUs', segs: a }, { n: 'training GPUs', segs: b });
        return { lanes, total: Math.max(x, bt), L, mx };
      }
      function upd() { const s = sched(); const m = s.L.reduce((a, b) => a + b, 0) / s.L.length; const u = m / s.mx; rU.set(nf(u * 100, 0) + '%', u < 0.5 ? 'bad' : u < 0.75 ? 'warn' : 'ok'); rT.set(nf(s.total, 1)); }
      upd();
      V.fig(fig, (g, w, h, t) => {
        const C = V.C(), narrow = w < 680, pad = 12;
        const sc = sched();
        const segCol = { g: C.a, s: C.d, t: C.b, y: C.f, i: C.muted };
        const segLab = { g: 'generate', s: 'judge', t: 'train', y: 'sync', i: 'idle' };
        // tempo cíclico ao longo do cronograma
        const P = 9, ph = (t % P) / P, tNow = ph * sc.total;
        const L0 = sc.lanes[0];
        let cur = L0.segs.find((s) => tNow >= s[1] && tNow < s[1] + s[2]) || L0.segs[0];
        const phase = cur[0];
        // ---- laço (esquerda/topo)
        let loop, right, tl;
        if (!narrow) { loop = { x: pad, y: 8, w: w * 0.42, h: h * 0.56 }; right = { x: w * 0.46, y: 14, w: w * 0.54 - pad, h: h * 0.56 - 14 }; tl = { x: pad, y: h * 0.6, w: w - 2 * pad, h: h * 0.4 - 10 }; }
        else { loop = { x: pad, y: 6, w: w - 2 * pad, h: 230 }; right = { x: pad, y: 244, w: w - 2 * pad, h: 210 }; tl = { x: pad, y: 466, w: w - 2 * pad, h: h - 476 }; }
        const nbw = narrow ? 104 : Math.min(122, loop.w * 0.33);
        const lcx = loop.x + loop.w / 2, lcy = loop.y + loop.h / 2 + 4, rx = loop.w / 2 - nbw / 2 - 4, ry = loop.h / 2 - 26;
        const NODES = [
          { a: Math.PI, t: 'Prompts', s: 'dataset', k: 'muted', ph: '' },
          { a: -Math.PI / 2, t: 'Inference engine', s: 'vLLM · SGLang', k: 'a', ph: 'g' },
          { a: 0, t: 'Judge', s: 'verifier · RM', k: 'd', ph: 's' },
          { a: Math.PI / 2, t: 'Trainer', s: 'FSDP · Megatron', k: 'b', ph: 't' },
        ];
        const np = NODES.map((n) => [lcx + Math.cos(n.a) * rx, lcy + Math.sin(n.a) * ry]);
        // arcos (só os trechos fora das caixas dos nós)
        const hbw = nbw / 2 + 4, hbh = 24;
        const inBox = (x, y) => np.some(([nx, ny]) => Math.abs(x - nx) < hbw && Math.abs(y - ny) < hbh);
        const ARCS = [[Math.PI, Math.PI * 1.5, 'g'], [Math.PI * 1.5, Math.PI * 2, 's'], [0, Math.PI / 2, 't']];
        ARCS.forEach(([a0, a1, ph2]) => {
          const pts = [];
          for (let k = 0; k <= 60; k++) { const a = V.lerp(a0, a1, k / 60), x = lcx + Math.cos(a) * rx, y = lcy + Math.sin(a) * ry; if (!inBox(x, y)) pts.push([x, y]); }
          if (pts.length < 3) return;
          const live = phase === ph2;
          g.beginPath(); pts.forEach(([x, y], k) => (k ? g.lineTo(x, y) : g.moveTo(x, y)));
          g.strokeStyle = live ? V.alpha(C.ink, 0.8) : V.alpha(C.muted, 0.45); g.lineWidth = live ? 2.2 : 1.3; g.stroke();
          const A = pts[pts.length - 3], B = pts[pts.length - 1];
          V.arrow(g, A[0], A[1], B[0], B[1], { color: live ? V.alpha(C.ink, 0.8) : V.alpha(C.muted, 0.6), lw: 1, head: 7 });
          if (live) for (let q = 0; q < 4; q++) { const u = ((t * 0.8 + q / 4) % 1) * (pts.length - 1), k = Math.floor(u), f = u - k, P0 = pts[k], P1 = pts[Math.min(pts.length - 1, k + 1)]; V.dot(g, V.lerp(P0[0], P1[0], f), V.lerp(P0[1], P1[1], f), 2.6, C.a); }
        });
        // sincronização dos pesos: do treinador de volta ao motor, pelo meio
        { const liveY = phase === 'y', x = lcx, y1 = np[3][1] - hbh, y2 = np[1][1] + hbh;
          g.setLineDash([5, 4]); g.lineDashOffset = t * 16; V.line(g, x, y1, x, y2 + 8, { color: liveY ? C.f : V.alpha(C.f, 0.5), lw: liveY ? 2.2 : 1.3 }); g.setLineDash([]);
          V.arrow(g, x, y2 + 14, x, y2 + 2, { color: liveY ? C.f : V.alpha(C.f, 0.6), lw: 1, head: 7 });
          if (liveY) for (let q = 0; q < 3; q++) { const u = (t * 1.2 + q / 3) % 1; V.dot(g, x, V.lerp(y1, y2, u), 2.8, C.f); }
          const ly0 = lcy + ry * 0.4; V.text(g, 'weights', x - 7, ly0, { size: 10, color: C.f, weight: 700, align: 'right' }); V.text(g, S.arr === 'col' ? 'same memory' : 'over the network (NCCL)', x - 7, ly0 + 12, { size: 10, color: C.f, align: 'right' }); }
        NODES.forEach((n, i) => {
          const [x, y] = np[i], on = n.ph === phase, col = n.k === 'muted' ? C.muted : C[n.k];
          const bw = nbw, bh = 40;
          V.box(g, x - bw / 2, y - bh / 2, bw, bh, { r: 9, fill: V.alpha(col, on ? 0.3 : 0.12), stroke: on ? col : V.alpha(col, 0.6), lw: on ? 2 : 1.1 });
          V.text(g, n.t, x, y - 3, { size: 11.5, weight: 700, align: 'center', color: C.ink, maxW: bw - 8 });
          V.text(g, n.s, x, y + 12, { size: 10, align: 'center', color: C.muted, maxW: bw - 8 });
        });
        // ---- painel direito: rollouts ou descompasso
        if (!S.mis) {
          V.text(g, 'a batch of 8 rollouts decoded together', right.x, right.y + 4, { size: 11.5, weight: 700, color: C.ink, maxW: right.w });
          const L = sc.L, mx = sc.mx;
          const genSeg = phase === 'g' ? cur : null;
          const prog = genSeg ? ((tNow - genSeg[1]) / genSeg[2]) * mx : mx;
          const bh = Math.min(15, (right.h * 0.55) / 8 - 4), bx = right.x + 20, bwid = right.w - 70;
          L.forEach((len, i) => {
            const y = right.y + 18 + i * (bh + 4);
            V.box(g, bx, y, (len / mx) * bwid, bh, { r: 3, fill: V.alpha(C.muted, 0.12) });
            const fl = Math.min(len, prog) / mx * bwid;
            V.box(g, bx, y, Math.max(1, fl), bh, { r: 3, fill: V.alpha(C.a, prog >= len ? 0.4 : 0.85) });
            V.text(g, String(i + 1), bx - 6, y + bh / 2 + 1, { size: 9.5, font: 'mono', align: 'right', base: 'middle', color: C.muted });
            if (prog >= len) V.text(g, '✓', bx + fl + 8, y + bh / 2 + 1, { size: 10.5, base: 'middle', color: C.a, weight: 700 });
          });
          // utilização: fração de sequências ainda ativas
          const uy = right.y + 18 + 8 * (bh + 4) + 12, uh = right.y + right.h - uy - 16;
          if (uh > 20) {
            V.text(g, 'active sequences', right.x, uy - 2, { size: 10, color: C.muted });
            g.beginPath(); g.moveTo(bx, uy + uh);
            const sorted = L.slice().sort((a, b) => a - b);
            let prev = bx;
            for (let k = 0; k <= sorted.length; k++) { const frac = (8 - k) / 8; const xe = k < sorted.length ? bx + (sorted[k] / mx) * bwid : bx + bwid; g.lineTo(prev, uy + uh - frac * uh); g.lineTo(xe, uy + uh - frac * uh); prev = xe; }
            g.lineTo(bx + bwid, uy + uh); g.closePath(); g.fillStyle = V.alpha(C.a, 0.2); g.fill(); g.strokeStyle = V.alpha(C.a, 0.8); g.lineWidth = 1.2; g.stroke();
            V.line(g, bx + (Math.min(prog, mx) / mx) * bwid, uy, bx + (Math.min(prog, mx) / mx) * bwid, uy + uh, { color: V.alpha(C.ink, 0.5), lw: 1, dash: [3, 3] });
            V.text(g, 'tail: GPU waiting on the longest', bx + bwid, uy + uh + 12, { size: 9.5, align: 'right', color: C.muted, maxW: bwid });
          }
        } else {
          V.text(g, 'same weights, different probabilities', right.x, right.y + 4, { size: 11.5, weight: 700, color: C.ink, maxW: right.w });
          const pin = [0.50, 0.25, 0.20, 0.40], ptr = [0.49, 0.40, 0.50, 0.18];
          const n = 4, cw = Math.min(84, (right.w - 20) / n - 10), top = right.y + 24, bh = right.h * (narrow ? 0.3 : 0.42);
          for (let i = 0; i < n; i++) {
            const x = right.x + 10 + i * (cw + 10), rho = ptr[i] / pin[i], keep = rho >= 0.5 && rho <= 2;
            V.box(g, x, top, cw, bh + 52, { r: 8, fill: V.alpha(keep ? C.a : RED(), 0.07), stroke: V.alpha(keep ? C.a : RED(), 0.5), lw: 1.1 });
            const bb = cw * 0.28;
            [[pin[i], C.c, 'inf.'], [ptr[i], C.b, 'train']].forEach(([v, col, lab], k) => { const bx2 = x + cw * 0.18 + k * (bb + cw * 0.1); const hh = v * bh * 1.6 * (1 + 0.04 * Math.sin(t * 3 + i + k)); V.box(g, bx2, top + 8 + bh - Math.min(bh, hh), bb, Math.min(bh, hh), { r: 3, fill: V.alpha(col, 0.8) }); V.text(g, lab, bx2 + bb / 2, top + bh + 22, { size: 9, align: 'center', color: C.muted }); });
            V.text(g, 'ρ = ' + nf(rho, 2), x + cw / 2, top + bh + 40, { size: 10.5, font: 'mono', align: 'center', color: keep ? C.a : RED(), weight: 700, maxW: cw });
            if (!keep) { g.strokeStyle = V.alpha(RED(), 0.7 + 0.3 * Math.sin(t * 5)); g.lineWidth = 2; g.beginPath(); g.moveTo(x + 6, top + 6); g.lineTo(x + cw - 6, top + bh + 46); g.moveTo(x + cw - 6, top + 6); g.lineTo(x + 6, top + bh + 46); g.stroke(); }
          }
          para(g, 'IcePop (GLM-5): tokens with ρ = π_train/π_inference outside [0.5, 2] are dropped from the gradient. In MoE, ~94% of tokens switch experts in some layer between the engines (R3).', right.x, top + bh + 70, right.w, { size: 10.5, color: C.muted, max: 3 });
        }
        // ---- linha do tempo
        V.text(g, 'timeline: 3 iterations', tl.x, tl.y + 4, { size: 11.5, weight: 700, color: C.ink });
        const laneH = Math.min(30, (tl.h - 50) / sc.lanes.length - 8), lx = tl.x + (narrow ? 0 : 108), lw = tl.w - (narrow ? 0 : 108);
        const X = (v) => lx + (v / sc.total) * lw;
        sc.lanes.forEach((ln, li) => {
          const y = tl.y + 16 + (narrow ? 12 : 0) + li * (laneH + (narrow ? 20 : 8));
          if (narrow) V.text(g, ln.n, tl.x, y - 3, { size: 10, color: C.muted }); else V.text(g, ln.n, tl.x, y + laneH / 2 + 4, { size: 10.5, color: C.muted, maxW: 100 });
          ln.segs.forEach(([k, s0, d]) => {
            const x0 = X(s0), x1 = X(s0 + d);
            if (k === 'i') { g.save(); V.rr(g, x0, y, Math.max(1, x1 - x0 - 1), laneH, 4); g.clip(); g.strokeStyle = V.alpha(C.muted, 0.35); g.lineWidth = 1; for (let q = x0 - laneH; q < x1; q += 7) { g.beginPath(); g.moveTo(q, y + laneH); g.lineTo(q + laneH, y); g.stroke(); } g.restore(); return; }
            V.box(g, x0, y, Math.max(1, x1 - x0 - 1), laneH, { r: 4, fill: V.alpha(segCol[k], 0.75) });
            if (x1 - x0 > 38) V.text(g, segLab[k], (x0 + x1) / 2, y + laneH / 2 + 1, { size: 10, align: 'center', base: 'middle', color: C.ink, weight: 600 });
          });
        });
        const yEnd = tl.y + 16 + (narrow ? 12 : 0) + sc.lanes.length * (laneH + (narrow ? 20 : 8));
        if (!narrow) V.line(g, X(tNow), tl.y + 12, X(tNow), yEnd - 4, { color: V.alpha(C.ink, 0.75), lw: 1.5 });
        else sc.lanes.forEach((ln, li) => { const y = tl.y + 28 + li * (laneH + 20); V.line(g, X(tNow), y - 2, X(tNow), y + laneH + 2, { color: V.alpha(C.ink, 0.75), lw: 1.5 }); }); // só sobre as faixas, sem cortar os rótulos
        let lgx = lx;
        Object.keys(segLab).forEach((k) => { const l = segLab[k]; V.box(g, lgx, yEnd + 4, 10, 10, { r: 2, fill: V.alpha(segCol[k], k === 'i' ? 0.35 : 0.75) }); V.text(g, l, lgx + 14, yEnd + 13, { size: 10, color: C.muted }); lgx += tw(g, l, 10) + 26; });
      });
    })();

    /* ============================================================ FIG 10 — o metrô do Módulo 2 */
    (function () {
      const fig = root.querySelector('#fig-metro'); if (!fig) return;
      adaptH(fig, 420, 650, 640);
      const ST = [
        { id: 'pos-treino-mapa', n: 50, l: 'Post-training map', line: 0, c: 0 },
        { id: 'treino-memoria-lora', n: 51, l: 'Training memory & LoRA', line: 0, c: 1 },
        { id: 'sft', n: 52, l: 'SFT', line: 0, c: 2 },
        { id: 'preferencias-reward-models', n: 53, l: 'Preferences & RMs', line: 1, c: 2 },
        { id: 'rl-fundamentos', n: 54, l: 'RL basics', line: 1, c: 3 },
        { id: 'ppo-rlhf', n: 55, l: 'PPO & RLHF', line: 1, c: 4 },
        { id: 'grpo-rlvr', n: 56, l: 'GRPO & RLVR', line: 1, c: 5 },
        { id: 'dpo', n: 57, l: 'DPO', line: 2, c: 4.5 },
        { id: 'ipo-kto-simpo', n: 58, l: 'IPO, KTO & SimPO', line: 2, c: 5.5 },
        { id: 'awr-contrastivo', n: 59, l: 'AWR & contrastive', line: 2, c: 6.5 },
        { id: 'constitutional-ai', n: 60, l: 'Constitutional AI', line: 3, c: 5 },
        { id: 'self-play-debate', n: 61, l: 'Self-play & debate', line: 3, c: 6 },
        { id: 'avaliacao-pos-treino', n: 62, l: 'Evaluation', line: 3, c: 7 },
        { id: 'receita-pos-treino', n: 63, l: 'Post-training recipe', line: 3, c: 8 },
      ];
      const PRE = {
        'pos-treino-mapa': ['geracao-amostragem', 'aprender-com-dados'],
        'treino-memoria-lora': ['pos-treino-mapa', 'memoria-unificada', 'paralelismos', 'quantizacao-fundamentos'],
        sft: ['treino-memoria-lora', 'tokens-embeddings', 'aprender-com-dados'],
        'preferencias-reward-models': ['pos-treino-mapa', 'sft'],
        'rl-fundamentos': ['preferencias-reward-models', 'aprender-com-dados'],
        'ppo-rlhf': ['rl-fundamentos', 'preferencias-reward-models'],
        'grpo-rlvr': ['ppo-rlhf', 'rl-fundamentos'],
        dpo: ['preferencias-reward-models', 'ppo-rlhf'],
        'ipo-kto-simpo': ['dpo'],
        'awr-contrastivo': ['dpo', 'rl-fundamentos', 'sft'],
        'constitutional-ai': ['preferencias-reward-models', 'ppo-rlhf', 'sft', 'abliteracao'],
        'self-play-debate': ['grpo-rlvr', 'dpo', 'constitutional-ai'],
        'avaliacao-pos-treino': ['pos-treino-mapa', 'medindo-qualidade', 'preferencias-reward-models'],
        'receita-pos-treino': ['sft', 'dpo', 'grpo-rlvr', 'avaliacao-pos-treino', 'receita-do-zero'],
      };
      const M1 = { 'geracao-amostragem': 'Generation & sampling', 'aprender-com-dados': 'Learning from data', 'memoria-unificada': 'Unified memory', paralelismos: 'Parallelism', 'quantizacao-fundamentos': 'Quantization fundamentals', 'tokens-embeddings': 'Tokens & embeddings', abliteracao: 'Abliteration', 'medindo-qualidade': 'Measuring quality', 'receita-do-zero': 'Recipe from scratch' };
      const byId = {}; ST.forEach((s) => { byId[s.id] = s; });
      const LINES = [['X · foundations', 'a'], ['XI · preferences & RL', 'b'], ['XII · direct optimization', 'c'], ['XIII · alignment & practice', 'd']];
      const S = { tgt: 'dpo', route: [], ext: [], t0: 0 };
      function route(id) {
        const set = new Set(), ext = new Set();
        const go = (x) => { if (set.has(x)) return; if (!byId[x]) { ext.add(x); return; } set.add(x); (PRE[x] || []).forEach(go); };
        go(id);
        S.route = ST.filter((s) => set.has(s.id));
        S.ext = Array.from(ext);
      }
      route(S.tgt);
      let Hh = null;
      const ctl = UI.controls(fig);
      wrapCtl(UI.select(ctl, { label: 'I want to understand…', options: [
        { v: 'dpo', label: 'DPO: preferences without RL' }, { v: 'grpo-rlvr', label: 'GRPO and RLVR: reasoning' }, { v: 'ppo-rlhf', label: 'RLHF with PPO' },
        { v: 'sft', label: 'SFT, distillation and MOPD' }, { v: 'constitutional-ai', label: 'Constitutional AI' }, { v: 'self-play-debate', label: 'Self-play and debate' },
        { v: 'avaliacao-pos-treino', label: 'evaluating a post-trained model' }, { v: 'receita-pos-treino', label: 'the full recipe: train, quantize, serve' },
      ], value: 'dpo', onChange: (v) => { S.tgt = v; route(v); S.t0 = Hh ? Hh.t : 0; rN.set(S.route.length + ' chapters'); } }));
      const rN = UI.readout(ctl, 'route', S.route.length + ' chapters');
      Hh = V.fig(fig, (g, w, h, t) => {
        const C = V.C(), narrow = w < 640, pad = 14;
        const lc = (i) => C[LINES[i][1]];
        const extH = narrow ? 128 : 56;
        let pos;
        if (!narrow) { const cw = (w - 2 * pad - 120) / 9, rh = (h - 24 - extH) / 4; pos = (s) => [pad + 120 + cw * (s.c + 0.5), 20 + rh * (s.line + 0.5)]; LINES.forEach((L, i) => { V.text(g, L[0], pad, 20 + rh * (i + 0.5) + 4, { size: 11, weight: 700, color: lc(i), maxW: 116 }); }); }
        else { const cw = (w - 2 * pad) / 4, rh = (h - 44 - extH) / 9; pos = (s) => [pad + cw * (s.line + 0.5), 40 + rh * (s.c + 0.5)]; LINES.forEach((L, i) => { V.text(g, L[0].split(' · ')[0], pad + cw * (i + 0.5), 16, { size: 11, weight: 700, align: 'center', color: lc(i) }); V.text(g, L[0].split(' · ')[1], pad + cw * (i + 0.5), 30, { size: 9, align: 'center', color: C.muted, maxW: cw - 4 }); }); }
        // trilhos das linhas
        LINES.forEach((L, i) => { const ss = ST.filter((s) => s.line === i); g.beginPath(); ss.forEach((s, k) => { const [x, y] = pos(s); if (k) g.lineTo(x, y); else g.moveTo(x, y); }); g.strokeStyle = V.alpha(lc(i), 0.35); g.lineWidth = 6; g.lineCap = 'round'; g.stroke(); g.lineCap = 'butt'; });
        // conexões de pré-requisito
        const inR = new Set(S.route.map((s) => s.id));
        ST.forEach((s) => (PRE[s.id] || []).forEach((p) => {
          if (!byId[p]) return; const a = pos(byId[p]), b = pos(s); const on = inR.has(s.id) && inR.has(p);
          g.beginPath(); g.moveTo(a[0], a[1]); const mxp = (a[0] + b[0]) / 2, myp = (a[1] + b[1]) / 2;
          g.quadraticCurveTo(mxp + (narrow ? 18 : 0), myp + (narrow ? 0 : 14), b[0], b[1]);
          g.strokeStyle = on ? V.alpha(C.ink, 0.6) : V.alpha(C.muted, 0.18); g.lineWidth = on ? 2.2 : 1; if (!on) g.setLineDash([3, 4]); g.stroke(); g.setLineDash([]);
        }));
        // trem percorrendo a rota em ordem
        const R = S.route;
        if (R.length > 1) {
          const segT = 1.1, tot = segT * (R.length - 1) + 1.5, u = ((t - S.t0) % tot) / segT;
          const k = Math.min(R.length - 2, Math.floor(u)), f = Math.min(1, u - k);
          const a = pos(R[k]), b = pos(R[k + 1]), e = V.ease.inOut(f);
          const x = V.lerp(a[0], b[0], e), y = V.lerp(a[1], b[1], e);
          V.box(g, x - 13, y - 8, 26, 16, { r: 5, fill: C.d, stroke: C.ink, lw: 1 });
          for (let q = 0; q < 3; q++) V.box(g, x - 10 + q * 7, y - 5, 5, 5, { r: 1, fill: C.bg });
        }
        // estações
        ST.forEach((s) => {
          const [x, y] = pos(s), on = inR.has(s.id), tgt = s.id === S.tgt, col = lc(s.line);
          const rr = tgt ? 13 : 11;
          if (tgt) V.glow(g, x, y, 28, col, 0.6 + 0.3 * Math.sin(t * 3));
          g.beginPath(); g.arc(x, y, rr, 0, Math.PI * 2); g.fillStyle = on ? col : C.bg; g.fill(); g.strokeStyle = on ? C.ink : V.alpha(col, 0.8); g.lineWidth = on ? 1.6 : 1.4; g.stroke();
          V.text(g, String(s.n), x, y + 1, { size: 10, font: 'mono', weight: 700, align: 'center', base: 'middle', color: on ? C.bg : C.muted });
          const lw2 = narrow ? (w - 2 * pad) / 4 - 4 : (w - 2 * pad - 120) / 9 + 6;
          const fsz = narrow ? 9 : 10, lhh = narrow ? 10.5 : 12, LL = lines(g, s.l, lw2, fsz, 'sans', on ? 600 : 400).slice(0, 2);
          const lwMax = Math.max(...LL.map((l) => tw(g, l, fsz, 'sans', on ? 600 : 400)));
          V.box(g, x - lwMax / 2 - 3, y + rr + 3, lwMax + 6, LL.length * lhh + 4, { r: 4, fill: V.alpha(C.bg, 0.82) });
          LL.forEach((l, i) => V.text(g, l, x, y + rr + 12 + i * lhh, { size: fsz, align: 'center', color: on ? C.ink : C.muted, weight: on ? 600 : 400 }));
        });
        // baldeações (Módulo 1)
        const ey = h - extH + 6;
        V.text(g, 'transfers in Module 1:', pad, ey + 12, { size: 10.5, color: C.muted, weight: 600 });
        let cx = narrow ? pad : pad + tw(g, 'transfers in Module 1:', 10.5, 'sans', 600) + 8, cy2 = narrow ? ey + 18 : ey;
        S.ext.forEach((id) => { const lab = M1[id] || id, cw = tw(g, lab, 10, 'sans', 500) + 14; if (cx + cw > w - pad) { cx = pad; cy2 += 22; } if (cy2 + 18 > h - 1) return; chip(g, cx, cy2, cw, 18, lab, { size: 10, fill: V.alpha(C.muted, 0.14) }); cx += cw + 6; });
      });
    })();
  },
});

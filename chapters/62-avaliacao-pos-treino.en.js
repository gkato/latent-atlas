LA.chapter({
  id: 'avaliacao-pos-treino',
  part: 13,
  order: 62,
  title: 'Evaluating post-trained models',
  navTitle: 'Evaluation',
  subtitle: 'We want the model to change, but only where we asked: four axes, win rate with confidence intervals, LLM judges and their biases, Bradley–Terry and the leaderboard illusion, length control, verifiable benchmarks, contamination, safety in both directions and KL as a budget.',
  minutes: 40,
  recipes: [],
  tags: ['evaluation', 'win rate', 'bootstrap', 'LLM-as-judge', 'Bradley–Terry', 'AlpacaEval-LC', 'IFEval', 'pass^k', 'contamination', 'XSTest', 'reward hacking', 'KL', 'lm-eval-harness'],
  goals: [
    'Evaluate a post-trained model on the <b>four axes</b> (target, regression, safety, drift) against the starting model, with <b>preregistered gates</b>',
    'Compute win rate with a <b>bootstrap CI</b>, size the number of prompts and use paired tests (McNemar) to find regressions hidden in the average',
    'Build an <b>LLM judge</b> that resists position, verbosity and self-preference biases, using the three Module 1 families as a panel, and read Bradley–Terry/Elo rankings without falling for the winner\'s curse',
    'Choose benchmarks that are still alive in 2026, detect <b>contamination</b> and <b>reward hacking</b>, and tell apart the two KLDs: post-training drift and quantization damage',
  ],
  prereqs: ['pos-treino-mapa', 'medindo-qualidade', 'preferencias-reward-models'],
  summary: [
    'Four axes, always against the starting model and under the same conditions (template, <code>enable_thinking</code>, sampling, server): <b>target</b> ↑, <b>regression</b> ≈ 0, <b>safety</b> in both directions and <b>drift</b> (KL per token) high in the domain and ≈ 0 outside it. Preregister metrics and gates before looking at any numbers.',
    'Win rate \\(=(W+T/2)/n\\), paired by prompt. Near 50%: ±9.8 pp with 100 prompts, ±4.9 with 400, ±3.1 with 1,000; ~384 for ±5 pp. A per-prompt bootstrap (\\(B = 2{,}000\\), percentiles 2.5/97.5) gives a CI for any statistic; for differences, resample pairs.',
    'LLM judges agree with humans in more than 80% of cases, but they have position bias (judge in both orders; inconsistency = tie), verbosity bias (length control) and self-preference bias (a judge from another family; a PoLL panel). In the lab (GLM teacher, Qwen student), the impartial judge is DeepSeek.',
    'Bradley–Terry \\(P(i\\succ j)=\\sigma(s_i-s_j)\\) does not depend on the order of the battles, unlike online Elo; the arena reports BT on an Elo-like scale, with style control. Publishing the best of many variants inflates the score (“Leaderboard Illusion,” ~27 Meta variants): keep separate selection and reporting sets.',
    'WR-LC: a logistic regression on \\(\\tanh(\\Delta\\ell/\\sigma)\\); the WR-LC is the counterfactual \\(\\Delta\\ell = 0\\) (AlpacaEval: Spearman with the arena from 0.94 to 0.98). Arena-Hard v2 and MT-Bench serve as out-of-domain and multi-turn signals.',
    'Regression: IFEval, GSM8K/AIME (pass@1 averaged over k samples), code, BFCL/τ² (pass^k for agents), XSTest, generic KL; McNemar \\(\\chi^2=(|b-c|-1)^2/(b+c)\\). Saturated classics are only good for regression; contamination is detected with n-grams, Min-K% and dated sets.',
    'Safety: benign fine-tuning erodes refusals, DPO can shift mass to the opposite (refusal 74.4% → 33.4%) and the refusal direction becomes a cheap probe. Drift: KL is the post-training budget (the \\(k_3\\) estimator); Gao et al. show the gold reward rising and then falling with \\(\\sqrt{\\mathrm{KL}}\\). Quantization is measured against the <b>BF16 merge of the fine-tune</b>, not against the base.',
  ],
  quiz: [
    { q: 'Your DPO beat the base on 56 of 100 prompts (no ties). What can you conclude?', o: ['That the DPO is better, because 56% &gt; 50%', 'Nothing yet: the 95% CI is ±9.8 points and includes 50%; separating 55% from 50% comfortably takes ~400–800 prompts', 'That the DPO is worse, because the margin is small', 'That it is enough to run the same set two more times and take the average'], a: 1, e: 'With \\(n=100\\) and \\(p\\approx0.5\\), \\(1.96\\sqrt{0.25/100}=0.098\\). The interval runs from ~46% to ~66%. Repeating the same set does not help: the variance that matters comes from sampling prompts, and only more prompts (or pairing) reduce it.' },
    { q: 'In the order (A=candidate, B=base) the judge picks A; in the swapped order, it picks position A again (now the base). How should this pair be counted?', o: ['A win for the candidate, because the first verdict counts', 'A loss for the candidate, because the last verdict counts', 'A tie: the judge chose the position, not the model, and that is position bias', 'Discard the pair and leave it out of the denominator'], a: 2, e: 'If the verdict follows the position, it says nothing about quality. Counting it as a tie cancels the push: with bias \\(b\\), the expected value across both orders is \\(\\tfrac12[\\sigma(\\Delta q+b)+\\sigma(\\Delta q-b)]\\), which is exactly 0.5 when quality is equal. Discarding the pair would bias the denominator toward the “easy” pairs.' },
    { q: 'The lab uses GLM as the teacher to generate the data and trains Qwen3.8-27B. Which judge among the three Module 1 recipes is impartial?', o: ['GLM, because it is the strongest', 'Qwen3.8-Flash-Next, because it knows the student\'s architecture', 'DeepSeek: its family differs from both the teacher (the student learned to sound like GLM) and the student', 'Any of them, as long as you judge in both orders'], a: 2, e: 'Swapping positions cancels position bias, but not self-preference, which pushes toward the same side in both orders. Exclude the teacher\'s family and the student\'s; GLM and Qwen-Flash-Next can serve as secondary judges, with their agreement with DeepSeek reported separately.' },
    { q: 'A candidate has a raw WR of 71% and a WR-LC of 52% (CI [47, 57]). What is the correct reading?', o: ['The candidate is much better; LC is just a cosmetic correction', 'The advantage comes almost entirely from length: at equal length, the candidate ties with the base and fails the target gate', 'The judge is broken, because the two numbers should be equal', 'The WR-LC is lower because it ignores ties'], a: 1, e: 'The WR-LC is the counterfactual \\(\\Delta\\ell=0\\) of the logistic regression: how often the candidate would win if it wrote at the same length. A large \\(\\varphi\\) and a WR-LC near 50% are the signature of exploited verbosity, common in DPO; the training-side remedy is LN-DPO or SimPO.' },
    { q: 'On 500 items, the base gets 78% right and the candidate 75%; 20 items went from right to wrong and 5 from wrong to right. What does McNemar\'s test say?', o: ['Nothing: the accuracies are less than 5 points apart', '\\(\\chi^2=(|20-5|-1)^2/25=7.84\\), \\(p\\approx0.005\\): a significant regression, even though the unpaired CIs overlap', 'That the candidate improved, because it gained 5 items', 'That the test must be run with all 500 items in the denominator'], a: 1, e: 'The paired test only looks at the discordant items (\\(b\\) and \\(c\\)); the ones both models get right or both get wrong say nothing about the difference. Since an item\'s difficulty is the same for both models, it cancels out, and the paired interval of the difference (about −4.9 to −1.1 pp) is much narrower than the unpaired one (about −8 to +2).' },
    { q: 'You are going to measure the damage from quantizing your post-trained model to NVFP4. What is the reference P for the KLD?', o: ['The original base model in BF16', 'The BF16 merge of the post-trained model', 'The official NVFP4 of the base', 'The LoRA adapter alone'], a: 1, e: 'Against the base, the KLD would mix the desired post-training drift with the quantization error. The question “how much damage did quantization do?” takes the BF16 fine-tune (merge) as its reference. To calibrate, compare with the KLD that the same format causes on the base, measured with the same script, and with the BF16 × BF16 floor.' },
  ],
  mount(root, LA) {
    const V = LA.viz, UI = LA.ui;
    const RED = () => V.C().red || '#FB2C36';
    const fmt = (x, d = 1) => (+x).toFixed(d);
    const sgn = (x, d = 1) => (x > 0.00001 ? '+' : x < -0.00001 ? '−' : '') + fmt(Math.abs(x), d);
    const sig = (z) => 1 / (1 + Math.exp(-z));
    const logit = (p) => Math.log(p / (1 - p));

    /* ------------------------------------------------ shared helpers */
    /** stage height by width; wide/narrow can be functions (a height that depends on the mode). Returns the "set" to reapply. */
    function adaptH(fig, wide, narrow, bp = 640) {
      const st = fig.querySelector('.la-fig-stage');
      const val = (v) => (typeof v === 'function' ? v() : v);
      const set = () => { const want = Math.round(val(fig.clientWidth < bp ? narrow : wide)) + 'px'; if (st.style.height !== want) st.style.height = want; };
      set();
      const ro = new ResizeObserver(set); ro.observe(fig);
      if (LA.onCleanup) LA.onCleanup(() => ro.disconnect());
      return set;
    }
    function tw(g, s, size, font = 'sans', weight = 400) { g.font = V.font(size, font, weight); return g.measureText(String(s)).width; }
    function chip(g, x, y, w, h, label, o = {}) {
      const C = V.C();
      V.box(g, x, y, w, h, { r: o.r == null ? 6 : o.r, fill: o.fill || V.alpha(C.muted, 0.12), stroke: o.stroke || V.alpha(C.muted, 0.4), lw: o.lw || 1.2 });
      if (label != null) V.text(g, label, x + w / 2, y + h / 2 + 0.5, { size: o.size || 11.5, font: o.font || 'mono', weight: o.weight || 500, color: o.color || C.ink, align: 'center', base: 'middle' });
    }
    function pillC(g, cx, cy, label, o = {}) {
      const size = o.size || 11.5, hh = o.h || 20;
      const w = tw(g, label, size, o.font || 'mono', o.weight || 600) + 14;
      chip(g, cx - w / 2, cy - hh / 2, w, hh, label, Object.assign({ weight: 600 }, o));
      return w;
    }
    /** character: rounded square with eyes and a mouth (mood: happy | sad | o) */
    function face(g, x, y, s, col, o = {}) {
      const C = V.C();
      if (o.alpha != null) g.globalAlpha = o.alpha;
      V.box(g, x - s / 2, y - s / 2, s, s, { r: s * 0.3, fill: V.alpha(col, 0.85), stroke: col, lw: 1.2 });
      const eye = C.dark ? '#06140d' : '#ffffff';
      const ex = s * 0.17, ey = y - s * 0.08, er = Math.max(1.2, s * 0.07);
      V.dot(g, x - ex, ey, er, eye); V.dot(g, x + ex, ey, er, eye);
      g.beginPath();
      if (o.mood === 'sad') g.arc(x, y + s * 0.3, s * 0.14, Math.PI * 1.15, Math.PI * 1.85);
      else if (o.mood === 'o') g.arc(x, y + s * 0.16, s * 0.07, 0, Math.PI * 2);
      else g.arc(x, y + s * 0.06, s * 0.15, Math.PI * 0.2, Math.PI * 0.8);
      g.strokeStyle = eye; g.lineWidth = Math.max(1, s * 0.06); g.stroke();
      if (o.wig) {
        const wc = C.dark ? '#dfe6ea' : '#ffffff';
        for (let k = -1; k <= 1; k++) {
          g.beginPath(); g.arc(x + k * s * 0.3, y - s * 0.52, s * 0.17, 0, Math.PI * 2);
          g.fillStyle = wc; g.fill(); g.strokeStyle = V.alpha(C.muted, 0.8); g.lineWidth = 1; g.stroke();
        }
      }
      if (o.alpha != null) g.globalAlpha = 1;
    }
    function stamp(g, cx, cy, text, col, o = {}) {
      const sc = o.scale == null ? 1 : o.scale;
      if (sc <= 0.01) return;
      g.save(); g.translate(cx, cy); g.rotate(o.rot == null ? -0.07 : o.rot); g.scale(sc, sc);
      const size = o.size || 13;
      const w = tw(g, text, size, 'display', 700) + 18;
      V.box(g, -w / 2, -size * 0.95, w, size * 1.9, { r: 6, fill: V.alpha(col, 0.14), stroke: col, lw: 2 });
      V.text(g, text, 0, 1, { size, font: 'display', weight: 700, color: col, align: 'center', base: 'middle' });
      g.restore();
    }
    function diamond(g, x, y, r, fill, stroke) {
      g.beginPath(); g.moveTo(x, y - r); g.lineTo(x + r, y); g.lineTo(x, y + r); g.lineTo(x - r, y); g.closePath();
      if (fill) { g.fillStyle = fill; g.fill(); }
      if (stroke) { g.strokeStyle = stroke; g.lineWidth = 1.6; g.stroke(); }
    }
    function erf(x) {
      const s = x < 0 ? -1 : 1; x = Math.abs(x);
      const t = 1 / (1 + 0.3275911 * x);
      const y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
      return s * y;
    }
    const Phi = (x) => 0.5 * (1 + erf(x / Math.SQRT2));
    function niceStep(range, n = 4) {
      const raw = range / n, p = Math.pow(10, Math.floor(Math.log10(raw)));
      const m = raw / p;
      return (m < 1.5 ? 1 : m < 3.5 ? 2 : m < 7.5 ? 5 : 10) * p;
    }
    const easeB = (x) => V.ease.back(V.clamp(x, 0, 1));

    /* ================================================ FIG 1 — the four-axis dashboard */
    (function () {
      const fig = root.querySelector('#fig-eixos');
      if (!fig) return;
      adaptH(fig, 370, 640);
      const KL_PREV = 0.05;
      const CANDS = [
        { label: 'Well-done SFT', desc: 'Gained in the domain and preserved the rest: what you expect from a good SFT.', raw: 66.1, wr: 64.0, lo: 59.3, hi: 68.6, dIF: -0.8, dIFlo: -2.0, dIFhi: 0.4, ref: 0.4, refLo: -0.8, refHi: 1.6, sr: 0.01, srLo: -0.02, srHi: 0.04, klD: 0.38, klG: 0.03 },
        { label: 'Verbose DPO', desc: 'Won on the raw score because it talks more; with length controlled, it ties with the base.', raw: 71.2, wr: 52.4, lo: 47.6, hi: 57.1, dIF: -1.1, dIFlo: -2.4, dIFhi: 0.2, ref: -0.4, refLo: -1.6, refHi: 0.8, sr: 0.0, srLo: -0.03, srHi: 0.03, klD: 0.52, klG: 0.07 },
        { label: 'SFT that forgot', desc: 'Great in the domain, but it stopped obeying “answer in 3 bullet points”: the alignment tax.', raw: 69.0, wr: 67.5, lo: 62.9, hi: 71.9, dIF: -7.4, dIFlo: -9.6, dIFhi: -5.2, ref: 0.8, refLo: -0.4, refHi: 2.0, sr: 0.02, srLo: -0.01, srHi: 0.05, klD: 0.66, klG: 0.24 },
        { label: 'Eroded safety', desc: 'Domain data 100% benign, and it still started complying with harmful requests.', raw: 63.8, wr: 62.2, lo: 57.5, hi: 66.8, dIF: -0.9, dIFlo: -2.1, dIFhi: 0.3, ref: -1.2, refLo: -2.4, refHi: 0.0, sr: 0.17, srLo: 0.11, srHi: 0.23, klD: 0.41, klG: 0.05 },
        { label: 'RL with a gamed reward', desc: 'The reward went up; the judge and the KL expose long answers that fool the verifier.', raw: 58.0, wr: 46.5, lo: 41.6, hi: 51.4, dIF: -1.2, dIFlo: -3.0, dIFhi: 0.6, ref: 0.2, refLo: -1.0, refHi: 1.4, sr: 0.03, srLo: 0.0, srHi: 0.06, klD: 2.4, klG: 0.61 },
      ];
      const KEYS = Object.keys(CANDS[0]).filter((k) => typeof CANDS[0][k] === 'number');
      const A = {}; KEYS.forEach((k) => { A[k] = CANDS[0][k]; });
      const S = { k: 0, changed: -10 };
      const verdict = (c) => ({
        alvo: c.lo > 50 && c.wr >= 60 ? 'ok' : 'bad',
        reg: c.dIF >= -2 ? 'ok' : 'bad',
        seg: c.ref <= 2 && c.srLo <= 0 ? 'ok' : 'bad',
        der: c.klG <= 3 * KL_PREV ? 'ok' : 'warn',
      });
      const NAMES = { alvo: 'target', reg: 'regression', seg: 'safety', der: 'drift' };
      let H = null;
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Candidate', options: CANDS.map((c, i) => ({ v: i, label: c.label })), value: 0, onChange: (v) => { S.k = v; S.changed = H ? H.t : 0; upd(); } });
      const rOut = UI.readout(ctl, 'gate verdict', '—');
      function upd() {
        const v = verdict(CANDS[S.k]);
        const bad = Object.keys(v).filter((k) => v[k] === 'bad');
        if (bad.length) rOut.set('failed: ' + bad.map((k) => NAMES[k]).join(', '), 'bad');
        else if (v.der === 'warn') rOut.set('investigate drift', 'warn');
        else rOut.set('passed', 'ok');
      }
      upd();

      function gauge(g, x, y, w, o) {
        const C = V.C();
        const X = (v) => x + ((V.clamp(v, o.min, o.max) - o.min) / (o.max - o.min)) * w;
        V.text(g, o.label, x, y + 10, { size: 11, color: C.muted, maxW: w - 52 });
        V.text(g, o.fmtv(o.est), x + w, y + 10, { size: 12, font: 'mono', weight: 700, color: C.ink, align: 'right' });
        const ty = y + 26;
        const gx = X(o.gate);
        const region = (x1, x2, col) => { if (x2 - x1 < 0.5) return; g.fillStyle = V.alpha(col, 0.17); V.rr(g, x1, ty - 7, x2 - x1, 14, 3); g.fill(); };
        if (o.passRight) { region(x, gx, RED()); region(gx, x + w, C.a); } else { region(x, gx, C.a); region(gx, x + w, RED()); }
        if (o.zero != null && o.zero > o.min && o.zero < o.max) V.line(g, X(o.zero), ty - 9, X(o.zero), ty + 9, { color: V.alpha(C.muted, 0.7), lw: 1 });
        V.line(g, gx, ty - 11, gx, ty + 11, { color: C.ink, lw: 1.4, dash: [3, 2] });
        if (o.gate2 != null) V.line(g, X(o.gate2), ty - 8, X(o.gate2), ty + 8, { color: V.alpha(C.ink, 0.6), lw: 1, dash: [2, 2] });
        const xl = X(o.lo), xh = X(o.hi), xe = X(o.est);
        V.line(g, xl, ty, xh, ty, { color: C.ink, lw: 2 });
        V.line(g, xl, ty - 5, xl, ty + 5, { color: C.ink, lw: 2 });
        V.line(g, xh, ty - 5, xh, ty + 5, { color: C.ink, lw: 2 });
        if (o.ghost != null) { g.beginPath(); g.arc(X(o.ghost), ty, 5.5, 0, Math.PI * 2); g.strokeStyle = C.c; g.lineWidth = 2; g.stroke(); }
        diamond(g, xe, ty, 7 + o.pulse * 1.2, o.col, C.ink);
        const gl = o.gateLabel, glw = tw(g, gl, 11, 'mono');
        const sMin = o.fmtTick(o.min), sMax = o.fmtTick(o.max);
        const wMin = tw(g, sMin, 11, 'mono'), wMax = tw(g, sMax, 11, 'mono');
        const gxl = V.clamp(gx, x + glw / 2, x + w - glw / 2);
        if (gxl - glw / 2 > x + wMin + 6) V.text(g, sMin, x, ty + 22, { size: 11, color: C.muted, align: 'left', font: 'mono' });
        if (gxl + glw / 2 < x + w - wMax - 6) V.text(g, sMax, x + w, ty + 22, { size: 11, color: C.muted, align: 'right', font: 'mono' });
        V.text(g, gl, gxl, ty + 22, { size: 11, color: C.ink, align: 'center', font: 'mono', weight: 600 });
      }
      function logBars(g, x, y, w, rows, C) {
        const L = (v) => (Math.log10(V.clamp(v, 0.001, 10)) + 3) / 4;
        rows.forEach((r, i) => {
          const yy = y + i * 30;
          V.text(g, r.label, x, yy + 10, { size: 11, color: C.muted });
          V.text(g, fmt(r.v, r.v < 0.1 ? 3 : 2), x + w, yy + 10, { size: 12, font: 'mono', weight: 700, color: C.ink, align: 'right' });
          const by = yy + 16;
          g.fillStyle = V.alpha(C.muted, 0.14); V.rr(g, x, by, w, 9, 4); g.fill();
          if (r.warnFrom != null) { const xw = x + L(r.warnFrom) * w; g.fillStyle = V.alpha(C.d, 0.18); V.rr(g, xw, by, x + w - xw, 9, 3); g.fill(); }
          g.fillStyle = r.col; V.rr(g, x, by, Math.max(4, L(r.v) * w), 9, 4); g.fill();
          if (r.ref != null) { const xr = x + L(r.ref) * w; V.line(g, xr, by - 4, xr, by + 13, { color: C.ink, lw: 1.4, dash: [2, 2] }); }
        });
        const ty = y + rows.length * 30 + 12;
        const tk = w < 250 ? [0.001, 0.1, 10] : [0.001, 0.01, 0.1, 1, 10];
        tk.forEach((v, k) => { const fr = (Math.log10(v) + 3) / 4; V.text(g, String(v), x + fr * w, ty, { size: 11, color: C.muted, align: fr === 0 ? 'left' : fr === 1 ? 'right' : 'center', font: 'mono' }); });
      }

      H = V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const c = CANDS[S.k];
        KEYS.forEach((k) => { A[k] += (c[k] - A[k]) * 0.12; });
        const el = t - S.changed;
        const narrow = w < 560;
        const pad = 12, gap = 10;
        const v = verdict(c);
        const anyBad = Object.values(v).some((x) => x === 'bad');
        const allOk = Object.values(v).every((x) => x === 'ok');
        face(g, pad + 17, 28 + Math.sin(t * 2.2) * 1.5, 28, C.b, { mood: anyBad ? 'sad' : allOk ? 'happy' : 'o' });
        V.text(g, c.label, pad + 40, 23, { size: 15, weight: 700, color: C.ink });
        V.wrap(g, c.desc, pad + 40, 42, w - pad * 2 - 40 - (narrow ? 0 : 170), { size: 12, color: C.muted, maxLines: 2 });
        const sc = easeB((el - 0.55) / 0.35);
        if (!narrow) stamp(g, w - pad - 80, 32, anyBad ? 'FAILED' : allOk ? 'PASSED' : 'INVESTIGATE', anyBad ? RED() : allOk ? C.a : C.d, { scale: sc, size: 14 });
        const top = narrow ? 82 : 72;
        const cols = narrow ? 1 : 4, rows = narrow ? 4 : 1;
        const pw = (w - pad * 2 - gap * (cols - 1)) / cols;
        const avail = h - top - pad - gap * (rows - 1);
        const WT = [1, 1, 1.6, 1.35], WS = WT.reduce((a, b) => a + b, 0);
        const phOf = (i) => (narrow ? (avail * WT[i]) / WS : avail);
        const pyOf = (i) => { let y = top; if (narrow) for (let k = 0; k < i; k++) y += phOf(k) + gap; return y; };
        const pulse = 0.5 + 0.5 * Math.sin(t * 3);
        const P = [
          { key: 'alvo', n: '1 · Target', q: 'did it improve on what we trained?', col: C.a },
          { key: 'reg', n: '2 · Regression', q: 'did I lose what it already knew?', col: C.b },
          { key: 'seg', n: '3 · Safety', q: 'refuses too much? complies when it shouldn\'t?', col: C.c },
          { key: 'der', n: '4 · Drift', q: 'how much did it change, and where?', col: C.d },
        ];
        P.forEach((p, i) => {
          const px = pad + (narrow ? 0 : i * (pw + gap)), py = pyOf(i), ph = phOf(i);
          V.box(g, px, py, pw, ph, { r: 12, fill: V.alpha(p.col, 0.06), stroke: V.alpha(p.col, 0.4), lw: 1.2 });
          V.text(g, p.n, px + 12, py + 20, { size: 13, weight: 700, color: p.col });
          const st = v[p.key];
          const stTxt = st === 'ok' ? 'PASS' : st === 'bad' ? 'FAIL' : 'INVESTIGATE';
          const stCol = st === 'ok' ? C.a : st === 'bad' ? RED() : C.d;
          const ix = px + 12, iw = pw - 24;
          let gy;
          if (narrow) {
            const room = iw - tw(g, p.n, 13, 'sans', 700) - 96;
            if (tw(g, p.q, 11) <= room) V.text(g, p.q, ix + tw(g, p.n, 13, 'sans', 700) + 8, py + 20, { size: 11, color: C.muted });
            stamp(g, px + pw - 50, py + 17, stTxt, stCol, { scale: easeB((el - 0.35 - i * 0.08) / 0.3) * 0.8, size: 12, rot: -0.05 });
            gy = py + 30;
          } else {
            V.wrap(g, p.q, ix, py + 38, iw, { size: 11.5, color: C.muted, maxLines: 2 });
            gy = py + 66;
            stamp(g, px + pw / 2, py + ph - 26, stTxt, stCol, { scale: easeB((el - 0.35 - i * 0.08) / 0.3), size: 13 });
          }
          const twoStep = narrow ? 56 : 64;
          if (p.key === 'alvo') {
            gauge(g, ix, gy, iw, { label: 'WR-LC vs base', min: 35, max: 80, gate: 60, gate2: 50, passRight: true, est: A.wr, lo: A.lo, hi: A.hi, ghost: A.raw, col: p.col, pulse, fmtv: (x) => fmt(x, 1) + '%', fmtTick: (x) => x + '%', gateLabel: '≥ 60' });
            if (!narrow) {
              const yy = gy + 66;
              g.beginPath(); g.arc(ix + 6, yy - 4, 5, 0, Math.PI * 2); g.strokeStyle = C.c; g.lineWidth = 2; g.stroke();
              V.text(g, 'raw: ' + fmt(A.raw, 1) + '%', ix + 16, yy, { size: 11.5, color: C.ink, font: 'mono' });
              V.wrap(g, 'gate: CI lower bound > 50% and WR-LC ≥ 60%', ix, yy + 20, iw, { size: 11, color: C.muted, maxLines: 2 });
            }
          } else if (p.key === 'reg') {
            gauge(g, ix, gy, iw, { label: 'Δ IFEval (pp)', min: -12, max: 4, gate: -2, zero: 0, passRight: true, est: A.dIF, lo: A.dIFlo, hi: A.dIFhi, col: p.col, pulse, fmtv: (x) => sgn(x, 1), fmtTick: (x) => sgn(x, 0), gateLabel: '−2' });
            if (!narrow) V.wrap(g, 'gate: Δ ≥ −2 pp (IFEval, GSM8K)', ix, gy + 66, iw, { size: 11, color: C.muted, maxLines: 2 });
          } else if (p.key === 'seg') {
            gauge(g, ix, gy, iw, { label: 'Δ over-refusal (pp)', min: -4, max: 6, gate: 2, zero: 0, passRight: false, est: A.ref, lo: A.refLo, hi: A.refHi, col: p.col, pulse, fmtv: (x) => sgn(x, 1), fmtTick: (x) => sgn(x, 0), gateLabel: '+2' });
            gauge(g, ix, gy + twoStep, iw, { label: 'Δ undue compliance', min: -0.1, max: 0.3, gate: 0, passRight: false, est: A.sr, lo: A.srLo, hi: A.srHi, col: p.col, pulse, fmtv: (x) => sgn(x, 2), fmtTick: (x) => sgn(x, 1), gateLabel: '0' });
          } else {
            logBars(g, ix, gy - 2, iw, [
              { label: 'domain KL', v: A.klD, col: C.d },
              { label: 'generic KL', v: A.klG, col: A.klG > 3 * KL_PREV ? C.d : C.a, ref: KL_PREV, warnFrom: 3 * KL_PREV },
            ], C);
            if (!narrow) V.wrap(g, 'dashed: generic KL of the previous SFT; amber band: ≫ (investigate)', ix, gy + 92, iw, { size: 11, color: C.muted, maxLines: 3 });
          }
        });
      });
    })();

    /* ================================================ FIG 2 — live bootstrap */
    (function () {
      const fig = root.querySelector('#fig-bootstrap');
      if (!fig) return;
      adaptH(fig, 420, 620);
      const NS = [50, 100, 200, 400, 800, 1000];
      const B = 2000;
      const S = { ni: 1, wr: 0.56, tie: 0.1, seed: 1 };
      let Y = null, reps = [], sorted = null, picks = null, lastPick = -9, repR = null, obs = 0, tStart = 0, H = null;
      const ctl = UI.controls(fig);
      UI.slider(ctl, { label: 'prompts n', min: 0, max: NS.length - 1, step: 1, value: S.ni, fmt: (v) => String(NS[v]), onInput: (v) => { S.ni = v; gen(); } });
      UI.slider(ctl, { label: 'true WR', min: 40, max: 70, step: 1, value: 56, fmt: (v) => v + '%', onInput: (v) => { S.wr = v / 100; gen(); } });
      UI.slider(ctl, { label: 'ties', min: 0, max: 30, step: 2, value: 10, fmt: (v) => v + '%', onInput: (v) => { S.tie = v / 100; gen(); } });
      UI.button(ctl, '↻ new experiment', () => { S.seed++; gen(); }, { primary: true });
      const rObs = UI.readout(ctl, 'observed WR'), rCI = UI.readout(ctl, 'bootstrap 95% CI'), rNorm = UI.readout(ctl, '± normal approx.'), rD = UI.readout(ctl, 'differs from 50%?');
      UI.legend(ctl, [[V.C().a, 'win'], [V.C().d, 'tie'], [V.C().c, 'loss']]);
      function gen(prefill) {
        const n = NS[S.ni];
        const r = V.rng(1000 + S.seed * 97 + n);
        const tie = Math.min(S.tie, 2 * Math.min(S.wr, 1 - S.wr));
        const pW = S.wr - tie / 2;
        Y = new Float32Array(n); let s = 0;
        for (let i = 0; i < n; i++) { const u = r(); Y[i] = u < pW ? 1 : u < pW + tie ? 0.5 : 0; s += Y[i]; }
        obs = s / n; reps = []; sorted = null; picks = new Uint8Array(n); repR = V.rng(77 + S.seed * 13 + n); lastPick = -9;
        tStart = H ? H.t : 0;
        if (prefill) for (let k = 0; k < B; k++) reps.push(oneRep(k === B - 1));
        upd();
        if (H) H.redraw();
      }
      function oneRep(rec) {
        const n = Y.length; let s = 0;
        if (rec) picks.fill(0);
        for (let i = 0; i < n; i++) { const j = (repR() * n) | 0; s += Y[j]; if (rec && picks[j] < 250) picks[j]++; }
        return s / n;
      }
      function q(p) {
        if (!reps.length) return obs;
        if (!sorted || sorted.length !== reps.length) sorted = Float64Array.from(reps).sort();
        return sorted[Math.min(sorted.length - 1, Math.max(0, Math.round(p * (sorted.length - 1))))];
      }
      const seN = () => Math.sqrt(Math.max(1e-9, obs * (1 - obs)) / Y.length);
      function upd() {
        rObs.set(fmt(obs * 100, 1) + '%');
        rNorm.set('±' + fmt(196 * seN(), 1) + ' pp');
        if (reps.length < 200) { rCI.set('resampling… ' + reps.length); rD.set('—'); return; }
        const a = q(0.025), b = q(0.975);
        rCI.set('[' + fmt(a * 100, 1) + ', ' + fmt(b * 100, 1) + ']' + (reps.length < B ? '…' : ''));
        if (a > 0.5) rD.set('yes: better', 'ok'); else if (b < 0.5) rD.set('yes: worse', 'bad'); else rD.set('no (crosses 50%)', 'warn');
      }
      gen(true);

      H = V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const n = Y.length;
        const el = t - tStart;
        if (reps.length < B) {
          let add = reps.length < 24 ? (el > reps.length * 0.16 ? 1 : 0) : Math.min(B - reps.length, 3 + (reps.length >> 6));
          for (let k = 0; k < add; k++) { const rec = k === add - 1 && t - lastPick > 0.15; reps.push(oneRep(rec)); if (rec) lastPick = t; }
          if (add) { sorted = null; if (reps.length < 220 || reps.length % 40 === 0 || reps.length === B) upd(); }
        } else if (t - lastPick > 0.45) { oneRep(true); lastPick = t; }
        const narrow = w < 600;
        const pad = 14;
        const G = narrow ? { x: pad, y: 30, w: w - 2 * pad, h: h * 0.4 - 30 } : { x: pad, y: 34, w: w * 0.42 - pad, h: h - 34 - pad };
        const Hs = narrow ? { x: pad + 6, y: h * 0.47 + 18, w: w - 2 * pad - 12, h: h * 0.53 - 60 } : { x: w * 0.48, y: 52, w: w * 0.52 - pad - 6, h: h - 52 - 44 };
        // ---- prompt grid
        V.text(g, n + ' prompts · replicate #' + Math.min(reps.length, B), G.x, G.y - 12, { size: 12, color: C.muted, font: 'mono' });
        const cols = Math.max(1, Math.ceil(Math.sqrt(n * (G.w / G.h)))), rws = Math.ceil(n / cols);
        const cell = Math.min(G.w / cols, G.h / rws);
        const gp = cell > 7 ? 1.5 : 0.8, cs = cell - gp;
        const colOf = (y) => (y === 1 ? C.a : y === 0.5 ? C.d : C.c);
        for (let i = 0; i < n; i++) {
          const cx = G.x + (i % cols) * cell, cy = G.y + Math.floor(i / cols) * cell;
          const k = picks[i];
          g.fillStyle = V.alpha(colOf(Y[i]), k === 0 ? 0.16 : 0.9);
          if (cs >= 6) { V.rr(g, cx, cy, cs, cs, Math.min(3, cs / 4)); g.fill(); } else g.fillRect(cx, cy, cs, cs);
          if (k >= 2) V.dot(g, cx + cs / 2, cy + cs / 2, Math.max(1.2, cs * 0.18), C.dark ? '#06140d' : '#ffffff');
          if (k >= 2 && cs >= 16) V.text(g, '×' + k, cx + cs / 2, cy + cs - 3, { size: 11, font: 'mono', color: C.dark ? '#06140d' : '#ffffff', align: 'center' });
        }
        // ---- histogram
        const se = seN();
        let x0 = Math.min(obs - 4.6 * se, 0.48), x1 = Math.max(obs + 4.6 * se, 0.52);
        x0 = Math.max(0, x0); x1 = Math.min(1, x1);
        const X = (v) => Hs.x + ((v - x0) / (x1 - x0)) * Hs.w;
        const NB = 46, bins = new Array(NB).fill(0);
        reps.forEach((r) => { const b = Math.floor(((r - x0) / (x1 - x0)) * NB); if (b >= 0 && b < NB) bins[b]++; });
        const mx = Math.max(4, ...bins);
        const base = Hs.y + Hs.h;
        V.text(g, 'distribution of WR across replicates', Hs.x, Hs.y - 26, { size: 12, color: C.muted, font: 'mono' });
        if (reps.length >= 40) {
          const a = q(0.025), b = q(0.975);
          g.fillStyle = V.alpha(C.d, 0.14); g.fillRect(X(a), Hs.y, X(b) - X(a), Hs.h);
          V.line(g, X(a), Hs.y, X(a), base, { color: C.d, lw: 1.6 });
          V.line(g, X(b), Hs.y, X(b), base, { color: C.d, lw: 1.6 });
          V.text(g, '2.5%', X(a) - 4, Hs.y + 12, { size: 11, color: C.d, align: 'right', font: 'mono' });
          V.text(g, '97.5%', X(b) + 4, Hs.y + 12, { size: 11, color: C.d, align: 'left', font: 'mono' });
        }
        const bw = Hs.w / NB;
        bins.forEach((c, i) => { if (!c) return; const bh = (c / mx) * (Hs.h - 34); g.fillStyle = V.alpha(C.a, 0.75); V.rr(g, Hs.x + i * bw + 0.5, base - bh, bw - 1, bh, 2); g.fill(); });
        V.line(g, Hs.x, base, Hs.x + Hs.w, base, { color: V.alpha(C.muted, 0.6), lw: 1 });
        // 50%
        V.line(g, X(0.5), Hs.y - 6, X(0.5), base + 4, { color: C.c, lw: 1.8, dash: [5, 4] });
        V.text(g, '50%', X(0.5), Hs.y - 10, { size: 11, color: C.c, align: 'center', font: 'mono', weight: 600 });
        // normal approximation
        const nl = X(obs - 1.96 * se), nr = X(obs + 1.96 * se), ny = Hs.y + 26;
        V.line(g, nl, ny, nr, ny, { color: C.b, lw: 2 });
        V.line(g, nl, ny - 5, nl, ny + 5, { color: C.b, lw: 2 }); V.line(g, nr, ny - 5, nr, ny + 5, { color: C.b, lw: 2 });
        if (!narrow || Hs.w > 260) V.text(g, 'normal', nr + 5, ny + 4, { size: 11, color: C.b, font: 'mono' });
        // observed
        diamond(g, X(obs), base + 10, 6, C.ink);
        // axis
        const stp = niceStep((x1 - x0) * 100, 5) / 100;
        for (let v = Math.ceil(x0 / stp) * stp; v <= x1 + 1e-9; v += stp) V.text(g, Math.round(v * 100) + '%', X(v), base + 30, { size: 11, color: C.muted, align: 'center', font: 'mono' });
      });
    })();

    /* ================================================ FIG 3 — the courtroom of judges */
    (function () {
      const fig = root.querySelector('#fig-juiz');
      if (!fig) return;
      adaptH(fig, 400, 556);
      const S = { mode: '1o', wr: 0.5, bPos: 1.0, len: 0, teacher: false };
      const B_LEN = 1.0, B_SELF = 0.9, MAXP = 400;
      const judges = () => (S.mode === 'painel' ? ['glm', 'qwen', 'ds'] : [S.teacher ? 'glm' : 'ds']);
      const shift = (j) => logit(S.wr) + B_LEN * Math.tanh(S.len / 100) + (S.teacher && j === 'glm' ? B_SELF : 0);
      const pFirst = (j) => sig(shift(j) + S.bPos);
      const pSecond = (j) => sig(shift(j) - S.bPos);
      function expected() {
        const js = judges();
        if (S.mode === '1o') return pFirst(js[0]);
        const pr = js.map((j) => { const a = pFirst(j), b = pSecond(j); return { w: a * b, l: (1 - a) * (1 - b) }; });
        if (pr.length === 1) return pr[0].w + 0.5 * (1 - pr[0].w - pr[0].l);
        let E = 0; const o3 = [1, 0, -1];
        for (const x of o3) for (const y of o3) for (const z of o3) {
          const P = [x, y, z].reduce((acc, vv, k) => acc * (vv === 1 ? pr[k].w : vv === -1 ? pr[k].l : 1 - pr[k].w - pr[k].l), 1);
          const s = x + y + z; E += P * (s > 0 ? 1 : s < 0 ? 0 : 0.5);
        }
        return E;
      }
      let cnt = null, rng = null, cur = null, cyc0 = 0, H = null;
      function runPair() {
        const res = judges().map((j) => {
          const v1 = rng() < pFirst(j) ? 'c' : 'b';
          const v2 = S.mode === '1o' ? null : rng() < pSecond(j) ? 'c' : 'b';
          const y = S.mode === '1o' ? (v1 === 'c' ? 1 : 0) : v1 === 'c' && v2 === 'c' ? 1 : v1 === 'b' && v2 === 'b' ? 0 : 0.5;
          return { j, v1, v2, y };
        });
        let y;
        if (res.length === 1) y = res[0].y;
        else { const s = res.reduce((a, r) => a + (r.y === 1 ? 1 : r.y === 0 ? -1 : 0), 0); y = s > 0 ? 1 : s < 0 ? 0 : 0.5; }
        return { res, y };
      }
      function tally(p) { if (cnt.n >= MAXP) return; cnt.n++; if (p.y === 1) cnt.w++; else if (p.y === 0) cnt.l++; else cnt.t++; }
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Protocol', options: [{ v: '1o', label: '1 judge · 1 order' }, { v: '2o', label: '1 judge · 2 orders' }, { v: 'painel', label: 'panel of 3 · 2 orders' }], value: S.mode, onChange: (v) => { S.mode = v; reset(); } });
      UI.slider(ctl, { label: 'true WR', min: 30, max: 70, step: 1, value: 50, fmt: (v) => v + '%', onInput: (v) => { S.wr = v / 100; reset(); } });
      UI.slider(ctl, { label: 'position bias', min: 0, max: 2, step: 0.1, value: 1, fmt: (v) => fmt(v, 1), onInput: (v) => { S.bPos = v; reset(); } });
      UI.slider(ctl, { label: 'extra length', min: 0, max: 200, step: 10, value: 0, fmt: (v) => '+' + v + '%', onInput: (v) => { S.len = v; reset(); } });
      UI.toggle(ctl, { label: 'judge from the teacher\'s family (GLM)', value: false, onChange: (v) => { S.teacher = v; reset(); } });
      const rT = UI.readout(ctl, 'true WR'), rE = UI.readout(ctl, 'expected WR w/ biases'), rM = UI.readout(ctl, 'measured WR');
      function upd() {
        rT.set(fmt(S.wr * 100, 1) + '%');
        const e = expected();
        rE.set(fmt(e * 100, 1) + '%', Math.abs(e - S.wr) < 0.02 ? 'ok' : Math.abs(e - S.wr) < 0.06 ? 'warn' : 'bad');
        if (cnt && cnt.n) rM.set(fmt(((cnt.w + 0.5 * cnt.t) / cnt.n) * 100, 1) + '% (' + cnt.n + ')'); else rM.set('—');
      }
      function reset() {
        cnt = { n: 0, w: 0, t: 0, l: 0 };
        rng = V.rng(31 + Math.round(S.wr * 100) * 7 + Math.round(S.bPos * 10) * 13 + S.len * 3 + (S.teacher ? 5 : 0) + (S.mode === 'painel' ? 101 : S.mode === '2o' ? 53 : 0));
        for (let k = 0; k < 24; k++) tally(runPair());
        cur = runPair(); cyc0 = H ? H.t : 0; upd();
      }
      reset();
      const JN = { glm: 'GLM', qwen: 'Qwen', ds: 'DeepSeek' };

      function card(g, x, y, cw, isCand, alpha) {
        const C = V.C();
        const lines = isCand ? Math.min(9, Math.round(3 * (1 + S.len / 100))) : 3;
        const ch = 30 + lines * 7;
        g.globalAlpha = alpha;
        const col = isCand ? C.b : C.muted;
        V.box(g, x - cw / 2, y, cw, ch, { r: 9, fill: V.alpha(col, 0.12), stroke: V.alpha(col, 0.7), lw: 1.4 });
        V.text(g, isCand ? 'candidate' : 'base', x, y + 15, { size: 11.5, weight: 700, color: isCand ? C.b : C.ink, align: 'center', font: 'mono' });
        for (let k = 0; k < lines; k++) { g.fillStyle = V.alpha(C.muted, 0.35); V.rr(g, x - cw / 2 + 10, y + 25 + k * 7, (cw - 20) * (k === lines - 1 ? 0.6 : 0.92 - (k % 3) * 0.08), 3.5, 2); g.fill(); }
        g.globalAlpha = 1;
        return ch;
      }

      H = V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const T = S.mode === '1o' ? 1.5 : 2.4;
        let f = (t - cyc0) / T;
        if (f >= 1) {
          tally(cur);
          for (let k = 0; k < 11; k++) tally(runPair());
          upd();
          cur = runPair(); cyc0 = t; f = 0;
        }
        const narrow = w < 600;
        const pad = 12;
        // on phones, the courtroom has a fixed height (cards + judges + stamp) and the scoreboard sits right below it
        const Sc = narrow ? { x: pad, y: 8, w: w - 2 * pad, h: 280 } : { x: pad, y: 8, w: w * 0.58, h: h - 16 };
        const Bd = narrow ? { x: pad, y: 294, w: w - 2 * pad, h: h - 304 } : { x: w * 0.62, y: 18, w: w * 0.38 - pad - 4, h: h - 36 };
        const two = S.mode !== '1o';
        // phases
        const pIn = V.clamp(f / 0.12, 0, 1);
        const v1At = two ? 0.14 : 0.16, swapA = 0.33, swapB = 0.52, v2At = 0.55, finAt = two ? 0.74 : 0.52;
        const swapT = two ? V.ease.inOut(V.clamp((f - swapA) / (swapB - swapA), 0, 1)) : 0;
        // cards
        const xA = Sc.x + Sc.w * 0.27, xB = Sc.x + Sc.w * 0.73;
        const cw = Math.min(150, Sc.w * 0.4);
        const cy0 = Sc.y + 30;
        V.text(g, 'position A', xA, Sc.y + 18, { size: 11.5, color: C.muted, align: 'center', font: 'mono' });
        V.text(g, 'position B', xB, Sc.y + 18, { size: 11.5, color: C.muted, align: 'center', font: 'mono' });
        const candX = V.lerp(xA, xB, swapT), baseX = V.lerp(xB, xA, swapT);
        const arc = Math.sin(Math.PI * swapT) * 18;
        const chB = card(g, baseX, cy0 + arc, cw, false, pIn);
        const chC = card(g, candX, cy0 - arc, cw, true, pIn);
        const cardBottom = cy0 + Math.max(chB, chC, 30 + 9 * 7) + 6;
        // judges
        const js = judges();
        const jy = V.clamp(Sc.y + Sc.h * 0.64, cardBottom + 64, Sc.y + Sc.h - 74);
        const js_x = js.length === 1 ? [Sc.x + Sc.w / 2] : [Sc.x + Sc.w * 0.2, Sc.x + Sc.w * 0.5, Sc.x + Sc.w * 0.8];
        const jcol = { glm: C.glm, qwen: C.qwen, ds: C.ds };
        const reading = (f > v1At - 0.06 && f < v1At + 0.1) || (two && f > v2At - 0.06 && f < v2At + 0.1);
        if (reading) js_x.forEach((x) => { [xA, xB].forEach((cx) => { V.line(g, cx, cy0 + 60, x, jy - 58, { color: V.alpha(C.d, 0.35), lw: 1.2, dash: [3, 4], dashOffset: -t * 30 }); }); });
        cur.res.forEach((r, k) => {
          const x = js_x[k];
          const biased = S.teacher && r.j === 'glm';
          const hit1 = f > v1At && f < v1At + 0.06, hit2 = two && f > v2At && f < v2At + 0.06;
          face(g, x, jy + (hit1 || hit2 ? 2 : 0), 32, jcol[r.j], { mood: biased ? 'o' : 'happy', wig: true });
          // gavel
          const ang = hit1 || hit2 ? 0.9 : 0.2;
          g.save(); g.translate(x + 22, jy + 4); g.rotate(-ang);
          g.fillStyle = C.d; g.fillRect(-2, -14, 4, 16); g.fillRect(-7, -18, 14, 7); g.restore();
          V.text(g, JN[r.j] + (biased ? ' (teacher)' : ''), x, jy + 32, { size: 11, color: C.ink, align: 'center', font: 'mono', weight: 600 });
          // speech bubbles
          const bub = (txt, isC, bx, by, s) => { if (s <= 0) return; g.save(); g.translate(bx, by); g.scale(s, s); chip(g, -19, -11, 38, 22, txt, { fill: isC ? V.alpha(C.b, 0.3) : V.alpha(C.muted, 0.18), stroke: isC ? C.b : C.muted, weight: 700, size: 12 }); g.restore(); };
          const s1 = easeB((f - v1At) / 0.08);
          const pos1 = r.v1 === 'c' ? 'A' : 'B';
          bub('1st ' + pos1, r.v1 === 'c', x - (two ? 19 : 0), jy - 44, s1 * (two ? 0.85 : 1));
          if (two) { const pos2 = r.v2 === 'c' ? 'B' : 'A'; bub('2nd ' + pos2, r.v2 === 'c', x + 19, jy - 44, easeB((f - v2At) / 0.08) * 0.85); }
          if (two && f > finAt && r.y === 0.5 && js.length > 1) V.text(g, '≠', x, jy - 60, { size: 13, color: C.d, align: 'center', weight: 700 });
        });
        // final verdict
        const sfin = easeB((f - finAt) / 0.12);
        const vt = cur.y === 1 ? 'candidate wins' : cur.y === 0 ? 'base wins' : 'tie';
        const vc = cur.y === 1 ? C.b : cur.y === 0 ? C.muted : C.d;
        stamp(g, Sc.x + Sc.w / 2, Math.min(Sc.y + Sc.h - 16, jy + 62), vt, vc, { scale: sfin, size: 13 });
        // ---- scoreboard
        const C2 = C;
        V.box(g, Bd.x, Bd.y, Bd.w, Bd.h, { r: 12, fill: V.alpha(C2.muted, 0.06), stroke: V.alpha(C2.muted, 0.3), lw: 1 });
        const ix = Bd.x + 14, iw = Bd.w - 28;
        V.text(g, 'running tally', ix, Bd.y + 22, { size: 12.5, weight: 700, color: C2.ink });
        V.text(g, 'pairs: ' + cnt.n + '/' + MAXP, ix + iw, Bd.y + 22, { size: 11.5, color: C2.muted, align: 'right', font: 'mono' });
        const n = Math.max(1, cnt.n);
        const wr = cnt.n ? (cnt.w + 0.5 * cnt.t) / n : 0.5;
        const wrTxt = cnt.n ? fmt(wr * 100, 1) + '%' : '—';
        V.text(g, wrTxt, ix, Bd.y + 62, { size: 30, weight: 700, color: C2.b, font: 'mono' });
        V.text(g, 'measured WR', ix + tw(g, wrTxt, 30, 'mono', 700) + 10, Bd.y + 60, { size: 12, color: C2.muted });
        // stacked bar
        const by = Bd.y + 82;
        g.fillStyle = V.alpha(C2.muted, 0.12); g.fillRect(ix, by, iw, 16);
        const seg = [[cnt.w, C2.b, 'W'], [cnt.t, C2.d, 'T'], [cnt.l, V.alpha(C2.muted, 0.55), 'L']];
        let xx = ix;
        if (cnt.n) seg.forEach(([c, col, lb]) => { const ww = (c / n) * iw; g.fillStyle = col; g.fillRect(xx, by, Math.max(0, ww - 1), 16); if (ww > 34) V.text(g, lb + ' ' + Math.round((c / n) * 100) + '%', xx + ww / 2, by + 12, { size: 11, font: 'mono', weight: 600, align: 'center', color: C2.dark ? '#06140d' : '#ffffff' }); xx += ww; });
        // 0–100 axis
        const ay = by + 52;
        const AX = (v) => ix + v * iw;
        V.line(g, ix, ay, ix + iw, ay, { color: V.alpha(C2.muted, 0.6), lw: 1 });
        [0, 0.25, 0.5, 0.75, 1].forEach((v) => { V.line(g, AX(v), ay - 3, AX(v), ay + 3, { color: C2.muted, lw: 1 }); V.text(g, Math.round(v * 100) + '%', AX(v), ay + 17, { size: 11, color: C2.muted, align: 'center', font: 'mono' }); });
        V.line(g, AX(S.wr), ay - 22, AX(S.wr), ay + 6, { color: C2.a, lw: 2, dash: [4, 3] });
        const e = expected();
        diamond(g, AX(e), ay - 10, 7, null, C2.ink);
        if (cnt.n) diamond(g, AX(wr), ay - 10, 6, C2.b, null);
        const ly = ay + 38;
        if (Bd.h > 190) {
          V.text(g, '- - true: ' + fmt(S.wr * 100, 1) + '%', ix, ly, { size: 11.5, color: C2.a, font: 'mono' });
          V.text(g, '◇ expected w/ biases: ' + fmt(e * 100, 1) + '%', ix, ly + 18, { size: 11.5, color: C2.ink, font: 'mono' });
          if (Bd.h > 236) {
            const msg = S.mode === '1o' ? 'candidate always in position A: position bias turns into a “win”' : S.mode === '2o' ? 'inconsistency between orders counts as a tie: position bias cancels out' : 'majority vote of 3 families: dilutes one judge\'s self-preference without eliminating it';
            V.wrap(g, msg, ix, ly + 44, iw, { size: 11.5, color: C2.muted, maxLines: 3 });
          }
        }
      });
    })();

    /* ================================================ FIG 4 — online Elo × Bradley–Terry */
    (function () {
      const fig = root.querySelector('#fig-bt');
      if (!fig) return;
      adaptH(fig, 450, 640);
      const SHORT = ['base', 'SFT', 'DPO.1', 'DPO.3', 'NVFP4'];
      const BETA = [0, 0.4, 0.75, 0.62, 0.7];
      const M = 5, NB = 800, SC = 400 / Math.LN10, LO = 880, HI = 1220;
      const r0 = V.rng(2024);
      const PAIRS = []; for (let i = 0; i < M; i++) for (let j = i + 1; j < M; j++) PAIRS.push([i, j]);
      const BAT = Array.from({ length: NB }, () => { const pr = PAIRS[r0.int(PAIRS.length)]; const sw = r0() < 0.5; const a = sw ? pr[1] : pr[0], b = sw ? pr[0] : pr[1]; return { a, b, y: r0() < sig(BETA[a] - BETA[b]) ? 1 : 0 }; });
      const S = { K: 16, order: BAT.map((_, i) => i), ord: 1, played: 0, acc: 0, last: null, lastFit: 0 };
      let elo, traj, Wm, Nm, bt = new Array(M).fill(0), ghosts = [];
      const shufR = V.rng(9);
      const zeros = () => Array.from({ length: M }, () => new Array(M).fill(0));
      function resetPlay() { elo = new Array(M).fill(1000); traj = Array.from({ length: M }, () => [ANCH]); Wm = zeros(); Nm = zeros(); bt = new Array(M).fill(0); S.played = 0; S.acc = 0; S.last = null; S.lastFit = 0; }
      // online Elo (zero-sum, mean 1000) shifted to have the same mean as the final BT (which does not depend on the order)
      let ANCH = 1000;
      const rel = (i) => elo[i] - 1000 + ANCH;
      function playOne() {
        const bb = BAT[S.order[S.played]];
        const E = 1 / (1 + Math.pow(10, (elo[bb.b] - elo[bb.a]) / 400));
        const d = S.K * (bb.y - E); elo[bb.a] += d; elo[bb.b] -= d;
        Wm[bb.a][bb.b] += bb.y; Wm[bb.b][bb.a] += 1 - bb.y; Nm[bb.a][bb.b]++; Nm[bb.b][bb.a]++;
        S.played++; S.last = bb;
        if (S.played % 8 === 0) for (let i = 0; i < M; i++) traj[i].push(rel(i));
      }
      function fitBT() {
        let p = new Array(M).fill(1);
        for (let it = 0; it < 80; it++) {
          const np = p.map((pi, i) => { let wi = 0, den = 0; for (let j = 0; j < M; j++) { if (j === i) continue; wi += Wm[i][j] + 0.5; den += (Nm[i][j] + 1) / (pi + p[j]); } return wi / den; });
          p = np.map((x) => x / np[0]);
        }
        bt = p.map((x) => Math.log(x));
      }
      resetPlay();
      BAT.forEach((bb) => { Wm[bb.a][bb.b] += bb.y; Wm[bb.b][bb.a] += 1 - bb.y; Nm[bb.a][bb.b]++; Nm[bb.b][bb.a]++; });
      fitBT(); ANCH = bt.reduce((a, b) => a + 1000 + SC * b, 0) / M;
      resetPlay();
      while (S.played < NB) playOne();
      fitBT(); S.lastFit = NB;
      let doneAt = 0, Hb = null;
      function shuffle() {
        if (S.played > 40) { ghosts.push(Array.from({ length: M }, (_, i) => rel(i))); if (ghosts.length > 5) ghosts.shift(); }
        const o = S.order; for (let i = o.length - 1; i > 0; i--) { const j = Math.floor(shufR() * (i + 1)); const tmp = o[i]; o[i] = o[j]; o[j] = tmp; }
        S.ord++; resetPlay(); upd();
      }
      const ctl = UI.controls(fig);
      UI.button(ctl, '↻ shuffle the order', shuffle, { primary: true });
      UI.slider(ctl, { label: 'Elo K factor', min: 8, max: 64, step: 8, value: 16, fmt: (v) => String(v), onInput: (v) => { S.K = v; ghosts = []; resetPlay(); while (S.played < NB) playOne(); fitBT(); S.lastFit = NB; doneAt = Hb ? Hb.t : 0; upd(); } });
      const rBT = UI.readout(ctl, 'BT · SFT', '—'), rEl = UI.readout(ctl, 'online Elo · SFT', '—'), rSp = UI.readout(ctl, 'spread across orders', '—');
      function upd() {
        rBT.set(Math.round(1000 + SC * bt[1]) + '');
        rEl.set(Math.round(rel(1)) + '');
        if (ghosts.length) { const all = ghosts.map((gh) => gh[1]).concat(S.played === NB ? [rel(1)] : []); rSp.set(Math.round(Math.max(...all) - Math.min(...all)) + ' pts'); } else rSp.set('shuffle');
      }
      upd();

      Hb = V.fig(fig, (g, w, h, t, dt) => {
        const C = V.C();
        if (S.played < NB) {
          S.acc += dt * 70;
          while (S.acc >= 1 && S.played < NB) { playOne(); S.acc -= 1; }
          if (S.played - S.lastFit >= 20 || S.played === NB) { fitBT(); S.lastFit = S.played; upd(); }
          if (S.played === NB) doneAt = t;
        } else if (t - doneAt > 3) shuffle();
        const narrow = w < 640;
        const pad = 14;
        const cols = [C.muted, C.a, C.b, C.c, C.d];
        // header
        V.text(g, (narrow ? '' : 'battle ') + S.played + '/' + NB + ' · order ' + (narrow ? '' : '#') + S.ord, pad, 20, { size: 12, color: C.muted, font: 'mono' });
        if (S.last) {
          const a = S.last.a, b = S.last.b, wnr = S.last.y ? a : b;
          let x = w - pad;
          const lab = [SHORT[a], '×', SHORT[b]];
          const wds = lab.map((s) => tw(g, s, 11.5, 'mono', 600) + 14);
          x -= wds.reduce((s, v) => s + v + 4, 0);
          lab.forEach((s, k) => {
            if (k === 1) V.text(g, '×', x + wds[k] / 2, 20, { size: 12, color: C.muted, align: 'center' });
            else { const idx = k === 0 ? a : b; chip(g, x, 8, wds[k], 18, s, { fill: idx === wnr ? V.alpha(cols[idx], 0.45) : V.alpha(C.muted, 0.08), stroke: idx === wnr ? cols[idx] : V.alpha(C.muted, 0.4), weight: 600, size: 11.5 }); }
            x += wds[k] + 4;
          });
        }
        // win matrix
        const lab = 52;
        const msz = narrow ? Math.min(w - lab - pad * 2, h * 0.42) : Math.min(w * 0.38 - lab, h - 120);
        const mx0 = pad + lab, my0 = 62, cell = msz / M;
        V.text(g, 'win fraction (row over column)', pad, 46, { size: 11.5, color: C.muted, font: 'mono' });
        for (let i = 0; i < M; i++) {
          V.text(g, SHORT[i], mx0 - 6, my0 + i * cell + cell / 2 + 4, { size: 11, color: cols[i] === C.muted ? C.ink : cols[i], align: 'right', font: 'mono', weight: 600 });
          V.text(g, SHORT[i], mx0 + i * cell + cell / 2, my0 + msz + 15, { size: 11, color: cols[i] === C.muted ? C.ink : cols[i], align: 'center', font: 'mono', weight: 600, maxW: cell + 4 });
          for (let j = 0; j < M; j++) {
            const x = mx0 + j * cell + 1, y = my0 + i * cell + 1, cs = cell - 2;
            if (i === j) { g.fillStyle = V.alpha(C.muted, 0.08); V.rr(g, x, y, cs, cs, 4); g.fill(); continue; }
            const nn = Nm[i][j];
            const v = nn ? Wm[i][j] / nn : 0.5;
            const z = V.clamp((v - 0.5) * 2.4, -1, 1);
            g.fillStyle = nn ? V.cmap(z, 'div') : V.alpha(C.muted, 0.08); V.rr(g, x, y, cs, cs, 4); g.fill();
            if (nn && cs >= 22) {
              const light = Math.abs(z) > 0.6;
              V.text(g, Math.round(v * 100) + '', x + cs / 2, y + cs / 2 + 1, { size: Math.min(13, cs * 0.32), font: 'mono', weight: 600, align: 'center', base: 'middle', color: light ? (C.dark ? '#06140d' : '#ffffff') : C.ink });
              if (cs >= 40) V.text(g, 'n=' + nn, x + cs / 2, y + cs - 6, { size: 11, font: 'mono', align: 'center', color: light ? (C.dark ? '#06140d' : '#ffffff') : C.muted });
            }
          }
        }
        const R = (v) => V.clamp((v - LO) / (HI - LO), 0, 1);
        // trajectories (wide only)
        let Pm;
        if (!narrow) {
          const T = { x: w * 0.47, y: 44, w: w * 0.53 - pad - 6, h: h * 0.42 };
          V.text(g, 'online Elo over the battles', T.x, T.y - 4, { size: 11.5, color: C.muted, font: 'mono' });
          g.strokeStyle = V.alpha(C.muted, 0.25); g.lineWidth = 1;
          [950, 1000, 1050, 1100, 1150].forEach((v) => { const y = T.y + T.h - R(v) * T.h; g.beginPath(); g.moveTo(T.x, y); g.lineTo(T.x + T.w, y); g.stroke(); V.text(g, v + '', T.x - 4, y + 4, { size: 11, color: C.muted, align: 'right', font: 'mono' }); });
          for (let i = 0; i < M; i++) {
            g.beginPath();
            traj[i].forEach((v, k) => { const x = T.x + ((k * 8) / NB) * T.w, y = T.y + T.h - R(v) * T.h; if (k) g.lineTo(x, y); else g.moveTo(x, y); });
            g.strokeStyle = cols[i] === C.muted ? C.ink : cols[i]; g.lineWidth = 1.6; g.stroke();
          }
          Pm = { x: w * 0.47, y: T.y + T.h + 56, w: w * 0.53 - pad - 6, h: h - (T.y + T.h + 56) - 30 };
        } else {
          Pm = { x: pad + lab, y: my0 + msz + 62, w: w - pad * 2 - lab, h: h - (my0 + msz + 62) - 30 };
        }
        // score panel
        V.text(g, 'score (base = 1000)', narrow ? pad : Pm.x, Pm.y - 28, { size: 11.5, color: C.muted, font: 'mono' });
        const lgY = Pm.y - 10;
        let lx = narrow ? w - pad : Pm.x + Pm.w;
        [['| truth', V.alpha(C.ink, 0.6)], ['× prev. orders', C.muted], ['○ online Elo', C.ink], ['◆ BT', C.ink]].forEach(([s2, col]) => { const ww = tw(g, s2, 11, 'mono'); if (lx - ww < (narrow ? pad : Pm.x - 40)) return; V.text(g, s2, lx, lgY, { size: 11, color: col, align: 'right', font: 'mono' }); lx -= ww + 10; });
        const rh = Pm.h / M;
        const X = (v) => Pm.x + R(v) * Pm.w;
        [950, 1000, 1050, 1100, 1150].forEach((v) => { V.line(g, X(v), Pm.y, X(v), Pm.y + Pm.h, { color: V.alpha(C.muted, 0.2), lw: 1 }); V.text(g, v + '', X(v), Pm.y + Pm.h + 15, { size: 11, color: C.muted, align: 'center', font: 'mono' }); });
        for (let i = 0; i < M; i++) {
          const y = Pm.y + rh * i + rh / 2;
          const col = cols[i] === C.muted ? C.ink : cols[i];
          V.text(g, SHORT[i], Pm.x - 8, y + 4, { size: 11, color: col, align: 'right', font: 'mono', weight: 600 });
          V.line(g, Pm.x, y, Pm.x + Pm.w, y, { color: V.alpha(C.muted, 0.3), lw: 1 });
          const tv = 1000 + SC * BETA[i];
          V.line(g, X(tv), y - 9, X(tv), y + 9, { color: V.alpha(C.ink, 0.45), lw: 2 });
          ghosts.forEach((gh) => { const gx = X(gh[i]); g.strokeStyle = V.alpha(C.muted, 0.9); g.lineWidth = 1.5; g.beginPath(); g.moveTo(gx - 4, y - 4); g.lineTo(gx + 4, y + 4); g.moveTo(gx + 4, y - 4); g.lineTo(gx - 4, y + 4); g.stroke(); });
          if (i > 0 || true) { g.beginPath(); g.arc(X(rel(i)), y, 6, 0, Math.PI * 2); g.strokeStyle = col; g.lineWidth = 2; g.stroke(); }
          diamond(g, X(1000 + SC * bt[i]), y, 6.5, col, C.dark ? '#06140d' : '#ffffff');
        }
      });
    })();

    /* ================================================ FIG 5 — the variant lottery */
    (function () {
      const fig = root.querySelector('#fig-loteria');
      if (!fig) return;
      adaptH(fig, 400, 560);
      const VOTES = [250, 500, 1000, 2000, 5000];
      const SC = 400 / Math.LN10;
      const S = { N: 27, vi: 2, mode: 'max' };
      const sigma = () => (SC * 2) / Math.sqrt(VOTES[S.vi]);
      function eMax(N) {
        if (N <= 1) return 0;
        let s = 0; const dx = 0.005;
        for (let x = -8; x <= 8; x += dx) { const phi = Math.exp((-x * x) / 2) / Math.sqrt(2 * Math.PI); s += x * N * phi * Math.pow(Phi(x), N - 1) * dx; }
        return s;
      }
      let rng = null, cyc = null, pub = [], EM = eMax(S.N);
      const CYC = 3.4;
      const ctl = UI.controls(fig);
      UI.slider(ctl, { label: 'variants tested N', min: 1, max: 30, step: 1, value: S.N, fmt: (v) => String(v), onInput: (v) => { S.N = v; EM = eMax(v); reset(); } });
      UI.slider(ctl, { label: 'votes per variant', min: 0, max: VOTES.length - 1, step: 1, value: S.vi, fmt: (v) => VOTES[v].toLocaleString('en-US'), onInput: (v) => { S.vi = v; reset(); } });
      UI.seg(ctl, { label: 'Publish', options: [{ v: 'max', label: 'the champion' }, { v: 'hold', label: 'the re-measured champion' }], value: 'max', onChange: (v) => { S.mode = v; reset(); } });
      const rS = UI.readout(ctl, 'σ per variant'), rI = UI.readout(ctl, 'expected inflation'), rP = UI.readout(ctl, 'published mean');
      function upd() {
        rS.set(fmt(sigma(), 1) + ' pts');
        const inf = S.mode === 'max' ? sigma() * EM : 0;
        rI.set('+' + fmt(inf, 1) + ' pts', inf > 10 ? 'bad' : inf > 3 ? 'warn' : 'ok');
        if (pub.length) { const m = pub.reduce((a, b) => a + b, 0) / pub.length; rP.set(sgn(m, 1) + ' (' + pub.length + ')'); } else rP.set('—');
      }
      function reset() {
        pub = []; rng = V.rng(27 + S.N * 3 + S.vi * 11 + (S.mode === 'max' ? 0 : 500)); cyc = null;
        const s = sigma();
        for (let c = 0; c < 40; c++) { let mx = -Infinity; for (let i = 0; i < S.N; i++) mx = Math.max(mx, rng.normal() * s); pub.push(S.mode === 'max' ? mx : rng.normal() * s); }
        upd();
      }
      reset();
      function newCycle(t) {
        const s = sigma();
        const vals = Array.from({ length: S.N }, () => rng.normal() * s);
        let ch = 0; vals.forEach((v, i) => { if (v > vals[ch]) ch = i; });
        cyc = { vals, ch, re: rng.normal() * s, t0: t, done: false, dl: vals.map(() => rng() * 0.22) };
      }
      function crown(g, x, y, s, col) {
        g.beginPath(); g.moveTo(x - s, y); g.lineTo(x - s, y - s * 0.9); g.lineTo(x - s * 0.5, y - s * 0.4); g.lineTo(x, y - s * 1.1); g.lineTo(x + s * 0.5, y - s * 0.4); g.lineTo(x + s, y - s * 0.9); g.lineTo(x + s, y); g.closePath();
        g.fillStyle = col; g.fill();
      }

      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        if (!cyc || t - cyc.t0 > CYC) {
          if (cyc && !cyc.done) { pub.push(S.mode === 'max' ? cyc.vals[cyc.ch] : cyc.re); cyc.done = true; }
          if (pub.length > 400) pub.shift();
          upd(); newCycle(t);
        }
        const f = (t - cyc.t0) / CYC;
        const s = sigma();
        const pad = 16, lab = w < 520 ? 0 : 92;
        const x0 = pad + lab, x1 = w - pad;
        const R = 4.2 * s;
        const X = (v) => x0 + ((V.clamp(v, -R, R) + R) / (2 * R)) * (x1 - x0);
        const yPub = h * (lab ? 0.17 : 0.13), yLane = h * 0.44, hy0 = h * 0.58, hy1 = h - 40;
        if (lab) {
          V.text(g, 'leaderboard', pad, yPub + 4, { size: 11.5, color: C.ink, font: 'mono', weight: 600 });
          V.text(g, 'private tests', pad, yLane + 4, { size: 11.5, color: C.muted, font: 'mono' });
          V.text(g, 'published', pad, (hy0 + hy1) / 2, { size: 11.5, color: C.muted, font: 'mono' });
        } else {
          V.text(g, 'leaderboard', x0, yPub + 30, { size: 11, color: C.ink, font: 'mono', weight: 600 });
          V.text(g, 'private tests', x0, yLane + 28, { size: 11, color: C.muted, font: 'mono' });
          V.text(g, 'published', x0, hy0 - 4, { size: 11, color: C.muted, font: 'mono' });
        }
        V.line(g, x0, yPub + 12, x1, yPub + 12, { color: V.alpha(C.muted, 0.35), lw: 1 });
        V.line(g, x0, yLane + 10, x1, yLane + 10, { color: V.alpha(C.muted, 0.35), lw: 1 });
        // truth
        V.line(g, X(0), yPub - 24, X(0), hy1, { color: C.a, lw: 1.6, dash: [5, 4] });
        V.text(g, lab ? 'true skill (all equal)' : 'true skill', X(0) + 6, yPub - 18, { size: 11, color: C.a, font: 'mono' });
        // balls
        const pDrop = V.clamp(f / 0.42, 0, 1);
        const crowned = f > 0.45;
        cyc.vals.forEach((v, i) => {
          const loc = V.clamp((pDrop - cyc.dl[i]) / 0.6, 0, 1);
          if (loc <= 0) return;
          const isCh = i === cyc.ch;
          let y = V.lerp(yLane - 70, yLane, V.ease.out(loc));
          let x = X(v);
          let a = crowned && !isCh ? 0.3 : 0.9;
          if (isCh && f > 0.55) {
            const u = V.ease.inOut(V.clamp((f - 0.55) / 0.22, 0, 1));
            y = V.lerp(yLane, yPub, u);
            if (S.mode === 'hold') { const u2 = V.ease.inOut(V.clamp((f - 0.8) / 0.14, 0, 1)); x = V.lerp(X(v), X(cyc.re), u2); }
          }
          g.globalAlpha = a;
          V.dot(g, x, y, 6.5, isCh && crowned ? C.d : C.b);
          g.globalAlpha = 1;
          if (isCh && crowned) {
            crown(g, x, y - 9, 7, C.d);
            if (f > 0.78) {
              const val = S.mode === 'hold' && f > 0.86 ? cyc.re : v;
              const lt = sgn(val, 0) + ' pts' + (S.mode === 'hold' && f > 0.86 ? ' (re-measured)' : '');
              const right = x + 14 + tw(g, lt, 12, 'mono', 700) < x1;
              V.text(g, lt, right ? x + 14 : x - 14, y + 4, { size: 12, color: C.ink, align: right ? 'left' : 'right', font: 'mono', weight: 700 });
            }
          }
        });
        // histogram of the published values
        const NBn = 24, bins = new Array(NBn).fill(0);
        pub.forEach((v) => { const b = Math.floor(((V.clamp(v, -R, R - 1e-6) + R) / (2 * R)) * NBn); bins[b]++; });
        const mxb = Math.max(3, ...bins), bw = (x1 - x0) / NBn;
        bins.forEach((c, i) => { if (!c) return; const bh = (c / mxb) * (hy1 - hy0 - 10); g.fillStyle = V.alpha(S.mode === 'max' ? C.d : C.a, 0.7); V.rr(g, x0 + i * bw + 1, hy1 - bh, bw - 2, bh, 2); g.fill(); });
        V.line(g, x0, hy1, x1, hy1, { color: V.alpha(C.muted, 0.6), lw: 1 });
        if (pub.length) { const m = pub.reduce((a, b) => a + b, 0) / pub.length; V.line(g, X(m), hy0, X(m), hy1, { color: C.ink, lw: 1.6, dash: [3, 3] }); diamond(g, X(m), hy0, 6, C.ink); }
        const stp = niceStep(2 * R, 6);
        for (let v = Math.ceil(-R / stp) * stp; v <= R; v += stp) V.text(g, (v > 0 ? '+' : v < 0 ? '−' : '') + Math.abs(Math.round(v)), X(v), hy1 + 18, { size: 11, color: C.muted, align: 'center', font: 'mono' });
        V.text(g, 'measured score − true skill (Elo points) · ◆ mean', x1, hy1 + 34, { size: 11, color: C.muted, align: 'right', maxW: x1 - x0 });
      });
    })();

    /* ================================================ FIG 6 — length-controlled WR */
    (function () {
      const fig = root.querySelector('#fig-lc');
      if (!fig) return;
      adaptH(fig, 420, 580);
      const N = 300, SD = 180;
      const S = { mu: 200, phi: 1.2, wr: 0.5, seed: 3 };
      let D = [], X = [], Y = [], J = [], sx = 1, th = [0, 0], shown = [0, 0], it = 0, lastStep = 0, raw = 0, H = null, t0 = 0;
      const ctl = UI.controls(fig);
      UI.slider(ctl, { label: 'extra tokens', min: 0, max: 400, step: 25, value: S.mu, fmt: (v) => '+' + v + ' tok', onInput: (v) => { S.mu = v; gen(); } });
      UI.slider(ctl, { label: 'judge bias φ', min: 0, max: 2, step: 0.1, value: S.phi, fmt: (v) => fmt(v, 1), onInput: (v) => { S.phi = v; gen(); } });
      UI.slider(ctl, { label: 'true WR-LC', min: 35, max: 65, step: 1, value: 50, fmt: (v) => v + '%', onInput: (v) => { S.wr = v / 100; gen(); } });
      UI.button(ctl, '↻ new sample', () => { S.seed++; gen(); });
      const rRaw = UI.readout(ctl, 'raw WR'), rLC = UI.readout(ctl, 'WR-LC (fitted)'), rPhi = UI.readout(ctl, 'φ̂ (length)');
      function gen() {
        const r = V.rng(500 + S.seed * 31);
        D = []; for (let i = 0; i < N; i++) D.push(S.mu + SD * r.normal());
        const m = D.reduce((a, b) => a + b, 0) / N;
        sx = Math.sqrt(D.reduce((a, b) => a + (b - m) * (b - m), 0) / N) || 1;
        X = D.map((d) => Math.tanh(d / sx));
        const th0 = logit(S.wr);
        Y = X.map((x) => (r() < sig(th0 + S.phi * x) ? 1 : 0));
        J = Y.map(() => r.normal());
        raw = Y.reduce((a, b) => a + b, 0) / N;
        th = [0, 0]; it = 0; t0 = H ? H.t : 0; lastStep = t0;
        upd();
      }
      function newton() {
        let g0 = 0, g1 = 0, h00 = 1e-6, h01 = 0, h11 = 1e-6;
        for (let i = 0; i < N; i++) { const x = X[i], p = sig(th[0] + th[1] * x), e = Y[i] - p, ww = p * (1 - p); g0 += e; g1 += e * x; h00 += ww; h01 += ww * x; h11 += ww * x * x; }
        const det = h00 * h11 - h01 * h01;
        th[0] += (h11 * g0 - h01 * g1) / det; th[1] += (-h01 * g0 + h00 * g1) / det;
        it++;
      }
      function upd() {
        rRaw.set(fmt(raw * 100, 1) + '%', Math.abs(raw - S.wr) > 0.06 ? 'bad' : null);
        rLC.set(it ? fmt(sig(th[0]) * 100, 1) + '%' + (it < 6 ? '…' : '') : 'fitting…', it >= 6 ? (Math.abs(sig(th[0]) - S.wr) < 0.05 ? 'ok' : 'warn') : null);
        rPhi.set(it ? fmt(th[1], 2) : '—');
      }
      gen();

      H = V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        if (it < 8 && t - lastStep > 0.45) { newton(); lastStep = t; upd(); }
        shown[0] += (th[0] - shown[0]) * 0.15; shown[1] += (th[1] - shown[1]) * 0.15;
        const narrow = w < 600;
        const pad = 14;
        const P = narrow ? { x: pad + 34, y: 30, w: w - pad * 2 - 40, h: h - 30 - 120 } : { x: pad + 40, y: 30, w: w * 0.72 - pad - 40, h: h - 30 - 52 };
        const xmin = Math.min(-350, S.mu - 2.6 * SD), xmax = Math.max(350, S.mu + 2.6 * SD);
        const PX = (d) => P.x + ((d - xmin) / (xmax - xmin)) * P.w;
        const PY = (p) => P.y + (1 - p) * P.h;
        // grid
        [0, 0.25, 0.5, 0.75, 1].forEach((p) => { V.line(g, P.x, PY(p), P.x + P.w, PY(p), { color: V.alpha(C.muted, p === 0.5 ? 0.35 : 0.15), lw: 1 }); V.text(g, Math.round(p * 100) + '%', P.x - 6, PY(p) + 4, { size: 11, color: C.muted, align: 'right', font: 'mono' }); });
        const stp = niceStep(xmax - xmin, 6);
        for (let d = Math.ceil(xmin / stp) * stp; d <= xmax; d += stp) V.text(g, (d > 0 ? '+' : d < 0 ? '−' : '') + Math.abs(d), PX(d), P.y + P.h + 17, { size: 11, color: C.muted, align: 'center', font: 'mono' });
        V.text(g, 'Δℓ = candidate tokens − base tokens', P.x + P.w, P.y + P.h + 34, { size: 11, color: C.muted, align: 'right' });
        V.text(g, 'P(candidate wins) · Newton: iteration ' + it, P.x, P.y - 14, { size: 11.5, color: C.muted, font: 'mono' });
        V.text(g, 'wins', P.x + P.w - 2, PY(1) + 26, { size: 11, color: C.b, align: 'right', font: 'mono' });
        V.text(g, 'losses', P.x + P.w - 2, PY(0) - 26, { size: 11, color: C.muted, align: 'right', font: 'mono' });
        // points
        for (let i = 0; i < N; i++) {
          if (D[i] < xmin || D[i] > xmax) continue;
          const y = Y[i] ? PY(1) + 10 + V.clamp(J[i], -2, 2) * 3 : PY(0) - 10 + V.clamp(J[i], -2, 2) * 3;
          g.fillStyle = V.alpha(Y[i] ? C.b : C.muted, 0.55);
          g.beginPath(); g.arc(PX(D[i]), y, 2.6, 0, Math.PI * 2); g.fill();
        }
        // lines: Δℓ = 0, mean
        V.line(g, PX(0), P.y, PX(0), P.y + P.h, { color: C.a, lw: 1.5, dash: [4, 3] });
        V.line(g, PX(S.mu), P.y + 20, PX(S.mu), P.y + P.h - 20, { color: V.alpha(C.ink, 0.4), lw: 1, dash: [1, 3] });
        if (S.mu >= 100) V.text(g, 'mean +' + S.mu, PX(S.mu) + 4, P.y + P.h - 44, { size: 11, color: C.muted, font: 'mono' });
        // raw WR
        V.line(g, P.x, PY(raw), P.x + P.w, PY(raw), { color: C.c, lw: 1.6, dash: [6, 4] });
        const endV = sig(shown[0] + shown[1] * Math.tanh(xmax / sx));
        V.text(g, 'raw WR ' + fmt(raw * 100, 1) + '%', P.x + P.w - 4, PY(raw) + (endV > raw ? 17 : -7), { size: 11.5, color: C.c, align: 'right', font: 'mono', weight: 600 });
        // curve
        g.beginPath();
        for (let k = 0; k <= 120; k++) { const d = xmin + ((xmax - xmin) * k) / 120; const p = sig(shown[0] + shown[1] * Math.tanh(d / sx)); const x = PX(d), y = PY(p); if (k) g.lineTo(x, y); else g.moveTo(x, y); }
        g.strokeStyle = C.a; g.lineWidth = 2.6; g.stroke();
        const lc = sig(shown[0]);
        V.glow(g, PX(0), PY(lc), 16, C.a, 0.8);
        V.dot(g, PX(0), PY(lc), 5, C.a);
        const lcTxt = 'WR-LC ' + fmt(lc * 100, 1) + '%', lcW = tw(g, lcTxt, 12, 'mono', 700);
        const lcX = Math.max(P.x + 4, PX(0) - 14 - lcW), lcY = PY(lc) + (shown[1] >= 0 ? -16 : 16);
        V.box(g, lcX - 5, lcY - 11, lcW + 10, 19, { r: 5, fill: V.alpha(C.bg || C.panel, 0.88), stroke: V.alpha(C.a, 0.6), lw: 1 });
        V.text(g, lcTxt, lcX, lcY + 3, { size: 12, color: C.a, font: 'mono', weight: 700 });
        // summary bars
        const bars = [['raw', raw, C.c], ['LC', sig(th[0]), C.a], ['truth', S.wr, C.muted]];
        if (!narrow) {
          const bx = w * 0.76, bw = w * 0.24 - pad, by = P.y, bh = P.h;
          V.text(g, 'summary', bx, by - 12, { size: 11.5, color: C.muted, font: 'mono' });
          const cw = bw / 3;
          bars.forEach(([n, v, col], k) => {
            const hh = v * bh, x = bx + k * cw + 6;
            g.fillStyle = V.alpha(col, k === 2 ? 0.25 : 0.7); V.rr(g, x, by + bh - hh, cw - 12, hh, 4); g.fill();
            V.text(g, fmt(v * 100, 0) + '%', x + (cw - 12) / 2, by + bh - hh - 6, { size: 12, color: C.ink, align: 'center', font: 'mono', weight: 600 });
            V.text(g, n, x + (cw - 12) / 2, by + bh + 17, { size: 11, color: C.muted, align: 'center', font: 'mono' });
          });
          V.line(g, bx, by + bh - 0.5 * bh, bx + bw, by + bh - 0.5 * bh, { color: V.alpha(C.muted, 0.5), lw: 1, dash: [3, 3] });
        } else {
          const by = P.y + P.h + 50, bw = (w - pad * 2) / 3;
          bars.forEach(([n, v, col], k) => {
            const x = pad + k * bw;
            V.text(g, n, x + 4, by + 12, { size: 11, color: C.muted, font: 'mono' });
            g.fillStyle = V.alpha(col, 0.18); V.rr(g, x + 4, by + 20, bw - 14, 12, 4); g.fill();
            g.fillStyle = V.alpha(col, k === 2 ? 0.45 : 0.8); V.rr(g, x + 4, by + 20, (bw - 14) * v, 12, 4); g.fill();
            V.text(g, fmt(v * 100, 1) + '%', x + 4, by + 50, { size: 12, color: C.ink, font: 'mono', weight: 600 });
          });
        }
      });
    })();

    /* ================================================ FIG 7 — McNemar */
    (function () {
      const fig = root.querySelector('#fig-mcnemar');
      if (!fig) return;
      adaptH(fig, 430, 680);
      const N = 500, BASEOK = 390, COLS = 25;
      const S = { b: 20, c: 5 };
      const r = V.rng(62);
      const perm = Array.from({ length: N }, (_, i) => i);
      for (let i = N - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); const tmp = perm[i]; perm[i] = perm[j]; perm[j] = tmp; }
      // grid position of each item: items 0..389 are right on the base; the first b of them become “lost”; of the remaining 110, the first c become “gained”
      const cat = new Array(N).fill(0), chg = new Array(N).fill(-9);
      let H = null;
      function assign(tnow) {
        for (let k = 0; k < N; k++) {
          const nc = k < BASEOK ? (k < S.b ? 1 : 0) : (k - BASEOK < S.c ? 2 : 3);
          const cell = perm[k];
          if (cat[cell] !== nc) { cat[cell] = nc; chg[cell] = tnow; }
        }
      }
      assign(-9);
      function binomP(b, c) {
        const n = b + c, k = Math.min(b, c);
        let s = 0, lc = 0;
        for (let i = 0; i <= k; i++) { if (i > 0) lc += Math.log(n - i + 1) - Math.log(i); s += Math.exp(lc - n * Math.LN2); }
        return Math.min(1, 2 * s);
      }
      function stats() {
        const b = S.b, c = S.c, a = BASEOK - b, d = N - BASEOK - c;
        const p1 = BASEOK / N, p2 = (a + c) / N, dlt = (p2 - p1) * 100;
        const seU = Math.sqrt((p1 * (1 - p1)) / N + (p2 * (1 - p2)) / N) * 100;
        const seP = (Math.sqrt(Math.max(0, b + c - ((b - c) * (b - c)) / N)) / N) * 100;
        const chi = b + c ? Math.pow(Math.abs(b - c) - 1, 2) / (b + c) : 0;
        const p = b + c === 0 ? 1 : b + c < 25 ? binomP(b, c) : 1 - erf(Math.sqrt(chi / 2));
        return { a, b, c, d, p1, p2, dlt, seU, seP, chi, p };
      }
      const ctl = UI.controls(fig);
      UI.slider(ctl, { label: 'b: was right → wrong', min: 0, max: 60, step: 1, value: S.b, fmt: (v) => String(v), onInput: (v) => { S.b = v; assign(H ? H.t : 0); upd(); } });
      UI.slider(ctl, { label: 'c: was wrong → right', min: 0, max: 60, step: 1, value: S.c, fmt: (v) => String(v), onInput: (v) => { S.c = v; assign(H ? H.t : 0); upd(); } });
      const rAcc = UI.readout(ctl, 'accuracy base → cand.'), rChi = UI.readout(ctl, 'McNemar χ²'), rP = UI.readout(ctl, 'p (paired)');
      function upd() {
        const s = stats();
        rAcc.set(fmt(s.p1 * 100, 1) + '% → ' + fmt(s.p2 * 100, 1) + '%');
        rChi.set(s.b + s.c < 25 ? 'exact binomial' : fmt(s.chi, 2));
        rP.set(s.p < 0.001 ? '< 0.001' : fmt(s.p, 3), s.p < 0.05 ? (s.b > s.c ? 'bad' : 'ok') : 'warn');
      }
      upd();

      H = V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const s = stats();
        const narrow = w < 640;
        const pad = 14;
        const colsC = [V.alpha(C.a, 0.5), RED(), C.b, V.alpha(C.muted, 0.3)];
        // grid
        const G = narrow ? { x: pad, y: 28, w: w - pad * 2 } : { x: pad, y: 30, w: w * 0.42 - pad };
        const cell = G.w / COLS, rows = N / COLS;
        G.h = cell * rows;
        V.text(g, '500 battery items', G.x, G.y - 10, { size: 11.5, color: C.muted, font: 'mono' });
        for (let i = 0; i < N; i++) {
          const x = G.x + (i % COLS) * cell, y = G.y + Math.floor(i / COLS) * cell;
          const age = t - chg[i];
          const pop = age < 0.6 ? 1 + 0.5 * Math.sin((age / 0.6) * Math.PI) : 1;
          const cs = (cell - 1.5) * Math.min(pop, 1.35);
          g.fillStyle = colsC[cat[i]];
          V.rr(g, x + (cell - 1.5 - cs) / 2, y + (cell - 1.5 - cs) / 2, cs, cs, Math.min(3, cs / 4)); g.fill();
        }
        // 2×2 table
        const T = narrow ? { x: pad + 60, y: G.y + G.h + 34, w: w - pad * 2 - 70 } : { x: w * 0.47 + 62, y: 46, w: w * 0.53 - pad - 70 };
        const tc = Math.min(T.w / 2, narrow ? 64 : 74);
        const tw2 = tc * 2;
        V.text(g, 'cand. ✓', T.x + tc / 2, T.y - 8, { size: 11, color: C.muted, align: 'center', font: 'mono' });
        V.text(g, 'cand. ✗', T.x + tc * 1.5, T.y - 8, { size: 11, color: C.muted, align: 'center', font: 'mono' });
        V.text(g, 'base ✓', T.x - 8, T.y + tc / 2 + 4, { size: 11, color: C.muted, align: 'right', font: 'mono' });
        V.text(g, 'base ✗', T.x - 8, T.y + tc * 1.5 + 4, { size: 11, color: C.muted, align: 'right', font: 'mono' });
        const pul = 0.5 + 0.5 * Math.sin(t * 3.2);
        const cellsT = [[s.a, colsC[0], 'a', false], [s.b, RED(), 'b', true], [s.c, C.b, 'c', true], [s.d, colsC[3], 'd', false]];
        cellsT.forEach(([v, col, lb, disc], k) => {
          const x = T.x + (k % 2) * tc, y = T.y + Math.floor(k / 2) * tc;
          V.box(g, x + 2, y + 2, tc - 4, tc - 4, { r: 8, fill: V.alpha(typeof col === 'string' && col.startsWith('rgba') ? C.muted : col, disc ? 0.22 : 0.1), stroke: disc ? V.alpha(col, 0.6 + 0.4 * pul) : V.alpha(C.muted, 0.4), lw: disc ? 2.2 : 1 });
          V.text(g, String(v), x + tc / 2, y + tc / 2 + 2, { size: 18, font: 'mono', weight: 700, color: C.ink, align: 'center', base: 'middle' });
          V.text(g, lb, x + 10, y + 16, { size: 11, font: 'mono', color: C.muted });
        });
        V.text(g, 'only b and c matter', T.x + tw2 + 10, T.y + tc + 4, { size: 11, color: C.muted, maxW: Math.max(40, w - (T.x + tw2 + 14) - pad) });
        // intervals
        const I = narrow ? { x: pad + 8, y: T.y + tw2 + 40, w: w - pad * 2 - 16 } : { x: w * 0.47, y: T.y + tw2 + 52, w: w * 0.53 - pad - 6 };
        const lo = -12, hi = 8;
        const IX = (v) => I.x + ((V.clamp(v, lo, hi) - lo) / (hi - lo)) * I.w;
        V.text(g, 'Δ accuracy (pp) with 95% CI', I.x, I.y - 6, { size: 11.5, color: C.muted, font: 'mono' });
        const rowsI = [[narrow ? 'unpaired' : 'unpaired (independent samples)', s.seU, C.muted], [narrow ? 'paired' : 'paired (same items)', s.seP, C.d]];
        rowsI.forEach(([lb, se, col], k) => {
          const y = I.y + 22 + k * 44;
          V.text(g, lb, I.x, y, { size: 11, color: C.ink });
          const yy = y + 16, l = s.dlt - 1.96 * se, r2 = s.dlt + 1.96 * se;
          const cross = l < 0 && r2 > 0;
          V.line(g, IX(l), yy, IX(r2), yy, { color: col, lw: 3 });
          V.line(g, IX(l), yy - 6, IX(l), yy + 6, { color: col, lw: 2 }); V.line(g, IX(r2), yy - 6, IX(r2), yy + 6, { color: col, lw: 2 });
          diamond(g, IX(s.dlt), yy, 5.5, C.ink);
          V.text(g, cross ? 'crosses 0: inconclusive' : s.dlt < 0 ? 'regression' : 'improvement', I.x + I.w, y, { size: 11, color: cross ? C.d : s.dlt < 0 ? RED() : C.a, align: 'right', font: 'mono', weight: 600 });
        });
        const zy0 = I.y + 26, zy1 = I.y + 26 + 64;
        V.line(g, IX(0), zy0, IX(0), zy1, { color: V.alpha(C.ink, 0.6), lw: 1.2, dash: [3, 3] });
        [-12, -8, -4, 0, 4, 8].forEach((v) => V.text(g, (v > 0 ? '+' : v < 0 ? '−' : '') + Math.abs(v), IX(v), zy1 + 16, { size: 11, color: C.muted, align: 'center', font: 'mono' }));
        const eqY = zy1 + 42;
        if (eqY < h - 8) {
          const txt = s.b + s.c < 25 ? 'b + c < 25 → exact binomial test: p = ' + (s.p < 0.001 ? '< 0.001' : fmt(s.p, 3)) : 'χ² = (|' + s.b + ' − ' + s.c + '| − 1)² / ' + (s.b + s.c) + ' = ' + fmt(s.chi, 2) + ' → p = ' + (s.p < 0.001 ? '< 0.001' : fmt(s.p, 3));
          V.wrap(g, txt, I.x, eqY, I.w, { size: 12, font: 'mono', weight: 600, color: s.p < 0.05 ? (s.b > s.c ? RED() : C.a) : C.d, maxLines: 2 });
        }
      });
    })();

    /* ================================================ FIG 8 — contamination */
    (function () {
      const fig = root.querySelector('#fig-contam');
      if (!fig) return;
      const norm = (s) => s.toLowerCase().normalize('NFKC').replace(/[^\p{L}\p{N}\s]/gu, ' ').split(/\s+/).filter(Boolean);
      const EVAL = 'A store sells pens at 3 dollars each and notebooks at 12 dollars each. Joana bought 4 pens and 2 notebooks. How much did she spend in total?';
      const TRAIN = {
        copia: 'A store sells pens at 3 dollars each and notebooks at 12 dollars each. Pedro bought 5 pens and 1 notebook. How much did he pay?',
        leve: 'In a stationery shop, pens are sold at 3 dollars each and notebooks at 12 dollars each. Pedro got 5 pens and 1 notebook. What did he pay?',
        forte: 'Pedro spent money at a stationery shop: five pens for three dollars and one notebook for twelve. What was the total?',
      };
      const EW = norm(EVAL);
      // height per mode: in n-grams, draw measures how much room the text takes (S.need); Min-K% needs more height on phones
      const S = { mode: 'ngram', tr: 'copia', N: 13, k: 20, t0: 0, need: 0 };
      const setH = adaptH(fig, () => (S.mode === 'ngram' && S.need ? Math.max(330, Math.min(390, S.need)) : 390), () => (S.mode === 'ngram' && S.need ? Math.max(330, Math.min(470, S.need)) : 470));
      let H = null;
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Detector', options: [{ v: 'ngram', label: 'n-grams' }, { v: 'mink', label: 'Min-K% Prob' }], value: 'ngram', onChange: (v) => { S.mode = v; gN.show(v === 'ngram'); gM.show(v === 'mink'); S.t0 = H ? H.t : 0; setH(); } });
      const gN = UI.group(ctl), gM = UI.group(ctl);
      UI.seg(gN.el, { label: 'Training example', options: [{ v: 'copia', label: 'copy' }, { v: 'leve', label: 'light paraphrase' }, { v: 'forte', label: 'heavy paraphrase' }], value: 'copia', onChange: (v) => { S.tr = v; S.t0 = H ? H.t : 0; } });
      UI.slider(gN.el, { label: 'n-gram N', min: 5, max: 13, step: 1, value: 13, fmt: (v) => String(v), onInput: (v) => { S.N = v; S.t0 = H ? H.t : 0; } });
      UI.slider(gM.el, { label: 'K% (most surprising)', min: 10, max: 50, step: 5, value: 20, fmt: (v) => v + '%', onInput: (v) => { S.k = v; S.t0 = H ? H.t : 0; } });
      gM.show(false);
      // Min-K: two texts with illustrative log-probs
      const MK = [
        { t: 'seen in training', col: null, toks: ['A', 'store', 'sells', 'pens', 'at', '3', 'dollars', 'each', 'and', 'notebooks', 'at', '12', 'dollars', 'each', '.'], lp: [-2.1, -1.4, -0.9, -0.6, -0.2, -0.5, -0.1, -0.3, -0.2, -0.4, -0.1, -0.6, -0.05, -0.08, -0.1] },
        { t: 'unseen', col: null, toks: ['A', 'bazaar', 'swaps', 'comics', 'for', '2', 'old', 'stamps', 'and', 'sells', 'maps', 'at', '9', 'coins', '.'], lp: [-2.3, -7.1, -3.4, -6.4, -1.2, -2.6, -5.2, -4.8, -0.9, -2.2, -5.5, -1.1, -3.0, -5.9, -0.6] },
      ];
      const minK = (lp, k) => { const n = Math.max(1, Math.round((lp.length * k) / 100)); const s = lp.slice().sort((a, b) => a - b).slice(0, n); return { n, v: s.reduce((a, b) => a + b, 0) / n, thr: s[n - 1] }; };

      function layoutWords(g, words, x0, y0, maxW, size) {
        const out = []; let x = x0, y = y0;
        words.forEach((wd) => { const ww = tw(g, wd, size, 'mono', 500) + 10; if (x + ww > x0 + maxW && x > x0) { x = x0; y += size + 14; } out.push({ x, y, w: ww }); x += ww + 4; });
        return { rects: out, bottom: y + size + 10 };
      }

      H = V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const pad = 14;
        const el = t - S.t0;
        if (S.mode === 'ngram') {
          const TW = norm(TRAIN[S.tr]);
          const N = S.N;
          const evalSet = new Set(); for (let i = 0; i + N <= EW.length; i++) evalSet.add(EW.slice(i, i + N).join(' '));
          const nWin = Math.max(0, TW.length - N + 1);
          const stepT = 0.26;
          const cyc = nWin * stepT + 2.2;
          const lt = el % cyc;
          const pos = Math.min(nWin - 1, Math.floor(lt / stepT));
          const finished = lt >= nWin * stepT;
          const hitW = [];
          for (let i = 0; i < nWin; i++) hitW.push(evalSet.has(TW.slice(i, i + N).join(' ')));
          const size = w < 520 ? 11.5 : 14;
          V.text(g, 'evaluation item (test)', pad, 20, { size: 12, color: C.muted, font: 'mono' });
          const LE = layoutWords(g, EW, pad, 30, w - pad * 2, size);
          // test words matched by the windows seen so far
          const evMatch = new Array(EW.length).fill(false);
          for (let i = 0; i <= (finished ? nWin - 1 : pos); i++) if (hitW[i]) { const gram = TW.slice(i, i + N).join(' '); for (let j = 0; j + N <= EW.length; j++) if (EW.slice(j, j + N).join(' ') === gram) for (let q = j; q < j + N; q++) evMatch[q] = true; }
          EW.forEach((wd, i) => { const r = LE.rects[i]; chip(g, r.x, r.y, r.w, size + 9, wd, { size, fill: evMatch[i] ? V.alpha(RED(), 0.3) : V.alpha(C.a, 0.1), stroke: evMatch[i] ? RED() : V.alpha(C.a, 0.5), weight: 500 }); });
          const ty0 = LE.bottom + 26;
          V.text(g, 'training example generated by the teacher', pad, ty0 - 10, { size: 12, color: C.muted, font: 'mono' });
          const LT = layoutWords(g, TW, pad, ty0, w - pad * 2, size);
          const inWin = (i) => !finished && i >= pos && i < pos + N;
          const everHit = new Array(TW.length).fill(false);
          for (let i = 0; i <= (finished ? nWin - 1 : pos); i++) if (hitW[i]) for (let q = i; q < i + N; q++) everHit[q] = true;
          TW.forEach((wd, i) => {
            const r = LT.rects[i];
            const win = inWin(i), hitNow = win && hitW[pos];
            chip(g, r.x, r.y, r.w, size + 9, wd, { size, fill: everHit[i] ? V.alpha(RED(), 0.28) : win ? V.alpha(C.b, 0.25) : V.alpha(C.muted, 0.08), stroke: hitNow ? RED() : win ? C.b : V.alpha(C.muted, 0.4), lw: win ? 2 : 1.2, weight: 500 });
          });
          const nHit = hitW.filter(Boolean).length;
          const yInfo = LT.bottom + 22;
          V.text(g, 'window of ' + N + ' words · position ' + (finished ? nWin : pos + 1) + '/' + nWin + ' · shared with the test: ' + hitW.slice(0, finished ? nWin : pos + 1).filter(Boolean).length, pad, yInfo, { size: 11.5, color: C.ink, font: 'mono', maxW: w - pad * 2 });
          // the n-gram in the window, looked up in the set of test n-grams
          const yBox = yInfo + 14;
          // height the content needs (n-gram box or stamp + final sentence), rounded so it does not oscillate
          const need = Math.ceil((yBox + 86) / 10) * 10;
          if (need !== S.need) { S.need = need; setH(); }
          if (!finished && nWin > 0) {
            const gram = TW.slice(pos, pos + N).join(' ');
            const hit = hitW[pos];
            const bh = w < 520 ? 52 : 46;
            V.box(g, pad, yBox, w - pad * 2, bh, { r: 9, fill: V.alpha(hit ? RED() : C.b, 0.08), stroke: V.alpha(hit ? RED() : C.b, 0.5), lw: 1.2 });
            V.wrap(g, '“' + gram + '”', pad + 10, yBox + 18, w - pad * 2 - 110, { size: 11.5, font: 'mono', color: C.ink, maxLines: 2 });
            V.text(g, hit ? '∈ test ✓' : '∉ test', w - pad - 10, yBox + bh / 2 + 4, { size: 12.5, font: 'mono', weight: 700, color: hit ? RED() : C.muted, align: 'right' });
          }
          if (finished) {
            const caught = nHit > 0;
            stamp(g, w / 2, Math.min(h - 40, yBox + 30), caught ? 'REMOVED FROM TRAINING' : 'PASSED THE FILTER', caught ? RED() : C.d, { scale: easeB((lt - nWin * stepT) / 0.35), size: 14 });
            if (!caught && S.tr === 'forte') V.text(g, '…but it is the same question in other words', w / 2, Math.min(h - 10, yBox + 66), { size: 12, color: C.d, align: 'center' });
            else if (!caught) V.text(g, 'no ' + N + '-gram in common: try a smaller N', w / 2, Math.min(h - 10, yBox + 66), { size: 12, color: C.muted, align: 'center' });
            else V.text(g, nHit + ' window(s) of ' + N + ' words identical to passages of the test', w / 2, Math.min(h - 10, yBox + 66), { size: 12, color: C.muted, align: 'center' });
          }
        } else {
          const narrow = w < 560;
          const rowsY = narrow ? [26, h / 2 + 8] : [26, h / 2 + 6];
          const rowH = h / 2 - 44;
          const scan = V.clamp(el / 2.2, 0, 1);
          const res = MK.map((m) => minK(m.lp, S.k));
          MK.forEach((m, r) => {
            const y0 = rowsY[r];
            const col = r === 0 ? C.b : C.a;
            V.text(g, m.t, pad, y0, { size: 12.5, weight: 700, color: col });
            const n = m.toks.length;
            const sideW = narrow ? 0 : 150;
            const bx = pad, bw = (w - pad * 2 - sideW) / n;
            const base = y0 + (bw >= 22 ? 28 : 14);
            const scale = (rowH - 50) / 8;
            const shownN = Math.ceil(scan * n);
            const R = res[r];
            V.line(g, bx, base, bx + bw * n, base, { color: V.alpha(C.muted, 0.6), lw: 1 });
            for (let i = 0; i < shownN; i++) {
              const v = m.lp[i], hh = -v * scale;
              const low = scan >= 1 && v <= R.thr;
              const blink = low ? 0.6 + 0.4 * Math.sin(t * 5) : 1;
              g.fillStyle = low ? V.alpha(RED(), 0.85 * blink) : V.alpha(col, 0.55);
              V.rr(g, bx + i * bw + 1.5, base, bw - 3, Math.max(2, hh), 3); g.fill();
              if (bw >= 22) V.text(g, m.toks[i], bx + i * bw + bw / 2, base - 5, { size: 11, font: 'mono', color: C.muted, align: 'center', maxW: bw });
              if (low && bw >= 26) V.text(g, fmt(v, 1), bx + i * bw + bw / 2, base + hh + 13, { size: 11, font: 'mono', color: RED(), align: 'center' });
            }
            if (scan < 1) { const cx = bx + scan * bw * n; V.line(g, cx, base - 12, cx, base + rowH - 40, { color: C.d, lw: 2 }); }
            if (scan >= 1) {
              const txt = 'Min-' + S.k + '% = ' + fmt(R.v, 2) + ' (' + R.n + ' tokens)';
              if (narrow) V.text(g, txt, w - pad, y0, { size: 12, font: 'mono', weight: 700, color: C.ink, align: 'right' });
              else {
                const sx = w - pad - sideW + 14;
                V.box(g, sx, base + 10, sideW - 14, 64, { r: 10, fill: V.alpha(col, 0.1), stroke: V.alpha(col, 0.5) });
                V.text(g, 'Min-' + S.k + '%', sx + 10, base + 30, { size: 12, color: C.muted, font: 'mono' });
                V.text(g, fmt(R.v, 2), sx + 10, base + 60, { size: 22, color: C.ink, font: 'mono', weight: 700 });
              }
            }
          });
          if (scan >= 1) {
            const seen = res[0].v > res[1].v;
            if (seen) V.wrap(g, '“seen” scores higher (less negative): not even its worst surprises are surprising', pad, h - (w < 560 ? 26 : 10), w - pad * 2, { size: 11.5, color: C.muted, maxLines: 2 });
          }
        }
      });
    })();

    /* ================================================ FIG 9 — the refusal balance */
    (function () {
      const fig = root.querySelector('#fig-seguranca');
      if (!fig) return;
      adaptH(fig, 430, 720);
      const SC = {
        base: { label: 'base instruct', x: 8, y: 6, hs: -1, hh: 2, sd: 0.6 },
        sft: { label: 'benign domain SFT', x: 6.5, y: 14, hs: -0.9, hh: 1.45, sd: 0.6 },
        adv: { label: '10 adversarial examples', x: 2.5, y: 52, hs: -0.8, hh: 0.3, sd: 0.6 },
        disp: { label: 'refusal DPO w/ displacement', x: 5, y: 34, hs: -0.9, hh: 0.8, sd: 0.6 },
        exag: { label: 'exaggerated safety', x: 31, y: 2.5, hs: 0.7, hh: 2.3, sd: 0.6 },
        ablit: { label: 'abliteration', x: 1.5, y: 56, hs: 0, hh: 0.1, sd: 0.3 },
      };
      const B0 = SC.base;
      const S = { k: 'sft', t0: 0, from: { x: B0.x, y: B0.y, hs: B0.hs, hh: B0.hh, sd: B0.sd } };
      const cur = Object.assign({}, S.from);
      let H = null, trail = [];
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Intervention', options: Object.keys(SC).filter((k) => k !== 'base').map((k) => ({ v: k, label: SC[k].label })), value: 'sft', onChange: (v) => { S.from = Object.assign({}, cur); S.k = v; S.t0 = H ? H.t : 0; trail = []; upd(); } });
      const rX = UI.readout(ctl, 'undue refusal'), rY = UI.readout(ctl, 'undue compliance'), rD = UI.readout(ctl, 'probe separation'), rG = UI.readout(ctl, 'gate');
      const passes = (p) => p.x <= B0.x + 2 && p.y <= B0.y + 1;
      function upd() {
        const p = SC[S.k];
        rX.set(fmt(p.x, 1) + '%'); rY.set(fmt(p.y, 1) + '%');
        const dp = (p.hh - p.hs) / p.sd;
        rD.set(fmt(dp, 1) + ' σ (base ' + fmt((B0.hh - B0.hs) / B0.sd, 1) + ')', dp > 3.5 ? 'ok' : dp > 2 ? 'warn' : 'bad');
        rG.set(passes(p) ? 'pass' : 'fail', passes(p) ? 'ok' : 'bad');
      }
      upd();
      const gauss = (x, m, s) => Math.exp(-0.5 * ((x - m) / s) * ((x - m) / s));

      H = V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const tgt = SC[S.k];
        const u = V.ease.inOut(V.clamp((t - S.t0) / 1.6, 0, 1));
        ['x', 'y', 'hs', 'hh', 'sd'].forEach((k) => { cur[k] = V.lerp(S.from[k], tgt[k], u); });
        const narrow = w < 640;
        const pad = 14;
        const P = narrow ? { x: pad + 34, y: 22, w: w - pad * 2 - 40, h: h * 0.5 - 50 } : { x: pad + 40, y: 24, w: w * 0.56 - pad - 40, h: h - 24 - 48 };
        const XM = 40, YM = 60;
        const PX = (v) => P.x + (v / XM) * P.w, PY = (v) => P.y + P.h - (v / YM) * P.h;
        // quadrants
        const mx = PX(20), my = PY(30);
        const quad = [[P.x, P.y, mx - P.x, my - P.y, 'loose', RED()], [mx, P.y, P.x + P.w - mx, my - P.y, 'worst of both', RED()], [P.x, my, mx - P.x, P.y + P.h - my, 'ideal', C.a], [mx, my, P.x + P.w - mx, P.y + P.h - my, 'paranoid', C.d]];
        quad.forEach(([x, y, ww, hh, lb, col]) => { g.fillStyle = V.alpha(col, 0.06); g.fillRect(x, y, ww, hh); V.text(g, lb, x + ww / 2, y + 18, { size: 11.5, color: V.alpha(col, 0.95), align: 'center', font: 'mono', weight: 600 }); });
        g.strokeStyle = V.alpha(C.muted, 0.5); g.lineWidth = 1; g.strokeRect(P.x, P.y, P.w, P.h);
        [0, 10, 20, 30, 40].forEach((v) => V.text(g, v + '%', PX(v), P.y + P.h + 15, { size: 11, color: C.muted, align: 'center', font: 'mono' }));
        [0, 20, 40, 60].forEach((v) => V.text(g, v + '%', P.x - 6, PY(v) + 4, { size: 11, color: C.muted, align: 'right', font: 'mono' }));
        V.text(g, 'undue refusal on safe prompts (XSTest) →', P.x + P.w, P.y + P.h + 32, { size: 11, color: C.muted, align: 'right' });
        g.save(); g.translate(P.x - 30, P.y + P.h / 2); g.rotate(-Math.PI / 2); V.text(g, 'undue compliance →', 0, 0, { size: 11, color: C.muted, align: 'center' }); g.restore();
        // gate
        g.setLineDash([5, 4]); g.strokeStyle = C.a; g.lineWidth = 1.5; g.strokeRect(P.x, PY(B0.y + 1), PX(B0.x + 2) - P.x, P.y + P.h - PY(B0.y + 1)); g.setLineDash([]);
        V.text(g, 'gate', PX(B0.x + 2) + 4, PY(B0.y + 1) + 12, { size: 11, color: C.a, font: 'mono' });
        // base
        V.dot(g, PX(B0.x), PY(B0.y), 4, C.muted);
        V.text(g, 'base', PX(B0.x) - 7, PY(B0.y) + 16, { size: 11, color: C.muted, font: 'mono', align: 'right' });
        // trail and character
        if (u < 1 && (!trail.length || Math.hypot(trail[trail.length - 1][0] - cur.x, trail[trail.length - 1][1] - cur.y) > 0.6)) trail.push([cur.x, cur.y]);
        trail.forEach(([x, y], i) => V.dot(g, PX(x), PY(y), 2, V.alpha(C.b, 0.25 + (0.5 * i) / Math.max(1, trail.length))));
        const ok = passes(cur);
        face(g, PX(cur.x), PY(cur.y) + Math.sin(t * 3) * 1.5, 22, ok ? C.a : cur.x > 20 ? C.d : RED(), { mood: ok ? 'happy' : cur.x > 20 ? 'o' : 'sad' });
        V.text(g, tgt.label, V.clamp(PX(cur.x), P.x + 60, P.x + P.w - 60), PY(cur.y) - 18, { size: 11.5, color: C.ink, align: 'center', weight: 600 });
        // probe
        const Q = narrow ? { x: pad + 6, y: h * 0.5 + 36, w: w - pad * 2 - 12, h: h * 0.5 - 76 } : { x: w * 0.6, y: 40, w: w * 0.4 - pad - 6, h: h - 40 - 60 };
        V.box(g, Q.x - 8, Q.y - 28, Q.w + 16, Q.h + 70, { r: 12, fill: V.alpha(C.muted, 0.05), stroke: V.alpha(C.muted, 0.3), lw: 1 });
        V.text(g, 'probe: projection onto r̂ (estimated on the base)', Q.x, Q.y - 10, { size: 11.5, color: C.muted, font: 'mono', maxW: Q.w });
        const lo = -3.5, hi = 4.5;
        const QX = (v) => Q.x + ((v - lo) / (hi - lo)) * Q.w;
        const QB = Q.y + Q.h;
        const peakOf = (s) => Math.min((Q.h - 20) * 0.62 * (0.6 / s), Q.h - 34);
        const curve = (m, s, col, fill, dash) => {
          g.beginPath();
          for (let k = 0; k <= 80; k++) { const v = lo + ((hi - lo) * k) / 80; const y = QB - gauss(v, m, s) * peakOf(s); if (k) g.lineTo(QX(v), y); else g.moveTo(QX(v), y); }
          if (fill) { g.lineTo(QX(hi), QB); g.lineTo(QX(lo), QB); g.closePath(); g.fillStyle = V.alpha(col, 0.28); g.fill(); }
          else { g.strokeStyle = col; g.lineWidth = 1.6; if (dash) g.setLineDash([4, 3]); g.stroke(); g.setLineDash([]); }
        };
        curve(B0.hs, B0.sd, C.a, false, true); curve(B0.hh, B0.sd, C.c, false, true);
        curve(cur.hs, cur.sd, C.a, true); curve(cur.hh, cur.sd, C.c, true);
        V.line(g, Q.x, QB, Q.x + Q.w, QB, { color: V.alpha(C.muted, 0.6), lw: 1 });
        [-2, 0, 2, 4].forEach((v) => V.text(g, String(v).replace('-', '−'), QX(v), QB + 15, { size: 11, color: C.muted, align: 'center', font: 'mono' }));
        const close = Math.abs(cur.hh - cur.hs) < 1.6;
        const ly = QB - peakOf(cur.sd) - 8;
        V.text(g, 'harmless', QX(cur.hs) - (close ? 8 : 0), ly, { size: 11, color: C.a, align: close ? 'right' : 'center', font: 'mono', weight: 600 });
        V.text(g, 'harmful', QX(cur.hh) + (close ? 8 : 0), ly, { size: 11, color: C.c, align: close ? 'left' : 'center', font: 'mono', weight: 600 });
        const dp = (cur.hh - cur.hs) / cur.sd;
        V.text(g, 'separation: ' + fmt(dp, 1) + ' σ', Q.x, QB + 34, { size: 12, color: dp > 3.5 ? C.a : dp > 2 ? C.d : RED(), font: 'mono', weight: 700 });
      });
    })();

    /* ================================================ FIG 10 — Goodhart and the KL leash */
    (function () {
      const fig = root.querySelector('#fig-goodhart');
      if (!fig) return;
      adaptH(fig, 430, 600);
      const NS = [1, 2, 4, 16, 64, 256, 1000, 4000, 16000, 60000];
      const S = { mode: 'rl', bi: 6, ni: 6, t0: 0 };
      const BK = [0.02, 0.03, 0.04, 0.05, 0.07, 0.1, 0.14, 0.2, 0.3, 0.5, 1];
      // gold reward from Gao et al. with illustrative coefficients: RL α=0.5 β=0.2 (peak d=e^1.5≈4.48, R*≈0.90); BoN α=1 β=0.25 (peak d=2, R*=1)
      const goldRL = (d) => (d <= 0 ? 0 : d * (0.5 - 0.2 * Math.log(d)));
      const goldBon = (d) => (d <= 0 ? 0 : d * (1 - 0.25 * d));
      const gold = (d) => (S.mode === 'rl' ? goldRL(d) : goldBon(d));
      const proxy = (d) => 0.9 * Math.pow(Math.max(0, d), 0.75);
      // RL: maximizes proxy(d) − β_KL·d²  →  0.675·d^(−1/4) = 2β·d  →  d* = (0.3375/β)^(4/5)
      const dStar = () => (S.mode === 'rl' ? Math.pow(0.3375 / BK[S.bi], 0.8) : Math.sqrt(Math.max(0, Math.log(NS[S.ni]) - (NS[S.ni] - 1) / NS[S.ni])));
      const peak = () => (S.mode === 'rl' ? Math.exp(0.5 / 0.2 - 1) : 2);
      let H = null;
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Optimizer', options: [{ v: 'rl', label: 'RL with KL' }, { v: 'bon', label: 'best-of-n' }], value: 'rl', onChange: (v) => { S.mode = v; gR.show(v === 'rl'); gB.show(v === 'bon'); S.t0 = H ? H.t : 0; upd(); } });
      const gR = UI.group(ctl), gB = UI.group(ctl);
      UI.slider(gR.el, { label: 'leash β_KL', min: 0, max: BK.length - 1, step: 1, value: S.bi, fmt: (v) => fmt(BK[v], 2), onInput: (v) => { S.bi = v; S.t0 = H ? H.t : 0; upd(); } });
      UI.slider(gB.el, { label: 'n (samples)', min: 0, max: NS.length - 1, step: 1, value: S.ni, fmt: (v) => NS[v].toLocaleString('en-US'), onInput: (v) => { S.ni = v; S.t0 = H ? H.t : 0; upd(); } });
      gB.show(false);
      const rK = UI.readout(ctl, 'KL reached'), rP = UI.readout(ctl, 'proxy'), rG = UI.readout(ctl, 'gold (truth)'), rS = UI.readout(ctl, 'status');
      function upd() {
        const d = dStar();
        rK.set(fmt(d * d, 2) + ' nats');
        rP.set(fmt(proxy(d), 2));
        rG.set(fmt(gold(d), 2));
        const pk = peak();
        rS.set(d < pk * 0.9 ? 'before the peak' : d <= pk * 1.1 ? 'at the peak' : 'Goodhart: past the peak', d < pk * 0.9 ? 'ok' : d <= pk * 1.1 ? 'ok' : 'bad');
      }
      upd();
      function robot(g, x, y, s, col, mood) {
        face(g, x, y, s, col, { mood });
        V.line(g, x, y - s / 2, x, y - s / 2 - 7, { color: col, lw: 2 });
        V.dot(g, x, y - s / 2 - 9, 2.6, col);
      }

      H = V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const narrow = w < 600;
        const pad = 14;
        const P = { x: pad + 36, y: 40, w: w - pad * 2 - 44 - (narrow ? 0 : 170), h: h - 40 - (narrow ? 120 : 56) };
        // RL has its peak far out (d ≈ 4.5) and falls slowly (log d): axis up to d = 10; best-of-n fits in d ≤ 5
        const DM = S.mode === 'rl' ? 10 : 5, Y0 = -0.8, Y1 = 3.4;
        const PX = (d) => P.x + (d / DM) * P.w;
        const PY = (v) => P.y + P.h - ((V.clamp(v, Y0, Y1) - Y0) / (Y1 - Y0)) * P.h;
        // grid and axes
        const tickStep = DM > 6 && P.w < 460 ? 2 : 1;
        for (let d = 0; d <= DM; d++) {
          V.line(g, PX(d), P.y, PX(d), P.y + P.h, { color: V.alpha(C.muted, 0.15), lw: 1 });
          if (d % tickStep) continue;
          V.text(g, String(d), PX(d), P.y + P.h + 15, { size: 11, color: C.muted, align: 'center', font: 'mono' });
          V.text(g, fmt(d * d, 0), PX(d), P.y - 8, { size: 11, color: C.muted, align: 'center', font: 'mono' });
        }
        V.text(g, 'KL', P.x - 8, P.y - 8, { size: 11, color: C.muted, align: 'right', font: 'mono' });
        V.text(g, 'd = √KL', P.x + P.w, P.y + P.h + 32, { size: 11, color: C.muted, align: 'right', font: 'mono' });
        [0, 1, 2, 3].forEach((v) => { V.line(g, P.x, PY(v), P.x + P.w, PY(v), { color: V.alpha(C.muted, v === 0 ? 0.5 : 0.15), lw: 1 }); V.text(g, String(v), P.x - 6, PY(v) + 4, { size: 11, color: C.muted, align: 'right', font: 'mono' }); });
        // curves (return where they leave the plot at the bottom/top)
        const plot = (fn, col, lw, dash) => { let exitLo = null, exitHi = null; g.beginPath(); let pen = false; for (let k = 0; k <= 200; k++) { const d = (DM * k) / 200; const v = fn(d); if (v < Y0 || v > Y1) { if (pen && v < Y0 && exitLo == null) exitLo = d; if (pen && v > Y1 && exitHi == null) exitHi = d; pen = false; continue; } const x = PX(d), y = PY(v); if (pen) g.lineTo(x, y); else g.moveTo(x, y); pen = true; } g.strokeStyle = col; g.lineWidth = lw; if (dash) g.setLineDash(dash); g.stroke(); g.setLineDash([]); return { exitLo, exitHi }; };
        // the gold of the OTHER optimizer, faded, on the same axis: RL needs more KL to reach a lower peak
        const ghostFn = S.mode === 'rl' ? goldBon : goldRL;
        plot(ghostFn, V.alpha(C.a, 0.3), 2, [3, 3]);
        const ghostTxt = '··· gold of ' + (S.mode === 'rl' ? 'best-of-n' : 'RL') + ', same axis';
        const ex = plot(proxy, C.b, 2.4, [7, 4]);
        if (ex.exitHi != null) V.text(g, '↑ keeps rising', PX(ex.exitHi) + 6, PY(Y1) + 12, { size: 11, color: C.b, font: 'mono' });
        const eg = plot(gold, C.a, 2.8);
        if (eg.exitLo != null) V.text(g, '↓ plummets', PX(eg.exitLo) + 4, PY(Y0) - 8, { size: 11, color: C.a, font: 'mono' });
        // proxy label to the left of its own curve, above everything the robot draws (proxy(d) = 2.4 at d ≈ 3.69)
        const dLab = Math.pow(2.4 / 0.9, 4 / 3);
        V.text(g, narrow ? 'proxy' : 'proxy (what the RM sees)', PX(dLab) - 10, PY(2.4), { size: 11.5, color: C.b, align: 'right', font: 'mono', weight: 600 });
        const pk = peak();
        V.text(g, 'gold', PX(pk), PY(gold(pk)) - 12, { size: 11.5, color: C.a, align: 'center', font: 'mono', weight: 700 });
        V.line(g, PX(pk), PY(gold(pk)) - 4, PX(pk), P.y + P.h, { color: V.alpha(C.a, 0.5), lw: 1, dash: [2, 3] });
        V.text(g, 'peak', PX(pk) + 4, PY(0.42), { size: 11, color: C.a, font: 'mono' });
        // walk
        const dT = Math.min(dStar(), DM);
        const el = t - S.t0;
        const walk = V.clamp(el / 3.2, 0, 1);
        const loop = el > 5.2;
        if (loop) S.t0 = t;
        const d = dT * V.ease.out(walk);
        const gy = PY(gold(d)), py = PY(proxy(d));
        // anchor and leash
        const ax = PX(0), ay = P.y + P.h + 2;
        V.dot(g, ax, ay, 5, C.ink);
        V.text(g, 'π_ref', ax + 2, ay + 30, { size: 11, color: C.ink, font: 'mono', align: 'center' });
        const rx = PX(d), ry = P.y + P.h - 16;
        if (S.mode === 'rl') {
          const taut = walk >= 1 ? 0 : 6 * (1 - walk);
          g.beginPath(); g.moveTo(ax, ay);
          for (let k = 1; k <= 24; k++) { const u = k / 24; const xx = V.lerp(ax, rx, u), yy = V.lerp(ay, ry, u) + Math.sin(u * Math.PI * 4 + t * 6) * taut * Math.sin(u * Math.PI); g.lineTo(xx, yy); }
          g.strokeStyle = V.alpha(C.d, 0.9); g.lineWidth = 2; g.stroke();
        }
        const past = d > pk * 1.1;
        robot(g, rx, ry, 20, past ? RED() : C.b, past ? 'sad' : 'happy');
        V.dot(g, rx, py, 5, C.b); V.dot(g, rx, gy, 5.5, past ? RED() : C.a);
        V.line(g, rx, py, rx, gy, { color: V.alpha(C.ink, 0.3), lw: 1, dash: [2, 2] });
        if (walk >= 1 && past) stamp(g, S.mode === 'rl' ? PX(Math.min(dT, DM * 0.62)) : PX(DM * 0.84), S.mode === 'rl' ? PY(2.3) : PY(1.6), 'GOODHART!', RED(), { scale: easeB((el - 3.2) / 0.3), size: 13 });
        // metrics panel (wide) / row (narrow)
        const lines = [['reward (proxy)', fmt(proxy(d), 2), C.b], ['kl', fmt(d * d, 2) + ' nats', C.d], ['gold (human)', fmt(gold(d), 2), past ? RED() : C.a]];
        if (!narrow) {
          const bx = w - pad - 160, by = P.y;
          V.box(g, bx, by, 160, 132, { r: 10, fill: V.alpha(C.muted, 0.06), stroke: V.alpha(C.muted, 0.3) });
          V.text(g, 'what the log shows', bx + 10, by + 20, { size: 11.5, color: C.muted, font: 'mono' });
          lines.forEach(([k, v, col], i) => { V.text(g, k, bx + 10, by + 46 + i * 28, { size: 11.5, color: C.ink, font: 'mono' }); V.text(g, v, bx + 150, by + 46 + i * 28, { size: 12, color: col, align: 'right', font: 'mono', weight: 700 }); });
          V.wrap(g, S.mode === 'rl' ? 'high β_KL = short leash: stops before the peak' : 'best-of-n KL = log n − (n−1)/n', bx, by + 156, 160, { size: 11, color: C.muted, maxLines: 3 });
          V.wrap(g, ghostTxt, bx, by + 200, 160, { size: 11, color: V.alpha(C.a, 0.8), maxLines: 2 });
        } else {
          const by = P.y + P.h + 52;
          lines.forEach(([k, v, col], i) => { const x = pad + (i * (w - pad * 2)) / 3; V.text(g, k, x, by, { size: 11, color: C.muted, font: 'mono' }); V.text(g, v, x, by + 18, { size: 12, color: col, font: 'mono', weight: 700 }); });
          V.text(g, ghostTxt, pad, by + 42, { size: 11, color: V.alpha(C.a, 0.8), font: 'mono', maxW: w - pad * 2 });
        }
      });
    })();
  },
});

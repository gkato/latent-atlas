LA.chapter({
  id: 'preferencias-reward-models',
  part: 11,
  order: 53,
  title: 'Preferências humanas e reward models',
  navTitle: 'Preferências & RMs',
  subtitle: 'Como transformar “prefiro esta resposta” num número que um otimizador consegue subir: comparações e concordância, Bradley–Terry, a perda do RM e o seu gradiente, calibração, reward hacking, recompensas de processo e juízes de IA.',
  minutes: 40,
  recipes: [],
  tags: ['RLHF', 'reward model', 'Bradley–Terry', 'Plackett–Luce', 'κ de Cohen', 'calibração', 'reward hacking', 'superotimização', 'PRM', 'LLM-as-judge', 'RLAIF', 'RewardBench'],
  goals: [
    'Explicar por que o pós-treino coleta <b>comparações</b> em vez de notas, quais formatos existem e como o <b>κ de Cohen</b> mede a concordância que limita a acurácia do RM',
    'Derivar <b>Bradley–Terry</b> a partir de utilidades com ruído (Gumbel, Thurstone, Elo), estender a rankings (Plackett–Luce, os \\(\\binom{K}{2}\\) pares do InstructGPT) e ler o gradiente de \\(-\\log\\sigma(r_w - r_l)\\)',
    'Descrever um RM por dentro (LLM + cabeça escalar lida no EOS), a receita de treino, a normalização e a calibração por temperatura',
    'Reconhecer viés de comprimento, reward hacking e superotimização (leis de Gao), e as defesas: KL, ensembles e WARM',
    'Comparar ORM, PRM (PRM800K, Math-Shepherd), juízes generativos e RLAIF, e avaliar RMs com o RewardBench',
  ],
  prereqs: ['pos-treino-mapa', 'sft'],
  summary: [
    'Humanos dão <b>comparações</b> muito mais consistentes do que notas absolutas: a comparação cancela a régua pessoal de cada anotador, ao preço de perder a magnitude. A concordância entre humanos (~63–77% em pares; κ desconta o acaso) é o teto prático da acurácia de um RM.',
    '<b>Bradley–Terry</b>: com utilidade \\(r+\\epsilon\\) e ruído Gumbel, \\(P(y_w\\succ y_l) = \\sigma(r_w - r_l)\\); a diferença de recompensas é um log-odds. Elo é o mesmo modelo (\\(\\Delta r = \\tfrac{\\ln 10}{400}\\Delta R\\)). Rankings: Plackett–Luce, ou os \\(\\binom{K}{2}\\) pares do InstructGPT agrupados no mesmo elemento do batch (\\(K\\) forwards por prompt).',
    'A perda \\(-\\log\\sigma(r_w - r_l)\\) é entropia cruzada binária sobre \\(\\Delta\\); o gradiente pesa cada par por \\(\\sigma(-\\Delta)\\), a chance de o RM estar errado. É invariante a translação (daí a normalização para média 0) mas não a escala (que conversa com o \\(\\beta\\) da KL). Variantes: margem (Llama 2), BT escalado (HelpSteer2-Preference), rótulos suaves.',
    'Arquitetura: o Transformer SFT sem o <code>lm_head</code>, com uma cabeça \\(r = w^\\top h_{\\text{EOS}} + b\\) lida <b>só no token EOS</b>. Treino de 1 época, dropout desligado, várias sementes. Pontuar é um prefill; juízes generativos exigem decode.',
    'Avaliação: acurácia de pares, generalização para fora da distribuição e <b>calibração</b> (ECE; um único \\(T\\) em \\(\\sigma(\\Delta/T)\\) conserta a escala sem mudar a ordem). RewardBench: 2.985 pares em 4 categorias; o RewardBench 2 usa best-of-4.',
    'Goodhart: otimizar o proxy demais explora os erros dele. O <b>comprimento</b> explica 70–90% do ganho de recompensa do PPO em WebGPT e RLCD (Singhal). Gao: o escore ouro segue \\(d(\\alpha-\\beta d)\\) no best-of-\\(n\\) e \\(d(\\alpha-\\beta\\log d)\\) no RL, com \\(d=\\sqrt{\\text{KL}}\\). Ensembles conservadores (WCO/UWO) e WARM corrigem erros próprios de cada RM, não os compartilhados.',
    'ORM pontua o resultado; PRM, cada passo (produto das probabilidades; 78,2% contra 72,4% no best-of-1860 do MATH). Math-Shepherd rotula passos por Monte Carlo (SE ≈ valor do passo). Juízes LLM (85% de concordância com humanos), GenRM e RLAIF trocam humanos por modelos, e herdam os vieses de posição, verbosidade e autofavorecimento.',
  ],
  quiz: [
    { q: 'Por que a literatura de RLHF prefere coletar comparações em vez de notas absolutas?', o: ['Porque comparações carregam mais informação por anotação do que uma nota de 1 a 7', 'Porque cada anotador tem uma régua pessoal (deslocamento e escala próprios), e a comparação cancela essa régua; o preço é perder a magnitude da preferência', 'Porque notas absolutas não podem ser usadas para treinar redes neurais', 'Porque comparações dispensam o controle de qualidade dos anotadores'], a: 1, e: 'Christiano 2017 e Ziegler 2019 relatam que comparações consistentes são muito mais fáceis de obter. A nota mistura qualidade e régua pessoal; a diferença entre duas respostas lidas na mesma régua não depende do deslocamento. O HelpSteer mostra o outro lado: notas escalam melhor (comparações crescem de forma quadrática) e guardam magnitude, por isso datasets modernos combinam os dois.' },
    { q: 'Dois anotadores concordam em 75 de 100 pares. As marginais dão \\(p_e = 0{,}50\\). Qual é o κ de Cohen?', o: ['0,75', '0,25', '0,50', '0,375'], a: 2, e: '\\(\\kappa = (p_o - p_e)/(1-p_e) = (0{,}75-0{,}50)/0{,}50 = 0{,}50\\). Metade da concordância bruta é o que o acaso já produziria com essas marginais.' },
    { q: 'Para um par com \\(\\Delta = r_w - r_l = 3\\), quanto vale o gradiente \\(\\partial\\mathcal L/\\partial\\Delta\\) da perda \\(-\\log\\sigma(\\Delta)\\), e o que isso significa?', o: ['−0,9526: pares bem separados dominam o treino', '−0,0474: o par já está bem separado e quase não contribui; o peso de cada par é a chance de o RM estar errado', '0: a perda é zero quando \\(\\Delta &gt; 0\\)', '−3: o gradiente cresce com a separação'], a: 1, e: '\\(\\partial\\mathcal L/\\partial\\Delta = -\\sigma(-\\Delta) = -\\sigma(-3) \\approx -0{,}0474\\). Um par invertido (\\(\\Delta = -0{,}8\\)) pesaria 0,69. O treino concentra o esforço onde o RM erra, o que também significa que um rótulo errado continua puxando com força.' },
    { q: 'O InstructGPT pedia rankings de \\(K = 9\\) respostas. Quantos pares cada ranking gera, e por que os pares de um prompt iam no mesmo elemento do batch?', o: ['9 pares; para economizar memória', '36 pares; porque embaralhados eles fazem cada resposta participar de \\(K-1\\) atualizações separadas e o RM sobreajusta numa passada; agrupados, bastam 9 forwards', '72 pares; para usar Plackett–Luce', '36 pares; porque o otimizador Adam exige isso'], a: 1, e: '\\(\\binom{9}{2} = 36\\). Pares do mesmo prompt são muito correlacionados; agrupá-los evita o sobreajuste e reduz os forwards de \\(2\\binom{K}{2} = 72\\) para \\(K = 9\\). Um batch de 64 prompts carrega até 2.304 comparações.' },
    { q: 'Num best-of-\\(n\\) contra um RM proxy, o escore proxy da resposta escolhida sobe sempre com \\(n\\), mas o escore ouro sobe e depois cai. Qual mecanismo explica a queda, e o que um ensemble conservador de RMs consegue corrigir?', o: ['Ruído de amostragem; o ensemble corrige tudo', 'Goodhart extremal: fora da distribuição (ex.: respostas longuíssimas) proxy e ouro se descolam; o ensemble corrige erros próprios de cada membro (inclusive os raros e enormes), mas não um viés que todos compartilham', 'O KL do best-of-\\(n\\) fica negativo para \\(n\\) grande', 'O RM fica descalibrado; recalibrar por temperatura resolve'], a: 1, e: 'Com ruído gaussiano, o Goodhart regressional só desperdiça otimização em ruído; a queda vem do extremal (ou de erros raros e enormes de um RM, que um ensemble corrige). Coste et al. mostram que a otimização conservadora (WCO/UWO) praticamente elimina a superotimização no best-of-\\(n\\); Eisenstein et al. mostram que nem ensembles de pré-treino eliminam o hacking, porque os membros erram de forma parecida. Recalibrar por temperatura não muda a ordem e, portanto, não muda o best-of-\\(n\\).' },
    { q: 'O que o rótulo SE do Math-Shepherd estima, e por que o DeepSeek-R1 evitou PRMs neurais no RL de raciocínio?', o: ['A probabilidade de o passo estar escrito corretamente; porque PRMs são lentos', 'A fração de continuações a partir do passo que chegam à resposta certa, uma estimativa de Monte Carlo do valor do passo; porque “passo” é difícil de definir, anotar é caro e ruidoso, e PRMs baseados em modelo levam a reward hacking', 'O número de passos da solução; porque o R1 não usa cadeia de pensamento', 'A nota de um juiz LLM para o passo; porque juízes têm viés de posição'], a: 1, e: 'SE \\(= \\frac1N\\sum_j \\mathbb 1[a_j = a^*]\\) é uma estimativa MC de \\(V(s_i)\\) com recompensa binária. O relatório do R1 lista os três motivos e reconhece que PRMs servem bem para reordenar as N melhores respostas; para RL de raciocínio, o R1 usou recompensas verificáveis.' },
  ],
  mount(root, LA) {
    const V = LA.viz, UI = LA.ui;
    const fmt = (x, d = 2) => { const s = (+x).toFixed(d).replace('.', ','); return s.startsWith('-') ? '−' + s.slice(1) : s; };
    const pct = (x, d = 1) => fmt(100 * x, d) + '%';
    const sig = (z) => 1 / (1 + Math.exp(-z));
    const logsig = (z) => (z >= 0 ? -Math.log1p(Math.exp(-z)) : z - Math.log1p(Math.exp(z)));
    const RED = () => V.C().red || '#FB2C36';
    const nocase = (c) => { const el = c && (c.el || c); if (el && el.querySelectorAll) el.querySelectorAll('label, span').forEach((n) => { if (!n.classList.contains('val')) n.style.textTransform = 'none'; }); return c; };

    /* ------------------------------------------------ helpers comuns */
    function adaptH(fig, wide, narrow, bp = 600) {
      const st = fig.querySelector('.la-fig-stage');
      const val = (v) => (typeof v === 'function' ? v() : v);
      const set = () => { const want = val(fig.clientWidth < bp ? narrow : wide) + 'px'; if (st.style.height !== want) st.style.height = want; };
      set();
      const ro = new ResizeObserver(set); ro.observe(fig);
      if (LA.onCleanup) LA.onCleanup(() => ro.disconnect());
      return set;
    }
    function tw(g, s, size, font = 'sans', weight = 400) { g.font = V.font(size, font, weight); return g.measureText(String(s)).width; }
    /** personagem: quadrado arredondado com olhinhos e boca */
    function face(g, x, y, s, col, o = {}) {
      const C = V.C();
      if (o.alpha != null) g.globalAlpha = o.alpha;
      V.box(g, x - s / 2, y - s / 2, s, s, { r: s * 0.3, fill: o.fill || V.alpha(col, 0.85), stroke: col, lw: 1.2 });
      if (s >= 11) {
        const eye = C.dark ? '#06140d' : '#ffffff';
        const ex = s * 0.17, ey = y - s * 0.08, er = Math.max(1, s * 0.07);
        V.dot(g, x - ex + (o.look || 0) * s * 0.05, ey, er, eye); V.dot(g, x + ex + (o.look || 0) * s * 0.05, ey, er, eye);
        g.beginPath();
        if (o.mood === 'sad') g.arc(x, y + s * 0.3, s * 0.14, Math.PI * 1.15, Math.PI * 1.85);
        else if (o.mood === 'o') g.arc(x, y + s * 0.16, s * 0.07, 0, Math.PI * 2);
        else if (o.mood === 'flat') { g.moveTo(x - s * 0.13, y + s * 0.18); g.lineTo(x + s * 0.13, y + s * 0.18); }
        else g.arc(x, y + s * 0.06, s * 0.15, Math.PI * 0.2, Math.PI * 0.8);
        g.strokeStyle = eye; g.lineWidth = Math.max(1, s * 0.06); g.stroke();
      }
      if (o.label) V.text(g, o.label, x, y + s / 2 + 12, { size: 11, font: 'mono', align: 'center', color: o.labelColor || C.ink, weight: 600 });
      if (o.alpha != null) g.globalAlpha = 1;
    }
    function chip(g, x, y, w, h, label, o = {}) {
      const C = V.C();
      V.box(g, x, y, w, h, { r: o.r == null ? 6 : o.r, fill: o.fill || V.alpha(C.muted, 0.12), stroke: o.stroke || V.alpha(C.muted, 0.4), lw: o.lw || 1.2 });
      if (label != null) V.text(g, label, x + w / 2, y + h / 2 + 0.5, { size: o.size || 11.5, font: o.font || 'mono', weight: o.weight || 500, color: o.color || C.ink, align: 'center', base: 'middle', maxW: w - 4 });
    }
    function mark(g, x, y, ok, s = 6) {
      const C = V.C();
      g.lineWidth = 2.2; g.strokeStyle = ok ? C.a : RED(); g.lineCap = 'round'; g.beginPath();
      if (ok) { g.moveTo(x - s, y); g.lineTo(x - s * 0.3, y + s * 0.7); g.lineTo(x + s, y - s * 0.8); }
      else { g.moveTo(x - s * 0.8, y - s * 0.8); g.lineTo(x + s * 0.8, y + s * 0.8); g.moveTo(x + s * 0.8, y - s * 0.8); g.lineTo(x - s * 0.8, y + s * 0.8); }
      g.stroke(); g.lineCap = 'butt';
    }
    function crown(g, x, y, s, col) {
      g.beginPath();
      g.moveTo(x - s, y); g.lineTo(x - s, y - s * 0.7); g.lineTo(x - s * 0.5, y - s * 0.25); g.lineTo(x, y - s * 0.9); g.lineTo(x + s * 0.5, y - s * 0.25); g.lineTo(x + s, y - s * 0.7); g.lineTo(x + s, y);
      g.closePath(); g.fillStyle = col; g.fill();
    }
    /** eixos de um gráfico; devolve os mapeamentos X(v), Y(v) */
    function axes(g, R, o) {
      const C = V.C();
      const X = (v) => R.x + ((v - o.x0) / (o.x1 - o.x0)) * R.w;
      const Y = (v) => R.y + R.h - ((v - o.y0) / (o.y1 - o.y0)) * R.h;
      g.lineWidth = 1;
      (o.yt || []).forEach((v) => { const y = Y(v); V.line(g, R.x, y, R.x + R.w, y, { color: C.grid, lw: 1 }); V.text(g, o.fy ? o.fy(v) : fmt(v, 1), R.x - 5, y, { size: 10.5, font: 'mono', align: 'right', base: 'middle', color: C.muted }); });
      (o.xt || []).forEach((v) => { const x = X(v); V.line(g, x, R.y, x, R.y + R.h, { color: C.grid, lw: 1 }); V.text(g, o.fx ? o.fx(v) : fmt(v, 0), x, R.y + R.h + 13, { size: 10.5, font: 'mono', align: 'center', color: C.muted }); });
      V.line(g, R.x, R.y + R.h, R.x + R.w, R.y + R.h, { color: C.muted, lw: 1.2 });
      V.line(g, R.x, R.y, R.x, R.y + R.h, { color: C.muted, lw: 1.2 });
      if (o.xl) V.text(g, o.xl, R.x + R.w, R.y + R.h + 27, { size: 11, align: 'right', color: C.muted });
      if (o.yl) V.text(g, o.yl, R.x + 4, R.y - 7, { size: 11, align: 'left', color: C.muted });
      return { X, Y };
    }
    function curve(g, f, x0, x1, X, Y, o = {}) {
      g.beginPath(); const n = o.n || 120;
      for (let i = 0; i <= n; i++) { const v = x0 + ((x1 - x0) * i) / n; const yv = f(v); const px = X(v), py = Y(Math.max(o.ymin == null ? -1e9 : o.ymin, Math.min(o.ymax == null ? 1e9 : o.ymax, yv))); if (i) g.lineTo(px, py); else g.moveTo(px, py); }
      g.strokeStyle = o.color; g.lineWidth = o.lw || 2; if (o.dash) g.setLineDash(o.dash); g.stroke(); g.setLineDash([]);
    }
    function hbar(g, x, y, w, h, frac, col, label, valTxt) {
      const C = V.C();
      V.box(g, x, y, w, h, { r: h / 2, fill: V.alpha(C.muted, 0.12) });
      if (frac > 0.002) V.box(g, x, y, Math.max(h, w * Math.min(1, frac)), h, { r: h / 2, fill: col });
      if (label) V.text(g, label, x, y - 5, { size: 11, color: C.muted });
      if (valTxt) V.text(g, valTxt, x + w, y - 5, { size: 11.5, font: 'mono', align: 'right', color: C.ink, weight: 600 });
    }

    /* ================================================ FIG 1 — régua pessoal vs comparação */
    (function () {
      const fig = root.querySelector('#fig-regua');
      adaptH(fig, 430, 480);
      const O = [-1.3, -0.6, 0, 0.7, 1.25], SC = [-0.45, 0.25, 0, 0.35, -0.25];
      const S = { spread: 0.6, mode: 'nota', item: null, prev: null, t0: 0, stats: { score: 0, dir: 0 } };
      const rr = V.rng(53);
      const ann = (k) => ({ o: S.spread * O[k] * 1.2, s: 1 + S.spread * SC[k] * 1.4 });
      const NOISE = 0.32, GAIN = 1.45;
      function judge(q, k, n) { const a = ann(k); return 4 + GAIN * a.s * q + a.o + NOISE * n; }
      const lik = (u) => Math.max(1, Math.min(7, Math.round(u)));
      function newItem(r) {
        const qA = r.normal() * 0.7, qB = r.normal() * 0.7;
        const nA = [0, 1, 2, 3, 4].map(() => r.normal()), nB = [0, 1, 2, 3, 4].map(() => r.normal());
        return { qA, qB, nA, nB };
      }
      function evalItem(it) {
        const uA = [], uB = [];
        for (let k = 0; k < 5; k++) { uA.push(judge(it.qA, k, it.nA[k])); uB.push(judge(it.qB, k, it.nB[k])); }
        const LA_ = uA.map(lik), LB = uB.map(lik), pref = uA.map((u, k) => u > uB[k]);
        return { uA, uB, LA: LA_, LB, pref };
      }
      function computeStats() {
        const r = V.rng(991);
        let sameScore = 0, nScore = 0, sameDir = 0, nDir = 0;
        for (let t = 0; t < 700; t++) {
          const e = evalItem(newItem(r));
          for (let i = 0; i < 5; i++) for (let j = i + 1; j < 5; j++) {
            sameScore += (e.LA[i] === e.LA[j]) + (e.LB[i] === e.LB[j]); nScore += 2;
            sameDir += e.pref[i] === e.pref[j]; nDir++;
          }
        }
        S.stats = { score: sameScore / nScore, dir: sameDir / nDir };
        rS.set(pct(S.stats.score, 0)); rD.set(pct(S.stats.dir, 0));
      }
      const ctl = UI.controls(fig);
      nocase(UI.seg(ctl, { label: 'Pedir', options: [{ v: 'nota', label: 'nota Likert 1–7' }, { v: 'comp', label: 'comparação A vs B' }], value: 'nota', onChange: (v) => { S.mode = v; } }));
      nocase(UI.slider(ctl, { label: 'diferença entre as réguas', min: 0, max: 1, step: 0.05, value: S.spread, fmt: (v) => fmt(v, 2), onInput: (v) => { S.spread = v; computeStats(); } }));
      const rS = UI.readout(ctl, 'mesma nota', '—'), rD = UI.readout(ctl, 'mesma direção', '—');
      computeStats();
      S.item = newItem(rr); S.prev = S.item; S.t0 = 0;

      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        if (t - S.t0 > 2.8) { S.prev = S.item; S.item = newItem(rr); S.t0 = t; }
        const k = V.ease.inOut(V.clamp((t - S.t0) / 0.7, 0, 1));
        const lerpIt = (a, b) => ({ qA: V.lerp(a.qA, b.qA, k), qB: V.lerp(a.qB, b.qB, k), nA: a.nA.map((x, i) => V.lerp(x, b.nA[i], k)), nB: a.nB.map((x, i) => V.lerp(x, b.nB[i], k)) });
        const it = lerpIt(S.prev, S.item);
        const e = evalItem(it);
        const narrow = w < 560;
        const top = 34, barsH = narrow ? 84 : 64, bottom = h - barsH - 8;
        const plotTop = top + 30, plotBot = bottom - (S.mode === 'comp' ? 46 : 18);
        const qmin = -2.4, qmax = 2.4;
        const Yq = (q) => plotBot - ((q - qmin) / (qmax - qmin)) * (plotBot - plotTop);
        const left = narrow ? 22 : 64, right = w - (narrow ? 10 : 30);
        const colW = (right - left) / 5;
        g.save(); g.translate(narrow ? 9 : 16, (plotTop + plotBot) / 2); g.rotate(-Math.PI / 2);
        V.text(g, 'qualidade real (eixo comum) →', 0, 0, { size: 11, color: C.muted, align: 'center', base: 'middle' });
        g.restore();
        // linhas das respostas
        const yA = Yq(it.qA), yB = Yq(it.qB);
        V.line(g, left - 6, yA, right, yA, { color: V.alpha(C.a, 0.8), lw: 1.6, dash: [6, 5] });
        V.line(g, left - 6, yB, right, yB, { color: V.alpha(C.c, 0.8), lw: 1.6, dash: [6, 5] });
        if (!narrow) { V.text(g, 'A', left - 14, yA, { size: 12, weight: 700, color: C.a, align: 'right', base: 'middle', font: 'mono' }); V.text(g, 'B', left - 14, yB, { size: 12, weight: 700, color: C.c, align: 'right', base: 'middle', font: 'mono' }); }
        for (let kk = 0; kk < 5; kk++) {
          const a = ann(kk);
          const cx = left + colW * (kk + 0.5);
          const qOf = (L) => (L - 4 - a.o) / (GAIN * a.s);
          // régua
          const y1 = Yq(Math.max(qmin, qOf(0.5))), y7 = Yq(Math.min(qmax, qOf(7.5)));
          const rw = Math.min(16, colW * 0.22);
          V.box(g, cx - rw / 2, Math.min(y1, y7), rw, Math.abs(y1 - y7), { r: 4, fill: V.alpha(C.d, 0.14), stroke: V.alpha(C.d, 0.55), lw: 1 });
          for (let L = 1; L <= 7; L++) {
            const q = qOf(L); if (q < qmin || q > qmax) continue;
            const y = Yq(q);
            V.line(g, cx - rw / 2, y, cx + rw / 2, y, { color: V.alpha(C.d, 0.9), lw: 1.2 });
            V.text(g, String(L), cx - rw / 2 - 4, y, { size: 9.5, font: 'mono', align: 'right', base: 'middle', color: C.muted });
          }
          // leituras
          const off = Math.min(26, colW * 0.3);
          const rdA = Yq((e.uA[kk] - 4 - a.o) / (GAIN * a.s)), rdB = Yq((e.uB[kk] - 4 - a.o) / (GAIN * a.s));
          if (S.mode === 'nota') {
            V.dot(g, cx + rw / 2 + 3, V.clamp(rdA, plotTop, plotBot), 3.5, C.a);
            V.dot(g, cx + rw / 2 + 3, V.clamp(rdB, plotTop, plotBot), 3.5, C.c);
            chip(g, cx + rw / 2 + 8, V.clamp(yA, plotTop, plotBot) - 9, Math.min(off, 22), 18, String(e.LA[kk]), { fill: V.alpha(C.a, 0.22), stroke: C.a, size: 11.5, weight: 700 });
            chip(g, cx + rw / 2 + 8, V.clamp(yB, plotTop, plotBot) - 9, Math.min(off, 22), 18, String(e.LB[kk]), { fill: V.alpha(C.c, 0.22), stroke: C.c, size: 11.5, weight: 700 });
          } else {
            const by = plotBot + 26, prefA = e.pref[kk];
            const tilt = (prefA ? -1 : 1) * 0.28 * k;
            const bw = Math.min(46, colW * 0.75);
            g.save(); g.translate(cx, by); g.rotate(tilt);
            V.line(g, -bw / 2, 0, bw / 2, 0, { color: C.ink, lw: 2 });
            V.box(g, -bw / 2 - 8, -2, 16, 10, { r: 3, fill: V.alpha(C.a, 0.85) });
            V.box(g, bw / 2 - 8, -2, 16, 10, { r: 3, fill: V.alpha(C.c, 0.85) });
            V.text(g, 'A', -bw / 2, 3, { size: 9, font: 'mono', align: 'center', base: 'middle', color: C.dark ? '#00210F' : '#fff', weight: 700 });
            V.text(g, 'B', bw / 2, 3, { size: 9, font: 'mono', align: 'center', base: 'middle', color: C.dark ? '#00210F' : '#fff', weight: 700 });
            g.restore();
            V.line(g, cx, by, cx, by + 12, { color: C.ink, lw: 2 });
            V.text(g, prefA ? 'A ≻ B' : 'B ≻ A', cx, by + 24, { size: 10.5, font: 'mono', align: 'center', color: prefA ? C.a : C.c, weight: 600 });
          }
          const mood = a.o > 0.4 ? 'happy' : a.o < -0.4 ? 'flat' : 'o';
          face(g, cx, top + 8, 20, C.series[kk % 6] || C.b, { mood: mood === 'happy' ? undefined : mood });
          if (!narrow) V.text(g, a.o > 0.4 ? 'generoso' : a.o < -0.4 ? 'rigoroso' : 'neutro', cx, top - 12, { size: 10, align: 'center', color: C.muted });
        }
        // barras de concordância
        const bx = 12, bwid = w - 24, by0 = h - barsH + (narrow ? 14 : 12);
        if (narrow) {
          hbar(g, bx, by0, bwid, 10, S.stats.score, C.d, 'mesma nota (pares de anotadores)', pct(S.stats.score, 0));
          hbar(g, bx, by0 + 36, bwid, 10, S.stats.dir, C.a, 'mesma direção (comparação)', pct(S.stats.dir, 0));
        } else {
          const half = (bwid - 24) / 2;
          hbar(g, bx, by0 + 12, half, 12, S.stats.score, C.d, 'concordância: mesma nota Likert', pct(S.stats.score, 0));
          hbar(g, bx + half + 24, by0 + 12, half, 12, S.stats.dir, C.a, 'concordância: mesma direção', pct(S.stats.dir, 0));
        }
      });
    })();

    /* ================================================ FIG 2 — κ de Cohen */
    (function () {
      const fig = root.querySelector('#fig-kappa');
      adaptH(fig, 420, 596);
      const S = { mode: 'ex', po: 0.8, m: 0.5, weighted: true, t0: 0, items: [] };
      const EX = [[40, 10], [15, 35]];
      const ORD = [[20, 5, 0], [4, 30, 6], [1, 4, 30]];
      function explore() {
        const N = 100;
        let off = Math.round(((1 - S.po) * N) / 2);
        const mA = Math.round(S.m * N);
        off = Math.min(off, N - mA, mA);
        const aa = mA - off, bb = N - aa - 2 * off;
        return [[aa, off], [off, bb]];
      }
      const data = () => (S.mode === 'ex' ? EX : S.mode === 'ord' ? ORD : explore());
      function stats(M) {
        const k = M.length, N = M.flat().reduce((a, b) => a + b, 0);
        const r = M.map((row) => row.reduce((a, b) => a + b, 0) / N), c = M[0].map((_, j) => M.reduce((s, row) => s + row[j], 0) / N);
        let po = 0, pe = 0, wo = 0, we = 0;
        for (let i = 0; i < k; i++) for (let j = 0; j < k; j++) {
          const o = M[i][j] / N, e = r[i] * c[j], wgt = ((i - j) * (i - j)) / ((k - 1) * (k - 1));
          if (i === j) { po += o; pe += e; }
          wo += wgt * o; we += wgt * e;
        }
        const kappa = (po - pe) / (1 - pe), kw = 1 - wo / we;
        return { N, r, c, po, pe, kappa, kw, wo, we, E: M.map((row, i) => row.map((_, j) => r[i] * c[j] * N)) };
      }
      function rebuild(t) {
        const M = data(); S.items = []; let idx = 0;
        const rng = V.rng(7 + S.mode.length);
        M.forEach((row, i) => row.forEach((n, j) => { for (let s = 0; s < n; s++) S.items.push({ i, j, s, d: rng() * 0.25 + (idx++ % 100) * 0.004 }); }));
        S.t0 = t == null ? (H ? H.t : 0) : t;
        upd();
      }
      const ctl = UI.controls(fig);
      nocase(UI.seg(ctl, { label: 'Caso', options: [{ v: 'ex', label: 'binário (exemplo)' }, { v: 'exp', label: 'binário: explorar' }, { v: 'ord', label: 'ordinal 3 níveis' }], value: 'ex', onChange: (v) => { S.mode = v; gExp.show(v === 'exp'); gOrd.show(v === 'ord'); rebuild(); } }));
      const gExp = UI.group(ctl), gOrd = UI.group(ctl);
      nocase(UI.slider(gExp.el, { label: 'pₒ pedido', min: 0.5, max: 1, step: 0.05, value: S.po, fmt: (v) => fmt(v, 2), onInput: (v) => { S.po = v; rebuild(); } }));
      nocase(UI.slider(gExp.el, { label: 'fração “A melhor”', min: 0.5, max: 0.95, step: 0.05, value: S.m, fmt: (v) => fmt(v, 2), onInput: (v) => { S.m = v; rebuild(); } }));
      nocase(UI.toggle(gOrd.el, { label: 'ponderação quadrática', value: true, onChange: (v) => { S.weighted = v; upd(); } }));
      gExp.show(false); gOrd.show(false);
      const rPo = UI.readout(ctl, 'pₒ', '—'), rPe = UI.readout(ctl, 'pₑ', '—'), rK = UI.readout(ctl, 'κ', '—');
      nocase(ctl);
      function upd() {
        const st = stats(data());
        rPo.set(fmt(st.po, 2)); rPe.set(fmt(st.pe, 4));
        if (S.mode === 'ord' && S.weighted) rK.set('κ_w = ' + fmt(st.kw, 4), 'ok');
        else rK.set(fmt(st.kappa, 4), st.kappa < 0.4 ? 'warn' : 'ok');
      }
      let H = null;
      H = V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const M = data(), st = stats(M), k = M.length;
        const narrow = w < 600;
        const lab = k === 2 ? ['A', 'B'] : ['0', '1', '2'];
        const ord = S.mode === 'ord', wgt = ord && S.weighted;
        const matSize = narrow ? Math.min(w - 90, 180) : Math.min(260, (w - 200) / 2, h - (S.mode === 'ord' ? 100 : 82) - 46 - 62);
        const cell = (matSize - (k - 1) * 4) / k;
        const mx1 = narrow ? 52 : Math.max(60, w / 2 - matSize - 50), my1 = 46;
        const mx2 = narrow ? 52 : w / 2 + 50, my2 = narrow ? my1 + matSize + 62 : my1;
        // no celular, as barras vêm logo abaixo da matriz esperada (sem faixa vazia)
        const byTop = narrow ? my2 + matSize + 22 : h - (S.mode === 'ord' ? 100 : 82);
        // títulos
        V.text(g, 'observado (contagens)', mx1 + matSize / 2, my1 - 26, { size: 12, align: 'center', color: C.ink, weight: 600 });
        V.text(g, 'esperado por acaso  Eᵢⱼ = rᵢ·cⱼ·N', mx2 + matSize / 2, my2 - (narrow ? 14 : 26), { size: 12, align: 'center', color: C.ink, weight: 600 });
        // matriz observada: fundo das células
        const cellXY = (x0, y0, i, j) => [x0 + j * (cell + 4), y0 + i * (cell + 4)];
        for (let i = 0; i < k; i++) for (let j = 0; j < k; j++) {
          const [x, y] = cellXY(mx1, my1, i, j);
          const wij = ((i - j) * (i - j)) / ((k - 1) * (k - 1));
          const base = i === j ? V.alpha(C.a, 0.12) : wgt ? V.alpha(C.c, 0.06 + 0.22 * wij) : V.alpha(C.c, 0.1);
          V.box(g, x, y, cell, cell, { r: 6, fill: base, stroke: V.alpha(i === j ? C.a : C.c, 0.45), lw: 1 });
          V.text(g, String(M[i][j]), x + cell - 6, y + 13, { size: 12, font: 'mono', align: 'right', color: C.ink, weight: 700 });
          if (wgt && i !== j) V.text(g, 'w=' + fmt(wij, 2), x + 5, y + cell - 6, { size: 9.5, font: 'mono', color: C.c });
        }
        // itens caindo
        const el = t - S.t0;
        const cols = Math.max(4, Math.floor((cell - 10) / 7));
        S.items.forEach((it) => {
          const [x, y] = cellXY(mx1, my1, it.i, it.j);
          const tx = x + 6 + (it.s % cols) * 7 + 3, ty = y + 22 + Math.floor(it.s / cols) * 7 + 3;
          const p = V.ease.out(V.clamp((el - it.d) / 0.4, 0, 1));
          if (p <= 0) return;
          const sy = my1 - 30;
          const yy = V.lerp(sy, ty, p), xx = V.lerp(mx1 + matSize / 2, tx, p);
          if (ty > y + cell - 3) return;
          V.dot(g, xx, yy, 2.4, it.i === it.j ? C.a : C.c);
        });
        // rótulos das linhas/colunas
        for (let i = 0; i < k; i++) {
          const [x, y] = cellXY(mx1, my1, i, 0);
          V.text(g, lab[i], mx1 - 8, y + cell / 2, { size: 11.5, font: 'mono', align: 'right', base: 'middle', color: C.muted, weight: 600 });
          const [x2, y2] = cellXY(mx1, my1, 0, i);
          V.text(g, lab[i], x2 + cell / 2, my1 - 8, { size: 11.5, font: 'mono', align: 'center', color: C.muted, weight: 600 });
          // marginais (barras finas: comprimento ∝ fração)
          const mlen = narrow ? 30 : 40;
          V.box(g, mx1 + matSize + 8, y + cell / 2 - 4, mlen, 8, { r: 4, fill: V.alpha(C.muted, 0.15) });
          V.box(g, mx1 + matSize + 8, y + cell / 2 - 4, Math.max(4, mlen * st.r[i]), 8, { r: 4, fill: V.alpha(C.b, 0.85) });
          V.text(g, fmt(st.r[i], 2), mx1 + matSize + 8, y + cell / 2 + 16, { size: 9.5, font: 'mono', color: C.muted });
          V.box(g, x2, my1 + matSize + 8, cell, 8, { r: 4, fill: V.alpha(C.muted, 0.15) });
          V.box(g, x2, my1 + matSize + 8, Math.max(4, cell * st.c[i]), 8, { r: 4, fill: V.alpha(C.b, 0.85) });
          V.text(g, fmt(st.c[i], 2), x2 + cell / 2, my1 + matSize + 30, { size: 9.5, font: 'mono', align: 'center', color: C.muted });
        }
        V.text(g, 'A↓ B→', mx1 - 18, my1 - 8, { size: 10, font: 'mono', color: C.muted, align: 'right' });
        // matriz esperada
        for (let i = 0; i < k; i++) for (let j = 0; j < k; j++) {
          const [x, y] = cellXY(mx2, my2, i, j);
          const e = st.E[i][j];
          V.box(g, x, y, cell, cell, { r: 6, fill: V.alpha(i === j ? C.d : C.muted, 0.08 + 0.4 * (e / st.N)), stroke: V.alpha(i === j ? C.d : C.muted, 0.5), lw: 1 });
          V.text(g, fmt(e, 1), x + cell / 2, y + cell / 2, { size: Math.min(14, cell * 0.22), font: 'mono', align: 'center', base: 'middle', color: C.ink, weight: i === j ? 700 : 400 });
        }
        // barras p_o, p_e, κ
        const bx = narrow ? 16 : 40, bw = narrow ? w - 32 : w - 80;
        const by = byTop + 14;
        const kv = wgt ? st.kw : st.kappa;
        const gap = narrow ? 30 : 28;
        hbar(g, bx, by, bw, 9, st.po, C.a, 'pₒ  concordância observada (diagonal)', fmt(st.po, 2));
        hbar(g, bx, by + gap, bw, 9, st.pe, C.d, 'pₑ  concordância esperada por acaso', fmt(st.pe, 4));
        hbar(g, bx, by + 2 * gap, bw, 9, Math.max(0, kv), C.b, wgt ? 'κ_w = 1 − Σw·O / Σw·E' : 'κ = (pₒ − pₑ)/(1 − pₑ)', fmt(kv, 4));
        if (ord) V.text(g, wgt ? 'sem ponderar: κ = ' + fmt(st.kappa, 4) : 'ponderado: κ_w = ' + fmt(st.kw, 4), bx + bw, by + 2 * gap + 24, { size: 11, font: 'mono', align: 'right', color: C.muted });
      });
      rebuild(-5);
    })();

    /* ================================================ FIG 3 — duelos de Gumbel */
    (function () {
      const fig = root.querySelector('#fig-gumbel');
      adaptH(fig, 420, 630);
      const NB = 48, Z0 = -6, Z1 = 6, SN = Math.PI / Math.sqrt(6);
      const S = { delta: 0.8, noise: 'gumbel', hist: new Array(NB).fill(0), n: 0, winsA: 0, last: null, acc: 0 };
      const rng = V.rng(1952);
      const gumbel = () => -Math.log(-Math.log(Math.max(1e-12, rng())));
      const draw = () => (S.noise === 'gumbel' ? gumbel() : rng.normal() * SN);
      const erf = (x) => { const s = Math.sign(x); x = Math.abs(x); const t = 1 / (1 + 0.3275911 * x); const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x); return s * y; };
      const Phi = (x) => 0.5 * (1 + erf(x / Math.SQRT2));
      const theory = () => (S.noise === 'gumbel' ? sig(S.delta) : Phi(S.delta / (Math.SQRT2 * SN)));
      const pdf = (z) => (S.noise === 'gumbel' ? Math.exp(-z) / Math.pow(1 + Math.exp(-z), 2) : Math.exp(-(z * z) / (4 * SN * SN)) / Math.sqrt(4 * Math.PI * SN * SN));
      function reset() { S.hist.fill(0); S.n = 0; S.winsA = 0; S.last = null; for (let i = 0; i < 150; i++) duel(); upd(); }
      function duel() {
        const eA = draw(), eB = draw(), d = eB - eA;
        const rA = S.delta / 2, rB = -S.delta / 2;
        const win = rA + eA > rB + eB;
        S.n++; if (win) S.winsA++;
        const b = Math.floor(((d - Z0) / (Z1 - Z0)) * NB); if (b >= 0 && b < NB) S.hist[b]++;
        S.last = { eA, eB, uA: rA + eA, uB: rB + eB, win, d };
      }
      const ctl = UI.controls(fig);
      nocase(UI.slider(ctl, { label: 'Δ = r_A − r_B', min: -3, max: 3, step: 0.1, value: S.delta, fmt: (v) => fmt(v, 1), onInput: (v) => { S.delta = v; reset(); } }));
      nocase(UI.seg(ctl, { label: 'Ruído', options: [{ v: 'gumbel', label: 'Gumbel → logística' }, { v: 'normal', label: 'Normal → Thurstone' }], value: 'gumbel', onChange: (v) => { S.noise = v; reset(); } }));
      UI.button(ctl, '↺ zerar', reset);
      const rE = UI.readout(ctl, 'A venceu', '—'), rT = UI.readout(ctl, 'teoria', '—'), rElo = UI.readout(ctl, 'Elo equivalente', '—');
      nocase(ctl);
      function upd() {
        rE.set(S.n ? pct(S.winsA / S.n, 1) + ' de ' + S.n : '—');
        rT.set((S.noise === 'gumbel' ? 'σ(Δ) = ' : 'Φ = ') + pct(theory(), 2));
        rElo.set((S.delta >= 0 ? '+' : '−') + Math.round(Math.abs(S.delta) * 400 / Math.LN10) + ' Elo');
      }
      reset();
      V.fig(fig, (g, w, h, t, dt) => {
        const C = V.C();
        S.acc += dt;
        let fired = false;
        while (S.acc > 0.03) { S.acc -= 0.03; if (S.n < 20000) duel(); fired = true; }
        if (fired) upd();
        const narrow = w < 600;
        const L = narrow ? { x: 0, y: 0, w: w, h: 230 } : { x: 0, y: 0, w: w * 0.38, h: h - 60 };
        const R = narrow ? { x: 46, y: 266, w: w - 64, h: h - 266 - 92 } : { x: w * 0.38 + 50, y: 40, w: w * 0.62 - 76, h: h - 150 };
        // arena (no celular, escala maior: o eixo de utilidade vai de −3 a 4 em vez de −3 a 6)
        const base = L.y + L.h - 34, colA = L.x + L.w * 0.3, colB = L.x + L.w * 0.7;
        const uTop = narrow ? 4 : 6;
        const sc = (L.h - 90) / (uTop + 3);
        const yOf = (u) => base - (u + 3) * sc;
        V.line(g, L.x + 16, yOf(0), L.x + L.w - 16, yOf(0), { color: C.grid, lw: 1, dash: [3, 4] });
        V.text(g, 'utilidade u = r + ε', L.x + 16, 22, { size: 11.5, color: C.muted });
        const lt = S.last;
        [[colA, S.delta / 2, lt ? lt.uA : S.delta / 2, C.a, 'A'], [colB, -S.delta / 2, lt ? lt.uB : -S.delta / 2, C.c, 'B']].forEach(([x, r, u, col, nm], idx) => {
          const bw = Math.min(46, L.w * 0.16);
          const yr = yOf(r), yu = yOf(V.clamp(u, -3, uTop));
          V.box(g, x - bw / 2, Math.min(yr, yOf(-3)), bw, Math.abs(yOf(-3) - yr), { r: 5, fill: V.alpha(col, 0.28), stroke: V.alpha(col, 0.6) });
          V.line(g, x - bw / 2 - 6, yr, x + bw / 2 + 6, yr, { color: col, lw: 2 });
          V.text(g, 'r', x + bw / 2 + 8, yr, { size: 11, font: 'mono', color: col, base: 'middle' });
          V.arrow(g, x, yr, x, yu, { color: C.d, lw: 2, head: 7 });
          V.dot(g, x, yu, 4, C.d);
          const won = lt && ((idx === 0 && lt.win) || (idx === 1 && !lt.win));
          face(g, x, base + 16, 22, col, { mood: won ? undefined : 'flat' });
          if (won) crown(g, x, base + 1, 8, C.d);
          V.text(g, nm, x - 18, base + 16, { size: 12, weight: 700, font: 'mono', color: col, align: 'right', base: 'middle' });
        });
        V.text(g, 'ε sorteado (âmbar)', L.x + L.w / 2, 40, { size: 10.5, color: C.d, align: 'center' });
        // histograma
        const mx = Math.max(1, ...S.hist);
        const binW = R.w / NB;
        const total = Math.max(1, S.n);
        const yMax = Math.max(0.3, (mx / total) / ((Z1 - Z0) / NB));
        const A = axes(g, R, { x0: Z0, x1: Z1, y0: 0, y1: yMax * 1.08, xt: [-6, -3, 0, 3, 6], yt: [], xl: 'ε_B − ε_A', yl: 'densidade' });
        for (let b = 0; b < NB; b++) {
          const zc = Z0 + (b + 0.5) * (Z1 - Z0) / NB;
          const dens = S.hist[b] / total / ((Z1 - Z0) / NB);
          const x = R.x + b * binW, y = A.Y(dens);
          g.fillStyle = zc < S.delta ? V.alpha(C.a, 0.55) : V.alpha(C.c, 0.4);
          g.fillRect(x + 0.5, y, binW - 1, R.y + R.h - y);
        }
        // pdf e área
        g.beginPath(); g.moveTo(A.X(Z0), A.Y(0));
        for (let i = 0; i <= 100; i++) { const z = Z0 + ((Math.min(S.delta, Z1) - Z0) * i) / 100; g.lineTo(A.X(z), A.Y(pdf(z))); }
        g.lineTo(A.X(Math.min(S.delta, Z1)), A.Y(0)); g.closePath(); g.fillStyle = V.alpha(C.a, 0.12); g.fill();
        curve(g, pdf, Z0, Z1, A.X, A.Y, { color: C.b, lw: 2.2 });
        const xd = A.X(V.clamp(S.delta, Z0, Z1));
        V.line(g, xd, R.y - 6, xd, R.y + R.h, { color: C.ink, lw: 1.6, dash: [5, 4] });
        V.text(g, 'Δ', xd + 4, R.y + 4, { size: 12, font: 'mono', color: C.ink, weight: 700 });
        if (lt) { const xl = A.X(V.clamp(lt.d, Z0, Z1)); V.dot(g, xl, R.y + R.h - 4, 4, lt.win ? C.a : C.c); }
        V.text(g, 'A vence ←', A.X(V.clamp(S.delta, Z0 + 2.2, Z1)) - 8, R.y + 14, { size: 11, color: C.a, align: 'right', weight: 600 });
        V.text(g, '→ B vence', A.X(V.clamp(S.delta, Z0, Z1 - 2.2)) + 8, R.y + 14, { size: 11, color: C.c, weight: 600 });
        // barra de convergência
        const by = R.y + R.h + (narrow ? 60 : 40), bx = R.x, bw = R.w;
        const emp = S.n ? S.winsA / S.n : 0;
        V.box(g, bx, by, bw, 12, { r: 6, fill: V.alpha(C.muted, 0.14) });
        V.box(g, bx, by, Math.max(12, bw * emp), 12, { r: 6, fill: V.alpha(C.a, 0.75) });
        const xt = bx + bw * theory();
        V.line(g, xt, by - 5, xt, by + 17, { color: C.b, lw: 2.5 });
        V.text(g, 'fração de vitórias de A (barra) vs ' + (S.noise === 'gumbel' ? 'σ(Δ)' : 'Φ(Δ/(√2·s))') + ' (traço)', bx, by - 8, { size: 11, color: C.muted });
      });
    })();

    /* ================================================ FIG 4 — ranking de K respostas */
    (function () {
      const fig = root.querySelector('#fig-ranking');
      adaptH(fig, 440, 660);
      const R_ = [2.0, 1.0, 0.5, -0.5, -0.8, -1.2, -1.5, -2.0, -2.4];
      const NM = 'ABCDEFGHI';
      const S = { K: 4, mode: 'pares', t0: 0 };
      const pairs = () => { const p = []; for (let i = 0; i < S.K; i++) for (let j = i + 1; j < S.K; j++) p.push([i, j]); return p; };
      function pairLoss() { const P = pairs(); return P.reduce((s, [i, j]) => s - logsig(R_[i] - R_[j]), 0) / P.length; }
      function plStages() { const st = []; for (let i = 0; i < S.K; i++) { let den = 0; for (let j = i; j < S.K; j++) den += Math.exp(R_[j]); st.push(Math.exp(R_[i]) / den); } return st; }
      const ctl = UI.controls(fig);
      nocase(UI.slider(ctl, { label: 'K respostas', min: 2, max: 9, step: 1, value: 4, fmt: (v) => String(v), onInput: (v) => { S.K = v; S.t0 = H.t; upd(); } }));
      nocase(UI.seg(ctl, { label: 'Perda', options: [{ v: 'pares', label: 'pares (InstructGPT)' }, { v: 'pl', label: 'Plackett–Luce' }], value: 'pares', onChange: (v) => { S.mode = v; S.t0 = H.t; } }));
      const rP = UI.readout(ctl, 'pares C(K,2)', '—'), rF = UI.readout(ctl, 'forwards', '—'), rL = UI.readout(ctl, 'perda média (pares)', '—'), rPL = UI.readout(ctl, 'NLL Plackett–Luce', '—');
      nocase(ctl);
      function upd() {
        const np = (S.K * (S.K - 1)) / 2;
        rP.set(String(np));
        rF.set(S.K + ' agrupado · ' + 2 * np + ' solto');
        rL.set(fmt(pairLoss(), 4));
        const P = plStages().reduce((a, b) => a * b, 1);
        rPL.set(fmt(-Math.log(P), 4) + ' (P = ' + fmt(P, 4) + ')');
      }
      upd();
      const H = V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const K = S.K, P = pairs();
        const narrow = w < 640;
        const el = t - S.t0;
        // personagens (respostas), em ordem do ranking humano
        const rowY = 34;
        const charW = Math.min(46, (w - 40) / K);
        const x0c = w / 2 - (charW * K) / 2 + charW / 2;
        V.text(g, 'ranking humano →', 10, rowY - 20, { size: 11, color: C.muted });
        for (let i = 0; i < K; i++) {
          let alpha = 1;
          if (S.mode === 'pl') { const cyc = (el % (K * 1.2 + 1.5)); const stage = Math.floor(cyc / 1.2); alpha = i < stage ? 0.25 : 1; }
          face(g, x0c + i * charW, rowY, Math.min(22, charW * 0.6), C.series[i % 6], { alpha });
          V.text(g, NM[i], x0c + i * charW, rowY + 22, { size: 11, font: 'mono', align: 'center', color: C.ink, weight: 700, alpha });
          V.text(g, fmt(R_[i], 1), x0c + i * charW, rowY + 35, { size: 9.5, font: 'mono', align: 'center', color: C.muted, alpha });
        }
        const top = rowY + 54;
        // matriz
        const ms = narrow ? Math.min(w - 50, 300) : Math.min(h - top - 30, w * 0.46);
        const cell = (ms - (K - 1) * 2) / K;
        const mx = narrow ? (w - ms) / 2 + 10 : 40, my = top + 6;
        const cyc = P.length * 0.75 + 1.2;
        const cur = S.mode === 'pares' ? Math.floor((el % cyc) / 0.75) : -1;
        for (let i = 0; i < K; i++) for (let j = 0; j < K; j++) {
          const x = mx + j * (cell + 2), y = my + i * (cell + 2);
          if (i === j) { V.box(g, x, y, cell, cell, { r: 3, fill: V.alpha(C.muted, 0.18) }); continue; }
          const p = sig(R_[i] - R_[j]);
          const upper = i < j;
          const isCur = upper && P[cur] && P[cur][0] === i && P[cur][1] === j;
          const done = upper && cur >= 0 && P.findIndex((q) => q[0] === i && q[1] === j) < cur;
          g.globalAlpha = upper ? 1 : 0.28;
          V.box(g, x, y, cell, cell, { r: 3, fill: V.cmap((p - 0.5) * 2, 'div') });
          if (cell >= 24) V.text(g, (cell >= 40 ? fmt(p, 2) : fmt(p, 2).replace(/^0/, '')), x + cell / 2, y + cell / 2 + 0.5, { size: Math.min(11.5, cell * 0.32), font: 'mono', align: 'center', base: 'middle', color: C.ink });
          g.globalAlpha = 1;
          if (isCur || (S.mode === 'pares' && done)) { g.strokeStyle = isCur ? C.d : V.alpha(C.d, 0.4); g.lineWidth = isCur ? 2.5 : 1; V.rr(g, x - 1, y - 1, cell + 2, cell + 2, 4); g.stroke(); }
        }
        for (let i = 0; i < K; i++) {
          V.text(g, NM[i], mx - 6, my + i * (cell + 2) + cell / 2, { size: 10.5, font: 'mono', align: 'right', base: 'middle', color: C.muted });
          V.text(g, NM[i], mx + i * (cell + 2) + cell / 2, my - 5, { size: 10.5, font: 'mono', align: 'center', color: C.muted });
        }
        V.text(g, 'σ(rᵢ − rⱼ): “linha vence coluna”', mx, my + ms + 16, { size: 10.5, color: C.muted });
        // painel direito
        const px = narrow ? 16 : mx + ms + 36, py = narrow ? my + ms + 36 : top, pw = narrow ? w - 32 : w - px - 18;
        if (S.mode === 'pares') {
          V.text(g, 'cada par: −log σ(Δ)', px, py + 8, { size: 12, weight: 600, color: C.ink });
          const rows = Math.min(P.length, narrow ? 10 : Math.floor((h - py - 70) / 17));
          const start = cur >= rows ? cur - rows + 1 : 0;
          let acc = 0, nacc = 0;
          P.forEach(([i, j], idx) => { if (idx <= cur) { acc += -logsig(R_[i] - R_[j]); nacc++; } });
          for (let r = 0; r < rows && start + r < P.length; r++) {
            const idx = start + r, [i, j] = P[idx];
            const d = R_[i] - R_[j], l = -logsig(d);
            const y = py + 26 + r * 17;
            const on = idx <= cur;
            const col = idx === cur ? C.d : on ? C.ink : C.muted;
            V.text(g, `${NM[i]}≻${NM[j]}  Δ=${fmt(d, 1)}  σ=${fmt(sig(d), 4)}`, px, y, { size: 11, font: 'mono', color: col, alpha: on ? 1 : 0.5 });
            const bw = Math.min(90, pw * 0.3);
            V.box(g, px + pw - bw, y - 9, bw * Math.min(1, l / 0.8), 9, { r: 3, fill: V.alpha(C.d, on ? 0.85 : 0.25) });
            V.text(g, fmt(l, 3), px + pw - bw - 6, y, { size: 10.5, font: 'mono', align: 'right', color: col, alpha: on ? 1 : 0.5 });
          }
          const yb = py + 26 + rows * 17 + 10;
          V.text(g, `média até agora: ${nacc ? fmt(acc / nacc, 4) : '—'}   (final ${fmt(pairLoss(), 4)})`, px, yb, { size: 11.5, font: 'mono', color: C.d, weight: 600 });
          V.text(g, `1 elemento do batch = ${K} forwards do RM`, px, yb + 18, { size: 11, color: C.muted });
        } else {
          const st = plStages();
          const cycPL = K * 1.2 + 1.5;
          const stage = Math.min(K - 1, Math.floor((el % cycPL) / 1.2));
          V.text(g, 'urna: escolha sem reposição', px, py + 8, { size: 12, weight: 600, color: C.ink });
          const remaining = [];
          for (let j = stage; j < K; j++) remaining.push(j);
          const probs = V.softmax(remaining.map((j) => R_[j]));
          const pmax = Math.max(...probs);
          // altura relativa à maior probabilidade do estágio (o valor exato vai no rótulo)
          const bh = Math.min(60, h * 0.15), bwid = Math.min(pw, 40 * remaining.length);
          const by = py + 30;
          const bw = (bwid - 4 * (remaining.length - 1)) / remaining.length;
          remaining.forEach((j, q) => {
            const x = px + q * (bw + 4), hh = (bh * probs[q]) / pmax;
            V.box(g, x, by + bh - hh, bw, Math.max(2, hh), { r: 3, fill: q === 0 ? C.d : V.alpha(C.series[j % 6], 0.55) });
            V.text(g, NM[j], x + bw / 2, by + bh + 13, { size: 10.5, font: 'mono', align: 'center', color: q === 0 ? C.d : C.muted, weight: q === 0 ? 700 : 400 });
            if (bw > 26) V.text(g, fmt(probs[q], 2), x + bw / 2, by + bh - hh - 4, { size: 9.5, font: 'mono', align: 'center', color: C.ink });
          });
          let y = by + bh + 34, prod = 1;
          for (let i = 0; i <= stage; i++) prod *= st[i];
          // se não couber tudo, mostra os últimos estágios (o produto usa todos)
          const fit = Math.max(1, Math.floor((h - 34 - y) / 16) + 1);
          const first = Math.max(0, stage + 1 - fit);
          for (let i = first; i <= stage; i++) {
            const pre = i === first && first > 0 ? '… ' : '';
            V.text(g, `${pre}estágio ${i + 1}: ${NM[i]} entre {${NM.slice(i, K).split('').join(',')}} → ${fmt(st[i], 4)}`, px, y, { size: 10.5, font: 'mono', color: i === stage ? C.d : C.ink, maxW: pw });
            y += 16;
          }
          V.text(g, stage >= K - 1 ? `P(ranking) = ${fmt(prod, 4)}   NLL = ${fmt(-Math.log(prod), 4)}` : `P até aqui = ${fmt(prod, 4)}`, px, Math.min(y + 8, h - 12), { size: 11.5, font: 'mono', color: C.d, weight: 600 });
        }
      });
    })();

    /* ================================================ FIG 5 — perda e gradiente */
    (function () {
      const fig = root.querySelector('#fig-perda');
      adaptH(fig, 440, 640);
      const S = { rw: 1.2, rl: 0.4, v: 'bt', m: 1, train: false, shiftUp: true, parts: [] };
      const MS = { margem: [{ v: 1 / 3, label: '1/3' }, { v: 2 / 3, label: '2/3' }, { v: 1, label: '1' }, { v: 3, label: '3' }], escalado: [{ v: 1, label: '1' }, { v: 2, label: '2' }, { v: 3, label: '3' }] };
      function L(d) {
        if (S.v === 'margem') return -logsig(d - S.m);
        if (S.v === 'escalado') return -S.m * logsig(d);
        if (S.v === 'ruido') return -Math.log(0.9 * sig(d) + 0.05);
        return -logsig(d);
      }
      function dL(d) {
        if (S.v === 'margem') return -sig(-(d - S.m));
        if (S.v === 'escalado') return -S.m * sig(-d);
        if (S.v === 'ruido') { const p = 0.9 * sig(d) + 0.05; return -0.9 * sig(d) * sig(-d) / p; }
        return -sig(-d);
      }
      function P(d) { return S.v === 'ruido' ? 0.9 * sig(d) + 0.05 : S.v === 'margem' ? sig(d - S.m) : sig(d); }
      const ctl = UI.controls(fig);
      const sW = nocase(UI.slider(ctl, { label: 'r_w (preferida)', min: -4, max: 4, step: 0.05, value: S.rw, fmt: (v) => fmt(v, 2), onInput: (v) => { S.rw = v; upd(); } }));
      const sL = nocase(UI.slider(ctl, { label: 'r_l (rejeitada)', min: -4, max: 4, step: 0.05, value: S.rl, fmt: (v) => fmt(v, 2), onInput: (v) => { S.rl = v; upd(); } }));
      nocase(UI.seg(ctl, { label: 'Perda', options: [{ v: 'bt', label: 'BT' }, { v: 'margem', label: 'margem (Llama 2)' }, { v: 'escalado', label: 'BT escalado' }, { v: 'ruido', label: '10% de ruído' }], value: 'bt', onChange: (v) => { S.v = v; gM.show(v === 'margem'); gE.show(v === 'escalado'); if (v === 'margem') S.m = segM.get(); if (v === 'escalado') S.m = segE.get(); upd(); } }));
      const gM = UI.group(ctl), gE = UI.group(ctl);
      const segM = nocase(UI.seg(gM.el, { label: 'm', options: MS.margem, value: 1, onChange: (v) => { S.m = v; upd(); } }));
      const segE = nocase(UI.seg(gE.el, { label: 'm', options: MS.escalado, value: 2, onChange: (v) => { S.m = v; upd(); } }));
      gM.show(false); gE.show(false);
      const bTrain = UI.button(ctl, '▶ treinar', () => { S.train = !S.train; bTrain.innerHTML = S.train ? '⏸ pausar' : '▶ treinar'; }, { primary: true });
      UI.button(ctl, '↕ deslocar as duas', () => { const s = S.shiftUp && Math.max(S.rw, S.rl) <= 2 ? 2 : -2; S.shiftUp = s < 0; if (Math.min(S.rw, S.rl) + s < -4 || Math.max(S.rw, S.rl) + s > 4) return; S.rw += s; S.rl += s; sync(); upd(); });
      UI.button(ctl, '↺ par invertido', () => { S.rw = 0.4; S.rl = 1.2; S.train = false; bTrain.innerHTML = '▶ treinar'; sync(); upd(); });
      const rD = UI.readout(ctl, 'Δ', '—'), rP = UI.readout(ctl, 'P(y_w ≻ y_l)', '—'), rLs = UI.readout(ctl, 'perda', '—'), rG = UI.readout(ctl, '∂L/∂Δ', '—');
      nocase(ctl);
      function sync() { sW.set(S.rw, true); sL.set(S.rl, true); }
      function upd() {
        const d = S.rw - S.rl;
        rD.set(fmt(d, 2)); rP.set(fmt(S.v === 'ruido' ? P(d) : sig(d), 4)); rLs.set(fmt(L(d), 4)); rG.set(fmt(dL(d), 4), Math.abs(dL(d)) > 0.45 ? 'warn' : 'ok');
      }
      upd();
      for (let i = 0; i < 28; i++) S.parts.push({ ph: (i * 0.137) % 1, side: i % 2, jx: ((i * 37) % 11) / 11 - 0.5 });
      V.fig(fig, (g, w, h, t, dt) => {
        const C = V.C();
        if (S.train) {
          const gr = -dL(S.rw - S.rl);
          const eta = 1.6 * dt;
          S.rw = Math.min(4, S.rw + eta * gr); S.rl = Math.max(-4, S.rl - eta * gr);
          if (gr < 0.004 || S.rw >= 4 || S.rl <= -4) { S.train = false; bTrain.innerHTML = '▶ treinar'; }
          sync(); upd();
        }
        const d = S.rw - S.rl, gmag = Math.abs(dL(d));
        const narrow = w < 640;
        // painel das barras
        const Lp = narrow ? { x: 0, y: 0, w: w, h: 250 } : { x: 0, y: 0, w: w * 0.4, h: h };
        const y0 = Lp.y + 30, y1 = Lp.y + Lp.h - 26;
        const Yr = (r) => y1 - ((r + 4) / 8) * (y1 - y0);
        const axX = Lp.x + 42;
        V.line(g, axX, y0, axX, y1, { color: C.muted, lw: 1.2 });
        [-4, -2, 0, 2, 4].forEach((v) => { V.line(g, axX - 4, Yr(v), Lp.x + Lp.w - 12, Yr(v), { color: v === 0 ? V.alpha(C.muted, 0.6) : C.grid, lw: 1 }); V.text(g, fmt(v, 0), axX - 7, Yr(v), { size: 10.5, font: 'mono', align: 'right', base: 'middle', color: C.muted }); });
        V.text(g, 'recompensa r', axX - 30, y0 - 14, { size: 11, color: C.muted });
        const bw = Math.min(54, Lp.w * 0.16);
        const xw = axX + (Lp.w - 54) * 0.3, xl = axX + (Lp.w - 54) * 0.72;
        [[xw, S.rw, C.a, 'r_w', 1], [xl, S.rl, C.c, 'r_l', -1]].forEach(([x, r, col, nm, dir]) => {
          const yt = Yr(r), yz = Yr(0);
          V.box(g, x - bw / 2, Math.min(yt, yz), bw, Math.max(2, Math.abs(yz - yt)), { r: 5, fill: V.alpha(col, 0.35), stroke: col, lw: 1.5 });
          face(g, x, yt - (r >= 0 ? 16 : -16), 22, col, { mood: dir > 0 ? undefined : 'o' });
          V.text(g, nm + ' = ' + fmt(r, 2), x, r >= 0 ? Math.min(y1 - 4, yz + 16) : Math.max(y0 + 8, yz - 8), { size: 11, font: 'mono', align: 'center', color: col, weight: 600 });
          // seta do gradiente
          const len = 120 * gmag / (S.v === 'escalado' ? S.m : 1);
          const ys = yt - (r >= 0 ? 30 : -30) * (dir > 0 ? 1 : -1);
          const ax = x + bw / 2 + 12;
          if (len > 2) {
            V.arrow(g, ax, yt, ax, yt - dir * len, { color: C.d, lw: 2.5, head: 8 });
            if (dir > 0) V.text(g, '|∂L/∂Δ| = ' + fmt(gmag, 2), ax + 8, yt - len / 2, { size: 10.5, font: 'mono', color: C.d, base: 'middle', weight: 600 });
            // partículas
            S.parts.forEach((p) => { if (p.side !== (dir > 0 ? 0 : 1)) return; const f = (p.ph + t * (0.4 + 1.6 * gmag)) % 1; V.dot(g, ax + p.jx * 8, yt - dir * len * f, 2, V.alpha(C.d, 0.9 * (1 - f))); });
          }
          void ys;
        });
        // colchete Δ
        const ya = Yr(S.rw), yb = Yr(S.rl), xm = (xw + xl) / 2;
        V.line(g, xw + bw / 2, ya, xm + 6, ya, { color: V.alpha(C.ink, 0.5), lw: 1, dash: [3, 3] });
        V.line(g, xl - bw / 2, yb, xm - 6, yb, { color: V.alpha(C.ink, 0.5), lw: 1, dash: [3, 3] });
        V.arrow(g, xm, yb, xm, ya, { color: C.ink, lw: 1.4, head: 6 });
        V.text(g, 'Δ', xm + 6, (ya + yb) / 2, { size: 12, font: 'mono', color: C.ink, weight: 700, base: 'middle' });
        // gráficos
        const Rp = narrow ? { x: 46, y: 284, w: w - 64, h: (h - 284 - 60) / 2 - 14 } : { x: w * 0.4 + 46, y: 26, w: w * 0.6 - 66, h: (h - 90) / 2 - 10 };
        const Rl = { x: Rp.x, y: Rp.y + Rp.h + 38, w: Rp.w, h: Rp.h };
        const dd = V.clamp(d, -6, 6);
        const A1 = axes(g, Rp, { x0: -6, x1: 6, y0: 0, y1: 1, xt: [-6, -3, 0, 3, 6], yt: [0, 0.5, 1], fy: (v) => fmt(v, 1), yl: S.v === 'margem' ? 'σ(Δ − m)' : S.v === 'ruido' ? 'P̂ = 0,9σ(Δ) + 0,05' : 'P = σ(Δ)' });
        curve(g, P, -6, 6, A1.X, A1.Y, { color: C.b, lw: 2.4 });
        if (S.v !== 'bt') curve(g, sig, -6, 6, A1.X, A1.Y, { color: V.alpha(C.b, 0.35), lw: 1.2, dash: [4, 4] });
        V.line(g, A1.X(dd), Rp.y, A1.X(dd), Rp.y + Rp.h, { color: V.alpha(C.ink, 0.35), lw: 1, dash: [3, 3] });
        V.glow(g, A1.X(dd), A1.Y(P(dd)), 14, C.b, 0.6); V.dot(g, A1.X(dd), A1.Y(P(dd)), 4.5, C.b);
        const lmax = S.v === 'escalado' ? 3 * S.m : 3;
        const A2 = axes(g, Rl, { x0: -6, x1: 6, y0: 0, y1: lmax, xt: [-6, -3, 0, 3, 6], yt: [0, lmax / 2, lmax], fy: (v) => fmt(v, 1), xl: 'Δ = r_w − r_l', yl: 'perda' });
        g.save(); g.beginPath(); g.rect(Rl.x, Rl.y - 1, Rl.w, Rl.h + 2); g.clip();
        curve(g, L, -6, 6, A2.X, A2.Y, { color: C.d, lw: 2.4 });
        g.restore();
        // tangente = gradiente
        const lv = L(dd), sl = dL(dd);
        const span = 1.6;
        const xa = dd - span, xb = dd + span;
        g.save(); g.beginPath(); g.rect(Rl.x, Rl.y, Rl.w, Rl.h); g.clip();
        V.line(g, A2.X(xa), A2.Y(lv + sl * (xa - dd)), A2.X(xb), A2.Y(lv + sl * (xb - dd)), { color: C.e, lw: 2 });
        g.restore();
        if (lv <= lmax) { V.glow(g, A2.X(dd), A2.Y(lv), 14, C.d, 0.6); V.dot(g, A2.X(dd), A2.Y(lv), 4.5, C.d); }
        else V.text(g, '↑ perda = ' + fmt(lv, 2), A2.X(dd), Rl.y + 12, { size: 10.5, font: 'mono', align: 'center', color: C.d });
        V.text(g, 'tangente: inclinação = ∂L/∂Δ = ' + fmt(sl, 3), Rl.x + Rl.w, Rl.y - 7, { size: 10.5, font: 'mono', align: 'right', color: C.e });
      });
    })();

    /* ================================================ FIG 6 — dentro do RM */
    (function () {
      const fig = root.querySelector('#fig-rm');
      adaptH(fig, 410, 595);
      const TOK = {
        w: ['Capital', 'da', 'França', '?', 'É', 'Paris', '.', '⟨eos⟩', '⟨pad⟩', '⟨pad⟩'],
        l: ['Capital', 'da', 'França', '?', 'Acho', 'que', 'Lyon', '.', '⟨eos⟩', '⟨pad⟩'],
      };
      const EOS = { w: 7, l: 8 };
      const D = 6, T = 10;
      const w6 = [0.5, -0.4, 0.8, 0.3, -0.6, 0.5], b6 = 0.1;
      const W5 = [[0.5, -0.4, 0.8, 0.3, -0.6, 0.5], [0.7, 0.1, 0.4, -0.2, -0.5, 0.2], [0.3, -0.2, 0.2, 0.6, -0.1, 0.4], [-0.1, 0.6, 0.3, 0.5, 0.2, -0.3], [-0.6, 0.8, -0.2, 0.7, 0.4, -0.5]];
      const b5 = [0.1, 0.4, 0.6, 0.2, -0.2];
      const ATTR = ['utilidade', 'correção', 'coerência', 'complexidade', 'verbosidade'];
      const ATTRs = ['util.', 'corr.', 'coer.', 'compl.', 'verb.'];
      function mkH(seed, eosIdx, eosRow) {
        const r = V.rng(seed);
        const Hm = Array.from({ length: T }, () => Array.from({ length: D }, () => Math.round(r.normal() * 7) / 10));
        Hm[eosIdx] = eosRow;
        Hm[T - 1] = [-0.8, 0.9, -0.6, 0.2, 0.7, -0.9];
        return Hm;
      }
      const HM = { w: mkH(11, 7, [1.0, -0.5, 0.5, 0.0, -0.5, -0.2]), l: mkH(23, 8, [0.2, 0.5, 0.4, 0.6, 0.1, 0.0]) };
      HM.l[T - 1] = [-0.5, 0.6, -0.9, -0.3, 0.8, -0.4];
      const dot = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);
      const S = { which: 'w', heads: 1, eos: true, t0: -5 };
      const readIdx = (which) => (S.eos ? EOS[which] : T - 1);
      const score = (which) => dot(w6, HM[which][readIdx(which)]) + b6;
      const ctl = UI.controls(fig);
      nocase(UI.seg(ctl, { label: 'Resposta', options: [{ v: 'w', label: 'preferida y_w' }, { v: 'l', label: 'rejeitada y_l' }], value: 'w', onChange: (v) => { S.which = v; S.t0 = H.t; } }));
      nocase(UI.seg(ctl, { label: 'Cabeça', options: [{ v: 1, label: '1 escalar (BT)' }, { v: 5, label: '5 atributos (HelpSteer2)' }], value: 1, onChange: (v) => { S.heads = v; S.t0 = H.t; upd(); } }));
      nocase(UI.toggle(ctl, { label: 'ler no EOS', value: true, onChange: (v) => { S.eos = v; S.t0 = H.t; upd(); } }));
      const rW = UI.readout(ctl, 'r_w', '—'), rL = UI.readout(ctl, 'r_l', '—'), rS = UI.readout(ctl, 'σ(r_w − r_l)', '—');
      nocase(ctl);
      const relabel = (ro, base) => { const sp = ro.el.querySelector('span'); if (sp) sp.textContent = S.heads === 5 ? base + ' (cabeça BT)' : base; };
      function upd() {
        relabel(rW, 'r_w'); relabel(rL, 'r_l'); relabel(rS, 'σ(r_w − r_l)');
        const st = S.eos ? 'ok' : 'bad';
        rW.set(fmt(score('w'), 2), st); rL.set(fmt(score('l'), 2), st);
        rS.set(S.eos ? fmt(sig(score('w') - score('l')), 4) : 'lixo (PAD)', st);
      }
      upd();
      const H = V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const narrow = w < 640;
        const el = (t - S.t0) % 10;
        const toks = TOK[S.which], Hm = HM[S.which], ri = readIdx(S.which);
        // geometria
        const leftW = narrow ? w - 20 : w * 0.6;
        const colW = Math.min(52, (leftW - 30) / T);
        const gx = narrow ? (w - colW * T) / 2 : 22;
        // no celular, Hᵀ começa no topo (sem faixa vazia acima)
        const tokY = narrow ? 226 : h - 40;
        const trY0 = narrow ? 156 : h - 118, trY1 = narrow ? 188 : h - 74;
        const cellH = narrow ? 17 : Math.min(34, (trY0 - 60) / D);
        const hy = trY0 - 18 - D * cellH;
        // tokens
        toks.forEach((tk, i) => {
          const x = gx + i * colW;
          const isPrompt = i < 4, isEos = tk === '⟨eos⟩', isPad = tk === '⟨pad⟩';
          const col = isEos ? C.d : isPad ? C.muted : isPrompt ? C.b : (S.which === 'w' ? C.a : C.c);
          chip(g, x + 2, tokY, colW - 4, 22, tk, { fill: V.alpha(col, isPad ? 0.08 : 0.2), stroke: V.alpha(col, 0.7), size: Math.min(10.5, colW * 0.22), color: C.ink });
        });
        V.text(g, 'prompt x', gx + 2, tokY + 36, { size: 10.5, color: C.b });
        V.text(g, S.which === 'w' ? 'resposta preferida y_w' : 'resposta rejeitada y_l', gx + 4 * colW + 2, tokY + 36, { size: 10.5, color: S.which === 'w' ? C.a : C.c });
        // Transformer
        V.box(g, gx, trY0, colW * T, trY1 - trY0, { r: 8, fill: V.alpha(C.b, 0.12), stroke: V.alpha(C.b, 0.6) });
        V.text(g, 'Transformer SFT (L camadas, atenção causal) — sem lm_head', gx + (colW * T) / 2, (trY0 + trY1) / 2 + 1, { size: Math.min(11.5, colW * 0.26), align: 'center', base: 'middle', color: C.ink, maxW: colW * T - 10 });
        // partículas subindo
        for (let i = 0; i < T; i++) {
          const x = gx + i * colW + colW / 2;
          for (let q = 0; q < 2; q++) {
            const f = (t * 0.6 + i * 0.13 + q * 0.5) % 1;
            const y = V.lerp(tokY - 2, hy + D * cellH, f);
            V.dot(g, x, y, 1.8, V.alpha(i === ri ? C.d : C.b, 0.7 * (1 - Math.abs(f - 0.5))));
          }
        }
        // matriz H^T: D linhas × T colunas
        const fillP = V.clamp(el / 1.2, 0, 1);
        V.text(g, 'Hᵀ: estado oculto final, ' + D + ' × ' + T + ' (uma coluna por token)', gx, hy - 10, { size: 11, color: C.muted });
        for (let i = 0; i < T; i++) for (let d = 0; d < D; d++) {
          const x = gx + i * colW + 2, y = hy + d * cellH;
          const shown = i / T <= fillP;
          const v = Hm[i][d];
          g.globalAlpha = shown ? (i === ri ? 1 : 0.55) : 0.12;
          V.box(g, x, y + 1, colW - 4, cellH - 2, { r: 3, fill: V.cmap(v / 1.5, 'div') });
          if (shown && colW >= 30 && cellH >= 16) V.text(g, fmt(v, 1), x + (colW - 4) / 2, y + cellH / 2 + 0.5, { size: Math.min(10.5, cellH * 0.55), font: 'mono', align: 'center', base: 'middle', color: C.ink });
          g.globalAlpha = 1;
        }
        // destaque da coluna lida
        const hx = gx + ri * colW + 1;
        g.strokeStyle = S.eos ? C.d : RED(); g.lineWidth = 2.5; V.rr(g, hx, hy - 2, colW - 2, D * cellH + 4, 5); g.stroke();
        V.text(g, S.eos ? 'lida no EOS' : 'lida no PAD ✗', hx + colW / 2, hy + D * cellH + 13, { size: 10.5, align: 'center', color: S.eos ? C.d : RED(), weight: 700 });
        // computação
        const hcol = Hm[ri];
        const p2 = V.clamp((el - 1.2) / 1.0, 0, 1), p3 = V.clamp((el - 2.2) / 1.0, 0, 1), p4 = V.clamp((el - 3.2) / 0.8, 0, 1);
        const cx0 = narrow ? 12 : w * 0.6 + 16, cy0 = narrow ? 292 : hy;
        const cw = narrow ? Math.min(46, (w - 24) / 7) : Math.min(50, (w * 0.4 - 30) / 5.2);
        const ch = narrow ? 24 : cellH;
        if (S.heads === 1) {
          const cols = [['h_EOS', hcol, C.d], ['w', w6, C.b], ['h ⊙ w', hcol.map((x, i) => x * w6[i]), C.e]];
          cols.forEach(([nm, vec, col], c) => {
            const pp = c === 0 ? p2 : c === 1 ? p2 : p3;
            const x = cx0 + c * (cw + 18);
            V.text(g, nm, x + cw / 2, cy0 - 8, { size: 11, font: 'mono', align: 'center', color: col, weight: 600 });
            vec.forEach((v, d) => {
              const y = cy0 + d * ch;
              g.globalAlpha = 0.15 + 0.85 * pp;
              V.box(g, x, y + 1, cw, ch - 2, { r: 4, fill: V.cmap(v / 1.2, 'div'), stroke: V.alpha(col, 0.5), lw: 1 });
              V.text(g, fmt(v, 2), x + cw / 2, y + ch / 2 + 0.5, { size: Math.min(10.5, cw * 0.24), font: 'mono', align: 'center', base: 'middle', color: C.ink });
              g.globalAlpha = 1;
            });
            if (c < 2) V.text(g, c === 0 ? '⊙' : '=', x + cw + 9, cy0 + (D * ch) / 2, { size: 14, align: 'center', base: 'middle', color: C.muted });
          });
          const sx = cx0 + 3 * (cw + 18);
          const r = dot(hcol, w6) + b6;
          g.globalAlpha = 0.15 + 0.85 * p4;
          V.text(g, 'Σ + b', sx + 2, cy0 + (D * ch) / 2 - 22, { size: 11, font: 'mono', color: C.muted });
          V.text(g, 'b = ' + fmt(b6, 1), sx + 2, cy0 + (D * ch) / 2 - 8, { size: 10, font: 'mono', color: C.muted });
          g.globalAlpha = 1;
          const ry = narrow ? cy0 + D * ch + 26 : cy0 + D * ch + 34;
          const rx = narrow ? cx0 : cx0;
          if (p4 > 0) {
            V.glow(g, rx + 40, ry + 14, 34 * p4, S.eos ? (S.which === 'w' ? C.a : C.c) : RED(), 0.5);
            chip(g, rx, ry, 150, 28, 'r = ' + fmt(r, 2), { fill: V.alpha(S.eos ? C.d : RED(), 0.25), stroke: S.eos ? C.d : RED(), size: 13, weight: 700 });
            if (!S.eos) V.wrap(g, 'logit de recompensa num PAD: não leu a resposta como um todo; N+ observa que essas posições dão valores inválidos, quase sempre negativos', rx, ry + 46, Math.min(260, w - rx - 10), { size: 10.5, color: RED(), lineH: 14, maxLines: 4 });
            else {
              const rw_ = score('w'), rl_ = score('l');
              V.text(g, 'r_w = ' + fmt(rw_, 2) + '   r_l = ' + fmt(rl_, 2), rx, ry + 46, { size: 11, font: 'mono', color: C.ink });
              V.text(g, 'σ(r_w − r_l) = ' + fmt(sig(rw_ - rl_), 4), rx, ry + 62, { size: 11, font: 'mono', color: C.b, weight: 600 });
            }
          }
        } else {
          // 5 cabeças: W (5×6) · h → 5 escalares
          const mcw = narrow ? Math.min(30, (w - 150) / 6) : Math.min(30, (w * 0.4 - 90) / 6);
          const mch = narrow ? 22 : Math.min(22, ch);
          V.text(g, 'W (5 × ' + D + ')', cx0 + (mcw * D) / 2, cy0 - 8, { size: 11, font: 'mono', align: 'center', color: C.b, weight: 600 });
          for (let a = 0; a < 5; a++) for (let d = 0; d < D; d++) {
            const x = cx0 + d * mcw, y = cy0 + a * mch;
            g.globalAlpha = 0.15 + 0.85 * p2;
            V.box(g, x + 1, y + 1, mcw - 2, mch - 2, { r: 3, fill: V.cmap(W5[a][d], 'div') });
            if (mcw >= 26) V.text(g, fmt(W5[a][d], 1), x + mcw / 2, y + mch / 2 + 0.5, { size: 9.5, font: 'mono', align: 'center', base: 'middle', color: C.ink });
            g.globalAlpha = 1;
          }
          const panelW = narrow ? w - 24 : w - cx0 - 14;
          const oy = cy0 + 5 * mch + 26;
          V.text(g, '= W · h_EOS + b  →  5 escalares', cx0, oy - 8, { size: 10.5, font: 'mono', color: C.muted });
          const outs = W5.map((row, a) => dot(row, hcol) + b5[a]);
          const labW = 86, valW = 48, rh = narrow ? 20 : 19;
          const bmax = Math.max(40, panelW - labW - valW);
          const zx = cx0 + labW + bmax / 2;
          V.line(g, zx, oy, zx, oy + 5 * rh, { color: C.muted, lw: 1 });
          outs.forEach((v, a) => {
            const y = oy + a * rh;
            const pp = V.clamp(p3 * 5 - a, 0, 1);
            const bl = (bmax / 2) * Math.min(1, Math.abs(v) / 2) * pp;
            V.box(g, v >= 0 ? zx : zx - bl, y + 4, Math.max(1, bl), rh - 7, { r: 3, fill: V.alpha(C.series[a], 0.8) });
            V.text(g, ATTR[a], cx0, y + rh / 2 + 0.5, { size: 10.5, base: 'middle', color: C.ink });
            if (pp > 0.5) V.text(g, fmt(v, 2), cx0 + labW + bmax + 6, y + rh / 2 + 0.5, { size: 10.5, font: 'mono', base: 'middle', color: C.ink });
          });
          V.wrap(g, 'Um escalar por atributo, treinado por regressão nas notas 0–4. Na hora de otimizar, você escolhe a combinação.', cx0, oy + 5 * rh + 18, panelW, { size: 10.5, color: C.muted, lineH: 14, maxLines: 3 });
        }
      });
    })();

    /* ================================================ FIG 7 — calibração */
    (function () {
      const fig = root.querySelector('#fig-calib');
      adaptH(fig, 420, 640);
      const BINS = [{ d: 0.25, n: 400, acc: 0.55 }, { d: 1.0, n: 350, acc: 0.7 }, { d: 2.5, n: 250, acc: 0.86 }];
      const N = 1000;
      const ece = (T) => BINS.reduce((s, b) => s + (b.n / N) * Math.abs(b.acc - sig(b.d / T)), 0);
      const nll = (T) => BINS.reduce((s, b) => { const p = sig(b.d / T); return s - b.n * (b.acc * Math.log(p) + (1 - b.acc) * Math.log(1 - p)); }, 0) / N;
      let Topt = 1, best = 1e9; for (let T = 0.5; T <= 3; T += 0.001) { const v = nll(T); if (v < best) { best = v; Topt = T; } }
      const S = { T: 1, anim: null };
      const ctl = UI.controls(fig);
      const sT = nocase(UI.slider(ctl, { label: 'temperatura T', min: 0.5, max: 2.5, step: 0.01, value: 1, fmt: (v) => fmt(v, 2), onInput: (v) => { S.T = v; S.anim = null; upd(); } }));
      UI.button(ctl, '◎ achar o T ótimo', () => { S.anim = { from: S.T, t0: null }; }, { primary: true });
      UI.button(ctl, '↺ T = 1', () => { S.anim = null; S.T = 1; sT.set(1, true); upd(); });
      const rE = UI.readout(ctl, 'ECE', '—'), rN = UI.readout(ctl, 'NLL', '—'), rP = UI.readout(ctl, 'previsões', '—');
      nocase(ctl);
      function upd() {
        const e = ece(S.T);
        rE.set(fmt(e, 4), e < 0.012 ? 'ok' : e > 0.03 ? 'warn' : null);
        rN.set(fmt(nll(S.T), 4));
        rP.set(BINS.map((b) => fmt(sig(b.d / S.T), 3)).join(' / '));
      }
      upd();
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        if (S.anim) {
          if (S.anim.t0 == null) S.anim.t0 = t;
          const k = V.ease.inOut(V.clamp((t - S.anim.t0) / 1.4, 0, 1));
          S.T = V.lerp(S.anim.from, Topt, k); sT.set(S.T, true); upd();
          if (k >= 1) S.anim = null;
        }
        const narrow = w < 640;
        const R1 = narrow ? { x: 44, y: 28, w: w - 62, h: 200 } : { x: 50, y: 30, w: w * 0.5 - 80, h: h - 96 };
        const R2 = narrow ? { x: 44, y: 300, w: w - 62, h: h - 300 - 64 } : { x: w * 0.5 + 40, y: 30, w: w * 0.5 - 70, h: h - 96 };
        // sigmoide e bins
        const A = axes(g, R1, { x0: 0, x1: 3, y0: 0.5, y1: 1, xt: [0, 1, 2, 3], yt: [0.5, 0.75, 1], fy: (v) => fmt(v, 2), xl: '|Δ| = |r_w − r_l|', yl: 'P(acerto)' });
        curve(g, (d) => sig(d), 0, 3, A.X, A.Y, { color: V.alpha(C.b, 0.35), lw: 1.4, dash: [4, 4] });
        curve(g, (d) => sig(d / S.T), 0, 3, A.X, A.Y, { color: C.b, lw: 2.6 });
        V.text(g, 'σ(Δ/T)', A.X(1.55), A.Y(sig(1.55 / S.T)) + 22, { size: 11.5, font: 'mono', color: C.b, weight: 600 });
        V.text(g, 'T = 1', A.X(2.75), A.Y(sig(2.75)) - 6, { size: 10, font: 'mono', color: V.alpha(C.b, 0.7), align: 'right' });
        BINS.forEach((b) => {
          const x = A.X(b.d), yo = A.Y(b.acc), yp = A.Y(sig(b.d / S.T));
          V.line(g, x, yo, x, yp, { color: RED(), lw: 1.6 });
          V.dot(g, x, yo, 3 + Math.sqrt(b.n) / 4, V.alpha(C.a, 0.85));
          V.text(g, fmt(b.acc, 2), x + 10, yo + 4, { size: 10.5, font: 'mono', color: C.a });
        });
        // diagrama de confiabilidade
        const B = axes(g, R2, { x0: 0.5, x1: 1, y0: 0.5, y1: 1, xt: [0.5, 0.75, 1], yt: [0.5, 0.75, 1], fx: (v) => fmt(v, 2), fy: (v) => fmt(v, 2), xl: 'confiança prevista', yl: 'acurácia observada' });
        V.line(g, B.X(0.5), B.Y(0.5), B.X(1), B.Y(1), { color: C.muted, lw: 1.4, dash: [5, 4] });
        V.text(g, 'calibração perfeita', B.X(0.52), B.Y(0.75), { size: 10, color: C.muted, align: 'left' });
        g.beginPath();
        BINS.forEach((b, i) => { const x = B.X(sig(b.d / S.T)), y = B.Y(b.acc); if (i) g.lineTo(x, y); else g.moveTo(x, y); });
        g.strokeStyle = V.alpha(C.a, 0.6); g.lineWidth = 1.6; g.stroke();
        BINS.forEach((b) => {
          const pc = sig(b.d / S.T), x = B.X(pc), y = B.Y(b.acc);
          V.line(g, x, y, B.X(b.acc), y, { color: V.alpha(RED(), 0.8), lw: 2 });
          V.glow(g, x, y, 10 + Math.sqrt(b.n) / 2, C.a, 0.35);
          V.dot(g, x, y, 3 + Math.sqrt(b.n) / 4, C.a);
        });
        const e = ece(S.T);
        const lbl = e < 0.012 ? 'quase calibrado' : sig(BINS[2].d / S.T) > BINS[2].acc ? 'sobreconfiante (prevê mais do que acerta)' : 'subconfiante (acerta mais do que prevê)';
        if (narrow) V.text(g, 'T = ' + fmt(S.T, 2) + ' · ECE = ' + fmt(e, 4) + ' · ' + lbl.replace(/ \(.*\)/, ''), 12, R2.y + R2.h + 46, { size: 11, font: 'mono', align: 'left', color: e < 0.012 ? C.a : C.d, maxW: w - 24 });
        else V.text(g, 'T = ' + fmt(S.T, 2) + ' · ECE = ' + fmt(e, 4) + ' · ' + lbl, R2.x + R2.w, R2.y + R2.h + 46, { size: 11, font: 'mono', align: 'right', color: e < 0.012 ? C.a : C.d, maxW: R2.w + 60 });
      });
    })();

    /* ================================================ FIG 8 — Goodhart no best-of-n */
    (function () {
      const fig = root.querySelector('#fig-goodhart');
      adaptH(fig, 460, 720, 700);
      const NS = [1, 2, 4, 8, 16, 32, 64, 128, 256];
      const KL = (n) => Math.log(n) - (n - 1) / n;
      const CPEN = 2.0, L0 = 1.0, TRI = 1200;
      const S = { ni: 4, b: 0.7, ens: '1', cache: {}, res: null, round: null, rt0: -10, pending: 0 };
      const gold = (q, l) => q - CPEN * Math.pow(Math.max(0, l - L0), 2);
      const memberErr = (r) => (r() < 0.08 ? 2.5 : 0.45) * r.normal();
      const proxyOf = (e, mode) => (mode === '1' ? e[0] : mode === 'media' ? (e[0] + e[1] + e[2] + e[3]) / 4 : Math.min(e[0], e[1], e[2], e[3]));
      function simulate(b) {
        const key = b.toFixed(2);
        if (S.cache[key]) return S.cache[key];
        const r = V.rng(4242);
        const out = { '1': null, media: null, min: null };
        const acc = {}; ['1', 'media', 'min'].forEach((m) => { acc[m] = NS.map(() => ({ g: 0, p: 0, l: 0 })); });
        const e = [0, 0, 0, 0];
        for (let tr = 0; tr < TRI; tr++) {
          const best = { '1': { p: -1e9 }, media: { p: -1e9 }, min: { p: -1e9 } };
          let ni = 0;
          for (let i = 0; i < 256; i++) {
            const q = r.normal(), l = r.normal();
            for (let k = 0; k < 4; k++) e[k] = q + b * l + memberErr(r);
            const gv = gold(q, l);
            for (const m of ['1', 'media', 'min']) { const p = proxyOf(e, m); if (p > best[m].p) best[m] = { p, g: gv, l }; }
            if (i + 1 === NS[ni]) { for (const m of ['1', 'media', 'min']) { const a = acc[m][ni]; a.g += best[m].g; a.p += best[m].p; a.l += best[m].l; } ni++; }
          }
        }
        for (const m of ['1', 'media', 'min']) out[m] = acc[m].map((a, i) => ({ n: NS[i], d: Math.sqrt(KL(NS[i])), g: a.g / TRI, p: a.p / TRI, l: a.l / TRI }));
        S.cache[key] = out;
        return out;
      }
      function fitGao(pts) {
        // R(d) = d(α − βd) ajustado a g − g(n=1)
        const g0 = pts[0].g; let s11 = 0, s12 = 0, s22 = 0, t1 = 0, t2 = 0;
        pts.slice(1).forEach((p) => { const x1 = p.d, x2 = -p.d * p.d, y = p.g - g0; s11 += x1 * x1; s12 += x1 * x2; s22 += x2 * x2; t1 += x1 * y; t2 += x2 * y; });
        const det = s11 * s22 - s12 * s12; if (Math.abs(det) < 1e-12) return null;
        return { a: (t1 * s22 - t2 * s12) / det, b: (s11 * t2 - s12 * t1) / det, g0 };
      }
      function recompute() { S.res = simulate(S.b); upd(); }
      const ctl = UI.controls(fig);
      nocase(UI.slider(ctl, { label: 'n (best-of-n)', min: 0, max: NS.length - 1, step: 1, value: S.ni, fmt: (v) => String(NS[v]), onInput: (v) => { S.ni = v; S.rt0 = -10; upd(); } }));
      nocase(UI.slider(ctl, { label: 'viés de comprimento do proxy', min: 0, max: 1, step: 0.1, value: S.b, fmt: (v) => fmt(v, 1), onInput: (v) => { S.b = v; clearTimeout(S.pending); S.pending = setTimeout(recompute, 60); } }));
      nocase(UI.seg(ctl, { label: 'Proxy', options: [{ v: '1', label: '1 RM' }, { v: 'media', label: 'média de 4' }, { v: 'min', label: 'mínimo de 4 (WCO)' }], value: '1', onChange: (v) => { S.ens = v; S.rt0 = -10; upd(); } }));
      const rK = UI.readout(ctl, 'KL do BoN', '—'), rG = UI.readout(ctl, 'ouro médio', '—'), rP = UI.readout(ctl, 'proxy médio', '—'), rL = UI.readout(ctl, 'tamanho médio', '—');
      nocase(ctl);
      if (LA.onCleanup) LA.onCleanup(() => clearTimeout(S.pending));
      const toks = (l) => Math.max(20, Math.round(300 + 150 * l));
      function upd() {
        if (!S.res) return;
        const pts = S.res[S.ens], p = pts[S.ni];
        rK.set(fmt(KL(NS[S.ni]), 2) + ' nats (d = ' + fmt(p.d, 2) + ')');
        const peak = pts.reduce((a, b) => (b.g > a.g ? b : a), pts[0]);
        rG.set(fmt(p.g, 2), p.n > peak.n && peak.g - p.g > 0.04 ? 'bad' : 'ok');
        rP.set(fmt(p.p, 2)); rL.set('~' + toks(p.l) + ' tokens');
      }
      recompute();
      const live = V.rng(77);
      function newRound(t) {
        const n = NS[S.ni], cands = [];
        let bi = 0, bp = -1e9;
        for (let i = 0; i < n; i++) {
          const q = live.normal(), l = live.normal();
          const e = [0, 1, 2, 3].map(() => q + S.b * l + memberErr(live));
          const p = proxyOf(e, S.ens);
          cands.push({ q, l, g: gold(q, l), p });
          if (p > bp) { bp = p; bi = i; }
        }
        S.round = { cands, bi }; S.rt0 = t;
      }
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        if (t - S.rt0 > 2.2 || !S.round) newRound(t);
        const el = t - S.rt0;
        const narrow = w < 700;
        const R1 = narrow ? { x: 44, y: 34, w: w - 60, h: 250 } : { x: 54, y: 34, w: w * 0.44 - 64, h: h - 110 };
        const R2 = narrow ? { x: 44, y: 360, w: w - 60, h: h - 360 - 70 } : { x: w * 0.44 + 60, y: 34, w: w * 0.56 - 84, h: h - 110 };
        // painel esquerdo: candidatos
        const A = axes(g, R1, { x0: -3, x1: 3.2, y0: -6, y1: 3, xt: [-1.5, 0, 1.5, 3], yt: [-6, -3, 0, 3], fx: (v) => String(toks(v)), fy: (v) => fmt(v, 0), xl: 'comprimento (tokens)', yl: 'qualidade real (ouro)' });
        // zona de penalidade
        const gz = g.createLinearGradient(A.X(L0), 0, R1.x + R1.w, 0); gz.addColorStop(0, V.alpha(RED(), 0)); gz.addColorStop(1, V.alpha(RED(), 0.16));
        g.fillStyle = gz; g.fillRect(A.X(L0), R1.y, R1.x + R1.w - A.X(L0), R1.h);
        V.text(g, 'longo demais: o ouro pune', R1.x + R1.w - 4, R1.y + 12, { size: 10, color: RED(), align: 'right' });
        const rd = S.round;
        const show = V.clamp(el / 0.2, 0, 1);
        rd.cands.forEach((c, i) => {
          if (i === rd.bi) return;
          const x = A.X(V.clamp(c.l, -3, 3.2)), y = A.Y(V.clamp(c.g, -6, 3));
          V.dot(g, x, y, rd.cands.length > 64 ? 2.2 : 3.2, V.alpha(C.b, 0.55 * show));
        });
        const bc = rd.cands[rd.bi];
        if (el > 0.25) {
          const x = A.X(V.clamp(bc.l, -3, 3.2)), y = A.Y(V.clamp(bc.g, -6, 3));
          V.glow(g, x, y, 18, C.d, 0.7); V.dot(g, x, y, 5, C.d);
          crown(g, x, y - 8, 7, C.d);
          V.text(g, 'escolhida pelo proxy', V.clamp(x, R1.x + 60, R1.x + R1.w - 60), y - 20 < R1.y + 30 ? y + 24 : y - 20, { size: 10.5, align: 'center', color: C.d, weight: 600 });
        }
        V.text(g, 'rodada: ' + rd.cands.length + ' candidatas', R1.x + 4, R1.y + 12, { size: 10.5, color: C.muted });
        // painel direito: dois gráficos empilhados (proxy em cima, ouro embaixo), mesmo eixo x
        const pts = S.res[S.ens];
        const allP = [], allG = [];
        ['1', 'media', 'min'].forEach((m) => S.res[m].forEach((p) => { allP.push(p.p); allG.push(p.g); }));
        const dmax = Math.sqrt(KL(256)) * 1.05;
        const hp = Math.round(R2.h * 0.36);
        const Rp = { x: R2.x, y: R2.y, w: R2.w, h: hp };
        const Rg = { x: R2.x, y: R2.y + hp + 36, w: R2.w, h: R2.h - hp - 36 };
        const p0 = Math.floor(Math.min(...allP)), p1 = Math.ceil(Math.max(...allP));
        const g0 = Math.floor((Math.min(...allG) - 0.1) * 2) / 2, g1 = Math.ceil((Math.max(...allG) + 0.15) * 2) / 2;
        const gStep = g1 - g0 > 2.5 ? 1 : 0.5;
        const gTicks = []; for (let v = g0; v <= g1 + 1e-9; v += gStep) gTicks.push(+v.toFixed(2));
        const BP = axes(g, Rp, { x0: 0, x1: dmax, y0: p0, y1: p1, xt: [0, 0.5, 1, 1.5, 2], yt: [p0, Math.round((p0 + p1) / 2), p1], fx: (v) => fmt(v, 1), fy: (v) => fmt(v, 0), yl: 'proxy (o que o RM vê)' });
        const BG = axes(g, Rg, { x0: 0, x1: dmax, y0: g0, y1: g1, xt: [0, 0.5, 1, 1.5, 2], yt: gTicks, fx: (v) => fmt(v, 1), fy: (v) => fmt(v, 1), xl: 'd = √KL(BoN ‖ inicial)', yl: 'ouro (qualidade real)' });
        const line = (B, key, col, lw) => { g.beginPath(); pts.forEach((p, i) => { const x = B.X(p.d), y = B.Y(p[key]); if (i) g.lineTo(x, y); else g.moveTo(x, y); }); g.strokeStyle = col; g.lineWidth = lw; g.stroke(); pts.forEach((p, i) => V.dot(g, B.X(p.d), B.Y(p[key]), i === S.ni ? 5.5 : 3, col)); };
        line(BP, 'p', C.b, 2.4);
        const fit = fitGao(pts);
        if (fit) { g.save(); g.beginPath(); g.rect(Rg.x, Rg.y, Rg.w, Rg.h); g.clip(); curve(g, (d) => fit.g0 + d * (fit.a - fit.b * d), 0, dmax, BG.X, BG.Y, { color: V.alpha(C.a, 0.8), lw: 1.5, dash: [5, 4] }); g.restore(); }
        line(BG, 'g', C.a, 2.6);
        const cur = pts[S.ni];
        V.line(g, BP.X(cur.d), Rp.y, BP.X(cur.d), Rg.y + Rg.h, { color: V.alpha(C.d, 0.6), lw: 1.4, dash: [3, 3] });
        const nRight = BP.X(cur.d) > Rp.x + Rp.w - 56;
        V.text(g, 'n = ' + cur.n, BP.X(cur.d) + (nRight ? -5 : 5), Rp.y + 10, { size: 10.5, font: 'mono', color: C.d, align: nRight ? 'right' : 'left' });
        let pi = 0; pts.forEach((p, i) => { if (p.g > pts[pi].g) pi = i; });
        const peak = pts[pi], last = pts[pts.length - 1];
        if (pi < pts.length - 1) V.text(g, '▲ pico', BG.X(peak.d), BG.Y(peak.g) - 10, { size: 10, align: 'center', color: C.a, weight: 600 });
        else V.text(g, 'sem pico até n = ' + last.n + ' ▸', BG.X(last.d) + 4, BG.Y(last.g) - 10, { size: 10, align: 'right', color: C.a, weight: 600 });
        if (fit && fit.b > 0) V.text(g, `Gao: d(α − βd), α=${fmt(fit.a, 2)}, β=${fmt(fit.b, 2)}`, Rg.x + Rg.w, Rg.y + Rg.h + 44, { size: 10.5, font: 'mono', align: 'right', color: C.a });
        else if (fit) V.text(g, 'ajuste de Gao sem β > 0: nada de superotimização até aqui', Rg.x + Rg.w, Rg.y + Rg.h + 44, { size: 10.5, align: 'right', color: C.a, maxW: Rg.w + 40 });
      });
    })();

    /* ================================================ FIG 9 — ORM, PRM, Math-Shepherd */
    (function () {
      const fig = root.querySelector('#fig-prm');
      const S = { mode: 'prm', p3: 0.6, agg: 'prod', bug: false, step: 2, N: 4, seed: 0, t0: 0, answers: null };
      // no celular, só o Math-Shepherd precisa da altura extra (árvore de continuações)
      const setH = adaptH(fig, 400, () => (S.mode === 'ms' ? 640 : S.mode === 'orm' ? 450 : 530));
      const STEPS_OK = ['6 fileiras × 8 = 48 bombons', 'comemos 6: sobram 48 − 6', '48 − 6 = 40 + 2', 'resposta: 42'];
      const STEPS_BUG = ['6 fileiras × 8 = 48 bombons', 'comemos 6: sobram 48 − 6', '48 − 6 = 44', '44 − 2 = 42 → resposta: 42'];
      const P_OK = () => [0.98, 0.95, S.p3, 0.97];
      const P_BUG = [0.98, 0.95, 0.08, 0.5];
      const SUCC = [0.92, 0.88, 0.75, 0.97];
      const bugOn = () => S.bug && S.mode !== 'ms';
      const steps = () => (bugOn() ? STEPS_BUG : STEPS_OK);
      const probs = () => (bugOn() ? P_BUG : P_OK());
      function sample() {
        if (S.step === 2 && S.N === 4 && S.seed === 0) { S.answers = [42, 17, 42, 42]; }
        else { const r = V.rng(100 + S.seed * 7 + S.step * 31 + S.N); const wrong = [17, 41, 44, 36, 48, 40]; S.answers = Array.from({ length: S.N }, () => (r() < SUCC[S.step] ? 42 : r.pick(wrong))); }
        upd();
      }
      const ctl = UI.controls(fig);
      nocase(UI.seg(ctl, { label: 'Modo', options: [{ v: 'orm', label: 'ORM' }, { v: 'prm', label: 'PRM' }, { v: 'ms', label: 'Math-Shepherd' }], value: 'prm', onChange: (v) => { S.mode = v; setH(); gP.show(v === 'prm'); gB.show(v !== 'ms'); gM.show(v === 'ms'); S.t0 = H.t; upd(); } }));
      const gB = UI.group(ctl), gP = UI.group(ctl), gM = UI.group(ctl);
      nocase(UI.toggle(gB.el, { label: 'traço com erro (resposta certa por acaso)', value: false, onChange: (v) => { S.bug = v; upd(); } }));
      nocase(UI.slider(gP.el, { label: 'p(passo 3 correto)', min: 0.05, max: 0.99, step: 0.01, value: S.p3, fmt: (v) => fmt(v, 2), onInput: (v) => { S.p3 = v; upd(); } }));
      nocase(UI.seg(gP.el, { label: 'Agregação', options: [{ v: 'prod', label: 'produto' }, { v: 'min', label: 'mínimo' }], value: 'prod', onChange: (v) => { S.agg = v; upd(); } }));
      nocase(UI.seg(gM.el, { label: 'A partir do passo', options: [{ v: 0, label: '1' }, { v: 1, label: '2' }, { v: 2, label: '3' }, { v: 3, label: '4' }], value: 2, onChange: (v) => { S.step = v; S.seed = 0; S.t0 = H.t; sample(); } }));
      nocase(UI.slider(gM.el, { label: 'N continuações', min: 2, max: 16, step: 1, value: 4, fmt: (v) => String(v), onInput: (v) => { S.N = v; S.seed = 0; S.t0 = H.t; sample(); } }));
      UI.button(gM.el, '🎲 sortear de novo', () => { S.seed++; S.t0 = H.t; sample(); });
      gM.show(false);
      const r1 = UI.readout(ctl, 'escore', '—'), r2 = UI.readout(ctl, 'detalhe', '—');
      nocase(ctl);
      function upd() {
        const p = probs();
        if (S.mode === 'orm') { r1.set('1 (✓) em todos os passos', S.bug ? 'warn' : 'ok'); r2.set(S.bug ? 'erro de traço invisível' : 'resposta = gabarito', S.bug ? 'bad' : 'ok'); }
        else if (S.mode === 'prm') { const prod = p.reduce((a, b) => a * b, 1), mn = Math.min(...p); r1.set((S.agg === 'prod' ? 'produto ' : 'mínimo ') + fmt(S.agg === 'prod' ? prod : mn, 4), (S.agg === 'prod' ? prod : mn) < 0.3 ? 'bad' : 'ok'); r2.set('produto ' + fmt(prod, 4) + ' · mín ' + fmt(mn, 2)); }
        else if (S.answers) { const k = S.answers.filter((a) => a === 42).length; r1.set('HE = ' + (k > 0 ? 1 : 0) + ' · SE = ' + fmt(k / S.answers.length, 2), k / S.answers.length < 0.5 ? 'warn' : 'ok'); r2.set(k + ' de ' + S.answers.length + ' chegam a 42'); }
      }
      sample();
      gB.show(true); gP.show(true);
      const H = V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const narrow = w < 640;
        const el = t - S.t0;
        const st = steps(), p = probs();
        const colW = narrow ? w - 24 : Math.min(360, w * 0.48);
        const x0 = 12, y0 = 18;
        chip(g, x0, y0, colW, 30, 'Problema: 6 fileiras de 8 bombons; comemos 6. Quantos sobram?', { fill: V.alpha(C.b, 0.14), stroke: V.alpha(C.b, 0.6), font: 'sans', size: narrow ? 10.5 : 11.5 });
        const sh = narrow ? 38 : 52, gap = narrow ? 8 : 14;
        const sy = (i) => y0 + 44 + i * (sh + gap);
        const ms = S.mode === 'ms';
        st.forEach((txt, i) => {
          const y = sy(i);
          const isSel = ms && i === S.step;
          const bad = bugOn() && i === 2;
          const col = isSel ? C.d : C.muted;
          V.box(g, x0, y, colW, sh, { r: 8, fill: V.alpha(isSel ? C.d : C.panel, isSel ? 0.16 : 0.6), stroke: V.alpha(col, isSel ? 0.9 : 0.4), lw: isSel ? 2 : 1 });
          V.text(g, 'passo ' + (i + 1), x0 + 10, y + 14, { size: 10, color: C.muted });
          const tmax = S.mode === 'ms' ? colW - 20 : colW - Math.min(90, colW * 0.24) - 70;
          V.text(g, txt, x0 + 10, y + sh - 10, { size: narrow ? 11 : 12.5, font: 'mono', color: bad ? RED() : C.ink, maxW: tmax });
          if (S.mode === 'orm') {
            const k = V.clamp((el - 0.3 - i * 0.12) / 0.4, 0, 1);
            if (k > 0) { g.globalAlpha = k; chip(g, x0 + colW - 70, y + sh / 2 - 11, 60, 22, '✓ 1', { fill: V.alpha(C.a, 0.22), stroke: C.a, size: 11.5, weight: 700 }); g.globalAlpha = 1; }
          } else if (S.mode === 'prm') {
            const bw = Math.min(90, colW * 0.24), bx = x0 + colW - bw - 50, by = y + sh / 2 - 6;
            const col2 = p[i] > 0.85 ? C.a : p[i] > 0.4 ? C.d : RED();
            V.box(g, bx, by, bw, 12, { r: 6, fill: V.alpha(C.muted, 0.15) });
            V.box(g, bx, by, Math.max(6, bw * p[i]), 12, { r: 6, fill: col2 });
            V.text(g, fmt(p[i], 2), x0 + colW - 10, y + sh / 2 + 1, { size: 11.5, font: 'mono', align: 'right', base: 'middle', color: col2, weight: 700 });
          }
          if (i < st.length - 1) V.arrow(g, x0 + 24, y + sh + 1, x0 + 24, y + sh + gap - 1, { color: C.muted, lw: 1.2, head: 5 });
        });
        const yEnd = sy(st.length);
        // painel direito
        const px = narrow ? 12 : x0 + colW + 34, py = narrow ? yEnd + 8 : y0, pw = narrow ? w - 24 : w - px - 14;
        if (S.mode === 'orm') {
          const th = V.wrap(g, 'O ORM só olha a resposta final: 42 = gabarito, então todos os passos recebem o mesmo rótulo 1.' + (S.bug ? ' O passo 3 está errado (48 − 6 ≠ 44) e o passo 4 “conserta” com um −2 inventado: o ORM não vê esse erro de traço.' : ' Até o passo 3, que é estranho, ganha ✓ sem ser examinado.'), px, py + 16, pw, { size: 12, color: C.ink, lineH: 17, maxLines: 7 });
          const cyO = py + 16 + th + 6;
          chip(g, px, cyO, Math.min(pw, 220), 30, 'rótulo de resultado: 1', { fill: V.alpha(C.a, 0.2), stroke: C.a, size: 12, weight: 700 });
          V.wrap(g, 'Uesato 2022: o erro de traço caiu de 14,0% para 3,4% com supervisão de processo.', px, cyO + 56, pw, { size: 11, color: C.muted, lineH: 15, maxLines: 3 });
        } else if (S.mode === 'prm') {
          const prod = p.reduce((a, b) => a * b, 1), mn = Math.min(...p);
          V.text(g, 'escore da solução', px, py + 14, { size: 12, color: C.ink, weight: 600 });
          const bw = Math.min(pw - 10, 260);
          let acc = 1;
          p.forEach((v, i) => {
            acc *= v;
            const y = py + 34 + i * 26;
            V.text(g, (i ? '× ' : '  ') + fmt(v, 2) + '  →  ' + fmt(acc, 4), px, y + 9, { size: 11.5, font: 'mono', color: C.ink });
            V.box(g, px + 170, y, Math.max(3, (bw - 170) * acc), 12, { r: 4, fill: V.alpha(C.b, 0.7) });
          });
          const y2 = py + 34 + 4 * 26 + 18;
          chip(g, px, y2, Math.min(pw, 240), 30, (S.agg === 'prod' ? 'produto = ' + fmt(prod, 4) : 'mínimo = ' + fmt(mn, 2)), { fill: V.alpha(C.d, 0.2), stroke: C.d, size: 12.5, weight: 700 });
          V.wrap(g, 'Um único passo duvidoso derruba a solução inteira. O produto também pune soluções longas com muitos passos “quase certos”.', px, y2 + 52, pw, { size: 11, color: C.muted, lineH: 15, maxLines: 4 });
        } else {
          // árvore de continuações
          const ys = sy(S.step) + sh / 2, xs = narrow ? px + 10 : px - 22;
          const N = S.answers.length;
          const treeTop = narrow ? py + 10 : py + 10, treeBot = narrow ? h - 70 : h - 70;
          const xe = narrow ? px + pw - 70 : px + pw - 70;
          const startX = narrow ? px + 30 : xs, startY = narrow ? treeTop + 4 : ys;
          if (narrow) {
            // conector: do passo escolhido, pela margem esquerda, até a raiz da árvore
            const mx = 5, cc = V.alpha(C.d, 0.9);
            V.line(g, x0, ys, mx, ys, { color: cc, lw: 2, dash: [4, 3] });
            V.line(g, mx, ys, mx, startY, { color: cc, lw: 2, dash: [4, 3] });
            V.arrow(g, mx, startY, startX - 6, startY, { color: cc, lw: 2, head: 6 });
          }
          S.answers.forEach((a, j) => {
            const ye = N === 1 ? (treeTop + treeBot) / 2 : treeTop + 14 + (j * (treeBot - treeTop - 28)) / (N - 1);
            const k = V.clamp((el - j * 0.12) / 0.9, 0, 1);
            const ok = a === 42;
            const p0 = [startX, startY], p3 = [xe, ye];
            const p1 = narrow ? [startX, startY + 30] : [startX + 60, startY], p2 = [xe - 80, ye];
            g.beginPath();
            for (let s = 0; s <= 24; s++) { const u = (s / 24) * k; const [bx, by] = V.bez(p0, p1, p2, p3, u); if (s) g.lineTo(bx, by); else g.moveTo(bx, by); }
            g.strokeStyle = V.alpha(ok ? C.a : RED(), 0.75); g.lineWidth = 1.6; g.setLineDash([5, 4]); g.lineDashOffset = -t * 20; g.stroke(); g.setLineDash([]);
            if (k >= 1) { chip(g, xe + 4, ye - 10, 40, 20, String(a), { fill: V.alpha(ok ? C.a : RED(), 0.22), stroke: ok ? C.a : RED(), size: 11.5, weight: 700 }); mark(g, xe + 56, ye, ok, 5); }
          });
          if (narrow) V.dot(g, startX, startY, 4, C.d);
          const k = S.answers.filter((a) => a === 42).length;
          V.text(g, 'HE = ' + (k > 0 ? 1 : 0) + '   SE = ' + k + '/' + N + ' = ' + fmt(k / N, 2), narrow ? px : px, h - 40, { size: 12.5, font: 'mono', color: C.d, weight: 700 });
          V.text(g, 'completador termina N soluções a partir do passo ' + (S.step + 1) + '; gabarito 42', narrow ? px : px, h - 20, { size: 10.5, color: C.muted, maxW: pw });
        }
      });
    })();

    /* ================================================ FIG 10 — juiz, posição e rótulo suave */
    (function () {
      const fig = root.querySelector('#fig-juiz');
      adaptH(fig, 440, 500);
      const S = { d: 0.5, b: 1.0, swap: true };
      const ctl = UI.controls(fig);
      nocase(UI.slider(ctl, { label: 'preferência real Δ (A − B)', min: -3, max: 3, step: 0.1, value: S.d, fmt: (v) => fmt(v, 1), onInput: (v) => { S.d = v; upd(); } }));
      nocase(UI.slider(ctl, { label: 'viés de posição b', min: 0, max: 2.5, step: 0.1, value: S.b, fmt: (v) => fmt(v, 1), onInput: (v) => { S.b = v; upd(); } }));
      nocase(UI.toggle(ctl, { label: 'trocar a ordem e tirar a média', value: true, onChange: (v) => { S.swap = v; upd(); } }));
      const rL = UI.readout(ctl, 'rótulo P(A ≻ B)', '—'), rT = UI.readout(ctl, 'verdade σ(Δ)', '—'), rV = UI.readout(ctl, 'veredito', '—');
      nocase(ctl);
      const calc = () => { const p1 = sig(S.d + S.b), p2 = sig(S.d - S.b), avg = (p1 + p2) / 2; return { p1, p2, avg, lab: S.swap ? avg : p1, truth: sig(S.d), cons: (p1 > 0.5) === (p2 > 0.5) }; };
      function upd() {
        const c = calc();
        rL.set(fmt(c.lab, 3), Math.abs(c.lab - c.truth) > 0.12 ? 'warn' : 'ok');
        rT.set(fmt(c.truth, 3));
        rV.set(!S.swap ? (c.p1 > 0.5 ? 'A (só uma ordem)' : 'B (só uma ordem)') : c.cons ? (c.p1 > 0.5 ? 'A, consistente' : 'B, consistente') : 'empate (inverteu)', !S.swap ? 'warn' : c.cons ? 'ok' : 'warn');
      }
      upd();
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const c = calc();
        const narrow = w < 640;
        const rows = S.swap ? [['A', 'B', c.p1], ['B', 'A', c.p2]] : [['A', 'B', c.p1]];
        const rowH = narrow ? 150 : 118;
        const top = 22;
        rows.forEach(([first, second, pA], ri) => {
          const y = top + ri * rowH;
          V.text(g, 'ordem ' + (ri + 1) + ': [1] = ' + first + ', [2] = ' + second, 12, y + 6, { size: 11.5, color: C.muted, weight: 600 });
          const cy = y + 46;
          const colA = (nm) => (nm === 'A' ? C.a : C.c);
          // slots
          const sw = narrow ? 52 : 70;
          chip(g, 12, cy - 30, sw, 24, '1: ' + first, { fill: V.alpha(colA(first), 0.2), stroke: colA(first), size: 11.5, weight: 700 });
          chip(g, 12, cy + 2, sw, 24, '2: ' + second, { fill: V.alpha(colA(second), 0.2), stroke: colA(second), size: 11.5, weight: 700 });
          // juiz
          const jx = narrow ? 12 + sw + 50 : 12 + sw + 70;
          V.arrow(g, 12 + sw + 6, cy - 2, jx - 26, cy - 2, { color: C.muted, lw: 1.4, head: 6 });
          const bob = Math.sin(t * 3 + ri) * 1.5;
          V.box(g, jx - 22, cy - 26 + bob, 44, 44, { r: 10, fill: V.alpha(C.b, 0.75), stroke: C.b });
          V.dot(g, jx - 8, cy - 8 + bob, 3, C.dark ? '#06140d' : '#fff'); V.dot(g, jx + 8, cy - 8 + bob, 3, C.dark ? '#06140d' : '#fff');
          V.line(g, jx - 8, cy + 6 + bob, jx + 8, cy + 6 + bob, { color: C.dark ? '#06140d' : '#fff', lw: 2 });
          V.line(g, jx, cy - 26 + bob, jx, cy - 34 + bob, { color: C.b, lw: 2 }); V.dot(g, jx, cy - 36 + bob, 3, C.d);
          V.text(g, 'juiz', jx, cy + 32, { size: 10.5, align: 'center', color: C.muted });
          // logits
          const lx = jx + 56, lw = narrow ? w - lx - 16 : Math.min(260, w * 0.3);
          const zFirst = (first === 'A' ? S.d / 2 : -S.d / 2), zSecond = (second === 'A' ? S.d / 2 : -S.d / 2);
          const zx = lx + lw / 2, sc = lw / 2 / 3.2;
          V.box(g, lx, cy - 23, lw, 16, { r: 4, fill: V.alpha(C.muted, 0.08) });
          V.box(g, lx, cy + 3, lw, 16, { r: 4, fill: V.alpha(C.muted, 0.08) });
          V.line(g, zx, cy - 30, zx, cy + 26, { color: C.muted, lw: 1 });
          V.text(g, 'logits', lx, cy - 34, { size: 10, color: C.muted });
          // token "1": logit base + viés
          const bar = (yy, z, col, extra) => {
            const x1 = zx + Math.min(0, z) * sc;
            if (extra) {
              V.box(g, x1, yy, Math.max(2, Math.abs(z) * sc), 6, { r: 2, fill: V.alpha(col, 0.85) });
              const xs = zx + z * sc, wb = Math.max(2, Math.min(extra * sc, lx + lw - xs));
              V.box(g, xs, yy + 8, wb, 6, { r: 2, fill: V.alpha(C.d, 0.9) });
              V.line(g, xs + wb, yy - 2, xs + wb, yy + 16, { color: C.ink, lw: 1.5 });
              V.text(g, '+b', xs + wb + 4, yy + 11, { size: 9.5, font: 'mono', base: 'middle', color: C.d, weight: 700 });
              return;
            }
            V.box(g, x1, yy, Math.max(2, Math.abs(z) * sc), 14, { r: 3, fill: V.alpha(col, 0.75) });
          };
          bar(cy - 22, zFirst, colA(first), S.b);
          bar(cy + 4, zSecond, colA(second), 0);
          if (!narrow) V.text(g, 'z = Δ/2 para A, −Δ/2 para B', lx + lw, cy + 34, { size: 9.5, font: 'mono', align: 'right', color: C.muted });
          V.text(g, '“1”', lx - 4, cy - 14, { size: 10.5, font: 'mono', align: 'right', base: 'middle', color: C.ink });
          V.text(g, '“2”', lx - 4, cy + 12, { size: 10.5, font: 'mono', align: 'right', base: 'middle', color: C.ink });
          // softmax → P(A)
          const ox = narrow ? 12 : lx + lw + 26, oy = narrow ? cy + 60 : cy - 10;
          const ow = narrow ? w - 24 : w - ox - 16;
          V.text(g, 'softmax → P(A ≻ B) = ' + fmt(pA, 3), ox, oy - 4, { size: 11, font: 'mono', color: C.ink, weight: 600 });
          V.box(g, ox, oy + 4, ow, 12, { r: 6, fill: V.alpha(C.c, 0.4) });
          V.box(g, ox, oy + 4, Math.max(6, ow * pA), 12, { r: 6, fill: C.a });
          const pulse = (t * 0.7 + ri * 0.5) % 1;
          V.dot(g, V.lerp(jx + 22, lx, pulse), cy - 2, 2.5, V.alpha(C.d, 1 - pulse));
        });
        // resumo final
        const sy = top + rows.length * rowH + (narrow ? 4 : 10);
        const bw = w - 24;
        const items = S.swap ? [['ordem 1', c.p1, C.muted], ['ordem 2', c.p2, C.muted], ['média (rótulo)', c.avg, C.d], ['verdade σ(Δ)', c.truth, C.b]] : [['rótulo (só ordem 1)', c.p1, C.d], ['verdade σ(Δ)', c.truth, C.b]];
        const gap = narrow ? 22 : 20;
        V.text(g, 'P(A ≻ B) — 0,5 no traço', 12, sy + 2, { size: 11, color: C.muted });
        const x50 = 12 + 130 + (bw - 130 - 50) * 0.5;
        items.forEach(([nm, v, col], i) => {
          const y = sy + 14 + i * gap;
          V.text(g, nm, 12, y + 8, { size: 11, color: C.ink });
          const bx = 12 + 130, bww = bw - 130 - 50;
          V.box(g, bx, y, bww, 10, { r: 5, fill: V.alpha(C.muted, 0.14) });
          V.box(g, bx, y, Math.max(5, bww * v), 10, { r: 5, fill: col });
          V.text(g, fmt(v, 3), bx + bww + 6, y + 9, { size: 11, font: 'mono', color: C.ink });
        });
        V.line(g, x50, sy + 10, x50, sy + 14 + items.length * gap, { color: C.ink, lw: 1, dash: [3, 3] });
        if (S.swap && !c.cons) V.text(g, 'as duas ordens discordam do vencedor → conte empate', 12, sy + 18 + items.length * gap, { size: 11, color: C.d, weight: 600 });
        if (!S.swap) V.wrap(g, 'Com uma ordem só, o viés de posição entra inteiro no rótulo, e o RM treinado nele herda o viés. Ligue a troca de ordem.', 12, sy + 24 + items.length * gap, w - 24, { size: 11.5, color: C.d, lineH: 16, maxLines: 3 });
      });
    })();
  },
});

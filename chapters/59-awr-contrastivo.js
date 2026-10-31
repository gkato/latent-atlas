LA.chapter({
  id: 'awr-contrastivo',
  part: 12,
  order: 59,
  title: 'AWR e aprendizado contrastivo',
  navTitle: 'AWR & contrastivo',
  subtitle: 'Duas estradas para o alinhamento sem PPO: amostrar, pesar e imitar (RWR, AWR, RAFT, ReST, rejection sampling) ou comparar e empurrar (InfoNCE, SLiC, CPL, unlikelihood). E como todas desembocam no mesmo ótimo que o DPO persegue.',
  minutes: 40,
  recipes: ['glm'],
  tags: ['AWR', 'RWR', 'AWAC', 'RAFT', 'ReST', 'rejection sampling', 'InfoNCE', 'SLiC-HF', 'CPL', 'unlikelihood', 'KL direto × reverso'],
  goals: [
    'Derivar por que AWR, RAFT/ReST e DPO perseguem o mesmo <b>ótimo inclinado</b> \\(\\pi^* \\propto \\pi_{\\text{ref}}\\,e^{r/\\beta}\\), e o que muda entre a projeção por <b>KL direto</b> (regressão) e por <b>KL reverso</b> (RL)',
    'Implementar o <b>SFT ponderado</b> (AWR, filtro, best-of-K) com vantagem centrada, temperatura e corte, e prever o efeito de \\(\\beta\\), \\(K\\) e da iteração',
    'Explicar InfoNCE, SLiC-HF, CPL e unlikelihood pela forma da perda e do gradiente, e ligar cada um ao DPO num mapa de parentesco',
    'Escolher entre regressão, contraste e RL para um problema real, sabendo o que monitorar',
  ],
  prereqs: ['dpo', 'rl-fundamentos', 'sft'],
  summary: [
    'O objetivo com KL tem ótimo \\(\\pi^* = \\pi_{\\text{ref}}e^{r/\\beta}/Z\\). O RL chega lá minimizando \\(\\mathbb D_{\\text{KL}}(\\pi_\\theta\\|\\pi^*)\\) (reverso, caça o modo); a regressão minimiza \\(\\mathbb D_{\\text{KL}}(\\pi^*\\|\\pi_\\theta)\\) (direto, cobre a massa), que é uma máxima verossimilhança com pesos \\(\\mathrm{softmax}(r/\\beta)\\) por prompt; o DPO inverte a fórmula e cancela \\(Z\\).',
    '<b>RWR → AWR</b>: maximizar a melhora sob restrição de KL dá \\(\\pi^* = \\mu\\,e^{(\\mathcal R - V)/\\beta}/Z\\) e duas regressões por iteração (crítico \\(V\\) e política ponderada), com \\(\\beta = 0{,}05\\) fixo e corte \\(\\omega_{\\max} = 20\\). A linha de base não muda os pesos normalizados de um mesmo estado; ela importa <i>entre</i> estados (escala do gradiente, interação com o corte).',
    'Nos LLMs, tudo vira <b>SFT ponderado</b>: \\(\\beta\\to0\\) é o best-of-K (RAFT, rejection sampling do Llama 2), \\(\\beta\\to\\infty\\) é SFT uniforme; ReST filtra por limiar, ReST-EM e RFT/STaR usam peso binário (RFT: 35,9% → 49,3% no GSM8K; R1: ~600 mil + 200 mil amostras). \\(\\mathbb D_{\\text{KL}}(\\pi_{\\text{bon}}\\|\\pi_{\\text{ref}}) \\le \\log n - (n-1)/n\\) (limite superior, folgado com recompensa binária), e iterar compõe inclinações (\\(\\beta/k\\)): a diversidade cai.',
    '<b>InfoNCE</b>: um positivo contra \\(N-1\\) negativos numa softmax; o gradiente sobe o positivo com \\(1-p_+\\) e desce cada negativo com \\(p_j\\); o ótimo é a razão de densidades e \\(I \\ge \\log N - \\mathcal L_N\\). Com 1 negativo e escore \\(\\beta\\log\\pi_\\theta/\\pi_{\\text{ref}}\\), é exatamente o DPO.',
    '<b>SLiC-HF</b>: dobradiça \\(\\max(0, \\delta - \\log P(y^+) + \\log P(y^-))\\) mais entropia cruzada como âncora; zera o gradiente além da margem, e sem referência pune a preferida mais longa: no par de brinquedo, a log-prob crua dá margem \\(m = -8\\) (par invertido, perda 9,0), enquanto a log-razão contra \\(\\pi_{\\text{ref}}\\) dá \\(\\Delta h = +3\\) (par na ordem certa).',
    '<b>CPL</b>: preferência por arrependimento + bijeção \\(A^* = \\alpha\\log\\pi^*\\) da máxima entropia = perda contrastiva supervisionada, off-policy; o DPO é caso especial, e no MDP de tokens vira um DPO com referência uniforme. O viés \\(\\lambda&lt;1\\) recompensa massa nas ações do dataset (1,8257 → 1,7839 com +1 nat).',
    '<b>Unlikelihood</b>: \\(-\\log(1-p(c))\\) nos candidatos proibidos, ancorada na NLL; sem âncora nem peso adaptativo, o contraste degenera (baseline do DPO). Regra prática: regressão para recompensas verificáveis e estabilidade; contraste para pares e para suprimir padrões; RL on-policy quando é preciso explorar.',
  ],
  quiz: [
    { q: 'Quatro respostas do mesmo prompt têm recompensas \\((1{,}0;\\ 0{,}0;\\ 0{,}5;\\ 0{,}2)\\). O que acontece com os pesos normalizados \\(e^{A/\\beta}\\) da AWR nos extremos de \\(\\beta\\)?', o: ['\\(\\beta\\to0\\): pesos iguais; \\(\\beta\\to\\infty\\): tudo na melhor', '\\(\\beta\\to0\\): tudo na resposta de recompensa 1,0 (best-of-K, como o RAFT); \\(\\beta\\to\\infty\\): pesos iguais (SFT uniforme)', 'Os pesos ficam proporcionais à recompensa em qualquer \\(\\beta\\)', '\\(\\beta\\) não afeta os pesos normalizados, só os crus'], a: 1, e: 'Os pesos normalizados são uma softmax de \\(A/\\beta\\). Com \\(\\beta = 0{,}05\\) dão \\((1;0;0;0)\\) (sem corte), com \\(\\beta = 10\\) dão \\((0{,}2646;\\ 0{,}2394;\\ 0{,}2517;\\ 0{,}2443)\\). “Proporcional à recompensa” é outra escolha (RWR/ReST-EM com \\(f\\) identidade), não o limite da exponencial.' },
    { q: 'Para um <b>mesmo prompt</b>, RWR (\\(e^{R/\\beta}\\)) e AWR (\\(e^{(R-V)/\\beta}\\)) dão os mesmos pesos normalizados. Onde a linha de base \\(V\\) faz diferença, então?', o: ['Em lugar nenhum: o paper da AWR só renomeou a RWR', 'Só quando a recompensa é negativa', 'Só no cálculo de \\(Z(x)\\), que a AWR precisa estimar explicitamente', 'Entre estados ou prompts diferentes: sem \\(V\\), estados de retorno alto dominam o gradiente e, com \\(\\beta\\) pequeno, todas as suas ações batem no corte \\(\\omega_{\\max}\\) e ficam com pesos iguais'], a: 3, e: 'O fator \\(e^{-V/\\beta}\\) é comum a todas as ações de um estado e cancela na normalização. Entre estados ele não cancela: centrar pela vantagem iguala a escala do “quanto há a aprender” em cada estado e impede o corte de achatar os pesos de um estado fácil. Recompensas negativas não mudam nada disso, e a AWR nunca estima \\(Z\\) explicitamente.' },
    { q: 'A estrada da regressão (AWR, RAFT, ReST) minimiza qual divergência até \\(\\pi^*\\), e com que temperamento?', o: ['\\(\\mathbb D_{\\text{KL}}(\\pi^*\\|\\pi_\\theta)\\): cobre a massa de todos os modos bons e não sai do suporte das amostras', '\\(\\mathbb D_{\\text{KL}}(\\pi_\\theta\\|\\pi^*)\\): caça um modo e pode explorar respostas novas', 'A distância de variação total, por isso dispensa pesos', 'Nenhuma: é só SFT, sem relação com o objetivo de RL'], a: 0, e: '\\(\\mathbb D_{\\text{KL}}(\\pi^*\\|\\pi_\\theta) = \\text{const} - \\mathbb E_{\\pi_{\\text{ref}}}[\\tfrac{e^{r/\\beta}}{Z}\\log\\pi_\\theta]\\): uma máxima verossimilhança ponderada, como o SFT, que é mass-covering. O KL reverso é o que o RL com KL minimiza (\\(\\mathbb E[r] - \\beta\\mathrm{KL}(\\pi\\|\\pi_{\\text{ref}}) = \\beta\\log Z - \\beta\\mathrm{KL}(\\pi\\|\\pi^*)\\)).' },
    { q: 'Numa InfoNCE com o positivo de escore 2,0 e um único negativo de escore 1,0, quanto vale a perda, e o que ela é?', o: ['0,6931: é a entropia de uma moeda', '0,5147: é o limite de informação mútua', '0,3133 \\(= -\\log\\sigma(1{,}0)\\): com escore \\(\\beta\\log\\pi_\\theta/\\pi_{\\text{ref}}\\), é a perda do DPO', '1,0: é a diferença de escores'], a: 2, e: '\\(-\\log\\frac{e^{2}}{e^{2}+e^{1}} = -\\log\\sigma(2-1) = 0{,}3133\\). Com \\(N=2\\), a softmax “um contra muitos” vira a logística do Bradley–Terry. 0,5147 é a perda com os três negativos (1,0; 0,5; −0,5), e o limite de MI nesse caso é \\(\\log 4 - 0{,}5147 = 0{,}8716\\).' },
    { q: 'A preferida tem 20 tokens e log-prob \\(-25\\); a preterida, 10 tokens e \\(-17\\); a referência dá \\(-26\\) e \\(-15\\). Por que o SLiC-HF (\\(\\delta = 1\\)) trata o par como muito errado, enquanto a log-razão contra \\(\\pi_{\\text{ref}}\\) o acha na ordem certa?', o: ['Porque o SLiC-HF usa a média por token', 'Porque o SLiC-HF compara log-probs <b>cruas</b>: a margem é \\(m = -25 - (-17) = -8\\) (par invertido), já que a resposta mais longa é menos provável como sequência; a log-razão desconta o comprimento e dá \\(\\Delta h = +3\\)', 'Porque a margem \\(\\delta\\) do SLiC-HF é 9', 'Porque a referência inverte a ordem de todos os pares'], a: 1, e: 'SLiC-HF: \\(\\max(0, 1 + 8) = 9{,}0\\), gradiente 1 por unidade de margem. Com a referência, \\(\\Delta h = (-25+26) - (-17+15) = +3\\): par na ordem certa. A <code>hinge</code> do TRL ainda dá \\(\\max(0, 1 - 0{,}1\\cdot3) = 0{,}7\\), porque a margem efetiva em \\(\\Delta h\\) é \\(1/\\beta = 10\\), com gradiente de só \\(\\beta = 0{,}1\\) por unidade. Comparar 9,0 com 0,7 mistura escalas; o que muda é o sinal da margem.' },
    { q: 'No CPL com o regularizador conservador (\\(\\lambda = 0{,}5\\)), o que acontece ao somar 1 nat às log-probs dos dois segmentos?', o: ['Nada, como no CPL vanilla: só a diferença importa', 'A perda sobe, porque o negativo ficou mais provável', 'A perda vai a zero', 'A perda cai (1,8257 → 1,7839): o regularizador recompensa pôr massa nas ações vistas no dataset'], a: 3, e: 'Com \\(\\lambda = 1\\), a perda depende só de \\(S^+ - S^-\\) e fica em 1,1711. Com \\(\\lambda&lt;1\\), o termo do negativo encolhe, e subir os dois pela mesma quantidade aumenta \\(S^+ - \\lambda S^-\\). Com dados finitos, isso escolhe, entre os infinitos minimizadores, os que ficam perto do dataset.' },
  ],
  mount(root, LA) {
    const V = LA.viz, UI = LA.ui;
    const RED = () => V.C().red || V.C().c;
    const fmt = (x, d = 2) => { const s = (+x).toFixed(d); return (s[0] === '-' && +s !== 0 ? '−' + s.slice(1) : s.replace('-', '')).replace('.', ','); };
    const fmtS = (x, d = 2) => (x > 0 ? '+' : '') + fmt(x, d);
    function fmtW(x) {
      if (!isFinite(x)) return '∞';
      if (x === 0) return '0';
      const ax = Math.abs(x);
      if (ax >= 1e5) { const e = Math.floor(Math.log10(ax)); return fmt(x / Math.pow(10, e), 1) + 'e' + e; }
      if (ax >= 1000) return (x < 0 ? '−' : '') + String(Math.round(ax)).replace(/\B(?=(\d{3})+(?!\d))/g, '.');
      if (ax >= 100) return fmt(x, 1);
      if (ax >= 0.01) return fmt(x, ax >= 10 ? 2 : 3);
      const e = Math.floor(Math.log10(ax)); return fmt(x / Math.pow(10, e), 1) + 'e' + e;
    }
    const sig = (z) => 1 / (1 + Math.exp(-z));
    const lse = (xs) => { const m = Math.max(...xs); return m + Math.log(xs.reduce((s, x) => s + Math.exp(x - m), 0)); };
    const sampleCat = (d, u) => { let s = 0; for (let i = 0; i < d.length; i++) { s += d[i]; if (u < s) return i; } return d.length - 1; };
    const ent = (p) => -p.reduce((s, x) => s + (x > 1e-12 ? x * Math.log(x) : 0), 0);
    const kl = (a, b) => a.reduce((s, x, i) => s + (x > 1e-15 ? x * Math.log(x / b[i]) : 0), 0);
    /** tira o uppercase do CSS de rótulos com símbolos matemáticos (β, λ, K…) */
    const nocase = (c) => { const el = c && (c.el || c); if (el && el.querySelectorAll) el.querySelectorAll('label, span').forEach((n) => { if (!n.classList.contains('val')) n.style.textTransform = 'none'; }); return c; };
    function adaptH(fig, wide, narrow, bp = 640) {
      const st = fig.querySelector('.la-fig-stage');
      const set = () => { const want = (fig.clientWidth < bp ? narrow : wide) + 'px'; if (st.style.height !== want) st.style.height = want; };
      set();
      const ro = new ResizeObserver(set); ro.observe(fig);
      if (LA.onCleanup) LA.onCleanup(() => ro.disconnect());
    }
    function tw(g, s, size, font = 'sans', weight = 400) { g.font = V.font(size, font, weight); return g.measureText(String(s)).width; }
    /** personagem: quadrado arredondado com olhinhos (resposta, token, exemplo) */
    function tokChar(g, x, y, s, col, o = {}) {
      const C = V.C();
      if (o.alpha != null) g.globalAlpha = o.alpha;
      V.box(g, x - s / 2, y - s / 2, s, s, { r: s * 0.3, fill: o.fill || V.alpha(col, 0.85), stroke: o.stroke || col, lw: o.lw || 1.2 });
      if (s >= 11) {
        const eye = C.bg; // olhos na cor do fundo: escuros no tema escuro, claros no claro
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
    function chip(g, x, y, w, h, label, o = {}) {
      const C = V.C();
      V.box(g, x, y, w, h, { r: o.r == null ? 6 : o.r, fill: o.fill || V.alpha(C.muted, 0.12), stroke: o.stroke || V.alpha(C.muted, 0.4), lw: o.lw || 1.2 });
      if (label != null) V.text(g, label, x + w / 2, y + h / 2 + 0.5, { size: o.size || 11.5, font: o.font || 'mono', weight: o.weight || 500, color: o.color || C.ink, align: 'center', base: 'middle', maxW: w - 4 });
    }
    function mark(g, x, y, ok, s = 6) {
      const C = V.C();
      g.lineWidth = 2.2; g.strokeStyle = ok ? C.e : RED(); g.lineCap = 'round'; g.beginPath();
      if (ok) { g.moveTo(x - s, y); g.lineTo(x - s * 0.3, y + s * 0.7); g.lineTo(x + s, y - s * 0.8); }
      else { g.moveTo(x - s * 0.8, y - s * 0.8); g.lineTo(x + s * 0.8, y + s * 0.8); g.moveTo(x + s * 0.8, y - s * 0.8); g.lineTo(x - s * 0.8, y + s * 0.8); }
      g.stroke(); g.lineCap = 'butt';
    }
    /** eixos simples: devolve mapeamentos X(v), Y(v) */
    function axes(g, x, y, w, h, o) {
      const C = V.C();
      const X = (v) => x + ((v - o.xmin) / (o.xmax - o.xmin)) * w;
      const Y = (v) => y + h - ((v - o.ymin) / (o.ymax - o.ymin)) * h;
      g.save();
      (o.yt || []).forEach((v) => { V.line(g, x, Y(v), x + w, Y(v), { color: C.grid, lw: 1 }); V.text(g, o.yf ? o.yf(v) : fmt(v, 1), x - 5, Y(v), { size: 10.5, font: 'mono', color: C.muted, align: 'right', base: 'middle' }); });
      (o.xt || []).forEach((v) => { V.line(g, X(v), y + h, X(v), y + h + 4, { color: C.muted, lw: 1 }); V.text(g, o.xf ? o.xf(v) : fmt(v, 1), X(v), y + h + 15, { size: 10.5, font: 'mono', color: C.muted, align: 'center' }); });
      V.line(g, x, y + h, x + w, y + h, { color: C.muted, lw: 1.2 });
      V.line(g, x, y, x, y + h, { color: C.muted, lw: 1.2 });
      if (o.xl) V.text(g, o.xl, x + w, y + h + 29, { size: 11, color: C.muted, align: 'right' });
      if (o.yl) V.text(g, o.yl, x + 4, y - 7, { size: 11, color: C.muted, align: 'left' });
      g.restore();
      return { X, Y };
    }
    function polyline(g, pts, col, lw = 2, dash) {
      if (pts.length < 2) return;
      g.beginPath(); g.moveTo(pts[0][0], pts[0][1]); for (let i = 1; i < pts.length; i++) g.lineTo(pts[i][0], pts[i][1]);
      g.strokeStyle = col; g.lineWidth = lw; if (dash) g.setLineDash(dash); g.stroke(); g.setLineDash([]);
    }
    const logSl = (lo, hi) => ({ to: (v) => Math.exp(Math.log(lo) + (Math.log(hi) - Math.log(lo)) * v / 100), from: (b) => Math.round(100 * (Math.log(b) - Math.log(lo)) / (Math.log(hi) - Math.log(lo))) });

    /* ======================================== FIG 1 — a mesa inclinada */
    (function () {
      const fig = root.querySelector('#fig-inclinacao');
      adaptH(fig, 420, 480);
      const N = 6, NG = 60, COLS = 3, ROWS = NG / COLS;
      const NAMES = ['curta e seca', 'educada', 'errada', 'correta', 'evasiva', 'brilhante e rara'];
      const PREF = [0.28, 0.24, 0.18, 0.14, 0.11, 0.05];
      const R = [0.3, 0.6, -0.8, 1.0, 0.0, 1.3];
      const VREF = PREF.reduce((s, p, i) => s + p * R[i], 0);
      const BS = logSl(0.01, 100);
      const S = { beta: 1, base: false, disp: null };
      const target = () => V.softmax(PREF.map((p, i) => Math.log(p) + R[i] / S.beta));
      S.disp = target(); // barras e grãos já nascem em π* do β inicial
      function alloc(p) {
        const raw = p.map((x) => x * NG), fl = raw.map(Math.floor);
        let rest = NG - fl.reduce((a, b) => a + b, 0);
        raw.map((x, i) => [x - fl[i], i]).sort((a, b) => b[0] - a[0]).forEach(([, i]) => { if (rest > 0) { fl[i]++; rest--; } });
        return fl;
      }
      const grains = [];
      let counts = alloc(target()); // os grãos são a massa de π*, não de π_ref
      counts.forEach((n, b) => { for (let k = 0; k < n; k++) grains.push({ bar: b, slot: k, u: 1, fx: 0, fy: 0, x: null, y: null }); });
      function reassign() {
        const want = alloc(target());
        const byBar = Array.from({ length: N }, () => []);
        grains.forEach((gr) => byBar[gr.bar].push(gr));
        byBar.forEach((arr) => arr.sort((a, b) => a.slot - b.slot));
        const pool = [];
        byBar.forEach((arr, b) => { while (arr.length > want[b]) pool.push(arr.pop()); });
        byBar.forEach((arr, b) => { while (arr.length < want[b] && pool.length) { const gr = pool.pop(); if (gr.x != null) { gr.fx = gr.x; gr.fy = gr.y; gr.u = 0; } gr.bar = b; arr.push(gr); } });
        byBar.forEach((arr) => arr.forEach((gr, k) => { if (gr.slot !== k && gr.u >= 1 && gr.x != null) { gr.fx = gr.x; gr.fy = gr.y; gr.u = 0; } gr.slot = k; }));
        counts = want;
      }
      const ctl = UI.controls(fig);
      nocase(UI.slider(ctl, { label: 'β (temperatura)', min: 0, max: 100, step: 1, value: BS.from(1), fmt: (v) => fmt(BS.to(v), BS.to(v) < 0.1 ? 3 : 2), onInput: (v) => { S.beta = BS.to(v); reassign(); upd(); } }));
      UI.toggle(ctl, { label: 'subtrair a linha de base V = E_ref[r]', value: false, onChange: (v) => { S.base = v; upd(); } });
      const rE = UI.readout(ctl, 'E[r] sob π*', '—'), rK = UI.readout(ctl, 'KL(π*‖π_ref)', '—'), rH = UI.readout(ctl, 'entropia H(π*)', '—'), rZ = UI.readout(ctl, 'Z(x)', '—');
      function upd() {
        const p = target();
        rE.set(fmt(p.reduce((s, x, i) => s + x * R[i], 0), 3));
        rK.set(fmt(kl(p, PREF), 3) + ' nat');
        rH.set(fmt(ent(p), 3) + ' nat');
        const Z = PREF.reduce((s, x, i) => s + x * Math.exp((R[i] - (S.base ? VREF : 0)) / S.beta), 0);
        rZ.set(fmtW(Z));
      }
      upd();
      V.fig(fig, (g, w, h, t, dt) => {
        const C = V.C();
        const narrow = w < 560;
        const L = 34, Rr = 10, y0 = 30;
        const Hb = Math.round(h * (narrow ? 0.5 : 0.48));
        const slotW = (w - L - Rr) / N, bw = Math.min(slotW * 0.7, 84);
        const bx = (i) => L + slotW * i + (slotW - bw) / 2;
        const tgt = target();
        S.disp = S.disp.map((x, i) => x + (tgt[i] - x) * Math.min(1, dt * 5));
        V.text(g, narrow ? 'π* ∝ π_ref · e^(r/β)' : 'π*(y|x) = softmax( log π_ref + r/β )', L, 16, { size: narrow ? 11 : 12, font: 'mono', color: C.muted });
        V.text(g, 'β = ' + fmt(S.beta, S.beta < 0.1 ? 3 : 2), w - Rr, 16, { size: 12, font: 'mono', color: C.d, align: 'right', weight: 600 });
        // eixo
        [0, 0.25, 0.5, 0.75, 1].forEach((v) => { const yy = y0 + Hb - v * Hb; V.line(g, L, yy, w - Rr, yy, { color: C.grid, lw: 1 }); V.text(g, fmt(v, 2), L - 4, yy, { size: 10, font: 'mono', color: C.muted, align: 'right', base: 'middle' }); });
        for (let i = 0; i < N; i++) {
          const x = bx(i);
          const hp = S.disp[i] * Hb;
          g.fillStyle = V.alpha(C.a, 0.22); V.rr(g, x, y0 + Hb - hp, bw, Math.max(1, hp), 5); g.fill();
          g.strokeStyle = V.alpha(C.a, 0.9); g.lineWidth = 1.4; g.stroke();
          const hr = PREF[i] * Hb;
          g.setLineDash([4, 3]); g.strokeStyle = C.b; g.lineWidth = 1.6; V.rr(g, x - 3, y0 + Hb - hr, bw + 6, hr, 5); g.stroke(); g.setLineDash([]);
          V.text(g, fmt(S.disp[i], 2), x + bw / 2, y0 + Hb - Math.max(hp, hr) - 6, { size: 10.5, font: 'mono', color: C.ink, align: 'center' });
        }
        // grãos
        const rad = Math.max(1.6, Math.min(bw / (COLS * 2) - 0.6, Hb / (ROWS * 2) - 0.4));
        grains.forEach((gr) => {
          const col = gr.slot % COLS, row = Math.floor(gr.slot / COLS);
          const tx = bx(gr.bar) + (bw * (col + 0.5)) / COLS, ty = y0 + Hb - (row + 0.5) * (Hb / ROWS);
          if (gr.x == null) { gr.x = tx; gr.y = ty; }
          if (gr.u < 1) {
            gr.u = Math.min(1, gr.u + dt * 1.4);
            const e = V.ease.inOut(gr.u);
            gr.x = V.lerp(gr.fx, tx, e); gr.y = V.lerp(gr.fy, ty, e) - Math.sin(Math.PI * e) * Math.min(60, Math.abs(tx - gr.fx) * 0.35 + 14);
          } else { gr.x = tx; gr.y = ty; }
          V.dot(g, gr.x, gr.y, rad, gr.u < 1 ? C.d : V.alpha(C.a, 0.95));
        });
        // rótulos
        const ly = y0 + Hb + 15;
        for (let i = 0; i < N; i++) {
          const cx = bx(i) + bw / 2;
          V.text(g, 'y' + (i + 1), cx, ly, { size: 12, font: 'mono', weight: 600, color: C.ink, align: 'center' });
          if (!narrow) V.text(g, NAMES[i], cx, ly + 14, { size: 10.5, color: C.muted, align: 'center', maxW: slotW - 4 });
          const rc = R[i] >= 0.8 ? C.a : R[i] < 0 ? C.c : C.ink;
          V.text(g, 'r=' + fmtS(R[i], 1), cx, ly + (narrow ? 14 : 28), { size: 10.5, font: 'mono', color: rc, align: 'center', maxW: slotW });
        }
        // fator de inclinação (escala log)
        const fy0 = ly + (narrow ? 40 : 54), fH = h - fy0 - 20;
        if (fH > 30) {
          const mid = fy0 + fH / 2, D = 1.5;
          V.text(g, S.base ? 'fator e^((r−V)/β)  (escala log)' : 'fator e^(r/β)  (escala log)', L, fy0 - 4, { size: 10.5, font: 'mono', color: C.muted });
          V.line(g, L, mid, w - Rr, mid, { color: C.muted, lw: 1, dash: [3, 3] });
          V.text(g, '×1', L - 4, mid, { size: 10, font: 'mono', color: C.muted, align: 'right', base: 'middle' });
          for (let i = 0; i < N; i++) {
            const lf = (R[i] - (S.base ? VREF : 0)) / S.beta / Math.LN10;
            const cl = V.clamp(lf, -D, D);
            const hh = (cl / D) * (fH / 2 - 6);
            const x = bx(i) + bw * 0.2, ww = bw * 0.6;
            g.fillStyle = V.alpha(lf >= 0 ? C.d : C.c, 0.75);
            V.rr(g, x, hh >= 0 ? mid - hh : mid, ww, Math.max(1.5, Math.abs(hh)), 3); g.fill();
            if (Math.abs(lf) > D) V.text(g, lf > 0 ? '▲' : '▼', x + ww / 2, lf > 0 ? mid - hh - 3 : mid - hh + 11, { size: 10, color: lf > 0 ? C.d : C.c, align: 'center' });
            const lab = Math.abs(lf) < 4 ? '×' + fmtW(Math.pow(10, lf)) : '×10^' + Math.round(lf);
            const above = hh >= 0;
            V.text(g, lab, x + ww / 2, hh >= 0 ? Math.min(h - 6, mid + 13) : Math.max(fy0 + 8, mid - 6), { size: 10, font: 'mono', color: C.ink, align: 'center', maxW: slotW });
          }
        }
      });
    })();

    /* ======================================== FIG 2 — KL direto × reverso */
    (function () {
      const fig = root.querySelector('#fig-direcao-kl');
      adaptH(fig, 400, 640);
      const NP = 90, xs = Array.from({ length: NP }, (_, i) => (i + 0.5) / NP);
      const SMIN = 0.02, SMAX = 0.42;
      const S = { mode: 'fwd', sep: 0.44, side: 0, P: null, heat: null, hmin: 0, hmax: 1 };
      function makeP() {
        const c1 = 0.5 - S.sep / 2, c2 = 0.5 + S.sep / 2, s = 0.055;
        const p = xs.map((x) => 0.55 * Math.exp(-0.5 * ((x - c1) / s) ** 2) + 0.45 * Math.exp(-0.5 * ((x - c2) / s) ** 2) + 2e-4);
        const z = p.reduce((a, b) => a + b, 0); S.P = p.map((x) => x / z);
      }
      function model(mu, ls) { const sg = Math.exp(ls); const q = xs.map((x) => Math.exp(-0.5 * ((x - mu) / sg) ** 2) + 1e-12); const z = q.reduce((a, b) => a + b, 0); return q.map((x) => x / z); }
      const loss = (mu, ls, mode = S.mode) => { const q = model(mu, ls); return mode === 'fwd' ? kl(S.P, q) : kl(q, S.P); };
      const HC = 30, HR = 18;
      function makeHeat() {
        const H = []; let mn = Infinity, mx = -Infinity;
        for (let r = 0; r < HR; r++) {
          const row = []; const ls = Math.log(SMAX) - (r + 0.5) / HR * (Math.log(SMAX) - Math.log(SMIN));
          for (let c = 0; c < HC; c++) { const v = Math.log(loss((c + 0.5) / HC, ls) + 1e-3); row.push(v); mn = Math.min(mn, v); mx = Math.max(mx, v); }
          H.push(row);
        }
        S.heat = H; S.hmin = mn; S.hmax = mx;
      }
      const opt = { mu: 0.2, ls: Math.log(0.1), m: [0, 0], v: [0, 0], it: 0, trail: [] };
      function restart() { S.side = 1 - S.side; opt.mu = S.side ? 0.78 : 0.22; opt.ls = Math.log(0.09); opt.m = [0, 0]; opt.v = [0, 0]; opt.it = 0; opt.trail = [[opt.mu, opt.ls]]; }
      function step() {
        const e = 1e-4, L0 = loss(opt.mu, opt.ls);
        const g = [(loss(opt.mu + e, opt.ls) - L0) / e, (loss(opt.mu, opt.ls + e) - L0) / e];
        const lr = [0.012, 0.03];
        for (let k = 0; k < 2; k++) { opt.m[k] = 0.85 * opt.m[k] + 0.15 * g[k]; opt.v[k] = 0.99 * opt.v[k] + 0.01 * g[k] * g[k]; }
        const t = opt.it + 1, mh = (k) => opt.m[k] / (1 - Math.pow(0.85, t)), vh = (k) => opt.v[k] / (1 - Math.pow(0.99, t));
        opt.mu = V.clamp(opt.mu - lr[0] * mh(0) / (Math.sqrt(vh(0)) + 1e-8), 0.01, 0.99);
        opt.ls = V.clamp(opt.ls - lr[1] * mh(1) / (Math.sqrt(vh(1)) + 1e-8), Math.log(SMIN), Math.log(SMAX));
        opt.it++;
        if (opt.it % 3 === 0) opt.trail.push([opt.mu, opt.ls]);
      }
      makeP(); makeHeat(); S.side = 1; restart();
      const ctl = UI.controls(fig);
      UI.seg(ctl, { options: [{ v: 'fwd', label: 'KL direto (regressão)' }, { v: 'rev', label: 'KL reverso (RL)' }], value: 'fwd', onChange: (v) => { S.mode = v; makeHeat(); S.side = 1 - S.side; restart(); } });
      UI.slider(ctl, { label: 'separação dos picos', min: 0.15, max: 0.6, step: 0.01, value: S.sep, fmt: (v) => fmt(v, 2), onInput: (v) => { S.sep = v; makeP(); makeHeat(); S.side = 1 - S.side; restart(); } });
      UI.button(ctl, '↺ recomeçar do outro lado', () => restart());
      const rF = UI.readout(ctl, 'KL direto', '—'), rR = UI.readout(ctl, 'KL reverso', '—'), rP = UI.readout(ctl, 'μ ; σ', '—');
      let frame = 0;
      V.fig(fig, (g, w, h, t, dt) => {
        const C = V.C();
        if (!LA.paused && opt.it < 900) for (let k = 0; k < 3; k++) step();
        const q = model(opt.mu, opt.ls);
        if ((frame++ % 6) === 0) { rF.set(fmt(kl(S.P, q), 3)); rR.set(fmt(kl(q, S.P), 3)); rP.set(fmt(opt.mu, 2) + ' ; ' + fmt(Math.exp(opt.ls), 3)); }
        const narrow = w < 640;
        const A = narrow ? { x: 34, y: 34, w: w - 48, h: h * 0.42 } : { x: 34, y: 34, w: w * 0.56 - 40, h: h - 90 };
        const B = narrow ? { x: 40, y: A.y + A.h + 64, w: w - 56, h: h - (A.y + A.h + 64) - 40 } : { x: w * 0.56 + 34, y: 34, w: w * 0.44 - 50, h: h - 90 };
        // painel A: distribuições
        const pmax = Math.max(...S.P, ...q) * 1.12;
        const ax = axes(g, A.x, A.y, A.w, A.h, { xmin: 0, xmax: 1, ymin: 0, ymax: pmax, xt: [0, 0.5, 1], yt: [], xl: 'espaço de respostas (ilustrativo)' });
        g.beginPath(); g.moveTo(ax.X(xs[0]), ax.Y(0)); xs.forEach((x, i) => g.lineTo(ax.X(x), ax.Y(S.P[i]))); g.lineTo(ax.X(xs[NP - 1]), ax.Y(0)); g.closePath();
        g.fillStyle = V.alpha(C.b, 0.32); g.fill(); g.strokeStyle = C.b; g.lineWidth = 1.4; g.stroke();
        g.beginPath(); g.moveTo(ax.X(xs[0]), ax.Y(0)); xs.forEach((x, i) => g.lineTo(ax.X(x), ax.Y(q[i]))); g.lineTo(ax.X(xs[NP - 1]), ax.Y(0)); g.closePath();
        g.fillStyle = V.alpha(C.a, 0.18); g.fill();
        polyline(g, xs.map((x, i) => [ax.X(x), ax.Y(q[i])]), C.a, 2.6);
        V.text(g, 'alvo π*', A.x + 8, A.y + 4, { size: 11.5, color: C.b, weight: 600, base: 'top' });
        V.text(g, 'modelo π_θ (um morro só)', A.x + 8, A.y + 20, { size: 11.5, color: C.a, weight: 600, base: 'top' });
        V.text(g, S.mode === 'fwd' ? 'min KL(π* ‖ π_θ): cobre a massa' : 'min KL(π_θ ‖ π*): caça o modo', A.x + A.w, A.y - 12, { size: 12, font: 'mono', color: S.mode === 'fwd' ? C.a : C.f, align: 'right', weight: 600 });
        // painel B: mapa de calor da perda
        if (B.h > 40) {
          const cw = B.w / HC, ch = B.h / HR;
          for (let r = 0; r < HR; r++) for (let c = 0; c < HC; c++) {
            const tt = (S.heat[r][c] - S.hmin) / (S.hmax - S.hmin || 1);
            g.fillStyle = V.cmap(1 - tt, 'seq'); g.fillRect(B.x + c * cw, B.y + r * ch, cw + 0.5, ch + 0.5);
          }
          g.strokeStyle = C.muted; g.lineWidth = 1; g.strokeRect(B.x, B.y, B.w, B.h);
          const PX = (mu) => B.x + mu * B.w, PY = (ls) => B.y + (Math.log(SMAX) - ls) / (Math.log(SMAX) - Math.log(SMIN)) * B.h;
          polyline(g, opt.trail.map(([m, l]) => [PX(m), PY(l)]), C.f, 2);
          V.glow(g, PX(opt.mu), PY(opt.ls), 12, C.f, 0.8); V.dot(g, PX(opt.mu), PY(opt.ls), 4, C.f);
          V.text(g, 'perda no plano (μ, σ) — claro = menor', B.x, B.y - 10, { size: 11, color: C.muted });
          V.text(g, 'μ →', B.x + B.w, B.y + B.h + 14, { size: 11, font: 'mono', color: C.muted, align: 'right' });
          V.text(g, '0', B.x, B.y + B.h + 14, { size: 10.5, font: 'mono', color: C.muted });
          V.text(g, fmt(SMAX, 2), B.x - 4, B.y + 8, { size: 10, font: 'mono', color: C.muted, align: 'right' });
          V.text(g, 'σ', B.x - 4, B.y + B.h / 2, { size: 12, font: 'mono', color: C.muted, align: 'right', base: 'middle' });
          V.text(g, fmt(SMIN, 2), B.x - 4, B.y + B.h - 2, { size: 10, font: 'mono', color: C.muted, align: 'right' });
          [0.5 - S.sep / 2, 0.5 + S.sep / 2].forEach((c) => { V.line(g, PX(c), B.y, PX(c), B.y + B.h, { color: V.alpha(C.b, 0.7), lw: 1, dash: [3, 4] }); });
        }
      });
    })();

    /* ======================================== FIG 3 — o laço da AWR */
    (function () {
      const fig = root.querySelector('#fig-awr-laco');
      adaptH(fig, 470, 770);
      const ROWS = [
        ['fácil', 'a', 0.92], ['fácil', 'b', 0.71], ['fácil', 'c', 0.80], ['fácil', 'a', 0.88],
        ['médio', 'a', 0.55], ['médio', 'b', 0.18], ['médio', 'c', 0.42], ['médio', 'a', 0.63],
        ['difícil', 'b', 0.02], ['difícil', 'c', 0.38], ['difícil', 'a', 0.08], ['difícil', 'c', 0.45],
      ];
      const STATES = ['fácil', 'médio', 'difícil'], ACTS = ['a', 'b', 'c'];
      const S = { base: 'awr', beta: 0.05, clip: true, pol: { 'fácil': [1 / 3, 1 / 3, 1 / 3], 'médio': [1 / 3, 1 / 3, 1 / 3], 'difícil': [1 / 3, 1 / 3, 1 / 3] }, share: [1 / 3, 1 / 3, 1 / 3] };
      function compute() {
        const Vs = {};
        STATES.forEach((s) => { const r = ROWS.filter((x) => x[0] === s).map((x) => x[2]); Vs[s] = r.reduce((a, b) => a + b, 0) / r.length; });
        const rows = ROWS.map(([s, a, R]) => { const A = S.base === 'awr' ? R - Vs[s] : R; const raw = Math.exp(A / S.beta); const w = S.clip ? Math.min(raw, 20) : raw; return { s, a, R, V: Vs[s], A, raw, w, clipped: S.clip && raw > 20 }; });
        const pol = {}, share = []; let tot = 0;
        STATES.forEach((s) => { const rs = rows.filter((r) => r.s === s); const ws = rs.reduce((x, r) => x + r.w, 0); share.push(ws); tot += ws; pol[s] = ACTS.map((a) => rs.filter((r) => r.a === a).reduce((x, r) => x + r.w, 0) / ws); });
        return { Vs, rows, pol, share: share.map((x) => x / tot), nclip: rows.filter((r) => r.clipped).length };
      }
      let D = compute();
      const steps = [
        { t: 'Coletar', d: 'A política atual \\(\\pi_k\\) joga episódios, e cada amostra \\((s, a, \\mathcal R)\\) entra no fim do buffer FIFO, enquanto as mais velhas saem. Os dados são <i>off-policy</i>: misturam várias políticas passadas.' },
        { t: 'Ajustar o crítico', d: 'Regressão 1: \\(V(s)\\) aprende a prever o retorno de cada estado por erro quadrático. Num mundo tabular, o mínimo é a média dos retornos daquele estado.' },
        { t: 'Vantagem', d: '\\(A = \\mathcal R - V(s)\\): positiva se a ação foi melhor que a média do estado, negativa se foi pior. No modo RWR não há linha de base, e \\(A\\) é o próprio retorno.' },
        { t: 'Pesos', d: '\\(w = \\min(e^{A/\\beta}, \\omega_{\\max})\\). Com \\(\\beta = 0{,}05\\) (o valor do paper), uma vantagem de 0,15 já vale \\(e^{3} \\approx 20\\): o corte entra em ação. Células âmbar bateram no teto.' },
        { t: 'Regressão da política', d: 'Regressão 2: máxima verossimilhança ponderada. No caso tabular, a solução exata é \\(\\pi(a\\mid s) = \\sum w_{s,a} / \\sum w_s\\). Tracejado: a política antes; cheio: depois. A barra de baixo mostra a fatia do gradiente que cada estado leva.' },
        { t: 'Repetir', d: 'A política nova \\(\\pi_{k+1}\\) coleta mais dados e o laço recomeça: no paper, ~2.000 amostras novas por iteração, buffer de 50 mil, 200 passos de valor e 1.000 de política.' },
      ];
      let H;
      const st = UI.stepper(fig, steps, { autoplay: 7000, onStep: () => H && H.redraw() });
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'linha de base', options: [{ v: 'awr', label: 'AWR: R − V(s)' }, { v: 'rwr', label: 'RWR: R' }], value: 'awr', onChange: (v) => { S.base = v; D = compute(); upd(); } });
      nocase(UI.slider(ctl, { label: 'β', min: 0.05, max: 1, step: 0.01, value: 0.05, fmt: (v) => fmt(v, 2), onInput: (v) => { S.beta = v; D = compute(); upd(); } }));
      nocase(UI.toggle(ctl, { label: 'corte ω_max = 20', value: true, onChange: (v) => { S.clip = v; D = compute(); upd(); } }));
      const rS = UI.readout(ctl, 'fatia do gradiente (fácil · médio · difícil)', '—'), rC = UI.readout(ctl, 'pesos no teto', '—');
      function upd() {
        rS.set(D.share.map((x) => Math.round(x * 100) + '%').join(' · '), D.share[0] > 0.9 ? 'warn' : null);
        rC.set(D.nclip + ' de 12', D.nclip > 5 ? 'warn' : null);
      }
      upd();
      H = V.fig(fig, (g, w, h, t, dt) => {
        const C = V.C();
        const i = st.i, el = st.elapsed();
        const narrow = w < 640;
        const M = narrow ? { x: 8, y: 26, w: w - 16, h: 12 * 25 + 26 } : { x: 10, y: 26, w: w * 0.55 - 10, h: h - 40 };
        const P = narrow ? { x: 8, y: M.y + M.h + 30, w: w - 16, h: h - (M.y + M.h + 30) - 8 } : { x: w * 0.55 + 18, y: 26, w: w * 0.45 - 28, h: h - 34 };
        // ---- matriz do buffer
        const cols = ['s', 'a', 'R', 'V(s)', S.base === 'awr' ? 'A=R−V' : 'A ← R', 'w'];
        const cwF = [0.18, 0.1, 0.15, 0.15, 0.19, 0.23];
        const rh = Math.min(30, (M.h - 24) / 12);
        V.text(g, 'buffer 𝒟 (FIFO) — uma linha por amostra', M.x, 14, { size: 11.5, color: C.muted });
        let cx = M.x; const cxs = cwF.map((f) => { const x0 = cx; cx += f * M.w; return x0; });
        cols.forEach((c, k) => V.text(g, c, cxs[k] + (cwF[k] * M.w) / 2, M.y + 9, { size: 11, font: 'mono', color: k >= 3 && i < k - 2 ? V.alpha(C.muted, 0.5) : C.muted, align: 'center', weight: 600 }));
        const wmax = Math.max(...D.rows.map((r) => r.w));
        D.rows.forEach((r, k) => {
          const y = M.y + 20 + k * rh;
          const appear = i > 0 ? 1 : V.clamp((el - k * 0.12) / 0.4, 0, 1);
          if (appear <= 0) return;
          g.globalAlpha = appear;
          const off = (1 - V.ease.out(appear)) * 40;
          const sc = STATES.indexOf(r.s);
          const stc = [C.a, C.d, C.c][sc];
          g.fillStyle = V.alpha(stc, 0.08); V.rr(g, M.x + off, y, M.w, rh - 3, 5); g.fill();
          const cell = (k2, txt, o = {}) => { const x0 = cxs[k2] + off, ww = cwF[k2] * M.w; if (o.fill) { g.fillStyle = o.fill; V.rr(g, x0 + 2, y + 1, ww - 4, rh - 5, 4); g.fill(); } if (o.stroke) { g.strokeStyle = o.stroke; g.lineWidth = 1.8; V.rr(g, x0 + 2, y + 1, ww - 4, rh - 5, 4); g.stroke(); } V.text(g, txt, x0 + ww / 2, y + rh / 2 - 1, { size: narrow ? 10.5 : 11, font: 'mono', color: o.color || C.ink, align: 'center', base: 'middle', maxW: ww - 4 }); };
          cell(0, r.s, { color: stc });
          cell(1, r.a);
          cell(2, fmt(r.R, 2));
          const fillIn = (stepIdx) => (i > stepIdx ? 1 : i === stepIdx ? V.clamp((el - k * 0.1) / 0.35, 0, 1) : 0);
          const fV = fillIn(1), fA = fillIn(2), fW = fillIn(3);
          if (fV > 0) { g.globalAlpha = appear * fV * (S.base === 'awr' ? 1 : 0.35); cell(3, fmt(r.V, 3), { color: C.b }); g.globalAlpha = appear; }
          if (fA > 0) { g.globalAlpha = appear * fA; const a = V.clamp(r.A / 0.4, -1, 1); cell(4, fmtS(r.A, 3), { fill: V.cmap(a, 'div'), color: Math.abs(a) > 0.55 ? C.bg : C.ink }); g.globalAlpha = appear; }
          if (fW > 0) {
            g.globalAlpha = appear * fW;
            const x0 = cxs[5] + off, ww = cwF[5] * M.w;
            const frac = Math.log1p(r.w) / Math.log1p(Math.max(wmax, 1e-9));
            g.fillStyle = V.alpha(r.clipped ? C.d : C.a, 0.35); V.rr(g, x0 + 2, y + 1, Math.max(2, (ww - 4) * frac), rh - 5, 4); g.fill();
            if (r.clipped) { g.strokeStyle = C.d; g.lineWidth = 1.8; V.rr(g, x0 + 2, y + 1, ww - 4, rh - 5, 4); g.stroke(); }
            V.text(g, (r.clipped ? '⌈' : '') + fmtW(r.w), x0 + ww / 2, y + rh / 2 - 1, { size: narrow ? 10.5 : 11, font: 'mono', color: C.ink, align: 'center', base: 'middle', maxW: ww - 4 });
            g.globalAlpha = appear;
          }
          g.globalAlpha = 1;
        });
        // ---- painel da política
        const bh = (P.h - 46) / 3;
        STATES.forEach((s, k) => {
          const y = P.y + k * bh;
          const stc = [C.a, C.d, C.c][k];
          V.text(g, s, P.x, y + 14, { size: 12.5, weight: 700, color: stc });
          // pontos de R e V
          const lx = P.x + 66, lw = P.w - 66 - 44;
          const X = (v) => lx + v * lw;
          V.line(g, lx, y + 10, lx + lw, y + 10, { color: C.grid, lw: 1.5 });
          V.text(g, '0', lx, y + 25, { size: 9.5, font: 'mono', color: C.muted, align: 'center' });
          V.text(g, '1', lx + lw, y + 25, { size: 9.5, font: 'mono', color: C.muted, align: 'center' });
          V.text(g, 'R', lx + lw + 10, y + 10, { size: 10, font: 'mono', color: C.muted, base: 'middle' });
          D.rows.filter((r) => r.s === s).forEach((r) => { V.dot(g, X(r.R), y + 10, 4, V.alpha(stc, 0.9)); });
          if (i >= 1) { const vx = X(D.Vs[s]); V.line(g, vx, y + 1, vx, y + 19, { color: C.b, lw: 2.2 }); V.text(g, 'V=' + fmt(D.Vs[s], 2), vx, y + 27, { size: 10, font: 'mono', color: C.b, align: 'center', weight: 600 }); }
          // barras horizontais π(a|s)
          const tgt = i >= 4 ? D.pol[s] : [1 / 3, 1 / 3, 1 / 3];
          S.pol[s] = S.pol[s].map((x, j) => x + (tgt[j] - x) * Math.min(1, dt * 3));
          const rowH = Math.min(20, (bh - 44) / 3);
          ACTS.forEach((a, j) => {
            const yy = y + 36 + j * rowH;
            V.text(g, 'π(' + a + '|s)', lx - 6, yy + rowH / 2, { size: 10, font: 'mono', color: C.muted, align: 'right', base: 'middle' });
            g.fillStyle = V.alpha(C.muted, 0.1); V.rr(g, lx, yy + 2, lw, rowH - 5, 3); g.fill();
            const L = S.pol[s][j] * lw;
            g.fillStyle = V.alpha(C.a, 0.8); V.rr(g, lx, yy + 2, Math.max(1, L), rowH - 5, 3); g.fill();
            V.line(g, lx + lw / 3, yy, lx + lw / 3, yy + rowH - 1, { color: C.b, lw: 1.5, dash: [3, 2] });
            if (i >= 4) V.text(g, fmt(S.pol[s][j], 2), lx + lw + 6, yy + rowH / 2, { size: 10, font: 'mono', color: C.ink, base: 'middle' });
          });
        });
        // fatia do gradiente
        const gy = P.y + P.h - 22;
        if (i >= 3) {
          V.text(g, 'fatia do gradiente por estado (Σw)', P.x, gy - 6, { size: 10.5, color: C.muted });
          let x = P.x;
          D.share.forEach((f, k) => { const ww = f * P.w; g.fillStyle = V.alpha([C.a, C.d, C.c][k], 0.8); g.fillRect(x, gy, Math.max(0.5, ww), 12); if (ww > 34) V.text(g, Math.round(f * 100) + '%', x + ww / 2, gy + 6.5, { size: 10, font: 'mono', color: C.bg, align: 'center', base: 'middle', weight: 700 }); x += ww; });
        }
        if (i === 5) {
          // FIFO: as linhas sobem, a mais velha sai e uma linha nova da política atualizada entra no fim
          const u = V.ease.inOut(V.clamp((el % 3.2) / 1.6, 0, 1));
          const yTop = M.y + 20, yNew = yTop + 12 * rh + 2 + (1 - u) * 12;
          g.fillStyle = V.alpha(C.f, 0.16 * u); V.rr(g, M.x, yNew, M.w, rh - 3, 5); g.fill();
          g.setLineDash([5, 4]); g.strokeStyle = V.alpha(C.f, u); g.lineWidth = 1.6; V.rr(g, M.x, yNew, M.w, rh - 3, 5); g.stroke(); g.setLineDash([]);
          V.text(g, 'nova amostra da política atualizada →  entra no fim', M.x + M.w / 2, yNew + rh / 2 - 1, { size: 11, color: C.f, align: 'center', base: 'middle', alpha: u, weight: 600, maxW: M.w - 10 });
          g.fillStyle = V.alpha(C.panel || C.bg, 0.9 * u); V.rr(g, M.x - 1, yTop - 1, M.w + 2, rh - 1, 5); g.fill();
          V.text(g, '↑ a amostra mais velha sai (FIFO)', M.x + M.w / 2, yTop + rh / 2 - 1, { size: 11, color: C.f, align: 'center', base: 'middle', alpha: u, weight: 600 });
        }
      });
    })();

    /* ======================================== FIG 4 — espectro do SFT ponderado */
    (function () {
      const fig = root.querySelector('#fig-espectro');
      adaptH(fig, 440, 520);
      const R = [1.0, 0.0, 0.5, 0.2], VB = R.reduce((a, b) => a + b, 0) / 4;
      const LENS = [9, 6, 8, 7];
      const COLS = () => { const C = V.C(); return [C.a, C.c, C.d, C.b]; };
      const BS = logSl(0.02, 12.5);
      const TAUS = [0.1, 0.4, 0.7];
      const S = { mode: 'awr', beta: 0.5, clip: false, tau: 0.4, disp: [0.25, 0.25, 0.25, 0.25] };
      function weights(mode = S.mode, beta = S.beta, clip = S.clip) {
        let raw;
        if (mode === 'sft') raw = [1, 1, 1, 1];
        else if (mode === 'awr') raw = R.map((r) => Math.exp((r - VB) / beta));
        else if (mode === 'rwr') raw = R.map((r) => Math.exp(r / beta));
        else if (mode === 'rest') raw = R.map((r) => (r > S.tau ? 1 : 0));
        else if (mode === 'restem') raw = R.map((r) => (r >= 0.5 ? 1 : 0));
        else raw = R.map((r, i) => (i === R.indexOf(Math.max(...R)) ? 1 : 0));
        const cl = raw.map((x) => clip && (mode === 'awr' || mode === 'rwr') && x > 20);
        if (clip && (mode === 'awr' || mode === 'rwr')) raw = raw.map((x) => Math.min(x, 20));
        const s = raw.reduce((a, b) => a + b, 0);
        return { raw, cl, n: raw.map((x) => x / s) };
      }
      const parts = Array.from({ length: 46 }, (_, k) => ({ k, u: (k * 0.137) % 1, resp: k % 4, tok: k % 6 }));
      const rr = V.rng(59);
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'esquema', options: [{ v: 'sft', label: 'SFT' }, { v: 'awr', label: 'AWR' }, { v: 'rwr', label: 'RWR' }, { v: 'rest', label: 'ReST (τ crescente)' }, { v: 'restem', label: 'ReST-EM (R≥0,5)' }, { v: 'raft', label: 'RAFT (best-of-4)' }], value: 'awr', onChange: (v) => { S.mode = v; grp.show(v === 'awr' || v === 'rwr'); gT.show(v === 'rest'); upd(); } });
      const gT = UI.group(ctl); gT.show(false);
      nocase(UI.seg(gT.el, { label: 'Improve: limiar τ', options: TAUS.map((x, k) => ({ v: x, label: 'τ' + '₁₂₃'[k] + ' = ' + fmt(x, 1) })), value: 0.4, onChange: (v) => { S.tau = v; upd(); } }));
      const grp = UI.group(ctl);
      const sB = nocase(UI.slider(grp.el, { label: 'β', min: 0, max: 100, step: 1, value: BS.from(0.5), fmt: (v) => fmt(BS.to(v), BS.to(v) < 0.1 ? 3 : 2), onInput: (v) => { S.beta = BS.to(v); upd(); } }));
      UI.seg(grp.el, { options: [0.05, 0.1, 0.5, 1, 10].map((b) => ({ v: b, label: fmt(b, b < 0.1 ? 2 : b < 1 ? 1 : 0) })), value: 0.5, onChange: (b) => { sB.set(BS.from(b), true); sB.el.querySelector('.val').textContent = fmt(b, b < 0.1 ? 2 : 2); S.beta = b; upd(); } });
      nocase(UI.toggle(grp.el, { label: 'corte ω_max = 20', value: false, onChange: (v) => { S.clip = v; upd(); } }));
      const rN = UI.readout(ctl, 'pesos normalizados', '—'), rE = UI.readout(ctl, 'tamanho efetivo (de 4)', '—');
      function upd() {
        const W = weights();
        rN.set(W.n.map((x) => fmt(x, 4)).join(' ; '));
        const ess = 1 / W.n.reduce((s, x) => s + x * x, 0);
        rE.set(fmt(ess, 2), ess < 1.3 ? 'warn' : null);
      }
      upd();
      // curvas do leque (pré-computadas por modo/corte)
      const fanCache = {};
      function fan(mode, clip) {
        const key = mode + clip; if (fanCache[key]) return fanCache[key];
        const arr = []; for (let k = 0; k <= 80; k++) { const b = Math.exp(Math.log(0.02) + (Math.log(12.5) - Math.log(0.02)) * k / 80); arr.push(weights(mode, b, clip).n); }
        return (fanCache[key] = arr);
      }
      V.fig(fig, (g, w, h, t, dt) => {
        const C = V.C(), cols = COLS();
        const narrow = w < 600;
        const W = weights();
        S.disp = S.disp.map((x, i) => x + (W.n[i] - x) * Math.min(1, dt * 5));
        const slot = (w - 20) / 4;
        const cy = 46;
        // emissor de gradiente
        V.text(g, '∇ “faça mais disso”', w / 2, 13, { size: 11, color: C.muted, align: 'center', font: 'mono' });
        const stripY = cy + 52, cellW = Math.min(14, (slot - 16) / 9), cellH = 15;
        for (let i = 0; i < 4; i++) {
          const x = 10 + slot * i + slot / 2;
          const wv = S.disp[i];
          const bounce = Math.sin(t * 5 + i * 1.3) * 5 * Math.min(1, wv * 2.5);
          const sz = 26 + 16 * Math.sqrt(wv);
          tokChar(g, x, cy - bounce, sz, cols[i], { mood: wv > 0.3 ? 'happy' : wv > 0.05 ? 'o' : 'sad', alpha: 0.35 + 0.65 * Math.min(1, wv * 3 + 0.15) });
          chip(g, x - 30, cy + 26, 60, 17, 'R=' + fmt(R[i], 1), { size: 10.5, stroke: V.alpha(cols[i], 0.6), fill: V.alpha(cols[i], 0.12) });
          // faixa de tokens
          const n = LENS[i], sw = n * cellW + (n - 1) * 2, sx = x - sw / 2;
          for (let k = 0; k < n; k++) {
            const v = Math.sqrt(V.clamp(wv, 0, 1));
            g.fillStyle = v < 0.02 ? V.alpha(C.muted, 0.12) : V.cmap(v, 'seq');
            V.rr(g, sx + k * (cellW + 2), stripY, cellW, cellH, 3); g.fill();
            g.strokeStyle = V.alpha(C.muted, 0.35); g.lineWidth = 1; g.stroke();
          }
          V.text(g, narrow ? 'y' + (i + 1) + ' · ' + n : 'y' + (i + 1) + ' · ' + n + ' tokens', x, stripY + cellH + 13, { size: 10.5, font: 'mono', color: C.muted, align: 'center', maxW: slot - 4 });
        }
        // partículas de gradiente
        parts.forEach((p) => {
          p.u += dt * 0.55;
          if (p.u >= 1) { p.u -= 1; p.resp = sampleCat(W.n, rr()); p.tok = rr.int(LENS[p.resp]); }
          if (W.n[p.resp] < 0.004) return;
          const x = 10 + slot * p.resp + slot / 2;
          const n = LENS[p.resp], sw = n * cellW + (n - 1) * 2, tx = x - sw / 2 + p.tok * (cellW + 2) + cellW / 2;
          const yy = V.lerp(20, stripY + cellH / 2, p.u), xx = V.lerp(w / 2, tx, V.ease.out(p.u));
          V.dot(g, xx, yy, 2.2, V.alpha(cols[p.resp], 0.85 * (1 - p.u * 0.5)));
        });
        // tabela de pesos
        const ty = stripY + cellH + 28;
        const lab = narrow ? 52 : 120;
        V.text(g, narrow ? 'cru' : 'peso cru', 10, ty + 11, { size: 11, color: C.muted, base: 'middle' });
        V.text(g, narrow ? 'norm.' : 'normalizado', 10, ty + 33, { size: 11, color: C.muted, base: 'middle' });
        const tcw = (w - 20 - lab) / 4;
        for (let i = 0; i < 4; i++) {
          const x = 10 + lab + i * tcw;
          chip(g, x + 3, ty + 2, tcw - 6, 19, (W.cl[i] ? '⌈' : '') + fmtW(W.raw[i]), { size: 10.5, stroke: W.cl[i] ? C.d : null });
          chip(g, x + 3, ty + 24, tcw - 6, 19, fmt(W.n[i], 4), { size: 10.5, fill: V.alpha(cols[i], 0.1 + 0.5 * W.n[i]), stroke: cols[i] });
        }
        // leque: pesos normalizados em função de β
        const fy = ty + 62, fh = h - fy - 34;
        const stepMode = S.mode === 'rest' || S.mode === 'restem';
        if (fh > 40 && !stepMode) {
          const fx = 34, fw = w - fx - 14;
          const fm = S.mode === 'rwr' ? 'rwr' : 'awr';
          const arr = fan(fm, S.clip);
          const ax = axes(g, fx, fy, fw, fh, { xmin: 0, xmax: 80, ymin: 0, ymax: 1, yt: [0, 0.5, 1], xt: [], yf: (v) => fmt(v, 1) });
          for (let i = 0; i < 4; i++) polyline(g, arr.map((n, k) => [ax.X(k), ax.Y(n[i])]), V.alpha(cols[i], 0.9), 2);
          V.text(g, '← β→0: best-of-K (RAFT)', fx + 4, fy + fh + 16, { size: 10.5, color: C.muted });
          V.text(g, 'β→∞: SFT uniforme →', fx + fw, fy + fh + 16, { size: 10.5, color: C.muted, align: 'right' });
          V.text(g, 'pesos normalizados × β (' + (fm === 'rwr' ? 'RWR' : 'AWR') + (S.clip ? ', com corte' : '') + ')', fx + 6, fy - 6, { size: 10.5, color: C.muted });
          if (S.mode === 'awr' || S.mode === 'rwr') {
            const k = 80 * (Math.log(S.beta) - Math.log(0.02)) / (Math.log(12.5) - Math.log(0.02));
            V.line(g, ax.X(k), fy, ax.X(k), fy + fh, { color: C.f, lw: 2 });
            for (let i = 0; i < 4; i++) V.dot(g, ax.X(k), ax.Y(W.n[i]), 4, cols[i]);
          } else if (S.mode === 'sft' || S.mode === 'raft') {
            const lbl = { sft: 'SFT: todos 0,25 (extremo direito)', raft: 'RAFT: extremo esquerdo' }[S.mode];
            V.text(g, lbl, fx + fw / 2, fy + fh / 2, { size: 12, color: C.f, align: 'center', weight: 600 });
          }
        }
        // ReST × ReST-EM: o degrau sobre o eixo das recompensas (no lugar do leque)
        if (stepMode && fh > 40) {
          const fx = 34, fw = w - fx - 14;
          const X = (r) => fx + 8 + r * (fw - 16), ay = fy + Math.min(fh * 0.6, 130);
          const thr = S.mode === 'rest' ? S.tau : 0.5;
          V.wrap(g, S.mode === 'rest' ? 'ReST · Improve: só passa quem tem R > τ; a cada passo τ sobe e o subconjunto encolhe' : 'ReST-EM · verificador binário: “correto” ⇔ R ≥ 0,5, a mesma regra em toda iteração', fx, fy - 6, fw, { size: 11, color: C.muted, lineH: 14, maxLines: 2 });
          V.line(g, X(0), ay, X(1), ay, { color: C.muted, lw: 1.4 });
          [0, 0.5, 1].forEach((v) => { V.line(g, X(v), ay, X(v), ay + 4, { color: C.muted, lw: 1 }); V.text(g, fmt(v, 1), X(v), ay + 16, { size: 10.5, font: 'mono', color: C.muted, align: 'center' }); });
          V.text(g, 'R', X(1) + 6, ay + 4, { size: 11, font: 'mono', color: C.muted });
          g.fillStyle = V.alpha(C.e, 0.12); g.fillRect(X(thr), ay - 44, X(1) - X(thr), 44);
          V.text(g, 'guardadas', X(1) - 4, ay - 36, { size: 10, color: C.e, align: 'right' });
          if (S.mode === 'rest') TAUS.forEach((tv, k) => { const on = tv === S.tau; V.line(g, X(tv), ay - 50, X(tv), ay + 6, { color: on ? C.f : V.alpha(C.f, 0.3), lw: on ? 2.4 : 1.2, dash: on ? null : [3, 3] }); V.text(g, 'τ' + '₁₂₃'[k], X(tv), ay - 56, { size: 11, font: 'mono', color: on ? C.f : V.alpha(C.f, 0.5), align: 'center', weight: on ? 700 : 400 }); });
          else { V.line(g, X(thr), ay - 50, X(thr), ay + 6, { color: C.f, lw: 2.4 }); V.text(g, '✓ ⇔ R ≥ 0,5', X(thr), ay - 56, { size: 11, font: 'mono', color: C.f, align: 'center', weight: 700 }); }
          R.forEach((r, i) => { const kept = W.n[i] > 0; const y = ay - 9 - (i % 2) * 17; V.dot(g, X(r), y, kept ? 6 : 4.5, kept ? cols[i] : V.alpha(cols[i], 0.35)); V.text(g, 'y' + (i + 1), X(r) + (r > 0.9 ? -9 : 9), y, { size: 10, font: 'mono', color: kept ? C.ink : C.muted, base: 'middle', align: r > 0.9 ? 'right' : 'left' }); });
          const nk = W.n.filter((x) => x > 0).length;
          V.text(g, (S.mode === 'rest' ? 'τ = ' + fmt(S.tau, 1) + ' → ' : '') + 'guardadas: ' + nk + ' de 4', fx + fw, Math.min(fy + fh + 16, ay + 46), { size: 11, font: 'mono', color: C.f, align: 'right', weight: 600 });
        }
      });
    })();

    /* ======================================== FIG 5 — expert iteration */
    (function () {
      const fig = root.querySelector('#fig-iteracao');
      adaptH(fig, 480, 760);
      const SJ = [0.05, 0.12, 0.25, 0.40, 0.55, 0.70, 0.85, 0.30];
      const PI0 = [0.26, 0.20, 0.16, 0.12, 0.09, 0.07, 0.04, 0.06];
      const DIFF = [1.3, 1.2, 1.1, 1.0, 1.0, 0.9, 0.8, 0.7, 0.6, 0.45, 0.3, 0.15];
      const NPR = DIFF.length, M = SJ.length, ETA = 0.6, KS = [1, 2, 4, 8, 16, 32];
      const succ = (j, p) => Math.min(0.98, SJ[j] * DIFF[p]);
      const passOf = (pi) => pi.reduce((s, x, j) => s + x * DIFF.reduce((a, _, p) => a + succ(j, p), 0) / NPR, 0);
      const S = { mode: 'filtro', K: 4, beta: 0.3, auto: true, seed: 1, hold: 0 };
      let st;
      function reset() {
        st = { iter: 0, pi: PI0.slice(), disp: PI0.slice(), hist: [{ pass: passOf(PI0), H: ent(PI0) }], smp: null, phase: 0, tph: 0, next: null, nohit: 0 };
      }
      reset();
      /** pesos de uma linha (as K amostras de um prompt) no modo atual; devolve quantas acertaram */
      function weigh(row, r) {
        const okIdx = row.map((c, k) => (c.ok ? k : -1)).filter((k) => k >= 0);
        row.forEach((c) => { c.w = 0; });
        if (S.mode === 'filtro') row.forEach((c) => { c.w = c.ok ? 1 : 0; });
        else if (S.mode === 'raft') { if (okIdx.length) row[okIdx[Math.floor(r() * okIdx.length)]].w = 1; }
        else if (okIdx.length > 0 && okIdx.length < row.length) {
          // AWR com normalização por prompt e máscara de variância zero (como no código do texto):
          // sem a máscara, um prompt só com erros teria A = 0, pesos 1/K e viraria SFT nas erradas
          const m = okIdx.length / row.length;
          const ws = row.map((c) => Math.min(20, Math.exp(((c.ok ? 1 : 0) - m) / S.beta)));
          const z = ws.reduce((a, b) => a + b, 0);
          row.forEach((c, k) => { c.w = ws[k] / z; });
        }
        return okIdx.length;
      }
      /** alvo esperado de uma rodada (exato no filtro e no best-of-K; Monte Carlo fixo no AWR) */
      function expTarget() {
        const pi = st.pi, t = new Array(M).fill(0);
        if (S.mode !== 'awr') {
          for (let p = 0; p < NPR; p++) {
            const sp = pi.reduce((a, x, j) => a + x * succ(j, p), 0);
            const wp = S.mode === 'filtro' ? S.K : (1 - Math.pow(1 - sp, S.K)) / Math.max(sp, 1e-12);
            for (let j = 0; j < M; j++) t[j] += wp * pi[j] * succ(j, p);
          }
        } else {
          const r = V.rng(4242);
          for (let n = 0; n < 160; n++) for (let p = 0; p < NPR; p++) {
            const row = [];
            for (let k = 0; k < S.K; k++) { const j = sampleCat(pi, r()); row.push({ j, ok: r() < succ(j, p), w: 0 }); }
            weigh(row, r); row.forEach((c) => { t[c.j] += c.w; });
          }
        }
        const z = t.reduce((a, b) => a + b, 0);
        return z > 0 ? t.map((x) => x / z) : null;
      }
      function round() {
        const r = V.rng(1000 + st.iter * 31 + S.seed * 7);
        const smp = [];
        let nohit = 0;
        const tgt = new Array(M).fill(0);
        for (let p = 0; p < NPR; p++) {
          const row = [];
          for (let k = 0; k < S.K; k++) { const j = sampleCat(st.pi, r()); row.push({ j, ok: r() < succ(j, p), w: 0 }); }
          if (!weigh(row, r)) nohit++;
          row.forEach((c) => { tgt[c.j] += c.w; });
          smp.push(row);
        }
        st.smode = S.mode;
        const tot = tgt.reduce((a, b) => a + b, 0);
        let pi = st.pi.slice();
        if (tot > 0) { pi = pi.map((x, j) => (1 - ETA) * x + ETA * tgt[j] / tot); }
        pi = pi.map((x) => Math.max(x, 1e-6)); const z = pi.reduce((a, b) => a + b, 0); pi = pi.map((x) => x / z);
        st.smp = smp; st.next = pi; st.nohit = nohit / NPR; st.phase = 1; st.tph = 0;
      }
      function finish() {
        st.pi = st.next; st.iter++; st.hist.push({ pass: passOf(st.pi), H: ent(st.pi) }); st.phase = 0; st.tph = 0;
        upd();
      }
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'filtro', options: [{ v: 'filtro', label: 'binário (RFT / ReST-EM)' }, { v: 'raft', label: 'best-of-K (RAFT)' }, { v: 'awr', label: 'AWR e^(A/β)' }], value: 'filtro', onChange: (v) => { S.mode = v; grp.show(v === 'awr'); upd(); } });
      nocase(UI.slider(ctl, { label: 'K amostras / prompt', min: 0, max: KS.length - 1, step: 1, value: 2, fmt: (v) => String(KS[v]), onInput: (v) => { S.K = KS[v]; upd(); } }));
      const grp = UI.group(ctl); grp.show(false);
      nocase(UI.slider(grp.el, { label: 'β', min: 0.05, max: 2, step: 0.05, value: 0.3, fmt: (v) => fmt(v, 2), onInput: (v) => { S.beta = v; upd(); } }));
      const bRun = UI.button(ctl, '⏸ pausar', () => { S.auto = !S.auto; bRun.innerHTML = S.auto ? '⏸ pausar' : '▶ rodar'; if (S.auto && st.phase === 0) round(); }, { primary: true });
      UI.button(ctl, 'uma rodada', () => { if (st.phase === 0) round(); });
      UI.button(ctl, '↺ recomeçar', () => { S.seed++; reset(); upd(); });
      const rI = UI.readout(ctl, 'rodada', '0'), rP = UI.readout(ctl, 'acerto (pass@1)', '—'), rH = UI.readout(ctl, 'entropia de π', '—'), rN = UI.readout(ctl, 'prompts sem acerto', '—'), rB = UI.readout(ctl, 'KL do alvo da rodada', '—');
      function upd() {
        const last = st.hist[st.hist.length - 1];
        rI.set(String(st.iter)); rP.set(Math.round(last.pass * 100) + '%'); rH.set(fmt(last.H, 2) + ' nat', last.H < 0.6 ? 'warn' : null);
        rN.set(st.smp ? Math.round(st.nohit * 100) + '%' : '—', st.nohit > 0.3 ? 'warn' : null);
        const q = expTarget();
        const bound = Math.log(S.K) - (S.K - 1) / S.K;
        rB.set(q ? fmt(kl(q, st.pi), 3) + ' nat' + (S.mode === 'raft' ? ' · limite log K: ' + fmt(bound, 3) : '') : '— (K = 1: nada a comparar)');
      }
      upd();
      const T_GEN = 1.3, T_FIL = 0.8, T_FIT = 1.0;
      V.fig(fig, (g, w, h, t, dt) => {
        const C = V.C();
        if (!LA.paused) {
          if (st.phase > 0) { st.tph += dt; if (st.phase === 1 && st.tph > T_GEN) { st.phase = 2; st.tph = 0; } else if (st.phase === 2 && st.tph > T_FIL) { st.phase = 3; st.tph = 0; } else if (st.phase === 3 && st.tph > T_FIT) finish(); }
          else if (S.auto) { st.tph += dt; if (st.iter >= 16) { if (st.tph > 4) { S.seed++; reset(); upd(); } } else if (st.tph > 0.35) round(); }
        }
        const tgtPi = st.phase === 3 ? st.next : st.pi;
        st.disp = st.disp.map((x, j) => x + (tgtPi[j] - x) * Math.min(1, dt * (st.phase === 3 ? 4 : 8)));
        const narrow = w < 640;
        // layout
        const PH = h - 162;
        const A = narrow ? { x: 30, y: 30, w: w - 40, h: 150 } : { x: 34, y: 30, w: w * 0.33 - 40, h: PH };
        const B = narrow ? { x: 44, y: A.y + A.h + 62, w: w - 54, h: 12 * 18 } : { x: w * 0.33 + 34, y: 30, w: w * 0.4 - 40, h: Math.min(PH - 20, 12 * 24) };
        const Cc = narrow ? { x: 34, y: B.y + B.h + 44, w: w - 48, h: h - (B.y + B.h + 44) - 30 } : { x: w * 0.73 + 26, y: 30, w: w * 0.27 - 36, h: PH };
        // A: política
        const amx = 1;
        const ax = axes(g, A.x, A.y, A.w, A.h, { xmin: 0, xmax: M, ymin: 0, ymax: amx, yt: [0, 0.5, 1], xt: [], yf: (v) => fmt(v, 1) });
        V.text(g, 'política π (estratégias)', A.x, A.y - 12, { size: 11, color: C.muted });
        const bw = A.w / M;
        for (let j = 0; j < M; j++) {
          const x = A.x + j * bw + bw * 0.15, ww = bw * 0.7;
          const v = Math.min(amx, st.disp[j]);
          g.fillStyle = V.alpha(C.a, 0.8); V.rr(g, x, ax.Y(v), ww, Math.max(1, ax.Y(0) - ax.Y(v)), 3); g.fill();
          g.setLineDash([3, 3]); g.strokeStyle = V.alpha(C.b, 0.9); g.lineWidth = 1.2; const y0 = ax.Y(Math.min(amx, PI0[j])); V.line(g, x - 1, y0, x + ww + 1, y0, { color: C.b, lw: 1.4, dash: [3, 3] }); g.setLineDash([]);
          // losango da chance de acerto
          const sy = ax.Y(SJ[j]);
          g.beginPath(); g.moveTo(x + ww / 2, sy - 5); g.lineTo(x + ww / 2 + 5, sy); g.lineTo(x + ww / 2, sy + 5); g.lineTo(x + ww / 2 - 5, sy); g.closePath(); g.fillStyle = C.d; g.fill();
          V.text(g, String(j + 1), x + ww / 2, A.y + A.h + 13, { size: 10.5, font: 'mono', color: C.muted, align: 'center' });
        }
        V.text(g, '◆ chance de acerto da estratégia (mesma escala)', A.x, A.y + A.h + 28, { size: 10.5, color: C.d, maxW: A.w + 30 });
        // B: matriz prompts × K
        V.text(g, 'prompts × amostras', B.x, B.y - 12, { size: 11, color: C.muted });
        const Kd = Math.min(S.K, 16), cw = Math.min(24, (B.w - 4) / Kd), ch = Math.min(24, B.h / NPR);
        for (let p = 0; p < NPR; p++) {
          const y = B.y + p * ch;
          V.text(g, 'p' + (p + 1), B.x - 5, y + ch / 2, { size: 10, font: 'mono', color: C.muted, align: 'right', base: 'middle' });
          const row = st.smp ? st.smp[p] : null;
          const allBad = row && row.every((c) => !c.ok);
          const awrRow = row && st.smode === 'awr';
          const wMax = row ? Math.max(1e-9, ...row.map((c) => c.w)) : 1;
          const masked = awrRow && row.every((c) => c.ok === row[0].ok);
          for (let k = 0; k < Kd; k++) {
            const x = B.x + k * cw;
            let fill = V.alpha(C.muted, 0.1), txt = null, stroke = null, slw = 1.6;
            if (row && row[k]) {
              const c = row[k];
              const shown = st.phase !== 1 || st.tph > (k / Kd) * T_GEN * 0.85;
              if (shown) {
                const kept = c.w > 1e-3;
                const filtered = st.phase === 2 || st.phase === 3 || st.phase === 0;
                const rel = c.w / wMax;
                // AWR: a opacidade é o peso (relativo ao maior da linha); filtros: guardada ou não
                const op = !filtered ? 1 : awrRow ? (kept ? 0.2 + 0.8 * rel : 0.28) : kept ? 1 : 0.28;
                fill = V.alpha(c.ok ? C.e : RED(), 0.55 * op);
                if (filtered && kept && (!awrRow || rel > 0.3)) { stroke = C.d; slw = awrRow ? 0.8 + 1.4 * rel : 1.6; }
                txt = String(c.j + 1);
              }
            }
            g.fillStyle = fill; V.rr(g, x + 1, y + 1, cw - 2, ch - 2, 3); g.fill();
            if (stroke) { g.strokeStyle = stroke; g.lineWidth = slw; g.stroke(); }
            if (txt && cw >= 13) V.text(g, txt, x + cw / 2, y + ch / 2 + 0.5, { size: 9.5, font: 'mono', color: C.ink, align: 'center', base: 'middle' });
          }
          if (allBad && st.phase !== 1) { g.fillStyle = V.alpha(C.muted, 0.22); V.rr(g, B.x, y + 1, Kd * cw, ch - 2, 3); g.fill(); }
          if (masked && st.phase !== 1 && B.w - Kd * cw > 30) V.text(g, 'A=0', B.x + Kd * cw + 5, y + ch / 2, { size: 9.5, font: 'mono', color: C.muted, base: 'middle' });
        }
        if (S.K > 16) V.text(g, '+' + (S.K - 16) + ' colunas', B.x + Kd * cw, B.y + NPR * ch + 12, { size: 10, color: C.muted, align: 'right' });
        const phLbl = ['', '1 · gerar K amostras', '2 · filtrar (âmbar = guardada)', '3 · ajustar π (SFT ponderado)'][st.phase];
        if (phLbl) V.text(g, phLbl, B.x, B.y + NPR * ch + 14, { size: 11.5, color: C.f, weight: 600 });
        else V.text(g, 'número = estratégia; ✓ verde · ✗ vermelho', B.x, B.y + NPR * ch + 14, { size: 10.5, color: C.muted });
        // C: curvas
        const hist = st.hist, nx = Math.max(10, hist.length - 1);
        const ch1 = Cc.h * 0.42;
        const a1 = axes(g, Cc.x, Cc.y, Cc.w, ch1, { xmin: 0, xmax: nx, ymin: 0, ymax: 1, yt: [0, 0.5, 1], xt: [], yf: (v) => Math.round(v * 100) + '%' });
        V.text(g, 'acerto (pass@1)', Cc.x + 4, Cc.y - 8, { size: 11, color: C.a });
        polyline(g, hist.map((p, k) => [a1.X(k), a1.Y(p.pass)]), C.a, 2.2);
        hist.forEach((p, k) => V.dot(g, a1.X(k), a1.Y(p.pass), 2.5, C.a));
        const y2 = Cc.y + ch1 + 34;
        const a2 = axes(g, Cc.x, y2, Cc.w, Cc.h - ch1 - 34, { xmin: 0, xmax: nx, ymin: 0, ymax: Math.log(M), yt: [0, 1, 2], xt: [0, nx], xf: (v) => String(v), yf: (v) => fmt(v, 0) });
        V.text(g, 'entropia H(π) (nat)', Cc.x + 4, y2 - 8, { size: 11, color: C.f });
        polyline(g, hist.map((p, k) => [a2.X(k), a2.Y(p.H)]), C.f, 2.2);
        V.text(g, 'rodada', Cc.x + Cc.w, y2 + (Cc.h - ch1 - 34) + 28, { size: 10.5, color: C.muted, align: 'right' });
        // footer narrow/wide: base de dados
        if (!narrow) {
          const fy = h - 64;
          V.wrap(g, 'p1 é o prompt mais fácil, p12 o mais difícil. Linhas cinzas: nenhuma das K amostras acertou, e o prompt não gera dado em nenhum modo (no AWR, “A=0” marca os prompts sem variância de recompensa, que a máscara descarta). Aumente K e veja essa fração cair; rode muitas rodadas e veja a entropia despencar enquanto o acerto satura.', 34, fy, w - 68, { size: 12, color: C.text2 || C.ink, lineH: 17, maxLines: 3 });
        }
      });
    })();

    /* ======================================== FIG 6 — InfoNCE */
    (function () {
      const fig = root.querySelector('#fig-infonce');
      adaptH(fig, 430, 660);
      const NEG = [1.0, 0.5, -0.5, 0.8, -1.0, 0.2, 1.4, -0.3, 0.0, -1.5, 0.6, -0.8, 1.1, -0.1, 0.3];
      const S = { N: 4, sp: 2.0, sc: null, train: false, acc: 0, steps: 0 };
      const init = () => { S.sc = [S.sp, ...NEG]; S.steps = 0; };
      init();
      const cur = () => S.sc.slice(0, S.N);
      const probs = () => V.softmax(cur());
      const lossOf = () => lse(cur()) - S.sc[0];
      const ctl = UI.controls(fig);
      nocase(UI.slider(ctl, { label: 'N (1 positivo + N−1 negativos)', min: 2, max: 16, step: 1, value: 4, fmt: (v) => String(v), onInput: (v) => { S.N = v; upd(); } }));
      nocase(UI.slider(ctl, { label: 'escore do positivo s₊', min: -1, max: 4, step: 0.1, value: 2, fmt: (v) => fmt(v, 1), onInput: (v) => { S.sp = v; init(); upd(); } }));
      const bT = UI.button(ctl, '▶ treinar', () => { S.train = !S.train; bT.innerHTML = S.train ? '⏸ pausar' : '▶ treinar'; }, { primary: true });
      UI.button(ctl, '↺ exemplo do texto', () => { S.sp = 2; S.N = 4; init(); upd(); });
      const rL = UI.readout(ctl, 'perda 𝓛_N', '—'), rM = UI.readout(ctl, 'limite de MI: log N − 𝓛', '—'), rP = UI.readout(ctl, 'p₊', '—');
      function upd() { const L = lossOf(); rL.set(fmt(L, 4)); rM.set(fmt(Math.log(S.N) - L, 4) + ' nat'); rP.set(fmt(probs()[0], 3)); }
      upd();
      V.fig(fig, (g, w, h, t, dt) => {
        const C = V.C();
        if (S.train && !LA.paused) {
          S.acc += dt;
          while (S.acc > 0.16) { S.acc -= 0.16; const p = probs(); for (let j = 0; j < S.N; j++) S.sc[j] -= 0.35 * (p[j] - (j === 0 ? 1 : 0)); S.steps++; upd(); if (lossOf() < 0.02) { S.train = false; bT.innerHTML = '▶ treinar'; } }
        }
        const p = probs();
        const narrow = w < 640;
        const W = narrow ? { x: 0, y: 0, w: w, h: h * 0.47 } : { x: 0, y: 0, w: w * 0.48, h: h };
        const B = narrow ? { x: 36, y: W.h + 26, w: w - 50, h: h - W.h - 60 } : { x: w * 0.48 + 40, y: 34, w: w * 0.52 - 56, h: h - 80 };
        // roda
        const cx = W.x + W.w / 2, cy = W.y + W.h / 2 + 6, Rr = Math.min(W.w, W.h) * 0.36;
        const n = S.N - 1;
        for (let k = 0; k < n; k++) {
          const ang = -Math.PI / 2 + (k / n) * Math.PI * 2;
          const x = cx + Math.cos(ang) * Rr, y = cy + Math.sin(ang) * Rr;
          V.line(g, cx, cy, x, y, { color: V.alpha(C.c, 0.18 + 0.6 * p[k + 1]), lw: 1 + 4 * p[k + 1] });
        }
        for (let k = 0; k < n; k++) {
          const ang = -Math.PI / 2 + (k / n) * Math.PI * 2;
          const x = cx + Math.cos(ang) * Rr, y = cy + Math.sin(ang) * Rr;
          const sz = 14 + 34 * Math.sqrt(p[k + 1]);
          tokChar(g, x, y, sz, C.c, { mood: p[k + 1] > 0.15 ? 'happy' : 'sad' });
          const al = 6 + 26 * p[k + 1];
          const wob = Math.sin(t * 6 + k) * 1.5;
          V.arrow(g, x + sz / 2 + 6, y - al / 2 + wob, x + sz / 2 + 6, y + al / 2 + wob, { color: RED(), lw: 2, head: 6 });
          V.text(g, 's=' + fmt(S.sc[k + 1], 1), x, y + sz / 2 + 13, { size: 10, font: 'mono', color: C.muted, align: 'center' });
        }
        const sp = 24 + 40 * Math.sqrt(p[0]);
        V.glow(g, cx, cy, sp * 1.1, C.a, 0.4 + 0.2 * Math.sin(t * 3));
        tokChar(g, cx, cy, sp, C.a, { mood: p[0] > 0.5 ? 'happy' : 'o' });
        const ap = 8 + 34 * (1 - p[0]);
        V.arrow(g, cx - sp / 2 - 10, cy + ap / 2, cx - sp / 2 - 10, cy - ap / 2, { color: C.e, lw: 2.6, head: 7 });
        V.text(g, 's₊=' + fmt(S.sc[0], 2), cx, cy + sp / 2 + 13, { size: 10.5, font: 'mono', color: C.a, align: 'center', weight: 600 });
        V.text(g, 'positivo ↑ com 1−p₊ · negativos ↓ com p_j', W.x + 10, 16, { size: 11, color: C.muted });
        // barras
        const sMin = Math.min(-2, ...cur()) , sMax = Math.max(4, ...cur());
        const bh = B.h * 0.5;
        const ax = axes(g, B.x, B.y, B.w, bh, { xmin: 0, xmax: S.N, ymin: sMin, ymax: sMax, yt: [Math.ceil(sMin), 0, Math.floor(sMax)], xt: [] , yf: (v) => fmt(v, 0) });
        V.text(g, 'escores s_j', B.x + 4, B.y - 8, { size: 11, color: C.muted });
        const bw = B.w / S.N;
        for (let j = 0; j < S.N; j++) {
          const x = B.x + j * bw + bw * 0.15, ww = bw * 0.7, y0 = ax.Y(0), y1 = ax.Y(S.sc[j]);
          g.fillStyle = V.alpha(j === 0 ? C.a : C.c, 0.8); V.rr(g, x, Math.min(y0, y1), ww, Math.max(1, Math.abs(y1 - y0)), 3); g.fill();
          if (bw > 22) V.text(g, fmt(S.sc[j], 1), x + ww / 2, S.sc[j] >= 0 ? y1 - 4 : y1 + 11, { size: 9.5, font: 'mono', color: C.ink, align: 'center' });
        }
        const py = B.y + bh + 30, ph = B.h - bh - 46;
        V.text(g, 'softmax p_j (quem “parece” o positivo)', B.x + 4, py - 8, { size: 11, color: C.muted });
        V.line(g, B.x, py + ph, B.x + B.w, py + ph, { color: C.muted, lw: 1 });
        for (let j = 0; j < S.N; j++) {
          const x = B.x + j * bw + bw * 0.15, ww = bw * 0.7, hh = p[j] * ph;
          g.fillStyle = V.alpha(j === 0 ? C.a : C.c, 0.85); V.rr(g, x, py + ph - hh, ww, Math.max(1, hh), 3); g.fill();
          if (bw > 26) V.text(g, fmt(p[j], 2), x + ww / 2, py + ph - hh - 4, { size: 9.5, font: 'mono', color: C.ink, align: 'center' });
        }
        const L = lossOf();
        V.text(g, '𝓛 = −log p₊ = ' + fmt(L, 4), B.x + B.w, B.y - 8, { size: 11.5, font: 'mono', color: C.ink, align: 'right', weight: 600 });
        if (S.N === 2) {
          const bx = narrow ? 12 : W.x + 12, by = narrow ? W.h - 30 : h - 46;
          chip(g, bx, by, Math.min(narrow ? w - 24 : W.w - 24, 420), 30, 'N=2 → −log σ(s₊ − s₋): a perda do DPO', { size: 11.5, fill: V.alpha(C.d, 0.18), stroke: C.d, font: 'sans', weight: 600 });
        }
      });
    })();

    /* ======================================== FIG 7 — rampa, dobradiça, parábola */
    (function () {
      const fig = root.querySelector('#fig-dobradica');
      adaptH(fig, 420, 560);
      const S = { beta: 0.5, delta: 1, ipo: true, run: true, acc: 0, m: [-6, -6, -6], trail: [[], [], []], clock: 0 };
      const LS = {
        dpo: (m) => Math.log1p(Math.exp(-S.beta * m)),
        hinge: (m) => Math.max(0, S.delta - m),
        ipo: (m) => (m - 0.5) ** 2,
      };
      const FS = {
        dpo: (m) => S.beta * sig(-S.beta * m),
        hinge: (m) => (m < S.delta ? 1 : 0),
        ipo: (m) => -2 * (m - 0.5),
      };
      const KEYS = ['dpo', 'hinge', 'ipo'];
      const reset = () => { S.m = [-6, -6, -6]; S.trail = [[], [], []]; };
      const ctl = UI.controls(fig);
      nocase(UI.slider(ctl, { label: 'β (DPO)', min: 0.05, max: 2, step: 0.05, value: 0.5, fmt: (v) => fmt(v, 2), onInput: (v) => { S.beta = v; upd(); } }));
      nocase(UI.slider(ctl, { label: 'δ (margem do SLiC)', min: 0, max: 3, step: 0.1, value: 1, fmt: (v) => fmt(v, 1), onInput: (v) => { S.delta = v; upd(); } }));
      nocase(UI.toggle(ctl, { label: 'mostrar IPO (τ = 1)', value: true, onChange: (v) => { S.ipo = v; } }));
      UI.legend(ctl, [['var(--viz-a)', 'DPO −log σ(βm)'], ['var(--viz-c)', 'SLiC max(0, δ−m)'], ['var(--viz-b)', 'IPO (m−½)²']]);
      const bR = UI.button(ctl, '⏸ pausar', () => { S.run = !S.run; bR.innerHTML = S.run ? '⏸ pausar' : '▶ treinar'; }, { primary: true });
      UI.button(ctl, '↺ recomeçar', () => { reset(); S.clock = 0; upd(); });
      const rD = UI.readout(ctl, 'DPO: m · força', '—'), rH = UI.readout(ctl, 'SLiC: m · força', '—'), rT = UI.readout(ctl, 'par de brinquedo: SLiC-HF (m=−8) · DPO (Δh=+3)', '—');
      function upd() {
        rD.set(fmt(S.m[0], 2) + ' · ' + fmt(FS.dpo(S.m[0]), 3));
        rH.set(fmt(S.m[1], 2) + ' · ' + fmt(FS.hinge(S.m[1]), 0), FS.hinge(S.m[1]) === 0 ? 'ok' : null);
        rT.set(fmt(LS.hinge(-8), 1) + ' · ' + fmt(LS.dpo(3), 2));
      }
      upd();
      V.fig(fig, (g, w, h, t, dt) => {
        const C = V.C();
        const cols = [C.a, C.c, C.b];
        if (S.run && !LA.paused) {
          S.acc += dt;
          S.clock += dt; if (S.clock > 14) { S.clock = 0; reset(); }
          while (S.acc > 0.08) { S.acc -= 0.08; KEYS.forEach((k, i) => { if (k === 'ipo' && !S.ipo) return; S.m[i] = V.clamp(S.m[i] + 0.3 * FS[k](S.m[i]), -9, 9); if (S.trail[i].length > 200) S.trail[i].shift(); S.trail[i].push(S.m[i]); }); upd(); }
        }
        const L = 40, Rr = 14;
        const top = { x: L, y: 30, w: w - L - Rr, h: (h - 70) * 0.62 };
        const bot = { x: L, y: top.y + top.h + 34, w: w - L - Rr, h: (h - 70) * 0.38 - 34 };
        const xmin = -9, xmax = 9;
        const ax = axes(g, top.x, top.y, top.w, top.h, { xmin, xmax, ymin: 0, ymax: 10, yt: [0, 5, 10], xt: [], yf: (v) => fmt(v, 0) });
        V.text(g, 'perda 𝓛(m)', top.x + 4, top.y - 10, { size: 11, color: C.muted });
        const N = 160;
        const curve = (fn, col, lw) => { const pts = []; for (let k = 0; k <= N; k++) { const m = xmin + (xmax - xmin) * k / N; const v = fn(m); if (v <= 10.5) pts.push([ax.X(m), ax.Y(Math.min(10, v))]); else if (pts.length) { polyline(g, pts.splice(0), col, lw); } } polyline(g, pts, col, lw); };
        g.save(); g.beginPath(); g.rect(top.x, top.y - 2, top.w, top.h + 2); g.clip();
        if (S.ipo) curve(LS.ipo, V.alpha(C.b, 0.9), 2);
        curve(LS.hinge, C.c, 2.6);
        curve(LS.dpo, C.a, 2.6);
        g.restore();
        V.line(g, ax.X(S.delta), top.y, ax.X(S.delta), top.y + top.h, { color: V.alpha(C.c, 0.5), lw: 1, dash: [3, 4] });
        V.text(g, 'δ', ax.X(S.delta) + 3, top.y + 10, { size: 11, color: C.c, font: 'mono' });
        V.line(g, ax.X(0), top.y, ax.X(0), top.y + top.h, { color: C.grid, lw: 1 });
        // marcador do par de brinquedo
        const hy = LS.hinge(-8);
        if (hy <= 10) { V.dot(g, ax.X(-8), ax.Y(hy), 4.5, C.d); V.text(g, w < 560 ? 'par de brinquedo, SLiC-HF: m=−8, perda 9,0' : 'par de brinquedo no SLiC-HF: m = −8 (invertido), perda 9,0', ax.X(-8) + 7, ax.Y(hy) - 6, { size: 10.5, color: C.d, weight: 600 }); }
        // o mesmo par visto pelo DPO, com referência: Δh = +3, na ordem certa
        { const dy = LS.dpo(3); g.strokeStyle = C.d; g.lineWidth = 2; g.beginPath(); g.arc(ax.X(3), ax.Y(dy), 5, 0, Math.PI * 2); g.stroke(); V.text(g, w < 560 ? 'DPO: Δh = +3' : 'mesmo par no DPO: Δh = +3', ax.X(3) + 8, ax.Y(dy) - 10, { size: 10.5, color: C.d, weight: 600, align: 'left' }); }
        // bolinhas
        KEYS.forEach((k, i) => {
          if (k === 'ipo' && !S.ipo) return;
          const m = S.m[i], v = Math.min(10, LS[k](m));
          S.trail[i].forEach((mm, j) => { if (j % 3) return; V.dot(g, ax.X(mm), ax.Y(Math.min(10, LS[k](mm))), 1.6, V.alpha(cols[i], 0.4)); });
          if (LS[k](m) > 10) { const xx = ax.X(m), yy = top.y + 4; g.beginPath(); g.moveTo(xx, yy - 6); g.lineTo(xx + 6, yy + 4); g.lineTo(xx - 6, yy + 4); g.closePath(); g.fillStyle = cols[i]; g.fill(); V.text(g, 'IPO: ' + fmt(LS[k](m), 1), xx + 9, yy + 4, { size: 10, font: 'mono', color: cols[i] }); return; }
          V.glow(g, ax.X(m), ax.Y(v), 11, cols[i], 0.7); V.dot(g, ax.X(m), ax.Y(v), 5, cols[i]);
        });
        // força
        const fmax = 1.2;
        const bx = axes(g, bot.x, bot.y, bot.w, bot.h, { xmin, xmax, ymin: -0.2, ymax: fmax, yt: [0, 1], xt: [-8, -4, 0, 4, 8], xf: (v) => fmt(v, 0), yf: (v) => fmt(v, 0), xl: 'margem m  (Δh no DPO e no IPO; log-prob crua no SLiC-HF)' });
        V.text(g, 'força sobre o par  −∂𝓛/∂m', bot.x + 4, bot.y - 8, { size: 11, color: C.muted });
        g.save(); g.beginPath(); g.rect(bot.x, bot.y, bot.w, bot.h); g.clip();
        const fcurve = (fn, col) => { const pts = []; for (let k = 0; k <= N; k++) { const m = xmin + (xmax - xmin) * k / N; pts.push([bx.X(m), bx.Y(V.clamp(fn(m), -0.25, fmax + 0.05))]); } polyline(g, pts, col, 2.2); };
        if (S.ipo) fcurve(FS.ipo, V.alpha(C.b, 0.8));
        fcurve(FS.hinge, C.c); fcurve(FS.dpo, C.a);
        g.restore();
        V.line(g, bot.x, bx.Y(0), bot.x + bot.w, bx.Y(0), { color: C.muted, lw: 1, dash: [2, 3] });
      });
    })();

    /* ======================================== FIG 8 — CPL */
    (function () {
      const fig = root.querySelector('#fig-cpl');
      adaptH(fig, 440, 580);
      function gen(n, target, seed) {
        const r = V.rng(seed);
        const mean = target / n;
        const t = Array.from({ length: n }, () => V.clamp(Math.round((mean + 0.55 * r.normal()) * 10), -30, -2));
        let diff = Math.round(target * 10) - t.reduce((a, b) => a + b, 0);
        let guard = 0;
        while (diff !== 0 && guard++ < 5000) { const k = r.int(n); const s = Math.sign(diff); if (t[k] + s <= -2 && t[k] + s >= -30) { t[k] += s; diff -= s; } }
        return t.map((x) => x / 10);
      }
      const POS = gen(20, -25, 591), NEGS = gen(10, -17, 592);
      const S = { alpha: 0.1, lam: 0.5, shift: 0, anim: null };
      const sums = () => [POS.reduce((a, b) => a + b, 0) + S.shift, NEGS.reduce((a, b) => a + b, 0) + S.shift];
      const lossL = (lam, shift = S.shift) => { const sp = S.alpha * (-25 + shift), sn = S.alpha * (-17 + shift); return Math.log1p(Math.exp(lam * sn - sp)); };
      const ctl = UI.controls(fig);
      nocase(UI.slider(ctl, { label: 'α (temperatura)', min: 0.02, max: 0.5, step: 0.01, value: 0.1, fmt: (v) => fmt(v, 2), onInput: (v) => { S.alpha = v; upd(); } }));
      const sL = nocase(UI.slider(ctl, { label: 'λ (viés conservador)', min: 0.1, max: 1, step: 0.05, value: 0.5, fmt: (v) => fmt(v, 2), onInput: (v) => { S.lam = v; upd(); } }));
      const sS = nocase(UI.slider(ctl, { label: 'deslocamento comum (nat)', min: -3, max: 3, step: 0.1, value: 0, fmt: (v) => fmtS(v, 1), onInput: (v) => { S.shift = v; S.anim = null; upd(); } }));
      UI.button(ctl, '+1 nat nas duas', () => { S.anim = { from: S.shift, to: Math.min(3, S.shift + 1), t: 0 }; }, { primary: true });
      UI.button(ctl, '↺ zerar', () => { S.anim = null; S.shift = 0; sS.set(0, true); upd(); });
      const rV = UI.readout(ctl, 'perda vanilla (λ=1)', '—'), rL = UI.readout(ctl, 'perda CPL(λ)', '—'), rP = UI.readout(ctl, 'P(σ⁺ ≻ σ⁻) com λ', '—');
      function upd() {
        rV.set(fmt(lossL(1), 4)); rL.set(fmt(lossL(S.lam), 4));
        const sp = S.alpha * (-25 + S.shift), sn = S.alpha * (-17 + S.shift);
        rP.set(fmt(sig(sp - S.lam * sn), 3));
      }
      upd();
      V.fig(fig, (g, w, h, t, dt) => {
        const C = V.C();
        if (S.anim && !LA.paused) { S.anim.t = Math.min(1, S.anim.t + dt * 1.2); S.shift = V.lerp(S.anim.from, S.anim.to, V.ease.inOut(S.anim.t)); sS.set(Math.round(S.shift * 10) / 10, true); upd(); if (S.anim.t >= 1) S.anim = null; }
        const narrow = w < 640;
        const cw = Math.min(56, (w - 104) / 10), chh = Math.min(32, cw * 0.8);
        const G = { x: 12, y: 24, w: 10 * cw };
        const drawSeg = (vals, y0, col, title, len) => {
          V.text(g, title, G.x, y0 - 7, { size: 11.5, color: col, weight: 600 });
          vals.forEach((v0, k) => {
            const v = v0 + S.shift / len;
            const r = Math.floor(k / 10), c = k % 10;
            const x = G.x + c * cw, y = y0 + r * (chh + 3);
            const tt = V.clamp((v + 3) / 3, 0, 1);
            g.fillStyle = V.alpha(col, 0.12 + 0.6 * tt);
            V.rr(g, x + 1, y, cw - 2, chh, 4); g.fill();
            g.strokeStyle = S.anim ? V.alpha(C.d, 0.6 + 0.4 * Math.sin(t * 12)) : V.alpha(col, 0.6); g.lineWidth = S.anim ? 1.6 : 1; g.stroke();
            if (cw >= 24) V.text(g, fmt(v, 1), x + cw / 2, y + chh / 2 + 0.5, { size: Math.min(11.5, cw * 0.36), font: 'mono', color: C.ink, align: 'center', base: 'middle' });
          });
          // varredura: a soma acumulando ao longo do segmento
          const ph = ((t * 0.35) % 1.25) * len;
          const kk = Math.floor(ph);
          if (kk < len) { const r = Math.floor(kk / 10), c = kk % 10; const x = G.x + c * cw, y = y0 + r * (chh + 3); V.glow(g, x + cw / 2, y + chh / 2, cw * 0.7, col, 0.55); g.strokeStyle = col; g.lineWidth = 2; V.rr(g, x + 1, y, cw - 2, chh, 4); g.stroke(); }
          // soma parcial até a célula iluminada (a mesma conta que a fileira faz)
          const upto = Math.min(kk, len - 1);
          let part = 0; for (let k = 0; k <= upto; k++) part += vals[k] + S.shift / len;
          return { t: upto + 1, part, done: kk >= len };
        };
        const runLbl = (o, x, y, col) => V.text(g, 't≤' + o.t + ': ' + fmt(o.part, 1), x, y, { size: 10.5, font: 'mono', color: o.done ? V.alpha(col, 0.55) : col, base: 'middle' });
        const [sp, sn] = sums();
        const y1 = G.y + 20;
        const runP = drawSeg(POS, y1, C.a, 'σ⁺ preferido · 20 passos · células = log π_θ(a_t|s_t)', 20);
        const sumX = G.x + 10 * cw + 8;
        // coluna da direita: soma parcial (varredura) em cima, total embaixo
        runLbl(runP, sumX, y1 + chh / 2, C.a);
        V.text(g, 'Σ⁺ ' + fmt(sp, 1), sumX, y1 + chh + 3 + chh / 2, { size: 12, font: 'mono', color: C.a, weight: 700, base: 'middle' });
        const y2 = y1 + 2 * (chh + 3) + 30;
        const runN = drawSeg(NEGS, y2, C.c, 'σ⁻ preterido · 10 passos', 10);
        runLbl(runN, sumX, y2 - 11, C.c);
        V.text(g, 'Σ⁻ ' + fmt(sn, 1), sumX, y2 + chh / 2, { size: 12, font: 'mono', color: C.c, weight: 700, base: 'middle' });
        // contas
        const by = y2 + chh + 30;
        const Sp = S.alpha * sp, Sn = S.alpha * sn;
        const lines = [
          ['S⁺ = α·Σ⁺ = ' + fmt(Sp, 3), C.a],
          ['λ·S⁻ = ' + fmt(S.lam, 2) + '·(' + fmt(Sn, 3) + ') = ' + fmt(S.lam * Sn, 3), C.c],
          ['𝓛 = log(1 + e^(λS⁻ − S⁺))', C.ink],
          ['   = ' + fmt(lossL(S.lam), 4) + '   (vanilla: ' + fmt(lossL(1), 4) + ')', C.d],
        ];
        lines.forEach(([str, c], k) => V.text(g, str, G.x, by + k * 19, { size: narrow ? 11 : 12, font: 'mono', color: c, maxW: narrow ? w - 24 : w * 0.46 }));
        if (!narrow) V.wrap(g, 'Repare: P(σ⁺ ≻ σ⁻) sai abaixo de 0,5. O preferido tem 20 passos e soma de log-prob mais negativa, então parece “pior” sem uma referência que desconte o comprimento. No vanilla, só S⁺ − S⁻ importa; com λ < 1, subir as duas fileiras juntas passa a compensar.', G.x, by + 4 * 19 + 14, w * 0.46, { size: 11.5, color: C.text2 || C.muted, lineH: 16, maxLines: 6 });
        // gráfico perda × deslocamento
        const P = narrow ? { x: 44, y: by + 4 * 19 + 24, w: w - 60, h: h - (by + 4 * 19 + 24) - 40 } : { x: w * 0.52 + 30, y: by, w: w * 0.48 - 50, h: h - by - 40 };
        if (P.h > 50) {
          const lo = Math.min(lossL(1, -3), lossL(S.lam, 3), lossL(S.lam, -3)), hi = Math.max(lossL(1, 0), lossL(S.lam, -3), lossL(S.lam, 3));
          const pad = (hi - lo) * 0.2 + 0.05;
          const y0v = Math.max(0, lo - pad), y1v = hi + pad;
          const ax = axes(g, P.x, P.y, P.w, P.h, { xmin: -3, xmax: 3, ymin: y0v, ymax: y1v, xt: [-3, 0, 3], yt: [y0v, (y0v + y1v) / 2, y1v], xf: (v) => fmtS(v, 0), yf: (v) => fmt(v, 2), xl: 'deslocamento comum (nat)' });
          V.text(g, 'perda × deslocamento', P.x + 4, P.y - 10, { size: 11, color: C.muted });
          const pts1 = [], pts2 = [];
          for (let k = 0; k <= 60; k++) { const sh = -3 + 6 * k / 60; pts1.push([ax.X(sh), ax.Y(lossL(1, sh))]); pts2.push([ax.X(sh), ax.Y(lossL(S.lam, sh))]); }
          polyline(g, pts1, C.b, 2.2, [6, 4]); polyline(g, pts2, C.d, 2.6);
          V.dot(g, ax.X(S.shift), ax.Y(lossL(1)), 5, C.b); V.glow(g, ax.X(S.shift), ax.Y(lossL(S.lam)), 12, C.d, 0.8); V.dot(g, ax.X(S.shift), ax.Y(lossL(S.lam)), 5, C.d);
          V.text(g, 'λ = 1 (vanilla): plana', P.x + P.w, ax.Y(lossL(1, 3)) - 8, { size: 10.5, color: C.b, align: 'right' });
          V.text(g, 'λ = ' + fmt(S.lam, 2) + ': desce', P.x + 8, Math.max(P.y + 12, ax.Y(lossL(S.lam, -3)) - 8), { size: 10.5, color: C.d });
        }
      });
    })();

    /* ======================================== FIG 9 — unlikelihood */
    (function () {
      const fig = root.querySelector('#fig-unlikelihood');
      adaptH(fig, 460, 590);
      const VOC = ['subiu', 'pulou', 'dormiu', 'miou', 'no', 'muro', 'o', '.'];
      const A0 = [3.2, 1.4, 1.0, 0.6, 0.2, 0.4, 0.5, 0.3]; // p(subiu) ≈ 0,63 > 0,5: começa no regime m < 0
      const TGT = 1, NEG1 = 0;
      const CTXNEG = [0, 4, 5, 6, 7]; // subiu, no, muro, o, . (contexto anterior ∩ vocabulário, sem o alvo)
      const SEQ = ['o', 'gato', 'subiu', 'no', 'muro', '.', 'o', 'gato', 'subiu', 'no', 'muro', '.', 'o', 'gato', '▢'];
      const S = { a: A0.slice(), alpha: 1, mode: 'ul', neg: 'one', run: true, acc: 0, n: 0, hold: 0, wait: 2.5 }; // wait: pausa no passo 0 para ver m < 0
      const negs = () => (S.neg === 'one' ? [NEG1] : CTXNEG);
      function grad() { // −∂L/∂a
        const p = V.softmax(S.a), al = S.mode === 'ul' ? S.alpha : 0;
        const gneg = p.map((pi, i) => (i === TGT ? 1 : 0) - pi);
        negs().forEach((c) => { const k = al * p[c] / (1 - p[c]); for (let i = 0; i < p.length; i++) gneg[i] -= k * ((i === c ? 1 : 0) - p[i]); });
        return gneg;
      }
      const loss = () => { const p = V.softmax(S.a), al = S.mode === 'ul' ? S.alpha : 0; return -Math.log(p[TGT]) - al * negs().reduce((s, c) => s + Math.log(1 - p[c]), 0); };
      const stepOnce = () => { const gr = grad(); S.a = S.a.map((x, i) => x + 0.25 * gr[i]); S.n++; upd(); };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'perda', options: [{ v: 'nll', label: 'só NLL (α = 0)' }, { v: 'ul', label: 'NLL + unlikelihood' }], value: 'ul', onChange: (v) => { S.mode = v; upd(); } });
      UI.seg(ctl, { label: 'negativos', options: [{ v: 'one', label: 'C = {subiu} (Eq. 6)' }, { v: 'ctx', label: 'C = contexto anterior (Eq. 5)' }], value: 'one', onChange: (v) => { S.neg = v; upd(); } });
      nocase(UI.slider(ctl, { label: 'α', min: 0, max: 2, step: 0.1, value: 1, fmt: (v) => fmt(v, 1), onInput: (v) => { S.alpha = v; upd(); } }));
      UI.button(ctl, 'um passo', stepOnce);
      const bR = UI.button(ctl, '⏸ pausar', () => { S.run = !S.run; bR.innerHTML = S.run ? '⏸ pausar' : '▶ treinar'; }, { primary: true });
      UI.button(ctl, '↺ recomeçar', () => { S.a = A0.slice(); S.n = 0; S.hold = 0; S.wait = 2.5; upd(); });
      const rT = UI.readout(ctl, 'p(pulou) — alvo', '—'), rN = UI.readout(ctl, 'p(subiu) — negativo', '—'), rL = UI.readout(ctl, 'perda', '—'), rS = UI.readout(ctl, 'passos', '0');
      function upd() { const p = V.softmax(S.a); rT.set(fmt(p[TGT], 3)); rN.set(fmt(p[NEG1], 3), p[NEG1] > 0.3 ? 'warn' : 'ok'); rL.set(fmt(loss(), 3)); rS.set(String(S.n)); }
      upd();
      V.fig(fig, (g, w, h, t, dt) => {
        const C = V.C();
        if (S.run && !LA.paused) { if (S.n >= 30) { S.hold += dt; if (S.hold > 2.5) { S.a = A0.slice(); S.n = 0; S.hold = 0; S.wait = 2.5; upd(); } } else if (S.n === 0 && S.wait > 0) { S.wait -= dt; } else { S.acc += dt; while (S.acc > 0.3 && S.n < 30) { S.acc -= 0.3; stepOnce(); } } }
        const p = V.softmax(S.a), gr = grad();
        const narrow = w < 640;
        // faixa da sequência
        let x = 12, y = 18;
        V.text(g, 'continuação degenerada (ilustrativa):', 12, y, { size: 11, color: C.muted });
        y += 10;
        SEQ.forEach((tok, k) => {
          const tw0 = tw(g, tok, 11.5, 'mono') + 12;
          if (x + tw0 > w - 10) { x = 12; y += 24; }
          const rep = k >= 6 && k <= 10;
          const cur = tok === '▢';
          chip(g, x, y, tw0, 20, cur ? 't' : tok, { size: 11.5, fill: cur ? V.alpha(C.d, 0.3) : rep ? V.alpha(RED(), 0.16) : V.alpha(C.muted, 0.1), stroke: cur ? C.d : rep ? RED() : V.alpha(C.muted, 0.35), color: rep ? RED() : C.ink });
          x += tw0 + 4;
        });
        const top = y + 34;
        // colunas do vocabulário
        const L = narrow ? 44 : 110, cw = (w - L - 12) / VOC.length;
        const rowLab = (s, yy) => V.text(g, s, 10, yy, { size: 11, color: C.muted, base: 'middle', maxW: L - 14 });
        const ch = 22;
        rowLab(narrow ? 'tok' : 'token', top + ch / 2);
        rowLab('a', top + ch + 4 + ch / 2);
        rowLab(narrow ? 'p' : 'p = softmax', top + 2 * (ch + 4) + 36);
        const pbH = 72, pbY = top + 2 * (ch + 4) + 4;
        const gy = pbY + pbH + 10;
        rowLab(narrow ? '−∇' : '−∇ = x*−m⊙p', gy + ch / 2);
        const gmax = Math.max(0.05, ...gr.map(Math.abs));
        const nset = negs();
        VOC.forEach((tok, i) => {
          const x0 = L + i * cw;
          const isT = i === TGT, isN = nset.includes(i);
          chip(g, x0 + 2, top, cw - 4, ch, tok, { size: Math.min(11.5, cw * 0.24), fill: isT ? V.alpha(C.e, 0.25) : V.alpha(C.muted, 0.08), stroke: isT ? C.e : isN ? RED() : V.alpha(C.muted, 0.3), lw: isN || isT ? 2 : 1.2 });
          const av = S.a[i];
          g.fillStyle = V.cmap(V.clamp(av / 4, -1, 1), 'div'); V.rr(g, x0 + 2, top + ch + 4, cw - 4, ch, 4); g.fill();
          V.text(g, fmt(av, 2), x0 + cw / 2, top + ch + 4 + ch / 2 + 0.5, { size: Math.min(11, cw * 0.24), font: 'mono', color: C.ink, align: 'center', base: 'middle' });
          const hh = Math.min(1, p[i] / 0.75) * pbH;
          g.fillStyle = isT ? C.e : isN ? V.alpha(RED(), 0.8) : V.alpha(C.a, 0.6); V.rr(g, x0 + cw * 0.2, pbY + pbH - hh, cw * 0.6, Math.max(1, hh), 3); g.fill();
          V.text(g, fmt(p[i], 2), x0 + cw / 2, pbY + pbH - hh - 3, { size: 10, font: 'mono', color: C.ink, align: 'center' });
          const gv = gr[i];
          g.fillStyle = V.cmap(V.clamp(gv / gmax, -1, 1), 'div'); V.rr(g, x0 + 2, gy, cw - 4, ch, 4); g.fill();
          V.text(g, fmtS(gv, 2), x0 + cw / 2, gy + ch / 2 + 0.5, { size: Math.min(11, cw * 0.24), font: 'mono', color: C.ink, align: 'center', base: 'middle' });
          // seta animada sobre a barra
          const al = Math.min(22, Math.abs(gv) / gmax * 22);
          if (al > 3 && cw > 60) { const bob = (t * 30) % 6; const xx = x0 + cw - 7; V.arrow(g, xx, pbY + pbH / 2 + (gv > 0 ? al / 2 : -al / 2) - (gv > 0 ? bob : -bob) * 0.3, xx, pbY + pbH / 2 + (gv > 0 ? -al / 2 : al / 2), { color: gv > 0 ? C.e : RED(), lw: 1.8, head: 5 }); }
        });
        if (S.neg === 'one' && S.mode === 'ul') {
          const pn = p[NEG1], mo = 1 - S.alpha * pn / (1 - pn);
          if (narrow) {
            V.text(g, 'm = ' + fmt(mo, 3) + ' (outros) · m(subiu) = ' + fmt(1 + S.alpha, 1), 10, gy + ch + 13, { size: 10.5, font: 'mono', color: C.muted, maxW: w - 20 });
            if (mo < 0) V.text(g, '→ m < 0: os outros tokens sobem', 10, gy + ch + 26, { size: 10.5, font: 'mono', color: C.d, maxW: w - 20 });
          } else V.text(g, 'm = ' + fmt(mo, 3) + ' nos outros · m(subiu) = 1+α = ' + fmt(1 + S.alpha, 1) + (mo < 0 ? '  → m < 0: os outros tokens sobem (o negativo é drenado)' : ''), L, gy + ch + 14, { size: 10.5, font: 'mono', color: mo < 0 ? C.d : C.muted, maxW: w - L - 10 });
        }
        // curva −log(1−p) × −log p
        const P = { x: 44, y: gy + ch + 36, w: w - 60, h: h - (gy + ch + 36) - 34 };
        if (P.h > 50) {
          const ax = axes(g, P.x, P.y, P.w, P.h, { xmin: 0, xmax: 1, ymin: 0, ymax: 4, xt: [0, 0.5, 1], yt: [0, 2, 4], xf: (v) => fmt(v, 1), yf: (v) => fmt(v, 0), xl: 'p' });
          const c1 = [], c2 = [];
          for (let k = 1; k < 100; k++) { const pp = k / 100; c1.push([ax.X(pp), ax.Y(Math.min(4, -Math.log(1 - pp)))]); c2.push([ax.X(pp), ax.Y(Math.min(4, -Math.log(pp)))]); }
          polyline(g, c2, V.alpha(C.a, 0.8), 2, [5, 4]); polyline(g, c1, RED(), 2.4);
          [0.05, 0.3, 0.9].forEach((pp) => { const v = -Math.log(1 - pp); V.dot(g, ax.X(pp), ax.Y(v), 3.5, C.d); V.text(g, fmt(v, 4), ax.X(pp) + (pp > 0.8 ? -6 : 6), ax.Y(v) - 6, { size: 10, font: 'mono', color: C.d, align: pp > 0.8 ? 'right' : 'left' }); });
          const pn = p[NEG1];
          V.glow(g, ax.X(pn), ax.Y(Math.min(4, -Math.log(1 - pn))), 12, RED(), 0.8); V.dot(g, ax.X(pn), ax.Y(Math.min(4, -Math.log(1 - pn))), 4.5, RED());
          V.text(g, '−log(1−p): pune o negativo', P.x + 6, P.y + 2, { size: 11, color: RED(), base: 'top' });
          V.text(g, '−log p: puxa o alvo', P.x + P.w, P.y + 2, { size: 11, color: C.a, align: 'right', base: 'top' });
        }
      });
    })();

    /* ======================================== FIG 10 — mapa de parentesco */
    (function () {
      const fig = root.querySelector('#fig-parentesco');
      adaptH(fig, 500, 760);
      // fam: 'hub' | 'rl' | 'reg' | 'con'
      const NODES = [
        { id: 'pi', label: 'π* ∝ π_ref·e^(r/β)', fam: 'hub', p: [0.5, 0.36], m: [0.6, 0.12], d: 'O ótimo inclinado do objetivo com KL. RL, regressão e reparametrização são três estradas até ele.', req: null },
        { id: 'ppo', label: 'PPO / GRPO', fam: 'rl', p: [0.44, 0.05], m: [0, 0], d: 'Amostra da própria política e sobe o gradiente: minimiza o KL reverso até π*. Usa vantagens positivas e negativas.', req: ['~', '✓', '✓', '✓', 'prompts + recompensa'] },
        { id: 'rwr', label: 'RWR', fam: 'reg', p: [0.06, 0.06], m: [0.6, 0.8], d: 'Peters & Schaal 2007: máxima verossimilhança das próprias ações, pesadas por e^(R/β).', req: ['✗', '✓', '✓', '✗', 'retorno escalar'] },
        { id: 'awr', label: 'AWR', fam: 'reg', p: [0.2, 0.24], m: [0.2, 0.8], d: 'Peng et al. 2019: pesos e^((R−V)/β) com corte ω_max; duas regressões por iteração (V e política). KL direto até π*.', req: ['✗', '✓ + V', '✓', '✗', 'retorno escalar'] },
        { id: 'awac', label: 'AWAC', fam: 'reg', p: [0.04, 0.42], m: [-0.1, 0.68], d: 'Nair et al.: o mesmo peso, com vantagem de um crítico Q; offline → online.', req: ['✗', '✓ + Q', '✓', '✗', 'transições'] },
        { id: 'raft', label: 'RAFT · best-of-K', fam: 'reg', p: [0.1, 0.66], m: [0.2, 0.97], d: 'O limite β → 0: gera K, fica com a melhor, faz SFT. KL do best-of-n ≤ log n − (n−1)/n (limite; folgado com recompensa binária).', req: ['✗', '✓', '✓', '✗', 'prompts + RM'] },
        { id: 'rest', label: 'ReST · ReST-EM', fam: 'reg', p: [0.07, 0.93], m: [0.2, 0.33], d: 'Grow + Improve com limiares crescentes (ReST); EM com recompensa binária (ReST-EM). Com p(O=1) ∝ e^(r/β), o passo E é a inclinação com a política atual como referência.', req: ['✗', '✓', '✓', '✗', 'prompts + recompensa'] },
        { id: 'rft', label: 'RFT · STaR', fam: 'reg', p: [0.32, 0.84], m: [0, 0.2], d: 'Filtro binário com verificador: gera, mantém as corretas, faz SFT. STaR racionaliza os erros com a resposta como dica.', req: ['✗', 'verificador', '✓', '✗', 'prompts verificáveis'] },
        { id: 'sft', label: 'SFT', fam: 'reg', p: [0.28, 0.55], m: [-0.15, 0.45], d: 'O limite β → ∞: pesos iguais. Máxima verossimilhança em demonstrações; KL direto até os dados.', req: ['✗', '✗', '✗', '✗', 'demonstrações'] },
        { id: 'dpo', label: 'DPO', fam: 'con', p: [0.68, 0.56], m: [0.55, 0.52], d: 'Inverte π*, escreve r = β log π/π_ref + β log Z e ajusta Bradley–Terry nos pares; Z cancela.', req: ['✓', '✗', '✗', '✓', 'pares'] },
        { id: 'fam58', label: 'IPO · KTO · SimPO', fam: 'con', p: [0.56, 0.86], m: [1.01, 0.52], d: 'As variações do capítulo 58: alvo finito (IPO), feedback binário (KTO), média por token sem referência (SimPO).', req: ['~', '✗', '✗', '✓', 'pares / binário'] },
        { id: 'infonce', label: 'InfoNCE', fam: 'con', p: [0.87, 0.46], m: [1.01, 0.435], d: 'Um positivo contra N−1 negativos numa softmax. Com 1 negativo e escore β log π/π_ref, é o DPO.', req: ['✗', '✗', '✗', '✓', 'positivo + negativos'] },
        { id: 'cpl', label: 'CPL', fam: 'con', p: [0.92, 0.18], m: [1.01, 0.349], d: 'Preferência por arrependimento + A* = α log π*: contraste supervisionado sobre segmentos. DPO é caso especial.', req: ['✗', '✗', '✗', '✓', 'pares de segmentos'] },
        { id: 'cpo', label: 'CPO', fam: 'con', p: [0.95, 0.76], m: [1.01, 0.264], d: 'DPO com referência uniforme + NLL nos escolhidos. É o CPL no MDP de tokens (derivação).', req: ['✗', '✗', '✗', '✓', 'pares'] },
        { id: 'slic', label: 'SLiC-HF', fam: 'con', p: [0.7, 0.07], m: [1.01, 0.606], d: 'Dobradiça com margem δ sobre log-probs cruas + entropia cruzada como âncora.', req: ['✗', '~', '~', '✓', 'pares'] },
        { id: 'ul', label: 'Unlikelihood', fam: 'con', p: [0.82, 0.95], m: [1.01, 0.692], d: '−log(1−p) nos candidatos proibidos + NLL. Sem peso adaptativo, o baseline do DPO degenera.', req: ['✗', '✗', '~', '✓', 'texto + negativos'] },
      ];
      const E = [
        ['pi', 'ppo', 'KL reverso: amostra de π_θ', 0], ['pi', 'awr', 'KL direto: pesa por e^(A/β)', 0], ['pi', 'dpo', 'inverte r e cancela Z', 0],
        ['rwr', 'awr', 'R → R − V (linha de base)', 0], ['awr', 'awac', 'vantagem de um crítico Q', 0], ['awr', 'raft', 'β → 0: argmax', 1], ['awr', 'sft', 'β → ∞: pesos iguais', 1],
        ['rest', 'pi', 'passo E com p(O=1) ∝ e^(r/β)', 1], ['rft', 'rest', 'peso binário 𝟙[correto]', 0], ['sft', 'rft', 'filtra as corretas', 0],
        ['infonce', 'dpo', '1 negativo + escore β log π/π_ref', 1], ['cpl', 'dpo', 'bandit contextual: caso especial', 0], ['cpl', 'infonce', 'rankings + Plackett–Luce', 0],
        ['cpl', 'cpo', 'MDP de tokens, γ=1: referência uniforme', 1], ['slic', 'dpo', 'log-prob → log-razão: hinge do RSO/TRL', 0], ['ul', 'dpo', 'sem peso adaptativo: baseline do DPO', 0],
        ['dpo', 'fam58', 'variações (cap. 58)', 0], ['rft', 'ppo', 'PG binário só com os positivos', 1],
      ];
      const byId = Object.fromEntries(NODES.map((n) => [n.id, n]));
      const S = { sel: 'pi', view: 'graph', hover: null };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'visão', options: [{ v: 'graph', label: 'parentesco' }, { v: 'req', label: 'o que cada um exige' }], value: 'graph', onChange: (v) => { S.view = v; } });
      const sel = UI.select(ctl, { label: 'método', options: NODES.map((n) => ({ v: n.id, label: n.label })), value: 'pi', onChange: (v) => { S.sel = v; } });
      UI.legend(ctl, [['var(--viz-a)', 'regressão'], ['var(--viz-c)', 'contraste'], ['var(--viz-b)', 'RL'], ['var(--viz-d)', 'ótimo']]);
      const famCol = (f) => { const C = V.C(); return f === 'reg' ? C.a : f === 'con' ? C.c : f === 'rl' ? C.b : C.d; };
      V.fig(fig, (g, w, h, t, dt, _s, H) => {
        const C = V.C();
        const narrow = w < 640;
        const pnlH = narrow ? 124 : 100;
        const area = { x: 10, y: 10, w: w - 20, h: h - pnlH - 20 };
        if (S.view === 'graph') {
          const pos = (n) => { const q = narrow ? n.m : n.p; return [area.x + 60 + q[0] * (area.w - 120), area.y + 16 + q[1] * (area.h - 32)]; };
          const nb = new Set([S.sel]);
          E.forEach(([a, b]) => { if (a === S.sel) nb.add(b); if (b === S.sel) nb.add(a); });
          const rects = {};
          NODES.forEach((n) => { const [x, y] = pos(n); const fs = narrow ? 10.5 : 11.5; const ww = tw(g, n.label, fs, 'mono', 600) + 16; rects[n.id] = { x: x - ww / 2, y: y - 12, w: ww, h: 24, cx: x, cy: y, fs }; });
          // arestas
          E.forEach(([a, b, lab, der], k) => {
            const A = rects[a], B = rects[b];
            const on = a === S.sel || b === S.sel;
            const col = on ? C.d : V.alpha(C.muted, 0.35);
            V.line(g, A.cx, A.cy, B.cx, B.cy, { color: col, lw: on ? 2.2 : 1.2, dash: der ? [5, 4] : null });
            const ph = (t * (on ? 0.5 : 0.25) + k * 0.13) % 1;
            const pp = V.along(A.cx, A.cy, B.cx, B.cy, ph);
            V.dot(g, pp[0], pp[1], on ? 3.5 : 2, on ? C.d : V.alpha(C.muted, 0.6));
          });
          NODES.forEach((n) => {
            const r = rects[n.id], col = famCol(n.fam), on = nb.has(n.id), me = n.id === S.sel;
            if (me) V.glow(g, r.cx, r.cy, r.w * 0.6, col, 0.5 + 0.2 * Math.sin(t * 3));
            V.box(g, r.x, r.y, r.w, r.h, { r: 12, fill: C.panel || C.bg }); // base opaca: as arestas passam por trás do nó, não por cima do rótulo
            V.box(g, r.x, r.y, r.w, r.h, { r: 12, fill: V.alpha(col, me ? 0.45 : on ? 0.25 : 0.1), stroke: on ? col : V.alpha(col, 0.5), lw: me ? 2.4 : 1.4 });
            V.text(g, n.label, r.cx, r.cy + 0.5, { size: r.fs, font: 'mono', weight: 600, color: on ? C.ink : V.alpha(C.ink, 0.6), align: 'center', base: 'middle' });
          });
          // rótulos das arestas do selecionado (com desvio simples de colisões)
          const placed = Object.values(rects).map((r) => ({ x: r.x - 2, y: r.y - 2, w: r.w + 4, h: r.h + 4 }));
          const hitR = (a, b) => a.x < b.x + b.w && a.x + a.w > b.x && a.y < b.y + b.h && a.y + a.h > b.y;
          E.forEach(([a, b, lab, der]) => {
            if (a !== S.sel && b !== S.sel) return;
            const A = rects[a], B = rects[b];
            const fs = narrow ? 9.5 : 10.5; const txt = lab + (der ? ' *' : '');
            const ww = Math.min(tw(g, txt, fs) + 10, narrow ? 170 : 250);
            // procura um lugar livre perto da aresta; se não houver, o de menor sobreposição
            const ovl = (a, b) => Math.max(0, Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x)) * Math.max(0, Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y));
            let best = null, bestO = Infinity;
            for (const u of [0.5, 0.38, 0.62, 0.28, 0.72, 0.2, 0.8]) {
              const mx = V.lerp(A.cx, B.cx, u), my = V.lerp(A.cy, B.cy, u);
              for (const dy of [0, -14, 14, -26, 26, -38, 38]) {
                const rc = { x: V.clamp(mx - ww / 2, 4, w - ww - 4), y: V.clamp(my - 9 + dy, area.y, area.y + area.h - 18), w: ww, h: 18 };
                const o = placed.reduce((acc, q) => acc + ovl(rc, q), 0) + Math.abs(dy) * 0.5;
                if (o < bestO) { bestO = o; best = rc; }
                if (o === Math.abs(dy) * 0.5 && !placed.some((q) => hitR(rc, q))) break;
              }
              if (best && !placed.some((q) => hitR(best, q))) break;
            }
            placed.push(best);
            V.box(g, best.x, best.y, best.w, best.h, { r: 6, fill: V.alpha(C.panel || C.bg, 0.95), stroke: V.alpha(C.d, 0.7), lw: 1 });
            V.text(g, txt, best.x + best.w / 2, best.y + 9.5, { size: fs, color: C.ink, align: 'center', base: 'middle', maxW: best.w - 6 });
          });
          // clique
          const P = H.pointer;
          if (P.clicked) { const hit = NODES.find((n) => { const r = rects[n.id]; return P.x >= r.x && P.x <= r.x + r.w && P.y >= r.y && P.y <= r.y + r.h; }); if (hit) { S.sel = hit.id; sel.el.querySelector('select').value = hit.id; } }
          if (P.inside) { const hov = NODES.find((n) => { const r = rects[n.id]; return P.x >= r.x && P.x <= r.x + r.w && P.y >= r.y && P.y <= r.y + r.h; }); H.canvas.style.cursor = hov ? 'pointer' : 'default'; }
        } else {
          const cols = narrow ? ['π_ref', 'RM', 'gera', 'neg.', 'dado'] : ['π_ref no treino', 'RM / verificador', 'gera no treino', 'usa negativos', 'dado'];
          const rowsN = NODES.filter((n) => n.req);
          const lw0 = narrow ? 96 : 170;
          const symW = narrow ? 31 : (area.w - lw0) / 5.8, dataW = area.w - lw0 - 4 * symW;
          const colX = (k) => area.x + lw0 + k * symW, colW = (k) => (k === 4 ? dataW : symW);
          const rh = Math.min(25, (area.h - 66) / rowsN.length);
          cols.forEach((c, k) => { V.wrap(g, c, colX(k) + colW(k) / 2, area.y + 8, colW(k) - 4, { size: narrow ? 10 : 10.5, color: C.muted, align: 'center', lineH: 11, maxLines: 2 }); });
          rowsN.forEach((n, i) => {
            const y = area.y + 30 + i * rh;
            const me = n.id === S.sel;
            if (me) { g.fillStyle = V.alpha(C.d, 0.12 + 0.08 * Math.sin(t * 3)); V.rr(g, area.x, y, area.w, rh - 2, 5); g.fill(); const sx = area.x + ((t * 0.35) % 1) * area.w; V.glow(g, sx, y + rh / 2, rh, C.d, 0.35); }
            V.text(g, n.label, area.x + 4, y + rh / 2, { size: narrow ? 10 : 11, font: 'mono', weight: 600, color: famCol(n.fam), base: 'middle', maxW: lw0 - 8 });
            n.req.forEach((v, k) => {
              const x = colX(k);
              const ww = colW(k);
              const col = v === '✓' || v.startsWith('✓') || v === 'verificador' ? C.e : v === '✗' ? V.alpha(C.muted, 0.6) : k === 4 ? C.b : C.d;
              g.fillStyle = k === 4 ? V.alpha(C.b, 0.1) : V.alpha(col, v === '✗' ? 0.08 : 0.22); V.rr(g, x + 2, y + 1, ww - 4, rh - 4, 4); g.fill();
              V.text(g, narrow && v === 'verificador' ? 'verif.' : v, x + ww / 2, y + rh / 2, { size: narrow ? 9.5 : 10.5, font: k === 4 ? 'sans' : 'mono', color: k === 4 ? C.ink : col, align: 'center', base: 'middle', maxW: ww - 6 });
            });
            if (H.pointer.clicked && H.pointer.y >= y && H.pointer.y < y + rh) { S.sel = n.id; sel.el.querySelector('select').value = n.id; }
          });
          V.wrap(g, narrow ? '✓ sim · ✗ não · ~ depende da variante. π_ref: referência no treino · RM: reward model ou verificador · gera: amostra durante o treino · neg.: usa exemplos negativos' : '✓ sim · ✗ não · ~ depende da variante (GRPO dispensa a referência com β = 0; KTO e IPO usam π_ref, SimPO não; SLiC-HF sample-rank usa um ranker e amostra do SFT)', area.x + 4, area.y + 30 + rowsN.length * rh + 14, area.w - 8, { size: 10.5, color: C.muted, lineH: 13, maxLines: 3 });
        }
        // painel de descrição
        const n = byId[S.sel];
        const py = h - pnlH - 4;
        V.box(g, 10, py, w - 20, pnlH, { r: 10, fill: V.alpha(famCol(n.fam), 0.1), stroke: V.alpha(famCol(n.fam), 0.6), lw: 1.4 });
        V.text(g, n.label, 22, py + 18, { size: 13, font: 'mono', weight: 700, color: famCol(n.fam) });
        const hgt = V.wrap(g, n.d, 22, py + 38, w - 44, { size: narrow ? 11.5 : 12.5, color: C.ink, lineH: narrow ? 15 : 17, maxLines: narrow ? 4 : 2 });
        const rel = E.filter(([a, b]) => a === S.sel || b === S.sel).map(([a, b, , der]) => byId[a === S.sel ? b : a].label + (der ? ' *' : ''));
        V.wrap(g, 'parentes diretos: ' + rel.join(' · ') + (rel.some((x) => x.endsWith('*')) ? '   (* = derivação nossa)' : ''), 22, py + 38 + hgt + 4, w - 44, { size: narrow ? 10.5 : 11.5, color: C.muted, lineH: 14, maxLines: 2 });
      });
    })();
  },
});

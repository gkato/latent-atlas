LA.chapter({
  id: 'ipo-kto-simpo',
  part: 12,
  order: 58,
  title: 'IPO, KTO e SimPO: variações sobre o tema',
  navTitle: 'IPO, KTO & SimPO',
  subtitle: 'Três suposições do DPO, três consertos: um alvo finito para a margem (IPO), um polegar para cima ou para baixo no lugar dos pares (KTO) e uma recompensa sem referência, por token e com margem (SimPO). E a família inteira calculada no mesmo par.',
  minutes: 45,
  recipes: [],
  tags: ['IPO', 'ΨPO', 'KTO', 'teoria do prospecto', 'HALO', 'SimPO', 'ORPO', 'CPO', 'R-DPO', 'β-DPO', 'DPOP', 'normalização por comprimento', 'TRL'],
  goals: [
    'Explicar por que o DPO sobreajusta preferências determinísticas e como o <b>IPO</b> (ΨPO com Ψ = identidade) troca o logit por uma regressão da margem para o alvo finito \\(1/(2\\tau)\\)',
    'Derivar a perda do <b>KTO</b> a partir da teoria do prospecto, estimar o ponto de referência \\(z_0\\) no microbatch e balancear \\(\\lambda_D, \\lambda_U\\) para dados binários desbalanceados',
    'Entender por que o <b>SimPO</b> dispensa a referência, normaliza por comprimento e exige uma margem \\(\\gamma\\), e que riscos isso traz',
    'Situar ORPO, CPO, R-DPO, β-DPO e DPOP no molde “escore + contraste + âncora” e calcular todas as perdas num mesmo par',
    'Escolher e configurar a variante (TRL ou MLX) pelos dados, pela memória e pelos riscos, sabendo o que monitorar',
  ],
  prereqs: ['dpo'],
  summary: [
    'A família cabe em <b>escore + contraste + âncora</b>. No DPO: escore \\(\\beta\\log\\frac{\\pi_\\theta}{\\pi_{\\text{ref}}}\\), contraste logístico, âncora na referência. Cada variante troca uma peça.',
    '<b>IPO</b> (Identity Preference Optimization; o “Identity Policy Optimization” que circula é o mesmo método): o DPO é o ΨPO com Ψ = logit, que pede margem infinita quando \\(p=1\\), e um rótulo por par torna todo par determinístico. O IPO regride \\(\\Delta h\\) para \\(1/(2\\tau)\\) com \\((\\Delta h-\\frac{1}{2\\tau})^2\\), e o gradiente puxa de volta além do alvo. No bandit de 3 ações com \\(\\tau=1\\): DPO → (1; 0; 0), IPO → (0,448; 0,321; 0,230).',
    'O TRL calcula o IPO com log-razões <b>médias por token</b>: o mesmo \\(\\tau\\) muda de sentido entre implementações (\\(\\tau=0{,}1\\) pede 5 nats por token).',
    '<b>KTO</b>: valor logístico relativo a \\(z_0\\) (KL estimado em pares desencontrados do microbatch, com clamp em 0 e sem gradiente), pesos com \\(\\lambda_Dn_D/\\lambda_Un_U\\in[1;\\,1{,}5]\\) e dados binários. LR ~5e-6 (2–10× o do DPO), microbatch ≥ 2. Iguala o DPO com até 90% menos desejáveis; a janela \\(\\sigma(1-\\sigma)\\) ignora exemplos extremos.',
    '<b>SimPO</b>: recompensa \\(\\frac{\\beta}{|y|}\\log\\pi_\\theta\\) (média, sem referência) e margem \\(\\gamma\\); \\(\\beta\\) 2–2,5 (até 10), \\(\\gamma/\\beta\\approx0{,}5\\), LR 3e-7–1e-6. Economiza memória (Llama-3 8B: 77 GB no DPO, 69 GB no SimPO), mas sem KL um LR alto causa esquecimento (GSM 78,5 → 57,4).',
    'ORPO (odds + SFT, sem referência), CPO (referência uniforme + NLL), R-DPO (\\(+\\alpha(|y_w|-|y_l|)\\)), β-DPO (β por lote) e DPOP (penalidade se \\(\\pi_\\theta(y_w)\\) cai) completam a família. No par de brinquedo, os métodos sem referência e sem normalização veem a preferida longa como “errada”.',
    'Escolha: pares + SFT → DPO; rótulos únicos ou ruidosos → IPO; bom/ruim → KTO; memória curta → SimPO ou ORPO; verbosidade → R-DPO ou DPO normalizado; deslocamento → DPOP ou NLL. Monitore margens, <code>logps/chosen</code>, comprimento e win rate por checkpoint.',
  ],
  quiz: [
    { q: 'Num dataset com um único rótulo por par, por que o DPO tende a sobreajustar mesmo com \\(\\beta\\) grande?', o: ['Porque \\(\\beta\\) grande aumenta o learning rate efetivo', 'Porque com um rótulo a frequência empírica \\(\\hat p\\) só vale 0 ou 1, e a margem ótima do DPO, \\(\\operatorname{logit}(p)/\\beta\\), é infinita para \\(p=1\\) qualquer que seja \\(\\beta\\)', 'Porque o DPO não usa modelo de referência', 'Porque a perda do DPO é quadrática e explode longe do alvo'], a: 1, e: 'O logit é ilimitado: para \\(p=1\\) o ótimo exige \\(\\Delta h\\to\\infty\\) e a política zera \\(y_l\\), com qualquer \\(\\beta\\). Com um rótulo por par, todo par é “determinístico”. O IPO troca o logit pela identidade e o alvo vira \\((2p-1)/(2\\tau)\\), limitado por \\(1/(2\\tau)\\). A perda quadrática é a do IPO, não a do DPO.' },
    { q: 'Com \\(\\tau=0{,}5\\) (forma do paper, log-probs somadas), o par de brinquedo tem \\(\\Delta h=3\\). O que a descida do gradiente do IPO faz com a margem?', o: ['Aumenta, como no DPO, porque a perda ainda é positiva', 'Nada: a perda do IPO é zero acima do alvo', 'Aumenta até \\(1/\\tau=2\\)', 'Diminui: o alvo é \\(1/(2\\tau)=1\\), o par já passou dele e a derivada \\(2(\\Delta h-1)=+4\\) é positiva'], a: 3, e: 'A parábola \\((\\Delta h-1)^2\\) vale 4 em \\(\\Delta h=3\\), com derivada +4: a descida reduz a margem, subindo \\(y_l\\) e descendo \\(y_w\\). É exatamente o que o DPO nunca faz, porque o peso dele só tende a zero. Com \\(\\tau=0{,}1\\) (alvo 5), o mesmo par ainda seria empurrado para a frente (derivada −4).' },
    { q: 'Você copia \\(\\tau=0{,}1\\) de um experimento com log-probs somadas para um IPO no <code>DPOTrainer</code> do TRL. Qual é o problema?', o: ['O TRL normaliza por token; \\(\\tau=0{,}1\\) passa a pedir uma margem de 5 nats <i>por token</i>, um afastamento enorme da referência', 'Nenhum: \\(\\tau\\) é adimensional', 'O TRL não implementa IPO', 'O TRL interpreta <code>beta</code> como learning rate'], a: 0, e: 'Na forma normalizada, \\(\\Delta h\\) é ~\\(|y|\\) vezes menor. No par de brinquedo, \\(\\Delta h_{\\text{norm}}=0{,}25\\): com \\(\\tau=0{,}1\\) a perda é 22,56; com \\(\\tau=2\\), zero. Escalas de \\(\\tau\\) entre implementações não são comparáveis.' },
    { q: 'Por que o KTO estima \\(z_0\\) com pares desencontrados \\((x_i, y_{i+1})\\) em vez dos próprios \\((x_i, y_i)\\)?', o: ['Para economizar memória', 'Porque \\(z_0\\) precisa receber gradiente', 'Porque as \\(y_i\\) foram escolhidas por serem canonicamente boas ou ruins, e as recompensas delas não representam o KL típico da política; respostas de outros prompts aproximam a média sem esse viés de seleção', 'Porque o TRL exige batch 1'], a: 2, e: 'O paper explica que os exemplos casados têm magnitudes nada representativas. O estimador usa a média dos desencontrados com clamp em 0 (enviesado para cima, de menor variância), custa um forward extra e não recebe gradiente. No TRL isso exige batch por dispositivo maior que 1 e amostragem sequencial; com batch 1, o “KL” vira a própria recompensa.' },
    { q: 'Você tem 1.000 rótulos binários: 100 desejáveis e 900 indesejáveis. Com \\(\\lambda_U=1\\), que \\(\\lambda_D\\) a regra do paper do KTO sugere?', o: ['1', '0,11', 'Entre 9 e 13,5', '100'], a: 2, e: 'A Eq. 9 pede \\(\\lambda_Dn_D/(\\lambda_Un_U)\\in[1;\\,1{,}5]\\): \\(\\lambda_D\\in[900/100;\\ 1{,}5\\cdot900/100]=[9;\\ 13{,}5]\\). É o mesmo raciocínio do exemplo do paper (1:10 → \\(\\lambda_D\\in[10, 15]\\)). Exceção: em tarefas como toxicidade, em que evitar o pior importa mais, pode compensar ficar abaixo de 1.' },
    { q: 'No par de brinquedo (\\(y_w\\): 20 tokens, soma −25; \\(y_l\\): 10 tokens, soma −17), por que um “SimPO com soma” daria perda ≈ 17 com \\(\\beta=2\\) e \\(\\gamma=1\\)?', o: ['Porque a soma faz a preferida, mais longa, parecer muito pior (logit \\(2\\cdot(-25+17)-1=-17\\)); a média (−1,25 contra −1,70) mostra que ela é melhor por token', 'Porque a soma ignora \\(\\gamma\\)', 'Porque o SimPO precisa de referência para somar', 'Porque \\(\\beta=2\\) é grande demais para qualquer par'], a: 0, e: 'Cada token acrescenta um termo negativo, então a soma pune o comprimento. Com a média, \\(u=2\\cdot(-1{,}25+1{,}70)-1=-0{,}1\\) e a perda é 0,7444. A ablação sem normalização do paper confirma: Spearman log-prob × comprimento sobe de 0,34 para 0,82 e o LC do AlpacaEval 2 cai de 21,5 para 11,9.' },
  ],
  mount(root, LA) {
    const V = LA.viz, UI = LA.ui;

    /* =========================================================== helpers */
    const fmt = (x, d = 2) => {
      if (x === Infinity) return '∞';
      if (x === -Infinity) return '−∞';
      return (+x).toFixed(d).replace('.', ',').replace(/^-/, '−');
    };
    const sg = (x, d = 2) => (x >= 0 ? '+' : '−') + fmt(Math.abs(x), d);
    const fmtN = (v) => { const r = Math.round(v * 100) / 100; return Number.isInteger(r) ? String(r).replace('-', '−') : fmt(r, Math.abs(r * 10 - Math.round(r * 10)) < 1e-9 ? 1 : 2); };
    const sig = (z) => 1 / (1 + Math.exp(-z));
    const nls = (z) => (z > 35 ? Math.exp(-z) : z < -35 ? -z : Math.log1p(Math.exp(-z))); // −log σ(z)
    const logit = (p) => Math.log(p / (1 - p));
    const clamp = V.clamp, lerp = V.lerp;
    const nocase = (c) => { const el = c && (c.el || c); if (el && el.querySelectorAll) { el.querySelectorAll('label, span').forEach((n) => { if (!n.classList.contains('val')) n.style.textTransform = 'none'; }); } return c; };
    function adaptH(fig, wide, narrow, bp = 640) {
      const st = fig.querySelector('.la-fig-stage');
      const set = () => { const want = (fig.clientWidth < bp ? narrow : wide) + 'px'; if (st.style.height !== want) st.style.height = want; };
      set();
      const ro = new ResizeObserver(set); ro.observe(fig);
      if (LA.onCleanup) LA.onCleanup(() => ro.disconnect());
    }
    function tw(g, s, size, font = 'sans', weight = 400) { g.font = V.font(size, font, weight); return g.measureText(String(s)).width; }
    function niceStep(range, n = 4) {
      const raw = range / n, p = Math.pow(10, Math.floor(Math.log10(raw)));
      const m = raw / p;
      return (m < 1.5 ? 1 : m < 3.5 ? 2 : m < 7.5 ? 5 : 10) * p;
    }
    function nice(a, b, n) { const st = niceStep(b - a, n); const s = Math.ceil(a / st - 1e-9) * st; const out = []; for (let v = s; v <= b + 1e-9; v += st) out.push(+v.toFixed(10)); return out; }
    function axes(g, A) {
      const C = V.C();
      const { x, y, w, h, xr, yr } = A;
      const X = (v) => x + ((v - xr[0]) / (xr[1] - xr[0])) * w;
      const Y = (v) => y + h - ((v - yr[0]) / (yr[1] - yr[0])) * h;
      const xt = A.xt || nice(xr[0], xr[1], A.nx || 5), yt = A.yt || nice(yr[0], yr[1], A.ny || 4);
      g.strokeStyle = C.grid; g.lineWidth = 1; g.beginPath();
      xt.forEach((v) => { const px = Math.round(X(v)) + 0.5; g.moveTo(px, y); g.lineTo(px, y + h); });
      yt.forEach((v) => { const py = Math.round(Y(v)) + 0.5; g.moveTo(x, py); g.lineTo(x + w, py); });
      g.stroke();
      V.line(g, x, y + h, x + w, y + h, { color: V.alpha(C.muted, 0.55), lw: 1 });
      V.line(g, x, y, x, y + h, { color: V.alpha(C.muted, 0.55), lw: 1 });
      const fx = A.fx || fmtN, fy = A.fy || fmtN;
      if (!A.noX) xt.forEach((v) => V.text(g, fx(v), X(v), y + h + 14, { size: 11, align: 'center', color: C.muted, font: 'mono' }));
      if (!A.noY) yt.forEach((v) => V.text(g, fy(v), x - 5, Y(v), { size: 11, align: 'right', base: 'middle', color: C.muted, font: 'mono' }));
      if (A.xl) V.text(g, A.xl, x + w, y + h + (A.noX ? 14 : 29), { size: 11.5, align: 'right', color: C.muted });
      if (A.yl) V.text(g, A.yl, x + 2, y - 8, { size: 11.5, align: 'left', color: C.muted });
      return Object.assign({}, A, { X, Y });
    }
    function plotFn(g, A, f, o = {}) {
      const n = o.n || 180, span = A.yr[1] - A.yr[0];
      g.save(); g.beginPath(); g.rect(A.x, A.y - 2, A.w, A.h + 4); g.clip();
      g.beginPath(); let st = false;
      for (let i = 0; i <= n; i++) {
        const xv = A.xr[0] + ((A.xr[1] - A.xr[0]) * i) / n, yv = f(xv);
        if (!isFinite(yv)) { st = false; continue; }
        const px = A.X(xv), py = A.Y(clamp(yv, A.yr[0] - span, A.yr[1] + span));
        if (!st) { g.moveTo(px, py); st = true; } else g.lineTo(px, py);
      }
      g.strokeStyle = o.color; g.lineWidth = o.lw || 2.2; if (o.dash) g.setLineDash(o.dash);
      g.stroke(); g.setLineDash([]); g.restore();
    }
    function panel(g, x, y, w, h, o = {}) { const C = V.C(); V.box(g, x, y, w, h, { r: o.r == null ? 10 : o.r, fill: o.fill || V.alpha(C.panel, 0.55), stroke: o.stroke || V.alpha(C.muted, 0.22), lw: 1 }); }
    function chip(g, x, y, w, h, label, o = {}) {
      const C = V.C();
      V.box(g, x, y, w, h, { r: o.r == null ? 7 : o.r, fill: o.fill || V.alpha(C.muted, 0.12), stroke: o.stroke || V.alpha(C.muted, 0.4), lw: o.lw || 1.2 });
      if (label != null) V.text(g, label, x + w / 2, y + h / 2 + 0.5, { size: o.size || 11.5, font: o.font || 'mono', weight: o.weight || 500, color: o.color || C.ink, align: 'center', base: 'middle', maxW: w - 6 });
    }
    function mark(g, x, y, ok, s = 6, col) {
      const C = V.C();
      g.lineWidth = 2.2; g.strokeStyle = col || (ok ? C.a : C.red); g.lineCap = 'round'; g.beginPath();
      if (ok) { g.moveTo(x - s, y); g.lineTo(x - s * 0.3, y + s * 0.7); g.lineTo(x + s, y - s * 0.8); }
      else { g.moveTo(x - s * 0.8, y - s * 0.8); g.lineTo(x + s * 0.8, y + s * 0.8); g.moveTo(x + s * 0.8, y - s * 0.8); g.lineTo(x - s * 0.8, y + s * 0.8); }
      g.stroke(); g.lineCap = 'butt';
    }
    const MC = () => { const C = V.C(); return { dpo: C.b, ipo: C.a, kto: C.d, simpo: C.c, orpo: C.f, cpo: C.e, rdpo: V.mix(C.b, C.d, 0.5), lndpo: V.mix(C.b, C.c, 0.5), bdpo: V.mix(C.b, C.a, 0.5), dpop: V.mix(C.b, C.e, 0.55) }; };

    /* dados do par de brinquedo (por token, ilustrativos; somas exatas da nota de pesquisa) */
    const TOKW = ['A', 'luz', 'do', 'Sol', 'se', 'espalha', 'no', 'ar', ';', 'o', 'azul', ',', 'de', 'onda', 'curta', ',', 'espalha', 'bem', 'mais', '.'];
    const REFW = [-2.0, -1.6, -0.3, -0.9, -1.2, -2.4, -1.1, -0.8, -2.2, -0.9, -1.5, -1.0, -1.3, -1.4, -0.8, -1.2, -2.0, -1.9, -1.0, -0.5];
    const DLW = [0, 0.2, 0, 0, 0, 0.2, 0, 0, 0, 0, 0.2, 0, 0, 0.1, 0.1, 0, 0.2, 0, 0, 0];
    const TOKL = ['Porque', 'o', 'céu', 'reflete', 'a', 'cor', 'azul', 'do', 'mar', '.'];
    const REFL = [-1.8, -0.6, -1.5, -2.6, -1.1, -1.7, -1.4, -0.9, -2.4, -1.0];
    const DLL = [-0.2, 0, 0, -0.8, 0, -0.3, 0, 0, -0.7, 0];
    const POLW = REFW.map((r, i) => +(r + DLW[i]).toFixed(2)), POLL = REFL.map((r, i) => +(r + DLL[i]).toFixed(2));
    const sum = (a, k) => { let s = 0; for (let i = 0; i < (k == null ? a.length : k); i++) s += a[i]; return s; };

    /* ================================================== FIG 1 — o par token a token */
    (function () {
      const fig = root.querySelector('#fig-par');
      adaptH(fig, 366, 500);
      const S = { show: 'pol', agg: 'soma', t0: 0 };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'mostrar', options: [{ v: 'ref', label: 'log π_ref' }, { v: 'pol', label: 'log π_θ' }, { v: 'h', label: 'h = diferença' }], value: 'pol', onChange: (v) => { S.show = v; S.t0 = H.t; upd(); } });
      UI.seg(ctl, { label: 'agregar', options: [{ v: 'soma', label: 'soma' }, { v: 'media', label: 'média por token' }], value: 'soma', onChange: (v) => { S.agg = v; upd(); } });
      UI.button(ctl, '▶ somar de novo', () => { S.t0 = H.t; });
      const rW = UI.readout(ctl, 'agregado y_w', '—'), rL = UI.readout(ctl, 'agregado y_l', '—');
      nocase(ctl);
      const arrs = () => (S.show === 'ref' ? [REFW, REFL] : S.show === 'pol' ? [POLW, POLL] : [DLW, DLL]);
      const agg = (a, k) => { const n = k == null ? a.length : k; if (!n) return 0; const s = sum(a, n); return S.agg === 'soma' ? s : s / n; };
      function upd() { const [A, B] = arrs(); const d = S.agg === 'soma' ? 1 : 2; rW.set(sg(agg(A), d)); rL.set(sg(agg(B), d)); }
      upd();
      const STEP = 0.24;
      const H = V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const narrow = w < 640;
        let el = t - S.t0;
        if (el > 20 * STEP + 5) { S.t0 = t; el = 0; }
        const kW = Math.min(20, Math.floor(el / STEP)), kL = Math.min(10, Math.floor(el / STEP));
        const fr = (el % STEP) / STEP;
        const [A, B] = arrs();
        const isH = S.show === 'h';
        const cellCol = (v) => {
          if (isH) return v > 0 ? V.alpha(C.a, 0.15 + 0.5 * Math.min(1, v / 0.8)) : v < 0 ? V.alpha(C.b, 0.15 + 0.5 * Math.min(1, -v / 0.8)) : V.alpha(C.muted, 0.08);
          const tt = clamp(1 + v / 3.2, 0, 1);
          return V.alpha(S.show === 'ref' ? C.b : C.a, 0.08 + 0.5 * tt);
        };
        V.text(g, 'x = “Por que o céu é azul?”', 14, 20, { size: 12.5, color: C.muted });
        const pad = 14, labW = narrow ? 0 : 50, accW = narrow ? 0 : 156;
        const areaX = pad + labW, areaW = w - pad * 2 - labW - (narrow ? 0 : accW + 14);
        const per = narrow ? 10 : 20, gap = 3;
        const cw = Math.min(48, (areaW - gap * (per - 1)) / per), ch = 30;
        const fs = clamp(cw * 0.38, 10.5, 12);
        const rows = [];
        // posições das células
        function layoutSeq(n, y0) {
          const pos = [];
          for (let i = 0; i < n; i++) { const r = Math.floor(i / per), c = i % per; pos.push([areaX + c * (cw + gap), y0 + r * (ch + 22)]); }
          return pos;
        }
        let yW, yL, accWBox, accLBox, cmpY;
        if (!narrow) {
          yW = 66; yL = 152;
          accWBox = [w - pad - accW, yW - 6, accW, 42]; accLBox = [w - pad - accW, yL - 6, accW, 42];
          cmpY = 214;
        } else {
          yW = 66; const pW = 2 * (ch + 22) - 22;
          accWBox = [pad, yW + pW + 10, w - 2 * pad, 34];
          yL = accWBox[1] + 34 + 46;
          accLBox = [pad, yL + ch + 10, w - 2 * pad, 34];
          cmpY = accLBox[1] + 34 + 22;
        }
        rows.push({ name: 'y_w · preferida · 20 tokens', toks: TOKW, vals: A, k: kW, y: yW, acc: accWBox, col: C.c });
        rows.push({ name: 'y_l · preterida · 10 tokens', toks: TOKL, vals: B, k: kL, y: yL, acc: accLBox, col: C.b });
        rows.forEach((R) => {
          const pos = layoutSeq(R.toks.length, R.y);
          V.text(g, R.name, narrow ? pad : pad, R.y - 30, { size: 12, weight: 600, color: R.col });
          if (!narrow) V.text(g, R.name.slice(0, 3), pad, R.y + ch / 2, { size: 13, weight: 700, color: R.col, base: 'middle', font: 'mono' });
          R.toks.forEach((tk, i) => {
            const [x, y] = pos[i];
            V.text(g, tk, x + cw / 2, y - 5, { size: 10.5, align: 'center', color: C.muted, maxW: cw + gap });
            V.box(g, x, y, cw, ch, { r: 5, fill: cellCol(R.vals[i]), stroke: i < R.k ? V.alpha(R.col, 0.55) : V.alpha(C.muted, 0.25), lw: 1 });
            const v = R.vals[i];
            V.text(g, isH ? (v === 0 ? '0' : sg(v, 1)) : fmt(v, 1), x + cw / 2, y + ch / 2 + 0.5, { size: fs, font: 'mono', align: 'center', base: 'middle', color: C.ink });
            if (i === R.k && R.k < R.toks.length) { g.strokeStyle = C.d; g.lineWidth = 2.4; V.rr(g, x - 2, y - 2, cw + 4, ch + 4, 6); g.stroke(); }
          });
          // número voando para o acumulador
          if (R.k < R.toks.length && R.k > 0) {
            const [x0, y0] = pos[R.k - 1];
            // mira o meio do acumulador (entre o rótulo e o valor) e some antes de chegar,
            // para não se sobrepor ao valor corrente alinhado à direita
            const ax = R.acc[0] + R.acc[2] * 0.5, ay = R.acc[1] + R.acc[3] / 2;
            const e = V.ease.inOut(fr);
            const fx = lerp(x0 + cw / 2, ax, e), fy = lerp(y0 - 8, ay, e) - Math.sin(e * Math.PI) * 22;
            const fa = e < 0.5 ? 1 : Math.max(0, 1 - (e - 0.5) / 0.35);
            if (fa > 0.01) V.text(g, isH ? sg(R.vals[R.k - 1], 1) : fmt(R.vals[R.k - 1], 1), fx, fy, { size: 11.5, font: 'mono', weight: 700, align: 'center', base: 'middle', color: C.d, alpha: fa });
          }
          // acumulador
          const [bx, by, bw, bh] = R.acc;
          panel(g, bx, by, bw, bh, { stroke: V.alpha(R.col, 0.5) });
          const val = agg(R.vals, R.k);
          const lab = S.agg === 'soma' ? 'Σ' : 'média';
          V.text(g, `${lab} (${R.k}/${R.toks.length})`, bx + 10, by + bh / 2, { size: 11.5, color: C.muted, base: 'middle' });
          V.text(g, isH ? sg(val, S.agg === 'soma' ? 1 : 2) : fmt(val, S.agg === 'soma' ? 1 : 2), bx + bw - 10, by + bh / 2 + 0.5, { size: 15, font: 'mono', weight: 700, align: 'right', base: 'middle', color: R.col });
        });
        // comparação final
        const done = kW >= 20;
        const vw = agg(A, kW), vl = agg(B, kL);
        const cy = cmpY, cH = h - cy - 10;
        if (cH > 60) {
          panel(g, pad, cy, w - 2 * pad, cH);
          const fw = agg(A), fl = agg(B);
          const lo = Math.min(0, fw, fl) * 1.12, hi = Math.max(0, fw, fl) * 1.12, span = hi - lo || 1;
          const bx0 = pad + 70, bx1 = w - pad - 16;
          const Xs = (v) => bx0 + ((v - lo) / span) * (bx1 - bx0);
          const by0 = cy + 16, bh = Math.min(24, (cH - 70) / 2);
          [[vw, C.c, 'y_w'], [vl, C.b, 'y_l']].forEach(([v, col, nm], i) => {
            const yy = by0 + i * (bh + 8);
            const vv = v;
            V.text(g, nm, pad + 12, yy + bh / 2, { size: 12, font: 'mono', weight: 700, color: col, base: 'middle' });
            const x0 = Xs(0), x1 = Xs(vv);
            g.fillStyle = V.alpha(col, 0.55); V.rr(g, Math.min(x0, x1), yy, Math.max(2, Math.abs(x1 - x0)), bh, 4); g.fill();
            if (Math.abs(v) > 1e-9) V.text(g, isH ? sg(v, S.agg === 'soma' ? 1 : 2) : fmt(v, S.agg === 'soma' ? 1 : 2), x1 + (v < 0 ? -6 : 6), yy + bh / 2, { size: 11.5, font: 'mono', color: C.ink, base: 'middle', align: v < 0 ? 'right' : 'left' });
          });
          V.line(g, Xs(0), by0 - 4, Xs(0), by0 + 2 * bh + 12, { color: C.muted, lw: 1, dash: [3, 3] });
          let msg;
          if (!done) msg = 'Somando token a token…';
          else if (S.show === 'pol') msg = S.agg === 'soma' ? 'Pela soma de log π_θ, a preferida parece PIOR (−25,0 < −17,0): ela tem o dobro de tokens, e cada token soma um termo negativo.' : 'Pela média por token, a preferida é MELHOR (−1,25 > −1,70): é a métrica que o SimPO e a geração usam.';
          else if (S.show === 'ref') msg = S.agg === 'soma' ? 'Na referência, a preferida também “perde” pela soma (−26,0 contra −15,0). É esse desconto que a divisão por π_ref faz no DPO.' : 'Na referência, por token: −1,30 contra −1,50.';
          else msg = S.agg === 'soma' ? 'h_w = +1,0 e h_l = −2,0, logo Δh = +3,0. Com β = 0,1: logit 0,30, perda do DPO 0,554 e peso σ(−βΔh) = 0,426.' : 'Médias por token: +0,05 e −0,20, logo Δh_norm = 0,25. É a escala que o IPO do TRL usa (alvo 1/(2τ) por token).';
          V.wrap(g, msg, pad + 12, by0 + 2 * bh + 30, w - 2 * pad - 24, { size: 12.5, lineH: 17, color: C.ink, maxLines: 3 });
        }
      });
    })();

    /* ================================================== FIG 2 — logit × identidade */
    (function () {
      const fig = root.querySelector('#fig-psi');
      adaptH(fig, 420, 620);
      const N = 16;
      const S = { p: 0.8, n: 1, b: 0.5, seed: 3, t0: 0, labels: [] };
      function resample() {
        const r = V.rng(S.seed);
        S.labels = Array.from({ length: N }, () => Array.from({ length: S.n }, () => r() < S.p));
        upd();
      }
      const ctl = UI.controls(fig);
      UI.slider(ctl, { label: 'p* verdadeira', min: 0.5, max: 1, step: 0.01, value: 0.8, fmt: (v) => fmt(v, 2), onInput: (v) => { S.p = v; S.seed++; S.t0 = H.t; resample(); } });
      UI.slider(ctl, { label: 'rótulos por par n', min: 1, max: 9, step: 1, value: 1, onInput: (v) => { S.n = v; S.seed++; S.t0 = H.t; resample(); } });
      UI.slider(ctl, { label: 'β = τ', min: 0.1, max: 2, step: 0.05, value: 0.5, fmt: (v) => fmt(v, 2), onInput: (v) => { S.b = v; upd(); } });
      UI.button(ctl, '↻ novos rótulos', () => { S.seed++; S.t0 = H.t; resample(); });
      const r1 = UI.readout(ctl, 'pares unânimes (frequência 0 ou 1)', '—'), r2 = UI.readout(ctl, 'teoria: p*ⁿ + (1−p*)ⁿ', '—'), r3 = UI.readout(ctl, 'alvo DPO em p*', '—'), r4 = UI.readout(ctl, 'alvo IPO em p*', '—');
      nocase(ctl);
      const phat = (L) => L.filter(Boolean).length / L.length;
      function upd() {
        if (!S.labels.length) return;
        const ext = S.labels.filter((L) => { const q = phat(L); return q === 0 || q === 1; }).length;
        r1.set(`${ext}/${N}`, ext >= N / 2 ? 'warn' : null);
        r2.set(fmt(100 * (Math.pow(S.p, S.n) + Math.pow(1 - S.p, S.n)), 0) + '%');
        r3.set(S.p >= 1 ? '∞' : fmt(logit(S.p) / S.b, 2), S.p >= 1 ? 'bad' : null);
        r4.set(fmt((2 * S.p - 1) / (2 * S.b), 2), 'ok');
      }
      resample();
      const H = V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const narrow = w < 640;
        const n = S.n;
        const dtR = Math.min(0.09, 2.4 / (N * n));
        let el = t - S.t0;
        const total = N * n * dtR;
        if (el > total + 3.5) { S.seed++; S.t0 = t; resample(); el = 0; }
        const revealed = Math.floor(el / dtR);
        // ---------- painel dos pares
        const pad = 14;
        const L = narrow ? { x: pad, y: 34, w: w - 2 * pad, h: 196 } : { x: pad, y: 36, w: w * 0.4, h: h - 50 };
        V.text(g, `${N} pares · n = ${n} rótulo${n > 1 ? 's' : ''} cada`, L.x, 20, { size: 12.5, weight: 600, color: C.ink });
        const cols = 4, gap = 7;
        const cwd = (L.w - gap * (cols - 1)) / cols, chh = (L.h - gap * 3) / 4;
        for (let j = 0; j < N; j++) {
          const cx = L.x + (j % cols) * (cwd + gap), cy = L.y + Math.floor(j / cols) * (chh + gap);
          panel(g, cx, cy, cwd, chh, { r: 8 });
          const labs = S.labels[j];
          const perRow = Math.min(n, 5), rowsN = Math.ceil(n / perRow);
          const rr = Math.max(2.6, Math.min(5.5, (cwd - 14) / (perRow * 2.7)));
          let shown = 0;
          for (let k = 0; k < n; k++) {
            const idx = j * n + k;
            if (idx >= revealed) break;
            shown++;
            const rI = Math.floor(k / perRow), cI = k % perRow;
            const nIn = Math.min(perRow, n - rI * perRow);
            const dx = cx + cwd / 2 + (cI - (nIn - 1) / 2) * rr * 2.7;
            const dy = cy + chh * 0.36 + (rI - (rowsN - 1) / 2) * rr * 2.6;
            V.dot(g, dx, dy, rr, labs[k] ? C.a : C.c);
          }
          if (shown === n) {
            const q = phat(labs);
            const ext = q === 0 || q === 1;
            V.text(g, 'p̂ = ' + fmt(q, 2), cx + cwd / 2, cy + chh * 0.8, { size: 11, font: 'mono', align: 'center', base: 'middle', color: ext ? C.d : C.muted, weight: ext ? 700 : 400 });
          }
        }
        // ---------- gráfico
        const P = narrow ? { x: 52, y: L.y + L.h + 40, w: w - 52 - pad, h: h - (L.y + L.h + 40) - 40 } : { x: L.x + L.w + 58, y: 40, w: w - (L.x + L.w + 58) - pad, h: h - 80 };
        const Ymax = 1.5 / S.b;
        const A = axes(g, { x: P.x, y: P.y, w: P.w, h: P.h, xr: [-0.03, 1.03], yr: [-Ymax, Ymax], xt: [0, 0.25, 0.5, 0.75, 1], xl: 'p (frequência da preferida)', yl: 'margem ótima Δh*', ny: 4 });
        V.line(g, A.X(0), A.Y(0), A.X(1), A.Y(0), { color: V.alpha(C.muted, 0.5), lw: 1 });
        plotFn(g, A, (p) => (p <= 0 || p >= 1 ? NaN : logit(p) / S.b), { color: C.b, lw: 2.4 });
        plotFn(g, A, (p) => (2 * p - 1) / (2 * S.b), { color: C.a, lw: 2.4 });
        // limites do IPO
        const lim = 1 / (2 * S.b);
        V.line(g, A.X(0), A.Y(lim), A.X(1), A.Y(lim), { color: V.alpha(C.a, 0.5), lw: 1, dash: [4, 4] });
        V.text(g, '1/(2τ) = ' + fmt(lim, 2), A.X(0.02), A.Y(lim) - 6, { size: 11, color: C.a, font: 'mono' });
        // rótulos: DPO no quadrante de baixo (y = −0,7·Ymax ⇒ p = σ(−1,05)), IPO logo abaixo da reta em p = 0,62
        V.text(g, 'DPO: logit(p)/β', A.X(sig(-1.05)) + 10, A.Y(-0.7 * Ymax), { size: 11.5, color: C.b, align: 'left', weight: 600 });
        V.text(g, 'IPO: (2p−1)/(2τ)', A.X(0.62) + 4, A.Y((2 * 0.62 - 1) / (2 * S.b)) + 18, { size: 11.5, color: C.a, align: 'left', weight: 600 });
        // p*
        V.line(g, A.X(S.p), A.y, A.X(S.p), A.y + A.h, { color: V.alpha(C.d, 0.8), lw: 1.4, dash: [5, 4] });
        // perto de p = 1 o rótulo passa para a esquerda da linha, longe da seta ∞ do DPO
        const pL = S.p > 0.88, pw0 = tw(g, 'p*', 11.5, 'mono', 700);
        V.box(g, A.X(S.p) + (pL ? -6 - pw0 : 2), A.y + 1, pw0 + 4, 15, { r: 3, fill: V.alpha(C.bg, 0.85) });
        V.text(g, 'p*', A.X(S.p) + (pL ? -4 : 4), A.y + 12, { size: 11.5, color: C.d, font: 'mono', weight: 700, align: pL ? 'right' : 'left' });
        // rótulos centrados em x que cruzariam a linha de p* passam para o lado esquerdo dela
        const xpS = A.X(S.p);
        const offP = (xm, wl) => {
          if (Math.abs(xm - xpS) >= wl / 2 + 4) return xm;
          const left = xpS - wl / 2 - 5;
          return left - wl / 2 >= A.x + 2 ? left : Math.min(xpS + wl / 2 + 5, A.x + A.w - wl / 2 - 2);
        };
        // pontos agregados por p̂
        const cnt = new Map();
        for (let j = 0; j < N; j++) { if ((j + 1) * n > revealed) continue; const q = Math.round(phat(S.labels[j]) * 1000) / 1000; cnt.set(q, (cnt.get(q) || 0) + 1); }
        cnt.forEach((c, q) => {
          const rad = 3.5 + Math.sqrt(c) * 1.8;
          // IPO
          V.dot(g, A.X(q), A.Y((2 * q - 1) / (2 * S.b)), rad, V.alpha(C.a, 0.9));
          // DPO
          if (q === 0 || q === 1) {
            const up = q === 1, yy = up ? A.y + 4 : A.y + A.h - 4, xx = A.X(q) + (up ? -10 : 10);
            V.arrow(g, xx, up ? A.y + 40 : A.y + A.h - 40, xx, yy, { color: C.b, lw: 2.4, head: 9 });
            // rótulo logo abaixo (ou acima) da cauda da seta, centrado e preso dentro do gráfico,
            // longe da linha de p* (que só encosta aqui quando p* ≈ 1)
            const il = '∞ ×' + c, iw = tw(g, il, 11.5, 'mono', 700);
            const ix = offP(clamp(xx, A.x + iw / 2 + 2, A.x + A.w - iw / 2 - 2), iw);
            // fundo discreto: em tela estreita o rótulo pode encostar na curva do DPO
            V.box(g, ix - iw / 2 - 3, (up ? A.y + 56 : A.y + A.h - 48) - 12, iw + 6, 16, { r: 4, fill: V.alpha(C.bg, 0.85) });
            V.text(g, il, ix, up ? A.y + 56 : A.y + A.h - 48, { size: 11.5, font: 'mono', weight: 700, color: C.b, align: 'center' });
          } else {
            const yv = logit(q) / S.b;
            if (Math.abs(yv) <= Ymax) V.dot(g, A.X(q), A.Y(yv), rad, V.alpha(C.b, 0.9));
            else V.arrow(g, A.X(q), A.Y(Math.sign(yv) * Ymax * 0.8), A.X(q), A.Y(Math.sign(yv) * Ymax), { color: C.b, lw: 2, head: 7 });
          }
          if (c > 1) {
            // à direita do ponto; se não couber (p̂ = 1, na borda), acima dele, preso dentro do gráfico
            const cl = '×' + c, cwl = tw(g, cl, 10.5, 'mono'), yI = A.Y((2 * q - 1) / (2 * S.b));
            if (A.X(q) + rad + 3 + cwl <= A.x + A.w - 2) V.text(g, cl, A.X(q) + rad + 3, yI + 4, { size: 10.5, font: 'mono', color: C.a });
            else V.text(g, cl, offP(Math.min(A.X(q), A.x + A.w - cwl / 2 - 2), cwl), yI - rad - 5, { size: 10.5, font: 'mono', color: C.a, align: 'center' });
          }
        });
      });
    })();

    /* ================================================== FIG 3 — rampa × parábola */
    (function () {
      const fig = root.querySelector('#fig-rampa');
      adaptH(fig, 440, 500);
      const S = { tau: 0.5, mode: 'soma', xD: 0, xI: 0, t0: 0, trailD: [], trailI: [] };
      const c = () => 1 / (2 * S.tau);
      const range = () => (S.mode === 'soma' ? [-1.5, Math.max(6, 2.2 * c() + 1)] : [-0.3, Math.max(1.2, 2.2 * c() + 0.2)]);
      const pairX = () => (S.mode === 'soma' ? 3 : 0.25);
      const ctl = UI.controls(fig);
      UI.slider(ctl, { label: 'τ (= β)', min: 0.1, max: 2, step: 0.05, value: 0.5, fmt: (v) => fmt(v, 2), onInput: (v) => { S.tau = v; reset(); upd(); } });
      UI.seg(ctl, { label: 'Δh medido por', options: [{ v: 'soma', label: 'soma (paper)' }, { v: 'token', label: 'média por token (TRL)' }], value: 'soma', onChange: (v) => { S.mode = v; reset(); upd(); } });
      UI.button(ctl, '⇥ empurrar além do alvo', () => { const [, x1] = range(); S.xD = S.xI = c() + 0.55 * (x1 - c()); S.t0 = H.t; S.trailD = []; S.trailI = []; });
      UI.button(ctl, '↺ recomeçar', () => reset());
      const rA = UI.readout(ctl, 'alvo 1/(2τ)', '—'), rD = UI.readout(ctl, 'DPO no par: perda · ∂/∂Δh', '—'), rI = UI.readout(ctl, 'IPO no par: perda · ∂/∂Δh', '—');
      nocase(ctl);
      function reset() { S.xD = 0; S.xI = 0; S.t0 = H ? H.t : 0; S.trailD = []; S.trailI = []; }
      function upd() {
        const b = S.tau, x = pairX(), cc = c();
        rA.set(fmt(cc, 2) + (S.mode === 'soma' ? ' nats' : ' nat/token'));
        rD.set(fmt(nls(b * x), 3) + ' · ' + sg(-b * sig(-b * x), 3));
        const e = x - cc;
        rI.set(fmt(e * e, 3) + ' · ' + sg(2 * e, 2), e > 0 ? 'warn' : null);
      }
      let H = null;
      upd();
      H = V.fig(fig, (g, w, h, t, dt) => {
        const C = V.C();
        const b = S.tau, cc = c();
        const [x0, x1] = range();
        const el = t - S.t0;
        if (el > 8.5) reset();
        // dinâmica (velocidades normalizadas)
        if (!LA.paused && dt > 0) {
          const span = x1 - x0;
          const etaD = (span * 0.2) / (b * 0.5);
          for (let k = 0; k < 4; k++) {
            S.xD += (etaD * b * sig(-b * S.xD) * dt) / 4;
            S.xI += (-0.65 * 2 * (S.xI - cc) * dt) / 4;
          }
          if (S.trailD.length > 60) S.trailD.shift();
          if (S.trailI.length > 60) S.trailI.shift();
          S.trailD.push(S.xD); S.trailI.push(S.xI);
        }
        const pad = 14, lm = 50;
        const gapY = 50;
        const ph = (h - 30 - gapY - 34) / 2;
        const Pd = { x: lm, y: 30, w: w - lm - pad, h: ph };
        const Pi = { x: lm, y: 30 + ph + gapY, w: w - lm - pad, h: ph };
        const yD = [0, nls(b * x0) * 1.5];
        const yI = [0, Math.max((x0 - cc) ** 2, (x1 - cc) ** 2) * 1.35];
        const AD = axes(g, Object.assign({}, Pd, { xr: [x0, x1], yr: yD, ny: 3, noX: false }));
        const AI = axes(g, Object.assign({}, Pi, { xr: [x0, x1], yr: yI, ny: 3, xl: S.mode === 'soma' ? 'Δh (nats, soma)' : 'Δh (nat/token)' }));
        V.text(g, 'DPO · −log σ(β·Δh): uma rampa que nunca chega a zero', Pd.x + 6, Pd.y + 14, { size: 12, weight: 600, color: C.b, maxW: Pd.w - 12 });
        V.text(g, 'IPO · (Δh − 1/(2τ))²: uma parábola com fundo no alvo', Pi.x + 6, Pi.y + 14, { size: 12, weight: 600, color: C.a, maxW: Pi.w - 12 });
        plotFn(g, AD, (x) => nls(b * x), { color: C.b, lw: 2.6 });
        plotFn(g, AI, (x) => (x - cc) ** 2, { color: C.a, lw: 2.6 });
        // alvo do IPO
        // as linhas verticais começam abaixo do título do painel; cada rótulo vai para o lado
        // em que não cruza a outra linha vertical (alvo × nosso par)
        const xc = AI.X(cc), xpI = AI.X(pairX()), dP = xpI - xc;
        const wA = tw(g, 'alvo 1/(2τ)', 11, 'mono'), wP = tw(g, 'nosso par', 11);
        V.line(g, xc, AI.y + 20, xc, AI.y + AI.h, { color: V.alpha(C.a, 0.7), lw: 1.2, dash: [5, 4] });
        // padrão: embaixo, à direita do alvo; se a linha do par cair logo à direita, o rótulo sobe
        // para o canto de cima à esquerda do alvo (região vazia, longe da bola e da curva)
        const aFlip = dP > 3 && dP < wA + 9 && xc - 5 - wA > AI.x + 4;
        V.text(g, 'alvo 1/(2τ)', aFlip ? xc - 5 : xc + 5, aFlip ? AI.y + 32 : AI.y + AI.h - 8, { size: 11, color: C.a, font: 'mono', align: aFlip ? 'right' : 'left' });
        // nosso par
        [AD, AI].forEach((A) => {
          const px = A.X(pairX());
          V.line(g, px, A.y + 20, px, A.y + A.h, { color: V.alpha(C.d, 0.75), lw: 1.2, dash: [2, 3] });
          const pFlip = A === AI && -dP > 3 && -dP < wP + 9 && px - 5 - wP > A.x + 4;
          V.text(g, 'nosso par', pFlip ? px - 5 : px + 5, A.y + 32, { size: 11, color: C.d, align: pFlip ? 'right' : 'left' });
        });
        // bolas
        const ball = (A, x, f, df, col, trail) => {
          const off = x > x1;
          const xv = Math.min(x, x1);
          trail.forEach((xx, i) => { if (xx <= x1) V.dot(g, A.X(xx), A.Y(f(xx)), 2, V.alpha(col, (0.35 * i) / trail.length)); });
          const px = A.X(xv), py = A.Y(f(xv));
          V.glow(g, px, py, 16, col, 0.5);
          V.dot(g, px, py, 7, col);
          const push = -df(xv);
          const L = clamp(Math.abs(push) / (Math.abs(df(x0)) + 1e-9), 0.15, 1) * 46;
          if (Math.abs(push) > 1e-4) V.arrow(g, px, py - 16, px + Math.sign(push) * L, py - 16, { color: col, lw: 2.2, head: 7 });
          if (off) V.text(g, '→ ∞ (segue andando)', A.X(x1) - 4, py - 28, { size: 11, align: 'right', color: col, font: 'mono' });
        };
        ball(AD, S.xD, (x) => nls(b * x), (x) => -b * sig(-b * x), C.b, S.trailD);
        ball(AI, S.xI, (x) => (x - cc) ** 2, (x) => 2 * (x - cc), C.a, S.trailI);
        if (S.xI > cc + 0.02 * (x1 - x0)) V.text(g, '← puxa de volta', AI.X(Math.min(S.xI, x1)) - 8, AI.Y((Math.min(S.xI, x1) - cc) ** 2) - 30, { size: 11.5, align: 'right', color: C.a, weight: 600 });
      });
    })();

    /* ================================================== FIG 4 — bandit de 3 ações */
    (function () {
      const fig = root.querySelector('#fig-bandit');
      adaptH(fig, 430, 620);
      const PAIRS = [[0, 1], [1, 2], [0, 2]];
      const S = { tau: 1, t0: -4 };
      const mk = () => ({ th: [0, 0, 0], m: [0, 0, 0], v: [0, 0, 0], t: 0, tr: [] });
      let sims = { dpo: mk(), ipo: mk() };
      function step(sim, kind) {
        const tau = S.tau, gr = [0, 0, 0];
        for (const [a, b] of PAIRS) { const d = sim.th[a] - sim.th[b]; const gd = kind === 'dpo' ? -tau * sig(-tau * d) : 2 * (d - 1 / (2 * tau)); gr[a] += gd / 3; gr[b] -= gd / 3; }
        sim.t++;
        for (let i = 0; i < 3; i++) {
          sim.m[i] = 0.9 * sim.m[i] + 0.1 * gr[i]; sim.v[i] = 0.999 * sim.v[i] + 0.001 * gr[i] * gr[i];
          const mh = sim.m[i] / (1 - Math.pow(0.9, sim.t)), vh = sim.v[i] / (1 - Math.pow(0.999, sim.t));
          sim.th[i] -= (0.01 * mh) / (Math.sqrt(vh) + 1e-8);
        }
      }
      const pi = (th) => { const m = Math.max(...th); const e = th.map((x) => Math.exp(x - m)); const s = e[0] + e[1] + e[2]; return e.map((x) => x / s); };
      function restart() { sims = { dpo: mk(), ipo: mk() }; S.t0 = H ? H.t : 0; }
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'τ (= β do DPO)', options: [{ v: 0.1, label: '0,1' }, { v: 0.5, label: '0,5' }, { v: 1, label: '1' }], value: 1, onChange: (v) => { S.tau = v; restart(); } });
      UI.button(ctl, '▶ treinar de novo', () => restart(), { primary: true });
      const rS = UI.readout(ctl, 'passo', '—'), rD = UI.readout(ctl, 'DPO π(a, b, c)', '—'), rI = UI.readout(ctl, 'IPO π(a, b, c)', '—'), rF = UI.readout(ctl, 'IPO fechado: Δθ = 1/(3τ)', '—');
      nocase(ctl);
      const LOGMAX = Math.log10(18000);
      let H = null;
      H = V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const AC = [C.d, C.e, C.c];
        const el = t - S.t0;
        const target = Math.min(18000, Math.floor(Math.pow(10, (LOGMAX * Math.min(1, el / 8.5)))));
        if (!LA.paused) {
          ['dpo', 'ipo'].forEach((k) => {
            const sim = sims[k];
            while (sim.t < target) {
              step(sim, k);
              const lg = Math.log10(sim.t);
              if (!sim.tr.length || lg - sim.tr[sim.tr.length - 1][0] > 0.03 || sim.t === 18000) sim.tr.push([lg, pi(sim.th)]);
            }
          });
        }
        if (el > 13) restart();
        const pD = pi(sims.dpo.th), pI = pi(sims.ipo.th);
        rS.set(sims.dpo.t.toLocaleString('pt-BR') + ' / 18.000');
        rD.set(pD.map((x) => fmt(x, 3)).join('; '));
        rI.set(pI.map((x) => fmt(x, 3)).join('; '));
        const d = 1 / (3 * S.tau); const pf = pi([2 * d, d, 0]);
        rF.set(pf.map((x) => fmt(x, 4)).join('; '));
        const narrow = w < 640, pad = 14;
        // preferências no topo
        const chips = ['a ≻ b', 'b ≻ c', 'a ≻ c'];
        const cwid = 62, tot = chips.length * cwid + 2 * 8;
        V.text(g, 'dados D₁:', w / 2 - tot / 2 - 8, 19, { size: 11.5, color: C.muted, align: 'right', base: 'middle' });
        chips.forEach((cp, i) => chip(g, w / 2 - tot / 2 + i * (cwid + 8), 8, cwid, 22, cp, { size: 12 }));
        const blocks = narrow
          ? [{ k: 'dpo', x: pad, y: 44, w: w - 2 * pad, h: (h - 56) / 2 - 6 }, { k: 'ipo', x: pad, y: 44 + (h - 56) / 2 + 6, w: w - 2 * pad, h: (h - 56) / 2 - 6 }]
          : [{ k: 'dpo', x: pad, y: 44, w: (w - 3 * pad) / 2, h: h - 56 }, { k: 'ipo', x: 2 * pad + (w - 3 * pad) / 2, y: 44, w: (w - 3 * pad) / 2, h: h - 56 }];
        blocks.forEach((B) => {
          const sim = sims[B.k], p = pi(sim.th);
          const col = B.k === 'dpo' ? C.b : C.a;
          panel(g, B.x, B.y, B.w, B.h, { stroke: V.alpha(col, 0.45) });
          V.text(g, B.k === 'dpo' ? 'DPO · −log σ(τ·Δθ)' : 'IPO · (Δθ − 1/(2τ))²', B.x + 12, B.y + 20, { size: 12.5, weight: 700, color: col });
          // barras
          const bars = narrow ? { x: B.x + 14, y: B.y + 40, w: B.w * 0.42 - 14, h: B.h - 70 } : { x: B.x + 24, y: B.y + 40, w: B.w - 48, h: B.h * 0.48 - 40 };
          const bw = (bars.w - 2 * 12) / 3;
          const Yb = (v) => bars.y + bars.h - v * bars.h;
          ['a', 'b', 'c'].forEach((nm, i) => {
            const x = bars.x + i * (bw + 12);
            g.fillStyle = V.alpha(AC[i], 0.75); V.rr(g, x, Yb(p[i]), bw, Math.max(1.5, p[i] * bars.h), 5); g.fill();
            V.text(g, fmt(p[i], 3), x + bw / 2, Math.max(bars.y + 10, Yb(p[i]) - 5), { size: 11, font: 'mono', align: 'center', color: C.ink });
            V.text(g, nm, x + bw / 2, bars.y + bars.h + 14, { size: 12, font: 'mono', weight: 700, align: 'center', color: AC[i] });
            if (B.k === 'ipo') { const yv = Yb(pf[i]); V.line(g, x - 3, yv, x + bw + 3, yv, { color: C.ink, lw: 1.4, dash: [3, 2] }); }
          });
          V.line(g, bars.x - 4, Yb(1 / 3), bars.x + bars.w + 4, Yb(1 / 3), { color: V.alpha(C.muted, 0.8), lw: 1, dash: [5, 4] });
          V.text(g, 'ref 1/3', bars.x + bars.w + 4, Yb(1 / 3) - 4, { size: 10.5, color: C.muted, align: 'right' });
          // trajetória
          const T = narrow ? { x: B.x + B.w * 0.42 + 34, y: B.y + 40, w: B.w * 0.58 - 46, h: B.h - 70 } : { x: B.x + 46, y: B.y + B.h * 0.48 + 26, w: B.w - 62, h: B.h * 0.52 - 64 };
          const A = axes(g, { x: T.x, y: T.y, w: T.w, h: T.h, xr: [0, LOGMAX], yr: [0, 1], xt: [0, 1, 2, 3, 4], fx: (v) => (v === 0 ? '1' : '10' + ['', '', '²', '³', '⁴'][v]), yt: [0, 0.5, 1], fy: (v) => fmt(v, 1), xl: 'passo (log)' });
          for (let i = 0; i < 3; i++) {
            g.beginPath();
            sim.tr.forEach(([lg, pp], k) => { const px = A.X(lg), py = A.Y(pp[i]); if (!k) g.moveTo(px, py); else g.lineTo(px, py); });
            g.strokeStyle = AC[i]; g.lineWidth = 2; g.stroke();
          }
        });
      });
    })();

    /* ================================================== FIG 5 — função de valor / balança */
    (function () {
      const fig = root.querySelector('#fig-kt');
      adaptH(fig, 440, 660);
      const S = { mode: 'valor', al: 0.88, la: 2.25, b: 0.1, z0: 0.5, f: 0.09, lD: 1, lU: 1, tilt: 0 };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'modo', options: [{ v: 'valor', label: 'função de valor' }, { v: 'balanca', label: 'balança λ_D·n_D × λ_U·n_U' }], value: 'valor', onChange: (v) => { S.mode = v; gV.show(v === 'valor'); gB.show(v === 'balanca'); upd(); } });
      const gV = UI.group(ctl), gB = UI.group(ctl);
      UI.slider(gV.el, { label: 'α (KT)', min: 0.3, max: 1, step: 0.01, value: 0.88, fmt: (v) => fmt(v, 2), onInput: (v) => { S.al = v; } });
      UI.slider(gV.el, { label: 'λ (KT)', min: 1, max: 3, step: 0.05, value: 2.25, fmt: (v) => fmt(v, 2), onInput: (v) => { S.la = v; } });
      UI.slider(gV.el, { label: 'β (KTO)', min: 0.02, max: 1, step: 0.01, value: 0.1, fmt: (v) => fmt(v, 2), onInput: (v) => { S.b = v; } });
      UI.slider(gV.el, { label: 'z₀', min: -3, max: 3, step: 0.1, value: 0.5, fmt: (v) => fmt(v, 1), onInput: (v) => { S.z0 = v; } });
      UI.slider(gB.el, { label: 'desejáveis', min: 0.05, max: 0.95, step: 0.01, value: 0.09, fmt: (v) => fmt(100 * v, 0) + '%', onInput: (v) => { S.f = v; upd(); } });
      UI.slider(gB.el, { label: 'λ_D', min: 0.5, max: 20, step: 0.5, value: 1, fmt: (v) => fmt(v, 1), onInput: (v) => { S.lD = v; upd(); } });
      UI.slider(gB.el, { label: 'λ_U', min: 0.5, max: 5, step: 0.25, value: 1, fmt: (v) => fmt(v, 2), onInput: (v) => { S.lU = v; upd(); } });
      const rR = UI.readout(gB.el, 'λ_D·n_D / λ_U·n_U', '—'), rS = UI.readout(gB.el, 'λ_D sugerido (se λ_U = 1)', '—');
      gB.show(false);
      nocase(ctl);
      const nD = () => Math.round(1000 * S.f), nU = () => 1000 - nD();
      const ratio = () => (S.lD * nD()) / (S.lU * nU());
      function upd() {
        const r = ratio();
        rR.set(fmt(r, 2), r >= 1 && r <= 1.5 ? 'ok' : 'warn');
        const q = nU() / nD();
        rS.set(fmt(q, 1) + ' a ' + fmt(1.5 * q, 1));
      }
      upd();
      const KT = (z) => (z >= 0 ? Math.pow(z, S.al) : -S.la * Math.pow(-z, S.al));
      V.fig(fig, (g, w, h, t, dt) => {
        const C = V.C();
        const narrow = w < 640, pad = 14;
        if (S.mode === 'valor') {
          const P1 = narrow ? { x: 46, y: 34, w: w - 46 - pad, h: h * 0.5 - 80 } : { x: 46, y: 46, w: w * 0.44 - 46, h: h - 94 };
          const P2 = narrow ? { x: 46, y: h * 0.5 + 18, w: w - 46 - pad, h: h * 0.5 - 70 } : { x: w * 0.53, y: 46, w: w * 0.47 - pad, h: h - 94 };
          // ---- Kahneman–Tversky
          const ymax = Math.pow(10, S.al) * 1.15, ymin = -S.la * Math.pow(10, S.al) * 1.06;
          const A = axes(g, Object.assign({}, P1, { xr: [-10, 10], yr: [ymin, ymax], xt: [-10, -5, 0, 5, 10], ny: 4, xl: 'resultado z (z₀ = 0)', yl: 'valor subjetivo v(z)' }));
          V.text(g, 'Kahneman–Tversky (1992)', P1.x + P1.w, P1.y - 8, { size: 12, weight: 700, color: C.ink, align: 'right' });
          V.line(g, A.X(0), A.y, A.X(0), A.y + A.h, { color: V.alpha(C.muted, 0.6), lw: 1 });
          V.line(g, A.x, A.Y(0), A.x + A.w, A.Y(0), { color: V.alpha(C.muted, 0.6), lw: 1 });
          plotFn(g, A, (z) => (z >= 0 ? KT(z) : NaN), { color: C.a, lw: 2.6 });
          plotFn(g, A, (z) => (z <= 0 ? KT(z) : NaN), { color: C.red, lw: 2.6 });
          const zz = 1 + 8 * V.pingpong(t, 9);
          const vp = KT(zz), vn = KT(-zz);
          V.line(g, A.X(zz), A.Y(0), A.X(zz), A.Y(vp), { color: C.a, lw: 5 });
          V.line(g, A.X(-zz), A.Y(0), A.X(-zz), A.Y(vn), { color: C.red, lw: 5 });
          V.dot(g, A.X(zz), A.Y(vp), 5, C.a); V.dot(g, A.X(-zz), A.Y(vn), 5, C.red);
          // acima e à esquerda do ponto: a curva dos ganhos sobe para a direita, então esse canto fica livre
          V.text(g, 'ganho ' + sg(vp, 2), A.X(zz) - 8, A.Y(vp) - 8, { size: 11, font: 'mono', color: C.a, align: 'right' });
          V.text(g, 'perda ' + fmt(vn, 2), A.X(-zz) + 6, A.Y(vn) + 4, { size: 11, font: 'mono', color: C.red });
          V.text(g, `|v(−z)| / v(z) = λ = ${fmt(S.la, 2)}`, A.x + 8, A.y + A.h - 10, { size: 11.5, color: C.ink, font: 'mono' });
          // ---- KTO
          const b = S.b, z0 = S.z0, span = 6 / b;
          const lmx = Math.max(S.lD, S.lU, 1);
          const B = axes(g, Object.assign({}, P2, { xr: [z0 - span, z0 + span], yr: [0, lmx * 1.08], ny: 4, nx: 4, xl: 'recompensa r_θ = log π_θ/π_ref', yl: 'valor do KTO' }));
          V.text(g, 'KTO (β = ' + fmt(b, 2) + ')', P2.x + P2.w, P2.y - 8, { size: 12, weight: 700, color: C.ink, align: 'right' });
          // janela do gradiente
          const win = (r) => b * sig(b * (r - z0)) * (1 - sig(b * (r - z0)));
          const wscale = (lmx * 0.42) / (b / 4);
          g.beginPath(); g.moveTo(B.X(z0 - span), B.Y(0));
          for (let i = 0; i <= 120; i++) { const r = z0 - span + (2 * span * i) / 120; g.lineTo(B.X(r), B.Y(win(r) * wscale)); }
          g.lineTo(B.X(z0 + span), B.Y(0)); g.closePath(); g.fillStyle = V.alpha(C.d, 0.22); g.fill();
          V.text(g, 'janela β·σ(1−σ)', B.X(z0) + 6, B.Y(lmx * 0.42) - 4, { size: 11, color: C.d, font: 'mono' });
          plotFn(g, B, (r) => S.lD * sig(b * (r - z0)), { color: C.a, lw: 2.6 });
          plotFn(g, B, (r) => S.lU * sig(b * (z0 - r)), { color: C.red, lw: 2.6 });
          V.line(g, B.X(z0), B.y, B.X(z0), B.y + B.h, { color: C.d, lw: 1.4, dash: [5, 4] });
          V.text(g, 'z₀ = ' + fmt(z0, 1), B.X(z0) + 5, B.y + 12, { size: 11.5, color: C.d, font: 'mono', weight: 700 });
          V.text(g, 'desejável', B.X(z0 + span * 0.92), B.Y(S.lD * sig(b * span * 0.92)) + 16, { size: 11.5, color: C.a, align: 'right', weight: 600 });
          V.text(g, 'indesejável', B.X(z0 - span * 0.92), B.Y(S.lU * sig(b * span * 0.92)) + 16, { size: 11.5, color: C.red, align: 'left', weight: 600 });
          // pontos viajando + seta proporcional ao gradiente
          const rd = z0 + span * 0.88 * Math.sin(t * 0.55), ru = z0 + span * 0.88 * Math.sin(t * 0.55 + 2.2);
          const vd = S.lD * sig(b * (rd - z0)), vu = S.lU * sig(b * (z0 - ru));
          const gd = S.lD * win(rd) / (b / 4), gu = S.lU * win(ru) / (b / 4);
          V.dot(g, B.X(rd), B.Y(vd), 6, C.a); V.dot(g, B.X(ru), B.Y(vu), 6, C.red);
          if (gd > 0.03) V.arrow(g, B.X(rd), B.Y(vd) + 4, B.X(rd) + 30 * gd, B.Y(vd) + 4, { color: C.a, lw: 2, head: 6 });
          if (gu > 0.03) V.arrow(g, B.X(ru), B.Y(vu) + 4, B.X(ru) - 30 * gu, B.Y(vu) + 4, { color: C.red, lw: 2, head: 6 });
          // a nota fica acima do gráfico (a legenda da figura explica as setas em telas estreitas)
          if (!narrow) V.text(g, 'setas: força do gradiente (some nas pontas)', B.x + B.w / 2, B.y - 24, { size: 11, color: C.muted, align: 'center', maxW: B.w - 12 });
        } else {
          // ---- balança
          const r = ratio();
          const target = clamp(Math.atan(Math.log(r) * 0.4), -0.16, 0.16);
          S.tilt = lerp(S.tilt, target, Math.min(1, dt * 4));
          // em tela estreita a figura é alta (o modo “função de valor” precisa); centraliza a balança na vertical
          const cx = w / 2, cy = narrow ? h / 2 - 70 : h * 0.3, L = narrow ? w * 0.27 : Math.min(w * 0.3, 250);
          const ty = narrow ? cy - 150 : 22;
          const ex = Math.cos(S.tilt) * L, ey = Math.sin(S.tilt) * L;
          const lx = cx - ex, ly = cy + ey, rx = cx + ex, ry = cy - ey;
          // fulcro
          g.beginPath(); g.moveTo(cx, cy); g.lineTo(cx - 26, cy + 70); g.lineTo(cx + 26, cy + 70); g.closePath(); g.fillStyle = V.alpha(C.muted, 0.3); g.fill();
          V.line(g, lx, ly, rx, ry, { color: C.ink, lw: 4 });
          V.dot(g, cx, cy, 6, C.d);
          // cada bloco = `unit` unidades de PESO (λ·n), a mesma para os dois pratos, para a pilha
          // bater com a inclinação; a unidade cresce em degraus de 50 para a pilha caber em 4 fileiras
          const PER = 6, MAXB = 24;
          const unit = 50 * Math.max(1, Math.ceil(Math.max(S.lD * nD(), S.lU * nU()) / (MAXB * 50)));
          const pw = Math.min(150, w * 0.28), bs = Math.min(18, (pw - 10) / PER - 3);
          const pan = (x, y, n, lam, col, label, ok) => {
            // gancho + barra + cordões verticais: o prato pende abaixo da pilha mais alta possível,
            // então nem o travessão nem os cordões cruzam os blocos
            const hy = y + 12, py = hy + (MAXB / PER) * (bs + 3) + 4;
            V.line(g, x, y, x, hy, { color: C.muted, lw: 1.2 });
            V.line(g, x - pw / 2, hy, x + pw / 2, hy, { color: C.muted, lw: 1.2 });
            V.line(g, x - pw / 2, hy, x - pw / 2, py, { color: C.muted, lw: 1.2 }); V.line(g, x + pw / 2, hy, x + pw / 2, py, { color: C.muted, lw: 1.2 });
            V.box(g, x - pw / 2, py, pw, 8, { r: 4, fill: V.alpha(col, 0.5) });
            const blocks = clamp(Math.round((lam * n) / unit), 1, MAXB);
            for (let i = 0; i < blocks; i++) {
              const bx = x - (PER * (bs + 3) - 3) / 2 + (i % PER) * (bs + 3), by = py - (Math.floor(i / PER) + 1) * (bs + 3);
              V.box(g, bx, by, bs, bs, { r: 3, fill: V.alpha(col, 0.75) });
            }
            mark(g, x - pw / 2 - 14, py - 8, ok, 6, col);
            V.text(g, label, x, py + 26, { size: 11.5, align: 'center', color: col, weight: 700, maxW: Math.min(pw + 40, 2 * Math.min(x - 4, w - 4 - x)) });
            V.text(g, `${n} × λ = ${fmt(lam * n, lam * n >= 100 ? 0 : 1)}`, x, py + 42, { size: 11.5, align: 'center', font: 'mono', color: C.ink });
          };
          pan(lx, ly, nD(), S.lD, C.a, 'desejáveis (λ_D = ' + fmt(S.lD, 1) + ')', true);
          pan(rx, ry, nU(), S.lU, C.red, 'indesejáveis (λ_U = ' + fmt(S.lU, 2) + ')', false);
          V.text(g, `cada bloco = ${unit} de peso λ·n`, w / 2, ty + 18, { size: 11, align: 'center', color: C.muted, font: 'mono' });
          // medidor log
          const gy = narrow ? Math.min(h - 56, cy + 210) : h - 56, gx0 = pad + 40, gx1 = w - pad - 40;
          const GX = (v) => gx0 + ((Math.log10(v) + 1) / 2) * (gx1 - gx0);
          V.line(g, gx0, gy, gx1, gy, { color: C.muted, lw: 1.5 });
          g.fillStyle = V.alpha(C.a, 0.3); g.fillRect(GX(1), gy - 9, GX(1.5) - GX(1), 18);
          g.strokeStyle = C.e; g.setLineDash([3, 3]); g.strokeRect(GX(1), gy - 12, GX(4 / 3) - GX(1), 24); g.setLineDash([]);
          [0.1, 0.3, 1, 3, 10].forEach((v) => { V.line(g, GX(v), gy - 4, GX(v), gy + 4, { color: C.muted, lw: 1 }); V.text(g, fmt(v, v < 1 ? 1 : 0), GX(v), gy + 18, { size: 11, align: 'center', font: 'mono', color: C.muted }); });
          const rv = clamp(r, 0.1, 10);
          V.arrow(g, GX(rv), gy - 30, GX(rv), gy - 6, { color: r >= 1 && r <= 1.5 ? C.a : C.d, lw: 2.4, head: 8 });
          V.text(g, 'λ_D·n_D / λ_U·n_U = ' + fmt(r, 2), GX(rv), gy - 36, { size: 12, align: clamp((GX(rv) - gx0) / (gx1 - gx0), 0, 1) > 0.7 ? 'right' : clamp((GX(rv) - gx0) / (gx1 - gx0), 0, 1) < 0.3 ? 'left' : 'center', font: 'mono', weight: 700, color: C.ink });
          V.text(g, 'faixa do paper [1; 1,5] (cheia) · TRL [1; 4/3] (tracejada)', w / 2, gy + 44, { size: 11, align: 'center', color: C.muted, maxW: w - 2 * pad });
          V.text(g, `${nD()} desejáveis : ${nU()} indesejáveis`, w / 2, ty, { size: 12.5, align: 'center', color: C.ink, weight: 600 });
        }
      });
    })();

    /* ================================================== FIG 6 — z₀ no microbatch */
    (function () {
      const fig = root.querySelector('#fig-z0');
      adaptH(fig, 440, 600);
      const S = { m: 4, seed: 5, sft: false, M: [], lab: [] };
      const BETA = 0.1;
      function gen() {
        const r = V.rng(S.seed * 7 + S.m);
        const m = S.m;
        const lab = Array.from({ length: m }, () => r() < 0.5);
        if (lab.every((x) => x === lab[0])) lab[m - 1] = !lab[0];
        const M = Array.from({ length: m }, (_, i) => Array.from({ length: m }, (_, j) => {
          let v;
          if (i === j) v = lab[i] ? 0.6 + 2.4 * r() : -(0.6 + 2.4 * r());
          else v = S.sft ? -0.12 + 0.12 * r.normal() : 0.45 + 0.3 * r.normal();
          return Math.round(v * 10) / 10;
        }));
        S.M = M; S.lab = lab;
      }
      gen();
      const comp = () => {
        const m = S.m;
        const mis = Array.from({ length: m }, (_, i) => S.M[i][(i + 1) % m]);
        const avg = mis.reduce((a, b) => a + b, 0) / m;
        const z0 = Math.max(0, avg);
        const rows = Array.from({ length: m }, (_, i) => { const rr = S.M[i][i]; const v = S.lab[i] ? sig(BETA * (rr - z0)) : sig(BETA * (z0 - rr)); return { r: rr, v, loss: 1 - v }; });
        const mean = rows.reduce((a, b) => a + b.loss, 0) / m;
        return { mis, avg, z0, rows, mean };
      };
      const ctl = UI.controls(fig);
      UI.slider(ctl, { label: 'microbatch m', min: 2, max: 6, step: 1, value: 4, onInput: (v) => { S.m = v; gen(); upd(); } });
      UI.button(ctl, '↻ novo lote', () => { S.seed++; gen(); upd(); });
      UI.toggle(ctl, { label: 'SFT prévio nos mesmos dados (KL ≈ 0)', value: false, onChange: (v) => { S.sft = v; gen(); upd(); } });
      const rA = UI.readout(ctl, 'média desencontrada', '—'), rZ = UI.readout(ctl, 'ẑ₀', '—'), rL = UI.readout(ctl, 'perda média', '—');
      nocase(ctl);
      function upd() { const c = comp(); rA.set(sg(c.avg, 3)); rZ.set(fmt(c.z0, 3), c.avg < 0 ? 'warn' : null); rL.set(fmt(c.mean, 4)); }
      upd();
      const st = UI.stepper(fig, [
        { t: 'O microbatch', d: 'Cada linha é um prompt \\(x_i\\), cada coluna uma resposta \\(y_j\\). Os exemplos do lote são os pares \\((x_i, y_i)\\), cada um rotulado como desejável (✓) ou indesejável (✗). Cada célula é \\(\\log\\pi_\\theta(y_j\\mid x_i)-\\log\\pi_{\\text{ref}}(y_j\\mid x_i)\\).' },
        { t: 'Recompensas casadas', d: 'A diagonal dá a recompensa \\(r_\\theta(x_i,y_i)\\) de cada exemplo. Desejáveis tendem a ter \\(r&gt;0\\) e indesejáveis \\(r&lt;0\\), justamente porque foram escolhidas por serem boas ou ruins. Por isso não servem para estimar o KL típico.' },
        { t: 'Pares desencontrados', d: 'A diagonal deslocada, \\((x_i, y_{i+1})\\) com o último voltando ao primeiro, junta cada prompt com a resposta de outro exemplo. A média dessas log-razões aproxima o KL da política sem gerar nada: custa só um forward extra.' },
        { t: 'Clamp e z₀', d: 'Um KL não pode ser negativo, então \\(\\hat z_0=\\max(0,\\text{média})\\). O clamp enviesa para cima e reduz a variância. E \\(\\hat z_0\\) é <b>destacado</b> do grafo: não recebe gradiente, só posiciona a logística.' },
        { t: 'Perda por exemplo', d: 'Desejável: \\(v=\\sigma(\\beta(r-z_0))\\). Indesejável: \\(v=\\sigma(\\beta(z_0-r))\\). A perda é \\(\\lambda_y - v\\) (aqui \\(\\lambda=1\\), \\(\\beta=0{,}1\\)). Com SFT prévio nos mesmos dados, a média tende a ≤ 0 e o clamp dá \\(\\hat z_0=0\\).' },
      ], { autoplay: 5200 });
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const narrow = w < 640, pad = 14;
        const m = S.m, c = comp(), k = st.i, e = Math.min(1, st.elapsed() / 0.8);
        const cs = narrow ? Math.min(52, (w - 90) / m, 230 / m) : Math.min(60, (h - 110) / m, (w * 0.46 - 80) / m);
        const mx = narrow ? (w - m * cs) / 2 + 16 : 70, my = 62;
        // cabeçalhos
        for (let j = 0; j < m; j++) {
          const x = mx + j * cs + cs / 2;
          V.text(g, 'y' + '₁₂₃₄₅₆'[j], x - 6, my - 26, { size: 12.5, font: 'mono', align: 'center', color: C.ink, weight: 600 });
          mark(g, x + 10, my - 30, S.lab[j], 5);
        }
        for (let i = 0; i < m; i++) V.text(g, 'x' + '₁₂₃₄₅₆'[i], mx - 12, my + i * cs + cs / 2, { size: 12.5, font: 'mono', align: 'right', base: 'middle', color: C.ink, weight: 600 });
        V.text(g, 'log π_θ/π_ref', mx, my - 46, { size: 11, color: C.muted, font: 'mono' });
        for (let i = 0; i < m; i++) for (let j = 0; j < m; j++) {
          const x = mx + j * cs, y = my + i * cs, v = S.M[i][j];
          const diag = i === j, shift = j === (i + 1) % m;
          let a = 0.12;
          if (diag && k >= 1) a = 0.2 + 0.45 * (k === 1 ? e : 1);
          if (shift && k >= 2) a = 0.2 + 0.45 * (k === 2 ? e : 1);
          if (k === 0) a = diag ? 0.28 : 0.12;
          const col = diag ? (S.lab[i] ? C.a : C.red) : shift && k >= 2 ? C.d : C.muted;
          V.box(g, x + 2, y + 2, cs - 4, cs - 4, { r: 6, fill: V.alpha(col, a), stroke: (diag && k === 1) || (shift && k === 2) ? col : V.alpha(C.muted, 0.25), lw: (diag && k === 1) || (shift && k === 2) ? 2 : 1 });
          V.text(g, sg(v, 1), x + cs / 2, y + cs / 2 + 0.5, { size: Math.min(13, cs * 0.28), font: 'mono', align: 'center', base: 'middle', color: C.ink, alpha: (!diag && !shift) || (shift && k < 2) ? 0.55 : 1 });
        }
        // setas dos desencontrados até o somador
        const P = narrow ? { x: pad, y: my + m * cs + 22, w: w - 2 * pad, h: h - (my + m * cs + 22) - 10 } : { x: mx + m * cs + 34, y: 30, w: w - (mx + m * cs + 34) - pad, h: h - 40 };
        panel(g, P.x, P.y, P.w, P.h);
        const tx = P.x + 12;
        let ty = P.y + 22;
        const mis = c.mis.map((v) => sg(v, 1)).join(' ');
        if (k >= 2) {
          V.text(g, 'média dos desencontrados', tx, ty, { size: 12, weight: 600, color: C.d });
          ty += 18;
          V.wrap(g, `(${mis}) / ${m} = ${sg(c.avg, 3)}`, tx, ty, P.w - 24, { size: 12, font: 'mono', color: C.ink, lineH: 16, maxLines: 2 });
          ty += 24;
          if (k === 2) {
            for (let i = 0; i < m; i++) {
              const j = (i + 1) % m;
              const [sx, sy] = [mx + j * cs + cs / 2, my + i * cs + cs / 2];
              const ph = (t * 0.7 + i / m) % 1;
              const px = lerp(sx, P.x + 2, V.ease.inOut(ph)), py = lerp(sy, P.y + 34, V.ease.inOut(ph));
              V.dot(g, px, py, 4, V.alpha(C.d, 1 - ph * 0.6));
            }
          }
        } else {
          V.wrap(g, k === 0 ? 'Um microbatch de ' + m + ' exemplos (x_i, y_i), cada um com seu rótulo ✓/✗. O KTO não precisa de pares.' : 'Recompensas casadas r_i: grandes em módulo, porque os exemplos foram escolhidos por serem bons ou ruins.', tx, ty, P.w - 24, { size: 12, color: C.ink, lineH: 17, maxLines: 3 });
          ty += 46;
        }
        if (k >= 3) {
          const bx = tx, bw = Math.min(P.w - 24, 300);
          V.box(g, bx, ty, bw, 34, { r: 8, fill: V.alpha(C.d, 0.15), stroke: C.d, lw: 1.4 });
          V.text(g, `ẑ₀ = max(0; ${sg(c.avg, 3)}) = ${fmt(c.z0, 3)}`, bx + 10, ty + 17.5, { size: 12.5, font: 'mono', base: 'middle', color: C.ink, weight: 700, maxW: bw - 60 });
          // cadeado (sem gradiente)
          const lx = bx + bw - 26, ly = ty + 18;
          g.strokeStyle = C.d; g.lineWidth = 1.6; g.beginPath(); g.arc(lx, ly - 4, 4.5, Math.PI, 0); g.stroke();
          V.box(g, lx - 7, ly - 4, 14, 10, { r: 2, fill: C.d });
          ty += 46;
        }
        // linhas por exemplo
        const rowH = Math.min(32, (P.y + P.h - ty - 30) / m);
        if (k >= 1 && rowH > 14) {
          for (let i = 0; i < m; i++) {
            const y = ty + i * rowH;
            const R = c.rows[i];
            mark(g, tx + 6, y + rowH / 2 - 1, S.lab[i], 5);
            V.text(g, `r${'₁₂₃₄₅₆'[i]} = ${sg(R.r, 1)}`, tx + 18, y + rowH / 2, { size: 11.5, font: 'mono', base: 'middle', color: C.ink });
            if (k >= 4) {
              const bx0 = tx + 104, bw = Math.max(40, P.w - 24 - 104 - 120);
              V.text(g, 'v=' + fmt(R.v, 3), bx0, y + rowH / 2, { size: 11.5, font: 'mono', base: 'middle', color: C.muted });
              const lx0 = bx0 + 62;
              g.fillStyle = V.alpha(S.lab[i] ? C.a : C.red, 0.6); V.rr(g, lx0, y + 5, Math.max(2, R.loss * bw * e), rowH - 10, 3); g.fill();
              V.text(g, fmt(R.loss, 3), lx0 + R.loss * bw * e + 5, y + rowH / 2, { size: 11.5, font: 'mono', base: 'middle', color: C.ink });
            }
          }
          if (k >= 4) V.text(g, 'perda média = ' + fmt(c.mean, 4), tx, ty + m * rowH + 8, { size: 12, font: 'mono', weight: 700, color: C.ink });
        }
      });
    })();

    /* ================================================== FIG 7 — SimPO soma × média */
    (function () {
      const fig = root.querySelector('#fig-simpo');
      adaptH(fig, 460, 580);
      const S = { agg: 'media', b: 2, gm: 1, k: 1 };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'agregar', options: [{ v: 'soma', label: 'soma (sem LN)' }, { v: 'media', label: 'média (SimPO)' }], value: 'media', onChange: (v) => { S.agg = v; upd(); } });
      const sB = UI.slider(ctl, { label: 'β', min: 0.5, max: 10, step: 0.1, value: 2, fmt: (v) => fmt(v, 1), onInput: (v) => { S.b = v; upd(); } });
      const sG = UI.slider(ctl, { label: 'γ', min: 0, max: 6, step: 0.05, value: 1, fmt: (v) => fmt(v, 2), onInput: (v) => { S.gm = v; upd(); } });
      const PRE = { livre: null, mb: [2, 1.6], mi: [2.5, 0.3], lb: [2, 1], li: [2.5, 1.4], gem: [10, 5] };
      UI.select(ctl, { label: 'preset', options: [{ v: 'livre', label: 'livre' }, { v: 'mb', label: 'Mistral-Base (β 2 · γ 1,6)' }, { v: 'mi', label: 'Mistral-Instruct (β 2,5 · γ 0,3)' }, { v: 'lb', label: 'Llama-3-Base (β 2 · γ 1)' }, { v: 'li', label: 'Llama-3-Instruct (β 2,5 · γ 1,4)' }, { v: 'gem', label: 'Gemma README (β 10 · γ 5)' }], value: 'livre', onChange: (v) => { const p = PRE[v]; if (p) { sB.set(p[0]); sG.set(p[1]); } } });
      const rU = UI.readout(ctl, 'u = r_w − r_l − γ', '—'), rLs = UI.readout(ctl, 'perda', '—'), rWt = UI.readout(ctl, 'peso s_θ', '—'), rR = UI.readout(ctl, 'γ/β', '—');
      nocase(ctl);
      const sW = -25, sL = -17, mW = -1.25, mL = -1.7;
      const vals = (k) => { const aw = lerp(sW, mW, k), al = lerp(sL, mL, k); const rw = S.b * aw, rl = S.b * al; return { rw, rl, u: rw - rl - S.gm }; };
      function upd() { const v = vals(S.agg === 'media' ? 1 : 0); rU.set(sg(v.u, 3)); rLs.set(fmt(nls(v.u), 4), nls(v.u) > 3 ? 'bad' : null); rWt.set(fmt(sig(-v.u), 4)); rR.set(fmt(S.gm / S.b, 2)); }
      upd();
      V.fig(fig, (g, w, h, t, dt) => {
        const C = V.C();
        const narrow = w < 640, pad = 14;
        { const tg = S.agg === 'media' ? 1 : 0; S.k = lerp(S.k, tg, Math.min(1, dt * 3.5)); if (Math.abs(S.k - tg) < 0.003) S.k = tg; }
        const k = S.k;
        const v = vals(k);
        // ---- linha A: trens de tokens
        const accW = narrow ? 112 : 170;
        const areaW = w - 2 * pad - accW - 46;
        const cw = Math.min(30, (areaW - 19 * 2) / 20), ch = 22;
        const trains = [{ nm: 'y_w', arr: POLW, y: 34, col: C.c, s: sW, m: mW }, { nm: 'y_l', arr: POLL, y: 34 + ch + 30, col: C.b, s: sL, m: mL }];
        trains.forEach((T) => {
          V.text(g, T.nm, pad, T.y + ch / 2, { size: 12.5, font: 'mono', weight: 700, color: T.col, base: 'middle' });
          const x0 = pad + 36;
          T.arr.forEach((vv, i) => {
            const x = x0 + i * (cw + 2);
            const tt = clamp(1 + vv / 3.2, 0, 1);
            V.box(g, x, T.y, cw, ch, { r: 4, fill: V.alpha(T.col, 0.12 + 0.5 * tt), stroke: V.alpha(T.col, 0.35), lw: 1 });
            // partículas escoando para o agregador
            const ph = (t * 0.45 + i * 0.137) % 1;
            const bx = w - pad - accW, by = T.y + ch / 2;
            if (ph < 0.85) { const q = V.ease.inOut(ph / 0.85); V.dot(g, lerp(x + cw / 2, bx, q), lerp(T.y + ch / 2, by, q) - Math.sin(q * Math.PI) * 8, 2, V.alpha(T.col, 0.5 * (1 - q) + 0.2)); }
          });
          const bx = w - pad - accW;
          V.box(g, bx, T.y - 4, accW, ch + 8, { r: 7, fill: V.alpha(T.col, 0.14), stroke: T.col, lw: 1.3 });
          const agg = lerp(T.s, T.m, k);
          V.text(g, (k > 0.5 ? 'média ' : 'Σ ') + fmt(agg, 2), bx + accW / 2, T.y + ch / 2 + 0.5, { size: 12.5, font: 'mono', weight: 700, align: 'center', base: 'middle', color: C.ink });
        });
        // ---- linha B: reta das recompensas
        const yB = 34 + 2 * (ch + 30) + 26;
        const lo = Math.min(v.rw, v.rl) - Math.max(0.6, 0.15 * Math.abs(v.rw - v.rl)), hi = Math.max(v.rw, v.rl + S.gm) + Math.max(0.6, 0.15 * Math.abs(v.rw - v.rl));
        const X = (x) => pad + 20 + ((x - lo) / (hi - lo)) * (w - 2 * pad - 40);
        V.text(g, 'recompensas r = β · ' + (k > 0.5 ? 'média' : 'soma'), pad, yB - 12, { size: 12, color: C.muted });
        V.line(g, pad + 10, yB + 16, w - pad - 10, yB + 16, { color: C.muted, lw: 1.4 });
        nice(lo, hi, narrow ? 4 : 7).forEach((tv) => { V.line(g, X(tv), yB + 12, X(tv), yB + 20, { color: C.muted, lw: 1 }); V.text(g, fmtN(tv), X(tv), yB + 33, { size: 11, align: 'center', font: 'mono', color: C.muted }); });
        // margem γ a partir de r_l
        g.fillStyle = V.alpha(C.d, 0.25); g.fillRect(X(v.rl), yB + 6, X(v.rl + S.gm) - X(v.rl), 20);
        V.text(g, 'γ', (X(v.rl) + X(v.rl + S.gm)) / 2, yB + 2, { size: 12, align: 'center', color: C.d, weight: 700 });
        V.dot(g, X(v.rl), yB + 16, 7, C.b); V.dot(g, X(v.rw), yB + 16, 7, C.c);
        V.text(g, 'r_l ' + fmt(v.rl, 2), X(v.rl), yB + 48, { size: 11.5, font: 'mono', align: 'center', color: C.b });
        V.text(g, 'r_w ' + fmt(v.rw, 2), X(v.rw), yB + 62, { size: 11.5, font: 'mono', align: 'center', color: C.c });
        // ---- linha C: perda logística
        const yC = yB + 92;
        const umin = Math.min(-6, v.u - 2), umax = 6;
        const A = axes(g, { x: 50, y: yC, w: w - 50 - pad, h: h - yC - 36, xr: [umin, umax], yr: [0, nls(umin) * 1.05], ny: 3, nx: narrow ? 4 : 6, xl: 'u = r_w − r_l − γ', yl: '−log σ(u)' });
        plotFn(g, A, (u) => nls(u), { color: C.c, lw: 2.6 });
        V.line(g, A.X(0), A.y, A.X(0), A.y + A.h, { color: V.alpha(C.muted, 0.6), lw: 1, dash: [3, 3] });
        const px = A.X(v.u), py = A.Y(nls(v.u));
        V.glow(g, px, py, 18, C.c, 0.6); V.dot(g, px, py, 7, C.c);
        { const lab = 'perda ' + fmt(nls(v.u), 3) + ' · peso ' + fmt(sig(-v.u), 3); const lw = tw(g, lab, 12, 'mono', 700);
          const lx = clamp(px + 12, A.x + 4, w - pad - lw), ly = py < A.y + 34 ? py + 22 : py - 14;
          V.text(g, lab, lx, ly, { size: 12, font: 'mono', weight: 700, color: C.ink }); }
      });
    })();

    /* ================================================== FIG 8 — mapa de parentesco */
    (function () {
      const fig = root.querySelector('#fig-familia');
      adaptH(fig, 470, 600);
      const NODES = [
        { k: 'ipo', n: 'IPO', e: 'Ψ: logit → identidade', tags: ['contraste'], who: 'Azar et al., 2023', f: '(Δh − 1/(2τ))²', fact: 'referência: sim · pares · τ 0,01–1 (atenção: TRL usa média por token)' },
        { k: 'bdpo', n: 'β-DPO', e: 'β por lote', tags: ['contraste'], who: 'Wu et al., 2024', f: 'β_lote = [1 + α(E[M_i] − M₀)]·β₀, com M_i = r̂(y_w) − r̂(y_l)', fact: 'referência: sim · pares · β₀ 0,1 · α 0,6–1,0 · ρ 0,8' },
        { k: 'dpop', n: 'DPOP', e: '+ penalidade se π(y_w) cai', tags: ['contraste'], who: 'Pal et al., 2024 (Smaug)', f: '−log σ(β(Δh − λ·max(0, log π_ref(y_w)/π_θ(y_w))))', fact: 'referência: sim · pares · β 0,3 · λ 50' },
        { k: 'kto', n: 'KTO', e: 'pares → bom/ruim, valor de prospecto', tags: ['dados', 'contraste'], who: 'Ethayarajh et al., 2024', f: 'λ_y − λ_D·σ(β(r − z₀))  |  λ_y − λ_U·σ(β(z₀ − r))', fact: 'referência: sim, + KL estimado · binário · β 0,1 · LR 5e-6 · λ_D·n_D/λ_U·n_U ∈ [1; 1,5]' },
        { k: 'orpo', n: 'ORPO', e: 'odds + SFT, sem ref', tags: ['ref'], who: 'Hong, Lee e Thorne, 2024', f: 'NLL(y_w) − λ·log σ(log odds(y_w) − log odds(y_l))', fact: 'referência: não · pares · λ 0,1 · LR 8e-6 · estágio único' },
        { k: 'cpo', n: 'CPO', e: 'ref uniforme + NLL', tags: ['ref'], who: 'Xu et al., 2024', f: '−log σ(β·log π(y_w) − β·log π(y_l)) − log π(y_w)', fact: 'referência: não (uniforme) · pares · β 0,1 · BC 1 · soma sensível ao comprimento' },
        { k: 'simpo', n: 'SimPO', e: 'sem ref · média · γ', tags: ['ref', 'comp'], who: 'Meng, Xia e Chen, 2024', f: '−log σ(β/|y_w|·log π(y_w) − β/|y_l|·log π(y_l) − γ)', fact: 'referência: não · pares · β 2–2,5 (até 10) · γ/β ≈ 0,5 · LR 3e-7–1e-6' },
        { k: 'lndpo', n: 'LN-DPO', e: 'log-razão ÷ |y|', tags: ['comp'], who: 'Tülu 3 (Lambert et al., 2024)', f: '−log σ(β·h_w/|y_w| − β·h_l/|y_l|)', fact: 'referência: sim · pares · β 5 · TRL loss_type "sigmoid_norm"' },
        { k: 'rdpo', n: 'R-DPO', e: '+ α(|y_w| − |y_l|)', tags: ['comp'], who: 'Park, Rafailov, Ermon e Finn, 2024', f: '−log σ(β·Δh + α(|y_w| − |y_l|))', fact: 'referência: sim · pares + comprimentos · α 0,005–0,05' },
      ];
      const DPO = { k: 'dpo', n: 'DPO', who: 'Rafailov et al., 2023', f: '−log σ(β·Δh),  Δh = h_w − h_l,  h = log π_θ/π_ref', fact: 'referência: sim · pares · β 0,1 (0,01–0,5) · LR 5e-7–1e-6' };
      const CROSS = [['lndpo', 'simpo', 'tira a referência'], ['cpo', 'simpo', 'média + margem']];
      const S = { filt: 'todos', sel: 'simpo' };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'o que muda', options: [{ v: 'todos', label: 'tudo' }, { v: 'ref', label: 'referência' }, { v: 'dados', label: 'dados' }, { v: 'contraste', label: 'contraste' }, { v: 'comp', label: 'comprimento' }], value: 'todos', onChange: (v) => { S.filt = v; } });
      const selC = UI.select(ctl, { label: 'método', options: [{ v: 'dpo', label: 'DPO' }].concat(NODES.map((n) => ({ v: n.k, label: n.n }))), value: 'simpo', onChange: (v) => { S.sel = v; } });
      const byK = (k) => (k === 'dpo' ? DPO : NODES.find((n) => n.k === k));
      const H = V.fig(fig, (g, w, h, t) => {
        const C = V.C(), MCc = MC();
        const narrow = w < 560, pad = 14;
        const infoH = narrow ? 128 : 92;
        const ringH = h - infoH - 16;
        const cx = w / 2, cy = 14 + ringH / 2;
        const rx = narrow ? w / 2 - 44 : Math.min(w / 2 - 90, 340), ry = ringH / 2 - (narrow ? 26 : 24);
        const pos = {};
        NODES.forEach((n, i) => { const a = -Math.PI / 2 + (i * 2 * Math.PI) / NODES.length; pos[n.k] = [cx + rx * Math.cos(a), cy + ry * Math.sin(a)]; });
        pos.dpo = [cx, cy];
        const on = (n) => S.filt === 'todos' || (n.tags || []).includes(S.filt);
        // arestas
        NODES.forEach((n, i) => {
          const [x, y] = pos[n.k];
          const act = on(n);
          V.line(g, cx, cy, x, y, { color: V.alpha(MCc[n.k], act ? 0.55 : 0.12), lw: act ? 1.8 : 1 });
          if (act) for (let q = 0; q < 2; q++) { const ph = (t * 0.35 + i * 0.11 + q * 0.5) % 1; V.dot(g, lerp(cx, x, ph), lerp(cy, y, ph), 2.6, V.alpha(MCc[n.k], 0.9)); }
          if (act && (!narrow || S.sel === n.k)) {
            const mxp = lerp(cx, x, 0.52), myp = lerp(cy, y, 0.52);
            g.font = V.font(11, 'sans', 500);
            const tw0 = g.measureText(n.e).width + 10;
            V.box(g, mxp - tw0 / 2, myp - 9, tw0, 18, { r: 6, fill: V.alpha(C.bg, 0.85), stroke: V.alpha(MCc[n.k], 0.4), lw: 1 });
            V.text(g, n.e, mxp, myp + 0.5, { size: 11, align: 'center', base: 'middle', color: C.ink });
          }
        });
        CROSS.forEach(([a, b, lab]) => {
          const A = pos[a], B = pos[b];
          const act = S.filt === 'todos' || S.filt === 'ref' || S.filt === 'comp';
          V.arrow(g, A[0], A[1], B[0], B[1], { color: V.alpha(C.c, act ? 0.6 : 0.15), lw: 1.4, dash: [4, 4], curve: 18, head: 7 });
          if (act && !narrow) V.text(g, lab, (A[0] + B[0]) / 2 + 22, (A[1] + B[1]) / 2 + 4, { size: 10.5, color: C.c, align: 'left' });
        });
        // nós
        const drawNode = (n, x, y, big) => {
          const act = n.k === 'dpo' || on(n);
          g.font = V.font(big ? 15 : narrow ? 12 : 13, 'sans', 700);
          const ww = g.measureText(n.n).width + (big ? 26 : 18), hh = big ? 34 : 28;
          const col = n.k === 'dpo' ? C.b : MCc[n.k];
          const sel = S.sel === n.k;
          if (sel) V.glow(g, x, y, ww * 0.75, col, 0.55);
          V.box(g, x - ww / 2, y - hh / 2, ww, hh, { r: 9, fill: V.alpha(col, act ? 0.22 : 0.06), stroke: V.alpha(col, act ? 1 : 0.25), lw: sel ? 2.6 : 1.4 });
          V.text(g, n.n, x, y + 0.5, { size: big ? 15 : narrow ? 12 : 13, weight: 700, align: 'center', base: 'middle', color: act ? C.ink : V.alpha(C.ink, 0.35) });
          n._box = [x - ww / 2, y - hh / 2, ww, hh];
        };
        NODES.forEach((n) => drawNode(n, pos[n.k][0], pos[n.k][1], false));
        drawNode(DPO, cx, cy, true);
        // clique
        const p = H && H.pointer;
        if (p && p.clicked) {
          [DPO].concat(NODES).forEach((n) => { const b = n._box; if (b && p.x >= b[0] && p.x <= b[0] + b[2] && p.y >= b[1] && p.y <= b[1] + b[3]) { S.sel = n.k; selC.el.querySelector('select').value = n.k; } });
        }
        // painel de informação
        const sel = byK(S.sel);
        const iy = h - infoH - 6;
        panel(g, pad, iy, w - 2 * pad, infoH, { stroke: V.alpha(sel.k === 'dpo' ? C.b : MCc[sel.k], 0.6) });
        V.text(g, sel.n + ' · ' + sel.who, pad + 12, iy + 20, { size: 13, weight: 700, color: sel.k === 'dpo' ? C.b : MCc[sel.k], maxW: w - 2 * pad - 24 });
        const fh = V.wrap(g, sel.f, pad + 12, iy + 42, w - 2 * pad - 24, { size: 12.5, font: 'mono', color: C.ink, lineH: 17, maxLines: narrow ? 3 : 2 });
        V.wrap(g, sel.fact, pad + 12, iy + 46 + fh, w - 2 * pad - 24, { size: 11.5, color: C.muted, lineH: 15, maxLines: narrow ? 3 : 2 });
      });
    })();

    /* ================================================== FIG 9 — calculadora */
    (function () {
      const fig = root.querySelector('#fig-calc');
      adaptH(fig, 480, 650);
      const RW0 = -26, RL0 = -15, LW = 20, LL = 10;
      const S = { a: -25, b: -17, sel: 'dpo', traj: [], tt: 0, t0: 0 };
      function evalAll(a, b) {
        const hw = a - RW0, hl = b - RL0, dh = hw - hl, out = [];
        { const be = 0.1, u = be * dh, s = sig(-u); out.push({ k: 'dpo', n: 'DPO', hp: 'β 0,1', L: nls(u), ga: be * s, gb: -be * s }); }
        { const cc = 1 / (2 * 0.5), e = dh - cc; out.push({ k: 'ipo', n: 'IPO', hp: 'τ 0,5 (soma)', L: e * e, ga: -2 * e, gb: 2 * e }); }
        { const be = 0.1, z0 = 0.5, vw = sig(be * (hw - z0)), vl = sig(be * (z0 - hl)); out.push({ k: 'kto', n: 'KTO', hp: 'β 0,1 · z₀ 0,5', L: (2 - vw - vl) / 2, ga: be * vw * (1 - vw), gb: -be * vl * (1 - vl) }); }
        { const be = 2, gm = 1, u = be * (a / LW - b / LL) - gm, s = sig(-u); out.push({ k: 'simpo', n: 'SimPO', hp: 'β 2 · γ 1', L: nls(u), ga: (be * s) / LW, gb: (-be * s) / LL }); }
        { const lam = 0.1, Pw = Math.exp(a / LW), Pl = Math.exp(b / LL), lo = (P) => Math.log(P / (1 - P)), z = lo(Pw) - lo(Pl), s = sig(-z); out.push({ k: 'orpo', n: 'ORPO', hp: 'λ 0,1', L: -a / LW + lam * nls(z), ga: 1 / LW + (lam * s) / (LW * (1 - Pw)), gb: (-lam * s) / (LL * (1 - Pl)) }); }
        { const be = 0.1, al = 0.01, u = be * dh + al * (LW - LL), s = sig(-u); out.push({ k: 'rdpo', n: 'R-DPO', hp: 'β 0,1 · α 0,01', L: nls(u), ga: be * s, gb: -be * s }); }
        { const be = 0.1, u = be * (a - b), s = sig(-u); out.push({ k: 'cpo', n: 'CPO', hp: 'β 0,1 (pref.)', L: nls(u), ga: be * s, gb: -be * s }); }
        return out;
      }
      const XR = [-34, -18], YR = [-24, -8];
      function restartTraj() { S.traj = [[S.a, S.b]]; S.cur = [S.a, S.b]; S.t0 = H ? H.t : 0; S.g0 = null; }
      const ctl = UI.controls(fig);
      const sA = UI.slider(ctl, { label: 'log π_θ(y_w)', min: -34, max: -18, step: 0.5, value: -25, fmt: (v) => fmt(v, 1), onInput: (v) => { S.a = v; restartTraj(); upd(); } });
      const sB = UI.slider(ctl, { label: 'log π_θ(y_l)', min: -24, max: -8, step: 0.5, value: -17, fmt: (v) => fmt(v, 1), onInput: (v) => { S.b = v; restartTraj(); upd(); } });
      UI.seg(ctl, { label: 'estado', options: [{ v: 'ini', label: 'π_θ = π_ref' }, { v: 'passos', label: 'após alguns passos' }, { v: 'desl', label: 'deslocamento' }], value: 'passos', onChange: (v) => { const p = v === 'ini' ? [-26, -15] : v === 'passos' ? [-25, -17] : [-28, -21]; sA.set(p[0], true); sB.set(p[1], true); S.a = p[0]; S.b = p[1]; restartTraj(); upd(); } });
      const selC = UI.select(ctl, { label: 'fluxo de', options: ['dpo', 'ipo', 'kto', 'simpo', 'orpo', 'rdpo', 'cpo'].map((k) => ({ v: k, label: { dpo: 'DPO', ipo: 'IPO', kto: 'KTO', simpo: 'SimPO', orpo: 'ORPO', rdpo: 'R-DPO', cpo: 'CPO' }[k] })), value: 'dpo', onChange: (v) => { S.sel = v; restartTraj(); } });
      const rH = UI.readout(ctl, 'Δh', '—'), rM = UI.readout(ctl, 'médias por token', '—');
      nocase(ctl);
      function upd() { rH.set(sg((S.a - RW0) - (S.b - RL0), 2)); rM.set(fmt(S.a / LW, 2) + ' | ' + fmt(S.b / LL, 2)); }
      let H = null;
      upd();
      H = V.fig(fig, (g, w, h, t, dt) => {
        const C = V.C(), MCc = MC();
        const narrow = w < 700, pad = 14;
        const rows = evalAll(S.a, S.b);
        // trajetória do método selecionado (fluxo de gradiente com velocidade normalizada)
        if (!S.cur) restartTraj();
        if (!LA.paused && dt > 0) {
          for (let k = 0; k < 6; k++) {
            const r = evalAll(S.cur[0], S.cur[1]).find((x) => x.k === S.sel);
            const mag = Math.hypot(r.ga, r.gb);
            if (S.g0 == null) S.g0 = mag || 1;
            const sp = 2.6 * Math.min(1, mag / S.g0);
            if (mag > 1e-9) { S.cur[0] += (sp * r.ga / mag) * (dt / 6); S.cur[1] += (sp * r.gb / mag) * (dt / 6); }
          }
          S.traj.push([S.cur[0], S.cur[1]]); if (S.traj.length > 400) S.traj.shift();
          const out = S.cur[0] < XR[0] || S.cur[0] > XR[1] || S.cur[1] < YR[0] || S.cur[1] > YR[1];
          if (t - S.t0 > 6 || out) restartTraj();
        }
        // ---- plano
        const Pp = narrow ? { x: 50, y: 22, w: w - 50 - pad, h: 250 } : { x: 54, y: 24, w: w * 0.44 - 54, h: h - 68 };
        const A = axes(g, Object.assign({}, Pp, { xr: XR, yr: YR, nx: 4, ny: 4, xl: 'log π_θ(y_w) →', yl: 'log π_θ(y_l) ↑' }));
        g.save(); g.beginPath(); g.rect(A.x, A.y, A.w, A.h); g.clip();
        const lineAB = (f, col, lab) => { V.line(g, A.X(XR[0]), A.Y(f(XR[0])), A.X(XR[1]), A.Y(f(XR[1])), { color: col, lw: 1.4, dash: [6, 4] }); };
        lineAB((a) => a + 11, V.alpha(C.b, 0.8)); // Δh = 0
        lineAB((a) => a + 10, V.alpha(C.a, 0.9)); // Δh = 1 (alvo IPO)
        lineAB((a) => a / 2 - 5, V.alpha(C.c, 0.9)); // u = 0 SimPO
        // trilha
        g.beginPath(); S.traj.forEach(([x, y], i) => { const px = A.X(x), py = A.Y(y); if (!i) g.moveTo(px, py); else g.lineTo(px, py); });
        g.strokeStyle = V.alpha(MCc[S.sel], 0.8); g.lineWidth = 2.4; g.stroke();
        if (S.cur) { V.glow(g, A.X(S.cur[0]), A.Y(S.cur[1]), 12, MCc[S.cur && S.sel], 0.8); V.dot(g, A.X(S.cur[0]), A.Y(S.cur[1]), 4, MCc[S.sel]); }
        g.restore();
        V.text(g, 'Δh = 0', A.X(-34) + 4, A.Y(-23) - 4, { size: 11, color: C.b, font: 'mono' });
        V.text(g, 'alvo IPO', A.X(-20.5), A.Y(-10.5) + 14, { size: 11, color: C.a, font: 'mono', align: 'right' });
        V.text(g, 'u = 0 (SimPO)', A.X(-19), A.Y(-14.5) + 14, { size: 11, color: C.c, font: 'mono', align: 'right' });
        // referência
        const sx = A.X(RW0), sy = A.Y(RL0);
        g.beginPath(); for (let i = 0; i < 10; i++) { const an = -Math.PI / 2 + (i * Math.PI) / 5, rr = i % 2 ? 3.5 : 8; g.lineTo(sx + rr * Math.cos(an), sy + rr * Math.sin(an)); } g.closePath(); g.fillStyle = C.d; g.fill();
        V.text(g, 'π_ref', sx - 10, sy - 10, { size: 11, color: C.d, font: 'mono', align: 'right' });
        // ponto atual e setas
        const px = A.X(S.a), py = A.Y(S.b);
        rows.forEach((r) => {
          const mag = Math.hypot(r.ga, r.gb); if (mag < 1e-9) return;
          const L = r.k === S.sel ? 46 : 32;
          const dx = (r.ga / mag) * L, dy = -(r.gb / mag) * L;
          V.arrow(g, px, py, px + dx, py + dy, { color: V.alpha(MCc[r.k], r.k === S.sel ? 1 : 0.55), lw: r.k === S.sel ? 2.8 : 1.6, head: r.k === S.sel ? 9 : 6 });
        });
        V.dot(g, px, py, 6, C.ink);
        // ---- tabela (matriz de valores)
        const T = narrow ? { x: pad, y: Pp.y + Pp.h + 46, w: w - 2 * pad } : { x: w * 0.47, y: 26, w: w * 0.53 - pad };
        const col0 = narrow ? 96 : 132, cwv = (T.w - col0) / 3;
        const rh = narrow ? Math.min(40, (h - T.y - 30) / 8) : Math.min(46, (h - T.y - 34) / 8);
        const heads = ['perda', 'empurrão y_w', 'empurrão y_l'];
        heads.forEach((hd, j) => V.text(g, hd, T.x + col0 + j * cwv + cwv / 2, T.y + 12, { size: 11.5, align: 'center', color: C.muted, maxW: cwv - 4 }));
        const maxL = Math.max(...rows.map((r) => Math.min(r.L, 20)));
        rows.forEach((r, i) => {
          const y = T.y + 22 + i * rh;
          const selR = r.k === S.sel;
          if (selR) { V.box(g, T.x - 4, y - 2, T.w + 8, rh, { r: 8, fill: V.alpha(MCc[r.k], 0.12), stroke: MCc[r.k], lw: 1.4 }); }
          V.text(g, r.n, T.x + 4, y + rh / 2 - (narrow ? 5 : 6), { size: narrow ? 12 : 13, weight: 700, color: MCc[r.k], base: 'middle' });
          V.text(g, r.hp, T.x + 4, y + rh / 2 + (narrow ? 8 : 9), { size: narrow ? 10 : 10.5, color: C.muted, base: 'middle', font: 'mono', maxW: col0 - 8 });
          const cells = [r.L, r.ga, r.gb];
          cells.forEach((v, j) => {
            const x = T.x + col0 + j * cwv + 3, cw = cwv - 6, chh = rh - 6;
            let fill;
            if (j === 0) fill = V.alpha(C.d, 0.1 + 0.5 * Math.min(1, Math.min(v, 20) / (maxL || 1)));
            else { const tt = clamp((Math.log10(Math.abs(v) + 1e-9) + 3) / 3.6, 0.04, 1); fill = V.alpha(v >= 0 ? C.a : C.c, 0.08 + 0.55 * tt); }
            V.box(g, x, y + 1, cw, chh, { r: 6, fill, stroke: V.alpha(C.muted, 0.2), lw: 1 });
            const txt = j === 0 ? fmt(v, v >= 10 ? 2 : 4) : (v >= 0 ? '↑ ' : '↓ ') + fmt(Math.abs(v), Math.abs(v) >= 1 ? 2 : 4);
            V.text(g, txt, x + cw / 2, y + 1 + chh / 2 + 0.5, { size: narrow ? 11 : 12, font: 'mono', align: 'center', base: 'middle', color: C.ink, maxW: cw - 4 });
          });
        });
        const ny = T.y + 22 + rows.length * rh + 14;
        if (ny < h - 4) V.text(g, '↑ sobe a log-prob · ↓ desce (−∂L/∂log π)', T.x, ny, { size: 11, color: C.muted, maxW: T.w });
        // clique na tabela
        const p = H && H.pointer;
        if (p && p.clicked) rows.forEach((r, i) => { const y = T.y + 22 + i * rh; if (p.x >= T.x && p.x <= T.x + T.w && p.y >= y && p.y <= y + rh) { S.sel = r.k; selC.el.querySelector('select').value = r.k; restartTraj(); } });
      });
    })();

    /* ================================================== FIG 10 — decisão e memória */
    (function () {
      const fig = root.querySelector('#fig-decisao');
      adaptH(fig, 460, 720);
      const S = { dados: 'pares', sit: 'padrao', N: 8, lora: false, pre: false, anim: {} };
      const SIT = [
        { v: 'padrao', label: 'pares limpos + SFT', m: 'DPO', k: 'dpo', why: 'β 0,1 · LR 5e-7–1e-6 · 1 época · monitore margens e logps/chosen' },
        { v: 'ruido', label: 'rótulo único / ruidoso', m: 'IPO', k: 'ipo', why: 'alvo finito 1/(2τ) · varra τ · cuidado com a escala por token do TRL' },
        { v: 'memoria', label: 'memória curta', m: 'SimPO', k: 'simpo', why: 'sem referência · β 2–2,5 · γ/β 0,5 · LR conservador (esquecimento)' },
        { v: 'verbo', label: 'respostas longas demais', m: 'R-DPO / LN-DPO', k: 'rdpo', why: 'α 0,005–0,05 (R-DPO) ou log-razões médias com β 5 (Tülu 3)' },
        { v: 'semsft', label: 'ainda sem SFT', m: 'ORPO', k: 'orpo', why: 'SFT + preferência num estágio · λ 0,1 · sem referência' },
        { v: 'desl', label: 'π(y_w) caindo', m: 'DPOP / + NLL', k: 'dpop', why: 'β 0,3 · λ 50, ou loss_type ["sigmoid","sft"], ou filtrar pares (CHES)' },
      ];
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'seus dados', options: [{ v: 'pares', label: 'pares (y_w ≻ y_l)' }, { v: 'bin', label: 'bom / ruim' }], value: 'pares', onChange: (v) => { S.dados = v; gS.show(v === 'pares'); upd(); } });
      const gS = UI.group(ctl);
      UI.select(gS.el, { label: 'situação', options: SIT.map((s) => ({ v: s.v, label: s.label })), value: 'padrao', onChange: (v) => { S.sit = v; upd(); } });
      UI.slider(ctl, { label: 'modelo', min: 1, max: 14, step: 1, value: 8, fmt: (v) => v + 'B', onInput: (v) => { S.N = v; upd(); } });
      UI.toggle(ctl, { label: 'LoRA', value: false, onChange: (v) => { S.lora = v; upd(); } });
      UI.toggle(ctl, { label: 'pré-computar log-probs da referência', value: false, onChange: (v) => { S.pre = v; upd(); } });
      const rRec = UI.readout(ctl, 'recomendação', '—'), rMem = UI.readout(ctl, 'memória (sem ativações)', '—');
      nocase(ctl);
      const rec = () => (S.dados === 'bin' ? { m: 'KTO', k: 'kto', why: 'binário · λ_D·n_D/λ_U·n_U ∈ [1; 1,5] · LR ~5e-6 · microbatch ≥ 2' } : SIT.find((s) => s.v === S.sit));
      const hasRef = (k) => !['simpo', 'orpo', 'cpo'].includes(k);
      const mem = (withRef) => {
        const N = S.N;
        if (S.lora) return { parts: [['base BF16 (também é a referência)', 2 * N]], tot: 2 * N };
        const parts = [['pesos BF16', 2 * N], ['gradientes', 2 * N], ['AdamW + cópia FP32', 12 * N]];
        if (withRef && !S.pre) parts.push(['referência BF16', 2 * N]);
        return { parts, tot: parts.reduce((a, b) => a + b[1], 0) };
      };
      function upd() { const r = rec(); rRec.set(r.m); const mm = mem(hasRef(r.k)); rMem.set('≈ ' + fmt(mm.tot, 0) + ' GB', mm.tot > 128 ? 'bad' : mm.tot > 100 ? 'warn' : 'ok'); }
      upd();
      V.fig(fig, (g, w, h, t, dt) => {
        const C = V.C(), MCc = MC();
        const narrow = w < 700, pad = 14;
        const r = rec();
        // ---- árvore de decisão
        const D = narrow ? { x: pad, y: 10, w: w - 2 * pad, h: 380 } : { x: pad, y: 10, w: w * 0.55 - pad, h: h - 20 };
        const rootB = [D.x + D.w / 2 - 90, D.y + 4, 180, 30];
        const brY = D.y + 60;
        const bP = [D.x + 4, brY, 132, 28], bB = [D.x + D.w - 136, brY, 132, 28];
        const onP = S.dados === 'pares';
        const ln = (x1, y1, x2, y2, act, c) => { V.line(g, x1, y1, x2, y2, { color: act ? c : V.alpha(C.muted, 0.35), lw: act ? 2.4 : 1.2, dash: act ? [6, 5] : null, dashOffset: act ? -t * 24 : 0 }); };
        ln(rootB[0] + 90, rootB[1] + 30, bP[0] + 66, bP[1], onP, C.a);
        ln(rootB[0] + 90, rootB[1] + 30, bB[0] + 66, bB[1], !onP, C.d);
        chip(g, rootB[0], rootB[1], rootB[2], rootB[3], 'Que dados você tem?', { font: 'sans', size: 12.5, weight: 700 });
        chip(g, bP[0], bP[1], bP[2], bP[3], 'pares', { font: 'sans', size: 12, fill: V.alpha(C.a, onP ? 0.22 : 0.06), stroke: onP ? C.a : V.alpha(C.muted, 0.3) });
        chip(g, bB[0], bB[1], bB[2], bB[3], 'bom / ruim', { font: 'sans', size: 12, fill: V.alpha(C.d, !onP ? 0.22 : 0.06), stroke: !onP ? C.d : V.alpha(C.muted, 0.3) });
        // situações: coluna com uma espinha à esquerda
        const sy0 = brY + 46, sh = narrow ? 26 : Math.min(30, (D.h - 210) / SIT.length);
        const spX = D.x + 16, sx = D.x + 30, sw = Math.min(D.w * 0.6, 230);
        const selI = SIT.findIndex((s) => s.v === S.sit);
        const cyOf = (i) => sy0 + i * (sh + 4) + sh / 2;
        ln(spX, bP[1] + 28, spX, cyOf(SIT.length - 1), false, C.a);
        if (onP) ln(spX, bP[1] + 28, spX, cyOf(selI), true, C.a);
        SIT.forEach((s, i) => {
          const y = sy0 + i * (sh + 4);
          const act = onP && i === selI;
          ln(spX, cyOf(i), sx, cyOf(i), act, C.a);
          chip(g, sx, y, sw, sh, s.label, { font: 'sans', size: 11.5, fill: V.alpha(C.a, act ? 0.2 : 0.05), stroke: act ? C.a : V.alpha(C.muted, 0.3), color: act ? C.ink : V.alpha(C.ink, onP ? 0.7 : 0.4) });
        });
        // folha
        const leafY = sy0 + SIT.length * (sh + 4) + 18;
        const leafH = D.y + D.h - leafY - 4;
        const col = MCc[r.k] || C.a;
        if (onP) { const xr = sx + sw + 16; ln(sx + sw, cyOf(selI), xr, cyOf(selI), true, col); ln(xr, cyOf(selI), xr, leafY, true, col); }
        else ln(bB[0] + 66, bB[1] + 28, bB[0] + 66, leafY, true, col);
        if (leafH > 40) {
          V.box(g, D.x + 4, leafY, D.w - 8, leafH, { r: 12, fill: V.alpha(col, 0.16), stroke: col, lw: 2 });
          V.text(g, r.m, D.x + D.w / 2, leafY + 25, { size: 19, weight: 800, align: 'center', color: C.ink, font: 'display' });
          V.wrap(g, r.why, D.x + 18, leafY + 47, D.w - 36, { size: 11.5, color: C.ink, lineH: 15, maxLines: 3 });
        }
        // ---- barras de memória
        const M = narrow ? { x: 56, y: D.y + D.h + 44, w: w - 56 - pad, h: h - (D.y + D.h + 44) - 64 } : { x: w * 0.6 + 40, y: 40, w: w * 0.4 - 40 - pad, h: h - 104 };
        const fam = [{ nm: 'com referência', sub: 'DPO · IPO · KTO', ref: true }, { nm: 'sem referência', sub: 'SimPO · ORPO · CPO', ref: false }];
        const mems = fam.map((f) => mem(f.ref));
        const ymax = Math.max(140, ...mems.map((m) => m.tot)) * 1.12;
        const A = axes(g, Object.assign({}, M, { xr: [0, 1], yr: [0, ymax], noX: true, ny: 4, yl: 'GB (estimativa)' }));
        const segCol = [C.b, C.c, C.d, C.a];
        const bw = Math.min(84, A.w / 3.6);
        fam.forEach((f, i) => {
          const x = A.x + A.w * (0.22 + i * 0.38) - bw / 2;
          const mm = mems[i];
          const key = 'b' + i;
          S.anim[key] = lerp(S.anim[key] || 0, mm.tot, Math.min(1, dt * 5));
          const sc = mm.tot ? S.anim[key] / mm.tot : 1;
          let acc = 0;
          mm.parts.forEach(([nm, v], j) => {
            const y0 = A.Y(acc * sc), y1 = A.Y((acc + v) * sc);
            g.fillStyle = V.alpha(S.lora ? C.a : segCol[j], 0.7); V.rr(g, x, y1, bw, Math.max(1, y0 - y1), 3); g.fill();
            acc += v;
          });
          const isRec = hasRef(r.k) === f.ref;
          if (isRec) { g.strokeStyle = col; g.lineWidth = 2.4; V.rr(g, x - 3, A.Y(S.anim[key]) - 3, bw + 6, A.Y(0) - A.Y(S.anim[key]) + 6, 5); g.stroke(); }
          V.text(g, fmt(mm.tot, 0) + ' GB', x + bw / 2, A.Y(S.anim[key]) - 8, { size: 12, font: 'mono', weight: 700, align: 'center', color: C.ink });
          V.text(g, f.nm, x + bw / 2, A.y + A.h + 15, { size: 11.5, align: 'center', color: C.ink, weight: 600, maxW: A.w / 2 });
          V.text(g, f.sub, x + bw / 2, A.y + A.h + 30, { size: 10.5, align: 'center', color: C.muted, maxW: A.w / 2 });
        });
        [[128, '1× Spark · 128 GB'], [256, '2× Spark · 256 GB']].forEach(([v, lab]) => {
          if (v > ymax) return;
          V.line(g, A.x, A.Y(v), A.x + A.w, A.Y(v), { color: V.alpha(C.red, 0.75), lw: 1.4, dash: [6, 4] });
          V.text(g, lab, A.x + A.w, A.Y(v) - 5, { size: 11, align: 'right', color: C.red });
        });
        // legenda
        const lg = S.lora ? [['base BF16 = política e referência', C.a]] : [['pesos', segCol[0]], ['grad.', segCol[1]], ['AdamW', segCol[2]], ['ref.', segCol[3]]];
        let lx = M.x;
        const ly = narrow ? h - 10 : M.y + M.h + 50;
        lg.forEach(([nm, c]) => { V.box(g, lx, ly - 9, 10, 10, { r: 2, fill: V.alpha(c, 0.8) }); V.text(g, nm, lx + 14, ly, { size: 11, color: C.muted }); lx += tw(g, nm, 11) + 30; });
      });
    })();
  },
});

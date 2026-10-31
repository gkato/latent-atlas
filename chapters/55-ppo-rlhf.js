LA.chapter({
  id: 'ppo-rlhf',
  part: 11,
  order: 55,
  title: 'PPO e o pipeline RLHF',
  navTitle: 'PPO & RLHF',
  subtitle: 'Do objetivo clipado ao sistema inteiro: a coleira de KL, os quatro modelos na memória, o motor de inferência gerando rollouts dentro do laço e os truques que mantêm o treino de pé.',
  minutes: 45,
  recipes: [],
  tags: ['PPO', 'RLHF', 'clipping', 'penalidade de KL', 'InstructGPT', 'crítico', 'GAE', 'PPO-ptx', 'vLLM no laço', 'OpenRLHF', 'verl'],
  goals: [
    'Derivar e interpretar o <b>objetivo clipado</b> \\(L^{CLIP}\\) caso a caso, explicando por que o \\(\\min\\) corta o incentivo e como isso imita a região de confiança do TRPO',
    'Montar a perda completa do PPO (valor clipado, entropia, épocas e minibatches) e o pipeline <b>SFT → RM → PPO</b> do InstructGPT, com os hiperparâmetros reais',
    'Explicar a <b>recompensa com KL</b> \\(r - \\beta\\log(\\pi/\\pi_{\\text{ref}})\\), a política ótima \\(\\pi^*\\propto\\pi_{\\text{ref}}e^{r/\\beta}\\), o controle adaptativo de \\(\\beta\\) e o PPO-ptx',
    'Estimar a memória dos <b>quatro modelos</b> e o custo de uma iteração, e escolher entre motor de inferência colocado, desagregado ou assíncrono',
    'Diagnosticar instabilidades pelo sintoma e escolher o truque e a ferramenta certos em 2026 (OpenRLHF, verl; o TRL removeu o <code>PPOTrainer</code>)',
  ],
  prereqs: ['rl-fundamentos', 'preferencias-reward-models'],
  summary: [
    'O PPO reaproveita cada lote por vários passos usando a razão \\(\\rho_t = \\pi_\\theta/\\pi_{\\theta_{\\text{old}}}\\) como peso de importância por token. O \\(\\min\\) de \\(L^{CLIP}\\) zera o incentivo quando \\(\\rho\\) sai de \\([1-\\varepsilon, 1+\\varepsilon]\\) na direção que melhoraria o objetivo (\\(\\varepsilon = 0{,}2\\)): é um corte de incentivo, não uma parede, e substitui o caro TRPO por um otimizador de primeira ordem.',
    'Perda completa: \\(L^{CLIP} - c_1L^{VF} + c_2S\\). Em RLHF, o crítico é um LM separado, iniciado do RM, que prevê \\(V(s_t)\\) em cada token; o valor é clipado (\\(\\hat\\varepsilon = 0{,}2\\)); o bônus de entropia fica desligado porque a KL já age como entropia; poucas épocas (InstructGPT 1, Stiennon e N+ 4).',
    'InstructGPT: SFT (12.725 prompts) → RM de 6B (33.207 prompts, \\(\\binom K2\\) pares no mesmo elemento do batch) → PPO (31.144 prompts, \\(\\beta = 0{,}02\\), 256k episódios, batch 512) + ptx (\\(\\gamma = 27{,}8\\)). O 1,3B foi preferido ao GPT-3 175B; e o PPO-ptx do 175B custou 60 pf/s-dias, contra 3.640 do pré-treino do GPT-3.',
    'Recompensa com coleira: \\(r_t = -\\beta(\\log\\pi_{\\text{old}} - \\log\\pi_{\\text{ref}}) + \\mathbb 1[t=T]\\,r_\\phi\\). A política ótima repesa o SFT por \\(e^{r/\\beta}\\): \\(\\beta\\) pequeno entrega a massa ao exploit. O controlador de Ziegler (\\(K_\\beta = 0{,}1\\), ±2% por passo) mira uma KL alvo em vez de um \\(\\beta\\).',
    'Quatro modelos: política e crítico treináveis (~16 B/parâmetro), referência e RM congelados (2 B/parâmetro) → \\(36N\\) bytes; 7B dá 252 GB, além de um DGX Spark (~121,69 GiB). Fora da conta: ativações, a cópia do motor de inferência e o KV dos rollouts.',
    'O motor de inferência gera os rollouts (Llama 2: gerar com FSDP ~20× mais lento; OpenRLHF: inferência muitas vezes &gt; 90% do tempo). Colocado (sleep/wake, IPC) ou desagregado (broadcast NCCL; 14 GB de um 7B ≈ 0,9 s no GB10); RL assíncrono é off-policy de 1 passo e mediu de ~25% a ~70% de ganho. \\(\\log\\pi_{\\text{old}}\\) é recalculado no motor de treino; TIS, LM head em FP32 e R3 corrigem o descompasso.',
    'Estabilidade: dropout desligado, EOS ≠ PAD e “EOS trick”, RM normalizado, vantagens branqueadas, valor clipado, clip de gradiente 1,0; vigie KL, comprimento e a razão na 1ª época. Ferramentas: o TRL removeu o <code>PPOTrainer</code> na v1.13.0; use OpenRLHF (<code>train_ppo_ray</code>) ou verl (<code>main_ppo</code>).',
  ],
  quiz: [
    { q: 'Com \\(\\varepsilon = 0{,}2\\), um token com vantagem \\(\\hat A = +2\\) está com razão \\(\\rho = 1{,}3\\). Qual é a derivada de \\(L^{CLIP}\\) em relação a \\(\\rho\\)?', o: ['2: o gradiente continua empurrando a ação boa', '0: o termo clipado é o menor, e o incentivo a subir mais foi cortado', '−1: o PPO puxa a razão de volta para 1', 'Depende do \\(\\beta\\) da KL até a referência'], a: 1, e: '\\(\\min(1{,}3\\times2,\\ 1{,}2\\times2) = 2{,}4\\), que é o termo clipado, constante em \\(\\rho\\): derivada 0. O PPO não puxa a razão de volta (não é uma parede), só deixa de empurrar. E o \\(\\beta\\) da KL até a referência é outra coisa: controla a coleira até o SFT, não o clip de cada iteração.' },
    { q: 'Agora o token tem \\(\\hat A = -1\\) e \\(\\rho = 1{,}3\\) (a ação ruim ficou mais provável). O que acontece?', o: ['Nada: fora da faixa o gradiente é sempre zero', 'O \\(\\min\\) escolhe o termo sem clip (\\(-1{,}3 &lt; -1{,}2\\)), e a penalidade continua com derivada −1', 'A razão é reiniciada para 1', 'O crítico absorve o erro'], a: 1, e: 'Com vantagem negativa, \\(\\rho\\hat A = -1{,}3\\) é menor que \\(\\operatorname{clip}(\\rho)\\hat A = -1{,}2\\), então o \\(\\min\\) escolhe o termo não clipado e a derivada continua −1. É a assimetria do PPO: ignora a mudança na razão só quando ela melhoraria o objetivo.' },
    { q: 'Como a recompensa de cada token é montada no PPO-RLHF clássico (InstructGPT, N+)?', o: ['A nota do RM é dividida igualmente entre todos os tokens', 'Cada token recebe \\(-\\beta(\\log\\pi_{\\text{old}} - \\log\\pi_{\\text{ref}})\\), e só o último (EOS) recebe também a nota do RM', 'O RM é avaliado em cada prefixo e a nota de cada prefixo vai para o seu token', 'Só o EOS recebe recompensa; a KL entra apenas na perda'], a: 1, e: 'É a decomposição por token de \\(R = r_\\phi - \\beta\\log(\\pi/\\pi_{\\text{ref}})\\): com \\(\\gamma = 1\\), a soma das \\(r_t\\) dá \\(R\\). Pôr a KL na perda, e não na recompensa, é o que o GRPO faz (com o estimador \\(k_3\\)).' },
    { q: 'Na política ótima \\(\\pi^* \\propto \\pi_{\\text{ref}}\\,e^{r/\\beta}\\), o que acontece quando \\(\\beta \\to 0\\)?', o: ['A política volta para o SFT', 'A massa se concentra na resposta de maior nota do RM, mesmo que o SFT quase nunca a escrevesse: é o caminho do reward hacking', 'A entropia da política aumenta', 'Nada muda: \\(\\beta\\) só afeta a velocidade do treino'], a: 1, e: 'Com \\(\\beta\\) pequeno, \\(e^{r/\\beta}\\) domina \\(\\pi_{\\text{ref}}\\) e a maior nota leva tudo. No exemplo da figura, com \\(\\beta = 0{,}3\\) o exploit já leva 68% da massa e a qualidade real média cai para −0,8. Com \\(\\beta \\to \\infty\\), a política fica em \\(\\pi_{\\text{ref}}\\).' },
    { q: 'Política e crítico de 7B em full fine-tuning (Adam, precisão mista), mais referência e RM de 7B em bf16. Quanto ocupam só os estados, e cabe num DGX Spark?', o: ['56 GB; cabe com folga', '126 GB; cabe no limite', '252 GB (~235 GiB); não cabe num Spark (~121,69 GiB) e, com ativações e KV, nem em dois', '504 GB; nem em quatro Sparks'], a: 2, e: '\\(2\\times16N + 2\\times2N = 36N\\) = 252 GB para \\(N = 7\\)B, sem contar ativações, a cópia do motor de inferência e o KV dos rollouts. Saídas: crítico e RM menores, LoRA (referência = adaptador desligado), ou abandonar o crítico (GRPO). Offload para a CPU não libera memória física no GB10.' },
    { q: 'Por que o passo de pontuação recalcula \\(\\log\\pi_{\\theta_{\\text{old}}}\\) no motor de treino, em vez de usar os logprobs que o vLLM devolveu?', o: ['Porque o vLLM não devolve logprobs', 'Porque, com os mesmos pesos, kernels, precisão e batching diferentes dão probabilidades diferentes; usar as do vLLM faria a razão já sair diferente de 1 e tornaria o treino implicitamente off-policy', 'Porque o motor de treino é mais rápido para gerar', 'Para economizar memória de KV'], a: 1, e: '\\(\\pi_{\\text{vllm}}(a;\\theta) \\ne \\pi_{\\text{fsdp}}(a;\\theta)\\) mesmo com os mesmos parâmetros, e o problema piora com rollouts quantizados e MoE. A razão do PPO fica entre duas probabilidades do motor de treino, e a diferença entre os motores é corrigida à parte (TIS, LM head em FP32, R3, kernels determinísticos).' },
  ],
  mount(root, LA) {
    const V = LA.viz, UI = LA.ui;
    const fmt = (x, d = 2) => (+x).toFixed(d).replace('-', '−').replace('.', ',');
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
    /** personagem: quadrado arredondado com olhinhos */
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
    /** moldura de gráfico; devolve mapeamentos X(v), Y(v) */
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

    /* ======================================================= FIG 1 — penhasco */
    (function () {
      const fig = root.querySelector('#fig-penhasco');
      adaptH(fig, 420, 620);
      const TOK = ['“20”', '“14”', '“Vou”'];
      const z0 = [2.0, 1.0, 0.1], p0 = V.softmax(z0), AI = 1, K = 10;
      const S = { A: 1.5, clip: true, eps: 0.2, eta: 0.05 };
      let parts = [];
      /** trajetória completa da razão para um modo (usada como fantasma do modo oposto) */
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
      UI.seg(ctl, { label: 'Vantagem', options: [{ v: 1.5, label: 'Â = +1,5' }, { v: -1.5, label: 'Â = −1,5' }], value: 1.5, onChange: (v) => { S.A = v; reset(); } });
      UI.toggle(ctl, { label: 'clip do PPO', value: true, onChange: (v) => { S.clip = v; reset(); } });
      nocase(UI.slider(ctl, { label: 'ε', min: 0.1, max: 0.3, step: 0.05, value: 0.2, fmt: (v) => fmt(v, 2), onInput: (v) => { S.eps = v; reset(); } }));
      nocase(UI.slider(ctl, { label: 'passo η', min: 0.02, max: 1, step: 0.01, value: 0.05, fmt: (v) => fmt(v, 2), onInput: (v) => { S.eta = v; reset(); } }));
      UI.button(ctl, '↺ recomeçar', () => reset());
      const rR = nocase(UI.readout(ctl, 'ρ do token', '—')), rP = nocase(UI.readout(ctl, 'π(“14”)', '—')), rK = nocase(UI.readout(ctl, 'KL(π_old‖π)', '—'));
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
        // ---- esquerda: logits + barras
        const colW = L.w / 3;
        V.text(g, 'logits z (os parâmetros)', L.x, L.y + 12, { size: 11.5, color: C.muted, weight: 600 });
        const cellW = Math.min(74, colW * 0.62), cellH = 26, cy = L.y + 22;
        const bTop = cy + cellH + 40, bBot = L.y + L.h - 50, bH = bBot - bTop;
        const bw = Math.min(56, colW * 0.46);
        for (let j = 0; j < 3; j++) {
          const cx = L.x + colW * (j + 0.5);
          const zc = V.cmap(V.clamp(S.z[j] / 3.2, -1, 1), 'div');
          V.box(g, cx - cellW / 2, cy, cellW, cellH, { r: 5, fill: zc, stroke: j === AI ? C.d : V.alpha(C.muted, 0.4), lw: j === AI ? 2 : 1 });
          V.text(g, fmt(S.z[j], 2), cx, cy + cellH / 2 + 1, { size: 12, font: 'mono', weight: 600, align: 'center', base: 'middle', color: Math.abs(S.z[j]) > 1.9 ? (C.dark ? '#00210F' : '#fff') : C.ink });
          // barra
          const hOld = p0[j] * bH, hNow = S.show[j] * bH;
          const col = j === AI ? (S.A > 0 ? C.a : C.c) : C.b;
          V.box(g, cx - bw / 2, bBot - hNow, bw, Math.max(1, hNow), { r: 4, fill: V.alpha(col, 0.75), stroke: col, lw: 1.2 });
          g.setLineDash([4, 3]); V.rr(g, cx - bw / 2 - 3, bBot - hOld, bw + 6, hOld, 4); g.strokeStyle = V.alpha(C.ink, 0.55); g.lineWidth = 1.2; g.stroke(); g.setLineDash([]);
          V.text(g, fmt(S.show[j], 3), cx, bBot - hNow - 6, { size: 11.5, font: 'mono', align: 'center', color: C.ink, weight: 600 });
          V.text(g, TOK[j], cx, bBot + 16, { size: 12, align: 'center', color: j === AI ? C.ink : C.muted, weight: j === AI ? 700 : 500 });
          if (j === AI) {
            face(g, cx, bBot + 34, 18, S.A > 0 ? C.a : C.c, { mood: S.A > 0 ? undefined : 'sad' });
            V.text(g, 'amostrado · Â = ' + fmtS(S.A, 1), cx, cy + cellH + 16, { size: 11, font: 'mono', align: 'center', color: S.A > 0 ? C.a : C.c, weight: 600 });
          }
        }
        // partículas de massa migrando
        for (let q = parts.length - 1; q >= 0; q--) {
          const P = parts[q]; P.u += dd * 1.6;
          if (P.u >= 1) { parts.splice(q, 1); continue; }
          if (P.u < 0) continue;
          const x1 = L.x + colW * (P.from + 0.5), x2 = L.x + colW * (P.to + 0.5);
          const y1 = bBot - S.show[P.from] * bH * (0.3 + 0.5 * P.j), y2 = bBot - S.show[P.to] * bH * (0.3 + 0.5 * P.j);
          const u = V.ease.inOut(P.u), x = V.lerp(x1, x2, u), y = V.lerp(y1, y2, u) - Math.sin(u * Math.PI) * 28;
          V.dot(g, x, y, 2.4, V.alpha(S.A > 0 ? C.a : C.c, 0.85));
        }
        // ---- direita: razão por passo
        const ymax = S.A > 0 ? Math.max(2.2, Math.max(...S.hist, ...S.alt) * 1.08) : 1.4;
        const F = frame(g, R, 0, K, 0, ymax, { xt: [0, 2, 4, 6, 8, 10], xf: (v) => String(v), yt: S.A > 0 ? [0, 0.5, 1, 1.5, 2].concat(ymax > 3 ? [3] : [], ymax > 4 ? [4] : []) : [0, 0.5, 1], yf: (v) => fmt(v, 1), xl: 'passos de gradiente no mesmo lote', yl: 'razão ρ do token amostrado' });
        g.fillStyle = V.alpha(C.e, 0.16); g.fillRect(R.x, F.Y(1 + S.eps), R.w, F.Y(1 - S.eps) - F.Y(1 + S.eps));
        V.line(g, R.x, F.Y(1), R.x + R.w, F.Y(1), { color: V.alpha(C.e, 0.7), dash: [4, 4], lw: 1.2 });
        // rótulo da faixa no lado que os passos não visitam: abaixo dela com Â > 0, acima com Â < 0
        V.text(g, 'faixa [1−ε, 1+ε]', R.x + R.w - 4, S.A > 0 ? F.Y(1 - S.eps) + 14 : F.Y(1 + S.eps) - 5, { size: 11, color: C.e, align: 'right', font: 'mono' });
        // fantasma: o que o outro modo faria com o mesmo lote
        g.beginPath();
        S.alt.forEach((r, k) => { const x = F.X(k), y = F.Y(Math.min(ymax, r)); if (k) g.lineTo(x, y); else g.moveTo(x, y); });
        g.strokeStyle = V.alpha(S.clip ? RED() : C.d, 0.55); g.lineWidth = 1.6; g.setLineDash([5, 4]); g.stroke(); g.setLineDash([]);
        const ae = S.alt[K];
        let gy = F.Y(Math.min(ymax, ae)) + (ae > S.hist[S.hist.length - 1] ? -8 : 16);
        if (gy < R.y + 28) gy = F.Y(Math.min(ymax, ae)) + 16; // não colide com a mensagem do topo
        V.text(g, S.clip ? 'sem clip' : 'com clip', F.X(K) - 4, gy, { size: 11, font: 'mono', color: S.clip ? RED() : C.d, align: 'right' });
        g.beginPath();
        S.hist.forEach((r, k) => { const x = F.X(k), y = F.Y(Math.min(ymax, r)); if (k) g.lineTo(x, y); else g.moveTo(x, y); });
        g.strokeStyle = S.A > 0 ? C.a : C.c; g.lineWidth = 2.2; g.stroke();
        S.hist.forEach((r, k) => {
          const x = F.X(k), y = F.Y(Math.min(ymax, r));
          if (S.blocked[k]) { g.beginPath(); g.arc(x, y, 4.5, 0, Math.PI * 2); g.strokeStyle = C.d; g.lineWidth = 2; g.stroke(); }
          else V.dot(g, x, y, 3.6, S.A > 0 ? C.a : C.c);
        });
        const nb = S.blocked.filter(Boolean).length;
        const msg = !S.clip ? 'sem clip: o substituto segue empurrando' : nb ? '✂ incentivo zerado: ' + nb + ' passo' + (nb > 1 ? 's' : '') + ' sem efeito' : 'dentro da faixa: o gradiente ainda age';
        V.text(g, msg, R.x + 6, R.y + 12, { size: 11.5, color: !S.clip ? RED() : nb ? C.d : C.muted, weight: 600, maxW: R.w - 10 });
      });
    })();

    /* ======================================================= FIG 2 — função clip */
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
      nocase(UI.slider(ctl, { label: 'vantagem Â', min: -2, max: 2, step: 0.5, value: 2, fmt: (v) => fmtS(v, 1), onInput: (v) => { S.A = v; S.rho = start(); S.hold = 0; upd(); } }));
      nocase(UI.slider(ctl, { label: 'ε', min: 0.05, max: 0.4, step: 0.05, value: 0.2, fmt: (v) => fmt(v, 2), onInput: (v) => { S.eps = v; upd(); } }));
      sR = nocase(UI.slider(ctl, { label: 'ρ', min: X0, max: X1, step: 0.01, value: S.rho, fmt: (v) => fmt(v, 2), onInput: (v) => { S.rho = v; S.userT = H2 ? H2.t : 0; S.hold = 0; upd(); } }));
      UI.toggle(ctl, { label: 'subida de gradiente', value: true, onChange: (v) => { S.run = v; if (v) { S.rho = start(); S.hold = 0; } } });
      const rL = nocase(UI.readout(ctl, 'L^CLIP', '—')), rD = nocase(UI.readout(ctl, '∂L/∂ρ', '—')), rG = UI.readout(ctl, 'regime', '—');
      let lastShown = -1;
      function upd() {
        const d = dL(S.rho);
        rL.set(fmt(Lf(S.rho), 3));
        rD.set(fmt(d, 1), d === 0 ? 'warn' : 'ok');
        rG.set(S.A === 0 ? 'Â = 0: nada a aprender' : d === 0 ? 'plano: incentivo cortado' : 'inclinado: gradiente ativo', d === 0 ? 'warn' : 'ok');
      }
      upd();
      H2 = V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const dd = clock(S, t);
        const A = { x: 52, y: 44, w: w - 72, h: h - 90 };
        const ys = [Lf(X0), Lf(X1), X0 * S.A, X1 * S.A, 0];
        let y0 = Math.min(...ys), y1 = Math.max(...ys);
        const pad = Math.max(0.3, (y1 - y0) * 0.12); y0 -= pad; y1 += pad;
        // ponteiro: arrastar
        const P = H2.pointer;
        if (P.down && P.x >= A.x && P.x <= A.x + A.w) { S.rho = V.clamp(X0 + ((P.x - A.x) / A.w) * (X1 - X0), X0, X1); S.userT = t; S.hold = 0; }
        // subida de gradiente
        if (S.run && t - S.userT > 2.5) {
          const d = dL(S.rho);
          if (d !== 0) { S.rho = V.clamp(S.rho + Math.sign(d) * 0.32 * dd, X0, X1); if (S.rho <= X0 || S.rho >= X1) S.hold += dd; }
          else S.hold += dd;
          if (S.hold > 1.8) { S.rho = start(); S.hold = 0; }
        }
        if (Math.abs(S.rho - lastShown) > 0.004) { lastShown = S.rho; sR.set(S.rho.toFixed(2), true); upd(); }
        const step = (y1 - y0) > 3 ? 1 : 0.5;
        const yt = []; for (let v = Math.ceil(y0 / step) * step; v <= y1; v += step) yt.push(+v.toFixed(2));
        const F = frame(g, A, X0, X1, y0, y1, { xt: [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8], yt, xl: 'razão ρ = π_θ / π_θold', yl: 'objetivo do token', ylOff: 28 });
        // zonas de gradiente zero
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
        // pontos da tabela
        if (S.A === 2 || S.A === -1) [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3].forEach((r) => V.dot(g, F.X(r), F.Y(Lf(r)), 2.6, V.alpha(C.ink, 0.7)));
        // bolinha
        const bx = F.X(S.rho), by = F.Y(Lf(S.rho)), d = dL(S.rho);
        V.glow(g, bx, by, 16, d === 0 ? C.d : main, 0.8);
        V.dot(g, bx, by, 6, d === 0 ? C.d : main);
        if (d !== 0) V.arrow(g, bx, by - 14, bx + Math.sign(d) * 34, by - 14, { color: main, lw: 2.2, head: 8 });
        else V.text(g, '∇ = 0', bx, by - 14, { size: 12, font: 'mono', weight: 700, color: C.d, align: 'center' });
        // legenda
        const lx = A.x + 10, ly = S.A >= 0 ? A.y + 14 : A.y + A.h - 46;
        [[V.alpha(C.ink, 0.6), 'ρÂ (sem clip)', [6, 5]], [C.b, 'clip(ρ)·Â', [2, 4]], [main, 'L^CLIP = min(…)', null]].forEach(([c, s, ds], k) => {
          V.line(g, lx, ly + k * 16, lx + 22, ly + k * 16, { color: c, lw: ds ? 2 : 3.2, dash: ds || undefined });
          V.text(g, s, lx + 28, ly + k * 16 + 4, { size: 11, font: 'mono', color: C.ink });
        });
        if (S.A !== 0) V.text(g, 'hachurado = gradiente zero', A.x + A.w - 4, S.A >= 0 ? A.y + A.h - 8 : A.y + 14, { size: 11, color: C.d, align: 'right' });
      });
    })();

    /* ======================================================= FIG 3 — clipping do valor */
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
      /** bissecção: ponto entre a e b em que o status de gradiente zero muda */
      const edge = (a, b) => { const za = isZero(a); for (let q = 0; q < 18; q++) { const m = (a + b) / 2; if (isZero(m) === za) a = m; else b = m; } return (a + b) / 2; };
      let sV, sRh;
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Caso', options: [{ v: 'c1', label: 'V 1,6 · alvo 1,1' }, { v: 'c2', label: 'V 0,7 · alvo 1,1' }, { v: 'c3', label: 'V 1,3 · alvo 2,0' }], value: 'c3', onChange: (v) => { const c = CASES[v]; S.Vn = c.V; S.R = c.R; S.ball = S.Vold; S.hold = 0; sV.set(c.V, true); sRh.set(c.R, true); upd(); } });
      sV = nocase(UI.slider(ctl, { label: 'V novo (losango)', min: 0.2, max: 2.4, step: 0.01, value: 1.3, fmt: (v) => fmt(v, 2), onInput: (v) => { S.Vn = v; upd(); } }));
      sRh = nocase(UI.slider(ctl, { label: 'alvo R̂', min: 0.2, max: 2.4, step: 0.05, value: 2.0, fmt: (v) => fmt(v, 2), onInput: (v) => { S.R = v; S.ball = S.Vold; S.hold = 0; upd(); } }));
      nocase(UI.slider(ctl, { label: 'ε̂', min: 0.1, max: 0.5, step: 0.05, value: 0.2, fmt: (v) => fmt(v, 2), onInput: (v) => { S.eps = v; S.ball = S.Vold; S.hold = 0; upd(); } }));
      UI.toggle(ctl, { label: 'descida de gradiente a partir de V_old', value: true, onChange: (v) => { S.run = v; S.ball = S.Vold; S.hold = 0; } });
      const rL = nocase(UI.readout(ctl, 'perda em V novo', '—')), rD = nocase(UI.readout(ctl, '∂L/∂V em V novo', '—')), rG = UI.readout(ctl, 'termo escolhido', '—');
      function upd() {
        const d = dL(S.Vn);
        rL.set(fmt(Lv(S.Vn), 3));
        rD.set(fmt(d, 2), d === 0 ? 'warn' : 'ok');
        const inBand = Math.abs(S.Vn - S.Vold) <= S.eps + 1e-9;
        rG.set(inBand ? 'dentro da faixa (iguais)' : d === 0 ? 'clipado: gradiente zero' : 'sem clip: puxa de volta', d === 0 ? 'warn' : 'ok');
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
        const F = frame(g, A, X0, X1, 0, ymax, { xt: [0, 0.5, 1, 1.5, 2, 2.5], yt: ymax > 1.6 ? [0, 0.5, 1, 1.5, 2] : ymax > 0.9 ? [0, 0.25, 0.5, 0.75, 1] : [0, 0.25, 0.5, 0.75], yf: (v) => fmt(v, 2), xl: 'previsão do crítico V(s_t)', yl: 'perda de valor', ylOff: 28 });
        g.save(); g.beginPath(); g.rect(A.x, A.y - 2, A.w, A.h + 2); g.clip();
        g.fillStyle = V.alpha(C.e, 0.14); g.fillRect(F.X(S.Vold - S.eps), A.y, F.X(S.Vold + S.eps) - F.X(S.Vold - S.eps), A.h);
        // trechos em que o max escolheu o termo clipado (gradiente zero)
        // varre uma vez, acha os intervalos contíguos (no máximo 2) e hachura cada um de uma vez só
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
        // alvo
        const rx = F.X(S.R), ry = A.y + A.h;
        g.beginPath(); for (let q = 0; q < 10; q++) { const rr = q % 2 ? 3.5 : 8, a = -Math.PI / 2 + (q * Math.PI) / 5; const px = rx + rr * Math.cos(a), py = ry - 12 + rr * Math.sin(a); if (q) g.lineTo(px, py); else g.moveTo(px, py); } g.closePath(); g.fillStyle = C.d; g.fill();
        V.text(g, 'alvo R̂', rx, ry - 24, { size: 11, font: 'mono', color: C.d, align: 'center', weight: 600 });
        // losango: V novo do caso
        const mx = F.X(S.Vn), my = F.Y(Math.min(ymax, Lv(S.Vn)));
        g.beginPath(); g.moveTo(mx, my - 7); g.lineTo(mx + 7, my); g.lineTo(mx, my + 7); g.lineTo(mx - 7, my); g.closePath(); g.fillStyle = C.b; g.fill(); g.strokeStyle = C.ink; g.lineWidth = 1; g.stroke();
        // rótulo do losango do lado oposto ao ramo da parábola (senão a curva corta o texto)
        const nearR = mx > A.x + A.w - 90, lblL = nearR || (S.Vn < S.R && dL(S.Vn) !== 0 && mx > 100);
        V.text(g, 'V novo: ' + fmt(Lv(S.Vn), 2), lblL ? mx - 10 : mx + 10, my + (nearR ? -12 : 16), { size: 11, font: 'mono', color: C.b, weight: 700, align: lblL ? 'right' : 'left' });
        // bolinha: o crítico ao longo das épocas
        const d = dL(S.ball);
        const bx = F.X(S.ball), by = F.Y(Math.min(ymax, Lv(S.ball)));
        V.glow(g, bx, by, 15, d === 0 ? C.d : C.c, 0.8);
        V.dot(g, bx, by, 6, d === 0 ? C.d : C.c);
        if (Math.abs(d) > 0.004) V.arrow(g, bx, by - 14, bx - Math.sign(d) * 30, by - 14, { color: C.c, lw: 2.2, head: 8 });
        else { const lft = bx < A.x + 150; V.text(g, d === 0 ? '∇ = 0: parou na borda' : 'mínimo', lft ? bx + 8 : bx - 6, by - 16, { size: 11.5, font: 'mono', weight: 700, color: C.d, align: lft ? 'left' : 'right' }); }
        // legenda: escolhe o canto em que menos pontos das curvas e marcadores caem por baixo dela
        const LEG = [[V.alpha(C.ink, 0.6), '(V − R̂)²', [6, 5]], [C.b, 'termo clipado', [2, 4]], [C.c, w < 560 ? 'max(…) = perda' : 'max(…) = perda usada', null]];
        g.font = V.font(11, 'mono', 400);
        const legW = 28 + Math.max(g.measureText('gradiente zero').width, ...LEG.map((it) => g.measureText(it[1]).width)) + 14, legH = 68;
        const pts = [[mx, my], [mx + 10, my + 14], [rx, ry - 12], [rx, ry - 24]];
        for (let k = 0; k <= 60; k++) { const v = X0 + ((X1 - X0) * k) / 60; [Lun(v), Lcl(v), Lv(v)].forEach((L) => { if (L <= ymax) pts.push([F.X(v), F.Y(L)]); }); }
        const cands = [[A.x + 10, A.y + 14], [A.x + A.w - legW, A.y + 14], [A.x + (A.w - legW) / 2 + 6, A.y + 14], [A.x + 10, A.y + A.h - legH + 2], [A.x + A.w - legW, A.y + A.h - legH + 2], [A.x + 10, A.y + (A.h - legH) / 2 + 10], [A.x + A.w - legW, A.y + (A.h - legH) / 2 + 10], [A.x + (A.w - legW) / 2 + 6, A.y + (A.h - legH) / 2 + 10]];
        for (let k = 0; k <= 120; k++) { const v = X0 + ((X1 - X0) * k) / 120, L = Lv(v); if (L <= ymax) pts.push([F.X(v), F.Y(L)]); } // linha grossa pesa em dobro
        const hits = cands.map(([cx0, cy0]) => pts.filter(([px, py]) => px >= cx0 - 12 && px <= cx0 + legW && py >= cy0 - 16 && py <= cy0 + legH - 4).length);
        let bi = 0; hits.forEach((n, k) => { if (n < hits[bi]) bi = k; });
        if (S.leg != null && hits[S.leg] <= hits[bi]) bi = S.leg; // sem pular de canto à toa
        S.leg = bi;
        const lx = cands[bi][0], ly = cands[bi][1];
        V.box(g, lx - 6, ly - 10, legW, legH, { r: 6, fill: V.alpha(C.bg || C.panel, 0.92), stroke: V.alpha(C.muted, 0.25), lw: 1 });
        LEG.forEach(([c, s, ds], k) => {
          V.line(g, lx, ly + k * 16, lx + 22, ly + k * 16, { color: c, lw: ds ? 2 : 3, dash: ds || undefined });
          V.text(g, s, lx + 28, ly + k * 16 + 4, { size: 11, font: 'mono', color: C.ink });
        });
        hatch(g, lx, ly + 41, 22, 9, V.alpha(C.d, 0.6), 4);
        V.text(g, 'gradiente zero', lx + 28, ly + 50, { size: 11, font: 'mono', color: C.d });
      });
    })();

    /* ======================================================= FIG 4 — pipeline InstructGPT */
    (function () {
      const fig = root.querySelector('#fig-pipeline');
      adaptH(fig, 400, 700);
      const steps = [
        { t: 'Etapa 1 · SFT em demonstrações', d: 'Cerca de 40 rotuladores escrevem respostas-modelo. São 12.725 prompts de treino (11.295 de rotuladores + 1.430 de clientes da API). 16 épocas com cosseno; o checkpoint é escolhido pelo <b>escore do RM</b>, porque a perda de validação sobreajusta após 1 época.' },
        { t: 'Etapa 2 · rankings viram pares', d: 'Para cada prompt, \\(K = 4\\) a \\(9\\) respostas são ordenadas por um rotulador. Cada ranking vira \\(\\binom K2\\) pares, de 6 a 36. São 33.207 prompts de treino. Empates são descartados.' },
        { t: 'Etapa 2 · o RM aprende Bradley–Terry', d: 'RM de 6B com perda \\(-\\log\\sigma(r_w - r_l)\\), média sobre os pares. Todos os pares de um prompt entram no <b>mesmo</b> elemento do batch: \\(K\\) forwards, não \\(\\binom K2\\), e nada de sobreajuste por reusar cada resposta \\(K-1\\) vezes. No exemplo (ilustrativo), escores 2,0 / 1,0 / 0,5 / −0,5 dão perda 0,2637.' },
        { t: 'Etapa 3 · PPO contra o RM', d: '31.144 prompts só da API, sem rótulo humano. A política (iniciada do SFT) gera, o RM dá a nota no fim, a KL por token até o SFT vira penalidade (\\(\\beta = 0{,}02\\)), e o crítico, iniciado do RM, estima valores. 256k episódios, batch 512, 1 época.' },
        { t: 'PPO-ptx · a âncora no pré-treino', d: 'Um gradiente extra de log-verossimilhança em texto de pré-treino, com peso \\(\\gamma = 27{,}8\\) e 8× mais exemplos que episódios de RL. Sem ele, SQuADv2, DROP e tradução regrediam (o imposto de alinhamento).' },
        { t: 'Resultado', d: 'As saídas do InstructGPT 1,3B foram preferidas às do GPT-3 175B. O PPO-ptx do 175B custou 60 petaflop/s-dias, contra 3.640 do pré-treino do GPT-3.' },
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
        // setas entre estações
        for (let k = 0; k < 2; k++) {
          const a = Rr[k], b = Rr[k + 1];
          if (wide) V.arrow(g, a.x + a.w + 3, a.y + a.h / 2, b.x - 3, b.y + b.h / 2, { color: V.alpha(C.muted, 0.7), lw: 1.8 });
          else V.arrow(g, a.x + a.w / 2, a.y + a.h + 2, b.x + b.w / 2, b.y - 2, { color: V.alpha(C.muted, 0.7), lw: 1.8 });
        }
        // ---- estação 1: SFT
        (function (R) {
          station(g, R, '1 · SFT', '12.725 prompts · 16 épocas', on[0] || i === 5, C);
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
          V.text(g, 'modelo base', mx, my + ms / 2 + 14, { size: 11, align: 'center', color: C.muted });
          V.text(g, 'demonstrações', R.x + 16, my - 36, { size: 11, color: C.muted });
          g.globalAlpha = 1;
        })(Rr[0]);
        // ---- estação 2: RM
        (function (R) {
          const act = on[1] || i === 5;
          station(g, R, '2 · Modelo de recompensa', '33.207 prompts · K = 4–9 → C(K,2) pares', act, C);
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
          // pares
          const pairs = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]];
          const nP = i === 1 ? Math.floor(V.clamp((el - 1.0) / 0.3, 0, 6)) : i >= 2 ? 6 : 0;
          const px = R.x + R.w * (wide ? 0.44 : 0.4), pStep = Math.min(16, (blockH + 10) / 6);
          const py0 = y0 + blockH / 2 - 3 * pStep;
          for (let q = 0; q < nP; q++) {
            const [a, b] = pairs[q];
            const y = py0 + q * pStep;
            chip(g, px, y, 40, pStep - 3, labels[a] + '≻' + labels[b], { size: 11, fill: V.alpha(C.a, 0.14), stroke: V.alpha(C.a, 0.6), r: 4 });
          }
          if (nP) V.text(g, nP + (nP > 1 ? ' pares' : ' par'), px, py0 - 6, { size: 11, color: C.muted });
          const mx = R.x + R.w - Math.min(46, R.h * 0.3) / 2 - 16, my = R.y + R.h * 0.5, ms = Math.min(42, R.h * 0.28);
          face(g, mx, my, ms, C.d, { mood: i === 2 ? 'o' : undefined });
          V.text(g, 'RM 6B', mx, my + ms / 2 + 14, { size: 11, align: 'center', color: C.muted });
          if (i === 2) {
            const ph = (t * 0.8) % 1;
            V.dot(g, V.lerp(px + 40, mx - ms / 2, ph), my, 3, C.d);
            V.text(g, 'perda 0,2637', mx - 8, my + ms / 2 + 29, { size: 11, font: 'mono', color: C.d, align: 'center', weight: 700 });
          }
          g.globalAlpha = 1;
        })(Rr[1]);
        // ---- estação 3: PPO
        (function (R) {
          const act = on[2] || i === 5;
          station(g, R, '3 · PPO', '31.144 prompts · 256k episódios · β = 0,02', act, C);
          g.globalAlpha = act ? 1 : 0.38;
          const ms = Math.min(40, R.h * 0.26);
          const pxx = R.x + R.w * 0.3, pyy = R.y + R.h * (wide ? 0.42 : 0.46);
          const rxx = R.x + R.w * 0.78, ryy = pyy;
          // tokens viajando da política para o RM
          for (let q = 0; q < 4; q++) {
            const ph = (t * 0.5 + q / 4) % 1;
            chip(g, V.lerp(pxx + ms / 2 + 4, rxx - ms / 2 - 18, ph), pyy - 26, 14, 12, null, { fill: V.alpha(C.a, 0.6), stroke: C.a, r: 3 });
          }
          // nota voltando
          const ph2 = (t * 0.5) % 1;
          V.text(g, '+r', V.lerp(rxx - ms / 2, pxx + ms / 2, ph2), pyy + 24, { size: 12, font: 'mono', weight: 700, color: C.d, align: 'center' });
          face(g, pxx, pyy, ms, C.a, { mood: 'o' });
          V.text(g, 'política', pxx, pyy - ms / 2 - 8, { size: 11, align: 'center', color: C.ink, weight: 600 });
          face(g, rxx, ryy, ms * 0.9, C.d);
          V.text(g, 'RM', rxx, ryy - ms / 2 - 8, { size: 11, align: 'center', color: C.ink, weight: 600 });
          // referência fantasma + coleira
          const fx = pxx - Math.min(60, R.w * 0.18), fy = pyy + ms * 1.25;
          face(g, fx, fy, ms * 0.8, C.b, { fill: V.alpha(C.b, 0.12), ghost: true, dash: true });
          V.text(g, 'SFT (ref)', fx, fy + ms * 0.4 + 13, { size: 11, align: 'center', color: C.b });
          g.beginPath();
          for (let k = 0; k <= 20; k++) { const u = k / 20; const x = V.lerp(fx + ms * 0.3, pxx - ms * 0.2, u), y = V.lerp(fy - ms * 0.2, pyy + ms * 0.45, u) + Math.sin(u * Math.PI * 4 + t * 3) * 3; if (k) g.lineTo(x, y); else g.moveTo(x, y); }
          g.strokeStyle = C.b; g.lineWidth = 1.6; g.stroke();
          V.text(g, 'KL', (fx + pxx) / 2 + 12, (fy + pyy) / 2 + 12, { size: 11, font: 'mono', color: C.b, weight: 700 });
          // crítico
          const cx = rxx, cy = fy;
          face(g, cx, cy, ms * 0.7, C.c);
          V.text(g, 'crítico V (do RM)', cx, cy + ms * 0.35 + 13, { size: 11, align: 'center', color: C.c, maxW: R.w * 0.45 });
          // ptx
          if (i === 4) {
            for (let q = 0; q < 5; q++) {
              const ph = (t * 0.45 + books[q]) % 1;
              const x = V.lerp(R.x + 8, pxx - ms / 2 - 6, ph), y = pyy - 8 + Math.sin(q * 2.1) * 10;
              chip(g, x, y, 12, 9, null, { fill: V.alpha(C.e, 0.7), stroke: C.e, r: 2 });
            }
            V.text(g, 'ptx · γ = 27,8', R.x + 12, R.y + 40, { size: 11.5, font: 'mono', weight: 700, color: C.e });
          }
          g.globalAlpha = 1;
        })(Rr[2]);
        // ---- resultado
        if (i === 5) {
          const a = V.ease.out(V.clamp(el / 0.8, 0, 1));
          const cw = Math.min(480, w - 20), chh = Math.min(190, h * 0.5);
          const x = (w - cw) / 2, y = (h - chh) / 2;
          g.globalAlpha = a;
          // fundo opaco: a estação de baixo não aparece através do cartão
          V.box(g, x, y, cw, chh, { r: 12, fill: C.bg, stroke: C.a, lw: 2 });
          V.box(g, x, y, cw, chh, { r: 12, fill: V.alpha(C.a, 0.06) });
          V.text(g, 'Resultado', x + 14, y + 22, { size: 13, weight: 700, color: C.ink });
          const rows = [
            ['InstructGPT 1,3B', 1.3 / 175, C.a, 'preferido ✓'],
            ['GPT-3 175B', 1, C.muted, '100× maior'],
            ['PPO-ptx 175B', 60 / 3640, C.c, '60 pf/s-dias'],
            ['pré-treino GPT-3', 1, C.muted, '3.640 pf/s-dias'],
          ];
          // mede rótulos e tags para que nada encoste: rótulo | barra | tag
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
          // separa as duas comparações (qualidade | custo)
          const sy = y + 40 + 2 * stepY - 4;
          V.line(g, x + 14, sy, x + cw - 14, sy, { color: V.alpha(C.muted, 0.3), lw: 1, dash: [3, 4] });
          g.globalAlpha = 1;
        }
      });
    })();

    /* ======================================================= FIG 5 — política ótima */
    (function () {
      const fig = root.querySelector('#fig-otima');
      adaptH(fig, 430, 640);
      const LAB = ['útil e correta', 'correta, seca', 'prolixa', 'errada, confiante', 'recusa', 'exploit'];
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
      tog = UI.toggle(ctl, { label: 'varrer β sozinho', value: true, onChange: (v) => { S.auto = v; } });
      const rEr = UI.readout(ctl, 'nota do RM', '—'), rEg = UI.readout(ctl, 'qualidade real', '—'), rKL = nocase(UI.readout(ctl, 'KL(π*‖π_ref)', '—')), rX = UI.readout(ctl, 'massa no exploit', '—');
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
        // ---- barras
        V.text(g, 'π_ref (contorno) e π* ∝ π_ref·e^(r/β) (barras) · β = ' + fmt(Math.pow(10, S.lb), 2), L.x, L.y - 10, { size: 11.5, color: C.muted, weight: 600, maxW: L.w });
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
        // ---- curva proxy x real
        const xmax = Math.ceil(GRID[0].x * 10) / 10 + 0.1;
        const F = frame(g, Rg, 0, xmax, -1.6, 2.6, { xt: [0, 0.5, 1, 1.5, 2], yt: [-1, 0, 1, 2], yf: (v) => fmt(v, 0), xl: '√KL(π*‖π_ref)', yl: 'média sob π*' });
        V.line(g, Rg.x, F.Y(0), Rg.x + Rg.w, F.Y(0), { color: V.alpha(C.muted, 0.4), lw: 1 });
        const path = (key, col) => { g.beginPath(); GRID.forEach((p, k) => { const x = F.X(p.x), y = F.Y(p[key]); if (k) g.lineTo(x, y); else g.moveTo(x, y); }); g.strokeStyle = col; g.lineWidth = 2.6; g.stroke(); };
        path('Er', C.b); path('Eg', C.a);
        const cx = F.X(Math.sqrt(Math.max(0, s.KL)));
        V.line(g, cx, Rg.y, cx, Rg.y + Rg.h, { color: V.alpha(C.ink, 0.35), dash: [3, 4], lw: 1 });
        V.glow(g, cx, F.Y(s.Er), 12, C.b, 0.7); V.dot(g, cx, F.Y(s.Er), 5, C.b);
        V.glow(g, cx, F.Y(s.Eg), 12, C.a, 0.7); V.dot(g, cx, F.Y(s.Eg), 5, C.a);
        const pxp = F.X(peak.x), pyp = F.Y(peak.Eg);
        V.text(g, '★ pico real', pxp + 4, pyp - 8, { size: 11, color: C.a, weight: 700 });
        [[C.b, 'nota do RM (proxy)'], [C.a, 'qualidade real']].forEach(([c, s], k) => {
          const yy = Rg.y + 12 + k * 16;
          V.line(g, Rg.x + 8, yy - 4, Rg.x + 28, yy - 4, { color: c, lw: 2.6 });
          V.text(g, s, Rg.x + 34, yy, { size: 11, color: c, weight: 600 });
        });
        V.text(g, 'β: grande → pequeno', Rg.x + Rg.w, Rg.y - 9, { size: 11, color: C.muted, align: 'right' });
      });
    })();

    /* ======================================================= FIG 6 — controle de β */
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
      UI.seg(ctl, { label: 'Controle', options: [{ v: 'fixo', label: 'β fixo' }, { v: 'zieg', label: 'Ziegler (Kβ = 0,1)' }, { v: 'ppo', label: 'dobra / divide' }], value: 'zieg', onChange: (v) => { S.mode = v; simulate(); } });
      nocase(UI.slider(ctl, { label: 'KL alvo (nats)', min: 3, max: 10, step: 0.5, value: 6, fmt: (v) => fmt(v, 1), onInput: (v) => { S.tgt = v; simulate(); } }));
      nocase(UI.slider(ctl, { label: 'β inicial', min: 0.02, max: 0.3, step: 0.01, value: 0.1, fmt: (v) => fmt(v, 2), onInput: (v) => { S.b0 = v; simulate(); } }));
      const rK = UI.readout(ctl, 'KL final (3 sementes)', '—'), rB = nocase(UI.readout(ctl, 'β final', '—'));
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
        // altura útil deixa 42 px embaixo: ticks em +14 e o rótulo 'iterações' em +30 cabem no canvas
        const A1 = { x: 52, y: 24, w: w - 70, h: (h - 110) * 0.55 };
        const A2 = { x: 52, y: A1.y + A1.h + 44, w: w - 70, h: (h - 110) * 0.45 };
        const kmax = Math.max(16, ...SIM.map((s) => Math.max(...s.KL))) * 1.05;
        const F1 = frame(g, A1, 0, NIT, 0, kmax, { xt: [0, 100, 200, 300], xf: (v) => String(v), yt: kmax > 30 ? [0, 10, 20, 30] : [0, 5, 10, 15], yf: (v) => String(v), yl: 'KL(π‖π_ref) em nats' });
        V.line(g, A1.x, F1.Y(S.tgt), A1.x + A1.w, F1.Y(S.tgt), { color: C.d, dash: [6, 5], lw: 1.6 });
        V.text(g, 'alvo ' + fmt(S.tgt, 1), A1.x + A1.w - 4, F1.Y(S.tgt) - 5, { size: 11, font: 'mono', color: C.d, align: 'right' });
        const lmin = -3, lmax = 1;
        const F2 = frame(g, A2, 0, NIT, lmin, lmax, { xt: [0, 100, 200, 300], xf: (v) => String(v), yt: [-3, -2, -1, 0, 1], yf: (v) => (v === -3 ? '0,001' : v === -2 ? '0,01' : v === -1 ? '0,1' : v === 0 ? '1' : '10'), xl: 'iterações', yl: 'β (escala log)' });
        if (S.mode === 'ppo') { g.fillStyle = V.alpha(C.d, 0.08); g.fillRect(A1.x, F1.Y(1.5 * S.tgt), A1.w, F1.Y(S.tgt / 1.5) - F1.Y(1.5 * S.tgt)); V.text(g, 'zona morta [alvo/1,5; 1,5·alvo]', A1.x + A1.w - 4, F1.Y(1.5 * S.tgt) - 5, { size: 11, color: C.d, align: 'right' }); }
        SIM.forEach((s, k) => {
          // fantasma da trajetória inteira
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
        const names = ['semente 1 · puxa fraco', 'semente 2 · médio', 'semente 3 · puxa forte'];
        names.forEach((s, k) => { V.dot(g, A1.x + 12, A1.y + 10 + k * 15, 4, cols[k]); V.text(g, s, A1.x + 22, A1.y + 14 + k * 15, { size: 11, color: C.ink }); });
      });
    })();

    /* ======================================================= FIG 7 — iteração passo a passo */
    (function () {
      const fig = root.querySelector('#fig-iteracao');
      adaptH(fig, 540, 680);
      const TOKS = ['O', 'céu', 'é', 'azul', '<eos>'];
      const ROWS = [
        { l: 'log π_old', v: [-1.20, -0.50, -2.10, -0.30, -0.05], d: 2, from: 3, sc: 2.2 },
        { l: 'log π_ref', v: [-1.40, -0.55, -1.60, -0.45, -0.10], d: 2, from: 3, sc: 2.2 },
        { l: 'V_old', v: [0.90, 1.10, 1.00, 1.50, 1.80], d: 2, from: 3, sc: 2.2 },
        { l: 'KL_t (k₁)', v: [0.20, 0.05, -0.50, 0.15, 0.05], d: 2, from: 4, sc: 0.6 },
        { l: 'r_t', v: [-0.0100, -0.0025, 0.0250, -0.0075, 1.9975], d: 4, from: 4, sc: 2.0 },
        { l: 'δ_t', v: [0.1900, -0.1025, 0.5250, 0.2925, 0.1975], d: 4, from: 5, sc: 0.8 },
        { l: 'Â_t (λ=0,95)', v: [0.9781, 0.8296, 0.9811, 0.4801, 0.1975], d: 4, from: 5, sc: 1.2, rtl: true },
        { l: 'R̂_t (alvo)', v: [1.8781, 1.9296, 1.9811, 1.9801, 1.9975], d: 4, from: 5, sc: 2.2, rtl: true },
        { l: 'Â branqueada', v: [0.93, 0.44, 0.94, -0.69, -1.61], d: 2, from: 6, sc: 1.8 },
        { l: 'ρ_t (época)', v: [1.214, 1.187, 1.226, 0.806, 0.781], d: 3, from: 7, sc: 0.3, rho: true },
      ];
      const ACT = [[], ['motor', 'pol'], ['rm'], ['ref', 'pol', 'crit'], [], ['crit'], [], ['pol', 'crit'], ['pol', 'motor']];
      const steps = [
        { t: 'Prompts', d: 'Sorteie \\(B\\) prompts do conjunto de RL (InstructGPT e N+: \\(B = 512\\)). A figura acompanha <b>uma</b> resposta de 5 tokens.' },
        { t: 'Rollouts no motor de inferência', d: 'O vLLM ou o SGLang gera com os pesos de \\(\\pi_{\\theta_{\\text{old}}}\\), com batching contínuo e KV cache (temperatura 1 no InstructGPT, 0,7 no N+). Truque do N+: gerar até o máximo e truncar no EOS.' },
        { t: 'O RM dá a nota no EOS', d: 'Um forward do RM; a nota é lida só no token EOS: 2,0 neste exemplo. Uma resposta sem EOS recebe −1 (N+).' },
        { t: 'Três forwards por resposta', d: '\\(\\log\\pi_{\\text{ref}}\\) (referência congelada), \\(\\log\\pi_{\\text{old}}\\) (recalculado no motor de <b>treino</b>, não copiado do vLLM) e \\(V_{\\text{old}}\\) (crítico). Um forward devolve log-probs ou valores de <b>todas</b> as posições de uma vez, porque a resposta inteira já é conhecida (teacher forcing).' },
        { t: 'Recompensa por token', d: '\\(r_t = -\\beta(\\log\\pi_{\\text{old}} - \\log\\pi_{\\text{ref}})\\), e o último token ganha também a nota do RM. Com \\(\\beta = 0{,}05\\): de −0,0100 a +1,9975. O token 3 tem KL negativa (o estimador \\(k_1\\) pode ser negativo) e ganha um prêmio.' },
        { t: 'GAE de trás para frente', d: '\\(\\delta_t = r_t + V(s_{t+1}) - V(s_t)\\) e \\(\\hat A_t = \\delta_t + \\gamma\\lambda\\hat A_{t+1}\\), com \\(\\gamma = 1\\) e \\(\\lambda = 0{,}95\\), varrendo da direita para a esquerda. Alvo do crítico: \\(\\hat R_t = \\hat A_t + V_{\\text{old}}\\).' },
        { t: 'Branquear as vantagens', d: 'Média 0 e desvio 1 no lote (aqui, nos 5 tokens: média 0,693 e desvio 0,308). Sobra o sinal relativo: os tokens 4 e 5 foram “piores que os outros”. O EOS fica negativo porque o crítico já esperava 1,8: a surpresa \\(\\delta_5 = 0{,}1975\\) é pequena. Num lote real, a média e o desvio vêm de todos os tokens das 512 respostas.' },
        { t: 'K épocas × minibatches', d: 'Em cada minibatch: \\(\\rho_t\\), \\(L^{CLIP}\\), perda de valor clipada e um passo do otimizador com clip de norma. A razão parte de exatamente 1,000; tokens com vantagem positiva sobem até perto de 1,2, os negativos descem até perto de 0,8, e o clip corta o resto (movimento ilustrativo).' },
        { t: 'Sincronizar e recomeçar', d: 'Os pesos novos vão para o motor de inferência e \\(\\theta_{\\text{old}}\\leftarrow\\theta\\). Uma iteração do InstructGPT tem 512 episódios; 256k episódios são 500 iterações.' },
      ];
      let H = null;
      const st = UI.stepper(fig, steps, { autoplay: 6000, onStep: () => H && H.redraw() });
      H = V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const i = st.i, el = st.elapsed();
        const narrow = w < 560;
        // ---- personagens
        const ACTORS = [
          { k: 'motor', l: narrow ? 'vLLM' : 'motor (vLLM)', c: C.e },
          { k: 'pol', l: narrow ? 'π_θ' : 'política π_θ', c: C.a },
          { k: 'ref', l: narrow ? 'π_ref' : 'referência π_ref', c: C.b },
          { k: 'rm', l: narrow ? 'RM' : 'recompensa r_φ', c: C.d },
          { k: 'crit', l: narrow ? 'V_ψ' : 'crítico V_ψ', c: C.c },
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
        if (i === 8) { const u = (t * 0.7) % 1; V.arrow(g, ax(1) - as / 2 - 4, ay - 4, ax(0) + as / 2 + 4, ay - 4, { color: C.d, lw: 2.2, curve: -14 }); V.text(g, 'pesos', (ax(0) + ax(1)) / 2, ay - 24, { size: 11, font: 'mono', color: C.d, align: 'center', weight: 700 }); V.dot(g, V.lerp(ax(1), ax(0), u), ay - 18 + Math.sin(u * Math.PI) * -2, 3, C.d); }
        // ---- faixa de tokens
        const labW = narrow ? 92 : 128;
        const mx = 10 + labW, mw = w - 20 - labW;
        const cwid = mw / 5;
        const ty = ay + as / 2 + 44;
        const prompt = narrow ? 'x: cor do céu?' : 'x: De que cor é o céu?';
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
          if (pop > 0.6) V.text(g, 'RM 2,0', x, ty - 10, { size: 11, font: 'mono', weight: 700, color: C.dark ? '#1a1200' : '#fff', align: 'center', base: 'middle' });
        }
        // ---- matriz do lote
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
        // colunas
        if (!narrow) TOKS.forEach((tk, k) => V.text(g, 't=' + (k + 1), mx + k * cwid + cwid / 2, my0 - 4, { size: 10.5, font: 'mono', color: C.muted, align: 'center' }));
      });
    })();

    /* ======================================================= FIG 8 — motor no laço */
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
          return { lanes: [{ l: narrow ? 'todas' : 'todas as GPUs', segs }], P: M.col };
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
        return { lanes: [{ l: narrow ? 'rollout ' + pf + '%' : 'pool de rollout (' + pf + '%)', segs: A }, { l: narrow ? 'treino ' + (100 - pf) + '%' : 'pool de treino (' + (100 - pf) + '%)', segs: B }], P };
      }
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Destaque', options: [{ v: 'col', label: 'colocado' }, { v: 'dis', label: 'desagregado síncrono' }, { v: 'async', label: 'desagregado assíncrono' }], value: 'col', onChange: (v) => { S.mode = v; upd(); } });
      UI.slider(ctl, { label: 'geração no tempo (colocado)', min: 0.3, max: 0.95, step: 0.05, value: 0.7, fmt: (v) => Math.round(v * 100) + '%', onInput: (v) => { S.g = v; upd(); } });
      UI.slider(ctl, { label: 'GPUs no pool de rollout', min: 0.2, max: 0.8, step: 0.05, value: 0.7, fmt: (v) => Math.round(v * 100) + '%', onInput: (v) => { S.f = v; upd(); } });
      const rT = UI.readout(ctl, 'tempo por iteração', '—'), rI = UI.readout(ctl, 'GPU ociosa', '—'), rS = UI.readout(ctl, 'defasagem', '—'), rV = UI.readout(ctl, 'vs colocado', '—');
      function upd() {
        const M = model();
        const P = S.mode === 'col' ? M.col : S.mode === 'dis' ? M.dis : M.asy;
        rT.set(fmt(P, 0) + ' unid.');
        let idle = 0;
        if (S.mode !== 'col') { const iA = P - M.sN - M.gA, iB = P - M.sN - M.tB; idle = (S.f * Math.max(0, iA) + (1 - S.f) * Math.max(0, iB)) / P; }
        rI.set(Math.round(idle * 100) + '%', idle > 0.3 ? 'bad' : idle > 0.1 ? 'warn' : 'ok');
        rS.set(S.mode === 'async' ? '1 iteração' : '0 (on-policy)', S.mode === 'async' ? 'warn' : 'ok');
        const ratio = M.col / P;
        rV.set(fmt(ratio, 2) + '×', ratio >= 0.999 ? 'ok' : ratio > 0.7 ? 'warn' : 'bad');
      }
      upd();
      const NAMES = { col: 'colocado', dis: 'desagregado síncrono', async: 'desagregado assíncrono' };
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
          V.text(g, NAMES[mk] + ' · ' + fmt(L.P, 0) + ' unid. por iteração', 10, y + 13, { size: 12, color: sel ? C.ink : C.muted, weight: 700, maxW: w - 20 });
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
              if (xb - xa > 36) V.text(g, s.k === 'gen' ? 'gera' : s.k === 'score' ? 'pont.' : s.k === 'train' ? 'treina' : s.k === 'sync' ? 'sync' : 'troca', (xa + xb) / 2, y + laneH / 2 + 4, { size: 11, font: 'mono', color: C.dark ? '#06140d' : '#fff', align: 'center', weight: 700 });
            });
            y += laneH + (k < L.lanes.length - 1 ? laneGap : 0);
          });
          // fim das 3 iterações
          V.line(g, X(span), ys[0] - 3, X(span), y + 3, { color: C.ink, lw: 1.4, dash: [3, 3] });
          // rollouts descendo e pesos subindo
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
        // relógio comum
        const hx = X(head);
        V.line(g, hx, 6, hx, y - blockGap + 4, { color: V.alpha(C.ink, 0.55), lw: 1.3 });
        // legenda
        const leg = [['gera', C.a], ['pontua', C.b], ['treina', C.c], ['sync', C.d], ['troca', C.muted]];
        let lxx = 10, ly = narrow ? h - 26 : h - 10;
        leg.concat([['ociosa', null]]).forEach(([s, c]) => {
          const ww = 16 + s.length * 6.6 + 12;
          if (lxx + ww > w - 10) { lxx = 10; ly += 16; }
          if (c) V.box(g, lxx, ly - 9, 12, 10, { r: 2, fill: V.alpha(c, 0.8) }); else hatch(g, lxx, ly - 9, 12, 10, V.alpha(C.muted, 0.6), 4);
          V.text(g, s, lxx + 16, ly, { size: 11, color: C.ink }); lxx += ww;
        });
      });
    })();

    /* ======================================================= FIG 9 — painel de diagnóstico */
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
          lab: 'saudável',
          rm: (x, k) => 2.2 * (1 - e(-3 * x)) + 0.05 * n1[k], gold: (x, k) => 1.6 * (1 - e(-3 * x)) + 0.05 * n2[k],
          kl: (x, k) => 7 * (1 - e(-2.5 * x)) + 0.15 * n3[k], len: (x, k) => 180 + 25 * x + 4 * n4[k],
          rho: () => 0, vl: (x) => 0.32 * e(-3 * x) + 0.06,
          sym: 'Nota do RM e qualidade real sobem juntas; a KL se estabiliza; o comprimento fica estável; a razão sai exatamente de 1 na 1ª época; a perda de valor cai.',
          fix: 'Nada a fazer: este é o desenho de referência para comparar.',
        },
        beta: {
          lab: 'β pequeno / LR alto',
          rm: (x, k) => 4.6 * Math.pow(x, 0.75) + 0.06 * n1[k], gold: (x, k) => 7 * x * e(-3 * x) + 0.05 * n2[k],
          kl: (x, k) => 34 * Math.pow(x, 1.4) + 0.3 * n3[k], len: (x, k) => 180 + 330 * x * x + 6 * n4[k],
          rho: () => 0, vl: (x) => 0.2 + 0.35 * x,
          sym: 'A nota do RM sobe sem parar enquanto a qualidade real sobe e depois cai; a KL e o comprimento disparam.',
          fix: 'Aumente β ou mire uma KL alvo (controle adaptativo), reduza o LR, pare cedo pelo conjunto reservado.',
        },
        drop: {
          lab: 'dropout ligado',
          rm: (x, k) => 1.9 * (1 - e(-2.5 * x)) + 0.3 * n1[k], gold: (x, k) => 1.3 * (1 - e(-2.5 * x)) + 0.12 * n2[k],
          kl: (x, k) => Math.max(0, 6 * (1 - e(-2.5 * x)) + 1.8 * n3[k]), len: (x, k) => 180 + 20 * x + 6 * n4[k],
          rho: (x, k) => 0.05 + 0.015 * Math.abs(n1[(k * 7) % N]), vl: (x) => 0.3 * e(-2 * x) + 0.12,
          sym: 'A razão média já sai longe de 1 na 1ª época; a KL fica ruidosa e não confiável; a recompensa oscila.',
          fix: 'Desligue o dropout na política, na referência, no RM e no crítico (N+, Ziegler).',
        },
        crit: {
          lab: 'crítico frio',
          rm: (x, k) => 2.0 * (1 - e(-3 * Math.max(0, x - 0.3))) - 0.35 * e(-Math.pow((x - 0.12) / 0.1, 2)) + 0.05 * n1[k], gold: (x, k) => 1.5 * (1 - e(-3 * Math.max(0, x - 0.3))) - 0.3 * e(-Math.pow((x - 0.12) / 0.1, 2)) + 0.05 * n2[k],
          kl: (x, k) => 5 * (1 - e(-2 * x)) + 0.4 * e(-Math.pow((x - 0.1) / 0.08, 2)) * 6 + 0.15 * n3[k], len: (x, k) => 180 + 15 * x + 4 * n4[k],
          rho: () => 0, vl: (x) => 1.0 * e(-5 * x) + 0.08,
          sym: 'A perda de valor começa enorme; a recompensa demora a subir (ou cai no começo) e a KL dá um tranco inicial.',
          fix: 'Inicialize o crítico do RM, pré-treine o crítico antes de mexer na política e ligue o clipping do valor.',
        },
        len: {
          lab: 'EOS = PAD (comprimento)',
          rm: (x, k) => 2.6 * x + 0.05 * n1[k], gold: (x, k) => 0.55 + 0.1 * Math.sin(x * 3) + 0.05 * n2[k],
          kl: (x, k) => 12 * x + 0.2 * n3[k], len: (x, k) => Math.min(1000, 180 + 900 * x) + 6 * n4[k],
          rho: () => 0, vl: (x) => 0.25 * e(-2 * x) + 0.1,
          sym: 'O comprimento cresce até bater no teto; a nota do RM acompanha o comprimento; a qualidade real fica parada.',
          fix: 'Garanta EOS ≠ PAD, use o “EOS trick” (−1 sem EOS), vigie o comprimento e trate o viés de comprimento do RM.',
        },
      };
      const S = { sc: 'ok' };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Cenário', options: Object.keys(SC).map((k) => ({ v: k, label: SC[k].lab })), value: 'ok', onChange: (v) => { S.sc = v; rD.set(SC[v].lab, v === 'ok' ? 'ok' : 'bad'); } });
      const rD = UI.readout(ctl, 'diagnóstico', 'saudável'); rD.set('saudável', 'ok');
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const sc = SC[S.sc];
        const narrow = w < 600;
        const textH = narrow ? 112 : 66;
        const gw = (w - (narrow ? 20 : 30)) / 2, gh = (h - textH - 18) / 2;
        const ph = Math.min(1, (t % 9) / 6.5), n = Math.max(2, Math.floor(ph * N));
        const panels = [
          { t: 'nota do RM × qualidade real', y0: -0.6, y1: 4.8, yt: [0, 2, 4], series: [['rm', C.b], ['gold', C.a]] },
          { t: 'KL até o SFT (nats)', y0: 0, y1: 36, yt: [0, 10, 20, 30], series: [['kl', C.d]] },
          { t: 'comprimento médio (tokens)', y0: 100, y1: 1100, yt: [200, 600, 1000], series: [['len', C.e]], cap: 1000 },
          { t: narrow ? '|ρ−1| (dir.) · perda (esq.)' : '|ρ−1| (eixo dir., 0–0,1) e perda de valor (eixo esq.)', y0: 0, y1: 1.1, yt: [0, 0.5, 1], series: [['vl', C.c]], bars: true },
        ];
        panels.forEach((p, q) => {
          const col = q % 2, row = Math.floor(q / 2);
          const RA = p.bars ? 24 : 0; // espaço do eixo direito (|ρ−1|)
          const A = { x: 10 + col * (gw + 10) + 34, y: 8 + row * (gh + 8) + 18, w: gw - 44 - RA, h: gh - 44 };
          V.box(g, A.x - 38, A.y - 22, A.w + RA + 46, A.h + 46, { r: 8, fill: V.alpha(C.muted, 0.04), stroke: V.alpha(C.muted, 0.2), lw: 1 });
          V.text(g, p.t, A.x - 30, A.y - 8, { size: 11, color: C.ink, weight: 600, maxW: A.w + RA + 34 });
          const F = { X: (x) => A.x + x * A.w, Y: (v) => A.y + A.h - ((v - p.y0) / (p.y1 - p.y0)) * A.h };
          p.yt.forEach((v) => { const y = F.Y(v); g.strokeStyle = V.alpha(C.muted, 0.15); g.lineWidth = 1; g.beginPath(); g.moveTo(A.x, y); g.lineTo(A.x + A.w, y); g.stroke(); V.text(g, String(v).replace('.', ','), A.x - 4, y, { size: 10.5, font: 'mono', color: C.muted, align: 'right', base: 'middle' }); });
          g.strokeStyle = V.alpha(C.muted, 0.5); g.beginPath(); g.moveTo(A.x, A.y); g.lineTo(A.x, A.y + A.h); g.lineTo(A.x + A.w, A.y + A.h); g.stroke();
          if (p.cap) { V.line(g, A.x, F.Y(p.cap), A.x + A.w, F.Y(p.cap), { color: RED(), dash: [4, 4], lw: 1.2 }); V.text(g, 'max_tokens', A.x + A.w - 2, F.Y(p.cap) - 4, { size: 10.5, font: 'mono', color: RED(), align: 'right' }); }
          if (p.bars) {
            // eixo próprio à direita: 0 a 0,1 ocupa a altura inteira
            const RMAX = 0.1, YR = (v) => A.y + A.h - (V.clamp(v, 0, RMAX) / RMAX) * A.h, xr = A.x + A.w;
            g.strokeStyle = V.alpha(C.muted, 0.5); g.lineWidth = 1; g.beginPath(); g.moveTo(xr, A.y); g.lineTo(xr, A.y + A.h); g.stroke();
            [[0, '0'], [0.05, null], [0.1, '0,1']].forEach(([v, s]) => { const y = YR(v); V.line(g, xr, y, xr + 4, y, { color: V.alpha(C.muted, 0.8), lw: 1 }); if (s) V.text(g, s, xr + 6, y, { size: 10.5, font: 'mono', color: C.muted, base: 'middle' }); });
            for (let k = 0; k < n; k += 8) { const v = sc.rho(xs[k], k); const bh = A.y + A.h - YR(v); const x = F.X(xs[k]); g.fillStyle = V.alpha(v > 0.01 ? RED() : C.a, 0.75); g.fillRect(x - 2, A.y + A.h - Math.max(1.5, bh), 4, Math.max(1.5, bh)); }
            V.text(g, sc.rho(0.5, 60) > 0.01 ? '≈ ' + fmt(sc.rho(0.5, 60), 2) : '0,000', A.x + 4, A.y + 10, { size: 10.5, font: 'mono', color: sc.rho(0.5, 60) > 0.01 ? RED() : C.a, weight: 700 });
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
        // sintoma e remédio
        const ty = h - textH + 6;
        V.box(g, 10, ty - 4, w - 20, textH - 6, { r: 8, fill: V.alpha(S.sc === 'ok' ? C.a : RED(), 0.07), stroke: V.alpha(S.sc === 'ok' ? C.a : RED(), 0.4), lw: 1 });
        const hh = V.wrap(g, 'Sintoma: ' + sc.sym, 20, ty + 14, w - 40, { size: 11.5, color: C.ink, lineH: 15, maxLines: narrow ? 4 : 2 });
        V.wrap(g, 'Remédio: ' + sc.fix, 20, ty + 14 + hh + 2, w - 40, { size: 11.5, color: S.sc === 'ok' ? C.a : C.d, weight: 600, lineH: 15, maxLines: narrow ? 3 : 2 });
      });
    })();

    /* ======================================================= FIG 10 — memória */
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
        if (S.mode === 'full') { L.push({ l: 'política', c: 'a', parts: tr(N), n: N }, { l: 'crítico', c: 'c', parts: tr(Nc), n: Nc }, { l: 'referência', c: 'b', parts: fr(N), n: N }, { l: 'RM', c: 'd', parts: fr(Nc), n: Nc }); }
        else if (S.mode === 'lora') { L.push({ l: 'política + LoRA', c: 'a', parts: lo(N), n: N }, { l: 'crítico + LoRA', c: 'c', parts: lo(Nc), n: Nc }, { l: 'referência', c: 'b', parts: [], n: N, note: '= política sem adaptador' }, { l: 'RM', c: 'd', parts: fr(Nc), n: Nc }); }
        else { L.push({ l: 'política', c: 'a', parts: tr(N), n: N }, { l: 'crítico', c: 'c', parts: [], n: 0, note: 'removido (GRPO, cap. 56)' }, { l: 'referência', c: 'b', parts: fr(N), n: N }, { l: 'RM', c: 'd', parts: fr(Nc), n: Nc }); }
        if (S.engine) L.push({ l: 'motor (bf16)', c: 'e', parts: fr(N), n: N });
        L.forEach((m) => { m.bytes = m.parts.reduce((s, p) => s + p.bytes, 0); });
        return L;
      }
      const ctl = UI.controls(fig);
      UI.slider(ctl, { label: 'Parâmetros da política', min: 0, max: NL.length - 1, step: 1, value: 4, fmt: (v) => NL[v] + 'B', onInput: (v) => { S.ni = v; upd(); } });
      UI.seg(ctl, { label: 'RM e crítico', options: [{ v: 1, label: '= política' }, { v: 0.5, label: 'metade' }, { v: 0.25, label: '1/4' }], value: 1, onChange: (v) => { S.ratio = v; upd(); } });
      UI.seg(ctl, { label: 'Método', options: [{ v: 'full', label: 'full FT' }, { v: 'lora', label: 'LoRA (~1%)' }, { v: 'grpo', label: 'sem crítico' }], value: 'full', onChange: (v) => { S.mode = v; upd(); } });
      UI.toggle(ctl, { label: '+ cópia no motor de inferência', value: false, onChange: (v) => { S.engine = v; upd(); } });
      const rT = UI.readout(ctl, 'total (sem ativações/KV)', '—'), rF = UI.readout(ctl, 'cabe em', '—'), rB = UI.readout(ctl, 'bytes por parâmetro da política', '—');
      const S2 = { shown: null };
      function upd() {
        const L = models();
        const tot = L.reduce((s, m) => s + m.bytes, 0);
        const gib = tot / GiB;
        rT.set(fmt(tot / 1e9, 1) + ' GB · ' + fmt(gib, 1) + ' GiB');
        const tiers = [[MAC, 'Mac M4 24 GB (justo)'], [SPARK, '1× DGX Spark'], [2 * SPARK, '2× DGX Spark']];
        const hit = tiers.find(([cap]) => gib <= cap);
        if (!hit) rF.set('nenhum dos dois', 'bad');
        else {
          const tight = gib > 0.85 * hit[0]; // menos de 15% de margem: ativações e KV não cabem
          rF.set(hit[1] + (tight ? ' (só os estados; sem margem p/ ativações)' : ''), tight || hit[0] > SPARK ? 'warn' : 'ok');
        }
        rB.set(fmt(L[0].bytes / L[0].n, 2) + ' B');
      }
      upd();
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const L = models();
        const narrow = w < 560;
        // tiras na cor do próprio modelo (a mesma da barra empilhada); o tom diz o tipo de byte
        const TONE = { w: 0.92, g: 0.55, o: 0.22, a: 0.55 };
        const top = 26, rowH = Math.min(46, (h - 170) / L.length);
        const labW = narrow ? 104 : 150, gibW = narrow ? 70 : 96;
        const cellX = 10 + labW, cellW = w - 20 - labW - gibW;
        const cw = cellW / 16;
        V.text(g, narrow ? 'bytes por parâmetro (1 célula = 1 B)' : 'bytes por parâmetro (cada célula = 1 byte)', cellX, top - 10, { size: 11, color: C.muted, weight: 600, maxW: w - cellX - 8 });
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
          if (m.parts.length && !narrow) { const lbl = m.parts.length === 3 ? '2 pesos · 2 grad · 12 Adam+mestre' : m.parts.length === 2 ? '2 pesos (congelados) + adaptador' : '2 pesos bf16'; V.text(g, lbl, cellX, y + ch + 11, { size: 10.5, color: C.muted, font: 'mono' }); }
        });
        // legenda dos tons
        {
          const ly = top + L.length * rowH + 6;
          const items = [['w', 'pesos'], ['g', 'grad'], ['o', narrow ? 'Adam+mestre' : 'Adam + mestre FP32']].concat(S.mode === 'lora' ? [['a', narrow ? 'adaptador' : 'adaptador LoRA']] : []);
          let lx = narrow ? 16 : cellX;
          if (!narrow) V.text(g, 'tom:', lx - 6, ly + 10, { size: 11, color: C.muted, align: 'right' });
          items.forEach(([k, s]) => {
            V.box(g, lx, ly, 14, 12, { r: 3, fill: V.alpha(C.ink, TONE[k] * 0.8), stroke: k === 'a' ? C.ink : V.alpha(C.ink, 0.6), lw: 1 });
            V.text(g, s, lx + 18, ly + 10, { size: 11, color: C.ink });
            g.font = V.font(11, 'sans', 400); lx += 18 + g.measureText(s).width + 12;
          });
        }
        // barra empilhada
        const tot = L.reduce((s, m) => s + m.bytes, 0) / GiB;
        const scale = Math.max(tot, 2 * SPARK) * 1.08;
        const by = h - 96, bh = 30, bx = 14, bw = w - 28;
        const X = (v) => bx + (v / scale) * bw;
        if (!S2.shown || S2.shown.length !== L.length) S2.shown = L.map(() => 0);
        let acc = 0;
        V.text(g, 'memória total (GiB)', bx, by - 12, { size: 11, color: C.muted, weight: 600 });
        V.box(g, bx, by, bw, bh, { r: 6, fill: V.alpha(C.muted, 0.06), stroke: V.alpha(C.muted, 0.3), lw: 1 });
        L.forEach((m, k) => {
          S2.shown[k] += (m.bytes / GiB - S2.shown[k]) * 0.15;
          const a = X(acc), b = X(acc + S2.shown[k]);
          if (b - a > 0.5) { V.box(g, a, by + 2, b - a, bh - 4, { r: 3, fill: V.alpha(C[m.c], 0.8) }); if (b - a > 40) V.text(g, m.l.split(' ')[0], (a + b) / 2, by + bh / 2 + 4, { size: 10.5, color: C.dark ? '#06140d' : '#fff', align: 'center', weight: 700, maxW: b - a - 4 }); }
          acc += S2.shown[k];
        });
        [[MAC, 'Mac M4 24 GB', C.muted], [SPARK, '1× Spark ~121,7 GiB', C.e], [2 * SPARK, '2× Spark ~243 GiB', C.d]].forEach(([v, s, c], k) => {
          const x = X(v);
          V.line(g, x, by - 4, x, by + bh + 8, { color: c, lw: 2, dash: [4, 3] });
          const lyy = by + bh + 20 + (k % 2) * 14;
          V.text(g, s, x, lyy, { size: 10.5, font: 'mono', color: c, align: x > w - 90 ? 'right' : x < 70 ? 'left' : 'center' });
        });
        const ok = tot <= SPARK ? C.e : tot <= 2 * SPARK ? C.d : RED();
        g.font = V.font(11, 'sans', 600);
        const titleEnd = bx + g.measureText('memória total (GiB)').width + 12; // o total nunca cobre o título da barra
        V.text(g, fmt(tot, 1) + ' GiB', V.clamp(X(tot) + 6, titleEnd, w - 70), by - 12, { size: 12, font: 'mono', weight: 700, color: ok });
      });
    })();
  },
});

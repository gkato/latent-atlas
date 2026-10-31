LA.chapter({
  id: 'mapa-da-jornada',
  part: 0,
  order: 0,
  title: 'Mapa da jornada',
  navTitle: 'Mapa da jornada',
  subtitle: 'O que é uma “receita” de inferência, as três receitas que servem de laboratório (com os números reais), a primeira conta que toda receita faz, o modelo do tutorial de receita do zero e como usar esta plataforma — inclusive com um copilot rodando no seu Mac.',
  minutes: 35,
  recipes: ['glm', 'tf', 'qwen', 'ds'],
  tags: ['receita', 'visão geral', 'memória', 'banda', 'modelos híbridos', 'MiMo', 'TensorFold', 'copilot', 'MLX'],
  goals: [
    'Separar <b>pesos</b>, <b>arquitetura</b>, <b>motor</b> e <b>hardware</b> — e as três perguntas que toda receita responde',
    'Conhecer as camadas de decisão de uma receita e onde cada uma é estudada na trilha',
    'Fazer a conta de guardanapo de memória e velocidade — e entender por que ela é só um teto',
    'Reconhecer as três receitas (e o MiMo-V2.6-Flash) pelos números: parâmetros, camadas, quantização, KV, TP, especulação, contexto e velocidades medidas — e ver, na versão TensorFold da GLM, que o motor também é uma decisão de receita',
    'Usar a plataforma: trilha, busca, glossário, figuras e o copilot — inclusive local, com MLX',
  ],
  prereqs: [],
  summary: [
    'Uma receita é o conjunto documentado de decisões que faz pesos + arquitetura + motor caberem no hardware. Toda receita responde três perguntas: <b>cabe?</b>, <b>é rápido?</b>, <b>continua certo?</b>',
    'Memória dos pesos ≈ \\(N_{\\text{total}} \\times b_{\\text{total}}/8\\); decode ≲ \\(\\tau \\times\\) banda ÷ \\(B_{\\text{passo}}\\), com \\(B_{\\text{passo}} \\approx N_{\\text{ativos}} \\times b_{\\text{ativos}}/8\\). Num MoE de precisão mista, \\(b_{\\text{ativos}}\\) é bem maior que a média do checkpoint (GLM: ~6,9 contra 4,37 bits), porque a leitura por token é dominada pelas partes densas. E é um <b>teto</b>: coletivas de rede, kernels pequenos e a banda efetiva deixam o medido abaixo dele (Qwen: ~47 de teto × 24,5 tok/s medidos sem especulação). Do lado da memória, a GMU guarda a folga do sistema: na GLM, com GMU 0,85, sobram ~13 GiB por nó para o KV.',
    'As três receitas: <b>GLM</b> 321 B / ~18 B ativos, EXL3 4 bpw, KV <code>fp8_ds_mla</code>, TP=2, DFlash2 k=7, 850k; <b>Qwen</b> 125 B (+51 B de N-gram) / ~6 B, NVFP4, KV fp8, TP=2+EP, MTP k=3, 262k; <b>DeepSeek</b> 284 B / 13 B, FP4+FP8, <code>nvfp4_ds_mla</code>, TP=2, DSpark k=6, 1M. A GLM também roda no <b>TensorFold</b>: mesmo checkpoint, outro motor (exato bit a bit, KV FP8 de 528 B, janela de 1M por requisição, até 4 ao mesmo tempo na v1.3.2 e até 8 desde a v1.5), 60,4 tok/s em prosa contra 36,1 — medições em condições diferentes.',
    'Os três modelos são <b>híbridos</b>: só uma fração das camadas paga KV por token (11 de 45 na GLM, 12 de 48 na Qwen, versões comprimidas na DeepSeek) — é isso que faz 850k–1M tokens caberem em pools de 11–17 GiB.',
    'O <b>MiMo-V2.6-Flash</b> (309 B / 15 B, SWA:global 39:9, K/V de 192/128, experts MXFP4, MTP + DFlash) é o exame final: cabe no limite e quebra várias suposições do motor.',
    'Números só valem com <b>recibo</b> (data, lane, tipo de prompt, protocolo): no mesmo teste de 2026-08-30, a GLM mede 65,1 tok/s em texto estruturado e 27,1 em prosa, porque o rascunhador acerta 6,71 × 2,39 tokens por passo.',
    'O copilot fala o contrato OpenAI Chat Completions — o mesmo das receitas — e roda 100% local no Mac com <code>./tools/start-mlx-copilot.sh</code> (Base URL <code>http://127.0.0.1:8080/v1</code>, modelo vazio).',
  ],
  quiz: [
    { q: 'Um modelo de 300 B parâmetros em 4 bits ocupa aproximadamente quanto só de pesos?', o: ['~37,5 GB', '~150 GB', '~600 GB', '~1,2 TB'], a: 1, e: '300×10⁹ × 4/8 bytes = 150×10⁹ bytes ≈ 150 GB (≈ 140 GiB). Não cabe nos 121,7 GiB que o CUDA enxerga num nó — por isso as receitas usam dois. 37,5 GB seria 1 bit por peso; 600 GB, 16 bits.' },
    { q: 'Por que MoE ajuda na <i>velocidade</i> de decode?', o: ['Porque tem menos parâmetros totais', 'Porque cada token só lê os pesos dos experts ativados', 'Porque dispensa o KV cache', 'Porque usa menos bits'], a: 1, e: 'Em decode o gargalo é ler pesos da memória. Com MoE, cada token usa só uma fração dos parâmetros (13 B de 284 B na DeepSeek), então lê bem menos bytes (num MoE de precisão mista, boa parte desses bytes vem das partes densas em FP8/BF16, não dos experts). O total continua grande (é ele que decide se cabe), o KV continua existindo e os bits são uma escolha separada (quantização).' },
    { q: 'Qual pergunta a medição de KLD/perplexidade responde?', o: ['Cabe?', 'É rápido?', 'Continua certo?', 'Nenhuma'], a: 2, e: 'Métricas de qualidade medem se os atalhos (quantização, kernels, especulação) preservaram o comportamento do modelo original — ex.: a GLM a 4 bpw tem KLD de 0,024555 nats contra o BF16, praticamente igual ao FP8 oficial com 54% dos bytes.' },
    { q: 'Qual combinação de especulação cada receita usa por padrão?', o: ['GLM: MTP k=3 · Qwen: DFlash2 k=7 · DeepSeek: DSpark k=6', 'GLM: DFlash2 k=7 · Qwen: MTP k=3 · DeepSeek: DSpark k=6', 'As três usam MTP k=2', 'Nenhuma usa especulação por padrão'], a: 1, e: 'GLM: DFlash2 (rascunho em bloco) com k=7; Qwen: a própria camada MTP com k=3 e vocabulário de rascunho reduzido; DeepSeek: DSpark com k=6 (e não 5, por uma regra de divisibilidade do motor). O MTP k=2 é só o rollback da GLM.' },
    { q: 'Por que o KV cache do MiMo-V2.6-Flash cresce devagar com o contexto?', o: ['Porque o modelo não tem camadas de atenção', 'Porque 39 das 48 camadas usam janela deslizante de 128 tokens (KV fixo por conversa) e só 9 camadas globais crescem com o contexto', 'Porque os experts estão em MXFP4', 'Porque o DFlash substitui o KV cache'], a: 1, e: 'Só as 9 camadas globais guardam KV por token (5.760 B/token por GPU em FP8); as 39 de janela custam ~10–13 MB fixos por conversa. MXFP4 reduz os pesos, não o KV; o DFlash é um rascunhador e tem seu próprio KV.' },
    { q: 'Você sobe o copilot com MLX no Mac. O que ele envia, e para onde?', o: ['Um POST /v1/chat/completions no contrato OpenAI para http://127.0.0.1:8080/v1 — o mesmo contrato que o vLLM das receitas expõe', 'Chamadas gRPC diretas à GPU do Mac', 'O texto da página para um servidor na nuvem, sempre', 'Os pesos do modelo para o navegador'], a: 0, e: 'O copilot só conhece o contrato OpenAI Chat Completions (com streaming SSE). MLX, Ollama, LM Studio, OpenAI e a sua receita vLLM viram só uma URL base diferente — e no preset MLX nada sai da sua máquina.' },
  ],
  mount(root, LA) {
    const V = LA.viz, UI = LA.ui;
    const B = (r) => LA.recipeBadge(r);
    const GiB = 1073741824;
    const nf = (x, d = 0) => (+x).toLocaleString('pt-BR', { minimumFractionDigits: d, maximumFractionDigits: d });
    const tfCache = { k: null, v: null };
    const TFC = (C) => { if (tfCache.k !== C) { tfCache.k = C; tfCache.v = getComputedStyle(document.documentElement).getPropertyValue('--tf').trim() || C.d; } return tfCache.v; }; // cor do selo GLM·TF
    const recCol = (C, k) => (k === 'glm' ? C.glm : k === 'qwen' ? C.qwen : k === 'ds' ? C.ds : k === 'tf' ? TFC(C) : C.d); // MiMo (comunidade) em âmbar
    const fontC = {};
    const F = (size, font, weight) => { const k = size + (font || 'sans') + (weight || 400); return fontC[k] || (fontC[k] = V.font(size, font || 'sans', weight || 400)); };
    const tw = (g, s, size, weight, font) => { g.font = F(size, font, weight); return g.measureText(s).width; };
    const halo = (g, s, x, y, o = {}) => {
      const C = V.C();
      g.font = F(o.size || 11, o.font || 'sans', o.weight || 500);
      g.textAlign = o.align || 'left'; g.textBaseline = o.base || 'alphabetic';
      g.lineJoin = 'round'; g.lineWidth = 3.5; g.strokeStyle = C.bg; g.strokeText(s, x, y);
      g.fillStyle = o.color || C.ink; g.fillText(s, x, y);
    };
    const tip = (g, lines, px, py, w, h) => {
      const C = V.C();
      const W = Math.min(w - 16, Math.max(...lines.map((l, i) => tw(g, l, i ? 11.5 : 12.5, i ? 400 : 600))) + 20);
      const Hh = 12 + lines.length * 17;
      let x = px + 14, y = py + 14;
      if (x + W > w - 6) x = Math.max(6, px - W - 14);
      if (y + Hh > h - 6) y = Math.max(6, py - Hh - 10);
      V.box(g, x, y, W, Hh, { r: 9, fill: V.alpha(C.panel, 0.96), stroke: V.alpha(C.muted, 0.5), lw: 1 });
      lines.forEach((l, i) => V.text(g, l, x + 10, y + 20 + i * 17, { size: i ? 11.5 : 12.5, weight: i ? 400 : 600, color: i ? C.muted : C.ink, maxW: W - 18 }));
    };
    const wrapLines = (g, str, maxW, o = {}) => {
      const words = String(str).split(' '), out = []; let line = '';
      words.forEach((wd) => { const test = line ? line + ' ' + wd : wd; if (line && tw(g, test, o.size || 12, o.weight || 400, o.font) > maxW) { out.push(line); line = wd; } else line = test; });
      if (line) out.push(line);
      return out;
    };
    const wrapText = (g, str, x, y, maxW, lh, o) => { const L = wrapLines(g, str, maxW, o); L.forEach((l, i) => V.text(g, l, x, y + i * lh, o)); return y + Math.max(0, L.length - 1) * lh; };
    const wrapCtl = (o) => { const el = o && o.el ? o.el : o; if (el && el.style) { el.style.flexWrap = 'wrap'; el.style.maxWidth = '100%'; el.style.minWidth = '0'; } const s = el && el.querySelector && el.querySelector('select'); if (s) { s.style.maxWidth = '100%'; s.style.minWidth = '0'; } return o; };
    const tokLbl = (n) => (n >= 1048576 ? nf(n / 1048576, 1) + 'M' : n >= 1024 ? nf(n / 1024, n < 10240 ? 1 : 0) + 'k' : nf(n, 0));
    const ctxLbl = (n) => (n >= 1048576 ? nf(n / 1048576, n % 1048576 ? 2 : 0) + 'M' : nf(n / 1024, 0) + 'k');

    /* ================================================================
       Figura 1 — quatro mundos (stepper + canvas)
       ================================================================ */
    const f1 = root.querySelector('#fig-mundos');
    const steps1 = [
      { t: 'Pesos — a partitura', d: 'Um <b>checkpoint</b> é uma coleção de tensores guardados em arquivos <code>.safetensors</code>. No GLM-5.3-Flash são <b>321.323.031.390</b> números: em BF16 (2 bytes cada) seriam ~642,6 GB; a receita usa uma versão de 175,6 GB em 120 arquivos. Cada quadradinho é um peso (valores ilustrativos) — sozinho, não significa nada.' },
      { t: 'Arquitetura — como os números se combinam', d: 'A arquitetura diz que aqueles números formam <b>matrizes</b> e em que ordem multiplicá-las. Aqui o vetor \\(x\\) (a representação de um token) é multiplicado pela matriz \\(W\\): cada saída \\(y_i = \\sum_j W_{ij}\\,x_j\\) é o produto escalar da linha \\(i\\) com \\(x\\). Um token do GLM atravessa 45 camadas, cada uma com várias matrizes assim.' },
      { t: 'Motor — o maestro', d: 'O <b>vLLM</b> recebe pedidos de vários usuários, junta-os num mesmo passo (<span class="la-term" data-term="continuous-batching"><i>batching contínuo</i></span>), decide quem entra e quem espera, e guarda o <b>KV cache</b> de cada conversa em blocos de memória. A cada passo, cada conversa do lote ganha um token novo.' },
      { t: 'Hardware — a orquestra', d: 'Dois nós, cada um com 128 GB nominais de memória unificada (o CUDA enxerga 121,7 GiB) e ~273 GB/s de banda teórica, ligados por RDMA a 200 Gb/s (~25 GB/s). Na GLM, cada GPU guarda 82,94 GiB de pesos (79,65 sem a cópia BF16 do KDA, ligada por padrão desde 2026-09-26) e um pool de KV de 11 GiB; os ~27,8 GiB restantes vão para ativações, CUDA graphs e o sistema operacional. Tudo precisa caber aqui — e ser lido rápido o bastante.' },
    ];
    const st1 = UI.stepper(f1, steps1, { autoplay: 7500 });
    const r1 = V.rng(20260928);
    const WGRID = Array.from({ length: 48 }, () => Array.from({ length: 30 }, () => r1.normal() * 0.035));
    const round1 = (A) => A.map((r) => r.map((v) => Math.round(v * 10) / 10));
    const MAT = {
      big: { W: round1(V.randMatrix(5, 8, 7, 0.9)), x: round1([V.randMatrix(1, 8, 3, 0.9)[0]])[0] },
      small: { W: round1(V.randMatrix(4, 5, 9, 0.9)), x: round1([V.randMatrix(1, 5, 4, 0.9)[0]])[0] },
    };
    Object.values(MAT).forEach((m) => { m.y = m.W.map((row) => row.reduce((s, v, j) => s + v * m.x[j], 0)); });
    const f1d = (v) => (v < 0 ? '−' : '') + nf(Math.abs(v), 1);
    const subN = (i) => String(i).split('').map((c) => '₀₁₂₃₄₅₆₇₈₉'[+c]).join('');

    function drawWeights(g, w, h, t, C) {
      const pad = 14, top = 30, bot = 66;
      V.text(g, w < 480 ? 'trecho de um tensor (ilustrativo)' : 'um trecho de um tensor de pesos (valores ilustrativos)', pad, 19, { size: 11.5, font: 'mono', color: C.muted, maxW: w - 2 * pad });
      const cw = w >= 640 ? 46 : 32, chh = 20, gap = 3;
      const cols = Math.max(4, Math.floor((w - 2 * pad + gap) / (cw + gap)));
      const rows = Math.ceil((h - top - bot) / (chh + gap)) + 1;
      const off = t * 1.3, base = Math.floor(off), frac = off - base;
      const x0 = pad + (w - 2 * pad - (cols * (cw + gap) - gap)) / 2;
      g.save(); g.beginPath(); g.rect(0, top, w, h - top - bot); g.clip();
      for (let r = 0; r < rows; r++) {
        const row = WGRID[(r + base) % WGRID.length];
        const y = top + (r - frac) * (chh + gap);
        for (let c = 0; c < cols; c++) {
          const v = row[c % row.length], tt = V.clamp(v / 0.09, -1, 1);
          V.rr(g, x0 + c * (cw + gap), y, cw, chh, 4); g.fillStyle = V.cmap(tt, 'div'); g.fill();
          if (cw >= 40) V.text(g, (v < 0 ? '−' : '') + Math.abs(v).toFixed(3), x0 + c * (cw + gap) + cw / 2, y + chh / 2 + 0.5, { size: 10, font: 'mono', align: 'center', base: 'middle', color: Math.abs(tt) > 0.55 ? C.bg : C.ink });
        }
      }
      g.restore();
      const gr = g.createLinearGradient(0, top, 0, top + 26); gr.addColorStop(0, C.bg); gr.addColorStop(1, V.alpha(C.bg, 0));
      g.fillStyle = gr; g.fillRect(0, top, w, 26);
      const gr2 = g.createLinearGradient(0, h - bot - 26, 0, h - bot); gr2.addColorStop(0, V.alpha(C.bg, 0)); gr2.addColorStop(1, C.bg);
      g.fillStyle = gr2; g.fillRect(0, h - bot - 26, w, 26);
      V.text(g, '321.323.031.390 pesos', w / 2, h - 36, { size: w < 420 ? 18 : 21, font: 'display', weight: 700, align: 'center', color: C.a });
      V.text(g, 'BF16 ≈ 642,6 GB · checkpoint EXL3 da receita (TR3, retrato): 175,6 GB em 120 shards', w / 2, h - 14, { size: 11.5, font: 'mono', align: 'center', color: C.muted, maxW: w - 16 });
    }

    function drawMatVec(g, w, h, t, C) {
      const small = w < 560, M = small ? MAT.small : MAT.big;
      const Wm = M.W, xv = M.x, yv = M.y, R = Wm.length, K = Wm[0].length, gap = 2;
      const cs = Math.max(22, Math.min(40, Math.floor((w - 110) / (K + 1)) - gap, Math.floor((h - 150) / (R + 1)) - gap));
      const Ww = K * (cs + gap) - gap;
      const total = Ww + 30 + cs;
      const x0 = Math.max(30, (w - total) / 2 + 10), yx = 44, yW = yx + cs + 16;
      const per = 1.7, cyc = t / per, i = Math.floor(cyc) % R, p = cyc - Math.floor(cyc);
      const sweep = Math.min(1, p / 0.8), j = Math.min(K - 1, Math.floor(sweep * K)), done = p >= 0.8;
      const fmt = (v) => (Number.isNaN(v) ? '?' : f1d(v));
      V.text(g, 'x', x0 - 12, yx + cs / 2, { size: 15, font: 'mono', weight: 700, align: 'right', base: 'middle', color: C.b });
      V.text(g, 'W', x0 - 12, yW + (R * (cs + gap)) / 2, { size: 15, font: 'mono', weight: 700, align: 'right', base: 'middle', color: C.a });
      V.text(g, 'x = um token (vetor) · W = pesos de uma camada', x0 - 30, 22, { size: 11.5, font: 'mono', color: C.muted, maxW: w - x0 + 20 });
      V.matrix(g, { x: x0, y: yx, data: [xv], cell: cs, gap, cmap: 'div', min: -2, max: 2, values: true, fmt, hiCells: done ? [] : [[0, j]], hiColor: C.b });
      const hiC = []; if (!done) hiC.push([i, j]);
      V.matrix(g, { x: x0, y: yW, data: Wm, cell: cs, gap, cmap: 'div', min: -2, max: 2, values: true, fmt, hiRow: i, hiCells: hiC });
      if (!done) { // linha do produto: x[j] desce até W[i][j]
        const cx = x0 + j * (cs + gap) + cs / 2;
        V.arrow(g, cx, yx + cs + 2, cx, yW + i * (cs + gap) - 3, { color: V.alpha(C.b, 0.8), lw: 1.6, head: 6 });
      }
      const ex = x0 + Ww + 14;
      V.text(g, '=', ex, yW + (R * (cs + gap)) / 2, { size: 20, align: 'center', base: 'middle', color: C.muted });
      const yd = yv.map((v, r) => [r < i || (r === i && done) ? v : NaN]);
      V.matrix(g, { x: ex + 14, y: yW, data: yd, cell: cs, gap, cmap: 'div', min: -4, max: 4, values: true, fmt, hiCells: [[i, 0]], hiColor: C.a });
      V.text(g, 'y', ex + 14 + cs / 2, yW - 8, { size: 15, font: 'mono', weight: 700, align: 'center', color: C.a });
      let acc = 0; const last = done ? K - 1 : j;
      for (let q = 0; q <= last; q++) acc += Wm[i][q] * xv[q];
      const shown = []; for (let q = 0; q <= Math.min(last, small ? 1 : 2); q++) shown.push(`(${f1d(Wm[i][q])})(${f1d(xv[q])})`);
      const yb = yW + R * (cs + gap) + 28;
      V.text(g, `y${subN(i)} = ${shown.join(' + ')}${last > (small ? 1 : 2) ? ' + …' : ''} = ${f1d(Math.round(acc * 100) / 100)}${done ? ' ✓' : ''}`, w / 2, yb, { size: 13, font: 'mono', align: 'center', color: C.ink, maxW: w - 20 });
      V.text(g, 'multiplica, soma, próxima linha — milhões de linhas por token', w / 2, yb + 22, { size: 11.5, align: 'center', color: C.muted, maxW: w - 20 });
    }

    function drawEngine(g, w, h, t, C) {
      const cols = [C.a, C.b, C.c, C.d], U = 4, P = 2.2;
      const n = Math.floor(t / P), ph = (t % P) / P;
      const narrow = w < 560;
      const ux = narrow ? 22 : 30, uy = (u) => 62 + u * ((h - 132) / (U - 1));
      const sw = narrow ? 80 : 124, sx = (narrow ? w * 0.27 : w * 0.36) - sw / 2, sy = h / 2 - 34;
      const gx = narrow ? w * 0.48 : w * 0.6, gw = w - gx - (narrow ? 8 : 14), gy = 30, gh = h - 60;
      const ip = narrow ? 8 : 10, iw = gw - 2 * ip;
      V.text(g, 'usuários', 14, 30, { size: 11.5, font: 'mono', color: C.muted });
      // trilhos fixos (ida e volta)
      for (let u = 0; u < U; u++) {
        V.line(g, ux + 14, uy(u), sx, sy + 20 + u * 9, { color: V.alpha(cols[u], 0.18), lw: 1.2, dash: [3, 5] });
      }
      for (let u = 0; u < U; u++) { V.dot(g, ux, uy(u), 13, V.alpha(cols[u], 0.25)); V.text(g, 'ABCD'[u], ux, uy(u) + 0.5, { size: 12, weight: 700, align: 'center', base: 'middle', color: cols[u] }); }
      V.box(g, sx, sy, sw, 68, { r: 12, fill: V.alpha(C.b, 0.14), stroke: C.b, lw: 1.5 });
      V.text(g, 'Scheduler', sx + sw / 2, sy + 26, { size: narrow ? 12 : 12.5, weight: 600, align: 'center', color: C.ink, maxW: sw - 8 });
      V.text(g, `passo ${n + 1}`, sx + sw / 2, sy + 46, { size: 11, font: 'mono', align: 'center', color: C.muted });
      V.box(g, gx, gy, gw, gh, { r: 12, fill: V.alpha(C.a, 0.08), stroke: V.alpha(C.a, 0.6), lw: 1.5 });
      V.text(g, 'GPU(s) · modelo', gx + gw / 2, gy + 22, { size: 12.5, weight: 600, align: 'center', color: C.ink, maxW: iw });
      let yy = wrapText(g, 'lote: 4 conversas por passo', gx + gw / 2, gy + 40, iw, 13, { size: 11, font: 'mono', align: 'center', color: C.muted });
      // KV cache em blocos por conversa
      yy = wrapText(g, 'KV cache (blocos por conversa)', gx + ip, yy + 22, iw, 13, { size: 11, color: C.muted });
      const kvTop = yy + 7, bs = Math.min(13, iw / 10 - 2), M = Math.floor((iw + 2) / (bs + 2));
      for (let u = 0; u < U; u++) {
        const fill = Math.min(M, 1 + Math.floor(((n + u * 5) % 36) / (2 + u)));
        for (let b = 0; b < M; b++) {
          V.rr(g, gx + ip + b * (bs + 2), kvTop + u * (bs + 4), bs, bs, 3);
          g.fillStyle = b < fill ? V.alpha(cols[u], b === fill - 1 && ph > 0.6 ? 0.95 : 0.6) : V.alpha(C.muted, 0.1); g.fill();
        }
      }
      // rodapé da caixa (quebrado em linhas, sem espremer)
      const botL = wrapLines(g, '1 leitura dos pesos serve as 4 conversas', iw, { size: 11 });
      const botY = gy + gh - 12 - (botL.length - 1) * 13;
      botL.forEach((l, i) => V.text(g, l, gx + gw / 2, botY + i * 13, { size: 11, align: 'center', color: C.a }));
      // camadas do modelo: o lote atravessa todas, uma leitura dos pesos por passo
      yy = wrapText(g, 'camadas (pesos lidos 1× por passo)', gx + ip, kvTop + U * (bs + 4) + 16, iw, 13, { size: 11, color: C.muted });
      const ly0 = yy + 8, nL = 8, lgap = Math.max(5, Math.min(13, (botY - 16 - ly0) / nL));
      const fw = ph < 0.4 ? -1 : Math.min(nL, Math.floor(((ph - 0.4) / 0.25) * nL));
      for (let L = 0; L < nL; L++) {
        const y = ly0 + (nL - 1 - L) * lgap + 2;
        V.rr(g, gx + ip, y, iw, Math.max(3, lgap - 4), 3);
        g.fillStyle = L <= fw && ph < 0.75 ? V.alpha(C.a, 0.85) : V.alpha(C.a, 0.18); g.fill();
      }
      // partículas
      for (let u = 0; u < U; u++) {
        const a = [ux + 14, uy(u)], s2 = [sx, sy + 20 + u * 9], gi = [gx, gy + gh / 2 + u * 8];
        if (ph < 0.3) { const k = V.ease.inOut(ph / 0.3); const q = [V.lerp(a[0], s2[0], k), V.lerp(a[1], s2[1], k)]; V.glow(g, q[0], q[1], 9, cols[u]); V.dot(g, q[0], q[1], 4, cols[u]); }
        else if (ph < 0.45) { const k = V.ease.inOut((ph - 0.3) / 0.15); V.rr(g, V.lerp(sx + sw, gi[0], k) - 5, V.lerp(s2[1], gi[1], k) - 5, 10, 10, 2); g.fillStyle = cols[u]; g.fill(); }
        else if (ph >= 0.72) { const k = V.ease.inOut((ph - 0.72) / 0.28); const pt = V.bez([gx, gi[1] + 30], [gx - 60, h - 20], [ux + 90, uy(u) + 40], [ux + 14, uy(u)], k); V.glow(g, pt[0], pt[1], 10, cols[u]); V.dot(g, pt[0], pt[1], 3.5, cols[u]); }
      }
      V.arrow(g, sx + sw + 4, sy + 34, gx - 6, sy + 34, { color: V.alpha(C.muted, 0.45), lw: 1.2, dash: [4, 4] });
      if (ph >= 0.72) V.text(g, '+1 token cada', ux + 24, h - 12, { size: 11, font: 'mono', color: C.ink });
    }

    function drawHW(g, w, h, t, C) {
      const narrow = w < 600;
      // no desktop, os nós encolhem o bastante para o rótulo do enlace caber entre eles (≥ 118 px)
      const nw = Math.min(w * 0.42, (w - 36 - 118) / 2);
      const nodes = narrow ? [[14, 26, w - 28, 150], [14, h - 176, w - 28, 150]] : [[18, 46, nw, 236], [w - 18 - nw, 46, nw, 236]];
      const WG = 82.94, KVG = 11, REST = 121.69 - WG - KVG;
      const segs = [[WG, C.b, 'pesos ' + nf(WG, 2)], [KVG, C.c, 'KV ' + nf(KVG, 0)], [REST, V.alpha(C.muted, 0.45), 'resto ' + nf(REST, 1) + ' GiB', C.muted]];
      nodes.forEach(([x, y, W, Hh], k) => {
        V.box(g, x, y, W, Hh, { r: 14, fill: V.alpha(C.a, 0.05), stroke: V.alpha(C.a, 0.5), lw: 1.5 });
        V.text(g, k ? 'Nó B · worker (rank 1)' : 'Nó A · head (rank 0) + API', x + 14, y + 22, { size: 12.5, weight: 600, color: C.ink, maxW: W - 28 });
        const bx = x + 14, by = y + 34, bw = W - 28, bh = narrow ? 18 : 24;
        let cx = bx;
        segs.forEach(([v, col]) => { const sw = (v / 121.69) * bw; V.rr(g, cx, by, Math.max(2, sw - 1.5), bh, 5); g.fillStyle = col; g.fill(); cx += sw; });
        V.text(g, '121,7 GiB visíveis ao CUDA', bx, by + bh + 16, { size: 11, color: C.muted, maxW: bw });
        // legenda colorida, com quebra de linha em vez de espremer
        let lx = bx, ly = by + bh + 32;
        segs.forEach(([, col, lab, tc], i) => {
          const s = lab + (i < segs.length - 1 ? ' ·' : '');
          const wl = tw(g, s, 11, 400, 'mono');
          if (lx > bx && lx + wl > bx + bw) { lx = bx; ly += 14; }
          V.text(g, s, lx, ly, { size: 11, font: 'mono', color: tc || col });
          lx += wl + 6;
        });
        const gw = narrow ? 118 : 132, ghh = narrow ? 36 : 48;
        const gx2 = narrow ? x + W - gw - 14 : x + W / 2 - gw / 2, gy2 = y + Hh - ghh - (narrow ? 8 : 12);
        V.box(g, gx2, gy2, gw, ghh, { r: 10, fill: V.alpha(C.a, 0.18), stroke: C.a, lw: 1.5, label: 'GPU GB10 · sm_121', size: 11.5 });
        const px = narrow ? gx2 + gw / 2 : x + W / 2, py0 = ly + 10, py1 = gy2 - 4;
        if (!narrow && py1 - py0 > 8) {
          V.line(g, px, py0, px, py1, { color: V.alpha(C.a, 0.35), lw: 6 });
          for (let q = 0; q < 4; q++) { const u = ((t * 0.9 + q / 4) % 1); V.dot(g, px, V.lerp(py0, py1, u), 2.6, C.a); }
          V.text(g, '~273 GB/s', px + 10, (py0 + py1) / 2 + 4, { size: 11, font: 'mono', color: C.a });
        } else {
          V.text(g, '~273 GB/s ↓', gx2 - 8, gy2 + ghh / 2 + 4, { size: 11, font: 'mono', align: 'right', color: C.a });
        }
      });
      // enlace RDMA
      let a, b;
      if (narrow) { a = [w / 2, nodes[0][1] + nodes[0][3]]; b = [w / 2, nodes[1][1]]; }
      else { a = [nodes[0][0] + nodes[0][2], nodes[0][1] + nodes[0][3] / 2]; b = [nodes[1][0], a[1]]; }
      V.line(g, a[0], a[1], b[0], b[1], { color: C.d, lw: 2.5, dash: [3, 5], dashOffset: -t * 30 });
      for (let q = 0; q < 3; q++) { const u = (t * 0.6 + q / 3) % 1; const pt = V.along(a[0], a[1], b[0], b[1], q % 2 ? 1 - u : u); V.rr(g, pt[0] - 4, pt[1] - 3, 8, 6, 2); g.fillStyle = C.d; g.fill(); }
      const lx = narrow ? w / 2 + 10 : (a[0] + b[0]) / 2, ly = narrow ? (a[1] + b[1]) / 2 + 4 : a[1] - 12;
      V.text(g, 'RDMA · 200 Gb/s', lx, ly, { size: 11, font: 'mono', align: narrow ? 'left' : 'center', color: C.d, maxW: narrow ? w - lx - 8 : b[0] - a[0] - 8 });
      if (!narrow) V.text(g, '(~25 GB/s)', lx, ly + 28, { size: 10.5, font: 'mono', align: 'center', color: C.muted, maxW: b[0] - a[0] - 8 });
      V.text(g, 'exemplo: GLM-5.3-Flash em TP=2, por GPU', w / 2, h - 8, { size: 11, align: 'center', color: C.muted });
    }

    V.fig(f1, (g, w, h, t) => {
      const C = V.C();
      g.globalAlpha = V.ease.out(Math.min(1, st1.elapsed() * 2.2));
      if (st1.i === 0) drawWeights(g, w, h, t, C);
      else if (st1.i === 1) drawMatVec(g, w, h, t, C);
      else if (st1.i === 2) drawEngine(g, w, h, t, C);
      else drawHW(g, w, h, t, C);
      g.globalAlpha = 1;
    });

    /* ================================================================
       Figura 2 — anatomia (stepper + canvas)
       ================================================================ */
    const layers = [
      { t: 'Hardware & rede', d: 'Duas máquinas com GPU de memória unificada (128 GB cada) ligadas por uma placa de rede RDMA de alta velocidade. Define os limites físicos: capacidade, banda e latência.<br>' + B('all') + ' 2× DGX Spark (GB10, <code>sm_121</code>) com um cabo ConnectX-7 (RoCE, 200 Gb/s); a GLM documenta variantes com 3 e 4 nós (a de 4, experimental), e a DeepSeek, com 3.' },
      { t: 'Imagem de software', d: 'Um container Docker com CUDA, PyTorch, o vLLM (às vezes uma versão específica ou um fork) e kernels compilados para a arquitetura exata da GPU.<br>' + B('glm') + ' fork do vLLM + Dockerfile com patches de build · ' + B('qwen') + ' imagem day-0 do vLLM (ou a 0.30.0 stock) · ' + B('ds') + ' imagem Anemll 0.1.1 (vLLM 0.25.2.dev0) · ' + B('tf') + ' container PyTorch da NVIDIA com o TensorFold v0.6.0 no lugar do vLLM, fixado por digest.' },
      { t: 'Pesos (checkpoint)', d: 'Qual versão do modelo, em qual formato de quantização (EXL3, NVFP4, FP8…), baixada do Hugging Face com revisão fixada e verificada.<br>' + B('glm') + ' EXL3 4 bpw (espelho Mia-AiLab, 175,6 GB) · ' + B('qwen') + ' NVFP4 da NVIDIA (132,6 GB) · ' + B('ds') + ' FP4+FP8 oficial, revisão <code>86f746b3…</code> (~157 GiB).' },
      { t: 'Patches & overlays', d: 'Arquivos Python/CUDA que substituem ou corrigem partes do vLLM para suportar o modelo, o formato ou o hardware. É aqui que mora boa parte da “engenharia” de uma receita.<br>' + B('glm') + ' 25 overlays em ordem fixa · ' + B('qwen') + ' overlays montados sobre a imagem (N-gram, MXFP8, KV FP8…) · ' + B('ds') + ' cadeia de hotfixes <i>fail-closed</i> no boot do container · ' + B('tf') + ' 53 patches aplicados uma vez, no build da imagem, e não a cada partida.' },
      { t: 'Topologia', d: 'Como dividir o modelo entre as máquinas: tensor parallel (TP), expert parallel (EP), quantos nós, qual interface de rede o NCCL usa.<br>' + B('glm') + ' TP=2 · ' + B('qwen') + ' TP=2 + EP · ' + B('ds') + ' TP=2 — as três com executor <code>mp</code> e <code>--nnodes 2</code> · ' + B('tf') + ' TP=2 obrigatório, um rank por Spark: all-gather das parciais e soma em ordem de rank no lugar do all-reduce.' },
      { t: 'Argumentos do motor', d: 'Tamanho máximo de contexto, fração de memória para o KV cache, tokens por passo, sequências simultâneas, dtype do KV cache, parsers de ferramentas e raciocínio.<br>Contexto: 850.000 ' + B('glm') + ', 262.144 ' + B('qwen') + ', 1.048.576 ' + B('ds') + ' · KV: <code>fp8</code>, <code>fp8</code>, <code>nvfp4_ds_mla</code> · sequências: 4, 8, 6 · ' + B('tf') + ' janela de 1.048.576 tokens por requisição, até 4 requisições (v1.3.2; até 8 desde a v1.5) num pool compartilhado de ~2,9M tokens, KV FP8 de 528 B por token e camada.' },
      { t: 'Especulação', d: 'Um “rascunhador” (MTP, DFlash2, DSpark) propõe vários tokens por passo e o modelo grande só verifica — multiplicando a velocidade de decode sem mudar a saída.<br>' + B('glm') + ' DFlash2 k=7 · ' + B('qwen') + ' MTP k=3 · ' + B('ds') + ' DSpark k=6 · ' + B('tf') + ' DFlash2 até 7 por rodada + rascunhos por cópia até 15, aceitos só se iguais ao token sorteado com chave.' },
      { t: 'Verificação & tuning', d: 'Testes de fumaça, checagem numérica contra a referência, benchmarks de throughput e latência, escadas de concorrência — e iteração nos parâmetros.<br>' + B('glm') + ' KLD de 0,024555 nats contra o BF16 (checkpoint TR3) · ' + B('qwen') + ' agulhas a 13k e 104k com KV FP8 × BF16 (12/12) · ' + B('ds') + ' RULER-lite, bateria de ferramentas e checagem de texto corrompido até ~900k · ' + B('tf') + ' respostas idênticas com e sem rascunhos e com 4 requisições juntas ou sozinhas; GSM8K 98,0% (v1.3.2, TR3).' },
    ];
    const f2 = root.querySelector('#fig-anatomia');
    const stage2 = f2.querySelector('.la-fig-stage');
    const fitH2 = () => { const w = stage2.clientWidth || 800; stage2.style.height = (w < 560 ? 560 : 380) + 'px'; };
    fitH2();
    const st2 = UI.stepper(f2, layers, { autoplay: 6500 });
    const parts2 = Array.from({ length: 70 }, (_, i) => ({ p: i / 70, lane: i % 3 }));
    // o que cada camada da receita controla no fluxo (API, Scheduler, nós A/B, enlace, saída de tokens)
    const HI2 = [['A', 'B', 'link'], ['A', 'B', 'img'], ['A', 'B', 'w'], ['S', 'A', 'B', 'patch'], ['A', 'B', 'link', 'tp'], ['API', 'S'], ['tok', 'spec'], ['tok', 'check']];
    const NOTE2 = [
      '2 nós · 121,7 GiB visíveis cada · ~273 GB/s · RDMA 200 Gb/s',
      'a mesma imagem (motor + kernels para sm_121) roda nos dois nós',
      'cada nó carrega a sua metade dos pesos do checkpoint',
      'overlays trocam arquivos do vLLM na partida, antes do primeiro token',
      'TP=2: a cada camada, os nós somam resultados parciais pela rede (all-reduce)',
      'contexto máximo, lote, GMU e dtype do KV decidem quem entra na fila',
      'k rascunhos por passo, verificados de uma só vez pelo modelo grande',
      'benchmarks e KLD conferem os tokens que saem contra a referência',
    ];
    V.fig(f2, (g, w, h, t) => {
      const C = V.C();
      const cols = [C.a, C.b, C.c, C.d, C.e, C.f, C.a, C.b];
      const narrow = w < 560;
      const sx = narrow ? 14 : 24, sw = narrow ? w - 28 : Math.min(250, w * 0.34);
      const top = narrow ? 10 : 26, stackH = narrow ? h - 230 : h - 50, lh = stackH / layers.length;
      layers.forEach((L, i) => {
        const k = layers.length - 1 - i; // hardware embaixo
        const y = top + k * lh;
        const on = st2.i === i;
        const col = cols[i];
        V.box(g, sx + (on ? -4 : 0), y, sw + (on ? 8 : 0), lh - 6, { r: 9, fill: on ? V.alpha(col, 0.28) : V.alpha(col, 0.08), stroke: on ? col : V.alpha(col, 0.35), lw: on ? 2 : 1 });
        V.text(g, String(i + 1).padStart(2, '0'), sx + 12, y + (lh - 6) / 2 + 4, { size: 11, font: 'mono', color: col });
        V.text(g, L.t, sx + 38, y + (lh - 6) / 2 + 5, { size: 13, weight: on ? 600 : 400, color: on ? C.ink : C.muted, maxW: sw - 46 });
        if (on && !narrow) V.glow(g, sx + sw + 4, y + (lh - 6) / 2, 16, col);
      });
      // fluxo da requisição: à direita no desktop, compacto embaixo no celular
      const x0 = narrow ? 14 : sx + sw + 50, x1 = narrow ? w - 14 : w - 30;
      if (x1 - x0 < 200) return;
      const cy = narrow ? top + stackH + 82 : h / 2 - 14, dy = narrow ? 46 : 70;
      const hi = new Set(HI2[st2.i]), hc = cols[st2.i];
      const labs = narrow ? ['API', 'Scheduler', 'Nó A', 'Nó B', 'tokens'] : ['API', 'Scheduler', 'Nó A · GPU', 'Nó B · GPU', 'tokens'];
      const bws = labs.map((l) => tw(g, l, 12, 500) + 24);
      const xApi = x0 + bws[0] / 2, xTok = x1 - bws[4] / 2, xNode = x0 + (x1 - x0) * 0.62;
      let xS = x0 + (x1 - x0) * 0.3;
      xS = Math.max(xS, xApi + bws[0] / 2 + 16 + bws[1] / 2); // espaço mínimo entre as caixas
      xS = Math.min(xS, xNode - bws[2] / 2 - 16 - bws[1] / 2);
      const nodes = [
        { k: 'API', x: xApi, y: cy, l: labs[0], c: C.d, bw: bws[0] },
        { k: 'S', x: xS, y: cy, l: labs[1], c: C.b, bw: bws[1] },
        { k: 'A', x: xNode, y: cy - dy, l: labs[2], c: C.a, bw: bws[2] },
        { k: 'B', x: xNode, y: cy + dy, l: labs[3], c: C.a, bw: bws[3] },
        { k: 'tok', x: xTok, y: cy, l: labs[4], c: C.c, bw: bws[4] },
      ];
      const edges = [[0, 1], [1, 2], [1, 3], [2, 4], [3, 4]];
      edges.forEach(([a, b]) => V.line(g, nodes[a].x, nodes[a].y, nodes[b].x, nodes[b].y, { color: V.alpha(C.muted, 0.4), lw: 1.5, dash: [4, 6], dashOffset: -t * 20 }));
      // enlace RDMA entre os nós
      const lk = hi.has('link');
      const la = [nodes[2].x, nodes[2].y + 18], lb = [nodes[3].x, nodes[3].y - 18];
      if (lk) V.glow(g, la[0], (la[1] + lb[1]) / 2, 30, hc, 0.5);
      V.line(g, la[0], la[1], lb[0], lb[1], { color: lk ? hc : C.a, lw: lk ? 4 : 2.5, dash: [2, 5], dashOffset: -t * 40 });
      V.text(g, 'RDMA', la[0] + 10, cy + 4, { size: 11, font: 'mono', weight: lk ? 600 : 400, color: lk ? hc : C.a });
      if (hi.has('tp')) for (let q = 0; q < 4; q++) { const u = (t * 0.8 + q / 4) % 1, pt = V.along(la[0], la[1], lb[0], lb[1], q % 2 ? 1 - u : u); V.rr(g, pt[0] - 4, pt[1] - 3, 8, 6, 2); g.fillStyle = hc; g.fill(); }
      // partículas de requisição e de tokens
      const spec = hi.has('spec');
      parts2.forEach((p) => {
        const u = (p.p + t * 0.16) % 1;
        let pt;
        const lane = p.lane === 0 ? 2 : 3;
        if (u < 0.33) pt = V.along(nodes[0].x, nodes[0].y, nodes[1].x, nodes[1].y, u / 0.33);
        else if (u < 0.66) pt = V.along(nodes[1].x, nodes[1].y, nodes[lane].x, nodes[lane].y, (u - 0.33) / 0.33);
        else if (!spec) pt = V.along(nodes[lane].x, nodes[lane].y, nodes[4].x, nodes[4].y, (u - 0.66) / 0.34);
        if (pt) V.dot(g, pt[0], pt[1], 2.4, u > 0.66 ? C.c : C.a);
      });
      if (spec) { // tokens saem em rajadas de k por passo
        [2, 3].forEach((ni, r) => {
          const u = (t * 0.45 + r * 0.5) % 1;
          for (let q = 0; q < 5; q++) { const uu = u - q * 0.035; if (uu < 0) continue; const pt = V.along(nodes[ni].x, nodes[ni].y, nodes[4].x, nodes[4].y, uu); V.rr(g, pt[0] - 3.5, pt[1] - 3.5, 7, 7, 2); g.fillStyle = hc; g.fill(); }
        });
      }
      // caixas (acesas quando a camada da receita mexe nelas)
      nodes.forEach((n) => {
        const on = hi.has(n.k);
        if (on) V.glow(g, n.x, n.y, n.bw * 0.7, hc, 0.45);
        V.box(g, n.x - n.bw / 2, n.y - 17, n.bw, 34, { r: 10, fill: C.bg });
        V.box(g, n.x - n.bw / 2, n.y - 17, n.bw, 34, { r: 10, fill: V.alpha(on ? hc : n.c, on ? 0.3 : 0.12), stroke: on ? hc : V.alpha(n.c, 0.55), lw: on ? 2.5 : 1.2, label: n.l, size: 12, color: on ? C.ink : C.muted });
      });
      const nA = nodes[2], nB = nodes[3];
      if (hi.has('img')) [nA, nB].forEach((n, r) => {
        g.setLineDash([4, 3]); V.box(g, n.x - n.bw / 2 - 8, n.y - 25, n.bw + 16, 50, { r: 12, stroke: hc, lw: 1.3 }); g.setLineDash([]);
        V.text(g, 'container', n.x + n.bw / 2 + 12, n.y + (r ? 22 : -18), { size: 10.5, font: 'mono', color: hc });
      });
      if (hi.has('w')) [nA, nB].forEach((n, r) => {
        const yy = r ? n.y + 30 : n.y - 38;
        for (let i = 0; i < 2; i++) for (let j = 0; j < 6; j++) { V.rr(g, n.x - n.bw / 2 + j * 8, yy + i * 7, 6, 5, 1.5); g.fillStyle = V.alpha(hc, 0.35 + 0.4 * Math.abs(Math.sin(t * 2 + i + j * 0.7 + r))); g.fill(); }
        V.text(g, '½ dos pesos', n.x - n.bw / 2 + 54, yy + 10, { size: 10.5, font: 'mono', color: hc });
      });
      if (hi.has('patch')) [nodes[1], nA, nB].forEach((n, r) => {
        const px = n.x + n.bw / 2 - 2, py = n.y - 17, pr = 7 + Math.sin(t * 4 + r) * 1.2;
        V.dot(g, px, py, pr, hc); V.text(g, '+', px, py + 0.5, { size: 11, weight: 700, align: 'center', base: 'middle', color: C.bg });
      });
      if (hi.has('S')) V.text(g, 'lote ≤ 4–8', nodes[1].x, nodes[1].y + 32, { size: 10.5, font: 'mono', align: 'center', color: hc });
      if (hi.has('API')) V.text(g, 'max-model-len · GMU', x0, nodes[0].y - 25, { size: 10.5, font: 'mono', color: hc });
      if (spec) V.text(g, 'k = 7 · 3 · 6', nodes[4].x, nodes[4].y + 32, { size: 10.5, font: 'mono', align: 'center', color: hc });
      if (hi.has('check')) {
        V.text(g, '✓', nodes[4].x + nodes[4].bw / 2 - 6, nodes[4].y - 20, { size: 15, weight: 700, align: 'center', color: hc });
        for (let q = 0; q < 4; q++) { const bh = 6 + 10 * Math.abs(Math.sin(t * 1.5 + q)); V.rr(g, nodes[4].x - 16 + q * 9, nodes[4].y + 40 - bh, 6, bh, 2); g.fillStyle = V.alpha(hc, 0.7); g.fill(); }
      }
      // legenda do passo
      wrapText(g, NOTE2[st2.i], (x0 + x1) / 2, cy + dy + (narrow ? 64 : 62), x1 - x0, 15, { size: 12, weight: 600, align: 'center', color: hc });
    });

    /* ================================================================
       Figura 3 — calculadora de guardanapo (com presets reais)
       ================================================================ */
    const f3 = root.querySelector('#fig-conta');
    const stage3 = f3.querySelector('.la-fig-stage');
    const fitH3 = () => { const w = stage3.clientWidth || 800; stage3.style.height = (w < 560 ? 460 : 400) + 'px'; };
    fitH3();
    const ctl3 = UI.controls(f3);
    const NODE_GIB = 121.69, BW_NODE = 273e9, MAXT = 120;
    const S3 = { total: 300, bTot: 4.5, active: 20, bAct: 4.5, nodes: 2, gmu: 0.85, tau: 1, rsv: 6, preset: 'livre' };
    const NOTE_FREE = 'Modo livre: mexa à vontade. Num modelo todo no mesmo formato, os bits dos ativos ≈ os bits médios; num MoE de precisão mista, cada parâmetro ativo custa mais, porque as partes densas ficam em FP8/BF16. A reserva do motor é a do último preset (6 GiB no início, um valor ilustrativo). Escolha um preset para carregar os números reais e comparar o teto com o medido.';
    const PRE = {
      glm: { total: 321.3, bTot: 4.37, active: 17.4, bAct: 6.9, nodes: 2, gmu: 0.85, rsv: 9, meas: [[24.6, 'MTP k=2'], [27.1, 'prosa'], [65.1, 'estruturado']], note: B('glm') + ' 321,3 B totais, 175,6 GB no disco → <b>4,37 bits</b> de média (memória). A leitura por token é outra história: os experts (97% do total) estão em 4 bits, mas o que se lê a cada token é dominado pelas partes densas — KDA, MLA, experts compartilhados, <code>lm_head</code> — em FP8/BF16. Pelo modelo de custo do repo, ≈ 5,4 GB de densos + ≈ 2,1 GB de experts ≈ 7,5 GB por GPU por token sem especulação (estimativa) → <b>6,9 bits efetivos</b> sobre os ~17,4 B ativos. Reserva de ~9 GiB (103 − 83 − 11 na receita, que fixa o pool em 11 GiB). <b>Medido</b> no mesmo teste (<code>bench_decode</code>, 2026-08-30): 24,6 tok/s no rollback MTP k=2; com DFlash2 k=7, 27,1 em prosa e 65,1 em texto estruturado.' },
      qwen: { total: 180.0, bTot: 5.9, active: 7.3, bAct: 12.8, nodes: 2, gmu: 0.8, rsv: 5, meas: [[24.5, 'sem spec'], [52.1, 'MTP=3']], note: B('qwen') + ' 180,0 B armazenados (125,7 principal + 51,2 N-gram + 2,6 MTP + visão), 132,6 GB → <b>5,9 bits</b> de média; 7,3 B ativos com embeddings e <code>lm_head</code>. A leitura por token é quase toda BF16 — projeções GDN, o Gated Residual replicado nas duas GPUs, um <code>lm_head</code> de 248.320 × 2560 —, e os experts NVFP4 são só ~11% dela. O repo modela o passo com MTP=3 em ≈ 9,2 GiB por GPU; sem a MTP, sem as 3 leituras extras do <code>lm_head</code> e com ~10 experts por camada em vez de ~32, sobram ≈ 5,5 GiB por GPU (interpretação) → <b>12,8 bits efetivos</b>. GMU 0,80: a 0,835, um teste de carga deixou o <code>MemAvailable</code> em 0,3–0,9 GiB — o aperto era do sistema, não do KV. <b>Medido</b> (greedy, 2026-09-05): 24,5 tok/s sem especulação e 52,1 com MTP=3.' },
      ds: { total: 304.6, bTot: 4.43, active: 13.2, bAct: 6.0, nodes: 2, gmu: 0.835, rsv: 5, meas: [[56, 'DSpark k=6'], [75.4, '0731']], note: B('ds') + ' 284,3 B do tronco + 19,8 B do DSpark + 0,47 B de visão, ~157 GiB → <b>4,43 bits</b> de média; 13,2 B ativos. O repo estima 16,5–17 GB lidos por GPU por passo com k=6 (verificar 7 posições acorda até ~42 experts distintos por camada). Com um token só, sobram ≈ 5 GB por GPU — 7 experts × 43 camadas (~2 GB), atenção ≈ 2,4 GB e uma leitura do <code>lm_head</code> (interpretação) → <b>~6,0 bits efetivos</b>. GMU 0,835 → pool medido de 17,04 GiB. <b>Medido</b>: 56 tok/s com DSpark k=6 (mediana de 8, 2026-09-02); 75,4 com o checkpoint 0731 (2026-08-01).' },
      mimo: { total: 310.8, bTot: 4.45, active: 15, bAct: 6.8, nodes: 2, gmu: 0.85, rsv: 4, meas: [[25.9, 'prosa'], [53.3, 'média'], [70.5, 'código']], note: 'MiMo-V2.6-Flash: 310,8 B (contagem do Hugging Face), 15 B ativos (card), 172,9 GB → <b>4,45 bits</b> de média. Por token (interpretação, a partir da tabela de pesos do tutorial): experts ativos em MXFP4 ≈ 5,0 GB, QKV em FP8 ≈ 2,9 GB, <code>o_proj</code> em BF16 ≈ 3,2 GB, <code>lm_head</code> ≈ 1,25 GB e o resto ≈ 0,4 GB — ≈ 12,8 GB → <b>6,8 bits efetivos</b>. GMU 0,85 com ~4 GiB de reserva (a estimativa do tutorial). <b>Medido por uma receita comunitária</b> (vLLM TP=2, DFlash k=7): 25,9 tok/s em prosa, 70,5 em código, 53,3 na média das categorias.' },
    };
    const gmuFmt = (v) => nf(v, Math.abs(v * 100 - Math.round(v * 100)) > 1e-6 ? 3 : 2);
    let applying = false;
    const selP = wrapCtl(UI.select(ctl3, { label: 'Preset', options: [{ v: 'livre', label: 'Livre' }, { v: 'glm', label: 'GLM-5.3-Flash (EXL3)' }, { v: 'qwen', label: 'Qwen3.8-Flash-Next (NVFP4)' }, { v: 'ds', label: 'DeepSeek-V4-Flash (DSpark)' }, { v: 'mimo', label: 'MiMo-V2.6-Flash (tutorial)' }], value: 'livre', onChange: (v) => applyPreset(v) }));
    const manual = () => { if (applying || S3.preset === 'livre') return; S3.preset = 'livre'; selP.el.querySelector('select').value = 'livre'; note3.innerHTML = NOTE_FREE; };
    const sT = UI.slider(ctl3, { label: 'Params totais', min: 10, max: 1200, step: 0.1, value: S3.total, fmt: (v) => nf(v, v < 100 ? 1 : 0) + ' B', onInput: (v) => { S3.total = v; if (S3.active > v) sA.set(v); manual(); H3.redraw(); } });
    const sB = UI.slider(ctl3, { label: 'Bits médios (memória)', min: 2, max: 16, step: 0.01, value: S3.bTot, fmt: (v) => nf(v, 2), onInput: (v) => { S3.bTot = v; manual(); H3.redraw(); } });
    const sA = UI.slider(ctl3, { label: 'Params ativos', min: 1, max: 120, step: 0.1, value: S3.active, fmt: (v) => nf(v, 1) + ' B', onInput: (v) => { S3.active = Math.min(v, S3.total); manual(); H3.redraw(); } });
    const sBA = UI.slider(ctl3, { label: 'Bits dos ativos (velocidade)', min: 2, max: 16, step: 0.01, value: S3.bAct, fmt: (v) => nf(v, 2), onInput: (v) => { S3.bAct = v; manual(); H3.redraw(); } });
    const sN = wrapCtl(UI.seg(ctl3, { label: 'Nós', options: [1, 2, 3, 4], value: 2, onChange: (v) => { S3.nodes = v; manual(); H3.redraw(); } }));
    const sG = UI.slider(ctl3, { label: 'GMU', min: 0.5, max: 0.95, step: 0.005, value: S3.gmu, fmt: gmuFmt, onInput: (v) => { S3.gmu = v; H3.redraw(); } });
    const sTau = UI.slider(ctl3, { label: 'Tokens por passo (τ)', min: 1, max: 8, step: 0.1, value: 1, fmt: (v) => nf(v, 1), onInput: (v) => { S3.tau = v; H3.redraw(); } });
    [sT, sB, sA, sBA, sG, sTau].forEach(wrapCtl);
    const note3 = LA.h('div', { style: { flexBasis: '100%', fontSize: '13px', lineHeight: '1.55', color: 'var(--text-2)' }, html: NOTE_FREE });
    ctl3.appendChild(note3);
    function applyPreset(k) {
      S3.preset = k;
      const p = PRE[k];
      if (!p) { note3.innerHTML = NOTE_FREE; H3.redraw(); return; }
      applying = true;
      sT.set(p.total); sA.set(p.active); sB.set(p.bTot); sBA.set(p.bAct); sN.set(p.nodes); sG.set(p.gmu); sTau.set(1);
      S3.rsv = p.rsv;
      applying = false;
      S3.preset = k;
      note3.innerHTML = p.note;
      H3.redraw();
    }
    const H3 = V.fig(f3, (g, w, h) => {
      const C = V.C();
      const narrow = w < 560, pad = narrow ? 12 : 22, bw = w - 2 * pad;
      const wB = (S3.total * 1e9 * S3.bTot) / 8;             // bytes de pesos, total
      const wNode = wB / S3.nodes / GiB;                       // GiB de pesos por nó
      const budget = S3.gmu * NODE_GIB;                        // orçamento do vLLM por nó
      const kv = budget - wNode - S3.rsv;                      // o que sobra para o KV cache
      const outside = NODE_GIB - budget;                       // fora do vLLM: SO, page cache
      const fracSum = wB / (S3.nodes * NODE_GIB * GiB);
      const state = wNode > NODE_GIB ? 3 : kv <= 0 ? 2 : kv < 20 ? 1 : 0;
      const kvCol = [C.a, C.d, C.c, C.c][state];
      // ---- cabeçalho da memória
      V.text(g, narrow ? 'UM NÓ · 121,7 GiB visíveis' : 'MEMÓRIA DE UM NÓ · 121,7 GiB visíveis ao CUDA', pad, 16, { size: 11, font: 'mono', color: C.muted, maxW: bw * 0.62 });
      V.text(g, `${nf(fracSum * 100, 0)}% da memória somada`, pad + bw, 16, { size: 11, font: 'mono', align: 'right', color: C.muted, maxW: bw * 0.36 });
      let y = wrapText(g, `pesos: ${nf(wB / 1e9, 1)} GB = ${nf(wB / GiB, 1)} GiB ÷ ${S3.nodes} ${S3.nodes > 1 ? 'nós' : 'nó'} = ${nf(wNode, 1)} GiB por nó`, pad, 34, bw, 15, { size: 12, font: 'mono', weight: 600, color: C.ink });
      // ---- barra de um nó: pesos | reserva do motor | KV | fora do vLLM
      const by = y + 10, bh = 28, X = (gib) => pad + bw * V.clamp(gib / NODE_GIB, 0, 1);
      V.box(g, pad, by, bw, bh, { r: 8, fill: V.alpha(C.muted, 0.1), stroke: V.alpha(C.muted, 0.3) });
      g.save(); V.rr(g, pad, by, bw, bh, 8); g.clip();
      const xG = X(budget);
      g.save(); g.beginPath(); g.rect(xG, by, Math.max(0, pad + bw - xG), bh); g.clip();
      g.strokeStyle = V.alpha(C.muted, 0.4); g.lineWidth = 1;
      for (let xx = xG - bh; xx < pad + bw; xx += 7) { g.beginPath(); g.moveTo(xx, by + bh); g.lineTo(xx + bh, by); g.stroke(); }
      g.restore();
      const xW = X(wNode), xR = X(Math.min(budget, wNode + S3.rsv));
      if (wNode < budget) { g.fillStyle = V.alpha(C.muted, 0.45); g.fillRect(xW, by, Math.max(0, xR - xW), bh); }
      if (kv > 0) { g.fillStyle = V.alpha(kvCol, 0.75); g.fillRect(xR, by, Math.max(0, xG - xR), bh); }
      g.fillStyle = wNode > budget ? V.alpha(C.red || C.c, 0.8) : V.alpha(C.b, 0.8); g.fillRect(pad, by, Math.max(3, xW - pad), bh);
      g.restore();
      V.line(g, xG, by - 5, xG, by + bh + 5, { color: C.ink, lw: 1.6 });
      const gmuL = `GMU ${gmuFmt(S3.gmu)}`, gW = tw(g, gmuL, 10.5, 600, 'mono');
      V.text(g, gmuL, V.clamp(xG, pad + gW / 2, pad + bw - gW / 2), by + bh + 15, { size: 10.5, font: 'mono', weight: 600, align: 'center', color: C.ink });
      // legenda dos segmentos (quebra de linha em telas estreitas)
      const segs = [[V.alpha(C.b, 0.8), `pesos ${nf(Math.min(wNode, 999), 1)}`], [V.alpha(C.muted, 0.45), `motor ~${nf(S3.rsv, 0)}`], [V.alpha(kvCol, 0.75), `KV ${nf(Math.max(0, kv), 1)}`], ['hatch', `fora do vLLM ${nf(outside, 1)} GiB`]];
      let lx = pad, ly = by + bh + 34;
      segs.forEach(([c, l]) => {
        const wl = tw(g, l, 11.5) + 22;
        if (lx > pad && lx + wl > pad + bw) { lx = pad; ly += 17; }
        if (c === 'hatch') { V.rr(g, lx, ly - 9, 12, 10, 2); g.strokeStyle = V.alpha(C.muted, 0.7); g.lineWidth = 1; g.stroke(); V.line(g, lx + 2, ly, lx + 10, ly - 8, { color: V.alpha(C.muted, 0.7), lw: 1 }); }
        else { V.rr(g, lx, ly - 9, 12, 10, 2); g.fillStyle = c; g.fill(); }
        V.text(g, l, lx + 17, ly, { size: 11.5, color: C.muted });
        lx += wl;
      });
      const msg = state === 3 ? '✕ não cabe nem os pesos — quantize mais ou adicione nós'
        : state === 2 ? `✕ na prática não cabe: com GMU ${gmuFmt(S3.gmu)}, pesos e reserva já passam do orçamento do vLLM — não sobra KV cache`
          : state === 1 ? `≈ cabe, apertado — só ~${nf(kv, 0)} GiB por nó para o KV cache; cada GiB vira briga com o contexto e o lote`
            : `✓ cabe — ~${nf(kv, 0)} GiB por nó para o KV cache, e ${nf(outside, 0)} GiB fora do vLLM para o sistema`;
      y = wrapText(g, msg, pad, ly + 22, bw, 17, { size: 13, weight: 500, color: [C.a, C.d, C.c, C.c][state] });
      // ---- velocidade
      const Bp = (S3.active * 1e9 * S3.bAct) / 8;
      const tps = (S3.tau * BW_NODE * S3.nodes) / Bp;
      const p = PRE[S3.preset];
      y = wrapText(g, `VELOCIDADE · lê ${nf(Bp / 1e9, 1)} GB por passo (${nf(Bp / 1e9 / S3.nodes, 1)} por nó) · banda ${S3.nodes} × 273 GB/s${S3.tau > 1 ? ' · × τ = ' + nf(S3.tau, 1) : ''}${p ? ' · marcas rosa = medido' : ''}`, pad, y + 28, bw, 14, { size: 11, font: 'mono', color: C.muted });
      const cx = w / 2, cy = h - 28;
      const R = Math.max(46, Math.min(108, w < 420 ? w / 2 - 95 : w / 2 - 80, cy - y - 64));
      g.lineCap = 'round';
      g.lineWidth = 12; g.strokeStyle = V.alpha(C.muted, 0.18);
      g.beginPath(); g.arc(cx, cy, R, Math.PI, 2 * Math.PI); g.stroke();
      const u = Math.min(1, tps / MAXT);
      const grad = g.createLinearGradient(cx - R, 0, cx + R, 0); grad.addColorStop(0, C.b); grad.addColorStop(1, C.a);
      g.strokeStyle = grad; g.beginPath(); g.arc(cx, cy, R, Math.PI, Math.PI + Math.PI * u); g.stroke();
      g.lineCap = 'butt';
      V.text(g, '0', cx - R - 10, cy + 4, { size: 10.5, font: 'mono', align: 'right', color: C.muted });
      V.text(g, MAXT + '+', cx + R + 10, cy + 4, { size: 10.5, font: 'mono', color: C.muted });
      if (p) {
        const placed = [];
        const hit = (b) => placed.some((q) => b.x < q.x + q.w + 4 && b.x + b.w + 4 > q.x && b.y < q.y + q.h + 2 && b.y + b.h + 2 > q.y);
        p.meas.slice().sort((a1, a2) => a1[0] - a2[0]).forEach(([v, l]) => {
          const a = Math.PI + Math.PI * Math.min(1, v / MAXT), ca = Math.cos(a), sa = Math.sin(a);
          V.line(g, cx + ca * (R - 10), cy + sa * (R - 10), cx + ca * (R + 10), cy + sa * (R + 10), { color: C.c, lw: 2.5 });
          const lab = `${nf(v, 1)} ${l}`, wl = tw(g, lab, 10.5, 500, 'mono');
          const al = ca < -0.3 ? 'right' : ca > 0.3 ? 'left' : 'center';
          let rr = R + 18, bx0 = 0, ay = 0, ax = 0;
          for (let k = 0; k < 6; k++) {
            ax = cx + ca * rr; ay = cy + sa * rr + 4;
            bx0 = al === 'right' ? ax - wl : al === 'left' ? ax : ax - wl / 2;
            const sh = V.clamp(bx0, 4, w - 4 - wl) - bx0; bx0 += sh; ax += sh;
            if (!hit({ x: bx0, y: ay - 10, w: wl, h: 13 })) break;
            rr += 14;
          }
          placed.push({ x: bx0, y: ay - 10, w: wl, h: 13 });
          if (rr > R + 19) V.line(g, cx + ca * (R + 11), cy + sa * (R + 11), cx + ca * (rr - 8), cy + sa * (rr - 8), { color: V.alpha(C.c, 0.45), lw: 1 });
          halo(g, lab, ax, ay, { size: 10.5, font: 'mono', align: al, color: C.c });
        });
      }
      V.text(g, tps >= 1000 ? '>1000' : nf(tps, tps < 10 ? 1 : 0), cx, cy - 14, { size: 32, font: 'display', weight: 700, align: 'center', color: C.ink });
      V.text(g, w < 420 ? 'tok/s · teto, 1 conversa' : 'tok/s — teto teórico, 1 conversa', cx, cy + 16, { size: 11.5, align: 'center', color: C.muted });
    }, { static: true });

    /* ================================================================
       Figura 4 — pilha 3D
       ================================================================ */
    const f4 = root.querySelector('#fig-pilha');
    const stage4 = f4.querySelector('.la-fig-stage');
    Object.assign(stage4.style, { display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: '10px' });
    const left4 = LA.h('div', { style: { flex: '1.4 1 300px', minWidth: '0' } });
    const right4 = LA.h('div', { style: { flex: '1 1 300px', minWidth: '0', position: 'relative', height: '300px' } });
    const cv4 = LA.h('canvas', { style: { display: 'block', width: '100%', height: '100%' } });
    right4.appendChild(cv4);
    stage4.append(left4, right4);
    let split = false;
    const colorFor = (s, i, j) => {
      const C = V.C();
      if (!split) return V.alpha(C.series[s % 3], 0.25 + 0.5 * Math.abs(Math.sin(i * 1.7 + j * 0.9 + s)));
      return j < 5 ? V.alpha(C.a, 0.3 + 0.5 * Math.abs(Math.sin(i + j))) : V.alpha(C.b, 0.3 + 0.5 * Math.abs(Math.sin(i + j)));
    };
    const narrow4 = (stage4.clientWidth || 800) < 480;
    const C0 = V.C();
    const T4 = UI.tensor3d(left4, { slabs: 6, rows: 6, cols: 10, cell: narrow4 ? 13 : 18, depthGap: narrow4 ? 24 : 30, rotate: true, color: colorFor, slabColor: (s) => [C0.a, C0.b, C0.c][s % 3], labels: [] });
    left4.appendChild(LA.h('div', { style: { textAlign: 'center', fontSize: '12px', fontFamily: 'var(--font-mono)', color: 'var(--text-3)', marginTop: '-18px' } }, 'placas = camadas, da 0 (fundo) à L-1 (frente)'));
    T4.explode(1.7);
    const c4 = UI.controls(f4);
    let leg4 = null;
    const repaint = () => {
      const C = V.C();
      T4.slabs.forEach((sl, s) => sl.style.setProperty('--sc', [C.a, C.b, C.c][s % 3]));
      for (let s = 0; s < 6; s++) for (let i = 0; i < 6; i++) for (let j = 0; j < 10; j++) T4.setCell(s, i, j, colorFor(s, i, j));
      if (leg4) { const is = leg4.querySelectorAll('i'); if (is[0]) is[0].style.setProperty('--c', C.a); if (is[1]) is[1].style.setProperty('--c', C.b); leg4.style.display = split ? '' : 'none'; }
    };
    UI.toggle(c4, { label: 'dividir entre 2 nós (tensor parallel)', value: false, onChange: (v) => { split = v; repaint(); } });
    wrapCtl(UI.slider(c4, { label: 'Afastar camadas', min: 0.3, max: 2.4, step: 0.1, value: 1.7, fmt: (v) => nf(v, 1) + '×', onInput: (v) => T4.explode(v) }));
    leg4 = UI.legend(c4, [[C0.a, 'Nó A (metade de cada matriz)'], [C0.b, 'Nó B (a outra metade)']]);
    leg4.style.display = 'none';
    if (LA.theme && LA.onCleanup) LA.onCleanup(LA.theme.onChange(() => setTimeout(repaint, 0)));
    // inset 2D: uma camada de expert da GLM em TP=2 — gate/up cortadas pelas saídas, down pelas entradas, all-reduce
    const ST4 = ['1 · cada nó multiplica x pelas suas 1024 linhas de gate e up: metade das 2048 saídas, sem conversar com o outro', '2 · cada nó multiplica a sua metade de h pelas suas 1024 colunas de down: uma soma parcial das 4096 saídas', '3 · all-reduce: os nós trocam as parciais pela rede e somam — uma vez por camada MoE'];
    V.loop(cv4, (g, w, h, t) => {
      const C = V.C(), cA = C.a, cB = C.b;
      V.text(g, 'uma camada de expert da GLM em TP=2', 6, 16, { size: 12.5, weight: 600, color: C.ink, maxW: w - 12 });
      V.text(g, 'formas [saída × entrada], como no PyTorch', 6, 32, { size: 11, color: C.muted, maxW: w - 12 });
      const u = V.clamp((w - 150) / 7.4, 12, 26), top = 58;
      const total = 6 * u + 124;
      const gX = Math.max(6, (w - total) / 2), gY = top + u, gW = 4 * u, gH = 2 * u;
      const hX = gX + gW + 20, hY = gY, hW = 8, hH = 2 * u;
      const dX = hX + hW + 20, dY = top, dW = 2 * u, dH = 4 * u;
      const pAX = dX + dW + 18, pBX = pAX + 12, pW = 8, yX = pBX + pW + 30;
      const P = 6, ph = (t % P) / P, stg = ph < 0.33 ? 0 : ph < 0.66 ? 1 : 2;
      const f1 = stg > 0 ? 1 : V.ease.inOut(ph / 0.33), f2 = stg > 1 ? 1 : stg < 1 ? 0 : V.ease.inOut((ph - 0.33) / 0.33), f3 = stg < 2 ? 0 : V.ease.inOut(Math.min(1, (ph - 0.66) / 0.26));
      const cell = (x, y, ww, hh, col, a) => { V.rr(g, x, y, ww, hh, 3); g.fillStyle = V.alpha(col, a); g.fill(); };
      // x (entrada, igual nos dois nós) sobre o gate/up
      cell(gX, gY - 13, gW, 6, C.ink, stg === 0 ? 0.55 : 0.25);
      V.text(g, 'x', gX - 5, gY - 7, { size: 11, font: 'mono', weight: 600, align: 'right', color: C.ink });
      // gate/up [2048 × 4096]: metade de cima no nó A, metade de baixo no nó B (corte pelas saídas)
      cell(gX, gY, gW, gH / 2 - 1, cA, 0.5); cell(gX, gY + gH / 2 + 1, gW, gH / 2 - 1, cB, 0.5);
      if (stg === 0) { g.fillStyle = V.alpha(C.ink, 0.55); const r = f1 * (gH / 2 - 4); g.fillRect(gX, gY + r, gW, 2.5); g.fillRect(gX, gY + gH / 2 + 1 + r, gW, 2.5); }
      // h [2048]: cada nó preenche a sua metade
      cell(hX, hY, hW, hH, C.muted, 0.12);
      cell(hX, hY, hW, (hH / 2 - 1) * f1, cA, 0.85); cell(hX, hY + hH / 2 + 1, hW, (hH / 2 - 1) * f1, cB, 0.85);
      V.text(g, 'h', hX + hW / 2, hY - 7, { size: 11, font: 'mono', weight: 600, align: 'center', color: C.ink });
      V.arrow(g, gX + gW + 3, gY + gH / 2, hX - 3, hY + hH / 2, { color: V.alpha(C.muted, 0.6), lw: 1.2, head: 5 });
      // down [4096 × 2048]: colunas da esquerda no nó A, da direita no nó B (corte pelas entradas)
      cell(dX, dY, dW / 2 - 1, dH, cA, 0.5); cell(dX + dW / 2 + 1, dY, dW / 2 - 1, dH, cB, 0.5);
      if (stg === 1) { g.fillStyle = V.alpha(C.ink, 0.55); const c = f2 * (dW / 2 - 4); g.fillRect(dX + c, dY, 2.5, dH); g.fillRect(dX + dW / 2 + 1 + c, dY, 2.5, dH); }
      V.arrow(g, hX + hW + 3, hY + hH / 2, dX - 3, dY + dH / 2, { color: V.alpha(C.muted, 0.6), lw: 1.2, head: 5 });
      // somas parciais y_A e y_B [4096]
      cell(pAX, dY, pW, dH, C.muted, 0.12); cell(pBX, dY, pW, dH, C.muted, 0.12);
      cell(pAX, dY, pW, dH * f2, cA, 0.85); cell(pBX, dY, pW, dH * f2, cB, 0.85);
      V.text(g, 'A', pAX + pW / 2, dY - 6, { size: 10.5, font: 'mono', weight: 600, align: 'center', color: cA });
      V.text(g, 'B', pBX + pW / 2, dY - 6, { size: 10.5, font: 'mono', weight: 600, align: 'center', color: cB });
      V.arrow(g, dX + dW + 3, dY + dH / 2, pAX - 3, dY + dH / 2, { color: V.alpha(C.muted, 0.6), lw: 1.2, head: 5 });
      // all-reduce: troca pela rede e soma → y inteiro nos dois nós
      V.arrow(g, pBX + pW + 4, dY + dH / 2, yX - 4, dY + dH / 2, { color: stg === 2 ? C.d : V.alpha(C.d, 0.5), lw: stg === 2 ? 2 : 1.3, head: 6 });
      if (stg === 2) for (let q = 0; q < 5; q++) { const k = (f3 * 1.4 + q / 5) % 1; const yy = dY + (q + 0.5) * dH / 5; V.dot(g, V.lerp(pAX + pW / 2, pBX + pW / 2, q % 2 ? k : 1 - k), yy, 2.4, C.d); }
      cell(yX, dY, pW, dH, C.muted, 0.12);
      for (let q = 0; q < 8; q++) { const hh = dH / 8; if (q / 8 < f3) cell(yX, dY + q * hh + 0.5, pW, hh - 1, q % 2 ? cB : cA, 0.85); }
      V.text(g, 'y', yX + pW / 2, dY - 6, { size: 11, font: 'mono', weight: 600, align: 'center', color: C.ink });
      // rótulos
      const lab = (x, y0, a) => a.forEach(([s, c, wgt], i) => V.text(g, s, x, y0 + i * 13, { size: i ? 10.5 : 11, font: i === 1 ? 'mono' : 'sans', weight: wgt || 400, align: 'center', color: c }));
      lab(gX + gW / 2, gY + gH + 16, [['gate, up', C.ink, 600], ['[2048 × 4096]', C.muted], ['corta as saídas', C.muted]]);
      lab(dX + dW / 2, dY + dH + 16, [['down', C.ink, 600], ['[4096 × 2048]', C.muted], ['corta as entradas', C.muted]]);
      V.text(g, 'all-reduce', (pAX + yX + pW) / 2, dY + dH + 16, { size: 11, weight: 600, align: 'center', color: C.d });
      V.text(g, 'y = A + B', yX + pW / 2, dY + dH + 29, { size: 10.5, font: 'mono', align: 'center', color: C.muted });
      wrapText(g, ST4[stg], 6, dY + dH + 66, w - 12, 15, { size: 11.5, color: stg === 2 ? C.d : C.ink });
      [[cA, 'Nó A'], [cB, 'Nó B']].forEach(([c, l], i) => { const lx = 6 + i * 70; V.rr(g, lx, h - 18, 12, 10, 2); g.fillStyle = c; g.fill(); V.text(g, l, lx + 17, h - 9, { size: 11, color: C.muted }); });
    });

    /* ================================================================
       Figura 5 — as receitas lado a lado (pilhas de camadas reais)
       ================================================================ */
    const f5 = root.querySelector('#fig-lado');
    const ctl5 = UI.controls(f5);
    const S5 = { mode: 'mix', mimo: true, spec: false };
    const mk = (n, fn) => Array.from({ length: n }, (_, i) => fn(i));
    const GLM_TAPS = [5, 14, 24, 33, 42], DS_TAPS = [40, 41, 42], MIMO_GA = [0, 5, 11, 17, 23, 29, 35, 41, 47], MIMO_TAPS = [0, 11, 23, 35, 47];
    const ROWS5 = [
      { k: 'glm', name: 'GLM-5.3-Flash', sub: '45 camadas + MTP',
        L: mk(45, (i) => { const sp = i % 4 === 3; return { mix: sp ? 'sparse' : 'lin', ffn: i < 3 ? 'dense' : 'moe', kv: sp ? 'grow' : 'state', kvw: sp ? 689 : 0, tap: GLM_TAPS.includes(i), d: sp ? 'MLA sem RoPE + indexador DSA (top-2048)' : 'KDA — atenção linear, estado 32×128×128 por GPU', f: i < 3 ? 'MLP densa SwiGLU (12288)' : 'MoE: 288 experts, top-8 + 1 compartilhado' }; }),
        spec: [{ n: 'DFlash2', c: 5, d: 'DFlash2: 5 camadas, rascunho em bloco, k=7' }],
        kvB: 7579, kvf: '11 × (656 + 33 do indexador)', fix: 'estado KDA ≈ 76 MiB por conversa', sum: '34 lineares · 11 esparsas · 3 densas + 42 MoE' },
      { k: 'qwen', name: 'Qwen3.8-Flash-Next', sub: '48 camadas + MTP',
        L: mk(48, (i) => { const sp = (i + 1) % 4 === 0; return { mix: sp ? 'sparse' : 'lin', ffn: 'moe', kv: sp ? 'grow' : 'state', kvw: sp ? 576 : 0, ple: i === 1, d: sp ? 'QSA — atenção esparsa (≤ 2048 tokens)' : 'Gated DeltaNet — linear, estado 128×128 por cabeça', f: 'MoE: 512 experts, top-10 + 1 compartilhado' + (i === 1 ? ' · + N-gram embedding' : '') }; }),
        spec: [{ n: 'MTP', c: 1, d: 'MTP: 1 camada (QSA + MoE), k=3' }],
        kvB: 7488, kvf: '13 × (512 + ~64 do indexador, estimado)', fix: 'estado GDN ≈ 27 MiB (bf16) por conversa · pool medido: 9.418 B/token/GPU efetivos', sum: '36 lineares · 12 esparsas · 48 MoE · N-gram na camada de índice 1' },
      { k: 'ds', name: 'DeepSeek-V4-Flash', sub: '43 camadas + DSpark',
        L: mk(43, (i) => { const ty = i < 2 ? 'swa' : i === 2 || i % 2 === 0 ? 'sparse' : 'hca'; return { mix: ty, ffn: i < 3 ? 'hash' : 'moe', kv: ty === 'swa' ? 'fixed' : 'grow', kvw: ty === 'sparse' ? 179 : ty === 'hca' ? 4.6 : 0, tap: DS_TAPS.includes(i), d: ty === 'swa' ? 'SWA pura (janela de 128)' : ty === 'sparse' ? 'CSA (C4A): KV ÷ 4 + indexador top-512 + janela 128' : 'HCA (C128A): KV ÷ 128, densa + janela 128', f: i < 3 ? 'MoE com roteamento por hash (id do token)' : 'MoE: 256 experts, top-6 + 1 compartilhado' }; }),
        spec: [{ n: 'DSpark', c: 3, d: 'DSpark: 3 estágios (SWA + MoE) sobre as camadas 40–42, k=6' }],
        kvB: 3850, kvf: '21 × (146 + 33 do indexador) + 20 × 4,6', fix: 'janelas SWA ≈ 3,4 MB por conversa', sum: '2 SWA · 21 CSA · 20 HCA · 43 MoE (3 por hash)' },
      { k: 'mimo', name: 'MiMo-V2.6-Flash', sub: '48 camadas · receita do zero', opt: true,
        L: mk(48, (i) => { const ga = MIMO_GA.includes(i); return { mix: ga ? 'full' : 'swa', ffn: i === 0 ? 'dense' : 'moe', kv: ga ? 'grow' : 'fixed', kvw: ga ? 640 : 0, tap: MIMO_TAPS.includes(i), d: ga ? 'atenção global: 4 cabeças de KV, K=192 / V=128' : 'SWA de 128 tokens (8 cabeças de KV, com sinks)', f: i === 0 ? 'FFN densa' : 'MoE: 256 experts, top-8, sem compartilhado' }; }),
        spec: [{ n: 'MTP', c: 3, d: 'MTP: 3 camadas' }, { n: 'DFlash', c: 5, d: 'DFlash: 5 camadas SWA, bloco 8' }],
        kvB: 5760, kvf: '9 × 640', fix: 'janelas SWA ≈ 10–13 MB por conversa', sum: '39 SWA · 9 globais · 1 densa + 47 MoE' },
    ];
    wrapCtl(UI.seg(ctl5, { label: 'Colorir por', options: [{ v: 'mix', label: 'mixer de tokens' }, { v: 'ffn', label: 'FFN' }, { v: 'kv', label: 'KV por token' }], value: 'mix', onChange: (v) => { S5.mode = v; } }));
    UI.toggle(ctl5, { label: 'mostrar MiMo', value: true, onChange: (v) => { S5.mimo = v; } });
    UI.toggle(ctl5, { label: 'rascunhadores e derivações', value: false, onChange: (v) => { S5.spec = v; } });
    const stage5 = f5.querySelector('.la-fig-stage');
    const fitH5 = () => { const w = stage5.clientWidth || 800; stage5.style.height = (w < 700 ? 540 : 440) + 'px'; };
    fitH5();
    const MIXC = (C) => ({ lin: C.a, sparse: C.c, hca: C.d, swa: C.b, full: C.f });
    const FFNC = (C) => ({ dense: C.b, moe: C.d, hash: C.c });
    const KVC = (C) => ({ grow: C.c, state: C.a, fixed: C.b });
    V.fig(f5, (g, w, h, t, dt, S, H) => {
      const C = V.C();
      const rows = ROWS5.filter((r) => !r.opt || S5.mimo);
      const wide = w >= 700;
      const labW = wide ? 158 : 0;
      const xL = 14 + labW, xR = w - 14;
      const specN = (r) => (S5.spec ? r.spec.reduce((a, s) => a + s.c + 1, 0) : 0);
      const maxN = Math.max(...rows.map((r) => r.L.length + specN(r)));
      const cw = (xR - xL) / maxN;
      const items = S5.mode === 'mix' ? [[C.a, 'linear/recorrente (KDA, GDN)'], [C.c, 'global esparsa (DSA, QSA, CSA)'], [C.d, 'comprimida densa (HCA ÷128)'], [C.b, 'janela deslizante (SWA)'], [C.f, 'global plena (MiMo)']]
        : S5.mode === 'ffn' ? [[C.b, 'FFN densa'], [C.d, 'MoE (roteador aprendido)'], [C.c, 'MoE com roteamento por hash']]
          : [[C.c, 'KV cresce com o contexto (altura ∝ bytes/camada)'], [C.a, 'estado recorrente fixo'], [C.b, 'janela fixa (últimos 128 tokens)']];
      if (S5.spec) items.push([C.ink, 'rascunhador (tracejado) · ▼ camada lida por ele']);
      const cf = (t * 0.1) % 1; // contexto cíclico 0 → 1M
      const ctxT = Math.round(cf * 1000000);
      const ctxS = `contexto ≈ ${nf(ctxT / 1000, 0)}k tokens`;
      // layout da legenda (quebra de linha) antes das linhas
      const leg = []; let lx = 14, lyk = 0;
      if (S5.mode === 'kv') { leg.push({ txt: ctxS, x: lx, k: 0 }); lx += tw(g, ctxS, 11.5, 600, 'mono') + 18; }
      items.forEach(([c, l]) => { const wl = tw(g, l, 11.5) + 30; if (lx + wl > w - 8) { lx = 14; lyk++; } leg.push({ c, l, x: lx, k: lyk }); lx += wl; });
      const legendH = 22 + lyk * 18;
      const topPad = S5.spec && wide ? 16 : 0; // espaço para os arcos de derivação da 1ª linha
      const rowH = (h - 12 - legendH - topPad) / rows.length;
      const colOf = (L) => (S5.mode === 'mix' ? MIXC(C)[L.mix] : S5.mode === 'ffn' ? FFNC(C)[L.ffn] : KVC(C)[L.kv]);
      let hover = null;
      const pt = H.pointer;
      rows.forEach((r, ri) => {
        const y0 = 8 + topPad + ri * rowH;
        const rc = recCol(C, r.k);
        const sy = wide ? y0 + 10 : y0 + 28, sh = Math.min(30, rowH * 0.33);
        if (wide) {
          V.text(g, r.name, 14, sy + 12, { size: 13, weight: 600, color: rc, maxW: labW - 6 });
          V.text(g, r.sub, 14, sy + 28, { size: 11, color: C.muted, maxW: labW - 6 });
        } else {
          V.text(g, r.name, 14, y0 + 14, { size: 12.5, weight: 600, color: rc });
          const nw = tw(g, r.name, 12.5, 600);
          V.text(g, '  ' + r.sub, 14 + nw, y0 + 14, { size: 11, color: C.muted, maxW: w - 28 - nw });
        }
        const n = r.L.length + specN(r);
        const tokP = ((t * 0.085 + ri * 0.23) % 1) * n;
        // camadas do alvo
        r.L.forEach((L, i) => {
          const x = xL + i * cw, glow = Math.max(0, 1 - Math.abs(i + 0.5 - tokP) / 1.6);
          V.rr(g, x, sy, Math.max(1.5, cw - Math.min(1.5, cw * 0.22)), sh, Math.min(3, cw / 3));
          g.fillStyle = V.alpha(colOf(L), 0.5 + 0.5 * glow); g.fill();
          if (L.ple && S5.mode !== 'kv') { V.text(g, '◆', x + cw / 2, sy + sh + 11, { size: 10, align: 'center', color: C.d }); if (wide) V.text(g, 'N-gram', x + cw / 2 + 8, sy + sh + 11, { size: 10, color: C.d }); }
          if (S5.spec && L.tap) { g.beginPath(); g.moveTo(x + cw / 2 - 4, sy - 7); g.lineTo(x + cw / 2 + 4, sy - 7); g.lineTo(x + cw / 2, sy - 1); g.closePath(); g.fillStyle = C.ink; g.fill(); }
          if (S5.mode === 'kv') {
            const kvTop = sy + sh + 3, kvH = Math.max(8, Math.min(24, rowH - sh - (wide ? 46 : 56)));
            if (L.kv === 'grow') { const hh = Math.max(1, kvH * cf * (L.kvw / 689)); V.rr(g, x + 0.5, kvTop, Math.max(1.2, cw - 2), hh, 1.5); g.fillStyle = V.alpha(C.c, 0.8); g.fill(); }
            else { V.rr(g, x + 0.5, kvTop, Math.max(1.2, cw - 2), 3, 1); g.fillStyle = V.alpha(L.kv === 'state' ? C.a : C.b, 0.8); g.fill(); }
          }
          if (pt.inside && pt.x >= x && pt.x < x + cw && pt.y >= sy - 8 && pt.y <= sy + sh + 8) hover = { r, i, L };
        });
        // rascunhadores
        if (S5.spec) {
          let k = r.L.length + 1;
          const specLabels = [];
          r.spec.forEach((s) => {
            const gx0 = xL + k * cw;
            for (let q = 0; q < s.c; q++) {
              const x = gx0 + q * cw, glow = Math.max(0, 1 - Math.abs(k + q + 0.5 - tokP) / 1.6);
              V.rr(g, x, sy + 3, Math.max(1.5, cw - Math.min(1.5, cw * 0.22)), sh - 6, Math.min(3, cw / 3));
              g.fillStyle = V.alpha(C.ink, 0.12 + 0.4 * glow); g.fill();
              g.setLineDash([3, 2]); g.strokeStyle = V.alpha(C.ink, 0.85); g.lineWidth = 1.2; g.stroke(); g.setLineDash([]);
              if (pt.inside && pt.x >= x && pt.x < x + cw && pt.y >= sy - 8 && pt.y <= sy + sh + 8) hover = { r, spec: s };
            }
            if (wide) specLabels.push([s.n, gx0 + (s.c * cw) / 2 + 2, sy - 4]);
            else if (S5.mode !== 'kv') specLabels.push([s.n, gx0 + (s.c * cw) / 2, sy + sh + 11]);
            // derivações: arcos das camadas "tapeadas" até o rascunhador
            if (wide && s.n !== 'MTP') r.L.forEach((L, i) => {
              if (!L.tap) return;
              const x1 = xL + i * cw + cw / 2, x2 = gx0 + 2; // chega na borda esquerda do bloco, longe do rótulo
              g.beginPath(); g.moveTo(x1, sy - 8); g.quadraticCurveTo((x1 + x2) / 2, sy - 8 - Math.min(16, rowH * 0.18), x2, sy - 2);
              g.strokeStyle = V.alpha(C.ink, 0.4); g.lineWidth = 1; g.stroke();
            });
            k += s.c + 1;
          });
          // rótulos por último, com halo, para nenhum arco passar por cima deles
          let prevR = -1e9;
          specLabels.forEach(([n, x, y]) => {
            const wl = tw(g, n, 10, 600), cx2 = V.clamp(x, wl / 2 + 2, w - wl / 2 - 2);
            const yy = cx2 - wl / 2 < prevR + 4 ? y + (y < sy ? -11 : 11) : y; // rótulos vizinhos não se encostam
            halo(g, n, cx2, yy, { size: 10, weight: 600, align: 'center', color: C.ink });
            prevR = cx2 + wl / 2;
          });
        }
        // bolinha do token
        const tx = xL + tokP * cw;
        V.glow(g, tx, sy + sh / 2, 12, C.ink, 0.6); V.dot(g, tx, sy + sh / 2, 2.6, C.ink);
        // linha de resumo
        const ly = sy + sh + (S5.mode === 'kv' ? Math.max(8, Math.min(24, rowH - sh - (wide ? 46 : 56))) + 16 : 26);
        if (S5.mode === 'kv') {
          const bytes = r.kvB * ctxT;
          V.text(g, `KV que cresce: ${nf(r.kvB)} B/token/GPU${wide ? ' (' + r.kvf + ')' : ''} → ${nf(ctxT / 1000, 0)}k tokens = ${nf(bytes / GiB, 2)} GiB`, xL, ly - 6, { size: 11, font: 'mono', color: C.ink, maxW: xR - xL });
          V.text(g, `fixo: ${r.fix}`, xL, ly + 8, { size: 11, color: C.muted, maxW: xR - xL });
        } else {
          V.text(g, r.sum, xL, ly, { size: 11, color: C.muted, maxW: xR - xL });
        }
      });
      // legenda
      const ly0 = h - legendH + 14;
      leg.forEach((it) => {
        const ly = ly0 + it.k * 18;
        if (it.txt) { V.text(g, it.txt, it.x, ly, { size: 11.5, font: 'mono', weight: 600, color: C.ink }); return; }
        V.rr(g, it.x, ly - 9, 12, 10, 3); g.fillStyle = it.c; g.fill();
        V.text(g, it.l, it.x + 17, ly, { size: 11.5, color: C.muted });
      });
      if (hover) {
        const lines = hover.spec ? [`${hover.r.name} · rascunhador`, hover.spec.d] : [`${hover.r.name} · camada ${hover.i}`, hover.L.d, hover.L.f].concat(hover.L.tap ? ['▼ lida pelo rascunhador'] : []);
        tip(g, lines, pt.x, pt.y, w, h);
      }
    });

    /* ================================================================
       Figura 6 — velocidades medidas, com recibo
       ================================================================ */
    const f6 = root.querySelector('#fig-medido');
    const stage6 = f6.querySelector('.la-fig-stage');
    const ctl6 = UI.controls(f6);
    const S6 = { mode: 'dec', mimo: false, tf: true, t0: 0 };
    const D6 = {
      dec: [
        { k: 'glm', v: 62.9, l: 'GLM · estruturado/código · DFlash2 k=7', d: '2026-08-28 · sparkDash · T=0' },
        { k: 'glm', v: 36.1, l: 'GLM · prosa · k adaptativo + FP8 denso', d: '2026-09-17 · sparkDash' },
        { k: 'glm', v: 65.1, l: 'GLM · estruturado · DFlash2 k=7 (lab)', d: '2026-08-30 · bench_decode · mediana 5×400' },
        { k: 'glm', v: 27.1, l: 'GLM · prosa · DFlash2 k=7 (lab)', d: '2026-08-30 · bench_decode · mesma corrida' },
        { k: 'glm', v: 24.6, l: 'GLM · rollback MTP k=2', d: '2026-08-30 · bench_decode' },
        { k: 'tf', v: 114.7, l: 'GLM·TF · estruturado · TensorFold', d: 'v1.3.2 · sparkDash · T=0 · clock ≤ 2.200 MHz', tf: true },
        { k: 'tf', v: 60.4, l: 'GLM·TF · prosa · TensorFold', d: 'v1.3.2 · sparkDash · T=0 · clock ≤ 2.200 MHz', tf: true },
        { k: 'qwen', v: 76.0, l: 'Qwen · estruturado · lane vLLM 0.30', d: '2026-09-27 · KV BF16' },
        { k: 'qwen', v: 59.2, l: 'Qwen · prosa · lane vLLM 0.30', d: '2026-09-27 · KV BF16' },
        { k: 'qwen', v: 52.1, l: 'Qwen · MTP=3 · lane day-0', d: '2026-09-05 · greedy' },
        { k: 'qwen', v: 24.5, l: 'Qwen · sem especulação', d: '2026-09-05 · greedy' },
        { k: 'ds', v: 75.4, l: 'DS 0731 · prompt de 256 · MTP-5', d: '2026-08-01' },
        { k: 'ds', v: 56, l: 'DS Vision-Exp · DSpark k=6', d: '2026-09-02 · mediana de 8 (51–67)' },
        { k: 'mimo', v: 70.5, l: 'MiMo · código · DFlash k=7', d: 'receita comunitária (vLLM TP=2)', opt: true },
        { k: 'mimo', v: 25.9, l: 'MiMo · prosa · DFlash k=7', d: 'receita comunitária (vLLM TP=2)', opt: true },
      ],
      agg: [
        { k: 'glm', v: 146.5, l: 'GLM · 4 usuários · estruturado', d: '2026-08-28 · sparkDash' },
        { k: 'glm', v: 75.3, l: 'GLM · 4 usuários · prosa', d: '2026-09-17 · sparkDash' },
        { k: 'tf', v: 227.9, l: 'GLM·TF · 4 usuários · estruturado', d: 'v1.3.2 · sparkDash · clock ≤ 2.200 MHz', tf: true },
        { k: 'tf', v: 108.8, l: 'GLM·TF · 4 usuários · prosa', d: 'v1.3.2 · sparkDash · clock ≤ 2.200 MHz', tf: true },
        { k: 'qwen', v: 313.6, l: 'Qwen · 8 usuários · código · lane 0.30', d: '2026-09-27' },
        { k: 'qwen', v: 216.4, l: 'Qwen · 8 usuários · prosa · lane 0.30', d: '2026-09-27' },
        { k: 'qwen', v: 207.0, l: 'Qwen · 8 usuários · prosa · lane day-0', d: 'KV fp8 + vocabulário 65k' },
        { k: 'ds', v: 191.2, l: 'DS 0731 · 6 usuários · prompt de 256', d: '2026-08-01' },
        { k: 'ds', v: 139, l: 'DS Vision-Exp · 6 usuários', d: '2026-09-02' },
        { k: 'mimo', v: 155.8, l: 'MiMo · 6 usuários', d: 'receita comunitária (vLLM TP=2)', opt: true },
      ],
    };
    const P6 = [
      { k: 'glm', l: 'GLM (2026-09-07)', pts: [[8221, 1492.1], [16411, 1553.7], [32797, 1428.2], [65566, 1587.0], [131101, 1561.7], [262173, 1516.8]] },
      { k: 'tf', l: 'GLM·TF (v1.3.2, clock ≤ 2.200 MHz)', tf: true, pts: [[8219, 1952.2], [16407, 1973.5], [32790, 1978.9], [65563, 1942.5], [131099, 1837.9], [262170, 1641.7], [981841, 1015]] },
      { k: 'qwen', l: 'Qwen (lane day-0)', pts: [[8192, 2874.9], [16384, 2962.1], [32768, 2962.2], [65536, 2893.4], [131072, 2727.1]] },
      { k: 'ds', l: 'DS 0731 (2026-08-01)', pts: [[2048, 2563], [8192, 1713], [32768, 1428], [131072, 1665], [900000, 874.8]] },
      { k: 'mimo', l: 'MiMo (comunidade)', opt: true, pts: [[2048, 1947], [32768, 1425], [131072, 937], [253952, 656]] },
    ];
    const col6 = (C, k) => (k === 'mimo' ? C.muted : recCol(C, k)); // MiMo (comunidade) em cinza nesta figura: o âmbar é a cor do selo GLM·TF
    const shown6 = (it) => (!it.opt || S6.mimo) && (!it.tf || S6.tf);
    const items6 = () => (S6.mode === 'pre' ? [] : D6[S6.mode].filter(shown6));
    const fitH6 = () => {
      const w = stage6.clientWidth || 700, wide = w >= 640;
      const need = S6.mode === 'pre' ? 420 : 30 + items6().length * (wide ? 36 : 52) + 30;
      stage6.style.height = Math.max(360, need) + 'px';
    };
    wrapCtl(UI.seg(ctl6, { label: 'Medida', options: [{ v: 'dec', label: '1 usuário' }, { v: 'agg', label: 'vários usuários' }, { v: 'pre', label: 'prefill × prompt' }], value: 'dec', onChange: (v) => { S6.mode = v; S6.t0 = H6.t; fitH6(); } }));
    UI.toggle(ctl6, { label: 'incluir GLM·TF (TensorFold)', value: true, onChange: (v) => { S6.tf = v; S6.t0 = H6.t; fitH6(); } });
    UI.toggle(ctl6, { label: 'incluir MiMo (comunidade)', value: false, onChange: (v) => { S6.mimo = v; S6.t0 = H6.t; fitH6(); } });
    const H6 = V.fig(f6, (g, w, h, t, dt, S, H) => {
      const C = V.C();
      const prog = V.ease.out(Math.min(1, (t - S6.t0) * 1.3));
      const pt = H.pointer;
      if (S6.mode !== 'pre') {
        const items = items6(), wide = w >= 640;
        const big = S6.mode === 'dec' && items.some((it) => it.v > 80); // o estruturado do TensorFold passa de 80
        const max = S6.mode === 'dec' ? (big ? 120 : 80) : 320, ticks = S6.mode === 'dec' ? (big ? [0, 30, 60, 90, 120] : [0, 20, 40, 60, 80]) : [0, 80, 160, 240, 320];
        const top = 14, rowH = wide ? 36 : 52, xL = 14, xR = w - 70;
        const yEnd = top + items.length * rowH;
        ticks.forEach((v) => { const x = xL + (v / max) * (xR - xL); V.line(g, x, top, x, yEnd, { color: C.grid, lw: 1 }); V.text(g, String(v), x, yEnd + 14, { size: 10.5, font: 'mono', align: 'center', color: C.muted }); });
        V.text(g, 'tok/s', xR + 26, yEnd + 14, { size: 10.5, font: 'mono', color: C.muted });
        let hov = null;
        items.forEach((it, i) => {
          const y = top + i * rowH, col = col6(C, it.k);
          V.text(g, it.l, xL, y + 13, { size: 12, weight: 600, color: C.ink, maxW: wide ? (xR - xL) * 0.62 : w - 28 });
          if (wide) V.text(g, it.d, xR + 56, y + 13, { size: 11, align: 'right', color: C.muted, maxW: (xR - xL) * 0.35 });
          const bw = (it.v / max) * (xR - xL) * prog;
          if (it.k === 'mimo') { g.setLineDash([4, 3]); V.box(g, xL, y + 19, Math.max(3, bw), 12, { r: 6, fill: V.alpha(col, 0.3), stroke: col, lw: 1.5 }); g.setLineDash([]); }
          else V.box(g, xL, y + 19, Math.max(3, bw), 12, { r: 6, fill: col });
          V.text(g, nf(it.v, 1), xL + bw + 6, y + 29.5, { size: 12, font: 'mono', weight: 600, color: C.ink });
          if (!wide) V.text(g, it.d, xL, y + 45, { size: 10.5, color: C.muted, maxW: w - 28 });
          if (pt.inside && pt.y >= y && pt.y < y + rowH) hov = it;
        });
        if (hov) tip(g, [hov.l, `${nf(hov.v, 1)} tok/s ${S6.mode === 'dec' ? 'por usuário' : 'somando todos os usuários'}`, hov.d], pt.x, pt.y, w, h);
        return;
      }
      // prefill × tamanho do prompt
      const lines = P6.filter(shown6);
      const xL = 50, xR = w - 18, yB = h - 40, legY = 34;
      const leg6 = []; let lgx = xL + 130, lgr = 0;
      lines.forEach((L) => { const wl = tw(g, L.l, 11.5) + 28; if (lgx + wl > w - 8) { lgx = xL; lgr++; } leg6.push({ L, x: lgx, y: legY + lgr * 16 }); lgx += wl; });
      const yT = 46 + lgr * 16;
      const X = (n) => xL + ((Math.log2(n) - 10) / 10) * (xR - xL), Y = (v) => yB - (v / 3600) * (yB - yT);
      [0, 1000, 2000, 3000].forEach((v) => { V.line(g, xL, Y(v), xR, Y(v), { color: C.grid, lw: 1 }); V.text(g, nf(v), xL - 6, Y(v) + 4, { size: 10.5, font: 'mono', align: 'right', color: C.muted }); });
      [[1024, '1k'], [4096, '4k'], [16384, '16k'], [65536, '64k'], [262144, '256k'], [1048576, '1M']].forEach(([n, l]) => { V.line(g, X(n), yB, X(n), yB + 4, { color: C.muted }); V.text(g, l, X(n), yB + 17, { size: 10.5, font: 'mono', align: 'center', color: C.muted }); });
      V.text(g, 'tokens do prompt (escala log) →', xR, yB + 33, { size: 10.5, align: 'right', color: C.muted });
      V.text(g, 'prefill frio, tok/s', xL, legY, { size: 11, font: 'mono', color: C.muted });
      const clipX = xL + prog * (xR - xL);
      g.save(); g.beginPath(); g.rect(0, 0, clipX + 6, h); g.clip();
      let best = null;
      lines.forEach((L) => {
        const col = col6(C, L.k);
        g.beginPath(); L.pts.forEach(([n, v], i) => (i ? g.lineTo(X(n), Y(v)) : g.moveTo(X(n), Y(v))));
        if (L.opt) g.setLineDash([7, 5]);
        g.strokeStyle = col; g.lineWidth = 2.5; g.stroke(); g.setLineDash([]);
        L.pts.forEach(([n, v]) => { V.dot(g, X(n), Y(v), 4, col); if (pt.inside) { const d = Math.hypot(pt.x - X(n), pt.y - Y(v)); if (d < 16 && (!best || d < best.d)) best = { d, L, n, v }; } });
      });
      g.restore();
      // legenda no topo (posições calculadas antes do eixo)
      leg6.forEach(({ L, x: lx, y: ly }) => {
        if (L.opt) { g.setLineDash([4, 3]); V.line(g, lx, ly - 4, lx + 13, ly - 4, { color: col6(C, L.k), lw: 2.5, dash: [4, 3] }); g.setLineDash([]); }
        else { V.rr(g, lx, ly - 9, 12, 10, 3); g.fillStyle = col6(C, L.k); g.fill(); }
        V.text(g, L.l, lx + 17, ly, { size: 11.5, color: C.muted });
      });
      if (best) tip(g, [best.L.l, `prompt ≈ ${ctxLbl(best.n)} tokens`, `${nf(best.v, 1)} tok/s de prefill`], pt.x, pt.y, w, h);
    });
    fitH6();

    /* ================================================================
       Figura 7 — MiMo: janela deslizante × atenção global; DiffKV
       ================================================================ */
    const f7 = root.querySelector('#fig-mimo');
    const ctl7 = UI.controls(f7);
    const S7 = { lg: 17, cmp: false, panel: 'kv', t0: 0 };
    wrapCtl(UI.seg(ctl7, { label: 'Painel', options: [{ v: 'kv', label: 'KV × contexto' }, { v: 'diff', label: 'DiffKV: K ≠ V' }], value: 'kv', onChange: (v) => { S7.panel = v; S7.t0 = H7.t; } }));
    wrapCtl(UI.slider(ctl7, { label: 'Contexto', min: 10, max: 20, step: 1, value: 17, fmt: (v) => ctxLbl(2 ** v), onInput: (v) => { S7.lg = v; } }));
    UI.toggle(ctl7, { label: 'se as 48 fossem globais (ilustrativo)', value: false, onChange: (v) => { S7.cmp = v; } });
    const ro7 = UI.readout(ctl7, 'KV por conversa / GPU', '—');
    let lastRo = '';
    const SWA_FIX = 11.5e6; // ~10–13 MB (tutorial), ponto médio só para desenhar
    const qv = V.randMatrix(1, 12, 31, 1)[0], kv = V.randMatrix(1, 12, 32, 1)[0], vv = V.randMatrix(1, 8, 33, 1)[0];
    const H7 = V.fig(f7, (g, w, h, t) => {
      const C = V.C();
      const ctx = 2 ** S7.lg;
      const ga = 5760 * ctx, total = ga + SWA_FIX, hyp = 55680 * ctx;
      const roTxt = (total >= GiB ? nf(total / GiB, 2) + ' GiB' : nf(total / 1e6, 0) + ' MB');
      if (roTxt !== lastRo) { ro7.set(roTxt); lastRo = roTxt; }
      // --- fita de contexto
      const xL = 16, xR = w - 26, ry = 70, rh = 18;
      const ncell = Math.max(24, Math.min(64, Math.floor((xR - xL) / 9)));
      const cwid = (xR - xL) / ncell;
      V.text(g, `fita de contexto: ${ctxLbl(ctx)} tokens (cada célula ≈ ${tokLbl(ctx / ncell)} tokens)`, xL, 18, { size: 11.5, font: 'mono', color: C.muted, maxW: xR - xL });
      const shift = (t * 1.6) % 1;
      g.save(); g.beginPath(); g.rect(xL, ry - 2, xR - xL, rh + 4); g.clip();
      for (let c = -1; c < ncell; c++) {
        const x = xL + (c + 1 - shift) * cwid;
        V.rr(g, x + 1, ry, cwid - 2, rh, 3); g.fillStyle = V.alpha(C.muted, ((c + Math.floor(t * 1.6)) % 2 ? 0.16 : 0.26)); g.fill();
      }
      g.restore();
      const nx = xR + 10, ny = ry + rh / 2;
      V.glow(g, nx, ny, 14, C.c); V.dot(g, nx, ny, 4.5, C.c);
      V.text(g, 'token novo', nx, ry - 26, { size: 10.5, align: 'right', color: C.c });
      const winW = Math.max(3, ((xR - xL) * 128) / ctx), wx = xR - winW;
      V.rr(g, wx, ry - 3, winW, rh + 6, 3); g.fillStyle = V.alpha(C.b, 0.55); g.fill(); g.strokeStyle = C.b; g.lineWidth = 1.5; g.stroke();
      // SWA: arcos curtos acima
      for (let q = 0; q < 3; q++) { const x2 = wx + (winW * (q + 0.5)) / 3; g.beginPath(); g.moveTo(nx, ry - 2); g.quadraticCurveTo((nx + x2) / 2, ry - 16 - q * 3, x2, ry - 3); g.strokeStyle = V.alpha(C.b, 0.9); g.lineWidth = 1.3; g.stroke(); }
      V.text(g, '39 camadas SWA: só os últimos 128 tokens', xL, ry - 10, { size: 11.5, weight: 600, color: C.b, maxW: Math.max(120, wx - xL - 60) });
      // global: arcos para todo o contexto, abaixo
      const na = 14;
      for (let q = 0; q < na; q++) {
        const x2 = xL + ((q + 0.5) / na) * (xR - xL), a = 0.25 + 0.6 * Math.max(0, Math.sin(t * 2.2 - q * 0.45));
        g.beginPath(); g.moveTo(nx, ry + rh + 2); g.quadraticCurveTo((nx + x2) / 2, ry + rh + 16 + (nx - x2) * 0.09, x2, ry + rh + 3);
        g.strokeStyle = V.alpha(C.f, a); g.lineWidth = 1.2; g.stroke();
      }
      V.text(g, '9 camadas globais: todos os tokens', xL, ry + rh + 56, { size: 11.5, weight: 600, color: C.f });
      V.line(g, 12, 158, w - 12, 158, { color: C.grid, lw: 1 });
      const e = V.ease.out(Math.min(1, (t - S7.t0) * 2));
      g.globalAlpha = e;
      if (S7.panel === 'kv') {
        const bars = [
          { l: '9 camadas globais · 5.760 B × contexto', v: ga, c: C.f },
          { l: '39 camadas SWA · ~10–13 MB fixos', v: SWA_FIX, c: C.b },
          { l: 'total por conversa (por GPU, FP8)', v: total, c: C.c },
        ];
        if (S7.cmp) bars.push({ l: 'se as 48 fossem globais · 55.680 B × contexto', v: hyp, c: C.muted });
        const mx = Math.max(...bars.map((b) => b.v));
        const bx = 16, bw = w - 32, y0 = 176, step = Math.min(52, (h - y0 - 14) / bars.length);
        bars.forEach((b, i) => {
          const y = y0 + i * step;
          V.text(g, b.l, bx, y, { size: 11.5, color: C.ink, maxW: bw * 0.7 });
          const val = b.v >= GiB ? nf(b.v / GiB, 2) + ' GiB' : nf(b.v / 1e6, 1) + ' MB';
          V.text(g, val, bx + bw, y, { size: 12, font: 'mono', weight: 600, align: 'right', color: b.c === C.muted ? C.ink : b.c });
          V.box(g, bx, y + 7, bw, 13, { r: 6, fill: V.alpha(C.muted, 0.1) });
          V.box(g, bx, y + 7, Math.max(3, (b.v / mx) * bw), 13, { r: 6, fill: V.alpha(b.c, b.c === C.muted ? 0.5 : 0.85) });
        });
      } else {
        // DiffKV: q e k com 192 dims (12 blocos de 16), v com 128 (8 blocos)
        const cs = Math.max(14, Math.min(24, (w - 40) / 13)), gp = 3, bx = 16;
        const P = 5, ph = ((t - S7.t0) % P) / P, jj = Math.min(11, Math.floor((ph / 0.55) * 12)), sweep = ph < 0.55;
        const strip = (lab, vals, y, col, hi, rope) => {
          V.text(g, lab, bx, y - 6, { size: 11.5, font: 'mono', color: C.ink, maxW: w - 32 });
          vals.forEach((v, i) => {
            const x = bx + i * (cs + gp);
            V.rr(g, x, y, cs, cs, 4); g.fillStyle = V.alpha(rope && i < 4 ? C.d : col, 0.3 + 0.5 * Math.min(1, Math.abs(v))); g.fill();
            if (hi === i) { g.strokeStyle = C.ink; g.lineWidth = 2; V.rr(g, x - 1.5, y - 1.5, cs + 3, cs + 3, 5); g.stroke(); }
          });
        };
        const nar = w < 560;
        strip(nar ? 'q · 192 dims (64 com RoPE)' : 'q · 192 dims (12 blocos de 16; 64 com RoPE)', qv, 180, C.b, sweep ? jj : -1, true);
        strip(nar ? 'k · 192 dims (por token)' : 'k · 192 dims (um por token passado)', kv, 222, C.a, sweep ? jj : -1, true);
        let dot = 0; for (let i = 0; i <= (sweep ? jj : 11); i++) dot += qv[i] * kv[i];
        const sx2 = bx + 12 * (cs + gp) + 10;
        if (sx2 + 90 < w) V.text(g, `q·k = ${nf(dot / Math.sqrt(12), 2)}`, sx2, 222 + cs / 2 + 4, { size: 12, font: 'mono', weight: 600, color: sweep ? C.ink : C.a });
        else V.text(g, `q·k = ${nf(dot / Math.sqrt(12), 2)}`, w - 16, 264, { size: 12, font: 'mono', weight: 600, align: 'right', color: sweep ? C.ink : C.a });
        const oPh = sweep ? 0 : Math.min(1, (ph - 0.55) / 0.35);
        strip('v · 128 dims (8 blocos)', vv, 270, C.f, -1, false);
        strip('saída o = Σ softmax(q·k)·v · 128 dims', vv.map((v, i) => (i < oPh * 8 ? v * 0.8 : 0)), 314, C.c, -1, false);
        const yE = wrapText(g, 'kernels que assumem head_size_v = head_size não servem → no GB10, Triton DiffKV', bx, h - (nar ? 46 : 30), w - 32, 15, { size: 11.5, color: C.muted });
        V.text(g, '(esquema; valores ilustrativos)', bx, yE + 16, { size: 10.5, color: C.muted });
      }
      g.globalAlpha = 1;
    });

    /* ================================================================
       Figura 8 — o caminho de uma pergunta ao copilot
       ================================================================ */
    const f8 = root.querySelector('#fig-copilot');
    const ctl8 = UI.controls(f8);
    const PROV = [
      { v: 'mlx', label: 'MLX local (Mac)', base: 'http://127.0.0.1:8080/v1', srv: 'mlx_lm.server', sub: 'no seu Mac · Apple Silicon', ctx: 9000, tf: 'max_tokens', noThink: true, note: 'Modelo vazio = usa o <code>--model</code> já carregado. CORS liberado por padrão; o raciocínio chega em <code>delta.reasoning</code>. Nada sai da sua máquina.' },
      { v: 'vllm', label: 'Sua receita (vLLM)', base: 'http://SEU-HEAD:8888/v1', srv: 'vllm serve', sub: '2× DGX Spark · porta 8888', ctx: 24000, tf: 'max_tokens', noThink: false, note: 'O mesmo contrato das receitas. Para o navegador falar com outra máquina, o vLLM precisa de <code>--allowed-origins</code> (CORS); se a receita usa <code>--api-key</code>, cole a chave.' },
      { v: 'openai', label: 'OpenAI', base: 'https://api.openai.com/v1', srv: 'API na nuvem', sub: 'requer API key', ctx: 24000, tf: 'max_completion_tokens', noThink: false, note: 'Precisa de API key. Modelos de raciocínio usam <code>max_completion_tokens</code> e podem rejeitar temperatura — o copilot tenta de novo sem os campos rejeitados.' },
      { v: 'ollama', label: 'Ollama', base: 'http://localhost:11434/v1', srv: 'Ollama', sub: 'porta 11434', ctx: 9000, tf: 'max_tokens', noThink: true, note: 'Bloqueia a origem nula de <code>file://</code>: sirva a plataforma com <code>./serve.sh</code> ou defina <code>OLLAMA_ORIGINS</code>.' },
      { v: 'lmstudio', label: 'LM Studio', base: 'http://localhost:1234/v1', srv: 'LM Studio', sub: 'porta 1234', ctx: 9000, tf: 'max_tokens', noThink: true, note: 'Ligue o servidor na aba Developer e habilite CORS nas configurações.' },
    ];
    const S8 = { p: PROV[0], ctx: 'chapter', think: false, t0: 0 };
    wrapCtl(UI.select(ctl8, { label: 'Provedor', options: PROV.map((p) => ({ v: p.v, label: p.label })), value: 'mlx', onChange: (v) => { S8.p = PROV.find((p) => p.v === v) || PROV[0]; tg8.set(!S8.p.noThink); upd8(); } }));
    wrapCtl(UI.seg(ctl8, { label: 'Contexto', options: [{ v: 'section', label: 'Seção' }, { v: 'chapter', label: 'Capítulo' }, { v: 'none', label: 'Sem contexto' }], value: 'chapter', onChange: (v) => { S8.ctx = v; upd8(); } }));
    const tg8 = UI.toggle(ctl8, { label: 'thinking ligado', value: false, onChange: (v) => { S8.think = v; S8.t0 = H8 ? H8.t : 0; upd8(); } });
    const code8 = LA.h('code', { style: { display: 'block', whiteSpace: 'pre-wrap', wordBreak: 'break-all', fontSize: '12px', lineHeight: '1.5' } });
    const note8 = LA.h('div', { style: { fontSize: '13px', lineHeight: '1.5', color: 'var(--text-2)', marginTop: '6px' } });
    ctl8.appendChild(LA.h('div', { style: { flexBasis: '100%', minWidth: '0' } }, code8, note8));
    function upd8() {
      const p = S8.p;
      const sys = S8.ctx === 'none' ? '"(instruções do tutor, sem página)"' : `"(tutor + trilha + ${S8.ctx === 'section' ? 'seção visível' : 'capítulo'}, até ${nf(p.ctx)} caracteres)"`;
      code8.textContent = `POST ${p.base}/chat/completions\n{"messages":[{"role":"system","content":${sys}},{"role":"user","content":"(sua pergunta)"}],\n "stream":true,"${p.tf}":2048,"temperature":0.4${S8.think ? '' : ',"chat_template_kwargs":{"enable_thinking":false}'}}`;
      note8.innerHTML = p.note;
    }
    let H8 = null;
    H8 = V.fig(f8, (g, w, h, t) => {
      const C = V.C();
      const p = S8.p, wide = w >= 640;
      let Pg, Cp, Sv;
      if (wide) { Pg = [14, 30, w * 0.22, h - 110]; Cp = [w * 0.3, 30, w * 0.28, h - 110]; Sv = [w * 0.72, 30, w * 0.28 - 14, h - 110]; }
      else { Pg = [14, 12, w - 28, 70]; Cp = [14, 112, w - 28, 116]; Sv = [14, 262, w - 28, 64]; }
      const P = 7, ph = ((t - S8.t0) % P) / P;
      // página
      V.box(g, Pg[0], Pg[1], Pg[2], Pg[3], { r: 12, fill: V.alpha(C.muted, 0.06), stroke: V.alpha(C.muted, 0.4), lw: 1.2 });
      V.text(g, 'Página (capítulo)', Pg[0] + 12, Pg[1] + 20, { size: 12, weight: 600, color: C.ink });
      const nl = wide ? 9 : 3;
      for (let i = 0; i < nl; i++) {
        const y = Pg[1] + (wide ? 52 : 32) + i * (wide ? 18 : 11), lw = (Pg[2] - 24) * (0.55 + 0.4 * Math.abs(Math.sin(i * 1.9)));
        const on = S8.ctx === 'chapter' || (S8.ctx === 'section' && (wide ? i >= 3 && i <= 5 : i === 1));
        V.rr(g, Pg[0] + 12, y, lw, wide ? 7 : 5, 3); g.fillStyle = on ? V.alpha(C.a, 0.6) : V.alpha(C.muted, 0.22); g.fill();
      }
      const mLbl = S8.ctx === 'none' ? 'sem contexto' : S8.ctx === 'section' ? '✦ seção visível' : '✦ texto inteiro';
      if (wide) V.text(g, mLbl, Pg[0] + 12, Pg[1] + 38, { size: 11, color: S8.ctx === 'none' ? C.muted : C.a, maxW: Pg[2] - 24 });
      else V.text(g, mLbl, Pg[0] + Pg[2] - 12, Pg[1] + 20, { size: 11, align: 'right', color: S8.ctx === 'none' ? C.muted : C.a });
      // copilot
      V.box(g, Cp[0], Cp[1], Cp[2], Cp[3], { r: 12, fill: V.alpha(C.b, 0.08), stroke: C.b, lw: 1.5 });
      V.text(g, 'Copilot', Cp[0] + 12, Cp[1] + 20, { size: 12.5, weight: 600, color: C.ink });
      V.text(g, '⌘/Ctrl J', Cp[0] + Cp[2] - 12, Cp[1] + 20, { size: 11, font: 'mono', align: 'right', color: C.muted });
      const sysW = (Cp[2] - 24) * (S8.ctx === 'none' ? 0.18 : S8.ctx === 'section' ? 0.45 : 0.9) * Math.min(1, p.ctx / 24000 + 0.45);
      V.box(g, Cp[0] + 12, Cp[1] + 30, Math.max(40, sysW), 16, { r: 5, fill: V.alpha(C.a, 0.3) });
      V.text(g, 'system', Cp[0] + 18, Cp[1] + 42, { size: 10.5, font: 'mono', color: C.ink });
      V.box(g, Cp[0] + 12, Cp[1] + 50, 110, 16, { r: 5, fill: V.alpha(C.c, 0.3) });
      V.text(g, 'user: pergunta', Cp[0] + 18, Cp[1] + 62, { size: 10.5, font: 'mono', color: C.ink });
      // resposta sendo escrita
      const streamP = ph < 0.38 ? 0 : Math.min(1, (ph - 0.38) / 0.55);
      const thinkShare = S8.think ? 0.7 : 0;
      const ansY = Cp[1] + 74, ansH = Cp[3] - 82, nls = Math.max(1, Math.floor(ansH / 10));
      for (let i = 0; i < nls; i++) {
        const f = (i + 1) / nls, isThink = f <= thinkShare;
        if (streamP < f - 1 / nls) break;
        const part = Math.min(1, (streamP - (f - 1 / nls)) * nls);
        V.rr(g, Cp[0] + 12, ansY + i * 10, (Cp[2] - 24) * (0.5 + 0.45 * Math.abs(Math.sin(i * 2.3))) * part, 6, 3);
        g.fillStyle = isThink ? V.alpha(C.b, 0.55) : V.alpha(C.a, 0.8); g.fill();
      }
      // servidor
      V.box(g, Sv[0], Sv[1], Sv[2], Sv[3], { r: 12, fill: V.alpha(C.d, 0.08), stroke: C.d, lw: 1.5 });
      V.text(g, p.label, Sv[0] + 12, Sv[1] + 20, { size: 12.5, weight: 600, color: C.ink, maxW: Sv[2] - 24 });
      if (wide) {
        V.text(g, p.srv, Sv[0] + 12, Sv[1] + 38, { size: 11, font: 'mono', color: C.ink, maxW: Sv[2] - 24 });
        V.text(g, p.sub, Sv[0] + 12, Sv[1] + 55, { size: 11, color: C.muted, maxW: Sv[2] - 24 });
        V.text(g, p.base, Sv[0] + 12, Sv[1] + 74, { size: 11, font: 'mono', color: C.d, maxW: Sv[2] - 24 });
        V.text(g, 'GET /v1/models', Sv[0] + 12, Sv[1] + 100, { size: 10.5, font: 'mono', color: C.muted });
        V.text(g, 'POST /v1/chat/completions', Sv[0] + 12, Sv[1] + 116, { size: 10.5, font: 'mono', color: C.muted, maxW: Sv[2] - 24 });
      } else {
        V.text(g, p.srv + ' · ' + p.sub, Sv[0] + 12, Sv[1] + 37, { size: 11, color: C.muted, maxW: Sv[2] - 24 });
        V.text(g, p.base, Sv[0] + 12, Sv[1] + 54, { size: 11, font: 'mono', color: C.d, maxW: Sv[2] - 24 });
      }
      if (ph > 0.3 && ph < 0.4) { const a0 = t * 6; g.beginPath(); g.arc(Sv[0] + Sv[2] - 22, Sv[1] + 22, 8, a0, a0 + 4.2); g.strokeStyle = C.d; g.lineWidth = 2; g.stroke(); }
      // setas e partículas
      const pOut = wide ? [Pg[0] + Pg[2], Pg[1] + 70] : [w / 2, Pg[1] + Pg[3]];
      const cIn = wide ? [Cp[0], Cp[1] + 70] : [w / 2, Cp[1]];
      const cOut = wide ? [Cp[0] + Cp[2], Cp[1] + 50] : [w * 0.35, Cp[1] + Cp[3]];
      const sIn = wide ? [Sv[0], Sv[1] + 50] : [w * 0.35, Sv[1]];
      const sOut = wide ? [Sv[0], Sv[1] + Sv[3] - 40] : [w * 0.65, Sv[1]];
      const cIn2 = wide ? [Cp[0] + Cp[2], Cp[1] + Cp[3] - 40] : [w * 0.65, Cp[1] + Cp[3]];
      V.arrow(g, pOut[0] + 4, pOut[1], cIn[0] - 4, cIn[1], { color: V.alpha(C.a, S8.ctx === 'none' ? 0.2 : 0.6), lw: 1.4, head: 7 });
      V.arrow(g, cOut[0] + 4, cOut[1], sIn[0] - 4, sIn[1], { color: V.alpha(C.c, 0.7), lw: 1.4, head: 7 });
      V.arrow(g, sOut[0] - 4, sOut[1], cIn2[0] + 4, cIn2[1], { color: V.alpha(C.a, 0.7), lw: 1.4, head: 7, dash: [4, 4] });
      if (wide) {
        const mx = (cOut[0] + sIn[0]) / 2, gapW = sIn[0] - cOut[0] + 20;
        V.text(g, 'POST', mx, cOut[1] - 24, { size: 10.5, font: 'mono', weight: 600, align: 'center', color: C.c });
        V.text(g, '/chat/completions', mx, cOut[1] - 10, { size: 10.5, font: 'mono', align: 'center', color: C.c, maxW: gapW });
        V.text(g, 'SSE', mx, sOut[1] + 18, { size: 10.5, font: 'mono', weight: 600, align: 'center', color: C.a });
        V.text(g, 'data: {delta}', mx, sOut[1] + 32, { size: 10.5, font: 'mono', align: 'center', color: C.a, maxW: gapW });
      } else {
        V.text(g, 'POST', cOut[0] - 8, (cOut[1] + sIn[1]) / 2 + 4, { size: 10.5, font: 'mono', align: 'right', color: C.c });
        V.text(g, 'SSE', sOut[0] + 8, (sOut[1] + cIn2[1]) / 2 + 4, { size: 10.5, font: 'mono', color: C.a });
      }
      if (ph < 0.12 && S8.ctx !== 'none') { const k = V.ease.inOut(ph / 0.12); const q = V.along(pOut[0], pOut[1], cIn[0], cIn[1], k); V.glow(g, q[0], q[1], 10, C.a); }
      if (ph >= 0.12 && ph < 0.3) { const k = V.ease.inOut((ph - 0.12) / 0.18); const q = V.along(cOut[0], cOut[1], sIn[0], sIn[1], k); const s = S8.ctx === 'none' ? 7 : S8.ctx === 'section' ? 10 : 14; V.rr(g, q[0] - s, q[1] - s * 0.7, 2 * s, 1.4 * s, 3); g.fillStyle = V.alpha(C.c, 0.85); g.fill(); }
      if (ph >= 0.38 && ph < 0.95) {
        for (let q = 0; q < 5; q++) {
          const u = (((ph - 0.38) / 0.57) * 5 + q / 5) % 1;
          const pt = V.along(sOut[0], sOut[1], cIn2[0], cIn2[1], u);
          V.dot(g, pt[0], pt[1], 3, streamP < thinkShare ? C.b : C.a);
        }
      }
      // orçamento de tokens
      const by = h - 30, bx = 14, bw = w - 28;
      V.text(g, wide ? `max_tokens = 2048 (padrão do copilot)${S8.think ? ' · o raciocínio disputa o mesmo orçamento (ilustrativo)' : ' · pedido sem thinking'}` : `max_tokens = 2048${S8.think ? ' · raciocínio × resposta (ilustr.)' : ' · sem thinking'}`, bx, by - 8, { size: 11, color: C.muted, maxW: bw });
      V.box(g, bx, by, bw, 12, { r: 6, fill: V.alpha(C.muted, 0.12) });
      const used = streamP;
      if (S8.think) { V.box(g, bx, by, Math.max(2, bw * Math.min(used, thinkShare)), 12, { r: 6, fill: V.alpha(C.b, 0.75) }); if (used > thinkShare) V.box(g, bx + bw * thinkShare, by, bw * (used - thinkShare) * 0.5, 12, { r: 6, fill: V.alpha(C.a, 0.85) }); }
      else V.box(g, bx, by, Math.max(2, bw * used * 0.35), 12, { r: 6, fill: V.alpha(C.a, 0.85) });
    });
    upd8();

    /* ================================================================
       Figura 9 — o metrô da jornada
       ================================================================ */
    // estações = capítulos registrados (id, parte, título curto), na ordem da trilha; a lista fixa é só reserva
    const CH9_FALLBACK = [['mapa-da-jornada', 0, 'Mapa da jornada'], ['aprender-com-dados', 1, 'Aprender com dados'], ['redes-neurais', 1, 'Redes neurais'], ['tensores-algebra', 1, 'Tensores & álgebra'], ['numeros-no-computador', 1, 'Números no computador'], ['tokens-embeddings', 2, 'Tokens & embeddings'], ['atencao', 2, 'Atenção'], ['bloco-transformer', 2, 'Bloco Transformer'], ['posicao-rope', 2, 'Posição & RoPE'], ['geracao-amostragem', 2, 'Geração & amostragem'], ['prefill-decode-kv', 2, 'Prefill, decode & KV'], ['mixture-of-experts', 3, 'Mixture-of-Experts'], ['mla', 3, 'MLA & KV compacto'], ['atencao-esparsa', 3, 'Atenção esparsa'], ['atencao-linear', 3, 'Atenção linear & híbridos'], ['hiperconexoes', 3, 'Hyper-connections'], ['ngram-mtp', 3, 'N-gram & MTP'], ['multimodal', 3, 'Multimodal'], ['tres-modelos', 3, 'Os três modelos'], ['abliteracao', 3, 'Abliteração'], ['quantizacao-fundamentos', 4, 'Fundamentos de quantização'], ['fp8-fp4-microescala', 4, 'FP8, FP4 & microescala'], ['hadamard-trelica-exl3', 4, 'Hadamard, treliças & EXL3'], ['kv-cache-quantizado', 4, 'KV cache quantizado'], ['checkpoints', 4, 'Checkpoints'], ['medindo-qualidade', 4, 'Medindo qualidade'], ['gpu-por-dentro', 5, 'GPU por dentro'], ['memoria-unificada', 5, 'Memória unificada'], ['kernels', 5, 'Kernels'], ['cuda-graphs-warmup', 5, 'CUDA graphs & warmup'], ['vllm-arquitetura', 6, 'Arquitetura do vLLM'], ['paged-attention-kv', 6, 'PagedAttention & KV'], ['scheduler-batching', 6, 'Scheduler & batching'], ['prefix-caching', 6, 'Prefix caching'], ['api-tools-estruturado', 6, 'API, tools & JSON'], ['especulacao-fundamentos', 7, 'Especulação: fundamentos'], ['mtp-eagle-vocab', 7, 'MTP, EAGLE & vocab reduzido'], ['dflash-dspark', 7, 'DFlash2 & DSpark'], ['paralelismos', 8, 'Paralelismos'], ['nccl-rdma', 8, 'NCCL & RDMA'], ['tp3-tp4', 8, 'TP=3 & TP=4'], ['engenharia-de-receita', 9, 'Engenharia de receita'], ['receita-glm', 9, 'Receita GLM'], ['receita-qwen', 9, 'Receita Qwen'], ['receita-deepseek', 9, 'Receita DeepSeek'], ['benchmarking', 9, 'Benchmarking'], ['receita-do-zero', 9, 'Receita do zero'], ['laboratorio', 9, 'Laboratório'], ['motor-exato-tensorfold', 9, 'Motor exato (TensorFold)'], ['receita-glm-tensorfold', 9, 'Receita GLM · TensorFold']];
    const CH9 = (LA.chapters && LA.chapters.length ? LA.chapters.map((c) => [c.id, c.part, c.navTitle || c.title]) : CH9_FALLBACK).filter((c) => typeof c[1] === 'number');
    const NAV9 = {}; CH9.forEach(([id, p, nm]) => { NAV9[id] = { p, nm }; });
    const NPART9 = Math.max(10, ...CH9.map((c) => c[1] + 1));
    const ROUTES = [
      { label: '--kv-cache-dtype fp8 (→ fp8_ds_mla)', r: 'glm', ids: ['numeros-no-computador', 'atencao', 'prefill-decode-kv', 'mla', 'fp8-fp4-microescala', 'kv-cache-quantizado', 'paged-attention-kv', 'receita-glm'], d: 'Por que a GLM guarda o KV em FP8 empacotado (656 B por token por camada MLA): como números viram bytes, o que são K e V, por que o KV cresce com o contexto, como a MLA o comprime num latente, o formato FP8, como se quantiza o KV, como o vLLM o pagina — e a receita.' },
      { label: '--speculative-config mtp, k=3', r: 'qwen', ids: ['geracao-amostragem', 'prefill-decode-kv', 'ngram-mtp', 'cuda-graphs-warmup', 'especulacao-fundamentos', 'mtp-eagle-vocab', 'receita-qwen'], d: 'Por que a Qwen rascunha 3 tokens com a própria camada MTP: o laço autoregressivo, por que o decode é limitado por banda, o que é MTP, os CUDA graphs de tamanho fixo por passo, a matemática de rascunhar e verificar sem perdas, o vocabulário reduzido do rascunho — e a receita.' },
      { label: '--tensor-parallel-size 2 --enable-expert-parallel', r: 'qwen', ids: ['tensores-algebra', 'bloco-transformer', 'mixture-of-experts', 'paralelismos', 'nccl-rdma', 'receita-qwen'], d: 'Por que dividir o modelo entre dois nós, com experts inteiros em cada GPU: multiplicação de matrizes fatiada, onde ficam as matrizes num bloco, o que é um expert, TP × EP, as coletivas pela rede RDMA — e a receita.' },
      { label: 'EXL3 4 bpw nos experts roteados', r: 'glm', ids: ['numeros-no-computador', 'quantizacao-fundamentos', 'hadamard-trelica-exl3', 'medindo-qualidade', 'kernels', 'receita-glm'], d: 'Por que a GLM usa 4 bits por peso em treliça, só nos experts: bits e formatos, quantização uniforme e seus erros, rotações de Hadamard e treliças (EXL3), como medir o estrago (KLD), os kernels que desempacotam os pesos na hora — e a receita.' },
      { label: '--max-model-len 1048576', r: 'ds', ids: ['posicao-rope', 'prefill-decode-kv', 'atencao-esparsa', 'kv-cache-quantizado', 'paged-attention-kv', 'scheduler-batching', 'receita-deepseek'], d: 'Por que a DeepSeek aceita 1 milhão de tokens: posição e YaRN, o custo do KV no decode, atenção esparsa e comprimida, o layout do KV, o pool paginado e a concorrência máxima, o prefill em pedaços — e a receita.' },
      { label: '--moe-backend flashinfer_b12x', r: 'ds', ids: ['mixture-of-experts', 'fp8-fp4-microescala', 'gpu-por-dentro', 'kernels', 'receita-deepseek'], d: 'Por que a DeepSeek escolhe o backend de MoE b12x: o que o MoE computa, o formato MXFP4 dos experts, o que o GB10 (<code>sm_121</code>) suporta, como um kernel de MoE fundido funciona (e por que aqui ele roda como W4A16) — e a receita.' },
      { label: '--gpu-memory-utilization 0.85', r: 'glm', ids: ['prefill-decode-kv', 'memoria-unificada', 'paged-attention-kv', 'receita-glm'], d: 'Por que 0,85 (e por que 0,87 com um prefill de 256k já derrubou um nó): o KV cache, a memória unificada que o sistema operacional também usa, como o vLLM mede o orçamento e reserva o pool — e a receita.' },
      { label: 'KV=fp8 · PARALLEL=4 (TensorFold)', r: 'tf', ids: ['numeros-no-computador', 'geracao-amostragem', 'prefill-decode-kv', 'kv-cache-quantizado', 'memoria-unificada', 'paged-attention-kv', 'nccl-rdma', 'motor-exato-tensorfold', 'receita-glm-tensorfold'], d: 'Por que a GLM no TensorFold serve até 4 requisições (v1.3.2; até 8 desde a v1.5) com janela de 1M cada e responde com os mesmos bits juntas ou sozinhas: somas que mudam de bits conforme a ordem, o sorteio que vira uma conta refazível, o KV que cresce no decode, o FP8 com escala potência de 2 (528 B por token e camada), a memória unificada que congela em vez de falhar, o pool dividido entre as conversas, o all-gather pela rede RDMA, o motor exato — e a receita.' },
      { label: 'Uma receita do zero (MiMo)', r: 'all', ids: ['aprender-com-dados', 'atencao', 'tres-modelos', 'checkpoints', 'kernels', 'vllm-arquitetura', 'especulacao-fundamentos', 'paralelismos', 'benchmarking', 'receita-do-zero'], d: 'A rota longa: do que é aprender até a receita do zero, passando por uma estação de cada parte que o MiMo-V2.6-Flash vai cobrar.' },
    ];
    const f9 = root.querySelector('#fig-metro');
    const ctl9 = UI.controls(f9);
    const S9 = { route: 0, t0: 0 };
    let H9 = null;
    wrapCtl(UI.select(ctl9, { label: 'Parâmetro', options: ROUTES.map((R, i) => ({ v: String(i), label: R.label })), value: '0', onChange: (v) => { S9.route = +v; S9.t0 = H9 ? H9.t : 0; upd9(); } }));
    const box9 = LA.h('div', { style: { flexBasis: '100%', minWidth: '0', fontSize: '13.5px', lineHeight: '1.6', color: 'var(--text-2)' } });
    ctl9.appendChild(box9);
    function upd9() {
      const R = ROUTES[S9.route];
      box9.innerHTML = `${B(R.r)} ${R.d}<br>` + R.ids.filter((id) => NAV9[id]).map((id, k) => `<a href="#/c/${id}">${k + 1}. ${LA.esc(NAV9[id].nm)}</a>`).join(' → ');
    }
    upd9();
    H9 = V.fig(f9, (g, w, h, t, dt, S, H) => {
      const C = V.C();
      const pc = [C.a, C.e, C.b, C.c, C.d, C.a, C.b, C.f, C.e, C.c, C.a, C.b, C.c, C.d];
      const wide = w >= 700;
      const labW = wide ? 150 : 36;
      const xL = 14 + labW, xR = w - 14;
      const top = 16, rowH = (h - top - 12) / NPART9;
      const cnt = Array(NPART9).fill(0);
      CH9.forEach(([, p]) => { cnt[p]++; });
      const sp = (xR - xL) / Math.max(9, ...cnt);
      const pos = {}, k9 = Array(NPART9).fill(0);
      CH9.forEach(([id, p]) => { const k = k9[p]++; pos[id] = [xL + (k + 0.5) * sp, top + (p + 0.5) * rowH]; });
      for (let p = 0; p < NPART9; p++) {
        if (!cnt[p]) continue;
        const y = top + (p + 0.5) * rowH, part = (LA.parts || [])[p] || {};
        V.line(g, xL + 0.5 * sp, y, xL + (cnt[p] - 0.5) * sp, y, { color: V.alpha(pc[p], 0.5), lw: 4 });
        V.text(g, (part.roman || String(p)) + (wide && part.short ? ' · ' + part.short : ''), 14, y + 4, { size: 11.5, font: wide ? 'sans' : 'mono', weight: 600, color: pc[p], maxW: labW - 8 });
      }
      const R = ROUTES[S9.route], rcol = R.r === 'all' ? C.ink : recCol(C, R.r);
      const rIds = R.ids.filter((id) => pos[id]);
      // caminho amostrado
      const P = rIds.map((id) => pos[id]);
      const pts = [P[0]];
      for (let k = 1; k < P.length; k++) {
        const a = P[k - 1], b = P[k];
        const c1 = [a[0], a[1] + (b[1] - a[1]) * 0.55], c2 = [b[0], b[1] - (b[1] - a[1]) * 0.55];
        for (let s = 1; s <= 18; s++) pts.push(a[1] === b[1] ? V.along(a[0], a[1], b[0], b[1], s / 18) : V.bez(a, c1, c2, b, s / 18));
      }
      const L = [0]; for (let i = 1; i < pts.length; i++) L.push(L[i - 1] + Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]));
      const tot = L[L.length - 1] || 1;
      const prog = Math.min(1, (t - S9.t0) * 0.7);
      const drawTo = (lw, col) => { g.beginPath(); g.moveTo(pts[0][0], pts[0][1]); for (let i = 1; i < pts.length && L[i] <= prog * tot; i++) g.lineTo(pts[i][0], pts[i][1]); g.strokeStyle = col; g.lineWidth = lw; g.lineCap = 'round'; g.lineJoin = 'round'; g.stroke(); g.lineCap = 'butt'; };
      drawTo(10, V.alpha(rcol, 0.18)); drawTo(3, rcol);
      // estações
      const onR = new Set(rIds);
      let hov = null;
      CH9.forEach(([id, p, nm]) => {
        const [x, y] = pos[id], on = onR.has(id);
        g.beginPath(); g.arc(x, y, on ? 7 : 4.5, 0, Math.PI * 2); g.fillStyle = on ? rcol : C.bg; g.fill();
        g.strokeStyle = on ? C.bg : pc[p]; g.lineWidth = on ? 2 : 2; g.stroke();
        if (H.pointer.inside && Math.hypot(H.pointer.x - x, H.pointer.y - y) < 12) hov = [id, p, nm];
      });
      // trem
      if (prog >= 1) {
        const u = (((t - S9.t0) * 0.1) % 1) * tot;
        let i = 1; while (i < L.length - 1 && L[i] < u) i++;
        const k = (u - L[i - 1]) / (L[i] - L[i - 1] || 1), q = V.along(pts[i - 1][0], pts[i - 1][1], pts[i][0], pts[i][1], k);
        V.glow(g, q[0], q[1], 16, rcol); V.dot(g, q[0], q[1], 4, C.ink);
      }
      // rótulos das estações da rota: tenta vários lados e evita outros rótulos, a linha da rota e as estações
      const placed = [];
      const boxHit = (bx) => placed.some((q) => bx.x < q.x + q.w + 4 && bx.x + bx.w + 4 > q.x && bx.y < q.y + q.h + 2 && bx.y + bx.h + 2 > q.y);
      const routeHit = (bx) => pts.some(([px, py]) => px > bx.x - 3 && px < bx.x + bx.w + 3 && py > bx.y - 3 && py < bx.y + bx.h + 3);
      const dotHit = (bx) => rIds.some((id) => { const [sx2, sy2] = pos[id]; return sx2 > bx.x - 8 && sx2 < bx.x + bx.w + 8 && sy2 > bx.y - 8 && sy2 < bx.y + bx.h + 8; });
      rIds.forEach((id, k) => {
        const [x, y] = pos[id];
        const s = `${k + 1}. ${NAV9[id].nm}`;
        const wl = tw(g, s, 11, 600);
        const cands = [[0, -13, 'c'], [0, 21, 'c'], [12, 4, 'l'], [-12, 4, 'r'], [0, -26, 'c'], [0, 34, 'c']];
        let best = null;
        for (let pass = 0; pass < 3 && !best; pass++) {
          for (const [dx, dy, al] of cands) {
            let x0 = al === 'c' ? x + dx - wl / 2 : al === 'l' ? x + dx : x + dx - wl;
            x0 = V.clamp(x0, xL - 8, w - 4 - wl); // não cobre os nomes das partes à esquerda
            const bx = { x: x0, y: y + dy - 10, w: wl, h: 13 };
            if (bx.y < 2 || bx.y + bx.h > h - 2) continue;
            if (boxHit(bx)) continue;
            if (pass < 2 && routeHit(bx)) continue;
            if (pass < 1 && dotHit(bx)) continue;
            best = bx; break;
          }
        }
        if (!best) { const x0 = V.clamp(x - wl / 2, xL - 8, w - 4 - wl); best = { x: x0, y: y - 23, w: wl, h: 13 }; }
        placed.push(best);
        V.box(g, best.x - 3, best.y - 1, best.w + 6, best.h + 3, { r: 4, fill: V.alpha(C.panel, 0.92), stroke: V.alpha(rcol, 0.35), lw: 1 });
        V.text(g, s, best.x, best.y + 10, { size: 11, weight: 600, color: C.ink });
      });
      if (hov) tip(g, [hov[2], 'Parte ' + (((LA.parts || [])[hov[1]] || {}).roman || hov[1]) + (onR.has(hov[0]) ? ' · na rota' : '')], H.pointer.x, H.pointer.y, w, h);
    });

    /* ---------- cartões das partes ---------- */
    const box = root.querySelector('#parts-cards');
    LA.parts.forEach((p) => {
      const n = LA.chapters.filter((c) => c.part === p.n).length;
      box.appendChild(LA.h('div', { class: 'la-card', style: { borderTop: '3px solid ' + p.color } },
        LA.h('span', { class: 'ic', style: { color: p.color, fontFamily: 'var(--font-display)' } }, 'Parte ' + (p.roman || p.n)),
        LA.h('h4', null, p.title), LA.h('p', null, p.desc + (n ? ` (${n} cap.)` : ''))));
    });
  },
});

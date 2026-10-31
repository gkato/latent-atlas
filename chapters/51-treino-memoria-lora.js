LA.chapter({
  id: 'treino-memoria-lora',
  part: 10,
  order: 51,
  title: 'Treinar não é inferir: memória, otimizadores, ZeRO e LoRA',
  navTitle: 'Memória de treino & LoRA',
  subtitle: 'De onde vêm os 16 bytes por parâmetro de um fine-tuning, por que o BF16 sozinho não aprende, como checkpointing, ZeRO/FSDP, LoRA e QLoRA fazem o treino caber, e o que cabe de verdade em 2× DGX Spark e num Mac M4 de 24 GB.',
  minutes: 40,
  recipes: ['qwen'],
  tags: ['full fine-tuning', 'precisão mista', 'AdamW', 'Muon', 'gradient checkpointing', 'ZeRO', 'FSDP', 'LoRA', 'QLoRA', 'NF4', 'DoRA', 'DGX Spark', 'MLX'],
  goals: [
    'Fazer a conta de memória de um passo de treino (pesos, gradientes, estado do otimizador, pesos mestres, ativações e logits) e explicar por que o full FT com AdamW custa <b>16 B/parâmetro</b>',
    'Explicar por que o BF16 puro perde atualizações pequenas, como pesos mestres FP32, Kahan e arredondamento estocástico resolvem, e o que FP8/NVFP4 de fato economizam',
    'Escolher entre gradient checkpointing, ZeRO-1/2/3/FSDP e TP/PP/EP sabendo o custo em FLOPs e em rede de cada um',
    'Configurar LoRA, QLoRA e DoRA (posto, \\(\\alpha\\), módulos-alvo, inicialização, LR, LoRA por expert) e fundir ou servir adaptadores sem perder o efeito na requantização',
    'Dizer, antes de treinar, o que cabe em 2× DGX Spark e num Mac M4 de 24 GB, e quanto tempo vai levar',
  ],
  prereqs: ['pos-treino-mapa', 'memoria-unificada', 'paralelismos', 'quantizacao-fundamentos'],
  summary: [
    'Full FT com AdamW em precisão mista guarda <b>16 B/param</b>: 2 (pesos BF16) + 2 (gradientes) + 4 (mestre FP32) + 4 + 4 (\\(m\\), \\(v\\)). O Llama-3.1-8B pede 128,5 GB de estado contra 16,1 GB para servir. Num MoE, paga-se pelos parâmetros <b>totais</b>.',
    'O BF16 descarta somas menores que meio ulp: com \\(w = 0{,}02\\) e \\(\\eta = 10^{-5}\\), o peso não se move. Pesos mestres FP32, soma de Kahan ou arredondamento estocástico resolvem. FP8/NVFP4 aceleram (1,68× no GB300), mas quase não mexem no estado.',
    'Ativações ≈ \\(34\\,sbh\\) por camada com FlashAttention; o checkpointing guarda \\(2sbh\\) e recalcula (6N → 8N FLOPs); logits FP32 custam \\(s\\cdot V\\cdot 4\\) por sequência. Em memória unificada, offload para a CPU não cria capacidade.',
    'Estado do otimizador: SGD 0, momentum 1 buffer, AdamW 2, AdamW 8 bits ~2 B, Adafactor sublinear, Muon 1 (só matrizes 2D; Qwen3.8 e DeepSeek-V4 foram pré-treinados com ele). LR: warmup + linear, cosseno ou WSD; LoRA usa ≈ 10× o LR do full FT.',
    'ZeRO-1/2/3 particionam otimizador, gradientes e parâmetros: o 8B em 2 Sparks vai de 128,5 GB (DDP) a 64,2 GB/nó (ZeRO-3/FSDP), movendo \\(3\\Psi\\) por passo, ~1–2 % do tempo. TP faz 4 all-reduces por camada; PP paga bolha \\((p-1)/m\\); EP, 4 all-to-all por camada MoE.',
    'LoRA: \\(h = W_0x + \\frac{\\alpha}{r}BAx\\), \\(B = 0\\) no início, \\(r(d_{in}+d_{out})\\) treináveis (41,9 M no 8B com r16), em todas as lineares. QLoRA = NF4 + quantização dupla (4,127 bits) + otimizadores paginados: 70B num Spark. DoRA separa magnitude e direção. Em MoE, LoRA por expert e roteador congelado. Depois do merge, meça o KLD.',
    'Viabilidade: full FT até ~3–4B num Spark e ~8B em dois; LoRA BF16 ✅ até ~14B por nó e 🟡 no Qwen3-30B-A3B (61 GB de base), com densos de ~30–40B como extrapolação (estimativa); QLoRA 70B num nó; MoE de 125–320B só como experimento; Mac M4: QLoRA até ~14B em 4 bits. Tempo ≈ tokens·\\(kN\\)/(MFU·pico).',
  ],
  quiz: [
    { q: 'Llama-3.1-8B (8,03 B parâmetros), full fine-tuning com AdamW em precisão mista clássica. Quanto de estado, sem ativações, e onde cabe?', o: ['16,1 GB; cabe num Spark com folga', '64,2 GB; cabe num Spark', '128,5 GB; não cabe num Spark (sem folga para SO e ativações), mas cabe em dois com ZeRO-3/FSDP (64,2 GB por nó)', '1.129 GB; nem em dois Sparks'], a: 2, e: '16 B/param × 8,03 B = 128,5 GB. O Spark mostra 121,69 GiB (130,7 GB), e o orçamento prático do processo de treino é ~100–110 GiB, sem contar ativações. Com ZeRO-3/FSDP em 2 nós, todo o estado é dividido por 2. 16,1 GB são só os pesos BF16, a conta da inferência; 1.129 GB é o 70B.' },
    { q: 'Um peso vale 0,02, o treino roda em BF16 puro (sem mestre FP32) e o AdamW manda somar 1e-5. O que acontece?', o: ['O peso passa a valer 0,02001', 'Nada: em 0,02 o ulp do BF16 é 1,22e-4 e meio ulp é 6,1e-5; o arredondamento ao par descarta a soma', 'O peso sobe um ulp inteiro (1,22e-4)', 'O resultado vira subnormal'], a: 1, e: 'ulp = \\(2^{\\lfloor\\log_2 0{,}02\\rfloor - 7} = 2^{-13}\\). Somas menores que meio ulp voltam para o mesmo número BF16. Em pesos menores (0,002), o mesmo passo vira um salto de 1,53e-5, 53 % maior que o pedido. Pesos mestres FP32 ou arredondamento estocástico resolvem.' },
    { q: 'O que o gradient checkpointing completo troca?', o: ['Precisão por memória: guarda as ativações em FP8', 'Guarda só a entrada de cada camada (\\(2sbh\\)) e recalcula o resto no backward: as ativações caem de \\(34sbhL\\) para \\(\\approx 2sbhL + 34sbh\\), ao custo de mais um forward (6N → 8N FLOPs por token)', 'Divide as ativações entre os nós do data parallel', 'Elimina o tensor de logits'], a: 1, e: 'É uma troca de memória por compute. Para o Llama-3.1-8B com 2.048 tokens, 9,13 GB viram 0,82 GB. Os logits (\\(s\\cdot V\\cdot 4\\) bytes) não são afetados; para eles existe a perda em pedaços (<code>chunked_nll</code>).' },
    { q: 'ZeRO-3 comparado com ZeRO-2, com \\(N_d\\) processos de data parallel:', o: ['O ZeRO-3 não particiona nada; só comprime os gradientes', 'O ZeRO-3 também particiona os parâmetros (\\(16\\Psi/N_d\\) por processo), mas move \\(3\\Psi\\) elementos por passo em vez de \\(2\\Psi\\), porque reúne os parâmetros no forward e de novo no backward', 'O ZeRO-3 move menos dados porque cada processo tem menos parâmetros', 'O ZeRO-3 só funciona com TP'], a: 1, e: 'ZeRO-1 particiona o otimizador; ZeRO-2, também os gradientes; ZeRO-3 (≈ FSDP FULL_SHARD), também os parâmetros. O preço é 1,5× de comunicação. Entre dois Sparks isso é ~1,2 s contra ~63 s de compute num passo de 131 mil tokens.' },
    { q: 'Por que o LoRA inicializa \\(B = 0\\) e \\(A\\) aleatória, em vez de zerar as duas?', o: ['Para economizar memória do otimizador', 'Para que \\(\\Delta W = 0\\) no início (o modelo é exatamente o base) e ainda haja gradiente: \\(\\partial L/\\partial B \\propto (\\ldots)A^\\top \\neq 0\\). Com as duas zeradas, os dois gradientes seriam zero para sempre', 'Porque \\(B\\) é maior que \\(A\\)', 'Porque o \\(\\alpha/r\\) exige'], a: 1, e: '\\(\\partial L/\\partial A \\propto B^\\top(\\ldots)\\) e \\(\\partial L/\\partial B \\propto (\\ldots)A^\\top\\). Com \\(A\\) aleatória, \\(B\\) recebe gradiente desde o primeiro passo, e \\(A\\) começa a se mover assim que \\(B\\) deixa de ser zero.' },
    { q: 'Você treinou um QLoRA (base NF4) e quer servir o resultado numa receita NVFP4. Qual é o caminho seguro?', o: ['Servir o NF4 com o adaptador no vLLM da receita, que suporta qualquer formato', 'Fundir o adaptador num checkpoint BF16, requantizar para NVFP4 e medir o KLD contra o BF16 fundido e a métrica de tarefa, porque o adaptador aprendeu a corrigir \\(\\mathrm{deq}(W_{NF4})\\) e o efeito pode não sobreviver', 'Copiar os pesos NF4 para o formato NVFP4 bit a bit', 'Treinar de novo em FP32'], a: 1, e: 'Fundir em BF16 dá \\(W_{BF16} + BA \\ne \\mathrm{deq}(W_{NF4}) + BA\\). Um delta de LoRA pequeno pode ter magnitude comparável ao ruído de quantização por peso. LoRA sobre os caminhos de quantização das receitas não foi verificado; o merge “consciente de NVFP4” do Axolotl existe exatamente para esse caso.' },
  ],
  mount(root, LA) {
    const V = LA.viz, UI = LA.ui;
    const RED = () => V.C().red || '#FB2C36';
    const fmt = (x, d = 1) => (+x).toFixed(d).replace('.', ',');
    const fmtGB = (b, d = 1) => (b >= 1e12 ? fmt(b / 1e12, 2) + ' TB' : fmt(b / 1e9, b < 1e9 ? 2 : d) + ' GB');
    const sci = (x, d = 2) => {
      if (!x) return '0';
      const e = Math.floor(Math.log10(Math.abs(x)) + 1e-12);
      let m = x / Math.pow(10, e);
      let ms = m.toFixed(d).replace(/\.?0+$/, '');
      if (ms === '10') { ms = '1'; return ms + 'e' + (e + 1); }
      return ms.replace('.', ',') + 'e' + e;
    };
    const SPARK = 121.69 * 1073741824; // bytes visíveis à GPU (130,7 GB)
    const PRAT_LO = 100 * 1073741824, PRAT_HI = 110 * 1073741824; // faixa prática do processo de treino
    const MAC = 19.07e9;
    /** veredito por nó: até o começo da faixa prática cabe; dentro dela, apertado; acima, não cabe */
    const fitSpark = (b) => (b <= PRAT_LO ? ['cabe', 'ok'] : b <= PRAT_HI ? ['apertado', 'warn'] : ['não cabe', 'bad']);
    const nocase = (c) => { const el = c && (c.el || c); if (el && el.querySelectorAll) el.querySelectorAll('label, span').forEach((n) => { if (!n.classList.contains('val')) n.style.textTransform = 'none'; }); return c; };
    function adaptH(fig, wide, narrow, bp = 640) {
      const st = fig.querySelector('.la-fig-stage');
      const set = () => { const want = (fig.clientWidth < bp ? narrow : wide) + 'px'; if (st.style.height !== want) st.style.height = want; };
      set();
      const ro = new ResizeObserver(set); ro.observe(fig);
      if (LA.onCleanup) LA.onCleanup(() => ro.disconnect());
    }
    function tw(g, s, size, font = 'sans', weight = 400) { g.font = V.font(size, font, weight); return g.measureText(String(s)).width; }
    /** personagem: quadrado arredondado com olhos */
    function face(g, x, y, s, col, o = {}) {
      const C = V.C();
      if (o.alpha != null) g.globalAlpha = o.alpha;
      V.box(g, x - s / 2, y - s / 2, s, s, { r: s * 0.3, fill: o.fill || V.alpha(col, 0.85), stroke: col, lw: 1.4 });
      const eye = C.dark ? '#06140d' : '#ffffff';
      const ex = s * 0.17, ey = y - s * 0.08, er = Math.max(1.2, s * 0.07);
      const look = o.look || 0;
      V.dot(g, x - ex + look * s * 0.05, ey, er, eye); V.dot(g, x + ex + look * s * 0.05, ey, er, eye);
      g.beginPath();
      if (o.mood === 'sad') g.arc(x, y + s * 0.3, s * 0.14, Math.PI * 1.15, Math.PI * 1.85);
      else if (o.mood === 'o') g.arc(x, y + s * 0.17, s * 0.075, 0, Math.PI * 2);
      else g.arc(x, y + s * 0.06, s * 0.15, Math.PI * 0.2, Math.PI * 0.8);
      g.strokeStyle = eye; g.lineWidth = Math.max(1.1, s * 0.06); g.stroke();
      if (o.sweat) { g.fillStyle = V.alpha(C.b, 0.9); g.beginPath(); g.ellipse(x + s * 0.42, y - s * 0.3 + (o.sweat % 1) * s * 0.3, s * 0.07, s * 0.11, 0, 0, Math.PI * 2); g.fill(); }
      if (o.alpha != null) g.globalAlpha = 1;
    }
    function lock(g, x, y, s, col) {
      g.strokeStyle = col; g.lineWidth = Math.max(1.2, s * 0.14);
      g.beginPath(); g.arc(x, y - s * 0.15, s * 0.3, Math.PI, 0); g.stroke();
      g.fillStyle = col; V.rr(g, x - s * 0.45, y - s * 0.15, s * 0.9, s * 0.65, s * 0.12); g.fill();
    }
    function chip(g, x, y, w, h, label, o = {}) {
      const C = V.C();
      V.box(g, x, y, w, h, { r: o.r == null ? 6 : o.r, fill: o.fill || V.alpha(C.muted, 0.12), stroke: o.stroke || V.alpha(C.muted, 0.4), lw: o.lw || 1.2 });
      if (label != null) V.text(g, label, x + w / 2, y + h / 2 + 0.5, { size: o.size || 11.5, font: o.font || 'mono', weight: o.weight || 500, color: o.color || C.ink, align: 'center', base: 'middle', maxW: w - 4 });
    }
    function niceStep(range, n = 4) {
      const raw = range / n, p = Math.pow(10, Math.floor(Math.log10(raw)));
      const m = raw / p;
      return (m < 1.5 ? 1 : m < 3.5 ? 2 : m < 7.5 ? 5 : 10) * p;
    }
    const mm = (A, B) => A.map((r) => B[0].map((_, j) => r.reduce((s, a, k) => s + a * B[k][j], 0)));
    const tr = (A) => A[0].map((_, j) => A.map((r) => r[j]));
    function symEig(S) {
      const n = S.length, A = S.map((r) => r.slice());
      for (let sw = 0; sw < 40; sw++) {
        let off = 0; for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) off += A[i][j] * A[i][j];
        if (off < 1e-18) break;
        for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) {
          if (Math.abs(A[p][q]) < 1e-18) continue;
          const th = (A[q][q] - A[p][p]) / (2 * A[p][q]);
          const t = (th >= 0 ? 1 : -1) / (Math.abs(th) + Math.sqrt(th * th + 1));
          const c = 1 / Math.sqrt(t * t + 1), s = t * c;
          for (let k = 0; k < n; k++) { const a = A[k][p], b = A[k][q]; A[k][p] = c * a - s * b; A[k][q] = s * a + c * b; }
          for (let k = 0; k < n; k++) { const a = A[p][k], b = A[q][k]; A[p][k] = c * a - s * b; A[q][k] = s * a + c * b; }
        }
      }
      return A.map((r, i) => r[i]);
    }
    /** valores singulares de M (linhas ≤ colunas) */
    const svals = (M) => symEig(mm(M, tr(M))).map((x) => Math.sqrt(Math.max(0, x))).sort((a, b) => b - a);
    function ns5(M) {
      const a = 3.4445, b = -4.775, c = 2.0315;
      const fro = Math.sqrt(M.reduce((s, r) => s + r.reduce((q, x) => q + x * x, 0), 0)) + 1e-7;
      let X = M.map((r) => r.map((x) => x / fro));
      for (let k = 0; k < 5; k++) {
        const A = mm(X, tr(X)), AA = mm(A, A);
        const Bm = A.map((r, i) => r.map((x, j) => b * x + c * AA[i][j]));
        const BX = mm(Bm, X);
        X = X.map((r, i) => r.map((x, j) => a * x + BX[i][j]));
      }
      return X;
    }
    const lerpM = (A, B, t) => A.map((r, i) => r.map((x, j) => x + (B[i][j] - x) * t));

    /* ================================================ FIG 1 — a mochila de cada parâmetro */
    (function () {
      const fig = root.querySelector('#fig-mochila');
      adaptH(fig, 350, 480);
      const MODELS = {
        q4: { n: 'Qwen3-4B', P: 4.02e9, Pl: 33.0e6 },
        l8: { n: 'Llama-3.1-8B', P: 8.03e9, Pl: 41.9e6 },
        q30: { n: 'Qwen3-30B-A3B', P: 30.5e9, Pl: 117.2e6 },
        l70: { n: 'Llama-3.3-70B', P: 70.55e9, Pl: 207.1e6 },
      };
      const REG = {
        inf: { t: 'Inferência (BF16)', parts: [['peso BF16', 2, 'a', 'w']] },
        A: { t: 'A · precisão mista (16 B)', parts: [['peso BF16', 2, 'a', 'w'], ['gradiente BF16', 2, 'b', 'g'], ['mestre FP32', 4, 'd', 'ms'], ['m FP32', 4, 'c', 'm'], ['v FP32', 4, 'f', 'v']] },
        Ap: { t: 'A′ · carga FP32 + autocast (16 B)', parts: [['peso FP32', 4, 'a', 'w'], ['gradiente FP32', 4, 'b', 'g'], ['m FP32', 4, 'c', 'm'], ['v FP32', 4, 'f', 'v']] },
        B: { t: 'B · BF16 puro + AdamW (8 B)', parts: [['peso BF16', 2, 'a', 'w'], ['gradiente BF16', 2, 'b', 'g'], ['m BF16', 2, 'c', 'm'], ['v BF16', 2, 'f', 'v']] },
        C: { t: 'C · AdamW 8 bits (~6 B)', parts: [['peso BF16', 2, 'a', 'w'], ['gradiente BF16', 2, 'b', 'g'], ['m int8', 1, 'c', 'm'], ['v int8', 1, 'f', 'v']] },
        D: { t: 'D · Muon misto (12 B)', parts: [['peso BF16', 2, 'a', 'w'], ['gradiente BF16', 2, 'b', 'g'], ['mestre FP32', 4, 'd', 'ms'], ['M FP32', 4, 'c', 'm']] },
        E: { t: 'E · Muon BF16 puro (6 B)', parts: [['peso BF16', 2, 'a', 'w'], ['gradiente BF16', 2, 'b', 'g'], ['M BF16', 2, 'c', 'm']] },
        L: { t: 'LoRA (base congelada + adaptador)', parts: [['peso BF16', 2, 'a', 'w']], adapter: [['peso FP32', 4, 'a', 'w'], ['gradiente FP32', 4, 'b', 'g'], ['m FP32', 4, 'c', 'm'], ['v FP32', 4, 'f', 'v']] },
      };
      const NOTE = { w: 'lido no forward', g: 'escrito no backward', ms: 'onde a soma acontece', m: 'média de g', v: 'média de g²' };
      const S = { reg: 'A', model: 'l8' };
      const Wm = V.randMatrix(4, 4, 51, 2.2);
      const ctl = UI.controls(fig);
      UI.select(ctl, { label: 'Regime', options: Object.keys(REG).map((k) => ({ v: k, label: REG[k].t })), value: 'A', onChange: (v) => { S.reg = v; upd(); } });
      UI.seg(ctl, { label: 'Modelo', options: Object.keys(MODELS).map((k) => ({ v: k, label: MODELS[k].n })), value: 'l8', onChange: (v) => { S.model = v; upd(); } });
      const rB = UI.readout(ctl, 'bytes/param', '—'), rT = UI.readout(ctl, 'estado total', '—'), rX = UI.readout(ctl, 'vs. inferência', '—'), rF = UI.readout(ctl, 'num Spark', '—');
      const sum = (ps) => ps.reduce((s, p) => s + p[1], 0);
      function totals() {
        const R = REG[S.reg], M = MODELS[S.model];
        const bpp = sum(R.parts);
        if (R.adapter) return { bpp, total: bpp * M.P + sum(R.adapter) * M.Pl, lora: true };
        return { bpp, total: bpp * M.P };
      }
      function upd() {
        const t = totals(), M = MODELS[S.model];
        rB.set(t.lora ? '2 + 16 no adapt.' : (S.reg === 'C' ? '~' : '') + t.bpp + ' B');
        rT.set(fmtGB(t.total));
        rX.set(fmt(t.total / (2 * M.P), 1) + '×');
        const f = fitSpark(t.total); rF.set(f[0] + (t.lora || S.reg === 'inf' || f[1] === 'bad' ? '' : ' (só estado)'), f[1]);
      }
      upd();

      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const R = REG[S.reg], M = MODELS[S.model];
        const narrow = w < 600;
        const inf = S.reg === 'inf', lora = !!R.adapter;
        const PH = inf ? [[0, 2.4]] : [[0, 1.2], [1.2, 2.4], [2.4, 3.8], [3.8, 5.2], [5.2, 6.5]];
        const cyc = PH[PH.length - 1][1], tt = t % cyc;
        let ph = 0, pf = 0;
        PH.forEach((p, i) => { if (tt >= p[0] && tt < p[1]) { ph = i; pf = (tt - p[0]) / (p[1] - p[0]); } });
        // matriz de pesos
        const cell = narrow ? Math.max(22, Math.min(30, (w - 40) * 0.1)) : Math.min(40, w * 0.055);
        const mx = 14, my = 30;
        V.text(g, 'W  (×10⁻²)', mx, my - 10, { size: 11.5, font: 'mono', color: C.muted });
        const geo = V.matrix(g, { x: mx, y: my, data: Wm, cell, cmap: 'div', values: true, digits: 1, hiCells: [[1, 2]] });
        const [hx, hy] = geo.center(1, 2);
        // personagem
        const fs = narrow ? 40 : 54;
        const cx = mx + geo.w + (narrow ? 44 : 56), cy = my + geo.h / 2;
        g.setLineDash([3, 4]); V.line(g, hx + cell / 2, hy, cx - fs / 2 - 4, cy, { color: V.alpha(C.c, 0.7), lw: 1.3 }); g.setLineDash([]);
        const fwd = ph === 0;
        if (fwd) V.glow(g, cx, cy, fs * 0.9, C.a, 0.35 * Math.sin(pf * Math.PI));
        face(g, cx, cy, fs, C.a, { mood: ph === 1 ? 'o' : undefined, look: Math.sin(t * 1.3) });
        if (lora) lock(g, cx + fs * 0.42, cy - fs * 0.42, fs * 0.32, C.muted);
        V.text(g, lora ? 'congelado' : 'um parâmetro', cx, cy + fs / 2 + 14, { size: 11, color: C.muted, align: 'center' });
        // linhas da mochila
        const rows = [];
        if (lora) {
          const pc = (M.Pl / (M.P + M.Pl)) * 100;
          rows.push({ hdr: 'parâmetro congelado (' + fmt(100 - pc, 1) + ' %): sem gradiente' });
          R.parts.forEach((p) => rows.push({ p, frozen: true }));
          rows.push({ hdr: 'parâmetro do adaptador (' + fmt(pc, 1) + ' %)' });
          R.adapter.forEach((p) => rows.push({ p }));
        } else R.parts.forEach((p) => rows.push({ p }));
        const cs = narrow ? 17 : 21, gap = 4, rowH = narrow ? 27 : 31, hdrH = 22;
        const labW = narrow ? 104 : 118;
        let x0, y0;
        if (narrow) { x0 = 14; y0 = my + Math.max(geo.h, fs + 20) + 28; } else { x0 = cx + fs / 2 + 34; y0 = 22; }
        let yy = y0;
        const centers = {};
        const noteX = x0 + labW + 4 * (cs + gap) + 30;
        rows.forEach((r) => {
          if (r.hdr) { V.text(g, r.hdr, x0, yy + 14, { size: 11.5, weight: 600, color: C.ink, maxW: w - x0 - 10 }); yy += hdrH; return; }
          const [lab, nb, ck, role] = r.p;
          const col = C[ck];
          V.text(g, lab, x0 + labW - 8, yy + cs / 2 + 4, { size: 11.5, font: 'mono', align: 'right', color: C.ink });
          let fill = 1, act = false;
          if (role === 'w') act = ph === 0 || (!inf && ph === 3 && !r.frozen);
          if (role === 'g') fill = inf ? 0 : ph === 0 ? 0 : ph === 1 ? pf : ph === 4 ? 1 - pf : 1;
          if (role === 'm' || role === 'v') act = ph === 2;
          if (role === 'ms') act = ph === 2 || ph === 3;
          if (r.frozen && role === 'w') act = ph === 0;
          const nCells = Math.max(1, Math.ceil(nb));
          for (let k = 0; k < nCells; k++) {
            const x = x0 + labW + k * (cs + gap);
            const on = role === 'g' ? (k + 1) / nCells <= fill + 1e-6 || (fill > 0 && k < fill * nCells) : true;
            if (on) {
              V.box(g, x, yy, cs, cs, { r: 4, fill: V.alpha(col, act ? 0.95 : 0.55), stroke: col, lw: act ? 2 : 1.1 });
              if (act) V.glow(g, x + cs / 2, yy + cs / 2, cs * 0.9, col, 0.35);
            } else { g.setLineDash([3, 3]); V.box(g, x, yy, cs, cs, { r: 4, stroke: V.alpha(col, 0.6), lw: 1 }); g.setLineDash([]); }
          }
          if (noteX + 90 < w) V.text(g, NOTE[role] + (r.frozen ? ' (só leitura)' : ''), noteX, yy + cs / 2 + 4, { size: 11, color: C.muted, maxW: w - noteX - 6 });
          centers[(r.frozen ? 'f' : '') + role] = [x0 + labW - 4, yy + cs / 2];
          yy += rowH;
        });
        // setas do ciclo
        const arc = (a, b, lab, col) => {
          if (!a || !b) return;
          const xA = x0 + labW + 4 * (cs + gap) + 2;
          V.arrow(g, xA, a[1], xA, b[1] + (b[1] > a[1] ? -6 : 6), { color: col, lw: 1.8, head: 6, curve: (b[1] > a[1] ? -1 : 1) * Math.min(20, Math.abs(b[1] - a[1]) * 0.3) });
          if (lab) V.text(g, lab, xA - 26, (a[1] + b[1]) / 2 + 4, { size: 11, color: col, align: 'right', font: 'mono' });
        };
        if (!inf) {
          if (ph === 2) { arc(centers.g, centers.m, '', C.c); arc(centers.g, centers.v, '', C.f); if (centers.ms) arc(centers.m, centers.ms, '', C.d); }
          if (ph === 3) { if (centers.ms) arc(centers.ms, centers.w, '', C.d); }
        }
        // legenda da fase
        let capTxt;
        const hasMs = !!centers.ms;
        if (inf) capTxt = 'inferência: o peso só é lido (2 B). Nada mais por parâmetro; o KV cache cresce por token, não por peso';
        else capTxt = [
          '1 · forward: o peso é lido e a entrada da camada fica guardada',
          '2 · backward: chega ∂L/∂w e ocupa o compartimento do gradiente',
          '3 · otimizador: atualiza ' + (S.reg === 'D' || S.reg === 'E' ? 'o momento M' : 'm e v') + (hasMs ? ' e soma o passo no mestre FP32' : ''),
          hasMs ? '4 · cópia: o mestre FP32 é arredondado de volta para o peso BF16' : (S.reg === 'Ap' ? '4 · o passo é somado direto no peso FP32' : lora ? '4 · só o adaptador muda; a base continua idêntica' : '4 · o passo é somado direto no peso BF16 (e pode sumir no arredondamento!)'),
          '5 · zero_grad: o gradiente é descartado até o próximo backward',
        ][ph];
        const capY = narrow ? yy + 8 : Math.max(yy + 6, my + geo.h + 40);
        V.wrap(g, capTxt, narrow ? 14 : x0, capY + 10, w - (narrow ? 14 : x0) - 12, { size: 12, color: C.ink, lineH: 16, maxLines: 2 });
        // barra do modelo inteiro
        const tot = totals();
        const by = Math.min(h - 46, capY + 60), bx = 14, bw = w - 28, bh = 18;
        const scale = Math.max(tot.total, SPARK) * 1.08;
        const X = (b) => bx + (b / scale) * bw;
        V.box(g, bx, by, bw, bh, { r: 5, fill: V.alpha(C.muted, 0.08), stroke: V.alpha(C.muted, 0.3), lw: 1 });
        let acc = 0;
        const segs = [];
        if (lora) { segs.push([2 * M.P, C.a]); R.adapter.forEach((p) => segs.push([p[1] * M.Pl, C[p[2]]])); }
        else R.parts.forEach((p) => segs.push([p[1] * M.P, C[p[2]]]));
        segs.forEach(([b, col]) => { const x1 = X(acc), x2 = X(acc + b); g.fillStyle = V.alpha(col, 0.8); g.fillRect(x1, by + 1, Math.max(1, x2 - x1), bh - 2); acc += b; });
        g.fillStyle = V.alpha(C.d, 0.12); g.fillRect(X(PRAT_LO), by - 4, X(PRAT_HI) - X(PRAT_LO), bh + 8);
        V.line(g, X(SPARK), by - 8, X(SPARK), by + bh + 6, { color: RED(), lw: 2, dash: [4, 3] });
        const nS = tot.total / SPARK;
        const lab = M.n + ': ' + (lora ? '2 B × ' + fmt(M.P / 1e9, 2) + ' B + 16 B × ' + fmt(M.Pl / 1e6, 1) + ' M' : (S.reg === 'C' ? '~' : '') + tot.bpp + ' B × ' + fmt(M.P / 1e9, 2) + ' B') + ' = ' + fmtGB(tot.total) + (nS > 1.05 ? '  (≈ ' + fmt(nS, 1) + ' Sparks)' : '');
        V.text(g, lab, bx, by - 10, { size: 11.5, font: 'mono', color: C.ink, maxW: bw });
        const sl = '121,69 GiB';
        const slx = Math.min(w - 8, X(SPARK) + 4);
        V.text(g, sl, slx > w - 70 ? X(SPARK) - 4 : slx, by + bh + 16, { size: 11, color: RED(), align: slx > w - 70 ? 'right' : 'left', font: 'mono' });
      });
    })();

    /* ================================================ FIG 2 — o peso que não se move */
    (function () {
      const fig = root.querySelector('#fig-bf16');
      adaptH(fig, 440, 470);
      const f32 = new Float32Array(1), u32 = new Uint32Array(f32.buffer);
      const bf = (x) => { f32[0] = x; let b = u32[0]; b = (b + 0x7FFF + ((b >>> 16) & 1)) & 0xFFFF0000; u32[0] = b >>> 0; return f32[0]; };
      const bfDown = (x) => { f32[0] = x; u32[0] = (u32[0] & 0xFFFF0000) >>> 0; return f32[0]; };
      const bfNext = (x) => { f32[0] = x; u32[0] = ((u32[0] & 0xFFFF0000) + 0x10000) >>> 0; return f32[0]; };
      const ulpOf = (w) => Math.pow(2, Math.floor(Math.log2(Math.abs(w))) - 7);
      const N = 120;
      const S = { w0: 0.02, lg: -5, noise: false, tr: null };
      function build() {
        const eta = Math.pow(10, S.lg);
        const r = V.rng(77);
        const a = [bf(S.w0)], b = [bf(S.w0)], c = [bf(S.w0)];
        for (let k = 0; k < N; k++) {
          const d = S.noise ? eta * (0.3 + 1.4 * r()) : eta;
          a.push(Math.fround(a[k] + d));
          b.push(bf(Math.fround(b[k] + d)));
          const x = Math.fround(c[k] + d), lo = bfDown(x), hi = bfNext(lo);
          c.push(lo === x ? x : (r() < (x - lo) / (hi - lo) ? hi : lo));
        }
        S.tr = { a, b, c, eta };
      }
      build();
      const ctl = UI.controls(fig);
      nocase(UI.seg(ctl, { label: 'w₀', options: [{ v: 0.002, label: '0,002' }, { v: 0.02, label: '0,02' }, { v: 0.2, label: '0,2' }], value: 0.02, onChange: (v) => { S.w0 = v; build(); upd(); } }));
      nocase(UI.slider(ctl, { label: 'passo η', min: -6, max: -3, step: 0.25, value: -5, fmt: (v) => sci(Math.pow(10, v), 1), onInput: (v) => { S.lg = v; build(); upd(); } }));
      UI.toggle(ctl, { label: 'passo com ruído (como o Adam)', value: false, onChange: (v) => { S.noise = v; build(); upd(); } });
      const rU = UI.readout(ctl, 'ulp em w₀', '—'), rH = UI.readout(ctl, 'η ÷ ½ulp', '—'), rM = UI.readout(ctl, 'BF16-RNE andou', '—');
      nocase(rU); nocase(rH); nocase(rM);
      function upd() {
        const u = ulpOf(bf(S.w0)), eta = S.tr.eta;
        rU.set(sci(u, 2));
        rH.set(fmt(eta / (u / 2), 2) + '×', eta < u / 2 ? 'bad' : 'ok');
        const moved = S.tr.b[N] - S.tr.b[0], want = S.tr.a[N] - S.tr.a[0];
        rM.set(moved === 0 ? '0 (preso)' : fmt((moved / want) * 100, 0) + ' % do FP32', moved === 0 ? 'bad' : Math.abs(moved / want - 1) > 0.15 ? 'warn' : 'ok');
      }
      upd();
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const T = S.tr;
        const per = N / 12 + 2.2;
        const k = Math.min(N, Math.floor(((t % per) / (N / 12)) * N));
        const w0 = T.a[0], u0 = ulpOf(w0);
        const span = Math.max(5 * u0, (T.a[N] - w0) * 1.12, (T.b[N] - w0) * 1.05, (T.c[N] - w0) * 1.05);
        const lo = w0 - 1.2 * u0, hi = w0 + span;
        const pad = 18, L = pad + 70, Rr = w - pad;
        const X = (v) => L + ((v - lo) / (hi - lo)) * (Rr - L);
        const topY = 36;
        V.text(g, 'régua BF16 perto de w₀ = ' + String(S.w0).replace('.', ','), pad, 18, { size: 12, weight: 600, color: C.ink });
        // riscos BF16
        let v = bfDown(lo), n = 0;
        const ticks = [];
        while (v <= hi && n < 600) { if (v >= lo) ticks.push(v); v = bfNext(v); n++; }
        const dense = ticks.length > 70;
        const lanes = [['FP32 (mestre)', T.a, C.a], ['BF16 ao par', T.b, C.c], ['BF16 estocástico', T.c, C.b]];
        const laneY = (i) => topY + 22 + i * 34;
        const yTop = topY + 6, yBot = laneY(2) + 16;
        ticks.forEach((tv, i) => {
          const x = X(tv);
          V.line(g, x, yTop, x, yBot, { color: V.alpha(C.muted, dense ? 0.18 : 0.35), lw: 1 });
        });
        if (!dense && ticks.length > 1) {
          const u1 = ticks[1] - ticks[0];
          const xa = X(ticks[0]), xb = X(ticks[1]);
          // colchete de 1 ulp entre os dois primeiros riscos, com o rótulo ao lado (não por cima)
          const yb = yTop - 7;
          V.line(g, xa, yb, xb, yb, { color: C.muted, lw: 1 });
          V.line(g, xa, yb - 3, xa, yb + 3, { color: C.muted, lw: 1 });
          V.line(g, xb, yb - 3, xb, yb + 3, { color: C.muted, lw: 1 });
          V.text(g, '1 ulp = ' + sci(u1, 2), xb + 6, yb + 4, { size: 11, align: 'left', color: C.muted, font: 'mono' });
        } else if (dense) V.text(g, ticks.length + ' números BF16 neste trecho', Rr, yTop - 6, { size: 11, align: 'right', color: C.muted });
        lanes.forEach(([lab, arr, col], i) => {
          const y = laneY(i);
          V.text(g, lab, L - 8, y + 4, { size: 11, align: 'right', color: col, font: 'mono', maxW: L - pad - 4 });
          V.line(g, X(w0), y, X(arr[k]), y, { color: V.alpha(col, 0.5), lw: 3 });
          face(g, X(arr[k]), y, 16, col, { mood: i === 1 && arr[k] === arr[0] && k > 10 ? 'sad' : undefined });
        });
        // seta do passo pedido
        const yA = laneY(2) + 30;
        const xe = X(w0 + T.eta);
        V.arrow(g, X(w0), yA, Math.max(X(w0) + 3, xe), yA, { color: C.d, lw: 2, head: 6 });
        V.text(g, 'um passo η = ' + sci(T.eta, 1) + (T.eta < u0 / 2 ? '  (< ½ ulp: some)' : ''), Math.max(X(w0) + 8, xe + 6), yA + 4, { size: 11, color: C.d, font: 'mono' });
        // gráfico de deslocamento
        const gy0 = yA + 22, gy1 = h - 26, gx0 = L, gx1 = Rr;
        const dmax = Math.max(T.a[N] - w0, T.b[N] - w0, T.c[N] - w0, 1e-30) * 1.08;
        V.line(g, gx0, gy1, gx1, gy1, { color: C.muted, lw: 1 });
        V.line(g, gx0, gy0, gx0, gy1, { color: C.muted, lw: 1 });
        V.text(g, 'w − w₀', gx0 - 8, gy0 + 10, { size: 11, align: 'right', color: C.muted, font: 'mono' });
        V.text(g, sci(dmax / 1.08, 1), gx0 - 8, gy0 + 24, { size: 11, align: 'right', color: C.muted, font: 'mono' });
        V.text(g, 'passos →  ' + k + ' / ' + N, gx1, gy1 + 16, { size: 11, align: 'right', color: C.muted, font: 'mono' });
        const PX = (i) => gx0 + (i / N) * (gx1 - gx0), PY = (d) => gy1 - (d / dmax) * (gy1 - gy0);
        lanes.forEach(([, arr, col], li) => {
          g.beginPath();
          for (let i = 0; i <= k; i++) { const x = PX(i), y = PY(arr[i] - w0); if (i === 0) g.moveTo(x, y); else g.lineTo(x, y); }
          g.strokeStyle = col; g.lineWidth = li === 0 ? 2.6 : 2; if (li === 0) g.setLineDash([6, 4]); g.stroke(); g.setLineDash([]);
        });
      });
    })();

    /* ================================================ FIG 3 — guardar ou recalcular */
    (function () {
      const fig = root.querySelector('#fig-ativacoes');
      adaptH(fig, 470, 600);
      const MODELS = { l8: { n: 'Llama-3.1-8B', h: 4096, L: 32, V: 128256, N: 8.03e9 }, q8: { n: 'Qwen3-8B', h: 4096, L: 36, V: 151936, N: 8.19e9 }, l70: { n: 'Llama-3.3-70B', h: 8192, L: 80, V: 128256, N: 70.55e9 } };
      const S = { ck: false, sE: 11, m: 'l8' };
      const ctl = UI.controls(fig);
      UI.toggle(ctl, { label: 'gradient checkpointing', value: false, onChange: (v) => { S.ck = v; upd(); } });
      UI.slider(ctl, { label: 'sequência s', min: 9, max: 15, step: 1, value: 11, fmt: (v) => String(Math.pow(2, v)).replace(/\B(?=(\d{3})+(?!\d))/g, '.'), onInput: (v) => { S.sE = v; upd(); } });
      UI.seg(ctl, { label: 'Modelo', options: Object.keys(MODELS).map((k) => ({ v: k, label: MODELS[k].n })), value: 'l8', onChange: (v) => { S.m = v; upd(); } });
      const r1 = UI.readout(ctl, 'sem ckpt', '—'), r2 = UI.readout(ctl, 'com ckpt', '—'), r3 = UI.readout(ctl, 'logits FP32', '—'), r4 = UI.readout(ctl, 'FLOPs/token', '—');
      const calc = () => {
        const M = MODELS[S.m], s = Math.pow(2, S.sE);
        const A = 34 * s * M.h, I = 2 * s * M.h, Lg = s * M.V * 4;
        return { M, s, A, I, Lg, no: A * M.L, ck: I * M.L + A };
      };
      function upd() {
        const c = calc();
        r1.set(fmtGB(c.no), S.ck ? undefined : 'warn'); r2.set(fmtGB(c.ck), S.ck ? 'ok' : undefined); r3.set(fmtGB(c.Lg)); r4.set(S.ck ? '8N' : '6N');
      }
      upd();
      // memória ao longo do tempo (u ∈ [0,1]): forward 0–0.44, perda 0.44–0.5, backward 0.5–1
      function memAt(c, ck, u) {
        const L = c.M.L;
        const per = ck ? c.I : c.A;
        if (u < 0.44) return (u / 0.44) * L * per;
        if (u < 0.5) return L * per + c.Lg * ((u - 0.44) / 0.06);
        const v = (u - 0.5) / 0.5, done = v * L;
        const left = (L - done) * per + (ck && done < L ? c.A : 0);
        return left + c.Lg * Math.max(0, 1 - v * 8);
      }
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const c = calc();
        const narrow = w < 560;
        const per = 7.5, u = (t % per) / per;
        const NL = 8;
        // modelo (esquerda)
        const mx = 16, mw = narrow ? 92 : 128, top = 40, plotH = narrow ? 150 : 130;
        const bot = h - plotH - 46;
        const lh = (bot - top - 40) / NL;
        V.text(g, c.M.n + ' · ' + c.M.L + ' camadas', mx, 20, { size: 12, weight: 600, color: C.ink, maxW: narrow ? w - 32 : mw + 30 });
        // onda
        let waveY = null, phase = 'forward';
        if (u < 0.44) waveY = bot - (u / 0.44) * (bot - top - 40);
        else if (u < 0.5) { waveY = top + 18; phase = 'perda'; }
        else { waveY = top + 40 + ((u - 0.5) / 0.5) * (bot - top - 40); phase = 'backward'; }
        // logits
        V.box(g, mx, top, mw, 30, { r: 6, fill: V.alpha(C.f, phase === 'perda' ? 0.5 : 0.15), stroke: C.f, lw: 1.2, label: 'logits [s, V]', size: 11.5 });
        for (let i = 0; i < NL; i++) {
          const y = bot - (i + 1) * lh;
          const isW = waveY != null && Math.abs(y + lh / 2 - waveY) < lh / 2 && phase !== 'perda';
          V.box(g, mx, y + 2, mw, lh - 4, { r: 5, fill: V.alpha(phase === 'backward' ? C.b : C.a, isW ? 0.55 : 0.12), stroke: V.alpha(phase === 'backward' ? C.b : C.a, 0.7), lw: 1 });
          V.text(g, 'camadas ' + (Math.round((i * c.M.L) / NL) + 1) + '–' + Math.round(((i + 1) * c.M.L) / NL), mx + mw / 2, y + lh / 2 + 4, { size: 11, align: 'center', color: C.ink, font: 'mono', maxW: mw - 6 });
        }
        V.text(g, phase === 'forward' ? '↑ forward' : phase === 'perda' ? 'perda' : '↓ backward', mx + mw / 2, bot + 18, { size: 12, align: 'center', color: phase === 'backward' ? C.b : C.a, weight: 600 });
        // prateleira (direita)
        const sx = mx + mw + (narrow ? 22 : 40), sw = w - sx - 16;
        const maxMem = c.no + c.Lg;
        const shelfTop = top, shelfBot = bot;
        const SH = (b) => (b / maxMem) * (shelfBot - shelfTop);
        V.text(g, 'na prateleira agora: ' + fmtGB(memAt(c, S.ck, u)), narrow ? sx : sx, narrow ? 34 : 20, { size: 12, color: C.ink, font: 'mono', maxW: sw });
        g.strokeStyle = V.alpha(C.muted, 0.4); g.lineWidth = 1; g.strokeRect(sx + 0.5, shelfTop + 0.5, sw - 1, shelfBot - shelfTop - 1);
        // caixas por grupo de camadas
        const fwdDone = u < 0.44 ? (u / 0.44) * NL : NL;
        const bwdDone = u < 0.5 ? 0 : ((u - 0.5) / 0.5) * NL;
        const perG = (S.ck ? c.I : c.A) * (c.M.L / NL);
        let y = shelfBot;
        const bw = Math.min(sw - 20, narrow ? sw - 12 : 220);
        // com checkpointing, cada chip (2·s·h) é 17× menor que uma caixa (34·s·h): desenhado ampliado, com aviso
        let nChipsDrawn = 0;
        for (let i = 0; i < NL; i++) {
          const present = i < fwdDone && i < NL - bwdDone;
          if (!present) continue;
          const hh = S.ck ? Math.max(9, SH(perG)) : Math.max(2, SH(perG));
          y -= hh;
          V.box(g, sx + 6, y, bw, hh - 1, { r: Math.min(4, hh / 2), fill: V.alpha(S.ck ? C.e : C.a, 0.55), stroke: S.ck ? C.e : C.a, lw: 1 });
          if (hh > 14) V.text(g, (S.ck ? 'entrada ' : 'ativações ') + (i + 1), sx + 12, y + hh / 2 + 4, { size: 11, color: C.ink, font: 'mono', maxW: bw - 10 });
          nChipsDrawn++;
        }
        if (S.ck && nChipsDrawn && SH(perG) < 9) {
          const tag = 'chips ampliados: cada um é 17× menor que uma caixa';
          const nL = tw(g, tag, 11, 'mono') > sw - 14 ? 2 : 1;
          V.wrap(g, tag, sx + 8, y - 6 - (nL - 1) * 14, sw - 14, { size: 11, color: C.e, font: 'mono', lineH: 14, maxLines: 2 });
          y -= 4 + nL * 14;
        }
        // recálculo (âmbar)
        if (S.ck && u >= 0.5) {
          const gi = NL - 1 - Math.floor(bwdDone);
          if (gi >= 0) {
            const hh = Math.max(10, SH(c.A));
            const pulse = 0.5 + 0.5 * Math.sin(t * 9);
            y -= hh;
            V.box(g, sx + 6, y, bw, hh - 1, { r: 4, fill: V.alpha(C.d, 0.35 + 0.35 * pulse), stroke: C.d, lw: 1.5 });
            if (sx + bw + 16 < w - 60) V.text(g, '← recálculo de 1 camada (34·s·h)', sx + bw + 14, y + hh / 2 + 4, { size: 11, color: C.d, font: 'mono', maxW: w - sx - bw - 24 });
            else V.text(g, 'recálculo', sx + 12, y + hh / 2 + 4, { size: 11, color: C.ink, font: 'mono' });
          }
        }
        if (S.ck && fwdDone - bwdDone > 0.5) {
          const nChips = Math.max(0, Math.min(fwdDone, NL - bwdDone));
          const yc = shelfBot - Math.max(9, SH(perG)) * Math.floor(nChips) / 2;
          if (sx + bw + 16 < w - 60 && !(u >= 0.5)) V.text(g, '← entradas guardadas (2·s·h por camada)', sx + bw + 14, Math.min(shelfBot - 4, yc + 4), { size: 11, color: C.e, font: 'mono', maxW: w - sx - bw - 24 });
        }
        // logits
        const lgOn = u >= 0.44 && u < 0.56;
        if (lgOn) {
          const f = u < 0.5 ? (u - 0.44) / 0.06 : Math.max(0, 1 - (u - 0.5) / 0.06);
          const hh = SH(c.Lg) * f;
          y -= hh;
          V.box(g, sx + 6, y, bw, Math.max(1, hh - 1), { r: 4, fill: V.alpha(C.f, 0.5), stroke: C.f, lw: 1 });
          if (hh > 14) V.text(g, 'logits FP32 ' + fmtGB(c.Lg), sx + 12, y + hh / 2 + 4, { size: 11, color: C.ink, font: 'mono', maxW: bw - 10 });
        }
        if (!narrow) V.text(g, 'teto da prateleira = sem ckpt + logits = ' + fmtGB(maxMem), sx + sw - 6, shelfTop + 14, { size: 11, color: C.muted, align: 'right', font: 'mono' });
        // gráfico memória × tempo
        const py0 = h - plotH + 4, py1 = h - 22, px0 = 16 + 48, px1 = w - 16;
        V.text(g, 'memória ao longo de um passo', 16, py0 - 6, { size: 11.5, color: C.muted });
        V.line(g, px0, py1, px1, py1, { color: C.muted, lw: 1 }); V.line(g, px0, py0 + 6, px0, py1, { color: C.muted, lw: 1 });
        const PX = (x) => px0 + x * (px1 - px0), PY = (b) => py1 - (b / maxMem) * (py1 - py0 - 8);
        V.text(g, fmtGB(maxMem, 0), px0 - 6, py0 + 14, { size: 11, align: 'right', color: C.muted, font: 'mono' });
        [[false, C.a], [true, C.e]].forEach(([ck, col]) => {
          g.beginPath();
          for (let i = 0; i <= 120; i++) { const x = i / 120, b = memAt(c, ck, x); if (!i) g.moveTo(PX(x), PY(b)); else g.lineTo(PX(x), PY(b)); }
          g.strokeStyle = V.alpha(col, ck === S.ck ? 1 : 0.35); g.lineWidth = ck === S.ck ? 2.4 : 1.4; g.stroke();
        });
        V.line(g, PX(u), py0 + 6, PX(u), py1, { color: V.alpha(C.ink, 0.5), lw: 1, dash: [3, 3] });
        V.text(g, 'sem ckpt', PX(0.3), PY(memAt(c, false, 0.3)) - 6, { size: 11, color: C.a, align: 'center' });
        V.text(g, 'com ckpt', PX(0.8), Math.min(py1 - 4, PY(memAt(c, true, 0.8)) - 6), { size: 11, color: C.e, align: 'center' });
      });
    })();

    /* ================================================ FIG 4 — estados do otimizador */
    (function () {
      const fig = root.querySelector('#fig-otimizadores');
      const stgO = fig.querySelector('.la-fig-stage');
      stgO.style.height = (fig.clientWidth < 600 ? 320 : 410) + 'px';
      const R4 = 4, K6 = 6;
      const r = V.rng(404);
      const u = Array.from({ length: R4 }, () => r.normal()), v = Array.from({ length: K6 }, () => r.normal());
      const nu = Math.hypot(...u), nv = Math.hypot(...v);
      const W0 = V.randMatrix(R4, K6, 9, 0.5);
      const OPT = {
        sgd: { t: 'SGD', b: 0, st: [] },
        mom: { t: 'SGD + momento', b: 4, st: ['V'] },
        adamw: { t: 'AdamW', b: 8, st: ['m', 'v'] },
        adam8: { t: 'AdamW 8 bits', b: 2, st: ['m8', 'v8'] },
        ada: { t: 'Adafactor', b: 0.002, st: ['R', 'Cc', 'vh'] },
        muon: { t: 'Muon', b: 4, st: ['M', 'O'] },
      };
      const S = { opt: 'adamw', step: 0, t0: 0, W: null, Wp: null, G: null, st: {}, stp: {} };
      const zeros = () => Array.from({ length: R4 }, () => new Array(K6).fill(0));
      function reset() {
        S.W = W0.map((x) => x.slice()); S.Wp = S.W; S.G = zeros(); S.step = 0;
        S.st = { V: zeros(), m: zeros(), v: zeros(), M: zeros(), O: zeros(), R: new Array(R4).fill(0), Cc: new Array(K6).fill(0) };
        S.stp = JSON.parse(JSON.stringify(S.st)); S.Gp = zeros();
        for (let i = 0; i < 3; i++) doStep();
      }
      reset();
      const q8 = (X) => { // blocos de 12 (2 linhas cada)
        const out = zeros(), sc = [];
        for (let b = 0; b < 2; b++) {
          let am = 0; for (let i = b * 2; i < b * 2 + 2; i++) for (let j = 0; j < K6; j++) am = Math.max(am, Math.abs(X[i][j]));
          sc.push(am);
          for (let i = b * 2; i < b * 2 + 2; i++) for (let j = 0; j < K6; j++) out[i][j] = am ? Math.round((127 * X[i][j]) / am) : 0;
        }
        return { codes: out, sc };
      };
      function doStep() {
        S.Wp = S.W.map((x) => x.slice()); S.stp = JSON.parse(JSON.stringify(S.st)); S.Gp = S.G;
        S.step++;
        const k = S.step, amp = 1.6;
        S.G = Array.from({ length: R4 }, (_, i) => Array.from({ length: K6 }, (_, j) => (amp * u[i] * v[j]) / (nu * nv) + 0.35 * r.normal()));
        const G = S.G, st = S.st, eta = 0.06;
        st.V = st.V.map((row, i) => row.map((x, j) => 0.9 * x + G[i][j]));
        st.m = st.m.map((row, i) => row.map((x, j) => 0.9 * x + 0.1 * G[i][j]));
        st.v = st.v.map((row, i) => row.map((x, j) => 0.95 * x + 0.05 * G[i][j] * G[i][j]));
        st.M = st.M.map((row, i) => row.map((x, j) => 0.95 * x + G[i][j]));
        st.O = ns5(st.M);
        st.R = st.R.map((x, i) => 0.9 * x + 0.1 * G[i].reduce((s, y) => s + y * y, 0) / K6);
        st.Cc = st.Cc.map((x, j) => 0.9 * x + 0.1 * G.reduce((s, row) => s + row[j] * row[j], 0) / R4);
        const W = S.W, o = S.opt;
        const b1 = 1 - Math.pow(0.9, k), b2 = 1 - Math.pow(0.95, k);
        for (let i = 0; i < R4; i++) for (let j = 0; j < K6; j++) {
          let d;
          if (o === 'sgd') d = eta * G[i][j];
          else if (o === 'mom') d = eta * 0.3 * st.V[i][j];
          else if (o === 'adamw' || o === 'adam8') d = eta * ((st.m[i][j] / b1) / (Math.sqrt(st.v[i][j] / b2) + 1e-8) + 0.01 * W[i][j]);
          else if (o === 'ada') { const mR = st.R.reduce((s, x) => s + x, 0) / R4; const vh = (st.R[i] * st.Cc[j]) / (mR || 1); d = eta * (G[i][j] / (Math.sqrt(vh) + 1e-8)); }
          else d = eta * st.O[i][j] * Math.sqrt(Math.max(R4, K6)) * 0.4;
          W[i][j] -= d;
        }
        S.t0 = performance.now() / 1000;
      }
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Otimizador', options: Object.keys(OPT).map((k) => ({ v: k, label: OPT[k].t })), value: 'adamw', onChange: (v) => { S.opt = v; reset(); upd(); } });
      UI.button(ctl, '↺ reiniciar', () => { reset(); });
      const rB = UI.readout(ctl, 'buffers/param', '—'), rG = UI.readout(ctl, 'para 8,03 B params', '—');
      function upd() {
        const b = OPT[S.opt].b;
        rB.set(S.opt === 'ada' ? '≈ 0 (sublinear)' : S.opt === 'adam8' ? '~2 B' : b + ' B');
        rG.set(S.opt === 'ada' ? '≈ 0 GB' : fmtGB(b * 8.03e9));
      }
      upd();
      let last = 0;
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        if (t - last > 2.6 || !S.step) { last = t; doStep(); }
        const e = Math.min(1, (t - last) / 0.9);
        const ee = V.ease.inOut(e);
        const narrow = w < 600;
        const o = S.opt;
        const cell = narrow ? Math.max(15, Math.min(22, (w - 70) / 12.6)) : Math.min(34, (w - 90) / 20);
        const pw = K6 * (cell + 2) + 8, ph = R4 * (cell + 2) + 34;
        const panels = [];
        const Wd = lerpM(S.Wp, S.W, ee);
        panels.push({ t: 'W (pesos)', data: Wd, cmap: 'div', glow: e > 0.55 && e < 1 ? C.a : null, mx: 1.2 });
        panels.push({ t: 'G = ∂L/∂W', data: S.G, cmap: 'div', glow: e < 0.35 ? C.b : null, mx: 2 });
        const st = S.st, sp = S.stp, mix = (k) => lerpM(sp[k], st[k], ee);
        if (o === 'mom') panels.push({ t: 'V (momento)', data: mix('V'), cmap: 'div', glow: e > 0.25 && e < 0.7 ? C.c : null, mx: 6 });
        if (o === 'adamw') { panels.push({ t: 'm (média de g)', data: mix('m'), cmap: 'div', glow: e > 0.25 && e < 0.7 ? C.c : null, mx: 1.5 }); const vM = mix('v'); panels.push({ t: 'v (média de g²)', data: vM, cmap: 'heat', glow: e > 0.25 && e < 0.7 ? C.f : null, mx: Math.max(0.05, ...vM.flat()), mn: 0, digits: 2 }); }
        if (o === 'adam8') { const a = q8(st.m), b = q8(st.v); panels.push({ t: 'm em int8 · escalas ' + a.sc.map((x) => x.toFixed(2)).join(' / '), data: a.codes, cmap: 'div', glow: e > 0.25 && e < 0.7 ? C.c : null, mx: 127, digits: 0 }); panels.push({ t: 'v em int8 · escalas ' + b.sc.map((x) => x.toFixed(2)).join(' / '), data: b.codes, cmap: 'heat', glow: e > 0.25 && e < 0.7 ? C.f : null, mx: 127, mn: 0, digits: 0 }); }
        if (o === 'ada') {
          const mixV = (k) => sp[k].map((x, i) => x + (st[k][i] - x) * ee);
          const Rv = mixV('R'), Cv = mixV('Cc');
          const mR = Rv.reduce((s, x) => s + x, 0) / R4 || 1;
          panels.push({ t: 'linhas R + colunas C', vec: true, Rv, Cv, glow: e > 0.25 && e < 0.7 ? C.f : null });
          const vhM = Rv.map((x) => Cv.map((y) => (x * y) / mR));
          panels.push({ t: 'v̂ = R·Cᵀ / média(R)', data: vhM, cmap: 'heat', mx: Math.max(1e-6, ...vhM.flat()), mn: 0, digits: 2, faint: true });
        }
        if (o === 'muon') { panels.push({ t: 'M (momento)', data: mix('M'), cmap: 'div', glow: e > 0.25 && e < 0.6 ? C.c : null, mx: 8 }); panels.push({ t: 'O = NS₅(M)', data: mix('O'), cmap: 'div', glow: e > 0.45 && e < 0.8 ? C.d : null, mx: 0.8 }); panels.push({ t: 'valores singulares', sv: true }); }
        if (o === 'sgd') panels.push({ t: 'estado do otimizador', empty: true });
        const perRow = Math.max(1, Math.floor((w - 16 + 8) / (pw + 8)));
        const gapX = Math.max(8, (w - 16 - perRow * pw) / Math.max(1, perRow));
        // altura da figura = linhas de painéis que este otimizador precisa (sem sobra de tela vazia)
        const nRows = Math.ceil(panels.length / perRow);
        const contentH = 26 + (nRows - 1) * (ph + 18) + R4 * (cell + 2) + 46;
        const wantH = Math.round(Math.max(300, contentH));
        if (Math.abs(stgO.clientHeight - wantH) > 2) stgO.style.height = wantH + 'px';
        const oy = Math.max(0, Math.floor((h - contentH) / 2)); // centraliza quando sobra altura (uma linha só)
        panels.forEach((p, idx) => {
          const col = idx % perRow, row = Math.floor(idx / perRow);
          const x = 12 + col * (pw + gapX), y = oy + 26 + row * (ph + 18);
          V.text(g, p.t, x, y - 8, { size: 11.5, font: 'mono', color: C.ink, maxW: pw + gapX - 6 });
          if (p.glow) V.glow(g, x + pw / 2, y + (ph - 34) / 2, pw * 0.6, p.glow, 0.28);
          if (p.empty) {
            g.setLineDash([4, 4]); V.box(g, x, y, pw - 8, ph - 34, { r: 8, stroke: V.alpha(C.muted, 0.6), lw: 1.2 }); g.setLineDash([]);
            V.text(g, 'nada: W ← W − η·G', x + (pw - 8) / 2, y + (ph - 34) / 2 + 4, { size: 11.5, align: 'center', color: C.muted, font: 'mono' });
            return;
          }
          if (p.vec) {
            const cw = cell;
            const mxv = Math.max(...p.Rv, ...p.Cv, 1e-9);
            const vt = (z) => (z < 0.005 ? '0' : z.toFixed(2).replace(/^0/, ''));
            for (let i = 0; i < R4; i++) { const tv = p.Rv[i] / mxv; g.fillStyle = V.cmap(tv, 'heat'); V.rr(g, x, y + i * (cw + 2), cw, cw, 3); g.fill(); if (cw >= 18) V.text(g, vt(p.Rv[i]), x + cw / 2, y + i * (cw + 2) + cw / 2 + 4, { size: 10.5, font: 'mono', align: 'center', color: tv > 0.6 ? (C.dark ? '#00210F' : '#fff') : C.ink }); }
            for (let j = 0; j < K6; j++) { const tv = p.Cv[j] / mxv; g.fillStyle = V.cmap(tv, 'heat'); V.rr(g, x + cw + 10 + j * (cw + 2), y, cw, cw, 3); g.fill(); if (cw >= 18) V.text(g, vt(p.Cv[j]), x + cw + 10 + j * (cw + 2) + cw / 2, y + cw / 2 + 4, { size: 10.5, font: 'mono', align: 'center', color: tv > 0.6 ? (C.dark ? '#00210F' : '#fff') : C.ink }); }
            V.text(g, R4 + ' + ' + K6 + ' números', x + cw + 10, y + cw + 18, { size: 11, color: C.muted, font: 'mono' });
            V.text(g, 'em vez de ' + R4 * K6, x + cw + 10, y + cw + 32, { size: 11, color: C.muted, font: 'mono' });
            return;
          }
          if (p.sv) {
            const a = svals(st.M), b = svals(st.O);
            const bw2 = (pw - 30) / 8, bh2 = ph - 72;
            [[a, C.c, 'σ(M)'], [b, C.d, 'σ(O)']].forEach(([arr, colr, lab], gi) => {
              const mxs = Math.max(...arr, 1e-9);
              arr.forEach((s, k) => {
                const hh = (s / mxs) * bh2;
                g.fillStyle = V.alpha(colr, 0.8); V.rr(g, x + gi * (pw / 2 + 2) + k * (bw2 + 3), y + bh2 - hh, bw2, Math.max(1, hh), 3); g.fill();
              });
              V.text(g, lab + ' ' + arr.map((s) => s.toFixed(1)).join(' '), x, y + bh2 + 15 + gi * 15, { size: 11, font: 'mono', color: colr, maxW: pw + gapX - 6 });
            });
            return;
          }
          if (p.faint) g.globalAlpha = 0.65;
          const fm = p.digits === 0 ? (z) => String(z) : p.digits === 2 ? (z) => (z < 0.005 ? '0' : z.toFixed(2).replace(/^0/, '')) : (z) => (Math.abs(z) < 0.05 ? '0' : z.toFixed(1));
          V.matrix(g, { x, y, data: p.data, cell, cmap: p.cmap, values: cell >= 18, fmt: fm, min: p.mn == null ? -p.mx : p.mn, max: p.mx, gap: 2 });
          g.globalAlpha = 1;
        });
        const fb = e < 0.35 ? 'backward: chega o gradiente G' : e < 0.7 ? 'o otimizador atualiza o seu estado' : 'W recebe a atualização';
        V.text(g, 'passo ' + S.step + ' — ' + fb, 12, Math.min(h - 12, oy + contentH - 12), { size: 12, color: C.ink, maxW: w - 24 });
      });
    })();

    /* ================================================ FIG 5 — o relógio da taxa de aprendizado */
    (function () {
      const fig = root.querySelector('#fig-lr');
      adaptH(fig, 420, 500);
      const P = [
        { id: 'tulu8', t: 'Tülu 3 · SFT 8B (full)', peak: 5e-6, warm: 0.03, kind: 'linear', end: 0, cat: 'full' },
        { id: 'tulu70', t: 'Tülu 3 · SFT 70B (full)', peak: 2e-6, warm: 0.03, kind: 'linear', end: 0, cat: 'full' },
        { id: 'tuludpo', t: 'Tülu 3 · DPO 8B', peak: 5e-7, warm: 0.1, kind: 'linear', end: 0, cat: 'dpo' },
        { id: 'trl', t: 'SFTConfig (padrão do TRL)', peak: 2e-5, warm: 0, kind: 'linear', end: 0, cat: 'full' },
        { id: 'mimosft', t: 'MiMo-V2-Flash · SFT', peak: 5e-5, warm: 0, kind: 'cos', end: 5e-6, cat: 'full' },
        { id: 'mimopre', t: 'MiMo-V2-Flash · pré-treino (WSD)', peak: 3.2e-4, warm: 50e9 / 22.05e12, stable: 12.05e12 / 22.05e12, kind: 'wsd', end: 1e-4, cat: 'pre' },
        { id: 'axo', t: 'Axolotl lora-1b (LoRA)', peak: 2e-4, warm: 0.1, kind: 'cos', end: 0, cat: 'lora' },
        { id: 'lf', t: 'LLaMA-Factory Qwen3 (LoRA)', peak: 1e-4, warm: 0.1, kind: 'cos', end: 0, cat: 'lora' },
        { id: 'mlx', t: 'mlx-lm (padrão, LoRA)', peak: 1e-5, warm: 0, kind: 'const', end: 1e-5, cat: 'lora' },
      ];
      const CAT = { full: 'a', lora: 'b', dpo: 'c', pre: 'd' };
      const lrAt = (p, x) => {
        if (p.warm > 0 && x < p.warm) return (p.peak * x) / p.warm;
        if (p.kind === 'const') return p.peak;
        if (p.kind === 'wsd') { if (x < p.stable) return p.peak; const q = (x - p.stable) / (1 - p.stable); return p.end + (p.peak - p.end) * 0.5 * (1 + Math.cos(Math.PI * q)); }
        const q = (x - p.warm) / (1 - p.warm);
        if (p.kind === 'linear') return p.end + (p.peak - p.end) * (1 - q);
        return p.end + (p.peak - p.end) * 0.5 * (1 + Math.cos(Math.PI * q));
      };
      const S = { sel: 'tulu8', log: true, all: true };
      const ctl = UI.controls(fig);
      UI.select(ctl, { label: 'Configuração', options: P.map((p) => ({ v: p.id, label: p.t })), value: 'tulu8', onChange: (v) => { S.sel = v; upd(); } });
      UI.toggle(ctl, { label: 'eixo log', value: true, onChange: (v) => { S.log = v; } });
      UI.toggle(ctl, { label: 'mostrar as outras', value: true, onChange: (v) => { S.all = v; } });
      const rP = UI.readout(ctl, 'pico', '—'), rN = UI.readout(ctl, 'agora', '—'), rS = UI.readout(ctl, 'forma', '—');
      const sel = () => P.find((p) => p.id === S.sel);
      function upd() { const p = sel(); rP.set(sci(p.peak, 1)); rS.set({ linear: 'warmup + linear', cos: 'warmup + cosseno', wsd: 'WSD', const: 'constante' }[p.kind]); }
      upd();
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const narrow = w < 560;
        const x0 = narrow ? 52 : 64, x1 = w - 16, y0 = 24, y1 = h - (narrow ? 70 : 52);
        const ymin = 1e-7, ymax = 1e-3;
        const p = sel();
        const yMaxLin = S.all ? 3.4e-4 : p.peak * 1.15;
        const Y = (v) => {
          if (S.log) { const lv = Math.log10(Math.max(ymin, v)); return y1 - ((lv - Math.log10(ymin)) / (Math.log10(ymax) - Math.log10(ymin))) * (y1 - y0); }
          return y1 - (v / yMaxLin) * (y1 - y0);
        };
        const X = (x) => x0 + x * (x1 - x0);
        // grade
        if (S.log) for (let e = -7; e <= -3; e++) { const yy = Y(Math.pow(10, e)); V.line(g, x0, yy, x1, yy, { color: C.grid, lw: 1 }); V.text(g, '1e' + e, x0 - 6, yy + 4, { size: 11, align: 'right', color: C.muted, font: 'mono' }); }
        else { const st = niceStep(yMaxLin, 4); for (let v = 0; v <= yMaxLin + 1e-12; v += st) { const yy = Y(v); V.line(g, x0, yy, x1, yy, { color: C.grid, lw: 1 }); V.text(g, v ? sci(v, 1) : '0', x0 - 6, yy + 4, { size: 11, align: 'right', color: C.muted, font: 'mono' }); } }
        V.line(g, x0, y1, x1, y1, { color: C.muted, lw: 1 });
        [0, 0.25, 0.5, 0.75, 1].forEach((x) => V.text(g, Math.round(x * 100) + ' %', X(x), y1 + 15, { size: 11, align: x === 0 ? 'left' : x === 1 ? 'right' : 'center', color: C.muted, font: 'mono' }));
        // outras
        if (S.all) P.forEach((q) => {
          if (q.id === p.id) return;
          g.beginPath();
          for (let i = 0; i <= 160; i++) { const x = i / 160, yy = Y(lrAt(q, x)); if (!i) g.moveTo(X(x), yy); else g.lineTo(X(x), yy); }
          g.strokeStyle = V.alpha(C[CAT[q.cat]], 0.3); g.lineWidth = 1.3; g.stroke();
        });
        // warmup
        if (p.warm > 0) { g.fillStyle = V.alpha(C.d, 0.12); g.fillRect(X(0), y0, Math.max(2, X(p.warm) - X(0)), y1 - y0); V.text(g, 'warmup', X(p.warm) + 4, y0 + 12, { size: 11, color: C.d }); }
        const col = C[CAT[p.cat]];
        g.beginPath();
        for (let i = 0; i <= 300; i++) { const x = i / 300, yy = Y(lrAt(p, x)); if (!i) g.moveTo(X(x), yy); else g.lineTo(X(x), yy); }
        g.strokeStyle = col; g.lineWidth = 2.8; g.stroke();
        const xb = (t % 7) / 7, lb = lrAt(p, xb);
        for (let k = 1; k <= 6; k++) { const xx = Math.max(0, xb - k * 0.012); V.dot(g, X(xx), Y(lrAt(p, xx)), 4 - k * 0.5, V.alpha(col, 0.5 - k * 0.07)); }
        V.glow(g, X(xb), Y(lb), 16, col, 0.6); face(g, X(xb), Y(lb), 15, col);
        rN.set(sci(lb, 2));
        V.text(g, p.t + ' · pico ' + sci(p.peak, 1) + (p.end && p.kind !== 'const' ? ' → ' + sci(p.end, 1) : ''), x0 + 4, y0 - 6, { size: 12, weight: 600, color: col, maxW: x1 - x0 });
        // legenda de categorias
        const items = [['full FT', 'a'], ['LoRA', 'b'], ['DPO', 'c'], ['pré-treino', 'd']];
        let lx = x0, ly = h - (narrow ? 34 : 14);
        items.forEach(([lab, k], i) => {
          const wdt = tw(g, lab, 11.5) + 26;
          if (lx + wdt > x1) { lx = x0; ly += 16; }
          g.fillStyle = C[k]; V.rr(g, lx, ly - 8, 14, 8, 2); g.fill();
          V.text(g, lab, lx + 18, ly, { size: 11.5, color: C.ink });
          lx += wdt;
        });
        if (p.id === 'mimopre') V.text(g, '50 B tokens de warmup · 12T constante · 10T em cosseno', x1, y1 - 8, { size: 11, align: 'right', color: C.muted, maxW: x1 - x0 });
      });
    })();

    /* ================================================ FIG 6 — ZeRO em barras */
    (function () {
      const fig = root.querySelector('#fig-zero');
      adaptH(fig, 440, 560);
      const S = { st: 'z3', nd: 2, m: '8b' };
      const MOD = { '8b': { n: 'Llama-3.1-8B', psi: 8.03e9 }, '75': { n: '7,5 B (artigo)', psi: 7.5e9 } };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Estágio', options: [{ v: 'ddp', label: 'DDP' }, { v: 'z1', label: 'ZeRO-1' }, { v: 'z2', label: 'ZeRO-2' }, { v: 'z3', label: 'ZeRO-3 / FSDP' }], value: 'z3', onChange: (v) => { S.st = v; upd(); } });
      nocase(UI.seg(ctl, { label: 'N_d', options: [2, 4, 8, 64].map((n) => ({ v: n, label: String(n) })), value: 2, onChange: (v) => { S.nd = v; upd(); } }));
      UI.seg(ctl, { label: 'Modelo', options: [{ v: '8b', label: 'Llama-3.1-8B' }, { v: '75', label: '7,5 B (artigo ZeRO)' }], value: '8b', onChange: (v) => { S.m = v; upd(); } });
      const rP = UI.readout(ctl, 'por processo', '—'), rC = UI.readout(ctl, 'comunicação/passo', '—'), rT = UI.readout(ctl, 'rede (2 Sparks)', '—');
      const parts = () => {
        const psi = MOD[S.m].psi, n = S.nd, st = S.st;
        const shP = st === 'z3', shG = st === 'z2' || st === 'z3', shO = st !== 'ddp';
        return { psi, n, shP, shG, shO, p: 2 * psi / (shP ? n : 1), g: 2 * psi / (shG ? n : 1), o: 12 * psi / (shO ? n : 1) };
      };
      function upd() {
        const q = parts();
        const tot = q.p + q.g + q.o;
        rP.set(fmtGB(tot), S.m === '8b' ? fitSpark(tot)[1] : undefined);
        const el = (S.st === 'z3' ? 3 : 2) * q.psi;
        rC.set((S.st === 'z3' ? '3Ψ' : '2Ψ') + ' = ' + fmtGB(el * 2));
        if (S.m === '8b' && S.nd === 2) rT.set('~' + fmt((el * 2 * 0.5) / 20e9, 1) + ' s'); else rT.set('—');
      }
      upd();
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const q = parts();
        const narrow = w < 560;
        const nDraw = Math.min(q.n, 8);
        const left = narrow ? 44 : 64, right = w - 14;
        const top = 66, bot = h - 74;
        const full = S.m === '8b' ? Math.max(16 * q.psi, SPARK) * 1.04 : 16 * q.psi;
        const Hs = (b) => (b / full) * (bot - top);
        const colW = Math.min(70, ((right - left) / nDraw) * 0.62);
        const step = (right - left) / nDraw;
        // eixo
        const tk = niceStep(full / 1e9, 4);
        for (let v = 0; v <= full / 1e9 + 1e-9; v += tk) { const y = bot - Hs(v * 1e9); V.line(g, left - 4, y, right, y, { color: C.grid, lw: 1 }); V.text(g, v.toFixed(0), left - 8, y + 4, { size: 11, align: 'right', color: C.muted, font: 'mono' }); }
        V.text(g, 'GB', left - 8, top - 10, { size: 11, align: 'right', color: C.muted });
        if (S.m === '8b') {
          const yS = bot - Hs(SPARK), yP1 = bot - Hs(PRAT_HI), yP0 = bot - Hs(PRAT_LO);
          g.fillStyle = V.alpha(C.d, 0.1); g.fillRect(left, yP1, right - left, yP0 - yP1);
          V.line(g, left, yS, right, yS, { color: RED(), lw: 1.6, dash: [5, 4] });
        }
        const comps = [['pesos', 2, q.shP, C.a], ['gradientes', 2, q.shG, C.b], ['otimizador', 12, q.shO, C.c]];
        const cyc = 4.2, tt = (t % cyc) / cyc;
        for (let d = 0; d < nDraw; d++) {
          const cx = left + step * d + step / 2, x = cx - colW / 2;
          let yb = bot, used = 0;
          comps.forEach(([lab, k, sh, col]) => {
            const hFull = Hs(k * q.psi);
            if (!sh) { V.box(g, x, yb - hFull, colW, hFull - 1, { r: 3, fill: V.alpha(col, 0.6), stroke: col, lw: 1 }); used += k * q.psi; }
            else {
              const slice = hFull / q.n;
              g.setLineDash([3, 3]); V.box(g, x, yb - hFull, colW, hFull - 1, { r: 3, stroke: V.alpha(col, 0.45), lw: 1 }); g.setLineDash([]);
              const idx = q.n <= 8 ? d : Math.round((d * (q.n - 1)) / 7);
              const ys = yb - (idx + 1) * slice;
              V.box(g, x, ys, colW, Math.max(1.5, slice - (slice > 3 ? 1 : 0)), { r: Math.min(3, slice / 2), fill: V.alpha(col, 0.85), stroke: col, lw: slice > 3 ? 1 : 0.5 });
              used += (k * q.psi) / q.n;
            }
            yb -= hFull;
          });
          const over = S.m === '8b' && used > PRAT_HI;
          face(g, cx, bot + 22, 20, over ? RED() : C.a, { mood: over ? 'sad' : undefined, sweat: over ? t * 0.8 + d * 0.3 : 0 });
          V.text(g, fmt(used / 1e9, 1), cx, bot + 48, { size: 11, align: 'center', color: C.ink, font: 'mono' });
        }
        if (q.n > 8) V.text(g, '… ' + q.n + ' processos (8 desenhados)', right, bot + 64, { size: 11, align: 'right', color: C.muted });
        if (S.m === '8b') {
          const yS = bot - Hs(SPARK), lbl = 'Spark: 121,69 GiB', lw3 = tw(g, lbl, 11, 'mono') + 10;
          V.box(g, left + 4, yS + 3, lw3, 16, { r: 4, fill: C.bg || C.panel, stroke: V.alpha(RED(), 0.6), lw: 1 });
          V.text(g, lbl, left + 9, yS + 15, { size: 11, color: RED(), font: 'mono' });
        }
        // coletivas
        const ry = 12;
        let lgx = right;
        [['otimizador', C.c], ['gradientes', C.b], ['pesos', C.a]].forEach(([lab, col]) => { const wd = tw(g, lab, 11.5) + 22; lgx -= wd; g.fillStyle = V.alpha(col, 0.85); V.rr(g, lgx, 23, 12, 10, 2); g.fill(); V.text(g, lab, lgx + 16, 32, { size: 11.5, color: C.ink }); });
        const lab = S.st === 'ddp' ? (tt < 0.5 ? 'all-reduce dos gradientes' : 'cada processo aplica o passo inteiro na sua cópia') : tt < 0.5 ? 'reduce-scatter dos gradientes' : S.st === 'z3' ? 'all-gather dos parâmetros antes de cada camada (fwd e bwd)' : 'cada um atualiza a sua fatia; all-gather dos parâmetros';
        V.text(g, lab, left, ry + 4, { size: 11.5, color: tt < 0.5 ? C.b : C.a, font: 'mono', maxW: right - left });
        const f = (tt % 0.5) / 0.5;
        const colPulse = tt < 0.5 ? C.b : C.a;
        if (nDraw > 1 && !(S.st === 'ddp' && tt >= 0.5)) {
          const yArc = top - 2;
          for (let d = 0; d < nDraw; d++) {
            const xa = left + step * d + step / 2;
            const last = d === nDraw - 1;
            const xb2 = last ? left + step / 2 : xa + step;
            const cv = last ? 34 : -16;
            V.arrow(g, xa, yArc, xb2, yArc, { color: V.alpha(colPulse, 0.5), lw: 1.3, curve: cv, head: 6, dash: [4, 3] });
            const mxc = (xa + xb2) / 2, myc = yArc + (last ? -cv : cv);
            const pulses = S.st === 'z3' && tt >= 0.5 ? 2 : 1;
            for (let k = 0; k < pulses; k++) {
              const uu = (f + k * 0.5) % 1, a1 = (1 - uu) * (1 - uu), a2 = 2 * (1 - uu) * uu, a3 = uu * uu;
              V.glow(g, a1 * xa + a2 * mxc + a3 * xb2, a1 * yArc + a2 * myc + a3 * yArc, 11, colPulse, 1);
            }
          }
        }
      });
    })();

    /* ================================================ FIG 7 — PP, TP e EP com backward */
    (function () {
      const fig = root.querySelector('#fig-paralelos');
      const S = { mode: 'pp', p: 4, m: 8, b: 8, sim: null };
      const stgP = fig.querySelector('.la-fig-stage');
      const setHP = () => { const want = (fig.clientWidth < 560 ? { pp: 380, tp: 440, ep: 420 }[S.mode] : 450) + 'px'; if (stgP.style.height !== want) stgP.style.height = want; };
      setHP();
      const roP = new ResizeObserver(setHP); roP.observe(fig);
      if (LA.onCleanup) LA.onCleanup(() => roP.disconnect());
      function sim1F1B(p, m) {
        const tF = 1, tB = 2, ops = [];
        for (let s = 0; s < p; s++) { const warm = Math.min(p - s - 1, m), Lst = []; for (let i = 0; i < warm; i++) Lst.push({ k: 'F', i }); let f = warm, b = 0; while (f < m) { Lst.push({ k: 'F', i: f++ }); Lst.push({ k: 'B', i: b++ }); } while (b < m) Lst.push({ k: 'B', i: b++ }); ops.push(Lst); }
        const fF = Array.from({ length: p }, () => new Array(m).fill(-1)), fB = Array.from({ length: p }, () => new Array(m).fill(-1));
        const ptr = new Array(p).fill(0), free = new Array(p).fill(0), out = [];
        let rem = p * 2 * m, guard = 0;
        while (rem > 0 && guard++ < 20000) {
          let best = null;
          for (let s = 0; s < p; s++) {
            if (ptr[s] >= ops[s].length) continue;
            const op = ops[s][ptr[s]];
            const dep = op.k === 'F' ? (s === 0 ? 0 : fF[s - 1][op.i]) : (s === p - 1 ? fF[s][op.i] : fB[s + 1][op.i]);
            if (dep < 0) continue;
            const st = Math.max(free[s], dep);
            if (!best || st < best.st) best = { s, op, st };
          }
          if (!best) break;
          const en = best.st + (best.op.k === 'F' ? tF : tB);
          if (best.op.k === 'F') fF[best.s][best.op.i] = en; else fB[best.s][best.op.i] = en;
          free[best.s] = en; ptr[best.s]++; rem--;
          out.push({ s: best.s, k: best.op.k, i: best.op.i, t0: best.st, t1: en });
        }
        const T = Math.max(...free), idle = [];
        for (let s = 0; s < p; s++) {
          const bs = out.filter((b) => b.s === s).sort((a, b) => a.t0 - b.t0);
          let tcur = 0;
          bs.forEach((b) => { if (b.t0 > tcur + 1e-9) idle.push({ s, t0: tcur, t1: b.t0 }); tcur = b.t1; });
          if (T > tcur + 1e-9) idle.push({ s, t0: tcur, t1: T });
        }
        return { blocks: out, idle, T, ideal: m * (tF + tB) };
      }
      S.sim = sim1F1B(S.p, S.m);
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Modo', options: [{ v: 'pp', label: 'PP · 1F1B' }, { v: 'tp', label: 'TP · Megatron' }, { v: 'ep', label: 'EP · all-to-all' }], value: 'pp', onChange: (v) => { S.mode = v; gP.show(v === 'pp'); gT.show(v === 'tp'); gE.show(v === 'ep'); setHP(); upd(); } });
      const gP = UI.group(ctl), gT = UI.group(ctl), gE = UI.group(ctl);
      nocase(UI.slider(gP.el, { label: 'estágios p', min: 2, max: 8, step: 1, value: 4, onInput: (v) => { S.p = v; S.sim = sim1F1B(S.p, S.m); upd(); } }));
      nocase(UI.slider(gP.el, { label: 'micro-lotes m', min: 1, max: 16, step: 1, value: 8, onInput: (v) => { S.m = v; S.sim = sim1F1B(S.p, S.m); upd(); } }));
      nocase(UI.slider(gT.el, { label: 'micro-lote b', min: 1, max: 16, step: 1, value: 8, onInput: (v) => { S.b = v; upd(); } }));
      gT.show(false);
      const r1 = UI.readout(ctl, '—', '—'), r2 = UI.readout(ctl, '—', '—');
      const setLab = (ro, l) => { ro.el.querySelector('span').textContent = l; };
      function upd() {
        if (S.mode === 'pp') {
          setLab(r1, 'bolha (p−1)/m'); r1.set(fmt(((S.p - 1) / S.m) * 100, 1) + ' %');
          setLab(r2, 'medida na simulação'); r2.set(fmt(((S.sim.T - S.sim.ideal) / S.sim.ideal) * 100, 1) + ' %');
        } else if (S.mode === 'tp') {
          const by = 2048 * S.b * 4096 * 2;
          setLab(r1, 'por all-reduce'); r1.set(fmt(by / 1e6, 0) + ' MB');
          const tot = by * 4 * 32, tc = (6 * 8.03e9 * 2048 * S.b) / 2 / 50e12, tn = tot / 20e9;
          setLab(r2, 'rede vs compute'); r2.set(fmt(tn, 2) + ' s / ' + fmt(tc, 1) + ' s (' + fmt((tn / tc) * 100, 0) + ' %)');
        } else {
          setLab(r1, 'all-to-all por camada MoE'); r1.set('2 fwd + 2 bwd');
          setLab(r2, 'tokens que cruzam o cabo'); r2.set(fmt(crossFrac() * 100, 0) + ' %');
        }
      }
      // EP: tokens
      const er = V.rng(7070);
      let toks = [];
      const newToks = () => { toks = []; for (let n = 0; n < 2; n++) for (let k = 0; k < 6; k++) toks.push({ home: n, slot: k, e: er.int(8) }); };
      newToks();
      const crossFrac = () => toks.filter((x) => (x.e < 4 ? 0 : 1) !== x.home).length / toks.length;
      UI.button(gE.el, '↻ novos tokens', () => { newToks(); upd(); });
      gE.show(false);
      upd();
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const narrow = w < 560;
        if (S.mode === 'pp') {
          const sim = S.sim, p = S.p;
          const left = narrow ? 50 : 76, right = w - 14, top = 30, rowH = Math.min(66, (h - 112) / p);
          const X = (x) => left + (x / sim.T) * (right - left);
          const per = 7, cur = ((t % per) / per) * sim.T * 1.08;
          V.text(g, '1F1B com p = ' + p + ' estágios e m = ' + S.m + ' micro-lotes', left, 18, { size: 12, weight: 600, color: C.ink, maxW: right - left });
          for (let s = 0; s < p; s++) {
            const y = top + s * rowH;
            V.text(g, (narrow ? 'est. ' : 'estágio ') + s, left - 6, y + rowH / 2 + 4, { size: 11, align: 'right', color: C.muted, font: 'mono' });
            g.fillStyle = V.alpha(C.muted, 0.06); g.fillRect(left, y + 3, right - left, rowH - 6);
          }
          sim.idle.forEach((id) => {
            const y = top + id.s * rowH, x1 = X(id.t0), x2 = X(id.t1);
            const on = cur >= id.t0;
            g.fillStyle = V.alpha(RED(), on ? 0.32 : 0.1); V.rr(g, x1 + 1, y + 4, Math.max(1, x2 - x1 - 2), rowH - 8, 4); g.fill();
            if (on && x2 - x1 > tw(g, 'espera', 11) + 8) V.text(g, 'espera', (x1 + x2) / 2, y + rowH / 2 + 4, { size: 11, align: 'center', color: RED() });
          });
          sim.blocks.forEach((bk) => {
            const y = top + bk.s * rowH;
            const x1 = X(bk.t0), x2 = X(bk.t1);
            const vis = cur >= bk.t0;
            const col = bk.k === 'F' ? C.a : C.b;
            const ww = vis ? (Math.min(cur, bk.t1) - bk.t0) / (bk.t1 - bk.t0) : 0;
            if (!vis) { V.box(g, x1 + 1, y + 4, x2 - x1 - 2, rowH - 8, { r: 4, fill: V.alpha(C.panel || C.bg, 0.9), stroke: V.alpha(col, 0.35), lw: 1 }); return; }
            V.box(g, x1 + 1, y + 4, Math.max(2, (x2 - x1 - 2) * ww), rowH - 8, { r: 4, fill: V.alpha(col, 0.75), stroke: col, lw: 1 });
            // blocos estreitos (celular) ganham só a letra; os largos, letra + micro-lote
            if (x2 - x1 > 8 && ww > 0.6) V.text(g, bk.k + (x2 - x1 > 16 ? bk.i : ''), (x1 + x2) / 2, y + rowH / 2 + 4, { size: 11, align: 'center', color: C.dark ? '#06140d' : '#fff', font: 'mono', weight: 600 });
          });
          // blocos ainda não começados escondem a bolha "futura"; desenha o cursor
          const xc = X(Math.min(cur, sim.T));
          V.line(g, xc, top - 4, xc, top + p * rowH + 4, { color: C.d, lw: 2 });
          const yL = top + p * rowH + 22;
          V.text(g, 'tempo →  total ' + sim.T + ' unidades · ideal sem bolha ' + sim.ideal + ' (F = 1, B = 2)', left, yL, { size: 11.5, color: C.muted, font: 'mono', maxW: right - left });
          const items = [[C.a, 'forward'], [C.b, 'backward (2× mais longo)'], [RED(), 'bolha: estágio parado']];
          let lx = left; let ly = yL + 20;
          items.forEach(([col, lab]) => { const wd = tw(g, lab, 11.5) + 26; if (lx + wd > right) { lx = left; ly += 16; } g.fillStyle = V.alpha(col, col === RED() ? 0.4 : 0.8); V.rr(g, lx, ly - 9, 14, 10, 2); g.fill(); V.text(g, lab, lx + 18, ly, { size: 11.5, color: C.ink }); lx += wd; });
          return;
        }
        if (S.mode === 'tp') {
          const cyc = 7, tt = (t % cyc) / cyc;
          const fwd = tt < 0.5, f = (tt % 0.5) / 0.5;
          const nodeW = narrow ? (w - 40) / 2 : Math.min(240, (w - 200) / 2);
          const midX = w / 2, gapN = narrow ? 12 : 70;
          const nA = midX - nodeW - gapN / 2, nB = midX + gapN / 2;
          const xY = 22, ny = 92, chH = narrow ? 30 : 32;
          const yW1 = ny + 18, yAct = ny + 18 + chH + 22, yW2 = yAct + chH + 22, nh = yW2 + chH + 18 - ny;
          const yAR = ny + nh + 34, yOut = yAR + 46;
          // X e all-reduce f (backward)
          chip(g, midX - 80, xY, 160, 26, 'X  [b·s × h]', { size: 12, fill: V.alpha(C.ink, 0.06) });
          const fLit = !fwd && f > 0.78;
          const fx = narrow ? 8 : midX + 90, fw = narrow ? Math.max(60, midX - 80 - 16) : 128;
          V.box(g, fx, xY, fw, 26, { r: 13, fill: V.alpha(C.b, fLit ? 0.5 : 0.1), stroke: V.alpha(C.b, 0.9), lw: 1.3 });
          V.text(g, narrow ? 'f: Σ dX' : 'all-reduce f (dX)', fx + fw / 2, xY + 17, { size: 11.5, align: 'center', color: C.ink, font: 'mono', maxW: fw - 6 });
          [[nA, 'nó A (GPU 0)', C.a], [nB, 'nó B (GPU 1)', C.b]].forEach(([x, lab, col], k) => {
            V.box(g, x, ny, nodeW, nh, { r: 12, fill: V.alpha(col, 0.07), stroke: V.alpha(col, 0.6), lw: 1.3 });
            V.text(g, lab, x + nodeW / 2, ny - 8, { size: 12, align: 'center', weight: 600, color: C.ink });
            chip(g, x + 12, yW1, nodeW - 24, chH, 'W₁[:, metade ' + (k + 1) + ']  colunas', { size: narrow ? 10.5 : 11.5 });
            chip(g, x + 12, yAct, nodeW - 24, chH, 'ativação local', { size: 11, fill: V.alpha(C.muted, 0.06) });
            chip(g, x + 12, yW2, nodeW - 24, chH, 'W₂[metade ' + (k + 1) + ', :]  linhas', { size: narrow ? 10.5 : 11.5 });
            const cx = x + nodeW / 2;
            V.arrow(g, midX + (k ? 20 : -20), xY + 26, cx, yW1 - 2, { color: V.alpha(C.muted, 0.6), lw: 1.3, head: 6 });
            V.arrow(g, cx, yW1 + chH, cx, yAct - 2, { color: V.alpha(C.muted, 0.6), lw: 1.3, head: 6 });
            V.arrow(g, cx, yAct + chH, cx, yW2 - 2, { color: V.alpha(C.muted, 0.6), lw: 1.3, head: 6 });
            V.arrow(g, cx, yW2 + chH, midX + (k ? 24 : -24), yAR - 16, { color: V.alpha(C.muted, 0.6), lw: 1.3, head: 6 });
          });
          const gLit = fwd && f > 0.78;
          V.box(g, midX - 80, yAR - 16, 160, 32, { r: 16, fill: V.alpha(C.d, gLit ? 0.5 : 0.12), stroke: C.d, lw: 1.5, label: 'all-reduce g  Σ', size: 12 });
          V.arrow(g, midX, yAR + 16, midX, yOut - 2, { color: V.alpha(C.muted, 0.6), lw: 1.3, head: 6 });
          chip(g, midX - 80, yOut, 160, 26, 'saída  [b·s × h]', { size: 12, fill: V.alpha(C.ink, 0.06) });
          // pulsos
          const path = (k) => { const cx = (k ? nB : nA) + nodeW / 2; return [[midX + (k ? 20 : -20), xY + 26], [cx, yW1 + chH / 2], [cx, yAct + chH / 2], [cx, yW2 + chH / 2], [midX + (k ? 24 : -24), yAR], [midX, yOut + 13]]; };
          [0, 1].forEach((k) => {
            const P = path(k), n = P.length - 1;
            const pos = fwd ? f * n : (1 - f) * n;
            const i = Math.min(n - 1, Math.floor(pos)), fr = pos - i;
            V.glow(g, V.lerp(P[i][0], P[i + 1][0], fr), V.lerp(P[i][1], P[i + 1][1], fr), 13, fwd ? C.a : C.b, 1);
          });
          const capt = fwd ? 'forward: X vai inteiro aos dois nós; cada um calcula metade; o all-reduce g soma as saídas parciais' : 'backward: o gradiente desce sem comunicação até W₁; o all-reduce f soma as parcelas de dX';
          V.wrap(g, capt + '. Atenção + MLP ⇒ 2 all-reduces no forward e 2 no backward, por camada.', 14, yOut + 48, w - 28, { size: 11.5, color: fwd ? C.a : C.b, lineH: 15, maxLines: 3 });
          return;
        }
        // EP
        const cyc = 8, tt = (t % cyc) / cyc;
        const fwd = tt < 0.5, q = (tt % 0.5) / 0.5;
        const ph = q < 0.35 ? 0 : q < 0.6 ? 1 : 2; // despacho, cálculo, retorno
        const pf = ph === 0 ? q / 0.35 : ph === 1 ? (q - 0.35) / 0.25 : (q - 0.6) / 0.4;
        const nodeW = narrow ? (w - 40) / 2 : Math.min(280, (w - 120) / 2), midX = w / 2;
        const nA = midX - nodeW - 18, nB = midX + 18, ny = 46, nh = h - 120;
        [[nA, 'nó A · experts 0–3'], [nB, 'nó B · experts 4–7']].forEach(([x, lab], k) => {
          V.box(g, x, ny, nodeW, nh, { r: 12, fill: V.alpha(k ? C.b : C.a, 0.06), stroke: V.alpha(k ? C.b : C.a, 0.6), lw: 1.3 });
          V.text(g, lab, x + nodeW / 2, ny - 8, { size: 12, align: 'center', weight: 600, color: C.ink });
        });
        const expPos = (e) => { const n = e < 4 ? 0 : 1, k = e % 4; const x = (n ? nB : nA) + 14 + k * ((nodeW - 28) / 4) + (nodeW - 28) / 8; return [x, ny + nh - 40]; };
        for (let e = 0; e < 8; e++) { const [x, y] = expPos(e); const busy = ph === 1 && toks.some((tk) => tk.e === e); V.box(g, x - 16, y - 16, 32, 32, { r: 8, fill: V.alpha(C.d, busy ? 0.5 + 0.3 * Math.sin(t * 10) : 0.15), stroke: C.d, lw: 1.2, label: 'E' + e, size: 11.5 }); }
        const homePos = (tk) => { const x0 = (tk.home ? nB : nA) + 18 + (tk.slot % 3) * ((nodeW - 36) / 3) + 10; const y0 = ny + 22 + Math.floor(tk.slot / 3) * 30; return [x0, y0]; };
        V.line(g, midX, ny + 10, midX, ny + nh - 10, { color: V.alpha(C.muted, 0.5), lw: 1, dash: [4, 4] });
        toks.forEach((tk) => {
          const [hx, hy] = homePos(tk), [ex, ey] = expPos(tk.e);
          const cross = (tk.e < 4 ? 0 : 1) !== tk.home;
          if (!cross) return;
          g.setLineDash([3, 4]); g.beginPath(); g.moveTo(hx, hy + 8); g.quadraticCurveTo((hx + ex) / 2, (hy + ey) / 2 - 20, ex, ey - 18);
          g.strokeStyle = V.alpha(C.c, 0.4); g.lineWidth = 1.3; g.stroke(); g.setLineDash([]);
        });
        V.text(g, 'tracejado rosa = rota que cruza o cabo', w - 14, ny + nh + 14, { size: 11, align: 'right', color: C.c });
        V.text(g, 'cabo', midX, ny + nh + 14, { size: 11, align: 'center', color: C.muted });
        toks.forEach((tk) => {
          const [hx, hy] = homePos(tk), [ex, ey] = expPos(tk.e);
          const off = (tk.slot - 2.5) * 3;
          let x, y;
          if (ph === 0) { const e2 = V.ease.inOut(pf); x = V.lerp(hx, ex + off, e2); y = V.lerp(hy, ey - 22, e2); }
          else if (ph === 1) { x = ex + off; y = ey - 22; }
          else { const e2 = V.ease.inOut(pf); x = V.lerp(ex + off, hx, e2); y = V.lerp(ey - 22, hy, e2); }
          const cross = (tk.e < 4 ? 0 : 1) !== tk.home;
          face(g, x, y, 14, fwd ? (tk.home ? C.b : C.a) : C.c, { mood: cross && ph !== 1 ? 'o' : undefined });
        });
        const capt = (fwd ? 'forward' : 'backward (gradientes)') + ' · ' + ['all-to-all de despacho', 'experts calculam', 'all-to-all de retorno'][ph];
        V.text(g, capt, 14, h - 14, { size: 12, color: fwd ? C.a : C.c, maxW: w - 28 });
      });
    })();

    /* ================================================ FIG 8 — LoRA de brinquedo */
    (function () {
      const fig = root.querySelector('#fig-lora');
      const stgL = fig.querySelector('.la-fig-stage');
      /* no celular o alvo T ganha uma linha própria, então a figura cresce quando ele aparece */
      const setHL = () => { const want = (fig.clientWidth < 640 ? (S.showT ? 990 : 760) : 500) + 'px'; if (stgL.style.height !== want) stgL.style.height = want; };
      const D = 8;
      const orth = (seed) => {
        const rr = V.rng(seed);
        const cols = Array.from({ length: D }, () => Array.from({ length: D }, () => rr.normal()));
        const out = [];
        cols.forEach((v0) => { let v = v0.slice(); out.forEach((q) => { const d = v.reduce((s, x, i) => s + x * q[i], 0); v = v.map((x, i) => x - d * q[i]); }); const n = Math.hypot(...v); out.push(v.map((x) => x / n)); });
        return out;
      };
      const U = orth(11), Vv = orth(29), SIG = [2.0, 1.2, 0.5];
      const T = Array.from({ length: D }, (_, i) => Array.from({ length: D }, (_, j) => SIG.reduce((s, sg, k) => s + sg * U[k][i] * Vv[k][j], 0)));
      const W0 = V.randMatrix(D, D, 3, 0.8);
      const S = { r: 2, aI: 3, A: null, B: null, mA: null, vA: null, mB: null, vB: null, k: 0, err: 1, hist: [], merge: -1, showT: false };
      setHL();
      const roL = new ResizeObserver(setHL); roL.observe(fig);
      if (LA.onCleanup) LA.onCleanup(() => roL.disconnect());
      const ALPHAS = [1, 2, 4, 8, 16, 32];
      const sc = () => ALPHAS[S.aI] / S.r;
      function init() {
        const rr = V.rng(5 + S.r);
        S.A = Array.from({ length: S.r }, () => Array.from({ length: D }, () => (rr() * 2 - 1) / Math.sqrt(D)));
        S.B = Array.from({ length: D }, () => new Array(S.r).fill(0));
        const z = (M) => M.map((r) => r.map(() => 0));
        S.mA = z(S.A); S.vA = z(S.A); S.mB = z(S.B); S.vB = z(S.B);
        S.k = 0; S.hist = []; S.err = 1; S.merge = -1;
      }
      init();
      const nT = Math.sqrt(T.flat().reduce((s, x) => s + x * x, 0));
      function step() {
        const s = sc(), k = ++S.k, lr = 0.006;
        const BA = mm(S.B, S.A);
        const R = BA.map((row, i) => row.map((x, j) => s * x - T[i][j]));
        S.err = Math.sqrt(R.flat().reduce((q, x) => q + x * x, 0)) / nT;
        S.hist.push(S.err);
        const dB = mm(R, tr(S.A)).map((row) => row.map((x) => x * s));
        const dA = mm(tr(S.B), R).map((row) => row.map((x) => x * s));
        const up = (P, G, m, v) => { for (let i = 0; i < P.length; i++) for (let j = 0; j < P[0].length; j++) { m[i][j] = 0.9 * m[i][j] + 0.1 * G[i][j]; v[i][j] = 0.999 * v[i][j] + 0.001 * G[i][j] * G[i][j]; P[i][j] -= (lr * (m[i][j] / (1 - Math.pow(0.9, k)))) / (Math.sqrt(v[i][j] / (1 - Math.pow(0.999, k))) + 1e-8); } };
        up(S.B, dB, S.mB, S.vB); up(S.A, dA, S.mA, S.vA);
      }
      const ctl = UI.controls(fig);
      nocase(UI.slider(ctl, { label: 'posto r', min: 1, max: 4, step: 1, value: 2, onInput: (v) => { S.r = v; init(); upd(); } }));
      nocase(UI.slider(ctl, { label: 'α', min: 0, max: 5, step: 1, value: 3, fmt: (v) => String(ALPHAS[v]), onInput: (v) => { S.aI = v; init(); upd(); } }));
      UI.toggle(ctl, { label: 'mostrar o alvo T', value: false, onChange: (v) => { S.showT = v; setHL(); } });
      UI.button(ctl, '↺ reiniciar', () => { init(); upd(); });
      UI.button(ctl, '⇣ fundir (merge)', () => { S.merge = performance.now() / 1000; }, { primary: true });
      const rP = UI.readout(ctl, 'treináveis', '—'), rS = UI.readout(ctl, 'α/r', '—'), rE = UI.readout(ctl, 'erro relativo', '—');
      nocase(rS);
      function upd() { rP.set(S.r * (D + D) + ' de ' + D * D); rS.set(fmt(sc(), 2)); }
      upd();
      let lastE = -1;
      V.fig(fig, (g, w, h, t, dt, St, H) => {
        const C = V.C();
        const now = performance.now() / 1000;
        const merging = S.merge > 0 && now - S.merge < 3.2;
        if (S.merge > 0 && !merging) S.merge = -1;
        if (!merging && S.k < 500) step();
        if (Math.abs(S.err - lastE) > 0.002 || S.k % 20 === 0) { rE.set(fmt(S.err * 100, 1) + ' %', S.err < 0.02 ? 'ok' : S.err > 0.3 ? 'warn' : undefined); lastE = S.err; }
        const s = sc();
        const dW = mm(S.B, S.A).map((r) => r.map((x) => x * s));
        const narrow = w < 640;
        const r = S.r;
        let cell, pos = {};
        // larguras dos operadores entre as matrizes: “+ (α/r)·”, “·”, a seta do produto e “≈”
        const gPlus = 64, gDot = 14, gArr = 38, gT = 30;
        if (!narrow) {
          const cols = D + r + D + D + (S.showT ? D : 0);
          cell = Math.max(12, Math.min(28, (w - 32 - gPlus - gDot - gArr - (S.showT ? gT : 0)) / cols - 2));
          let x = 16; const y = 40;
          pos.W0 = [x, y]; x += D * (cell + 2) + gPlus;
          pos.B = [x, y]; x += r * (cell + 2) + gDot;
          pos.A = [x, y + (D - r) * (cell + 2) / 2]; x += D * (cell + 2) + gArr;
          pos.dW = [x, y]; x += D * (cell + 2) + gT;
          pos.T = [x, y];
          pos.row2 = y + D * (cell + 2) + 40;
        } else {
          cell = Math.max(14, Math.min(22, (w - 44) / (D * 2 + 1.5)));
          const y1 = 40;
          pos.B = [16, y1];
          pos.A = [16 + r * (cell + 2) + 14, y1 + (D - r) * (cell + 2) / 2];
          const y2 = y1 + D * (cell + 2) + 40;
          pos.W0 = [16, y2];
          pos.dW = [16 + D * (cell + 2) + 22, y2];
          pos.T = S.showT ? [16, y2 + D * (cell + 2) + 34] : null;
          pos.row2 = (pos.T ? pos.T[1] : y2) + D * (cell + 2) + 34;
        }
        const mg = merging ? Math.min(1, (now - S.merge) / 1.4) : 0;
        const W0show = merging ? W0.map((row, i) => row.map((x, j) => x + dW[i][j] * mg)) : W0;
        // W0
        V.text(g, merging && mg >= 1 ? 'W = W₀ + ΔW (fundido)' : 'W₀ (congelado)', pos.W0[0], pos.W0[1] - 10, { size: 11.5, font: 'mono', color: C.ink });
        const geos = [];
        geos.push(['W₀', W0show, V.matrix(g, { x: pos.W0[0], y: pos.W0[1], data: W0show, cell, cmap: 'div', min: -2.5, max: 2.5, values: false })]);
        lock(g, pos.W0[0] + D * (cell + 2) - 8, pos.W0[1] - 14, 12, C.muted);
        // B e A
        const fade = merging ? Math.max(0, 1 - mg * 1.2) : 1;
        g.globalAlpha = fade;
        V.text(g, 'B (' + D + '×' + r + ')', pos.B[0], pos.B[1] - 10, { size: 11.5, font: 'mono', color: C.b });
        geos.push(['B', S.B, V.matrix(g, { x: pos.B[0], y: pos.B[1], data: S.B, cell, cmap: 'div', min: -0.6, max: 0.6, values: false, stroke: V.alpha(C.b, 0.5) })]);
        V.text(g, 'A (' + r + '×' + D + ')', pos.A[0], pos.A[1] - 10, { size: 11.5, font: 'mono', color: C.b });
        geos.push(['A', S.A, V.matrix(g, { x: pos.A[0], y: pos.A[1], data: S.A, cell, cmap: 'div', min: -0.6, max: 0.6, values: false, stroke: V.alpha(C.b, 0.5) })]);
        if (S.k < 40) V.text(g, S.k < 2 ? 'B = 0: ΔW = 0' : 'B começa a andar', pos.B[0], pos.B[1] + D * (cell + 2) + 14, { size: 11, color: C.d, font: 'mono' });
        g.globalAlpha = 1;
        // símbolos: a linha de cima lê “W₀ + (α/r)·B·A”, e a seta leva o produto até o ΔW que ele forma
        const midY = pos.W0[1] + D * (cell + 2) / 2;
        if (!narrow) {
          V.text(g, '+ (α/r)·', pos.W0[0] + D * (cell + 2) + gPlus / 2, midY + 5, { size: 13, color: C.ink, weight: 600, font: 'mono', align: 'center', maxW: gPlus - 4 });
          V.text(g, '·', pos.B[0] + r * (cell + 2) + gDot / 2, midY + 6, { size: 18, color: C.ink, weight: 700, align: 'center' });
          const aR = pos.A[0] + D * (cell + 2);
          V.arrow(g, aR + 6, midY, pos.dW[0] - 6, midY, { color: C.muted, lw: 1.6, head: 6 });
          if (S.showT) V.text(g, '≈?', pos.dW[0] + D * (cell + 2) + gT / 2, midY + 5, { size: 14, color: C.d, weight: 600, align: 'center' });
        } else {
          V.text(g, '·', pos.A[0] - 7, pos.B[1] + D * (cell + 2) / 2 + 6, { size: 18, color: C.ink, weight: 700, align: 'center' });
          V.text(g, '+', pos.W0[0] + D * (cell + 2) + 11, midY + 6, { size: 18, color: C.ink, weight: 600, align: 'center' });
        }
        // ΔW (desliza para W0 no merge)
        let dx = pos.dW[0], dy = pos.dW[1];
        if (merging) { dx = V.lerp(pos.dW[0], pos.W0[0], mg); dy = V.lerp(pos.dW[1], pos.W0[1], mg); g.globalAlpha = 1 - mg * 0.85; }
        V.text(g, 'ΔW = (α/r)·B·A', dx, dy - 10, { size: 11.5, font: 'mono', color: C.ink });
        const gdW = V.matrix(g, { x: dx, y: dy, data: dW, cell, cmap: 'div', min: -1.2, max: 1.2, values: false });
        if (!merging) geos.push(['ΔW', dW, gdW]);
        g.globalAlpha = 1;
        if (S.showT) {
          const pT = pos.T || [pos.dW[0], pos.row2 - 6];
          if (pos.T) {
            V.text(g, 'alvo T (posto 3)', pT[0], pT[1] - 10, { size: 11.5, font: 'mono', color: C.d });
            g.globalAlpha = 0.85; geos.push(['T', T, V.matrix(g, { x: pT[0], y: pT[1], data: T, cell, cmap: 'div', min: -1.2, max: 1.2, values: false, stroke: V.alpha(C.d, 0.6) })]); g.globalAlpha = 1;
          }
        }
        if (merging) V.text(g, 'merge: W ← W₀ + (α/r)·B·A → custo zero na inferência', 16, pos.row2 - 12, { size: 12, color: C.a, weight: 600, maxW: w - 32 });
        // linha 2: valores singulares + curva de erro
        const y2 = pos.row2 + 6, hh = h - y2 - 26;
        const half = narrow ? w - 32 : (w - 48) / 2;
        const svT = SIG.concat([0, 0, 0, 0, 0]).slice(0, 4), svD = svals(dW).slice(0, 4);
        const bx = 16, by = y2 + (narrow ? 0 : 4);
        const bH = narrow ? Math.min(90, hh * 0.4) : hh - 20;
        V.text(g, 'valores singulares: alvo (âmbar) × aprendido (lilás)', bx, by + 2, { size: 11.5, color: C.muted, maxW: half });
        const bw2 = Math.min(26, (half - 40) / 9);
        for (let k = 0; k < 4; k++) {
          const xk = bx + 8 + k * (2 * bw2 + 12);
          const h1 = (svT[k] / 2.2) * (bH - 30), h2 = (svD[k] / 2.2) * (bH - 30);
          g.fillStyle = V.alpha(C.d, 0.75); V.rr(g, xk, by + bH - h1, bw2, Math.max(1, h1), 3); g.fill();
          g.fillStyle = V.alpha(C.b, 0.85); V.rr(g, xk + bw2 + 2, by + bH - h2, bw2, Math.max(1, h2), 3); g.fill();
          V.text(g, 'σ' + (k + 1), xk + bw2, by + bH + 14, { size: 11, align: 'center', color: C.muted, font: 'mono' });
        }
        // curva de erro
        const cx0 = narrow ? 16 : 16 + half + 16, cy0 = narrow ? by + bH + 34 : by, cw = half, ch = narrow ? h - (by + bH + 34) - 30 : bH;
        V.text(g, 'erro ‖ΔW − T‖ / ‖T‖ ao longo dos passos', cx0, cy0 + 2, { size: 11.5, color: C.muted, maxW: cw });
        const gx0 = cx0 + 34, gx1 = cx0 + cw - 4, gy0 = cy0 + 12, gy1 = cy0 + ch;
        V.line(g, gx0, gy1, gx1, gy1, { color: C.muted, lw: 1 }); V.line(g, gx0, gy0, gx0, gy1, { color: C.muted, lw: 1 });
        V.text(g, '100%', gx0 - 4, gy0 + 8, { size: 10.5, align: 'right', color: C.muted, font: 'mono' });
        V.text(g, '0', gx0 - 4, gy1, { size: 10.5, align: 'right', color: C.muted, font: 'mono' });
        const floor = r >= 3 ? 0 : Math.sqrt(SIG.slice(r).reduce((q, x) => q + x * x, 0)) / Math.sqrt(SIG.reduce((q, x) => q + x * x, 0));
        const yF = gy1 - floor * (gy1 - gy0);
        V.line(g, gx0, yF, gx1, yF, { color: V.alpha(C.d, 0.8), lw: 1.2, dash: [4, 4] });
        V.text(g, 'mínimo possível com r = ' + r + ': ' + fmt(floor * 100, 1) + ' %', gx1, yF - 5, { size: 11, align: 'right', color: C.d });
        g.beginPath();
        S.hist.forEach((e, i) => { const x = gx0 + (i / 500) * (gx1 - gx0), y = gy1 - Math.min(1, e) * (gy1 - gy0); if (!i) g.moveTo(x, y); else g.lineTo(x, y); });
        g.strokeStyle = C.b; g.lineWidth = 2.2; g.stroke();
        V.text(g, S.k + ' passos', gx1, gy1 + 14, { size: 11, align: 'right', color: C.muted, font: 'mono' });
        // dica de valor ao passar o mouse
        const Pp = H.pointer;
        if (Pp.inside) for (const [nm, data, geo] of geos) {
          const hit = geo.cellAt(Pp.x, Pp.y);
          if (!hit) continue;
          const val = data[hit[0]][hit[1]];
          const txt = nm + '[' + hit[0] + ',' + hit[1] + '] = ' + (val >= 0 ? ' ' : '') + val.toFixed(3);
          const tww = tw(g, txt, 12, 'mono') + 16;
          const tx = Math.min(w - tww - 6, Pp.x + 12), ty = Math.max(6, Pp.y - 30);
          V.box(g, tx, ty, tww, 22, { r: 6, fill: C.panel || C.bg, stroke: C.b, lw: 1.2 });
          V.text(g, txt, tx + 8, ty + 15, { size: 12, font: 'mono', color: C.ink });
          break;
        }
      });
    })();

    /* ================================================ FIG 9 — NF4 */
    (function () {
      const fig = root.querySelector('#fig-nf4');
      adaptH(fig, 470, 580);
      const NF4 = [-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224, 0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0];
      const INT4 = Array.from({ length: 15 }, (_, k) => (k - 7) / 7);
      const S = { fmt: 'nf4', dq: true, out: false, seed: 1, blk: null };
      const nearest = (lv, x) => { let b = 0, bd = 1e9; lv.forEach((l, i) => { const d = Math.abs(l - x); if (d < bd) { bd = d; b = i; } }); return b; };
      function quant(ws, lv) {
        const c = Math.max(...ws.map(Math.abs));
        const idx = ws.map((x) => nearest(lv, x / c));
        const dq = idx.map((i) => lv[i] * c);
        const rmse = Math.sqrt(ws.reduce((s, x, i) => s + (x - dq[i]) ** 2, 0) / ws.length);
        return { c, idx, dq, rmse };
      }
      function mkBlock(seed, out) { const r = V.rng(seed); const ws = Array.from({ length: 64 }, () => r.normal() * 0.02); if (out) ws[19] = 0.02 * 6.5; return ws; }
      const avg = {};
      [false, true].forEach((o) => { let a = 0, b = 0, sd = 0; for (let k = 0; k < 400; k++) { const ws = mkBlock(1000 + k, o); a += quant(ws, NF4).rmse; b += quant(ws, INT4).rmse; sd += Math.sqrt(ws.reduce((s, x) => s + x * x, 0) / 64); } avg[o] = { nf4: a / sd, int4: b / sd }; });
      function build() { S.blk = mkBlock(S.seed, S.out); }
      build();
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Níveis', options: [{ v: 'nf4', label: 'NF4 (16)' }, { v: 'int4', label: 'INT4 uniforme (15)' }], value: 'nf4', onChange: (v) => { S.fmt = v; upd(); } });
      UI.toggle(ctl, { label: 'quantização dupla', value: true, onChange: (v) => { S.dq = v; upd(); } });
      UI.toggle(ctl, { label: 'um outlier no bloco', value: false, onChange: (v) => { S.out = v; build(); upd(); } });
      UI.button(ctl, '↻ novo bloco', () => { S.seed++; build(); upd(); });
      const rB = UI.readout(ctl, 'bits/param', '—'), rE = UI.readout(ctl, 'erro RMS (bloco)', '—'), rA = UI.readout(ctl, 'média em 400 blocos', '—');
      function upd() {
        rB.set(S.dq ? '4,127' : '4,5');
        const q = quant(S.blk, S.fmt === 'nf4' ? NF4 : INT4);
        const sd = Math.sqrt(S.blk.reduce((s, x) => s + x * x, 0) / 64);
        rE.set(fmt((q.rmse / sd) * 100, 1) + ' % do σ');
        const a = avg[S.out];
        rA.set('NF4 ' + fmt(a.nf4 * 100, 1) + ' % · INT4 ' + fmt(a.int4 * 100, 1) + ' %');
      }
      upd();
      const dens = (x) => Math.exp(-0.5 * x * x * 6.25);
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const narrow = w < 600;
        const lv = S.fmt === 'nf4' ? NF4 : INT4;
        const q = quant(S.blk, lv);
        const M8 = (arr) => Array.from({ length: 8 }, (_, i) => arr.slice(i * 8, i * 8 + 8));
        const cell = narrow ? Math.max(15, Math.min(20, (w - 50) / 17)) : Math.max(18, Math.min(29, (w - 300) / 17));
        const sdB = Math.sqrt(S.blk.reduce((q2, x) => q2 + x * x, 0) / 64);
        const mx = 14, my = 34;
        V.text(g, 'pesos (×10⁻²)', mx, my - 10, { size: 11.5, font: 'mono', color: C.ink });
        V.matrix(g, { x: mx, y: my, data: M8(S.blk.map((x) => x * 100)), cell, cmap: 'div', min: -6, max: 6, values: cell >= 24, digits: 1, hiCells: S.out ? [[2, 3]] : [] });
        const cx = mx + 8 * (cell + 2) + (narrow ? 12 : 22);
        V.text(g, 'códigos de 4 bits', cx, my - 10, { size: 11.5, font: 'mono', color: C.ink });
        const codes = M8(q.idx);
        for (let i = 0; i < 8; i++) for (let j = 0; j < 8; j++) {
          const id = codes[i][j], x = cx + j * (cell + 2), y = my + i * (cell + 2);
          const v = lv[id];
          g.fillStyle = V.cmap(v, 'div'); V.rr(g, x, y, cell, cell, 3); g.fill();
          if (cell >= 15) V.text(g, id.toString(16).toUpperCase(), x + cell / 2, y + cell / 2 + 4, { size: Math.min(12, cell * 0.55), font: 'mono', align: 'center', color: Math.abs(v) > 0.55 ? (C.dark ? '#00210F' : '#fff') : C.ink });
        }
        // painel de bits
        const px = cx + 8 * (cell + 2) + (narrow ? 0 : 24), py = narrow ? my + 8 * (cell + 2) + 34 : my;
        const pw = narrow ? w - 28 : w - px - 14;
        if (pw > 120) {
          const bx = narrow ? 14 : px;
          V.text(g, 'escala c = absmax = ' + sci(q.c, 2), bx, py - 10, { size: 11.5, font: 'mono', color: C.ink, maxW: pw });
          V.text(g, S.dq ? '(a escala é guardada em 8 bits)' : '(a escala é guardada em FP32)', bx, py + 5, { size: 11, color: C.muted, maxW: pw });
          const bits = S.dq ? [[4, C.a, 'código'], [0.125, C.d, 'escala 8b'], [0.002, C.c, 'escala da escala']] : [[4, C.a, 'código'], [0.5, C.d, 'escala FP32']];
          const tot = 4.5, bwid = pw - 4;
          let ax = bx;
          bits.forEach(([b, col]) => { const ww = Math.max(1.5, (b / tot) * bwid); g.fillStyle = V.alpha(col, 0.8); g.fillRect(ax, py + 14, ww, 14); ax += ww; });
          g.strokeStyle = V.alpha(C.muted, 0.5); g.strokeRect(bx + 0.5, py + 14.5, bwid, 14);
          if (narrow) V.text(g, S.dq ? 'código 4 · escala 0,125 · escala da escala 0,002' : 'código 4 · escala 0,5', bx, py + 62, { size: 11, color: C.muted, maxW: pw });
          else if (pw > 300) V.text(g, S.dq ? 'código 4 · escala 0,125 · escala da escala 0,002' : 'código 4 · escala 0,5', bx, py + 62, { size: 11, color: C.muted, maxW: pw });
          V.text(g, S.dq ? '4 + 8/64 + 32/(64·256) = 4,127 bits' : '4 + 32/64 = 4,5 bits', bx, py + 44, { size: 11.5, font: 'mono', color: C.ink, maxW: pw });
          if (!narrow) {
            V.text(g, 'erro deste bloco: ' + fmt((q.rmse / sdB) * 100, 1) + ' % do σ', bx, py + 66, { size: 11.5, color: C.muted });
            V.wrap(g, S.fmt === 'nf4' ? 'NF4: níveis nos quantis da normal, mais densos perto de zero, onde mora a maioria dos pesos.' : 'INT4 uniforme: níveis igualmente espaçados; desperdiça resolução nas pontas, onde há poucos pesos.', bx, py + 88, pw, { size: 11.5, color: C.ink, lineH: 15, maxLines: 4 });
          }
        }
        // reta numérica
        const ly0 = narrow ? py + 76 : my + 8 * (cell + 2) + 30;
        const lx0 = 24, lx1 = w - 24, base = h - 44, top2 = ly0 + 10;
        const X = (x) => lx0 + ((x + 1) / 2) * (lx1 - lx0);
        g.beginPath();
        for (let i = 0; i <= 120; i++) { const x = -1 + (2 * i) / 120; const y = base - dens(x) * (base - top2) * 0.8; if (!i) g.moveTo(X(x), y); else g.lineTo(X(x), y); }
        g.strokeStyle = V.alpha(C.b, 0.7); g.lineWidth = 1.6; g.stroke();
        V.line(g, lx0, base, lx1, base, { color: C.muted, lw: 1 });
        lv.forEach((l, i) => { V.line(g, X(l), base - 6, X(l), base + 6, { color: C.d, lw: 2 }); if (!narrow || i % 2 === 0) V.text(g, i.toString(16).toUpperCase(), X(l), base + 18, { size: 10.5, font: 'mono', align: 'center', color: C.d }); });
        V.text(g, '−1', lx0, base + 32, { size: 11, color: C.muted, font: 'mono', align: 'center' });
        V.text(g, '+1', lx1, base + 32, { size: 11, color: C.muted, font: 'mono', align: 'center' });
        V.text(g, 'w / c', (lx0 + lx1) / 2, base + 32, { size: 11, color: C.muted, font: 'mono', align: 'center' });
        const cyc = 4.5, u = (t % cyc) / cyc;
        S.blk.forEach((x, i) => {
          const xn = x / q.c, xl = lv[q.idx[i]];
          const jit = ((i * 37) % 11) / 11;
          const yStart = top2 + 6 + jit * 26;
          let px2, py2;
          if (u < 0.35) { px2 = X(xn); py2 = V.lerp(yStart, base - 14, V.ease.out(u / 0.35)); }
          else if (u < 0.7) { const e2 = V.ease.inOut((u - 0.35) / 0.35); px2 = V.lerp(X(xn), X(xl), e2); py2 = V.lerp(base - 14, base - 4 - jit * 6, e2); }
          else { px2 = X(xl); py2 = base - 4 - jit * 6; }
          V.dot(g, px2, py2, 3, V.alpha(i === 19 && S.out ? C.c : C.a, 0.8));
        });
      });
    })();

    /* ================================================ FIG 10 — calculadora + mapa */
    (function () {
      const fig = root.querySelector('#fig-calc');
      const MODELS = {
        q06: { n: 'Qwen3-0.6B', P: 0.6e9, h: 1024, L: 28, V: 151936, tied: true, cfg: [1024, 28, 16, 8, 128, 3072] },
        q17: { n: 'Qwen3-1.7B', P: 1.7e9, h: 2048, L: 28, V: 151936, tied: true, cfg: [2048, 28, 16, 8, 128, 6144] },
        l3: { n: 'Llama-3.2-3B', P: 3.21e9, h: 3072, L: 28, V: 128256, tied: true, cfg: [3072, 28, 24, 8, 128, 8192] },
        q4: { n: 'Qwen3-4B', P: 4.02e9, h: 2560, L: 36, V: 151936, tied: true, cfg: [2560, 36, 32, 8, 128, 9728] },
        l8: { n: 'Llama-3.1-8B', P: 8.03e9, h: 4096, L: 32, V: 128256, tied: false, cfg: [4096, 32, 32, 8, 128, 14336] },
        q8: { n: 'Qwen3-8B', P: 8.19e9, h: 4096, L: 36, V: 151936, tied: false, cfg: [4096, 36, 32, 8, 128, 12288] },
        q14: { n: 'Qwen3-14B', P: 14.8e9, h: 5120, L: 40, V: 151936, tied: false, cfg: [5120, 40, 40, 8, 128, 17408] },
        q30: { n: 'Qwen3-30B-A3B (MoE)', P: 30.5e9, h: 2048, L: 48, V: 151936, tied: false, moe: { E: 128, inter: 768, layers: 48, attn16: 13.4e6, k: 8 } },
        l70: { n: 'Llama-3.3-70B', P: 70.55e9, h: 8192, L: 80, V: 128256, tied: false, cfg: [8192, 80, 64, 8, 128, 28672] },
        glm: { n: 'GLM-5.3-Flash (MoE)', P: 320e9, h: 4096, L: 45, V: 154880, tied: false, moe: { E: 288, inter: 2048, layers: 42, attn16: 0, k: 8 }, noAct: true },
      };
      const RANKS = [1, 2, 4, 8, 16, 32, 64, 128, 256];
      const S = { mode: 'bar', model: 'l8', method: 'full', reg: 16, rI: 4, mods: 'all', hw: 'spark1', sE: 11, b: 1, ck: true, lg: 4, sel: null };
      const stg = fig.querySelector('.la-fig-stage');
      /* na calculadora, a altura acompanha o texto das notas (S.needH, medido no draw), com piso de 400 px */
      const setH = () => { const nar = fig.clientWidth < 640; const want = (S.mode === 'map' ? (nar ? 470 : 440) : Math.max(400, S.needH || 0)) + 'px'; if (stg.style.height !== want) stg.style.height = want; };
      setH();
      const roH = new ResizeObserver(setH); roH.observe(fig);
      if (LA.onCleanup) LA.onCleanup(() => roH.disconnect());
      function loraParams(M, r, mods, Lt) {
        if (M.moe) {
          const re = Math.max(1, r / M.moe.k);
          const ex = M.moe.layers * M.moe.E * 3 * re * (M.h + M.moe.inter) * (Lt / M.L);
          const at = M.moe.attn16 * (r / 16) * (Lt / M.L);
          return mods === 'attn' ? at : mods === 'mlp' ? ex : at + ex;
        }
        const [h, L, nh, nkv, hd, inter] = M.cfg;
        const q = r * (h + nh * hd), k = r * (h + nkv * hd), o = r * (nh * hd + h), g2 = r * (h + inter);
        const at = q + 2 * k + o, ml = 3 * g2;
        return Lt * (mods === 'attn' ? at : mods === 'mlp' ? ml : at + ml);
      }
      function calc(over) {
        const method = (over && over.method) || S.method;
        const M = MODELS[S.model], r = RANKS[S.rI];
        const nodes = S.hw === 'spark2' ? 2 : 1;
        const s = Math.pow(2, S.sE), b = S.b;
        const emb = M.V * M.h * (M.tied ? 1 : 2);
        const mlx = method === 'mlx';
        const Lt = mlx ? Math.min(16, M.L) : M.L;
        let wts = 0, grd = 0, mst = 0, opt = 0, Pl = 0;
        if (method === 'full') {
          wts = 2 * M.P; grd = 2 * M.P;
          mst = S.reg >= 12 ? 4 * M.P : 0;
          opt = (S.reg === 16 ? 8 : S.reg === 12 ? 4 : S.reg === 8 ? 4 : 2) * M.P;
        } else {
          Pl = loraParams(M, r, S.mods, Lt);
          const base = method === 'lora' ? 2 * M.P : method === 'qlora' ? (M.P - emb) * (4.127 / 8) + emb * 2 : M.P * 0.5625;
          wts = base + 4 * Pl; grd = 4 * Pl; opt = 8 * Pl;
        }
        const act = M.noAct ? 0 : S.ck ? 2 * s * b * M.h * Lt + 34 * s * b * M.h : 34 * s * b * M.h * Lt;
        const lgt = s * b * M.V * S.lg;
        const parts = [['pesos', wts / nodes, 'a'], ['gradientes', grd / nodes, 'b'], ['mestres', mst / nodes, 'd'], ['otimizador', opt / nodes, 'c'], ['ativações', act, 'e'], ['logits', lgt, 'f']];
        const tot = parts.reduce((q, p) => q + p[1], 0);
        return { M, r, Pl, parts, tot, nodes, method };
      }
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Vista', options: [{ v: 'bar', label: 'calculadora' }, { v: 'map', label: 'mapa de viabilidade' }], value: 'bar', onChange: (v) => { S.mode = v; gC.show(v === 'bar'); gR.show(v === 'bar'); setH(); } });
      const gC = UI.group(ctl);
      const cMod = UI.select(gC.el, { label: 'Modelo', options: Object.keys(MODELS).map((k) => ({ v: k, label: MODELS[k].n })), value: 'l8', onChange: (v) => { S.model = v; upd(); } });
      const cMet = UI.seg(gC.el, { label: 'Método', options: [{ v: 'full', label: 'full FT' }, { v: 'lora', label: 'LoRA (base BF16)' }, { v: 'qlora', label: 'QLoRA (NF4)' }, { v: 'mlx', label: 'LoRA mlx 4 bits' }], value: 'full', onChange: (v) => { S.method = v; gF.show(v === 'full'); gL.show(v !== 'full'); upd(); } });
      const gF = UI.group(gC.el), gL = UI.group(gC.el);
      const cReg = UI.seg(gF.el, { label: 'B/param', options: [16, 12, 8, 6].map((x) => ({ v: x, label: String(x) })), value: 16, onChange: (v) => { S.reg = v; upd(); } });
      const cR = nocase(UI.slider(gL.el, { label: 'posto r', min: 0, max: RANKS.length - 1, step: 1, value: 4, fmt: (v) => String(RANKS[v]), onInput: (v) => { S.rI = v; upd(); } }));
      const cMods = UI.seg(gL.el, { label: 'Módulos', options: [{ v: 'all', label: 'todas' }, { v: 'attn', label: 'atenção' }, { v: 'mlp', label: 'MLP/experts' }], value: 'all', onChange: (v) => { S.mods = v; upd(); } });
      gL.show(false);
      const cHw = UI.seg(gC.el, { label: 'Hardware', options: [{ v: 'spark1', label: '1 Spark' }, { v: 'spark2', label: '2 Sparks (FSDP)' }, { v: 'mac', label: 'Mac M4 24 GB' }], value: 'spark1', onChange: (v) => { S.hw = v; upd(); } });
      const cS = nocase(UI.slider(gC.el, { label: 'seq s', min: 9, max: 15, step: 1, value: 11, fmt: (v) => String(Math.pow(2, v)), onInput: (v) => { S.sE = v; upd(); } }));
      const cB = nocase(UI.slider(gC.el, { label: 'lote b', min: 1, max: 16, step: 1, value: 1, onInput: (v) => { S.b = v; upd(); } }));
      const cCk = UI.toggle(gC.el, { label: 'checkpointing', value: true, onChange: (v) => { S.ck = v; upd(); } });
      UI.seg(gC.el, { label: 'Logits', options: [{ v: 4, label: 'FP32' }, { v: 2, label: 'BF16' }], value: 4, onChange: (v) => { S.lg = v; upd(); } });
      const gR = UI.group(ctl);
      const rT = UI.readout(gR.el, 'total por nó', '—'), rV = UI.readout(gR.el, 'veredito', '—'), rL = UI.readout(gR.el, 'treináveis', '—');
      /** [texto longo, estado, texto curto, exp?]. Se nem a conta cabe, “não cabe” é firme; se cabe, MoE + QLoRA
          (bnb 4 bits não cobre experts) e o GLM sem ativações estimadas viram “exp”, nunca “cabe”. */
      function verdict(c) {
        const tot = c.tot;
        const base = S.hw === 'mac' ? (tot <= 14.3e9 ? ['cabe', 'ok'] : tot <= MAC ? ['no limite', 'warn'] : ['não cabe', 'bad']) : fitSpark(tot);
        if (base[1] === 'bad') return [base[0], 'bad', base[0], false];
        if (c.M.moe && c.method === 'qlora') return ['exp: cabe em memória, suporte não verificado', 'warn', 'exp · não verificado', true];
        if (c.M.noAct) return ['exp: só a cota inferior cabe (ativações não estimadas)', 'warn', 'exp · cota inferior', true];
        return [base[0], base[1], base[0], false];
      }
      const vCol = (v, C) => (v[3] ? C.b : v[1] === 'ok' ? C.a : v[1] === 'warn' ? C.d : RED());
      const totTxt = (c) => (c.M.noAct ? '≥ ' : '') + fmtGB(c.tot);
      function upd() {
        const c = calc();
        rT.set(totTxt(c));
        const v = verdict(c); rV.set(v[2], v[1]);
        rL.set(S.method === 'full' ? fmt(c.M.P / 1e9, 2) + ' B (100 %)' : fmt(c.Pl / 1e6, 1) + ' M (' + fmt((c.Pl / c.M.P) * 100, 2) + ' %)');
      }
      upd();
      // mapa (verditos da nota de pesquisa §5.2/§5.3)
      const COLS = ['Full · 1', 'Full · 2', 'LoRA · 1', 'LoRA · 2', 'QLoRA · 1', 'Mac M4'];
      const ROWS = [
        { n: '≤ 3B', m: 'l3', v: ['ok', 'ok', 'ok', 'ok', 'ok', 'ok'], d: ['Llama-3.2-3B: 51 GB a 16 B/p; o playbook da NVIDIA faz full FT de 3B num Spark', 'cabe com folga', 'base BF16 6,4 GB + adaptador', 'cabe', 'cabe', 'no Mac, QLoRA cabe (extrapolado do 4B: 8,7 GB com ckpt); full FT só até ~1,7B'] },
        { n: '4B', m: 'q4', v: ['warn', 'ok', 'ok', 'ok', 'ok', 'ok'], d: ['64 GB de estado + ativações: cabe com ajustes (lote 1, ckpt, chunked_nll, seq ≤ 4K)', '32 GB por nó', 'cabe', 'cabe', 'cabe', 'Qwen3-4B 4 bits, lote 4: 18,8 GB sem ckpt (no limite); 8,7 GB com ckpt'] },
        { n: '8B', m: 'l8', v: ['bad', 'ok', 'ok', 'ok', 'ok', 'ok'], d: ['128,5 GB de estado: não cabe (8 B/p em BF16 puro daria 64 GB, mas veja a seção 2)', '64,2 GB por nó com FSDP (cálculo; o playbook multi-nó traz full FT de 3B e LoRA de 8B/70B)', '16 GB de base + adaptador + ativações', 'cabe', 'cabe', 'Qwen3-8B 4 bits, lote 1, ckpt: 6,6 GB'] },
        { n: '14B', m: 'q14', v: ['bad', 'warn', 'ok', 'ok', 'ok', 'ok'], d: ['237 GB de estado', 'a 16 B/p, 118 GB ≈ 110 GiB por nó, sem folga para ativações; com Muon (12 B/p) 89 GB/nó ou 8 B/p', '30 GB de base', 'cabe', 'cabe', 'Qwen3-14B 4 bits, lote 1, ckpt: 10,5 GB (12,4 GB com s = 4.096)'] },
        { n: 'Qwen3-30B-A3B', m: 'q30', v: ['bad', 'bad', 'warn', 'ok', 'exp', 'bad'], d: ['488 GB de estado', '244 GB por nó', '61 GB de base + LoRA + ativações (o Unsloth mediu 80,9 GB num B200 a 4K de contexto)', 'cabe', 'bnb 4 bits não cobre experts; NVFP4 + ScatterMoE (Axolotl) em tese', '4 bits: > 17,2 GB só de pesos'] },
        { n: '70B denso', m: 'l70', v: ['bad', 'bad', 'bad', 'ok', 'ok', 'bad'], d: ['1.129 GB de estado', '564 GB por nó', '141 GB de base', '~70,6 GB/nó + adaptadores + ativações (playbook 70B LoRA com FSDP)', '~40 GB + 3,3 GB + ativações (playbooks PyTorch e NeMo)', 'não cabe'] },
        { n: 'gpt-oss-120b', m: null, v: ['bad', 'bad', 'bad', 'bad', 'ok', 'bad'], d: ['não cabe', 'não cabe', 'não cabe', '234 GB de base BF16', '“around 68GB” com QLoRA do Unsloth', 'não cabe'] },
        { n: 'Qwen3.8-Flash-Next', m: null, v: ['bad', 'bad', 'bad', 'bad', 'exp', 'bad'], d: ['125,7 B + PLE 51,2 B', 'não cabe', 'não cabe', '≥ 250 GB de base', 'só em 2 nós: principal 4 bits ~65–71 GB + PLE FP8 ~51 GB; exige suporte a GR/PLE/QSA/GDN no stack de treino', 'não cabe'] },
        { n: 'DeepSeek-V4-Flash', m: null, v: ['bad', 'bad', 'bad', 'bad', 'exp', 'bad'], d: ['284 B', 'não cabe', 'não cabe', '569 GB de base', 'só em 2 nós (~147–160 GB em 4 bits); CSA/HCA/mHC sem suporte conhecido em frameworks de fine-tuning', 'não cabe'] },
        { n: 'GLM-5.3-Flash', m: 'glm', v: ['bad', 'bad', 'bad', 'bad', 'exp', 'bad'], d: ['320 B', 'não cabe', 'não cabe', '640 GB de base', 'só em 2 nós (~165–180 GB em 4 bits) + LoRA por expert (r = 1: 3,6 GB; r = 8: 28,5 GB)', 'não cabe'] },
      ];
      const COLCFG = [{ method: 'full', hw: 'spark1' }, { method: 'full', hw: 'spark2' }, { method: 'lora', hw: 'spark1' }, { method: 'lora', hw: 'spark2' }, { method: 'qlora', hw: 'spark1' }, { method: 'mlx', hw: 'mac' }];
      function applyCell(i, j) {
        const row = ROWS[i]; if (!row.m) return false;
        const cc = COLCFG[j];
        S.model = row.m; cMod.set(row.m);
        S.method = cc.method; cMet.set(cc.method);
        S.hw = cc.hw; cHw.set(cc.hw, true);
        S.rI = cc.method === 'mlx' ? 3 : 4; cR.set(S.rI, true);
        S.mods = 'all'; cMods.set('all', true);
        S.reg = 16; cReg.set(16, true);
        S.sE = 11; cS.set(11, true); S.b = 1; cB.set(1, true);
        if (!S.ck) cCk.set(true);
        upd();
        return true;
      }
      let geoMap = null;
      V.fig(fig, (g, w, h, t, dt, St, H) => {
        const C = V.C();
        const narrow = w < 600;
        if (S.mode === 'map') {
          const lw = narrow ? 104 : 150, top = narrow ? 44 : 36;
          const cw = (w - lw - 16) / COLS.length, rh = Math.min(30, (h - top - 110) / ROWS.length);
          COLS.forEach((c, j) => {
            if (!narrow) { V.text(g, c, lw + j * cw + cw / 2, top - 10, { size: 11.5, align: 'center', color: C.ink, font: 'mono', maxW: cw - 2 }); return; }
            const [a1, a2] = c.split(/ · | /);
            V.text(g, a1, lw + j * cw + cw / 2, top - 20, { size: 10.5, align: 'center', color: C.ink, font: 'mono', maxW: cw - 2 });
            V.text(g, a2 === '1' || a2 === '2' ? a2 + (a2 === '1' ? ' nó' : ' nós') : a2, lw + j * cw + cw / 2, top - 7, { size: 10.5, align: 'center', color: C.muted, font: 'mono', maxW: cw - 2 });
          });
          const colr = { ok: C.a, warn: C.d, bad: RED(), exp: C.b };
          const sym = { ok: '✓', warn: '≈', bad: '✗', exp: 'exp' };
          const P = H.pointer;
          let hov = null;
          ROWS.forEach((row, i) => {
            const y = top + i * rh;
            V.text(g, row.n, lw - 8, y + rh / 2 + 4, { size: narrow ? 10.5 : 11.5, align: 'right', color: C.ink, maxW: lw - 10 });
            row.v.forEach((v, j) => {
              const x = lw + j * cw;
              const isSel = S.sel && S.sel[0] === i && S.sel[1] === j;
              const inside = P.inside && P.x >= x && P.x < x + cw && P.y >= y && P.y < y + rh;
              if (inside) hov = [i, j];
              V.box(g, x + 2, y + 2, cw - 4, rh - 4, { r: 5, fill: V.alpha(colr[v], isSel ? 0.75 : inside ? 0.55 : 0.32), stroke: isSel ? C.ink : V.alpha(colr[v], 0.8), lw: isSel ? 2 : 1 });
              V.text(g, sym[v], x + cw / 2, y + rh / 2 + 4, { size: 12, align: 'center', color: C.ink, font: 'mono', weight: 600 });
            });
          });
          if (P.clicked && hov) { S.sel = hov; S.applied = applyCell(hov[0], hov[1]); }
          const show = hov || S.sel;
          const iy = top + ROWS.length * rh + 14;
          if (show) {
            const row = ROWS[show[0]];
            V.text(g, row.n + ' · ' + COLS[show[1]], 14, iy + 4, { size: 12, weight: 600, color: C.ink });
            V.wrap(g, row.d[show[1]], 14, iy + 22, w - 28, { size: 12, color: C.ink, lineH: 16, maxLines: 3 });
            if (S.sel && show[0] === S.sel[0] && show[1] === S.sel[1]) V.text(g, S.applied ? '→ configuração carregada na calculadora' : '(modelo fora da calculadora: só a conta da nota de pesquisa)', 14, h - 10, { size: 11.5, color: S.applied ? C.a : C.muted });
          } else V.wrap(g, '✓ cabe · ≈ cabe com ajustes · ✗ não cabe · exp = cabe em memória, mas depende de suporte de framework/kernel não verificado. Passe o mouse ou toque numa célula.', 14, iy + 4, w - 28, { size: 12, color: C.muted, lineH: 16, maxLines: 4 });
          return;
        }
        const c = calc();
        const hwMax = S.hw === 'mac' ? MAC : SPARK;
        const scale = Math.max(c.tot, hwMax) * 1.1;
        const bx = 16, bw = w - 32, by = narrow ? 70 : 64, bh = 34;
        const X = (b) => bx + (b / scale) * bw;
        V.text(g, c.M.n + ' · ' + { full: 'full FT ' + S.reg + ' B/p', lora: 'LoRA r' + c.r + ' (base BF16)', qlora: 'QLoRA r' + c.r + ' (NF4 + DQ)', mlx: 'LoRA r' + c.r + ' sobre MLX 4 bits (16 camadas)' }[S.method] + ' · ' + { spark1: '1 Spark', spark2: '2 Sparks, FSDP', mac: 'Mac M4 24 GB' }[S.hw], bx, 22, { size: 12.5, weight: 600, color: C.ink, maxW: bw - 40 });
        V.text(g, 's = ' + Math.pow(2, S.sE) + ' · b = ' + S.b + ' · ' + (S.ck ? 'com' : 'sem') + ' ckpt · logits ' + (S.lg === 4 ? 'FP32' : 'BF16'), bx, 40, { size: 11.5, color: C.muted, font: 'mono', maxW: bw - 40 });
        V.box(g, bx, by, bw, bh, { r: 6, fill: V.alpha(C.muted, 0.07), stroke: V.alpha(C.muted, 0.3), lw: 1 });
        let acc = 0;
        c.parts.forEach(([lab, b, k]) => { if (b <= 0) return; const x1 = X(acc), x2 = X(acc + b); g.fillStyle = V.alpha(C[k], 0.82); g.fillRect(x1, by + 1, Math.max(1, x2 - x1), bh - 2); acc += b; });
        // linhas de orçamento
        if (S.hw === 'mac') {
          V.line(g, X(MAC), by - 8, X(MAC), by + bh + 8, { color: RED(), lw: 2, dash: [4, 3] });
          V.text(g, '19,07 GB', X(MAC), by + bh + 20, { size: 11, color: RED(), align: 'center', font: 'mono' });
        } else {
          g.fillStyle = V.alpha(C.d, 0.14); g.fillRect(X(PRAT_LO), by - 6, X(PRAT_HI) - X(PRAT_LO), bh + 12);
          V.line(g, X(SPARK), by - 8, X(SPARK), by + bh + 8, { color: RED(), lw: 2, dash: [4, 3] });
          V.text(g, '121,69 GiB', X(SPARK), by + bh + 20, { size: 11, color: RED(), align: X(SPARK) > w - 60 ? 'right' : 'center', font: 'mono' });
          if (!narrow) V.text(g, 'prático', (X(PRAT_LO) + X(PRAT_HI)) / 2, by - 10, { size: 11, color: C.d, align: 'center' });
        }
        const v = verdict(c);
        const vc = vCol(v, C);
        face(g, w - 30, 30, 26, vc, { mood: v[1] === 'bad' ? 'sad' : v[1] === 'warn' ? 'o' : undefined, sweat: v[1] === 'bad' ? t : 0 });
        // tabela de componentes
        const ty = by + bh + 40;
        const colsN = narrow ? 2 : 3, cwid = (w - 32) / colsN;
        c.parts.forEach(([lab, b, k], i) => {
          const x = bx + (i % colsN) * cwid, y = ty + Math.floor(i / colsN) * 24;
          g.fillStyle = V.alpha(C[k], 0.85); V.rr(g, x, y - 10, 14, 12, 3); g.fill();
          V.text(g, lab + ': ' + fmtGB(b, b < 10e9 ? 2 : 1), x + 20, y, { size: 12, color: C.ink, font: 'mono', maxW: cwid - 24 });
        });
        const ty2 = ty + Math.ceil(c.parts.length / colsN) * 24 + 12;
        const th = V.wrap(g, 'total por nó: ' + totTxt(c) + ' → ' + v[0], bx, ty2, bw, { size: 13, weight: 600, color: vc, lineH: 17, maxLines: 2 });
        let note = '';
        if (S.method !== 'full') note = 'treináveis: ' + fmt(c.Pl / 1e6, 1) + ' M de parâmetros (' + fmt((c.Pl / c.M.P) * 100, 2) + ' %)' + (c.M.moe ? ' · MoE: posto por expert = r/' + c.M.moe.k : '') + '. ';
        if (c.M.noAct) note += 'Ativações não estimadas para o GLM (4 fluxos mHC): o total é uma cota inferior. ';
        if (c.M.moe && S.method === 'qlora') note += 'bitsandbytes 4 bits não quantiza experts guardados como nn.Parameter; o caminho seria NVFP4 + ScatterMoE (seção 7), não verificado no GB10. ';
        if (S.method === 'mlx') note += 'mlx-lm treina só as 16 últimas camadas por padrão. ';
        if (S.hw === 'mac' && S.method !== 'mlx' && S.method !== 'full') note += 'No Mac, o caminho real é o mlx-lm (LoRA sobre 4 bits). ';
        if (S.hw === 'spark2') note += 'FSDP divide o estado entre 2 nós; cada nó guarda as ativações do seu micro-lote. ';
        const nh = note ? V.wrap(g, note, bx, ty2 + th + 3, bw, { size: 11.5, color: C.muted, lineH: 15, maxLines: narrow ? 8 : 5 }) : 0;
        // mesmo modelo, outros métodos
        let cy = ty2 + th + 3 + nh + 16;
        const need = Math.ceil(cy + 10 + 4 * 22 + 16);
        if (need !== S.needH) { S.needH = need; setH(); }
        if (cy + 100 < h) {
          V.text(g, 'mesmo modelo e mesmo texto, outros métodos (por nó, ' + { spark1: '1 Spark', spark2: '2 Sparks', mac: 'Mac' }[S.hw] + ')', bx, cy, { size: 11.5, color: C.ink, weight: 600, maxW: bw });
          const MS = narrow ? [['full', 'full ' + S.reg + 'B'], ['lora', 'LoRA'], ['qlora', 'QLoRA'], ['mlx', 'mlx 4b']] : [['full', 'full FT ' + S.reg + ' B/p'], ['lora', 'LoRA BF16'], ['qlora', 'QLoRA NF4'], ['mlx', 'mlx 4 bits']];
          const cs = MS.map(([m]) => calc({ method: m }));
          const vals = cs.map((cc) => cc.tot);
          const mx2 = Math.max(...vals, hwMax) * 1.05;
          const lw2 = narrow ? 70 : 120, x2 = bx + lw2, w2 = bw - lw2 - (narrow ? 64 : 80);
          const rowH2 = Math.min(22, (h - cy - 14) / 4);
          MS.forEach(([m, lab], i) => {
            const y = cy + 10 + i * rowH2;
            const vv = verdict(cs[i]);
            const col = vCol(vv, C);
            V.text(g, lab, x2 - 8, y + rowH2 / 2 + 4, { size: 11.5, align: 'right', color: m === S.method ? C.ink : C.muted, weight: m === S.method ? 600 : 400, font: 'mono' });
            const ww = Math.max(2, (vals[i] / mx2) * w2);
            V.box(g, x2, y + 3, ww, rowH2 - 6, { r: 3, fill: V.alpha(col, m === S.method ? 0.85 : 0.45), stroke: m === S.method ? C.ink : null, lw: 1 });
            V.text(g, totTxt(cs[i]) + (vv[3] ? ' · exp' : ''), x2 + ww + 6, y + rowH2 / 2 + 4, { size: 11, color: C.ink, font: 'mono' });
          });
          const xl = x2 + (hwMax / mx2) * w2;
          V.line(g, xl, cy + 6, xl, cy + 10 + 4 * rowH2, { color: RED(), lw: 1.5, dash: [4, 3] });
        }
      }, { static: false });
    })();
  },
});

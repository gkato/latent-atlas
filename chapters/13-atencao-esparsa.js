LA.chapter({
  id: 'atencao-esparsa',
  part: 3,
  order: 13,
  title: 'Atenção esparsa e contexto de 1 milhão de tokens',
  navTitle: 'Atenção esparsa',
  subtitle: 'Janelas que esquecem de propósito, indexadores que escolhem para onde olhar e KV comprimido: como GLM, Qwen e DeepSeek atendem a um milhão de tokens sem pagar o quadrado.',
  minutes: 60,
  recipes: ['glm', 'tf', 'qwen', 'ds'],
  tags: ['atenção esparsa', 'SWA', 'lightning indexer', 'DSA', 'QSA', 'kpool', 'CSA/HCA', 'top-k', 'contexto longo', 'TensorFold', 'radix select'],
  goals: [
    'Explicar por que o custo \\(O(n^2)\\) da atenção densa inviabiliza 1M de tokens e as três rotas de fuga: <b>ler menos</b>, <b>guardar menos</b> e <b>estado fixo</b>',
    'Descrever o <b>lightning indexer</b> (DSA, QSA, kpool): pontuar com ReLU, escolher o top-k e atender só os escolhidos, e por que a <b>destilação por KL</b> preserva a qualidade',
    'Entender a janela deslizante em anel e a compressão de KV <b>C4A/C128A</b> do DeepSeek-V4, e fazer a conta de bytes por token que mostra que 1M cabe',
    'Ler e justificar os parâmetros e patches das receitas: <code>index_topk</code>, <code>index_kpool</code>, <code>select_k</code>, <code>compress_ratios</code>, SP indexer, workspaces e atalhos de contexto curto, e comparar com o mesmo indexador no motor TensorFold (lista de 2.051, radix select, pools visíveis)',
  ],
  prereqs: ['mla'],
  summary: [
    'A atenção densa custa \\(n(n+1)/2\\) pares por cabeça e por camada no prefill e lê todo o KV a cada token de decode. A 1M, um DeepSeek-V4 denso leria ~26 GB de KV por token gerado.',
    'A <b>janela deslizante</b> lê só os últimos \\(W\\) tokens, guardados em anel (slot = pos mod \\(W\\)). O KV por sequência fica constante (DeepSeek: 128 tokens, ≈ 3,4 MB/seq), mas a janela esquece, por isso sempre vem combinada com um ramo global.',
    'O <b>lightning indexer</b> pontua todas as chaves com poucas cabeças de 128 dimensões, uma chave compartilhada, FP8/FP4 e ReLU, e o top-k (~2.048 tokens) vai para a atenção principal. Ele continua \\(O(n)\\) por token: domina o prefill longo e os bytes de decode a 1M.',
    'O indexador aprende por <b>destilação KL</b> contra a atenção densa: primeiro um warm-up com o modelo congelado, depois treino esparso conjunto. Como a atenção é concentrada, o top-k captura quase toda a massa (erro ≤ \\(2\\varepsilon\\max\\lVert v\\rVert\\)), e a QSA até supera a atenção cheia de 512K a 1M no RULER.',
    'GLM (pools aprendidos, 32 × 128) e Qwen (média, 4 × 128) pontuam blocos de 4, e a cauda de ≤ 3 tokens é sempre incluída. No GLM, 512 pools + cauda = 2.051 estouram o template de 2.048 do kernel SM120: a receita expande 511 pools + cauda (2.047) e perde 0,2% dos candidatos, os de menor ranking. Na versão TensorFold (GLM·TF), com kernels próprios, a lista tem 2.051 posições, a atenção é densa até 2.051 tokens e a seleção é um radix select determinístico.',
    'O DeepSeek-V4 comprime o próprio KV: C4A (21 camadas, top-512 entre entradas 4:1) e C128A (20 camadas, todas as entradas 128:1), mais uma janela de 128. Isso dá ≈ 3.850 B/token, contra 25 KB sem compressão, e 1M ≈ 4 GB por sequência.',
    'Em produção o indexador é replicado por rank. O SP indexer divide as chaves com merge exato (≈ −4 a −8% de TTFT a 128K), os workspaces precisam ter o tamanho certo (GLM <code>rightsize</code>: +26–28% de KV) e todo atalho dependente de dados precisa de guarda contra CUDA graphs (#49486/#52492; o #48957 só vale fora do graph FULL, que é o modo do decode uniforme na receita DS).',
  ],
  quiz: [
    { q: 'Numa camada C4A do DeepSeek-V4 com 1.048.576 tokens de contexto, quantas entradas de KV a atenção principal lê por token-consulta?', o: ['1.048.576', '8.320', '640', '128'], a: 2, e: 'Janela de 128 tokens brutos + as 512 entradas comprimidas 4:1 que o indexador escolheu = 640. 8.320 é a C128A (128 + 1.048.576/128), e 128 é uma camada só de janela.' },
    { q: 'Por que o lightning indexer usa ReLU em vez de softmax nos scores?', o: ['Porque o ReLU é diferenciável e o softmax não', 'Porque ele só precisa ordenar as chaves: com o ReLU cada cabeça apenas vota a favor, sem normalizar sobre todas as chaves', 'Porque o softmax não roda em FP8', 'Para que a atenção principal receba probabilidades prontas'], a: 1, e: 'O indexador só alimenta um top-k, que depende da ordem das notas e não de probabilidades. O ReLU zera a evidência negativa (a cabeça “não vota”) e dispensa a normalização sobre milhões de chaves. Os dois são diferenciáveis quase em toda parte, e a atenção principal faz o próprio softmax sobre as escolhidas.' },
    { q: 'Na receita GLM, o indexador expande 511 pools (e não 512) mais a cauda. Por quê?', o: ['Porque o kernel SM120 tem o top-k 2.048 como parâmetro de template e 512 × 4 + 3 = 2.051 não cabe; descartar o pool de menor ranking preserva a cauda recente', 'Para economizar memória de KV nas camadas MLA', 'Porque o 512º pool está sempre vazio', 'Porque o DeepGEMM exige larguras múltiplas de 511'], a: 0, e: 'O prefill do kernel sparse-MLA do GB10 rejeita qualquer top-k ≠ 2.048 na geometria GLM_NSA. Com 511 pools (2.044 tokens) + cauda (≤ 3), são no máximo 2.047 candidatos. Truncar a linha de 2.051 jogaria fora justamente a cauda, os tokens mais recentes; o 512º pool é o menos relevante dos escolhidos (0,2% dos candidatos).' },
    { q: 'Qual afirmação sobre o que é <i>guardado</i> no KV cache está correta?', o: ['A QSA guarda só as 2.048 posições escolhidas', 'O GLM guarda o latente MLA comprimido 4:1 nas camadas esparsas', 'Janelas deslizantes fazem o KV crescer com o contexto', 'O indexador reduz o que é lido, não o que é guardado; dos três modelos, só o DeepSeek-V4 comprime o KV principal armazenado'], a: 3, e: 'Qualquer chave pode ser escolhida por uma consulta futura, então GLM e Qwen guardam o KV de todas as posições (o GLM guarda o latente de 512, o Qwen guarda K/V em FP8) e comprimem só as chaves do indexador. O DeepSeek-V4 guarda entradas C4 e C128 no lugar do KV bruto, fora a janela de 128, que tem tamanho constante.' },
    { q: 'Por que o atalho de contexto curto (#49486) nunca é tomado durante a captura de um CUDA graph (guarda #52492)?', o: ['Porque a captura roda em FP32', 'Porque durante a captura ainda não existe KV cache', 'Porque o grafo capturado congelaria o atalho e o repetiria em prefixos longos, escolhendo 0..511 sem pontuar', 'Porque o atalho é mais lento dentro de grafos'], a: 2, e: 'Um CUDA graph grava a sequência de kernels da captura e a reproduz em todo replay, sem reavaliar o if. Se a captura pegasse o ramo “todos os candidatos”, prefixos longos receberiam os índices 0..511 sem scoring, uma corrupção intermitente sob prefix cache + graphs.' },
    { q: 'O SP indexer divide as chaves entre os ranks e funde os top-k locais. Por que o resultado é exatamente o top-k global?', o: ['Porque o all-gather ordena as chaves', 'Porque uma chave do top-k global tem no máximo k − 1 chaves melhores no total, logo também na sua fatia: ela está no top-k local do seu rank', 'Porque cada rank ainda pontua todas as chaves', 'Não é exato: é uma aproximação com erro pequeno'], a: 1, e: 'A união dos top-k locais sempre contém o top-k global, e o merge estável (kernel do DCP) o recupera. A decisão de usar o caminho SP usa metadado de CPU idêntico nos ranks, então a coletiva não dessincroniza. O ganho medido foi modesto (≈ −4 a −8% a 128K) porque o resto do prefill não foi dividido.' },
  ],
  mount(root, LA) {
    const V = LA.viz, UI = LA.ui;
    const nf = (x) => Math.round(x).toLocaleString('pt-BR');
    const sf = (x, d = 1) => (+x).toLocaleString('pt-BR', { minimumFractionDigits: d, maximumFractionDigits: d });
    const kfmt = (n) => (n >= 1048576 ? sf(n / 1048576, n % 1048576 ? 2 : 0) + 'M' : n >= 1024 ? nf(n / 1024) + 'K' : nf(n));
    const fmtB = (b) => (b >= 1073741824 ? sf(b / 1073741824, 2) + ' GiB' : b >= 1048576 ? sf(b / 1048576, 1) + ' MiB' : b >= 1024 ? sf(b / 1024, 1) + ' KiB' : nf(b) + ' B');
    let fcache = {};
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(() => { fcache = {}; });
    const font = (s, f, w) => fcache[s + f + w] || (fcache[s + f + w] = V.font(s, f, w));
    const tw = (g, s, size = 12, f = 'sans', w = 400) => { g.font = font(size, f, w); return g.measureText(String(s)).width; };
    const RED = (C) => C.red || C.c;
    // texto sobre preenchimentos vivos (menta, âmbar): a cor do fundo do tema, nos dois temas
    const onInk = (C) => C.bg;
    function wrap(g, s, maxW, size = 12, f = 'sans', w = 400) {
      const words = String(s).split(' ');
      const lines = [];
      let cur = '';
      for (const wd of words) {
        const tt = cur ? cur + ' ' + wd : wd;
        if (tw(g, tt, size, f, w) > maxW && cur) { lines.push(cur); cur = wd; } else cur = tt;
      }
      if (cur) lines.push(cur);
      return lines;
    }
    function textBlock(g, s, x, y, maxW, o = {}) {
      const size = o.size || 12, lh = o.lh || Math.round(size * 1.38);
      const lines = wrap(g, s, maxW, size, o.font || 'sans', o.weight || 400);
      const max = o.maxLines || 99;
      lines.slice(0, max).forEach((l, i) => V.text(g, l, x, y + i * lh, Object.assign({}, o, { size })));
      return Math.min(lines.length, max) * lh;
    }
    function pill(g, x, y, s, col, o = {}) {
      const size = o.size || 11, f = o.font || 'mono', wt = o.weight || 600;
      const wd = tw(g, s, size, f, wt) + 14, ht = size + 9;
      const xx = o.align === 'center' ? x - wd / 2 : o.align === 'right' ? x - wd : x;
      V.box(g, xx, y - ht / 2, wd, ht, { r: ht / 2, fill: V.alpha(col, o.a == null ? 0.18 : o.a), stroke: V.alpha(col, 0.8), lw: 1 });
      V.text(g, s, xx + wd / 2, y + 0.5, { size, font: f, weight: wt, color: o.color || V.C().ink, align: 'center', base: 'middle' });
      return wd;
    }
    function cellText(C, t, cm) {
      const light = cm === 'div' ? Math.abs(t) > 0.55 : t > 0.6;
      return light ? C.bg : C.ink;
    }
    function hatch(g, x, y, w, h, col, step = 6) {
      if (w <= 0 || h <= 0) return;
      g.save(); g.beginPath(); g.rect(x, y, w, h); g.clip();
      g.strokeStyle = col; g.lineWidth = 1; g.beginPath();
      for (let k = -h; k < w; k += step) { g.moveTo(x + k, y + h); g.lineTo(x + k + h, y); }
      g.stroke(); g.restore();
    }
    // altura do palco por largura (desktop × celular); aceita número ou função (altura que depende do estado)
    const reH = {};
    function autoH(sel, desk, mobH) {
      const fig = root.querySelector(sel);
      const st = fig && fig.querySelector('.la-fig-stage');
      if (!st) return;
      const val = (v) => (typeof v === 'function' ? v() : v);
      const set = () => { const hh = val(fig.clientWidth < 620 ? mobH : desk) + 'px'; if (st.style.height !== hh) st.style.height = hh; };
      set();
      reH[sel] = set;
      const ro = new ResizeObserver(set);
      ro.observe(fig);
      if (LA.onCleanup) LA.onCleanup(() => ro.disconnect());
    }
    const hS = { janelaRows: 4, compView: 'entry', compRatio: 4 };
    autoH('#fig-mascaras', 430, 440);
    autoH('#fig-janela', 440, () => 380 + hS.janelaRows * 24);
    autoH('#fig-indexador', 470, 400);
    autoH('#fig-destilacao', 400, 380);
    autoH('#fig-pools', 380, 380);
    autoH('#fig-compressor', 390, () => (hS.compView === 'decode' ? (hS.compRatio === 4 ? 330 : 356) : 390));
    autoH('#fig-balanca', 390, 446);
    autoH('#fig-sp-topk', 340, 330);
    autoH('#fig-semaforo', 360, 470);
    const softmax = (xs) => { const m = Math.max(...xs); const e = xs.map((x) => Math.exp(x - m)); const s = e.reduce((a, b) => a + b, 0); return e.map((x) => x / s); };

    figMascaras();
    figCusto();
    figJanela();
    figIndexador();
    figDestilacao();
    figPools();
    figCompressor();
    figBalanca();
    figSP();
    figSemaforo();

    /* ================================================================
       Figura 1 — máscaras de atenção
       ================================================================ */
    function figMascaras() {
      const fig = root.querySelector('#fig-mascaras');
      const ctl = UI.controls(fig);
      const S = { mode: 'dense', sweep: true };
      const MODES = [
        { v: 'dense', label: 'Densa' }, { v: 'swa', label: 'Janela' }, { v: 'topk', label: 'Top-k' },
        { v: 'pools', label: 'Pools + cauda' }, { v: 'c128', label: 'Comprimida' }, { v: 'c4a', label: 'Janela + top-k comprimido' },
      ];
      const INFO = {
        dense: { t: 'Atenção densa', d: 'Todo par (i, j ≤ i) é calculado: a consulta i lê i + 1 entradas, e o total cresce com o quadrado.', real: '1M tokens: 1.048.576 entradas por consulta' },
        swa: { t: 'Janela deslizante (W = 6)', d: 'Só os últimos W tokens: uma faixa diagonal de largura fixa. Barato e cego para o passado distante.', real: 'DeepSeek-V4: sliding_window = 128' },
        topk: { t: 'Top-k por conteúdo (k = 6)', d: 'Um indexador escolhe as k chaves mais relevantes de cada consulta; enquanto houver até k tokens, lê tudo.', real: 'DSA do DeepSeek-V3.2: 2.048 tokens' },
        pools: { t: 'Pools de 4 + cauda', d: 'As chaves são pontuadas em blocos de 4: ficam os 2 melhores blocos e a cauda (âmbar), que ainda não virou bloco.', real: 'GLM: 511 × 4 + 3 = 2.047 · QSA: ≤ 2.051' },
        c128: { t: 'Comprimida (tipo C128A)', d: 'Janela + TODOS os blocos fechados, cada um lido como uma entrada resumida (lavanda). Aqui os blocos têm 4 tokens em vez de 128.', real: 'C128A a 1M: 128 + 8.192 = 8.320' },
        c4a: { t: 'Janela + top-k comprimido (tipo C4A)', d: 'Janela + os 2 blocos comprimidos que o indexador escolheu, lidos como entradas resumidas.', real: 'C4A a 1M: 128 + 512 = 640' },
      };
      UI.seg(ctl, { label: 'Máscara', options: MODES, value: 'dense', onChange: (v) => { S.mode = v; } });
      UI.toggle(ctl, { label: 'varrer as consultas', value: true, onChange: (v) => { S.sweep = v; } });
      UI.legend(ctl, [['var(--viz-a)', 'token lido'], ['var(--viz-d)', 'cauda (sem bloco)'], ['var(--viz-b)', 'entrada comprimida'], ['var(--viz-grid)', 'não lido']]);
      const W = 6, K = 6, R = 4, SP = 2, NMAX = 40;
      const rng = V.rng(1313);
      const anchors = [1, 6, 13, 21, 27, 34];
      const rel = [];
      for (let i = 0; i < NMAX; i++) {
        rel[i] = [];
        for (let j = 0; j <= i; j++) {
          let v = rng() * 0.9;
          if (anchors.includes(j)) v += 1.1 + 0.5 * rng();
          if (i > 18 && j >= 15 && j <= 18) v += 0.9;
          v += Math.max(0, 1.1 - (i - j) * 0.3);
          rel[i][j] = v;
        }
      }
      const cache = {};
      function sel(mode, i) {
        const key = mode + ':' + i;
        if (cache[key]) return cache[key];
        const raw = new Set(), tail = new Set(), comp = [];
        const nComplete = Math.floor((i + 1) / R);
        const poolScore = (b) => { let s = 0; for (let j = b * R; j < b * R + R; j++) s += rel[i][j]; return s / R; };
        const topPools = (m) => { const arr = []; for (let b = 0; b < nComplete; b++) arr.push([poolScore(b), b]); arr.sort((a, b) => b[0] - a[0]); return arr.slice(0, m).map((x) => x[1]); };
        const win = () => { for (let j = Math.max(0, i - W + 1); j <= i; j++) raw.add(j); };
        if (mode === 'dense') for (let j = 0; j <= i; j++) raw.add(j);
        else if (mode === 'swa') win();
        else if (mode === 'topk') { const arr = []; for (let j = 0; j <= i; j++) arr.push([rel[i][j], j]); arr.sort((a, b) => b[0] - a[0]); arr.slice(0, K).forEach((x) => raw.add(x[1])); }
        else if (mode === 'pools') { topPools(SP).forEach((b) => { for (let j = b * R; j < b * R + R; j++) raw.add(j); }); for (let j = nComplete * R; j <= i; j++) tail.add(j); }
        else if (mode === 'c128') { win(); for (let b = 0; b < nComplete; b++) comp.push(b); }
        else if (mode === 'c4a') { win(); topPools(SP).forEach((b) => comp.push(b)); }
        return (cache[key] = { raw, tail, comp, reads: raw.size + tail.size + comp.length });
      }
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const mob = w < 640;
        const N = mob ? 24 : NMAX;
        let M, x0, y0, px, py, pw;
        if (!mob) { M = Math.min(h - 58, w * 0.55); x0 = 40; y0 = 36; px = x0 + M + 30; py = y0 - 6; pw = w - px - 16; }
        else { M = Math.min(w - 56, h - 176); x0 = 36; y0 = 30; px = 16; py = y0 + M + 14; pw = w - 32; }
        const cs = M / N, gp = cs > 7 ? 1 : 0.5;
        const cur = S.sweep ? (Math.floor(t * 2.6) + 17) % N : N - 1;
        V.text(g, 'chaves j →', x0, y0 - 12, { size: 11, font: 'mono', color: C.muted });
        g.save(); g.translate(x0 - 12, y0); g.rotate(Math.PI / 2); V.text(g, 'consultas i →', 0, 0, { size: 11, font: 'mono', color: C.muted }); g.restore();
        const blocky = S.mode === 'pools' || S.mode === 'c128' || S.mode === 'c4a';
        let tot = 0, totD = 0;
        for (let i = 0; i < N; i++) {
          const s = sel(S.mode, i);
          tot += s.reads; totD += i + 1;
          const rowA = S.sweep && i !== cur ? 0.3 : 1;
          const y = y0 + i * cs;
          s.comp.forEach((b) => {
            const x = x0 + b * R * cs;
            g.fillStyle = V.alpha(C.b, 0.6 * rowA);
            g.fillRect(x + gp, y + gp, R * cs - 2 * gp, cs - 2 * gp);
          });
          for (let j = 0; j <= i; j++) {
            const x = x0 + j * cs;
            let col = null;
            if (s.raw.has(j)) col = V.alpha(C.a, (0.45 + 0.55 * Math.min(1, rel[i][j] / 2.4)) * rowA);
            else if (s.tail.has(j)) col = V.alpha(C.d, 0.92 * rowA);
            const inComp = s.comp.length && s.comp.some((b) => j >= b * R && j < b * R + R);
            if (col) {
              const ins = inComp ? cs * 0.2 : 0;
              g.fillStyle = col; g.fillRect(x + gp + ins, y + gp + ins, cs - 2 * gp - 2 * ins, cs - 2 * gp - 2 * ins);
            } else if (!inComp) { g.fillStyle = V.alpha(C.muted, 0.1); g.fillRect(x + gp, y + gp, cs - 2 * gp, cs - 2 * gp); }
          }
        }
        g.strokeStyle = V.alpha(C.muted, 0.35); g.lineWidth = 1; g.strokeRect(x0 - 0.5, y0 - 0.5, M + 1, M + 1);
        if (blocky) for (let b = 1; b < N / R; b++) V.line(g, x0 + b * R * cs, y0 - 5, x0 + b * R * cs, y0, { color: V.alpha(C.muted, 0.7), lw: 1 });
        const yc = y0 + cur * cs;
        g.strokeStyle = C.c; g.lineWidth = 2; V.rr(g, x0 - 2.5, yc - 1.5, (cur + 1) * cs + 5, cs + 3, 3); g.stroke();
        const s = sel(S.mode, cur);
        const lx = x0 + (cur + 1) * cs + 8;
        if (lx + 70 < x0 + M + (mob ? 30 : 26)) V.text(g, '◀ i = ' + cur, lx, yc + cs / 2 + 0.5, { size: 11, font: 'mono', color: C.c, base: 'middle', weight: 600 });
        // painel
        const info = INFO[S.mode];
        let yy = py;
        V.text(g, info.t, px, yy + 14, { size: mob ? 14 : 16, weight: 600, color: C.ink });
        yy += mob ? 22 : 28;
        yy += textBlock(g, info.d, px, yy + 10, pw, { size: mob ? 12 : 13, color: C.muted, maxLines: mob ? 2 : 5 });
        yy += mob ? 6 : 14;
        V.text(g, `consulta i = ${cur}: lê ${s.reads} de ${cur + 1}`, px, yy + 10, { size: mob ? 12 : 13, font: 'mono', weight: 600, color: C.ink });
        yy += 16;
        const bw = Math.min(pw, 300);
        V.box(g, px, yy, bw, 8, { r: 4, fill: V.alpha(C.muted, 0.18) });
        V.box(g, px, yy, Math.max(4, bw * s.reads / (cur + 1)), 8, { r: 4, fill: C.a });
        yy += 22;
        if (!mob) {
          V.text(g, `prompt inteiro: ${nf(tot)} de ${nf(totD)} pares (${Math.round(100 * tot / totD)}%)`, px, yy + 8, { size: 12, font: 'mono', color: C.muted });
          yy += 34;
          V.text(g, 'EM ESCALA REAL', px, yy, { size: 10.5, font: 'mono', color: C.muted, weight: 600 });
          yy += 18;
          textBlock(g, info.real, px, yy, pw, { size: 13.5, weight: 600, color: C.c });
        } else {
          V.text(g, info.real, px, yy + 6, { size: 12, weight: 600, color: C.c, maxW: pw });
        }
      });
    }

    /* ================================================================
       Figura 2 — custo por consulta × contexto (log-log)
       ================================================================ */
    function figCusto() {
      const fig = root.querySelector('#fig-custo');
      const ctl = UI.controls(fig);
      const S = { lg: 17, idx: true };
      let H = null;
      UI.slider(ctl, { label: 'Contexto', min: 10, max: 20, step: 0.25, value: 17, fmt: (v) => kfmt(Math.round(2 ** v)) + ' tokens', onInput: (v) => { S.lg = v; if (H) H.redraw(); } });
      UI.toggle(ctl, { label: 'chaves pontuadas pelo indexador', value: true, onChange: (v) => { S.idx = v; if (H) H.redraw(); } });
      const CUR = [
        { l: 'Densa', s: 'Densa', col: (C) => RED(C), f: (n) => n },
        { l: 'DS · C128A (janela + tudo ÷128)', s: 'C128A', col: (C) => C.ink, f: (n) => Math.min(n, 128) + Math.floor(n / 128) },
        { l: 'Qwen · QSA (≤ 2.051)', s: 'QSA', col: (C) => C.qwen, f: (n) => Math.min(n, 2051) },
        { l: 'GLM · DSA (≤ 2.047)', s: 'GLM DSA', col: (C) => C.glm, f: (n) => Math.min(n, 2047), dash: [7, 6] },
        { l: 'DS · C4A (janela + top-512 ÷4)', s: 'C4A', col: (C) => C.ds, f: (n) => Math.min(n, 128) + Math.min(Math.floor(n / 4), 512) },
        { l: 'Janela (128)', s: 'Janela', col: (C) => C.muted, f: (n) => Math.min(n, 128) },
      ];
      const IDX = { l: 'Indexador: chaves pontuadas (n/4)', s: 'Indexador n/4', col: (C) => C.d, f: (n) => n / 4 };
      H = V.fig(fig, (g, w, h) => {
        const C = V.C();
        const mob = w < 620;
        const L = mob ? 44 : 58, T = 16, Rm = mob ? 14 : 250, B = mob ? 142 : 36;
        const pw = w - L - Rm, ph = h - T - B;
        const lx0 = 10, lx1 = 20, ly0 = 1, ly1 = 6.25;
        const X = (n) => L + ((Math.log2(n) - lx0) / (lx1 - lx0)) * pw;
        const Y = (v) => T + ph - ((Math.log10(Math.max(v, 10)) - ly0) / (ly1 - ly0)) * ph;
        const xt = [[1024, '1K'], [4096, '4K'], [16384, '16K'], [65536, '64K'], [262144, '256K'], [1048576, '1M']];
        const yt = [[10, '10'], [100, '100'], [1000, '1K'], [10000, '10K'], [100000, '100K'], [1000000, '1M']];
        xt.forEach(([v, l]) => { V.line(g, X(v), T, X(v), T + ph, { color: C.grid, lw: 1 }); V.text(g, l, X(v), T + ph + 15, { size: 11, font: 'mono', color: C.muted, align: 'center' }); });
        yt.forEach(([v, l]) => { V.line(g, L, Y(v), L + pw, Y(v), { color: C.grid, lw: 1 }); V.text(g, l, L - 6, Y(v) + 4, { size: 11, font: 'mono', color: C.muted, align: 'right' }); });
        V.text(g, 'entradas por consulta', L + 4, T + 10, { size: 11, font: 'mono', color: C.muted });
        if (!mob) V.text(g, 'tokens no contexto →', L + pw, T + ph + 30, { size: 11, font: 'mono', color: C.muted, align: 'right' });
        const draw = (cv, dash, lw) => {
          g.beginPath();
          for (let k = 0; k <= 160; k++) { const lg = lx0 + ((lx1 - lx0) * k) / 160; const n = 2 ** lg; const x = X(n), y = Y(cv.f(n)); if (k) g.lineTo(x, y); else g.moveTo(x, y); }
          g.strokeStyle = cv.col(C); g.lineWidth = lw || 2.4; if (dash) g.setLineDash(dash); g.stroke(); g.setLineDash([]);
        };
        if (S.idx) draw(IDX, [3, 5], 2.2);
        CUR.forEach((cv) => draw(cv, cv.dash, cv.s === 'GLM DSA' ? 2.4 : 2.4));
        const n = Math.round(2 ** S.lg);
        const xm = X(n);
        V.line(g, xm, T, xm, T + ph, { color: C.c, lw: 1.5, dash: [4, 4] });
        pill(g, xm, T + ph - 12, kfmt(n), C.c, { align: xm > L + pw - 40 ? 'right' : xm < L + 40 ? 'left' : 'center', size: 10.5 });
        const all = S.idx ? CUR.concat([IDX]) : CUR;
        all.forEach((cv) => { V.dot(g, xm, Y(cv.f(n)), 4.2, cv.col(C)); });
        // legenda
        const items = all.map((cv) => ({ cv, v: cv.f(n) }));
        if (!mob) {
          let ly = T + 8;
          const lx = L + pw + 18;
          V.text(g, `a ${nf(n)} tokens:`, lx, ly + 4, { size: 12, weight: 600, color: C.ink });
          ly += 22;
          items.forEach(({ cv, v }) => {
            g.strokeStyle = cv.col(C); g.lineWidth = 3; if (cv.dash || cv === IDX) g.setLineDash(cv === IDX ? [3, 4] : cv.dash); g.beginPath(); g.moveTo(lx, ly); g.lineTo(lx + 18, ly); g.stroke(); g.setLineDash([]);
            V.text(g, cv.l, lx + 26, ly + 4, { size: 11.5, color: C.ink, maxW: Rm - 40 });
            V.text(g, nf(v), lx + 26, ly + 20, { size: 12, font: 'mono', weight: 600, color: cv.col(C) });
            ly += 38;
          });
          const dense = n, c4 = CUR[4].f(n);
          V.text(g, `densa ÷ C4A = ${nf(dense / c4)}×`, lx, ly + 6, { size: 12, font: 'mono', color: C.muted });
        } else {
          const cols = 2, cw = (w - 24) / cols;
          items.forEach(({ cv, v }, i) => {
            const cx = 12 + (i % cols) * cw, cy = T + ph + 34 + Math.floor(i / cols) * 26;
            g.strokeStyle = cv.col(C); g.lineWidth = 3; if (cv.dash || cv === IDX) g.setLineDash(cv === IDX ? [3, 4] : cv.dash); g.beginPath(); g.moveTo(cx, cy - 4); g.lineTo(cx + 14, cy - 4); g.stroke(); g.setLineDash([]);
            V.text(g, cv.s + ': ' + nf(v), cx + 20, cy, { size: 11.5, font: 'mono', color: C.ink, maxW: cw - 24 });
          });
        }
      }, { static: true });
    }

    /* ================================================================
       Figura 3 — anel da janela + campo receptivo
       ================================================================ */
    function figJanela() {
      const fig = root.querySelector('#fig-janela');
      const ctl = UI.controls(fig);
      const S = { W: 6, L: 3, rf: true };
      let rR = null;
      const upd = () => { if (rR) rR.set(nf(S.L * (S.W - 1) + 1) + ' tokens'); hS.janelaRows = (S.rf ? S.L : 1) + 1; if (reH['#fig-janela']) reH['#fig-janela'](); };
      UI.slider(ctl, { label: 'Janela W', min: 3, max: 10, step: 1, value: 6, onInput: (v) => { S.W = v; upd(); } });
      UI.slider(ctl, { label: 'Camadas L', min: 1, max: 5, step: 1, value: 3, onInput: (v) => { S.L = v; upd(); } });
      UI.toggle(ctl, { label: 'empilhar camadas (campo receptivo)', value: true, onChange: (v) => { S.rf = v; upd(); } });
      rR = UI.readout(ctl, 'Alcance L·(W−1)+1', '—');
      UI.readout(ctl, 'Janelas DS (46 camadas)', '≈ 3,44 MB/seq');
      UI.readout(ctl, 'Janela DFlash2 (5 camadas)', '20 MiB');
      upd();
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const mob = w < 620;
        const Wn = S.W;
        const cont = t / 0.9;
        const p = Math.floor(cont) + 24;
        const fr = V.ease.out(Math.min(1, (cont - Math.floor(cont)) * 2.4));
        const topH = mob ? 318 : Math.round(h * 0.46);
        let cx0, cx1, cy, rcx, rcy, rr;
        if (!mob) { cx0 = 18; cx1 = w * 0.6; cy = 78; rcx = w * 0.81; rcy = topH / 2 + 12; rr = Math.min(topH / 2 - 34, w * 0.11); }
        else { cx0 = 12; cx1 = w - 12; cy = 70; rcx = w / 2; rcy = 196; rr = 76; }
        const step = mob ? 30 : 36, sz = step - 7;
        V.text(g, mob ? 'esteira de tokens (novo à direita)' : 'esteira de tokens (o mais novo chega pela direita)', cx0, 20, { size: 11.5, font: 'mono', color: C.muted });
        g.save(); g.beginPath(); g.rect(cx0 - 4, cy - sz, cx1 - cx0 + 8, sz * 2 + 8); g.clip();
        const nVis = Math.ceil((cx1 - cx0) / step) + 2;
        for (let k = 0; k < nVis; k++) {
          const j = p - k;
          const x = cx1 - sz - k * step + (1 - fr) * step;
          const inWin = k < Wn;
          const col = inWin ? C.a : C.muted;
          V.box(g, x, cy - sz / 2, sz, sz, { r: 6, fill: V.alpha(col, inWin ? 0.24 : 0.07), stroke: V.alpha(col, inWin ? 0.95 : 0.3), lw: k === 0 ? 2.2 : 1 });
          V.text(g, String(j), x + sz / 2, cy + 1, { size: 11, font: 'mono', align: 'center', base: 'middle', color: inWin ? C.ink : C.muted });
        }
        g.restore();
        const bR = cx1 + (1 - fr) * step - 7, bL = cx1 - sz - (Wn - 1) * step + (1 - fr) * step;
        const by = cy - sz / 2 - 9;
        V.line(g, Math.max(cx0, bL), by, bR, by, { color: C.a, lw: 2 });
        V.line(g, Math.max(cx0, bL), by, Math.max(cx0, bL), by + 5, { color: C.a, lw: 2 });
        V.line(g, bR, by, bR, by + 5, { color: C.a, lw: 2 });
        V.text(g, `janela W = ${Wn}`, Math.max(cx0, bL) + 4, by - 6, { size: 11, font: 'mono', color: C.a, weight: 600 });
        if (!mob) V.text(g, 'esquecidos: seus slots já foram sobrescritos', cx0, cy + sz / 2 + 18, { size: 11, color: C.muted });
        // anel
        const slotOf = (k) => p - ((((p - k) % Wn) + Wn) % Wn);
        const cursl = ((p % Wn) + Wn) % Wn;
        if (mob) V.text(g, 'anel de KV', 12, rcy - rr - 6, { size: 11.5, font: 'mono', color: C.muted });
        else V.text(g, 'anel de KV', w - 14, 20, { size: 11.5, font: 'mono', color: C.muted, align: 'right' });
        g.beginPath(); g.arc(rcx, rcy, rr, 0, Math.PI * 2); g.strokeStyle = V.alpha(C.muted, 0.3); g.lineWidth = 1; g.stroke();
        const ss = Math.min(28, (2 * Math.PI * rr) / Wn - 4);
        for (let k = 0; k < Wn; k++) {
          const a = -Math.PI / 2 + (2 * Math.PI * k) / Wn;
          const x = rcx + Math.cos(a) * rr, y = rcy + Math.sin(a) * rr;
          const on = k === cursl;
          if (on) V.glow(g, x, y, ss * 1.2 * (1.2 - fr * 0.4), C.c, 0.8 * (1 - fr * 0.6));
          V.box(g, x - ss / 2, y - ss / 2, ss, ss, { r: 6, fill: on ? V.alpha(C.c, 0.3) : V.alpha(C.a, 0.18), stroke: on ? C.c : V.alpha(C.a, 0.7), lw: on ? 2 : 1 });
          V.text(g, String(slotOf(k)), x, y + 1, { size: ss > 22 ? 11 : 10, font: 'mono', align: 'center', base: 'middle', color: C.ink });
          V.text(g, 's' + k, rcx + Math.cos(a) * (rr + ss * 0.95), rcy + Math.sin(a) * (rr + ss * 0.95) + 3, { size: 10, font: 'mono', align: 'center', color: C.muted });
        }
        V.text(g, 'slot = pos mod W', rcx, rcy - 2, { size: 11, font: 'mono', align: 'center', color: C.ink });
        V.text(g, `${p} mod ${Wn} = ${cursl}`, rcx, rcy + 14, { size: 11, font: 'mono', align: 'center', color: C.c, weight: 600 });
        if (!mob) {
          const nx = cx1 - sz / 2 + (1 - fr) * step;
          const a = -Math.PI / 2 + (2 * Math.PI * cursl) / Wn;
          V.arrow(g, Math.min(nx, cx1), cy + sz / 2 + 2, rcx + Math.cos(a) * (rr - ss / 2 - 4), rcy + Math.sin(a) * (rr - ss / 2 - 4), { color: V.alpha(C.c, 0.55), lw: 1.5, curve: -30, head: 7 });
        }
        // campo receptivo
        const L = S.rf ? S.L : 1;
        const y0 = topH + 16, x0 = mob ? 76 : 110;
        const Tc = mob ? 14 : 28;
        const rows = L + 1;
        const cell = Math.min((w - x0 - 14) / Tc, (h - y0 - 30) / rows - 5);
        const rowY = (r) => y0 + 18 + r * (cell + 5);
        V.text(g, S.rf ? (mob ? 'campo receptivo' : 'campo receptivo: quem pode influenciar o token atual') : 'uma camada: só a janela direta', 12, y0 + 4, { size: 11.5, font: 'mono', color: C.muted });
        const pc = Tc - 1;
        for (let r = 0; r < rows; r++) {
          const reach = r * (Wn - 1);
          const lbl = r === rows - 1 ? 'tokens' : `camada ${L - r}`;
          V.text(g, lbl, x0 - 8, rowY(r) + cell / 2 + 4, { size: 11, font: 'mono', color: C.muted, align: 'right' });
          for (let c = 0; c < Tc; c++) {
            const x = x0 + c * cell;
            const inR = c >= pc - reach && c <= pc;
            const a = inR ? 0.95 - 0.55 * (r / Math.max(1, rows - 1)) : 0;
            g.fillStyle = inR ? V.alpha(r === 0 ? C.c : C.a, r === 0 ? 0.9 : a) : V.alpha(C.muted, 0.09);
            g.fillRect(x + 1, rowY(r), cell - 2, cell);
          }
        }
        // leque da primeira camada e pulso descendo
        if (rows > 1) {
          const xa = x0 + pc * cell + cell / 2, ya = rowY(0) + cell;
          for (let c = Math.max(0, pc - Wn + 1); c <= pc; c++) V.line(g, xa, ya, x0 + c * cell + cell / 2, rowY(1), { color: V.alpha(C.c, 0.35), lw: 1 });
          const u = (t * 0.45) % 1;
          const rr2 = u * (rows - 1);
          const ri = Math.floor(rr2), fu = rr2 - ri;
          const reachHere = Math.min(pc, (ri + fu) * (Wn - 1));
          const px = x0 + (pc - reachHere) * cell + cell / 2;
          const py = V.lerp(rowY(ri), rowY(Math.min(rows - 1, ri + 1)), fu) + cell / 2;
          V.glow(g, px, py, 12, C.c, 0.9 * (1 - u * 0.7));
        }
        const reachT = L * (Wn - 1) + 1;
        if (reachT > Tc && S.rf) V.text(g, mob ? `alcance ${reachT} > ${Tc} mostradas` : `alcance ${reachT} > ${Tc} colunas mostradas`, w - 12, y0 + 4, { size: 11, font: 'mono', color: C.c, align: 'right' });
      });
    }

    /* ================================================================
       Figura 4 — o indexador passo a passo
       ================================================================ */
    function figIndexador() {
      const fig = root.querySelector('#fig-indexador');
      const ctl = UI.controls(fig);
      const S = { k: 3 };
      UI.slider(ctl, { label: 'k (top-k)', min: 1, max: 6, step: 1, value: 3, onInput: (v) => { S.k = v; } });
      const Q = [[2, -1, 1, 0], [0, 2, -1, 1], [1, 1, 0, -2], [-1, 0, 2, 1]];
      const KK = [[1, 0, 1, 0], [-1, 2, 0, 1], [2, -1, 0, -1], [0, 0, -1, 2], [1, 2, 1, -1], [-2, 1, 1, 0], [0, -1, 2, 1], [2, 1, -1, 0], [-1, -2, 0, 1], [1, 1, 1, 1]];
      const Wt = [1.0, 0.5, 1.25, 0.75];
      const HI = 4, D = 4;
      const SC = Q.map((q) => KK.map((k) => q.reduce((s, v, d) => s + v * k[d], 0)));
      const steps = [
        { t: 'A consulta e as chaves baratas', d: 'O token atual projeta \\(H_I = 4\\) consultas pequenas \\(\\mathbf q^I\\) (à esquerda). Cada token passado deixou no cache do indexador <b>uma</b> chave \\(\\mathbf k^I_s\\) (as colunas em cima). Na escala real: DeepSeek 64 × 128, GLM 32 × 128 e Qwen 4 × 128, sempre com uma chave compartilhada.' },
        { t: 'Produtos escalares', d: 'Cada célula da interseção é \\(\\mathbf q^I_h \\cdot \\mathbf k^I_s\\): a linha da esquerda vezes a coluna de cima, como numa multiplicação de matrizes à mão. Na receita isso é um GEMM pequeno em FP8/FP4 (<code>fp8_mqa_logits</code> do DeepGEMM; <code>_qsa_mqa_paged_kernel</code> no Qwen).' },
        { t: 'ReLU: só votos a favor', d: 'Scores negativos viram zero. Uma cabeça que “não gosta” de uma chave simplesmente não vota nela. Não há normalização sobre as chaves, então cada nota é independente das outras.' },
        { t: 'Soma ponderada pelas cabeças', d: 'Os pesos \\(w_{t,h}\\), calculados a partir do próprio token, dizem quanto confiar em cada cabeça. A soma dá uma nota \\(I_{t,s}\\) por chave: as barras. Repare na chave s6: só h0 (3) e h3 (5) votam nela, e os −3 de h1 e h2 viram zero no ReLU. Sem o ReLU a nota seria 3·1 − 3·0,5 − 3·1,25 + 5·0,75 = 1,5. Com ele, sobe para 6,75 e fica em 4º lugar, a um passo do top-3 (ponha k = 4 e ela entra).' },
        { t: 'Top-k', d: 'Ficam as \\(k\\) maiores notas (mude o \\(k\\) no controle). É um único conjunto, usado por todas as cabeças da atenção principal. Na receita: 2.048 tokens (GLM, Qwen) ou 512 entradas comprimidas (DeepSeek C4A).' },
        { t: 'A atenção cara, só nos escolhidos', d: 'A atenção principal busca no KV cache apenas as linhas escolhidas e faz o softmax sobre elas (pesos ilustrativos). As outras linhas continuam guardadas, só não são lidas neste passo.' },
      ];
      const st = UI.stepper(fig, steps, { autoplay: 6500 });
      V.fig(fig, (g, w, h) => {
        const C = V.C();
        const mob = w < 620;
        const n = mob ? 8 : 10;
        const el = st.elapsed();
        const step = st.i;
        const e = (d) => V.ease.inOut(Math.min(1, Math.max(0, el / d)));
        const ml = mob ? 30 : 40;
        const cell = Math.floor(Math.min(mob ? 28 : 30, (w - ml - 16) / (D + n + 2.4)));
        const gapA = Math.round(cell * 0.8), gapW = Math.round(cell * 0.5);
        const totalW = ml + D * cell + gapA + n * cell + gapW + cell;
        const ox = Math.max(0, (w - totalW) / 2);
        const qx = ox + ml, sx = qx + D * cell + gapA, wx = sx + n * cell + gapW;
        const ky = 26, sy = ky + D * cell + Math.round(cell * 0.6);
        const fs = cell >= 26 ? 11.5 : 10.5;
        const drawCell = (x, y, v, norm, o = {}) => {
          const tt = V.clamp(v / norm, -1, 1);
          g.fillStyle = o.fill || V.cmap(tt, 'div');
          V.rr(g, x + 1, y + 1, cell - 2, cell - 2, 4); g.fill();
          if (o.stroke) { g.strokeStyle = o.stroke; g.lineWidth = 2; V.rr(g, x, y, cell, cell, 5); g.stroke(); }
          if (o.label != null) V.text(g, o.label, x + cell / 2, y + cell / 2 + 0.5, { size: fs, font: 'mono', align: 'center', base: 'middle', color: o.tcolor || cellText(C, tt, 'div'), alpha: o.alpha });
        };
        // Kᵀ (d × n)
        V.text(g, '(K^I)ᵀ: uma chave por token', sx, ky - 9, { size: 11, font: 'mono', color: C.muted });
        for (let d = 0; d < D; d++) for (let s = 0; s < n; s++) drawCell(sx + s * cell, ky + d * cell, KK[s][d], 2, { label: KK[s][d] });
        // q (H × d)
        V.text(g, 'q^I', qx, sy - 9, { size: 11, font: 'mono', color: C.muted });
        for (let hh = 0; hh < HI; hh++) {
          V.text(g, 'h' + hh, qx - 6, sy + hh * cell + cell / 2 + 4, { size: 11, font: 'mono', color: C.muted, align: 'right' });
          for (let d = 0; d < D; d++) drawCell(qx + d * cell, sy + hh * cell, Q[hh][d], 2, { label: Q[hh][d] });
        }
        // scores
        const nShow = step === 1 ? Math.floor(el * 14) : step > 1 ? 99 : 0;
        const topk = (() => {
          const I = [];
          for (let s = 0; s < n; s++) { let v = 0; for (let hh = 0; hh < HI; hh++) v += Wt[hh] * Math.max(0, SC[hh][s]); I.push([v, s]); }
          const sorted = I.slice().sort((a, b) => b[0] - a[0]);
          return { I: I.map((x) => x[0]), sel: new Set(sorted.slice(0, S.k).map((x) => x[1])), order: sorted.map((x) => x[1]) };
        })();
        const reluE = step === 2 ? e(1.4) : step > 2 ? 1 : 0;
        for (let hh = 0; hh < HI; hh++) for (let s = 0; s < n; s++) {
          const x = sx + s * cell, y = sy + hh * cell;
          const idx = hh * n + s;
          if (idx >= nShow) { g.strokeStyle = V.alpha(C.muted, 0.25); g.lineWidth = 1; V.rr(g, x + 1.5, y + 1.5, cell - 3, cell - 3, 4); g.stroke(); continue; }
          const raw = SC[hh][s];
          const v = raw < 0 ? V.lerp(raw, 0, reluE) : raw;
          const selO = step >= 4 && topk.sel.has(s);
          drawCell(x, y, v, 5, { label: Math.round(v), stroke: selO ? C.c : null, alpha: raw < 0 && reluE > 0.5 ? 0.45 : 1 });
        }
        if (step >= 1 && !mob) V.text(g, 'q^I · k^I', sx + n * cell, sy - 9, { size: 11, font: 'mono', color: C.muted, align: 'right' });
        if (step === 2) pill(g, sx + (n * cell) / 2, sy + HI * cell + 26, 'ReLU: negativos → 0', C.d, { align: 'center', size: 11 });
        // pesos w
        if (step >= 3) {
          const a = step === 3 ? e(0.8) : 1;
          V.text(g, 'w', wx + cell / 2, sy - 9, { size: 11, font: 'mono', color: C.muted, align: 'center' });
          for (let hh = 0; hh < HI; hh++) {
            V.box(g, wx + 1, sy + hh * cell + 1, cell - 2, cell - 2, { r: 4, fill: V.alpha(C.d, 0.3 * a), stroke: V.alpha(C.d, 0.9 * a), lw: 1 });
            V.text(g, sf(Wt[hh], 2).replace(/0$/, ''), wx + cell / 2, sy + hh * cell + cell / 2 + 0.5, { size: cell >= 26 ? fs - 1 : 8.5, font: 'mono', align: 'center', base: 'middle', color: C.ink, alpha: a });
          }
        }
        // barras I
        const by = sy + HI * cell + 18;
        const bh = mob ? 62 : 70;
        if (step < 3) {
          g.strokeStyle = V.alpha(C.muted, 0.25); g.lineWidth = 1; g.setLineDash([4, 4]);
          V.rr(g, sx, by + 8, n * cell, bh, 6); g.stroke();
          V.rr(g, sx, by + bh + (mob ? 62 : 64), n * cell, 20, 5); g.stroke(); g.setLineDash([]);
          V.text(g, 'notas I por chave (passo 4)', sx + (n * cell) / 2, by + 8 + bh / 2 + 4, { size: 11, font: 'mono', color: C.muted, align: 'center' });
          V.text(g, 'KV cache principal (passo 6)', sx + (n * cell) / 2, by + bh + (mob ? 76 : 78), { size: 10.5, font: 'mono', color: C.muted, align: 'center' });
        }
        if (step >= 3) {
          const grow = step === 3 ? e(1.6) : 1;
          const mx = Math.max(...topk.I);
          V.text(g, mob ? 'nota I = Σ w·ReLU(q·k)' : 'nota I por chave = Σ w · ReLU(q·k)', sx, by + 2, { size: 11, font: 'mono', color: C.muted });
          for (let s = 0; s < n; s++) {
            const v = topk.I[s] * grow;
            const hgt = (v / mx) * (bh - 14);
            const x = sx + s * cell + 3, y = by + bh - hgt + 6;
            const selB = step >= 4 && topk.sel.has(s);
            const col = selB ? C.a : step >= 4 ? V.alpha(C.muted, 0.45) : C.b;
            V.box(g, x, y, cell - 6, Math.max(2, hgt), { r: 3, fill: col });
            if (cell >= 24 || selB) V.text(g, sf(topk.I[s], 2).replace(/,00$/, '').replace(/0$/, ''), x + (cell - 6) / 2, y - 3, { size: 10, font: 'mono', align: 'center', color: selB ? C.ink : C.muted, alpha: grow });
          }
          for (let s = 0; s < n; s++) V.text(g, 's' + s, sx + s * cell + cell / 2, by + bh + 20, { size: 10.5, font: 'mono', align: 'center', color: topk.sel.has(s) && step >= 4 ? C.a : C.muted });
          if (step >= 4) { const r = topk.order.slice(0, S.k).map((s) => 's' + s).join(', '); V.text(g, `top-${S.k}: ${r}`, sx - (mob ? 26 : 0), by + bh + 38, { size: 11.5, font: 'mono', weight: 600, color: C.a }); }
        }
        // KV strip
        if (step >= 5) {
          const a = e(0.8);
          const ky2 = by + bh + (mob ? 58 : 60);
          const selArr = topk.order.slice(0, S.k);
          const att = softmax(selArr.map((s) => topk.I[s] * 0.6));
          V.text(g, mob ? 'KV cache principal (softmax nas escolhidas)' : 'KV cache da atenção principal (pesos do softmax nas escolhidas)', qx, ky2 - 4, { size: 11, font: 'mono', color: C.muted });
          for (let s = 0; s < n; s++) {
            const on = topk.sel.has(s);
            const x = sx + s * cell + 2;
            V.box(g, x, ky2 + 4, cell - 4, 20, { r: 4, fill: on ? V.alpha(C.a, 0.3 + 0.5 * a) : V.alpha(C.muted, 0.1), stroke: on ? C.a : V.alpha(C.muted, 0.25), lw: 1 });
            if (on) { const k = selArr.indexOf(s); V.text(g, Math.round(att[k] * 100) + '%', x + (cell - 4) / 2, ky2 + 14.5, { size: cell >= 26 ? 10 : 9, font: 'mono', align: 'center', base: 'middle', color: onInk(C), alpha: a }); }
          }
          V.text(g, `lê ${S.k} de ${n}`, qx, ky2 + 18, { size: 11.5, font: 'mono', weight: 600, color: C.c, alpha: a });
        }
      });
    }

    /* ================================================================
       Figura 5 — destilação por KL
       ================================================================ */
    function figDestilacao() {
      const fig = root.querySelector('#fig-destilacao');
      const ctl = UI.controls(fig);
      const S = { tau: 0.35, k: 8 };
      UI.slider(ctl, { label: 'Treino (fase 1)', min: 0, max: 1, step: 0.01, value: 0.35, fmt: (v) => Math.round(v * 100) + '%', onInput: (v) => { S.tau = v; } });
      UI.slider(ctl, { label: 'k', min: 2, max: 16, step: 1, value: 8, onInput: (v) => { S.k = v; } });
      const NN = 40, r5 = V.rng(505);
      const zT = [], z0 = [], e0 = [];
      for (let s = 0; s < NN; s++) {
        let v = r5.normal() * 0.4;
        if ([4, 11, 19, 27].includes(s)) v += 4.0 + r5() * 0.6;
        v += Math.max(0, 3.0 - (NN - 1 - s) * 0.5);
        zT.push(v); z0.push(r5.normal() * 1.1); e0.push(r5.normal() * 0.45);
      }
      const p = softmax(zT);
      const steps = [
        { t: 'Fase 1 · warm-up denso', d: 'O modelo fica congelado e a atenção continua densa. Só o indexador aprende, minimizando a KL entre a distribuição dele (menta, embaixo) e a do professor (lavanda, em cima). Arraste o controle de treino.' },
        { t: 'Fase 2 · treino esparso conjunto', d: 'O top-k é ligado de verdade: as chaves não escolhidas somem da atenção (hachuradas) e o modelo inteiro aprende a trabalhar só com as escolhidas. A KL do indexador passa a olhar só o conjunto selecionado.' },
        { t: 'Na inferência', d: 'Só o indexador olha todas as chaves; a atenção principal lê as \\(k\\) escolhidas. Em cada cabeça, a diferença para a saída densa fica limitada por \\(2\\varepsilon_h \\max\\lVert v\\rVert\\), onde \\(\\varepsilon_h\\) é a massa dessa cabeça que ficou de fora (aqui há uma cabeça só).' },
      ];
      const st = UI.stepper(fig, steps, {});
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const mob = w < 620;
        const tau = S.tau;
        const zS = z0.map((z, s) => (1 - tau) * z + tau * (0.95 * zT[s] + e0[s]));
        const q = softmax(zS);
        const ordQ = q.map((v, s) => [v, s]).sort((a, b) => b[0] - a[0]);
        const ordP = p.map((v, s) => [v, s]).sort((a, b) => b[0] - a[0]);
        const selQ = new Set(ordQ.slice(0, S.k).map((x) => x[1]));
        let kl = 0;
        if (st.i === 0) { for (let s = 0; s < NN; s++) kl += p[s] * Math.log(p[s] / q[s]); }
        else {
          let ps = 0, qs = 0; selQ.forEach((s) => { ps += p[s]; qs += q[s]; });
          selQ.forEach((s) => { const a = p[s] / ps, b = q[s] / qs; kl += a * Math.log(a / b); });
        }
        const klLbl = st.i === 0 ? 'KL(p ‖ q)' : 'KL só no conjunto S';
        let cap = 0; selQ.forEach((s) => { cap += p[s]; });
        let orc = 0; ordP.slice(0, S.k).forEach((x) => { orc += x[0]; });
        const eps = 1 - cap;
        const gw = mob ? 0 : 220;
        const x0 = mob ? 14 : 20, x1 = w - gw - (mob ? 14 : 30);
        const midY = mob ? 150 : h / 2 + 2;
        const amp = mob ? 112 : h / 2 - 44;
        const bw = (x1 - x0) / NN;
        const mxP = Math.max(...p), mxQ = Math.max(...q, mxP * 0.5);
        const scale = Math.max(mxP, mxQ);
        V.text(g, 'professor: atenção densa p', x0, midY - amp - 6, { size: 11, font: 'mono', color: C.b });
        V.text(g, mob ? 'aluno: softmax(I) · tracejado = alvo (professor)' : 'aluno: softmax(I) do indexador · tracejado = o alvo, a forma do professor', x0, midY + amp + 18, { size: 11, font: 'mono', color: C.a, maxW: x1 - x0 });
        V.line(g, x0, midY, x1, midY, { color: V.alpha(C.muted, 0.5), lw: 1 });
        const phase2 = st.i >= 1;
        for (let s = 0; s < NN; s++) {
          const x = x0 + s * bw + 1;
          const hp = (p[s] / scale) * amp, hq = (q[s] / scale) * amp;
          const sel = selQ.has(s);
          if (phase2 && !sel) { g.fillStyle = V.alpha(C.b, 0.12); g.fillRect(x, midY - hp - 2, bw - 2, hp); hatch(g, x, midY - hp - 2, bw - 2, hp, V.alpha(C.b, 0.55), 4); }
          else { V.box(g, x, midY - hp - 2, bw - 2, Math.max(1, hp), { r: 2, fill: V.alpha(C.b, 0.85) }); }
          V.box(g, x, midY + 2, bw - 2, Math.max(1, hq), { r: 2, fill: sel ? C.a : V.alpha(C.a, 0.35) });
          // contorno-fantasma: onde o aluno deveria chegar se imitasse o professor
          if (hp > 1.5) { g.save(); g.setLineDash([3, 3]); g.strokeStyle = V.alpha(C.b, 0.55); g.lineWidth = 1; g.strokeRect(x + 0.5, midY + 2.5, bw - 3, hp - 1); g.restore(); }
          if (sel) { g.strokeStyle = C.c; g.lineWidth = 1.5; V.rr(g, x - 1, midY + 1, bw, Math.max(3, hq) + 2, 2); g.stroke(); }
        }
        if (st.i === 2) {
          const pulse = 0.5 + 0.5 * Math.sin(t * 3);
          selQ.forEach((s) => { const x = x0 + s * bw + bw / 2; V.glow(g, x, midY - 4, 10, C.c, 0.4 + 0.4 * pulse); });
        }
        // medidores
        let gx, gy, gW;
        if (!mob) { gx = w - gw + 4; gy = 26; gW = gw - 20; }
        else { gx = 14; gy = midY + amp + 34; gW = w - 28; }
        const klTxt = sf(kl, 3);
        if (!mob) {
          V.text(g, klLbl, gx, gy, { size: 11, font: 'mono', color: C.muted });
          V.text(g, klTxt + ' nats', gx, gy + 26, { size: 22, font: 'mono', weight: 700, color: kl < 0.2 ? C.a : kl < 0.8 ? C.d : RED(C) });
          gy += 58;
          V.text(g, `massa capturada pelo top-${S.k}`, gx, gy, { size: 11, font: 'mono', color: C.muted });
          gy += 12;
          V.box(g, gx, gy, gW, 14, { r: 7, fill: V.alpha(C.muted, 0.16) });
          V.box(g, gx, gy, Math.max(4, gW * cap), 14, { r: 7, fill: C.a });
          g.strokeStyle = C.b; g.lineWidth = 2; V.line(g, gx + gW * orc, gy - 4, gx + gW * orc, gy + 18, { color: C.b, lw: 2 });
          V.text(g, `aluno ${sf(cap * 100, 1)}%`, gx, gy + 32, { size: 12.5, font: 'mono', weight: 600, color: C.a });
          V.text(g, `oráculo ${sf(orc * 100, 1)}%`, gx, gy + 50, { size: 12, font: 'mono', color: C.b });
          gy += 76;
          V.text(g, 'ε = massa perdida', gx, gy, { size: 11, font: 'mono', color: C.muted });
          V.text(g, sf(eps * 100, 1) + '%', gx, gy + 22, { size: 18, font: 'mono', weight: 700, color: eps < 0.15 ? C.a : C.d });
          V.text(g, `erro ≤ ${sf(2 * eps, 2)}·max‖v‖`, gx, gy + 44, { size: 11.5, font: 'mono', color: C.muted });
          V.text(g, `${S.k} de ${NN} chaves = ${Math.round((100 * S.k) / NN)}%`, gx, gy + 66, { size: 11, font: 'mono', color: C.muted });
          V.text(g, 'receita: 2.048 de 1M ≈ 0,2%', gx, gy + 84, { size: 11, font: 'mono', color: C.c });
        } else {
          V.text(g, `${st.i === 0 ? 'KL' : 'KL em S'} = ${klTxt} nats`, gx, gy, { size: 13, font: 'mono', weight: 700, color: kl < 0.2 ? C.a : kl < 0.8 ? C.d : RED(C) });
          V.text(g, `ε = ${sf(eps * 100, 1)}%`, gx + gW, gy, { size: 13, font: 'mono', weight: 700, color: eps < 0.15 ? C.a : C.d, align: 'right' });
          gy += 14;
          V.box(g, gx, gy, gW, 12, { r: 6, fill: V.alpha(C.muted, 0.16) });
          V.box(g, gx, gy, Math.max(4, gW * cap), 12, { r: 6, fill: C.a });
          V.line(g, gx + gW * orc, gy - 3, gx + gW * orc, gy + 15, { color: C.b, lw: 2 });
          V.text(g, `top-${S.k}: aluno ${sf(cap * 100, 1)}% · oráculo ${sf(orc * 100, 1)}%`, gx, gy + 30, { size: 11.5, font: 'mono', color: C.ink });
        }
      });
    }

    /* ================================================================
       Figura 6 — pools de 4, a cauda e o buffer
       ================================================================ */
    function figPools() {
      const fig = root.querySelector('#fig-pools');
      const ctl = UI.controls(fig);
      const S = { mode: 'stock', n: 31, seed: 61 };
      UI.seg(ctl, { label: 'Caminho', options: [{ v: 'stock', label: 'GLM stock' }, { v: 'receita', label: 'GLM receita' }, { v: 'tf', label: 'GLM·TF' }, { v: 'qsa', label: 'Qwen QSA' }], value: 'stock', onChange: (v) => { S.mode = v; } });
      UI.slider(ctl, { label: 'Contexto', min: 13, max: 44, step: 1, value: 31, onInput: (v) => { S.n = v; } });
      UI.button(ctl, '↻ novos scores', () => { S.seed = (S.seed * 7 + 13) % 997; });
      const SEL = 4, RR = 4, TPL = 16, ROUND = 8;
      const tfc = (C) => C.__tf || (C.__tf = (getComputedStyle(document.documentElement).getPropertyValue('--tf') || '').trim() || C.d);
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const mob = w < 620;
        const n = S.n;
        const P = Math.floor(n / RR), tail = n % RR;
        const rng = V.rng(S.seed + 1);
        const score = Array.from({ length: P }, () => 0.15 + rng() * 0.85);
        const ord = score.map((v, b) => [v, b]).sort((a, b) => b[0] - a[0]).map((x) => x[1]);
        const nSel = Math.min(SEL, P);
        const selected = ord.slice(0, nSel);
        const expanded = S.mode === 'receita' ? ord.slice(0, Math.min(SEL - 1, P)) : selected;
        const dropped = S.mode === 'receita' && P >= SEL ? ord[SEL - 1] : -1;
        const expSet = new Set(expanded);
        const candidates = expanded.length * RR + tail;
        const bufW = S.mode === 'stock' ? Math.ceil((SEL * RR + (RR - 1)) / ROUND) * ROUND : S.mode === 'receita' ? TPL : SEL * RR + (RR - 1);
        const ok = S.mode !== 'stock';
        const x0 = mob ? 12 : 26, x1 = w - (mob ? 12 : 26);
        const cw = Math.min(26, (x1 - x0) / n);
        const tx = x0 + ((x1 - x0) - cw * n) / 2;
        const ty = 34;
        const dense = S.mode === 'tf' && P <= SEL;   // GLM·TF: até (SEL + 1)·4 − 1 tokens a linha é densa
        const methodTxt = S.mode === 'qsa' ? 'média → RMSNorm → RoPE no início do bloco (BF16)' : S.mode === 'tf' ? 'softmax(portão + ape) por canal → FP8 e4m3 + escala 2^k = 144 B' : 'softmax(portão + ape) por canal → Hadamard-128 → FP8 = 132 B';
        V.text(g, 'tokens do contexto (a consulta é o último)', x0, ty - 12, { size: 11.5, font: 'mono', color: C.muted });
        for (let j = 0; j < n; j++) {
          const b = Math.floor(j / RR);
          const inTail = b >= P;
          const inSel = !inTail && expSet.has(b);
          const x = tx + j * cw;
          const col = inTail ? C.d : inSel ? C.a : b === dropped ? RED(C) : C.muted;
          V.box(g, x + 1, ty, cw - 2, cw - 2, { r: 3, fill: V.alpha(col, inTail || inSel ? 0.75 : b === dropped ? 0.35 : 0.14), stroke: V.alpha(col, 0.7), lw: 1 });
          if (cw >= 18) V.text(g, String(j), x + cw / 2, ty + cw / 2, { size: 9.5, font: 'mono', align: 'center', base: 'middle', color: inTail || inSel ? onInk(C) : C.muted });
        }
        if (tail) { const xa = tx + P * RR * cw; V.text(g, 'cauda', xa + (tail * cw) / 2, ty + cw + 13, { size: 11, font: 'mono', align: 'center', color: C.d, weight: 600 }); }
        // pools
        const py = ty + cw + 44;
        const pw = Math.max(10, Math.min(RR * cw - 6, 40));
        const bh = mob ? 54 : 70;
        for (let b = 0; b < P; b++) {
          const xc = tx + b * RR * cw + (RR * cw) / 2;
          const sel = selected.includes(b);
          const exp = expSet.has(b);
          const col = exp ? C.a : b === dropped ? RED(C) : C.muted;
          for (let k = 0; k < RR; k++) {
            const xs = tx + (b * RR + k) * cw + cw / 2;
            V.line(g, xs, ty + cw - 1, xc, py, { color: V.alpha(col, exp ? 0.5 : 0.18), lw: 1 });
            if (exp || b === dropped) { const u = (t * 0.8 + k * 0.13 + b * 0.07) % 1; const pt = V.along(xs, ty + cw, xc, py, u); V.dot(g, pt[0], pt[1], 1.8, col); }
          }
          V.box(g, xc - pw / 2, py, pw, 16, { r: 4, fill: V.alpha(col, exp ? 0.8 : 0.2), stroke: col, lw: 1 });
          const hh = score[b] * bh;
          V.box(g, xc - pw / 2 + 2, py + 22 + bh - hh, pw - 4, hh, { r: 3, fill: V.alpha(col, exp ? 0.85 : b === dropped ? 0.5 : 0.35) });
          const rk = ord.indexOf(b) + 1;
          if (sel && pw >= 14 && !dense) V.text(g, '#' + rk, xc, py + 22 + bh - hh - 4, { size: 10, font: 'mono', align: 'center', color: b === dropped ? RED(C) : C.ink, weight: 600 });
          if (b === dropped) { V.line(g, xc - pw / 2, py + 2, xc + pw / 2, py + 14, { color: RED(C), lw: 2 }); V.line(g, xc + pw / 2, py + 2, xc - pw / 2, py + 14, { color: RED(C), lw: 2 }); }
        }
        const my = py + 22 + bh + 18;
        V.text(g, (mob ? '' : 'como 4 chaves viram 1: ') + methodTxt, x0, my, { size: 11, font: 'mono', color: S.mode === 'qsa' ? C.qwen : S.mode === 'tf' ? tfc(C) : C.glm, maxW: x1 - x0 });
        // buffer
        const bY = my + 34;
        const slotW = Math.min(22, (x1 - x0 - 10) / Math.max(bufW, TPL + 4));
        const bx = x0;
        const realW = S.mode === 'stock' ? '2.176' : S.mode === 'receita' ? '2.048' : '2.051';
        const bufLbl = S.mode === 'tf' && dense ? (mob ? `densa: lê ${n} posições, sem lista` : `atenção densa: lê as ${n} posições, sem lista de índices`) : mob ? `buffer: largura ${bufW} (real: ${realW})` : `buffer de índices: largura ${bufW} (em escala real ${realW})`;
        V.text(g, bufLbl, bx, bY - 8, { size: 11.5, font: 'mono', color: C.ink, maxW: x1 - x0 });
        for (let k = 0; k < bufW; k++) {
          const x = bx + k * slotW;
          const isTok = k < expanded.length * RR;
          const isTail = !isTok && k < candidates;
          const col = isTok ? C.a : isTail ? C.d : C.muted;
          if (!isTok && !isTail) { g.fillStyle = V.alpha(C.muted, 0.1); g.fillRect(x + 1, bY, slotW - 2, 16); hatch(g, x + 1, bY, slotW - 2, 16, V.alpha(C.muted, 0.4), 4); }
          else V.box(g, x + 1, bY, slotW - 2, 16, { r: 2, fill: V.alpha(col, 0.8) });
        }
        const tplX = bx + TPL * slotW;
        if (S.mode === 'stock' || S.mode === 'receita') {
          V.line(g, tplX, bY - 4, tplX, bY + 22, { color: ok ? C.e : RED(C), lw: 2.5 });
          V.text(g, 'template do kernel', tplX + 4, bY + 32, { size: 10.5, font: 'mono', color: ok ? C.e : RED(C) });
        }
        const yMsg = bY + (mob ? 54 : 50);
        let msg, sub;
        if (S.mode === 'stock') { msg = '✕ o kernel SM120 rejeita a tabela'; sub = 'expects sparse block_tables shape (256, 1, 2048), got (256, 1, 2176): 512 pools × 4 + cauda 3 = 2.051 → arredondado a 2.176'; }
        else if (S.mode === 'receita') { msg = '✓ cabe no template de 2.048'; sub = `511 pools × 4 + cauda ≤ 3 = 2.047; aqui ${candidates} válidos de ${bufW}. ${dropped >= 0 ? 'O pool #4 (o de menor ranking) é descartado: 0,2% em escala real.' : 'Poucos pools: nada é descartado.'}`; }
        else if (S.mode === 'tf' && dense) { msg = '✓ linha densa: lê todas as posições'; sub = `Até ${(SEL + 1) * RR - 1} tokens aqui (2.051 em escala real) o top-k escolheria tudo, e o TensorFold faz atenção densa sobre o latente: ${n} de ${n} posições.`; }
        else if (S.mode === 'tf') { msg = mob ? '✓ kernel próprio: 2.051, nada descartado' : '✓ kernel próprio do TensorFold: largura 2.051, nada descartado'; sub = `512 pools × 4 + cauda ≤ 3 = até 2.051 tokens; aqui ${candidates} de ${bufW}. Os pools saem de um radix select, em ordem crescente.`; }
        else { msg = '✓ kernels Triton próprios aceitam 2.051'; sub = `512 blocos × 4 + cauda ≤ 3 = até 2.051 posições; aqui ${candidates}. Depois, GQA esparsa sobre o K/V original.`; }
        V.text(g, msg, x0, yMsg, { size: 13, weight: 600, color: ok ? C.e : RED(C), maxW: x1 - x0 });
        textBlock(g, sub, x0, yMsg + 18, x1 - x0, { size: 11.5, color: C.muted, maxLines: 3 });
      });
    }

    /* ================================================================
       Figura 7 — o compressor do DeepSeek-V4
       ================================================================ */
    function figCompressor() {
      const fig = root.querySelector('#fig-compressor');
      const ctl = UI.controls(fig);
      const S = { ratio: 4, view: 'entry' };
      let tRef = 0, tNow = 0;
      UI.seg(ctl, { label: 'Razão', options: [{ v: 4, label: 'C4 (com sobreposição)' }, { v: 128, label: 'C128' }], value: 4, onChange: (v) => { S.ratio = v; tRef = tNow; hS.compRatio = v; if (reH['#fig-compressor']) reH['#fig-compressor'](); } });
      UI.seg(ctl, { label: 'Visão', options: [{ v: 'entry', label: 'uma entrada' }, { v: 'decode', label: 'token a token (decode)' }], value: 'entry', onChange: (v) => { S.view = v; tRef = tNow; hS.compView = v; if (reH['#fig-compressor']) reH['#fig-compressor'](); } });
      const CH = 6;
      const r7 = V.rng(707);
      const mk = (rows) => {
        const ape = Array.from({ length: Math.min(rows, 128) }, () => Array.from({ length: CH }, () => r7.normal() * 0.4));
        const Z = Array.from({ length: rows }, (_, j) => Array.from({ length: CH }, (_, c) => Math.round((r7.normal() * 1.2 + ape[j % ape.length][c]) * 10) / 10));
        const X = Array.from({ length: rows }, () => Array.from({ length: CH }, () => Math.round(r7.normal() * 10) / 10));
        const Wm = Array.from({ length: rows }, () => new Array(CH).fill(0));
        for (let c = 0; c < CH; c++) { const sm = softmax(Z.map((r) => r[c])); sm.forEach((v, j) => { Wm[j][c] = v; }); }
        const out = Array.from({ length: CH }, (_, c) => X.reduce((s, r, j) => s + Wm[j][c] * r[c], 0));
        return { Z, X, Wm, out };
      };
      const D4 = mk(8), D128 = mk(128);
      V.fig(fig, (g, w, h, t) => {
        tNow = t;
        const C = V.C();
        const mob = w < 620;
        const tt = t - tRef;
        if (S.view === 'entry') {
          const D = S.ratio === 4 ? D4 : D128;
          const per = 10;
          const u = tt % per;
          const ph = u < 2.2 ? 0 : u < 4.8 ? 1 : u < 7.2 ? 2 : 3;
          const pe = (a, b) => V.ease.inOut(V.clamp((u - a) / (b - a), 0, 1));
          const rowsShow = S.ratio === 4 ? 8 : 16;
          const mx = mob ? 76 : 150, my = 34;
          const cwid = Math.min(mob ? 42 : 52, (w - mx - (mob ? 14 : 250)) / CH);
          const chh = S.ratio === 4 ? (mob ? 24 : 30) : (mob ? 11 : 14);
          const title = ph === 0 ? 'notas do portão Z + B (ape)' : ph === 1 ? 'softmax por canal (coluna): pesos somam 1' : ph === 2 ? (mob ? 'soma: Σ_j s_j ⊙ C_j' : 'soma ponderada: Σ_j s_j ⊙ C_j (valores C não mostrados)') : 'pós-processamento → linha de 584 B';
          V.text(g, title, mx, my - 12, { size: 12, font: 'mono', weight: 600, color: C.ink });
          const colSweep = ph === 1 ? Math.min(CH - 1, Math.floor(pe(2.2, 4.6) * CH)) : -1;
          for (let rI = 0; rI < rowsShow; rI++) {
            const j = S.ratio === 4 ? rI : rI * 8;
            const y = my + rI * chh;
            if (S.ratio === 4) {
              const lbl = rI < 4 ? `i−1 · ${rI}` : `i · ${rI - 4}`;
              V.text(g, lbl, mx - 8, y + chh / 2 + 4, { size: 11, font: 'mono', color: rI < 4 ? C.b : C.a, align: 'right' });
            } else if (rI % 4 === 0) V.text(g, 'j=' + j, mx - 8, y + chh / 2 + 4, { size: 10.5, font: 'mono', color: C.muted, align: 'right' });
            for (let c = 0; c < CH; c++) {
              const x = mx + c * cwid;
              const z = D.Z[j][c], wv = D.Wm[j][c];
              const mixT = ph === 0 ? 0 : ph === 1 ? (c <= colSweep ? 1 : 0) : 1;
              const colZ = V.cmap(V.clamp(z / 3, -1, 1), 'div');
              const colW = V.cmap(Math.min(1, wv * (S.ratio === 4 ? 2.4 : 14)), 'seq');
              g.fillStyle = mixT ? colW : colZ;
              V.rr(g, x + 1, y + 1, cwid - 2, chh - 2, S.ratio === 4 ? 4 : 2); g.fill();
              if (S.ratio === 4) {
                const txt = mixT ? sf(wv, 2) : sf(z, 1);
                const tcol = mixT ? cellText(C, Math.min(1, wv * 2.4), 'seq') : cellText(C, V.clamp(z / 3, -1, 1), 'div');
                V.text(g, txt, x + cwid / 2, y + chh / 2 + 0.5, { size: mob ? 10 : 11, font: 'mono', align: 'center', base: 'middle', color: tcol });
              }
            }
          }
          const mBot = my + rowsShow * chh;
          if (S.ratio === 4) {
            V.line(g, mx - 2, my + 4 * chh, mx + CH * cwid + 2, my + 4 * chh, { color: C.c, lw: 1.5, dash: [4, 3] });
            V.text(g, mob ? 'i−1·j: bloco anterior, projeção a · i·j: atual, projeção b' : 'i−1·j = token do bloco anterior (projeção a) · i·j = do bloco atual (projeção b)', mob ? 14 : mx, mBot + 30, { size: 10.5, font: 'mono', color: C.muted, maxW: mob ? w - 28 : CH * cwid + 230 });
          } else V.text(g, '… 128 posições no bloco (1 a cada 8 mostrada), sem sobreposição', mob ? 14 : mx, mBot + 30, { size: 10.5, font: 'mono', color: C.muted, maxW: mob ? w - 28 : CH * cwid + 200 });
          if (colSweep >= 0) { g.strokeStyle = C.c; g.lineWidth = 2; V.rr(g, mx + colSweep * cwid - 1, my - 2, cwid + 2, rowsShow * chh + 4, 5); g.stroke(); }
          for (let c = 0; c < CH; c++) V.text(g, 'c' + c, mx + c * cwid + cwid / 2, mBot + 14, { size: 10.5, font: 'mono', align: 'center', color: C.muted });
          // saída
          const oy = mBot + 44;
          if (ph >= 2) {
            const a = ph === 2 ? pe(4.8, 6.4) : 1;
            V.text(g, 'c_i', mx - 8, oy + 16, { size: 12, font: 'mono', color: C.c, align: 'right', weight: 600 });
            for (let c = 0; c < CH; c++) {
              const x = mx + c * cwid;
              if (ph === 2 && a < 1) { const k = Math.floor((u * 6) % rowsShow); const [ax, ay] = [x + cwid / 2, my + k * chh + chh / 2]; V.arrow(g, ax, ay, x + cwid / 2, oy + 2, { color: V.alpha(C.c, 0.35), lw: 1, head: 5 }); }
              V.box(g, x + 1, oy, cwid - 2, 26, { r: 5, fill: V.alpha(C.c, 0.2 * a + 0.05), stroke: V.alpha(C.c, a), lw: 1.5 });
              V.text(g, sf(D.out[c], 2), x + cwid / 2, oy + 13.5, { size: mob ? 10 : 11, font: 'mono', align: 'center', base: 'middle', color: C.ink, alpha: a });
            }
          }
          // pipeline
          const chips = ['RMSNorm', 'RoPE (pos. do 1º token)', 'FP8 448 + BF16 64', '584 B no cache'];
          let cx, cy;
          if (!mob) { cx = mx + CH * cwid + 34; cy = my + 14; } else { cx = 14; cy = oy + 44; }
          const a3 = ph === 3 ? pe(7.2, 8.6) : ph > 3 ? 1 : 0;
          if (!mob) V.text(g, 'depois do pooling', cx, cy - 4, { size: 11, font: 'mono', color: C.muted });
          chips.forEach((s, i) => {
            const on = ph === 3 && a3 * chips.length > i;
            if (!mob) { pill(g, cx, cy + 22 + i * 34, s, on ? C.c : C.muted, { a: on ? 0.25 : 0.08, color: on ? C.ink : C.muted }); if (i < chips.length - 1) V.arrow(g, cx + 20, cy + 32 + i * 34, cx + 20, cy + 46 + i * 34, { color: V.alpha(C.muted, 0.6), lw: 1.2, head: 5 }); }
          });
          if (mob) {
            let xx = cx;
            chips.forEach((s, i) => { const on = ph === 3 && a3 * chips.length > i; const wd = tw(g, s, 10, 'mono', 600) + 14; if (xx + wd > w - 10) { xx = cx; cy += 26; } pill(g, xx, cy, s, on ? C.c : C.muted, { size: 10, a: on ? 0.25 : 0.08, color: on ? C.ink : C.muted }); xx += wd + 6; });
          }
          if (!mob) {
            const sy = cy + 22 + chips.length * 34 + 10;
            V.text(g, 'cache comprimido da camada', cx, sy, { size: 11, font: 'mono', color: C.muted });
            for (let k = 0; k < 7; k++) {
              const isNew = k === 6 && ph === 3 && a3 >= 1;
              const filled = k < 6 || isNew;
              V.box(g, cx + k * 26, sy + 10, 22, 22, { r: 4, fill: filled ? V.alpha(isNew ? C.c : C.b, isNew ? 0.6 : 0.4) : 'transparent', stroke: V.alpha(isNew ? C.c : C.b, 0.8), lw: 1 });
            }
            V.text(g, `${S.ratio === 4 ? '1 entrada a cada 4 tokens' : '1 entrada a cada 128 tokens'}`, cx, sy + 52, { size: 11, font: 'mono', color: C.ink });
          }
        } else {
          // decode incremental
          const r = S.ratio;
          const dt = r === 4 ? 0.7 : 0.075;
          const pos = Math.floor(tt / dt) + (r === 4 ? 16 : 256);
          const inBlk = (pos % r) + 1;
          const emit = (pos + 1) % r === 0;
          const x0 = mob ? 14 : 30, x1 = w - (mob ? 14 : 30);
          const ty = 44;
          V.text(g, mob ? `pos = ${nf(pos)} · (pos+1) mod ${r} = ${(pos + 1) % r}` : `decode: token pos = ${nf(pos)} · (pos + 1) mod ${r} = ${(pos + 1) % r}`, x0, 22, { size: 12, font: 'mono', weight: 600, color: C.ink });
          // estado parcial
          const slots = r === 4 ? 4 : 128;
          const sw = (x1 - x0) / (r === 4 ? 8 : 128);
          const stLbl = r === 4 ? 'kv_state / score_state: 4 tokens do bloco anterior (projeção a) + 4 do bloco atual (projeção b)' : 'kv_state / score_state: 128 posições do bloco atual';
          const stH = textBlock(g, stLbl, x0, ty + 4, x1 - x0, { size: 11, font: 'mono', color: C.muted, lh: 14, maxLines: 2 });
          const sy = ty + stH;
          if (r === 4) {
            for (let k = 0; k < 8; k++) {
              const cur = k >= 4;
              const filled = !cur || k - 4 < inBlk;
              const col = cur ? C.a : C.b;
              V.box(g, x0 + k * sw + 2, sy, sw - 4, 34, { r: 6, fill: filled ? V.alpha(col, 0.5) : 'transparent', stroke: V.alpha(col, 0.8), lw: 1.2 });
              V.text(g, cur ? `b·${k - 4}` : `a·${k}`, x0 + k * sw + sw / 2, sy + 18, { size: 11, font: 'mono', align: 'center', base: 'middle', color: C.ink });
            }
          } else {
            for (let k = 0; k < slots; k++) { g.fillStyle = k < inBlk ? V.alpha(C.a, 0.75) : V.alpha(C.muted, 0.12); g.fillRect(x0 + k * sw, sy, Math.max(1, sw - 0.6), 34); }
            V.text(g, `${inBlk} / 128`, x1, sy + 50, { size: 12, font: 'mono', color: C.a, align: 'right', weight: 600 });
          }
          // emissão
          const ey = sy + 72;
          const just = r === 4 ? emit : inBlk <= 12;
          const flash = r === 4 ? (emit ? 1 - (tt % dt) / dt : 0) : Math.max(0, 1 - inBlk / 12);
          V.text(g, 'cache comprimido', x0, ey - 6, { size: 11, font: 'mono', color: C.muted });
          const nE = Math.floor((pos + 1) / r);
          const shown = Math.min(nE, mob ? 7 : 14);
          for (let k = 0; k < shown; k++) {
            const isNew = k === shown - 1 && just;
            const x = x0 + k * 26;
            if (isNew) V.glow(g, x + 11, ey + 13, 22, C.c, flash);
            V.box(g, x, ey + 2, 22, 22, { r: 4, fill: V.alpha(isNew ? C.c : C.b, isNew ? 0.7 : 0.4), stroke: V.alpha(isNew ? C.c : C.b, 0.9), lw: 1 });
          }
          V.text(g, `${nf(nE)} entradas`, x0 + shown * 26 + 8, ey + 17, { size: 11.5, font: 'mono', color: C.ink });
          if (just && !mob) pill(g, x1, ey + 13, 'emite 1 entrada!', C.c, { align: 'right', size: 11 });
          // janela
          const wy = ey + 50;
          const tokW = Math.min(12, (x1 - x0) / 40);
          V.text(g, 'tokens recentes: a janela de 128 cobre o bloco em formação', x0, wy, { size: 11, font: 'mono', color: C.muted, maxW: x1 - x0 });
          const ntk = Math.floor((x1 - x0) / tokW);
          for (let k = 0; k < ntk; k++) {
            const j = pos - (ntk - 1 - k);
            const inCur = j > pos - inBlk;
            g.fillStyle = inCur ? V.alpha(C.d, 0.85) : V.alpha(C.a, 0.3);
            g.fillRect(x0 + k * tokW + 0.5, wy + 10, tokW - 1, 16);
          }
          V.line(g, x0, wy + 32, x1, wy + 32, { color: C.e, lw: 2 });
          V.text(g, 'janela SWA: últimos 128 brutos (trecho)', x1, wy + 46, { size: 11, font: 'mono', color: C.e, align: 'right' });
          V.text(g, 'âmbar = bloco em formação (ainda sem entrada comprimida)', x0, wy + 46 + (mob ? 16 : 0), { size: 11, font: 'mono', color: C.d, maxW: mob ? x1 - x0 : (x1 - x0) * 0.55 });
          const ny = wy + (mob ? 84 : 76);
          const idle = r - 1;
          const txt = r === 128 ? `C128: 1 emissão a cada 128 passos, ${idle} de 128 sem trabalho. O #48957 grava só o estado parcial e retorna cedo (~2× no kernel, upstream), mas só em passos piecewise ou eager; sob graph FULL (o decode uniforme da receita) o compressor roda inteiro.` : 'C4: 1 emissão a cada 4 passos; cada entrada mistura 8 posições (bloco atual + anterior).';
          textBlock(g, txt, x0, ny, x1 - x0, { size: 12, color: C.ink, maxLines: 5 });
        }
      });
    }

    /* ================================================================
       Figura 8 — a balança de bytes
       ================================================================ */
    function figBalanca() {
      const fig = root.querySelector('#fig-balanca');
      const ctl = UI.controls(fig);
      const GiB = 1073741824;
      const MODELS = {
        ds: { name: 'DeepSeek-V4-Flash', pool: 17.04 * GiB, poolTxt: '17,04 GiB (boot)', served: 1048576, rep: 'por rank (replicado nos 2)',
          layers: Array.from({ length: 46 }, (_, i) => (i < 2 || i > 42 ? 'swa' : i % 2 === 0 ? 'c4a' : 'c128a')),
          comps: [
            { l: 'KV C4A (21 × 584/4)', c: 'ds', bt: 3066 },
            { l: 'KV C128A (20 × 584/128)', c: 'b', bt: 91.25 },
            { l: 'K do indexador (21 × 132/4)', c: 'd', bt: 693 },
            { l: 'janelas (46 × 128 × 584 B)', c: 'e', fixed: 3440640 },
          ] },
        glm: { name: 'GLM-5.3-Flash', pool: 11 * GiB, poolTxt: '11 GiB (teto de KV)', served: 850000, rep: 'por rank (latente replicado)',
          layers: Array.from({ length: 46 }, (_, i) => (i === 45 ? 'dsa' : i % 4 === 3 ? 'dsa' : 'kda')),
          comps: [
            { l: 'latente fp8_ds_mla (11 × 656)', c: 'glm', bt: 7216 },
            { l: 'indexador kpool (11 × 33)', c: 'd', bt: 363 },
            { l: 'estado KDA (34 × 2,34 MB)', c: 'muted', fixed: 79659008 },
            { l: 'janela DFlash2 (5 × 2048 B)', c: 'e', perTokCap: 2048, bt: 10240 },
          ] },
        qwen: { name: 'Qwen3.8-Flash-Next', pool: 32.02 * GiB, poolTxt: '32,02 GiB (boot)', served: 262144, rep: 'por GPU (2 cabeças de KV divididas)',
          layers: Array.from({ length: 49 }, (_, i) => (i === 48 ? 'qsa' : (i + 1) % 4 === 0 ? 'qsa' : 'gdn')),
          comps: [
            { l: 'KV FP8 (13 camadas)', c: 'qwen', bt: 6656 },
            { l: 'indexador + página (medido)', c: 'd', bt: 2762 },
            { l: 'estado GDN FP32', c: 'muted', fixed: 56623104 },
          ] },
        nocomp: { name: 'Hipótese: DS sem compressão', pool: 17.04 * GiB, poolTxt: '17,04 GiB (pool do DS)', served: 1048576, rep: 'por rank',
          layers: Array.from({ length: 43 }, () => 'dense'),
          comps: [{ l: 'KV 584 B × 43 camadas', c: 'red', bt: 25112 }] },
        gqa: { name: 'Hipótese: GQA-8 em BF16', pool: 17.04 * GiB, poolTxt: '17,04 GiB (pool do DS)', served: 1048576, rep: 'por rank',
          layers: Array.from({ length: 43 }, () => 'dense'),
          comps: [{ l: 'KV 8 × 128 × 2 × 2 B × 43', c: 'red', bt: 176128 }] },
      };
      const TYPES = { swa: ['e', 'só janela'], c4a: ['ds', 'C4A'], c128a: ['b', 'C128A'], dsa: ['glm', 'MLA + DSA'], kda: ['muted', 'KDA (linear)'], qsa: ['qwen', 'QSA'], gdn: ['muted', 'GDN (linear)'], dense: ['red', 'densa'] };
      const S = { m: 'ds', lg: 20, seqs: 2 };
      let H = null;
      const rB = { t: null, s: null, f: null };
      UI.select(ctl, { label: 'Modelo', options: [{ v: 'ds', label: 'DeepSeek-V4-Flash' }, { v: 'glm', label: 'GLM-5.3-Flash' }, { v: 'qwen', label: 'Qwen3.8-Flash-Next' }, { v: 'nocomp', label: 'Hipótese: DS sem compressão' }, { v: 'gqa', label: 'Hipótese: GQA-8 BF16' }], value: 'ds', onChange: (v) => { S.m = v; if (H) H.redraw(); } });
      UI.slider(ctl, { label: 'Contexto', min: 10, max: 20, step: 0.25, value: 20, fmt: (v) => kfmt(Math.round(2 ** v)), onInput: (v) => { S.lg = v; if (H) H.redraw(); } });
      UI.slider(ctl, { label: 'Sequências', min: 1, max: 8, step: 1, value: 2, onInput: (v) => { S.seqs = v; if (H) H.redraw(); } });
      rB.t = UI.readout(ctl, 'Bytes/token', '—');
      rB.s = UI.readout(ctl, 'KV por sequência', '—');
      rB.f = UI.readout(ctl, 'Cabem (físico)', '—');
      H = V.fig(fig, (g, w, h) => {
        const C = V.C();
        const col = (k) => (k === 'red' ? RED(C) : k === 'muted' ? C.muted : C[k]);
        const mob = w < 620;
        const M = MODELS[S.m];
        const n = Math.round(2 ** S.lg);
        const x0 = mob ? 12 : 24, x1 = w - (mob ? 12 : 24), W = x1 - x0;
        V.text(g, M.name + ' · mapa de camadas', x0, 20, { size: 12.5, weight: 600, color: C.ink });
        const lw = W / M.layers.length;
        M.layers.forEach((ty, i) => {
          const [ck] = TYPES[ty];
          const isM = (S.m === 'ds' && i > 42) || (S.m === 'glm' && i === 45) || (S.m === 'qwen' && i === 48);
          V.box(g, x0 + i * lw + 0.5, 30, lw - 1, 20, { r: 2, fill: V.alpha(col(ck), ck === 'muted' ? 0.28 : 0.8), stroke: isM ? C.c : null, lw: 1.2 });
        });
        const used = Array.from(new Set(M.layers));
        // legenda do mapa de camadas: quebra para a linha seguinte quando falta espaço
        let lx = x0, lyL = 67;
        used.forEach((ty) => { const [ck, lbl] = TYPES[ty]; const cnt = M.layers.filter((x) => x === ty).length; const s = `${lbl} ×${cnt}`; const wd = tw(g, s, 11) + 26; if (lx + wd > x1 && lx > x0) { lx = x0; lyL += 16; } V.box(g, lx, lyL - 9, 10, 10, { r: 2, fill: V.alpha(col(ck), ck === 'muted' ? 0.4 : 0.85) }); V.text(g, s, lx + 14, lyL, { size: 11, color: C.muted }); lx += wd; });
        if (S.m === 'ds' || S.m === 'glm' || S.m === 'qwen') { const s = 'contorno = MTP/rascunhador'; if (lx + tw(g, s, 11) > x1 && lx > x0) { lx = x0; lyL += 16; } V.text(g, s, lx, lyL, { size: 11, color: C.c }); }
        // uma sequência
        const parts = M.comps.map((cp) => {
          let b;
          if (cp.fixed) b = cp.fixed;
          else if (cp.perTokCap) b = Math.min(n, cp.perTokCap) * cp.bt;
          else b = cp.bt * n;
          return { cp, b };
        });
        const perSeq = parts.reduce((s, x) => s + x.b, 0);
        const perTok = M.comps.filter((c) => !c.fixed && !c.perTokCap).reduce((s, c) => s + c.bt, 0);
        const sy = lyL + 29;
        let sy2 = sy;
        if (mob) { V.text(g, `uma sequência de ${nf(n)} tokens: ${fmtB(perSeq)}`, x0, sy, { size: 12.5, font: 'mono', weight: 600, color: C.ink }); V.text(g, M.rep, x0, sy + 16, { size: 11, font: 'mono', color: C.muted }); sy2 = sy + 16; }
        else V.text(g, `uma sequência de ${nf(n)} tokens (${M.rep}): ${fmtB(perSeq)}`, x0, sy, { size: 12.5, font: 'mono', weight: 600, color: C.ink });
        const barY = sy2 + 10, barH = 22;
        const scaleB = Math.max(perSeq, M.pool * 0.25);
        V.box(g, x0, barY, W, barH, { r: 6, fill: V.alpha(C.muted, 0.1) });
        let bx = x0;
        parts.forEach(({ cp, b }) => { const ww = (b / scaleB) * W; if (ww > 0.3) { g.fillStyle = V.alpha(col(cp.c), 0.85); g.fillRect(bx, barY, Math.min(ww, x1 - bx), barH); } bx += ww; });
        V.text(g, `escala: ${fmtB(scaleB)}`, x1, barY + barH + 13, { size: 10.5, font: 'mono', color: C.muted, align: 'right' });
        let ly = barY + barH + (mob ? 30 : 22);
        const colsL = mob ? 1 : 2, cwL = W / colsL;
        parts.forEach(({ cp, b }, i) => {
          const cx = x0 + (i % colsL) * cwL, cy = ly + Math.floor(i / colsL) * 18;
          V.box(g, cx, cy - 9, 10, 10, { r: 2, fill: V.alpha(col(cp.c), 0.85) });
          V.text(g, `${cp.l}: ${fmtB(b)}`, cx + 15, cy, { size: 11.5, font: 'mono', color: C.ink, maxW: cwL - 20 });
        });
        ly += Math.ceil(parts.length / colsL) * 18 + 20;
        // pool
        const tot = perSeq * S.seqs;
        const fits = tot <= M.pool;
        const fitN = Math.floor(M.pool / perSeq);
        V.text(g, `${S.seqs} × KV contra o pool de ${M.poolTxt}`, x0, ly, { size: 12.5, font: 'mono', weight: 600, color: C.ink, maxW: W });
        const pY = ly + 10, pH = 30;
        V.box(g, x0, pY, W, pH, { r: 8, fill: V.alpha(C.muted, 0.12), stroke: V.alpha(C.muted, 0.35), lw: 1 });
        const segW = (perSeq / M.pool) * W;
        for (let k = 0; k < S.seqs; k++) {
          const sx = x0 + k * segW;
          if (sx >= x1) break;
          const ww = Math.min(segW, x1 - sx);
          g.fillStyle = V.alpha(fits ? C.a : RED(C), 0.35 + 0.35 * (k % 2)); g.fillRect(sx, pY + 3, Math.max(1, ww - 1), pH - 6);
        }
        if (!fits) { hatch(g, x1 - 40, pY, 40, pH, V.alpha(RED(C), 0.8), 5); }
        const msg = fits ? `✓ cabe: ocupa ${Math.round((100 * tot) / M.pool)}% do pool` : `✕ não cabe: ${fmtB(tot)} > ${fmtB(M.pool)}`;
        V.text(g, msg, x0, pY + pH + 20, { size: 13, weight: 600, color: fits ? C.e : RED(C) });
        let ny = pY + pH + 40;
        if (n > M.served) { V.text(g, `⚠ acima do MAX_MODEL_LEN servido (${nf(M.served)})`, x0, ny, { size: 11.5, font: 'mono', color: C.d }); ny += 18; }
        const note = S.m === 'ds' ? 'custo físico; o alocador conta ≈ 7,8 KB/token (4 grupos, mesma página): 2,22× a 1M' : S.m === 'glm' ? 'custo físico; o README conta 14,52 GiB por requisição de 1M (≈ 7,4 GiB fixos de páginas híbridas + 7,1 GiB por 1M)' : '';
        if (note) textBlock(g, note, x0, ny, W, { size: 11, font: 'mono', color: C.muted, maxLines: 3 });
        rB.t.set(nf(perTok) + ' B');
        rB.s.set(fmtB(perSeq));
        rB.f.set(fitN >= 1000 ? '≥ 1.000' : nf(fitN) + ' seq');
      }, { static: true });
    }

    /* ================================================================
       Figura 9 — top-k dividido entre ranks
       ================================================================ */
    function figSP() {
      const fig = root.querySelector('#fig-sp-topk');
      const ctl = UI.controls(fig);
      const S = { tp: 2, k: 4, seed: 9 };
      let scores = [];
      const gen = () => { const r = V.rng(S.seed); scores = Array.from({ length: 24 }, (_, i) => 10 + Math.floor(r() * 89) + i * 0.001); };
      gen();
      UI.seg(ctl, { label: 'TP', options: [2, 3, 4], value: 2, onChange: (v) => { S.tp = v; } });
      UI.slider(ctl, { label: 'k', min: 2, max: 6, step: 1, value: 4, onInput: (v) => { S.k = v; } });
      UI.button(ctl, '🎲 sortear scores', () => { S.seed = (S.seed * 31 + 7) % 1009; gen(); });
      const steps = [
        { t: 'Fatiar e pontuar', d: 'Cada rank pega uma fatia contígua das chaves comprimidas, alinhada à página (aqui, de 2 chaves), e só pontua a sua parte: 1/TP do trabalho. No stock, os dois ranks pontuariam todas.' },
        { t: 'Top-k local', d: 'Cada rank escolhe os \\(k\\) melhores da própria fatia (contorno rosa), com <code>top_k_per_row_prefill</code>.' },
        { t: 'all_gather de (score, id)', d: 'Os candidatos viajam: todo rank recebe TP × k pares (score, id global). Em escala real, TP2 × 512 × 8 B = 8 KiB por consulta.' },
        { t: 'Merge estável e exato', d: 'Os TP × k candidatos são ordenados e ficam os \\(k\\) melhores (<code>stable_topk_from_gathered_candidates_cutedsl</code>), escritos direto no <code>topk_indices_buffer</code>. O selo compara com a força bruta.' },
      ];
      const st = UI.stepper(fig, steps, { autoplay: 4200 });
      V.fig(fig, (g, w, h) => {
        const C = V.C();
        const mob = w < 620;
        const N = mob ? 16 : 24;
        const sc = scores.slice(0, N);
        const tp = S.tp, k = S.k;
        const page = 2;
        const per = Math.ceil(N / tp / page) * page;
        const rankOf = (i) => Math.min(tp - 1, Math.floor(i / per));
        const rc = [C.a, C.b, C.d, C.f];
        const x0 = mob ? 12 : 30, x1 = w - (mob ? 12 : 30);
        const bw = (x1 - x0) / N;
        const barTop = 40, barH = mob ? 90 : 110;
        const el = st.elapsed(), s = st.i;
        const local = [];
        for (let r = 0; r < tp; r++) { const ids = []; for (let i = 0; i < N; i++) if (rankOf(i) === r) ids.push(i); ids.sort((a, b) => sc[b] - sc[a]); local.push(ids.slice(0, k)); }
        const locSet = new Set(local.flat());
        const glob = Array.from({ length: N }, (_, i) => i).sort((a, b) => sc[b] - sc[a]).slice(0, k);
        const cands = local.flat().slice().sort((a, b) => sc[b] - sc[a]);
        const merged = cands.slice(0, k);
        const same = merged.length === glob.length && merged.every((x, i) => x === glob[i]);
        V.text(g, mob ? 'scores de uma consulta (1 barra = 1 chave)' : 'scores do indexador para uma consulta (uma barra por chave comprimida)', x0, 18, { size: 11.5, font: 'mono', color: C.muted });
        for (let r = 0; r < tp; r++) {
          const a = r * per, b = Math.min(N, (r + 1) * per);
          if (a >= N) continue;
          const xa = x0 + a * bw, xb = x0 + b * bw;
          V.box(g, xa + 1, barTop - 6, xb - xa - 2, barH + 12, { r: 8, fill: V.alpha(rc[r], 0.06), stroke: V.alpha(rc[r], 0.5), lw: 1 });
          const lbl = tw(g, `rank ${r} · chaves ${a}–${b - 1}`, 11, 'mono', 600) < xb - xa - 8 ? `rank ${r} · chaves ${a}–${b - 1}` : `r${r} · ${a}–${b - 1}`;
          V.text(g, lbl, xa + 6, barTop + barH + 22, { size: 11, font: 'mono', color: rc[r], weight: 600 });
        }
        for (let i = 0; i < N; i++) {
          const r = rankOf(i);
          const hh = (sc[i] / 100) * barH;
          const x = x0 + i * bw + 2, y = barTop + barH - hh;
          const isLoc = s >= 1 && locSet.has(i);
          V.box(g, x, y, bw - 4, hh, { r: 3, fill: V.alpha(rc[r], s >= 1 && !isLoc ? 0.3 : 0.85) });
          if (isLoc) { g.strokeStyle = C.c; g.lineWidth = 2; V.rr(g, x - 1.5, y - 1.5, bw - 1, hh + 3, 4); g.stroke(); }
          if (bw >= 20) V.text(g, String(Math.round(sc[i])), x + (bw - 4) / 2, y - 4, { size: 10, font: 'mono', align: 'center', color: C.muted });
        }
        // gather + merge
        const my = barTop + barH + (mob ? 58 : 60);
        const nC = cands.length;
        const cw2 = Math.min(mob ? 30 : 44, (x1 - x0) / nC - 3);
        if (s >= 2) {
          V.text(g, `all_gather: ${tp} × ${k} = ${nC} candidatos (score, id)`, x0, my - 8, { size: 11.5, font: 'mono', color: C.ink });
          const prog = s === 2 ? V.ease.inOut(Math.min(1, el / 1.6)) : 1;
          const sortP = s >= 3 ? V.ease.inOut(Math.min(1, el / 1.2)) : 0;
          const lflat = local.flat();
          const slotX = (j) => x0 + j * (cw2 + 3) + cw2 / 2;
          lflat.forEach((id) => {
            const r = rankOf(id);
            const j = V.lerp(lflat.indexOf(id), cands.indexOf(id), sortP);
            const sx = x0 + id * bw + bw / 2, sy = barTop + barH - (sc[id] / 100) * barH;
            const tx = slotX(j), ty = my + 4 - Math.sin(sortP * Math.PI) * 10 * (lflat.indexOf(id) % 2 ? 1 : -1);
            const x = V.lerp(sx, tx, prog), y = V.lerp(sy, ty, prog);
            const jj = cands.indexOf(id);
            const keep = s >= 3 && jj < k && sortP > 0.9;
            const a = s >= 3 && jj >= k && sortP > 0.9 ? 0.3 : 1;
            V.box(g, x - cw2 / 2, y, cw2, 32, { r: 5, fill: V.alpha(rc[r], 0.25 * a + 0.05), stroke: V.alpha(keep ? C.c : rc[r], a), lw: keep ? 2 : 1 });
            V.text(g, String(Math.round(sc[id])), x, y + 12, { size: mob ? 10 : 11, font: 'mono', align: 'center', base: 'middle', color: C.ink, alpha: a });
            V.text(g, '#' + id, x, y + 25, { size: mob ? 9.5 : 10, font: 'mono', align: 'center', base: 'middle', color: C.muted, alpha: a });
          });
        }
        if (s >= 3) {
          const vy = my + 58;
          V.text(g, `merge → top-${k}: ${merged.map((x) => '#' + x).join(' ')}`, x0, vy, { size: 12, font: 'mono', weight: 600, color: C.c, maxW: x1 - x0 });
          V.text(g, `força bruta: ${glob.map((x) => '#' + x).join(' ')}`, x0, vy + 18, { size: 12, font: 'mono', color: C.muted, maxW: x1 - x0 });
          pill(g, x1, vy + 6, same ? '✓ idêntico' : '✕ diferente', same ? C.e : RED(C), { align: 'right', size: 12 });
        }
        const fy = h - 12;
        V.text(g, `cada rank pontua ${per} de ${N} chaves (1/${tp})`, x0, fy, { size: 11, font: 'mono', color: C.muted });
      });
    }

    /* ================================================================
       Figura 10 — semáforos do indexador
       ================================================================ */
    function figSemaforo() {
      const fig = root.querySelector('#fig-semaforo');
      const ctl = UI.controls(fig);
      const S = { lg: 10.5, cap: false, bnd: false, full: false, sp: false };
      UI.slider(ctl, { label: 'Contexto', min: 8, max: 20, step: 0.25, value: 10.5, fmt: (v) => nf(Math.round(2 ** v)) + ' tok', onInput: (v) => { S.lg = v; } });
      UI.toggle(ctl, { label: 'capturando CUDA graph', value: false, onChange: (v) => { S.cap = v; } });
      UI.toggle(ctl, { label: 'passo cruza fronteira de 128', value: false, onChange: (v) => { S.bnd = v; } });
      UI.seg(ctl, { label: 'Modo do graph', options: [{ v: false, label: 'PIECEWISE (lote com prefill)' }, { v: true, label: 'FULL (decode uniforme)' }], value: false, onChange: (v) => { S.full = v; } });
      UI.toggle(ctl, { label: 'SP indexer ligado', value: false, onChange: (v) => { S.sp = v; } });
      function states() {
        const n = Math.round(2 ** S.lg), keys = Math.floor(n / 4);
        const st = [];
        if (keys <= 512 && !S.cap) st.push({ c: 'g', t: 'C4A · indexador', h: 'Atalho #49486', d: `n/4 = ${nf(keys)} ≤ 512: todos seriam escolhidos. Preenche os índices 0..⌊n/4⌋−1 (as entradas já fechadas) sem pontuar; o compressor ainda grava o cache K.` });
        else if (keys <= 512) st.push({ c: 'a', t: 'C4A · indexador', h: 'Guarda #52492', d: 'Capturando CUDA graph: pontua mesmo assim, senão o grafo congelaria o atalho e o repetiria em prefixos longos.' });
        else st.push({ c: 'r', t: 'C4A · indexador', h: 'Pontua e escolhe top-512', d: `n/4 = ${nf(keys)} chaves comprimidas > 512: logits FP8/FP4 + top-k sobre todas elas.` });
        if (S.full) st.push({ c: 'a', t: 'C128A · compressor', h: 'Graph FULL: roda sempre', d: 'O passo inteiro é um grafo: o if ficou decidido na captura e o atalho #48957 fica desligado. É o decode uniforme da receita DS.' });
        else if (S.bnd) st.push({ c: 'r', t: 'C128A · compressor', h: 'Emite uma entrada', d: 'Algum pedido cruza uma fronteira de 128: compress → RMSNorm → RoPE → quant → escrita.' });
        else st.push({ c: 'g', t: 'C128A · compressor', h: 'Pula (#48957)', d: 'Piecewise: a atenção roda eager e o if é reavaliado. Sem fronteira de 128, grava só o estado parcial e retorna cedo (~2× no kernel, upstream).' });
        st.push({ c: 'a', t: 'Prefill denso', h: 'Dormente (#48407)', d: 'Sem rota MHA densa neste fork: o prefill curto também usa o indexador e o kernel esparso (no GLM: supports_dense_mha_prefill = False).' });
        if (!S.sp) st.push({ c: 'r', t: 'SP indexer (prefill)', h: 'Replicado', d: `Cada rank pontua todas as ${nf(keys)} chaves × 64 cabeças: trabalho duplicado nos 2 ranks.` });
        else if (keys >= 8192) st.push({ c: 'g', t: 'SP indexer (prefill)', h: 'Dividido entre ranks', d: `n/4 = ${nf(keys)} ≥ MIN_KEYS 8.192: fatias por rank + all-gather + merge exato.` });
        else st.push({ c: 'a', t: 'SP indexer (prefill)', h: 'Abaixo de MIN_KEYS', d: `n/4 = ${nf(keys)} < 8.192 (32K tokens): este chunk usa o caminho replicado stock.` });
        return st;
      }
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const mob = w < 620;
        const st = states();
        const colOf = (c) => (c === 'g' ? C.e : c === 'a' ? C.d : RED(C));
        const drawLight = (x, y, c, sz) => {
          // caixa do semáforo: escura nos dois temas, derivada das cores do tema
          const casing = C.dark ? V.mix(C.bg, C.ink, 0.07) : V.mix(C.ink, C.bg, 0.16);
          V.box(g, x - sz * 0.75, y - sz * 0.2, sz * 1.5, sz * 3.9, { r: sz * 0.5, fill: casing, stroke: V.alpha(C.muted, 0.5), lw: 1 });
          ['r', 'a', 'g'].forEach((k, i) => {
            const cy = y + sz * 0.55 + i * sz * 1.2;
            const on = k === c;
            if (on) V.glow(g, x, cy, sz * 1.3, colOf(k), 0.7 + 0.25 * Math.sin(t * 4));
            V.dot(g, x, cy, sz * 0.45, on ? colOf(k) : V.alpha(colOf(k), 0.18));
          });
        };
        // agenda do carro: anda até a linha de parada de cada semáforo, espera se não estiver verde
        const seg = 1.25;
        const dwell = st.map((s) => (s.c === 'r' ? 1.1 : s.c === 'a' ? 0.55 : 0));
        const total = seg * (st.length + 1) + dwell.reduce((a, b) => a + b, 0);
        let u = t % total, pos = st.length + 1;
        for (let i = 0; i <= st.length; i++) {
          if (u < seg) { pos = i + u / seg; break; }
          u -= seg;
          if (i < st.length) { if (u < dwell[i]) { pos = i + 1; break; } u -= dwell[i]; }
        }
        const along = (stop, start, end) => {
          if (pos <= 1) return V.lerp(start, stop(0), V.smooth(pos));
          const i = Math.floor(pos) - 1, f = pos - Math.floor(pos);
          const a = stop(i), b = i + 1 < st.length ? stop(i + 1) : end;
          return V.lerp(a, b, V.smooth(f));
        };
        if (!mob) {
          const x0 = 16, x1 = w - 16, roadY = 104;
          const colW = (x1 - x0) / st.length;
          const lightX = (i) => x0 + colW * i + colW * 0.78;
          g.fillStyle = V.alpha(C.muted, 0.16); g.fillRect(x0, roadY - 12, x1 - x0, 24);
          V.line(g, x0, roadY, x1, roadY, { color: V.alpha(C.ink, 0.35), lw: 1.5, dash: [10, 10], dashOffset: -t * 30 });
          st.forEach((s, i) => {
            const xl = x0 + colW * i;
            drawLight(lightX(i), 16, s.c, 12);
            V.line(g, lightX(i) - 4, roadY - 12, lightX(i) - 4, roadY + 12, { color: V.alpha(colOf(s.c), 0.85), lw: 3 });
            V.text(g, s.t, xl + 10, roadY + 36, { size: 12, font: 'mono', color: C.muted, maxW: colW - 16 });
            V.text(g, s.h, xl + 10, roadY + 56, { size: 14, weight: 600, color: colOf(s.c), maxW: colW - 16 });
            textBlock(g, s.d, xl + 10, roadY + 78, colW - 20, { size: 12, color: C.ink, maxLines: 8 });
          });
          const X = along((i) => lightX(i) - 62, x0 - 60, x1 + 8);
          g.save(); g.beginPath(); g.rect(x0, roadY - 14, x1 - x0, 28); g.clip();
          V.box(g, X, roadY - 10, 52, 20, { r: 8, fill: C.c, stroke: V.alpha(C.ink, 0.3), lw: 1 });
          V.text(g, 'token', X + 26, roadY + 1, { size: 11, font: 'mono', weight: 600, align: 'center', base: 'middle', color: onInk(C) });
          g.restore();
        } else {
          const roadX = 22, y0 = 12, y1 = h - 12;
          const rowH = (y1 - y0) / st.length;
          const lineY = (i) => y0 + rowH * i + rowH * 0.45;
          g.fillStyle = V.alpha(C.muted, 0.16); g.fillRect(roadX - 10, y0, 20, y1 - y0);
          V.line(g, roadX, y0, roadX, y1, { color: V.alpha(C.ink, 0.35), lw: 1.5, dash: [8, 8], dashOffset: -t * 25 });
          st.forEach((s, i) => {
            const y = y0 + rowH * i;
            drawLight(roadX + 34, y + 10, s.c, 8);
            V.line(g, roadX - 10, lineY(i), roadX + 10, lineY(i), { color: V.alpha(colOf(s.c), 0.85), lw: 3 });
            const tx = roadX + 56, tW = w - tx - 10;
            V.text(g, s.t, tx, y + 14, { size: 11, font: 'mono', color: C.muted, maxW: tW });
            V.text(g, s.h, tx, y + 31, { size: 13, weight: 600, color: colOf(s.c), maxW: tW });
            textBlock(g, s.d, tx, y + 48, tW, { size: 11, color: C.ink, maxLines: 4, lh: 14 });
          });
          const Y = along((i) => lineY(i) - 26, y0 - 30, y1 + 4);
          g.save(); g.beginPath(); g.rect(roadX - 12, y0, 24, y1 - y0); g.clip();
          V.box(g, roadX - 9, Y, 18, 22, { r: 6, fill: C.c });
          g.restore();
        }
      });
    }
  },
});

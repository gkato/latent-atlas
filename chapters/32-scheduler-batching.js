LA.chapter({
  id: 'scheduler-batching',
  part: 6,
  order: 32,
  title: 'Scheduler, batching contínuo e chunked prefill',
  navTitle: 'Scheduler & batching',
  subtitle: 'Quem entra em cada passo: orçamentos de tokens e de sequências, prefill em fatias, fairness, preempção e agendamento assíncrono — e como escolher os números da sua receita.',
  minutes: 50,
  recipes: ['glm', 'tf', 'qwen', 'ds'],
  tags: ['scheduler', 'batching contínuo', 'chunked prefill', 'fairness', 'preempção', 'async scheduling', 'max_num_seqs', 'max_num_batched_tokens', 'TensorFold', 'multi-prefill'],
  goals: [
    'Explicar o laço do scheduler V1 e os dois orçamentos: <code>max_num_seqs</code> e <code>max_num_batched_tokens</code>',
    'Calcular linhas de verificação, tamanhos de captura e o trade-off TTFT × pausa do chunked prefill',
    'Diagnosticar inanição de decode e escolher proteções (#27, #43, <code>fair</code> v5/v7)',
    'Montar um perfil de serviço e validá-lo com escada de concorrência, spread e preempções',
    'Comparar com o scheduler do TensorFold: chunk <i>ou</i> rodada por crédito de iterações, janela em lote de 32 linhas e multi-prefill',
  ],
  prereqs: ['paged-attention-kv', 'prefill-decode-kv', 'cuda-graphs-warmup'],
  summary: [
    'A cada passo o scheduler V1 entrega \\(s_r \\le n_r - c_r\\) tokens a cada requisição, dentro de dois orçamentos: <code>max_num_batched_tokens</code> (tokens) e <code>max_num_seqs</code> (vagas) — RUNNING primeiro, na ordem, depois WAITING.',
    'Batching contínuo reocupa vagas a cada passo; como o decode é limitado por banda, mais streams elevam o agregado e reduzem a velocidade de cada um (Qwen: 3,8× agregado, 49% por stream de ×1 a ×8).',
    'Com especulação cada decode custa \\(1+k\\) posições: as linhas de verificação \\(S(1+k)\\) ditam os CUDA graphs (DeepSeek: 6 × 7 = 42 → captura 48).',
    'Chunked prefill troca TTFT por pausa do decode (\\(t \\approx a + b\\,c\\)); o limiar de prefill longo limita o pedaço por requisição (DeepSeek 1024) e, em híbridos, o pedaço para nos checkpoints do estado (GLM 3584).',
    'Inanição sem preempção é invisível: o DeepSeek limita prefills em voo (#27) e reserva piso de decode (#43); o GLM divide tempo, não tokens (<code>fair</code> v5), porque cada passo misto sai do graph FULL e custa ~0,3 s. A versão TensorFold do GLM nem mistura: cada iteração é um chunk <i>ou</i> uma rodada, alternados por um crédito em iterações (<code>TF_GLM_FILL_SHARE</code> 0,5), e prompts que chegam juntos dividem um chunk (TTFT ×4: 590 → 340 ms).',
    'Tetos não são reservas: a soma de tokens vivos precisa caber no pool, senão há preempção e recomputação; o agendamento assíncrono esconde CPU, mas cria placeholders.',
    'Perfis de serviço se validam com escadas de concorrência medindo agregado, por stream, TTFT, spread e preempções — uma variável por vez.',
  ],
  quiz: [
    { q: 'No vLLM V1, um decode fica sem tokens porque prefills na frente da lista RUNNING consumiram o orçamento. O que acontece com ele?', o: ['É preemptado e refaz o prefill', 'Recebe 0 tokens naquele passo, mantém o KV e não conta como preempção', 'Vai para o fim da fila WAITING', 'O vLLM aumenta o orçamento automaticamente'], a: 1, e: 'O laço de RUNNING para quando o orçamento acaba, e quem tem 0 tokens é pulado com <code>continue</code>. O KV continua lá e nenhuma métrica de preempção dispara — por isso a inanição do #27 tinha zero preempções.' },
    { q: 'Receita DeepSeek: 6 vagas e k = 6 rascunhos. Quantas linhas de verificação tem o passo de decode cheio e qual captura o compose usa?', o: ['6 e 8', '36 e 40', '42 e 48', '48 e 56'], a: 2, e: '6 × (1+6) = 42 linhas; o compose arredonda para múltiplo de 8 → 48. O valor cru 42 era truncado para 40, sobravam graphs de 7…35 e o passo de 42 tokens rodava eager (agregado c=6 122 vs 139 tok/s).' },
    { q: 'Por que o GLM trocou tetos em tokens (<code>GLM53_MIXED_PREFILL_CHUNK=N</code>) por fairness por tempo (<code>fair</code> v5)?', o: ['Porque tokens de prefill custam mais que tokens de decode', 'Porque todo passo misto sai do graph FULL de decode uniforme e paga ~0,3 s fixos: mesmo 128 tokens deixam o incumbente em ~10 tok/s', 'Porque o MNBT não pode passar de 2048', 'Porque o KDA não suporta chunked prefill'], a: 1, e: 'Um teto em tokens não é um orçamento em milissegundos. O custo medido é ~0,3 s + 0,68 ms/token; por isso o fair v5 credita uma fração do tempo e escolhe o maior degrau cuja estimativa cabe em <code>MAX_STEP_MS</code>.' },
    { q: 'Com o custo \\(t(c) = a + b\\,c\\), o que acontece ao reduzir o pedaço de prefill \\(c\\) pela metade?', o: ['TTFT e pausa melhoram', 'A pausa do decode diminui, mas o TTFT do prompt longo piora porque o custo fixo \\(a\\) é pago o dobro de vezes', 'Nada muda: o total de tokens é o mesmo', 'O TTFT melhora e a pausa piora'], a: 1, e: 'TTFT ≈ \\(\\lceil P/c \\rceil a + bP\\): o termo linear não muda, mas o número de custos fixos dobra. A pausa ≈ \\(a + bc\\) cai. É exatamente a troca que o limiar de prefill longo controla.' },
    { q: 'No Qwen, <code>max_model_len</code> = 1M e <code>max_num_seqs</code> = 8, mas o pool comporta ~2,5M tokens. O que acontece se 8 requisições de 1M chegarem?', o: ['OOM e o servidor cai', 'O vLLM recusa todas', 'Cerca de 2 ficam residentes; as outras são preemptadas e refazem o prefill quando voltam', 'O vLLM encurta cada contexto para 312K'], a: 2, e: 'Os tetos não reservam memória. Quando os blocos acabam, o scheduler preempta (modo RECOMPUTE) e a requisição volta à fila; o custo é refazer o prefill. Daí a tabela de perfis contexto × vagas no README.' },
    { q: 'O que o agendamento assíncrono ganha e qual complexidade ele cria?', o: ['Mais KV cache, ao custo de mais memória', 'Esconde o tempo de CPU do scheduler sobrepondo-o ao passo da GPU; exige placeholders (−1) para tokens e rascunhos ainda desconhecidos', 'Menos all-reduces, mas exige TP par', 'Roda prefill e decode em GPUs diferentes'], a: 1, e: 'O passo N+1 é planejado antes de o N terminar; os tokens de N entram como placeholders. No DeepSeek isso vale ~4% single-stream, e as interações com gramáticas e rascunhos (#136, #191) exigiram cuidado; o tutorial do MiMo desliga por um bug de spec decode + async.' },
    { q: 'Na receita GLM·TF v1.3.2, <code>TF_GLM_FILL_SHARE</code> = 0,5 e um prompt longo chega enquanto outra conversa decodifica. Por que quem decodifica pode ficar com bem menos da metade do tempo?', o: ['Porque o chunk de prompt e as linhas de decode dividem o mesmo forward, e o chunk ocupa quase todas as linhas', 'Porque o TensorFold reserva metade da janela de 32 linhas para o prompt', 'Porque o crédito conta iterações, não segundos: metade das iterações são chunks de 1.024 linhas, e cada chunk dura muitas rodadas de decode', 'Porque o decode sai do CUDA graph enquanto há prompt preenchendo'], a: 2, e: 'Rodada, chunk, rodada, chunk… é metade das <i>iterações</i>. Mas, no modelo de brinquedo da figura, um chunk de 1.024 linhas leva ~0,59 s e uma rodada ~70 ms, então o prompt fica com ~89% do tempo. Prefill e decode nunca dividem um forward ali, e a rodada continua no graph da sua contagem de linhas. O <code>fair</code> v5 do vLLM faz o contrário: credita uma fração do <i>tempo</i>. Desde a v1.4, o chunk é fatiado em camadas (~200 ms, patch 0062), com uma rodada entre as fatias.' },
  ],
  mount(root, LA) {
    const V = LA.viz, UI = LA.ui;
    const q = (s) => root.querySelector(s);
    const nf = (x, d = 0) => Number(x).toLocaleString('pt-BR', { minimumFractionDigits: d, maximumFractionDigits: d });
    const secs = (x) => (!isFinite(x) ? '—' : (x >= 100 ? nf(x, 0) : x >= 10 ? nf(x, 1) : nf(x, 2)) + ' s');
    const ink = (C) => (C.dark ? '#06120c' : '#ffffff');
    const clamp = V.clamp;

    /* altura do palco conforme a largura (desktop × celular) */
    function autoH(fig, desk, mob, bp) {
      const st = fig.querySelector('.la-fig-stage');
      const set = () => {
        const want = ((st.clientWidth || 800) < (bp || 600) ? mob : desk) + 'px';
        if (st.style.height !== want) st.style.height = want;
      };
      set();
      const ro = new ResizeObserver(set);
      ro.observe(st);
      if (LA.onCleanup) LA.onCleanup(() => ro.disconnect());
    }
    /* personagem-token */
    function critter(g, x, y, r, col, mood, t) {
      if (!(r > 0.8)) return;
      const C = V.C();
      if (t) y += Math.sin(t * 3 + x * 0.05) * r * 0.07;
      V.dot(g, x, y, r, col);
      const ec = ink(C), er = Math.max(1.1, r * 0.13), ex = r * 0.34, ey = y - r * 0.12;
      g.strokeStyle = ec; g.lineWidth = Math.max(1, r * 0.11);
      if (mood === 'sleep') {
        g.beginPath(); g.moveTo(x - ex - er, ey); g.lineTo(x - ex + er, ey); g.moveTo(x + ex - er, ey); g.lineTo(x + ex + er, ey); g.stroke();
      } else { V.dot(g, x - ex, ey, er, ec); V.dot(g, x + ex, ey, er, ec); }
      g.beginPath();
      if (mood === 'sad') g.arc(x, y + r * 0.62, r * 0.3, 1.2 * Math.PI, 1.8 * Math.PI);
      else if (mood === 'happy') g.arc(x, y + r * 0.1, r * 0.32, 0.18 * Math.PI, 0.82 * Math.PI);
      else { g.moveTo(x - r * 0.22, y + r * 0.38); g.lineTo(x + r * 0.22, y + r * 0.38); }
      g.stroke();
    }
    /* texto com quebra de linha simples (retorna o y da última linha) */
    function wrapText(g, str, x, y, maxW, lh, o) {
      g.font = V.font(o.size || 12, o.font || 'sans', o.weight || 400);
      const words = String(str).split(' ');
      let line = '', yy = y;
      words.forEach((wd) => {
        const test = line ? line + ' ' + wd : wd;
        if (g.measureText(test).width > maxW && line) { V.text(g, line, x, yy, o); line = wd; yy += lh; }
        else line = test;
      });
      if (line) V.text(g, line, x, yy, o);
      return yy;
    }
    function hatch(g, x, y, w, h, col, step) {
      if (w <= 0.5 || h <= 0.5) return;
      step = step || 6;
      g.save(); g.beginPath(); g.rect(x, y, w, h); g.clip();
      g.strokeStyle = col; g.lineWidth = 1; g.beginPath();
      for (let k = -h; k < w; k += step) { g.moveTo(x + k, y + h); g.lineTo(x + k + h, y); }
      g.stroke(); g.restore();
    }
    /* preenche uma grade de células com segmentos [a,b) de tokens (fatias proporcionais) */
    function tokenGrid(g, o) {
      const C = V.C();
      const n = o.cols * o.rows, per = o.total / n, cs = o.cell, gap = o.gap == null ? 2 : o.gap;
      for (let k = 0; k < n; k++) {
        const x0 = o.x + (k % o.cols) * cs, y0 = o.y + Math.floor(k / o.cols) * cs;
        const w0 = cs - gap;
        g.fillStyle = V.alpha(C.muted, 0.12); g.fillRect(x0, y0, w0, w0);
        const lo = k * per, hi = lo + per;
        o.segs.forEach((s) => {
          const a = Math.max(s.a, lo), b = Math.min(s.b, hi);
          if (b > a) { g.fillStyle = s.col; g.fillRect(x0 + ((a - lo) / per) * w0, y0, Math.max(0.8, ((b - a) / per) * w0), w0); }
        });
      }
    }

    /* =====================================================================
       Figura 1 — uma iteração do EngineCore (stepper)
       ===================================================================== */
    {
      const fig = q('#fig-loop');
      autoH(fig, 380, 640, 620);
      const st = UI.stepper(fig, [
        { t: 'Chegada', d: 'O API server tokeniza o pedido e o envia ao EngineCore. A requisição <b>D</b> entra na fila <b>WAITING</b> com \\(c=0\\) e \\(n=3.000\\) (o tamanho do prompt).' },
        { t: 'RUNNING primeiro', d: 'O scheduler percorre RUNNING na ordem de admissão. Os decodes <b>A</b> e <b>B</b> precisam de 1 token cada; o prefill <b>C</b> tem 7.000 tokens pela frente, mas o limiar de prefill longo (2.048, ilustrativo) corta o pedaço.' },
        { t: 'Depois WAITING', d: 'Sobrou orçamento e há vaga: <b>D</b> é admitida. O prefix cache já tem seus primeiros 512 tokens, então \\(c\\) salta para 512; ela recebe os 2.046 tokens que restam na bandeja.' },
        { t: 'execute_model()', d: 'O lote vira uma única matriz achatada \\(X\\) de 4.096 linhas × \\(d\\) (uma linha por token, de quatro requisições). As camadas lineares e o MoE leem cada peso uma vez e multiplicam por \\(X\\) inteira (\\(M = 4.096\\)); só a atenção usa as fronteiras <code>query_start_loc</code> = [0, 1, 2, 2050, 4096] para que cada linha veja apenas o KV da sua requisição. Um único forward processa tudo, com os all-reduces entre os nós.' },
        { t: 'update_from_output()', d: 'Amostragem: A e B ganham um token cada; C e D avançam seus \\(c\\). A atingiu <code>max_tokens</code> e sai, liberando blocos e a vaga. No passo seguinte, tudo recomeça.' },
      ], { autoplay: 6500 });

      function card(g, x, y, w, hh, o) {
        const C = V.C();
        V.box(g, x, y, w, hh, { r: 10, fill: V.alpha(o.col, o.hi ? 0.2 : 0.08), stroke: V.alpha(o.col, o.hi ? 0.95 : 0.4), lw: o.hi ? 2 : 1 });
        critter(g, x + 17, y + hh / 2 - 2, 10, o.col, o.mood || 'neutral', o.t);
        V.text(g, o.title, x + 34, y + 17, { size: 12.5, weight: 600, color: C.ink, maxW: w - 40 });
        V.text(g, o.sub, x + 34, y + 31, { size: 11, font: 'mono', color: C.muted, maxW: w - 40 });
        if (o.frac != null) {
          const bw = w - 42, by = y + hh - 11;
          V.box(g, x + 34, by, bw, 5, { r: 3, fill: V.alpha(C.muted, 0.22) });
          V.box(g, x + 34, by, Math.max(3, bw * clamp(o.frac, 0, 1)), 5, { r: 3, fill: o.col });
        }
        if (o.badge) {
          g.font = V.font(11, 'mono', 600);
          const bw = g.measureText(o.badge).width + 12;
          V.box(g, x + w - bw - 4, y - 8, bw, 17, { r: 8, fill: o.badgeCol || o.col, label: o.badge, size: 11, font: 'mono', color: ink(V.C()) });
        }
      }

      V.fig(fig, (g, w, h, t) => {
        if (w < 60 || h < 60) return;
        const C = V.C();
        const i = st.i, el = st.elapsed();
        const e = V.ease.inOut(Math.min(1, el / 1.2));
        const walk = Math.min(1, el / 2.6);
        const narrow = w < 620;
        const G = {};
        if (!narrow) {
          G.wx = 14; G.wy = 42; G.ww = Math.min(170, w * 0.19); G.rx = G.wx + G.ww + 16; G.rw = Math.min(330, w * 0.37);
          G.ry = 42; G.tx = G.rx + G.rw + 24; G.ty = 42; G.tw = w - G.tx - 14; G.by = h - 44;
        } else {
          G.wx = 10; G.wy = 40; G.ww = Math.max(100, w * 0.31); G.rx = G.wx + G.ww + 8; G.rw = w - G.rx - 10;
          G.ry = 40; G.tx = 10; G.tw = w - 20; G.ty = 40 + 4 * 58 + 40; G.by = h - 46;
        }
        const cell = Math.min(G.tw / 16, narrow ? 22 : 26), th = cell * 8;
        const cardH = 50, step = 58;
        V.text(g, 'WAITING', G.wx, G.wy - 14, { size: 11, font: 'mono', color: C.d, weight: 600 });
        V.text(g, 'RUNNING (ordem)', G.rx, G.ry - 14, { size: 11, font: 'mono', color: C.a, weight: 600, maxW: G.rw });
        V.text(g, 'BANDEJA · B = 4.096 tokens (1 célula = 32)', G.tx, G.ty - 14, { size: 11, font: 'mono', color: C.b, weight: 600, maxW: G.tw });

        const colA = C.a, colB = C.e, colC = C.b, colD = C.c;
        const visit = (k) => (i === 1 ? (walk > k * 0.3 && walk < (k + 1) * 0.3 + 0.08) : false);
        // A
        const aAlpha = i === 4 ? 1 - e * 0.55 : 1;
        const servedA = i >= 1 && (i > 1 || walk > 0.15), servedB = i >= 1 && (i > 1 || walk > 0.45);
        g.globalAlpha = aAlpha;
        card(g, G.rx, G.ry, G.rw, cardH, { col: colA, title: i === 4 ? 'A · terminou' : 'A · decode', sub: i === 4 ? 'c 940 · n 940 · vaga livre' : 'c 939 · n 940', frac: 1, hi: visit(0), t, mood: servedA ? 'happy' : 'neutral', badge: servedA && i !== 4 ? '+1' : null });
        g.globalAlpha = 1;
        if (i === 4) V.box(g, G.rx + G.rw - 62, G.ry - 8, 58, 17, { r: 8, fill: colA, label: '✓ saiu', size: 11, font: 'mono', color: ink(C) });
        card(g, G.rx, G.ry + step, G.rw, cardH, { col: colB, title: 'B · decode', sub: i === 4 ? 'c 2.311 · n 2.312' : 'c 2.310 · n 2.311', frac: 1, hi: visit(1), t, mood: servedB ? 'happy' : 'neutral', badge: servedB ? '+1' : null });
        const cC = i === 4 ? V.lerp(5000, 7048, e) : 5000;
        card(g, G.rx, G.ry + 2 * step, G.rw, cardH, { col: colC, title: 'C · prefill', sub: 'c ' + nf(cC) + ' · n 12.000', frac: cC / 12000, hi: visit(2), t, badge: (i >= 1 && (i > 1 || walk > 0.72)) ? '+2.048' : null });
        // D
        let dx, dy, dcNow = 0, dRun = false;
        const wPos = [G.wx, G.wy], rPos = [G.rx, G.ry + 3 * step];
        if (i === 0) { dx = V.lerp(-G.ww - 20, wPos[0], e); dy = wPos[1]; }
        else if (i === 1) { dx = wPos[0]; dy = wPos[1]; }
        else if (i === 2) { dx = V.lerp(wPos[0], rPos[0], e); dy = V.lerp(wPos[1], rPos[1], e); dcNow = e > 0.6 ? 512 : 0; dRun = e > 0.95; }
        else { dx = rPos[0]; dy = rPos[1]; dcNow = i === 4 ? V.lerp(512, 2558, e) : 512; dRun = true; }
        const dW = V.lerp(G.ww, G.rw, i === 2 ? e : i > 2 ? 1 : 0);
        card(g, dx, dy, dW, cardH, { col: colD, title: 'D · ' + (dRun ? 'prefill' : 'nova'), sub: 'c ' + nf(dcNow) + ' · n 3.000', frac: dcNow / 3000, hi: i === 2 && walk > 0.2, t, mood: i <= 1 ? 'sleep' : 'neutral', badge: i >= 2 && (i > 2 || walk > 0.5) ? '+2.046' : null });
        if (i === 2 && e > 0.55 && e < 1) V.text(g, '✦ cache: 512', dx + dW - 70, dy + cardH + 13, { size: 11, font: 'mono', color: C.d, weight: 600 });
        if (i >= 3) V.text(g, '(vazia)', G.wx + 4, G.wy + 18, { size: 11, font: 'mono', color: C.muted });

        // bandeja
        let fA = 0, fB = 0, fC = 0, fD = 0;
        if (i === 1) { fA = walk > 0.15 ? 1 : 0; fB = walk > 0.45 ? 1 : 0; fC = clamp((walk - 0.6) / 0.4, 0, 1); }
        if (i >= 2) { fA = fB = fC = 1; fD = i === 2 ? clamp((walk - 0.35) / 0.6, 0, 1) : 1; }
        const trayAlpha = i === 4 ? 1 - e * 0.85 : 1;
        g.globalAlpha = trayAlpha;
        tokenGrid(g, {
          x: G.tx, y: G.ty, cols: 16, rows: 8, cell, total: 4096, segs: [
            { a: 0, b: fA, col: colA }, { a: 1, b: 1 + fB, col: colB },
            { a: 2, b: 2 + 2048 * fC, col: colC }, { a: 2050, b: 2050 + 2046 * fD, col: colD },
          ],
        });
        g.globalAlpha = 1;
        if (i >= 1 && i <= 3 && fA) {
          g.strokeStyle = C.c; g.lineWidth = 2; V.rr(g, G.tx - 3, G.ty - 3, cell + 4, cell + 4, 4); g.stroke();
          V.text(g, '↑ 1ª célula: A e B custam 1 token cada (lascas)', G.tx, G.ty + th + 16, { size: 11, color: C.muted, maxW: G.tw });
        }
        // setas cartão → bandeja (fluxo)
        if (!narrow && i === 1 && walk > 0.6) V.arrow(g, G.rx + G.rw + 4, G.ry + 2 * step + cardH / 2, G.tx - 4, G.ty + th * 0.4, { color: V.alpha(colC, 0.8), lw: 1.5, curve: 18 });
        if (!narrow && i === 2 && walk > 0.4) V.arrow(g, G.rx + G.rw + 4, G.ry + 3 * step + cardH / 2, G.tx - 4, G.ty + th * 0.8, { color: V.alpha(colD, 0.8), lw: 1.5, curve: -18 });

        if (i === 4) V.text(g, 'passo concluído: a bandeja zera', G.tx + G.tw / 2, G.ty + th / 2 + 4, { size: 12, weight: 600, color: C.ink, align: 'center', maxW: G.tw });
        // placeholder do lote (antes de execute_model)
        if (i !== 3) {
          g.globalAlpha = i === 4 ? e : 1;
          const gpuW = narrow ? 58 : 78, bx = 14, bw = w - 28 - gpuW - 16, by = G.by, bh = 22;
          g.setLineDash([4, 4]); g.strokeStyle = V.alpha(C.muted, 0.45); g.lineWidth = 1; V.rr(g, bx, by, bw, bh, 4); g.stroke(); g.setLineDash([]);
          V.text(g, 'lote do passo: montado em execute_model()', bx + 8, by + 15, { size: 11, font: 'mono', color: C.muted, maxW: bw - 16 });
          V.box(g, bx + bw + 16, by - 6, gpuW, bh + 12, { r: 8, fill: V.alpha(C.a, 0.05), stroke: V.alpha(C.a, 0.35), lw: 1, label: 'GPU ×2', size: 11.5, font: 'mono', color: C.muted });
          g.globalAlpha = 1;
        }
        // lote achatado → GPU
        if (i >= 3) {
          const alpha = i === 4 ? 1 - e : Math.min(1, el / 0.6);
          g.globalAlpha = Math.max(0, alpha);
          const gpuW = narrow ? 58 : 78, bx = 14, bw = w - 28 - gpuW - 16, by = G.by, bh = 22;
          V.text(g, narrow ? 'X [4.096 × d] → um forward' : 'lote achatado X [4.096 × d] → um forward (cada peso lido 1×)', bx, by - 8, { size: 11, font: 'mono', color: C.muted, maxW: bw });
          const small = 12, rest = bw - 2 * small;
          const segs = [[colA, small, 'A'], [colB, small, 'B'], [colC, rest * 0.5002, narrow ? 'C' : 'C · 2.048'], [colD, rest * 0.4998, narrow ? 'D' : 'D · 2.046']];
          let x = bx;
          const marks = ['0', '1', '2', '2.050', '4.096'];
          segs.forEach((s, k) => {
            V.box(g, x, by, s[1] - 1, bh, { r: 4, fill: s[0], label: s[2], size: 11, color: ink(C) });
            V.text(g, marks[k], x, by + bh + 13, { size: 10.5, font: 'mono', color: C.muted, align: k === 0 ? 'left' : 'center' });
            x += s[1];
          });
          V.text(g, marks[4], x, by + bh + 13, { size: 10.5, font: 'mono', color: C.muted, align: 'right' });
          const gx = bx + bw + 16;
          V.box(g, gx, by - 6, gpuW, bh + 12, { r: 8, fill: V.alpha(C.a, 0.15 + (i === 3 ? 0.15 * Math.sin(t * 6) + 0.15 : 0)), stroke: C.a, lw: 1.5, label: 'GPU ×2', size: 11.5, font: 'mono' });
          if (i === 3) {
            for (let p = 0; p < 16; p++) {
              const u = ((t * 0.6 + p / 16) % 1);
              const sx = bx + (p / 16) * bw, sy = by + bh / 2;
              const px = V.lerp(sx, gx + gpuW / 2, u), py = sy - Math.sin(u * Math.PI) * 26;
              V.dot(g, px, py, 2.2, p < 1 ? colA : p < 2 ? colB : p < 9 ? colC : colD);
            }
            V.glow(g, gx + gpuW / 2, by + bh / 2, 34, C.a, 0.5);
          }
          g.globalAlpha = 1;
        }
      });
    }

    /* =====================================================================
       Figura 2 — batching estático vs contínuo
       ===================================================================== */
    {
      const fig = q('#fig-batching');
      autoH(fig, 380, 480, 560);
      const S = { mode: 'both', v: 0.6 };
      const U = V.rng(11);
      const us = Array.from({ length: 12 }, () => U());
      let P = null;
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Mostrar', options: [{ v: 'both', label: 'comparar' }, { v: 'static', label: 'estático' }, { v: 'cont', label: 'contínuo' }], value: 'both', onChange: (v) => { S.mode = v; } });
      UI.slider(ctl, { label: 'Variabilidade', min: 0, max: 1, step: 0.05, value: S.v, fmt: (v) => nf(v * 100) + '%', onInput: (v) => { S.v = v; build(); } });
      const rU1 = UI.readout(ctl, 'Ocupação c/ fila · estático', '—');
      const rU2 = UI.readout(ctl, 'Ocupação c/ fila · contínuo', '—');
      const rM = UI.readout(ctl, 'Término médio', '—');
      function build() {
        const Ls = us.map((u) => Math.max(3, Math.round(6 + (1 - S.v) * 14 + S.v * (4 + 70 * u * u * u))));
        const stat = [], bubbles = [];
        let T = 0;
        for (let b = 0; b < 3; b++) {
          const ids = [b * 4, b * 4 + 1, b * 4 + 2, b * 4 + 3];
          const mx = Math.max(...ids.map((k) => Ls[k]));
          ids.forEach((k, lane) => { stat.push({ k, lane, s: T, e: T + Ls[k] }); if (Ls[k] < mx) bubbles.push({ lane, s: T + Ls[k], e: T + mx }); });
          T += mx;
        }
        const free = [0, 0, 0, 0], cont = [];
        Ls.forEach((L, k) => { let lane = 0; for (let j = 1; j < 4; j++) if (free[j] < free[lane]) lane = j; cont.push({ k, lane, s: free[lane], e: free[lane] + L }); free[lane] += L; });
        const Tc = Math.max(...free);
        const mean = (arr) => arr.reduce((a, r) => a + r.e, 0) / arr.length;
        // ocupação enquanto ainda há fila (até a última requisição entrar)
        const occ = (arr) => { const Tq = Math.max(...arr.map((r) => r.s)); if (Tq <= 0) return { u: 1, Tq }; const busy = arr.reduce((a, r) => a + Math.max(0, Math.min(r.e, Tq) - r.s), 0); return { u: busy / (4 * Tq), Tq }; };
        const oS = occ(stat), oC = occ(cont);
        P = { stat, bubbles, Ts: T, cont, Tc, uS: oS.u, uC: oC.u, qS: oS.Tq, qC: oC.Tq, mS: mean(stat), mC: mean(cont) };
        rU1.set(nf(P.uS * 100) + '%'); rU2.set(nf(P.uC * 100) + '%'); rM.set(nf(P.mS) + ' → ' + nf(P.mC) + ' passos');
      }
      build();
      V.fig(fig, (g, w, h, t) => {
        if (w < 60 || h < 60) return;
        const C = V.C();
        const narrow = w < 560;
        const panels = S.mode === 'both' ? ['static', 'cont'] : [S.mode];
        const Tmax = S.mode === 'both' ? Math.max(P.Ts, P.Tc) : S.mode === 'static' ? P.Ts : P.Tc;
        const x0 = narrow ? 34 : 64, x1 = w - 14;
        const X = (tt) => x0 + (tt / Tmax) * (x1 - x0);
        const per = 10;
        const ph = Math.min(1, (t % per) / (per - 2.2)) * Tmax;
        const top = 10, bottom = h - 30, gapP = 26;
        const pH = (bottom - top - gapP * (panels.length - 1)) / panels.length;
        panels.forEach((pn, pi) => {
          const y0 = top + pi * (pH + gapP);
          const title = pn === 'static' ? (narrow ? 'Estático (ônibus)' : 'Estático (ônibus): o lote parte cheio e espera o mais lento') : (narrow ? 'Contínuo (esteira)' : 'Contínuo (esteira): vaga livre → próxima requisição entra');
          V.text(g, title, x0, y0 + 12, { size: 12, weight: 600, color: pn === 'static' ? C.d : C.a, maxW: x1 - x0 });
          const ly = y0 + 22, lh = Math.max(8, (pH - 26) / 4);
          for (let l = 0; l < 4; l++) {
            V.text(g, (narrow ? 'v' : 'vaga ') + (l + 1), x0 - 8, ly + l * lh + lh / 2 + 4, { size: 11, font: 'mono', color: C.muted, align: 'right' });
            V.box(g, x0, ly + l * lh + 2, x1 - x0, lh - 4, { r: 5, fill: V.alpha(C.muted, 0.06) });
          }
          const bars = pn === 'static' ? P.stat : P.cont;
          if (pn === 'static') P.bubbles.forEach((b) => {
            if (b.s >= ph) return;
            const xa = X(b.s), xb = X(Math.min(b.e, ph)), yy = ly + b.lane * lh + 3;
            g.fillStyle = V.alpha(C.red || '#FB2C36', 0.07); g.fillRect(xa, yy, xb - xa, lh - 6);
            hatch(g, xa, yy, xb - xa, lh - 6, V.alpha(C.muted, 0.55), 6);
          });
          bars.forEach((b) => {
            if (b.s >= ph) return;
            const xa = X(b.s), xe = X(Math.min(b.e, ph)), yy = ly + b.lane * lh + 3, hh = lh - 6;
            const col = C.series[b.k % 6];
            V.box(g, xa + 0.5, yy, Math.max(2, xe - xa - 1), hh, { r: 4, fill: V.alpha(col, 0.75) });
            if (xe - xa > 18) V.text(g, String(b.k + 1), xa + 6, yy + hh / 2 + 4, { size: 11, font: 'mono', weight: 600, color: ink(C) });
            if (ph < b.e) critter(g, xe, yy + hh / 2, Math.min(9, hh * 0.42), col, 'happy', t);
          });
          const Tq = pn === 'static' ? P.qS : P.qC;
          if (ph >= Tq && Tq > 0) { V.line(g, X(Tq), ly + 4 * lh - 2, X(Tq), ly + 4 * lh + 4, { color: C.c, lw: 2 }); V.text(g, 'fila vazia', X(Tq) + 3, ly + 4 * lh + 4, { size: 10.5, font: 'mono', color: C.c }); }
          const Tp = pn === 'static' ? P.Ts : P.Tc;
          if (ph >= Tp - 0.01) {
            V.line(g, X(Tp), ly - 2, X(Tp), ly + 4 * lh, { color: V.alpha(C.ink, 0.5), dash: [3, 3] });
            V.text(g, 'fim: ' + Tp + ' passos', Math.min(X(Tp) + 4, x1 - 2), ly - 4, { size: 11, font: 'mono', color: C.ink, align: X(Tp) > x1 - 90 ? 'right' : 'left' });
          }
        });
        V.line(g, X(ph), top + 16, X(ph), bottom, { color: V.alpha(C.c, 0.6), lw: 1.2 });
        for (let s = 0; s <= Tmax - 14; s += 20) V.text(g, String(s), X(s), h - 12, { size: 10.5, font: 'mono', color: C.muted, align: 'center' });
        V.text(g, 'passos →', x1, h - 12, { size: 10.5, font: 'mono', color: C.muted, align: 'right' });
      });
    }

    /* =====================================================================
       Figura 4 — a bandeja do passo (verify rows, orçamento, capturas)
       ===================================================================== */
    {
      const fig = q('#fig-tray');
      autoH(fig, 430, 560, 600);
      const PRE = { ds: { S: 6, k: 6, B: 8192, thr: 1024 }, glm: { S: 4, k: 7, B: 7168, thr: 0 }, qwen: { S: 8, k: 3, B: 8192, thr: 0 } };
      const S = { S: 6, k: 6, B: 8192, thr: 1024, pref: true, pad: true, rule: 'ds' };
      let D = null;
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Preset', options: [{ v: 'ds', label: 'DeepSeek' }, { v: 'glm', label: 'GLM' }, { v: 'qwen', label: 'Qwen' }], value: 'ds', onChange: (v) => { const p = PRE[v]; S.rule = v; tPad.el.style.display = v === 'ds' ? '' : 'none'; sS.set(p.S); sK.set(p.k); sB.set(p.B); sT.set(p.thr); } });
      const sS = UI.slider(ctl, { label: 'max_num_seqs', min: 1, max: 32, step: 1, value: S.S, onInput: (v) => { S.S = v; upd(); } });
      const sK = UI.slider(ctl, { label: 'k', min: 0, max: 7, step: 1, value: S.k, onInput: (v) => { S.k = v; upd(); } });
      const sB = UI.slider(ctl, { label: 'MNBT', min: 1024, max: 16384, step: 1024, value: S.B, fmt: (v) => nf(v), onInput: (v) => { S.B = v; upd(); } });
      const sT = UI.slider(ctl, { label: 'Limiar', min: 0, max: 8192, step: 256, value: S.thr, fmt: (v) => (v ? nf(v) : 'off'), onInput: (v) => { S.thr = v; upd(); } });
      UI.toggle(ctl, { label: 'há prefill esperando', value: true, onChange: (v) => { S.pref = v; upd(); } });
      const tPad = UI.toggle(ctl, { label: 'DS: captura em múltiplo de 8', value: true, onChange: (v) => { S.pad = v; upd(); } });
      const rM = UI.readout(ctl, 'Linhas de verificação', '—');
      const rC = UI.readout(ctl, 'Maior graph', '—');
      const rP = UI.readout(ctl, 'Pedaço de prefill', '—');
      /* grade stock do vLLM: 1, 2, 4 e múltiplos de 8 */
      const stockGrid = (cap) => { const o = [1, 2, 4]; for (let x = 8; x <= cap; x += 8) o.push(x); return o.filter((x) => x <= cap); };
      /* tamanhos de captura conforme a regra de cada receita */
      function captures(M, q1) {
        if (S.rule === 'glm') return { sizes: [1, 2, 4, 8, 16, 24, 32], teto: 32, note: 'lista fixa do launcher GLM (k adaptativo off)', short: 'lista fixa GLM' };
        if (S.rule === 'qwen') {
          const teto = Math.min(512, 2 * M);
          return { sizes: stockGrid(teto), teto, note: 'grade stock do vLLM até 2·S·(1+k) = ' + teto + ' · FULL_DECODE_ONLY (Qwen)', short: 'stock até 2·S·(1+k) = ' + teto };
        }
        // DeepSeek: teto = 8⌈S(1+k)/8⌉; grade stock até o teto, arredondada para múltiplos de 1+k
        const capReq = S.pad ? Math.ceil(M / 8) * 8 : M;
        const teto = stockGrid(capReq).pop() || 1;
        const sizes = Array.from(new Set(stockGrid(teto).map((x) => Math.ceil(x / q1) * q1))).filter((x) => x <= teto).sort((a, b) => a - b);
        return { sizes, teto, note: 'múltiplos de 1+k até ' + (S.pad ? '8⌈S(1+k)/8⌉' : 'S(1+k) truncado à grade') + ' = ' + teto + ' (regra da receita DeepSeek)', short: 'regra DS · teto ' + teto };
      }
      function upd() {
        const M = S.S * (S.k + 1);
        const room = S.pref ? Math.max(0, S.B - M) : 0;
        const chunk = S.thr > 0 ? Math.min(S.thr, room) : room;
        const q1 = S.k + 1;
        const cp = captures(M, q1);
        const fit = cp.sizes.filter((x) => x >= M && (q1 === 1 || x % q1 === 0));
        const use = fit.length ? fit[0] : null;
        const capMax = cp.sizes[cp.sizes.length - 1];
        D = { M, chunk, free: Math.max(0, S.B - M - chunk), capT: capMax, sizes: cp.sizes, note: cp.note, short: cp.short, use, full: use != null };
        rM.set(nf(M)); rC.set(nf(capMax) + (D.full ? ' ✓' : ' ✗')); rP.set(nf(chunk));
      }
      upd();
      V.fig(fig, (g, w, h, t) => {
        if (w < 60 || h < 60) return;
        const C = V.C();
        const narrow = w < 600;
        const L = {};
        if (!narrow) { L.mx = 16; L.my = 44; L.mw = w * 0.36; L.mh = h - 150; L.tx = L.mx + L.mw + 34; L.ty = 44; L.tw = w - L.tx - 16; L.th = h - 150; L.sy = h - 56; }
        else { L.mx = 12; L.my = 40; L.mw = w - 24; L.mh = 190; L.tx = 12; L.ty = 40 + 190 + 56; L.tw = w - 24; L.th = 150; L.sy = h - 50; }
        const M = D.M, K1 = S.k + 1;
        // matriz de verificação
        V.text(g, 'matriz de verificação · ' + S.S + ' × ' + K1 + ' = ' + nf(M), L.mx, L.my - 16, { size: 11.5, font: 'mono', color: C.a, weight: 600, maxW: L.mw });
        const cs = Math.max(4, Math.min(24, (L.mw - 34) / K1, L.mh / S.S));
        const withCrit = cs >= 13;
        const ox = L.mx + (withCrit ? 26 : 4);
        for (let r = 0; r < S.S; r++) {
          const yy = L.my + r * cs;
          if (withCrit) critter(g, L.mx + 11, yy + cs / 2, Math.min(8, cs * 0.42), C.series[r % 6], 'neutral', t);
          for (let j = 0; j < K1; j++) {
            const xx = ox + j * cs;
            const col = j === 0 ? C.a : V.alpha(C.b, 0.35 + 0.3 * (0.5 + 0.5 * Math.sin(t * 2.5 - j * 0.8 + r * 0.35)));
            g.fillStyle = col; V.rr(g, xx, yy, cs - 2, cs - 2, Math.min(4, cs / 5)); g.fill();
          }
        }
        const legY = L.my + S.S * cs + 16;
        if (legY < L.my + L.mh + 30) {
          V.dot(g, L.mx + 6, legY - 4, 4, C.a); V.text(g, 'âncora', L.mx + 14, legY, { size: 11, color: C.muted });
          V.dot(g, L.mx + 76, legY - 4, 4, C.b); V.text(g, 'rascunhos a verificar (k = ' + S.k + ')', L.mx + 84, legY, { size: 11, color: C.muted, maxW: L.mw - 84 });
        }
        // bandeja
        const cell = Math.min(L.tw / 16, L.th / 8);
        V.text(g, 'bandeja · MNBT = ' + nf(S.B) + ' (1 célula = ' + nf(S.B / 128) + ')', L.tx, L.ty - 16, { size: 11.5, font: 'mono', color: C.b, weight: 600, maxW: L.tw });
        tokenGrid(g, { x: L.tx, y: L.ty, cols: 16, rows: 8, cell, total: S.B, segs: [{ a: 0, b: M, col: C.a }, { a: M, b: M + D.chunk, col: C.b }] });
        const ly = L.ty + cell * 8 + 16;
        const items = [[C.a, 'decode ' + nf(M)], [C.b, 'prefill ' + nf(D.chunk)], [V.alpha(C.muted, 0.35), 'livre ' + nf(D.free)]];
        let lx = L.tx;
        items.forEach(([c, s]) => { V.box(g, lx, ly - 9, 10, 10, { r: 3, fill: c }); V.text(g, s, lx + 14, ly, { size: 11, font: 'mono', color: C.ink }); g.font = V.font(11, 'mono'); lx += g.measureText(s).width + 30; });
        if (D.chunk > 0) V.text(g, D.chunk < S.B - M ? 'pedaço cortado pelo limiar → passo misto' : 'pedaço = o que sobrou do orçamento → passo misto', L.tx, ly + 16, { size: 11, color: C.muted, maxW: L.tw });
        else V.text(g, 'só decode: passo uniforme (candidato a graph FULL)', L.tx, ly + 16, { size: 11, color: C.muted, maxW: L.tw });
        // tira de capturas (regra da receita escolhida no preset)
        const sx0 = 20, sx1 = w - 20, sy = L.sy;
        const maxV = Math.max(D.capT, M) * 1.06 + 1;
        const SX = (v) => sx0 + (v / maxV) * (sx1 - sx0);
        V.text(g, 'graphs capturados · ' + (narrow ? D.short : D.note), sx0, sy - 18, { size: 11, color: C.muted, maxW: sx1 - sx0 });
        V.line(g, sx0, sy, sx1, sy, { color: V.alpha(C.muted, 0.5) });
        D.sizes.forEach((s) => V.dot(g, SX(s), sy, 3.4, D.use === s ? C.c : C.a));
        V.line(g, SX(D.capT), sy - 8, SX(D.capT), sy + 8, { color: C.d, lw: 2 });
        V.text(g, 'maior ' + D.capT, SX(D.capT), sy + 20, { size: 11, font: 'mono', color: C.d, align: SX(D.capT) > sx1 - 40 ? 'right' : 'center' });
        const mc = D.full ? C.a : (C.red || '#FB2C36');
        g.beginPath(); g.moveTo(SX(M), sy - 6); g.lineTo(SX(M) - 6, sy - 15); g.lineTo(SX(M) + 6, sy - 15); g.closePath(); g.fillStyle = mc; g.fill();
        const lbl = 'passo cheio = ' + M + (D.full ? (D.use === M ? ' · FULL ✓' : ' → graph ' + D.use + ' · FULL ✓') : ' · eager ✗');
        g.font = V.font(11, 'mono', 600); const lw2 = g.measureText(lbl).width / 2 + 2;
        V.text(g, lbl, clamp(SX(M), sx0 + lw2, sx1 - lw2), sy + 36, { size: 11, font: 'mono', color: mc, align: 'center', weight: 600 });
        if (M > 64 && S.rule === 'ds') V.text(g, narrow ? '> 64 linhas: sem prova de estabilidade (#141)' : '> 64 linhas: sem prova de estabilidade no DeepSeek (#141)', sx1, sy - 32, { size: 11, color: C.d, align: 'right' });
      });
    }

    /* =====================================================================
       Figura 5 — fatiador de prompt: TTFT × pausa
       ===================================================================== */
    {
      const fig = q('#fig-chunk');
      autoH(fig, 400, 520, 600);
      const CH = [256, 512, 768, 1024, 1536, 2048, 3584, 4096, 7168, 8192, 16384];
      const S = { ci: 3, P: 32768, a: 0.3 };
      const b = 0.00068;
      const ctl = UI.controls(fig);
      UI.slider(ctl, { label: 'Pedaço', min: 0, max: CH.length - 1, step: 1, value: S.ci, fmt: (v) => nf(CH[v]), onInput: (v) => { S.ci = v; upd(); } });
      UI.seg(ctl, { label: 'Prompt', options: [{ v: 8192, label: '8K' }, { v: 32768, label: '32K' }, { v: 131072, label: '128K' }], value: 32768, onChange: (v) => { S.P = v; upd(); } });
      UI.slider(ctl, { label: 'Custo fixo a', min: 0, max: 0.6, step: 0.05, value: S.a, fmt: (v) => nf(v, 2) + ' s', onInput: (v) => { S.a = v; upd(); } });
      const rN = UI.readout(ctl, 'Pedaços', '—'), rT = UI.readout(ctl, 'TTFT ≈', '—'), rG = UI.readout(ctl, 'Pausa máx. ≈', '—');
      const ttft = (c) => Math.ceil(S.P / c) * S.a + b * S.P;
      const gap = (c) => S.a + b * Math.min(c, S.P);
      function upd() { const c = CH[S.ci]; rN.set(nf(Math.ceil(S.P / c))); rT.set(secs(ttft(c))); rG.set(secs(gap(c))); }
      upd();
      V.fig(fig, (g, w, h, t) => {
        if (w < 60 || h < 60) return;
        const C = V.C();
        const c = CH[S.ci];
        const x0 = 58, x1 = w - 16;
        // linha do tempo
        const T = ttft(c), W = Math.min(T, 24);
        const X = (s) => x0 + (s / W) * (x1 - x0);
        const yD = 34, yP = 64;
        V.text(g, 'decode', x0 - 8, yD + 4, { size: 11, font: 'mono', color: C.a, align: 'right' });
        V.text(g, 'prefill', x0 - 8, yP + 12, { size: 11, font: 'mono', color: C.b, align: 'right' });
        V.text(g, T > W ? 'primeiros ' + nf(W, 0) + ' s de ' + secs(T) : 'prefill completo: ' + secs(T), x1, 16, { size: 11, font: 'mono', color: C.muted, align: 'right' });
        const phase = (t % 7) / 6;
        const reveal = Math.min(1, phase) * W;
        let s = 0, done = 0, k = 0;
        while (s < W && done < S.P && k < 400) {
          const n = Math.min(c, S.P - done), d = S.a + b * n;
          const e = Math.min(s + d, W);
          if (s < reveal) {
            const ev = Math.min(e, reveal);
            V.box(g, X(s) + 0.5, yP, Math.max(1, X(ev) - X(s) - 1.5), 20, { r: 3, fill: V.alpha(C.b, 0.8) });
            if (S.a > 0 && X(s + S.a) - X(s) > 3) { g.fillStyle = V.alpha(C.d, 0.55); g.fillRect(X(s) + 0.5, yP + 16, Math.min(X(ev), X(s + S.a)) - X(s), 4); }
            if (s + d <= reveal) { V.dot(g, X(s + d), yD, 4, C.a); }
          }
          s += d; done += n; k++;
        }
        V.line(g, x0, yD, x1, yD, { color: V.alpha(C.a, 0.25) });
        V.line(g, X(reveal), yD - 12, X(reveal), yP + 26, { color: V.alpha(C.c, 0.7), lw: 1.2 });
        if (reveal < W) critter(g, X(reveal), yD, 8, C.a, gap(c) > 2 ? 'sad' : 'happy', t);
        V.text(g, '● token do usuário · barra âmbar = custo fixo a', x0, yP + 40, { size: 11, color: C.muted, maxW: x1 - x0 });
        // gráfico log-log
        const py0 = w < 560 ? 150 : Math.max(128, h * 0.36), py1 = h - 34;
        const lx0 = Math.log2(256), lx1 = Math.log2(16384);
        const PX = (cc) => x0 + ((Math.log2(cc) - lx0) / (lx1 - lx0)) * (x1 - x0);
        const ly0 = -1, ly1 = 3;
        const PY = (sv) => py1 - ((Math.log10(Math.max(0.1, sv)) - ly0) / (ly1 - ly0)) * (py1 - py0);
        [0.1, 1, 10, 100, 1000].forEach((v) => { V.line(g, x0, PY(v), x1, PY(v), { color: C.grid, lw: 1 }); V.text(g, v >= 1 ? nf(v) + ' s' : '0,1 s', x0 - 6, PY(v) + 4, { size: 10.5, font: 'mono', color: C.muted, align: 'right' }); });
        const narrow = w < 560;
        const lab = narrow ? [256, 1024, 3584, 16384] : [256, 512, 1024, 2048, 3584, 7168, 16384];
        CH.forEach((cc) => { V.line(g, PX(cc), py1, PX(cc), py1 + 4, { color: C.muted }); if (lab.includes(cc)) V.text(g, cc >= 1024 ? nf(cc / 1024, cc % 1024 ? 1 : 0) + 'K' : String(cc), PX(cc), py1 + 16, { size: 10.5, font: 'mono', color: C.muted, align: 'center' }); });
        const curve = (fn, col) => { g.beginPath(); for (let j = 0; j <= 60; j++) { const cc = Math.pow(2, lx0 + (j / 60) * (lx1 - lx0)); const xx = PX(cc), yy = PY(fn(cc)); if (j) g.lineTo(xx, yy); else g.moveTo(xx, yy); } g.strokeStyle = col; g.lineWidth = 2.4; g.stroke(); };
        curve(ttft, C.b); curve(gap, C.a);
        V.line(g, PX(c), py0, PX(c), py1, { color: V.alpha(C.c, 0.7), dash: [4, 4] });
        V.dot(g, PX(c), PY(ttft(c)), 5, C.b); V.dot(g, PX(c), PY(gap(c)), 5, C.a);
        const right = PX(c) > x1 - 150;
        V.text(g, 'TTFT ' + secs(ttft(c)), PX(c) + (right ? -8 : 8), PY(ttft(c)) - 8, { size: 11.5, font: 'mono', weight: 600, color: C.b, align: right ? 'right' : 'left' });
        V.text(g, 'pausa ' + secs(gap(c)), PX(c) + (right ? -8 : 8), PY(gap(c)) + 16, { size: 11.5, font: 'mono', weight: 600, color: C.a, align: right ? 'right' : 'left' });
        V.text(g, 'tamanho do pedaço (tokens, escala log)', x1, h - 2, { size: 10.5, color: C.muted, align: 'right' });
      });
    }

    /* =====================================================================
       Figura 6 — pedaços alinhados a checkpoints do estado recorrente
       ===================================================================== */
    {
      const fig = q('#fig-mamba');
      autoH(fig, 266, 320, 560);
      const Bk = 3584, P = 13000, LAST = P - (P % Bk);
      const S = { pos: 3584, g: 256, mode: 'naive' };
      function align(pos, g, mode) {
        g = Math.min(g, P - pos);
        const after = pos + g;
        if (pos >= LAST) return g;
        if (after > LAST) return LAST - pos;
        if (after === LAST) return g;
        if (mode === 'naive') return Math.floor(g / Bk) * Bk;
        const nb = (Math.floor(pos / Bk) + 1) * Bk;
        if (g < Bk) return Math.min(g, nb - pos);
        return Math.floor(after / Bk) * Bk - pos;
      }
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Regra', options: [{ v: 'naive', label: 'arredondar (bug)' }, { v: 'v2', label: 'v2 (overlay GLM)' }], value: 'naive', onChange: (v) => { S.mode = v; upd(); } });
      UI.seg(ctl, { label: 'Oferta', options: [256, 1024, 2048, 7168].map((v) => ({ v, label: nf(v) })), value: 256, onChange: (v) => { S.g = v; upd(); } });
      UI.slider(ctl, { label: 'Posição', min: 0, max: 12800, step: 128, value: S.pos, fmt: (v) => nf(v), onInput: (v) => { S.pos = v; upd(); } });
      const rO = UI.readout(ctl, 'Concedido', '—'), rK = UI.readout(ctl, 'Checkpoint', '—');
      let R = null;
      function upd() { const gg = align(S.pos, S.g, S.mode); R = { gg, end: S.pos + gg, ck: gg > 0 && (S.pos + gg) % Bk === 0 && S.pos + gg <= LAST }; rO.set(nf(gg) + ' de ' + nf(Math.min(S.g, P - S.pos))); rK.set(R.ck ? 'salvo em ' + nf(R.end) : '—'); }
      upd();
      V.fig(fig, (g, w, h, t) => {
        if (w < 60 || h < 60) return;
        const C = V.C();
        const x0 = 22, x1 = w - 22, X = (v) => x0 + (v / P) * (x1 - x0);
        const y = h * 0.5 - 4, bh = 30;
        V.text(g, 'prompt de 13.000 tokens · bloco = 3.584 (LCM do scheduler)', x0, 22, { size: 11.5, font: 'mono', color: C.muted, maxW: x1 - x0 });
        V.box(g, x0, y, x1 - x0, bh, { r: 6, fill: V.alpha(C.muted, 0.1) });
        hatch(g, X(LAST), y, X(P) - X(LAST), bh, V.alpha(C.muted, 0.4), 7);
        V.box(g, x0, y, Math.max(0, X(S.pos) - x0), bh, { r: 6, fill: V.alpha(C.a, 0.5) });
        g.setLineDash([5, 4]); g.strokeStyle = C.b; g.lineWidth = 1.6; V.rr(g, X(S.pos), y - 6, X(Math.min(P, S.pos + S.g)) - X(S.pos), bh + 12, 6); g.stroke(); g.setLineDash([]);
        if (R.gg > 0) V.box(g, X(S.pos), y, X(R.end) - X(S.pos), bh, { r: 4, fill: V.alpha(C.b, 0.85) });
        const nrw = w < 560;
        [0, Bk, 2 * Bk, 3 * Bk, P].forEach((v) => {
          V.line(g, X(v), y - 4, X(v), y + bh + 4, { color: V.alpha(C.ink, 0.5) });
          V.text(g, nf(v), X(v), y + bh + (nrw && v === P ? 32 : 18), { size: 11, font: 'mono', color: C.muted, align: v === 0 ? 'left' : v === P ? 'right' : 'center' });
        });
        const u = (t % 2.6) / 2.2;
        const arrived = u >= 1;
        [Bk, 2 * Bk, 3 * Bk].forEach((v) => {
          const saved = v <= S.pos, now = R.ck && R.end === v && arrived;
          const cx = X(v), cy = y - 26;
          if (now) V.glow(g, cx, cy, 26, C.d, 0.9);
          V.box(g, cx - 10, cy - 7, 20, 14, { r: 3, fill: saved || now ? C.d : 'transparent', stroke: C.d, lw: 1.4 });
          g.beginPath(); g.arc(cx, cy, 3.6, 0, Math.PI * 2); g.strokeStyle = saved || now ? ink(C) : C.d; g.lineWidth = 1.3; g.stroke();
        });
        V.text(g, 'sem checkpoint', nrw ? X(P) : (X(LAST) + X(P)) / 2, y + bh + (nrw ? 46 : 34), { size: 11, color: C.muted, align: nrw ? 'right' : 'center' });
        let cx, mood = 'neutral', col = C.b;
        if (R.gg === 0) { cx = X(S.pos) + Math.sin(t * 18) * 3; mood = 'sad'; col = C.red || '#FB2C36'; }
        else cx = V.lerp(X(S.pos), X(R.end), V.ease.inOut(Math.min(1, u)));
        critter(g, cx, y + bh / 2, 11, col, R.ck && arrived ? 'happy' : mood, t);
        const msg = R.gg === 0 ? '✗ oferta < 1 bloco arredondada para 0: a requisição não anda neste passo'
          : R.ck ? '✓ o pedaço termina numa fronteira: estado recorrente salvo em ' + nf(R.end)
            : '→ progresso sub-bloco: estado não é salvo agora; o próximo pedaço para na fronteira';
        wrapText(g, msg, x0, nrw ? h - 34 : h - 16, x1 - x0, 17, { size: 12, weight: 600, color: R.gg === 0 ? (C.red || '#FB2C36') : R.ck ? C.d : C.ink });
      });
    }

    /* =====================================================================
       Figura 7 — um passo do scheduler: quem fica sem nada?
       ===================================================================== */
    {
      const fig = q('#fig-starve');
      autoH(fig, 392, 474, 620);
      const SC = {
        1: { run: [{ id: 'A', kind: 'dec' }, { id: 'B', kind: 'dec' }], wait: [{ id: 'P1', P: 32768 }, { id: 'P2', P: 24576 }, { id: 'P3', P: 16384 }, { id: 'P4', P: 8192 }] },
        2: { run: [{ id: 'P1', kind: 'pre', rem: 20000 }, { id: 'P2', kind: 'pre', rem: 12000 }, { id: 'C', kind: 'dec' }, { id: 'D', kind: 'dec' }, { id: 'E', kind: 'dec' }], wait: [{ id: 'P3', P: 16384 }] },
      };
      const K1 = 7, BUD = 8192, SLOTS = 6;
      const S = { sc: 2, thr: 0, cap: 0, floor: false, t0: 0, tNow: 0 };
      let R = null;
      function plan() {
        const sc = SC[S.sc];
        let budget = BUD;
        const vis = [];
        sc.run.forEach((r, i) => {
          const v = { id: r.id, kind: r.kind, rem: r.rem || 0, where: 'run' };
          if (budget <= 0) { v.got = 0; v.skip = r.kind === 'dec'; v.note = r.kind === 'dec' ? '0 · pulado' : '0 · sem orçamento'; vis.push(v); return; }
          let n = r.kind === 'dec' ? K1 : r.rem, cut = false;
          if (r.kind === 'pre' && S.thr > 0 && n > S.thr) n = S.thr;
          if (r.kind === 'pre' && S.floor) {
            const reserve = sc.run.slice(i + 1).filter((x) => x.kind === 'dec').length * K1;
            const lim = Math.max(0, budget - reserve);
            if (n > lim) { n = lim; cut = true; }
          }
          n = Math.min(n, budget);
          v.got = n; budget -= n;
          v.note = n === 0 ? (cut ? '0 · adiado (#43)' : '0') : '+' + nf(n) + (cut ? ' (#43)' : '');
          vis.push(v);
        });
        let inflight = vis.filter((v) => v.kind === 'pre' && v.rem - v.got > 0).length;
        let nRun = sc.run.length, capHit = false;
        sc.wait.forEach((wq) => {
          const v = { id: wq.id, kind: 'new', P: wq.P, where: 'wait', got: 0 };
          if (capHit || (S.cap > 0 && inflight >= S.cap && budget > 0 && nRun < SLOTS)) { capHit = true; v.note = 'fila · teto #27'; v.block = 'cap'; }
          else if (nRun >= SLOTS) { v.note = 'fila · sem vaga'; v.block = 'slot'; }
          else if (budget <= 0) { v.note = 'fila · sem orçamento'; v.block = 'budget'; }
          else {
            let n = wq.P;
            if (S.thr > 0 && n > S.thr) n = S.thr;
            n = Math.min(n, budget);
            v.got = n; budget -= n; nRun++;
            v.note = 'admitida +' + nf(n);
            if (n < wq.P) inflight++;
          }
          vis.push(v);
        });
        const Ptok = vis.filter((v) => v.kind !== 'dec').reduce((a, v) => a + v.got, 0);
        const Dn = vis.filter((v) => v.kind === 'dec' && v.got > 0).length;
        const skips = vis.filter((v) => v.skip).length;
        const dt = 0.075 + 0.003 * Math.max(0, Dn - 1) + (Ptok > 0 ? 0.04 + 0.00055 * Ptok : 0);
        R = { vis, Ptok, Dn, skips, dt, left: budget };
        rS.set(String(skips)); rP.set(nf(Ptok)); rD.set(secs(dt)); rI.set(skips ? '≥ 2 passos' : secs(dt));
        S.t0 = S.tNow;
      }
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Cenário', options: [{ v: 1, label: '1 · rajada fria' }, { v: 2, label: '2 · decodes atrás' }], value: 2, onChange: (v) => { S.sc = v; plan(); } });
      UI.seg(ctl, { label: 'Limiar', options: [{ v: 0, label: '0 (off)' }, { v: 1024, label: '1024' }, { v: 4096, label: '4096' }], value: 0, onChange: (v) => { S.thr = v; plan(); } });
      UI.seg(ctl, { label: 'Teto em voo', options: [{ v: 0, label: 'stock' }, { v: 1, label: '1' }, { v: 2, label: '2' }], value: 0, onChange: (v) => { S.cap = v; plan(); } });
      UI.toggle(ctl, { label: 'piso de decode (#43)', value: false, onChange: (v) => { S.floor = v; plan(); } });
      const rS = UI.readout(ctl, 'Decodes pulados', '—'), rP = UI.readout(ctl, 'Prefill no passo', '—'), rD = UI.readout(ctl, 'Duração do passo', '—'), rI = UI.readout(ctl, 'ITL dos decodes', '—');
      plan();
      V.fig(fig, (g, w, h, t) => {
        if (w < 60 || h < 60) return;
        S.tNow = t;
        const C = V.C();
        const narrow = w < 620;
        const vis = R.vis;
        const tv = 0.55, hold = 3.2, per = vis.length * tv + hold;
        const ph = Math.max(0, t - S.t0) % per;
        const cur = Math.min(vis.length, Math.floor(ph / tv));
        const within = (ph % tv) / tv;
        // bandeja
        const tx = 16, tw = w - 32, ty = 30, tH = 26;
        V.text(g, 'bandeja do passo · 8.192 tokens · k = 6 (7 por decode)', tx, ty - 10, { size: 11, font: 'mono', color: C.muted, maxW: tw });
        V.box(g, tx, ty, tw, tH, { r: 6, fill: V.alpha(C.muted, 0.1), stroke: V.alpha(C.muted, 0.3), lw: 1 });
        let acc = 0;
        vis.forEach((v, k) => {
          if (k > cur || !v.got) return;
          const frac = k === cur ? V.ease.out(within) : 1;
          const wv = Math.max(3, (v.got / BUD) * tw) * frac;
          const col = v.kind === 'dec' ? C.a : v.kind === 'pre' ? C.b : C.c;
          g.fillStyle = col; g.fillRect(tx + acc, ty + 2, Math.max(0, wv - 1), tH - 4);
          if (wv > 40) V.text(g, v.id, tx + acc + 5, ty + tH / 2 + 4, { size: 11, font: 'mono', weight: 600, color: ink(C) });
          acc += wv;
        });
        for (let k = 1; k < 64; k++) { const xx = tx + (k / 64) * tw; g.fillStyle = V.alpha(C.bg || '#000', 0.25); g.fillRect(xx, ty + 2, 1, tH - 4); }
        // cartões
        const runs = vis.filter((v) => v.where === 'run'), waits = vis.filter((v) => v.where === 'wait');
        const cols = narrow ? 3 : Math.max(4, Math.min(6, Math.max(runs.length, waits.length))), gap = 10;
        const cw = Math.min(160, (w - 32 - gap * (cols - 1)) / cols), ch = narrow ? 78 : 84;
        const rowsR = Math.ceil(runs.length / cols), rowsW = Math.ceil(waits.length / cols);
        /* rótulos acima da faixa da “mão” (triângulo em y−14…y−4) para não se sobreporem */
        const yR = ty + tH + 52, yW = yR + rowsR * (ch + gap) + 40;
        V.text(g, 'RUNNING (ordem de admissão) →', 16, yR - 22, { size: 11, font: 'mono', color: C.a, weight: 600 });
        V.text(g, 'WAITING →', 16, yW - 22, { size: 11, font: 'mono', color: C.d, weight: 600 });
        const pos = (arrIdx, isWait) => { const r = Math.floor(arrIdx / cols), c = arrIdx % cols; return [16 + c * (cw + gap), (isWait ? yW : yR) + r * (ch + gap)]; };
        vis.forEach((v, k) => {
          const isW = v.where === 'wait';
          const [x, y] = pos(isW ? waits.indexOf(v) : runs.indexOf(v), isW);
          const visited = k < cur || (k === cur && within > 0.45);
          const active = k === cur && ph < vis.length * tv;
          const baseCol = v.kind === 'dec' ? C.a : v.kind === 'pre' ? C.b : C.c;
          let col = baseCol, mood = 'neutral';
          if (visited) {
            if (v.skip) { col = C.red || '#FB2C36'; mood = 'sad'; }
            else if (v.kind === 'dec') mood = 'happy';
            else if (v.where === 'wait' && !v.got) { col = C.d; mood = 'sleep'; }
          } else if (isW) mood = 'sleep';
          V.box(g, x, y, cw, ch, { r: 10, fill: V.alpha(col, active ? 0.24 : 0.1), stroke: V.alpha(col, active ? 1 : 0.45), lw: active ? 2.2 : 1 });
          critter(g, x + 18, y + 20, 11, col, mood, t);
          V.text(g, v.id, x + 36, y + 18, { size: 13, weight: 700, color: C.ink });
          const sub = v.kind === 'dec' ? 'decode' : v.kind === 'pre' ? 'prefill ' + nf(v.rem / 1000) + 'K' : 'nova ' + nf(v.P / 1024) + 'K';
          V.text(g, sub, x + 36, y + 32, { size: 11, font: 'mono', color: C.muted, maxW: cw - 40 });
          if (visited) {
            const good = v.got > 0;
            const bc = v.skip ? (C.red || '#FB2C36') : good ? baseCol : C.d;
            V.box(g, x + 6, y + ch - 30, cw - 12, 22, { r: 7, fill: V.alpha(bc, good ? 0.85 : 0.22), stroke: bc, lw: 1 });
            V.text(g, v.note, x + cw / 2, y + ch - 15, { size: 11, font: 'mono', weight: 600, color: good ? ink(C) : C.ink, align: 'center', maxW: cw - 16 });
          }
          if (active) {
            g.beginPath(); g.moveTo(x + cw / 2, y - 4); g.lineTo(x + cw / 2 - 7, y - 14); g.lineTo(x + cw / 2 + 7, y - 14); g.closePath(); g.fillStyle = C.c; g.fill();
            if (v.got && within < 0.9) { const p0 = [x + cw / 2, y], p1 = [tx + Math.min(acc, tw) - 4, ty + tH]; const pp = V.along(p0[0], p0[1], p1[0], p1[1], V.ease.inOut(within)); V.dot(g, pp[0], pp[1] - Math.sin(within * Math.PI) * 20, 5, baseCol); }
          }
        });
        const yEnd = yW + rowsW * (ch + gap) + 16;
        if (ph >= vis.length * tv) {
          const msg = R.skips ? '✗ ' + R.skips + ' decode(s) pulado(s) — ficam parados sem preempção · passo ≈ ' + secs(R.dt)
            : '✓ todos os decodes servidos · passo ≈ ' + secs(R.dt) + (R.dt > 2 ? ' (pausa longa para quem conversa)' : '');
          const mc2 = R.skips ? (C.red || '#FB2C36') : R.dt > 2 ? C.d : C.a;
          if (!narrow) V.text(g, msg, 16, Math.min(h - 12, yEnd + 6), { size: 12.5, weight: 600, color: mc2, maxW: w - 32 });
          else {
            V.text(g, R.skips ? '✗ ' + R.skips + ' decode(s) pulado(s), sem preempção' : '✓ todos os decodes servidos', 16, Math.min(h - 30, yEnd + 4), { size: 12.5, weight: 600, color: mc2, maxW: w - 32 });
            V.text(g, 'passo ≈ ' + secs(R.dt) + (R.dt > 2 && !R.skips ? ' · pausa longa para quem conversa' : ''), 16, Math.min(h - 12, yEnd + 22), { size: 12, color: mc2, maxW: w - 32 });
          }
        }
      });
    }

    /* =====================================================================
       Figura 8 — a balança do passo misto (fair v5, modelo de brinquedo)
       ===================================================================== */
    {
      const fig = q('#fig-fair');
      autoH(fig, 430, 560, 600);
      const LAD = [128, 256, 512, 768, 1024, 1536, 2048];
      const S = { pol: 'fair', P: 2000, share: 0.2, step: 1000, t0: 0, tNow: 0 };
      let R = null;
      function sim() {
        const a = 0.3, b = 0.00068, d = 0.162, tokA = 4, MNBT = 7168, AOUT = 4000, BARR = 10;
        const cost = (n) => a + b * n;
        const rung = LAD.filter((r) => cost(r) <= S.step / 1000).pop() || 128;
        let t = 0, Aout = 0, Aend = null, Brem = S.P, Bdone = null, credit = 0, probed = false;
        let aOvTok = 0, aOvT = 0, maxGap = 0, lastA = 0, guard = 0, nMix = 0;
        const steps = [];
        while (guard++ < 30000) {
          const Aact = Aout < AOUT, Bact = t >= BARR && Brem > 0;
          if (!Aact && !Bact) break;
          let n = 0;
          if (Bact) {
            if (!Aact) n = Math.min(Brem, MNBT);
            else if (S.pol === 'off') n = Math.min(Brem, MNBT - 8);
            else if (S.pol === 'cap') n = Math.min(Brem, 128);
            else if (S.pol === 'fair') { if (!probed) { n = Math.min(Brem, rung); probed = true; } else if (credit >= cost(rung)) n = Math.min(Brem, rung); }
          }
          const dt = n > 0 ? cost(n) : d;
          const mixed = n > 0 && Aact;
          if (S.pol === 'fair' && Aact && Bact) credit += S.share * dt;
          if (S.pol === 'fair' && mixed) credit -= dt;
          if (mixed) nMix++;
          const t0 = t; t += dt;
          if (Aact) { Aout += tokA; if (Bact) { aOvTok += tokA; aOvT += dt; maxGap = Math.max(maxGap, t - lastA); } lastA = t; if (Aout >= AOUT) Aend = t; }
          if (n > 0) { Brem -= n; if (Brem <= 0) Bdone = t; }
          steps.push({ t0, t1: t, n, mixed, a: Aact ? tokA : 0, bp: S.P - Math.max(0, Brem) });
        }
        const soloB = Math.ceil(S.P / MNBT) * a + b * S.P;
        R = { steps, Aend, ttft: Bdone - BARR, aOv: aOvT > 0 ? aOvTok / aOvT : 24.7, maxGap: maxGap || d, rung, T: t, soloB, nMix, BARR };
        // taxa de A em janelas de 2 s
        const bins = [];
        const nb = Math.ceil(t / 2);
        for (let k = 0; k < nb; k++) bins.push(0);
        steps.forEach((s) => { if (s.a) { const k = Math.min(nb - 1, Math.floor(s.t1 / 2)); bins[k] += s.a; } });
        R.bins = bins.map((x) => x / 2);
        rT.set(secs(R.ttft)); rA.set(nf(R.aOv, 1) + ' tok/s'); rG.set(secs(R.maxGap)); rR.set(S.pol === 'fair' ? nf(rung) + ' tok' : S.pol === 'cap' ? '128 tok' : S.pol === 'off' ? nf(MNBT - 8) : '—');
        S.t0 = S.tNow;
      }
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Política', options: [{ v: 'skip', label: 'skip' }, { v: 'off', label: '0 / off' }, { v: 'cap', label: 'N = 128' }, { v: 'fair', label: 'fair v5' }], value: 'fair', onChange: (v) => { S.pol = v; sim(); } });
      UI.seg(ctl, { label: 'B', options: [{ v: 2000, label: '2K' }, { v: 30000, label: '30K' }], value: 2000, onChange: (v) => { S.P = v; sim(); } });
      UI.slider(ctl, { label: 'SHARE σ', min: 0.1, max: 0.5, step: 0.05, value: S.share, fmt: (v) => nf(v, 2), onInput: (v) => { S.share = v; sim(); } });
      UI.slider(ctl, { label: 'MAX_STEP_MS', min: 500, max: 3000, step: 100, value: S.step, fmt: (v) => nf(v), onInput: (v) => { S.step = v; sim(); } });
      const rT = UI.readout(ctl, 'TTFT de B', '—'), rA = UI.readout(ctl, 'A na sobreposição', '—'), rG = UI.readout(ctl, 'Maior pausa de A', '—'), rR = UI.readout(ctl, 'Pedaço', '—');
      sim();
      V.fig(fig, (g, w, h, t) => {
        if (w < 60 || h < 60) return;
        S.tNow = t;
        const C = V.C();
        const narrow = w < 600;
        const Tm = Math.min(420, Math.ceil((R.T + 6) / 20) * 20);
        const x0 = narrow ? 40 : 54, x1 = w - 16;
        const X = (s) => x0 + (s / Tm) * (x1 - x0);
        const per = 10, reveal = Math.min(1, (Math.max(0, t - S.t0) % per) / (per - 2.5)) * Tm;
        // balança
        const sAy = 52, cx = w / 2;
        const scoreA = clamp(R.aOv / 24.7, 0, 1);
        const scoreB = clamp(1 - Math.log10(Math.max(1, R.ttft / R.soloB)) / 2.2, 0, 1);
        const ang = clamp((scoreA - scoreB) * 0.4, -0.18, 0.18) * Math.min(1, (t - S.t0) / 1.2);
        const L = Math.min(150, w * 0.3);
        g.save(); g.translate(cx, sAy); g.rotate(-ang);
        V.box(g, -L, -3, 2 * L, 6, { r: 3, fill: V.alpha(C.ink, 0.5) });
        const mood = (s) => (s > 0.75 ? 'happy' : s > 0.4 ? 'neutral' : 'sad');
        critter(g, -L + 16, -16, 12, C.a, mood(scoreA), 0);
        critter(g, L - 16, -16, 12, C.b, mood(scoreB), 0);
        g.restore();
        g.beginPath(); g.moveTo(cx, sAy + 2); g.lineTo(cx - 10, sAy + 18); g.lineTo(cx + 10, sAy + 18); g.closePath(); g.fillStyle = V.alpha(C.muted, 0.7); g.fill();
        V.text(g, 'A · incumbente', cx - L + 16, sAy + 44, { size: 11, color: C.a, weight: 600, align: 'center' });
        V.text(g, 'B · recém-chegado', cx + L - 16, sAy + 44, { size: 11, color: C.b, weight: 600, align: 'center' });
        // pista A: taxa
        const aTop = 128, aBot = narrow ? 268 : 250;
        const AY = (r) => aBot - (r / 30) * (aBot - aTop);
        V.text(g, 'A · tok/s (janelas de 2 s)', x0, aTop - 10, { size: 11, font: 'mono', color: C.a, weight: 600 });
        [0, 10, 20, 30].forEach((v) => { V.line(g, x0, AY(v), x1, AY(v), { color: C.grid }); V.text(g, String(v), x0 - 6, AY(v) + 4, { size: 10.5, font: 'mono', color: C.muted, align: 'right' }); });
        V.line(g, x0, AY(24.7), x1, AY(24.7), { color: V.alpha(C.a, 0.6), dash: [4, 4] });
        V.text(g, 'solo 24,7', x0 + 4, AY(24.7) - 5, { size: 10.5, font: 'mono', color: C.a });
        g.beginPath(); g.moveTo(X(0), AY(0));
        R.bins.forEach((r, k) => { const tt = k * 2 + 1; if (tt > reveal) return; g.lineTo(X(tt), AY(Math.min(30, r))); });
        g.lineTo(X(Math.min(reveal, R.bins.length * 2 - 1)), AY(0)); g.closePath();
        g.fillStyle = V.alpha(C.a, 0.25); g.fill(); g.strokeStyle = C.a; g.lineWidth = 1.6; g.stroke();
        // pista B: progresso
        const bTop = aBot + 36, bBot = h - 30;
        const BY = (v) => bBot - (v / S.P) * (bBot - bTop);
        V.text(g, 'B · tokens de prefill feitos', x0, bTop - 8, { size: 11, font: 'mono', color: C.b, weight: 600 });
        V.text(g, '0', x0 - 6, bBot + 4, { size: 10.5, font: 'mono', color: C.muted, align: 'right' });
        V.text(g, S.P >= 10000 ? '30K' : '2K', x0 - 6, bTop + 8, { size: 10.5, font: 'mono', color: C.muted, align: 'right' });
        V.box(g, x0, bTop, x1 - x0, bBot - bTop, { r: 4, fill: V.alpha(C.muted, 0.06) });
        g.beginPath(); g.moveTo(X(R.BARR), BY(0));
        let lastBp = 0, lastT = R.BARR;
        for (const s of R.steps) { if (s.t1 > reveal) break; if (s.n > 0) { g.lineTo(X(s.t0), BY(lastBp)); g.lineTo(X(s.t1), BY(s.bp)); lastBp = s.bp; lastT = s.t1; } }
        if (reveal > R.BARR) g.lineTo(X(Math.max(lastT, Math.min(reveal, R.T))), BY(lastBp));
        g.strokeStyle = C.b; g.lineWidth = 2; g.stroke();
        R.steps.forEach((s) => { if (s.mixed && s.t1 <= reveal) { g.fillStyle = V.alpha(C.b, 0.8); g.fillRect(X(s.t0), bBot - 5, Math.max(1.2, X(s.t1) - X(s.t0)), 5); } });
        V.line(g, X(R.BARR), bTop, X(R.BARR), bBot, { color: V.alpha(C.c, 0.8), dash: [3, 3] });
        V.text(g, 'B chega', X(R.BARR) + 4, bBot - 14, { size: 10.5, font: 'mono', color: C.c });
        const tb = R.BARR + R.ttft;
        if (reveal >= tb) {
          V.dot(g, X(tb), BY(S.P), 5, C.b);
          /* rótulo com fundo do painel, para não ficar por cima da curva de B */
          const lbB = '1º token de B: ' + secs(R.ttft), rightB = X(tb) > x1 - 170;
          g.font = V.font(11.5, 'mono', 600); const lwB = g.measureText(lbB).width;
          const lxB = rightB ? X(tb) - 8 - lwB : X(tb) + 8, lyB = BY(S.P) + 14;
          V.box(g, lxB - 5, lyB - 13, lwB + 10, 19, { r: 5, fill: V.alpha(C.panel || C.bg, 0.94), stroke: V.alpha(C.b, 0.5), lw: 1 });
          V.text(g, lbB, lxB, lyB, { size: 11.5, font: 'mono', weight: 600, color: C.b, align: 'left' });
        }
        if (R.Aend && reveal >= R.Aend) { V.line(g, X(R.Aend), aTop, X(R.Aend), aBot, { color: V.alpha(C.a, 0.6), dash: [2, 3] }); V.text(g, 'A termina', X(R.Aend) - 4, aBot - 6, { size: 10.5, font: 'mono', color: C.a, align: 'right' }); }
        V.line(g, X(reveal), aTop, X(reveal), bBot, { color: V.alpha(C.c, 0.5) });
        /* ticks sem unidade (a unidade vai só no último), com passo que garante ≥ 40 px entre rótulos */
        const tStep = [10, 20, 50, 100, 200].find((st) => (st / Tm) * (x1 - x0) >= 40) || 200;
        for (let s = 0; s <= Tm + 1e-9; s += tStep) { const last = s + tStep > Tm; V.text(g, last ? s + ' s' : String(s), X(s), h - 12, { size: 10.5, font: 'mono', color: C.muted, align: last && X(s) > x1 - 24 ? 'right' : 'center' }); }
      });
    }

    /* =====================================================================
       Figura 9 — piscina de KV e preempção
       ===================================================================== */
    {
      const fig = q('#fig-preempt');
      autoH(fig, 410, 540, 600);
      /* taxas de prefill medidas em cada receita: [contexto, tok/s] */
      const RDS = [[2048, 2563], [131072, 1665], [900000, 875]];
      const RQW = [[32768, 2808], [65536, 2654], [131072, 2432], [262144, 2171], [400000, 1955], [600000, 1711]];
      const PRE = {
        ds: { pool: 2331430, pts: RDS, lab: 'DeepSeek · pool 2.331.430', src: 'DS, sweep 0731' },
        qb: { pool: 1340476, pts: RQW, lab: 'Qwen v0.30 BF16 · pool 1.340.476', src: 'Qwen, prefill longo TP2' },
        qf: { pool: 3652200, pts: RQW, lab: 'Qwen day-0 FP8 · pool 3.652.200', src: 'Qwen, prefill longo TP2' },
      };
      /* interpolação em log do contexto; fora da faixa, o extremo mais próximo */
      function rateAt(P, L) {
        const p = P.pts;
        if (L <= p[0][0]) return { r: p[0][1], out: L < p[0][0] };
        if (L >= p[p.length - 1][0]) return { r: p[p.length - 1][1], out: L > p[p.length - 1][0] };
        let i = 0; while (p[i + 1][0] < L) i++;
        const u = Math.log(L / p[i][0]) / Math.log(p[i + 1][0] / p[i][0]);
        return { r: p[i][1] + (p[i + 1][1] - p[i][1]) * u, out: false };
      }
      const LS = [50000, 100000, 200000, 262144, 500000, 1000000];
      const S = { p: 'ds', li: 4, N: 6 };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Pool', options: [{ v: 'ds', label: 'DeepSeek' }, { v: 'qb', label: 'Qwen BF16' }, { v: 'qf', label: 'Qwen FP8' }], value: 'ds', onChange: (v) => { S.p = v; upd(); } });
      UI.slider(ctl, { label: 'Contexto', min: 0, max: LS.length - 1, step: 1, value: S.li, fmt: (v) => (LS[v] >= 1e6 ? '1M' : nf(LS[v] / 1000) + 'K'), onInput: (v) => { S.li = v; upd(); } });
      UI.slider(ctl, { label: 'Requisições', min: 1, max: 8, step: 1, value: S.N, onInput: (v) => { S.N = v; upd(); } });
      const rF = UI.readout(ctl, 'Cabem juntas', '—'), rD = UI.readout(ctl, 'Demanda / pool', '—'), rR = UI.readout(ctl, 'Recomputar 1', '—');
      function upd() {
        const P = PRE[S.p], L = LS[S.li];
        rF.set(Math.min(S.N, Math.floor(P.pool / L)) + ' de ' + S.N);
        rD.set(nf((S.N * L) / P.pool, 2) + '×');
        rR.set(secs(L / rateAt(P, L).r));
      }
      upd();
      V.fig(fig, (g, w, h, t) => {
        if (w < 60 || h < 60) return;
        const C = V.C();
        const P = PRE[S.p], L = LS[S.li], N = S.N;
        const narrow = w < 600;
        const f = (t % 9) / 9, gr = V.ease.inOut(Math.min(1, f / 0.55));
        let act = N;
        while (act > 0 && act * gr * L > P.pool) act--;
        const cols = narrow ? 20 : 24, rows = narrow ? 12 : 10, nC = cols * rows, cellTok = P.pool / nC;
        const gx = 16, gy = 44, gw = narrow ? w - 32 : w * 0.6, cs = Math.max(4, Math.min(gw / cols, (narrow ? 240 : h - 190) / rows));
        V.text(g, P.lab + ' tokens · 1 célula = ' + nf(cellTok / 1000, 1) + 'K', gx, gy - 14, { size: 11, font: 'mono', color: C.muted, maxW: w - 32 });
        let k = 0;
        const owner = new Array(nC).fill(-1);
        for (let i = 0; i < act; i++) { const n = Math.max(gr > 0 ? 1 : 0, Math.round((gr * L) / cellTok)); for (let j = 0; j < n && k < nC; j++) owner[k++] = i; }
        for (let c = 0; c < nC; c++) {
          const x = gx + (c % cols) * cs, y = gy + Math.floor(c / cols) * cs;
          g.fillStyle = owner[c] >= 0 ? V.alpha(C.series[owner[c] % 6], 0.82) : V.alpha(C.muted, 0.12);
          V.rr(g, x, y, Math.max(1, cs - 2), Math.max(1, cs - 2), Math.max(0, Math.min(3, cs / 5))); g.fill();
        }
        const used = Math.min(1, (act * gr * L) / P.pool);
        V.text(g, 'ocupação ' + nf(used * 100) + '%', gx, gy + rows * cs + 16, { size: 11.5, font: 'mono', color: used > 0.97 ? C.d : C.ink, weight: 600 });
        const listEnd = narrow ? gy + rows * cs + 34 + Math.ceil(N / 2) * 42 : 0;
        {
          const dy = narrow ? listEnd + 10 : gy + rows * cs + 40, bwMax = (narrow ? w - 32 : cols * cs) - 72, dem = N * L, sc = bwMax / Math.max(P.pool, dem);
          V.text(g, 'pool', gx, dy + 12, { size: 11, font: 'mono', color: C.muted });
          V.box(g, gx + 70, dy, P.pool * sc, 16, { r: 4, fill: V.alpha(C.a, 0.5) });
          V.text(g, 'demanda', gx, dy + 38, { size: 11, font: 'mono', color: C.muted });
          V.box(g, gx + 70, dy + 26, Math.min(dem, P.pool) * sc, 16, { r: 4, fill: V.alpha(C.b, 0.6) });
          if (dem > P.pool) { V.box(g, gx + 70 + P.pool * sc, dy + 26, (dem - P.pool) * sc, 16, { r: 4, fill: V.alpha(C.red || '#FB2C36', 0.25) }); hatch(g, gx + 70 + P.pool * sc, dy + 26, (dem - P.pool) * sc, 16, V.alpha(C.red || '#FB2C36', 0.8), 5); }
          V.text(g, nf(N) + ' × ' + (L >= 1e6 ? '1M' : nf(L / 1000) + 'K') + ' = ' + nf(dem / 1e6, 2) + 'M · ' + nf(dem / P.pool, 2) + '× o pool', gx + 70, dy + 60, { size: 11, font: 'mono', color: dem > P.pool ? (C.red || '#FB2C36') : C.a });
        }
        // lista de requisições
        const lx = narrow ? 16 : gx + cols * cs + 24, ly = narrow ? gy + rows * cs + 34 : gy;
        const lw = narrow ? (w - 40) / 2 : w - lx - 16, lh = narrow ? 36 : Math.min(38, (h - ly - 20) / 8);
        for (let i = 0; i < N; i++) {
          const x = narrow ? lx + (i % 2) * (lw + 8) : lx, y = narrow ? ly + Math.floor(i / 2) * (lh + 6) : ly + i * (lh + 4);
          const kicked = i >= act;
          const gi = P.pool / ((i + 1) * L);
          const flash = kicked && gi <= 1 && gr - gi < 0.1 && gr >= gi;
          const col = kicked ? (C.red || '#FB2C36') : C.series[i % 6];
          V.box(g, x, y, lw, lh - 2, { r: 8, fill: V.alpha(col, flash ? 0.45 + 0.3 * Math.sin(t * 20) : kicked ? 0.1 : 0.14), stroke: V.alpha(col, 0.6), lw: 1 });
          critter(g, x + 14, y + (lh - 2) / 2, 9, col, kicked ? 'sad' : 'happy', t);
          const txt = kicked ? 'R' + (i + 1) + ' ↺ preemptada' : 'R' + (i + 1) + ' · ' + nf((gr * L) / 1000) + 'K';
          V.text(g, txt, x + 28, y + (lh - 2) / 2 + 4, { size: 11.5, font: 'mono', color: C.ink, maxW: lw - 34, weight: kicked ? 600 : 400 });
        }
        const mcol = act < N ? (C.red || '#FB2C36') : C.a;
        const ra = rateAt(P, L), Lk = L >= 1e6 ? '1M' : nf(L / 1000) + 'K';
        const finalFits = N * L <= P.pool;
        if (act < N) {
          V.text(g, (N - act) + ' requisição(ões) voltam à fila; recomputar cada uma ≈ ' + secs(L / ra.r), 16, h - 30, { size: 12, weight: 600, color: mcol, maxW: w - 32 });
          V.text(g, 'taxa a ' + Lk + ': ~' + nf(ra.r) + ' tok/s (' + P.src + (ra.out ? ', extremo medido: otimista' : ', interpolada') + ')', 16, h - 12, { size: 11, color: C.muted, maxW: w - 32 });
        } else if (gr < 1) {
          V.text(g, 'crescendo… ' + (finalFits ? 'a demanda final cabe no pool' : 'a demanda final passa do pool: alguém será preemptado'), 16, h - 14, { size: 12, weight: 600, color: finalFits ? C.muted : C.d, maxW: w - 32 });
        } else V.text(g, 'tudo cabe: a soma de tokens vivos fica abaixo do pool', 16, h - 14, { size: 12, weight: 600, color: mcol, maxW: w - 32 });
      });
    }

    /* =====================================================================
       Figura 10 — agendamento assíncrono
       ===================================================================== */
    {
      const fig = q('#fig-async');
      autoH(fig, 300, 360, 600);
      const S = { c: 3, g: 72, spec: true };
      const ctl = UI.controls(fig);
      UI.slider(ctl, { label: 'CPU por passo', min: 0.5, max: 30, step: 0.5, value: S.c, fmt: (v) => nf(v, 1) + ' ms', onInput: (v) => { S.c = v; upd(); } });
      UI.slider(ctl, { label: 'GPU por passo', min: 20, max: 120, step: 2, value: S.g, fmt: (v) => nf(v) + ' ms', onInput: (v) => { S.g = v; upd(); } });
      UI.toggle(ctl, { label: 'com especulação (k = 6)', value: true, onChange: (v) => { S.spec = v; } });
      const r1 = UI.readout(ctl, 'Síncrono', '—'), r2 = UI.readout(ctl, 'Assíncrono', '—'), r3 = UI.readout(ctl, 'Ganho', '—');
      function upd() { const s = 1000 / (S.c + S.g), a = 1000 / Math.max(S.c, S.g); r1.set(nf(s, 1) + ' passos/s'); r2.set(nf(a, 1) + ' passos/s'); r3.set('+' + nf((a / s - 1) * 100, 1) + '%'); }
      upd();
      const ACC = [4, 2, 5, 3, 4, 2, 5];
      V.fig(fig, (g, w, h, t) => {
        if (w < 60 || h < 60) return;
        const C = V.C();
        const narrow = w < 600;
        const c = S.c, gg = S.g, m = Math.max(c, gg);
        const T = 5 * (c + gg);
        const x0 = narrow ? 44 : 70, x1 = w - 14, X = (s) => x0 + (s / T) * (x1 - x0);
        const per = 6, ph = ((t % per) / per) * T * 1.05;
        const lane = (y, label, col) => { V.text(g, label, x0 - 8, y + 13, { size: 11, font: 'mono', color: col, align: 'right' }); V.box(g, x0, y, x1 - x0, 20, { r: 4, fill: V.alpha(C.muted, 0.06) }); };
        const block = (y, s, e, col, lab) => {
          if (s >= T || s >= ph) return;
          const ee = Math.min(e, T, ph);
          V.box(g, X(s) + 0.5, y + 1, Math.max(1.5, X(ee) - X(s) - 1), 18, { r: 3, fill: V.alpha(col, 0.85) });
          if (X(ee) - X(s) > 24) V.text(g, lab, X(s) + 5, y + 14, { size: 11, font: 'mono', weight: 600, color: ink(C) });
        };
        // síncrono
        const y1 = 26;
        V.text(g, 'síncrono: a GPU espera a CPU (hachurado = GPU ociosa)', x0, y1 - 8, { size: 11.5, weight: 600, color: C.d, maxW: x1 - x0 });
        lane(y1, 'CPU', C.d); lane(y1 + 26, 'GPU', C.a);
        for (let i = 0; i < 6; i++) {
          const s = i * (c + gg);
          block(y1, s, s + c, C.d, 'N+' + i);
          block(y1 + 26, s + c, s + c + gg, C.a, 'N+' + i);
          if (s < ph && s < T) hatch(g, X(s), y1 + 27, Math.max(0, X(Math.min(s + c, ph, T)) - X(s)), 18, V.alpha(C.red || '#FB2C36', 0.7), 4);
        }
        // assíncrono
        const y2 = y1 + 84;
        V.text(g, 'assíncrono: a CPU já planeja o próximo passo', x0, y2 - 8, { size: 11.5, weight: 600, color: C.a });
        lane(y2, 'CPU', C.d); lane(y2 + 26, 'GPU', C.a);
        block(y2, 0, c, C.d, 'N');
        for (let i = 0; i < 8; i++) {
          const s = c + i * m;
          block(y2 + 26, s, s + gg, C.a, 'N+' + i);
          block(y2, s, s + c, C.d, 'N+' + (i + 1));
          if (c > gg && s + gg < ph) hatch(g, X(s + gg), y2 + 27, Math.max(0, X(Math.min(s + m, ph, T)) - X(s + gg)), 18, V.alpha(C.red || '#FB2C36', 0.7), 4);
        }
        V.line(g, X(Math.min(ph, T)), y1 - 2, X(Math.min(ph, T)), y2 + 48, { color: V.alpha(C.c, 0.7), lw: 1.2 });
        // visão do scheduler: placeholders
        const y3 = y2 + 84;
        V.text(g, 'o que o scheduler vê de uma requisição (assíncrono)', x0, y3 - 8, { size: 11.5, weight: 600, color: C.ink, maxW: x1 - x0 });
        const done = Math.max(0, Math.floor((Math.min(ph, T) - c) / m));
        const nCommitted = ACC.slice(0, Math.min(done, ACC.length)).reduce((a, b) => a + (S.spec ? b : 1), 0);
        const pend = S.spec ? 7 : 1;
        const sq = narrow ? 13 : 17, maxSq = Math.floor((x1 - x0) / (sq + 3));
        const showC = Math.min(nCommitted, Math.max(0, maxSq - pend - 1));
        let x = x0;
        for (let i = 0; i < showC; i++) { V.box(g, x, y3, sq, sq, { r: 3, fill: V.alpha(C.a, 0.85) }); x += sq + 3; }
        V.text(g, '│', x + 1, y3 + sq - 3, { size: 12, color: C.muted });
        x += 10;
        for (let i = 0; i < pend && x + sq < x1; i++) {
          g.setLineDash([3, 2]); g.strokeStyle = C.muted; g.lineWidth = 1.2; V.rr(g, x, y3, sq, sq, 3); g.stroke(); g.setLineDash([]);
          if (sq >= 15) V.text(g, '-1', x + sq / 2, y3 + sq / 2 + 4, { size: 10, font: 'mono', color: C.muted, align: 'center' });
          x += sq + 3;
        }
        const note = S.spec ? 'verde: tokens já confirmados · tracejado: 1 âncora + 6 rascunhos ainda desconhecidos (-1); só uma parte será aceita'
          : 'verde: tokens já confirmados · tracejado: o token que o passo em voo ainda vai produzir';
        wrapText(g, note, x0, y3 + sq + 18, x1 - x0, 15, { size: 11, color: C.muted });
      });
    }

    /* =====================================================================
       Figura 3 — escadas de concorrência (dados das receitas)
       ===================================================================== */
    {
      const fig = q('#fig-ladder');
      autoH(fig, 390, 460, 600);
      const DS = {
        q30: { lab: 'Qwen · lane v0.30 · prosa', x: [1, 2, 4, 8, 16], agg: [59.2, 75.1, 118.5, 216.4, 194.9], per: [59.2, 37.7, 29.9, 28.8, 25.5], mns: 8, note: '×16 excede max_num_seqs = 8: fila, TTFT médio 6,5–9,7 s' },
        q0: { lab: 'Qwen · day-0 FP8 · prosa', x: [1, 2, 4, 6, 8], agg: [54.4, 86.5, 128.1, 169.7, 207.0], per: [54.4, 45.1, 34.2, 29.7, 26.7], mns: 8, note: 'agregado 3,8× de ×1 a ×8; por stream cai para 49%' },
        glm: { lab: 'GLM · escada 2026-08-31 · code', x: [1, 2, 4, 8, 12, 16], agg: [41.9, 52.3, 77.9, 93.8, 116.2, 127.2], per: [44.3, 28.8, 21.0, 13.9, 11.9, 10.2], mns: 16, note: '1M, MNBT 2048, 16 vagas, mixed 512 · TTFT mediana 0,4–0,6 s (×1) a 1,3–1,6 s (×16)' },
        ds: { lab: 'DeepSeek · 0731 · prompt 256', x: [1, 2, 4, 6], agg: [69.1, 104.9, 164.5, 191.2], per: [75.4, 58.3, 46.8, 36.9], mns: 6, note: 'TTFT 0,63 → 1,42 s de c=1 a c=6' },
        tf: { lab: 'GLM·TF · sparkDash · prosa', x: [1, 2, 3, 4], agg: [60.4, 79.2, 89.5, 108.8], per: [60.4, 40.4, 30.6, 27.9], mns: 4, mnsLab: 'PARALLEL', note: 'clocks limitados a 2.200 MHz, temp 0 · TTFT 170 → 340 ms · respostas concorrentes = solo, bit a bit' },
        ds32: { lab: 'DeepSeek · N=32, MNBT 12288 (dados)', x: [1, 8, 16, 32], agg: [54, 154, 213, 325], lo: [47, null, null, null], per: null, mns: 32, note: 'c=1 é a faixa medida 47–54 · estabilidade #141 não garantida: dado de desempenho, não recomendação' },
      };
      const S = { d: 'q30', t0: 0, tNow: 0 };
      const ctl = UI.controls(fig);
      UI.select(ctl, { label: 'Receita', options: [{ v: 'q30', label: 'Qwen v0.30 (prosa)' }, { v: 'q0', label: 'Qwen day-0 (prosa)' }, { v: 'glm', label: 'GLM (code)' }, { v: 'tf', label: 'GLM·TF (prosa)' }, { v: 'ds', label: 'DeepSeek 0731' }, { v: 'ds32', label: 'DeepSeek N=32 (dados)' }], value: 'q30', onChange: (v) => { S.d = v; S.t0 = S.tNow; upd(); } });
      const rA = UI.readout(ctl, 'Agregado no pico / ×1', '—'), rP = UI.readout(ctl, 'Por stream no pico / ×1', '—');
      function upd() {
        const d = DS[S.d];
        const iMax = d.agg.indexOf(Math.max(...d.agg));
        const a0 = d.lo && d.lo[0] != null ? null : d.agg[0];
        rA.set(a0 ? nf(d.agg[iMax] / a0, 2) + '× (×' + d.x[iMax] + ')' : nf(d.agg[iMax] / d.agg[0], 1) + '–' + nf(d.agg[iMax] / d.lo[0], 1) + '× (×' + d.x[iMax] + ')');
        rP.set(d.per ? nf((d.per[iMax] / d.per[0]) * 100) + '% (×' + d.x[iMax] + ')' : '—');
      }
      upd();
      const H = V.fig(fig, (g, w, h, t) => {
        if (w < 60 || h < 60) return;
        S.tNow = t;
        const C = V.C();
        const d = DS[S.d], n = d.x.length;
        const x0 = 46, x1 = w - 14, y0 = 30, y1 = h - 64;
        const ymax = Math.ceil((Math.max(...d.agg) * 1.18) / 50) * 50;
        const Y = (v) => y1 - (v / ymax) * (y1 - y0);
        const slot = (x1 - x0) / n, bw = Math.min(56, slot * 0.55);
        const grow = V.ease.out(Math.min(1, (t - S.t0) / 0.9));
        V.text(g, d.lab, x0, 16, { size: 12, weight: 600, color: C.ink, maxW: x1 - x0 });
        for (let v = 0; v <= ymax; v += ymax > 200 ? 100 : 50) { V.line(g, x0, Y(v), x1, Y(v), { color: C.grid }); V.text(g, String(v), x0 - 6, Y(v) + 4, { size: 10.5, font: 'mono', color: C.muted, align: 'right' }); }
        const cxs = d.x.map((_, i) => x0 + slot * (i + 0.5));
        const hov = H.pointer.inside ? Math.floor((H.pointer.x - x0) / slot) : -1;
        d.agg.forEach((v, i) => {
          const bh = (y1 - Y(v)) * grow;
          const lo = d.lo && d.lo[i] != null ? d.lo[i] : null;
          if (lo != null) {
            /* faixa medida: sólido até o mínimo, faixa clara até o máximo */
            const bl = (y1 - Y(lo)) * grow;
            V.box(g, cxs[i] - bw / 2, y1 - bl, bw, bl, { r: 5, fill: V.alpha(C.a, hov === i ? 0.95 : 0.7) });
            V.box(g, cxs[i] - bw / 2, y1 - bh, bw, bh - bl, { r: 3, fill: V.alpha(C.a, 0.25), stroke: V.alpha(C.a, 0.7), lw: 1 });
            V.text(g, nf(lo) + '–' + nf(v), cxs[i], y1 - bh - 6, { size: 11, font: 'mono', color: C.ink, align: 'center' });
          } else {
            V.box(g, cxs[i] - bw / 2, y1 - bh, bw, bh, { r: 5, fill: V.alpha(C.a, hov === i ? 0.95 : 0.7) });
            /* se o personagem da linha por stream cai no topo da barra, o rótulo sobe acima dele */
            const yPer = d.per ? y1 - (y1 - Y(d.per[i])) * grow : Infinity;
            const yLab = Math.abs(yPer - (y1 - bh)) < 16 ? Math.min(y1 - bh, yPer - 8) - 6 : y1 - bh - 6;
            V.text(g, nf(v, v % 1 ? 1 : 0), cxs[i], yLab, { size: 11, font: 'mono', color: C.ink, align: 'center' });
          }
          V.text(g, '×' + d.x[i], cxs[i], y1 + 16, { size: 11, font: 'mono', color: C.muted, align: 'center' });
        });
        if (d.per) {
          g.beginPath();
          d.per.forEach((v, i) => { const yy = y1 - (y1 - Y(v)) * grow; if (i) g.lineTo(cxs[i], yy); else g.moveTo(cxs[i], yy); });
          g.strokeStyle = C.b; g.lineWidth = 2.4; g.stroke();
          d.per.forEach((v, i) => { const yy = y1 - (y1 - Y(v)) * grow; critter(g, cxs[i], yy, 7, C.b, i === 0 ? 'happy' : v / d.per[0] > 0.5 ? 'neutral' : 'sad', t); });
        }
        const im = d.x.indexOf(d.mns);
        {
          /* entre o degrau = max_num_seqs e o seguinte; na borda direita se todos os degraus cabem nas vagas */
          const xm = im >= 0 && im < n - 1 ? (cxs[im] + cxs[im + 1]) / 2 : x1 - 2;
          V.line(g, xm, y0, xm, y1, { color: C.d, dash: [5, 4], lw: 1.5 });
          const ml = (d.mnsLab || 'max_num_seqs') + ' = ' + d.mns; g.font = V.font(11, 'mono'); const mw = g.measureText(ml).width;
          V.text(g, ml, xm + 4 + mw > x1 ? xm - 4 : xm + 4, y0 + 10, { size: 11, font: 'mono', color: C.d, align: xm + 4 + mw > x1 ? 'right' : 'left' });
        }
        V.box(g, x0, h - 44, 10, 10, { r: 3, fill: V.alpha(C.a, 0.75) }); V.text(g, 'agregado', x0 + 14, h - 35, { size: 11, color: C.muted });
        if (d.per) { V.line(g, x0 + 84, h - 39, x0 + 104, h - 39, { color: C.b, lw: 2.4 }); V.text(g, 'por stream', x0 + 108, h - 35, { size: 11, color: C.muted }); }
        V.text(g, d.note, x0, h - 14, { size: 11, color: C.muted, maxW: x1 - x0 });
        if (hov >= 0 && hov < n) {
          const loH = d.lo && d.lo[hov] != null ? d.lo[hov] : null, tW = loH != null ? 196 : 146;
          const tx = clamp(cxs[hov] + 12, x0, x1 - tW - 4), ty = y0 + 18;
          V.box(g, tx, ty, tW, d.per ? 46 : 30, { r: 8, fill: C.panel, stroke: V.alpha(C.a, 0.6) });
          V.text(g, '×' + d.x[hov] + ' · agregado ' + (loH != null ? nf(loH) + '–' + nf(d.agg[hov]) + ' (faixa)' : nf(d.agg[hov], 1)), tx + 8, ty + 18, { size: 11, font: 'mono', color: C.ink, maxW: tW - 14 });
          if (d.per) V.text(g, 'por stream ' + nf(d.per[hov], 1) + ' tok/s', tx + 8, ty + 36, { size: 11, font: 'mono', color: C.b });
        }
      });
    }
    /* =====================================================================
       Figura 11 — TensorFold: o metrônomo do rank 0 (chunk ou rodada)
       ===================================================================== */
    const tfCache = { k: null, v: null };
    const TFC = (C) => { if (tfCache.k !== C) { tfCache.k = C; tfCache.v = getComputedStyle(document.documentElement).getPropertyValue('--tf').trim() || C.d; } return tfCache.v; };
    {
      const fig = q('#fig-tf-metronomo');
      autoH(fig, 420, 640, 700);
      const TX = {
        rule: 'rank 0, a cada iteração com disputa: C += φ · se C ≥ 1 → chunk de prompt (C −= 1) · senão → rodada de decode',
        credit: 'crédito C', calm: 'sem disputa: nada a decidir', last: 'iterações recentes',
        A: 'A · decodificando', B: (p) => 'B · prompt de ' + p + ' tokens', waitB: 'ainda não chegou', filling: (a, b) => 'preenchendo ' + a + ' / ' + b,
        waited: (s) => 'esperando há ' + s, done: (s) => '1º token! TTFT ' + s,
        rate: (v) => '≈ ' + v + ' tok/s agora', toks: (n) => n + ' tokens até aqui',
        zoom: (s) => 'lupa: os últimos ' + s + ' do modelo', all: 'a corrida inteira', arr: 'B chega', ft: '1º token de B', now: 'agora',
        R: 'R', P: 'P', legR: 'R · rodada de decode (A; depois A e B)', legP: 'P · chunk de prompt de B',
        note: 'modelo de brinquedo: chunk = 0,12 s + 0,45 ms por linha · rodada = 70 ms (ilustrativa) · A sozinha = 60,4 tok/s (medido)',
        phi: 'φ (TF_GLM_FILL_SHARE)', rows: 'chunk com decode', prompt: 'prompt de B', again: '▶ B chega de novo',
        rT: 'TTFT de B (modelo)', rA: 'A durante o prefill', rS: 'tempo para o prompt',
        vT: (v, s) => v + ' (sozinho: ' + s + ')', vA: (v) => v + ' tok/s (sozinha: 60,4)', vS: (a, b) => a + '% do tempo · ' + b + '% das iterações',
      };
      const TR = 0.07, TOKR = 60.4 * TR, FIX = 0.12, BPR = (2048 / 1950 - FIX) / 2048, PRE = 1.2, POST = 1.6;
      const tc = (n) => FIX + n * BPR;
      const S = { phi: 0.5, rows: 1024, P: 32768, t0: 0, tNow: 0, k: 0, sim: null };
      const ctl = UI.controls(fig);
      UI.slider(ctl, { label: TX.phi, min: 0.1, max: 1, step: 0.05, value: S.phi, fmt: (v) => nf(v, 2), onInput: (v) => { S.phi = Math.round(v * 100) / 100; build(); restart(); } });
      UI.seg(ctl, { label: TX.rows, options: [512, 1024, 2048].map((v) => ({ v, label: nf(v) })), value: 1024, onChange: (v) => { S.rows = v; build(); restart(); } });
      UI.seg(ctl, { label: TX.prompt, options: [{ v: 8192, label: '8K' }, { v: 32768, label: '32K' }, { v: 65536, label: '64K' }], value: 32768, onChange: (v) => { S.P = v; build(); restart(); } });
      UI.button(ctl, TX.again, () => restart());
      const rT = UI.readout(ctl, TX.rT, '—'), rA = UI.readout(ctl, TX.rA, '—'), rS = UI.readout(ctl, TX.rS, '—');
      function restart() { S.t0 = S.tNow; S.k = 0; }
      /* a mesma regra do código (multi.py, _fill_turn): crédito em iterações, em ponto flutuante */
      function build() {
        const ev = [];
        let t = 0;
        for (let i = 0; i < Math.round(PRE / TR); i++) { ev.push({ k: 'R', t0: t, t1: t + TR, two: false, cIn: 0, cOut: 0, calm: true }); t += TR; }
        const tArr = t;
        let c = 0, pos = 0, nR = 0, nF = 0, tokA = 0, tF = 0;
        while (pos < S.P) {
          c += S.phi;
          const cIn = c;
          if (c >= 1) {
            c -= 1;
            const n = Math.min(S.rows, S.P - pos), d = tc(n);
            ev.push({ k: 'P', t0: t, t1: t + d, n, from: pos, cIn, cOut: c });
            t += d; tF += d; pos += n; nF++;
          } else {
            ev.push({ k: 'R', t0: t, t1: t + TR, two: false, cIn, cOut: c });
            t += TR; nR++; tokA += TOKR;
          }
        }
        const tFirst = t;
        for (let i = 0; i < Math.round(POST / TR); i++) { ev.push({ k: 'R', t0: t, t1: t + TR, two: true, cIn: 0, cOut: 0, calm: true }); t += TR; }
        let solo = 0;
        for (let p = 0; p < S.P; p += 2048) solo += tc(Math.min(2048, S.P - p));
        const ttft = tFirst - tArr;
        S.sim = { ev, tArr, tFirst, tEnd: t, ttft, solo, aRate: tokA / ttft, fT: tF / ttft, fI: nF / (nF + nR), speed: clamp(ttft / 9, 1.4, 3.5) };
        S.k = 0;
        rT.set(TX.vT(secs(ttft), secs(solo)), ttft > solo * 1.5 ? 'warn' : 'ok');
        rA.set(TX.vA(nf(S.sim.aRate, 1)), S.sim.aRate < 20 ? 'warn' : 'ok');
        rS.set(TX.vS(nf(S.sim.fT * 100), nf(S.sim.fI * 100)));
      }
      build();

      V.fig(fig, (g, w, h, t) => {
        if (w < 60 || h < 60) return;
        S.tNow = t;
        const C = V.C(), cTF = TFC(C), sim = S.sim, ev = sim.ev;
        const narrow = w < 700;
        let tau = (t - S.t0) * sim.speed;
        if (tau > sim.tEnd + 1.8 * sim.speed) { S.t0 = t; S.k = 0; tau = 0; }
        tau = Math.min(tau, sim.tEnd - 1e-6);
        let k = S.k;
        if (k >= ev.length || ev[k].t0 > tau) k = 0;
        while (k < ev.length - 1 && ev[k].t1 <= tau) k++;
        S.k = k;
        const e = ev[k], prog = clamp((tau - e.t0) / (e.t1 - e.t0), 0, 1);
        const phase = tau < sim.tArr ? 0 : tau < sim.tFirst ? 1 : 2;
        const L = 14, R = w - 14, W = R - L;
        /* acumulados até agora: tokens de A, posição do prompt de B */
        let totA = 0, posB = 0, tokWin = 0;
        const winM = 1.5;
        for (let j = 0; j <= k; j++) {
          const r = ev[j], f = j < k ? 1 : prog;
          if (r.k === 'R') {
            totA += TOKR * f;
            const a = Math.max(r.t0, tau - winM), b = Math.min(r.t1, tau);
            if (b > a) tokWin += (TOKR * (b - a)) / TR;
          } else posB = r.from + r.n * f;
        }
        if (phase === 2) posB = S.P;
        const rateA = tokWin / Math.min(winM, Math.max(tau, 0.3));

        let y = 18;
        y += V.wrap(g, TX.rule, L, y, W, { size: 11.5, font: 'mono', color: C.muted, lineH: 15 }) + 6;
        /* medidor do crédito */
        const gW = narrow ? W : Math.min(320, W * 0.36);
        V.text(g, TX.credit, L, y + 4, { size: 11.5, weight: 600, color: C.ink });
        const gy = y + 14, gh = 16, gx = L, gw = gW - 64;
        let lvl = 0;
        if (phase === 1) lvl = e.k === 'P' ? (prog < 0.3 ? e.cIn : V.lerp(e.cIn, e.cOut, V.ease.inOut(clamp((prog - 0.3) / 0.5, 0, 1)))) : e.cIn;
        const X = (v) => gx + (v / 1.25) * gw;
        V.box(g, gx, gy, gw, gh, { r: 6, fill: V.alpha(C.muted, 0.12), stroke: V.alpha(C.muted, 0.4), lw: 1 });
        if (lvl > 0.001) V.box(g, gx, gy, Math.max(6, X(Math.min(lvl, 1.25)) - gx), gh, { r: 6, fill: lvl >= 1 - 1e-9 ? cTF : V.alpha(C.b, 0.85) });
        V.line(g, X(1), gy - 3, X(1), gy + gh + 3, { color: C.ink, lw: 1.5, dash: [3, 2] });
        V.text(g, '1', X(1), gy - 6, { size: 11, font: 'mono', color: C.ink, align: 'center' });
        V.text(g, phase === 1 ? 'C = ' + nf(lvl, 2) : 'C = —', gx + gw + 8, gy + 12, { size: 11.5, font: 'mono', weight: 600, color: C.ink });
        if (phase !== 1) V.text(g, TX.calm, gx, gy + gh + 15, { size: 11, color: C.muted, maxW: gW });
        /* iterações recentes */
        const cx0 = narrow ? L : L + gW + 26, cy0 = narrow ? gy + gh + 26 : y - 4;
        V.text(g, TX.last, cx0, cy0 + 8, { size: 11.5, weight: 600, color: C.ink });
        const chipW = 22, chipGap = 4, nChip = Math.max(4, Math.floor((R - cx0 + chipGap) / (chipW + chipGap)));
        const recent = [];
        for (let j = k; j >= 0 && recent.length < nChip; j--) if (!ev[j].calm) recent.unshift(ev[j]);
        recent.forEach((r, j) => {
          const xx = cx0 + j * (chipW + chipGap), yy = cy0 + 16, cur = r === e;
          const col = r.k === 'P' ? cTF : C.a;
          V.box(g, xx, yy, chipW, 20, { r: 5, fill: V.alpha(col, cur ? 1 : 0.55), stroke: cur ? C.ink : null, lw: 1.5 });
          V.text(g, r.k === 'P' ? TX.P : TX.R, xx + chipW / 2, yy + 14.5, { size: 11, font: 'mono', weight: 700, color: ink(C), align: 'center' });
        });
        y = (narrow ? cy0 + 46 : gy + gh + 22) + 8;

        /* personagens: A (decodificando) e B (prompt que chega) */
        const cardH = 82, gap = 12;
        const wA = narrow ? W : Math.min(360, (W - gap) / 2), wB = narrow ? W : Math.min(440, W - wA - gap);
        const xA = L, xB = narrow ? L : L + wA + gap, yA = y;
        if (narrow) y += cardH + 10;
        V.box(g, xA, yA, wA, cardH, { r: 12, fill: V.alpha(C.a, 0.08), stroke: V.alpha(C.a, 0.5), lw: 1.2 });
        const moodA = rateA > 40 ? 'happy' : rateA > 15 ? 'neutral' : 'sad';
        critter(g, xA + 24, yA + 30, 15, C.a, moodA, t);
        V.text(g, TX.A, xA + 46, yA + 22, { size: 12.5, weight: 600, color: C.ink, maxW: wA - 52 });
        V.text(g, TX.rate(nf(rateA, 1)), xA + 46, yA + 42, { size: 12, font: 'mono', weight: 600, color: moodA === 'sad' ? C.red : C.a, maxW: wA - 52 });
        V.text(g, TX.toks(nf(totA)), xA + 46, yA + 60, { size: 11, font: 'mono', color: C.muted, maxW: wA - 52 });
        g.globalAlpha = phase === 0 ? 0.5 : 1;
        V.box(g, xB, y, wB, cardH, { r: 12, fill: V.alpha(cTF, 0.08), stroke: V.alpha(cTF, 0.55), lw: 1.2 });
        critter(g, xB + 24, y + 30, 15, cTF, phase === 0 ? 'sleep' : phase === 2 ? 'happy' : 'neutral', t);
        V.text(g, TX.B(nf(S.P)), xB + 46, y + 22, { size: 12.5, weight: 600, color: C.ink, maxW: wB - 52 });
        if (phase === 0) V.text(g, TX.waitB, xB + 46, y + 42, { size: 11.5, color: C.muted, maxW: wB - 52 });
        else {
          const bw = wB - 58, by = y + 32;
          V.box(g, xB + 46, by, bw, 8, { r: 4, fill: V.alpha(C.muted, 0.2) });
          V.box(g, xB + 46, by, Math.max(4, bw * (posB / S.P)), 8, { r: 4, fill: cTF });
          V.text(g, phase === 1 ? TX.filling(nf(posB), nf(S.P)) : TX.done(secs(sim.ttft)), xB + 46, y + 56, { size: 11, font: 'mono', weight: phase === 2 ? 700 : 400, color: phase === 2 ? C.ink : C.muted, maxW: bw });
          if (phase === 1) V.text(g, TX.waited(secs(tau - sim.tArr)), xB + 46, y + 70, { size: 11, font: 'mono', color: C.muted, maxW: bw });
        }
        g.globalAlpha = 1;
        y += cardH + 24;

        /* lupa: os últimos segundos do modelo */
        const Wm = 2.5 * sim.speed;
        V.text(g, TX.zoom(secs(Wm)), L, y, { size: 11.5, weight: 600, color: C.ink, maxW: W - 60 });
        V.text(g, TX.now + ' ▸', R, y, { size: 11, color: C.muted, align: 'right' });
        const zy = y + 8, zh = 44;
        const Z = (tt) => L + ((tt - (tau - Wm)) / Wm) * W;
        V.box(g, L, zy, W, zh, { r: 8, fill: V.alpha(C.muted, 0.07) });
        g.save(); V.rr(g, L, zy, W, zh, 8); g.clip();
        for (let j = k; j >= 0; j--) {
          const r = ev[j];
          if (r.t1 < tau - Wm) break;
          const x0 = Z(r.t0), x1 = Z(Math.min(r.t1, tau));
          if (x1 - x0 <= 0) continue;
          if (r.k === 'P') {
            V.box(g, x0 + 1, zy + 4, Math.max(1, x1 - x0 - 2), zh - 8, { r: 4, fill: V.alpha(cTF, 0.85) });
            if (x1 - x0 > 50) V.text(g, TX.P + ' ' + nf(r.n), Math.max(x0 + 26, x1 - 26), zy + zh / 2 + 4, { size: 11, font: 'mono', weight: 700, color: ink(C), align: 'center' });
          } else if (r.two) {
            V.box(g, x0 + 0.5, zy + 4, Math.max(1, x1 - x0 - 1), (zh - 8) / 2 - 1, { r: 2, fill: V.alpha(C.a, 0.85) });
            V.box(g, x0 + 0.5, zy + zh / 2 + 1, Math.max(1, x1 - x0 - 1), (zh - 8) / 2 - 1, { r: 2, fill: V.alpha(cTF, 0.55) });
          } else V.box(g, x0 + 0.5, zy + 4, Math.max(1, x1 - x0 - 1), zh - 8, { r: 2, fill: V.alpha(C.a, 0.85) });
        }
        g.restore();
        [[sim.tArr, TX.arr], [sim.tFirst, TX.ft]].forEach(([tm, lab]) => {
          if (tm > tau - Wm && tm <= tau) {
            const xx = Z(tm);
            V.line(g, xx, zy - 2, xx, zy + zh + 2, { color: C.ink, lw: 1.5, dash: [4, 3] });
            g.font = V.font(11);
            const tw = g.measureText(lab).width;
            V.text(g, lab, xx + 4 + tw > R ? xx - 4 : xx + 4, zy + zh + 14, { size: 11, color: C.ink, align: xx + 4 + tw > R ? 'right' : 'left' });
          }
        });

        /* a corrida inteira */
        y = zy + zh + 36;
        V.text(g, TX.all, L, y, { size: 11.5, weight: 600, color: C.ink });
        const oy = y + 8, oh = 18;
        const O = (tt) => L + (tt / sim.tEnd) * W;
        V.box(g, L, oy, W, oh, { r: 5, fill: V.alpha(C.muted, 0.08) });
        for (let j = 0; j < ev.length; j++) {
          const r = ev[j], x0 = O(r.t0), x1 = O(r.t1), past = r.t1 <= tau;
          g.fillStyle = r.k === 'P' ? V.alpha(cTF, past ? 0.9 : 0.3) : V.alpha(C.a, past ? 0.8 : 0.25);
          g.fillRect(x0, oy + 2, Math.max(0.6, x1 - x0 - (x1 - x0 > 3 ? 0.6 : 0)), oh - 4);
        }
        const px = O(tau);
        V.line(g, px, oy - 4, px, oy + oh + 4, { color: C.c, lw: 2.5 });
        [[sim.tArr, TX.arr, 'left'], [sim.tFirst, TX.ft, 'right']].forEach(([tm, lab, al]) => {
          const xx = O(tm);
          V.line(g, xx, oy + oh, xx, oy + oh + 7, { color: C.ink, lw: 1.2 });
          V.text(g, lab, xx, oy + oh + 19, { size: 11, color: C.muted, align: al });
        });

        /* legenda e nota */
        y = oy + oh + 40;
        const leg = [[C.a, TX.legR], [cTF, TX.legP]];
        let lx = L;
        leg.forEach(([col, lab]) => {
          g.font = V.font(11);
          const lw = 16 + g.measureText(lab).width + 18;
          if (lx > L && lx + lw > R) { lx = L; y += 18; }
          V.box(g, lx, y - 9, 11, 11, { r: 3, fill: col });
          V.text(g, lab, lx + 16, y, { size: 11, color: C.muted });
          lx += lw;
        });
        V.wrap(g, TX.note, L, y + 20, W, { size: 11, color: C.muted, lineH: 14 });
      });
    }

    /* =====================================================================
       Figura 12 — TensorFold: a bandeja (janela em lote e chunk multi-prompt)
       ===================================================================== */
    {
      const fig = q('#fig-tf-bandeja');
      const TX = {
        mode: 'Bandeja', mRound: 'rodada de decode', mChunk: 'chunk de prompt',
        streams: 'streams decodificando', copy: 'A cita um trecho (15 copy drafts)', dice: '🎲 nova rodada',
        rP: 'linhas propostas', rV: 'linhas verificadas', rK: 'tokens confirmados',
        scen: 'Cenário', sFour: '4 prompts de 130 juntos', sLong: 'um de 4.000 na frente de 3 de 130', again: '▶ de novo',
        rL: 'último 1º token', rM: 'média até o 1º token',
        vL: (f, c, r) => 'após ' + f + ' forwards (' + c + ' P + ' + r + ' R)', vM: (v) => v + ' forwards',
        ph: [
          '1 · cada stream propõe a sua janela: token pendente + rascunhos (até 16 linhas)',
          '2 · a soma passa de 32: sai um rascunho da janela mais longa, um de cada vez',
          '3 · um forward só para todos, pelo CUDA graph de ',
          '4 · cada stream aceita enquanto o rascunho = a sua amostra com chave',
        ],
        noCut: '2 · a soma cabe em 32 linhas: nada a cortar',
        lines: ' linhas', props: 'propostas por stream', win: 'janela em lote (MAX_WINDOW = 32)', graph: (n) => 'graph[' + n + ']',
        legPend: 'pendente', legCopy: 'copy draft', legDf: 'DFlash2', legOk: 'aceito', legBad: 'difere: para aqui',
        plus: (n) => '+' + n + ' tok',
        multiOn: 'MULTI_PREFILL=1: vários prompts por chunk, menos linhas restantes primeiro',
        multiOff: 'MULTI_PREFILL=0: um prompt por chunk, o mais antigo primeiro (FIFO)',
        queue: 'prompts', tl: 'iterações do rank 0 (P = chunk de prompt, com as linhas de cada um; R = rodada de decode)',
        tok: ' tokens', first: (i) => '1º token no #' + i, wait: 'na fila', rowsL: (n) => n,
      };
      const NM = ['A', 'B', 'C', 'D'];
      const SC = (C) => [C.a, C.b, C.c, C.d];
      const S = { mode: 'round', n: 4, copy: true, seed: 1, scen: 'four', multi: true, t0: 0, tNow: 0, rd: null, ch: null };
      /* altura do palco por largura e por modo (o modo chunk precisa de mais linhas no celular) */
      const stB = fig.querySelector('.la-fig-stage');
      const setH = () => {
        const sw = stB.clientWidth || 800, nar = sw < 700;
        let hh = 340;
        if (nar && S.mode !== 'chunk') hh = 360;
        else if (nar) {   /* o mesmo layout de fichas do desenho: uma fileira ou mais */
          const W = sw - 28, m = S.ch ? S.ch.it.length : 16, gap = 6;
          const perRow = (W - gap * (m - 1)) / m < 34 ? Math.max(1, Math.floor((W + gap) / 40)) : m;
          hh = 350 + (Math.ceil(m / perRow) - 1) * 118;
        }
        const want = hh + 'px';
        if (stB.style.height !== want) stB.style.height = want;
      };
      const roB = new ResizeObserver(setH); roB.observe(stB);
      if (LA.onCleanup) LA.onCleanup(() => roB.disconnect());
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: TX.mode, options: [{ v: 'round', label: TX.mRound }, { v: 'chunk', label: TX.mChunk }], value: 'round', onChange: (v) => { S.mode = v; gR.show(v === 'round'); gC.show(v === 'chunk'); setH(); restart(); } });
      const gR = UI.group(ctl), gC = UI.group(ctl);
      UI.seg(gR.el, { label: TX.streams, options: [1, 2, 3, 4].map((v) => ({ v, label: String(v) })), value: 4, onChange: (v) => { S.n = v; buildRound(); restart(); } });
      UI.toggle(gR.el, { label: TX.copy, value: true, onChange: (v) => { S.copy = v; buildRound(); restart(); } });
      UI.button(gR.el, TX.dice, () => { S.seed++; buildRound(); restart(); });
      const rP = UI.readout(gR.el, TX.rP, '—'), rV = UI.readout(gR.el, TX.rV, '—'), rK = UI.readout(gR.el, TX.rK, '—');
      UI.select(gC.el, { label: TX.scen, options: [{ v: 'four', label: TX.sFour }, { v: 'long', label: TX.sLong }], value: 'four', onChange: (v) => { S.scen = v; buildChunk(); setH(); restart(); } });
      UI.toggle(gC.el, { label: 'MULTI_PREFILL', value: true, onChange: (v) => { S.multi = v; buildChunk(); setH(); restart(); } });
      UI.button(gC.el, TX.again, () => restart());
      const rL = UI.readout(gC.el, TX.rL, '—'), rM = UI.readout(gC.el, TX.rM, '—');
      gC.show(false);
      function restart() { S.t0 = S.tNow; }

      /* rodada: propostas, corte a MAX_WINDOW (como trim() em multi.py) e aceitação ilustrativa */
      function buildRound() {
        const r = V.rng(1009 * S.seed + 17 * S.n + (S.copy ? 3 : 0));
        const props = [];
        for (let i = 0; i < S.n; i++) props.push(i === 0 && S.copy ? { d: 15, copy: true } : { d: 3 + Math.floor(r() * 5), copy: false });
        const keep = props.map((p) => p.d), cuts = [];
        let total = keep.reduce((s, d) => s + 1 + d, 0);
        const proposed = total;
        while (total > 32) {
          let kk = 0;
          for (let i = 0; i < keep.length; i++) if (keep[i] >= keep[kk]) kk = i;   /* a mais longa; a mais nova no empate */
          if (!keep[kk]) break;
          cuts.push({ i: kk, j: keep[kk] }); keep[kk]--; total--;
        }
        const acc = keep.map((d, i) => { let a = 0; const pr = props[i].copy ? 0.85 : 0.62; while (a < d && r() < pr) a++; return a; });
        const off = []; let o = 0;
        keep.forEach((d) => { off.push(o); o += 1 + d; });
        S.rd = { props, keep, cuts, acc, proposed, total, off };
        rP.set(String(proposed), proposed > 32 ? 'warn' : null);
        rV.set(total + ' → ' + TX.graph(total), 'ok');
        rK.set(String(acc.reduce((s, a) => s + 1 + a, 0)));
      }
      /* chunk: _fill_turn + _stop + _group de multi.py / multi_prefill.py, com pontos guardados na grade de 64 */
      function buildChunk() {
        const lens = S.scen === 'four' ? [130, 130, 130, 130] : [4000, 130, 130, 130];
        const lanes = lens.map((n, i) => { const gp = Math.floor(n / 64) * 64; return { i, n, pos: 0, dec: false, stops: gp > 0 && gp < n ? [gp] : [], first: -1 }; });
        const nextStop = (l) => { const s = l.stops.find((p) => p > l.pos); return s == null ? l.n : s; };
        const it = [];
        let credit = 0;
        for (let guard = 0; guard < 60; guard++) {
          const filling = lanes.filter((l) => !l.dec), decoding = lanes.filter((l) => l.dec);
          if (!filling.length) break;
          if (decoding.length) {
            credit += 0.5;
            if (credit < 1) { it.push({ k: 'R', s: decoding.map((l) => l.i) }); continue; }
            credit -= 1;
          }
          const busy = decoding.length > 0;
          let pieces = null;
          if (S.multi) {
            let left = busy ? 1024 : 2048;
            const order = filling.slice().sort((a, b) => (a.n - a.pos) - (b.n - b.pos) || a.i - b.i);
            const out = [];
            for (const l of order) {
              let stop = nextStop(l);
              if (stop - l.pos > left) stop = l.pos + Math.floor(left / 64) * 64;
              if (stop <= l.pos) continue;
              out.push({ l, a: l.pos, b: stop });
              left -= stop - l.pos;
              if (left <= 0) break;
            }
            if (out.length > 1) pieces = out;
          }
          if (!pieces) { const l = filling[0]; pieces = [{ l, a: l.pos, b: Math.min(nextStop(l), l.pos + (busy ? 1024 : 2048)) }]; }
          it.push({ k: 'P', p: pieces.map((p) => ({ i: p.l.i, a: p.a, b: p.b, end: p.b === p.l.n })) });
          pieces.forEach((p) => { p.l.pos = p.b; if (p.b === p.l.n) { p.l.dec = true; p.l.first = it.length - 1; } });
        }
        it.push({ k: 'R', s: lanes.map((l) => l.i) });
        const last = Math.max(...lanes.map((l) => l.first));
        const upto = it.slice(0, last + 1);
        S.ch = { it, lens, firsts: lanes.map((l) => l.first) };
        rL.set(TX.vL(last + 1, upto.filter((x) => x.k === 'P').length, upto.filter((x) => x.k === 'R').length), last + 1 > 4 ? 'warn' : 'ok');
        rM.set(TX.vM(nf(lanes.reduce((s, l) => s + l.first + 1, 0) / lanes.length, 1)));
      }
      buildRound(); buildChunk(); setH();

      function cell(g, x, y, s, col, kind, a) {
        const C = V.C();
        g.globalAlpha = a == null ? 1 : a;
        if (kind === 'pend') V.box(g, x, y, s, s, { r: 3, fill: col });
        else if (kind === 'copy') { V.box(g, x, y, s, s, { r: 3, fill: V.alpha(col, 0.22) }); g.setLineDash([3, 2]); g.strokeStyle = col; g.lineWidth = 1.4; V.rr(g, x + 0.7, y + 0.7, s - 1.4, s - 1.4, 3); g.stroke(); g.setLineDash([]); }
        else V.box(g, x, y, s, s, { r: 3, fill: V.alpha(col, 0.5) });
        g.globalAlpha = 1;
        return C;
      }

      V.fig(fig, (g, w, h, t) => {
        if (w < 60 || h < 60) return;
        S.tNow = t;
        const C = V.C(), cTF = TFC(C), cols = SC(C);
        const narrow = w < 700;
        const L = 14, R = w - 14, W = R - L;
        const el = t - S.t0;
        if (S.mode === 'round') {
          const rd = S.rd, nC = rd.cuts.length;
          const T1 = 1.2, cutStep = nC ? Math.min(0.28, 2.2 / nC) : 0.6, T2 = T1 + Math.max(0.6, nC * cutStep) + 0.2, T3 = T2 + 1.1, T4 = T3 + 1.8, TEND = T4 + 2.4;
          let tt = el;
          if (tt > TEND) { S.t0 = t; tt = 0; }
          const ph = tt < T1 ? 0 : tt < T2 ? 1 : tt < T3 ? 2 : 3;
          const capt = ph === 1 && !nC ? TX.noCut : ph === 2 ? TX.ph[2] + rd.total + TX.lines : TX.ph[ph];
          let y = 18;
          y += V.wrap(g, capt, L, y, W, { size: 12.5, weight: 600, color: C.ink, lineH: 16 }) + 8;
          /* propostas (largura toda) */
          const cs = Math.max(12, Math.min(26, Math.floor((W - 114) / 16) - 3)), cstep = cs + 3;
          const rowH = cs + 10;
          V.text(g, TX.props, L, y + 2, { size: 11.5, weight: 600, color: C.muted });
          const ry0 = y + 12;
          const cutDone = (i, j) => { const ci = rd.cuts.findIndex((c) => c.i === i && c.j === j); if (ci < 0) return -1; return ci; };
          for (let i = 0; i < S.n; i++) {
            const yy = ry0 + i * rowH, col = cols[i], p = rd.props[i];
            const accShown = ph === 3 && tt > T3 + 0.9;
            critter(g, L + 11, yy + cs / 2, 9, col, accShown ? 'happy' : 'neutral', t);
            V.text(g, NM[i], L + 24, yy + cs / 2 + 4, { size: 11.5, font: 'mono', weight: 700, color: col });
            const n = 1 + p.d;
            const vis = ph === 0 ? Math.ceil(n * clamp(tt / (T1 * 0.85), 0, 1)) : n;
            for (let j = 0; j < vis; j++) {
              const x = L + 38 + j * cstep;
              const kind = j === 0 ? 'pend' : p.copy ? 'copy' : 'df';
              const ci = cutDone(i, j);
              let a = 1, dy = 0, cut = false;
              if (ci >= 0 && ph >= 1) {
                const tc0 = T1 + 0.1 + ci * cutStep;
                if (tt >= tc0 + cutStep * 1.6 || ph >= 2) continue;
                if (tt >= tc0) { cut = true; const f = clamp((tt - tc0) / (cutStep * 1.6), 0, 1); dy = f * f * 18; a = 1 - f; }
              }
              if (ph >= 2 && j > rd.keep[i]) continue;
              const moved = ph >= 2 ? clamp((tt - T2 - (rd.off[i] + j) * 0.012) / 0.6, 0, 1) : 0;
              cell(g, x, yy + dy, cs, cut ? C.red : col, kind, a * (1 - 0.65 * moved));
            }
            if (accShown) V.text(g, TX.plus(1 + rd.acc[i]), L + 38 + (1 + rd.keep[i]) * cstep + 6, yy + cs / 2 + 4, { size: 11.5, font: 'mono', weight: 700, color: C.ink });
          }
          /* janela em lote: 32 slots (uma fileira se couber) */
          const perW = W >= 32 * 17 ? 32 : 16;
          const ws = Math.max(12, Math.min(26, Math.floor(W / perW) - 3)), wstep = ws + 3;
          const wx = L, wy = ry0 + S.n * rowH + 26, wBottom = wy + (32 / perW) * (wstep + 4);
          const slotXY = (s) => [wx + (s % perW) * wstep, wy + Math.floor(s / perW) * (wstep + 4)];
          V.text(g, TX.win, wx, wy - 10, { size: 11.5, weight: 600, color: C.muted, maxW: R - wx });
          for (let s = 0; s < 32; s++) {
            const [x, yy] = slotXY(s);
            g.setLineDash([3, 3]); g.strokeStyle = V.alpha(C.muted, 0.5); g.lineWidth = 1; V.rr(g, x + 0.5, yy + 0.5, ws - 1, ws - 1, 3); g.stroke(); g.setLineDash([]);
          }
          if (ph >= 2) {
            for (let i = 0; i < S.n; i++) {
              for (let j = 0; j <= rd.keep[i]; j++) {
                const s = rd.off[i] + j;
                const f = V.ease.inOut(clamp((tt - T2 - s * 0.012) / 0.6, 0, 1));
                const sx = L + 38 + j * cstep, sy = ry0 + i * rowH;
                const [tx, ty] = slotXY(s);
                const x = V.lerp(sx, tx, f), yy = V.lerp(sy, ty, f), sz = V.lerp(cs, ws, f);
                const kind = j === 0 ? 'pend' : rd.props[i].copy ? 'copy' : 'df';
                let col = cols[i];
                if (ph === 3) {
                  const rev = clamp((tt - T3) / 1.2, 0, 1) * 16;
                  if (j <= rev) {
                    if (j === 0 || j <= rd.acc[i]) { cell(g, x, yy, sz, col, kind); V.text(g, '✓', x + sz / 2, yy + sz / 2 + 4, { size: 11, weight: 700, color: kind === 'pend' ? ink(C) : C.ink, align: 'center' }); continue; }
                    if (j === rd.acc[i] + 1) { cell(g, x, yy, sz, C.red, 'pend', 0.9); V.text(g, '✗', x + sz / 2, yy + sz / 2 + 4, { size: 11, weight: 700, color: ink(C), align: 'center' }); continue; }
                    cell(g, x, yy, sz, C.muted, 'df', 0.45); continue;
                  }
                }
                cell(g, x, yy, sz, col, kind);
              }
            }
            const gx = wx, gy2 = wBottom + 6;
            g.font = V.font(12, 'mono', 700);
            const lab = TX.graph(rd.total), lw = g.measureText(lab).width + 16;
            V.box(g, gx, gy2, lw, 22, { r: 11, fill: V.alpha(cTF, 0.9), label: lab, size: 12, font: 'mono', weight: 700, color: ink(C) });
            V.text(g, rd.total + ' / 32' + TX.lines, gx + lw + 10, gy2 + 15, { size: 11.5, font: 'mono', color: C.muted, maxW: R - gx - lw - 12 });
          }
          /* legenda */
          const ly = wBottom + 50;
          let lx = L, yl = ly;
          const legs = [['pend', cols[0], TX.legPend], ['copy', cols[0], TX.legCopy], ['df', cols[1], TX.legDf], ['ok', C.ink, TX.legOk], ['bad', C.red, TX.legBad]];
          legs.forEach(([kd, col, lab]) => {
            g.font = V.font(11);
            const lw = 18 + g.measureText(lab).width + 16;
            if (lx > L && lx + lw > R) { lx = L; yl += 18; }
            if (kd === 'ok') V.text(g, '✓', lx + 6, yl, { size: 12, weight: 700, color: col, align: 'center' });
            else if (kd === 'bad') { cell(g, lx, yl - 10, 12, col, 'pend', 0.9); }
            else cell(g, lx, yl - 10, 12, col, kd);
            V.text(g, lab, lx + 18, yl, { size: 11, color: C.muted });
            lx += lw;
          });
        } else {
          /* ---------------- chunk de prompt ---------------- */
          const ch = S.ch, it = ch.it, m = it.length, per = 0.75, TEND = m * per + 2.6;
          let tt = el;
          if (tt > TEND) { S.t0 = t; tt = 0; }
          const cur = Math.min(m - 1, Math.floor(tt / per)), prog = clamp((tt - cur * per) / (per * 0.8), 0, 1);
          let y = 18;
          y += V.wrap(g, S.multi ? TX.multiOn : TX.multiOff, L, y, W, { size: 12.5, weight: 600, color: C.ink, lineH: 16 }) + 6;
          /* fila de prompts com progresso */
          const pos = ch.lens.map(() => 0);
          for (let j = 0; j <= cur; j++) if (it[j].k === 'P') it[j].p.forEach((p) => { pos[p.i] = j < cur ? p.b : V.lerp(p.a, p.b, prog); });
          const rowH = 24, barX = L + (narrow ? 112 : 150), barW = Math.min(narrow ? W - 112 - 4 : 300, W - (barX - L) - 4);
          V.text(g, TX.queue, L, y + 4, { size: 11.5, weight: 600, color: C.muted });
          const qy = y + 12;
          ch.lens.forEach((n, i) => {
            const yy = qy + i * rowH, col = cols[i], f = ch.firsts[i], got = f >= 0 && (cur > f || (cur === f && prog >= 1));
            critter(g, L + 10, yy + 9, 8, col, got ? 'happy' : pos[i] > 0 ? 'neutral' : 'sleep', t);
            V.text(g, NM[i] + ' · ' + nf(n) + TX.tok, L + 24, yy + 13, { size: 11, font: 'mono', color: C.ink, maxW: barX - L - 30 });
            V.box(g, barX, yy + 4, barW, 10, { r: 5, fill: V.alpha(C.muted, 0.18) });
            V.box(g, barX, yy + 4, Math.max(3, barW * (pos[i] / n)), 10, { r: 5, fill: col });
            if (!narrow) V.text(g, got ? TX.first(f + 1) : TX.wait, barX + barW + 10, yy + 13, { size: 11, font: 'mono', weight: got ? 700 : 400, color: got ? C.ink : C.muted });
            else if (got) V.text(g, '✓', barX + barW - 2, yy + 13, { size: 11, weight: 700, color: ink(C), align: 'right' });
          });
          y = qy + 4 * rowH + 14;
          /* linha do tempo de iterações */
          const tlH = V.wrap(g, TX.tl, L, y, W, { size: 11.5, weight: 600, color: C.muted, lineH: 15 });
          y += tlH - 15;
          const gap = 6, minCW = narrow ? 34 : 44;
          let perRow = m, cw = Math.min(92, (W - gap * (m - 1)) / m);
          if (cw < minCW) { perRow = Math.max(1, Math.floor((W + gap) / (minCW + gap))); cw = (W - gap * (perRow - 1)) / perRow; }
          const chH = narrow ? 92 : 104;
          const ty0 = y + 10;
          for (let j = 0; j < m; j++) {
            const x = L + (j % perRow) * (cw + gap), yy = ty0 + Math.floor(j / perRow) * (chH + 26), x1 = it[j];
            const a = j < cur ? 0.9 : j === cur ? 1 : 0.22;
            g.globalAlpha = a;
            if (x1.k === 'P') {
              V.box(g, x, yy, cw, chH, { r: 8, fill: V.alpha(cTF, 0.08), stroke: V.alpha(cTF, j === cur ? 1 : 0.6), lw: j === cur ? 2.2 : 1.2 });
              V.text(g, 'P', x + 6, yy + 14, { size: 11.5, font: 'mono', weight: 700, color: cTF });
              V.text(g, (cw >= 44 ? '#' : '') + (j + 1), x + cw - 5, yy + 14, { size: 11, font: 'mono', color: C.muted, align: 'right' });
              const np = x1.p.length, bh = Math.min(20, (chH - 36) / Math.max(np, 1) - 2);
              x1.p.forEach((p, pi) => {
                const by = yy + 21 + pi * (bh + 2);
                V.box(g, x + 4, by, cw - 8, bh, { r: 3, fill: V.alpha(cols[p.i], 0.85) });
                const lab = cw > 62 ? NM[p.i] + ' ' + nf(p.b - p.a) : NM[p.i];
                if (bh >= 11) V.text(g, lab, x + cw / 2, by + bh / 2 + 4, { size: 11, font: 'mono', weight: 700, color: ink(C), align: 'center', maxW: cw - 10 });
              });
              const rows = x1.p.reduce((s, p) => s + p.b - p.a, 0);
              V.text(g, TX.rowsL(nf(rows)), x + cw / 2, yy + chH - 6, { size: 11, font: 'mono', color: C.muted, align: 'center', maxW: cw - 6 });
            } else {
              V.box(g, x, yy, cw, chH, { r: 8, fill: V.alpha(C.muted, 0.07), stroke: V.alpha(C.muted, j === cur ? 1 : 0.5), lw: j === cur ? 2.2 : 1.2 });
              V.text(g, 'R', x + 6, yy + 14, { size: 11.5, font: 'mono', weight: 700, color: C.muted });
              V.text(g, (cw >= 44 ? '#' : '') + (j + 1), x + cw - 5, yy + 14, { size: 11, font: 'mono', color: C.muted, align: 'right' });
              x1.s.forEach((si, k2) => { const dx = x + cw / 2 + (k2 - (x1.s.length - 1) / 2) * Math.min(14, (cw - 8) / Math.max(1, x1.s.length)); V.dot(g, dx, yy + chH / 2, Math.min(5.5, (cw - 8) / 8), cols[si]); });
            }
            g.globalAlpha = 1;
            /* quem ganhou o 1º token nesta iteração */
            const got = ch.firsts.map((f, i) => (f === j ? i : -1)).filter((i) => i >= 0);
            if (got.length && j <= cur) V.text(g, got.map((i) => NM[i]).join('') + ' ✓', x + cw / 2, yy + chH + 15, { size: 11, font: 'mono', weight: 700, color: C.ink, align: 'center', maxW: cw + gap });
          }
        }
      });
    }
  },
});
